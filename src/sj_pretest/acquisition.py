"""采集：把图像来源的帧流按"段"落成 RAW，格式与旧项目采集目录完全一致。

为什么要分段
------------
需求六要求"在线线程只负责尽可能稳定地采集和保存"，角点识别全部放到实验之后。
所以这里的职责被压到最小：**接住帧、写盘、记账**。每段（一次试验、或静态基线）
写成一个独立的采集目录，字段格式与被复用代码的 ``_run_raw_capture_core`` 一致，
因此 ``camera.RawCaptureSource`` 之后能原样读回来做离线识别。

时间轴由帧自己的 ``host_ns`` 驱动
---------------------------------
段的结束条件是"某一帧的 ``host_ns`` 超过了本段截止时刻"，而不是墙上时钟。
这样三种来源走同一套逻辑：
* 真机 —— ``host_ns`` 是 perf_counter_ns，等于真实时间；
* 回放 —— ``host_ns`` 是当年记录的时间，于是回放跑得比实时快很多；
* 合成 —— ``host_ns`` 由虚拟时钟按 132.23 fps 精确推进，于是不用真的等 7.5 ms。

格式关键的那几步（时间戳补全、缺帧清单、摘要、样本图）直接调被复用代码的
私有函数，不自己重写一遍——自己重写就有和 ``_read_capture_timestamps``
对不上的风险，而那个错误在回放时才会暴露。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import AppConfig
from .vendor_shim import vendor_module


class CaptureError(RuntimeError):
    """采集出错。消息中文。"""


@dataclass
class PhasePlan:
    """一段采集里的一个阶段。

    为什么需要"阶段"而不是"整段定时长"
------------------------------------
    真机上运动要花多久，取决于控制器怎么规划，程序事先只能估。如果整段按时长采，
    估短了会把"保持"阶段切掉一半，估长了会白录一堆。所以每段被拆成：
    运动前静止 → 下发动作并等它真的停下来 → 保持 → 回程并等停 → 运动后记录。
    每个阶段的边界都由**实际发生的帧**决定，于是任何模式下都能得到准确的相位时间。

    字段含义：
    * ``min_duration_s``：这个阶段至少录多久（按帧自带时间戳算）；
    * ``action``：进入这个阶段时执行的动作用（通常就是下发一次 moveJ）；
    * ``wait_for``：在 ``min_duration_s`` 之后还要等它变成 True 才结束
      （真机上是"关节停稳了"）。返回 False 表示还在动，继续录。
    * ``max_duration_s``：兜底上限，防止 ``wait_for`` 永远不满足时把磁盘录满。
    """

    label: str
    min_duration_s: float
    action: Callable[[], None] | None = None
    wait_for: Callable[[], bool] | None = None
    max_duration_s: float = 60.0


@dataclass
class PhaseTiming:
    """一个阶段在实际帧时间轴上的起止（相对本段起点，秒）。"""

    label: str
    start_s: float
    end_s: float
    frames: int
    ended_by: str  # "duration" / "settled" / "timeout" / "stopped"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "duration_s": float(self.duration_s),
            "frames": int(self.frames),
            "ended_by": self.ended_by,
        }


@dataclass
class SegmentCapture:
    """一段采集的记账结果。"""

    segment_id: str
    kind: str
    dir: Path
    frame_count: int
    first_frame_id: int
    last_frame_id: int
    first_host_ns: int
    last_host_ns: int
    duration_s: float
    missing_frames: int
    dropped_ratio: float
    actual_camera_fps: float
    #: 这段的帧是不是合成的（dry_run）。
    synthetic: bool
    #: 采集期间的宿主时间（真机是真实耗时；回放/虚拟时钟下会远小于 duration_s）。
    wall_seconds: float
    #: 阶段时间表。静态基线只有一个阶段。
    phases: list[PhaseTiming] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False

    def phase(self, label: str) -> PhaseTiming | None:
        for timing in self.phases:
            if timing.label == label:
                return timing
        return None

    def phase_window(self, label: str) -> tuple[float, float] | None:
        timing = self.phase(label)
        return None if timing is None else (timing.start_s, timing.end_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "dir": str(self.dir),
            "frame_count": int(self.frame_count),
            "first_frame_id": int(self.first_frame_id),
            "last_frame_id": int(self.last_frame_id),
            "first_host_ns": int(self.first_host_ns),
            "last_host_ns": int(self.last_host_ns),
            "duration_s": float(self.duration_s),
            "missing_frames": int(self.missing_frames),
            "dropped_ratio": float(self.dropped_ratio),
            "actual_camera_fps": float(self.actual_camera_fps),
            "synthetic": bool(self.synthetic),
            "wall_seconds": float(self.wall_seconds),
            "stopped_early": bool(self.stopped_early),
            "phases": [timing.to_dict() for timing in self.phases],
        }

    def summary_line(self) -> str:
        tag = " [合成数据]" if self.synthetic else ""
        early = "（提前停止，已保住已采到的帧）" if self.stopped_early else ""
        phase_text = ""
        if self.phases:
            phase_text = "  阶段：" + "、".join(
                f"{timing.label} {timing.duration_s:.2f}s" for timing in self.phases
            )
        return (
            f"{self.segment_id}：{self.frame_count} 帧 / {self.duration_s:.2f} s，"
            f"缺帧 {self.missing_frames}（{self.dropped_ratio:.2%}）{tag}{early}{phase_text}"
        )


class CaptureEngine:
    """共用一个来源迭代器，按段把帧落盘。"""

    def __init__(
        self,
        config: AppConfig,
        source: Iterator[Any],
        *,
        synthetic: bool = False,
    ) -> None:
        self.config = config
        # 来源可能是"可迭代对象"（带 __iter__）而不是迭代器本身，统一成迭代器。
        self.source = source if hasattr(source, "__next__") else iter(source)
        self.synthetic = bool(synthetic)
        self._pending: Any = None
        self.segments: list[SegmentCapture] = []
        self.camera = vendor_module("camera")
        roi = config.camera.roi if config.mode != "dry_run" else None
        self.roi: tuple[int, int, int, int] | None = (
            tuple(int(value) for value in roi) if roi else None
        )  # type: ignore[assignment]

    # -- 主流程 -----------------------------------------------------------

    def capture_segment(
        self,
        out_dir: Path,
        *,
        segment_id: str,
        kind: str,
        phases: list[PhasePlan] | None = None,
        duration_s: float | None = None,
        stop_requested: Callable[[], bool] | None = None,
        metadata_extra: dict[str, Any] | None = None,
        on_frame: Callable[[Any, np.ndarray, int], None] | None = None,
    ) -> SegmentCapture:
        """采一段帧，写成一个采集目录。

        两种用法：
        * ``phases``：按阶段采，每个阶段的边界由实际帧决定（试验用这个）；
        * ``duration_s``：只采固定时长（静态基线用这个，它只有一个阶段）。

        时间长度一律**按帧自带的时间戳算**，不是"墙上时钟要等多久"。
        """
        if phases:
            plan_phases = list(phases)
        elif duration_s is not None:
            plan_phases = [PhasePlan(label="static", min_duration_s=float(duration_s))]
        else:
            raise CaptureError("capture_segment 需要 phases 或 duration_s 之一。")
        for item in plan_phases:
            if item.min_duration_s < 0:
                raise CaptureError(f"阶段 {item.label} 的最短时长不能为负。")
        if all(item.min_duration_s == 0 and item.wait_for is None for item in plan_phases):
            raise CaptureError("所有阶段都是零时长且没有等待条件，这段采集没有意义。")
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "frames.raw"
        temp_path = raw_path.with_name(raw_path.name + ".tmp")
        metadata_path = out_dir / "capture_metadata.json"
        timestamps_path = out_dir / "frame_timestamps.csv"
        missing_frames_path = out_dir / "missing_frames.csv"
        summary_path = out_dir / "capture_summary.txt"

        rows: list[dict[str, Any]] = []
        copy_times_ms: list[float] = []
        width = height = 0
        expected_fps = float(self.config.effective_fps())
        stopped_early = False
        wall_start = time.perf_counter()
        phase_timings: list[PhaseTiming] = []
        wait_poll_interval_s = 0.05

        raw_stream = temp_path.open("wb")
        try:
            first = self._next_frame()
            frame = self._prepare(first.frame)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if frame.ndim != 2 or frame.dtype != np.uint8:
                raise CaptureError(
                    "RAW 采集只支持 Mono8（二维 uint8）帧，"
                    f"实际 shape={frame.shape} dtype={frame.dtype}。"
                )
            segment_start_ns = int(first.host_ns)

            def content_time_s(packet: Any) -> float:
                """这一帧相对本段起点的时间（秒）。整段的时间轴就靠它。"""
                return (int(packet.host_ns) - segment_start_ns) / 1e9

            for index, plan in enumerate(plan_phases):
                if stop_requested is not None and stop_requested():
                    stopped_early = True
                    break
                if plan.action is not None:
                    # 动作在阶段**开始时**执行：先下命令，再录它怎么动。
                    plan.action()
                phase_start_ns = int(first.host_ns)
                phase_start_index = len(rows)
                ended_by = "duration"
                next_wait_check_ns = phase_start_ns
                max_deadline_ns = phase_start_ns + int(plan.max_duration_s * 1e9)
                while True:
                    write_start_ns = time.perf_counter_ns()
                    raw_stream.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                    copy_times_ms.append(
                        (time.perf_counter_ns() - write_start_ns) / 1e6
                    )
                    rows.append(
                        {
                            "capture_index": len(rows),
                            "frame_id": int(first.frame_id),
                            "host_ns": int(first.host_ns),
                            "camera_timestamp_raw": (
                                None
                                if first.camera_timestamp_raw is None
                                else int(first.camera_timestamp_raw)
                            ),
                        }
                    )
                    if on_frame is not None:
                        on_frame(first, frame, len(rows) - 1)

                    if stop_requested is not None and stop_requested():
                        stopped_early = True
                        ended_by = "stopped"
                        break

                    packet = self._next_frame()
                    elapsed_s = content_time_s(packet) - (
                        (phase_start_ns - segment_start_ns) / 1e9
                    )
                    min_ok = elapsed_s >= plan.min_duration_s
                    timed_out = int(packet.host_ns) >= max_deadline_ns
                    finished = False
                    if min_ok:
                        if plan.wait_for is None:
                            ended_by = "duration"
                            finished = True
                        elif int(packet.host_ns) >= next_wait_check_ns:
                            # 等待条件要花钱（真机上一次 RTDE 调用是毫秒级），
                            # 所以最多每 50 ms 才问一次，别让 132 fps 的采集被它拖慢。
                            next_wait_check_ns = int(packet.host_ns) + int(
                                wait_poll_interval_s * 1e9
                            )
                            if bool(plan.wait_for()):
                                ended_by = "settled"
                                finished = True
                    if finished:
                        # 这一帧属于下一个阶段了。放回待取位，
                        # 既不重复写盘，也不丢帧。
                        self._stash(packet)
                        break
                    if timed_out:
                        ended_by = "timeout"
                        self._stash(packet)
                        break

                    first = packet
                    frame = self._prepare(packet.frame)
                    if frame.shape != (height, width):
                        raise CaptureError(
                            "采集过程中帧尺寸发生变化："
                            f"开始是 {(height, width)}，现在是 {frame.shape}。"
                        )

                phase_timings.append(
                    PhaseTiming(
                        label=plan.label,
                        start_s=(phase_start_ns - segment_start_ns) / 1e9,
                        end_s=content_time_s(first),
                        frames=len(rows) - phase_start_index,
                        ended_by=ended_by,
                    )
                )
                if stopped_early:
                    break
                if index < len(plan_phases) - 1:
                    first = self._next_frame()
                    frame = self._prepare(first.frame)
        finally:
            raw_stream.close()

        if not rows:
            temp_path.unlink(missing_ok=True)
            raise CaptureError(f"段 {segment_id} 没有采到任何帧。")

        content_seconds = (int(rows[-1]["host_ns"]) - segment_start_ns) / 1e9
        for timing in phase_timings:
            # 最后一个阶段的结束时间用实际最后一帧，避免收尾差一帧。
            if timing is phase_timings[-1]:
                timing.end_s = content_seconds

        actual_fps = self._actual_fps()
        missing_rows, analysis_source = self.camera._enrich_capture_timestamps(rows, actual_fps)
        temp_path.replace(raw_path)
        self.camera._write_capture_timestamps(timestamps_path, rows)
        self.camera._write_missing_frames(missing_frames_path, missing_rows)
        if self.config.camera.save_sample_images:
            self.camera._save_capture_sample_images_from_raw(
                raw_path, len(rows), height, width, out_dir
            )
        raw_size = raw_path.stat().st_size
        self.camera._write_capture_summary(
            summary_path,
            rows,
            copy_times_ms,
            content_seconds,
            None,
            raw_size,
            actual_fps,
            self._actual_fps_source(),
            analysis_source,
            missing_frames_path,
        )

        gap_events = sum(1 for row in rows if int(row.get("missing_before", 0)) > 0)
        missing_total = sum(int(row.get("missing_before", 0)) for row in rows)
        nominal = len(rows) + missing_total
        dropped_ratio = (missing_total / nominal) if nominal else 0.0

        metadata: dict[str, Any] = {
            "width": width,
            "height": height,
            "channels": 1,
            "dtype": "uint8",
            "storage_mode": "stream_raw",
            "frame_count": len(rows),
            "frame_bytes": int(width * height),
            "raw_file": raw_path.name,
            "expected_fps": expected_fps,
            "actual_camera_fps": actual_fps,
            "actual_camera_fps_source": self._actual_fps_source(),
            "nominal_frame_period_s": 1.0 / actual_fps,
            "analysis_time_source": analysis_source,
            "capture_duration_s": float(content_seconds),
            "phase_count": len(phase_timings),
            "phases": [timing.to_dict() for timing in phase_timings],
            "first_frame_id": int(rows[0]["frame_id"]),
            "last_frame_id": int(rows[-1]["frame_id"]),
            "first_host_ns": int(rows[0]["host_ns"]),
            "last_host_ns": int(rows[-1]["host_ns"]),
            "first_camera_timestamp_raw": rows[0]["camera_timestamp_raw"],
            "last_camera_timestamp_raw": rows[-1]["camera_timestamp_raw"],
            "missing_frame_count": missing_total,
            "gap_event_count": gap_events,
            "maximum_missing_run": max(
                (int(row.get("missing_before", 0)) for row in rows), default=0
            ),
            "missing_frames_csv": missing_frames_path.name,
            "segment_id": segment_id,
            "segment_kind": kind,
            "synthetic": self.synthetic,
            "roi": list(self.roi) if self.roi else None,
            "content_seconds": content_seconds,
            "stopped_early": bool(stopped_early),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        metadata_path.write_text(
            np_style_json(metadata), encoding="utf-8"
        )

        record = SegmentCapture(
            segment_id=segment_id,
            kind=kind,
            dir=out_dir,
            frame_count=len(rows),
            first_frame_id=int(rows[0]["frame_id"]),
            last_frame_id=int(rows[-1]["frame_id"]),
            first_host_ns=int(rows[0]["host_ns"]),
            last_host_ns=int(rows[-1]["host_ns"]),
            duration_s=content_seconds,
            missing_frames=missing_total,
            dropped_ratio=dropped_ratio,
            actual_camera_fps=actual_fps,
            synthetic=self.synthetic,
            wall_seconds=time.perf_counter() - wall_start,
            phases=phase_timings,
            metadata=metadata,
            stopped_early=stopped_early,
        )
        self.segments.append(record)
        return record

    # -- 内部 -------------------------------------------------------------

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """裁剪 ROI（如果配了）。裁剪发生在写盘之前，所以 RAW 里存的就是裁过的图，
        这样磁盘占用和后续解析开销都按裁剪后的尺寸算。"""
        if self.roi is None:
            return frame
        x, y, width, height = self.roi
        if y + height > frame.shape[0] or x + width > frame.shape[1]:
            raise CaptureError(
                f"camera.roi={list(self.roi)} 超出帧尺寸 "
                f"{frame.shape[1]}×{frame.shape[0]}，无法裁剪。"
            )
        return np.ascontiguousarray(frame[y : y + height, x : x + width])

    def _stash(self, packet: Any) -> None:
        """把"已经取出、但属于下一个阶段/段"的帧放回去。

        如果来源本身能暂存（:class:`~sj_pretest.sources.FramePump`），就交给它——
        整条流只允许有一个暂存位，否则实时预览和采集会互相插队。
        """
        push_back = getattr(self.source, "push_back", None)
        if push_back is not None:
            push_back(packet)
        else:
            self._pending = packet

    def _next_frame(self) -> Any:
        if self._pending is not None:
            packet = self._pending
            self._pending = None
            return packet
        try:
            return next(self.source)
        except StopIteration as exc:
            raise CaptureError(
                "图像来源已经没有更多帧了，但这一段还没采够时长。"
                "回放模式请检查历史数据是否覆盖了完整实验；"
                "真机模式请检查相机是否掉线。"
            ) from exc

    def _actual_fps(self) -> float:
        value = getattr(self.source, "actual_camera_fps", None)
        if value:
            return float(value)
        return float(self.config.effective_fps())

    def _actual_fps_source(self) -> str:
        return str(
            getattr(self.source, "actual_camera_fps_source", "config.EXPECTED_VISION_FPS")
        )

    # -- 汇总 -------------------------------------------------------------

    def totals(self) -> dict[str, Any]:
        return {
            "segments": len(self.segments),
            "frames": sum(record.frame_count for record in self.segments),
            "seconds": sum(record.duration_s for record in self.segments),
            "missing_frames": sum(record.missing_frames for record in self.segments),
            "bytes": sum(
                int(record.metadata.get("frame_count", 0))
                * int(record.metadata.get("frame_bytes", 0))
                for record in self.segments
            ),
            "synthetic": bool(self.synthetic),
        }


def np_style_json(payload: dict[str, Any]) -> str:
    """和旧项目一样用 ``json.dumps(..., ensure_ascii=False, indent=2)`` 落盘。"""
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
