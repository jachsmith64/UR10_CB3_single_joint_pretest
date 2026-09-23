"""记录：实验目录结构、事件日志、RTDE 状态流、以及需求六要求的那十几项原始数据。

目录结构（一次运行一个时间戳目录，绝不覆盖旧实验）
--------------------------------------------------
::

    outputs/20260923_153012/
        config.json              完整配置（界面上的每一个数都在这里）
        run_manifest.json        运行元信息：模式、版本、是否合成、计划规模
        events.jsonl             事件日志：每次动作/等待/异常/中止，一行一条
        robot_states.csv         六个指令角、六个实际角、六个实际角速度、TCP、
                                 机器人与安全状态、RTDE 时间戳
        segments/<事件编号>/     每次试验或静态基线的采集目录
            frames.raw               原始帧（Mono8，逐帧连续）
            capture_metadata.json    尺寸/帧数/帧率/缺帧统计
            frame_timestamps.csv     逐帧相机时间戳、帧号、缺帧序号
            missing_frames.csv       缺帧清单
            capture_summary.txt      人可读摘要
            sample_*.png             首/中/末三张样本图
        vision/
            corners/<事件编号>.csv   88 个角点的离线识别结果
            metrics.csv              质心、二维转角、检测质量
        analysis/                §七 的三层分析结果
        notes.txt                运行结束时写下的状态与遗留问题

为什么要"不覆盖"
----------------
需求八要求"输出目录/文件不许覆盖旧实验"。这里用**时间戳目录 + 存在即报错**两层保证：
同一秒内重复启动会被拒绝，而不是安静地把上一份数据盖掉。
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import JOINT_NAMES, AppConfig, check_free_disk


class RecorderError(RuntimeError):
    """记录层出错。消息中文。"""


#: robot_states.csv 的列顺序。显式写死，避免"某次运行少了一列"这种事后才发现的问题。
ROBOT_STATE_COLUMNS: tuple[str, ...] = (
    ("host_ns", "event_id", "stage", "synthetic")
    + tuple(f"command_q_{name}_deg" for name in JOINT_NAMES)
    + tuple(f"actual_q_{name}_deg" for name in JOINT_NAMES)
    + tuple(f"actual_qd_{name}_deg_s" for name in JOINT_NAMES)
    + tuple(f"actual_tcp_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
    + ("robot_mode", "safety_mode", "runtime_state", "speed_scaling")
)


# --------------------------------------------------------------------------
# 事件日志
# --------------------------------------------------------------------------


@dataclass
class EventLog:
    """append-only 的事件日志（JSONL）。每次写都 flush，进程被杀也不丢已发生的事。"""

    path: Path
    synthetic: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host_ns": time.perf_counter_ns(),
            "event": str(event),
            "synthetic": bool(self.synthetic),
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# --------------------------------------------------------------------------
# RTDE 状态记录
# --------------------------------------------------------------------------


class RobotStateRecorder:
    """把机器人状态按时序写成 CSV。

    真机上用一个后台线程按 ``rtde_record_hz`` 采样——这样 132 fps 的相机线程
    不需要等 RTDE 读取，两边只共享时间戳（这正是被复用代码里的做法）。
    dry_run / replay 下同步采样就够了，不用起线程（也就不会有线程收尾问题）。
    """

    def __init__(
        self,
        path: Path,
        robot: Any,
        *,
        hz: float,
        threaded: bool,
        event_log: EventLog | None = None,
    ) -> None:
        self.path = path
        self.robot = robot
        self.hz = float(hz)
        self.threaded = bool(threaded)
        self.event_log = event_log
        self._handle: Any = None
        self._writer: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.row_count = 0
        self._current_event: dict[str, Any] = {"event_id": None, "stage": None}

    # -- 生命周期 ---------------------------------------------------------

    def open(self) -> "RobotStateRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(ROBOT_STATE_COLUMNS))
        self._writer.writeheader()
        self._handle.flush()
        if self.threaded:
            self._thread = threading.Thread(
                target=self._loop, name="rtde-recorder", daemon=True
            )
            self._thread.start()
            if self.event_log is not None:
                self.event_log.write("rtde_recorder_started", hz=self.hz)
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            if self._handle is not None:
                self._handle.flush()
                self._handle.close()
                self._handle = None
        if self.event_log is not None and self.threaded:
            self.event_log.write("rtde_recorder_stopped", rows=self.row_count)

    # -- 采样 -------------------------------------------------------------

    def mark(self, *, event_id: str, stage: str, label: str | None = None) -> None:
        """标记"接下来的状态属于哪个动作"。线程采样和同步采样都会读到它。"""
        with self._lock:
            self._current_event = {"event_id": event_id, "stage": stage}
        if self.event_log is not None and label is not None:
            self.event_log.write("event_begin", event_id=event_id, stage=stage, label=label)

    def sample(self, host_ns: int | None = None) -> None:
        """同步采一条。dry_run / replay 用。

        ``host_ns`` 给了就用它当行时间戳（调用方一般是"当前这一帧的时间戳"）。
        这一点很关键：干运行里帧的时间戳是**合成时钟**（虚拟时间轴），
        而 ``time.perf_counter_ns()`` 是真实时钟——两者在干运行下会越差越远，
        拿真实时钟去和帧时间轴对齐，会让分析层把保持阶段的行错配到运动前阶段，
        于是"RTDE 实际角变化"永远是 0。真机模式用的是后台线程采样，
        帧和 RTDE 都在真实时钟上，不存在这个问题。
        """
        if self._handle is None:
            return
        try:
            state = self.robot.read_state()
        except Exception as exc:
            if self.event_log is not None:
                self.event_log.write("rtde_read_failed", error=str(exc))
            return
        with self._lock:
            event = dict(self._current_event)
            row = state.to_row()
            if host_ns is not None:
                row["host_ns"] = int(host_ns)
            row["event_id"] = event["event_id"] or row.get("event_id") or ""
            row["stage"] = event["stage"] or row.get("stage") or ""
            self._writer.writerow(row)
            self._handle.flush()
            self.row_count += 1

    def _loop(self) -> None:
        period = 1.0 / self.hz if self.hz > 0 else 0.008
        next_time = time.perf_counter()
        while not self._stop.is_set():
            self.sample()
            next_time += period
            remaining = next_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # 采样跟不上设定频率时如实退让，不攒欠账（攒欠账会导致时间戳挤在一起）。
                next_time = time.perf_counter()


# --------------------------------------------------------------------------
# 运行目录
# --------------------------------------------------------------------------


@dataclass
class RunDirectory:
    """一次运行的输出目录。存在即报错，绝不覆盖。"""

    root: Path
    events: EventLog
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def vision_dir(self) -> Path:
        return self.root / "vision"

    @property
    def corners_dir(self) -> Path:
        return self.vision_dir / "corners"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    def segment_dir(self, segment_id: str) -> Path:
        return self.segments_dir / safe_name(segment_id)

    def corners_path(self, segment_id: str) -> Path:
        return self.corners_dir / f"{safe_name(segment_id)}.csv"

    def write_manifest(self) -> None:
        write_json(self.root / "run_manifest.json", self.manifest)

    def note(self, text: str) -> None:
        with (self.root / "notes.txt").open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


def create_run_directory(
    config: AppConfig,
    *,
    run_kind: str,
    extra: Mapping[str, Any] | None = None,
    stamp: str | None = None,
) -> RunDirectory:
    """建一次运行的输出目录。

    两步防覆盖：
    1. 目录名带时间戳，正常不会撞；
    2. 万一撞了（同一秒内再启动一次），直接报错，不往里写。
    """
    root = config.resolve_output_root()
    ok, message = check_free_disk(root, float(config.paths.min_free_disk_gb))
    if not ok:
        raise RecorderError(message)

    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    if run_dir.exists():
        raise RecorderError(
            f"输出目录已存在：{run_dir}\n"
            "为避免覆盖已有实验数据，本工具不会复用同名目录。"
            "请稍等一秒再启动，或改一个 outputs 路径。"
        )
    run_dir.mkdir(parents=True)
    (run_dir / "segments").mkdir()
    (run_dir / "vision" / "corners").mkdir(parents=True)
    (run_dir / "analysis").mkdir()

    from . import __version__

    manifest: dict[str, Any] = {
        "run_kind": str(run_kind),
        "run_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool_version": __version__,
        "mode": config.mode,
        "synthetic": config.mode == "dry_run",
        "collision_status": "unknown",
        "disk_check": message,
        "summary": config.describe(),
    }
    if extra:
        manifest.update(dict(extra))

    events = EventLog(path=run_dir / "events.jsonl", synthetic=config.mode == "dry_run")
    run = RunDirectory(root=run_dir, events=events, manifest=manifest)
    config.save(run_dir / "config.json")
    run.write_manifest()
    events.write("run_started", run_dir=str(run_dir), mode=config.mode)
    return run


def safe_name(text: str) -> str:
    """把事件编号变成安全的目录/文件名。

    只允许字母数字和 ``-_.``；其它字符换成下划线。需求九明确禁止乱码中文文件名，
    所以这里不用中文，也不做任何编码转换——事件编号本来就只有 ASCII。
    """
    cleaned = [ch if (ch.isalnum() and ch.isascii()) or ch in "-_." else "_" for ch in text]
    result = "".join(cleaned).strip("_")
    return result or "segment"


# --------------------------------------------------------------------------
# 通用写文件
# --------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - 防御
            pass
    raise TypeError(f"无法序列化为 JSON：{type(value).__name__}")


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """写 CSV。列顺序要么显式给出，要么取第一行的键顺序（保证可复现）。"""
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        if not materialized:
            raise RecorderError(f"{path}：没有任何行，也没有显式给出列名，无法决定表头。")
        columns = list(materialized[0].keys())
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)
    temp.replace(path)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)
    return path


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
