"""回放与复算：对**已经跑完**的一次运行目录做离线再处理。

这里有两个不同的"回放"，不要混为一谈
------------------------------------
1. **图像流回放**（在 :mod:`sj_pretest.sources` 里）——把历史图像当成"相机的输出"
   喂进和真机一样的采集流程，用来验证流程本身。它产出的是**新的采集数据**。
2. **结果复算**（本模块）——原始数据已经躺在磁盘上了，不需要再采一遍。
   改一个阈值、换一种剔除规则、或者只是想确认上次的结论能不能重现，
   都不应该要求重做实验。这是真实科研里最常用的那件事。

本模块做的是第 2 件事：读回 ``segments/`` 里的 RAW、``robot_states.csv``、
``analysis/trial_plan.json``，重新跑一遍离线角点识别和三层分析，
把结果写进**一个新的子目录**，绝不动原来的 ``analysis/``。

关于"绝不覆盖"
--------------
默认输出目录是 ``<原运行目录>/analysis/replay_<时间戳>/``。
原始采集目录一个字节都不改；上一次的复算结果也留着，可以对比"两次复算是否一致"。
如果用户显式给了 ``out_dir``，那个目录必须不存在——存在就报错，不往里写。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .analysis import (
    PretestReport,
    analyze_pretest,
    load_robot_states,
    write_report,
)
from .config import JOINT_NAMES, AppConfig
from .recorder import safe_name, write_text
from .vision import SegmentVision, process_segment


class ReplayError(RuntimeError):
    """回放/复算出错。消息中文。"""


#: 复算时读的试验清单（第一次分析时由实验会话写下）。
TRIAL_PLAN_NAME = "trial_plan.json"


@dataclass
class RunSummary:
    """一次历史运行的摘要。给人看，也给"选哪一次复算"用。"""

    path: Path
    run_kind: str
    mode: str
    created_at: str
    tool_version: str
    collision_status: str
    segments: int
    frames: int
    seconds: float
    has_trials: bool
    has_analysis: bool
    note: str = ""

    @property
    def synthetic(self) -> bool:
        return self.mode == "dry_run"

    def line(self) -> str:
        tag = " [合成数据]" if self.synthetic else ""
        bits = [
            self.created_at or "时间未知",
            f"{self.run_kind}/{self.mode}{tag}",
            f"{self.segments} 段 / {self.frames} 帧 / {self.seconds:.1f} s",
        ]
        if self.has_trials:
            bits.append("有试验清单")
        if self.has_analysis:
            bits.append("已有分析结果")
        if self.note:
            bits.append(self.note)
        return "｜".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "run_kind": self.run_kind,
            "mode": self.mode,
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "collision_status": self.collision_status,
            "segments": int(self.segments),
            "frames": int(self.frames),
            "seconds": float(self.seconds),
            "has_trials": bool(self.has_trials),
            "has_analysis": bool(self.has_analysis),
            "synthetic": self.synthetic,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# 发现与读取历史运行
# --------------------------------------------------------------------------


def discover_runs(root: str | Path) -> list[RunSummary]:
    """列出 ``root`` 下面所有带 ``run_manifest.json`` 的运行目录，按时间倒序。

    只认带清单的目录：没有清单的目录可能是用户手动拷进来的，宁可看不见，
    也不要猜它的结构然后分析出错误的结论。
    """
    base = Path(root).expanduser()
    if not base.exists():
        return []
    found: list[RunSummary] = []
    for manifest_path in sorted(base.glob("*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found.append(_summarize(manifest_path.parent, manifest))
    # 目录名就是时间戳，倒序即"最近的在前"；名字相同再按创建时间兜底。
    found.sort(key=lambda item: (item.path.name, item.created_at), reverse=True)
    return found


def _summarize(run_dir: Path, manifest: Mapping[str, Any]) -> RunSummary:
    segments, frames, seconds, note = _count_segments(run_dir)
    return RunSummary(
        path=run_dir,
        run_kind=str(manifest.get("run_kind", "")),
        mode=str(manifest.get("mode", "")),
        created_at=str(manifest.get("created_at", "")),
        tool_version=str(manifest.get("tool_version", "")),
        collision_status=str(manifest.get("collision_status", "unknown")),
        segments=segments,
        frames=frames,
        seconds=seconds,
        has_trials=(run_dir / "analysis" / TRIAL_PLAN_NAME).is_file(),
        has_analysis=(run_dir / "analysis" / "pretest_report.json").is_file(),
        note=note,
    )


def _count_segments(run_dir: Path) -> tuple[int, int, float, str]:
    """数一数这个运行目录里有多少段、多少帧。只读 metadata，不读 RAW。"""
    segments = frames = 0
    seconds = 0.0
    broken = 0
    for metadata_path in sorted((run_dir / "segments").glob("*/capture_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            broken += 1
            continue
        segments += 1
        frames += int(metadata.get("frame_count", 0) or 0)
        seconds += float(metadata.get("content_seconds", 0.0) or 0.0)
    note = f"{broken} 段元数据读不动" if broken else ""
    return segments, frames, seconds, note


def load_run(run_dir: str | Path) -> tuple[AppConfig, dict[str, Any]]:
    """读回一次运行当时的配置和清单。

    配置用的是**当时落盘的 config.json**，不是现在的默认值——否则"复算"就变成了
    "拿新参数套旧数据"，结论变化时说不清是数据的功劳还是参数的功劳。
    """
    directory = Path(run_dir).expanduser()
    if not directory.is_dir():
        raise ReplayError(f"运行目录不存在：{directory}")
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise ReplayError(
            f"{directory} 里没有 run_manifest.json，不像是本工具的运行目录。\n"
            "如果是别的程序采的数据，请用回放模式（把图像喂进采集流程），"
            "而不是复算模式。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = directory / "config.json"
    if config_path.is_file():
        config = AppConfig.load(config_path)
    else:
        raise ReplayError(
            f"{directory} 里没有 config.json，无法还原当时的参数，也就无法可靠复算。"
        )
    return config, manifest


def find_run_root(path: str | Path) -> Path | None:
    """从一份历史数据往上找它所属的运行目录（含 robot_states.csv 的那一层）。

    回放的数据可能是 ``<运行目录>/segments/<段名>/``，也可能是任意一个装着图片的
    目录。只有前者才找得到机器人记录。找不到就返回 None——由调用方决定怎么如实
    说明"这份数据里没有关节角记录"，而不是编一份出来。
    """
    current = Path(path).expanduser()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "robot_states.csv").is_file():
            return candidate
    return None


def load_joint_records(
    path: str | Path,
) -> list[tuple[int, dict[str, float], dict[str, Any]]]:
    """把一次历史运行的 robot_states.csv 还原成回放用的"机器人记录"。

    回放要能回答"这一刻关节角是多少"，否则第一层分析（指令→实际）无从算起。
    取**实际角**（没有实际角才退用指令角，并在 extra 里标出来）；时间用
    ``host_ns``——和 RAW 的逐帧时间戳同一个基准，两边对得上。
    """
    root = find_run_root(path)
    if root is None:
        return []
    rows = load_robot_states(root / "robot_states.csv")
    records: list[tuple[int, dict[str, float], dict[str, Any]]] = []
    for row in rows:
        host_ns = row.get("host_ns")
        if host_ns is None:
            continue
        angles: dict[str, float] = {}
        source = "actual"
        for name in JOINT_NAMES:
            value = row.get(f"actual_q_{name}_deg")
            if value is None:
                value = row.get(f"command_q_{name}_deg")
                source = "commanded"
            if value is None:
                angles = {}
                break
            angles[name] = float(value)
        if not angles:
            continue
        tcp = [row.get(f"actual_tcp_{axis}") for axis in ("x", "y", "z", "rx", "ry", "rz")]
        extra = {
            "actual_tcp_pose": None if any(v is None for v in tcp) else [float(v) for v in tcp],
            "robot_mode": row.get("robot_mode"),
            "safety_mode": row.get("safety_mode"),
            "runtime_state": row.get("runtime_state"),
            "event_id": row.get("event_id"),
            "stage": row.get("stage"),
            "angles_source": source,
        }
        records.append((int(host_ns), angles, extra))
    records.sort(key=lambda item: item[0])
    return records


def load_trials(run_dir: str | Path) -> list[dict[str, Any]]:
    """读回试验清单。没有就报错——没有清单就无法知道哪段对应哪个幅度/方向。"""
    path = Path(run_dir).expanduser() / "analysis" / TRIAL_PLAN_NAME
    if not path.is_file():
        raise ReplayError(
            f"没有试验清单：{path}\n"
            "这说明这次运行没有跑过（或没有跑完）三档预实验，没有可复算的统计试验。"
            "静态基线仍然可以用 read_static_segments() 单独看。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ReplayError(f"试验清单是空的：{path}")
    return [dict(item) for item in payload]


def find_static_segments(run_dir: str | Path) -> list[str]:
    """找出所有静态基线段（按目录名排序，多次基线按发生顺序返回）。"""
    found: list[tuple[str, str]] = []
    for metadata_path in sorted((Path(run_dir).expanduser() / "segments").glob("*/capture_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(metadata.get("segment_kind", "")) == "static":
            found.append((str(metadata.get("created_at", "")), str(metadata.get("segment_id", metadata_path.parent.name))))
    return [segment_id for _created, segment_id in found]


# --------------------------------------------------------------------------
# 复算
# --------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """一次复算的结果。"""

    run_dir: Path
    out_dir: Path
    report: PretestReport | None
    segments: dict[str, SegmentVision] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    stride: int = 1
    seconds: float = 0.0
    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def to_text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


def _default_out_dir(run_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return run_dir / "analysis" / f"replay_{stamp}"


def reanalyze_run(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    stride: int = 1,
    config: AppConfig | None = None,
    progress: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ReplayResult:
    """对一次已完成的运行做离线复算。

    与"重新采集一遍"的区别：这里一行运动指令都不会下发，也不碰相机。
    它只是把磁盘上的 RAW 重新识别一遍，重算三层分析和推荐步长。

    ``config`` 一般不用传（默认用当时落盘的 config.json）；显式传入是为了
    "同一份数据、换一组阈值"这种对比复算。
    """
    directory = Path(run_dir).expanduser()
    started = time.perf_counter()
    if config is None:
        config, _manifest = load_run(directory)
    trials = load_trials(directory)
    static_ids = find_static_segments(directory)

    target = Path(out_dir).expanduser() if out_dir is not None else _default_out_dir(directory)
    if target.exists():
        raise ReplayError(
            f"复算输出目录已存在：{target}\n"
            "本工具不覆盖已有结果（包括上一次的复算结果）。请换一个目录名。"
        )
    target.mkdir(parents=True)

    result = ReplayResult(
        run_dir=directory,
        out_dir=target,
        report=None,
        stride=int(stride),
    )
    result.add(f"复算来源：{directory}")
    result.add(f"复算输出：{target}（原目录里的东西一个都不改）")
    result.add(
        f"数据规模：{len(static_ids)} 段静态基线 + {len(trials)} 次统计试验；"
        f"抽帧步长 stride={int(stride)}。"
    )
    if config.mode == "dry_run":
        result.add("注意：这份数据是**合成数据**，复算只能验证流程与判据。")

    # 复算不再写 corners/ 到原目录：写进去会盖掉第一次的识别结果，
    # 而"两次识别是否一致"本身就是有用的信息，所以复算的角点单独放。
    corners_dir = target / "corners"
    metrics_path = target / "metrics.csv"

    todo: list[tuple[str, str]] = [(segment_id, "static") for segment_id in static_ids]
    todo += [(str(trial["segment_id"]), "trial") for trial in trials]

    for position, (segment_id, kind) in enumerate(todo, start=1):
        if stop_requested is not None and stop_requested():
            result.add(f"操作者要求停止：已在第 {position} 段之前停下，已完成的 {len(result.segments)} 段结果保留。")
            break
        # 目录名是**净化过**的（例如事件编号里的 “+1” 落盘成 “_1”），
        # 直接用事件编号去拼路径会找不到目录——正方向那一半试验会整批丢掉，
        # 而且丢得悄无声息。这里走和采集时同一个净化函数。
        segment_dir = directory / "segments" / safe_name(segment_id)
        if not segment_dir.is_dir():
            result.failed[segment_id] = "采集目录不存在"
            result.add(
                f"{segment_id}（{kind}）：采集目录不存在（找的是 {segment_dir.name}），跳过。"
            )
            continue
        message = f"离线识别 {position}/{len(todo)}：{segment_id}（{kind}）"
        result.add(message)
        if progress is not None:
            progress(message)
        try:
            result.segments[segment_id] = process_segment(
                segment_dir,
                config=config,
                segment_id=segment_id,
                save_corners=True,
                corners_dir=corners_dir,
                metrics_path=metrics_path,
                stride=int(stride),
                stop_requested=stop_requested,
            )
        except Exception as exc:
            result.failed[segment_id] = f"{type(exc).__name__}: {exc}"
            result.add(f"    {segment_id} 识别失败：{exc}")
            continue

    if not result.segments:
        result.add("一段都没识别成功，无法复算分析。原始数据没有被改动。")
        write_text(target / "replay_notes.txt", result.to_text())
        result.seconds = time.perf_counter() - started
        return result

    states_path = directory / "robot_states.csv"
    rows = load_robot_states(states_path) if states_path.is_file() else []
    if not rows:
        result.add(
            f"没有读到机器人状态（{states_path}）：只有第二、三层分析能算，"
            "第一层（指令→实际）会全部标为无法评估。"
        )

    report = analyze_pretest(
        config=config,
        segments={sid: vision for sid, vision in result.segments.items()},
        trials=trials,
        rtde_rows=rows,
        static_segment_ids=static_ids,
    )
    write_report(report, target)
    write_text(target / "replay_notes.txt", result.to_text())
    result.report = report
    result.seconds = time.perf_counter() - started
    return result


# --------------------------------------------------------------------------
# 回放的"能不能用"检查
# --------------------------------------------------------------------------


def describe_replay_data(path: str | Path) -> list[str]:
    """给一份历史数据做个"能不能拿来回放"的体检，返回人可读的几行。

    只报事实，不报结论。真正跑起来时上游代码还会再检查一次（这里是给人看提示用的）。
    """
    from .sources import resolve_replay_target

    target = resolve_replay_target(path)
    lines = [f"回放数据：{target.path}", f"形态：{target.note}"]

    if target.kind == "raw":
        metadata_path = target.path / "capture_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            lines.append(
                f"已采集：{metadata.get('frame_count')} 帧，"
                f"{metadata.get('width')}×{metadata.get('height')}，"
                f"{float(metadata.get('content_seconds', 0.0)):.2f} s，"
                f"缺帧 {metadata.get('missing_frame_count', 0)}。"
            )
            phases = metadata.get("phases") or []
            if phases:
                names = "、".join(str(item.get("label")) for item in phases)
                lines.append(f"阶段：{names}（回放时按相位切窗口，和真机同一套逻辑）。")
        else:
            lines.append(
                "这个目录里没有 capture_metadata.json，回放时按“时长已到”结束各段。"
            )
    elif target.kind == "video":
        lines.append("视频没有逐帧采集时间戳，各段只能按时长切分，相位时间会比真机粗。")
    else:
        images = sorted(
            child
            for child in target.path.iterdir()
            if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        names = ", ".join(child.name for child in images[:3])
        lines.append(f"前几张：{names}{' …' if len(images) > 3 else ''}")
        lines.append("图像目录没有原始采集时间戳，按文件名顺序当成等间隔处理。")

    if not has_frame_timestamps(target.path, target.kind):
        lines.append(
            "提醒：这份数据没有逐帧时间戳。本工具仍然能出曲线，但“秒”这一维只是"
            "按名义帧率换算出来的，涉及时间的判据（停稳时间、残余衰减）要打折看。"
        )
    return lines


def has_frame_timestamps(path: Path, kind: str) -> bool:
    """这份历史数据里有没有**逐帧的采集时间戳**。

    只有 RAW 采集目录有（``frame_timestamps.csv`` 是被复用代码写的格式）。
    视频和图像目录都没有，它们的"秒"是名义帧率换算出来的——这件事必须
    明确告诉用户，否则"停稳用了 0.13 s"这种结论会被当真。
    """
    if kind == "raw":
        return (Path(path) / "frame_timestamps.csv").is_file()
    return False


def missing_analysis_inputs(run_dir: str | Path) -> list[str]:
    """复算之前先看看缺什么，返回缺失清单（空列表表示齐了）。"""
    directory = Path(run_dir).expanduser()
    missing: list[str] = []
    if not (directory / "config.json").is_file():
        missing.append("config.json（无法还原当时的参数）")
    if not (directory / "robot_states.csv").is_file():
        missing.append("robot_states.csv（第一层分析算不了）")
    if not (directory / "analysis" / TRIAL_PLAN_NAME).is_file():
        missing.append(f"analysis/{TRIAL_PLAN_NAME}（不知道哪段是哪个幅度）")
    if not list((directory / "segments").glob("*/frames.raw")):
        missing.append("segments/*/frames.raw（没有任何原始帧）")
    return missing


def static_noise_of(result: ReplayResult) -> Sequence[Any]:
    """复算结果里的静态噪声（可能为 None）。给界面显示用。"""
    return [] if result.report is None or result.report.static_noise is None else [result.report.static_noise]
