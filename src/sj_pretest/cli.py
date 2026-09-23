"""命令行入口：不打开界面也能把同一套流程跑完。

为什么要有它
------------
1. 自测要能在没有任何窗口的环境里跑完整流程（需求八的十二条自测里有一半是流程性的）。
2. 真实实验里"复算"这件事，命令行比点界面顺手：改一个阈值、重算一遍、对比两份结果。
3. 界面上做错的判据，命令行里会以同样的方式错；两条路走同一套会话层，不会各说各话。

``dry-run`` 子命令**强制** ``mode="dry_run"``：它连 RTDE 都不连，
所以不能用来碰真机（想碰真机请用界面，那里有逐步确认）。这不是省事，
是"命令行里没有人工确认的闸门"这个事实决定的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig
from .experiment import ExperimentSession, SessionHooks
from .replay import (
    ReplayError,
    describe_replay_data,
    discover_runs,
    missing_analysis_inputs,
    reanalyze_run,
)
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sj-pretest",
        description="UR10 CB3 单关节微动预实验工具（默认不下发任何真实运动）",
    )
    parser.add_argument("--version", action="version", version=f"sj-pretest {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ui = sub.add_parser("ui", help="打开一体化界面")
    ui.add_argument("--config")
    ui.add_argument("--mode", choices=("dry_run", "replay", "hardware"))

    defaults = sub.add_parser("defaults", help="写出默认配置 JSON")
    defaults.add_argument("--out", default="configs/experiment_default.json")

    dry = sub.add_parser(
        "dry-run", help="用合成世界跑完整流程（不连任何硬件；自动回答人工确认）"
    )
    dry.add_argument("--config")
    dry.add_argument("--output-root", default="outputs")
    dry.add_argument("--stamp", default="dryrun_cli")
    dry.add_argument("--joints", default="J1", help="逗号分隔，例如 J1,J6")
    dry.add_argument("--amplitudes", default="0.01,0.05,0.2")
    dry.add_argument("--repeats", type=int, default=1)
    dry.add_argument("--stride", type=int, default=8, help="离线识别抽帧步长")
    dry.add_argument("--formal-joints", default="J1", help="正式实验跑哪些关节")
    dry.add_argument("--formal-step", type=float, default=0.2)
    dry.add_argument("--group", default="A", choices=("A", "B", "none"))
    dry.add_argument("--no-approach", action="store_true", help="跳过到位过程")
    dry.add_argument("--json", help="把结果摘要写成 JSON 到这个路径")

    analyze = sub.add_parser("reanalyze", help="对已完成的运行目录做离线复算")
    analyze.add_argument("run_dir")
    analyze.add_argument("--out")
    analyze.add_argument("--stride", type=int, default=1)
    analyze.add_argument("--config", help="用另一份配置复算（例如只改阈值）")

    report = sub.add_parser("report", help="打印某次运行已有的分析结果")
    report.add_argument("run_dir")

    check = sub.add_parser("replay-check", help="体检一份历史数据能不能回放")
    check.add_argument("path")

    runs = sub.add_parser("runs", help="列出输出目录下的历史运行")
    runs.add_argument("--root", default="outputs")

    return parser


# --------------------------------------------------------------------------
# 子命令实现
# --------------------------------------------------------------------------


def cmd_defaults(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.mode = "dry_run"
    config.validate()
    path = config.save(args.out)
    print(f"默认配置已写出：{path}")
    print(json.dumps(config.describe(), ensure_ascii=False, indent=2))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config) if args.config else AppConfig()
    # 硬保证：这个子命令永远不碰硬件。load 进来的配置若写着 hardware，也按合成跑。
    config.mode = "dry_run"
    config.paths.output_root = args.output_root
    config.dry_run.realtime = False
    config.dry_run.shorten_durations = True
    config.pretest.joints = _split(args.joints)
    config.pretest.amplitudes_deg = [float(x) for x in _split(args.amplitudes)]
    config.pretest.repeats_per_direction = int(args.repeats)
    formal_joints = _split(args.formal_joints) if args.formal_joints else []
    steps = {joint: float(args.formal_step) for joint in formal_joints}
    config.formal.step_deg = steps or dict(config.formal.step_deg)
    config.formal.staircase_n = min(int(config.formal.staircase_n), 2)
    config.formal.repeats = min(int(config.formal.repeats), 1)
    config.validate()

    log_lines: list[str] = []
    confirmations = {"count": 0, "labels": []}

    def confirm(label: str) -> bool:
        confirmations["count"] += 1
        confirmations["labels"].append(label.splitlines()[0])
        return True

    def log(message: str) -> None:
        log_lines.append(str(message))
        print(str(message), flush=True)

    hooks = SessionHooks(confirm=confirm, on_log=log)
    summary: dict[str, Any] = {"mode": config.mode}
    with ExperimentSession(config, hooks=hooks, stamp=args.stamp) as session:
        session.open(run_kind="dry_run")
        session.connect_devices()
        if not args.no_approach:
            result = session.run_approach()
            # 到位过程不产生"采集段"（它每步只落一张预览图、不存 RAW），
            # 所以这里报的是**下发了几个中间点**，数的是计划里的目标点数。
            # 用 result.segments 的话永远是 0，读起来像是"一步都没走"。
            summary["approach_steps"] = sum(
                1 for step in session.approach_plan().steps if step.target_joint_deg is not None
            )
            summary["approach_previews"] = len(session.approach_frames)
        session.run_static()
        probes, failed = session.run_quick_probes()
        summary["quick_probe_failed"] = list(failed)
        pretest = session.run_pretest()
        summary["pretest_segments"] = len(pretest.segments)
        report = session.analyze_offline(stride=int(args.stride))
        summary["recommended_deg"] = {
            item.joint: item.recommended_deg for item in report.recommendations
        }
        summary["needs_manual_input"] = [
            item.joint for item in report.recommendations if item.needs_manual_input
        ]
        if steps:
            summary["range_check"] = session.formal_range_checks(steps)
        if args.group.upper() in ("A", "B") and steps:
            formal = session.run_formal(args.group.upper(), step_deg=steps)
            summary["formal_segments"] = len(formal.segments)
        summary["run_dir"] = str(session.run.root if session.run else "")
        summary["aborted"] = session.aborted
    summary["confirmations"] = confirmations["count"]
    summary["synthetic"] = True

    print("\n===== 干运行摘要 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"摘要已写出：{args.json}")
    print("提醒：以上全部是**合成数据**，只能说明流程和判据能跑通，不是实验结果。")
    return 0


def cmd_reanalyze(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config) if args.config else None
    missing = missing_analysis_inputs(args.run_dir)
    if missing:
        print("这份运行缺少复算需要的文件：")
        for item in missing:
            print(f"  - {item}")
        print("仍然可以试着复算，能算多少算多少。")
    result = reanalyze_run(
        args.run_dir, out_dir=args.out, stride=int(args.stride), config=config
    )
    for line in result.lines:
        print(line)
    if result.report is not None:
        print("\n===== 分析摘要 =====")
        for line in result.report.summary_lines():
            print(line)
    if result.failed:
        print("\n识别失败的段（原始数据未改动）：")
        for key, value in result.failed.items():
            print(f"  - {key}：{value}")
    print(f"\n复算结果目录：{result.out_dir}")
    print(f"耗时 {result.seconds:.1f} s。")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    directory = Path(args.run_dir)
    text = directory / "analysis" / "pretest_report.txt"
    json_path = directory / "analysis" / "pretest_report.json"
    if text.is_file():
        print(text.read_text(encoding="utf-8"))
    elif json_path.is_file():
        print(json_path.read_text(encoding="utf-8"))
    else:
        print(f"{directory} 里还没有分析结果。可以先用 reanalyze 复算一次。")
        return 1
    print(f"（结果来自 {directory}；真实碰撞状态：unknown）")
    return 0


def cmd_replay_check(args: argparse.Namespace) -> int:
    for line in describe_replay_data(args.path):
        print(line)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    found = discover_runs(args.root)
    if not found:
        print(f"{args.root} 下面没有带 run_manifest.json 的运行目录。")
        return 0
    for item in found:
        print(f"{item.path.name}  {item.line()}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import main as ui_main

    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.mode:
        argv += ["--mode", args.mode]
    return ui_main(argv)


def _split(text: str) -> list[str]:
    return [piece.strip() for piece in str(text).replace("，", ",").split(",") if piece.strip()]


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handlers = {
        "defaults": cmd_defaults,
        "dry-run": cmd_dry_run,
        "reanalyze": cmd_reanalyze,
        "report": cmd_report,
        "replay-check": cmd_replay_check,
        "runs": cmd_runs,
        "ui": cmd_ui,
    }
    try:
        return int(handlers[args.command](args))
    except (ReplayError, RuntimeError, OSError, ValueError) as exc:
        print(f"出错了：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - 手动启动时才走到
    sys.exit(main())
