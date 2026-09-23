"""把一次真实的干运行结果收成 docs/dry_run_example/ 那份**删减**快照。

快照的原则（沿用 v1.0.0 起的老规矩，也符合"交付包少图多文字"）：
文本文件尽量齐、图片每类只留一张样例、RAW 一律不放。
每次重跑干运行之后手工执行本脚本，避免"抄文件时漏掉一个"这类错误。

    python refresh_dry_run_snapshot.py <跑出来的 run 目录>

脚本**只写** docs/dry_run_example/，不碰别的地方。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "docs" / "dry_run_example"

#: 文本文件整份照搬。
TEXT_FILES = [
    "config.json",
    "run_manifest.json",
    "notes.txt",
    "events.jsonl",
    "analysis/pretest_report.txt",
    "analysis/pretest_report.json",
    "analysis/amplitudes.csv",
    "analysis/trials.csv",
    "analysis/directions.csv",
    "analysis/sensitivity.csv",
    "analysis/trial_plan.csv",
    "analysis/trial_plan.json",
    "analysis/analysis_notes.txt",
    "vision/metrics.csv",
    # v1.0.3 新增：连上设备后的 5 s 全屏采集检查，只留汇总（RAW 和样本图另算）。
    "connection_check/connection_check.json",
    "connection_check/connection_check.txt",
    "connection_check/capture_summary.txt",
]

#: 每类只留一张样例图。
SAMPLE_FILES = [
    "approach_previews/06_approach-all-ana-d0-r06-move-pt06.png",
    "roi_checks/001_segment.png",
    "segments/formal_a-J1-a0p05-d-1-r01-return-down01/sample_middle.png",
    # 采集检查的样本帧也只留中间那一张。
    "connection_check/sample_middle.png",
]

#: 逐段结果，按文件名挑几个代表（每类留一个）。
SEGMENT_GLOBS = [
    "pretest-J1-a0p01-d-1-r01-move-to",
    "formal_a-J1-a0p05-d-1-r01-return-down01",
]
SEGMENT_FILES = ["segment_vision.json", "capture_summary.txt"]

#: 角点 CSV 单张就有几百 KB，只留一段的。
CORNERS = ["vision/corners/pretest-J1-a0p01-d-1-r01-move-to.csv"]

ROBOT_STATES_HEAD_LINES = 200


def copy_text(src: Path, rel: str) -> None:
    origin = src / rel
    assert origin.is_file(), f"干运行结果里没有 {rel}"
    target = DEST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]).resolve()
    assert src.is_dir(), f"找不到 run 目录：{src}"
    assert (src / "run_manifest.json").is_file(), f"{src} 不像 run 目录"

    # 先清掉上一版快照的**内容**，只留 README.txt（它是人写的说明）。
    for child in sorted(DEST.iterdir()):
        if child.name == "README.txt":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for rel in TEXT_FILES:
        copy_text(src, rel)
    for rel in SAMPLE_FILES:
        origin = src / rel
        assert origin.is_file(), f"干运行结果里没有 {rel}"
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    for segment in SEGMENT_GLOBS:
        for name in SEGMENT_FILES:
            copy_text(src, f"segments/{segment}/{name}")
    for rel in CORNERS:
        copy_text(src, rel)

    # RTDE 状态流只留前 200 行（完整的是几十万行）。
    lines = (src / "robot_states.csv").read_text(encoding="utf-8").splitlines()
    head = lines[:ROBOT_STATES_HEAD_LINES]
    (DEST / "robot_states_head.csv").write_text("\n".join(head) + "\n", encoding="utf-8")

    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"快照已刷新：{DEST}")
    print(f"  文件 {len(files)} 个，合计 {total / 1024:.0f} KB")
    for p in files:
        print(f"  {p.relative_to(DEST).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
