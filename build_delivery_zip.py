"""打 v1.0.3 源码交付包。

约定（沿用 v1.0.0 / v1.0.1 / v1.0.2）：
* 根目录扁平，条目清单以 v1.0.2 包为模板 + 1.0.3 新增的自测文件；
  ``docs/dry_run_example/`` 是**重跑过**的干运行快照，所以这一棵树按盘上实际情况
  重新扫；
* **没有动过的文件，一个字节都不重写**——直接从 v1.0.2 包里搬，
  这样复核的人拿两个包做二进制对比，看到的差异就只有真正改过的文件；
* 改过的文件按 .gitattributes 的换行策略写（文本一律 LF），
  vendor 五个原始文件因此和 v1.0.2 逐字节相同，证明"没有修改 vendor"。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ZIP = ROOT / "delivery_v1.0.2" / "UR10_CB3_single_joint_pretest_v1_src_v1.0.2.zip"
NEW_ZIP = ROOT / "delivery_v1.0.3" / "UR10_CB3_single_joint_pretest_v1_src_v1.0.3.zip"

#: 干运行快照那一棵树整体按盘上重扫（1.0.3 重跑过）。
EXAMPLE_PREFIX = "docs/dry_run_example/"

#: v1.0.3 新增的顶层文件（1.0.2 包里没有这一条，得显式列出来，
#: 否则"没在旧包清单里"的条目会被默默丢掉）。
#: 快照脚本跟着快照一起交付：复核的人要能自己把 `docs/dry_run_example/`
#: 从一次新的干运行里重做出来，而不是只能相信这份快照是手抄的。
NEW_ROOT_FILES = [
    "refresh_dry_run_snapshot.py",
    "SELF_REVIEW.md",  # 需求五：给复核者（GPT）看的那份自审，随包交付
]

#: v1.0.2 包里有、1.0.3 里已经删掉的条目。逐个列出来而不是"允许任何删除"：
#: 少一个文件都该是**有意**的，不能悄悄丢掉。
#: 1.0.3 没有删任何条目——RAW 一律整幅落盘，示例快照里的文件名也都还是 ASCII。
REMOVED: set[str] = set()

# v1.0.2 包里已有的 17 个自测文件 + v1.0.3 新增的 7 个，逐个列出来
# （而不是"从旧包里抄目录"），这样新增的文件漏写会在下面的 stale 断言里立刻暴露。
NEW_TESTS = [
    "tests/test_analysis.py",
    "tests/test_capture_check.py",      # 1.0.3 新增：连接设备后的 5 s 全屏采集检查
    "tests/test_config_roundtrip.py",
    "tests/test_defaults.py",           # 1.0.3 新增：交付默认值（删 RAW 默认开等）
    "tests/test_depth_gate.py",
    "tests/test_depth_stability.py",    # 1.0.3 新增：同一物理状态必须同一结论
    "tests/test_dry_run_flow.py",
    "tests/test_estimates.py",
    "tests/test_formal_repeat_grouping.py",  # 1.0.3 新增：正式实验按遍分组
    "tests/test_group_pipeline.py",
    "tests/test_hardware_units.py",
    "tests/test_j6_eccentric.py",
    "tests/test_plan_invariants.py",
    "tests/test_processing_state.py",   # 1.0.3 新增：处理期间 RTDE/相机与"不许运动"
    "tests/test_raw_status_order.py",   # 1.0.3 新增：处理→校验→删除→改写 JSON 的顺序
    "tests/test_replay.py",
    "tests/test_roi_consistency.py",
    "tests/test_rolling_delete.py",
    "tests/test_safety.py",
    "tests/test_settle_timeout.py",
    "tests/test_stepwise_estimates.py",  # 1.0.3 新增：按"上一目标姿态"逐段估算时长
    "tests/test_ui.py",
    "tests/test_vision_paths.py",
]

# .gitattributes：这些后缀在版本库里一律存 LF。
LF_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".ini", ".cfg", ".toml", ".jsonl"}
BINARY_SUFFIXES = {".png", ".jpg", ".raw", ".zip", ".pyc"}


def to_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _on_disk_bytes(name: str) -> bytes:
    """按 .gitattributes 的换行策略读盘上这一份（文本一律 LF）。"""
    data = (ROOT / name).read_bytes()
    if Path(name).suffix.lower() in LF_SUFFIXES:
        data = to_lf(data)
    return data


def _example_entries() -> list[str]:
    """按盘上实际情况列出干运行快照那一棵树（目录条目排在自己的内容之前）。"""
    root = ROOT / EXAMPLE_PREFIX.rstrip("/")
    assert root.is_dir(), f"干运行快照目录不见了：{root}"
    out: list[str] = [EXAMPLE_PREFIX]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        out.append(rel + "/" if path.is_dir() else rel)
    return out


def build() -> int:
    old = zipfile.ZipFile(OLD_ZIP)

    order: list[str] = []
    for name in old.namelist():
        if name.startswith(EXAMPLE_PREFIX):
            continue  # 这一棵树整体重扫（含根条目，由 _example_entries 重新给）
        if name.startswith("tests/") and name.endswith(".py") and name != "tests/conftest.py":
            continue  # 自测按 NEW_TESTS 重新排；conftest.py 不属于自测用例，原样搬
        order.append(name)
        if name == "docs/":
            order.extend(_example_entries())
        if name == "tests/":
            order.extend(sorted(set(NEW_TESTS)))
    for name in NEW_ROOT_FILES:
        assert name not in order, f"{name} 已经在旧包清单里了，别重复加"
        order.append(name)

    written_files = [n for n in order if not n.endswith("/")]
    stale = sorted(set(old.namelist()) - set(order))
    assert stale == sorted(REMOVED), (
        f"1.0.2 里有、1.0.3 里丢了的条目和预期不符："
        f"{sorted(set(stale) - REMOVED)} 是意外丢的，{sorted(REMOVED - set(stale))} 是预期要删却没删"
    )
    for name in REMOVED:
        assert not (ROOT / name).exists(), f"说好删掉的条目又回来了：{name}"
    missing_tests = sorted(set(NEW_TESTS) - set(order))
    assert missing_tests == [], f"清单里写了、包里没排进去的自测：{missing_tests}"

    dirty = [
        n for n in order
        if "__pycache__" in Path(n).parts or n.endswith(".pyc")
        or n.startswith("outputs/") or n.startswith("delivery_")
    ]
    assert dirty == [], f"包里混进了不该有的东西：{dirty}"

    for n in written_files:
        assert (ROOT / n).is_file(), f"磁盘上没有 {n}"

    NEW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    reused = rewritten = 0
    with zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 24, 0, 0, 0))
            info.flag_bits |= 0x800  # UTF-8 文件名
            if name.endswith("/"):
                info.external_attr = (0o40755 << 16) | 0x10
                z.writestr(info, b"")
                continue
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            disk = _on_disk_bytes(name)
            # ★ "没动过的文件原样搬"这件事**按内容判断，不看 git**：
            # 盘上这一份（按 .gitattributes 归一化过）与旧包里那一份逐字节相同
            # 就直接搬旧字节，否则就用盘上的。早先这里信的是 `git status`——
            # 一旦所有改动都提交了、工作区变干净，它就会把**改过的文件**也当成
            # "没动过"去旧包里翻，轻则 KeyError（新文件），重则悄悄发出旧内容。
            try:
                previous = old.read(name)
            except KeyError:
                previous = None  # 1.0.3 新增的条目，旧包里本来就没有
            if previous is not None and previous == disk:
                data = previous
                reused += 1
            else:
                data = disk
                rewritten += 1
            z.writestr(info, data)

    print(f"写出 {NEW_ZIP.relative_to(ROOT)}")
    print(f"  条目 {len(order)}（目录 {sum(1 for n in order if n.endswith('/'))}）")
    print(f"  文件 {len(written_files)}：与 1.0.2 逐字节相同 {reused}，本次重写 {rewritten}")
    print(f"  字节 {NEW_ZIP.stat().st_size}")
    return len(written_files)


if __name__ == "__main__":
    build()
