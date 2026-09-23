"""打 v1.0.2 源码交付包。

约定（沿用 v1.0.0 / v1.0.1）：
* 根目录扁平，条目清单以 v1.0.1 包为模板 + 1.0.2 新增的 2 个自测文件；
  ``docs/dry_run_example/`` 是**重跑过**的干运行快照，所以这一棵树按盘上实际情况
  重新扫（旧的三个中文名样例文件已经删了，新的 ASCII 名样例顶上来）；
* **没有动过的文件，一个字节都不重写**——直接从 v1.0.1 包里搬，
  这样复核的人拿两个包做二进制对比，看到的差异就只有真正改过的文件；
* 改过的文件按 .gitattributes 的换行策略写（文本一律 LF），
  vendor 五个原始文件因此和 v1.0.1 逐字节相同，证明"没有修改 vendor"。
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ZIP = ROOT / "delivery_v1.0.1" / "UR10_CB3_single_joint_pretest_v1_src_v1.0.1.zip"
NEW_ZIP = ROOT / "delivery_v1.0.2" / "UR10_CB3_single_joint_pretest_v1_src_v1.0.2.zip"

#: 干运行快照那一棵树整体按盘上重扫（1.0.2 重跑过，目录结构也变了）。
EXAMPLE_PREFIX = "docs/dry_run_example/"

#: v1.0.1 包里有、1.0.2 里已经删掉的条目。逐个列出来而不是"允许任何删除"：
#: 少一个文件都该是**有意**的，不能悄悄丢掉。
REMOVED = {
    "docs/dry_run_example/approach_previews/到位最后一点_预览图.png",
    "docs/dry_run_example/segments/组A_正方向第一级_中间帧.png",
    "docs/dry_run_example/segments/预实验_负方向0.2度_采集摘要.txt",
}

# v1.0.1 包里已有的 14 个自测文件，逐个列出来（而不是"从旧包里抄目录"），
# 这样 v1.0.2 新增的两个文件漏写会在下面的 stale 断言里立刻暴露。
NEW_TESTS = [
    "tests/test_analysis.py",
    "tests/test_config_roundtrip.py",
    "tests/test_depth_gate.py",
    "tests/test_dry_run_flow.py",
    "tests/test_estimates.py",
    "tests/test_group_pipeline.py",
    "tests/test_hardware_units.py",
    "tests/test_j6_eccentric.py",
    "tests/test_plan_invariants.py",
    "tests/test_replay.py",
    "tests/test_roi_consistency.py",
    "tests/test_rolling_delete.py",
    "tests/test_safety.py",
    "tests/test_settle_timeout.py",
    "tests/test_ui.py",
    "tests/test_vision_paths.py",
]

# .gitattributes：这些后缀在版本库里一律存 LF。
LF_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".ini", ".cfg", ".toml", ".jsonl"}
BINARY_SUFFIXES = {".png", ".jpg", ".raw", ".zip", ".pyc"}


def to_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def changed_files() -> set[str]:
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    out = set()
    for line in proc.stdout.splitlines():
        name = line[3:].strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        out.add(name.replace("\\", "/"))
    return out


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
    changed = changed_files()

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

    written_files = [n for n in order if not n.endswith("/")]
    stale = sorted(set(old.namelist()) - set(order))
    assert stale == sorted(REMOVED), (
        f"1.0.1 里有、1.0.2 里丢了的条目和预期不符："
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
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
            info.flag_bits |= 0x800  # UTF-8 文件名
            if name.endswith("/"):
                info.external_attr = (0o40755 << 16) | 0x10
                z.writestr(info, b"")
                continue
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            if name in changed:
                data = (ROOT / name).read_bytes()
                if Path(name).suffix.lower() in LF_SUFFIXES:
                    data = to_lf(data)
                rewritten += 1
            else:
                data = old.read(name)  # ★ 原样搬，保证未改动文件逐字节一致
                reused += 1
            z.writestr(info, data)

    print(f"写出 {NEW_ZIP.relative_to(ROOT)}")
    print(f"  条目 {len(order)}（目录 {sum(1 for n in order if n.endswith('/'))}）")
    print(f"  文件 {len(written_files)}：原样搬运 {reused}，本次重写 {rewritten}")
    print(f"  字节 {NEW_ZIP.stat().st_size}")
    return len(written_files)


if __name__ == "__main__":
    build()
