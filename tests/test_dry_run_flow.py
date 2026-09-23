"""需求八.2/八.12：干运行把整套流程走通，且输出目录绝不覆盖旧实验。

这里是"三个按钮"的端到端版本：到位 → 静态基线 → 快速几何检查 → 三档预实验
→ 离线分析 → 理论范围检查 → 组 A → 组 B。全程 dry_run，一次都不碰真机。

规模按自测缩小（关节数、幅度档数、重复次数、中间点数），**判据一条都不动**。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from sj_pretest.config import JOINT_NAMES
from sj_pretest.experiment import ExperimentSession, ExperimentError
from sj_pretest.recorder import RecorderError, create_run_directory

from conftest import (
    Recorder,
    build_config,
    open_session,
    read_csv_rows,
    read_json,
    run_dir_of,
    segments_of,
)


@pytest.fixture
def small_config(tmp_path: Path):
    """两个关节（一个平移路径、一个旋转路径）× 两档幅度 × 单次重复。"""
    config = build_config(
        tmp_path,
        joints=("J1", "J6"),
        amplitudes=(0.05, 0.2),
        repeats=1,
    )
    # 正式实验按"预实验推荐的那一档"走，规模压到两级阶梯、单次重复。
    config.formal.staircase_n = 2
    config.formal.repeats = 1
    config.formal.step_deg = {"J1": 0.2, "J6": 0.2}
    config.pretest.quick_probe_deg = 0.05
    config.validate()
    return config


def test_full_dry_run_flow_end_to_end(tmp_path: Path, small_config) -> None:
    """整套流程跑通，并且每一步的结论都能在落盘的文件里核对。"""
    recorder = Recorder(answer=True)
    session, recorder = open_session(small_config, recorder=recorder, run_kind="dry-run")

    # -- 按钮一：连接设备、到位 ----------------------------------------
    facts = session.connect_devices()
    assert facts["mode"] == "dry_run"
    assert facts["collision_status"] == "unknown"
    assert facts["camera_note"]
    assert facts["current_joint_deg"] == pytest.approx(
        list(small_config.robot.nominal_joint_deg), abs=1e-6
    )

    # 自测装置：先把机械臂停到别的位置，模拟"现场开机时它不在实验姿态上"。
    # 真机上这一步由示教器或上一个程序决定，本工具管不着；这里只是把
    # "到位过程要真的走一段"这件事放进流程里，否则中间点全是零位移。
    robot = session.robot
    assert robot is not None
    parked = list(small_config.robot.nominal_joint_deg)
    parked[0] += 2.0
    parked[1] -= 1.5
    robot.move_to_joint(
        parked, speed_deg_s=5.0, accel_deg_s2=5.0, event_id="test-park", label="自测：先停到别处"
    )
    assert robot.wait_until_settled(
        parked,
        tolerance_deg=float(small_config.robot.settle_tolerance_deg),
        hold_s=float(small_config.robot.settle_hold_s),
        timeout_s=float(small_config.robot.settle_timeout_s),
    ), "自测装置没走到停放姿态"
    assert session.approach_needed() is True, "已经离开了实验姿态，应当要求逐步到位"

    result = session.run_approach()
    assert session.approach_needed() is False, "走完到位过程，实际角应当已经在名义位姿上"
    assert session.approach_frames, "到位过程一张预览图都没落盘"
    assert [line for line in result.lines if "画面检查" in line]
    # 每一步都要显示"相对当前姿态"的增量，而且这一次是真的有增量的。
    deltas = [line for line in recorder.logs if "本次增量（相对当前姿态）" in line]
    assert len(deltas) == len(session.approach_frames)
    numbers = [
        float(value) for line in deltas for value in re.findall(r"[+-]\d+\.\d+", line)
    ]
    assert any(abs(value) > 0.5 for value in numbers), (
        "到位过程每一步都显示零增量——中间点计划是空的"
    )

    # -- 按钮二 A：静态噪声基线 ----------------------------------------
    static = session.capture_static(segment_id="static_base", duration_s=0.6)
    assert static.frame_count > 0
    assert static.dropped_ratio == 0.0

    # -- 按钮二 B：快速几何检查 ----------------------------------------
    probe_result, failed = session.run_quick_probes()
    assert probe_result.segments, "快速几何检查一段都没采"
    assert [line for line in probe_result.lines if "快速几何检查" in line]
    # 合成棋盘格是 16 px/格（半径约 90 px）：J6 转 0.05° 只让角点沿切向挪
    # 约 0.08 px，和合成世界的角点噪声同一个量级——所以 J6 这一档
    # **在合成世界里本来就看不出来**（真机上棋盘格在画面里大得多，是另一回事）。
    # 这正是需求三.B 要拦的那种情况：工具必须停下来把话说清楚，而不是自己往下走。
    # 所以这里断言的不是"全都通过"，而是"该拦的拦住了、话说清楚了、
    # 而且**没有**替人做决定继续跑"。
    assert failed == ["J6"], f"合成世界里应当只有 J6 的 0.05° 探针看不出来，实际 {failed}"
    assert probe_result.aborted is True, "快速几何检查没过却标成正常结束"
    assert "看不出来" in "\n".join(probe_result.lines), "没有说明是哪一条判据没过"
    assert "暂停" in "\n".join(probe_result.lines)
    # 拦下来之后仍然可以人工决定继续——需求三.B 说的是"暂停等人处理"，
    # 不是"禁止继续"。所以下面照常跑三档预实验。

    # -- 按钮二 C：三档微动预实验 --------------------------------------
    pretest = session.run_pretest()
    stats = [t for t in session.trials if t["stage"] == "pretest"]
    assert len(stats) == 2 * 2 * 2, (
        f"2 关节 × 2 档 × 2 方向 × 1 次重复 = 8 次统计试验，实际 {len(stats)}"
    )
    assert len(pretest.segments) == len(stats)
    for trial in session.trials:
        assert trial["dropped_ratio"] == 0.0, f"{trial['event_id']} 掉帧了"

    # -- 按钮三：离线分析 ----------------------------------------------
    report = session.analyze_offline(stride=4)
    assert report.static_noise is not None, "静态基线没有进分析"
    assert len(report.trials) == len(stats)
    by_joint = {r.joint: r for r in report.recommendations}
    assert set(by_joint) == set(JOINT_NAMES)
    candidates = {0.05, 0.2}
    for joint in ("J1", "J6"):
        recommendation = by_joint[joint]
        assert recommendation.recommended_deg in candidates or (
            recommendation.recommended_deg is None and recommendation.needs_manual_input
        ), f"{joint} 推荐了候选幅度以外的步长：{recommendation.recommended_deg}"
    # 没跑预实验的关节要说"没跑"，不能说"测了但都不合格"。
    for joint in ("J2", "J3", "J4", "J5"):
        recommendation = by_joint[joint]
        assert recommendation.recommended_deg is None
        assert "没有跑预实验" in recommendation.reason
        assert recommendation.detail_lines == []

    # 灵敏度要和合成世界的真值对得上（这是判据正确性的证据，不是拟合）。
    # 两条路径共用同一个字段名 px_per_deg，但 **J6 的单位是 °/°**
    # （二维转角 ÷ 关节角变化）——单位写在 source 和 note 里。
    j1 = report.sensitivities[("J1", 1)]
    assert j1.source == "px_per_deg_from_actual_q"
    assert j1.px_per_deg == pytest.approx(
        float(small_config.dry_run.centroid_px_per_deg["J1"]), rel=0.25
    ), f"J1 实测灵敏度 {j1.px_per_deg} px/° 与真值相差超过 25%"

    j6 = report.sensitivities[("J6", 1)]
    assert j6.source == "rotation_deg_per_deg"
    assert "°/°" in j6.note, "J6 的灵敏度没有写清单位是 °/°，容易被当成 px/°"
    # 这里只核对**量级**（真值 1.0 °/°，允许 0.3～2.0）。
    # 原因是本测试为了跑得动用了步长 4 的离线分析：保持窗口只有 0.4 s 的一半，
    # 步长 4 时窗口里只剩 3 帧，单条试验的转角估计噪声很大。
    # **J6 转角的精确度**由 test_vision_paths.py 用步长 1 单独卡（±30%），
    # 这里只保证流程把两条视觉路径接对了、没有量纲错误。
    assert j6.px_per_deg is not None and 0.3 <= j6.px_per_deg <= 2.0, (
        f"J6 实测旋转灵敏度 {j6.px_per_deg} °/° 离开真值 1.0 °/° 太远"
    )

    # -- 理论范围检查（不是碰撞检查） ----------------------------------
    lines = session.formal_range_checks()
    assert any("J1" in line for line in lines)
    assert any("unknown" in line for line in lines), "范围检查没有写明碰撞状态仍是 unknown"

    # -- 正式实验：组 A 与组 B -----------------------------------------
    group_a = session.run_formal("A")
    assert group_a.segments, "组 A 一段都没采"
    formal_a = [t for t in session.trials if t["stage"] == "formal_a"]
    assert formal_a, "组 A 没有登记任何试验"
    assert all(t["joint"] in ("J1", "J6") for t in formal_a)

    group_b = session.run_formal("B")
    assert group_b.segments, "组 B 一段都没采"
    formal_b = [t for t in session.trials if t["stage"] == "formal_b"]
    assert formal_b, "组 B 没有登记任何试验"

    # -- 落盘完整性（需求六：每个时间戳目录里该有的东西） ---------------
    run = run_dir_of(session)
    for name in ("run_manifest.json", "config.json", "events.jsonl", "robot_states.csv"):
        assert (run / name).is_file(), f"运行目录里缺少 {name}"
    for name in (
        "trials.csv",
        "amplitudes.csv",
        "sensitivity.csv",
        "directions.csv",
        "pretest_report.json",
        "pretest_report.txt",
        "trial_plan.json",
        "trial_plan.csv",
        "analysis_notes.txt",
    ):
        assert (run / "analysis" / name).is_file(), f"分析目录里缺少 {name}"
    assert (run / "vision" / "metrics.csv").is_file(), "逐帧视觉指标没有落盘"

    states = read_csv_rows(run / "robot_states.csv")
    assert states, "robot_states.csv 一行都没有"
    assert {"host_ns", "command_q_J1_deg", "actual_q_J1_deg"} <= set(states[0])

    missing = 0
    for meta in segments_of(session):
        assert Path(meta["dir"]).is_dir()
        assert (Path(meta["dir"]) / str(meta["raw_file"])).is_file()
        assert (Path(meta["dir"]) / "frame_timestamps.csv").is_file()
        assert (Path(meta["dir"]) / "capture_summary.txt").is_file()
        assert (Path(meta["dir"]) / "missing_frames.csv").is_file()
        assert meta["stopped_early"] is False
        missing += int(meta["missing_frame_count"])
    assert missing == 0, f"干运行不该缺帧，实际缺了 {missing} 帧"

    # 分析结果里 J1/J6 走的是两条不同的视觉路径（需求四）。
    trials_csv = read_csv_rows(run / "analysis" / "trials.csv")
    j1_rows = [row for row in trials_csv if row["joint"] == "J1"]
    j6_rows = [row for row in trials_csv if row["joint"] == "J6"]
    assert j1_rows and j6_rows
    assert all(row["sensitivity_source"].startswith("px_per_deg") for row in j1_rows)
    assert all(row["sensitivity_source"].startswith("rotation_deg_per_deg") for row in j6_rows)

    session.close()


def test_second_run_never_overwrites_the_first(tmp_path: Path, small_config) -> None:
    """需求八.12：再跑一次是**新目录**，旧目录里的文件一个都不许动。"""
    first, _rec = open_session(small_config, recorder=Recorder(answer=True))
    first.capture_static(segment_id="static_base", duration_s=0.4)
    first.run_pretest()
    first.analyze_offline(stride=8)
    first_root = run_dir_of(first)
    first_files = {
        path.relative_to(first_root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in first_root.rglob("*")
        if path.is_file()
    }
    assert first_files, "第一次运行什么都没写"
    first.close()

    # 时间戳目录名精确到秒：等一秒，确保第二次是另一个目录。
    time.sleep(1.1)
    second, _rec2 = open_session(small_config, recorder=Recorder(answer=True))
    try:
        second.capture_static(segment_id="static_base", duration_s=0.4)
        second_root = run_dir_of(second)
        assert second_root != first_root
        assert second_root.is_dir() and first_root.is_dir()
        after = {
            path.relative_to(first_root): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in first_root.rglob("*")
            if path.is_file()
        }
        assert after == first_files, "第二次运行改动了第一次运行的文件"
        assert not list(second_root.glob("analysis/trials.csv")), "第二次不该动第一次的分析结果"
    finally:
        second.close()


def test_reusing_a_run_stamp_is_refused(tmp_path: Path, small_config) -> None:
    """同一秒内再启动一次：直接拒绝，绝不往已有目录里写。"""
    stamp = "20990101_000000"
    first = create_run_directory(small_config, run_kind="dry-run", stamp=stamp)
    assert first.root.is_dir()
    with pytest.raises(RecorderError) as info:
        create_run_directory(small_config, run_kind="dry-run", stamp=stamp)
    assert "覆盖" in str(info.value)


def test_analysis_needs_a_pretest_first(tmp_path: Path, small_config) -> None:
    """没跑过预实验就点分析：要明确报错，不能给一份空报告。"""
    session, _rec = open_session(small_config, recorder=Recorder(answer=True))
    try:
        with pytest.raises(ExperimentError) as info:
            session.analyze_offline(stride=8)
        assert "预实验" in str(info.value)
    finally:
        session.close()


def test_cli_dry_run_forces_dry_run_mode(tmp_path: Path) -> None:
    """命令行 `dry-run` 子命令把模式钉死在 dry_run，配置文件里写 hardware 也不行。"""
    from sj_pretest import cli

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.mode = "hardware"  # 故意写错：cli 必须把它改回 dry_run
    path = tmp_path / "config.json"
    config.save(path)

    # 规模压到最小：一个关节、一档幅度、不跑正式实验——这条测试要验证的是
    # "模式被钉死"，不是流程本身（流程由上面那条端到端测试负责）。
    code = cli.main(
        [
            "dry-run",
            "--config",
            str(path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--joints",
            "J1",
            "--amplitudes",
            "0.2",
            "--repeats",
            "1",
            "--stride",
            "8",
            "--group",
            "none",
        ]
    )
    assert code == 0
    saved = read_json(path)
    # 落盘的配置不动，但这一次运行用的是 dry_run——用运行清单作证。
    outputs = sorted((tmp_path / "outputs").glob("*/run_manifest.json"))
    assert outputs, "dry-run 子命令没有建运行目录"
    manifest = read_json(outputs[-1])
    assert manifest["mode"] == "dry_run"
    assert manifest["synthetic"] is True
    assert saved["mode"] == "hardware"


def test_missing_dependency_is_reported_in_chinese(monkeypatch, tmp_path: Path) -> None:
    """依赖缺失要在动手之前说清楚，而不是跑到一半才炸。"""
    import builtins

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    recorder = Recorder(answer=True)
    session = ExperimentSession(config, hooks=recorder.hooks())
    session.open(run_kind="dry-run")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("自测：假装没装 opencv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ExperimentError) as info:
            session.connect_devices()
        assert "依赖" in str(info.value)
        assert "install_deps" in str(info.value)
    finally:
        session.close()
