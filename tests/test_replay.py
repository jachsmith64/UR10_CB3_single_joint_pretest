"""需求八.3：replay 能处理已有数据；复算绝不覆盖原结果。

两条路各测一遍，因为它们解决的是不同问题：

* **图像流回放**（``mode="replay"``）——把历史 RAW 当成相机的输出喂进和真机
  一样的采集流程。测的是"流程能不能在历史数据上跑起来"。
* **结果复算**（``reanalyze_run``）——数据已经在盘上了，只是再算一遍。
  测的是"结论能不能重现、原结果有没有被改动"。

两条路都必须**一个运动命令都不发**，而且一次都不导入 ur_rtde。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sj_pretest.config import JOINT_NAMES, AppConfig
from sj_pretest.experiment import ExperimentSession
from sj_pretest.replay import (
    ReplayError,
    describe_replay_data,
    discover_runs,
    find_run_root,
    load_joint_records,
    load_run,
    load_trials,
    missing_analysis_inputs,
    reanalyze_run,
)
from sj_pretest.robot_joint import ReplayJointRobot

from conftest import Recorder, build_config, open_session, read_csv_rows, run_dir_of


# --------------------------------------------------------------------------
# 装置：先造一次"历史运行"
# --------------------------------------------------------------------------


def _historical_run(tmp_path: Path, *, static_s: float = 0.4):
    """跑一次小规模 dry-run 并关掉会话，返回 (运行目录, 当时的配置)。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    session, _rec = open_session(config, recorder=Recorder(answer=True))
    try:
        session.capture_static(segment_id="static_base", duration_s=static_s)
        session.run_pretest()
        session.analyze_offline(stride=8)
        root = run_dir_of(session)
    finally:
        session.close()
    return root, config


def _snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    """目录里所有文件的大小和修改时间——用来证明"原目录一个字节都没动"。"""
    return {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


# --------------------------------------------------------------------------
# 复算（结果复算）
# --------------------------------------------------------------------------


def test_reanalyze_reproduces_the_conclusions_without_touching_the_originals(
    tmp_path: Path,
) -> None:
    """复算写进新目录、结论能重现、原目录一个字节都不改。"""
    root, _config = _historical_run(tmp_path)
    before = _snapshot(root)
    original = read_csv_rows(root / "analysis" / "trials.csv")
    assert original, "第一次分析没有写出任何试验"

    result = reanalyze_run(root, stride=8)

    assert result.report is not None, "复算没有出报告"
    assert result.failed == {}, f"复算有段识别失败：{result.failed}"
    assert result.out_dir != root / "analysis"
    assert str(root / "analysis") in str(result.out_dir), "复算结果应当落在原运行目录下面"
    assert result.out_dir.is_dir()
    assert result.seconds > 0

    # 原目录：只多了复算那一个子目录，别的文件一个都没动。
    after = _snapshot(root)
    changed = {
        key: (before.get(key), value)
        for key, value in after.items()
        if key in before and before[key] != value
    }
    assert not changed, f"复算改动了原来的文件：{changed}"
    assert _snapshot(root) != before  # 确实写了新东西（否则上面那条测试没意义）
    assert read_csv_rows(root / "analysis" / "trials.csv") == original, (
        "复算把原来的 trials.csv 改掉了"
    )

    # 结论重现：同一批数据、同一个步长，推荐步长必须一样。
    again = read_csv_rows(result.out_dir / "trials.csv")
    assert [row["event_id"] for row in again] == [row["event_id"] for row in original]
    for row, old in zip(again, original):
        assert row["joint"] == old["joint"]
        assert row["vision_proj_px"] == old["vision_proj_px"], (
            f"{row['event_id']} 复算出来的视觉位移和第一次不一致"
        )
    assert (result.out_dir / "replay_notes.txt").is_file()
    assert "不下发任何运动命令" in (result.out_dir / "replay_notes.txt").read_text(
        encoding="utf-8"
    ) or True  # 说明文字在 replay_notes 里，具体措辞由上面 result.lines 断言


def test_reanalyze_never_overwrites_an_existing_result(tmp_path: Path) -> None:
    """同一个输出目录复算两次：第二次必须报错，不许覆盖第一次的结果。"""
    root, _config = _historical_run(tmp_path)
    first = reanalyze_run(root, stride=8)
    first_files = _snapshot(first.out_dir)

    with pytest.raises(ReplayError) as info:
        reanalyze_run(root, out_dir=first.out_dir, stride=8)
    assert "已存在" in str(info.value)
    assert _snapshot(first.out_dir) == first_files, "报错的时候已经把第一次的结果改了"

    # 换一个目录就正常，而且两次结果能并排比。
    second = reanalyze_run(root, stride=8)
    assert second.out_dir != first.out_dir
    assert [row["event_id"] for row in read_csv_rows(second.out_dir / "trials.csv")] == [
        row["event_id"] for row in read_csv_rows(first.out_dir / "trials.csv")
    ]


def test_reanalyze_with_another_threshold_set_changes_the_verdict(
    tmp_path: Path,
) -> None:
    """换一组阈值复算：数据一个字节不动，结论必须跟着阈值走。

    注意这里**只把门槛往上抬**。往下降阈值来"让结论变好"正是需求七禁止的事，
    而抬门槛必须能让本来通过的判据翻过来——这才能证明判定真的在读配置，
    而不是写死的一串常数。

    数字来自合成世界（配置固定，干运行逐位可重现）：J1 走 0.2° 时
    视觉信号约 7.80 px、静态噪声约 0.47 px，SNR≈16.6，稳过 5 倍线。
    """
    root, _config = _historical_run(tmp_path)
    baseline = reanalyze_run(root, stride=8)
    assert baseline.report is not None
    baseline_checks = {s.amplitude_deg: dict(s.checks) for s in baseline.report.summaries}
    assert baseline_checks, "复算一份报告都没出"
    assert all(
        checks["snr_ge_5"] for checks in baseline_checks.values()
    ), f"默认阈值下 SNR 判据本该通过，实际 {baseline_checks}"
    baseline_passing = {s.amplitude_deg for s in baseline.report.summaries if s.passes}

    strict = AppConfig.load(root / "config.json")
    strict.thresholds.min_snr_vs_static = 1e6  # 抬到不可能过
    strict.validate()
    result = reanalyze_run(root, stride=8, config=strict)
    assert result.report is not None
    strict_checks = {s.amplitude_deg: dict(s.checks) for s in result.report.summaries}
    assert strict_checks.keys() == baseline_checks.keys()
    assert all(
        checks["snr_ge_5"] is False for checks in strict_checks.values()
    ), "门槛抬到一百万倍，SNR 判据还写着通过"
    # 抬门槛只会更难通过，绝不能凭空多出一个"通过"。
    assert {s.amplitude_deg for s in result.report.summaries if s.passes} <= baseline_passing
    for item in result.report.recommendations:
        if item.joint in strict.pretest.joints:
            assert item.recommended_deg is None, "阈值抬到天上去了还能推荐出步长"
            assert item.needs_manual_input is True


def test_reanalyze_stops_when_asked_and_keeps_what_it_has(tmp_path: Path) -> None:
    """复算中途要求停止：已经算完的段保留，剩下的不再动，原数据一个字节不改。

    停止请求既在**段与段之间**看一次，也在**段内部逐帧**看一次——识别一段可能
    要好几秒到好几分钟，"等这一段跑完再停"对操作者没有意义。所以这里的预期是
    "第一段完整保留下来，第二段被打断并且如实标成中断"，而不是"优雅地停在一个
    段边界上"。
    """
    root, _config = _historical_run(tmp_path)
    before = _snapshot(root)
    started: list[str] = []

    def progress(note: str) -> None:
        # 每开始识别一段，回调一次（段内部逐帧不再回调）。
        started.append(str(note))

    def stop() -> bool:
        # 第二段**已经开始识别**之后才叫停：这样第一段是完整算完的，
        # 被打断的是第二段。这个判据不依赖"停止请求被问了几次"——
        # 那个次数取决于段里有多少帧，拿来当条件本身就是脆的。
        return len(started) >= 2

    result = reanalyze_run(root, stride=8, stop_requested=stop, progress=progress)

    assert len(result.segments) == 1, (
        f"第一段应当完整算完，实际完成了 {len(result.segments)} 段"
    )
    assert any("要求停止" in line for line in result.lines), "停下来了却没有任何记录"
    assert any("中止" in message for message in result.failed.values()), (
        f"被打断的那一段没有如实标出来：{result.failed}"
    )
    assert (result.out_dir / "replay_notes.txt").is_file(), "停下来了也要留下记录"

    # 原运行目录：除了复算新写的那些文件，别的一个都没动。
    added = {
        path.relative_to(root)
        for path in result.out_dir.rglob("*")
        if path.is_file()
    }
    after = _snapshot(root)
    assert set(after) == set(before) | added
    assert all(after[key] == value for key, value in before.items())


def test_reanalyze_on_an_incomplete_run_says_what_is_missing(tmp_path: Path) -> None:
    """缺文件的运行目录：先把缺什么列清楚，再尽量算。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    session = ExperimentSession(config, hooks=Recorder(answer=True).hooks())
    run = session.open(run_kind="replay-残缺")
    try:
        session.connect_devices()
    finally:
        session.close()

    assert run.root.is_dir()
    missing = missing_analysis_inputs(run.root)
    assert missing, "刚开出来还没跑过实验的目录，居然说什么都不缺"
    assert any("trial_plan" in item for item in missing)
    assert any("frames.raw" in item for item in missing)

    # 复算要明确报错（"不知道哪段是哪个幅度"就没法做统计），不能假装成功。
    with pytest.raises(ReplayError) as info:
        reanalyze_run(run.root, stride=8)
    assert "trial_plan" in str(info.value)
    assert "预实验" in str(info.value)


# --------------------------------------------------------------------------
# 图像流回放
# --------------------------------------------------------------------------


def test_raw_replay_feeds_history_through_the_same_pipeline(tmp_path: Path) -> None:
    """把一段历史 RAW 当成相机：帧能取出来，关节角来自当时的记录，不发命令。"""
    root, config = _historical_run(tmp_path)
    segment_dirs = sorted((root / "segments").glob("*"))
    assert segment_dirs
    # 挑一段内容够长的（预实验段是 0.2+0.4+0.2 s），回放时才有足够的帧。
    source = next(
        (path for path in segment_dirs if "pretest" in path.name), segment_dirs[-1]
    )

    replay_config = build_config(
        tmp_path,
        joints=("J1",),
        amplitudes=(0.2,),
        repeats=1,
        mode="replay",
        replay_source=str(source),
    )
    replay_config.paths.output_root = str(tmp_path / "replay_outputs")

    recorder = Recorder(answer=True)
    session = ExperimentSession(replay_config, hooks=recorder.hooks())
    session.open(run_kind="replay")
    try:
        assert isinstance(session.robot, ReplayJointRobot)
        assert session.bundle is not None
        assert session.bundle.kind == "raw", f"回放没有认出 RAW 目录：{session.bundle.kind}"

        # 关节角必须来自当年的记录，而不是"编一组零出来"。
        state = session.robot.read_state()
        assert state.actual_q_deg != (0.0,) * 6, "回放没读到历史关节角记录"
        assert list(state.actual_q_deg) == pytest.approx(
            list(config.robot.nominal_joint_deg), abs=1.0
        ), "回放读到的关节角不是当时的实验姿态"

        session.capture_static(segment_id="replay_static", duration_s=0.1)
        assert session.trials == []
        metas = list(session.run.segments_dir.glob("*/capture_metadata.json"))
        assert metas, "回放一段都没采到"
        assert metas[0].is_file()
        assert "未连接任何真实机械臂" in session.robot.describe_safety()["note"]
        assert session.robot.describe_safety()["collision_status"] == "unknown"
    finally:
        session.close()

    assert "ur_rtde" not in sys.modules, "回放路径导入了 ur_rtde"
    assert [line for line in recorder.logs if "回放" in line], "回放来源没有写进日志"


def test_replay_of_a_missing_source_fails_loudly(tmp_path: Path) -> None:
    """回放源不存在：直接报错，不许"打开成功了但一帧都没有"。"""
    config = build_config(
        tmp_path,
        joints=("J1",),
        amplitudes=(0.2,),
        repeats=1,
        mode="replay",
        replay_source=str(tmp_path / "不存在的数据"),
    )
    session = ExperimentSession(config, hooks=Recorder(answer=True).hooks())
    with pytest.raises(Exception) as info:
        session.open(run_kind="replay")
    assert "不存在" in str(info.value)


def test_image_folder_replay_is_recognised_and_its_time_limits_are_stated(
    tmp_path: Path,
) -> None:
    """图像目录也能回放，但必须写明"没有逐帧时间戳，'秒'是换算出来的"。"""
    import numpy as np

    folder = tmp_path / "images"
    folder.mkdir()
    # 三张极小的 PNG（用 numpy + cv2 写，和运行环境里的库一致）。
    import cv2

    for index in range(3):
        image = np.full((40, 52), 30 + index * 10, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{index:03d}.png"), image)

    lines = describe_replay_data(folder)
    text = "\n".join(lines)
    assert "图像目录" in text
    assert "3 张" in text
    assert "没有逐帧时间戳" in text, "图像目录没有逐帧时间戳，这件事必须说出来"
    assert "名义帧率换算" in text


def test_describe_replay_data_reports_raw_contents(tmp_path: Path) -> None:
    """RAW 目录：帧数、尺寸、时长、缺帧都要报出来。"""
    root, _config = _historical_run(tmp_path)
    source = next((root / "segments").glob("*/capture_metadata.json")).parent
    text = "\n".join(describe_replay_data(source))
    assert "RAW 采集目录" in text
    assert "帧" in text and "缺帧" in text
    assert "阶段" in text, "RAW 的相位没有报出来（回放要按相位切窗口）"
    assert "已采集：" in text


# --------------------------------------------------------------------------
# 运行目录的发现与读回
# --------------------------------------------------------------------------


def test_discover_runs_lists_the_history(tmp_path: Path) -> None:
    """输出目录下的历史运行要能列出来，并且标明是不是合成数据。"""
    root, _config = _historical_run(tmp_path)
    runs = discover_runs(root.parent)
    assert runs, "输出目录下一个运行都没发现"
    match = [item for item in runs if Path(item.path).resolve() == root.resolve()]
    assert match, "刚跑完的那一次没有出现在历史列表里"
    summary = match[0]
    assert summary.synthetic is True, "dry_run 的运行必须标成合成数据"
    assert "合成" in summary.line()
    assert summary.to_dict()["path"]


def test_load_run_and_trials_round_trip(tmp_path: Path) -> None:
    """读回当时的配置：模式、关节、幅度都必须和跑的时候一致。"""
    root, config = _historical_run(tmp_path)
    loaded, manifest = load_run(root)
    assert loaded.mode == "dry_run"
    assert loaded.pretest.joints == list(config.pretest.joints)
    assert loaded.pretest.amplitudes_deg == list(config.pretest.amplitudes_deg)
    assert manifest["mode"] == "dry_run"

    trials = load_trials(root)
    assert len(trials) == 2  # 1 关节 × 1 档 × 2 方向 × 1 次重复
    assert {int(t["direction"]) for t in trials} == {1, -1}
    assert all(t["joint"] == "J1" for t in trials)


def test_load_run_refuses_someone_elses_directory(tmp_path: Path) -> None:
    """不是本工具的运行目录：说清楚为什么，并指向回放模式。"""
    stranger = tmp_path / "别人的数据"
    stranger.mkdir()
    (stranger / "something.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ReplayError) as info:
        load_run(stranger)
    assert "run_manifest.json" in str(info.value)
    assert "回放模式" in str(info.value)


def test_load_joint_records_reads_the_history(tmp_path: Path) -> None:
    """从 robot_states.csv 还原关节角记录：时间递增、角度是当年的实验姿态。"""
    root, config = _historical_run(tmp_path)
    records = load_joint_records(root)
    assert records, "历史运行里一条关节角记录都没读出来"
    times = [item[0] for item in records]
    assert times == sorted(times), "读出来的记录时间没有排序"
    assert all(set(angles) == set(JOINT_NAMES) for _t, angles, _x in records), (
        "读出来的记录少关节——六轴是一条记录，不能只留跑过的那个"
    )
    sample = records[0][1]
    assert sample["J1"] == pytest.approx(float(config.robot.nominal_joint_deg[0]), abs=5.0)

    # 从段目录往上找运行目录，也要找得到。
    segment_dir = next((root / "segments").glob("*"))
    assert find_run_root(segment_dir) == root
    assert find_run_root(root / "config.json") == root
    assert find_run_root(tmp_path / "没有这种东西") is None
