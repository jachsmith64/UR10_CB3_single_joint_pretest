"""需求八的安全条款：不确认就不动、中止后不再发命令、不自动回位、干运行不碰真机。

这些是**红线**，所以每一条都断言两件事：
一件是"该发生的发生了"（抛了中止、记了事件），
另一件是"不该发生的没发生"（合成世界里没有任何一次动作记录）。

只看前者不够——"抛了异常但其实已经把命令发下去了"正是这类代码最容易出的错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sj_pretest.config import JOINT_NAMES, AppConfig
from sj_pretest.experiment import ExperimentSession, ExperimentError
from sj_pretest.robot_joint import MotionAborted, SimulatedJointRobot

from conftest import Recorder, build_config, open_session, read_json, run_dir_of


def _events(session: ExperimentSession) -> list[dict]:
    assert session.run is not None
    return session.run.events.read_all()


def _moved(session: ExperimentSession) -> list[str]:
    robot = session.robot
    assert robot is not None
    return list(robot.moved_event_ids())


# --------------------------------------------------------------------------
# 人工确认
# --------------------------------------------------------------------------


def test_approach_declined_at_the_first_step_sends_nothing(session_factory) -> None:
    """到位过程每一步都要确认：人不点头，一步都不许动。

    同时核对"运动前显示目标角度"这条：确认问句之前，日志里必须已经有
    这一步相对**当前**姿态的增量。
    """
    recorder = Recorder(answer=False)
    session, _rec, config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1, recorder=recorder
    )
    assert isinstance(session.robot, SimulatedJointRobot)

    with pytest.raises(MotionAborted):
        session.run_approach()

    assert _moved(session) == [], "人没有确认，却已经发过动作了"
    assert session.aborted is False, "拒绝一步不等于中止整个会话"

    # 目标角度先于确认显示出来（需求三.1）。
    text = recorder.text()
    assert "本次增量（相对当前姿态）" in text, "运动前没有显示相对当前姿态的增量"
    assert "到位 1/" in text, "没有说明这是到位过程的第几步"
    # 拒绝要留下痕迹，不能只是"悄悄没动"。
    declined = [e for e in _events(session) if e["event"] == "motion_declined"]
    assert declined, "拒绝了运动，事件日志里却没有 motion_declined"
    assert any(JOINT_NAMES[0] in label for label in recorder.asked)


def test_pretest_declined_at_the_first_step_sends_nothing(session_factory) -> None:
    """关掉"按关节确认"时，每一步都会问：拒绝同样一步都不许动。"""
    recorder = Recorder(answer=False)
    session, _rec, _config = session_factory(
        joints=("J1",),
        amplitudes=(0.2,),
        repeats=1,
        recorder=recorder,
        pretest__confirm_each_joint=False,
    )

    with pytest.raises(MotionAborted):
        session.run_pretest()

    assert _moved(session) == [], "人没有确认，却已经发过动作了"
    assert [e for e in _events(session) if e["event"] == "motion_declined"]


def test_confirmation_label_names_the_joint_and_the_unknown_collision(
    session_factory,
) -> None:
    """关节级确认要问得清楚：是哪个关节、碰撞状态是什么。"""
    recorder = Recorder(answer=True)
    session, _rec, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1, recorder=recorder
    )
    session.run_pretest()
    trigger = [label for label in recorder.asked if "接下来是这个关节" in label]
    assert trigger, "没有在关节开始前停下来确认"
    assert "J1" in trigger[0]
    assert "unknown" in trigger[0], "确认问句里必须写明碰撞状态是 unknown"


# --------------------------------------------------------------------------
# 中止
# --------------------------------------------------------------------------


def test_stop_mid_trial_skips_the_return_and_keeps_the_data(session_factory) -> None:
    """中止发生在一次动作已经在飞的时候：

    * 剩下的动作（包括这一次的回程）一律不发；
    * 机器人停在当前姿态，**不自动回位**；
    * 已经采到的段照常落盘，并且登记进 trial_plan（数据保留、能用）。
    """
    recorder = Recorder(answer=True)
    recorder.arm_stop_before_move(after=1)
    session, _rec, config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1, recorder=recorder
    )

    with pytest.raises(MotionAborted):
        session.run_pretest()

    assert session.aborted is True, "按了中止，会话却没有进入中止状态"
    robot = session.robot
    assert robot is not None
    assert robot.abort_reason() is not None, "机器人层不知道已经中止"

    moved = _moved(session)
    assert moved, "这一次动作本来应该发出去（中止是在它之后按下的）"
    assert not any(event_id.endswith("return-back") for event_id in moved), (
        f"中止之后仍然发了回程动作：{moved}"
    )

    # 不自动回位：目标角停在偏出去的位置上，没有被悄悄改回名义位姿。
    target = robot.world.target  # type: ignore[attr-defined]
    nominal = config.robot.nominal_joint_deg
    index = JOINT_NAMES.index("J1")
    assert target[index] == pytest.approx(nominal[index] + 0.2, abs=1e-6), (
        "中止后目标角被改动了——不许自动回位"
    )

    # 数据保留：段目录在盘上，trial_plan 里也有它（否则分析侧用不上）。
    assert session.trials, "已经采到的试验没有登记"
    plan = read_json(run_dir_of(session) / "analysis" / "trial_plan.json")
    assert [entry["segment_id"] for entry in plan] == [
        str(trial["segment_id"]) for trial in session.trials
    ]
    for entry in plan:
        assert session.run is not None
        segment_dir = session.run.segment_dir(str(entry["segment_id"]))
        assert segment_dir.is_dir(), f"{entry['segment_id']} 的采集目录不见了"
        metadata = read_json(segment_dir / "capture_metadata.json")
        raw = segment_dir / str(metadata["raw_file"])
        assert raw.is_file() and raw.stat().st_size > 0, (
            f"中止后 {entry['segment_id']} 的 RAW 不见了或长度为 0"
        )
        assert (segment_dir / "frame_timestamps.csv").is_file()
    # 被跳过的那一段不该有目录（说明它连一帧都没采）。
    captured = [e for e in _events(session) if e["event"] == "segment_captured"]
    assert len(captured) == len(session.trials)


def test_no_commands_after_abort(session_factory) -> None:
    """中止之后：再点一次也不会发命令，而且不会再抛"未知错误"。"""
    recorder = Recorder(answer=True)
    recorder.arm_stop_before_move(after=1)
    session, _rec, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1, recorder=recorder
    )
    with pytest.raises(MotionAborted):
        session.run_pretest()

    before = _moved(session)
    with pytest.raises(MotionAborted):
        session.run_pretest()
    assert _moved(session) == before, "中止之后又发了动作"

    # 到位过程同样不许再动。
    recorder.stop = True
    with pytest.raises(MotionAborted):
        session.run_approach()
    assert _moved(session) == before

    # 分析这类**只读**操作不受影响：中止不该把已采到的数据也锁死。
    report = session.analyze_offline(stride=8)
    assert report.trials, "中止之后已经采到的数据应该仍然可以分析"


# --------------------------------------------------------------------------
# 干运行绝不碰真机
# --------------------------------------------------------------------------


def test_dry_run_never_imports_or_touches_rtde(session_factory) -> None:
    """干运行模式下：机器人端口是合成世界，ur_rtde 一次都不许被导入。"""
    session, _recorder, config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    assert config.mode == "dry_run"
    assert isinstance(session.robot, SimulatedJointRobot)
    assert not isinstance(session.robot, type(None))
    assert "ur_rtde" not in sys.modules, "干运行下导入了 ur_rtde"
    safety = session.robot.describe_safety()
    assert safety["collision_status"] == "unknown"
    assert "未连接任何真实机械臂" in safety["note"]
    assert safety["mode"] == "dry_run"


def test_hardware_mode_is_never_the_default() -> None:
    """真机必须**显式**选：默认配置一定是干运行。"""
    assert AppConfig().mode == "dry_run"
    assert AppConfig().validate() is None


def test_low_disk_blocks_the_run_before_any_motion(tmp_path: Path) -> None:
    """磁盘不够直接拒绝开始，不给"先跑起来再说"的机会。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.paths.min_free_disk_gb = 1e9  # 一亿 GB：任何真实磁盘都不够
    recorder = Recorder(answer=True)
    session = ExperimentSession(config, hooks=recorder.hooks())
    with pytest.raises(ExperimentError) as info:
        session.open(run_kind="test")
    assert "磁盘" in str(info.value)
    assert recorder.asked == [], "磁盘检查没过，却已经问过确认了"
    assert session.robot is None, "磁盘检查没过却已经把设备建起来了"


def test_min_free_disk_is_validated(tmp_path: Path) -> None:
    """这条阈值本身也要能被配置校验拦住负数。"""
    from sj_pretest.config import ConfigError

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.paths.min_free_disk_gb = -1.0
    with pytest.raises(ConfigError):
        config.validate()


def test_plan_disk_gate_refuses_a_stage_that_cannot_fit(tmp_path: Path) -> None:
    """按**计划**估算占用：写不下的那一段拒绝开始，而不是跑一半写满。

    这里故意把配置调成"真要写几十 TB"（满幅 + 50 级 × 50 遍 × 6 关节），
    任何真实磁盘都不够，所以判据不依赖跑测试这台机器的剩余空间。
    """
    from sj_pretest.config import check_plan_disk
    from sj_pretest.joint_space import planned_plans

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.mode = "hardware"           # 不用干运行的缩短时长
    config.camera.roi = None           # 满幅 1936×1096 ≈ 280 MB/s
    config.formal.staircase_n = 50
    config.formal.repeats = 50
    config.formal.step_deg = {"J1": 0.2}
    config.validate()

    ok, lines = check_plan_disk(config, planned_plans(config))
    text = "\n".join(lines)
    assert ok is False, f"几十 TB 的计划却说磁盘够：{text}"
    assert "预计" in text, "没给出任何估算"
    # 拒绝的时候必须说清"接下来能做什么"，否则现场只能干瞪眼。
    assert "roi" in text.lower() or "ROI" in text
    assert "save_raw" in text


def test_plan_disk_gate_is_proportional_not_a_blanket_refusal(tmp_path: Path) -> None:
    """同一套代码在小规模下必须放行——门槛是按计划算的，不是"一律拒绝"。"""
    from sj_pretest.config import check_plan_disk
    from sj_pretest.joint_space import planned_plans

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    ok, lines = check_plan_disk(config, planned_plans(config, include_formal=False))
    assert ok is True, "\n".join(lines)
    assert any("磁盘检查" in line for line in lines)


def test_disk_gate_stops_the_stage_before_any_motion(tmp_path: Path) -> None:
    """红线式断言：磁盘不够时，这一段**一个运动命令都没发**。

    只看"抛了异常"不够——真正要命的是"抛异常之前已经把命令发下去了"。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    session, recorder = open_session(config, recorder=Recorder(answer=True))
    try:
        # 会话已经打开（连设备那一步的固定门槛过了），现在把"按计划估算"这一关
        # 卡死：预实验那一段无论怎么算都不够。
        session.config.paths.min_free_disk_gb = 1e9
        with pytest.raises(ExperimentError) as info:
            session.run_pretest()
        assert "磁盘" in str(info.value)
        assert _moved(session) == [], "磁盘检查没过，却已经发了运动命令"
        assert session.trials == [], "磁盘检查没过，却已经登记了试验"
        # 拒绝这件事本身要落在事件流里，事后能查。
        assert any(
            event["event"] == "disk_checked" and event.get("ok") is False
            for event in _events(session)
        ), "拒绝了却没有留下 disk_checked 记录"
    finally:
        session.close()


# --------------------------------------------------------------------------
# 掉帧/断连如实记录（需求八.8）
# --------------------------------------------------------------------------


def test_dropped_frames_are_recorded_and_reported(tmp_path: Path) -> None:
    """注入丢帧：采集里要看得见缺口，日志里要提示，不能悄悄少几帧。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.dry_run.drop_every = 7  # 每 7 帧丢 1 帧
    recorder = Recorder(answer=True)
    session, recorder = open_session(config, recorder=recorder)
    try:
        session.capture_static(segment_id="static_drop", duration_s=0.4)
        session.run_pretest()
        assert session.run is not None
        rows = [
            line
            for line in (session.run.root / "robot_states.csv")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
            if line.strip()
        ]
        assert rows, "掉帧注入下 robot_states.csv 一行都没有"
        metas = list(session.run.segments_dir.glob("*/capture_metadata.json"))
        assert metas, "一段采集都没有落盘"
        drop_reported = False
        for path in metas:
            payload = read_json(path)
            if int(payload["missing_frame_count"]) > 0:
                drop_reported = True
                # 缺口要写进缺帧清单，事后能对上是哪几帧丢了。
                missing = path.parent / "missing_frames.csv"
                assert missing.is_file()
                assert missing.read_text(encoding="utf-8").strip()
        assert drop_reported, "注入了丢帧，采集元数据里却一个缺口都没有"
    finally:
        session.close()
