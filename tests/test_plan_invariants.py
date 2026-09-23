"""需求八.4/八.5/八.6：不累计、只动目标关节、正负方向与重复编号。

这三条是**发命令之前**就该成立的语义，所以先在计划层断言一遍
（:func:`verify_plan` 已经把"按事件应当到哪"逐条核对了），
再在真跑一遍的数据里断言一遍——计划对、实际没照做，同样是事故。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import JOINT_NAMES, AppConfig
from sj_pretest.joint_space import (
    ROLE_MOVE,
    ROLE_RETURN,
    STAGE_FORMAL_A,
    STAGE_PRETEST,
    build_formal_group_a,
    build_formal_group_b,
    build_pretest_plan,
    build_quick_probe_plan,
    joint_delta,
    verify_plan,
)

from conftest import build_config, read_csv_rows, rows_for_event, segments_of


# --------------------------------------------------------------------------
# 计划层
# --------------------------------------------------------------------------


@pytest.mark.parametrize("joints", [("J1",), ("J1", "J6"), tuple(JOINT_NAMES)])
def test_pretest_plan_passes_verify(tmp_path: Path, joints: tuple[str, ...]) -> None:
    """计划本身满足：编号唯一、方向标签对、只动目标关节、每次都从名义位姿出发。"""
    config = build_config(tmp_path, joints=joints, amplitudes=(0.01, 0.05, 0.2), repeats=2)
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    verify_plan(plan)  # 不通过就抛，不需要额外断言


def test_pretest_plan_covers_every_combination(tmp_path: Path) -> None:
    """每个关节 × 每档幅度 × 两个方向 × 每次重复，都要有统计试验。"""
    config = build_config(
        tmp_path, joints=("J1", "J6"), amplitudes=(0.01, 0.05, 0.2), repeats=2
    )
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    statistics = [step.event for step in plan.statistics_steps()]

    seen = {
        (event.joint, event.amplitude_deg, event.direction, event.repeat_index)
        for event in statistics
    }
    expected = {
        (joint, amplitude, direction, repeat)
        for joint in ("J1", "J6")
        for amplitude in (0.01, 0.05, 0.2)
        for direction in (1, -1)
        for repeat in (1, 2)
    }
    assert seen == expected
    # 6 关节 × 3 档 × 2 方向 × 2 次 = 72，正是需求三.2 的规模。
    assert len(statistics) == len(expected)


def test_amplitudes_never_accumulate_in_plan(tmp_path: Path) -> None:
    """需求八.4：三档幅度各自独立，绝不允许累计成 0.26°。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.01, 0.05, 0.2), repeats=1)
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    nominal = config.robot.nominal_joint_deg

    for step in plan.motion_steps():
        event = step.event
        if event.stage != STAGE_PRETEST:
            continue
        delta = joint_delta(nominal, step.target_joint_deg)[0]
        if event.role == ROLE_MOVE:
            # 每一次动作的**目标角度**都必须等于"名义 + 这一档的幅度"，
            # 而不是"上一档之后的位置再偏一点"。
            assert delta == pytest.approx(event.direction * event.amplitude_deg, abs=1e-9), (
                f"{event.event_id} 的目标角偏离名义 {delta}°，"
                f"而这一档应该只偏 {event.direction * event.amplitude_deg}°"
            )
        elif event.role == ROLE_RETURN:
            # 回程回到名义位姿（0 偏），这样下一档才是真的从头开始。
            assert delta == pytest.approx(0.0, abs=1e-9), f"{event.event_id} 没有回到名义位姿"


def test_every_statistics_move_is_preceded_by_return(tmp_path: Path) -> None:
    """每一次统计动作之前，上一步的落点都必须是名义位姿。

    做法是"沿着计划走一遍，记住上一步把关节放到了哪"，而不是要求前一步一定
    标着 return：计划开头本来就没有"回程"（因为按约定，实验开始时机器的实际
    角就等于名义角，回程是多余的）。真正的锚点是"上一个落点是不是名义位姿"。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.01, 0.05, 0.2), repeats=1)
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    nominal = config.robot.nominal_joint_deg

    # 计划开始时按约定已经在名义位姿上（真机上由到位过程保证，见按钮一）。
    last_target = tuple(nominal)
    checked = 0
    for step in plan.steps:
        event = step.event
        if event.stage == STAGE_PRETEST and event.role == ROLE_MOVE:
            delta = joint_delta(nominal, last_target)[0]
            assert delta == pytest.approx(0.0, abs=1e-9), (
                f"{event.event_id} 之前的落点偏离名义位姿 {delta}°——"
                "这一档是叠加在上一档之上的"
            )
            checked += 1
        if step.target_joint_deg is not None:
            last_target = step.target_joint_deg
    assert checked == 6, f"只检查到 {checked} 次统计动作，应该 3 档 × 2 方向"


def test_only_target_joint_changes(tmp_path: Path) -> None:
    """需求八.5：单关节微动，其余五个关节的目标角必须原地不动。"""
    config = build_config(tmp_path, joints=tuple(JOINT_NAMES), amplitudes=(0.2,), repeats=1)
    for plan in (
        build_pretest_plan(config, config.robot.nominal_joint_deg),
        build_quick_probe_plan(
            config.robot.nominal_joint_deg, ("J1", "J6"), probe_deg=0.05
        ),
        build_formal_group_a(config, config.robot.nominal_joint_deg, "J1", 0.2),
        build_formal_group_b(config, config.robot.nominal_joint_deg, "J6", 0.2),
    ):
        nominal = config.robot.nominal_joint_deg
        for step in plan.motion_steps():
            event = step.event
            if event.joint is None or event.expected_delta_deg == 0.0:
                continue
            delta = joint_delta(nominal, step.target_joint_deg)
            index = JOINT_NAMES.index(event.joint)
            for other in range(6):
                if other == index:
                    continue
                assert delta[other] == pytest.approx(0.0, abs=1e-9), (
                    f"{event.event_id}（{event.joint}）把 {JOINT_NAMES[other]} 也动了 "
                    f"{delta[other]}°"
                )


def test_event_ids_encode_direction_and_repeat(tmp_path: Path) -> None:
    """需求八.6：事件编号里的方向标记和重复序号必须与实际一致。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=3)
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    statistics = [step.event for step in plan.statistics_steps()]
    assert statistics, "没有统计试验"

    for event in statistics:
        tag = "d+1" if event.direction > 0 else "d-1"
        assert tag in event.event_id, f"{event.event_id} 的方向标记与 direction={event.direction} 不符"
        assert f"r{event.repeat_index:02d}" in event.event_id, (
            f"{event.event_id} 的重复序号与 repeat_index={event.repeat_index} 不符"
        )

    positive = [e for e in statistics if e.direction > 0]
    negative = [e for e in statistics if e.direction < 0]
    assert len(positive) == len(negative) == 3
    assert {e.repeat_index for e in positive} == {1, 2, 3}


def test_limit_margin_is_enforced_by_the_planner(tmp_path: Path) -> None:
    """关节限位余量由计划层把关：把名义位姿推到接近限位就该报错，而不是照发。"""
    from sj_pretest.joint_space import PlanError

    config = build_config(tmp_path, joints=("J1",), amplitudes=(5.0,), repeats=1)
    # UR10 CB3 的 J1 限位是 ±360°；把名义角放到 -358°，再加 5° 就越界了。
    config.robot.nominal_joint_deg = [-358.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    with pytest.raises(PlanError):
        build_pretest_plan(config, config.robot.nominal_joint_deg)


# --------------------------------------------------------------------------
# 到位过程的说明文字
# --------------------------------------------------------------------------


def test_approach_step_label_is_readable_when_nothing_needs_to_move() -> None:
    """已经在实验姿态上时，每一步的说明不能留下半截话。

    干运行每次都从名义位姿出发，"相对当前位姿"后面一个关节都列不出来；
    真机上偶尔也会碰上（比如重启后机器人还停在原位）。这时候写出来的
    应当是"不需要移动"，而不是一句以"相对当前位姿 "结尾、后面空着的
    说明——现场人对着一句读不通的话，没法判断这一步到底要动什么。
    """
    from sj_pretest.joint_space import build_approach_plan

    nominal = [-53.8365, -175.6992, 148.2887, -137.4699, -17.5120, -112.4909]

    still = build_approach_plan(nominal, nominal)
    assert len(still.steps) == 6
    for step in still.steps:
        assert not step.label.rstrip().endswith("相对当前位姿")
        assert "不需要移动" in step.label, step.label
        assert step.target_joint_deg is not None

    # 真要移动时，说明里必须逐关节列出"相对当前位姿"的差值。
    moved = build_approach_plan([0.0] * 6, nominal)
    first = moved.steps[0].label
    assert "相对当前位姿" in first
    assert "J1" in first
    assert "不需要移动" not in first
    # 最后一个点恰好落在实验姿态上。
    assert moved.steps[-1].target_joint_deg == pytest.approx(nominal)


# --------------------------------------------------------------------------
# 实际跑出来的数据
# --------------------------------------------------------------------------


def test_pretest_capture_starts_every_move_from_nominal(session_factory) -> None:
    """一边跑一边记账：每次统计动作开始时的**实际**指令角都对着名义位姿。

    这是"不累计"在真实数据里的样子：计划说从头出发，数据里也得是从头出发。
    """
    session, recorder, config = session_factory(
        joints=("J1",), amplitudes=(0.01, 0.05, 0.2), repeats=1
    )
    session.capture_static(segment_id="static_test", duration_s=0.3)
    session.run_pretest()

    nominal = config.robot.nominal_joint_deg
    assert session.run is not None
    rows = read_csv_rows(session.run.root / "robot_states.csv")
    assert rows, "robot_states.csv 一行都没有"

    checked = 0
    for trial in session.trials:
        event_id = str(trial["event_id"])
        rows_of_event = rows_for_event(rows, event_id)
        assert rows_of_event, f"{event_id} 在 robot_states.csv 里没有任何记录"
        first = rows_of_event[0]
        commanded = [float(first[f"command_q_{name}_deg"]) for name in JOINT_NAMES]
        delta = joint_delta(nominal, commanded)
        expected = float(trial["commanded_delta_deg"])
        assert delta[0] == pytest.approx(expected, abs=1e-6), (
            f"{event_id}：动作开始时 J1 指令角比名义位姿偏 {delta[0]}°，"
            f"而这一次只应该偏 {expected}°（说明动作是累计出来的）"
        )
        for other in range(1, 6):
            assert delta[other] == pytest.approx(0.0, abs=1e-6), (
                f"{event_id} 顺带动了 {JOINT_NAMES[other]}"
            )
        checked += 1
    # J1 单关节、三档幅度、两个方向、每方向 1 次重复 = 6 次统计试验。
    assert checked == len(session.trials) == 6, "统计试验的条数和实际记下的对不上"


def test_segments_match_the_plan(session_factory) -> None:
    """每段采集都落成一个目录，且段里的相位标签齐全（运动在保持之前）。"""
    session, _recorder, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    session.capture_static(segment_id="static_test", duration_s=0.3)
    session.run_pretest()

    assert session.run is not None
    metas = segments_of(session)
    assert len(metas) == 1 + len(session.trials)

    for meta in metas:
        assert Path(meta["dir"]).is_dir()
        phases = [entry["label"] for entry in meta["phases"]]
        if meta["segment_kind"] == "static":
            assert phases == ["static"]
            continue
        assert phases == ["pre_motion", "move", "hold", "return", "post_motion"], (
            f"{meta['segment_id']} 的相位是 {phases}"
        )
        timing = {entry["label"]: entry for entry in meta["phases"]}
        # 相位必须首尾相接、单调不减：有一处缝隙就说明窗口切错了。
        ordered = sorted(timing.values(), key=lambda entry: entry["start_s"])
        previous_end = 0.0
        for entry in ordered:
            assert entry["start_s"] >= previous_end - 1e-9
            assert entry["end_s"] >= entry["start_s"]
            previous_end = entry["end_s"]
