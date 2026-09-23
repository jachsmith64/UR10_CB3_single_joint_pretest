"""★ 需求一·5：时长与磁盘估算必须按**相邻两次目标姿态的差**算，不是按"离名义多远"。

这条为什么必须单独测
--------------------
v1.0.2 拿 ``event.expected_delta_deg``（相对**名义位姿**的偏移量）当"这次要走多少度"，
于是两个方向各错一次，而且正好是相反的方向：

* **组 A 的相邻阶梯（高估）**：第 3 级的目标是"名义 + 3Δ"，但机械臂是从
  "名义 + 2Δ"走过去的，真实运动量只有 **Δ**。按 3Δ 算，每一级的时长和磁盘
  都被放大到几倍；5 级阶梯一遍要按 1Δ+2Δ+3Δ+4Δ+5Δ 来估，而真实是 5×Δ。
* **组 B 的回程（低估）**：回程目标是名义位姿（``expected_delta_deg == 0.0``），
  机械臂其实是从"名义 ± Δ"走回名义，真实运动量是 **Δ**。按 0 算，
  回程时间被当成 0——而回程和去程一样一直在录 RAW。

两个错都在同一份"上一次目标关节角"的时间序里自然消掉。这里就按需求一·5 的原话
把三档幅度（0.01° / 0.05° / 0.2°）和两个组都过一遍，而且**手工把每一步的
目标角列出来、只看相邻差**，再和工具算出来的总时长对账。

为什么要三档都测：0.01° 小到速度规划处在**三角形**区间（加速到一半就得减速），
0.2° 已经在梯形区间，两种形状的时长公式不同；只测一档会把其中一种漏掉。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import (
    AppConfig,
    ROLE_RETURN,
    ROLE_WAIT,
    estimate_capture_seconds,
    motion_amount_deg,
    segment_step_groups,
    trapezoid_seconds,
)
from sj_pretest.joint_space import (
    STAGE_FORMAL_A,
    MotionEvent,
    MotionPlan,
    PlannedStep,
    build_formal_group_a,
    build_formal_group_b,
    offset_joint,
)

from conftest import build_config

BUILDERS = {"A": build_formal_group_a, "B": build_formal_group_b}
AMPLITUDES = (0.01, 0.05, 0.2)


def _hardware(config: AppConfig, *, step_deg: float) -> AppConfig:
    config.mode = "hardware"
    config.formal.staircase_n = 5
    config.formal.repeats = 3
    config.formal.step_deg = {"J1": step_deg}
    config.validate()
    return config


def _leg_amounts(plan: MotionPlan, *, start: tuple[float, ...] | None = None) -> list[float]:
    """按**时间顺序**手工列出每一段运动的实际运动量。

    规则照需求一·5 的原话：``本次实际运动量 = 本次目标关节角 − 上一次目标关节角``。
    起点是**名义位姿**（每个组都是站在名义位姿上开动的），这一步是手工写在
    测试里的、不从被测代码取——所以"第一条腿走多少"这件事在测试里是独立的。

    这一段实现**故意**不复用被测代码的循环：它只用 ``segment_step_groups``
    这个"哪几步算一段"的公开规则，然后自己维护"上一个目标"。
    """
    previous = tuple(float(v) for v in (start or plan.nominal_joint_deg))
    amounts: list[float] = []
    for group in segment_step_groups(list(plan.steps)):
        for step in group:
            target = step.target_joint_deg
            if target is None:
                continue
            amounts.append(motion_amount_deg(previous, target))
            previous = tuple(float(v) for v in target)
    return amounts


def _hand_summed_seconds(config: AppConfig, plan: MotionPlan) -> float:
    """按需求一·5 列的那几项，把总录制时长**手工加一遍**。

    每一项都写出来（pre / 去程梯形 / settle_hold / 保持 / 回程梯形 +
    settle_hold + 回程保持 / post），不调用 ``estimate_capture_seconds``——
    两边独立算，对上了才说明估算没有漏项或重复计数。
    """
    durations = config.effective_durations()
    pre_s = float(durations["pre_motion"])
    post_s = float(durations["post_motion"])
    settle_hold = float(config.robot.settle_hold_s)
    speed, accel = config.effective_speed(formal=True)
    total = 0.0
    previous = tuple(float(v) for v in plan.nominal_joint_deg)
    for group in segment_step_groups(list(plan.steps)):
        primary = group[0]
        if primary.target_joint_deg is None:
            # 纯等待步：一帧都不写盘，一秒都不计。
            assert primary.event.role == ROLE_WAIT
            continue
        target = tuple(float(v) for v in primary.target_joint_deg)
        amount = motion_amount_deg(previous, target)
        total += pre_s + trapezoid_seconds(amount, speed, accel) + settle_hold
        total += float(primary.hold_s)
        previous = target
        if len(group) > 1:
            follow = group[1]
            back_target = tuple(float(v) for v in follow.target_joint_deg)
            back = motion_amount_deg(previous, back_target)
            total += trapezoid_seconds(back, speed, accel) + settle_hold
            total += float(follow.hold_s) + post_s
            previous = back_target
    return total


@pytest.mark.parametrize("step_deg", AMPLITUDES)
@pytest.mark.parametrize("group", sorted(BUILDERS))
def test_every_leg_of_the_formal_groups_is_exactly_one_step(
    tmp_path: Path, group: str, step_deg: float
) -> None:
    """两组的**每一条腿**都恰好走 Δ——不是 kΔ，也不是 0。

    组 A：5 级阶梯去程 + 原路下来的回程，每一级的相邻差都是 Δ；
    组 B：每一级都是"走到 ±Δ 再回名义"，去程 Δ、回程也 Δ。
    所以正确估算下，"一遍 repeat 的运动量总和"应该正好等于 **腿数 × Δ**。
    """
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(step_deg,), repeats=1),
        step_deg=step_deg,
    )
    plan = BUILDERS[group](config, config.robot.nominal_joint_deg, "J1", step_deg)
    amounts = _leg_amounts(plan)

    assert amounts, f"组{group} 一条运动腿都没有"
    offenders = [value for value in amounts if abs(value - step_deg) > 1e-9]
    assert not offenders, (
        f"组{group} Δ={step_deg}°：有 {len(offenders)} 条腿的运动量不是 Δ，"
        f"例如 {offenders[:5]}——说明还在拿'相对名义的偏移'当运动量"
    )
    assert abs(sum(amounts) - len(amounts) * step_deg) < 1e-9

    # 反向证据：老口径（相对名义的偏移之和）在两组上都**明显不同**，
    # 而且方向相反——这正是需求一·5 说的"组A 高估、组B 低估"。
    old_total = sum(
        abs(float(step.event.expected_delta_deg or 0.0))
        for step in plan.steps
        if step.target_joint_deg is not None
    )
    if group == "A":
        assert old_total > sum(amounts) * 2.0, (
            f"组A 的老口径总运动量 {old_total} 没有明显大于真实量 {sum(amounts)}——"
            "相邻阶梯被当成'从名义走了 kΔ'了？"
        )
    else:
        assert old_total < sum(amounts) * 0.5, (
            f"组B 的老口径总运动量 {old_total} 没有明显小于真实量 {sum(amounts)}——"
            "回程被当成 0 了？"
        )


@pytest.mark.parametrize("step_deg", AMPLITUDES)
@pytest.mark.parametrize("group", sorted(BUILDERS))
def test_the_recorded_seconds_match_the_hand_summed_leg_list(
    tmp_path: Path, group: str, step_deg: float
) -> None:
    """总录制时长 = 手工按逐条腿加出来的那一份（含 pre/post/settle/保持）。"""
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(step_deg,), repeats=1),
        step_deg=step_deg,
    )
    plan = BUILDERS[group](config, config.robot.nominal_joint_deg, "J1", step_deg)
    estimated = estimate_capture_seconds(config, [plan])
    expected = _hand_summed_seconds(config, plan)
    assert abs(estimated - expected) < 1e-9, (
        f"组{group} Δ={step_deg}°：工具算出 {estimated:.6f} s，"
        f"逐条腿手工加出来是 {expected:.6f} s"
    )

    # 纯等待步（组间 3 s 等待）**不写 RAW**，所以它一秒都不该计入。
    wait_seconds = sum(
        float(step.hold_s)
        for step in plan.steps
        if step.event.role == ROLE_WAIT
    )
    assert wait_seconds > 0, "这一组里本来应该有组间等待步"
    without_waits = estimate_capture_seconds(
        config,
        [
            MotionPlan(
                name=plan.name,
                nominal_joint_deg=plan.nominal_joint_deg,
                steps=[s for s in plan.steps if s.event.role != ROLE_WAIT],
            )
        ],
    )
    assert abs(without_waits - estimated) < 1e-9, (
        f"把等待步删掉之后估算从 {estimated:.6f} s 变成 {without_waits:.6f} s——"
        f"说明那 {wait_seconds:.1f} s 的等待被算进了 RAW 录制时间"
    )


@pytest.mark.parametrize("step_deg", AMPLITUDES)
def test_the_wrong_old_estimate_is_wrong_in_the_documented_direction(
    tmp_path: Path, step_deg: float
) -> None:
    """把"按相对名义的偏移估时长"复现一遍，确认它偏的方向和文档说的一致。

    这不是在测一段死代码，而是在**钉住结论**：以后谁要是把估算改回老口径，
    这条断言会立刻炸，并且告诉他"组A 会高估 X 倍、组B 会低估 Y 倍"。
    """
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(step_deg,), repeats=1),
        step_deg=step_deg,
    )
    durations = config.effective_durations()
    speed, accel = config.effective_speed(formal=True)
    settle_hold = float(config.robot.settle_hold_s)
    pre_s = float(durations["pre_motion"])
    post_s = float(durations["post_motion"])

    def old_style(plan: MotionPlan) -> float:
        """老口径：每一段按 ``expected_delta_deg`` 算运动量，回程算 0。"""
        total = 0.0
        for group in segment_step_groups(list(plan.steps)):
            primary = group[0]
            if primary.target_joint_deg is None:
                continue
            amount = abs(float(primary.event.expected_delta_deg or 0.0))
            total += pre_s + trapezoid_seconds(amount, speed, accel) + settle_hold
            total += float(primary.hold_s)
            if len(group) > 1:
                # ★ 老口径的错处就在这里：回程目标就是名义位姿，
                #   expected_delta_deg == 0 → 回程被算成 0 秒。
                back = abs(float(group[1].event.expected_delta_deg or 0.0))
                total += trapezoid_seconds(back, speed, accel) + settle_hold
                total += float(group[1].hold_s) + post_s
        return total

    real_a = estimate_capture_seconds(
        config, [build_formal_group_a(config, config.robot.nominal_joint_deg, "J1", step_deg)]
    )
    real_b = estimate_capture_seconds(
        config, [build_formal_group_b(config, config.robot.nominal_joint_deg, "J1", step_deg)]
    )
    old_a = old_style(
        build_formal_group_a(config, config.robot.nominal_joint_deg, "J1", step_deg)
    )
    old_b = old_style(
        build_formal_group_b(config, config.robot.nominal_joint_deg, "J1", step_deg)
    )

    assert old_a > real_a, (
        f"Δ={step_deg}°：组A 的老口径 {old_a:.2f} s 居然不高估（真实 {real_a:.2f} s）——"
        "相邻阶梯被当成'从名义走了 kΔ'了"
    )
    assert old_b < real_b, (
        f"Δ={step_deg}°：组B 的老口径 {old_b:.2f} s 居然不低估（真实 {real_b:.2f} s）——"
        "回程被当成 0 了"
    )
    if abs(step_deg - 0.2) < 1e-12:
        # 定稿实测（Δ=0.2°、硬件口径）：组A 老口径 361.5 s vs 真实 183.0 s（1.98 倍），
        # 组B 老口径 96.0 s vs 真实 366.0 s（0.26 倍）。写成"实测值 ± 余量"的护栏。
        assert old_a > real_a * 1.8, f"Δ=0.2°：组A 高估倍数只有 {old_a / real_a:.2f}"
        assert old_b < real_b * 0.35, f"Δ=0.2°：组B 低估倍数只有 {old_b / real_b:.2f}"
        # 组B 的老口径下每一段运动量都被算成 0，所以它**完全不随 Δ 变**——
        # 这正是"低估"最纯粹的样子，也解释了为什么小 Δ 时它反而只低估 28%。
        assert abs(old_b - 96.0) < 1e-6, (
            f"老口径下组B 的估算应该恒等于固定开销 96 s，实际 {old_b:.3f} s"
        )


def _motion_event(role: str, delta: float, *, index: int) -> MotionEvent:
    return MotionEvent(
        event_id=f"hand-{index}",
        # 用正式的 stage 名：正式实验与预实验用的是**两套速度**
        # （0.05 vs 0.5 °/s），挑错一套会把时长差出一个量级。
        stage=STAGE_FORMAL_A,
        joint="J1",
        joint_index=1,
        amplitude_deg=abs(delta),
        direction=1 if delta >= 0 else -1,
        repeat_index=1,
        role=role,
        counts_for_statistics=True,
        describe=f"手工步骤 {index}",
        expected_delta_deg=delta,
    )


def test_direction_reversal_counts_as_a_full_two_step_travel(tmp_path: Path) -> None:
    """**正负换向**：从 +Δ 走到 −Δ 是 **2Δ** 的运动量，不是 Δ、更不是 0。

    这是"只看相邻两个目标"最容易被写错的一处：换向那一步的
    ``expected_delta_deg`` 是 −Δ，看着像"只走 Δ"，实际机械臂要走 2Δ。
    用一份手工搭出来的最小计划（+Δ → 换向到 −Δ）钉住它。
    """
    step_deg = 0.05
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(step_deg,), repeats=1),
        step_deg=step_deg,
    )
    nominal = config.robot.nominal_joint_deg
    plus = offset_joint(nominal, 1, +step_deg)
    minus = offset_joint(nominal, 1, -step_deg)

    def step(target, delta: float, index: int) -> PlannedStep:
        return PlannedStep(
            event=_motion_event("move", delta, index=index),
            target_joint_deg=target,
            settle_required=True,
            hold_s=float(config.camera.hold_s),
            delta_from_nominal_deg=tuple(
                float(b) - float(a) for a, b in zip(nominal, target)
            ),
            label=f"手工步骤 {index}",
        )

    nominal_tuple = tuple(float(v) for v in nominal)
    plan = MotionPlan(
        name="手工换向计划",
        nominal_joint_deg=nominal_tuple,
        steps=[step(plus, +step_deg, 1), step(minus, -step_deg, 2)],
    )
    amounts = _leg_amounts(plan)
    assert amounts == pytest.approx([step_deg, 2 * step_deg]), (
        f"逐条腿的运动量是 {amounts}，换向那一步应该是 2Δ={2 * step_deg}°"
    )

    # 时长上也要看得见这个 2Δ：换向计划必须比"两步各走 Δ"的同形计划更长。
    # 对照计划把第二个目标放在"名义 + 2Δ"，从 +Δ 走过去正好也是 Δ。
    same_shape = MotionPlan(
        name="手工同向计划",
        nominal_joint_deg=nominal_tuple,
        steps=[
            step(plus, +step_deg, 1),
            step(offset_joint(nominal, 1, +2 * step_deg), +step_deg, 2),
        ],
    )
    assert _leg_amounts(same_shape) == pytest.approx([step_deg, step_deg])

    durations = config.effective_durations()
    speed, accel = config.effective_speed(formal=True)
    fixed = 2 * (
        float(durations["pre_motion"])
        + float(config.robot.settle_hold_s)
        + float(config.camera.hold_s)
    )
    expected = (
        fixed
        + trapezoid_seconds(step_deg, speed, accel)
        + trapezoid_seconds(2 * step_deg, speed, accel)
    )
    estimated = estimate_capture_seconds(config, [plan])
    assert abs(estimated - _hand_summed_seconds(config, plan)) < 1e-9
    assert abs(estimated - expected) < 1e-9, (
        f"换向计划估了 {estimated:.6f} s，按 2Δ 手工算应该是 {expected:.6f} s"
    )
    assert estimated > estimate_capture_seconds(config, [same_shape]) + 1.0, (
        "换向那一步的 2Δ 在时长上没有体现出来"
    )
