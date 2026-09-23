"""关节空间的"动作计划"：把需求三/五说的时间顺序翻译成一串带编号的动作。

这个模块**只生成计划，不发命令**。任何模式（dry_run / replay / hardware）都先拿到
同一份计划对象，再各自决定怎么执行——这样 dry-run 自测真的能覆盖真机会走的顺序。

两条容易写错、所以在这里被强制保证的语义
---------------------------------------
1. **不允许累计**（需求三 B 明确写了"不允许累计成 0.26°"）：
   每一档步长、每一个方向、每一次重复，都从**名义位姿**独立出发。
   所以计划里不允许出现"从上一个目标点再走一小步"这种连续动作；
   :func:`verify_plan` 会逐条核对，一旦发现累计就报错。
2. **回程数据另算**：每次动作都跟着一次"回到名义位姿"。
   回程帧照样采集、照样落盘，但 ``counts_for_statistics=False``，
   分析时不计入目标步长的统计（需求三 B：回程数据保存但不计入主统计）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import JOINT_NAMES, AppConfig
from .kinematics import JOINT_LIMITS_DEG, check_joint_in_limits

#: 动作在整场实验里的阶段名。
STAGE_APPROACH = "approach"
STAGE_QUICK_PROBE = "quick_probe"
STAGE_PRETEST = "pretest"
STAGE_FORMAL_A = "formal_a"
STAGE_FORMAL_B = "formal_b"
STAGE_STATIC = "static"

#: 动作角色。``move``/``hold`` 计入统计，``return`` 不计入（但要保存）。
ROLE_MOVE = "move"
ROLE_HOLD = "hold"
ROLE_RETURN = "return"
ROLE_SETTLE = "settle"
ROLE_WAIT = "wait"


class PlanError(RuntimeError):
    """计划本身不合法。消息中文，直接显示给实验者。"""


@dataclass(frozen=True)
class MotionEvent:
    """一个动作的身份信息。事件编号是原始数据里的主键。"""

    event_id: str
    stage: str
    joint: str | None
    joint_index: int | None
    amplitude_deg: float | None
    direction: int  # +1 / -1；不涉及方向时为 0
    repeat_index: int  # 从 1 开始；不重复时固定 1
    role: str
    counts_for_statistics: bool
    describe: str
    #: 这一步目标关节应该相对名义位姿变化多少度。
    #: 单步微动＝``方向×步长``；组 A 阶梯＝``步长×级数``（所以不能只看方向×步长）。
    #: 到位过程是六个关节一起变，留 None 表示"不按这条核对"。
    expected_delta_deg: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "stage": self.stage,
            "joint": self.joint,
            "joint_index": self.joint_index,
            "amplitude_deg": self.amplitude_deg,
            "direction": self.direction,
            "repeat_index": self.repeat_index,
            "role": self.role,
            "counts_for_statistics": self.counts_for_statistics,
            "describe": self.describe,
            "expected_delta_deg": self.expected_delta_deg,
        }


@dataclass(frozen=True)
class PlannedStep:
    """计划里的一步：要么是运动到某个关节配置，要么是原地等待。"""

    event: MotionEvent
    #: 目标关节角（度，6 个）。纯等待步为 None。
    target_joint_deg: tuple[float, ...] | None
    #: 到位后是否要等关节停稳再继续。
    settle_required: bool
    #: 到位（或停稳）之后再保持/等待多久，单位秒。
    hold_s: float
    #: 这一步相对名义位姿的关节角变化（度，6 个），只用于显示与核对。
    delta_from_nominal_deg: tuple[float, ...] | None
    #: 界面/日志上一行说明。
    label: str

    @property
    def is_motion(self) -> bool:
        return self.target_joint_deg is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "is_motion": self.is_motion,
            "settle_required": self.settle_required,
            "hold_s": self.hold_s,
            "target_joint_deg": (
                None if self.target_joint_deg is None else list(self.target_joint_deg)
            ),
            "delta_from_nominal_deg": (
                None
                if self.delta_from_nominal_deg is None
                else list(self.delta_from_nominal_deg)
            ),
            **self.event.to_dict(),
        }


@dataclass
class MotionPlan:
    """一整段实验的计划。"""

    name: str
    nominal_joint_deg: tuple[float, ...]
    steps: list[PlannedStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nominal_joint_deg": list(self.nominal_joint_deg),
            "step_count": len(self.steps),
            "motion_step_count": sum(1 for step in self.steps if step.is_motion),
            "steps": [step.to_dict() for step in self.steps],
        }

    def motion_steps(self) -> list[PlannedStep]:
        return [step for step in self.steps if step.is_motion]

    def statistics_steps(self) -> list[PlannedStep]:
        return [
            step
            for step in self.steps
            if step.is_motion and step.event.counts_for_statistics
        ]

    def summary_lines(self, limit: int = 40) -> list[str]:
        lines = [
            f"计划「{self.name}」：共 {len(self.steps)} 步，"
            f"其中运动 {len(self.motion_steps())} 步，计入统计 "
            f"{len(self.statistics_steps())} 步。"
        ]
        for step in self.steps[:limit]:
            lines.append("  " + step.label)
        if len(self.steps) > limit:
            lines.append(f"  …… 其余 {len(self.steps) - limit} 步略。")
        return lines


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _fmt_amplitude(value: float | None) -> str:
    """把步长写进事件编号里，用 p 代替小数点，避免文件名里出现点号。"""
    if value is None:
        return "na"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def _clean_angle(value: float) -> float:
    """避免 -0.0 这种在 CSV 里看起来像负数的写法。"""
    return 0.0 if abs(value) < 1e-12 else float(value)


def _as_six(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(_clean_angle(float(value)) for value in values)


def joint_delta(
    reference: Sequence[float], target: Sequence[float]
) -> tuple[float, ...]:
    """目标相对参考的逐关节角度差（度）。"""
    return _as_six([float(b) - float(a) for a, b in zip(reference, target)])


def offset_joint(
    nominal: Sequence[float], joint_index: int, delta_deg: float
) -> tuple[float, ...]:
    """在名义位姿上只动一个关节。"""
    if not 0 <= joint_index < 6:
        raise PlanError(f"关节序号必须在 0～5，收到 {joint_index}")
    target = [float(value) for value in nominal]
    target[joint_index] += float(delta_deg)
    return _as_six(target)


def _check_target(
    target: Sequence[float], *, margin_deg: float, context: str
) -> None:
    ok, reason = check_joint_in_limits(
        target, margin_deg=margin_deg, limits=JOINT_LIMITS_DEG
    )
    if not ok:
        raise PlanError(f"{context} 不满足关节限位要求：{reason}")


def _event(
    *,
    stage: str,
    joint: str | None,
    amplitude_deg: float | None,
    direction: int,
    repeat_index: int,
    role: str,
    counts_for_statistics: bool,
    describe: str,
    suffix: str,
    expected_delta_deg: float | None = None,
) -> MotionEvent:
    joint_index = JOINT_NAMES.index(joint) if joint else None
    parts = [
        stage,
        joint or "all",
        f"a{_fmt_amplitude(amplitude_deg)}",
        f"d{direction:+d}" if direction else "d0",
        f"r{repeat_index:02d}",
        role,
        suffix,
    ]
    return MotionEvent(
        event_id="-".join(parts),
        stage=stage,
        joint=joint,
        joint_index=joint_index,
        amplitude_deg=amplitude_deg,
        direction=int(direction),
        repeat_index=int(repeat_index),
        role=role,
        counts_for_statistics=bool(counts_for_statistics),
        describe=describe,
        expected_delta_deg=expected_delta_deg,
    )


# --------------------------------------------------------------------------
# 按钮一：到位过程
# --------------------------------------------------------------------------


def build_approach_plan(
    current_joint_deg: Sequence[float],
    nominal_joint_deg: Sequence[float],
    *,
    points: int = 6,
    hold_s: float = 0.3,
    limit_margin_deg: float = 1.0,
) -> MotionPlan:
    """从"当前角度"到"推荐姿态"的关节空间中间点。

    需求三要求 5～6 个中间点、每个点显示相对当前位置的差值、每步人工确认。
    这里按线性插值给点，最后一个点**恰好**等于推荐姿态——不用运动学反解，
    因为全程只动关节角，越简单越不容易出意外。
    """
    if not 2 <= points <= 12:
        raise PlanError(f"中间点数量应在 2～12，收到 {points}")
    start = _as_six(current_joint_deg)
    target = _as_six(nominal_joint_deg)

    plan = MotionPlan(name="到达实验姿态", nominal_joint_deg=target)
    for index in range(1, points + 1):
        fraction = index / points
        waypoint = _as_six(
            [a + (b - a) * fraction for a, b in zip(start, target)]
        )
        _check_target(
            waypoint,
            margin_deg=limit_margin_deg,
            context=f"到位过程第 {index}/{points} 点",
        )
        delta_from_current = joint_delta(start, waypoint)
        delta_from_previous = joint_delta(
            plan.steps[-1].target_joint_deg if plan.steps else start, waypoint
        )
        biggest = max(range(6), key=lambda i: abs(delta_from_previous[i]))
        describe = (
            f"到位 {index}/{points}：整体走 {fraction * 100:.0f}%，"
            f"本步最大变化在 J{biggest + 1} {delta_from_previous[biggest]:+.4f}°"
        )
        moving_from_current = [
            f"J{i + 1}{delta_from_current[i]:+.4f}°"
            for i in range(6)
            if abs(delta_from_current[i]) > 1e-9
        ]
        if moving_from_current:
            describe = f"{describe}；相对当前位姿 " + "、".join(moving_from_current)
        else:
            # 当前姿态已经就是实验姿态（干运行每次都这样，真机上偶尔也会碰上）：
            # 这时"相对当前位姿 "后面一个关节都列不出来，会留下一句读不通的
            # 半截话。宁可明说"不需要动"。
            describe = f"{describe}；相对当前位姿不需要移动（已经在实验姿态上）"
        plan.steps.append(
            PlannedStep(
                event=_event(
                    stage=STAGE_APPROACH,
                    joint=None,
                    amplitude_deg=None,
                    direction=0,
                    repeat_index=index,
                    role=ROLE_MOVE,
                    counts_for_statistics=False,
                    describe=describe,
                    suffix=f"pt{index:02d}",
                ),
                target_joint_deg=waypoint,
                settle_required=True,
                hold_s=hold_s,
                delta_from_nominal_deg=delta_from_current,
                label=describe,
            )
        )
    return plan


# --------------------------------------------------------------------------
# 按钮二：静态噪声 + 快速几何确认 + 三档预实验
# --------------------------------------------------------------------------


def build_static_plan(
    nominal_joint_deg: Sequence[float],
    *,
    duration_s: float,
) -> MotionPlan:
    """静态噪声测量：完全不动，只录一段。

    还是做成"计划"而不是直接录，是为了让干运行能走同一条代码路径：
    干运行只需要检查"计划里没有任何运动步"就够了。
    """
    target = _as_six(nominal_joint_deg)
    plan = MotionPlan(name="静态噪声基线", nominal_joint_deg=target)
    plan.steps.append(
        PlannedStep(
            event=_event(
                stage=STAGE_STATIC,
                joint=None,
                amplitude_deg=None,
                direction=0,
                repeat_index=1,
                role=ROLE_HOLD,
                counts_for_statistics=False,
                describe=f"保持不动 {duration_s:.1f} s，测量静态噪声",
                suffix="hold",
            ),
            target_joint_deg=None,
            settle_required=False,
            hold_s=float(duration_s),
            delta_from_nominal_deg=(0.0,) * 6,
            label=f"[静态] 保持推荐姿态 {duration_s:.1f} s（不发送任何运动）",
        )
    )
    return plan


def build_quick_probe_plan(
    nominal_joint_deg: Sequence[float],
    joints: Sequence[str],
    *,
    probe_deg: float,
    repeat_index: int = 1,
    limit_margin_deg: float = 1.0,
    hold_s: float = 1.0,
    return_settle_s: float = 0.2,
) -> MotionPlan:
    """三档预实验之前的"快速几何确认"：每个关节用 0.05° 正负各试一次。

    需求三 B 要求先看：棋盘格是否完整、画面是否明显动了、方向是否大致符合理论、
    是否主要在面内、有没有严重掉帧。这里只生成动作；判定在 :mod:`analysis`。
    """
    nominal = _as_six(nominal_joint_deg)
    plan = MotionPlan(name="快速几何确认", nominal_joint_deg=nominal)
    for name in joints:
        index = _joint_index(name)
        for direction in (+1, -1):
            target = offset_joint(nominal, index, direction * probe_deg)
            _check_target(
                target,
                margin_deg=limit_margin_deg,
                context=f"快速确认 {name} {direction:+d}",
            )
            _append_trial(
                plan,
                stage=STAGE_QUICK_PROBE,
                joint=name,
                joint_index=index,
                nominal=nominal,
                target=target,
                amplitude_deg=float(probe_deg),
                direction=direction,
                repeat_index=repeat_index,
                hold_s=hold_s,
                return_settle_s=return_settle_s,
            )
    return plan


def build_pretest_plan(
    config: AppConfig,
    nominal_joint_deg: Sequence[float],
) -> MotionPlan:
    """按钮二的主体：六个关节 × 三档步长 × 正负两方向 × 每方向 N 次。

    默认规模 = 6 × 3 × 2 × 2 = 72 次动作（需求三 B 的数字），
    每次动作都从名义位姿独立出发，动作之间回到名义位姿再停稳。
    """
    nominal = _as_six(nominal_joint_deg)
    plan = MotionPlan(name="三档微动预实验", nominal_joint_deg=nominal)
    amplitudes = sorted(float(v) for v in config.pretest.amplitudes_deg)
    repeats = int(config.pretest.repeats_per_direction)
    margin = config.thresholds.joint_limit_margin_deg

    for name in config.pretest.joints:
        index = _joint_index(name)
        for amplitude in amplitudes:
            for direction in (+1, -1):
                for repeat in range(1, repeats + 1):
                    target = offset_joint(nominal, index, direction * amplitude)
                    _check_target(
                        target,
                        margin_deg=margin,
                        context=f"预实验 {name} {direction:+d}{amplitude}°",
                    )
                    _append_trial(
                        plan,
                        stage=STAGE_PRETEST,
                        joint=name,
                        joint_index=index,
                        nominal=nominal,
                        target=target,
                        amplitude_deg=amplitude,
                        direction=direction,
                        repeat_index=repeat,
                        hold_s=config.camera.hold_s,
                        return_settle_s=config.pretest.return_before_settle_s,
                    )
    return plan


def _append_trial(
    plan: MotionPlan,
    *,
    stage: str,
    joint: str,
    joint_index: int,
    nominal: tuple[float, ...],
    target: tuple[float, ...],
    amplitude_deg: float,
    direction: int,
    repeat_index: int,
    hold_s: float,
    return_settle_s: float,
) -> None:
    """把"去 → 停 → 回 → 停稳"四步追加到计划里。

    四步各有自己的事件编号：分析时可以精确指出"是去程的哪一次重复"。
    回程那一步 ``counts_for_statistics=False``，因为它测的是回差方向，
    混进目标步长统计会把正负两向的平均值拖回零。
    """
    sign_text = "正" if direction > 0 else "负"
    common = {
        "stage": stage,
        "joint": joint,
        "amplitude_deg": float(amplitude_deg),
        "direction": direction,
        "repeat_index": repeat_index,
    }
    plan.steps.append(
        PlannedStep(
            event=_event(
                **common,
                role=ROLE_MOVE,
                counts_for_statistics=True,
                describe=(
                    f"{joint} 从名义位姿走 {direction * amplitude_deg:+.4f}°"
                    f"（{sign_text}向第 {repeat_index} 次）"
                ),
                suffix="to",
                expected_delta_deg=direction * float(amplitude_deg),
            ),
            target_joint_deg=target,
            settle_required=True,
            hold_s=float(hold_s),
            delta_from_nominal_deg=joint_delta(nominal, target),
            label=(
                f"[去程] {joint} {direction * amplitude_deg:+.4f}° "
                f"（{sign_text}向 #{repeat_index}）"
            ),
        )
    )
    plan.steps.append(
        PlannedStep(
            event=_event(
                **common,
                role=ROLE_RETURN,
                counts_for_statistics=False,
                describe=f"{joint} 回到名义位姿（回程，不计入目标步长统计）",
                suffix="back",
                expected_delta_deg=0.0,
            ),
            target_joint_deg=nominal,
            settle_required=True,
            hold_s=float(return_settle_s),
            delta_from_nominal_deg=(0.0,) * 6,
            label=(
                f"[回程] {joint} 回名义位姿（{sign_text}向 #{repeat_index}）"
                "——数据保存但不计入主统计"
            ),
        )
    )


# --------------------------------------------------------------------------
# 按钮三之后：正式实验
# --------------------------------------------------------------------------


def build_formal_group_a(
    config: AppConfig,
    nominal_joint_deg: Sequence[float],
    joint: str,
    step_deg: float,
) -> MotionPlan:
    """组 A：单向阶梯 q0 → q0+Δq → … → q0+NΔq，再原路下来。

    去程和回程要**分别分析**（需求五），所以两段用不同的 role：
    去程每一步都是 ``move``，回程每一步是 ``return``。
    """
    nominal = _as_six(nominal_joint_deg)
    index = _joint_index(joint)
    n = int(config.formal.staircase_n)
    repeats = int(config.formal.repeats)
    margin = config.thresholds.joint_limit_margin_deg
    plan = MotionPlan(name=f"组A 阶梯 {joint} Δ={step_deg}°", nominal_joint_deg=nominal)

    targets = [offset_joint(nominal, index, step_deg * level) for level in range(0, n + 1)]
    for target in targets:
        _check_target(target, margin_deg=margin, context=f"组A {joint} 阶梯点")

    for repeat in range(1, repeats + 1):
        for level in range(1, n + 1):
            target = targets[level]
            plan.steps.append(
                PlannedStep(
                    event=_event(
                        stage=STAGE_FORMAL_A, joint=joint, amplitude_deg=step_deg,
                        direction=+1, repeat_index=repeat, role=ROLE_MOVE,
                        counts_for_statistics=True,
                        describe=f"组A 第 {level} 级 {joint} {(step_deg * level):+.4f}°",
                        suffix=f"up{level:02d}",
                        expected_delta_deg=step_deg * level,
                    ),
                    target_joint_deg=target,
                    settle_required=True,
                    hold_s=float(config.formal.hold_s),
                    delta_from_nominal_deg=joint_delta(nominal, target),
                    label=(
                        f"[组A 去程] #{repeat} 第 {level}/{n} 级 "
                        f"{joint} {step_deg * level:+.4f}°"
                    ),
                )
            )
        for level in range(n, 0, -1):
            target = targets[level - 1]
            plan.steps.append(
                PlannedStep(
                    event=_event(
                        stage=STAGE_FORMAL_A, joint=joint, amplitude_deg=step_deg,
                        direction=-1, repeat_index=repeat, role=ROLE_RETURN,
                        counts_for_statistics=False,
                        describe=f"组A 回程 降到第 {level - 1} 级",
                        suffix=f"down{level - 1:02d}",
                        expected_delta_deg=step_deg * (level - 1),
                    ),
                    target_joint_deg=target,
                    settle_required=True,
                    hold_s=float(config.formal.hold_s),
                    delta_from_nominal_deg=joint_delta(nominal, target),
                    label=(
                        f"[组A 回程] #{repeat} 降到第 {level - 1}/{n} 级 "
                        f"{joint} {step_deg * (level - 1):+.4f}°（单独分析）"
                    ),
                )
            )
        plan.steps.append(
            PlannedStep(
                event=_event(
                    stage=STAGE_FORMAL_A, joint=joint, amplitude_deg=step_deg,
                    direction=0, repeat_index=repeat, role=ROLE_WAIT,
                    counts_for_statistics=False,
                    describe=f"组A 第 {repeat} 遍结束，等待 {config.formal.group_wait_s} s",
                    suffix="wait",
                ),
                target_joint_deg=None,
                settle_required=False,
                hold_s=float(config.formal.group_wait_s),
                delta_from_nominal_deg=(0.0,) * 6,
                label=f"[组A] #{repeat} 遍结束，等待 {config.formal.group_wait_s:.1f} s",
            )
        )
    return plan


def build_formal_group_b(
    config: AppConfig,
    nominal_joint_deg: Sequence[float],
    joint: str,
    step_deg: float,
) -> MotionPlan:
    """组 B：换向实验 q0 → q0+Δq → q0 → q0−Δq → q0 → …，重复 N 遍。

    换向实验专门看回差和死区，所以每一步都"回到名义位姿再往另一边走"，
    中间不做连续换向。
    """
    nominal = _as_six(nominal_joint_deg)
    index = _joint_index(joint)
    n = int(config.formal.staircase_n)
    repeats = int(config.formal.repeats)
    margin = config.thresholds.joint_limit_margin_deg
    plus = offset_joint(nominal, index, step_deg)
    minus = offset_joint(nominal, index, -step_deg)
    for target in (plus, minus):
        _check_target(target, margin_deg=margin, context=f"组B {joint} 换向点")

    plan = MotionPlan(name=f"组B 换向 {joint} Δ={step_deg}°", nominal_joint_deg=nominal)
    for repeat in range(1, repeats + 1):
        for cycle in range(1, n + 1):
            for direction, target in ((+1, plus), (-1, minus)):
                sign_text = "正" if direction > 0 else "负"
                plan.steps.append(
                    PlannedStep(
                        event=_event(
                            stage=STAGE_FORMAL_B, joint=joint, amplitude_deg=step_deg,
                            direction=direction, repeat_index=repeat, role=ROLE_MOVE,
                            counts_for_statistics=True,
                            describe=(
                                f"组B #{repeat} 第 {cycle} 次换向，{sign_text}向 "
                                f"{direction * step_deg:+.4f}°"
                            ),
                            suffix=f"c{cycle:02d}",
                        ),
                        target_joint_deg=target,
                        settle_required=True,
                        hold_s=float(config.formal.hold_s),
                        delta_from_nominal_deg=joint_delta(nominal, target),
                        label=(
                            f"[组B] #{repeat} 换向 {cycle}/{n} "
                            f"{sign_text}向 {direction * step_deg:+.4f}°"
                        ),
                    )
                )
                plan.steps.append(
                    PlannedStep(
                        event=_event(
                            stage=STAGE_FORMAL_B, joint=joint, amplitude_deg=step_deg,
                            direction=direction, repeat_index=repeat, role=ROLE_RETURN,
                            counts_for_statistics=False,
                            describe="组B 回名义位姿",
                            suffix=f"c{cycle:02d}b",
                            expected_delta_deg=0.0,
                        ),
                        target_joint_deg=nominal,
                        settle_required=True,
                        hold_s=float(config.formal.hold_s),
                        delta_from_nominal_deg=(0.0,) * 6,
                        label=f"[组B] #{repeat} 换向 {cycle}/{n} 回名义位姿（单独分析）",
                    )
                )
        plan.steps.append(
            PlannedStep(
                event=_event(
                    stage=STAGE_FORMAL_B, joint=joint, amplitude_deg=step_deg,
                    direction=0, repeat_index=repeat, role=ROLE_WAIT,
                    counts_for_statistics=False,
                    describe=f"组B 第 {repeat} 遍结束，等待 {config.formal.group_wait_s} s",
                    suffix="wait",
                ),
                target_joint_deg=None,
                settle_required=False,
                hold_s=float(config.formal.group_wait_s),
                delta_from_nominal_deg=(0.0,) * 6,
                label=f"[组B] #{repeat} 遍结束，等待 {config.formal.group_wait_s:.1f} s",
            )
        )
    return plan


def _joint_index(name: str) -> int:
    if name not in JOINT_NAMES:
        raise PlanError(f"未知关节名 {name!r}，应为 {list(JOINT_NAMES)} 之一。")
    return JOINT_NAMES.index(name)


# --------------------------------------------------------------------------
# 自检：计划必须满足的硬语义
# --------------------------------------------------------------------------


def verify_plan(plan: MotionPlan, *, tolerance_deg: float = 1e-9) -> list[str]:
    """逐条核对计划的硬语义，返回中文问题清单（空列表 = 通过）。

    这些是需求里写死的语义，任何一条不满足都应该在**发命令之前**就报错：
    * 事件编号唯一；
    * 运动步的目标角度只改一个关节（预实验/组A/组B），其它五个关节必须等于名义位姿；
    * 预实验里每个"去程"都从名义位姿出发（＝不允许累计）；
    * 正负方向与目标角的符号一致；
    * 回程步一律 ``counts_for_statistics=False``。
    """
    problems: list[str] = []
    nominal = plan.nominal_joint_deg

    seen: dict[str, int] = {}
    for index, step in enumerate(plan.steps):
        event_id = step.event.event_id
        if event_id in seen:
            problems.append(
                f"第 {index + 1} 步的事件编号 {event_id} 与第 {seen[event_id] + 1} 步重复。"
            )
        else:
            seen[event_id] = index

    # 1) 事件编号格式里带上方向和重复序号，这里再核对一次"编号里写的 = 实际做的"。
    for index, step in enumerate(plan.steps):
        event = step.event
        if event.amplitude_deg is not None and event.direction:
            expected_tag = f"d{event.direction:+d}"
            if expected_tag not in event.event_id:
                problems.append(
                    f"第 {index + 1} 步的事件编号 {event.event_id} 里没有正确的方向标记 "
                    f"{expected_tag}。"
                )
        if f"r{event.repeat_index:02d}" not in event.event_id:
            problems.append(
                f"第 {index + 1} 步的事件编号 {event.event_id} 里没有正确的重复序号 "
                f"r{event.repeat_index:02d}。"
            )

    # 2) 运动步的目标角度核对。
    for index, step in enumerate(plan.steps):
        if not step.is_motion:
            continue
        target = step.target_joint_deg
        assert target is not None
        event = step.event
        changed = [
            i for i in range(6) if abs(target[i] - nominal[i]) > tolerance_deg
        ]
        # 回程步的要求和去程不同：
        # * 微动实验的回程目标就是名义位姿本身（expected_delta_deg=0），
        #   所以"必须变化一个关节"这条不适用；
        # * 组 A 的回程是"阶梯往下退"，目标不是名义位姿，而是上一级，
        #   所以也不能统一要求"必须等于名义位姿"。
        # 统一按事件里写明的 expected_delta_deg 核对，两种情形都对得上。
        if event.role == ROLE_RETURN:
            if event.counts_for_statistics:
                problems.append(
                    f"第 {index + 1} 步（{event.event_id}）是回程却被标成计入统计。"
                )
            if event.expected_delta_deg is None:
                problems.append(
                    f"第 {index + 1} 步（{event.event_id}）是回程，"
                    "但没有写明它应该退到哪个角度，无法核对。"
                )
                continue
            changed_return = [
                i for i in range(6) if abs(target[i] - nominal[i]) > tolerance_deg
            ]
            if len(changed_return) > 1:
                problems.append(
                    f"第 {index + 1} 步（{event.event_id}）是回程，"
                    f"但同时改了 {len(changed_return)} 个关节。"
                )
                continue
            if event.joint_index is not None and changed_return and (
                changed_return[0] != event.joint_index
            ):
                problems.append(
                    f"第 {index + 1} 步（{event.event_id}）是回程，"
                    "改动的关节和事件里记的不是同一个。"
                )
                continue
            if event.joint_index is None:
                continue
            actual_return = target[event.joint_index] - nominal[event.joint_index]
            if abs(actual_return - float(event.expected_delta_deg)) > 1e-6:
                problems.append(
                    f"第 {index + 1} 步（{event.event_id}）回程目标角度差 "
                    f"{actual_return:+.6f}° 与事件里的 "
                    f"{float(event.expected_delta_deg):+.6f}° 不一致。"
                )
                if abs(float(event.expected_delta_deg)) < tolerance_deg:
                    problems.append(
                        f"第 {index + 1} 步（{event.event_id}）按事件应当回到名义位姿，"
                        "但没有回到。"
                    )
            continue

        if event.joint_index is None:
            # 到位过程允许六个关节一起变，但要落在"当前位姿→名义位姿"的连线上，
            # 表现为逐点单调靠近名义位姿。
            if event.stage == STAGE_APPROACH:
                distance = max(abs(target[i] - nominal[i]) for i in range(6))
                if index > 0:
                    previous_event = plan.steps[index - 1].event
                    if previous_event.stage == STAGE_APPROACH:
                        previous_target = plan.steps[index - 1].target_joint_deg
                        assert previous_target is not None
                        previous_distance = max(
                            abs(previous_target[i] - nominal[i]) for i in range(6)
                        )
                        if distance > previous_distance + tolerance_deg:
                            problems.append(
                                f"到位过程第 {index + 1} 步离推荐姿态反而更远了，"
                                "插值顺序不对。"
                            )
            continue

        if len(changed) != 1 or changed[0] != event.joint_index:
            problems.append(
                f"第 {index + 1} 步（{event.event_id}）改动的不止目标关节："
                f"变化关节序号 {changed}，目标关节序号 {event.joint_index}。"
                "微动实验只允许动一个关节。"
            )
            continue

        if event.expected_delta_deg is not None:
            expected_delta = float(event.expected_delta_deg)
        elif event.amplitude_deg:
            expected_delta = event.direction * event.amplitude_deg
        else:
            expected_delta = 0.0
        actual_delta = target[event.joint_index] - nominal[event.joint_index]
        if abs(actual_delta - expected_delta) > 1e-6:
            problems.append(
                f"第 {index + 1} 步（{event.event_id}）目标角度差 "
                f"{actual_delta:+.6f}° 与事件里的 {expected_delta:+.6f}° 不一致。"
            )

        if event.direction > 0 and actual_delta < -tolerance_deg:
            problems.append(f"第 {index + 1} 步标了正向却往负方向走。")
        if event.direction < 0 and actual_delta > tolerance_deg:
            problems.append(f"第 {index + 1} 步标了负向却往正方向走。")

    # 3) 不允许累计：预实验的每个去程前面，必须是名义位姿（第一步除外）。
    pretest_steps = [
        step for step in plan.steps if step.event.stage == STAGE_PRETEST
    ]
    for position, step in enumerate(pretest_steps):
        if step.event.role != ROLE_MOVE:
            continue
        if position == 0:
            continue
        previous = pretest_steps[position - 1]
        if previous.event.role != ROLE_RETURN:
            problems.append(
                f"预实验里 {step.event.event_id} 不是紧跟在回程之后，"
                "存在把步长累计起来的风险。"
            )
        if previous.target_joint_deg is not None:
            drift = max(
                abs(previous.target_joint_deg[i] - nominal[i]) for i in range(6)
            )
            if drift > tolerance_deg:
                problems.append(
                    f"预实验里 {step.event.event_id} 的起点不是名义位姿"
                    f"（偏离 {drift:.6f}°）。"
                )
    if pretest_steps and pretest_steps[0].event.role != ROLE_MOVE:
        problems.append("预实验的第一步不是去程动作，顺序不对。")

    # 4) 回到名义位姿的步不许计入统计。
    for index, step in enumerate(plan.steps):
        if step.event.role == ROLE_RETURN and step.event.counts_for_statistics:
            problems.append(
                f"第 {index + 1} 步是回程（{step.event.event_id}）却被标成计入统计，"
                "会把正负方向的平均值拖回零。"
            )

    return problems


def describe_amplitudes(amplitudes: Iterable[float]) -> str:
    """把步长列表写成一行中文，用于日志。"""
    return "、".join(f"{value}°" for value in amplitudes)


def nominal_offset_summary(
    nominal: Sequence[float], target: Sequence[float]
) -> str:
    """把"目标相对名义位姿的差值"写成一行中文（界面按钮一要用）。"""
    deltas = joint_delta(nominal, target)
    parts = [
        f"J{index + 1}{deltas[index]:+.4f}°"
        for index in range(6)
        if not math.isclose(deltas[index], 0.0, abs_tol=1e-9)
    ]
    return "、".join(parts) if parts else "与名义位姿相同"


# --------------------------------------------------------------------------
# 开跑之前的规模估算
# --------------------------------------------------------------------------


def planned_plans(config: AppConfig, *, include_formal: bool = True) -> list[MotionPlan]:
    """按当前配置列出"整场实验"要跑的所有计划（不执行、不发命令）。

    只用配置，不需要连接任何设备——所以界面一打开就能把它算出来显示给人看。
    正式实验只算**已经填了步长**的关节：没填步长就没法生成组A/组B 的计划，
    这本身就是一条要显示给用户看的信息。
    """
    nominal = config.robot.nominal_joint_deg
    durations = config.effective_durations()
    plans = [
        build_static_plan(nominal, duration_s=float(durations["static"])),
        build_quick_probe_plan(
            nominal,
            config.pretest.joints,
            probe_deg=float(config.pretest.quick_probe_deg),
            hold_s=float(config.camera.hold_s),
            return_settle_s=float(config.pretest.return_before_settle_s),
        ),
        build_pretest_plan(config, nominal),
    ]
    if include_formal:
        for joint in config.pretest.joints:
            step = config.formal.step_deg.get(joint)
            if not step:
                continue
            if config.formal.enable_group_a:
                plans.append(build_formal_group_a(config, nominal, joint, float(step)))
            if config.formal.enable_group_b:
                plans.append(build_formal_group_b(config, nominal, joint, float(step)))
    return plans


def planned_scale_lines(config: AppConfig, *, measured: Any = None) -> list[str]:
    """整场实验预计要花多久、占多少盘。**估算**，不是承诺。

    这些数偏保守：运动时间按梯形速度规划算，停稳按判据的保持时间算，
    但真机上控制器怎么规划、现场要不要中途重来，都会让它变长。
    写在开始之前，是为了让人知道"现在按下去要占用多久"——而不是
    按下之后才发现是几十分钟。

    ★ v1.0.3：``measured`` 给了就用**实测**的分辨率和帧率算（连上设备之后那次
    5 s 全屏采集检查的结果）。交付默认的 1936×1096 只是名义值——相机实际出的
    画面尺寸和帧率都可能不一样，而磁盘估算差一点点就是"这一组能不能开始"。
    """
    from .config import MeasuredCapture, estimate_capture_seconds, estimate_disk_gb

    plans = planned_plans(config)
    seconds = estimate_capture_seconds(config, plans, measured=measured)
    size_gb = estimate_disk_gb(config, seconds, measured=measured)
    width, height = config.effective_camera_size(measured)
    fps = config.effective_fps(measured)
    origin = (
        "实测（连接设备后的 5 s 全屏采集检查）"
        if isinstance(measured, MeasuredCapture)
        else "配置里的名义值"
    )
    lines = [
        f"整场实验规模（按当前配置估算）：录制约 {seconds / 60.0:.0f} 分钟"
        f"（{seconds:.0f} 秒），RAW 约 {size_gb:.1f} GB"
        f"（全屏 {width}×{height} @ {fps:.2f} fps，尺寸与帧率取自{origin}）。"
    ]
    if config.camera.resolved_analysis_roi() is not None:
        lines.append(
            f"（整场估算按**整幅**算：camera.analysis_roi="
            f"{list(config.camera.analysis_roi or config.camera.roi or [])} "
            "只影响机器人停住之后的离线识别，不改变 RAW 的大小。）"
        )
    missing = [
        joint
        for joint in config.pretest.joints
        if not config.formal.step_deg.get(joint)
    ]
    if missing:
        lines.append(
            "上面这个数**没有算**正式实验里还没填步长的关节："
            + "、".join(missing)
            + "。填好步长之后再看一次这个数。"
        )
    else:
        lines.append("上面这个数已经把组A/组B 的正式实验算进去了。")
    lines.append(
        "这只是估算：真机上控制器怎么规划、现场要不要重来，都会让它变长。"
    )
    return lines
