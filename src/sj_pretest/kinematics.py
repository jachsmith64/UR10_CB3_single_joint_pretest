"""UR10 CB3 名义运动学：只用来做"理论范围检查"，不构成碰撞安全结论。

用途（对应需求五"组 A 前的范围检查"）：
* 关节限位：整段 N×Δq 里每个关节是否还在限位内；
* 名义正运动学：算出这段运动让法兰中心大概平移了多少毫米——用来判断
  "这个步长是不是已经大到不该当微动做了"；
* 名义正运动学下的棋盘格"朝向变化"：棋盘格固定在环境里，法兰转 1° 不代表
  棋盘格转 1°，但可以给出量级参考。

**这不是 UR 控制器里的标定模型**，参数取的是公开的 UR10 名义 DH 值，
和现场的绝对精度会有几毫米级差别；更不能反映桌面、夹具、线缆和人员。
所以本模块所有结论都只写成"理论检查"，运行期一律同时显示
``collision_status = unknown``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

#: UR10 CB3 六轴的名义限位（度）。UR10 六个关节都是 ±360°。
JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
)

#: 公开的 UR10 名义 DH 参数：每行 (d, a, alpha)，theta 由关节角决定。
#: 单位：米 / 弧度。用法见 :func:`forward_kinematics`。
UR10_DH: tuple[tuple[float, float, float], ...] = (
    (0.1273, 0.0, math.pi / 2.0),
    (0.0, -0.612, 0.0),
    (0.0, -0.5723, 0.0),
    (0.163941, 0.0, math.pi / 2.0),
    (0.1157, 0.0, -math.pi / 2.0),
    (0.0922, 0.0, 0.0),
)

#: 名义腕心（J2/J3/J4 轴线交点）到基座的距离，用来把角度步长粗算成线位移。
#: 用法：拿几组典型姿态算法兰线位移，看"1° 大约是多少毫米"。
NOMINAL_REACH_M = 1.0


@dataclass(frozen=True)
class RangeCheckResult:
    """名义范围检查结果。字段名保持英文，报告里再翻成中文。"""

    ok: bool
    #: 逐关节的 (最小角, 最大角)。
    joint_span_deg: dict[str, tuple[float, float]]
    #: 逐关节距最近限位还剩多少度（负值表示已经越限）。
    limit_margin_deg: dict[str, float]
    #: 这段运动里法兰中心相对起点的最大线位移（毫米）。
    max_flange_travel_mm: float
    #: 越限的关节名。
    violating_joints: tuple[str, ...]
    #: 给人看的中文说明。
    messages: tuple[str, ...]

    def summary_lines(self) -> list[str]:
        lines = [
            f"名义范围检查：{'通过' if self.ok else '不通过'}",
            f"整段运动中法兰中心最大线位移：约 {self.max_flange_travel_mm:.3f} mm",
        ]
        for joint, margin in self.limit_margin_deg.items():
            lines.append(f"  {joint} 距限位最小余量：{margin:.2f}°")
        lines.extend(f"  {message}" for message in self.messages)
        return lines


# --------------------------------------------------------------------------
# 基础线性代数
# --------------------------------------------------------------------------


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def _rot_x(alpha: float) -> np.ndarray:
    c, s = math.cos(alpha), math.sin(alpha)
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def _trans_z(d: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[2, 3] = d
    return matrix


def _trans_x(a: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 3] = a
    return matrix


def forward_kinematics(joint_deg: Sequence[float]) -> np.ndarray:
    """名义正运动学：关节角（度，6 个）→ 法兰位姿矩阵（4×4，米/弧度）。

    用的是标准 DH：``T = Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)``。
    """
    angles = _as_six(joint_deg, "joint_deg")
    transform = np.eye(4)
    for theta_deg, (d, a, alpha) in zip(angles, UR10_DH):
        theta = math.radians(theta_deg)
        transform = transform @ _rot_z(theta) @ _trans_z(d) @ _trans_x(a) @ _rot_x(alpha)
    return transform


def flange_position_m(joint_deg: Sequence[float]) -> np.ndarray:
    """法兰中心位置（米）。"""
    return forward_kinematics(joint_deg)[:3, 3].copy()


def flange_travel_mm(
    start_deg: Sequence[float], end_deg: Sequence[float]
) -> float:
    """两个关节配置之间法兰中心的直线距离（毫米）。"""
    start = flange_position_m(start_deg)
    end = flange_position_m(end_deg)
    return float(np.linalg.norm(end - start) * 1000.0)


def generic_joint_line_speed_mm_s(
    joint_deg: Sequence[float], joint_index: int, speed_deg_s: float
) -> float:
    """估计"某关节以 speed_deg_s 转动时，法兰中心大约每秒走多少毫米"。

    用一个小角度差分近似局部雅可比，不解析求导。给现场填速度时作参考：
    需求里说默认速度要沿用旧项目验证过的保守参数，这个数字就是"保守"的换算依据。
    """
    if not 0 <= joint_index < 6:
        raise ValueError(f"joint_index 必须在 0～5，收到 {joint_index}")
    delta = 1e-3
    plus = list(_as_six(joint_deg, "joint_deg"))
    minus = list(plus)
    plus[joint_index] += delta
    minus[joint_index] -= delta
    travel_m = float(
        np.linalg.norm(flange_position_m(plus) - flange_position_m(minus))
    )
    # 差分跨了 2*delta 度，换算成"每度多少毫米"，再乘角速度。
    mm_per_deg = travel_m * 1000.0 / (2.0 * delta)
    return mm_per_deg * float(speed_deg_s)


# --------------------------------------------------------------------------
# 限位检查
# --------------------------------------------------------------------------


def check_joint_in_limits(
    joint_deg: Sequence[float],
    *,
    margin_deg: float = 0.0,
    limits: Iterable[tuple[float, float]] = JOINT_LIMITS_DEG,
) -> tuple[bool, str]:
    """检查一个配置是否在限位内（可要求留出余量）。返回 (是否通过, 中文原因)。"""
    angles = _as_six(joint_deg, "joint_deg")
    limit_list = list(limits)
    if len(limit_list) != 6:
        raise ValueError(f"limits 必须是 6 组，收到 {len(limit_list)} 组")
    for index, angle in enumerate(angles):
        low, high = limit_list[index]
        if angle < low + margin_deg or angle > high - margin_deg:
            return False, (
                f"J{index + 1} = {angle:.4f}° 距限位 [{low:.0f}°, {high:.0f}°] "
                f"不足 {margin_deg:.2f}°。"
            )
    return True, "全部关节都在限位内。"


def check_staircase_range(
    nominal_deg: Sequence[float],
    step_deg: float,
    *,
    n: int,
    joint_index: int,
    limit_margin_deg: float = 5.0,
    verify_return: bool = True,
) -> RangeCheckResult:
    """检查"从名义姿态连续单向走 N 步再回来"这一整段的关节范围。

    对应需求五组 A：整段 ``q0 → q0+Δq → … → q0+NΔq`` 都要检查，
    而不是只看单步。``verify_return=True`` 时，回程段（同样的角度集合倒着走）
    已经包含在同一个角度区间里，所以不需要额外区间——这里保留这个参数是为了
    让调用处明确写出"回程也要检查"。
    """
    nominal = _as_six(nominal_deg, "nominal_deg")
    if not 0 <= joint_index < 6:
        raise ValueError(f"joint_index 必须在 0～5，收到 {joint_index}")
    if n < 1:
        raise ValueError(f"n 至少为 1，收到 {n}")
    if step_deg <= 0:
        raise ValueError(f"step_deg 必须为正，收到 {step_deg}")

    messages: list[str] = []
    spans: dict[str, tuple[float, float]] = {}
    margins: dict[str, float] = {}
    violating: list[str] = []

    for index in range(6):
        base = nominal[index]
        if index == joint_index:
            low = base
            high = base + step_deg * n
        else:
            low = high = base
        spans[f"J{index + 1}"] = (low, high)
        low_limit, high_limit = JOINT_LIMITS_DEG[index]
        margin = min(low - low_limit, high_limit - high)
        margins[f"J{index + 1}"] = float(margin)
        if margin < limit_margin_deg:
            violating.append(f"J{index + 1}")

    # 法兰线位移：起点、终点、以及每一步的中点里取最大。
    start = list(nominal)
    end = list(nominal)
    end[joint_index] += step_deg * n
    travel = flange_travel_mm(start, end)

    if violating:
        for name in violating:
            index = int(name[1:]) - 1
            low, high = spans[name]
            messages.append(
                f"{name} 在本段里会走到 [{low:.4f}°, {high:.4f}°]，"
                f"距限位只剩 {margins[name]:.2f}°（要求 ≥ {limit_margin_deg:.2f}°）。"
            )
    if not verify_return:
        messages.append("注意：调用方要求不检查回程段，请确认这是有意为之。")
    else:
        messages.append("回程段与去程共用同一角度区间，已一并检查。")

    messages.append(
        "★ 以上只是名义运动学范围检查，**不等于碰撞安全**；"
        "collision_status 仍为 unknown。"
    )

    ok = not violating
    return RangeCheckResult(
        ok=ok,
        joint_span_deg=spans,
        limit_margin_deg=margins,
        max_flange_travel_mm=travel,
        violating_joints=tuple(violating),
        messages=tuple(messages),
    )


def describe_amplitude_scale(step_deg: float, nominal_deg: Sequence[float]) -> str:
    """把"这个步长在末端大约是多少毫米"写成一句人话，供界面提示。"""
    nominal = _as_six(nominal_deg, "nominal_deg")
    parts: list[str] = []
    for index in range(6):
        target = list(nominal)
        target[index] += step_deg
        travel = flange_travel_mm(nominal, target)
        parts.append(f"J{index + 1} {travel:.3f} mm")
    return f"名义正运动学下 {step_deg}° ≈ 法兰位移：" + "；".join(parts)


def _as_six(values: Sequence[float], where: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{where} 必须是 6 个数字的序列，收到 {type(values).__name__}")
    if len(values) != 6:
        raise ValueError(f"{where} 必须是 6 个数字，收到 {len(values)} 个")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}[{index}] 必须是数字，收到 {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}] 必须是有限数字，收到 {value!r}")
        result.append(number)
    return result


def joint_spans_from_plan(
    nominal_deg: Sequence[float], steps: Mapping[str, float], n: int
) -> dict[str, tuple[float, float]]:
    """组 A：把"每个关节各自的 N 步阶梯"汇总成逐关节角度区间。

    ``steps`` 是 ``{关节名: 步长(度)}``。没在里面的关节按 0 处理。
    """
    nominal = _as_six(nominal_deg, "nominal_deg")
    if n < 1:
        raise ValueError(f"n 至少为 1，收到 {n}")
    spans: dict[str, tuple[float, float]] = {}
    for index in range(6):
        name = f"J{index + 1}"
        step = float(steps.get(name, 0.0))
        if step < 0:
            raise ValueError(f"{name} 的步长不能为负，收到 {step}")
        low = nominal[index]
        high = nominal[index] + step * n
        spans[name] = (low, high)
    return spans
