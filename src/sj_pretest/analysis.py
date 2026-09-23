"""离线分析：把"采到的帧 + RTDE 记录"变成"每个关节每个步长的结论"。

这一层只做三件事，对应需求七的三层，**不许混成一句"关节误差与振动"**：

第一层 关节执行（指令 → 实际）
    拿 RTDE 里的 ``command_q``（控制器认下的目标）和 ``actual_q``（实际关节角）
    比。这一层回答"命令发出去了，关节真的转了吗、转了多少、有没有死区/跳变/超调"。
    它**不可能**区分"伺服误差"和"负载变形"——那需要力矩信息，本工具没有。

第二层 端视觉运动（实际 → 画面）
    把棋盘格 88 个内角点压成一个标量（J1–J5 用质心沿该关节局部图像方向的位置投影；
    J6 用绕质心的二维刚体旋转角），再除以灵敏度换成"视觉等效角度"。
    灵敏度优先用"视觉位移 ÷ RTDE 实际关节角变化"，退而用"÷ 指令角变化"——
    用了退化版本一定在报告里标出来。

第三层 动态残差（RTDE 已稳但画面仍在动）
    保持阶段（RTDE 判据已经认为停稳）里画面还动多少。这一层能和静态噪声基线
    直接比，所以"是结构在振还是检测在抖"分得开。

反过来，本工具**不声称**一次实验就能把"关节误差"和"结构振动"严格分离：
第二层的信号里本来就同时含着这两者，只是第三层能给出"其中至少有多少是
在关节停稳之后还在动的"这个下界。

判据（需求七）
-------------
推荐步长 = 三个候选幅度里**最小的、同时满足下面全部条件**的那个：
稳定检测、符号正确、重复大致一致、视觉信号 ≥ 5× 静态噪声、RTDE 有真实响应、
不出画且不接近限位。都不满足就如实说"需要人工填写更大步长"，不做自动外扩。
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import (
    CENTROID_JOINTS,
    JOINT_NAMES,
    ROTATION_JOINTS,
    AppConfig,
    ThresholdConfig,
)
from .kinematics import (
    check_joint_in_limits,
    flange_travel_mm,
    joint_spans_from_plan,
)
from .vision import (
    FrameVision,
    SegmentVision,
    estimate_depth_in_plane,
    expected_image_direction_deg,
)

#: 试验段里各阶段的标签。experiment.py 用同样这几个名字建 PhasePlan，
#: analysis.py 用它们在本段自己的时间轴上切窗口——两边必须一致。
PHASE_PRE = "pre_motion"
PHASE_MOVE = "move"
PHASE_HOLD = "hold"
PHASE_RETURN = "return"
PHASE_POST = "post_motion"


class AnalysisError(RuntimeError):
    """分析层出错。消息中文。"""


# --------------------------------------------------------------------------
# 读 RTDE 记录
# --------------------------------------------------------------------------


def load_robot_states(path: Path) -> list[dict[str, Any]]:
    """读 ``robot_states.csv``，把数值列转成 float（空字符串转 None）。"""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 一个试验的结果
# --------------------------------------------------------------------------


@dataclass
class TrialMetrics:
    """一次统计试验（去程或回程、某个幅度、某个方向、某次重复）的全部结论。"""

    event_id: str
    joint: str
    stage: str
    amplitude_deg: float
    direction: int
    repeat: int
    segment_id: str

    # -- 第一层：指令 → 实际 ---------------------------------------------
    commanded_delta_deg: float
    rtde_command_delta_deg: float | None = None
    rtde_actual_delta_deg: float | None = None
    #: 实际响应 ÷ 指令（1.0 表示完全跟上；远小于 1 表示没动或死区）。
    rtde_response_ratio: float | None = None
    #: 实际角变化的量化台阶（相邻不同值的最小正差），°。
    quantization_deg: float | None = None
    #: 一次采样内跳过多个量化台阶的次数（"跳变"）。
    jump_count: int = 0
    max_jump_deg: float | None = None
    #: 最大超出最终值的量 ÷ 步长（>0 表示有超调）。
    overshoot_ratio: float | None = None
    #: 到位后实际角穿过目标值的次数（振荡计数）。
    oscillation_count: int = 0
    #: 从下发到 RTDE 判据认为停稳的时长（秒）；判据超时则为 None。
    rtde_settle_s: float | None = None
    rtde_settle_ended_by: str | None = None

    # -- 第二层：实际 → 画面 ---------------------------------------------
    vision_dx_px: float | None = None
    vision_dy_px: float | None = None
    vision_proj_px: float | None = None
    vision_equiv_deg: float | None = None
    rotation_deg: float | None = None
    rotation_equiv_deg: float | None = None
    detection_valid_ratio: float = 0.0
    direction_deg_used: float | None = None
    direction_source: str = "theoretical"
    sensitivity_px_per_deg: float | None = None
    sensitivity_source: str = "unknown"
    #: 视觉信号 ÷ 静态噪声。**两条视觉路径都要给这一个数**，但单位随关节不同：
    #: J1–J5 是"质心投影位移 ÷ 同一方向的投影噪声"（无量纲，两边都是像素）；
    #: J6 是"二维转角 ÷ 静态转角噪声"（两边都是度，同样无量纲）。
    #: 设成 None 只有一种情况：静态基线没进分析，或者静态噪声恰为 0。
    snr: float | None = None
    #: 第二层两个窗口各用了多少**有效帧**。留这两个数是为了让"窗口里到底有几帧"
    #: 看得见：离线分析的步长一大，保持窗口（本来就只有保持段的一半长）里可能
    #: 只剩一两帧，这时候报出来的不是均值而是一次抽样——必须先能看见，才谈得上
    #: 判断这个数能不能用（见 thresholds.min_window_frames）。
    pre_frames: int = 0
    steady_frames: int = 0

    # -- 轴向 / 面内分解（需求四） ---------------------------------------
    #: 由棋盘格相似变换的尺度变化估的轴向位移（mm）。见 vision.estimate_depth_in_plane。
    depth_mm: float | None = None
    #: 由质心二维位移 × 现场 mm/px 得到的面内位移（mm）。
    in_plane_mm: float | None = None
    #: depth / in_plane（仅在尺度可分辨时给出）。
    depth_ratio: float | None = None
    #: 用深度**上限**算出的比值——判据用的就是它。
    depth_ratio_upper: float | None = None
    #: 参与分解的有效帧数。
    depth_frames: int = 0
    #: resolved / below_resolution / unavailable（含义见 vision 里的同名常量）。
    depth_confidence: str = ""
    #: 一句话结论（含"无法判断"这类如实表述）。
    depth_note: str = ""

    # -- 第三层：动态残差 -------------------------------------------------
    residual_px: float | None = None
    residual_ratio: float | None = None
    residual_decaying: bool | None = None
    residual_deg: float | None = None
    vision_moves_after_settle: bool | None = None

    static_noise_px: float | None = None
    static_noise_deg: float | None = None

    issues: list[str] = field(default_factory=list)

    @property
    def signed_proj_px(self) -> float | None:
        """沿"该方向为正"的投影。回程（direction=-1）时投影已按方向翻转。"""
        return self.vision_proj_px

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "joint": self.joint,
            "stage": self.stage,
            "amplitude_deg": self.amplitude_deg,
            "direction": self.direction,
            "direction_sign": "+" if self.direction > 0 else "-",
            "repeat": self.repeat,
            "segment_id": self.segment_id,
            "commanded_delta_deg": _r(self.commanded_delta_deg, 6),
            "rtde_command_delta_deg": _r(self.rtde_command_delta_deg, 6),
            "rtde_actual_delta_deg": _r(self.rtde_actual_delta_deg, 6),
            "rtde_response_ratio": _r(self.rtde_response_ratio, 4),
            "quantization_deg": _r(self.quantization_deg, 6),
            "jump_count": self.jump_count,
            "max_jump_deg": _r(self.max_jump_deg, 6),
            "overshoot_ratio": _r(self.overshoot_ratio, 4),
            "oscillation_count": self.oscillation_count,
            "rtde_settle_s": _r(self.rtde_settle_s, 4),
            "rtde_settle_ended_by": self.rtde_settle_ended_by or "",
            "vision_dx_px": _r(self.vision_dx_px, 5),
            "vision_dy_px": _r(self.vision_dy_px, 5),
            "vision_proj_px": _r(self.vision_proj_px, 5),
            "vision_equiv_deg": _r(self.vision_equiv_deg, 6),
            "rotation_deg": _r(self.rotation_deg, 6),
            "rotation_equiv_deg": _r(self.rotation_equiv_deg, 6),
            "detection_valid_ratio": _r(self.detection_valid_ratio, 4),
            "direction_deg_used": _r(self.direction_deg_used, 3),
            "direction_source": self.direction_source,
            "sensitivity_px_per_deg": _r(self.sensitivity_px_per_deg, 4),
            "sensitivity_source": self.sensitivity_source,
            "snr": _r(self.snr, 4),
            "pre_frames": self.pre_frames,
            "steady_frames": self.steady_frames,
            "depth_mm": _r(self.depth_mm, 5),
            "in_plane_mm": _r(self.in_plane_mm, 5),
            "depth_ratio": _r(self.depth_ratio, 4),
            "depth_ratio_upper": _r(self.depth_ratio_upper, 4),
            "depth_frames": self.depth_frames,
            "depth_confidence": self.depth_confidence,
            "depth_note": self.depth_note,
            "residual_px": _r(self.residual_px, 5),
            "residual_deg": _r(self.residual_deg, 6),
            "residual_ratio": _r(self.residual_ratio, 4),
            "residual_decaying": "" if self.residual_decaying is None else int(self.residual_decaying),
            "vision_moves_after_settle": (
                "" if self.vision_moves_after_settle is None else int(self.vision_moves_after_settle)
            ),
            "static_noise_px": _r(self.static_noise_px, 5),
            "static_noise_deg": _r(self.static_noise_deg, 6),
            "issues": "；".join(self.issues),
        }


def _r(value: Any, digits: int) -> Any:
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


# --------------------------------------------------------------------------
# 静态噪声基线
# --------------------------------------------------------------------------


@dataclass
class StaticNoise:
    """静态基线的噪声水平。这是后面所有"信号有没有意义"的尺子。"""

    segment_id: str
    frames: int
    valid_ratio: float
    std_x_px: float
    std_y_px: float
    std_radial_px: float
    std_rotation_deg: float
    #: 首尾各一成帧的均值之差，用来识别"其实框架在动"。
    #: ★ 不要用"最大最小值之差"当漂移：那量的是纯噪声的极差（约 6σ），
    #: 哪怕真的完全静止也会给出一个远大于 σ 的数，然后误报"有人在碰设备"。
    drift_x_px: float
    drift_y_px: float
    drift_rotation_deg: float
    #: 极差（仅供参考，不参与判断）。
    span_x_px: float
    span_y_px: float
    span_rotation_deg: float
    #: ★ **静态基线实际录了多久**（采集口径）。报告里"基线多少秒"说的就是它。
    #: 早先这里装的其实是"离线处理算了多久"，于是同一段 3 s 的静止采集
    #: 会因为算力不同被写成 14.5 s——单位没错、量纲没错，但量错了东西，
    #: 而且写进报告时没人看得出来。
    seconds: float
    #: 这一段 RAW 里**一共**多少帧（采集帧数）。stride > 1 时大于 ``frames``，
    #: 报告要把两个数都写出来，"统计了多少帧"和"录了多少帧"是两回事。
    captured_frames: int = 0
    #: 离线处理这段花掉的 wall clock 秒数。**不是实验量**，仅供现场估时间。
    process_seconds: float = 0.0

    def noise_along_px(self, direction_deg: float) -> float:
        """把二维噪声投到某个图像方向上。

        投影的方差 = σx²cos²θ + σy²sin²θ（默认 x/y 独立）。
        沿关节运动方向的那一份噪声才是和信号同一量纲的尺子。
        """
        theta = math.radians(float(direction_deg))
        variance = (self.std_x_px * math.cos(theta)) ** 2 + (
            self.std_y_px * math.sin(theta)
        ) ** 2
        return float(math.sqrt(variance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "frames": int(self.frames),
            "valid_ratio": float(self.valid_ratio),
            # seconds = 这一段**录了多久**（采集口径）；process_seconds = 处理耗时。
            # 两个都写出来，"基线时长"和"算力开销"以后不会再有第二种读法。
            "seconds": float(self.seconds),
            "captured_frames": int(self.captured_frames),
            "process_seconds": float(self.process_seconds),
            "std_x_px": float(self.std_x_px),
            "std_y_px": float(self.std_y_px),
            "std_radial_px": float(self.std_radial_px),
            "std_rotation_deg": float(self.std_rotation_deg),
            "drift_x_px": float(self.drift_x_px),
            "drift_y_px": float(self.drift_y_px),
            "drift_rotation_deg": float(self.drift_rotation_deg),
            "span_x_px": float(self.span_x_px),
            "span_y_px": float(self.span_y_px),
            "span_rotation_deg": float(self.span_rotation_deg),
        }

    def summary_lines(self) -> list[str]:
        # 采集帧数 / 统计帧数 / 采集时长 三件事分开说：stride > 1 时
        # "录了 397 帧、统计了 50 帧"和"录了 50 帧"是完全不同的实验。
        recorded = (
            f"{self.captured_frames} 帧 / {self.seconds:.2f} s"
            if self.captured_frames
            else f"{self.frames} 帧 / {self.seconds:.2f} s"
        )
        stats = (
            f"参与统计 {self.frames} 帧"
            + ("" if self.captured_frames in (0, self.frames) else "（按处理步长抽样）")
            + f"，有效角点帧 {self.valid_ratio:.1%}"
        )
        return [
            f"静态基线 {self.segment_id}：本段录了 {recorded}；{stats}",
            (
                f"  质心噪声 σx={self.std_x_px:.4f} px，σy={self.std_y_px:.4f} px，"
                f"径向 {self.std_radial_px:.4f} px"
            ),
            f"  二维转角噪声 σ={self.std_rotation_deg:.5f}°",
            (
                f"  首尾漂移（首/尾各一成帧的均值之差）："
                f"x {self.drift_x_px:+.4f} px，y {self.drift_y_px:+.4f} px，"
                f"转角 {self.drift_rotation_deg:+.5f}°"
            ),
            (
                f"  整段极差（仅供参考）：x {self.span_x_px:.4f} px，"
                f"y {self.span_y_px:.4f} px，转角 {self.span_rotation_deg:.5f}°"
            ),
        ]


def analyze_static(segment: SegmentVision) -> StaticNoise:
    """从静态基线段算噪声。"""
    # 注意返回顺序：centroids() 给的是 (时间, x, y)，rotations() 给的是 (时间, 转角)。
    # 这两个数组都只含有效帧。把时间当成坐标用会让"噪声"变成一个纯粹的时间跨度，
    # 数值上还看着挺合理——所以这里按名字拆开，不用位置猜。
    _cx_time, cx, cy = segment.centroids()
    _rot_time, angles = segment.rotations()
    if cx.size < 2:
        raise AnalysisError(
            f"静态基线 {segment.segment_id} 几乎没有检到角点，无法给出噪声基线。"
            "请检查现场光照、棋盘格是否在视野内、以及是否真的静止。"
        )
    dx = cx - float(np.mean(cx))
    dy = cy - float(np.mean(cy))
    radial = np.hypot(dx, dy)

    def decile_drift(values: np.ndarray) -> float:
        """首/尾各一成帧的均值之差。每边平均了约 n/10 帧，噪声被压掉约 √(n/10) 倍，
        所以这个量能真的反映"框架有没有整体移动"，而不会被逐帧抖动淹没。"""
        if values.size < 20:
            return 0.0
        chunk = max(1, values.size // 10)
        return float(np.mean(values[-chunk:]) - np.mean(values[:chunk]))

    return StaticNoise(
        segment_id=segment.segment_id,
        frames=segment.frame_count,
        valid_ratio=segment.valid_ratio,
        std_x_px=float(np.std(dx, ddof=1)) if dx.size > 1 else 0.0,
        std_y_px=float(np.std(dy, ddof=1)) if dy.size > 1 else 0.0,
        std_radial_px=float(np.std(radial, ddof=1)) if radial.size > 1 else 0.0,
        std_rotation_deg=(
            float(np.std(angles - float(np.mean(angles)), ddof=1))
            if angles.size > 1
            else 0.0
        ),
        drift_x_px=decile_drift(cx),
        drift_y_px=decile_drift(cy),
        drift_rotation_deg=decile_drift(angles),
        span_x_px=float(np.max(cx) - np.min(cx)) if cx.size else 0.0,
        span_y_px=float(np.max(cy) - np.min(cy)) if cy.size else 0.0,
        span_rotation_deg=(
            float(np.max(angles) - np.min(angles)) if angles.size else 0.0
        ),
        # ★ 报告里的"基线多少秒"必须是**采集时长**，不是这段离线算了多久。
        # 采集时长取不到（老段目录没写 content_seconds）时，退回用本段分析时间轴
        # 的实际跨度——那也是采集侧的时间，量纲和物理含义都对；
        # 绝不退回 ``segment.seconds``（那是 wall clock，换台机器就变）。
        seconds=_captured_seconds(segment),
        captured_frames=int(segment.captured_frames),
        process_seconds=float(segment.seconds),
    )


def _captured_seconds(segment: "SegmentVision") -> float:
    """这一段 RAW 实际录了多久（优先采集层写的 ``content_seconds``）。"""
    if segment.captured_seconds > 0:
        return float(segment.captured_seconds)
    times = segment.times()
    if times.size >= 2:
        return float(times[-1] - times[0])
    return 0.0


# --------------------------------------------------------------------------
# 窗口与提取
# --------------------------------------------------------------------------


def _window_frames(
    segment: SegmentVision, start_s: float, end_s: float, *, margin_s: float = 0.05
) -> list[FrameVision]:
    """取窗口内**有效**的帧，两端各留一点余量避开相位边界那一两帧。"""
    low = float(start_s) + float(margin_s)
    high = float(end_s) - float(margin_s)
    if high <= low:
        low, high = float(start_s), float(end_s)
    return [f for f in segment.window(low, high) if f.valid]


def _mean_dx_dy(frames: Sequence[FrameVision]) -> tuple[float, float] | None:
    xs = [f.shift_x_px for f in frames if f.shift_x_px is not None]
    ys = [f.shift_y_px for f in frames if f.shift_y_px is not None]
    if not xs or not ys:
        return None
    return float(statistics.fmean(xs)), float(statistics.fmean(ys))


def _mean_rotation(frames: Sequence[FrameVision]) -> float | None:
    values = [f.rotation_deg for f in frames if f.rotation_deg is not None]
    if not values:
        return None
    return float(statistics.fmean(values))


def _std_rotation(frames: Sequence[FrameVision]) -> float | None:
    values = [f.rotation_deg for f in frames if f.rotation_deg is not None]
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


# --------------------------------------------------------------------------
# 关节局部图像方向
# --------------------------------------------------------------------------


@dataclass
class JointDirection:
    """某个关节在图像里的局部运动方向（度和来源）。"""

    joint: str
    direction_deg: float | None
    #: "measured_0.2deg" / "theoretical" / "unavailable"
    source: str
    note: str
    #: 正方向、负方向各自的单位向量（用于分别投影，避免把回程间隙混进去）。
    plus_unit: tuple[float, float] | None = None
    minus_unit: tuple[float, float] | None = None
    #: 用于说明"这条路有没有被分辨出来"的裕量。
    separation_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "direction_deg": self.direction_deg,
            "source": self.source,
            "note": self.note,
            "plus_unit": list(self.plus_unit) if self.plus_unit else None,
            "minus_unit": list(self.minus_unit) if self.minus_unit else None,
            "separation_deg": self.separation_deg,
        }


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return None
    return (dx / norm, dy / norm)


def _angle_between(u: tuple[float, float], v: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(dot))


def estimate_joint_direction(
    joint: str,
    in_plane: Sequence[tuple[float, float]],
    out_of_plane: Sequence[tuple[float, float]],
    *,
    config: AppConfig,
    direction_tolerance_deg: float,
) -> JointDirection:
    """从 0.2° 的位移向量里找出这个关节的图像局部运动方向。

    §四 的要求：方向要从 **0.2° 那一档**的数据里量出来，并且正负分别标定——
    正负合并成一个方向会把回程间隙（齿轮反向误差）搅进来，而间隙恰恰是
    小步长实验最想看清楚的东西之一。
    """
    index = JOINT_NAMES.index(joint)
    nominal = config.robot.nominal_joint_deg
    theory_deg, theory_note = expected_image_direction_deg(
        index, nominal, config=config
    )

    plus_unit = None
    minus_unit = None
    if joint in CENTROID_JOINTS:
        # 正负分开求平均位移：in_plane 是"朝正方向"的那些试验。
        plus_pairs = [p for p in in_plane if p is not None]
        minus_pairs = [p for p in out_of_plane if p is not None]
        if plus_pairs:
            plus_unit = _unit(
                float(statistics.fmean([p[0] for p in plus_pairs])),
                float(statistics.fmean([p[1] for p in plus_pairs])),
            )
        if minus_pairs:
            minus_unit = _unit(
                float(statistics.fmean([p[0] for p in minus_pairs])),
                float(statistics.fmean([p[1] for p in minus_pairs])),
            )

        if plus_unit is not None and minus_unit is not None:
            separation = _angle_between(plus_unit, minus_unit)
            # 正负应该大致反向（夹角接近 180°）。差太多说明这一档的信号
            # 还淹在噪声里，方向不可信——那时退回理论方向。
            if abs(180.0 - separation) <= direction_tolerance_deg:
                measured = math.degrees(
                    math.atan2(plus_unit[1], plus_unit[0]) % (2 * math.pi)
                )
                return JointDirection(
                    joint=joint,
                    direction_deg=measured,
                    source="measured_0.2deg",
                    note=(
                        f"由 0.2° 档正负两组位移分别标定，正负夹角 {separation:.1f}°"
                        f"（越接近 180° 越可信）；理论预测 {theory_deg:.1f}°。"
                    ),
                    plus_unit=plus_unit,
                    minus_unit=minus_unit,
                    separation_deg=separation,
                )
            return JointDirection(
                joint=joint,
                direction_deg=theory_deg,
                source="theoretical",
                note=(
                    f"0.2° 档正负位移的夹角只有 {separation:.1f}°，与反向差"
                    f"{abs(180.0 - separation):.1f}°，方向量大过噪声，退用理论方向。"
                    f"{theory_note}"
                ),
                plus_unit=plus_unit,
                minus_unit=minus_unit,
                separation_deg=separation,
            )
        return JointDirection(
            joint=joint,
            direction_deg=theory_deg,
            source="theoretical",
            note=f"0.2° 档没有量到完整位移，退用理论方向。{theory_note}",
            plus_unit=plus_unit,
            minus_unit=minus_unit,
        )

    # J6：不用平移，靠旋转。
    return JointDirection(
        joint=joint,
        direction_deg=0.0,
        source="rotation_only",
        note=(
            "J6 走二维刚体旋转路径（88 角点绕质心 Kabsch），不使用质心平移，"
            "所以这里的方向只作为“不得使用”的标记。"
        ),
    )


# --------------------------------------------------------------------------
# 单个试验的分析
# --------------------------------------------------------------------------


def analyze_trial(
    *,
    trial: Mapping[str, Any],
    segment: SegmentVision,
    rtde_rows: Sequence[Mapping[str, Any]],
    static_noise: StaticNoise | None,
    direction: JointDirection,
    config: AppConfig,
) -> TrialMetrics:
    """把一个统计试验的 RTDE 记录和视觉记录算成 :class:`TrialMetrics`。

    ``trial`` 里应有的键：``event_id`` / ``joint`` / ``stage`` /
    ``amplitude_deg`` / ``direction`` / ``repeat`` / ``commanded_delta_deg``。
    """
    thresholds = config.thresholds
    joint = str(trial["joint"])
    index = JOINT_NAMES.index(joint)
    stage = str(trial.get("stage", ""))
    direction_sign = 1 if float(trial.get("direction", 1)) >= 0 else -1

    metrics = TrialMetrics(
        event_id=str(trial["event_id"]),
        joint=joint,
        stage=stage,
        amplitude_deg=float(trial.get("amplitude_deg", 0.0)),
        direction=direction_sign,
        repeat=int(trial.get("repeat", 0)),
        segment_id=segment.segment_id,
        commanded_delta_deg=float(trial.get("commanded_delta_deg", 0.0)),
        detection_valid_ratio=segment.valid_ratio,
    )

    # -- 第一层：指令 → 实际 ---------------------------------------------
    _fill_joint_layer(metrics, rtde_rows, segment, index, thresholds)

    # -- 第二层：实际 → 画面 ---------------------------------------------
    _fill_vision_layer(metrics, segment, direction, static_noise)

    # -- 轴向/面内分解（需求四） -----------------------------------------
    _fill_depth_layer(metrics, segment, config=config)

    # -- 第三层：动态残差 -------------------------------------------------
    _fill_residual_layer(metrics, segment, direction, config)

    _collect_issues(metrics, segment, thresholds)
    return metrics


def _fill_depth_layer(
    metrics: TrialMetrics, segment: SegmentVision, *, config: AppConfig
) -> None:
    """把这一段分解成"轴向位移 / 面内位移"，并如实标出置信状态。

    窗口口径与第二层一致：**保持窗口的后半段**当信号（前半段还带着到位余振），
    运动前 + 运动后两段静止帧当噪声池（同时也用来估尺度噪声）。
    这样"深度比"和"位移"是同一段画面上量出来的，不会互相打架。
    """
    hold_window = _phase_window(segment, PHASE_HOLD)
    if hold_window is None:
        metrics.depth_note = "缺少保持段相位信息，无法分解"
        return
    hold_start, hold_end = hold_window
    steady_start = hold_start + 0.5 * (hold_end - hold_start)
    signal = _window_frames(segment, steady_start, hold_end)
    noise: list[FrameVision] = []
    for phase in (PHASE_PRE, PHASE_POST):
        window = _phase_window(segment, phase)
        if window is not None:
            noise.extend(_window_frames(segment, *window))

    estimate = estimate_depth_in_plane(
        signal,
        config=config,
        noise_frames=noise,
        # ★ J6 的面内位移要加上"转动扫过的位移"（见 vision.in_plane_px_of_frame）：
        # 棋盘格可能偏心装，只看质心平移会把它的面内运动量小看甚至量成 0，
        # 那样深度/面内比值就会被抬高，一次纯转动可能被误判成"轴向偏大"。
        rotation_joint=str(metrics.joint) in set(config.vision.rotation_joints),
        # ★ 基准是几帧的平均进了可分辨下限的公式，复算时必须原样带过来。
        reference_frames=int(getattr(segment, "reference_frames", 1) or 1),
    )
    metrics.depth_mm = estimate.depth_mm
    metrics.in_plane_mm = estimate.in_plane_mm
    metrics.depth_ratio = estimate.ratio
    metrics.depth_ratio_upper = estimate.ratio_upper
    metrics.depth_frames = estimate.frames
    metrics.depth_confidence = estimate.confidence
    metrics.depth_note = estimate.judgement_text


def check_segment_analyzable(
    segment: SegmentVision, *, config: AppConfig, kind: str = "", joint: str = ""
) -> tuple[bool, str]:
    """删 RAW 之前的最后一道：**本段的基本分析真的跑得完吗**。

    为什么不能只看"有效帧比例"：比例高不代表算得出来。一个典型的坏情况是
    "保持段一帧都没有"——valid_ratio 仍然可能很高（运动前/运动后都是静止帧），
    但第二层的信号窗口是空的，第三层的残差也没有基准，这一段事后就是废数据，
    而那时 RAW 已经删了。所以这里按**分析层真正用的那几件事**逐件试跑一遍：

    1. 相位表里声明的每个阶段，窗口内都要有 ≥ ``thresholds.min_window_frames`` 帧；
    2. 保持窗口的后半段（信号窗）要够；
    3. 运动前 + 运动后（噪声池）要够——没有它就没有"信噪比"这把尺子；
    4. ``estimate_depth_in_plane`` 要能给出结论（不能是 ``unavailable``）；
    5. 质心与二维转角至少有一个算得出来（否则第二层无从下手）。

    静态基线段走另一条路：它的"基本分析"就是 ``analyze_static``，
    所以这里直接调它一次，调得通就算过。

    ``joint`` 由调用方给出（段自己不带关节名——``SegmentVision`` 是视觉层类型，
    只管像素）。它只影响第 4 步的面内位移口径：J6 要把转动扫过的位移算进去。

    返回 ``(能不能分析, 中文原因)``。**失败一律保留 RAW**，由调用方暂停。
    """
    thresholds = config.thresholds
    min_frames = int(thresholds.min_window_frames)
    if str(kind) == "static" or not getattr(segment, "phases", None):
        try:
            noise = analyze_static(segment)
        except Exception as exc:
            return False, f"静态基线的基本分析跑不完：{exc}"
        if noise.frames <= 0 or not np.isfinite(noise.std_radial_px):
            return False, "静态基线算不出噪声水平"
        return True, "静态基线分析通过"

    # 1) 声明了的每个阶段都要有足够有效帧。
    for entry in segment.phases:
        label = str(entry.get("label") or "")
        if not label:
            return False, "相位表里有条目没有 label"
        try:
            start_s, end_s = float(entry["start_s"]), float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            return False, f"相位 {label} 缺 start_s/end_s"
        count = len(_window_frames(segment, start_s, end_s))
        if count < min_frames:
            return False, (
                f"阶段「{label}」只有 {count} 帧有效帧（要求 ≥ {min_frames}）："
                "这个阶段的分析做不了，保留 RAW"
            )

    hold_window = _phase_window(segment, PHASE_HOLD)
    if hold_window is None:
        return False, "相位表里没有保持段（hold），第二层分析没有信号窗"
    hold_start, hold_end = hold_window
    steady_start = hold_start + 0.5 * (hold_end - hold_start)
    signal = _window_frames(segment, steady_start, hold_end)
    if len(signal) < min_frames:
        return False, (
            f"保持段后半（信号窗）只有 {len(signal)} 帧有效帧"
            f"（要求 ≥ {min_frames}）"
        )

    noise: list[FrameVision] = []
    for phase in (PHASE_PRE, PHASE_POST):
        window = _phase_window(segment, phase)
        if window is not None:
            noise.extend(_window_frames(segment, *window))
    if len(noise) < min_frames:
        return False, (
            f"运动前 + 运动后的静止帧只有 {len(noise)} 帧"
            f"（要求 ≥ {min_frames}）：没有噪声池就算不出信噪比和尺度噪声"
        )

    estimate = estimate_depth_in_plane(
        signal,
        config=config,
        noise_frames=noise,
        rotation_joint=str(joint) in set(config.vision.rotation_joints),
        reference_frames=int(getattr(segment, "reference_frames", 1) or 1),
    )
    if estimate.confidence == "unavailable":
        return False, f"轴向/面内分解算不出来：{estimate.note}"
    if _mean_dx_dy(signal) is None and _mean_rotation(signal) is None:
        return False, "保持段既算不出质心也算不出转角，第二层分析无从下手"
    return True, (
        f"本段基本分析通过（信号 {len(signal)} 帧、噪声池 {len(noise)} 帧、"
        f"轴向/面内置信状态 {estimate.confidence}）"
    )


def _phase_window(segment: SegmentVision, phase: str) -> tuple[float, float] | None:
    """从采集元数据里取某个阶段在本段时间轴上的起止。

    相位信息存在 ``capture_metadata.json`` 的 ``phases`` 里（采集层写的），
    ``SegmentVision`` 把它带过来了，所以这里不需要再去读一遍文件。
    """
    phases = getattr(segment, "phases", None) or []
    for entry in phases:
        if str(entry.get("label")) == phase:
            return (float(entry["start_s"]), float(entry["end_s"]))
    return None


def _fill_joint_layer(
    metrics: TrialMetrics,
    rows: Sequence[Mapping[str, Any]],
    segment: SegmentVision,
    index: int,
    thresholds: ThresholdConfig,
) -> None:
    """第一层：拿 RTDE 的指令角和实际角说清楚"关节到底动了多少"。"""
    key_cmd = f"command_q_{JOINT_NAMES[index]}_deg"
    key_act = f"actual_q_{JOINT_NAMES[index]}_deg"

    pre_window = _phase_window(segment, PHASE_PRE)
    hold_window = _phase_window(segment, PHASE_HOLD)
    if not rows or pre_window is None or hold_window is None:
        metrics.issues.append("缺少 RTDE 记录或相位信息，第一层无法计算。")
        return

    # 相位时间是"本段的时间轴"（从段首帧起算），RTDE 记录是 host_ns。
    # 用本段首帧的 host_ns 把两者对齐——这是唯一一个能对齐的锚点，
    # 因为相位边界本身就是按帧时间戳量出来的。
    base_ns = int(segment.frames[0].host_ns) if segment.frames else None
    if base_ns is None:
        metrics.issues.append("这一段一帧都没有，无法对齐 RTDE 与画面时间。")
        return

    def in_window(row: Mapping[str, Any], window: tuple[float, float]) -> bool:
        value = row.get("host_ns")
        if value is None:
            return False
        t = (float(value) - base_ns) / 1e9
        return window[0] <= t <= window[1]

    pre_rows = [r for r in rows if in_window(r, pre_window)]
    hold_rows = [r for r in rows if in_window(r, hold_window)]
    if not pre_rows or not hold_rows:
        metrics.issues.append(
            "RTDE 记录没有覆盖到运动前/保持阶段，第一层无法计算（记录频率太稀？）。"
        )
        return

    def mean_of(collection: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [float(r[key]) for r in collection if isinstance(r.get(key), float)]
        return float(statistics.fmean(values)) if values else None

    pre_act = mean_of(pre_rows, key_act)
    hold_act = mean_of(hold_rows, key_act)
    pre_cmd = mean_of(pre_rows, key_cmd)
    hold_cmd = mean_of(hold_rows, key_cmd)
    if pre_act is None or hold_act is None:
        metrics.issues.append("RTDE 记录里这一列全是空的，第一层无法计算。")
        return

    metrics.rtde_actual_delta_deg = hold_act - pre_act
    if pre_cmd is not None and hold_cmd is not None:
        metrics.rtde_command_delta_deg = hold_cmd - pre_cmd
    if metrics.commanded_delta_deg:
        metrics.rtde_response_ratio = (
            metrics.rtde_actual_delta_deg / metrics.commanded_delta_deg
        )

    # 量化台阶：只看这一段里不同的实际角取值，取相邻差的最小正值。
    values = sorted({float(r[key_act]) for r in rows if isinstance(r.get(key_act), float)})
    diffs = [b - a for a, b in zip(values, values[1:]) if b - a > 1e-9]
    metrics.quantization_deg = min(diffs) if diffs else None

    # 跳变：一次采样内越过多个量化台阶。
    step = metrics.quantization_deg or 0.0
    if step > 0:
        seq = [float(r[key_act]) for r in rows if isinstance(r.get(key_act), float)]
        jumps = [abs(b - a) for a, b in zip(seq, seq[1:]) if abs(b - a) > 1e-9]
        big = [j for j in jumps if j > 1.5 * step]
        metrics.jump_count = len(big)
        metrics.max_jump_deg = max(jumps) if jumps else None

    # 超调与振荡：都在保持窗口之前的那段里看。
    move_window = _phase_window(segment, PHASE_MOVE)
    if move_window is not None:
        span = (move_window[0], hold_window[1])
        span_rows = [r for r in rows if in_window(r, span)]
        target_delta = metrics.rtde_actual_delta_deg
        if span_rows and target_delta:
            seq = [
                float(r[key_act]) - pre_act
                for r in span_rows
                if isinstance(r.get(key_act), float)
            ]
            if seq:
                # 沿运动方向的超出量
                extreme = max(seq) if target_delta > 0 else min(seq)
                overshoot = extreme - target_delta
                if target_delta > 0:
                    overshoot = max(overshoot, 0.0)
                else:
                    overshoot = max(-overshoot, 0.0)
                metrics.overshoot_ratio = overshoot / abs(target_delta)
                # 振荡计数：到位之后穿过最终值多少次
                crossed = 0
                after = [v for v in seq if (v - target_delta) * (1 if target_delta > 0 else -1) >= 0]
                for a, b in zip(after, after[1:]):
                    if (a - target_delta) * (b - target_delta) < 0:
                        crossed += 1
                metrics.oscillation_count = crossed

    if move_window is not None:
        metrics.rtde_settle_s = max(0.0, hold_window[0] - move_window[0])
    phases = getattr(segment, "phases", None) or []
    for entry in phases:
        # 动作阶段（move/return）的 ended_by 就是"这次等待是被什么结束的"：
        # settled=真停稳了，timeout=到时间还在动，stopped=被中止。
        if str(entry.get("label")) in (PHASE_MOVE, PHASE_RETURN) and entry.get("ended_by"):
            metrics.rtde_settle_ended_by = str(entry["ended_by"])


def _fill_vision_layer(
    metrics: TrialMetrics,
    segment: SegmentVision,
    direction: JointDirection,
    static_noise: StaticNoise | None,
) -> None:
    """第二层：把 88 角点压成一个标量，再换成"视觉等效角度"。"""
    pre_window = _phase_window(segment, PHASE_PRE)
    hold_window = _phase_window(segment, PHASE_HOLD)
    if pre_window is None or hold_window is None:
        metrics.issues.append("缺少相位信息，第二层无法计算。")
        return

    pre_frames = _window_frames(segment, *pre_window)
    # 保持窗口只取**后半段**作为"到位后的稳态"：前半段还带着到位余振，
    # 那部分属于第三层，不该混进"这个步长产生了多大位移"里。
    hold_start, hold_end = hold_window
    steady_start = hold_start + 0.5 * (hold_end - hold_start)
    hold_frames = _window_frames(segment, steady_start, hold_end)
    metrics.pre_frames = len(pre_frames)
    metrics.steady_frames = len(hold_frames)
    if not pre_frames or not hold_frames:
        metrics.issues.append("运动前或保持阶段没有有效角点帧，第二层无法计算。")
        return

    pre_pair = _mean_dx_dy(pre_frames)
    hold_pair = _mean_dx_dy(hold_frames)
    if pre_pair is None or hold_pair is None:
        metrics.issues.append("质心位移这一列是空的，第二层无法计算。")
        return
    dx = hold_pair[0] - pre_pair[0]
    dy = hold_pair[1] - pre_pair[1]
    metrics.vision_dx_px = dx
    metrics.vision_dy_px = dy

    if metrics.joint in ROTATION_JOINTS:
        # J6：二维刚体旋转，绝不用平移路径。
        pre_rot = _mean_rotation(pre_frames)
        hold_rot = _mean_rotation(hold_frames)
        if pre_rot is None or hold_rot is None:
            metrics.issues.append("J6 的二维转角这一列是空的，第二层无法计算。")
            return
        metrics.rotation_deg = hold_rot - pre_rot
        metrics.direction_deg_used = None
        metrics.direction_source = "rotation_only"
        metrics.sensitivity_source = "rotation_deg_per_deg"
        metrics.sensitivity_px_per_deg = None
        if static_noise is not None:
            metrics.static_noise_deg = static_noise.std_rotation_deg
            # J6 的信噪比用转角比静止时的转角噪声（单位都是度）。
            # 不这么写的话，J6 的 trials.csv 里 snr 一列会空着，
            # 看表的人分不清"这条判据不适用于 J6"还是"这一格没算出来"。
            if metrics.static_noise_deg > 0:
                metrics.snr = abs(metrics.rotation_deg) / metrics.static_noise_deg
        return

    if direction.direction_deg is None:
        metrics.issues.append("这个关节的图像方向既量不到、也推不出，第二层无法计算。")
        return

    unit = direction.plus_unit
    if unit is None and direction.direction_deg is not None:
        theta = math.radians(direction.direction_deg)
        unit = (math.cos(theta), math.sin(theta))
    if unit is None:
        metrics.issues.append("这个关节没有可用的图像方向向量，第二层无法计算。")
        return
    # 投影到**正方向**的单位向量上。这样正负两个方向的试验得到的是同一个坐标轴上的
    # 带符号位移，符号本身就有意义——"符号对不对"这条判据才立得住。
    # 正负分开标定只用在"求方向"和"看回程间隙"上（见 AmplitudeSummary.plus_minus_diff）。
    projected = dx * unit[0] + dy * unit[1]
    metrics.vision_proj_px = projected
    metrics.direction_deg_used = math.degrees(math.atan2(unit[1], unit[0])) % 360.0
    metrics.direction_source = direction.source

    if static_noise is not None:
        metrics.static_noise_px = static_noise.noise_along_px(metrics.direction_deg_used)
        if metrics.static_noise_px > 0:
            metrics.snr = abs(projected) / metrics.static_noise_px


def _fill_residual_layer(
    metrics: TrialMetrics,
    segment: SegmentVision,
    direction: JointDirection,
    config: AppConfig,
) -> None:
    """第三层：RTDE 判据已经认为停稳之后，画面还在动多少。"""
    hold_window = _phase_window(segment, PHASE_HOLD)
    if hold_window is None:
        return
    hold_start, hold_end = hold_window
    span = hold_end - hold_start
    if span <= 0.05:
        return
    early = _window_frames(segment, hold_start, hold_start + 0.5 * span, margin_s=0.0)
    late = _window_frames(segment, hold_start + 0.5 * span, hold_end, margin_s=0.0)
    if metrics.joint in ROTATION_JOINTS:
        resid = _std_rotation(early)
        resid_late = _std_rotation(late)
        metrics.residual_deg = resid
        if metrics.static_noise_deg:
            metrics.residual_ratio = (
                None if resid is None else resid / metrics.static_noise_deg
            )
        if resid is not None and resid_late is not None:
            metrics.residual_decaying = resid_late < resid
        metrics.vision_moves_after_settle = bool(
            resid is not None
            and metrics.static_noise_deg is not None
            and resid > metrics.static_noise_deg * config.thresholds.residual_factor
        )
        return

    if direction.direction_deg is None:
        return
    unit = direction.plus_unit
    if unit is None:
        theta = math.radians(direction.direction_deg)
        unit = (math.cos(theta), math.sin(theta))

    def projected_std(frames: Sequence[FrameVision]) -> float | None:
        values = [
            f.shift_x_px * unit[0] + f.shift_y_px * unit[1]
            for f in frames
            if f.shift_x_px is not None and f.shift_y_px is not None
        ]
        if len(values) < 3:
            return None
        return float(statistics.stdev(values))

    resid = projected_std(early)
    resid_late = projected_std(late)
    metrics.residual_px = resid
    if resid is not None and resid_late is not None:
        metrics.residual_decaying = resid_late < resid
    if metrics.static_noise_px:
        metrics.residual_ratio = None if resid is None else resid / metrics.static_noise_px
    metrics.vision_moves_after_settle = bool(
        resid is not None
        and metrics.static_noise_px is not None
        and resid > metrics.static_noise_px * config.thresholds.residual_factor
    )


def _collect_issues(
    metrics: TrialMetrics,
    segment: SegmentVision,
    thresholds: ThresholdConfig,
) -> None:
    if segment.valid_ratio < thresholds.min_valid_frame_ratio:
        metrics.issues.append(
            f"有效角点帧只占 {segment.valid_ratio:.1%}"
            f"（要求 ≥ {thresholds.min_valid_frame_ratio:.0%}）。"
        )
    if metrics.rtde_settle_ended_by == "timeout":
        metrics.issues.append(
            "RTDE 等待停稳超时：到时间还在动，这一条不作为“关节已到位”的证据。"
        )
    if 0 < metrics.steady_frames < thresholds.min_window_frames:
        # 措辞刻意避开“无法计算”和“超时”：这一条是提醒，不是判废。
        # 数已经算出来了，只是它是几次抽样的平均，不该拿来下结论。
        metrics.issues.append(
            f"保持窗口（后半段）只有 {metrics.steady_frames} 帧有效数据，"
            f"少于 {thresholds.min_window_frames} 帧：这里报出的数是一两次抽样的结果，"
            "不是一个可以下结论的均值。数据不用重采——把离线分析的步长调小"
            "（例如 1～2）再分析一遍即可。"
        )
    if metrics.rtde_response_ratio is not None and metrics.rtde_response_ratio < (
        thresholds.min_rtde_response_ratio
    ):
        metrics.issues.append(
            f"RTDE 实际响应只有指令的 {metrics.rtde_response_ratio:.2f} 倍"
            f"（要求 ≥ {thresholds.min_rtde_response_ratio}）。"
        )
    if metrics.commanded_delta_deg:
        if abs(metrics.commanded_delta_deg) > 1e-9 and metrics.vision_proj_px is None and (
            metrics.rotation_deg is None
        ):
            metrics.issues.append("视觉这一路没有算出任何位移。")
    # 轴向分量偏大：写进这一条的 issues 里，人在看 trials.csv 时能一眼找到它
    # （它不参与 no_blocking_problem 那个"致命问题"的判定，由 in_plane_dominant
    #  这条专门的判据负责，见 summarize_amplitudes）。
    if metrics.depth_ratio_upper is not None and (
        metrics.depth_ratio_upper > float(thresholds.max_depth_ratio)
    ):
        metrics.issues.append(
            f"轴向分量偏大：深度/面内 = {metrics.depth_ratio_upper:.3f} > "
            f"{thresholds.max_depth_ratio}（{metrics.depth_note}）"
        )


# --------------------------------------------------------------------------
# 幅值 × 关节 的汇总与推荐
# --------------------------------------------------------------------------


@dataclass
class AmplitudeSummary:
    """某个关节、某个幅度、两个方向合起来的汇总。"""

    joint: str
    amplitude_deg: float
    trials: list[TrialMetrics] = field(default_factory=list)

    #: 符号正确 / 重复一致 / 信号够大 / RTDE 有响应 / 不出问题 —— 逐条判据。
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def mean_abs_proj_px(self) -> float | None:
        values = [
            abs(t.vision_proj_px) for t in self.trials if t.vision_proj_px is not None
        ]
        return float(statistics.fmean(values)) if values else None

    def mean_abs_rotation_deg(self) -> float | None:
        values = [abs(t.rotation_deg) for t in self.trials if t.rotation_deg is not None]
        return float(statistics.fmean(values)) if values else None

    def mean_abs_actual_deg(self) -> float | None:
        values = [
            abs(t.rtde_actual_delta_deg)
            for t in self.trials
            if t.rtde_actual_delta_deg is not None
        ]
        return float(statistics.fmean(values)) if values else None

    def repeat_spread(self, accessor) -> float | None:
        """同一方向两次重复之间的差（取各方向里的最大者）。

        重复之间的"差"比"标准差"更能说明问题：两次就是两次，
        报标准差会让人误以为样本很多。
        """
        worst: float | None = None
        for direction in (1, -1):
            values = [
                accessor(t)
                for t in self.trials
                if t.direction == direction and accessor(t) is not None
            ]
            if len(values) >= 2:
                spread = max(values) - min(values)
                worst = spread if worst is None else max(worst, spread)
        return worst

    def sign_consistency(self, accessor) -> float:
        """符号正确率：与指令方向同号的试验占比。"""
        values = [accessor(t) for t in self.trials]
        pairs = [
            (v, t.direction)
            for v, t in zip(values, self.trials)
            if v is not None and abs(v) > 0
        ]
        if not pairs:
            return 0.0
        good = sum(1 for v, d in pairs if (v > 0) == (d > 0))
        return good / len(pairs)

    def plus_minus_diff(self, accessor) -> float | None:
        """正负两个方向的**位移量**之差（取绝对值）。

        §七 要求给出 "+/- 差异"：同一个关节、同一个幅度，正走和反走产生的
        画面位移量不一样大，通常就是反向间隙（回程间隙）在作怪。这是小步长
        实验最想知道的事情之一，所以单列，不并进平均值里。

        这里比的是**大小**（两边都取绝对值再相减），不是带符号的两个数相减：
        带符号相减的话，完全对称的 ±0.4 px 会给出 0.8 px，读起来像是"差了一大截"，
        而其实两边一模一样。符号对不对由 ``sign_consistency`` 单独判定，
        不该混进这个数里。
        """
        plus = [accessor(t) for t in self.trials if t.direction > 0]
        minus = [accessor(t) for t in self.trials if t.direction < 0]
        plus = [abs(v) for v in plus if v is not None]
        minus = [abs(v) for v in minus if v is not None]
        if not plus or not minus:
            return None
        return abs(statistics.fmean(plus) - statistics.fmean(minus))

    def to_row(self) -> dict[str, Any]:
        signal = (
            (lambda t: t.rotation_deg)
            if self.joint in ROTATION_JOINTS
            else (lambda t: t.vision_proj_px)
        )
        return {
            "joint": self.joint,
            "amplitude_deg": self.amplitude_deg,
            "trials": len(self.trials),
            "mean_abs_proj_px": _r(self.mean_abs_proj_px(), 5),
            "mean_abs_rotation_deg": _r(self.mean_abs_rotation_deg(), 6),
            "mean_abs_actual_deg": _r(self.mean_abs_actual_deg(), 6),
            "plus_minus_diff": _r(self.plus_minus_diff(signal), 5),
            "passes": int(self.passes),
            **{f"check_{k}": int(v) for k, v in self.checks.items()},
            "notes": "；".join(self.notes),
        }


def summarize_amplitudes(
    trials: Sequence[TrialMetrics], config: AppConfig
) -> list[AmplitudeSummary]:
    """按 (关节, 幅度) 汇总，并逐条判定需求七的那几个条件。"""
    thresholds = config.thresholds
    buckets: dict[tuple[str, float], AmplitudeSummary] = {}
    for trial in trials:
        key = (trial.joint, float(trial.amplitude_deg))
        bucket = buckets.get(key)
        if bucket is None:
            bucket = AmplitudeSummary(joint=trial.joint, amplitude_deg=key[1])
            buckets[key] = bucket
        bucket.trials.append(trial)

    summaries: list[AmplitudeSummary] = []
    for key in sorted(buckets, key=lambda k: (JOINT_NAMES.index(k[0]), k[1])):
        bucket = buckets[key]
        rotation_joint = bucket.joint in ROTATION_JOINTS
        signal = (
            (lambda t: t.rotation_deg) if rotation_joint else (lambda t: t.vision_proj_px)
        )
        checks: dict[str, bool] = {}

        # 1) 稳定检测
        min_ratio = min((t.detection_valid_ratio for t in bucket.trials), default=0.0)
        checks["detection_stable"] = min_ratio >= thresholds.min_valid_frame_ratio

        # 2) 符号正确
        consistency = bucket.sign_consistency(signal)
        checks["sign_correct"] = consistency >= thresholds.min_sign_consistency

        # 3) 重复大致一致
        spread = bucket.repeat_spread(signal)
        mean_abs = (
            bucket.mean_abs_rotation_deg() if rotation_joint else bucket.mean_abs_proj_px()
        )
        noise = _noise_level(bucket, rotation_joint)
        if spread is None or mean_abs is None or noise is None:
            checks["repeat_consistent"] = False
        else:
            allowed = max(
                thresholds.max_repeat_relative_spread * mean_abs,
                thresholds.repeat_noise_multiple * noise,
            )
            checks["repeat_consistent"] = spread <= allowed

        # 4) 视觉信号 ≥ 5× 静态噪声
        if mean_abs is None or noise is None or noise <= 0:
            checks["snr_ge_5"] = False
        else:
            checks["snr_ge_5"] = mean_abs >= thresholds.min_snr_vs_static * noise

        # 5) RTDE 有真实响应
        min_response = min(
            (
                t.rtde_response_ratio
                for t in bucket.trials
                if t.rtde_response_ratio is not None
            ),
            default=None,
        )
        mean_actual = bucket.mean_abs_actual_deg()
        checks["rtde_responded"] = bool(
            min_response is not None
            and min_response >= thresholds.min_rtde_response_ratio
            and mean_actual is not None
            and mean_actual >= thresholds.min_rtde_response_deg
        )

        # 6) 不出画 / 不接近限位 / 没有致命问题
        fatal = [
            issue
            for t in bucket.trials
            for issue in t.issues
            if "超时" in issue or "无法计算" in issue
        ]
        checks["no_blocking_problem"] = not fatal

        # 7) 以面内运动为主（需求四）：深度/面内 比值不超过 max_depth_ratio，
        #    而且判据用的是深度的**上限**——分辨不出来时按最坏情况算，不许蒙混过关。
        depth_trials = [t for t in bucket.trials if t.depth_confidence]
        if not depth_trials:
            # 整批都没有轴向/面内数据（缺静态参考帧之类）：这一条**不判**，
            # 但要说清楚为什么不判，并且不把它算进"通过"里当作已经验过。
            bucket.notes.append(
                "轴向/面内分解：本批数据全部没有可用结果（缺静止参考帧或算不出尺度），"
                "这一条**未评估**——不要当成“已确认以面内运动为主”。"
            )
        else:
            worst = None
            for trial in depth_trials:
                if trial.depth_ratio_upper is None:
                    worst = float("inf")
                elif worst is None or trial.depth_ratio_upper > worst:
                    worst = trial.depth_ratio_upper
            checks["in_plane_dominant"] = bool(
                worst is not None and worst <= float(thresholds.max_depth_ratio)
            )
            if worst is not None and worst != float("inf"):
                bucket.notes.append(
                    f"轴向/面内比值（取上限）最大 {worst:.4f}，"
                    f"上限 {thresholds.max_depth_ratio}"
                )
            for trial in depth_trials:
                if trial.depth_note:
                    bucket.notes.append(f"{trial.event_id}：{trial.depth_note}")

        bucket.checks = checks
        if noise is not None and mean_abs is not None:
            bucket.notes.append(
                f"视觉信号 {mean_abs:.4f}，静态噪声 {noise:.4f}，"
                f"比值 {(mean_abs / noise if noise else float('inf')):.2f}"
            )
        if spread is not None:
            bucket.notes.append(f"重复间最大差 {spread:.4f}")
        else:
            bucket.notes.append(
                "重复一致性**无法评估**（这个方向只有 1 次重复）。"
                "本工具不把“无法评估”当成“通过”：要拿到推荐步长，"
                "请把重复次数设成 2 次或以上再跑一遍。"
            )
        bucket.notes.append(f"符号正确率 {consistency:.0%}")
        if min_response is not None:
            bucket.notes.append(f"RTDE 最小响应比 {min_response:.3f}")
        bucket.notes.extend(fatal)
        summaries.append(bucket)
    return summaries


def _noise_level(bucket: AmplitudeSummary, rotation_joint: bool) -> float | None:
    values = []
    for trial in bucket.trials:
        if rotation_joint:
            if trial.static_noise_deg:
                values.append(float(trial.static_noise_deg))
        elif trial.static_noise_px:
            values.append(float(trial.static_noise_px))
    if not values:
        return None
    # 取最大者：判据要过，就得在所有方向上都过。
    return max(values)


# --------------------------------------------------------------------------
# 灵敏度
# --------------------------------------------------------------------------


@dataclass
class Sensitivity:
    """局部灵敏度：图像位移 ÷ 关节角变化。

    §四 要求优先用 **RTDE 实际关节角变化**做分母，因为那才是"画面真的转了多少"的
    因；只有拿不到实际角时才退回到指令角，并且如实标注退化。
    """

    joint: str
    direction: int
    px_per_deg: float | None
    source: str
    samples: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "direction": self.direction,
            "px_per_deg": self.px_per_deg,
            "source": self.source,
            "samples": self.samples,
            "note": self.note,
        }


def estimate_sensitivity(
    trials: Sequence[TrialMetrics], config: AppConfig
) -> dict[tuple[str, int], Sensitivity]:
    """用**最大的那一档**幅度来估灵敏度（信噪比最高的一档）。"""
    results: dict[tuple[str, int], Sensitivity] = {}
    for joint in JOINT_NAMES:
        joint_trials = [t for t in trials if t.joint == joint]
        if not joint_trials:
            continue
        biggest = max(t.amplitude_deg for t in joint_trials)
        for direction in (1, -1):
            subset = [
                t
                for t in joint_trials
                if t.amplitude_deg == biggest and t.direction == direction
            ]
            if not subset:
                continue
            rotation_joint = joint in ROTATION_JOINTS
            signal = (
                (lambda t: t.rotation_deg) if rotation_joint else (lambda t: t.vision_proj_px)
            )
            # 优先用实际角变化做分母
            use_actual = [
                (signal(t), t.rtde_actual_delta_deg)
                for t in subset
                if signal(t) is not None and t.rtde_actual_delta_deg is not None
            ]
            ratios = [
                abs(s) / abs(d) for s, d in use_actual if d and abs(d) > 1e-9
            ]
            if ratios:
                results[(joint, direction)] = Sensitivity(
                    joint=joint,
                    direction=direction,
                    px_per_deg=float(statistics.fmean(ratios)),
                    source=(
                        "rotation_deg_per_deg"
                        if rotation_joint
                        else "px_per_deg_from_actual_q"
                    ),
                    samples=len(ratios),
                    note=(
                        f"用 {biggest}° 档 {len(ratios)} 次试验估计，"
                        "分母是 RTDE 实际关节角变化。"
                        + (
                            "这个数是 °/°（画面二维转角 ÷ 关节角变化），不是 px/°。"
                            if rotation_joint
                            else ""
                        )
                    ),
                )
                continue
            use_command = [
                (signal(t), t.commanded_delta_deg)
                for t in subset
                if signal(t) is not None and t.commanded_delta_deg
            ]
            ratios = [
                abs(s) / abs(d) for s, d in use_command if d and abs(d) > 1e-9
            ]
            results[(joint, direction)] = Sensitivity(
                joint=joint,
                direction=direction,
                px_per_deg=float(statistics.fmean(ratios)) if ratios else None,
                source=(
                    "rotation_deg_per_deg_degraded"
                    if rotation_joint
                    else "px_per_deg_from_command_degraded"
                ),
                samples=len(ratios),
                note=(
                    "**退化估计**：RTDE 实际角拿不到（或全为 0），"
                    f"改用指令角做分母（{len(ratios)} 次）。"
                    "这个数字里混着伺服没跟上的部分，只作参考。"
                    + ("单位是 °/°。" if rotation_joint else "")
                ),
            )
    return results


def apply_sensitivity(
    trials: Sequence[TrialMetrics],
    sensitivities: Mapping[tuple[str, int], Sensitivity],
) -> None:
    """把灵敏度套回每个试验，算出"视觉等效角度"。"""
    for trial in trials:
        key = (trial.joint, trial.direction)
        sensitivity = sensitivities.get(key)
        if sensitivity is None:
            continue
        trial.sensitivity_px_per_deg = sensitivity.px_per_deg
        trial.sensitivity_source = sensitivity.source
        if sensitivity.px_per_deg is None or abs(sensitivity.px_per_deg) < 1e-12:
            continue
        if trial.joint in ROTATION_JOINTS:
            if trial.rotation_deg is not None:
                trial.rotation_equiv_deg = (
                    trial.rotation_deg / sensitivity.px_per_deg
                )
        elif trial.vision_proj_px is not None:
            trial.vision_equiv_deg = trial.vision_proj_px / sensitivity.px_per_deg


# --------------------------------------------------------------------------
# 推荐步长
# --------------------------------------------------------------------------


@dataclass
class StepRecommendation:
    """一个关节的正式实验步长建议。"""

    joint: str
    recommended_deg: float | None
    reason: str
    #: 三个候选幅度各自的通过情况（人可读）。
    detail_lines: list[str] = field(default_factory=list)
    needs_manual_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "recommended_step_deg": self.recommended_deg,
            "reason": self.reason,
            "needs_manual_input": bool(self.needs_manual_input),
            "detail": list(self.detail_lines),
        }


def recommend_steps(
    summaries: Sequence[AmplitudeSummary], config: AppConfig
) -> list[StepRecommendation]:
    """推荐步长 = **最小**的、全部判据都过的那个幅度。"""
    recommendations: list[StepRecommendation] = []
    for joint in JOINT_NAMES:
        rows = sorted(
            [s for s in summaries if s.joint == joint], key=lambda s: s.amplitude_deg
        )
        if not rows:
            # 这个关节这次压根没跑预实验。不能说"没有一档通过判据"——
            # 那是"测了但都不合格"，跟"没测"是两回事，混在一起会让人以为
            # 数据不好，而其实只是配置里没勾这个关节。
            recommendations.append(
                StepRecommendation(
                    joint=joint,
                    recommended_deg=None,
                    reason=(
                        "这个关节这次没有跑预实验（不在 pretest.joints 里），"
                        "没有候选幅度可推荐。要它也有推荐值，请在参数里把它加进"
                        "预实验关节、重新采一遍三档幅度。"
                    ),
                    detail_lines=[],
                    needs_manual_input=True,
                )
            )
            continue
        detail = []
        chosen: float | None = None
        for row in rows:
            failed = [name for name, ok in row.checks.items() if not ok]
            detail.append(
                f"{row.amplitude_deg}°："
                + ("全部判据通过" if not failed else "未通过 " + "、".join(failed))
                + "（" + "；".join(row.notes) + "）"
            )
            if row.passes and chosen is None:
                chosen = row.amplitude_deg
        if chosen is not None:
            reason = (
                f"在候选幅度 {[r.amplitude_deg for r in rows]} 中，"
                f"{chosen}° 是最小的、全部判据都通过的一档。"
            )
        else:
            reason = (
                "三个候选幅度没有任何一档全部通过判据。"
                "**需要人工填写更大步长**——本工具不做自动外扩"
                "（自动加大步长会越过现场已验证的范围）。"
            )
        recommendations.append(
            StepRecommendation(
                joint=joint,
                recommended_deg=chosen,
                reason=reason,
                detail_lines=detail,
                needs_manual_input=chosen is None,
            )
        )
    return recommendations


# --------------------------------------------------------------------------
# 出画 / 限位（第六七八条判据里的最后一条）
# --------------------------------------------------------------------------


@dataclass
class RangeVerdict:
    """正式实验前对整段行程的理论检查结果。"""

    joint: str
    step_deg: float
    levels: int
    span_deg: float
    ok: bool
    lines: list[str]
    limit_margin_deg: float
    max_flange_travel_mm: float
    in_pretest_range: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "step_deg": self.step_deg,
            "levels": self.levels,
            "span_deg": self.span_deg,
            "ok": bool(self.ok),
            "limit_margin_deg": self.limit_margin_deg,
            "max_flange_travel_mm": self.max_flange_travel_mm,
            "in_pretest_range": bool(self.in_pretest_range),
            "note": self.note,
            "lines": list(self.lines),
        }


def check_formal_range(
    joint: str,
    step_deg: float,
    levels: int,
    *,
    config: AppConfig,
    pretest_max_step_deg: float | None = None,
) -> RangeVerdict:
    """正式实验前，按 UR10 CB3 名义运动学检查整段行程。

    检查四件事：关节限位、理论可达的最低/最高位、行程相对预实验范围的倍数、
    以及末端位移量级。**这是理论检查，不是碰撞检查**——碰撞状态一直是
    ``unknown``，报告里必须一直这么写着。
    """
    index = JOINT_NAMES.index(joint)
    limits = config.robot.nominal_joint_deg
    extreme_plus = list(limits)
    extreme_minus = list(limits)
    extreme_plus[index] = limits[index] + step_deg * levels
    extreme_minus[index] = limits[index] - step_deg * levels

    lines: list[str] = []
    ok = True
    margin = float(config.thresholds.joint_limit_margin_deg)
    for label, target in (("正向末端", extreme_plus), ("负向末端", extreme_minus)):
        good, why = check_joint_in_limits(target, margin_deg=margin)
        lines.append(f"{label}：{why}")
        if not good:
            ok = False

    spans = joint_spans_from_plan(
        config.robot.nominal_joint_deg, {joint: float(step_deg)}, int(levels)
    )
    low, high = spans.get(joint, (limits[index], limits[index]))
    travel = max(
        flange_travel_mm(config.robot.nominal_joint_deg, extreme_plus),
        flange_travel_mm(config.robot.nominal_joint_deg, extreme_minus),
    )
    lines.append(
        f"{joint} 名义关节行程 {low:.4f}° → {high:.4f}°"
        f"（{levels} 级 × {step_deg}°，含负向到 {extreme_minus[index]:.4f}°），"
        f"按名义运动学对应末端位移约 {travel:.3f} mm。"
    )
    lines.append(
        "理论检查只说明“按名义 DH 参数算得出来”，**不等于撞不到东西**；"
        "本工具不做碰撞检查，collision_status 永远是 unknown。"
    )

    in_pretest = True
    if pretest_max_step_deg is not None:
        in_pretest = step_deg * levels <= pretest_max_step_deg + 1e-9
        if not in_pretest:
            lines.append(
                f"注意：整段行程 ±{step_deg * levels:.4f}° 超过了预实验里验证过的"
                f"最大幅度 {pretest_max_step_deg}°，属于**超出已验证范围**。"
                "超出不等于不能做，但现场必须格外小心，并保留人工逐步确认。"
            )

    return RangeVerdict(
        joint=joint,
        step_deg=float(step_deg),
        levels=int(levels),
        span_deg=float(step_deg * levels),
        ok=ok,
        lines=lines,
        limit_margin_deg=margin,
        max_flange_travel_mm=travel,
        in_pretest_range=in_pretest,
        note="碰撞状态：unknown。",
    )


# --------------------------------------------------------------------------
# 整批分析
# --------------------------------------------------------------------------


@dataclass
class PretestReport:
    """一次预实验分析的全部产物。"""

    static_noise: StaticNoise | None
    directions: dict[str, JointDirection]
    sensitivities: dict[tuple[str, int], Sensitivity]
    trials: list[TrialMetrics]
    summaries: list[AmplitudeSummary]
    recommendations: list[StepRecommendation]
    warnings: list[str] = field(default_factory=list)
    mode: str = "dry_run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "collision_status": "unknown",
            "static_noise": None if self.static_noise is None else self.static_noise.to_dict(),
            "directions": {k: v.to_dict() for k, v in self.directions.items()},
            "sensitivity": [
                value.to_dict()
                for _, value in sorted(self.sensitivities.items())
            ],
            "trials": [t.to_row() for t in self.trials],
            "amplitudes": [s.to_row() for s in self.summaries],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "warnings": list(self.warnings),
        }

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append("=" * 68)
        lines.append("预实验分析结果（三层分开，不合并成一句话）")
        lines.append("=" * 68)
        if self.static_noise is not None:
            lines.extend(self.static_noise.summary_lines())
        else:
            lines.append("静态基线：缺失（没有可用的静态段）。")
        lines.append("")
        lines.append("关节局部图像方向")
        for joint in JOINT_NAMES:
            entry = self.directions.get(joint)
            if entry is None:
                lines.append(f"  {joint}：未估计")
                continue
            degree = "—" if entry.direction_deg is None else f"{entry.direction_deg:.1f}°"
            lines.append(f"  {joint}：{degree}（{entry.source}）{entry.note}")
        lines.append("")
        lines.append("灵敏度")
        for key in sorted(self.sensitivities, key=lambda k: (JOINT_NAMES.index(k[0]), k[1])):
            entry = self.sensitivities[key]
            value = "—" if entry.px_per_deg is None else f"{entry.px_per_deg:.4f}"
            lines.append(
                f"  {entry.joint} {'+' if entry.direction > 0 else '-'}：{value}"
                f"（{entry.source}，{entry.samples} 次）{entry.note}"
            )
        lines.append("")
        lines.append("逐档判定")
        for summary in self.summaries:
            flag = "通过" if summary.passes else "不通过"
            lines.append(
                f"  {summary.joint} {summary.amplitude_deg}°：[{flag}] "
                + "；".join(summary.notes)
            )
        lines.append("")
        lines.append("推荐正式实验步长")
        for recommendation in self.recommendations:
            if recommendation.recommended_deg is None:
                lines.append(f"  {recommendation.joint}：{recommendation.reason}")
            else:
                lines.append(
                    f"  {recommendation.joint}：{recommendation.recommended_deg}°"
                    f" —— {recommendation.reason}"
                )
        if self.warnings:
            lines.append("")
            lines.append("需要注意")
            for warning in self.warnings:
                lines.append(f"  * {warning}")
        return lines


def analyze_pretest(
    *,
    config: AppConfig,
    segments: Mapping[str, SegmentVision],
    trials: Sequence[Mapping[str, Any]],
    rtde_rows: Sequence[Mapping[str, Any]],
    static_segment_ids: Sequence[str],
    progress: Any = None,
) -> PretestReport:
    """把整批预实验的数据算成一个 :class:`PretestReport`。

    ``segments`` 以 segment_id 为键；``trials`` 是每个统计试验的元信息
    （实验层在跑的时候记下来，见 :mod:`sj_pretest.experiment`）。
    """
    warnings: list[str] = []

    static_noise: StaticNoise | None = None
    for segment_id in static_segment_ids:
        segment = segments.get(segment_id)
        if segment is None:
            continue
        try:
            candidate = analyze_static(segment)
        except AnalysisError as exc:
            warnings.append(str(exc))
            continue
        if static_noise is None or candidate.frames > static_noise.frames:
            static_noise = candidate
    if static_noise is None:
        warnings.append(
            "没有任何可用的静态基线，所有“信号够不够大”的判据都会以静态噪声为尺子——"
            "现在这把尺子没有刻度，相关判据一律按不通过处理。"
        )

    # 先按 (关节, 幅度, 方向) 收集原始位移向量，用来定方向。
    raw: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
    for trial in trials:
        segment = segments.get(str(trial.get("segment_id", "")))
        if segment is None:
            continue
        pre_window = _phase_window(segment, PHASE_PRE)
        hold_window = _phase_window(segment, PHASE_HOLD)
        if pre_window is None or hold_window is None:
            continue
        pre_frames = _window_frames(segment, *pre_window)
        hold_start, hold_end = hold_window
        hold_frames = _window_frames(
            segment, hold_start + 0.5 * (hold_end - hold_start), hold_end
        )
        pre_pair = _mean_dx_dy(pre_frames)
        hold_pair = _mean_dx_dy(hold_frames)
        if pre_pair is None or hold_pair is None:
            continue
        key = (
            str(trial["joint"]),
            float(trial.get("amplitude_deg", 0.0)),
            int(trial.get("direction", 1)),
        )
        raw.setdefault(key, []).append(
            (hold_pair[0] - pre_pair[0], hold_pair[1] - pre_pair[1])
        )

    # 方向：只用 0.2° 那一档（§四 明确指定）。
    directions: dict[str, JointDirection] = {}
    for joint in JOINT_NAMES:
        amplitudes = sorted({key[1] for key in raw if key[0] == joint})
        if not amplitudes:
            theory_deg, theory_note = expected_image_direction_deg(
                JOINT_NAMES.index(joint), config.robot.nominal_joint_deg, config=config
            )
            directions[joint] = JointDirection(
                joint=joint,
                direction_deg=theory_deg,
                source="theoretical",
                note=f"没有可用位移，退用理论方向。{theory_note}",
            )
            continue
        reference = max(amplitudes)
        plus = raw.get((joint, reference, 1), [])
        minus = raw.get((joint, reference, -1), [])
        directions[joint] = estimate_joint_direction(
            joint,
            plus,
            minus,
            config=config,
            direction_tolerance_deg=config.thresholds.direction_tolerance_deg,
        )
        if reference != 0.2:
            directions[joint].note += f"（注意：实际用的是 {reference}° 档，而不是 0.2°。）"

    # 正式逐条计算
    metrics_list: list[TrialMetrics] = []
    for trial in trials:
        segment = segments.get(str(trial.get("segment_id", "")))
        if segment is None:
            warnings.append(
                f"试验 {trial.get('event_id')} 的采集段找不到，跳过。"
            )
            continue
        metrics_list.append(
            analyze_trial(
                trial=trial,
                segment=segment,
                rtde_rows=rtde_rows,
                static_noise=static_noise,
                direction=directions[str(trial["joint"])],
                config=config,
            )
        )
        if progress is not None:
            progress(metrics_list[-1])

    sensitivities = estimate_sensitivity(metrics_list, config)
    apply_sensitivity(metrics_list, sensitivities)
    summaries = summarize_amplitudes(metrics_list, config)
    recommendations = recommend_steps(summaries, config)

    # 首尾漂移是"每边 n/10 帧的均值之差"，它的噪声是 σ/√(n/10)，
    # 所以拿 σ 本身当尺子已经足够保守：超过 3σ 基本不可能是纯抖动。
    if static_noise is not None:
        noise_scale = max(static_noise.std_x_px, static_noise.std_y_px, 1e-9)
        drift = max(abs(static_noise.drift_x_px), abs(static_noise.drift_y_px))
        if drift > 3.0 * noise_scale:
            warnings.append(
                f"静态段首尾漂移（{drift:.3f} px）超过噪声（{noise_scale:.3f} px）的 3 倍："
                "这段时间里相机或台面可能真的动过，噪声基线偏保守，"
                "请复核现场是否有人在碰设备。"
            )

    if not metrics_list:
        # 到了这里说明一段统计试验都没有（例如只采了静态基线就按了分析，
        # 或者预实验在第一段之前就被中止）。这时报告里除了静态噪声什么都没有，
        # 必须把"为什么没有推荐步长"说成"没测"，而不是让人对着空表猜。
        warnings.append(
            "这次没有任何统计试验（预实验没跑，或在第一段之前就中止了）："
            "静态噪声算得出来，但没有可用的候选幅度，也就没有推荐步长。"
            "已采到的数据都还在，补跑预实验之后不必重采静态基线。"
        )

    return PretestReport(
        static_noise=static_noise,
        directions=directions,
        sensitivities=sensitivities,
        trials=metrics_list,
        summaries=summaries,
        recommendations=recommendations,
        warnings=warnings,
        mode=config.mode,
    )


#: 空结果也要写出 CSV，而 write_csv 在"一行都没有"时是没法自己决定表头的。
#: 下面这几个占位实例**只用来取列名**，它们的数值一个都不会进报告。
#: 之所以用真实例而不是手写一份列名清单：手写的会慢慢和数据类脱节，
#: 而这里只要给 TrialMetrics 加了必填字段，构造这两个占位实例时会立刻报错。
_TRIALS_PROTOTYPE = TrialMetrics(
    event_id="",
    joint="",
    stage="",
    amplitude_deg=0.0,
    direction=1,
    repeat=1,
    segment_id="",
    commanded_delta_deg=0.0,
)
_AMPLITUDES_PROTOTYPE = AmplitudeSummary(joint="", amplitude_deg=0.0)
_SENSITIVITY_PROTOTYPE = Sensitivity(
    joint="", direction=1, px_per_deg=None, source="", samples=0, note=""
)


def write_report(report: PretestReport, out_dir: Path) -> dict[str, Path]:
    """把报告写成 CSV + JSON + 人可读文本。返回写出的文件。

    空结果（例如只采了静态基线、预实验被中止）也要能写出**带表头**的 CSV：
    分析结果目录缺文件会让人以为是丢了东西，而"这次没有统计试验"本身
    就是要如实写下来的事实之一。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    from .recorder import write_csv, write_json, write_text

    paths: dict[str, Path] = {}
    paths["trials"] = write_csv(
        out_dir / "trials.csv",
        [t.to_row() for t in report.trials],
        columns=list(_TRIALS_PROTOTYPE.to_row()),
    )
    paths["amplitudes"] = write_csv(
        out_dir / "amplitudes.csv",
        [s.to_row() for s in report.summaries],
        columns=list(_AMPLITUDES_PROTOTYPE.to_row()),
    )
    paths["sensitivity"] = write_csv(
        out_dir / "sensitivity.csv",
        [report.sensitivities[key].to_dict() for key in sorted(report.sensitivities)],
        columns=list(_SENSITIVITY_PROTOTYPE.to_dict()),
    )
    paths["directions"] = write_csv(
        out_dir / "directions.csv",
        [report.directions[joint].to_dict() for joint in JOINT_NAMES if joint in report.directions],
    )
    paths["json"] = write_json(out_dir / "pretest_report.json", report.to_dict())
    paths["text"] = write_text(
        out_dir / "pretest_report.txt", "\n".join(report.summary_lines()) + "\n"
    )
    return paths


def load_report(path: Path) -> dict[str, Any]:
    """读回 JSON 报告（界面刷新"推荐步长"输入框时用）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
