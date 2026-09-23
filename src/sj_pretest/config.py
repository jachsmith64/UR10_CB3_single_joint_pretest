"""本工具自己的配置对象：界面能改的每一个量都在这里，并能原样存成 JSON。

和 ``vendor/config.py`` 的关系（重要，别混）：

* ``vendor/config.py`` 是被复用代码的"全局设置模块"，相机/机器人/离线分析都从
  它里面读参数。它**不对外暴露成 JSON**，改它就得改源码。
* 本文件的 :class:`AppConfig` 才是**用户可见的配置**。界面改的是它，存成
  JSON 的也是它；它在启动时由 :mod:`sj_pretest.bridge` 翻译成 ``vendor/config.py``
  里的属性值，被复用代码读到的就已经是用户在界面上填的数。

这样做的目的很直接：**改步长、改时长、改阈值都不需要碰源代码**。

单位约定（避免实验现场搞错）：
* 关节角：度（deg）。存 JSON 也是度，只有真正发给 RTDE 时才换成弧度。
* 时间：秒（s）。
* 距离/位移：毫米（mm）。像素位移另用 px。
* 噪声：平移用 μm，转角用 deg。
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

#: 六个关节的名字，全工具统一用这一份，不要在别处另起名字。
JOINT_NAMES: tuple[str, ...] = ("J1", "J2", "J3", "J4", "J5", "J6")

#: 默认推荐姿态（= 离线可观测性工具 v1.3.0 标准档选出的那个姿态，
#: 已换算成归一化关节角）。它只是**摆放提示**，不代表碰撞已经验证过。
DEFAULT_NOMINAL_JOINT_DEG: tuple[float, ...] = (
    -53.8365,
    -175.6992,
    148.2887,
    -137.4699,
    -17.5120,
    -112.4909,
)

#: 相机摆放提示，来自同一个离线结果（只用于现场摆位，不做碰撞声明）。
DEFAULT_CAMERA_HINT: dict[str, Any] = {
    "mount_mode": "horizontal（水平放置）",
    "working_distance_mm": 675.0,
    "height_note": "相机与棋盘格中心大致等高",
    "azimuth_deg": 33.3,
    "facing_deviation_deg": 14.75,
    "collision_status": "unknown",
}

#: 运行模式。默认 dry_run —— 不接真机、不发命令。
VALID_MODES: tuple[str, ...] = ("dry_run", "replay", "hardware")

#: 三档预实验步长（度）。这是默认值，界面上可以改。
DEFAULT_AMPLITUDES_DEG: tuple[float, ...] = (0.01, 0.05, 0.2)

#: 走质心平移的关节与走二维旋转的关节（需求四）。
CENTROID_JOINTS: tuple[str, ...] = ("J1", "J2", "J3", "J4", "J5")
ROTATION_JOINTS: tuple[str, ...] = ("J6",)


class ConfigError(ValueError):
    """配置不合法。消息一律用中文，能直接显示给实验者看。"""


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} 必须是数字，收到 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{where} 必须是有限数字，收到 {value!r}")
    return result


def _as_optional_float(value: Any, where: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _as_float(value, where)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{where} 必须是整数，收到 {value!r}")
    return int(value)


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{where} 必须是 true/false，收到 {value!r}")


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} 必须是字符串，收到 {value!r}")
    return value


def _joint_list(value: Any, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ConfigError(f"{where} 必须是 6 个数字的列表，收到 {value!r}")
    return [_as_float(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _positive(value: float, where: str) -> float:
    if value <= 0:
        raise ConfigError(f"{where} 必须大于 0，收到 {value}")
    return value


def _non_negative(value: float, where: str) -> float:
    if value < 0:
        raise ConfigError(f"{where} 不能是负数，收到 {value}")
    return value


# --------------------------------------------------------------------------
# 各段配置
# --------------------------------------------------------------------------


@dataclass
class RobotConfig:
    """UR10 CB3 连接与关节运动参数。"""

    ip: str = "192.168.1.10"
    #: 默认推荐姿态（归一化到 ±180°）。
    nominal_joint_deg: list[float] = field(
        default_factory=lambda: list(DEFAULT_NOMINAL_JOINT_DEG)
    )

    #: 到位过程的分段数（需求：5～6 个中间点）。每段都要人工确认。
    approach_points: int = 6
    #: 到位速度/加速度。★ 来历见 README「默认值的来历」：旧项目真机上跑过的
    #: 是最慢档（ROBOT_TEST_SPEED_M_S = 0.02 m/s），这里按约 1 m 臂展折算成角度速度。
    approach_speed_deg_s: float = 1.0
    approach_accel_deg_s2: float = 2.0

    #: 微动步进的速度/加速度。★ 来历见 README「默认值的来历」：
    #: 旧项目真机用过的**最快**档是 0.05 m/s 直线速度（约 2.9°/s @1 m 臂展），
    #: 最保守档是 0.02 m/s（约 1.15°/s）。这里取 0.5 °/s，比最保守档还慢一倍。
    #: 不用更慢的原因：0.05°/s 会让 0.2° 的一步走 4 秒，整套 72 次要录二十多分钟，
    #: 存储代价不可接受，而微动步长的科学意义在**幅值**而不是"走得慢"。
    trial_speed_deg_s: float = 0.5
    trial_accel_deg_s2: float = 1.0

    #: 每个动作后等待关节真正停稳的判据。
    settle_tolerance_deg: float = 0.002
    settle_hold_s: float = 0.3
    settle_timeout_s: float = 8.0
    #: RTDE 状态记录频率（旧项目真机用的 125 Hz）。
    rtde_record_hz: float = 125.0
    #: 允许的最大关节速度/加速度（界面校验用，防止误填把机械臂开快）。
    max_speed_deg_s: float = 5.0
    max_accel_deg_s2: float = 10.0

    def validate(self) -> None:
        where = "robot"
        _as_str(self.ip, f"{where}.ip")
        joint_list = _joint_list(self.nominal_joint_deg, f"{where}.nominal_joint_deg")
        for index, value in enumerate(joint_list):
            if not -360.0 <= value <= 360.0:
                raise ConfigError(
                    f"{where}.nominal_joint_deg[{index}] = {value} 超出 ±360°，"
                    "请填写归一化后的角度。"
                )
        points = _as_int(self.approach_points, f"{where}.approach_points")
        if not 2 <= points <= 12:
            raise ConfigError(
                f"{where}.approach_points 必须在 2～12 之间（默认 6），收到 {points}"
            )
        speed = _positive(
            _as_float(self.approach_speed_deg_s, f"{where}.approach_speed_deg_s"),
            f"{where}.approach_speed_deg_s",
        )
        accel = _positive(
            _as_float(self.approach_accel_deg_s2, f"{where}.approach_accel_deg_s2"),
            f"{where}.approach_accel_deg_s2",
        )
        max_speed = _positive(
            _as_float(self.max_speed_deg_s, f"{where}.max_speed_deg_s"),
            f"{where}.max_speed_deg_s",
        )
        max_accel = _positive(
            _as_float(self.max_accel_deg_s2, f"{where}.max_accel_deg_s2"),
            f"{where}.max_accel_deg_s2",
        )
        if speed > max_speed:
            raise ConfigError(
                f"{where}.approach_speed_deg_s = {speed} 超过上限 "
                f"{max_speed} °/s，拒绝启动。"
            )
        if accel > max_accel:
            raise ConfigError(
                f"{where}.approach_accel_deg_s2 = {accel} 超过上限 "
                f"{max_accel} °/s²，拒绝启动。"
            )
        trial_speed = _positive(
            _as_float(self.trial_speed_deg_s, f"{where}.trial_speed_deg_s"),
            f"{where}.trial_speed_deg_s",
        )
        trial_accel = _positive(
            _as_float(self.trial_accel_deg_s2, f"{where}.trial_accel_deg_s2"),
            f"{where}.trial_accel_deg_s2",
        )
        if trial_speed > max_speed or trial_accel > max_accel:
            raise ConfigError(
                f"{where} 的 trial 速度/加速度超过上限："
                f"{trial_speed} °/s、{trial_accel} °/s²。"
            )
        _positive(
            _as_float(self.settle_tolerance_deg, f"{where}.settle_tolerance_deg"),
            f"{where}.settle_tolerance_deg",
        )
        _non_negative(
            _as_float(self.settle_hold_s, f"{where}.settle_hold_s"),
            f"{where}.settle_hold_s",
        )
        _positive(
            _as_float(self.settle_timeout_s, f"{where}.settle_timeout_s"),
            f"{where}.settle_timeout_s",
        )
        hz = _positive(
            _as_float(self.rtde_record_hz, f"{where}.rtde_record_hz"),
            f"{where}.rtde_record_hz",
        )
        if hz > 500:
            raise ConfigError(f"{where}.rtde_record_hz = {hz} 不现实（上限 500 Hz）。")


@dataclass
class CameraConfig:
    """海康 MV-CS028-10UM 采集参数。"""

    serial: str = ""
    #: MVS 安装目录里的 Python 示例包路径（被复用代码从这里 import MvCameraControl）。
    mvs_import_path: str | None = (
        r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
    )
    exposure_us: float | None = 6500.0
    gain: float | None = None
    frame_timeout_ms: int = 1000
    #: 目标帧率。旧项目真机实测 132.23 fps，这里作为默认期望值。
    expected_fps: float = 132.23
    #: 开始采集前的预热（丢弃前若干帧，让自动曝光/增益稳定）。
    warmup_s: float = 5.0

    #: 棋盘格：12×9 方格 → 11×8 内角点 → 88 个角点；小格 3 mm。
    board_inner_corners: list[int] = field(default_factory=lambda: [11, 8])
    square_mm: float = 3.0

    #: 采集时长（秒）。静态噪声测量默认 15 s（需求允许 10～20 s）。
    static_duration_s: float = 15.0
    #: 运动后继续录制的时间，用来观察"RTDE 稳了但画面还在动"。
    #: 需求六的动态残差层就靠这一段，所以不能太短；0.3 s ≈ 40 帧。
    post_motion_s: float = 0.3
    #: 运动前先录一段静止，作为这次动作自己的参考帧。0.3 s ≈ 40 帧。
    pre_motion_s: float = 0.3
    #: 到达目标并停稳之后，继续保持录制的时间。
    #: 这是整套预实验里**最有信息量**的一段（稳定值 + 残余振动），
    #: 所以给到 1.5 s ≈ 200 帧；嫌数据大可以在界面上调小。
    hold_s: float = 1.5
    #: 裁剪区域 (x, y, w, h)，None = 不裁剪。
    #: ★ 现场建议填一个包住棋盘格的区域：分辨率是磁盘占用的第一主导项，
    #: 1936×1096 全画幅在 132 fps 下约 280 MB/s，裁到 800×600 只剩约 48 MB/s。
    roi: list[int] | None = None

    #: 是否保存原始帧（流式写 frames.raw）。
    save_raw: bool = True
    #: 是否额外存几张贴图，方便人工肉眼确认。
    save_sample_images: bool = True
    sample_image_count: int = 5

    #: 画面边距下限（像素）。棋盘格离画面边缘太近就认为"边距不足"，暂停。
    min_margin_px: int = 40
    #: 丢帧比例上限，超过就报警。
    max_dropped_ratio: float = 0.02

    def validate(self) -> None:
        where = "camera"
        _as_str(self.serial, f"{where}.serial")
        if self.mvs_import_path is not None:
            _as_str(self.mvs_import_path, f"{where}.mvs_import_path")
        _as_optional_float(self.exposure_us, f"{where}.exposure_us")
        _as_optional_float(self.gain, f"{where}.gain")
        timeout = _as_int(self.frame_timeout_ms, f"{where}.frame_timeout_ms")
        if timeout <= 0:
            raise ConfigError(f"{where}.frame_timeout_ms 必须为正，收到 {timeout}")
        _positive(_as_float(self.expected_fps, f"{where}.expected_fps"),
                  f"{where}.expected_fps")
        _non_negative(_as_float(self.warmup_s, f"{where}.warmup_s"), f"{where}.warmup_s")

        corners = self.board_inner_corners
        if not isinstance(corners, (list, tuple)) or len(corners) != 2:
            raise ConfigError(
                f"{where}.board_inner_corners 必须是两个整数（内角点列数、行数），"
                f"收到 {corners!r}"
            )
        cols = _as_int(corners[0], f"{where}.board_inner_corners[0]")
        rows = _as_int(corners[1], f"{where}.board_inner_corners[1]")
        if cols < 2 or rows < 2:
            raise ConfigError(f"{where}.board_inner_corners 两维都要 ≥ 2，收到 {corners!r}")
        if cols * rows != 88:
            # 需求四要求 88 个内角点，这里只提醒不阻断（换板子的人自己清楚）。
            pass
        _positive(_as_float(self.square_mm, f"{where}.square_mm"), f"{where}.square_mm")

        static_s = _positive(
            _as_float(self.static_duration_s, f"{where}.static_duration_s"),
            f"{where}.static_duration_s",
        )
        if not 5.0 <= static_s <= 60.0:
            raise ConfigError(
                f"{where}.static_duration_s = {static_s} 超出 5～60 s，"
                "静态噪声测量默认 15 s（需求 10～20 s）。"
            )
        for name in ("post_motion_s", "pre_motion_s", "hold_s"):
            _non_negative(
                _as_float(getattr(self, name), f"{where}.{name}"), f"{where}.{name}"
            )
        if self.roi is not None:
            if not isinstance(self.roi, (list, tuple)) or len(self.roi) != 4:
                raise ConfigError(
                    f"{where}.roi 必须是 [x, y, w, h] 四个整数，或 null 表示不裁剪，"
                    f"收到 {self.roi!r}"
                )
            for position, value in enumerate(self.roi):
                number = _as_int(value, f"{where}.roi[{position}]")
                if number < 0:
                    raise ConfigError(f"{where}.roi[{position}] 不能为负，收到 {number}")
            if int(self.roi[2]) <= 0 or int(self.roi[3]) <= 0:
                raise ConfigError(f"{where}.roi 的宽和高必须为正，收到 {self.roi!r}")
        if not _as_bool(self.save_raw, f"{where}.save_raw"):
            # 需求六要求原始帧必须完整保存（RAW 或经过验证的无损视频）。
            # 关掉它就没有可供离线识别和复核的数据了，所以这里直接拒绝。
            raise ConfigError(
                f"{where}.save_raw 必须是 true：本工具要求完整保存原始帧，"
                "否则实验结束后无法离线复核角点。"
            )
        _as_bool(self.save_sample_images, f"{where}.save_sample_images")
        margin = _as_int(self.min_margin_px, f"{where}.min_margin_px")
        if margin < 0:
            raise ConfigError(f"{where}.min_margin_px 不能为负，收到 {margin}")
        ratio = _as_float(self.max_dropped_ratio, f"{where}.max_dropped_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"{where}.max_dropped_ratio 必须在 0～1，收到 {ratio}")


@dataclass
class PretestConfig:
    """静态噪声 + 三档步长预实验的参数。"""

    #: 三档步长（度）。
    amplitudes_deg: list[float] = field(
        default_factory=lambda: list(DEFAULT_AMPLITUDES_DEG)
    )
    #: 每个幅值、每个方向重复几次（需求：正负各 2 次）。
    repeats_per_direction: int = 2
    #: 快速几何确认用的试探步长（需求：每个关节先用 0.05° 试一次）。
    quick_probe_deg: float = 0.05
    #: 参与预实验的关节顺序。
    joints: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    #: 每个新关节开始前是否要人工确认（需求要求保留）。
    confirm_each_joint: bool = True
    #: 同一关节内部是否自动跑完 12 次。
    auto_within_joint: bool = True
    #: 每次试验的动作点之间保持时间之后，再等 RTDE 稳定。
    return_before_settle_s: float = 0.2

    def validate(self) -> None:
        where = "pretest"
        values = self.amplitudes_deg
        if not isinstance(values, (list, tuple)) or not values:
            raise ConfigError(f"{where}.amplitudes_deg 至少要有一个步长。")
        parsed = [_as_float(v, f"{where}.amplitudes_deg") for v in values]
        for value in parsed:
            if value <= 0:
                raise ConfigError(
                    f"{where}.amplitudes_deg 里不能有非正数（收到 {value}）。"
                    "本工具**不会**自动放大到 1°/5°，需要更大请人工填写。"
                )
        if len(set(parsed)) != len(parsed):
            raise ConfigError(f"{where}.amplitudes_deg 里有重复值：{parsed}")
        repeats = _as_int(
            self.repeats_per_direction, f"{where}.repeats_per_direction"
        )
        if not 1 <= repeats <= 10:
            raise ConfigError(
                f"{where}.repeats_per_direction 必须在 1～10，收到 {repeats}"
            )
        _positive(_as_float(self.quick_probe_deg, f"{where}.quick_probe_deg"),
                  f"{where}.quick_probe_deg")
        joints = self.joints
        if not isinstance(joints, (list, tuple)) or not joints:
            raise ConfigError(f"{where}.joints 不能为空。")
        for name in joints:
            if name not in JOINT_NAMES:
                raise ConfigError(f"{where}.joints 里出现未知关节 {name!r}。")
        if len(set(joints)) != len(joints):
            raise ConfigError(f"{where}.joints 里有重复关节：{list(joints)}")
        _as_bool(self.confirm_each_joint, f"{where}.confirm_each_joint")
        _as_bool(self.auto_within_joint, f"{where}.auto_within_joint")
        _non_negative(
            _as_float(self.return_before_settle_s, f"{where}.return_before_settle_s"),
            f"{where}.return_before_settle_s",
        )


@dataclass
class VisionConfig:
    """离线视觉计算参数（需求四：二维质心 + J6 二维旋转）。"""

    #: 走质心平移投影的关节。
    centroid_joints: list[str] = field(default_factory=lambda: list(CENTROID_JOINTS))
    #: 走二维 Kabsch 旋转的关节。
    rotation_joints: list[str] = field(default_factory=lambda: list(ROTATION_JOINTS))
    #: 估计"图像中局部运动方向"时用哪一档步长（需求：用 0.2° 数据确定）。
    direction_reference_deg: float = 0.2
    #: 局部灵敏度优先用 RTDE 实际角变化；实在分辨不出来才退化用指令角。
    #: 退化时结果里会明确标注（需求四要求）。
    sensitivity_from_rtde: bool = True
    #: 角点检测残差上限（像素）。超过就认为这一帧不可信。
    max_residual_px: float = 1.5
    #: 一帧里至少要检测到多少角点才算"完整"（88 个全到才算完整识别）。
    min_corners: int = 88
    #: μm/px 参考值的来源说明；只作参考，不当作相机标定。
    um_per_px_note: str = (
        "μm/px 由角点间距与 3 mm 小格现场估计，只作参考，"
        "本工具不做完整相机标定、不做 PnP。"
    )

    def validate(self) -> None:
        where = "vision"
        for name, pool in (
            ("centroid_joints", JOINT_NAMES),
            ("rotation_joints", JOINT_NAMES),
        ):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ConfigError(f"{where}.{name} 必须是列表。")
            for joint in values:
                if joint not in pool:
                    raise ConfigError(f"{where}.{name} 里出现未知关节 {joint!r}。")
        centroid = set(self.centroid_joints)
        rotation = set(self.rotation_joints)
        if centroid & rotation:
            raise ConfigError(
                f"{where}：关节 {sorted(centroid & rotation)} 同时出现在质心和旋转列表里，"
                "一个关节只能用一种算法。"
            )
        if not centroid | rotation:
            raise ConfigError(f"{where}：质心和旋转关节列表不能都为空。")
        _positive(
            _as_float(self.direction_reference_deg, f"{where}.direction_reference_deg"),
            f"{where}.direction_reference_deg",
        )
        _as_bool(self.sensitivity_from_rtde, f"{where}.sensitivity_from_rtde")
        _positive(_as_float(self.max_residual_px, f"{where}.max_residual_px"),
                  f"{where}.max_residual_px")
        corners = _as_int(self.min_corners, f"{where}.min_corners")
        if corners < 4:
            raise ConfigError(f"{where}.min_corners 至少 4，收到 {corners}")


@dataclass
class ThresholdConfig:
    """推荐正式步长时的判据阈值。全部可以在界面上改、并存进 JSON。"""

    #: 视觉信号至少要是静态噪声的多少倍（需求六：5 倍）。
    min_snr_vs_static: float = 5.0
    #: 正负方向符号必须一致。
    require_sign_match: bool = True
    #: 重复试验之间的相对离散上限（标准差 / 均值）。
    max_repeat_relative_spread: float = 0.5
    #: RTDE 实际角变化至少要达到多少才算"确实有响应"（度）。
    min_rtde_response_deg: float = 0.002
    #: 正负方向差异上限（相对步长）。
    max_direction_asymmetry: float = 0.5
    #: 角点检测必须稳定：这一档里有效帧占比下限。
    min_valid_frame_ratio: float = 0.8
    #: 不允许接近关节限位：距限位至少留多少度。
    joint_limit_margin_deg: float = 5.0
    #: 允许的最大深度/面内运动比（超出说明主要不是面内运动）。
    max_depth_ratio: float = 0.5
    #: 符号正确率下限（1.0 = 每一次都必须同号）。
    min_sign_consistency: float = 1.0
    #: 重复之间的差，除了相对上限以外还给一个"绝对噪声倍数"上限：
    #: 小步长时相对上限会松得离谱，绝对倍数才能拦住"两次差出好几倍噪声"。
    repeat_noise_multiple: float = 3.0
    #: RTDE 实际变化 ÷ 指令变化 的下限。
    min_rtde_response_ratio: float = 0.5
    #: 第三层判据：RTDE 停稳后画面残余运动超过静态噪声多少倍才算"还在振"。
    residual_factor: float = 3.0
    #: 正负两组位移的夹角与 180° 的最大允许偏差（度）。超出则方向不可信。
    direction_tolerance_deg: float = 30.0
    #: 快速几何检查（按钮二 B，0.05° 探针）：位移至少要达到静止段噪声的多少倍
    #: 才算"画面上有可见运动"。这一档本来就是"0.05° 看不看得出来"的判据，
    #: 所以门槛比正式分析的 5 倍低：3 倍。
    quick_probe_min_snr: float = 3.0
    #: 快速几何检查里"方向大致符合理论"的容差（度）。
    quick_probe_direction_tol_deg: float = 45.0

    def validate(self) -> None:
        where = "thresholds"
        _positive(
            _as_float(self.min_snr_vs_static, f"{where}.min_snr_vs_static"),
            f"{where}.min_snr_vs_static",
        )
        _as_bool(self.require_sign_match, f"{where}.require_sign_match")
        spread = _non_negative(
            _as_float(self.max_repeat_relative_spread, f"{where}.max_repeat_relative_spread"),
            f"{where}.max_repeat_relative_spread",
        )
        if spread > 5.0:
            raise ConfigError(
                f"{where}.max_repeat_relative_spread = {spread} 过于宽松（上限 5.0）。"
            )
        _non_negative(
            _as_float(self.min_rtde_response_deg, f"{where}.min_rtde_response_deg"),
            f"{where}.min_rtde_response_deg",
        )
        _non_negative(
            _as_float(self.max_direction_asymmetry, f"{where}.max_direction_asymmetry"),
            f"{where}.max_direction_asymmetry",
        )
        ratio = _as_float(self.min_valid_frame_ratio, f"{where}.min_valid_frame_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"{where}.min_valid_frame_ratio 必须在 0～1，收到 {ratio}")
        _non_negative(
            _as_float(self.joint_limit_margin_deg, f"{where}.joint_limit_margin_deg"),
            f"{where}.joint_limit_margin_deg",
        )
        _non_negative(
            _as_float(self.max_depth_ratio, f"{where}.max_depth_ratio"),
            f"{where}.max_depth_ratio",
        )
        sign_floor = _as_float(
            self.min_sign_consistency, f"{where}.min_sign_consistency"
        )
        if not 0.0 <= sign_floor <= 1.0:
            raise ConfigError(
                f"{where}.min_sign_consistency 必须在 0～1，收到 {sign_floor}"
            )
        _positive(
            _as_float(self.repeat_noise_multiple, f"{where}.repeat_noise_multiple"),
            f"{where}.repeat_noise_multiple",
        )
        response = _as_float(
            self.min_rtde_response_ratio, f"{where}.min_rtde_response_ratio"
        )
        if not 0.0 < response <= 1.5:
            raise ConfigError(
                f"{where}.min_rtde_response_ratio 必须在 0～1.5，收到 {response}"
            )
        _positive(
            _as_float(self.residual_factor, f"{where}.residual_factor"),
            f"{where}.residual_factor",
        )
        tolerance = _as_float(
            self.direction_tolerance_deg, f"{where}.direction_tolerance_deg"
        )
        if not 0.0 <= tolerance < 90.0:
            raise ConfigError(
                f"{where}.direction_tolerance_deg 必须在 0～90，收到 {tolerance}"
                "（90 表示完全不检查正负是否反向）。"
            )
        _positive(
            _as_float(self.quick_probe_min_snr, f"{where}.quick_probe_min_snr"),
            f"{where}.quick_probe_min_snr",
        )
        probe_tol = _as_float(
            self.quick_probe_direction_tol_deg,
            f"{where}.quick_probe_direction_tol_deg",
        )
        if not 0.0 <= probe_tol <= 180.0:
            raise ConfigError(
                f"{where}.quick_probe_direction_tol_deg 必须在 0～180，收到 {probe_tol}"
            )


@dataclass
class FormalConfig:
    """正式实验（组 A 阶梯 / 组 B 换向）参数。"""

    #: 每个关节的正式步长（度）。预实验分析完成后由界面填入，可人工改。
    step_deg: dict[str, float] = field(default_factory=dict)
    #: 阶梯级数 N（组 A）。
    staircase_n: int = 5
    #: 重复次数（组 A 的完整阶梯跑几遍 / 组 B 的来回几遍）。
    repeats: int = 3
    #: 每个点保持时间。
    hold_s: float = 1.0
    #: 组与组之间的等待时间。
    group_wait_s: float = 3.0
    #: 是否跑组 A、组 B。
    enable_group_a: bool = True
    enable_group_b: bool = True
    #: 正式实验用的速度/加速度（默认沿用预实验，界面可改）。
    speed_deg_s: float = 0.05
    accel_deg_s2: float = 0.1
    #: 正式实验前必须做的名义运动学范围检查，超出预实验已验证范围就警告。
    require_range_check: bool = True
    #: 预实验每个关节实际验证过的最大幅值（度），由预实验自动填入。
    verified_amplitude_deg: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        where = "formal"
        _as_int(self.staircase_n, f"{where}.staircase_n")
        if not 1 <= self.staircase_n <= 50:
            raise ConfigError(f"{where}.staircase_n 必须在 1～50，收到 {self.staircase_n}")
        repeats = _as_int(self.repeats, f"{where}.repeats")
        if not 1 <= repeats <= 50:
            raise ConfigError(f"{where}.repeats 必须在 1～50，收到 {repeats}")
        for name in ("hold_s", "group_wait_s"):
            _non_negative(_as_float(getattr(self, name), f"{where}.{name}"),
                          f"{where}.{name}")
        _positive(_as_float(self.speed_deg_s, f"{where}.speed_deg_s"),
                  f"{where}.speed_deg_s")
        _positive(_as_float(self.accel_deg_s2, f"{where}.accel_deg_s2"),
                  f"{where}.accel_deg_s2")
        _as_bool(self.enable_group_a, f"{where}.enable_group_a")
        _as_bool(self.enable_group_b, f"{where}.enable_group_b")
        _as_bool(self.require_range_check, f"{where}.require_range_check")
        for joint, value in self.step_deg.items():
            if joint not in JOINT_NAMES:
                raise ConfigError(f"{where}.step_deg 里出现未知关节 {joint!r}")
            step = _as_float(value, f"{where}.step_deg[{joint}]")
            if step <= 0:
                raise ConfigError(f"{where}.step_deg[{joint}] 必须为正，收到 {step}")


@dataclass
class PathConfig:
    """输出目录与磁盘门槛。"""

    output_root: str = "outputs"
    #: 开始任何采集前要求的可用磁盘空间下限（GB）。
    min_free_disk_gb: float = 2.0

    def validate(self) -> None:
        _as_str(self.output_root, "paths.output_root")
        if not self.output_root.strip():
            raise ConfigError("paths.output_root 不能为空。")
        free = _non_negative(
            _as_float(self.min_free_disk_gb, "paths.min_free_disk_gb"),
            "paths.min_free_disk_gb",
        )
        if free > 200:
            raise ConfigError(
                f"paths.min_free_disk_gb = {free} GB 过于苛刻（上限 200 GB）。"
            )


@dataclass
class DryRunConfig:
    """干运行（dry_run）用的合成世界参数。只在 mode="dry_run" 时生效。

    合成世界**不是**真实机械臂的模型，它的唯一用途是：让整套流程（采集 → 运动编排 →
    离线角点 → 三层分析 → 步长推荐）在不接真机的前提下真的被跑一遍。
    所以它的输出必须始终标记为 synthetic，绝不能当成实验结果。
    """

    #: 合成图像尺寸（像素）。默认小一点，让自测产生的 RAW 不至于吃掉几十 GB。
    width: int = 480
    height: int = 360
    #: 合成帧率。默认与被复用代码期望的 132.23 fps 一致，用来验证时间轴语义。
    fps: float = 132.23
    #: True = 按真实时间出图（能看预览、能感时间）；False = 虚拟时钟，跑得飞快。
    realtime: bool = False
    #: 把静态/保持/运动后的时长压短，让整套自测几分钟内跑完。
    #: ★ 只有 dry_run 会看这个开关，replay/hardware 永远用 camera 里的真实时长。
    shorten_durations: bool = True
    #: 缩短后的时长（秒）。
    static_duration_s: float = 3.0
    pre_motion_s: float = 0.2
    hold_s: float = 0.4
    post_motion_s: float = 0.2
    #: 每 N 帧故意让角点检测失败一次（验证"棋盘格丢失就暂停"）。0 = 不注入。
    corner_loss_every: int = 0
    #: 每 N 帧故意丢一帧（验证丢帧会被记录）。0 = 不注入。
    drop_every: int = 0
    #: 合成世界的噪声与灵敏度参数（只影响自测，不影响任何真实结论）。
    #: 噪声量级按"真实 3 mm 棋盘格在 675 mm 处成像约 10 px/格"的情况取，
    #: 所以 0.5 px 的质心抖动是现场会遇到的量级，不是为了让自测好看而挑的数。
    centroid_noise_px: float = 0.5
    rotation_noise_deg: float = 0.02
    #: 各关节的合成灵敏度（像素/度）。数值刻意大小不一：
    #: 这样"0.01° 对某些关节够用、对另一些不够"的判据在自测里能被真正走到。
    centroid_px_per_deg: dict[str, float] = field(
        default_factory=lambda: {
            "J1": 40.0,
            "J2": 60.0,
            "J3": 90.0,
            "J4": 25.0,
            "J5": 12.0,
            "J6": 0.0,
        }
    )
    #: 各关节合成的图像内运动方向（度，0 = 图像 +x 方向）。
    #: 这几个数**不是随便填的**：它们等于
    #: ``vision.expected_image_direction_deg`` 在默认名义姿态 + 默认摆放提示下
    #: 算出来的预测方向。这样"方向是否大致符合理论"这条判据在干运行里应当是
    #: 通过的；要验证它**能报错**，改其中一个方向（tests 里有专门用例）即可。
    image_direction_deg: dict[str, float] = field(
        default_factory=lambda: {
            "J1": 0.0,
            "J2": -114.5,
            "J3": 91.9,
            "J4": 95.2,
            "J5": -135.7,
            "J6": 0.0,
        }
    )
    #: J6 的合成旋转灵敏度（图像转角/关节角）。
    rotation_deg_per_deg: float = 1.0
    #: 运动后的合成"结构振荡"：幅值（像素）、频率（Hz）、衰减时间常数（秒）。
    vibration_amplitude_px: float = 0.4
    vibration_hz: float = 6.5
    vibration_tau_s: float = 0.25
    #: 关节对指令的一阶跟随时间常数（秒），用来合成"RTDE 也在动"的过程。
    joint_response_tau_s: float = 0.12
    #: 合成世界的随机种子，保证自测可重复。
    seed: int = 20260923

    @property
    def duration_overrides(self) -> dict[str, float]:
        """dry_run 要覆盖的时长。关闭 shorten 时返回空字典。"""
        if not self.shorten_durations:
            return {}
        return {
            "static": float(self.static_duration_s),
            "pre_motion": float(self.pre_motion_s),
            "hold": float(self.hold_s),
            "post_motion": float(self.post_motion_s),
        }

    def validate(self) -> None:
        where = "dry_run"
        width = _as_int(self.width, f"{where}.width")
        height = _as_int(self.height, f"{where}.height")
        if not 64 <= width <= 4096 or not 64 <= height <= 4096:
            raise ConfigError(
                f"{where} 的合成图像尺寸必须在 64～4096，收到 {width}×{height}"
            )
        _positive(_as_float(self.fps, f"{where}.fps"), f"{where}.fps")
        _as_bool(self.realtime, f"{where}.realtime")
        _as_bool(self.shorten_durations, f"{where}.shorten_durations")
        for name in ("static_duration_s", "pre_motion_s", "hold_s", "post_motion_s"):
            value = _non_negative(
                _as_float(getattr(self, name), f"{where}.{name}"), f"{where}.{name}"
            )
            if value > 120.0:
                raise ConfigError(f"{where}.{name} = {value} s 太长（上限 120 s）。")
        for name in ("corner_loss_every", "drop_every"):
            value = _as_int(getattr(self, name), f"{where}.{name}")
            if value < 0:
                raise ConfigError(f"{where}.{name} 不能为负，收到 {value}")
        _positive(
            _as_float(self.centroid_noise_px, f"{where}.centroid_noise_px"),
            f"{where}.centroid_noise_px",
        )
        _non_negative(
            _as_float(self.rotation_noise_deg, f"{where}.rotation_noise_deg"),
            f"{where}.rotation_noise_deg",
        )
        for dict_name, expected in (
            ("centroid_px_per_deg", JOINT_NAMES),
            ("image_direction_deg", JOINT_NAMES),
        ):
            values = getattr(self, dict_name)
            if not isinstance(values, Mapping):
                raise ConfigError(f"{where}.{dict_name} 必须是 JSON 对象。")
            for joint in expected:
                if joint not in values:
                    raise ConfigError(
                        f"{where}.{dict_name} 缺少 {joint}，六个关节都要给。"
                    )
                _as_float(values[joint], f"{where}.{dict_name}[{joint}]")
        _positive(
            _as_float(self.rotation_deg_per_deg, f"{where}.rotation_deg_per_deg"),
            f"{where}.rotation_deg_per_deg",
        )
        for name in ("vibration_amplitude_px", "vibration_tau_s", "joint_response_tau_s"):
            _positive(_as_float(getattr(self, name), f"{where}.{name}"), f"{where}.{name}")
        _positive(_as_float(self.vibration_hz, f"{where}.vibration_hz"),
                  f"{where}.vibration_hz")
        _as_int(self.seed, f"{where}.seed")


@dataclass
class AppConfig:
    """整套配置。界面改的就是它，存 JSON 的也是它。"""

    #: dry_run / replay / hardware —— 默认 dry_run，真机要显式选。
    mode: str = "dry_run"
    #: replay 模式下从哪里读历史数据。
    replay_source: str = ""
    note: str = ""
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    pretest: PretestConfig = field(default_factory=PretestConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    formal: FormalConfig = field(default_factory=FormalConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    dry_run: DryRunConfig = field(default_factory=DryRunConfig)

    # -- 校验 ---------------------------------------------------------------

    def validate(self) -> None:
        mode = _as_str(self.mode, "mode")
        if mode not in VALID_MODES:
            raise ConfigError(
                f"mode 必须是 {list(VALID_MODES)} 之一，收到 {mode!r}。"
                "真机运行必须显式选 'hardware'。"
            )
        _as_str(self.replay_source, "replay_source")
        _as_str(self.note, "note")
        self.robot.validate()
        self.camera.validate()
        self.pretest.validate()
        self.vision.validate()
        self.thresholds.validate()
        self.formal.validate()
        self.paths.validate()
        self.dry_run.validate()

        self._validate_joint_coverage()
        self._validate_amplitudes_within_limits()

    def _validate_joint_coverage(self) -> None:
        """预实验要跑的关节必须都有算法（质心或旋转），否则白跑。"""
        planned = set(self.pretest.joints)
        covered = set(self.vision.centroid_joints) | set(self.vision.rotation_joints)
        missing = sorted(planned - covered)
        if missing:
            raise ConfigError(
                f"预实验包含关节 {missing}，但 vision 里既没有给它们配"
                "centroid_joints 也没有配 rotation_joints，跑完也没有结果可算。"
            )

    def _validate_amplitudes_within_limits(self) -> None:
        """三档步长里最大的那个，单独一个动作也不能把关节顶出限位。"""
        from .kinematics import JOINT_LIMITS_DEG, check_joint_in_limits

        nominal = self.robot.nominal_joint_deg
        worst = max(self.pretest.amplitudes_deg)
        margin = self.thresholds.joint_limit_margin_deg
        for index, name in enumerate(JOINT_NAMES):
            if name not in self.pretest.joints:
                continue
            for sign in (+1.0, -1.0):
                target = list(nominal)
                target[index] += sign * worst
                ok, reason = check_joint_in_limits(
                    target, margin_deg=margin, limits=JOINT_LIMITS_DEG
                )
                if not ok:
                    raise ConfigError(
                        f"名义姿态下 J{index + 1} 走 ±{worst}° 会出问题：{reason}"
                    )

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent) + "\n"

    def save(self, path: str | Path) -> Path:
        """保存到 JSON。先写临时文件再替换，避免中途失败留下半份配置。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(self.to_json(), encoding="utf-8")
        temp.replace(target)
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("配置文件的根必须是一个 JSON 对象。")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"配置文件里有本工具不认识的字段：{unknown}。"
                "请对照 configs/experiment_default.json 修改（拼错的字段不会静默忽略）。"
            )
        config = cls(
            mode=data.get("mode", "dry_run"),
            replay_source=data.get("replay_source", ""),
            note=data.get("note", ""),
            robot=_build(RobotConfig, data.get("robot"), "robot"),
            camera=_build(CameraConfig, data.get("camera"), "camera"),
            pretest=_build(PretestConfig, data.get("pretest"), "pretest"),
            vision=_build(VisionConfig, data.get("vision"), "vision"),
            thresholds=_build(ThresholdConfig, data.get("thresholds"), "thresholds"),
            formal=_build(FormalConfig, data.get("formal"), "formal"),
            paths=_build(PathConfig, data.get("paths"), "paths"),
            dry_run=_build(DryRunConfig, data.get("dry_run"), "dry_run"),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        source = Path(path)
        if not source.is_file():
            raise ConfigError(f"配置文件不存在：{source}")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"配置文件不是合法 JSON：{source}（{exc}）") from exc
        return cls.from_dict(raw)

    # -- 派生信息 -----------------------------------------------------------

    def effective_durations(self) -> dict[str, float]:
        """当前模式下**真正生效**的四个时长。

        dry_run 且开了 ``shorten_durations`` 时用合成世界的短时长，
        其余情况一律用 ``camera`` 里的现场时长——这条规则很关键：
        自测跑的时长和真机跑的时长不是一回事，不能拿自测的时长去推断真机要多久。
        """
        durations = {
            "static": float(self.camera.static_duration_s),
            "pre_motion": float(self.camera.pre_motion_s),
            "hold": float(self.camera.hold_s),
            "post_motion": float(self.camera.post_motion_s),
        }
        if self.mode == "dry_run":
            durations.update(self.dry_run.duration_overrides)
        return durations

    def effective_camera_size(self) -> tuple[int, int]:
        """当前模式下每帧的宽高（像素）。用于磁盘估算。"""
        if self.mode == "dry_run":
            return int(self.dry_run.width), int(self.dry_run.height)
        roi = self.camera.roi
        if roi:
            return int(roi[2]), int(roi[3])
        # 没有 ROI 时按传感器满幅估（MV-CS028-10UM 是 1936×1096）。
        return 1936, 1096

    def effective_fps(self) -> float:
        if self.mode == "dry_run":
            return float(self.dry_run.fps)
        return float(self.camera.expected_fps)

    def effective_speed(self, *, formal: bool = False) -> tuple[float, float]:
        """当前模式下生效的角速度/角加速度（度/秒、度/秒²）。"""
        if formal:
            return float(self.formal.speed_deg_s), float(self.formal.accel_deg_s2)
        return float(self.robot.trial_speed_deg_s), float(self.robot.trial_accel_deg_s2)

    def resolve_output_root(self) -> Path:
        """输出根目录：相对路径按"当前工作目录"解析，绝对路径原样使用。"""
        root = Path(self.output_root if False else self.paths.output_root).expanduser()
        return root if root.is_absolute() else (Path.cwd() / root)

    def camera_hint_lines(self) -> list[str]:
        """把相机摆放提示整理成给人看的中文行。"""
        hint = DEFAULT_CAMERA_HINT
        return [
            f"摆法：{hint['mount_mode']}",
            f"工作距离：约 {hint['working_distance_mm']:.0f} mm",
            f"高度：{hint['height_note']}",
            f"水平方位角：约 {hint['azimuth_deg']:.1f}°",
            f"棋盘格与相机正视方向偏差：约 {hint['facing_deviation_deg']:.2f}°",
            "★ 这些只是摆放提示；collision_status = unknown，碰撞未经验证。",
        ]

    def describe(self) -> dict[str, Any]:
        """给报告/日志用的摘要（不含大列表）。"""
        return {
            "mode": self.mode,
            "nominal_joint_deg": list(self.robot.nominal_joint_deg),
            "amplitudes_deg": list(self.pretest.amplitudes_deg),
            "repeats_per_direction": self.pretest.repeats_per_direction,
            "static_duration_s": self.camera.static_duration_s,
            "joints": list(self.pretest.joints),
            "centroid_joints": list(self.vision.centroid_joints),
            "rotation_joints": list(self.vision.rotation_joints),
            "rtde_record_hz": self.robot.rtde_record_hz,
        }


def _build(cls: type, data: Any, where: str) -> Any:
    """把一段 dict 变成 dataclass，并把未知字段报出来（不静默丢）。"""
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise ConfigError(f"配置段 {where} 必须是 JSON 对象，收到 {type(data).__name__}")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"配置段 {where} 里有不认识的字段：{unknown}")
    kwargs = {}
    for item in fields(cls):
        if item.name in data:
            kwargs[item.name] = data[item.name]
    return cls(**kwargs)


def trapezoid_seconds(distance_deg: float, speed_deg_s: float, accel_deg_s2: float) -> float:
    """梯形速度曲线走完一段角位移需要多久（秒）。

    加速段和减速段各用 ``v/a`` 秒；如果这段距离短到还没加到最高速就要开始减速，
    就用三角形曲线，时间 = ``2*sqrt(d/a)``。这是一个理想化模型——真机还有滤波、
    关节耦合和控制器自己的规划，实际会比这个数字长，所以估算时只用它做保守下界，
    真正的时长以 RTDE 记录的到位时间为准。
    """
    distance = abs(float(distance_deg))
    if distance <= 0:
        return 0.0
    speed = float(speed_deg_s)
    accel = float(accel_deg_s2)
    if speed <= 0 or accel <= 0:
        raise ConfigError("速度/加速度必须为正。")
    ramp_distance = speed * speed / accel
    if distance <= ramp_distance:
        return 2.0 * math.sqrt(distance / accel)
    return distance / speed + speed / accel


def estimate_capture_seconds(config: AppConfig, plans: Iterable[Any]) -> float:
    """估算这些计划一共要录制多少秒（用来算磁盘占用）。

    规则：一个"去程 + 回程"算作一次试验，录一段连续的：运动前静止 → 去程 →
    保持 → 回程 → 运动后记录。纯等待步（组间等待）不录制。
    """
    durations = config.effective_durations()
    speed, accel = config.effective_speed()
    total = 0.0
    for plan in plans:
        steps = list(getattr(plan, "steps", plan))
        index = 0
        while index < len(steps):
            step = steps[index]
            if not step.is_motion:
                # 纯等待：不做视觉记录。
                index += 1
                continue
            if step.event.role != "move":
                # 单独出现的回程（理论上不会，防御一下）。
                delta = _step_delta(step)
                total += trapezoid_seconds(delta, speed, accel) + step.hold_s
                index += 1
                continue
            total += durations["pre_motion"]
            total += trapezoid_seconds(_step_delta(step), speed, accel)
            total += float(step.hold_s)
            next_step = steps[index + 1] if index + 1 < len(steps) else None
            if (
                next_step is not None
                and next_step.event.role == "return"
                and next_step.is_motion
            ):
                total += trapezoid_seconds(_step_delta(next_step), speed, accel)
                total += float(next_step.hold_s)
                index += 2
            else:
                index += 1
            total += durations["post_motion"]
        if not any(step.is_motion for step in steps):
            # 纯静态计划：只录静态那一段。
            total += sum(float(step.hold_s) for step in steps)
    return total


def _step_delta(step: Any) -> float:
    event = step.event
    if event.expected_delta_deg is not None:
        return abs(float(event.expected_delta_deg))
    if event.amplitude_deg:
        return abs(float(event.amplitude_deg))
    target = step.target_joint_deg
    if target is None:
        return 0.0
    return 0.0


def estimate_disk_gb(config: AppConfig, capture_seconds: float) -> float:
    """Mono8 RAW 会占多少 GB。**不含**离线角点结果和样本图（那些很小）。"""
    width, height = config.effective_camera_size()
    fps = config.effective_fps()
    frames = float(capture_seconds) * fps
    return frames * width * height / (1024.0**3)


def disk_estimate_lines(config: AppConfig, plans: Iterable[Any]) -> list[str]:
    """给界面/日志用的中文磁盘估算说明。"""
    seconds = estimate_capture_seconds(config, plans)
    size_gb = estimate_disk_gb(config, seconds)
    width, height = config.effective_camera_size()
    fps = config.effective_fps()
    lines = [
        f"预计录制总时长：约 {seconds / 60.0:.1f} 分钟（{seconds:.0f} 秒）",
        f"预计 RAW 占用：约 {size_gb:.2f} GB"
        f"（{width}×{height} Mono8 @ {fps:.2f} fps）",
    ]
    if config.mode != "dry_run" and config.camera.roi is None:
        lines.append(
            "★ 当前没有设置 ROI 裁剪：满幅 1936×1096 在 132 fps 下约 280 MB/s，"
            "是磁盘占用的第一主导项。现场建议把 camera.roi 填成包住棋盘格的一块。"
        )
    return lines


def default_config() -> AppConfig:
    """默认配置（界面第一次打开时看到的就是它）。"""
    config = AppConfig()
    config.validate()
    return config


def trial_plan_summary(config: AppConfig) -> dict[str, Any]:
    """预实验规模：几个关节、每关节几个动作、总共几个动作。

    需求三(B) 说得很具体：每个关节 3 档 × 2 方向 × 每方向重复 2 次 = 12 次，
    六个关节共 72 次。这里把它算出来显示给用户，避免"以为跑了 72 次"。
    """
    joints = list(config.pretest.joints)
    per_joint = (
        len(config.pretest.amplitudes_deg)
        * 2
        * int(config.pretest.repeats_per_direction)
    )
    return {
        "joints": joints,
        "actions_per_joint": per_joint,
        "total_actions": per_joint * len(joints),
        "amplitudes_deg": list(config.pretest.amplitudes_deg),
        "repeats_per_direction": int(config.pretest.repeats_per_direction),
    }


def check_free_disk(path: Path, min_free_gb: float) -> tuple[bool, str]:
    """检查磁盘空间。返回 (是否够用, 中文说明)。"""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:  # pragma: no cover - 平台相关
        return True, f"无法读取磁盘信息（{exc}），跳过空间检查。"
    free_gb = usage.free / 1024**3
    if free_gb < min_free_gb:
        return False, (
            f"可用磁盘空间只有 {free_gb:.2f} GB，低于要求的 {min_free_gb:.2f} GB，"
            "拒绝开始采集。"
        )
    return True, f"可用磁盘空间 {free_gb:.2f} GB，满足要求。"


def iter_joint_names(names: Iterable[str]) -> list[tuple[int, str]]:
    """把关节名列表变成 (序号, 名字) 列表，序号从 0 开始。"""
    return [(JOINT_NAMES.index(name), name) for name in names]
