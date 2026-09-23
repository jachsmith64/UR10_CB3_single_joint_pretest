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

    #: 默认值取自被复用的旧预实验代码（``vendor/config.py`` 的 ``ROBOT_HOST``），
    #: 那一套是在真机上跑通过的。现场若不是这个地址，请在界面的参数面板里改，
    #: 或者另存一份 ``configs/local_site.json``（该文件名已在 .gitignore 里）。
    ip: str = "192.168.125.12"
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
    #: 相机到棋盘格的**竖直**工作距离（mm）。默认 675 = 需求二给的摆放提示。
    #:
    #: ★ 它只用来把"棋盘格在画面里变大/变小了多少"换算成**轴向位移**：
    #: depth_mm ≈ working_distance_mm × |scale − 1|
    #: （小孔成像的一阶近似：Z' = Z·s，所以 ΔZ = Z·(s−1)）。
    #: 现场实际测一次相机到板面的距离填进来即可；填错只影响轴向位移的**比例**，
    #: 不影响"面内为主 / 有轴向分量"这个判断的方向（比例是单调的）。
    working_distance_mm: float = float(DEFAULT_CAMERA_HINT["working_distance_mm"])

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
        _positive(
            _as_float(self.working_distance_mm, f"{where}.working_distance_mm"),
            f"{where}.working_distance_mm",
        )

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
    #: 快速几何确认用的试探步长（度）。
    #: ★ 默认为 0.2 而不是 0.05：快速确认要同时判"面内位移够不够"和
    #: "轴向位移占比大不大"，0.05° 的位移在真机上只有几个像素，
    #: 轴向那一项根本分辨不出来，会一路报"无法判断"。
    #: 0.2° 是"仍属微动、但图像上信噪比够看"的折中，界面上可以改。
    quick_probe_deg: float = 0.2
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
    #: 一个窗口里至少要凑够多少**有效帧**才允许把它当成"均值"。
    #: 第二层的窗口是"保持段的后半段"，本来就不长（默认 0.4 s 的一半），
    #: 离线分析的步长一大，窗口里就剩一两帧——那时候报出来的数字不是均值，
    #: 是一次抽样。这一条**不是**放行/拦停的门槛（不参与推荐步长的判据），
    #: 而是提示"这个数不可信、请把步长调小重算"，所以不会把试验判成无效。
    min_window_frames: int = 5

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
        frames = _as_int(self.min_window_frames, f"{where}.min_window_frames")
        if frames < 2:
            raise ConfigError(
                f"{where}.min_window_frames 至少 2（1 帧算不出离散度），收到 {frames}"
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
    #: 正式实验用的速度/加速度。★ 故意比预实验更慢（预实验默认 0.5 °/s）：
    #: 组A 的行程是预实验的 N 倍（5 级阶梯 = 5×步长），运动本身激起的振动
    #: 更容易混进"几何"里，所以正式实验走得更慢。代价是慢——0.05 °/s 下
    #: 0.2° 的一级要走近 3 s，一个关节的组A 大约两分半。嫌慢可以在界面的
    #: 参数面板里改，上限仍是 ``robot.max_speed_deg_s``（5.0 °/s）；
    #: 改快了请自己承担"振动与几何分不开"的判读代价。
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

    #: ★ 边采边清（分组流水线）：一组动作全部采完之后、机械臂停着不动的时候，
    #: 把这一组**整组**处理掉，角点表与逐帧几何**先落盘并校验**，
    #: 校验通过才删掉这一组的 frames.raw，然后才进入下一组运动。
    #: 目的：让盘上任何时刻只有"当前这一组"的几十 GB 原始视频，
    #: 而不是整场几百 GB（整场 800×600 约 222 GB，950×800 约 352 GB）。
    #:
    #: 默认 False（保持"原始数据一律留盘"的交付行为）。
    #: 打开它换来的代价必须说清楚：RAW 删掉之后**只能**用保存下来的角点复算，
    #: 不能再换一套角点检测参数重跑识别；所以每段都会写一条 raw_deleted 事件，
    #: 记下删除的大小、帧数和 sha256，事后能证明"这一段处理过什么、删了什么"。
    #: 处理失败或校验不过时**不删**，宁可占盘，并且**暂停**等人处理。
    delete_raw_after_process: bool = False
    #: 就地处理这一遍的抽帧步长（1 = 每帧都识别）。
    #:
    #: ★ 默认 **1**：删 RAW 之前必须按 132 Hz 把**每一帧**的角点和视觉结果算出来并落盘。
    #: 一旦抽帧，被抽掉的那些帧的原始像素随 RAW 一起消失，事后**无法**补算——
    #: 那不是"省时间"，是永久丢数据。步长 > 1 只允许出现在**现场快速预览**这一类
    #: 不删数据的场合（见 ``experiment.QUICK_PROBE_STRIDE``）。
    #: 所以 ``delete_raw_after_process=True`` 时本项必须为 1，``validate`` 会硬拦。
    process_stride: int = 1
    #: 就地处理阶段的磁盘余量系数：处理是有中间产物的，别把盘顶死。
    process_headroom: float = 1.15
    #: 派生数据（88 角点 CSV + 逐帧 segment_vision.json + 元数据）相对 RAW 的比例。
    #: 现场实测约 1%：每帧 RAW 是 800×600=480000 字节，角点 88×~45 字节、
    #: 逐帧 JSON 约 400 字节。这里取 5% 是**故意往多了算**——留出写盘临时文件
    #: （.tmp）、事件日志、RTDE 状态流和样本图的空间，宁可早一点拦。
    derived_overhead_ratio: float = 0.05
    #: ★ 硬上限（GB）：盘上 RAW + 临时文件的总占用**任何时刻**都不得超过它。
    #: 超过就**不得开始下一组**——在一组开始之前按这一组的计划估算，超了直接拒绝。
    max_peak_disk_gb: float = 50.0
    #: ★ 预警线（GB）：估算超过它就放行但**明确预警**，让人先去清盘或裁 ROI。
    #:
    #: 现场要求 35～40 GB；取区间上限 40，理由是两个交付 ROI 档的**最坏一组**
    #: 都在它下面（800×600 约 22.4 GB、950×800 约 35.5 GB，见
    #: ``tests/test_group_pipeline.py`` 的实测断言）。取 35 的话 950×800 这一档
    #: 自己就会天天踩线报警，报警变成噪声就没人看了；取 40 则它只在"比交付
    #: 建议的 ROI 还大"时才响，正好是"该清盘或该裁 ROI 了"的时刻。
    disk_warn_gb: float = 40.0

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
        delete_raw = _as_bool(
            self.delete_raw_after_process, "paths.delete_raw_after_process"
        )
        stride = _as_int(self.process_stride, "paths.process_stride")
        if stride < 1:
            raise ConfigError(f"paths.process_stride 必须 ≥ 1，收到 {stride}")
        if delete_raw and stride != 1:
            # 需求四：stride>1 只能用于**不删数据**的现场快速预览。
            # 删 RAW 之前按步长抽帧 = 把没算的那几分之几帧的原始像素永久丢掉，
            # 事后既不能补算、也不能换检测参数重识别。这一条必须硬拦，
            # 因为它错起来是**静默**的：报告照样出，只是每 4 帧才有一帧。
            raise ConfigError(
                f"paths.delete_raw_after_process=true 时 paths.process_stride 必须为 1，"
                f"收到 {stride}。删 RAW 之前必须按 132 Hz 把每一帧的角点和视觉结果"
                "算出来落盘；抽帧只允许用在“不删数据”的现场快速预览上。"
                "（要么把 process_stride 改回 1，要么先把边采边清关掉。）"
            )
        headroom = _as_float(self.process_headroom, "paths.process_headroom")
        if headroom < 1.0:
            raise ConfigError(
                f"paths.process_headroom 必须 ≥ 1，收到 {headroom}（1.15 表示留 15% 余量）"
            )
        ratio = _non_negative(
            _as_float(self.derived_overhead_ratio, "paths.derived_overhead_ratio"),
            "paths.derived_overhead_ratio",
        )
        if ratio > 1.0:
            raise ConfigError(
                f"paths.derived_overhead_ratio = {ratio} 过大（派生数据不该超过 RAW 本身）。"
            )
        peak = _positive(
            _as_float(self.max_peak_disk_gb, "paths.max_peak_disk_gb"),
            "paths.max_peak_disk_gb",
        )
        warn = _non_negative(
            _as_float(self.disk_warn_gb, "paths.disk_warn_gb"), "paths.disk_warn_gb"
        )
        if warn > peak:
            raise ConfigError(
                f"paths.disk_warn_gb（{warn} GB）不能高于 paths.max_peak_disk_gb"
                f"（{peak} GB）：预警线画在硬上限外面就永远不会被触发。"
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
        root = Path(self.paths.output_root).expanduser()
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


#: 正式实验的阶段名。正式计划走的是 ``formal.speed_deg_s``，比预实验慢得多，
#: 所以估时长必须挑对速度——否则会把一小时估成几分钟，磁盘门槛就形同虚设。
#: 这里用字面量而不是从 joint_space 导入，是因为 joint_space 反过来要导入本模块。
_FORMAL_STAGES = ("formal_a", "formal_b")


def estimate_capture_seconds(config: AppConfig, plans: Iterable[Any]) -> float:
    """估算这些计划一共要录制多少秒（用来算磁盘占用）。

    规则：一个"去程 + 回程"算作一次试验，录一段连续的：运动前静止 → 去程 →
    保持 → 回程 → 运动后记录。纯等待步（组间等待）不录制。
    """
    durations = config.effective_durations()
    total = 0.0
    for plan in plans:
        steps = list(getattr(plan, "steps", plan))
        stage = next(
            (str(step.event.stage) for step in steps if step.is_motion), None
        )
        speed, accel = config.effective_speed(formal=stage in _FORMAL_STAGES)
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


def plan_disk_gb(config: AppConfig, plans: Iterable[Any]) -> tuple[float, float, float]:
    """返回 (总占用 GB, 单段最大占用 GB, 总时长 秒)。

    单段最大值是给"边采边清"用的：RAW 处理完就删的话，盘上任何时刻只有
    **当前这一段**，峰值不是总和。
    """
    plan_list = list(plans)
    seconds = estimate_capture_seconds(config, plan_list)
    total = estimate_disk_gb(config, seconds)
    per_plan = [
        estimate_disk_gb(config, estimate_capture_seconds(config, [plan]))
        for plan in plan_list
    ]
    return total, (max(per_plan) if per_plan else 0.0), seconds


def disk_estimate_lines(config: AppConfig, plans: Iterable[Any]) -> list[str]:
    """给界面/日志用的中文磁盘估算说明。"""
    total_gb, peak_gb, seconds = plan_disk_gb(config, plans)
    width, height = config.effective_camera_size()
    fps = config.effective_fps()
    lines = [
        f"预计录制总时长：约 {seconds / 60.0:.1f} 分钟（{seconds:.0f} 秒）",
        f"预计 RAW 占用：约 {total_gb:.2f} GB"
        f"（{width}×{height} Mono8 @ {fps:.2f} fps）",
    ]
    if config.mode != "dry_run" and config.camera.roi is None:
        lines.append(
            "★ 当前没有设置 ROI 裁剪：满幅 1936×1096 在 132 fps 下约 280 MB/s，"
            "是磁盘占用的第一主导项。现场建议把 camera.roi 填成包住棋盘格的一块。"
        )
    if config.paths.delete_raw_after_process:
        lines.append(
            f"★ 分组流水线已打开（paths.delete_raw_after_process=true）："
            f"一组动作全部走完、机械臂停稳之后，整组按步长 "
            f"{int(config.paths.process_stride)} 逐帧识别、落盘角点并**回读校验**，"
            f"校验通过才删除这一组的 RAW，然后才进下一组。"
            f"所以盘上同时只有当前这一组（单段最大约 {peak_gb:.2f} GB），"
            f"而不是整场的 {total_gb:.2f} GB。"
            f"硬上限 {config.paths.max_peak_disk_gb:.0f} GB、"
            f"预警线 {config.paths.disk_warn_gb:.0f} GB（见 check_group_disk）。"
        )
    else:
        lines.append(
            "（原始帧全程留盘：camera.save_raw 必须为 true，本工具不提供关闭选项。"
            "盘不够时请改用 paths.delete_raw_after_process 分组流水线，或裁小 ROI。）"
        )
    return lines


# --------------------------------------------------------------------------
# 分组流水线的磁盘估算（需求一）
# --------------------------------------------------------------------------


def segment_raw_gb(config: AppConfig, segment: Any) -> float:
    """一段采集的 RAW 会占多少 GB。

    ``segment`` 可以是 ``SegmentPlan``，也可以是任何有 ``steps`` 的对象
    （``SegmentPlan.steps`` 就是"这一段包含的那一到两个计划步"）。
    """
    return float(estimate_disk_gb(config, estimate_capture_seconds(config, [segment])))


def group_disk_gb(config: AppConfig, segments: Iterable[Any]) -> tuple[float, float]:
    """一组动作的占用，返回 ``(RAW GB, 含派生文件 GB)``。

    为什么一组要单独算：分组流水线下，盘上同时存在的是**当前这一组**的全部段
    （整组采完才处理、才删），而不是单段、也不是整场。组的划分见
    ``experiment.ExperimentSession`` 的 ``begin_group`` 调用点。
    """
    raw = sum(segment_raw_gb(config, segment) for segment in segments)
    ratio = float(config.paths.derived_overhead_ratio)
    return raw, raw * (1.0 + ratio)


def group_peak_gb(config: AppConfig, groups: Mapping[str, Iterable[Any]]) -> tuple[float, str, dict[str, float]]:
    """整场跑下来，盘上的峰值占用出现在哪一组。返回 ``(峰值 GB, 组名, 每组 GB)``。"""
    per_group = {
        str(name): group_disk_gb(config, segments)[1] for name, segments in groups.items()
    }
    if not per_group:
        return 0.0, "", {}
    name = max(per_group, key=lambda key: per_group[key])
    return per_group[name], name, per_group


def check_group_disk(
    config: AppConfig, segments: Iterable[Any], *, what: str = "本组动作"
) -> tuple[bool, list[str]]:
    """进入下一组动作之前的磁盘闸门（需求一）。

    三件事，按严重程度从小到大：

    1. **估算**：这一组的 RAW + 派生数据一共多少 GB。
    2. **预警**：超过 ``paths.disk_warn_gb``（默认 40 GB）就放行但明确预警——
       让操作者有机会先清盘，而不是等下一组被硬拦。
    3. **硬拦**：超过 ``paths.max_peak_disk_gb``（默认 50 GB）**或**可用空间不够，
       直接拒绝开始这一组。这一条是"不得开始下一组"的落地处。

    为什么用"这一组"而不是"这一段"：一组是流水线上"同时驻留"的单位，
    整组采完才处理，所以峰值就是这一组的总和。单段算会低估几十倍。

    ★ **硬上限只在分组流水线打开时才算。** ``max_peak_disk_gb`` 说的是
    "任意时刻盘上的 RAW + 临时文件"——只有"处理完就删"时才等于"当前这一组"。
    开关关着的时候 RAW 是**全程累积**的，拿 50 GB 去卡每一组会让默认配置
    根本跑不起来（整场就是 200～350 GB 的量级），那不是在保护数据，
    是在逼人关掉检查。所以那种情况下这里只查"这一组要写的量 + 绝对下限"
    够不够，并把估算如实报出来。
    """
    segment_list = list(segments)
    raw_gb, total_gb = group_disk_gb(config, segment_list)
    rolling = bool(config.paths.delete_raw_after_process)
    peak = float(config.paths.max_peak_disk_gb)
    warn = float(config.paths.disk_warn_gb)
    headroom = float(config.paths.process_headroom)
    need_gb = max(float(config.paths.min_free_disk_gb), total_gb * headroom)
    lines = [
        f"{what}：{len(segment_list)} 段，RAW 约 {raw_gb:.2f} GB，"
        f"含派生文件约 {total_gb:.2f} GB（派生按 RAW 的 "
        f"{float(config.paths.derived_overhead_ratio):.0%} 计）",
    ]
    if rolling:
        lines.append(f"硬上限 {peak:.0f} GB / 预警线 {warn:.0f} GB")
    else:
        lines.append(
            "（分组流水线未打开：RAW 全程累积，整场总和才是峰值，"
            "因此不拿单组硬上限来卡；本组估算仅供参考。）"
        )
    lines.append(
        f"按计划需要约 {need_gb:.2f} GB（含 {int((headroom - 1) * 100)}% 余量）"
    )
    if rolling and total_gb > peak:
        lines.append(
            f"磁盘检查未通过：本组估算 {total_gb:.2f} GB 超过硬上限 {peak:.0f} GB，"
            "**不得开始这一组**。可做的三件事：把 camera.roi 裁到刚好包住棋盘格；"
            "把这一组再拆小（例如减少 formal.staircase_n 或 repeats）；"
            "或换一个更大的输出盘。"
        )
        return False, lines
    ok, message = check_free_disk(config.resolve_output_root(), need_gb)
    if ok and rolling and total_gb > warn:
        lines.append(
            f"★ 磁盘预警：本组估算 {total_gb:.2f} GB 已超过预警线 {warn:.0f} GB"
            f"（硬上限 {peak:.0f} GB）。本次放行，但建议先清盘或裁小 ROI，"
            "否则再大一点就会被硬拦。"
        )
    if ok:
        lines.append(f"磁盘检查：{message}")
    else:
        lines.append(
            f"磁盘检查未通过：{message}"
            "可做的三件事：把 camera.roi 裁到刚好包住棋盘格；"
            "把这一组再拆小；或换一个更大的输出盘。"
        )
    return bool(ok), lines


def check_plan_disk(
    config: AppConfig,
    plans: Iterable[Any],
    *,
    headroom: float = 1.15,
) -> tuple[bool, list[str]]:
    """按**这一次要跑的计划**算占用，再和可用空间比。返回 (是否够, 中文说明)。

    为什么不能只看 ``paths.min_free_disk_gb``：那是个固定门槛（默认 2 GB），
    而真实预实验 + 正式实验的 RAW 是几十到几百 GB 的量级——满幅 1936×1096
    在 132 fps 下约 280 MB/s，跑一小时就是约 1 TB。固定门槛定在 2 GB，
    等于没有门槛：磁盘写满的时候人已经离开一小时了，采到的数据还是断的。

    所以这里按计划算"接下来这一段要多少"，乘一个余量（默认 15%），
    再和"绝对下限"取较大者。不够就返回 False，由调用方**拒绝开始**这一段。
    """
    plan_list = list(plans)
    lines = disk_estimate_lines(config, plan_list)
    total_gb, peak_gb, _seconds = plan_disk_gb(config, plan_list)
    # 边采边清时，盘上只要放得下"最大的那一段"，因为上一段在处理完就被删了；
    # 但**不能**因此把门槛降到 0——正在录的那一段仍然要一次写完。
    size_gb = peak_gb if config.paths.delete_raw_after_process else total_gb
    what = "最大的一段" if config.paths.delete_raw_after_process else "本段计划"
    need_gb = max(float(config.paths.min_free_disk_gb), size_gb * float(headroom))
    ok, message = check_free_disk(config.resolve_output_root(), need_gb)
    if ok:
        lines.append(
            f"磁盘检查：{message}（{what}按计划需要约 {need_gb:.2f} GB，"
            f"含 {int((headroom - 1) * 100)}% 余量）"
        )
    else:
        lines.append(
            f"磁盘检查未通过：{message}"
            f"{what}按计划需要约 {need_gb:.2f} GB。"
            "可做的三件事：把 camera.roi 裁到刚好包住棋盘格；"
            "或打开 paths.delete_raw_after_process（边采边清：每段处理完就删 RAW，"
            "盘上只留当前这一段）；或换一个更大的输出盘。"
        )
    return ok, lines


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
