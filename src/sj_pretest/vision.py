"""离线视觉：实验跑完之后再做的角点识别与二维运动量提取。

在线线程一个角点都不算（需求六），所以这里是**唯一**做图像处理的地方。

按需求四，两套算法分开、不混用
------------------------------
* **J1～J5：二维质心沿局部方向的标量投影。**
  88 个内角点的质心位移是一个二维向量；再投影到"这个关节自己的局部图像运动方向"上，
  得到一个标量。需求明确说了这只是"二维质心沿局部方向的标量投影，
  不是高维角点特征投影"，代码里也只做这一件事。
* **J6：88 个角点的二维 Kabsch 刚体旋转（绕质心）。**
  绕质心做正交 Procrustes，取旋转矩阵的角度。**J6 绝不走平移通道**——
  J6 绕自身轴线转，棋盘格中心几乎不动，用平移会得到噪声。

灵敏度（px/度）
---------------
优先用 ``视觉位移 ÷ RTDE 实际关节角变化``；当 RTDE 那一档变化小到分不出来时，
退回用 ``÷ 指令角变化``，并在结果里**明确标注退化了**（需求四要求）。

角点检测本身复用被复用代码的 ``CheckerboardTracker``：棋盘格规格、SB/传统两级检测、
亚像素细化、rigid motion 估计都是原来那套，没有重写。本模块只是在它返回的
88 个角点上再做质心与 Kabsch 两件小事。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import AppConfig
from .recorder import write_csv
from .vendor_shim import vendor_module


class VisionError(RuntimeError):
    """离线视觉出错。消息中文。"""


#: 几何基准（位移/转角/尺度的零点）用静止阶段里多少帧的平均值来定。
#: 见 :func:`_reference_points`：单帧作基准会把那一帧的识别噪声灌进整段的每一个
#: 几何量，而且不会被平均掉。取 12 帧能把基准噪声压到单帧的约 1/3.5，
#: 同时仍然只占第一个静止窗口的一小部分（不会跨到运动里去）。
REFERENCE_FRAMES = 12


# --------------------------------------------------------------------------
# 单帧结果
# --------------------------------------------------------------------------


@dataclass
class FrameVision:
    """一帧的离线识别结果。"""

    frame_id: int
    host_ns: int
    analysis_time_s: float
    valid: bool
    corner_count: int
    #: 88 个角点的质心（像素）。无效帧为 None。
    centroid_x_px: float | None
    centroid_y_px: float | None
    #: 相对本段参考帧的二维转角（度，绕质心 Kabsch）。无效帧为 None。
    rotation_deg: float | None
    #: Kabsch 拟合残差（像素）与质量分。质量分来自被复用代码的 rigid motion 估计。
    residual_px: float | None
    quality: float | None
    #: 被复用代码估计的毫米/像素比例（由角点间距算出）。
    mm_per_pixel: float | None
    #: 相对参考帧的整体平移（像素），直接由质心差算，不经过被复用代码的 dx/dy。
    shift_x_px: float | None
    shift_y_px: float | None
    #: ★ 相对本段参考帧的**相似变换尺度**（1.0 = 和参考帧一样大）。
    #: 它由 Kabsch 顺带算出（最小二乘最优缩放），是棋盘格在画面里"变大/变小"的
    #: 直接度量，也就是相机与板之间**轴向**距离变化的一阶观测量：
    #:     深度变化 ≈ 工作距离 × |scale − 1|
    #: 以前这个量算出来就被丢掉了（``_scale``），轴向运动因此完全没被检查。
    scale: float | None
    #: 图像质量指标（来自 preprocess_frame）。
    mean_brightness: float | None
    blur_variance: float | None
    #: ★ 本帧 88 个内角点到自身质心的**均方根半径**（像素）。
    #:
    #: 这是 J6 面内位移公式里的 ``r_rms``。J6 绕自身轴转，棋盘格**不保证**
    #: 装在旋转轴中心（现场可能偏心 5～15 mm），所以面内运动量不能只看质心平移：
    #: 质心平移只反映"偏心量 × 转角"，板自己转了多少一点没进去。
    #: 绕轴转 θ 时，距轴 r 的一点在画面里扫过的弧长是 2·r·sin(|θ|/2)；
    #: 板有大小，取"转动惯量"意义下的等效半径（均方根半径）：
    #:     r_rms = √( mean_i |p_i − 质心|² )
    #: 参考帧的角点在本段内是常量，所以这个值每帧一样；存下来是为了
    #: RAW 删掉之后仍能从 JSON 复算面内位移，不依赖角点 CSV 是否还在。
    #: （带默认值，所以放在所有无默认值字段的后面。）
    r_rms_px: float | None = None
    drop_before: int = 0
    #: 这一帧的角点坐标，写 CSV 用；不写时留 None 以省内存。
    points_px: np.ndarray | None = None

    def to_row(self, segment_id: str) -> dict[str, Any]:
        return {
            "segment_id": segment_id,
            "frame_id": self.frame_id,
            "host_ns": self.host_ns,
            "analysis_time_s": _fmt(self.analysis_time_s, 6),
            "valid": int(self.valid),
            "corner_count": self.corner_count,
            "centroid_x_px": _fmt(self.centroid_x_px, 4),
            "centroid_y_px": _fmt(self.centroid_y_px, 4),
            "shift_x_px": _fmt(self.shift_x_px, 4),
            "shift_y_px": _fmt(self.shift_y_px, 4),
            "rotation_deg": _fmt(self.rotation_deg, 5),
            "residual_px": _fmt(self.residual_px, 4),
            "scale": _fmt(self.scale, 7),
            "r_rms_px": _fmt(self.r_rms_px, 4),
            "quality": _fmt(self.quality, 4),
            "mm_per_pixel": _fmt(self.mm_per_pixel, 6),
            "mean_brightness": _fmt(self.mean_brightness, 3),
            "blur_variance": _fmt(self.blur_variance, 3),
            "drop_before": int(self.drop_before),
        }

    def to_json_dict(self) -> dict[str, Any]:
        """不带角点的存盘形式（角点单独存 CSV，见 :func:`save_vision_json`）。"""
        return {
            "frame_id": int(self.frame_id),
            "host_ns": int(self.host_ns),
            "analysis_time_s": float(self.analysis_time_s),
            "valid": bool(self.valid),
            "corner_count": int(self.corner_count),
            "centroid_x_px": self.centroid_x_px,
            "centroid_y_px": self.centroid_y_px,
            "shift_x_px": self.shift_x_px,
            "shift_y_px": self.shift_y_px,
            "rotation_deg": self.rotation_deg,
            "residual_px": self.residual_px,
            "scale": self.scale,
            "r_rms_px": self.r_rms_px,
            "quality": self.quality,
            "mm_per_pixel": self.mm_per_pixel,
            "mean_brightness": self.mean_brightness,
            "blur_variance": self.blur_variance,
            "drop_before": int(self.drop_before),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "FrameVision":
        frame = cls(
            frame_id=int(payload["frame_id"]),
            host_ns=int(payload["host_ns"]),
            analysis_time_s=float(payload["analysis_time_s"]),
            valid=bool(payload["valid"]),
            corner_count=int(payload["corner_count"]),
            centroid_x_px=payload.get("centroid_x_px"),
            centroid_y_px=payload.get("centroid_y_px"),
            rotation_deg=payload.get("rotation_deg"),
            residual_px=payload.get("residual_px"),
            scale=payload.get("scale"),
            r_rms_px=payload.get("r_rms_px"),
            quality=payload.get("quality"),
            mm_per_pixel=payload.get("mm_per_pixel"),
            shift_x_px=payload.get("shift_x_px"),
            shift_y_px=payload.get("shift_y_px"),
            mean_brightness=payload.get("mean_brightness"),
            blur_variance=payload.get("blur_variance"),
            drop_before=int(payload.get("drop_before", 0) or 0),
        )
        return frame


@dataclass
class SegmentVision:
    """一段采集的全部逐帧结果。"""

    segment_id: str
    segment_dir: Path
    frames: list[FrameVision] = field(default_factory=list)
    #: 用的哪一帧当参考（本段第一帧有效帧）。
    reference_frame_id: int | None = None
    #: ★ v1.0.3：几何基准（位移/转角/尺度的零点）是**多少帧的平均**（见
    #: :func:`_reference_points`）。这不是装饰性字段：深度可分辨下限要按
    #: "窗口均值 − 基准均值"这个**估计量**的误差棒算，基准用了多少帧直接
    #: 决定下限（见 :func:`estimate_depth_in_plane`）。老数据没有这个字段，
    #: 按 1 帧算——那时基准确实就是单帧。
    reference_frames: int = 1
    mm_per_pixel: float | None = None
    #: 采集层写下的相位时间表（``capture_metadata.json`` 的 ``phases``）。
    #: 分析层靠它在本段自己的时间轴上切"运动前/保持/回程"窗口。
    phases: list[dict[str, Any]] = field(default_factory=list)
    #: **这一段离线处理算了多久**（wall clock），不是"这一段录了多久"。
    #: ★ 名字容易被误读，所以任何"面向实验报告的时长"都不许用它——
    #: 用下面的 :attr:`captured_seconds`。见 ``analysis.StaticNoise``。
    seconds: float = 0.0
    #: ★ v1.0.3：**这一段 RAW 实际录了多久**（``capture_metadata.json`` 的
    #: ``content_seconds``）。报告里凡是要说"录了多久"的地方一律用它。
    #: 早先 ``StaticNoise.seconds`` 直接拿上面的 ``seconds``（处理耗时），
    #: 报告里于是出现"静态基线 3.0 s 的采集写成 14.5 s"——采集时长被写成了
    #: 算力耗时，同一场上换个处理步长这个数还会变。
    captured_seconds: float = 0.0
    #: ★ v1.0.3：这一段的 RAW 里一共有多少帧（**采集**帧数，不是参与统计的
    #: 处理帧数）。``process_stride > 1`` 时两者不同，报告必须分开写。
    captured_frames: int = 0
    #: 这段是不是合成数据。
    synthetic: bool = False
    #: ★ v1.0.3：离线分析用的 ROI（``(x, y, w, h)``），None = 整幅。
    analysis_roi: tuple[int, int, int, int] | None = None
    #: ★ 分析窗口左上角在**原始整幅帧**里的位置。把 ROI 内坐标加回它就得到原图坐标。
    roi_origin: tuple[int, int] = (0, 0)
    #: ★ 本段所有角点/质心坐标属于哪个坐标系。
    #: v1.0.3 起固定为 ``"full_image"``——角点在存盘之前已经加回 ``roi_origin``，
    #: 所以这一份数据不论当时有没有配 ROI，坐标口径都一样，可以互相比较。
    #: 这个字段必须写进 JSON：将来有人拿不同批次的角点 CSV 拼在一起算，
    #: 没有它就无法判断两组坐标是不是同一个基准。
    corners_frame: str = "full_image"

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def valid_count(self) -> int:
        return sum(1 for frame in self.frames if frame.valid)

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.frame_count if self.frame_count else 0.0

    def times(self) -> np.ndarray:
        return np.array([frame.analysis_time_s for frame in self.frames], dtype=float)

    def centroids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (time, x, y)，只含有有效帧。"""
        valid = [frame for frame in self.frames if frame.valid]
        return (
            np.array([frame.analysis_time_s for frame in valid], dtype=float),
            np.array([float(frame.centroid_x_px) for frame in valid], dtype=float),
            np.array([float(frame.centroid_y_px) for frame in valid], dtype=float),
        )

    def rotations(self) -> tuple[np.ndarray, np.ndarray]:
        valid = [
            frame
            for frame in self.frames
            if frame.valid and frame.rotation_deg is not None
        ]
        return (
            np.array([frame.analysis_time_s for frame in valid], dtype=float),
            np.array([float(frame.rotation_deg) for frame in valid], dtype=float),
        )

    def window(self, start_s: float, end_s: float) -> list[FrameVision]:
        return [
            frame
            for frame in self.frames
            if start_s <= frame.analysis_time_s <= end_s and frame.valid
        ]

    def subsampled(self, step: int) -> "SegmentVision":
        """按 ``step`` 抽帧，返回**新的一份**（不动自己）。

        ★ 这是"RAW 已经删了，但下游要求按 step 抽帧看"的唯一正路。

        ``process_segment(..., stride=step)`` 保留的是原始采集里下标
        ``0, step, 2·step, …`` 的帧（见那里的 ``index % step``），参考帧是第一帧——
        所以对一份 stride=1 的**完整**逐帧结果做同样的抽取，得到的帧集合、
        参考帧、每一帧的转角/尺度/位移都和"直接对 RAW 抽帧跑一遍"逐位相同。
        这条性质是必需的：边采边清之后 RAW 没了，任何"RAW 在"与"RAW 不在"
        两条路径给出的结论必须一致，否则就等于拿数据换了磁盘。

        抽帧只允许用在**已经完整算过一遍**的数据上（含内存里刚算的那一份）；
        它不会、也不能凭空补出没算过的帧。
        """
        count = max(1, int(step))
        if count == 1:
            return self
        picked = list(self.frames[::count])
        return SegmentVision(
            segment_id=self.segment_id,
            segment_dir=self.segment_dir,
            frames=picked,
            reference_frame_id=self.reference_frame_id,
            # 抽帧只是少看几帧，基准还是原来那几帧的平均——照抄，不要重算。
            reference_frames=self.reference_frames,
            mm_per_pixel=self.mm_per_pixel,
            phases=list(self.phases),
            seconds=self.seconds,
            # ★ 采集口径照抄，**不随抽帧变小**：抽帧少的只是"参与统计的帧"，
            # 这一段 RAW 依然录了这么多帧、这么久。抽帧之后报告还是得写
            # "录了 397 帧 / 3.00 s"，不然"少看几帧"就变成了"少录了"。
            captured_seconds=self.captured_seconds,
            captured_frames=self.captured_frames,
            synthetic=self.synthetic,
            analysis_roi=self.analysis_roi,
            roi_origin=self.roi_origin,
            corners_frame=self.corners_frame,
        )


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


# --------------------------------------------------------------------------
# 二维 Kabsch：绕质心的刚体旋转
# --------------------------------------------------------------------------


def kabsch_rotation_about_centroid(
    reference_points: np.ndarray, current_points: np.ndarray
) -> tuple[float, float, float]:
    """两组一一对应的二维点之间的刚体旋转。

    返回 ``(转角_deg, 残差_px, 尺度)``。转角是"把参考点转到当前点"所需的角度，
    绕各自质心计算，所以**平移不影响它**——这正是 J6 需要的性质。
    尺度只作为质量检查（刚体假设下应当接近 1），不参与转角计算。
    """
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    if ref.shape != cur.shape or len(ref) < 3:
        raise VisionError(
            f"Kabsch 需要两组一一对应且不少于 3 个的点，收到 {ref.shape} 与 {cur.shape}。"
        )
    ref_centered = ref - ref.mean(axis=0)
    cur_centered = cur - cur.mean(axis=0)
    covariance = ref_centered.T @ cur_centered
    u_matrix, singular, vt_matrix = np.linalg.svd(covariance)
    # 二维刚体旋转：允许反射时取伪旋转，这里强制 det=+1，不允许镜像。
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix = vt_matrix.copy()
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    angle_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    predicted = ref_centered @ rotation.T
    residual = float(np.sqrt(np.mean(np.sum((predicted - cur_centered) ** 2, axis=1))))
    # 尺度：最小二乘最优缩放 = <cur, predicted> / <predicted, predicted>。
    denominator = float(np.sum(predicted**2))
    scale = float(np.sum(predicted * cur_centered) / denominator) if denominator > 0 else 1.0
    return angle_deg, residual, scale


def rms_radius_px(points: np.ndarray) -> float:
    """一组点到它们自身质心的均方根半径（像素）。

    ``√( mean_i |p_i − 质心|² )``。用均方根半径而不是最大半径或平均半径：
    它正是"绕质心转动"在均方意义下的等效半径——转动惯量意义下，板有大小这件事
    只能用一个等效半径表达，而均方根半径给出的就是转动量的均方根。
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) == 0:
        return 0.0
    centered = array - array.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def rotation_in_plane_px(rotation_deg: float | None, r_rms_px: float | None) -> float:
    """J6 转动在画面里扫出的等效面内位移（像素）。

    ``d_rot = 2 · r_rms · sin(|θ|/2)``

    这是"板上离轴最远的那些点在均方意义下扫过多远"：绕轴转 θ 的点，其弦长是
    ``2·r·sin(θ/2)``。取弦长而不是弧长，是因为画面里量到的就是**首末位置的位移**，
    不是轨迹长度。

    为什么不能只看质心平移：J6 绕自身轴转，棋盘格如果恰好装在轴中心，
    质心**一点都不动**（平移是 0，而画面明明在转）；偏心 5～15 mm 时质心才动一点，
    而且动多少取决于偏心量——那是安装公差，不是运动量。所以转动量必须单独算，
    再和质心平移**合成**，见 :func:`in_plane_px_of_frame`。
    """
    if rotation_deg is None or r_rms_px is None:
        return 0.0
    half_rad = math.radians(abs(float(rotation_deg))) / 2.0
    return float(2.0 * float(r_rms_px) * math.sin(half_rad))


def in_plane_px_of_frame(frame: FrameVision, *, rotation_joint: bool) -> float:
    """一帧相对参考帧的**面内位移**（像素）。

    * J1～J5：就是质心二维位移的模。
    * J6：把质心平移和**转动扫过的位移**合成——
      ``d_plane = √(质心位移² + d_rot²)``。两项都算进去之后，
      棋盘格居中（质心位移 0）和偏心 15 mm（质心位移不为 0）两种装法
      都会被算成同一个量级的面内运动，判据不再跟着安装公差跑。
    """
    shift = math.hypot(float(frame.shift_x_px or 0.0), float(frame.shift_y_px or 0.0))
    if not rotation_joint:
        return float(shift)
    return float(math.hypot(shift, rotation_in_plane_px(frame.rotation_deg, frame.r_rms_px)))


def project_onto_direction(
    shift_x_px: float, shift_y_px: float, direction_deg: float
) -> float:
    """把二维位移投影到给定方向上，得到一个标量（像素，可正可负）。"""
    radians = math.radians(float(direction_deg))
    return float(shift_x_px) * math.cos(radians) + float(shift_y_px) * math.sin(radians)


def direction_angle_deg(shift_x_px: float, shift_y_px: float, *, previous: float | None) -> float:
    """把二维位移向量变成一个角度。

    ``previous`` 给出时，结果会被折到"与 previous 相差不超过 180°"的那一支，
    这样同一档步长的多次重复不会因为点积为负而跳变 180°。
    """
    angle = math.degrees(math.atan2(float(shift_y_px), float(shift_x_px)))
    if previous is None:
        return angle
    while angle - previous > 180.0:
        angle -= 360.0
    while angle - previous < -180.0:
        angle += 360.0
    return angle


# --------------------------------------------------------------------------
# 逐段离线识别
# --------------------------------------------------------------------------


def process_segment(
    segment_dir: Path,
    *,
    config: AppConfig,
    segment_id: str | None = None,
    save_corners: bool = True,
    corners_dir: Path | None = None,
    metrics_path: Path | None = None,
    vision_json_path: Path | None = None,
    raw_available_after: bool = True,
    progress: Any = None,
    stop_requested: Any = None,
    stride: int = 1,
) -> SegmentVision:
    """对一段采集做离线角点识别。

    输入是采集目录（frames.raw + frame_timestamps.csv），输出是逐帧结果。
    棋盘格检测与亚像素细化全部走被复用代码，本函数只追加质心和 Kabsch 两件事。

    ``stride > 1`` 时只处理每隔 ``stride`` 帧的一帧。这是给**现场快速几何检查**
    用的：识别一帧约 150 ms，整段跑完要几十秒，操作者等不起；抽样跑一次能在一两秒
    内回答"棋盘格还在不在、画面动没动、方向对不对"，而正式分析仍然用 ``stride=1``
    把每一帧都算一遍。抽掉的帧不是"跳过"，是**留到正式分析再算**——这一点写在这里，
    避免以后有人以为抽样结果就是最终结果。

    ★ **要删 RAW 的路径必须 stride=1。** 抽帧处理之后若把 frames.raw 删掉，
    被抽掉的那些帧的像素就永久没有了，"留到正式分析再算"这条退路随之消失。
    所以生产路径（边采边清、离线分析）一律 stride=1，stride>1 只用于
    **不删数据的现场预览**；``config.validate`` 会把两者的组合硬拦下来。

    ★ **离线分析 ROI 只在这里生效**（v1.0.3 需求一·2）。实现方式是把 ROI 交给
    被复用代码自己的 ``config.VISION_ROI``（它是 ``preprocess_frame`` 支持的
    官方入口，vendor 一个字节都不用改），于是"减少背景干扰"这件事由被复用代码
    完成，本模块不自己裁图。识别完之后**把 ``roi_origin`` 加回每一个角点**，
    所以本模块对外给出的坐标一律是**原图坐标**，和 RAW 整幅保存的口径一致。
    """
    camera = vendor_module("camera")
    vendor_config = vendor_module("config")
    segment_id = segment_id or segment_dir.name
    capture = camera.RawCaptureSource(segment_dir)
    tracker = camera.CheckerboardTracker()
    result = SegmentVision(segment_id=segment_id, segment_dir=segment_dir)
    metadata = _read_metadata(segment_dir)
    result.synthetic = bool(metadata.get("synthetic", False))
    # 相位表原样带过来：它和本段的帧时间轴是同一套时间，分析层不用再读文件。
    result.phases = list(metadata.get("phases", []) or [])
    # ★ 采集口径（录了多久、一共几帧）：取自采集层写的 metadata，**不是**处理耗时。
    # 找不到（老段目录没这个字段）就退回 0，由下游用"处理帧数"兜底并标注。
    result.captured_seconds = _maybe_float(metadata.get("content_seconds")) or 0.0
    result.captured_frames = int(metadata.get("frame_count") or 0)

    # ★ 离线分析 ROI：整幅时为 None（整幅保存的 RAW 全都会被识别）。
    roi = config.camera.resolved_analysis_roi()
    result.analysis_roi = tuple(int(v) for v in roi) if roi is not None else None

    corner_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    min_corners = int(config.vision.min_corners)

    step = max(1, int(stride))
    # 记住进来之前的全局值，处理完原样还回去——这是个模块级全局量，
    # 不还原的话会悄悄影响同一个进程里后面跑的别的段。
    previous_vision_roi = getattr(vendor_config, "VISION_ROI", None)
    vendor_config.VISION_ROI = (
        (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])) if roi is not None else None
    )
    try:
        _process_frames(
            capture,
            camera=camera,
            tracker=tracker,
            result=result,
            config=config,
            step=step,
            min_corners=min_corners,
            segment_id=segment_id,
            progress=progress,
            stop_requested=stop_requested,
        )
    finally:
        vendor_config.VISION_ROI = previous_vision_roi

    result.seconds = time.perf_counter() - started

    # 以本段的**静止基准**（见 _reference_points）为参考，补上相对它的位移与 Kabsch 转角。
    reference, baseline_clouds = _reference_points(result, tracker)
    if reference is not None:
        # ★ 几何零点 = 基准窗口里若干帧的平均。注意实现方式：**先在"单帧参考"
        # 上把每一帧的位移/转角/尺度算出来，再把这一批基准帧自己的平均值减掉**，
        # 而不是把 12 帧角点平均成一个虚构的参考点云。
        # 两者对"均值"的效果一样（基准噪声都降到 1/√n），但前者保持了一个
        # 必须保住的性质：**转角差是可加的**——
        #     Kabsch(参考→末帧) − Kabsch(参考→首帧) ≡ Kabsch(首帧→末帧)
        # 用虚构点云当参考时这个恒等式只在数值上近似成立（实测差 3e-6°），
        # 而"删掉 RAW 之后能用角点 CSV 把转角复算回来"正是靠它：
        # 复算路径手里只有真实帧的角点，造不出那个虚构点云。
        # 基准帧数记下来：深度可分辨下限按"窗口均值 − 基准均值"这个估计量的
        # 误差棒算（见 estimate_depth_in_plane），基准帧数进了公式。
        result.reference_frames = max(1, len(baseline_clouds) or 1)
        # ★ 参考点也要在**原图坐标**里（见 _reference_points）：只给当前帧加 ROI 偏移、
        # 不给参考点加的话，两边的基准差一个偏移，算出来的位移会正好等于 ROI 左上角
        # 坐标——一个巨大的假位移。
        reference_centroid = reference.mean(axis=0)
        zero_shift, zero_angle, zero_scale = _baseline_zero_point(
            reference, reference_centroid, baseline_clouds
        )
        for frame in result.frames:
            if not frame.valid or frame.points_px is None:
                continue
            centroid = frame.points_px.mean(axis=0)
            frame.shift_x_px = float(centroid[0] - reference_centroid[0]) - zero_shift[0]
            frame.shift_y_px = float(centroid[1] - reference_centroid[1]) - zero_shift[1]
            try:
                angle, residual, scale = kabsch_rotation_about_centroid(
                    reference, frame.points_px
                )
                frame.rotation_deg = float(angle) - zero_angle
                frame.residual_px = residual
                # ★ 尺度留着，不再丢掉：轴向运动检查全靠它（需求四）。
                frame.scale = float(scale) / zero_scale
            except VisionError:
                frame.rotation_deg = None
                frame.residual_px = None
                frame.scale = None

    if save_corners:
        corner_rows, metric_rows = _collect_rows(result)
        if corners_dir is not None:
            from .recorder import safe_name

            write_corners(corners_dir / f"{safe_name(segment_id)}.csv", corner_rows)
        if metrics_path is not None:
            append_metrics(metrics_path, metric_rows)
    if vision_json_path is not None:
        # 逐帧结果单独存一份：这是"边采边清"删掉 RAW 之后唯一能复算几何量的依据。
        # raw_available_after 由调用方给：只有边采边清那条路径处理完会删 RAW。
        save_vision_json(
            Path(vision_json_path), result, stride=step, raw_available=raw_available_after
        )

    return result


def _process_frames(
    capture: Any,
    *,
    camera: Any,
    tracker: Any,
    result: SegmentVision,
    config: AppConfig,
    step: int,
    min_corners: int,
    segment_id: str,
    progress: Any = None,
    stop_requested: Any = None,
) -> None:
    """逐帧识别的主循环（从 :func:`process_segment` 抽出来，只为让 ROI 的
    "设进去 / 还回来" 用 ``try/finally`` 包得干净，不改变任何算法）。

    ``camera.preprocess_frame`` 已经按 ``config.VISION_ROI`` 裁好了，
    所以这里拿到的 ``corners`` 是 **ROI 局部坐标**；立刻加回 ``roi_origin``，
    之后本函数往 ``result.frames`` 里写的每一个坐标都是**原图坐标**。
    """
    with capture:
        for index, packet in enumerate(capture):
            if step > 1 and index % step:
                continue
            try:
                gray, metrics, roi_origin, _timing = camera.preprocess_frame(packet.frame)
            except ValueError as exc:
                # 被复用代码对"ROI 越界"抛的是 ValueError。这里换成中文的 VisionError，
                # 并且把真实的帧尺寸报出来——离线处理时才越界，多半是换了相机分辨率
                # 或者 ROI 是照着另一台机器的画面填的。
                raise VisionError(
                    f"{segment_id}：离线分析 ROI 和这段 RAW 的实际帧尺寸对不上（{exc}）。"
                    "请按实际分辨率重填 camera.analysis_roi，或留空按整幅分析。"
                ) from exc
            result.roi_origin = (int(roi_origin[0]), int(roi_origin[1]))
            offset = np.asarray(result.roi_origin, dtype=np.float64)
            checker, corners, _timing = tracker.process(gray)
            frame = FrameVision(
                frame_id=int(packet.frame_id),
                host_ns=int(packet.host_ns),
                analysis_time_s=float(packet.analysis_time_s or 0.0),
                valid=False,
                corner_count=0,
                centroid_x_px=None,
                centroid_y_px=None,
                rotation_deg=None,
                residual_px=None,
                scale=None,
                quality=None,
                mm_per_pixel=None,
                shift_x_px=None,
                shift_y_px=None,
                mean_brightness=float(metrics.get("mean_brightness", float("nan"))),
                blur_variance=float(metrics.get("blur_variance", float("nan"))),
                drop_before=int(getattr(packet, "missing_before", 0) or 0),
            )
            if corners is not None:
                points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                # ★ 加回 ROI 偏移：本模块对外只给原图坐标（见 SegmentVision.corners_frame）。
                points = points + offset
                frame.corner_count = int(len(points))
                if frame.corner_count >= min_corners:
                    centroid = points.mean(axis=0)
                    frame.centroid_x_px = float(centroid[0])
                    frame.centroid_y_px = float(centroid[1])
                    frame.valid = True
                    frame.points_px = points
                    # 角点到自身质心的均方根半径（J6 面内位移要用，见 FrameVision）。
                    frame.r_rms_px = rms_radius_px(points)
                    if result.reference_frame_id is None:
                        result.reference_frame_id = frame.frame_id
                        # mm_per_pixel 直接取被复用代码算好的那个值，
                        # 保证"像素→毫米"这一步和原来的口径完全一致。
                        result.mm_per_pixel = _maybe_float(
                            getattr(tracker, "mm_per_pixel", None)
                        )
            frame.mm_per_pixel = result.mm_per_pixel
            frame.quality = _maybe_float(checker.get("checker_quality"))
            result.frames.append(frame)

            if progress is not None and len(result.frames) % 200 == 0:
                progress(result.segment_id, len(result.frames))
            if stop_requested is not None and stop_requested():
                raise VisionError(
                    f"{segment_id}：离线识别被中止（已完成 {len(result.frames)} 帧）。"
                )


def _reference_points(
    result: SegmentVision, tracker: Any
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """本段的几何基准：单帧参考角点（**原图坐标**），以及用来定零点的若干帧角点。

    ★ v1.0.3：零点取"第一个静止阶段里有效帧的**平均**"，而不是单帧。

    为什么非改不可：以前直接用 ``CheckerboardTracker`` 记下的"本段第一帧"角点。
    单帧作基准的问题不在于它偏了多少，而在于**这一帧的识别噪声会变成整段所有
    几何量的公共偏移**——位移、Kabsch 转角、尺度全都是相对它算出来的，
    于是"这一帧的角点偏了 0.02°"会原样出现在整段的每一个转角里，
    而且因为它对每一帧都一样，**不会被平均掉**。J6 的轴向/面内判据正好卡在
    分辨边界上（0.2° 的探针本来就只比噪声大一档），基准噪声因此直接决定了
    判据是"看得出来"还是"看不出来"——同一个动作、同一台设备，换一次采集
    （帧号变了、噪声实现就变了）结论就可能反过来。现场表现就是快速几何检查
    时而通过时而不通过，而这正是它最不该有的行为。

    棋盘格在第一个静止窗口（``pre_motion``）里保证没动，把那个窗口的头若干帧
    平均起来当零点，基准噪声降到 1/√n，整段的位移/转角/尺度口径同时变准、变稳。

    实现上返**两样**东西：单帧的参考角点 + 用来定零点的那几帧角点。
    真正的"减掉平均值"在 :func:`_baseline_zero_point` 里做（在位移/转角/尺度
    这些**派生量**上做，而不是把角点平均成一个虚构点云），原因见那里的说明。

    退回规则：没有相位表（历史数据、回放）或静止窗口里一帧都没识别出来时，
    仍然用被复用代码记下的第一帧、并且不做零点平移——和有相位表时是同一个
    坐标系（都已经是加回过 ROI 偏移的原图坐标），不会把老数据读歪。
    退回时基准帧数是 1，调用方（:func:`estimate_depth_in_plane`）会据此把
    可分辨下限放宽回去。
    """
    baseline_clouds = _stationary_reference_frames(result)

    reference = getattr(tracker, "reference_corners", None)
    if reference is not None:
        points = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
        if len(points) >= 3:
            # 被复用代码给的是 ROI 局部坐标，加回偏移才是原图坐标。
            return points + np.asarray(result.roi_origin, dtype=np.float64), baseline_clouds

    if not baseline_clouds:
        return None, []
    # 拿不到被复用代码记下的那一帧时（老数据、回放），就用基准窗口的第一帧当
    # 单帧参考——零点随后由这一批帧的均值定出来，效果等价。
    return np.asarray(baseline_clouds[0], dtype=np.float64), baseline_clouds


def _baseline_zero_point(
    reference: np.ndarray,
    reference_centroid: np.ndarray,
    baseline_clouds: Sequence[np.ndarray],
) -> tuple[tuple[float, float], float, float]:
    """基准窗口的"零点"，在位移/转角/尺度这三个**派生量**上取平均。

    为什么不在角点空间里平均：把 12 帧角点平均成一个点云，再拿它当参考，
    得到的位移/尺度与"逐帧算完再取平均"是一样的（质心平均、尺度平均都线性），
    但**转角不一样**——转角的可加性
    ``Kabsch(参考→末) − Kabsch(参考→首) ≡ Kabsch(首→末)``
    只在参考是**真实帧**时严格成立。删掉 RAW 之后，复算路径手里只有角点 CSV
    里的真实帧，造不出虚构点云；一旦基准是虚构点云，这条恒等式就只剩数值近似
    （实测差 3e-6°），"从 CSV 复算转角"的精度会跟着变差。
    所以在派生量上平移零点是既等价又更保真的做法。

    返回 ``((零点位移 x, y), 零点转角, 零点尺度)``；没有基准帧时返回"不平移"。
    """
    if not baseline_clouds:
        return (0.0, 0.0), 0.0, 1.0
    shifts_x: list[float] = []
    shifts_y: list[float] = []
    angles: list[float] = []
    scales: list[float] = []
    for cloud in baseline_clouds:
        points = np.asarray(cloud, dtype=np.float64)
        if points.shape != np.asarray(reference).shape:
            # 角点数对不上就不能比：Kabsch 需要一一对应的点。
            continue
        centroid = points.mean(axis=0)
        shifts_x.append(float(centroid[0] - reference_centroid[0]))
        shifts_y.append(float(centroid[1] - reference_centroid[1]))
        try:
            angle, _residual, scale = kabsch_rotation_about_centroid(reference, points)
        except VisionError:
            continue
        angles.append(float(angle))
        scales.append(float(scale))
    if not shifts_x:
        return (0.0, 0.0), 0.0, 1.0
    mean_scale = float(np.mean(scales)) if scales else 1.0
    if abs(mean_scale) < 1e-12:
        # 尺度平均到 0 是不物理的（说明数据坏了）；这种情况下不平移，宁可保守。
        mean_scale = 1.0
    return (
        (float(np.mean(shifts_x)), float(np.mean(shifts_y))),
        float(np.mean(angles)) if angles else 0.0,
        mean_scale,
    )


def _stationary_reference_frames(result: SegmentVision) -> list[np.ndarray]:
    """基准用的若干帧角点（都要求**角点数相同**才敢平均）。

    取第一个标着 ``pre_motion`` 的阶段；没有相位表（历史数据）时退回"本段头几帧"。
    退回也是安全的：本工具采的每一段都是"先在名义位姿上静一会儿再动"，
    静态基线更是整段都不动——所以段首那几帧一定是静止的。
    """
    from .analysis import PHASE_PRE  # 局部导入：analysis 依赖本模块，顶部导入会成环

    bounds: tuple[float, float] | None = None
    for entry in result.phases:
        if entry.get("label") != PHASE_PRE:
            continue
        try:
            bounds = (float(entry["start_s"]), float(entry["end_s"]))
        except (KeyError, TypeError, ValueError):
            bounds = None
        break

    picked: list[np.ndarray] = []
    count: int | None = None
    for frame in result.frames:
        if not frame.valid or frame.points_px is None:
            continue
        if bounds is not None and not (
            bounds[0] <= float(frame.analysis_time_s) <= bounds[1]
        ):
            continue
        points = np.asarray(frame.points_px, dtype=np.float64)
        if count is None:
            count = len(points)
        elif len(points) != count:
            # 角点数不同的帧不能并进平均：那说明棋盘格被挡了一部分，
            # 硬平均出来的是一个不存在的"角点"。
            continue
        picked.append(points)
        if len(picked) >= REFERENCE_FRAMES:
            break
    return picked


def _collect_rows(
    result: SegmentVision,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corner_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for frame in result.frames:
        metric_rows.append(frame.to_row(result.segment_id))
        if frame.valid and frame.points_px is not None:
            for index, (x_coord, y_coord) in enumerate(frame.points_px):
                corner_rows.append(
                    {
                        "segment_id": result.segment_id,
                        "frame_id": frame.frame_id,
                        "host_ns": frame.host_ns,
                        "analysis_time_s": _fmt(frame.analysis_time_s, 6),
                        "corner_index": index,
                        # 6 位小数（1e-6 px），不是 4 位：角点 CSV 是"RAW 删掉之后
                        # 复算几何量"的**唯一**观测量来源，4 位小数会在 200 px 力臂上
                        # 留下约 3e-6° 的转角误差——比它要复现的那个量（0.2° 探针的
                        # 图像转角约 6e-3°）小得多，但仍然是可以避免的。
                        # 6 位之后从 CSV 复算的转角与存下来的值在 1e-6° 内一致。
                        "x_px": _fmt(float(x_coord), 6),
                        "y_px": _fmt(float(y_coord), 6),
                    }
                )
    return corner_rows, metric_rows


def save_vision_json(
    path: Path, result: SegmentVision, *, stride: int = 1, raw_available: bool = True
) -> Path:
    """把这一段的逐帧结果存成 JSON（**不含角点**，角点单独存 CSV）。

    这是"边采边清"能成立的关键：RAW 删掉之后，要复算这一段的视觉量，
    靠的就是这份 JSON（逐帧质心/尺度/转角）+ 角点 CSV（原始观测量）。
    两者加起来足以重算位移、转角和三层判据；但**不能**换一套角点检测
    参数重新识别——那需要原始像素。这件事在 ``stride`` 和事件里都要写清楚。

    ``raw_available`` 说的是**这份文件写完之后**段目录里还有没有 ``frames.raw``，
    必须由调用方如实告诉它：只有"边采边清"那条路径才会删 RAW，
    离线分析和回放路径写这份 JSON 的时候 RAW 明明还在。写死成 False
    会让这些路径的 JSON 里躺着一句"RAW 已删除"的假话——将来有人拿它判断
    "这段还能不能重新识别像素"，就会得出错的结论。

    ★ **原子写**（v1.0.3 需求一·6）：先写同目录下的临时文件，``fsync`` 之后
    ``os.replace`` 覆盖。分组流水线要**改写两次**同一个文件（先 ``true``、
    删完 RAW 再改成 ``false``），如果直接 ``write_text`` 覆盖，进程在写到一半时
    被 Ctrl-C 或断电，留下的是半截 JSON——既没有 ``raw_available=true`` 的旧值，
    也没有 ``false`` 的新值，这一段的"RAW 还在不在"就彻底说不清了。
    ``os.replace`` 在同一文件系统内是原子的，任何时刻读到的要么是旧文件、
    要么是新文件，不会读到中间态。
    """
    import json
    import os
    import tempfile

    if raw_available:
        note = (
            "本文件是这一段的逐帧结果（含质心、二维转角、尺度、r_rms、时间戳）。"
            "段目录里的 frames.raw 仍在，换角点检测参数重识别随时可以重来；"
            "本文件只是省掉重复识别，不是唯一副本。"
            f"（处理步长 {int(stride)}；生产路径固定为 1，即每一帧都算过。）"
        )
    else:
        note = (
            "本文件由“分组流水线”就地处理生成：本组校验通过后 RAW 已删除。"
            "逐帧几何量可直接复算（含质心、二维转角、尺度、r_rms、时间戳）；"
            "若要换角点检测参数重识别，需要原始帧，本段已不可得"
            f"（当时按步长 {int(stride)} 处理；生产路径固定为 1，即每一帧都算过）。"
        )
    payload = {
        "segment_id": result.segment_id,
        "segment_dir": str(result.segment_dir),
        "reference_frame_id": result.reference_frame_id,
        # ★ 基准是几帧的平均：复算深度可分辨下限要用它（见 estimate_depth_in_plane）。
        # 老 JSON 没有这一项，读回时按 1 帧算——那时的基准确实就是单帧。
        "reference_frames": int(result.reference_frames),
        "mm_per_pixel": result.mm_per_pixel,
        # ★ 两个"时长"必须分开写，别让下游拿处理耗时当采集时长：
        # seconds = 本段离线处理算了多久；captured_seconds = 本段 RAW 录了多久。
        "seconds": float(result.seconds),
        "captured_seconds": float(result.captured_seconds),
        "captured_frames": int(result.captured_frames),
        "synthetic": bool(result.synthetic),
        "phases": list(result.phases),
        "processed_stride": int(stride),
        "raw_available": bool(raw_available),
        # ★ 坐标系与 ROI 口径：这三项和角点 CSV 一起，才构成"这批坐标能不能和
        # 别的批次比"的完整说明。角点一律是原图坐标（见 SegmentVision.corners_frame）。
        "analysis_roi": list(result.analysis_roi) if result.analysis_roi else None,
        "roi_origin": [int(result.roi_origin[0]), int(result.roi_origin[1])],
        "corners_frame": result.corners_frame,
        "note": note,
        "frames": [frame.to_json_dict() for frame in result.frames],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_segment_vision(
    segment_dir: Path,
    *,
    segment_id: str | None = None,
    corners_path: Path | None = None,
) -> SegmentVision:
    """从"边采边清"留下的 JSON（+角点 CSV）重建一段的视觉结果。

    用途：RAW 被删掉之后再跑离线分析。没有 RAW 也没有 JSON 就明确报错，
    **不返回空结果**——空结果会被下游当成"这段没动"，那是科研事故。
    """
    import json

    segment_dir = Path(segment_dir)
    name = segment_id or segment_dir.name
    json_path = segment_dir / VISION_JSON_NAME
    if not json_path.is_file():
        raise VisionError(
            f"{name}：既没有原始帧也没有逐帧结果（缺 {VISION_JSON_NAME}），无法复算。"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    result = SegmentVision(
        segment_id=str(payload.get("segment_id") or name),
        segment_dir=segment_dir,
    )
    result.reference_frame_id = payload.get("reference_frame_id")
    # 基准帧数：缺字段（老 JSON）按 1 帧，也就是"基准就是单帧"那个更保守的口径。
    try:
        result.reference_frames = max(1, int(payload.get("reference_frames") or 1))
    except (TypeError, ValueError):
        result.reference_frames = 1
    result.mm_per_pixel = payload.get("mm_per_pixel")
    result.seconds = float(payload.get("seconds") or 0.0)
    # ★ 采集口径：新 JSON 直接有；老 JSON 没有这两项，就从段目录的
    # capture_metadata.json 补回来（它一直在，删 RAW 不会删它）。
    metadata = _read_metadata(segment_dir)
    result.captured_seconds = float(
        payload.get("captured_seconds")
        or _maybe_float(metadata.get("content_seconds"))
        or 0.0
    )
    result.captured_frames = int(
        payload.get("captured_frames") or metadata.get("frame_count") or 0
    )
    result.synthetic = bool(payload.get("synthetic", False))
    result.phases = list(payload.get("phases") or [])
    # ROI 口径一并读回：它决定这批角点/质心坐标是原图坐标还是某个窗口的局部坐标。
    # 缺字段（老 JSON）按"整幅、原图坐标、偏移 0"处理——老版本存的就是整幅坐标，
    # 这个默认值对它们恰好是对的，不会把老数据读歪。
    roi = payload.get("analysis_roi")
    result.analysis_roi = tuple(int(v) for v in roi) if roi else None
    origin = payload.get("roi_origin") or [0, 0]
    result.roi_origin = (int(origin[0]), int(origin[1]))
    result.corners_frame = str(payload.get("corners_frame") or "full_image")
    result.frames = [
        FrameVision.from_json_dict(item) for item in payload.get("frames") or []
    ]

    points = read_corners(corners_path) if corners_path is not None else {}
    if points:
        for frame in result.frames:
            frame.points_px = points.get(int(frame.frame_id))
    return result


#: 段目录里那份逐帧结果的固定文件名（"边采边清"要能按名找到它）。
VISION_JSON_NAME = "segment_vision.json"


def read_corners(path: Path | None) -> dict[int, np.ndarray]:
    """读回角点 CSV，返回 ``{frame_id: 角点数组}``。文件不在就返回空字典。"""
    import csv

    if path is None or not Path(path).is_file():
        return {}
    grouped: dict[int, list[tuple[int, float, float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_id = int(row["frame_id"])
                index = int(row["corner_index"])
                x_coord = float(row["x_px"])
                y_coord = float(row["y_px"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(frame_id, []).append((index, x_coord, y_coord))
    out: dict[int, np.ndarray] = {}
    for frame_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        out[frame_id] = np.asarray(
            [(x_coord, y_coord) for _index, x_coord, y_coord in items], dtype=np.float64
        )
    return out


def write_corners(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns = (
        "segment_id",
        "frame_id",
        "host_ns",
        "analysis_time_s",
        "corner_index",
        "x_px",
        "y_px",
    )
    return write_csv(path, rows, columns=columns)


def append_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """把逐帧指标追加到总表。表头只在文件不存在时写一次。"""
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    columns = (
        "segment_id",
        "frame_id",
        "host_ns",
        "analysis_time_s",
        "valid",
        "corner_count",
        "centroid_x_px",
        "centroid_y_px",
        "shift_x_px",
        "shift_y_px",
        "rotation_deg",
        "residual_px",
        # ★ scale 与 r_rms_px 以前没列进来，于是 DictWriter 的 extrasaction="ignore"
        # 把它们**静默**丢掉了：metrics.csv 里根本没有尺度列，而轴向/面内判据
        # 恰恰要用它。列名写在这里，少一列就该看得见。
        "scale",
        "r_rms_px",
        "quality",
        "mm_per_pixel",
        "mean_brightness",
        "blur_variance",
        "drop_before",
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def _read_metadata(segment_dir: Path) -> dict[str, Any]:
    import json

    path = segment_dir / "capture_metadata.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _maybe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------
# 基于摆放提示的"理论方向"（只作方向核对，不是精确标定）
# --------------------------------------------------------------------------


#: 理论方向"无意义"时提示语里的标记词。
#: 判断方向是否要跟理论比对时必须先看这个标记——否则会拿一个占位的 0.0°
#: 当真方向去比对，把一个根本不产生面内位移的关节判成"方向错误"。
#: 调用方（experiment.py 的快速几何检查）用 ``in`` 检查这个标记。
THEORY_DIRECTION_MEANINGLESS_MARK = "无意义"


def expected_image_direction_deg(
    joint_index: int,
    nominal_joint_deg: Sequence[float],
    *,
    config: AppConfig,
) -> tuple[float, str]:
    """按名义运动学和现场摆放提示，粗算"这个关节正向动 1° 时画面该往哪走"。

    **这不是精确相机标定。** 用到的只是需求二给的那几条摆放提示
    （水平放置、工作距离约 675 mm、相机与棋盘格中心大致等高、水平方位角约 33.3°）。
    它的用途只有一个：需求三 B 要求"方向是否大致符合理论"，
    这里给出一个可比较的角度，再在结果里显示两者相差多少度。

    做法：假设棋盘格固定在相机光轴附近的工作空间中心，取相机在基座坐标系里的
    粗略位姿（按方位角摆一圈），算出关节正向微动时 TCP 的位移方向，
    再投到相机的图像平面上（图像 x 向右、y 向下）。
    """
    from .config import DEFAULT_CAMERA_HINT

    if not 0 <= joint_index < 6:
        raise VisionError(f"关节序号必须在 0～5，收到 {joint_index}")

    hint = DEFAULT_CAMERA_HINT
    azimuth = math.radians(float(hint["azimuth_deg"]))
    # 只要**方向**，不要量级，所以用不到工作距离：距离只影响"多少像素/度"，
    # 不影响"往哪个方向动"。这一点在报告里也说明过（不假装是精确标定）。
    delta = 0.5  # 度；取小量做差分，避免名义运动学的非线性混进来
    start = list(nominal_joint_deg)
    plus = list(start)
    minus = list(start)
    plus[joint_index] += delta
    minus[joint_index] -= delta
    from .kinematics import flange_position_m

    displacement = (flange_position_m(plus) - flange_position_m(minus)) * 1000.0
    # 相机在水平面内朝内看：图像向右 = 世界水平面上与视线垂直的右手方向；
    # 图像向下 = 世界 −z（图像 y 轴朝下）。
    right = np.array([math.cos(azimuth), math.sin(azimuth)], dtype=float)
    image_x = float(displacement[0] * right[0] + displacement[1] * right[1])
    image_y = -float(displacement[2])
    if abs(image_x) < 1e-12 and abs(image_y) < 1e-12:
        return (
            0.0,
            f"该关节在此姿态下对画面中心几乎不产生面内位移"
            f"（预测方向{THEORY_DIRECTION_MEANINGLESS_MARK}）",
        )
    angle = math.degrees(math.atan2(image_y, image_x))
    note = (
        "基于现场摆放提示（水平、约 675 mm、方位角约 33.3°）的粗略方向预测，"
        "不是相机标定结果；只在判断“方向是否大致符合理论”时作参考。"
    )
    return angle, note


def in_plane_ratio_from_scale_change(
    scale_change: float, working_distance_mm: float, in_plane_px: float, mm_per_pixel: float
) -> float:
    """用"棋盘格看起来变大/变小"估计深度位移，再和面内位移比一比。

    一阶近似：深度位移 ≈ 工作距离 × 相对尺度变化。这只是量级估计，
    所以结果只用来判断"这次运动是不是主要在面内"，不当作位移测量值。
    """
    depth_mm = abs(float(scale_change)) * float(working_distance_mm)
    in_plane_mm = abs(float(in_plane_px)) * float(mm_per_pixel)
    if in_plane_mm <= 1e-9:
        return float("inf") if depth_mm > 1e-9 else 0.0
    return depth_mm / in_plane_mm


# --------------------------------------------------------------------------
# 轴向 / 面内运动判别（需求四）
# --------------------------------------------------------------------------

#: 判据里用的置信状态。三种都要能出现在报告里，不能只留"通过/不通过"。
CONFIDENCE_RESOLVED = "resolved"
CONFIDENCE_BELOW_RESOLUTION = "below_resolution"
CONFIDENCE_UNAVAILABLE = "unavailable"

#: 尺度分辨率的经验系数：分辨率取"参考窗口内尺度散布"的若干倍。
#: 取 3 是常规的"三倍标准差"，低于它的尺度变化不可与噪声区分。
SCALE_RESOLUTION_SIGMA = 3.0


@dataclass
class DepthInPlane:
    """一次运动的轴向 / 面内分解结果。"""

    #: 深度位移估计（mm）。置信状态为 below_resolution 时是 None
    #: （那时只能给上限，见 ``depth_limit_mm``）。
    depth_mm: float | None
    #: 面内位移（mm），由质心二维位移 × 现场 mm/px 得到。
    in_plane_mm: float | None
    #: depth / in_plane。置信状态下才是实测比值；否则是 None。
    ratio: float | None
    #: 深度位移的**上限**（mm）：把"分辨不出来"的那一部分按最坏情况算进去。
    depth_limit_mm: float | None
    #: 用上限算出来的比值——判据就用它，因为它永远不会低估深度。
    ratio_upper: float | None
    #: 有效帧数（参与统计的帧）。
    frames: int
    #: 实测相对尺度（1.0 = 与参考帧同大）。
    scale: float | None
    #: 尺度噪声水平（单位与 scale 相同）。低于它就说"分辨不出来"。
    scale_noise: float | None
    #: 上面那个噪声对应的深度（mm）。
    resolution_mm: float | None
    #: resolved / below_resolution / unavailable
    confidence: str = CONFIDENCE_UNAVAILABLE
    note: str = ""
    #: ★ v1.0.3：几何基准是几帧的平均（``SegmentVision.reference_frames``）。
    #: 它进了可分辨下限的公式，所以必须跟着结果一起报出来，否则这个下限
    #: 事后没法复核——"下限为什么是 0.0415 而不是 0.135"要能一眼看出来。
    reference_frames: int = 1

    @property
    def decided(self) -> bool:
        return self.confidence == CONFIDENCE_RESOLVED

    @property
    def judgement_text(self) -> str:
        """一句话结论（报告和日志共用，措辞保持一致）。"""
        if self.confidence == CONFIDENCE_UNAVAILABLE:
            return f"无法判断（{self.note}）"
        if self.confidence == CONFIDENCE_BELOW_RESOLUTION:
            return (
                f"深度变化低于可分辨下限（{self.resolution_mm:.4f} mm），"
                f"无法判断实际深度，只能给上限 {self.depth_limit_mm:.4f} mm"
            )
        return f"深度 {self.depth_mm:.4f} mm，面内 {self.in_plane_mm:.4f} mm"

    def to_lines(self) -> list[str]:
        lines = [
            f"轴向（深度）位移：{self._fmt_or(self.depth_mm, '低于可分辨下限')} mm",
            f"面内位移：{self._fmt_or(self.in_plane_mm)} mm",
            f"深度/面内 比值：{self._fmt_or(self.ratio)}"
            + ("" if self.decided else f"（上限 {self._fmt_or(self.ratio_upper)}）"),
            f"有效帧数：{self.frames}",
            f"置信状态：{self.confidence}（{self.judgement_text}）",
        ]
        if self.scale is not None and self.scale_noise is not None:
            lines.append(
                f"实测尺度：{self.scale:.7f}，尺度噪声 ±{self.scale_noise:.7f}"
                f"（≈ {self._fmt_or(self.resolution_mm)} mm 深度）"
            )
            lines.append(
                f"可分辨下限口径：{SCALE_RESOLUTION_SIGMA:g}σ·√(1/N运动 + 1/N基准)，"
                f"N运动={int(self.frames)}、N基准={int(self.reference_frames)}"
            )
        return lines

    @staticmethod
    def _fmt_or(value: float | None, fallback: str = "无法判断") -> str:
        return fallback if value is None else f"{float(value):.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_mm": self.depth_mm,
            "in_plane_mm": self.in_plane_mm,
            "ratio": self.ratio,
            "depth_limit_mm": self.depth_limit_mm,
            "ratio_upper": self.ratio_upper,
            "frames": int(self.frames),
            "scale": self.scale,
            "scale_noise": self.scale_noise,
            "resolution_mm": self.resolution_mm,
            "confidence": self.confidence,
            "note": self.note,
            "reference_frames": int(self.reference_frames),
        }


def scale_noise_level(frames: Sequence[FrameVision]) -> float | None:
    """从一组**静止**帧估尺度噪声：尺度散布的标准差。

    为什么不用一个拍脑袋的常数：尺度噪声取决于棋盘格在画面里有多大、角点定位
    精度多高（半像素级），换一块板子、换一个 ROI 就全变了。用同一段数据自己算，
    判据才跟着现场走。
    """
    values = [
        float(frame.scale)
        for frame in frames
        if frame.valid and frame.scale is not None
    ]
    if len(values) < 3:
        return None
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def estimate_depth_in_plane(
    frames: Sequence[FrameVision],
    *,
    config: AppConfig,
    noise_frames: Sequence[FrameVision] | None = None,
    rotation_joint: bool = False,
    reference_frames: int = 1,
) -> DepthInPlane:
    """把一段运动里的逐帧结果分解成"轴向位移 / 面内位移"（需求四）。

    口径（三条都要写清楚，否则数字没法复核）：

    1. **深度**：``depth_mm = working_distance_mm × |scale − 1|``。
       小孔成像的一阶近似——物体离相机近 s 倍，成像就大 s 倍。
       这是**量级估计**，不当作位移测量值用。
    2. **面内**：``in_plane_mm = d_plane_px × mm_per_pixel``，
       其中 ``d_plane_px`` 按关节分两种（见 :func:`in_plane_px_of_frame`）：

       * J1～J5：``|质心二维位移|``；
       * **J6**：``√(质心二维位移² + (2·r_rms·sin(|θ|/2))²)``。
         J6 绕自身轴转，棋盘格可能**偏心** 5～15 mm，所以既不能只看质心平移
         （棋盘格装得越正、质心越不动，运动反而被判成 0），也不能假设它居中
         （偏心量是安装公差，不该进运动量）。两项合成之后，
         "居中"和"偏心 15 mm"的纯转动会得到同一个量级的面内位移。

       mm_per_pixel 用被复用代码由角点间距算出的那个值，和"像素→毫米"的旧口径一致。
       **J6 的角度本身仍然只用 88 角点去质心后的二维 Kabsch 旋转**，
       绝不用"质心圆弧位移 ÷ 假定偏心距离"反推——那样量到的是安装偏心，
       不是关节转角。
    3. **可分辨性**：尺度噪声取 ``noise_frames``（静止段）里尺度的标准差 σ，
       分辨率 = ``SCALE_RESOLUTION_SIGMA × σ × √(1/N运动 + 1/N基准)``。
       若窗口均值 |mean(scale)−1| 小于它，就**不报实测深度**，
       而是报"低于可分辨下限 + 一个上限"：

           depth_limit_mm = working_distance_mm × max(|scale−1|, 分辨率)

       判据（:attr:`DepthInPlane.ratio_upper`）永远用这个上限，
       所以"分辨不出来"**永远不会**被当成"深度很小、放心通过"——
       只有当上限本身都远小于面内位移时，才敢说"以面内运动为主"。
       如果上限已经大到压不住，就如实报 ``below_resolution`` 并由调用方暂停。

       ★ v1.0.3：那个 ``√(1/N运动 + 1/N基准)`` 因子不是可选的修饰，而是
       **估计量自己的误差棒**。深度这个量是

           d = mean(本窗口的 scale) − mean(基准的 scale)

       两项都是**均值**：前者用本窗口 N运动 帧，后者用基准 N基准 帧
       （见 :attr:`SegmentVision.reference_frames`）。两批帧互相独立，
       各自的标准误是 σ/√N，所以 d 的标准误是 ``σ·√(1/N运动 + 1/N基准)``。
       三倍标准差的分辨下限就该按它算。

       以前直接用 σ（也就是把"单帧"当估计量），在 N运动=52、N基准=12 时
       把下限放大了 3.1 倍：真实能分辨的深度被报成"分辨不出来"，
       于是判据只能拿一个偏大的上限去比面内位移，**同一个物理状态**会
       因为"这次看了多少帧、基准用了单帧还是多帧"而时过时不过。
       注意：判据门限 ``thresholds.max_depth_ratio`` 一个字都没动，
       这里改的是"下限本身算得对不对"，不是"多少算通过"。
    """
    signal = [
        frame for frame in frames if frame.valid and frame.centroid_x_px is not None
    ]
    result = DepthInPlane(
        depth_mm=None,
        in_plane_mm=None,
        ratio=None,
        depth_limit_mm=None,
        ratio_upper=None,
        frames=len(signal),
        scale=None,
        scale_noise=None,
        resolution_mm=None,
    )
    working_distance = float(config.camera.working_distance_mm)
    if not signal:
        result.note = "这一段没有有效帧"
        return result

    # 运动量取"窗口内相对参考帧的最大位移"：微动是一去一回的动作，
    # 用末帧差值会被回程抵消掉，用最大值才代表这次运动到底走了多少。
    # ★ J6 走"平移 + 转动合成"，见 in_plane_px_of_frame：棋盘格居中时质心不动，
    # 只看平移会把一次真实的转动量成 0。
    in_plane_px = max(
        in_plane_px_of_frame(frame, rotation_joint=rotation_joint) for frame in signal
    )
    mm_per_pixel = None
    for frame in signal:
        if frame.mm_per_pixel:
            mm_per_pixel = float(frame.mm_per_pixel)
            break
    if mm_per_pixel is None:
        mm_per_pixel = 0.0
        result.note = "没有 mm/像素 标定值（棋盘格识别没给出格距），面内位移按 0 计"
    result.in_plane_mm = float(in_plane_px * mm_per_pixel)

    scales = [float(frame.scale) for frame in signal if frame.scale is not None]
    if not scales:
        result.note = (result.note + "；" if result.note else "") + "这一段算不出尺度"
        return result
    # ★ v1.0.3：深度取窗口内的**均值**，不取"偏离 1 最远的那一帧"。
    # 运动量那一句（用最大值）说的是**面内位移**：微动是一去一回的，只有最大值
    # 才代表这次到底走了多少。深度不一样——这个窗口是"走完停住"的保持段，
    # 真实尺度变化在窗口里是同一个常数，所以它的正确估计量是均值。
    # 取最大值等于"在 N 帧噪声里挑最偏的那一帧"：N 越大挑到的越偏（52 帧时
    # 能挑到约 2.5σ），同一个物理量会因为"看了多少帧"得出不同的深度。
    # 快速几何检查看的是逐帧结果（52 帧），抽样预览看的是每 8 帧一帧（7 帧），
    # 两条路本来必须结论一致，取最大值就做不到。
    scale = float(np.mean(scales))
    result.scale = float(scale)
    deviation = abs(float(scale) - 1.0)

    noise_source = [
        frame
        for frame in (noise_frames or [])
        if frame.valid and frame.scale is not None
    ]
    result.scale_noise = scale_noise_level(noise_source) if len(noise_source) >= 3 else None
    if result.scale_noise is None:
        # 没有静止段可以参考，就不能声称自己分辨得出深度——如实说。
        result.confidence = CONFIDENCE_UNAVAILABLE
        result.note = "本段没有可用的静止参考帧，尺度噪声未知，无法判断"
        return result

    # ★ v1.0.3：估计量是"窗口均值 − 基准均值"，误差棒按两个均值各自的
    # 标准误合成（见上面 docstring 第 3 条）。基准帧数由 process_segment 落盘，
    # 复算时从 JSON 读回来；老数据/单帧基准时它等于 1，因子退化成
    # √(1/N运动 + 1)，也就是"基准那一帧的噪声照单全收"的旧口径——保守方向。
    signal_frames = max(1, len(scales))
    reference_frame_count = max(1, int(reference_frames))
    result.reference_frames = reference_frame_count
    standard_error = math.sqrt(
        1.0 / float(signal_frames) + 1.0 / float(reference_frame_count)
    )
    resolution = (
        float(SCALE_RESOLUTION_SIGMA) * float(result.scale_noise) * standard_error
    )
    result.resolution_mm = float(resolution * working_distance)
    resolved = deviation >= resolution

    if resolved:
        result.confidence = CONFIDENCE_RESOLVED
        result.depth_mm = float(deviation * working_distance)
        result.depth_limit_mm = result.depth_mm
    else:
        result.confidence = CONFIDENCE_BELOW_RESOLUTION
        result.depth_limit_mm = float(resolution * working_distance)
    result.ratio_upper = _safe_ratio(result.depth_limit_mm, result.in_plane_mm)
    if resolved:
        result.ratio = _safe_ratio(result.depth_mm, result.in_plane_mm)
    if result.in_plane_mm <= 1e-9:
        result.note = "面内位移为 0（画面没动），比值无意义"
    return result


def _safe_ratio(depth_mm: float | None, in_plane_mm: float | None) -> float | None:
    if depth_mm is None or in_plane_mm is None:
        return None
    if in_plane_mm <= 1e-9:
        return float("inf") if depth_mm > 1e-9 else 0.0
    return float(depth_mm / in_plane_mm)


def judge_in_plane_dominant(
    estimate: DepthInPlane, *, max_depth_ratio: float
) -> tuple[bool, str]:
    """按 ``thresholds.max_depth_ratio`` 放行或暂停。返回 (是否放行, 中文原因)。

    判据用的是**上限**比值：宁可把深度算大，也不许因为"分辨不出来"而漏过轴向运动。
    """
    limit = float(max_depth_ratio)
    if estimate.ratio_upper is None:
        return False, f"无法判断（{estimate.note or '缺少数据'}）"
    if estimate.ratio_upper <= limit:
        if estimate.confidence == CONFIDENCE_RESOLVED:
            return True, (
                f"深度/面内 = {estimate.ratio_upper:.3f} ≤ {limit}"
                f"（深度 {estimate.depth_mm:.4f} mm，面内 {estimate.in_plane_mm:.4f} mm）"
            )
        return True, (
            f"深度变化低于可分辨下限，无法判断实际深度；"
            f"即使按上限 {estimate.depth_limit_mm:.4f} mm 算，深度/面内 = "
            f"{estimate.ratio_upper:.3f} 仍 ≤ {limit}，认定以面内运动为主"
        )
    return False, (
        f"深度/面内 = {estimate.ratio_upper:.3f} > {limit}，轴向分量偏大："
        f"{estimate.judgement_text}"
    )
