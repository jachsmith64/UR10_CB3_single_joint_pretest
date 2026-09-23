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
    #: 图像质量指标（来自 preprocess_frame）。
    mean_brightness: float | None
    blur_variance: float | None
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
            "quality": _fmt(self.quality, 4),
            "mm_per_pixel": _fmt(self.mm_per_pixel, 6),
            "mean_brightness": _fmt(self.mean_brightness, 3),
            "blur_variance": _fmt(self.blur_variance, 3),
            "drop_before": int(self.drop_before),
        }


@dataclass
class SegmentVision:
    """一段采集的全部逐帧结果。"""

    segment_id: str
    segment_dir: Path
    frames: list[FrameVision] = field(default_factory=list)
    #: 用的哪一帧当参考（本段第一帧有效帧）。
    reference_frame_id: int | None = None
    mm_per_pixel: float | None = None
    #: 采集层写下的相位时间表（``capture_metadata.json`` 的 ``phases``）。
    #: 分析层靠它在本段自己的时间轴上切"运动前/保持/回程"窗口。
    phases: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    #: 这段是不是合成数据。
    synthetic: bool = False

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
    """
    camera = vendor_module("camera")
    segment_id = segment_id or segment_dir.name
    capture = camera.RawCaptureSource(segment_dir)
    tracker = camera.CheckerboardTracker()
    result = SegmentVision(segment_id=segment_id, segment_dir=segment_dir)
    metadata = _read_metadata(segment_dir)
    result.synthetic = bool(metadata.get("synthetic", False))
    # 相位表原样带过来：它和本段的帧时间轴是同一套时间，分析层不用再读文件。
    result.phases = list(metadata.get("phases", []) or [])

    corner_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    min_corners = int(config.vision.min_corners)

    step = max(1, int(stride))
    with capture:
        for index, packet in enumerate(capture):
            if step > 1 and index % step:
                continue
            gray, metrics, _roi_origin, _timing = camera.preprocess_frame(packet.frame)
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
                frame.corner_count = int(len(points))
                if frame.corner_count >= min_corners:
                    centroid = points.mean(axis=0)
                    frame.centroid_x_px = float(centroid[0])
                    frame.centroid_y_px = float(centroid[1])
                    frame.valid = True
                    frame.points_px = points
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

    result.seconds = time.perf_counter() - started

    # 以本段第一帧有效帧为参考，补上相对参考帧的位移与 Kabsch 转角。
    reference = _reference_points(result, tracker)
    if reference is not None:
        reference_centroid = reference.mean(axis=0)
        for frame in result.frames:
            if not frame.valid or frame.points_px is None:
                continue
            centroid = frame.points_px.mean(axis=0)
            frame.shift_x_px = float(centroid[0] - reference_centroid[0])
            frame.shift_y_px = float(centroid[1] - reference_centroid[1])
            try:
                angle, residual, _scale = kabsch_rotation_about_centroid(
                    reference, frame.points_px
                )
                frame.rotation_deg = angle
                frame.residual_px = residual
            except VisionError:
                frame.rotation_deg = None
                frame.residual_px = None

    if save_corners:
        corner_rows, metric_rows = _collect_rows(result)
        if corners_dir is not None:
            from .recorder import safe_name

            write_corners(corners_dir / f"{safe_name(segment_id)}.csv", corner_rows)
        if metrics_path is not None:
            append_metrics(metrics_path, metric_rows)

    return result


def _reference_points(result: SegmentVision, tracker: Any) -> np.ndarray | None:
    """本段参考帧的角点。

    直接用 ``CheckerboardTracker`` 自己记下的参考角点——它记的就是"本段第一帧成功识别"
    的那一组，和被复用代码内部算 dx/dy 用的是同一个基准，所以两边口径一致。
    """
    reference = getattr(tracker, "reference_corners", None)
    if reference is None:
        return None
    points = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    return points if len(points) >= 3 else None


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
                        "x_px": _fmt(float(x_coord), 4),
                        "y_px": _fmt(float(y_coord), 4),
                    }
                )
    return corner_rows, metric_rows


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
