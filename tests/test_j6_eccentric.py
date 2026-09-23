"""J6 偏心棋盘格：面内运动量必须**平移和转动一起算**，不能假设板装在轴心。

为什么这条要单独测
------------------
J6 是绕自身轴旋转的关节。棋盘格是**人手贴上去的**，不保证落在旋转轴中心，
现场偏心 5～15 mm 是常态。

v1.0.1 之前，J6 的"面内运动量"只取质心平移的模。这条口径有两个致命后果：

1. **板正好居中时读到零**：质心不动，而画面明明在小幅旋转，
   判据会说"这个关节 0.2° 在画面上几乎看不出来"，把一次真实有效的运动判成无效；
2. **同一个转角读出不同的数**：读数正比于偏心量——那是安装公差，不是运动量。
   偏心 15 mm 和偏心 5 mm 的两次实验得出完全不同的"面内位移"，
   而两者的物理运动一模一样。

现在的口径（需求五）：

    d_rot_px   = 2 · r_rms_px · sin(|θ| / 2)        ← 板自己转过的等效面内位移
    d_plane_px = √(质心位移² + d_rot_px²)            ← 两项合成
    d_plane_mm = d_plane_px × mm_per_pixel

J6 的**角度**仍然只用 88 角点去质心后的二维 Kabsch 转角，不用"质心圆弧位移 ÷
假定偏心"反推——那样反推出来的角度会随偏心假设变化，而偏心恰恰是不知道的。

这里测三件事：

1. 居中（偏心 0）与偏心 5/10/15 mm 的**纯旋转**都必须通过轴向/面内判据；
2. 加上明显的尺度变化（= 轴向位移）后必须被拦下来；
3. RAW 删掉之后，J6 的角度和动态过程仍能从**保存的角点 CSV** 复算出来。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from sj_pretest.config import AppConfig
from sj_pretest.recorder import safe_name
from sj_pretest.vision import (
    CONFIDENCE_RESOLVED,
    FrameVision,
    estimate_depth_in_plane,
    in_plane_px_of_frame,
    judge_in_plane_dominant,
    kabsch_rotation_about_centroid,
    read_corners,
    rms_radius_px,
)

from conftest import build_config, open_session

#: 界面上量分辨率用的棋盘格内角点数（11×8）。
BOARD_COLS, BOARD_ROWS = 11, 8
#: 格距（毫米）与现场标称的毫米/像素。
SQUARE_MM = 30.0
MM_PER_PIXEL = 0.1875
#: 现场常见的偏心量（毫米），需求五里写的就是 5～15 mm。
ECCENTRICITIES_MM = (0.0, 5.0, 10.0, 15.0)


def _board_points() -> Any:
    """一张 11×8 内角点的理想网格（像素），质心在原点。"""
    import numpy as np

    spacing_px = SQUARE_MM / MM_PER_PIXEL
    xs = (np.arange(BOARD_COLS) - (BOARD_COLS - 1) / 2.0) * spacing_px
    ys = (np.arange(BOARD_ROWS) - (BOARD_ROWS - 1) / 2.0) * spacing_px
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def _rotate(points: Any, angle_deg: float, center: Any) -> Any:
    import numpy as np

    theta = math.radians(float(angle_deg))
    matrix = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    return (np.asarray(points) - np.asarray(center)) @ matrix.T + np.asarray(center)


def _frame(
    *,
    rotation_deg: float,
    shift_x_px: float,
    shift_y_px: float,
    r_rms_px: float,
    scale: float = 1.0,
    analysis_time_s: float = 0.0,
) -> FrameVision:
    """手搭一帧 J6 的逐帧结果（不跑采集，纯算判据）。"""
    return FrameVision(
        frame_id=0,
        host_ns=0,
        analysis_time_s=float(analysis_time_s),
        valid=True,
        corner_count=BOARD_COLS * BOARD_ROWS,
        centroid_x_px=100.0,
        centroid_y_px=80.0,
        rotation_deg=float(rotation_deg),
        residual_px=0.05,
        quality=1.0,
        mm_per_pixel=MM_PER_PIXEL,
        shift_x_px=float(shift_x_px),
        shift_y_px=float(shift_y_px),
        scale=float(scale),
        mean_brightness=120.0,
        blur_variance=200.0,
        r_rms_px=float(r_rms_px),
    )


def _noise(count: int = 9) -> list[FrameVision]:
    """静止段：只留尺度噪声，量级和合成世界实测同阶（±3e-5）。"""
    deltas = (-3e-5, -1e-5, 0.0, 1e-5, 3e-5, -2e-5, 2e-5, 0.0, 1e-5)[:count]
    return [
        _frame(
            rotation_deg=0.0,
            shift_x_px=0.0,
            shift_y_px=0.0,
            r_rms_px=float(r_rms_px_cached()),
            scale=1.0 + delta,
            analysis_time_s=index * 0.0075,
        )
        for index, delta in enumerate(deltas)
    ]


_R_RMS_CACHE: list[float] = []


def r_rms_px_cached() -> float:
    if not _R_RMS_CACHE:
        _R_RMS_CACHE.append(rms_radius_px(_board_points()))
    return _R_RMS_CACHE[0]


def _eccentric_signal(
    *, eccentricity_mm: float, rotation_deg: float, scale: float = 1.0
) -> list[FrameVision]:
    """纯旋转（可选加一点尺度变化）时的"运动段"逐帧结果。

    偏心 ``e`` 的板绕轴转 θ，质心沿弦走 ``2·e·sin(θ/2)``；
    板自身相对质心的转动就是 θ。两项都按真实几何算出来，不许拍脑袋填。
    """
    r_rms = r_rms_px_cached()
    eccentric_px = float(eccentricity_mm) / MM_PER_PIXEL
    chord_px = 2.0 * eccentric_px * math.sin(math.radians(abs(rotation_deg)) / 2.0)
    # ★ 窗口里的每一帧都**已经到位**（保持段），所以每一帧的几何量一样。
    # 以前这里放了一帧名义位姿（转角 0、位移 0）当"参考帧"：那是把参考帧混进了
    # 运动窗口。深度取窗口均值，混一帧名义位姿会把均值（也就是深度）拉走一半，
    # 而那一半跟"转了多少"毫无关系。
    return [
        _frame(
            rotation_deg=float(rotation_deg),
            shift_x_px=chord_px,
            shift_y_px=0.0,
            r_rms_px=r_rms,
            scale=float(scale),
        )
        for _ in range(3)
    ]


def _estimate(tmp_path: Path, **kwargs):
    config = build_config(
        tmp_path, joints=("J6",), amplitudes=(0.2,), repeats=1
    )
    return estimate_depth_in_plane(
        _eccentric_signal(**kwargs),
        config=config,
        noise_frames=_noise(),
        rotation_joint=True,
    )


# --------------------------------------------------------------------------
# 1) 居中 + 偏心 5/10/15 mm 的纯旋转都要通过
# --------------------------------------------------------------------------


@pytest.mark.parametrize("eccentricity_mm", ECCENTRICITIES_MM)
def test_a_pure_j6_rotation_passes_whatever_the_eccentricity(
    tmp_path: Path, eccentricity_mm: float
) -> None:
    """偏心 0（居中）、5、10、15 mm 的纯旋转都必须通过轴向/面内判据。

    0.2° 是需求四给这个几何确认的推荐默认值；J6 在 0.2° 时板自身转过的
    等效面内位移约 ``2·r_rms·sin(0.1°)``，与 J1 的面内位移同量级——
    这正是"合成之后判据不再跟着偏心跑"要证明的事。
    """
    estimate = _estimate(
        tmp_path, eccentricity_mm=eccentricity_mm, rotation_deg=0.2
    )
    assert estimate.in_plane_mm is not None and estimate.in_plane_mm > 0.0, estimate
    allowed, why = judge_in_plane_dominant(estimate, max_depth_ratio=0.5)
    assert allowed is True, (
        f"偏心 {eccentricity_mm} mm 的 J6 纯旋转被判成「轴向分量偏大」：{why}"
    )
    assert "以面内运动为主" in why, why
    # 纯旋转在画面上不该有显著的尺度变化：深度要么分辨不出来（按上限判），
    # 要么虽然分辨出来了也远小于面内位移。两种情况都必须是"放行"。
    assert estimate.confidence != CONFIDENCE_RESOLVED or (
        estimate.depth_mm is not None and estimate.depth_mm < estimate.in_plane_mm
    ), estimate


def test_a_centred_board_is_not_read_as_no_motion(tmp_path: Path) -> None:
    """棋盘格完全居中时，质心一点不动——但**不能**因此判成"没有面内运动"。

    这是旧口径最直接的错误：质心位移 0 → 面内位移 0 → 判据无话可说。
    """
    estimate = _estimate(tmp_path, eccentricity_mm=0.0, rotation_deg=0.2)
    assert estimate.in_plane_mm is not None
    # 居中时质心位移确实是 0，面内位移全部来自转动分量。
    signal = _eccentric_signal(eccentricity_mm=0.0, rotation_deg=0.2)
    assert signal[-1].shift_x_px == 0.0 and signal[-1].shift_y_px == 0.0
    expected_px = 2.0 * r_rms_px_cached() * math.sin(math.radians(0.2) / 2.0)
    assert estimate.in_plane_mm == pytest.approx(expected_px * MM_PER_PIXEL, rel=1e-6)


@pytest.mark.parametrize("eccentricity_mm", ECCENTRICITIES_MM)
def test_the_composition_is_at_least_the_rotation_term(
    tmp_path: Path, eccentricity_mm: float
) -> None:
    """合成值必须 ≥ 转动分量：偏心越大，质心平移补进来，总数只增不减。

    这条盯的是"两个分量是**合成**的，不是二选一"：如果实现里写成
    "有偏心就只取平移、没偏心就只取转动"，居中那组会漏掉转动分量，
    偏心那组会漏掉转动分量——两种错法在这里都会露出来。
    """
    signal = _eccentric_signal(eccentricity_mm=eccentricity_mm, rotation_deg=0.2)
    composed = max(in_plane_px_of_frame(f, rotation_joint=True) for f in signal)
    rotation_only = 2.0 * r_rms_px_cached() * math.sin(math.radians(0.2) / 2.0)
    shift_only = max(
        math.hypot(float(f.shift_x_px or 0.0), float(f.shift_y_px or 0.0))
        for f in signal
    )
    assert composed >= rotation_only - 1e-9
    assert composed >= shift_only - 1e-9
    assert composed == pytest.approx(
        math.hypot(rotation_only, shift_only), rel=1e-6
    )


def test_j5_style_joints_still_use_the_centroid_shift(tmp_path: Path) -> None:
    """非旋转关节（J1～J5）的口径**不许**被改动：还是质心位移。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    frame = _frame(rotation_deg=3.0, shift_x_px=8.0, shift_y_px=6.0, r_rms_px=200.0)
    assert in_plane_px_of_frame(frame, rotation_joint=False) == pytest.approx(10.0)
    assert in_plane_px_of_frame(frame, rotation_joint=True) > 10.0, (
        "J6 的口径必须把转动分项算进去"
    )
    assert "J6" in config.vision.rotation_joints
    assert "J1" not in config.vision.rotation_joints


# --------------------------------------------------------------------------
# 2) 加了明显的尺度变化 → 轴向判据必须拦下来
# --------------------------------------------------------------------------


def test_a_scale_change_is_blocked_by_the_axial_criterion(tmp_path: Path) -> None:
    """同一批几何，只加一个明显的尺度变化（轴向位移），必须被拦。

    0.001 的尺度变化在工作距离 675 mm 上就是约 0.68 mm 的轴向位移；
    而 0.2° 的 J6 转动在 950×800 这一档上的面内位移只有约 0.1 mm 量级，
    轴向/面内比值远大于 0.5——判据必须说"轴向分量偏大"。
    """
    clean = _estimate(tmp_path / "clean", eccentricity_mm=10.0, rotation_deg=0.2)
    dirty = _estimate(
        tmp_path / "dirty", eccentricity_mm=10.0, rotation_deg=0.2, scale=1.001
    )
    assert clean.confidence in (CONFIDENCE_RESOLVED, "below_resolution")
    assert dirty.confidence == CONFIDENCE_RESOLVED, dirty
    assert dirty.depth_mm is not None and dirty.depth_mm > 0.5, dirty
    assert clean.in_plane_mm == pytest.approx(dirty.in_plane_mm, rel=1e-9), (
        "面内分量不该因为尺度变化而改变——它只由平移和转角决定"
    )
    assert dirty.ratio is not None and dirty.ratio > 0.5, dirty

    allowed_clean, why_clean = judge_in_plane_dominant(clean, max_depth_ratio=0.5)
    allowed_dirty, why_dirty = judge_in_plane_dominant(dirty, max_depth_ratio=0.5)
    assert allowed_clean is True, why_clean
    assert allowed_dirty is False, f"加了轴向位移却没被拦下来：{why_dirty}"
    assert "轴向" in why_dirty, why_dirty


# --------------------------------------------------------------------------
# 3) RAW 删掉之后，J6 的角度与动态过程仍能复算
# --------------------------------------------------------------------------


def _j6_group(tmp_path: Path):
    config = build_config(
        tmp_path,
        joints=("J6",),
        amplitudes=(0.2,),
        repeats=1,
        paths__delete_raw_after_process=True,
    )
    session, _recorder = open_session(config, run_kind="j6_eccentric")
    return session, config


def test_the_j6_angle_survives_the_raw_deletion(tmp_path: Path) -> None:
    """删了 RAW 之后：J6 的角度、r_rms 和逐帧过程都还能从保存的数据复算。

    复算路径要求得很具体：**只用角点 CSV**（去掉质心后的二维 Kabsch）
    重新算一遍转角，结果必须和 ``segment_vision.json`` 里存的一致。
    只断言"文件还在"是不够的——文件在但算不出角度，等于数据没留住。
    """
    session, config = _j6_group(tmp_path)
    try:
        session.run_static()
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    deleted = [
        item
        for item in session.run.events.read_all()
        if item.get("event") == "raw_deleted"
    ]
    j6_deleted = [
        item
        for item in deleted
        if str(item["segment_id"]).startswith("pretest-J6")
    ]
    assert j6_deleted, "J6 的预实验段没有被处理掉"

    from sj_pretest.vision import load_segment_vision

    for event in j6_deleted:
        segment_id = str(event["segment_id"])
        segment_dir = session.run.segment_dir(segment_id)
        assert not (segment_dir / "frames.raw").is_file()

        vision = load_segment_vision(
            segment_dir,
            segment_id=segment_id,
            corners_path=session.run.corners_dir / f"{safe_name(segment_id)}.csv",
        )
        valid = [f for f in vision.frames if f.valid and f.rotation_deg is not None]
        assert len(valid) >= 50, f"{segment_id}：留存的逐帧结果太少（{len(valid)} 帧）"
        assert all(f.r_rms_px is not None and f.r_rms_px > 0 for f in valid), (
            f"{segment_id}：逐帧结果里没有 r_rms_px，面内位移复算不出来"
        )

        # ★ 只用角点 CSV 复算：取存下来的首末两帧角点，去质心后做二维 Kabsch。
        corners = read_corners(
            session.run.corners_dir / f"{safe_name(segment_id)}.csv"
        )
        assert corners, f"{segment_id}：角点 CSV 是空的"
        first, last = valid[0], valid[-1]
        assert first.frame_id in corners and last.frame_id in corners, (
            f"{segment_id}：逐帧结果里的帧在角点 CSV 里找不到"
        )
        # ★ 复算的是**转角的变化量**，不是绝对转角。
        #
        # 绝对转角永远是"相对某一帧参考"的：``process_segment`` 用的是
        # CheckerboardTracker 自己记下的参考角点，那一帧未必逐帧落在 CSV 里
        # （参考帧可能是一帧角点数不够 88、因而没被写进 CSV 的帧）。
        # 而"从第 i 帧到第 j 帧转了多少"与参考帧**无关**：
        #     Kabsch(ref→j) − Kabsch(ref→i) ≡ Kabsch(i→j)
        # 所以这个量可以从角点 CSV 精确复算，也必须精确对上——
        # 它才是"删了 RAW 之后 J6 的动态过程还算得出来"要证明的东西。
        delta_csv, residual, scale = kabsch_rotation_about_centroid(
            corners[first.frame_id], corners[last.frame_id]
        )
        delta_json = float(last.rotation_deg) - float(first.rotation_deg)
        assert scale == pytest.approx(1.0, abs=0.05), (
            f"{segment_id}：纯旋转段的 Kabsch 尺度不该偏离 1：{scale}"
        )
        assert abs(delta_csv) > 1e-4, (
            f"{segment_id}：J6 这一段几乎没转（{delta_csv}°），测不出复算是否成立"
        )
        assert delta_csv == pytest.approx(delta_json, abs=1e-6), (
            f"{segment_id}：从角点 CSV 复算的转角变化 {delta_csv}° 和存下来的 "
            f"{delta_json}° 对不上——RAW 删掉之后角度复算不出来了"
        )
        # 绝对转角也应当对得上，只是基准可能差一点点（见上），所以给宽一点的容差。
        assert float(last.rotation_deg) == pytest.approx(
            delta_csv + float(first.rotation_deg), abs=1e-3
        )
        # r_rms 也要能从角点复算出来（它只是角点到质心的均方根半径）。
        assert float(last.r_rms_px) == pytest.approx(
            rms_radius_px(corners[last.frame_id]), rel=1e-6
        )


def test_the_j6_dynamics_survive_the_raw_deletion(tmp_path: Path) -> None:
    """快速几何检查（= J6 的动态过程判据）在删了 RAW 之后必须照样跑完。

    需求二点名了这条：不能在 RAW 被删掉之后再回头去 ``process_segment``
    读 frames.raw。这里跑完整的快速几何检查（开着边采边清），
    检查它用的是留存结果、并且 J6 的正负方向转角反号。
    """
    session, config = _j6_group(tmp_path)
    try:
        result, failed = session.run_quick_probes()
        text = "\n".join(result.lines)
    finally:
        session.close()

    assert "图像二维转角" in text, text
    assert "J6" in text, text
    assert failed == [], (
        f"删了 RAW 之后 J6 的快速几何检查没跑通：{failed}\n{text}"
    )
    # 而且真的是"删了之后才算的"：J6 那一段的 RAW 确实不在了。
    assert session.run is not None
    j6_segments = [
        path
        for path in session.run.segments_dir.glob("quick_probe-J6*")
        if path.is_dir()
    ]
    assert j6_segments, "没找到 J6 的快速探针目录"
    for path in j6_segments:
        assert not (path / "frames.raw").is_file(), (
            f"{path.name}：RAW 还在，这条测试没测到「删了之后复算」这件事"
        )
