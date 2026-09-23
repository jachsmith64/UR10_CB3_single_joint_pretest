"""``thresholds.max_depth_ratio`` 必须**真的**能通过、也**真的**能拦截。

为什么这条要单独测
------------------
1.0.0 里"以面内运动为主"只写在文档和需求里，代码里没有任何一处读它：
``max_depth_ratio`` 是个**死参数**，把它改成 0.0001 或者 10 都毫无影响，
``in_plane_ratio_from_scale_change`` 也是一段没人调的备用函数。
这种"参数在、判据不在"的状态最危险：报告里会显示"已按阈值判定"，
而实际上什么都没判。

这里测两个层次：

1. **端到端**：真的跑一遍快速几何检查，同一批数据在**默认判据**下通过；
   把 ``max_depth_ratio`` 收紧到远小于实测比值，同一批数据必须被拦下来，
   并且报告里要出现实测的 depth / in_plane / 比值 / 有效帧数 / 置信状态。
2. **单元**：判据本身在"低于可分辨下限"时用**上限**去判，
   而且用到的正是 ``camera.working_distance_mm``——把它改大 10 倍，
   同一个尺度变化的深度估计必须跟着大 10 倍（证明这个参数没被架空）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import AppConfig
from sj_pretest.vision import (
    CONFIDENCE_BELOW_RESOLUTION,
    CONFIDENCE_RESOLVED,
    DepthInPlane,
    estimate_depth_in_plane,
    in_plane_ratio_from_scale_change,
    judge_in_plane_dominant,
)

from conftest import build_config, open_session


def _quick_probe(tmp_path: Path, *, max_depth_ratio: float | None):
    """跑一遍快速几何检查，返回 (报告文本, 未通过的关节, 配置)。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    # 0.2°：需求四给这个几何确认的推荐默认值（0.05° 在 J1 上面内位移
    # 只有约 0.375 mm，和轴向可分辨极限同量级，方向判不出来）。
    config.pretest.quick_probe_deg = 0.2
    if max_depth_ratio is not None:
        config.thresholds.max_depth_ratio = float(max_depth_ratio)
    config.validate()
    session, _recorder = open_session(config, run_kind="depth_gate")
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        result, failed = session.run_quick_probes()
        return "\n".join(result.lines), list(failed), config
    finally:
        session.close()


def test_the_gate_passes_with_the_delivered_threshold(tmp_path: Path) -> None:
    """交付默认判据下，J1 的 0.2° 微动必须判为"以面内运动为主"。"""
    text, failed, config = _quick_probe(tmp_path, max_depth_ratio=None)
    assert config.thresholds.max_depth_ratio == 0.5, (
        "交付默认值被改过了——这条测试要盯的就是它"
    )
    assert "轴向/面内" in text, "报告里没有轴向/面内分解这一行"
    assert "深度/面内 = " in text
    assert "有效帧数" in text
    assert "置信状态" in text
    assert failed == [], f"默认判据下 J1 不该被判不合格：{failed}\n{text}"


def test_the_gate_blocks_when_the_threshold_is_tightened(tmp_path: Path) -> None:
    """把判据收紧到实测比值以下，**同一批数据**必须被拦下来。

    这是"参数真的在用"的正面证据：数据一个字节没动，只有这一个数变了。
    """
    text, failed, _config = _quick_probe(tmp_path, max_depth_ratio=0.001)
    assert failed == ["J1"], (
        f"把 max_depth_ratio 收到 0.001 之后 J1 居然还通过了：{failed}\n{text}"
    )
    assert "轴向分量偏大" in text or "轴向/面内" in text, text
    # 报告里要能看见"是哪个数超了哪个数"。
    assert "0.001" in text
    assert "未通过" in text


def test_the_gate_is_monotone_in_the_threshold(tmp_path: Path) -> None:
    """判据要跟着阈值单调走：松的过、紧的不过，中间不许出现"紧了反而过"。"""
    loose, failed_loose, _ = _quick_probe(tmp_path / "loose", max_depth_ratio=2.0)
    tight, failed_tight, _ = _quick_probe(tmp_path / "tight", max_depth_ratio=0.05)
    assert failed_tight and not failed_loose, (
        f"阈值 2.0 时 {failed_loose}，阈值 0.05 时 {failed_tight}：判据方向反了\n"
        f"{loose}\n{tight}"
    )


def _frame(scale: float, shift_x_px: float, shift_y_px: float):
    """手搭一帧离线识别结果（不跑采集，纯算判据）。"""
    from sj_pretest.vision import FrameVision

    return FrameVision(
        frame_id=0,
        host_ns=0,
        analysis_time_s=0.0,
        valid=True,
        corner_count=88,
        centroid_x_px=100.0,
        centroid_y_px=80.0,
        rotation_deg=0.0,
        residual_px=0.1,
        quality=1.0,
        mm_per_pixel=0.1875,
        shift_x_px=shift_x_px,
        shift_y_px=shift_y_px,
        scale=scale,
        mean_brightness=120.0,
        blur_variance=200.0,
    )


def _estimate(
    tmp_path: Path,
    *,
    working_distance_mm: float,
    scale: float,
    shift_x_px: float,
    shift_y_px: float = 0.0,
) -> DepthInPlane:
    """静止段噪声固定（±3e-5 量级，和合成世界实测同量级），运动段给一组 (scale, shift)。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.working_distance_mm = float(working_distance_mm)
    config.validate()
    noise = [_frame(1.0 + delta, 0.0, 0.0) for delta in (-3e-5, 0.0, 3e-5, -1e-5, 2e-5)]
    signal = [_frame(1.0, 0.0, 0.0), _frame(scale, shift_x_px, shift_y_px)]
    return estimate_depth_in_plane(signal, config=config, noise_frames=noise)


def test_depth_estimate_scales_with_the_working_distance(tmp_path: Path) -> None:
    """深度估计必须真的用 ``camera.working_distance_mm``，而不是写死的 675。"""
    near = _estimate(tmp_path / "a", working_distance_mm=675.0, scale=1.001, shift_x_px=8.0, shift_y_px=0.0)
    far = _estimate(tmp_path / "b", working_distance_mm=1350.0, scale=1.001, shift_x_px=8.0, shift_y_px=0.0)
    assert near.confidence == CONFIDENCE_RESOLVED, near
    assert near.depth_mm is not None and far.depth_mm is not None
    assert far.depth_mm == pytest.approx(near.depth_mm * 2.0, rel=1e-6), (
        f"工作距离翻倍，深度估计没跟着变：{near.depth_mm} → {far.depth_mm}"
    )
    assert far.resolution_mm == pytest.approx(near.resolution_mm * 2.0, rel=1e-6)
    # 面内位移跟工作距离无关（它是角度 × mm/px 的量），所以比值必须翻倍。
    assert far.in_plane_mm == pytest.approx(near.in_plane_mm, rel=1e-9)
    assert far.ratio == pytest.approx(near.ratio * 2.0, rel=1e-6)


def test_below_resolution_says_so_and_gates_on_the_upper_bound(tmp_path: Path) -> None:
    """分辨不出来时：说"无法判断"，并且判据用的是**上限**而不是 0。"""
    tiny = _estimate(
        tmp_path / "tiny", working_distance_mm=675.0, scale=1.0 + 1e-9, shift_x_px=8.0, shift_y_px=0.0
    )
    assert tiny.confidence == CONFIDENCE_BELOW_RESOLUTION
    assert tiny.depth_mm is None, "低于可分辨下限时不该报一个实测深度"
    assert "无法判断" in tiny.judgement_text
    assert tiny.depth_limit_mm is not None and tiny.depth_limit_mm > 0
    assert tiny.ratio_upper is not None and tiny.ratio_upper > 0, (
        "上限比值不该是 0——那样等于把'看不出来'当成'没有轴向运动'"
    )
    assert "无法判断" in "\n".join(tiny.to_lines())

    # 同上，但把工作距离拉大到让上限压不住 0.5：判据必须拦下来。
    allowed, why = judge_in_plane_dominant(tiny, max_depth_ratio=0.5)
    assert allowed is True, why  # 675 mm、8 px、噪声 3e-5 → 上限比值很小
    blocked, why_blocked = judge_in_plane_dominant(tiny, max_depth_ratio=1e-6)
    assert blocked is False and "轴向分量偏大" in why_blocked, why_blocked


def test_unavailable_is_never_treated_as_passing(tmp_path: Path) -> None:
    """算不出来（没有静止参考帧）时必须报"无法判断"，并且**不许放行**。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    result = estimate_depth_in_plane(
        [_frame(1.002, 8.0, 0.0)], config=config, noise_frames=[]
    )
    assert result.confidence == "unavailable"
    allowed, why = judge_in_plane_dominant(result, max_depth_ratio=0.5)
    assert allowed is False
    assert "无法判断" in why


def test_the_legacy_ratio_helper_agrees_with_the_new_one(tmp_path: Path) -> None:
    """旧的 ``in_plane_ratio_from_scale_change`` 与新的深度估计口径一致。

    它以前是死代码；现在两条路径必须给出同一个量级，否则报告和判据会各说各话。
    """
    config = AppConfig()
    estimate = _estimate(
        tmp_path, working_distance_mm=config.camera.working_distance_mm,
        scale=1.001, shift_x_px=8.0, shift_y_px=0.0,
    )
    legacy = in_plane_ratio_from_scale_change(
        0.001, float(config.camera.working_distance_mm), 8.0, 0.1875
    )
    assert estimate.confidence == CONFIDENCE_RESOLVED
    assert estimate.ratio == pytest.approx(legacy, rel=1e-6), (
        f"新口径 {estimate.ratio} 和旧函数 {legacy} 对不上"
    )
