"""★ 需求一·5、一·6 的隐藏前提：**同一个物理状态必须总是同一个结论**。

为什么要单独测"结论稳不稳"
--------------------------
快速几何检查的 J6 那一档本来就贴着自己的分辨下限：0.2° 的探针在 675 mm 工作
距离上只有 0.24 mm 量级的面内位移，而相机看到的尺度变化小到 1e-4 量级。
这种量级上，"结论"对**估计量怎么算**极其敏感：

* 用"窗口里偏离 1 最远的那一帧"当深度 → 看的帧越多，挑到的极值越大，
  同一个动作会因为"这次采了多少帧、抽没抽样"得出不同的深度；
* 用"第一帧"当几何零点 → 那一帧的识别噪声变成整段的公共偏移，
  0.02° 的基准噪声落在 0.2° 的探针上就是 10%，判据时过时不过；
* 用单帧的标准差当"可分辨下限" → 估计量明明是 52 帧的均值，
  误差棒却按 1 帧算，把下限放大 3 倍，于是"其实分辨得出来"被报成
  "分辨不出来"，判据只能拿一个偏大的上限去比，结论就又跟着噪声跑了。

前两条在 v1.0.3 里改掉了（见 ``vision._reference_points`` 与
``estimate_depth_in_plane`` 的 docstring）。这个文件把**第三条的性质**钉死：
拿同一批实测噪声参数（合成世界实测 σ≈6.4e-5、52 帧窗口、12 帧基准、
面内 0.24 mm 量级）做多次噪声实现，判据的结论必须**一次都不翻**；
而按旧口径（单帧基准 + 单帧标准差）做的同一批实现必须**翻**——
后者证明这个"稳"不是碰巧，是真的把结论从噪声手里拿了回来。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sj_pretest.vision import (
    CONFIDENCE_RESOLVED,
    FrameVision,
    estimate_depth_in_plane,
    judge_in_plane_dominant,
)

from conftest import build_config

#: 现场量分辨率用的棋盘格内角点数（11×8），与 test_j6_eccentric 一致。
BOARD_CORNERS = 88
#: 现场标称的毫米/像素与 J6 的均方根半径（合成世界实测 62.5 px）。
MM_PER_PIXEL = 0.1875
R_RMS_PX = 62.5
#: 合成世界实测的单帧识别噪声（尺度 / 质心各分量 / 转角）。
SIGMA_SCALE = 6.4e-5
SIGMA_SHIFT_PX = 0.416
SIGMA_ROT_DEG = 0.027
#: 0.2° 的 J6 探针在名义位姿上的真实几何量（合成世界实测）。
TRUE_SHIFT_PX = 0.6
TRUE_ROT_DEG = 0.1756
TRUE_SCALE_DEV = 4.0e-5
#: 窗口与基准各用多少帧（生产路径：保持段 52 帧、基准 12 帧）。
WINDOW_FRAMES = 52
REFERENCE_FRAMES = 12
#: 噪声池（运动前 + 运动后）的帧数。
POOL_FRAMES = 52
#: 跑多少种噪声实现。
DRAWS = 40


def _frame(
    *,
    scale: float,
    shift_x_px: float,
    shift_y_px: float,
    rotation_deg: float,
    analysis_time_s: float = 0.0,
) -> FrameVision:
    return FrameVision(
        frame_id=0,
        host_ns=0,
        analysis_time_s=float(analysis_time_s),
        valid=True,
        corner_count=BOARD_CORNERS,
        centroid_x_px=170.0,
        centroid_y_px=130.0,
        rotation_deg=float(rotation_deg),
        residual_px=0.05,
        quality=1.0,
        mm_per_pixel=MM_PER_PIXEL,
        shift_x_px=float(shift_x_px),
        shift_y_px=float(shift_y_px),
        scale=float(scale),
        mean_brightness=120.0,
        blur_variance=200.0,
        r_rms_px=R_RMS_PX,
    )


def _arc_px(rotation_deg: float) -> float:
    """转动扫过的等效面内位移（像素）。判据内部用的是同一个式子。

    这里留一份是为了让"面内位移由哪两项合成"在测试文件里也是一眼可见的。
    """
    return 2.0 * R_RMS_PX * math.sin(math.radians(abs(float(rotation_deg))) / 2.0)


def _simulate(seed: int, *, single_frame_baseline: bool, scale_dev: float):
    """造一次"J6 0.2° 探针"的逐帧结果。

    ``single_frame_baseline`` 打开时，几何零点是**一帧**（旧口径）：
    那一帧的噪声会变成整段位移/转角/尺度的公共偏移，而且它自己对每帧都一样，
    不会被平均掉。关掉时是 12 帧的平均（v1.0.3 口径）。
    """
    rng = np.random.default_rng(seed)
    baseline_frames = 1 if single_frame_baseline else REFERENCE_FRAMES
    # 基准自己的误差：一帧就是全额噪声，多帧降到 1/√n。
    b_scale = 1.0 + rng.normal(0.0, SIGMA_SCALE) / math.sqrt(baseline_frames)
    b_shift_x = rng.normal(0.0, SIGMA_SHIFT_PX) / math.sqrt(baseline_frames)
    b_shift_y = rng.normal(0.0, SIGMA_SHIFT_PX) / math.sqrt(baseline_frames)
    b_rot = rng.normal(0.0, SIGMA_ROT_DEG) / math.sqrt(baseline_frames)

    window = []
    for index in range(WINDOW_FRAMES):
        # 尺度是"窗口测量 ÷ 基准测量"——两边的噪声都进得来。
        scale = (1.0 + scale_dev + rng.normal(0.0, SIGMA_SCALE)) / b_scale
        shift_x = TRUE_SHIFT_PX + rng.normal(0.0, SIGMA_SHIFT_PX) - b_shift_x
        shift_y = rng.normal(0.0, SIGMA_SHIFT_PX) - b_shift_y
        rotation = TRUE_ROT_DEG + rng.normal(0.0, SIGMA_ROT_DEG) - b_rot
        window.append(
            _frame(
                scale=scale,
                shift_x_px=shift_x,
                shift_y_px=shift_y,
                rotation_deg=rotation,
                analysis_time_s=index * 0.00756,
            )
        )
    pool = [
        _frame(
            scale=1.0 + rng.normal(0.0, SIGMA_SCALE),
            shift_x_px=rng.normal(0.0, SIGMA_SHIFT_PX),
            shift_y_px=rng.normal(0.0, SIGMA_SHIFT_PX),
            rotation_deg=rng.normal(0.0, SIGMA_ROT_DEG),
        )
        for _ in range(POOL_FRAMES)
    ]
    return window, pool


def _verdict(tmp_path: Path, seed: int, *, single_frame_baseline: bool, scale_dev: float):
    config = build_config(tmp_path, joints=("J6",), amplitudes=(0.2,), repeats=1)
    window, pool = _simulate(
        seed, single_frame_baseline=single_frame_baseline, scale_dev=scale_dev
    )
    estimate = estimate_depth_in_plane(
        window,
        config=config,
        noise_frames=pool,
        rotation_joint=True,
        # 旧口径 = 基准只有一帧。v1.0.3 的基准帧数是从逐帧结果里带过来的。
        reference_frames=1 if single_frame_baseline else REFERENCE_FRAMES,
    )
    allowed, why = judge_in_plane_dominant(estimate, max_depth_ratio=0.5)
    return estimate, allowed, why


def test_the_pure_rotation_verdict_never_flips_across_noise_draws(tmp_path: Path) -> None:
    """★ 纯转动（真实深度 0.027 mm）在 40 种噪声实现里必须**全部**放行。

    这条就是"结论不跟着噪声跑"的正面证明：判据要么真的量出了深度
    （``resolved``，0.03 mm 量级，远小于面内），要么如实说"低于下限"
    并拿上限去比——两种情况下都必须是放行，一次都不许翻。
    """
    verdicts = [
        _verdict(tmp_path / f"seed{seed}", seed, single_frame_baseline=False, scale_dev=TRUE_SCALE_DEV)
        for seed in range(DRAWS)
    ]
    blocked = [
        (index, why)
        for index, (_estimate, allowed, why) in enumerate(verdicts)
        if not allowed
    ]
    assert not blocked, (
        f"{len(blocked)}/{DRAWS} 次噪声实现把一次纯转动判成了「轴向分量偏大」："
        f"{blocked[:3]}"
    )
    # 而且判据用的比值要有余量，不能是"刚好压线过"。
    worst = max(float(estimate.ratio_upper) for estimate, _a, _w in verdicts)
    assert worst <= 0.5 * 0.6, (
        f"最差一次的上限比值 {worst:.3f} 离门限 0.5 太近——这种余量撑不住换一台设备"
    )
    # 再看上限本身的涨落：它只应该随"σ 估得准不准"变（52 帧估一个标准差，抽样
    # 涨落本来就有 ±20% 量级），不该有别的结构性依赖。真实的 0.027 mm 只有
    # 下限的 0.4～0.7 倍，所以大多数实现会如实报"低于下限、按上限判"——
    # 这是保守路径，不是兜底失败。
    limits = [float(estimate.depth_limit_mm) for estimate, _a, _w in verdicts]
    assert all(value > 0.0 for value in limits)
    assert max(limits) <= 2.0 * min(limits), (
        f"上限在实现之间从 {min(limits):.4f} 涨到 {max(limits):.4f} mm——"
        "涨落超过「σ 估计误差」能解释的范围，说明还有别的因素在决定结论"
    )


def test_the_single_frame_baseline_is_what_made_the_verdict_flip(tmp_path: Path) -> None:
    """旧口径（单帧基准 + 单帧标准差）在同一批实现里**会翻**。

    这不是"为了证明改动有理"编出来的对比：两项都是合成世界实测参数的直接后果——
    单帧基准的噪声（0.42 px 位移、0.027° 转角）与 0.2° 探针的真实量同量级，
    所以"面内位移"的读数会被基准噪声抬高或压低一大截，而单帧标准差又把可分辨
    下限放大 3 倍。读数的分子分母同时在抖，结论自然时过时不过。
    """
    outcomes = set()
    for seed in range(DRAWS):
        _estimate, allowed, _why = _verdict(
            tmp_path / f"old{seed}", seed, single_frame_baseline=True, scale_dev=TRUE_SCALE_DEV
        )
        outcomes.add(bool(allowed))
    assert outcomes == {True, False}, (
        "旧口径在这批实现里没有出现「时过时不过」——这个对比的前提不成立了，"
        f"实际结果集合是 {outcomes}"
    )


def test_a_real_axial_motion_is_still_blocked_in_every_draw(tmp_path: Path) -> None:
    """★ 反向保险：真的轴向运动（尺度变化 1e-3 ≈ 0.68 mm）必须**每一次**都被拦。

    收紧下限的改动必须只让"其实分辨得出来"的那部分被算出来，
    不许把真实的轴向运动放进来。这里把同一个真实位移放进 40 种噪声实现里，
    判据一次都不许放行。
    """
    escaped = []
    for seed in range(DRAWS):
        estimate, allowed, why = _verdict(
            tmp_path / f"axial{seed}",
            seed,
            single_frame_baseline=False,
            scale_dev=1.0e-3,
        )
        if allowed:
            escaped.append((seed, why))
        else:
            assert "轴向" in why, why
        assert estimate.confidence == CONFIDENCE_RESOLVED, (
            f"seed={seed}：0.68 mm 的轴向位移竟然被报成「分辨不出来」：{estimate.note}"
        )
    assert not escaped, f"{len(escaped)}/{DRAWS} 次真的轴向运动没被拦住：{escaped[:3]}"


def test_the_resolution_uses_the_estimator_s_own_error_bar(tmp_path: Path) -> None:
    """可分辨下限必须按"窗口均值 − 基准均值"的误差棒算，且随帧数按 √(1/N) 变。

    三条一起看，才能说明这个下限是**算出来的**而不是拍出来的：

    * 窗口 52 帧、基准 12 帧 → 因子 √(1/52 + 1/12)；
    * 基准帧数从 12 降到 1 → 因子变大（基准那一帧的噪声照单全收）；
    * 窗口帧数变多 → 因子变小，但不能变成 0（基准那一项是地板）。
    """
    config = build_config(tmp_path, joints=("J6",), amplitudes=(0.2,), repeats=1)
    window, pool = _simulate(0, single_frame_baseline=False, scale_dev=TRUE_SCALE_DEV)
    base = estimate_depth_in_plane(
        window, config=config, noise_frames=pool, rotation_joint=True,
        reference_frames=REFERENCE_FRAMES,
    )
    factor = math.sqrt(1.0 / WINDOW_FRAMES + 1.0 / REFERENCE_FRAMES)
    expected = 3.0 * float(base.scale_noise) * factor * float(config.camera.working_distance_mm)
    assert base.resolution_mm == pytest.approx(expected, rel=1e-9), (
        f"下限 {base.resolution_mm} 不等于 3σ·√(1/N运动+1/N基准)·工作距离 = {expected}"
    )
    assert base.reference_frames == REFERENCE_FRAMES, "基准帧数没有跟着结果报出来"

    single = estimate_depth_in_plane(
        window, config=config, noise_frames=pool, rotation_joint=True, reference_frames=1
    )
    assert single.resolution_mm > base.resolution_mm, (
        "把基准退回单帧之后，下限必须**变宽**（更保守），否则说明这一项根本没进公式"
    )

    more = estimate_depth_in_plane(
        window * 4, config=config, noise_frames=pool, rotation_joint=True,
        reference_frames=REFERENCE_FRAMES,
    )
    assert more.resolution_mm < base.resolution_mm, "窗口帧数变多，下限应当变紧"
    assert more.resolution_mm > float(base.scale_noise) * 3.0 * math.sqrt(
        1.0 / REFERENCE_FRAMES
    ) * float(config.camera.working_distance_mm) * 0.99, (
        "窗口无限多帧时下限也不能低于「基准那一项」的地板"
    )
