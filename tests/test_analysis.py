"""需求四/六/七：三层分析的数算得对不对，推荐步长的规矩守没守住。

这里分两类测试，刻意分开：

* **数据层**（直接喂 ``TrialMetrics`` 给 ``summarize_amplitudes`` /
  ``recommend_steps``）。判据的规矩是"最小且全过的那一档"，只有把数字摆明了
  才看得清它到底挑了哪一档；跑一整套采集反而把这件事盖住了。
* **合成世界层**（真的采一段静态基线）。噪声基线是后面所有"信号够不够大"的
  尺子，这把尺子准不准，只能拿合成世界里**注入的真值**去对。

一条红线：**不放宽任何阈值**。要改的只有规模与时长（见 ``conftest.py``）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.analysis import (
    AmplitudeSummary,
    RangeVerdict,
    TrialMetrics,
    check_formal_range,
    recommend_steps,
    summarize_amplitudes,
)
from sj_pretest.config import JOINT_NAMES

from conftest import Recorder, build_config, open_session, read_csv_rows, run_dir_of


# --------------------------------------------------------------------------
# 装置：手搓一条试验记录
# --------------------------------------------------------------------------


def _trial(
    joint: str,
    amplitude: float,
    direction: int,
    repeat: int,
    *,
    signal: float,
    noise: float,
    actual: float | None = None,
    response: float = 0.9,
    valid: float = 1.0,
    issues: tuple[str, ...] = (),
) -> TrialMetrics:
    """造一条"看起来像真跑出来"的试验记录。

    ``signal`` 是这一档的视觉信号（J1–J5 是 px，J6 是 °），``noise`` 是静态噪声
    （同一个关节同量纲）。两者之比就是判据里的那个 SNR，所以造数据时心里要有数。
    """
    rotation = joint in ("J6",)
    trial = TrialMetrics(
        event_id=f"pretest-{joint}-a{amplitude}-d{direction:+d}-r{repeat:02d}-move-to",
        joint=joint,
        stage="pretest",
        amplitude_deg=float(amplitude),
        direction=int(direction),
        repeat=int(repeat),
        segment_id=f"seg-{joint}-{amplitude}-{direction}-{repeat}",
        commanded_delta_deg=float(amplitude) * int(direction),
    )
    trial.rtde_actual_delta_deg = (
        float(actual) * int(direction) if actual is not None else float(amplitude) * int(direction)
    )
    trial.rtde_response_ratio = float(response)
    trial.detection_valid_ratio = float(valid)
    if rotation:
        trial.static_noise_deg = float(noise)
        trial.rotation_deg = float(signal) * int(direction)
    else:
        trial.static_noise_px = float(noise)
        trial.vision_proj_px = float(signal) * int(direction)
    trial.issues = list(issues)
    return trial


def _bucket(
    joint: str,
    amplitude: float,
    *,
    signal: float,
    noise: float,
    repeats: int = 2,
    jitter: float = 0.0,
) -> list[TrialMetrics]:
    """一个关节、一个幅度、两个方向 × ``repeats`` 次重复。"""
    out: list[TrialMetrics] = []
    for direction in (1, -1):
        for repeat in range(1, repeats + 1):
            # 重复之间的差：故意让第 2 次和第 1 次差 ``jitter``，用来验证
            # "重复一致性"这条判据真的在读数，而不是永远返回通过。
            factor = 1.0 + (jitter if repeat > 1 else 0.0)
            out.append(
                _trial(
                    joint,
                    amplitude,
                    direction,
                    repeat,
                    signal=signal * factor,
                    noise=noise,
                )
            )
    return out


# --------------------------------------------------------------------------
# 判据的规矩：最小且全部通过的那一档
# --------------------------------------------------------------------------


def test_the_smallest_amplitude_that_passes_everything_wins(tmp_path: Path) -> None:
    """三档都过 → 推荐最小的那一档，不是最大的、也不是平均值。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.01, 0.05, 0.2))
    trials = (
        # 0.01°：信号 0.05 px，噪声 0.05 px → SNR 1，远低于 5 倍，不通过。
        _bucket("J1", 0.01, signal=0.05, noise=0.05)
        # 0.05°：SNR 8，通过。
        + _bucket("J1", 0.05, signal=0.40, noise=0.05)
        # 0.2°：SNR 32，也通过，但它不是最小的。
        + _bucket("J1", 0.2, signal=1.60, noise=0.05)
    )
    summaries = summarize_amplitudes(trials, config)
    by_amp = {s.amplitude_deg: s for s in summaries}
    assert by_amp[0.01].passes is False
    assert by_amp[0.01].checks["snr_ge_5"] is False
    assert by_amp[0.05].passes is True
    assert by_amp[0.2].passes is True

    recommendations = recommend_steps(summaries, config)
    j1 = next(r for r in recommendations if r.joint == "J1")
    assert j1.recommended_deg == pytest.approx(0.05)
    assert j1.needs_manual_input is False
    assert "最小" in j1.reason
    # 逐档的明细要留着：人要看得出为什么 0.01° 不行。
    assert len(j1.detail_lines) == 3
    assert "snr_ge_5" in " ".join(j1.detail_lines)


def test_only_the_snr_check_can_be_loosened_to_change_the_verdict(
    tmp_path: Path,
) -> None:
    """判据真的在读配置：把 SNR 门槛抬高，同一批数据就不通过了。

    这一条同时是"阈值可调"（需求七）的证据——如果判定是写死的，改阈值不会有
    任何反应。门槛**只往上抬**，绝不为了通过而往下调。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.05,))
    trials = _bucket("J1", 0.05, signal=0.40, noise=0.05)  # SNR = 8

    assert summarize_amplitudes(trials, config)[0].passes is True

    config.thresholds.min_snr_vs_static = 20.0  # 门槛抬到 20 倍
    stricter = summarize_amplitudes(trials, config)[0]
    assert stricter.passes is False
    assert stricter.checks["snr_ge_5"] is False


def test_larger_step_sizes_are_never_invented(tmp_path: Path) -> None:
    """都不通过时：如实说"需要人工填写更大步长"，绝不自作主张放大到 1° 或 5°。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.01,))
    trials = _bucket("J1", 0.01, signal=0.05, noise=0.05)  # SNR 1，过不了
    summaries = summarize_amplitudes(trials, config)

    j1 = next(r for r in recommend_steps(summaries, config) if r.joint == "J1")
    assert j1.recommended_deg is None, "没有一档通过，却给出了推荐步长"
    assert j1.needs_manual_input is True
    assert "人工填写更大步长" in j1.reason
    assert "自动外扩" in j1.reason
    # 推荐值（如果有）必须落在跑过的候选幅度里，不能凭空造一个更大的数。
    candidates = {0.01}
    for recommendation in recommend_steps(summaries, config):
        assert recommendation.recommended_deg in candidates or (
            recommendation.recommended_deg is None
        )


def test_repeat_inconsistency_blocks_a_step_that_is_otherwise_fine(
    tmp_path: Path,
) -> None:
    """信号够大、符号也对，但同一方向两次差太多 → 这一档不算通过。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.05,))
    clean = _bucket("J1", 0.05, signal=0.40, noise=0.05)
    assert summarize_amplitudes(clean, config)[0].passes is True

    # 第 2 次重复比第 1 次大 3 倍：远超 0.5 的相对上限，也超过 3 倍噪声。
    noisy = _bucket("J1", 0.05, signal=0.40, noise=0.05, jitter=3.0)
    verdict = summarize_amplitudes(noisy, config)[0]
    assert verdict.checks["repeat_consistent"] is False
    assert verdict.passes is False


def test_sign_consistency_is_required(tmp_path: Path) -> None:
    """回程走出来和去程同方向的位移 → 符号不一致，这一档不算通过。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.05,))
    trials = _bucket("J1", 0.05, signal=0.40, noise=0.05)
    for trial in trials:
        if trial.direction < 0:
            trial.vision_proj_px = abs(trial.vision_proj_px or 0.0)  # 该负却正
    verdict = summarize_amplitudes(trials, config)[0]
    assert verdict.sign_consistency(lambda t: t.vision_proj_px) == 0.5
    assert verdict.checks["sign_correct"] is False


def test_j6_verdict_uses_rotation_not_translation(tmp_path: Path) -> None:
    """J6 的判据只认二维转角：把平移量做得很大也不该改变结论。"""
    config = build_config(tmp_path, joints=("J6",), amplitudes=(0.05,))
    trials = _bucket("J6", 0.05, signal=0.5, noise=0.02)  # SNR = 25
    # 故意给一个同样很大的平移量：如果判据误用平移，下面改小转角就拦不住。
    for trial in trials:
        trial.vision_proj_px = 100.0 * trial.direction
    verdict = summarize_amplitudes(trials, config)[0]
    assert verdict.passes is True, "J6 的判定没有走转角，被平移量带偏了"

    for trial in trials:
        trial.rotation_deg = 0.01 * trial.direction  # 转角 SNR = 0.5
    verdict = summarize_amplitudes(trials, config)[0]
    assert verdict.passes is False, "转角小到看不见，却仍然通过了"


def test_separate_directions_are_calibrated_separately(tmp_path: Path) -> None:
    """+/- 差异单列出来（回程间隙的迹象），比的是**位移量**，不并进平均值里。

    完全对称的 ±0.40 px 必须给出 0：如果这个数报出 0.8，读起来就成了
    "正负差了 0.8 px"，而实际上两边一模一样——那是把符号当成了不对称。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.05,))
    symmetric = summarize_amplitudes(
        _bucket("J1", 0.05, signal=0.40, noise=0.05), config
    )[0]
    assert symmetric.plus_minus_diff(lambda t: t.vision_proj_px) == pytest.approx(
        0.0, abs=1e-9
    ), "对称的正负位移被报成了不对称"

    trials = _bucket("J1", 0.05, signal=0.40, noise=0.05)
    for trial in trials:
        if trial.direction < 0:
            trial.vision_proj_px = -0.25  # 回程比去程小
    summary = summarize_amplitudes(trials, config)[0]
    difference = summary.plus_minus_diff(lambda t: t.vision_proj_px)
    assert difference == pytest.approx(0.15, abs=1e-9)
    assert "plus_minus_diff" in summary.to_row()


# --------------------------------------------------------------------------
# 静态噪声基线：拿注入的真值对
# --------------------------------------------------------------------------


def test_static_noise_matches_the_injected_level(tmp_path: Path) -> None:
    """合成世界里注入的噪声是多少，量出来的就该差不多是多少。

    机制：合成世界把噪声直接加在**整张棋盘格**的平移/转角上，所以质心
    （88 个内角点的平均）量到的就是那个平移量本身，不会被平均掉。
    转角噪声多出来的一截来自角点检测本身（半径约 90 px 的画面上，
    亚像素级的检测误差换算成转角就是千分之几度）。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    injected_px = float(config.dry_run.centroid_noise_px)
    injected_deg = float(config.dry_run.rotation_noise_deg)

    session, _rec = open_session(config, recorder=Recorder(answer=True))
    try:
        session.capture_static(segment_id="static_base", duration_s=1.2)
        report = session.analyze_offline(stride=1)
    finally:
        session.close()

    noise = report.static_noise
    assert noise is not None, "采了静态基线，分析里却没有噪声基线"
    assert noise.valid_ratio > 0.95, f"静态段有效帧只有 {noise.valid_ratio:.1%}"
    assert noise.std_x_px == pytest.approx(injected_px, rel=0.2), (
        f"质心噪声 σx={noise.std_x_px:.4f} px 与注入的 {injected_px} px 差太多"
    )
    assert injected_deg * 0.9 <= noise.std_rotation_deg <= injected_deg * 2.5, (
        f"转角噪声 σ={noise.std_rotation_deg:.5f}° 超出"
        f"注入值 {injected_deg}° 的合理范围（含检测引入的那一截）"
    )
    # 漂移比噪声小一个量级才是"没人碰设备"的样子。
    assert abs(noise.drift_x_px) < 3.0 * noise.std_x_px


def test_the_baseline_duration_in_the_report_is_the_capture_not_the_processing(
    tmp_path: Path,
) -> None:
    """报告里的"静态基线 N 秒"必须是**录了多久**，不是这段离线算了多久。

    为什么单独立一条：这两个数很容易互相冒充——都是正数、都是秒、量级也只差几倍。
    干运行里 3.0 s 的静止采集曾经被写成 14.5 s（那是处理 397 帧的 wall clock），
    而这个数正好写在"静态基线"那一行的开头，读报告的人没有任何线索能看出来
    它其实在说算力。所以这里两头都验：采集时长要对得上**采集层**写下的
    ``content_seconds``，处理耗时单独一个字段，谁也别顶替谁。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    session, _rec = open_session(config, recorder=Recorder(answer=True))
    try:
        record = session.capture_static(segment_id="static_base", duration_s=1.2)
        report = session.analyze_offline(stride=1)
    finally:
        session.close()

    noise = report.static_noise
    assert noise is not None
    captured = float(record.metadata["content_seconds"])
    assert noise.seconds == pytest.approx(captured, abs=1e-6), (
        f"报告里写的是静态基线 {noise.seconds:.3f} s，采集层实际录了 {captured:.3f} s"
    )
    assert noise.captured_frames == int(record.metadata["frame_count"])
    # 处理耗时是另一件事：它必须真的是这一段的离线处理时间（> 0），
    # 而且**不能**被当成采集时长写进去。
    assert noise.process_seconds >= 0.0
    payload = noise.to_dict()
    assert payload["seconds"] == pytest.approx(captured, abs=1e-6)
    assert payload["captured_frames"] == int(record.metadata["frame_count"])
    # 文本报告里"录了多久"和"统计了多少帧"都要出现，且不能张冠李戴。
    text = "\n".join(noise.summary_lines())
    assert f"本段录了 {int(record.metadata['frame_count'])} 帧" in text, text

    # ★ 抽帧之后"录了多少"不许跟着变小。抽帧少的只是**参与统计**的帧，
    # RAW 里依然是那么多帧——这一条是反向用例：subsampled() 漏抄采集口径时，
    # 报告会写成"本段录了 50 帧"，而那一段其实录了 397 帧。
    session2, _rec2 = open_session(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,)),
        recorder=Recorder(answer=True),
    )
    try:
        record2 = session2.capture_static(segment_id="static_base", duration_s=1.2)
        report2 = session2.analyze_offline(stride=3)
    finally:
        session2.close()
    noise2 = report2.static_noise
    assert noise2 is not None
    assert noise2.captured_frames == int(record2.metadata["frame_count"]), (
        f"抽帧之后报告的采集帧数变成了 {noise2.captured_frames}，"
        f"实际录了 {int(record2.metadata['frame_count'])} 帧"
    )
    assert noise2.frames < noise2.captured_frames or noise2.captured_frames == 0
    assert noise2.seconds == pytest.approx(
        float(record2.metadata["content_seconds"]), abs=1e-6
    )


# --------------------------------------------------------------------------
# 只采了静态基线就分析（预实验被中止）：要出带表头的空表，并说清是"没测"
# --------------------------------------------------------------------------


def test_analysis_without_any_trial_says_not_measured(tmp_path: Path) -> None:
    """没有统计试验时：CSV 要有表头、要说"没测"、不能给推荐步长。

    这是现场真会走到的一条路：按钮二把静态基线和快速检查采完，操作者看到
    快速检查没过、选择暂停，然后顺手点了按钮三。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    session, _rec = open_session(config, recorder=Recorder(answer=True))
    try:
        session.capture_static(segment_id="static_base", duration_s=0.4)
        report = session.analyze_offline(stride=8)
    finally:
        session.close()

    assert report.trials == []
    assert report.static_noise is not None, "静态基线还是应该算出来的"
    assert any("没有任何统计试验" in warning for warning in report.warnings), (
        f"没有统计试验却没有任何提示：{report.warnings}"
    )

    run = run_dir_of(session)
    trials_rows = read_csv_rows(run / "analysis" / "trials.csv")
    assert trials_rows == []
    # 表头必须在：空文件会让人以为"丢东西了"，而这只是"这次没有试验"。
    header = (run / "analysis" / "trials.csv").read_text(encoding="utf-8").strip()
    assert header.startswith("event_id,"), f"trials.csv 没有表头：{header[:60]!r}"
    assert (run / "analysis" / "amplitudes.csv").read_text(encoding="utf-8").strip()
    assert (run / "analysis" / "sensitivity.csv").read_text(encoding="utf-8").strip()

    for recommendation in report.recommendations:
        assert recommendation.recommended_deg is None
        assert recommendation.needs_manual_input is True
        assert "没有跑预实验" in recommendation.reason


# --------------------------------------------------------------------------
# 正式实验前的理论范围检查（不是碰撞检查）
# --------------------------------------------------------------------------


def test_range_check_reports_the_unknown_collision_status(tmp_path: Path) -> None:
    """范围检查必须一直写着 collision unknown，不能让人误以为是碰撞检查。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    # 一级 0.2°：整段行程正好等于预实验验证过的最大幅度，仍在已验证范围内。
    verdict = check_formal_range(
        "J1", 0.2, 1, config=config, pretest_max_step_deg=0.2
    )
    assert isinstance(verdict, RangeVerdict)
    assert verdict.ok is True
    assert verdict.in_pretest_range is True
    assert verdict.note == "碰撞状态：unknown。"
    text = "\n".join(verdict.lines)
    assert "unknown" in text
    assert "不等于撞不到东西" in text, "范围检查没有写明它不等于碰撞检查"


def test_staircase_beyond_the_pretest_span_is_flagged_not_blocked(tmp_path: Path) -> None:
    """阶梯爬高本身就会超过预实验的单步幅度：必须标出来，但不拒绝执行。

    这是**设计如此**：预实验只验证过"单步 0.2°"，而正式实验是 0.2° 往上爬
    n 级，整段行程必然更大。工具该做的是说清楚"你正在离开已验证范围"，
    而不是替你决定不做——更不是悄悄放过。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    verdict = check_formal_range(
        "J1", 0.2, 3, config=config, pretest_max_step_deg=0.2
    )
    assert verdict.span_deg == pytest.approx(0.6)
    assert verdict.in_pretest_range is False
    assert verdict.ok is True, "超出预实验范围不等于理论检查不通过"
    assert "超出已验证范围" in "\n".join(verdict.lines)


def test_range_check_flags_a_step_beyond_the_pretest_range(tmp_path: Path) -> None:
    """超出预实验验证过的幅度：不能拒绝，但必须标出来并提醒谨慎。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    verdict = check_formal_range(
        "J1", 5.0, 2, config=config, pretest_max_step_deg=0.2
    )
    assert verdict.in_pretest_range is False
    assert verdict.span_deg == pytest.approx(10.0)
    text = "\n".join(verdict.lines)
    assert "超出已验证范围" in text
    assert "保留人工逐步确认" in text


def test_range_check_refuses_a_span_that_hits_the_joint_limit(tmp_path: Path) -> None:
    """行程撞到关节限位：理论上就不通过。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,))
    limit = float(config.robot.nominal_joint_deg[0])
    verdict = check_formal_range(
        "J1", 30.0, 20, config=config, pretest_max_step_deg=0.2
    )
    assert verdict.ok is False, "跨过关节限位的行程不该判成通过"
    assert any("限位" in line for line in verdict.lines)
    # 越界的那个末端要能看出来在哪个关节上。
    assert f"J1" in "\n".join(verdict.lines)
    assert abs(limit) < 360.0  # 名义位姿本身是个正常的角度


def test_session_range_checks_cover_the_pretest_joints(tmp_path: Path) -> None:
    """会话级的范围检查：只查跑过预实验的关节，并且总是以 unknown 收尾。"""
    config = build_config(tmp_path, joints=("J1", "J6"), amplitudes=(0.2,))
    session, _rec = open_session(config, recorder=Recorder(answer=True))
    try:
        lines = session.formal_range_checks({"J1": 0.2, "J6": 0.2})
    finally:
        session.close()
    assert any("J1" in line for line in lines)
    assert any("J6" in line for line in lines)
    assert not any("J2" in line for line in lines), "没跑预实验的关节不该出现在范围检查里"
    assert lines[-1].strip().startswith("以上是名义运动学检查")
    assert "collision_status 始终是 unknown" in lines[-1]
