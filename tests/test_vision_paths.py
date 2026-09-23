"""需求八.9/八.10 与需求四：J1–J5 走质心平移，J6 走二维旋转。

需求四写死了两条不同的视觉路径，走错一条不会报错，只会给出一个看着像真的、
其实没意义的数字。所以这里断言的不是"有个数"，而是**这个数是哪条路算出来的**：

* J1–J5：灵敏度来自 ``质心位移 ÷ RTDE 实际角变化``；J6 的旋转量不得参与定性。
* J6：只有 88 角点绕质心的二维 Kabsch 转角，绝不套用质心平移那条路。

合成世界里的"真值"是配置里显式写下的（``dry_run.centroid_px_per_deg`` 等），
所以这里可以拿实测灵敏度去和真值比——**这是判据正确性的证据，不是拟合**。
"""

from __future__ import annotations

import statistics

import pytest

from sj_pretest.config import CENTROID_JOINTS, JOINT_NAMES, ROTATION_JOINTS
from sj_pretest.experiment import angle_difference

from conftest import all_trials, pretest_report, trial_of


def test_joint_partition_is_the_documented_one() -> None:
    """需求四的分工：J1–J5 用质心平移，J6 用二维旋转。"""
    assert tuple(CENTROID_JOINTS) == ("J1", "J2", "J3", "J4", "J5")
    assert tuple(ROTATION_JOINTS) == ("J6",)
    assert sorted(list(CENTROID_JOINTS) + list(ROTATION_JOINTS)) == list(JOINT_NAMES)


@pytest.mark.parametrize("joint", ["J1", "J6"])
def test_analysis_uses_the_documented_path(session_factory, joint: str) -> None:
    # 只跑 0.2° 这一档：路径对不对和有多少档幅度无关，档数越多只是越慢。
    # J6 用步长 1，理由见下面 J6 分支里的注释（保持窗口里帧太少就没法取平均）。
    session, _recorder, config = session_factory(
        joints=(joint,), amplitudes=(0.2,), repeats=2
    )
    report = pretest_report(session, stride=1 if joint in ROTATION_JOINTS else 4)
    biggest = trial_of(report, joint, amplitude=0.2, direction=1)

    assert biggest.detection_valid_ratio >= 0.8, "识别有效率太低，后面的数字都不可信"
    # 两条路径都要给信噪比：J1–J5 是"像素÷像素"，J6 是"度÷度"。
    assert biggest.snr is not None, "没有算信噪比：说明静态基线没进分析"

    if joint in ROTATION_JOINTS:
        # J6：走旋转。旋转量必须有值，质心平移那条路必须一个字都没参与。
        assert biggest.rotation_deg is not None, "J6 没有量到二维转角"
        assert biggest.direction_source == "rotation_only"
        assert biggest.sensitivity_source == "rotation_deg_per_deg"
        assert biggest.rotation_equiv_deg is not None, "J6 没有把转角换算成等效角度"
        assert biggest.vision_proj_px is None, (
            "J6 也算了沿方向的质心投影——需求四要求 J6 只走二维旋转"
        )

        # 为什么这里对**全部 J6 试验取平均**，而不是逐条卡同一个容差：
        # 合成棋盘格是 16 px/格，0.2° 的 J6 旋转只让角点沿切向挪动约 0.35 px，
        # 单帧转角估计的散布与信号本身同量级（实测每帧 ±0.1° 量级）。
        # 保持窗口只有 0.4 s 的一半、再扣掉两端各 0.05 s 余量，132 fps 下十几帧，
        # 步长一大就只剩一两帧——那时候报出来的是一次抽样，不是均值。
        # 所以这里用步长 1 取满帧数，再对所有 J6 试验取平均：这才是分析报告
        # 真正在说的那句话"这一档量到的转角是多少"。
        trials = all_trials(report, joint, amplitude=0.2)
        assert len(trials) == 4, f"应当有 2 方向 × 2 次重复 = 4 条 J6 试验，实际 {len(trials)}"
        for trial in trials:
            assert trial.steady_frames >= config.thresholds.min_window_frames, (
                f"{trial.event_id} 的保持窗口只有 {trial.steady_frames} 帧有效数据"
            )
            assert not any("帧有效数据" in issue for issue in trial.issues), (
                f"{trial.event_id} 报了“窗口帧数太少”：这个测试的步长没取对"
            )
            # 符号是精确判据：+方向的转角必须为正，-方向必须为负。
            assert trial.rotation_deg is not None
            assert (trial.rotation_deg > 0) == (trial.direction > 0), (
                f"{trial.event_id}（direction={trial.direction:+d}）量到的转角 "
                f"{trial.rotation_deg:+.4f}° 符号反了"
            )
        measured = statistics.fmean(abs(t.rotation_deg) for t in trials)
        actual = statistics.fmean(abs(t.rtde_actual_delta_deg) for t in trials)
        # 合成世界的旋转灵敏度是 1.0 °/°：0.2° 的 J6 动作应当量到约 0.2° 的转角。
        assert measured == pytest.approx(actual, rel=0.3), (
            f"J6 量到的平均画面转角 {measured:.4f}° 与 RTDE 实际角变化 "
            f"{actual:.4f}° 相差超过 30%"
        )
        return

    # J1–J5：走质心平移。方向由 0.2° 档正负两组分别标定（或退回理论方向），
    # 灵敏度分母必须是 RTDE 实际角变化，不能是别的。
    assert biggest.vision_proj_px is not None, "没有沿局部方向的标量投影"
    assert biggest.direction_source in ("measured_0.2deg", "theoretical")
    assert biggest.sensitivity_source == "px_per_deg_from_actual_q", (
        "灵敏度不是用 RTDE 实际角算的；需求四要求优先用实际角，退化时必须标注"
    )
    assert biggest.sensitivity_px_per_deg is not None
    truth = float(config.dry_run.centroid_px_per_deg[joint])
    assert biggest.sensitivity_px_per_deg == pytest.approx(truth, rel=0.25), (
        f"{joint} 实测灵敏度 {biggest.sensitivity_px_per_deg:.1f} px/° "
        f"与合成世界的真值 {truth:.1f} px/° 相差超过 25%"
    )
    assert biggest.vision_equiv_deg is not None, "没有把像素换算成等效角度"


def test_coarse_stride_is_reported_as_too_few_frames(session_factory) -> None:
    """步长太大时保持窗口里只剩一两帧——必须报出来，不能闷声给个数字。

    J6 的转角就是从"保持段后半段"这一个窗口里算出来的，而它本来就只有
    0.4 s 的一半、再扣两端余量。步长一大，里面的帧数就掉到个位数，
    报出来的是一次抽样而不是均值。这一条是提醒，**不是判废**：
    数据没坏，换个步长重算就行，所以它不许出现在"拦停"那类措辞里。
    """
    session, _recorder, config = session_factory(
        joints=("J6",), amplitudes=(0.2,), repeats=1
    )
    report = pretest_report(session, stride=8)
    trial = trial_of(report, "J6", amplitude=0.2, direction=1)

    assert trial.steady_frames < config.thresholds.min_window_frames, (
        f"步长 8 时保持窗口居然还有 {trial.steady_frames} 帧，"
        "那这条提示就永远触发不了——先确认窗口是怎么切的"
    )
    assert any("抽样" in issue for issue in trial.issues), (
        f"步长 8 时窗口只剩 {trial.steady_frames} 帧，分析里却没有任何提示"
    )
    assert not any(
        "无法计算" in issue or "超时" in issue for issue in trial.issues
    ), "这只是提醒，不该把试验判成无效"


def test_j1_5_do_not_use_rotation_as_the_verdict(session_factory) -> None:
    """J1–J5 的结论不能建立在旋转量上（旋转是 J6 的路径）。

    画面转角对 J1–J5 来说是"顺手算出来的副产品"，它不参与定性；
    这里用"J1 的旋转等效角度与平移等效角度不是同一个数"来证明两条路是分开的。
    """
    session, _recorder, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    report = pretest_report(session)
    trial = trial_of(report, "J1", amplitude=0.2, direction=1)
    assert trial.vision_equiv_deg is not None
    assert trial.rotation_equiv_deg is None, (
        "J1 也算了旋转等效角度——说明 apply_sensitivity 没有按关节分工"
    )


def test_direction_units_are_calibrated_per_side(session_factory) -> None:
    """需求四：正负方向分别标定，两者应当大致反向（夹角接近 180°）。"""
    session, _recorder, config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=2
    )
    report = pretest_report(session)
    direction = report.directions["J1"]
    assert direction.plus_unit is not None and direction.minus_unit is not None, (
        "正负方向没有分别标定"
    )
    assert direction.separation_deg is not None
    assert direction.separation_deg > 90.0, (
        f"正负位移夹角只有 {direction.separation_deg:.1f}°，正负方向没有分开"
    )
    # 合成世界里 J1 的画面位移方向是 0°（+J1 把棋盘格推向图像 +x）。
    # 角度要按圆周比：357° 和 0° 只差 3°，直接用 approx 比会当成"差 357°"。
    truth = float(config.dry_run.image_direction_deg["J1"]) % 360.0
    if direction.source == "measured_0.2deg":
        difference = abs(angle_difference(direction.direction_deg, truth))
        assert difference <= 25.0 or abs(difference - 180.0) <= 25.0, (
            f"实测方向 {direction.direction_deg:.1f}° 与真值 {truth:.1f}° "
            f"相差 {difference:.1f}°"
        )


def test_vision_layer_reports_noise_relative_signal(session_factory) -> None:
    """第三层要看"停稳之后画面还动不动"：残余量必须与静态噪声一起给出。"""
    session, _recorder, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    report = pretest_report(session)
    trial = trial_of(report, "J1", amplitude=0.2, direction=1)
    assert report.static_noise is not None, "静态基线没有进报告"
    assert trial.static_noise_px is not None
    assert trial.residual_px is not None, "没有算停稳后的残余位移"
    assert trial.residual_ratio is not None
    assert trial.vision_moves_after_settle is not None, (
        "没有给出“RTDE 稳了之后画面还动不动”的结论"
    )


def test_sensitivity_falls_back_to_command_when_actual_is_missing(session_factory) -> None:
    """需求四的降级要求：拿不到 RTDE 实际角时用指令角，并且**必须标注**降级。

    做法是把 robot_states.csv 里的实际角清空——这正是"RTDE 没采到"在数据里的样子。
    """
    from sj_pretest.analysis import analyze_pretest
    from sj_pretest.vision import process_segment

    session, _recorder, _config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    session.capture_static(segment_id="static_base", duration_s=0.4)
    session.run_pretest()
    assert session.run is not None

    segments = {}
    for segment_id in ["static_base"] + [str(t["segment_id"]) for t in session.trials]:
        segments[segment_id] = process_segment(
            session.run.segment_dir(segment_id),
            config=session.config,
            segment_id=segment_id,
            save_corners=False,
            stride=2,
        )
    rows = [
        dict(row, actual_q_J1_deg=None, actual_q_J2_deg=None, actual_q_J3_deg=None,
             actual_q_J4_deg=None, actual_q_J5_deg=None, actual_q_J6_deg=None)
        for row in _read_rows(session.run.root / "robot_states.csv")
    ]
    report = analyze_pretest(
        config=session.config,
        segments=segments,
        trials=session.trials,
        rtde_rows=rows,
        static_segment_ids=["static_base"],
    )
    trial = trial_of(report, "J1", amplitude=0.2, direction=1)
    assert trial.sensitivity_source == "px_per_deg_from_command_degraded", (
        "没有实际角时应当降级用指令角，并标注出来"
    )
    sensitivity = report.sensitivities[("J1", 1)]
    assert "退化" in sensitivity.note, "降级估计必须写清楚是退化的，不能长得像正常结果"


def _read_rows(path):
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
