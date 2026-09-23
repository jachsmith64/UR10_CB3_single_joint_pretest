"""开跑之前的规模估算：整场要多久、要占多少盘。

为什么要专门测它：这个数会显示在界面日志的第一屏，也会被当成磁盘门槛
（写不下就拒绝开始）。它错的方向只有两个，而两个都很贵——
**估小了**，人以为二十分钟的事结果做了两小时，中途磁盘写满；
**估大了**，工具会拒绝一次其实跑得完的实验，现场只能干瞪眼。

所以这里不测"某个具体数字对不对"（那取决于配置），测三条性质：
估算跟着帧尺寸走、跟着正式实验的**慢速度**走、缺步长时如实说明没算它。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import (
    AppConfig,
    ConfigError,
    estimate_capture_seconds,
    estimate_disk_gb,
)
from sj_pretest.joint_space import (
    build_formal_group_a,
    build_pretest_plan,
    build_static_plan,
    planned_plans,
    planned_scale_lines,
)

from conftest import build_config


def _hardware(config: AppConfig, roi: list[int] | None = None) -> AppConfig:
    config.mode = "hardware"
    config.camera.roi = roi
    config.validate()
    return config


def test_disk_estimate_uses_the_full_frame_not_the_analysis_roi(tmp_path: Path) -> None:
    """★ v1.0.3：RAW 一律整幅保存，所以估算**不许**跟着 analysis_roi 变小。

    以前这里正相反（配了 ROI 就按 ROI 的宽高估，估出来小好几倍）。现在那是错的：
    离线 ROI 只在机器人停住之后用来减少背景干扰、加快识别，RAW 里存的仍是整幅。
    拿 ROI 尺寸估会把这个组"要不要开始"的闸门整个架空——估 20 GB，实际写 200 GB。
    """
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    )
    plan = build_static_plan(config.robot.nominal_joint_deg, duration_s=15.0)
    seconds = estimate_capture_seconds(config, [plan])

    no_roi = estimate_disk_gb(config, seconds)
    config.camera.analysis_roi = [0, 0, 800, 600]
    config.validate()
    with_roi = estimate_disk_gb(config, seconds)
    assert with_roi == no_roi, (
        f"配了分析 ROI 之后估算从 {no_roi:.2f} GB 变成了 {with_roi:.2f} GB——"
        "分析 ROI 不该影响 RAW 大小"
    )

    # 真正决定估算的是**传感器满幅**：换一块更大的传感器，估算必须跟着涨。
    config.camera.sensor_width = 800
    config.camera.sensor_height = 600
    smaller = estimate_disk_gb(config, seconds)
    assert no_roi > smaller * 3.0, (
        f"把传感器改成 800×600 之后估算没跟着变（{smaller:.2f} vs {no_roi:.2f}）——"
        "说明估算没读传感器尺寸"
    )


def test_the_legacy_roi_field_still_works_but_never_shrinks_the_raw(tmp_path: Path) -> None:
    """旧字段 ``camera.roi`` 还能用（当分析 ROI），但**不再**让 RAW 变小。"""
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    )
    plan = build_static_plan(config.robot.nominal_joint_deg, duration_s=15.0)
    seconds = estimate_capture_seconds(config, [plan])
    baseline = estimate_disk_gb(config, seconds)

    # 构造之后再赋值（界面的参数面板就是这么干的）：必须照样生效。
    config.camera.roi = [10, 20, 700, 500]
    config.validate()
    assert config.camera.resolved_analysis_roi() == (10, 20, 700, 500)
    assert estimate_disk_gb(config, seconds) == baseline
    assert "旧字段 camera.roi" in config.camera.roi_migration_note()

    # 两个字段都写且不一致：必须报错，不许悄悄挑一个。
    config.camera.analysis_roi = [1, 2, 3, 4]
    with pytest.raises(ConfigError):
        config.validate()


def test_capture_estimate_uses_the_formal_speed_for_formal_plans(
    tmp_path: Path,
) -> None:
    """正式实验比预实验慢一个量级，估时长必须挑对速度。

    挑错方向很危险：正式实验每级要走近 3 s，用预实验的 0.5 °/s 去估会
    把一个小时的组A 估成几分钟，磁盘门槛就形同虚设。
    """
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    )
    config.formal.step_deg = {"J1": 0.2}
    config.validate()

    formal = build_formal_group_a(
        config, config.robot.nominal_joint_deg, "J1", step_deg=0.2
    )
    legs = sum(1 for step in formal.steps if step.is_motion)
    assert legs == (
        config.formal.staircase_n * 2 * config.formal.repeats
    ), f"组A 的动作数不对：{legs}"

    seconds = estimate_capture_seconds(config, [formal])
    # 每级 0.2°、0.05 °/s、0.1 °/s² 的三角速度规划 ≈ 2.83 s，再加上保持段。
    per_leg = seconds / legs
    assert per_leg > 2.5, (
        f"每级只估了 {per_leg:.2f} s，像是用了预实验的 0.5 °/s 而不是正式的 0.05 °/s"
    )

    # 换成预实验的速度去估，同样的计划必须明显更短——证明速度真的被读进去了。
    config.formal.speed_deg_s = float(config.robot.trial_speed_deg_s)
    config.formal.accel_deg_s2 = float(config.robot.trial_accel_deg_s2)
    faster = estimate_capture_seconds(config, [formal])
    assert faster < seconds * 0.6, f"把正式速度改快之后估算没变：{faster} vs {seconds}"


def test_planned_scale_lines_reports_minutes_and_gigabytes(tmp_path: Path) -> None:
    """整场规模要能用一句中文说清楚：多少分钟、多少 GB、按哪个尺寸算的。"""
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1),
        roi=[0, 0, 800, 600],
    )
    config.formal.step_deg = {"J1": 0.2}
    config.validate()
    text = "\n".join(planned_scale_lines(config))
    assert "整场实验规模" in text
    assert "分钟" in text and "GB" in text
    # ★ v1.0.3：这里的尺寸是**全屏**尺寸，不是分析 ROI 的尺寸。
    assert "1936×1096" in text
    assert "800×600" not in text, "整场估算还在按分析 ROI 的尺寸报数"
    assert "估算" in text, "要明说这是估算，不是承诺"


def test_planned_scale_lines_names_the_joints_it_did_not_count(tmp_path: Path) -> None:
    """没填正式步长的关节，报告里必须明说"这个数没算它"。

    默认配置的 ``formal.step_deg`` 是空的（步长要等按钮三给出推荐值、
    由人确认），所以整场估算天然缺正式实验那一大块——这件事必须说出来，
    否则人会把"四分钟"当成整场的时间。
    """
    config = AppConfig()
    text = "\n".join(planned_scale_lines(config))
    assert "没有算" in text
    for joint in config.pretest.joints:
        assert joint in text, f"漏说了没算哪个关节（{joint}）"

    config.formal.step_deg = {joint: 0.2 for joint in config.pretest.joints}
    config.validate()
    text = "\n".join(planned_scale_lines(config))
    assert "没有算" not in text
    assert "组A/组B" in text


def test_planned_plans_covers_every_stage_that_records(tmp_path: Path) -> None:
    """整场估算要覆盖所有会录盘的计划：静态、快速确认、预实验、组A、组B。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    plans = planned_plans(config)
    stages = {
        step.event.stage
        for plan in plans
        for step in plan.steps
        if step.is_motion
    }
    assert {"quick_probe", "pretest", "formal_a", "formal_b"} <= stages, (
        f"整场估算漏了阶段：{stages}"
    )
    # 静态基线是"只等待、不运动"的一段，所以它不在上面那个集合里——
    # 但它必须在计划清单里，否则整场估算会漏掉静态噪声那一大段。
    static_plans = [plan for plan in plans if plan.name.startswith("静态")]
    assert static_plans, "整场估算里没有静态基线的计划"
    assert sum(float(step.hold_s) for step in static_plans[0].steps) > 0

    # 关掉组B 就真的不算它——估算要跟着配置走。
    config.formal.enable_group_b = False
    config.validate()
    stages_off = {
        step.event.stage
        for plan in planned_plans(config)
        for step in plan.steps
        if step.is_motion
    }
    assert "formal_b" not in stages_off
    assert "formal_a" in stages_off


def test_estimate_never_touches_devices() -> None:
    """估算是纯计算：不建会话、不连设备、不导入 ur_rtde。"""
    import sys

    config = AppConfig()  # 交付默认：6 关节 × 3 档 × 2 方向 × 2 次 = 72 次统计动作
    for line in planned_scale_lines(config):
        assert line
    assert "ur_rtde" not in sys.modules
    plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
    assert sum(1 for step in plan.steps if step.event.counts_for_statistics) == 72
