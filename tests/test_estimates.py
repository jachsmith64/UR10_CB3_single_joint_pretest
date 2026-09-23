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

from sj_pretest.config import (
    AppConfig,
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


def test_disk_estimate_follows_the_frame_size(tmp_path: Path) -> None:
    """同样长的一段，满幅的占用必须是裁剪后的好几倍——这才是 ROI 的意义。"""
    config = _hardware(
        build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    )
    plan = build_static_plan(config.robot.nominal_joint_deg, duration_s=15.0)

    config.camera.roi = [0, 0, 800, 600]
    cropped = estimate_disk_gb(config, estimate_capture_seconds(config, [plan]))
    config.camera.roi = None
    full = estimate_disk_gb(config, estimate_capture_seconds(config, [plan]))

    assert cropped > 0
    assert full > cropped * 3.0, (
        f"满幅({full:.2f} GB)居然没比裁剪({cropped:.2f} GB)大多少，"
        "说明估算没读相机尺寸"
    )


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
    assert "800×600" in text
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
