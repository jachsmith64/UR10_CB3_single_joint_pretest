"""分组流水线：**一组动作走完 → 整组处理 → 校验 → 删这一组 RAW → 下一组**。

为什么这条要单独测
------------------
"边采边清"（v1.0.1）是**按段**处理：一段采完就地识别、就地删。到了现场发现两件事：

1. 处理期间机械臂必须停着，按段处理会把"停机"切碎成几十次，人得等几十回；
2. 峰值一点没降下来——峰值本来就是"同时驻留在盘上的那些段"，
   按段处理只是把删除提前了，整组驻留的字节数不变。

所以 v1.0.2 改成**分组**：静态基线一组、快速几何检查一组、预实验每个关节一组、
正式实验"一个关节的组A/组B"各一组。任何时刻盘上只有当前这一组。

这里测四件事：

1. **组的划分**就是上面那几条（看事件流里的 ``group_started``）；
2. **处理发生在整组采完之后**——事件顺序必须是"组内全部 ``segment_captured``
   → ``group_processing`` → ``raw_deleted``"，不许中间插队；
3. **峰值落在预警线以内**：按交付默认参数（全屏 1936×1096 @ 132.23 fps、
   正式实验 Δ=0.2°）算出每一遍 repeat 的估算峰值，都要 ≤ 预警线；
   ★ v1.0.3 起磁盘估算**一律按全屏 RAW 算**，``camera.analysis_roi``
   再怎么改都**不得**改变估算值；
4. **超估算就不得开始下一组**：闸门拒绝时**一个运动命令都不能发出去**；
   单遍 repeat 自己就超硬上限时，只在"已回到名义姿态"的边界上再拆，
   拆不动（组 A）就拒绝开始——见 ``test_formal_repeat_grouping.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import (
    AppConfig,
    check_group_disk,
    group_forecast,
    group_peak_gb,
)
from sj_pretest.experiment import ExperimentError

from conftest import build_config, open_session


# --------------------------------------------------------------------------
# 1) 组的划分 + 2) 处理时机
# --------------------------------------------------------------------------


def _group_events(session) -> list[dict]:
    assert session.run is not None
    return [
        item
        for item in session.run.events.read_all()
        if item.get("event", "").startswith("group_")
    ]


def test_the_groups_are_exactly_the_documented_ones(tmp_path: Path) -> None:
    """组就是需求一写的那几组，不多不少。"""
    config = build_config(
        tmp_path,
        joints=("J1", "J6"),
        amplitudes=(0.2,),
        repeats=1,
        paths__delete_raw_after_process=True,
    )
    session, _recorder = open_session(config, run_kind="groups")
    try:
        session.run_static()
        session.run_quick_probes()
        session.run_pretest()
    finally:
        session.close()

    started = [
        str(item["group"]) for item in _group_events(session) if item["event"] == "group_started"
    ]
    assert started == [
        "静态基线",
        "快速几何检查",
        "预实验 J1",
        "预实验 J6",
    ], f"组的划分和需求一对不上：{started}"

    # 每一组都要有始有终：开了就要有 group_processing / group_finished。
    finished = [
        str(item["group"]) for item in _group_events(session) if item["event"] == "group_finished"
    ]
    assert finished == started, f"有组开了没收尾：{started} vs {finished}"


def test_processing_happens_after_the_whole_group_was_captured(tmp_path: Path) -> None:
    """处理必须在**整组采完、机械臂停下**之后，中间不许插队。

    现场要求的原话是"采集一组动作 RAW → 机械臂保持停止 → 完整处理该组"。
    如果处理插在组内两段之间，那处理的时候机械臂正在准备下一个动作，
    既不是"保持停止"，峰值也没有降下来。
    """
    config = build_config(
        tmp_path,
        joints=("J1",),
        amplitudes=(0.2, 0.5),
        repeats=2,
        paths__delete_raw_after_process=True,
    )
    session, _recorder = open_session(config, run_kind="order")
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    events = session.run.events.read_all()

    # 顺着事件流走一遍：遇到 group_started 就换组，中间的 segment_captured
    # 就算这一组的；遇到 group_processing 就记下"这一组从哪一刻开始处理"。
    captured_in: dict[str, list[str]] = {}
    processing_at: dict[str, int] = {}
    deleted_at: dict[str, int] = {}
    current: str | None = None
    for index, item in enumerate(events):
        event = item.get("event")
        if event == "group_started":
            current = str(item["group"])
            captured_in.setdefault(current, [])
        elif event == "segment_captured" and current is not None:
            captured_in[current].append(str(item["segment_id"]))
        elif event == "group_processing":
            processing_at[str(item["group"])] = index
        elif event == "raw_deleted":
            deleted_at[str(item["segment_id"])] = index

    assert processing_at, "一个 group_processing 事件都没有：分组流水线没跑起来"
    assert len(captured_in) == len(processing_at), (
        f"开了没处理的组：{set(captured_in) - set(processing_at)}"
    )
    # 预实验那一组至少要有两段（两次动作），才谈得上"整组采完再处理"。
    pretest_group = next(name for name in captured_in if name.startswith("预实验"))
    assert len(captured_in[pretest_group]) >= 2, (
        f"{pretest_group} 只有 {len(captured_in[pretest_group])} 段，测不出组内的先后"
    )

    for group, segment_ids in captured_in.items():
        assert segment_ids, f"{group}：一段都没采"
        index = processing_at[group]
        # 1) 处理开始时，本组的段**全部**已经采完。
        for segment_id in segment_ids:
            assert segment_id in deleted_at, (
                f"{group}：{segment_id} 采了却没处理（也没有 raw_kept 之类的下落）"
            )
            assert deleted_at[segment_id] > index, (
                f"{group}：{segment_id} 在本组处理开始（事件 {index}）之前就被删了"
                "——说明还是按段删的，不是按组"
            )
        # 2) 本组最后一段的采集必须早于处理开始。
        last_captured = max(
            i
            for i, item in enumerate(events)
            if item.get("event") == "segment_captured"
            and str(item.get("segment_id")) in segment_ids
        )
        assert last_captured < index, (
            f"{group}：处理在事件 {index} 开始，但最后一段到 {last_captured} 才采完"
        )


def test_the_group_is_flushed_at_the_session_boundary(tmp_path: Path) -> None:
    """会话收尾时，还挂着的那一组必须被处理掉——不许留着 RAW 就走人。"""
    config = build_config(
        tmp_path,
        joints=("J1",),
        amplitudes=(0.2,),
        repeats=1,
        paths__delete_raw_after_process=True,
    )
    session, _recorder = open_session(config, run_kind="tail")
    try:
        # 只采，不跑任何 runner：这一组没有任何显式的组边界。
        session.capture_static(segment_id="static_base", duration_s=0.6)
    finally:
        session.close()
    assert session.run is not None
    deleted = [
        item for item in session.run.events.read_all() if item.get("event") == "raw_deleted"
    ]
    assert deleted, "会话都关了，最后那一组的 RAW 还在盘上"


def test_the_group_boundaries_are_paired_even_when_the_switch_is_off(tmp_path: Path) -> None:
    """开关关着（**交付默认**）时，组头也要成对收口。

    分组流水线关着的时候组里没有待处理的段，但 ``group_started`` 照样会写——
    组边界的磁盘闸门和日志都在。要是收尾只写在"有待处理段"的分支里，
    事件流里就会留下一串配不上 ``group_finished`` 的 ``group_started``，
    事后按组对账（"这几组到底走完没有"）会得出错的结论。
    默认配置下跑一遍默认路径，事件必须成对。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    assert config.paths.delete_raw_after_process is False, (
        "这条测的就是默认（开关关着）的路径，别把开关打开"
    )
    session, _recorder = open_session(config, run_kind="pairs")
    try:
        session.run_static()
        session.run_pretest()
    finally:
        session.close()

    events = _group_events(session)
    started = [str(item["group"]) for item in events if item["event"] == "group_started"]
    finished = [str(item["group"]) for item in events if item["event"] == "group_finished"]
    assert started, "默认配置下连组都没有声明"
    assert finished == started, f"开关关着时组边界没收口：{started} vs {finished}"
    # 空组收尾必须如实记成 0 段，不能拿它冒充"处理过一段"。
    for item in events:
        if item["event"] == "group_finished":
            assert int(item["segments"]) == 0, item
            assert int(item["released"]) == 0 and int(item["kept"]) == 0, item


# --------------------------------------------------------------------------
# 3) 峰值估算：交付默认参数必须落在预警线以内
# --------------------------------------------------------------------------


def _group_plan_map(config: AppConfig) -> dict[str, list]:
    """把"整场实验"按需求一的划分拆成组，供峰值估算用。"""
    from sj_pretest.experiment import iter_segment_plans
    from sj_pretest.joint_space import (
        build_formal_group_a,
        build_formal_group_b,
        build_pretest_plan,
        build_quick_probe_plan,
        build_static_plan,
    )

    nominal = config.robot.nominal_joint_deg
    durations = config.effective_durations()
    groups: dict[str, list] = {
        "静态基线": iter_segment_plans(
            build_static_plan(nominal, duration_s=float(durations["static"]))
        ),
        "快速几何检查": iter_segment_plans(
            build_quick_probe_plan(
                nominal,
                config.pretest.joints,
                probe_deg=float(config.pretest.quick_probe_deg),
                hold_s=float(config.camera.hold_s),
                return_settle_s=float(config.pretest.return_before_settle_s),
            )
        ),
    }
    for segment in iter_segment_plans(build_pretest_plan(config, nominal)):
        groups.setdefault(f"预实验 {segment.primary.event.joint}", []).append(segment)
    for joint in config.pretest.joints:
        step = config.formal.step_deg.get(joint)
        if not step:
            continue
        if config.formal.enable_group_a:
            groups[f"组A {joint}"] = iter_segment_plans(
                build_formal_group_a(config, nominal, joint, float(step))
            )
        if config.formal.enable_group_b:
            groups[f"组B {joint}"] = iter_segment_plans(
                build_formal_group_b(config, nominal, joint, float(step))
            )
    return groups


def _formal_group_plans(config: AppConfig, group: str, joint: str) -> list:
    """按需求一·4 把"这一关节的组 A / 组 B"切成本次真正要跑的组。"""
    from sj_pretest.experiment import iter_segment_plans, plan_formal_groups
    from sj_pretest.joint_space import build_formal_group_a, build_formal_group_b

    builder = build_formal_group_a if group.upper() == "A" else build_formal_group_b
    nominal = config.robot.nominal_joint_deg
    step = float(config.formal.step_deg[joint])
    plans = iter_segment_plans(builder(config, nominal, joint, step))
    return plan_formal_groups(config, plans, nominal, group, joint)


def test_the_delivered_group_peaks_stay_under_the_warning_line() -> None:
    """交付默认参数下，每一个正式实验组的估算峰值都要在预警线以内。

    ★ v1.0.3 的组边界是**每一遍 repeat 一组**，尺寸口径是**全屏 RAW**
    （1936×1096 @ 132.23 fps、每帧 2 122 336 字节、约 280 MB/s），
    而且比的是 40/50 GB 两条线真正卡的**那个**数：同时驻留峰值
    （RAW + 派生 + 处理期临时余量，见 ``GroupForecast.peak_gb``——
    需求二的原话是"同时驻留的 RAW + 临时文件 + 本组派生文件估算值 ≤ 50 GB"）。
    按真机口径（``mode="hardware"``、Δ=0.2°、阶梯 5 级、重复 3 遍）算出来：
    组 A 一遍约 19.3 GB、组 B 一遍约 38.5 GB，六个关节都一样
    （帧数只由时长决定，跟哪个关节无关）。

    ★ 组 B 一遍 38.49 GB 距 40 GB 预警线只剩 1.51 GB。这不是余量不够，
    是"交付举例用的那个 Δ=0.2° 本来就在预警线附近"：Δ 更大时组 B 会被预警，
    再大就会被硬拦并自动在"回到名义姿态"的地方（正负循环之间）再拆。
    现场要跑更大的 Δ，就先清盘或换一个更大的盘。

    注意 ``mode`` 必须是 hardware：干运行把画面缩到 340×260，算出来的占用
    和现场不是一个量级，拿它验"现场放不放得下"没有意义。
    """
    config = AppConfig()
    config.mode = "hardware"
    config.formal.step_deg = {joint: 0.2 for joint in config.pretest.joints}
    config.validate()
    # 交付参数下**不需要**再拆细一级：按遍分就已经放得下。
    # 这一条要是失败，说明默认参数已经大到要现场多停好几次了。
    assert config.camera.analysis_roi is None, (
        "交付默认不该带分析 ROI：ROI 只影响分析速度，不影响 RAW 尺寸"
    )
    cap_gb = float(config.paths.max_peak_disk_gb)
    warn_gb = float(config.paths.disk_warn_gb)

    per_group: dict[str, float] = {}
    for joint in config.pretest.joints:
        for group in ("A", "B"):
            plans = _formal_group_plans(config, group, joint)
            for item in plans:
                assert item.level == "repeat", (
                    f"交付参数下 {item.name} 竟然要拆到「{item.level}」这一级："
                    "默认参数不该离硬上限这么近"
                )
                forecast = group_forecast(config, item.segments)
                per_group[item.name] = forecast.peak_gb
                # ★ 界面上"是否低于 40/50 GB"那两行，必须与闸门真正据以放行的
                # 判据**同源**（都是 peak_gb）。显示一个数、执行另一个数的话，
                # 现场看到的"是"就不是程序实际用的那个结论。
                assert forecast.under_cap == (forecast.peak_gb <= cap_gb), item.name
                assert forecast.under_warn == (forecast.peak_gb <= warn_gb), item.name
                assert forecast.resident_gb < forecast.peak_gb, (
                    "驻留量与同时驻留峰值应当差着处理期那部分余量，"
                    "两个数相等说明 process_headroom 没起作用"
                )

    assert per_group, "一个正式实验组都没算出来"
    worst = max(per_group, key=lambda name: per_group[name])
    limits = {"A": 19.5, "B": 39.5}
    for name, peak_gb in sorted(per_group.items()):
        limit = limits["A"] if "组A " in name else limits["B"]
        assert peak_gb <= limit, (
            f"「{name}」估算峰值 {peak_gb:.2f} GB，超过 v1.0.3 定稿实测上限 {limit} GB"
        )
        assert peak_gb <= warn_gb, (
            f"「{name}」估算峰值 {peak_gb:.2f} GB 越过预警线 "
            f"{warn_gb:.0f} GB——交付默认参数会一路报警"
        )
        assert peak_gb <= cap_gb
    assert "组B" in worst, f"最坏一组竟然是「{worst}」——组 B 才是最大的那一组"
    assert per_group[worst] == pytest.approx(38.49, abs=0.5), (
        f"最坏一组「{worst}」估算 {per_group[worst]:.2f} GB，与定稿的 38.49 GB "
        "差得太多——估算口径（全屏尺寸/帧率/时长模型/处理余量）被改动过？"
    )
    assert warn_gb - per_group[worst] >= 1.0, (
        f"最坏一组「{worst}」距预警线只剩 {warn_gb - per_group[worst]:.2f} GB："
        "交付举例用的 Δ 不该贴着预警线"
    )
    assert float(config.paths.max_peak_disk_gb) == 50.0, (
        "硬上限被改过了——需求一写的是 50 GB"
    )
    assert 35.0 <= float(config.paths.disk_warn_gb) <= 40.0, (
        "预警线要落在需求一建议的 35～40 GB 区间里"
    )


def test_the_analysis_roi_never_changes_the_disk_estimate() -> None:
    """★ 需求一·2：磁盘估算**一律按全屏 RAW 算**，分析 ROI 怎么改都不许动它。

    这一条是"RAW 全屏、ROI 只用于离线分析"在**估算口径**上的体现，
    也正好是 v1.0.2 的错处：那时候按 ROI 裁着存，估算也跟着 ROI 缩，
    于是"盘够不够"这个判断是拿一个被缩小的画面做的——现场的真实画面大 4～5 倍，
    等发现放不下的时候，RAW 已经写下去一半了。

    反面写法（"改了 ROI 估算就变小"）在这里被明确断言为**假**。
    """
    config = AppConfig()
    config.mode = "hardware"
    config.formal.step_deg = {joint: 0.2 for joint in config.pretest.joints}
    config.validate()
    baseline = group_peak_gb(config, _group_plan_map(config))[0]

    for roi in ([0, 0, 800, 600], [0, 0, 950, 800], [0, 0, 1936, 1096]):
        config.camera.analysis_roi = list(roi)
        config.validate()
        again, _worst, _per = group_peak_gb(config, _group_plan_map(config))
        assert abs(again - baseline) < 1e-9, (
            f"把 analysis_roi 改成 {roi} 之后磁盘估算从 {baseline:.4f} GB "
            f"变成了 {again:.4f} GB——估算跟着分析窗口缩了，"
            "而现场真实要写的是全屏 RAW，这会把磁盘闸门架在一次误判上"
        )

    # 旧字段 camera.roi 也只是 analysis_roi 的兼容写法，同样不许影响估算。
    config.camera.analysis_roi = None
    config.camera.roi = [0, 0, 800, 600]
    config.validate()
    legacy, _worst, _per = group_peak_gb(config, _group_plan_map(config))
    assert abs(legacy - baseline) < 1e-9, (
        "旧字段 camera.roi 又把磁盘估算拉小了：v1.0.2 的裁剪行为被带回来了"
    )


# --------------------------------------------------------------------------
# 4) 超估算：拒绝开始，而且一个命令都不许发
# --------------------------------------------------------------------------


def test_a_group_over_the_estimate_is_refused_before_any_motion(
    tmp_path: Path, session_factory
) -> None:
    """红线式断言：本组估算超过可用空间时，这一组**一个运动命令都没发**。"""
    session, recorder, cfg = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    assert session.robot is not None
    cfg.paths.min_free_disk_gb = 1e9
    with pytest.raises(ExperimentError) as info:
        session.run_pretest()
    text = str(info.value)
    assert "磁盘" in text, text
    assert "按计划需要约" in text, text
    # 运动命令一条都没发：机器人层的"动过哪几段"必须是空的。
    assert session.robot.moved_event_ids() == [], (
        f"磁盘闸门拒绝之前已经把运动命令发出去了：{session.robot.moved_event_ids()}"
    )


def test_the_group_gate_reports_the_group_not_the_whole_run(tmp_path: Path) -> None:
    """闸门报的是"这一组"的占用，不是整场总和——否则现场会被误拦。

    整场 200～350 GB、单组 20～34 GB，把总和报给闸门就等于永远不放行。
    """
    from sj_pretest.experiment import capture_segment_count

    config = build_config(
        tmp_path,
        joints=("J1", "J6"),
        amplitudes=(0.2,),
        repeats=1,
        paths__delete_raw_after_process=True,
    )
    groups = _group_plan_map(config)
    whole = sum(
        group_peak_gb(config, {name: segments})[0] for name, segments in groups.items()
    )
    peak_gb, worst, _per = group_peak_gb(config, groups)
    assert peak_gb < whole, (
        f"单组峰值 {peak_gb:.2f} GB 不小于整场总和 {whole:.2f} GB：分组没有起作用"
    )
    ok, lines = check_group_disk(config, groups[worst], what=worst)
    text = "\n".join(lines)

    # ★ 报的是**真正会采的那几段**，不是计划里所有条目（需求一·5：纯等待步骤
    # 只要不写 RAW，就不算进录制时间）。组 B 的一组里有一步是"停下来等稳定"，
    # 它不写 RAW，所以"动作数量"要比 plan 条目少一个。
    captures = capture_segment_count(groups[worst])
    assert 0 < captures < len(groups[worst]), (
        f"这一组有 {len(groups[worst])} 个计划条目、{captures} 段会采——"
        "这条断言本来就要求两者不等，好证明等待步骤被排除了"
    )
    assert f"动作数量：{captures} 段采集" in text, text
    assert f"{len(groups[worst])} 段采集" not in text, (
        "把不写 RAW 的等待步骤也算成了动作"
    )
