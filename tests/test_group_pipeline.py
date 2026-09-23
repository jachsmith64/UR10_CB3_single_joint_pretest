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
3. **峰值落在预警线以内**：按交付默认参数（800×600 ROI 与 950×800 ROI）
   算出每一组的估算峰值，都要 ≤ 预警线，而全幅 1936×1096 要被硬拦；
4. **超估算就不得开始下一组**：闸门拒绝时**一个运动命令都不能发出去**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import (
    AppConfig,
    check_group_disk,
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


@pytest.mark.parametrize(
    "roi, expected_max_gb",
    [((0, 0, 800, 600), 22.6), ((0, 0, 950, 800), 35.8)],
)
def test_the_delivered_rois_stay_under_the_warning_line(
    roi: tuple[int, int, int, int], expected_max_gb: float
) -> None:
    """800×600 与 950×800 两种 ROI 下，**每一组**的估算峰值都要在预警线以内。

    数字是 1.0.2 定稿时按**真机口径**（``mode="hardware"``、132.23 fps、
    正式实验 Δ=0.2°、阶梯 5 级、重复 3 遍）算出来的实测值：
    800×600 最坏一组约 22.4 GB，950×800 约 35.5 GB，两档都在 40 GB 预警线以内。
    这里写成"实测值 + 一点余量"的上限，同时**必须**低于 ``disk_warn_gb``——
    否则交付的默认参数自己就会一路报警，报警变成噪声就没人看了。

    注意 ``mode`` 必须是 hardware：干运行把画面缩到 340×260，算出来的占用
    和现场不是一个量级，拿它验"现场放不放得下"没有意义。
    """
    config = AppConfig()
    config.mode = "hardware"
    config.camera.roi = list(roi)
    config.formal.step_deg = {joint: 0.2 for joint in config.pretest.joints}
    config.validate()

    peak_gb, worst, per_group = group_peak_gb(config, _group_plan_map(config))
    assert per_group, "一组都没算出来"
    assert peak_gb <= expected_max_gb, (
        f"{roi[2]}×{roi[3]}：最坏一组是「{worst}」{peak_gb:.2f} GB，"
        f"超过定稿实测上限 {expected_max_gb} GB"
    )
    assert peak_gb <= float(config.paths.disk_warn_gb), (
        f"{roi[2]}×{roi[3]}：最坏一组 {peak_gb:.2f} GB 已经越过预警线 "
        f"{config.paths.disk_warn_gb:.0f} GB——交付默认参数会一路报警"
    )
    assert peak_gb <= float(config.paths.max_peak_disk_gb)


def test_full_frame_is_refused_by_the_hard_cap() -> None:
    """不裁 ROI 的全幅画面（1936×1096）必须被硬上限拦下来。

    这一条是"分组流水线真的在按峰值把关"的反向证据：同一套代码、
    同一套计划，只是画面从 950×800 变成满幅，就必须从"放行"变成"拒绝开始"。
    满幅一组约 99 GB（整场约 1 TB），正是需求一说的"不得开始下一组"。

    硬上限只在分组流水线打开时才算（开关关着的时候 RAW 是全程累积的，
    拿 50 GB 卡每一组等于禁用默认配置，见 ``check_group_disk`` 的说明），
    所以这里把开关打开——这也正是现场要控峰值时的用法。
    顺便：这条断言因此**不依赖跑测试那台机器的剩余磁盘**，
    超硬上限是先判的，轮不到"可用空间够不够"。
    """
    config = AppConfig()
    config.mode = "hardware"
    config.camera.roi = [0, 0, 1936, 1096]
    config.paths.delete_raw_after_process = True
    config.formal.step_deg = {joint: 0.2 for joint in config.pretest.joints}
    config.validate()
    ok, lines = check_group_disk(
        config, _group_plan_map(config)["组A J1"], what="组A J1 正式实验"
    )
    assert ok is False, "全幅画面居然被放行了：\n" + "\n".join(lines)
    assert "不得开始这一组" in "\n".join(lines)
    assert float(config.paths.max_peak_disk_gb) == 50.0, (
        "硬上限被改过了——需求一写的是 50 GB"
    )
    assert 35.0 <= float(config.paths.disk_warn_gb) <= 40.0, (
        "预警线要落在需求一建议的 35～40 GB 区间里"
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
    assert f"{len(groups[worst])} 段" in text, text
