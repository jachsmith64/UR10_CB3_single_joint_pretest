"""★ 需求一·4：正式实验的组边界就是"每一遍 repeat"，不许把三遍塞成一组。

需求原文：正式组 A **每关节的每一遍 repeat 单独一组**；正式组 B 同样。
"不能把 5 循环 × 2 方向 × 3 遍补成一组"。

同一个文件里也管三档预实验的分组（需求一·4 的另一半）：默认**每个关节一组**；
一个关节的一组自己放不下时，在"已经回到名义姿态"的边界上再拆（预实验每次动作
都从名义姿态独立出发再回来），而不是直接拒绝——只有连单次动作都放不下才拒绝。

为什么这条要单独测
------------------
分组的错法与后果都很具体：

* **切得太粗**（三遍一组）：一遍 repeat 的 RAW 就够大，三遍叠起来直接撞硬上限；
  而且"停下来处理"的位置落在了**没有回到名义姿态**的地方——处理期间机械臂
  停在一个偏离名义姿态的位置上，人要靠过去换盘、查线，风险凭空变大。
* **切得太细**（每个动作一组）：一趟实验要停下来处理几百次，现场没法用。
* **切错位置**（在非名义姿态处切）：同上，安全边界被破坏。
* **偷偷少走几步**：为了把这一组塞进磁盘上限而减少动作数量/重复次数/帧率——
  需求一·4 **明文禁止**，这里也要证明它没有发生（切完的步骤数、顺序、
  身份都与原计划完全一致）。

所以这个文件既测"切够了"，也测"没切过头、没切错地方、没切掉东西"。

关于磁盘数值
------------
用真机口径（``mode="hardware"``、1936×1096 @ 132.23 fps、Δ=0.2°、5 级、3 遍）：
组 A 一遍约 16.7 GB、组 B 一遍约 33.5 GB，都低于 50 GB 硬上限，所以
**交付默认下不需要再拆**——这一条由 ``test_group_pipeline`` 盯住。
这里要测的是"某一遍自己就超上限"时会发生什么，所以把硬上限调低到一个
真机口径下必然超的值（相当于换一台更小的盘）。判据（50 GB 那条规则本身）
一个字都没改：变的只是这台机器的容量。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import (
    AppConfig,
    MeasuredCapture,
    group_forecast,
)
from sj_pretest.experiment import (
    ExperimentError,
    iter_segment_plans,
    nominal_split_segment_groups,
    plan_formal_groups,
    repeat_segment_groups,
)
from sj_pretest.joint_space import build_formal_group_a, build_formal_group_b

#: 交付默认的正式实验规模（与需求一·4 的文字一致：5 级阶梯、3 遍）。
STAIRCASE_N = 5
REPEATS = 3
STEP_DEG = 0.2
#: 真机口径的满幅采集实测值（现场 5 s 全屏采集检查会量到这一组数）。
MEASURED = MeasuredCapture(
    width=1936,
    height=1096,
    fps=132.23,
    frames=661,
    seconds=5.0,
    source="自测：按真机口径写死的实测值",
)
#: 偏小的盘：真机口径下组 A 一遍(≈17 GB)、组 B 一遍(≈34 GB) 都超。
TIGHT_CAP_GB = 8.0


def _config(*, cap_gb: float = 50.0) -> AppConfig:
    config = AppConfig()
    config.mode = "hardware"
    config.formal.staircase_n = STAIRCASE_N
    config.formal.repeats = REPEATS
    config.formal.step_deg = {joint: STEP_DEG for joint in config.pretest.joints}
    config.paths.max_peak_disk_gb = float(cap_gb)
    # 预警线必须画在硬上限**里面**（配置层会校验），所以跟着一起降。
    config.paths.disk_warn_gb = min(40.0, float(cap_gb) * 0.75)
    config.validate()
    return config


def _plans(config: AppConfig, group: str, joint: str = "J1") -> list:
    builder = build_formal_group_a if group.upper() == "A" else build_formal_group_b
    return iter_segment_plans(
        builder(config, config.robot.nominal_joint_deg, joint, STEP_DEG)
    )


def _flat(groups) -> list:
    return [segment for item in groups for segment in item.segments]


def _repeat_index(segment) -> int:
    return int(segment.primary.event.repeat_index or 1)


def _ends_at_nominal(segment, config: AppConfig) -> bool:
    target = segment.end_target
    if target is None:
        return False
    nominal = config.robot.nominal_joint_deg
    return all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(target, nominal))


@pytest.fixture(autouse=True)
def no_real_disk_probe(monkeypatch):
    """把"真实可用空间"探测换成固定答复。

    ★ 只换**探测**，不换判据：本文件要看的是"分组策略"（切在哪、切多细、
    有没有丢步骤），而"这块盘现在剩多少 GB"取决于跑测试的机器，和分组逻辑无关。
    硬上限那一条（``resident > cap``）照旧生效——它才是分组策略的输入。
    真实空间/断盘的处理由 test_safety、test_group_pipeline 里的磁盘用例覆盖。
    """
    from sj_pretest import config as config_module

    monkeypatch.setattr(
        config_module,
        "check_free_disk",
        lambda path, min_free_gb: (True, "自测：跳过真实磁盘空间探测（只看分组策略）"),
    )


# --------------------------------------------------------------------------
# 1) 默认：每一遍 repeat 单独一组
# --------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["A", "B"])
def test_each_repeat_is_its_own_group(group: str) -> None:
    """组 A / 组 B 都按**每一遍 repeat** 切：3 遍就是 3 组，一组不多一组不少。"""
    config = _config()
    segments = _plans(config, group)
    groups = plan_formal_groups(
        config, segments, config.robot.nominal_joint_deg, group, "J1", measured=MEASURED
    )

    assert len(groups) == REPEATS, (
        f"组{group} 切成了 {len(groups)} 组，而 repeats={REPEATS}——"
        "需求一·4 要的是每一遍 repeat 单独一组"
    )
    assert [item.level for item in groups] == ["repeat"] * REPEATS, (
        f"组{group} 用了更细的切法（{ [item.level for item in groups] }），"
        "而交付参数下应当是每遍一组"
    )
    for position, item in enumerate(groups, start=1):
        indices = {_repeat_index(segment) for segment in item.segments}
        assert indices == {position}, (
            f"{item.name}：包含的 repeat 编号是 {indices}，应当只有第 {position} 遍"
        )
        assert f"第{position}遍" in item.name, (
            f"组名没有写清是第几遍：{item.name}"
        )
    # 一段都没多、没少、没换顺序（切分只能决定"在哪儿停"）。
    assert _flat(groups) == segments, "切完之后步骤数/顺序/身份与原计划不一致"


def test_no_group_mixes_cycles_directions_and_repeats() -> None:
    """需求一·4 点名禁止的那件事：5 循环 × 2 方向 × 3 遍不许塞成一组。"""
    config = _config()
    segments = _plans(config, "B")
    groups = plan_formal_groups(
        config, segments, config.robot.nominal_joint_deg, "B", "J1", measured=MEASURED
    )
    for item in groups:
        repeats = {_repeat_index(segment) for segment in item.segments}
        assert len(repeats) == 1, (
            f"{item.name}：把 {sorted(repeats)} 遍混进了一组"
        )
        # 每一组里的运动段数应当是"一遍"的量级（5 循环 × 2 方向 = 10 段），
        # 而不是三遍叠起来的 30 段。
        moves = [segment for segment in item.segments if not segment.is_wait_only]
        assert len(moves) <= 2 * STAIRCASE_N, (
            f"{item.name}：一组里有 {len(moves)} 段运动，超过一遍 repeat 的 "
            f"{2 * STAIRCASE_N} 段（{STAIRCASE_N} 个循环 × 正负两个方向）"
        )


@pytest.mark.parametrize("group", ["A", "B"])
def test_every_group_boundary_is_a_return_to_the_nominal_pose(group: str) -> None:
    """切点必须落在"已经回到名义姿态"的地方——那是**唯一**允许停下来处理的位置。

    同时检查下一组的第一段是**从名义姿态**继续走的（这一步最多走一个 Δ）：
    如果切点落在别处，下一组的第一段就会从某个偏离名义的姿态出发，
    运动量会突然变成好几倍——那是把"相邻目标差"算错的典型症状。
    """
    config = _config()
    nominal = tuple(float(v) for v in config.robot.nominal_joint_deg)
    segments = _plans(config, group)
    groups = plan_formal_groups(
        config, segments, config.robot.nominal_joint_deg, group, "J1", measured=MEASURED
    )
    assert len(groups) > 1, "只有一组，没有切点可查"
    for item in groups[:-1]:
        last_motion = [s for s in item.segments if not s.is_wait_only]
        assert last_motion, f"{item.name}：一组里一个运动段都没有"
        assert _ends_at_nominal(last_motion[-1], config), (
            f"{item.name} 的最后一段走完没有停在名义姿态，"
            f"而是停在 {last_motion[-1].end_target}——这是在不安全的位置停下来处理"
        )
        # 纯等待段只能跟在它前面那一组（它录的是"停下来这段时间"）。
        assert not item.segments[-1].is_wait_only or _ends_at_nominal(
            last_motion[-1], config
        )
    for item in groups[1:]:
        first = item.segments[0].primary
        assert first.target_joint_deg is not None
        step = abs(
            float(first.target_joint_deg[0]) - float(nominal[0])
        )
        assert step <= STEP_DEG + 1e-9, (
            f"{item.name} 的第一段从名义姿态走了 {step:.4f}°，超过一个 Δ="
            f"{STEP_DEG}°——说明上一组的切点不在名义姿态上"
        )


# --------------------------------------------------------------------------
# 2) 某一遍自己就超上限：只在安全边界上再拆，拆不动就拒绝开始
# --------------------------------------------------------------------------


def test_group_b_is_split_further_only_at_the_nominal_boundaries() -> None:
    """组 B 的一遍自带若干个"回到名义姿态"的点，所以超上限时能在那里再拆。

    断言四件事：每组的驻留估算都压到上限以内；组数比"每遍一组"更多；
    每一组的最后一段仍然停在名义姿态上（切点没跑偏）；步骤一段不差。
    """
    config = _config(cap_gb=TIGHT_CAP_GB)
    segments = _plans(config, "B")
    groups = plan_formal_groups(
        config, segments, config.robot.nominal_joint_deg, "B", "J1", measured=MEASURED
    )

    assert len(groups) > REPEATS, (
        f"上限压到 {TIGHT_CAP_GB:.0f} GB 之后组 B 还是 {len(groups)} 组——没有再拆"
    )
    assert {item.level for item in groups} <= {"repeat", "nominal"}
    assert "nominal" in {item.level for item in groups}, (
        "既然后拆了，就应该标成在「已回到名义姿态处」再拆的那一级"
    )
    for item in groups:
        # ★ 比的就是闸门真正比的那个数（含处理期临时余量的驻留峰值），
        # 不是只含 RAW+派生的中间量：两者差 15%，用错量会把"还是超了"的组
        # 看成"已经压下去了"。
        peak = group_forecast(config, item.segments, measured=MEASURED).peak_gb
        assert peak <= TIGHT_CAP_GB, (
            f"{item.name}：驻留峰值估算 {peak:.2f} GB 仍然超过上限 "
            f"{TIGHT_CAP_GB:.0f} GB"
        )
        assert len({_repeat_index(segment) for segment in item.segments}) == 1, (
            f"{item.name}：再拆之后跨了遍"
        )
    # 中间那些组的最后一段必须停在名义姿态（末尾那一组到一遍结束自然也在名义上）。
    for item in groups:
        motion = [s for s in item.segments if not s.is_wait_only]
        assert _ends_at_nominal(motion[-1], config), (
            f"{item.name} 停在 {motion[-1].end_target}，不是名义姿态——"
            "再拆的切点跑偏了"
        )
    assert _flat(groups) == segments, "再拆之后步骤数/顺序/身份与原计划不一致"


def test_group_a_cannot_be_split_further_so_it_refuses_before_anything_starts() -> None:
    """组 A 的一遍中途没有回名义的点，所以拆不动——那就**拒绝开始并说明原因**。

    这是需求一·4 的最后一道兜底："不能为了满足磁盘限制偷偷减少动作数量、
    重复次数或采集帧率；如果不存在安全的拆分方式，就在运动开始之前拒绝。"

    所以这里要看到：抛的是 ``ExperimentError``（不是"跑一半才报错"）、
    说清了是哪一组、报出了上限数值、点明了"再拆只能拆在没有回到名义姿态的地方"、
    并且明确写着不许减少动作/重复/帧率。整个过程**没有会话、没有机器人**——
    这些函数只做计划，连设备都不碰。
    """
    config = _config(cap_gb=TIGHT_CAP_GB)
    segments = _plans(config, "A")
    # 先确认这条路确实被走到：逐遍分组会有超上限的组。
    default_over = [
        item
        for item in repeat_segment_groups(segments)
        if group_forecast(config, item[1], measured=MEASURED).peak_gb > TIGHT_CAP_GB
    ]
    assert default_over, (
        "把上限压到 8 GB 之后组 A 的每一遍都没超——这条测试的前提不成立"
    )
    with pytest.raises(ExperimentError) as info:
        plan_formal_groups(
            config,
            segments,
            config.robot.nominal_joint_deg,
            "A",
            "J1",
            measured=MEASURED,
        )
    text = str(info.value)
    assert "组A J1" in text, text
    assert "没有开始" in text, text
    assert f"{TIGHT_CAP_GB:.0f} GB" in text, text
    assert "不能再往下拆" in text or "不能再拆" in text, text
    assert "名义姿态" in text, text
    for banned in ("动作数量", "重复次数", "帧率"):
        assert banned in text, f"没有点明不许减少{banned}：{text}"
    # 也顺便看一眼：组 A 的"更细一级"确实等于按遍分组（中途没有安全切点）。
    finer = nominal_split_segment_groups(
        config.robot.nominal_joint_deg, segments
    )
    assert len(finer) == len(repeat_segment_groups(segments)), (
        "组 A 竟然能在中途再切——那它就不是「拆不动」的那一组了，"
        "这条测试的结论要重写"
    )


# --------------------------------------------------------------------------
# 3) 切分本身的性质：是划分，不是过滤
# --------------------------------------------------------------------------


def test_the_two_splitters_are_partitions() -> None:
    """两级切分都只是"在哪儿停"：段数、顺序、身份一个都不许变。

    ``repeat_segment_groups`` / ``nominal_split_segment_groups`` 内部都有
    ``_assert_partition`` 把关；这里再从外面核对一遍，免得以后有人在这里
    顺手加个 filter（那正是"偷偷少走几步"）。
    """
    config = _config()
    for group in ("A", "B"):
        segments = _plans(config, group)
        by_repeat = [items for _index, items in repeat_segment_groups(segments)]
        assert [segment for items in by_repeat for segment in items] == segments
        by_nominal = nominal_split_segment_groups(
            config.robot.nominal_joint_deg, segments
        )
        assert [segment for items in by_nominal for segment in items] == segments
        assert sum(len(items) for items in by_nominal) == len(segments)
        # 纯等待段不单独成组，也不被丢掉。
        waits = [s for s in segments if s.is_wait_only]
        assert waits, f"组{group} 里没有等待段，这条性质就测不到了"
        for wait in waits:
            assert any(wait in items for items in by_nominal), (
                "等待段在按名义姿态分组时被丢掉了——它录的是「停下来这段时间」，"
                "不能丢"
            )


def test_repeat_groups_follow_the_repeat_index_in_order() -> None:
    """按遍分组必须**顺序**输出、且每遍只出现一次（乱序会让事件流对不上账）。"""
    config = _config()
    segments = _plans(config, "A")
    indices = [index for index, _items in repeat_segment_groups(segments)]
    assert indices == list(range(1, REPEATS + 1)), indices


# --------------------------------------------------------------------------
# 4) 三档预实验：默认每关节一组；放不下时也在名义姿态边界上再拆
# --------------------------------------------------------------------------


def _pretest_plans(config: AppConfig) -> list:
    from sj_pretest.experiment import iter_segment_plans
    from sj_pretest.joint_space import build_pretest_plan

    return iter_segment_plans(
        build_pretest_plan(config, config.robot.nominal_joint_deg)
    )


def test_the_pretest_is_one_group_per_joint_by_default() -> None:
    """★ 需求一·4：三档预实验**每个关节一组**，交付默认尺寸下不需要再拆。"""
    from sj_pretest.experiment import plan_pretest_groups

    config = _config()
    segments = _pretest_plans(config)
    groups = plan_pretest_groups(config, segments, measured=MEASURED)

    assert len(groups) == len(config.pretest.joints), (
        f"预实验切成了 {len(groups)} 组，而有 {len(config.pretest.joints)} 个关节"
    )
    assert [item.name for item in groups] == [
        f"预实验 {joint}" for joint in config.pretest.joints
    ]
    assert {item.level for item in groups} == {"repeat"}, (
        "交付默认尺寸下预实验还要拆到「小段」这一级——说明默认参数离上限太近"
    )
    assert [s for item in groups for s in item.segments] == segments, (
        "分组把预实验的段数/顺序/身份改掉了"
    )


def test_a_pretest_group_over_the_cap_is_split_at_the_nominal_boundaries() -> None:
    """预实验一组自己放不下时：在"已回到名义姿态"的边界上再拆，而不是拒绝。

    预实验每次动作都从名义姿态独立出发、再回到名义姿态，所以这种边界到处都是
    （组 A 的阶梯没有这种点，是"拆不动就拒绝"的那一组）。
    """
    from sj_pretest.experiment import plan_pretest_groups

    config = _config(cap_gb=TIGHT_CAP_GB)
    segments = _pretest_plans(config)
    groups = plan_pretest_groups(config, segments, measured=MEASURED)

    assert len(groups) > len(config.pretest.joints), (
        f"上限压到 {TIGHT_CAP_GB:.0f} GB 之后预实验还是每组一个关节——没有拆"
    )
    assert {item.level for item in groups} == {"nominal"}, (
        "既然拆了，就该标成在「已回到名义姿态处」再拆的那一级"
    )
    for item in groups:
        peak = group_forecast(config, item.segments, measured=MEASURED).peak_gb
        assert peak <= TIGHT_CAP_GB, f"{item.name}：驻留峰值 {peak:.2f} GB 仍然超上限"
        motion = [s for s in item.segments if not s.is_wait_only]
        assert motion and _ends_at_nominal(motion[-1], config), (
            f"{item.name} 停在 {motion[-1].end_target}，不是名义姿态"
        )
    assert [s for item in groups for s in item.segments] == segments, (
        "再拆之后预实验的段数/顺序/身份与原计划不一致"
    )


def test_a_pretest_action_that_does_not_fit_refuses_before_anything_starts() -> None:
    """连**单次动作**都放不下时：在发任何运动命令之前拒绝，并说清为什么不能再拆。"""
    from sj_pretest.experiment import plan_pretest_groups

    config = _config(cap_gb=0.05)
    segments = _pretest_plans(config)
    with pytest.raises(ExperimentError) as info:
        plan_pretest_groups(config, segments, measured=MEASURED)
    text = str(info.value)
    assert "预实验 J1" in text, text
    assert "没有开始" in text, text
    assert "名义姿态" in text, text
    assert "不能再往下拆" in text, text
    for banned in ("动作数量", "重复次数", "帧率"):
        assert banned in text, f"没有点明不许减少{banned}：{text}"


def test_the_pretest_session_actually_runs_the_split_groups(tmp_path: Path) -> None:
    """分组不是纸上算的：真跑一遍按钮二，事件流里的组名必须就是拆出来的那些。

    这一条防的是"``plan_pretest_groups`` 算得很漂亮，``run_pretest`` 却还在按
    关节走"。所以这里把硬上限压到"每个关节一组"的峰值之下（模拟一台更小的盘），
    然后直接看 ``group_started`` 事件和最终采到的段数。
    """
    from sj_pretest.experiment import (
        iter_segment_plans,
        plan_pretest_groups,
    )
    from sj_pretest.joint_space import build_pretest_plan

    from conftest import build_config, open_session

    config = build_config(
        tmp_path, joints=("J1", "J6"), amplitudes=(0.2,), repeats=1
    )
    config.paths.delete_raw_after_process = True
    config.validate()
    session, _recorder = open_session(config, run_kind="pretest-split")
    try:
        plans = iter_segment_plans(
            build_pretest_plan(config, config.robot.nominal_joint_deg)
        )
        # 会话已经连过设备，``session.measured`` 就是这台"机器"的实测值——
        # 用它算出"每关节一组"的峰值，再把硬上限压到六成。
        one_group = plan_pretest_groups(
            config, plans, measured=session.measured
        )
        joint_peak = max(
            group_forecast(config, item.segments, measured=session.measured).peak_gb
            for item in one_group
        )
        config.paths.max_peak_disk_gb = joint_peak * 0.6
        config.paths.disk_warn_gb = config.paths.max_peak_disk_gb * 0.5

        result = session.run_pretest()
        assert session.run is not None
        started = [
            str(item.get("group") or "")
            for item in session.run.events.read_all()
            if item.get("event") == "group_started"
        ]
        pretest_groups = [name for name in started if name.startswith("预实验")]
        assert len(pretest_groups) >= 2, (
            f"硬上限压到 {config.paths.max_peak_disk_gb:.3f} GB 之后预实验还只开了 "
            f"{len(pretest_groups)} 组：{started}"
        )
        assert any("小段" in name for name in pretest_groups), pretest_groups
        assert len(result.segments) == len(plans), (
            f"折了 {len(plans)} 段计划，只采到 {len(result.segments)} 段——"
            "分组把动作弄丢了"
        )
        assert session.aborted is False
    finally:
        session.close()
