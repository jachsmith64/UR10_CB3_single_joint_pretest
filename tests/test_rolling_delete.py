"""边采边清：处理完就删 RAW，但**派生数据必须真的能顶替它**。

现场要求：一趟完整实验的原始帧是几百 GB，D 盘会被塞满；希望"每个动作结束就停下来
把这一段处理掉，然后删掉视频"，盘上任何时刻只有当前几十 GB。

这个功能会**删数据**，所以自测不能只测"删了没有"，必须测三条：

1. **删之前真的处理过**：删掉的段必须已经落了逐帧结果（``segment_vision.json``）
   和角点 CSV，事件流里要留下 sha256、字节数、处理步长——事后可追。
2. **删掉的只是 RAW**：时间戳、缺帧清单、采集摘要、样本图、元数据**一个都不能少**
   （它们是"这一段当时是怎么采的"的证据，而且很小）。
3. **删完之后还能复算**：``analyze_offline`` 必须能只靠留下的东西跑完，
   并且算出的结果和"RAW 还在时"一致——否则"边采边清"就是拿数据换磁盘。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.recorder import safe_name
from sj_pretest.vision import load_segment_vision

from conftest import build_config, open_session, read_json


def _run(tmp_path: Path, *, delete_raw: bool):
    config = build_config(
        tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    config.paths.delete_raw_after_process = bool(delete_raw)
    config.validate()
    session, _recorder = open_session(config, run_kind="rolling")
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
        report = session.analyze_offline(stride=4)
        return session, config, report
    finally:
        session.close()


def _deleted_events(session) -> list[dict]:
    assert session.run is not None
    return [
        item
        for item in session.run.events.read_all()
        if item.get("event") == "raw_deleted"
    ]


def test_raw_is_deleted_only_after_the_segment_was_processed(tmp_path: Path) -> None:
    """删掉的每一段，都必须先有逐帧结果落盘、回读通过，才允许删。"""
    session, _config, _report = _run(tmp_path, delete_raw=True)
    assert session.run is not None
    deleted = _deleted_events(session)
    assert deleted, "开了边采边清，却一个 raw_deleted 事件都没有"

    for event in deleted:
        segment_id = str(event["segment_id"])
        segment_dir = session.run.segment_dir(segment_id)
        assert not (segment_dir / "frames.raw").is_file(), (
            f"{segment_id} 写了 raw_deleted，RAW 却还在"
        )
        # 1) 逐帧结果在，而且能读得回来（这就是"删之前校验过"的物证）。
        vision = load_segment_vision(segment_dir, segment_id=segment_id)
        assert vision.frame_count > 0
        assert vision.valid_count > 0
        # 2) 角点 CSV 在（默认落在 run/vision/corners/ 下，文件名走 safe_name）。
        corners = session.run.corners_dir / f"{safe_name(segment_id)}.csv"
        assert corners.is_file() and corners.stat().st_size > 0
        # 3) 事件里记得住"删的是什么"：大小、哈希、步长、帧数。
        assert int(event["raw_bytes"]) > 0
        assert len(str(event["raw_sha256"])) == 64, "没记 sha256，事后没法对上账"
        assert int(event["stride"]) == int(session.config.paths.process_stride)
        assert event["processed"] is True


def test_only_the_raw_file_is_removed(tmp_path: Path) -> None:
    """只删 RAW：其余小文件一个都不能少。"""
    session, _config, _report = _run(tmp_path, delete_raw=True)
    assert session.run is not None
    for event in _deleted_events(session):
        segment_dir = session.run.segment_dir(str(event["segment_id"]))
        for name in (
            "capture_metadata.json",
            "frame_timestamps.csv",
            "missing_frames.csv",
            "capture_summary.txt",
        ):
            assert (segment_dir / name).is_file(), f"{event['segment_id']} 少了 {name}"
        assert list(segment_dir.glob("sample_*.png")), (
            f"{event['segment_id']} 的样本图被删掉了——它只有几十 KB"
        )
        # 也不许留下写盘中断的 .tmp。
        assert not list(segment_dir.glob("frames.raw*"))


def _signature(report) -> list[tuple]:
    """一份报告里"该逐位对上"的那些量。"""
    return sorted(
        (
            item.joint,
            item.stage,
            round(float(item.amplitude_deg), 6),
            int(item.direction),
            # 视觉量：位移、信噪比、轴向/面内比值——这些数全靠这段的逐帧结果，
            # RAW 删掉之后必须一模一样地重算出来。
            None if item.snr is None else round(float(item.snr), 6),
            None if item.vision_proj_px is None else round(float(item.vision_proj_px), 6),
            None
            if item.depth_ratio_upper is None
            else round(float(item.depth_ratio_upper), 6),
            item.depth_confidence,
        )
        for item in report.trials
    )


def test_offline_reanalysis_still_works_without_the_raw(tmp_path: Path) -> None:
    """删完之后必须还能复算，而且结果和"RAW 还在"时**逐位**一致。

    这是整个功能的前提：RAW 换来的就是"事后还能算"，换不到就不该删。

    ★ 为什么在**同一个会话**里先算一遍、把 RAW 删掉再算一遍，而不是跑两个会话
    （一个删一个不删）去比：合成世界的逐帧噪声是按**帧号**播种的，而 v1.0.3 在
    连接之后会先做一次 5 s 全屏采集检查、每组处理完还要丢几帧重建基线——
    两个会话的帧号序列因此不同，"同一个实验"其实落在了两份不同的噪声实现上。
    拿它们互相比，比出来的差异是噪声实现不同，不是"RAW 删了以后算不出来"。
    同一个会话里两遍复算的**输入**完全一样，差异只可能来自"用不用 RAW"——
    这才是这一条要证的事，而且比原来的比法更严（要求逐位相同，不是统计意义上相近）。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.paths.delete_raw_after_process = False
    config.validate()
    session, recorder = open_session(config, run_kind="rolling")
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
        assert session.run is not None
        # 第一遍：RAW 还在，逐帧几何是从**像素**里现算出来的。
        kept_report = session.analyze_offline(stride=1)
        assert "离线识别" in recorder.text()
        # 模拟"边采边清已经走完"：只删 RAW，别的一个字节都不动。
        raw_files = sorted(session.run.segments_dir.glob("*/frames.raw"))
        assert raw_files, "这一段流程本该写出过 RAW"
        for raw in raw_files:
            raw.unlink()
        recorder.logs.clear()
        # 第二遍：输入完全一样，只是 RAW 没了——必须能算出逐位相同的结论。
        deleted_report = session.analyze_offline(stride=1)
        text = recorder.text()
    finally:
        session.close()

    assert "原始帧已按边采边清删除" in text, (
        "第二遍分析没有走「用段内已保存的逐帧结果复算」这条路——"
        f"那它比的就还是像素，不是留下的派生数据。\n{text}"
    )
    assert _signature(kept_report) == _signature(deleted_report), (
        "只靠留下的逐帧结果复算出来的结论，和 RAW 还在时不一致"
    )
    assert [item.to_dict() for item in kept_report.recommendations] == [
        item.to_dict() for item in deleted_report.recommendations
    ], "步长推荐不一样，说明复算路径和原路径口径不同"
    assert kept_report.warnings == deleted_report.warnings, (
        "两次分析的告警不一样，说明有一条路径缺数据"
    )



def test_the_deleted_raw_is_the_one_that_was_hashed(tmp_path: Path) -> None:
    """事件里记的字节数 = 帧数 × 裁剪后每帧字节数（说明删的就是那一段 RAW）。"""
    session, _config, _report = _run(tmp_path, delete_raw=True)
    assert session.run is not None
    for event in _deleted_events(session):
        segment_dir = session.run.segment_dir(str(event["segment_id"]))
        metadata = read_json(segment_dir / "capture_metadata.json")
        expected = int(metadata["frame_count"]) * int(metadata["frame_bytes"])
        assert int(event["raw_bytes"]) == expected, (
            f"{event['segment_id']}：删掉的字节数和元数据对不上"
        )


def test_the_vision_json_does_not_lie_about_the_raw(tmp_path: Path) -> None:
    """``segment_vision.json`` 里的 ``raw_available`` 必须与盘上事实一致。

    这份文件同时由**两条**路径写出来：边采边清（写完就删 RAW）和离线分析 /
    回放（RAW 还在）。要是把 ``raw_available`` 写死成 false、注解写死成
    "RAW 已删除"，那离线分析留下的 JSON 里就躺着一句假话——将来有人拿它
    判断"这段还能不能换检测参数重新识别像素"，会得出反的结论。
    """
    kept_session, _config, _report = _run(tmp_path / "kept", delete_raw=False)
    deleted_session, _config2, _report2 = _run(tmp_path / "deleted", delete_raw=True)

    def flag(session, segment_id: str) -> bool:
        assert session.run is not None
        payload = read_json(
            session.run.segment_dir(segment_id) / "segment_vision.json"
        )
        return bool(payload["raw_available"])

    for session in (kept_session, deleted_session):
        assert session.run is not None
        for segment_dir in session.run.segments_dir.iterdir():
            if not (segment_dir / "segment_vision.json").is_file():
                continue
            has_raw = (segment_dir / "frames.raw").is_file()
            assert flag(session, segment_dir.name) is has_raw, (
                f"{segment_dir.name}：JSON 说 raw_available="
                f"{flag(session, segment_dir.name)}，盘上 frames.raw "
                f"{'在' if has_raw else '不在'}"
            )
            note = str(
                read_json(segment_dir / "segment_vision.json")["note"]
            )
            assert ("RAW 已删除" in note) is (not has_raw), (
                f"{segment_dir.name}：注解和盘上事实对不上——{note}"
            )


def test_nothing_is_deleted_while_the_switch_is_off(tmp_path: Path) -> None:
    """默认（开关关着）一个字节都不许删，而且要用中文说清为什么保留。

    用户要的是"处理完就删"，但这是**默认关**的开关：默认路径必须还是
    "证据全留"，否则任何一次误操作都会让实验数据不可回放。
    """
    session, config, _report = _run(tmp_path, delete_raw=False)
    assert config.paths.delete_raw_after_process is False
    assert session.run is not None
    assert _deleted_events(session) == []
    for segment_dir in session.run.segments_dir.iterdir():
        assert (segment_dir / "frames.raw").is_file(), (
            f"{segment_dir.name} 的 RAW 在开关关着的时候被删了"
        )


def test_the_quick_probe_check_runs_after_the_raw_is_gone(tmp_path: Path, monkeypatch) -> None:
    """★ 删了 RAW 之后，完整跑一遍快速几何检查必须照常出结论。

    这是 v1.0.1 的一个真实缺陷：``_measure_probe`` 无条件调 ``process_segment``
    去读 ``frames.raw``。边采边清把这个文件删掉之后，同一个会话里紧接着跑
    快速几何检查就会直接报"找不到 frames.raw"——**同一个动作组刚采完就检查不了**。
    现在它必须：整组采完 → 处理 → 删 RAW → **再**量，而且量的时候一个字节都
    不从 RAW 读。

    断言分两半：

    1. 结论照常：正负两个方向都量到了、J1 和 J6 都出现在结论里；
    2. 真的没重读 RAW：整趟 ``run_quick_probes`` 里 ``process_segment`` 只被调
       每段一次（都是分组处理那一次）。要是 ``_measure_probe`` 又去读一遍，
       计数会翻倍。
    """
    from sj_pretest import experiment as experiment_module

    config = build_config(
        tmp_path, joints=("J1", "J6"), amplitudes=(0.2,), repeats=1
    )
    config.paths.delete_raw_after_process = True
    config.validate()
    session, _recorder = open_session(config, run_kind="quick_after_delete")

    calls: list[str] = []
    original = experiment_module.process_segment

    def counting_process(segment_dir, **kwargs):
        calls.append(str(kwargs.get("segment_id") or segment_dir))
        return original(segment_dir, **kwargs)

    monkeypatch.setattr(experiment_module, "process_segment", counting_process)
    try:
        session.run_static()
        calls.clear()
        result, failed = session.run_quick_probes()
        text = "\n".join(result.lines)
    finally:
        monkeypatch.undo()
        session.close()

    assert session.run is not None
    quick_segments = sorted(
        path for path in session.run.segments_dir.glob("quick_probe-*") if path.is_dir()
    )
    assert quick_segments, "一段快速探针都没采到"
    for path in quick_segments:
        assert not (path / "frames.raw").is_file(), f"{path.name}：RAW 还在"
    assert len(calls) == len(quick_segments), (
        f"process_segment 被调了 {len(calls)} 次，而快速探针只有 "
        f"{len(quick_segments)} 段——多出来的就是在 RAW 删掉之后又回头去读了一遍"
    )
    assert "图像二维转角" in text and "J6" in text, text
    assert failed == [], f"删了 RAW 之后快速几何检查没通过：{failed}\n{text}"


def test_the_switch_makes_the_gate_use_the_single_segment_peak() -> None:
    """开了边采边清，磁盘门槛按**单段峰值**算，而不是整场总和。

    否则现场会出现"盘明明够大却拒绝开始"——总和是几百 GB，而实际同一时刻
    只有一段。这条直接盯住 ``check_plan_disk`` 用的是哪个数。
    """
    from sj_pretest.config import AppConfig, check_plan_disk, plan_disk_gb
    from sj_pretest.joint_space import planned_plans

    config = AppConfig()
    plans = planned_plans(config)  # 静态 + 快速确认 + 预实验 + 组A + 组B
    total, peak, _seconds = plan_disk_gb(config, plans)
    assert total > 0 and peak > 0
    assert peak < total, "单段峰值不该大于整场总和"

    config.paths.delete_raw_after_process = False
    _ok_off, lines_off = check_plan_disk(config, plans)
    config.paths.delete_raw_after_process = True
    _ok_on, lines_on = check_plan_disk(config, plans)

    def need(lines) -> float:
        text = next(line for line in lines if "按计划需要约" in line)
        return float(text.split("按计划需要约 ")[1].split(" GB")[0])

    assert need(lines_on) < need(lines_off), (
        f"打开边采边清之后门槛没降下来：{need(lines_on)} vs {need(lines_off)}"
    )
    assert "最大的一段" in lines_on[0] or any("最大的一段" in line for line in lines_on)
    # 而且**不是**降到 0：正在录的那一段仍然要一次写完。
    assert need(lines_on) >= float(config.paths.min_free_disk_gb)
