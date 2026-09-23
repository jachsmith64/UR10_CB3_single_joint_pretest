"""★ 需求一·6：删 RAW 的动作顺序本身就是数据安全。

需求一·6 规定的六步是：

1. **RAW 还在的时候**逐帧处理完，写 ``segment_vision.json``（``raw_available=true``）；
2. 跑完删除前的六项校验；
3. 校验不过 → 保留 RAW，JSON 保持 ``true``，注解里**绝不出现**"RAW 已删除"，停下等人；
4. 校验通过 → 才真正删除 RAW；
5. 确认 RAW 真的没了；
6. 确认之后才原子改写 JSON 为 ``raw_available=false`` 并换成"已删除"的注解。

为什么顺序不能调
----------------
``raw_available`` 是给**将来的人**看的："这一段还能不能换检测参数重新识别像素"。
两种写反的方式都会造成实打实的损失：

* 一开始就写 ``false`` → 第 3 步校验失败、RAW 被留下来时，盘上出现"RAW 在、
  JSON 说不在"的矛盾，后人据此以为没法重识别了，把那一段能用的原始像素白白丢掉；
* 删之前就改写注解 → 删除失败（文件被占用）时 JSON 已经在说"已删除"，
  于是没有任何线索提示"其实还在"，而人也不会去查。

所以这个文件测两件事：

1. **顺序**：在四个关键时刻（处理刚结束 / 校验时 / 调删除时 / 改写时）各取一次
   现场快照，看 JSON 和盘上事实是否一致——快照顺序必须与上面六步一致；
2. **反向**：**故意**让删除前校验不过，然后逐条证明四件事——
   ``frames.raw`` 还在、``raw_available`` 还是 ``true``、注解里没有"RAW 已删除"、
   会话**停下来等人**（而且要证明"停下来"是真的把运动也拦住了，不只是打个日志）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.robot_joint import MotionAborted
from sj_pretest.vision import load_segment_vision

from conftest import build_config, open_session, read_json


# --------------------------------------------------------------------------
# 装置
# --------------------------------------------------------------------------


def _config(tmp_path: Path, **overrides):
    config = build_config(
        tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1, **overrides
    )
    config.paths.delete_raw_after_process = True
    config.validate()
    return config


def _release_events(session, name: str) -> list[dict]:
    assert session.run is not None
    return [
        item
        for item in session.run.events.read_all()
        if item.get("event") == name
    ]


def _vision_payload(segment_dir: Path) -> dict:
    return read_json(segment_dir / "segment_vision.json")


def test_the_release_order_is_processed_then_checked_then_deleted_then_rewritten(
    tmp_path: Path, monkeypatch
) -> None:
    """四个关键时刻的现场快照必须与需求一·6 的六步一一对上。

    快照点：

    * ``处理刚结束``：``process_segment`` 返回时——此刻**只允许**是
      "RAW 在 / JSON 说在 / 注解没说删"；
    * ``校验时``：``_verify_releasable`` 里——同上（校验是在删之前跑的）；
    * ``调删除时``：``remove_raw_artifacts`` 被调用的那一刻——RAW 还在，
      而且 JSON **仍然**说在（此刻还没到第 6 步）；
    * ``改写时``：``save_vision_json(raw_available=False)`` 被调用的那一刻——
      RAW **必须已经不在**了，否则就是"还没确认删掉就改了说法"。
    """
    from sj_pretest import experiment as experiment_module

    config = _config(tmp_path)
    session, _recorder = open_session(config, run_kind="order_probe")

    snapshots: list[tuple[str, str, bool]] = []  # (时刻, 段, JSON 里的 raw_available)
    seen_processed: set[str] = set()

    original_process = experiment_module.process_segment
    original_verify = experiment_module.ExperimentSession._verify_releasable
    original_remove = experiment_module.remove_raw_artifacts
    original_save = experiment_module.save_vision_json

    def checking_process(segment_dir, **kwargs):
        result = original_process(segment_dir, **kwargs)
        segment_dir = Path(segment_dir)
        payload = _vision_payload(segment_dir)
        raw_there = (segment_dir / "frames.raw").is_file()
        assert raw_there, f"{segment_dir.name}：处理刚结束 RAW 就没了，顺序反了"
        assert payload["raw_available"] is True, (
            f"{segment_dir.name}：处理刚结束时 JSON 已经写着 raw_available=false——"
            "校验还没跑呢。校验一旦不过，盘上就会留下「RAW 在、JSON 说不在」"
            f"的矛盾记录。实际：{payload['raw_available']}"
        )
        assert "已删除" not in str(payload.get("note") or ""), (
            f"{segment_dir.name}：处理刚结束注解就在说删除——{payload.get('note')}"
        )
        snapshots.append(("处理刚结束", segment_dir.name, True))
        seen_processed.add(segment_dir.name)
        return result

    def checking_verify(self, record, processed, *, corners_path, vision_json, **_kw):
        payload = _vision_payload(Path(vision_json).parent)
        # ★ 快照一律按**采集目录名**归类，不用 segment_id：段号里可能有
        # ``+`` 这类字符，落盘时被 safe_name 换掉（``+1`` → ``_1``），
        # 两个键混用会把同一个段拆成两条互不相认的记录，顺序链就断了。
        name = Path(record.dir).name
        assert Path(record.dir, "frames.raw").is_file(), (
            f"{name}：跑到校验时 RAW 已经不在盘上了"
        )
        assert payload["raw_available"] is True, (
            f"{name}：校验时 JSON 的 raw_available 已经是 "
            f"{payload['raw_available']}——删除前的校验必须发生在改写之前"
        )
        snapshots.append(("校验时", name, True))
        return original_verify(
            self,
            record,
            processed,
            corners_path=corners_path,
            vision_json=vision_json,
            **_kw,
        )

    def checking_remove(path, *args, **kwargs):
        path = Path(path)
        vision_json = path.parent / "segment_vision.json"
        if vision_json.is_file():
            payload = _vision_payload(path.parent)
            assert path.is_file(), (
                f"{path.parent.name}：调删除的时候 RAW 已经不在了——"
                "那这一刀是别处砍的，不是这条流水线"
            )
            assert payload["raw_available"] is True, (
                f"{path.parent.name}：调删除的时候 JSON 已经改成 "
                f"raw_available={payload['raw_available']}——"
                "还没确认删干净就先改了说法"
            )
            assert "已删除" not in str(payload.get("note") or ""), (
                f"{path.parent.name}：还没删，注解就在说已删除——{payload.get('note')}"
            )
            snapshots.append(("调删除时", path.parent.name, True))
        return original_remove(path, *args, **kwargs)

    def checking_save(path, result, **kwargs):
        path = Path(path)
        if kwargs.get("raw_available") is False:
            assert not (path.parent / "frames.raw").exists(), (
                f"{path.parent.name}：改写 raw_available=false 的时候 frames.raw "
                "还在——第 5 步（确认删掉了）被跳过了"
            )
            snapshots.append(("改写时", path.parent.name, False))
        return original_save(path, result, **kwargs)

    monkeypatch.setattr(experiment_module, "process_segment", checking_process)
    monkeypatch.setattr(
        experiment_module.ExperimentSession, "_verify_releasable", checking_verify
    )
    monkeypatch.setattr(experiment_module, "remove_raw_artifacts", checking_remove)
    monkeypatch.setattr(experiment_module, "save_vision_json", checking_save)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        monkeypatch.undo()
        session.close()

    assert session.run is not None
    deleted = _release_events(session, "raw_deleted")
    assert deleted, "开了边采边清却一段都没删，这条测试的现场快照就取不全"

    expected = {"处理刚结束", "校验时", "调删除时", "改写时"}
    per_segment: dict[str, list[str]] = {}
    for moment, segment, _flag in snapshots:
        per_segment.setdefault(segment, []).append(moment)
    for segment, moments in per_segment.items():
        if segment not in seen_processed or segment == "connection_check":
            continue
        assert expected <= set(moments), (
            f"{segment}：只走过了 {moments}，没走全 {sorted(expected)}——"
            "顺序链断了一环（多半是某一步被别处调用绕过去了）"
        )
        assert moments == ["处理刚结束", "校验时", "调删除时", "改写时"], (
            f"{segment}：快照顺序是 {moments}，与需求一·6 的"
            "「处理 → 校验 → 删除 → 改写」不一致"
        )

    # 收尾状态：RAW 真的没了，JSON 如实说 false，注解换成"已删除"。
    for event in deleted:
        segment_dir = session.run.segment_dir(str(event["segment_id"]))
        assert not (segment_dir / "frames.raw").exists()
        payload = _vision_payload(segment_dir)
        assert payload["raw_available"] is False
        assert "已删除" in str(payload.get("note") or "")


def test_a_failed_pre_deletion_check_keeps_everything_and_stops_the_session(
    tmp_path: Path,
) -> None:
    """★ 反向测试：**故意**让删除前校验不过，四条事实逐一见分晓。

    做法是把 ``thresholds.min_window_frames`` 抬到一个真实数据永远达不到的帧数，
    于是第 2 条校验（"每个声明过的阶段都要有 ≥ N 帧有效帧"）必然失败。
    这里**没有**注入假函数、也没有改判据口径——是走真实路径、真实数据、
    真实失败原因，只不过把一个上限类参数调到了不可能满足的值。

    校验失败之后必须同时成立：

    1. ``frames.raw`` **还在**（哪怕开了"分组处理并删除 RAW"）；
    2. ``segment_vision.json`` 里 ``raw_available`` 仍是 ``true``；
    3. 注解里**没有**"RAW 已删除"（否则后人会被一句假话骗走原始像素）；
    4. 会话**停下来等人**——不只是写一条日志，而是真的中止：
       再想发运动命令会被拦，而且事件流里有 ``aborted``。
    """
    config = _config(tmp_path)
    # 真实数据不可能满足的帧数：每一阶段都会判"有效帧不够"。
    config.thresholds.min_window_frames = 100000
    config.validate()
    session, _recorder = open_session(config, run_kind="reverse_check")
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        with pytest.raises(MotionAborted) as info:
            session.run_pretest()
        text = str(info.value)
        assert "校验" in text or "保留" in text, text
        # 会话真的中止了。
        assert session.aborted is True, (
            "校验失败之后会话居然还是「没中止」的状态——按需求三要停下来等人"
        )
        assert session.run is not None
        assert _release_events(session, "aborted"), "事件流里没有 aborted 这一笔"
        assert _release_events(session, "raw_deleted") == [], (
            "明明校验不过，还是有段被删了"
        )

        kept = _release_events(session, "raw_kept")
        assert kept, "校验不过的段没有写 raw_kept 事件，事后查不出下落"
        moved_before = set(session.robot.moved_event_ids()) if session.robot else set()

        # 逐段核对四条事实。
        segmented = [
            path
            for path in session.run.segments_dir.iterdir()
            if path.is_dir() and (path / "capture_metadata.json").is_file()
        ]
        assert segmented, "一段都没采到，这条测试没测到东西"
        for segment_dir in segmented:
            assert (segment_dir / "frames.raw").is_file(), (
                f"{segment_dir.name}：校验没过，RAW 却被删了——这是最严重的一类错误"
            )
            payload = _vision_payload(segment_dir)
            assert payload["raw_available"] is True, (
                f"{segment_dir.name}：RAW 还在盘上，JSON 却说 "
                f"raw_available={payload['raw_available']}——"
                "后人会据此以为没法重识别，把那一段原始像素白白丢掉"
            )
            note = str(payload.get("note") or "")
            assert "RAW 已删除" not in note and "已删除" not in note, (
                f"{segment_dir.name}：RAW 好端端在盘上，注解却写着已删除——{note}"
            )
            # 而且逐帧结果本身是可用的（校验失败只是"这一档不够严"，不是处理崩了）。
            vision = load_segment_vision(segment_dir, segment_id=segment_dir.name)
            assert vision.frame_count > 0 and vision.valid_count > 0

        # 4) 真的停下来了：再想往下跑仍然被拦，也不会凭空多出动作编号。
        #    中止是**粘住**的——不是"写一条日志然后继续"。现场要的正是这个：
        #    停下来，等人看过段目录里的相位表和帧时间戳之后再决定下一步。
        assert session.aborted
        with pytest.raises(MotionAborted):
            session.run_pretest()
        moved_after = set(session.robot.moved_event_ids()) if session.robot else set()
        assert moved_after == moved_before, (
            f"中止之后机器人还收到了新的运动命令：{sorted(moved_after - moved_before)}"
        )
    finally:
        session.close()

    # "停下来等人"不等于这一趟报废：人把原因解决之后（这里是把帧数上限改回默认），
    # 换一个会话重跑同一套流程必须能正常走完并删掉 RAW。
    # 这一段同时也是"校验不过不是把工具卡死"的正面证据。
    fixed = build_config(
        tmp_path / "retry", joints=("J1",), amplitudes=(0.2,), repeats=1
    )
    fixed.paths.delete_raw_after_process = True
    fixed.validate()
    retry, _recorder2 = open_session(fixed, run_kind="after_human_fix")
    try:
        retry.capture_static(segment_id="static_base", duration_s=0.6)
        retry.run_pretest()
    finally:
        retry.close()
    assert _release_events(retry, "raw_deleted"), (
        "把原因解决之后重跑，还是没删掉 RAW"
    )
    assert _release_events(retry, "raw_kept") == []


def test_the_note_is_rewritten_only_after_the_raw_is_gone(tmp_path: Path, monkeypatch) -> None:
    """改写失败必须响亮地报出来，而且不能把 RAW 已经删掉这件事说成没删。

    第 6 步（原子改写 JSON）理论上会失败（盘满、文件被占）。真发生的时候：
    RAW 确实删了，JSON 还写着 ``true``——这是**唯一**允许的方向性误差
    （保守：不会让人误以为"还能重识别"而删掉别的东西），但必须有
    ``raw_deleted_note_stale`` 事件，让后人知道要手工改哪一份文件。
    """
    from sj_pretest import experiment as experiment_module

    config = _config(tmp_path)
    session, _recorder = open_session(config, run_kind="stale_note")

    original_save = experiment_module.save_vision_json
    saved_with = {"false": 0, "true": 0}

    def flaky_save(path, result, **kwargs):
        if kwargs.get("raw_available") is False:
            saved_with["false"] += 1
            raise OSError("自测：模拟改写落盘失败")
        saved_with["true"] += 1
        return original_save(path, result, **kwargs)

    monkeypatch.setattr(experiment_module, "save_vision_json", flaky_save)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        monkeypatch.undo()
        session.close()

    assert saved_with["false"] > 0, "改写路径一次都没走到，这条测试没测到东西"
    assert session.run is not None
    stale = _release_events(session, "raw_deleted_note_stale")
    assert stale, "改写失败却没有任何 raw_deleted_note_stale 事件"
    for event in stale:
        segment_dir = session.run.segment_dir(str(event["segment_id"]))
        assert not (segment_dir / "frames.raw").exists(), (
            "改写失败了，但 RAW 也没删掉——那不是这条路径"
        )
        payload = _vision_payload(segment_dir)
        assert payload["raw_available"] is True, (
            "改写失败时 JSON 应该保持 true（保守方向），实际 "
            f"{payload['raw_available']}"
        )
        assert "人工" in str(event.get("reason") or ""), (
            f"事件里没有提示人工处理：{event.get('reason')}"
        )
