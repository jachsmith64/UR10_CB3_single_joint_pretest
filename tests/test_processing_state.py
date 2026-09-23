"""★ 需求一·7：处理一组的时候，RTDE 和相机各自怎么处置。

处理一组要几十秒到几分钟。这段时间里机械臂停着不动，但另外两样东西还在跑：

* **RTDE 状态流**：继续往里写只会把"处理耗时"混进机器人状态流——第一层判据
  （"指令 → 实际"）会平白多出一段没有任何动作的时间。所以处理期间**暂停采样**
  （先刷盘再暂停，见 ``RobotStateRecorder.pause``），处理完**恢复**，两头都留事件。
* **相机**：没人取帧，缓冲里压着处理期间的旧画面。直接接着取会有两个后果——
  下一段的头几帧其实是处理期间的画面（相位边界被顶歪），以及这几十秒的帧号
  跳号被当成下一段的掉帧。所以恢复之前**丢几帧**并重建帧号基线。

还有一条硬不变量：**处理期间不得发送任何机械臂运动命令**。这一条不能只靠
"代码里没写"来保证——本文件里故意在处理过程中塞一次运动，看它会不会被抓住。

这个文件测四件事：

1. 处理期间 RTDE 真的暂停了、处理完真的恢复了，而且暂停期间一行都没写进去；
2. 相机恢复时丢掉的帧数 = 配置要求，并记下新的帧号基线；
3. 处理期间**不丢帧、不少帧**：掉帧清单里不许出现"处理期间的缺口"；
4. 处理期间一旦有运动命令 → 立刻中止会话（不是写条日志就算了）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.robot_joint import MotionAborted

from conftest import build_config, open_session, read_json


# --------------------------------------------------------------------------
# 装置
# --------------------------------------------------------------------------


def _open(tmp_path: Path, **overrides):
    config = build_config(
        tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1, **overrides
    )
    config.paths.delete_raw_after_process = True
    config.validate()
    session, recorder = open_session(config, run_kind="processing_state")
    return session, recorder, config


def _events(session, *names: str) -> list[dict]:
    assert session.run is not None
    wanted = set(names)
    return [
        item for item in session.run.events.read_all() if item.get("event") in wanted
    ]


def _rtde_row_count(session) -> int:
    assert session.run is not None
    path = session.run.root / "robot_states.csv"
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()])


def test_the_rtde_recorder_pauses_and_resumes_around_every_group(tmp_path: Path) -> None:
    """每一组处理前后都要有暂停/恢复事件，而且配对。"""
    session, _recorder, _config = _open(tmp_path)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    paused = _events(session, "rtde_recorder_paused")
    resumed = _events(session, "rtde_recorder_resumed")
    assert paused, "处理期间没有暂停 RTDE 采样——处理耗时会被混进机器人状态流"
    assert len(resumed) == len(paused), (
        f"暂停 {len(paused)} 次、恢复 {len(resumed)} 次：有一组处理完之后没恢复采样"
    )
    for index, (stop, start) in enumerate(zip(paused, resumed), start=1):
        reason = str(stop.get("reason") or "")
        assert reason, "暂停事件里没写原因，事后查不出是哪一组"
        assert str(start.get("reason") or ""), "恢复事件里没写原因"
        # 先暂停、后恢复（事件流按写入顺序读）。
        assert stop["rows"] is not None and start["rows"] is not None
        # ★ 暂停期间**一行都没写**：恢复时的行数必须等于暂停时的行数。
        assert int(start["rows"]) == int(stop["rows"]), (
            f"第 {index} 次处理：暂停时 {stop['rows']} 行、恢复时 {start['rows']} 行——"
            "处理期间还在往 robot_states.csv 里写"
        )
    # 每一组的处理都要夹在暂停与恢复之间（按事件顺序核对）。
    events = session.run.events.read_all()
    for group_start in range(len(paused)):
        stop_at = events.index(paused[group_start])
        start_at = events.index(resumed[group_start])
        between = [
            item
            for item in events[stop_at + 1 : start_at]
            if item.get("event") == "segment_captured"
        ]
        assert not between, (
            f"第 {group_start + 1} 次处理期间采了新段：{[item['segment_id'] for item in between]}"
        )


def test_no_rtde_rows_are_lost_by_the_pause(tmp_path: Path) -> None:
    """暂停/恢复不能把该采的状态弄丢：每一段在 RTDE 里仍然有行，校验照过。

    （``_verify_releasable`` 第 4 条要求"这一段在 robot_states.csv 里有 ≥ 下限的行"。
    如果暂停把行刷丢了、或者恢复之后的段一条都没采到，那些段就会在校验里被判失败——
    这条用例盯的就是"暂停没有副作用"。）
    """
    session, _recorder, config = _open(tmp_path)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    assert _events(session, "raw_kept") == [], (
        "暂停/恢复之后有段在校验里被判失败（RAW 被保留）——先看 pause 的刷盘顺序"
    )
    assert _events(session, "raw_deleted"), "一段都没处理，这条用例没测到东西"
    rows = _rtde_row_count(session)
    assert rows > 0, "robot_states.csv 里一行都没有"
    metadata = [
        read_json(path)
        for path in sorted(session.run.segments_dir.glob("*/capture_metadata.json"))
    ]
    assert metadata, "一段采集都没落盘"
    for payload in metadata:
        assert int(payload["frame_count"]) > 0
    assert float(config.camera.min_write_headroom) > 1.0


# --------------------------------------------------------------------------
# 相机：处理之后丢帧、重建帧号基线
# --------------------------------------------------------------------------


def test_the_camera_drops_the_old_buffer_and_rebuilds_the_frame_id_baseline(
    tmp_path: Path,
) -> None:
    """每一组处理完之后：丢掉配置要求的帧数，并记下新的帧号基线。"""
    session, _recorder, config = _open(tmp_path)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    resumed = _events(session, "camera_resumed")
    assert resumed, "处理结束后没有相机恢复事件"
    want = int(config.camera.resume_drain_frames)
    assert want > 0
    for item in resumed:
        assert int(item["requested"]) == want, (
            f"丢的帧数不是配置里的 {want}：{item['requested']}"
        )
        assert int(item["dropped_frames"]) == want, (
            f"要求丢 {want} 帧、实际丢了 {item['dropped_frames']} 帧"
        )
        assert item["frame_id_baseline"] is not None, (
            "没有记下新的帧号基线——下一段的掉帧就无法界定"
        )
    # 相机恢复与 RTDE 恢复要成对出现（否则会出现"相机在跑、状态流停着"的段）。
    assert len(_events(session, "rtde_recorder_resumed")) >= len(resumed)


def test_the_drained_frames_are_not_counted_as_drops_in_the_next_segment(
    tmp_path: Path,
) -> None:
    """处理期间没被读的那些帧**不许**算成下一段的掉帧（需求一·7 最后一句）。

    处理一组要几十秒，这段时间相机会继续出帧（真机上压在缓冲里）。如果恢复时
    不丢、不重建基线，这几十秒的帧号跳号就会原封不动出现在下一段的掉帧统计里，
    于是每一组之后都会"凭空掉几千帧"，报告和日志立刻失去意义。
    """
    session, _recorder, config = _open(tmp_path)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()

    assert session.run is not None
    resumed = _events(session, "camera_resumed")
    assert resumed, "没有相机恢复事件"
    baselines = [int(item["frame_id_baseline"]) for item in resumed]
    # 每一段自己的缺帧统计必须是干净的：合成世界里没有注入丢帧，而且处理期间的
    # 帧号都已经被丢掉了，所以"处理过之后采的段"不该有任何缺口。
    for path in sorted(session.run.segments_dir.glob("*/capture_metadata.json")):
        payload = read_json(path)
        assert int(payload["missing_frame_count"]) == 0, (
            f"{path.parent.name} 记了 {payload['missing_frame_count']} 帧缺口——"
            "多半是把处理期间没读的帧算成了掉帧"
        )
        assert int(payload["gap_event_count"]) == 0, payload["segment_id"]
        assert int(payload["maximum_missing_run"]) == 0, payload["segment_id"]
        assert float(payload["actual_camera_fps"]) > 0.0
    # 基线是**单调递增**的：处理之后帧号只会往前走，倒退就说明有段的帧号错了。
    assert baselines == sorted(baselines), (
        f"帧号基线不是单调的：{baselines}——说明有段的帧号倒退了"
    )
    assert len(set(baselines)) == len(baselines), (
        f"有两组用了同一个帧号基线：{baselines}——基线没有重建"
    )
    # ★ 最要紧的一条：按事件顺序走一遍，**每一次相机恢复之后采到的第一段，
    # 必须从基线之后的第一帧开始**。
    # 合成世界的帧号是连续的（没有注入丢帧），所以这个等式是精确的：
    # 丢掉 5 帧、最后一帧号就是基线，下一段的头一帧自然就是"基线 + 1"。
    # 反过来说——如果恢复时没丢帧，下一段会**接着上一段的帧号往下走**
    # （头一帧就是处理期间压在缓冲里的旧画面），这条断言当场就会失败。
    events = session.run.events.read_all()
    pending: int | None = None
    checked = 0
    for item in events:
        name = item.get("event")
        if name == "camera_resumed":
            pending = int(item["frame_id_baseline"])
        elif name == "segment_captured" and pending is not None:
            first = int(item["first_frame_id"])
            assert first == pending + 1, (
                f"{item['segment_id']} 的头一帧是 {first}，而上一组处理完"
                f"丢完帧留下的基线是 {pending}——这一段要么用了处理期间的旧帧，"
                "要么丢帧数没落到实处"
            )
            pending = None
            checked += 1
    assert checked, "一次「恢复之后紧接着采一段」都没有发生，这条用例没测到东西"


def test_zero_drain_is_reported_loudly_instead_of_silently_reusing_frames(
    tmp_path: Path,
) -> None:
    """把丢帧数配成 0 时，必须**明说**"这样不推荐"，而不是悄悄接着用旧帧。"""
    session, recorder, _config = _open(tmp_path, camera__resume_drain_frames=0)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        session.run_pretest()
    finally:
        session.close()
    text = recorder.text()
    assert "resume_drain_frames=0" in text and "不推荐" in text, (
        f"配成 0 之后日志里没有明确提示：\n{text}"
    )


# --------------------------------------------------------------------------
# 硬不变量：处理期间不许发运动命令
# --------------------------------------------------------------------------


def test_a_motion_during_processing_is_caught_and_aborts_the_session(
    tmp_path: Path, monkeypatch
) -> None:
    """★ 故意在处理过程中发一次运动命令：必须**被抓到并中止**，而不是写条日志。

    为什么这条要用"注入违规"的方式测：需求一·7 写的是"处理期间不得发送任何
    机械臂运动命令"。只检查"当前代码里没有这种调用"是测不出保护是否存在的——
    将来任何一次重构都可能把某个 helper 挪进处理路径，而现场看到的现象会是
    "机械臂在处理期间动了一下"。所以这里真的发一条运动命令，看闸门会不会响。

    预期：会话立刻中止（``aborted``）、抛 ``MotionAborted``、中止原因里点明
    "处理期间"和"运动命令"。
    """
    from sj_pretest import experiment as experiment_module

    session, _recorder, _config = _open(tmp_path)
    robot = session.robot
    assert robot is not None
    original_process = experiment_module.process_segment
    fired = {"count": 0}

    def rogue_process(segment_dir, **kwargs):
        result = original_process(segment_dir, **kwargs)
        if fired["count"] == 0:
            fired["count"] = 1
            # 处理期间"发一条运动命令"（目标就是名义姿态，本身不危险——
            # 这里测的是**闸门**，不是让机械臂真动）。
            robot.move_to_joint(
                session.config.robot.nominal_joint_deg,
                speed_deg_s=1.0,
                accel_deg_s2=1.0,
                event_id="自测注入的运动",
                label="自测：处理期间注入的运动命令",
            )
        return result

    monkeypatch.setattr(experiment_module, "process_segment", rogue_process)
    try:
        session.capture_static(segment_id="static_base", duration_s=0.6)
        with pytest.raises(MotionAborted) as info:
            session.run_pretest()
        monkeypatch.undo()

        assert fired["count"] == 1, "注入的运动没有发生，这条用例没测到东西"
        text = str(info.value)
        assert "处理" in text and "运动命令" in text, text
        assert session.aborted is True, "处理期间发了运动命令，会话居然没有中止"
        assert session.run is not None
        aborted = _events(session, "aborted")
        assert aborted, "事件流里没有 aborted 这一笔"
        assert "处理" in str(aborted[0].get("reason") or ""), aborted[0]
        # 中止是**粘住**的：再按一次"继续实验"也走不动，而且不会再发出任何命令。
        before = set(robot.moved_event_ids())
        with pytest.raises(MotionAborted):
            session.run_pretest()
        assert set(robot.moved_event_ids()) == before, (
            "会话已经中止，重跑却还是发出了新的运动命令"
        )
    finally:
        monkeypatch.undo()
        session.close()
