"""等待停稳超时：**立刻停、判失败、保住数据、不自动回位、不再发下一条**。

为什么这条要单独测
------------------
1.0.0 里"等停稳超时"只写一行日志就继续往下走 hold → return → 下一条动作。
在真机上这意味着两件坏事同时发生：

* 关节还在动的时候，采集层**继续按"已到位"采保持段**——那一段数据会被当成
  "微动之后的稳定值"参与统计，而它其实还在动；
* 紧接着又发一条新命令，等于在未知状态下叠加下一次运动。

需求二要求改成：超时那一刻就 ``stopJ``、本段判失败、已写的 RAW/RTDE/时间戳
**全部保留**、界面明确提示人工检查、**绝不自动回位**。

这里用干运行 + 一个"永远说没停稳"的机器人包装来触发这条路（不是放宽阈值：
``settle_tolerance_deg`` 保持交付默认 0.002°，只是让询问永远得不到 True；
``settle_timeout_s`` 压到 1 s 只影响"愿意等多久"，不影响超时之后怎么做）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from sj_pretest.analysis import PHASE_HOLD, PHASE_MOVE, PHASE_POST, PHASE_PRE, PHASE_RETURN
from sj_pretest.experiment import ExperimentError
from sj_pretest.robot_joint import MotionAborted

from conftest import read_csv_rows, read_json, rows_for_event


class NeverSettles:
    """包一层机器人：``settled()`` 永远返回 False，其余原样转发。

    这就是"计时到了关节还在动"这件事的替身。**只替换询问的答案**，
    判据本身（容差、连续保持时间）一个都没动。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
    ) -> bool:
        del target_joint_deg, tolerance_deg, hold_s
        return False


def _trigger_timeout(session_factory, **overrides: Any):
    """跑一段预实验，让第一段就停在"等停稳超时"上。

    返回 ``(会话, 记录器, 底层机器人, 配置, 抛出的异常 或 None)``。
    """
    session, recorder, config = session_factory(
        joints=("J1",), amplitudes=(0.2,), repeats=1, **overrides
    )
    config.robot.settle_timeout_s = 1.0  # 只改"等多久"，不改"等不到怎么办"
    config.validate()
    inner = session.robot
    session.robot = NeverSettles(inner)
    session.capture_static(segment_id="static_base", duration_s=0.4)

    error: BaseException | None = None
    try:
        session.run_pretest()
    except BaseException as exc:  # noqa: BLE001 - 这里要的就是"到底抛了什么"
        error = exc
    return session, recorder, inner, config, error


def test_no_second_movej_after_a_settle_timeout(session_factory) -> None:
    """超时之后：**没有第二条运动命令**，尤其没有回程。"""
    session, _recorder, inner, _config, error = _trigger_timeout(session_factory)
    assert session.run is not None

    moved = inner.moved_event_ids()
    assert len(moved) == 1, (
        f"超时之后还发了别的运动命令：{moved}。"
        "需求二要求'不继续 hold、不回程、不下一条'。"
    )
    segment_id = moved[0]
    assert segment_id.startswith("pretest-"), f"第一条不是预实验动作：{segment_id}"

    metadata = read_json(session.run.segment_dir(segment_id) / "capture_metadata.json")
    planned_return = metadata["return_event_id"]
    assert planned_return, (
        "这一段本来就没有计划回程，这条断言就失去意义了——换个动作再测"
    )
    assert planned_return not in moved, "超时之后把回程发出去了"
    assert "return" not in [str(item) for item in moved]

    events = session.run.events.read_all()
    aborted = [item for item in events if item.get("event") == "aborted"]
    assert aborted, "会话没有被中止"
    assert "停稳超时" in str(aborted[0]["reason"]), (
        f"中止原因没写清楚是停稳超时：{aborted[0]['reason']}"
    )
    # 中止是"硬停"：后面任何动作都会被拦住（抛 MotionAborted / ExperimentError）。
    assert isinstance(error, (MotionAborted, ExperimentError)), (
        f"超时之后没有把流程停住，而是抛了 {error!r}"
    )


def test_timeout_is_recorded_and_the_human_is_told_to_inspect(session_factory) -> None:
    """事件流和界面日志要同时说明：做了什么、**没做什么**、人该干什么。"""
    session, recorder, inner, _config, _error = _trigger_timeout(session_factory)
    assert session.run is not None
    segment_id = inner.moved_event_ids()[0]

    events = session.run.events.read_all()
    kinds = [str(item.get("event")) for item in events]
    assert "settle_timeout" in kinds, f"事件流里没有 settle_timeout：{kinds}"
    timeout_event = next(item for item in events if item.get("event") == "settle_timeout")
    assert timeout_event["action"] == "stopJ", "超时那一刻没有立刻请求 stopJ"
    assert timeout_event["event_id"] == segment_id

    failed = [item for item in events if item.get("event") == "segment_failed"]
    assert [item.get("reason") for item in failed] == ["settle_timeout"], (
        f"没有把这一段判失败：{failed}"
    )
    assert Path(str(failed[0]["kept_dir"])).is_dir(), (
        "判失败时没有记下'数据留在哪个目录'"
    )
    assert failed[0]["aborted_by_timeout"] is True

    text = recorder.text()
    for phrase in (
        "立刻 stopJ",
        "RAW、RTDE 状态流和时间戳**已全部保留**",
        "没有继续 hold",
        "没有自动回位",
        "请人工检查",
    ):
        assert phrase in text, f"界面日志里缺少这句说明：{phrase}"
    assert "已中止" in text
    # notes.txt 里也要留一份（界面日志会滚掉，事后翻目录要能看见）。
    notes = (session.run.root / "notes.txt").read_text(encoding="utf-8")
    assert "停稳超时" in notes


def test_raw_rtde_and_timestamps_survive_the_timeout(session_factory) -> None:
    """超时那一段的 RAW / RTDE 状态流 / 每帧时间戳，一个都不许少。"""
    session, _recorder, inner, _config, _error = _trigger_timeout(session_factory)
    assert session.run is not None
    segment_id = inner.moved_event_ids()[0]
    segment_dir = session.run.segment_dir(segment_id)

    raw = segment_dir / "frames.raw"
    assert raw.is_file() and raw.stat().st_size > 0, "超时那一段的 RAW 不见了"
    metadata = read_json(segment_dir / "capture_metadata.json")
    assert int(metadata["frame_count"]) > 0
    rows = read_csv_rows(segment_dir / "frame_timestamps.csv")
    assert len(rows) == int(metadata["frame_count"]), "每帧时间戳的条数和帧数对不上"
    assert int(metadata["missing_frame_count"]) == 0

    state_rows = read_csv_rows(session.run.root / "robot_states.csv")
    mine = rows_for_event(state_rows, segment_id)
    assert mine, "RTDE 状态流里没有这一段：超时那一段的机器人状态没留住"
    assert any(row["command_q_J1_deg"] not in ("", "nan") for row in mine)


def test_the_move_phase_is_marked_as_ended_by_timeout(session_factory) -> None:
    """采集层要把"这一段是超时结束的"如实标出来，不冒充停稳。"""
    session, _recorder, inner, _config, _error = _trigger_timeout(session_factory)
    assert session.run is not None
    segment_id = inner.moved_event_ids()[0]
    captured = [
        item
        for item in session.run.events.read_all()
        if item.get("event") == "segment_captured"
        and item.get("segment_id") == segment_id
    ]
    assert captured, f"没有 {segment_id} 的 segment_captured 事件"
    record = captured[0]
    assert record["aborted_by_timeout"] is True
    assert record["stopped_early"] is True
    metadata = read_json(session.run.segment_dir(segment_id) / "capture_metadata.json")
    phases = {item["label"]: item for item in metadata["phases"]}
    assert phases[PHASE_MOVE]["ended_by"] == "timeout", phases[PHASE_MOVE]
    # 超时之后**没有**再进保持段和回程段（不是跑了却漏了标记）。
    assert PHASE_RETURN not in phases, "超时之后居然还进了回程相位"
    assert PHASE_HOLD not in phases, "超时之后居然还进了保持相位"
    assert PHASE_POST not in phases
    assert PHASE_PRE in phases  # 运动前那一段是正常跑完的


def test_a_failed_segment_keeps_its_raw_even_with_rolling_delete(session_factory) -> None:
    """开了"边采边清"也不许删这一段——它是留给人工检查的。"""
    session, _recorder, inner, config, _error = _trigger_timeout(
        session_factory, paths__delete_raw_after_process=True
    )
    assert config.paths.delete_raw_after_process is True
    assert session.run is not None
    segment_id = inner.moved_event_ids()[0]
    raw = session.run.segment_dir(segment_id) / "frames.raw"
    assert raw.is_file(), (
        "判失败（停稳超时）的段在边采边清下被删掉了；人工检查要看的就是这一段"
    )
    events = session.run.events.read_all()
    mine = [item for item in events if item.get("segment_id") == segment_id]
    deleted = [item for item in mine if item.get("event") == "raw_deleted"]
    assert not deleted, f"判失败的段被边采边清删掉了：{deleted}"
    kept = [item for item in mine if item.get("event") == "raw_kept"]
    assert kept, "既没删也没写 raw_kept：这一段的下落不明"
    assert "停稳超时" in str(kept[0]["reason"]) or "失败" in str(kept[0]["reason"]), (
        f"保留 RAW 的理由说得不清楚：{kept[0]['reason']}"
    )
