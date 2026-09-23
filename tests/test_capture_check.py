"""★ 需求一·3：连上设备之后、任何运动之前，先做一次 5 s 全屏采集检查。

这条检查要回答的问题只有一个：**这台机器能不能以全屏 RAW 采得动、写得下。**
答不上来就不许开工——因为后面每一组的磁盘峰值都要拿这次实测的分辨率和帧率去算，
拿配置里的名义值去卡一个可能大得多的真实画面，正是"跑一半磁盘写满"的成因。

所以这个文件盯住五件事：

1. **时机**：连上设备就做，而且**不动机器人**（一次运动命令都没有）；
2. **量到的东西**：实际原始分辨率、实际帧率、帧号连续性（缺帧数量与比例）、
   实际 RAW 字节数、实际每秒写盘速度——五个值都要落在汇总里，而且**互相自洽**
   （字节数 = 帧数 × 每帧字节数之类）；
3. **通过就轻装上路**：把这次测的 RAW 删掉，只留汇总 JSON/TXT、帧率/缺帧/写盘
   速度、少量样本图、实际分辨率；
4. **不通过就不许动**：三种原因（帧率不足 / 掉帧 / 磁盘写入不足）分开判、分开说，
   而且闸门真的拦住运动；
5. **实测值要顶替名义值**：后面算每一组峰值时用的是这次实测的尺寸和帧率。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sj_pretest.config import AppConfig, group_forecast
from sj_pretest.experiment import ExperimentError

from conftest import build_config, open_session, read_json


# --------------------------------------------------------------------------
# 装置
# --------------------------------------------------------------------------


def _open(tmp_path: Path, **overrides):
    """开一个会话（连设备时会自动跑那次 5 s 全屏采集检查）。

    自测把检查时长压到 1 s（配置允许的最短值）——这是**规模**选择：
    帧率下限比例、缺帧上限、写盘余量三个判据一个都没动。
    """
    config = build_config(
        tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1, **overrides
    )
    config.validate()
    session, recorder = open_session(config, run_kind="capture_check")
    return session, recorder, config


def _check_dir(session) -> Path:
    assert session.run is not None
    return session.run.root / "connection_check"


def _summary(session) -> dict:
    return read_json(_check_dir(session) / "connection_check.json")


def _recheck(session) -> dict:
    """再跑一次检查（用来测三种失败原因）。"""
    return session.run_connection_check()


# --------------------------------------------------------------------------
# 1) 时机：连上设备就做，而且不动机器人
# --------------------------------------------------------------------------


def test_the_check_runs_right_after_connecting_without_any_motion(tmp_path: Path) -> None:
    """连上设备之后立刻就有检查结果，而且这一次**一条运动命令都没发**。"""
    session, _recorder, _config = _open(tmp_path)
    try:
        assert session.capture_check is not None, (
            "连上设备之后没有 5 s 全屏采集检查的结果——它应该发生在 connect_devices 里"
        )
        assert session.measured is not None, "实测值没有被留下来给后面的估算用"
        # 检查自己声明过"这一次没有运动"。
        assert session.capture_check.get("segment_id") == "connection_check"
        # ★ 一条运动命令都没有：机器人层的"动过哪几段"必须是空的。
        assert session.robot is not None
        assert session.robot.moved_event_ids() == [], (
            "5 s 采集检查期间机器人收到了运动命令："
            f"{session.robot.moved_event_ids()}（需求一·3：这 5 s 里不动机器人）"
        )
        assert session.run is not None
        events = [
            item
            for item in session.run.events.read_all()
            if item.get("event") == "connection_check"
        ]
        assert events, "事件流里没有 connection_check 这一笔"
    finally:
        session.close()


def test_repeating_the_check_still_sends_no_motion_command(tmp_path: Path) -> None:
    """重复做检查（现场改完曝光/帧率会重做）同样不许发运动命令。"""
    session, _recorder, _config = _open(tmp_path)
    try:
        before = set(session.robot.moved_event_ids())
        _recheck(session)
        assert set(session.robot.moved_event_ids()) == before, (
            "重做采集检查时发了运动命令"
        )
    finally:
        session.close()


# --------------------------------------------------------------------------
# 2) 量到的五个值：互相自洽
# --------------------------------------------------------------------------


def test_the_check_reports_the_five_real_values_and_they_agree(tmp_path: Path) -> None:
    """五个实测值齐全，而且彼此自洽；分辨率是**量出来的**，不是抄配置的。"""
    session, _recorder, config = _open(tmp_path)
    try:
        summary = _summary(session)
        measured = summary["measured"]
        frames = int(summary["frames"])

        # (1) 实际原始分辨率：合成世界出多大就报多大（真机上是相机报的原始尺寸）。
        assert int(measured["width"]) == int(config.dry_run.width)
        assert int(measured["height"]) == int(config.dry_run.height)
        # ★ 它**不等于**相机型号的满幅名义值——真机上两者可能不同（binning/裁切），
        # 所以"报的是实测值"这件事本身要能被看见。
        assert (int(measured["width"]), int(measured["height"])) != (
            int(config.camera.sensor_width),
            int(config.camera.sensor_height),
        ), "自测的合成尺寸恰好等于满幅名义值，这条断言就失去意义了（请调开两者）"

        # (2) 实际帧率：合成世界的真实出帧节奏。
        assert float(measured["fps"]) == pytest.approx(float(config.dry_run.fps), rel=1e-6)
        assert float(measured["fps"]) >= float(summary["min_fps"]), summary["failures"]

        # (3) 帧号连续性：采到的帧数、缺的帧数、缺失比例三者要对得上。
        assert frames > 0
        assert int(summary["missing_frames"]) == int(measured["missing_frames"])
        assert float(summary["dropped_ratio"]) == pytest.approx(
            float(measured["dropped_ratio"]), rel=1e-9
        )
        assert int(summary["last_frame_id"]) - int(summary["first_frame_id"]) + 1 >= frames

        # (4) 实际 RAW 字节数：= 帧数 × 每帧字节数（整幅、不含任何裁切）。
        assert int(summary["raw_bytes"]) == frames * int(measured["frame_bytes"])
        assert int(measured["frame_bytes"]) == int(measured["width"]) * int(
            measured["height"]
        ), "每帧字节数不是「宽 × 高」（单通道 8 位整幅），RAW 尺寸口径被改了"
        # 而且盘上真写过这么多字节（删之前的实测证据来自文件本身）。
        assert int(summary["raw_bytes"]) > 0

        # (5) 实际每秒写盘速度：字节 ÷ 这一段真实花掉的墙上时间。
        assert float(summary["raw_seconds_wall"]) > 0
        assert float(summary["write_mbps"]) == pytest.approx(
            int(summary["raw_bytes"]) / 1e6 / float(summary["raw_seconds_wall"]),
            rel=1e-6,
        )
        assert float(summary["write_mbps"]) >= float(summary["need_mbps"]), (
            f"写盘 {summary['write_mbps']} 低于需要的 {summary['need_mbps']}"
        )

        # 五项都要在给人看的那几行里出现（现场是照着这几行读数的）。
        text = "\n".join(summary["lines"])
        for label in (
            "实际原始分辨率",
            "实际帧率",
            "帧号连续性",
            "实际 RAW 字节数",
            "实际每秒写盘速度",
        ):
            assert label in text, f"汇总里没有「{label}」这一行：\n{text}"
        assert "结论：通过" in text, text
    finally:
        session.close()


# --------------------------------------------------------------------------
# 3) 通过之后：删掉测试 RAW，只留下小东西
# --------------------------------------------------------------------------


def test_the_test_raw_is_deleted_on_pass_and_the_evidence_stays(tmp_path: Path) -> None:
    """通过就删掉这一份测试 RAW；汇总、样本图、实测值一个都不能少。"""
    session, _recorder, _config = _open(tmp_path)
    try:
        summary = _summary(session)
        assert summary["ok"] is True, summary["failures"]
        out_dir = _check_dir(session)
        assert not (out_dir / "frames.raw").exists(), (
            "检查通过了，测试 RAW 却还留在盘上（需求一·3：通过就删掉）"
        )
        assert not list(out_dir.glob("frames.raw*")), "留下了写盘中断的 .tmp"
        assert summary["raw_deleted"] is True, "汇总里没有如实记下「测试 RAW 已删除」"
        # 保留下来的四样：汇总 JSON/TXT、样本图、实际分辨率（在汇总里）。
        assert (out_dir / "connection_check.json").is_file()
        assert (out_dir / "connection_check.txt").is_file()
        samples = sorted(out_dir.glob("sample_*.png"))
        assert samples, "样本图被一起删掉了——它们只有几十 KB，是现场复核的证据"
        assert [path.name for path in samples] == list(summary["sample_images"])
        assert int(summary["measured"]["width"]) > 0
        # 文本汇总里也要写清楚"删了什么、留了什么"。
        assert "测试 RAW 已删除" in "\n".join(summary["lines"])
    finally:
        session.close()
    # 补一句：删掉的只是**测试** RAW，之后正式采集的段一个字节都不受影响
    # （那由分组流水线管，见 test_group_pipeline / test_raw_status_order）。
    assert not list(_check_dir(session).glob("frames.raw*"))


# --------------------------------------------------------------------------
# 4) 不通过：三种原因分开说，而且真的拦住运动
# --------------------------------------------------------------------------


def _fail_with(session, *, why: str, monkeypatch) -> dict:
    """按原因制造一种真实的失败，然后重做检查。"""
    config = session.config
    if why == "fps":
        # 这台机器实际只能跑 60 fps，而**配置里的期望帧率**是交付默认的 132.23
        # （真机上这个期望值来自 ``camera.expected_fps``，由相机型号和现场设定决定）。
        # 出帧节奏在开会话之前就调成 60 fps（所以帧时间戳与帧号仍然自洽，
        # 不会把"时间轴校验"搅进来），这里只把"期望帧率"保持成交付默认值。
        # 判据本身（期望值的 95%）一个字都没动。
        assert float(config.dry_run.fps) == pytest.approx(60.0), (
            "这一条要求会话是在 dry_run.fps=60 下开的（见调用方），否则模拟不出来"
        )
        monkeypatch.setattr(
            config,
            "effective_fps",
            lambda measured=None: (
                float(measured.fps)
                if measured is not None
                else float(config.camera.expected_fps)
            ),
            raising=False,
        )
    elif why == "drop":
        # 每 3 帧丢 1 帧 ≈ 33%，远超缺帧上限。
        config.dry_run.drop_every = 3
    elif why == "disk":
        # 要求写盘速度达到实际能力的上千倍：模拟"盘比相机慢"这一种失败。
        config.camera.min_write_headroom = 1e4
    else:  # pragma: no cover - 调用方只传上面三种
        raise AssertionError(why)
    config.validate()
    return _recheck(session)


@pytest.mark.parametrize(
    "why, keyword",
    [("fps", "帧率不足"), ("drop", "掉帧"), ("disk", "磁盘写入不足")],
)
def test_each_failure_reason_is_worded_separately_and_keeps_the_raw(
    tmp_path: Path, why: str, keyword: str, monkeypatch
) -> None:
    """三种失败各自单独说清是哪一种，并且**保留**测试 RAW 供排查。

    分开判是有现场意义的：帧率不足要去看相机/曝光/带宽；掉帧要去看线缆和触发；
    写盘不足要换盘或者降低采集规模。三种原因的处置完全不同，
    笼统报一句"采集检查没通过"等于让人自己猜。
    """
    overrides = {"dry_run__fps": 60.0} if why == "fps" else {}
    session, _recorder, _config = _open(tmp_path, **overrides)
    try:
        summary = _fail_with(session, why=why, monkeypatch=monkeypatch)
        assert summary["ok"] is False, f"{why}：制造出来的失败居然通过了"
        assert summary["failures"], "未通过却没写原因"
        text = "\n".join(summary["failures"])
        assert keyword in text, (
            f"{why} 这一种失败没有用中文说清是「{keyword}」：{summary['failures']}"
        )
        # 另外两类原因不该被误报（分开判=不混着报）。
        for other in ("帧率不足", "掉帧", "磁盘写入不足"):
            if other != keyword:
                assert other not in text, f"{why} 的报错里混进了「{other}」：{text}"
        assert "结论：通过" not in "\n".join(summary["lines"])
        assert "未通过" in "\n".join(summary["lines"])
        # 未通过时**保留**测试 RAW：为什么采成这样要看原始帧。
        assert (_check_dir(session) / "frames.raw").is_file(), (
            f"{why}：检查没过却把测试 RAW 删了——排查现场的第一手材料就没了"
        )
        assert summary["raw_deleted"] is False
        assert session.capture_check_ok is False

        # ★ 硬闸门：此刻**任何**运动都被拦住，而且一条命令都没发出去。
        for runner in (session.run_approach, session.run_pretest):
            with pytest.raises(ExperimentError) as info:
                runner()
            assert keyword in str(info.value), str(info.value)
            assert "禁止运动" in str(info.value) or "没有开始" in str(info.value), str(
                info.value
            )
        assert session.robot.moved_event_ids() == [], (
            f"{why}：检查没过，机器人却收到了运动命令 "
            f"{session.robot.moved_event_ids()}"
        )
    finally:
        session.close()


def test_the_gate_refuses_when_no_check_was_done_at_all(tmp_path: Path) -> None:
    """连检查都没做过也要拦住——"没量过这台机器"绝不能等于"默认没问题"。"""
    session, _recorder, _config = _open(tmp_path)
    try:
        session.capture_check = None
        session.capture_check_ok = False
        session.measured = None
        with pytest.raises(ExperimentError) as info:
            session.require_capture_check("正式实验")
        assert "5 s" in str(info.value) or "5 秒" in str(info.value), str(info.value)
        assert session.robot.moved_event_ids() == []
    finally:
        session.close()


# --------------------------------------------------------------------------
# 5) 实测值顶替名义值：每一组的峰值按实测尺寸和帧率重算
# --------------------------------------------------------------------------


def test_the_measured_size_replaces_the_nominal_one_in_every_group_estimate(
    tmp_path: Path,
) -> None:
    """按实测分辨率/帧率重算每一组峰值；名义满幅值不得再出现。

    真机上这一步很要紧：相机可能以 binning 或者裁切输出，实际写下去的画面
    和"型号满载 1936×1096"不一致。拿名义值算，等于把磁盘闸门架在一次猜上。
    """
    from sj_pretest.experiment import iter_segment_plans
    from sj_pretest.joint_space import build_formal_group_a

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    # 合成尺寸刻意取一个和满幅名义值都不同的数（两边都不是），证明用的是实测值。
    config.dry_run.width = 512
    config.dry_run.height = 384
    config.dry_run.fps = 100.0
    config.validate()
    session, _recorder = open_session(config, run_kind="measured_scale")
    try:
        measured = session.measured
        assert measured is not None
        assert (int(measured.width), int(measured.height)) == (512, 384)
        assert float(measured.fps) == pytest.approx(100.0, rel=1e-6)

        segments = iter_segment_plans(
            build_formal_group_a(config, config.robot.nominal_joint_deg, "J1", 0.2)
        )
        forecast = group_forecast(
            config, segments, name="组A J1", measured=measured
        )
        assert (forecast.width, forecast.height) == (512, 384), (
            f"本组预报用的是 {forecast.width}×{forecast.height}，不是实测的 512×384"
        )
        assert float(forecast.fps) == pytest.approx(100.0, rel=1e-6)
        assert (forecast.width, forecast.height) != (
            config.camera.sensor_width,
            config.camera.sensor_height,
        )
        # 帧数 = 预计录制秒数 × 实测帧率（不是名义 132.23）。
        assert int(forecast.frames) == pytest.approx(
            float(forecast.seconds) * 100.0, abs=2.0
        ), f"预计帧数 {forecast.frames} 和「秒数 × 实测帧率」对不上"
        # 界面/日志上要看得见这个尺寸是从哪儿来的。
        text = "\n".join(forecast.lines())
        assert "512×384" in text, text
        assert "连接设备后的" in text or "实测" in text or "measured" in forecast.measured_source, (
            f"没有说明这个尺寸是实测来的：{forecast.measured_source}"
        )
    finally:
        session.close()


def test_the_summary_is_valid_json_and_can_be_reread(tmp_path: Path) -> None:
    """汇总必须是能复算的纯文本：JSON 可解析、TXT 有同样的几行。

    现场出问题时人拿到的是这两个文件（RAW 已经删了），所以它们必须自解释。
    """
    session, _recorder, _config = _open(tmp_path)
    try:
        out_dir = _check_dir(session)
        payload = json.loads((out_dir / "connection_check.json").read_text(encoding="utf-8"))
        text = (out_dir / "connection_check.txt").read_text(encoding="utf-8")
        for line in payload["lines"]:
            assert line in text, f"TXT 汇总里缺了这一行：{line}"
        assert "连接设备后的" in text.splitlines()[0]
        # 不许出现乱码中文文件名（需求七）。
        for path in out_dir.iterdir():
            assert path.name.isascii(), f"落盘文件名不是纯 ASCII：{path.name}"
    finally:
        session.close()
