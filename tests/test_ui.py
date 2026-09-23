"""需求八.1：界面能起来，三个按钮按顺序点一遍能走完全流程。

``threaded=False`` 时界面里所有长任务都是**同步**跑的，人机确认由
``UiOptions.confirm_policy`` 回答——所以这里能在不开窗口主循环的情况下
把"点按钮"这件事真的跑一遍，而不是只看控件在不在。

两条界面层的红线在这里各有一条测试：
* 没有人能回答确认时，一步都不许动；
* 切到 hardware 必须有人明确点头（默认配置永远是 dry_run）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sj_pretest.config import AppConfig
from sj_pretest.ui import (
    BUTTON_ANALYZE,
    BUTTON_CONNECT,
    BUTTON_GROUP_A,
    BUTTON_GROUP_B,
    BUTTON_PRETEST,
    BUTTON_STOP,
    ExperimentApp,
    ParameterDialog,
    UiOptions,
    preflight,
)

from conftest import build_config


# --------------------------------------------------------------------------
# 自测不许弹真窗口
# --------------------------------------------------------------------------

#: 自测期间 messagebox 记下的内容。任何一条弹框都会被记在这里而不是弹出来。
BOXES: list[str] = []


@pytest.fixture(autouse=True)
def _no_real_dialogs(monkeypatch):
    """把所有模态对话框换成记录器。

    界面自测里必须**一次都不弹真窗口**：``messagebox`` 是**阻塞**的，一旦弹出来
    就会把整个测试进程挂在那里等人点，而且窗口会出现在用户的桌面上——自测弹窗
    到别人屏幕上还卡住，这本身就是缺陷。被测的东西是"弹框说了什么"和"配置被改
    成什么样"，不是"窗口长得对不对"。
    """
    import tkinter.messagebox as messagebox

    BOXES.clear()

    def record(kind):
        def inner(title=None, message=None, **kwargs):
            BOXES.append(f"{kind}|{title}|{message}")
            return True

        return inner

    for name in ("showerror", "showwarning", "showinfo", "askyesno", "askokcancel"):
        monkeypatch.setattr(messagebox, name, record(name), raising=False)
    yield BOXES


@pytest.fixture(scope="module")
def tk_root():
    """整个模块共用一个 Tk 根窗口。

    每个测试各建一个 ``Tk()`` 在同一个进程里会反复初始化 Tcl，跑几条之后
    就会以 "This probably means that tk wasn't installed properly" 收场
    （看着像环境坏了，其实是自测自己的问题）。共用一个根，每次用完把里面的
    控件清干净即可。
    """
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    yield root
    for child in list(root.winfo_children()):
        try:
            child.destroy()
        except Exception:  # pragma: no cover - 收尾尽力而为
            pass
    try:
        root.destroy()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _clean_root(tk_root):
    """每条测试前后都把根窗口里的控件清掉，测试之间互不影响。"""
    yield
    for child in list(tk_root.winfo_children()):
        try:
            child.destroy()
        except Exception:  # pragma: no cover
            pass


def _app(tk_root, tmp_path: Path, *, answer=True, **kwargs):
    """建一个同步模式的界面：不真的 mainloop，一切按调用顺序发生。

    日志不清空——界面一打开写的那几行安全边界本身就是被测对象之一。
    """
    config = build_config(tmp_path, **kwargs)
    options = UiOptions(
        threaded=False,
        confirm_policy=(lambda label: bool(answer)),
        analysis_stride=8,
    )
    return ExperimentApp(config, root=tk_root, options=options)


def _log(app: ExperimentApp) -> str:
    return app.log_text.get("1.0", "end")


def _labels(app: ExperimentApp) -> set[str]:
    """窗口里所有按钮的文字。"""
    found: set[str] = set()

    def walk(widget) -> None:
        for child in widget.winfo_children():
            try:
                text = child.cget("text")
            except Exception:
                text = None
            if text:
                found.add(str(text))
            walk(child)

    walk(app.root)
    return found


# --------------------------------------------------------------------------
# 界面能起来，该有的控件都在
# --------------------------------------------------------------------------


def test_ui_builds_with_the_three_main_buttons(tk_root, tmp_path: Path) -> None:
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    try:
        labels = _labels(app)
        assert {
            BUTTON_CONNECT,
            BUTTON_PRETEST,
            BUTTON_ANALYZE,
            BUTTON_GROUP_A,
            BUTTON_GROUP_B,
            BUTTON_STOP,
        } <= labels, f"少了按钮：{labels}"
        # 三个主按钮的编号顺序就是操作顺序，不能乱。
        assert BUTTON_CONNECT.startswith("①")
        assert BUTTON_PRETEST.startswith("②")
        assert BUTTON_ANALYZE.startswith("③")

        # 默认干运行：界面一打开就不能是 hardware。
        assert app.config.mode == "dry_run"
        assert "dry_run" in app.root.title()
        assert app.session is None, "还没点按钮一就把设备建起来了"
        # 按钮二/三在会话打开前没有意义，应当是灰的。
        assert str(app.btn_pretest.cget("state")) == "disabled"
        assert str(app.btn_analyze.cget("state")) == "disabled"

        # 安全边界要写在日志第一屏里，而不是藏在帮助菜单。
        text = _log(app)
        assert "没有碰撞模型" in text
        assert "unknown" in text
        assert "中止后不会自动回位" in text

        # 六个正式步长输入框都在。
        assert set(app.step_vars) == {"J1", "J2", "J3", "J4", "J5", "J6"}
    finally:
        app.destroy()


def test_preflight_reports_disk_and_history(tmp_path: Path) -> None:
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    lines = preflight(config)
    assert any("磁盘" in line for line in lines)
    assert not any("失败" in line for line in lines)


# --------------------------------------------------------------------------
# 三个按钮按顺序点一遍
# --------------------------------------------------------------------------


def test_ui_drives_the_whole_flow_without_touching_hardware(
    tk_root, tmp_path: Path
) -> None:
    """按钮一 → 二 → 三 → 组A：全程干运行，每个阶段都有可核对的落盘结果。"""
    # 重复 2 次：这样"重复一致性"这条判据才算得出来，推荐步长才有机会真的填进
    # 上面的输入框——否则这一条测试就测不到"填推荐值"那一步。
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=2)
    try:
        app.on_connect()
        assert app.session is not None, "点按钮一没有把会话建起来"
        assert app.session.run is not None
        assert "运行目录" in _log(app)

        app.on_pretest()
        assert app.session.trials, "按钮二一次试验都没登记"
        assert all(
            trial["stage"] == "pretest" for trial in app.session.trials
        ), "按钮二混进了别的阶段的试验"
        # 1 关节 × 1 档 × 2 方向 × 2 次重复 = 4 次。
        assert len(app.session.trials) == 4, f"试验数不对：{len(app.session.trials)}"

        app.on_analyze()
        assert app.report is not None, "按钮三没有出报告"
        text = _log(app)
        assert "推荐正式实验步长" in text or "推荐正式实验" in text
        recommended = app.report.recommendations
        assert {item.joint for item in recommended} == {
            "J1",
            "J2",
            "J3",
            "J4",
            "J5",
            "J6",
        }
        j1 = next(item for item in recommended if item.joint == "J1")
        if j1.recommended_deg is not None:
            assert app.step_vars["J1"].get() == f"{j1.recommended_deg:.12g}", (
                "推荐步长没有填进界面上的输入框"
            )
        else:
            assert "人工填写更大步长" in text
            assert "不做自动外扩" in text

        # 正式步长由人确认：这里用配置里给的那一档，界面不替人放大。
        assert float(app.step_vars["J1"].get()) == pytest.approx(0.2)

        app.on_formal("A")
        formal = [t for t in app.session.trials if t["stage"] == "formal_a"]
        assert formal, "组A一段都没跑"
        assert "组A采集结束" in _log(app)
        assert "原始数据已落盘" in _log(app)
    finally:
        app.destroy()

    # 干运行绝不碰真机：整个界面流程跑完，ur_rtde 一次都没被导入。
    assert "ur_rtde" not in sys.modules


def test_stop_button_aborts_and_never_returns_home(tk_root, tmp_path: Path) -> None:
    """中止按钮：说清"不自动回位"，会话进入中止状态，之后的按钮都点不动了。"""
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    try:
        app.on_connect()
        app.on_stop()
        assert app.session is not None
        assert app.session.aborted is True
        text = _log(app)
        assert "不会自动回位" in text
        assert "已请求中止" in text
        assert str(app.btn_group_a.cget("state")) == "disabled"
        assert str(app.btn_pretest.cget("state")) == "disabled"
        assert "已中止" in app.status_var.get()
    finally:
        app.destroy()


def test_ui_refuses_every_motion_when_nobody_can_confirm(tk_root, tmp_path: Path) -> None:
    """没有人能回答确认（例如无人值守）：一步都不许动，而且要写明白为什么。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    app = ExperimentApp(
        config,
        root=tk_root,
        options=UiOptions(threaded=False, confirm_policy=None, analysis_stride=8),
    )
    try:
        app.on_connect()
        text = _log(app)
        assert "[未确认]" in text, "没有人确认时应当留下明确记录"
        assert "拒绝这一步" in text
        assert app.session is not None
        assert app.session.trials == []
        moves = [
            event
            for event in (app.session.run.events.read_all() if app.session.run else [])
            if event["event"] == "segment_captured"
        ]
        assert moves == [], "没有人确认，却已经采到东西了"
    finally:
        app.destroy()


# --------------------------------------------------------------------------
# 模式切换
# --------------------------------------------------------------------------


def test_switching_to_hardware_needs_an_explicit_yes(tk_root, tmp_path: Path) -> None:
    """切到 hardware：人点了"否"就必须切回去，配置里不能留下 hardware。"""
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1, answer=False)
    try:
        app.mode_var.set("hardware")
        app.on_mode_change()
        assert app.config.mode == "dry_run", "人没有点头，模式却变成了 hardware"
        assert app.mode_var.get() == "dry_run", "下拉框没有弹回原值"
        assert "模式已切换为 hardware" not in _log(app)
    finally:
        app.destroy()

    # 换成"是"：这时才允许切过去（仍然不会去做任何硬件动作——只是切了个开关）。
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1, answer=True)
    try:
        app.mode_var.set("hardware")
        app.on_mode_change()
        assert app.config.mode == "hardware"
        assert "hardware" in app.root.title()
        assert "模式已切换为 hardware" in _log(app)
        # 切了模式也不代表连上了真机：会话仍然是空的。
        assert app.session is None
        assert "ur_rtde" not in sys.modules
        # 收尾时切回干运行，别把状态留在 hardware 上。
        app.config.mode = "dry_run"
    finally:
        app.destroy()


# --------------------------------------------------------------------------
# 参数面板
# --------------------------------------------------------------------------


def test_parameter_dialog_shows_every_parameter_and_applies(
    tk_root, tmp_path: Path, _no_real_dialogs
) -> None:
    """参数面板：每一项都在，改一项就真的写回配置。"""
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    try:
        dialog = ParameterDialog(app)
        dialog.show()
        assert dialog.entries, "参数面板一个输入框都没建出来"
        assert "robot.ip" in dialog.entries
        assert "thresholds.min_window_frames" in dialog.entries
        # ★ v1.0.3：面板上的 ROI 是 camera.analysis_roi（离线分析用）。
        # 旧名 camera.roi 只作为老配置文件的兼容口，不在面板里——
        # 面板上留一个叫 roi 的框，现场会以为它能裁 RAW 尺寸。
        assert "camera.analysis_roi" in dialog.entries
        assert "camera.roi" not in dialog.entries
        # 面板里的每一格都必须是配置里真有的字段——少一个就是现场改不了。
        assert set(dialog.entries) == {
            path for _section, path, _kind in __import__(
                "sj_pretest.ui", fromlist=["PARAMETER_SPEC"]
            ).PARAMETER_SPEC
        }

        # 4.5 °/s 在 CB3 的上限（5.0）以内，是真机上合法的值。
        dialog.entries["robot.trial_speed_deg_s"].set("4.5")
        dialog.entries["camera.serial"].set("K1234567")
        dialog.apply()
        assert not _no_real_dialogs, f"合法的值却被拒绝了：{list(_no_real_dialogs)}"
        assert app.config.robot.trial_speed_deg_s == pytest.approx(4.5)
        assert app.config.camera.serial == "K1234567"
        assert "参数已更新" in _log(app)
    finally:
        app.destroy()


def test_parameter_dialog_refuses_a_bad_value_without_changing_anything(
    tk_root, tmp_path: Path, _no_real_dialogs
) -> None:
    """填错一格：整体不生效，配置保持原样。"""
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    try:
        before = AppConfig.describe(app.config)
        dialog = ParameterDialog(app)
        dialog.show()
        dialog.entries["robot.trial_speed_deg_s"].set("4.5")  # 这一格是合法的
        dialog.entries["pretest.amplitudes_deg"].set("这不是数")  # 这一格不是
        dialog.apply()
        assert _no_real_dialogs, "填了不合法的一格，却没有报错"
        assert AppConfig.describe(app.config) == before, "一格不合法，别的格却已经被改了"
    finally:
        app.destroy()


def test_parameter_dialog_refuses_a_value_that_breaks_the_speed_limit(
    tk_root, tmp_path: Path, _no_real_dialogs
) -> None:
    """合法数字但不合法的值：超过 CB3 速度上限时，配置**一个字段都不许改**。

    这条是现场最容易碰到的一种：界面会弹"trial 速度/加速度超过上限"，
    但如果实现是"先写进配置、再校验"，弹框就只是提示——那个超限的速度
    已经留在配置里了，下一次点"连接设备"才在真机上炸出来。
    """
    app = _app(tk_root, tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    try:
        limit = float(app.config.robot.max_speed_deg_s)
        original_speed = float(app.config.robot.trial_speed_deg_s)
        assert original_speed <= limit, "自测前提：默认的 trial 速度本来就该在上限以内"
        before = AppConfig.describe(app.config)
        dialog = ParameterDialog(app)
        dialog.show()
        dialog.entries["robot.trial_speed_deg_s"].set(str(limit + 1.5))
        dialog.entries["camera.serial"].set("K1234567")  # 同时改一格合法的
        dialog.apply()

        boxes = list(_no_real_dialogs)
        assert boxes, "速度超过上限却没有报错"
        assert any("上限" in box for box in boxes), f"报错里没说清是上限问题：{boxes}"
        assert app.config.robot.trial_speed_deg_s == pytest.approx(
            original_speed
        ), "超限的速度值被留在配置里了"
        assert app.config.camera.serial == "", "整体校验没过，合法的另一格也被写进去了"
        assert AppConfig.describe(app.config) == before
    finally:
        app.destroy()
