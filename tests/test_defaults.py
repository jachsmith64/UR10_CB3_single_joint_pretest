"""★ 需求一·1、一·2、一·3：**交付默认值本身**也是被验收的东西。

为什么默认值要单独测
--------------------
前面那些流程测试都会显式改配置（自测跑的是缩小的规模），所以它们测的是
"功能对不对"，测不出"打开工具时它默认是什么样"。而需求一·1 要的恰恰是
**默认就是删 RAW**、一·3 要的是**连上设备就做 5 s 全屏采集检查**——
这两条一旦回退，现场会在"以为留了 RAW 结果被删"或"根本没做采集检查"的状态下
开实验。所以这里直接盯住三处**交付给用户的东西**：

1. 配置类的默认值（``AppConfig()`` 一构造出来是什么样）；
2. 交付 JSON（``configs/experiment_default.json`` 与干运行样例）；
3. 界面上的字（默认打开时那一行状态是不是写着"已开启"）。

三条都对上，才能说"默认值是真的改了"，而不是"某个测试里改了一下"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sj_pretest.config import AppConfig
from sj_pretest.ui import PARAMETER_SPEC, pipeline_status_line

ROOT = Path(__file__).resolve().parents[1]
DELIVERY_JSON = ROOT / "configs" / "experiment_default.json"
DRY_RUN_JSON = ROOT / "docs" / "dry_run_example" / "config.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 一·1：分组处理并删除 RAW 是**默认开**的
# --------------------------------------------------------------------------


def test_the_delivered_default_is_delete_raw_after_process_on() -> None:
    """配置类默认开：一构造就是 True，不需要谁去显式打开。"""
    config = AppConfig()
    assert config.paths.delete_raw_after_process is True, (
        "交付默认必须是「分组处理并删除 RAW：开启」（需求一·1）"
    )
    # 开着的默认必须自洽：删 RAW 的路径只能 stride=1（抽帧删 RAW 会把没算过的
    # 像素丢掉，validate 会拦）。
    assert int(config.paths.process_stride) == 1
    config.validate()


@pytest.mark.parametrize("path", [DELIVERY_JSON, DRY_RUN_JSON])
def test_the_delivered_json_files_agree_with_the_class_default(path: Path) -> None:
    """交付的两个 JSON 也必须是 true（现场按 JSON 建配置，不看代码）。"""
    payload = _load(path)
    assert payload["paths"]["delete_raw_after_process"] is True, (
        f"{path.name} 里 delete_raw_after_process 不是 true——"
        "现场用这份 JSON 建出来的配置就不会删 RAW"
    )
    assert int(payload["paths"]["process_stride"]) == 1


def test_the_ui_says_out_loud_that_the_raw_will_be_deleted() -> None:
    """界面那一行必须一眼看见"已开启"，而且要说清"校验过才删"。"""
    config = AppConfig()
    line = pipeline_status_line(config)
    assert "分组处理并删除RAW：已开启" in line
    assert "校验" in line, f"没有说清删除前有校验：{line}"
    # 关掉之后措辞必须跟着变——不能写死一句"已开启"糊弄过去。
    config.paths.delete_raw_after_process = False
    assert "已关闭" in pipeline_status_line(config)


# --------------------------------------------------------------------------
# 一·3：连上设备之后那次全屏采集检查的参数
# --------------------------------------------------------------------------


def test_the_connection_check_defaults_match_the_requirement() -> None:
    """5 s、帧率下限 95%、写盘余量 1.2、恢复时丢 5 帧——都在配置里，可改。"""
    camera = AppConfig().camera
    assert float(camera.connection_check_s) == pytest.approx(5.0), (
        "需求一·3 要的是**严格 5 s** 的全屏采集检查"
    )
    assert float(camera.min_fps_ratio) == pytest.approx(0.95)
    assert float(camera.min_write_headroom) == pytest.approx(1.2)
    assert int(camera.resume_drain_frames) == 5


@pytest.mark.parametrize("path", [DELIVERY_JSON, DRY_RUN_JSON])
def test_the_json_files_carry_the_same_check_parameters(path: Path) -> None:
    camera = _load(path)["camera"]
    assert float(camera["connection_check_s"]) == pytest.approx(5.0)
    assert float(camera["min_fps_ratio"]) == pytest.approx(0.95)
    assert float(camera["min_write_headroom"]) == pytest.approx(1.2)
    assert int(camera["resume_drain_frames"]) == 5


def test_the_check_duration_is_not_hard_coded_in_the_business_logic(tmp_path: Path) -> None:
    """5 s 只能是**默认值**，不能写死在流程里（现场要能改）。

    做法：真开一个会话（干运行），把检查时长改成 2 s，看落盘的
    ``connection_check.json`` 里记的是不是 2 s——写死的话这里永远是 5。
    """
    from conftest import build_config, open_session

    config = build_config(tmp_path)
    config.camera.connection_check_s = 2.0
    config.validate()
    session, _recorder = open_session(config, run_kind="check_duration")
    try:
        assert session.run is not None
        payload = json.loads(
            (session.run.root / "connection_check" / "connection_check.json").read_text(
                encoding="utf-8"
            )
        )
    finally:
        session.close()
    assert float(payload["seconds"]) == pytest.approx(2.0), (
        "检查时长没有跟着配置走——它被写死在业务逻辑里了"
    )


# --------------------------------------------------------------------------
# 一·2：全屏 RAW + 只影响离线识别的 analysis_roi
# --------------------------------------------------------------------------


def test_the_analysis_roi_defaults_to_the_whole_image() -> None:
    camera = AppConfig().camera
    assert camera.analysis_roi is None, "默认应当是整幅分析（不裁）"
    assert camera.resolved_analysis_roi() is None
    # 旧字段默认也是空的，而且不会凭空迁移出一个 ROI 来。
    assert camera.roi is None


@pytest.mark.parametrize("path", [DELIVERY_JSON, DRY_RUN_JSON])
def test_the_json_files_use_the_new_roi_field(path: Path) -> None:
    camera = _load(path)["camera"]
    assert "analysis_roi" in camera, (
        f"{path.name} 还在用旧的 camera.roi 字段——现场会以为 ROI 改了 RAW 尺寸"
    )
    assert camera["analysis_roi"] in (None, [],)

    config = AppConfig()
    config.camera.analysis_roi = None
    if camera["analysis_roi"]:
        config.camera.analysis_roi = list(camera["analysis_roi"])
    config.validate()


def test_the_ui_can_edit_the_roi_and_the_check_parameters() -> None:
    """界面上必须能改：离线分析 ROI、检查时长、帧率下限、写盘余量、恢复丢帧数。"""
    paths = {item[1] for item in PARAMETER_SPEC}
    for expected in (
        "camera.analysis_roi",
        "camera.connection_check_s",
        "camera.min_fps_ratio",
        "camera.min_write_headroom",
        "camera.resume_drain_frames",
    ):
        assert expected in paths, f"界面上没有 {expected} 这一项，现场改不了"


def test_the_ui_states_that_the_check_runs_right_after_connecting() -> None:
    """界面要说清"连上设备就自动做一次 5 s 检查，不通过禁止运动"。"""
    text = (ROOT / "src" / "sj_pretest" / "ui.py").read_text(encoding="utf-8")
    assert "5 s" in text or "5 秒" in text
    assert "不动机器人" in text and "禁止" in text, (
        "界面上没有说清「这一次检查不动机器人、不通过就禁止运动」"
    )
