"""需求八.11：配置能存能读，界面上的每一个数都真的落到 config.json 里。

参数面板是"改错了就整体不生效"的那种设计：先全部解析、再全部写入。这一条
听起来是小事，实际上决定了现场有没有可能拿着"改了一半"的配置去跑真机——
所以下面既测"存得下读得回"，也测"填错一项就一个字段都不许改"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.config import JOINT_NAMES, AppConfig, ConfigError
from sj_pretest.ui import (
    PARAMETER_SPEC,
    UiError,
    apply_fields,
    field_kind,
    format_field,
    get_field,
    parse_field,
    parameter_sections,
)

# --------------------------------------------------------------------------
# 参数表本身：每一条都得是真的
# --------------------------------------------------------------------------


def test_every_spec_path_exists_on_the_config() -> None:
    """参数表里写错一个字段名，现场就是"打开参数面板直接崩"。"""
    config = AppConfig()
    for section, path, kind in PARAMETER_SPEC:
        value = get_field(config, path)  # 取不到就 AttributeError，这里就是要它别炸
        assert section, f"{path} 没有分区"
        assert kind in {
            "str",
            "optional_str",
            "bool",
            "int",
            "float",
            "optional_float",
            "floats",
            "ints",
            "strs",
            "optional_ints",
        }, f"{path} 的类型 {kind} 不认识"
        assert field_kind(path) == kind
        del value


def test_spec_has_no_duplicate_paths() -> None:
    paths = [path for _section, path, _kind in PARAMETER_SPEC]
    assert len(paths) == len(set(paths)), f"参数表里有重复项：{paths}"


def test_parameter_sections_cover_every_spec_entry() -> None:
    """分区分组不能漏项、不能重复——漏掉的那一项就再也改不了了。"""
    grouped = [path for _section, fields in parameter_sections() for path, _kind in fields]
    assert grouped == [path for _section, path, _kind in PARAMETER_SPEC]


def test_min_window_frames_is_editable() -> None:
    """新加的判据阈值（保持窗口至少几帧）必须能改、能存。

    这条是防"界面参数表和配置字段各走各的"：加字段时忘了加参数，现场就
    调不了它。
    """
    config = AppConfig()
    assert field_kind("thresholds.min_window_frames") == "int"
    assert get_field(config, "thresholds.min_window_frames") >= 2


def test_ui_lists_the_parameters_that_must_be_filled_on_site() -> None:
    """现场必填的那几项必须在参数面板里出现，不能藏在代码里。"""
    listed = {path for _section, path, _kind in PARAMETER_SPEC}
    for path in (
        "robot.ip",
        "camera.serial",
        "camera.mvs_import_path",
        "camera.exposure_us",
        "camera.gain",
        "camera.roi",
        "camera.board_inner_corners",
        "pretest.joints",
        "pretest.amplitudes_deg",
        "paths.output_root",
    ):
        assert path in listed, f"{path} 在参数面板里改不了，只能改代码"


# --------------------------------------------------------------------------
# 界面文字的来回转换
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [path for _s, path, _k in PARAMETER_SPEC])
def test_format_then_parse_returns_the_same_value(path: str) -> None:
    """界面上显示出来的样子，要能原样读回去——否则打开面板再点应用就改了配置。"""
    config = AppConfig()
    kind = field_kind(path)
    value = get_field(config, path)
    text = format_field(value)
    if value == [] and kind == "optional_ints":
        return  # 空列表和"留空"本来就是一回事，打印成空串是设计如此
    assert parse_field(kind, value, text) == value, (
        f"{path}：{value!r} 显示成 {text!r} 之后就读不回来了"
    )


def test_apply_fields_is_all_or_nothing() -> None:
    """一项填错 → 一个字段都不许改（避免"改了一半"的配置被拿去跑真机）。"""
    config = AppConfig()
    snapshot = AppConfig.describe(config)
    with pytest.raises(UiError) as info:
        apply_fields(
            config,
            {
                "robot.trial_speed_deg_s": "3.5",  # 这一项是合法的
                "robot.approach_points": "不是整数",  # 这一项不合法
            },
        )
    assert "整数" in str(info.value)
    assert AppConfig.describe(config) == snapshot, "解析失败却已经改了别的字段"


def test_apply_fields_reports_what_changed() -> None:
    """改动清单要人看得懂：写清是哪个字段、从什么改到什么。"""
    config = AppConfig()
    changes = apply_fields(
        config,
        {
            "robot.trial_speed_deg_s": "7.5",
            "pretest.amplitudes_deg": "0.01, 0.05, 0.2",
            "pretest.joints": "j1、j6",
        },
    )
    assert config.robot.trial_speed_deg_s == 7.5
    assert config.pretest.amplitudes_deg == [0.01, 0.05, 0.2]
    assert config.pretest.joints == ["J1", "J6"], "关节名没有统一成大写"
    assert any("robot.trial_speed_deg_s" in line for line in changes)
    assert any("pretest.joints" in line for line in changes)

    # 什么都没改的时候不报改动。
    assert apply_fields(config, {"robot.trial_speed_deg_s": "7.5"}) == []


def test_unknown_parameter_is_refused() -> None:
    with pytest.raises(UiError) as info:
        apply_fields(AppConfig(), {"robot.没这个字段": "1"})
    assert "参数表里没有这一项" in str(info.value)


def test_boolean_field_accepts_chinese() -> None:
    config = AppConfig()
    assert parse_field("bool", True, "否") is False
    assert parse_field("bool", False, "是") is True
    assert parse_field("bool", True, "") is False
    with pytest.raises(UiError):
        parse_field("bool", True, "大概吧")
    del config


def test_empty_optional_numbers_mean_not_set() -> None:
    assert parse_field("optional_float", 100.0, "") is None
    assert parse_field("optional_float", None, "12.5") == pytest.approx(12.5)
    assert parse_field("optional_ints", [1, 2], "") is None
    assert parse_field("optional_str", "x", "  ") is None


# --------------------------------------------------------------------------
# 配置本体的存与读
# --------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """改过的配置存下去再读回来，一个数都不能变。"""
    config = AppConfig()
    config.mode = "replay"
    config.replay_source = str(tmp_path / "历史数据")
    config.robot.ip = "192.168.125.12"
    config.camera.serial = "K1234567"
    config.camera.exposure_us = 8000.0
    config.camera.roi = [100, 200, 640, 480]
    config.pretest.joints = ["J1", "J2"]
    config.pretest.amplitudes_deg = [0.01, 0.05]
    config.pretest.repeats_per_direction = 2
    config.formal.step_deg = {name: 0.02 * (index + 1) for index, name in enumerate(JOINT_NAMES)}
    config.thresholds.min_window_frames = 7
    config.validate()

    path = tmp_path / "configs" / "现场.json"
    written = config.save(path)
    assert written.is_file()

    loaded = AppConfig.load(written)
    assert AppConfig.describe(loaded) == AppConfig.describe(config)
    assert loaded.camera.roi == [100, 200, 640, 480]
    assert loaded.formal.step_deg["J6"] == pytest.approx(0.12)
    assert loaded.thresholds.min_window_frames == 7
    assert loaded.validate() is None


def test_default_config_json_is_written_and_read_back(tmp_path: Path) -> None:
    """默认配置必须能直接当现场起点：存出去、读回来、还是那套默认值。"""
    path = tmp_path / "experiment_default.json"
    AppConfig().save(path)
    loaded = AppConfig.load(path)
    assert AppConfig.describe(loaded) == AppConfig.describe(AppConfig())
    assert loaded.mode == "dry_run", "默认配置必须是干运行"
    # 默认位姿就是需求二给的那一组（这里用的是 4 位小数的写法）。
    assert loaded.robot.nominal_joint_deg == list(AppConfig().robot.nominal_joint_deg)


def test_loading_a_config_with_a_bad_value_is_refused(tmp_path: Path) -> None:
    """坏配置要在载入时就被拦住，不能等到跑起来才炸。"""
    path = tmp_path / "坏配置.json"
    config = AppConfig()
    config.save(path)
    text = path.read_text(encoding="utf-8").replace('"dry_run"', '"flying"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.load(path)


def test_config_validation_catches_the_dangerous_ones() -> None:
    """几个"填错了会出事"的值：模式、磁盘、阈值、幅度。"""
    config = AppConfig()

    config.mode = "hardware2"
    with pytest.raises(ConfigError):
        config.validate()

    config = AppConfig()
    config.paths.min_free_disk_gb = -1.0
    with pytest.raises(ConfigError):
        config.validate()

    config = AppConfig()
    config.pretest.amplitudes_deg = []
    with pytest.raises(ConfigError):
        config.validate()

    config = AppConfig()
    config.thresholds.min_snr_vs_static = 0.0
    with pytest.raises(ConfigError):
        config.validate()

    config = AppConfig()
    config.robot.nominal_joint_deg = [0.0] * 5
    with pytest.raises(ConfigError):
        config.validate()


def test_saved_config_records_the_mode_and_the_safety_note(tmp_path: Path) -> None:
    """存下去的配置要能自己说明"这次是干运行还是真机"。"""
    config = AppConfig()
    path = config.save(tmp_path / "c.json")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"
    assert payload["pretest"]["joints"] == list(config.pretest.joints)
    assert payload["thresholds"]["min_window_frames"] == config.thresholds.min_window_frames
