"""★ 需求一·2：RAW 一律**整幅**，ROI 只用于**离线分析**。

为什么这条要单独测
------------------
1.0.0/1.0.2 里 ROI 是**采集参数**：采集引擎裁一份写进 RAW，棋盘格完整性检查
用另一份，到位预览又用未裁剪的整幅——三套互不相干的裁剪。后果现场才看得出来：
界面上棋盘格完整、离线识别却在裁过的图里找不齐角点；或者预览里没越界、
RAW 却因为 ROI 写大了而写不出来。更要命的是"裁掉的那部分**永久没有了**"：
棋盘格一旦因为碰撞、松动或机械臂挪动跑出 ROI，现场再也查不出原因。

v1.0.3 把两件事彻底分开：

* ``frames.raw`` **逐帧保存整幅原始帧**，尺寸只由相机决定，任何配置都改不了它；
* ``camera.analysis_roi`` 只决定**离线识别在画面里的哪一块找棋盘格**，
  角点在存盘前已经加回 ROI 偏移，所以对外（角点 CSV、质心、逐帧几何、
  位移、转角）**一律是原图坐标**，和 RAW 的口径一致。

这里测的就是这两条，以及它们与"磁盘估算"的关系（估算也一律按全屏算，
见 ``test_group_pipeline.test_the_analysis_roi_never_changes_the_disk_estimate``）。

测的都是可核对的事实：RAW 的**实际字节数**、``capture_metadata.json`` 里的
尺寸五件套、预览 PNG 的**实际像素尺寸**、角点 CSV 里的**实际坐标**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.experiment import ExperimentError

from conftest import build_config, open_session, read_json

#: 一块**完整包住棋盘格、又比棋盘格大出一圈**的分析 ROI（原图坐标）。
#: 自测画面是 340×260，合成世界里棋盘格的内角点落在 x∈[89.3, 249.4]、
#: y∈[72.6, 184.6]（实测），而检查要求内角点到分析窗口边缘至少留 40 px
#: （``camera.min_margin_px`` 的交付默认值，一个字都没改）。
#: 所以这块 ROI 的四条边余量分别是 41.3 / 46.6 / 40.6 / 47.4 px —— 刚刚够，
#: 也就意味着"余量确实是在分析窗口里算的"这件事真的被走到了。
ROI_BOARD = [48, 32, 248, 200]


def _png_size(path: Path) -> tuple[int, int]:
    """读 PNG 的真实像素尺寸（不靠 PIL：Pillow 只是可选依赖）。"""
    with Path(path).open("rb") as handle:
        header = handle.read(33)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} 不是 PNG"
    assert header[12:16] == b"IHDR", f"{path} 的 PNG 头不完整"
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def _segments(session) -> list[dict]:
    assert session.run is not None
    return [
        read_json(path)
        for path in sorted(session.run.segments_dir.glob("*/capture_metadata.json"))
    ]


def _read_corners_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# --------------------------------------------------------------------------
# 1) RAW 是全屏，ROI 改不了它
# --------------------------------------------------------------------------


def test_the_raw_is_always_the_full_frame_whatever_the_roi_says(tmp_path: Path) -> None:
    """配了分析 ROI 也一样：RAW 的字节数 = 整幅尺寸 × 帧数。

    三种写法各采一段，比的都是**盘上 frames.raw 的真实字节数**：
    不配 ROI、配一块 248×200 的分析 ROI、以及用 v1.0.2 的旧字段 ``camera.roi``
    配同一块区域。三段必须**一模一样大**——只要有一段的 RAW 变小了，
    就说明"ROI 又在裁采集"这条 v1.0.2 的行为被带回来了，而裁掉的那部分
    在这台机器上就永久没有了。
    """
    from conftest import read_json as _read

    cases = {
        "no_roi": (None, None),
        "analysis_roi": (list(ROI_BOARD), None),
        "legacy_roi_field": (None, list(ROI_BOARD)),
    }
    sizes: dict[str, int] = {}
    for name, (analysis_roi, legacy_roi) in cases.items():
        config = build_config(
            tmp_path / name, joints=("J1",), amplitudes=(0.2,), repeats=1
        )
        config.camera.analysis_roi = analysis_roi
        config.camera.roi = legacy_roi
        config.validate()
        session, _recorder = open_session(config, run_kind=f"raw_{name}")
        try:
            record = session.capture_static(segment_id="static_raw", duration_s=0.4)
            metadata = _read(Path(record.dir) / "capture_metadata.json")
            raw = Path(record.dir) / "frames.raw"
            full_w = int(config.dry_run.width)
            full_h = int(config.dry_run.height)

            # 尺寸五件套必须自洽：原始尺寸 = RAW 实际尺寸 = 全屏。
            assert metadata["source_size"] == [full_w, full_h]
            assert metadata["raw_size"] == [full_w, full_h], (
                f"{name}：RAW 实际保存尺寸被改成了 {metadata['raw_size']}，"
                f"应该是全屏 {[full_w, full_h]}"
            )
            assert metadata["cropped_size"] == [full_w, full_h], (
                "兼容字段 cropped_size 也应该等于全屏（v1.0.3 起不再裁剪）"
            )
            assert int(metadata["frame_bytes"]) == full_w * full_h
            expected = int(metadata["frame_count"]) * full_w * full_h
            assert raw.stat().st_size == expected, (
                f"{name}：RAW 有 {raw.stat().st_size} 字节，"
                f"应该是 {metadata['frame_count']} 帧 × {full_w}×{full_h} = {expected}"
            )
            sizes[name] = int(raw.stat().st_size)
        finally:
            session.close()

    assert len(set(sizes.values())) == 1, (
        f"换了 ROI 写法之后 RAW 大小变了：{sizes}——"
        "ROI 又跑到采集路径上去了，RAW 不再是整幅"
    )


def test_metadata_says_where_the_corners_live(tmp_path: Path) -> None:
    """元数据必须记下"离线分析 ROI 是哪一块、偏移多少、角点是哪套坐标"。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.analysis_roi = list(ROI_BOARD)
    config.validate()
    session, _recorder = open_session(config, run_kind="roi_meta")
    try:
        session.capture_static(segment_id="static_meta", duration_s=0.4)
        for metadata in _segments(session):
            assert metadata["analysis_roi"] == ROI_BOARD
            assert metadata["roi_origin"] == ROI_BOARD[:2], (
                "ROI 偏移没记下来：后来的人没法把 ROI 局部坐标换算回原图"
            )
            assert metadata["corners_frame"] == "full_image", (
                "角点坐标属于哪套坐标系必须写死在元数据里，"
                "否则同一批数据里两段坐标口径不同都看不出来"
            )
            assert metadata["source_size"] == [
                config.dry_run.width,
                config.dry_run.height,
            ]
    finally:
        session.close()


# --------------------------------------------------------------------------
# 2) 界面预览：整幅 + ROI 框，且不得改变 RAW
# --------------------------------------------------------------------------


def test_the_preview_shows_the_full_frame_with_the_roi_box(tmp_path: Path) -> None:
    """预览图是**整幅**画面（和 RAW 一致），ROI 只是画在上面的红框。

    这一条正是需求一·2 的原话："预览可以显示整幅画面并在上面画 ROI 框，
    但不得改变 RAW"。所以断言两件事：

    * 预览 PNG 的实际像素尺寸 = **全屏**尺寸（不是 ROI 尺寸）；
    * 框确实画上去了（框线上的像素是红的）。
    """
    import numpy as np
    from PIL import Image

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.analysis_roi = list(ROI_BOARD)
    config.validate()
    session, _recorder = open_session(config, run_kind="roi_preview")
    try:
        lines = session.require_roi_and_board("静态基线采集")
        text = "\n".join(lines)
        full = (config.dry_run.width, config.dry_run.height)
        assert f"原始帧尺寸：{full[0]}×{full[1]} px" in text, text
        assert f"RAW 实际保存尺寸：{full[0]}×{full[1]} px" in text, text
        assert f"离线分析 ROI = {ROI_BOARD}" in text, text

        assert session.roi_check_previews, "ROI 检查没有落下任何预览图"
        preview = session.roi_check_previews[0]
        assert _png_size(preview) == full, (
            f"预览图是 {_png_size(preview)}，不是全屏 {full}——"
            "预览被裁成分析窗口了，人会把它当成 RAW 的样子"
        )

        image = np.asarray(Image.open(preview).convert("RGB"))
        x, y, w, h = ROI_BOARD
        # 框线是纯红（BGR(0,0,255) → RGB(255,0,0)），灰度底图上不会自然出现。
        edge = image[y, x : x + w]
        assert (edge == np.array([255, 0, 0])).all(axis=1).any(), (
            "分析 ROI 的红框没画在预览图上，人看不出离线分析只看哪一块"
        )
        # 框**外**必须还是原样的画面（灰度 → R=G=B），不能被涂改。
        outside = image[: y - 4, : x - 4]
        assert (outside[:, :, 0] == outside[:, :, 1]).all() and (
            outside[:, :, 1] == outside[:, :, 2]
        ).all(), "ROI 框以外的像素被改过了：预览不只是一张示意图，它得是真画面"
    finally:
        session.close()


# --------------------------------------------------------------------------
# 3) 离线处理确实按 analysis_roi 走，且角点是原图坐标
# --------------------------------------------------------------------------


def test_the_analysis_window_is_the_roi_and_corners_stay_in_full_image_coordinates(
    tmp_path: Path,
) -> None:
    """★ 需求一·2 + 三·「分析结果无法复算」：ROI 生效，但坐标口径不变。

    做法：同一片静止画面采两段，一段不配 ROI（整幅识别）、一段配 248×200 的
    分析 ROI，两段都真跑一次 ``process_segment``，然后比**角点坐标**。

    * 若 ROI 没生效（分析还看整幅），两段结果一样 ⇒ 这条测不出差别，但下面的
      ``vision.json`` 里的 ``analysis_roi`` 会暴露"配了却没生效"；
    * 若 ROI 生效但坐标**没加回偏移**，两段角点会整整差一个 ``roi_origin``
      （几十像素）——这正是"坐标口径不一致"最危险的样子，因为位移、转角
      都是拿角点算的，差一个常量偏移会让每一段的位移都凭空多出一个假值。

    所以断言：两段角点逐点相差 < 1 px（同一片静止画面 + 传感器噪声），
    并且角点确实落在 ROI 所覆盖的原图范围里。
    """
    import numpy as np

    from sj_pretest.vision import load_segment_vision, process_segment

    def capture_and_process(tag: str, roi: list[int] | None):
        config = build_config(
            tmp_path / tag, joints=("J1",), amplitudes=(0.2,), repeats=1
        )
        config.camera.analysis_roi = roi
        config.validate()
        session, _recorder = open_session(config, run_kind=f"roi_{tag}")
        try:
            record = session.capture_static(segment_id="static_vision", duration_s=0.4)
            assert session.run is not None
            corners = session.run.root / "corners"
            vision = process_segment(
                Path(record.dir),
                config=config,
                segment_id="static_vision",
                stride=1,
                save_corners=True,
                corners_dir=corners,
                vision_json_path=Path(record.dir) / "segment_vision.json",
                raw_available_after=False,
            )
            rows = _read_corners_csv(corners / "static_vision.csv")
            points = np.array(
                [
                    [float(row["x_px"]), float(row["y_px"])]
                    for row in rows
                    if str(row.get("valid", "1")) not in ("0", "False", "false")
                ],
                dtype=float,
            )
            saved = load_segment_vision(
                Path(record.dir), segment_id="static_vision"
            )
            return points, vision, saved
        finally:
            session.close()

    full_points, full_vision, full_saved = capture_and_process("full", None)
    roi_points, roi_vision, roi_saved = capture_and_process("window", list(ROI_BOARD))

    assert full_points.size and roi_points.size, "有一段一个角点都没读到"

    # (a) ROI 真的被离线识别用上了：vision 结果里记着它，偏移也对。
    assert full_vision.analysis_roi is None, "没配 ROI，却记成了有"
    assert roi_vision.analysis_roi == tuple(ROI_BOARD)
    assert roi_vision.roi_origin == (ROI_BOARD[0], ROI_BOARD[1])
    assert roi_saved.analysis_roi == tuple(ROI_BOARD)
    assert roi_saved.roi_origin == (ROI_BOARD[0], ROI_BOARD[1])
    # 两种口径写进去的都是"原图坐标"这件事本身。
    assert full_saved.corners_frame == roi_saved.corners_frame == "full_image"

    # (b) 角点是**原图坐标**：两段的角点必须基本重合。
    #     如果 roi_origin 没加回去，差值会正好等于 ROI 左上角 (48, 32)。
    assert full_points.shape[1:] == roi_points.shape[1:], (
        f"两段读到的角点数不一样：整幅 {full_points.shape}，"
        f"分析窗口 {roi_points.shape}——ROI 不该改变能读出多少个点"
    )
    deviation = np.abs(full_points - roi_points)
    # 逐点差值里有**传感器噪声**：两段取自不同帧（合成世界每帧都带噪声，真机也一样），
    # 亚像素细化会把噪声变成一点几像素的抖动，所以逐点差不能当作"偏移"的证据。
    shift = roi_points.mean(axis=0) - full_points.mean(axis=0)
    assert np.abs(shift).max() <= 0.5, (
        f"配 ROI 前后角点**均值**偏了 {shift} px（典型症状是整整偏一个 ROI 偏移量"
        f" {(ROI_BOARD[0], ROI_BOARD[1])}）——说明 ROI 生效了，但角点坐标没有加回偏移，"
        "位移与转角会凭空多出一个常量假值。噪声会让单点抖动，但不会让均值偏移。"
    )
    assert deviation.max() <= 2.0, (
        f"逐点差最大 {deviation.max():.2f} px，超过传感器噪声能解释的范围——"
        "除了噪声之外还有别的东西在动这些坐标"
    )

    # (c) 角点确实落在 ROI 覆盖的那块原图范围里（确实是原图坐标，不是 ROI 局部坐标）。
    x, y, w, h = ROI_BOARD
    assert roi_points[:, 0].min() >= x and roi_points[:, 0].max() <= x + w
    assert roi_points[:, 1].min() >= y and roi_points[:, 1].max() <= y + h


# --------------------------------------------------------------------------
# 4) 分析 ROI 越界：不许开始
# --------------------------------------------------------------------------


def test_an_out_of_range_roi_blocks_before_any_capture(tmp_path: Path) -> None:
    """分析 ROI 越界：**在采集之前**就拒绝，不浪费一段的磁盘和时间。

    为什么越界要拦在采集之前而不是等离线处理时再报：RAW 全屏之后，采集本身
    跟 ROI 没关了，一段照样能采下来、照样占几 GB。等处理时才失败，等于白采一趟，
    而且那时人已经离开工位了。越界是**当场就能算出来**的事，所以当场拦。

    这一条同时反向证明了"ROI 不影响 RAW"：拦住的是**分析窗口**，
    报错信息里说的是 RAW 是全幅、被卡的是离线分析窗口。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.analysis_roi = [300, 200, 200, 160]  # x+w=500 > 340，y+h=360 > 260
    config.validate()
    session, recorder = open_session(config, run_kind="roi_bad")
    assert session.run is not None

    with pytest.raises(ExperimentError) as excinfo:
        session.require_roi_and_board("静态基线采集")
    message = str(excinfo.value)
    assert "静态基线采集没有开始" in message
    assert "ROI" in message
    assert "越界" in message
    assert "[ROI]" in recorder.text(), "越界原因没有打到界面上"

    # 事件流里要如实记下"这次检查没过"。
    checked = [
        item
        for item in session.run.events.read_all()
        if item.get("event") == "roi_checked"
    ]
    assert checked and checked[-1]["ok"] is False

    # 越界之后一段都不许采：段目录里什么都没有。
    assert _segments(session) == []
    assert session.trials == []


def test_the_check_runs_on_the_analysis_window_not_the_full_one(tmp_path: Path) -> None:
    """完整性检查必须看**分析窗口**——离线识别真正会看的那一块。

    做法：把分析 ROI 裁到只剩棋盘格的一角。整幅画面里棋盘格是完整的，
    分析窗口里必然不完整；检查如果还在看整幅就会误判"通过"，然后跑起来
    才发现识别不到角点——那时 RAW 已经采完、盘已经花掉了。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.analysis_roi = [0, 0, 60, 50]  # 只包住棋盘格左上角的一小块
    config.validate()
    session, _recorder = open_session(config, run_kind="roi_partial")

    with pytest.raises(ExperimentError) as excinfo:
        session.require_roi_and_board("静态基线采集")
    message = str(excinfo.value)
    assert "棋盘格" in message, (
        "分析窗口里只剩一角，检查却说通过了——它看的不是分析窗口"
    )
    assert "内角点" in message or "没有检出" in message, message
    assert "没有开始" in message

    # 反向对照：同一套配置把分析窗口放成全幅就必须通过——
    # 证明上面那次失败是 ROI 造成的，不是棋盘格本身没摆好。
    session.close()
    full = build_config(tmp_path / "full", joints=("J1",), amplitudes=(0.2,), repeats=1)
    full.camera.analysis_roi = None
    full.validate()
    ok_session, _ = open_session(full, run_kind="roi_full")
    try:
        lines = ok_session.require_roi_and_board("静态基线采集")
        assert any("通过" in line for line in lines), lines
    finally:
        ok_session.close()


def test_the_legacy_roi_field_only_moves_the_analysis_window(tmp_path: Path) -> None:
    """v1.0.2 的旧字段 ``camera.roi`` 现在只是 ``analysis_roi`` 的兼容写法。

    旧配置文件不该因为升版而读不进来，但也**不许**把旧行为（裁 RAW）带回来。
    所以：旧字段能让检查按那一块画面判，RAW 却仍然是整幅。
    """
    from conftest import read_json as _read

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.roi = list(ROI_BOARD)
    config.camera.analysis_roi = None
    config.validate()

    # 兜底也生效：迁移之后两处读到的是同一块。
    assert config.camera.resolved_analysis_roi() == tuple(ROI_BOARD)
    assert "camera.roi" in config.camera.roi_migration_note()

    session, _recorder = open_session(config, run_kind="roi_legacy")
    try:
        record = session.capture_static(segment_id="static_legacy", duration_s=0.4)
        metadata = _read(Path(record.dir) / "capture_metadata.json")
        assert metadata["raw_size"] == [
            config.dry_run.width,
            config.dry_run.height,
        ], "旧字段把 RAW 又裁小了——v1.0.2 的行为被带回来了"
        assert metadata["analysis_roi"] == ROI_BOARD, (
            "旧字段没有迁移成分析 ROI，检查和分析会各看各的"
        )
    finally:
        session.close()


def test_the_two_roi_fields_disagreeing_is_an_error(tmp_path: Path) -> None:
    """两个 ROI 字段都写又不一致：**报错**，不悄悄挑一个。

    现场最有价值的排查线索就是"我以为它按 roi 分析，其实按 analysis_roi 分析"
    这种不一致，所以宁可拒绝也不猜。
    """
    from sj_pretest.config import ConfigError

    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.analysis_roi = list(ROI_BOARD)
    config.camera.roi = [0, 0, 100, 80]
    with pytest.raises(ConfigError) as excinfo:
        config.validate()
    message = str(excinfo.value)
    assert "camera.roi" in message and "analysis_roi" in message
    assert "不一致" in message
