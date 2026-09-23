"""ROI 只有一份：界面预览 = 检查用的图 = RAW 里真正存下的图。

为什么这条要单独测
------------------
1.0.0 里有**三套**互不相干的裁剪：采集引擎裁一份写进 RAW，到位预览用的是
**未裁剪**的整幅图，棋盘格完整性检查又用另一份（而且 vendored 的 ``VISION_ROI``
默认是 None）。后果现场才看得出来：界面上棋盘格完整、离线识别却在裁过的图里
找不齐角点；或者预览里没越界、RAW 却因为 ROI 写大了而写不出来。

需求三要求：到位预览、棋盘格完整性检查、余量检查、快速几何检查、RAW 落盘
**全部**用同一份 ROI；界面要显示「原始画面尺寸 / ROI 坐标 / 裁剪后尺寸」和
真实的裁剪后预览；ROI 越界或裁剪后棋盘格不完整时**不许开始**。

这里测的是可核对的事实：RAW 的字节数、``capture_metadata.json`` 里的三个尺寸、
预览 PNG 的实际像素尺寸、以及越界时的拒绝。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sj_pretest.experiment import ExperimentError

from conftest import build_config, open_session, read_json

#: 一块**裁得进合成画面、又完整包住棋盘格**的 ROI。
#: 自测画面是 340×260，合成世界里棋盘格的内角点落在 x∈[89.3, 249.4]、
#: y∈[72.6, 184.6]（实测），而检查要求内角点到裁剪边缘至少留 40 px
#: （``camera.min_margin_px`` 的交付默认值，一个字都没改）。
#: 所以这块 ROI 的四条边余量分别是 41.3 / 46.6 / 40.6 / 47.4 px —— 刚刚够，
#: 也就意味着"检查确实是在裁剪后的图上算余量"这件事真的被走到了。
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


def test_preview_and_raw_use_the_same_roi(tmp_path: Path) -> None:
    """预览 PNG、检查结论和 RAW 的字节数必须指向**同一块**画面。"""
    roi = list(ROI_BOARD)
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.roi = list(roi)
    config.validate()
    session, _recorder = open_session(config, run_kind="roi")

    # 按钮二第一件事就是这次检查：它会捞一帧 → 报三个尺寸 → 落一张裁剪预览。
    lines = session.require_roi_and_board("静态基线采集")
    text = "\n".join(lines)
    assert f"原始画面尺寸：{config.dry_run.width}×{config.dry_run.height} px" in text
    assert f"ROI = {roi}" in text
    assert f"裁剪后尺寸：{roi[2]}×{roi[3]} px" in text

    assert session.roi_check_previews, "ROI 检查没有落下任何裁剪预览图"
    for path in session.roi_check_previews:
        assert _png_size(path) == (roi[2], roi[3]), (
            f"{Path(path).name} 的尺寸和 ROI 裁剪尺寸不一致"
        )

    record = session.capture_static(segment_id="static_roi", duration_s=0.4)
    metadata = read_json(Path(record.dir) / "capture_metadata.json")
    assert metadata["roi"] == roi
    assert metadata["source_size"] == [config.dry_run.width, config.dry_run.height]
    assert metadata["cropped_size"] == [roi[2], roi[3]]
    assert int(metadata["frame_bytes"]) == roi[2] * roi[3], (
        "元数据里记的每帧字节数不是裁剪后的尺寸"
    )
    raw = Path(record.dir) / "frames.raw"
    assert raw.stat().st_size == int(metadata["frame_count"]) * roi[2] * roi[3], (
        "RAW 的实际字节数和'裁剪后尺寸 × 帧数'对不上——"
        "说明写进盘里的不是裁剪后的图，或者尺寸记错了"
    )

    # 界面看到的那张预览图，和 RAW 里存下的第一帧，必须是**落在同一个窗口里的
    # 同一块画面**。这里不要求逐像素相等：合成世界（和真机相机一样）每帧都带
    # 传感器噪声，而这两张图取自不同帧——逐像素相等本来就不该成立。
    # 能要求、也真正说明问题的是：棋盘格的 88 个内角点在这两张图里的位置一致。
    # 如果预览用了未裁剪的整幅图（1.0.0 的错误），角点坐标会差几十个像素。
    import numpy as np

    first = np.fromfile(raw, dtype=np.uint8, count=roi[2] * roi[3]).reshape(
        roi[3], roi[2]
    )
    from PIL import Image  # Pillow 是 requirements.txt 里的依赖，不是可选项

    preview = np.asarray(Image.open(session.roi_check_previews[0]).convert("L"))
    assert preview.shape == first.shape, (
        f"预览图 {preview.shape} 和 RAW 每帧的尺寸 {first.shape} 不一致"
    )

    from sj_pretest.vendor_shim import vendor_module

    camera = vendor_module("camera")
    tracker = camera.CheckerboardTracker()

    def corners_of(image):
        gray, _metrics, _origin, _timing = camera.preprocess_frame(image)
        _checker, found, _timing = tracker.process(gray)
        assert found is not None, "这张图里没有检出棋盘格"
        return np.asarray(found, dtype=float).reshape(-1, 2)

    from_preview = corners_of(preview)
    from_raw = corners_of(first)
    assert from_preview.shape == from_raw.shape
    assert len(from_raw) == 88, f"内角点数不对：{len(from_raw)}"
    deviation = np.abs(from_preview - from_raw)
    assert deviation.max() <= 1.0, (
        f"预览图和 RAW 里的棋盘格位置差最多 {deviation.max():.2f} px——"
        "两张图不是同一块画面（预览是不是用了未裁剪的整幅图？）"
    )


def test_an_out_of_range_roi_blocks_every_capture(tmp_path: Path) -> None:
    """ROI 越界：**不许开始**，且不许留下"看起来跑过"的痕迹。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.roi = [300, 200, 200, 160]  # x+w=500 > 340，y+h=360 > 260
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

    # 越界的 ROI 连预览都给不出来，所以一张图都不该落。
    assert session.roi_check_previews == []
    assert not list((session.run.root / "roi_checks").glob("*.png"))

    # 事件流里也要如实记下"这次检查没过"。
    checked = [
        item
        for item in session.run.events.read_all()
        if item.get("event") == "roi_checked"
    ]
    assert checked and checked[-1]["ok"] is False


def test_the_check_runs_on_the_cropped_frame_not_the_full_one(tmp_path: Path) -> None:
    """完整性检查必须看**裁剪后**的图——这是 1.0.0 出错的地方。

    做法：把 ROI 裁到只剩棋盘格的一角。整幅图里棋盘格是完整的，
    裁剪之后必然不完整；检查如果还在看整幅图就会误判"通过"。
    """
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.roi = [0, 0, 60, 50]  # 只包住棋盘格左上角的一小块
    config.validate()
    session, _recorder = open_session(config, run_kind="roi_partial")

    with pytest.raises(ExperimentError) as excinfo:
        session.require_roi_and_board("静态基线采集")
    message = str(excinfo.value)
    assert "棋盘格" in message, (
        "裁剪后只剩一角，检查却说通过了——它看的不是裁剪后的图"
    )
    assert "内角点" in message or "没有检出" in message, message
    assert "没有开始" in message

    # 反向对照：同一套配置把 ROI 放成全幅就必须通过——
    # 证明上面那次失败是 ROI 造成的，不是棋盘格本身没摆好。
    session.close()
    full = build_config(tmp_path / "full", joints=("J1",), amplitudes=(0.2,), repeats=1)
    full.camera.roi = None
    full.validate()
    ok_session, _ = open_session(full, run_kind="roi_full")
    try:
        lines = ok_session.require_roi_and_board("静态基线采集")
        assert any("通过" in line for line in lines), lines
    finally:
        ok_session.close()


def test_segments_record_the_roi_they_actually_used(tmp_path: Path) -> None:
    """每一段的元数据都要能回答"这段用的哪份 ROI、原始多大、裁完多大"。"""
    config = build_config(tmp_path, joints=("J1",), amplitudes=(0.2,), repeats=1)
    config.camera.roi = list(ROI_BOARD)
    config.validate()
    session, _recorder = open_session(config, run_kind="roi_meta")
    try:
        session.capture_static(segment_id="static_meta", duration_s=0.4)
        for metadata in _segments(session):
            assert metadata["roi"] == ROI_BOARD
            assert metadata["cropped_size"] == ROI_BOARD[2:]
            assert metadata["source_size"] == [
                config.dry_run.width,
                config.dry_run.height,
            ]
    finally:
        session.close()
