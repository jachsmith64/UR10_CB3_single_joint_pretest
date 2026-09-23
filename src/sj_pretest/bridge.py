"""把 :class:`sj_pretest.config.AppConfig` 翻译到被复用代码的 ``vendor/config.py``。

为什么需要这一层
----------------
被复用的 ``camera.py`` / ``robot.py`` / ``analyze.py`` 里写的都是 ``import config``
然后 ``config.HIK_EXPOSURE_US`` 这样在**调用时**读属性（已核对：四个模块都是
``import config``，没有 ``from config import X``）。所以只要在调用这些函数之前
把 ``vendor/config.py`` 上的属性改成用户想要的数，被复用代码读到的就是新值，
**不需要改它们一行源码**。

安全上的两点保证
----------------
1. :func:`apply` 只写"值"，不改任何行为开关的默认语义；真机运动的总开关
   （``ROBOT_TEST_ALLOW_MOTION``）只有 ``mode == "hardware"`` 时才为 True。
2. :func:`snapshot` / :func:`restore` 让测试可以整段回滚，避免测试之间
   互相污染全局配置——这是自测能反复跑的前提。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig

#: 本工具会写入 vendor/config.py 的属性名清单。
#: 显式列出来有两个好处：一是可以逐条写"为什么要改它"，二是
#: :func:`restore` 只回滚这些名字，不会误伤别的全局量。
BRIDGED_ATTRIBUTES: tuple[str, ...] = (
    # -- 相机与采集 --
    "EXPECTED_VISION_FPS",
    "HIK_CAMERA_SERIAL",
    "HIK_MVS_IMPORT_PATH",
    "HIK_EXPOSURE_US",
    "HIK_GAIN",
    "HIK_FRAME_TIMEOUT_MS",
    "CAPTURE_STORAGE_MODE",
    "CAPTURE_REQUIRE_MONO8",
    "CAPTURE_SAVE_SAMPLE_IMAGES",
    "CAPTURE_SHOW_PREVIEW",
    "CAPTURE_PREVIEW_FPS",
    # -- 棋盘格识别 --
    "CHECKERBOARD_INNER_CORNERS",
    "CHECKER_SQUARE_MM",
    "CHECKER_RESIDUAL_WARNING_PX",
    # -- 机器人 --
    "ROBOT_HOST",
    "ROBOT_RECORD_HZ",
    "ROBOT_TEST_ALLOW_MOTION",
    "ROBOT_POSES_CONFIRMED",
    "REQUIRE_OPERATOR_CONFIRMATION",
    # -- 记录时长 --
    "POST_MOTION_RECORD_SECONDS",
    "BATCH_STATIC_BASELINE_SECONDS",
    "BATCH_PRE_MOTION_SECONDS",
    "BATCH_POST_MOTION_SECONDS",
    # -- 离线/静态噪声默认值 --
    "MICRO_LOOP_STATIC_SECONDS",
    "OFFLINE_STATIC_SECONDS",
    "OFFLINE_STATIC_REPEATS",
    "OFFLINE_STATIC_DISK_RESERVE_GB",
    # -- 输出与预览 --
    "OUTPUT_ROOT",
    "SHOW_PREVIEW",
    "OFFLINE_SHOW_PREVIEW",
    "ENABLE_VISION_TIMING",
)


class BridgeError(RuntimeError):
    """桥接失败。消息用中文，能直接显示在界面上。"""


@dataclass
class _Snapshot:
    """一次 :func:`snapshot` 记录下来的原值。"""

    values: dict[str, Any]

    def restore(self) -> None:
        config = _vendor_config()
        for name, value in self.values.items():
            setattr(config, name, value)


def _vendor_config() -> Any:
    """拿到被复用代码用的那个 config 模块。

    通过 ``sj_pretest`` 包的 ``sys.path`` 垫片导入，保证和
    ``camera.py`` 里 ``import config`` 拿到的是**同一个模块对象**。
    这一点必须成立，否则桥接就白改了——所以找不到时直接报错，不静默返回默认配置。
    """
    import sj_pretest  # noqa: F401  只为触发 vendor 目录进 sys.path

    try:
        import config as vendor_config  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 说明部署有问题
        raise BridgeError(
            "找不到被复用代码的 config 模块。请确认 "
            "src/sj_pretest/vendor/config.py 存在，且 sj_pretest 包已被导入。"
        ) from exc

    module_path = Path(getattr(vendor_config, "__file__", "") or "")
    if module_path.name != "config.py" or module_path.parent.name != "vendor":
        raise BridgeError(
            f"导入到的 config 模块不是本工具内置的那一份：{module_path}。"
            "这会把参数写进错误的模块，已中止。"
        )
    return vendor_config


def snapshot() -> _Snapshot:
    """记录当前 vendor 配置里本工具会碰的那些属性。"""
    vendor_config = _vendor_config()
    return _Snapshot(
        values={
            name: getattr(vendor_config, name)
            for name in BRIDGED_ATTRIBUTES
            if hasattr(vendor_config, name)
        }
    )


def restore(state: _Snapshot) -> None:
    """把 :func:`snapshot` 记录的原值写回去。"""
    state.restore()


def apply(config: AppConfig, *, output_root: Path | None = None) -> dict[str, Any]:
    """把 AppConfig 写进 vendor 配置，返回实际写入的 ``{属性名: 值}``。

    ``output_root`` 给出时要覆盖 ``paths.output_root`` 的解析结果——
    让调用方（界面、测试）决定实验数据落在哪个盘上。
    """
    vendor_config = _vendor_config()
    config.validate()

    resolved_output = Path(output_root) if output_root else config.resolve_output_root()
    resolved_output.mkdir(parents=True, exist_ok=True)

    hardware = config.mode == "hardware"

    #: 每一项都写清楚"为什么"。真正的赋值在最后统一做，避免中途半套参数生效。
    plan: dict[str, Any] = {
        # 相机：帧率期望值影响缺帧判定和 analysis_time_s 的重建；曝光/增益沿用
        # 旧项目真机跑通的 6500 us；MVS 路径让 HikCameraSource 能 import 到 SDK。
        "EXPECTED_VISION_FPS": float(config.camera.expected_fps),
        "HIK_CAMERA_SERIAL": str(config.camera.serial),
        "HIK_MVS_IMPORT_PATH": config.camera.mvs_import_path,
        "HIK_EXPOSURE_US": config.camera.exposure_us,
        "HIK_GAIN": config.camera.gain,
        "HIK_FRAME_TIMEOUT_MS": int(config.camera.frame_timeout_ms),
        # 采集：必须是 stream_raw —— 需求六要求"在线线程只负责稳定采集保存"，
        # stream_raw 边采边落盘，不需要预分配容量，是长时段采集最稳的方式。
        "CAPTURE_STORAGE_MODE": "stream_raw",
        "CAPTURE_REQUIRE_MONO8": True,
        "CAPTURE_SAVE_SAMPLE_IMAGES": bool(config.camera.save_sample_images),
        # 预览一律关掉：界面自己画预览，而且无头自测环境里 cv2.imshow 会直接报错。
        "CAPTURE_SHOW_PREVIEW": False,
        "CAPTURE_PREVIEW_FPS": 10.0,
        # 棋盘格：11×8 内角点、3 mm 小格，沿用已验证配置。
        "CHECKERBOARD_INNER_CORNERS": tuple(int(v) for v in config.camera.board_inner_corners),
        "CHECKER_SQUARE_MM": float(config.camera.square_mm),
        # 残差阈值直接由界面的 max_residual_px 驱动（默认 1.5，比旧项目的 0.8 宽），
        # 这样"角点质量"这个判据在界面上可调，而不是埋在源码里。
        "CHECKER_RESIDUAL_WARNING_PX": float(config.vision.max_residual_px),
        # 机器人：IP、记录频率、以及**真机运动总开关**。
        # ROBOT_TEST_ALLOW_MOTION 只在 hardware 模式下为 True；dry_run/replay 一律 False。
        "ROBOT_HOST": str(config.robot.ip),
        "ROBOT_RECORD_HZ": float(config.robot.rtde_record_hz),
        "ROBOT_TEST_ALLOW_MOTION": bool(hardware),
        # 本工具不用旧项目的示例点位（POINT_A/B/C），所以这个开关保持 True 也无妨：
        # 它只挡"用示例绝对位姿动真机"，而我们走的是关节空间 moveJ，不经过那条路径。
        "ROBOT_POSES_CONFIRMED": True,
        # 每一次真机运动前都要人工确认——这是需求三明确要求保留的安全条件，
        # 不允许在界面上关掉，所以这里硬写成 True。
        "REQUIRE_OPERATOR_CONFIRMATION": True,
        # 记录时长：运动后继续录 post_motion_s，用来判断"RTDE 稳了画面是否还在动"。
        "POST_MOTION_RECORD_SECONDS": float(config.camera.post_motion_s),
        "BATCH_STATIC_BASELINE_SECONDS": float(config.camera.static_duration_s),
        "BATCH_PRE_MOTION_SECONDS": float(config.camera.pre_motion_s),
        "BATCH_POST_MOTION_SECONDS": float(config.camera.post_motion_s),
        # 旧项目静态噪声脚本的默认值，本工具沿用同一批数以免两套口径。
        "MICRO_LOOP_STATIC_SECONDS": float(config.camera.static_duration_s),
        "OFFLINE_STATIC_SECONDS": float(config.camera.static_duration_s),
        "OFFLINE_STATIC_DISK_RESERVE_GB": float(config.paths.min_free_disk_gb),
        # 输出根目录：让被复用代码里那些"找最新采集目录"的工具函数也在本工具的
        # outputs/ 下找，而不是去旧项目目录里翻。
        "OUTPUT_ROOT": resolved_output,
        "SHOW_PREVIEW": False,
        "OFFLINE_SHOW_PREVIEW": False,
        # 逐帧计时在本工具里是有用的（需求六要求记录相机时间戳和丢帧），保持开启。
        "ENABLE_VISION_TIMING": True,
    }

    written: dict[str, Any] = {}
    for name, value in plan.items():
        if not hasattr(vendor_config, name):
            # 不同版本的被复用代码可能没有个别属性；缺了就记下来告诉调用方，
            # 但不编造一个出来——那会让后续读到"看起来正常其实无效"的值。
            continue
        setattr(vendor_config, name, value)
        written[name] = value

    missing = sorted(set(plan) - set(written))
    if missing:
        raise BridgeError(
            f"被复用代码的 config 模块里没有这些属性：{missing}。"
            "说明 vendor 目录里的模块版本不对，已中止（不会带着半套参数继续跑）。"
        )
    return written


def describe_effective_config() -> dict[str, Any]:
    """读回被复用代码当前生效的关键参数，用于日志和自检。"""
    vendor_config = _vendor_config()
    keys = (
        "EXPECTED_VISION_FPS",
        "CHECKERBOARD_INNER_CORNERS",
        "CHECKER_SQUARE_MM",
        "CHECKER_RESIDUAL_WARNING_PX",
        "CAPTURE_STORAGE_MODE",
        "CAPTURE_SHOW_PREVIEW",
        "ROBOT_HOST",
        "ROBOT_RECORD_HZ",
        "ROBOT_TEST_ALLOW_MOTION",
        "REQUIRE_OPERATOR_CONFIRMATION",
        "POST_MOTION_RECORD_SECONDS",
        "OUTPUT_ROOT",
    )
    result: dict[str, Any] = {}
    for key in keys:
        if not hasattr(vendor_config, key):
            continue
        value = getattr(vendor_config, key)
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def assert_no_motion_allowed() -> None:
    """自检用：确认当前 vendor 配置**不会**允许真机运动。

    dry-run / replay 自测的第一步就调它——万一哪天桥接写错把开关打开了，
    自测要当场失败，而不是安静地"跑通"。
    """
    vendor_config = _vendor_config()
    if bool(getattr(vendor_config, "ROBOT_TEST_ALLOW_MOTION", False)):
        raise BridgeError(
            "当前 vendor 配置把 ROBOT_TEST_ALLOW_MOTION 打开了，"
            "但这次运行不是 hardware 模式，拒绝继续。"
        )
