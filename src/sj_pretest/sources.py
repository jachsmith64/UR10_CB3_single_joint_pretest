"""图像来源：真机相机、合成世界、历史数据回放，三种来源给出同一种帧包。

为什么要统一
------------
需求八要求"干运行能跑完整流程""回放能处理已有数据"。如果给干运行单写一条
"假的"采集通道，那自测就只证明了假通道能跑，证明不了真通道的流程对。
所以三种来源都实现同一个 :class:`ImageSource` 协议，产出被复用代码定义的
``FramePacket``，后面的采集、落盘、离线角点、分析全都不需要知道帧是从哪来的。

合成世界（dry_run）
-------------------
:class:`SyntheticWorld` 会渲染真的棋盘格图像（12×9 方格 → 11×8 内角点 → 88 个角点），
再按"当前关节角"做仿射变化。**所以离线角点检测是在真图上跑的**，不是打桩返回数字。
``SyntheticWorld`` 只是替代了"机械臂 + 相机"这一层物理，不替代任何算法。

**合成世界的输出一律标记 synthetic**，它验证的是流程和判据，不能当作实验结果。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import numpy as np

from .config import JOINT_NAMES, AppConfig, DryRunConfig


class SourceError(RuntimeError):
    """图像来源不可用。消息中文，直接显示给实验者。"""


class ImageSource(Protocol):
    """三种来源共用的最小接口。"""

    actual_camera_fps: float | None
    actual_camera_fps_source: str

    def __enter__(self) -> "ImageSource": ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    def __iter__(self) -> Iterator[Any]: ...


# --------------------------------------------------------------------------
# 真机相机：直接把被复用代码的 HikCameraSource 包一层
# --------------------------------------------------------------------------


def open_hardware_source() -> ImageSource:
    """打开海康相机。MVS SDK 或相机不在时抛中文错误，不会静默退回合成。"""
    from .vendor_shim import vendor_module

    camera = vendor_module("camera")
    source = camera.HikCameraSource()
    try:
        source.__enter__()
    except Exception as exc:
        raise SourceError(
            "打开海康相机失败："
            f"{type(exc).__name__}: {exc}\n"
            "请检查：1) 相机是否上电并插在同一台电脑上；"
            "2) MVS 客户端能否看到相机；"
            "3) config 里的 camera.mvs_import_path 是否指向 MVS 的 MvImport 目录；"
            "4) 串口号 camera.serial 是否选对了（留空表示用第一台）。"
        ) from exc
    return source


# --------------------------------------------------------------------------
# 合成世界（dry_run）
# --------------------------------------------------------------------------


@dataclass
class BoardRender:
    """合成世界在某一时刻给"相机"看到的棋盘格状态。"""

    offset_x_px: float
    offset_y_px: float
    rotation_deg: float
    scale: float


class SyntheticWorld:
    """把关节角变成"相机里看到的棋盘格位置"的合成世界。

    模型（全部写在这里，方便复核它到底假设了什么）：
    * 棋盘格在图像里的平移 = Σ_关节 灵敏度(px/°) × (该关节实际角 − 名义角) × 单位方向；
    * 图像转角 = 旋转灵敏度 × J6 的角偏差；
    * 关节本身对指令做一阶跟随（时间常数 ``joint_response_tau_s``），
      所以"指令已到、关节还在动"这件事在合成世界里也存在；
    * 每次下发新指令，叠加一个衰减振荡，模拟结构在运动后的余振——
      这正是需求七第三层（"RTDE 稳了但画面还在动"）需要被验证的那件事；
    * 叠一个按帧号播种的高斯抖动，模拟检测噪声。
    """

    def __init__(self, config: DryRunConfig, nominal_joint_deg: Sequence[float]) -> None:
        self.config = config
        self.nominal = np.asarray(nominal_joint_deg, dtype=np.float64)
        if self.nominal.shape != (6,):
            raise SourceError("名义姿态必须是 6 个角度。")
        gains = np.array(
            [float(config.centroid_px_per_deg[name]) for name in JOINT_NAMES],
            dtype=np.float64,
        )
        directions = np.array(
            [math.radians(float(config.image_direction_deg[name])) for name in JOINT_NAMES],
            dtype=np.float64,
        )
        #: 每个关节的单位位移向量（px/° 已经乘进增益里）。
        self.gain_x = gains * np.cos(directions)
        self.gain_y = gains * np.sin(directions)
        self.gain_rotation = float(config.rotation_deg_per_deg)
        #: 合成深度变化：用一个温和的尺度项，让"深度/面内比"这一判据有东西可算。
        self.gain_scale = 2.0e-4

        self.actual = self.nominal.copy()
        self.target = self.nominal.copy()
        self.command_time_s: float | None = None
        self._last_step_s: float | None = None
        self.vibration_time_s: float | None = None
        self.vibration_axis = np.array([1.0, 0.0])
        self._rng = np.random.default_rng(int(config.seed))

    # -- 被"机器人"调用 --------------------------------------------------

    def reset(self, joint_deg: Sequence[float]) -> None:
        """把世界退回某个姿态（连接、或重新开始时用）。"""
        q = np.asarray(joint_deg, dtype=np.float64)
        if q.shape != (6,):
            raise SourceError("复位姿态必须是 6 个角度。")
        self.actual = q.copy()
        self.target = q.copy()
        self.command_time_s = None
        self._last_step_s = None
        self.vibration_time_s = None

    def command(self, target_joint_deg: Sequence[float], now_s: float) -> None:
        """下发一个新的目标关节角（合成世界里的 moveJ）。"""
        target = np.asarray(target_joint_deg, dtype=np.float64)
        if target.shape != (6,):
            raise SourceError("目标姿态必须是 6 个角度。")
        self.target = target.copy()
        self.command_time_s = float(now_s)
        self._last_step_s = float(now_s)
        self.vibration_time_s = float(now_s)
        # 余振方向取这次运动的主方向，确定性播种，保证自测可重复。
        delta_x = float((target - self.nominal) @ self.gain_x)
        delta_y = float((target - self.nominal) @ self.gain_y)
        norm = math.hypot(delta_x, delta_y)
        self.vibration_axis = (
            np.array([delta_x / norm, delta_y / norm]) if norm > 1e-9 else np.array([1.0, 0.0])
        )

    def step(self, now_s: float) -> None:
        """把"关节实际角"按一阶跟随推进到当前时刻。

        关键：衰减是按**距上一次推进的时间差**算的，不是距下发指令的时间差。
        （后者会在每次推进时把残差再乘一遍同样的衰减，等于把时间常数人为缩小，
        于是"还没到位"这件事在自测里就看不到了——那样控速判据就是假的。）
        推进到同一时刻两次是幂等的，所以渲染和判据可以各自调用。
        """
        now = float(now_s)
        if self.command_time_s is None:
            self._last_step_s = now
            return
        tau = float(self.config.joint_response_tau_s)
        last = self._last_step_s
        if last is None:
            last = float(self.command_time_s)
        dt = max(now - last, 0.0)
        self._last_step_s = max(now, last)
        if tau <= 0:
            self.actual = self.target.copy()
            return
        alpha = math.exp(-dt / tau)
        self.actual = self.target + (self.actual - self.target) * alpha

    def is_settled(self, now_s: float, tolerance_deg: float) -> bool:
        """关节是否已经贴住目标（合成世界里的"到位"判据）。"""
        del now_s
        return bool(np.max(np.abs(self.actual - self.target)) <= tolerance_deg)

    # -- 被"相机"调用 ----------------------------------------------------

    def render_state(self, now_s: float, frame_id: int) -> BoardRender:
        """当前这一帧应该看到的棋盘格状态（含噪声）。"""
        self.step(now_s)
        delta = self.actual - self.nominal
        offset_x = float(delta @ self.gain_x)
        offset_y = float(delta @ self.gain_y)
        rotation = float(delta[5] * self.gain_rotation)
        scale = 1.0 + float(delta @ np.array([self.gain_scale] * 6))

        vib_x = vib_y = 0.0
        if self.vibration_time_s is not None:
            amplitude = float(self.config.vibration_amplitude_px)
            tau = float(self.config.vibration_tau_s)
            freq = float(self.config.vibration_hz)
            dt = max(float(now_s) - self.vibration_time_s, 0.0)
            envelope = amplitude * math.exp(-dt / tau)
            phase = math.cos(2.0 * math.pi * freq * dt)
            vib_x = envelope * phase * float(self.vibration_axis[0])
            vib_y = envelope * phase * float(self.vibration_axis[1])

        # 噪声按帧号播种：同一帧号永远是同一个抖动，自测可重复。
        noise = np.random.default_rng(
            (int(self.config.seed) * 1000003 + int(frame_id)) % (2**32)
        ).normal(
            0.0,
            [
                float(self.config.centroid_noise_px),
                float(self.config.centroid_noise_px),
                float(self.config.rotation_noise_deg),
            ],
        )
        return BoardRender(
            offset_x_px=offset_x + vib_x + float(noise[0]),
            offset_y_px=offset_y + vib_y + float(noise[1]),
            rotation_deg=rotation + float(noise[2]),
            scale=scale,
        )


def _build_board_image(config: DryRunConfig) -> np.ndarray:
    """画一张真的棋盘格图像：12×9 方格 → 11×8 内角点，带白边。"""
    columns, rows = 11, 8  # 内角点数
    square_px = 16
    board_w = (columns + 1) * square_px
    board_h = (rows + 1) * square_px
    board = np.full((board_h, board_w), 240, dtype=np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                y0 = row * square_px
                x0 = column * square_px
                board[y0 : y0 + square_px, x0 : x0 + square_px] = 25
    # 棋盘格外面再留一圈白边，免得检测算法把图像边界当成方格边界。
    border = 8
    padded = np.full(
        (board_h + 2 * border, board_w + 2 * border), 240, dtype=np.uint8
    )
    padded[border : border + board_h, border : border + board_w] = board

    frame_w = int(config.width)
    frame_h = int(config.height)
    if padded.shape[0] > frame_h or padded.shape[1] > frame_w:
        raise SourceError(
            f"合成棋盘格 {padded.shape[1]}×{padded.shape[0]} 比帧 "
            f"{frame_w}×{frame_h} 还大，请把 dry_run.width/height 调大。"
        )
    canvas = np.full((frame_h, frame_w), 128, dtype=np.uint8)
    y0 = (frame_h - padded.shape[0]) // 2
    x0 = (frame_w - padded.shape[1]) // 2
    canvas[y0 : y0 + padded.shape[0], x0 : x0 + padded.shape[1]] = padded
    return canvas


class SyntheticSource:
    """合成图像来源：按 ``dry_run`` 的参数出帧，帧包字段和真机完全一致。"""

    def __init__(
        self,
        config: AppConfig,
        world: SyntheticWorld,
        *,
        start_frame_id: int = 0,
        start_host_ns: int | None = None,
    ) -> None:
        self.dry = config.dry_run
        self.world = world
        self.master = _build_board_image(self.dry)
        self._center = (self.master.shape[1] / 2.0, self.master.shape[0] / 2.0)
        self.actual_camera_fps: float | None = float(self.dry.fps)
        self.actual_camera_fps_source = "dry_run.synthetic"
        self.frame_id = int(start_frame_id)
        self._period_ns = int(round(1_000_000_000 / float(self.dry.fps)))
        self._start_host_ns = (
            int(start_host_ns) if start_host_ns is not None else time.perf_counter_ns()
        )
        self._started = False
        self._wall_start_ns = 0
        self.frames_emitted = 0
        self.dropped = 0
        self._rng = np.random.default_rng(int(self.dry.seed) + 7)

    # -- 上下文管理 -------------------------------------------------------

    def __enter__(self) -> "SyntheticSource":
        self._wall_start_ns = time.perf_counter_ns()
        self._started = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._started = False

    # -- 帧生成 -----------------------------------------------------------

    def _host_ns_for(self, index: int) -> int:
        """第 index 帧的时间戳。

        虚拟时钟下时间戳完全由帧号决定（所以"132.23 fps"这件事被如实表达，
        而不用真的等 7.5 ms）；真实时间模式下取墙上时钟。
        """
        if self.dry.realtime:
            return time.perf_counter_ns()
        return self._start_host_ns + index * self._period_ns

    def _pace(self, index: int) -> None:
        if not self.dry.realtime:
            return
        target = self._wall_start_ns + index * self._period_ns
        remaining = target - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1_000_000_000)

    def __iter__(self) -> Iterator[Any]:
        if not self._started:
            raise SourceError("SyntheticSource 必须放在 with 语句中使用。")
        from .vendor_shim import vendor_module

        camera = vendor_module("camera")
        emitted = 0
        while True:
            self._pace(emitted)
            host_ns = self._host_ns_for(self.frame_id)
            now_s = (host_ns - self._start_host_ns) / 1_000_000_000.0

            # 注入丢帧：让 frame_id 跳号，用来验证缺帧记录链路。
            drop_every = int(self.dry.drop_every)
            if drop_every > 0 and emitted > 0 and emitted % drop_every == 0:
                self.frame_id += 1
                self.dropped += 1

            frame = self._render(now_s, self.frame_id)
            packet = camera.FramePacket(
                frame=frame,
                frame_id=int(self.frame_id),
                host_ns=int(host_ns),
                camera_timestamp_raw=int(host_ns - self._start_host_ns),
                source_name="dry_run:synthetic",
            )
            self.frame_id += 1
            emitted += 1
            self.frames_emitted = emitted
            yield packet

    def _render(self, now_s: float, frame_id: int) -> np.ndarray:
        import cv2

        state = self.world.render_state(now_s, frame_id)
        center_x, center_y = self._center
        # 注入角点丢失：把整帧压成均匀灰，检测必然失败。
        lost = False
        loss_every = int(self.dry.corner_loss_every)
        if loss_every > 0 and frame_id > 0 and frame_id % loss_every == 0:
            lost = True

        angle = math.radians(state.rotation_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        scale = state.scale
        matrix = np.array(
            [
                [scale * cos_a, -scale * sin_a, 0.0],
                [scale * sin_a, scale * cos_a, 0.0],
            ],
            dtype=np.float64,
        )
        # warpAffine 默认绕原点旋转，这里把它改成绕图像中心，并叠加平移。
        matrix[0, 2] = center_x - (matrix[0, 0] * center_x + matrix[0, 1] * center_y)
        matrix[1, 2] = center_y - (matrix[1, 0] * center_x + matrix[1, 1] * center_y)
        matrix[0, 2] += state.offset_x_px
        matrix[1, 2] += state.offset_y_px

        frame = cv2.warpAffine(
            self.master,
            matrix,
            (self.master.shape[1], self.master.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=128,
        )
        if lost:
            frame = np.full_like(frame, 128)
            return frame
        # 一点轻微模糊和传感器噪声，别让检测器吃到"过于干净"的理想图。
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
        noise = self._rng.normal(0.0, 2.5, size=frame.shape)
        return np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 历史数据回放
# --------------------------------------------------------------------------


@dataclass
class ReplayTarget:
    """回放源解析结果。"""

    kind: str  # "raw" / "image_folder" / "video"
    path: Path
    note: str


def resolve_replay_target(path: str | Path) -> ReplayTarget:
    """判断这份历史数据该用哪种来源读。判断错就直接报错，不猜。"""
    target = Path(path).expanduser()
    if not target.exists():
        raise SourceError(f"回放数据不存在：{target}")
    if target.is_dir():
        if (target / "frames.raw").is_file():
            return ReplayTarget("raw", target, "RAW 采集目录（frames.raw + frame_timestamps.csv）")
        images = sorted(
            child
            for child in target.iterdir()
            if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if images:
            return ReplayTarget(
                "image_folder", target, f"图像目录（{len(images)} 张）"
            )
        raise SourceError(
            f"目录里既没有 frames.raw 也没有图像文件：{target}\n"
            "回放需要：RAW 采集目录，或者一个装着逐帧图像的目录，或者一个视频文件。"
        )
    if target.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        return ReplayTarget("video", target, "视频文件")
    raise SourceError(f"不认识的回放数据：{target}")


def open_replay_source(path: str | Path, *, fps: float) -> tuple[ImageSource, ReplayTarget]:
    """按历史数据的形态打开对应的来源。"""
    from .vendor_shim import vendor_module

    camera = vendor_module("camera")
    target = resolve_replay_target(path)
    if target.kind == "raw":
        return camera.RawCaptureSource(target.path), target
    if target.kind == "image_folder":
        return camera.ImageFolderSource(target.path, float(fps)), target
    return camera.VideoSource(target.path), target


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------


class FramePump:
    """唯一的取帧入口：既给采集器供帧，也给"到位预览"抽帧。

    为什么需要它
    ------------
    到位过程中要不断抽最新的帧给操作者看（否则相机流会停），而这些帧又不该按
    采集目录落盘（一个到位步骤录 0.5 s 就要几十 MB，纯粹浪费）。所以"抽帧"和
    "采集"必须能共用同一条流。

    共用的危险在于**两套缓冲**：采集器遇到段边界时会把"下一段的头一帧"暂存起来，
    如果抽帧器也有自己的暂存，两边的帧就会插队。所以这里只留**一个**暂存位
    （:meth:`push_back`），采集器遇到它时就用它，不再自己存一份。
    """

    def __init__(self, source: Any) -> None:
        # ★ v1.0.3：**留下原始来源对象**。三种来源（真机 / 合成 / 回放）的
        # ``__iter__`` 都是生成器函数，所以 ``iter(来源)`` 拿到的是一个生成器——
        # 帧率（``actual_camera_fps``）和上下文管理（``__exit__``）都在**来源**
        # 身上，生成器上一个都没有。只留迭代器就会把这两样一起丢掉，
        # 而丢掉的后果不是"少个字段"，是采集层把**期望帧率**当成**实测帧率**用
        # （见下面两个属性的说明）。
        self._origin = source
        self.source = source if hasattr(source, "__next__") else iter(source)
        self._pending: Any = None
        self.read_count = 0

    def __iter__(self) -> "FramePump":
        return self

    def __next__(self) -> Any:
        if self._pending is not None:
            packet = self._pending
            self._pending = None
        else:
            packet = next(self.source)
        self.read_count += 1
        return packet

    def push_back(self, packet: Any) -> None:
        """把已经取出但属于下一段的帧放回去。"""
        if self._pending is not None:
            raise SourceError(
                "暂存位已经被占用，同一时刻只允许一帧回退。"
                "出现这个错误说明采集层和抽帧层同时在使用这条流。"
            )
        self._pending = packet

    def drain_frames(self, count: int, *, stop_requested: Any = None) -> Any:
        """读掉 ``count`` 帧，返回最后一帧（给实时预览用）。

        按**帧数**而不是秒数：三种来源的"一秒"含义不同（真机是真实时间、回放是
        历史时间、合成是虚拟时间），按帧数走就不用区分它们。
        """
        total = max(1, int(count))
        last: Any = None
        for _ in range(total):
            if stop_requested is not None and stop_requested():
                break
            last = self.__next__()
        if last is None:
            raise SourceError("抽帧时一帧都没读到，图像来源可能已经结束或相机掉线。")
        return last

    @property
    def actual_camera_fps(self) -> float | None:
        """★ v1.0.3：把**来源实测**的帧率透出去。

        为什么必须透：采集层拿这个值干两件事——写进采集元数据、
        以及当"实际帧率"去核对帧时间戳与帧号的对应关系
        （见 :func:`vendor.camera._enrich_capture_timestamps`）。
        而这一层包装（本类）曾经把它挡住：``CaptureEngine`` 拿到的是**本对象**，
        ``getattr(本对象, "actual_camera_fps", None)`` 取不到就回退到
        ``config.effective_fps()``——也就是**配置里的期望帧率**。
        后果不是"少个字段"，而是：

        * 需求一·3 的 5 s 全屏采集检查里，"实际帧率"会变成"期望帧率"，
          一台真的只跑 90 fps 的相机会被报成 132.23 fps 并**判为通过**——
          检查因此形同虚设（阈值是期望值的 95%，自己比自己永远过）；
        * 时间轴核对也会拿错帧率去比对。

        真机上相机的实测帧率来自 GenICam 的 ``ResultingFrameRate`` 等节点
        （见 ``vendor.camera.HikCameraSource._refresh_actual_camera_fps``），
        它每帧刷新；这里每次都转发现读，不做缓存。
        """
        return getattr(self._origin, "actual_camera_fps", None)

    @property
    def actual_camera_fps_source(self) -> str:
        """帧率的出处（GenICam 节点名 / 合成世界 / 配置回退），同样转发给来源。"""
        return str(
            getattr(
                self._origin, "actual_camera_fps_source", "config.EXPECTED_VISION_FPS"
            )
        )

    def close(self) -> None:
        """关掉来源（真机上就是释放相机句柄）。

        必须对**来源对象**调 ``__exit__``：迭代器（生成器）上没有它，
        以前写在 ``self.source`` 上会静默失败——真机上相机就悬着没释放。
        """
        closer = getattr(self._origin, "__exit__", None)
        if closer is None:
            closer = getattr(self.source, "__exit__", None)
        if closer is None:
            return
        try:
            closer(None, None, None)
        except Exception:
            pass


@dataclass
class SourceBundle:
    """一次运行用到的来源，以及它带来的附属对象（合成世界）。"""

    source: ImageSource
    kind: str
    note: str
    world: SyntheticWorld | None = None
    replay_target: ReplayTarget | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        try:
            self.source.__exit__(None, None, None)
        except Exception:
            pass


def open_source(
    config: AppConfig,
    *,
    nominal_joint_deg: Sequence[float],
    start_frame_id: int = 0,
) -> SourceBundle:
    """按 ``config.mode`` 打开图像来源。三种模式走同一条下游流程。"""
    if config.mode == "hardware":
        return SourceBundle(
            source=open_hardware_source(),
            kind="hik_camera",
            note="海康 MV-CS028-10UM 实时采集",
        )
    if config.mode == "replay":
        source, target = open_replay_source(
            config.replay_source, fps=float(config.camera.expected_fps)
        )
        source.__enter__()
        return SourceBundle(
            source=source,
            kind=target.kind,
            note=f"回放：{target.note}（{target.path}）",
            replay_target=target,
        )
    world = SyntheticWorld(config.dry_run, nominal_joint_deg)
    source = SyntheticSource(config, world, start_frame_id=start_frame_id)
    source.__enter__()
    return SourceBundle(
        source=source,
        kind="synthetic",
        note="合成世界（dry_run）：验证流程与判据，不是实验结果",
        world=world,
    )
