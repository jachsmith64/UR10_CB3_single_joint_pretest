"""实验编排：三个主按钮 + 正式实验两组。

这一层是唯一知道"整个实验长什么样"的地方。它把三个东西串起来：
:mod:`sj_pretest.joint_space`（该做什么动作）、:mod:`sj_pretest.robot_joint`
（怎么把动作发出去）、:mod:`sj_pretest.acquisition`（边动边录）。

安全边界（这是本工具最不能出错的部分）
--------------------------------------
* **没有碰撞模型。** 所有运动都靠人工在每一步之前确认；本模块不会绕过它。
  ``confirm`` 回调返回 False 就抛 :class:`~sj_pretest.robot_joint.MotionAborted`，
  整段停止。
* **中止之后不再发任何命令、也不自动回位。** 中止只是把"不再发送"这件事写死，
  机器人停在当前姿态等人处理。自动回位是另一种运动，同样需要人确认，
  所以不能藏在"中止"里替人做决定。
* **拿不到安全状态就不动。** 机器人层已经要求 ``safety_mode == NORMAL``；
  这里再加一层：任何一步运动前如果画面检查不通过（棋盘格丢了、余量不够），
  直接不发这一步。
* **干运行（dry_run）永远不发真命令。** 这一条由
  :class:`~sj_pretest.robot_joint.SimulatedJointRobot` 保证，它连 RTDE 都没连。

"一段采集"是什么
----------------
一次试验被录成**一个采集目录**，里面按阶段切开：
运动前静止 → 下发动作并等停稳 → 保持 → 回程并等停稳 → 运动后记录。
每个阶段的边界由实际帧的时间戳决定，所以分析层能在本段自己的时间轴上准确切窗口，
不需要把相机时钟和 RTDE 时钟对齐（那是做不到的，两个时钟没有公共原点）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .acquisition import CaptureEngine, PhasePlan, SegmentCapture
from .analysis import (
    PHASE_HOLD,
    PHASE_MOVE,
    PHASE_POST,
    PHASE_PRE,
    PHASE_RETURN,
    PretestReport,
    analyze_pretest,
    write_report,
)
from .config import (
    JOINT_NAMES,
    AppConfig,
    check_free_disk,
    trial_plan_summary,
)
from .joint_space import (
    ROLE_RETURN,
    MotionPlan,
    PlannedStep,
    build_approach_plan,
    build_formal_group_a,
    build_formal_group_b,
    build_pretest_plan,
    build_quick_probe_plan,
    build_static_plan,
    joint_delta,
    nominal_offset_summary,
)
from .recorder import (
    RobotStateRecorder,
    RunDirectory,
    create_run_directory,
    safe_name,
    write_csv,
    write_json,
    write_text,
)
from .robot_joint import (
    JointRobotPort,
    MotionAborted,
    make_robot,
)
from .sources import FramePump, SourceBundle, open_source
from .vision import SegmentVision, process_segment

#: 启动前必须能 import 到的模块。缺哪个就直接说缺哪个，不要让界面半死不活。
REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "数值计算"),
    ("cv2", "OpenCV 图像处理（角点识别用）"),
    ("scipy", "统计与优化（分析用）"),
)
#: 到位预览时抓几帧来检查画面（每步只留最后一张图）。
APPROACH_PREVIEW_FRAMES = 5
#: 快速几何检查的抽样间隔：每 stride 帧识别一帧。识别一帧约 150 ms，
#: 整段逐帧跑要几十秒，操作者等不起；抽样跑一两秒就够回答"还看不看得见棋盘格"。
QUICK_PROBE_STRIDE = 8


class ExperimentError(RuntimeError):
    """编排层出错。消息中文，能直接显示在界面上。"""


# --------------------------------------------------------------------------
# 界面注入的回调
# --------------------------------------------------------------------------


@dataclass
class SessionHooks:
    """界面（或自测脚本）注入的回调。

    不给回调也能跑：这时"人工确认"默认**拒绝**——宁可什么都不做，
    也不能在没人确认的情况下自己动起来。
    """

    #: 每一步运动前问人。返回 True 才动。
    confirm: Callable[[str], bool] | None = None
    #: 运动前的额外检查（画面里棋盘格还在不在）。返回 (通过, 原因)。
    precheck: Callable[[], tuple[bool, str]] | None = None
    #: 是否请求中止。
    stop_requested: Callable[[], bool] | None = None
    #: 一行日志（界面显示、控制台打印）。
    on_log: Callable[[str], None] | None = None
    #: 进度（当前/总数，说明文字）。
    on_progress: Callable[[int, int, str], None] | None = None
    #: 到位预览图落地后回调（图片路径，说明文字）。
    on_preview: Callable[[Path, str], None] | None = None

    def log(self, message: str) -> None:
        if self.on_log is not None:
            self.on_log(message)

    def should_stop(self) -> bool:
        return bool(self.stop_requested is not None and self.stop_requested())

    def ask(self, label: str) -> bool:
        if self.confirm is None:
            # 没有人可以确认 = 不允许运动。这是默认拒绝，不是默认允许。
            return False
        return bool(self.confirm(label))


# --------------------------------------------------------------------------
# 一段采集 = 一个主步（可选带一个收尾回程步）
# --------------------------------------------------------------------------


@dataclass
class SegmentPlan:
    """一个采集目录对应的一到两个计划步。"""

    primary: PlannedStep
    follow: PlannedStep | None = None

    @property
    def label(self) -> str:
        return self.primary.label

    @property
    def event_id(self) -> str:
        return self.primary.event.event_id

    @property
    def is_wait_only(self) -> bool:
        """纯等待步（没有目标角）。这种步不采集，只等。"""
        return self.primary.target_joint_deg is None

    @property
    def statistics_event_id(self) -> str | None:
        """计入统计的那一步的事件编号（本段是它的采集）。"""
        if self.primary.event.counts_for_statistics:
            return self.primary.event.event_id
        return None


def capture_segment_count(plans: Sequence[SegmentPlan]) -> int:
    """真正会落盘的段数（纯等待步不采集，不算进去）。"""
    return sum(1 for plan in plans if not plan.is_wait_only)


def iter_segment_plans(plan: MotionPlan) -> list[SegmentPlan]:
    """把一个计划切成"每段采集"。切法由计划自己的语义决定。

    规则：一步之后如果紧跟一个"回到名义位姿"的收尾步（``role=return`` 且
    ``expected_delta_deg == 0``），就把它并进同一段——因为分析要看的正是
    "去程走了多少、回程回到哪里"，这两件事必须在同一段画面里。

    组 A 的阶梯回程**不是**这种收尾步（它降到第 k−1 级，不是回名义），
    所以不会被并进来，而是各自成段——需求五要求去程和回程分别分析。
    """
    steps = list(plan.steps)
    consumed: set[int] = set()
    segments: list[SegmentPlan] = []
    for index, step in enumerate(steps):
        if index in consumed:
            continue
        if step.target_joint_deg is None:
            segments.append(SegmentPlan(primary=step))
            continue
        follow: PlannedStep | None = None
        if index + 1 < len(steps):
            candidate = steps[index + 1]
            if (
                candidate.target_joint_deg is not None
                and candidate.event.role == ROLE_RETURN
                and candidate.event.expected_delta_deg == 0.0
            ):
                follow = candidate
                consumed.add(index + 1)
        segments.append(SegmentPlan(primary=step, follow=follow))
    return segments


# --------------------------------------------------------------------------
# 一次运行
# --------------------------------------------------------------------------


@dataclass
class SessionResult:
    """一次按钮运行的结果摘要，供界面显示。"""

    title: str
    lines: list[str] = field(default_factory=list)
    segments: list[SegmentCapture] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str | None = None

    def add(self, line: str) -> None:
        self.lines.append(line)

    def to_text(self) -> str:
        return "\n".join(self.lines)


class ExperimentSession:
    """一次实验会话：持有配置、来源、机器人、记录器和采集器。"""

    def __init__(
        self,
        config: AppConfig,
        *,
        hooks: SessionHooks | None = None,
        stamp: str | None = None,
    ) -> None:
        self.config = config
        self.hooks = hooks or SessionHooks()
        self.stamp = stamp

        self.run: RunDirectory | None = None
        self.robot: JointRobotPort | None = None
        self.bundle: SourceBundle | None = None
        self.pump: FramePump | None = None
        self.engine: CaptureEngine | None = None
        self.recorder: RobotStateRecorder | None = None

        self.trials: list[dict[str, Any]] = []
        self.static_segment_ids: list[str] = []
        self.approach_frames: list[Path] = []
        self._confirmed: set[str] = set()
        self._abort_reason: str | None = None
        self._opened = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def aborted(self) -> bool:
        return self._abort_reason is not None or bool(
            self.robot is not None and self.robot.abort_reason() is not None
        )

    def abort(self, reason: str) -> None:
        """中止：不再发任何命令，也不自动回位。已采集的数据一律保留。"""
        if self._abort_reason is None:
            self._abort_reason = str(reason)
            if self.robot is not None:
                self.robot.abort(str(reason))
            if self.run is not None:
                self.run.events.write("aborted", reason=str(reason))
                self.run.note(f"[中止] {reason}")
            self.hooks.log(f"已中止：{reason}（不会自动回位，机器人停在当前姿态）")

    def _check_stop(self) -> None:
        if self.hooks.should_stop() and not self.aborted:
            self.abort("操作者在界面上按了中止")
        if self.aborted:
            raise MotionAborted(f"已中止（{self._abort_reason or '未知原因'}），不再发送命令。")

    def open(self, *, run_kind: str) -> RunDirectory:
        """建运行目录、连设备、开图像来源和记录器。"""
        if self._opened:
            raise ExperimentError("这个会话已经开过了；一个会话只对应一次运行目录。")
        config = self.config
        ok, message = check_free_disk(
            config.resolve_output_root(), float(config.paths.min_free_disk_gb)
        )
        if not ok:
            raise ExperimentError(message)
        self.hooks.log(message)

        run = create_run_directory(
            config,
            run_kind=run_kind,
            extra={
                "plan": trial_plan_summary(config),
                "amplitudes_deg": list(config.pretest.amplitudes_deg),
                "formal_steps_deg": dict(config.formal.step_deg),
                "camera_hint": config.camera_hint_lines(),
                "reused_modules": REUSED_MODULE_NOTE,
            },
            stamp=self.stamp,
        )
        self.run = run

        # 图像来源 + 机器人（两种模式各自决定）
        self.bundle = open_source(
            config,
            nominal_joint_deg=config.robot.nominal_joint_deg,
        )
        self.pump = FramePump(self.bundle.source)
        self.engine = CaptureEngine(
            config, self.pump, synthetic=config.mode == "dry_run"
        )
        self.robot = make_robot(
            config,
            world=self.bundle.world,
            confirm=lambda label: self._confirm_motion(label),
            precheck=self.hooks.precheck,
        )
        self.robot.connect(require_control=True)
        self.recorder = RobotStateRecorder(
            run.root / "robot_states.csv",
            self.robot,
            hz=float(config.robot.rtde_record_hz),
            threaded=config.mode == "hardware",
            event_log=run.events,
        ).open()
        self._opened = True
        run.events.write(
            "session_opened",
            source=self.bundle.kind,
            source_note=self.bundle.note,
            safety=self.robot.describe_safety(),
        )
        self.hooks.log(f"图像来源：{self.bundle.note}")
        return run

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as exc:  # pragma: no cover - 断开失败不该掩盖主错误
                self.hooks.log(f"断开机器人时出错（已忽略）：{exc}")
            self.robot = None
        if self.pump is not None:
            self.pump.close()
            self.pump = None
        if self.bundle is not None:
            self.bundle.close()
            self.bundle = None
        self._opened = False

    def __enter__(self) -> "ExperimentSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 确认策略
    # ------------------------------------------------------------------

    def _confirm_motion(self, label: str) -> bool:
        """机器人层每一步都会调到这里。

        需求三：到位过程**每一步**都要确认；预实验/正式实验里**每个关节**开始前
        确认一次，同一个关节内部自动（否则 72 次动作要按 144 次，人会按到手酸，
        反而更容易误按）。这个策略写在配置里，现场可以改。
        """
        if self.aborted:
            return False
        stage = self._stage_from_label(label)
        joint = self._joint_from_label(label)
        per_joint = bool(
            self.config.pretest.confirm_each_joint
            and self.config.pretest.auto_within_joint
        )
        if not per_joint:
            approved = self.hooks.ask(label)
        elif stage in ("approach", "static", "quick_probe"):
            approved = self.hooks.ask(label)
        elif joint is None:
            approved = self.hooks.ask(label)
        elif joint in self._confirmed:
            approved = True
        else:
            approved = self.hooks.ask(
                f"{label}\n\n"
                f"确认后，本关节（{joint}）余下的同组动作将不再逐步询问；"
                "随时可以在界面上按中止。"
            )
            if approved:
                self._confirmed.add(joint)
                if self.run is not None:
                    self.run.events.write("joint_confirmed", joint=joint, label=label)
        if not approved:
            if self.run is not None:
                self.run.events.write("motion_declined", label=label, stage=stage)
            self.hooks.log(f"操作者没有确认这一步，不发送运动：{label}")
        return approved

    @staticmethod
    def _stage_from_label(label: str) -> str:
        for token, name in (
            ("[到位", "approach"),
            ("[预实验", "pretest"),
            ("[组A", "formal_a"),
            ("[组B", "formal_b"),
            ("[快速几何", "quick_probe"),
            ("[静态", "static"),
        ):
            if label.startswith(token):
                return name
        return "unknown"

    @staticmethod
    def _joint_from_label(label: str) -> str | None:
        for name in JOINT_NAMES:
            if name in label:
                return name
        return None

    # ------------------------------------------------------------------
    # 按钮一：连接设备并到达实验姿态
    # ------------------------------------------------------------------

    def connect_devices(self) -> dict[str, Any]:
        """检查依赖/磁盘/输出目录，连机器人、连相机、读当前状态。"""
        config = self.config
        facts: dict[str, Any] = {
            "mode": config.mode,
            "output_root": str(config.resolve_output_root()),
            "collision_status": "unknown",
        }
        for need, purpose in REQUIRED_MODULES:
            try:
                __import__(str(need))
            except ImportError as exc:
                raise ExperimentError(
                    f"缺少 Python 依赖 {need}（{purpose}）：{exc}\n"
                    "请双击 install_deps.bat 安装后再启动。"
                ) from exc
        ok, message = check_free_disk(
            config.resolve_output_root(), float(config.paths.min_free_disk_gb)
        )
        if not ok:
            raise ExperimentError(message)
        facts["disk"] = message

        if self.robot is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        state = self.robot.read_state()
        facts["current_joint_deg"] = [
            round(float(v), 4) for v in state.actual_q_deg
        ]
        facts["commanded_joint_deg"] = (
            None
            if state.commanded_q_deg is None
            else [round(float(v), 4) for v in state.commanded_q_deg]
        )
        facts["tcp_pose"] = (
            None
            if state.actual_tcp_pose is None
            else [round(float(v), 4) for v in state.actual_tcp_pose]
        )
        facts["safety"] = self.robot.describe_safety()
        facts["camera_note"] = self.bundle.note if self.bundle else ""
        if self.run is not None:
            self.run.events.write("devices_connected", **facts)
        self.hooks.log(
            "已读取当前关节角：" + "、".join(f"{v:.4f}" for v in state.actual_q_deg)
        )
        return facts

    def approach_plan(self) -> MotionPlan:
        """按当前实际角生成到位计划。"""
        if self.robot is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        state = self.robot.read_state()
        config = self.config
        return build_approach_plan(
            state.actual_q_deg,
            config.robot.nominal_joint_deg,
            points=int(config.robot.approach_points),
            hold_s=0.3,
            limit_margin_deg=1.0,
        )

    def run_approach(self) -> SessionResult:
        """逐步走到实验姿态：每一步都确认、每步后给一张预览图。"""
        if self.robot is None or self.pump is None or self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        config = self.config
        plan = self.approach_plan()
        result = SessionResult(title="到位过程")
        result.add(
            f"到位计划：{len(plan.steps)} 个中间点，"
            f"速度 {config.robot.approach_speed_deg_s}°/s，"
            f"加速度 {config.robot.approach_accel_deg_s2}°/s²（保守值，可用界面改）"
        )
        for step in plan.steps:
            if step.target_joint_deg is None:
                continue
            result.add(
                f"{step.label}：相对实验姿态 "
                + nominal_offset_summary(config.robot.nominal_joint_deg, step.target_joint_deg)
            )
        if self.run is not None:
            for line in result.lines:
                self.run.note(line)

        frames_per_preview = max(
            APPROACH_PREVIEW_FRAMES,
            int(config.camera.sample_image_count),
        )
        for index, step in enumerate(plan.steps, start=1):
            self._check_stop()
            if step.target_joint_deg is None:
                continue
            # 需求三.1 要"显示每个点相对当前位姿的增量"：这里现读一次实际关节角，
            # 算的是"从**现在这个姿态**要动多少"，而不是相对名义位姿的名义差值。
            # 两者在第二步之后就完全不同了，混用会让人以为后面的点没在动。
            current = self.robot.read_state().actual_q_deg
            delta_text = "、".join(
                f"{name} {value:+.4f}°"
                for name, value in zip(
                    JOINT_NAMES, joint_delta(current, step.target_joint_deg)
                )
            )
            self.hooks.log(
                f"[到位 {index}/{len(plan.steps)}] {step.label}\n"
                f"  本次增量（相对当前姿态）：{delta_text}\n"
                f"  到位后相对实验姿态："
                + nominal_offset_summary(
                    config.robot.nominal_joint_deg, step.target_joint_deg
                )
            )
            self.robot.move_to_joint(
                step.target_joint_deg,
                speed_deg_s=float(config.robot.approach_speed_deg_s),
                accel_deg_s2=float(config.robot.approach_accel_deg_s2),
                event_id=step.event.event_id,
                label=step.label,
            )
            if self.recorder is not None:
                self.recorder.mark(
                    event_id=step.event.event_id, stage="approach", label=step.label
                )
                if not self.recorder.threaded:
                    # 到位过程没有采集帧，同步模式下要自己采一条，
                    # 否则这一步在 robot_states.csv 里没有任何记录。
                    self.recorder.sample()
            settled = self.robot.wait_until_settled(
                step.target_joint_deg,
                tolerance_deg=float(config.robot.settle_tolerance_deg),
                hold_s=float(config.robot.settle_hold_s),
                timeout_s=float(config.robot.settle_timeout_s),
            )
            if not settled:
                result.add(f"第 {index} 步等待停稳超时；画面检查仍然照做，但要人工留意。")
                self.hooks.log("等待停稳超时——不建议继续，请人工确认姿态是否已经稳定。")
            # 抽一小段帧（只留最后一张预览图，不落 RAW）做画面检查。
            packet = self.pump.drain_frames(
                frames_per_preview, stop_requested=self.hooks.stop_requested
            )
            preview, check = self._preview_and_check(packet.frame, step.event.event_id, index)
            result.add(f"第 {index} 步画面检查：{check}")
            self.hooks.log(f"第 {index} 步画面检查：{check}")
            if preview is not None and self.hooks.on_preview is not None:
                self.hooks.on_preview(preview, f"到位 {index}/{len(plan.steps)}")
            if not check.startswith("通过"):
                self.hooks.log(
                    "画面检查没通过。下一步之前请人工确认棋盘格完整、余量足够。"
                )

        self.hooks.log("到位过程结束：已到达实验姿态（请人工核对示教器与现场）。")
        if self.run is not None:
            self.run.events.write("approach_finished", steps=len(plan.steps))
        return result

    def _preview_and_check(
        self, frame: Any, event_id: str, index: int
    ) -> tuple[Path | None, str]:
        """把预览图写盘并做一次轻量画面检查。

        检查内容：88 个内角点是否全部检出、角点到图像边缘的余量够不够。
        这两件事只用**一张**帧，所以耗时是"一次识别"（约 150 ms），
        不会拖慢采集——因为此刻并没有在采集。
        """
        import cv2  # 局部导入：只有真要落图时才需要

        config = self.config
        preview_path: Path | None = None
        if self.run is not None:
            preview_path = self.run.root / "approach_previews" / f"{index:02d}_{safe_name(event_id)}.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), np_as_uint8(frame))
        corners_note, margin_ok = self._check_board_frame(frame)
        verdict = "通过" if margin_ok else "未通过"
        text = (
            f"{verdict}——{corners_note}"
            f"（要求内角点 {config.camera.board_inner_corners[0]}×"
            f"{config.camera.board_inner_corners[1]} 全部检出，"
            f"并留出 ≥ {config.camera.min_margin_px} px 余量）"
        )
        if self.run is not None:
            self.run.events.write(
                "preview_check", index=index, event_id=event_id, ok=margin_ok, note=text
            )
        return preview_path, text

    def _check_board_frame(self, frame: Any) -> tuple[str, bool]:
        from .vendor_shim import vendor_module

        camera = vendor_module("camera")
        tracker = camera.CheckerboardTracker()
        gray, _metrics, _origin, _timing = camera.preprocess_frame(frame)
        _checker, corners, _timing = tracker.process(gray)
        want = int(self.config.camera.board_inner_corners[0]) * int(
            self.config.camera.board_inner_corners[1]
        )
        if corners is None:
            return "没有检出棋盘格", False
        points = np_as_array(corners).reshape(-1, 2)
        if len(points) < want:
            return f"只检出 {len(points)}/{want} 个内角点", False
        height, width = gray.shape[:2]
        margin = float(
            min(
                points[:, 0].min(),
                points[:, 1].min(),
                width - 1 - points[:, 0].max(),
                height - 1 - points[:, 1].max(),
            )
        )
        required = float(self.config.camera.min_margin_px)
        if margin < required:
            return (
                f"内角点 {len(points)}/{want} 全部检出，但边缘余量只有 {margin:.0f} px"
                f"（要求 ≥ {required:.0f} px）",
                False,
            )
        return (
            f"内角点 {len(points)}/{want} 全部检出，边缘余量 {margin:.0f} px",
            True,
        )

    # ------------------------------------------------------------------
    # 按钮二（A）：静态噪声基线
    # ------------------------------------------------------------------

    def run_static(self) -> SegmentCapture:
        config = self.config
        if self.robot is None or self.engine is None or self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        self._check_stop()
        duration = float(config.effective_durations()["static"])
        plan = build_static_plan(config.robot.nominal_joint_deg, duration_s=duration)
        step = plan.steps[0]
        self.hooks.log(
            f"[静态基线] 保持不动录 {duration:.1f} s："
            "这段时间里不要碰相机、台面和机械臂。"
        )
        if self.recorder is not None:
            self.recorder.mark(event_id=step.event.event_id, stage="static", label="静态基线")
        return self.capture_static(
            segment_id=step.event.event_id, duration_s=duration
        )

    def capture_static(self, *, segment_id: str, duration_s: float) -> SegmentCapture:
        """只录不动的一段。抽出来是为了让"预览抽帧"和"静态采集"解耦。"""
        if self.engine is None or self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        before = self.hooks.log
        before(f"开始静态采集：{duration_s:.1f} s（按帧时间戳计时）")
        record = self.engine.capture_segment(
            self.run.segment_dir(segment_id),
            segment_id=segment_id,
            kind="static",
            duration_s=duration_s,
            stop_requested=self.hooks.stop_requested,
            metadata_extra={
                "event_id": segment_id,
                "stage": "static",
                "counts_for_statistics": False,
            },
            on_frame=self._on_frame,
        )
        before(record.summary_line())
        self.static_segment_ids.append(segment_id)
        if self.run is not None:
            self.run.events.write(
                "segment_captured", **{k: v for k, v in record.to_dict().items() if k != "phases"}
            )
        return record

    def _on_frame(self, packet: Any, frame: Any, index: int) -> None:
        """每接一帧：推进机器人的虚拟时间，并同步采一条机器人状态。

        真机上 RTDE 由后台线程按 ``rtde_record_hz`` 采（相机线程不等它）；
        干运行/回放没有线程，就在**每帧回调里同步采一条**——这样机器人状态和
        画面帧是同一个节拍，第一层分析才能按相位窗口把两边对齐。
        """
        if self.robot is not None:
            self.robot.sync_host_ns(int(packet.host_ns))
        if self.recorder is not None and not self.recorder.threaded:
            # 用**这一帧的时间戳**给 RTDE 行打时间，两边就落在同一根时间轴上。
            self.recorder.sample(host_ns=int(packet.host_ns))

    # ------------------------------------------------------------------
    # 按钮二（B）：快速几何检查
    # ------------------------------------------------------------------

    def run_quick_probes(self) -> tuple[SessionResult, list[str]]:
        """每个关节正负各走 0.05°，然后抽样识别几帧做几何确认。"""
        config = self.config
        if self.run is None or self.robot is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        plan = build_quick_probe_plan(
            config.robot.nominal_joint_deg,
            config.pretest.joints,
            probe_deg=float(config.pretest.quick_probe_deg),
            hold_s=float(config.camera.hold_s),
            return_settle_s=float(config.pretest.return_before_settle_s),
        )
        result = SessionResult(title="快速几何检查")
        failed_joints: list[str] = []
        plans = iter_segment_plans(plan)
        triggers = self._joint_triggers(plans)
        measurements: list[ProbeMeasurement] = []
        for position, segment_plan in enumerate(plans, start=1):
            self._check_stop()
            joint = segment_plan.primary.event.joint
            if trigger := triggers.get(position):
                if not self.hooks.ask(trigger):
                    raise MotionAborted(f"操作者没有确认进入 {joint} 的快速几何检查。")
                # 这次确认就代表"本关节被批准了"，不要让机器人层再问一遍同一个关节。
                if joint:
                    self._confirmed.add(joint)
            record = self._run_segment(
                segment_plan, kind="quick_probe", position=position, total=len(plans)
            )
            if record is None:
                continue
            result.segments.append(record)
            measurement = self._measure_probe(record, segment_plan)
            measurements.append(measurement)

        # 逐关节汇总：正负两个方向都量到了才能判"方向对不对"。
        for joint in dict.fromkeys(m.joint for m in measurements):
            ok, lines = self._judge_joint_probe(
                joint, [m for m in measurements if m.joint == joint]
            )
            result.add(f"{joint} 快速几何检查：{'通过' if ok else '未通过'}")
            result.lines.extend(f"    {line}" for line in lines)
            self.hooks.log(f"{joint} 快速几何检查：{'通过' if ok else '未通过'}")
            for line in lines:
                self.hooks.log(f"    {line}")
            if not ok:
                failed_joints.append(joint)
        if failed_joints:
            result.add(
                "以下关节的快速几何检查没通过："
                + "、".join(failed_joints)
                + "。按需求三 B，这里**暂停**下来等人处理；"
                "三档预实验要不要继续，由人决定，程序不自动往下走。"
            )
            result.aborted = True
            result.abort_reason = "快速几何检查未通过：" + "、".join(failed_joints)
        return result, failed_joints

    def _measure_probe(
        self, record: SegmentCapture, segment_plan: SegmentPlan
    ) -> ProbeMeasurement:
        """抽样离线识别一段快速探针，量出"这个方向动了多少"。

        抽样的时间窗取三段：运动前（pre_motion）、保持（hold）、运动后（post_motion）。
        pre 和 post 都是"回到名义位姿的静止段"，把它们并起来当噪声池；
        hold 是"走完 0.05° 停稳后"的静止段，拿它跟噪声池比，得到信噪比。
        这样不需要额外的静态基线，也不会拿"首末帧"去比——首末帧一个是运动前、
        一个是回程之后，两个都在名义位姿，位移自然是零（这是最初版本的错误）。
        """
        config = self.config
        segment = process_segment(
            record.dir,
            config=config,
            segment_id=record.segment_id,
            save_corners=False,
            stride=QUICK_PROBE_STRIDE,
        )
        joint = segment_plan.primary.event.joint or "?"
        direction = int(segment_plan.primary.event.direction)
        rotation_joint = joint in set(config.vision.rotation_joints)
        signal = segment.window(*(segment_phase_window(segment, PHASE_HOLD) or (0.0, 0.0)))
        noise_pool: list[Any] = []
        for label in (PHASE_PRE, PHASE_POST):
            bounds = segment_phase_window(segment, label)
            if bounds is not None:
                noise_pool.extend(segment.window(*bounds))
        signal = [f for f in signal if f.valid]
        noise_frames = [f for f in noise_pool if f.valid]

        if rotation_joint:
            values = [float(f.rotation_deg) for f in signal if f.rotation_deg is not None]
            noise_values = [
                float(f.rotation_deg) for f in noise_frames if f.rotation_deg is not None
            ]
            unit = "deg"
        else:
            values = [
                (float(f.centroid_x_px), float(f.centroid_y_px))
                for f in signal
                if f.centroid_x_px is not None and f.centroid_y_px is not None
            ]
            noise_values = [
                (float(f.centroid_x_px), float(f.centroid_y_px))
                for f in noise_frames
                if f.centroid_x_px is not None and f.centroid_y_px is not None
            ]
            unit = "px"
        return ProbeMeasurement(
            joint=joint,
            direction=direction,
            label=segment_plan.label,
            kind="rotation" if rotation_joint else "translation",
            unit=unit,
            signal=values,
            noise=noise_values,
            corner_max=max((f.corner_count for f in segment.frames if f.valid), default=0),
            valid_ratio=segment.valid_ratio,
            dropped_ratio=float(record.dropped_ratio),
        )

    def _judge_joint_probe(
        self, joint: str, measurements: Sequence["ProbeMeasurement"]
    ) -> tuple[bool, list[str]]:
        """按需求三 B 的五条判据给出"过/不过 + 理由"。

        需求三 B 要看的是：棋盘格完整、图像有明显运动、方向大致符合理论、
        以面内运动为主、没有严重掉帧。这里逐条给出结论，并且**只做放行/暂停**，
        不替代按钮三的正式分析。
        """
        from .vision import (
            THEORY_DIRECTION_MEANINGLESS_MARK,
            expected_image_direction_deg,
        )

        config = self.config
        want = int(config.camera.board_inner_corners[0]) * int(
            config.camera.board_inner_corners[1]
        )
        lines: list[str] = []
        ok = True
        rotation_joint = joint in set(config.vision.rotation_joints)

        for measurement in measurements:
            tag = "正向" if measurement.direction > 0 else "负向"
            lines.append(
                f"{tag} 0.05°（{measurement.label}）："
                f"棋盘格完整度 {measurement.valid_ratio:.0%} 抽样帧有效，"
                f"最多 {measurement.corner_max}/{want} 个内角点，"
                f"掉帧率 {measurement.dropped_ratio:.2%}"
            )
            if measurement.corner_max < want:
                ok = False
                lines.append(
                    f"    {tag}：最多只检出 {measurement.corner_max}/{want} 个内角点，"
                    "棋盘格不完整或不在视野内。"
                )
            if measurement.dropped_ratio > float(config.camera.max_dropped_ratio):
                ok = False
                lines.append(
                    f"    {tag}：掉帧率超过上限 {config.camera.max_dropped_ratio:.2%}，"
                    "先解决采集稳定性再继续。"
                )
            metric = probe_metric(measurement)
            if metric is None:
                ok = False
                lines.append(
                    f"    {tag}：静止段或保持段的有效帧太少，这次量不出来。"
                )
                continue
            value, sigma, ratio = metric
            lines.append(
                f"    {tag}位移 {value:+.3f} {measurement.unit}，"
                f"静止段噪声 {sigma:.3f} {measurement.unit}，信噪比 {ratio:.1f}"
            )
            if ratio < float(config.thresholds.quick_probe_min_snr):
                ok = False
                lines.append(
                    f"    {tag}：位移不到噪声的 {config.thresholds.quick_probe_min_snr:.1f} 倍——"
                    "这个关节的 0.05° 在画面上几乎看不出来（可能是正常物理现象，"
                    "也可能是关节/摆放不对，需要人工判断）。"
                )
            if ratio < 5.0:
                lines.append(f"    {tag}：信噪比 {ratio:.1f} 偏低，方向结论仅供参考。")

        # 方向：正负两个方向应当大致相反（夹角接近 180°），
        # 而且各自要对得上名义运动学预测的方向。
        usable = [m for m in measurements if probe_metric(m) is not None]
        if len(usable) < 2:
            ok = False
            lines.append("正负两个方向没有都量到，无法判断方向是否一致。")
        elif rotation_joint:
            first, second = usable[0], usable[1]
            v1 = probe_metric(first)[0]
            v2 = probe_metric(second)[0]
            lines.append(
                f"图像二维转角：{first.label} {v1:+.4f}°，{second.label} {v2:+.4f}°"
            )
            if v1 * v2 >= 0:
                ok = False
                lines.append(
                    "    J6 正负两个方向的图像转角同号：转角方向与关节方向对不上，"
                    "请人工确认关节没认错。"
                )
            else:
                lines.append(
                    "    正负方向的图像转角反号，符合“关节反向转、图像反向转”的预期。"
                )
        else:
            index = JOINT_NAMES.index(joint)
            theory_deg, theory_note = expected_image_direction_deg(
                index, config.robot.nominal_joint_deg, config=config
            )
            if THEORY_DIRECTION_MEANINGLESS_MARK in theory_note:
                lines.append(
                    "理论方向无意义（这个关节在此姿态下对画面中心几乎不产生面内位移），"
                    "跳过方向比对，只保留“位移看不看得出来”这一条。"
                )
            else:
                for measurement in usable:
                    if measurement.kind != "translation":
                        continue
                    if probe_metric(measurement)[2] < 5.0:
                        lines.append(
                            f"    {measurement.label}：信噪比不足 5，跳过方向比对。"
                        )
                        continue
                    measured = probe_direction_deg(measurement)
                    expected = theory_deg + (0.0 if measurement.direction > 0 else 180.0)
                    if measured is None:
                        continue
                    difference = abs(angle_difference(measured, expected))
                    lines.append(
                        f"    {measurement.label}：实测方向 {measured:.1f}°，"
                        f"按理论应为 {expected:.1f}°，相差 {difference:.1f}°"
                    )
                    if difference > float(config.thresholds.quick_probe_direction_tol_deg):
                        ok = False
                        lines.append(
                            "       方向与理论相差过大：请人工看画面确认关节没错、摆放没错。"
                        )
                lines.append(f"    方向依据：{theory_note}")
        lines.append(
            "这是**抽样**检查（每 "
            f"{QUICK_PROBE_STRIDE} 帧抽 1 帧，只用来放行/暂停）；"
            "正式结论在按钮三的离线分析里。"
        )
        return ok, lines

    # ------------------------------------------------------------------
    # 按钮二（C）：三档微动预实验
    # ------------------------------------------------------------------

    def run_pretest(self) -> SessionResult:
        config = self.config
        if self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        plan = build_pretest_plan(config, config.robot.nominal_joint_deg)
        plans = iter_segment_plans(plan)
        statistics = [p for p in plans if p.statistics_event_id]
        result = SessionResult(title="三档微动预实验")
        result.add(
            f"计划：{len(statistics)} 次统计试验（"
            f"{len(config.pretest.joints)} 个关节 × {len(config.pretest.amplitudes_deg)} 档幅度"
            f" × 2 个方向 × {config.pretest.repeats_per_direction} 次重复），"
            f"每次动作都从名义位姿独立出发（不累计）。"
        )
        self.hooks.log(result.lines[-1])
        triggers = self._joint_triggers(plans)
        for position, segment_plan in enumerate(plans, start=1):
            self._check_stop()
            joint = segment_plan.primary.event.joint
            if trigger := triggers.get(position):
                if not self.hooks.ask(trigger):
                    raise MotionAborted(f"操作者没有确认进入 {joint} 的预实验。")
                # 同上：关节级确认就是本关节其余动作的通行证。
                if joint:
                    self._confirmed.add(joint)
            record = self._run_segment(
                segment_plan, kind="pretest", position=position, total=len(plans)
            )
            if record is None:
                continue
            result.segments.append(record)
            self._record_trial(segment_plan, record)
            self.hooks.log(record.summary_line())
        result.add(f"预实验采集完成：{len(result.segments)} 段。")
        self._write_trial_plan()
        return result

    def _record_trial(self, segment_plan: SegmentPlan, record: SegmentCapture) -> None:
        """登记一次统计试验，供按钮三的离线分析使用。"""
        event = segment_plan.primary.event
        if not event.counts_for_statistics:
            return
        if self.run is not None:
            self.trials.append(
                {
                    "event_id": event.event_id,
                    "segment_id": record.segment_id,
                    "stage": event.stage,
                    "joint": event.joint,
                    "amplitude_deg": event.amplitude_deg,
                    "direction": event.direction,
                    "repeat": event.repeat_index,
                    "commanded_delta_deg": (
                        event.expected_delta_deg
                        if event.expected_delta_deg is not None
                        else (event.direction or 0) * float(event.amplitude_deg or 0.0)
                    ),
                    "settle_ended_by": (
                        None
                        if record.phase(PHASE_MOVE) is None
                        else record.phase(PHASE_MOVE).ended_by
                    ),
                    "dropped_ratio": record.dropped_ratio,
                    "frame_count": record.frame_count,
                }
            )

    def _write_trial_plan(self) -> None:
        if self.run is None or not self.trials:
            return
        write_json(self.run.analysis_dir / "trial_plan.json", self.trials)
        write_csv(self.run.analysis_dir / "trial_plan.csv", self.trials)

    # ------------------------------------------------------------------
    # 正式实验
    # ------------------------------------------------------------------

    def formal_plan(self, group: str, joint: str, step_deg: float) -> MotionPlan:
        config = self.config
        if group.upper() == "A":
            return build_formal_group_a(
                config, config.robot.nominal_joint_deg, joint, float(step_deg)
            )
        if group.upper() == "B":
            return build_formal_group_b(
                config, config.robot.nominal_joint_deg, joint, float(step_deg)
            )
        raise ExperimentError(f"正式实验只有 A、B 两组，收到 {group!r}。")

    def run_formal(
        self, group: str, *, step_deg: Mapping[str, float] | None = None
    ) -> SessionResult:
        """跑一组正式实验。每个关节用自己的步长（界面上确认过的那组）。"""
        config = self.config
        if self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        if group.upper() == "A" and not config.formal.enable_group_a:
            raise ExperimentError("配置里关掉了组 A（formal.enable_group_a = false）。")
        if group.upper() == "B" and not config.formal.enable_group_b:
            raise ExperimentError("配置里关掉了组 B（formal.enable_group_b = false）。")
        steps = dict(step_deg or config.formal.step_deg)
        result = SessionResult(title=f"正式实验 组{group.upper()}")
        for joint in config.pretest.joints:
            step = steps.get(joint)
            if not step:
                result.add(f"{joint}：没有填步长，跳过。")
                continue
            self._check_stop()
            plan = self.formal_plan(group, joint, float(step))
            plans = iter_segment_plans(plan)
            result.add(
                f"组{group.upper()} {joint} Δ={step}°：{capture_segment_count(plans)} 段采集"
                f"（阶梯 {config.formal.staircase_n} 级，重复 {config.formal.repeats} 遍）"
            )
            self.hooks.log(result.lines[-1])
            if not self.hooks.ask(
                f"[组{group.upper()}] 开始 {joint} 的正式实验 Δ={step}°。\n\n"
                "理论范围检查已通过，但碰撞状态仍是 unknown。"
            ):
                raise MotionAborted(f"操作者没有确认开始 组{group.upper()} {joint}。")
            self._confirmed.add(joint)
            for position, segment_plan in enumerate(plans, start=1):
                self._check_stop()
                record = self._run_segment(
                    segment_plan,
                    kind=f"formal_{group.lower()}",
                    position=position,
                    total=len(plans),
                )
                if record is None:
                    continue
                result.segments.append(record)
                self._record_trial(segment_plan, record)
        self._write_trial_plan()
        result.add(f"组{group.upper()} 采集完成：{len(result.segments)} 段。")
        return result

    # ------------------------------------------------------------------
    # 通用：跑一段采集
    # ------------------------------------------------------------------

    def _joint_triggers(self, plans: Sequence[SegmentPlan]) -> dict[int, str]:
        """找出"每个关节开始前的那一次确认"在哪一段。"""
        triggers: dict[int, str] = {}
        if not bool(self.config.pretest.confirm_each_joint):
            return triggers
        seen: set[str | None] = set()
        for position, segment_plan in enumerate(plans, start=1):
            joint = segment_plan.primary.event.joint
            if joint is None or joint in seen:
                continue
            seen.add(joint)
            if segment_plan.is_wait_only:
                continue
            triggers[position] = (
                f"接下来是这个关节（{joint}）的成组动作。"
                "现在会在关节之间停下等你确认，关节内部不再逐步询问。\n\n"
                "请先确认现场：棋盘格在视野内、余量足够、没有人和障碍物在运动范围内、"
                "急停随手可及。碰撞状态：unknown。"
            )
        return triggers

    def _run_segment(
        self,
        segment_plan: SegmentPlan,
        *,
        kind: str,
        position: int,
        total: int,
    ) -> SegmentCapture | None:
        """跑一段采集：一个主步（+可选的收尾回程步）。"""
        if self.engine is None or self.run is None or self.robot is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        config = self.config
        step = segment_plan.primary
        follow = segment_plan.follow

        if segment_plan.is_wait_only:
            # 纯等待：按帧时间推进（干运行/回放也一样），不采集不落盘。
            self.hooks.log(step.label)
            self._wait_frames(float(step.hold_s))
            return None

        self._check_stop()
        if self.hooks.on_progress is not None:
            self.hooks.on_progress(position, total, step.label)

        target = list(step.target_joint_deg or ())
        return_target = (
            list(follow.target_joint_deg) if follow is not None else None
        )
        settle_tol = float(config.robot.settle_tolerance_deg)
        settle_hold = float(config.robot.settle_hold_s)
        settle_timeout = float(config.robot.settle_timeout_s)
        event_id = step.event.event_id
        robot = self.robot
        is_formal = step.event.stage in ("formal_a", "formal_b")
        speed_deg_s, accel_deg_s2 = config.effective_speed(formal=is_formal)

        def do_move() -> None:
            robot.move_to_joint(
                target,
                speed_deg_s=float(speed_deg_s),
                accel_deg_s2=float(accel_deg_s2),
                event_id=event_id,
                label=step.label,
            )
            if self.recorder is not None:
                self.recorder.mark(
                    event_id=event_id, stage=step.event.stage, label=step.label
                )

        def do_return() -> None:
            assert return_target is not None and follow is not None
            robot.move_to_joint(
                return_target,
                speed_deg_s=float(speed_deg_s),
                accel_deg_s2=float(accel_deg_s2),
                event_id=follow.event.event_id,
                label=follow.label,
            )
            if self.recorder is not None:
                self.recorder.mark(
                    event_id=follow.event.event_id,
                    stage=follow.event.stage,
                    label=follow.label,
                )

        # 时长一律走 effective_durations()：干运行可以整体压短做自测，
        # 真机永远用 camera 里的现场时长（这条规则写在 config 里，不在这里复制）。
        durations = config.effective_durations()
        pre_s = float(durations["pre_motion"])
        post_s = float(durations["post_motion"])
        hold_s = float(durations["hold"] if config.mode == "dry_run" else step.hold_s)

        phases = [
            PhasePlan(
                label=PHASE_PRE,
                min_duration_s=pre_s,
                max_duration_s=pre_s + 5.0,
            ),
            PhasePlan(
                label=PHASE_MOVE,
                min_duration_s=0.02,
                action=do_move,
                wait_for=lambda: robot.settled(
                    target, tolerance_deg=settle_tol, hold_s=settle_hold
                ),
                max_duration_s=float(settle_timeout) + float(durations["hold"]),
            ),
            PhasePlan(
                label=PHASE_HOLD,
                min_duration_s=hold_s,
                max_duration_s=hold_s + float(settle_timeout) + 5.0,
            ),
        ]
        if return_target is not None:
            phases.extend(
                [
                    PhasePlan(
                        label=PHASE_RETURN,
                        min_duration_s=0.02,
                        action=do_return,
                        wait_for=lambda: robot.settled(
                            return_target, tolerance_deg=settle_tol, hold_s=settle_hold
                        ),
                        max_duration_s=float(settle_timeout) + float(durations["hold"]),
                    ),
                    PhasePlan(
                        label=PHASE_POST,
                        min_duration_s=post_s,
                        max_duration_s=post_s + 5.0,
                    ),
                ]
            )

        metadata = {
            "event_id": event_id,
            "stage": step.event.stage,
            "joint": step.event.joint,
            "amplitude_deg": step.event.amplitude_deg,
            "direction": step.event.direction,
            "repeat": step.event.repeat_index,
            "role": step.event.role,
            "counts_for_statistics": step.event.counts_for_statistics,
            "expected_delta_deg": step.event.expected_delta_deg,
            "target_joint_deg": [round(float(v), 6) for v in target],
            "delta_from_nominal_deg": [
                round(float(v), 6) for v in step.delta_from_nominal_deg
            ],
            "return_event_id": None if follow is None else follow.event.event_id,
            "return_target_joint_deg": (
                None if return_target is None else [round(float(v), 6) for v in return_target]
            ),
            "collision_status": "unknown",
        }
        try:
            record = self.engine.capture_segment(
                self.run.segment_dir(event_id),
                segment_id=event_id,
                kind=kind,
                phases=phases,
                stop_requested=self.hooks.stop_requested,
                metadata_extra=metadata,
                on_frame=self._on_frame,
            )
        except MotionAborted:
            # 人工拒绝或中止：不是错误，是要立刻停下来。
            raise
        except Exception as exc:
            # 采集失败也要保住已经落盘的数据，并把原因写进事件日志和 notes。
            if self.run is not None:
                self.run.events.write("segment_failed", event_id=event_id, error=str(exc))
                self.run.note(f"[采集失败] {event_id}：{exc}")
            self.hooks.log(f"采集失败（{event_id}）：{exc}")
            raise
        if self.run is not None:
            self.run.events.write(
                "segment_captured",
                **{k: v for k, v in record.to_dict().items() if k != "phases"},
            )
        move_phase = record.phase(PHASE_MOVE)
        if move_phase is not None and move_phase.ended_by == "timeout":
            self.hooks.log(
                f"{event_id}：等待停稳超时（到时间还在动）。这一条会被如实记录，"
                "分析时不会当作已到位。"
            )
        if record.dropped_ratio > float(config.camera.max_dropped_ratio):
            self.hooks.log(
                f"{event_id}：掉帧率 {record.dropped_ratio:.2%} 偏高"
                f"（上限 {config.camera.max_dropped_ratio:.2%}），已记录。"
            )
        return record

    def _wait_frames(self, seconds: float) -> None:
        """按帧时间推进等一段时间（组间等待用）。"""
        if self.pump is None:
            return
        count = max(1, int(round(float(seconds) * self.config.effective_fps())))
        self.pump.drain_frames(count, stop_requested=self.hooks.stop_requested)

    # ------------------------------------------------------------------
    # 按钮三：离线分析
    # ------------------------------------------------------------------

    def analyze_offline(
        self, *, stride: int = 1, progress: Callable[[str], None] | None = None
    ) -> PretestReport:
        """把预实验的采集全部离线识别一遍，算出推荐步长。"""
        if self.run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        if not self.trials and not self.static_segment_ids:
            raise ExperimentError(
                "还没有跑过预实验，没有可分析的数据。请先执行按钮二的静态基线与三档预实验。"
            )
        config = self.config
        run = self.run
        segments: dict[str, SegmentVision] = {}

        todo: list[str] = list(self.static_segment_ids) + [
            str(t["segment_id"]) for t in self.trials
        ]
        for position, segment_id in enumerate(todo, start=1):
            directory = run.segment_dir(segment_id)
            if not directory.is_dir():
                self.hooks.log(f"{segment_id}：采集目录不存在，跳过。")
                continue
            if progress is not None:
                progress(f"离线识别 {position}/{len(todo)}：{segment_id}")
            self.hooks.log(f"离线识别 {position}/{len(todo)}：{segment_id}")
            try:
                segments[segment_id] = process_segment(
                    directory,
                    config=config,
                    segment_id=segment_id,
                    save_corners=True,
                    corners_dir=run.corners_dir,
                    metrics_path=run.vision_dir / "metrics.csv",
                    stride=stride,
                )
            except Exception as exc:
                # 单段识别失败不推翻整批：如实记下，继续算其余的。
                message = f"{segment_id} 离线识别失败：{exc}"
                self.hooks.log(message)
                run.note(f"[离线识别失败] {message}")
                run.events.write("vision_failed", segment_id=segment_id, error=str(exc))

        rows = load_states(run.root / "robot_states.csv")
        report = analyze_pretest(
            config=config,
            segments=segments,
            trials=self.trials,
            rtde_rows=rows,
            static_segment_ids=self.static_segment_ids,
        )
        write_report(report, run.analysis_dir)
        write_text(
            run.analysis_dir / "analysis_notes.txt",
            "\n".join(report.summary_lines()) + "\n",
        )
        run.events.write(
            "analysis_finished",
            trials=len(report.trials),
            recommended={
                r.joint: r.recommended_deg for r in report.recommendations
            },
        )
        self.hooks.log("分析完成，结果已写入 analysis/ 目录。")
        for line in report.summary_lines():
            self.hooks.log(line)
        return report

    # ------------------------------------------------------------------
    # 正式实验之前的理论范围检查
    # ------------------------------------------------------------------

    def formal_range_checks(
        self, step_deg: Mapping[str, float] | None = None
    ) -> list[str]:
        """按名义运动学检查整段行程。**不是碰撞检查**。"""
        from .analysis import check_formal_range

        config = self.config
        steps = dict(step_deg or config.formal.step_deg)
        pretest_max = max(float(v) for v in config.pretest.amplitudes_deg)
        lines: list[str] = []
        for joint in config.pretest.joints:
            step = steps.get(joint)
            if not step:
                lines.append(f"{joint}：没有填步长，无法检查。")
                continue
            verdict = check_formal_range(
                joint,
                float(step),
                int(config.formal.staircase_n),
                config=config,
                pretest_max_step_deg=pretest_max,
            )
            lines.append(
                f"{joint} Δ={step}° × {config.formal.staircase_n} 级："
                + ("理论检查通过" if verdict.ok else "理论检查**未通过**")
            )
            lines.extend(f"    {line}" for line in verdict.lines)
        lines.append(
            "以上是名义运动学检查，**不等于碰撞安全检查**；"
            "collision_status 始终是 unknown。"
        )
        if self.run is not None:
            self.run.events.write("formal_range_checked", steps=steps, lines=lines)
        return lines


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


@dataclass
class ProbeMeasurement:
    """一次快速探针量到的东西（按钮二 B 用，不参与正式统计）。

    ``signal`` 是"走完 0.05° 停稳后"的静止段样本，
    ``noise`` 是"运动前 + 运动后"两段静止样本（都停在名义位姿）。
    两者相减才是这一步真正动了多少。
    """

    joint: str
    direction: int
    label: str
    kind: str  # "translation"（J1–J5）或 "rotation"（J6）
    unit: str  # "px" 或 "deg"
    signal: list[Any]
    noise: list[Any]
    corner_max: int
    valid_ratio: float
    dropped_ratio: float


def _mean_point(samples: Sequence[Any]) -> Any:
    import numpy as np

    return np.mean(np.asarray(samples, dtype=float), axis=0)


def _std_point(samples: Sequence[Any]) -> Any:
    import numpy as np

    return np.std(np.asarray(samples, dtype=float), axis=0)


def probe_metric(
    measurement: ProbeMeasurement, *, min_samples: int = 4
) -> tuple[float, float, float] | None:
    """把一次探针量成 (有符号位移, 噪声, 信噪比)。

    平移：位移是"保持段质心均值 − 静止段质心均值"的模长，
    噪声取**沿位移方向**的静止段标准差（其它方向的噪声不参与这次判断）；
    转角：位移就是转角差，噪声是静止段转角标准差。
    """
    import math as _math

    if len(measurement.signal) < min_samples or len(measurement.noise) < min_samples:
        return None
    signal_mean = _mean_point(measurement.signal)
    noise_mean = _mean_point(measurement.noise)
    noise_std = _std_point(measurement.noise)
    if measurement.kind == "rotation":
        value = float(signal_mean - noise_mean)
        sigma = float(noise_std)
        if sigma <= 1e-12:
            return (value, 0.0, float("inf") if abs(value) > 0 else 0.0)
        return (value, sigma, abs(value) / sigma)
    dx = float(signal_mean[0] - noise_mean[0])
    dy = float(signal_mean[1] - noise_mean[1])
    magnitude = _math.hypot(dx, dy)
    sigma_x = float(noise_std[0])
    sigma_y = float(noise_std[1])
    if magnitude <= 1e-12:
        # 没有位移时噪声取两个方向的最大值（保守）。
        return (0.0, max(sigma_x, sigma_y), 0.0)
    cos = dx / magnitude
    sin = dy / magnitude
    sigma = _math.sqrt((sigma_x * cos) ** 2 + (sigma_y * sin) ** 2)
    if sigma <= 1e-12:
        return (magnitude, 0.0, float("inf"))
    return (magnitude, sigma, magnitude / sigma)


def probe_direction_deg(measurement: ProbeMeasurement) -> float | None:
    """这次探针的**图像运动方向**（度，0 = 图像 +x）。"""
    if measurement.kind != "translation":
        return None
    if len(measurement.signal) < 1 or len(measurement.noise) < 1:
        return None
    signal_mean = _mean_point(measurement.signal)
    noise_mean = _mean_point(measurement.noise)
    return math_degrees(
        float(signal_mean[0] - noise_mean[0]), float(signal_mean[1] - noise_mean[1])
    )


def segment_phase_window(
    segment: SegmentVision, label: str
) -> tuple[float, float] | None:
    """从采集时记下的阶段边界里取出某个阶段的时间窗（本段时间轴，秒）。"""
    for entry in segment.phases:
        if entry.get("label") == label:
            try:
                return (float(entry["start_s"]), float(entry["end_s"]))
            except (KeyError, TypeError, ValueError):
                return None
    return None


def np_as_uint8(frame: Any) -> Any:
    import numpy as np

    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def np_as_array(points: Any) -> Any:
    import numpy as np

    return np.asarray(points, dtype=np.float64)


def math_degrees(dx: float, dy: float) -> float | None:
    import math

    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angle_difference(a: float, b: float) -> float:
    """两个角度之间的最小夹角（度，0～180）。"""
    difference = (float(a) - float(b)) % 360.0
    return difference if difference <= 180.0 else 360.0 - difference


def load_states(path: Path) -> list[dict[str, Any]]:
    from .analysis import load_robot_states

    return load_robot_states(path)


#: 交付说明里要写清楚的"哪些模块是被复用代码原样搬过来的"。
REUSED_MODULE_NOTE: dict[str, Any] = {
    "重点复用（一行未改，在 src/sj_pretest/vendor/）": [
        "camera.py —— 海康相机采集、RAW 落盘格式、缺帧记账、棋盘格检测与亚像素细化、"
        "rigid motion 估计",
        "robot.py —— RTDE 连接、Dashboard 安全检查、停止、脚本收尾",
        "analyze.py —— 离线分析里可复用的统计与绘图工具",
        "calibration.py —— 棋盘格标定相关工具",
        "config.py —— 被复用代码的参数面（本工具通过 src/sj_pretest/bridge.py "
        "在运行时改属性，不改文件）",
    ],
    "新写的（本工具特有）": [
        "关节空间 moveJ 微动与停稳判据（被复用代码只发过 moveL）",
        "相位感知的采集（运动前/运动/保持/回程/运动后五段在同一段时间轴上）",
        "三层分析与步长推荐判据",
        "三个主按钮的一体化界面与逐步人工确认",
    ],
}
