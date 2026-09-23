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
    check_group_disk,
    check_plan_disk,
    group_peak_gb,
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
    remove_raw_artifacts,
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
from .vision import (
    VISION_JSON_NAME,
    SegmentVision,
    estimate_depth_in_plane,
    judge_in_plane_dominant,
    load_segment_vision,
    process_segment,
    save_vision_json,
)

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
    def steps(self) -> list[PlannedStep]:
        """本段包含的那一到两个计划步。

        有它之后，``config.estimate_capture_seconds`` / ``plan_disk_gb`` /
        ``check_plan_disk`` 这些"按计划估时长、估磁盘"的函数可以直接吃
        ``SegmentPlan``（它们取步的方式就是 ``getattr(plan, "steps", plan)``），
        不必再为"段"和"计划"维护两套估算路径——两套口径迟早会不一致。
        """
        return [self.primary] + ([self.follow] if self.follow is not None else [])

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
        #: ROI/棋盘格检查时落下的裁剪预览图（和 RAW 同一份 ROI）。
        self.roi_check_previews: list[Path] = []
        self._confirmed: set[str] = set()
        self._abort_reason: str | None = None
        self._opened = False

        # -- 分组流水线（需求一） --------------------------------------------
        #: 本组已经采完、还等着处理的段。整组走完、机械臂停稳之后才一起处理。
        self._pending: list[SegmentCapture] = []
        #: 当前组的名字（日志和事件里用它说清"盘上现在是哪一组"）。
        self._group_name: str | None = None
        #: 就地处理时算出来的逐帧结果，按段编号存着。
        #: ★ 这是"RAW 删掉之后不要再去读 frames.raw"（需求二）的落地处：
        #: 快速几何检查等下游直接用这一份，而不是重新 process_segment 一遍。
        self._released_vision: dict[str, SegmentVision] = {}

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
        # 回放模式下机器人不发任何命令，它只回答"这一刻关节角是多少"。
        # 答案来自这份历史数据所属运行目录里的 robot_states.csv；
        # 找不到就传空表，回放会如实报"没有关节角记录"，而不是编一组角出来。
        records: list[Any] | None = None
        if config.mode == "replay" and config.replay_source:
            from .replay import load_joint_records

            records = list(load_joint_records(config.replay_source))
            self.hooks.log(
                f"回放：从 {config.replay_source} 所在的运行目录读到 "
                f"{len(records)} 条关节角记录。"
                + ("" if records else "（没有记录，第一层分析会标为无法评估。）")
            )
        self.robot = make_robot(
            config,
            world=self.bundle.world,
            records=records,
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
        # ★ 需求一：会话收尾时必须把还没处理的那一组处理掉。
        # 顺序有讲究——先关记录器（把 robot_states.csv 落完），再处理，
        # 否则校验去读 RTDE 状态流会少最后几行，误判成"缺 RTDE 行"。
        # 已经中止的会话不在这里动手：那是"等人处理"的状态，RAW 一律保留。
        # 也收没有待处理段的"空组"：分组流水线关着的时候组头只会被记下来，
        # 不在这里收口就会在事件流里留下配不上对的 group_started。
        if (self._pending or self._group_name is not None) and not self.aborted:
            try:
                self.flush_group(reason="会话收尾")
            except Exception as exc:
                self.hooks.log(
                    f"[分组] 会话收尾时处理最后一组失败，RAW 全部保留：{exc}"
                )
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

    def approach_needed(self, tolerance_deg: float = 0.02) -> bool:
        """当前实际角与实验姿态差得远不远。只读实际关节角，不发任何命令。

        阈值取得很小（默认 0.02°）：这里是"要不要再走一遍到位过程"的判断，
        宁可多走一遍，也不要在其实没到位的时候以为已经站在实验姿态上了。
        返回 True 就说明**需要**逐步走到位。
        """
        if self.robot is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        current = self.robot.read_state().actual_q_deg
        delta = joint_delta(current, self.config.robot.nominal_joint_deg)
        biggest = max(abs(float(value)) for value in delta)
        self.hooks.log(
            f"当前姿态与实验姿态的最大偏差：{biggest:.4f}°"
            f"（超过 {float(tolerance_deg):g}° 就需要重新逐步到位）"
        )
        return biggest > float(tolerance_deg)

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
                f"{step.label}；到位后相对实验姿态 "
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
            preview, check, roi_lines = self._preview_and_check(
                packet.frame, step.event.event_id, index
            )
            if preview is not None:
                # 记下来：到位过程一共落了哪几张预览图，"现场回头看"时要有据可查。
                self.approach_frames.append(preview)
            result.add(f"第 {index} 步画面检查：{check}")
            self.hooks.log(f"第 {index} 步画面检查：{check}")
            if preview is not None and self.hooks.on_preview is not None:
                # ★ 界面上除了一张**裁剪后**的真实预览图，还要同时给出
                # 原始画面尺寸 / ROI 坐标 / 裁剪后尺寸这三个数（需求三）：
                # 只看裁剪后的图，人没法判断 ROI 坐标是不是写歪了。
                note = f"到位 {index}/{len(plan.steps)}（预览图 = RAW 里实际存下的裁剪画面）"
                if roi_lines:
                    note = note + "\n" + "\n".join(roi_lines)
                self.hooks.on_preview(preview, note)
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
    ) -> tuple[Path | None, str, list[str]]:
        """把预览图写盘并做一次轻量画面检查。

        检查内容：88 个内角点是否全部检出、角点到图像边缘的余量够不够。
        这两件事只用**一张**帧，所以耗时是"一次识别"（约 150 ms），
        不会拖慢采集——因为此刻并没有在采集。

        ★ 这里的裁剪**必须**和 RAW 落盘用同一份 ROI（``engine.crop``）：
        预览图里看到的画面 = RAW 里真正存下的画面 = 离线识别要处理的画面。
        以前预览用的是**未裁剪**的整幅图，于是"预览里棋盘格好好的、
        离线分析却在裁过的图里找不齐角点"这种事会在现场发生。

        返回 ``(预览图路径 或 None, 检查结论文字, ROI 三行说明)``。
        第三项是给界面直接显示的（需求三：原始画面尺寸 / ROI 坐标 / 裁剪后尺寸），
        和日志里 ``[ROI]`` 那几行是同一份措辞。
        """
        import cv2  # 局部导入：只有真要落图时才需要

        config = self.config
        preview_path: Path | None = None
        cropped, roi_lines, roi_ok = self._crop_with_roi(frame)
        if cropped is None:
            text = "未通过——ROI 越界，无法给出预览：" + "；".join(roi_lines)
            self.hooks.log(f"[ROI] {text}")
            if self.run is not None:
                self.run.events.write(
                    "preview_check", index=index, event_id=event_id, ok=False, note=text
                )
            return None, text, roi_lines
        if self.run is not None:
            preview_path = (
                self.run.root / "approach_previews" / f"{index:02d}_{safe_name(event_id)}.png"
            )
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            # 存的就是裁过的图（和 RAW 一致）。
            cv2.imwrite(str(preview_path), np_as_uint8(cropped))
        corners_note, margin_ok = self._check_board_frame(cropped)
        verdict = "通过" if margin_ok else "未通过"
        text = (
            f"{verdict}——{corners_note}"
            f"（要求内角点 {config.camera.board_inner_corners[0]}×"
            f"{config.camera.board_inner_corners[1]} 全部检出，"
            f"并留出 ≥ {config.camera.min_margin_px} px 余量）"
            + ("" if roi_ok else "；ROI 越界：" + "；".join(roi_lines))
        )
        for line in roi_lines:
            self.hooks.log(f"[ROI] {line}")
        if self.run is not None:
            self.run.events.write(
                "preview_check", index=index, event_id=event_id, ok=margin_ok, note=text,
                roi=roi_lines,
            )
        return preview_path, text, roi_lines

    def _crop_with_roi(self, frame: Any) -> tuple[Any | None, list[str], bool]:
        """按 RAW 的那一份 ROI 裁一帧，并给出"原始尺寸/ROI/裁剪后尺寸"三行说明。

        返回 ``(裁剪后的帧 或 None, 中文说明行, 是否成功)``。
        越界时返回 None 而不是抛异常：调用方要在界面上把**为什么**说清楚，
        而不是丢一个栈。
        """
        if self.engine is None:
            return frame, [], True
        report = self.engine.roi_report(frame)
        lines = list(CaptureEngine.roi_report_lines(report))
        if not report.get("ok", True):
            return None, lines, False
        try:
            return self.engine.crop(frame), lines, True
        except Exception as exc:  # pragma: no cover - roi_report 已经先拦过一遍
            lines.append(f"裁剪失败：{exc}")
            return None, lines, False

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
        # ★ 按钮二的第一件事：在**任何采集开始之前**把 ROI 和棋盘格检查一遍。
        # 这一步过了，后面每一段用的画面才和预览里看到的是同一张。
        self.require_roi_and_board("静态基线采集")
        # ★ 需求一：静态基线**单独一组**——它本来就是"机械臂完全不动"的一段，
        # 自己成组最省事：采完立刻处理掉，盘上不留任何 RAW。
        self.begin_group("静态基线", [plan], what="静态基线采集")
        self.hooks.log(
            f"[静态基线] 保持不动录 {duration:.1f} s："
            "这段时间里不要碰相机、台面和机械臂。"
        )
        if self.recorder is not None:
            self.recorder.mark(event_id=step.event.event_id, stage="static", label="静态基线")
        record = self.capture_static(
            segment_id=step.event.event_id, duration_s=duration
        )
        self.flush_group(reason="静态基线录完")
        return record

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
        # ★ 静态基线是**单独一组**（需求一）：这里只登记，处理在组边界上做。
        # 顺序：先登记归属（没有显式分组时会自动开一个组），再写 segment_captured——
        # 这样事件流里"哪一段属于哪一组"按顺序就能读出来，不用事后猜。
        self.stage_segment(record)
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

    def _require_disk(self, plans: Sequence[Any], what: str) -> list[str]:
        """开跑之前，按**这一次要跑的计划**检查磁盘；不够就拒绝开始这一段。

        为什么不是"连设备时查一次就完了"：连设备时只知道一个固定门槛，
        而真正的占用取决于"这一段要录几分钟、每帧多大"。而且一段正式实验
        是几十分钟，中途被别的进程写满磁盘是很现实的事——所以每一段开始前
        都按那一段自己的计划重新估一次，宁可多花一次系统调用。

        ``plans`` 传计划对象（有 ``steps``）或步骤序列都行。
        """
        ok, lines = check_plan_disk(self.config, plans)
        for line in lines:
            self.hooks.log(f"[磁盘] {line}")
        if self.run is not None:
            self.run.events.write("disk_checked", what=what, ok=bool(ok), lines=lines)
        if not ok:
            raise ExperimentError(
                f"{what}没有开始：磁盘空间不够。\n" + "\n".join(lines)
            )
        return lines

    # ------------------------------------------------------------------
    # ROI / 棋盘格：开始任何一段采集之前的统一检查
    # ------------------------------------------------------------------

    def require_roi_and_board(self, what: str, *, require_board: bool = True) -> list[str]:
        """抓一帧，按**和 RAW 完全相同**的 ROI 检查这件事能不能开始。

        检查两件事，任何一件不过就**拒绝开始这一段**（抛 :class:`ExperimentError`）：

        1. ROI 是否越界（``camera.roi`` 的 x+w、y+h 是否落在原始画面内）；
        2. 裁剪**之后**的图里，棋盘格内角点是否齐全、边缘余量是否够。

        第 2 条必须按裁剪后的图判：人在预览里看到的是裁过的图，离线识别处理的
        也是裁过的图，如果拿未裁剪的整幅图去判"棋盘格完整"，就会出现
        "检查说没问题、跑起来识别不到角点"。

        ``require_board=False`` 只查 ROI（正式实验中途复查用，那时棋盘格必须仍然完整，
        所以实际上还是 True；留这个参数是为了静态基线之前的检查可以只报不拦）。
        """
        if self.engine is None or self.run is None or self.pump is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        packet = self.pump.drain_frames(1, stop_requested=self.hooks.stop_requested)
        frame = packet.frame
        cropped, roi_lines, roi_ok = self._crop_with_roi(frame)
        lines = list(roi_lines)
        board_ok = True
        if cropped is not None and require_board:
            corners_note, board_ok = self._check_board_frame(cropped)
            config = self.config
            lines.append(
                f"裁剪后棋盘格检查：{'通过' if board_ok else '未通过'}——{corners_note}"
                f"（要求内角点 {config.camera.board_inner_corners[0]}×"
                f"{config.camera.board_inner_corners[1]} 全部检出，"
                f"余量 ≥ {config.camera.min_margin_px} px）"
            )
        ok = bool(roi_ok and board_ok)
        for line in lines:
            self.hooks.log(f"[ROI] {line}")
        # ★ 需求三：界面上除了日志里的三个尺寸，还要给出**真实的裁剪后预览**。
        # 这里落的图就是刚才用来判棋盘格的那张裁剪图，和 RAW 里存的是同一个裁剪，
        # 所以"界面上看到的" = "检查用的" = "RAW 里有的"。
        preview_path = self._write_roi_check_preview(cropped, what)
        if preview_path is not None and self.hooks.on_preview is not None:
            self.hooks.on_preview(
                preview_path,
                f"{what}：ROI 与棋盘格检查（{'通过' if ok else '未通过'}）\n"
                + "\n".join(lines),
            )
        if self.run is not None:
            self.run.events.write(
                "roi_checked", what=what, ok=ok, lines=lines, board_checked=require_board,
                preview=self.run.relative(preview_path) if preview_path else None,
            )
        if not ok:
            raise ExperimentError(
                f"{what}没有开始：ROI 或棋盘格检查未通过。\n" + "\n".join(lines)
                + "\n（ROI 越界请改小/移动 camera.roi；棋盘格不完整请调整相机或棋盘格位置后重试。）"
            )
        return lines

    def _write_roi_check_preview(self, cropped: Any, what: str) -> Path | None:
        """把"这次检查到底看了哪块画面"存成一小张 PNG，并回给界面显示。

        存图失败（盘满、编码器异常）**不算检查失败**——检查的结论是数值判出来的，
        图只是给人看的佐证；这里如实写一条日志然后继续。
        """
        if cropped is None or self.run is None:
            return None
        index = len(self.roi_check_previews) + 1
        path = self.run.root / "roi_checks" / f"{index:03d}_{safe_name(what)}.png"
        try:
            import cv2

            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np_as_uint8(cropped))
        except Exception as exc:
            self.hooks.log(f"[ROI] 预览图存盘失败（不影响本次检查结论）：{exc}")
            return None
        self.roi_check_previews.append(path)
        return path

    # ------------------------------------------------------------------
    # 按钮二（B）：快速几何检查
    # ------------------------------------------------------------------

    def run_quick_probes(self) -> tuple[SessionResult, list[str]]:
        """每个关节正负各走一个试探步长（默认 0.2°），然后抽样识别几帧做几何确认。"""
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
        self.require_roi_and_board("快速几何检查")
        result = SessionResult(title="快速几何检查")
        failed_joints: list[str] = []
        plans = iter_segment_plans(plan)
        triggers = self._joint_triggers(plans)
        # ★ 需求一：快速几何检查**单独一组**（它不进统计，纯放行用）。
        self.begin_group("快速几何检查", [plan], what="快速几何检查")
        captured: list[tuple[SegmentCapture, SegmentPlan]] = []
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
            captured.append((record, segment_plan))

        # 一组动作全部走完、机械臂停稳 → 处理这一组 → 校验 → 删 RAW（需求一）。
        self.flush_group(reason="快速几何检查动作全部走完")

        # ★ 需求二：量的时候**不再去读 frames.raw**——上面 flush 之后它已经删了。
        # _measure_probe 走 _segment_vision()：先看内存里刚算的那一份，
        # 没有就读 segment_vision.json + 角点 CSV（离线复算走的就是这条路）。
        measurements: list[ProbeMeasurement] = []
        for record, segment_plan in captured:
            measurements.append(self._measure_probe(record, segment_plan))

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

    def _segment_vision(self, record: SegmentCapture) -> SegmentVision:
        """拿一段的逐帧结果，**尽量不碰 frames.raw**（需求二）。

        三条路，按"离得最近"排序：

        1. 本会话内存里刚算出来的那一份（``process_and_release`` 处理完就留着），
           分组流水线正常跑完就是这条路；
        2. 从段目录里 **回读** ``segment_vision.json`` + 角点 CSV——RAW 删掉之后
           复算走的就是这条路，所以它必须是可用的，不能只是"理论上存在"；
        3. 只有前两条都不成立（没开边采边清、还没处理过）才现算。

        以前这里无条件第 3 条：删了 RAW 之后跑快速几何检查会直接报
        "找不到 frames.raw"。现在第 1、2 条覆盖了那个场景。
        """
        cached = self._released_vision.get(record.segment_id)
        if cached is not None:
            return cached
        config = self.config
        if not bool(config.paths.delete_raw_after_process):
            # 没开边采边清：RAW 还在，按老规矩现算（抽帧即可，不删数据）。
            return process_segment(
                record.dir,
                config=config,
                segment_id=record.segment_id,
                save_corners=False,
                stride=QUICK_PROBE_STRIDE,
            )
        corners_path = (
            self.run.corners_dir / f"{safe_name(record.segment_id)}.csv"
            if self.run is not None
            else record.dir / f"{safe_name(record.segment_id)}.csv"
        )
        try:
            loaded = load_segment_vision(
                record.dir, segment_id=record.segment_id, corners_path=corners_path
            )
        except Exception as exc:
            raise ExperimentError(
                f"{record.segment_id}：RAW 已删除，回读 segment_vision.json / "
                f"角点表也失败（{exc}）。这一段没法再复算——"
                "这正是删除前校验要拦住的情况，请把段目录留好交给人工处理。"
            ) from exc
        # 回读出来的一份也缓存住：快速几何检查可能对它取好几个窗口。
        self._released_vision[record.segment_id] = loaded
        return loaded

    def _measure_probe(
        self, record: SegmentCapture, segment_plan: SegmentPlan
    ) -> ProbeMeasurement:
        """抽样离线识别一段快速探针，量出"这个方向动了多少"。

        抽样的时间窗取三段：运动前（pre_motion）、保持（hold）、运动后（post_motion）。
        pre 和 post 都是"回到名义位姿的静止段"，把它们并起来当噪声池；
        hold 是"走完试探步长停稳后"的静止段，拿它跟噪声池比，得到信噪比。
        这样不需要额外的静态基线，也不会拿"首末帧"去比——首末帧一个是运动前、
        一个是回程之后，两个都在名义位姿，位移自然是零（这是最初版本的错误）。

        ★ 需求二：这一段的逐帧结果**不再从 frames.raw 现算**。分组流水线跑完
        就已经把它处理成 ``segment_vision.json`` + 角点 CSV 了，RAW 也已经删掉；
        这里走 :meth:`_segment_vision` 拿结果。只有"没开边采边清、RAW 还在"
        的情况下才回落到现算——那条路上 RAW 本来就还在，没有重复读取的问题。
        """
        config = self.config
        segment = self._segment_vision(record)
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
            # ★ 轴向/面内分解（需求四）：用棋盘格相似变换的尺度变化估轴向位移，
            # 用质心二维位移估面内位移。噪声池（pre+post 两段静止帧）同时给了
            # "尺度噪声"——判据要知道自己分辨不分辨得出来。
            depth_in_plane=estimate_depth_in_plane(
                signal,
                config=config,
                noise_frames=noise_frames,
                rotation_joint=rotation_joint,
            ),
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
        probe_deg = float(config.pretest.quick_probe_deg)
        lines: list[str] = []
        ok = True
        rotation_joint = joint in set(config.vision.rotation_joints)

        for measurement in measurements:
            tag = "正向" if measurement.direction > 0 else "负向"
            lines.append(
                f"{tag} {probe_deg:g}°（{measurement.label}）："
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
                    f"这个关节的 {probe_deg:g}° 在画面上几乎看不出来（可能是正常物理现象，"
                    "也可能是关节/摆放不对，需要人工判断）。"
                )
            if ratio < 5.0:
                lines.append(f"    {tag}：信噪比 {ratio:.1f} 偏低，方向结论仅供参考。")

            # ★ 需求四的"以面内运动为主"：这一条以前只写在文档里，没实现。
            # 判据用的是深度位移的**上限**比值，见 vision.judge_in_plane_dominant。
            estimate = measurement.depth_in_plane
            if estimate is None:
                ok = False
                lines.append(
                    f"    {tag}：这一趟没算出轴向/面内分解，无法判断是否以面内运动为主。"
                )
                continue
            allowed, why = judge_in_plane_dominant(
                estimate, max_depth_ratio=float(config.thresholds.max_depth_ratio)
            )
            lines.append(f"    {tag} 轴向/面内：{why}")
            for line in estimate.to_lines():
                lines.append(f"        {line}")
            if not allowed:
                ok = False

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
        if bool(config.paths.delete_raw_after_process):
            source = (
                "这一段用的是删 RAW 之前**逐帧**算好并落盘的结果"
                "（步长 1，每一帧都算过）"
            )
        else:
            source = f"这是**抽样**检查（每 {QUICK_PROBE_STRIDE} 帧抽 1 帧）"
        lines.append(
            source + "，只用来放行/暂停；正式结论在按钮三的离线分析里。"
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
        # ★ 需求一：预实验**每个关节一组**。计划本身就是按关节排的（见
        # joint_space.build_pretest_plan 的循环顺序），所以这里按关节切片即可；
        # 切片算的是"这一组要录多久、要占多少盘"，用来做分组磁盘闸门。
        by_joint: dict[Any, list[SegmentPlan]] = {}
        for item in plans:
            by_joint.setdefault(item.primary.event.joint, []).append(item)
        triggers = self._joint_triggers(plans)
        active: Any = None
        try:
            for position, segment_plan in enumerate(plans, start=1):
                self._check_stop()
                joint = segment_plan.primary.event.joint
                if joint != active:
                    if trigger := triggers.get(position):
                        # ★ 需求五：每个关节的成组动作开始前，对**实际 ROI** 重新检查一次
                        # 棋盘格。上一次检查可能是十几分钟前的事，中间人可能碰过相机、
                        # 机械臂也可能已经动过了——隔了这么多动作还拿旧结论当依据不算数。
                        self.require_roi_and_board(f"预实验 {joint} 的成组动作")
                        if not self.hooks.ask(trigger):
                            raise MotionAborted(f"操作者没有确认进入 {joint} 的预实验。")
                        # 同上：关节级确认就是本关节其余动作的通行证。
                        if joint:
                            self._confirmed.add(joint)
                    # 换关节 = 换组：上一组在这里收尾（处理→校验→删 RAW），
                    # 本组再按自己的计划过磁盘闸门。
                    self.begin_group(
                        f"预实验 {joint}", by_joint.get(joint, []), what=f"预实验 {joint}"
                    )
                    active = joint
                record = self._run_segment(
                    segment_plan, kind="pretest", position=position, total=len(plans)
                )
                if record is None:
                    continue
                result.segments.append(record)
                self._record_trial(segment_plan, record)
                self.hooks.log(record.summary_line())
            # 最后一个关节的那一组也要收尾，不能留在盘上。
            self.flush_group(reason="预实验最后一个关节走完")
            result.add(f"预实验采集完成：{len(result.segments)} 段。")
        finally:
            # 中止时也要落盘。已经采到的段必须登记进 trial_plan：
            # 采集目录里 RAW、时间戳、元数据都还在，但没有 trial_plan
            # 分析侧就不知道哪些目录算统计试验——原始数据没丢，却用不上。
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
        # ★ 需求五：组 A 和组 B **都要**先做理论范围检查，不能只查 A。
        # 放在这里而不是只放界面上，是为了让命令行、脚本、任何入口都绕不过去。
        if bool(config.formal.require_range_check):
            for line in self.formal_range_checks(steps):
                result.add(line)
                self.hooks.log("[范围检查] " + line)
        else:
            self.hooks.log(
                "[范围检查] 配置里关掉了 formal.require_range_check；"
                "本次没有做理论范围检查（碰撞状态仍然是 unknown）。"
            )
        try:
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
                # ★ 需求五：每个关节成组动作开始前，对实际 ROI 重新检查棋盘格。
                self.require_roi_and_board(f"组{group.upper()} {joint} 正式实验")
                if not self.hooks.ask(
                    f"[组{group.upper()}] 开始 {joint} 的正式实验 Δ={step}°。\n\n"
                    "理论范围检查已通过，但碰撞状态仍是 unknown。"
                ):
                    raise MotionAborted(f"操作者没有确认开始 组{group.upper()} {joint}。")
                self._confirmed.add(joint)
                # ★ 需求一：正式实验**"一个关节的组A" 或 "一个关节的组B" 各一组**。
                # begin_group 会先把上一组收尾（处理→校验→删 RAW），
                # 再按本组自己的计划过磁盘闸门——正式实验一组就是几十分钟、
                # 十几 GB，前一组没删干净的时候查磁盘等于没查。
                self.begin_group(
                    f"组{group.upper()} {joint}",
                    plans,
                    what=f"组{group.upper()} {joint} 正式实验",
                )
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
                # 本关节这一组走完，机械臂停在名义位姿，就地处理掉。
                self.flush_group(reason=f"组{group.upper()} {joint} 动作全部走完")
            result.add(f"组{group.upper()} 采集完成：{len(result.segments)} 段。")
        finally:
            # 同 run_pretest：中途中止也要把已采到的段登记进 trial_plan。
            self._write_trial_plan()
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
                # ★ 没停稳就地停住 + 掐断整段（需求二）。
                # 以前这里只写一行日志就继续 hold → return → 下一条，
                # 等于把"还在动"的画面当成到位值采下来，还在关节运动时又发新命令。
                on_timeout=lambda: self._on_settle_timeout(event_id, robot),
                abort_on_timeout=True,
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
                        on_timeout=lambda: self._on_settle_timeout(
                            follow.event.event_id if follow is not None else event_id,
                            robot,
                        ),
                        abort_on_timeout=True,
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
        # ★ 分组流水线（需求一）：这一段先登记归属，**不在**这里处理。
        # 组内所有动作走完、机械臂真的停住之后，由 flush_group 整组处理、
        # 校验通过再删这一组的 RAW。（停稳超时那一段上面已判失败，
        # process_and_release 会保留它的 RAW 给人工检查。）
        # 顺序：先登记再写 segment_captured，事件流按顺序就能读出组的归属。
        self.stage_segment(record)
        if self.run is not None:
            self.run.events.write(
                "segment_captured",
                **{k: v for k, v in record.to_dict().items() if k != "phases"},
            )
        if record.aborted_by_timeout:
            # 采集层已经在超时那一刻 stopJ 并掐断了整段；这里负责把"这一段失败"
            # 传递出去，让编排层立刻停手（不再 hold / 不回程 / 不下一条）。
            self._fail_segment_on_settle_timeout(event_id, record)
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

    def _on_settle_timeout(self, event_id: str, robot: JointRobotPort) -> None:
        """等停稳超时的**第一反应**：立刻 stopJ，并把这件事写进事件流。

        这一条必须在发现超时的那一刻执行（采集层保证），不能在整段收尾时补做。
        """
        robot.stop()
        if self.run is not None:
            self.run.events.write(
                "settle_timeout", event_id=event_id, action="stopJ"
            )
        self.hooks.log(
            f"{event_id}：等待停稳超时（到时间还在动）——已就地请求 stopJ 停止。"
        )

    def _fail_segment_on_settle_timeout(
        self, event_id: str, record: SegmentCapture
    ) -> None:
        """把"停稳超时"这一段判为失败，并**中止整个会话**。

        为什么是中止会话而不是"跳过这一条继续"：关节没在预期时间内停住，
        可能的原因（负载、摩擦、限位、碰撞、控制异常）都不是"这条数据不要了"
        就能带过的；而且此刻机器人不在名义位姿上，继续下一条动作就是在未知状态下
        叠加下一次运动。所以这里的行为和中止完全一致：
        不再发任何命令、**不自动回位**、数据全部保留、由人去现场检查。

        （需求二：不继续 hold、不 return、不下一条；RAW / RTDE / 时间戳全部保留。）
        """
        minutes = float(self.config.robot.settle_timeout_s)
        message = (
            f"{event_id}：等待停稳超时（{minutes:g} s 内没停住），本段判为失败。\n"
            f"已做的处理：立刻 stopJ；这一段的 RAW、RTDE 状态流和时间戳**已全部保留**"
            f"（{record.dir}）。\n"
            "没有做的处理：没有继续 hold、没有发回程、**没有自动回位**，"
            "后面的动作一条都不会再发。\n"
            "请人工检查：关节为什么没停住（负载/摩擦/限位/碰撞/控制异常），"
            "示教器上确认姿态与安全状态，处理完再决定要不要重跑这一段。"
        )
        if self.run is not None:
            self.run.events.write(
                "segment_failed",
                event_id=event_id,
                reason="settle_timeout",
                kept_dir=str(record.dir),
                frame_count=int(record.frame_count),
                aborted_by_timeout=True,
            )
            self.run.note(f"[停稳超时] {event_id}：{message}")
        self.hooks.log(message)
        self.abort(f"{event_id} 等待停稳超时，需人工检查（不会自动回位）")

    # ------------------------------------------------------------------
    # 边采边清：一段结束就地处理，校验通过后删掉 RAW
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 分组流水线（需求一）：一组动作走完 → 整组处理 → 校验 → 删 RAW → 下一组
    # ------------------------------------------------------------------

    def begin_group(
        self, name: str, segments: Sequence[Any], *, what: str | None = None
    ) -> None:
        """进入一组新的动作：先把上一组收尾，再按**本组**的计划查磁盘。

        流程（需求一原文）：
        采集一组动作 RAW → 机械臂保持停止 → 完整处理该组 →
        保存逐帧角点/质心/二维转角/尺度/时间戳/RTDE 和分析结果 → 回读校验 →
        校验通过后删除该组 RAW → 再进入下一组运动。

        组的划分：
        * 静态基线**单独**一组；
        * 快速几何检查**单独**一组；
        * 三档预实验**每个关节**一组；
        * 正式实验**"一个关节的组A" 或 "一个关节的组B"** 各一组。

        ★ 顺序很重要：**先 flush 上一组，再查本组磁盘**。反过来的话，
        磁盘检查看到的可用空间里还含着上一组没删的 RAW，会算出"够"，
        然后删完上一组才发现不够——那时本组已经在飞了。
        """
        self.flush_group(reason=f"进入「{name}」之前收尾上一组")
        label = what or name
        # 磁盘闸门**无论开关在不在都要过**：开着的时候卡的是"本组峰值 ≤ 硬上限"，
        # 关着的时候卡的是"这一组要写的量 + 绝对下限"够不够。
        self._require_group_disk(segments, label)
        self._group_name = str(name)
        if self.run is not None:
            self.run.events.write(
                "group_started",
                group=str(name),
                segments=len(list(segments)),
                release_enabled=bool(self.config.paths.delete_raw_after_process),
            )
        self.hooks.log(
            f"[分组] 开始「{name}」：{len(list(segments))} 段；"
            + (
                "本组全部走完后整组处理、校验通过再删这一组的 RAW。"
                if self.config.paths.delete_raw_after_process
                else "（分组流水线未打开，本轮 RAW 全程保留。）"
            )
        )

    def stage_segment(self, record: SegmentCapture) -> None:
        """本段采完了，先攒着——等整组动作走完、机械臂停稳再一起处理。

        为什么不在每段结束时立刻处理：现场要的是"一个动作组结束就停下来处理"，
        处理期间机械臂必须真的停着。按段处理会把"停机时间"切碎成几十次，
        而且组内每段处理完就删，峰值并没有降下来（峰值本来就是整组驻留）。
        按组处理只有一个停机窗口，人也只需要等一次。
        """
        if not bool(self.config.paths.delete_raw_after_process):
            return
        if record.aborted_by_timeout or record.stopped_early:
            # ★ 判失败/提前停止的段**不进组**：它们的 RAW 一律保留给人工检查，
            # 而且下落要**立刻**落账（写 raw_kept），不能只挂在内存里——
            # 这时候会话通常已经中止，后面不会再有 flush 来替它们记账。
            self.process_and_release(record)
            return
        self._ensure_group_for(record)
        self._pending.append(record)

    def _ensure_group_for(self, record: SegmentCapture) -> None:
        """给"没有显式 ``begin_group`` 就采到的段"补一个组头。

        正常流程里每个 runner 都会先声明组（静态基线 / 快速几何检查 /
        每个关节的预实验 / 正式实验的每个组A、组B）。但界面上的单个按钮
        可以被单独按下去，脚本也可以分步调，那就没有组边界了。
        补一条 ``group_started`` 是为了让事件流保持一条不变式：
        **每一段采集都隶属于某个已声明的组**，事后按组对账不会漏掉任何一段。
        """
        if self._group_name is not None:
            return
        name = f"自动分组（{record.kind or '未分类'}）"
        self._group_name = name
        if self.run is not None:
            self.run.events.write(
                "group_started",
                group=name,
                segments=None,
                automatic=True,
                release_enabled=bool(self.config.paths.delete_raw_after_process),
            )
        self.hooks.log(
            f"[分组] {record.segment_id} 没有显式分组，归入「{name}」"
            "（段一结束就地处理）。"
        )

    def flush_group(self, *, reason: str = "") -> list[dict[str, Any]]:
        """把本组攒下的段全部处理掉：处理 → 校验 → 删 RAW。

        任何一段校验不过就**保留它的 RAW 并暂停整个会话**（需求三）：
        处理管线一旦出问题（相位缺帧、时间戳缺、回读对不上），
        继续往下跑只会让更多段落在同样的坏状态里，
        而且此刻机器人正停在组边界上，正是停下来最安全的时刻。
        """
        run = self.run
        group = self._group_name
        if not self._pending:
            # 分组流水线没打开时 ``_pending`` 永远是空的，但组边界仍然要**成对收口**：
            # 事件流里只剩一串没有 ``group_finished`` 的 ``group_started``，
            # 事后按组对账会以为这些组都没走完。这里只补一条收尾事件，不做任何处理。
            if group is not None and run is not None:
                run.events.write(
                    "group_finished",
                    group=group,
                    segments=0,
                    released=0,
                    kept=0,
                    release_enabled=bool(self.config.paths.delete_raw_after_process),
                )
            self._group_name = None
            return []
        pending = list(self._pending)
        self._pending.clear()
        group = group or "未命名组"
        if run is not None:
            run.events.write("group_processing", group=group, segments=len(pending))
        self.hooks.log(
            f"[分组]「{group}」采集结束，机械臂保持停止；就地处理这一组的 "
            f"{len(pending)} 段（{reason or '组边界'}）……"
        )
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        # RTDE 状态流每读一次要遍历整个 CSV，所以**一组只读一次**，
        # 给组内每段的校验共用（见 _verify_releasable 的 rtde_rows 参数）。
        rtde_rows = self._load_rtde_rows()
        for position, record in enumerate(pending, start=1):
            info = self.process_and_release(record, rtde_rows=rtde_rows)
            results.append(info)
            if info.get("kept"):
                failures.append(f"{record.segment_id}：{info.get('reason', '')}")
            self.hooks.log(
                f"[分组] {position}/{len(pending)} {record.segment_id}："
                + ("已处理并删除 RAW" if info.get("deleted") else str(info.get("reason")))
            )
        if run is not None:
            released = sum(1 for item in results if item.get("deleted"))
            run.events.write(
                "group_finished",
                group=group,
                segments=len(pending),
                released=released,
                kept=len(pending) - released,
            )
        # 本组到此为止：后面再采到的段属于新的一组，
        # 不能继续挂着上一组的名字（否则日志和事件会串组）。
        self._group_name = None
        if failures:
            message = (
                f"「{group}」里有 {len(failures)} 段**校验没通过，RAW 已保留**：\n"
                + "\n".join(f"  · {item}" for item in failures)
                + "\n按需求三，这里**暂停**，等人工处理："
                "先看段目录里的 capture_metadata.json 相位表和 frame_timestamps.csv，"
                "确认是采集出了问题还是只是这一段的相位不全，处理完再决定要不要重跑这一段。"
            )
            self.hooks.log(message)
            self.abort(message)
            raise MotionAborted(message)
        return results

    def _load_rtde_rows(self) -> list[dict[str, Any]]:
        """读一次 robot_states.csv，给本组所有段的校验共用。"""
        if self.run is None:
            return []
        path = self.run.root / "robot_states.csv"
        if not path.is_file():
            return []
        try:
            return load_states(path)
        except Exception as exc:  # pragma: no cover - 读不动就当没有，校验会如实报
            self.hooks.log(f"[分组] 读取 robot_states.csv 失败：{exc}")
            return []

    def _require_group_disk(self, segments: Sequence[Any], what: str) -> list[str]:
        """进入下一组之前的磁盘闸门：超硬上限或空间不够就**不得开始**。"""
        ok, lines = check_group_disk(self.config, segments, what=what)
        for line in lines:
            self.hooks.log(f"[磁盘] {line}")
        if self.run is not None:
            self.run.events.write(
                "disk_checked", what=f"{what}（分组）", ok=bool(ok), lines=lines,
                group=self._group_name,
            )
        if not ok:
            hint = (
                f"{what}没有开始：磁盘不允许。\n" + "\n".join(lines)
                + "\n（需求一：任意时刻 RAW + 临时文件不得超过 "
                f"{self.config.paths.max_peak_disk_gb:.0f} GB；"
                "超了就不得开始下一组。）"
            )
            # 磁盘不允许**不是**运行出错，是"这一步先别做"：如实记一笔然后**拒绝开始**。
            # 这里抛 ExperimentError 而不是中止会话：一条运动命令都还没发出去，
            # 人清完盘/改完 ROI 之后仍然可以继续用这个会话，不必整个重来。
            self.hooks.log(hint)
            if self.run is not None:
                self.run.note(f"[磁盘] {hint}")
            raise ExperimentError(hint)
        return lines

    def process_and_release(
        self, record: SegmentCapture, *, rtde_rows: Sequence[Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        """就地处理一段，然后把 RAW 删掉（只有 ``paths.delete_raw_after_process`` 打开时）。

        现场要解决的问题：一趟完整实验的原始帧是几百 GB，D 盘会被塞满。
        这里把它改成分组流水线：**一组动作全部走完、机械臂停稳之后**，
        整组一起离线识别，角点表和逐帧几何**先落盘并回读校验**，
        确认这一组已经可以复算了，才删除 frames.raw。这样任何一个时刻盘上
        只有"当前这一组"，而不是整场的总和（800×600 整场约 222 GB、
        单组峰值约 21 GB；950×800 整场约 352 GB、单组峰值约 34 GB）。

        四条不妥协的规矩：

        1. **校验不过就不删，而且暂停。** 处理失败、有效帧不够、必要相位缺帧、
           回读对不上、时间戳/RTDE 缺、本段基本分析跑不完，一律保留 RAW
           并写 ``raw_kept`` 事件，然后由 ``flush_group`` 中止整个会话等人处理。
        2. **删除留痕。** 删之前算 sha256 和字节数，删之后写 ``raw_deleted``
           事件（含处理步长、帧数、派生文件路径），事后可追。
        3. **判失败的段不删。** 停稳超时/提前停止的段是给人工检查用的，一律保留。
        4. **步长永远是 1。** 删 RAW 之前必须把每一帧都算过（见 config 里
           ``process_stride`` 的说明）；配置层已经把"删数据 + 抽帧"的组合拦掉了。

        返回一份说明用的字典（调用方拿它写日志）。
        """
        import hashlib

        config = self.config
        run = self.run
        if run is None:
            raise ExperimentError("会话还没打开：请先调用 open()。")
        raw_path = record.dir / "frames.raw"
        info: dict[str, Any] = {
            "segment_id": record.segment_id,
            "enabled": bool(config.paths.delete_raw_after_process),
            "processed": False,
            "deleted": False,
            "kept": False,
            "reason": "",
            "raw_bytes": 0,
            "raw_sha256": "",
            "stride": int(config.paths.process_stride),
        }
        if not info["enabled"]:
            return info
        if not raw_path.is_file():
            info["reason"] = "没有找到 frames.raw，跳过"
            return info
        if record.aborted_by_timeout or record.stopped_early:
            info["kept"] = True
            info["reason"] = (
                "这一段是被判失败/提前停止的（停稳超时或人工中止），"
                "原始帧留给人工检查，不删"
            )
            self._record_release(run, "raw_kept", info)
            self.hooks.log(f"[分组] {record.segment_id}：{info['reason']}")
            return info

        # ★ 需求四：删 RAW 之前**必须**是 132 Hz 逐帧结果，不能抽帧。
        # 配置层已经拦掉了"删数据 + 抽帧"的组合，这里再钉一次：真到了这里
        # 还拿到 >1 的步长，那是有人在代码里绕过了配置校验，宁可报错也不能
        # 悄悄按步长删掉那些没算过的帧。
        stride = int(config.paths.process_stride)
        if stride != 1:
            info["kept"] = True
            info["reason"] = (
                f"处理步长为 {stride}，而这一步会删除 RAW：抽帧删数据会永久丢掉"
                "被抽掉那些帧的原始像素。需求四要求删 RAW 前必须逐帧算完，"
                "所以拒绝删除，RAW 保留。"
            )
            self._record_release(run, "raw_kept", info)
            self.hooks.log(f"[分组] {record.segment_id}：{info['reason']}")
            return info

        digest = hashlib.sha256()
        with raw_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        info["raw_bytes"] = int(raw_path.stat().st_size)
        info["raw_sha256"] = digest.hexdigest()

        self.hooks.log(
            f"[分组] {record.segment_id}：机械臂保持不动，就地逐帧离线识别"
            f"（步长 {stride} = 每一帧都算，{record.frame_count} 帧，"
            f"RAW {info['raw_bytes'] / 1024**3:.3f} GB）……"
        )
        corners_path = run.corners_dir / f"{safe_name(record.segment_id)}.csv"
        vision_json = record.dir / VISION_JSON_NAME
        try:
            processed = process_segment(
                record.dir,
                config=config,
                segment_id=record.segment_id,
                save_corners=True,
                corners_dir=run.corners_dir,
                metrics_path=run.vision_dir / "metrics.csv",
                stride=stride,
                vision_json_path=vision_json,
                # 这一条路径处理完、校验通过就会删掉 RAW，所以 JSON 里如实写
                # raw_available=false；离线分析/回放写同一份文件时 RAW 还在，
                # 那里用默认的 true（见 save_vision_json 的说明）。
                raw_available_after=False,
            )
        except Exception as exc:
            info["kept"] = True
            info["reason"] = f"就地处理失败，RAW 保留：{exc}"
            self._record_release(run, "raw_kept", info)
            self.hooks.log(f"[分组] {record.segment_id}：{info['reason']}")
            return info
        info["processed"] = True
        # ★ 需求二：把刚算出来的逐帧结果**留在内存里**。下游（快速几何检查、
        # 离线分析）直接用这一份，不再去读 frames.raw——那时 RAW 已经删了。
        self._released_vision[record.segment_id] = processed

        ok, why = self._verify_releasable(
            record,
            processed,
            corners_path=corners_path,
            vision_json=vision_json,
            rtde_rows=rtde_rows,
        )
        if not ok:
            info["kept"] = True
            info["reason"] = f"处理结果校验不过，RAW 保留：{why}"
            self._record_release(run, "raw_kept", info)
            self.hooks.log(f"[分组] {record.segment_id}：{info['reason']}")
            return info

        remove_raw_artifacts(raw_path)
        info["deleted"] = True
        info["corners"] = str(corners_path)
        info["vision_json"] = str(vision_json)
        info["valid_frames"] = int(processed.valid_count)
        info["processed_frames"] = int(processed.frame_count)
        info["valid_ratio"] = float(processed.valid_ratio)
        self._record_release(run, "raw_deleted", info)
        self.hooks.log(
            f"[分组] {record.segment_id}：已删除原始帧（释放 "
            f"{info['raw_bytes'] / 1024**3:.3f} GB）。"
            f"逐帧角点、质心、二维转角、尺度、时间戳留在 "
            f"{vision_json.name}、{corners_path.name}（步长 {stride}，"
            f"有效帧比例 {info['valid_ratio']:.1%}）；"
            "复算可用，但不能换检测参数重识别。"
        )
        return info

    def _verify_releasable(
        self,
        record: SegmentCapture,
        processed: SegmentVision,
        *,
        corners_path: Path,
        vision_json: Path,
        rtde_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """删 RAW 之前的校验（需求三）。**回读**一遍，确认这一段真的还能用。

        需求三列了六条，缺一条就保留 RAW 并暂停（由 ``flush_group`` 中止）：

        1. ``valid_ratio ≥ thresholds.min_valid_frame_ratio``——
           "至少有一帧识别成功"是不够的：一段 400 帧里只有 1 帧认出来，
           第二层、第三层都没有可用的量。
        2. **每个声明过的阶段**都要有 ≥ ``thresholds.min_window_frames`` 帧有效帧
           （运动前/运动/保持/回程/运动后）。哪一个阶段空掉，
           事后就没有那一段的信息——而那时 RAW 已经删了。
        3. 相机时间戳文件与相位表都在（否则整段没有时间轴）。
        4. RTDE 状态流里有这一段的行（第一层"指令 → 实际"全靠它）。
        5. ``segment_vision.json`` 与角点 CSV 能**回读**，且数值与刚算的一致。
        6. 本段的基本分析跑得完（``analysis.check_segment_analyzable``）。

        ``rtde_rows`` 由 ``flush_group`` 每組读一次传进来，免得每段都把
        robot_states.csv 整份解析一遍。
        """
        from .analysis import check_segment_analyzable

        config = self.config
        thresholds = config.thresholds

        if not vision_json.is_file():
            return False, "逐帧结果文件没写成"
        if not corners_path.is_file() or corners_path.stat().st_size <= 0:
            return False, "角点表没写成或为空"
        if processed.frame_count <= 0:
            return False, "一段帧都没处理"

        # -- 1) 有效帧比例 --------------------------------------------------
        want_ratio = float(thresholds.min_valid_frame_ratio)
        if processed.valid_ratio < want_ratio:
            return False, (
                f"有效帧比例 {processed.valid_ratio:.1%} 低于要求 {want_ratio:.0%}"
                f"（{processed.valid_count}/{processed.frame_count} 帧）——"
                "不是“至少有一帧认出来”就算过，比例不够这段事后算不出可信的量"
            )

        # -- 2) 必要阶段 ------------------------------------------------
        metadata = self._segment_metadata(record)
        phases = list(metadata.get("phases") or [])
        if not phases:
            return False, "capture_metadata.json 里没有相位表，事后无法按相位切窗"
        min_frames = int(thresholds.min_window_frames)
        for entry in phases:
            label = str(entry.get("label") or "")
            try:
                start_s, end_s = float(entry["start_s"]), float(entry["end_s"])
            except (KeyError, TypeError, ValueError):
                return False, f"相位 {label or '?'} 缺 start_s/end_s"
            count = sum(
                1
                for frame in processed.frames
                if start_s <= frame.analysis_time_s <= end_s and frame.valid
            )
            if count < min_frames:
                return False, (
                    f"阶段「{label}」只有 {count} 帧有效帧（要求 ≥ {min_frames}）"
                )

        # -- 3) 相机时间戳 -----------------------------------------------
        stamps = record.dir / "frame_timestamps.csv"
        if not stamps.is_file() or stamps.stat().st_size <= 0:
            return False, "frame_timestamps.csv 缺失或为空：这一段没有相机时间轴"
        source = str(metadata.get("analysis_time_source") or "")
        if not source:
            return False, "capture_metadata.json 里没有 analysis_time_source"

        # -- 4) RTDE 状态流 ----------------------------------------------
        if rtde_rows is not None:
            count = self._rtde_rows_for(record, rtde_rows, metadata)
            if count < min_frames:
                return False, (
                    f"robot_states.csv 里这一段只有 {count} 行（要求 ≥ {min_frames}）："
                    "第一层“指令 → 实际”没有数据可对"
                )

        # -- 5) 回读一致 --------------------------------------------------
        try:
            reloaded = load_segment_vision(
                record.dir, segment_id=record.segment_id, corners_path=corners_path
            )
        except Exception as exc:
            return False, f"回读失败：{exc}"
        if reloaded.frame_count != processed.frame_count:
            return False, (
                f"回读的帧数对不上：{reloaded.frame_count} vs {processed.frame_count}"
            )
        original = [f for f in processed.frames if f.valid]
        again = [f for f in reloaded.frames if f.valid]
        if len(original) != len(again):
            return False, f"回读的有效帧数对不上：{len(again)} vs {len(original)}"
        # 首末各抽查一帧的全部几何量：质心、位移、二维转角、尺度、r_rms。
        # 少一个量，RAW 删掉之后就复算不出来了，所以逐个比。
        for before, after in ((original[0], again[0]), (original[-1], again[-1])):
            for name in (
                "centroid_x_px",
                "centroid_y_px",
                "shift_x_px",
                "shift_y_px",
                "rotation_deg",
                "scale",
                "r_rms_px",
            ):
                left, right = getattr(before, name), getattr(after, name)
                if (left is None) != (right is None):
                    return False, f"回读的 {name} 有无对不上"
                if left is not None and abs(float(left) - float(right)) > 1e-6:
                    return False, f"回读的 {name} 数值对不上：{right} vs {left}"
        if reloaded.mm_per_pixel is None:
            return False, "回读的 mm/像素 为空：面内位移换算不出来了"

        # -- 6) 本段的基本分析跑得完 --------------------------------------
        ok, why = check_segment_analyzable(
            reloaded,
            config=config,
            kind=str(metadata.get("stage") or ""),
            joint=str(metadata.get("joint") or ""),
        )
        if not ok:
            return False, f"本段基本分析跑不完：{why}"
        return True, ""

    def _segment_metadata(self, record: SegmentCapture) -> dict[str, Any]:
        """读这一段的 capture_metadata.json（读不动就给空字典，校验会如实报）。"""
        path = record.dir / "capture_metadata.json"
        if not path.is_file():
            return {}
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _rtde_rows_for(
        self,
        record: SegmentCapture,
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> int:
        """数一数这一段在 robot_states.csv 里有多少行。

        两道找法，取多的那个：

        * 按 ``event_id`` 精确匹配（记录器每段开始前会 ``mark`` 一次）；
        * 按时间窗匹配（``first_host_ns`` 起、段落时长内）。

        两道都留着是因为干运行里 RTDE 行的时间戳直接取帧时间戳，
        而真机是后台线程用真实时钟——两种模式的时间轴口径不同，
        只认一种会在另一种模式下误报"没有 RTDE 行"，然后错误地暂停整场实验。
        """
        if not rows:
            return 0
        by_event = sum(
            1 for row in rows if str(row.get("event_id") or "") == record.segment_id
        )
        first_ns = metadata.get("first_host_ns")
        window = 0
        if first_ns is not None:
            try:
                start_ns = int(first_ns)
            except (TypeError, ValueError):
                start_ns = None
            if start_ns is not None:
                end_ns = start_ns + int(
                    float(metadata.get("content_seconds") or record.seconds or 0.0) * 1e9
                )
                for row in rows:
                    try:
                        stamp = int(row.get("host_ns"))
                    except (TypeError, ValueError):
                        continue
                    if start_ns <= stamp <= end_ns:
                        window += 1
        return max(by_event, window)

    def _record_release(self, run: RunDirectory, event: str, info: dict[str, Any]) -> None:
        run.events.write(event, **{k: v for k, v in info.items() if k != "enabled"})

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

        # ★ 需求一：分析之前先把还没处理的那一组收尾。正常流程走完按钮二
        # 不会剩下东西（每个组边界都 flush 过），只有会话被中断或分步调用
        # 时才可能留着——那也得在分析之前落到"处理+校验+删 RAW"的终点上。
        if self._pending and not self.aborted:
            try:
                self.flush_group(reason="离线分析之前")
            except MotionAborted as exc:
                self.hooks.log(f"分组收尾失败，RAW 已保留，先按现有数据继续分析：{exc}")

        todo: list[str] = list(self.static_segment_ids) + [
            str(t["segment_id"]) for t in self.trials
        ]
        for position, segment_id in enumerate(todo, start=1):
            directory = run.segment_dir(segment_id)
            if not directory.is_dir():
                self.hooks.log(f"{segment_id}：采集目录不存在，跳过。")
                continue
            # ★ 需求二：本会话已经就地处理过的段，直接用内存里那一份，
            # 既不重读 RAW，也不重读 JSON——它和落盘的那份是同一个对象算出来的。
            # 调用方要抽帧看（stride>1）就在这份**完整**结果上抽取：
            # 存下来的是步长 1 的逐帧结果，抽帧是"少看几帧"，不是"少算几帧"。
            cached = self._released_vision.get(segment_id)
            if cached is not None:
                self.hooks.log(
                    f"{segment_id}：用分组流水线里刚算出的逐帧结果，不重复识别。"
                )
                segments[segment_id] = cached.subsampled(stride)
                continue
            if progress is not None:
                progress(f"离线识别 {position}/{len(todo)}：{segment_id}")
            self.hooks.log(f"离线识别 {position}/{len(todo)}：{segment_id}")
            try:
                if (directory / "frames.raw").is_file():
                    segments[segment_id] = process_segment(
                        directory,
                        config=config,
                        segment_id=segment_id,
                        save_corners=True,
                        corners_dir=run.corners_dir,
                        metrics_path=run.vision_dir / "metrics.csv",
                        stride=stride,
                        vision_json_path=directory / VISION_JSON_NAME,
                    )
                else:
                    # ★ 这一段是"边采边清"处理过的：RAW 已经删了，但逐帧结果和角点
                    # 都在段目录里。用它们复算，并在日志里**明说**这是在复算，
                    # 免得有人以为又重新识别了一遍像素。
                    self.hooks.log(
                        f"    {segment_id}：原始帧已按边采边清删除，"
                        "改用段内已保存的角点与逐帧几何复算（不重新识别像素）。"
                    )
                    # 存下来的是步长 1 的完整结果；调用方要 stride>1 就在它上面抽帧，
                    # 抽法和 process_segment(stride=...) 完全一样（见 subsampled）。
                    stored = load_segment_vision(
                        directory,
                        segment_id=segment_id,
                        corners_path=run.corners_dir / f"{safe_name(segment_id)}.csv",
                    )
                    segments[segment_id] = stored.subsampled(stride)
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

    ``signal`` 是"走完试探步长（默认 0.2°）停稳后"的静止段样本，
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
    #: 轴向/面内分解（需求四）。None = 这一趟没算（比如没有可用尺度）。
    depth_in_plane: Any = None


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
