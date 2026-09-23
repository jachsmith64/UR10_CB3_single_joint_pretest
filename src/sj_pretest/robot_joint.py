"""机器人层：在关节空间里走微动，并如实记录"指令角 / 实际角 / 实际角速度"。

被复用代码的边界（必须先说清楚）
--------------------------------
``vendor/robot.py`` 里的 ``URRobot`` **只发过 moveL（TCP 直线运动）**，
它读的状态里也没有 ``actual_qd`` / ``target_q``。本工具需要的是**关节空间微动**，
所以：

* 不改 vendored 文件一行；:class:`HardwareJointRobot` 把 ``URRobot`` 当成一个
  被包住的连接对象，在外面追加 ``moveJ`` 和关节状态读取；
* 连接、安全检查、停止、断开、脚本收尾全部继续走 ``URRobot`` 里已经跑通的代码；
* **moveJ 这条路径在本工具之前没有在真机上跑过**——这一点写进交付说明，
  现场第一次跑必须用最小步长、最低速度、手放急停。

三种模式
--------
* ``HardwareJointRobot`` —— 真机，发 moveJ；
* ``SimulatedJointRobot`` —— dry_run，把指令交给合成世界，合成世界自己按一阶跟随
  产生"实际角"，所以"指令到了、关节还在动"这件事在自测里是真的会发生的；
* ``ReplayJointRobot`` —— 回放，从历史 RTDE 记录里按时间取状态，**绝不发命令**。

三者实现同一个 :class:`JointRobotPort`，所以实验编排、记录、分析都不用区分模式。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .config import JOINT_NAMES
from .kinematics import JOINT_LIMITS_DEG, check_joint_in_limits
from .vendor_shim import vendor_module


class RobotError(RuntimeError):
    """机器人层出错。消息中文，能直接显示在界面上。"""


class MotionAborted(RobotError):
    """用户中止（或急停/掉线）导致的取消。区别于普通错误：它不该被当成失败重试。"""


@dataclass
class RobotState:
    """一条机器人状态记录。字段对应需求六要求的原始数据项 5～9。"""

    host_ns: int
    #: 指令关节角（度）。真机取 ``target_q``；合成/回放取当时下发的目标。
    commanded_q_deg: tuple[float, ...] | None
    actual_q_deg: tuple[float, ...]
    actual_qd_deg_s: tuple[float, ...] | None
    actual_tcp_pose: tuple[float, ...] | None
    robot_mode: Any = None
    safety_mode: Any = None
    runtime_state: Any = None
    speed_scaling: float | None = None
    #: 这一条状态属于哪个事件（需求六第 11 项：实验阶段/关节/步长/方向/重复编号）。
    event_id: str | None = None
    stage: str | None = None
    synthetic: bool = False

    def to_row(self) -> dict[str, Any]:
        """铺平成一行，方便写 CSV。"""
        row: dict[str, Any] = {
            "host_ns": int(self.host_ns),
            "event_id": self.event_id or "",
            "stage": self.stage or "",
            "synthetic": int(bool(self.synthetic)),
            "robot_mode": "" if self.robot_mode is None else self.robot_mode,
            "safety_mode": "" if self.safety_mode is None else self.safety_mode,
            "runtime_state": "" if self.runtime_state is None else self.runtime_state,
            "speed_scaling": "" if self.speed_scaling is None else self.speed_scaling,
        }
        for index, name in enumerate(JOINT_NAMES):
            row[f"command_q_{name}_deg"] = _fmt(
                None if self.commanded_q_deg is None else self.commanded_q_deg[index]
            )
            row[f"actual_q_{name}_deg"] = _fmt(self.actual_q_deg[index])
            row[f"actual_qd_{name}_deg_s"] = _fmt(
                None if self.actual_qd_deg_s is None else self.actual_qd_deg_s[index]
            )
        for index, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            row[f"actual_tcp_{axis}"] = _fmt(
                None if self.actual_tcp_pose is None else self.actual_tcp_pose[index]
            )
        return row


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


class JointRobotPort(Protocol):
    """实验编排需要的机器人能力。"""

    mode: str
    synthetic: bool

    def connect(self, *, require_control: bool) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self, *, event_id: str | None = None, stage: str | None = None) -> RobotState: ...

    def move_to_joint(
        self,
        target_joint_deg: Sequence[float],
        *,
        speed_deg_s: float,
        accel_deg_s2: float,
        event_id: str,
        label: str,
    ) -> None: ...

    def sync_host_ns(self, host_ns: int) -> None:
        """把采集层当前帧的时间戳告诉机器人。

        真机/回放忽略它；合成世界用它推进虚拟时间——这样"关节在动"这件事
        由帧的时间轴驱动，而不用真的 sleep。采集层每接一帧都会调一次。
        """
        ...

    def settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
    ) -> bool:
        """**非阻塞**地问一次"停稳了吗"。

        为什么要非阻塞版本：采集线程在运动期间必须继续接帧，不能为了等停稳
        而阻塞（那样运动过程就没有画面了）。所以"等停稳"被拆成
        "每帧问一次" + 下面那个阻塞包装。
        """
        ...

    def wait_until_settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
        timeout_s: float,
    ) -> bool: ...

    def stop(self) -> None: ...

    def abort(self, reason: str) -> None: ...

    def abort_reason(self) -> str | None: ...

    def describe_safety(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# 真机
# --------------------------------------------------------------------------


@dataclass
class HardwareJointRobot:
    """真机：**组合**（不是继承）被复用代码的 ``URRobot``，只追加关节空间动作。

    用组合而不是继承，是因为 ``URRobot`` 是"平铺脚本"风格写的类，直接继承会让
    本类同时背上面向接口的两套状态；组合把边界划得很清楚：连接/断开/安全检查/
    停止/脚本收尾全部转发给它，新增的只有 moveJ 和关节状态字段。

    **没有做的事**（写在这里避免误解）：本类不做碰撞检查、不做工作空间限制、
    不自己实现运动规划。它只做三件事：查限位、问控制器能不能到、发 moveJ。
    真机安全依赖：示教器急停、现场防护、以及操作者对每一步的人工确认。
    """

    #: 每一步运动前的人机确认入口。真机上是"要人在界面上点"的那个回调。
    confirm: Callable[[str], bool] | None = None
    #: 每一步运动前的额外检查（比如"画面里棋盘格还在"）。返回 (通过, 原因)。
    precheck: Callable[[], tuple[bool, str]] | None = None
    mode: str = "hardware"
    synthetic: bool = False
    settle_poll_s: float = 0.05

    def __post_init__(self) -> None:
        self._robot: Any = None
        self._abort: str | None = None
        self._last_command: tuple[float, ...] | None = None
        self._last_safety: dict[str, Any] = {}
        self._moved_event_ids: list[str] = []
        # "从什么时候开始已经安静了"，用于 hold_s 的连续判据（非阻塞询问之间保持）。
        self._stable_since: float | None = None

    # -- 连接 -------------------------------------------------------------

    def connect(self, *, require_control: bool) -> None:
        robot_module = vendor_module("robot")
        self._robot = robot_module.URRobot()
        # 连接、Dashboard 只读、RTDE 建链与重试全部沿用被复用代码里跑通过的那套。
        self._robot.connect(require_control=require_control)
        state = self._robot.read_state()
        self._last_safety = self._safety_snapshot(state)

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()

    # -- 状态 -------------------------------------------------------------

    def read_state(
        self, *, event_id: str | None = None, stage: str | None = None
    ) -> RobotState:
        if self._robot is None or self._robot.receive is None:
            raise RobotError("尚未连接 RTDE，无法读取机器人状态。")
        raw = self._robot.read_state()
        # ``actual_qd`` / ``target_q`` 是被复用代码没有写的两个字段，在这里补上。
        actual_qd = self._safe(self._robot.receive, "getActualQd", None)
        target_q = self._safe(self._robot.receive, "getTargetQ", None)
        self._last_safety = self._safety_snapshot(raw)
        return RobotState(
            host_ns=int(raw["host_ns"]),
            commanded_q_deg=_deg_tuple(target_q),
            actual_q_deg=_deg_tuple(raw["actual_q_rad"]) or (0.0,) * 6,
            actual_qd_deg_s=_deg_tuple(actual_qd),
            actual_tcp_pose=tuple(float(v) for v in raw["actual_tcp_pose"]),
            robot_mode=raw.get("robot_mode"),
            safety_mode=raw.get("safety_mode"),
            runtime_state=raw.get("runtime_state"),
            speed_scaling=raw.get("speed_scaling"),
            event_id=event_id,
            stage=stage,
        )

    @staticmethod
    def _safe(interface: Any, method_name: str, default: Any) -> Any:
        method = getattr(interface, method_name, None)
        if method is None:
            return default
        try:
            return method()
        except Exception:
            return default

    def _safety_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        robot_module = vendor_module("robot")
        dashboard = getattr(self._robot, "dashboard_info", {}) or {}
        return {
            "robot_mode": state.get("robot_mode"),
            "safety_mode": state.get("safety_mode"),
            "runtime_state": state.get("runtime_state"),
            "speed_scaling": state.get("speed_scaling"),
            "dashboard_safety_ok": robot_module._dashboard_safety_status_is_normal(dashboard),
            "rtde_safety_ok": robot_module._rtde_safety_mode_is_normal(state),
        }

    def describe_safety(self) -> dict[str, Any]:
        return dict(self._last_safety)

    # -- 中止 -------------------------------------------------------------

    def abort(self, reason: str) -> None:
        """请求中止。**不会**自动回到任何位置（需求八：中止后不自动回位）。"""
        if self._abort is None:
            self._abort = str(reason)
            self.stop()

    def abort_reason(self) -> str | None:
        return self._abort

    def stop(self) -> None:
        if self._robot is not None:
            # 关节空间用 stopJ；被复用代码里的 stop_motion() 走的是 stopL。
            control = getattr(self._robot, "control", None)
            if control is not None:
                stop_j = getattr(control, "stopJ", None)
                if stop_j is not None:
                    try:
                        stop_j(1.0)
                        print("[机器人] 已请求 stopJ 减速停止。", flush=True)
                        return
                    except Exception as exc:
                        print(f"[机器人警告] stopJ 失败：{exc}", flush=True)
            self._robot.stop_motion()

    # -- 运动 -------------------------------------------------------------

    def move_to_joint(
        self,
        target_joint_deg: Sequence[float],
        *,
        speed_deg_s: float,
        accel_deg_s2: float,
        event_id: str,
        label: str,
    ) -> None:
        if self._abort is not None:
            raise MotionAborted(f"已中止（{self._abort}），不再发送任何运动命令。")
        if self._robot is None or self._robot.control is None:
            raise RobotError("控制连接未建立，不能发送 moveJ。")

        target = tuple(float(v) for v in target_joint_deg)
        if len(target) != 6:
            raise RobotError(f"目标关节角必须是 6 个，收到 {len(target)} 个。")

        # 第一层：本地的名义限位检查。
        ok, reason = check_joint_in_limits(target, margin_deg=0.5, limits=JOINT_LIMITS_DEG)
        if not ok:
            raise RobotError(f"目标姿态未通过本地限位检查：{reason}")

        # 第二层：让 UR 控制器自己判断（和 moveL 路径一样的思路）。
        checker = getattr(self._robot.control, "isJointsWithinSafetyLimits", None)
        if checker is None:
            raise RobotError(
                "当前 ur_rtde 版本没有 isJointsWithinSafetyLimits；"
                "为了安全，程序不允许跳过该检查后运动。"
            )
        if not bool(checker(list(target))):
            raise RobotError(
                f"{label}：UR 控制器判定该关节姿态超出当前安全限制，已拒绝发送。"
            )

        # 第三层：安全检查回调（棋盘格还在不在、磁盘够不够之类）。
        if self.precheck is not None:
            passed, why = self.precheck()
            if not passed:
                raise MotionAborted(f"{label}：运动前检查未通过 —— {why}")

        # 第四层：人工确认。需求三要求每一步都要人确认，这一步不允许被绕过。
        if self.confirm is not None and not self.confirm(label):
            raise MotionAborted(f"{label}：操作者取消了这一次运动。")

        # 第五层：读一次状态，确认安全模式正常。
        state = self.read_state(event_id=event_id)
        if state.safety_mode is not None and int(state.safety_mode) != 1:
            raise MotionAborted(
                f"{label}：当前 safety_mode={state.safety_mode}（非 NORMAL），"
                "拒绝运动。请先在示教器上恢复。"
            )

        speed_rad = math.radians(float(speed_deg_s))
        accel_rad = math.radians(float(accel_deg_s2))
        print(
            f"[机器人] moveJ → {label}\n"
            f"         目标(度)：{['%.4f' % v for v in target]}\n"
            f"         速度 {speed_deg_s} °/s（{speed_rad:.6f} rad/s），"
            f"加速度 {accel_deg_s2} °/s²（{accel_rad:.6f} rad/s²）",
            flush=True,
        )
        accepted = bool(
            self._robot.control.moveJ(list(target), speed_rad, accel_rad, True)
        )
        if not accepted:
            raise RobotError(f"{label}：UR 控制器拒绝了异步 moveJ。")
        self._last_command = target
        self._moved_event_ids.append(event_id)

    def sync_host_ns(self, host_ns: int) -> None:
        """真机不需要外部时钟：RTDE 自己带时间戳。"""
        return None

    def settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
    ) -> bool:
        """**非阻塞**地问一次"停稳了吗"。读一次 RTDE，立刻返回。

        判据（两条都要满足，并连续保持 ``hold_s`` 秒）：
        * 六个关节的实际角与目标的偏差 ≤ ``tolerance_deg``；
        * 最大实际角速度 ≤ 0.01 °/s —— 比我们用的 0.5 °/s 慢 50 倍，
          意味着"确实停下来了"，不是"还在慢慢爬"。
        """
        target = [float(v) for v in target_joint_deg]
        if self._abort is not None:
            return False
        try:
            state = self.read_state()
        except Exception as exc:
            # 读不到状态就当作"没停稳"，让上层按超时处理，而不是假装停稳了。
            self._last_safety = {**self._last_safety, "settle_read_error": str(exc)}
            return False
        error = max(abs(state.actual_q_deg[i] - target[i]) for i in range(6))
        speed = (
            max(abs(v) for v in state.actual_qd_deg_s) if state.actual_qd_deg_s else 0.0
        )
        quiet = error <= float(tolerance_deg) and speed <= 0.01
        now = time.perf_counter()
        if not quiet:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = now
            return False
        return (now - self._stable_since) >= float(hold_s)

    def wait_until_settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
        timeout_s: float,
    ) -> bool:
        """阻塞版：没有采集在跑的时候用它（例如到位序列的逐步确认）。

        返回 True 表示"停稳了"，False 表示超时（超时不算错误——分析层会把
        "到时间还在动"如实记录下来，而不是替它圆场）。
        """
        self._stable_since = None
        deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline:
            if self._abort is not None:
                return False
            if self.settled(
                target_joint_deg, tolerance_deg=float(tolerance_deg), hold_s=float(hold_s)
            ):
                return True
            time.sleep(self.settle_poll_s)
        return False

    def motion_in_progress(self) -> bool:
        if self._robot is None:
            return False
        return bool(self._robot.motion_in_progress())

    def moved_event_ids(self) -> list[str]:
        return list(self._moved_event_ids)


# --------------------------------------------------------------------------
# 干运行（合成世界）
# --------------------------------------------------------------------------


@dataclass
class SimulatedJointRobot:
    """dry_run 用的"机器人"：把指令交给合成世界，不发任何真实命令。

    关键点：它**不假装一切完美**。合成世界的关节做一阶跟随，所以
    "指令已经下发了，但关节还没到位"这件事会真实发生；下发指令时还会触发余振。
    这样"RTDE 稳了但画面还在动"这条判据在自测里才能被走到。
    """

    world: Any
    nominal_joint_deg: Sequence[float]
    #: 合成时间（秒）。虚拟时钟下由帧时间戳驱动，所以这里只做累计。
    clock_s: float = 0.0
    mode: str = "dry_run"
    synthetic: bool = True
    confirm: Callable[[str], bool] | None = None
    precheck: Callable[[], tuple[bool, str]] | None = None

    def __post_init__(self) -> None:
        self._abort: str | None = None
        self._current: tuple[float, ...] = tuple(float(v) for v in self.nominal_joint_deg)
        self._target: tuple[float, ...] = self._current
        self._event_id: str | None = None
        self._stage: str | None = None
        self._moved_event_ids: list[str] = []
        self._move_started_s: float | None = None
        self._t0_ns: int | None = None
        self._stable_since_s: float | None = None

    # -- 连接（合成世界里没有连接这回事，但接口要一致） --------------------

    def connect(self, *, require_control: bool) -> None:
        self._current = tuple(float(v) for v in self.nominal_joint_deg)
        self._target = self._current
        self.world.reset(self._current)
        self._t0_ns = None
        self.clock_s = 0.0
        self._stable_since_s = None

    def disconnect(self) -> None:
        return None

    # -- 状态 -------------------------------------------------------------

    def set_time(self, now_s: float) -> None:
        """由采集层在每段结束时推进一次，保证合成时间和帧时间轴一致。"""
        self.clock_s = float(now_s)
        self.world.step(self.clock_s)

    def sync_host_ns(self, host_ns: int) -> None:
        """用帧的时间戳推进虚拟时间。

        取单调不减（``max``）：到位序列里的阻塞等待会自己把时钟往前推，
        之后帧的时间戳可能比它略小，不能由此让时间倒流。
        """
        ns = int(host_ns)
        if self._t0_ns is None:
            self._t0_ns = ns
        self.set_time(max(self.clock_s, (ns - self._t0_ns) / 1e9))

    def read_state(
        self, *, event_id: str | None = None, stage: str | None = None
    ) -> RobotState:
        self.world.step(self.clock_s)
        actual = tuple(float(v) for v in self.world.actual)
        target = tuple(float(v) for v in self.world.target)
        tau = float(getattr(self.world.config, "joint_response_tau_s", 0.12))
        # 合成角速度＝一阶跟随的瞬时速度，量级上和真机 RTDE 的 actual_qd 同义。
        vd = tuple(
            (target[i] - actual[i]) / tau if tau > 0 else 0.0 for i in range(6)
        )
        return RobotState(
            host_ns=time.perf_counter_ns(),
            commanded_q_deg=target,
            actual_q_deg=actual,
            actual_qd_deg_s=vd,
            actual_tcp_pose=_synthetic_tcp(actual),
            robot_mode="SIMULATED",
            safety_mode=1,
            runtime_state=0,
            speed_scaling=1.0,
            event_id=event_id,
            stage=stage,
            synthetic=True,
        )

    def describe_safety(self) -> dict[str, Any]:
        return {
            "mode": "dry_run",
            "robot_mode": "SIMULATED",
            "safety_mode": 1,
            "collision_status": "unknown",
            "note": "干运行：未连接任何真实机械臂，未发送任何运动命令。",
        }

    # -- 中止 -------------------------------------------------------------

    def abort(self, reason: str) -> None:
        if self._abort is None:
            self._abort = str(reason)

    def abort_reason(self) -> str | None:
        return self._abort

    def stop(self) -> None:
        return None

    # -- 运动 -------------------------------------------------------------

    def move_to_joint(
        self,
        target_joint_deg: Sequence[float],
        *,
        speed_deg_s: float,
        accel_deg_s2: float,
        event_id: str,
        label: str,
    ) -> None:
        if self._abort is not None:
            raise MotionAborted(f"已中止（{self._abort}），不再下发任何动作。")
        target = tuple(float(v) for v in target_joint_deg)
        ok, reason = check_joint_in_limits(target, margin_deg=0.5, limits=JOINT_LIMITS_DEG)
        if not ok:
            raise RobotError(f"目标姿态未通过限位检查：{reason}")
        if self.precheck is not None:
            passed, why = self.precheck()
            if not passed:
                raise MotionAborted(f"{label}：运动前检查未通过 —— {why}")
        if self.confirm is not None and not self.confirm(label):
            raise MotionAborted(f"{label}：操作者取消了这一次运动。")
        del speed_deg_s, accel_deg_s2  # 合成世界不模拟速度曲线，只用一阶跟随
        self._event_id = event_id
        self._target = target
        self._move_started_s = self.clock_s
        self._stable_since_s = None
        self.world.command(target, self.clock_s)
        self._moved_event_ids.append(event_id)

    def settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
    ) -> bool:
        """合成世界里的"停稳了吗"：在**虚拟时间**轴上看一阶跟随的偏差和速度。

        因为虚拟时间由帧时间戳推进，所以运动期间采集照常进行——
        这和真机上"边录边等停稳"是同一条逻辑。
        """
        self.world.step(self.clock_s)
        tau = float(getattr(self.world.config, "joint_response_tau_s", 0.12))
        error = max(
            abs(float(self.world.actual[i]) - float(target_joint_deg[i]))
            for i in range(6)
        )
        speed = error / tau if tau > 0 else 0.0
        quiet = error <= float(tolerance_deg) and speed <= 0.01
        if not quiet:
            self._stable_since_s = None
            return False
        if self._stable_since_s is None:
            self._stable_since_s = self.clock_s
            return False
        return (self.clock_s - self._stable_since_s) >= float(hold_s)

    def wait_until_settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
        timeout_s: float,
    ) -> bool:
        """阻塞版：没有采集在跑的时候用它（例如到位序列的逐步确认）。

        不 sleep 真实时间，而是把虚拟时钟一小步一小步往前推——
        干运行要跑得动整套 72 次动作。
        """
        del timeout_s
        self._stable_since_s = None
        for _ in range(20000):
            self.clock_s += 0.005
            if self.settled(
                target_joint_deg,
                tolerance_deg=float(tolerance_deg),
                hold_s=float(hold_s),
            ):
                return True
        return False

    def motion_in_progress(self) -> bool:
        return False

    def moved_event_ids(self) -> list[str]:
        return list(self._moved_event_ids)


# --------------------------------------------------------------------------
# 回放
# --------------------------------------------------------------------------


@dataclass
class ReplayJointRobot:
    """回放用的"机器人"：从历史记录里取状态，**一个命令都不会发**。

    如果回放的数据里没有机器人记录，就读不出状态；这时如实返回一个空状态，
    让上层把"没有 RTDE 数据"这件事显示出来，而不是编造关节角。
    """

    #: 历史 RTDE 记录：``[(host_ns, {关节名: 角度}, ...)]``，可为空。
    records: list[tuple[int, dict[str, float], dict[str, Any]]] = field(
        default_factory=list
    )
    mode: str = "replay"
    synthetic: bool = False
    #: 一次动作后最多消费多少条历史记录当作"等待时间"。
    settle_records: int = 64

    def __post_init__(self) -> None:
        self._abort: str | None = None
        self._index = 0
        self._target: tuple[float, ...] | None = None
        self._event_id: str | None = None
        self._stage: str | None = None
        self._moved_event_ids: list[str] = []
        self._records_since_move = 0
        self._last_settle_reason: str | None = None

    def connect(self, *, require_control: bool) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read_state(
        self, *, event_id: str | None = None, stage: str | None = None
    ) -> RobotState:
        if not self.records:
            return RobotState(
                host_ns=time.perf_counter_ns(),
                commanded_q_deg=None,
                actual_q_deg=(0.0,) * 6,
                actual_qd_deg_s=None,
                actual_tcp_pose=None,
                event_id=event_id,
                stage=stage,
            )
        if self._index < len(self.records):
            host_ns, angles, extra = self.records[self._index]
            self._index += 1
        else:
            host_ns, angles, extra = self.records[-1]
        actual = tuple(float(angles.get(name, 0.0)) for name in JOINT_NAMES)
        return RobotState(
            host_ns=int(host_ns),
            commanded_q_deg=self._target or actual,
            actual_q_deg=actual,
            actual_qd_deg_s=None,
            actual_tcp_pose=extra.get("actual_tcp_pose"),
            robot_mode=extra.get("robot_mode"),
            safety_mode=extra.get("safety_mode"),
            runtime_state=extra.get("runtime_state"),
            event_id=event_id,
            stage=stage,
        )

    def describe_safety(self) -> dict[str, Any]:
        return {
            "mode": "replay",
            "collision_status": "unknown",
            "note": "回放历史数据：未连接任何真实机械臂，未发送任何运动命令。",
        }

    def abort(self, reason: str) -> None:
        if self._abort is None:
            self._abort = str(reason)

    def abort_reason(self) -> str | None:
        return self._abort

    def stop(self) -> None:
        return None

    def move_to_joint(
        self,
        target_joint_deg: Sequence[float],
        *,
        speed_deg_s: float,
        accel_deg_s2: float,
        event_id: str,
        label: str,
    ) -> None:
        if self._abort is not None:
            raise MotionAborted(f"已中止（{self._abort}）。")
        del speed_deg_s, accel_deg_s2, label
        self._target = tuple(float(v) for v in target_joint_deg)
        self._event_id = event_id
        self._records_since_move = 0
        self._last_settle_reason = None
        self._moved_event_ids.append(event_id)

    def sync_host_ns(self, host_ns: int) -> None:
        """回放的时间轴来自历史记录本身，不需要外部时钟。"""
        return None

    def settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
    ) -> bool:
        """回放里的"停稳"只能**从历史数据里看出来**，不能凭空假设。

        两条结束条件，都不假装：
        * 历史记录里实际角已经进到容差以内 —— 说明当年确实到位了；
        * 已经消费了 ``settle_records`` 条历史记录 —— 数据就这么多，
          再等也没有新信息。这时把 ``_last_settle_reason`` 记成
          "history_exhausted"，让上层知道这个"停稳"是数据到头，不是真停稳。
        """
        del hold_s
        self._records_since_move += 1
        if not self.records:
            self._last_settle_reason = "no_history"
            return True
        state = self.read_state()
        error = max(
            abs(state.actual_q_deg[i] - float(target_joint_deg[i])) for i in range(6)
        )
        if error <= max(float(tolerance_deg), 0.01):
            self._last_settle_reason = "reached_in_history"
            return True
        if self._records_since_move >= int(self.settle_records):
            self._last_settle_reason = "history_exhausted"
            return True
        return False

    def wait_until_settled(
        self,
        target_joint_deg: Sequence[float],
        *,
        tolerance_deg: float,
        hold_s: float,
        timeout_s: float,
    ) -> bool:
        del timeout_s
        for _ in range(max(1, int(self.settle_records))):
            if self.settled(
                target_joint_deg,
                tolerance_deg=float(tolerance_deg),
                hold_s=float(hold_s),
            ):
                return True
        return False

    def motion_in_progress(self) -> bool:
        return False

    def moved_event_ids(self) -> list[str]:
        return list(self._moved_event_ids)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _deg_tuple(values: Any) -> tuple[float, ...] | None:
    if values is None:
        return None
    try:
        items = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(items) != 6:
        return None
    return tuple(math.degrees(value) for value in items)


def _np_array(values: Sequence[float]) -> Any:
    import numpy as np

    return np.asarray(values, dtype=np.float64)


def _synthetic_tcp(joint_deg: Sequence[float]) -> tuple[float, ...]:
    """合成世界里的 TCP 位姿：用名义正运动学算位置，姿态固定为 0 旋转向量。

    只用于让"记录里有个 TCP 位姿"这件事成立；合成世界的 TCP 不代表真机姿态。
    """
    from .kinematics import forward_kinematics

    matrix = forward_kinematics(joint_deg)
    position = matrix[:3, 3]
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        0.0,
        0.0,
        0.0,
    )


def make_robot(
    config: Any,
    *,
    world: Any = None,
    records: list[tuple[int, dict[str, float], dict[str, Any]]] | None = None,
    confirm: Callable[[str], bool] | None = None,
    precheck: Callable[[], tuple[bool, str]] | None = None,
) -> JointRobotPort:
    """按运行模式造出对应的机器人对象。"""
    if config.mode == "hardware":
        return HardwareJointRobot(confirm=confirm, precheck=precheck)
    if config.mode == "replay":
        return ReplayJointRobot(records=list(records or []))
    if world is None:
        raise RobotError("dry_run 模式必须传入合成世界（SyntheticWorld）。")
    return SimulatedJointRobot(
        world=world,
        nominal_joint_deg=config.robot.nominal_joint_deg,
        confirm=confirm,
        precheck=precheck,
    )
