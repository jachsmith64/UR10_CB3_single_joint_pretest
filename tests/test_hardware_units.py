"""真机角度单位：**度**是内部单位，**弧度**只在下发命令那一刻出现。

为什么这条必须单独测
--------------------
ur_rtde 的关节接口是 SI 单位制（rad），而本工具内部、日志、界面和"停稳"判据
全部用度。1.0.0 里 ``HardwareJointRobot.move_to_joint`` 把度值**直接**喂给了
``isJointsWithinSafetyLimits`` 和 ``moveJ``：0.2° 会被控制器当成 0.2 rad
（≈ 11.5°），每条试运动的幅度放大 57 倍，而安全范围检查查的也是被放大的角度
——真机上这是一条会撞到东西的错误，而它**不会**在任何干运行测试里露头。

所以这里塞一个**手写的假控制器**进去（``robot_factory`` 那道缝），把
"到底送出去的是什么数"逐字断言。这个假控制器不连任何真机，也不依赖装了
ur_rtde：``vendor/robot.py`` 里 ur_rtde 是**懒导入**（连接时才 import）。

断言三件事：
1. 安全范围检查收到的是弧度；
2. ``moveJ`` 收到的是弧度，且速度/加速度也是 rad/s、rad/s²；
3. 日志、界面链路（``_last_command``）和 "停稳" 判据**仍然是度**。
"""

from __future__ import annotations

import math
import sys

import pytest

from sj_pretest.config import JOINT_NAMES, AppConfig
from sj_pretest.robot_joint import HardwareJointRobot, RobotError

# 一个刻意"好认"的目标：J1 从名义位姿加 0.2°，其余不变。
# 0.2° 和 0.2 rad 差了 57 倍，断言里一眼能看出是哪个单位。
NOMINAL_DEG = tuple(AppConfig().robot.nominal_joint_deg)
TARGET_DEG = (NOMINAL_DEG[0] + 0.2,) + NOMINAL_DEG[1:]
SPEED_DEG_S = 0.5
ACCEL_DEG_S2 = 1.0


class FakeReceive:
    """假 RTDE 接收接口。只提供真机路径真正会调的那两个方法。"""

    def __init__(self, owner: "FakeURRobot") -> None:
        self._owner = owner

    def getActualQd(self) -> list[float]:
        return [0.0] * 6

    def getTargetQ(self) -> list[float]:
        return list(self._owner.actual_q_rad)


class FakeControl:
    """假控制接口。**把收到的每一个参数原样记下来**，这就是被测的事实。"""

    def __init__(self, owner: "FakeURRobot") -> None:
        self._owner = owner
        self.safety_limit_calls: list[list[float]] = []
        self.movej_calls: list[tuple[list[float], float, float, bool]] = []
        self.stopj_calls: list[float] = []

    def isJointsWithinSafetyLimits(self, joints_rad: list[float]) -> bool:
        self.safety_limit_calls.append(list(joints_rad))
        # 假控制器照单全收；被验证的是"送进来的是什么"，不是它怎么判。
        return True

    def moveJ(
        self, joints_rad: list[float], speed: float, accel: float, asynchronous: bool
    ) -> bool:
        self.movej_calls.append((list(joints_rad), float(speed), float(accel), bool(asynchronous)))
        # 真机语义：命令下发后，实际角会朝目标走。这里直接"到位"，
        # 好让 settled() 的判据在**度数**上成立（如果它拿度去比弧度，就会失败）。
        self._owner.actual_q_rad = list(joints_rad)
        return True

    def stopJ(self, deceleration: float) -> None:
        self.stopj_calls.append(float(deceleration))


class FakeURRobot:
    """和 ``vendor.robot.URRobot`` 同接口的假连接对象：**不连任何真机**。

    只实现本工具真机路径会用到的那部分：connect / disconnect / read_state /
    receive / control / dashboard_info / stop_motion / motion_in_progress。
    """

    def __init__(self) -> None:
        self.actual_q_rad: list[float] = [math.radians(v) for v in NOMINAL_DEG]
        self.receive = FakeReceive(self)
        self.control = FakeControl(self)
        self.dashboard_info: dict[str, str] = {}
        self.connected = False
        self.require_control = False
        self.stop_motion_calls = 0

    # -- 连接 -------------------------------------------------------------

    def connect(self, *, require_control: bool) -> None:
        self.connected = True
        self.require_control = bool(require_control)

    def disconnect(self) -> None:
        self.connected = False

    # -- 状态 -------------------------------------------------------------

    def read_state(self) -> dict[str, object]:
        return {
            "host_ns": 123456789,
            "actual_q_rad": list(self.actual_q_rad),
            "actual_tcp_pose": [0.0] * 6,
            "robot_mode": 7,
            "safety_mode": 1,
            "runtime_state": 0,
            "speed_scaling": 1.0,
        }

    def stop_motion(self) -> None:
        self.stop_motion_calls += 1

    def motion_in_progress(self) -> bool:
        return False


def _hardware_robot() -> tuple[HardwareJointRobot, FakeURRobot]:
    holder: dict[str, FakeURRobot] = {}

    def factory() -> FakeURRobot:
        holder["robot"] = FakeURRobot()
        return holder["robot"]

    robot = HardwareJointRobot(robot_factory=factory)
    robot.connect(require_control=True)
    return robot, holder["robot"]


def _move(robot: HardwareJointRobot) -> None:
    robot.move_to_joint(
        TARGET_DEG,
        speed_deg_s=SPEED_DEG_S,
        accel_deg_s2=ACCEL_DEG_S2,
        event_id="test_move",
        label="自测：单位边界",
    )


def test_safety_check_and_movej_receive_radians() -> None:
    """送进 ``isJointsWithinSafetyLimits`` 和 ``moveJ`` 的必须是**弧度**。"""
    robot, fake = _hardware_robot()
    _move(robot)

    expected = [math.radians(v) for v in TARGET_DEG]
    assert len(fake.control.safety_limit_calls) == 1, "安全范围检查没被调用"
    got_check = fake.control.safety_limit_calls[0]
    assert len(got_check) == 6
    for index, (sent, want, deg) in enumerate(zip(got_check, expected, TARGET_DEG)):
        assert sent == pytest.approx(want, abs=1e-12), (
            f"{JOINT_NAMES[index]} 送进安全范围检查的是 {sent}，"
            f"期望 {want} rad（{deg}°）。"
            "看起来是把度当弧度送出去了——真机上这条命令会大 57 倍。"
        )

    assert len(fake.control.movej_calls) == 1, "moveJ 没被调用（或调用了不止一次）"
    sent_move, speed, accel, asynchronous = fake.control.movej_calls[0]
    for index, (sent, want) in enumerate(zip(sent_move, expected)):
        assert sent == pytest.approx(want, abs=1e-12), (
            f"{JOINT_NAMES[index]} 的 moveJ 目标 {sent}，期望 {want} rad"
        )
    assert speed == pytest.approx(math.radians(SPEED_DEG_S))
    assert accel == pytest.approx(math.radians(ACCEL_DEG_S2))
    assert asynchronous is True

    # 反向自查：这些数**不能**等于度值本身（否则上面那些断言就成了摆设）。
    assert max(abs(v) for v in sent_move) < 3.2, "送出去的数看着像度，不像弧度"


def test_degrees_stay_degrees_in_logs_ui_and_settle_criterion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """日志、界面链接（``_last_command``）和"停稳"判据**仍然用度**。"""
    robot, fake = _hardware_robot()
    _move(robot)

    # 1) 日志：现场看的是度，同时把真正下发的弧度也打出来，好对上账。
    printed = capsys.readouterr().out
    assert "目标(度)" in printed
    assert TARGET_DEG[0] and f"{TARGET_DEG[0]:.4f}" in printed
    assert "目标(弧度，实际下发)" in printed

    # 2) 界面链路：机器人记住的"上一次命令"是度——界面拿它显示、拿它比。
    assert robot._last_command == pytest.approx(TARGET_DEG, abs=0.0), (
        "上一次命令记的不是度，界面和停稳判据会跟着错"
    )

    # 3) 停稳判据：假控制器此刻的实际角就是目标的**弧度**值。
    #    判据必须把两边的单位对齐到度，才能判出"到位"；
    #    如果它拿目标度去减实际弧度（0.2 对 0.0035），误差会接近名义角本身。
    tolerance = 0.002
    assert robot.settled(TARGET_DEG, tolerance_deg=tolerance, hold_s=0.0) is False, (
        "第一次询问应当先记下'已安静'的起点"
    )
    assert robot.settled(TARGET_DEG, tolerance_deg=tolerance, hold_s=0.0) is True, (
        "实际角已经等于目标（只是单位是弧度），停稳判据却判不出到位——"
        "说明它拿度去比弧度了"
    )

    # 4) 假控制器一件真事都没做：没连真机、没装 ur_rtde。
    assert fake.connected is True  # 连的是假对象，不是真机
    assert "ur_rtde" not in sys.modules, "这条测试不该依赖（更不该导入）ur_rtde"


def test_degrees_are_converted_exactly_once() -> None:
    """换算只发生在边界上：本类内部存的是度，发出去的是弧度，一一对应。

    这条防的是"有人为了防止单位错，在多处各转一次"，那会让 0.2° 变成
    0.2 rad 再变成 11.5° 喂给安全检查，反而更危险。
    """
    robot, fake = _hardware_robot()
    _move(robot)
    sent = fake.control.movej_calls[0][0]
    back_to_deg = [math.degrees(v) for v in sent]
    assert back_to_deg == pytest.approx(list(TARGET_DEG), abs=1e-9), (
        "把下发的弧度换回度，必须和用户给的目标一致（不能被转两次）"
    )


def test_local_limit_check_still_rejects_a_nonsense_pose() -> None:
    """单位换算是加在原有防线**之内**的，不是把防线换掉。

    这条同时证明：假控制器返回 True 也不会让一个超限位的姿态发出去——
    第一层本地限位检查在换算之前就拦住了。
    """
    robot, fake = _hardware_robot()
    bad = (1000.0,) + TARGET_DEG[1:]
    with pytest.raises(RobotError):
        robot.move_to_joint(
            bad,
            speed_deg_s=SPEED_DEG_S,
            accel_deg_s2=ACCEL_DEG_S2,
            event_id="test_bad",
            label="自测：超限位",
        )
    assert fake.control.safety_limit_calls == []
    assert fake.control.movej_calls == []
