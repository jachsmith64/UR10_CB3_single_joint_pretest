"""自测用的公共装置。

两条铁律
--------
1. **不连真实机械臂**（需求一的硬约束）。所有流程测试跑在 ``mode="dry_run"``
   的合成世界上；合成世界渲染的是真棋盘格图，离线角点识别也是真跑，
   被替换掉的只有"机械臂 + 相机"这一层物理。
2. **不允许为了通过而放宽判据**（需求七）。所以这里的配置只改"规模"和"时长"——
   关节数、幅度档数、重复次数、帧尺寸、帧率——**不改任何阈值**：
   SNR 门限、奇异余量、方向容差、残差倍数都保持交付默认值。
   如果某条断言只有放宽阈值才能过，那是代码的问题，不是测试的问题。

时长与规模
----------
一次真机规模的预实验是 72 次动作、每次几秒，自测不可能那样跑。这里用
``dry_run.shorten_durations`` 把静态基线缩短到 0.6 s，帧尺寸缩到 340×260，
让"秒"和"帧数"都小下来；帧率与相位时长保持交付默认值，
因为相位窗口里有没有帧，正是要被验证的事。相位边界仍然由**帧时间戳**决定，
所以三段式相位切分逻辑一条都没少走。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sj_pretest.config import AppConfig  # noqa: E402
from sj_pretest.experiment import ExperimentSession, SessionHooks  # noqa: E402


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


def build_config(
    tmp_path: Path,
    *,
    joints: Sequence[str] = ("J1", "J6"),
    amplitudes: Sequence[float] = (0.2,),
    repeats: int = 1,
    mode: str = "dry_run",
    **overrides: Any,
) -> AppConfig:
    """自测用的**快速**配置。只改规模与时长，一个阈值都不动。"""
    config = AppConfig()
    config.mode = mode
    config.paths.output_root = str(Path(tmp_path) / "outputs")
    # 绝对门槛压到 0.05 GB：它只是"下限"，自测不该因为跑测试的机器磁盘紧张而失败。
    # ★ 但真正把关的是"按计划估算占用"（config.check_plan_disk），它**不**受这个值
    # 影响：干运行一整趟预实验的估算在几个 GB 量级，所以跑流程自测的机器至少要有
    # 几个 GB 空闲。这是**有意的**——估算门槛要是能被测试配置随手关掉，
    # 就等于没有门槛，而"跑一半磁盘写满"正是这条门槛要防的事。
    config.paths.min_free_disk_gb = 0.05

    # 合成世界：帧尺寸缩小到"棋盘格 + 两边各 ≥40 px 余量"，省下识别开销。
    # **帧率不动**（保持交付默认 132.23）：它决定余振在一帧里被采到哪个相位，
    # 是合成世界的物理参数，不是规模参数；动它等于换了一个世界。
    config.dry_run.width = 340
    config.dry_run.height = 260
    config.dry_run.realtime = False
    config.dry_run.shorten_durations = True
    # 相位时长**保持交付默认值**（0.2/0.4/0.2）：它们决定了相位窗口里有多少帧，
    # 而窗口切分正是要被验证的东西。自测只缩静态基线和帧尺寸。
    config.dry_run.static_duration_s = 0.6

    # 到位只用 2 个中间点：点数是流程参数，不是判据。
    config.robot.approach_points = 2

    # 连上设备后那次全屏采集检查：交付默认 5.0 s，自测取**配置允许的最短值** 1.0 s
    # （``camera.connection_check_s`` 的校验下限就是 1 s，再短会被 ``validate`` 拦下）。
    # 同样是**规模**选择——它只决定"采多少帧"，不决定任何判据：
    # 帧率下限比例（min_fps_ratio）、缺帧上限（max_dropped_ratio）、写盘余量
    # （min_write_headroom）一个都没动，检查本身也照常真跑（真采、真算、真落盘、
    # 真删）。"交付默认是 5.0 s" 由 test_defaults.py 直接盯住配置类和交付 JSON。
    config.camera.connection_check_s = 1.0

    config.pretest.joints = list(joints)
    config.pretest.amplitudes_deg = list(amplitudes)
    config.pretest.repeats_per_direction = int(repeats)
    config.formal.staircase_n = 2
    config.formal.repeats = 1
    config.formal.step_deg = {joint: 0.2 for joint in joints}

    # ★ 分组流水线（采完一组就处理、校验、删 RAW）在**交付默认里是开的**，
    # 自测里显式关掉——这是**规模**选择，不是放宽判据：开着它意味着每一段都要按
    # stride=1 把每一帧重新识别一遍（实测一段约 10 s），十来段的流程自测会从
    # 七分钟涨到一小时以上。打开时的行为由 test_rolling_delete、
    # test_group_pipeline、test_raw_status_order 用**显式打开**的配置覆盖；
    # "交付默认就是 true" 由 test_defaults.py 直接盯住配置类和交付 JSON。
    # 想跑打开的那条路径，照常传 paths__delete_raw_after_process=True 覆盖即可。
    config.paths.delete_raw_after_process = False

    for path, value in overrides.items():
        target: Any = config
        parts = path.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    config.validate()
    return config


@pytest.fixture
def dry_config(tmp_path: Path) -> AppConfig:
    return build_config(tmp_path)


# --------------------------------------------------------------------------
# 会话与钩子
# --------------------------------------------------------------------------


class Recorder:
    """记下钩子的每一次调用，供断言"到底有没有问人""问了哪几步"。"""

    def __init__(self, *, answer: bool | Callable[[str], bool] = True) -> None:
        self.answer = answer
        self.asked: list[str] = []
        self.logs: list[str] = []
        self.previews: list[tuple[str, str]] = []
        self.stop = False
        self.stopped_after: int | None = None
        #: 运动前检查（机器人层每次 move_to_joint 之前调一次）。
        #: 默认 None = 不额外检查；自测里要"在某一次运动前把中止按下去"时用它。
        self.precheck: Callable[[], tuple[bool, str]] | None = None

    def confirm(self, label: str) -> bool:
        self.asked.append(label)
        if self.stopped_after is not None and len(self.asked) > self.stopped_after:
            return False
        if callable(self.answer):
            return bool(self.answer(label))
        return bool(self.answer)

    def log(self, message: str) -> None:
        self.logs.append(str(message))

    def preview(self, path: Path, note: str) -> None:
        self.previews.append((str(path), str(note)))

    def hooks(self) -> SessionHooks:
        return SessionHooks(
            confirm=self.confirm,
            precheck=self.precheck,
            stop_requested=lambda: self.stop,
            on_log=self.log,
            on_preview=self.preview,
        )

    def arm_stop_before_move(self, *, after: int = 1) -> None:
        """在第 ``after`` 次运动前检查时把"中止"按下去，并放行这一次运动。

        这样中止发生在**一次运动已经在飞**的时候，才能验证
        "中止后不再发命令、也不自动回位"这两条。
        """
        calls = {"n": 0}

        def precheck() -> tuple[bool, str]:
            calls["n"] += 1
            if calls["n"] >= int(after):
                self.stop = True
            return True, "自测：运动前检查一律通过"

        self.precheck = precheck

    def text(self) -> str:
        return "\n".join(self.logs)

    def asked_labels(self) -> str:
        return "\n".join(self.asked)


def open_session(
    config: AppConfig,
    *,
    recorder: Recorder | None = None,
    run_kind: str = "test",
) -> tuple[ExperimentSession, Recorder]:
    """开一个已经连上设备的会话（dry_run 下"设备"是合成世界）。"""
    recorder = recorder or Recorder()
    session = ExperimentSession(config, hooks=recorder.hooks())
    session.open(run_kind=run_kind)
    session.connect_devices()
    return session, recorder


@pytest.fixture
def session_factory(tmp_path: Path):
    """返回一个工厂：建配置 + 开会话，退出时自动关闭。"""

    opened: list[ExperimentSession] = []

    def factory(
        *,
        joints: Sequence[str] = ("J1", "J6"),
        amplitudes: Sequence[float] = (0.2,),
        repeats: int = 1,
        recorder: Recorder | None = None,
        run_kind: str = "test",
        config: AppConfig | None = None,
        **overrides: Any,
    ) -> tuple[ExperimentSession, Recorder, AppConfig]:
        cfg = config or build_config(
            tmp_path,
            joints=joints,
            amplitudes=amplitudes,
            repeats=repeats,
            **overrides,
        )
        session, rec = open_session(cfg, recorder=recorder, run_kind=run_kind)
        opened.append(session)
        return session, rec, cfg

    yield factory
    for session in opened:
        try:
            session.close()
        except Exception:  # pragma: no cover - 关不掉不该掩盖测试结论
            pass


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def pretest_report(
    session: ExperimentSession,
    *,
    stride: int = 4,
    static_s: float = 0.4,
    static_id: str = "static_base",
):
    """跑一段静态基线 + 三档预实验 + 离线分析，返回报告。

    静态段是必须的：第二层判据要拿"视觉信号 ÷ 静态噪声"当信噪比，
    没有静态基线就只能出曲线、判不出"看不看得出来"。
    """
    session.capture_static(segment_id=static_id, duration_s=static_s)
    session.run_pretest()
    return session.analyze_offline(stride=int(stride))


def trial_of(report, joint: str, *, amplitude: float | None = None, direction: int | None = None):
    """从报告里挑出符合条件的试验（挑不到就报错，不让断言在 None 上打转）。"""
    matches = [
        item
        for item in report.trials
        if item.joint == joint
        and (amplitude is None or abs(float(item.amplitude_deg) - float(amplitude)) < 1e-12)
        and (direction is None or int(item.direction) == int(direction))
    ]
    assert matches, f"报告里没有 {joint} 的这一条试验"
    return matches[0]


def all_trials(report, joint: str, *, amplitude: float | None = None, direction: int | None = None):
    return [
        item
        for item in report.trials
        if item.joint == joint
        and (amplitude is None or abs(float(item.amplitude_deg) - float(amplitude)) < 1e-12)
        and (direction is None or int(item.direction) == int(direction))
    ]


def read_json(path: Path) -> Any:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dir_of(session: ExperimentSession) -> Path:
    assert session.run is not None, "会话还没打开"
    return session.run.root


def segments_of(session: ExperimentSession) -> list[dict[str, Any]]:
    """运行目录里所有段的元数据，按目录名排序。

    ``capture_metadata.json`` 里没有采集目录的路径（那是被复用代码定义的格式，
    本工具不改它），所以这里把所在目录补进去，省得每个测试各自去拼。
    """
    assert session.run is not None
    out: list[dict[str, Any]] = []
    for metadata in sorted(session.run.segments_dir.glob("*/capture_metadata.json")):
        payload = dict(read_json(metadata))
        payload.setdefault("dir", str(metadata.parent))
        out.append(payload)
    return out


def rows_for_event(rows: Iterable[dict[str, Any]], event_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("event_id") or "") == event_id]
