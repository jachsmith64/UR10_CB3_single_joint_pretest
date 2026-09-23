"""一体化界面：三个主按钮 + 两组正式实验。

界面只做三件事：显示、收参数、把点击转成 :class:`~sj_pretest.experiment.ExperimentSession`
的调用。所有判据、时序、安全闸门都在会话层，界面不重新实现一遍——
否则"界面显示通过、实际没通过"这种事迟早会发生。

一条重要的分工
--------------
长时间的动作（连接、到位、预实验、离线分析）跑在**工作线程**里，界面线程只负责
刷新。但"人工确认"必须回到界面线程弹窗（Tkinter 的控件不是线程安全的），
所以 :meth:`ExperimentApp._ask` 用 ``root.after`` 把弹窗排进界面线程，
工作线程在 Event 上等结果。关闭窗口时等待会被打断，返回 False——
**没有人能确认的时候，答案是不许动**。

模式
----
默认 ``dry_run``（合成世界，不连任何硬件）。切到 ``hardware`` 需要用户在弹出的
对话框里亲手勾选确认，并且默认值不会自动变成 hardware。切到 ``replay`` 需要
先选一份历史数据目录。
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import JOINT_NAMES, AppConfig, ConfigError, check_free_disk
from .experiment import ExperimentSession, SessionHooks
from .joint_space import planned_scale_lines
from .replay import describe_replay_data, discover_runs, reanalyze_run

#: 界面标题。版本号写在 __init__ 里，这里只给名字。
WINDOW_TITLE = "UR10 CB3 单关节微动预实验与正式实验一体化工具"

#: 三个主按钮的文案。测试会按这些字符串找控件，改名时一起改。
BUTTON_CONNECT = "① 连接设备并到达实验姿态"
BUTTON_PRETEST = "② 运行静态噪声与三档预实验"
BUTTON_ANALYZE = "③ 离线分析并生成正式实验参数"
BUTTON_GROUP_A = "组A 正式实验（单向爬梯→返回）"
BUTTON_GROUP_B = "组B 正式实验（反向实验）"
BUTTON_STOP = "中止 / 停止"

#: 参数面板的字段规格：(分区, 配置路径, 类型)。类型显式写死，不去猜——
#: ``exposure_us=None`` 和 ``gain=None`` 光看值是分不出"浮点数"还是"字符串"的。
PARAMETER_SPEC: tuple[tuple[str, str, str], ...] = (
    ("机器人", "robot.ip", "str"),
    ("机器人", "robot.nominal_joint_deg", "floats"),
    ("机器人", "robot.approach_points", "int"),
    ("机器人", "robot.approach_speed_deg_s", "float"),
    ("机器人", "robot.approach_accel_deg_s2", "float"),
    ("机器人", "robot.trial_speed_deg_s", "float"),
    ("机器人", "robot.trial_accel_deg_s2", "float"),
    ("机器人", "robot.settle_tolerance_deg", "float"),
    ("机器人", "robot.settle_hold_s", "float"),
    ("机器人", "robot.settle_timeout_s", "float"),
    ("机器人", "robot.rtde_record_hz", "float"),
    ("相机", "camera.serial", "str"),
    ("相机", "camera.mvs_import_path", "optional_str"),
    ("相机", "camera.exposure_us", "optional_float"),
    ("相机", "camera.gain", "optional_float"),
    ("相机", "camera.expected_fps", "float"),
    ("相机", "camera.warmup_s", "float"),
    ("相机", "camera.board_inner_corners", "ints"),
    ("相机", "camera.square_mm", "float"),
    ("相机", "camera.roi", "optional_ints"),
    ("相机", "camera.min_margin_px", "int"),
    ("相机", "camera.max_dropped_ratio", "float"),
    ("相机", "camera.save_raw", "bool"),
    ("相机", "camera.save_sample_images", "bool"),
    ("预实验", "pretest.joints", "strs"),
    ("预实验", "pretest.amplitudes_deg", "floats"),
    ("预实验", "pretest.repeats_per_direction", "int"),
    ("预实验", "pretest.quick_probe_deg", "float"),
    ("预实验", "pretest.confirm_each_joint", "bool"),
    ("预实验", "pretest.auto_within_joint", "bool"),
    ("采集时长", "camera.static_duration_s", "float"),
    ("采集时长", "camera.pre_motion_s", "float"),
    ("采集时长", "camera.hold_s", "float"),
    ("采集时长", "camera.post_motion_s", "float"),
    ("正式实验", "formal.staircase_n", "int"),
    ("正式实验", "formal.repeats", "int"),
    ("正式实验", "formal.hold_s", "float"),
    ("正式实验", "formal.speed_deg_s", "float"),
    ("正式实验", "formal.accel_deg_s2", "float"),
    ("正式实验", "formal.enable_group_a", "bool"),
    ("正式实验", "formal.enable_group_b", "bool"),
    ("判据", "thresholds.joint_limit_margin_deg", "float"),
    ("判据", "thresholds.min_snr_vs_static", "float"),
    ("判据", "thresholds.min_valid_frame_ratio", "float"),
    # 掉帧率的上限在**相机**那一组（camera.max_dropped_ratio），
    # 这里原来还挂了一条 thresholds.max_dropped_ratio——阈值组里没有这个字段，
    # 参数对话框一打开就会 AttributeError。删掉。
    ("判据", "thresholds.min_window_frames", "int"),
    ("判据", "thresholds.max_repeat_relative_spread", "float"),
    ("判据", "thresholds.direction_tolerance_deg", "float"),
    ("判据", "thresholds.residual_factor", "float"),
    ("输出", "paths.output_root", "str"),
    ("输出", "paths.min_free_disk_gb", "float"),
)


class UiError(RuntimeError):
    """界面层出错。消息中文。"""


# --------------------------------------------------------------------------
# 配置读写（界面上的每一个数都走这里，保证"界面显示=落盘内容"）
# --------------------------------------------------------------------------


def get_field(config: AppConfig, path: str) -> Any:
    target: Any = config
    for part in path.split("."):
        target = getattr(target, part)
    return target


def set_field(config: AppConfig, path: str, value: Any) -> None:
    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _format_float(value: float) -> str:
    """浮点数在界面上怎么显示。

    这里必须**可逆**：操作者打开参数面板、什么都没改、点一下应用，
    配置不能因此发生任何变化。``g`` 默认只有 6 位有效数字，
    ``-175.6992`` 会被显示成 ``-175.699``，一点应用就悄悄改了 0.0002°。
    所以先用 12 位有效数字试，读回来不相等就退回 ``repr``（它一定可逆）。
    """
    text = f"{value:.12g}"
    try:
        if float(text) == value:
            return text
    except ValueError:  # pragma: no cover - 12 位有效数字不可能转不回来
        pass
    return repr(value)


def format_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple)):
        return "、".join(format_field(item) for item in value)
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def parse_field(kind: str, current: Any, text: str) -> Any:
    """把界面上的一行文字转回该字段的类型。转不了就抛 UiError（消息给用户看）。"""
    raw = (text or "").strip()
    if kind == "str":
        return raw
    if kind == "optional_str":
        return raw or None
    if kind == "bool":
        lowered = raw.lower()
        if lowered in ("是", "true", "1", "y", "yes", "开"):
            return True
        if lowered in ("否", "false", "0", "n", "no", "关", ""):
            return False
        raise UiError(f"看不懂的开关值：{raw}（请填 是/否）")
    if kind in ("floats", "ints", "strs", "optional_ints"):
        items = [piece for piece in _split_list(raw) if piece != ""]
        if not items:
            return [] if kind != "optional_ints" else None
        if kind == "strs":
            return [item.strip().upper() for item in items]
        if kind in ("ints", "optional_ints"):
            try:
                return [int(item) for item in items]
            except ValueError as exc:
                raise UiError(f"这里要填整数：{raw}") from exc
        try:
            return [float(item) for item in items]
        except ValueError as exc:
            raise UiError(f"这里要填数字：{raw}") from exc
    if kind == "optional_float":
        if raw == "":
            return None
        return _as_float(raw)
    if kind == "float":
        return _as_float(raw)
    if kind == "int":
        return int(_as_float(raw))
    raise UiError(f"未知的字段类型：{kind}")


def _as_float(text: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise UiError(f"这里要填数字：{text}") from exc


def _split_list(text: str) -> list[str]:
    """列表字段的分隔：中英文逗号、分号、空白都认。"""
    normalized = text.replace("，", ",").replace("、", ",").replace(";", ",")
    return [piece.strip() for piece in normalized.replace("\t", ",").split(",")]


def parse_fields(
    config: AppConfig, values: Mapping[str, str]
) -> list[tuple[str, Any, Any]]:
    """把界面上的一批文字解析成配置值，**但先不写**。

    返回 ``[(路径, 新值, 旧值), ...]``。解析失败直接抛 :class:`UiError`，
    此时一个字段都没动过。
    """
    staged: list[tuple[str, Any, Any]] = []
    for path, text in values.items():
        kind = field_kind(path)
        current = get_field(config, path)
        staged.append((path, parse_field(kind, current, text), current))
    return staged


def commit_fields(config: AppConfig, staged: Sequence[tuple[str, Any, Any]]) -> list[str]:
    """把解析好的值写进配置，返回人可读的改动清单。"""
    changes: list[str] = []
    for path, value, current in staged:
        if value != current:
            changes.append(f"{path}：{format_field(current)} → {format_field(value)}")
    for path, value, _current in staged:
        set_field(config, path, value)
    return changes


def apply_fields(config: AppConfig, values: Mapping[str, str]) -> list[str]:
    """把界面上的一批文字写回配置，返回人可读的改动清单。

    先全部解析、再全部写入：任何一项填错就整体不生效，避免"改了一半"的配置
    被拿去跑实验。

    注意：这一条只保证**解析**失败时不动配置。像"速度填了 6.5 °/s，而 CB3
    的上限是 5"这种**合法数字但不合法的值**，只有在 ``config.validate()``
    里才会被拦住——所以参数对话框走的是 :func:`apply_fields_with_validation`，
    它先在副本上校验、通过了才落到真配置上。
    """
    return commit_fields(config, parse_fields(config, values))


def apply_fields_with_validation(
    config: AppConfig, values: Mapping[str, str]
) -> list[str]:
    """像 :func:`apply_fields`，但先在**副本**上过一遍 ``validate()``。

    副本校验通过之后才写进真配置，所以"填了超过上限的速度"这类错误不会
    留下半套改动。返回改动清单；任何一步失败都抛异常，配置保持原样。
    """
    import copy

    staged = parse_fields(config, values)
    candidate = copy.deepcopy(config)
    commit_fields(candidate, staged)
    candidate.validate()
    return commit_fields(config, staged)


def field_kind(path: str) -> str:
    for _section, spec_path, kind in PARAMETER_SPEC:
        if spec_path == path:
            return kind
    raise UiError(f"参数表里没有这一项：{path}")


def parameter_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    """按分区整理参数表，给参数对话框按顺序摆。"""
    order: list[str] = []
    grouped: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    for section, path, kind in PARAMETER_SPEC:
        if path in seen:
            continue
        seen.add(path)
        if section not in grouped:
            grouped[section] = []
            order.append(section)
        grouped[section].append((path, kind))
    return [(section, grouped[section]) for section in order]


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------


@dataclass
class UiOptions:
    """界面的非配置项。"""

    #: 是否用工作线程跑长任务。自测时设 False，一切都变成同步调用。
    threaded: bool = True
    #: 同步模式下"人工确认"由谁回答。None 表示拒绝（安全默认）。
    confirm_policy: Callable[[str], bool] | None = None
    #: 分析抽帧步长。逐帧识别约 150 ms/帧，整批要有数小时，现场一般都抽帧。
    analysis_stride: int = 8


class ExperimentApp:
    """主窗口。构造它就会建出控件；``threaded=False`` 时不需要真的 mainloop。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        root: Any = None,
        options: UiOptions | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.options = options or UiOptions()
        self.session: ExperimentSession | None = None
        self.report: Any = None
        self._stop = threading.Event()
        self._closing = False
        self._busy = False
        self._log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._preview_image: Any = None
        self._last_run_dir: Path | None = None

        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root if root is not None else tk.Tk()
        self._owns_root = root is None
        self._build()
        if self.options.threaded:
            self.root.after(80, self._pump)

    # -- 控件 -------------------------------------------------------------

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.title(f"{WINDOW_TITLE}（{self.config.mode}）")
        try:
            self.root.geometry("1180x780")
        except Exception:  # pragma: no cover - 无显示环境下给不出尺寸也无所谓
            pass

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # 顶栏：模式、输出目录
        top = ttk.LabelFrame(outer, text="运行设置", padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="模式：").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=self.config.mode)
        self.mode_box = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=("dry_run", "replay", "hardware"),
            width=10,
            state="readonly",
        )
        self.mode_box.grid(row=0, column=1, sticky="w")
        self.mode_box.bind("<<ComboboxSelected>>", self.on_mode_change)
        ttk.Label(top, text="输出目录：").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.output_var = tk.StringVar(value=str(self.config.resolve_output_root()))
        ttk.Entry(top, textvariable=self.output_var, width=52).grid(
            row=0, column=3, sticky="we"
        )
        ttk.Button(top, text="选目录…", command=self.on_choose_output).grid(
            row=0, column=4, padx=4
        )
        ttk.Button(top, text="参数…", command=self.on_parameters).grid(row=0, column=5)
        ttk.Button(top, text="载入配置…", command=self.on_load_config).grid(
            row=0, column=6, padx=4
        )
        ttk.Button(top, text="另存配置…", command=self.on_save_config).grid(row=0, column=7)
        top.columnconfigure(3, weight=1)

        # 三个主按钮
        main = ttk.LabelFrame(outer, text="主流程（按顺序点，每一步都会先问再动）", padding=6)
        main.pack(fill="x", pady=6)
        self.btn_connect = ttk.Button(main, text=BUTTON_CONNECT, command=self.on_connect)
        self.btn_pretest = ttk.Button(main, text=BUTTON_PRETEST, command=self.on_pretest)
        self.btn_analyze = ttk.Button(main, text=BUTTON_ANALYZE, command=self.on_analyze)
        for column, button in enumerate((self.btn_connect, self.btn_pretest, self.btn_analyze)):
            button.grid(row=0, column=column, sticky="we", padx=4, pady=2)
            main.columnconfigure(column, weight=1)

        # 正式实验
        formal = ttk.LabelFrame(outer, text="正式实验（先做理论范围检查，不是碰撞检查）", padding=6)
        formal.pack(fill="x", pady=(0, 6))
        ttk.Label(formal, text="正式步长(°)：").grid(row=0, column=0, sticky="w")
        self.step_vars: dict[str, Any] = {}
        for index, name in enumerate(JOINT_NAMES):
            ttk.Label(formal, text=f"{name}").grid(row=0, column=1 + index * 2, sticky="e")
            var = tk.StringVar(
                value=format_field(float(self.config.formal.step_deg.get(name, 0.0)))
            )
            self.step_vars[name] = var
            ttk.Entry(formal, textvariable=var, width=7).grid(
                row=0, column=2 + index * 2, sticky="w", padx=(2, 6)
            )
        self.btn_group_a = ttk.Button(formal, text=BUTTON_GROUP_A, command=lambda: self.on_formal("A"))
        self.btn_group_b = ttk.Button(formal, text=BUTTON_GROUP_B, command=lambda: self.on_formal("B"))
        self.btn_group_a.grid(row=1, column=0, columnspan=6, sticky="we", padx=4, pady=4)
        self.btn_group_b.grid(row=1, column=7, columnspan=6, sticky="we", padx=4, pady=4)
        for column in range(13):
            formal.columnconfigure(column, weight=1 if column in (5, 12) else 0)

        # 回放与复算
        replay = ttk.LabelFrame(outer, text="历史数据（回放 / 复算，都不连硬件）", padding=6)
        replay.pack(fill="x", pady=(0, 6))
        ttk.Button(replay, text="选历史数据…", command=self.on_choose_replay).grid(
            row=0, column=0, padx=4
        )
        self.replay_var = tk.StringVar(value=self.config.replay_source or "（未选择）")
        ttk.Label(replay, textvariable=self.replay_var).grid(row=0, column=1, sticky="w")
        ttk.Button(replay, text="体检这份数据", command=self.on_check_replay).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(replay, text="复算历史运行…", command=self.on_reanalyze).grid(
            row=0, column=3, padx=4
        )
        ttk.Label(
            replay,
            text="复算＝拿磁盘上已有的 RAW 重新识别一遍，不下发任何运动命令，也不覆盖原结果。",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        replay.columnconfigure(1, weight=1)

        # 中止
        control = ttk.Frame(outer)
        control.pack(fill="x", pady=(0, 6))
        self.btn_stop = ttk.Button(control, text=BUTTON_STOP, command=self.on_stop)
        self.btn_stop.grid(row=0, column=0, padx=4)
        self.status_var = tk.StringVar(value="就绪。默认干运行：不会连接任何真实设备。")
        ttk.Label(control, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=8)
        control.columnconfigure(1, weight=1)

        # 日志 + 预览
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)
        log_frame = ttk.LabelFrame(middle, text="日志", padding=4)
        log_frame.pack(side="left", fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=20, width=78)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set, state="disabled")

        preview_frame = ttk.LabelFrame(middle, text="画面预览", padding=4)
        preview_frame.pack(side="right", fill="both")
        self.preview_label = ttk.Label(preview_frame, text="（还没有画面）", anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_note = tk.StringVar(value="")
        ttk.Label(preview_frame, textvariable=self.preview_note).pack(fill="x")

        self.append_log(f"{WINDOW_TITLE}")
        self.append_log(
            "安全边界：本工具没有碰撞模型。每一步运动都要人工确认，"
            "确认窗口之外不会有任何自动运动；中止后不会自动回位；"
            "真实碰撞状态一律记为 unknown。"
        )
        self.append_log(f"当前模式：{self.config.mode}；输出目录：{self.output_var.get()}")
        # 开跑之前就把"这场实验要占多久、多少盘"摆出来。按钮按下之后再知道
        # 是几十分钟、几百 GB，就太晚了。
        try:
            for line in planned_scale_lines(self.config):
                self.append_log(line)
        except Exception as exc:  # pragma: no cover - 估算失败不该拦住界面
            self.append_log(f"（规模估算没算出来，不影响使用：{exc}）")
        self.refresh_buttons()

    # -- 界面刷新 ---------------------------------------------------------

    def append_log(self, message: str) -> None:
        """写一行日志。线程安全：只往队列里放，界面线程去取。"""
        text = str(message).rstrip()
        if self.options.threaded:
            self._log_queue.put(("log", text))
        else:
            self._write_log(text)

    def _write_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        for line in text.splitlines() or [""]:
            self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status_var.set(str(text))

    def show_preview(self, path: str | Path, note: str = "") -> None:
        if self.options.threaded:
            self._log_queue.put(("preview", (str(path), str(note))))
        else:
            self._show_preview(str(path), str(note))

    def _show_preview(self, path: str, note: str) -> None:
        tk = self.tk
        target = Path(path)
        if not target.is_file():
            self.preview_note.set(f"预览图不存在：{target}")
            return
        try:
            image = tk.PhotoImage(file=str(target))
        except Exception as exc:
            # PNG/GIF 之外要 Pillow；没有就只写文字，不假装显示了。
            self.preview_note.set(f"无法显示预览图（{type(exc).__name__}）：{target.name}")
            return
        factor = max(1, int(max(image.width() / 520, image.height() / 400)) + 1)
        if factor > 1:
            image = image.subsample(factor, factor)
        self._preview_image = image  # 必须留引用，否则 Tk 会把它回收掉
        self.preview_label.configure(image=self._preview_image, text="")
        self.preview_note.set(note or target.name)

    def refresh_buttons(self) -> None:
        normal = "normal" if not self._busy else "disabled"
        for button in (
            self.btn_connect,
            self.btn_pretest,
            self.btn_analyze,
            self.btn_group_a,
            self.btn_group_b,
        ):
            button.configure(state=normal)
        # 按钮二/三必须在会话打开之后才有意义。
        if self.session is None:
            self.btn_pretest.configure(state="disabled")
            self.btn_analyze.configure(state="disabled")
        elif self._busy:
            self.btn_pretest.configure(state="disabled")
            self.btn_analyze.configure(state="disabled")
        if self.session is not None and self.session.aborted:
            self.btn_connect.configure(state="disabled")
            self.btn_pretest.configure(state="disabled")
            self.btn_analyze.configure(state="disabled")
            self.btn_group_a.configure(state="disabled")
            self.btn_group_b.configure(state="disabled")
        self.btn_stop.configure(state="normal")

    def _pump(self) -> None:
        """界面线程：把队列里的东西画出来。"""
        drained = 0
        while drained < 200:
            try:
                kind, payload = self._log_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if kind == "log":
                self._write_log(payload)
            elif kind == "preview":
                path, note = payload
                self._show_preview(path, note)
            elif kind == "status":
                self.set_status(payload)
            elif kind == "done":
                self._on_task_done(*payload)
        if not self._closing:
            self.root.after(80, self._pump)

    # -- 人工确认 ---------------------------------------------------------

    def _ask(self, label: str) -> bool:
        """问人。返回 True 才允许运动；任何异常一律返回 False。"""
        if self.options.threaded:
            if self._closing:
                return False
            done = threading.Event()
            box: dict[str, bool] = {}

            def show() -> None:
                from tkinter import messagebox

                try:
                    box["value"] = bool(
                        messagebox.askyesno(
                            "请确认这一步",
                            label + "\n\n确认之后才会发送这一条运动命令。",
                            parent=self.root,
                        )
                    )
                except Exception:
                    box["value"] = False
                finally:
                    done.set()

            try:
                self.root.after(0, show)
            except Exception:
                return False
            while not done.wait(0.1):
                if self._closing:
                    return False
            return bool(box.get("value", False))
        if self.options.confirm_policy is None:
            self.append_log(f"[未确认] 没有人可以确认，拒绝这一步：{label.splitlines()[0]}")
            return False
        return bool(self.options.confirm_policy(label))

    def _make_hooks(self) -> SessionHooks:
        return SessionHooks(
            confirm=self._ask,
            precheck=None,
            stop_requested=self._stop.is_set,
            on_log=self.append_log,
            on_progress=lambda index, total, note: self.set_status(f"[{index}/{total}] {note}"),
            on_preview=self.show_preview,
        )

    # -- 任务调度 ---------------------------------------------------------

    def _run_task(self, title: str, work: Callable[[], None]) -> None:
        """跑一个长任务。threaded=False 时直接同步跑（自测用）。"""
        if self._busy:
            self.append_log("上一步还没结束，请等它跑完或按中止。")
            return
        self._busy = True
        self.refresh_buttons()
        self.set_status(title)
        self.append_log(f"===== {title} =====")

        def runner() -> None:
            error: str | None = None
            try:
                work()
            except KeyboardInterrupt:  # pragma: no cover - 交互时才可能
                error = "操作者中断"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.append_log(f"[出错] {error}")
                self.append_log(traceback.format_exc())
            finally:
                if self.options.threaded:
                    self._log_queue.put(("done", (title, error)))
                else:
                    self._on_task_done(title, error)

        if self.options.threaded:
            threading.Thread(target=runner, name="sj-task", daemon=True).start()
        else:
            runner()

    def _on_task_done(self, title: str, error: str | None) -> None:
        self._busy = False
        if error:
            self.set_status(f"{title} 结束（有错误）：{error}")
            if self.options.threaded and not self._closing:
                try:
                    from tkinter import messagebox

                    messagebox.showerror("这一步出错了", f"{title}\n\n{error}", parent=self.root)
                except Exception:
                    pass
        else:
            self.set_status(f"{title} 已完成")
        self.refresh_buttons()

    # -- 会话 -------------------------------------------------------------

    def _open_session(self, run_kind: str) -> ExperimentSession:
        if self.session is not None:
            return self.session
        session = ExperimentSession(self.config, hooks=self._make_hooks())
        session.open(run_kind=run_kind)
        session.connect_devices()
        self.session = session
        self._last_run_dir = session.run.root if session.run is not None else None
        self.append_log(f"运行目录：{self._last_run_dir}")
        return session

    def _close_session(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            except Exception as exc:
                self.append_log(f"关闭会话时出错（已忽略）：{exc}")
            self.session = None
            self.refresh_buttons()

    # -- 按钮一 -----------------------------------------------------------

    def on_connect(self) -> None:
        def work() -> None:
            session = self._open_session("approach")
            result = session.run_approach()
            for line in result.lines:
                self.append_log(line)
            self.append_log(
                "到位流程结束。请**人工核对**示教器上的实际姿态与现场，"
                "确认棋盘格完整、周围没有障碍，再点按钮二。"
            )
            for line in session.config.camera_hint_lines():
                self.append_log(line)

        self._run_task("按钮一：连接设备并到达实验姿态", work)

    # -- 按钮二 -----------------------------------------------------------

    def on_pretest(self) -> None:
        def work() -> None:
            if self.session is None:
                raise UiError("请先点按钮一（连接设备并到达实验姿态）。")
            session = self.session
            session.run_static()
            probes, failed = session.run_quick_probes()
            self.append_log(
                f"快速几何检查：{len(probes.segments)} 段；"
                f"未通过关节 = {failed or '无'}"
            )
            if failed:
                # 需求三.2：不通过就停下来问人，不替人决定继续。
                question = (
                    f"以下关节的快速几何检查没通过：{'、'.join(failed)}\n\n"
                    "继续跑完整的三档预实验吗？\n"
                    "（不继续也不影响：已采到的数据全部保留，可以稍后单独复算。）"
                )
                proceed = self._ask(question)
                if not proceed:
                    raise UiError(
                        "操作者选择暂停。已采集的静态基线与快速检查数据都保留了，"
                        "没有发送任何新的运动命令。"
                    )
            result = session.run_pretest()
            for line in result.lines:
                self.append_log(line)
            self.append_log("预实验结束。接下来点按钮三做离线分析。")

        self._run_task("按钮二：静态噪声与三档预实验", work)

    # -- 按钮三 -----------------------------------------------------------

    def on_analyze(self) -> None:
        def work() -> None:
            if self.session is None:
                raise UiError("请先点按钮一、按钮二。")
            report = self.session.analyze_offline(
                stride=int(self.options.analysis_stride),
                progress=lambda note: self.set_status(note),
            )
            self.report = report
            for line in report.summary_lines():
                self.append_log(line)
            self._fill_recommended_steps(report)
            self.append_log(
                "推荐步长已填进上面的输入框。**请人工确认或修改**之后再点组A/组B——"
                "本工具不会自动把步长放大到 1° 或 5°。"
            )

        self._run_task("按钮三：离线分析并生成正式实验参数", work)

    def _fill_recommended_steps(self, report: Any) -> None:
        missing: list[str] = []
        for item in getattr(report, "recommendations", []):
            joint = str(getattr(item, "joint", ""))
            if joint not in self.step_vars:
                continue
            value = getattr(item, "recommended_deg", None)
            if value is None:
                missing.append(joint)
                continue
            self.step_vars[joint].set(format_field(float(value)))
            self.append_log(f"{joint}：推荐正式步长 {value}°（{item.reason}）")
        if missing:
            self.append_log(
                "以下关节没有任何一档全部通过，"
                f"需要人工填写更大步长：{'、'.join(missing)}。"
                "本工具不做自动外扩。"
            )

    def collect_steps(self, *, require_all: bool = True) -> dict[str, float]:
        """读界面上的六个正式步长。"""
        steps: dict[str, float] = {}
        waiting: list[str] = []
        for name, var in self.step_vars.items():
            text = (var.get() or "").strip()
            if text == "":
                waiting.append(name)
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise UiError(f"{name} 的步长不是数字：{text}") from exc
            if value <= 0:
                waiting.append(name)
                continue
            steps[name] = value
        if require_all and waiting:
            raise UiError(
                f"这些关节还没有正式步长：{'、'.join(waiting)}。\n"
                "请先跑按钮三拿到推荐值，或者自己填一个（工具不会替你猜）。"
            )
        return steps

    # -- 正式实验 ---------------------------------------------------------

    def on_formal(self, group: str) -> None:
        group = group.upper()

        def work() -> None:
            steps = self.collect_steps(require_all=False)
            enable = (
                self.config.formal.enable_group_a
                if group == "A"
                else self.config.formal.enable_group_b
            )
            if not enable:
                raise UiError(f"配置里已关闭组{group}（formal.enable_group_{group.lower()}）。")
            session = self._open_session(f"formal_{group}")
            if group == "A":
                self._range_check_then_confirm(session, steps)
            if self.session is session and session.approach_needed():
                self.append_log("当前姿态不在实验姿态附近，先走到位（每一步仍会问人）。")
                result = session.run_approach()
                for line in result.lines:
                    self.append_log(line)
            session.run_formal(group, step_deg=steps)
            self.append_log(f"组{group}采集结束。原始数据已落盘，可以随时复算。")

        self._run_task(f"正式实验 组{group}", work)

    def _range_check_then_confirm(
        self, session: ExperimentSession, steps: Mapping[str, float]
    ) -> None:
        """组A 之前的理论范围检查。**不是碰撞检查**，措辞不能含糊。"""
        if not self.config.formal.require_range_check:
            self.append_log("配置里关掉了理论范围检查（formal.require_range_check=false）。")
            return
        lines = session.formal_range_checks(steps)
        for line in lines:
            self.append_log(line)
        question = (
            "组A 之前的理论范围检查结果：\n\n"
            + "\n".join(lines)
            + "\n\n提醒：理论检查用的是名义运动学，**不等于碰撞安全**。"
            "真实碰撞状态记为 unknown，请人工核对现场。\n\n"
            "确认之后开始组A（单向爬梯后返回）吗？"
        )
        if not self._ask(question):
            raise UiError("操作者没有确认理论范围检查，组A 不开始。")

    # -- 停止 -------------------------------------------------------------

    def on_stop(self) -> None:
        self._stop.set()
        self.append_log(
            "已请求中止：当前这一小步结束后不再发送任何运动命令，"
            "也不会自动回位。已采到的数据全部保留。"
        )
        if self.session is not None:
            self.session.abort("操作者在界面上按了中止")
        self.set_status("已中止（不会自动回位）")
        self.refresh_buttons()

    def reset_stop(self) -> None:
        """重新开始一批动作前清掉中止标志。只由用户显式点击触发。"""
        self._stop.clear()

    # -- 配置与数据 -------------------------------------------------------

    def on_mode_change(self, _event: Any = None) -> None:
        chosen = self.mode_var.get()
        if chosen == self.config.mode:
            return
        if chosen == "hardware":
            if not self._ask(
                "切换到 hardware 模式会**连接真实机械臂**。\n"
                "本工具没有碰撞模型；所有运动都要你在每一步之前确认。\n"
                "请确认：急停按钮在你手边、工作区里没有人、机器人和相机都已经上电。\n\n"
                "确认切换到硬件模式吗？"
            ):
                self.mode_var.set(self.config.mode)
                return
            self.config.mode = "hardware"
        elif chosen == "replay":
            if not self.config.replay_source:
                self.append_log("回放模式需要先选一份历史数据（右侧“选历史数据…”）。")
                self.on_choose_replay()
            if not self.config.replay_source:
                self.mode_var.set(self.config.mode)
                return
            self.config.mode = "replay"
        else:
            self.config.mode = "dry_run"
        self.append_log(f"模式已切换为 {self.config.mode}。")
        self.root.title(f"{WINDOW_TITLE}（{self.config.mode}）")

    def on_choose_output(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(title="选择输出目录（每次运行会在下面建一个时间戳子目录）")
        if chosen:
            self.config.paths.output_root = chosen
            self.output_var.set(chosen)
            self.append_log(f"输出目录改为：{chosen}")

    def on_choose_replay(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(title="选择历史数据目录（含 frames.raw 或逐帧图像）")
        if chosen:
            self.config.replay_source = chosen
            self.replay_var.set(chosen)
            self.append_log(f"回放数据：{chosen}")

    def on_check_replay(self) -> None:
        source = self.config.replay_source
        if not source:
            self.append_log("还没有选历史数据。")
            return
        try:
            for line in describe_replay_data(source):
                self.append_log(line)
        except Exception as exc:
            self.append_log(f"这份数据用不了：{exc}")

    def on_reanalyze(self) -> None:
        from tkinter import filedialog

        root_dir = filedialog.askdirectory(title="选择要复算的运行目录（含 run_manifest.json）")
        if not root_dir:
            return
        directory = Path(root_dir)

        def work() -> None:
            runs = discover_runs(directory.parent)
            match = [item for item in runs if item.path.resolve() == directory.resolve()]
            if match:
                self.append_log("这次运行：" + match[0].line())
            result = reanalyze_run(
                directory,
                stride=int(self.options.analysis_stride),
                progress=lambda note: self.set_status(note),
                stop_requested=self._stop.is_set,
            )
            for line in result.lines:
                self.append_log(line)
            if result.report is not None:
                self.report = result.report
                for line in result.report.summary_lines():
                    self.append_log(line)
                self._fill_recommended_steps(result.report)
            if result.failed:
                self.append_log(
                    "以下段识别失败（原始数据没有改动）："
                    + "；".join(f"{key}（{value}）" for key, value in result.failed.items())
                )
            self.append_log(f"复算结果目录：{result.out_dir}")

        self._run_task("复算历史运行", work)

    def on_load_config(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askopenfilename(
            title="载入实验配置", filetypes=[("JSON 配置", "*.json"), ("全部文件", "*.*")]
        )
        if not chosen:
            return
        try:
            self.config = AppConfig.load(chosen)
        except (ConfigError, OSError, ValueError) as exc:
            self.append_log(f"载入配置失败：{exc}")
            return
        self.mode_var.set(self.config.mode)
        self.output_var.set(str(self.config.resolve_output_root()))
        self.replay_var.set(self.config.replay_source or "（未选择）")
        for name, var in self.step_vars.items():
            var.set(format_field(float(self.config.formal.step_deg.get(name, 0.0))))
        self._close_session()
        self.append_log(f"已载入配置：{chosen}")
        self.refresh_buttons()

    def on_save_config(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.asksaveasfilename(
            title="另存实验配置",
            defaultextension=".json",
            filetypes=[("JSON 配置", "*.json"), ("全部文件", "*.*")],
        )
        if not chosen:
            return
        try:
            self.config.formal.step_deg = dict(self.collect_steps(require_all=False))
            self.config.validate()
            path = self.config.save(chosen)
        except (ConfigError, UiError, OSError) as exc:
            self.append_log(f"保存配置失败：{exc}")
            return
        self.append_log(f"配置已保存：{path}")

    def on_parameters(self) -> None:
        ParameterDialog(self).show()

    # -- 关闭 -------------------------------------------------------------

    def destroy(self) -> None:
        self._closing = True
        self._stop.set()
        self._close_session()
        if self._owns_root:
            try:
                self.root.destroy()
            except Exception:
                pass

    def mainloop(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.root.mainloop()


# --------------------------------------------------------------------------
# 参数对话框
# --------------------------------------------------------------------------


class ParameterDialog:
    """把所有参数摆出来的滚动窗口。

    "界面上的每一个数都在 config.json 里"——这句话要成立，就得有一个地方能看到
    全部的数。填错任何一项都会整体不生效，不会出现"改了一半"的配置。
    """

    def __init__(self, app: ExperimentApp) -> None:
        self.app = app
        self.window: Any = None
        self.entries: dict[str, Any] = {}

    def show(self) -> None:
        tk, ttk = self.app.tk, self.app.ttk
        window = tk.Toplevel(self.app.root)
        self.window = window
        window.title("实验参数（全部写进 config.json）")
        window.geometry("760x620")

        canvas = tk.Canvas(window, borderwidth=0)
        scroll = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=8)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        for section, fields in parameter_sections():
            box = ttk.LabelFrame(body, text=section, padding=6)
            box.pack(fill="x", pady=4)
            for row, (path, _kind) in enumerate(fields):
                ttk.Label(box, text=path, width=34, anchor="w").grid(
                    row=row, column=0, sticky="w"
                )
                value = tk.StringVar(value=format_field(get_field(self.app.config, path)))
                ttk.Entry(box, textvariable=value, width=48).grid(
                    row=row, column=1, sticky="we"
                )
                self.entries[path] = value
            box.columnconfigure(1, weight=1)

        note = ttk.Label(
            body,
            text=(
                "提示：列表用逗号分隔（中英文逗号都认）。留空的曝光/增益表示不设置。"
                "标定、标定板尺寸这些只做参考的量请按现场量到的填。"
            ),
            wraplength=680,
            justify="left",
        )
        note.pack(fill="x", pady=6)

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="应用", command=self.apply).pack(side="left", padx=4)
        ttk.Button(buttons, text="关闭", command=self.window.destroy).pack(side="left")

    def collect(self) -> dict[str, str]:
        return {path: var.get() for path, var in self.entries.items()}

    def apply(self) -> None:
        from tkinter import messagebox

        try:
            # 先在副本上解析并校验，通过了才落到真配置上。不能先写再校验：
            # "速度超过上限"这种错误如果发生在写之后，用户会看到报错框，
            # 但配置里已经留下了那个超限的值——下一按"连设备"才炸。
            changes = apply_fields_with_validation(self.app.config, self.collect())
        except (UiError, ConfigError) as exc:
            messagebox.showerror("参数没通过检查", str(exc), parent=self.window)
            return
        if changes:
            self.app.append_log("参数已更新：\n  " + "\n  ".join(changes))
            self.app.set_status(f"参数已更新（{len(changes)} 项）")
        else:
            self.app.append_log("参数没有变化。")
        self.app.output_var.set(str(self.app.config.resolve_output_root()))
        self.window.destroy()


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def preflight(config: AppConfig) -> list[str]:
    """启动前的检查，返回人可读的几行。只报事实，不阻止启动。"""
    lines: list[str] = []
    try:
        ok, message = check_free_disk(
            config.resolve_output_root(), float(config.paths.min_free_disk_gb)
        )
        lines.append(("磁盘：" if ok else "磁盘不足：") + message)
    except Exception as exc:  # pragma: no cover - 目录都建不出来时才会走到
        lines.append(f"磁盘检查失败：{exc}")
    try:
        lines.extend(planned_scale_lines(config))
    except Exception as exc:  # pragma: no cover - 估算失败不该拦住启动
        lines.append(f"规模估算失败（不影响使用）：{exc}")
    runs = discover_runs(config.resolve_output_root())
    if runs:
        lines.append(f"这个输出目录下已有 {len(runs)} 次历史运行，最近一次：{runs[0].line()}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sj_pretest.ui",
        description="UR10 CB3 单关节微动预实验与正式实验一体化界面",
    )
    parser.add_argument("--config", help="启动时载入的配置文件（JSON）")
    parser.add_argument(
        "--mode",
        choices=("dry_run", "replay", "hardware"),
        help="覆盖配置里的模式。默认沿用配置，配置默认 dry_run。",
    )
    parser.add_argument("--replay-source", help="回放模式的数据目录或视频文件")
    parser.add_argument("--output-root", help="输出目录（每次运行会在下面建时间戳子目录）")
    parser.add_argument("--analysis-stride", type=int, default=8, help="离线识别抽帧步长")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只做启动检查并建一次窗口就退出（自测用，不进界面主循环）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = AppConfig.load(args.config) if args.config else AppConfig()
    if args.mode:
        config.mode = args.mode
    if args.replay_source:
        config.replay_source = args.replay_source
    if args.output_root:
        config.paths.output_root = args.output_root
    config.validate()

    for line in preflight(config):
        print(line)
    options = UiOptions(analysis_stride=max(1, int(args.analysis_stride)))
    app = ExperimentApp(config, options=options)
    if args.check:
        # 建了窗口、控件、日志就算通过；跑一遍事件循环再退出，确认没有延迟报错。
        app.root.update()
        app.destroy()
        print("界面自检通过（窗口已建立并销毁，未进入主循环）。")
        return 0
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - 手动启动时才走到
    sys.exit(main())
