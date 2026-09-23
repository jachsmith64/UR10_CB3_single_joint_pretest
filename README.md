# UR10 CB3 单关节微动预实验与正式实验一体化工具

**版本 1.0.2**（2026-09-23）

一个窗口、三个主按钮，把"到位 → 静态噪声 → 三档预实验 → 离线分析 → 确认步长 →
组A/组B 正式实验"这条链子跑完。默认干运行；真机运动必须人工逐步确认。

> 第一次拿到这个包，请先看 **[START_HERE.md](START_HERE.md)**（上手指引、现场参数、
> 时间与磁盘的诚实数字、干运行的已知局限）。本文件写给要读代码/跑自测的人。

---

## 安全边界（写在最前面）

* **没有碰撞模型**：`collision_status` 永远 `unknown`；理论范围检查 ≠ 碰撞安全。
* **默认 `dry_run`**：切到 `hardware` 必须有人在界面上明确点头。
* **每一步运动都要人工确认**，确认窗口之外不会自动运动。
* **中止后不自动回位**。
* 到位过程不能"一键无确认"连续执行——这是保留的安全条件，不是待办项。

## 安装与运行

```bat
install_deps.bat        :: 建 .venv 并装依赖（numpy/opencv/scipy/matplotlib/Pillow/pytest）
run_ui.bat              :: 打开界面（优先读 configs\local_site.json）
run_dry_run_example.bat :: 合成数据跑一整套流程，结果落在 outputs\
```

命令行等价物（`PYTHONPATH=src`）：

```bash
python -m sj_pretest.cli defaults  --out configs/experiment_default.json
python -m sj_pretest.cli dry-run   --joints J1 --amplitudes 0.01,0.05,0.2 --group A
python -m sj_pretest.cli reanalyze outputs/<运行目录> --stride 8
python -m sj_pretest.cli report    outputs/<运行目录>
python -m sj_pretest.cli replay-check <历史数据目录>
python -m sj_pretest.cli runs --root outputs
```

`dry-run` 子命令**内部强制** `mode="dry_run"`：命令行没有逐步确认的闸门，
所以它连 RTDE 都不会连。

## 目录结构

```
configs/experiment_default.json   默认配置（阈值、时长、现场参数都在里面）
docs/reused_modules.md            复用了哪些原始模块、为什么选那一份做基础
docs/dry_run_example/             一次真实干运行的落盘快照（文本为主）
docs/messages/                    BAT 里的中文提示（BAT 本身必须纯 ASCII，见下）
src/sj_pretest/
  vendor/        一行未改的被复用代码（camera/robot/analyze/calibration/config）
  vendor_shim.py 取用 vendor 的唯一入口（校验 import 到的是自带的那一份）
  bridge.py      把 AppConfig 翻译到 vendor/config.py 的属性上
  config.py      全部可调参数 + 校验 + 磁盘估算
  kinematics.py  名义运动学（只用于理论范围检查）
  joint_space.py 动作计划（到位/静态/快速确认/预实验/组A/组B）+ 计划自检
  acquisition.py 相位感知采集引擎
  sources.py     图像来源：相机 / 合成世界 / RAW 回放 / 图像目录
  robot_joint.py 机器人：真机 RTDE / 合成 / 回放
  recorder.py    原始数据落盘（14 项）
  experiment.py  会话编排（三个按钮背后的动作）
  analysis.py    三层分析与步长推荐
  vision.py      两条视觉通道（J1–J5 质心投影 / J6 二维旋转）
  replay.py      历史运行读取、复算（写新目录，绝不覆盖）
  ui.py          一体化界面
  cli.py         命令行
tests/           自测：不连任何硬件，全部跑在合成世界/回放上
outputs/         每次运行的时间戳目录（不进版本库）
```

## 跑自测

```bash
.venv\Scripts\python.exe -m pytest -q
```

自测的两条铁律（写在 `tests/conftest.py` 里）：

1. **不连真实机械臂**。所有流程测试跑在 `mode="dry_run"` 的合成世界上；
   合成世界渲染的是真棋盘格图，离线角点识别也是真跑，被替换掉的只有
   "机械臂 + 相机"这一层物理。有一条测试专门断言整个界面流程跑完
   `ur_rtde` **没有被 import**。
2. **不为通过而放宽判据**。测试配置只改"规模"和"时长"（关节数、幅度档数、
   重复次数、帧尺寸/帧率），**不改任何阈值**——SNR 门限、方向容差、残差倍数
   全部保持交付默认值。如果某条断言只有放宽阈值才能过，那是代码的问题。

界面测试**不弹任何真窗口**：所有 `messagebox` 都被换成记录器。模态框是阻塞的，
自测把它弹到操作者屏幕上再卡住等人点，本身就是缺陷。

## 默认值的来历

默认值不是随手填的，逐条说明：

| 参数 | 默认 | 来历 |
|---|---|---|
| 名义位姿 | J1 −53.8365°、J2 −175.6992°、J3 148.2887°、J4 −137.4699°、J5 −17.5120°、J6 −112.4909° | 需求二给定的推荐姿态 |
| 摆放提示 | 水平、约 675 mm、方位角 33.3°、面内偏差 14.75° | 需求二给定的相机摆放提示；**只是提示**，没做碰撞验证 |
| 三档幅度 | 0.01 / 0.05 / 0.2° | 需求二给定的预实验幅度 |
| `robot.approach_speed_deg_s` | 1.0 °/s | 由旧项目真机验证过的**最慢档**（0.02 m/s 直线）按约 1 m 臂展折算 |
| `robot.trial_speed_deg_s` | 0.5 °/s | 比旧项目最保守档还慢一倍；不再更慢的原因见 `config.py` 里的注释 |
| `formal.speed_deg_s` | 0.05 °/s | **故意比预实验更慢**：组A 行程是预实验的 N 倍，走慢一点让运动自身激起的振动更小 |
| `robot.settle_tolerance_deg` / `hold_s` / `timeout_s` | 0.002° / 0.3 s / 8 s | "停稳"的判据；不影响采集，只影响什么时候开始算保持段 |
| `robot.rtde_record_hz` | 125 Hz | 旧项目真机用的频率 |
| `camera.expected_fps` | 132.23 | 旧项目真机实测帧率 |
| `camera.board_inner_corners` / `square_mm` | 11×8 / 3 mm | 12×9 方格 → 88 个内角点；小格 3 mm |
| `camera.working_distance_mm` | 675 mm | 需求二的竖直工作距离提示；只用来把棋盘格"变大/变小多少"换算成轴向位移（见「轴向/面内」） |
| `camera.static_duration_s` | 15 s | 需求三允许 10～20 s |
| `camera.hold_s` | 1.5 s | 保持段是整套预实验里最有信息量的一段（稳定值 + 残余振动） |
| `camera.pre_motion_s` / `post_motion_s` | 0.3 s | 各约 40 帧：运动前参考帧、运动后残差观察段 |
| `pretest.quick_probe_deg` | 0.2° | 快速几何确认的幅度。**不用 0.05°**：0.05° 在 J1 上只造成约 0.375 mm 面内位移，和轴向可分辨极限同量级，方向判不出来（见「轴向/面内」） |
| `thresholds.*` | 见 `config.py` | 全部可在参数面板里改；改完会存进配置 JSON |
| `paths.min_free_disk_gb` | 2 GB | **绝对下限**；真正把关的是"按计划估算占用"的检查（见下） |
| `paths.delete_raw_after_process` | `false` | 分组流水线开关：一组动作采完 → 停下 → 整组处理并校验 → 删掉这一组的 RAW → 才走下一组。**默认关**（默认路径全程留 RAW），代价见 `START_HERE.md` |
| `paths.process_stride` | 1 | 处理步长。**默认逐帧**：删除 RAW 前必须已经生成 132 Hz 的逐帧角点和视觉结果，所以正式保留数据的路径固定用 1；`4` 只留给现场快速预览（`analyze_offline(stride=4)`），不能用它做删除前的最终提取 |
| `paths.max_peak_disk_gb` | 50 GB | **硬上限**：分组流水线下任何一组的估算占用超过它就**不允许开始这一组**（一条运动命令都不会发出去） |
| `paths.disk_warn_gb` | 40 GB | **软件预警线**：超过就告警并提示换 ROI / 清盘。定 40 而不是 38，是因为 950×800 的实测最坏一组是 35.53 GB（见下表），交付的两种 ROI 都不该自己触发预警 |
| `paths.derived_overhead_ratio` | 0.05 | 派生数据（角点 CSV、逐帧 JSON、样本图）相对 RAW 的体量比例，算组占用时一并计入 |

**实测的组峰值**（真机参数：132.23 fps、Δ=0.2°、阶梯 5 档、重复 3 次）：

| ROI | 最坏一组 | 整场总和 |
|---|---|---|
| 800×600 | 22.44 GB（组A J1） | 233 GB |
| 950×800 | 35.53 GB | 369 GB |
| 1936×1096（满幅） | 99.18 GB | 1031 GB |

所以满幅 ROI 在分组流水线下会**被 50 GB 硬上限拒绝**——这是有意的，不是 bug；
要用满幅就必须先关掉分组流水线并按整场总和准备磁盘。

## 几处值得单独说的实现

**磁盘不是看一个固定门槛。** `paths.min_free_disk_gb` 只是下限；
每一段采集**开始之前**都会按这一段自己的计划估算要写多少 GB
（`config.check_plan_disk`），乘 15% 余量后和可用空间比，不够就**拒绝开始这一段**。
理由：满幅 1936×1096 @ 132 fps 约 280 MB/s，一趟完整实验是几百 GB 量级，
固定门槛定成 2 GB 等于没有门槛——等磁盘写满时人已经离开一小时了。
分组流水线打开后，每组开头还有一道更严的 `check_group_disk`（50 GB 硬上限 +
40 GB 预警线，只按这一组算），两道都过才动。

**分组流水线：先处理完、校验过，才删。** 打开
`paths.delete_raw_after_process` 之后，流程不再是"采一段处理一段"，而是**按组**走：

```
begin_group(组名, 这一组的全部段)      ← 先按组的估算查盘，不够就直接拒绝，一条命令都不发
  → 采集组内每一段 RAW（stage_segment 只登记，不处理）
  → 机械臂在这组动作上停住
  → flush_group：一次读完这组的 RTDE 行，逐段 process_and_release
      （逐帧角点 + segment_vision.json + 二维转角/尺度/时间戳 + 本段分析）
  → 逐段回读校验（见下），通过才删这一段的 RAW，不通过就保留并中止会话
  → 才进入下一组
```

分组的划分是固定的：**静态基线一组、快速几何检查一组、三档预实验每个关节一组、
正式实验每个关节的组A / 组B 各一组**。这样任意时刻盘上只有"当前这一组"的 RAW
加派生数据，实测最坏一组就是上表里的数——而不是整场总和。事件流里
`group_started → segment_captured* → group_processing → raw_deleted` 的顺序是
可检查的，`tests/test_group_pipeline.py` 就是沿事件流走一遍来验证这件事。

**删之前校验的是六件事，不是"至少有一帧认出来"。** `_verify_releasable` 逐条查：

1. `valid_ratio >= thresholds.min_valid_frame_ratio`（有效帧比例）；
2. `capture_metadata.json` 里的相位表存在，且**运动前/运动/保持/回程/运动后**
   每个声明阶段都有 ≥ `min_window_frames` 帧有效——只有"中间那些帧"认出来是不够的；
3. `segment_vision.json` 和角点 CSV 能**回读**，且回读出来的量与刚算完的一致
   （含质心位移与 `r_rms_px`）；
4. 相机时间戳 CSV（`frame_timestamps.csv`）存在；
5. RTDE 行存在（按 `event_id` 或主机时间窗匹配，取两者中较大的那个数）；
6. 本段的基本分析（`analysis.check_segment_analyzable`）能正常跑完。

**任一条不通过：RAW 保留、写 `raw_kept`、会话中止、等人工处理。** 不通过时绝不是
"少删一个文件"，而是"这一段的数据此后不可复算"。

**RAW 删掉之后不许再回头读它。** 删 RAW 之前的处理是唯一一次读
`frames.raw`；之后的快速几何检查、J6 角度、动态过程一律复用已经生成的
`SegmentVision`（或从 `segment_vision.json` + 角点 CSV 读回）。
`SegmentVision.subsampled(step)` 保证"对完整逐帧数据取每 N 帧"和
`process_segment(stride=N)` 逐位相同，所以"RAW 在"和"RAW 不在"两条路径的结果
一模一样——`tests/test_rolling_delete.py` 里有一条测试直接断言这一点，
并且用 `process_segment` 的调用计数证明删完之后没有第二次读 RAW。

**计划自检先于发命令。** `joint_space.verify_plan` 会逐条核对：事件编号唯一、
正负方向标签正确、每个统计动作只动目标关节、每次都从名义位姿独立出发
（不累计）、回程不计入统计。计划层错了就不要发命令。

**单位只在最后一刻换。** 内部、日志、界面、以及"停稳"判据**全部用度**；
换成弧度只发生在 `HardwareJointRobot.move_to_joint` 里的一处，安全范围检查和
`moveJ` 拿到的都是同一个弧度列表。ur_rtde 是 SI 单位制（rad），1.0.0 里把度
直接喂给了它——0.2° 会被当成 0.2 rad（≈11.5°），每条试运动都比计划大 57 倍，
而且安全范围检查查的也是被放大 57 倍的角度。这条有独立测试用手写的假控制器
盯着（`tests/test_hardware_units.py`），它断言的就是"送进 `isJointsWithinSafetyLimits`
和 `moveJ` 的是弧度，而日志/界面/停稳判据仍是度"，且全程不 import ur_rtde。

**ROI 只有一份。** `CaptureEngine.crop` 是**唯一**的裁剪实现，到位预览、
棋盘格完整性/余量检查、快速几何检查、RAW 落盘四处都调它；RAW 里存的就是
裁过的图（`capture_metadata.json` 里同时记 `source_size` 和 `cropped_size`）。
界面上的预览图因此和 RAW 是同一张画面，并同时显示
「原始画面尺寸 / ROI 坐标 / 裁剪后尺寸」三个数。1.0.0 里预览用的是**未裁剪**
的整幅图，于是会出现"界面上棋盘格好好的、离线识别却在裁过的图里找不齐角点"。

**轴向/面内怎么分开。** 棋盘格在画面里只有"变大/变小"能反映**轴向**位移，
所以用相似变换的尺度做一阶估计：`depth_mm ≈ working_distance_mm × |scale − 1|`；
面内位移用内角点质心的二维位移乘现场的 mm/px。判据是
`thresholds.max_depth_ratio`：轴向/面内比超过它就**停下来**，不往下跑。
尺度噪声取前后静态窗的标准差，可分辨极限取 3σ；当 `|scale − 1|` 小于这个
极限时，工具显示**无法判断**，并改用**上界**（`working_distance_mm × max(|scale−1|, 3σ)`)
去过判据——这样"看不出来"绝不会被当成"轴向很小"而放过去。
干运行合成世界实测：尺度噪声约 ±6.3e-5 → 675 mm 处可分辨约 0.128 mm，
J1 在 0.2° 下面内约 1.5–1.7 mm，比值上界 0.086–0.098 ≪ 0.5，判为以面内为主。

**J6 的面内量不能只看质心动了多少。** 棋盘格不保证装在 J6 旋转轴的**中心**上，
偏心 5～15 mm 是正常装配。偏心时 J6 一转，质心会**沿圆弧走**一大截，
而棋盘格本身可能几乎没平移——把质心位移当成"面内运动"会把纯旋转误判成"轴向有问题"，
也会让"棋盘格正好居中（质心不动）"的纯旋转被误判成"没动"。
所以 J6 的面内量按刚体面内位移合成：

```
r_rms_px   = sqrt(mean(|pᵢ − 质心|²))          # 角点到质心的均方根半径
d_rot_px   = 2 · r_rms_px · sin(|θ| / 2)       # 纯转动引起的面内线位移（弦长）
d_plane_px = sqrt(质心位移² + d_rot_px²)        # 平动 ⊕ 转动
d_plane_mm = d_plane_px · mm_per_pixel         # 用来和 depth_mm 比
```

`d_rot_px` 用的是**弦长**而不是"弧长÷偏心距离反推"：偏心量在真机上是未知的，
弦长只依赖已经量到的 `r_rms_px` 和转角，居中与偏心都成立
（居中时 `r_rms_px` 仍等于棋盘格自身半径，纯转动照样给出非零的 `d_rot_px`）。
**J6 的角度本身仍然用 88 个角点去质心之后的二维 Kabsch 旋转结果**，
不用质心位移反推——质心圆弧位移要除以假定偏心距离，那个距离恰恰是不知道的。

`tests/test_j6_eccentric.py` 覆盖：偏心 0 / 5 / 10 / 15 mm 的纯旋转全部判为
"以面内为主"；居中棋盘的纯旋转**不会**被当成没动；转动 + 平动的合成量
≥ 各自分量且 ≈ 两者平方和开根；同样的尺度变化在 J1–J5 上仍走原来的质心判据；
明显尺度变化会被轴向判据拦下；J6 角度能从角点 CSV 复算出与 JSON 一致的值；
以及 RAW 删掉之后 J6 角度和动态过程仍可复算。

## 版本说明

* **1.0.2**（2026-09-23）定点修复，**不重构、不改 vendor**。五组，全部只动
  `experiment.py` / `vision.py` / `config.py` / `ui.py` 和自测：
  1. **RAW 分组流水线**：采集按"组"走，一组采完机械臂停住 → 整组处理并落盘
     → 回读校验 → 删这一组的 RAW → 才走下一组。分组固定为静态基线、快速几何检查、
     预实验每关节一组、正式实验每关节的组A/组B各一组。打开开关后任意时刻盘上
     只有当前一组的 RAW，配合 800×600（最坏 22.44 GB）或 950×800（最坏 35.53 GB）
     的 ROI，峰值远在 50 GB 硬上限和 40 GB 预警线之内；超过估算空间时
     **不得开始下一组**（`_require_group_disk` 抛 `ExperimentError`，一条运动命令都
     没发出去，清完盘还能接着用这个会话）。
  2. **修掉删 RAW 后重复读取**：`_measure_probe` 原来无条件调 `process_segment`
     去读 `frames.raw`，边采边清把它删掉之后，**同一个动作组刚采完就检查不了**。
     现在后续计算只复用已经生成的 `SegmentVision` 或读 `segment_vision.json` /
     角点 CSV；新增端到端测试证明 `delete_raw_after_process=true` 下
     `run_quick_probes` 全程只调 `process_segment` 每段一次。
  3. **删除前的校验**：从"至少 1 帧识别成功"改成六条硬条件——有效帧比例、各必要
     阶段（运动前/运动/保持/回程/运动后）都有足够有效帧、`segment_vision.json`
     与角点 CSV 能回读且数值一致、相机时间戳存在、RTDE 行存在、本段分析能跑完。
     任一条不过就保留 RAW 并暂停等人工处理。
  4. **处理步长**：`paths.process_stride` 默认 4 → **1**。删除 RAW 前必须已经生成
     132 Hz 的逐帧角点和视觉结果；`stride=4` 只能用于现场快速预览。
  5. **J6 偏心棋盘格几何**：J6 面内量改用
     `d_rot_px = 2·r_rms_px·sin(|θ|/2)`、`d_plane_px = √(质心位移² + d_rot_px²)`
     合成，`d_plane_mm` 作为轴向/面内比的分母；J6 角度仍用去质心后的二维
     Kabsch 结果，不用"质心圆弧位移÷假定偏心距离"反推。偏心 5～15 mm 不再
     被误判，居中棋盘的纯旋转也不再被当成"没动"。
  **自测**：新增 `tests/test_group_pipeline.py`（9 条）与 `tests/test_j6_eccentric.py`
  （13 条），`tests/test_rolling_delete.py` 增补 2 条用例。
  全套 16 个自测文件 **203 条用例**，连续两次全绿。
  新用例都做过反向验证（把被修的行为改回旧写法，它们会红）：
  `stride` 改回 4 → 分组峰值用例红；`_measure_probe` 改回无条件 `process_segment`
  → 调用计数用例红；J6 面内量退回质心平移 → 偏心和居中两组用例红。
* **1.0.1**（2026-09-23）定点修复，**不重构、不改 vendor**。六项：
  1. **真机角度单位**（严重）：度被直接喂给 ur_rtde 的弧度接口，试运动幅度被放大
     57 倍，安全范围检查查的也是错的角度。现在只在 `move_to_joint` 里换一次弧度。
  2. **停稳超时**：超时后立刻 `stopJ`、本段判失败、**不继续 hold、不回程、不自动回位**、
     已写的 RAW/RTDE/时间戳全部保留，界面明确提示人工检查。
  3. **ROI 统一**：四处共用 `CaptureEngine.crop`；界面显示原始尺寸/ROI/裁剪后尺寸
     与**真实裁剪预览**；ROI 越界或裁剪后棋盘格不完整一律**拒绝开始**。
  4. **轴向/面内真正实现**：新增 `camera.working_distance_mm`（默认 675）与
     `thresholds.max_depth_ratio` 判据；输出 depth_mm、in_plane_mm、比值、有效帧数、
     置信状态；分辨不出来时明说**无法判断**并用上界过判据。`quick_probe_deg` 默认
     0.05° → **0.2°**（仍可在界面改），不做逐步自动搜步长。
  5. **其他**：组A 和组B 开始前**都**做理论范围检查；每个关节成组动作前按实际 ROI
     复查棋盘格；修掉"可以关掉 `save_raw`"的错误说法（本工具**不提供**这个选项）。
  6. **自测**：新增硬件单位边界、停稳超时无回程、ROI 预览与 RAW 裁剪一致、
     `max_depth_ratio` 既能通过也能拦截四类测试；另加"边采边清删了 RAW 之后
     离线复算结果不变"一条。全套 14 个自测文件 **176 条用例**，连续两次全绿。
     四条新测试都做过反向验证（把被修的行为改回旧写法，它们会红）。
  另新增**边采边清**选项 `paths.delete_raw_after_process`（默认关）：段处理完、
  校验通过即删该段 RAW，盘上只留当前一段视频；代价与保留规则见 `START_HERE.md`。
* **1.0.0**（2026-09-23）首个交付版本。包含：界面（三个主按钮 + 参数面板）、
  命令行、干燥/回放/真机三种模式、相位感知采集、两层视觉通道的三层分析、
  步长推荐、组A/组B 正式实验、复算、9 个自测文件（137 条用例）。
* 已知局限（都写在 `START_HERE.md` 的「干运行的三个已知局限」）：合成世界
  棋盘格只有 16 px/格，J6 的 0.05° 快速确认在干运行下会报"看不出来"；
  干运行里机器人一开始就在名义位姿，到位过程是 6 步零位移。
* 未验证项：**真机**上的相机取流、RTDE 通信、实际运动与安全行为——
  交付前按需求一的硬约束，全部自测都在 dry-run / 回放上完成，**没有连接过真实机械臂**。
