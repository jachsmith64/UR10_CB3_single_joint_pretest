# SELF_REVIEW —— v1.0.3 实验室候选版（给复核者）

这份文件只说三件事：**改了什么、自己查出来的问题按 A/B/C 怎么分的、哪些事本机
验证不了**。不是验收报告，不重复 README 里的操作说明。

* 基线：`UR10_CB3_single_joint_pretest_v1_src_v1.0.2.zip`（唯一基础，没有从更旧的版本继承）
* `src/sj_pretest/vendor/` 下五个被复用的文件 **一个字节都没改**（打 ZIP 时未改动文件
  逐字节从 v1.0.2 包里搬运，vendor 五个文件走的就是这条路；改动过的文件按
  `.gitattributes` 的 LF 策略重写，所以 vendor 与 v1.0.2 逐字节相同这件事可以在两个包之间直接比对）
* 自测：全套 **278 条，四次运行全部通过、无 skip 无失败**（含一次随机顺序，
  以及从交付 ZIP 解压后重跑）。明细见文末「测试与验证」。
  **本机没有连过真实 UR10，也没有连过真实海康相机**——所有"真机"结论都来自
  dry-run / mock / 历史数据，文末单独列了哪些话不能说。

---

## 一、按需求一·1–7 逐条对应的行为变化

| 需求 | 落在哪里 | 关键点 |
|---|---|---|
| 一·1 默认删 RAW | `config.py` `paths.delete_raw_after_process=True`；`configs/experiment_default.json`；`ui.py` 初始显示；`docs/dry_run_example/config.json`；README / START_HERE | 五处同步；界面顶行显示"分组处理并删除RAW：已开启"；删除前的六项校验一条都没放宽 |
| 一·2 全屏采集 | `acquisition.py` `raw_frame()`（原 `crop()` 的裁剪路径整段删掉）、`analysis_crop()`；`vision.py` 只在离线识别时把 ROI 交给 vendor；元数据加 `source_size` / `raw_size` / `analysis_roi` / `roi_origin` / `corners_frame` | RAW 尺寸与 ROI 无关；导出坐标一律原图坐标；磁盘估算按整幅 |
| 一·3 5 s 采集检查 | `experiment.py::run_connection_check`（在 `connect_devices()` 里、**任何运动命令之前**）；UI 重做按钮 | 五项实测值 + 三条失败分支（帧率不足 / 掉帧 / 写盘不足）；门槛可配 |
| 一·4 重新分组 | `experiment.py::repeat_segment_groups` / `nominal_split_segment_groups` / `plan_formal_groups` / `plan_pretest_groups` | 正式 A/B **每遍一组**；预实验每关节一组；超限只在"已回到名义姿态"处再拆；拆不动则**运动前**拒绝 |
| 一·5 时长与磁盘估算 | `joint_space.py::estimate_plan`（按"上一目标姿态"逐段）、`config.py::group_forecast` | 去程/回程/阶梯/换向/保持/pre-post/settle 全覆盖；不写 RAW 的纯等待步骤不计入录制时长 |
| 一·6 RAW 状态顺序 | `experiment.py::process_and_release`、`vision.py::save_vision_json`（临时文件 + fsync + 原子替换） | 先写 true → 校验 → 不过就保留 → 过了才删 → 确认没了 → 原子改写 false |
| 一·7 处理期状态 | `recorder.py::pause/resume`、`experiment.py::flush_group`、`acquisition.py::drain_frames` | RTDE 暂停/恢复行数相等；处理期"已发运动命令"快照比对；相机丢帧重建帧号基线 |

---

## 二、A 类：实验前必须修改的问题（**已修，并各配了测试**）

### A1. 5 s 采集检查会把"期望帧率"当成"实测帧率"，一台跑不动的相机也会判通过

* 现象：`CaptureEngine` 拿到的帧源是 `FramePump` 这个包装对象，
  `getattr(pump, "actual_camera_fps", None)` 取不到就回退到
  `config.effective_fps()`——**配置里的期望帧率**。
* 后果：需求一·3 的核心判据（实际帧率 ≥ 95% 期望帧率）变成"自己比自己"，
  永远通过；一台真实只跑 90 fps 的相机在这一步会被报成 132.23 fps。
  同理，帧时间戳与帧号的对应关系也会拿错帧率去核对。
* 修法：`FramePump` 保留**来源对象**，把 `actual_camera_fps` /
  `actual_camera_fps_source` 透传出去（真机侧来自 GenICam 的
  `ResultingFrameRate`，每帧刷新，不缓存）。
* 附带修掉同一处的第二个问题：`FramePump.close()` 对**迭代器**调 `__exit__`，
  而迭代器上没有这个方法——真机上相机句柄会悬着不释放。现在对来源对象调。
* 测试：`tests/test_capture_check.py`（含"慢相机必须判不通过"）。
* 反向验证：把透传改回 `getattr(self.source, ...)`（即旧写法）→ 慢相机用例红。

### A2. 50 GB 闸门比错了量：界面上显示的是"RAW + 派生"，真正拦的是同一个数，
### 而需求二写的是"RAW + 临时文件 + 派生"

* 现象：`GroupForecast.under_cap` 与 `check_group_disk` 拿 `resident_gb`
  （RAW + 派生）去比 50 GB，而 `process_peak_gb = resident × process_headroom(1.15)`
  才是"处理期同时驻留"的量。
* 后果：驻留 49 GB 的一组会被放行，而它处理时的估算峰值是 56 GB——
  闸门被悄悄放宽 15%，且界面上的"是否低于硬上限"那一行与实际执行的判据用的是
  两个数（这正是"显示与实际执行不一致"）。
* 修法：`under_warn` / `under_cap` / `check_group_disk` 统一改比
  `peak_gb = process_peak_gb`；`group_peak_gb()` 的返回口径在 docstring 里写明
  "这是 RAW + 派生的驻留量，不含处理期临时余量，别拿它比 50 GB"。
* 测试：`tests/test_group_pipeline.py::test_the_delivered_group_peaks_stay_under_the_warning_line`
  （逐组断言 `under_cap == (peak_gb <= cap)`、`under_warn == (peak_gb <= warn)`、
  `resident_gb < peak_gb`，交付参数下最坏一组是正式组B 的 38.49 GB）。
* 反向验证：把比较改回 `resident_gb` → 该用例红。

### A3. 预实验没有"名义姿态再拆一级"的兜底，超限直接拒绝

* 现象：只有正式实验会先尝试在名义姿态边界再拆一级；预实验一超限就抛拒绝。
* 后果：预实验的**每一次动作都是从名义姿态独立出发再回来的**——每一段边界都是
  安全切点，本来可以拆；直接拒绝等于让现场无谓地失去一个可用选项。
* 修法：抽出共享的 `_over_hard_cap` / `_over_cap_groups` /
  `_refine_at_nominal_boundaries` / `_over_cap_refusal`，预实验走
  `plan_pretest_groups`，规则与正式实验完全一致；只有"连单次动作都放不下"
  才拒绝，且拒绝理由说明"单次动作本身已是最小安全单位"。
* 测试：`tests/test_formal_repeat_grouping.py` 末尾四条（默认每关节一组、
  超限时在名义边界再拆、拆不动则运动前拒绝、真实会话确实按拆完的组跑）。
* 反向验证：去掉预实验的 refine 分支 → "超限时再拆"用例红。

### A4. `_over_hard_cap` 曾把"盘满"误报成"这一组太大"

* 现象：判断"要不要再拆一级"时，一度把"可用空间不够"也算成"超上限"。
* 后果：盘满时工具会说"这一组太大、不能再拆"——现场会去改参数/减动作，
  而正确的处置是清盘。两件事用同一句话，等于给了错的处置方向。
* 修法：`_over_hard_cap` **只看尺寸、不看可用空间**；盘满仍由
  `_require_group_disk` 用"按计划需要约 X GB、可用空间不够"的话拒绝。
* 测试：`tests/test_group_pipeline.py::test_a_group_over_the_estimate_is_refused_before_any_motion`
  就是构造 `min_free_disk_gb` 极大、断言拒绝语是"按计划需要约…"而不是上限那一套。

### A5. 报告里"静态基线多少秒"写的是**离线处理耗时**，不是采集时长

* 现象：`StaticNoise.seconds` 直接取 `SegmentVision.seconds`，而后者是
  `time.perf_counter()` 的差值（"这一段算了多久"）。
  一次 3.0 s 的静止采集在报告里被写成 **14.52 s**。
* 后果：这是要写进论文方法段的量。单位没错、量级也像，但量错了对象；
  而且它随算力浮动（换台机器、改处理步长数字就变）。
* 修法：`SegmentVision` 增加采集口径的 `captured_seconds` /
  `captured_frames`（取自 `capture_metadata.json` 的 `content_seconds` /
  `frame_count`，删 RAW 不会删它；老段目录缺字段时退回本段分析时间轴跨度，
  **绝不**退回 wall clock）；`StaticNoise.seconds` 改用采集口径，处理耗时
  单独放进 `process_seconds`；文本改成
  "本段录了 397 帧 / 2.99 s；参与统计 50 帧（按处理步长抽样）"。
  `subsampled()` 照抄采集口径——抽帧少的只是参与统计的帧。
* 测试：`tests/test_analysis.py::test_the_baseline_duration_in_the_report_is_the_capture_not_the_processing`
  （含"抽帧后采集帧数不许变小"）。
* 反向验证：把 `seconds` 改回 `float(segment.seconds)` → 用例红
  （实测 2.79 s vs 1.19 s）。

### A6. 连接检查的汇总把"0×0 @ 0.00 fps"写成结论

* 现象：检查失败（一帧都没接住）时，汇总行仍然按"分辨率 W×H @ 帧率"的模板排版，
  印出 `0×0 @ 0.00 fps`，紧跟着的结论却是"不通过"。
* 后果：现场看到的是自相矛盾的一行，第一反应是怀疑工具而不是相机。
* 修法：汇总按"有没有量到东西"分两种写法，没量到就直接说"一帧都没接住，
  请检查相机是否在出图 / 触发是否配错"，不再印 0 值模板。
* 测试：`tests/test_capture_check.py` 的零帧分支。

---

## 三、B 类：可以以后修改（**本次不动**）

1. `group_started.segments` 计的是"本组计划里的段数"，与随后
   `group_processing.segments`（实际处理段数）在含纯等待段时相差 1
   （例：组A 一遍 = 2 去程 + 1 回程 + 1 等待 → started 4 / processing 3）。
   `group_forecast` 的 `action_count` 已正确排除等待段，事件流对账不受影响。
   措辞上把两者都叫 `segments` 容易让人以为丢了段，建议改名为
   `planned_segments` / `processed_segments`。
2. `analysis/pretest_report.txt` 里静态基线那一行现在偏长（"本段录了…；参与统计…"），
   排版可以再收一收。
3. `refresh_dry_run_snapshot.py` 与 `build_delivery_zip.py` 里各有一小段
   重复的"文本/二进制后缀"判断，可以合到一个模块里。
4. `docs/dry_run_example/` 里 `vision/corners/*.csv` 单张 1.5 MB（stride 8 下仍很大），
   快照体积主要来自它；真要压可以只留前若干行，但要另加"这是节选"的标注。
5. 干运行里 J6 的 0.2° 快速确认在合成尺度下接近可分辨极限，同样的配置有时判
   "以面内为主"、有时判"看不出来"。已在 `START_HERE.md` 的"干运行的已知局限"
   里写明；这不是待修的缺陷（真机棋盘格成像大得多），但**不建议**把它当成
   真机结论的预演。

## 四、C 类：与本实验无关，**未改动**

SFC 控制、MuJoCo 仿真、旧微动闭环工程、完整数字孪生、新的复杂视觉算法、
与本任务无关的代码风格重构——全部没碰。

---

## 五、测试与验证

* 命令与顺序（工程根目录，`PYTHONPATH=src`）：
  `python -m compileall -q src tests` →
  `pytest -q -p no:randomly` →（再跑一次）→ `pytest -q`；
  最后把交付 ZIP 解到新的临时目录，在**解出来的那份代码**上再跑一次全套。
* 结果：全套 **278 passed**，**没有 skip、没有失败**。开发过程中连续三次全绿
  （含一次 pytest-randomly 随机顺序：592.11 s / 587.44 s / 588.47 s），
  之后才打 ZIP；随后把交付 ZIP 解到新的临时目录，在**解出来的那份代码**上
  又跑一遍，同为 278 passed、无 skip、无失败。
  （这里不写第四次的秒数：那一趟的耗时对复核没有意义，写死了反而会跟
  包外重跑的结果对不上。）
* 反转验证（把被修的行为改回旧写法，用例必须红）：A1、A2、A3、A5 四条都做过，
  见上面各条末尾。
* 硬件依赖的覆盖方式：真机 RTDE 与海康相机用 mock/合成来源覆盖
  （`tests/test_hardware_units.py`、`test_capture_check.py`、`test_processing_state.py`）；
  这些用例证明的是"链路与判据"，**不是**"真机跑过了"。

## 六、本机验证不了、所以不能声称的事

1. 没有连过真实 UR10 CB3，也没有发过任何真实 RTDE 运动命令。
   `collision_status` 在全部输出里仍然是 `unknown`。
2. 没有连过真实海康相机。真机满幅 1936×1096 @ 132.23 fps 以及
   2 122 336 字节/帧、约 280 MB/s 这些数来自相机手册与既有工程，
   不是本机测出来的；真机上的"实际帧率 / 实际写盘速度"要以现场那一次
   5 s 全屏采集检查的输出为准。
3. 交付里的分组尺寸（静态 3.92 GB、快速几何检查 14.70 GB、预实验 12.32 GB、
   组A 每遍 15.94 GB、组B 每遍 31.88 GB / 峰值 38.49 GB，Δ=0.2°）是
   按上面的名义字节率与**实测动作时长公式**算出来的估算值；
   真机上控制器怎么规划会让它们浮动。闸门本身（50 GB / 40 GB）在执行时
   用的是现场 5 s 检查的实测分辨率与帧率。
4. 干运行快照（`docs/dry_run_example/`）里的位移、噪声、方向都是合成世界
   算出来的，只能证明"链路是通的、判据是活的"。
5. 最长的一次端到端是干运行（约 2 分钟）；真机整场 60 分钟量级的连续运行
   没有做过。
