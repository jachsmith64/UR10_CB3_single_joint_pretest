# UR10 CB3 单关节微动预实验与正式实验一体化工具

**版本 1.0.0**（2026-09-23）

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
| `camera.static_duration_s` | 15 s | 需求三允许 10～20 s |
| `camera.hold_s` | 1.5 s | 保持段是整套预实验里最有信息量的一段（稳定值 + 残余振动） |
| `camera.pre_motion_s` / `post_motion_s` | 0.3 s | 各约 40 帧：运动前参考帧、运动后残差观察段 |
| `thresholds.*` | 见 `config.py` | 全部可在参数面板里改；改完会存进配置 JSON |
| `paths.min_free_disk_gb` | 2 GB | **绝对下限**；真正把关的是"按计划估算占用"的检查（见下） |

## 两处值得单独说的实现

**磁盘不是看一个固定门槛。** `paths.min_free_disk_gb` 只是下限；
每一段采集**开始之前**都会按这一段自己的计划估算要写多少 GB
（`config.check_plan_disk`），乘 15% 余量后和可用空间比，不够就**拒绝开始这一段**。
理由：满幅 1936×1096 @ 132 fps 约 280 MB/s，一趟完整实验是几百 GB 量级，
固定门槛定成 2 GB 等于没有门槛——等磁盘写满时人已经离开一小时了。

**计划自检先于发命令。** `joint_space.verify_plan` 会逐条核对：事件编号唯一、
正负方向标签正确、每个统计动作只动目标关节、每次都从名义位姿独立出发
（不累计）、回程不计入统计。计划层错了就不要发命令。

## 版本说明

* **1.0.0**（2026-09-23）首个交付版本。包含：界面（三个主按钮 + 参数面板）、
  命令行、干燥/回放/真机三种模式、相位感知采集、两层视觉通道的三层分析、
  步长推荐、组A/组B 正式实验、复算、9 个自测文件（137 条用例）。
* 已知局限（都写在 `START_HERE.md` 的「干运行的三个已知局限」）：合成世界
  棋盘格只有 16 px/格，J6 的 0.05° 快速确认在干运行下会报"看不出来"；
  干运行里机器人一开始就在名义位姿，到位过程是 6 步零位移。
* 未验证项：**真机**上的相机取流、RTDE 通信、实际运动与安全行为——
  交付前按需求一的硬约束，全部自测都在 dry-run / 回放上完成，**没有连接过真实机械臂**。
