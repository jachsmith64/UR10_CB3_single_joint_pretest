"""UR10 CB3 单关节微动预实验与正式实验一体化工具。

设计原则（对应需求一/二/三）：
1. **复用而不是重写**：相机采集、RTDE 通信、离线角点分析这些"已经跑通过"的
   代码原封不动放在 :mod:`sj_pretest.vendor` 里（见该目录的模块说明），
   新代码只做编排、序列规划、离线分析和界面。
2. **默认不动真机**：``AppConfig.mode`` 默认 ``"dry_run"``。任何真机运动都必须
   由用户在界面上显式切到 ``"hardware"``，并且每一步都有人工确认。
3. 三重可替换件（图像来源、机器人、时钟）都通过接口注入，
   所以同样的实验流程既能在真机上跑，也能在 dry-run / replay 下跑完整套。

导入这个包会自动把 ``vendor`` 目录放到 ``sys.path`` 末尾，好让被复用的
``camera.py`` / ``robot.py`` / ``config.py`` 里那句 ``import config`` 仍然成立
——那几个模块是按"平铺脚本 + 全局配置模块"的方式写的，不改动它们的前提
就是保持这个导入环境。
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "1.0.3"

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

if str(VENDOR_DIR) not in sys.path:
    # 放在末尾：万一将来本包内出现同名模块，优先用新代码，再说清楚来源。
    sys.path.append(str(VENDOR_DIR))

__all__ = ["__version__", "VENDOR_DIR"]
