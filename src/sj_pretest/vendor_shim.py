"""访问 :mod:`sj_pretest.vendor` 里被复用模块的唯一入口。

为什么不直接 ``import camera``
-----------------------------
被复用的 ``camera.py`` / ``robot.py`` / ``config.py`` 是按"平铺脚本目录"写的：
它们自己写 ``import config``，靠的是 ``sys.path`` 里有那个目录。
直接从本包各处 ``import camera`` 有两个隐患：

1. 名字太普通，一旦别处也有个 ``camera`` 模块，就会安静地 import 错；
2. 每个调用点都要自己记得先让 vendor 目录进 ``sys.path``。

所以统一走这里：校验"import 到的文件确实在 vendor 目录里"，不对就报错。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from . import VENDOR_DIR

#: vendor 目录里允许被取用的模块名。
ALLOWED_MODULES: tuple[str, ...] = ("analyze", "calibration", "camera", "config", "robot")


class VendorError(RuntimeError):
    """被复用模块不可用。消息中文。"""


def vendor_module(name: str) -> Any:
    """取 vendor 里的一个模块，并确认取到的是本工具自带的那一份。"""
    if name not in ALLOWED_MODULES:
        raise VendorError(
            f"{name!r} 不在允许的被复用模块清单里：{list(ALLOWED_MODULES)}"
        )

    if str(VENDOR_DIR) not in __import__("sys").path:
        raise VendorError(
            "vendor 目录没有被加入 sys.path。请先 import sj_pretest（包初始化会加）。"
        )

    module = importlib.import_module(name)
    module_file = Path(getattr(module, "__file__", "") or "").resolve()
    expected_dir = VENDOR_DIR.resolve()
    if module_file.parent != expected_dir:
        raise VendorError(
            f"import 到的 {name} 不是本工具自带的那一份：{module_file}\n"
            f"期望在 {expected_dir} 下面。"
            "这会把参数和结果写到错误的模块上，已中止。"
        )
    return module


def reload_vendor() -> None:
    """强制重新导入 vendor 模块。

    只有测试会用到：合成世界和相机来源之间切换时，想确认拿到的是干净模块对象。
    """
    for name in ALLOWED_MODULES:
        module = vendor_module(name)
        importlib.reload(module)
