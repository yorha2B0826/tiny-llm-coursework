"""测试环境兼容层（本仓库为纯 Python 实践仓库，不含已构建的原生扩展）。

`tiny_llm/__init__.py` 会 import `week2_kernels`，后者在模块级导入
`extensions.tiny_llm_ext`（需要在完整课程仓库中构建的扩展）。

Week 2 Day 2 的测试还依赖课程 harness（`benches/`、`tiny_llm_ref`），
其中 `tiny_llm_ref` 在模块级导入参考扩展 `extensions_ref.tiny_llm_ext_ref`。

本仓库只包含扩展的 C++/Metal 源码，没有构建产物（`_ext*.so` 需在 macOS 上
用 `pdm run build-ext` / `pdm run build-ext-ref` 现场编译）。这里给未构建的
扩展包挂上惰性报错的 `_ext`，让导入链正常工作；一旦真正调用扩展算子，
会得到明确的报错提示。
"""

import importlib
import sys
import types


class _MissingExtension:
    def __init__(self, module: str) -> None:
        self._module = module

    def __getattr__(self, name):
        raise RuntimeError(
            f"课程扩展 {self._module} 未构建：Week 2+ 的算子测试需要先在完整课程"
            f"仓库中构建扩展（pdm run build-ext / pdm run build-ext-ref）。"
            f"（尝试访问：{name}）"
        )


def _register(package: str, module: str, leaf: str) -> None:
    try:
        imported = importlib.import_module(module)
    except ImportError:
        imported = None

    if imported is not None and hasattr(imported, "_ext"):
        return  # 已构建的真实扩展

    if imported is None:
        imported = types.ModuleType(module)
        sys.modules.setdefault(module, imported)

    imported._ext = _MissingExtension(module)

    try:
        pkg = importlib.import_module(package)
    except ImportError:
        pkg = types.ModuleType(package)
        pkg.__path__ = []
        sys.modules.setdefault(package, pkg)
    setattr(pkg, leaf, imported)


# 学习者扩展（src/extensions）与参考扩展（src/extensions_ref）。
_register("extensions", "extensions.tiny_llm_ext", "tiny_llm_ext")
_register("extensions_ref", "extensions_ref.tiny_llm_ext_ref", "tiny_llm_ext_ref")
