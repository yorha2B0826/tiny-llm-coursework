"""测试环境兼容层（本仓库不含课程 C++/Metal 扩展）。

`tiny_llm/__init__.py` 会 import `week2_kernels`，后者在模块级导入
`extensions.tiny_llm_ext`（需要在完整课程仓库中构建的扩展）。
本仓库只包含 Python 实现，Week 1 的测试不依赖该扩展；
扩展缺失时这里注册一个占位模块，让 `import tiny_llm` 正常工作，
一旦真正调用扩展算子会得到明确的报错提示。
"""

import sys
import types

try:
    import extensions.tiny_llm_ext  # noqa: F401
except ImportError:

    class _MissingExtension:
        def __getattr__(self, name):
            raise RuntimeError(
                "课程扩展 tiny_llm_ext 未构建：Week 2+ 的内核测试需要先在完整课程"
                f"仓库中构建扩展（pdm run build-ext）。（尝试访问：{name}）"
            )

    _extensions = types.ModuleType("extensions")
    _extensions.__path__ = []
    _tiny_llm_ext = types.ModuleType("extensions.tiny_llm_ext")
    _tiny_llm_ext._ext = _MissingExtension()
    _extensions.tiny_llm_ext = _tiny_llm_ext

    sys.modules.setdefault("extensions", _extensions)
    sys.modules.setdefault("extensions.tiny_llm_ext", _tiny_llm_ext)
