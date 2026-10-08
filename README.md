# tiny-llm 课程实践代码

[skyzh/tiny-llm](https://github.com/skyzh/tiny-llm)（面向系统工程师的 LLM 推理系统课程）的个人实践实现。

- **课程与教程正体**：<https://github.com/skyzh/tiny-llm>。本仓库不是 fork：包含课程 starter 包 `src/tiny_llm/` 的个人实现、课程测试（`tests/`，取自上游）与 Week 2 起需要的原生扩展源码（`src/extensions/`）。
- **进度**：Week 1（Day 1–7）已完成 ✅；Week 2 Day 1（KV cache）、Day 2（量化算子量化路径 + W4A16 Metal kernel）已完成 ✅；Week 2 其余天与 Week 3–4 未开始。
- **骨架版本**：对应上游提交 `da3e841`。
- **提交历史**：按课程 task 逐个推进（Day 2 的 traditional / non-traditional 合并为一个 commit）。
- **更新时间**：2026-10-08。

## 完成度

### Week 1 ✅

| 天 | Task | 内容 | 状态 |
| --- | --- | --- | --- |
| Day 1 | Task 1 | softmax、simple attention | ✅ |
| Day 1 | Task 2 | linear、multi-head attention | ✅ |
| Day 2 | Task 1–2 | RoPE（traditional / non-traditional） | ✅ |
| Day 3 | Task 1 | grouped attention（GQA 广播实现） | ✅ |
| Day 3 | Task 2 | causal mask（矩形因果） | ✅ |
| Day 3 | Task 3 | Qwen3 attention（GQA 接入） | ✅ |
| Day 4 | Task 1 | RMSNorm | ✅ |
| Day 4 | Task 2 | SiLU、Qwen3 MLP | ✅ |
| Day 5 | Task 1 | Qwen3TransformerBlock | ✅ |
| Day 5 | Task 2 | Embedding（lookup / as_linear） | ✅ |
| Day 5 | Task 3 | Qwen3ModelWeek1（端到端跑通 Qwen3-0.6B） | ✅ |
| Day 6 | Task 1 | 生成循环（simple_generate：prefill/decode、EOS、流式输出） | ✅ |
| Day 7 | Task 1 | 采样器（make_sampler：temperature / top-k / top-p） | ✅ |

- **Week 1 完成（Day 1–7）**：day1 44 例、day2 32 例、day3 84 例、day4 44 例、day5 27 例（另有 2 例 4B / 1.7B 模型用例未下载自动 skip）、day6 7 例、day7 10 例，共 **248 例全绿**。

### Week 2

| 天 | Task | 内容 | 状态 |
| --- | --- | --- | --- |
| Day 1 | Task 1 | KV cache（追加拼接路径） | ✅ |
| Day 1 | Task 2 | 带 cache 的 Week 2 模型（offset / cache 数据流） | ✅ |
| Day 1 | Task 3 | create_kv_cache 请求级缓存工厂 | ✅ |
| Day 1 | Task 4 | simple_generate_with_kv_cache serving loop | ✅ |
| Day 1 | — | capacity-cache checkpoint（定长缓冲、切片写入） | ✅ |
| Day 1 | — | profiler 归因（capacity-cache:decode:128） | ✅ |
| Day 2 | Task 1 | 权重解包（`dequantize_weights`）与量化 embedding | ✅ |
| Day 2 | Task 2 | W4A16 quantized_matmul（C++ primitive + `eval_cpu`/`eval_gpu`） | ✅ |
| Day 2 | Task 3 | Metal kernel：`quantized_matmul_vanilla_w4a16_g128` | ✅ |
| Day 2 | Task 4 | Metal kernel：`quantized_matvec_x4_fast_w4a16_g128` | ✅ |
| Day 3–7 | — | 尚未开始 | ⬜ |

- **Week 2 Day 1 验证口径**：`pdm run test --week 2 --day 1` 6 例全绿；`tests_refsol/test_dense_kv_capacity.py` 7 例全绿；`pdm run profile-week2-kernels --case capacity-cache:decode:128` 可运行。
- **Week 2 Day 2 验证口径**：在完整课程仓库中 `pdm run build-ext && pdm run test --week 2 --day 2` **14 例全绿**（对应上游 `tests_refsol/test_week_2_day_2.py`）。

## 前置条件

| 项 | 要求 | 说明 |
| --- | --- | --- |
| 平台 | macOS + Apple Silicon（M1 及以上） | MLX 只支持 Apple silicon |
| Python | 3.10–3.12（已用 `.python-version` 固定为 3.11） | 无需手动安装，uv 会自动下载/选用 |
| [uv](https://docs.astral.sh/uv/) | 最新版 | 安装：`brew install uv`，或 `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Xcode 与 Metal 工具链 | 完整 Xcode + Metal 编译器 + CMake ≥ 3.27 | **Week 2+ 构建课程原生扩展时需要**（见[课程 setup](https://github.com/skyzh/tiny-llm/blob/main/book/src/setup.md)与 Week 1 Day 7 的工具链检查）；本仓库的 Week 1 测试无需构建扩展，但建议提前装好 |
| 网络与磁盘 | 首次装依赖；Day 5 的模型用例需下载 Qwen3-0.6B（约 0.4 GB） | 见下文"模型依赖" |

## 如何运行测试

本仓库自带 Week 1（Day 1–7）的课程测试，可在仓库根目录直接运行。

### 1. 安装依赖

```bash
uv sync          # 按 pyproject.toml + uv.lock 创建 .venv 并安装全部依赖
```

### 2. 运行

```bash
uv run pytest                                        # 全部
uv run pytest tests/test_week_1_day_1.py             # 跑某一天
uv run pytest tests/test_week_1_day_1.py tests/test_week_1_day_4.py   # 跑多天
uv run pytest tests/test_week_1_day_4.py -k task_1   # 只跑某个 task
```

（也可以 `source .venv/bin/activate` 激活环境后直接用 `pytest`。）

### 3. 说明

- pytest 配置在 `pyproject.toml`（`--import-mode=importlib`、`pythonpath = ["src", "."]`），在仓库根目录执行即可。
- **模型依赖**：Day 5 的 task2（部分用例）与 task3 需要本地已缓存 Qwen3-0.6B；未缓存时对应用例会失败并提示下载命令：

  ```bash
  uv run hf download Qwen/Qwen3-0.6B-MLX-4bit
  ```

  4B / 1.7B 模型用例在其未下载时自动跳过（skip）。
- **扩展依赖**：Week 2 起需要原生的 C++/Metal 扩展。本仓库包含扩展**源码**（`src/extensions/` 为学习者实现，`src/extensions_ref/` 为课程参考实现，均取自上游），但**不含构建产物**（`_ext*.so`、`*.dylib`、`*.metallib` 均被 `.gitignore` 排除，需要在 macOS + 完整 Xcode 工具链下现场编译）。`tests/conftest.py` 会在扩展未构建时给对应的 `_ext` 挂上惰性报错对象：Week 1 测试与 Week 2 Day 1 测试不受影响，真正调用扩展算子时会给出明确报错（提示先构建扩展）。
- **测试不自足的部分**：`tests/test_week_2_day_2.py` 取自上游并原样保留，它在**收集阶段**会 import 课程 harness（`benches/bench_course_progression.py`、`benches/bench_week2_operators.py`、`benches/profile_week2_kernels.py`、`benches/week2_gpudebug.py`、`src/tiny_llm_ref/`）。这些同样是课程-provided 文件，为可跑通而一并纳入本仓库，不是本仓库的实践产物；其中 `src/tiny_llm_ref/` 是 Week 2+ 的**参考实现**（含剧透，请按需回避）。
- 也可以回到完整课程仓库工作：用本仓库的 `src/tiny_llm/` 覆盖上游同名目录后，按[课程 setup](https://github.com/skyzh/tiny-llm/blob/main/book/src/setup.md) 用 `pdm run test --week 1 --day N` 跑同一批测试。

## 备注

- 提交历史为按 task 重建（本地开发时未逐步提交）：每个 commit 只包含该 task 的实现范围。
- 测试文件取自课程上游仓库 `tests_refsol/`（Week 1 七天、Week 2 Day 1、Day 2），逐字节原样保留；仅新增 `tests/conftest.py` 作为扩展未构建时的兼容层。
- 基准输出（`week2-day*-*.json`）属于本机测量结果，已加入 `.gitignore` 不入库。
- 依赖版本：`mlx-lm` 固定为 `0.31.3`（与课程环境一致）。更新的 0.32.x 中 `TokenizerWrapper` 会要求被测 tokenizer 提供 `apply_chat_template`，与课程 Day 6 测试使用的 `MinimalTokenizer` 不兼容。`mlx` 未固定：Day 2 验证在课程环境的 0.32.0 与解析出的 0.32.3 上均通过。
- Week 2 Day 2 的 C++/Metal 侧还包含两处与课程上游不同的修正：①`src/extensions/CMakeLists.txt`、`src/extensions_ref/CMakeLists.txt` 修正 MLX wheel 里写死的 Xcode 框架路径（否则链接期出现 `ld: warning: search path '...' not found`）；②扩展的 C++/Metal 实现按课程 task 自行完成。
