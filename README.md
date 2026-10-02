# tiny-llm 课程实践代码

[skyzh/tiny-llm](https://github.com/skyzh/tiny-llm)（面向系统工程师的 LLM 推理系统课程）的个人实践实现。

- **课程与教程正体**：<https://github.com/skyzh/tiny-llm>。本仓库不是 fork，仅保留课程 starter 包 `src/tiny_llm/` 的实现进度，方便独立查看和复盘。
- **骨架版本**：对应上游提交 `da3e841`。
- **提交历史**：按课程 task 逐个推进（Day 2 的 traditional / non-traditional 合并为一个 commit）。
- **更新时间**：2026-10-02。

## 完成度（Week 1）

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
| Day 5 | Task 3 | Qwen3ModelWeek1（端到端跑通 0.6B） | 🚧 进行中 |

- Week 2–4：未开始。
- Day 5 Task 3 的实现还在本地进行中，本仓库暂未包含半成品代码。
- 验证口径：day1 44 例、day2 32 例、day3 84 例、day4 44 例、day5 task1/2 26 例，共 230 例全绿（task3 需下载 Qwen3-0.6B 模型，未运行）。

## 怎么跑测试

本仓库只含实现代码；运行课程测试需要完整的上游仓库：

```bash
git clone https://github.com/skyzh/tiny-llm
cd tiny-llm
pdm install -v
# 用本仓库的 src/tiny_llm/ 覆盖上游仓库的同名目录，然后：
pdm run test --week 1 --day 5
```

## 备注

- 提交历史为按 task 重建（本地开发时未逐步提交）：每个 commit 只包含该 task 的实现范围。
