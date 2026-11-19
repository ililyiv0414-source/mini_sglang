# mini_sglang 单卡推理引擎

面向 Qwen3-0.6B 的 BF16 贪心推理实验。目标是在固定 KV 显存预算下提高请求接纳容量，并衡量批量执行对输出吞吐和排队延迟的影响。项目参考 mini-sglang 的设计，参考源码版本为 `9a91cfafe754aa85daee49998176275667eb58f2`。下列数字均来自本项目的实现间对照。

## 执行路径

[模型前向](mini_engine/qwen3.py) 加载权重并执行 prefill 和 decode。[页分配器](mini_engine/page_pool.py) 维护空闲页与请求页表，[分页缓存](mini_engine/paged_cache.py) 管理物理 K/V、写入位置和请求释放。[Triton attention](mini_engine/paged_attention.py) 在单 token decode 时按页表读取历史 K/V。FIFO 调度器按请求的生成上限预留额度，逐请求执行 prefill，然后把同轮活跃请求的位置和页表组成 decode batch；请求结束后归还额度和物理页。[调度器与对照实验](scripts/benchmark_paged_refill.py)

## 设计指标与结果

单张共享 A800-SXM4-80GB，KV 张量预算 64 MiB。端到端实验使用同时到达的 16 个请求，共用一条 24-token prompt，短、长请求各 8 个，实际分别生成 8 和 34 个 token。吞吐按所有输出 token 除以本轮墙钟时间计算，含 prefill、decode 与调度。TTFT 从整组到达计至请求首 token，包含排队。

| 对照 | 固定条件 | 结果 |
|---|---|---|
| 连续槽位 → 分页分配 | 相同 KV 预算，逐请求执行 | 峰值接纳 6 → 9，吞吐 51.83 → 49.30 tokens/s |
| 收集 K/V → 直接读页 | 单活跃请求，1024-token 输入，固定后续 token | decode 延迟 27.05 → 25.75 ms/token，降低 4.8% |
| 逐请求 → 批量 decode | 两组均直接读页、峰值接纳 9 个请求 | 49.47 → 193.20 tokens/s，达到 3.91 倍；P95 TTFT 4.740 → 1.055 s |

容量实验取三轮中位数。批量实验预热后交替执行两种模式，取四轮中位数。每轮 336 个输出 token 全部与串行参考一致；批量路径用 67 次 decode forward 完成 320 个后续 token，平均每次处理 4.78 个请求。16 个请求下，最近秩 P95 等于该轮最大 TTFT。[容量实验](artifacts/direct-refill.json) · [端到端原始记录](artifacts/batched-refill.json)

## 正确性与取舍

双请求验收从不同长度开始，每步交换 batch 行顺序，检查跨页增长、预测 token 和长度。20 个预测 token 全部匹配，logits 最大绝对差为 2.125。该验收覆盖选定输入上的贪心解码。[逐步记录](artifacts/batched-paged-check.json)

分页先解决容量问题。Profiler 显示旧路径每层分别收集 K 和 V，直接读页消除了每步 56 次额外的 index_select。短请求上的端到端吞吐只从 49.30 增至 49.81 tokens/s；更大的收益来自随后将多个请求合并执行。共享 GPU、单模型、固定到达模式限制了这些结果的适用范围。

## 在已配置的 course-gpu 环境复现

在项目根目录准备 Qwen3-0.6B 权重、项目源码、已配置的虚拟环境和 artifacts/reference.json。依次执行以下命令；static batch 会生成后续实验使用的串行参考。

```bash
cd /huhaifeng2/ljl/sgl
source scripts/activate_sgl.sh
export CUDA_VISIBLE_DEVICES=1
python scripts/verify_batched_paged.py
python scripts/benchmark_static_batch.py
python scripts/benchmark_paged_refill.py --direct
python scripts/benchmark_paged_refill.py --batched
python scripts/benchmark_workload_sweep.py
```

## 负载敏感性实测

同一张 A800、64 MiB KV 预算，所有请求同时到达，短、长请求分别输出 8、34 个 token。每组预热后交替测量四轮。吞吐和 P95 TTFT 取四轮中位数，平均 decode 宽度按实际 decode 步统计。40 个计时样本中每个请求的完整输出均与串行参考一致。[原始报告](artifacts/workload-sweep.json)

| 请求数（短∶长） | 峰值接纳 | 平均 decode 宽度 | 吞吐，逐请求 → 批量（tokens/s） | 吞吐倍数 | P95 TTFT，逐请求 → 批量（s） |
|---|---:|---:|---:|---:|---:|
| 4（1∶1） | 4 | 2.42 | 49.03 → 108.38 | 2.21× | 0.0885 → 0.0887 |
| 16（1∶1） | 9 | 4.78 | 49.42 → 194.45 | 3.93× | 4.746 → 1.050 |
| 32（1∶1） | 9 | 5.47 | 49.45 → 214.99 | 4.35× | 10.655 → 2.259 |
| 16（3∶1） | 12 | 5.27 | 49.48 → 194.01 | 3.92× | 2.096 → 0.519 |
| 16（1∶3） | 7 | 5.65 | 49.54 → 231.39 | 4.67× | 5.545 → 1.222 |

均衡负载从 16 增至 32 个请求时，峰值接纳仍为 9，持续补位让平均 decode 宽度从 4.78 增至 5.47。4 请求组可以一次全部接纳，吞吐提高了 2.21 倍，首 token 延迟几乎相同。长请求占多数时，生成上限预留压低了峰值接纳数，但更多后续 token 可以合批。16 请求均衡组复测为 3.93 倍，与独立实验的 3.91 倍接近。

## 简历与面试

> 为 Qwen3-0.6B 实现分页 KV cache、Triton 直接读页 attention 与变长请求批量 decode。在单卡 A800、64 MiB KV 预算下，将峰值接纳容量从 6 提升至 9；在 16 请求离线负载下，相较逐请求分页 decode，将输出吞吐从 49.47 提升至 193.20 tokens/s，四轮生成结果均与串行参考逐 token 一致。

面试时可沿三个问题展开：固定槽位为何浪费容量？为什么分页接纳更多请求后吞吐反而下降？为什么直接读页只带来有限提升，而批量 decode 在同一负载上达到 3.91 倍？分别用容量、算子 profiling 和实际 batch 宽度解释。

[项目设计](docs/project-plan.md) · [参考模型](docs/reference-model.md) · [KV 基线](docs/kv-baseline-results.md) · [CUDA Graph 实验](docs/cuda-graph-fixed-results.md)
