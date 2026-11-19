# RMSNorm 部分融合：首轮负结果

来源：artifacts/rmsnorm-benchmark-summary.json。BF16、A800、单请求连续 KV Cache、固定 32 个预测位置，31 步 decode；预热两轮，交替测量五轮。所有实际测评序列的逐步 logits 与原始路径完全一致。Profiler 在全部计时结束后采集。

| 输入长度 | 原始 decode 中位数 ms | 部分融合 decode 中位数 ms | 原始/融合 |
| --- | ---: | ---: | ---: |
| 128 | 591.949 | 872.748 | 0.678× |
| 512 | 593.181 | 877.743 | 0.676× |
| 1024 | 590.947 | 878.338 | 0.673× |

四步 trace 的 kernel 数从 7252 降至 6348（每步 1813→1587，减少约 12.47%），但未开启 profiler 的 decode 时间增加约 47%–49%。所有长度下两组五轮耗时范围不重叠，不能将本轮回退解释为微小测量波动。

实现是保留 PyTorch FP32 均方归约与 rsqrt，只融合乘逆 RMS、BF16 舍入与权重相乘；不是完整单 kernel RMSNorm。完整融合此前在归约阶段引入微小 FP32 差异，在 BF16 舍入边界造成分叉，未通过严格正确性门槛。

结论：不启用为默认优化，原始 torch 路径保留默认。减少 kernel 数不必然减少端到端时延。Python 包装、参数校验、device context 和 Triton launch 开销是待验证方向；现有摘要不能将回退精确归因于其中一个。不能将 profile 下的 kernel 时间与未 profile 的 wall time 相减得到 CPU 时间。后续先隔离 host 包装和 launch 开销，再决定精简热路径或改做 CUDA Graph。

## 调用开销消融结果

来源：artifacts/rmsnorm-overhead/summary.json 及 cpu-*.txt。相同 kernel，预热后每样本 100 次调用、五轮轮换顺序。形状 [1,1024] 的 wall-time 中位数：原始 torch 55.52 µs，生产部分融合 99.21 µs，受控完整包装 100.03 µs，去参数检查 95.60 µs，去 device guard 90.59 µs，同时去除 86.63 µs。其他三种形状表现相近。

去除两类包装仅改善受控版本约 13.4%，仍比原始 torch 慢约 56%。cProfile 中 100 次调用的 Triton jit.run 累计约 7 ms，包含目标查询、参数特化与 launcher 等调用；这是含 profiling 开销的调用栈证据，不等同于 JIT 编译耗时，也不能与未 profile 时延直接相减。

决策：不将无检查、无 device guard 版本投入模型默认路径。下一项实验优先验证 CUDA Graph replay 是否能降低重复 host 派发，先做固定形状原型，再处理动态 KV 长度与缓存状态。原型收益不能直接声称为完整动态服务收益。
