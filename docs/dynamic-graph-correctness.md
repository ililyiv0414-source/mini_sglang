# 动态位置 Graph 正确性复核

报告：artifacts/dynamic-graph-check.json（BF16）和 artifacts/dynamic-graph-check-fp32.json（FP32）。容量 128，batch=1，两个固定 prompt 按 A→B→A 复用同一个 graph。

- BF16：Graph 与同形状固定容量 eager 完全一致；位置、反馈、EOS、长度上限、容量与重置检查通过；三个生成结果匹配原参考。但相对变长切片 eager，45 步 logits 不完全一致，最大绝对误差 0.9375。
- FP32（禁用 TF32）：Graph 与固定容量及变长切片 eager 的逐步 logits 均完全一致，最大绝对误差 0；同精度生成结果一致，边界检查通过。

证据支持 BF16 计算形状引起数值差异，而非已测用例中的 graph 状态推进错误。该结论仅覆盖本次容量与输入，不能推广为所有容量、输入、精度均验证。

后续性能实验应区分两个问题：相同固定容量 BF16 eager→graph replay 的派发收益，可以在逐步严格对齐后测量；原变长 BF16 eager→固定容量 graph 同时改变 attention 形状，不是纯 graph 消融，其完全一致门槛仍未通过。不要混合 FP32 和 BF16 的时延计算加速比。

本版本 generate 每步通过 token.item() 同步 CPU 以检查 EOS。下一轮完整生成测评应在两条路径中使用相同 EOS 和同步策略，计入 token 回馈和停止检查，并分别报告捕获、prefill/重置和 decode 成本。
