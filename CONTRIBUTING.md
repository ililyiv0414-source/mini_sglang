# 参与 mini_sglang

欢迎提交可以复现的正确性问题、新的请求负载或性能对照。第一次阅读可以从 README 的请求生命周期和在线实验台开始。

- **报告结果差异**：附上仓库提交、GPU、PyTorch/Triton 版本、命令、期望结果，以及第一次出现差异的位置。可以用合成 token 代替私有 prompt。
- **提交性能改动**：说明改动位于 prefill、decode、页分配还是调度。用相同模型、输入、KV 预算和采样设置测量改动前后结果，注明预热和重复次数。保留逐 token 正确性检查。
- **修改演示**：浏览器直接打开 `docs/demo/index.html`，切换五种负载。展示数据来自 `artifacts/workload-sweep.json` 的各组 summary。请同时核对对应的样本和单位。

GPU 校验入口为 `python scripts/verify_batched_paged.py`。硬件和模型准备方式见[实验文档](docs/benchmarks.md)。提交 PR 时注明实际运行过的检查以及尚待验证的环境。
