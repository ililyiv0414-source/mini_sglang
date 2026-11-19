# 第一阶段模型与正确性基线

模型：Qwen/Qwen3-0.6B，来源为 ModelScope 的同名仓库。下载记录保留逐文件 revision 与 SHA256；下载成功后才能作为已验证模型使用。

从下载的 config.json 读取：28 层、16 个 query heads、8 个 KV heads、head dimension 128、hidden size 1024，使用 BF16。注意 head dimension 不能简单用 hidden size / query heads 推导，本模型配置显式指定为 128。

不考虑分页填充、分配器与元数据时，每 token 的 KV Cache 字节数：

`2 (K,V) × 28 × 8 × 128 × 2 (BF16 bytes) = 114688 bytes = 112 KiB`

单请求总序列长度 4096 时，KV Cache 理论容量为 448 MiB；长度预算需包含输入和已生成 token。这是配置推算，不是实测显存峰值。权重、激活、logits、临时工作区及框架开销另计。

第一阶段运行 scripts/reference_inference.py，固定两条 prompt、聊天模板关闭 thinking、BF16、eager attention、贪心解码、最多 64 个输出 token。保存输入/输出 token IDs、首个预测位置 logits、软件版本、GPU 型号与模型清单指纹。

该脚本是 Hugging Face 参考实现，不是自研引擎，也不是性能 benchmark。自研实现后续使用相同 token IDs、权重、精度进行数值及生成结果对比。不能将运行参考脚本描述为实现了推理引擎。
