"""Dense Qwen3 forward using PyTorch primitives, without Transformers execution.

Scope: default RoPE, full causal attention, tied embeddings, unpadded batches.
Equations and checkpoint names follow Qwen3 and the Transformers reference.
Optional contiguous KV cache and Triton RMSNorm; no scheduler yet.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class KVCache:
    """One equal-length batch, owned by one model; reset before reuse.

    Length is committed only after a successful forward. Uncommitted slots
    are overwritten on retry and never included in attention.
    """
    def __init__(self, model, batch_size, capacity):
        if batch_size < 1 or not 1 <= capacity <= model.config["max_position_embeddings"]:
            raise ValueError("Invalid batch size or cache capacity")
        self.owner, self.batch_size, self.capacity = model, batch_size, capacity
        self.length = 0
        shape = (model.config["num_hidden_layers"], batch_size,
                 model.kv_heads, capacity, model.dim)
        self.k = torch.empty(shape, device=model.device, dtype=model.dtype)
        self.v = torch.empty_like(self.k)

    def reset(self):
        # Old bytes need not be cleared: only the committed prefix is readable.
        self.length = 0

    @property
    def allocated_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


class Qwen3:
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16, norm_backend="torch"):
        self.set_norm_backend(norm_backend)
        directory = Path(model_dir)
        c = json.loads((directory / "config.json").read_text())
        if (c["model_type"] != "qwen3" or c.get("rope_scaling") is not None
                or c.get("use_sliding_window", False) or c.get("attention_bias", False)
                or c["hidden_act"] != "silu" or not c["tie_word_embeddings"]):
            raise ValueError("Only dense Qwen3 with default RoPE, full attention, no bias and tied embeddings is supported")
        self.config, self.device, self.dtype = c, torch.device(device), dtype
        self.heads, self.kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
        self.dim = c["head_dim"]
        if self.heads % self.kv_heads or self.dim % 2:
            raise ValueError("Invalid GQA or rotary dimensions")
        checkpoint = load_file(str(directory / "model.safetensors"))
        # Some exporters store both names even when embeddings are tied.
        # Validate before dtype conversion; never silently discard unequal weights.
        if "lm_head.weight" in checkpoint:
            embedding = checkpoint["model.embed_tokens.weight"]
            output = checkpoint["lm_head.weight"]
            if output.dtype != embedding.dtype or not torch.equal(output, embedding):
                raise ValueError("Tied lm_head.weight differs from model.embed_tokens.weight")
            del checkpoint["lm_head.weight"]
        self.weights = {name: value.to(device=self.device, dtype=dtype)
                        for name, value in checkpoint.items()}
        expected = {"model.embed_tokens.weight": (c["vocab_size"], c["hidden_size"]),
                    "model.norm.weight": (c["hidden_size"],)}
        h, m = c["hidden_size"], c["intermediate_size"]
        for i in range(c["num_hidden_layers"]):
            shapes = {"input_layernorm": (h,), "post_attention_layernorm": (h,),
                      "self_attn.q_proj": (self.heads * self.dim, h),
                      "self_attn.k_proj": (self.kv_heads * self.dim, h),
                      "self_attn.v_proj": (self.kv_heads * self.dim, h),
                      "self_attn.o_proj": (h, self.heads * self.dim),
                      "self_attn.q_norm": (self.dim,), "self_attn.k_norm": (self.dim,),
                      "mlp.gate_proj": (m, h), "mlp.up_proj": (m, h), "mlp.down_proj": (h, m)}
            expected.update({f"model.layers.{i}.{k}.weight": v for k, v in shapes.items()})
        if set(expected) != set(self.weights):
            raise ValueError(f"Checkpoint keys differ: {set(expected) ^ set(self.weights)}")
        for name, shape in expected.items():
            if tuple(self.weights[name].shape) != shape:
                raise ValueError(f"Wrong shape: {name}")
        # Compute frequencies in FP32 on CPU, as the reference initialization does.
        self.inv_freq = (1.0 / (c["rope_theta"] **
                               (torch.arange(0, self.dim, 2).float() / self.dim))).to(self.device)

    def set_norm_backend(self, backend):
        if backend not in ("torch", "triton"):
            raise ValueError("norm_backend must be torch or triton")
        self.norm_backend = backend

    def norm(self, x, name):
        if self.norm_backend == "triton":
            from .rmsnorm import rmsnorm
            return rmsnorm(x, self.weights[name + ".weight"], self.config["rms_norm_eps"])
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.config["rms_norm_eps"])
        return self.weights[name + ".weight"] * normalized.to(x.dtype)

    def linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"])

    @staticmethod
    def rotate(x, cos, sin):
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-right, left), dim=-1) * sin

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, graph_position=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0 or input_ids.shape[0] == 0:
            raise ValueError("Expected nonempty [batch, sequence] token IDs without padding")
        b, t = input_ids.shape
        from .paged_cache import PagedDecodeBatch
        batch_decode = isinstance(cache, PagedDecodeBatch)
        if batch_decode:
            if t != 1 or graph_position is not None or cache.batch_size != b or cache.owner is not self:
                raise ValueError("Invalid paged decode batch")
            graph_position = cache.prepare()
        static_decode = graph_position is not None
        if static_decode and (cache is None or t != 1):
            raise ValueError("Graph position requires a cache and a single input token")
        start = 0
        if cache is not None:
            if cache.owner is not self or cache.batch_size != b:
                raise ValueError("Cache model or batch mismatch")
            start = 0 if static_decode else cache.length
            if start + t > cache.capacity:
                raise ValueError("KV cache capacity exceeded")
        end = cache.capacity if static_decode else start + t
        if end > self.config["max_position_embeddings"]:
            raise ValueError("Context limit exceeded")
        paged = hasattr(cache, "indices")
        if paged and static_decode:
            raise ValueError("Paged request requires eager execution")
        direct = (paged and cache.store.direct and t == 1 and self.dtype == torch.bfloat16
                  and self.dim == 128 and end <= 2048)
        direct = direct or batch_decode
        table = cache.direct_table(end) if direct and not batch_decode else None
        indices = cache.indices(end) if paged and not direct else None
        x = F.embedding(input_ids, self.weights["model.embed_tokens.weight"])
        positions = (graph_position.float().view(-1, 1, 1) if static_decode else
                     torch.arange(start, end, device=self.device).float().view(1, 1, t))
        frequencies = (self.inv_freq[None, :, None].float() @ positions).transpose(1, 2)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)
        if static_decode and not batch_decode:
            mask = torch.where(torch.arange(end, device=self.device)[None, None, None, :] <= graph_position.view(-1, 1, 1, 1),
                               0.0, torch.finfo(x.dtype).min).to(x.dtype)
        elif not direct:
            mask = torch.full((t, end), torch.finfo(x.dtype).min, device=self.device, dtype=x.dtype).triu(start + 1)
        for i in range(self.config["num_hidden_layers"]):
            p = f"model.layers.{i}"
            a = p + ".self_attn"
            z = self.norm(x, p + ".input_layernorm")
            q = self.norm(self.linear(z, a + ".q_proj").view(b, t, self.heads, self.dim), a + ".q_norm").transpose(1, 2)
            k = self.norm(self.linear(z, a + ".k_proj").view(b, t, self.kv_heads, self.dim), a + ".k_norm").transpose(1, 2)
            v = self.linear(z, a + ".v_proj").view(b, t, self.kv_heads, self.dim).transpose(1, 2)
            q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
            if direct:
                attended = cache.attend(i, table, end, q, k, v).reshape(b, t, -1)
            else:
                if paged:
                    k, v = cache.update(i, indices, start, k, v)
                elif cache is not None:
                    if static_decode and graph_position.numel() > 1:
                        indices = graph_position.view(b, 1, 1, 1).expand_as(k)
                        cache.k[i].scatter_(2, indices, k)
                        cache.v[i].scatter_(2, indices, v)
                    elif static_decode:
                        cache.k[i].index_copy_(2, graph_position, k)
                        cache.v[i].index_copy_(2, graph_position, v)
                    else:
                        cache.k[i, :, :, start:end].copy_(k)
                        cache.v[i, :, :, start:end].copy_(v)
                    k, v = cache.k[i, :, :, :end], cache.v[i, :, :, :end]
                repeats = self.heads // self.kv_heads
                k = k[:, :, None].expand(b, self.kv_heads, repeats, end, self.dim).reshape(b, self.heads, end, self.dim)
                v = v[:, :, None].expand(b, self.kv_heads, repeats, end, self.dim).reshape(b, self.heads, end, self.dim)
                scores = (q @ k.transpose(-2, -1)) * (self.dim ** -0.5) + mask
                probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
                attended = (probabilities @ v).transpose(1, 2).contiguous().reshape(b, t, -1)
            x = x + self.linear(attended, a + ".o_proj")
            z = self.norm(x, p + ".post_attention_layernorm")
            gated = F.silu(self.linear(z, p + ".mlp.gate_proj")) * self.linear(z, p + ".mlp.up_proj")
            x = x + self.linear(gated, p + ".mlp.down_proj")
        logits = F.linear(self.norm(x, "model.norm"), self.weights["model.embed_tokens.weight"])
        if cache is not None and not static_decode:
            cache.length = end
        if batch_decode:
            cache.commit()
        return logits


