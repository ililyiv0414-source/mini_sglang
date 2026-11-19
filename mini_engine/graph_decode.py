"""Single-request fixed-capacity graph decoder with GPU-resident position.

Attention processes the whole capacity with a validity mask. This is not
paged attention; padding can alter floating-point reduction vs eager slices.
"""
import torch
from .qwen3 import KVCache


class GraphDecoder:
    @torch.inference_mode()
    def __init__(self, model, capacity):
        self.model = model
        self.cache = KVCache(model, 1, capacity)
        self.cache.k.zero_()
        self.cache.v.zero_()
        self.position = torch.zeros(1, dtype=torch.long, device=model.device)
        self.token = torch.zeros((1, 1), dtype=torch.long, device=model.device)
        self.length = 0
        self.ready = False

        def body():
            logits = model.forward(self.token, self.cache, graph_position=self.position)
            selected = logits[:, -1].argmax(-1)
            self.token.copy_(selected[:, None])
            self.position.add_(1)
            return logits, selected

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.position.zero_()
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.position.zero_()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits, self.selected = body()
        torch.cuda.synchronize()
        self.cache.k.zero_()
        self.cache.v.zero_()
        self.cache.reset()
        self.position.zero_()

    @torch.inference_mode()
    def prefill(self, ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= self.cache.capacity:
            raise ValueError("Invalid single-request prompt or capacity")
        self.ready = False
        # Full zeroing is essential: masked uninitialized V could contain NaN.
        self.cache.k.zero_()
        self.cache.v.zero_()
        self.cache.reset()
        output = self.model.forward(ids, self.cache)
        self.length = ids.shape[1]
        self.position.fill_(self.length)
        self.token.copy_(output[:, -1].argmax(-1)[:, None])
        self.ready = True
        return output

    @torch.inference_mode()
    def step(self):
        if not self.ready:
            raise RuntimeError("Prefill required")
        if self.length >= self.cache.capacity:
            raise ValueError("KV cache capacity exceeded")
        self.graph.replay()
        self.length += 1
        self.cache.length = self.length
        # Returned tensors are graph-owned and overwritten by the next replay.
        return self.logits

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens, eos_ids):
        if max_new_tokens < 1 or ids.shape[1] + max_new_tokens - 1 > self.cache.capacity:
            raise ValueError("Invalid generation length or insufficient capacity")
        self.prefill(ids)
        generated = []
        for step in range(max_new_tokens):
            if step:
                self.step()
            token = self.token.item()  # Explicit host synchronization for EOS in v1.
            generated.append(token)
            if token in eos_ids:
                return generated, "eos"
        return generated, "length"
