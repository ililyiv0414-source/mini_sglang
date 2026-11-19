"""Single-request BF16 grouped-query decode over physical KV pages."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scores(Q, K, Table, Scores, length, stride, GROUP: tl.constexpr,
            KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr,
            Lengths, HEADS: tl.constexpr, TABLE_WIDTH: tl.constexpr):
    row = tl.program_id(0)
    head = row % HEADS
    if TABLE_WIDTH:
        batch = row // HEADS
        Table += batch * TABLE_WIDTH
        length = tl.load(Lengths + batch)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, D)
    page = tl.load(Table + pos // PAGE, pos < length, other=0)
    physical = page * PAGE + pos % PAGE
    offsets = (physical * KV + head // GROUP) * D
    keys = tl.load(K + offsets[:, None] + dim[None, :], (pos < length)[:, None], other=0)
    q = tl.load(Q + row * D + dim)
    dot = tl.sum(keys.to(tl.float32) * q[None, :].to(tl.float32), 1)
    rounded = dot.to(tl.bfloat16).to(tl.float32)
    tl.store(Scores + row * stride + pos, rounded * (D ** -0.5), pos < length)


@triton.jit
def _values(Scores, V, Table, Out, length, stride, GROUP: tl.constexpr,
            KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, N: tl.constexpr,
            Lengths, HEADS: tl.constexpr, TABLE_WIDTH: tl.constexpr):
    row = tl.program_id(0)
    head = row % HEADS
    if TABLE_WIDTH:
        batch = row // HEADS
        Table += batch * TABLE_WIDTH
        length = tl.load(Lengths + batch)
    pos = tl.arange(0, N)
    dims = tl.program_id(1) * 8 + tl.arange(0, 8)
    scores = tl.load(Scores + row * stride + pos, pos < length, other=-float("inf")).to(tl.float32)
    weights = tl.softmax(scores).to(tl.bfloat16).to(tl.float32)
    page = tl.load(Table + pos // PAGE, pos < length, other=0)
    offsets = ((page * PAGE + pos % PAGE) * KV + head // GROUP) * D
    values = tl.load(V + offsets[:, None] + dims[None, :], (pos < length)[:, None], other=0)
    tl.store(Out + row * D + dims, tl.sum(weights[:, None] * values.to(tl.float32), 0))


def attend(q, k, v, table, length, page_size, lengths=None):
    heads, dim = q.shape[1], q.shape[-1]
    kv_heads = k.shape[1]
    rows = heads * q.shape[0]
    width = table.shape[1] if lengths is not None else 0
    metadata = (table if lengths is None else lengths, heads, width)
    scores = torch.empty((rows, length), dtype=q.dtype, device=q.device)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    args = (heads // kv_heads, kv_heads, dim, page_size)
    _scores[(rows, triton.cdiv(length, 16))](q.contiguous(), k, table, scores,
                                               length, length, *args, 16, *metadata)
    _values[(rows, dim // 8)](scores, v, table, out, length, length, *args,
                                triton.next_power_of_2(length), *metadata)
    return out
