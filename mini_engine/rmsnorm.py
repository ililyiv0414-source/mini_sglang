"""Partial RMSNorm fusion: reference reduction/rsqrt, fused pointwise tail.

Full Triton reduction is retained as an experimental kernel, not selected:
different FP32 reduction order can cross BF16 rounding boundaries.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / N
    normalized = values * tl.rsqrt(variance + EPS)
    # Reference casts normalized activations BEFORE multiplying the weight.
    rounded = normalized.to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, rounded * weight, cols < N)


@triton.jit
def _rmsnorm_tail(X, W, INV, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    inv = tl.load(INV + row)
    rounded = (values * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, rounded * weight, cols < N)


def rmsnorm(x, weight, eps):
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Fused RMSNorm requires contiguous CUDA tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("Fused RMSNorm currently supports BF16 on one CUDA device only")
    if x.ndim < 1 or x.numel() == 0 or x.shape[-1] not in (128, 1024):
        raise ValueError("Supported nonempty last dimensions: 128, 1024")
    if tuple(weight.shape) != (x.shape[-1],) or eps <= 0:
        raise ValueError("Invalid weight shape or epsilon")
    output = torch.empty_like(x)
    n = x.shape[-1]
    # Keep the original shape as well as operation order for PyTorch reduction.
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    with torch.cuda.device(x.device):
        _rmsnorm_tail[(x.numel() // n,)](x, weight, inv, output, n,
            triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return output
