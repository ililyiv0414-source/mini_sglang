"""Controlled wrapper ablation; unchecked paths are diagnostic-only."""
import argparse
import cProfile
import hashlib
import io
import json
import pstats
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from mini_engine.rmsnorm import _rmsnorm_tail, rmsnorm


def reference(x, weight, eps):
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def variant(x, weight, eps, checks, guard):
    # Same validation and operations as production, with exactly two switches.
    if checks:
        if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("Requires contiguous CUDA tensors")
        if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or weight.device != x.device:
            raise ValueError("Requires BF16 on one device")
        if x.ndim < 1 or x.numel() == 0 or x.shape[-1] not in (128, 1024):
            raise ValueError("Invalid shape")
        if tuple(weight.shape) != (x.shape[-1],) or eps <= 0:
            raise ValueError("Invalid weight or epsilon")
    out = torch.empty_like(x)
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    n = x.shape[-1]
    with torch.cuda.device(x.device) if guard else nullcontext():
        _rmsnorm_tail[(x.numel() // n,)](x, weight, inv, out, n,
            triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return out


def measure(fn, x, w, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    begin = time.perf_counter()
    for _ in range(calls):
        out = fn(x, w, 1e-6)
    submitted = time.perf_counter()
    end.record()
    torch.cuda.synchronize()
    done = time.perf_counter()
    return {"enqueue_us_per_call": (submitted - begin) * 1e6 / calls,
            "wall_us_per_call": (done - begin) * 1e6 / calls,
            "cuda_span_us_per_call": start.elapsed_time(end) * 1000 / calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    torch.cuda.set_device(0)
    torch.manual_seed(2026)
    functions = {
        "torch": reference,
        "production": rmsnorm,
        "controlled_full": lambda x, w, e: variant(x, w, e, True, True),
        "no_checks": lambda x, w, e: variant(x, w, e, False, True),
        "no_guard": lambda x, w, e: variant(x, w, e, True, False),
        "neither": lambda x, w, e: variant(x, w, e, False, False),
    }
    folder = Path("artifacts/rmsnorm-overhead")
    folder.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0), "calls_per_sample": args.calls,
        "notes": "5 rotated-order samples; all JIT warmed before measurement. "
                 "Enqueue wall time includes Python, allocation, dispatch and possible backpressure, not pure CPU compute. "
                 "CUDA event span includes device idle gaps; not pure kernel runtime. "
                 "cProfile collected after all timing samples. No model speedup claim.",
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).resolve().parents[1] / "mini_engine/rmsnorm.py"]},
        "cases": []}
    with torch.inference_mode():
        for shape in [(1, 1024), (1, 1, 16, 128), (1, 1, 8, 128), (1, 512, 1024)]:
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
            expected = reference(x, w, 1e-6)
            for name, fn in functions.items():
                for _ in range(10):
                    actual = fn(x, w, 1e-6)
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"Exact correctness failed: {name}, {shape}")
            row = {"shape": list(shape), "exact_correctness": True, "samples": []}
            names = list(functions)
            for repeat in range(5):
                order = names[repeat:] + names[:repeat]
                for name in order:
                    sample = measure(functions[name], x, w, args.calls)
                    sample.update(variant=name, repeat=repeat)
                    row["samples"].append(sample)
            row["summary"] = {}
            for name in names:
                samples = [s for s in row["samples"] if s["variant"] == name]
                row["summary"][name] = {metric: {"median": statistics.median(s[metric] for s in samples),
                    "min": min(s[metric] for s in samples), "max": max(s[metric] for s in samples)}
                    for metric in ("wall_us_per_call", "enqueue_us_per_call", "cuda_span_us_per_call")}
                print(f"shape={shape} {name}: {row['summary'][name]['wall_us_per_call']['median']:.2f} us/call", flush=True)
            report["cases"].append(row)
            (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        x = torch.randn((1, 1024), device="cuda", dtype=torch.bfloat16)
        w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
        for name, fn in functions.items():
            torch.cuda.synchronize()
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(args.calls):
                fn(x, w, 1e-6)
            profiler.disable()
            torch.cuda.synchronize()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
            (folder / f"cpu-{name}.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(f"Overhead ablation complete: {folder}")


if __name__ == "__main__":
    main()
