"""Find the first differing RMSNorm with identical inputs; inspect FP32 stages."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
import triton.language as tl
from mini_engine.qwen3 import Qwen3
from mini_engine.rmsnorm import rmsnorm


@triton.jit
def stages(X, VAR, INV, NORMALIZED, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, other=0).to(tl.float32)
    variance = tl.sum(x * x, 0) / N
    inv = tl.rsqrt(variance + EPS)
    tl.store(VAR + row, variance)
    tl.store(INV + row, inv)
    tl.store(NORMALIZED + row * N + cols, x * inv, cols < N)


def difference(a, b):
    d = (a.float() - b.float()).abs()
    return {"exact": torch.equal(a, b), "different_elements": int((a != b).sum().item()),
            "max_abs": d.max().item(), "mean_abs": d.mean().item()}


class FoundDifference(Exception):
    pass


def main():
    model = Qwen3(sys.argv[1])
    original = model.norm
    report = {"torch": torch.__version__, "triton": triton.__version__, "checked": 0}

    def inspect(x, name):
        expected = original(x, name)
        weight = model.weights[name + ".weight"]
        actual = rmsnorm(x, weight, model.config["rms_norm_eps"])
        report["checked"] += 1
        if torch.equal(expected, actual):
            return expected
        flat = x.reshape(-1, x.shape[-1])
        count, n = flat.shape
        variance = torch.empty(count, device="cuda", dtype=torch.float32)
        inv = torch.empty_like(variance)
        normalized = torch.empty_like(flat, dtype=torch.float32)
        stages[(count,)](flat, variance, inv, normalized, n, model.config["rms_norm_eps"],
                        triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
        xf = flat.float()
        torch_var = xf.square().mean(-1)
        torch_inv = torch.rsqrt(torch_var + model.config["rms_norm_eps"])
        torch_normalized = xf * torch_inv[:, None]
        bad = (actual.reshape_as(flat) != expected.reshape_as(flat)).nonzero()[:8]
        samples = []
        for r, col in bad.tolist():
            samples.append({"row": r, "column": col, "x": flat[r, col].item(),
                "weight": weight[col].item(), "torch_var": torch_var[r].item(),
                "triton_var": variance[r].item(), "fp64_var": xf[r].double().square().mean().item(),
                "torch_inv": torch_inv[r].item(), "triton_inv": inv[r].item(),
                "torch_normalized_fp32": torch_normalized[r, col].item(),
                "triton_normalized_fp32": normalized[r, col].item(),
                "torch_normalized_bf16": torch_normalized[r, col].bfloat16().item(),
                "triton_normalized_bf16": normalized[r, col].bfloat16().item(),
                "expected": expected.reshape_as(flat)[r, col].item(),
                "actual": actual.reshape_as(flat)[r, col].item()})
        report.update(name=name, shape=list(x.shape), output=difference(actual, expected),
            variance=difference(variance, torch_var), inverse_rms=difference(inv, torch_inv),
            normalized_fp32=difference(normalized, torch_normalized), samples=samples)
        reconstructed = (normalized.bfloat16() * weight).reshape_as(expected)
        report["debug_reconstruction_vs_kernel"] = difference(reconstructed, actual)
        torch.save({"x": x.cpu(), "weight": weight.cpu(), "eps": model.config["rms_norm_eps"]},
                   "artifacts/rmsnorm-first-input.pt")
        raise FoundDifference()

    model.norm = inspect
    ref = json.loads(Path("artifacts/reference.json").read_text(encoding="utf-8"))
    with torch.inference_mode():
        try:
            for index, case in enumerate(ref["results"]):
                report["prompt_index"] = index
                model.forward(torch.tensor([case["input_ids"]], device="cuda"))
        except FoundDifference:
            pass
    Path("artifacts/rmsnorm-diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
