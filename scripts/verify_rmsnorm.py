"""Exact correctness gate for fused RMSNorm; no automatic tolerance relaxation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_engine.qwen3 import Qwen3, KVCache
from mini_engine.rmsnorm import rmsnorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    output = Path("artifacts/rmsnorm-tail-check.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, actual, expected):
        error = (actual.float() - expected.float()).abs()
        row = {"case": name, "exact": torch.equal(actual, expected),
               "finite": bool(torch.isfinite(actual).all()), "max_abs": error.max().item(),
               "mean_abs": error.mean().item(),
               "different_elements": int((actual != expected).sum().item())}
        row["passed"] = row["exact"] and row["finite"]
        rows.append(row)
        print(json.dumps(row), flush=True)

    with torch.inference_mode():
        for n in (128, 1024):
            for count in (1, 17, 512):
                for scale in (0.0, 1e-4, 1.0, 100.0):
                    x = (torch.randn(count, n, device="cuda") * scale).bfloat16()
                    weight = torch.randn(n, device="cuda").bfloat16()
                    xf = x.float()
                    expected = weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
                    check(f"norm_n{n}_rows{count}_scale{scale}", rmsnorm(x, weight, 1e-6), expected)
        model = Qwen3(args.model)
        reference = json.loads(Path("artifacts/reference.json").read_text(encoding="utf-8"))
        cases = [(f"prompt_{i}", torch.tensor([c["input_ids"]], device="cuda"))
                 for i, c in enumerate(reference["results"])]
        for batch, length in ((1, 1), (1, 127), (1, 512), (1, 1024), (2, 17)):
            cases.append((f"random_b{batch}_t{length}", torch.randint(model.config["vocab_size"],
                         (batch, length), device="cuda")))
        for name, ids in cases:
            model.set_norm_backend("torch")
            expected = model.forward(ids)
            model.set_norm_backend("triton")
            check(name, model.forward(ids), expected)
        # Teacher force reference histories, comparing all prefill/decode logits.
        for index, case in enumerate(reference["results"]):
            base_cache = KVCache(model, 1, 128)
            fused_cache = KVCache(model, 1, 128)
            ids = torch.tensor([case["input_ids"]], device="cuda")
            for step, token in enumerate(case["output_ids"]):
                model.set_norm_backend("torch")
                expected = model.forward(ids, base_cache)
                model.set_norm_backend("triton")
                check(f"cached_prompt{index}_step{step}", model.forward(ids, fused_cache), expected)
                assert base_cache.length == fused_cache.length
                ids = torch.tensor([[token]], device="cuda")
    passed = all(r["passed"] for r in rows)
    output.write_text(json.dumps({"passed": passed, "implementation": "torch reduction/rsqrt + Triton pointwise tail", "acceptance": "exact BF16 equality",
                                 "cases": rows}, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(f"Numerical differences detected: {output}. Benchmark must not proceed yet.")
    print(f"Fused RMSNorm verification OK: {output}")


if __name__ == "__main__":
    main()
