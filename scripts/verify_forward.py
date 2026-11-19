"""Compare custom full-sequence forward against HF eager attention, no cache."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM
from mini_engine.qwen3 import Qwen3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", default="artifacts/reference.json")
    parser.add_argument("--output", default="artifacts/forward-check.json")
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    custom = Qwen3(args.model)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).to("cuda").eval()
    cases = [(f"prompt_{i}", torch.tensor([r["input_ids"]], device="cuda"))
             for i, r in enumerate(ref["results"])]
    rng = torch.Generator().manual_seed(2026)
    for batch, length in [(1, 1), (1, 7), (1, 32), (1, 127), (2, 17)]:
        ids = torch.randint(custom.config["vocab_size"], (batch, length), generator=rng).cuda()
        cases.append((f"random_b{batch}_t{length}", ids))
    records = []
    with torch.inference_mode():
        for name, ids in cases:
            expected = hf(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            actual = custom.forward(ids)
            difference = (actual.float() - expected.float()).abs()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            exact = torch.equal(actual, expected)
            agreement = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
            row = {"case": name, "shape": list(ids.shape), "exact_logits": exact,
                   "finite": finite, "max_abs_error": difference.max().item(),
                   "mean_abs_error": difference.mean().item(), "argmax_agreement": agreement,
                   "passed": finite and exact}
            records.append(row)
            print(json.dumps(row), flush=True)
    # Invalid empty inputs must be rejected explicitly.
    try:
        custom.forward(torch.empty((1, 0), dtype=torch.long, device="cuda"))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty input was accepted")
    passed = all(r["passed"] for r in records)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": passed, "acceptance": "exact BF16 logits against HF eager",
                               "torch": torch.__version__, "cases": records}, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(f"Numerical differences detected; inspect {path}; do not relax tolerances without analysis")
    print(f"Custom Qwen3 forward verification OK: {path}")


if __name__ == "__main__":
    main()
