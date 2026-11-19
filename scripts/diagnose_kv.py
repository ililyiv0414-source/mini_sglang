"""Teacher-forced four-path comparison on a failed benchmark prompt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM
from mini_engine.qwen3 import Qwen3, KVCache


def metrics(a, b):
    delta = (a.float() - b.float()).abs()
    return {"exact": torch.equal(a, b), "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(), "same_argmax": a.argmax().item() == b.argmax().item()}


def top(logits):
    values, indices = logits.float().topk(5)
    return {"ids": indices.tolist(), "logits": values.tolist(),
            "top2_margin": (values[0] - values[1]).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", default="artifacts/benchmark-kv.json")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    case = next(c for c in report["cases"] if c["status"] == "correctness_failed")
    steps = report["arguments"]["output_length"]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    custom = Qwen3(args.model, dtype=dtype)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
        attn_implementation="eager", local_files_only=True, trust_remote_code=False).cuda().eval()
    history = torch.tensor([case["input_ids"]], device="cuda")
    cache = KVCache(custom, 1, history.shape[1] + steps - 1)
    hf_cache = None
    rows, selected = [], []
    output = Path(f"artifacts/kv-diagnostic-{args.dtype}.json")
    result = {"dtype": args.dtype, "input_length": history.shape[1],
              "history_policy": "All paths receive identical history selected by custom full recomputation",
              "torch": torch.__version__, "rows": rows}
    with torch.inference_mode():
        for step in range(steps):
            ids = history if step == 0 else history[:, -1:]
            cf = custom.forward(history)[0, -1]
            cc = custom.forward(ids, cache)[0, -1]
            hf_full = hf(input_ids=history, use_cache=False).logits[0, -1]
            cached = hf(input_ids=ids, past_key_values=hf_cache, use_cache=True,
                        attention_mask=torch.ones_like(history))
            hf_cache = cached.past_key_values
            hc = cached.logits[0, -1]
            assert cache.length == history.shape[1] == hf_cache.get_seq_length()
            row = {"step": step, "history_length": history.shape[1],
                   "custom_full_vs_hf_full": metrics(cf, hf_full),
                   "custom_cache_vs_hf_cache": metrics(cc, hc),
                   "custom_full_vs_cache": metrics(cf, cc),
                   "hf_full_vs_cache": metrics(hf_full, hc),
                   "top5": {name: top(value) for name, value in
                            [("custom_full", cf), ("custom_cache", cc), ("hf_full", hf_full), ("hf_cache", hc)]}}
            rows.append(row)
            token = cf.argmax().item()
            selected.append(token)
            print(f"step={step} full_vs_cache_argmax={row['custom_full_vs_cache']['same_argmax']} "
                  f"custom_cache_vs_hf_exact={row['custom_cache_vs_hf_cache']['exact']} "
                  f"custom_full_vs_hf_exact={row['custom_full_vs_hf_full']['exact']}", flush=True)
            if step + 1 < steps:
                history = torch.cat([history, torch.tensor([[token]], device="cuda")], dim=1)
            result["full_recompute_output_ids"] = selected
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["first_argmax_divergence"] = next((r["step"] for r in rows
        if not r["custom_full_vs_cache"]["same_argmax"]), None)
    result["custom_matches_hf_both_paths_exactly"] = all(
        r["custom_full_vs_hf_full"]["exact"] and r["custom_cache_vs_hf_cache"]["exact"] for r in rows)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Diagnostic saved: {output}")


if __name__ == "__main__":
    main()
