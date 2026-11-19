"""Compare custom contiguous KV cache with HF's cached eager forward."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM
from mini_engine.qwen3 import Qwen3, KVCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    reference = json.loads(Path("artifacts/reference.json").read_text(encoding="utf-8"))
    custom = Qwen3(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True, trust_remote_code=False).cuda().eval()
    records = []

    def compare(name, actual, expected):
        error = (actual.float() - expected.float()).abs()
        row = {"case": name, "exact_logits": torch.equal(actual, expected),
               "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
               "argmax_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()}
        row["passed"] = row["exact_logits"] and bool(torch.isfinite(actual).all())
        records.append(row)
        print(json.dumps(row), flush=True)

    with torch.inference_mode():
        # Match the reference's prefill and one-token execution shapes at every step.
        for index, case in enumerate(reference["results"]):
            cache = KVCache(custom, 1, len(case["input_ids"]) + reference["max_new_tokens"])
            ids = torch.tensor([case["input_ids"]], device="cuda")
            state, generated = None, []
            for step in range(len(case["output_ids"])):
                previous = cache.length
                actual = custom.forward(ids, cache)
                expected = hf(input_ids=ids, past_key_values=state, use_cache=True,
                    attention_mask=torch.ones((1, cache.length), dtype=torch.long, device="cuda"))
                state = expected.past_key_values
                assert cache.length == previous + ids.shape[1] == state.get_seq_length()
                compare(f"prompt_{index}_step_{step}", actual, expected.logits)
                token = actual[0, -1].argmax().item()
                generated.append(token)
                ids = torch.tensor([[token]], device="cuda")
            records.append({"case": f"prompt_{index}_tokens", "passed": generated == case["output_ids"],
                            "output_ids": generated, "cache_length": cache.length,
                            "allocated_bytes": cache.allocated_bytes})

        # Batch two, nonzero RoPE offset, multi-token causal masking, exact capacity.
        rng = torch.Generator().manual_seed(42)
        ids = torch.randint(custom.config["vocab_size"], (2, 13), generator=rng).cuda()
        cache = KVCache(custom, 2, 13)
        pointers = (cache.k.data_ptr(), cache.v.data_ptr())
        state, offset = None, 0
        for length in (5, 3, 1, 4):
            chunk = ids[:, offset:offset + length]
            actual = custom.forward(chunk, cache)
            expected = hf(input_ids=chunk, past_key_values=state, use_cache=True,
                attention_mask=torch.ones((2, offset + length), dtype=torch.long, device="cuda"))
            state = expected.past_key_values
            offset += length
            assert cache.length == offset == state.get_seq_length()
            compare(f"batch2_chunk_end_{offset}", actual, expected.logits)
        assert pointers == (cache.k.data_ptr(), cache.v.data_ptr()), "Cache reallocated"
        before_k, before_v = cache.k.clone(), cache.v.clone()
        try:
            custom.forward(ids[:, :1], cache)
        except ValueError as exc:
            assert "capacity" in str(exc)
        else:
            raise AssertionError("Overflow was accepted")
        assert cache.length == 13 and torch.equal(cache.k, before_k) and torch.equal(cache.v, before_v)
        cache.reset()
        assert cache.length == 0
        other = ids.flip(1)[:, :7]
        compare("reset_request_isolation", custom.forward(other, cache),
                hf(input_ids=other, use_cache=False).logits)
        assert cache.length == 7
        old_length = cache.length
        try:
            custom.forward(ids[:1, :1], cache)
        except ValueError as exc:
            assert "batch" in str(exc)
        else:
            raise AssertionError("Batch mismatch was accepted")
        assert cache.length == old_length
        records.append({"case": "capacity_reset_batch_and_storage_checks", "passed": True})
    passed = all(r["passed"] for r in records)
    output = Path("artifacts/kv-cache-check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": passed, "cases": records}, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(f"Numerical mismatch: inspect {output}; tolerance unchanged")
    print(f"Custom KV cache verification OK: {output}")


if __name__ == "__main__":
    main()
