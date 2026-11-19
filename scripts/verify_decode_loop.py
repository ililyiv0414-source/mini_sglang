"""Compare an explicit cached decode loop with saved HF generate results."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/reference.json"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/decode-check.json"))
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    assert torch.__version__ == ref["torch"], "PyTorch version differs from reference"
    assert transformers.__version__ == ref["transformers"], "Transformers version differs"
    assert ref["dtype"] == "bfloat16" and ref["attention"] == "eager"
    assert ref["do_sample"] is False
    digest = hashlib.sha256((args.model / "SHA256SUMS").read_bytes()).hexdigest()
    assert digest == ref["model_manifest_sha256"], "Model manifest differs"
    # Reference paths are recorded relative to the project working directory.
    saved_logits = torch.load(ref["first_token_logits_file"], map_location="cpu", weights_only=True)
    assert len(saved_logits) == len(ref["results"])
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    cases = []
    with torch.inference_mode():
        for case_index, reference in enumerate(ref["results"]):
            prompt = reference["input_ids"]
            input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
            # Prefill: all prompt tokens enter the model exactly once.
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        use_cache=True)
            cache = out.past_key_values
            first = out.logits[:, -1, :].float().cpu()
            first_equal = torch.equal(first, saved_logits[case_index])
            error = (first - saved_logits[case_index]).abs()
            generated, trace = [], []
            stop_reason = "length"
            for step in range(ref["max_new_tokens"]):
                # Before predicting token n, the cache contains prompt + n prior tokens.
                expected_length = len(prompt) + step
                cache_length = int(cache.get_seq_length())
                assert cache_length == expected_length, (case_index, step, cache_length)
                token = int(out.logits[0, -1].argmax().item())
                generated.append(token)
                trace.append({"step": step, "input_tokens": len(prompt) if step == 0 else 1,
                              "cache_length": cache_length, "predicted_token": token})
                if token in eos_ids:
                    stop_reason = "eos"
                    break
                if step + 1 == ref["max_new_tokens"]:
                    break
                # Decode: only the selected token enters; earlier K/V come from cache.
                out = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device="cuda"),
                    attention_mask=torch.ones((1, expected_length + 1),
                                              dtype=torch.long, device="cuda"),
                    past_key_values=cache, use_cache=True,
                )
                cache = out.past_key_values
            expected = reference["output_ids"]
            mismatch = next((i for i in range(max(len(generated), len(expected)))
                             if i >= len(generated) or i >= len(expected)
                             or generated[i] != expected[i]), None)
            passed = mismatch is None and first_equal
            cases.append({"case": case_index, "passed": passed,
                          "prompt_tokens": len(prompt), "generated_tokens": len(generated),
                          "first_logits_exact_match": first_equal,
                          "first_logits_max_abs_error": error.max().item(),
                          "first_logits_mean_abs_error": error.mean().item(),
                          "first_token_mismatch_index": mismatch,
                          "output_ids": generated, "stop_reason": stop_reason, "trace": trace})
            print(f"case={case_index} passed={passed} output_tokens={len(generated)} "
                  f"stop={stop_reason} max_logit_error={error.max().item()}", flush=True)
    passed = all(case["passed"] for case in cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": passed, "cases": cases,
                                      "scope": "HF model with manual cached generation; not a custom engine"},
                                     indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(f"Comparison failed; inspect {args.output}; no tolerance was relaxed")
    print(f"Explicit decode verification OK: {args.output}")


if __name__ == "__main__":
    main()
