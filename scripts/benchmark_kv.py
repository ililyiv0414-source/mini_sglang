"""Single-request offline KV ablation. Not a serving/SLO benchmark."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM
from mini_engine.qwen3 import Qwen3, KVCache


def gpu_snapshot():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


@torch.inference_mode()
def run(model, prompt, output_length, cached, continuation):
    # Setup excluded from timed phases; report preallocated cache explicitly.
    length = prompt.shape[1]
    tokens = torch.empty((1, length + output_length), dtype=torch.long, device="cuda")
    tokens[:, :length].copy_(prompt)
    tokens[:, length:].copy_(continuation)
    predictions = torch.empty((1, output_length), dtype=torch.long, device="cuda")
    cache = KVCache(model, 1, length + output_length - 1) if cached else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    start = time.perf_counter()
    logits = model.forward(tokens[:, :length], cache)
    predictions[:, 0].copy_(logits[:, -1].argmax(-1))
    del logits
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    for step in range(1, output_length):
        current = length + step
        ids = tokens[:, current - 1:current] if cached else tokens[:, :current]
        logits = model.forward(ids, cache)
        predictions[:, step].copy_(logits[:, -1].argmax(-1))
        del logits
    torch.cuda.synchronize()
    end = time.perf_counter()
    result = {
        "mode": "cache" if cached else "recompute",
        "prefill_ms": (prefill_end - start) * 1000,
        "decode_ms": (end - prefill_end) * 1000,
        "total_ms": (end - start) * 1000,
        "decode_steps": output_length - 1,
        "decode_tokens_per_second": (output_length - 1) / (end - prefill_end),
        "output_tokens_per_second": output_length / (end - start),
        "allocated_before_timing_bytes": baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cache_capacity_bytes": cache.allocated_bytes if cache else 0,
        "output_ids": predictions.cpu().tolist()[0],
    }
    return result


@torch.inference_mode()
def validate_paths(model, model_dir, prompt, continuation):
    """Untimed exact comparison against HF for each execution shape separately."""
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True, trust_remote_code=False).cuda().eval()
    tokens = torch.cat([prompt, continuation], dim=1)
    length, count = prompt.shape[1], continuation.shape[1]
    expected = {}
    for cached in (False, True):
        cache = KVCache(model, 1, length + count - 1) if cached else None
        state, predictions = None, []
        for step in range(count):
            end = length + step
            ids = tokens[:, end - 1:end] if cached and step else tokens[:, :end]
            actual = model.forward(ids, cache)
            reference = hf(input_ids=ids, use_cache=cached, past_key_values=state,
                attention_mask=torch.ones((1, end), dtype=torch.long, device="cuda"))
            state = reference.past_key_values if cached else None
            if not torch.equal(actual, reference.logits):
                error = (actual.float() - reference.logits.float()).abs().max().item()
                raise RuntimeError(f"HF comparison failed: cached={cached}, step={step}, max_abs={error}")
            predictions.append(actual[0, -1].argmax().item())
        expected["cache" if cached else "recompute"] = predictions
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="artifacts/benchmark-kv-fixed.json")
    args = parser.parse_args()
    if min(args.input_lengths) < 1 or args.output_length < 2 or args.repeats < 3 or args.warmup < 1:
        parser.error("Require positive input lengths, output >=2, repeats >=3 and warmup >=1")
    torch.manual_seed(2026)
    model = Qwen3(args.model)
    if max(args.input_lengths) + args.output_length - 1 > model.config["max_position_embeddings"]:
        parser.error("Context limit exceeded")
    generator = torch.Generator().manual_seed(2026)
    continuation_generator = torch.Generator().manual_seed(2027)
    manifest = Path(args.model) / "SHA256SUMS"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "batch_size": 1,
        "arguments": vars(args), "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), Path(__file__).resolve().parents[1] / "mini_engine/qwen3.py"]},
        "protocol": "Fixed synthetic token replay, NOT free-running generation. Both paths see identical histories. "
                    "Each path is separately validated with exact logits against HF before measurement. "
                    "CUDA-synchronized wall time includes Python and token selection. "
                    "Token-buffer/cache allocation and model loading excluded from timings. "
                    "Prefill includes first token selection; decode counts output_length-1 steps. "
                    "Full logits computed for every input position in both paths. "
                    "Memory includes resident model and cache; reserved memory is allocator-history-dependent. "
                    "Shared GPU, no isolation guarantee; no serving latency or SLO claim.",
        "cases": [],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    for length in args.input_lengths:
        prompt = torch.randint(model.config["vocab_size"], (1, length), generator=generator).cuda()
        continuation = torch.randint(model.config["vocab_size"], (1, args.output_length),
                                     generator=continuation_generator).cuda()
        case = {"input_length": length, "input_ids": prompt.cpu().tolist()[0],
                "fixed_continuation_ids": continuation.cpu().tolist()[0],
                "gpu_before": gpu_snapshot(), "measurements": [], "status": "running"}
        report["cases"].append(case)
        save()
        print(f"input={length}: correctness check and warmup", flush=True)
        try:
            expected = validate_paths(model, args.model, prompt, continuation)
        except Exception as exc:
            case.update(status="correctness_failed", error=str(exc))
            save()
            raise
        case["hf_validation"] = "exact logits at all replay steps, separately for each mode"
        case["expected_output_ids_by_mode"] = expected
        # HF model and validation temporaries are released before memory/timing runs.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for _ in range(args.warmup):
            for cached in (False, True):
                result = run(model, prompt, args.output_length, cached, continuation)
                if result["output_ids"] != expected[result["mode"]]:
                    case.update(status="correctness_failed", mismatch=result)
                    save()
                    raise SystemExit("Generated tokens differ; benchmark aborted")
        for repeat in range(args.repeats):
            for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
                result = run(model, prompt, args.output_length, cached, continuation)
                result["repeat"] = repeat
                result["tokens_match"] = result["output_ids"] == expected[result["mode"]]
                case["measurements"].append(result)
                save()
                if not result["tokens_match"]:
                    case["status"] = "correctness_failed"
                    save()
                    raise SystemExit("Generated tokens changed during timing")
                print(f"input={length} repeat={repeat} mode={result['mode']} "
                      f"prefill={result['prefill_ms']:.2f}ms decode={result['decode_ms']:.2f}ms", flush=True)
        summary = {}
        for mode in ("recompute", "cache"):
            rows = [r for r in case["measurements"] if r["mode"] == mode]
            summary[mode] = {key: {"median": statistics.median(r[key] for r in rows),
                                  "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
                             for key in ("prefill_ms", "decode_ms", "total_ms",
                                         "decode_tokens_per_second", "output_tokens_per_second",
                                         "peak_allocated_bytes", "peak_reserved_bytes")}
        summary["decode_speedup_ratio"] = summary["recompute"]["decode_ms"]["median"] / summary["cache"]["decode_ms"]["median"]
        case.update(status="complete", summary=summary, gpu_after=gpu_snapshot())
        save()
        print(f"input={length}: decode speedup={summary['decode_speedup_ratio']:.3f}x", flush=True)
    print(f"Benchmark complete: {path}")


if __name__ == "__main__":
    main()
