"""Same-model ablation of torch RMSNorm vs partially fused RMSNorm."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from benchmark_kv import run, gpu_snapshot
from profile_decode import execute
from mini_engine.qwen3 import Qwen3, KVCache


@torch.inference_mode()
def verify(model, prompt, continuation):
    caches = [KVCache(model, 1, prompt.shape[1] + continuation.shape[1]) for _ in range(2)]
    ids = prompt
    for step in range(continuation.shape[1]):
        model.set_norm_backend("torch")
        expected = model.forward(ids, caches[0])
        model.set_norm_backend("triton")
        actual = model.forward(ids, caches[1])
        if not torch.equal(expected, actual):
            raise RuntimeError(f"Exact logits check failed at step {step}")
        ids = continuation[:, step:step + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    gate = json.loads(Path("artifacts/rmsnorm-tail-check.json").read_text())
    if not gate["passed"]:
        raise SystemExit("RMSNorm correctness gate has not passed")
    source = json.loads(Path("artifacts/benchmark-kv-fixed.json").read_text())
    model = Qwen3(args.model)
    folder = Path("artifacts/rmsnorm-benchmark")
    folder.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name("benchmark_kv.py"),
             Path(__file__).with_name("profile_decode.py"),
             Path(__file__).resolve().parents[1] / "mini_engine/qwen3.py",
             Path(__file__).resolve().parents[1] / "mini_engine/rmsnorm.py"]
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "protocol": "BF16, contiguous KV cache, fixed token replay, 2 warmups, "
                          "5 alternating repetitions. Same model weights; only norm backend changes. "
                          "Loading/allocation/JIT excluded. Profiles collected AFTER timing. "
                          "Not a full single-kernel RMSNorm; reduction/rsqrt remain in torch.",
              "cases": []}
    def save():
        (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for original in source["cases"]:
        prompt = torch.tensor([original["input_ids"]], device="cuda")
        continuation = torch.tensor([original["fixed_continuation_ids"]], device="cuda")
        count = continuation.shape[1]
        row = {"input_length": prompt.shape[1], "prediction_count": count,
               "gpu_before": gpu_snapshot(), "measurements": [], "status": "validating"}
        report["cases"].append(row)
        save()
        print(f"Validating input={prompt.shape[1]}", flush=True)
        verify(model, prompt, continuation)
        row["exact_logits_verified"] = True
        for _ in range(2):
            for backend in ("torch", "triton"):
                model.set_norm_backend(backend)
                run(model, prompt, count, True, continuation)
        for repeat in range(5):
            for backend in (("torch", "triton") if repeat % 2 == 0 else ("triton", "torch")):
                model.set_norm_backend(backend)
                result = run(model, prompt, count, True, continuation)
                assert result["output_ids"] == original["expected_output_ids_by_mode"]["cache"]
                result.update(backend=backend, repeat=repeat)
                row["measurements"].append(result)
                save()
                print(f"input={prompt.shape[1]} repeat={repeat} {backend} decode={result['decode_ms']:.3f}ms", flush=True)
        row["summary"] = {}
        for backend in ("torch", "triton"):
            values = [r for r in row["measurements"] if r["backend"] == backend]
            row["summary"][backend] = {key: {"median": statistics.median(r[key] for r in values),
                "min": min(r[key] for r in values), "max": max(r[key] for r in values)}
                for key in ("prefill_ms", "decode_ms", "total_ms", "peak_allocated_bytes")}
        row["decode_speedup"] = row["summary"]["torch"]["decode_ms"]["median"] / row["summary"]["triton"]["decode_ms"]["median"]
        row.update(status="timing_complete", gpu_after=gpu_snapshot())
        save()
    # All performance samples finish before any profiler is enabled.
    for row, original in zip(report["cases"], source["cases"]):
        prompt = torch.tensor([original["input_ids"]], device="cuda")
        continuation = torch.tensor([original["fixed_continuation_ids"]], device="cuda")
        row["profiles"] = {}
        for backend in ("torch", "triton"):
            model.set_norm_backend(backend)
            trace = folder / f"t{prompt.shape[1]}-{backend}.json"
            result = execute(model, prompt, continuation, True, 4, trace)
            assert result["predicted_ids"] == original["expected_output_ids_by_mode"]["cache"][1:5]
            row["profiles"][backend] = result
            save()
        row["status"] = "complete"
        save()
    print(f"RMSNorm benchmark complete: {folder}/summary.json")


if __name__ == "__main__":
    main()
