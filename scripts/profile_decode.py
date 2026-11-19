"""Decode-only diagnostic traces; profiler timings are not benchmark results."""
import argparse
import gzip
import hashlib
import json
import shutil
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from mini_engine.qwen3 import Qwen3, KVCache


@torch.inference_mode()
def execute(model, prompt, continuation, cached, steps, trace_path=None):
    # Fresh state for every run. Prefill and allocations precede measurement.
    tokens = torch.cat([prompt, continuation], dim=1)
    length = prompt.shape[1]
    cache = KVCache(model, 1, length + steps) if cached else None
    prefill = model.forward(prompt, cache)
    del prefill
    predictions = torch.empty((steps,), device="cuda", dtype=torch.long)
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=True, profile_memory=False, with_stack=False) if trace_path else None
    with profiler if profiler is not None else nullcontext():
        wall_start = time.perf_counter()
        start_event.record()
        for step in range(steps):
            end = length + step + 1
            ids = tokens[:, end - 1:end] if cached else tokens[:, :end]
            with record_function(f"decode_step_{step}") if profiler is not None else nullcontext():
                logits = model.forward(ids, cache)
                predictions[step].copy_(logits[0, -1].argmax())
                del logits
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
    result = {"wall_ms": wall_ms, "cuda_event_span_ms": start_event.elapsed_time(end_event),
              "predicted_ids": predictions.cpu().tolist()}
    if profiler is not None:
        profiler.export_chrome_trace(str(trace_path))
        averages = profiler.key_averages(group_by_input_shape=True)
        result["operators"] = [{"name": item.key, "count": item.count,
            "input_shapes": item.input_shapes, "self_cpu_us": item.self_cpu_time_total,
            "cpu_total_us": item.cpu_time_total,
            "self_device_us": getattr(item, "self_device_time_total", 0),
            "device_total_us": getattr(item, "device_time_total", 0)} for item in averages]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        events = trace.get("traceEvents", [])
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        result["cuda_kernel_events"] = len(kernels)
        result["kernel_duration_sum_us"] = sum(e.get("dur", 0) for e in kernels)
        # Union across streams: this is trace coverage, NOT SM utilization.
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
        coverage, right = 0.0, float("-inf")
        for left, end in intervals:
            coverage += max(0.0, end - max(left, right))
            right = max(right, end)
        result["kernel_union_us"] = coverage
        result["kernel_envelope_us"] = max(b for a, b in intervals) - intervals[0][0] if intervals else None
        result["cuda_trace_available"] = bool(kernels)
        with trace_path.open("rb") as source, gzip.open(str(trace_path) + ".gz", "wb") as dest:
            shutil.copyfileobj(source, dest)
        # Keep raw trace too, so a local trace viewer can open it directly.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="artifacts/benchmark-kv-fixed.json")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    source = json.loads(Path(args.benchmark).read_text())
    if args.steps < 1 or args.steps >= source["arguments"]["output_length"]:
        parser.error("steps must be positive and less than the benchmark output length")
    model = Qwen3(args.model)
    destination = Path("artifacts/profile-decode")
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "steps": args.steps, "cases": [],
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).resolve().parents[1] / "mini_engine/qwen3.py"]},
               "notes": "Prefill excluded. CUDA event span can include GPU idle gaps waiting for CPU. "
                        "Profiler adds overhead; use unprofiled timings for speed comparisons. "
                        "Kernel union/envelope is trace coverage, not SM utilization. Shared GPU."}
    for length in args.lengths:
        case = next(c for c in source["cases"] if c["input_length"] == length)
        prompt = torch.tensor([case["input_ids"]], device="cuda")
        continuation = torch.tensor([case["fixed_continuation_ids"]], device="cuda")
        row = {"input_length": length, "modes": {}}
        expected = case["expected_output_ids_by_mode"]
        for cached in (False, True):
            for _ in range(2):
                execute(model, prompt, continuation, cached, args.steps)
        for repeat in range(3):
            for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
                mode = "cache" if cached else "recompute"
                result = execute(model, prompt, continuation, cached, args.steps)
                assert result["predicted_ids"] == expected[mode][1:args.steps + 1], "Replay outputs changed"
                row["modes"].setdefault(mode, {"unprofiled": []})["unprofiled"].append(result)
        for cached in (False, True):
            mode = "cache" if cached else "recompute"
            print(f"Profiling input={length} mode={mode}", flush=True)
            trace = destination / f"t{length}-{mode}.json"
            result = execute(model, prompt, continuation, cached, args.steps, trace)
            assert result["predicted_ids"] == expected[mode][1:args.steps + 1], "Profile outputs changed"
            row["modes"][mode]["profiled"] = result
            row["modes"][mode]["median_unprofiled_ms_per_step"] = statistics.median(
                r["wall_ms"] for r in row["modes"][mode]["unprofiled"]) / args.steps
            print(f"CUDA kernel events: {result['cuda_kernel_events']}", flush=True)
        summary["cases"].append(row)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Profile complete: {destination}/summary.json")


if __name__ == "__main__":
    main()
