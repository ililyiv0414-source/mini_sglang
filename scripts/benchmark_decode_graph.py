"""Fixed-position decode graph experiment, NOT an autoregressive graph engine."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_engine.qwen3 import Qwen3, KVCache


def measure(fn, repeats):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin.record()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    submitted = time.perf_counter()
    end.record()
    torch.cuda.synchronize()
    done = time.perf_counter()
    return {"wall_ms_per_call": (done - start) * 1000 / repeats,
            "enqueue_ms_per_call": (submitted - start) * 1000 / repeats,
            "cuda_span_ms_per_call": begin.elapsed_time(end) / repeats}


@torch.inference_mode()
def experiment(model, case, calls):
    prefix = torch.tensor([case["input_ids"]], device="cuda")
    length = prefix.shape[1]
    cache = KVCache(model, 1, length + 1)
    model.forward(prefix, cache)
    # Graph always writes slot L and attends positions 0..L. Prefix never changes.
    prefix_k, prefix_v = cache.k[:, :, :, :length].clone(), cache.v[:, :, :, :length].clone()
    static_input = torch.tensor([[case["fixed_continuation_ids"][0]]], device="cuda")

    def eager():
        cache.length = length
        logits = model.forward(static_input, cache)
        return logits, logits[:, -1].argmax(-1)

    # Side-stream warmup initializes kernels/handles before graph capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    before = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    cache.length = length
    start = time.perf_counter()
    with torch.cuda.graph(graph):
        graph_logits = model.forward(static_input, cache)
        graph_token = graph_logits[:, -1].argmax(-1)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - start) * 1000
    after = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
             "capture_peak_allocated": torch.cuda.max_memory_allocated()}
    checks = []
    original_token = int(static_input.item())
    # Distinct token values, identical addresses, shapes and prefix length.
    for token in [original_token, (original_token + 17) % model.config["vocab_size"],
                  (original_token + 103) % model.config["vocab_size"], original_token]:
        static_input.fill_(token)
        expected, expected_token = eager()
        expected = expected.clone()
        expected_token = expected_token.clone()
        graph.replay()
        torch.cuda.synchronize()
        exact = torch.equal(graph_logits, expected) and torch.equal(graph_token, expected_token)
        checks.append({"input_token": token, "exact_logits_and_argmax": exact,
                       "max_abs_error": (graph_logits.float() - expected.float()).abs().max().item()})
        if not exact:
            raise RuntimeError(f"Graph output mismatch at length={length}, token={token}")
    assert torch.equal(prefix_k, cache.k[:, :, :, :length]), "Graph corrupted historical K"
    assert torch.equal(prefix_v, cache.v[:, :, :, :length]), "Graph corrupted historical V"
    del prefix_k, prefix_v, expected, expected_token
    static_input.fill_(original_token)
    # Capture and validation are excluded from throughput samples.
    for _ in range(3):
        eager()
        graph.replay()
    samples = []
    for repeat in range(5):
        for mode in (("eager", "graph") if repeat % 2 == 0 else ("graph", "eager")):
            sample = measure(eager if mode == "eager" else graph.replay, calls)
            sample.update(mode=mode, repeat=repeat)
            samples.append(sample)
            print(f"L={length} repeat={repeat} {mode}: {sample['wall_ms_per_call']:.4f} ms/call", flush=True)
    summary = {}
    for mode in ("eager", "graph"):
        values = [s["wall_ms_per_call"] for s in samples if s["mode"] == mode]
        summary[mode] = {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}
    summary["speedup"] = summary["eager"]["median_ms"] / summary["graph"]["median_ms"]
    return {"prefix_length": length, "checks": checks, "prefix_unchanged": True,
            "cache_capacity_bytes": cache.allocated_bytes,
            "capture_ms": capture_ms, "memory_before_capture": before, "memory_after_capture": after,
            "samples": samples, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calls", type=int, default=30)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    model = Qwen3(args.model, norm_backend="torch")
    source = json.loads(Path("artifacts/benchmark-kv-fixed.json").read_text())
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "calls_per_sample": args.calls,
              "norm_backend": "torch", "cases": [],
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).resolve().parents[1] / "mini_engine/qwen3.py"]},
              "scope": "Fixed prefix and fixed cache write position, batch=1, BF16. "
                       "No advancing positions or autoregressive token feedback. "
                       "Token upload, prefill, allocation, warmup and capture excluded from timing. "
                       "Input updates validated separately. Eager measured while graph pool remains resident. "
                       "CUDA event spans may include idle gaps. Shared GPU; not serving throughput."}
    output = Path("artifacts/decode-graph-fixed.json")
    for case in source["cases"]:
        if case["input_length"] not in (512, 1024):
            continue
        print(f"Capture and validate L={case['input_length']}", flush=True)
        report["cases"].append(experiment(model, case, args.calls))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"Fixed-position CUDA Graph benchmark complete: {output}")


if __name__ == "__main__":
    main()
