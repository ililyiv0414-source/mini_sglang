"""Paired eager decode with one active request and fixed input tokens."""
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from benchmark_static_batch import KVCache, Qwen3, torch
from mini_engine.paged_cache import PagedStore
from torch.profiler import ProfilerActivity, profile


@torch.inference_mode()
def replay(model, prompt, inputs, expected, paged, capture=False, direct=False):
    capacity = prompt.shape[1] + len(inputs)
    if paged:
        cache = PagedStore(model, math.ceil(capacity / 16), direct=direct).request()
    else:
        cache = KVCache(model, 1, capacity)
    model.forward(prompt, cache)
    predictions = []
    torch.cuda.synchronize()

    def decode():
        for ids in inputs:
            predictions.append(model.forward(ids, cache)[0, -1].argmax())
        torch.cuda.synchronize()

    if capture:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            decode()
        operations = {event.key: event for event in prof.key_averages()}
        result = {}
        for key in ("aten::index_select", "aten::index_copy_"):
            event = operations.get(key)
            result[key] = dict(count=event.count if event else 0,
                               cuda_total_us=event.device_time_total if event else 0)
    else:
        start = perf_counter()
        decode()
        result = (perf_counter() - start) * 1000 / len(inputs)
    if torch.stack(predictions).tolist() != expected:
        raise ValueError(f"Decode mismatch at prompt length {prompt.shape[1]}, paged={paged}")
    if paged:
        cache.release()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    modes = ("paged", "direct") if args.direct else ("contiguous", "paged")
    cases = json.loads(Path("artifacts/benchmark-kv-fixed.json").read_text())["cases"]
    model = Qwen3("Qwen3-0.6B")
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  decode_steps=8, profiled_steps=2, cases=[])
    for case in cases:
        prompt = torch.tensor([case["input_ids"]], device=model.device)
        inputs = torch.tensor(case["fixed_continuation_ids"][:8], device=model.device).view(-1, 1, 1).unbind()
        expected = case["expected_output_ids_by_mode"]["cache"][1:9]
        row = dict(input_length=case["input_length"], modes={})
        for mode in modes:
            row["modes"][mode] = dict(ms_per_token=[])
        for repeat in range(-1, 6):
            for mode in (modes[::-1] if repeat % 2 else modes):
                latency = replay(model, prompt, inputs, expected, mode != "contiguous", direct=mode == "direct")
                if repeat >= 0:
                    row["modes"][mode]["ms_per_token"].append(latency)
        for mode, data in row["modes"].items():
            data["median_ms_per_token"] = median(data["ms_per_token"])
            data["profile"] = replay(model, prompt, inputs[:2], expected[:2], mode != "contiguous", True, direct=mode == "direct")
            print(row["input_length"], mode, data["median_ms_per_token"], data["profile"], flush=True)
        report["cases"].append(row)
    Path("artifacts/direct-decode-profile.json" if args.direct else "artifacts/paged-decode-profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
