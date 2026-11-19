"""Offline closed workload: all requests arrive at time zero, FIFO static batches."""
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_engine.qwen3 import Qwen3, KVCache


@torch.inference_mode()
def run(model, cases, workload, width, limit=64):
    # Bucket by equal prompt length within each batch; workload order is fixed.
    batches = []
    for case_id in workload:
        if (not batches or len(batches[-1]) == width or
                len(cases[batches[-1][0]]["input_ids"]) != len(cases[case_id]["input_ids"])):
            batches.append([])
        batches[-1].append(case_id)
    rows, slots, idle = [], 0, 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in batches:
        admitted = time.perf_counter() - start
        prompts = [cases[i]["input_ids"] for i in batch]
        ids = torch.tensor(prompts, device="cuda")
        cache = KVCache(model, len(batch), ids.shape[1] + limit - 1)
        records = [dict(case=i, queue_s=admitted, output=[]) for i in batch]
        for step in range(limit):
            pred = model.forward(ids, cache)[:, -1].argmax(-1)
            values = pred.tolist()
            now = time.perf_counter() - start
            for row, token in zip(records, values):
                if "end_s" in row:
                    idle += 1
                    continue
                if step == 0:
                    row["ttft_s"] = now
                row["output"].append(token)
                if token == model.config["eos_token_id"] or step + 1 == cases[row["case"]]["max_new_tokens"]:
                    row["end_s"] = now
            slots += len(batch)
            if all("end_s" in row for row in records):
                break
            ids = pred[:, None]
        rows.extend(records)
        del cache
    elapsed = time.perf_counter() - start
    def p95(key):
        return sorted(r[key] for r in rows)[math.ceil(.95 * len(rows)) - 1]
    return dict(requests=rows, wall_s=elapsed,
                output_tok_s=sum(len(r["output"]) for r in rows)/elapsed,
                requests_s=len(rows)/elapsed, p95_ttft_s=p95("ttft_s"),
                p95_completion_s=p95("end_s"), retired_slot_steps=idle,
                executed_slot_steps=slots)


@torch.inference_mode()
def main():
    model = Qwen3("Qwen3-0.6B")
    cases = json.loads(Path("artifacts/reference.json").read_text(encoding="utf8"))["results"]
    # Same intact prompt; per-request limits isolate decode-length imbalance.
    prompt = cases[1]
    cases = [dict(input_ids=prompt['input_ids'], max_new_tokens=n) for n in (8, 64)]
    workload = [0, 1] * 4
    oracle = run(model, cases, [0, 1], 1)["requests"]
    expected = [r["output"] for r in oracle]
    assert expected == [prompt["output_ids"][:8], prompt["output_ids"]], "Serial reference changed"
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  prompts=cases, workload=workload, serial_expected=expected,
                  policy="all arrive together; FIFO; no refill until entire batch finishes; BF16 eager",
                  repeats=[])
    dest = Path("artifacts/static-batch.json")
    for repeat in range(-1, 6):
        for width in ((1, 2) if repeat % 2 == 0 else (2, 1)):
            sample = run(model, cases, workload, width)
            sample.update(width=width, repeat=repeat)
            ok = all(r["output"] == expected[r["case"]] for r in sample["requests"])
            if not ok:
                report["failed_sample"] = sample
                dest.write_text(json.dumps(report, indent=2))
                raise SystemExit("Batch/serial tokens differ; inspect artifacts/static-batch.json")
            if repeat >= 0:
                report["repeats"].append(sample)
                dest.write_text(json.dumps(report, indent=2))
                print(width, repeat, round(sample["output_tok_s"], 2), "output tokens/s", flush=True)
    for width in (1, 2):
        samples = [r for r in report["repeats"] if r["width"] == width]
        print("width", width, "median output tokens/s", statistics.median(r["output_tok_s"] for r in samples))
    print("Static batch baseline complete:", dest)


if __name__ == "__main__":
    main()

