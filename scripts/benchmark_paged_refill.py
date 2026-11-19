"""Equal KV budget FIFO refill with identical per-request eager execution."""
from collections import deque
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from benchmark_static_batch import KVCache, Qwen3, torch
from mini_engine.paged_cache import PagedStore, PagedDecodeBatch


@torch.inference_mode()
def run(model, baseline, workload, paged, direct=False, batched=False):
    budget = 64 << 20
    unit = (2 * model.config["num_hidden_layers"] * model.kv_heads * model.dim
            * torch.empty((), dtype=model.dtype).element_size())
    cases = baseline["prompts"]
    limits = [len(c["input_ids"]) + c["max_new_tokens"] - 1 for c in cases]
    capacity = max(limits)
    pages = budget // (16 * unit)
    if paged:
        store = PagedStore(model, pages, direct=direct)
        available = pages
        allocated = (store.k.numel() + store.v.numel()) * store.k.element_size()
    else:
        slots = [KVCache(model, 1, capacity) for _ in range(budget // (unit * capacity))]
        available = len(slots)
        allocated = sum(slot.allocated_bytes for slot in slots)
    pending = deque(enumerate(workload))
    active = {}
    rows = []
    peak = 0
    torch.cuda.synchronize()
    start = perf_counter()
    widths = []
    def emit(request, token):
        nonlocal available
        row, cache, cost = active[request]
        case = cases[row["case"]]
        if not row["output"]:
            row["ttft_s"] = perf_counter() - start
        row["output"].append(token)
        if token == model.config["eos_token_id"] or len(row["output"]) == case["max_new_tokens"]:
            del active[request]
            available += cost
            if paged:
                cache.release()
            else:
                cache.reset()
                slots.append(cache)

    while pending or active:
        while pending:
            request, case = pending[0]
            # Reserve sufficient credits for the declared generation limit.
            cost = math.ceil(limits[case] / 16) if paged else 1
            if cost > available:
                break
            pending.popleft()
            available -= cost
            cache = store.request() if paged else slots.pop()
            row = dict(request=request, case=case, output=[])
            active[request] = (row, cache, cost)
            rows.append(row)
        if not active:
            raise RuntimeError("Head request exceeds the KV budget")
        peak = max(peak, len(active))
        current = list(active)
        if batched:
            decoding = [r for r in current if active[r][0]["output"]]
            current = [r for r in current if not active[r][0]["output"]]
            if decoding:
                inputs = torch.tensor([[active[r][0]["output"][-1]] for r in decoding],
                                      device=model.device)
                batch = PagedDecodeBatch([active[r][1] for r in decoding])
                tokens = model.forward(inputs, batch)[:, -1].argmax(-1).tolist()
                widths.append(len(decoding))
                for request, token in zip(decoding, tokens):
                    emit(request, token)
        for request in current:
            row, cache, cost = active[request]
            case = cases[row["case"]]
            ids = row["output"][-1:] or case["input_ids"]
            x = torch.tensor([ids], device=model.device)
            token = model.forward(x, cache)[0, -1].argmax().item()
            emit(request, token)
    elapsed = perf_counter() - start
    if paged:
        assert len(store.pages.free) == pages and not store.pages.tables
    assert allocated <= budget
    for row in rows:
        if row["output"] != baseline["serial_expected"][row["case"]]:
            raise ValueError(f"Token mismatch for request {row['request']} in paged={paged}")
    return dict(decode_batch_widths=widths, peak_admitted=peak, kv_tensor_bytes=allocated, wall_s=elapsed,
                output_tok_s=sum(len(r["output"]) for r in rows) / elapsed,
                p95_ttft_s=sorted(r["ttft_s"] for r in rows)[math.ceil(.95 * len(rows)) - 1],
                requests=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="Include direct paged attention")
    parser.add_argument("--batched", action="store_true", help="Compare serial and batched direct decode")
    args = parser.parse_args()
    baseline = json.loads(Path("artifacts/static-batch.json").read_text())
    workload = baseline["workload"] * 2
    model = Qwen3("Qwen3-0.6B")
    samples = []
    modes = ["contiguous", "paged", "direct"] if args.direct else ["contiguous", "paged"]
    if args.batched:
        modes = ["direct", "batched"]
    for repeat in range(-1, 4 if args.batched else 3):
        shift = repeat % len(modes)
        for mode in modes[shift:] + modes[:shift]:
            paged = mode != "contiguous"
            sample = run(model, baseline, workload, paged, direct=mode in ("direct", "batched"),
                         batched=mode == "batched")
            if repeat >= 0:
                samples.append(dict(sample, paged=paged, mode=mode, repeat=repeat))
    report = dict(gpu=torch.cuda.get_device_name(), budget_bytes=64 << 20,
                  workload=workload, samples=samples)
    name = "direct-refill.json" if args.direct else "paged-refill.json"
    if args.batched:
        name = "batched-refill.json"
    (Path("artifacts") / name).write_text(json.dumps(report, indent=2))
    for mode in modes:
        subset = [s for s in samples if s["mode"] == mode]
        print(mode, {k: median(s[k] for s in subset)
                     for k in ("peak_admitted", "output_tok_s", "p95_ttft_s")})


if __name__ == "__main__":
    main()
