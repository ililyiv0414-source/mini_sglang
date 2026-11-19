"""FIFO refill baseline, two physical slots and per-slot KV positions."""
from collections import deque
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from benchmark_static_batch import run as static_run, Qwen3, KVCache, torch


@torch.inference_mode()
def continuous(model, cases, workload):
    pending = deque(enumerate(workload))
    rows, events = [], []
    active = [None, None]
    capacity = max(len(c["input_ids"]) + c["max_new_tokens"] - 1 for c in cases)
    torch.cuda.synchronize()
    start = time.perf_counter()
    cache = KVCache(model, 2, capacity)
    cache.k.zero_()
    cache.v.zero_()
    positions = [0, 0]
    executed = 0

    def emit(slot, token, now):
        row = active[slot]
        if not row["output"]:
            row["ttft_s"] = now
        row["output"].append(token)
        if token == model.config["eos_token_id"] or len(row["output"]) == cases[row["case"]]["max_new_tokens"]:
            row["end_s"] = now
            active[slot] = None

    while pending or any(active):
        for slot in range(2):
            if active[slot] is not None or not pending:
                continue
            request, case = pending.popleft()
            row = dict(request=request, case=case, queue_s=time.perf_counter()-start, output=[])
            rows.append(row)
            events.append(dict(slot=slot, request=request, peer=active[1-slot]["request"] if active[1-slot] else None,
                               peer_length=positions[1-slot] if active[1-slot] else None))
            active[slot] = row
            prompt = torch.tensor([cases[case]["input_ids"]], device="cuda")
            # Prefill directly into a view of the vacant slot.
            view = SimpleNamespace(owner=model, batch_size=1, capacity=capacity, length=0,
                                   k=cache.k[:, slot:slot+1], v=cache.v[:, slot:slot+1])
            view.k.zero_()
            view.v.zero_()
            token = model.forward(prompt, view)[0,-1].argmax().item()
            positions[slot] = view.length
            executed += 1
            emit(slot, token, time.perf_counter()-start)
        if pending and None in active:
            continue
        slots = [i for i in range(2) if active[i] is not None]
        if not slots:
            continue
        first, last = slots[0], slots[-1]+1
        view = SimpleNamespace(owner=model, batch_size=len(slots), capacity=capacity, length=0,
                               k=cache.k[:, first:last], v=cache.v[:, first:last])
        ids = torch.tensor([[active[i]["output"][-1]] for i in slots], device="cuda")
        pos = torch.tensor([positions[i] for i in slots], device="cuda")
        assert max(positions[i] for i in slots) < capacity
        values = model.forward(ids, view, graph_position=pos)[:, -1].argmax(-1).tolist()
        now = time.perf_counter()-start
        executed += len(slots)
        for i, token in zip(slots, values):
            positions[i] += 1
            emit(i, token, now)
    elapsed = time.perf_counter()-start
    return dict(requests=rows, admissions=events, executed_slot_steps=executed, retired_slot_steps=0,
                wall_s=elapsed, output_tok_s=sum(len(r["output"]) for r in rows)/elapsed,
                requests_s=len(rows)/elapsed, p95_ttft_s=max(r["ttft_s"] for r in rows),
                p95_completion_s=max(r["end_s"] for r in rows))


def main():
    baseline = json.loads(Path("artifacts/static-batch.json").read_text())
    model = Qwen3("Qwen3-0.6B")
    cases, workload = baseline["prompts"], baseline["workload"]
    report = dict(policy="FIFO, individual prefill, fixed-capacity masked batched decode", samples=[])
    output = Path("artifacts/continuous-batch.json")
    for repeat in range(-1, 6):
        for mode in (("static","continuous") if repeat % 2 == 0 else ("continuous","static")):
            sample = continuous(model,cases,workload) if mode=="continuous" else static_run(model,cases,workload,2)
            sample.update(mode=mode, repeat=repeat)
            if ([r["case"] for r in sample["requests"]] != workload or
                any(r["output"] != baseline["serial_expected"][r["case"]] for r in sample["requests"])):
                report["failed_sample"] = sample
                output.write_text(json.dumps(report,indent=2))
                raise SystemExit("Output mismatch; diagnostic saved")
            if repeat >= 0:
                report["samples"].append(sample)
                output.write_text(json.dumps(report,indent=2))
    for mode in ("static","continuous"):
        print(mode, statistics.median(s["output_tok_s"] for s in report["samples"] if s["mode"]==mode), "output tokens/s")


if __name__ == "__main__":
    main()
