"""Replay verified two-slot admissions; count logical BF16 K/V storage."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--check-cuda", action="store_true")
    args = parser.parse_args()
    cfg = read(Path(args.model) / "config.json")
    unit = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    base = read("artifacts/static-batch.json")
    run = next(s for s in read("artifacts/continuous-batch.json")["samples"] if s["mode"] == "continuous")
    req = run["requests"]
    slots = [None, None]
    trace = []
    cursor = 0
    while cursor < len(req) or any(slots):
        for i in range(2):
            if slots[i] is None and cursor < len(req):
                row = req[cursor]
                event = run["admissions"][cursor]
                peer = slots[1-i]
                assert (event["slot"], event["request"], event["peer"], event["peer_length"]) == (
                    i, cursor, peer[0] if peer else None, peer[1] if peer else None)
                n = len(base["prompts"][row["case"]]["input_ids"])
                slots[i] = [cursor, n, len(row["output"])-1]
                cursor += 1
                trace.append([s[1] if s else 0 for s in slots])
                if slots[i][2] == 0:
                    slots[i] = None
        if cursor < len(req) and None in slots:
            continue
        for slot in slots:
            if slot:
                slot[1] += 1
                slot[2] -= 1
        trace.append([s[1] if s else 0 for s in slots])
        slots = [s if s and s[2] else None for s in slots]
    peak = max(map(sum, trace))
    cap = max(len(p["input_ids"]) + p["max_new_tokens"]-1 for p in base["prompts"])
    report = dict(bytes_per_token=unit, peak_live_tokens=peak, peak_live_bytes=peak*unit,
                  observed_peak_concurrency=max(sum(n>0 for n in t) for t in trace),
                  slot_lengths=trace, budget_bytes=64*1024**2, cases=[])
    if args.check_cuda:
        import torch
        from types import SimpleNamespace
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from mini_engine.qwen3 import KVCache
        model = SimpleNamespace(config=cfg,kv_heads=cfg["num_key_value_heads"],dim=cfg["head_dim"],device="cuda",dtype=torch.bfloat16)
    for capacity in (cap,128,256,512):
        reserved = 2*capacity*unit
        case = dict(capacity_per_slot=capacity, two_slot_bytes=reserved,
                    peak_live_fraction=peak*unit/reserved,
                    slots_in_budget=report["budget_bytes"]//(capacity*unit))
        if args.check_cuda:
            cache = KVCache(model,2,capacity)
            case["cuda_tensor_bytes"] = cache.allocated_bytes
            assert cache.allocated_bytes == reserved
            del cache
        report["cases"].append(case)
    path = Path("artifacts/contiguous-kv-capacity.json")
    path.write_text(json.dumps(report,indent=2))
    print(peak, "peak live tokens;", report["cases"])


if __name__ == "__main__":
    main()
