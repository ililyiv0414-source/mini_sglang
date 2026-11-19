"""Verify cross-prompt slot reuse using the production continuous scheduler."""
import json
from pathlib import Path

from benchmark_continuous import continuous, Qwen3
from benchmark_static_batch import run


def main():
    model = Qwen3("Qwen3-0.6B")
    ref = json.loads(Path("artifacts/reference.json").read_text(encoding="utf-8"))["results"]
    a, b = ref[:2]
    assert a["input_ids"] != b["input_ids"]
    cases = [dict(input_ids=a["input_ids"], max_new_tokens=3),
             dict(input_ids=b["input_ids"], max_new_tokens=64),
             dict(input_ids=b["input_ids"], max_new_tokens=5),
             dict(input_ids=a["input_ids"], max_new_tokens=64)]
    oracle = run(model, cases, list(range(4)), 1)["requests"]
    expected = [r["output"] for r in oracle]
    assert expected == [a["output_ids"][:3], b["output_ids"], b["output_ids"][:5], a["output_ids"]]
    report = []
    output = Path("artifacts/slot-isolation.json")
    for order in ([0, 1, 2, 3], [1, 0, 2, 3]):
        sample = continuous(model, cases, order)
        admissions = sample["admissions"]
        checks = []
        for rid, case in enumerate(order):
            row = sample["requests"][rid]
            actual, target = row["output"], expected[case]
            checks.append(dict(request=rid, case=case, matches=(row["request"] == rid and
                row["case"] == case and actual == target), expected=target, actual=actual))
        short, survivor = order.index(0), order.index(1)
        old, peer, new = admissions[short], admissions[survivor], admissions[2]
        # The initial prefill emits one token. Two decode steps retire the short request.
        handoff = (new["request"] == 2 and new["slot"] == old["slot"] != peer["slot"]
                   and new["peer"] == survivor and new["peer_length"] == len(b["input_ids"]) + 2
                   and sample["requests"][short]["end_s"] <= sample["requests"][2]["queue_s"]
                   < sample["requests"][survivor]["end_s"])
        passed = handoff and all(c["matches"] for c in checks)
        report.append(dict(order=order, passed=passed, handoff=handoff, admissions=admissions, checks=checks))
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("order", order, "handoff", handoff, "token checks", [c["matches"] for c in checks])
        if not passed:
            raise SystemExit(f"Isolation check failed: {output}")
    print("Cross-prompt slot isolation OK")


if __name__ == "__main__":
    main()
