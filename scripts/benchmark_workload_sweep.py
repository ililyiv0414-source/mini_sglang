"""Sensitivity to request count and output-length mix on one shared GPU."""
import json
from pathlib import Path
from statistics import median

from benchmark_paged_refill import Qwen3, run, torch


def main():
    root = Path("artifacts")
    baseline = json.loads((root / "static-batch.json").read_text())
    model = Qwen3("Qwen3-0.6B")
    workloads = {
        "4-balanced": [0, 1] * 2,
        "16-balanced": [0, 1] * 8,
        "32-balanced": [0, 1] * 16,
        "16-short-heavy": [0, 0, 0, 1] * 4,
        "16-long-heavy": [0, 1, 1, 1] * 4,
    }
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  budget_bytes=64 << 20, cases={})
    for label, workload in workloads.items():
        samples = []
        for repeat in range(-1, 4):
            for batched in ((False, True) if repeat % 2 == 0 else (True, False)):
                sample = run(model, baseline, workload, paged=True, direct=True, batched=batched)
                if repeat >= 0:
                    samples.append(dict(sample, batched=batched, repeat=repeat))
        summary = {}
        for batched, mode in ((False, "serial"), (True, "batched")):
            group = [s for s in samples if s["batched"] == batched]
            summary[mode] = {k: median(s[k] for s in group)
                             for k in ("output_tok_s", "p95_ttft_s", "peak_admitted")}
            if batched:
                widths = [w for s in group for w in s["decode_batch_widths"]]
                summary[mode]["mean_decode_width"] = sum(widths) / len(widths)
        summary["speedup"] = summary["batched"]["output_tok_s"] / summary["serial"]["output_tok_s"]
        report["cases"][label] = dict(workload=workload, samples=samples, summary=summary)
        (root / "workload-sweep.json").write_text(json.dumps(report, indent=2))
        print(label, summary, flush=True)


if __name__ == "__main__":
    main()
