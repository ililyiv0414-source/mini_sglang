"""Replay synchronized copies of observed two-slot demand under one KV budget."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "artifacts"
baseline = json.loads((root / "contiguous-kv-capacity.json").read_text())
page_size = 16
unit = baseline["bytes_per_token"]
budget = baseline["budget_bytes"]
page_count = budget // (unit * page_size)
peak_pages = max(
    sum((length + page_size - 1) // page_size for length in row)
    for row in baseline["slot_lengths"]
)
copies = page_count // peak_pages
result = {
    "budget_bytes": budget,
    "page_size": page_size,
    "available_pages": page_count,
    "peak_pages_per_trace": peak_pages,
    "paged_trace_copies": copies,
    "paged_peak_requests": copies * baseline["observed_peak_concurrency"],
    "fixed_slots": {case["capacity_per_slot"]: case["slots_in_budget"]
                    for case in baseline["cases"]},
}
(root / "equal-budget-capacity.json").write_text(json.dumps(result, indent=2))
print(result)
