import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mini_engine.page_pool import PagePool

root = Path("artifacts")
base = json.loads((root/"contiguous-kv-capacity.json").read_text())
run = next(r for r in json.loads((root/"continuous-batch.json").read_text())["samples"] if r["mode"]=="continuous")
unit = base["bytes_per_token"]
results = []
for size in (4,8,16,32):
    count = base["budget_bytes"]//(unit*size)
    pool = PagePool(size,count)
    prev = [0,0]
    owners = [None,None]
    admissions = iter(run["admissions"])
    peak = waste = reused = 0
    seen = set()
    for lengths in base["slot_lengths"]:
        # A slot that starts a new request loses all pages of its prior owner.
        for i,n in enumerate(lengths):
            if owners[i] is not None and (not n or n < prev[i]):
                pool.release(owners[i])
                owners[i] = None
        for i,n in enumerate(lengths):
            if not n:
                continue
            if owners[i] is None:
                event = next(admissions)
                assert event["slot"] == i
                owners[i] = event["request"]
            owner = owners[i]
            old = len(pool.tables.get(owner,[]))
            pool.grow(owner,n)
            new = pool.tables[owner][old:]
            reused += sum(p in seen for p in new)
            seen.update(new)
        live = [p for table in pool.tables.values() for p in table]
        assert len(set(live + pool.free)) == len(live)+len(pool.free) == count
        peak = max(peak,len(live))
        waste = max(waste,len(live)*size-sum(lengths))
        prev = lengths
    assert next(admissions,None) is None
    for owner in list(pool.tables):
        pool.release(owner)
    assert len(pool.free)==count
    results.append(dict(page_tokens=size,peak_pages=peak,peak_assigned_bytes=peak*size*unit,
                        max_unused_assigned_tokens=waste,reused_page_allocations=reused))
(root/"page-replay.json").write_text(json.dumps(results,indent=2))
print(results)
