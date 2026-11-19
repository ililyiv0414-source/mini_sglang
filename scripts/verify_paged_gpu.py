import json
from pathlib import Path
from benchmark_static_batch import Qwen3, KVCache, torch
from mini_engine.paged_cache import PagedStore


@torch.inference_mode()
def main():
    model = Qwen3("Qwen3-0.6B")
    store = PagedStore(model, pages=4)
    prompts = [r["input_ids"] for r in json.loads(Path("artifacts/reference.json").read_text())["results"]]
    rows = []
    def create():
        return store.request(), KVCache(model, 1, 64)
    def append(pair, ids):
        paged, contiguous = pair
        tokens = torch.tensor([ids], device="cuda")
        expected = model.forward(tokens, contiguous)
        actual = model.forward(tokens, paged)
        table = list(store.pages.tables[paged.key])
        # Independently reconstruct the logical prefix from physical page slices.
        for physical, baseline in ((store.k, contiguous.k), (store.v, contiguous.v)):
            blocks = [physical[:, p*16:(p+1)*16] for p in table]
            rebuilt = torch.cat(blocks, dim=1)[:, :paged.length].transpose(1, 2)
            assert torch.equal(rebuilt, baseline[:, 0, :, :paged.length]), "KV mismatch"
        assert torch.equal(expected, actual), "Logits mismatch"
        rows.append(dict(length=paged.length, pages=table, exact=True))
        return table
    a, b = create(), create()
    first = append(a, prompts[0][:16])
    survivor = append(b, prompts[1][:16])
    old = append(a, prompts[0][16:17])
    assert len(first)==1 and len(old)==2 and old[0]==first[0] and old[1]!=first[0]+1
    assert not set(old).intersection(survivor)
    a[0].release()
    c = create()
    reused = append(c, prompts[1][3:20])
    assert set(reused)==set(old)
    append(b, prompts[1][16:17])
    append(c, prompts[1][20:21])
    b[0].release()
    c[0].release()
    assert len(set(store.pages.free))==4 and not store.pages.tables
    Path("artifacts/paged-gpu-check.json").write_text(json.dumps(rows, indent=2))
    print("Paged GPU KV verification OK")


if __name__ == "__main__":
    main()
