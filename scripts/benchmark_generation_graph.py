import json
import sys
import time
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_engine.qwen3 import Qwen3
from mini_engine.graph_decode import GraphDecoder


@torch.inference_mode()
def main():
    model = Qwen3("Qwen3-0.6B")
    cases = json.loads(Path("artifacts/reference.json").read_text(encoding="utf8"))["results"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    decoder = GraphDecoder(model, 128)
    torch.cuda.synchronize()
    report = {"setup_including_warmup_and_capture_ms": 1000*(time.perf_counter()-start),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "capacity": 128,
              "dtype": "bf16", "cases": []}

    def request(ids, graph, capture_logits=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = decoder.prefill(ids)
        token = decoder.token.item()
        first = time.perf_counter()
        tokens = [token]
        snapshots = [logits.clone()] if capture_logits else []
        while token != model.config["eos_token_id"] and len(tokens) < 64:
            if graph:
                logits = decoder.step()
            else:
                logits = model.forward(decoder.token, decoder.cache, graph_position=decoder.position)
                decoder.token.copy_(logits[:, -1].argmax(-1)[:, None])
                decoder.position.add_(1)
                decoder.length += 1
                decoder.cache.length = decoder.length
            token = decoder.token.item()
            tokens.append(token)
            if capture_logits:
                snapshots.append(logits.clone())
        end = time.perf_counter()
        return tokens, snapshots, dict(reset_prefill_first_ms=1000*(first-start),
                                      decode_ms=1000*(end-first), total_ms=1000*(end-start))

    for case in cases:
        ids = torch.tensor([case["input_ids"]], device="cuda")
        assert ids.shape[1]+63 <= decoder.cache.capacity
        expected, baseline, _ = request(ids, False, True)
        generated, replay, _ = request(ids, True, True)
        assert generated == expected == case["output_ids"]
        assert all(torch.equal(a,b) and torch.isfinite(a).all() for a,b in zip(baseline,replay))
        del baseline, replay
        rows = {"graph": [], "eager": []}
        for repeat in range(6):
            for mode in (("graph","eager") if repeat%2 else ("eager","graph")):
                tokens, _, timing = request(ids, mode=="graph")
                assert tokens == expected
                rows[mode].append(timing)
        med = {mode:statistics.median(r["total_ms"] for r in times) for mode,times in rows.items()}
        report["cases"].append(dict(prompt=case["prompt"], generated_ids=expected, exact_logits=True,
                                    samples=rows, median_total_ms=med, speedup=med["eager"]/med["graph"]))
        Path("artifacts/generation-graph.json").write_text(json.dumps(report, indent=2))
        print(len(expected), "tokens; total ms:", med, flush=True)


if __name__ == "__main__":
    main()
