"""Validate evolving graph state against fixed-capacity and sliced eager paths."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_engine.qwen3 import Qwen3, KVCache
from mini_engine.graph_decode import GraphDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3(args.model, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    decoder = GraphDecoder(model, 128)
    reference = json.loads(Path("artifacts/reference.json").read_text(encoding="utf-8"))
    rows = []

    def eager_generate(ids):
        cache = KVCache(model, 1, 128)
        output = model.forward(ids, cache)
        generated = []
        for step in range(64):
            token = output[0, -1].argmax().item()
            generated.append(token)
            if token == model.config["eos_token_id"]:
                break
            if step < 63:
                output = model.forward(torch.tensor([[token]], device="cuda"), cache)
        return generated
    with torch.inference_mode():
        # Reuse one graph across A -> B -> A to exercise reset and isolation.
        for index in (0, 1, 0):
            case = reference["results"][index]
            ids = torch.tensor([case["input_ids"]], device="cuda")
            decoder.prefill(ids)
            fixed = KVCache(model, 1, 128)
            fixed.k.zero_(); fixed.v.zero_()
            sliced = KVCache(model, 1, 128)
            model.forward(ids, fixed)
            model.forward(ids, sliced)
            for step in range(1, len(case["output_ids"])):
                # Identical history for all three paths, independent of argmax.
                token = torch.tensor([[case["output_ids"][step - 1]]], device="cuda")
                decoder.token.copy_(token)
                position = torch.tensor([len(case["input_ids"]) + step - 1], device="cuda")
                expected_fixed = model.forward(token, fixed, graph_position=position)
                expected_sliced = model.forward(token, sliced)
                actual = decoder.step()
                row = {"case": index, "step": step, "position": decoder.position.item(),
                    "graph_vs_fixed_exact": torch.equal(actual, expected_fixed),
                    "graph_vs_sliced_exact": torch.equal(actual, expected_sliced),
                    "graph_vs_sliced_max_abs": (actual.float() - expected_sliced.float()).abs().max().item(),
                    "graph_vs_sliced_same_argmax": actual.argmax(-1).item() == expected_sliced.argmax(-1).item(),
                    "token_feedback_correct": decoder.token.item() == actual.argmax(-1).item()}
                assert row["position"] == decoder.length == sliced.length
                rows.append(row)
            generated, reason = decoder.generate(ids, 64, {model.config["eos_token_id"]})
            eager_tokens = eager_generate(ids)
            rows.append({"case": index, "generation_exact": generated == eager_tokens,
                         "bf16_saved_reference_match": generated == case["output_ids"],
                         "eager_same_dtype_tokens": eager_tokens,
                         "generated": generated, "stop_reason": reason})
        ids = torch.tensor([[1, 2, 3]], device="cuda")
        generated, reason = decoder.generate(ids, 3, set())
        assert len(generated) == 3 and reason == "length" and decoder.length == 5
        first = decoder.prefill(ids).argmax(-1)[0, -1].item()
        generated, reason = decoder.generate(ids, 3, {first})
        assert generated == [first] and reason == "eos" and decoder.length == 3
        decoder.prefill(torch.ones((1, 127), device="cuda", dtype=torch.long))
        decoder.step()
        assert decoder.length == 128 and decoder.position.item() == 128
        try:
            decoder.step()
        except ValueError:
            pass
        else:
            raise AssertionError("Capacity overflow accepted")
    graph_ok = all(r.get("graph_vs_fixed_exact", True) and r.get("token_feedback_correct", True) for r in rows)
    sliced_ok = all(r.get("graph_vs_sliced_exact", True) and r.get("generation_exact", True) for r in rows)
    result = {"dtype": args.dtype, "graph_state_checks_passed": graph_ok, "sliced_eager_exact_passed": sliced_ok,
              "boundary_checks_passed": True, "capacity": 128, "rows": rows}
    path = Path(f"artifacts/dynamic-graph-check-{args.dtype}.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"graph_state={graph_ok}, sliced_eager_exact={sliced_ok}; report: {path}")
    if not graph_ok or not sliced_ok:
        raise SystemExit("Comparison difference: inspect report before performance measurement")
    print("Dynamic CUDA Graph verification OK")


if __name__ == "__main__":
    main()
