"""Check ragged batched decode against independent contiguous-cache execution."""
import json
from pathlib import Path

from benchmark_static_batch import KVCache, Qwen3, torch
from mini_engine.paged_cache import PagedStore, PagedDecodeBatch


@torch.inference_mode()
def main():
    model = Qwen3("Qwen3-0.6B")
    store = PagedStore(model, pages=8, direct=True)
    prompts = json.loads(Path("artifacts/reference.json").read_text())["results"]
    rows = []
    requests, references, tokens = [], [], []
    for prompt, length in zip(prompts, (15, 23)):
        ids = torch.tensor([prompt["input_ids"][:length]], device=model.device)
        request = store.request()
        reference = KVCache(model, 1, 40)
        expected = model.forward(ids, reference)
        actual = model.forward(ids, request)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        tokens.append(expected[0, -1].argmax().item())
        requests.append(request)
        references.append(reference)
    report = Path("artifacts/batched-paged-check.json")
    for step in range(10):
        order = [0, 1] if step % 2 == 0 else [1, 0]
        ids = torch.tensor([[tokens[i]] for i in order], device=model.device)
        expected = torch.cat([model.forward(ids[j:j+1], references[i]) for j, i in enumerate(order)])
        actual = model.forward(ids, PagedDecodeBatch([requests[i] for i in order]))
        predicted = actual[:, -1].argmax(-1).tolist()
        oracle = expected[:, -1].argmax(-1).tolist()
        error = (expected.float() - actual.float()).abs()
        lengths = [r.length for r in requests]
        passed = (predicted == oracle and error.isfinite().all().item()
                  and lengths == [r.length for r in references])
        rows.append(dict(step=step, order=order, lengths=lengths, predicted=predicted,
                         expected=oracle, max_abs_error=error.max().item(), passed=passed))
        report.write_text(json.dumps(rows, indent=2))
        if not passed:
            raise RuntimeError(f"Step {step} failed; inspect {report}")
        for i, token in zip(order, oracle):
            tokens[i] = token
    for request in requests:
        request.release()
    assert not store.pages.tables and len(store.pages.free) == 8
    print("10 batched decode steps passed; final lengths", lengths, flush=True)


if __name__ == "__main__":
    main()
