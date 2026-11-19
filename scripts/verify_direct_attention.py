"""Compare direct attention with gathered BF16 attention on reused, nonadjacent pages."""
from benchmark_static_batch import torch
from mini_engine.page_pool import PagePool
from mini_engine.paged_attention import attend


def main():
    torch.manual_seed(42)
    pool = PagePool(16, 132)
    pool.grow("first", 16)
    pool.grow("survivor", 16)
    pool.grow("first", 17)
    pool.release("first")
    shape = (132 * 16, 8, 128)
    k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    q = torch.randn((1, 16, 1, 128), device="cuda", dtype=k.dtype)
    for length in (1, 16, 17, 33, 129, 513, 1025):
        pool.grow("new", length)
        pages = pool.tables["new"]
        if length >= 17:
            assert pages[:2] == [2, 0]
        table = torch.tensor(pages, device=q.device)
        pos = torch.arange(length, device=q.device)
        indices = table[pos // 16] * 16 + pos % 16
        keys, values = [x[indices].transpose(0, 1).repeat_interleave(2, dim=0)
                        for x in (k, v)]
        probs = ((q @ keys.transpose(-2, -1)) * (128 ** -0.5)).float().softmax(-1).to(q.dtype)
        expected = probs @ values
        actual = attend(q, k, v, table, length, 16)
        error = (actual.float() - expected.float()).abs().max().item()
        print("length", length, "max_abs_error", error, flush=True)
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.016)


if __name__ == "__main__":
    main()
