"""GPU KV storage with logical-to-physical indexing and eager gather attention."""
import torch
from .page_pool import PagePool


class PagedStore:
    def __init__(self, model, pages, page_size=16, direct=False):
        self.model = model
        self.direct = direct
        self.pages = PagePool(page_size, pages)
        shape = (model.config["num_hidden_layers"], pages * page_size, model.kv_heads, model.dim)
        self.k = torch.empty(shape, device=model.device, dtype=model.dtype)
        self.v = torch.empty_like(self.k)

    def request(self):
        return PagedRequest(self)


class PagedRequest:
    batch_size = 1

    def __init__(self, store):
        self.store = store
        self.owner = store.model
        self.capacity = self.owner.config["max_position_embeddings"]
        self.length = 0
        self.key = object()
        self.closed = False

    def direct_table(self, end):
        if self.closed:
            raise ValueError("Request already released")
        pool = self.store.pages
        pool.grow(self.key, end)
        return torch.tensor(pool.tables[self.key], device=self.owner.device)

    def attend(self, layer, table, end, q, k, v):
        from .paged_attention import attend
        store = self.store
        page_size = store.pages.page_size
        position = end - 1
        physical = store.pages.tables[self.key][position // page_size] * page_size + position % page_size
        for storage, new in ((store.k, k), (store.v, v)):
            storage[layer, physical].copy_(new[0, :, 0])
        return attend(q, store.k[layer], store.v[layer], table, end, page_size)

    def indices(self, end):
        if self.closed:
            raise ValueError("Request already released")
        pool = self.store.pages
        pool.grow(self.key, end)
        pages = torch.tensor(pool.tables[self.key], device=self.owner.device)
        positions = torch.arange(end, device=self.owner.device)
        return pages[positions // pool.page_size] * pool.page_size + positions % pool.page_size

    def update(self, layer, indices, start, k, v):
        result = []
        for storage, values in ((self.store.k, k), (self.store.v, v)):
            storage[layer].index_copy_(0, indices[start:], values[0].transpose(0, 1))
            result.append(storage[layer].index_select(0, indices).transpose(0, 1).unsqueeze(0))
        return result

    def release(self):
        if not self.closed and self.key in self.store.pages.tables:
            self.store.pages.release(self.key)
        self.closed = True
        self.length = 0

class PagedDecodeBatch:
    """One decode iteration over independent live requests in one physical store."""

    def __init__(self, requests):
        if not requests or len(set(requests)) != len(requests):
            raise ValueError("Expected distinct live requests")
        self.requests = list(requests)
        self.store = requests[0].store
        self.owner = self.store.model
        self.batch_size = len(requests)
        if any(r.store is not self.store for r in requests):
            raise ValueError("Requests must share the same KV store")

    def prepare(self):
        pool = self.store.pages
        starts = [r.length for r in self.requests]
        if any(r.closed or not r.length for r in self.requests):
            raise ValueError("Decode requires live prefilled requests")
        self.capacity = max(starts) + 1
        if (self.capacity > min(2048, self.owner.config["max_position_embeddings"])
                or self.owner.dtype != torch.bfloat16 or self.owner.dim != 128):
            raise ValueError("Unsupported paged decode dimensions")
        width = (self.capacity + pool.page_size - 1) // pool.page_size
        needed = sum(max(0, n // pool.page_size + 1 - len(pool.tables[r.key]))
                     for r, n in zip(self.requests, starts))
        if needed > len(pool.free):
            raise MemoryError("KV page pool exhausted")
        tables, destinations = [], []
        for r, n in zip(self.requests, starts):
            pool.grow(r.key, n + 1)
            pages = pool.tables[r.key]
            tables.append(pages + [0] * (width - len(pages)))
            destinations.append(pages[n // pool.page_size] * pool.page_size + n % pool.page_size)
        device = self.owner.device
        self.tables = torch.tensor(tables, device=device)
        self.destinations = torch.tensor(destinations, device=device)
        positions = torch.tensor(starts, device=device)
        self.lengths = positions + 1
        return positions

    def attend(self, layer, table, end, q, k, v):
        from .paged_attention import attend
        keys, values = self.store.k[layer], self.store.v[layer]
        keys.index_copy_(0, self.destinations, k[:, :, 0])
        values.index_copy_(0, self.destinations, v[:, :, 0])
        return attend(q, keys, values, self.tables, end, self.store.pages.page_size, self.lengths)

    def commit(self):
        for request in self.requests:
            request.length += 1
