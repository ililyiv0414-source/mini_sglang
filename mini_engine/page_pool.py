"""CPU metadata allocator for fixed-size KV pages."""
class PagePool:
    def __init__(self, page_size, pages):
        if min(page_size, pages) < 1:
            raise ValueError("Invalid pool dimensions")
        self.page_size = page_size
        self.free = list(reversed(range(pages)))
        self.tables = {}

    def grow(self, request, tokens):
        if tokens < 1:
            raise ValueError("Nonpositive token count")
        table = self.tables.get(request, [])
        needed = (tokens + self.page_size - 1) // self.page_size - len(table)
        if needed < 0:
            raise ValueError("Cannot shrink a live page table")
        if needed > len(self.free):
            raise MemoryError("KV page pool exhausted")
        self.tables.setdefault(request, table).extend(self.free.pop() for _ in range(needed))

    def release(self, request):
        self.free.extend(self.tables.pop(request))
