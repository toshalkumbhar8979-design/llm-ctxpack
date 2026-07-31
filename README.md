# ctxpack

Pack the most important text into a fixed LLM context-window token budget.

Every RAG pipeline or agent eventually hits the same problem: you have more
candidate context (retrieved chunks, chat history, tool output, file
contents) than tokens to spend. `ctxpack` solves the "what do I keep, what do
I trim, what do I drop" decision for you — deterministically, and without
pulling in a full LLM framework.

## Install

```bash
pip install ctxpack

# optional: exact token counts via tiktoken (otherwise a ~4-char/token estimate is used)
pip install "ctxpack[tiktoken]"
```

## Quickstart

```python
from ctxpack import ContextItem, pack

items = [
    ContextItem(id="system-notes", text=system_notes, priority=10, truncatable=False),
    ContextItem(id="retrieved-doc-1", text=doc1, priority=5),
    ContextItem(id="retrieved-doc-2", text=doc2, priority=3),
    ContextItem(id="chat-history", text=history, priority=1, truncate_strategy="head"),
]

result = pack(items, budget=6000)  # tokens

print(result.total_tokens, "/", result.budget, "tokens used")
print([i.id for i in result.items])   # what got kept
print(result.dropped_ids)             # what got dropped entirely
prompt_context = result.render()      # joined text, ready to drop into a prompt
```

## How it works

1. Every item has a `priority` (higher = more important).
2. Items are considered highest-priority first. Ties are broken by
   "value density" — priority per token — so a small high-value item isn't
   starved by one large item of equal priority.
3. Each item is included **whole** if it fits in what's left of the budget.
4. If it doesn't fit and `truncatable=True`, it's trimmed to whatever budget
   remains, as long as the trimmed size clears `min_tokens` (otherwise
   trimming it further would produce useless mush, so it's dropped instead).
5. If `truncatable=False`, it's included whole or not at all.

Truncation can keep the `"tail"` (default — good for docs, keep the intro),
the `"head"` (good for chat history — keep the most recent turns), or the
`"middle"` (keep both ends, cut the middle — good for long files where the
top and bottom carry the most signal).

## Reserving room for a system prompt or expected output

```python
result = pack(items, budget=8000, reserve_tokens=1500)  # only 6500 available to items
```

## CLI

```bash
ctxpack doc1.md doc2.md notes.txt --budget 4000 --stats
# earlier files = higher priority; --stats prints what was kept/trimmed/dropped to stderr
```

```
ctxpack --help
```

## Why not just truncate the whole prompt?

Naive truncation (cut the end of the final concatenated string) treats every
token as equally disposable — you might cut the system prompt or the most
relevant retrieved doc just because it happened to be assembled last.
`ctxpack` decides *what* to cut based on *what you told it matters*, before
concatenation happens.

## License

MIT
