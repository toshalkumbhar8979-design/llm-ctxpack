"""Core packing logic for llm-ctxpack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

from .tokenizers import count_tokens, truncate_to_tokens

TruncateStrategy = Literal["head", "tail", "middle"]


@dataclass
class ContextItem:
    """One candidate piece of text competing for a slot in the context window.

    Attributes:
        id: Any identifier you'll recognize later (filename, doc id, message id...).
        text: The raw text.
        priority: Higher = more important. Ties broken by best token-efficiency
            (more "value" per token), then by original order.
        truncatable: If False, the item is included whole or not at all.
            If True and it doesn't fully fit, it may be trimmed to fit the
            remaining budget (subject to min_tokens).
        min_tokens: Smallest useful size after truncation. If the remaining
            budget can't fit at least this many tokens, the item is dropped
            instead of being truncated into uselessness.
        truncate_strategy: "head" | "tail" | "middle" — which part of the
            text to keep when trimming.
    """

    id: str
    text: str
    priority: float = 1.0
    truncatable: bool = True
    min_tokens: int = 40
    truncate_strategy: TruncateStrategy = "tail"


@dataclass
class PackedItem:
    """A ContextItem after packing, possibly truncated."""

    id: str
    text: str
    tokens: int
    priority: float
    truncated: bool
    original_tokens: int


@dataclass
class PackResult:
    items: List[PackedItem] = field(default_factory=list)
    dropped_ids: List[str] = field(default_factory=list)
    total_tokens: int = 0
    budget: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.budget - self.total_tokens)

    def render(self, separator: str = "\n\n---\n\n") -> str:
        """Join the packed items' text back into one context blob, in the
        order they were packed (highest priority first)."""
        return separator.join(item.text for item in self.items)


def pack(
    items: Sequence[ContextItem],
    budget: int,
    encoding: str = "cl100k_base",
    reserve_tokens: int = 0,
) -> PackResult:
    """Pack ``items`` into a token ``budget``, keeping the highest-priority
    content and trimming or dropping the rest.

    Algorithm: priority-sorted greedy fill. Items are considered highest
    priority first; ties are broken by "value density" (priority per token)
    so small high-value items aren't starved by one large item of equal
    priority. Each item is included whole if it fits; if it doesn't fit but
    is truncatable, it's trimmed to whatever budget remains (as long as the
    trimmed size clears ``min_tokens``); otherwise it's dropped.

    Args:
        items: Candidate ContextItems.
        budget: Total token budget available for packed content.
        encoding: tiktoken encoding name (ignored if tiktoken isn't installed).
        reserve_tokens: Tokens to hold back off the top (e.g. for a system
            prompt or expected model output), subtracted from ``budget``.

    Returns:
        PackResult with the selected/trimmed items, in priority order.
    """
    effective_budget = max(0, budget - reserve_tokens)

    sized = []
    for idx, item in enumerate(items):
        tok = count_tokens(item.text, encoding=encoding)
        density = (item.priority / tok) if tok > 0 else item.priority
        sized.append((item, tok, density, idx))

    # Highest priority first; among equal priority, prefer denser value;
    # stable on original order beyond that.
    sized.sort(key=lambda t: (-t[0].priority, -t[2], t[3]))

    result = PackResult(budget=effective_budget)
    remaining = effective_budget

    for item, tok, _density, _idx in sized:
        if remaining <= 0:
            result.dropped_ids.append(item.id)
            continue

        if tok <= remaining:
            result.items.append(
                PackedItem(
                    id=item.id,
                    text=item.text,
                    tokens=tok,
                    priority=item.priority,
                    truncated=False,
                    original_tokens=tok,
                )
            )
            remaining -= tok
            continue

        if item.truncatable and remaining >= item.min_tokens:
            trimmed_text = truncate_to_tokens(
                item.text,
                max_tokens=remaining,
                encoding=encoding,
                strategy=item.truncate_strategy,
            )
            trimmed_tok = count_tokens(trimmed_text, encoding=encoding)
            result.items.append(
                PackedItem(
                    id=item.id,
                    text=trimmed_text,
                    tokens=trimmed_tok,
                    priority=item.priority,
                    truncated=True,
                    original_tokens=tok,
                )
            )
            remaining -= trimmed_tok
        else:
            result.dropped_ids.append(item.id)

    result.total_tokens = effective_budget - remaining
    return result
