"""Token counting utilities.

Uses tiktoken when it's installed (accurate, matches OpenAI-family models).
Falls back to a dependency-free heuristic otherwise, so llm-ctxpack never
requires a hard dependency just to estimate size.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised in envs without tiktoken
    _TIKTOKEN_AVAILABLE = False


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in ``text``.

    Uses tiktoken's ``encoding`` if the library is installed, otherwise
    falls back to a ~4-chars-per-token heuristic (a widely used, decent
    approximation for English text across most modern tokenizers).
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = _get_encoding(encoding)
            return len(enc.encode(text))
        except Exception:
            pass
    # Dependency-free fallback heuristic.
    return max(1, round(len(text) / 4))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding: str = "cl100k_base",
    strategy: str = "tail",
) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens.

    strategy:
      - "tail":   keep the beginning, cut the end (default; good for docs)
      - "head":   keep the end, cut the beginning (good for recent chat history)
      - "middle": keep both ends, cut the middle (good for long files where
                   the start and end carry the most signal, e.g. code files)
    """
    if max_tokens <= 0:
        return ""

    if _TIKTOKEN_AVAILABLE:
        try:
            enc = _get_encoding(encoding)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            if strategy == "head":
                return enc.decode(tokens[-max_tokens:])
            if strategy == "middle":
                half = max_tokens // 2
                return (
                    enc.decode(tokens[:half])
                    + "\n...\n"
                    + enc.decode(tokens[-(max_tokens - half):])
                )
            return enc.decode(tokens[:max_tokens])
        except Exception:
            pass

    # Fallback heuristic: ~4 chars/token.
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    if strategy == "head":
        return text[-max_chars:]
    if strategy == "middle":
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-(max_chars - half):]
    return text[:max_chars]
