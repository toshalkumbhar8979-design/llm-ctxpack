"""Command-line interface: pack a list of files into a token budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .packer import ContextItem, pack


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-ctxpack",
        description="Pack files into a fixed LLM context-window token budget, "
        "keeping the highest-priority content and trimming the rest.",
    )
    parser.add_argument("files", nargs="+", help="Text files to pack, highest priority first.")
    parser.add_argument(
        "--budget", "-b", type=int, required=True, help="Token budget for the packed context."
    )
    parser.add_argument(
        "--reserve",
        type=int,
        default=0,
        help="Tokens to reserve off the top (e.g. for a system prompt).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=40,
        help="Minimum tokens a file must get to be included when truncated.",
    )
    parser.add_argument(
        "--strategy",
        choices=["head", "tail", "middle"],
        default="tail",
        help="Which part of an over-budget file to keep.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Never trim files; include them whole or drop them entirely.",
    )
    parser.add_argument(
        "-o", "--output", help="Write packed context to this file instead of stdout."
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print packing stats (tokens used, dropped/truncated files) to stderr.",
    )
    args = parser.parse_args(argv)

    items = []
    n = len(args.files)
    for i, path_str in enumerate(args.files):
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"llm-ctxpack: error reading {path}: {e}", file=sys.stderr)
            return 1
        # Earlier files on the command line get higher priority.
        priority = n - i
        items.append(
            ContextItem(
                id=str(path),
                text=text,
                priority=priority,
                truncatable=not args.no_truncate,
                min_tokens=args.min_tokens,
                truncate_strategy=args.strategy,
            )
        )

    result = pack(items, budget=args.budget, reserve_tokens=args.reserve)
    rendered = result.render()

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    if args.stats:
        print(
            f"\n[llm-ctxpack] {result.total_tokens}/{result.budget} tokens used "
            f"({len(result.items)} included, {len(result.dropped_ids)} dropped)",
            file=sys.stderr,
        )
        for pi in result.items:
            flag = " (truncated)" if pi.truncated else ""
            print(f"  + {pi.id}: {pi.tokens} tok{flag}", file=sys.stderr)
        for did in result.dropped_ids:
            print(f"  - {did}: dropped", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
