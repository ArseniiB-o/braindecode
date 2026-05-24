"""Report the number of ``assert`` statements in the source tree.

Used to track progress on the multi-PR ``assert`` → typed-exception
migration documented in ``ROADMAP.md`` (P1). ``assert`` statements are
stripped by ``python -O`` / ``PYTHONOPTIMIZE``, so input-validation
guards expressed as asserts silently disappear from optimised
deployments — this script gives reviewers a single number to watch
shrink over time.

By default it walks ``braindecode/`` and prints both a summary line
and a per-file breakdown sorted by count. Pass ``--threshold N`` to
exit non-zero when the count exceeds ``N``; that lets CI / pre-commit
fail loudly if a PR re-introduces ``assert`` guards.

Example::

    python scripts/report_asserts.py                       # human-readable
    python scripts/report_asserts.py --threshold 50        # CI gate
    python scripts/report_asserts.py --json                # machine-readable

The check is intentionally text-based (it uses :mod:`ast` to find
``Assert`` nodes, not regex) so it does not double-count multi-line
asserts and does not flag the literal string ``"assert"`` appearing
in docstrings or comments.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "braindecode"


def count_asserts(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))


def collect(root: Path) -> Counter:
    counts: Counter = Counter()
    for py_file in root.rglob("*.py"):
        n = count_asserts(py_file)
        if n:
            counts[str(py_file.relative_to(root.parent))] = n
    return counts


def render_text(counts: Counter) -> str:
    total = sum(counts.values())
    lines = [f"Total asserts in source: {total} across {len(counts)} files."]
    if counts:
        lines.append("")
        lines.append("Top files:")
        for path, n in counts.most_common(20):
            lines.append(f"  {n:>4}  {path}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory to walk (default: ``braindecode/`` next to this script).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Exit with code 1 when the total assert count is strictly greater "
        "than this value. Useful as a regression gate in CI.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON document instead of the text report.",
    )
    args = parser.parse_args()

    counts = collect(args.root)
    total = sum(counts.values())

    if args.json:
        payload = {
            "total": total,
            "files": dict(counts.most_common()),
            "root": str(args.root),
            "threshold": args.threshold,
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(counts))

    if args.threshold is not None and total > args.threshold:
        sys.stderr.write(
            f"::error::assert count {total} exceeds threshold {args.threshold}.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
