"""Compare two numbers.json files from runs of the same figure script.

A figure script whose numbers feed a deck is run twice and the two numbers.json are
diffed before the deck is trusted; ``make verify-<stem>`` reruns a script into
``build/verify/<stem>/`` and calls this tool against ``build/fig/<stem>/numbers.json``.

- Numbers are equal when they differ by at most 1e-12, relative to the larger magnitude, or
  absolutely near zero: ``|a - b| <= 1e-12 * max(1, |a|, |b|)``. An int and a float are a
  difference even at the same value (12 against 12.0: a count turned into a float, as a pandas
  NaN upcast does). Strings, booleans and nulls must be identical.
- Timings are ignored: a key named in ``TIMING_KEYS``, with everything under it, at any depth,
  except at the paths of ``DATA_PATHS``, which hold data under a timing's name (L20's window
  length ``fourier.seconds``, L28's GPU ledger, whose minutes are the other lectures' recorded
  run times). The last line says how many values were ignored, under which key names.
  A new timing key must be added to ``TIMING_KEYS``: until it is, it shows as a difference.
- A key or list element present on one side only is reported as one-sided.

Usage: python tools/diff_numbers.py OLD.json NEW.json
(prints each difference; exit 1 if there is any, 0 if none)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TypeGuard

TOL = 1e-12
# Wall-clock timings, which change from run to run, by key name (the lectures that use each).
TIMING_KEYS = frozenset(
    {
        "minutes",  # every script's run time; L15-L16, L25 stages
        "seconds",  # L19-L23 stages
        "embed_minutes",  # L15
        "resnet18_deck_seconds",  # L17
        "vit_b_16_deck_seconds",  # L17
        "qwen_scoring_seconds",  # L19
        "pooled_seconds",  # L20
        "fd_seconds_max",  # L23
        "seconds_ratio_to_fd",  # L23: a ratio of two timings
        "time_ratio_to_fd",  # L23's name for it until 2026-11-20 (in older numbers.json)
        "train_seconds",  # L24, L26
        "test_seconds",  # L24
        "seconds_per_episode",  # L25
        "bridge_seconds",  # L26
    }
)
# Data stored under a timing's name: compared. Shell-style patterns on the dotted key path.
DATA_PATHS = (
    "fourier.seconds",  # L20: the Fourier window, 4 s of the bridge record
    "ledger.*",  # L28: the GPU ledger, from the other lectures' numbers.json
)


def ignored(path: str, key: str) -> bool:
    """Whether a key is a timing (``TIMING_KEYS``) at a path that holds no data (``DATA_PATHS``).

    Args:
        path: The key's dotted path, e.g. ``training.MLP.seconds``.
        key: The key itself, e.g. ``seconds``.

    Returns:
        True when the key and everything under it are skipped.
    """
    return key in TIMING_KEYS and not any(fnmatchcase(path, p) for p in DATA_PATHS)


def same_number(a: float, b: float) -> bool:
    """Equal within ``TOL``, relative to the larger magnitude (absolute below 1)."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def is_number(x: object) -> TypeGuard[int | float]:
    """An int or float that is not a bool."""
    return isinstance(x, int | float) and not isinstance(x, bool)


def diff(old: object, new: object, path: str, skipped: Counter[str]) -> list[str]:
    """Every difference between two JSON values, as one line each.

    Args:
        old: The reference value.
        new: The value from the new run.
        path: Where these values sit, e.g. ``results.mlp[2]`` ("" at the top).
        skipped: Counts the timing keys left out, by key name (updated in place).

    Returns:
        Lines naming the path and both values; empty when the two agree.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in [*old, *(k for k in new if k not in old)]:
            sub = f"{path}.{k}" if path else k
            if ignored(sub, k):
                skipped[k] += 1
            elif k not in new:
                out.append(f"{sub}: only in OLD ({json.dumps(old[k])[:80]})")
            elif k not in old:
                out.append(f"{sub}: only in NEW ({json.dumps(new[k])[:80]})")
            else:
                out += diff(old[k], new[k], sub, skipped)
        return out
    if isinstance(old, list) and isinstance(new, list):
        pairs = enumerate(zip(old, new, strict=False))  # the longer list's tail is reported below
        out = [line for i, (a, b) in pairs for line in diff(a, b, f"{path}[{i}]", skipped)]
        for i in range(len(new), len(old)):
            out.append(f"{path}[{i}]: only in OLD ({json.dumps(old[i])[:80]})")
        for i in range(len(old), len(new)):
            out.append(f"{path}[{i}]: only in NEW ({json.dumps(new[i])[:80]})")
        return out
    return diff_leaf(old, new, path or "(top)")


def diff_leaf(old: object, new: object, where: str) -> list[str]:
    """The difference between two JSON values that are not both dicts or both lists.

    Args:
        old: The reference value.
        new: The value from the new run.
        where: Their path, for the message.

    Returns:
        One line, or none when the two agree: numbers within ``TOL`` and of one type (int or
        float), anything else identical.
    """
    if is_number(old) and is_number(new):
        retyped = type(old) is not type(new)
        if same_number(float(old), float(new)) and not retyped:
            return []
        kind = f", {type(old).__name__} -> {type(new).__name__}" if retyped else ""
        return [f"{where}: {old!r} -> {new!r} (difference {float(new) - float(old):+.3g}{kind})"]
    if type(old) is type(new) and old == new:
        return []
    return [f"{where}: {json.dumps(old)[:80]} -> {json.dumps(new)[:80]}"]


def main() -> int:
    """Diff the two files named on the command line."""
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    skipped: Counter[str] = Counter()
    lines = diff(old, new, "", skipped)
    for line in lines:
        print(f"  {line}")
    verdict = f"{len(lines)} differences" if lines else "identical (timings aside)"
    print(f"{old_path} vs {new_path}: {verdict}")
    if skipped:
        keys = ", ".join(f"{k} {n}" for k, n in sorted(skipped.items()))
        print(f"  timings ignored, by key: {keys}")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
