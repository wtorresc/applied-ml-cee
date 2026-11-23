"""Derive a sample of the CRAS lab point cloud and per-class statistics (CEE Lecture 22).

Reads ``data/raw/craslab/craslabannotated.zip`` (from ``scripts/fetch_craslab.sh``; never
modified or extracted to disk): one tab-separated ASCII file, ``CRASLAB_annotated.asc``, of
584,701,977 points with columns X, Y, Z (m), R, G, B, Intensity, Classification, after a header
line and a line with the point count. It is streamed once, in chunks, and writes to
``data/derived/``:

- ``craslab_sample.npz``: every point kept with probability ``KEEP`` (a seeded draw in file
  order): ``xyz`` float32 (n, 3), ``rgb`` uint8 (n, 3), ``intensity`` float32 (n,), ``label``
  uint8 (n,).
- ``craslab_stats.json``: over all points, the count per classification code. The file gives
  codes only; their names are the paper's Table 4, in ``lectures/figures/craslab.py``, which
  checks these counts against it. (Until 2026-11-20 it also held each code's bounding box, mean,
  and a Z histogram over -3 to 3 m that dropped class 0's points above 3 m; nothing read them.)

Each output is written to a ``.part`` file and renamed over the old one (``os.replace``), so a
reader never sees half a file. Colours and codes are checked to be integers in 0-255 before the
uint8 casts, which would otherwise wrap silently.

Usage: PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_craslab.py
(a few minutes: one pass over 26.8 GB of text)
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "craslab" / "craslabannotated.zip"
DERIVED = ROOT / "data" / "derived"
MEMBER = "CRASLAB_annotated.asc"
N_POINTS = 584_701_977  # the file's own count line
KEEP = 0.02  # share of points kept in the sample
SEED = 22
CHUNK = 5_000_000
COLUMNS = ["x", "y", "z", "r", "g", "b", "intensity", "label"]


def checked_codes(a: np.ndarray) -> np.ndarray:
    """A chunk's classification codes, once its codes and colours are known to fit in uint8.

    Args:
        a: A chunk of the file, float64, shape (n, 8), columns in ``COLUMNS``' order.

    Returns:
        The codes, int64, shape (n,).

    Raises:
        ValueError: If a code is not an integer, or a code or a colour is outside the integers
            0-255 (the sample stores both as uint8, and the cast would wrap or truncate).
    """
    lab = np.rint(a[:, 7]).astype(np.int64)
    if not np.allclose(a[:, 7], lab):
        raise ValueError("non-integer classification codes")
    for name, v in (("classification code", a[:, 7:8]), ("colour", a[:, 3:6])):
        if v.min() < 0 or v.max() > 255 or not np.array_equal(v, np.rint(v)):
            raise ValueError(f"a {name} is not an integer in 0-255: the uint8 cast would wrap")
    return lab


def main() -> None:
    """Stream the file once; write the sample and the statistics."""
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW}: run scripts/fetch_craslab.sh first")
    rng = np.random.default_rng(SEED)
    counts: dict[int, int] = {}
    kept: list[np.ndarray] = []
    n = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(RAW) as zf, zf.open(MEMBER) as raw:
        text = io.TextIOWrapper(raw, encoding="ascii")
        header = text.readline().strip()
        count = int(text.readline().strip())
        if not header.startswith("//X") or count != N_POINTS:
            raise ValueError(f"unexpected header {header!r} or count {count}")
        reader = pd.read_csv(
            text,
            sep="\t",
            header=None,
            names=COLUMNS,
            dtype=np.float64,
            chunksize=CHUNK,
            engine="c",
        )
        for chunk in reader:
            a = chunk.to_numpy()
            lab = checked_codes(a)
            for c, k in zip(*np.unique(lab, return_counts=True), strict=True):
                counts[int(c)] = counts.get(int(c), 0) + int(k)
            kept.append(a[rng.random(len(a)) < KEEP])
            n += len(a)
            print(f"  {n:,} points, {time.perf_counter() - t0:.0f} s", flush=True)
    if n != N_POINTS:
        raise ValueError(f"read {n:,} points, the file says {N_POINTS:,}")
    s = np.concatenate(kept)
    DERIVED.mkdir(parents=True, exist_ok=True)
    part = DERIVED / "craslab_sample.npz.part"
    with part.open("wb") as fh:
        np.savez(
            fh,
            xyz=s[:, :3].astype(np.float32),
            rgb=s[:, 3:6].astype(np.uint8),  # checked_codes: integers in 0-255
            intensity=s[:, 6].astype(np.float32),
            label=np.rint(s[:, 7]).astype(np.uint8),
        )
    part.replace(DERIVED / "craslab_sample.npz")  # os.replace: atomic within one file system
    out = {
        "points": n,
        "keep": KEEP,
        "seed": SEED,
        "sample": len(s),
        "classes": {str(c): {"count": k} for c, k in sorted(counts.items())},
    }
    if sum(counts.values()) != n:
        raise ValueError(f"per-code counts sum to {sum(counts.values()):,}, not {n:,}")
    part = DERIVED / "craslab_stats.json.part"
    part.write_text(json.dumps(out) + "\n")
    part.replace(DERIVED / "craslab_stats.json")
    print(f"{n:,} points, {len(s):,} kept, {len(counts)} codes, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
