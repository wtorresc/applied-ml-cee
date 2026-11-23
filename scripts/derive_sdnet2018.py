"""Index SDNET2018's tiles and cache their ImageNet ResNet-18 features (CEE Lecture 15).

Reads ``data/raw/sdnet2018/SDNET2018`` (from ``scripts/fetch_sdnet2018.sh``; never modified) and
writes, in ``data/derived/``:

- ``sdnet2018_tiles.csv``: one row per tile, in a fixed order (surface, label folder, file name):
  ``path`` (relative to the raw folder), ``surface`` (deck, wall, pavement), ``cracked`` (0/1),
  ``photo`` (surface letter + photo number, e.g. ``D7001``), ``tile`` (1-based index in the
  photo), ``dup`` (1 for the 50 tiles with a ``_2`` suffix), and ``row``, ``col`` (the tile's
  place in its photo: the tiles of a photo run row by row, 18 to a row; checked by reassembling
  photos of all three surfaces). Of the 50 ``_2`` tiles, 13 have the same pixels as the tile
  with the same index (their JPEG files differ), two of them filed under the other label; the
  other 37 fill places that have no other tile.
- ``sdnet2018_resnet18.npy``: float32, shape (56,092, 512), the global-average-pooled output of
  torchvision's ResNet-18 with ImageNet weights (``IMAGENET1K_V1``; He et al., 2016), one row per
  tile in the CSV's order. Each 256 x 256 tile is resized to 224 x 224 (bilinear, antialiased,
  the resolution the weights were trained at) and normalized with ImageNet's mean and std.
- ``sdnet2018_resnet18.json``: the weights, preprocessing, and library versions.

The weights' licence (BSD-3 code, ImageNet terms for research and education) suits this course
(candidate, 2026-11-16). On the 28-core CPU this takes about 6 minutes at 14 threads.

Usage: PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_sdnet2018.py [--threads 14]
(string hashing is fixed only when PYTHONHASHSEED is set before Python starts; nothing
here depends on it, but the course's scripts all run that way)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.v2 import functional as tvf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sdnet2018" / "SDNET2018"
DERIVED = ROOT / "data" / "derived"
TILES_CSV = DERIVED / "sdnet2018_tiles.csv"
FEATURES = DERIVED / "sdnet2018_resnet18.npy"
SURFACES = {"D": "deck", "W": "wall", "P": "pavement"}
COLS = 18  # tiles per row of a source photograph (4,608 px / 256 px)
# Tile counts per folder, as extracted by fetch_sdnet2018.sh (cracked, uncracked).
COUNTS = {"D": (2025, 11595), "P": (2608, 21726), "W": (3851, 14287)}
SIZE = 224
MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics the weights expect
STD = (0.229, 0.224, 0.225)


def tile_index() -> pd.DataFrame:
    """List every tile with its surface, label, photo, and place in the photo.

    Returns:
        One row per tile, in a fixed order.

    Raises:
        ValueError: If a folder's count or a file name differs from what the lectures expect.
    """
    rows = []
    for letter, surface in SURFACES.items():
        for cracked, prefix in ((1, "C"), (0, "U")):
            folder = RAW / letter / f"{prefix}{letter}"
            files = sorted(folder.glob("*.jpg"))
            if len(files) != COUNTS[letter][1 - cracked]:
                raise ValueError(f"{folder}: {len(files)} tiles, expected {COUNTS[letter]}")
            for f in files:
                photo, rest = f.stem.split("-")
                tile, _, suffix = rest.partition("_")
                if suffix not in ("", "2") or not (photo.isdigit() and tile.isdigit()):
                    raise ValueError(f"unexpected tile name {f.name}")
                k = int(tile) - 1
                rows.append(
                    {
                        "path": str(f.relative_to(RAW)),
                        "surface": surface,
                        "cracked": cracked,
                        "photo": f"{letter}{photo}",
                        "tile": int(tile),
                        "dup": int(suffix == "2"),
                        "row": k // COLS,
                        "col": k % COLS,
                    }
                )
    df = pd.DataFrame(rows)
    n_photos = df.groupby("surface")["photo"].nunique().to_dict()
    if len(df) != 56092 or n_photos != {"deck": 54, "pavement": 104, "wall": 72}:
        raise ValueError(f"{len(df)} tiles in {n_photos} photos; expected 56,092 in 54/104/72")
    if int(df["dup"].sum()) != 50:
        raise ValueError(f"{df['dup'].sum()} tiles with a _2 suffix, expected 50")
    return df


class Tiles(Dataset):
    """Decoded, resized, normalized tiles (C, 224, 224), float32."""

    def __init__(self, paths: list[str]) -> None:
        """Tiles to read, in order.

        Args:
            paths: Tile paths relative to the raw folder.
        """
        self.paths = paths

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """The i-th tile, ready for ResNet-18."""
        img = decode_jpeg(read_file(str(RAW / self.paths[i])), mode=ImageReadMode.RGB)
        return preprocess(img)


def preprocess(img: torch.Tensor) -> torch.Tensor:
    """Resize a uint8 tile (3, H, W) to 224 x 224 and normalize it with ImageNet's statistics.

    Args:
        img: uint8 image, shape (3, H, W).

    Returns:
        float32 tensor, shape (3, 224, 224).
    """
    x = tvf.to_dtype(img, torch.float32, scale=True)
    x = tvf.resize(x, [SIZE, SIZE], antialias=True)
    return tvf.normalize(x, MEAN, STD)


def features(paths: list[str], threads: int) -> np.ndarray:
    """ResNet-18 (ImageNet) pooled features for each tile.

    Args:
        paths: Tile paths relative to the raw folder.
        threads: PyTorch threads.

    Returns:
        float32 array, shape (len(paths), 512).
    """
    torch.set_num_threads(threads)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # keep the 512 pooled features, drop ImageNet's classifier
    model = model.eval().to(memory_format=torch.channels_last)  # ~1.8x faster on this CPU
    loader = DataLoader(Tiles(paths), batch_size=128, num_workers=6, shuffle=False)
    out = []
    with torch.inference_mode():
        for i, x in enumerate(loader):
            out.append(model(x.to(memory_format=torch.channels_last)).numpy())
            if i % 50 == 0:
                print(f"  {i * 128:,} / {len(paths):,}", flush=True)
    feats = np.concatenate(out)
    assert feats.shape == (len(paths), 512), feats.shape
    return feats.astype(np.float32)


def main() -> None:
    """Write the tile index and the feature cache."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=14)
    threads = parser.parse_args().threads
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 (the global generator, for any library that draws from it)
    torch.manual_seed(0)
    DERIVED.mkdir(parents=True, exist_ok=True)

    df = tile_index()
    part = TILES_CSV.with_suffix(".csv.part")
    df.to_csv(part, index=False)
    part.rename(TILES_CSV)
    print(f"{TILES_CSV.name}: {len(df):,} tiles, {df['photo'].nunique()} photos")

    feats = features(df["path"].tolist(), threads)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite features")
    part = FEATURES.with_suffix(".npy.part")
    with part.open("wb") as fh:
        np.save(fh, feats)
    part.rename(FEATURES)
    meta = {
        "weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "url": ResNet18_Weights.IMAGENET1K_V1.url,
        "preprocess": f"resize {SIZE}x{SIZE} bilinear antialias; ImageNet mean {MEAN} std {STD}",
        "layer": "global average pool (fc replaced by identity)",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "shape": list(feats.shape),
    }
    FEATURES.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{FEATURES.name}: {feats.shape}")


if __name__ == "__main__":
    main()
