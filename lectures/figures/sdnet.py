"""SDNET2018 tiles, the ImageNet ResNet-18 that reads them, and Lecture 15's hand-made filter.

SDNET2018 (Maguire, Dorafshan & Thomas, 2018; CC BY 4.0) from ``scripts/fetch_sdnet2018.sh``:
230 photographs of bridge decks, walls, and pavements, cut into 56,092 tiles of 256 px, each
labelled cracked or not. ``scripts/derive_sdnet2018.py`` writes the tile index and the cached
ResNet-18 features; it is standalone, so ``load`` checks that the cache's recorded
preprocessing is the one ``preprocess`` applies here.

Also here, for Lectures 15 and 16: ImageNet's channel statistics, and the 3 x 3 "darker than its
neighbours" filter.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.v2 import functional as tvf

from common import DATA_DIR, require

RAW = DATA_DIR / "raw" / "sdnet2018" / "SDNET2018"
TILES_CSV = DATA_DIR / "derived" / "sdnet2018_tiles.csv"
FEATURES = DATA_DIR / "derived" / "sdnet2018_resnet18.npy"
SURFACES: tuple[str, ...] = ("deck", "wall", "pavement")
SPLIT: tuple[float, float] = (0.6, 0.8)  # cumulative share of each surface's photos: train, val
COLS: int = 18  # tiles per row of a photograph
SIZE: int = 224  # ResNet-18's input size; tiles are 256 px
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # what the weights expect
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
# The hand-made filter: the 8 neighbours minus the centre, so a pixel darker than its
# neighbours (a crack is a dark line) gives a positive response.
DARK_LINE: np.ndarray = np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]) / 8.0


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """The tile index and the cached ResNet-18 features, row for row.

    Returns:
        The index (one row per tile) and the features, shape (56,092, 512).

    Raises:
        ValueError: If the cache was made with other weights or preprocessing than this
            module's.
    """
    hint = "fetch_sdnet2018.sh; then derive_sdnet2018.py"
    df = pd.read_csv(require(TILES_CSV, hint))
    feats = np.load(require(FEATURES, hint))
    meta = json.loads(require(FEATURES.with_suffix(".json"), hint).read_text())
    want = {
        "weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "url": WEIGHTS.url,
        "preprocess": f"resize {SIZE}x{SIZE} bilinear antialias; "
        f"ImageNet mean {IMAGENET_MEAN} std {IMAGENET_STD}",
        "shape": [len(df), 512],
    }
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"feature cache does not match this preprocessing: {bad}")
    assert feats.shape == (len(df), 512), feats.shape
    return df, feats


def photo_split(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign whole photographs to train/val/test: 60/20/20% of each surface's photos.

    Args:
        df: The tile index.
        rng: Shuffles each surface's photos.

    Returns:
        ``"train"``, ``"val"``, or ``"test"`` for every tile.
    """
    part = np.empty(len(df), dtype=object)
    for surface in SURFACES:
        g = df[df["surface"] == surface]
        photos = np.array(sorted(g["photo"].unique()))
        rng.shuffle(photos)
        a, b = (round(f * len(photos)) for f in SPLIT)
        role = dict.fromkeys(photos[:a], "train") | dict.fromkeys(photos[a:b], "val")
        role |= dict.fromkeys(photos[b:], "test")
        part[g.index] = g["photo"].map(role).to_numpy()
    return part


def read_tile(path: str, gray: bool = False) -> np.ndarray:
    """One tile as an array.

    Args:
        path: The tile's path relative to the raw folder.
        gray: Convert to grey levels (ITU-R 601 luma), as floats 0-255.

    Returns:
        uint8 (256, 256, 3), or float (256, 256) if ``gray``.
    """
    im = Image.open(RAW / path)
    if gray:
        return np.array(im.convert("L"), dtype=np.float64)
    return np.array(im.convert("RGB"))  # a writable copy, so torch.from_numpy can share it


def mosaic(df: pd.DataFrame, photo: str, px: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Reassemble a photograph from its tiles, each shrunk to ``px`` pixels.

    A tile with a ``_2`` suffix fills its place only if no other tile does (13 of them are
    copies of the tile at the same place).

    Args:
        df: The tile index.
        photo: Photo id, e.g. ``"D7001"``.
        px: Pixels per tile in the mosaic.

    Returns:
        The image (rows * px, 18 * px, 3) and the tiles placed, with their index rows.
    """
    g = df[df["photo"] == photo].sort_values("dup").drop_duplicates(["row", "col"])
    rows = int(g["row"].max()) + 1
    img = np.full((rows * px, COLS * px, 3), 255, dtype=np.uint8)
    for r in g.itertuples():
        tile = Image.open(RAW / r.path).convert("RGB").resize((px, px), Image.Resampling.BOX)
        img[r.row * px : (r.row + 1) * px, r.col * px : (r.col + 1) * px] = np.asarray(tile)
    return img, g


def preprocess(img: np.ndarray) -> torch.Tensor:
    """A uint8 tile (H, W, 3) as ResNet-18's input, as ``scripts/derive_sdnet2018.py`` makes it.

    Args:
        img: uint8 RGB image.

    Returns:
        float32 tensor (3, 224, 224), resized (bilinear, antialiased) and normalized.
    """
    x = tvf.to_dtype(torch.from_numpy(img).permute(2, 0, 1), torch.float32, scale=True)
    x = tvf.resize(x, [SIZE, SIZE], antialias=True)
    return tvf.normalize(x, list(IMAGENET_MEAN), list(IMAGENET_STD))


def backbone() -> torch.nn.Module:
    """ImageNet ResNet-18 without its classifier: 512 pooled features per image (eval mode)."""
    model = resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    return model.eval()
