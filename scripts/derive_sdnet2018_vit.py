"""Cache ImageNet ViT-B/16 features of every SDNET2018 tile (CEE Lecture 17).

Reads the tile index ``data/derived/sdnet2018_tiles.csv`` and the tiles under
``data/raw/sdnet2018`` (both from ``scripts/derive_sdnet2018.py`` and
``scripts/fetch_sdnet2018.sh``; never modified) and writes, in ``data/derived/``:

- ``sdnet2018_vit_b_16.npy``: float32, shape (56,092, 768), the class token of torchvision's
  ViT-B/16 with ImageNet weights (``IMAGENET1K_V1``; Dosovitskiy et al., 2021) after its final
  layer norm, one row per tile in the index's order. Each tile gets exactly the preprocessing
  Lecture 15 gave ResNet-18 (resize 256 -> 224, bilinear, antialiased; ImageNet mean and std), so
  the two backbones are compared on the same inputs.
- ``sdnet2018_vit_b_16.json``: the weights, preprocessing, and library versions.

The weights' terms are those of ResNet-18's (BSD-3 code; ImageNet weights for research and
education), which suit this course (candidate, 2026-11-16 and 2026-11-18).

Usage: PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_sdnet2018_vit.py
(on the GPU, a few minutes; decoding the JPEGs dominates)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torchvision
from derive_sdnet2018 import DERIVED, MEAN, SIZE, STD, TILES_CSV, Tiles
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_16_Weights, vit_b_16

FEATURES = DERIVED / "sdnet2018_vit_b_16.npy"
WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1


def features(paths: list[str]) -> np.ndarray:
    """ViT-B/16 class-token features for each tile, shape (len(paths), 768)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=WEIGHTS)
    model.heads = torch.nn.Identity()  # keep the 768-d class token, drop ImageNet's classifier
    model = model.eval().to(device)
    loader = DataLoader(Tiles(paths), batch_size=256, num_workers=12, shuffle=False)
    out = []
    with torch.inference_mode():
        for i, x in enumerate(loader):
            out.append(model(x.to(device)).float().cpu().numpy())
            if i % 40 == 0:
                print(f"  {i * 256:,} / {len(paths):,}", flush=True)
    feats = np.concatenate(out)
    assert feats.shape == (len(paths), 768), feats.shape
    return feats.astype(np.float32)


def main() -> None:
    """Write the ViT feature cache and its metadata."""
    torch.manual_seed(0)
    df = pd.read_csv(TILES_CSV)
    feats = features(df["path"].tolist())
    if not np.isfinite(feats).all():
        raise ValueError("non-finite features")
    part = FEATURES.with_suffix(".npy.part")
    with part.open("wb") as fh:
        np.save(fh, feats)
    part.rename(FEATURES)
    meta = {
        "weights": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
        "url": WEIGHTS.url,
        "preprocess": f"resize {SIZE}x{SIZE} bilinear antialias; ImageNet mean {MEAN} std {STD}",
        "layer": "class token after the encoder's final layer norm (heads replaced by identity)",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "shape": list(feats.shape),
    }
    FEATURES.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{FEATURES.name}: {feats.shape}")


if __name__ == "__main__":
    main()
