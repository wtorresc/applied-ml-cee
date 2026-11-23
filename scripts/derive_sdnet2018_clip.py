"""Cache CLIP image embeddings of every SDNET2018 tile (CEE Lecture 18).

Reads the tile index ``data/derived/sdnet2018_tiles.csv`` and the tiles under
``data/raw/sdnet2018`` (from ``scripts/derive_sdnet2018.py`` and ``scripts/fetch_sdnet2018.sh``;
never modified) and writes, in ``data/derived/``:

- ``sdnet2018_clip_l14_336.npy``: float32, shape (56,092, 768), CLIP ViT-L/14 at 336 px
  (Radford et al., 2021; ``openai/clip-vit-large-patch14-336``, a pinned revision) image
  embeddings after the projection into the shared image-text space (not normalized), one row
  per tile in the index's order. Each tile gets CLIP's own preprocessing (its processor: resize
  the shorter side to 336, centre crop 336, CLIP's mean and std), as zero-shot classification
  compares these embeddings with text embeddings made the way CLIP was trained.
- ``sdnet2018_clip_l14_336.json``: the model, revision, preprocessing, and library versions.

The weights are OpenAI's (MIT-licensed repository; the model card scopes them to research use,
which suits a course).

Usage: PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_sdnet2018_clip.py
(about 20 minutes on one GPU; longer when the GPU is shared)
"""

from __future__ import annotations

import json
import os

# Read once, when huggingface_hub is first imported (by transformers): no usage reports.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
import transformers
from derive_sdnet2018 import DERIVED, RAW, TILES_CSV
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL = "openai/clip-vit-large-patch14-336"
REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
FEATURES = DERIVED / "sdnet2018_clip_l14_336.npy"
# What the cache holds, as its .json records it: transformers 5's get_image_features returns the
# vision tower's output with pooler_output replaced by visual_projection(pooled), 768 numbers.
LAYER = "get_image_features(...).pooler_output: the projected image embedding (768), not normalized"


class Tiles(Dataset):
    """Tiles as CLIP's processor prepares them (3, 336, 336)."""

    def __init__(self, paths: list[str], processor: transformers.CLIPImageProcessor) -> None:
        """Tiles to read, in order, and the processor to apply."""
        self.paths, self.processor = paths, processor

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """The i-th tile, preprocessed."""
        img = Image.open(RAW / self.paths[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def features(paths: list[str]) -> np.ndarray:
    """CLIP image embeddings for each tile, shape (len(paths), 768)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.CLIPModel.from_pretrained(MODEL, revision=REVISION).eval().to(device)
    processor = transformers.CLIPImageProcessor.from_pretrained(MODEL, revision=REVISION)
    loader = DataLoader(Tiles(paths, processor), batch_size=128, num_workers=12, shuffle=False)
    out = []
    with torch.inference_mode():
        for i, x in enumerate(loader):
            emb = model.get_image_features(
                pixel_values=x.to(device)
            ).pooler_output  # projected (n, 768)
            out.append(emb.float().cpu().numpy())
            if i % 80 == 0:
                print(f"  {i * 128:,} / {len(paths):,}", flush=True)
    feats = np.concatenate(out)
    assert feats.shape == (len(paths), 768), feats.shape
    return feats.astype(np.float32)


def main() -> None:
    """Write the CLIP embedding cache and its metadata."""
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
        "model": MODEL,
        "revision": REVISION,
        "preprocess": "CLIPImageProcessor: shorter side 336, centre crop 336, CLIP mean/std",
        "layer": LAYER,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "shape": list(feats.shape),
    }
    FEATURES.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{FEATURES.name}: {feats.shape}")


if __name__ == "__main__":
    main()
