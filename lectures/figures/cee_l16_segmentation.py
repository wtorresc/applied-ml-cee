"""Figures and numbers for CEE Lecture 16: segmentation, detection, and measuring from pixels.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l16_segmentation.py --out build/fig/cee_l16_segmentation

Writes the PNG figures and ``numbers.json`` (the numbers the slides quote from this lecture's
data and runs; constants, citations, and other lectures' numbers are not in it). Data:

- 3DCrack (Zhang, Wang, Hsieh, Yang, Yezzi & Tsai, 2025; doi:10.21227/2etq-xh36; CC BY 4.0),
  read by ``crack3d.py``: 1,634 range images of Georgia pavements from a 3D line-laser survey
  system, 512 x 624 px at about 4 mm per pixel, with crack masks and crack centrelines that
  carry two widths per point (one from the mask, one from the range image).
- SpaceNet 8, Germany (Hänsch et al., 2022; CC BY-SA 4.0, Maxar imagery under a non-commercial
  licence), read by ``spacenet8.py``: satellite tiles before and after the July 2021 flood, with
  building footprints and roads marked flooded or not.

The setting: a pavement survey van scans a lane; the agency rates cracking by the LTPP
Distress Identification Manual (FHWA-HRT-13-092), which records the length of cracking at each
severity level, and the severity of a longitudinal or transverse crack in asphalt by its mean
width (low up to 6 mm, moderate up to 19 mm, high above). Lecture 15's classifier said whether a
tile holds a crack; here a U-Net labels every pixel, and the mask is turned into lengths and
widths in millimetres. The same network then maps flooded buildings and roads from space, where
buildings must be counted one by one (detection) and pixels in degrees must become metres.

Training runs on the GPU when there is one (about 8 min per U-Net on an RTX 3090 Ti; the whole
script about 50 min, recorded as ``minutes``) and on the CPU otherwise: many hours (the notebook's
header gives a measured estimate), and the numbers then differ slightly.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from scipy.ndimage import correlate, distance_transform_edt, maximum_filter
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.models import resnet18
from torchvision.ops import box_iou

import aml_style
import crack3d
import spacenet8
import title_art
from common import parse_out_dir, seed_all, write_numbers
from crack3d import MASK_CUT, MM_PER_PX, SEVERITIES, SPLITS, Branch, H, Split, W
from sdnet import DARK_LINE, IMAGENET_MEAN, IMAGENET_STD, WEIGHTS
from spacenet8 import Tile

SEED: Final[int] = 0
TOL: Final[int] = 2  # px; the tolerance of the "within 2 px" F1 (8 mm)
BASE: Final[int] = 16  # channels at the U-Net's first level; doubled at each of 4 halvings
# Passes over the training images. The one-cycle rate falls to zero at the last pass, and the
# validation F1 levels off as it does; a longer schedule was not tried.
EPOCHS: Final[int] = 40
BATCH: Final[int] = 8
LR: Final[float] = 2e-3  # peak of a one-cycle schedule (Smith & Topin, 2019), AdamW
WD: Final[float] = 1e-4
THRESHOLDS: Final[np.ndarray] = np.round(np.arange(0.05, 0.951, 0.05), 2)  # tried on validation
BOOT: Final[int] = 2000  # image-level bootstrap resamples for the test scores' 95% intervals
HUTHWOHL_MM: Final[float] = 0.1  # mm per px to see a 0.3 mm crack at 3 px (Hüthwohl 2019)
PHONE_F_MM: Final[float] = 4.2  # the example phone camera: focal length
PHONE_PITCH_MM: Final[float] = 1.4e-3  # and pixel pitch (1.4 µm)
WINDOW_PX: Final[int] = 64  # the sliding-window classifier's window
CROP_PX: Final[int] = 160  # the crop the skips and Dice slides zoom into
TWIN_DELTA_MAX: Final[float] = 0.005  # leaving out the test twin may move the U-Net's F1 by less
# numbers.json keys (slugs) of the crack U-Net and its two variants, and their display labels
RUN_LABELS: Final[dict[str, str]] = {"unet": "U-Net", "no_skips": "no skips", "dice": "+ Dice loss"}
# Overlay colours: nothing, found (TP), false (FP), missed (FN)
CMAP_TFN: Final[ListedColormap] = ListedColormap(["#00000000", "#009E73", "#D55E00", "#0072B2"])
# Test images shown on the slides (chosen from a contact sheet of the 250 test images).
SHOW: Final[tuple[str, ...]] = ("sample_000047", "sample_000184", "sample_000257", "sample_000201")
ATTEMPT_STEM: Final[str] = "sample_000209"  # a crack beside a dark lane edge
UNET_STEM: Final[str] = "sample_000236"
EDT_STEM: Final[str] = "sample_000091"
EDT_CROP: Final[int] = 40  # px around the widest point of that crack
TITLE_STEM: Final[str] = "sample_000257"  # an alligator-cracked patch
LTPP_STEM: Final[str] = "sample_000609"  # longitudinal and transverse cracks, all three severities
# SpaceNet 8
SN8_COLORS: Final[ListedColormap] = ListedColormap(
    ["#00000000", "#F0E442", "#0072B2", "#E69F00", "#CC79A7"]
)
SN8_CROP: Final[int] = 512  # training crops
SN8_CROPS: Final[int] = 4  # random crops per training tile per pass
SN8_EPOCHS: Final[int] = 60
SN8_LR: Final[float] = 1e-3  # one-cycle peak; lower than from scratch, the encoder is pretrained
SN8_RECALL: Final[float] = 0.8  # the responder's target: flag 8 of 10 flooded buildings
SN8_SHOW: Final[str] = "0_41_58"  # the town on the river in the test block
SN8_BOX: Final[tuple[int, int, int]] = (300, 300, 400)  # crop for the detection and merge panels


def device() -> torch.device:
    """The GPU when there is one, else the CPU (students without a GPU: same code, slower)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# Metrics


def counts(b: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    """True positives, false positives, false negatives over all pixels."""
    return int((b & y).sum()), int((b & ~y).sum()), int((~b & y).sum())


@dataclass
class Counts:
    """Per-image pixel counts behind the strict and the tolerant scores (for the bootstrap)."""

    tp: np.ndarray  # (N,) true positives
    fp: np.ndarray  # (N,) false positives
    fn: np.ndarray  # (N,) false negatives
    hit_b: np.ndarray  # (N,) predicted pixels within ``TOL`` px of a true one
    hit_y: np.ndarray  # (N,) true pixels within ``TOL`` px of a predicted one
    nb: np.ndarray  # (N,) predicted pixels
    ny: np.ndarray  # (N,) true pixels

    def pooled(self, idx: np.ndarray | None = None) -> dict[str, float]:
        """Strict and tolerant scores over the images ``idx`` (all by default), pooled.

        Args:
            idx: Image indices; repeats allowed (a bootstrap resample).

        Returns:
            precision, recall, f1, iou (strict), and tol_precision, tol_recall, tol_f1.
        """
        sel = slice(None) if idx is None else idx
        tp, fp, fn = (float(v[sel].sum()) for v in (self.tp, self.fp, self.fn))
        tol_p = float(self.hit_b[sel].sum()) / max(float(self.nb[sel].sum()), 1.0)
        tol_r = float(self.hit_y[sel].sum()) / max(float(self.ny[sel].sum()), 1.0)
        return {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "tol_precision": tol_p,
            "tol_recall": tol_r,
            "tol_f1": 2 * tol_p * tol_r / max(tol_p + tol_r, 1e-12),
        }


def image_counts(b: np.ndarray, y: np.ndarray, tol: int = TOL) -> Counts:
    """Count, image by image, what the strict and the tolerant scores pool.

    Thin structures are drawn a pixel to either side by different annotators (``twins``), so
    crack benchmarks also score with a tolerance: a pixel within ``tol`` px of the other mask
    counts as a hit.

    Args:
        b: Predicted masks, (N, H, W) bool.
        y: True masks, (N, H, W) bool.
        tol: Tolerance in pixels.

    Returns:
        The counts.
    """
    names = ("tp", "fp", "fn", "hit_b", "hit_y", "nb", "ny")
    arr = {k: np.zeros(len(b), np.int64) for k in names}
    for i, (bi, yi) in enumerate(zip(b, y, strict=True)):
        arr["tp"][i], arr["fp"][i], arr["fn"][i] = counts(bi, yi)
        arr["nb"][i], arr["ny"][i] = int(bi.sum()), int(yi.sum())
        if bi.any() and yi.any():
            arr["hit_b"][i] = int((distance_transform_edt(~yi)[bi] <= tol).sum())
            arr["hit_y"][i] = int((distance_transform_edt(~bi)[yi] <= tol).sum())
    return Counts(**arr)


def tolerant(b: np.ndarray, y: np.ndarray, tol: int = TOL) -> tuple[float, float]:
    """Precision and recall when a pixel within ``tol`` px of the other mask counts as a hit."""
    s = image_counts(b, y, tol).pooled()
    return s["tol_precision"], s["tol_recall"]


def scores(b: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Strict pixel F1 and IoU, and the F1 within ``TOL`` px, pooled over the images."""
    return image_counts(b, y).pooled()


def check_interior(grid: np.ndarray | range, value: float, name: str) -> None:
    """Raise unless a value chosen on a grid sits strictly inside it.

    Args:
        grid: The values tried, in increasing order.
        value: The value kept.
        name: What was chosen (for the message).

    Raises:
        ValueError: If ``value`` is the grid's first or last value.
    """
    lo, hi = float(grid[0]), float(grid[-1])
    if not lo < float(value) < hi:
        raise ValueError(f"{name}: kept {value} at the edge of its grid [{lo}, {hi}]")


def resamples(n: int) -> np.ndarray:
    """``BOOT`` image-level bootstrap resamples of n images, the same for every model."""
    return np.random.default_rng((SEED, 16, 9)).integers(0, n, size=(BOOT, n))


def interval(c: Counts, idx: np.ndarray) -> dict[str, list[float]]:
    """95% bootstrap intervals (percentile) of the pooled F1, IoU, and F1 within 2 px.

    Args:
        c: Per-image counts.
        idx: ``resamples``.

    Returns:
        [low, high] for f1, iou, and tol_f1.
    """
    vals = [c.pooled(i) for i in idx]
    return {k: [float(np.percentile([v[k] for v in vals], q)) for q in (2.5, 97.5)]
            for k in ("f1", "iou", "tol_f1")}  # fmt: skip


def paired(a: Counts, b: Counts, idx: np.ndarray, key: str = "f1") -> dict[str, float]:
    """The difference score(a) - score(b) on the same test images, with its 95% interval.

    Args:
        a: Per-image counts of one model.
        b: Per-image counts of the other, on the same images.
        idx: ``resamples``.
        key: The score.

    Returns:
        The difference and its interval.
    """
    d = [a.pooled(i)[key] - b.pooled(i)[key] for i in idx]
    lo, hi = (float(np.percentile(d, q)) for q in (2.5, 97.5))
    return {"diff": a.pooled()[key] - b.pooled()[key], "low": lo, "high": hi}


def f1_by_threshold(p: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Strict F1 and IoU at every threshold in ``THRESHOLDS``, computed where ``p`` lives.

    Args:
        p: Probabilities, any shape.
        y: True mask, same shape, bool.

    Returns:
        (f1, iou), one value per threshold.
    """
    f1, iou = [], []
    for t in THRESHOLDS:
        b = p > float(t)
        tp = (b & y).sum().item()
        fp = (b & ~y).sum().item()
        fn = (~b & y).sum().item()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        iou.append(tp / max(tp + fp + fn, 1))
    return np.array(f1), np.array(iou)


# ---------------------------------------------------------------------------------------------
# The U-Net


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3 x 3 convolutions, each followed by batch normalization and a ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
    )  # fmt: skip


class UNet(nn.Module):
    """A U-Net (Ronneberger, Fischer & Brox, 2015).

    It halves the map four times, then doubles it back with transposed convolutions, joining
    each level to its encoder twin (the skips).

    Args:
        c_in: Input channels (1 for a range image; 6 for a before-and-after satellite pair).
        c_out: Output channels: 1 logit per pixel, or one per class.
        base: Channels at the first level; doubled at each halving.
        skips: Whether the decoder sees the encoder's maps; False gives a plain
            encoder-decoder.
    """

    def __init__(
        self, c_in: int = 1, c_out: int = 1, base: int = BASE, *, skips: bool = True
    ) -> None:
        """Build the five encoder levels, four decoder levels, and the 1 x 1 head."""
        super().__init__()
        ch = [base * 2**i for i in range(5)]  # 16, 32, 64, 128, 256
        self.skips = skips
        self.down = nn.ModuleList(
            [conv_block(c_in, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(4)]
        )
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))]
        )
        self.dec = nn.ModuleList(
            [conv_block(2 * ch[i] if skips else ch[i], ch[i]) for i in reversed(range(4))]
        )
        self.head = nn.Conv2d(ch[0], c_out, 1)

    def forward(self, x: torch.Tensor, keep: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Map an image batch (B, c_in, H, W), H and W divisible by 16, to logits (B, c_out, H, W).

        Args:
            x: The input batch.
            keep: If given, every level's map is appended to it (encoder, then decoder).

        Returns:
            The logits.
        """
        feats = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        if keep is not None:
            keep.extend(feats)
        x = feats.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            x = up(x)
            skip = feats.pop()
            assert x.shape == skip.shape, (x.shape, skip.shape)
            x = dec(torch.cat([skip, x], dim=1) if self.skips else x)
            if keep is not None:
                keep.append(x)
        return self.head(x)


def normalize(u8: torch.Tensor) -> torch.Tensor:
    """Range image bytes to roughly zero mean, unit spread (the images centre on 126)."""
    return (u8.float() / 255.0 - 0.5) / 0.25


@dataclass
class Run:
    """A trained U-Net and its training record."""

    net: UNet
    loss: list[float]  # mean training loss per pass
    val_f1: list[float]  # best strict validation F1 over THRESHOLDS, per pass
    val_iou: list[float]
    threshold: float  # chosen on validation after the last pass
    seconds: float


def dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss, 1 - 2 sum(p y) / (sum p + sum y), over the whole batch (+1 smoothing)."""
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def predict(net: UNet, x: np.ndarray) -> np.ndarray:
    """Crack probabilities for range images.

    Args:
        net: A trained U-Net.
        x: (N, h, w) uint8 images, h and w divisible by 16.

    Returns:
        (N, h, w) float32 probabilities.
    """
    dev = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
        for i in range(0, len(x), 16):
            xb = normalize(torch.from_numpy(x[i : i + 16]).to(dev)[:, None])
            out.append(torch.sigmoid(net(xb).float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_unet(
    xt: np.ndarray, yt: np.ndarray, xv: np.ndarray, yv: np.ndarray, *,
    skips: bool = True, loss: str = "bce", epochs: int = EPOCHS, name: str = "",
) -> Run:  # fmt: skip
    """Train a U-Net on range images with random flips; pick the threshold on validation.

    Args:
        xt: Training images, (N, h, w) uint8.
        yt: Training masks, (N, h, w) bool.
        xv: Validation images.
        yv: Validation masks.
        skips: U-Net (True) or plain encoder-decoder (False).
        loss: ``"bce"`` (cross-entropy at every pixel; kept after a one-seed near-tie on
            validation, see ``crack_runs``) or ``"bce+dice"`` (plus the soft Dice loss, a common
            recipe for thin structures).
        epochs: Passes over the training images.
        name: Name printed with the progress lines.

    Returns:
        The trained network and its record.

    Raises:
        ValueError: If the threshold kept after the last pass is at an end of ``THRESHOLDS``.
    """
    dev = device()
    torch.manual_seed(SEED)
    net = UNet(skips=skips).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = epochs * math.ceil(len(xt) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(SEED)  # batches and flips, on the CPU on every device
    X, Y = torch.from_numpy(xt).to(dev)[:, None], torch.from_numpy(yt).to(dev)[:, None]
    yv_t = torch.from_numpy(yv).to(dev)
    run = Run(net, [], [], [], 0.5, 0.0)
    t0 = time.perf_counter()
    for epoch in range(epochs):
        net.train()
        perm, total = torch.randperm(len(X), generator=gen), 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i : i + BATCH].to(dev)
            xb, yb = normalize(X[idx]), Y[idx].float()
            flip = torch.rand(2, generator=gen)  # mirror across and along the lane
            if flip[0] < 0.5:
                xb, yb = xb.flip(3), yb.flip(3)
            if flip[1] < 0.5:
                xb, yb = xb.flip(2), yb.flip(2)
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                logits = net(xb).float()
            value = F.binary_cross_entropy_with_logits(logits, yb)
            if loss == "bce+dice":
                value = value + dice_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
            sched.step()
            total += value.item() * len(idx)
        pv = torch.from_numpy(predict(net, xv)).to(dev)
        f1, iou = f1_by_threshold(pv, yv_t)
        run.loss.append(total / len(X))
        run.val_f1.append(float(f1.max()))
        run.val_iou.append(float(iou[f1.argmax()]))
        run.threshold = float(THRESHOLDS[f1.argmax()])
        if epoch % 10 == 9:
            print(f"{name} pass {epoch + 1}: loss {run.loss[-1]:.4f}, val F1 {f1.max():.3f}, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    run.seconds = time.perf_counter() - t0
    check_interior(THRESHOLDS, run.threshold, f"{name} threshold")
    return run


# ---------------------------------------------------------------------------------------------
# The question and the attempt


def overlay(ax: plt.Axes, img: np.ndarray, b: np.ndarray | None, y: np.ndarray) -> None:
    """A range image with true positives (green), false positives (orange), misses (blue).

    Args:
        ax: Axes to draw on.
        img: The range image.
        b: Predicted mask, or None to draw the true mask alone (green).
        y: True mask.
    """
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    code = y.astype(int) if b is None else np.select([b & y, b & ~y, ~b & y], [1, 2, 3], 0)
    # On a whole image a 1-px line is thinner than a slide pixel: thicken it; zooms stay exact.
    shown = maximum_filter(code, size=3) if img.shape[0] > 200 else code
    ax.imshow(np.ma.masked_equal(shown, 0), cmap=CMAP_TFN, vmin=0, vmax=3,
              interpolation="nearest")  # fmt: skip
    ax.axis("off")


def samples_figure(test: Split, show: list[int], out_dir: Path) -> dict[str, float]:
    """Four test range images with their masks: what the data looks like.

    Args:
        test: The test split.
        show: Indices of the four images.
        out_dir: Figure directory.

    Returns:
        Each image's crack share (the panel titles).
    """
    fig, axes = plt.subplots(1, 4, figsize=aml_style.FULL)
    shares = {}
    for ax, i in zip(axes, show, strict=True):
        overlay(ax, test.x[i], None, test.y[i])
        shares[test.stems[i]] = float(test.y[i].mean())
        ax.set_title(f"{100 * shares[test.stems[i]]:.1f}% crack", fontsize=12)
    aml_style.save(fig, out_dir, "samples")
    return shares


def ltpp_figure(test: Split, branches: list[Branch], out_dir: Path) -> dict[str, object]:
    """What the agency wants: each crack branch of one image coloured by its LTPP severity.

    The image (``LTPP_STEM``) was picked by eye from a contact sheet of the test images:
    longitudinal and transverse cracks of all three severities by the range-image widths, and
    no potholes (their annotated outlines are not cracks).

    Args:
        test: The test split.
        branches: Test centreline branches.
        out_dir: Figure directory.

    Returns:
        The image and its metres of cracking at each severity.
    """
    i = test.stems.index(LTPP_STEM)
    metres = np.zeros(3)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.imshow(test.x[i], cmap="gray", vmin=0, vmax=255)
    colors = (aml_style.PALETTE[2], aml_style.PALETTE[0], aml_style.PALETTE[1])
    for b in branches:
        if b.key == (test.name, LTPP_STEM) and b.inner.any():
            k = int(crack3d.band(b.range_mm[b.inner].mean()))
            metres[k] += b.length_m
            ax.plot(b.rc[:, 1], b.rc[:, 0], color=colors[k], lw=2.2, ls="-")
    for k, (name, rng) in enumerate(zip(SEVERITIES, ("≤ 6 mm", "6–19 mm", "> 19 mm"),
                                        strict=True)):  # fmt: skip
        ax.plot([], [], color=colors[k], lw=3, ls="-", label=f"{name} ({rng}): {metres[k]:.1f} m")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=10, ncol=1)
    ax.axis("off")
    aml_style.save(fig, out_dir, "ltpp")
    return {"stem": LTPP_STEM, "metres": [float(v) for v in metres]}


def attempt(splits: dict[str, Split]) -> dict[str, object]:
    """The attempt before any learning: a crack is darker, so threshold the range image.

    The grey level is chosen on validation for the best F1. Lecture 15's 3 x 3 "darker than
    its neighbours" filter, thresholded the same way, is the second attempt.

    Args:
        splits: The three splits.

    Returns:
        Each attempt's threshold and test scores (the first with its 95% interval).

    Raises:
        ValueError: If a level kept on validation is at an end of its grid.
    """
    val, test = splits["Val"], splits["Test"]
    levels, filter_levels = range(20, 130, 2), np.arange(2.0, 40.0)
    best = max(levels, key=lambda t: scores_f1(val.x < t, val.y))
    check_interior(levels, best, "grey level")
    filt_v = np.stack([correlate(a.astype(float), DARK_LINE, mode="nearest") for a in val.x])
    filt_t = np.stack([correlate(a.astype(float), DARK_LINE, mode="nearest") for a in test.x])
    best_f = max(filter_levels, key=lambda t: scores_f1(filt_v > t, val.y))
    check_interior(filter_levels, best_f, "filter level")
    dark = image_counts(test.x < best, test.y)
    return {
        "dark": {"level": best, **dark.pooled(), "ci": interval(dark, resamples(len(test.x)))},
        "filter": {"level": float(best_f), **scores(filt_t > best_f, test.y)},
    }


def scores_f1(b: np.ndarray, y: np.ndarray) -> float:
    """Strict F1 alone (for threshold searches)."""
    tp, fp, fn = counts(b, y)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def attempt_figure(test: Split, i: int, level: int, out_dir: Path) -> None:
    """The attempt on one test image: the image, "darker than the level", and the errors.

    Args:
        test: The test split.
        i: Index of the image.
        level: The grey level chosen on validation.
        out_dir: Figure directory.
    """
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    axes[0].imshow(test.x[i], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("range image (dark = low)", fontsize=12)
    axes[1].imshow(test.x[i] < level, cmap="gray_r")
    axes[1].set_title(f"darker than {level}", fontsize=12)
    overlay(axes[2], test.x[i], test.x[i] < level, test.y[i])
    axes[2].set_title("found / false / missed", fontsize=12)
    for ax in axes[:2]:
        ax.axis("off")
    aml_style.save(fig, out_dir, "attempt")


# ---------------------------------------------------------------------------------------------
# One picture: label every pixel; the U


def macs(net: nn.Module, shape: tuple[int, ...]) -> int:
    """Multiply-adds of every convolution in one forward pass on an input of ``shape``.

    Args:
        net: The network.
        shape: Input shape, (1, C, H, W).

    Returns:
        The count.
    """
    total = 0

    def hook(m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        nonlocal total
        k = math.prod(m.kernel_size)
        if isinstance(m, nn.ConvTranspose2d):  # each input value stamps a c_out x k x k kernel
            total += _inp[0].numel() * m.out_channels * k
        else:
            total += out.numel() * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in net.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]  # fmt: skip
    with torch.no_grad():
        net.eval()(torch.zeros(shape))
    for h in hs:
        h.remove()
    return total


class Encoder(nn.Module):
    """The U-Net's encoder alone (its five levels with the 2 x 2 pooling between them)."""

    def __init__(self, net: UNet) -> None:
        """Share the given U-Net's encoder blocks."""
        super().__init__()
        self.down = net.down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, 1, h, w) to the coarsest level's features (B, 256, h / 16, w / 16)."""
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
        return x


def sliding_cost() -> dict[str, float]:
    """Labelling every pixel: one window per pixel versus one fully convolutional pass.

    A classifier that looks at a ``WINDOW_PX`` (64) square window around a pixel (the U-Net's
    encoder followed by a linear layer, a Lecture 15 network) must run once per pixel: 319,488
    times per image. The U-Net computes each convolution once and shares it among all the windows
    that contain it.

    Returns:
        Multiply-adds per image both ways, and their ratio.
    """
    net = UNet()
    whole = macs(net, (1, 1, H, W))
    window = macs(Encoder(net), (1, 1, WINDOW_PX, WINDOW_PX))
    return {"windows": H * W, "unet_gmac": whole / 1e9, "sliding_gmac": H * W * window / 1e9,
            "ratio": H * W * window / whole}  # fmt: skip


def sliding_figure(test: Split, i: int, cost: dict[str, float], out_dir: Path) -> None:
    """One 64 x 64 window for one pixel, and the cost of a window per pixel versus one pass.

    Args:
        test: The test split.
        i: Index of the image.
        cost: ``sliding_cost``'s result.
        out_dir: Figure directory.
    """
    rows, cols = np.nonzero(test.y[i])
    r, c = int(rows[len(rows) // 2]), int(cols[len(cols) // 2])
    grid = {"width_ratios": [1, 1.5], "wspace": 0.45}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw=grid)
    ax0.imshow(test.x[i], cmap="gray", vmin=0, vmax=255)
    half = WINDOW_PX // 2
    ax0.add_patch(Rectangle((c - half, r - half), WINDOW_PX, WINDOW_PX, fill=False,
                            ec=aml_style.PALETTE[1], lw=2))  # fmt: skip
    ax0.plot(c, r, "o", color=aml_style.PALETTE[1], ms=4)
    ax0.set_title("one window, one pixel", fontsize=12)
    ax0.axis("off")
    vals = [cost["sliding_gmac"], cost["unet_gmac"]]
    ax1.barh(["a window\nper pixel", "U-Net,\none pass"], vals,
             color=[aml_style.PALETTE[1], aml_style.PALETTE[0]])  # fmt: skip
    for k, v in enumerate(vals):
        ax1.text(v * 1.3, k, f"{v:,.0f} billion", va="center", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 3e6)
    ax1.set_xlabel("billion multiply-adds (log scale)")
    aml_style.save(fig, out_dir, "sliding")


def unet_levels(
    net: UNet, test: Split, i: int
) -> tuple[list[np.ndarray], list[tuple[int, int, int]], np.ndarray]:
    """Each level's map that best matches the crack, on one test image.

    At each level, the channel whose map correlates best with the mask (resampled to that
    level), signed so that the crack is high.

    Args:
        net: The trained U-Net.
        test: The test split.
        i: Index of the image.

    Returns:
        The maps (encoder levels, then decoder levels), each level's (height, width, channels),
        and the predicted probability of crack.
    """
    dev = next(net.parameters()).device
    keep: list[torch.Tensor] = []
    with torch.no_grad():
        net.eval()
        logit = net(normalize(torch.from_numpy(test.x[i : i + 1]).to(dev)[:, None]), keep)
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    yt = torch.from_numpy(test.y[i]).float()[None, None]
    maps, sizes = [], []
    for level in keep:
        f = level[0].float().cpu()
        y_lvl = F.adaptive_avg_pool2d(yt, f.shape[-2:])[0, 0].flatten()
        z = f.flatten(1)
        z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-8)
        corr = (z @ (y_lvl - y_lvl.mean())) / (y_lvl.std() + 1e-8) / z.shape[1]
        best = int(corr.abs().argmax())
        maps.append(f[best].numpy() * torch.sign(corr[best]).item())
        sizes.append((int(f.shape[1]), int(f.shape[2]), int(f.shape[0])))
    return maps, sizes, prob


def unet_figure(run: Run, test: Split, i: int, out_dir: Path) -> dict[str, object]:
    """The one picture: a U-Net's maps on a real image, laid out as the U.

    Levels are drawn to scale (``unet_levels`` picks each level's map), so the bottom of the U
    is a coarse 39 x 32 grid that knows *what* is in the image but not exactly *where*.

    Args:
        run: The trained U-Net.
        test: The test split.
        i: Index of the image.
        out_dir: Figure directory.

    Returns:
        The map size and channel count at each level, and the number of weights.
    """
    net = run.net
    maps, sizes, prob = unet_levels(net, test, i)
    fig_w, fig_h = aml_style.FULL[0], 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Tiles shrink by 2^0.55 a level (the maps halve; full scale would hide the deep levels).
    hs = [1.5 * 2 ** (-0.55 * k) for k in range(5)]  # tile heights, inches
    ws = [h * W / H for h in hs]
    gap, step, top = 0.12, 0.34, fig_h - 0.3
    tops = [top - step * k for k in range(5)]
    # Left to right: encoder levels 0-4, decoder levels 3-0, then the output.
    tiles = [("enc", k, maps[k], sizes[k]) for k in range(5)]
    tiles += [("dec", 3 - j, maps[5 + j], sizes[5 + j]) for j in range(4)]
    tiles += [("out", 0, prob, None)]
    x, left = 0.05, {}
    for kind, k, _, _ in tiles:
        left[kind, k] = x
        x += ws[k] + gap
    assert x <= fig_w + gap, f"tiles need {x:.2f} in, the figure is {fig_w} in wide"
    for kind, k, img, size in tiles:
        ax = fig.add_axes((left[kind, k] / fig_w, (tops[k] - hs[k]) / fig_h,
                           ws[k] / fig_w, hs[k] / fig_h))  # fmt: skip
        if kind == "out":
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap="magma")
        ax.axis("off")
        text = "P(crack)" if size is None else f"{size[0]}×{size[1]}\n{size[2]} ch"
        fig.text((left[kind, k] + ws[k] / 2) / fig_w, (tops[k] - hs[k] - 0.04) / fig_h, text,
                 ha="center", va="top", fontsize=9)  # fmt: skip
    for k in range(4):  # skips: just under each level's top edge, above the deeper tiles
        y = (tops[k] - 0.1) / fig_h
        fig.add_artist(FancyArrowPatch(
            ((left["enc", k] + ws[k]) / fig_w, y), (left["dec", k] / fig_w, y),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10, lw=1.2,
            color=aml_style.PALETTE[2], linestyle="--"))  # fmt: skip
    mid = (left["enc", 4] + ws[4] / 2) / fig_w
    fig.text(mid, (top + 0.02) / fig_h, "skips copy the where", ha="center", va="bottom",
             fontsize=11, color=aml_style.PALETTE[2])  # fmt: skip
    low = (min(t - h for t, h in zip(tops, hs, strict=True)) - 0.45) / fig_h
    fig.text((left["enc", 1] + ws[1]) / fig_w, low, "encoder: halve, deepen", ha="center",
             fontsize=11, color=aml_style.PALETTE[0])  # fmt: skip
    fig.text(left["dec", 1] / fig_w, low, "decoder: double, sharpen", ha="center",
             fontsize=11, color=aml_style.PALETTE[1])  # fmt: skip
    aml_style.save(fig, out_dir, "unet_maps")
    return {"levels": [list(s) for s in sizes], "weights": sum(p.numel() for p in net.parameters())}


# ---------------------------------------------------------------------------------------------
# Derive: transposed convolution, the overlap losses, thin structures


def conv_transpose2d(x: np.ndarray, w: np.ndarray, stride: int) -> np.ndarray:
    """Transposed convolution from scratch: every input value stamps a scaled copy of the kernel.

    Args:
        x: Input, (C_in, h, w).
        w: Kernel, (C_in, C_out, k, k) (PyTorch's ``ConvTranspose2d`` layout).
        stride: Step between stamps in the output.

    Returns:
        Output, (C_out, (h - 1) * stride + k, (w - 1) * stride + k).
    """
    _, h, wd = x.shape
    _, c_out, k, _ = w.shape
    out = np.zeros((c_out, (h - 1) * stride + k, (wd - 1) * stride + k))
    for i in range(h):
        for j in range(wd):
            # (C_in,) . (C_in, C_out, k, k) -> (C_out, k, k): the stamp of input pixel (i, j)
            out[:, i * stride : i * stride + k, j * stride : j * stride + k] += np.einsum(
                "c,cokl->okl", x[:, i, j], w
            )
    return out


def verify_transpose(seed: int = SEED) -> dict[str, object]:
    """Check the from-scratch transposed convolution two ways.

    (1) Against ``torch.nn.functional.conv_transpose2d``. (2) Against the backward pass of an
    ordinary strided convolution: the gradient of <conv2d(z, w), x> with respect to z is the
    transposed convolution of x. It is the dx of Lecture 15's backward pass.

    Args:
        seed: Seed of the random input and kernel.

    Returns:
        The largest absolute differences.
    """
    rng = np.random.default_rng((seed, 16))
    x = rng.normal(size=(3, 5, 4))
    w = rng.normal(size=(3, 2, 4, 4))
    ours = conv_transpose2d(x, w, stride=2)
    ref = F.conv_transpose2d(torch.from_numpy(x)[None], torch.from_numpy(w), stride=2)[0]
    z = torch.zeros((1, 2, *ours.shape[1:]), dtype=torch.float64, requires_grad=True)
    y = F.conv2d(z, torch.from_numpy(w), stride=2)  # (1, 3, 5, 4): the shapes of x
    assert y.shape[1:] == x.shape, (y.shape, x.shape)
    (y * torch.from_numpy(x)[None]).sum().backward()
    return {
        "vs_pytorch": float(np.abs(ours - ref.numpy()).max()),
        "vs_backward": float(np.abs(ours - z.grad[0].numpy()).max()),
        "shape": list(ours.shape),
    }


BUILD_X: Final[np.ndarray] = np.array([[2.0, 4.0], [6.0, 0.0]])  # a 2 x 2 map to double
BUILD_K: Final[np.ndarray] = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 4.0


def transpose_build(out_dir: Path) -> dict[str, object]:
    """A four-frame build: each input value stamps the kernel, scaled, two pixels apart.

    The kernel is the tent [1 2 1] x [1 2 1] / 4, so the result is bilinear upsampling: a
    learned transposed convolution can at least interpolate, and learns more.

    Args:
        out_dir: Figure directory.

    Returns:
        The input, kernel, and output.
    """
    full = conv_transpose2d(BUILD_X[None], BUILD_K[None, None], stride=2)[0]
    for frame in range(1, 5):
        part = np.zeros_like(BUILD_X)
        part.flat[:frame] = BUILD_X.flat[:frame]
        out = conv_transpose2d(part[None], BUILD_K[None, None], stride=2)[0]
        i, j = divmod(frame - 1, 2)
        fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL,
                                 gridspec_kw={"width_ratios": [2, 3, 5]})  # fmt: skip
        _grid(axes[0], BUILD_X, "input 2×2", hi=(i, j, 1))
        _grid(axes[1], BUILD_K * BUILD_X[i, j], f"{BUILD_X[i, j]:g} × kernel", hi=None)
        _grid(axes[2], out, "output 5×5 (sum of stamps)", hi=(2 * i, 2 * j, 3))
        aml_style.save(fig, out_dir, f"tconv_{frame}")
    return {"x": BUILD_X.tolist(), "k": (BUILD_K * 4).tolist(), "out": full.tolist()}


def _grid(ax: plt.Axes, a: np.ndarray, title: str, hi: tuple[int, int, int] | None) -> None:
    """Draw a small array as numbered cells; ``hi`` (row, col, size) outlines a square."""
    ax.imshow(a, cmap="Blues", vmin=0, vmax=max(float(np.abs(a).max()), 1e-9) * 1.6)
    for (r, c), v in np.ndenumerate(a):
        ax.text(c, r, f"{v:g}", ha="center", va="center", fontsize=11 if a.shape[0] < 5 else 10,
                color=aml_style.INK)  # fmt: skip
    ax.set_xticks(np.arange(-0.5, a.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, a.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12)
    if hi is not None:
        r, c, size = hi
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), size, size, fill=False,
                               ec=aml_style.PALETTE[1], lw=3))  # fmt: skip


def matrix_figure(out_dir: Path) -> dict[str, list[int]]:
    """Convolution as a matrix C, and its transpose: the same weights, the other direction.

    A 3 x 3 kernel on a 4 x 4 image gives a 2 x 2 map: C is 4 x 16, each row the kernel laid
    along the flattened image. C^T is 16 x 4 and maps a 2 x 2 map back to 4 x 4.

    Args:
        out_dir: Figure directory.

    Returns:
        The two shapes.
    """
    k = np.arange(1, 10, dtype=float).reshape(3, 3)
    C = np.zeros((4, 16))
    for r in range(2):
        for c in range(2):
            img = np.zeros((4, 4))
            img[r : r + 3, c : c + 3] = k
            C[2 * r + c] = img.ravel()
    # The transposed convolution of a 2 x 2 map equals C^T times its flattened values.
    z = np.array([1.0, -2.0, 0.5, 3.0])
    assert np.allclose(C.T @ z, conv_transpose2d(z.reshape(1, 2, 2), k[None, None], 1)[0].ravel())
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"width_ratios": [4, 1.6]})
    for ax, m, title in ((axes[0], C, "C: 4×16, image to map"),
                         (axes[1], C.T, r"$C^{\top}$: 16×4, map to image")):  # fmt: skip
        ax.imshow(np.ma.masked_equal(m, 0), cmap="viridis", vmin=0, vmax=9)
        for (r, c), v in np.ndenumerate(m):
            if v:
                ax.text(c, r, f"{v:g}", ha="center", va="center", fontsize=8, color="white")
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    aml_style.save(fig, out_dir, "matrix")
    return {"C": list(C.shape), "CT": list(C.T.shape)}


def shift_figure(
    test: Split, twins: dict[str, dict[str, object]], out_dir: Path
) -> dict[str, list[float]]:
    """Thin structures: a perfect mask moved by s pixels, scored strictly and within 2 px.

    Args:
        test: The test split.
        twins: The agreement of the twin annotations (drawn as reference lines).
        out_dir: Figure directory.

    Returns:
        Strict IoU and F1 within 2 px at each shift.
    """
    shifts = [0, 1, 2, 3]
    ys = test.y[test.y.reshape(len(test.y), -1).any(1)]
    iou, tol = [], []
    for s in shifts:
        b = np.zeros_like(ys)
        b[:, :, s:] = ys[:, :, : W - s]  # every crack moved s px to the right
        sc = scores(b, ys)
        iou.append(sc["iou"])
        tol.append(sc["tol_f1"])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(shifts, iou, marker="o", label="strict IoU")
    ax.plot(shifts, tol, marker="s", label=f"F1 within {TOL} px")
    agree = np.mean([t["iou"] for t in twins.values()])
    ax.axhline(agree, color=aml_style.MUTED, lw=1.2, ls=":")
    ax.text(3, agree + 0.02, "two annotators", ha="right", fontsize=10, color=aml_style.MUTED)
    ax.set_xticks(shifts)
    ax.set_xlabel("true mask shifted by (px)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    aml_style.save(fig, out_dir, "shift")
    return {"shifts": shifts, "iou": iou, "tol_f1": tol}


def twins_figure(
    splits: dict[str, Split], twins: list[list[tuple[str, int]]], out_dir: Path
) -> dict[str, dict[str, object]]:
    """The same scan annotated twice: the labels' own agreement (the noise floor).

    Args:
        splits: The three splits.
        twins: Pairs of (split, index) with identical pixels.
        out_dir: Figure directory.

    Returns:
        Agreement of each pair: strict IoU and F1, F1 within 1 and 2 px.
    """
    out = {}
    fig, axes = plt.subplots(1, len(twins), figsize=aml_style.FULL)
    for ax, ((s1, i1), (s2, i2)) in zip(np.atleast_1d(axes), twins, strict=True):
        a, b = splits[s1].y[i1], splits[s2].y[i2]
        stem = splits[s1].stems[i1]
        sc = scores(b[None], a[None])
        p1, r1 = tolerant(b[None], a[None], tol=1)
        out[stem] = {"splits": [s1, s2], "iou": sc["iou"], "f1": sc["f1"],
                     "tol1_f1": 2 * p1 * r1 / max(p1 + r1, 1e-12), "tol2_f1": sc["tol_f1"],
                     "pixels": [int(a.sum()), int(b.sum())]}  # fmt: skip
        rows, cols = np.nonzero(a & ~b)  # zoom where the two disagree
        n = 96
        r0 = int(np.clip(np.median(rows) - n // 2, 0, H - n))
        c0 = int(np.clip(np.median(cols) - n // 2, 0, W - n))
        sl = (slice(r0, r0 + n), slice(c0, c0 + n))
        overlay(ax, splits[s1].x[i1][sl], b[sl], a[sl])
        ax.set_title(f"{stem} ({s1} vs {s2}): IoU {sc['iou']:.2f}", fontsize=11)
    aml_style.save(fig, out_dir, "twins")
    return out


# ---------------------------------------------------------------------------------------------
# Measuring from pixels


@dataclass
class Measured:
    """One branch's length and mean width under four rulers (NaN when not measured)."""

    length_m: float
    mask_mm: float  # the dataset's width_mm: the mask ruler at the true centreline
    range_mm: float  # the dataset's width_mm_rangeimg
    same_mm: float  # the ridge ruler on the true mask: the control for pred_mm
    pred_mm: float  # the ridge ruler on the U-Net's mask


def measure(test: Split, pred: np.ndarray, branches: list[Branch],
            skip: set[tuple[str, str]]) -> tuple[dict[str, float], list[Measured]]:  # fmt: skip
    """Widths at the test centrelines: check the mask ruler, then one ruler for both masks.

    The mask ruler at the true centreline reproduces the dataset's ``width_mm``. A predicted
    mask has its own centreline, so both masks are read with the ridge ruler
    (``crack3d.Ridge``); on the true mask it is the control: what it disagrees with there is
    the ruler's doing, not the network's.

    Args:
        test: The test split.
        pred: The U-Net's test masks, (N, H, W) bool.
        branches: Test centreline branches, in file order (grouped by image).
        skip: (split, stem) of the faint masks, whose centrelines are noise (``mask_facts``).

    Returns:
        The agreement counts, and every measured branch.
    """
    index = {s: i for i, s in enumerate(test.stems)}
    n = dict.fromkeys(("points", "exact", "same_true", "found", "same_pred", "agree", "wider",
                       "narrower"), 0)  # fmt: skip
    out = []
    for stem, same_image in itertools.groupby(branches, key=lambda b: b.stem):
        group = [b for b in same_image if b.key not in skip and b.inner.any()]
        if not group:
            continue
        i = index[stem]
        true_r, pred_r = crack3d.Ridge.of(test.y[i]), crack3d.Ridge.of(pred[i])
        for br in group:
            rc, want = br.rc[br.inner], br.mask_mm[br.inner]
            at_point = crack3d.width_mm(true_r.d[rc[:, 0], rc[:, 1]])  # the mask ruler
            same, pw = true_r.width(rc), pred_r.width(rc)
            hit = pw > 0
            n["points"] += len(rc)
            n["exact"] += int(np.isclose(at_point, want, atol=0.05).sum())
            n["same_true"] += int(np.isclose(same, want, atol=0.05).sum())
            n["found"] += int(hit.sum())
            n["same_pred"] += int(np.isclose(pw[hit], want[hit], atol=0.05).sum())
            n["agree"] += int(np.isclose(pw[hit], same[hit], atol=0.05).sum())
            n["wider"] += int((pw[hit] > same[hit] + 0.05).sum())
            n["narrower"] += int((pw[hit] < same[hit] - 0.05).sum())
            same_mm = float(same[same > 0].mean()) if (same > 0).any() else math.nan
            pred_mm = float(pw[hit].mean()) if hit.mean() >= 0.5 else math.nan
            out.append(Measured(br.length_m, float(want.mean()),
                                float(br.range_mm[br.inner].mean()), same_mm, pred_mm))  # fmt: skip
    pts, found = n["points"], max(n["found"], 1)
    numbers = {
        "points": pts,
        "ruler_exact": n["exact"] / pts,  # the mask ruler reproduces width_mm
        "same_ruler_true": n["same_true"] / pts,  # the ridge ruler on the true mask
        "pred_found": n["found"] / pts,
        "pred_same_width": n["same_pred"] / found,  # U-Net, ridge ruler, against width_mm
        "pred_vs_true": n["agree"] / found,  # U-Net against the true mask, the same ruler
        "pred_wider": n["wider"] / found,
        "pred_narrower": n["narrower"] / found,
    }
    return numbers, out


def lengths(test: Split, pred: np.ndarray, branches: list[Branch],
            skip: set[tuple[str, str]]) -> dict[str, object]:  # fmt: skip
    """Crack length per test image: the U-Net mask's skeleton against the reference centrelines.

    The skeleton length of the true mask is the control: the estimator's own bias. Per-image
    errors are over the images with at least 1 m of reference cracking; totals over all test
    images but the faint one (crack-free images add only predicted length).

    Args:
        test: The test split.
        pred: The U-Net's test masks.
        branches: Test centreline branches.
        skip: (split, stem) of the faint masks.

    Returns:
        Total metres three ways, and the median and quartiles of the per-image relative error.
    """
    ref: dict[str, float] = defaultdict(float)
    for br in branches:
        ref[br.stem] += br.length_m
    rows = [(ref.get(stem, 0.0), crack3d.skeleton_length_m(test.y[i]),
             crack3d.skeleton_length_m(pred[i]))
            for i, stem in enumerate(test.stems) if (test.name, stem) not in skip]  # fmt: skip
    a = np.array(rows)
    big = a[:, 0] >= 1.0

    def spread(col: int) -> dict[str, float]:  # relative error of column col against the reference
        e = a[big, col] / a[big, 0] - 1
        return {"median": float(np.median(e)), "q25": float(np.percentile(e, 25)),
                "q75": float(np.percentile(e, 75))}  # fmt: skip

    return {"images": len(a), "images_1m": int(big.sum()), "ref_m": float(a[:, 0].sum()),
            "true_skeleton_m": float(a[:, 1].sum()), "pred_skeleton_m": float(a[:, 2].sum()),
            "true_err": spread(1), "pred_err": spread(2)}  # fmt: skip


def edt_figure(test: Split, i: int, box: tuple[int, int, int], out_dir: Path) -> dict[str, float]:
    """The distance transform over a crack as a roof: its ridge height is half the width.

    Args:
        test: The test split.
        i: Index of the image.
        box: (row, column, size) of the crop.
        out_dir: Figure directory.

    Returns:
        The largest distance in the crop and the width it gives.
    """
    r, c, n = box
    m = test.y[i, r : r + n, c : c + n]
    d = distance_transform_edt(m)
    fig = plt.figure(figsize=aml_style.FULL)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05)
    ax0 = fig.add_subplot(grid[0])
    ax0.imshow(test.x[i, r : r + n, c : c + n], cmap="gray", vmin=0, vmax=255)
    ax0.contour(m, levels=[0.5], colors=[aml_style.PALETTE[2]], linewidths=1.5)
    ax0.axis("off")
    ax0.set_title(f"mask on the range image ({n * MM_PER_PX / 10:.0f} cm square)", fontsize=12)
    ax1 = fig.add_subplot(grid[1], projection="3d")
    Y, X = np.mgrid[0:n, 0:n]
    aml_style.surface3d(ax1, X, Y, d, cmap="viridis")
    aml_style.style3d(ax1, elev=38, azim=-60)
    ax1.set_box_aspect(None, zoom=1.25)
    ax1.set_zlabel("d (px)")
    ax1.set_title("distance to the edge, d", fontsize=12, pad=18)  # above the zoomed box
    # The rule sits in the figure, so the slide holds the picture alone, at full width.
    rule = r"width $w = (2d - 1) \times 4$ mm: a one-pixel line ($d = 1$) is 4 mm wide"
    fig.text(0.5, 1.02, rule, ha="center", va="bottom", fontsize=14)
    aml_style.save(fig, out_dir, "edt")
    return {"max_d": float(d.max()), "max_width_mm": float((2 * d.max() - 1) * MM_PER_PX)}


def widths_figure(branches: list[Branch], skip: set[tuple[str, str]],
                  out_dir: Path) -> dict[str, object]:  # fmt: skip
    """Point widths from the mask come in steps of the pixel; the range image's do not.

    Args:
        branches: Test centreline branches.
        skip: (split, stem) of the faint masks, left out.
        out_dir: Figure directory.

    Returns:
        The most common mask widths and their shares; medians of both rulers; the range
        ruler's 5th and 95th percentiles; the share of points wider than 8 mm (two pixels) by
        each ruler.
    """
    a = np.concatenate([b.mask_mm[b.inner] for b in branches if b.key not in skip])
    r = np.concatenate([b.range_mm[b.inner] for b in branches if b.key not in skip])
    values, n = np.unique(np.round(a, 1), return_counts=True)
    top = np.argsort(-n)[:4]
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    bins = np.arange(0, 41, 0.5)
    ax.hist(r, bins=bins, color=aml_style.PALETTE[0], alpha=0.55, label="range-image ruler")
    ax.hist(a, bins=bins, color=aml_style.PALETTE[1], alpha=0.75, label="mask ruler")
    for x, name in zip((0.0, *crack3d.LTPP_MM), SEVERITIES, strict=True):
        if x:
            ax.axvline(x, color=aml_style.MUTED, lw=1, ls=":")
        ax.text(x + 0.4, ax.get_ylim()[1] * 0.92, name, fontsize=11, color=aml_style.MUTED)
    ax.set_xlabel("crack width at a centreline point (mm)")
    ax.set_ylabel("points (test)")
    ax.set_xlim(0, 40)
    ax.legend(loc="center right")
    aml_style.save(fig, out_dir, "widths")
    return {"mask_steps": [float(values[k]) for k in top],
            "mask_step_share": [float(n[k] / n.sum()) for k in top],
            "mask_median": float(np.median(a)), "range_median": float(np.median(r)),
            "range_p5": float(np.percentile(r, 5)), "range_p95": float(np.percentile(r, 95)),
            "mask_over_8mm": float((a > 2 * MM_PER_PX).mean()),
            "range_over_8mm": float((r > 2 * MM_PER_PX).mean()),
            "corr": float(np.corrcoef(a, r)[0, 1]), "points": len(a)}  # fmt: skip


def severity_table(measured: list[Measured], attr: str) -> list[float]:
    """Metres of cracking at each LTPP severity under one ruler (all cracking treated as linear).

    Args:
        measured: Branches.
        attr: ``"mask_mm"``, ``"range_mm"``, ``"same_mm"``, or ``"pred_mm"``.

    Returns:
        Metres at low, moderate, high, and (last) metres not measured.
    """
    out = [0.0, 0.0, 0.0, 0.0]
    for m in measured:
        w = getattr(m, attr)
        out[3 if math.isnan(w) else int(crack3d.band(w))] += m.length_m
    return out


def rulers_figure(measured: list[Measured], out_dir: Path) -> dict[str, object]:
    """Two rulers, two severities: metres of cracking by severity under each ruler.

    Args:
        measured: Test branches.
        out_dir: Figure directory.

    Returns:
        Metres by severity under the dataset's two rulers, the ridge ruler on the true mask (the
        control), and the ridge ruler on the U-Net's mask; the control again on the branches the
        U-Net found (``same_found``, like for like with ``pred``) and on those it missed
        (``same_missed``); the confusion of the first two (length-weighted).
    """
    found = [m for m in measured if not math.isnan(m.pred_mm)]
    missed = [m for m in measured if math.isnan(m.pred_mm)]
    conf = np.zeros((3, 3))
    for m in measured:
        conf[int(crack3d.band(m.mask_mm)), int(crack3d.band(m.range_mm))] += m.length_m
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.imshow(conf, cmap="Blues")
    for (r, c), v in np.ndenumerate(conf):
        ax.text(c, r, f"{v:.0f} m", ha="center", va="center", fontsize=13,
                color="white" if v > conf.max() / 2 else aml_style.INK)  # fmt: skip
    ax.set_xticks(range(3), SEVERITIES)
    ax.set_yticks(range(3), SEVERITIES)
    ax.set_xlabel("severity by the range-image ruler")
    ax.set_ylabel("severity by the mask ruler")
    aml_style.save(fig, out_dir, "rulers")
    return {"mask": severity_table(measured, "mask_mm"),
            "range": severity_table(measured, "range_mm"),
            "same": severity_table(measured, "same_mm"),
            "pred": severity_table(measured, "pred_mm"),
            "same_found": severity_table(found, "same_mm"),
            "same_missed": severity_table(missed, "same_mm"),
            "agree": float(np.trace(conf) / conf.sum()),
            "confusion": [[float(v) for v in row] for row in conf]}  # fmt: skip


def pinhole_figure(out_dir: Path) -> dict[str, float]:
    """Millimetres per pixel = distance x pixel pitch / focal length, for four sensors.

    The scanner's own spacing (1 mm), the dataset as released (4 mm), Hüthwohl et al.'s 0.1 mm
    for a 0.3 mm crack, and an example phone camera (f = 4.2 mm, 1.4 µm pixels) at 1 m and 3 m.

    Args:
        out_dir: Figure directory.

    Returns:
        The phone example's millimetres per pixel at the two distances.
    """
    phone = {z: z * 1000 * PHONE_PITCH_MM / PHONE_F_MM for z in (1.0, 3.0)}
    grid = {"width_ratios": [1.2, 1], "wspace": 0.55}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw=grid)
    ax0.plot([0, 0], [-0.4, 0.4], color=aml_style.INK, ls="-", lw=4)  # sensor
    ax0.plot([1, 1], [-0.15, 0.15], color=aml_style.INK, ls="-", lw=2)  # lens
    ax0.plot([6, 6], [-2.0, 2.0], color=aml_style.MUTED, ls="-", lw=4)  # surface
    for s in (-1, 1):
        ax0.plot([0, 6], [s * 0.4, -s * 2.0], color=aml_style.PALETTE[0], ls="-", lw=1.2)
    ax0.annotate("", (1, -0.75), (0, -0.75), arrowprops={"arrowstyle": "<->"})
    ax0.text(0.5, -0.9, "f", ha="center", va="top", fontsize=13)  # below its arrow
    ax0.annotate("", (6, -2.4), (1, -2.4), arrowprops={"arrowstyle": "<->"})
    ax0.text(3.5, -2.55, "Z (distance)", ha="center", va="top", fontsize=12)
    ax0.text(-0.15, 0.55, "pixel\npitch p", ha="center", fontsize=11)
    ax0.text(6.25, 0, "ground\npixel\nZ p / f", fontsize=12, va="center")
    ax0.set_xlim(-0.8, 8.2)
    ax0.set_ylim(-3.3, 2.4)
    ax0.axis("off")
    names = ["0.3 mm crack at 3 px", "phone at 1 m", "the scanner", "phone at 3 m",
             "3DCrack release"]  # fmt: skip
    vals = [HUTHWOHL_MM, phone[1.0], crack3d.NATIVE_MM, phone[3.0], MM_PER_PX]
    ax1.barh(names, vals, color=[aml_style.PALETTE[2]] + [aml_style.PALETTE[0]] * 3
             + [aml_style.PALETTE[1]])  # fmt: skip
    for k, v in enumerate(vals):
        ax1.text(v * 1.15, k, f"{v:.2g} mm", va="center", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_xlim(0.05, 20)
    ax1.set_xlabel("mm per pixel (log scale)")
    ax1.invert_yaxis()
    aml_style.save(fig, out_dir, "pinhole")
    return {"phone_1m": phone[1.0], "phone_3m": phone[3.0], "ratio": MM_PER_PX / HUTHWOHL_MM}


# ---------------------------------------------------------------------------------------------
# Break it


def found_by_width(test: Split, pred: np.ndarray, branches: list[Branch],
                   skip: set[tuple[str, str]], factor: int) -> dict[str, float]:  # fmt: skip
    """Share of true centreline points with a predicted crack within one (coarse) pixel.

    Hairline points (mask width 4 mm: a one- or two-pixel line under the (2d - 1) rule) and
    wider ones are counted apart.

    Args:
        test: The test split (for the stem index).
        pred: Predicted masks on a grid ``factor`` times coarser.
        branches: Test centreline branches.
        skip: (split, stem) of the faint masks, left out.
        factor: 1 (4 mm) or 2 (8 mm).

    Returns:
        Shares for "hairline" and "wider".
    """
    index = {s: i for i, s in enumerate(test.stems)}
    near = maximum_filter(pred, size=(1, 3, 3))
    hit = {"hairline": [0, 0], "wider": [0, 0]}
    for br in branches:
        if br.key in skip:
            continue
        rc = br.rc[br.inner] // factor
        h = near[index[br.stem], rc[:, 0], rc[:, 1]]
        for key, sel in (("hairline", np.isclose(br.mask_mm[br.inner], MM_PER_PX)),
                         ("wider", br.mask_mm[br.inner] > MM_PER_PX + 0.05)):  # fmt: skip
            hit[key][0] += int(h[sel].sum())
            hit[key][1] += int(sel.sum())
    return {k: v[0] / v[1] for k, v in hit.items()}


def coarse_figure(test: Split, i: int, box: tuple[int, int, int], *, pred4: np.ndarray,
                  pred8: np.ndarray, out_dir: Path) -> None:  # fmt: skip
    """One crop at 4 mm and at 8 mm per pixel, with each model's errors.

    Args:
        test: The test split.
        i: Index of the image.
        box: (row, column, size) of the crop at 4 mm.
        pred4: The 4 mm model's mask for image i.
        pred8: The 8 mm model's mask for image i (on the 8 mm grid).
        out_dir: Figure directory.
    """
    r, c, n = box
    x8, y8 = crack3d.coarsen(Split("one", [test.stems[i]], test.x[i : i + 1], test.y[i : i + 1]))
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    overlay(axes[0], test.x[i, r : r + n, c : c + n], pred4[r : r + n, c : c + n],
            test.y[i, r : r + n, c : c + n])  # fmt: skip
    axes[0].set_title("4 mm per pixel", fontsize=12)
    s = slice(r // 2, (r + n) // 2), slice(c // 2, (c + n) // 2)
    overlay(axes[1], x8[0][s], pred8[s], y8[0][s])
    axes[1].set_title("8 mm per pixel", fontsize=12)
    aml_style.save(fig, out_dir, "coarse")


def faint_figure(splits: dict[str, Split], faint: list[dict], out_dir: Path) -> None:
    """A faint mask: what ``> 0`` and ``> 127`` call crack on the same image.

    Args:
        splits: The three splits.
        faint: The faint masks found by ``mask_facts``.
        out_dir: Figure directory.
    """
    f = faint[0]
    s = splits[f["split"]]
    i = s.stems.index(f["stem"])
    m = np.asarray(Image.open(crack3d.RAW / s.name / "anno" / f"{f['stem']}.png"))
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    axes[0].imshow(s.x[i], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{f['stem']} ({s.name})", fontsize=11)
    axes[1].imshow(m > 0, cmap="gray_r")
    axes[1].set_title(f"mask > 0: {100 * f['any']:.0f}% crack", fontsize=11)
    axes[2].imshow(m > MASK_CUT, cmap="gray_r")
    axes[2].set_title(f"mask > {MASK_CUT}: {100 * f['cut']:.1f}% crack", fontsize=11)
    for ax in axes:
        ax.axis("off")
    aml_style.save(fig, out_dir, "faint")


# ---------------------------------------------------------------------------------------------
# SpaceNet 8: the siamese U-Net (the data, matching, and boxes are in spacenet8.py)


def imagenet6(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """ImageNet's mean and std for six channels (before, then after), on the 0-255 scale."""
    mean = torch.tensor(IMAGENET_MEAN * 2, device=dev).view(1, 6, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD * 2, device=dev).view(1, 6, 1, 1) * 255
    return mean, std


class SiameseUNet(nn.Module):
    """A U-Net whose encoder is ImageNet's ResNet-18, shared by the before and after images.

    Each image goes through the same encoder (Lecture 15's transfer, fine-tuned here); at every
    level the two maps are concatenated, and a U-Net decoder doubles back to full resolution.
    The design of the SpaceNet 8 baseline (Hänsch et al., 2022), smaller.

    Args:
        c_out: Classes per pixel.
    """

    def __init__(self, c_out: int = 5) -> None:
        """Load the ImageNet weights (torchvision IMAGENET1K_V1) and build the decoder."""
        super().__init__()
        r = resnet18(weights=WEIGHTS)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # 1/2 size, 64 channels
        self.layers = nn.ModuleList(
            [nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4]
        )  # 1/4 64, 1/8 128, 1/16 256, 1/32 512
        enc = [64, 64, 128, 256, 512]
        dec = [32, 64, 128, 256]
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        c = 2 * enc[4]
        for lvl in (3, 2, 1, 0):  # back up to 1/16, 1/8, 1/4, 1/2
            self.up.append(nn.ConvTranspose2d(c, dec[lvl], 2, stride=2))
            self.dec.append(conv_block(dec[lvl] + 2 * enc[lvl], dec[lvl]))
            c = dec[lvl]
        self.last_up = nn.ConvTranspose2d(c, 32, 2, stride=2)  # 1/2 -> full size
        self.last = conv_block(32, 32)
        self.head = nn.Conv2d(32, c_out, 1)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """ResNet-18's maps at 1/2, 1/4, 1/8, 1/16, 1/32 of the input size."""
        feats = [self.stem(x)]
        for layer in self.layers:
            feats.append(layer(feats[-1]))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, 6, H, W), before then after (each ImageNet-normalized), to (B, c_out, H, W)."""
        fa, fb = self.encode(x[:, :3]), self.encode(x[:, 3:])
        feats = [torch.cat([a, b], dim=1) for a, b in zip(fa, fb, strict=True)]
        y = feats.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            y = up(y)
            skip = feats.pop()
            assert y.shape[-2:] == skip.shape[-2:], (y.shape, skip.shape)
            y = dec(torch.cat([skip, y], dim=1))
        return self.head(self.last(self.last_up(y)))


@dataclass
class Sn8Run:
    """The trained siamese U-Net and its record."""

    net: nn.Module
    loss: list[float]  # mean training loss per pass
    seconds: float


def sn8_train(train: list[Tile], val: list[Tile]) -> Sn8Run:
    """Train the siamese U-Net on random 512 px crops of the training tiles.

    The loss is cross-entropy plus soft Dice over the four object classes (each is 0.3-1.3%
    of the pixels).

    Args:
        train: Training tiles.
        val: Validation tiles (pixel F1 of each object class, printed every 10 passes).

    Returns:
        The network, its losses per pass, and the seconds it took.
    """
    dev = device()
    xt, yt = spacenet8.stack(train)
    mean, std = imagenet6(dev)  # the normalization the pretrained encoder expects
    X = torch.from_numpy(xt).to(dev).permute(0, 3, 1, 2)  # (N, 6, 1312, 1312) uint8
    Y = torch.from_numpy(yt).to(dev)  # (N, 1312, 1312) uint8; widened per batch in sn8_batch
    torch.manual_seed(SEED)
    classes = len(spacenet8.CLASSES)
    net = SiameseUNet(c_out=classes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=SN8_LR, weight_decay=WD)
    per_pass = math.ceil(len(X) * SN8_CROPS / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=SN8_LR, total_steps=SN8_EPOCHS * per_pass, pct_start=0.1
    )
    gen = torch.Generator().manual_seed(SEED)
    losses: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(SN8_EPOCHS):
        net.train()
        tiles = torch.randperm(len(X) * SN8_CROPS, generator=gen) % len(X)
        # ``high`` is exclusive, so no crop reaches a tile's last row and column (1 px of 1,300).
        # Left as it is: the fix would redraw every crop and move every SpaceNet 8 number.
        corners = torch.randint(0, spacenet8.SIZE - SN8_CROP, (len(tiles), 2), generator=gen)
        flips = torch.rand(len(tiles), 2, generator=gen) < 0.5
        total = 0.0
        for i in range(0, len(tiles), BATCH):
            ks = range(i, min(i + BATCH, len(tiles)))
            xb, yb = sn8_batch(X, Y, ks, tiles=tiles, corners=corners, flips=flips)
            xb = (xb.float() - mean) / std
            with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                logits = net(xb).float()
            p = logits.softmax(1)
            one = F.one_hot(yb, classes).permute(0, 3, 1, 2).float()
            inter, size = (p * one).sum((0, 2, 3)), (p + one).sum((0, 2, 3))
            dice = 1 - ((2 * inter + 1) / (size + 1))[1:].mean()
            # Cross-entropy written out: CUDA's nll_loss2d has no deterministic kernel.
            ce = -(logits.log_softmax(1) * one).sum(1).mean()
            loss = ce + dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        losses.append(total / per_pass)
        if epoch % 10 == 9:
            f1 = sn8_pixel_f1(net, val)
            print(
                f"SN8 pass {epoch + 1}: loss {losses[-1]:.3f}, val F1 by class "
                f"{np.round(f1, 3).tolist()}, {time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    return Sn8Run(net, losses, time.perf_counter() - t0)


def sn8_batch(X: torch.Tensor, Y: torch.Tensor, ks: range, *, tiles: torch.Tensor,
              corners: torch.Tensor, flips: torch.Tensor) -> tuple[torch.Tensor, ...]:  # fmt: skip
    """Cut, flip, and stack one batch of training crops.

    Args:
        X: Training images, (N, 6, 1312, 1312) uint8.
        Y: Training labels, (N, 1312, 1312) uint8.
        ks: Positions of this batch in the pass's crop list.
        tiles: Tile index of every crop in the pass.
        corners: Upper-left (row, column) of every crop.
        flips: Whether to mirror each crop across and along.

    Returns:
        Images (B, 6, 512, 512) uint8 and labels (B, 512, 512) int64.
    """
    xb, yb = [], []
    for k in ks:
        r, c = corners[k].tolist()
        xa = X[tiles[k], :, r : r + SN8_CROP, c : c + SN8_CROP]
        ya = Y[tiles[k], r : r + SN8_CROP, c : c + SN8_CROP]
        if flips[k, 0]:
            xa, ya = xa.flip(2), ya.flip(1)
        if flips[k, 1]:
            xa, ya = xa.flip(1), ya.flip(0)
        xb.append(xa)
        yb.append(ya)
    return torch.stack(xb), torch.stack(yb).long()


def sn8_predict(net: nn.Module, tiles: list[Tile]) -> np.ndarray:
    """Class probabilities for whole tiles.

    Args:
        net: The trained network.
        tiles: Tiles to predict.

    Returns:
        (N, 5, 1300, 1300) float32 probabilities.
    """
    dev = next(net.parameters()).device
    mean, std = imagenet6(dev)
    x, _ = spacenet8.stack(tiles)
    size = spacenet8.SIZE
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
        for k in range(len(x)):
            xb = torch.from_numpy(x[k : k + 1]).to(dev).permute(0, 3, 1, 2).float()
            p = net((xb - mean) / std).float().softmax(1)
            out.append(p[0, :, :size, :size].cpu().numpy())
    return np.stack(out)


def sn8_pixel_f1(net: nn.Module, tiles: list[Tile]) -> np.ndarray:
    """Pixel F1 of each object class (argmax labels) on whole tiles."""
    p = sn8_predict(net, tiles).argmax(1)
    y = np.stack([t.lab for t in tiles])
    return np.array([2 * ((p == c) & (y == c)).sum() / max((p == c).sum() + (y == c).sum(), 1)
                     for c in range(1, len(spacenet8.CLASSES))])  # fmt: skip


def degrees_figure(tiles: list[Tile], out_dir: Path) -> dict[str, float]:
    """A pixel in degrees is not square on the ground: road lengths three ways.

    Each road's length from its pixel geometry: (a) pixels times one pixel size (the north-south
    size, the "Max GSD" the dataset lists), (b) east-west and north-south steps converted
    separately, against (c) the reference length in metres the dataset provides.

    Args:
        tiles: All tiles.
        out_dir: Figure directory.

    Returns:
        Kilometres three ways, the pixel's sides in degrees and metres, and errors by direction.
    """
    sizes = {t.deg for t in tiles}
    assert len(sizes) == 1, sizes  # every pre-event tile has the same pixel size in degrees
    sx, sy = sizes.pop()
    rows = []
    for t in tiles:
        m_lon, m_lat = spacenet8.metres_per_degree(t.lat)
        for road in t.roads:
            c = np.asarray(road["line"])
            dx, dy = np.diff(c[:, 0]), np.diff(c[:, 1])
            naive = np.hypot(dx, dy).sum() * sy * m_lat
            proper = np.hypot(dx * sx * m_lon, dy * sy * m_lat).sum()
            bearing = math.degrees(math.atan2(np.abs(dy * sy * m_lat).sum(),
                                              np.abs(dx * sx * m_lon).sum()))  # fmt: skip
            rows.append((road["length_m"], naive, proper, bearing))
    every = np.array(rows)
    r = every[every[:, 0] > 20]  # direction is ill-defined for the short pieces at tile edges
    lat = float(np.mean([t.lat for t in tiles]))
    m_lon, m_lat = spacenet8.metres_per_degree(lat)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=aml_style.FULL,
                                   gridspec_kw={"width_ratios": [1, 2.2]})  # fmt: skip
    px_x, px_y = sx * m_lon, sy * m_lat
    ax0.add_patch(Rectangle((0, 0), px_x, px_y, color=aml_style.PALETTE[0], alpha=0.5))
    ax0.text(px_x / 2, -0.05, f"{px_x:.2f} m", ha="center", va="top", fontsize=12)
    ax0.text(px_x + 0.02, px_y / 2, f"{px_y:.2f} m", va="center", fontsize=12)
    ax0.set_xlim(-0.05, 0.6)
    ax0.set_ylim(-0.15, 0.5)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.set_title("one pixel on the ground", fontsize=12)
    ax1.scatter(r[:, 3], 100 * (r[:, 1] / r[:, 0] - 1), s=6, alpha=0.4,
                color=aml_style.PALETTE[1], label=f"pixels × {px_y:.2f} m")  # fmt: skip
    ax1.scatter(r[:, 3], 100 * (r[:, 2] / r[:, 0] - 1), s=6, alpha=0.4,
                color=aml_style.PALETTE[2], label="each axis in metres")  # fmt: skip
    ax1.axhline(0, color=aml_style.MUTED, lw=1)
    ax1.set_xlabel("road direction (0° east-west, 90° north-south)")
    ax1.set_ylabel("length error (%)")
    ax1.set_xticks([0, 30, 60, 90])
    ax1.legend(loc="upper right", markerscale=3)
    aml_style.save(fig, out_dir, "degrees")
    ew, ns = r[:, 3] < 15, r[:, 3] > 75
    km = every[:, :3].sum(0) / 1000
    axes_err = np.abs(r[:, 2] / r[:, 0] - 1)
    return {"km_reference": float(km[0]), "km_naive": float(km[1]), "km_axes": float(km[2]),
            "pixel_deg": float(sx), "mean_lat": lat, "pixel_x_m": px_x, "pixel_y_m": px_y,
            "naive_ew": float(np.median(r[ew, 1] / r[ew, 0] - 1)),
            "naive_ns": float(np.median(r[ns, 1] / r[ns, 0] - 1)),
            "axes_abs": float(np.median(axes_err)),
            "axes_abs_p95": float(np.percentile(axes_err, 95)),
            "roads": len(every), "roads_plotted": len(r)}  # fmt: skip


def sn8_pair_figures(t: Tile, p: np.ndarray, out_dir: Path) -> None:
    """One test tile: before, after, and truth; then the truth beside the U-Net's map.

    Args:
        t: The tile.
        p: Its class probabilities, (5, 1300, 1300).
        out_dir: Figure directory.
    """
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    axes[0].imshow(t.pre)
    axes[0].set_title("before (pre-event)", fontsize=12)
    axes[1].imshow(t.post)
    axes[1].set_title("after (post-event)", fontsize=12)
    axes[2].imshow(t.post)
    axes[2].imshow(np.ma.masked_equal(t.lab, 0), cmap=SN8_COLORS, vmin=0, vmax=4,
                   interpolation="nearest", alpha=0.85)  # fmt: skip
    axes[2].set_title("truth: flooded in blue / pink", fontsize=12)
    for ax in axes:
        ax.axis("off")
    aml_style.save(fig, out_dir, "sn8_pair")
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    for ax, lab, title in ((axes[0], t.lab, "truth"), (axes[1], p.argmax(0), "U-Net")):
        ax.imshow(t.post)
        ax.imshow(np.ma.masked_equal(lab, 0), cmap=SN8_COLORS, vmin=0, vmax=4,
                  interpolation="nearest", alpha=0.85)  # fmt: skip
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    aml_style.save(fig, out_dir, "sn8_pred")


def nms_figure(t: Tile, boxes: spacenet8.TileBoxes, box: tuple[int, int, int],
               out_dir: Path) -> dict[str, int]:  # fmt: skip
    """A crop's proposals, the boxes NMS keeps, and the true boxes found or missed.

    Args:
        t: The tile.
        boxes: Its proposals, kept boxes, and true boxes (``spacenet8.detection``).
        box: (row, column, size) of the crop.
        out_dir: Figure directory.

    Returns:
        Proposals, kept boxes, and true boxes inside the crop.
    """
    b, tb = boxes.boxes, boxes.true
    r0, c0, n = box
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    inside = (b[:, 0] >= c0) & (b[:, 2] <= c0 + n) & (b[:, 1] >= r0) & (b[:, 3] <= r0 + n)
    kept = np.zeros(len(b), bool)
    kept[boxes.kept] = True
    shown = [(np.nonzero(inside)[0], "proposals"), (np.nonzero(inside & kept)[0], "after NMS")]
    for ax, (idx, title) in zip(axes[:2], shown, strict=True):
        ax.imshow(t.pre[r0 : r0 + n, c0 : c0 + n])
        for j in idx:
            x0, y0, x1, y1 = b[j]
            ax.add_patch(Rectangle((x0 - c0, y0 - r0), x1 - x0, y1 - y0, fill=False,
                                   ec=aml_style.PALETTE[1], lw=1.2))  # fmt: skip
        ax.set_title(f"{title}: {len(idx)}", fontsize=12)
    ax = axes[2]
    ax.imshow(t.pre[r0 : r0 + n, c0 : c0 + n])
    kept_in = b[np.nonzero(inside & kept)[0]]
    iou = box_iou(torch.from_numpy(kept_in), torch.from_numpy(tb)).numpy() if len(kept_in) else None
    for g, (x0, y0, x1, y1) in enumerate(tb):
        if x0 >= c0 and x1 <= c0 + n and y0 >= r0 and y1 <= r0 + n:
            hit = iou is not None and iou[:, g].max() >= 0.5
            ec = aml_style.PALETTE[2] if hit else aml_style.PALETTE[0]
            ax.add_patch(Rectangle((x0 - c0, y0 - r0), x1 - x0, y1 - y0, fill=False, lw=1.5,
                                   ec=ec))  # fmt: skip
    ax.set_title("true boxes: found / missed", fontsize=12)
    for ax in axes:
        ax.axis("off")
    aml_style.save(fig, out_dir, "nms")
    true_in = (tb[:, 0] >= c0) & (tb[:, 2] <= c0 + n) & (tb[:, 1] >= r0) & (tb[:, 3] <= r0 + n)
    return {"proposals": int(inside.sum()), "kept": int((inside & kept).sum()),
            "true": int(true_in.sum())}  # fmt: skip


def merged_figure(t: Tile, p: np.ndarray, box: tuple[int, int, int],
                  out_dir: Path) -> dict[str, int]:  # fmt: skip
    """Touching houses become one blob: true footprints and the U-Net's connected components.

    Args:
        t: The tile.
        p: Its class probabilities.
        box: (row, column, size) of the crop.
        out_dir: Figure directory.

    Returns:
        True buildings and merged ones in the crop, and in the tile.
    """
    comps = spacenet8.blobs(np.isin(p.argmax(0), (1, 2)))
    m = spacenet8.match(t.inst, comps)
    r0, c0, n = box
    sl = (slice(r0, r0 + n), slice(c0, c0 + n))
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    rng = np.random.default_rng((SEED, 16))
    for ax, lab, title in ((axes[0], t.inst[sl], "true footprints"),
                           (axes[1], comps[sl], "U-Net blobs")):  # fmt: skip
        ids = np.unique(lab[lab > 0])
        colors = np.zeros((int(lab.max()) + 1, 4))
        colors[ids] = np.c_[rng.uniform(0.25, 1, (len(ids), 3)), np.full(len(ids), 0.75)]
        ax.imshow(t.pre[sl])
        ax.imshow(colors[lab], interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    merged_here = np.isin(t.inst[sl], list(m.merged_ids))
    axes[0].contour(merged_here, levels=[0.5], colors=["white"], linewidths=1.0)
    aml_style.save(fig, out_dir, "merged")
    ids = {int(v) for v in np.unique(t.inst[sl])} - {0}
    return {"true": len(ids), "merged": len(ids & m.merged_ids), "tile_true": m.true,
            "tile_merged": m.merged}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Results, decide, title art


def curves_figure(runs: dict[str, Run], out_dir: Path) -> None:
    """Training loss and validation F1 per pass for the U-Net and its variants.

    Args:
        runs: The runs, keyed as ``RUN_LABELS``.
        out_dir: Figure directory.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=aml_style.FULL)
    for key, run in runs.items():
        x = np.arange(1, len(run.loss) + 1)
        if key == "unet":
            ax0.plot(x, run.loss, ls="-")
        ax1.plot(x, run.val_f1, label=RUN_LABELS[key])
    ax0.set_xlabel("pass over the 1,139 training images")
    ax0.set_ylabel("training loss (cross-entropy)")
    ax1.set_xlabel("pass")
    ax1.set_ylabel("validation F1 (best threshold)")
    ax1.set_ylim(0.4, 0.9)
    ax1.legend(loc="lower right")
    aml_style.save(fig, out_dir, "curves")


def dice_figure(test: Split, pred: np.ndarray, i: int, box: tuple[int, int, int],
                out_dir: Path) -> dict[str, float]:  # fmt: skip
    """Overlap on one crop: true positives, false positives, misses, and the scores they give.

    Args:
        test: The test split.
        pred: U-Net test masks.
        i: Index of the image.
        box: (row, column, size) of the crop.
        out_dir: Figure directory.

    Returns:
        The crop's counts, Dice (F1), and IoU.
    """
    r, c, n = box
    b, y = pred[i, r : r + n, c : c + n], test.y[i, r : r + n, c : c + n]
    tp, fp, fn = counts(b, y)
    grid = {"width_ratios": [1, 1.1], "wspace": 0.7}  # the bars' y label clears the crop
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.2, 3.0), gridspec_kw=grid)
    overlay(ax0, test.x[i, r : r + n, c : c + n], b, y)
    ax1.bar(["found\n(TP)", "false\n(FP)", "missed\n(FN)"], [tp, fp, fn],
            color=[aml_style.PALETTE[2], aml_style.PALETTE[1], aml_style.PALETTE[0]])  # fmt: skip
    for k, v in enumerate((tp, fp, fn)):
        ax1.text(k, v, f"{v}", ha="center", va="bottom", fontsize=12)
    ax1.set_ylabel("pixels in the crop")
    aml_style.save(fig, out_dir, "dice")
    return {"tp": tp, "fp": fp, "fn": fn, "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1)}  # fmt: skip


def overlays_figure(
    test: Split, pred: np.ndarray, show: list[int], out_dir: Path
) -> list[dict[str, object]]:
    """Three test images: found (green), false (orange), missed (blue) pixels.

    Args:
        test: The test split.
        pred: U-Net test masks.
        show: Indices of the images.
        out_dir: Figure directory.

    Returns:
        Strict and tolerant F1 of each shown image.
    """
    fig, axes = plt.subplots(1, len(show), figsize=aml_style.FULL)
    out = []
    for ax, i in zip(axes, show, strict=True):
        overlay(ax, test.x[i], pred[i], test.y[i])
        sc = scores(pred[i : i + 1], test.y[i : i + 1])
        out.append({"stem": test.stems[i], "f1": sc["f1"], "tol_f1": sc["tol_f1"]})
        ax.set_title(f"F1 {sc['f1']:.2f}; within 2 px {sc['tol_f1']:.2f}", fontsize=11)
    aml_style.save(fig, out_dir, "overlays")
    return out


def skips_figure(test: Split, preds: dict[str, np.ndarray], i: int, box: tuple[int, int, int],
                 out_dir: Path) -> None:  # fmt: skip
    """The same crop decoded without and with skip connections.

    Args:
        test: The test split.
        preds: Test masks of the no-skip network and the U-Net.
        i: Index of the image.
        box: (row, column, size) of the crop.
        out_dir: Figure directory.
    """
    r, c, n = box
    fig, axes = plt.subplots(1, len(preds), figsize=aml_style.FULL)
    for ax, (name, p) in zip(axes, preds.items(), strict=True):
        overlay(ax, test.x[i, r : r + n, c : c + n], p[i, r : r + n, c : c + n],
                test.y[i, r : r + n, c : c + n])  # fmt: skip
        ax.set_title(name, fontsize=12)
    aml_style.save(fig, out_dir, "skips")


def decide_figure(sev: dict[str, list[float]], out_dir: Path) -> None:
    """Metres of test cracking at each LTPP severity under four rulers.

    The fourth group is the length on branches the U-Net mostly missed (only its ruler leaves
    length unmeasured): the U-Net's bars plus that length add up to the others'.

    Args:
        sev: Metres by severity (and not measured) for "mask", "same", "pred", "range".
        out_dir: Figure directory.
    """
    names = {"mask": "true mask, dataset ruler", "same": "true mask, ridge ruler",
             "pred": "U-Net mask, ridge ruler", "range": "range image"}  # fmt: skip
    hatches = ("", "//", "..", "xx")
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    xs = np.arange(4)
    for k, (key, name) in enumerate(names.items()):
        ax.bar(xs + (k - 1.5) * 0.2, sev[key][:4], width=0.2, label=name,
               color=aml_style.PALETTE[k], hatch=hatches[k], edgecolor="white")  # fmt: skip
    bands = zip(SEVERITIES, ("≤ 6 mm", "6–19 mm", "> 19 mm"), strict=True)
    ax.set_xticks(xs, [f"{s}\n({lo})" for s, lo in bands] + ["branch\nmissed"])
    ax.set_ylabel("metres of cracking (250 test\nimages, all treated as linear)")
    ax.legend(loc="upper right", fontsize=11)
    aml_style.save(fig, out_dir, "decide")


def title_art_figure(branches: list[Branch], stem: str, out_dir: Path) -> None:
    """The title art: a crack network as lines of light, and a dark sun at the bottom of a U.

    Args:
        branches: Centreline branches.
        stem: The image whose cracks are drawn.
        out_dir: Figure directory.
    """
    c = title_art.Canvas(seed=16)
    top = c.top

    def to_xy(row: float, col: float) -> tuple[float, float]:  # image px -> canvas units
        return 0.30 + 0.62 * col / W, top - 0.04 - (top - 0.08) * row / H

    c.ground(("#0b0d14", "#12101c", "#2a1014", "#08070b"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.02, 0.04, 0.34, 0.46, 4), soft=4), "#5a2216", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.64, 0.04, 0.34, 0.30, 3), soft=4), "#3a1912", 0.85)
    segs = [[to_xy(r, col) for r, col in b.rc[::3]] for b in branches if b.stem == stem]
    c.glow(c.lines(segs, 2.0, soft=1.0), "#f3c98a", 0.75, spread=10)
    u = [(0.10, top - 0.10), (0.10, 0.22), (0.18, 0.12), (0.26, 0.22), (0.26, top - 0.10)]
    c.glow(c.lines([u], 3.0, soft=1.0), "#e39a45", 0.5, spread=12)
    centre = (0.18, 0.17)
    rim = c.disc(centre, 0.05, soft=3) - c.disc(centre, 0.04, soft=3)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.6, spread=20)
    c.glaze(c.disc(centre, 0.04, soft=2), "#1d0a10", 0.95)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean forms in rust at the bottom; "
        "across the right of the canvas runs a pale golden network of lines, the real cracks of "
        "a scanned pavement; on the left a glowing amber U descends to a small dark sun with a "
        "burning rim, the bottom of a U-Net.",
    )


def crack_runs(splits: dict[str, Split]) -> dict[str, Run]:
    """The U-Net and two variants, same data, seed, and passes.

    In a first sweep (one seed, 40 passes) plain cross-entropy scored a slightly higher
    validation F1 than cross-entropy + Dice, so the U-Net uses plain cross-entropy and Dice is
    the variant; with one seed, that difference is no evidence either way.

    Args:
        splits: The three splits.

    Returns:
        The runs, keyed as ``RUN_LABELS`` ("unet", "no_skips", "dice").
    """
    tr, va = splits["Train"], splits["Val"]
    return {
        "unet": train_unet(tr.x, tr.y, va.x, va.y, name="U-Net"),
        "no_skips": train_unet(tr.x, tr.y, va.x, va.y, skips=False, name="no skips"),
        "dice": train_unet(tr.x, tr.y, va.x, va.y, loss="bce+dice", name="+ Dice"),
    }


def break_runs(splits: dict[str, Split], branches: list[Branch], skip: set[tuple[str, str]],
               pred: np.ndarray) -> tuple[dict[str, object], np.ndarray]:  # fmt: skip
    """Masks read with ``> 0`` (for training, and for scoring), and the survey at 8 mm.

    Args:
        splits: The three splits.
        branches: Test centrelines.
        skip: (split, stem) of the faint masks.
        pred: The U-Net's test masks (4 mm).

    Returns:
        Test scores of each, and the 8 mm model's test masks.
    """
    va, te = splits["Val"], splits["Test"]
    tr0 = crack3d.load_split("Train", cut=0)
    run0 = train_unet(tr0.x, tr0.y, va.x, va.y, name="mask > 0")
    faint = scores(predict(run0.net, te.x) > run0.threshold, te.y) | {
        "threshold": run0.threshold, "val_f1": run0.val_f1[-1],
        "train_crack_share": float(tr0.y.mean())}  # fmt: skip
    del tr0, run0
    te0 = crack3d.load_split("Test", cut=0)  # the same U-Net, scored against masks read with > 0
    faint |= {"scored_on_any": scores(pred, te0.y), "test_crack_share_any": float(te0.y.mean()),
              "test_crack_share": float(te.y.mean())}  # fmt: skip
    del te0
    (x8, y8), (xv8, yv8), (xt8, yt8) = (crack3d.coarsen(splits[s]) for s in SPLITS)
    run8 = train_unet(x8, y8, xv8, yv8, name="8 mm")
    p8 = predict(run8.net, xt8)[:, : H // 2] > run8.threshold
    coarse = {
        "found4": found_by_width(te, pred, branches, skip, 1),
        "found8": found_by_width(te, p8, branches, skip, 2),
        "f1_8": scores(p8, yt8[:, : H // 2]),
        "smallest_mm": 2 * MM_PER_PX,
        "threshold": run8.threshold,
        "val_f1": run8.val_f1[-1],
    }
    return {"faint_run": faint, "coarse": coarse}, p8


def finding_and_flooding(tiles: list[Tile], prob: np.ndarray) -> dict[str, float]:
    """Finding buildings, and telling flooded from dry where one was found, kept apart.

    The per-class pixel F1s count a flooded roof called dry as an error of both classes, so
    neither measures finding a roof. Here: pixel F1 of "building, flooded or not" against the
    true buildings, and, on the true building pixels the network calls building, the share
    whose flooded-or-dry call is right.

    Args:
        tiles: Test tiles.
        prob: Their class probabilities, (N, 5, 1300, 1300).

    Returns:
        ``building_any_f1`` and ``flood_call_on_found`` (with the pixels it counts).
    """
    a = prob.argmax(1)
    y = np.stack([t.lab for t in tiles])
    pred_b, true_b = np.isin(a, (1, 2)), np.isin(y, (1, 2))
    tp = int((pred_b & true_b).sum())
    found = pred_b & true_b
    return {"building_any_f1": 2 * tp / max(int(pred_b.sum() + true_b.sum()), 1),
            "flood_call_on_found": int((a[found] == y[found]).sum()) / max(tp, 1),
            "found_pixels": tp}  # fmt: skip


def sn8_section(out_dir: Path) -> dict[str, object]:
    """SpaceNet 8: data counts, degrees versus metres, the flood U-Net, detection, merges.

    Args:
        out_dir: Figure directory.

    Returns:
        Every SpaceNet 8 number on the slides.

    Raises:
        ValueError: If the from-scratch NMS keeps other boxes than torchvision's.
    """
    t0 = time.perf_counter()
    tiles = spacenet8.tiles()
    part = spacenet8.split(tiles)
    print(f"SN8 tiles read in {time.perf_counter() - t0:.0f} s", flush=True)
    out: dict[str, object] = {"data": {
        k: {"tiles": len(v), "buildings": sum(len(t.buildings) for t in v),
            "flooded": sum(1 for t in v for b in t.buildings if b["flooded"]),
            "road_km": sum(r["length_m"] for t in v for r in t.roads) / 1000,
            "flooded_road_km": sum(r["length_m"] for t in v for r in t.roads if r["flooded"])
            / 1000} for k, v in part.items()}}  # fmt: skip
    out["degrees"] = degrees_figure(tiles, out_dir)
    run = sn8_train(part["train"], part["val"])
    net, test = run.net, part["test"]
    prob = sn8_predict(net, test)
    print(f"SN8 trained in {time.perf_counter() - t0:.0f} s", flush=True)
    out["model"] = spacenet8.evaluate(test, prob) | {"minutes": run.seconds / 60,
                                                     "final_loss": run.loss[-1]}  # fmt: skip
    out["model"] |= finding_and_flooding(test, prob)
    det = spacenet8.detection(test, prob)
    if not det.numbers["nms_matches_torchvision"]:
        raise ValueError("the from-scratch NMS keeps other boxes than torchvision.ops.nms")
    test_scores = spacenet8.building_scores(test, prob)
    out["target"] = spacenet8.flood_target(
        spacenet8.building_scores(part["val"], sn8_predict(net, part["val"])), test_scores,
        recall=SN8_RECALL)  # fmt: skip
    out["model"]["flood_auc"] = float(roc_auc_score(test_scores[1], test_scores[0]))
    print(f"SN8 scored in {time.perf_counter() - t0:.0f} s", flush=True)
    k = [t.name for t in test].index(SN8_SHOW)
    sn8_pair_figures(test[k], prob[k], out_dir)
    out["crop"] = nms_figure(test[k], det.per_tile[SN8_SHOW], SN8_BOX, out_dir)
    out["detection"] = det.numbers
    out["merged"] = merged_figure(test[k], prob[k], SN8_BOX, out_dir)
    return out


def grey_near_crack(splits: dict[str, Split], skip: set[tuple[str, str]]) -> dict[str, float]:
    """Where the masks' grey values sit: the share within one pixel of a crack pixel.

    Args:
        splits: The three splits (masks cut at ``MASK_CUT``).
        skip: (split, stem) of the faint masks, left out (their grey covers the image).

    Returns:
        Grey pixels counted, and the share of them within one pixel (8-neighbours) of a pixel
        above ``MASK_CUT``.
    """
    grey = near = 0
    for sp in splits.values():
        for i, stem in enumerate(sp.stems):
            if (sp.name, stem) in skip:
                continue
            m = np.asarray(Image.open(crack3d.RAW / sp.name / "anno" / f"{stem}.png"))
            g = (m > 0) & (m < 255)
            if g.any():
                grey += int(g.sum())
                near += int((g & maximum_filter(sp.y[i], size=3)).sum())
    return {"grey_pixels": grey, "near_crack": near / max(grey, 1)}


def twins_unet(splits: dict[str, Split], pred: np.ndarray,
               twins: list[list[tuple[str, int]]]) -> dict[str, dict[str, object]]:  # fmt: skip
    """The U-Net on a test image that was annotated twice, against each annotation.

    The same image, the same model: a like-for-like comparison with the two annotations'
    agreement on that image (``twins_figure``).

    Args:
        splits: The three splits.
        pred: The U-Net's test masks.
        twins: Groups of (split, index) with identical pixels.

    Returns:
        Per test twin: strict F1, IoU, and F1 within 2 px against the test mask and against
        the other split's mask.
    """
    out = {}
    for group in twins:
        tests = [i for sp, i in group if sp == "Test"]
        others = [(sp, i) for sp, i in group if sp != "Test"]
        for i in tests:
            row: dict[str, object] = {"other_split": [sp for sp, _ in others]}
            for name, y in [("test_mask", splits["Test"].y[i])] + [
                (f"{sp}_mask", splits[sp].y[j]) for sp, j in others
            ]:
                sc = scores(pred[i][None], y[None])
                row[name] = {k: sc[k] for k in ("f1", "iou", "tol_f1")}
            out[splits["Test"].stems[i]] = row
    return out


def concepts(splits: dict[str, Split], at: dict[str, int],
             out_dir: Path) -> tuple[dict[str, object], crack3d.MaskFacts]:  # fmt: skip
    """The data, the attempt, and the derive step's pictures (no training).

    Args:
        splits: The three splits.
        at: Test index of each named stem.
        out_dir: Figure directory.

    Returns:
        Their numbers, and the mask facts (faint masks and twins) the later steps need.
    """
    test = splits["Test"]
    facts = crack3d.mask_facts(splits)
    out: dict[str, object] = {"data": {
        "images": {s: len(v.stems) for s, v in splits.items()},
        "crack_free": {s: int((~v.y.reshape(len(v.y), -1).any(1)).sum()) for s, v in
                       splits.items()},
        "images_total": sum(len(v.stems) for v in splits.values()),
        "crack_share": float(splits["Train"].y.mean()),
        "not_crack_share": 1 - float(splits["Train"].y.mean()),
        "extent_m": [H * MM_PER_PX / 1000, W * MM_PER_PX / 1000],
        "grey_masks": facts.grey_masks, "faint": facts.faint,
        "grey": grey_near_crack(splits, facts.faint_keys)}}  # fmt: skip
    out["twins"] = twins_figure(splits, facts.twins, out_dir)
    out["attempt"] = attempt(splits)
    out["samples"] = samples_figure(test, [at[s] for s in SHOW], out_dir)
    attempt_figure(test, at[ATTEMPT_STEM], out["attempt"]["dark"]["level"], out_dir)
    out["sliding"] = sliding_cost()
    sliding_figure(test, at[UNET_STEM], out["sliding"], out_dir)
    out["transpose"] = transpose_build(out_dir) | {"verify": verify_transpose()}
    out["matrix"] = matrix_figure(out_dir)
    out["shift"] = shift_figure(test, out["twins"], out_dir)
    out["ltpp"] = ltpp_figure(test, crack3d.load_branches("Test"), out_dir)
    return out, facts


@dataclass
class CrackResults:
    """The crack U-Nets' numbers, the U-Net's test masks, and the crop the slides zoom into."""

    numbers: dict[str, object]
    pred: np.ndarray
    box: tuple[int, int, int]


def crack_results(splits: dict[str, Split], at: dict[str, int], twins_test: list[int],
                  out_dir: Path) -> CrackResults:  # fmt: skip
    """Train the U-Net and its two variants; score them, with intervals; draw the results.

    Args:
        splits: The three splits.
        at: Test index of each named stem.
        twins_test: Test images whose pixels are also in another split (sample_000131).
        out_dir: Figure directory.

    Returns:
        Their numbers, the U-Net's test masks, and the crop shown.

    Raises:
        ValueError: If leaving out the test twin moves the U-Net's F1 by ``TWIN_DELTA_MAX``
            or more.
    """
    test = splits["Test"]
    runs = crack_runs(splits)
    preds = {k: predict(r.net, test.x) > r.threshold for k, r in runs.items()}
    cnt = {k: image_counts(preds[k], test.y) for k in runs}
    idx = resamples(len(test.stems))
    out: dict[str, object] = {"runs": {
        k: {"val_f1": r.val_f1[-1], "val_iou": r.val_iou[-1], "threshold": r.threshold,
            "minutes": r.seconds / 60, "final_loss": r.loss[-1], **cnt[k].pooled(),
            "ci": interval(cnt[k], idx), "val_f1_curve": list(r.val_f1)}
        for k, r in runs.items()}}  # fmt: skip
    # Same test images, same resamples: is either variant's difference from the U-Net real?
    out["paired_f1"] = {
        "unet_minus_dice": paired(cnt["unet"], cnt["dice"], idx),
        "unet_minus_noskips": paired(cnt["unet"], cnt["no_skips"], idx),
    }
    # sample_000131's pixels are also in training (annotated by someone else): leave it out.
    keep = np.setdiff1d(np.arange(len(test.stems)), twins_test)
    alone = cnt["unet"].pooled(keep)
    out["without_twin"] = {"images": len(keep), "f1": alone["f1"], "iou": alone["iou"],
                           "tol_f1": alone["tol_f1"],
                           "delta_f1": alone["f1"] - out["runs"]["unet"]["f1"]}  # fmt: skip
    if not abs(out["without_twin"]["delta_f1"]) < TWIN_DELTA_MAX:
        raise ValueError(f"the test twin moves the U-Net's F1: {out['without_twin']}")
    out["unet"] = unet_figure(runs["unet"], test, at[UNET_STEM], out_dir)
    curves_figure(runs, out_dir)
    pred = preds["unet"]
    cracked = [i for i in range(len(pred)) if test.y[i].any()]
    per_image = np.array([cnt["unet"].pooled(np.array([i]))["tol_f1"] for i in cracked])
    order = np.argsort(per_image, kind="stable")
    typical = [cracked[order[int(f * (len(order) - 1))]] for f in (0.25, 0.5, 0.75)]
    out["overlays"] = overlays_figure(test, pred, typical, out_dir)
    out["per_image_tol_f1"] = {
        k: float(np.quantile(per_image, q))
        for k, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))
    }
    rows, cols = np.nonzero(test.y[at[UNET_STEM]])
    half = CROP_PX // 2
    box = (int(np.clip(rows.mean() - half, 0, H - CROP_PX)),
           int(np.clip(cols.mean() - half, 0, W - CROP_PX)), CROP_PX)  # fmt: skip
    skips_figure(test, {"no skips": preds["no_skips"], "U-Net": pred}, at[UNET_STEM], box,
                 out_dir)  # fmt: skip
    out["dice"] = dice_figure(test, pred, at[UNET_STEM], box, out_dir)
    return CrackResults(out, pred, box)


def measuring(test: Split, pred: np.ndarray, branches: list[Branch],
              skip: set[tuple[str, str]], out_dir: Path) -> dict[str, object]:  # fmt: skip
    """Widths and lengths in millimetres: the rulers, their control, severity, the pinhole.

    Args:
        test: The test split.
        pred: The U-Net's test masks.
        branches: Test centrelines.
        skip: (split, stem) of the faint masks.
        out_dir: Figure directory.

    Returns:
        Their numbers.
    """
    meas, measured = measure(test, pred, branches, skip)
    i = test.stems.index(EDT_STEM)
    d = distance_transform_edt(test.y[i])
    r, c = np.unravel_index(int(d.argmax()), d.shape)
    box = (int(np.clip(r - EDT_CROP // 2, 0, H - EDT_CROP)),
           int(np.clip(c - EDT_CROP // 2, 0, W - EDT_CROP)), EDT_CROP)  # fmt: skip
    out = {
        "measure": meas,
        "lengths": lengths(test, pred, branches, skip),
        "edt": edt_figure(test, i, box, out_dir),
        "widths": widths_figure(branches, skip, out_dir),
        "rulers": rulers_figure(measured, out_dir),
        "pinhole": pinhole_figure(out_dir),
    }
    decide_figure(out["rulers"], out_dir)
    return out


def main() -> None:
    """Generate every figure and number for CEE Lecture 16."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    t0 = time.perf_counter()
    splits = {s: crack3d.load_split(s) for s in SPLITS}
    test = splits["Test"]
    at = {s: test.stems.index(s) for s in (*SHOW, ATTEMPT_STEM, UNET_STEM, EDT_STEM)}
    branches = crack3d.load_branches("Test")
    numbers, facts = concepts(splits, at, out_dir)
    skip = facts.faint_keys
    numbers["data"]["test_branches"] = len(branches)
    twins_test = [i for group in facts.twins for split, i in group if split == "Test"]
    print(f"concepts in {time.perf_counter() - t0:.0f} s", flush=True)
    res = crack_results(splits, at, twins_test, out_dir)
    numbers |= res.numbers
    numbers["twins_unet"] = twins_unet(splits, res.pred, facts.twins)
    print(f"crack U-Nets in {time.perf_counter() - t0:.0f} s", flush=True)
    broken, p8 = break_runs(splits, branches, skip, res.pred)
    numbers |= broken
    i = at[UNET_STEM]
    coarse_figure(test, i, res.box, pred4=res.pred[i], pred8=p8[i], out_dir=out_dir)
    faint_figure(splits, numbers["data"]["faint"], out_dir)
    numbers |= measuring(test, res.pred, branches, skip, out_dir)
    print(f"measuring in {time.perf_counter() - t0:.0f} s", flush=True)
    numbers["sn8"] = sn8_section(out_dir)
    title_art_figure(branches, TITLE_STEM, out_dir)
    aml_style.thumbnails(out_dir, ("unet_maps", "shift", "widths", "nms"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60  # read by Lecture 28's GPU ledger
    write_numbers(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
