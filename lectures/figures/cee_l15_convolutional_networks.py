"""Figures and numbers for CEE Lecture 15: convolutional networks and transfer learning.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l15_convolutional_networks.py \
        --out build/fig/cee_l15_convolutional_networks

Writes the PNG figures and ``numbers.json`` (the numbers the slides quote from this lecture's
runs; arithmetic on them, such as 160 / 252, is shown on the slides). Data: SDNET2018 (Maguire,
Dorafshan & Thomas, 2018; CC BY 4.0) from ``scripts/fetch_sdnet2018.sh``, with its tile index and
ImageNet ResNet-18 features from ``scripts/derive_sdnet2018.py`` (read by ``sdnet.py``).

The setting: an inspector photographs bridge decks, walls, and pavements. Can a network flag the
256-px tiles that show a crack, so the inspector reviews only those? 230 photographs, cut into
56,092 tiles, each labelled cracked or not. Photos, not tiles, are split into training,
validation, and test sets. A 3 x 3 filter and convolution are written from scratch and checked
against PyTorch; a small network is trained from scratch, and on 16 photos its schedule length is
chosen on validation by a bracketing search; ImageNet's ResNet-18 is reused as a frozen feature
extractor under a logistic regression (Lecture 6), also on tiles at half the resolution the small
network sees. The break-it tests add half of each test photo's tiles to the training tiles
(siblings, against as many tiles of other photos), train on one surface and test on another, and
fly the camera higher.
"""

from __future__ import annotations

import copy
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import correlate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.transforms.v2 import functional as tvf

import aml_style
import title_art
from common import parse_out_dir, seed_all, write_numbers
from probes import C_GRID, Probe, fit_probe, scores, threshold_for
from sdnet import (
    DARK_LINE,
    RAW,
    SIZE,
    SURFACES,
    backbone,
    load,
    mosaic,
    photo_split,
    preprocess,
    read_tile,
)

SEED: int = 0
# The numbers were made with 14 threads: sums run in an order that depends on the thread count.
THREADS: int = 14
FILTER_TILE: str = "D/CD/7001-21.jpg"  # a deck tile with one clean crack (photo D7001)
TEXTURE_TILE: str = "P/CP/001-104.jpg"  # a pavement tile: a crack in coarse aggregate
SHOW_PHOTOS: dict[str, str] = {"deck": "D7021", "wall": "W7102", "pavement": "P001"}
HAND_CUT: float = 10.0  # the hand filter's response that counts a pixel as "line-like"
# Processes that decode tiles; capped at 4 on the shared machine (pool.map keeps the order, so
# the count does not change any number).
WORKERS: int = 4
MOSAIC_PX: int = 48  # pixels per tile when a photograph is rebuilt from its tiles
RIDGE_CAP: float = 40.0  # the ridge surfaces' height cap: one-pixel ridges stay visible
ERF_TILES: int = 32  # random tiles the effective receptive field is averaged over
RESIDUAL_DEPTH: int = 32  # blocks in the gradient-flow figure's deepest stacks
RESIDUAL_SEEDS: int = 5  # weight seeds per stack; the figure shows their median
SHIFT_PX: int = 32  # the equivariance check's shift
# Inverse L2 strengths a probe's search may widen to, past probes.C_GRID's ends.
C_LADDER: tuple[float, ...] = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, *C_GRID, 3.0, 10.0, 100.0)
SPACE_TRAIN, SPACE_VAL, SPACE_SHOW = 6000, 2000, 300  # pixel-probe rows; tiles shown per class
OCCLUSION_PX, OCCLUSION_STRIDE = 48, 16  # the grey square's side and step
CONTROL_DRAWS: int = 3  # draws of the siblings' more-data control
BOOT_DRAWS: int = 2000  # photo-bootstrap resamples of the main probe's test numbers
SCRATCH_PX: int = 128  # the from-scratch network sees tiles at half resolution (CPU budget)
# Passes over the training tiles. Set after a 4-pass pilot whose validation AP was still rising;
# at 12 passes it still rises (see the curve), so the slides say so.
SCRATCH_EPOCHS: int = 12
SCRATCH_BATCH: int = 64
SCRATCH_LR: float = 3e-3  # peak of a one-cycle schedule (Smith & Topin, 2019), AdamW
SCRATCH_DECAY: float = 1e-4  # AdamW's weight decay
BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)  # shares of the training photos
SMALL_BUDGET: float = 0.1  # the from-scratch network is also trained on this share
SMALL_STEPS: int = 2000  # the first recipe's optimizer steps: 33 passes over the smaller set
SMALL_SEEDS: tuple[int, ...] = (0, 1, 2)  # seeds of the small-budget network (0 is the slides')
# The small network's schedule length (passes of one one-cycle schedule) is a
# bracketing search on the median validation AP of the three seeds. It starts from SMALL_START
# and, while the best length sits at an edge, adds the next length of this ladder past it; the
# ladder's ends raise. 33 passes is the first recipe's 2,000 steps (2,013).
SMALL_LADDER: tuple[int, ...] = (8, 16, 33, 66, 132, 264)
SMALL_START: tuple[int, ...] = (16, 33, 66)


# ---------------------------------------------------------------------------------------------
# Data (the loaders are in sdnet.py)


def outline_tiles(ax: plt.Axes, tiles: pd.DataFrame, px: int, **style: object) -> None:
    """Draw a square around each tile in ``tiles`` on a mosaic of ``px``-pixel tiles.

    Args:
        ax: The axes holding the mosaic.
        tiles: Tiles with ``row`` and ``col`` columns.
        px: Pixels per tile in the mosaic.
        **style: Passed to ``Rectangle`` (edge colour, line width).
    """
    for r in tiles.itertuples():
        ax.add_patch(Rectangle((r.col * px - 0.5, r.row * px - 0.5), px, px, fill=False, **style))


def probe_search(
    feats: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, name: str
) -> Probe:
    """``probes.fit_probe``, widened along ``C_LADDER`` while the kept C is at an edge.

    ``fit_probe`` tries ``probes.C_GRID``. If its pick is the grid's smallest or largest C, the
    next C of the ladder past that edge is fitted too (same scaler, same solver), and so on until
    the kept C has a tried value on both sides. When the pick is interior, the result is
    ``fit_probe``'s exactly.

    Args:
        feats: Features, one row per tile.
        y: 1 if cracked.
        train: Boolean mask of training rows.
        val: Boolean mask of validation rows.
        name: What the probe is, for messages.

    Returns:
        The probe with the best validation AP, its C interior to the values tried.

    Raises:
        ValueError: If the search runs off the ladder's end.
    """
    probe = fit_probe(feats, y, train, val)
    tried = list(C_GRID)
    zt = zv = None
    while probe.C in (tried[0], tried[-1]):
        kept = probe.C
        k = C_LADDER.index(kept) + (-1 if kept <= tried[0] else 1)
        if not 0 <= k < len(C_LADDER):
            raise ValueError(f"{name}: C = {probe.C} is at the end of {C_LADDER}")
        if zt is None:
            zt, zv = probe.scaler.transform(feats[train]), probe.scaler.transform(feats[val])
        c = C_LADDER[k]
        m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000).fit(zt, y[train])
        ap = float(average_precision_score(y[val], m.predict_proba(zv)[:, 1]))
        print(f"{name}: C = {probe.C} at an edge; C = {c} gives validation AP {ap:.4f} "
              f"against {probe.val_ap:.4f}", flush=True)  # fmt: skip
        tried = sorted([*tried, c])
        if ap > probe.val_ap:
            probe = Probe(probe.scaler, m, c, ap)
    return probe


# ---------------------------------------------------------------------------------------------
# The question and the one picture


def photos_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Three photographs rebuilt from their tiles, the cracked tiles outlined.

    Args:
        df: The tile index.
        out_dir: Figure directory.

    Returns:
        Per photo, its cracked tiles and its tiles.
    """
    px = MOSAIC_PX
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.04})
    shares = {}
    for ax, (surface, photo) in zip(axes, SHOW_PHOTOS.items(), strict=True):
        img, tiles = mosaic(df, photo, px)
        ax.imshow(img, interpolation="lanczos")
        outline_tiles(ax, tiles[tiles["cracked"] == 1], px, edgecolor="#D55E00", linewidth=0.9)
        n_c, n = int(tiles["cracked"].sum()), len(tiles)
        shares[photo] = {"cracked": n_c, "tiles": n}
        ax.set_title(f"{surface.capitalize()}: {n_c} of {n} cracked", fontsize=12)
        ax.set_axis_off()
    aml_style.save(fig, out_dir, "photos")
    return shares


def filter_build(out_dir: Path) -> dict[str, object]:
    """The one picture, in two frames: a 3 x 3 filter at one pixel, then over the whole tile.

    Panels: the tile with a zoom box; the zoomed pixels with the 3 x 3 window; the arithmetic
    (pixel values times weights, summed); and (frame 2) the response at every pixel.

    Args:
        out_dir: Figure directory.

    Returns:
        The window's pixel, its 3 x 3 grey levels, the weighted sum times 8 and divided by 8,
        and the tile's largest response.
    """
    g = read_tile(FILTER_TILE, gray=True)
    resp = correlate(g, DARK_LINE, mode="nearest")
    # the window: the strongest response near the crack's middle, away from the borders
    inner = resp[96:160, 96:160]
    i, j = np.unravel_index(np.argmax(inner), inner.shape)
    ci, cj = int(i) + 96, int(j) + 96
    patch = g[ci - 1 : ci + 2, cj - 1 : cj + 2]
    value = float((patch * DARK_LINE).sum())
    assert np.isclose(value, resp[ci, cj])
    assert np.isclose(8 * value, round(8 * value))  # integer pixels: the sum is n / 8 exactly
    z = 8  # zoom half-width, px
    for frame in (1, 2):
        fig, axes = plt.subplots(
            1, 4, figsize=aml_style.FULL, gridspec_kw={"width_ratios": [1, 1, 1.25, 1]}
        )
        ax = axes[0]
        ax.imshow(g, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(Rectangle((cj - z, ci - z), 2 * z, 2 * z, fill=False, ec="#D55E00", lw=2))
        ax.set_title("Deck tile, 256 px", fontsize=13)
        ax = axes[1]
        ax.imshow(g[ci - z : ci + z, cj - z : cj + z], cmap="gray", vmin=0, vmax=255)
        ax.add_patch(Rectangle((z - 1.5, z - 1.5), 3, 3, fill=False, ec="#D55E00", lw=2.5))
        ax.set_title(f"Zoom, {2 * z} x {2 * z} px", fontsize=13)
        ax = axes[2]
        _arithmetic(ax, patch, value)
        ax = axes[3]
        if frame == 2:
            ax.imshow(np.clip(resp, 0, None), cmap="magma", vmin=0, vmax=np.percentile(resp, 99.5))
            ax.plot(cj, ci, marker="o", ms=9, mfc="none", mec="white", mew=2)
            ax.set_title("Response everywhere", fontsize=13)
        else:
            ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=40, color=aml_style.MUTED)
            ax.set_title("Response everywhere", fontsize=13, color=aml_style.MUTED)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        aml_style.save(fig, out_dir, f"filter_{frame}")
    return {
        "pixel": [ci, cj],
        "patch": patch.astype(int).tolist(),
        "sum8": round(8 * value),
        "value": value,
        "max_response": round(float(resp.max()), 1),
    }


def _arithmetic(ax: plt.Axes, patch: np.ndarray, value: float) -> None:
    """Draw the 3 x 3 pixel values, the filter's weights, and their weighted sum.

    Args:
        ax: The axes to draw on.
        patch: The 3 x 3 grey levels.
        value: Their weighted sum (the filter's response).
    """
    ax.set_xlim(0, 7.4)
    ax.set_ylim(-1.6, 3.3)
    ax.set_aspect("equal")
    for k, (grid, x0, fmt) in enumerate(((patch, 0.0, "{:.0f}"), (DARK_LINE * 8, 4.0, "{:+.0f}"))):
        for (r, c), v in np.ndenumerate(grid):
            shade = v / 255 if k == 0 else 1.0
            face = (shade, shade, shade) if k == 0 else ("#FDE0C5" if v < 0 else "white")
            ax.add_patch(Rectangle((x0 + c, 2 - r), 1, 1, fc=face, ec=aml_style.MUTED, lw=0.8))
            dark = k == 0 and shade < 0.55
            ax.text(x0 + c + 0.5, 2.5 - r, fmt.format(v), ha="center", va="center",
                    fontsize=9.5, color="white" if dark else aml_style.INK)  # fmt: skip
    ax.text(3.5, 1.5, "×", ha="center", va="center", fontsize=20)
    ax.text(5.5, 3.12, "weights", ha="center", va="bottom", fontsize=11, color=aml_style.MUTED)
    ax.text(1.5, 3.12, "pixels", ha="center", va="bottom", fontsize=11, color=aml_style.MUTED)
    ax.text(3.7, -0.9, f"sum ÷ 8 = {8 * value:.0f} ÷ 8 = {value:.2f}", ha="center",
            va="center", fontsize=13, fontweight="bold", color="#D55E00")  # fmt: skip
    ax.set_axis_off()


def ridge_figure(out_dir: Path) -> dict[str, object]:
    """The filter's response as a 3D surface, on a deck tile and a pavement tile.

    The deck's crack becomes one ridge; the pavement's texture, a range that hides its crack.

    Args:
        out_dir: Figure directory.

    Returns:
        Per tile, the share of pixels whose response is above ``HAND_CUT``.
    """
    out = {}
    fig = plt.figure(figsize=aml_style.FULL)
    for k, (name, path) in enumerate((("deck", FILTER_TILE), ("pavement", TEXTURE_TILE))):
        g = read_tile(path, gray=True)
        resp = np.clip(correlate(g, DARK_LINE, mode="nearest"), 0, None)
        # 4 x 4 max-pooling keeps one-pixel ridges visible on a 64 x 64 mesh
        pooled = resp.reshape(64, 4, 64, 4).max(axis=(1, 3))
        cap = RIDGE_CAP
        zz = np.minimum(pooled, cap)
        yy, xx = np.mgrid[0:64, 0:64]
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        aml_style.surface3d(ax, xx, -yy, zz, cmap="magma", vmin=0, vmax=cap)
        aml_style.style3d(ax, elev=50, azim=-120)  # from six views: the crack crosses the view
        ax.set_zlim(0, cap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0, 20, 40])
        share = float((resp > HAND_CUT).mean())
        out[name] = {"share_above_10": round(share, 3)}
        ax.set_title(f"{name.capitalize()}: {100 * share:.0f}% of pixels above {HAND_CUT:.0f}",
                     fontsize=13, pad=14)  # fmt: skip
        ax.set_box_aspect((1, 1, 0.45), zoom=1.0)  # 1.15 cut the right box at the image edge
    # One label for both panels, beside the left one's ticks: matplotlib put each panel's z label
    # on the far side of its box, where the left one ran into the right panel.
    fig.text(0.06, 0.62, "response", rotation=90, ha="center", va="center", fontsize=12)
    aml_style.save(fig, out_dir, "ridges")
    return out


# ---------------------------------------------------------------------------------------------
# Build and verify: convolution from scratch


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A convolutional layer's forward pass (stride 1, no padding), as PyTorch defines it.

    Args:
        x: Inputs, shape (N, C, H, W).
        w: Filters, shape (F, C, k, k).
        b: Biases, shape (F,).

    Returns:
        The outputs (N, F, H - k + 1, W - k + 1) and the windows (N, C, H', W', k, k), kept
        for the backward pass.
    """
    k = w.shape[-1]
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
    return np.einsum("nchwij,fcij->nfhw", win, w) + b[None, :, None, None], win


def conv2d_back(
    g: np.ndarray, w: np.ndarray, win: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The backward pass of ``conv2d``: gradients of the loss for x, w, and b.

    Args:
        g: Gradient of the loss for the outputs, shape (N, F, H', W').
        w: The filters.
        win: The windows ``conv2d`` returned.

    Returns:
        dx (N, C, H, W), dw (F, C, k, k), db (F,).
    """
    n, c, ho, wo, k, _ = win.shape
    dw = np.einsum("nchwij,nfhw->fcij", win, g)
    db = g.sum(axis=(0, 2, 3))
    dx = np.zeros((n, c, ho + k - 1, wo + k - 1))
    for i in range(k):  # each filter tap sends the output gradient back to the pixels it read
        for j in range(k):
            dx[:, :, i : i + ho, j : j + wo] += np.einsum("nfhw,fc->nchw", g, w[:, :, i, j])
    return dx, dw, db


def verify_conv() -> dict[str, float]:
    """``conv2d`` and its backward pass against PyTorch, in float64, on random inputs.

    Returns:
        The largest absolute differences, and that of the flipped filter (true convolution).
    """
    rng = np.random.default_rng((SEED, 15, 7))
    x, w, b = rng.normal(size=(4, 3, 32, 32)), rng.normal(size=(8, 3, 3, 3)), rng.normal(size=8)
    y, win = conv2d(x, w, b)
    g = rng.normal(size=y.shape)
    dx, dw, db = conv2d_back(g, w, win)
    xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    yt = torch.nn.functional.conv2d(xt, wt, bt)
    yt.backward(torch.tensor(g))
    flipped, _ = conv2d(x, np.ascontiguousarray(w[:, :, ::-1, ::-1]), b)
    return {
        "y": float(np.abs(y - yt.detach().numpy()).max()),
        "dx": float(np.abs(dx - xt.grad.numpy()).max()),
        "dw": float(np.abs(dw - wt.grad.numpy()).max()),
        "db": float(np.abs(db - bt.grad.numpy()).max()),
        "flipped": float(np.abs(flipped - yt.detach().numpy()).max()),
        "y_sd": float(y.std()),
    }


def equivariance(shift: int = SHIFT_PX) -> float:
    """Shift the deck tile and filter it: the largest difference from the shifted map.

    Compared away from the borders, where the shifted tile has no wrapped pixels.

    Args:
        shift: Pixels to shift the tile to the right.

    Returns:
        The largest absolute difference.
    """
    g = read_tile(FILTER_TILE, gray=True)
    a = correlate(g, DARK_LINE, mode="nearest")
    b = correlate(np.roll(g, shift, axis=1), DARK_LINE, mode="nearest")
    return float(np.abs(np.roll(a, shift, axis=1) - b)[:, shift + 2 : -2].max())


# ---------------------------------------------------------------------------------------------
# Inside ResNet-18


def tiles_tensor(paths: list[str]) -> torch.Tensor:
    """ResNet-18 inputs for a list of tiles.

    Args:
        paths: Tile paths.

    Returns:
        Shape (n, 3, 224, 224), preprocessed.
    """
    return torch.stack([preprocess(read_tile(p)) for p in paths])


def theoretical_rf(model: nn.Module) -> dict[str, int]:
    """Receptive field, in input pixels, of one unit after each stage, along the main path.

    Each k x k layer with stride s widens the field by (k - 1) x the input pixels between
    neighbouring units (the jump), then multiplies the jump by s. Shortcuts are 1 x 1 or
    identity and widen nothing.

    Args:
        model: A torchvision ResNet-18.

    Returns:
        Field width per stage, ``"layer1"`` to ``"layer4"``.
    """
    rf, jump, out = 1, 1, {}

    def step(k: int, s: int) -> None:
        nonlocal rf, jump
        rf += (k - 1) * jump
        jump *= s

    step(model.conv1.kernel_size[0], model.conv1.stride[0])
    step(model.maxpool.kernel_size, model.maxpool.stride)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        for block in getattr(model, name):
            for conv in (block.conv1, block.conv2):
                step(conv.kernel_size[0], conv.stride[0])
        out[name] = rf
    return out


def effective_rf(model: nn.Module, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Effective receptive field (Luo et al., 2016) of each stage's centre unit.

    The gradient of the centre unit's summed channels with respect to the input pixels,
    in absolute value, summed over colour and averaged over the tiles in ``x``.

    Args:
        model: ResNet-18 in eval mode.
        x: Tiles, shape (n, 3, 224, 224).

    Returns:
        Per stage, a (224, 224) map that sums to 1.
    """
    out = {}
    for name in ("layer1", "layer2", "layer3", "layer4"):
        acts: dict[str, torch.Tensor] = {}
        hook = getattr(model, name).register_forward_hook(
            lambda _m, _i, o, acts=acts: acts.update(a=o)
        )
        xx = x.clone().requires_grad_(True)
        model(xx)
        hook.remove()
        a = acts["a"]  # (n, C, h, w)
        c = a.shape[-1] // 2
        # the gradient for the input only: no .grad is left on the backbone's weights
        (grad,) = torch.autograd.grad(a[:, :, c, c].sum(), xx)
        g = grad.abs().sum(1).mean(0).numpy()  # (224, 224)
        out[name] = g / g.sum()
    return out


def mass_width(g: np.ndarray, mass: float = 0.9) -> int:
    """Side, in pixels, of the smallest square about the map's centre of mass holding ``mass``.

    Args:
        g: A map that sums to 1.
        mass: The share of the map the square must hold.

    Returns:
        The square's side, in pixels.
    """
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    cy, cx = (g * yy).sum(), (g * xx).sum()
    d = np.maximum(np.abs(yy - cy), np.abs(xx - cx)).ravel()
    order = np.argsort(d, kind="stable")
    k = np.searchsorted(np.cumsum(g.ravel()[order]), mass)
    return int(round(2 * d[order][k] + 1))


def rf_figure(model: nn.Module, df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Each stage's receptive field over the deck tile, in theory and in effect.

    Dashed: the theoretical square. Solid: the square holding 90% of the gradient, averaged
    over ``ERF_TILES`` random tiles.

    Args:
        model: ResNet-18 in eval mode.
        df: The tile index.
        out_dir: Figure directory.

    Returns:
        Per stage, the theoretical and the effective (90%) field, in pixels.
    """
    rng = np.random.default_rng((SEED, 15, 1))
    paths = df["path"].to_numpy()[rng.choice(len(df), ERF_TILES, replace=False)].tolist()
    erf = effective_rf(model, tiles_tensor(paths))
    rf = theoretical_rf(model)
    base = tvf.resize(torch.from_numpy(read_tile(FILTER_TILE, gray=True))[None], [SIZE, SIZE])[0]
    fig, axes = plt.subplots(1, 4, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.08})
    out = {}
    for ax, (name, g) in zip(axes, erf.items(), strict=True):
        w = mass_width(g)
        out[name] = {"theory": rf[name], "effective90": w}
        ax.imshow(base.numpy(), cmap="gray", vmin=0, vmax=255)
        ax.imshow(np.sqrt(g / g.max()), cmap="magma", alpha=0.7, vmin=0, vmax=1)
        c = (SIZE - 1) / 2
        side = min(rf[name], SIZE + 40)
        ax.add_patch(Rectangle((c - side / 2, c - side / 2), side, side, fill=False,
                               ec="white", lw=1.6, ls="--", clip_on=True))  # fmt: skip
        ax.add_patch(Rectangle((c - w / 2, c - w / 2), w, w, fill=False, ec="#D55E00", lw=2))
        ax.set_xlim(-0.5, SIZE - 0.5)
        ax.set_ylim(SIZE - 0.5, -0.5)
        ax.set_title(f"{name}: {w} px", fontsize=13)
        ax.set_xlabel(f"theory {rf[name]} px (dashed)", fontsize=12, color=aml_style.MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
    aml_style.save(fig, out_dir, "receptive")
    return out


def conv1_figure(model: nn.Module, out_dir: Path) -> dict[str, object]:
    """ImageNet's 64 first-layer filters (7 x 7 x 3), and the two that follow the deck crack.

    Args:
        model: ResNet-18 in eval mode.
        out_dir: Figure directory.

    Returns:
        The two filters whose maps correlate best with the hand-made filter's, and their r.
    """
    w = model.conv1.weight.detach().numpy()  # (64, 3, 7, 7)
    x = preprocess(read_tile(FILTER_TILE))[None]
    with torch.no_grad():
        maps = torch.relu(model.bn1(model.conv1(x)))[0].numpy()  # (64, 112, 112)
    # the crack as the hand-made filter sees it, at the first layer's resolution (stride 2)
    g = tvf.resize(torch.from_numpy(read_tile(FILTER_TILE, gray=True))[None], [SIZE, SIZE])
    dark = np.clip(correlate(g[0].numpy(), DARK_LINE, mode="nearest"), 0, None)[::2, ::2]
    # a constant map has no correlation: it scores -1, last, as in stages_figure
    corr = np.array([np.corrcoef(m.ravel(), dark.ravel())[0, 1] if m.std() > 0 else -1.0
                     for m in maps])  # fmt: skip
    best = np.argsort(-corr, kind="stable")[:2]
    fig = plt.figure(figsize=aml_style.FULL)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.12)
    ax = fig.add_subplot(grid[0])
    sheet = np.ones((8 * 8, 8 * 8, 3))
    for k in range(64):
        f = w[k].transpose(1, 2, 0)
        f = (f - f.min()) / (np.ptp(f) + 1e-8)
        r, c = divmod(k, 8)
        sheet[r * 8 : r * 8 + 7, c * 8 : c * 8 + 7] = f
    ax.imshow(sheet, interpolation="nearest")
    for k in best:
        r, c = divmod(int(k), 8)
        box = Rectangle((c * 8 - 0.9, r * 8 - 0.9), 8.8, 8.8, fill=False, ec="#D55E00", lw=2)
        ax.add_patch(box)
    ax.set_title("64 filters, 7 x 7 x 3", fontsize=13)
    ax.set_axis_off()
    for j, k in enumerate(best):
        ax = fig.add_subplot(grid[j + 1])
        ax.imshow(maps[k], cmap="magma")
        ax.set_title(f"Filter {k}: r = {corr[k]:.2f}", fontsize=13)
        ax.set_axis_off()
    aml_style.save(fig, out_dir, "conv1")
    return {"best": [int(k) for k in best], "r": [round(float(corr[k]), 2) for k in best]}


def stages_figure(model: nn.Module, out_dir: Path) -> dict[str, object]:
    """The deck tile through ResNet-18: at each stage, the channel that best follows the crack.

    Each stage halves the map and (from layer2) doubles the channels; the pooled 512 numbers
    are what the logistic regression sees.

    Args:
        model: ResNet-18 in eval mode.
        out_dir: Figure directory.

    Returns:
        Each stage's map shape and best r, the weights per stage, and the total.
    """
    names = ("relu", "layer1", "layer2", "layer3", "layer4")
    acts: dict[str, torch.Tensor] = {}
    hooks = [getattr(model, n).register_forward_hook(lambda _m, _i, o, n=n: acts.update({n: o}))
             for n in names]  # fmt: skip
    with torch.no_grad():
        model(preprocess(read_tile(FILTER_TILE))[None])
    for h in hooks:
        h.remove()
    g = read_tile(FILTER_TILE, gray=True)
    dark = np.clip(correlate(g, DARK_LINE, mode="nearest"), 0, None)
    params = {n: sum(p.numel() for p in getattr(model, n).parameters()) for n in names[1:]}
    params["relu"] = model.conv1.weight.numel() + 2 * model.bn1.num_features
    fig, axes = plt.subplots(1, 6, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.1})
    axes[0].imshow(read_tile(FILTER_TILE))
    axes[0].set_title("3 x 224²", fontsize=12)
    out = {"shapes": {}, "r": {}, "params": params,
           "total": sum(p.numel() for p in model.parameters())}  # fmt: skip
    for ax, n in zip(axes[1:], names, strict=True):
        a = acts[n][0].numpy()  # (C, h, w)
        side = a.shape[-1]
        target = tvf.resize(torch.from_numpy(dark)[None], [side, side], antialias=True)[0]
        corr = [np.corrcoef(m.ravel(), target.numpy().ravel())[0, 1] if m.std() > 0 else -1
                for m in a]  # fmt: skip
        k = int(np.argmax(corr))
        ax.imshow(a[k], cmap="magma", interpolation="nearest")
        label = "conv1" if n == "relu" else n
        ax.set_title(f"{a.shape[0]} x {side}²", fontsize=12)
        ax.set_xlabel(f"{label}\nr = {corr[k]:.2f}", fontsize=11)
        out["shapes"][label] = [int(a.shape[0]), side, side]
        out["r"][label] = round(float(corr[k]), 2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    aml_style.save(fig, out_dir, "stages")
    return out


class _Block(nn.Module):
    """Two 3 x 3 convolutions with batch normalization, with or without the identity shortcut."""

    def __init__(self, channels: int, residual: bool) -> None:
        """Two conv-BN pairs.

        Args:
            channels: Channels in and out.
            residual: Add the block's input to its output (He et al., 2016).
        """
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1, self.b2 = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """relu(x + F(x)) or relu(F(x)).

        Args:
            x: Feature maps, shape (n, channels, h, w).

        Returns:
            The block's output, the same shape.
        """
        h = self.b2(self.c2(torch.relu(self.b1(self.c1(x)))))
        return torch.relu(x + h if self.residual else h)


def flow_tiles(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """The 64 tiles (half cracked) the gradient-flow toy network sees, at 128 px, standardized.

    Args:
        df: The tile index.

    Returns:
        Inputs (64, 3, 128, 128) and labels (64,).
    """
    rng = np.random.default_rng((SEED, 15, 2))
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(df["cracked"] == c), 32, replace=False) for c in (0, 1)]
    )
    x = torch.stack([
        tvf.resize(tvf.to_dtype(torch.from_numpy(read_tile(df["path"].iloc[i])).permute(2, 0, 1),
                                torch.float32, scale=True), [128, 128], antialias=True)
        for i in idx
    ])  # fmt: skip
    x = (x - x.mean((0, 2, 3), keepdim=True)) / x.std((0, 2, 3), keepdim=True)
    return x, torch.from_numpy(df["cracked"].to_numpy()[idx])


def gradient_flow(
    x: torch.Tensor, y: torch.Tensor, blocks: int, residual: bool, seed: int
) -> np.ndarray:
    """Size of the loss's gradient at each block's input, at initialization.

    A toy network on ``flow_tiles``: a stride-2 stem of 16 channels, then ``blocks`` blocks,
    global average pooling, and a linear classifier; PyTorch's default initialization, batch
    statistics (training mode).

    Args:
        x: Inputs from ``flow_tiles``.
        y: Their labels.
        blocks: Number of blocks.
        residual: With identity shortcuts.
        seed: Seeds the weights.

    Returns:
        The gradient norm at each block's input, divided by the last block's, shape (blocks,).
    """
    torch.manual_seed(seed)
    stem = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16),
                         nn.ReLU())  # fmt: skip
    body = [_Block(16, residual) for _ in range(blocks)]
    head = nn.Linear(16, 2)
    h = stem(x)
    inputs = []
    for b in body:
        h.retain_grad()
        inputs.append(h)
        h = b(h)
    torch.nn.functional.cross_entropy(head(h.mean((2, 3))), y).backward()
    g = np.array([v.grad.norm().item() for v in inputs])
    return g / g[-1]


def residual_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Gradient size through plain vs residual blocks, median of ``RESIDUAL_SEEDS`` seeds.

    The two blocks are drawn beside the plot. Each (depth, kind, seed) is run once; the deepest
    stacks' runs give both the curves and their first-over-last ratios.

    Args:
        df: The tile index.
        out_dir: Figure directory.

    Returns:
        The first block's gradient over the last block's, per depth and kind (median of seeds,
        one decimal), and the deepest stacks' ratios to the nearest integer.
    """
    depth = RESIDUAL_DEPTH
    x, y = flow_tiles(df)
    flows = {
        (d, res): [gradient_flow(x, y, d, res, s) for s in range(RESIDUAL_SEEDS)]
        for d in (8, 16, depth)
        for res in (False, True)
    }
    curves = {res: np.median(flows[depth, res], axis=0) for res in (False, True)}
    first = {key: float(np.median([f[0] for f in runs])) for key, runs in flows.items()}
    ratios = {
        f"{'res' if res else 'plain'}_{d}": round(first[d, res], 1)
        for d in (8, 16, depth)
        for res in (False, True)
    }
    fig = plt.figure(figsize=aml_style.FULL)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax = fig.add_subplot(grid[0])
    _blocks_schematic(ax)
    ax = fig.add_subplot(grid[1])
    k = np.arange(1, depth + 1)
    ax.semilogy(k, curves[False], color=aml_style.PALETTE[1], ls="--", marker="s", ms=4,
                label="plain")  # fmt: skip
    ax.semilogy(k, curves[True], color=aml_style.PALETTE[0], ls="-", marker="o", ms=4,
                label="residual")  # fmt: skip
    ax.set_xlabel("block, from the input side to the output side")
    ax.set_ylabel("gradient size ÷ last block's")
    ax.annotate(f"×{curves[False][0]:,.0f}", (1, curves[False][0]), xytext=(6, -4),
                textcoords="offset points", color=aml_style.PALETTE[1], fontsize=13)  # fmt: skip
    ax.annotate(f"×{curves[True][0]:.0f}", (1, curves[True][0]), xytext=(6, 6),
                textcoords="offset points", color=aml_style.PALETTE[0], fontsize=13)  # fmt: skip
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "residual")
    return {"first_over_last": ratios, "plain32": round(float(curves[False][0])),
            "res32": round(float(curves[True][0]))}  # fmt: skip


def _blocks_schematic(ax: plt.Axes) -> None:
    """A plain block, x -> F -> ReLU -> y, and a residual block, x -> F -> (+ x) -> ReLU -> y.

    Args:
        ax: The axes to draw on.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    arrow = {"arrowstyle": "->", "lw": 1.6, "color": aml_style.INK}
    for y0, label, res in ((7.0, "plain", False), (1.8, "residual", True)):
        color = aml_style.PALETTE[0 if res else 1]
        ax.text(0.0, y0 + 2.2, label, fontsize=13, fontweight="bold", color=color)
        ax.text(0.15, y0, "x", fontsize=15, style="italic", va="center")
        ax.annotate("", (1.9, y0), (0.6, y0), arrowprops=arrow)
        ax.add_patch(Rectangle((1.9, y0 - 0.8), 4.0, 1.6, fc="#EEF3F8", ec=aml_style.NAVY, lw=1.4))
        ax.text(3.9, y0, "conv, BN, ReLU,\nconv, BN", ha="center", va="center", fontsize=9)
        out_x = 5.9
        if res:
            ax.add_patch(plt.Circle((7.0, y0), 0.42, fc="white", ec=aml_style.INK, lw=1.4))
            ax.text(7.0, y0 - 0.04, "+", ha="center", va="center", fontsize=15)
            ax.annotate("", (6.58, y0), (5.9, y0), arrowprops=arrow)
            top = y0 + 1.35
            ax.plot([1.2, 1.2, 7.0], [y0, top, top], color=color, lw=1.8)
            ax.annotate("", (7.0, y0 + 0.42), (7.0, top), arrowprops={**arrow, "color": color})
            ax.text(4.1, top + 0.15, "identity", ha="center", va="bottom", fontsize=11, color=color)
            out_x = 7.42
        ax.annotate("", (8.6, y0), (out_x, y0), arrowprops=arrow)
        ax.text(7.9 if not res else 8.2, y0 + 0.25, "ReLU", ha="center", va="bottom", fontsize=10,
                color=aml_style.MUTED)  # fmt: skip
        ax.text(8.85, y0, "y", fontsize=15, style="italic", va="center")


# ---------------------------------------------------------------------------------------------
# Classifiers: the transfer probe and the hand-filter baseline


def _hand_features(path: str) -> list[float]:
    """The hand filter's summary of one tile: share of line-like pixels, mean, 99th percentile.

    Args:
        path: A tile.

    Returns:
        The three summaries.
    """
    resp = np.clip(correlate(read_tile(path, gray=True), DARK_LINE, mode="nearest"), 0, None)
    return [float((resp > HAND_CUT).mean()), float(resp.mean()), float(np.percentile(resp, 99))]


def hand_features(df: pd.DataFrame) -> np.ndarray:
    """``_hand_features`` for every tile, in parallel processes.

    Args:
        df: The tile index.

    Returns:
        Shape (n, 3), in the index's order.
    """
    with ProcessPoolExecutor(WORKERS) as pool:
        return np.array(list(pool.map(_hand_features, df["path"], chunksize=512)))


def _tile_at(job: tuple[str, int]) -> np.ndarray:
    """A tile shrunk to 256 / factor px (box filter).

    Args:
        job: The tile's path and the factor.

    Returns:
        uint8 (h, w, 3).
    """
    path, factor = job
    img = read_tile(path)
    if factor > 1:
        img = np.array(Image.fromarray(img).resize((256 // factor,) * 2, Image.Resampling.BOX))
    return img


def embed(model: nn.Module, paths: list[str], factor: int = 1) -> np.ndarray:
    """ResNet-18 features of tiles, optionally seen from ``factor`` times farther away.

    Tiles are decoded in worker processes; the network runs on batches of 128, on a copy of
    ``model`` in channels-last memory format (the caller's model is left as it was).

    Args:
        model: The backbone in eval mode.
        paths: Tile paths.
        factor: Shrink each 256-px tile to 256 / factor px first (fewer pixels per millimetre,
            as from a camera ``factor`` times farther), then preprocess as usual.

    Returns:
        float32 features, shape (len(paths), 512).
    """
    model = copy.deepcopy(model).to(memory_format=torch.channels_last)
    out = []
    with ProcessPoolExecutor(WORKERS) as pool, torch.inference_mode():
        for i in range(0, len(paths), 2048):  # a multiple of 128: the same batches as one by one
            jobs = [(q, factor) for q in paths[i : i + 2048]]
            imgs = list(pool.map(_tile_at, jobs, chunksize=64))
            for j in range(0, len(imgs), 128):
                x = torch.stack([preprocess(im) for im in imgs[j : j + 128]])
                out.append(model(x.to(memory_format=torch.channels_last)).numpy())
    feats = np.concatenate(out).astype(np.float32)
    assert feats.shape == (len(paths), 512), feats.shape
    return feats


# ---------------------------------------------------------------------------------------------
# Results and break-it tests


@dataclass(frozen=True)
class ProbeRun:
    """The probe's numbers for the slides, its probabilities for every tile, and the probe."""

    numbers: dict[str, object]
    p: np.ndarray
    probe: Probe


def results(df: pd.DataFrame, feats: np.ndarray, part: np.ndarray) -> ProbeRun:
    """The probe on photo-split data, the hand-filter baseline, and the trivial baseline.

    Args:
        df: The tile index.
        feats: ResNet-18 features.
        part: The photo split.

    Returns:
        The numbers, the probabilities, and the probe.
    """
    y = df["cracked"].to_numpy()
    train, val, test = (part == r for r in ("train", "val", "test"))
    probe = probe_search(feats, y, train, val, "probe")
    p = probe.score(feats)
    t = threshold_for(p[val], y[val])
    hand = hand_features(df)
    hprobe = probe_search(hand, y, train, val, "hand-filter probe")
    ph = hprobe.score(hand)
    th = threshold_for(ph[val], y[val])
    half = p[test] >= 0.5  # the default threshold, against the inspector's
    out: dict[str, object] = {
        "counts": {
            r: {"tiles": int((part == r).sum()), "photos": int(df["photo"][part == r].nunique())}
            for r in ("train", "val", "test")
        },
        "C": probe.C,
        "val_ap": probe.val_ap,
        "threshold": t,
        "probe": {"all": scores(p[test], y[test], t)},
        "hand": {"all": scores(ph[test], y[test], th), "C": hprobe.C},
        "all_uncracked_acc": float(1 - y[test].mean()),
        "at_half": {"recall": float(half[y[test] == 1].mean()), "review": float(half.mean())},
        "bootstrap": photo_bootstrap(df, part, p, t),
    }
    for s in SURFACES:
        m = test & (df["surface"] == s).to_numpy()
        out["probe"][s] = scores(p[m], y[m], t)
        out["hand"][s] = scores(ph[m], y[m], th)
    return ProbeRun(out, p, probe)


def photo_bootstrap(df: pd.DataFrame, part: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Photo-level 95% intervals of the probe's test recall, review share, and AP.

    The 46 test photos are the unit of collection, so whole photos are resampled, within each
    surface (``BOOT_DRAWS`` draws). The threshold stays fixed: its own noise, from the validation
    photos, is what the five photo splits show.

    Args:
        df: The tile index.
        part: The photo split.
        p: The probe's probability for every tile.
        t: The threshold set on validation.

    Returns:
        Per measure, the 2.5th and 97.5th percentiles; the draws; and the share of draws whose
        recall reaches the 90% target.
    """
    y = df["cracked"].to_numpy()
    test = part == "test"
    photos, surface = df["photo"].to_numpy(), df["surface"].to_numpy()
    by = {s: sorted(set(photos[test & (surface == s)])) for s in SURFACES}
    rows_of = {ph: np.flatnonzero(test & (photos == ph)) for s in SURFACES for ph in by[s]}
    rng = np.random.default_rng((SEED, 15, 9))
    draws: dict[str, list[float]] = {"recall": [], "review": [], "ap": []}
    for _ in range(BOOT_DRAWS):
        picks = [ph for s in SURFACES for ph in rng.choice(by[s], len(by[s]))]
        rows = np.concatenate([rows_of[ph] for ph in picks])
        yy, pp = y[rows], p[rows]
        draws["recall"].append(float(np.mean(pp[yy == 1] >= t)))
        draws["review"].append(float(np.mean(pp >= t)))
        draws["ap"].append(float(average_precision_score(yy, pp)))
    out: dict[str, object] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in draws.items()}
    out["draws"] = BOOT_DRAWS
    out["share_recall_90"] = float(np.mean(np.array(draws["recall"]) >= 0.9))
    return out


def split_spread(df: pd.DataFrame, feats: np.ndarray, splits: int = 5) -> dict[str, object]:
    """The probe on several photo splits: how much the score depends on which photos test it.

    Split k shuffles the photos with the seed (k, 15); split 0 is the lecture's.

    Args:
        df: The tile index.
        feats: ResNet-18 features.
        splits: How many photo splits.

    Returns:
        Test AUC, AP, recall, and review share (at each split's validation threshold), and C,
        per split.
    """
    y = df["cracked"].to_numpy()
    rows = []
    for k in range(splits):
        part = photo_split(df, np.random.default_rng((k, 15)))
        train, val, test = (part == r for r in ("train", "val", "test"))
        probe = probe_search(feats, y, train, val, f"split {k}")
        p = probe.score(feats)
        rows.append(scores(p[test], y[test], threshold_for(p[val], y[val])) | {"C": probe.C})
    return {key: [r[key] for r in rows] for key in ("auc", "ap", "recall", "review", "C")}


def probe_budgets(df: pd.DataFrame, feats: np.ndarray, feats128: np.ndarray,
                  part: np.ndarray) -> dict[str, list]:  # fmt: skip
    """The probe trained on a share of the training photos, on 256-px and on 128-px tiles.

    Validation photos still choose C (they are labelled too). The curves are drawn on
    validation AP; each budget's test AP is that budget's single test, at its own C. The
    128-px probe sees the resolution the small network trained from scratch sees.

    Args:
        df: The tile index.
        feats: ResNet-18 features of the 256-px tiles.
        feats128: ResNet-18 features of the tiles shrunk to 128 px.
        part: The photo split.

    Returns:
        Per budget: photos, tiles, and validation and test AP of both probes.
    """
    y = df["cracked"].to_numpy()
    out: dict[str, list] = {"frac": list(BUDGETS), "photos": [], "tiles": []}
    for key in ("probe", "probe128"):
        out[f"{key}_val_ap"], out[f"{key}_test_ap"] = [], []
    for frac in BUDGETS:
        use = budget_photos(df, part, frac)
        out["photos"].append(int(df["photo"][use].nunique()))
        out["tiles"].append(int(use.sum()))
        for key, f in (("probe", feats), ("probe128", feats128)):
            probe = probe_search(f, y, use, part == "val", f"{key}, {frac:.0%} of the photos")
            p = probe.score(f[part == "test"])
            out[f"{key}_val_ap"].append(probe.val_ap)
            out[f"{key}_test_ap"].append(float(average_precision_score(y[part == "test"], p)))
    return out


def probe128(df: pd.DataFrame, feats128: np.ndarray, part: np.ndarray) -> dict[str, object]:
    """The probe on 128-px tiles, all training photos: C and threshold set on validation.

    Args:
        df: The tile index.
        feats128: ResNet-18 features of the tiles shrunk to 128 px.
        part: The photo split.

    Returns:
        C, validation AP, threshold, and the test scores at that threshold.
    """
    y = df["cracked"].to_numpy()
    train, val, test = (part == r for r in ("train", "val", "test"))
    probe = probe_search(feats128, y, train, val, "128-px probe")
    p = probe.score(feats128)
    t = threshold_for(p[val], y[val])
    return {"C": probe.C, "val_ap": probe.val_ap, "threshold": t,
            "test": scores(p[test], y[test], t)}  # fmt: skip


def leakage(df: pd.DataFrame, feats: np.ndarray, part: np.ndarray, c: float) -> dict[str, object]:
    """Does seeing other tiles of the test photos help?

    The same held-out tiles, scored by models trained without and with the other half of each
    test photo's tiles; and, as the control for more data alone, with as many random tiles of
    the validation photos instead (photos the held-out tiles do not share), ``CONTROL_DRAWS``
    draws.

    Args:
        df: The tile index.
        feats: Features.
        part: The photo split.
        c: The probe's C (chosen on validation photos).

    Returns:
        AP and AUC of the probe and of 5-nearest neighbours, each way, the control's APs per
        draw, and the neighbouring tiles' labels' AUC as a score.
    """
    y = df["cracked"].to_numpy()
    test = np.flatnonzero(part == "test")
    half = np.random.default_rng((SEED, 15, 3)).random(len(test)) < 0.5
    sibling, held = test[half], test[~half]
    train = np.flatnonzero(part == "train")
    val = np.flatnonzero(part == "val")

    def fit_score(rows: np.ndarray) -> dict[str, float]:
        scaler = StandardScaler().fit(feats[rows])
        z, zh = scaler.transform(feats[rows]), scaler.transform(feats[held])
        lr = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000).fit(z, y[rows])
        knn = KNeighborsClassifier(n_neighbors=5).fit(z, y[rows])
        p, pk = lr.predict_proba(zh)[:, 1], knn.predict_proba(zh)[:, 1]
        return {"probe_ap": float(average_precision_score(y[held], p)),
                "probe_auc": float(roc_auc_score(y[held], p)),
                "knn_ap": float(average_precision_score(y[held], pk))}  # fmt: skip

    out: dict[str, object] = {"held": len(held), "sibling_tiles": len(sibling)}
    for name, rows in (("photo", train), ("siblings", np.concatenate([train, sibling]))):
        out[name] = fit_score(rows)
    control = [
        fit_score(np.concatenate([train, np.random.default_rng((d, 15, 99)).choice(
            val, len(sibling), replace=False)]))
        for d in range(CONTROL_DRAWS)
    ]  # fmt: skip
    out["more_data"] = {k: [r[k] for r in control] for k in ("probe_ap", "knn_ap")}
    # the labels of a held tile's eight neighbours, as a score: cracks run across tiles
    lab = {(r.photo, r.row, r.col): r.cracked for r in df[df["dup"] == 0].itertuples()}
    nb = []
    for r in df.iloc[held].itertuples():
        v = [lab.get((r.photo, r.row + a, r.col + b)) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        v = [x for k, x in enumerate(v) if k != 4 and x is not None]
        nb.append(np.mean(v))
    out["neighbours_auc"] = float(roc_auc_score(y[held], nb))
    out["gain_probe"] = out["siblings"]["probe_ap"] - out["photo"]["probe_ap"]
    out["gain_knn"] = out["siblings"]["knn_ap"] - out["photo"]["knn_ap"]
    for k in ("probe", "knn"):  # the control's gains, and the siblings' gain beyond the median
        gains = [v - out["photo"][f"{k}_ap"] for v in out["more_data"][f"{k}_ap"]]
        out["more_data"][f"gain_{k}"] = gains
        out[f"leak_{k}"] = out[f"gain_{k}"] - float(np.median(gains))
    out["next_photo"] = nearest_photos(df, feats, part)
    return out


def nearest_photos(df: pd.DataFrame, feats: np.ndarray, part: np.ndarray) -> dict[str, int]:
    """For each test photo, the training photo whose mean feature is closest (cosine).

    Args:
        df: The tile index.
        feats: Features.
        part: The photo split.

    Returns:
        How many test photos there are, and for how many the nearest training photo's number is
        one more or one less (photos taken in sequence, often of the same slab).
    """
    z = StandardScaler().fit(feats[part == "train"]).transform(feats)
    mean = pd.DataFrame(z).groupby(df["photo"].to_numpy()).mean()
    mean = mean.div(np.linalg.norm(mean.to_numpy(), axis=1), axis=0)
    role = pd.Series(part).groupby(df["photo"].to_numpy()).first()
    te, tr = mean[role == "test"], mean[role == "train"]
    nearest = (te @ tr.T).idxmax(axis=1)
    number = {ph: int(ph[1:]) for ph in mean.index}
    same = [
        nearest[ph][0] == ph[0] and abs(number[nearest[ph]] - number[ph]) == 1 for ph in te.index
    ]
    return {"test_photos": len(te), "next_number": int(sum(same))}


def cross_surface(
    df: pd.DataFrame, feats: np.ndarray, part: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Train on one surface's photos, test on each surface's test photos: a 3 x 3 grid of AP.

    Args:
        df: The tile index.
        feats: Features.
        part: The photo split.
        out_dir: Figure directory.

    Returns:
        The grid of test AP (rows trained on, columns tested on), each probe's C, and each
        probe's recall and false-alarm rate on the pavement test tiles at its own threshold.
    """
    y = df["cracked"].to_numpy()
    grid = np.zeros((3, 3))
    cs = {}
    on_pavement = {}  # each probe at its own surface's threshold, on the pavement test tiles
    pave = (part == "test") & (df["surface"] == "pavement").to_numpy()
    for i, s_tr in enumerate(SURFACES):
        on = (df["surface"] == s_tr).to_numpy()
        probe = probe_search(feats, y, (part == "train") & on, (part == "val") & on, s_tr)
        cs[s_tr] = probe.C
        p = probe.score(feats)
        for j, s_te in enumerate(SURFACES):
            m = (part == "test") & (df["surface"] == s_te).to_numpy()
            grid[i, j] = average_precision_score(y[m], p[m])
        t = threshold_for(p[(part == "val") & on], y[(part == "val") & on])
        on_pavement[s_tr] = {"recall": float((p[pave & (y == 1)] >= t).mean()),
                             "false_alarm": float((p[pave & (y == 0)] >= t).mean())}  # fmt: skip
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.imshow(grid, cmap="cividis", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=16,
                color="white" if v < 0.55 else aml_style.INK,
                fontweight="bold" if i == j else "normal")  # fmt: skip
    ax.set_xticks(range(3), [s.capitalize() for s in SURFACES])
    ax.set_yticks(range(3), [s.capitalize() for s in SURFACES])
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    aml_style.save(fig, out_dir, "cross_surface")
    return {"ap": grid.tolist(), "C": cs, "on_pavement": on_pavement}


def resolution(
    model: nn.Module, df: pd.DataFrame, part: np.ndarray, probe: Probe, *, t: float,
    feats: np.ndarray, out_dir: Path,
) -> dict[str, object]:  # fmt: skip
    """Fly higher: the same test tiles with half and a quarter of the pixels per millimetre.

    Every cracked test tile and as many uncracked ones (a seeded sample) are shrunk, re-embedded,
    and scored by the unchanged probe at the unchanged threshold. The AUC at each scale tells a
    shift of every score (ranking kept) from a lost ranking.

    Args:
        model: ResNet-18 in eval mode.
        df: The tile index.
        part: The photo split.
        probe: The probe fitted on full-resolution features.
        t: Its threshold.
        feats: The cached full-resolution features.
        out_dir: Figure directory.

    Returns:
        Per scale, recall, false-alarm rate, and AUC on the sampled tiles; and the example tile.
    """
    y = df["cracked"].to_numpy()
    test = np.flatnonzero(part == "test")
    pos = test[y[test] == 1]
    rng = np.random.default_rng((SEED, 15, 4))
    neg = np.sort(rng.choice(test[y[test] == 0], len(pos), replace=False))
    rows = np.concatenate([pos, neg])
    paths = df["path"].to_numpy()[rows].tolist()
    # the cache was made by scripts/derive_sdnet2018.py; the same preprocessing here must agree
    check = embed(model, paths[:256])
    gap = float(np.abs(check - feats[rows[:256]]).max())
    if gap >= 1e-3:
        raise ValueError(f"features differ from the cache by {gap}")
    out: dict[str, object] = {"cracked": len(pos), "uncracked": len(neg), "cache_gap": gap}
    ps = {}
    for f in (1, 2, 4):
        ps[f] = p = probe.score(feats[rows] if f == 1 else embed(model, paths, f))
        out[f"x{f}"] = {"px": 256 // f, "recall": float((p[: len(pos)] >= t).mean()),
                        "false_alarm": float((p[len(pos) :] >= t).mean()),
                        "auc": float(roc_auc_score(y[rows], p))}  # fmt: skip
    # the example: the cracked tile caught at 256 px whose probability falls most at 64 px
    lost = np.flatnonzero((ps[1][: len(pos)] >= t) & (ps[4][: len(pos)] < t))
    k = int(lost[np.argmax(ps[1][lost] - ps[4][lost])])
    example = paths[k]
    out["example"] = {"path": example, "p": [float(ps[f][k]) for f in (1, 2, 4)],
                      "lost": len(lost)}  # fmt: skip
    fig = plt.figure(figsize=aml_style.FULL)
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.35, 2.2], wspace=0.15)
    img = read_tile(example)
    for j, f in enumerate((1, 2, 4)):
        ax = fig.add_subplot(grid[j])
        small = Image.fromarray(img).resize((256 // f,) * 2, Image.Resampling.BOX)
        ax.imshow(np.array(small), interpolation="nearest")
        ax.set_title(f"{256 // f} px", fontsize=13)
        ax.set_xlabel(f"p = {ps[f][k]:.2f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(grid[4])
    px = [256, 128, 64]
    rec = [out[f"x{f}"]["recall"] for f in (1, 2, 4)]
    fa = [out[f"x{f}"]["false_alarm"] for f in (1, 2, 4)]
    ax.plot(px, rec, marker="o", color=aml_style.PALETTE[0], label="cracked tiles flagged")
    ax.plot(px, fa, marker="s", ls="--", color=aml_style.PALETTE[1], label="clean tiles flagged")
    for x, a, b in zip(px, rec, fa, strict=True):
        ax.annotate(f"{100 * a:.0f}%", (x, a), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=aml_style.PALETTE[0], fontsize=12)  # fmt: skip
        ax.annotate(f"{100 * b:.0f}%", (x, b), xytext=(0, -18), textcoords="offset points",
                    ha="center", color=aml_style.PALETTE[1], fontsize=12)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks(px, [str(v) for v in px])
    ax.set_xlim(300, 54)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("pixels across a tile")
    ax.set_ylabel("share flagged")
    ax.legend(loc="lower left", fontsize=11)
    aml_style.save(fig, out_dir, "resolution")
    return out


def occlusion_map(model: nn.Module, probe: Probe, path: str, size: int = OCCLUSION_PX,
                  stride: int = OCCLUSION_STRIDE) -> tuple[float, np.ndarray]:  # fmt: skip
    """How much the log-odds of "cracked" fall when a square of the tile is greyed out.

    Log-odds rather than probability: near p = 1 a probability cannot show a drop.

    Args:
        model: The backbone.
        probe: The probe on its features.
        path: A tile.
        size: Side of the grey square, px.
        stride: Step between squares, px.

    Returns:
        The tile's probability and the drop in log-odds at each square's position.
    """
    img = read_tile(path)
    grey = np.round(img.reshape(-1, 3).mean(0)).astype(np.uint8)  # the tile's mean colour
    n = (256 - size) // stride + 1
    batch = [preprocess(img)]
    for i in range(n):
        for j in range(n):
            occ = img.copy()
            occ[i * stride : i * stride + size, j * stride : j * stride + size] = grey
            batch.append(preprocess(occ))
    with torch.inference_mode():
        f = model(torch.stack(batch)).numpy()
    logit = probe.logit(f)  # the decision function: exact where p would round to 1
    return float(probe.score(f)[0]), (logit[0] - logit[1:]).reshape(n, n)


def pick_tiles(df: pd.DataFrame, part: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Eight test tiles for the attempt: two of each kind, from different surfaces.

    Taken at fixed ranks of the probe's probability among the test tiles of one surface and one
    label, so the choice is reproducible: sure catches (deck, wall), sure passes (pavement,
    deck), false alarms (pavement, wall), and misses (pavement, deck).

    Args:
        df: The tile index.
        part: The photo split.
        p: The probe's probability for every tile.

    Returns:
        The eight tiles in a seeded random order, with ``p`` and a letter A-H.
    """
    y = df["cracked"].to_numpy()
    test = part == "test"
    want = [  # (surface, label, quantile of p within that surface and label)
        ("deck", 1, 0.95), ("wall", 1, 0.95),  # caught
        ("pavement", 0, 0.05), ("deck", 0, 0.05),  # passed
        ("pavement", 0, 0.995), ("wall", 0, 0.995),  # false alarms
        ("pavement", 1, 0.05), ("deck", 1, 0.05),  # misses
    ]  # fmt: skip
    rows = []
    for surface, label, q in want:
        m = np.flatnonzero(test & (y == label) & (df["surface"] == surface).to_numpy())
        order = m[np.argsort(p[m], kind="stable")]
        rows.append(int(order[int(round(q * (len(order) - 1)))]))
    rng = np.random.default_rng((SEED, 15, 5))
    rows = list(rng.permutation(rows))
    out = df.iloc[rows].copy()
    out["p"] = p[rows]
    out["letter"] = list("ABCDEFGH")
    return out


def attempt_figures(tiles: pd.DataFrame, out_dir: Path) -> None:
    """The attempt (eight tiles, A-H) and its reveal (label and probability under each).

    Args:
        tiles: ``pick_tiles``.
        out_dir: Figure directory.
    """
    for name, reveal in (("attempt", False), ("attempt_reveal", True)):
        fig, axes = plt.subplots(2, 4, figsize=aml_style.FULL, gridspec_kw={"hspace": 0.35})
        for ax, r in zip(axes.ravel(), tiles.itertuples(), strict=True):
            ax.imshow(read_tile(r.path))
            ax.set_xticks([])
            ax.set_yticks([])
            title = f"{r.letter}"
            if reveal:
                lab = "cracked" if r.cracked else "uncracked"
                title = f"{r.letter}: {lab}, p = {r.p:.2f}"
                right = (r.p >= 0.5) == bool(r.cracked)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(aml_style.PALETTE[0] if right else aml_style.PALETTE[1])
                    sp.set_linewidth(3 if right else 4)
                    sp.set_linestyle("-" if right else "--")
            ax.set_title(title, fontsize=12 if reveal else 14)
        aml_style.save(fig, out_dir, name)


def occlusion_figure(model: nn.Module, probe: Probe, tiles: pd.DataFrame,
                     out_dir: Path) -> dict[str, object]:  # fmt: skip
    """Where the network looks, on the four attempt tiles it flags (p >= 0.5).

    Each 48-px square of the tile is greyed out in turn; orange marks where the log-odds of
    "cracked" fall, the stronger the larger the fall.

    Args:
        model: ResNet-18 in eval mode.
        probe: The probe.
        tiles: ``pick_tiles``.
        out_dir: Figure directory.

    Returns:
        Per flagged tile, its probability, label, and largest drop in log-odds.
    """
    show = tiles[tiles["p"] >= 0.5].sort_values("letter")
    if len(show) != 4:  # the slide shows four: a catch and a false alarm on two surfaces each
        raise ValueError(f"{len(show)} attempt tiles at p >= 0.5, not 4")
    fig, axes = plt.subplots(1, 4, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.08})
    out = {}
    orange = np.array([0xD5, 0x5E, 0x00]) / 255
    for ax, r in zip(axes, show.itertuples(), strict=True):
        p0, drop = occlusion_map(model, probe, r.path)
        out[r.letter] = {"p": p0, "cracked": int(r.cracked), "max_drop": float(drop.max())}
        # each square's drop, spread over the pixels it covered (averaged where squares overlap)
        acc, cnt = np.zeros((256, 256)), np.zeros((256, 256))
        for (i, j), d in np.ndenumerate(drop):
            r0, c0 = i * OCCLUSION_STRIDE, j * OCCLUSION_STRIDE
            acc[r0 : r0 + OCCLUSION_PX, c0 : c0 + OCCLUSION_PX] += d
            cnt[r0 : r0 + OCCLUSION_PX, c0 : c0 + OCCLUSION_PX] += 1
        heat = np.clip(acc / np.maximum(cnt, 1) / max(float(drop.max()), 1e-6), 0, 1)
        ax.imshow(read_tile(r.path, gray=True), cmap="gray", vmin=0, vmax=255)
        rgba = np.dstack([np.broadcast_to(orange, (256, 256, 3)), 0.8 * heat])
        ax.imshow(rgba)
        lab = "cracked" if r.cracked else "uncracked"
        ax.set_title(f"{r.letter}: {lab}\np = {p0:.2f}", fontsize=12)
        ax.set_xlabel(f"largest drop {drop.max():.1f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    aml_style.save(fig, out_dir, "occlusion")
    return out


def twins_figure(df: pd.DataFrame, p: np.ndarray, out_dir: Path) -> dict[str, object]:
    """Tiles whose pixels appear twice in the dataset, once filed as cracked and once not.

    Every tile's decoded pixels are hashed too: how many images are stored twice, with both
    labels, and within one photo.

    Args:
        df: The tile index.
        p: The probe's probability for every tile.
        out_dir: Figure directory.

    Returns:
        The probe's p for each twin filed both ways, and the counts of copies.
    """
    keys = df.groupby(["photo", "tile"])["cracked"].agg(["nunique", "size"])
    both = keys[(keys["nunique"] > 1)].index.tolist()
    fig, axes = plt.subplots(1, len(both), figsize=aml_style.SIDE)
    out = {}
    for ax, (photo, tile) in zip(np.atleast_1d(axes), both, strict=True):
        g = df[(df["photo"] == photo) & (df["tile"] == tile)]
        a, b = (read_tile(x) for x in g["path"])
        assert np.array_equal(a, b), f"{photo}-{tile}: the two files differ"
        ax.imshow(a)
        ax.set_title(f"{photo}-{tile}", fontsize=13)
        ax.set_xlabel(f"filed as both;\nnetwork: p = {p[g.index[0]]:.2f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        out[f"{photo}-{tile}"] = round(float(p[g.index[0]]), 2)
    aml_style.save(fig, out_dir, "twins")
    with ProcessPoolExecutor(WORKERS) as pool:
        digest = pd.Series(list(pool.map(_pixel_hash, df["path"], chunksize=512)), index=df.index)
    groups = [g.index for _, g in digest.groupby(digest) if len(g) > 1]
    out["copies"] = len(groups)
    out["copies_both_labels"] = int(sum(df["cracked"].loc[g].nunique() > 1 for g in groups))
    out["copies_same_photo"] = int(sum(df["photo"].loc[g].nunique() == 1 for g in groups))
    return out


def _pixel_hash(path: str) -> str:
    """SHA-1 of a tile's decoded pixels: equal for identical images, whatever the JPEG bytes.

    Args:
        path: A tile.

    Returns:
        The hex digest.
    """
    return hashlib.sha1(read_tile(path).tobytes(), usedforsecurity=False).hexdigest()


def decide_figure(df: pd.DataFrame, part: np.ndarray, p: np.ndarray, t: float,
                  out_dir: Path) -> dict[str, object]:  # fmt: skip
    """The decision on one test photo: tiles flagged for review; cracked tiles caught or missed.

    The photo is the test deck photo with the most cracked tiles.

    Args:
        df: The tile index.
        part: The photo split.
        p: The probe's probability for every tile.
        t: The threshold.
        out_dir: Figure directory.

    Returns:
        The photo, its tiles, flagged tiles, cracked tiles caught and missed, and the shares.
    """
    test_decks = df[(part == "test") & (df["surface"] == "deck").to_numpy()]
    photo = test_decks.groupby("photo")["cracked"].sum().sort_values(kind="stable").index[-1]
    px = MOSAIC_PX
    img, tiles = mosaic(df, photo, px)
    tp = tiles.assign(p=p[tiles.index])
    flag = tp[tp["p"] >= t]
    cracked = tp[tp["cracked"] == 1]
    missed = cracked[cracked["p"] < t]
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.imshow(img, interpolation="lanczos")
    outline_tiles(ax, flag, px, edgecolor="#D55E00", linewidth=1.4)
    ax.plot(cracked["col"] * px + px / 2, cracked["row"] * px + px / 2, "o", ms=4,
            color=aml_style.PALETTE[0], label="labelled cracked")  # fmt: skip
    ax.plot(missed["col"] * px + px / 2, missed["row"] * px + px / 2, "X", ms=11, mew=1.2,
            mfc="white", mec=aml_style.INK, label="cracked, missed")  # fmt: skip
    ax.plot([], [], "s", mfc="none", mec="#D55E00", ms=10, mew=1.4, label="flagged for review")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_axis_off()
    aml_style.save(fig, out_dir, "decide")
    return {"photo": photo, "tiles": len(tiles), "flagged": len(flag), "cracked": len(cracked),
            "caught": len(cracked) - len(missed), "missed": len(missed),
            "flagged_share": len(flag) / len(tiles),
            "caught_share": (len(cracked) - len(missed)) / len(cracked)}  # fmt: skip


def pr_figure(df: pd.DataFrame, part: np.ndarray, p: np.ndarray, t: float,
              out_dir: Path) -> None:  # fmt: skip
    """Precision against recall on each surface's test photos, with the operating point.

    Args:
        df: The tile index.
        part: The photo split.
        p: The probe's probability for every tile.
        t: The threshold.
        out_dir: Figure directory.
    """
    y = df["cracked"].to_numpy()
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for k, s in enumerate(SURFACES):
        m = (part == "test") & (df["surface"] == s).to_numpy()
        order = np.argsort(-p[m], kind="stable")
        ys = y[m][order]
        tp = np.cumsum(ys)
        rec, prec = tp / ys.sum(), tp / np.arange(1, len(ys) + 1)
        ax.plot(rec, prec, color=aml_style.PALETTE[k], ls=aml_style.LINESTYLES[k], label=s)
        flag = p[m] >= t
        r0, p0 = flag[y[m] == 1].mean(), y[m][flag].mean()
        ax.plot(r0, p0, marker=aml_style.MARKERS[k], color=aml_style.PALETTE[k], ms=9)
        ax.axhline(y[m].mean(), color=aml_style.PALETTE[k], ls=":", lw=1)
    ax.set_xlabel("recall: cracked tiles flagged")
    ax.set_ylabel("precision: flags that are cracked")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "pr")


def _pixels64(path: str) -> np.ndarray:
    """A tile as 64 x 64 grey levels, flattened.

    Args:
        path: A tile.

    Returns:
        float32 (4,096,).
    """
    small = Image.open(RAW / path).convert("L").resize((64, 64), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32).ravel()


def space_figure(
    df: pd.DataFrame, feats: np.ndarray, part: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Pixels vs features: can a line separate the cracked test tiles?

    Both probes are fitted on the same ``SPACE_TRAIN`` training and ``SPACE_VAL`` validation
    tiles and scored on every test tile, so the two AUCs compare like with like. Each shown tile
    is placed by its probe's log-odds (x) and the representation's first principal component (y).
    Only the rows used are decoded (a compact array, not one row per tile of the dataset).

    Args:
        df: The tile index.
        feats: ResNet-18 features.
        part: The photo split.
        out_dir: Figure directory.

    Returns:
        Each probe's test AUC and C; for each panel, how well the first principal component
        follows the label (AUC, sign-free), the share of its variance between surfaces (eta
        squared), and its correlation with the tile's mean grey level (sign-free).
    """
    y = df["cracked"].to_numpy()
    rng = np.random.default_rng((SEED, 15, 6))
    train = np.sort(rng.choice(np.flatnonzero(part == "train"), SPACE_TRAIN, replace=False))
    val = np.sort(rng.choice(np.flatnonzero(part == "val"), SPACE_VAL, replace=False))
    test = np.flatnonzero(part == "test")
    show = np.sort(np.concatenate([rng.choice(test[y[test] == c], SPACE_SHOW, replace=False)
                                   for c in (0, 1)]))  # fmt: skip
    rows = np.concatenate([train, val, test])
    with ProcessPoolExecutor(WORKERS) as pool:
        pix = np.array(list(pool.map(_pixels64, df["path"].to_numpy()[rows], chunksize=256)))
    assert pix.shape == (len(rows), 64 * 64), pix.shape
    n_tr, n_va = len(train), len(val)
    k = np.arange(len(rows))
    tr, va, te = k < n_tr, (k >= n_tr) & (k < n_tr + n_va), k >= n_tr + n_va
    yr = y[rows]
    at_show = n_tr + n_va + np.searchsorted(test, show)  # the shown tiles' rows in ``pix``
    assert np.array_equal(rows[at_show], show)
    pix_probe = probe_search(pix, yr, tr, va, "pixel probe")
    feat_probe = probe_search(feats[rows], yr, tr, va, "feature probe, same rows")
    out: dict[str, object] = {
        "pixels_auc": float(roc_auc_score(yr[te], pix_probe.score(pix[te]))),
        "features_auc": float(roc_auc_score(yr[te], feat_probe.score(feats[test]))),
        "pixels_C": pix_probe.C,
        "features_C": feat_probe.C,
        "pc1": {},
    }
    brightness = pix[at_show].mean(axis=1)
    surface = df["surface"].to_numpy()[show]
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.25})
    for ax, (name, rep_train, rep_show, probe) in zip(axes, (
        ("Pixels", pix[:n_tr], pix[at_show], pix_probe),
        ("ResNet-18 features", feats[train], feats[show], feat_probe),
    ), strict=True):  # fmt: skip
        z = StandardScaler().fit(rep_train).transform(rep_show)
        u, sv, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
        pc1 = u[:, 0] * sv[0]
        logit = probe.logit(rep_show)
        auc = float(roc_auc_score(y[show], pc1))  # a component's sign is arbitrary
        between = sum((surface == s).sum() * (pc1[surface == s].mean() - pc1.mean()) ** 2
                      for s in SURFACES)  # fmt: skip
        out["pc1"]["pixels" if name == "Pixels" else "features"] = {
            "label_auc": max(auc, 1 - auc),
            "surface_eta2": float(between / ((pc1 - pc1.mean()) ** 2).sum()),
            "brightness_r": abs(float(np.corrcoef(pc1, brightness)[0, 1])),
        }
        for c, mk, col, lab in ((0, "o", aml_style.PALETTE[0], "uncracked"),
                                (1, "^", aml_style.PALETTE[1], "cracked")):  # fmt: skip
            m = y[show] == c
            ax.scatter(logit[m], pc1[m], s=14, marker=mk, color=col, alpha=0.6, label=lab,
                       edgecolors="none")  # fmt: skip
        key = "pixels_auc" if name == "Pixels" else "features_auc"
        ax.set_title(f"{name}: AUC {out[key]:.2f}", fontsize=13)
        ax.set_xlabel("logistic score (log-odds)")
        ax.set_ylabel("1st principal component")
        ax.set_yticks([])
    axes[1].legend(loc="upper left", fontsize=11, markerscale=1.6, frameon=True,
                   facecolor="white", edgecolor="none", framealpha=1)  # points behind  # fmt: skip
    aml_style.save(fig, out_dir, "space")
    return out


# ---------------------------------------------------------------------------------------------
# A network trained from scratch


def _pixels_rgb(path: str) -> np.ndarray:
    """A tile at ``SCRATCH_PX`` px (bilinear).

    Args:
        path: A tile.

    Returns:
        uint8 (h, w, 3).
    """
    im = Image.open(RAW / path).convert("RGB").resize((SCRATCH_PX,) * 2, Image.Resampling.BILINEAR)
    return np.asarray(im)


def small_cnn() -> nn.Sequential:
    """A small CNN for training from a random start.

    Four conv-BN-ReLU-pool stages (16, 32, 64, 128 channels), global average pooling, and one
    logit.

    Returns:
        The network, with PyTorch's default initialization (seeded by the caller).
    """

    def stage(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                             nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))  # fmt: skip

    return nn.Sequential(stage(3, 16), stage(16, 32), stage(32, 64), stage(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))  # fmt: skip


def budget_photos(df: pd.DataFrame, part: np.ndarray, frac: float) -> np.ndarray:
    """A share of each surface's training photos (nested: a larger share contains a smaller).

    Args:
        df: The tile index.
        part: The photo split.
        frac: Share of each surface's training photos, rounded up.

    Returns:
        Boolean mask of the tiles of the chosen training photos.
    """
    keep = []
    for surface in SURFACES:
        on = (part == "train") & (df["surface"] == surface).to_numpy()
        photos = np.array(sorted(df["photo"][on].unique()))
        order = np.random.default_rng((SEED, 15, 8)).permutation(photos)
        keep += list(order[: int(np.ceil(frac * len(order)))])
    return (part == "train") & df["photo"].isin(keep).to_numpy()


def scratch_images(df: pd.DataFrame, part: np.ndarray) -> dict[str, np.ndarray]:
    """Every tile of the training, validation, and test photos at ``SCRATCH_PX``.

    Args:
        df: The tile index.
        part: The photo split.

    Returns:
        Per part, uint8 (n, SCRATCH_PX, SCRATCH_PX, 3) in the index's order: 2.8 GB in all.
    """
    with ProcessPoolExecutor(WORKERS) as pool:
        return {r: np.stack(list(pool.map(_pixels_rgb, df["path"].to_numpy()[part == r],
                                          chunksize=256)))
                for r in ("train", "val", "test")}  # fmt: skip


def channel_stats(imgs: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of uint8 images, without a float64 copy.

    ``imgs.reshape(-1, 3).std(0)`` would allocate the whole array in float64 (8 times its size,
    13 GB for the training tiles). Sums of the values and of their squares are integers below
    2^53, so they are exact in int64 whatever the order; the result matches NumPy's.

    Args:
        imgs: (n, h, w, 3) uint8.
        chunk: Images per step.

    Returns:
        Mean and standard deviation (ddof 0) of each channel, on the 0-255 scale.
    """
    total, squares, count = np.zeros(3, np.int64), np.zeros(3, np.int64), 0
    for i in range(0, len(imgs), chunk):
        a = imgs[i : i + chunk].reshape(-1, 3).astype(np.int64)
        total += a.sum(0)
        squares += (a * a).sum(0)
        count += len(a)
    mean = total / count
    return mean, np.sqrt(np.maximum(squares / count - mean**2, 0.0))


@dataclass(frozen=True)
class ScratchRun:
    """One training run of ``small_cnn`` from a random start."""

    curve: list[dict[str, float]]  # per evaluated pass: epoch, mean training loss, val AP
    p_val: np.ndarray  # probabilities on the validation tiles after the last pass
    p_test: np.ndarray  # probabilities on the test tiles after the last pass
    seconds: float
    tiles: int
    steps: int
    weights: int


def scratch_cnn(df: pd.DataFrame, part: np.ndarray, imgs: dict[str, np.ndarray],
                use: np.ndarray, *, epochs: int, eval_every: int = 1,
                seed: int = SEED) -> ScratchRun:  # fmt: skip
    """Train ``small_cnn`` from a random start on the tiles in ``use``; score val and test.

    AdamW with a one-cycle schedule over exactly ``epochs`` passes, horizontal and vertical
    flips (drawn per batch: a batch is flipped as a whole), and the positive class weighted by
    the uncracked-to-cracked ratio, so both classes count equally. Validation AP is recorded
    every ``eval_every`` passes and after the last; the test tiles are scored once, at the end.

    Args:
        df: The tile index.
        part: The photo split.
        imgs: ``scratch_images``.
        use: Boolean mask of the training tiles to use (all, or a budget of photos).
        epochs: Passes over those tiles (the schedule's length).
        eval_every: Score the validation tiles after every this many passes (and the last).
        seed: Seeds the weights, the batches, and the flips.

    Returns:
        The run.
    """
    y = df["cracked"].to_numpy()
    sel = use[part == "train"]  # positions within the training images
    train_imgs = imgs["train"][sel]
    ytr = torch.from_numpy(y[use]).float()
    mean, std = (v / 255 for v in channel_stats(train_imgs))

    def tensor(a: np.ndarray) -> torch.Tensor:  # (n, h, w, 3) uint8 -> (n, 3, h, w) float
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255
        m, sd = (torch.tensor(v).float().view(1, 3, 1, 1) for v in (mean, std))
        x = (x - m) / sd
        return x.contiguous(memory_format=torch.channels_last)

    torch.manual_seed(seed)
    net = small_cnn().to(memory_format=torch.channels_last)
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    steps = epochs * (len(ytr) // SCRATCH_BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=SCRATCH_LR, weight_decay=SCRATCH_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SCRATCH_LR, total_steps=steps)

    def predict(a: np.ndarray) -> np.ndarray:
        net.eval()
        with torch.inference_mode():
            z = torch.cat([net(tensor(a[i : i + 512])).squeeze(1) for i in range(0, len(a), 512)])
        return torch.sigmoid(z).numpy()

    t0 = time.perf_counter()
    curve = []
    pv = None
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(ytr))
        losses = []
        for i in range(0, len(ytr) - SCRATCH_BATCH + 1, SCRATCH_BATCH):
            b = perm[i : i + SCRATCH_BATCH]
            x = tensor(train_imgs[b.numpy()])
            if torch.rand(1).item() < 0.5:
                x = x.flip(3)
            if torch.rand(1).item() < 0.5:
                x = x.flip(2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(x).squeeze(1), ytr[b], pos_weight=pos_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if epoch % eval_every == 0 or epoch == epochs:
            pv = predict(imgs["val"])
            ap = float(average_precision_score(y[part == "val"], pv))
            curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ap": ap})
    assert pv is not None  # the last pass is always evaluated
    return ScratchRun(curve=curve, p_val=pv, p_test=predict(imgs["test"]),
                      seconds=time.perf_counter() - t0, tiles=int(use.sum()), steps=steps,
                      weights=sum(q.numel() for q in net.parameters()))  # fmt: skip


def length_search(df: pd.DataFrame, part: np.ndarray, imgs: dict[str, np.ndarray],
                  use: np.ndarray) -> tuple[int, dict[int, dict[int, ScratchRun]]]:  # fmt: skip
    """The small network's schedule length, chosen on validation by a bracketing search.

    Each length is one one-cycle schedule of that many passes, trained with every seed of
    ``SMALL_SEEDS``; the score is the median validation AP over the seeds (one seed's
    AP moves by about as much as the lengths differ). The search starts from ``SMALL_START`` and,
    while the best length is the shortest or the longest tried, adds the next length of
    ``SMALL_LADDER`` past it. Only validation photos are read.

    Args:
        df: The tile index.
        part: The photo split.
        imgs: ``scratch_images``.
        use: Boolean mask of the training tiles (the small budget).

    Returns:
        The chosen number of passes, and every run, by passes and seed.

    Raises:
        ValueError: If the best length is at an end of ``SMALL_LADDER``.
    """
    runs: dict[int, dict[int, ScratchRun]] = {}
    grid = list(SMALL_START)
    while True:
        for n in grid:
            runs[n] = {s: scratch_cnn(df, part, imgs, use, epochs=n, eval_every=n, seed=s)
                       for s in SMALL_SEEDS}  # fmt: skip
            aps = [round(r.curve[-1]["val_ap"], 4) for r in runs[n].values()]
            minutes = sum(r.seconds for r in runs[n].values()) / 60
            print(f"small network, {n} passes: validation AP {aps}, {minutes:.1f} min", flush=True)
        tried = sorted(runs)
        median = {n: float(np.median([r.curve[-1]["val_ap"] for r in runs[n].values()]))
                  for n in tried}  # fmt: skip
        best = max(tried, key=lambda n: median[n])  # a tie keeps the shorter schedule
        if tried[0] < best < tried[-1]:
            return best, runs
        k = SMALL_LADDER.index(best) + (-1 if best == tried[0] else 1)
        if not 0 <= k < len(SMALL_LADDER):
            raise ValueError(f"best schedule, {best} passes, is at an end of {SMALL_LADDER}")
        grid = [SMALL_LADDER[k]]


def transfer_figure(scratch: ScratchRun, small_val: list[float], base_val: list[float],
                    budgets: dict[str, list], probe_val_ap: float, *, small_photos: int,
                    chosen: int, base: int, out_dir: Path) -> None:  # fmt: skip
    """Validation AP of the from-scratch network and the probes: by pass, and by budget.

    Left: after each pass over all training tiles. Right: against the number of labelled
    training photos, with the probe on 256-px and on 128-px tiles, and the small network at the
    schedule length chosen on validation (the median of its seeds, each seed a tick); if that is
    not the first recipe's length, the first recipe's median is a hollow square.

    Args:
        scratch: The run on all training photos.
        small_val: Validation AP of the small-budget network at the chosen length, per seed.
        base_val: The same at the first recipe's length.
        budgets: ``probe_budgets``.
        probe_val_ap: The probe's validation AP on all training photos.
        small_photos: The small budget's photos.
        chosen: The chosen length, in passes.
        base: The first recipe's length, in passes.
        out_dir: Figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    k = np.arange(1, len(scratch.curve) + 1)
    ax.plot(k, [c["val_ap"] for c in scratch.curve], marker="s", color=aml_style.PALETTE[1],
            ls="--", label="small CNN, from scratch")  # fmt: skip
    ax.axhline(probe_val_ap, color=aml_style.PALETTE[0], label="ResNet-18 + logistic")
    ax.set_xticks([1, 4, 8, 12] if len(k) == 12 else k)
    ax.set_xlabel("passes over the training tiles")
    ax.set_ylabel("validation AP")
    ax.set_ylim(0, 1)
    ax.set_title(f"All {budgets['photos'][-1]} training photos", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax = axes[1]
    ax.plot(budgets["photos"], budgets["probe_val_ap"], marker="o", color=aml_style.PALETTE[0],
            label="ResNet-18 + logistic")  # fmt: skip
    ax.plot(budgets["photos"], budgets["probe128_val_ap"], marker="^", ls=":",
            color=aml_style.PALETTE[2], label="same, 128-px tiles")  # fmt: skip
    ax.plot([small_photos, budgets["photos"][-1]],
            [float(np.median(small_val)), scratch.curve[-1]["val_ap"]], marker="s", ls="--",
            color=aml_style.PALETTE[1], label="small CNN, from scratch")  # fmt: skip
    # each seed of the small-budget network, as a tick at its budget
    ax.plot([small_photos] * len(small_val), small_val, ls="none", marker="_", ms=14, mew=2,
            color=aml_style.PALETTE[1])  # fmt: skip
    if chosen != base:
        ax.plot([small_photos], [float(np.median(base_val))], ls="none", marker="s", ms=8,
                mfc="none", mec=aml_style.MUTED, label=f"from scratch, {base} passes")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xticks(budgets["photos"], [str(v) for v in budgets["photos"]])
    ax.minorticks_off()
    ax.set_xlabel("labelled training photos")
    ax.set_ylabel("validation AP")
    ax.set_ylim(0, 1)
    ax.set_title("Fewer labels", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    aml_style.save(fig, out_dir, "transfer")


def scratch_runs(df: pd.DataFrame, part: np.ndarray, budgets: dict[str, list],
                 probe_val_ap: float, out_dir: Path) -> dict[str, dict]:  # fmt: skip
    """The from-scratch network on all training photos and on ``SMALL_BUDGET`` of them.

    On the small budget, the first recipe (``SMALL_STEPS``, 33 passes) is kept as it was
    reported, with its three seeds; then ``length_search`` chooses the schedule's length on
    validation, and the chosen length's seeds are tested once.

    Args:
        df: The tile index.
        part: The photo split.
        budgets: ``probe_budgets``.
        probe_val_ap: The probe's validation AP on all training photos.
        out_dir: Figure directory.

    Returns:
        ``scratch`` and ``small`` (seed 0 of the first recipe): training curve, minutes, passes,
        steps, tiles, weights, the validation threshold, and the test scores at it;
        ``small_seeds``: the first recipe's validation and test AP per seed; ``small_length``:
        the lengths tried, each seed's validation AP and minutes at each, the chosen length,
        and its seeds' test scores at their own validation thresholds.
    """
    y = df["cracked"].to_numpy()
    yv, yt = y[part == "val"], y[part == "test"]
    imgs = scratch_images(df, part)
    scratch = scratch_cnn(df, part, imgs, part == "train", epochs=SCRATCH_EPOCHS)
    small_use = budget_photos(df, part, SMALL_BUDGET)
    base = int(np.ceil(SMALL_STEPS / (small_use.sum() // SCRATCH_BATCH)))
    if base not in SMALL_START:
        raise ValueError(f"the first recipe's {base} passes are not in {SMALL_START}")
    chosen, runs = length_search(df, part, imgs, small_use)
    small_photos = int(df["photo"][small_use].nunique())
    out: dict[str, dict] = {}
    for name, run in (("scratch", scratch), ("small", runs[base][SEED])):
        ts = threshold_for(run.p_val, yv)
        out[name] = {
            "curve": run.curve,
            "minutes": run.seconds / 60,
            "epochs": run.curve[-1]["epoch"],
            "steps": run.steps,
            "tiles": run.tiles,
            "weights": run.weights,
            "threshold": ts,
            "test": scores(run.p_test, yt, ts),
        }
    out["small"]["photos"] = small_photos
    out["small_seeds"] = {
        "seeds": list(SMALL_SEEDS),
        "val_ap": [r.curve[-1]["val_ap"] for r in runs[base].values()],
        "test_ap": [float(average_precision_score(yt, r.p_test)) for r in runs[base].values()],
        "minutes": [r.seconds / 60 for r in runs[base].values()],
    }
    best = runs[chosen].values()
    last = {str(n): [r.curve[-1] for r in runs[n].values()] for n in sorted(runs)}
    out["small_length"] = {
        "ladder": list(SMALL_LADDER),
        "tried": sorted(runs),
        "val_ap": {n: [c["val_ap"] for c in cs] for n, cs in last.items()},
        "median_val_ap": {n: float(np.median([c["val_ap"] for c in cs])) for n, cs in last.items()},
        "train_loss": {n: [c["train_loss"] for c in cs] for n, cs in last.items()},
        "minutes": {str(n): [r.seconds / 60 for r in runs[n].values()] for n in sorted(runs)},
        "passes": chosen,
        "base_passes": base,
        "steps": runs[chosen][SEED].steps,
        "test": [scores(r.p_test, yt, threshold_for(r.p_val, yv)) for r in best],
    }
    out["small_length"]["test_ap"] = [t["ap"] for t in out["small_length"]["test"]]
    transfer_figure(scratch, [r.curve[-1]["val_ap"] for r in best],
                    out["small_seeds"]["val_ap"], budgets, probe_val_ap,
                    small_photos=small_photos, chosen=chosen, base=base,
                    out_dir=out_dir)  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------
# Title art, and the run


def title_art_figure(rf: dict[str, dict[str, int]], out_dir: Path) -> None:
    """The title art: the deck tile's real crack as a line of light, and a dark sun.

    Around one point of the crack, the receptive fields of ResNet-18's four stages are nested
    portals.

    Args:
        rf: ``rf_figure``'s fields, per stage.
        out_dir: Figure directory.
    """
    g = read_tile(FILTER_TILE, gray=True)
    resp = correlate(g, DARK_LINE, mode="nearest")
    rows, cols = np.nonzero(resp > 15)
    along = rows + cols  # the crack runs from the upper left to the lower right
    bins = np.linspace(along.min(), along.max(), 41)
    k = np.digitize(along, bins[1:-1])
    pts = [(np.median(cols[k == b]), np.median(rows[k == b])) for b in range(40) if (k == b).any()]
    c = title_art.Canvas(seed=15)
    top = c.top

    def to_xy(col: float, row: float) -> tuple[float, float]:  # tile px -> canvas units
        return 0.06 + 0.88 * col / 256, top - 0.12 - 0.88 * row / 256

    c.ground(("#0b0d14", "#14101c", "#2a0f14", "#09070b"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.05, 0.62, 0.40, 4), soft=4), "#5a2216", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.55, 0.05, 0.42, 0.26, 3), soft=4), "#3a1912", 0.85)
    centre = to_xy(133, 96)  # the pixel of the one picture's 3 x 3 window
    for name, color in zip(("layer1", "layer2", "layer3", "layer4"),
                           ("#f3c46a", "#e39a45", "#c8673a", "#9b3f2e"), strict=True):  # fmt: skip
        half = 0.44 * rf[name]["effective90"] / 256
        x0, y0 = centre[0] - half, centre[1] - half
        square = [(x0, y0), (x0 + 2 * half, y0), (x0 + 2 * half, y0 + 2 * half),
                  (x0, y0 + 2 * half), (x0, y0)]  # fmt: skip
        c.glow(c.lines([square], 3.0, soft=1.0), color, 0.55, spread=10)
    rim = c.disc(centre, 0.05, soft=3) - c.disc(centre, 0.04, soft=3)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.6, spread=20)
    c.glaze(c.disc(centre, 0.04, soft=2), "#1d0a10", 0.95)
    c.glow(c.lines([[to_xy(x, y) for x, y in pts]], 2.5, soft=1.0), "#f3c98a", 0.7, spread=10)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean forms in rust at the bottom; "
        "across the canvas runs a pale golden line of light, the real crack of a bridge-deck "
        "tile; around one of its points, a small dark sun with a burning rim sits inside four "
        "nested glowing squares, gold to deep red, the receptive fields of a convolutional "
        "network's four stages.",
    )


def main() -> None:
    """Generate every figure and number for CEE Lecture 15."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    t0 = time.perf_counter()
    df, feats = load()
    part = photo_split(df, np.random.default_rng((SEED, 15)))
    model = backbone()
    numbers: dict[str, object] = {
        "data": {
            "tiles": len(df),
            "photos": int(df["photo"].nunique()),
            "cracked_share": float(df["cracked"].mean()),
            "by_surface": {
                s: {
                    "photos": int(df["photo"][df["surface"] == s].nunique()),
                    "tiles": int((df["surface"] == s).sum()),
                    "cracked_share": float(df["cracked"][df["surface"] == s].mean()),
                }
                for s in SURFACES
            },
        }
    }
    numbers["photos"] = photos_figure(df, out_dir)
    numbers["filter"] = filter_build(out_dir)
    numbers["ridges"] = ridge_figure(out_dir)
    numbers["rf"] = rf_figure(model, df, out_dir)
    numbers["conv1"] = conv1_figure(model, out_dir)
    numbers["stages"] = stages_figure(model, out_dir)
    numbers["residual"] = residual_figure(df, out_dir)
    numbers["verify"] = verify_conv() | {"equivariance": equivariance()}
    inputs = 3 * SIZE * SIZE  # a colour tile at ResNet-18's input size (the warm-up)
    numbers["first_layer"] = {"inputs": inputs, "dense_weights": inputs * 64,
                              "dense": inputs * 64 + 64, "conv3x3": 3 * 3 * 3 * 64 + 64,
                              "resnet18_conv1": model.conv1.weight.numel()}  # fmt: skip
    print(f"concepts in {time.perf_counter() - t0:.0f} s", flush=True)
    res = results(df, feats, part)
    p, probe, t = res.p, res.probe, res.numbers["threshold"]
    numbers["results"] = res.numbers
    numbers["space"] = space_figure(df, feats, part, out_dir)
    numbers["spread"] = split_spread(df, feats)
    numbers["leakage"] = leakage(df, feats, part, probe.C)
    numbers["cross"] = cross_surface(df, feats, part, out_dir)
    print(f"probes in {time.perf_counter() - t0:.0f} s", flush=True)
    numbers["resolution"] = resolution(model, df, part, probe, t=t, feats=feats, out_dir=out_dir)
    tiles = pick_tiles(df, part, p)
    attempt_figures(tiles, out_dir)
    numbers["attempt"] = {r.letter: {"path": r.path, "cracked": int(r.cracked), "p": float(r.p)}
                          for r in tiles.itertuples()}  # fmt: skip
    numbers["occlusion"] = occlusion_figure(model, probe, tiles, out_dir)
    numbers["twins"] = twins_figure(df, p, out_dir)
    numbers["decide"] = decide_figure(df, part, p, t, out_dir)
    pr_figure(df, part, p, t, out_dir)
    print(f"break-it in {time.perf_counter() - t0:.0f} s", flush=True)
    t1 = time.perf_counter()
    feats128 = embed(model, df["path"].tolist(), factor=2)
    numbers["probe128"] = probe128(df, feats128, part) | {
        "embed_minutes": (time.perf_counter() - t1) / 60}  # fmt: skip
    numbers["budgets"] = budgets = probe_budgets(df, feats, feats128, part)
    numbers |= scratch_runs(df, part, budgets, probe.val_ap, out_dir)
    title_art_figure(numbers["rf"], out_dir)
    aml_style.thumbnails(out_dir, ("filter_2", "receptive", "transfer", "decide"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
