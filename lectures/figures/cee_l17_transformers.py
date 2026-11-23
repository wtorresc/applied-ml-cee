"""Figures and numbers for CEE Lecture 17: tokens, attention, and transformers.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l17_transformers.py --out build/fig/cee_l17_transformers

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides). The lecture
follows CS 441's "Word Representations and Attention Models" and "Transformers in Vision and
Language" (Fall 2026), in their order, on CEE data:

- Tokens and word vectors: OSHA's Severe Injury Reports (2015-2025; U.S. federal data), from
  ``scripts/fetch_osha_sir.sh``: byte-pair encoding from scratch, a small word2vec, and BERT's
  masked words (bert-base-uncased, Apache-2.0) on the incident descriptions.
- Attention: cross-attention as a soft nearest-neighbour average on Lecture 2's concrete mixes
  (UCI; ``concrete.py``), then self-attention and a transformer block written out and checked
  against the first block of torchvision's ViT-B/16 on a real crack tile.
- Vision transformers: SDNET2018's crack tiles (Lecture 15; ``sdnet.py``) as 16 x 16 patches;
  ViT-B/16's ImageNet features (``scripts/derive_sdnet2018_vit.py``) under Lecture 15's probe
  (``probes.py``) against ResNet-18's, at the inspector's 90% recall target; attention maps; the
  tiles cut into 32-pixel blocks and shuffled (three permutations).

Splits are Lecture 2's (30 design-held-out splits) and Lecture 15's (photos 60/20/20), so every
comparison is against a number students have already seen.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from torch import nn
from torch.nn import functional as F  # noqa: N812 (PyTorch's convention)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import aml_style
import concrete
import title_art
from common import DATA_DIR, parse_out_dir, require, seed_all, write_numbers, write_tex_macros
from osha import TRAIN_YEARS, osha, training
from probes import fit_probe, scores, threshold_for
from sdnet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIZE,
    backbone,
    load,
    photo_split,
    preprocess,
    read_tile,
)

SEED = 17
VIT_FEATURES = DATA_DIR / "derived" / "sdnet2018_vit_b_16.npy"
VIT_WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
VIT_CACHE = {  # what the ViT cache's JSON must record (scripts/derive_sdnet2018_vit.py)
    "weights": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
    "url": VIT_WEIGHTS.url,
    "preprocess": f"resize {SIZE}x{SIZE} bilinear antialias; "
    f"ImageNet mean {IMAGENET_MEAN} std {IMAGENET_STD}",
    "layer": "class token after the encoder's final layer norm (heads replaced by identity)",
}
TAUS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)  # attention temperatures tried (squared z-units)
TOP_KEYS = 8  # keys shown in the one picture
PATCH = 16  # ViT-B/16's patch, pixels
GRID = SIZE // PATCH  # 14 x 14 patches
# The shuffle test cuts each 224-px input into 32-px blocks: one block is 2 x 2 of ViT's patches
# and one cell of ResNet-18's last 7 x 7 feature map (stride 32), so the cut is aligned to both
# grids. ViT's patch embedding never reads across a seam between blocks; ResNet-18's filters
# cross the seams at every layer. Three random permutations, each applied to every tile.
BLOCK = 32
BLOCKS = SIZE // BLOCK  # 7 x 7 blocks
SHUFFLES = 3
SHOW_POSITIONS = ((0, 0), (0, 13), (7, 7), (13, 6))  # patches whose position similarity is shown
VERIFY_TILE = "D/CD/7001-21.jpg"  # Lecture 15's filter tile: the block is checked on it


def device() -> torch.device:
    """The GPU if there is one."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cache(path: Path, want: dict[str, object], rows: int, hint: str) -> np.ndarray:
    """A feature cache, loaded only if its JSON records the model and preprocessing expected.

    The same check as ``sdnet.load`` makes for Lecture 15's cache (duplicated in Lecture 18's
    script until ``sdnet`` offers it).

    Args:
        path: The ``.npy`` cache; its metadata is the ``.json`` beside it.
        want: Fields the metadata must record exactly (weights, preprocessing, layer).
        rows: Tiles in the index: the cache must have one row per tile.
        hint: The script that writes the cache.

    Returns:
        The features, one row per tile.

    Raises:
        ValueError: If a field or the shape differs from what is expected.
    """
    meta = json.loads(require(path.with_suffix(".json"), hint).read_text())
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"{path.name} was not made as this script expects: {bad}")
    feats = np.load(require(path, hint))
    if feats.shape[0] != rows or list(feats.shape) != meta.get("shape"):
        raise ValueError(f"{path.name}: shape {feats.shape}, metadata {meta.get('shape')}")
    return feats


# --------------------------------------------------------------------------------------------
# Attention as a soft nearest-neighbour average (Lecture 2's concrete)
# --------------------------------------------------------------------------------------------


def attend(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-attention with a distance score: softmax(-|q - k|^2 / tau) weights on the values.

    Args:
        q: Queries, shape (m, d).
        keys: Keys, shape (n, d).
        values: Values, shape (n,).
        tau: Temperature: small -> the nearest key alone (1-NN), large -> the mean of the values.

    Returns:
        The weights (m, n) and the attended values (m,).
    """
    d2 = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(-1)  # (m, n)
    s = -d2 / tau
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return w, w @ values


def choose_tau(z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """The temperature with the lowest error over 5 folds of whole designs (no twin leaks)."""
    err = {}
    for tau in TAUS:
        sq = []
        for a, b in GroupKFold(5).split(z, y, groups):
            _, pred = attend(z[b], z[a], y[a], tau)
            sq.append((pred - y[b]) ** 2)
        err[tau] = float(np.sqrt(np.mean(np.concatenate(sq))))
    tau = min(err, key=err.get)
    if tau in (TAUS[0], TAUS[-1]):
        raise ValueError(f"temperature {tau} is at the edge of TAUS; widen the grid")
    return tau


def concrete_attention() -> dict[str, object]:
    """Lecture 2's 30 design-held-out splits: 5-NN against attention over all training mixes."""
    df = concrete.load_concrete()
    x, y = concrete.mix_groups(df), df["strength"].to_numpy(float)
    groups = df["design"].to_numpy()
    knn, att, taus = [], [], []
    for tr, te in concrete.grouped_splits(df):
        mu, sd = x[tr].mean(0), x[tr].std(0)
        z = (x - mu) / sd
        p, _ = concrete.knn_predict(x, y, tr, te)
        tau = choose_tau(z[tr], y[tr], groups[tr])
        _, pa = attend(z[te], z[tr], y[tr], tau)
        knn.append(float(np.sqrt(np.mean((p - y[te]) ** 2))))
        att.append(float(np.sqrt(np.mean((pa - y[te]) ** 2))))
        taus.append(tau)
    return {
        "knn_rmse_median": float(np.median(knn)),
        "attention_rmse_median": float(np.median(att)),
        "attention_better": int(sum(a < k for a, k in zip(att, knn, strict=True))),
        "splits": len(knn),
        "tau_values": sorted(set(taus)),
        "tau_median": float(np.median(taus)),
    }


def attention_build(out_dir: Path) -> dict[str, object]:
    """The one picture: a proposed mix attends to every tested design (three frames)."""
    df = concrete.load_concrete()
    x, y = concrete.mix_groups(df), df["strength"].to_numpy(float)
    q = concrete.PROPOSED_ROW
    design = df["design"].to_numpy()
    keys = np.flatnonzero(design != design[q])  # every design but the proposed one
    mu, sd = x[keys].mean(0), x[keys].std(0)
    z = (x - mu) / sd
    tau = choose_tau(z[keys], y[keys], design[keys])
    w, pred = attend(z[[q]], z[keys], y[keys], tau)
    w = w[0]
    top = np.argsort(w)[::-1][:TOP_KEYS]
    p5, _ = concrete.knn_predict(x, y, keys, np.array([q]))
    for frame in (1, 2, 3):
        fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, width_ratios=(1.2, 1, 0.8))
        a, b, c = axes
        a.scatter(x[keys, 0], np.exp(x[keys, 5]), s=6, c="#C8C8C8", lw=0)
        a.scatter(
            x[keys[top], 0],
            np.exp(x[keys[top], 5]),
            s=40,
            facecolor="none",
            edgecolor=aml_style.PALETTE[0],
            lw=1.4,
            label="largest weights",
        )
        a.scatter(
            [x[q, 0]],
            [np.exp(x[q, 5])],
            marker="*",
            s=220,
            color=aml_style.PALETTE[1],
            label="query: the proposed mix",
        )
        a.set_yscale("log")
        a.set_xlabel("water / binder")
        a.set_ylabel("age (days)")
        a.legend(fontsize=9, frameon=False, loc="upper right")
        a.set_title(f"keys: {len(keys):,} tests of other designs", fontsize=12)
        for ax in (b, c):
            ax.set_axis_off()
        if frame >= 2:
            b.set_axis_on()
            b.barh(np.arange(TOP_KEYS), w[top], color=aml_style.PALETTE[0])
            b.set_yticks(np.arange(TOP_KEYS), [f"{y[keys[i]]:.1f} MPa" for i in top], fontsize=10)
            b.invert_yaxis()
            b.set_xlabel("weight")
            b.set_title(f"softmax weights (top {TOP_KEYS})", fontsize=12)
        if frame >= 3:
            c.text(0.0, 0.85, "attention", fontsize=12, fontweight="bold", color=aml_style.NAVY)
            c.text(0.0, 0.70, f"{pred[0]:.1f} MPa", fontsize=16)
            c.text(0.0, 0.48, "5-NN (Lecture 2)", fontsize=12, color=aml_style.MUTED)
            c.text(0.0, 0.33, f"{p5[0]:.1f} MPa", fontsize=16, color=aml_style.MUTED)
            c.text(0.0, 0.11, f"tested: {y[q]:.1f} MPa", fontsize=12)
        fig.tight_layout()
        aml_style.save(fig, out_dir, f"attention_{frame}")
    return {
        "tau": tau,
        "keys": len(keys),
        "key_mixes": int(df["mix"].iloc[keys].nunique()),  # tests come from fewer mixes
        "top_weights": w[top].tolist(),
        "top_values": y[keys[top]].tolist(),
        "top_share": float(w[top].sum()),
        "prediction": float(pred[0]),
        "knn5": float(p5[0]),
        "tested": float(y[q]),
        "row": int(q),
    }


# --------------------------------------------------------------------------------------------
# A transformer block written out, checked against torchvision's ViT-B/16
# --------------------------------------------------------------------------------------------


def vit() -> nn.Module:
    """The ViT-B/16 of torchvision, with ImageNet weights, classifier removed, in eval mode."""
    model = vit_b_16(weights=VIT_WEIGHTS)
    model.heads = nn.Identity()
    return model.eval()


def tile_tensor(path: str) -> torch.Tensor:
    """A tile resized to 224 and normalized as the cached features were (1, 3, 224, 224).

    Lecture 15's ``sdnet.preprocess``; an earlier copy here (``F.interpolate``) agreed with it to
    1.2e-6 on 200 tiles, float32 rounding.
    """
    return preprocess(read_tile(path))[None]


def tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Patch tokens plus the class token plus position embeddings: (n, 197, 768)."""
    t = model._process_input(x)  # (n, 196, 768): one token per 16 x 16 patch
    cls = model.class_token.expand(t.shape[0], -1, -1)  # (n, 1, 768)
    return torch.cat([cls, t], dim=1) + model.encoder.pos_embedding


def self_attention(
    x: torch.Tensor, mha: nn.MultiheadAttention
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head self-attention written out with ``mha``'s weights.

    Args:
        x: Tokens (n, t, d).
        mha: The layer whose projections to use (batch_first).

    Returns:
        The output (n, t, d) and the attention weights (n, heads, t, t).
    """
    n, t, d = x.shape
    h = mha.num_heads
    qkv = x @ mha.in_proj_weight.T + mha.in_proj_bias  # (n, t, 3d): queries, keys, values
    q, k, v = qkv.split(d, dim=-1)
    q, k, v = (z.view(n, t, h, d // h).transpose(1, 2) for z in (q, k, v))  # (n, h, t, d/h)
    a = torch.softmax(q @ k.transpose(-2, -1) / (d // h) ** 0.5, dim=-1)  # (n, h, t, t)
    out = (a @ v).transpose(1, 2).reshape(n, t, d)  # heads side by side again
    return out @ mha.out_proj.weight.T + mha.out_proj.bias, a


def block(x: torch.Tensor, blk: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """One pre-norm transformer block: x + attention(norm x), then + MLP(norm)."""
    a_out, a = self_attention(
        F.layer_norm(x, (x.shape[-1],), blk.ln_1.weight, blk.ln_1.bias, blk.ln_1.eps),
        blk.self_attention,
    )
    x = x + a_out
    y = F.layer_norm(x, (x.shape[-1],), blk.ln_2.weight, blk.ln_2.bias, blk.ln_2.eps)
    lin1, lin2 = blk.mlp[0], blk.mlp[3]
    return x + F.gelu(y @ lin1.weight.T + lin1.bias) @ lin2.weight.T + lin2.bias, a


def verify_block(path: str) -> dict[str, object]:
    """Our block against torchvision's first ViT block, in float64, on one real tile."""
    model = vit().double()
    with torch.no_grad():
        x0 = tokens(model, tile_tensor(path).double())
        ours, _ = block(x0, model.encoder.layers[0])
        ref = model.encoder.layers[0](x0)
    blk = model.encoder.layers[0]
    return {
        "max_abs": float((ours - ref).abs().max()),
        "tokens": int(x0.shape[1]),
        "dim": int(x0.shape[2]),
        "heads": int(blk.self_attention.num_heads),
        "block_weights": int(sum(p.numel() for p in blk.parameters())),
        "blocks": len(model.encoder.layers),
        "vit_weights": int(sum(p.numel() for p in model.parameters())),
        "resnet_weights": int(sum(p.numel() for p in backbone().parameters())),
        "mlp_weights": int(sum(p.numel() for p in blk.mlp.parameters())),
    }


def positions_figure(out_dir: Path) -> dict[str, float]:
    """ViT's learned position embeddings: how alike each patch's is to every other patch's."""
    pe = vit().encoder.pos_embedding.detach()[0, 1:]  # (196, 768), the class token dropped
    pe = F.normalize(pe, dim=1)
    sim = (pe @ pe.T).numpy()  # (196, 196) cosine similarity
    fig, axes = plt.subplots(1, len(SHOW_POSITIONS), figsize=aml_style.FULL)
    for ax, (r, c) in zip(axes, SHOW_POSITIONS, strict=True):
        m = sim[r * GRID + c].reshape(GRID, GRID)
        ax.imshow(m, cmap="cividis", vmin=-0.2, vmax=1.0)
        ax.scatter(
            [c], [r], marker="s", s=40, facecolor="none", edgecolor=aml_style.PALETTE[1], lw=1.6
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"patch (row {r}, col {c})", fontsize=11)
    aml_style.save(fig, out_dir, "positions")
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    dist = np.hypot(rows[:, None] - rows[None, :], cols[:, None] - cols[None, :])
    off = ~np.eye(len(sim), dtype=bool)
    return {
        "neighbour_similarity": float(sim[(dist == 1)].mean()),
        "far_similarity": float(sim[(dist >= 8) & off].mean()),
        "corr_with_distance": float(np.corrcoef(sim[off], dist[off])[0, 1]),
    }


# --------------------------------------------------------------------------------------------
# ViT against ResNet-18 at the inspector's target (Lecture 15's split and probe)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Screen:
    """One backbone's probe on Lecture 15's photo split."""

    p: np.ndarray  # probability of "cracked" for every tile
    t: float  # threshold that flags 90% of the validation photos' cracked tiles
    numbers: dict[str, object]


def screen(feats: np.ndarray, y: np.ndarray, part: np.ndarray, df: pd.DataFrame) -> Screen:
    """Fit the probe, set the threshold on validation, score the test photos once."""
    train, val, test = (part == r for r in ("train", "val", "test"))
    pr = fit_probe(feats, y, train, val)
    p = pr.score(feats)
    t = threshold_for(p[val], y[val])
    out = {"C": pr.C, "val_ap": pr.val_ap, "all": scores(p[test], y[test], t)}
    for s in ("deck", "wall", "pavement"):
        m = test & (df["surface"] == s).to_numpy()
        out[s] = scores(p[m], y[m], t)
    return Screen(p, t, out)


def shuffle_blocks(x: torch.Tensor, perm: np.ndarray, block: int = BLOCK) -> torch.Tensor:
    """An image (3, s, s) cut into ``block``-pixel squares, put back in the order ``perm``.

    Args:
        x: The image, its side a multiple of ``block``.
        perm: A permutation of the (s / block)^2 squares, in row-major order.
        block: The square's side, pixels.

    Returns:
        The shuffled image (3, s, s); the identity permutation returns ``x`` exactly.
    """
    c, s, _ = x.shape
    g = s // block
    p = x.unfold(1, block, block).unfold(2, block, block)  # (3, g, g, block, block)
    p = p.reshape(c, g * g, block, block)[:, torch.as_tensor(perm)]
    p = p.reshape(c, g, g, block, block).permute(0, 1, 3, 2, 4)  # rows of pixels again
    return p.reshape(c, s, s)


class Shuffled(Dataset):
    """Tiles with their 32 x 32 blocks (of the 224-px input) in a fixed shuffled order."""

    def __init__(self, paths: list[str], perm: np.ndarray | None) -> None:
        """Tiles to read; ``perm`` reorders the 49 blocks (None: as taken)."""
        self.paths, self.perm = paths, perm

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        """The i-th tile, (3, 224, 224), blocks permuted."""
        x = tile_tensor(self.paths[i])[0]
        return x if self.perm is None else shuffle_blocks(x, self.perm)


def embed(model: nn.Module, paths: list[str], perm: np.ndarray | None) -> np.ndarray:
    """Features of ``paths`` under ``model``, blocks in the order ``perm`` (None: as taken)."""
    dev = device()
    model = model.eval().to(dev)
    loader = DataLoader(Shuffled(paths, perm), batch_size=256, num_workers=12, shuffle=False)
    with torch.inference_mode():
        out = [model(x.to(dev)).float().cpu().numpy() for x in loader]
    return np.concatenate(out)


def shuffle_test(
    df: pd.DataFrame, part: np.ndarray, rfeat: np.ndarray, vfeat: np.ndarray
) -> dict[str, object]:
    """Break it: shuffle every tile's 32-px blocks; the probes are refitted on shuffled tiles.

    The same ``SHUFFLES`` permutations (seeds (17, 0), (17, 1), ...) for both networks; each
    network also re-embeds the tiles as taken, which must reproduce its cache (TF32 is off, so
    the GPU's convolutions run in float32, as the caches were made).

    Returns:
        Per network: the test scores as taken and under each permutation, the drops in average
        precision, the largest difference from the cache, and the time per tile.
    """
    y = df["cracked"].to_numpy()
    perms = [np.random.default_rng((SEED, i)).permutation(BLOCKS**2) for i in range(SHUFFLES)]
    paths = df["path"].tolist()
    out: dict[str, object] = {
        "block": BLOCK,
        "blocks": BLOCKS**2,
        "fixed_points": [int((p == np.arange(BLOCKS**2)).sum()) for p in perms],
    }
    for name, model, cached in (("resnet18", backbone(), rfeat), ("vit_b_16", vit(), vfeat)):
        t0 = time.perf_counter()
        plain = embed(model, paths, None)
        shuffled = [embed(model, paths, p) for p in perms]
        seconds = time.perf_counter() - t0
        base = screen(plain, y, part, df).numbers["all"]
        runs = [screen(f, y, part, df).numbers["all"] for f in shuffled]
        out[name] = {
            "plain": base,
            "shuffled": runs,
            "ap_drop": [base["ap"] - r["ap"] for r in runs],
            "review_rise": [r["review"] - base["review"] for r in runs],
            "cache_max_abs": float(np.abs(plain - cached).max()),
            "passes": (1 + len(perms)) * len(paths),  # tiles through the network, in all
            "ms_per_tile": 1e3 * seconds / ((1 + len(perms)) * len(paths)),
        }
        if out[name]["cache_max_abs"] > 1e-3:
            raise ValueError(f"{name}: tiles as taken differ from the cache by more than 1e-3")
    return out


def rollout(model: nn.Module, path: str) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): what the class token draws on, 14 x 14.

    Each block's attention, averaged over heads, is mixed half and half with the identity (the
    residual path), rows renormalized, and the blocks multiplied in order; the class token's row
    over the 196 patches is the map. Uses ``block``, the code checked against torchvision.
    """
    with torch.no_grad():
        x = tokens(model, tile_tensor(path))
        roll = torch.eye(x.shape[1])
        for blk in model.encoder.layers:
            x, a = block(x, blk)
            m = 0.5 * a[0].mean(0) + 0.5 * torch.eye(x.shape[1])  # (197, 197)
            roll = (m / m.sum(-1, keepdim=True)) @ roll
    return roll[0, 1:].reshape(GRID, GRID).numpy()


def attention_maps_figure(
    df: pd.DataFrame, part: np.ndarray, vit_screen: Screen, out_dir: Path
) -> dict[str, object]:
    """Four test tiles with cracks, the ViT probe's p, and the class token's attention."""
    y = df["cracked"].to_numpy()
    test = part == "test"
    picks = []
    for s in ("deck", "wall", "pavement"):
        m = np.flatnonzero(test & (y == 1) & (df["surface"] == s).to_numpy())
        picks.append(int(m[np.argmax(vit_screen.p[m])]))  # the surest catch on each surface
    missed = np.flatnonzero(test & (y == 1) & (vit_screen.p < vit_screen.t))
    picks.append(int(missed[np.argmin(vit_screen.p[missed])]))  # the surest miss
    model = vit()
    fig, axes = plt.subplots(2, len(picks), figsize=aml_style.FULL)
    for j, i in enumerate(picks):
        path = df["path"].iloc[i]
        axes[0, j].imshow(read_tile(path))
        axes[0, j].set_title(f"{df['surface'].iloc[i]}, p = {vit_screen.p[i]:.2f}", fontsize=10)
        axes[1, j].imshow(
            read_tile(path, gray=True), cmap="gray", alpha=0.6, extent=(0, GRID, GRID, 0)
        )
        axes[1, j].imshow(
            rollout(model, path), cmap="cividis", alpha=0.6, extent=(0, GRID, GRID, 0)
        )
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[1, 0].set_ylabel("attention", fontsize=11)
    aml_style.save(fig, out_dir, "attention_maps")
    return {
        "tiles": [df["path"].iloc[i] for i in picks],
        "p": [float(vit_screen.p[i]) for i in picks],
        "threshold": vit_screen.t,
    }


def decide_figure(res: Screen, vit_s: Screen, out_dir: Path) -> None:
    """The inspector's numbers for each backbone on the test photos."""
    names = ("ResNet-18 (Lecture 15)", "ViT-B/16")
    rows = [res.numbers["all"], vit_s.numbers["all"]]
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    for ax, key, label in zip(
        axes,
        ("ap", "recall", "review"),
        ("average precision", "cracks flagged", "tiles to review"),
        strict=True,
    ):
        vals = [r[key] for r in rows]
        ax.bar(
            [0, 1],
            vals,
            color=[aml_style.PALETTE[0], aml_style.PALETTE[1]],
            width=0.6,
            hatch=["", "//"],
        )
        for k, v in enumerate(vals):
            ax.text(
                k,
                v + 0.01,
                f"{v:.2f}" if key == "ap" else f"{100 * v:.0f}%",
                ha="center",
                fontsize=12,
            )
        ax.set_xticks([0, 1], [n.split(" (")[0] for n in names], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=12)
    aml_style.save(fig, out_dir, "decide")


# --------------------------------------------------------------------------------------------
# Text as tokens: OSHA's Severe Injury Reports
# --------------------------------------------------------------------------------------------

WORD = re.compile(r"\w+|[^\w\s]+")  # the pre-tokenizer: runs of letters/digits, or punctuation
LETTERS = re.compile(r"[a-z]+")
SHOW_MERGES = 12
BPE_VOCAB = 8000
CHECK_MERGES = 50
DOMAIN_WORDS = (
    "scaffold",
    "excavator",
    "trench",
    "forklift",
    "amputated",
    "lacerated",
    "skid",
    "rebar",
    "drywall",
    "trencher",
)
W2V_DIM, W2V_WINDOW, W2V_NEG, W2V_EPOCHS, W2V_MIN = 100, 5, 5, 3, 5
W2V_BATCH, W2V_LR, W2V_SUBSAMPLE = 4096, 3e-3, 1e-5
QUERY_WORDS = ("ladder", "scaffold", "trench", "forklift", "amputated", "finger")
BERT = "google-bert/bert-base-uncased"  # Devlin et al. (2019); Apache-2.0
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BARE = "The employee fell from a [MASK] and fractured his leg."  # no job, no tool, no task
FELL = re.compile(r"\b(fell (?:from|off) (?:a|an|the) )([a-z]+)\b")


def bpe_scratch(texts: pd.Series, merges: int) -> list[tuple[str, str]]:
    """Byte-pair encoding's first ``merges`` merges, written out (Sennrich et al., 2016).

    Start from characters; repeatedly fuse the most frequent adjacent pair of symbols, counting
    over the corpus's words (a word never merges across a space). Ties go to the smaller pair.
    """
    counts: dict[tuple[str, ...], int] = {}
    for t in texts:
        for w in WORD.findall(t):
            key = tuple(w)
            counts[key] = counts.get(key, 0) + 1
    done = []
    for _ in range(merges):
        pairs: dict[tuple[str, str], int] = {}
        for w, c in counts.items():
            for a, b in itertools.pairwise(w):
                pairs[(a, b)] = pairs.get((a, b), 0) + c
        best = max(pairs, key=lambda p: (pairs[p], tuple(-ord(ch) for ch in p[0] + "|" + p[1])))
        done.append(best)
        merged: dict[tuple[str, ...], int] = {}
        for w, c in counts.items():
            key = merge_pair(w, best)
            merged[key] = merged.get(key, 0) + c
        counts = merged
    return done


def merge_pair(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """``word`` with every adjacent ``pair`` fused, left to right.

    Args:
        word: A word as a tuple of symbols.
        pair: The two symbols to fuse.

    Returns:
        The word's symbols after the merge.
    """
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def bpe_library(texts: pd.Series, vocab: int) -> Tokenizer:
    """The same algorithm in Hugging Face ``tokenizers`` (lowercase, the same pre-tokenizer)."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))  # noqa: S106 (a token, not a password)
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, min_frequency=2, show_progress=False, special_tokens=["[UNK]"]
    )
    tok.train_from_iterator(texts.tolist(), trainer=trainer)
    return tok


def tokens_numbers(df: pd.DataFrame) -> dict[str, object]:
    """Words, subwords, and characters: vocabulary, unseen words, tokens per report, splits."""
    train = training(df)
    test = df.loc[df["year"] > TRAIN_YEARS[1], "lower"]
    words_train = [w for t in train for w in LETTERS.findall(t)]
    vocab = pd.Series(words_train).value_counts()
    words_test = [w for t in test for w in LETTERS.findall(t)]
    unseen = np.mean([w not in vocab.index for w in words_test])
    scratch = bpe_scratch(train, CHECK_MERGES)
    lib = bpe_library(train, BPE_VOCAB)
    lib_merges = [tuple(m) for m in json.loads(lib.to_str())["model"]["merges"][:CHECK_MERGES]]
    same = next(
        (i for i, (a, b) in enumerate(zip(scratch, lib_merges, strict=True)) if a != b),
        CHECK_MERGES,
    )
    bert = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    sample = test.sample(2000, random_state=SEED)
    per_report = {
        "characters": float(np.median(sample.str.len())),
        "words": float(np.median([len(WORD.findall(t)) for t in sample])),
        "bpe_osha": float(np.median([len(lib.encode(t).tokens) for t in sample])),
        "bert_wordpiece": float(np.median([len(bert.tokenize(t)) for t in sample])),
    }
    return {
        "reports": len(df),
        "train_reports": len(train),
        "test_reports": len(test),
        "words_total": int(df["lower"].str.count(r"[a-z]+").sum()),
        "word_vocab": len(vocab),
        "word_vocab_5plus": int((vocab >= W2V_MIN).sum()),
        "test_words_unseen": float(unseen),
        "first_merges": ["".join(m[:1]) + " + " + m[1] for m in scratch[:SHOW_MERGES]],
        "merges_identical": int(same),
        "per_report": per_report,
        "splits": {
            w: {"bert": bert.tokenize(w), "osha": lib.encode(w).tokens} for w in DOMAIN_WORDS
        },
        "bpe_vocab": BPE_VOCAB,
        "bert_vocab": bert.vocab_size,
    }


def word2vec(df: pd.DataFrame) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """Skip-gram with negative sampling (Mikolov et al., 2013), written out, on the CPU.

    Each word predicts the words within ``W2V_WINDOW`` of it; for every true pair, ``W2V_NEG``
    random words (drawn by frequency^0.75) are pushed away. Frequent words are subsampled.

    Returns:
        The vocabulary (words seen at least ``W2V_MIN`` times), unit-length vectors, and the
        training numbers.
    """
    rng = np.random.default_rng(SEED)
    seed_all(SEED)
    docs = [LETTERS.findall(t) for t in training(df)]
    counts = pd.Series([w for d in docs for w in d]).value_counts()
    counts = counts[counts >= W2V_MIN]
    vocab = counts.index.tolist()
    index = {w: i for i, w in enumerate(vocab)}
    freq = counts.to_numpy(float) / counts.sum()
    keep = np.minimum(1.0, np.sqrt(W2V_SUBSAMPLE / freq) + W2V_SUBSAMPLE / freq)
    neg_p = torch.tensor(freq**0.75 / (freq**0.75).sum())
    centre = nn.Embedding(len(vocab), W2V_DIM)
    context = nn.Embedding(len(vocab), W2V_DIM)
    nn.init.uniform_(centre.weight, -0.5 / W2V_DIM, 0.5 / W2V_DIM)
    nn.init.zeros_(context.weight)
    opt = torch.optim.Adam([*centre.parameters(), *context.parameters()], lr=W2V_LR)
    t0 = time.perf_counter()
    for _ in range(W2V_EPOCHS):
        pairs = skipgram_pairs(docs, index, keep, rng)
        for i in range(0, len(pairs), W2V_BATCH):
            b = pairs[i : i + W2V_BATCH]
            neg = torch.multinomial(neg_p, len(b) * W2V_NEG, replacement=True).view(len(b), W2V_NEG)
            u = centre(b[:, 0])  # (batch, dim)
            pos = (u * context(b[:, 1])).sum(-1)  # (batch,)
            negs = torch.einsum("bd,bkd->bk", u, context(neg))  # (batch, neg)
            loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-negs).sum(-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    vec = F.normalize(centre.weight.detach(), dim=1).numpy()
    return (
        vocab,
        vec,
        {
            "vocab": len(vocab),
            "pairs_per_epoch": len(pairs),
            "loss": float(loss),
            "minutes": (time.perf_counter() - t0) / 60,
        },
    )


def skipgram_pairs(
    docs: list[list[str]], index: dict[str, int], keep: np.ndarray, rng: np.random.Generator
) -> torch.Tensor:
    """One epoch's (centre, context) pairs, shuffled: every word and each word 1 to 5 away.

    Args:
        docs: The training reports as lists of words.
        index: Each vocabulary word's row.
        keep: Each word's chance of being kept (frequent words are subsampled).
        rng: Draws the subsample and the order.

    Returns:
        The pairs, (n, 2) word rows, both directions, in random order.
    """
    pairs = []
    for d in docs:
        ids = np.array([index[w] for w in d if w in index])
        ids = ids[rng.random(len(ids)) < keep[ids]]
        for off in range(1, W2V_WINDOW + 1):
            pairs.append(np.stack([ids[:-off], ids[off:]], 1))
            pairs.append(np.stack([ids[off:], ids[:-off]], 1))
    return torch.tensor(np.concatenate(pairs))[
        torch.tensor(rng.permutation(sum(len(p) for p in pairs)))
    ]


def neighbours(vocab: list[str], vec: np.ndarray, words: tuple[str, ...], k: int = 4) -> dict:
    """The ``k`` nearest words (cosine) to each word in ``words``."""
    index = {w: i for i, w in enumerate(vocab)}
    out = {}
    for w in words:
        sim = vec @ vec[index[w]]
        order = [j for j in np.argsort(sim)[::-1] if j != index[w]][:k]
        out[w] = [(vocab[j], float(sim[j])) for j in order]
    return out


def masked_words(df: pd.DataFrame) -> dict[str, object]:
    """BERT fills the blank in "fell from a ___" in the test years' reports.

    Every report from 2023-2025 with "fell from/off a/an/the X" has X replaced by [MASK]; BERT
    (pretrained on books and Wikipedia, never on these reports) ranks its vocabulary. Only X
    that are one WordPiece can be guessed in one step; the others count as misses. The baseline
    guesses the training years' most frequent X.
    """
    seen = training(df).str.extract(FELL)[1].dropna().value_counts()
    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    mlm = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval().to(device())
    rows = blanks(df.loc[df["year"] > TRAIN_YEARS[1], "lower"], tok.mask_token)
    first = bert_guesses(rows, tok, mlm)
    answers = pd.Series([r[1] for r in rows])
    top1 = sum(g[0] == a for g, (_, a) in zip(first, rows, strict=True))
    top5 = sum(a in g for g, (_, a) in zip(first, rows, strict=True))
    guesses = pd.Series([g[0] for g in first]).value_counts()
    base = seen.index[0]
    enc = tok(BARE.lower().replace("[mask]", tok.mask_token), return_tensors="pt").to(device())
    with torch.no_grad():
        logits = mlm(**enc).logits[0, (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]]
    ranked = [tok.convert_ids_to_tokens(int(k)) for k in logits.topk(50).indices]
    return {
        "test_cases": len(rows),
        "single_wordpiece": sum(len(tok.tokenize(a)) == 1 for _, a in rows),
        "bert_top1": top1 / len(rows),
        "bert_top5": top5 / len(rows),
        "baseline_word": base,
        "baseline_top1": float((answers == base).mean()),
        "baseline_top5": float(answers.isin(seen.index[:5]).mean()),
        "answers_top": answers.value_counts().head(8).to_dict(),
        "bert_guesses_top": guesses.head(8).to_dict(),
        # Every answer and every first guess, counted: the figure draws these, not the top 8s
        "answers_all": answers.value_counts().to_dict(),
        "bert_guesses_all": guesses.to_dict(),
        "bare_sentence": BARE,
        "bare_top8": ranked[:8],
        "bare_ladder_rank": ranked.index("ladder") + 1 if "ladder" in ranked else None,
    }


def blanks(texts: pd.Series, mask: str) -> list[tuple[str, str]]:
    """Each text with "fell from/off a/an/the X": X replaced by ``mask``, and X.

    Args:
        texts: Report narratives.
        mask: The mask token to put in X's place.

    Returns:
        (masked text, X) for every text that has the phrase, in order.
    """
    rows = []
    for t in texts:
        m = FELL.search(t)
        if m is not None:
            rows.append((t[: m.start(2)] + mask + t[m.end(2) :], m.group(2)))
    return rows


def bert_guesses(
    rows: list[tuple[str, str]], tok: PreTrainedTokenizerBase, mlm: PreTrainedModel
) -> list[list[str]]:
    """BERT's five most likely WordPieces for each blank, best first (batches of 64).

    Args:
        rows: ``blanks``'s (masked text, answer) pairs.
        tok: BERT's tokenizer.
        mlm: BERT with its masked-word head, in eval mode.

    Returns:
        Five WordPieces per row.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            batch = rows[i : i + 64]
            enc = tok(
                [r[0] for r in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(mlm.device)
            logits = mlm(**enc).logits
            pos = (enc["input_ids"] == tok.mask_token_id).nonzero()
            assert len(pos) == len(batch), "one [MASK] per report"
            out += [
                [tok.convert_ids_to_tokens(int(k)) for k in logits[b, j].topk(5).indices]
                for b, j in pos.tolist()
            ]
    return out


def jaccard(a: str, b: str) -> float:
    """Shared words over all words (sets of lowercase words)."""
    x, y = set(LETTERS.findall(a)), set(LETTERS.findall(b))
    return len(x & y) / len(x | y)


def similarity_attempt(df: pd.DataFrame, vocab: list[str], vec: np.ndarray) -> dict[str, object]:
    """Three construction reports from the test years, and three ways to compare them.

    A: a fall to a lower level from a ladder. B: a fall from a scaffold, in other words (the
    fewest words shared with A). C: not a fall (no "fell" or "fall" in it either), but sharing the
    most words with A. Each pair is
    scored by shared words, by the mean of word2vec vectors, and by BERT's mean output token.
    """
    t = df[(df["year"] > TRAIN_YEARS[1]) & df["naics"].str.startswith("23")].copy()
    t["n"] = t["lower"].str.count(r"[a-z]+")
    t = t[t["n"].between(20, 40)]
    fall = t["event"].str.startswith("43")
    a = t[fall & t["lower"].str.contains("ladder")].sample(1, random_state=SEED).iloc[0]
    b_pool = t[fall & t["lower"].str.contains("scaffold") & ~t["lower"].str.contains("ladder")]
    b = b_pool.loc[b_pool["lower"].map(lambda x: jaccard(x, a["lower"])).idxmin()]
    c_pool = t[
        ~t["event"].str.startswith("4")
        & t["lower"].str.contains("ladder")
        & ~t["lower"].str.contains(r"\bfell\b|\bfall")
    ]
    c = c_pool.loc[c_pool["lower"].map(lambda x: jaccard(x, a["lower"])).idxmax()]
    texts = {"A": a["text"], "B": b["text"], "C": c["text"]}
    index = {w: i for i, w in enumerate(vocab)}

    def w2v(x: str) -> np.ndarray:
        v = vec[[index[w] for w in LETTERS.findall(x.lower()) if w in index]].mean(0)
        return v / np.linalg.norm(v)

    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    enc_model = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval()

    def bert(x: str) -> np.ndarray:
        enc = tok(x, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            h = enc_model.base_model(**enc).last_hidden_state[0]  # (tokens, 768)
        v = h[1:-1].mean(0).numpy()  # the mean of the word tokens ([CLS], [SEP] dropped)
        return v / np.linalg.norm(v)

    pairs = (("A", "B"), ("A", "C"), ("B", "C"))
    emb = {
        "word2vec": {k: w2v(v) for k, v in texts.items()},
        "bert": {k: bert(v) for k, v in texts.items()},
    }
    return {
        "texts": texts,
        "events": {"A": a["event"], "B": b["event"], "C": c["event"]},
        "shared_words": {f"{x}{y}": jaccard(texts[x].lower(), texts[y].lower()) for x, y in pairs},
        "word2vec": {f"{x}{y}": float(emb["word2vec"][x] @ emb["word2vec"][y]) for x, y in pairs},
        "bert": {f"{x}{y}": float(emb["bert"][x] @ emb["bert"][y]) for x, y in pairs},
    }


SIM_PAIRS = 200_000  # random pairs of test reports for "Similar to what?" over all reports


def bert_means(
    texts: list[str], tok: PreTrainedTokenizerBase, model: PreTrainedModel, batch: int = 128
) -> np.ndarray:
    """BERT's mean output token of each text, unit length: [CLS], [SEP], and padding left out.

    The batched form of ``similarity_attempt``'s ``bert`` (one text at a time, on the CPU).

    Args:
        texts: The texts.
        tok: BERT's tokenizer.
        model: BERT (its ``base_model`` gives the output tokens), in eval mode.
        batch: Texts per batch.

    Returns:
        (len(texts), 768) unit vectors.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i : i + batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(model.device)
            h = model.base_model(**enc).last_hidden_state  # (b, tokens, 768)
            m = enc["attention_mask"].clone()
            m[:, 0] = 0  # [CLS]
            m[torch.arange(len(m)), enc["attention_mask"].sum(1) - 1] = 0  # [SEP]
            v = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            out.append(F.normalize(v, dim=1).float().cpu().numpy())
    return np.concatenate(out)


def similarity_all(df: pd.DataFrame, vocab: list[str], vec: np.ndarray) -> dict[str, object]:
    """Do the three similarities track what happened? Every test report, random pairs.

    The attempt's three reports are a worked example; this is the measurement. Every report of
    2023-2025 with a classifiable OIICS event (division 9, nonclassifiable, left out) and at
    least one word in word2vec's vocabulary; ``SIM_PAIRS`` random pairs of different reports.
    For each similarity, the AUC with which it ranks pairs of the same event division (falls;
    contact with objects; ...) above pairs of different divisions (0.5: no better than a
    coin), and the same for the event group (two digits: falls to a lower level, 43).

    Returns:
        Reports, pairs, the share of same-division pairs, each similarity's AUCs, and each
        similarity's median over the pairs.
    """
    index = {w: i for i, w in enumerate(vocab)}
    t = df[(df["year"] > TRAIN_YEARS[1]) & ~df["event"].str.startswith("9")]
    words = [[index[w] for w in LETTERS.findall(x) if w in index] for x in t["lower"]]
    t = t[[len(w) > 0 for w in words]]
    words = [w for w in words if w]
    sets = [set(LETTERS.findall(x)) for x in t["lower"]]
    w2v = np.stack([vec[w].mean(0) for w in words])
    w2v /= np.linalg.norm(w2v, axis=1, keepdims=True)
    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    model = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval().to(device())
    bert = bert_means(t["text"].tolist(), tok, model)
    rng = np.random.default_rng((SEED, 19))
    i, j = rng.integers(0, len(t), size=(2, SIM_PAIRS))
    i, j = i[i != j], j[i != j]
    ev = t["event"].to_numpy()
    same = {
        "division": np.array([ev[a][0] == ev[b][0] for a, b in zip(i, j, strict=True)]),
        "group": np.array([ev[a][:2] == ev[b][:2] for a, b in zip(i, j, strict=True)]),
    }
    sim = {
        "shared_words": np.array([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in
                                  zip(i, j, strict=True)]),
        "word2vec": np.einsum("nd,nd->n", w2v[i], w2v[j]),
        "bert": np.einsum("nd,nd->n", bert[i], bert[j]),
    }  # fmt: skip
    return {
        "reports": len(t),
        "pairs": len(i),
        "same_division_share": float(same["division"].mean()),
        "same_group_share": float(same["group"].mean()),
        "auc_division": {k: float(roc_auc_score(same["division"], v)) for k, v in sim.items()},
        "auc_group": {k: float(roc_auc_score(same["group"], v)) for k, v in sim.items()},
        "median": {k: float(np.median(v)) for k, v in sim.items()},
    }


TRY_SENTENCE = "the worker held the ladder there, then the ladder fell"


def pair_counts(words: list[tuple[str, ...]]) -> list[tuple[str, int]]:
    """Adjacent symbol pairs and their counts, most frequent first."""
    c: dict[tuple[str, str], int] = {}
    for w in words:
        for pair in itertools.pairwise(w):
            c[pair] = c.get(pair, 0) + 1
    return [(f"{a} + {b}", n) for (a, b), n in sorted(c.items(), key=lambda kv: -kv[1])]


def try_it() -> dict[str, object]:
    """The slide's exercise: the first two merges of one sentence, with the counts behind them."""
    words = [tuple(w) for w in WORD.findall(TRY_SENTENCE)]
    first = pair_counts(words)[:3]
    merges = bpe_scratch(pd.Series([TRY_SENTENCE]), 2)
    merged = [merge_pair(w, merges[0]) for w in words]
    return {
        "sentence": TRY_SENTENCE,
        "first_counts": first,
        "second_counts": pair_counts(merged)[:3],
        "merges": [f"{x} + {y}" for x, y in merges],
    }


def masked_figure(mw: dict[str, object], out_dir: Path) -> None:
    """What workers fell from, in the test years, against BERT's first guesses.

    The words are the eight commonest answers and the eight commonest guesses (the first ten
    of them); each bar is that word's full count, as answer and as guess.
    """
    words = list(dict.fromkeys([*mw["answers_top"], *mw["bert_guesses_top"]]))[:10]
    answers, guesses = mw["answers_all"], mw["bert_guesses_all"]
    mw["plotted"] = {w: [answers.get(w, 0), guesses.get(w, 0)] for w in words}
    y = np.arange(len(words))
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.barh(
        y - 0.2,
        [answers.get(w, 0) for w in words],
        height=0.38,
        color=aml_style.PALETTE[0],
        label="what the report said",
    )
    ax.barh(
        y + 0.2,
        [guesses.get(w, 0) for w in words],
        height=0.38,
        color=aml_style.PALETTE[1],
        hatch="//",
        label="BERT's first guess",
    )
    ax.set_yticks(y, words, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel(f'"fell from a ___": reports of 2023-2025 (of {mw["test_cases"]:,})')
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    aml_style.save(fig, out_dir, "masked")


def title_art_figure(out_dir: Path) -> None:
    """The title art: a grid of patches, one of them a dark sun, joined by threads of light."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0d14", "#14101c", "#22101a", "#08070b"), grain=0.08)
    rng = np.random.default_rng(SEED)
    n = 7
    cells = [(0.08 + 0.12 * j, 0.10 + 0.12 * i) for i in range(n) for j in range(n)]
    for k, (x, y) in enumerate(cells):
        sq = [(x, y), (x + 0.1, y), (x + 0.1, y + 0.1), (x, y + 0.1)]
        c.glaze(
            c.polygon(sq, soft=3),
            ("#5a2216", "#3a1912", "#4a2a1a")[k % 3],
            0.6 + 0.3 * rng.random(),
        )
    hub = cells[24]
    centre = (hub[0] + 0.05, hub[1] + 0.05)
    lines = [[centre, (x + 0.05, y + 0.05)] for x, y in cells[::5]]
    c.glow(c.lines(lines, 1.6, soft=1.0), "#f3c98a", 0.45, spread=8)
    c.glaze(c.disc(centre, 0.045, soft=2), "#1d0a10", 0.95)
    rim = c.disc(centre, 0.055, soft=3) - c.disc(centre, 0.045, soft=3)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.6, spread=18)
    c.glow(c.disc((0.72, top * 0.82), 0.06, soft=4), "#e39a45", 0.35, spread=30)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a seven-by-seven grid of rust and ochre squares, "
        "like the patches a vision transformer cuts an image into; from a dark sun in the middle "
        "square, thin threads of golden light reach out to other squares, the way attention "
        "joins every token to every other.",
    )


def text_numbers(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Tokens, word vectors, the attempt and its measurement on every report, BERT's blanks.

    Args:
        df: The OSHA reports (``osha()``).
        out_dir: Where the masked-word figure goes.

    Returns:
        The text half of numbers.json.
    """
    numbers: dict[str, object] = {"tokens": tokens_numbers(df)}
    vocab, vec, numbers["word2vec"] = word2vec(df)
    numbers["neighbours"] = neighbours(vocab, vec, QUERY_WORDS)
    numbers["try_it"] = try_it()
    numbers["attempt"] = similarity_attempt(df, vocab, vec)
    numbers["similarity_all"] = similarity_all(df, vocab, vec)
    numbers["masked"] = masked_words(df)
    masked_figure(numbers["masked"], out_dir)
    return numbers


def vision_numbers(out_dir: Path) -> dict[str, object]:
    """ViT-B/16 against ResNet-18 on the crack tiles: the screen, the block, maps, the shuffle.

    Args:
        out_dir: Where the figures go.

    Returns:
        The vision half of numbers.json.
    """
    tiles, rfeat = load()
    vfeat = load_cache(
        VIT_FEATURES,
        VIT_CACHE | {"shape": [len(tiles), 768]},
        len(tiles),
        "scripts/derive_sdnet2018_vit.py",
    )
    part = photo_split(tiles, np.random.default_rng((0, 15)))  # Lecture 15's split, exactly
    y = tiles["cracked"].to_numpy()
    res, vit_s = screen(rfeat, y, part, tiles), screen(vfeat, y, part, tiles)
    numbers: dict[str, object] = {"screen": {"resnet18": res.numbers, "vit_b_16": vit_s.numbers}}
    decide_figure(res, vit_s, out_dir)
    numbers["verify"] = v = verify_block(VERIFY_TILE)
    v["millions"] = {k: v[f"{k}_weights"] / 1e6 for k in ("block", "vit", "resnet")}  # slides
    numbers["positions"] = positions_figure(out_dir)
    numbers["maps"] = attention_maps_figure(tiles, part, vit_s, out_dir)
    numbers["shuffle"] = sh = shuffle_test(tiles, part, rfeat, vfeat)
    decks = int((tiles["surface"] == "deck").sum())
    nets = ("resnet18", "vit_b_16")
    numbers["timing"] = {
        "deck_tiles": decks,
        "deck_photos": int(tiles.loc[tiles["surface"] == "deck", "photo"].nunique()),
        **{f"{m}_deck_seconds": decks * sh[m]["ms_per_tile"] / 1e3 for m in nets},
        "ratio": sh["vit_b_16"]["ms_per_tile"] / sh["resnet18"]["ms_per_tile"],
    }
    return numbers


def main() -> None:
    """Generate every figure and number for CEE Lecture 17."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # cuDNN runs convolutions in TF32 by default on this GPU (10-bit mantissas): ResNet-18's
    # features then differ from its cache, made on the CPU, by up to about 0.01. Off, a GPU pass
    # agrees with the cache to about 1e-5.
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    numbers: dict[str, object] = {"concrete": concrete_attention()}
    numbers["picture"] = attention_build(out_dir)
    numbers |= text_numbers(osha(), out_dir)
    print(f"text in {time.perf_counter() - t0:.0f} s", flush=True)
    numbers |= vision_numbers(out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("attention_3", "masked", "positions", "decide"))
    v = numbers["verify"]
    write_tex_macros(
        {
            "Tokens": str(v["tokens"]),
            "Dim": str(v["dim"]),
            "Heads": str(v["heads"]),
            "Blocks": str(v["blocks"]),
            "BlockWeights": f"{v['block_weights'] / 1e6:.2f}",
            "VitWeights": f"{v['vit_weights'] / 1e6:.1f}",
        },
        out_dir,
        Path(__file__).name,
    )
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
