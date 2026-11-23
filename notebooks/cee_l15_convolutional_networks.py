"""Applied Machine Learning for CEE — Lecture 15: Convolutions and Transfer Learning.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l15_convolutional_networks.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_sdnet2018.sh`` once (SDNET2018, Maguire, Dorafshan & Thomas, 2018,
CC BY 4.0), then ``pixi run -e aml-env python scripts/derive_sdnet2018.py`` (the tile index and
the ImageNet ResNet-18 features: one pass of the network over every tile, about 10 minutes on 14
CPU cores, longer on a laptop).
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Applied Machine Learning for CEE — Lecture 15: Convolutions and Transfer Learning

    **Slides:** *Applied Machine Learning for CEE*, Lecture 15 (`lectures/cee_l15_convolutional_networks`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §17.1 (pattern detection by convolution; layers upon layers) and §18.1.6–18.1.9 (batch normalization, residual networks).

    **Data:** SDNET2018 (Maguire, Dorafshan & Thomas, 2018, CC BY 4.0): 230 photographs of concrete
    bridge decks, walls, and pavements at Utah State University, cut into 56,092 tiles of 256
    pixels, each labelled cracked or not. Fetch with `bash scripts/fetch_sdnet2018.sh`, then
    `python scripts/derive_sdnet2018.py` (tile index and ResNet-18 features). The ImageNet
    weights (torchvision, `IMAGENET1K_V1`) are for research and education.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. It trains a
    small network from scratch 16 times (once on all photos; on 16 photos, five schedule lengths
    with three seeds each) and embeds every tile a second time at half resolution, all on the
    CPU: about 4 h 45 min on 14 cores with a 9 GB peak (measured 2026-11-21), most of it the
    16-photo network's length search. `L15_SETTING=reduced` trains that network at two
    lengths (33 and 132 passes) with one seed instead: about 50 min, also with a 9 GB peak.

    **Your tasks** (from the slides): call eight tiles yourself, write a convolution and its
    backward pass and match PyTorch, compare a network trained from scratch with ImageNet's
    features, then break the classifier three ways and set the inspector's threshold.
    """)
    return


@app.cell
def _():
    import copy
    import hashlib
    import os
    import random
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from joblib import Parallel, delayed
    from matplotlib.patches import Rectangle
    from PIL import Image
    from scipy.ndimage import correlate
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms.v2 import functional as tvf

    SEED = 0
    # "full" (the default) trains every network the slides quote; "reduced" skips the schedule-
    # length search and trains the 16-photo network at two lengths with seed 0 only
    SETTING = os.environ.get("L15_SETTING", "full")
    assert SETTING in ("full", "reduced"), SETTING
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    THREADS = 14  # as in the slides' figure script: the same sums in the same order
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    RAW = ROOT / "data" / "raw" / "sdnet2018" / "SDNET2018"
    TILES_CSV = ROOT / "data" / "derived" / "sdnet2018_tiles.csv"
    FEATURES = ROOT / "data" / "derived" / "sdnet2018_resnet18.npy"
    SURFACES = ("deck", "wall", "pavement")
    SPLIT = (0.6, 0.8)  # cumulative share of each surface's photos: train, val (rest: test)
    COLS, SIZE = 18, 224  # tiles per photo row; ResNet-18's input size
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet's statistics
    DARK_LINE = np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]) / 8.0
    FILTER_TILE, TEXTURE_TILE = "D/CD/7001-21.jpg", "P/CP/001-104.jpg"
    C_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)  # inverse L2 strengths tried on validation
    RECALL = 0.90  # the inspector's target
    HAND_CUT = 10.0  # the hand filter's "line-like" response
    # processes that decode tiles (joblib's loky workers can pickle cell functions); the order
    # of the results, not their count, fixes the numbers
    WORKERS = 4
    # a probe's C search widens past C_GRID's ends along this ladder while its pick is at an edge
    C_LADDER = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, *C_GRID, 3.0, 10.0, 100.0)
    SCRATCH_PX, SCRATCH_EPOCHS, SCRATCH_BATCH, SCRATCH_LR = 128, 12, 64, 3e-3
    # the 16-photo network's schedule length, in passes: a bracketing search on validation
    # (median of three seeds), widened along the ladder while the best length is at an edge;
    # 33 passes is the first recipe's 2,000 steps
    SMALL_LADDER, SMALL_START, SMALL_SEEDS = (8, 16, 33, 66, 132, 264), (16, 33, 66), (0, 1, 2)
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    return (
        C_GRID,
        C_LADDER,
        COLS,
        DARK_LINE,
        FEATURES,
        FILTER_TILE,
        HAND_CUT,
        Image,
        KNeighborsClassifier,
        LogisticRegression,
        MEAN,
        PALETTE,
        Parallel,
        RAW,
        RECALL,
        Rectangle,
        ResNet18_Weights,
        SCRATCH_BATCH,
        SCRATCH_EPOCHS,
        SCRATCH_LR,
        SCRATCH_PX,
        SEED,
        SETTING,
        SIZE,
        SMALL_LADDER,
        SMALL_SEEDS,
        SMALL_START,
        SPLIT,
        STD,
        SURFACES,
        StandardScaler,
        TEXTURE_TILE,
        TILES_CSV,
        WORKERS,
        average_precision_score,
        copy,
        correlate,
        delayed,
        hashlib,
        np,
        pd,
        plt,
        resnet18,
        roc_auc_score,
        time,
        torch,
        tvf,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–14

    On your own, no notes, two minutes:

    1. Lecture 14's network took 2 inputs. A colour tile of 224 × 224 pixels has 150,528. How
       many weights does a first layer of 64 units need?
    2. Lecture 8's forest scored R² +0.16 on random folds and 0.00 with whole sites held out.
       Why?
    3. Lecture 6: missing a crack costs 20 times a needless review. Where does the threshold on
       $p$ go?

    **Today:** one layer that covers the whole tile with 1,792 weights.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. $150{,}528 \times 64 + 64 = 9{,}633{,}856$ (checked in §3). Every unit looks at
                   every pixel, so a crack in a corner and in the centre need different weights.
                2. Rows from one site share its climate, traffic, and construction: with the
                   site in the training folds, the forest recognized it. Today's tiles come from
                   230 photographs, the same question one level down (§5).
                3. Flag when $20p > 1 - p$: $p > 1/21 \approx 0.048$. Today the inspector sets the
                   share of cracks to catch instead, and the threshold follows (§6).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    An inspector photographs a bridge deck, a wall, a pavement. **Can a network flag the 256-pixel
    tiles that show a crack, so the inspector reviews only those?** A missed crack is not recorded
    until the next inspection, routinely 24 months later; a false alarm costs seconds. So the
    inspector sets the target, **flag 9 of every 10 cracked tiles**, and we minimize the share of
    tiles to review. Reviewing everything is the baseline: it catches all, saves nothing.

    SDNET2018 has 230 photographs (54 decks, 72 walls, 104 pavements), cut into 56,092 tiles, 18
    to a row. A photo is the unit of collection, one camera position and one light, so **whole
    photos** go to training, validation, or test: 60/20/20% of each surface's photos.
    """)
    return


@app.cell
def _(FEATURES, SURFACES, TILES_CSV, np, pd):
    df = pd.read_csv(TILES_CSV)
    feats = np.load(FEATURES)  # ResNet-18 (ImageNet) pooled features, one row per tile
    assert feats.shape == (len(df), 512), feats.shape
    y = df["cracked"].to_numpy()
    _by = df.groupby("surface").agg(
        photos=("photo", "nunique"), tiles=("path", "size"), cracked=("cracked", "mean")
    )
    assert len(df) == 56_092 and df["photo"].nunique() == 230
    assert _by.loc[list(SURFACES), "photos"].tolist() == [54, 72, 104]
    assert round(float(y.mean()), 3) == 0.151
    assert _by.loc[list(SURFACES), "cracked"].round(3).tolist() == [0.149, 0.212, 0.107]
    _by
    return df, feats, y


@app.cell
def _(
    COLS, Image, MEAN, RAW, ResNet18_Weights, SIZE, SPLIT, STD, SURFACES, np, resnet18, torch, tvf
):
    def read_tile(path, gray=False):
        """One tile: uint8 (256, 256, 3), or grey levels 0-255 as floats."""
        im = Image.open(RAW / path)
        return np.array(im.convert("L"), dtype=np.float64) if gray else np.array(im.convert("RGB"))

    def mosaic(df, photo, px):
        """A photograph rebuilt from its tiles, each shrunk to px pixels; and the tiles placed."""
        g = df[df["photo"] == photo].sort_values("dup").drop_duplicates(["row", "col"])
        img = np.full(((int(g["row"].max()) + 1) * px, COLS * px, 3), 255, dtype=np.uint8)
        for r in g.itertuples():
            t = Image.open(RAW / r.path).convert("RGB").resize((px, px), Image.Resampling.BOX)
            img[r.row * px : (r.row + 1) * px, r.col * px : (r.col + 1) * px] = np.asarray(t)
        return img, g

    def preprocess(img):
        """A uint8 tile as ResNet-18's input: 224 x 224, ImageNet-normalized, (3, 224, 224)."""
        x = tvf.to_dtype(torch.from_numpy(img).permute(2, 0, 1), torch.float32, scale=True)
        return tvf.normalize(tvf.resize(x, [SIZE, SIZE], antialias=True), list(MEAN), list(STD))

    def backbone():
        """ImageNet ResNet-18 without its classifier: 512 pooled features (eval mode)."""
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        return model.eval()

    def photo_split(df, rng):
        """Whole photos to train/val/test, 60/20/20% of each surface's photos."""
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

    return backbone, mosaic, photo_split, preprocess, read_tile


@app.cell
def _(SEED, df, np, pd, photo_split):
    part = photo_split(df, np.random.default_rng((SEED, 15)))
    train, val, test = (part == r for r in ("train", "val", "test"))
    _counts = pd.crosstab(df["surface"], part)
    assert (~(train & val) & ~(train & test) & ~(val & test)).all()
    # no photo in two sets
    assert (
        df.groupby("photo").apply(lambda g: len(set(part[g.index])), include_groups=False).max()
        == 1
    )
    _counts
    return part, test, train, val


@app.cell
def _(Rectangle, df, mosaic, plt):
    # Three photographs rebuilt from their tiles, the tiles labelled cracked outlined
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for _ax, _photo in zip(_axes, ("D7021", "W7102", "P001"), strict=True):
        _img, _tiles = mosaic(df, _photo, 48)
        _ax.imshow(_img)
        for _r in _tiles[_tiles["cracked"] == 1].itertuples():
            _ax.add_patch(Rectangle((_r.col * 48, _r.row * 48), 48, 48, fill=False, ec="#D55E00"))
        _ax.set_title(f"{_photo}: {_tiles['cracked'].sum()} of {len(_tiles)} cracked")
        _ax.set_axis_off()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Attempt (two minutes).** Eight tiles from test photographs. For each, write *cracked* or
    *not* and how sure you are (1 to 3). Compare with a neighbour. The labels and the network's
    answers are in §5.
    """)
    return


@app.cell
def _(plt, read_tile):
    # A, B, ..., H: chosen in §5 by the probe's ranks, two sure catches, two sure passes, two
    # false alarms, and two misses, each pair from different surfaces; shuffled with a seed
    ATTEMPT = [
        "P/UP/092-23.jpg", "W/CW/7125-154.jpg", "P/CP/095-111.jpg", "D/UD/7041-178.jpg",
        "W/UW/7118-141.jpg", "D/CD/7004-140.jpg", "D/CD/7004-82.jpg", "P/UP/024-207.jpg",
    ]  # fmt: skip
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for _ax, _path, _letter in zip(_axes.ravel(), ATTEMPT, "ABCDEFGH", strict=True):
        _ax.imshow(read_tile(_path))
        _ax.set_title(_letter)
        _ax.set_axis_off()
    _fig
    return (ATTEMPT,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: a filter slides over the tile

    Nine weights, $+1$ around a centre of $-8$ (divided by 8), ask one question at a pixel: *is
    it darker than its eight neighbours?* A crack is a thin dark line. Multiply the nine grey
    levels by the nine weights and add; then do the same at every pixel. The same small set of
    weights applied at every position is a **convolution**.
    """)
    return


@app.cell
def _(DARK_LINE, FILTER_TILE, correlate, np, plt, read_tile):
    g_tile = read_tile(FILTER_TILE, gray=True)
    response = correlate(g_tile, DARK_LINE, mode="nearest")
    # the slide's window: the strongest response in the tile's central 64 x 64
    _i, _j = np.unravel_index(np.argmax(response[96:160, 96:160]), (64, 64))
    ci, cj = int(_i) + 96, int(_j) + 96
    patch = g_tile[ci - 1 : ci + 2, cj - 1 : cj + 2]
    _sum8 = float((patch * DARK_LINE * 8).sum())
    assert (ci, cj) == (96, 133)
    assert patch.astype(int).tolist() == [[96, 104, 147], [115, 89, 115], [133, 97, 107]]
    assert _sum8 == 202 and _sum8 / 8 == 25.25 == response[ci, cj]
    assert round(float(response.max()), 1) == 29.1
    print(f"window at ({ci}, {cj}): sum = {_sum8:.0f}, / 8 = {_sum8 / 8}")
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _axes[0].imshow(g_tile, cmap="gray")
    _axes[1].imshow(np.clip(response, 0, None), cmap="magma", vmax=np.percentile(response, 99.5))
    _axes[1].plot(cj, ci, "o", mfc="none", mec="white", ms=10, mew=2)
    for _ax, _t in zip(_axes, ("deck tile", "response everywhere"), strict=True):
        _ax.set_title(_t)
        _ax.set_axis_off()
    _fig
    return (g_tile,)


@app.cell
def _(DARK_LINE, HAND_CUT, TEXTURE_TILE, correlate, g_tile, np, read_tile):
    # The same filter on a pavement tile: aggregate texture is "darker than its neighbours" too
    _share = {
        name: float((np.clip(correlate(g, DARK_LINE, mode="nearest"), 0, None) > HAND_CUT).mean())
        for name, g in (("deck", g_tile), ("pavement", read_tile(TEXTURE_TILE, gray=True)))
    }
    assert {k: round(v, 3) for k, v in _share.items()} == {"deck": 0.008, "pavement": 0.106}
    print({k: f"{100 * v:.1f}% of pixels above {HAND_CUT:.0f}" for k, v in _share.items()})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption: a crack looks alike anywhere in the tile.** Two parts: *locality* (whether a
    pixel belongs to a crack depends on its neighbourhood) and *stationarity* (the same pattern
    means the same thing at every position). Together they give **weight sharing**: one filter
    $w$ of size $k\times k$ over $C$ channels, applied at every position,

    $$y_{f,i,j} = b_f + \sum_{c=1}^{C}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} w_{f,c,u,v}\,x_{c,\,i+u,\,j+v}.$$

    PyTorch, like most of machine learning, calls this a convolution; signal processing would
    flip $w$ first and call this a cross-correlation (Forsyth §17.1.1 makes the same choice). Its
    consequence: **shift the input, and the output shifts with it** (translation equivariance).
    """)
    return


@app.cell
def _(DARK_LINE, correlate, g_tile, np, resnet18):
    # Weights of a first layer of 64 units on a 3 x 224 x 224 tile
    _dense = 3 * 224 * 224 * 64 + 64
    _conv = 3 * 3 * 3 * 64 + 64
    _r18 = resnet18().conv1.weight.numel()  # 7 x 7 x 3 x 64, no bias (batch norm follows)
    assert (_dense, _conv, _r18) == (9_633_856, 1_792, 9_408)
    # Equivariance: filter a tile shifted 32 px right; compare with the shifted map, away from
    # the borders (np.roll wraps the left edge around)
    _shift = 32
    _a = correlate(g_tile, DARK_LINE, mode="nearest")
    _b = correlate(np.roll(g_tile, _shift, axis=1), DARK_LINE, mode="nearest")
    _gap = float(np.abs(np.roll(_a, _shift, axis=1) - _b)[:, _shift + 2 : -2].max())
    assert _gap == 0.0
    print(f"dense {_dense:,}, conv {_conv:,}, ResNet-18 conv1 {_r18:,}; shift check: {_gap}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Stacks see farther.** A $3\times3$ layer widens what one unit sees by 2 steps of the layer
    below; a stride-2 layer doubles the step. Theory, from ResNet-18's layer sizes; in effect, the
    gradient of a central unit with respect to the input pixels concentrates in about half of it
    (Luo et al., 2016).
    """)
    return


@app.cell
def _(SEED, backbone, df, np, preprocess, read_tile, torch):
    r18 = backbone()

    def theoretical_rf(model):
        """Receptive field after each stage, along the main path (shortcuts widen nothing)."""
        rf, jump, out = 1, 1, {}
        for k, s in ((model.conv1.kernel_size[0], model.conv1.stride[0]),
                     (model.maxpool.kernel_size, model.maxpool.stride)):  # fmt: skip
            rf, jump = rf + (k - 1) * jump, jump * s
        for name in ("layer1", "layer2", "layer3", "layer4"):
            for block in getattr(model, name):
                for conv in (block.conv1, block.conv2):
                    rf += (conv.kernel_size[0] - 1) * jump
                    jump *= conv.stride[0]
            out[name] = rf
        return out

    def effective_width(model, x, name, mass=0.9):
        """Side of the square, about the centre of mass, holding `mass` of |d unit / d pixels|."""
        acts = {}
        hook = getattr(model, name).register_forward_hook(lambda _m, _i, o: acts.update(a=o))
        xx = x.clone().requires_grad_(True)
        model(xx)
        hook.remove()
        c = acts["a"].shape[-1] // 2
        # the gradient for the input only: no .grad is left on the network's weights
        (grad,) = torch.autograd.grad(acts["a"][:, :, c, c].sum(), xx)
        g = grad.abs().sum(1).mean(0).numpy()
        g = g / g.sum()
        yy, xg = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
        cy, cx = (g * yy).sum(), (g * xg).sum()
        d = np.maximum(np.abs(yy - cy), np.abs(xg - cx)).ravel()
        order = np.argsort(d, kind="stable")
        return int(round(2 * d[order][np.searchsorted(np.cumsum(g.ravel()[order]), mass)] + 1))

    _rng = np.random.default_rng((SEED, 15, 1))
    _x = torch.stack([preprocess(read_tile(p))
                      for p in df["path"].to_numpy()[_rng.choice(len(df), 32, replace=False)]])  # fmt: skip
    _theory = theoretical_rf(r18)
    _eff = {n: effective_width(r18, _x, n) for n in _theory}
    assert _theory == {"layer1": 43, "layer2": 99, "layer3": 211, "layer4": 435}
    assert _eff == {"layer1": 25, "layer2": 50, "layer3": 115, "layer4": 208}
    print("theory:", _theory, "\neffective (90%):", _eff)
    return (r18,)


@app.cell
def _(DARK_LINE, FILTER_TILE, correlate, np, preprocess, r18, read_tile, torch, tvf):
    # ResNet-18 stage by stage: halve the map, double the channels
    _names = ("relu", "layer1", "layer2", "layer3", "layer4")
    _acts = {}
    _hooks = [getattr(r18, n).register_forward_hook(lambda _m, _i, o, n=n: _acts.update({n: o}))
              for n in _names]  # fmt: skip
    with torch.no_grad():
        r18(preprocess(read_tile(FILTER_TILE))[None])
    for _h in _hooks:
        _h.remove()
    _shapes = {n: tuple(_acts[n].shape[1:]) for n in _names}
    assert _shapes == {"relu": (64, 112, 112), "layer1": (64, 56, 56), "layer2": (128, 28, 28),
                       "layer3": (256, 14, 14), "layer4": (512, 7, 7)}  # fmt: skip
    _total = sum(p.numel() for p in r18.parameters())
    _l4 = sum(p.numel() for p in r18.layer4.parameters())
    assert (_total, _l4) == (11_176_512, 8_393_728)  # without the 1,000-class layer
    print(_shapes, f"\n{_total:,} weights; layer4 {_l4:,} ({100 * _l4 / _total:.0f}%)")
    # at each stage, the channel whose map correlates best with the hand filter's map
    _dark = np.clip(correlate(read_tile(FILTER_TILE, gray=True), DARK_LINE, mode="nearest"), 0,
                    None)  # fmt: skip
    _r = {}
    for _n in _names:
        _a = _acts[_n][0].numpy()
        _target = tvf.resize(torch.from_numpy(_dark)[None], [_a.shape[-1]] * 2, antialias=True)[0]
        _c = [np.corrcoef(_m.ravel(), _target.numpy().ravel())[0, 1] if _m.std() > 0 else -1
              for _m in _a]  # fmt: skip
        _r[_n] = round(float(max(_c)), 2)
    print("best r per stage:", _r)
    assert _r == {"relu": 0.79, "layer1": 0.73, "layer2": 0.88, "layer3": 0.87, "layer4": 0.76}
    return


@app.cell
def _(DARK_LINE, FILTER_TILE, SIZE, correlate, np, preprocess, r18, read_tile, torch, tvf):
    # ImageNet's first layer: which of its 64 filters (7 x 7 x 3) follow the deck crack? Each
    # filter's map against the hand-made filter's, at the first layer's resolution (stride 2)
    with torch.no_grad():
        _maps = torch.relu(r18.bn1(r18.conv1(preprocess(read_tile(FILTER_TILE))[None])))[0].numpy()
    _g = tvf.resize(torch.from_numpy(read_tile(FILTER_TILE, gray=True))[None], [SIZE, SIZE])
    _dark = np.clip(correlate(_g[0].numpy(), DARK_LINE, mode="nearest"), 0, None)[::2, ::2]
    _corr = np.array([np.corrcoef(_m.ravel(), _dark.ravel())[0, 1] if _m.std() > 0 else -1.0
                      for _m in _maps])  # fmt: skip
    _best = np.argsort(-_corr, kind="stable")[:2]
    print("filters", _best.tolist(), "r", _corr[_best].round(2).tolist())
    assert _best.tolist() == [49, 45] and _corr[_best].round(2).tolist() == [0.76, 0.63]
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Skips keep the gradient in range.** A residual block outputs $\mathrm{relu}(x + F(x))$
    (He et al., 2016; Forsyth §18.1.9), so its Jacobian is $I + \partial F/\partial x$: the
    gradient reaches earlier blocks along the identity. Below, a toy network on 64 real tiles at
    initialization: the gradient's size at each block's input, divided by the last block's.
    """)
    return


@app.cell
def _(SEED, df, np, read_tile, torch, tvf):
    class Block(torch.nn.Module):
        """Two 3 x 3 conv + batch-norm layers, with or without the identity shortcut."""

        def __init__(self, channels, residual):
            super().__init__()
            nn = torch.nn
            self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.b1, self.b2 = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
            self.residual = residual

        def forward(self, x):
            h = self.b2(self.c2(torch.relu(self.b1(self.c1(x)))))
            return torch.relu(x + h if self.residual else h)

    _rng = np.random.default_rng((SEED, 15, 2))
    _idx = np.concatenate([_rng.choice(np.flatnonzero(df["cracked"] == c), 32, replace=False)
                           for c in (0, 1)])  # fmt: skip
    _x = torch.stack([
        tvf.resize(tvf.to_dtype(torch.from_numpy(read_tile(df["path"].iloc[i])).permute(2, 0, 1),
                                torch.float32, scale=True), [128, 128], antialias=True)
        for i in _idx
    ])  # fmt: skip
    _x = (_x - _x.mean((0, 2, 3), keepdim=True)) / _x.std((0, 2, 3), keepdim=True)
    _y = torch.from_numpy(df["cracked"].to_numpy()[_idx])

    def gradient_flow(blocks, residual, seed):
        """Gradient norm at each block's input over the last block's, at initialization."""
        torch.manual_seed(seed)
        nn = torch.nn
        stem = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(16), nn.ReLU())  # fmt: skip
        body = [Block(16, residual) for _ in range(blocks)]
        head = nn.Linear(16, 2)
        h, inputs = stem(_x), []
        for b in body:
            h.retain_grad()
            inputs.append(h)
            h = b(h)
        torch.nn.functional.cross_entropy(head(h.mean((2, 3))), _y).backward()
        g = np.array([v.grad.norm().item() for v in inputs])
        return g / g[-1]

    first_over_last = {
        (d, res): round(float(np.median([gradient_flow(d, res, s)[0] for s in range(5)])), 1)
        for d in (8, 16, 32)
        for res in (False, True)
    }
    assert first_over_last == {(8, False): 11.1, (8, True): 8.4, (16, False): 174.9,
                               (16, True): 14.8, (32, False): 36483.1, (32, True): 30.8}  # fmt: skip
    print("first block / last block, median of 5 seeds:", first_over_last)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Convolution from scratch.** `sliding_window_view` lists every $k\times k$ window without
    copying; one `einsum` multiplies each window by each filter and sums. The backward pass:
    each weight's gradient sums (window × output gradient) over every position; each pixel's
    gradient collects the output gradients of every window that read it.

    **Predict first:** against `torch.nn.functional.conv2d`, how many digits agree? And if we
    flip the filter, as the signal-processing definition does?
    """)
    return


@app.cell
def _(np):
    def conv2d(x, w, b):
        """x (N, C, H, W), w (F, C, k, k), b (F,) -> (N, F, H-k+1, W-k+1), and the windows."""
        k = w.shape[-1]
        win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
        # win (N, C, H-k+1, W-k+1, k, k): every k x k window of every channel
        return np.einsum("nchwij,fcij->nfhw", win, w) + b[None, :, None, None], win

    def conv2d_back(g, w, win):
        """g = dL/dy (N, F, H', W') -> dL/dx, dL/dw, dL/db."""
        n, c, ho, wo, k, _ = win.shape
        dw = np.einsum("nchwij,nfhw->fcij", win, g)
        db = g.sum(axis=(0, 2, 3))
        dx = np.zeros((n, c, ho + k - 1, wo + k - 1))
        for i in range(k):  # tap (i, j) read pixel (h + i, w + j) for output (h, w)
            for j in range(k):
                dx[:, :, i : i + ho, j : j + wo] += np.einsum("nfhw,fc->nchw", g, w[:, :, i, j])
        return dx, dw, db

    return conv2d, conv2d_back


@app.cell
def _(SEED, conv2d, conv2d_back, np, torch):
    _rng = np.random.default_rng((SEED, 15, 7))
    _x, _w, _b = (_rng.normal(size=s) for s in ((4, 3, 32, 32), (8, 3, 3, 3), (8,)))
    _y, _win = conv2d(_x, _w, _b)
    assert _y.shape == (4, 8, 30, 30) and _win.shape == (4, 3, 30, 30, 3, 3)
    _g = _rng.normal(size=_y.shape)
    _dx, _dw, _db = conv2d_back(_g, _w, _win)
    _xt, _wt, _bt = (torch.tensor(a, requires_grad=True) for a in (_x, _w, _b))
    _yt = torch.nn.functional.conv2d(_xt, _wt, _bt)
    _yt.backward(torch.tensor(_g))
    _flip, _ = conv2d(_x, np.ascontiguousarray(_w[:, :, ::-1, ::-1]), _b)
    conv_check = {
        "y": float(np.abs(_y - _yt.detach().numpy()).max()),
        "dx": float(np.abs(_dx - _xt.grad.numpy()).max()),
        "dw": float(np.abs(_dw - _wt.grad.numpy()).max()),
        "db": float(np.abs(_db - _bt.grad.numpy()).max()),
        "flipped": float(np.abs(_flip - _yt.detach().numpy()).max()),
        "y_sd": float(_y.std()),
    }
    # float64: agreement to round-off; the flipped filter is a different operation (a miss of
    # 36.7 against outputs whose standard deviation is 5.5)
    assert max(conv_check[k] for k in ("y", "dx", "dw", "db")) < 1e-12
    assert round(conv_check["flipped"], 1) == 36.7 and round(conv_check["y_sd"], 1) == 5.5
    print({k: f"{v:.0e}" for k, v in conv_check.items()})
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Transfer learning.** ImageNet's ResNet-18 (He et al., 2016), frozen: each tile's 512 pooled
    features, then Lecture 6's logistic regression with an L2 penalty, its strength $C$ chosen
    on the validation photos. `scripts/derive_sdnet2018.py` cached the features for all 56,092
    tiles; first, recompute some and check the cache.
    """)
    return


@app.cell
def _(Image, Parallel, WORKERS, backbone, copy, delayed, df, feats, np, preprocess, read_tile,
      torch):  # fmt: skip
    def tile_at(path, factor):
        """A tile shrunk to 256 / factor px (box filter), uint8."""
        img = read_tile(path)
        if factor > 1:
            img = np.array(Image.fromarray(img).resize((256 // factor,) * 2, Image.Resampling.BOX))
        return img

    def embed(model, paths, factor=1):
        """ResNet-18 features of tiles, each first shrunk to 256 / factor px if factor > 1.

        Tiles are decoded in worker processes; the network runs on batches of 128, on a
        channels-last copy of `model` (the caller's model is left as it was)."""
        model = copy.deepcopy(model).to(memory_format=torch.channels_last)
        out = []
        with torch.inference_mode():
            for i in range(0, len(paths), 2048):  # a multiple of 128: the same batches
                imgs = Parallel(n_jobs=WORKERS, batch_size=64)(
                    delayed(tile_at)(q, factor) for q in paths[i : i + 2048]
                )
                for j in range(0, len(imgs), 128):
                    x = torch.stack([preprocess(im) for im in imgs[j : j + 128]])
                    out.append(model(x.to(memory_format=torch.channels_last)).numpy())
        f = np.concatenate(out).astype(np.float32)
        assert f.shape == (len(paths), 512), f.shape
        return f

    r18_feat = backbone()
    _rows = np.arange(0, len(df), len(df) // 64)[:64]
    _gap = float(
        np.abs(embed(r18_feat, df["path"].to_numpy()[_rows].tolist()) - feats[_rows]).max()
    )
    assert _gap < 1e-3, _gap
    print(f"cache agrees within {_gap:.1e} on {len(_rows)} tiles")
    return embed, r18_feat


@app.cell
def _(
    C_GRID,
    C_LADDER,
    LogisticRegression,
    RECALL,
    StandardScaler,
    average_precision_score,
    np,
    roc_auc_score,
):
    class Probe:
        """Logistic regression (L2) on standardized features."""

        def __init__(self, scaler, model, c, val_ap):
            self.scaler, self.model, self.C, self.val_ap = scaler, model, c, val_ap

        def score(self, f):
            return self.model.predict_proba(self.scaler.transform(f))[:, 1]

        def logit(self, f):
            """Exact log-odds (a probability can round to 0 or 1)."""
            return self.model.decision_function(self.scaler.transform(f))

    def fit_probe(f, y, train, val):
        """Fit for each C in C_GRID on train; keep the best average precision on val.

        As the slides' probes.fit_probe, plus the widening below: when the kept C is
        interior (every probe of the slides but the pixel probe), the result is the same."""
        scaler = StandardScaler().fit(f[train])
        zt, zv = scaler.transform(f[train]), scaler.transform(f[val])
        best = None
        for c in C_GRID:
            # l1_ratio=0 is the L2 penalty (scikit-learn 1.8 deprecates penalty="l2")
            m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
            m.fit(zt, y[train])
            ap = average_precision_score(y[val], m.predict_proba(zv)[:, 1])
            if best is None or ap > best[0]:
                best = (ap, c, m)
        probe = Probe(scaler, best[2], best[1], float(best[0]))
        # a search whose pick sits at the grid's edge has not found its best: widen it along
        # C_LADDER, one value past that edge at a time, until the pick has neighbours both sides
        tried = list(C_GRID)
        while probe.C in (tried[0], tried[-1]):
            kept = probe.C
            k = C_LADDER.index(kept) + (-1 if kept <= tried[0] else 1)
            assert 0 <= k < len(C_LADDER), f"C = {probe.C} at the end of the ladder"
            c = C_LADDER[k]
            m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
            ap = average_precision_score(y[val], m.fit(zt, y[train]).predict_proba(zv)[:, 1])
            tried = sorted([*tried, c])
            if ap > probe.val_ap:
                probe = Probe(scaler, m, c, float(ap))
        return probe

    def threshold_for(p, y, recall=RECALL):
        """The highest threshold that flags at least `recall` of the cracked tiles (p >= t).

        Flag the top ceil(recall n) scores; floor((1 - 0.9) n) would be one too small whenever
        n is a multiple of 10 (1 - 0.9 is 0.0999... in floating point)."""
        pos = np.sort(p[y == 1])
        return float(pos[len(pos) - int(np.ceil(recall * len(pos) - 1e-9))])

    def scores(p, y, t):
        flag = p >= t
        return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                "recall": float(flag[y == 1].mean()), "review": float(flag.mean()),
                "acc": float(((p >= 0.5) == y).mean()), "prevalence": float(y.mean())}  # fmt: skip

    return fit_probe, scores, threshold_for


@app.cell
def _(SURFACES, df, feats, fit_probe, np, pd, scores, test, threshold_for, train, val, y):
    probe = fit_probe(feats, y, train, val)
    p = probe.score(feats)
    t = threshold_for(p[val], y[val])  # set on validation photos: flag 90% of their cracks
    probe_scores = {"all": scores(p[test], y[test], t)}
    for _s in SURFACES:
        _m = test & (df["surface"] == _s).to_numpy()
        probe_scores[_s] = scores(p[_m], y[_m], t)
    _table = pd.DataFrame(probe_scores).T.round(3)
    print(f"C = {probe.C}, validation AP {probe.val_ap:.3f}, threshold {t:.3f}")
    assert np.isfinite(p).all()
    assert probe.C == 0.03 and round(probe.val_ap, 3) == 0.737 and round(t, 3) == 0.057
    _want = {"all": (0.842, 0.666, 0.862, 0.506, 0.906, 0.144), "deck": (0.827, 0.655, 0.911, 0.66, 0.887, 0.166),
             "wall": (0.857, 0.744, 0.916, 0.655, 0.917, 0.162), "pavement": (0.835, 0.59, 0.77, 0.312, 0.909, 0.119)}  # fmt: skip
    for _k, _v in _want.items():
        _got = tuple(
            round(probe_scores[_k][m], 3)
            for m in ("auc", "ap", "recall", "review", "acc", "prevalence")
        )
        assert _got == _v, (_k, _got)
    _table
    return p, probe, probe_scores, t


@app.cell
def _(SEED, SURFACES, average_precision_score, df, np, p, part, t, test, y):
    # The default threshold, p >= 0.5, against the inspector's: what share of cracks it flags
    _half = p[test] >= 0.5
    at_half = {"recall": float(_half[y[test] == 1].mean()), "review": float(_half.mean())}
    assert (round(100 * at_half["recall"]), round(100 * at_half["review"])) == (50, 9)
    # How sure are the test numbers? Resample whole test photos (the unit of collection) within
    # each surface; the threshold stays fixed (its own noise is what the five splits show)
    _photos, _surface = df["photo"].to_numpy(), df["surface"].to_numpy()
    _by = {s: sorted(set(_photos[(part == "test") & (_surface == s)])) for s in SURFACES}
    _rows_of = {ph: np.flatnonzero((part == "test") & (_photos == ph))
                for s in SURFACES for ph in _by[s]}  # fmt: skip
    _rng = np.random.default_rng((SEED, 15, 9))
    _draws = {"recall": [], "review": [], "ap": []}
    for _ in range(2000):
        _picks = [ph for s in SURFACES for ph in _rng.choice(_by[s], len(_by[s]))]
        _r = np.concatenate([_rows_of[ph] for ph in _picks])
        _draws["recall"].append(float(np.mean(p[_r][y[_r] == 1] >= t)))
        _draws["review"].append(float(np.mean(p[_r] >= t)))
        _draws["ap"].append(float(average_precision_score(y[_r], p[_r])))
    bootstrap = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in _draws.items()}
    bootstrap["share_recall_90"] = float(np.mean(np.array(_draws["recall"]) >= 0.9))
    print(at_half, {k: np.round(v, 3).tolist() for k, v in bootstrap.items()})
    assert {k: np.round(bootstrap[k], 2).tolist() for k in ("recall", "review", "ap")} == {
        "recall": [0.83, 0.89],
        "review": [0.47, 0.54],
        "ap": [0.62, 0.71],
    }
    assert round(bootstrap["share_recall_90"], 3) == 0.006
    return at_half, bootstrap


@app.cell
def _(DARK_LINE, HAND_CUT, Parallel, WORKERS, correlate, delayed, df, fit_probe, np, read_tile,
      scores, test, threshold_for, train, val, y):  # fmt: skip
    # The naive baseline: the hand filter's summaries (share of line-like pixels, mean, 99th
    # percentile) under the same logistic regression
    def hand_features(path):
        r = np.clip(correlate(read_tile(path, gray=True), DARK_LINE, mode="nearest"), 0, None)
        return [float((r > HAND_CUT).mean()), float(r.mean()), float(np.percentile(r, 99))]

    hand = np.array(
        Parallel(n_jobs=WORKERS, batch_size=512)(delayed(hand_features)(q) for q in df["path"])
    )
    _hp = fit_probe(hand, y, train, val)
    _ph = _hp.score(hand)
    hand_scores = scores(_ph[test], y[test], threshold_for(_ph[val], y[val]))
    print({k: round(v, 3) for k, v in hand_scores.items()})
    assert tuple(round(hand_scores[m], 3) for m in ("ap", "recall", "review")) == (
        0.274,
        0.912,
        0.836,
    )
    return (hand_scores,)


@app.cell
def _(Image, Parallel, RAW, SEED, SURFACES, WORKERS, delayed, df, feats, fit_probe, np, part, plt,
      roc_auc_score, y):  # fmt: skip
    # Can a line separate cracks? In pixels (64 x 64 grey levels, 4,096 numbers) vs features. Both
    # probes are fitted on the same 6,000 training and 2,000 validation tiles, so the two AUCs
    # compare like with like; only the rows used are decoded.
    def pixels64(path):
        im = Image.open(RAW / path).convert("L").resize((64, 64), Image.Resampling.BOX)
        return np.asarray(im, dtype=np.float32).ravel()

    _rng = np.random.default_rng((SEED, 15, 6))
    _tr = np.sort(_rng.choice(np.flatnonzero(part == "train"), 6000, replace=False))
    _va = np.sort(_rng.choice(np.flatnonzero(part == "val"), 2000, replace=False))
    _te = np.flatnonzero(part == "test")
    _show = np.sort(np.concatenate([_rng.choice(_te[y[_te] == c], 300, replace=False)
                                    for c in (0, 1)]))  # fmt: skip
    _rows = np.concatenate([_tr, _va, _te])
    _pix = np.array(Parallel(n_jobs=WORKERS, batch_size=256)(
        delayed(pixels64)(q) for q in df["path"].to_numpy()[_rows]))  # fmt: skip
    assert _pix.shape == (len(_rows), 4096), _pix.shape
    _k = np.arange(len(_rows))
    _mtr, _mva = _k < len(_tr), (_k >= len(_tr)) & (_k < len(_tr) + len(_va))
    _mte = _k >= len(_tr) + len(_va)
    _at_show = len(_tr) + len(_va) + np.searchsorted(_te, _show)  # shown tiles' rows in _pix
    assert np.array_equal(_rows[_at_show], _show)
    _pp = fit_probe(_pix, y[_rows], _mtr, _mva)
    _fp = fit_probe(feats[_rows], y[_rows], _mtr, _mva)
    space_auc = {"pixels": float(roc_auc_score(y[_te], _pp.score(_pix[_mte]))),
                 "features": float(roc_auc_score(y[_te], _fp.score(feats[_te]))),
                 "pixels_C": _pp.C, "features_C": _fp.C}  # fmt: skip
    print(space_auc)
    # each test tile placed by its probe's log-odds; a line separates the classes if they split.
    # The first principal component (the direction of largest spread) on the y axis: how well
    # does it follow the label, the surface, the brightness?
    _bright = _pix[_at_show].mean(axis=1)
    _surf = df["surface"].to_numpy()[_show]
    pc1_stats = {}
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for _ax, (_name, _rtr, _rsh, _probe) in zip(_axes, (
        ("pixels", _pix[: len(_tr)], _pix[_at_show], _pp),
        ("features", feats[_tr], feats[_show], _fp),
    ), strict=True):  # fmt: skip
        _z = (_rsh - _rtr.mean(0)) / _rtr.std(0)  # standardized with the training rows
        _u, _sv, _ = np.linalg.svd(_z - _z.mean(0), full_matrices=False)
        _pc1 = _u[:, 0] * _sv[0]
        _auc = float(roc_auc_score(y[_show], _pc1))  # a component's sign is arbitrary
        _between = sum((_surf == s).sum() * (_pc1[_surf == s].mean() - _pc1.mean()) ** 2
                       for s in SURFACES)  # fmt: skip
        pc1_stats[_name] = {
            "label_auc": max(_auc, 1 - _auc),
            "surface_eta2": float(_between / ((_pc1 - _pc1.mean()) ** 2).sum()),
            "brightness_r": abs(float(np.corrcoef(_pc1, _bright)[0, 1])),
        }
        _logit = _probe.logit(_rsh)
        for _c, _mk in ((0, "o"), (1, "^")):
            _m = y[_show] == _c
            _ax.scatter(_logit[_m], _pc1[_m], marker=_mk, alpha=0.5, s=12,
                        label="cracked" if _c else "uncracked")  # fmt: skip
        _ax.set_title(f"{_name}: AUC {space_auc[_name]:.2f}")
        _ax.set_xlabel("logistic score (log-odds)")
        _ax.set_ylabel("1st principal component")
    _axes[1].legend()
    print({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in pc1_stats.items()})
    assert round(space_auc["pixels"], 2) == 0.54 and round(space_auc["features"], 2) == 0.83
    assert {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in pc1_stats.items()} == {
        "pixels": {"label_auc": 0.56, "surface_eta2": 0.22, "brightness_r": 1.0},
        "features": {"label_auc": 0.62, "surface_eta2": 0.14, "brightness_r": 0.3},
    }
    assert space_auc["pixels_C"] == 1e-3 and space_auc["features_C"] == 0.01  # both interior
    _fig
    return pc1_stats, space_auc


@app.cell
def _(mo):
    mo.md(r"""
    **A network trained from scratch**, for comparison: four conv–BN–ReLU–pool stages (16 to 128
    channels) on 128-pixel tiles, from a random start, AdamW with a one-cycle schedule, flips as
    augmentation, the cracked class weighted by the class ratio. Validation AP after every pass;
    the test tiles once, at the end. Trained on all 137 training photos (12 passes over 33,418
    tiles; one seed) and on 10% of them (16 photos; three seeds). On 16 photos the first recipe
    stopped at 2,000 steps (33 passes) without a check that validation had stopped improving,
    so the schedule's length is then chosen on validation too: a bracketing search over 8 to
    264 passes, widened while the best length is at an edge. Transfer learning's claim is about
    the 16-photo case. The network sees 128-pixel tiles, so the probe is also fitted on
    ResNet-18 features of the tiles shrunk to 128 pixels (a box filter, then upsampled to 224;
    the network's tiles are shrunk bilinearly). The resolution is then the same, but the two
    still differ in more than pretraining: a linear probe with 513 weights and its $C$ searched
    on validation, against a 97,809-weight network with one fixed recipe.
    """)
    return


@app.cell
def _(SEED, SURFACES, df, np, part):
    def budget_photos(frac):
        """A share of each surface's training photos, nested (a larger share contains a smaller)."""
        keep = []
        for s in SURFACES:
            on = (part == "train") & (df["surface"] == s).to_numpy()
            order = np.random.default_rng((SEED, 15, 8)).permutation(
                np.array(sorted(df["photo"][on].unique()))
            )
            keep += list(order[: int(np.ceil(frac * len(order)))])
        return (part == "train") & df["photo"].isin(keep).to_numpy()

    return (budget_photos,)


@app.cell
def _(df, embed, fit_probe, part, r18_feat, scores, threshold_for, time, y):
    # The same probe on the tiles shrunk to 128 px, the resolution the network from scratch sees
    _t0 = time.perf_counter()
    feats128 = embed(r18_feat, df["path"].tolist(), factor=2)
    _pr = fit_probe(feats128, y, part == "train", part == "val")
    _p = _pr.score(feats128)
    _t = threshold_for(_p[part == "val"], y[part == "val"])
    probe128 = {"C": _pr.C, "val_ap": _pr.val_ap,
                "test": scores(_p[part == "test"], y[part == "test"], _t)}  # fmt: skip
    print(f"embedded in {(time.perf_counter() - _t0) / 60:.0f} min;",
          {k: round(v, 3) for k, v in probe128["test"].items()}, "C =", _pr.C)  # fmt: skip
    assert _pr.C == 0.01 and round(_pr.val_ap, 3) == 0.736
    assert round(probe128["test"]["ap"], 3) == 0.662
    assert round(probe128["test"]["recall"], 3) == 0.863
    assert round(probe128["test"]["review"], 2) == 0.48  # the slides' "48% of tiles"
    return feats128, probe128


@app.cell
def _(average_precision_score, budget_photos, df, feats, feats128, fit_probe, part, y):
    # The probe with fewer labelled photos: 5% to 100% of the training photos (C still chosen on
    # the validation photos), on 256-px and on 128-px tiles. The curve uses validation AP; each
    # budget's test AP is that budget's one test.
    budgets = {"photos": [], "probe_val_ap": [], "probe_test_ap": [], "probe128_val_ap": [],
               "probe128_test_ap": []}  # fmt: skip
    for _frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        _use = budget_photos(_frac)
        budgets["photos"].append(int(df["photo"][_use].nunique()))
        for _key, _f in (("probe", feats), ("probe128", feats128)):
            _pr = fit_probe(_f, y, _use, part == "val")
            budgets[f"{_key}_val_ap"].append(_pr.val_ap)
            budgets[f"{_key}_test_ap"].append(
                float(average_precision_score(y[part == "test"], _pr.score(_f[part == "test"])))
            )
    print({k: [round(v, 3) for v in vals] for k, vals in budgets.items()})
    assert budgets["photos"] == [9, 16, 35, 69, 137]
    assert [round(v, 2) for v in budgets["probe_val_ap"]] == [0.55, 0.61, 0.67, 0.70, 0.74]
    assert [round(v, 2) for v in budgets["probe_test_ap"]] == [0.50, 0.56, 0.63, 0.63, 0.67]
    assert [round(v, 2) for v in budgets["probe128_val_ap"]] == [0.56, 0.61, 0.67, 0.69, 0.74]
    assert [round(v, 2) for v in budgets["probe128_test_ap"]] == [0.5, 0.54, 0.62, 0.62, 0.66]
    return (budgets,)


@app.cell
def _(Image, Parallel, RAW, SCRATCH_BATCH, SCRATCH_LR, SCRATCH_PX, SEED, WORKERS,
      average_precision_score, delayed, df, np, part, scores, threshold_for, time, torch,
      y):  # fmt: skip
    def pixels_rgb(path):
        im = (
            Image.open(RAW / path)
            .convert("RGB")
            .resize((SCRATCH_PX,) * 2, Image.Resampling.BILINEAR)
        )
        return np.asarray(im)

    def small_cnn():
        nn = torch.nn

        def stage(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))  # fmt: skip

        return nn.Sequential(stage(3, 16), stage(16, 32), stage(32, 64), stage(64, 128),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))  # fmt: skip

    scratch_imgs = {r: np.stack(Parallel(n_jobs=WORKERS, batch_size=256)(
        delayed(pixels_rgb)(q) for q in df["path"].to_numpy()[part == r]))
        for r in ("train", "val", "test")}  # fmt: skip

    def channel_stats(a, chunk=256):
        """Per-channel mean and std of uint8 images, in exact int64 sums: ``a.std()`` would
        allocate the whole array in float64 (13 GB for the training tiles)."""
        total, squares, count = np.zeros(3, np.int64), np.zeros(3, np.int64), 0
        for i in range(0, len(a), chunk):
            v = a[i : i + chunk].reshape(-1, 3).astype(np.int64)
            total, squares, count = total + v.sum(0), squares + (v * v).sum(0), count + len(v)
        m = total / count
        return m, np.sqrt(np.maximum(squares / count - m**2, 0.0))

    def train_scratch(use, epochs, eval_every=1, seed=SEED):
        """small_cnn from a random start on the tiles in `use`; validation AP every few passes;
        the test tiles scored once, at the end, at the validation threshold. One one-cycle
        schedule over exactly `epochs` passes; the flips are drawn per batch."""
        imgs = scratch_imgs["train"][use[part == "train"]]
        ytr = torch.from_numpy(y[use]).float()
        mean, std = (v / 255 for v in channel_stats(imgs))

        def tensor(a):  # (n, h, w, 3) uint8 -> (n, 3, h, w), standardized with the training tiles
            x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255
            m, sd = (torch.tensor(v).float().view(1, 3, 1, 1) for v in (mean, std))
            return ((x - m) / sd).contiguous(memory_format=torch.channels_last)

        torch.manual_seed(seed)
        net = small_cnn().to(memory_format=torch.channels_last)
        pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
        steps = epochs * (len(ytr) // SCRATCH_BATCH)
        opt = torch.optim.AdamW(net.parameters(), lr=SCRATCH_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SCRATCH_LR, total_steps=steps)

        def predict(a):
            net.eval()
            with torch.inference_mode():
                z = torch.cat(
                    [net(tensor(a[i : i + 512])).squeeze(1) for i in range(0, len(a), 512)]
                )
            return torch.sigmoid(z).numpy()

        t0, curve = time.perf_counter(), []
        for epoch in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(ytr))
            losses = []
            for i in range(0, len(ytr) - SCRATCH_BATCH + 1, SCRATCH_BATCH):
                b = perm[i : i + SCRATCH_BATCH]
                x = tensor(imgs[b.numpy()])
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
                pv = predict(scratch_imgs["val"])  # the last pass is always scored
                ap = float(average_precision_score(y[part == "val"], pv))
                curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ap": ap})
        pt = predict(scratch_imgs["test"])
        return {"curve": curve, "steps": steps, "minutes": (time.perf_counter() - t0) / 60,
                "weights": sum(q.numel() for q in net.parameters()),
                "test": scores(pt, y[part == "test"], threshold_for(pv, y[part == "val"]))}  # fmt: skip

    return (train_scratch,)


@app.cell
def _(
    SCRATCH_BATCH,
    SCRATCH_EPOCHS,
    SETTING,
    SMALL_LADDER,
    SMALL_SEEDS,
    SMALL_START,
    budget_photos,
    df,
    np,
    part,
    train_scratch,
):
    # The slow cell: all 137 training photos for 12 passes; then 10% of them (16 photos), first
    # with the first recipe's 2,000 steps (33 passes), then with the schedule's length chosen
    # on validation: a bracketing search over SMALL_LADDER, each length trained with three
    # seeds and scored by their median validation AP, widened while the best is at an edge
    scratch = train_scratch(part == "train", SCRATCH_EPOCHS)
    _use = budget_photos(0.1)
    _base = int(np.ceil(2000 / (_use.sum() // SCRATCH_BATCH)))
    assert _base == 33 and _base in SMALL_START
    small_runs = {}
    _grid = list(SMALL_START) if SETTING == "full" else [_base, 132]
    _seeds = SMALL_SEEDS if SETTING == "full" else SMALL_SEEDS[:1]
    while True:
        for _n in _grid:
            small_runs[_n] = [train_scratch(_use, _n, eval_every=_n, seed=_s) for _s in _seeds]
            print(_n, "passes:", [round(r["curve"][-1]["val_ap"], 3) for r in small_runs[_n]])
        _tried = sorted(small_runs)
        _median = {n: float(np.median([r["curve"][-1]["val_ap"] for r in small_runs[n]]))
                   for n in _tried}  # fmt: skip
        small_passes = max(_tried, key=lambda n: _median[n])  # a tie keeps the shorter one
        if SETTING == "reduced":  # the length the full search chose (132 passes)
            small_passes = 132
            break
        if _tried[0] < small_passes < _tried[-1]:
            break
        _k = SMALL_LADDER.index(small_passes) + (-1 if small_passes == _tried[0] else 1)
        assert 0 <= _k < len(SMALL_LADDER), f"best length {small_passes} at the ladder's end"
        _grid = [SMALL_LADDER[_k]]
    small = small_runs[_base][0]  # the first recipe, seed 0: the run the slides first showed
    small["photos"] = int(df["photo"][_use].nunique())
    small_seeds = {"val_ap": [r["curve"][-1]["val_ap"] for r in small_runs[_base]],
                   "test_ap": [r["test"]["ap"] for r in small_runs[_base]]}  # fmt: skip
    small_length = {
        "median_val_ap": _median,
        "val_ap": [r["curve"][-1]["val_ap"] for r in small_runs[small_passes]],
        "test_ap": [r["test"]["ap"] for r in small_runs[small_passes]],
    }
    print(
        "first recipe, seeds 0-2:", {k: [round(v, 3) for v in vs] for k, vs in small_seeds.items()}
    )
    print(f"chosen: {small_passes} passes;", {k: np.round(v, 3).tolist() if isinstance(v, list)
                                              else v for k, v in small_length.items()})  # fmt: skip
    for _name, _run in (("all photos", scratch), (f"{small['photos']} photos", small)):
        print(_name, f"{_run['minutes']:.0f} min,", [round(c["val_ap"], 3) for c in _run["curve"]],
              {k: round(v, 3) for k, v in _run["test"].items()})  # fmt: skip

    # the same code, seed, threads, and data as the slides' figure script: the same run
    def _near(a, b, tol=0.005):
        return abs(a - b) < tol

    assert scratch["weights"] == 97_809 and scratch["steps"] == 6_264
    assert small["photos"] == 16 and small["steps"] == 2_013 and int(_use.sum()) == 3_907
    assert _near(scratch["curve"][0]["val_ap"], 0.336) and _near(
        scratch["curve"][-1]["val_ap"], 0.740
    )
    assert _near(scratch["curve"][-2]["train_loss"], 0.650) and _near(
        scratch["curve"][-1]["train_loss"], 0.634
    )
    assert _near(scratch["curve"][-2]["val_ap"], 0.735)  # still rising at the last pass
    assert _near(scratch["test"]["ap"], 0.640) and _near(scratch["test"]["recall"], 0.878)
    assert _near(scratch["test"]["review"], 0.468)
    assert _near(small["curve"][-1]["val_ap"], 0.498) and _near(small["test"]["ap"], 0.418)
    # (strict=False: the reduced setting trains seed 0 only)
    assert all(
        _near(a, b) for a, b in zip(small_seeds["val_ap"], [0.498, 0.508, 0.544], strict=False)
    )
    assert all(
        _near(a, b) for a, b in zip(small_seeds["test_ap"], [0.418, 0.451, 0.465], strict=False)
    )
    # the longer-training check: validation AP of each seed at every length tried (the full
    # setting); the search keeps 132 passes (8,052 steps), inside the lengths tried
    _want_val = {
        "16": [0.422, 0.458, 0.459],
        "33": [0.498, 0.508, 0.544],
        "66": [0.605, 0.603, 0.622],
        "132": [0.622, 0.645, 0.636],
        "264": [0.609, 0.615, 0.628],
    }
    for _n, _runs in small_runs.items():
        _got = [r["curve"][-1]["val_ap"] for r in _runs]
        assert all(_near(a, b) for a, b in zip(_got, _want_val[str(_n)], strict=False)), (_n, _got)
    assert small_passes == 132 and small_runs[small_passes][0]["steps"] == 8052
    # tested once, at the chosen length: test AP per seed
    _want_test = [0.577, 0.601, 0.586]
    assert all(_near(a, b) for a, b in zip(small_length["test_ap"], _want_test, strict=False))
    return small, small_length, small_passes, small_seeds, scratch


@app.cell
def _(df, feats, fit_probe, np, photo_split, scores, threshold_for, y):
    # How much does the score depend on which photos test it? Five photo splits (split 0 is §1's)
    _rows = []
    for _k in range(5):
        _part = photo_split(df, np.random.default_rng((_k, 15)))
        _tr, _va, _te = (_part == r for r in ("train", "val", "test"))
        _pr = fit_probe(feats, y, _tr, _va)
        _p = _pr.score(feats)
        _rows.append(scores(_p[_te], y[_te], threshold_for(_p[_va], y[_va])) | {"C": _pr.C})
    spread = {key: [r[key] for r in _rows] for key in ("auc", "ap", "recall", "review", "C")}
    print({k: [round(v, 3) for v in vals] for k, vals in spread.items()})
    assert [round(v, 2) for v in (min(spread["ap"]), max(spread["ap"]))] == [0.67, 0.75]
    assert [round(100 * v) for v in (min(spread["recall"]), max(spread["recall"]))] == [86, 91]
    assert [round(100 * v) for v in (min(spread["review"]), max(spread["review"]))] == [43, 55]
    assert spread["ap"][0] == min(spread["ap"])  # the lecture's split is the hardest
    assert spread["C"] == [0.03, 0.03, 0.1, 0.1, 0.01]  # each split's C, inside the grid
    return (spread,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict first, in writing.**

    1. **Siblings.** Train also on half the tiles of every test photo, and score the other half.
       How much does average precision rise?
    2. **Another surface.** Train on decks only. What AP on pavements?
    3. **Fly higher.** The same tiles with half, then a quarter of the pixels. What share of the
       cracked tiles is still flagged?
    """)
    return


@app.cell
def _(KNeighborsClassifier, LogisticRegression, SEED, StandardScaler, average_precision_score,
      df, feats, np, part, pd, probe, roc_auc_score, y):  # fmt: skip
    # 1. Siblings: the same held-out tiles, scored by models trained without and with the other
    # half of each test photo's tiles
    _test = np.flatnonzero(part == "test")
    _half = np.random.default_rng((SEED, 15, 3)).random(len(_test)) < 0.5
    _sib, _held = _test[_half], _test[~_half]
    _train = np.flatnonzero(part == "train")
    _val = np.flatnonzero(part == "val")

    def _fit_score(rows):
        _sc = StandardScaler().fit(feats[rows])
        _z, _zh = _sc.transform(feats[rows]), _sc.transform(feats[_held])
        _lr = LogisticRegression(C=probe.C, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
        _pl = _lr.fit(_z, y[rows]).predict_proba(_zh)[:, 1]
        _pk = KNeighborsClassifier(n_neighbors=5).fit(_z, y[rows]).predict_proba(_zh)[:, 1]
        return {"probe_ap": float(average_precision_score(y[_held], _pl)),
                "probe_auc": float(roc_auc_score(y[_held], _pl)),
                "knn_ap": float(average_precision_score(y[_held], _pk))}  # fmt: skip

    leakage = {}
    for _name, _rows in (("photo", _train), ("siblings", np.concatenate([_train, _sib]))):
        leakage[_name] = _fit_score(_rows)
    # the control: as many tiles again, from the validation photos (which the held-out tiles do
    # not share), three draws. Their gain is what more data alone buys.
    _ctl = [_fit_score(np.concatenate([_train, np.random.default_rng((_d, 15, 99)).choice(
        _val, len(_sib), replace=False)])) for _d in range(3)]  # fmt: skip
    leakage["more_data"] = {k: [r[k] for r in _ctl] for k in ("probe_ap", "knn_ap")}
    # a held tile's eight neighbours' labels, as a score: cracks run across tiles
    _lab = {(r.photo, r.row, r.col): r.cracked for r in df[df["dup"] == 0].itertuples()}
    _nb = []
    for _r in df.iloc[_held].itertuples():
        _v = [_lab.get((_r.photo, _r.row + a, _r.col + b)) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        _nb.append(np.mean([x for k, x in enumerate(_v) if k != 4 and x is not None]))
    leakage["neighbours_auc"] = float(roc_auc_score(y[_held], _nb))
    # Photos in sequence: each test photo's nearest training photo, by mean feature (cosine)
    _z = StandardScaler().fit(feats[part == "train"]).transform(feats)
    _mean = pd.DataFrame(_z).groupby(df["photo"].to_numpy()).mean()
    _mean = _mean.div(np.linalg.norm(_mean.to_numpy(), axis=1), axis=0)
    _role = pd.Series(part).groupby(df["photo"].to_numpy()).first()
    _near = (_mean[_role == "test"] @ _mean[_role == "train"].T).idxmax(axis=1)
    leakage["next_number"] = int(sum(n[0] == ph[0] and abs(int(n[1:]) - int(ph[1:])) == 1
                                     for ph, n in _near.items()))  # fmt: skip
    print(leakage)
    assert (
        round(leakage["photo"]["probe_ap"], 3) == 0.655
        and round(leakage["siblings"]["probe_ap"], 3) == 0.676
    )
    assert (
        round(leakage["photo"]["knn_ap"], 3) == 0.464
        and round(leakage["siblings"]["knn_ap"], 3) == 0.510
    )
    assert round(leakage["neighbours_auc"], 2) == 0.86 and leakage["next_number"] == 14
    assert [round(v, 3) for v in leakage["more_data"]["probe_ap"]] == [0.659, 0.657, 0.660]
    assert [round(v, 3) for v in leakage["more_data"]["knn_ap"]] == [0.473, 0.473, 0.473]
    assert len(_held) == 5_580 and len(_sib) == 5_628
    return (leakage,)


@app.cell
def _(SURFACES, average_precision_score, df, feats, fit_probe, np, part, pd, threshold_for, y):
    # 2. Train on one surface, test on each; and each probe, at its own surface's threshold, on
    # the pavement test tiles: cracks flagged and clean tiles flagged
    _grid = np.zeros((3, 3))
    on_pavement = {}
    _pave = (part == "test") & (df["surface"] == "pavement").to_numpy()
    for _i, _s_tr in enumerate(SURFACES):
        _on = (df["surface"] == _s_tr).to_numpy()
        _pr = fit_probe(feats, y, (part == "train") & _on, (part == "val") & _on)
        _p = _pr.score(feats)
        for _j, _s_te in enumerate(SURFACES):
            _m = (part == "test") & (df["surface"] == _s_te).to_numpy()
            _grid[_i, _j] = average_precision_score(y[_m], _p[_m])
        _t = threshold_for(_p[(part == "val") & _on], y[(part == "val") & _on])
        on_pavement[_s_tr] = {"recall": float((_p[_pave & (y == 1)] >= _t).mean()),
                              "false_alarm": float((_p[_pave & (y == 0)] >= _t).mean())}  # fmt: skip
    cross = pd.DataFrame(_grid, index=[f"train {s}" for s in SURFACES], columns=SURFACES)
    print(on_pavement)
    assert cross.round(2).to_numpy().tolist() == [
        [0.65, 0.65, 0.37],
        [0.59, 0.75, 0.38],
        [0.52, 0.61, 0.62],
    ]
    assert [
        round(100 * on_pavement[k][m])
        for k in ("deck", "pavement")
        for m in ("recall", "false_alarm")
    ] == [43, 15, 88, 47]
    cross.round(2)
    return cross, on_pavement


@app.cell
def _(SEED, df, embed, feats, np, part, probe, r18_feat, roc_auc_score, t, y):
    # 3. Fly higher: every cracked test tile and as many uncracked, at 1/1, 1/2, 1/4 of the pixels
    _test = np.flatnonzero(part == "test")
    _pos = _test[y[_test] == 1]
    _neg = np.sort(np.random.default_rng((SEED, 15, 4)).choice(_test[y[_test] == 0], len(_pos),
                                                               replace=False))  # fmt: skip
    _rows = np.concatenate([_pos, _neg])
    _paths = df["path"].to_numpy()[_rows].tolist()
    _ps = {
        f: probe.score(feats[_rows] if f == 1 else embed(r18_feat, _paths, f)) for f in (1, 2, 4)
    }
    # the AUC at each scale: does every score shift (ranking kept), or is the ranking lost?
    resolution = {f: {"recall": float((_ps[f][: len(_pos)] >= t).mean()),
                      "false_alarm": float((_ps[f][len(_pos) :] >= t).mean()),
                      "auc": float(roc_auc_score(y[_rows], _ps[f]))}
                  for f in (1, 2, 4)}  # fmt: skip
    _lost = np.flatnonzero((_ps[1][: len(_pos)] >= t) & (_ps[4][: len(_pos)] < t))
    _k = int(_lost[np.argmax(_ps[1][_lost] - _ps[4][_lost])])
    resolution["example"] = {"path": _paths[_k], "p": [float(_ps[f][_k]) for f in (1, 2, 4)],
                             "lost": len(_lost)}  # fmt: skip
    print(resolution)
    assert [round(100 * resolution[f]["recall"]) for f in (1, 2, 4)] == [86, 84, 77]
    assert [round(100 * resolution[f]["false_alarm"]) for f in (1, 2, 4)] == [44, 39, 31]
    # the ranking barely moves until 64 px: AUC 0.844, 0.846, 0.824
    assert [round(resolution[f]["auc"], 3) for f in (1, 2, 4)] == [0.844, 0.846, 0.824]
    assert (
        resolution["example"]["path"] == "P/CP/089-22.jpg" and resolution["example"]["lost"] == 211
    )
    assert [round(v, 2) for v in resolution["example"]["p"]] == [1.0, 0.97, 0.01]
    return (resolution,)


@app.cell
def _(ATTEMPT, SEED, df, np, p, part, plt, read_tile, y):
    # The attempt, revealed. The eight tiles were picked by fixed ranks of the probe's p among
    # the test tiles of one surface and one label, two of each kind from different surfaces
    # (catches: deck, wall; passes: pavement, deck; false alarms: pavement, wall; misses:
    # pavement, deck), then shuffled with a seed.
    _want = [("deck", 1, 0.95), ("wall", 1, 0.95), ("pavement", 0, 0.05), ("deck", 0, 0.05),
             ("pavement", 0, 0.995), ("wall", 0, 0.995), ("pavement", 1, 0.05), ("deck", 1, 0.05)]  # fmt: skip
    _rows = []
    for _s, _lab, _q in _want:
        _m = np.flatnonzero((part == "test") & (y == _lab) & (df["surface"] == _s).to_numpy())
        _o = _m[np.argsort(p[_m], kind="stable")]
        _rows.append(int(_o[int(round(_q * (len(_o) - 1)))]))
    attempt_rows = list(np.random.default_rng((SEED, 15, 5)).permutation(_rows))
    assert df["path"].to_numpy()[attempt_rows].tolist() == ATTEMPT
    assert [round(float(p[r]), 2) for r in attempt_rows] == [
        0.0, 1.0, 0.01, 0.01, 0.68, 1.0, 0.04, 0.94
    ]  # fmt: skip
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for _ax, _r, _letter in zip(_axes.ravel(), attempt_rows, "ABCDEFGH", strict=True):
        _ax.imshow(read_tile(df["path"].iloc[_r]))
        _ax.set_title(f"{_letter}: {'cracked' if y[_r] else 'uncracked'}, p = {p[_r]:.2f}",
                      fontsize=10)  # fmt: skip
        _ax.set_axis_off()
    _fig
    return (attempt_rows,)


@app.cell
def _(attempt_rows, df, np, p, preprocess, probe, r18_feat, read_tile, torch):
    # Where the network looks: grey out each 48-px square (stride 16) of the four flagged attempt
    # tiles; the drop in log-odds of "cracked"
    def occlusion(path, size=48, stride=16):
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
            f = r18_feat(torch.stack(batch)).numpy()
        logit = probe.logit(f)  # exact log-odds, where a probability would round to 1
        return float(probe.score(f)[0]), (logit[0] - logit[1:]).reshape(n, n)

    occlusion_drops = {}
    for _r, _letter in zip(attempt_rows, "ABCDEFGH", strict=True):
        if p[_r] >= 0.5:
            _p0, _drop = occlusion(df["path"].iloc[_r])
            occlusion_drops[_letter] = {"p": _p0, "max_drop": float(_drop.max())}
    print({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in occlusion_drops.items()})
    assert {k: round(v["max_drop"], 1) for k, v in occlusion_drops.items()} == {
        "B": 4.2,
        "E": 0.9,
        "F": 5.8,
        "H": 5.1,
    }
    assert (
        round(occlusion_drops["E"]["p"], 2) == 0.68 and round(occlusion_drops["H"]["p"], 2) == 0.94
    )
    return (occlusion_drops,)


@app.cell
def _(Parallel, WORKERS, delayed, df, hashlib, p, pd, read_tile):
    # Labels are judgments: tiles whose pixels appear twice in the dataset
    def pixel_hash(path):
        return hashlib.sha1(read_tile(path).tobytes(), usedforsecurity=False).hexdigest()

    _digest = pd.Series(
        Parallel(n_jobs=WORKERS, batch_size=512)(delayed(pixel_hash)(q) for q in df["path"]),
        index=df.index,
    )
    _groups = [g.index for _, g in _digest.groupby(_digest) if len(g) > 1]
    copies = {
        "groups": len(_groups),
        "both_labels": int(sum(df["cracked"].loc[g].nunique() > 1 for g in _groups)),
        "same_photo": int(sum(df["photo"].loc[g].nunique() == 1 for g in _groups)),
        "both": {
            f"{df['photo'][g[0]]}-{df['tile'][g[0]]}": round(float(p[g[0]]), 2)
            for g in _groups
            if df["cracked"].loc[g].nunique() > 1
        },
    }
    print(copies)
    assert copies == {
        "groups": 13,
        "both_labels": 2,
        "same_photo": 13,
        "both": {"D7039-112": 0.14, "W7074-105": 0.34},
    }
    return (copies,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    The threshold was set on the validation photos to flag 90% of their cracked tiles. On the
    test photos: what share of cracks is caught, and what share of tiles must the inspector
    review? Then one test photo, tile by tile.
    """)
    return


@app.cell
def _(df, mosaic, p, part, plt, t):
    # The decision on the test deck photo with the most cracked tiles
    _decks = df[(part == "test") & (df["surface"] == "deck").to_numpy()]
    decide_photo = _decks.groupby("photo")["cracked"].sum().sort_values(kind="stable").index[-1]
    _img, _tiles = mosaic(df, decide_photo, 48)
    _tp = _tiles.assign(p=p[_tiles.index])
    decide = {"photo": decide_photo, "tiles": len(_tp), "flagged": int((_tp["p"] >= t).sum()),
              "cracked": int(_tp["cracked"].sum()),
              "caught": int(((_tp["p"] >= t) & (_tp["cracked"] == 1)).sum())}  # fmt: skip
    decide["missed"] = decide["cracked"] - decide["caught"]
    print(decide, f"review {decide['flagged'] / decide['tiles']:.0%},",
          f"caught {decide['caught'] / decide['cracked']:.0%}")  # fmt: skip
    assert decide == {"photo": "D7049", "tiles": 252, "flagged": 160, "cracked": 73, "caught": 56,
                      "missed": 17}  # fmt: skip
    _fig, _ax = plt.subplots(figsize=(9, 7))
    _ax.imshow(_img)
    _f = _tp[_tp["p"] >= t]
    _c = _tp[_tp["cracked"] == 1]
    _ax.scatter(_f["col"] * 48 + 24, _f["row"] * 48 + 24, marker="s", s=300, facecolors="none",
                edgecolors="#D55E00", label="flagged")  # fmt: skip
    _ax.scatter(_c["col"] * 48 + 24, _c["row"] * 48 + 24, s=12, color="#0072B2", label="cracked")
    _ax.legend(loc="upper right")
    _ax.set_axis_off()
    _fig
    return (decide,)


@app.cell
def _(mo):
    mo.md(r"""
    **When not to use a tile classifier.** When the crack's width matters (a tile says *cracked*,
    not *0.3 mm*; that needs a segmentation and a calibrated camera, Lecture 16); on a surface or
    camera it was not trained on (§5: decks do not teach pavements); when a missed crack is
    critical and the inspector would stop looking at unflagged tiles.

    **Impact.** Screening changes the inspector's work from looking at every square foot to
    reviewing flags. Who signs for a crack the network passed, and is the inspector still paid
    to look at the rest? Record which tiles were never reviewed, so a missed crack can be traced
    to the screen, not the inspector.

    **Vote.** The owner wants the screen on next season's deck inspections. (A) Deploy the
    ImageNet probe as is. (B) Deploy it, with the threshold re-set on the owner's own photos and
    a random sample of unflagged tiles reviewed. (C) Train a network from scratch first. (D) Do
    not deploy; inspect every tile. Vote alone, discuss, vote again.

    **Where CNNs live today.** U-Net segmentation (Lecture 16) and object detectors use ResNet
    backbones; vision transformers (Lecture 17) replaced the convolutions in many models, and
    foundation models supply frozen features, as ImageNet's ResNet-18 did here: Lecture 18
    compares CLIP's with ResNet-18's on these tiles and uses SAM.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "The vote, explained": mo.md(
                r"""
                **B.** The threshold must be re-set where the screen is used: the validation
                photos promised 90% of cracks, and other photos give a different share (§4's
                five splits). A random sample of unflagged tiles measures the misses in service,
                which is the only way to know the screen still works. A deploys a promise made
                on someone else's photos; C, trained long enough, matched the probe even with 16
                photos (§4), at half an hour a seed plus a search over lengths, and the owner's
                own photos would still be needed to set its threshold; D is the baseline,
                defensible if no one will audit the misses.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question.** A vendor reports 98% accuracy for a crack classifier, tested on a random
    20% of the tiles from its 300 training photos. Name two things you would ask before believing
    it, and the number you would want instead of accuracy.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Accuracy: with 15% cracked tiles, flagging nothing scores 85%; ask for recall and
                the share of tiles to review, at a threshold set before the test. A random 20%
                of tiles: the test photos were also training photos; ask for whole photos (or
                whole structures) held out, and for results per surface and camera. Then ask
                how the threshold was chosen.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Splitting tiles instead of photos.** Tiles of one photo share its light, camera, and
      surface. Here the leak was small for a frozen probe (§5), but a crack's neighbouring tiles
      predict it well: anything that can see them will look better than it is. Split by the unit
      of collection, and say which.
    - **Flipping the filter.** PyTorch's `conv2d` is a cross-correlation. A textbook
      convolution flips the filter; the two agree only for symmetric filters (§4).
    - **ImageNet normalization.** The pretrained weights expect ImageNet's mean and standard
      deviation, in RGB order, at about 224 pixels. Skip it and the features shift; the
      notebook checks the cache against a fresh pass.
    - **Accuracy on an imbalanced set.** 85% for flagging nothing. Report recall at a threshold
      fixed before the test, and the review share.
    - **Batch norm in eval mode.** Features are computed with `model.eval()`: in training mode
      batch norm uses each batch's statistics and the same tile gets different features.
    - **Reading per-epoch test scores.** Choose epochs and thresholds on validation photos;
      score the test photos once.
    - **Comparing before training is done.** At its first recipe's 2,000 steps the 16-photo
      network trailed the probe by 0.09-0.14 test AP; with its schedule's length chosen on
      validation it led by 0.02-0.04 (§4). Train each model until validation stops improving,
      then compare.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Forsyth §17.1.1–17.1.2:** pattern detection by convolution, and layers of convolutions
      on layers; he flips the kernel and calls the result convolution, as PyTorch does, and
      writes the layer as a map from a block of data to a block of data.
    - **Forsyth §18.1.6:** batch normalization, and why it lets deep stacks train at all.
    - **Forsyth §18.1.9:** residual networks; compare his argument with the gradient sizes in §3.
    - **Forsyth §18.1.3:** augmentation and ensembles; today's flips are the simplest augmentation.
    - **Exercise.** Fine-tune ResNet-18's last stage (`layer4`) on the training photos (a GPU
      helps) and compare with the frozen probe on the same test photos. Then repeat §5's sibling
      test: does the fine-tuned network gain more from siblings than the probe did?
    - **Exercise.** Re-set the threshold on each surface's validation photos separately. How do
      recall and the review share change on each surface's test photos?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A convolution applies one small filter everywhere: few weights, and shifts in, shifts out.
    - Stacks see farther; skip connections keep the gradient in range.
    - ImageNet's features transfer: with 16 labelled photos, a frozen network and a logistic
      regression reach test AP 0.56 in seconds of fitting. A network trained from scratch on
      the same photos reached 0.42-0.46 after 2,000 steps, and 0.58-0.60 once its schedule
      was lengthened on validation (132 passes, half an hour a seed). Train each model to its
      best before comparing them.
    - Hold out whole photos, set the threshold before the test, and expect another surface to
      break it.

    **Next:** segmentation and detection, and measuring cracks in millimetres.
    """)
    return


if __name__ == "__main__":
    app.run()
