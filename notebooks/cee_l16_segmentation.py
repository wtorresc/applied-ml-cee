"""Applied Machine Learning for CEE — Lecture 16: Segmentation and Measuring.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l16_segmentation.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_3dcrack.sh`` (3DCrack, Zhang et al., 2025, CC BY 4.0, 238 MB) and
``bash scripts/fetch_spacenet8.sh`` (SpaceNet 8 Germany, Hänsch et al., 2022, CC BY-SA 4.0,
1.4 GB) once.

Set ``L16_SETTING=reduced`` (the default: a laptop CPU) or ``L16_SETTING=slides`` (the slides'
full run, on a GPU) to choose the setting (see the header cell).
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
    # Applied Machine Learning for CEE — Lecture 16: Segmentation and Measuring

    **Slides:** *Applied Machine Learning for CEE*, Lecture 16 (`lectures/cee_l16_segmentation`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §19.2.1–19.2.2 (encoders, decoders, and making data blocks bigger) and §18.2 (object detection: non-maximum suppression, R-CNN, YOLO, evaluating detectors).

    **Data:**
    - 3DCrack (Zhang, Wang, Hsieh, Yang, Yezzi & Tsai, 2025; doi:10.21227/2etq-xh36; CC BY 4.0):
      1,634 range images of Georgia pavements from a 3D line-laser survey system, 512 × 624
      pixels at about 4 mm per pixel, with crack masks and centrelines that carry two widths
      per point. Fetch with `bash scripts/fetch_3dcrack.sh`.
    - SpaceNet 8, Germany (Hänsch et al., 2022; CC BY-SA 4.0; the Maxar imagery is under a
      non-commercial licence, fine for teaching): satellite tiles before and after the July 2021
      flood with buildings and roads marked flooded or not. Fetch with
      `bash scripts/fetch_spacenet8.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    The numbers on the slides that come from the data and from the from-scratch code are
    recomputed here and checked with an `assert` in both settings; the trained networks'
    numbers are checked in the `slides` setting.

    **Two settings**, chosen by the environment variable `L16_SETTING` (whatever the device):

    - `reduced` (the default): for a laptop CPU. The U-Net and the network without skips train
      on one random 256 × 256 crop of every fourth training image per pass, for 6 passes,
      choosing the threshold on every eighth validation image; the 8 mm U-Net on every fourth
      coarse image; SpaceNet 8's network on one 256-pixel crop per tile per pass, for 6 passes.
      The Dice variant and the training on masks read with `> 0` are left out (the clean U-Net
      is still scored against them). Every test image is still scored. 43 minutes on four CPU
      threads of the slides' machine (November 21, 2026), with up to 14 GB of memory (SpaceNet
      8's tiles and probabilities): a laptop with 16 GB. The trained numbers are printed, not checked: minutes of training on crops leave the
      networks undertrained. On four threads of the slides' machine the reduced U-Net scored
      test F1 0.50, about the same as the "darker than 96" threshold (0.50), and the network
      without skips 0.21. The reduced run shows the code working end to end; the slides' result
      (0.82, and every comparison between methods) needs the `slides` setting.
    - `slides`: the slides' full run: six networks, 40 passes over all 1,139 training images
      (60 for SpaceNet 8). About 50 minutes on the slides' GPU (the slides' figure script, the
      same work, took 46 minutes alone on an RTX 3090 Ti); on four CPU threads about 60 hours,
      estimated from measured steps (0.8 s per training image, so about 11.5 hours for one
      40-pass U-Net). On the slides' GPU (an RTX 3090 Ti) the slides' digits are checked; on another device the trained numbers move a little, and their
      asserts allow a stated difference.

    **Your tasks** (from the slides): threshold the range image yourself, write a transposed
    convolution and match PyTorch, train a U-Net, turn its mask into millimetres, then break it
    four ways and decide who rates crack severity.
    """)
    return


@app.cell
def _():
    import csv
    import hashlib
    import json
    import math
    import os
    import random
    import re
    import time
    from collections import defaultdict
    from pathlib import Path

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
    from matplotlib.colors import ListedColormap
    from PIL import Image, ImageDraw
    from scipy.ndimage import (
        correlate,
        distance_transform_edt,
        find_objects,
        label,
        maximum_filter,
    )
    from scipy.ndimage import mean as ndimage_mean
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torchvision.ops import box_iou, nms

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"
    # The slides were made on a GPU. There, the same code reproduces them exactly; on a CPU the
    # trained models' numbers move a little, so their checks allow a small difference.
    TRAINED_TOL = 0.0015 if GPU else 0.03
    # SpaceNet 8's network fine-tunes ImageNet weights on 134 tiles: a looser check on a CPU
    SN8_TOL = 0.01 if GPU else 0.05

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    RAW = ROOT / "data" / "raw" / "3dcrack" / "3DCrack_v1"
    SN8 = ROOT / "data" / "raw" / "spacenet8" / "Germany_Training_Public"
    SPLITS = ("Train", "Val", "Test")
    H, W = 624, 512  # rows (along the lane) x columns (across)
    MM_PER_PX = 4.0  # the record: "about 4 mm per pixel"
    MASK_CUT = 127  # a mask pixel is crack when its value exceeds this
    LTPP_MM = (6.0, 19.0)  # FHWA-HRT-13-092, ACP 4 and ACP 6: low <= 6 < moderate <= 19 < high
    SEVERITIES = ("low", "moderate", "high")
    TOL = 2  # px, the tolerance of "F1 within 2 px"
    # The setting, independent of the device: "reduced" (default) or "slides" (the header cell)
    SETTING = os.environ.get("L16_SETTING", "reduced").strip().lower()
    if SETTING not in {"reduced", "slides"}:
        raise ValueError(f"L16_SETTING must be 'reduced' or 'slides', not {SETTING!r}")
    SLIDES = SETTING == "slides"
    BASE, BATCH, LR, WD = 16, 8, 2e-3, 1e-4
    EPOCHS = 40 if SLIDES else 6  # passes over the training images
    TRAIN_EVERY = 1 if SLIDES else 4  # every k-th training image
    VAL_EVERY = 1 if SLIDES else 8  # every k-th validation image picks the threshold
    CROP = None if SLIDES else 256  # train on one random crop of this size per image and pass
    # SpaceNet 8: crop size, crops per tile per pass, passes
    SN8_CROP, SN8_CROPS, SN8_EPOCHS = (512, 4, 60) if SLIDES else (256, 1, 6)
    THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
    DARK_LINE = np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]) / 8.0
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    CMAP_TFN = ListedColormap(["#00000000", "#009E73", "#D55E00", "#0072B2"])  # -, TP, FP, FN
    print(f"device: {DEVICE}; setting: {SETTING}")
    return (
        CMAP_TFN,
        CROP,
        DARK_LINE,
        DEVICE,
        EPOCHS,
        F,
        GPU,
        H,
        Image,
        ImageDraw,
        LTPP_MM,
        LR,
        MASK_CUT,
        MM_PER_PX,
        PALETTE,
        RAW,
        SEED,
        SEVERITIES,
        SLIDES,
        SN8,
        SN8_CROP,
        SN8_CROPS,
        SN8_EPOCHS,
        SPLITS,
        BASE,
        BATCH,
        THRESHOLDS,
        TOL,
        TRAINED_TOL,
        TRAIN_EVERY,
        SN8_TOL,
        VAL_EVERY,
        W,
        WD,
        box_iou,
        correlate,
        csv,
        defaultdict,
        distance_transform_edt,
        find_objects,
        hashlib,
        json,
        label,
        math,
        maximum_filter,
        ndimage_mean,
        nms,
        nn,
        np,
        plt,
        re,
        roc_auc_score,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–15

    On your own, no notes, two minutes:

    1. Lecture 6: logistic regression gives one row a probability $p$ of class 1. What is the
       row's loss if its label is 1?
    2. Lecture 15: a convolution with 16 filters of 3 × 3 on a one-channel image. How many
       weights, biases included?
    3. Lecture 15: ResNet-18's last stage sees 435 pixels but keeps a 7 × 7 map. Can it say
       *where* in the tile the crack runs?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. $-\log p$, the cross-entropy. Today it is summed over every pixel: a U-Net is
           logistic regression at each pixel, on features a network learns.
        2. $16 \times 9 + 16 = 160$.
        3. Only to within 32 pixels: one cell of the 7 × 7 map covers a 32 × 32 block of the
           224-pixel input. A classifier knows *what*; today's decoder recovers *where*.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A survey van scans a lane with 3D line lasers. The agency rates cracking with the LTPP
    *Distress Identification Manual* (FHWA-HRT-13-092, 2014): for longitudinal and transverse
    cracks in asphalt, the **severity is set by the crack's mean width**, low up to 6 mm,
    moderate up to 19 mm, high above, and the rater records **metres of cracking at each
    level**. Lecture 15's classifier said *whether* a tile holds a crack. Here we need *where*
    every crack runs (its length) and *how wide* it is, in millimetres.

    3DCrack's range images store the surface height: darker is lower. The released images are
    512 × 624 pixels at about 4 mm per pixel (2.0 × 2.5 m of lane), downsampled from the
    scanner's 1 mm.
    """)
    return


@app.cell
def _(H, Image, MASK_CUT, RAW, W, np):
    def load_split(name, cut=MASK_CUT):
        """Every image (uint8, darker = lower) and mask (bool, value > cut) of one split."""
        d = RAW / name
        stems = sorted(p.stem for p in (d / "img").glob("*.png"))
        x = np.stack([np.asarray(Image.open(d / "img" / f"{s}.png")) for s in stems])
        y = np.stack([np.asarray(Image.open(d / "anno" / f"{s}.png")) > cut for s in stems])
        assert x.shape == y.shape == (len(stems), H, W), (x.shape, y.shape)
        return {"name": name, "stems": stems, "x": x, "y": y}

    return (load_split,)


@app.cell
def _(SPLITS, load_split):
    splits = {_s: load_split(_s) for _s in SPLITS}
    train, val, test = splits["Train"], splits["Val"], splits["Test"]
    crack_share = float(train["y"].mean())
    _free = {_s: int((~v["y"].reshape(len(v["y"]), -1).any(1)).sum()) for _s, v in splits.items()}
    print({_s: len(v["stems"]) for _s, v in splits.items()}, "crack-free:", _free)
    print(f"crack pixels in training: {crack_share:.4%}")
    assert [len(splits[_s]["stems"]) for _s in SPLITS] == [1139, 245, 250]
    assert _free == {"Train": 271, "Val": 60, "Test": 53}
    assert round(crack_share, 5) == 0.00742
    return crack_share, splits, test, train, val


@app.cell
def _(mo):
    mo.md(r"""
    ### Look at the labels' values, not their documentation

    The record says masks hold 0 or 255. Count what they really hold, and find images published
    twice (identical pixels in two splits).
    """)
    return


@app.cell
def _(Image, RAW, defaultdict, hashlib, np, splits):
    grey_masks, faint, _digests = 0, [], defaultdict(list)
    for _s in splits.values():
        for _i, _stem in enumerate(_s["stems"]):
            _m = np.asarray(Image.open(RAW / _s["name"] / "anno" / f"{_stem}.png"))
            grey_masks += bool(((_m > 0) & (_m < 255)).any())
            if (_m > 0).mean() > 0.5:
                faint.append((_s["name"], _stem, float((_m > 0).mean()), float(_s["y"][_i].mean())))
            _key = hashlib.sha1(_s["x"][_i].tobytes(), usedforsecurity=False).hexdigest()
            _digests[_key].append((_s["name"], _i))
    twins = [v for v in _digests.values() if len(v) > 1]
    # their centrelines are noise too; keyed by (split, stem), because names repeat across
    # splits (sample_000450 is one image in Train and another in Test)
    skip = {(f[0], f[1]) for f in faint}
    print("masks with grey values:", grey_masks)
    print("faint masks (split, stem, share > 0, share > 127):", faint)
    print("published twice:", [(splits[a[0]]["stems"][a[1]], a[0], b[0]) for a, b in twins])
    assert grey_masks == 233
    assert [f[1] for f in faint] == [
        "sample_000797",
        "sample_000798",
        "sample_000799",
        "sample_000800",
    ]
    assert len(twins) == 2
    return faint, skip, twins


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: cracks are dark, right?

    Call a pixel crack when it is darker than a grey level chosen on the validation images.
    **Predict** the test F1 before running the next cell. Then try Lecture 15's "darker than its
    eight neighbours" filter the same way.
    """)
    return


@app.cell
def _(DARK_LINE, correlate, np, test, val):
    def counts(b, y):
        """True positives, false positives, misses over all pixels."""
        return int((b & y).sum()), int((b & ~y).sum()), int((~b & y).sum())

    def f1_of(b, y):
        tp, fp, fn = counts(b, y)
        return 2 * tp / max(2 * tp + fp + fn, 1)

    dark_level = max(range(20, 130, 2), key=lambda t: f1_of(val["x"] < t, val["y"]))
    _fv = np.stack([correlate(a.astype(float), DARK_LINE, mode="nearest") for a in val["x"]])
    _ft = np.stack([correlate(a.astype(float), DARK_LINE, mode="nearest") for a in test["x"]])
    _lvl = max(np.arange(2.0, 40.0), key=lambda t: f1_of(_fv > t, val["y"]))
    dark_f1 = f1_of(test["x"] < dark_level, test["y"])
    filter_f1 = f1_of(_ft > _lvl, test["y"])
    print(f"darker than {dark_level}: test F1 {dark_f1:.3f}; L15 filter > {_lvl}: {filter_f1:.3f}")
    assert dark_level == 96
    assert round(dark_f1, 3) == 0.501 and round(filter_f1, 3) == 0.315
    return counts, dark_f1, dark_level, f1_of


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: shrink to see, grow back to point

    Labelling every pixel with Lecture 15's classifier means one window per pixel: 319,488
    windows an image, each recomputing convolutions its neighbours already computed. A **fully
    convolutional** network computes each convolution once over the whole image. The **U-Net**
    (Ronneberger, Fischer & Brox, 2015) halves the map four times (to know *what*), doubles it
    back four times (to say *where*), and joins each decoder level to its encoder twin with a
    **skip connection** that carries the fine position the pooling threw away.
    """)
    return


@app.cell
def _(BASE, F, nn, torch):
    def conv_block(c_in, c_out):
        """Two 3 x 3 convolutions, each with batch normalization and a ReLU."""
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    class UNet(nn.Module):
        """Five encoder levels (16 ... 256 channels), four decoder levels, a 1 x 1 head."""

        def __init__(self, c_in=1, c_out=1, base=BASE, *, skips=True):
            super().__init__()
            ch = [base * 2**i for i in range(5)]
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

        def forward(self, x, keep=None):
            feats = []
            for i, block in enumerate(self.down):  # halve, deepen
                x = block(x if i == 0 else F.max_pool2d(x, 2))
                feats.append(x)
            if keep is not None:
                keep.extend(feats)
            x = feats.pop()
            for up, dec in zip(self.up, self.dec, strict=True):  # double, sharpen
                x = up(x)
                skip = feats.pop()
                assert x.shape == skip.shape, (x.shape, skip.shape)
                x = dec(torch.cat([skip, x], dim=1) if self.skips else x)
                if keep is not None:
                    keep.append(x)
            return self.head(x)  # a logit per pixel

    class Encoder(nn.Module):
        """The U-Net's encoder alone, with its pooling: Lecture 15's kind of classifier body."""

        def __init__(self, net):
            super().__init__()
            self.down = net.down

        def forward(self, x):
            for i, block in enumerate(self.down):
                x = block(x if i == 0 else F.max_pool2d(x, 2))
            return x

    return Encoder, UNet


@app.cell
def _(Encoder, H, UNet, W, math, nn, torch):
    def macs(net, shape):
        """Multiply-adds of every convolution in one forward pass."""
        total = 0

        def hook(m, inp, out):
            nonlocal total
            k = math.prod(m.kernel_size)
            if isinstance(m, nn.ConvTranspose2d):  # each input value stamps c_out x k x k
                total += inp[0].numel() * m.out_channels * k
            else:
                total += out.numel() * (m.in_channels // m.groups) * k

        hs = [
            m.register_forward_hook(hook)
            for m in net.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        ]
        with torch.no_grad():
            net.eval()(torch.zeros(shape))
        for h in hs:
            h.remove()
        return total

    _net = UNet()
    unet_weights = sum(p.numel() for p in _net.parameters())
    _keep = []
    with torch.no_grad():
        _out = _net.eval()(torch.zeros(1, 1, H, W), _keep)
    print("level shapes:", [tuple(t.shape[1:]) for t in _keep], "output", tuple(_out.shape))
    _whole, _window = macs(_net, (1, 1, H, W)), macs(Encoder(_net), (1, 1, 64, 64))
    print(
        f"U-Net: {_whole / 1e9:.1f} billion multiply-adds per image; one 64-px window per pixel: "
        f"{H * W * _window / 1e9:,.0f} billion ({H * W * _window / _whole:,.0f} times)"
    )
    assert _out.shape == (1, 1, H, W)
    assert [tuple(t.shape[1:]) for t in _keep][:5] == [
        (16, 624, 512),
        (32, 312, 256),
        (64, 156, 128),
        (128, 78, 64),
        (256, 39, 32),
    ]
    assert unet_weights == 1942289
    assert round(H * W * _window / _whole) == 1449
    return (unet_weights,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** whether a pixel is crack depends on the image within the network's
    receptive field around it, and the same evidence means the same thing anywhere. Every pixel
    gets its own logistic regression on learned features,
    $p_{ij} = \sigma(f_\theta(x)_{ij})$, and the loss is Lecture 6's cross-entropy averaged
    over all pixels:

    $$L = -\frac{1}{HW}\sum_{i,j} y_{ij}\log p_{ij} + (1-y_{ij})\log(1-p_{ij}).$$

    ### Growing a map: the transposed convolution

    Each input value multiplies the whole kernel, and the copy is *added* into the output at
    `stride` times the input's position. An $h \times w$ map becomes
    $((h-1)s + k) \times ((w-1)s + k)$.
    """)
    return


@app.cell
def _(np):
    def conv_transpose2d(x, w, stride):
        """x (C_in, h, w), w (C_in, C_out, k, k) -> (C_out, (h-1)s+k, (w-1)s+k): stamps."""
        _, h, wd = x.shape
        _, c_out, k, _ = w.shape
        out = np.zeros((c_out, (h - 1) * stride + k, (wd - 1) * stride + k))
        for i in range(h):
            for j in range(wd):
                # (C_in,) . (C_in, C_out, k, k) -> (C_out, k, k): input pixel (i, j)'s stamp
                out[:, i * stride : i * stride + k, j * stride : j * stride + k] += np.einsum(
                    "c,cokl->okl", x[:, i, j], w
                )
        return out

    return (conv_transpose2d,)


@app.cell
def _(conv_transpose2d, np):
    # The slides' build: a tent kernel [1 2 1] x [1 2 1] / 4 makes it bilinear upsampling.
    build_x = np.array([[2.0, 4.0], [6.0, 0.0]])
    build_k = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 4.0
    build_out = conv_transpose2d(build_x[None], build_k[None, None], stride=2)[0]
    print(build_out)
    # Inside the border: each input copied at its own position, averages between them.
    assert build_out[1, 1] == 2 and build_out[1, 3] == 4 and build_out[3, 1] == 6
    assert build_out[1, 2] == 3 and build_out[2, 2] == 3  # (2 + 4) / 2 and (2 + 4 + 6 + 0) / 4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    Against `torch.nn.functional.conv_transpose2d`, and against the **gradient of an ordinary
    strided `conv2d`** with the same weights: how many digits agree? If the second holds, a
    decoder's upsampling layer is the backward pass of Lecture 15's convolution (its $dx$),
    run forward.
    """)
    return


@app.cell
def _(F, SEED, conv_transpose2d, np, torch):
    _rng = np.random.default_rng((SEED, 16))
    _x = _rng.normal(size=(3, 5, 4))
    _w = _rng.normal(size=(3, 2, 4, 4))
    _ours = conv_transpose2d(_x, _w, stride=2)
    _ref = F.conv_transpose2d(torch.from_numpy(_x)[None], torch.from_numpy(_w), stride=2)[0]
    _z = torch.zeros((1, 2, *_ours.shape[1:]), dtype=torch.float64, requires_grad=True)
    _y = F.conv2d(_z, torch.from_numpy(_w), stride=2)  # (1, 3, 5, 4): the shape of x
    assert _y.shape[1:] == _x.shape
    (_y * torch.from_numpy(_x)[None]).sum().backward()
    vs_pytorch = float(np.abs(_ours - _ref.numpy()).max())
    vs_backward = float(np.abs(_ours - _z.grad[0].numpy()).max())
    print(
        f"shape {_ours.shape}; vs PyTorch {vs_pytorch:.1e}; vs the gradient of conv2d {vs_backward:.1e}"
    )
    assert _ours.shape == (2, 12, 10)
    assert np.allclose(_ours, _ref.numpy(), rtol=0, atol=1e-12)
    assert np.allclose(_ours, _z.grad[0].numpy(), rtol=0, atol=1e-12)
    return


@app.cell
def _(conv_transpose2d, np):
    # Convolution as a matrix: a 3 x 3 kernel on a 4 x 4 image gives a 2 x 2 map, C is 4 x 16.
    _k = np.arange(1, 10, dtype=float).reshape(3, 3)
    C = np.zeros((4, 16))
    for _r in range(2):
        for _c in range(2):
            _img = np.zeros((4, 4))
            _img[_r : _r + 3, _c : _c + 3] = _k
            C[2 * _r + _c] = _img.ravel()
    _z = np.array([1.0, -2.0, 0.5, 3.0])
    # C^T maps the 2 x 2 map back to 4 x 4: the transposed convolution, hence the name.
    assert np.allclose(
        C.T @ _z, conv_transpose2d(_z.reshape(1, 2, 2), _k[None, None], 1)[0].ravel()
    )
    C
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Imbalance, overlap, and thin structures

    Crack pixels are 0.74% of the training pixels, so a network that never
    says crack is right on 99.26% of them: accuracy is useless. We score the
    crack pixels with

    $$\text{F1} = \text{Dice} = \frac{2\,TP}{2\,TP + FP + FN}, \qquad
    \text{IoU} = \frac{TP}{TP + FP + FN},$$

    which rank models the same way ($\text{F1} = 2\,\text{IoU} / (1 + \text{IoU})$). The
    **soft Dice loss** replaces the counts by sums of probabilities,
    $1 - 2\sum p\,y / (\sum p + \sum y)$; it ignores true negatives, so 99% background cannot
    dominate it (Milletari et al., 2016). Whether it helps here is measured in §4.

    A crack mask is one to three pixels wide, so a one-pixel disagreement is a large share of
    it. The next cells measure how much, and how much two annotators disagree.
    """)
    return


@app.cell
def _(TOL, counts, distance_transform_edt, np):
    def image_counts(b, y, tol=TOL):
        """Per-image counts behind the strict and the tolerant scores (kept for the bootstrap)."""
        c = {
            k: np.zeros(len(b), np.int64) for k in ("tp", "fp", "fn", "hit_b", "hit_y", "nb", "ny")
        }
        for i, (bi, yi) in enumerate(zip(b, y, strict=True)):
            c["tp"][i], c["fp"][i], c["fn"][i] = counts(bi, yi)
            c["nb"][i], c["ny"][i] = int(bi.sum()), int(yi.sum())
            if bi.any() and yi.any():  # a pixel within tol px of the other mask is a hit
                c["hit_b"][i] = int((distance_transform_edt(~yi)[bi] <= tol).sum())
                c["hit_y"][i] = int((distance_transform_edt(~bi)[yi] <= tol).sum())
        return c

    def pooled(c, idx=None):
        """Strict precision, recall, F1, IoU, and F1 within TOL px, over images idx (all)."""
        s = {k: float(v.sum() if idx is None else v[idx].sum()) for k, v in c.items()}
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        tp_, tr_ = s["hit_b"] / max(s["nb"], 1.0), s["hit_y"] / max(s["ny"], 1.0)
        return {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "tol_precision": tp_,
            "tol_recall": tr_,
            "tol_f1": 2 * tp_ * tr_ / max(tp_ + tr_, 1e-12),
        }

    def tolerant(b, y, tol=TOL):
        """Precision and recall when a pixel within tol px of the other mask counts as a hit."""
        _s = pooled(image_counts(b, y, tol))
        return _s["tol_precision"], _s["tol_recall"]

    def scores(b, y):
        """Strict precision, recall, F1, IoU, and F1 within TOL px, pooled over the images."""
        return pooled(image_counts(b, y))

    return image_counts, pooled, scores, tolerant


@app.cell
def _(W, np, scores, test):
    _ys = test["y"][test["y"].reshape(len(test["y"]), -1).any(1)]
    shift_iou, shift_tol = [], []
    for _s in range(4):
        _b = np.zeros_like(_ys)
        _b[:, :, _s:] = _ys[:, :, : W - _s]  # every crack moved s px to the right
        _sc = scores(_b, _ys)
        shift_iou.append(_sc["iou"])
        shift_tol.append(_sc["tol_f1"])
    print("strict IoU by shift:", np.round(shift_iou, 3), "F1 within 2 px:", np.round(shift_tol, 3))
    assert [round(v, 3) for v in shift_iou] == [1.0, 0.605, 0.38, 0.261]
    return


@app.cell
def _(scores, splits, tolerant, twins):
    twin_scores = {}
    for (_s1, _i1), (_s2, _i2) in twins:
        _a, _b = splits[_s1]["y"][_i1], splits[_s2]["y"][_i2]
        _sc = scores(_b[None], _a[None])
        _p1, _r1 = tolerant(_b[None], _a[None], tol=1)
        twin_scores[splits[_s1]["stems"][_i1]] = {
            "iou": _sc["iou"],
            "f1": _sc["f1"],
            "tol1_f1": 2 * _p1 * _r1 / (_p1 + _r1),
            "tol2_f1": _sc["tol_f1"],
        }
    for _k, _v in twin_scores.items():
        print(_k, {m: round(x, 3) for m, x in _v.items()})
    assert round(twin_scores["sample_000130"]["iou"], 3) == 0.703
    assert round(twin_scores["sample_000131"]["iou"], 3) == 0.686
    return (twin_scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify: train the U-Net

    AdamW with a one-cycle learning rate peaking at $2 \times 10^{-3}$, batches of 8 images,
    random flips across and along the lane, 40 passes, bf16 on a GPU. Three networks, same
    data, seed, and passes: the U-Net with plain cross-entropy, the same without skips, and the
    same with the soft Dice loss added (the `reduced` setting trains the first two, on crops,
    for 6 passes). The threshold on $p$ is chosen on validation after the last pass, as in
    Lecture 6, and must fall inside the grid of thresholds tried.
    """)
    return


@app.cell
def _(DEVICE, THRESHOLDS, np, torch):
    def normalize(u8):
        """Range image bytes to roughly zero mean, unit spread (the images centre on 126)."""
        return (u8.float() / 255.0 - 0.5) / 0.25

    def dice_loss(logits, y):
        p = torch.sigmoid(logits)
        return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)

    def predict(net, x):
        """Crack probabilities, (N, h, w) float32, for uint8 images (N, h, w)."""
        net.eval()
        out = []
        with (
            torch.no_grad(),
            torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"),
        ):
            for i in range(0, len(x), 16):
                xb = normalize(torch.from_numpy(x[i : i + 16]).to(DEVICE)[:, None])
                out.append(torch.sigmoid(net(xb).float())[:, 0].cpu().numpy())
        return np.concatenate(out)

    def f1_by_threshold(p, y):
        """Strict F1 and IoU at every threshold in THRESHOLDS."""
        f1, iou = [], []
        for t in THRESHOLDS:
            b = p > float(t)
            tp, fp, fn = (b & y).sum().item(), (b & ~y).sum().item(), (~b & y).sum().item()
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            iou.append(tp / max(tp + fp + fn, 1))
        return np.array(f1), np.array(iou)

    return dice_loss, f1_by_threshold, normalize, predict


@app.cell
def _(
    BATCH,
    CROP,
    DEVICE,
    EPOCHS,
    F,
    LR,
    SEED,
    SLIDES,
    THRESHOLDS,
    UNet,
    WD,
    dice_loss,
    f1_by_threshold,
    math,
    normalize,
    predict,
    time,
    torch,
):
    def train_unet(xt, yt, xv, yv, *, skips=True, loss="bce", epochs=EPOCHS, crop=CROP, name=""):
        """Train a U-Net with random flips; return it, its record, and the validation threshold.

        With ``crop`` (the reduced setting), each batch is cut to one random square of that
        size: the network is fully convolutional, so it still predicts whole images.
        """
        torch.manual_seed(SEED)
        net = UNet(skips=skips).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        steps = epochs * math.ceil(len(xt) / BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=steps, pct_start=0.1
        )
        gen = torch.Generator().manual_seed(SEED)  # batches and flips, on the CPU on every device
        X, Y = torch.from_numpy(xt).to(DEVICE)[:, None], torch.from_numpy(yt).to(DEVICE)[:, None]
        yv_t = torch.from_numpy(yv).to(DEVICE)
        rec = {"loss": [], "val_f1": [], "val_iou": [], "threshold": 0.5}
        t0 = time.perf_counter()
        for epoch in range(epochs):
            net.train()
            perm, total = torch.randperm(len(X), generator=gen), 0.0
            for i in range(0, len(X), BATCH):
                idx = perm[i : i + BATCH].to(DEVICE)
                xb, yb = normalize(X[idx]), Y[idx].float()
                flip = torch.rand(2, generator=gen)  # mirror across and along the lane
                if flip[0] < 0.5:
                    xb, yb = xb.flip(3), yb.flip(3)
                if flip[1] < 0.5:
                    xb, yb = xb.flip(2), yb.flip(2)
                if crop is not None:  # one random square for the whole batch
                    _r = int(torch.randint(0, xb.shape[2] - crop + 1, (1,), generator=gen))
                    _c = int(torch.randint(0, xb.shape[3] - crop + 1, (1,), generator=gen))
                    xb = xb[:, :, _r : _r + crop, _c : _c + crop]
                    yb = yb[:, :, _r : _r + crop, _c : _c + crop]
                with torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    logits = net(xb).float()
                value = F.binary_cross_entropy_with_logits(logits, yb)
                if loss == "bce+dice":
                    value = value + dice_loss(logits, yb)
                opt.zero_grad(set_to_none=True)
                value.backward()
                opt.step()
                sched.step()
                total += value.item() * len(idx)
            f1, iou = f1_by_threshold(torch.from_numpy(predict(net, xv)).to(DEVICE), yv_t)
            rec["loss"].append(total / len(X))
            rec["val_f1"].append(float(f1.max()))
            rec["val_iou"].append(float(iou[f1.argmax()]))
            rec["threshold"] = float(THRESHOLDS[f1.argmax()])
            if epoch % 10 == 9:
                print(
                    f"{name} pass {epoch + 1}: loss {rec['loss'][-1]:.4f}, val F1 {f1.max():.3f}, "
                    f"{time.perf_counter() - t0:.0f} s"
                )
        rec["minutes"] = (time.perf_counter() - t0) / 60
        # A threshold at an end of the grid means the grid was too narrow. The
        # slides' runs must pass; a short reduced run is reported, not stopped.
        if not THRESHOLDS[0] < rec["threshold"] < THRESHOLDS[-1]:
            _msg = f"{name}: threshold {rec['threshold']} at the edge of the grid"
            if SLIDES:
                raise ValueError(_msg)
            print("note:", _msg)
        return net, rec

    return (train_unet,)


@app.cell
def _(SLIDES, TRAIN_EVERY, VAL_EVERY, train, train_unet, val):
    _xt, _yt = train["x"][::TRAIN_EVERY], train["y"][::TRAIN_EVERY]
    _xv, _yv = val["x"][::VAL_EVERY], val["y"][::VAL_EVERY]
    runs = {
        "U-Net": train_unet(_xt, _yt, _xv, _yv, name="U-Net"),
        "no skips": train_unet(_xt, _yt, _xv, _yv, skips=False, name="no skips"),
    }
    if SLIDES:
        runs["+ Dice loss"] = train_unet(_xt, _yt, _xv, _yv, loss="bce+dice", name="+ Dice")
    for _k, (_net, _rec) in runs.items():
        print(f"{_k}: {_rec['minutes']:.1f} min, threshold {_rec['threshold']}")
    return (runs,)


@app.cell
def _(SLIDES, TRAINED_TOL, dark_f1, predict, runs, scores, test):
    preds = {k: predict(net, test["x"]) > rec["threshold"] for k, (net, rec) in runs.items()}
    results = {
        k: {"val_f1": rec["val_f1"][-1], **scores(preds[k], test["y"])}
        for k, (net, rec) in runs.items()
    }
    for _k, _v in results.items():
        print(
            f"{_k:12s} val F1 {_v['val_f1']:.3f} | test F1 {_v['f1']:.3f}, IoU {_v['iou']:.3f}, "
            f"F1 within 2 px {_v['tol_f1']:.3f}"
        )
    pred = preds["U-Net"]
    if SLIDES:
        _expect = {"U-Net": 0.8216, "no skips": 0.7498, "+ Dice loss": 0.8179}
        for _k, _f1 in _expect.items():
            assert abs(results[_k]["f1"] - _f1) <= TRAINED_TOL, (_k, results[_k]["f1"], _f1)
        assert abs(results["U-Net"]["tol_f1"] - 0.9451) <= TRAINED_TOL
    else:  # minutes of training on crops: printed, not checked (see the header)
        print(
            f"reduced setting: the U-Net is undertrained after {len(runs['U-Net'][1]['loss'])} "
            f"passes on crops; expect a test F1 near the darkness threshold's {dark_f1:.2f} (on "
            "four threads of the slides' machine: U-Net 0.50, no skips 0.21), not the slides' "
            "0.82, which needs the full run (L16_SETTING=slides)"
        )
    return pred, preds, results


@app.cell
def _(SEED, SLIDES, image_counts, np, pooled, preds, test, twins):
    # How sure are these scores? Resample the 250 test images (with replacement) 2,000 times and
    # recompute each pooled score: a 95% interval. The same resamples for every model, so the
    # difference between two models is measured on the same images.
    _counts = {k: image_counts(b, test["y"]) for k, b in preds.items()}
    _idx = np.random.default_rng((SEED, 16, 9)).integers(0, len(test["stems"]), size=(2000, 250))

    def _ci(values):
        return [round(float(np.percentile(values, q)), 3) for q in (2.5, 97.5)]

    _boot = {k: [pooled(c, i) for i in _idx] for k, c in _counts.items()}
    intervals = {k: {m: _ci([b[m] for b in v]) for m in ("f1", "iou", "tol_f1")}
                 for k, v in _boot.items()}  # fmt: skip
    for _k, _v in intervals.items():
        print(f"{_k:12s} 95% intervals:", _v)
    if SLIDES:
        _d = [a["f1"] - b["f1"] for a, b in zip(_boot["U-Net"], _boot["+ Dice loss"], strict=True)]
        print(
            "F1, U-Net minus U-Net + Dice:",
            _ci(_d),
            "(an interval containing 0: no gain either way)",
        )
        assert _ci(_d)[0] < 0 < _ci(_d)[1]
        assert all(
            abs(a - b) <= 0.01
            for a, b in zip(intervals["U-Net"]["f1"], [0.804, 0.839], strict=True)
        )
    # sample_000131's pixels are also in training, annotated by someone else: leave it out
    _twin = [i for g in twins for sp, i in g if sp == "Test"]
    _keep = np.setdiff1d(np.arange(len(test["stems"])), _twin)
    _delta = pooled(_counts["U-Net"], _keep)["f1"] - pooled(_counts["U-Net"])["f1"]
    print(f"test F1 without {test['stems'][_twin[0]]}: change {_delta:+.4f}")
    assert len(_twin) == 1 and (abs(_delta) < 0.005 or not SLIDES)
    return (intervals,)


@app.cell
def _(SLIDES, TRAINED_TOL, pred, scores, splits, twins):
    # The test image published twice (sample_000131, also in Train with another annotator's
    # mask): the U-Net against each annotation of the same image, the like-for-like comparison
    # with the two annotations' agreement on it (§3).
    twin_unet = {}
    for _g in twins:
        for _sp, _i in _g:
            if _sp == "Test":
                for _osp, _j in _g:
                    twin_unet[_osp] = scores(pred[_i][None], splits[_osp]["y"][_j][None])
    print("U-Net on the twin test image, against the annotation in",
          {k: {m: round(v[m], 3) for m in ("f1", "iou", "tol_f1")} for k, v in twin_unet.items()})  # fmt: skip
    assert sorted(twin_unet) == ["Test", "Train"]
    if SLIDES:
        _want = {
            ("Test", "f1"): 0.8668,
            ("Test", "iou"): 0.7649,
            ("Test", "tol_f1"): 0.9639,
            ("Train", "f1"): 0.8404,
            ("Train", "iou"): 0.7247,
            ("Train", "tol_f1"): 0.9638,
        }  # from numbers.json
        for (_osp, _m), _v in _want.items():
            assert abs(twin_unet[_osp][_m] - _v) <= TRAINED_TOL, (_osp, _m, twin_unet[_osp][_m])
    return


@app.cell
def _(PALETTE, plt, runs):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10, 3.6))
    for _name, (_net, _rec) in runs.items():
        if _name == "U-Net":
            _ax0.plot(range(1, len(_rec["loss"]) + 1), _rec["loss"], color=PALETTE[0])
        _ax1.plot(range(1, len(_rec["val_f1"]) + 1), _rec["val_f1"], label=_name)
    _ax0.set_xlabel("pass")
    _ax0.set_ylabel("training loss")
    _ax1.set_xlabel("pass")
    _ax1.set_ylabel("validation F1 (best threshold)")
    _ax1.set_ylim(0.4, 0.9)
    _ax1.legend()
    _fig
    return


@app.cell
def _(CMAP_TFN, maximum_filter, np, plt):
    def overlay(ax, img, b, y):
        """Range image with found (green), false (orange), missed (blue) pixels."""
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        code = y.astype(int) if b is None else np.select([b & y, b & ~y, ~b & y], [1, 2, 3], 0)
        shown = maximum_filter(code, size=3) * maximum_filter(code > 0, size=3)
        ax.imshow(
            np.ma.masked_equal(shown, 0), cmap=CMAP_TFN, vmin=0, vmax=3, interpolation="nearest"
        )
        ax.axis("off")

    def show(panels, titles):
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        for ax, (img, b, y), t in zip(np.atleast_1d(axes), panels, titles, strict=True):
            overlay(ax, img, b, y)
            ax.set_title(t)
        return fig

    return (show,)


@app.cell
def _(np, pred, scores, show, test):
    _cracked = [i for i in range(len(pred)) if test["y"][i].any()]
    _per = np.array([scores(pred[i : i + 1], test["y"][i : i + 1])["tol_f1"] for i in _cracked])
    _order = np.argsort(_per, kind="stable")
    _typical = [_cracked[_order[int(f * (len(_order) - 1))]] for f in (0.25, 0.5, 0.75)]
    print("per-image F1 within 2 px, quartiles:", np.round(np.quantile(_per, [0.25, 0.5, 0.75]), 3))
    show(
        [(test["x"][i], pred[i], test["y"][i]) for i in _typical],
        [test["stems"][i] for i in _typical],
    )
    return


@app.cell
def _(DEVICE, F, plt, runs, test, torch, normalize):
    # The one picture: at each level, the channel whose map best matches the crack.
    _net = runs["U-Net"][0]
    _i = test["stems"].index("sample_000236")
    _keep = []
    with torch.no_grad():
        _net.eval()
        _logit = _net(
            normalize(torch.from_numpy(test["x"][_i : _i + 1]).to(DEVICE)[:, None]), _keep
        )
    _yt = torch.from_numpy(test["y"][_i]).float()[None, None]
    _fig, _axes = plt.subplots(2, 5, figsize=(14, 6))
    for _k, (_ax, _f) in enumerate(zip(_axes.flat, [*_keep, torch.sigmoid(_logit)], strict=True)):
        _f = _f[0].float().cpu()
        _yl = F.adaptive_avg_pool2d(_yt, _f.shape[-2:])[0, 0].flatten()
        _z = _f.flatten(1)
        _z = (_z - _z.mean(1, keepdim=True)) / (_z.std(1, keepdim=True) + 1e-8)
        _corr = (_z @ (_yl - _yl.mean())) / (_yl.std() + 1e-8) / _z.shape[1]
        _c = int(_corr.abs().argmax())
        _ax.imshow(_f[_c].numpy() * torch.sign(_corr[_c]).item(), cmap="magma")
        _ax.set_title(
            ("enc " if _k < 5 else "dec " if _k < 9 else "") + "×".join(map(str, _f.shape))
        )
        _ax.axis("off")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Measuring: from a mask to millimetres

    The Euclidean distance transform gives each crack pixel its distance $d$ to the nearest
    non-crack pixel. The dataset's convention is

    $$w = (2d - 1) \times 4\ \text{mm} \qquad (d = 1 \text{ on a one-pixel line: } w = 4 \text{ mm}).$$

    On a straight band $w$ pixels wide the centre has $d = \lceil w/2 \rceil$, so the rule is
    exact for odd widths and one pixel short for even ones: one- and two-pixel lines both read
    4 mm. Each test centreline file lists points along every crack branch with two widths: the
    dataset's `width_mm` (from the mask) and `width_mm_rangeimg` (from the range image). Check
    the ruler against the first. A predicted mask has its own centreline, so to compare it with
    the true mask both are read with one **ridge ruler**: thin the mask to its skeleton (Zhang &
    Suen, 1984) and read $d$ at the skeleton pixel nearest each true centreline point.
    """)
    return


@app.cell
def _(H, MM_PER_PX, RAW, W, distance_transform_edt, json, label, math, np):
    def load_branches(name):
        """Every crack branch of a split: points [row, col], width_mm, width_mm_rangeimg."""
        out = []
        for g in sorted((RAW / name / "CVM").glob("*.geojson")):
            for f in json.loads(g.read_text())["features"]:
                rc = np.asarray(f["geometry"]["coordinates"], dtype=float)
                a = np.asarray(f["properties"]["width_mm"], float)
                b = np.asarray(f["properties"]["width_mm_rangeimg"], float)
                assert rc.shape == (len(a), 2) and len(a) == len(b)
                rc = rc.astype(int)
                assert (rc[:, 0] < H).all() and (rc[:, 1] < W).all() and (rc >= 0).all()
                out.append(
                    {
                        "split": name,
                        "stem": g.stem,
                        "rc": rc,
                        "mask_mm": a,
                        "range_mm": b,
                        "length_m": float(
                            np.linalg.norm(np.diff(rc, axis=0), axis=1).sum() * MM_PER_PX / 1000
                        ),
                    }
                )
        return out

    def width_mm(d):
        """(2 d - 1) x 4 mm, 0 where d is 0."""
        return np.where(d > 0, (2 * d - 1) * MM_PER_PX, 0.0)

    def ruler(mask, rc):
        """The mask ruler: width at the points from the mask's distance transform."""
        return width_mm(distance_transform_edt(mask)[rc[:, 0], rc[:, 1]])

    def thin(mask):
        """A one-pixel skeleton (Zhang & Suen, 1984); a component it erases keeps its deepest pixel."""
        img = np.pad(mask.astype(np.uint8), 1)
        changed = True
        while changed:
            changed = False
            for step in (0, 1):
                # the eight neighbours P2 ... P9, clockwise from the one above
                nb = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1],
                      img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]]  # fmt: skip
                b = sum(v.astype(np.int16) for v in nb)
                a = sum(((nb[k] == 0) & (nb[(k + 1) % 8] == 1)).astype(np.int16) for k in range(8))
                p2, p4, p6, p8 = nb[0], nb[2], nb[4], nb[6]
                if step == 0:
                    cond = ((p2 & p4 & p6) == 0) & ((p4 & p6 & p8) == 0)
                else:
                    cond = ((p2 & p4 & p8) == 0) & ((p2 & p6 & p8) == 0)
                rm = (img[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
                if rm.any():
                    img[1:-1, 1:-1][rm] = 0
                    changed = True
        skel = img[1:-1, 1:-1].astype(bool)
        comps, n = label(mask, structure=np.ones((3, 3), bool))
        lost = np.setdiff1d(np.arange(1, n + 1), np.unique(comps[skel]))
        if lost.size:
            from scipy.ndimage import maximum_position as _maximum_position

            for r, c in _maximum_position(distance_transform_edt(mask), comps, lost):
                skel[int(r), int(c)] = True
        return skel

    def ridge_ruler(mask):
        """The ridge ruler for one mask: widths(rc) reads d at the nearest skeleton pixel.

        A point counts as found when the mask has a pixel within 1 px (3 x 3)."""
        skel = thin(mask)
        d = distance_transform_edt(mask)
        if skel.any():
            _, near = distance_transform_edt(~skel, return_indices=True)
        else:
            near = np.zeros((2, *mask.shape), int)
        padded = np.pad(mask, 1)

        def widths(rc):
            r, c = rc[:, 0], rc[:, 1]
            found = np.zeros(len(rc), bool)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    found |= padded[r + 1 + dr, c + 1 + dc]
            return np.where(found, width_mm(d[near[0][r, c], near[1][r, c]]), 0.0)

        return widths

    def skeleton_length_m(mask):
        """Path length of the skeleton: 4-neighbour steps 1 px; a diagonal step sqrt(2) px where
        no 4-neighbour path joins the two pixels already."""
        sk = thin(mask)
        orth = int((sk[:, :-1] & sk[:, 1:]).sum() + (sk[:-1, :] & sk[1:, :]).sum())
        d1 = sk[:-1, :-1] & sk[1:, 1:] & ~sk[:-1, 1:] & ~sk[1:, :-1]
        d2 = sk[:-1, 1:] & sk[1:, :-1] & ~sk[:-1, :-1] & ~sk[1:, 1:]
        return (orth + math.sqrt(2) * int(d1.sum() + d2.sum())) * MM_PER_PX / 1000

    branches = load_branches("Test")
    return branches, ridge_ruler, ruler, skeleton_length_m


@app.cell
def _(SLIDES, TRAINED_TOL, branches, math, np, pred, ridge_ruler, ruler, skip, test):
    _index = {s: i for i, s in enumerate(test["stems"])}
    _n = dict.fromkeys(("total", "exact", "control", "found", "same", "agree", "wider"), 0)
    measured = []  # per branch: length; mean width by mask, range, true-ridge, U-Net-ridge rulers
    _done = None
    for _br in branches:  # in file order: an image's branches come together
        _inner = _br["mask_mm"] > 0  # 0 marks an end point
        if (_br["split"], _br["stem"]) in skip or not _inner.any():
            continue
        _i = _index[_br["stem"]]
        if _i != _done:  # one skeleton per image and mask
            _true_w, _pred_w, _done = ridge_ruler(test["y"][_i]), ridge_ruler(pred[_i]), _i
        _rc, _want = _br["rc"][_inner], _br["mask_mm"][_inner]
        _same, _pw = _true_w(_rc), _pred_w(_rc)
        _hit = _pw > 0
        _n["total"] += len(_rc)
        _n["exact"] += int(np.isclose(ruler(test["y"][_i], _rc), _want, atol=0.05).sum())
        _n["control"] += int(np.isclose(_same, _want, atol=0.05).sum())
        _n["found"] += int(_hit.sum())
        _n["same"] += int(np.isclose(_pw[_hit], _want[_hit], atol=0.05).sum())
        _n["agree"] += int(np.isclose(_pw[_hit], _same[_hit], atol=0.05).sum())
        _n["wider"] += int((_pw[_hit] > _same[_hit] + 0.05).sum())
        measured.append(
            (
                _br["length_m"],
                _want.mean(),
                _br["range_mm"][_inner].mean(),
                _same[_same > 0].mean() if (_same > 0).any() else math.nan,
                _pw[_hit].mean() if _hit.mean() >= 0.5 else math.nan,
            )
        )
    measured = np.array(measured)
    ruler_check = {k: v / (_n["total"] if k in ("exact", "control", "found") else max(_n["found"], 1))
                   for k, v in _n.items() if k != "total"}  # fmt: skip
    print(
        f"{_n['total']:,} points; mask ruler = width_mm at {ruler_check['exact']:.2%}; ridge ruler on"
        f" the true mask = width_mm at {ruler_check['control']:.1%}; U-Net finds"
        f" {ruler_check['found']:.1%}; at those, its width = the true mask's (same ruler) at"
        f" {ruler_check['agree']:.1%}, wider at {ruler_check['wider']:.1%}"
    )
    assert _n["total"] == 213065 and abs(ruler_check["exact"] - 0.9990707061225448) < 1e-12
    assert round(ruler_check["control"], 4) == 0.8896  # true masks only
    if SLIDES:
        assert abs(ruler_check["agree"] - 0.6607) <= 10 * TRAINED_TOL
    return measured, ruler_check


@app.cell
def _(GPU, LTPP_MM, SEVERITIES, SLIDES, measured, np):
    def band(width_mm):
        """LTPP severity: 0 low (<= 6 mm), 1 moderate (<= 19 mm), 2 high."""
        return np.digitize(width_mm, LTPP_MM, right=True)

    _L = measured[:, 0]
    severity = {}  # metres at each severity, all cracking treated as linear
    for _col, _name in ((1, "mask"), (2, "range"), (3, "same ruler"), (4, "U-Net")):
        _w = measured[:, _col]
        _ok = ~np.isnan(_w)
        severity[_name] = [float(_L[_ok & (band(np.nan_to_num(_w)) == k)].sum()) for k in range(3)]
        severity[_name].append(float(_L[~_ok].sum()))  # not measured (U-Net missed the branch)
    # Like for like: the true mask by the same ridge ruler on the branches the U-Net found, and
    # on those it missed (the U-Net's "missed" length is not a width effect)
    _found = ~np.isnan(measured[:, 4])
    for _name, _sel in (("same ruler, found", _found), ("same ruler, missed", ~_found)):
        _w, _Ls = measured[_sel, 3], _L[_sel]
        _ok = ~np.isnan(_w)
        severity[_name] = [float(_Ls[_ok & (band(np.nan_to_num(_w)) == k)].sum()) for k in range(3)]
        severity[_name].append(float(_Ls[~_ok].sum()))
    for _k, _v in severity.items():
        print(
            f"{_k:6s}", {s: round(m, 1) for s, m in zip((*SEVERITIES, "missed"), _v, strict=True)}
        )
    assert [round(v) for v in severity["mask"][:3]] == [427, 542, 45]
    assert [round(v) for v in severity["range"][:3]] == [63, 819, 131]
    assert [round(v) for v in severity["same ruler"][:3]] == [411, 556, 47]
    _parts = np.add(severity["same ruler, found"], severity["same ruler, missed"])
    assert np.allclose(_parts, severity["same ruler"], rtol=0, atol=1e-9)
    if SLIDES:  # the U-Net's metres; a CPU-trained U-Net may move them by several percent
        _rel = 0.02 if GPU else 0.15
        _want = {"U-Net": [360.5, 539.8, 43.0]}
        _want |= {
            "same ruler, found": [352.5, 546.5, 44.3],
            "same ruler, missed": [58.3, 9.4, 2.9],
        }  # from numbers.json
        for _k, _vals in _want.items():
            assert all(
                abs(a - b) <= _rel * max(b, 50)
                for a, b in zip(severity[_k][:3], _vals, strict=True)
            ), (_k, severity[_k])
    return (severity,)


@app.cell
def _(SLIDES, TRAINED_TOL, branches, defaultdict, np, pred, skeleton_length_m, skip, test):
    # Lengths: the U-Net mask's skeleton against the reference centrelines, per test image. The
    # true mask's skeleton is the control: the estimator's own bias.
    _ref = defaultdict(float)
    for _br in branches:
        _ref[_br["stem"]] += _br["length_m"]
    _rows = np.array([(_ref.get(s, 0.0), skeleton_length_m(test["y"][i]), skeleton_length_m(pred[i]))
                      for i, s in enumerate(test["stems"]) if ("Test", s) not in skip])  # fmt: skip
    _big = _rows[:, 0] >= 1.0  # images with at least 1 m of reference cracking
    length_err = {}
    for _col, _name in ((1, "true mask"), (2, "U-Net")):
        _e = _rows[_big, _col] / _rows[_big, 0] - 1
        length_err[_name] = [float(np.percentile(_e, q)) for q in (25, 50, 75)]
        print(f"{_name:9s} skeleton length vs reference, per image: quartiles",
              np.round(length_err[_name], 3))  # fmt: skip
    print("total metres: reference, true-mask skeleton, U-Net skeleton:",
          np.round(_rows.sum(0), 1), f"({_big.sum()} images with >= 1 m)")  # fmt: skip
    assert [round(v, 4) for v in length_err["true mask"]] == [0.0316, 0.0489, 0.0689]
    if SLIDES:
        assert abs(length_err["U-Net"][1] - -0.0448) <= 10 * TRAINED_TOL
    return (length_err,)


@app.cell
def _(mo):
    mo.md(r"""
    On a 4 mm grid the mask ruler's widths come in steps: 4.0 mm (a one- *or* two-pixel line,
    by the $(2d - 1)$ rule), 12.0 mm (three or four pixels), and 7.3 and 13.9 mm where the
    nearest background is diagonal ($d = \sqrt{2}, \sqrt{5}$). LTPP's 6 mm limit falls between
    4.0 and 7.3: *low* means "the annotator drew a line one or two pixels wide". The range-image
    ruler is continuous and much wider. Read with the same ridge ruler, the U-Net's mask puts
    less length at low than the true mask, but the two rows cover different branches: the
    U-Net's leaves out the branches it mostly missed. On the branches it found, compare it with
    the "same ruler, found" row; the missed length is a finding problem, not a width one. The
    gap between either and the range-image ruler is the rulers' own disagreement. LTPP's width rule is for longitudinal and transverse cracks;
    3DCrack does not label crack types, so every branch, alligator cracking included (which
    LTPP rates by area), is treated as linear here.

    **Millimetres per pixel** = distance × pixel pitch / focal length. A survey system fixes
    the distance and states the result; a phone or drone photo does not, and a mask then gives
    widths in pixels only. Hüthwohl et al. (2019): seeing a 0.3 mm crack at 3 pixels needs
    0.1 mm per pixel, 40 times finer than this release.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    Write a prediction for each before running it:

    1. Read the masks with `> 0` instead of `> 127`. Train, then score. What changes?
    2. Survey at 8 mm per pixel instead of 4. Which cracks go missing, and what happens to
       LTPP's "low"?
    3. Same U-Net, satellite pixels of 0.4 m, before and after a flood. What goes wrong when
       you *count* buildings?
    4. The satellite tiles' pixels are in degrees. Road length = pixels × 0.42 m?
    """)
    return


@app.cell
def _(SLIDES, TRAINED_TOL, load_split, pred, predict, scores, test, train_unet, val):
    # 1. Masks read with > 0: train on them (slides setting), and score the clean U-Net
    # against them (both settings).
    if SLIDES:
        _tr0 = load_split("Train", cut=0)
        _net0, _rec0 = train_unet(_tr0["x"], _tr0["y"], val["x"], val["y"], name="mask > 0")
        _faint_trained = scores(predict(_net0, test["x"]) > _rec0["threshold"], test["y"])
        print(
            f"trained on > 0: val F1 {_rec0['val_f1'][-1]:.3f}, test F1 {_faint_trained['f1']:.3f}"
        )
        assert abs(_faint_trained["f1"] - 0.7676) <= TRAINED_TOL
        del _tr0, _net0
    _te0 = load_split("Test", cut=0)
    faint_scored = scores(pred, _te0["y"])
    print(
        f"clean U-Net scored against > 0 masks: test F1 {faint_scored['f1']:.3f}; "
        f"test crack share {test['y'].mean():.2%} -> {_te0['y'].mean():.2%}"
    )
    assert round(float(_te0["y"].mean()), 5) == 0.01206  # the ruler moved, not the model
    if SLIDES:
        assert abs(faint_scored["f1"] - 0.6564) <= TRAINED_TOL
    del _te0
    return


@app.cell
def _(H, W, np):
    def coarsen(s):
        """The survey at 8 mm per pixel: 2 x 2 means; a block is crack if any pixel is.
        Rows padded 312 -> 320 so the U-Net can halve four times."""
        n = len(s["x"])
        x = s["x"].reshape(n, H // 2, 2, W // 2, 2).astype(np.float32).mean(axis=(2, 4))
        y = s["y"].reshape(n, H // 2, 2, W // 2, 2).any(axis=(2, 4))
        x = np.pad(np.round(x).astype(np.uint8), ((0, 0), (0, 8), (0, 0)), mode="edge")
        y = np.pad(y, ((0, 0), (0, 8), (0, 0)))
        assert x.shape == y.shape == (n, 320, W // 2)
        return x, y

    return (coarsen,)


@app.cell
def _(
    H,
    MM_PER_PX,
    SLIDES,
    TRAINED_TOL,
    TRAIN_EVERY,
    VAL_EVERY,
    branches,
    coarsen,
    maximum_filter,
    np,
    pred,
    predict,
    skip,
    splits,
    test,
    train_unet,
):
    # 2. The same survey at 8 mm: a second U-Net trained and tested on the coarser grid (whole
    # coarse images in both settings; the reduced setting on every fourth image).
    (_x8, _y8), (_xv8, _yv8), (_xt8, _yt8) = (coarsen(splits[s]) for s in ("Train", "Val", "Test"))
    _net8, _rec8 = train_unet(
        _x8[::TRAIN_EVERY], _y8[::TRAIN_EVERY], _xv8[::VAL_EVERY], _yv8[::VAL_EVERY],
        crop=None, name="8 mm",
    )  # fmt: skip
    pred8 = predict(_net8, _xt8)[:, : H // 2] > _rec8["threshold"]

    def found_by_width(p, factor):
        """Share of true centreline points with a predicted crack within one (coarse) pixel."""
        index = {s: i for i, s in enumerate(test["stems"])}
        near = maximum_filter(p, size=(1, 3, 3))
        hit = {"hairline": [0, 0], "wider": [0, 0]}
        for br in branches:
            if (br["split"], br["stem"]) in skip:
                continue
            inner = br["mask_mm"] > 0
            rc = br["rc"][inner] // factor
            h = near[index[br["stem"]], rc[:, 0], rc[:, 1]]
            for key, sel in (
                ("hairline", np.isclose(br["mask_mm"][inner], MM_PER_PX)),
                ("wider", br["mask_mm"][inner] > MM_PER_PX + 0.05),
            ):
                hit[key][0] += int(h[sel].sum())
                hit[key][1] += int(sel.sum())
        return {k: v[0] / v[1] for k, v in hit.items()}

    found4, found8 = found_by_width(pred, 1), found_by_width(pred8, 2)
    print(
        "found at 4 mm:",
        {k: round(v, 3) for k, v in found4.items()},
        "| at 8 mm:",
        {k: round(v, 3) for k, v in found8.items()},
    )
    print(f"narrowest crack an 8 mm mask can draw: {2 * MM_PER_PX:.0f} mm (LTPP low ends at 6 mm)")
    if SLIDES:
        assert abs(found4["hairline"] - 0.8434) <= TRAINED_TOL
        assert abs(found8["hairline"] - 0.9120) <= 2 * TRAINED_TOL
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3–4. Same network, a flooded valley (SpaceNet 8)

    Satellite tiles of the Ahr valley before and after the July 2021 flood, with OpenStreetMap
    building footprints and roads, each marked flooded or not. Labels are given in the
    pre-event image's pixels (`Wkt_Pix`); the post-event images have coarser pixels and are
    resampled onto the pre-event grid with both images' georeferencing (GeoTIFF tags 33550,
    pixel size, and 33922, tie point). Roads are centrelines; we draw them 12 pixels wide (an
    assumption). Tiles are split in blocks along the valley: grid column < 35 train, 35–39
    validation, ≥ 40 test.
    """)
    return


@app.cell
def _(Image, ImageDraw, SN8, csv, defaultdict, np, re):
    SN8_SIZE, SN8_PAD, SN8_ROAD_PX, SN8_MIN_PX = 1300, 1312, 12, 25
    SN8_CLASSES = ("background", "building", "flooded building", "road", "flooded road")

    def wkt_rings(wkt):
        return [
            np.array([[float(v) for v in p.split()] for p in s.split(",")])
            for s in re.findall(r"\(([^()]+)\)", wkt)
        ]

    def geotransform(im):
        """West edge, north edge, pixel size x, y (degrees) from the GeoTIFF tags."""
        sx, sy, _ = im.tag_v2[33550]
        tie = im.tag_v2[33922]
        assert tie[:3] == (0.0, 0.0, 0.0)
        return tie[3], tie[4], sx, sy

    def rasterize(objs):
        size = (SN8_SIZE, SN8_SIZE)
        lab, inst = Image.new("L", size, 0), Image.new("I", size, 0)
        dl, di = ImageDraw.Draw(lab), ImageDraw.Draw(inst)
        buildings, roads = [], []
        for r in objs:  # roads first, buildings on top
            if r["Object"] != "Road" or "EMPTY" in r["Wkt_Pix"]:
                continue
            line = [tuple(p) for p in wkt_rings(r["Wkt_Pix"])[0]]
            flooded = r["Flooded"] == "True"
            dl.line(line, fill=4 if flooded else 3, width=SN8_ROAD_PX, joint="curve")
            roads.append({"flooded": flooded, "length_m": float(r["length_m"]), "line": line})
        for r in objs:
            if r["Object"] != "Building" or "EMPTY" in r["Wkt_Pix"]:
                continue
            poly = [tuple(p) for p in wkt_rings(r["Wkt_Pix"])[0]]
            flooded = r["Flooded"] == "True"
            dl.polygon(poly, fill=2 if flooded else 1)
            di.polygon(poly, fill=len(buildings) + 1)
            buildings.append({"flooded": flooded})
        inst = np.asarray(inst).astype(np.int32)
        areas = np.bincount(inst.ravel(), minlength=len(buildings) + 1)
        for k, b in enumerate(buildings, start=1):
            b["area_px"] = int(areas[k])
        return np.asarray(lab), inst, buildings, roads

    def sn8_tiles():
        with (SN8 / "Germany_Training_Public_reference.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        with (SN8 / "Germany_Training_Public_label_image_mapping.csv").open(newline="") as f:
            maps = list(csv.DictReader(f))
        by = defaultdict(list)
        for r in rows:
            by[r["ImageId"]].append(r)
        tiles = []
        for m in maps:
            pre = Image.open(SN8 / "PRE-event" / m["pre-event image"])
            post = Image.open(SN8 / "POST-event" / m["post-event image 1"])
            lon0, lat0, sx, sy = geotransform(pre)
            plon, plat, psx, psy = geotransform(post)  # before convert(), which drops the tags
            # Pillow's affine transform maps each output pixel's centre, in continuous pixel
            # coordinates, to the input in the same convention: no half-pixel terms
            coef = (sx / psx, 0, (lon0 - plon) / psx, 0, sy / psy, (plat - lat0) / psy)
            post = post.convert("RGB").transform(
                pre.size, Image.Transform.AFFINE, coef, Image.Resampling.BILINEAR
            )
            name = m["pre-event image"][:-4]
            lab, inst, buildings, roads = rasterize(by[name])
            tiles.append(
                {
                    "name": name.split("_", 1)[1],
                    "col": int(name.split("_")[2]),
                    "pre": np.asarray(pre.convert("RGB")),
                    "post": np.asarray(post),
                    "lab": lab,
                    "inst": inst,
                    "buildings": buildings,
                    "roads": roads,
                    "lat": lat0 - sy * SN8_SIZE / 2,
                    "deg": (sx, sy),
                }
            )
        return tiles

    tiles = sn8_tiles()
    part = {
        "train": [t for t in tiles if t["col"] < 35],
        "val": [t for t in tiles if 35 <= t["col"] < 40],
        "test": [t for t in tiles if t["col"] >= 40],
    }
    print(
        {
            k: (
                len(v),
                sum(len(t["buildings"]) for t in v),
                sum(b["flooded"] for t in v for b in t["buildings"]),
            )
            for k, v in part.items()
        }
    )
    assert [len(part[k]) for k in ("train", "val", "test")] == [134, 31, 37]
    assert sum(len(t["buildings"]) for t in part["test"]) == 897
    return SN8_CLASSES, SN8_MIN_PX, SN8_PAD, SN8_ROAD_PX, SN8_SIZE, part, tiles


@app.cell
def _(math, np, tiles):
    # 4. A pixel in degrees is not square: road lengths three ways.
    def metres_per_degree(lat):
        """Metres per degree of longitude and latitude at lat (WGS 84 series)."""
        phi = math.radians(lat)
        m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
        m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
        return m_lon, m_lat

    _rows = []
    for _t in tiles:
        _sx, _sy = _t["deg"]
        _mlon, _mlat = metres_per_degree(_t["lat"])
        for _road in _t["roads"]:
            _c = np.asarray(_road["line"])
            _dx, _dy = np.diff(_c[:, 0]), np.diff(_c[:, 1])
            _rows.append(
                (
                    _road["length_m"],
                    np.hypot(_dx, _dy).sum() * _sy * _mlat,
                    np.hypot(_dx * _sx * _mlon, _dy * _sy * _mlat).sum(),
                )
            )
    _km = np.array(_rows).sum(0) / 1000
    print(
        f"reference {_km[0]:.1f} km; pixels x one size {_km[1]:.1f} km; each axis in metres {_km[2]:.1f} km"
    )
    assert round(_km[0], 1) == 163.3 and round(_km[1], 1) == 207.7
    assert abs(_km[2] / _km[0] - 1) < 0.005
    return


@app.cell
def _(F, nn, torch):
    from torchvision.models import ResNet18_Weights, resnet18

    def _block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    class SiameseUNet(nn.Module):
        """ImageNet's ResNet-18 reads the before and the after image (shared weights); each
        level's two maps are concatenated and decoded back to full size, U-Net style."""

        def __init__(self, c_out=5):
            super().__init__()
            r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # 1/2, 64
            self.layers = nn.ModuleList(
                [nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4]
            )
            enc, dec = [64, 64, 128, 256, 512], [32, 64, 128, 256]
            self.up, self.dec = nn.ModuleList(), nn.ModuleList()
            c = 2 * enc[4]
            for lvl in (3, 2, 1, 0):
                self.up.append(nn.ConvTranspose2d(c, dec[lvl], 2, stride=2))
                self.dec.append(_block(dec[lvl] + 2 * enc[lvl], dec[lvl]))
                c = dec[lvl]
            self.last_up = nn.ConvTranspose2d(c, 32, 2, stride=2)
            self.last = _block(32, 32)
            self.head = nn.Conv2d(32, c_out, 1)

        def encode(self, x):
            feats = [self.stem(x)]
            for layer in self.layers:
                feats.append(layer(feats[-1]))
            return feats

        def forward(self, x):
            fa, fb = self.encode(x[:, :3]), self.encode(x[:, 3:])
            feats = [torch.cat([a, b], dim=1) for a, b in zip(fa, fb, strict=True)]
            y = feats.pop()
            for up, dec in zip(self.up, self.dec, strict=True):
                y = dec(torch.cat([feats.pop(), up(y)], dim=1))
            return self.head(self.last(self.last_up(y)))

    _ = F  # the decoder uses only modules
    return (SiameseUNet,)


@app.cell
def _(
    DEVICE,
    SEED,
    SN8_CLASSES,
    SN8_CROP,
    SN8_CROPS,
    SN8_EPOCHS,
    SN8_PAD,
    SN8_SIZE,
    SiameseUNet,
    WD,
    BATCH,
    F,
    math,
    np,
    time,
    torch,
):
    SN8_LR = 1e-3  # crop size, crops per tile and passes come from the setting (setup cell)
    _mean = torch.tensor((0.485, 0.456, 0.406) * 2, device=DEVICE).view(1, 6, 1, 1) * 255
    _std = torch.tensor((0.229, 0.224, 0.225) * 2, device=DEVICE).view(1, 6, 1, 1) * 255

    def sn8_stack(ts):
        """Before and after as six channels, padded to 1312 px; labels padded with background."""
        pad = SN8_PAD - SN8_SIZE
        x = np.stack([np.concatenate([t["pre"], t["post"]], axis=2) for t in ts])
        x = np.pad(x, ((0, 0), (0, pad), (0, pad), (0, 0)), mode="reflect")
        y = np.pad(np.stack([t["lab"] for t in ts]), ((0, 0), (0, pad), (0, pad)))
        return x, y

    def sn8_predict(net, ts):
        """Class probabilities (N, 5, 1300, 1300) for whole tiles."""
        x, _ = sn8_stack(ts)
        net.eval()
        out = []
        with (
            torch.no_grad(),
            torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"),
        ):
            for k in range(len(x)):
                xb = torch.from_numpy(x[k : k + 1]).to(DEVICE).permute(0, 3, 1, 2).float()
                p = net((xb - _mean) / _std).float().softmax(1)
                out.append(p[0, :, :SN8_SIZE, :SN8_SIZE].cpu().numpy())
        return np.stack(out)

    def sn8_train(ts):
        """Random crops (512 px in the slides), flips; cross-entropy (written out: CUDA's
        nll_loss2d is not deterministic) plus soft Dice over the four object classes."""
        xt, yt = sn8_stack(ts)
        X = torch.from_numpy(xt).to(DEVICE).permute(0, 3, 1, 2)
        Y = torch.from_numpy(yt).to(DEVICE)  # uint8 (an eighth of int64); widened per batch
        torch.manual_seed(SEED)
        net = SiameseUNet(c_out=len(SN8_CLASSES)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=SN8_LR, weight_decay=WD)
        per_pass = math.ceil(len(X) * SN8_CROPS / BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=SN8_LR, total_steps=SN8_EPOCHS * per_pass, pct_start=0.1
        )
        gen = torch.Generator().manual_seed(SEED)
        t0 = time.perf_counter()
        for epoch in range(SN8_EPOCHS):
            net.train()
            tl = torch.randperm(len(X) * SN8_CROPS, generator=gen) % len(X)
            # high is exclusive: no crop reaches a tile's last row and column (1 px of 1,300),
            # as in the slides' script; left as it is, so the slides' numbers reproduce
            corners = torch.randint(0, SN8_SIZE - SN8_CROP, (len(tl), 2), generator=gen)
            flips = torch.rand(len(tl), 2, generator=gen) < 0.5
            total = 0.0
            for i in range(0, len(tl), BATCH):
                xb, yb = [], []
                for k in range(i, min(i + BATCH, len(tl))):
                    r, c = corners[k].tolist()
                    xa = X[tl[k], :, r : r + SN8_CROP, c : c + SN8_CROP]
                    ya = Y[tl[k], r : r + SN8_CROP, c : c + SN8_CROP]
                    if flips[k, 0]:
                        xa, ya = xa.flip(2), ya.flip(1)
                    if flips[k, 1]:
                        xa, ya = xa.flip(1), ya.flip(0)
                    xb.append(xa)
                    yb.append(ya)
                xb = (torch.stack(xb).float() - _mean) / _std
                yb = torch.stack(yb).long()
                with torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    logits = net(xb).float()
                p = logits.softmax(1)
                one = F.one_hot(yb, len(SN8_CLASSES)).permute(0, 3, 1, 2).float()
                inter, size = (p * one).sum((0, 2, 3)), (p + one).sum((0, 2, 3))
                dice = 1 - ((2 * inter + 1) / (size + 1))[1:].mean()
                loss = -(logits.log_softmax(1) * one).sum(1).mean() + dice
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
            if epoch % 10 == 9:
                print(
                    f"pass {epoch + 1}: loss {total / per_pass:.3f}, {time.perf_counter() - t0:.0f} s"
                )
        return net

    return sn8_predict, sn8_train


@app.cell
def _(part, sn8_predict, sn8_train):
    sn8_net = sn8_train(part["train"])
    sn8_prob = sn8_predict(sn8_net, part["test"])
    return sn8_net, sn8_prob


@app.cell
def _(
    Image,
    ImageDraw,
    SLIDES,
    SN8_CLASSES,
    SN8_MIN_PX,
    SN8_ROAD_PX,
    SN8_SIZE,
    SN8_TOL,
    label,
    np,
    part,
    sn8_prob,
):
    def blobs(mask):
        """Connected components, numbered 1..n, without those under SN8_MIN_PX (about 2.8 m2)."""
        comps, n = label(mask)
        if n == 0:
            return comps
        size = np.bincount(comps.ravel(), minlength=n + 1)
        keep = size >= SN8_MIN_PX
        keep[0] = False
        new = np.zeros(n + 1, dtype=comps.dtype)
        new[keep] = np.arange(1, int(keep.sum()) + 1)
        return new[comps]

    def match(gt, pred, iou_min=0.5):
        """One-to-one matches of predicted blobs and true footprints at IoU >= 0.5; merges."""
        ng, npred = int(gt.max()), int(pred.max())
        inter = np.bincount(
            (gt.astype(np.int64) * (npred + 1) + pred).ravel(), minlength=(ng + 1) * (npred + 1)
        ).reshape(ng + 1, npred + 1)
        area_g, area_p = inter.sum(1), inter.sum(0)
        iou = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
        iou[0, :] = iou[:, 0] = 0
        used_g, used_p = set(), set()
        for _, g, p in sorted(
            ((iou[g, p], g, p) for g, p in zip(*np.nonzero(iou >= iou_min), strict=True)),
            reverse=True,
        ):
            if g not in used_g and p not in used_p:
                used_g.add(int(g))
                used_p.add(int(p))
        owner = np.where(inter[1:, 1:] > area_g[1:, None] / 2)
        per_comp = np.bincount(owner[1], minlength=npred) if npred else np.zeros(0, int)
        merged = {int(g) + 1 for g, p in zip(*owner, strict=True) if per_comp[p] > 1}
        return {
            "hits": len(used_g),
            "true": ng,
            "pred": npred,
            "merged": len(merged),
            "merged_ids": merged,
        }

    det = {"hits": 0, "true": 0, "pred": 0, "merged": 0}
    calls = np.zeros((2, 2), int)  # [true flooded, called flooded]
    km = np.zeros((2, 2))
    for _t, _p in zip(part["test"], sn8_prob, strict=True):
        _m = match(_t["inst"], blobs(np.isin(_p.argmax(0), (1, 2))))
        for _k in det:
            det[_k] += _m[_k]
        _sums = np.bincount(
            _t["inst"].ravel(), weights=(_p[2] - _p[1]).ravel(), minlength=len(_t["buildings"]) + 1
        )
        for _k, _b in enumerate(_t["buildings"], start=1):
            if _b["area_px"]:
                calls[int(_b["flooded"]), int(_sums[_k] > 0)] += 1
        for _road in _t["roads"]:
            _img = Image.new("L", (SN8_SIZE, SN8_SIZE), 0)
            ImageDraw.Draw(_img).line(_road["line"], fill=1, width=SN8_ROAD_PX, joint="curve")
            _on = np.asarray(_img, dtype=bool)
            if _on.any():
                km[int(_road["flooded"]), int((_p[4][_on] - _p[3][_on]).sum() > 0)] += _road[
                    "length_m"
                ]
    _y = np.stack([t["lab"] for t in part["test"]])
    _a = sn8_prob.argmax(1)
    pixel_f1 = {
        SN8_CLASSES[c]: 2
        * ((_a == c) & (_y == c)).sum()
        / max((_a == c).sum() + (_y == c).sum(), 1)
        for c in range(1, 5)
    }
    print("pixel F1:", {k: round(float(v), 3) for k, v in pixel_f1.items()})
    # The per-class F1s count a flooded roof called dry as an error of both classes. Apart:
    # finding buildings (flooded or not), and the flooded-or-dry call where one was found.
    _pb, _tb = np.isin(_a, (1, 2)), np.isin(_y, (1, 2))
    _found = _pb & _tb
    building_any_f1 = 2 * int(_found.sum()) / max(int(_pb.sum() + _tb.sum()), 1)
    flood_call_on_found = int((_a[_found] == _y[_found]).sum()) / max(int(_found.sum()), 1)
    print(
        f"buildings found, flooded or not: pixel F1 {building_any_f1:.3f}; on found building "
        f"pixels, flooded-or-dry called right {flood_call_on_found:.1%}"
    )
    print(
        f"buildings found at IoU >= 0.5: {det['hits']} of {det['true']} ({det['hits'] / det['true']:.0%}), "
        f"blobs {det['pred']}, true buildings merged with a neighbour {det['merged']}"
    )
    print(
        f"flooded buildings called flooded: {calls[1, 1]} of {calls[1].sum()}; dry called flooded: {calls[0, 1]}"
    )
    print(
        f"flooded road: {km[1, 1] / 1000:.1f} of {km[1].sum() / 1000:.1f} km found; dry road called flooded "
        f"{km[0, 1] / 1000:.1f} km"
    )
    if SLIDES:
        assert abs(det["hits"] / det["true"] - 0.1996) <= SN8_TOL
        assert abs(calls[1, 1] / calls[1].sum() - 0.3369) <= SN8_TOL
        _want = {"building": 0.4444, "flooded building": 0.2791, "road": 0.4289,
                 "flooded road": 0.3018}  # fmt: skip
        _want_found = {
            "building_any_f1": 0.6249,
            "flood_call_on_found": 0.6239,
        }  # from numbers.json
        for _k, _v in _want.items():
            assert abs(pixel_f1[_k] - _v) <= SN8_TOL, (_k, pixel_f1[_k])
        for _k, _v in _want_found.items():
            _got = {"building_any_f1": building_any_f1, "flood_call_on_found": flood_call_on_found}
            assert abs(_got[_k] - _v) <= SN8_TOL, (_k, _got[_k])
        assert abs(km[1, 1] / 1000 - 6.076) <= 20 * SN8_TOL  # km of flooded road found
        assert abs(km[0, 1] / 1000 - 5.673) <= 20 * SN8_TOL  # km of dry road called flooded
    return blobs, match


@app.cell
def _(SLIDES, SN8_TOL, math, np, part, roc_auc_score, sn8_net, sn8_predict, sn8_prob):
    # The footprints are known before the flood: score each by its mean P(flooded) - P(dry), and
    # set the threshold on the validation block to flag 80% of its flooded buildings.
    def building_scores(ts, prob):
        score, flooded = [], []
        for t, p in zip(ts, prob, strict=True):
            sums = np.bincount(
                t["inst"].ravel(), weights=(p[2] - p[1]).ravel(), minlength=len(t["buildings"]) + 1
            )
            for k, b in enumerate(t["buildings"], start=1):
                if b["area_px"]:
                    score.append(sums[k] / b["area_px"])
                    flooded.append(b["flooded"])
        return np.array(score), np.array(flooded)

    _sv, _yv = building_scores(part["val"], sn8_predict(sn8_net, part["val"]))
    _ranked = np.sort(_sv[_yv])[::-1]
    flood_threshold = float(_ranked[math.ceil(0.8 * len(_ranked)) - 1])
    _s, _y = building_scores(part["test"], sn8_prob)
    _flag = _s >= flood_threshold
    _auc = float(roc_auc_score(_y, _s))  # flooded vs dry, ranked by score, per footprint
    print(
        f"test: catches {_flag[_y].mean():.0%} of flooded buildings, flags {_flag.mean():.0%} of all "
        f"(truly flooded: {_y.mean():.0%}), precision {_y[_flag].mean():.0%}; "
        f"footprint AUC {_auc:.3f}"
    )
    assert len(_y) == 897 and round(float(_y.mean()), 3) == 0.417  # 374 of 897 flooded
    if SLIDES:
        assert abs(_flag[_y].mean() - 0.5241) <= SN8_TOL
        assert abs(_flag.mean() - 0.4114) <= SN8_TOL and abs(_y[_flag].mean() - 0.5312) <= SN8_TOL
        _want_auc = {"flood_auc": 0.6052}  # from numbers.json
        for _v in _want_auc.values():
            assert abs(_auc - _v) <= SN8_TOL, _auc
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Detection: one box per building, and non-maximum suppression

    A detector outputs a box and a score per object. Make proposals the simple way: the
    connected blobs of $P(\text{building})$ at three thresholds (0.3, 0.5, 0.7), so one building
    gives up to three nested boxes. **Non-maximum suppression** keeps the best-scoring box,
    drops every box that overlaps it with IoU > 0.5, and repeats. Write it, then check it
    against `torchvision.ops.nms`.
    """)
    return


@app.cell
def _(
    SLIDES,
    SN8_SIZE,
    SN8_TOL,
    blobs,
    box_iou,
    find_objects,
    ndimage_mean,
    nms,
    np,
    part,
    sn8_prob,
    torch,
):
    def boxes_of(comps, score):
        """Boxes (x0, y0, x1, y1) and mean scores of labelled blobs."""
        n = int(comps.max())
        if n == 0:
            return np.zeros((0, 4)), np.zeros(0)
        sl = find_objects(comps)
        present = [k for k in range(n) if sl[k] is not None]
        boxes = [[sl[k][1].start, sl[k][0].start, sl[k][1].stop, sl[k][0].stop] for k in present]
        return np.array(boxes, float).reshape(-1, 4), np.asarray(
            ndimage_mean(score, comps, [k + 1 for k in present]), float
        )

    def nms_scratch(boxes, score, iou_max=0.5):
        """Keep the best box, drop every box overlapping it by more than iou_max, repeat."""
        order = np.argsort(-score, kind="stable")
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = []
        while order.size:
            i, rest = order[0], order[1:]
            keep.append(int(i))
            x0 = np.maximum(boxes[i, 0], boxes[rest, 0])
            y0 = np.maximum(boxes[i, 1], boxes[rest, 1])
            x1 = np.minimum(boxes[i, 2], boxes[rest, 2])
            y1 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
            order = rest[inter / (area[i] + area[rest] - inter) <= iou_max]
        return keep

    _rows, _n_true, _n_prop, _same = [], 0, 0, True
    for _t, _p in zip(part["test"], sn8_prob, strict=True):
        _pb = _p[1] + _p[2]
        _props, _sc = zip(*[boxes_of(blobs(_pb > lv), _pb) for lv in (0.3, 0.5, 0.7)], strict=True)
        _b, _s = np.concatenate(_props), np.concatenate(_sc)
        _n_prop += len(_b)
        _ours = nms_scratch(_b, _s)
        _ref = nms(torch.from_numpy(_b), torch.from_numpy(_s), 0.5).tolist()
        _same &= sorted(map(tuple, _b[_ours])) == sorted(map(tuple, _b[_ref]))  # ties: same boxes
        _tb, _ = boxes_of(_t["inst"].astype(np.int64), np.ones((SN8_SIZE, SN8_SIZE)))
        _n_true += len(_tb)
        _iou = (
            box_iou(torch.from_numpy(_b[_ours]), torch.from_numpy(_tb)).numpy()
            if len(_ours) and len(_tb)
            else np.zeros((len(_ours), len(_tb)))
        )
        _taken = set()
        for _j in range(len(_ours)):  # best first: a hit claims its best free true box
            _free = [g for g in np.argsort(-_iou[_j]) if g not in _taken and _iou[_j, g] >= 0.5]
            if _free:
                _taken.add(_free[0])
            _rows.append((_s[_ours[_j]], bool(_free)))
    _rows.sort(key=lambda r: -r[0])
    _hit = np.array([r[1] for r in _rows], float)
    _prec = np.cumsum(_hit) / np.arange(1, len(_hit) + 1)
    _rec = np.cumsum(_hit) / _n_true
    ap50 = float(
        np.sum(np.diff(np.concatenate([[0], _rec])) * np.maximum.accumulate(_prec[::-1])[::-1])
    )
    print(
        f"{_n_prop} proposals, {len(_rows)} kept; same boxes as torchvision: {_same}; AP at IoU 0.5: {ap50:.3f}"
    )
    assert _same  # the from-scratch NMS keeps torchvision's boxes, whatever the network
    if SLIDES:
        assert abs(_n_prop / 2487 - 1) <= SN8_TOL and abs(len(_rows) / 1224 - 1) <= SN8_TOL
        assert abs(ap50 - 0.1386) <= SN8_TOL
    return


@app.cell
def _(length_err, mo, severity):
    mo.md(rf"""
    ## 6. Decide: who rates severity next season?

    Metres of test cracking by LTPP severity (all cracking treated as linear):

    | Ruler | Low | Moderate | High |
    |:------|----:|---------:|-----:|
    | True mask, dataset ruler | {severity["mask"][0]:.0f} | {severity["mask"][1]:.0f} | {severity["mask"][2]:.0f} |
    | True mask, ridge ruler | {severity["same ruler"][0]:.0f} | {severity["same ruler"][1]:.0f} | {severity["same ruler"][2]:.0f} |
    | U-Net mask, ridge ruler | {severity["U-Net"][0]:.0f} | {severity["U-Net"][1]:.0f} | {severity["U-Net"][2]:.0f} |
    | Range image | {severity["range"][0]:.0f} | {severity["range"][1]:.0f} | {severity["range"][2]:.0f} |

    The U-Net finds *where* cracks run (strict F1 and IoU near the two annotations' agreement on
    the twin images; within 2 pixels the annotations agree more closely than the U-Net does);
    its skeleton's length per image is {100 * length_err["U-Net"][1]:+.0f}% from the reference
    at the median (the true mask's own skeleton: {100 * length_err["true mask"][1]:+.0f}%, so
    the estimator runs long and the U-Net's lengths are, if anything, short). Its
    widths are the annotation convention on a 4 mm grid, and LTPP's 6 mm limit falls inside one
    pixel.

    **Vote** (alone, then discuss, then again): A. the U-Net's mask with distance-transform
    widths; B. the U-Net for *where*, widths from the depth profile; C. the U-Net for *where*,
    widths re-annotated by inspectors; D. inspectors rate everything, as now.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(r"""
        **B.** A rates by pixel steps, and the three rulers disagree. B keeps the network's
        strength (location and length) and measures width where the information is, in the
        height profile, at the scanner's native 1 mm when the severity limit is close; it still
        needs a check against field measurements, which this dataset does not have. C repeats
        the convention problem unless inspectors measure rather than draw. D is the baseline.

        **When not to use a mask width at all:** whenever the limit is within a pixel or two of
        the pixel size, and for structural cracks of 0.3 mm, which need about 0.1 mm per pixel.

        **Impact:** a missed or under-rated crack is a liability question for the agency, and
        the inspector's work moves from drawing cracks to auditing a network's widths.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question.** A drone photographs a concrete bridge deck, and a U-Net reports a largest
    crack width of 0.4 mm, above the owner's 0.3 mm repair limit. Name the numbers you need
    before believing the width, and one check you would run on the masks.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        The ground pixel size (distance to the deck, focal length, pixel pitch, and the lens's
        distortion, or a target of known size in the frame), and that size relative to 0.3 mm:
        coarser than about 0.1 mm per pixel, a 0.4 mm crack is one pixel and its "width" is the
        annotator's convention. Checks: the masks' actual values (grey edges, faint masks) and
        two annotators on the same images (how far two annotations agree).
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas

    - **Masks are not always 0/255.** Read them with an explicit cut (`> 127`) and look at what
      `> 0` would have given; here four masks are faint noise on 98% of their pixels.
    - **Thin structures and strict IoU.** A one-pixel shift costs about 40% of the IoU; report
      a tolerance score too, and the annotators' agreement as a scale (not a hard ceiling).
    - **Input sizes.** Four halvings and four doublings must land on the same size: 624 and
      512 are multiples of 16; 312 is not (pad to 320); 1300 is not (pad to 1312).
    - **Determinism on a GPU.** `torch.use_deterministic_algorithms(True)` needs
      `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA starts, and some kernels have no
      deterministic version (2-D cross-entropy on CUDA): write the loss out from
      `log_softmax`.
    - **Degrees are not metres.** Reproject to a metric grid before measuring lengths or areas.

    ## Going deeper (Forsyth)

    - §19.2.1: encoders and decoders, the two halves of today's network.
    - §19.2.2: "making data blocks bigger": transposed convolution as zero insertion followed
      by a convolution. Show it equals the stamps here (insert zeros between input pixels, pad,
      and convolve with the flipped kernel).
    - §18.2.1–18.2.3: how detectors work: proposals, non-maximum suppression, bounding-box
      regression, R-CNN to Faster R-CNN's anchors.
    - §18.2.5: evaluating detectors: IoU thresholds and average precision.

    **Exercises.** (1) Implement transposed convolution by zero insertion and match
    `conv_transpose2d`. (2) Add a third class, "building edge", to the SpaceNet labels and
    measure how many merged buildings it separates.

    ## Things to remember

    - Shrink to see; grow back with skips.
    - Upsampling is convolution run backward.
    - Thin masks: score with a tolerance, and know the annotators' agreement.
    - Width needs millimetres per pixel, and pixels finer than the limit.
    - Count objects, not pixels; measure in metres, not degrees.
    """)
    return


if __name__ == "__main__":
    app.run()
