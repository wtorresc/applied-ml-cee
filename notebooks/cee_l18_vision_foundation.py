"""Applied Machine Learning for CEE — Lecture 18: Vision Foundation Models.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l18_vision_foundation.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: SDNET2018 (Maguire, Dorafshan & Thomas, 2018, CC BY 4.0) from
``bash scripts/fetch_sdnet2018.sh``, then ``pixi run -e aml-env python scripts/derive_sdnet2018.py``,
``pixi run -e aml-env python scripts/derive_sdnet2018_vit.py`` and
``PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_sdnet2018_clip.py`` (CLIP's image
embeddings of every tile, about 30 minutes on a GPU). 3DCrack (Zhang et al., 2025, CC BY 4.0) from
``bash scripts/fetch_3dcrack.sh``; SpaceNet 8 Germany (Hänsch et al., 2022, CC BY-SA 4.0) from
``bash scripts/fetch_spacenet8.sh``. CLIP ViT-L/14 at 336 px (OpenAI; MIT-licensed repository, the
model card scopes the model to research) and SAM 2.1 Hiera base-plus (Apache-2.0) come from the
Hugging Face Hub at pinned revisions on the first run.

Set ``L18_SETTING=reduced`` (the default: a laptop CPU or a free Colab GPU) or
``L18_SETTING=slides`` (the slides' full run) to choose the setting (see the header cell).
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
    # Applied Machine Learning for CEE — Lecture 18: Vision Foundation Models

    **Slides:** *Applied Machine Learning for CEE*, Lecture 18 (`lectures/cee_l18_vision_foundation`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §1.2 (nearest neighbours), §6.3.1 (the cosine distance), §7.1–7.2 (canonical correlation analysis: one space for words and pictures), §16.1.1 (the softmax as a classifier's last layer), and §18.2.5 (IoU, precision, recall, and average precision).

    **Data:**
    - SDNET2018 (Maguire, Dorafshan & Thomas, 2018; CC BY 4.0): Lecture 15's 56,092 tiles from
      230 photographs of bridge decks, walls, and pavements, split by photo. Fetch with
      `bash scripts/fetch_sdnet2018.sh`, then `python scripts/derive_sdnet2018.py` (the tile
      index and ResNet-18 features), `python scripts/derive_sdnet2018_vit.py` (Lecture 17's
      ViT-B/16 features), and `python scripts/derive_sdnet2018_clip.py` (CLIP's image
      embeddings of every tile; about 30 minutes on a GPU, so this notebook loads them and
      re-embeds a sample to check them).
    - 3DCrack (Zhang, Wang, Hsieh, Yang, Yezzi & Tsai, 2025; doi:10.21227/2etq-xh36; CC BY 4.0):
      Lecture 16's range images of Georgia pavements with crack masks. Fetch with
      `bash scripts/fetch_3dcrack.sh`.
    - SpaceNet 8, Germany (Hänsch et al., 2022; CC BY-SA 4.0; the Maxar imagery is under a
      non-commercial licence, fine for teaching): Lecture 16's satellite tiles of the Ahr valley
      with OpenStreetMap building footprints. Fetch with `bash scripts/fetch_spacenet8.sh`.
    - CLIP ViT-L/14 at 336 pixels (`openai/clip-vit-large-patch14-336`, Radford et al., 2021) at
      a pinned revision, about 1.7 GB from the Hugging Face Hub. OpenAI's repository is
      MIT-licensed; the model card scopes the model to research use, which suits a course and
      not, as is, a deployed screen (§7).
    - SAM 2.1 Hiera base-plus (`facebook/sam2.1-hiera-base-plus`, Ravi et al., 2024; Apache-2.0)
      at a pinned revision, about 0.3 GB.

    This notebook follows the lecture in the deck's order, §0 to §7: **warm-up → the question and
    the attempt → foundation models → derive → the picture revealed → zero-shot, few-shot, and
    three ways to adapt → SAM → decide and check yourself**. Every number on the slides is
    recomputed here and checked with an `assert`. Everything about CLIP runs on the CPU from the
    cached image embeddings (CLIP's text side is small), except a check that re-embeds a sample
    of tiles; SAM runs on crack images and buildings.

    **Two settings**, chosen by the environment variable `L18_SETTING` (whatever the device):

    - `reduced` (the default): for a laptop CPU or a free Colab GPU; 8 minutes on the CPU of the
      slides' machine (56 threads, no GPU, November 20, 2026), 30 minutes on four of its threads
      (November 19), most of it SAM, with up to 8 GB of memory. Everything about CLIP (zero-shot, the
      probes, nearest neighbours, the loss) is the slides' computation on the cached
      embeddings, checked to 1e-6, or to 1e-4 for what passes through CLIP's text tower (its
      float32 sums move with the number of CPU threads); the cache is checked on 32
      re-embedded tiles; SAM runs on every 4th crack image and all 300 buildings, with loose
      checks.
    - `slides`: the slides' full run, 17 minutes on the slides' GPU (an RTX 3090 Ti) shared
      with other jobs. There SAM's digits are checked; on another GPU or the CPU its masks
      move a little, and its asserts allow a stated difference.

    **Your tasks** (from the slides): pick a caption for six tiles, write CLIP's classifier and
    match Hugging Face's, write the contrastive loss and match it in float64, choose prompts on
    the validation photos (try your own), fit probes on 1 to 16 labelled photos per surface,
    click on cracks and buildings with SAM, then decide what screens a new owner's decks.
    """)
    return


@app.cell
def _():
    import csv
    import json
    import math
    import os
    import random
    import re
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub loads
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt, label, maximum_position
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from transformers import (
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizer,
        Sam2Model,
        Sam2Processor,
    )
    from transformers.models.clip.modeling_clip import image_text_contrastive_loss
    from transformers.utils import logging as hf_logging

    SEED = 18
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed):
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    # As the slides' figure script: deterministic kernels wherever PyTorch has one
    torch.use_deterministic_algorithms(True, warn_only=True)
    hf_logging.set_verbosity_error()  # SAM 2.1's checkpoint is the video model's; its note is noise
    hf_logging.disable_progress_bar()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"
    # The setting, independent of the device: "reduced" (default) or "slides" (the header cell)
    SETTING = os.environ.get("L18_SETTING", "reduced").strip().lower()
    if SETTING not in {"reduced", "slides"}:
        raise ValueError(f"L18_SETTING must be 'reduced' or 'slides', not {SETTING!r}")
    SLIDES = SETTING == "slides"
    # SAM's digits are checked only on the slides' GPU, in "slides"
    REFERENCE_GPU = SLIDES and GPU and "3090 Ti" in torch.cuda.get_device_name(0)
    # CLIP's text side, the probes, k-NN, and every zero-shot probability are computed on the CPU
    # from the cached image embeddings: they reproduce the slides to round-off.
    EXACT = 1e-6
    # What passes through CLIP's text tower (float32, on the CPU) moves with the number of CPU
    # threads: with 4 threads against the slides' 56, text embeddings by 3e-8, probabilities by
    # 1e-6, and one average precision by 1.5e-5
    CLIP_TOL = 1e-4
    LOSS_TOL = 1e-12  # our contrastive loss against Hugging Face's, float64 (0 on the slides' run)
    # Re-embedded tiles against the cache, relative difference per row: about 1e-5 on the slides'
    # GPU (batches of 32 here, 128 for the cache), about 1e-6 on a CPU; GPUs that use TF32
    # convolutions move further.
    CACHE_TOL = 1e-3
    LOGIT_TOL = 0.01  # CLIP's own forward pass (image tower on the CPU) against ours, in logits
    # SAM ran on the slides' GPU. There the same code reproduces the slides exactly; on another
    # GPU or on the CPU its masks move a little, so the IoU medians allow a stated difference
    SAM_TOL = 0.005 if GPU else 0.03
    # "reduced": SAM on every 4th crack image (49 of 197) and the cache checked on 32 tiles
    CRACK_EVERY = 1 if SLIDES else 4
    CACHE_SAMPLE = 256 if SLIDES else 32  # tiles re-embedded to check the cache
    CACHE_BATCH = 32  # tiles per pass: CLIP ViT-L/14 at 336 px fits in about 3 GB of GPU memory

    def shown(x, text):
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("0.66")."""
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    def require(path, hint):
        """``path`` if it exists; otherwise say which script writes it."""
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: {hint}")
        return path

    def find_root():
        """The repository, found from its root, ``notebooks/``, or ``ipynb/`` (Jupyter's cwd)."""
        rel = Path("data") / "derived" / "sdnet2018_tiles.csv"
        bases = [Path.cwd(), Path.cwd().parent]
        if "__file__" in globals():
            bases.insert(0, Path(__file__).resolve().parents[1])
        for base in bases:
            if (base / rel).exists():
                return base
        raise FileNotFoundError(
            f"{rel} not found from {Path.cwd()}: run `bash scripts/fetch_sdnet2018.sh`, then "
            "`python scripts/derive_sdnet2018.py`, from the repository root"
        )

    ROOT = find_root()
    DATA = ROOT / "data"
    RAW = DATA / "raw" / "sdnet2018" / "SDNET2018"
    TILES_CSV = DATA / "derived" / "sdnet2018_tiles.csv"
    R18_FEATURES = DATA / "derived" / "sdnet2018_resnet18.npy"
    VIT_FEATURES = DATA / "derived" / "sdnet2018_vit_b_16.npy"
    CLIP_FEATURES = DATA / "derived" / "sdnet2018_clip_l14_336.npy"
    CRACK3D = DATA / "raw" / "3dcrack" / "3DCrack_v1"
    SN8 = DATA / "raw" / "spacenet8" / "Germany_Training_Public"
    # Lecture 15's crack tiles, split, probe, and target
    SURFACES = ("deck", "wall", "pavement")
    SPLIT = (0.6, 0.8)  # cumulative share of each surface's photos: train, val (rest: test)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet's (the older caches)
    C_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)  # inverse L2 strengths tried on validation
    RECALL = 0.90  # the inspector's target: flag 9 of every 10 cracked tiles
    # CLIP ViT-L/14 at 336 px (Radford et al., 2021), OpenAI's weights at a pinned revision
    CLIP = "openai/clip-vit-large-patch14-336"
    CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    SURFACE_NOUN = {"deck": "bridge deck", "wall": "concrete wall", "pavement": "pavement"}
    # Prompt pairs (clean, cracked); "{s}" is the tile's surface. Chosen on validation photos.
    PROMPTS = {
        "concrete": ("a photo of concrete", "a photo of cracked concrete"),
        "surface": ("a photo of a {s}", "a photo of a cracked {s}"),
        "crack": (
            "a photo of a concrete surface without cracks",
            "a photo of a concrete surface with a crack",
        ),
        "close-up": (
            "a close-up photo of intact concrete",
            "a close-up photo of a crack in concrete",
        ),
    }
    CLASSES = [(s, c) for s in SURFACE_NOUN for c in (0, 1)]  # (surface, cracked): six captions
    SHOTS = (1, 2, 4, 8, 16)  # labelled training photos per surface for the few-shot probes
    SHOT_DRAWS = tuple(range(10))  # random draws of the photos for each k
    # The few-label probes' C: k photos leave no labels to tune it on. 0.01 is the C the probe on
    # all photos picks on all 47 validation photos, labels the few-label owner does not have
    FEW_C = 0.01
    FEW_C_CHECK = (0.003, 0.03)  # its neighbours on C_GRID, reported beside it as a check
    # The third way to adapt CLIP: nearest neighbours on its image embeddings, votes weighted by
    # 1 / cosine distance, k chosen on validation by a bracketing search over a fixed ladder
    KNN_START = (5, 10, 25, 50, 100)
    KNN_LADDER = (1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200)
    # SAM 2.1 (Ravi et al., 2024), Hiera base-plus, at a pinned revision
    SAM = "facebook/sam2.1-hiera-base-plus"
    SAM_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
    CLICKS = (1, 3, 5)  # positive clicks placed along a crack's centreline
    BUILDINGS = 300  # SpaceNet 8 test buildings sampled for the one-click test
    MIN_BUILDING_PX = 400  # about 45 m2 at 0.3-0.4 m pixels: a house, not a shed
    H3D, W3D, MASK_CUT = 624, 512, 127  # 3DCrack: rows x columns; a mask pixel is crack above 127
    SN8_SIZE, TEST_X = 1300, 40  # SpaceNet 8: pre-event tile size; test block: grid column >= 40
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    def caption(surface, cracked):
        """A caption for a tile, as a person might have written it."""
        noun = SURFACE_NOUN[surface]
        return f"a photo of a cracked {noun}" if cracked else f"a photo of a {noun}"

    print(f"device: {DEVICE}; setting: {SETTING}" + (" (the slides' GPU)" if REFERENCE_GPU else ""))
    return (
        BUILDINGS,
        CACHE_BATCH,
        CACHE_SAMPLE,
        CACHE_TOL,
        CLASSES,
        CLICKS,
        CLIP,
        CLIPImageProcessor,
        CLIPModel,
        CLIPTokenizer,
        CLIP_FEATURES,
        CLIP_REVISION,
        CLIP_TOL,
        CRACK3D,
        CRACK_EVERY,
        C_GRID,
        Counter,
        DEVICE,
        EXACT,
        F,
        FEW_C,
        FEW_C_CHECK,
        GPU,
        H3D,
        Image,
        ImageDraw,
        KNN_LADDER,
        KNN_START,
        KNeighborsClassifier,
        LOGIT_TOL,
        LOSS_TOL,
        LogisticRegression,
        MASK_CUT,
        MEAN,
        MIN_BUILDING_PX,
        PALETTE,
        PROMPTS,
        R18_FEATURES,
        RAW,
        RECALL,
        REFERENCE_GPU,
        SAM,
        SAM_REVISION,
        SAM_TOL,
        SEED,
        SETTING,
        SHOTS,
        SHOT_DRAWS,
        SLIDES,
        SN8,
        SN8_SIZE,
        SPLIT,
        STD,
        SURFACES,
        SURFACE_NOUN,
        Sam2Model,
        Sam2Processor,
        StandardScaler,
        TEST_X,
        TILES_CSV,
        VIT_FEATURES,
        W3D,
        average_precision_score,
        caption,
        csv,
        defaultdict,
        distance_transform_edt,
        image_text_contrastive_loss,
        json,
        label,
        math,
        maximum_position,
        np,
        pd,
        plt,
        re,
        require,
        roc_auc_score,
        shown,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–17

    On your own, no notes, two minutes:

    1. Lecture 6: cross-entropy for one example with 3 classes, when the right class gets
       $p = 0.5$?
    2. Lecture 15: what did the probe on frozen ResNet-18 features need that a network trained
       from scratch needed far more of?
    3. Lecture 17: BERT learned from books with no labels. Where did its "answers" come from?

    **Today:** a model that learned from captions, asked about cracks it has never been taught.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. $-\log 0.5 \approx 0.69$, whatever the other two classes get (checked below). Today
           CLIP is trained with this loss over a batch: each image must pick its own caption
           out of all the batch's captions (§3).
        2. Labelled photos. In Lecture 15, with 16 training photos the probe beat the network
           trained from scratch (that notebook checks the numbers). Today: how many labels does
           a newer model need, and can it start with none?
        3. From the text itself: hide a word, predict it. CLIP's labels are free in the same
           way: 400 million images that came with captions on the web (Radford et al., 2021).
        """)
        }
    )
    return


@app.cell
def _(math, shown):
    _ce = -math.log(0.5)
    print(f"-log 0.5 = {_ce:.4f}")
    assert shown(_ce, "0.69")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The question and the attempt

    ### A new owner, no labels yet

    Lecture 15's screen learned from 137 labelled photographs: a probe (logistic regression) on
    ImageNet's frozen ResNet-18, its threshold set to flag 90% of the cracked tiles of the
    validation photos. A new owner has photos of their own decks, walls, and pavements, and no
    labels. Screen now, or label first? How many?

    | Option | Labels needed |
    |:-------|:--------------|
    | Zero-shot: describe the classes in words | none |
    | Probe on a pretrained model | a few photos |
    | Lecture 15's screen | 137 photos, labelled |

    Labelling is the cost an owner sees first: an inspector marking every tile of a photo as
    cracked or not takes a while, and someone must check the marks. A foundation model trained
    on image–caption pairs from the web can be asked, in words, which tiles show cracks. How well
    does that work on our tiles, and how few labels beat it?

    First the tiles, Lecture 15's photo split, and the three caches of image features: ResNet-18
    (Lecture 15), ViT-B/16 (Lecture 17), and today's CLIP.
    """)
    return


@app.cell
def _(
    CLIP,
    CLIP_FEATURES,
    CLIP_REVISION,
    Image,
    MEAN,
    R18_FEATURES,
    RAW,
    SPLIT,
    STD,
    SURFACES,
    TILES_CSV,
    VIT_FEATURES,
    json,
    np,
    pd,
    require,
):
    def read_tile(path):
        """One tile as uint8 RGB (256, 256, 3)."""
        return np.array(Image.open(RAW / path).convert("RGB"))

    def photo_split(df, rng):
        """Whole photos to train/val/test: 60/20/20% of each surface's photos (Lecture 15)."""
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

    def load_tiles():
        """The tile index and the three feature caches, row for row, after checking how they were made.

        Raises:
            ValueError: If a cache was made with other weights, model, or preprocessing.
        """
        hint = "run `bash scripts/fetch_sdnet2018.sh`, then `python scripts/derive_sdnet2018.py`"
        df = pd.read_csv(require(TILES_CSV, hint))
        imagenet = f"resize 224x224 bilinear antialias; ImageNet mean {MEAN} std {STD}"
        caches = (  # what each cache's JSON must record: model, preprocessing, layer, shape
            (R18_FEATURES, hint, {"weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
                                  "preprocess": imagenet,
                                  "layer": "global average pool (fc replaced by identity)",
                                  "shape": [len(df), 512]}),
            (VIT_FEATURES, "run `python scripts/derive_sdnet2018_vit.py`",
             {"weights": "torchvision ViT_B_16_Weights.IMAGENET1K_V1", "preprocess": imagenet,
              "layer": "class token after the encoder's final layer norm (heads replaced by "
                       "identity)",
              "shape": [len(df), 768]}),
            (CLIP_FEATURES, "run `python scripts/derive_sdnet2018_clip.py`",
             {"model": CLIP, "revision": CLIP_REVISION,
              "preprocess": "CLIPImageProcessor: shorter side 336, centre crop 336, CLIP mean/std",
              "layer": "get_image_features(...).pooler_output: the projected image embedding "
                       "(768), not normalized",
              "shape": [len(df), 768]}),
        )  # fmt: skip
        feats = []
        for path, hint_f, want in caches:
            meta = json.loads(require(path.with_suffix(".json"), hint_f).read_text())
            bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
            if bad:
                raise ValueError(f"{path.name} was not made as this notebook expects: {bad}")
            f = np.load(require(path, hint_f))
            assert f.shape == tuple(want["shape"]), f.shape
            feats.append(f)
        return df, *feats

    return load_tiles, photo_split, read_tile


@app.cell
def _(SURFACES, load_tiles, np, photo_split):
    tiles, rfeat, vfeat, cfeat = load_tiles()
    # Lecture 15's split, exactly: its notebook's seed (0) and lecture number, not this SEED
    part = photo_split(tiles, np.random.default_rng((0, 15)))
    y_tile = tiles["cracked"].to_numpy()
    surf = tiles["surface"].to_numpy()
    is_train, is_val, is_test = (part == _r for _r in ("train", "val", "test"))
    _photos = {_r: tiles.loc[part == _r, "photo"].nunique() for _r in ("train", "val", "test")}
    train_photos = {
        _s: int(tiles.loc[is_train & (surf == _s), "photo"].nunique()) for _s in SURFACES
    }
    print(f"{len(tiles):,} tiles from {tiles['photo'].nunique()} photos; photos per set {_photos}")
    print(f"training photos per surface: {train_photos}; test tiles {int(is_test.sum()):,}")
    assert len(tiles) == 56_092 and tiles["photo"].nunique() == 230
    assert _photos == {"train": 137, "val": 47, "test": 46}  # "137 labelled photos"
    assert train_photos == {"deck": 32, "wall": 43, "pavement": 62}  # the notes: 32, 43, 62
    assert int(is_test.sum()) == 11_208
    assert (
        tiles.groupby("photo").apply(lambda g: len(set(part[g.index])), include_groups=False) == 1
    ).all()
    return cfeat, is_test, is_train, is_val, rfeat, surf, tiles, train_photos, vfeat, y_tile


@app.cell
def _(
    C_GRID,
    LogisticRegression,
    RECALL,
    StandardScaler,
    average_precision_score,
    math,
    np,
    roc_auc_score,
):
    class Probe:
        """Logistic regression with an L2 penalty on standardized features (Lecture 6)."""

        def __init__(self, scaler, model, C, val_ap):
            self.scaler, self.model, self.C, self.val_ap = scaler, model, C, val_ap

        def score(self, feats):
            """Probability of "cracked" for each row of ``feats``."""
            return self.model.predict_proba(self.scaler.transform(feats))[:, 1]

    def fit_probe(feats, y, train, val):
        """Fit on ``train`` for each C in C_GRID; keep the best average precision on ``val``."""
        scaler = StandardScaler().fit(feats[train])
        zt, zv = scaler.transform(feats[train]), scaler.transform(feats[val])
        best = None
        for c in C_GRID:
            # l1_ratio=0 is the L2 penalty (scikit-learn 1.8 deprecates penalty="l2")
            m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
            m.fit(zt, y[train])
            ap = average_precision_score(y[val], m.predict_proba(zv)[:, 1])
            if best is None or ap > best[0]:
                best = (ap, c, m)
        return Probe(scaler, best[2], best[1], float(best[0]))

    def threshold_for(p, y, recall=RECALL):
        """The highest threshold that flags at least ``recall`` of the cracked tiles (p >= t).

        Flag the top ceil(recall n) scores; floor((1 - 0.9) n) would be one too small whenever n
        is a multiple of 10 (1 - 0.9 is 0.0999... in floating point).
        """
        pos = np.sort(p[y == 1])
        return float(pos[len(pos) - math.ceil(recall * len(pos) - 1e-9)])

    def scores(p, y, t):
        """Ranking and decision metrics of probabilities ``p`` for labels ``y`` at threshold ``t``."""
        flag = p >= t
        return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                "recall": float(flag[y == 1].mean()), "review": float(flag.mean()),
                "acc": float(((p >= 0.5) == y).mean()), "prevalence": float(y.mean()),
                "n": len(y)}  # fmt: skip

    def close(got, want, tol):
        """Assert every metric in ``want`` is within ``tol`` of ``got``'s."""
        for k, v in want.items():
            assert abs(got[k] - v) <= tol, (k, got[k], v)

    return close, fit_probe, scores, threshold_for


@app.cell
def _(
    EXACT,
    cfeat,
    close,
    fit_probe,
    is_test,
    is_train,
    is_val,
    rfeat,
    scores,
    shown,
    threshold_for,
    vfeat,
    y_tile,
):
    # The probe on all 137 training photos, for each of the three image features
    full_probes = {}
    for _name, _f in (
        ("CLIP probe, all photos", cfeat),
        ("ViT-B/16 probe (Lecture 17)", vfeat),
        ("ResNet-18 probe (Lecture 15)", rfeat),
    ):
        _pr = fit_probe(_f, y_tile, is_train, is_val)
        _p = _pr.score(_f)
        full_probes[_name] = scores(
            _p[is_test], y_tile[is_test], threshold_for(_p[is_val], y_tile[is_val])
        )
        _s = full_probes[_name]
        print(
            f"{_name}: C = {_pr.C}; test AP {_s['ap']:.4f}, {100 * _s['recall']:.1f}% of cracks "
            f"flagged, {100 * _s['review']:.1f}% of tiles to review"
        )
    _want = {  # auc, ap, recall, review, acc (numbers.json)
        "CLIP probe, all photos": (0.8994333851612925, 0.7786404323098531, 0.877715704531347,
                                   0.38936473947180583, 0.9242505353319058),
        "ViT-B/16 probe (Lecture 17)": (0.8973519877765442, 0.7668746687857423,
                                        0.877715704531347, 0.41059957173447537,
                                        0.922733761598858),
        "ResNet-18 probe (Lecture 15)": (0.8415554674616079, 0.6659046754776506,
                                         0.861576660459342, 0.5059778729478943,
                                         0.9064061384725196),
    }  # fmt: skip
    for _k, _v in _want.items():
        close(
            full_probes[_k],
            dict(zip(("auc", "ap", "recall", "review", "acc"), _v, strict=True)),
            EXACT,
        )
        assert full_probes[_k]["n"] == 11_208 and shown(100 * full_probes[_k]["prevalence"], "14")
    _r18 = full_probes["ResNet-18 probe (Lecture 15)"]
    # Lecture 15's screen, quoted today: AP 0.67, 51% of tiles to review
    assert shown(_r18["ap"], "0.67") and shown(100 * _r18["review"], "51")
    return (full_probes,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CLIP's two encoders, and the cache checked

    CLIP (Radford et al., 2021) has an image encoder $f$ (here a ViT-L/14 at 336 pixels: 14-pixel
    patches, 24 transformer blocks) and a text encoder $g$, each followed by a linear projection
    into one shared 768-number space. Its learned inverse temperature $s$ is stored as a log.
    The text side runs here, on the CPU; the image side was run once over every tile by
    `scripts/derive_sdnet2018_clip.py`. The next cell re-embeds a random sample of tiles and
    compares them with the cache.
    """)
    return


@app.cell
def _(
    CLIP,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
    CLIP_REVISION,
    EXACT,
    F,
    cfeat,
    math,
    shown,
    torch,
):
    clip_model = CLIPModel.from_pretrained(CLIP, revision=CLIP_REVISION).eval()  # on the CPU
    clip_tok = CLIPTokenizer.from_pretrained(CLIP, revision=CLIP_REVISION)
    clip_proc = CLIPImageProcessor.from_pretrained(CLIP, revision=CLIP_REVISION)
    with torch.no_grad():
        _log_s = float(clip_model.logit_scale.detach())  # stored as a log
        clip_scale = float(clip_model.logit_scale.exp())  # s, CLIP's learned inverse temperature

    def clip_text(prompts):
        """Unit-length text embeddings (n, 768), float32, from CLIP's text side on the CPU."""
        enc = clip_tok(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            t = clip_model.get_text_features(**enc).pooler_output  # projected (n, 768)
        assert t.shape == (len(prompts), 768), t.shape
        return F.normalize(t, dim=1).numpy()

    _cfg = clip_model.config
    print(
        f"s = exp({_log_s:.4f}) = {clip_scale!r}; shared space "
        f"{_cfg.projection_dim}; image encoder: patch {_cfg.vision_config.patch_size}, "
        f"{_cfg.vision_config.image_size} px, {_cfg.vision_config.num_hidden_layers} blocks"
    )
    # The slide: s = 100. Training clips the log at log 100, and it ended there.
    assert abs(clip_scale - 100.00000762939453) < EXACT and shown(clip_scale, "100")
    assert abs(_log_s - math.log(100)) < 1e-6
    # "one 768-number space"; ViT-L/14 at 336 pixels
    assert _cfg.projection_dim == 768 == cfeat.shape[1]
    assert (_cfg.vision_config.patch_size, _cfg.vision_config.image_size) == (14, 336)
    assert _cfg.vision_config.num_hidden_layers == 24
    return clip_model, clip_proc, clip_scale, clip_text, clip_tok


@app.cell
def _(
    CACHE_BATCH,
    CACHE_SAMPLE,
    CACHE_TOL,
    CLIP,
    CLIPModel,
    CLIP_REVISION,
    DEVICE,
    GPU,
    Image,
    RAW,
    SEED,
    cfeat,
    clip_model,
    clip_proc,
    np,
    tiles,
    time,
    torch,
):
    # A second copy of CLIP on the GPU (the CPU copy above stays where it is)
    _model = (
        CLIPModel.from_pretrained(CLIP, revision=CLIP_REVISION).eval().to(DEVICE)
        if GPU
        else clip_model
    )
    _rows = np.sort(
        np.random.default_rng((SEED, 336)).choice(len(tiles), CACHE_SAMPLE, replace=False)
    )
    _x = torch.cat(
        [
            clip_proc(images=Image.open(RAW / tiles["path"].iloc[_r]).convert("RGB"),
                      return_tensors="pt")["pixel_values"]
            for _r in _rows
        ]
    )  # fmt: skip
    assert _x.shape == (CACHE_SAMPLE, 3, 336, 336), _x.shape
    with torch.no_grad():
        _model.get_image_features(pixel_values=_x[:2].to(DEVICE))  # warm-up, not timed
        if GPU:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _fresh = torch.cat(
            [
                _model.get_image_features(pixel_values=_x[_i : _i + CACHE_BATCH].to(DEVICE))
                .pooler_output.float()
                .cpu()
                for _i in range(0, CACHE_SAMPLE, CACHE_BATCH)
            ]
        ).numpy()
        if GPU:
            torch.cuda.synchronize()
        _sec = time.perf_counter() - _t0
    if GPU:
        del _model
        torch.cuda.empty_cache()
    assert _fresh.shape == (CACHE_SAMPLE, 768), _fresh.shape
    _norm = np.linalg.norm(cfeat[_rows], axis=1)
    _rel = np.linalg.norm(_fresh - cfeat[_rows], axis=1) / _norm
    cache_check = {
        "device": DEVICE.type,
        "rows": CACHE_SAMPLE,
        "max_rel": float(_rel.max()),
        "max_abs": float(np.abs(_fresh - cfeat[_rows]).max()),
        "norms": (float(_norm.min()), float(_norm.max())),
        # the network alone, all 56,092 tiles on this device (reading and resizing them is extra)
        "minutes_all": _sec / CACHE_SAMPLE * len(tiles) / 60,
    }
    print(
        f"{CACHE_SAMPLE} tiles re-embedded on the {DEVICE.type.replace('cuda', 'GPU').upper()}: largest relative difference from "
        f"the cache {cache_check['max_rel']:.1e} (absolute {cache_check['max_abs']:.1e}); "
        f"embedding norms {cache_check['norms'][0]:.1f} to {cache_check['norms'][1]:.1f}; "
        f"all tiles would take {cache_check['minutes_all']:.0f} min of network time here"
    )
    assert cache_check["max_rel"] <= CACHE_TOL
    return (cache_check,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: which caption fits each tile?

    Six test tiles, one of each kind, drawn at random (seed 18) from the test photos, and six
    captions:

    *a photo of a bridge deck*, *a photo of a cracked bridge deck*, *a photo of a concrete
    wall*, *a photo of a cracked concrete wall*, *a photo of a pavement*, *a photo of a cracked
    pavement*.

    Two minutes: for each tile, pick the caption you think fits best, and write it down. CLIP's
    answers come in §4.
    """)
    return


@app.cell
def _(CLASSES, SEED, caption, is_test, np, plt, read_tile, surf, tiles, y_tile):
    _rng = np.random.default_rng(SEED)
    six_rows = [
        int(_rng.choice(np.flatnonzero(is_test & (surf == _s) & (y_tile == _k))))
        for _s, _k in CLASSES
    ]
    six_captions = [caption(_s, _k) for _s, _k in CLASSES]
    print("rows:", six_rows)
    assert six_rows == [11576, 1053, 21946, 16591, 53868, 32517]
    # The notes: tiles 1 and 2 are a deck without and with a crack, 3 and 4 a wall, 5 and 6 a
    # pavement (tell students after they commit)
    assert [(surf[_r], int(y_tile[_r])) for _r in six_rows] == CLASSES
    _fig, _axes = plt.subplots(2, 3, figsize=(6.5, 4.6), layout="constrained")
    for _j, _r in enumerate(six_rows):
        _ax = _axes[_j % 2, _j // 2]  # as on the slide: columns deck, wall, pavement
        _ax.imshow(read_tile(tiles["path"].iloc[_r]))
        _ax.set_title(f"tile {_j + 1}", fontsize=11)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig
    return six_captions, six_rows


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Foundation models: train once, adapt

    A foundation model is trained once, on a very large amount of data and compute, on a broad
    task, for general use or to be adapted to other tasks. ImageNet networks (Lecture 15) and
    BERT (Lecture 17) are adapted by probing or fine-tuning; CLIP and GPT (Lecture 19) can also
    be used **zero-shot**, with no examples of the new task.

    | Way to use it | What it needs | Lecture |
    |:--------------|:--------------|:--------|
    | Zero-shot | the task in words, no examples | today |
    | Linear probe | a few labelled examples, a frozen network | 15, 17, today |
    | Nearest neighbours | labelled examples, a frozen network | 2, today |
    | Fine-tune | many examples, a GPU budget | 15 |

    Today runs the first three on the crack tiles (§5), then asks the same question of a model
    that draws masks, SAM (§6).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    ### A text encoder as the classifier

    **Assume** that an image and a caption that describes it land close together in the shared
    space, and that a caption written for a class describes the class. Then classify an image
    $x$ by the caption $t_c$ it is closest to, with Lecture 6's softmax over scaled cosines:

    $$p(c \mid x) = \frac{\exp\!\big(s \cos(f(x), g(t_c))\big)}{\sum_{c'} \exp\!\big(s \cos(f(x), g(t_{c'}))\big)} .$$

    With unit-length embeddings $z = f(x)/\lVert f(x) \rVert$ and $w_c = g(t_c)/\lVert g(t_c)
    \rVert$, the logits are $s\, w_c^\top z$: softmax regression whose weight vectors are not
    learned for the task but written in words, one caption per class. A new task needs new
    captions, not new training. $s = 100$ makes a cosine gap of 0.01 a logit gap of 1.

    The check: our formula on the cached embeddings against CLIP's own forward pass
    (`CLIPModel(...)`, which runs the image tower again, here on the CPU), for the six tiles and
    the six captions.
    """)
    return


@app.cell
def _(
    EXACT,
    Image,
    LOGIT_TOL,
    RAW,
    cfeat,
    clip_model,
    clip_proc,
    clip_scale,
    clip_text,
    clip_tok,
    np,
    six_captions,
    six_rows,
    tiles,
    torch,
):
    _enc = clip_tok(six_captions, padding=True, return_tensors="pt")
    _px = torch.cat(
        [
            clip_proc(images=Image.open(RAW / tiles["path"].iloc[_r]).convert("RGB"),
                      return_tensors="pt")["pixel_values"]
            for _r in six_rows
        ]
    )  # fmt: skip
    with torch.no_grad():
        _out = clip_model(
            input_ids=_enc["input_ids"], attention_mask=_enc["attention_mask"], pixel_values=_px
        )
    _ref = _out.logits_per_image.numpy()  # (6 tiles, 6 captions): CLIP's own forward pass
    _z = cfeat[six_rows] / np.linalg.norm(cfeat[six_rows], axis=1, keepdims=True)
    _W = clip_text(six_captions)  # (6, 768): the classifier's weights, one row per caption
    _ours = clip_scale * _z @ _W.T
    assert _ours.shape == _ref.shape == (6, 6)
    _d_logit = float(np.abs(_ours - _ref).max())
    _d_text = float(np.abs(_out.text_embeds.numpy() - _W).max())
    _d_img = float(np.abs(_out.image_embeds.numpy() - _z).max())
    print(
        f"largest difference: logits {_d_logit:.1e}; text embeddings {_d_text:.1e}; "
        f"unit image embeddings {_d_img:.1e} (image tower on the CPU against the cache)"
    )
    assert _d_text < EXACT  # the same text tower, the same batch
    assert _d_logit < LOGIT_TOL
    assert (np.argmax(_ours, axis=1) == np.argmax(_ref, axis=1)).all()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Trained to match images to captions

    CLIP learned $f$ and $g$ from 400 million image–caption pairs from the web, in batches of
    32,768 (Radford et al., 2021). In a batch of $N$ pairs, stack the unit image embeddings as the
    rows of $Z_I$ and the caption embeddings as the rows of $Z_T$; the $N \times N$ logits
    $s Z_I Z_T^\top$ have the true pairs on the diagonal. Cross-entropy along each row picks each
    image's caption among $N$; along each column, each caption's image among $N$:

    $$\mathcal{L} = \tfrac{1}{2}\big[\text{CE}_{\text{rows}}(sZ_IZ_T^\top) + \text{CE}_{\text{cols}}(sZ_IZ_T^\top)\big] .$$

    Lecture 6's loss, twice. Predicting a caption word by word is hard; picking it out of a batch
    is learnable, and every batch is a new 32,768-way classification task. The paper's cost: the
    largest ResNet took 18 days on 592 V100 GPUs, the largest ViT 12 days on 256. The model used
    here is that ViT-L/14, given one more epoch at 336 pixels.

    ### Your turn: predict the check

    ```python
    zi = img / img.norm(dim=1, keepdim=True)     # 16 tiles
    zt = txt / txt.norm(dim=1, keepdim=True)     # their 16 captions
    logits = scale * zi @ zt.T                   # (16, 16)
    target = torch.arange(16)                    # the diagonal
    loss = 0.5 * (cross_entropy(logits, target)
                  + cross_entropy(logits.T, target))
    ```

    The batch: 16 random SDNET2018 tiles, each with a caption from its labels ("a photo of a
    cracked concrete wall"). Against the loss in Hugging Face's CLIP, in float64: how close?
    Write a number.
    """)
    return


@app.cell
def _(
    Counter,
    EXACT,
    F,
    LOSS_TOL,
    SEED,
    caption,
    cfeat,
    clip_scale,
    clip_text,
    image_text_contrastive_loss,
    math,
    np,
    shown,
    surf,
    tiles,
    torch,
    y_tile,
):
    _rows = list(np.random.default_rng(SEED).choice(len(tiles), 16, replace=False))
    _caps = [caption(surf[_r], int(y_tile[_r])) for _r in _rows]
    _zi = torch.tensor(cfeat[_rows], dtype=torch.float64)
    _zi = _zi / _zi.norm(dim=1, keepdim=True)
    _zt = torch.tensor(clip_text(_caps), dtype=torch.float64)
    _logits = clip_scale * _zi @ _zt.T  # (batch, batch): image i against caption j
    assert _logits.shape == (16, 16)
    _target = torch.arange(16)
    _ours = 0.5 * (F.cross_entropy(_logits, _target) + F.cross_entropy(_logits.T, _target))
    # Hugging Face's function takes the text-by-image matrix
    _ref = image_text_contrastive_loss(_logits.T)
    _p = torch.softmax(_logits, dim=1)  # each image's probabilities over the 16 captions
    _same = torch.tensor([[_a == _b for _b in _caps] for _a in _caps])  # captions with equal text
    contrastive = {
        "batch": len(_rows),
        "loss": float(_ours),
        "diff": float(abs(_ours - _ref)),
        "chance": math.log(16),
        "captions": Counter(_caps),
        # a caption with other text is the most similar: equal captions tie up to round-off,
        # so the tile's own caption is compared through the best of its equal-text copies
        "other_text_best": int(
            (
                _logits.masked_fill(_same, -math.inf).max(dim=1).values
                > _logits.masked_fill(~_same, -math.inf).max(dim=1).values
            ).sum()
        ),
        # confidently wrong: under 5% of the probability on captions with the tile's own text
        "confident_wrong": int(((_p * _same).sum(dim=1) < 0.05).sum()),
    }
    print({_k: _v for _k, _v in contrastive.items() if _k != "captions"})
    print(dict(contrastive["captions"]))
    assert contrastive["batch"] == 16
    assert abs(contrastive["loss"] - 3.8438262241746335) < EXACT and shown(
        contrastive["loss"], "3.8438"
    )
    assert contrastive["diff"] < LOSS_TOL  # 0 on the slides' run: the same operations, same order
    assert shown(contrastive["chance"], "2.7726") and contrastive["loss"] > contrastive["chance"]
    # The notes: captions repeat in such a batch, several walls without cracks among them
    assert (
        len(contrastive["captions"]) < 16
        and contrastive["captions"]["a photo of a concrete wall"] >= 2
    )
    # ... the right caption is often not the most similar, and CLIP is confidently wrong on some
    assert contrastive["other_text_best"] == 8 and contrastive["confident_wrong"] == 4
    return (contrastive,)


@app.cell
def _(contrastive, mo):
    _c = contrastive["captions"]
    mo.md(f"""
    ### The same loss, to the last bit

    | Batch of 16 tiles and captions | Value |
    |:-------------------------------|------:|
    | Loss, ours | {contrastive["loss"]:.4f} |
    | Difference from Hugging Face's | {contrastive["diff"]:.0f} |
    | Chance level, log 16 | {contrastive["chance"]:.4f} |

    Identical in float64: the same operations in the same order. The loss is above log 16, the
    loss of a uniform guess. The batch holds only {len(_c)} distinct captions
    ({_c["a photo of a pavement"]} uncracked pavements, {_c["a photo of a concrete wall"]}
    uncracked walls), and the loss still counts only the diagonal as right, which is why CLIP
    needs captions as varied as the web's. For {contrastive["other_text_best"]} of the 16 tiles a
    caption with other text is the most similar one, and {contrastive["confident_wrong"]} tiles put
    less than 5% of their probability on any caption with their own text: CLIP is confidently
    wrong on them. The next section shows how.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The picture revealed: CLIP's softmax

    The attempt again: each row is one of the six tiles, and CLIP's softmax ($s = 100$) over the
    six captions. Compare with what you wrote.
    """)
    return


@app.cell
def _(
    CLIP_TOL,
    cfeat,
    clip_scale,
    clip_text,
    np,
    plt,
    read_tile,
    shown,
    six_captions,
    six_rows,
    tiles,
    torch,
):
    _z = cfeat[six_rows] / np.linalg.norm(cfeat[six_rows], axis=1, keepdims=True)
    six_prob = torch.softmax(
        torch.tensor(clip_scale * _z @ clip_text(six_captions).T), dim=1
    ).numpy()
    assert six_prob.shape == (6, 6)
    _want = np.array([
        [0.002464965218678117, 0.0009861350990831852, 0.7615768909454346, 0.018526015803217888,
         0.21237680315971375, 0.004069299437105656],
        [0.0010806693462654948, 0.006413720548152924, 0.34093791246414185, 0.0991724282503128,
         0.5204039812088013, 0.031991321593523026],
        [0.0007033486035652459, 0.00023729154781904072, 0.7891458868980408, 0.005754818674176931,
         0.20229287445545197, 0.0018657911568880081],
        [0.00028325128369033337, 0.0001530540466774255, 0.8747695684432983, 0.019012587144970894,
         0.10332947224378586, 0.0024520945735275745],
        [0.0010702002327889204, 0.0026037790812551975, 0.3141149878501892, 0.026929693296551704,
         0.6210379004478455, 0.03424348309636116],
        [0.0003633360320236534, 0.03263300284743309, 0.2481793761253357, 0.6335911154747009,
         0.04057588055729866, 0.04465731233358383],
    ])  # fmt: skip
    assert np.abs(six_prob - _want).max() < CLIP_TOL
    _best = six_prob.argmax(axis=1)
    _surface_right = int((_best // 2 == np.arange(6) // 2).sum())
    _crack_right = int((_best % 2 == np.arange(6) % 2).sum())
    _caption_right = int((_best == np.arange(6)).sum())
    print("each tile's most probable caption:", [six_captions[_b] for _b in _best])
    # The alt text, tile by tile: the most probable caption and its probability
    _alt = [("concrete wall", "0.76"), ("pavement", "0.52"), ("concrete wall", "0.79"),
            ("concrete wall", "0.87"), ("pavement", "0.62"), ("cracked concrete wall", "0.63")]  # fmt: skip
    for _i, (_cap, _pr) in enumerate(_alt):
        assert six_captions[_best[_i]] == f"a photo of a {_cap}", (_i, six_captions[_best[_i]])
        assert shown(six_prob[_i, _best[_i]], _pr), (_i, six_prob[_i, _best[_i]])
    # The notes: the right surface for 3 of 6 tiles; cracked or not, for 4 of 6; the whole
    # caption for 2 of 6 (the slide's title); no bridge deck is called a deck
    assert (_surface_right, _crack_right, _caption_right) == (3, 4, 2)
    assert not np.isin(_best, [0, 1]).any()
    _fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    _grid = _fig.add_gridspec(1, 2, width_ratios=(1.0, 1.6))
    _sub = _grid[0].subgridspec(2, 3)
    for _j, _r in enumerate(six_rows):
        _ax = _fig.add_subplot(_sub[_j % 2, _j // 2])
        _ax.imshow(read_tile(tiles["path"].iloc[_r]))
        _ax.set_title(f"{_j + 1}", fontsize=10)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _ax = _fig.add_subplot(_grid[1])
    _ax.imshow(six_prob, cmap="cividis", vmin=0, vmax=1)
    for _i in range(6):
        for _j in range(6):
            _ax.text(_j, _i, f"{six_prob[_i, _j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if six_prob[_i, _j] < 0.5 else "black")  # fmt: skip
    _ax.set_yticks(range(6), [f"tile {_i + 1}" for _i in range(6)])
    _ax.set_xticks(range(6), [_c.replace("a photo of a ", "") for _c in six_captions],
                   rotation=35, ha="right")  # fmt: skip
    _fig
    return (six_prob,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Six tiles: 2 of 6 captions right

    CLIP picks the whole caption, surface and crack, for 2 of the 6 tiles. The right surface
    gets the most weight for 3 of 6, cracked or not is right for 4 of 6, and neither deck tile
    is called a deck. Compare with the answers you wrote.

    Six tiles are an anecdote: before saying what CLIP "knows", ask all the test tiles. The next
    cell scores the six captions over the 11,208 test tiles in two halves: the surface of the
    most probable caption, and $P(\text{cracked})$, the three cracked captions' total, as a
    ranking.
    """)
    return


@app.cell
def _(
    CLIP_TOL,
    SURFACE_NOUN,
    cfeat,
    clip_scale,
    clip_text,
    is_test,
    np,
    pd,
    roc_auc_score,
    shown,
    six_captions,
    surf,
    torch,
    y_tile,
):
    _z = cfeat[is_test] / np.linalg.norm(cfeat[is_test], axis=1, keepdims=True)
    _p = torch.softmax(torch.tensor(clip_scale * _z @ clip_text(six_captions).T), dim=1).numpy()
    assert _p.shape == (int(is_test.sum()), 6)
    _names = list(SURFACE_NOUN)
    _s, _y = surf[is_test], y_tile[is_test]
    _called = np.array(_names)[_p.argmax(axis=1) // 2]  # the most probable caption's surface
    _by_surface = _p.reshape(-1, 3, 2).sum(axis=2)  # (n, 3): P(deck), P(wall), P(pavement)
    _wp = np.isin(_s, ["wall", "pavement"])
    _p_wall = _by_surface[_wp, 1] / (_by_surface[_wp, 1] + _by_surface[_wp, 2])
    _commonest = pd.Series(_s).value_counts()
    six_all = {
        "tiles": len(_s),
        "surface_accuracy": float(np.mean(_called == _s)),
        "commonest_surface": str(_commonest.index[0]),
        "commonest_share": float(_commonest.iloc[0] / len(_s)),
        "crack_auc": float(roc_auc_score(_y, _p[:, 1::2].sum(axis=1))),
        "wall_vs_pavement_auc": float(roc_auc_score(_s[_wp] == "wall", _p_wall)),
        "called": {_a: {_b: int(np.sum((_s == _a) & (_called == _b))) for _b in _names}
                   for _a in _names},
        "tiles_by_surface": {_a: int(np.sum(_s == _a)) for _a in _names},
    }  # fmt: skip
    print({_k: _v for _k, _v in six_all.items() if _k != "called"})
    print("rows: the tile's surface; columns: the surface CLIP names:", six_all["called"])
    _want = {"tiles": 11208, "surface_accuracy": 0.44557458957887225,
             "commonest_surface": "pavement", "commonest_share": 0.43843683083511775,
             "crack_auc": 0.777433001868536, "wall_vs_pavement_auc": 0.6247916462993124}  # fmt: skip
    for _k, _v in _want.items():
        if isinstance(_v, (int, str)):
            assert six_all[_k] == _v, (_k, six_all[_k])
        else:
            assert abs(six_all[_k] - _v) < CLIP_TOL, (_k, six_all[_k])
    assert six_all["called"] == {
        "deck": {"deck": 4, "wall": 2497, "pavement": 271},
        "wall": {"deck": 0, "wall": 2297, "pavement": 1225},
        "pavement": {"deck": 61, "wall": 2160, "pavement": 2693},
    }
    assert six_all["tiles_by_surface"] == {"deck": 2772, "wall": 3522, "pavement": 4914}
    # The slide: 45% named right, 44% by always answering "pavement"; 4 of 2,772 deck tiles called
    # "bridge deck"; cracked or not, AUC 0.78; wall or pavement, AUC 0.62. The notes: 2,497 deck
    # tiles called walls, "barely more often" than always answering "pavement"
    assert shown(100 * six_all["surface_accuracy"], "45")
    assert shown(100 * six_all["commonest_share"], "44")
    assert shown(six_all["crack_auc"], "0.78") and shown(six_all["wall_vs_pavement_auc"], "0.62")
    assert 0 < six_all["surface_accuracy"] - six_all["commonest_share"] < 0.01
    return (six_all,)


@app.cell
def _(mo, six_all):
    _c, _n = six_all["called"], six_all["tiles_by_surface"]
    mo.md(f"""
    ### Over all tiles: cracks, not surfaces

    | The six captions, {six_all["tiles"]:,} test tiles | Value |
    |:-----------------------------------------|------:|
    | Surface named right | {100 * six_all["surface_accuracy"]:.0f}% |
    | … always answering "pavement" | {100 * six_all["commonest_share"]:.0f}% |
    | Deck tiles called "bridge deck" | {_c["deck"]["deck"]} of {_n["deck"]:,} |
    | Cracked or not, as a ranking (AUC) | {six_all["crack_auc"]:.2f} |
    | Wall or pavement, as a ranking (AUC) | {six_all["wall_vs_pavement_auc"]:.2f} |

    CLIP names the surface barely more often than always answering "pavement" would, and deck
    tiles are nearly all called something else ({_c["deck"]["wall"]:,} of them walls): an
    SDNET2018 deck tile is a close-up of concrete with nothing in it that shows a bridge. Cracked
    against clean tiles rank well (AUC {six_all["crack_auc"]:.2f}, where 0.5 is a coin), walls
    against pavements barely ({six_all["wall_vs_pavement_auc"]:.2f}; the score is
    $P(\\text{{wall}}) / (P(\\text{{wall}}) + P(\\text{{pavement}}))$ over those tiles). A claim
    about what a model knows needs the test set, not a handful of examples. The screen in §5
    asks only about the crack.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Zero-shot, few-shot, and three ways to adapt

    ### Your turn: predict four failures

    1. Reword the zero-shot prompts.
    2. Screen with zero-shot and no labels at all.
    3. Click 1, 3, or 5 times on a crack's centreline (SAM, §6).
    4. Click once inside a building (SpaceNet 8, §6).

    Two minutes alone: write what happens to each, and why. The first two are revealed in this
    section, the last two in §6.

    ### Zero-shot: prompts, then a threshold

    For each tile, $p(\text{cracked})$ is the softmax over two prompts, a clean one and a cracked
    one; a prompt with `{s}` names the tile's surface. Four pairs, each scored by average
    precision on Lecture 15's validation photos; the best pair is tested once.
    """)
    return


@app.cell
def _(SURFACE_NOUN, clip_scale, clip_text, np, torch):
    def zero_shot(img, surfaces, prompts):
        """P(cracked) for each tile: a softmax over the two prompts' scaled cosine similarities.

        Args:
            img: Image embeddings (n, 768), not normalized.
            surfaces: Each tile's surface (deck, wall, pavement), for prompts that name it.
            prompts: (clean, cracked) templates.

        Returns:
            P(cracked), shape (n,).
        """
        z = img / np.linalg.norm(img, axis=1, keepdims=True)
        p = np.empty(len(img))
        for s, noun in SURFACE_NOUN.items():
            m = surfaces == s
            t = clip_text([prompts[0].format(s=noun), prompts[1].format(s=noun)])  # (2, 768)
            logits = clip_scale * z[m] @ t.T  # (tiles, 2): the text embeddings are the classifier
            p[m] = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return p

    return (zero_shot,)


@app.cell
def _(
    CLIP_TOL,
    PROMPTS,
    SURFACE_NOUN,
    cfeat,
    close,
    full_probes,
    is_test,
    is_val,
    scores,
    shown,
    surf,
    threshold_for,
    y_tile,
    zero_shot,
):
    zs_probs = {_n: zero_shot(cfeat, surf, _pr) for _n, _pr in PROMPTS.items()}
    _val_ap = {_n: scores(_p[is_val], y_tile[is_val], 0.5)["ap"] for _n, _p in zs_probs.items()}
    _best = max(_val_ap, key=_val_ap.get)
    _p = zs_probs[_best]
    _t90 = threshold_for(_p[is_val], y_tile[is_val])
    zs = {
        "val_ap": _val_ap,
        "best": _best,
        "at_half": scores(_p[is_test], y_tile[is_test], 0.5),  # no labels at all
        "at_90": scores(_p[is_test], y_tile[is_test], _t90),  # threshold set on labelled val photos
        "t90": _t90,
        "by_surface": {
            _s: scores(_p[is_test & (surf == _s)], y_tile[is_test & (surf == _s)], _t90)
            for _s in SURFACE_NOUN
        },
        "test_ap": {
            _n: scores(_q[is_test], y_tile[is_test], 0.5)["ap"] for _n, _q in zs_probs.items()
        },
    }
    print("validation AP:", {_k: round(_v, 4) for _k, _v in _val_ap.items()}, "-> best:", _best)
    for _k in ("at_half", "at_90"):
        print(
            f"{_k}: "
            + ", ".join(f"{_m} {zs[_k][_m]:.4f}" for _m in ("auc", "ap", "recall", "review"))
        )
    _want_val = {"concrete": 0.5084616536542934, "surface": 0.4307685366810713,
                 "crack": 0.48031241320679197, "close-up": 0.524357378969088}  # fmt: skip
    close(zs["val_ap"], _want_val, CLIP_TOL)
    assert all(
        shown(zs["val_ap"][_k], _t)
        for _k, _t in zip(PROMPTS, ("0.51", "0.43", "0.48", "0.52"), strict=True)
    )
    assert zs["best"] == "close-up"
    _metrics = ("auc", "ap", "recall", "review", "acc", "prevalence", "n")
    _half = (0.7604513734667885, 0.5180380882101326, 0.553072625698324, 0.21984296930763741,
             0.7954139900071378, 0.14373661670235546, 11208)  # fmt: skip
    close(zs["at_half"], dict(zip(_metrics, _half, strict=True)), CLIP_TOL)
    close(zs["at_90"], {"recall": 0.8448168839230292, "review": 0.6293718772305497}, CLIP_TOL)
    assert abs(zs["t90"] - 0.2077830582857132) < CLIP_TOL
    _by = {  # auc, ap, recall, review, acc, prevalence, n
        "deck": (0.7491605183169486, 0.5587959285648829, 0.8235294117647058, 0.6154401154401155,
                 0.8358585858585859, 0.16558441558441558, 2772),
        "wall": (0.7849632526452798, 0.5952400789663235, 0.8558875219683656, 0.6229415105053947,
                 0.8185689948892675, 0.16155593412833616, 3522),
        "pavement": (0.7540211321071553, 0.4336864424553386, 0.8507718696397941,
                     0.6418396418396418, 0.756003256003256, 0.11864061864061864, 4914),
    }  # fmt: skip
    for _s, _v in _by.items():
        close(zs["by_surface"][_s], dict(zip(_metrics, _v, strict=True)), CLIP_TOL)
    _want_test = {"concrete": 0.46859348305817367, "surface": 0.4196306204188425,
                  "crack": 0.4631100760251814, "close-up": 0.5180380882101326}  # fmt: skip
    close(zs["test_ap"], _want_test, CLIP_TOL)
    # The notes: test AP 0.52 (AUC 0.76) against Lecture 15's ResNet-18 probe, 0.67; at p = 0.5,
    # 55% of cracked tiles flagged and 22% of tiles to review; at the 90% threshold set on
    # validation, 84% and 63%, against ResNet-18's 51%
    _h, _n = zs["at_half"], zs["at_90"]
    assert shown(_h["ap"], "0.52") and shown(_h["auc"], "0.76")
    assert shown(full_probes["ResNet-18 probe (Lecture 15)"]["ap"], "0.67")
    assert shown(100 * _h["recall"], "55") and shown(100 * _h["review"], "22")
    assert shown(100 * _n["recall"], "84") and shown(100 * _n["review"], "63")
    assert shown(100 * full_probes["ResNet-18 probe (Lecture 15)"]["review"], "51")
    return zs, zs_probs


@app.cell
def _(PROMPTS, mo, zs):
    _rows = "\n".join(
        f"    | {_pr[0]} / {_pr[1]} | {zs['val_ap'][_n]:.2f} |" for _n, _pr in PROMPTS.items()
    )
    mo.md(f"""
    | Prompt pair (clean / cracked) | Validation AP |
    |:------------------------------|-------------:|
{_rows}

    The pair is chosen on the validation photos ({zs["best"]}), and the test photos are scored
    once: average precision {zs["at_half"]["ap"]:.2f} (AUC {zs["at_half"]["auc"]:.2f}), against
    Lecture 15's ResNet-18 probe, 0.67. With no labels the natural threshold is $p = 0.5$: it
    flags {100 * zs["at_half"]["recall"]:.0f}% of cracked tiles and sends
    {100 * zs["at_half"]["review"]:.0f}% of tiles to review. Set to flag 90% on labelled
    validation photos (so not label-free anymore), it flags {100 * zs["at_90"]["recall"]:.0f}%
    and sends {100 * zs["at_90"]["review"]:.0f}% to review, against ResNet-18's 51%.
    """)
    return


@app.cell
def _(mo):
    my_clean = mo.ui.text(
        value="a close-up photo of intact concrete", label="clean prompt", full_width=True
    )
    my_cracked = mo.ui.text(
        value="a close-up photo of a crack in concrete", label="cracked prompt", full_width=True
    )
    mo.vstack(
        [
            mo.md(
                "**Try your own pair** (use `{s}` for the tile's surface), on validation photos:"
            ),
            my_clean,
            my_cracked,
        ]
    )
    return my_clean, my_cracked


@app.cell
def _(
    CLIP_TOL,
    PROMPTS,
    average_precision_score,
    cfeat,
    is_val,
    mo,
    my_clean,
    my_cracked,
    surf,
    y_tile,
    zero_shot,
):
    _pair = (my_clean.value, my_cracked.value)
    _p = zero_shot(cfeat, surf, _pair)
    _ap = average_precision_score(y_tile[is_val], _p[is_val])
    if _pair == PROMPTS["close-up"]:  # the default: the slides' chosen pair
        assert abs(_ap - 0.524357378969088) < CLIP_TOL
    mo.md(
        f"Validation AP of your pair: **{_ap:.3f}** (the chosen pair: 0.524). Keep the test photos "
        "for the one pair you finally choose."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reveal 1: prompts move the answer

    The same images, the same model; only the words change. A post-hoc diagnostic on the test
    photos (every pair scored there; the choice was made on validation):
    """)
    return


@app.cell
def _(
    CLIP_TOL,
    PROMPTS,
    SURFACE_NOUN,
    is_test,
    mo,
    np,
    roc_auc_score,
    shown,
    surf,
    y_tile,
    zs,
    zs_probs,
):
    _order = sorted(PROMPTS, key=lambda _n: zs["test_ap"][_n])
    assert all(
        shown(zs["test_ap"][_n], _t)
        for _n, _t in zip(_order, ("0.42", "0.46", "0.47", "0.52"), strict=True)
    )
    assert _order == ["surface", "crack", "concrete", "close-up"]
    # The notes: the validation choice was also the test's best
    assert max(zs["test_ap"], key=zs["test_ap"].get) == zs["best"]
    # Within one surface (a ranking only) against all surfaces pooled (one scale)
    prompt_auc = {
        _n: {
            _s: float(roc_auc_score(y_tile[is_test & (surf == _s)], _p[is_test & (surf == _s)]))
            for _s in SURFACE_NOUN
        }
        for _n, _p in zs_probs.items()
    }
    prompt_median_p = {
        _n: {_s: float(np.median(_p[is_test & (surf == _s)])) for _s in SURFACE_NOUN}
        for _n, _p in zs_probs.items()
    }
    for _n in PROMPTS:
        print(f"{_n:9s} AUC by surface", {_s: round(_v, 4) for _s, _v in prompt_auc[_n].items()},
              "median p", {_s: round(_v, 4) for _s, _v in prompt_median_p[_n].items()})  # fmt: skip
    _want_auc = {
        "concrete": (0.7425237857068177, 0.7755486214311264, 0.7682737597590152),
        "surface": (0.7834076033257132, 0.7910780910301222, 0.7980077410728749),
        "crack": (0.7722558014895443, 0.7870923912234854, 0.81066431205403),
        "close-up": (0.7491605183169486, 0.7849632526452798, 0.7540211321071553),
    }
    _want_med = {
        "concrete": (0.05651376582682133, 0.05042310990393162, 0.16134191304445267),
        "surface": (0.5893510580062866, 0.04382927343249321, 0.043565673753619194),
        "crack": (0.16816718131303787, 0.24661484360694885, 0.3488715589046478),
        "close-up": (0.2529967129230499, 0.27074940502643585, 0.3020884245634079),
    }
    for _n in PROMPTS:
        for _s, _a, _m in zip(SURFACE_NOUN, _want_auc[_n], _want_med[_n], strict=True):
            assert (
                abs(prompt_auc[_n][_s] - _a) < CLIP_TOL
                and abs(prompt_median_p[_n][_s] - _m) < CLIP_TOL
            )
    _sa, _ca, _sm = prompt_auc["surface"], prompt_auc["close-up"], prompt_median_p["surface"]
    # The notes: naming the surface ranks a little better within each surface than the close-up
    # pair (AUC 0.78, 0.79, 0.80 against 0.75, 0.78, 0.75) ...
    assert all(_sa[_s] > _ca[_s] for _s in SURFACE_NOUN)
    assert all(
        shown(_sa[_s], _t) for _s, _t in zip(SURFACE_NOUN, ("0.78", "0.79", "0.80"), strict=True)
    )
    assert all(
        shown(_ca[_s], _t) for _s, _t in zip(SURFACE_NOUN, ("0.75", "0.78", "0.75"), strict=True)
    )
    # ... and lifts every deck tile instead: median P(cracked) 0.59 on decks, 0.04 on walls and
    # pavements; once the surfaces are pooled, clean decks outrank cracked walls
    assert all(
        shown(_sm[_s], _t) for _s, _t in zip(SURFACE_NOUN, ("0.59", "0.04", "0.04"), strict=True)
    )
    _p = zs_probs["surface"]
    _clean_deck = _p[is_test & (surf == "deck") & (y_tile == 0)]
    _cracked_wall = _p[is_test & (surf == "wall") & (y_tile == 1)]
    _deck_above = float((_clean_deck > np.median(_cracked_wall)).mean())
    _pairwise = float((_clean_deck[:, None] > _cracked_wall[None, :]).mean())
    print(f"clean deck tiles above the median cracked wall: {100 * _deck_above:.1f}%; "
          f"a clean deck tile outranks a cracked wall tile in {100 * _pairwise:.1f}% of pairs")  # fmt: skip
    assert _deck_above > 0.9 and _pairwise > 0.5
    _rows = "\n".join(
        f"    | {PROMPTS[_n][0]} / {PROMPTS[_n][1]} | {zs['test_ap'][_n]:.2f} |" for _n in _order
    )
    mo.md(f"""
    | Prompt pair (clean / cracked), test photos | Test AP |
    |:-------------------------------------------|--------:|
{_rows}

    Average precision moves from {zs["test_ap"]["surface"]:.2f} to {zs["test_ap"]["close-up"]:.2f}.
    Here the validation choice was also the test's best; it need not be. Prompt wording is a
    hyperparameter: choose it on validation photos, never on the test photos (CLIP's paper
    averages 80 prompts per class, *prompt ensembling*).

    Naming the surface ("a cracked bridge deck") does not hurt the ranking within a surface; it
    ranks a little better there: test AUC {_sa["deck"]:.2f}, {_sa["wall"]:.2f},
    {_sa["pavement"]:.2f} on decks, walls, pavements, against {_ca["deck"]:.2f},
    {_ca["wall"]:.2f}, {_ca["pavement"]:.2f} for the close-up pair. It lifts every deck tile
    instead: median $P(\\text{{cracked}})$ {_sm["deck"]:.2f} on decks, {_sm["wall"]:.2f} on walls,
    {_sm["pavement"]:.2f} on pavements. Once the surfaces are pooled, clean decks outrank cracked
    walls: {100 * _deck_above:.0f}% of clean deck tiles score above the median cracked wall. One
    threshold for all surfaces needs scores on one scale; each prompt pair is its own classifier,
    with its own offset.
    """)
    return (prompt_auc, prompt_median_p)


@app.cell
def _(SURFACE_NOUN, few_shot, mo, zs):
    _bs = zs["by_surface"]
    _rec = " / ".join(f"{100 * _bs[_s]['recall']:.0f}%" for _s in SURFACE_NOUN)
    _rev = " / ".join(f"{100 * _bs[_s]['review']:.0f}%" for _s in SURFACE_NOUN)
    _z2, _p2 = few_shot[2]["zero_shot"], few_shot[2]
    _pct = lambda _v: f"{100 * _v:.0f}"  # noqa: E731
    mo.md(f"""
    ### Reveal 2: no labels, no threshold

    | Zero-shot CLIP, test photos | Cracks flagged | Tiles to review |
    |:----------------------------|---------------:|----------------:|
    | Threshold 0.5 (no labels at all) | {_pct(zs["at_half"]["recall"])}% | {_pct(zs["at_half"]["review"])}% |
    | Threshold set on 47 labelled validation photos | {_pct(zs["at_90"]["recall"])}% | {_pct(zs["at_90"]["review"])}% |
    | ... on 2 labelled photos per surface (10 draws) | {_pct(_z2["recall"]["range"][0])}–{_pct(_z2["recall"]["range"][1])}% | {_pct(_z2["review"]["range"][0])}–{_pct(_z2["review"]["range"][1])}% |
    | By surface, the 47-photo threshold: decks / walls / pavements | {_rec} | {_rev} |

    Without labels, the only threshold is the softmax's 0.5, and it misses
    {_pct(1 - zs["at_half"]["recall"])}% of the cracks: nothing ties 0.5 to the inspector's 90%.
    Setting the threshold for 90% needs labelled photos, so zero-shot is not label-free once a
    target is set. With all 47 validation photos it sends {_pct(zs["at_90"]["review"])}% of tiles
    to review; with two labelled photos per surface (the draws of the few-label probes, below),
    what it flags depends on which two: {_pct(_z2["recall"]["range"][0])}% to
    {_pct(_z2["recall"]["range"][1])}% of the cracks over ten draws (median
    {_pct(_z2["recall"]["median"])}%; a median {_pct(_z2["review"]["median"])}% of tiles to
    review), against a median {_pct(_p2["review"]["median"])}% for a probe fitted on two photos
    per surface with the same threshold photos. A zero-shot score ranks tiles; it does not say
    how many to look at.
    """)
    return


@app.cell
def _(SURFACE_NOUN, few_shot, shown, zs):
    _bs = zs["by_surface"]
    assert all(
        shown(100 * _bs[_s]["recall"], _t)
        for _s, _t in zip(SURFACE_NOUN, ("82", "86", "85"), strict=True)
    )
    assert all(
        shown(100 * _bs[_s]["review"], _t)
        for _s, _t in zip(SURFACE_NOUN, ("62", "62", "64"), strict=True)
    )
    assert 0.4 < 1 - zs["at_half"]["recall"] < 0.5  # "misses nearly half the cracks"
    # The table and notes: on 2 photos per surface, 71-92% of cracks (median 86%), 39-78% of
    # tiles (median 65%); the probe with the same threshold photos, a median 52%
    _z2 = few_shot[2]["zero_shot"]
    _texts = ((_z2["recall"]["range"], ("71", "92")), (_z2["review"]["range"], ("39", "78")))
    assert all(shown(100 * _v, _t) for _r, _ts in _texts for _v, _t in zip(_r, _ts, strict=True))
    assert shown(100 * _z2["recall"]["median"], "86") and shown(100 * _z2["review"]["median"], "65")
    assert shown(100 * few_shot[2]["review"]["median"], "52")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A few labels beat no labels

    Probes (Lecture 15's logistic regression) on CLIP's image embeddings, fitted as an owner
    with $k$ labelled photos per surface would fit them, $k$ = 1, 2, 4, 8, 16. For each $k$, ten
    random draws: $k$ training photos of each surface to fit the probe (seed
    $(18, k, \text{draw})$), and $k$ validation photos of each surface to set the 90% threshold
    (seed $(18, k, \text{draw}, 1)$). $C$ is fixed at 0.01, since $k$ photos leave no labels to
    tune it on. Two choices borrow labels this owner would not have: 0.01 is the $C$ the probe on
    all photos chose on all 47 validation photos (§5's last table), and zero-shot's prompt pair
    was chosen on the same 47. The test photos are scored once. Zero-shot CLIP gets its threshold from the same $k$ photos, like for like. A photo is
    about 240 tiles, so one photo per surface is several hundred labelled tiles, some of them
    cracked.
    """)
    return


@app.cell
def _(
    EXACT,
    FEW_C,
    FEW_C_CHECK,
    LogisticRegression,
    SEED,
    SHOTS,
    SHOT_DRAWS,
    SURFACE_NOUN,
    StandardScaler,
    average_precision_score,
    cfeat,
    is_test,
    is_train,
    is_val,
    np,
    scores,
    surf,
    threshold_for,
    tiles,
    y_tile,
    zs_probs,
    zs,
):
    _photos = tiles["photo"].to_numpy()

    def draw_photos(among, k, rng):
        """The tiles of k photos of each surface, drawn at random from the photos of ``among``."""
        pick = []
        for s in SURFACE_NOUN:
            pool = np.unique(_photos[among & (surf == s)])
            pick += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        return among & np.isin(_photos, pick)

    def fixed_probe(feats, y, use, c):
        """Lecture 15's probe at a fixed C, fitted on the tiles ``use``: P(cracked) of every tile."""
        scaler = StandardScaler().fit(feats[use])
        m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
        m.fit(scaler.transform(feats[use]), y[use])
        return m.predict_proba(scaler.transform(feats))[:, 1]

    def spread(v):
        """Every draw's value, their median, and their range."""
        return {"draws": v, "median": float(np.median(v)), "range": [min(v), max(v)]}

    _zero_p = zs_probs[zs["best"]]  # zero-shot P(cracked), the pair chosen on validation
    few_shot = {}
    for _k in SHOTS:
        _rows = {_key: [] for _key in ("ap", "recall", "review", "zs_recall", "zs_review",
                                       "tiles", "cracked_tiles", "val_tiles")}  # fmt: skip
        _check = {_c: [] for _c in FEW_C_CHECK}
        for _draw in SHOT_DRAWS:
            # k training photos per surface to fit; k validation photos per surface for the
            # threshold: the only labels this owner has
            _use = draw_photos(is_train, _k, np.random.default_rng((SEED, _k, _draw)))
            _use_val = draw_photos(is_val, _k, np.random.default_rng((SEED, _k, _draw, 1)))
            assert 0 < y_tile[_use].sum() < _use.sum() and y_tile[_use_val].sum() > 0
            _p = fixed_probe(cfeat, y_tile, _use, FEW_C)
            _s = scores(_p[is_test], y_tile[is_test], threshold_for(_p[_use_val], y_tile[_use_val]))
            _z = scores(_zero_p[is_test], y_tile[is_test],
                        threshold_for(_zero_p[_use_val], y_tile[_use_val]))  # fmt: skip
            for _key, _v in (("ap", _s["ap"]), ("recall", _s["recall"]), ("review", _s["review"]),
                             ("zs_recall", _z["recall"]), ("zs_review", _z["review"]),
                             ("tiles", int(_use.sum())), ("cracked_tiles", int(y_tile[_use].sum())),
                             ("val_tiles", int(_use_val.sum()))):  # fmt: skip
                _rows[_key].append(_v)
            for _c in FEW_C_CHECK:  # C's neighbours on the grid: does the fixed C decide it?
                _pc = fixed_probe(cfeat, y_tile, _use, _c)
                _check[_c].append(float(average_precision_score(y_tile[is_test], _pc[is_test])))
        few_shot[_k] = {
            **{_m: spread(_rows[_m]) for _m in ("ap", "recall", "review")},
            "zero_shot": {
                "recall": spread(_rows["zs_recall"]),
                "review": spread(_rows["zs_review"]),
            },
            "tiles": _rows["tiles"],
            "cracked_tiles": _rows["cracked_tiles"],
            "val_tiles": _rows["val_tiles"],
            "ap_median_at_C": {str(_c): float(np.median(_v)) for _c, _v in _check.items()},
        }
        _f = few_shot[_k]
        print(
            f"k = {_k:2d}: test AP median {_f['ap']['median']:.3f} (range {_f['ap']['range'][0]:.3f}"
            f"-{_f['ap']['range'][1]:.3f}); cracks flagged {100 * _f['recall']['median']:.0f}% "
            f"({100 * _f['recall']['range'][0]:.0f}-{100 * _f['recall']['range'][1]:.0f}%); "
            f"review {100 * _f['review']['median']:.0f}% ({100 * _f['review']['range'][0]:.0f}-"
            f"{100 * _f['review']['range'][1]:.0f}%)"
        )
    # The slides' run, exactly: everything here runs on the CPU from the cached embeddings
    _ref = (  # few: from numbers.json
        {'1': {'ap': {'draws': [0.6482310807259785,
                                0.5930029052609871,
                                0.6511929803438764,
                                0.6186084428485974,
                                0.6012335411418624,
                                0.6485408637494385,
                                0.641732410153013,
                                0.6651337973566316,
                                0.6093593315333654,
                                0.557062672377311]},
               'recall': {'draws': [0.8423339540657977,
                                    0.7281191806331471,
                                    0.8094351334574799,
                                    0.8237119801365611,
                                    0.952203600248293,
                                    0.8870266914959652,
                                    0.8702669149596524,
                                    0.8144009931719429,
                                    0.9553072625698324,
                                    0.8758535071384234]},
               'review': {'draws': [0.4400428265524625,
                                    0.3487687366167024,
                                    0.4707351891506067,
                                    0.471627408993576,
                                    0.7805139186295503,
                                    0.5691470378301213,
                                    0.5161491791577445,
                                    0.44325481798715205,
                                    0.7615096359743041,
                                    0.5726266952177016]},
               'zero_shot': {'recall': {'draws': [0.8150217256362507,
                                                  0.8559900682805711,
                                                  0.8199875853507138,
                                                  0.7312228429546865,
                                                  0.9795158286778398,
                                                  0.8094351334574799,
                                                  0.7858472998137802,
                                                  0.8634388578522657,
                                                  0.8752327746741154,
                                                  0.8292985723153321]},
                             'review': {'draws': [0.5752141327623126,
                                                  0.6506067094932191,
                                                  0.5792291220556746,
                                                  0.4279978586723769,
                                                  0.932994289793005,
                                                  0.5668272662384012,
                                                  0.5263204853675946,
                                                  0.6625624553890078,
                                                  0.6917380442541042,
                                                  0.6025160599571735]}},
               'tiles': [738, 738, 738, 738, 738, 738, 737, 738, 738, 738],
               'cracked_tiles': [93, 135, 132, 154, 212, 112, 82, 92, 116, 63],
               'val_tiles': [738, 738, 738, 738, 738, 738, 738, 738, 738, 738],
               'ap_median_at_C': {'0.003': 0.6228918170097942, '0.03': 0.6196359968992886}},
         '2': {'ap': {'draws': [0.6669803498323823,
                                0.6947182362539238,
                                0.6793025721815654,
                                0.6930464846957,
                                0.6819010150348265,
                                0.694082126172939,
                                0.6750748573385793,
                                0.6926157020799798,
                                0.6668566295750732,
                                0.6833663774900925]},
               'recall': {'draws': [0.9515828677839852,
                                    0.7932960893854749,
                                    0.9031657355679702,
                                    0.8429546865301055,
                                    0.8659217877094972,
                                    0.792675356921167,
                                    0.8926132836747361,
                                    0.9428926132836747,
                                    0.8944754810676598,
                                    0.8721291123525761]},
               'review': {'draws': [0.7408993576017131,
                                    0.3193254817987152,
                                    0.555496074232691,
                                    0.4469129193433262,
                                    0.49009635974304067,
                                    0.32673090649536046,
                                    0.5246252676659529,
                                    0.6722876516773733,
                                    0.539971448965025,
                                    0.5073162027123483]},
               'zero_shot': {'recall': {'draws': [0.8746120422098076,
                                                  0.7082557417752948,
                                                  0.8212290502793296,
                                                  0.8559900682805711,
                                                  0.8280571073867163,
                                                  0.8249534450651769,
                                                  0.8634388578522657,
                                                  0.9186840471756673,
                                                  0.877715704531347,
                                                  0.8609559279950342]},
                             'review': {'draws': [0.689329050678087,
                                                  0.3945396145610278,
                                                  0.5836009992862241,
                                                  0.6506067094932191,
                                                  0.5998394004282656,
                                                  0.5937723054960742,
                                                  0.6654175588865097,
                                                  0.7787294789436117,
                                                  0.6964668094218416,
                                                  0.6579229122055674]}},
               'tiles': [1476, 1476, 1476, 1476, 1476, 1476, 1476, 1476, 1476, 1477],
               'cracked_tiles': [193, 199, 214, 235, 219, 291, 191, 190, 271, 243],
               'val_tiles': [1476, 1476, 1476, 1476, 1476, 1476, 1476, 1476, 1476, 1476],
               'ap_median_at_C': {'0.003': 0.685144279719949, '0.03': 0.6702253806946483}},
         '4': {'ap': {'draws': [0.7309112512943376,
                                0.7249974613845274,
                                0.7241219291510124,
                                0.7122585504692516,
                                0.6964242937727259,
                                0.6998314127024925,
                                0.7306502912158289,
                                0.7011612832594747,
                                0.70804882539376,
                                0.67643180301037]},
               'recall': {'draws': [0.9013035381750466,
                                    0.8292985723153321,
                                    0.8684047175667288,
                                    0.925512104283054,
                                    0.845437616387337,
                                    0.8653010552451893,
                                    0.8653010552451893,
                                    0.8286778398510242,
                                    0.8665425201738051,
                                    0.8609559279950342]},
               'review': {'draws': [0.5149000713775874,
                                    0.36143825838686655,
                                    0.45458600999286225,
                                    0.5922555317630264,
                                    0.43486795146324053,
                                    0.4629728765167737,
                                    0.39650249821556033,
                                    0.39569950035688795,
                                    0.4678800856531049,
                                    0.4822448251249108]},
               'zero_shot': {'recall': {'draws': [0.8249534450651769,
                                                  0.8280571073867163,
                                                  0.8292985723153321,
                                                  0.8286778398510242,
                                                  0.8367473618870267,
                                                  0.8696461824953445,
                                                  0.8466790813159528,
                                                  0.8075729360645562,
                                                  0.8261949099937926,
                                                  0.8311607697082557]},
                             'review': {'draws': [0.5937723054960742,
                                                  0.5998394004282656,
                                                  0.6025160599571735,
                                                  0.6022483940042827,
                                                  0.6115274803711634,
                                                  0.6784439685938616,
                                                  0.6340114204139901,
                                                  0.563169164882227,
                                                  0.5971627408993576,
                                                  0.6032298358315489]}},
               'tiles': [2951, 2952, 2952, 2952, 2952, 2951, 2952, 2960, 2952, 2952],
               'cracked_tiles': [362, 412, 445, 344, 440, 614, 415, 440, 449, 368],
               'val_tiles': [2952, 2952, 2952, 2952, 2952, 2952, 2952, 2952, 2952, 2952],
               'ap_median_at_C': {'0.003': 0.7108236781923278, '0.03': 0.6984013397617879}},
         '8': {'ap': {'draws': [0.7516470594275225,
                                0.7116897514734148,
                                0.7312538263256484,
                                0.684969881035506,
                                0.7257892313382849,
                                0.7472553390855375,
                                0.7318344022295142,
                                0.7033137982886096,
                                0.7252599300777026,
                                0.7314976017825967]},
               'recall': {'draws': [0.9273743016759777,
                                    0.8677839851024208,
                                    0.8603351955307262,
                                    0.8274363749224084,
                                    0.9050279329608939,
                                    0.9267535692116697,
                                    0.8814400993171942,
                                    0.8739913097454997,
                                    0.8559900682805711,
                                    0.8944754810676598]},
               'review': {'draws': [0.5640613847251963,
                                    0.47341184867951464,
                                    0.4121163454675232,
                                    0.4190756602426838,
                                    0.5140078515346181,
                                    0.545146324054247,
                                    0.4416488222698073,
                                    0.46850463954318344,
                                    0.4051570306923626,
                                    0.46770164168451106]},
               'zero_shot': {'recall': {'draws': [0.8504034761018001,
                                                  0.8472998137802608,
                                                  0.8268156424581006,
                                                  0.8361266294227188,
                                                  0.8479205462445686,
                                                  0.8367473618870267,
                                                  0.8547486033519553,
                                                  0.8472998137802608,
                                                  0.8609559279950342,
                                                  0.8634388578522657]},
                             'review': {'draws': [0.6426659528907923,
                                                  0.6343683083511777,
                                                  0.5980549607423269,
                                                  0.6097430406852249,
                                                  0.6369557458957887,
                                                  0.6115274803711634,
                                                  0.6478408279800143,
                                                  0.6341898643825838,
                                                  0.657209136331192,
                                                  0.6625624553890078]}},
               'tiles': [5909, 5907, 5903, 5904, 5908, 5904, 5912, 5904, 5908, 5904],
               'cracked_tiles': [749, 1149, 916, 1072, 940, 800, 910, 1028, 929, 872],
               'val_tiles': [5904, 5904, 5904, 5904, 5904, 5904, 5904, 5904, 5904, 5904],
               'ap_median_at_C': {'0.003': 0.726732038706901, '0.03': 0.7185691719330276}},
         '16': {'ap': {'draws': [0.7560398779240475,
                                 0.7562506880124541,
                                 0.7642219243163454,
                                 0.76379924424206,
                                 0.7631390465727758,
                                 0.7610926376123368,
                                 0.7650719657770234,
                                 0.7581773732859654,
                                 0.7597825163711974,
                                 0.7585217021430054]},
                'recall': {'draws': [0.8938547486033519,
                                     0.8752327746741154,
                                     0.8746120422098076,
                                     0.8783364369956549,
                                     0.8727498448168839,
                                     0.888268156424581,
                                     0.8621973929236499,
                                     0.8640595903165735,
                                     0.8721291123525761,
                                     0.8733705772811918]},
                'review': {'draws': [0.4562812276945039,
                                     0.40444325481798715,
                                     0.4034618129907209,
                                     0.4074768022840828,
                                     0.3924875089221984,
                                     0.44022127052105636,
                                     0.38900785153461814,
                                     0.39364739471805854,
                                     0.3875802997858672,
                                     0.3991791577444682]},
                'zero_shot': {'recall': {'draws': [0.8497827436374923,
                                                   0.8466790813159528,
                                                   0.8466790813159528,
                                                   0.8479205462445686,
                                                   0.846058348851645,
                                                   0.8472998137802608,
                                                   0.8448168839230292,
                                                   0.8466790813159528,
                                                   0.8497827436374923,
                                                   0.8504034761018001]},
                              'review': {'draws': [0.6415952890792291,
                                                   0.6340114204139901,
                                                   0.6340114204139901,
                                                   0.6349928622412563,
                                                   0.632762312633833,
                                                   0.6341898643825838,
                                                   0.6293718772305497,
                                                   0.6340114204139901,
                                                   0.6415952890792291,
                                                   0.6426659528907923]}},
                'tiles': [11813, 11817, 11812, 11815, 11808, 11812, 11813, 11809, 11808, 11811],
                'cracked_tiles': [1770, 1956, 1641, 1777, 1923, 1860, 1836, 2128, 2060, 1673],
                'val_tiles': [10296, 10296, 10296, 10296, 10296, 10296, 10296, 10296, 10296, 10296],
                'ap_median_at_C': {'0.003': 0.7606496839456707, '0.03': 0.7556451202908228}}}
    )  # fmt: skip
    for _k in SHOTS:
        _got, _want = few_shot[_k], _ref[str(_k)]
        for _m in ("ap", "recall", "review"):
            assert np.abs(np.array(_got[_m]["draws"]) - _want[_m]["draws"]).max() < EXACT, (_k, _m)
        for _m in ("recall", "review"):
            assert np.abs(np.array(_got["zero_shot"][_m]["draws"])
                          - _want["zero_shot"][_m]["draws"]).max() < 1e-3, (_k, _m)  # fmt: skip
        for _m in ("tiles", "cracked_tiles", "val_tiles"):
            assert _got[_m] == _want[_m], (_k, _m)
        for _c, _v in _want["ap_median_at_C"].items():
            assert abs(_got["ap_median_at_C"][_c] - _v) < EXACT, (_k, _c)
    return (few_shot,)


@app.cell
def _(PALETTE, SHOTS, SHOT_DRAWS, few_shot, full_probes, plt, shown, train_photos, zs):
    _zero = zs["at_half"]["ap"]
    _all = full_probes["CLIP probe, all photos"]["ap"]
    _vit = full_probes["ViT-B/16 probe (Lecture 17)"]["ap"]
    _r18 = full_probes["ResNet-18 probe (Lecture 15)"]["ap"]
    _med = {_k: few_shot[_k]["ap"]["median"] for _k in SHOTS}
    # The alt text and notes: the probe's median climbs from 0.63 (one photo per surface) to 0.76
    # (16) and 0.78 (all: 32, 43, 62 photos); zero-shot, 0.52, is below every draw (the lowest,
    # 0.56); two per surface, 0.68 (0.67-0.69), close to Lecture 15's ResNet-18 on all 137
    # (0.67); ViT-B/16 0.77; at C = 0.003 or 0.03 the two-photo median is 0.69 or 0.67
    assert all(shown(_med[_k], _t)
               for _k, _t in zip(SHOTS, ("0.63", "0.68", "0.71", "0.73", "0.76"), strict=True))  # fmt: skip
    assert (
        shown(_all, "0.78") and shown(_vit, "0.77") and shown(_r18, "0.67") and shown(_zero, "0.52")
    )
    assert sum(train_photos.values()) == 137 and len(SHOT_DRAWS) == 10
    _lowest = min(few_shot[_k]["ap"]["range"][0] for _k in SHOTS)
    assert _zero < _lowest and shown(_lowest, "0.56")
    _two = few_shot[2]["ap"]
    assert shown(_two["range"][0], "0.67") and shown(_two["range"][1], "0.69")
    assert 0 < _two["median"] - _r18 < 0.03  # "close" (one split, one probe)
    assert shown(few_shot[2]["ap_median_at_C"]["0.003"], "0.69")
    assert shown(few_shot[2]["ap_median_at_C"]["0.03"], "0.67")
    _fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _ks = list(SHOTS)
    _ax.fill_between(_ks, [few_shot[_k]["ap"]["range"][0] for _k in _ks],
                     [few_shot[_k]["ap"]["range"][1] for _k in _ks], color=PALETTE[0], alpha=0.15)  # fmt: skip
    _ax.plot(_ks, [_med[_k] for _k in _ks], color=PALETTE[0], marker="o", lw=2,
             label=f"CLIP probe, k photos per surface ({len(SHOT_DRAWS)} draws; band: range)")  # fmt: skip
    _ax.axhline(_zero, color=PALETTE[1], ls="--", lw=2, label="CLIP zero-shot: no labels")
    for (_name, _ap), _mk, _col in zip(
        ((("CLIP probe, all photos"), _all), ("ViT-B/16 probe (Lecture 17)", _vit),
         ("ResNet-18 probe (Lecture 15)", _r18)),
        ("s", "^", "D"), (PALETTE[2], PALETTE[3], "#1f3a5f"), strict=True,
    ):  # fmt: skip
        _ax.plot([45], [_ap], marker=_mk, ms=10, ls="", color=_col, label=_name)
    _ax.set_xscale("log")
    _ax.set_xticks([*_ks, 45], [*map(str, _ks), "all"])
    _ax.set_xlabel("k labelled training photos per surface (and k more for the threshold)")
    _ax.set_ylabel("test average precision")
    _ax.legend(fontsize=9, frameon=False, loc="lower right")
    _fig
    return


@app.cell
def _(few_shot, mo):
    _two, _one = few_shot[2], few_shot[1]
    mo.md(f"""
    One labelled photo per surface beats zero-shot in every draw (the lowest AP of any draw,
    {_one["ap"]["range"][0]:.2f}, against 0.52). Two photos per surface (median AP
    {_two["ap"]["median"]:.2f}, {_two["ap"]["range"][0]:.2f} to {_two["ap"]["range"][1]:.2f} over
    the draws) come close to Lecture 15's ResNet-18 probe trained on all 137 photos (0.67): one
    split and one probe, so "close", not "better". Checked after the fact on the test photos,
    the fixed $C$ does not decide it: at 0.003 or 0.03 the two-photo median is {_two["ap_median_at_C"]["0.003"]:.2f} or
    {_two["ap_median_at_C"]["0.03"]:.2f}. A "photo" is many labelled tiles: one photo per surface
    was {_one["tiles"][0]:,} tiles in the first draw ({_one["cracked_tiles"][0]} cracked;
    {min(_one["cracked_tiles"])} to {max(_one["cracked_tiles"])} over the ten draws). On everyday
    photos, CLIP's paper finds zero-shot matching a probe on CLIP's own features at about four
    labelled examples per class (Radford et al., 2021); on cracks, one photo per surface already
    wins.

    Average precision is a ranking: it does not depend on the threshold. What the inspector gets
    does, and a threshold set on two photos per surface depends on which two (§7).
    """)
    return


@app.cell
def _(
    EXACT,
    KNN_LADDER,
    KNN_START,
    KNeighborsClassifier,
    average_precision_score,
    cfeat,
    close,
    is_test,
    is_train,
    is_val,
    np,
    scores,
    threshold_for,
    y_tile,
):
    _z = cfeat / np.linalg.norm(cfeat, axis=1, keepdims=True)

    def knn_scores(k):
        """Lecture 2's k-NN on CLIP's unit embeddings (cosine distance), votes weighted by
        1 / distance: P(cracked) of every tile."""
        _knn = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute",
                                    metric="cosine").fit(_z[is_train], y_tile[is_train])  # fmt: skip
        return _knn.predict_proba(_z)[:, 1]

    # k chosen on validation by a bracketing search: start from KNN_START, and while the best k
    # is at an edge, add the next k of KNN_LADDER past it; the kept k must be interior
    _val_ap, _tried = {}, list(KNN_START)
    while True:
        for _k in _tried:
            if _k not in _val_ap:
                _val_ap[_k] = float(average_precision_score(y_tile[is_val], knn_scores(_k)[is_val]))
        _best, _ks = max(_val_ap, key=_val_ap.get), sorted(_val_ap)
        if _ks[0] < _best < _ks[-1]:
            break
        _nxt = KNN_LADDER.index(_best) + (-1 if _best == _ks[0] else 1)
        assert 0 <= _nxt < len(KNN_LADDER), "the best k is at the end of the ladder"
        _tried.append(KNN_LADDER[_nxt])
    _p = knn_scores(_best)
    _t = threshold_for(_p[is_val], y_tile[is_val])
    knn_res = scores(_p[is_test], y_tile[is_test], _t) | {
        "k": _best,
        "val_ap": _val_ap,
        "t": _t,
        "ties": int(np.sum(_p[is_test] == _t)),  # test tiles scored exactly at the threshold
    }
    print(f"validation AP by k: { {_k: round(_v, 4) for _k, _v in sorted(_val_ap.items())} }")
    print(f"k-NN, k = {_best}: threshold {_t:.4f}, {knn_res['ties']} test tiles exactly at it; "
          "test: " + ", ".join(f"{_m} {knn_res[_m]:.4f}" for _m in ("auc", "ap", "recall", "review")))  # fmt: skip
    _ref = (  # knn: from numbers.json
        {'k': 50,
         'val_ap': {'5': 0.6569547636224699,
                    '10': 0.6985851730607031,
                    '25': 0.7162996797629333,
                    '50': 0.7171032227276467,
                    '100': 0.7103506837181294},
         'auc': 0.8357966328578653,
         'ap': 0.6504355499151907,
         'recall': 0.8758535071384234,
         'review': 0.5462169878658102,
         'acc': 0.9001605995717344,
         'ties_at_threshold': 0}
    )  # fmt: skip
    assert _best == _ref["k"] and sorted(_val_ap) == sorted(map(int, _ref["val_ap"]))
    assert all(abs(_val_ap[int(_k)] - _v) < EXACT for _k, _v in _ref["val_ap"].items())
    close(knn_res, {_m: _ref[_m] for _m in ("auc", "ap", "recall", "review", "acc")}, EXACT)
    assert knn_res["ties"] == _ref["ties_at_threshold"]
    return (knn_res,)


@app.cell
def _(few_shot, full_probes, knn_res, mo, shown, zs):
    _clip = full_probes["CLIP probe, all photos"]
    _vit = full_probes["ViT-B/16 probe (Lecture 17)"]
    _r18 = full_probes["ResNet-18 probe (Lecture 15)"]
    _two = few_shot[2]
    # The table: 0.52 / 22%; k = 50: 0.65 / 55%; 0.78 / 39%; 2 + 2 photos: 0.68 / 52% (median).
    # The notes: k-NN flags 88%, no tile exactly at the threshold; the probe on all photos
    # reviews 39% of tiles at 88% of cracks, close to ViT-B/16's 41% (one split)
    assert (
        knn_res["k"] == 50 and shown(knn_res["ap"], "0.65") and shown(100 * knn_res["review"], "55")
    )
    assert shown(100 * knn_res["recall"], "88") and knn_res["ties"] == 0
    assert shown(_two["ap"]["median"], "0.68") and shown(100 * _two["review"]["median"], "52")
    assert shown(_clip["ap"], "0.78") and shown(100 * _clip["review"], "39")
    assert shown(100 * _clip["recall"], "88") and shown(100 * _vit["review"], "41")
    assert 0 < _vit["review"] - _clip["review"] < 0.03  # close: a gap of 2 points on one split
    mo.md(f"""
    ### Three ways to adapt, on the test photos

    | Adaptation of CLIP | Labels | AP | Tiles to review |
    |:-------------------|:-------|---:|----------------:|
    | Zero-shot, threshold 0.5 | none | {zs["at_half"]["ap"]:.2f} | {100 * zs["at_half"]["review"]:.0f}% |
    | Nearest neighbours (k = {knn_res["k"]}) | all training photos | {knn_res["ap"]:.2f} | {100 * knn_res["review"]:.0f}% |
    | Linear probe | all training photos | {_clip["ap"]:.2f} | {100 * _clip["review"]:.0f}% |
    | Linear probe | 2 + 2 photos per surface | {_two["ap"]["median"]:.2f} | {100 * _two["review"]["median"]:.0f}% |

    Review shares at the 90% target, the threshold set on the validation photos (all 47, or two
    per surface for the last row: the median of ten draws), except zero-shot at 0.5, which flags
    only {100 * zs["at_half"]["recall"]:.0f}% of cracks. Nearest neighbours (Lecture 2, cosine
    distance on CLIP's embeddings, each neighbour's vote weighted by one over its distance, $k$
    chosen on validation) flags {100 * knn_res["recall"]:.0f}% and asks for
    {100 * knn_res["review"]:.0f}% of tiles; with weighted votes, {knn_res["ties"]} test tiles
    score exactly at the threshold. (With one vote per neighbour, 25 neighbours give only 26
    possible scores for 11,208 test tiles, so whole blocks of tiles tie at any threshold.) The
    fourth way, fine-tuning CLIP's encoder, is not run here: it needs a
    GPU budget, and the model loses its general use. The linear probe on all photos,
    {100 * _clip["review"]:.0f}% of tiles at {100 * _clip["recall"]:.0f}% of cracks, is close to
    Lecture 17's ViT-B/16 probe ({100 * _vit["review"]:.0f}%) on this one split; ResNet-18,
    Lecture 15: {100 * _r18["review"]:.0f}%.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. SAM: masks from clicks

    ### Segment anything, from a click

    Segment Anything (Kirillov et al., 2023; trained on SA-1B, 11 million images and 1.1 billion
    masks) and SAM 2 (Ravi et al., 2024): an image encoder run once per image, a prompt encoder
    for clicks or boxes, and a light decoder that returns three candidate masks (a part, the
    object, the whole) with its own estimate of each one's IoU. Used here: SAM 2.1 Hiera
    base-plus (Apache-2.0). Because the image encoder runs once, the code below encodes each
    image once and asks it for 1, 3, and 5 clicks.

    | Lecture 16 | SAM |
    |:-----------|:----|
    | U-Net trained on 3DCrack | no training on cracks |
    | a class per pixel | the object at the click |
    | IoU 0.70 on the test images (Lecture 16's notebook) | ? |

    Can a click on a crack replace Lecture 16's trained U-Net? The test: 3DCrack's test images
    that hold a crack, clicks evenly spaced along the labelled centreline, SAM's own choice of its
    three masks (the highest predicted IoU), scored by IoU against the label. Then the same on
    SpaceNet 8's buildings, one click at each building's most interior pixel.
    """)
    return


@app.cell
def _(
    DEVICE,
    SAM,
    SAM_REVISION,
    Sam2Model,
    Sam2Processor,
    distance_transform_edt,
    label,
    maximum_position,
    np,
    torch,
):
    sam_model = Sam2Model.from_pretrained(SAM, revision=SAM_REVISION).eval().to(DEVICE)
    sam_proc = Sam2Processor.from_pretrained(SAM, revision=SAM_REVISION)

    def sam_embed(rgb):
        """SAM's image encoder, run once per image: its feature maps at three scales."""
        px = sam_proc(images=rgb, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            return sam_model.get_image_embeddings(px.to(DEVICE))

    def sam_masks(rgb, points, emb):
        """SAM's masks for one image and several prompts, each a list of positive clicks [x, y].

        Returns:
            Boolean masks (prompts, 3, h, w), SAM's three candidates per prompt, and SAM's
            predicted IoU of each (prompts, 3).
        """
        labels = [[[1] * len(p) for p in points]]
        enc = sam_proc(images=rgb, input_points=[points], input_labels=labels, return_tensors="pt")
        with torch.no_grad():
            out = sam_model(
                image_embeddings=emb,
                input_points=enc["input_points"].to(DEVICE),
                input_labels=enc["input_labels"].to(DEVICE),
                multimask_output=True,
            )
        masks = sam_proc.post_process_masks(out.pred_masks.cpu(), enc["original_sizes"])[0]
        return masks.numpy() > 0, out.iou_scores[0].float().cpu().numpy()

    def iou(a, b):
        """Intersection over union of two boolean masks."""
        return float((a & b).sum() / max((a | b).sum(), 1))

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
            for r, c in maximum_position(distance_transform_edt(mask), comps, lost):
                skel[int(r), int(c)] = True
        return skel

    def along(skeleton, k):
        """``k`` clicks [x, y] spread along a centreline: evenly spaced ranks, ends excluded.

        The centreline's pixels are ordered along the crack's main direction (their first
        principal axis, from the SVD).
        """
        rc = np.argwhere(skeleton).astype(float)
        axis = np.linalg.svd(rc - rc.mean(0), full_matrices=False)[2][0]
        order = rc[np.argsort((rc - rc.mean(0)) @ axis)]
        ranks = np.linspace(0, len(order) - 1, k + 2)[1:-1].round().astype(int)
        return [[int(order[r, 1]), int(order[r, 0])] for r in ranks]

    return along, iou, sam_embed, sam_masks, thin


@app.cell
def _(CRACK3D, H3D, Image, MASK_CUT, W3D, np, require):
    _d = require(CRACK3D / "Test", "run `bash scripts/fetch_3dcrack.sh`")
    _stems = sorted(_p.stem for _p in (_d / "img").glob("*.png"))
    crack_x = np.stack([np.asarray(Image.open(_d / "img" / f"{_s}.png")) for _s in _stems])
    crack_y = np.stack(
        [np.asarray(Image.open(_d / "anno" / f"{_s}.png")) > MASK_CUT for _s in _stems]
    )
    assert crack_x.shape == crack_y.shape == (250, H3D, W3D), (crack_x.shape, crack_y.shape)
    crack_free = int((~crack_y.reshape(len(crack_y), -1).any(axis=1)).sum())
    print(f"3DCrack test: {len(_stems)} range images, {crack_free} without a crack")
    assert crack_free == 53
    return crack_free, crack_x, crack_y


@app.cell
def _(CLICKS, CRACK_EVERY, along, crack_x, crack_y, iou, np, sam_embed, sam_masks, thin, time):
    _t0 = time.perf_counter()
    _iou = {_k: [] for _k in CLICKS}
    _oracle = {_k: [] for _k in CLICKS}
    _shown = []  # (image, clicks, SAM's mask, IoU) at 3 clicks
    for _i, (_img, _gt) in enumerate(zip(crack_x, crack_y, strict=True)):
        if _i % CRACK_EVERY:  # "reduced": every 4th image
            continue
        _skel = thin(_gt)
        if _skel.sum() < 2 * max(CLICKS):  # too short a centreline for 5 clicks: no crack
            continue
        _rgb = np.repeat(_img[..., None], 3, axis=2)
        _emb = sam_embed(_rgb)  # once per image; the clicks reuse it
        for _k in CLICKS:
            _pts = along(_skel, _k)
            _m, _pred = sam_masks(_rgb, [_pts], _emb)
            assert _m.shape == (1, 3, *_gt.shape) and _pred.shape == (1, 3)  # three candidates
            _pick = int(np.argmax(_pred[0]))  # SAM's own choice among its three masks
            _iou[_k].append(iou(_m[0, _pick], _gt))
            _oracle[_k].append(max(iou(_m[0, _j], _gt) for _j in range(3)))  # by the label
            if _k == 3:
                _shown.append(
                    (_i, _pts, _m[0, _pick], _iou[_k][-1], _m[0, _pick].sum() / _gt.sum())
                )
    _mid = float(np.median([_e[3] for _e in _shown]))
    sam_cracks = {
        "images": len(_iou[1]),
        "iou": _iou,
        "iou_median": {_k: float(np.median(_v)) for _k, _v in _iou.items()},
        "oracle_median": {_k: float(np.median(_v)) for _k, _v in _oracle.items()},
        "example": min(_shown, key=lambda _e: abs(_e[3] - _mid)),  # a typical image, not the best
        "area_ratio_median": float(np.median([_e[4] for _e in _shown])),
        "seconds": time.perf_counter() - _t0,
    }
    print(
        f"{sam_cracks['images']} images in {sam_cracks['seconds']:.0f} s; median IoU by clicks "
        f"{ {_k: round(_v, 4) for _k, _v in sam_cracks['iou_median'].items()} }; best of three by "
        f"the label { {_k: round(_v, 4) for _k, _v in sam_cracks['oracle_median'].items()} }; "
        f"SAM's mask / the crack's area, median at 3 clicks {sam_cracks['area_ratio_median']:.1f}"
    )
    return (sam_cracks,)


@app.cell
def _(CLICKS, CRACK_EVERY, REFERENCE_GPU, SAM_TOL, SLIDES, crack_y, sam_cracks, shown):
    _med, _orc = sam_cracks["iou_median"], sam_cracks["oracle_median"]
    _ex = sam_cracks["example"]
    # "3DCrack's test images with a crack (197)": every image with a crack has a centreline long
    # enough for five clicks
    _with_crack = sum(bool(crack_y[_i].any()) for _i in range(0, len(crack_y), CRACK_EVERY))
    assert sam_cracks["images"] == _with_crack and (_with_crack == 197 or not SLIDES)
    _want = (  # sam_cracks: from numbers.json
        {'iou_median': {'1': 0.04194260485651214,
                        '3': 0.09848484848484848,
                        '5': 0.12205691174346382},
         'iou_oracle_median': {'1': 0.17898832684824903,
                               '3': 0.2674437170684975,
                               '5': 0.24193548387096775}}
    )  # fmt: skip
    if SLIDES:
        for _k in CLICKS:
            assert abs(_med[_k] - _want["iou_median"][str(_k)]) <= SAM_TOL, (_k, _med[_k])
            assert abs(_orc[_k] - _want["iou_oracle_median"][str(_k)]) <= SAM_TOL, (_k, _orc[_k])
        print("largest difference from the slides' medians:",
              max(abs(_med[_k] - _want["iou_median"][str(_k)]) for _k in CLICKS))  # fmt: skip
        assert _med[1] < _med[3] < _med[5] < 0.15  # more clicks help, a little
    if REFERENCE_GPU:  # the notes: medians 0.04, 0.10, 0.12; the best of three only 0.27
        assert all(
            shown(_med[_k], _t) for _k, _t in zip(CLICKS, ("0.04", "0.10", "0.12"), strict=True)
        )
        assert shown(max(_orc.values()), "0.27")
        assert _ex[0] == 11 and shown(_ex[3], "0.10")  # the figure's typical image
    # Either setting: a click on a crack does not outline it
    assert max(_med.values()) < 0.2 and max(_orc.values()) < 0.4
    # The alt text: SAM's mask is "a blob that covers far more than the crack"
    assert _ex[2].sum() > 5 * crack_y[_ex[0]].sum() and sam_cracks["area_ratio_median"] > 2
    return


@app.cell
def _(
    BUILDINGS,
    ImageDraw,
    MIN_BUILDING_PX,
    SEED,
    SN8,
    SN8_SIZE,
    TEST_X,
    Image,
    csv,
    defaultdict,
    np,
    re,
    require,
):
    def wkt_rings(wkt):
        """The coordinate rings of a WKT polygon, each (n, 2) [x, y] in pixels."""
        return [np.array([[float(v) for v in p.split()] for p in s.split(",")])
                for s in re.findall(r"\(([^()]+)\)", wkt)]  # fmt: skip

    def sn8_test_tiles():
        """Lecture 16's test block of SpaceNet 8 (grid column >= 40): pre-event image, footprints.

        Building footprints are drawn in the reference file's order, each with its own index
        (a later footprint covers an earlier one), on the pre-event image's pixel grid.
        """
        d = require(SN8, "run `bash scripts/fetch_spacenet8.sh`")
        with (d / "Germany_Training_Public_reference.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        with (d / "Germany_Training_Public_label_image_mapping.csv").open(newline="") as f:
            maps = list(csv.DictReader(f))
        by = defaultdict(list)
        for r in rows:
            by[r["ImageId"]].append(r)
        out = []
        for m in maps:
            name = m["pre-event image"][:-4]
            if int(name.split("_")[2]) < TEST_X:
                continue
            pre = Image.open(d / "PRE-event" / m["pre-event image"])
            assert pre.size == (SN8_SIZE, SN8_SIZE), pre.size
            inst = Image.new("I", pre.size, 0)
            draw, n = ImageDraw.Draw(inst), 0
            for r in by[name]:
                if r["Object"] != "Building" or "EMPTY" in r["Wkt_Pix"]:
                    continue
                rings = wkt_rings(r["Wkt_Pix"])
                assert len(rings) == 1, "no holes in the Germany footprints"
                n += 1
                draw.polygon([tuple(p) for p in rings[0]], fill=n)
            out.append({"name": name.split("_", 1)[1], "pre": np.array(pre.convert("RGB")),
                        "inst": np.asarray(inst).astype(np.int32), "buildings": n})  # fmt: skip
        return out

    sn8_test = sn8_test_tiles()
    _cand = [
        (_ti, _b)
        for _ti, _t in enumerate(sn8_test)
        for _b in np.unique(_t["inst"])[1:]
        if (_t["inst"] == _b).sum() >= MIN_BUILDING_PX
    ]
    _sel = np.sort(
        np.random.default_rng(SEED).choice(
            len(_cand), size=min(BUILDINGS, len(_cand)), replace=False
        )
    )
    sn8_pick = [_cand[_i] for _i in _sel]
    sn8_candidates = len(_cand)
    print(f"{len(sn8_test)} test tiles, {sum(_t['buildings'] for _t in sn8_test)} buildings, "
          f"{sn8_candidates} of {MIN_BUILDING_PX} pixels or more; {len(sn8_pick)} sampled")  # fmt: skip
    assert len(sn8_test) == 37 and sum(_t["buildings"] for _t in sn8_test) == 897  # Lecture 16's
    assert sn8_candidates == 750 and len(sn8_pick) == 300  # "300 of 750 over 400 pixels"
    return sn8_candidates, sn8_pick, sn8_test


@app.cell
def _(
    REFERENCE_GPU,
    SAM_TOL,
    distance_transform_edt,
    iou,
    np,
    sam_embed,
    sam_masks,
    shown,
    sn8_pick,
    sn8_test,
    time,
):
    _t0 = time.perf_counter()
    _ious, _example = [], None
    _S = 3  # footprints moved by up to 3 pixels each way: are they off the imagery?
    _D = range(-_S, _S + 1)
    _shift_iou = []  # (buildings, 7, 7): IoU with the footprint moved down by dy, right by dx
    for _ti in sorted({_p[0] for _p in sn8_pick}):
        _t = sn8_test[_ti]
        _ids = [_b for _tj, _b in sn8_pick if _tj == _ti]
        _pts = []
        for _b in _ids:
            _d = distance_transform_edt(_t["inst"] == _b)  # the most interior pixel: a sure click
            _r, _c = np.unravel_index(np.argmax(_d), _d.shape)
            _pts.append([[int(_c), int(_r)]])
        _m, _pred = sam_masks(_t["pre"], _pts, sam_embed(_t["pre"]))
        assert _m.shape[:2] == (len(_ids), 3)
        for _j, _b in enumerate(_ids):
            _mj = _m[_j, int(np.argmax(_pred[_j]))]
            _gt = _t["inst"] == _b
            _ious.append(iou(_mj, _gt))
            # both masks cropped to their joint box, padded, and the footprint slid under the mask
            _rr, _cc = np.nonzero(_mj | _gt)
            _box = (slice(_rr.min(), _rr.max() + 1), slice(_cc.min(), _cc.max() + 1))
            _M, _G = np.pad(_mj[_box], 2 * _S), np.pad(_gt[_box], 2 * _S)
            _h, _w = _M.shape[0] - 2 * _S, _M.shape[1] - 2 * _S
            _Mw = _M[_S : _S + _h, _S : _S + _w]
            _shift_iou.append(
                [[iou(_Mw, _G[_S - _dy : _S - _dy + _h, _S - _dx : _S - _dx + _w]) for _dx in _D]
                 for _dy in _D]
            )  # fmt: skip
            if _example is None:
                _example = (_ti, _pts[_j][0], _mj, _t["inst"] == _b, _ious[-1])
    _grid = np.array(_shift_iou)
    assert _grid.shape == (len(_ious), 2 * _S + 1, 2 * _S + 1)
    assert np.allclose(_grid[:, _S, _S], _ious, rtol=0, atol=1e-12)  # no shift: the same IoU
    # One shift for every footprint, chosen on the even-numbered buildings, tested on the odd
    _sel = np.unravel_index(np.argmax(np.median(_grid[0::2], axis=0)), _grid.shape[1:])
    _best = np.array([np.unravel_index(np.argmax(_g), _g.shape) for _g in _grid]) - _S
    sam_buildings = {
        "buildings": len(_ious),
        "iou": _ious,
        "iou_median": float(np.median(_ious)),
        "share_above_half": float(np.mean(np.array(_ious) > 0.5)),
        "example": _example,
        "example_iou": _example[4],
        "example_mask_px": int(_example[2].sum()),
        "example_footprint_px": int(_example[3].sum()),
        "shift": (int(_sel[0]) - _S, int(_sel[1]) - _S),  # (down, right), pixels
        "held_out_median": float(np.median(_grid[1::2, _S, _S])),
        "held_out_shifted": float(np.median(_grid[1::2, _sel[0], _sel[1]])),
        "best_shift_median": tuple(float(_v) for _v in np.median(_best, axis=0)),
        "best_shift_zero": float(np.mean((_best == 0).all(axis=1))),
    }
    print(
        f"{sam_buildings['buildings']} buildings in {time.perf_counter() - _t0:.0f} s: median IoU "
        f"{sam_buildings['iou_median']:.4f}, {100 * sam_buildings['share_above_half']:.1f}% above "
        f"0.5; the figure's building: IoU {sam_buildings['example_iou']:.4f}, mask "
        f"{sam_buildings['example_mask_px']} px, footprint {sam_buildings['example_footprint_px']} px"
    )
    print(
        f"footprints moved by {sam_buildings['shift']} px (down, right), chosen on 150 buildings: "
        f"the other 150's median IoU {sam_buildings['held_out_median']:.3f} -> "
        f"{sam_buildings['held_out_shifted']:.3f}; each building's best shift, median "
        f"{sam_buildings['best_shift_median']}, zero for {100 * sam_buildings['best_shift_zero']:.0f}%"
    )
    assert sam_buildings["buildings"] == 300
    # The alt text: the figure's building, IoU 0.50 with the footprint, 1,655 pixels against 1,142
    assert sam_buildings["example_footprint_px"] == 1142  # the label alone: exact anywhere
    assert abs(sam_buildings["example_iou"] - 0.5013419216317767) <= SAM_TOL
    assert abs(sam_buildings["example_mask_px"] - 1655) <= (0 if REFERENCE_GPU else 0.05 * 1655)
    # A check of the labels: footprints drawn off the imagery in one direction would cost every
    # building alike, and one common shift would win it back. Chosen on half the buildings, the
    # best common shift barely moves the other half's median IoU (0.499 to 0.503 on the slides'
    # GPU): the misfit is building by building
    assert abs(sam_buildings["held_out_shifted"] - sam_buildings["held_out_median"]) < 0.02
    assert abs(sam_buildings["iou_median"] - 0.4954435519553991) <= SAM_TOL
    assert abs(sam_buildings["share_above_half"] - 0.49666666666666665) <= 2 * SAM_TOL
    if REFERENCE_GPU:  # the notes: median IoU 0.50, 50% above 0.5; the alt text's example 0.50
        assert shown(sam_buildings["iou_median"], "0.50") and shown(
            100 * sam_buildings["share_above_half"], "50"
        )
        assert shown(sam_buildings["example_iou"], "0.50")
    return (sam_buildings,)


@app.cell
def _(CLICKS, PALETTE, crack_x, crack_y, np, plt, sam_buildings, sam_cracks, sn8_test):
    _fig, _axes = plt.subplots(
        1, 3, figsize=(10, 3.8), width_ratios=(1, 1, 1.3), layout="constrained"
    )
    _i, _pts, _m, _iou, _ = sam_cracks["example"]
    _axes[0].imshow(crack_x[_i], cmap="gray")
    _axes[0].contour(crack_y[_i], levels=[0.5], colors=[PALETTE[2]], linewidths=0.8)
    _axes[0].imshow(np.ma.masked_where(~_m, _m), cmap="autumn", alpha=0.5)
    _axes[0].scatter([_p[0] for _p in _pts], [_p[1] for _p in _pts], marker="*", s=90,
                     color=PALETTE[1], edgecolor="white", lw=0.6)  # fmt: skip
    _axes[0].set_title(f"crack: 3 clicks, IoU {_iou:.2f}", fontsize=11)
    _ti, _pt, _mb, _gb, _ib = sam_buildings["example"]
    _r0, _c0 = max(_pt[1] - 120, 0), max(_pt[0] - 120, 0)
    _win = (slice(_r0, _r0 + 240), slice(_c0, _c0 + 240))
    _axes[1].imshow(sn8_test[_ti]["pre"][_win])
    _axes[1].contour(_gb[_win], levels=[0.5], colors=[PALETTE[2]], linewidths=1.0)
    _axes[1].imshow(np.ma.masked_where(~_mb[_win], _mb[_win]), cmap="autumn", alpha=0.45)
    _axes[1].scatter([_pt[0] - _c0], [_pt[1] - _r0], marker="*", s=90, color=PALETTE[1],
                     edgecolor="white", lw=0.6)  # fmt: skip
    _axes[1].set_title(f"building: 1 click, IoU {_ib:.2f}", fontsize=11)
    for _ax in _axes[:2]:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _data = [sam_cracks["iou"][_k] for _k in CLICKS] + [sam_buildings["iou"]]
    _labels = [f"crack, {_k}" for _k in CLICKS] + ["building, 1"]
    _axes[2].boxplot(_data, orientation="horizontal", widths=0.5, showfliers=False)
    _axes[2].set_yticks(range(1, len(_labels) + 1), _labels, fontsize=10)
    _axes[2].set_xlabel("IoU with the label")
    _axes[2].set_xlim(0, 1)
    _axes[2].set_title("clicks", fontsize=11)
    _fig
    return


@app.cell
def _(mo, sam_buildings, sam_cracks, sn8_candidates):
    _med, _orc = sam_cracks["iou_median"], sam_cracks["oracle_median"]
    _dy, _dx = sam_buildings["shift"]
    _moved = (
        " and ".join(
            f"{abs(_v)} pixel{'s' if abs(_v) > 1 else ''} {_w[_v > 0]}"
            for _v, _w in ((_dy, ("up", "down")), (_dx, ("left", "right")))
            if _v
        )
        or "not at all"
    )
    mo.md(f"""
    ### Reveals 3 and 4: SAM finds buildings, not cracks

    Green: the label; orange: SAM's mask; stars: the clicks. On 3DCrack's test images with a
    crack ({sam_cracks["images"]}), the median IoU of SAM's own choice is {_med[1]:.2f} with 1
    click, {_med[3]:.2f} with 3, {_med[5]:.2f} with 5; taking the best of the three candidates by
    the label (cheating) only reaches {max(_orc.values()):.2f}. SAM's mask is a blob: at 3 clicks,
    its median area is {sam_cracks["area_ratio_median"]:.0f} times the crack's. On SpaceNet 8's
    test buildings ({sam_buildings["buildings"]} of the {sn8_candidates} of 400 pixels or more,
    one click at the most interior pixel), the median IoU is {sam_buildings["iou_median"]:.2f},
    and {100 * sam_buildings["share_above_half"]:.0f}% are above 0.5; the figure's house: IoU
    {sam_buildings["example_iou"]:.2f}, SAM's mask {sam_buildings["example_mask_px"]:,} pixels
    against a {sam_buildings["example_footprint_px"]:,}-pixel footprint. SAM learned what people
    outline as objects; a crack is a line through a surface. Lecture 16's U-Net, trained on
    cracks: 0.70.

    A check of the building labels: footprints drawn off the imagery in one direction would cost
    every building alike, and one common shift would win it back. The best common shift, chosen
    on half of the buildings ({_moved}), moves the other half's median IoU only from
    {sam_buildings["held_out_median"]:.3f} to {sam_buildings["held_out_shifted"]:.3f}: where SAM's
    mask and the footprint disagree, they disagree building by building.
    """)
    return


@app.cell
def _(few_shot, full_probes, mo, shown, zs):
    _clip = full_probes["CLIP probe, all photos"]
    _r18 = full_probes["ResNet-18 probe (Lecture 15)"]
    _z2 = few_shot[2]["zero_shot"]
    _pct = lambda _v: f"{100 * _v:.0f}"  # noqa: E731
    _span = lambda _m: f"{_pct(_m['range'][0])}–{_pct(_m['range'][1])}%"  # noqa: E731
    # The table: zero-shot 0.52, 71-92%, 39-78%; 2 + 2: 0.68, 79-95%, 32-74%; 4 + 4: 0.71,
    # 83-93%, 36-59%; ResNet-18 0.67, 86%, 51%; CLIP on all photos 0.78, 88%, 39%
    _want = {2: ("0.68", ("79", "95"), ("32", "74")), 4: ("0.71", ("83", "93"), ("36", "59"))}
    for _k, (_ap, _rec, _rev) in _want.items():
        _f = few_shot[_k]
        assert shown(_f["ap"]["median"], _ap)
        assert all(shown(100 * _v, _t) for _v, _t in zip(_f["recall"]["range"], _rec, strict=True))
        assert all(shown(100 * _v, _t) for _v, _t in zip(_f["review"]["range"], _rev, strict=True))
    assert shown(_r18["ap"], "0.67") and shown(100 * _r18["recall"], "86")
    assert shown(100 * _r18["review"], "51")
    assert shown(_clip["ap"], "0.78") and shown(100 * _clip["recall"], "88")
    assert shown(100 * _clip["review"], "39") and shown(zs["at_half"]["ap"], "0.52")
    # "close": two photos per surface against ResNet-18 on all photos, a gap under 0.03
    assert 0 < few_shot[2]["ap"]["median"] - _r18["ap"] < 0.03
    # "the ranges narrow" from 2 to 4 photos per surface, and again with all photos
    for _m in ("recall", "review"):
        _w = {_k: few_shot[_k][_m]["range"][1] - few_shot[_k][_m]["range"][0] for _k in (2, 4, 16)}
        assert _w[2] > _w[4] > _w[16], (_m, _w)
    _rows = "\n".join(
        f"    | CLIP probe, {_k} + {_k} per surface | {few_shot[_k]['ap']['median']:.2f} | "
        f"{_span(few_shot[_k]['recall'])} | {_span(few_shot[_k]['review'])} |"
        for _k in (2, 4)
    )
    mo.md(f"""
    ## 7. Decide; check yourself

    ### Few labels rank; thresholds need more

    | Test photos, 90% target | AP | Cracks flagged | Tiles to review |
    |:------------------------|---:|---------------:|----------------:|
    | Zero-shot, 2 photos per surface | {zs["at_half"]["ap"]:.2f} | {_span(_z2["recall"])} | {_span(_z2["review"])} |
{_rows}
    | ResNet-18 probe, all (Lecture 15) | {_r18["ap"]:.2f} | {_pct(_r18["recall"])}% | {_pct(_r18["review"])}% |
    | CLIP probe, all photos | {_clip["ap"]:.2f} | {_pct(_clip["recall"])}% | {_pct(_clip["review"])}% |

    The decision for the new owner. "2 + 2": two labelled photos per surface fit the probe, two
    more set the threshold; the ranges are over ten draws of the photos, AP the median. Two per
    surface rank about as well as Lecture 15's screen on 137 photos (AP
    {few_shot[2]["ap"]["median"]:.2f} against {_r18["ap"]:.2f}: close, on one split), but a
    threshold from two photos per surface is a gamble: depending on which photos, it flags
    {_span(few_shot[2]["recall"])} of the cracks and sends {_span(few_shot[2]["review"])} of the
    tiles to review. With four per surface the ranges narrow; with all 137 training and 47
    validation photos the CLIP probe gives {_clip["ap"]:.2f}, {_pct(_clip["recall"])}%,
    {_pct(_clip["review"])}%. Zero-shot can rank photos for a first look, not set a threshold the
    inspector can sign. Label a few photos to fit, more to set the threshold, on the owner's own
    camera, and refit as labels accumulate in service.
    """)
    return


@app.cell
def _(cache_check, mo):
    _gpu_days = 12 * 256  # the paper: the largest ViT, 12 days on 256 V100 GPUs
    assert _gpu_days > 1000  # "thousands of GPU-days"
    # One pass over every tile, measured above: about 20 min of network time on the slides' RTX
    # 3090 Ti when the GPU is free, 45 min while another job shared it. Timing depends on the
    # device and its load, so the check is only the contrast: far below one GPU-day
    if cache_check["device"] == "cuda":
        assert cache_check["minutes_all"] < 24 * 60
    mo.md(f"""
    ### The model card says: not for deployment

    | | What we know |
    |:--|:-------------|
    | Training data | 400 million web pairs, not released |
    | Its card | "any deployed use case … out of scope" |
    | What we checked | our test photos, three surfaces |
    | What we did not | other cameras, other owners, bias in the web data |

    OpenAI's model card for CLIP states that any deployed use case of the model, commercial or
    not, is currently out of scope, and that the model was released for research. An agency
    screening bridge photos is a deployment: before using CLIP's features in service, read the
    card and the licence, and test on the owner's data; or use a model released for deployment
    (SigLIP, Apache-2.0, is one). The training data are web captions no one in this room chose:
    whatever they over- and under-represent comes along. The environmental note: one embedding
    pass over the 56,092 tiles took about 20 minutes on one GPU for the slides; measured above on
    {cache_check["rows"]} tiles, it is about {cache_check["minutes_all"]:.0f} minutes of network
    time on this machine's {cache_check["device"].upper()} (reading and resizing the tiles is
    extra). Training the largest CLIP ViT took {_gpu_days:,} GPU-days.

    ### Vote: what screens the new decks?

    - **A.** Zero-shot CLIP, threshold 0.5: no labels, start today.
    - **B.** Label 2 photos per surface, probe CLIP, set the threshold on a few more.
    - **C.** Lecture 15's ResNet-18 screen, trained on SDNET2018.
    - **D.** Wait until 137 photos are labelled.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(few_shot, full_probes, mo, zs):
    _clip = full_probes["CLIP probe, all photos"]
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        **B.** A flags only {100 * zs["at_half"]["recall"]:.0f}% of cracks, with no way to set a
        threshold an inspector can sign. C was trained on another owner's photos; its threshold
        still needs the new owner's. D ends with the best screen (AP {_clip["ap"]:.2f},
        {100 * _clip["review"]:.0f}% of tiles to review) but screens nothing until 137 photos are
        labelled. B ranks about as well as Lecture 15's screen from six photos (AP
        {few_shot[2]["ap"]["median"]:.2f} against 0.67), but its threshold needs more than two
        photos per surface: with two, the share of cracks flagged ranged from
        {100 * few_shot[2]["recall"]["range"][0]:.0f}% to
        {100 * few_shot[2]["recall"]["range"][1]:.0f}% over ten draws; with four, from
        {100 * few_shot[4]["recall"]["range"][0]:.0f}% to
        {100 * few_shot[4]["recall"]["range"][1]:.0f}%. B refits as labels accumulate toward D's.
        B also respects CLIP's card if the owner chooses a deployable model for the features, and
        the audit of unflagged tiles (Lecture 15) applies unchanged.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where vision foundation models live

    | Today | What it is | Lecture |
    |:------|:-----------|:--------|
    | Stable Diffusion, DALL-E 2 | CLIP's text side steers image generation | today |
    | LLaVA, GPT-4o, Claude | an image encoder feeding a language model | 19 |
    | Annotation tools (CVAT, Label Studio) | SAM proposes masks, people correct them | today |
    | Prithvi (NASA, IBM) | a foundation model for satellite images | not yet |

    CLIP's encoders sit inside text-to-image generators and early vision-language models; SAM is
    now a standard first draft in labelling tools, which is how it pays in practice: a person
    clicks, SAM drafts, the person fixes. Geospatial foundation models (Prithvi, trained on
    NASA's Harmonized Landsat-Sentinel data) map floods and burn scars from a few labelled
    scenes.

    ### Exit question

    A vendor says its screen "needs no training data: describe the defect in words." Name two
    things you would ask before using it, and the number you would want.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        On whose photos was it tested, and at what threshold? Without labels there is no
        threshold an inspector can sign. Ask for recall and the share of tiles to review at your
        target, on your photos, and what its model card allows. The number: tiles to review at
        90% of cracks flagged, on a few labelled photos of yours.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas

    - **Normalize before the cosine.** The cached embeddings are CLIP's projections, not unit
      vectors (norms about 17 to 22); the zero-shot logits are $s$ times the cosine.
    - **`logit_scale` is a log.** The model stores $\log s$; the softmax uses
      $s = e^{4.6052} = 100$.
    - **Transformers 5 returns objects.** `get_image_features` and `get_text_features` return
      an output whose `.pooler_output` is the projected embedding, not a tensor.
    - **A prompt pair is a classifier with its own offset.** Pairs that differ by surface put
      surfaces on different scales; compare scores only within one pair, and choose the pair on
      validation photos.
    - **0.5 is the softmax's threshold, not the inspector's.** A recall target needs labelled
      photos to set.
    - **Repeated captions in a batch** are counted wrong by the contrastive loss; the loss of a
      batch of labels-as-captions cannot reach zero.
    - **SAM's clicks are [x, y]** (column, row), not NumPy's [row, column]; SAM returns three
      masks per prompt, chosen here by its own predicted IoU (choosing by the label is cheating).
    - **Encode each image once.** SAM's image encoder is the expensive part; prompts reuse its
      output.

    ## Going deeper (Forsyth)

    - §7.1–7.2: canonical correlation analysis finds linear projections of image features and
      word vectors whose correlation is largest, and Forsyth's words-and-pictures example labels
      a new image by its nearest captions in that shared space. CLIP's recipe with linear maps
      instead of deep encoders, and correlation instead of a contrastive loss. Forsyth also shows
      the shared space looking good on a few images and poor on most, today's six tiles against
      11,208 in miniature.
    - §6.3.1: the cosine between word-count vectors of documents. Forsyth calls the cosine
      itself the "cosine distance" (near 1 for similar documents); scikit-learn's
      `metric="cosine"`, used for k-NN here, is one minus it.
    - §16.1.1: the softmax over units' outputs as class probabilities; zero-shot CLIP is that
      layer with weights written in words.
    - §1.2: nearest-neighbour classifiers and Forsyth's $(k, l)$ rule: the class with the most
      of the $k$ votes, but only if it has more than $l$ of them. Our 90% target asks for
      something else: flag a tile when the distance-weighted share of cracked neighbours passes
      a threshold set on validation, far below a majority.
    - §18.2.5: IoU and average precision. Forsyth's convention averages the *interpolated*
      precision at 11 recall levels (0, 0.1, …, 1); scikit-learn's `average_precision_score`,
      used here, sums the precision at each step of recall without interpolation, so the two
      can differ on the same ranking.

    **Exercises.** (1) Prompt ensembling: average the unit text embeddings of several prompts per
    class (clean and cracked), renormalize, and choose the set on validation photos; does it beat
    the best single pair on the test photos? (2) Give SAM a box around the labelled crack
    instead of clicks (`input_boxes`): how far does the median IoU move?

    ## Things to remember

    - CLIP: captions as the classifier.
    - Contrastive: pick your pair from a batch.
    - A few labels rank; thresholds need more.
    - SAM outlines objects; cracks are lines.
    - Read the model card before deploying.

    **Next:** language models, GPT, and retrieval.
    """)
    return


if __name__ == "__main__":
    app.run()
