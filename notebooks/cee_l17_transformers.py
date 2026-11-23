"""Applied Machine Learning for CEE — Lecture 17: Tokens, Attention, and Transformers.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l17_transformers.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: OSHA's Severe Injury Reports (U.S. government data, public): osha.gov refuses scripted
downloads, so save the ZIP with a browser and run ``bash scripts/fetch_osha_sir.sh <saved zip>``.
The UCI concrete data (Yeh, 1998, CC BY 4.0) from ``bash scripts/fetch_concrete.sh``. SDNET2018
(Maguire, Dorafshan & Thomas, 2018, CC BY 4.0) from ``bash scripts/fetch_sdnet2018.sh``, then
``pixi run -e aml-env python scripts/derive_sdnet2018.py`` and
``pixi run -e aml-env python scripts/derive_sdnet2018_vit.py``. BERT (bert-base-uncased,
Apache-2.0) comes from the Hugging Face Hub on the first run; ViT-B/16 and ResNet-18 (ImageNet
weights, research and education terms) from torchvision.

Set ``L17_SETTING=reduced`` (the default: a laptop CPU or a free Colab GPU) or
``L17_SETTING=slides`` (the slides' full run) to choose the setting (see the header cell).
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
    # Applied Machine Learning for CEE — Lecture 17: Tokens, Attention, and Transformers

    **Slides:** *Applied Machine Learning for CEE*, Lecture 17 (`lectures/cee_l17_transformers`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §1.2 (nearest neighbours), §6.3 (text models: bags of words, the cosine distance, and the meaning of words from the words near them), §16.1.1 (the softmax), §18.1.9 (residual networks), and §19.2.3 (the denoising autoencoder).

    **Data:**
    - OSHA Severe Injury Reports (U.S. Occupational Safety and Health Administration; U.S.
      government data, public): 105,996 reports, January 2015 to November 2025, each with the
      employer's description of the incident; federal OSHA jurisdiction only. osha.gov refuses
      scripted downloads: save the ZIP with a browser, then
      `bash scripts/fetch_osha_sir.sh <the saved zip>`. Read in place, never extracted.
    - UCI Concrete Compressive Strength (Yeh, 1998; CC BY 4.0): Lecture 2's 1,030 tests,
      `data/derived/Concrete_Data.csv` from `bash scripts/fetch_concrete.sh`.
    - SDNET2018 (Maguire, Dorafshan & Thomas, 2018; CC BY 4.0): Lecture 15's 56,092 tiles from
      230 photographs of bridge decks, walls, and pavements. Fetch with
      `bash scripts/fetch_sdnet2018.sh`, then `python scripts/derive_sdnet2018.py` (the tile
      index and ResNet-18 features) and `python scripts/derive_sdnet2018_vit.py` (ViT-B/16
      features).
    - BERT (`google-bert/bert-base-uncased`, Devlin et al., 2019; Apache-2.0) at a pinned
      revision, downloaded once from the Hugging Face Hub (about 0.4 GB).
    - torchvision's ViT-B/16 and ResNet-18 with ImageNet weights (`IMAGENET1K_V1`): research
      and education terms.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`.

    **Two settings**, chosen by the environment variable `L17_SETTING` (whatever the device):

    - `reduced` (the default): for a laptop CPU or a free Colab GPU; 6 minutes on the CPU of the
      slides' machine (56 threads, no GPU, November 20, 2026), 17 minutes on four of its threads
      (November 19), with up to 6 GB of memory.
      Everything computed on the CPU from the data and the cached features (byte pairs,
      word2vec, attention on concrete, the probes, the transformer block, the position
      embeddings, the attention maps) is the slides' computation and is checked exactly. The
      two heavy parts run on a sample: the shuffle test (§5.1) embeds every 64th tile under
      one permutation instead of all 56,092 under three (its numbers are printed, not
      checked: 877 tiles are too few), and "Similar to what?" (§5.2) compares 20,000 pairs
      of 2,000 reports instead of 200,000 pairs of all 26,599 (checked to within 0.03).
    - `slides`: the slides' full run: BERT on 26,599 reports and every tile embedded four
      times by each network. 28 minutes on the slides' GPU shared with other jobs; hours on a
      CPU. On the slides' GPU (an RTX 3090 Ti) the slides' digits are checked; elsewhere the
      numbers from a GPU move in their last digits, and their asserts allow a stated
      difference.

    **Your tasks** (from the slides): say which two injury reports are alike, merge byte pairs
    by hand, train word vectors, let a proposed concrete mix attend to the tested ones, write a
    transformer block and match torchvision's, let BERT fill blanks, then break the readers four
    ways and decide which backbone screens next season's decks.
    """)
    return


@app.cell
def _():
    import itertools
    import json
    import math
    import os
    import random
    import re
    import time
    import zipfile
    from dataclasses import dataclass
    from pathlib import Path

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub loads
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers
    from tokenizers import models as tk_models
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16
    from torchvision.transforms.v2 import functional as tvf
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    SEED = 17
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed):
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    # As the slides' figure script: deterministic kernels wherever PyTorch has one, and float32
    # convolutions on a GPU (cuDNN's default, TF32, keeps 10 mantissa bits: ResNet-18's features
    # then differ from its cache, made on the CPU, by up to about 0.01; without it, about 1e-5)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.allow_tf32 = False
    hf_logging.set_verbosity_error()  # BERT's checkpoint also holds a pooler and a next-sentence
    hf_logging.disable_progress_bar()  # head the masked-word model does not use; no report
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"
    WORKERS = min(12, os.cpu_count() or 1)  # processes that decode tiles for the network

    # The setting, independent of the device: "reduced" (default) or "slides" (the header cell)
    SETTING = os.environ.get("L17_SETTING", "reduced").strip().lower()
    if SETTING not in {"reduced", "slides"}:
        raise ValueError(f"L17_SETTING must be 'reduced' or 'slides', not {SETTING!r}")
    SLIDES = SETTING == "slides"
    # The slides' digits of what ran on a GPU are checked only on the slides' GPU, in "slides"
    REFERENCE_GPU = SLIDES and GPU and "3090 Ti" in torch.cuda.get_device_name(0)
    # Everything computed on the CPU (byte pairs, word2vec, the concrete attention, the probes on
    # the cached features, the position embeddings) reproduces the slides' numbers to round-off.
    EXACT = 1e-6
    BLOCK_TOL = 1e-14  # our block against torchvision's in float64 (the last bits vary by threads)
    EMB_TOL = 1e-5  # BERT's mean token vectors of the three reports (float32, on the CPU)
    # BERT's 1,102 blanks ran on the slides' GPU. On another GPU or on the CPU the numbers move a
    # little, so their checks allow a stated difference.
    MLM_TOL = 0.002 if GPU else 0.01  # BERT's share right, of 1,102 blanks (2 or 11 of them)
    GUESS_TOL = 2 if GPU else 8  # how often each word is BERT's first guess
    # The shuffle test (§5.1): every tile, three permutations ("slides"); every 64th tile, one
    # (on four CPU threads ViT-B/16 takes about 0.4 s a tile)
    SHUFFLE_EVERY, SHUFFLE_PERMS = (1, 3) if SLIDES else (64, 1)
    FEAT_TOL = 0.002  # "slides" off the slides' GPU: AP and shares from re-embedded tiles
    # "Similar to what?" over every report (§5.2): all 26,599 and 200,000 pairs, or a sample
    SIM_REPORTS, SIM_PAIRS = (None, 200_000) if SLIDES else (2_000, 20_000)
    SIM_TOL = 0.03  # the sample's AUCs against the slides'

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
    OSHA_ZIP = DATA / "raw" / "osha_sir" / "January2015toNovember2025.zip"
    CONCRETE_CSV = DATA / "derived" / "Concrete_Data.csv"
    RAW = DATA / "raw" / "sdnet2018" / "SDNET2018"
    TILES_CSV = DATA / "derived" / "sdnet2018_tiles.csv"
    R18_FEATURES = DATA / "derived" / "sdnet2018_resnet18.npy"
    VIT_FEATURES = DATA / "derived" / "sdnet2018_vit_b_16.npy"
    # Lecture 2's concrete: one encoder, 30 splits by design, 5-NN
    CONCRETE_COLUMNS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg",
                        "fine_agg", "age", "strength"]  # fmt: skip
    INGREDIENTS = CONCRETE_COLUMNS[:7]
    PROPOSED_ROW = 638  # the proposed mix of Lectures 5 and 10: the query
    REPEATS, TEST_SIZE, SPLIT_SEED, K = 30, 0.25, 0, 5
    TAUS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)  # attention temperatures tried (squared z-units)
    TOP_KEYS = 8  # keys shown in the one picture
    # Lecture 15's crack tiles, split, probe, and target
    SURFACES = ("deck", "wall", "pavement")
    SPLIT = (0.6, 0.8)  # cumulative share of each surface's photos: train, val (rest: test)
    SIZE, PATCH = 224, 16  # the networks' input; ViT-B/16's patch, pixels
    GRID = SIZE // PATCH  # 14 x 14 patches
    # The shuffle test's blocks: 2 x 2 of ViT's patches, one cell of ResNet-18's 7 x 7 last map
    BLOCK = 32
    BLOCKS = SIZE // BLOCK  # 7 x 7 blocks
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet's statistics
    C_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)  # inverse L2 strengths tried on validation
    RECALL = 0.90  # the inspector's target: flag 9 of every 10 cracked tiles
    VERIFY_TILE = "D/CD/7001-21.jpg"  # Lecture 15's filter tile: the block is checked on it
    SHOW_POSITIONS = ((0, 0), (0, 13), (7, 7), (13, 6))  # patches whose position map is shown
    # OSHA's reports: tokens, word vectors, BERT
    TRAIN_YEARS = (2015, 2022)  # tokenizers and word vectors learn from these years; test 2023-25
    WORD = re.compile(r"\w+|[^\w\s]+")  # the pre-tokenizer: runs of letters/digits, or punctuation
    LETTERS = re.compile(r"[a-z]+")
    SHOW_MERGES, CHECK_MERGES, BPE_VOCAB = 12, 50, 8000
    DOMAIN_WORDS = ("scaffold", "excavator", "trench", "forklift", "amputated", "lacerated",
                    "skid", "rebar", "drywall", "trencher")  # fmt: skip
    W2V_DIM, W2V_WINDOW, W2V_NEG, W2V_EPOCHS, W2V_MIN = 100, 5, 5, 3, 5
    W2V_BATCH, W2V_LR, W2V_SUBSAMPLE = 4096, 3e-3, 1e-5
    QUERY_WORDS = ("ladder", "scaffold", "trench", "forklift", "amputated", "finger")
    BERT = "google-bert/bert-base-uncased"  # Devlin et al. (2019); Apache-2.0
    BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
    FELL = re.compile(r"\b(fell (?:from|off) (?:a|an|the) )([a-z]+)\b")
    TRY_SENTENCE = "the worker held the ladder there, then the ladder fell"
    BARE = "The employee fell from a [MASK] and fractured his leg."  # no job, no tool, no task
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    print(f"device: {DEVICE}; setting: {SETTING}" + (" (the slides' GPU)" if REFERENCE_GPU else ""))
    return (
        AutoModelForMaskedLM,
        AutoTokenizer,
        BARE,
        BERT,
        BERT_REVISION,
        BLOCK,
        BLOCKS,
        BLOCK_TOL,
        BPE_VOCAB,
        CHECK_MERGES,
        CONCRETE_COLUMNS,
        CONCRETE_CSV,
        C_GRID,
        DEVICE,
        DOMAIN_WORDS,
        DataLoader,
        Dataset,
        EMB_TOL,
        EXACT,
        F,
        FEAT_TOL,
        FELL,
        GPU,
        GRID,
        GUESS_TOL,
        GroupKFold,
        GroupShuffleSplit,
        INGREDIENTS,
        Image,
        K,
        KNeighborsRegressor,
        LETTERS,
        LogisticRegression,
        MEAN,
        MLM_TOL,
        OSHA_ZIP,
        PALETTE,
        PATCH,
        PROPOSED_ROW,
        QUERY_WORDS,
        R18_FEATURES,
        RAW,
        RECALL,
        REFERENCE_GPU,
        REPEATS,
        ResNet18_Weights,
        SEED,
        SETTING,
        SHOW_MERGES,
        SHOW_POSITIONS,
        SHUFFLE_EVERY,
        SHUFFLE_PERMS,
        SIM_PAIRS,
        SIM_REPORTS,
        SIM_TOL,
        SIZE,
        SLIDES,
        SPLIT,
        SPLIT_SEED,
        STD,
        SURFACES,
        StandardScaler,
        TAUS,
        TEST_SIZE,
        TILES_CSV,
        TOP_KEYS,
        TRAIN_YEARS,
        TRY_SENTENCE,
        Tokenizer,
        VERIFY_TILE,
        VIT_FEATURES,
        ViT_B_16_Weights,
        W2V_BATCH,
        W2V_DIM,
        W2V_EPOCHS,
        W2V_LR,
        W2V_MIN,
        W2V_NEG,
        W2V_SUBSAMPLE,
        W2V_WINDOW,
        WORD,
        WORKERS,
        average_precision_score,
        dataclass,
        itertools,
        json,
        math,
        nn,
        normalizers,
        np,
        os,
        pd,
        plt,
        pre_tokenizers,
        require,
        resnet18,
        roc_auc_score,
        seed_all,
        shown,
        time,
        tk_models,
        torch,
        trainers,
        tvf,
        vit_b_16,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 2–15

    On your own, no notes, two minutes:

    1. Lecture 2: 5-NN predicts a mix's strength from its 5 nearest tested mixes. What weight
       does each of them get, and what weight does every other mix get?
    2. Lecture 6: softmax turns scores (2, 1, 0) into probabilities. Roughly what are they?
    3. Lecture 15: a 3 × 3 filter sees 9 pixels. How many pixels could one layer see at most?

    **Today:** every token looks at every other, with learned weights.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. One fifth each; every other mix gets exactly zero. Today's attention gives every
           tested mix a weight, larger for the more similar, and the weights still sum to one
           (§2).
        2. $e^2 : e : 1$ over their sum: 0.67, 0.24, 0.09 (checked below). Attention uses the
           same softmax on similarity scores.
        3. All of them. A convolution sees a window and needs many layers to see the whole tile
           (Lecture 15's receptive fields); one attention layer lets every patch draw on every
           other.
        """)
        }
    )
    return


@app.cell
def _(np, shown):
    _s = np.array([2.0, 1.0, 0.0])
    _p = np.exp(_s) / np.exp(_s).sum()
    print("softmax(2, 1, 0) =", _p.round(4))
    assert all(shown(_v, _t) for _v, _t in zip(_p, ("0.67", "0.24", "0.09"), strict=True))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    ### The screen again: a better reader?

    Lecture 15's screen: ImageNet's ResNet-18, frozen, under a logistic-regression probe ($C$
    chosen on the validation photos), with the threshold set to flag 90% of the validation
    photos' cracked tiles. Today the same question is asked of a newer architecture, on the same
    230 photographs, the same photo split, the same probe, and the same threshold rule: does a
    transformer's view of a tile leave the inspector fewer tiles to review?

    | What changes | Cost |
    |:-------------|:-----|
    | A crack missed | unrecorded until the next inspection |
    | A tile reviewed | seconds of the inspector's time |
    | A bigger model | slower screening, more energy |

    Before images, the lecture builds the machinery on text and on Lecture 2's mixes: tokens,
    attention, transformers. First, Lecture 15's numbers again.
    """)
    return


@app.cell
def _(
    Image,
    MEAN,
    R18_FEATURES,
    RAW,
    ResNet18_Weights,
    SIZE,
    SPLIT,
    STD,
    SURFACES,
    TILES_CSV,
    VIT_FEATURES,
    ViT_B_16_Weights,
    json,
    nn,
    np,
    pd,
    require,
    resnet18,
    torch,
    tvf,
    vit_b_16,
):
    def read_tile(path, gray=False):
        """One tile: uint8 (256, 256, 3), or grey levels 0-255 as floats (256, 256)."""
        im = Image.open(RAW / path)
        return np.array(im.convert("L"), dtype=np.float64) if gray else np.array(im.convert("RGB"))

    def tile_tensor(path):
        """A tile resized to 224 and normalized as the cached features were: (1, 3, 224, 224).

        Lecture 15's preprocessing, exactly (bilinear, antialiased; ImageNet's mean and std).
        """
        x = tvf.to_dtype(torch.from_numpy(read_tile(path)).permute(2, 0, 1), torch.float32,
                         scale=True)  # fmt: skip
        x = tvf.normalize(tvf.resize(x, [SIZE, SIZE], antialias=True), list(MEAN), list(STD))
        assert x.shape == (3, SIZE, SIZE), x.shape
        return x[None]

    def backbone():
        """ImageNet ResNet-18 without its classifier: 512 pooled features (eval mode)."""
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        return model.eval()

    def vit():
        """torchvision's ViT-B/16, ImageNet weights, classifier removed: the class token, 768."""
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
        return model.eval()

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
        """The tile index and both feature caches, row for row, after checking how they were made.

        Raises:
            ValueError: If a cache was made with other weights or preprocessing.
        """
        hint = "run `bash scripts/fetch_sdnet2018.sh`, then `python scripts/derive_sdnet2018.py`"
        df = pd.read_csv(require(TILES_CSV, hint))
        preprocess = f"resize {SIZE}x{SIZE} bilinear antialias; ImageNet mean {MEAN} std {STD}"
        feats = []
        for path, weights, dim, name, layer in (
            (R18_FEATURES, ResNet18_Weights.IMAGENET1K_V1, 512, "ResNet18",
             "global average pool (fc replaced by identity)"),
            (VIT_FEATURES, ViT_B_16_Weights.IMAGENET1K_V1, 768, "ViT_B_16",
             "class token after the encoder's final layer norm (heads replaced by identity)"),
        ):  # fmt: skip
            hint_f = hint if dim == 512 else "run `python scripts/derive_sdnet2018_vit.py`"
            meta = json.loads(require(path.with_suffix(".json"), hint_f).read_text())
            want = {
                "weights": f"torchvision {name}_Weights.IMAGENET1K_V1",
                "url": weights.url,
                "preprocess": preprocess,
                "layer": layer,
                "shape": [len(df), dim],
            }
            bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
            if bad:
                raise ValueError(f"{path.name} does not match this preprocessing: {bad}")
            f = np.load(require(path, hint_f))
            assert f.shape == (len(df), dim), f.shape
            feats.append(f)
        return df, feats[0], feats[1]

    return backbone, load_tiles, photo_split, read_tile, tile_tensor, vit


@app.cell
def _(load_tiles, np, photo_split):
    tiles, rfeat, vfeat = load_tiles()
    # Lecture 15's split, exactly: its notebook's seed (0) and lecture number, not this SEED
    part = photo_split(tiles, np.random.default_rng((0, 15)))
    y_tile = tiles["cracked"].to_numpy()
    _photos = {_r: tiles.loc[part == _r, "photo"].nunique() for _r in ("train", "val", "test")}
    print(f"{len(tiles):,} tiles from {tiles['photo'].nunique()} photos; photos per set {_photos}")
    assert len(tiles) == 56_092 and tiles["photo"].nunique() == 230
    assert _photos == {"train": 137, "val": 47, "test": 46}  # "137 training photos" (the vote)
    assert (
        tiles.groupby("photo").apply(lambda g: len(set(part[g.index])), include_groups=False) == 1
    ).all()
    return part, rfeat, tiles, vfeat, y_tile


@app.cell
def _(
    C_GRID,
    LogisticRegression,
    RECALL,
    SURFACES,
    StandardScaler,
    average_precision_score,
    dataclass,
    math,
    np,
    roc_auc_score,
):
    @dataclass(frozen=True)
    class Probe:
        """Logistic regression with an L2 penalty on standardized features (Lecture 6)."""

        scaler: StandardScaler
        model: LogisticRegression
        C: float
        val_ap: float

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

    @dataclass(frozen=True)
    class Screen:
        """One backbone's probe on Lecture 15's photo split."""

        p: np.ndarray  # probability of "cracked" for every tile
        t: float  # threshold that flags 90% of the validation photos' cracked tiles
        numbers: dict

    def screen(feats, y, part, df):
        """Fit the probe, set the threshold on validation, score the test photos once."""
        train, val, test = (part == r for r in ("train", "val", "test"))
        pr = fit_probe(feats, y, train, val)
        p = pr.score(feats)
        t = threshold_for(p[val], y[val])
        out = {"C": pr.C, "val_ap": pr.val_ap, "all": scores(p[test], y[test], t)}
        for s in SURFACES:
            m = test & (df["surface"] == s).to_numpy()
            out[s] = scores(p[m], y[m], t)
        return Screen(p, t, out)

    def close(got, want, tol):
        """Assert every metric in ``want`` is within ``tol`` of ``got``'s."""
        for k, v in want.items():
            assert abs(got[k] - v) <= tol, (k, got[k], v)

    return close, screen


@app.cell
def _(EXACT, close, part, rfeat, screen, shown, tiles, y_tile):
    res_screen = screen(rfeat, y_tile, part, tiles)
    _n = res_screen.numbers
    _all = _n["all"]
    print(
        f"ResNet-18 probe: C = {_n['C']}, validation AP {_n['val_ap']:.4f}; test photos: "
        f"{100 * _all['recall']:.1f}% of cracks flagged, {100 * _all['review']:.1f}% of tiles to review"
    )
    assert _n["C"] == 0.03 and abs(_n["val_ap"] - 0.7370948470874679) < EXACT
    _want = {  # auc, ap, recall, review, acc, n
        "all": (0.8415554674616079, 0.6659046754776506, 0.861576660459342,
                0.5059778729478943, 0.9064061384725196, 11208),
        "deck": (0.8268986414760938, 0.6545574306543193, 0.9106753812636166,
                 0.6601731601731602, 0.8874458874458875, 2772),
        "wall": (0.8572635019523799, 0.7444309721968776, 0.9156414762741653,
                 0.6553094832481544, 0.9170925610448609, 3522),
        "pavement": (0.8347067473592786, 0.5895616517327669, 0.7701543739279588,
                     0.31196581196581197, 0.9094424094424094, 4914),
    }  # fmt: skip
    _metrics = ("auc", "ap", "recall", "review", "acc", "n")
    for _k, _v in _want.items():
        close(_n[_k], dict(zip(_metrics, _v, strict=True)), EXACT)
    # the slide: "flags 86% of cracks; the inspector reviews 51% of tiles"
    assert shown(100 * _all["recall"], "86") and shown(100 * _all["review"], "51")
    return (res_screen,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: which two reports are alike?

    Three reports from OSHA's Severe Injury Reports, construction (NAICS 23), 2023–2025, chosen
    by a rule in code below: **A**, a fall from a ladder; **B**, a fall, mentioning a scaffold
    and no ladder, sharing the fewest words with A; **C**, not a fall (no "fell" or "fall" in
    it either), sharing the most words with A. Which pair is most alike, and why? Write it down.
    The answer comes back in §5.
    """)
    return


@app.cell
def _(LETTERS, OSHA_ZIP, TRAIN_YEARS, pd, require, zipfile):
    def osha():
        """The reports: event year, industry code, OIICS event code, and the narrative.

        Read from the published ZIP (latin-1 CSV), never extracted to disk.
        """
        hint = "save the ZIP from osha.gov, then `bash scripts/fetch_osha_sir.sh <saved zip>`"
        with (
            zipfile.ZipFile(require(OSHA_ZIP, hint)) as zf,
            zf.open("January2015toNovember2025.csv") as fh,
        ):
            df = pd.read_csv(fh, encoding="latin-1", low_memory=False, dtype={"Primary NAICS": str})
        out = pd.DataFrame(
            {
                "year": pd.to_datetime(df["EventDate"], format="%m/%d/%Y").dt.year,
                "naics": df["Primary NAICS"].fillna(""),
                "text": df["Final Narrative"]
                .fillna("")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip(),
                # OIICS event codes: 43xx is a fall to a lower level
                "event": df["Event"].astype("Int64").astype(str),
            }
        )
        out = out[out["text"].str.len() > 0].reset_index(drop=True)
        out["lower"] = out["text"].str.lower()
        return out

    def training(df):
        """Lowercased narratives of the training years (2015-2022)."""
        return df.loc[df["year"].between(*TRAIN_YEARS), "lower"]

    def jaccard(a, b):
        """Shared words over all words (sets of lowercase words)."""
        x, y = set(LETTERS.findall(a)), set(LETTERS.findall(b))
        return len(x & y) / len(x | y)

    return jaccard, osha, training


@app.cell
def _(TRAIN_YEARS, osha, training):
    osha_df = osha()
    _train = training(osha_df)
    _test = osha_df.loc[osha_df["year"] > TRAIN_YEARS[1]]
    _words = int(osha_df["lower"].str.count(r"[a-z]+").sum())
    print(
        f"{len(osha_df):,} reports, {osha_df['year'].min()}-{osha_df['year'].max()}: "
        f"{len(_train):,} to learn from (2015-2022), {len(_test):,} to test on (2023-2025); "
        f"{_words:,} words"
    )
    assert len(osha_df) == 105_996 and _words == 3_683_085
    assert len(_train) == 79_307 and len(_test) == 26_689
    assert (osha_df["year"].min(), osha_df["year"].max()) == (2015, 2025)
    return (osha_df,)


@app.cell
def _(SEED, TRAIN_YEARS, jaccard, mo, osha_df):
    _t = osha_df[(osha_df["year"] > TRAIN_YEARS[1]) & osha_df["naics"].str.startswith("23")].copy()
    _t["n"] = _t["lower"].str.count(r"[a-z]+")
    _t = _t[_t["n"].between(20, 40)]
    _fall = _t["event"].str.startswith("43")  # OIICS 43xx: a fall to a lower level
    _a = _t[_fall & _t["lower"].str.contains("ladder")].sample(1, random_state=SEED).iloc[0]
    _b_pool = _t[_fall & _t["lower"].str.contains("scaffold") & ~_t["lower"].str.contains("ladder")]
    _b = _b_pool.loc[_b_pool["lower"].map(lambda x: jaccard(x, _a["lower"])).idxmin()]
    _c_pool = _t[
        ~_t["event"].str.startswith("4")
        & _t["lower"].str.contains("ladder")
        & ~_t["lower"].str.contains(r"\bfell\b|\bfall")
    ]
    _c = _c_pool.loc[_c_pool["lower"].map(lambda x: jaccard(x, _a["lower"])).idxmax()]
    reports = {"A": _a["text"], "B": _b["text"], "C": _c["text"]}
    report_events = {"A": _a["event"], "B": _b["event"], "C": _c["event"]}
    _slide = {
        "A": "An employee was working from a ladder while installing roof trusses on a residential structure when they fell 9 feet off the ladder, resulting in neck, back, and leg injuries.",
        "B": "An employee was dismantling scaffolding. As he knelt down to lower the frame, he was pulled to the ground by the weight of the frame, resulting in a laceration to the side of his head.",
        "C": "An employee was setting up an extension ladder to access the roof of a house when the ladder contacted a power line, resulting in an electric shock and hospitalization.",
    }
    assert reports == _slide
    # A and B: falls to a lower level (OIICS 43xx); C: contact with electric current (51xx)
    assert report_events == {"A": "4332", "B": "4330", "C": "5122"}
    mo.md("\n".join(f"- **{_k}.** {_v}" for _k, _v in reports.items()))
    return report_events, reports


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: a proposed mix attends to every tested mix

    ### Assume: similar keys, similar values

    Lecture 2's question again: predict a proposed mix's 28-day strength from the tested ones.
    The **query** is the proposed mix (row 638, Lectures 5 and 10); the **keys** are the 1,028
    tests of other designs (427 mixes, some tested at several ages), each described by Lecture
    2's six mix groups, standardized; the
    **values** are their strengths. The assumption is Lecture 2's: mixes with similar keys have
    similar values.

    Score every key by its similarity to the query and turn the scores into weights with
    Lecture 6's softmax:

    $$s_i = -\frac{\lVert q - k_i \rVert^2}{\tau}, \qquad w_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad \hat y = \sum_i w_i v_i .$$

    Every tested mix gets a weight; $\tau$, the temperature, is chosen on whole held-out
    designs: a small $\tau$ gives all the weight to the nearest mix (1-NN), a large one spreads
    it evenly (the mean). Step through the picture's three frames with the slider.
    """)
    return


@app.cell
def _(
    CONCRETE_COLUMNS,
    CONCRETE_CSV,
    GroupKFold,
    GroupShuffleSplit,
    INGREDIENTS,
    K,
    KNeighborsRegressor,
    REPEATS,
    SPLIT_SEED,
    StandardScaler,
    TAUS,
    TEST_SIZE,
    np,
    pd,
    require,
):
    def load_concrete(tol=1.0):
        """Lecture 2's 1,030 tests with ``mix`` and ``design`` ids for grouped splits.

        Mixes whose seven proportions all agree within ``tol`` kg/m³ share a design: the data
        combine several laboratories' records, and a design often appears twice, once rounded.
        """
        df = pd.read_csv(require(CONCRETE_CSV, "run `bash scripts/fetch_concrete.sh`"))
        assert df.shape == (1030, 9), df.shape
        df.columns = CONCRETE_COLUMNS
        df["mix"] = df.groupby(INGREDIENTS, sort=False).ngroup()
        mixes = df.groupby("mix")[INGREDIENTS].first().to_numpy()
        parent = np.arange(len(mixes))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = int(parent[a])
            return a

        near = np.abs(mixes[:, None, :] - mixes[None, :, :]).max(axis=-1) <= tol
        for a, b in np.argwhere(np.triu(near, 1)):
            parent[find(int(a))] = find(int(b))
        _, design_of_mix = np.unique([find(i) for i in range(len(mixes))], return_inverse=True)
        df["design"] = design_of_mix[df["mix"].to_numpy()]
        assert df["mix"].nunique() == 428 and df["design"].nunique() == 315
        return df

    def mix_groups(df):
        """Lecture 2's encoder: w/b, slag/b, fly ash/b, SP/b, aggregate/b, ln age; (N, 6)."""
        binder = df["cement"] + df["slag"] + df["fly_ash"]
        z = np.column_stack([df["water"] / binder, df["slag"] / binder, df["fly_ash"] / binder,
                             df["superplasticizer"] / binder,
                             (df["coarse_agg"] + df["fine_agg"]) / binder, np.log(df["age"])])  # fmt: skip
        assert z.shape == (len(df), 6), z.shape
        return z

    def grouped_splits(df):
        """Lecture 2's 30 splits: a quarter of the designs held out each time."""
        gss = GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        return list(gss.split(df, groups=df["design"]))

    def knn_predict(x, y, tr, te, k=K):
        """Lecture 2's k-NN (uniform weights, Euclidean), standardized on ``tr`` only."""
        scaler = StandardScaler().fit(x[tr])
        model = KNeighborsRegressor(n_neighbors=k, weights="uniform").fit(
            scaler.transform(x[tr]), y[tr]
        )
        return model.predict(scaler.transform(x[te]))

    def attend(q, keys, values, tau):
        """Cross-attention with a distance score: softmax(-|q - k|² / tau) weights on the values.

        q (m, d), keys (n, d), values (n,) -> weights (m, n) and attended values (m,).
        """
        d2 = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(-1)  # (m, n)
        s = -d2 / tau
        w = np.exp(s - s.max(axis=1, keepdims=True))  # subtract the largest: no overflow
        w /= w.sum(axis=1, keepdims=True)
        return w, w @ values

    def choose_tau(z, y, groups):
        """The temperature with the lowest error over 5 folds of whole designs (no twin leaks)."""
        err = {}
        for tau in TAUS:
            sq = []
            for a, b in GroupKFold(n_splits=5).split(z, y, groups):
                _, pred = attend(z[b], z[a], y[a], tau)
                sq.append((pred - y[b]) ** 2)
            err[tau] = float(np.sqrt(np.mean(np.concatenate(sq))))
        tau = min(err, key=err.get)
        if tau in (TAUS[0], TAUS[-1]):
            raise ValueError(f"temperature {tau} is at the edge of TAUS; widen the grid")
        return tau

    return attend, choose_tau, grouped_splits, knn_predict, load_concrete, mix_groups


@app.cell
def _(
    EXACT,
    PROPOSED_ROW,
    TOP_KEYS,
    attend,
    choose_tau,
    knn_predict,
    load_concrete,
    mix_groups,
    np,
    shown,
):
    conc = load_concrete()
    xc = mix_groups(conc)
    yc = conc["strength"].to_numpy(float)
    _q = PROPOSED_ROW
    _design = conc["design"].to_numpy()
    keys = np.flatnonzero(_design != _design[_q])  # every test but the proposed design's
    _mu, _sd = xc[keys].mean(0), xc[keys].std(0)
    _z = (xc - _mu) / _sd
    tau_pic = choose_tau(_z[keys], yc[keys], _design[keys])
    _w, _pred = attend(_z[[_q]], _z[keys], yc[keys], tau_pic)
    w_pic = _w[0]
    top = np.argsort(w_pic)[::-1][:TOP_KEYS]
    picture = {
        "pred": float(_pred[0]),
        "knn5": float(knn_predict(xc, yc, keys, np.array([_q]))[0]),
        "tested": float(yc[_q]),
        "share": float(w_pic[top].sum()),
    }
    print(
        f"tau {tau_pic}; {len(keys):,} keys ({conc['mix'].iloc[keys].nunique()} mixes); top "
        f"{TOP_KEYS} weights {w_pic[top].round(3)} carry {100 * picture['share']:.1f}%; attention "
        f"{picture['pred']:.2f}, 5-NN {picture['knn5']:.2f}, tested {picture['tested']:.2f} MPa"
    )
    assert np.isclose(w_pic.sum(), 1.0, rtol=0, atol=EXACT)
    assert tau_pic == 0.2 and len(keys) == 1028 and conc["mix"].iloc[keys].nunique() == 427
    assert np.allclose(w_pic[top], [0.09565311718410417, 0.09460330723446443, 0.09405696336351288,
                                    0.09149389293258964, 0.0728157234284458, 0.0603139251343548,
                                    0.05801349105726131, 0.057533367463739785], rtol=0, atol=EXACT)  # fmt: skip
    assert np.allclose(yc[keys[top]], [37.42475728, 37.437857324, 37.42475728, 22.489328168,
                                       31.35047372, 39.604880392, 36.93522932, 33.082437432], rtol=0, atol=EXACT)  # fmt: skip
    assert abs(picture["share"] - 0.6244837877984728) < EXACT and shown(
        100 * picture["share"], "62"
    )
    assert abs(picture["pred"] - 35.259446089025275) < EXACT and shown(picture["pred"], "35.3")
    assert abs(picture["knn5"] - 33.225434754400005) < EXACT and shown(picture["knn5"], "33.2")
    assert abs(picture["tested"] - 38.21075992) < EXACT and shown(picture["tested"], "38.2")
    # The alt texts: the proposed mix sits at w/b near 0.5 and 28 days; 37.4 MPa for the three
    # largest weights; the largest weights are about 0.1 each; the circled keys sit beside the star
    assert all(shown(yc[keys[_i]], "37.4") for _i in top[:3])
    # The notes: 1,028 tests of other designs from 427 mixes, some tested at several ages
    assert conc.iloc[keys].groupby("mix")["age"].nunique().max() > 1
    assert shown(xc[_q, 0], "0.50") and conc["age"][_q] == 28
    assert shown(yc[keys[top[0]]], "37.4") and all(0.09 < _v < 0.1 for _v in w_pic[top[:4]])
    assert (conc["age"].to_numpy()[keys[top]] == 28).all()
    assert np.abs(xc[keys[top], 0] - xc[_q, 0]).max() < 0.03
    return conc, keys, picture, tau_pic, top, w_pic, xc, yc


@app.cell
def _(mo):
    frame = mo.ui.slider(1, 3, step=1, value=3, label="Frame of the build", show_value=True)
    frame
    return (frame,)


@app.cell
def _(PALETTE, PROPOSED_ROW, TOP_KEYS, frame, keys, np, picture, plt, top, w_pic, xc, yc):
    _fig, (_a, _b, _c) = plt.subplots(
        1, 3, figsize=(10, 4), width_ratios=(1.2, 1, 0.8), layout="constrained"
    )
    _q = PROPOSED_ROW
    _a.scatter(xc[keys, 0], np.exp(xc[keys, 5]), s=6, c="#C8C8C8", lw=0)
    _a.scatter(xc[keys[top], 0], np.exp(xc[keys[top], 5]), s=40, facecolor="none",
               edgecolor=PALETTE[0], lw=1.4, label="largest weights")  # fmt: skip
    _a.scatter([xc[_q, 0]], [np.exp(xc[_q, 5])], marker="*", s=220, color=PALETTE[1],
               label="query: the proposed mix")  # fmt: skip
    _a.set_yscale("log")
    _a.set_xlabel("water / binder")
    _a.set_ylabel("age (days)")
    _a.legend(fontsize=8, frameon=False, loc="upper right")
    _a.set_title(f"keys: {len(keys):,} tests of other designs", fontsize=11)
    for _ax in (_b, _c):
        _ax.set_axis_off()
    if frame.value >= 2:
        _b.set_axis_on()
        _b.barh(np.arange(TOP_KEYS), w_pic[top], color=PALETTE[0])
        _b.set_yticks(np.arange(TOP_KEYS), [f"{yc[keys[_i]]:.1f} MPa" for _i in top], fontsize=9)
        _b.invert_yaxis()
        _b.set_xlabel("weight")
        _b.set_title(f"softmax weights (top {TOP_KEYS})", fontsize=11)
    if frame.value >= 3:
        _c.text(0.0, 0.85, "attention", fontsize=12, fontweight="bold")
        _c.text(0.0, 0.70, f"{picture['pred']:.1f} MPa", fontsize=15)
        _c.text(0.0, 0.48, "5-NN (Lecture 2)", fontsize=12, color="#666666")
        _c.text(0.0, 0.33, f"{picture['knn5']:.1f} MPa", fontsize=15, color="#666666")
        _c.text(0.0, 0.11, f"tested: {picture['tested']:.1f} MPa", fontsize=12)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The proposed mix (w/b 0.50, 28 days) sits among tested mixes like it: the eight largest
    weights all go to 28-day tests within 0.03 of its water–binder ratio, and they carry 62% of
    the weight; $\tau = 0.2$. Attention's output, $\sum_i w_i v_i$, is 35.3 MPa; Lecture 2's
    5-NN says 33.2; the test said 38.2. This is **cross-attention**: the query comes from one set
    (new mixes), keys and values from another (tested mixes). It is a soft nearest-neighbour
    average, the instance-based regression of Lecture 2 with every neighbour weighted.

    *Counting note:* the 1,028 keys are tests, not distinct mixes: they come from 427 mixes, many
    tested at several ages.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    ### Text becomes tokens

    A model reads integers, so text is cut into **tokens** first. Characters: a tiny vocabulary,
    every text covered, but long sequences and little meaning per token. Words: short sequences,
    but the vocabulary grows without end and a new word has no integer. **Subwords** sit in
    between: frequent words stay whole, rare ones split into pieces that exist. The tokenizers
    and the word vectors learn from the reports of 2015–2022; 2023–2025 is held out.
    """)
    return


@app.cell
def _(
    BPE_VOCAB,
    Tokenizer,
    WORD,
    itertools,
    normalizers,
    pre_tokenizers,
    tk_models,
    trainers,
):
    def merge_pair(word, pair):
        """``word`` (a tuple of symbols) with every adjacent ``pair`` fused, left to right."""
        out, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def bpe_scratch(texts, merges):
        """Byte-pair encoding's first ``merges`` merges, written out (Sennrich et al., 2016).

        Start from characters; repeatedly fuse the most frequent adjacent pair of symbols,
        counting over the corpus's words (a word never merges across a space). Ties go to the
        smaller pair.
        """
        counts = {}
        for t in texts:
            for w in WORD.findall(t):
                counts[tuple(w)] = counts.get(tuple(w), 0) + 1
        done = []
        for _ in range(merges):
            pairs = {}
            for w, c in counts.items():
                for a, b in itertools.pairwise(w):
                    pairs[(a, b)] = pairs.get((a, b), 0) + c
            best = max(pairs, key=lambda p: (pairs[p], tuple(-ord(ch) for ch in p[0] + "|" + p[1])))
            done.append(best)
            merged = {}
            for w, c in counts.items():
                key = merge_pair(w, best)
                merged[key] = merged.get(key, 0) + c
            counts = merged
        return done

    def bpe_library(texts, vocab=BPE_VOCAB):
        """The same algorithm in Hugging Face ``tokenizers`` (lowercase, the same pre-tokenizer)."""
        tok = Tokenizer(tk_models.BPE(unk_token="[UNK]"))  # noqa: S106 - a token, not a password
        tok.normalizer = normalizers.Lowercase()
        tok.pre_tokenizer = pre_tokenizers.Whitespace()  # \w+|[^\w\s]+, as WORD
        trainer = trainers.BpeTrainer(
            vocab_size=vocab, min_frequency=2, show_progress=False, special_tokens=["[UNK]"]
        )
        tok.train_from_iterator(texts.tolist(), trainer=trainer)
        return tok

    def pair_counts(words):
        """Adjacent symbol pairs and their counts, most frequent first."""
        c = {}
        for w in words:
            for pair in itertools.pairwise(w):
                c[pair] = c.get(pair, 0) + 1
        return [(f"{a} + {b}", n) for (a, b), n in sorted(c.items(), key=lambda kv: -kv[1])]

    return bpe_library, bpe_scratch, merge_pair, pair_counts


@app.cell
def _(
    AutoTokenizer,
    BERT,
    BERT_REVISION,
    BPE_VOCAB,
    CHECK_MERGES,
    EXACT,
    LETTERS,
    SEED,
    SHOW_MERGES,
    TRAIN_YEARS,
    W2V_MIN,
    WORD,
    bpe_library,
    bpe_scratch,
    json,
    np,
    osha_df,
    pd,
    shown,
    training,
):
    _train = training(osha_df)
    _test = osha_df.loc[osha_df["year"] > TRAIN_YEARS[1], "lower"]
    word_vocab = pd.Series([_w for _t in _train for _w in LETTERS.findall(_t)]).value_counts()
    _test_words = [_w for _t in _test for _w in LETTERS.findall(_t)]
    _unseen = float(np.mean([_w not in word_vocab.index for _w in _test_words]))
    scratch_merges = bpe_scratch(_train, CHECK_MERGES)
    bpe = bpe_library(_train)
    _lib_merges = [tuple(_m) for _m in json.loads(bpe.to_str())["model"]["merges"][:CHECK_MERGES]]
    _same = next(
        (
            _i
            for _i, (_a, _b) in enumerate(zip(scratch_merges, _lib_merges, strict=True))
            if _a != _b
        ),
        CHECK_MERGES,
    )
    bert_tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    _sample = _test.sample(2000, random_state=SEED)
    per_report = {
        "characters": float(np.median(_sample.str.len())),
        "words": float(np.median([len(WORD.findall(_t)) for _t in _sample])),
        "bpe_osha": float(np.median([len(bpe.encode(_t).tokens) for _t in _sample])),
        "bert_wordpiece": float(np.median([len(bert_tok.tokenize(_t)) for _t in _sample])),
    }
    _letters_only = float(np.median([len(LETTERS.findall(_t)) for _t in _sample]))
    assert _letters_only == 34.0
    print(
        f"vocabulary: {len(word_vocab):,} words, {int((word_vocab >= W2V_MIN).sum()):,} seen 5+ times"
    )
    print(f"words of 2023-2025 never seen in 2015-2022: {100 * _unseen:.2f}%")
    print(f"first {SHOW_MERGES} merges: {['+'.join(_m) for _m in scratch_merges[:SHOW_MERGES]]}")
    print(f"identical to tokenizers' first merges: {_same} of {CHECK_MERGES}")
    print(
        f"median tokens per report (2,000 of 2023-2025): {per_report}; letter-words only {_letters_only}"
    )
    assert len(word_vocab) == 15_041 and int((word_vocab >= W2V_MIN).sum()) == 6_783
    assert abs(_unseen - 0.002072198512869625) < EXACT and shown(100 * _unseen, "0.2")
    assert [" + ".join(_m) for _m in scratch_merges[:SHOW_MERGES]] == [
        "h + e", "i + n", "a + n", "t + he", "in + g", "l + o",
        "e + r", "e + e", "m + p", "e + d", "e + mp", "a + s",
    ]  # fmt: skip
    assert _same == CHECK_MERGES == 50  # our 50 merges are the library's 50
    # "the" becomes one token at merge 4 (h + e, then t + he); "ing" at merge 5 (i + n, in + g)
    assert scratch_merges[0] == ("h", "e") and scratch_merges[3] == ("t", "he")
    assert scratch_merges[1] == ("i", "n") and scratch_merges[4] == ("in", "g")
    assert bpe.get_vocab_size() == BPE_VOCAB == 8000 and bert_tok.vocab_size == 30_522
    assert len(_sample) == 2000
    # The slide's table: 196.5 | 38 | 39 | 41 (the characters' median of an even sample)
    assert per_report == {
        "characters": 196.5,
        "words": 38.0,
        "bpe_osha": 39.0,
        "bert_wordpiece": 41.0,
    }
    assert shown(per_report["characters"], "196.5")
    return bert_tok, bpe, per_report, scratch_merges, word_vocab


@app.cell
def _(mo, per_report):
    mo.md(rf"""
    | One report, median of 2,000 (2023–2025) | Tokens |
    |:-----------------------------------------|-------:|
    | Characters | {per_report["characters"]:.1f} |
    | Words and punctuation marks | {per_report["words"]:.0f} |
    | Subwords learned from the reports (8,000) | {per_report["bpe_osha"]:.0f} |
    | BERT's subwords (30,522, books and Wikipedia) | {per_report["bert_wordpiece"]:.0f} |

    15,041 different words in the training years' 79,307 reports; in 2023–2025, 0.2% of the
    words are new to that vocabulary. The "words" row counts what the pre-tokenizer cuts: runs of
    letters and digits, and runs of punctuation (letter-words alone: a median of 34).

    ### Byte pairs: merge the most frequent

    Byte-pair encoding (Sennrich et al., 2016): start from characters; count every adjacent pair
    of symbols over the corpus (never across a space); fuse the most frequent pair into a new
    symbol; repeat until the vocabulary has the size you want. "the" becomes one token at merge
    4 (h + e, then t + he); "ing" at merge 5. `bpe_scratch` above is the whole algorithm, and its
    first 50 merges are identical to Hugging Face's `tokenizers` trained on the same 79,307
    reports. WordPiece (BERT's tokenizer; Wu et al., 2016) picks merges by likelihood instead of
    count; the idea is the same.

    ### Your turn: two merges by hand

    "the worker held the ladder there, then the ladder fell"

    Count the adjacent letter pairs inside the words. Which pair merges first? Then which?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        h + e first (6 times; next t + h, 5); then t + he (5 times, ahead of e + r with 3).
        After the first merge, t + h no longer exists: every t + h was followed by an e, which
        is now part of "he". The same arithmetic, over 8 years of reports, built the merges
        above. Checked below.
        """)
        }
    )
    return


@app.cell
def _(TRY_SENTENCE, WORD, bpe_scratch, merge_pair, pair_counts, pd):
    _words = [tuple(_w) for _w in WORD.findall(TRY_SENTENCE)]
    _first = pair_counts(_words)
    _merges = bpe_scratch(pd.Series([TRY_SENTENCE]), 2)
    _merged = [merge_pair(_w, _merges[0]) for _w in _words]
    _second = pair_counts(_merged)
    print("first counts:", _first[:3], "\nafter h + e:", _second[:3], "\nmerges:", _merges)
    assert _first[:3] == [("h + e", 6), ("t + h", 5), ("e + r", 4)]
    assert _second[:3] == [("t + he", 5), ("e + r", 3), ("l + a", 2)]
    assert [" + ".join(_m) for _m in _merges] == ["h + e", "t + he"]
    assert "t + h" not in dict(_second)  # no t + h is left once "he" is fused
    return


@app.cell
def _(DOMAIN_WORDS, bert_tok, bpe, mo):
    _splits = {
        _w: {"bert": bert_tok.tokenize(_w), "osha": bpe.encode(_w).tokens} for _w in DOMAIN_WORDS
    }
    _want = {
        "scaffold": ["sc", "##af", "##fold"], "excavator": ["ex", "##ca", "##vat", "##or"],
        "trench": ["trench"], "forklift": ["fork", "##lift"], "amputated": ["amp", "##uta", "##ted"],
        "lacerated": ["lace", "##rated"], "skid": ["ski", "##d"], "rebar": ["re", "##bar"],
        "drywall": ["dry", "##wall"], "trencher": ["trench", "##er"],
    }  # fmt: skip
    assert {_w: _s["bert"] for _w, _s in _splits.items()} == _want
    assert all(_s["osha"] == [_w] for _w, _s in _splits.items())  # ours keeps every one whole
    _rows = "\n".join(
        f"| {_w} | {', '.join(_s['bert'])} | {', '.join(_s['osha'])} |"
        for _w, _s in _splits.items()
    )
    mo.md(
        "### BERT's vocabulary splits the job site\n\n"
        "| Word | BERT (books, Wikipedia) | Learned from the reports |\n"
        "|:-----|:------------------------|:-------------------------|\n" + _rows + "\n\n"
        '"##" marks a piece that continues a word. A vocabulary reflects the text it was learned '
        "from: BERT's (Devlin et al., 2019; 30,522 pieces) from books and Wikipedia, ours from "
        'injury reports (8,000). "lace + rated" and "re + bar" are not wrong: the model still '
        "sees the whole sequence. But every split spends tokens and asks the model to rebuild a "
        "word from pieces whose other uses mean something else (§5.4)."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Words become vectors

    Word2vec's skip-gram with negative sampling (Mikolov et al., 2013): each word $w$ gets a
    vector $u_w$ that should predict the words $c$ within 5 positions of it ($\sigma$ of the dot
    product high), and not 5 random words $n_k$ (drawn by frequency$^{0.75}$):

    $$\max \sum_{(w, c)} \Big[\log \sigma(u_w \cdot v_c) + \sum_{k=1}^{5} \log \sigma(-u_w \cdot v_{n_k})\Big]$$

    Words used in the same contexts end up with similar vectors. Trained here from scratch on
    the training years, on the CPU: every word seen at least 5 times, 100 numbers each, 3
    passes, with Adam on minibatches of 4,096 pairs; very frequent words are subsampled.
    """)
    return


@app.cell
def _(
    F,
    LETTERS,
    SEED,
    W2V_BATCH,
    W2V_DIM,
    W2V_EPOCHS,
    W2V_LR,
    W2V_MIN,
    W2V_NEG,
    W2V_SUBSAMPLE,
    W2V_WINDOW,
    nn,
    np,
    osha_df,
    pd,
    seed_all,
    time,
    torch,
    training,
):
    def word2vec(df):
        """Skip-gram with negative sampling, written out, on the CPU.

        Returns the vocabulary (words seen at least W2V_MIN times), unit-length vectors, and the
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
        keep = np.minimum(1.0, np.sqrt(W2V_SUBSAMPLE / freq) + W2V_SUBSAMPLE / freq)  # subsample
        neg_p = torch.tensor(freq**0.75 / (freq**0.75).sum())  # negatives: frequency^0.75
        centre = nn.Embedding(len(vocab), W2V_DIM)  # u_w
        context = nn.Embedding(len(vocab), W2V_DIM)  # v_c
        nn.init.uniform_(centre.weight, -0.5 / W2V_DIM, 0.5 / W2V_DIM)
        nn.init.zeros_(context.weight)
        opt = torch.optim.Adam([*centre.parameters(), *context.parameters()], lr=W2V_LR)
        t0 = time.perf_counter()
        for _ in range(W2V_EPOCHS):
            pairs = []
            for d in docs:
                ids = np.array([index[w] for w in d if w in index])
                ids = ids[rng.random(len(ids)) < keep[ids]]
                for off in range(1, W2V_WINDOW + 1):  # (centre, context) both ways, 1-5 apart
                    pairs.append(np.stack([ids[:-off], ids[off:]], 1))
                    pairs.append(np.stack([ids[off:], ids[:-off]], 1))
            pairs = torch.tensor(np.concatenate(pairs))[
                torch.tensor(rng.permutation(sum(len(p) for p in pairs)))
            ]
            for i in range(0, len(pairs), W2V_BATCH):
                b = pairs[i : i + W2V_BATCH]
                neg = torch.multinomial(neg_p, len(b) * W2V_NEG, replacement=True).view(
                    len(b), W2V_NEG
                )
                u = centre(b[:, 0])  # (batch, dim)
                pos = (u * context(b[:, 1])).sum(-1)  # (batch,)
                negs = torch.einsum("bd,bkd->bk", u, context(neg))  # (batch, neg)
                loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-negs).sum(-1).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        vec = F.normalize(centre.weight.detach(), dim=1).numpy()
        return vocab, vec, {"vocab": len(vocab), "pairs_per_epoch": len(pairs), "loss": float(loss.detach()),
                            "minutes": (time.perf_counter() - t0) / 60}  # fmt: skip

    w2v_vocab, w2v_vec, w2v_numbers = word2vec(osha_df)
    print(w2v_numbers)
    return w2v_numbers, w2v_vec, w2v_vocab


@app.cell
def _(EXACT, QUERY_WORDS, mo, np, w2v_numbers, w2v_vec, w2v_vocab):
    def neighbours(vocab, vec, words, k=4):
        """The ``k`` nearest words (cosine) to each word in ``words``."""
        index = {w: i for i, w in enumerate(vocab)}
        out = {}
        for w in words:
            sim = vec @ vec[index[w]]
            order = [j for j in np.argsort(sim)[::-1] if j != index[w]][:k]
            out[w] = [(vocab[j], float(sim[j])) for j in order]
        return out

    _nb = neighbours(w2v_vocab, w2v_vec, QUERY_WORDS)
    _want = {
        "ladder": [("rung", 0.9465038776397705), ("rungs", 0.9373140931129456),
                   ("climbing", 0.9246635437011719), ("extension", 0.9238770008087158)],
        "scaffold": [("baker", 0.9374390840530396), ("plastering", 0.9195448756217957),
                     ("stucco", 0.9130181670188904), ("bakers", 0.9124215841293335)],
        "trench": [("excavation", 0.8931899666786194), ("caved", 0.8816922903060913),
                   ("soil", 0.8678572177886963), ("buried", 0.8056724667549133)],
        "forklift": [("standup", 0.8911076188087463), ("backing", 0.8791340589523315),
                     ("forklifts", 0.876851499080658), ("reversing", 0.8759035468101501)],
        "amputated": [("matcher", 0.9459586143493652), ("middle", 0.9388402104377747),
                      ("amputating", 0.935362696647644), ("fingertips", 0.9343288540840149)],
        "finger": [("fingertip", 0.9414889812469482), ("amputating", 0.9262454509735107),
                   ("amputation", 0.9241602420806885), ("ring", 0.9206337332725525)],
    }  # fmt: skip
    for _w, _ref in _want.items():
        assert [_n for _n, _ in _nb[_w]] == [_n for _n, _ in _ref], (_w, _nb[_w])
        assert np.allclose([_s for _, _s in _nb[_w]], [_s for _, _s in _ref], rtol=0, atol=EXACT)
    assert w2v_numbers["vocab"] == 6_783 and w2v_numbers["pairs_per_epoch"] == 1_853_654
    assert abs(w2v_numbers["loss"] - 2.257361888885498) < EXACT
    # the slides' run: 5.348325533300007 min (numbers.json), on a CPU shared with other jobs
    print(f"word2vec trained in {w2v_numbers['minutes']:.1f} min (the slides: 5.3 min)")
    _rows = "\n".join(
        f"| {_w} | {', '.join(f'{_n} ({_s:.2f})' for _n, _s in _nb[_w][:3])} |"
        for _w in QUERY_WORDS
    )
    mo.md(
        "| Word | Its nearest words in the reports (cosine) |\n|:-----|:------|\n" + _rows + "\n\n"
        '"baker" is a Baker scaffold, a rolling platform. BERT starts the same way, with a '
        "learned table of one vector per token (30,522 × 768, checked in §5.2), learned with the "
        "rest of the model rather than taken from here. ViT's tokens are patches, not words, so "
        "its first layer projects each patch's 16 × 16 × 3 pixels to 768 numbers (checked in §4) "
        "instead of looking a vector up."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Attention: a soft nearest-neighbour average

    The one picture (§2) on one mix; now on Lecture 2's 30 splits by design. In each split the
    temperature is chosen by 5 folds of whole designs inside the training rows, and the test
    designs are predicted once, against Lecture 2's 5-NN on the same split.
    """)
    return


@app.cell
def _(EXACT, attend, choose_tau, conc, grouped_splits, knn_predict, np, shown, xc, yc):
    _groups = conc["design"].to_numpy()
    _knn, _att, _taus = [], [], []
    for _tr, _te in grouped_splits(conc):
        _mu, _sd = xc[_tr].mean(0), xc[_tr].std(0)
        _z = (xc - _mu) / _sd
        _tau = choose_tau(_z[_tr], yc[_tr], _groups[_tr])
        _, _pa = attend(_z[_te], _z[_tr], yc[_tr], _tau)
        _knn.append(float(np.sqrt(np.mean((knn_predict(xc, yc, _tr, _te) - yc[_te]) ** 2))))
        _att.append(float(np.sqrt(np.mean((_pa - yc[_te]) ** 2))))
        _taus.append(_tau)
    concrete_att = {
        "knn": float(np.median(_knn)),
        "attention": float(np.median(_att)),
        "better": int(sum(_a < _k for _a, _k in zip(_att, _knn, strict=True))),
    }
    print(
        f"median RMSE over {len(_knn)} splits: attention {concrete_att['attention']:.3f}, 5-NN "
        f"{concrete_att['knn']:.3f} MPa; attention lower in {concrete_att['better']}; "
        f"temperatures chosen {sorted(set(_taus))}, median {np.median(_taus)}"
    )
    assert len(_knn) == 30 and concrete_att["better"] == 25
    assert abs(concrete_att["knn"] - 6.707673579012678) < EXACT and shown(
        concrete_att["knn"], "6.71"
    )
    assert abs(concrete_att["attention"] - 6.558275802990389) < EXACT and shown(
        concrete_att["attention"], "6.56"
    )
    assert sorted(set(_taus)) == [0.2, 0.5] and np.median(_taus) == 0.2  # inside the grid
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Self-attention: tokens query each other

    $$Q = XW_Q,\quad K = XW_K,\quad V = XW_V, \qquad \text{Attention}(X) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)\,V$$

    In **self-attention** every token is a query, a key, and a value at once, each by its own
    learned matrix ($W_Q, W_K, W_V$, learned by backpropagation, Lecture 14). Similarity is a
    scaled dot product instead of a distance; $\sqrt{d}$ keeps the scores from growing with the
    dimension $d$, so the softmax does not saturate. Each output token is a weighted average of
    all tokens' values, weighted by how well its query matches their keys. **Multi-head:** do
    this $h$ times with smaller matrices, side by side, so different heads can match on
    different things (ViT-B/16: 12 heads of 64 numbers). Vaswani et al. (2017).

    The code below is the slide's two equations, then one pre-norm transformer block: attend,
    add the input back, then an MLP on each token alone, and add again. §4 checks it against
    torchvision.
    """)
    return


@app.cell
def _(F, torch):
    def vit_tokens(model, x):
        """Patch tokens, plus the class token, plus position embeddings: (n, 197, 768)."""
        t = model._process_input(x)  # (n, 196, 768): one token per 16 x 16 patch
        cls = model.class_token.expand(t.shape[0], -1, -1)  # (n, 1, 768)
        out = torch.cat([cls, t], dim=1) + model.encoder.pos_embedding
        assert out.shape == (x.shape[0], 197, 768), out.shape
        return out

    def self_attention(x, mha):
        """Multi-head self-attention written out with ``mha``'s weights (batch_first).

        x (n, t, d) -> the output (n, t, d) and the attention weights (n, heads, t, t).
        """
        n, t, d = x.shape
        h = mha.num_heads
        qkv = x @ mha.in_proj_weight.T + mha.in_proj_bias  # (n, t, 3d): queries, keys, values
        q, k, v = qkv.split(d, dim=-1)
        q, k, v = (z.view(n, t, h, d // h).transpose(1, 2) for z in (q, k, v))  # (n, h, t, d/h)
        a = torch.softmax(q @ k.transpose(-2, -1) / (d // h) ** 0.5, dim=-1)  # (n, h, t, t)
        out = (a @ v).transpose(1, 2).reshape(n, t, d)  # heads side by side again
        assert out.shape == x.shape and a.shape == (n, h, t, t)
        return out @ mha.out_proj.weight.T + mha.out_proj.bias, a

    def block(x, blk):
        """One pre-norm transformer block: x + attention(norm x), then + MLP(norm x)."""
        a_out, a = self_attention(
            F.layer_norm(x, (x.shape[-1],), blk.ln_1.weight, blk.ln_1.bias, blk.ln_1.eps),
            blk.self_attention,
        )
        x = x + a_out  # attend, then add back
        y = F.layer_norm(x, (x.shape[-1],), blk.ln_2.weight, blk.ln_2.bias, blk.ln_2.eps)
        lin1, lin2 = blk.mlp[0], blk.mlp[3]
        return x + F.gelu(y @ lin1.weight.T + lin1.bias) @ lin2.weight.T + lin2.bias, a

    return block, self_attention, vit_tokens


@app.cell
def _(F, SEED, np, torch):
    # The formula against PyTorch's fused kernel, on random queries, keys, values in float64
    _g = torch.Generator().manual_seed(SEED)
    _q, _k, _v = (torch.randn(1, 4, 10, 8, dtype=torch.float64, generator=_g) for _ in range(3))
    _ours = torch.softmax(_q @ _k.transpose(-2, -1) / 8**0.5, dim=-1) @ _v
    _gap = float((_ours - F.scaled_dot_product_attention(_q, _k, _v)).abs().max())
    print(f"softmax(QK^T / sqrt(d)) V against F.scaled_dot_product_attention: {_gap:.1e}")
    assert _ours.shape == (1, 4, 10, 8) and _gap < 1e-14

    # Why sqrt(d): dot products of random unit-variance vectors grow like sqrt(d), and an
    # unscaled softmax over 197 keys then puts most of the weight on one of them
    def _top_weight(s):
        e = np.exp(s - s.max(1, keepdims=True))
        return float((e / e.sum(1, keepdims=True)).max(1).mean())

    _rng = np.random.default_rng((SEED, 3))
    for _d in (4, 64, 768):
        _s = _rng.normal(size=(2000, _d)) @ _rng.normal(size=(197, _d)).T
        _raw, _scaled = _top_weight(_s), _top_weight(_s / np.sqrt(_d))
        print(
            f"d = {_d:3d}: score sd {_s.std():5.2f}, scaled {(_s / np.sqrt(_d)).std():.2f}; "
            f"largest weight {_raw:.2f} unscaled, {_scaled:.2f} scaled"
        )
        assert abs(_s.std() / np.sqrt(_d) - 1) < 0.1 and _scaled < 0.1
        if _d >= 64:
            assert _raw > 0.5  # saturated: one key takes most of the weight
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Order needs a position

    Attention is a weighted sum over all tokens: shuffle the tokens and the outputs shuffle with
    them; nothing in it knows order (checked below on ViT's first block). So a **position
    embedding** is added to each token: fixed sines and cosines in the original transformer,
    learned numbers in ViT. ViT-B/16 learned its 196 from ImageNet alone. Each map below: how
    similar every patch's position embedding is to that of one patch (the square).
    """)
    return


@app.cell
def _(EXACT, F, GRID, PALETTE, SHOW_POSITIONS, np, plt, shown, vit):
    vit32 = vit()  # float32, on the CPU: positions here, attention maps in §6
    _pe = F.normalize(
        vit32.encoder.pos_embedding.detach()[0, 1:], dim=1
    )  # (196, 768), no class token
    _sim = (_pe @ _pe.T).numpy()  # (196, 196) cosine similarity
    assert _sim.shape == (GRID * GRID, GRID * GRID)
    _rows, _cols = np.divmod(np.arange(GRID * GRID), GRID)
    _dist = np.hypot(_rows[:, None] - _rows[None, :], _cols[:, None] - _cols[None, :])
    _off = ~np.eye(len(_sim), dtype=bool)
    positions = {
        "neighbour": float(_sim[_dist == 1].mean()),
        "far": float(_sim[(_dist >= 8) & _off].mean()),
        "corr": float(np.corrcoef(_sim[_off], _dist[_off])[0, 1]),
    }
    print(
        f"cosine similarity of position embeddings: neighbours {positions['neighbour']:.3f}, 8 or "
        f"more patches apart {positions['far']:.3f}; correlation with distance {positions['corr']:.3f}"
    )
    assert abs(positions["neighbour"] - 0.7313395142555237) < EXACT and shown(
        positions["neighbour"], "0.73"
    )
    assert abs(positions["far"] - -0.10481199622154236) < EXACT and shown(positions["far"], "-0.10")
    assert abs(positions["corr"] - -0.3236628843251102) < EXACT
    _fig, _axes = plt.subplots(1, len(SHOW_POSITIONS), figsize=(10, 2.9), layout="constrained")
    for _ax, (_r, _c) in zip(_axes, SHOW_POSITIONS, strict=True):
        _i = _r * GRID + _c
        _ax.imshow(_sim[_i].reshape(GRID, GRID), cmap="cividis", vmin=-0.2, vmax=1.0)
        _ax.scatter([_c], [_r], marker="s", s=40, facecolor="none", edgecolor=PALETTE[1], lw=1.6)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_title(f"patch (row {_r}, col {_c})", fontsize=10)
        # The alt text: each map lights up the patch's own row and column, brightest near it
        _line = (_rows == _r) | (_cols == _c)
        _same, _other = _sim[_i][_line & (np.arange(len(_sim)) != _i)], _sim[_i][~_line]
        _nearest = np.argmax(np.where(np.arange(len(_sim)) == _i, -np.inf, _sim[_i]))
        assert _same.mean() > _other.mean() + 0.4 and _dist[_i, _nearest] == 1
    _fig
    return positions, vit32


@app.cell
def _(mo):
    mo.md(r"""
    Neighbouring patches' position embeddings have cosine similarity 0.73 on average, patches 8
    or more apart −0.10, and the row-and-column pattern of the image grid appears on its own: for
    each patch shown, the patches in its own row and column are on average more than 0.4 more
    similar than the rest, and the most similar patch is a neighbour (Dosovitskiy et al., 2021,
    showed the same).
    """)
    return


@app.cell
def _(SEED, VERIFY_TILE, block, np, tile_tensor, torch, vit, vit_tokens):
    vit64 = vit().double()  # the same weights in float64, for exact checks (here and §4)
    with torch.no_grad():
        x0 = vit_tokens(vit64, tile_tensor(VERIFY_TILE).double())  # (1, 197, 768) with positions
        _perm = torch.from_numpy(np.random.default_rng(SEED).permutation(x0.shape[1]))
        _shuffled_out, _ = block(x0[:, _perm], vit64.encoder.layers[0])
        _out, _ = block(x0, vit64.encoder.layers[0])
    _gap = float((_shuffled_out - _out[:, _perm]).abs().max())
    print(f"block(shuffled tokens) against shuffled block(tokens): {_gap:.1e}")
    assert _gap < 1e-12  # permutation-equivariant: order reaches the block only through positions
    return vit64, x0


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### One block: attend, then think

    A transformer alternates two steps: **attention**, where tokens exchange information, and an
    **MLP**, applied to each token alone, where each token processes what it gathered (Lecture
    14's network, 768 → 3,072 → 768, two-thirds of the block's weights). Each step sits inside a
    residual path (Lecture 15's skips) after a layer norm. ViT-B/16 cuts the 224-pixel tile into
    16-pixel patches, giving 196 tokens of 768 numbers, plus a learned **class token** that
    gathers the tile's summary; Lecture 15's probe reads it. Twelve identical blocks.

    ### Your turn: predict the check

    ```python
    def block(x, blk):                     # x: (1, 197, 768) tokens
        a, _ = self_attention(norm(x, blk.ln_1), blk.self_attention)
        x = x + a                          # attend, then add back
        y = norm(x, blk.ln_2)
        h = gelu(y @ blk.mlp[0].weight.T + blk.mlp[0].bias)
        return x + h @ blk.mlp[3].weight.T + blk.mlp[3].bias
    ```

    Against torchvision's first ViT block, on a real crack tile (Lecture 15's filter tile, cut
    and embedded by ViT-B/16's own patch layer), in float64: how close? Write a number.
    """)
    return


@app.cell
def _(BLOCK_TOL, backbone, block, mo, shown, torch, vit64, x0):
    _blk = vit64.encoder.layers[0]
    with torch.no_grad():
        _ours, _ = block(x0, _blk)
        _ref = _blk(x0)
    verify = {
        "max_abs": float((_ours - _ref).abs().max()),
        "tokens": int(x0.shape[1]),
        "dim": int(x0.shape[2]),
        "heads": int(_blk.self_attention.num_heads),
        "block_weights": int(sum(_p.numel() for _p in _blk.parameters())),
        "mlp_weights": int(sum(_p.numel() for _p in _blk.mlp.parameters())),
        "blocks": len(vit64.encoder.layers),
        "vit_weights": int(sum(_p.numel() for _p in vit64.parameters())),
        "resnet_weights": int(sum(_p.numel() for _p in backbone().parameters())),
    }
    print(verify)
    assert verify["max_abs"] < BLOCK_TOL  # round-off; below 1e-15 on the slides' run
    assert (verify["tokens"], verify["dim"], verify["heads"], verify["blocks"]) == (
        197,
        768,
        12,
        12,
    )
    assert (_blk.mlp[0].in_features, _blk.mlp[0].out_features, _blk.mlp[3].out_features) == (
        768,
        3072,
        768,
    )
    assert verify["block_weights"] == 7_087_872 and shown(verify["block_weights"] / 1e6, "7.09")
    assert abs(verify["mlp_weights"] / verify["block_weights"] - 2 / 3) < 0.01  # two-thirds
    assert verify["vit_weights"] == 85_798_656 and shown(verify["vit_weights"] / 1e6, "85.8")
    assert verify["resnet_weights"] == 11_176_512
    # ViT's first layer: each 16 x 16 x 3 patch projected to 768 numbers, not a lookup table
    _proj = vit64.conv_proj
    assert (_proj.in_channels, _proj.out_channels, _proj.kernel_size, _proj.stride) == (
        3,
        768,
        (16, 16),
        (16, 16),
    )
    _below = "below" if verify["max_abs"] < 1e-15 else "not below"
    mo.md(f"""
    | Quantity, one tile | Value |
    |:-------------------|------:|
    | Tokens × numbers | {verify["tokens"]} × {verify["dim"]} |
    | Heads | {verify["heads"]} |
    | Largest difference, all outputs | {verify["max_abs"]:.1e} ({_below} 10⁻¹⁵) |

    Float64 round-off: the order of additions differs between our code and PyTorch's fused
    kernel, so the last bits can differ, and they differ between runs with different thread
    counts too (so the check is below 10⁻¹⁴, not a digit). The block is the whole architecture:
    ViT-B/16 is this function applied {verify["blocks"]} times with different weights,
    {verify["block_weights"]:,} weights each; {verify["vit_weights"]:,} in all against
    ResNet-18's {verify["resnet_weights"]:,}.
    """)
    return (verify,)


@app.cell
def _(mo):
    mo.md(r"""
    ### BERT: learn by filling blanks

    BERT (Devlin et al., 2019) is a stack of transformer blocks over text tokens, pretrained by
    hiding 15% of the tokens of books and Wikipedia and predicting them from both sides. No
    labels: the text is its own answer key. Test it on reports it never saw: every report of
    2023–2025 with "fell from/off a/an/the ___", the word replaced by `[MASK]`. Only answers
    that are one WordPiece can be guessed in one step; the others count as misses. The baseline
    always guesses the training years' most common answer.
    """)
    return


@app.cell
def _(
    AutoModelForMaskedLM,
    BARE,
    BERT,
    BERT_REVISION,
    DEVICE,
    EXACT,
    FELL,
    GPU,
    GUESS_TOL,
    MLM_TOL,
    TRAIN_YEARS,
    bert_tok,
    osha_df,
    pd,
    shown,
    torch,
    training,
):
    _seen = training(osha_df).str.extract(FELL)[1].dropna().value_counts()
    _test = osha_df.loc[osha_df["year"] > TRAIN_YEARS[1], "lower"]
    _mlm = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval().to(DEVICE)
    _rows = [
        (_t[: _m.start(2)] + bert_tok.mask_token + _t[_m.end(2) :], _m.group(2))
        for _t in _test
        if (_m := FELL.search(_t)) is not None
    ]
    _top1 = _top5 = _single = 0
    _guesses = {}
    with torch.no_grad():
        for _i in range(0, len(_rows), 64):
            _batch = _rows[_i : _i + 64]
            _enc = bert_tok([_r[0] for _r in _batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to(DEVICE)  # fmt: skip
            _logits = _mlm(**_enc).logits  # (batch, tokens, 30,522)
            _pos = (_enc["input_ids"] == bert_tok.mask_token_id).nonzero()  # one [MASK] each
            assert len(_pos) == len(_batch)
            for (_b, _j), (_, _answer) in zip(_pos.tolist(), _batch, strict=True):
                _best = bert_tok.convert_ids_to_tokens(_logits[_b, _j].topk(5).indices.tolist())
                _guesses[_best[0]] = _guesses.get(_best[0], 0) + 1
                _single += len(bert_tok.tokenize(_answer)) == 1
                _top1 += _best[0] == _answer
                _top5 += _answer in _best
        # The bare sentence, prepared as the reports were (lowercased), on the same model
        _enc = bert_tok(
            BARE.lower().replace("[mask]", bert_tok.mask_token), return_tensors="pt"
        ).to(DEVICE)
        _j = int((_enc["input_ids"][0] == bert_tok.mask_token_id).nonzero()[0, 0])
        _bare = bert_tok.convert_ids_to_tokens(_mlm(**_enc).logits[0, _j].topk(50).indices.tolist())
    del _mlm
    _answers = pd.Series([_r[1] for _r in _rows])
    masked = {
        "cases": len(_rows),
        "single": _single,
        "top1": _top1 / len(_rows),
        "top5": _top5 / len(_rows),
        "base": _seen.index[0],
        "base_top1": float((_answers == _seen.index[0]).mean()),
        "base_top5": float(_answers.isin(_seen.index[:5]).mean()),
        "answers": _answers.value_counts(),
        "guesses": dict(sorted(_guesses.items(), key=lambda kv: -kv[1])),
        "bare": _bare,  # BERT's 50 first guesses for the bare sentence, best first
    }
    print(
        f"{masked['cases']:,} blanks ({masked['single']} are one WordPiece): BERT right "
        f"{100 * masked['top1']:.1f}%, in its top five {100 * masked['top5']:.1f}%; always "
        f"'{masked['base']}' {100 * masked['base_top1']:.1f}% (top five {100 * masked['base_top5']:.1f}%)"
    )
    # From the data and the tokenizer alone: exact
    assert masked["cases"] == 1102 and masked["single"] == 956 and masked["base"] == "ladder"
    assert abs(masked["base_top1"] - 0.2159709618874773) < EXACT and shown(
        100 * masked["base_top1"], "22"
    )
    assert abs(masked["base_top5"] - 0.38656987295825773) < EXACT and shown(
        100 * masked["base_top5"], "39"
    )
    assert masked["answers"].head(8).to_dict() == {"ladder": 238, "roof": 73, "trailer": 49, "truck": 43,
                                                   "top": 28, "forklift": 27, "lift": 24, "back": 23}  # fmt: skip
    # From BERT on the GPU: within MLM_TOL, and the slide's digits on a GPU
    assert abs(masked["top1"] - 0.46098003629764067) <= MLM_TOL
    assert abs(masked["top5"] - 0.6234119782214156) <= MLM_TOL
    if GPU:
        assert shown(100 * masked["top1"], "46") and shown(100 * masked["top5"], "62")
    _want = {"ladder": 231, "roof": 124, "top": 64, "trailer": 60, "truck": 57, "building": 41,
             "side": 38, "ceiling": 30}  # fmt: skip
    assert list(masked["guesses"])[:2] == ["ladder", "roof"]
    assert all(abs(masked["guesses"].get(_w, 0) - _n) <= GUESS_TOL for _w, _n in _want.items())
    # The notes: BERT beats the frequency baseline (22%) by a factor of two
    assert 2 < masked["top1"] / masked["base_top1"] < 2.5
    # The bare sentence: building first, ladder 8th
    assert masked["bare"][:8] == [
        "building",
        "window",
        "roof",
        "tree",
        "car",
        "balcony",
        "bridge",
        "ladder",
    ]
    return (masked,)


@app.cell
def _(PALETTE, masked, np, plt):
    _ans, _gs = masked["answers"].head(8).to_dict(), dict(list(masked["guesses"].items())[:8])
    _words = list(dict.fromkeys([*_ans, *_gs]))[:10]
    _counts = masked["answers"].to_dict()
    _y = np.arange(len(_words))
    _fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _ax.barh(_y - 0.2, [_counts.get(_w, 0) for _w in _words], height=0.38, color=PALETTE[0],
             label="what the report said")  # fmt: skip
    _ax.barh(_y + 0.2, [masked["guesses"].get(_w, 0) for _w in _words], height=0.38,
             color=PALETTE[1], hatch="//", label="BERT's first guess")  # fmt: skip
    _ax.set_yticks(_y, _words)
    _ax.invert_yaxis()
    _ax.set_xlabel(f'"fell from a ___": reports of 2023-2025 (of {masked["cases"]:,})')
    _ax.legend(frameon=False, loc="lower right")
    print({_w: (_counts.get(_w, 0), masked["guesses"].get(_w, 0)) for _w in _words})
    # The alt text: ladder about as often as the reports; roof more often; and building, which the
    # reports almost never say (1 of 1,102). The notes: "building" and "side" far more often
    # than the reports say them (side 17 times)
    assert _counts["building"] == 1 and _counts["side"] == 17
    assert abs(masked["guesses"]["ladder"] / _counts["ladder"] - 1) < 0.1
    assert masked["guesses"]["roof"] > 1.5 * _counts["roof"]
    assert all(masked["guesses"][_w] > 2 * _counts.get(_w, 0) for _w in ("building", "side"))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    BERT's first guess is right 46% of the time on the 1,102 blanks, in its top five 62%;
    always guessing "ladder" (the most common answer in 2015–2022) is right 22% (top five 39%).
    Of the answers, 956 are one BERT token; the other 146 cannot be guessed in one step. BERT
    guesses "ladder" about as often as the reports say it, "roof" more often, and "building",
    which the reports almost never say, 41 times (the reports: once); "side" 38 times against
    the reports' 17.

    ## 5. Break it

    ### Your turn: predict four failures

    1. Cut each tile into 32-pixel blocks, shuffle them, refit both probes.
    2. Which of reports A, B, C does each representation call alike?
    3. BERT, given only "The employee fell from a [MASK] and fractured his leg."
    4. A tokenizer from books, on injury reports.

    Write what happens to each, and why. Each subsection below reveals one.

    ### 5.1 Shuffled tiles: both readers lose

    Each tile's 224-pixel input is cut into 49 blocks of 32 × 32 pixels, and the blocks are put
    back in a fixed random order, the same for every tile; both probes are refitted on shuffled
    tiles. A block is 2 × 2 of ViT's 16-pixel patches and one cell of ResNet-18's last 7 × 7
    feature map, so the cut is aligned to both grids, but the seams differ for the two: ViT's
    patch embedding never reads across a seam between blocks, while ResNet-18's filters cross
    the seams at every layer, so part of ResNet-18's larger loss may be the seams only its
    filters see (a test that isolates the seams was not run). Three permutations (seeds (17, 0),
    (17, 1), (17, 2)): in the slides' setting each network embeds every tile four times, as
    taken and under each permutation, about 25 minutes on the slides' GPU while other jobs
    shared it; reduced,
    every 64th tile under the first permutation.
    """)
    return


@app.cell
def _(
    BLOCK, BLOCKS, DEVICE, DataLoader, Dataset, SEED, VERIFY_TILE, WORKERS, np, tile_tensor, torch
):
    def shuffle_blocks(x, perm, block=BLOCK):
        """An image (3, s, s) cut into ``block``-pixel squares, put back in the order ``perm``."""
        c, s, _ = x.shape
        g = s // block
        p = x.unfold(1, block, block).unfold(2, block, block)  # (3, g, g, block, block)
        p = p.reshape(c, g * g, block, block)[:, torch.as_tensor(perm)]
        p = p.reshape(c, g, g, block, block).permute(0, 1, 3, 2, 4)  # rows of pixels again
        return p.reshape(c, s, s)

    class Shuffled(Dataset):
        """Tiles with their 32 x 32 blocks (of the 224-px input) in a fixed shuffled order."""

        def __init__(self, paths, perm):
            self.paths, self.perm = paths, perm

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            x = tile_tensor(self.paths[i])[0]  # (3, 224, 224)
            return x if self.perm is None else shuffle_blocks(x, self.perm)

    def embed(model, paths, perm):
        """Features of ``paths`` under ``model``, blocks in the order ``perm`` (None: as taken)."""
        model = model.eval().to(DEVICE)
        loader = DataLoader(
            Shuffled(paths, perm), batch_size=256, num_workers=WORKERS, shuffle=False
        )
        with torch.inference_mode():
            out = [model(_x.to(DEVICE)).float().cpu().numpy() for _x in loader]
        return np.concatenate(out)

    # The identity permutation reassembles a tile exactly; otherwise block k of the output is
    # block perm[k] of the input
    _x = tile_tensor(VERIFY_TILE)[0]
    assert torch.equal(Shuffled([VERIFY_TILE], np.arange(BLOCKS**2))[0], _x)
    _perm = np.random.default_rng((SEED, 0)).permutation(BLOCKS**2)
    _y = shuffle_blocks(_x, _perm)
    for _k in range(BLOCKS**2):
        (_r, _c), (_r0, _c0) = divmod(_k, BLOCKS), divmod(int(_perm[_k]), BLOCKS)
        assert torch.equal(_y[:, BLOCK * _r : BLOCK * (_r + 1), BLOCK * _c : BLOCK * (_c + 1)],
                           _x[:, BLOCK * _r0 : BLOCK * (_r0 + 1), BLOCK * _c0 : BLOCK * (_c0 + 1)])  # fmt: skip
    return (embed,)


@app.cell
def _(
    BLOCKS,
    EXACT,
    FEAT_TOL,
    REFERENCE_GPU,
    SEED,
    SHUFFLE_EVERY,
    SHUFFLE_PERMS,
    SLIDES,
    backbone,
    embed,
    np,
    os,
    part,
    rfeat,
    screen,
    tiles,
    time,
    vfeat,
    vit,
    y_tile,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # the tokenizers ran before these forks
    _perms = [
        np.random.default_rng((SEED, _i)).permutation(BLOCKS**2) for _i in range(SHUFFLE_PERMS)
    ]
    _rows = np.arange(0, len(tiles), SHUFFLE_EVERY)  # every tile ("slides") or every 64th
    _sub = tiles.iloc[_rows].reset_index(drop=True)
    _paths, _y, _part = _sub["path"].tolist(), y_tile[_rows], part[_rows]
    shuffle = {"tiles": len(_rows), "permutations": len(_perms)}
    for _name, _make, _cached in (("resnet18", backbone, rfeat), ("vit_b_16", vit, vfeat)):
        _model = _make()
        _t0 = time.perf_counter()
        _plain = embed(_model, _paths, None)
        _shuffled = [embed(_model, _paths, _p) for _p in _perms]
        _seconds = time.perf_counter() - _t0
        _base = screen(_plain, _y, _part, _sub).numbers["all"]
        _runs = [screen(_f, _y, _part, _sub).numbers["all"] for _f in _shuffled]
        shuffle[_name] = {
            "plain": _base,
            "shuffled": _runs,
            "ap_drop": [_base["ap"] - _r["ap"] for _r in _runs],
            "cache_gap": float(np.abs(_plain - _cached[_rows]).max()),
            "ms_per_tile": 1e3 * _seconds / ((1 + len(_perms)) * len(_paths)),
        }
        del _model
        print(
            f"{_name}: {_seconds:.0f} s for {(1 + len(_perms)) * len(_paths):,} passes; cache "
            f"agrees within {shuffle[_name]['cache_gap']:.1e}; AP {_base['ap']:.4f} as taken, "
            f"shuffled {[round(_r['ap'], 4) for _r in _runs]}; tiles to review "
            f"{100 * _base['review']:.1f}% -> {[round(100 * _r['review'], 1) for _r in _runs]}%"
        )
    # Fresh embeddings against the caches, on any device (TF32 off): float32 round-off
    assert all(shuffle[_n]["cache_gap"] < 1e-3 for _n in ("resnet18", "vit_b_16"))
    if SLIDES:
        # Every permutation costs both networks average precision and adds tiles to review (the
        # reduced sample's 877 tiles are too few to test that: its numbers are printed only)
        for _n in ("resnet18", "vit_b_16"):
            assert all(_d > 0 for _d in shuffle[_n]["ap_drop"]), _n
            assert all(
                _r["review"] > shuffle[_n]["plain"]["review"] for _r in shuffle[_n]["shuffled"]
            )
        _ref = (  # shuffle: from numbers.json
            {'resnet18': ((0.6647861327136738, 0.5022305496074233),
                          [(0.5326332474149454, 0.5988579586009993),
                           (0.5467533897914412, 0.5791399000713776),
                           (0.5384893036083364, 0.574768022840828)]),
             'vit_b_16': ((0.7668746687857423, 0.41059957173447537),
                          [(0.7016694399737103, 0.5053533190578159),
                           (0.7055565249769926, 0.48545681655960027),
                           (0.7009730242473222, 0.4988401142041399)])}
        )  # fmt: skip
        for _n, (_plain_ref, _runs_ref) in _ref.items():
            _got = [shuffle[_n]["plain"], *shuffle[_n]["shuffled"]]
            for _g, (_ap, _rev) in zip(_got, [_plain_ref, *_runs_ref], strict=True):
                _tol = EXACT if REFERENCE_GPU else FEAT_TOL
                assert abs(_g["ap"] - _ap) <= _tol and abs(_g["review"] - _rev) <= _tol, (_n, _g)
    # Timing: assert only the order (ResNet-18 is the faster reader)
    assert shuffle["resnet18"]["ms_per_tile"] < shuffle["vit_b_16"]["ms_per_tile"]
    return (shuffle,)


@app.cell
def _(SLIDES, mo, shown, shuffle):
    _nets = (("resnet18", "ResNet-18"), ("vit_b_16", "ViT-B/16"))
    _rng = lambda _v, _f: f"{_f(min(_v))}–{_f(max(_v))}" if len(_v) > 1 else _f(_v[0])  # noqa: E731
    _ap, _pc = (lambda _x: f"{_x:.2f}"), (lambda _x: f"{100 * _x:.0f}")
    _rows = "\n".join(
        f"| {_label}, tiles as taken | {_ap(shuffle[_n]['plain']['ap'])} | "
        f"{_pc(shuffle[_n]['plain']['review'])}% |\n"
        f"| {_label}, blocks shuffled | {_rng([_r['ap'] for _r in shuffle[_n]['shuffled']], _ap)} | "
        f"{_rng([_r['review'] for _r in shuffle[_n]['shuffled']], _pc)}% |"
        for _n, _label in _nets
    )
    _drop = {_n: shuffle[_n]["ap_drop"] for _n, _ in _nets}
    if SLIDES:  # the slide's table and notes (the ranges are over the three permutations)
        # every permutation: ResNet-18 loses about twice the ViT's average precision, and the
        # shuffled ViT still ranks above the intact ResNet-18
        assert all(1.5 < _r / _v < 2.5
                   for _r, _v in zip(_drop["resnet18"], _drop["vit_b_16"], strict=True))  # fmt: skip
        assert all(
            _r["ap"] > shuffle["resnet18"]["plain"]["ap"] for _r in shuffle["vit_b_16"]["shuffled"]
        )
        _want = {"resnet18": ("0.66", "50", ("0.53", "0.55"), ("57", "60"), ("0.12", "0.13")),
                 "vit_b_16": ("0.77", "41", ("0.70", "0.71"), ("49", "51"), ("0.06", "0.07"))}  # fmt: skip
        for _n, (_a, _v, _as, _vs, _ds) in _want.items():
            _sh = shuffle[_n]["shuffled"]
            assert shown(shuffle[_n]["plain"]["ap"], _a) and shown(
                100 * shuffle[_n]["plain"]["review"], _v
            )
            assert shown(min(_r["ap"] for _r in _sh), _as[0]) and shown(
                max(_r["ap"] for _r in _sh), _as[1]
            )
            assert shown(100 * min(_r["review"] for _r in _sh), _vs[0])
            assert shown(100 * max(_r["review"] for _r in _sh), _vs[1])
            assert shown(min(_drop[_n]), _ds[0]) and shown(max(_drop[_n]), _ds[1])
    _what = "the three permutations" if len(_drop["resnet18"]) > 1 else "one permutation"
    mo.md(
        "| Test photos, probe refitted | Average precision | Tiles to review |\n"
        "|:----------------------------|------------------:|----------------:|\n" + _rows + "\n\n"
        f"Ranges over {_what}, on {shuffle['tiles']:,} tiles. Both lose; neither is order-blind. "
        f"ResNet-18's average precision falls by {_rng(_drop['resnet18'], _ap)}, the ViT's by "
        f"{_rng(_drop['vit_b_16'], _ap)} (on the slides' run: ResNet-18 loses about twice as much "
        "under each permutation, one photo split, and the shuffled ViT still ranks above the "
        "intact ResNet-18). These are the measured drops; which features each network loses is "
        'not measured here. "Tiles as taken" re-embeds every tile, and its features agree with '
        f"the cache to within {max(shuffle[_n]['cache_gap'] for _n, _ in _nets):.0e}; yet the "
        "refitted ResNet-18 probe lands in the third decimal away from the cached screen of §1 "
        "(0.665 against 0.666 on the slides' run): a probe's fit moves with differences that "
        "small.\n\n"
        "### 5.2 Similar to what?\n\n"
        "The attempt's three reports, compared three ways: the share of words they have in "
        "common; the cosine of their mean word2vec vectors; and the cosine of BERT's mean output "
        "token (the [CLS] and [SEP] tokens dropped)."
    )
    return


@app.cell
def _(
    AutoModelForMaskedLM,
    BERT,
    BERT_REVISION,
    EMB_TOL,
    EXACT,
    LETTERS,
    bert_tok,
    jaccard,
    mo,
    np,
    reports,
    shown,
    torch,
    w2v_vec,
    w2v_vocab,
):
    # BERT on the CPU, as the slides' script computed these vectors
    _bert_model = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval()
    # Its first layer: a learned table of one vector per token
    assert tuple(_bert_model.base_model.embeddings.word_embeddings.weight.shape) == (30_522, 768)
    _index = {_w: _i for _i, _w in enumerate(w2v_vocab)}

    def _w2v(x):
        v = w2v_vec[[_index[w] for w in LETTERS.findall(x.lower()) if w in _index]].mean(0)
        return v / np.linalg.norm(v)

    def _bert(x):
        enc = bert_tok(x, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            h = _bert_model.base_model(**enc).last_hidden_state[0]  # (tokens, 768)
        v = h[1:-1].mean(0).numpy()  # the mean of the word tokens ([CLS], [SEP] dropped)
        return v / np.linalg.norm(v)

    _pairs = (("A", "B"), ("A", "C"), ("B", "C"))
    _e = {
        "w2v": {_k: _w2v(_v) for _k, _v in reports.items()},
        "bert": {_k: _bert(_v) for _k, _v in reports.items()},
    }
    similar = {
        "shared": {_x + _y: jaccard(reports[_x].lower(), reports[_y].lower()) for _x, _y in _pairs},
        "w2v": {_x + _y: float(_e["w2v"][_x] @ _e["w2v"][_y]) for _x, _y in _pairs},
        "bert": {_x + _y: float(_e["bert"][_x] @ _e["bert"][_y]) for _x, _y in _pairs},
    }
    print(similar)
    _want = {
        "shared": ({"AB": 0.15555555555555556, "AC": 0.275, "BC": 0.225}, EXACT, ("0.16", "0.28", "0.23")),
        "w2v": ({"AB": 0.937378466129303, "AC": 0.9655182957649231, "BC": 0.9466134905815125}, EXACT, ("0.94", "0.97", "0.95")),
        "bert": ({"AB": 0.8684964179992676, "AC": 0.9352075457572937, "BC": 0.8723939657211304}, EMB_TOL, ("0.87", "0.94", "0.87")),
    }  # fmt: skip
    for _how, (_ref, _tol, _texts) in _want.items():
        for (_pair, _v), _text in zip(_ref.items(), _texts, strict=True):
            assert abs(similar[_how][_pair] - _v) <= _tol and shown(similar[_how][_pair], _text), (
                _how,
                _pair,
            )
        assert max(similar[_how], key=similar[_how].get) == "AC"  # every one calls A and C alike
    assert shown(min(similar["w2v"].values()), "0.94") and shown(
        max(similar["w2v"].values()), "0.97"
    )
    _rows = "\n".join(
        f"| {_label} | {similar['shared'][_p]:.2f} | {similar['w2v'][_p]:.2f} | {similar['bert'][_p]:.2f} |"
        for _p, _label in (
            ("AB", "A, B: both falls"),
            ("AC", "A, C: both ladders at a house"),
            ("BC", "B, C"),
        )
    )
    mo.md(
        "| Pair | Shared words | Word vectors (mean) | BERT (mean) |\n"
        "|:-----|-------------:|--------------------:|------------:|\n" + _rows
    )
    return


@app.cell
def _(
    AutoModelForMaskedLM,
    BERT,
    BERT_REVISION,
    DEVICE,
    EXACT,
    F,
    LETTERS,
    REFERENCE_GPU,
    SEED,
    SIM_PAIRS,
    SIM_REPORTS,
    SIM_TOL,
    SLIDES,
    TRAIN_YEARS,
    bert_tok,
    np,
    osha_df,
    roc_auc_score,
    torch,
    w2v_vec,
    w2v_vocab,
):
    # The measurement behind the attempt: every report of 2023-2025 with a classifiable event
    # (OIICS division 9, nonclassifiable, left out) and at least one word word2vec knows
    _index = {_w: _i for _i, _w in enumerate(w2v_vocab)}
    _t = osha_df[(osha_df["year"] > TRAIN_YEARS[1]) & ~osha_df["event"].str.startswith("9")]
    _ids = [[_index[_w] for _w in LETTERS.findall(_x) if _w in _index] for _x in _t["lower"]]
    _t = _t[[len(_w) > 0 for _w in _ids]]
    _ids = [_w for _w in _ids if _w]
    _n_all = len(_t)
    if SIM_REPORTS is not None:  # reduced: a random sample of the reports
        _keep = np.sort(
            np.random.default_rng((SEED, 4)).choice(len(_t), SIM_REPORTS, replace=False)
        )
        _t, _ids = _t.iloc[_keep], [_ids[_k] for _k in _keep]
    _sets = [set(LETTERS.findall(_x)) for _x in _t["lower"]]
    _w2v = np.stack([w2v_vec[_w].mean(0) for _w in _ids])
    _w2v /= np.linalg.norm(_w2v, axis=1, keepdims=True)
    # BERT's mean output token of each report, in batches: [CLS], [SEP], and padding left out
    _model = AutoModelForMaskedLM.from_pretrained(BERT, revision=BERT_REVISION).eval().to(DEVICE)
    _texts, _out = _t["text"].tolist(), []
    with torch.no_grad():
        for _i in range(0, len(_texts), 128):
            _enc = bert_tok(_texts[_i : _i + 128], return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to(DEVICE)  # fmt: skip
            _h = _model.base_model(**_enc).last_hidden_state  # (batch, tokens, 768)
            _m = _enc["attention_mask"].clone()
            _m[:, 0] = 0  # [CLS]
            _m[torch.arange(len(_m)), _enc["attention_mask"].sum(1) - 1] = 0  # [SEP]
            _v = (_h * _m[..., None]).sum(1) / _m.sum(1, keepdim=True)
            _out.append(F.normalize(_v, dim=1).float().cpu().numpy())
    del _model
    _bert = np.concatenate(_out)
    assert _bert.shape == (len(_t), 768) and _w2v.shape == (len(_t), _w2v.shape[1])
    # Random pairs of different reports; same event division (first digit), group (two digits)
    _i, _j = np.random.default_rng((SEED, 19)).integers(0, len(_t), size=(2, SIM_PAIRS))
    _i, _j = _i[_i != _j], _j[_i != _j]
    _ev = _t["event"].to_numpy()
    _same = {
        "division": np.array([_ev[_a][0] == _ev[_b][0] for _a, _b in zip(_i, _j, strict=True)]),
        "group": np.array([_ev[_a][:2] == _ev[_b][:2] for _a, _b in zip(_i, _j, strict=True)]),
    }
    _sim = {
        "shared_words": np.array([len(_sets[_a] & _sets[_b]) / len(_sets[_a] | _sets[_b])
                                  for _a, _b in zip(_i, _j, strict=True)]),
        "word2vec": np.einsum("nd,nd->n", _w2v[_i], _w2v[_j]),
        "bert": np.einsum("nd,nd->n", _bert[_i], _bert[_j]),
    }  # fmt: skip
    similar_all = {
        "reports": len(_t),
        "pairs": len(_i),
        "same_division_share": float(_same["division"].mean()),
        "auc_division": {
            _k: float(roc_auc_score(_same["division"], _v)) for _k, _v in _sim.items()
        },
        "auc_group": {_k: float(roc_auc_score(_same["group"], _v)) for _k, _v in _sim.items()},
        "median": {_k: float(np.median(_v)) for _k, _v in _sim.items()},
    }
    print(
        f"{similar_all['reports']:,} of {_n_all:,} reports, {similar_all['pairs']:,} pairs "
        f"({100 * similar_all['same_division_share']:.1f}% of the same division):"
    )
    for _k in _sim:
        print(f"  {_k:12s} AUC, same division {similar_all['auc_division'][_k]:.3f}, same group "
              f"{similar_all['auc_group'][_k]:.3f}; median {similar_all['median'][_k]:.3f}")  # fmt: skip
    _ref = (  # sim_all: from numbers.json
        {'reports': 26599,
         'pairs': 199991,
         'same_division_share': 0.32331954937972207,
         'auc_division': {'shared_words': 0.6376439075596868,
                          'word2vec': 0.7877246675841449,
                          'bert': 0.6649013576097044},
         'auc_group': {'shared_words': 0.640781342709749,
                       'word2vec': 0.7679488300294675,
                       'bert': 0.6617229427040472},
         'median': {'shared_words': 0.16,
                    'word2vec': 0.9339096546173096,
                    'bert': 0.8575646281242371}}
    )  # fmt: skip
    assert _n_all == _ref["reports"]
    for _what in ("auc_division", "auc_group", "median"):
        for _k, _v in _ref[_what].items():
            if not SLIDES:
                _tol = SIM_TOL  # a sample of the reports and pairs
            elif _k != "bert" or REFERENCE_GPU:
                _tol = EXACT  # the CPU's measures, and BERT on the slides' GPU
            else:
                _tol = 1e-3  # BERT on another device
            assert abs(similar_all[_what][_k] - _v) <= _tol, (_what, _k, similar_all[_what][_k])
    return (similar_all,)


@app.cell
def _(SLIDES, mo, shown, similar_all):
    _auc, _med = similar_all["auc_division"], similar_all["median"]
    # The slide's last row and notes: AUC 0.64 / 0.79 / 0.66 over 26,599 reports and 199,991
    # pairs, 32% of one division; median cosine 0.93 (word vectors), 0.86 (BERT), 0.16 (shared)
    if SLIDES:
        _texts = {"shared_words": ("0.64", "0.16"), "word2vec": ("0.79", "0.93"),
                  "bert": ("0.66", "0.86")}  # fmt: skip
        assert all(shown(_auc[_k], _a) and shown(_med[_k], _m) for _k, (_a, _m) in _texts.items())
        assert similar_all["reports"] == 26_599 and similar_all["pairs"] == 199_991
        assert shown(100 * similar_all["same_division_share"], "32")
    # "All three track the event, none closely; word vectors best; BERT about as well as words"
    assert all(0.55 < _v < 0.85 for _v in _auc.values())
    assert (
        _auc["word2vec"] == max(_auc.values()) and abs(_auc["bert"] - _auc["shared_words"]) < 0.05
    )
    mo.md(f"""
    The attempt revealed: every representation calls A and C the closest pair; a safety engineer
    coding the event calls A and B alike (both falls to a lower level; C is an electric shock).
    Three reports are an example, not a measurement. Over {similar_all["reports"]:,} test
    reports here ({similar_all["pairs"]:,} random pairs,
    {100 * similar_all["same_division_share"]:.0f}% of them of the same event division: falls,
    contact with objects, ...), the AUC with which each similarity ranks a same-division pair
    above a pair of different divisions (0.5 is a coin, 1 always): shared words
    {_auc["shared_words"]:.2f}, word vectors {_auc["word2vec"]:.2f}, BERT's mean token
    {_auc["bert"]:.2f}. All three track the event, none closely: the mean word vectors best,
    BERT's mean token about as well as shared words. Averaged vectors are also all alike: two
    random reports have a median cosine of {_med["word2vec"]:.2f} with word vectors and
    {_med["bert"]:.2f} with BERT (shared words: {_med["shared_words"]:.2f}), and the three
    reports differ only in the second decimal. To get your meaning, train for it: a probe on
    frozen features with your labels (Lecture 15, and today's screen), or a model trained to
    embed sentences (Lecture 19).

    ### 5.3 Context fills the blank

    The same BERT, given a bare sentence instead of a report.
    """)
    return


@app.cell
def _(BARE, masked, mo):
    _rank = masked["bare"].index("ladder") + 1
    print(f"{BARE!r}: {masked['bare'][:12]}")
    assert masked["bare"][0] == "building" and _rank == 8  # the slide: building (ladder 8th)
    mo.md(f"""
    | BERT's input | First guess |
    |:-------------|:------------|
    | "{BARE}" | {masked["bare"][0]} (ladder {_rank}th) |
    | The full report, 2023–2025 | right {100 * masked["top1"]:.0f}% of {masked["cases"]:,} |

    A one-line sentence gives BERT only general English: people fall from buildings, windows,
    roofs. A full report gives it the job: the task, the tools, the injury; self-attention lets
    the blank draw on every one of those tokens. The same model, with more context, beats the
    frequency baseline (22%) by a factor of two. Its first guesses still include words the
    reports rarely end the phrase with: "building" {masked["guesses"]["building"]} times against
    the reports' {masked["answers"]["building"]}, "side" {masked["guesses"]["side"]} times
    against {masked["answers"]["side"]} (§4).
    """)
    return


@app.cell
def _(DOMAIN_WORDS, bert_tok, bpe, mo, per_report):
    _split_bert = [_w for _w in DOMAIN_WORDS if len(bert_tok.tokenize(_w)) > 1]
    _split_ours = [_w for _w in DOMAIN_WORDS if len(bpe.encode(_w).tokens) > 1]
    assert set(DOMAIN_WORDS) - set(_split_bert) == {"trench"} and _split_ours == []
    assert per_report["bert_wordpiece"] - per_report["bpe_osha"] == 2
    mo.md(f"""
    ### 5.4 A tokenizer from books, on injury reports

    The vocabulary slide (§3) was the reveal: BERT's WordPiece splits {len(_split_bert)} of our
    {len(DOMAIN_WORDS)} job-site words (only "trench" stays whole); the vocabulary learned from
    the reports keeps all of them whole. A median report costs BERT
    {per_report["bert_wordpiece"]:.0f} tokens against {per_report["bpe_osha"]:.0f}. The cost is
    not the two tokens: it is that "amp + uta + ted" must be rebuilt from pieces whose other uses
    mean something else, by a model that never read an injury report. A model for a field is
    often pretrained, or at least tokenized, on that field's text.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Patches are tokens: the screen

    The tile becomes 197 tokens: 196 patches of 16 × 16 pixels, each flattened and projected to
    768 numbers, plus the class token (Dosovitskiy et al., 2021). ViT-B/16's ImageNet weights,
    frozen; the same probe, the same split, the same threshold rule as Lecture 15; the cached
    features (`scripts/derive_sdnet2018_vit.py`).
    """)
    return


@app.cell
def _(EXACT, close, part, res_screen, screen, shown, tiles, vfeat, y_tile):
    vit_screen = screen(vfeat, y_tile, part, tiles)
    _n = vit_screen.numbers
    print(
        f"ViT-B/16 probe: C = {_n['C']}, validation AP {_n['val_ap']:.4f}, threshold {vit_screen.t:.4f}"
    )
    assert _n["C"] == 0.01 and abs(_n["val_ap"] - 0.8074819761044519) < EXACT
    _want = {  # auc, ap, recall, review, acc, n
        "all": (0.8973519877765442, 0.7668746687857423, 0.877715704531347,
                0.41059957173447537, 0.922733761598858, 11208),
        "deck": (0.8454402369104437, 0.7111627083924632, 0.8431372549019608,
                 0.4963924963924964, 0.8971861471861472, 2772),
        "wall": (0.9016215971723373, 0.816556679431446, 0.9226713532513181,
                 0.5576377058489494, 0.9304372515616127, 3522),
        "pavement": (0.9218783725608156, 0.7604049554365768, 0.8610634648370498,
                     0.2568172568172568, 0.9316239316239316, 4914),
    }  # fmt: skip
    _metrics = ("auc", "ap", "recall", "review", "acc", "n")
    for _k, _v in _want.items():
        close(_n[_k], dict(zip(_metrics, _v, strict=True)), EXACT)
    _r = res_screen.numbers
    # The slides, overall: AP 0.67 against 0.77; flagged 86% against 88%; review 51% against 41%
    assert shown(_r["all"]["ap"], "0.67") and shown(_n["all"]["ap"], "0.77")
    assert shown(100 * _n["all"]["recall"], "88") and shown(100 * _n["all"]["review"], "41")
    # By surface (review / flagged), ViT against ResNet: decks 50/84 against 66/91, walls 56/92
    # against 66/92, pavements 26/86 against 31/77
    _by = {
        "deck": ("50", "84", "66", "91"),
        "wall": ("56", "92", "66", "92"),
        "pavement": ("26", "86", "31", "77"),
    }
    for _s, _texts in _by.items():
        _got = (_n[_s]["review"], _n[_s]["recall"], _r[_s]["review"], _r[_s]["recall"])
        assert all(shown(100 * _g, _t) for _g, _t in zip(_got, _texts, strict=True)), _s
    return (vit_screen,)


@app.cell
def _(PALETTE, plt, res_screen, vit_screen):
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3.4), layout="constrained")
    _rows = [res_screen.numbers["all"], vit_screen.numbers["all"]]
    for _ax, _key, _label in zip(_axes, ("ap", "recall", "review"),
                                 ("average precision", "cracks flagged", "tiles to review"), strict=True):  # fmt: skip
        _vals = [_r[_key] for _r in _rows]
        _ax.bar([0, 1], _vals, color=[PALETTE[0], PALETTE[1]], width=0.6, hatch=["", "//"])
        for _k, _v in enumerate(_vals):
            _ax.text(
                _k, _v + 0.01, f"{_v:.2f}" if _key == "ap" else f"{100 * _v:.0f}%", ha="center"
            )
        _ax.set_xticks([0, 1], ["ResNet-18", "ViT-B/16"])
        _ax.set_ylim(0, 1)
        _ax.set_title(_label, fontsize=11)
    _fig
    return


@app.cell
def _(mo, res_screen, vit_screen):
    _r, _v = res_screen.numbers, vit_screen.numbers
    _rows = "\n".join(
        f"| {_s} | {100 * _r[_s]['recall']:.0f}% / {100 * _r[_s]['review']:.0f}% | "
        f"{100 * _v[_s]['recall']:.0f}% / {100 * _v[_s]['review']:.0f}% |"
        for _s in ("all", "deck", "wall", "pavement")
    )
    mo.md(
        "| Test photos | ResNet-18: flagged / review | ViT-B/16: flagged / review |\n"
        "|:------------|------------------:|------------------:|\n" + _rows + "\n\n"
        "Overall the ViT screen opens fewer tiles at a similar share of cracks flagged, but not "
        "on every surface: on decks it flags fewer cracks (84% against 91%) and reviews fewer "
        "tiles; on pavements it flags more and reviews fewer. One threshold, set to flag 90% of "
        "all cracks, flags different shares on each surface (ResNet-18: decks 91%, pavements "
        "77%): it moves the misses between surfaces; a deck owner sets it on decks.\n\n"
        "### What the class token draws on\n\n"
        "Attention rollout (Abnar & Zuidema, 2020): average each block's attention over its "
        "heads, add the residual path (half and half with the identity), renormalize the rows, "
        "and multiply the 12 blocks in order; the class token's row shows which patches its "
        "summary draws on. Computed with our `block`, the code checked against torchvision. The "
        "first three tiles are the screen's surest catches on each surface; the fourth is its "
        "surest miss."
    )
    return


@app.cell
def _(
    EXACT,
    F,
    GRID,
    SIZE,
    block,
    np,
    part,
    plt,
    read_tile,
    shown,
    tile_tensor,
    tiles,
    torch,
    vit32,
    vit_screen,
    vit_tokens,
    y_tile,
):
    def rollout(model, path):
        """Attention rollout: the class token's share of each of the 14 x 14 patches."""
        with torch.no_grad():
            x = vit_tokens(model, tile_tensor(path))
            roll = torch.eye(x.shape[1])
            for blk in model.encoder.layers:
                x, a = block(x, blk)
                m = 0.5 * a[0].mean(0) + 0.5 * torch.eye(x.shape[1])  # (197, 197): heads, residual
                roll = (m / m.sum(-1, keepdim=True)) @ roll
        return roll[0, 1:].reshape(GRID, GRID).numpy()

    _test = part == "test"
    _picks = []
    for _s in ("deck", "wall", "pavement"):
        _m = np.flatnonzero(_test & (y_tile == 1) & (tiles["surface"] == _s).to_numpy())
        _picks.append(int(_m[np.argmax(vit_screen.p[_m])]))  # the surest catch on each surface
    _missed = np.flatnonzero(_test & (y_tile == 1) & (vit_screen.p < vit_screen.t))
    _picks.append(int(_missed[np.argmin(vit_screen.p[_missed])]))  # the surest miss
    _paths = [tiles["path"].iloc[_i] for _i in _picks]
    _p = vit_screen.p[_picks]
    print(f"tiles {_paths}; p {_p}; threshold {vit_screen.t:.5f}")
    assert _paths == [
        "D/CD/7004-63.jpg",
        "W/CW/7118-98.jpg",
        "P/CP/002-100.jpg",
        "P/CP/048-109.jpg",
    ]
    assert np.allclose(_p, [0.9999994824131885, 0.9999997007223954, 0.9999540267536702,
                            0.0007577621215284619], rtol=0, atol=EXACT)  # fmt: skip
    assert abs(vit_screen.t - 0.04958539365764904) < EXACT and shown(vit_screen.t, "0.050")
    assert shown(_p[3], "0.0008") and (_p[:3] > 0.9999).all()  # "p near 1"; "p = 0.0008"
    _maps = [rollout(vit32, _q) for _q in _paths]
    _fig, _axes = plt.subplots(2, 4, figsize=(10, 5.2), layout="constrained")
    _dark = []
    for _j, (_q, _map) in enumerate(zip(_paths, _maps, strict=True)):
        _axes[0, _j].imshow(read_tile(_q))
        _axes[0, _j].set_title(
            f"{tiles['surface'].iloc[_picks[_j]]}, p = {_p[_j]:.2f}", fontsize=10
        )
        _gray = read_tile(_q, gray=True)
        _axes[1, _j].imshow(_gray, cmap="gray", alpha=0.6, extent=(0, GRID, GRID, 0))
        _axes[1, _j].imshow(_map, cmap="cividis", alpha=0.6, extent=(0, GRID, GRID, 0))
        for _ax in _axes[:, _j]:
            _ax.set_xticks([])
            _ax.set_yticks([])
        # Each patch's mean grey level at the network's input size (low = dark)
        _g = F.interpolate(torch.from_numpy(_gray)[None, None], size=(SIZE, SIZE), mode="bilinear",
                           antialias=True, align_corners=False)  # fmt: skip
        _g = F.avg_pool2d(_g, SIZE // GRID)[0, 0].numpy()
        _top = np.argsort(_map.ravel())[::-1][:5]
        _dark10 = [divmod(int(_k), GRID) for _k in np.argsort(_g.ravel())[:10]]
        _dark.append({"top5": [divmod(int(_k), GRID) for _k in _top], "darkest": _dark10[0], "dark10": _dark10,
                      "corr": float(np.corrcoef(_map.ravel(), -_g.ravel())[0, 1]),
                      "top5_grey": float(_g.ravel()[_top].mean()), "tile_grey": float(_g.mean())})  # fmt: skip
    _axes[1, 0].set_ylabel("attention")
    for _q, _d in zip(_paths, _dark, strict=True):
        print(_q, _d)
    # Deck: the most-attended patch is the tile's darkest (the crack's knot)
    assert _dark[0]["top5"][0] == _dark[0]["darkest"]
    # Wall: the attention runs along the crack, a vertical line: 4 of the top 5 in column 4,
    # the tile's darkest patch among them
    assert (
        sum(_c == 4 for _, _c in _dark[1]["top5"]) == 4 and _dark[1]["darkest"] in _dark[1]["top5"]
    )
    # The miss: the attention sits on the dark band at the right edge (column 13)
    assert sum(_c == GRID - 1 for _, _c in _dark[3]["top5"]) == 4
    assert _dark[3]["top5_grey"] < _dark[3]["tile_grey"] - 10
    assert _dark[3]["darkest"] in _dark[3]["top5"]
    # The caught pavement tile, beside the crack rather than on it: the crack is the row of dark
    # patches (6 of the tile's 10 darkest in row 8); the two most-attended patches are not among
    # the 10 darkest, each sits next to one of them, and the top five are no darker than the tile
    _pav = _dark[2]
    assert sum(_r == 8 for _r, _ in _pav["dark10"]) == 6
    for _r, _c in _pav["top5"][:2]:
        assert (_r, _c) not in _pav["dark10"]
        assert any(abs(_r - _a) + abs(_c - _b) == 1 for _a, _b in _pav["dark10"])
    assert abs(_pav["top5_grey"] - _pav["tile_grey"]) < 5
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    What the checks above can and cannot say (SDNET2018 labels tiles, not pixels): on the wall
    tile the attention runs down one column of patches, along the crack, the tile's darkest patch
    among the five most attended; on the deck tile the most-attended patch is the tile's darkest,
    on the crack; on the missed pavement tile it sits on the dark band at the tile's right edge
    (four of the five most attended in the last column, the tile's darkest patch among them),
    a band the screen did not read as a crack. On the caught pavement tile it sits beside the
    crack rather than on it: the two most-attended patches are each next to one of the crack's
    dark patches, not on one, and the five most attended are no darker than the tile. (The
    fourth, at the right edge, is the tile's second-darkest patch, perhaps the crack's end.) A map is
    evidence of what the model used, not of why; rollout ignores the MLPs and the values (Chefer
    et al., 2021).
    """)
    return


@app.cell
def _(mo, shown, shuffle, tiles, verify):
    _ms_r, _ms_v = shuffle["resnet18"]["ms_per_tile"], shuffle["vit_b_16"]["ms_per_tile"]
    _w_ratio = verify["vit_weights"] / verify["resnet_weights"]
    _deck = tiles["surface"] == "deck"
    _deck_tiles, _deck_photos = int(_deck.sum()), tiles.loc[_deck, "photo"].nunique()
    _deck_s = {"resnet18": _deck_tiles * _ms_r / 1e3, "vit_b_16": _deck_tiles * _ms_v / 1e3}
    _slides = (  # timing: from numbers.json
        {'resnet18_ms': 1.2459518163419039,
         'vit_b_16_ms': 5.374863871318549,
         'ratio': 4.313861740736548,
         'resnet18_deck_seconds': 16.969863738576734,
         'vit_b_16_deck_seconds': 73.20564592735863}
    )  # fmt: skip
    _passes = (1 + shuffle["permutations"]) * shuffle["tiles"]
    print(
        f"time per tile: ResNet-18 {_ms_r:.2f} ms, ViT-B/16 {_ms_v:.2f} ms, ratio {_ms_v / _ms_r:.1f}; "
        f"the deck photos {_deck_s['resnet18']:.0f} s and {_deck_s['vit_b_16']:.0f} s (the slides: "
        f"{_slides['resnet18_ms']:.2f} and {_slides['vit_b_16_ms']:.2f} ms, {_slides['ratio']:.1f} "
        f"times; {_slides['resnet18_deck_seconds']:.0f} s and {_slides['vit_b_16_deck_seconds']:.0f} s)"
    )
    assert _deck_tiles == 13_620 and _deck_photos == 54
    # Timings: only their order is asserted. The ratio depends on the device and on what else
    # shares it: the slides' "about 4 times the time per tile" is the slides' run's ratio (4.3,
    # copied above); this run prints its own
    assert _deck_s["vit_b_16"] > _deck_s["resnet18"]
    assert round(_w_ratio) == 8  # "about 8 times the weights" (7.7)
    assert shown(verify["resnet_weights"] / 1e6, "11") and shown(
        verify["vit_weights"] / 1e6, "86"
    )  # 11 M, 86 M
    mo.md(f"""
    ### CNNs and transformers, side by side

    | Aspect | ResNet-18 (Lecture 15) | ViT-B/16 (today) |
    |:-------|:-----------------------|:-----------------|
    | Position | the index in the feature map | a learned embedding per patch |
    | Locality | built in: small filters | learned: any patch, any layer |
    | Weights | {verify["resnet_weights"]:,} | {verify["vit_weights"]:,} |
    | Time per tile, this run | {_ms_r:.2f} ms | {_ms_v:.2f} ms |
    | Data needs | ImageNet suffices | far more, or ImageNet + care |

    A CNN assumes locality (Lecture 15's filters) and gets it for free; a transformer must learn
    which patches matter, which takes more data: ViT trained on ImageNet alone trails ResNets,
    and beats them after pretraining on 300 million images (Dosovitskiy et al., 2021); the
    weights used here were trained on ImageNet with strong augmentation (torchvision's recipe).
    Time per tile: reading, resizing, and the network, averaged over {_passes:,} passes each
    (the slides: {_slides["resnet18_ms"]:.2f} and {_slides["vit_b_16_ms"]:.2f} ms on an RTX 3090
    Ti, shared with other jobs; your numbers depend on your machine).

    ### Swap the screen's backbone?

    Overall the ViT screen opens fewer tiles at a similar share of cracks flagged, but not on
    every surface (above). The price: about {_w_ratio:.0f} times the weights and
    {_ms_v / _ms_r:.1f} times the time per tile; all {_deck_photos} deck photographs of SDNET2018
    ({_deck_tiles:,} tiles) take {_deck_s["vit_b_16"]:.0f} s with the ViT and
    {_deck_s["resnet18"]:.0f} s with ResNet-18 on this machine (the slides' GPU:
    {_slides["vit_b_16_deck_seconds"]:.0f} s and {_slides["resnet18_deck_seconds"]:.0f} s,
    {_slides["ratio"]:.1f} times). Lecture 15's caution holds unchanged: both thresholds were set on SDNET2018's
    validation photos; a new owner's camera needs its own check.

    ### Who answers for a borrowed model?

    | | Lecture 15's screen | Today's screen |
    |:--|:--------------------|:---------------|
    | Trained on | ImageNet, then our labels | ImageNet, then our labels |
    | Size, energy | 11 M weights | 86 M weights, {_ms_v / _ms_r:.1f}× the time |
    | What we can check | our test photos | our test photos |
    | What we cannot | ImageNet's contents | ImageNet's contents |

    Both screens rest on a pretrained network whose training data (ImageNet; for BERT, books and
    Wikipedia) no one in this room chose or audited; our evidence is our own test set and a
    random audit of unflagged tiles in service (Lecture 15). A larger model costs more energy
    for every tile screened; at a few milliseconds a tile it is small here, and large when an
    agency screens every photo of every bridge every cycle. The inspector who signs the report
    answers for both.

    ### Vote: what screens next season's decks?

    - **A.** Lecture 15's ResNet-18 screen, as is.
    - **B.** The ViT-B/16 screen, threshold re-set on the owner's photos, unflagged tiles audited.
    - **C.** Fine-tune a vision transformer on SDNET2018 first.
    - **D.** No screen: review every tile.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(r"""
        **B.** On these photos the ViT screen reviews 41% of tiles against 51%, flagging 88% of
        cracks against 86%; the extra cost is milliseconds per tile. On decks alone it flags
        fewer (84% against 91%), which is why the threshold must be re-set on the owner's own
        deck photos. The re-set threshold and the audit are Lecture 15's conditions, unchanged.
        C may gain more, but needs a GPU budget, a validation plan, and a reason to expect gains
        from 137 training photos (with moderate data, fine-tune; with little, probe). A is
        defensible if the owner already validated it; D is the baseline.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where transformers live today

    | Today | What it is | Lecture |
    |:------|:-----------|:--------|
    | ChatGPT, Claude, Gemini | decoder-only transformers on text tokens | 19 |
    | CLIP, SAM, Prithvi | transformers on image patches and text | 18 |
    | Chronos-2, TimesFM | transformers on time-series patches | 20 |
    | Whisper | a transformer on spectrogram patches | 20 |

    The block in §4 is the block inside all of them. Decoder-only models (GPT; Lecture 19) mask
    attention so each token sees only earlier ones and predict the next token; encoders (BERT,
    ViT) see both sides. Lecture 18 uses CLIP and SAM on construction images; Lecture 20 uses
    Chronos-2 on rivers.

    ## 7. Check yourself

    **Exit question.** A vendor's ViT screen reports 95% recall on "held-out tiles". Name two
    things you would ask before swapping your ResNet screen for it, and the number you would
    want instead.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        Held out how? Tiles of the same photographs leak (Lecture 15's siblings): ask for whole
        photos, or whole structures, held out. At what threshold, set on what? Ask for recall
        and the share of tiles to review at a threshold fixed on validation, by surface, and on
        your own camera's photos. The number to want: tiles to review at your recall target, on
        your photos.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas

    - **The tokenizer belongs to the model.** BERT's token ids mean something only with BERT's
      own WordPiece vocabulary; the uncased model lowercases first. A word that splits into
      several pieces cannot be the answer to one `[MASK]` (146 of the 1,102 blanks).
    - **Scale the scores.** Without the $1/\sqrt{d}$, dot products grow with the dimension and
      the softmax puts nearly all the weight on one key (§3).
    - **Attention has no order.** Without position embeddings a transformer block is
      permutation-equivariant (§3): add positions before the first block.
    - **`nn.MultiheadAttention` packs $W_Q, W_K, W_V$** in one `in_proj_weight` of shape
      $(3d, d)$, in that order, and computes `x @ W.T + b`; split the result before reshaping
      into heads, and transpose heads back before the output projection.
    - **Pre-norm or post-norm.** torchvision's ViT normalizes before attention and before the
      MLP (pre-norm); the original transformer normalized after. Match the model you check
      against.
    - **Average embeddings compress.** The mean of word vectors made every pair of reports
      0.94–0.97 alike: cosines of averages need a reference, not an absolute threshold.
    - **Attention maps are not explanations.** Rollout ignores the MLPs and the values; check a
      map against what you can measure, and say what the data cannot show.
    - **TF32 on the GPU.** On recent NVIDIA GPUs, cuDNN runs convolutions in TF32 by default
      (10 mantissa bits): a GPU pass of ResNet-18 then differs from its cache, made on the CPU,
      by up to about 0.01 (measured on 256 tiles). With `torch.backends.cudnn.allow_tf32 = False`,
      as here, it differs by about 1e-5. BERT's numbers still move in their last digits from
      one device to another; their asserts allow a stated difference.

    ## Going deeper (Forsyth)

    - §1.2: classifying with nearest neighbours. Attention (§2) is the soft version: every
      neighbour counts, weighted by a softmax of its distance, and the temperature sets how
      many matter.
    - §6.3–6.3.1: the bag-of-words model and the cosine distance between count vectors. Our
      "shared words" is the set version; both ignore order, as attention does without positions.
      Forsyth calls the count vectors "word vectors"; today's word vectors are learned
      embeddings, one per word, not counts per document.
    - §6.3.4–6.3.5: the words near a word suggest its meaning; latent semantic analysis builds
      word embeddings from smoothed document counts with the SVD, and Forsyth notes that
      methods using the nearby words themselves, word2vec among them, embed better.
    - §16.1.1: the softmax as the layer that turns scores into probabilities. Attention applies
      it across tokens instead of across classes.
    - §19.2.3: the denoising autoencoder with masking noise, trained to recover what was hidden:
      BERT's objective is that idea on text tokens.
    - §18.1.9: residual networks; each transformer block adds its attention and MLP outputs to
      its input the same way.

    **Exercises.** (1) Build the word–context count matrix of the training years (a window of 5),
    reduce it to 100 dimensions with a truncated SVD (§6.3.4), and compare the nearest words of
    "ladder", "trench", and "scaffold" with word2vec's. (2) Compute the rollout map of each of
    the last block's 12 heads separately for the wall tile: do all heads look at the crack?

    ## Things to remember

    - Text and images become tokens.
    - Attention: a softmax-weighted average.
    - A block: attend, then an MLP, with skips.
    - Position must be added.
    - Similar to what? Train for your meaning.

    **Next:** CLIP and SAM, foundation models for images.
    """)
    return


if __name__ == "__main__":
    app.run()
