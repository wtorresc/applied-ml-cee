"""Figures and numbers for CEE Lecture 18: vision foundation models (CLIP, SAM).

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l18_vision_foundation.py \
        --out build/fig/cee_l18_vision_foundation

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides). The lecture
follows the CLIP half of CS 441's "Foundation Models: CLIP and GPT" (Fall 2026), then adds SAM:

- CLIP (Radford et al., 2021; ViT-L/14 at 336 px, OpenAI's weights): a text encoder as the
  classifier, contrastive image-text training, zero-shot screening of SDNET2018's crack tiles
  (Lecture 15's split) by prompts, probes on CLIP's image embeddings from 1 to 16 labelled
  photos per surface (C fixed at the all-photo probe's choice, the threshold set on as many
  labelled validation photos per surface, ten draws), and distance-weighted nearest neighbours
  (k chosen on validation), against Lecture 15's ResNet-18 and Lecture 17's ViT-B/16.
- SAM 2.1 (Ravi et al., 2024; Apache-2.0): masks from clicks, on 3DCrack's thin cracks
  (Lecture 16) and on SpaceNet 8's buildings (Lecture 16), scored by IoU against the labels.

Image embeddings come from ``scripts/derive_sdnet2018_clip.py`` (and the Lecture 15 and 17
caches, each loaded only after its recorded model and preprocessing are checked); the photo
split is Lecture 15's, exactly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F  # noqa: N812 (PyTorch's convention)
from torchvision.models import ViT_B_16_Weights

import aml_style
import crack3d
import spacenet8
import title_art
from common import DATA_DIR, parse_out_dir, require, seed_all, write_numbers
from probes import fit_probe, scores, threshold_for
from sdnet import IMAGENET_MEAN, IMAGENET_STD, SIZE, load, photo_split, read_tile

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub is imported
import transformers

SEED = 18
# CLIP's model and revision, here once: the text side below and the image cache's check use them
# (scripts/derive_sdnet2018_clip.py, which wrote the cache, pins its own copy).
CLIP = "openai/clip-vit-large-patch14-336"
CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CLIP_FEATURES = DATA_DIR / "derived" / "sdnet2018_clip_l14_336.npy"
CLIP_CACHE = {  # what the cache's JSON must record: model, revision, preprocessing, layer
    "model": CLIP,
    "revision": CLIP_REVISION,
    "preprocess": "CLIPImageProcessor: shorter side 336, centre crop 336, CLIP mean/std",
    "layer": "get_image_features(...).pooler_output: the projected image embedding (768), "
    "not normalized",
}
VIT_FEATURES = DATA_DIR / "derived" / "sdnet2018_vit_b_16.npy"
VIT_CACHE = {  # Lecture 17's cache: torchvision's ViT-B/16 on Lecture 15's preprocessing
    "weights": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
    "url": ViT_B_16_Weights.IMAGENET1K_V1.url,
    "preprocess": f"resize {SIZE}x{SIZE} bilinear antialias; "
    f"ImageNet mean {IMAGENET_MEAN} std {IMAGENET_STD}",
    "layer": "class token after the encoder's final layer norm (heads replaced by identity)",
}
SURFACE_NOUN = {"deck": "bridge deck", "wall": "concrete wall", "pavement": "pavement"}
# Prompt sets: (clean, cracked). "{s}" is the tile's surface; the set is chosen on validation.
PROMPTS: dict[str, tuple[str, str]] = {
    "concrete": ("a photo of concrete", "a photo of cracked concrete"),
    "surface": ("a photo of a {s}", "a photo of a cracked {s}"),
    "crack": (
        "a photo of a concrete surface without cracks",
        "a photo of a concrete surface with a crack",
    ),
    "close-up": ("a close-up photo of intact concrete", "a close-up photo of a crack in concrete"),
}
SHOTS = (1, 2, 4, 8, 16)  # labelled training photos per surface for the few-shot probes
SHOT_DRAWS = tuple(range(10))  # random draws of the photos for each k
# The few-label probes' C: k photos leave no labels to tune it on. 0.01 is the C the probe on all
# photos picks on all 47 validation photos (full_probes), so it borrows labels the few-label owner
# does not have; FEW_C_CHECK's neighbours, scored on the test photos, are an after-the-fact check.
FEW_C = 0.01
FEW_C_CHECK = (0.003, 0.03)  # its neighbours on probes.C_GRID, reported beside it as a check
# CS 441's third way to adapt CLIP: nearest neighbours on its image embeddings, votes weighted by
# 1 / cosine distance (no ties at the threshold), k chosen on validation by a bracketing search
KNN_START = (5, 10, 25, 50, 100)
KNN_LADDER = (1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200)


def device() -> torch.device:
    """The GPU if there is one."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cache(path: Path, want: dict[str, object], rows: int, hint: str) -> np.ndarray:
    """A feature cache, loaded only if its JSON records the model and preprocessing expected.

    The same check as ``sdnet.load`` makes for Lecture 15's cache (duplicated in Lecture 17's
    script until ``sdnet`` offers it).

    Args:
        path: The ``.npy`` cache; its metadata is the ``.json`` beside it.
        want: Fields the metadata must record exactly (model, revision, preprocessing, layer).
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
# CLIP: a text encoder as the classifier
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """CLIP's text side and its learned temperature (the image side is cached)."""

    model: transformers.CLIPModel
    tokenizer: transformers.CLIPTokenizer
    scale: float  # exp(logit_scale): CLIP's learned inverse temperature

    def text(self, prompts: list[str]) -> np.ndarray:
        """Unit-length text embeddings (n, 768)."""
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            t = self.model.get_text_features(**enc).pooler_output  # projected (n, 768)
        return F.normalize(t, dim=1).numpy()


def clip() -> Clip:
    """CLIP ViT-L/14 at 336 px, on the CPU (only text is encoded here)."""
    model = transformers.CLIPModel.from_pretrained(CLIP, revision=CLIP_REVISION).eval()
    tok = transformers.CLIPTokenizer.from_pretrained(CLIP, revision=CLIP_REVISION)
    return Clip(model, tok, float(model.logit_scale.detach().exp()))


def zero_shot(
    c: Clip, img: np.ndarray, surfaces: np.ndarray, prompts: tuple[str, str]
) -> np.ndarray:
    """P(cracked) for each tile: a softmax over the two prompts' cosine similarities.

    Args:
        c: CLIP's text side.
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
        t = c.text([prompts[0].format(s=noun), prompts[1].format(s=noun)])  # (2, 768)
        logits = c.scale * z[m] @ t.T  # (tiles, 2): the text embeddings are the classifier
        p[m] = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return p


def zero_shot_screen(c: Clip, img: np.ndarray, df: pd.DataFrame, part: np.ndarray) -> dict:
    """Each prompt set on validation photos; the best one tested once, at two thresholds."""
    y = df["cracked"].to_numpy()
    surf = df["surface"].to_numpy()
    val, test = part == "val", part == "test"
    val_ap = {}
    probs = {}
    for name, pr in PROMPTS.items():
        probs[name] = zero_shot(c, img, surf, pr)
        val_ap[name] = scores(probs[name][val], y[val], 0.5)["ap"]
    best = max(val_ap, key=val_ap.get)
    p = probs[best]
    t90 = threshold_for(p[val], y[val])
    out = {
        "val_ap": val_ap,
        "best": best,
        "prompts": list(PROMPTS[best]),
        "at_half": scores(p[test], y[test], 0.5),  # no labels at all
        "at_90": scores(p[test], y[test], t90),  # threshold set on labelled validation photos
        "t90": t90,
        "by_surface": {
            s: scores(p[test & (surf == s)], y[test & (surf == s)], t90) for s in SURFACE_NOUN
        },
    }
    out["test_ap_all_prompts"] = {n: scores(probs[n][test], y[test], 0.5)["ap"] for n in PROMPTS}
    # Within one surface (ranking only) against across surfaces (one scale): pooled AP can fall
    # because a prompt shifts a whole surface's scores, while it still ranks within each surface.
    out["test_auc_by_surface"] = {
        n: {
            s: float(roc_auc_score(y[test & (surf == s)], probs[n][test & (surf == s)]))
            for s in SURFACE_NOUN
        }
        for n in PROMPTS
    }
    out["test_median_p_by_surface"] = {
        n: {s: float(np.median(probs[n][test & (surf == s)])) for s in SURFACE_NOUN}
        for n in PROMPTS
    }
    out["_p"] = p  # the chosen pair's P(cracked) for every tile (private: not in numbers.json)
    return out


def draw_photos(
    df: pd.DataFrame, among: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """The tiles of k photos of each surface, drawn at random from the photos of ``among``.

    Args:
        df: The tile index.
        among: Boolean mask of the tiles whose photos may be drawn (one split).
        k: Photos per surface.
        rng: Draws the photos, surface by surface in ``SURFACE_NOUN``'s order.

    Returns:
        Boolean mask of the drawn photos' tiles.
    """
    photos = df["photo"].to_numpy()
    pick = []
    for s in SURFACE_NOUN:
        pool = np.unique(photos[among & (df["surface"] == s).to_numpy()])
        pick += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return among & np.isin(photos, pick)


def fixed_probe(feats: np.ndarray, y: np.ndarray, use: np.ndarray, c: float) -> np.ndarray:
    """Lecture 15's probe at a fixed C, fitted on the tiles ``use``: P(cracked) for every tile.

    Args:
        feats: Features, one row per tile.
        y: 1 if cracked.
        use: Boolean mask of the labelled tiles it learns from (both classes present).
        c: Inverse L2 strength, fixed (not tuned on ``use``).

    Returns:
        P(cracked) for every row of ``feats``.
    """
    scaler = StandardScaler().fit(feats[use])
    # l1_ratio=0 is the L2 penalty (scikit-learn 1.8 deprecates penalty="l2"), as probes.py
    m = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
    m.fit(scaler.transform(feats[use]), y[use])
    return m.predict_proba(scaler.transform(feats))[:, 1]


def spread(v: list[float]) -> dict[str, object]:
    """Every draw's value, their median, and their range.

    Args:
        v: One value per draw.

    Returns:
        ``draws`` (the values), ``median``, and ``range`` ([min, max]).
    """
    return {"draws": v, "median": float(np.median(v)), "range": [min(v), max(v)]}


def few_shot(
    feats: np.ndarray, df: pd.DataFrame, part: np.ndarray, zero_p: np.ndarray
) -> dict[str, object]:
    """Probes on CLIP's image embeddings from k labelled photos per surface: the few-label screen.

    An owner with k labelled photos per surface has no other labels. For each k and draw: k
    training photos of each surface are drawn; the probe (Lecture 15's logistic regression on
    standardized features) is fitted on their tiles at C = ``FEW_C`` (see its comment); the
    threshold that flags 90% of the cracked tiles is set on k validation photos per surface,
    drawn at random; and the test photos are scored once. Zero-shot CLIP (``zero_p``) gets its
    threshold from the same k validation photos, like for like. The spread over the draws is
    the price of so few labels.

    Args:
        feats: CLIP's image embeddings, one row per tile.
        df: The tile index.
        part: Lecture 15's split of every tile.
        zero_p: Zero-shot P(cracked) of every tile, from the prompt pair chosen on validation.

    Returns:
        Per k: labelled tiles in each draw, and the test AP, recall, and share of tiles to
        review of each draw, with their median and range; zero-shot's recall and review at its
        k-photo threshold; and the median AP at C's neighbours on the grid.

    Raises:
        ValueError: If a draw's photos hold only one class (no probe can be fitted).
    """
    y = df["cracked"].to_numpy()
    train, val, test = (part == r for r in ("train", "val", "test"))
    out: dict[str, object] = {"C": FEW_C, "draws": len(SHOT_DRAWS)}
    for k in SHOTS:
        rows: dict[str, list] = {key: [] for key in ("ap", "recall", "review", "zs_recall",
                                                    "zs_review", "tiles", "cracked_tiles",
                                                    "val_tiles")}  # fmt: skip
        check: dict[float, list[float]] = {c: [] for c in FEW_C_CHECK}
        for draw in SHOT_DRAWS:
            use = draw_photos(df, train, k, np.random.default_rng((SEED, k, draw)))
            use_val = draw_photos(df, val, k, np.random.default_rng((SEED, k, draw, 1)))
            if y[use].min() == y[use].max() or y[use_val].max() == 0:
                raise ValueError(f"k = {k}, draw {draw}: one class only in the drawn photos")
            p = fixed_probe(feats, y, use, FEW_C)
            s_ = scores(p[test], y[test], threshold_for(p[use_val], y[use_val]))
            z_ = scores(zero_p[test], y[test], threshold_for(zero_p[use_val], y[use_val]))
            for key, v in (("ap", s_["ap"]), ("recall", s_["recall"]), ("review", s_["review"]),
                           ("zs_recall", z_["recall"]), ("zs_review", z_["review"]),
                           ("tiles", int(use.sum())), ("cracked_tiles", int(y[use].sum())),
                           ("val_tiles", int(use_val.sum()))):  # fmt: skip
                rows[key].append(v)
            for c in FEW_C_CHECK:
                pc = fixed_probe(feats, y, use, c)
                check[c].append(float(average_precision_score(y[test], pc[test])))
        out[str(k)] = {
            **{m: spread(rows[m]) for m in ("ap", "recall", "review")},
            "zero_shot": {"recall": spread(rows["zs_recall"]), "review": spread(rows["zs_review"])},
            "tiles": rows["tiles"],  # labelled training tiles in the drawn photos, per draw
            "cracked_tiles": rows["cracked_tiles"],
            "val_tiles": rows["val_tiles"],  # labelled validation tiles that set the threshold
            "ap_median_at_C": {str(c): float(np.median(v)) for c, v in check.items()},
        }
    return out


def knn_scores(z: np.ndarray, y: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """P(cracked) of every tile from its k nearest training tiles, votes weighted by 1 / distance.

    Args:
        z: Unit-length image embeddings, one row per tile.
        y: 1 if cracked.
        train: Boolean mask of the training tiles (the neighbours).
        k: Neighbours.

    Returns:
        The weighted share of cracked neighbours, for every tile (cosine distance).
    """
    knn = KNeighborsClassifier(k, weights="distance", algorithm="brute", metric="cosine")
    return knn.fit(z[train], y[train]).predict_proba(z)[:, 1]


def knn_screen(img: np.ndarray, y: np.ndarray, part: np.ndarray) -> dict[str, object]:
    """Nearest neighbours on CLIP's embeddings: k chosen on validation, then the 90% screen.

    A bracketing search: start from ``KNN_START``; while the best validation AP is
    at an edge, add the next k of ``KNN_LADDER`` past it; the kept k must be interior. The
    threshold is set on validation, as the probe's, and the test photos are scored once.

    Args:
        img: CLIP's image embeddings, not normalized.
        y: 1 if cracked.
        part: Lecture 15's split of every tile.

    Returns:
        The test scores at the chosen k, the k, every k's validation AP, and the test tiles
        whose score equals the threshold exactly (ties).

    Raises:
        ValueError: If the best k sits at an end of the ladder.
    """
    train, val, test = (part == r for r in ("train", "val", "test"))
    z = img / np.linalg.norm(img, axis=1, keepdims=True)
    val_ap: dict[int, float] = {}
    tried = list(KNN_START)
    while True:
        for k in tried:
            if k not in val_ap:
                val_ap[k] = float(average_precision_score(y[val], knn_scores(z, y, train, k)[val]))
        best, ks = max(val_ap, key=val_ap.get), sorted(val_ap)
        i = KNN_LADDER.index(best)
        if ks[0] < best < ks[-1]:
            break
        nxt = i - 1 if best == ks[0] else i + 1
        if not 0 <= nxt < len(KNN_LADDER):
            raise ValueError(f"k = {best} is at the end of KNN_LADDER; extend it")
        tried.append(KNN_LADDER[nxt])
    p = knn_scores(z, y, train, best)
    t = threshold_for(p[val], y[val])
    return scores(p[test], y[test], t) | {
        "k": best,
        "val_ap": {str(k): v for k, v in sorted(val_ap.items())},
        "threshold": t,
        "ties_at_threshold": int(np.sum(p[test] == t)),
    }


# --------------------------------------------------------------------------------------------
# SAM 2.1: masks from clicks, on thin cracks and on buildings
# --------------------------------------------------------------------------------------------

SAM = "facebook/sam2.1-hiera-base-plus"  # Ravi et al. (2024); Apache-2.0
SAM_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
CLICKS = (1, 3, 5)  # positive clicks placed along a crack's centreline
BUILDINGS = 300  # SpaceNet 8 test buildings sampled for the one-click test
MIN_BUILDING_PX = 400  # about 45 m2 at 0.3-0.4 m pixels: a house, not a shed


@dataclass(frozen=True)
class Sam:
    """SAM 2.1's model and processor on the GPU."""

    model: transformers.Sam2Model
    proc: transformers.Sam2Processor

    def embed(self, rgb: np.ndarray) -> list[torch.Tensor]:
        """SAM's image encoder, run once per image: its feature maps, which every prompt reuses.

        Args:
            rgb: The image (h, w, 3) uint8.

        Returns:
            The image embeddings, as ``Sam2Model.get_image_embeddings`` returns them.
        """
        px = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            return self.model.get_image_embeddings(px.to(device()))

    def masks(
        self, rgb: np.ndarray, points: list[list[list[int]]], emb: list[torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray]:
        """SAM's masks for one image and several prompts (each a list of positive clicks).

        Args:
            rgb: The image (h, w, 3) uint8; only its size is used here, the pixels are in ``emb``.
            points: For each prompt, its clicks as [x, y] pixels; every prompt has as many clicks.
            emb: The image's embeddings, from ``embed``.

        Returns:
            Boolean masks (prompts, 3, h, w), SAM's three candidates per prompt, and SAM's
            predicted IoU of each (prompts, 3).
        """
        labels = [[[1] * len(p) for p in points]]
        enc = self.proc(images=rgb, input_points=[points], input_labels=labels, return_tensors="pt")
        with torch.no_grad():
            out = self.model(
                image_embeddings=emb,
                input_points=enc["input_points"].to(device()),
                input_labels=enc["input_labels"].to(device()),
                multimask_output=True,
            )
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), enc["original_sizes"])[0]
        return masks.numpy() > 0, out.iou_scores[0].float().cpu().numpy()


def sam() -> Sam:
    """SAM 2.1 (Hiera base-plus) at a pinned revision."""
    model = transformers.Sam2Model.from_pretrained(SAM, revision=SAM_REVISION).eval().to(device())
    return Sam(model, transformers.Sam2Processor.from_pretrained(SAM, revision=SAM_REVISION))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    return float((a & b).sum() / max((a | b).sum(), 1))


def along(skeleton: np.ndarray, k: int) -> list[list[int]]:
    """``k`` clicks spread along a crack's centreline: evenly spaced in its pixel order.

    The centreline's pixels are ordered along the crack's main direction (their first principal
    axis); the clicks sit at evenly spaced ranks, ends excluded.
    """
    rc = np.argwhere(skeleton).astype(float)
    axis = np.linalg.svd(rc - rc.mean(0), full_matrices=False)[2][0]
    order = rc[np.argsort((rc - rc.mean(0)) @ axis)]
    ranks = np.linspace(0, len(order) - 1, k + 2)[1:-1].round().astype(int)
    return [[int(order[r, 1]), int(order[r, 0])] for r in ranks]  # [x, y]


def sam_cracks(s: Sam) -> dict[str, object]:
    """SAM on 3DCrack's 250 test range images: 1, 3, 5 clicks on the crack's centreline."""
    split = crack3d.load_split("Test")
    out: dict[str, list[float]] = {str(k): [] for k in CLICKS}
    oracle: dict[str, list[float]] = {str(k): [] for k in CLICKS}
    shown = []  # (image, clicks, SAM's mask, IoU) at 3 clicks, for the figure
    for i, (img, gt) in enumerate(zip(split.x, split.y, strict=True)):
        skel = crack3d.thin(gt)
        if skel.sum() < 2 * max(CLICKS):
            continue
        rgb = np.repeat(img[..., None], 3, axis=2)
        emb = s.embed(rgb)  # once per image; the 1, 3, and 5 clicks reuse it
        for k in CLICKS:
            pts = along(skel, k)
            m, pred = s.masks(rgb, [pts], emb)
            pick = int(np.argmax(pred[0]))  # SAM's own choice among its three masks
            out[str(k)].append(iou(m[0, pick], gt))
            oracle[str(k)].append(max(iou(m[0, j], gt) for j in range(m.shape[1])))
            if k == 3:
                shown.append((i, pts, m[0, pick], out[str(k)][-1]))
    mid = float(np.median([e[3] for e in shown]))
    example = min(shown, key=lambda e: abs(e[3] - mid))  # a typical image, not the best
    return {
        "images": len(out["1"]),
        "example_iou": example[3],
        "iou_median": {k: float(np.median(v)) for k, v in out.items()},
        "iou_oracle_median": {k: float(np.median(v)) for k, v in oracle.items()},
        "iou": out,
        "example": example,
        "split": split,
    }


def sam_buildings(s: Sam) -> dict[str, object]:
    """SAM on SpaceNet 8's test buildings (pre-event images): one click inside each building."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415 (only this test needs it)

    test = spacenet8.split(spacenet8.tiles())["test"]
    rng = np.random.default_rng(SEED)
    cand = [
        (ti, b)
        for ti, t in enumerate(test)
        for b in np.unique(t.inst)[1:]
        if (t.inst == b).sum() >= MIN_BUILDING_PX
    ]
    pick = [
        cand[i]
        for i in np.sort(rng.choice(len(cand), size=min(BUILDINGS, len(cand)), replace=False))
    ]
    ious, example = [], None
    for ti in sorted({p[0] for p in pick}):
        t = test[ti]
        ids = [b for tj, b in pick if tj == ti]
        pts = []
        for b in ids:
            d = distance_transform_edt(t.inst == b)  # the most interior pixel: a sure click
            r, c = np.unravel_index(np.argmax(d), d.shape)
            pts.append([[int(c), int(r)]])
        m, pred = s.masks(t.pre, pts, s.embed(t.pre))
        for j, b in enumerate(ids):
            mj = m[j, int(np.argmax(pred[j]))]
            ious.append(iou(mj, t.inst == b))
            if example is None:
                example = (ti, pts[j][0], mj, t.inst == b)
    return {
        "candidates": len(cand),
        "buildings": len(ious),
        "iou_median": float(np.median(ious)),
        "iou": ious,
        "share_above_half": float(np.mean(np.array(ious) > 0.5)),
        "example_iou": iou(example[2], example[3]) if example else None,
        "example_mask_px": int(example[2].sum()) if example else None,
        "example_footprint_px": int(example[3].sum()) if example else None,
        "example": example,
        "tile": test[example[0]] if example else None,
    }


# --------------------------------------------------------------------------------------------
# Contrastive training, checked; the one picture; figures
# --------------------------------------------------------------------------------------------

CLASSES = [(s, c) for s in SURFACE_NOUN for c in (0, 1)]  # (surface, cracked)


def caption(surface: str, cracked: int) -> str:
    """A caption for a tile, as a person might have written it."""
    noun = SURFACE_NOUN[surface]
    return f"a photo of a cracked {noun}" if cracked else f"a photo of a {noun}"


def contrastive_check(c: Clip, img: np.ndarray, df: pd.DataFrame, rows: list[int]) -> dict:
    """CLIP's training loss written out, against transformers' ``image_text_contrastive_loss``.

    The batch is the tiles ``rows``, each with a caption written from its labels (``caption``),
    so equal captions repeat in it. The cached image embeddings are normalized in float64; the
    caption embeddings come normalized from CLIP's text side (float32) and are cast to float64.
    The logits are ``c.scale`` times the cosine of every image with every caption (image i
    against caption j), and the loss averages the cross-entropy along the rows (each image picks
    its own caption) and along the columns (each caption picks its own image): Lecture 6's
    loss, twice, with only the diagonal counted right. Transformers' function takes the
    caption-by-image matrix, hence the transpose (the loss is the same either way).

    Args:
        c: CLIP's text side and scale.
        img: Cached image embeddings, one row per tile, not normalized.
        df: The tile index (surface and label of each tile, for its caption).
        rows: The tiles of the batch.

    Returns:
        The batch size, our loss, and its absolute difference from transformers' loss.
    """
    zi = torch.tensor(img[rows], dtype=torch.float64)
    zi = zi / zi.norm(dim=1, keepdim=True)
    zt = torch.tensor(
        c.text([caption(df["surface"].iloc[r], int(df["cracked"].iloc[r])) for r in rows]),
        dtype=torch.float64,
    )
    logits = c.scale * zi @ zt.T  # (batch, batch): image i against caption j
    target = torch.arange(len(rows))
    ours = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    ref = transformers.models.clip.modeling_clip.image_text_contrastive_loss(logits.T)
    return {"batch": len(rows), "loss": float(ours), "diff": float(abs(ours - ref))}


def six_tiles(df: pd.DataFrame, part: np.ndarray) -> list[int]:
    """One random test tile of each class in ``CLASSES`` (seed ``SEED``): the attempt's tiles.

    Args:
        df: The tile index.
        part: Lecture 15's split of every tile.

    Returns:
        Six row indices into ``df``, in ``CLASSES`` order.
    """
    rng = np.random.default_rng(SEED)
    test = part == "test"
    rows = []
    for s, k in CLASSES:
        m = np.flatnonzero(test & (df["surface"] == s).to_numpy() & (df["cracked"] == k).to_numpy())
        rows.append(int(rng.choice(m)))
    return rows


def caption_probs(c: Clip, img: np.ndarray, rows: list[int] | np.ndarray) -> np.ndarray:
    """CLIP's softmax over the six captions for the tiles ``rows``.

    Args:
        c: CLIP.
        img: CLIP's image embeddings, one row per tile.
        rows: The tiles.

    Returns:
        (len(rows), 6) probabilities, captions in ``CLASSES`` order.
    """
    z = img[rows] / np.linalg.norm(img[rows], axis=1, keepdims=True)
    caps = [caption(s, k) for s, k in CLASSES]
    return torch.softmax(torch.tensor(c.scale * z @ c.text(caps).T), dim=1).numpy()


def picture(
    c: Clip, img: np.ndarray, df: pd.DataFrame, part: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Six test tiles (each surface, cracked or not) against six captions: CLIP's softmax."""
    rows = six_tiles(df, part)
    prob = caption_probs(c, img, rows)  # (6, 6)
    picture_frames(df, rows, prob, out_dir)
    surface_right = sum(int(np.argmax(prob[i]) // 2 == i // 2) for i in range(6))
    return {
        "rows": rows,
        "prob": prob.tolist(),
        "diag": np.diag(prob).tolist(),
        "surface_right": surface_right,
        "crack_right": sum(int(np.argmax(prob[i]) == i) for i in range(6)),
        "crack_half_right": sum(int(np.argmax(prob[i]) % 2 == i % 2) for i in range(6)),
    }


def picture_frames(df: pd.DataFrame, rows: list[int], prob: np.ndarray, out_dir: Path) -> None:
    """The attempt's two frames: the six tiles and the caption grid, empty, then CLIP's answers.

    Args:
        df: The tile index.
        rows: The six tiles (``six_tiles``).
        prob: Their caption probabilities (``caption_probs``).
        out_dir: Where the frames go.
    """
    caps = [caption(s, k) for s, k in CLASSES]
    for frame in (1, 2):  # 1: the attempt, captions without CLIP's answers; 2: the answers
        fig = plt.figure(figsize=aml_style.FULL)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.6), wspace=0.05)
        ax_t = grid[0].subgridspec(2, 3, wspace=0.05, hspace=0.25)
        for j, r in enumerate(rows):
            ax = fig.add_subplot(ax_t[j % 2, j // 2])
            ax.imshow(read_tile(df["path"].iloc[r]))
            ax.set_title(f"{j + 1}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(grid[1])
        ax.imshow(prob if frame == 2 else np.zeros_like(prob), cmap="cividis", vmin=0, vmax=1)
        if frame == 2:
            for i in range(6):
                for j in range(6):
                    ax.text(
                        j,
                        i,
                        f"{prob[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if prob[i, j] < 0.5 else aml_style.INK,
                    )
        ax.set_yticks(range(6), [f"tile {i + 1}" for i in range(6)], fontsize=10)
        ax.set_xticks(
            range(6),
            [cp.replace("a photo of a ", "") for cp in caps],
            fontsize=9,
            rotation=35,
            ha="right",
        )
        aml_style.save(fig, out_dir, f"picture_{frame}")


def six_captions(c: Clip, img: np.ndarray, df: pd.DataFrame, part: np.ndarray) -> dict:
    """The picture's six captions over every test tile: does CLIP know surfaces, or cracks?

    Surface: the arg-max caption's surface, against always answering the commonest surface.
    Cracks: P(cracked) = the three cracked captions' total, as a ranking (AUC). Walls against
    pavements: P(wall) / (P(wall) + P(pavement)) over those tiles, as a ranking (AUC).
    """
    test = part == "test"
    prob = caption_probs(c, img, np.flatnonzero(test))  # (n, 6)
    names = list(SURFACE_NOUN)
    surf = df["surface"].to_numpy()[test]
    y = df["cracked"].to_numpy()[test]
    called = np.array(names)[prob.argmax(1) // 2]
    by_surface = prob.reshape(-1, 3, 2).sum(2)  # (n, 3): deck, wall, pavement
    wp = np.isin(surf, ["wall", "pavement"])
    p_wall = by_surface[wp, 1] / (by_surface[wp, 1] + by_surface[wp, 2])
    commonest = pd.Series(surf).value_counts()
    return {
        "tiles": int(test.sum()),
        "surface_accuracy": float(np.mean(called == surf)),
        "commonest_surface": str(commonest.index[0]),
        "commonest_share": float(commonest.iloc[0] / len(surf)),
        "crack_auc": float(roc_auc_score(y, prob[:, 1::2].sum(1))),
        "wall_vs_pavement_auc": float(roc_auc_score(surf[wp] == "wall", p_wall)),
        "called": {s: {t: int(np.sum((surf == s) & (called == t))) for t in names} for s in names},
        "tiles_by_surface": {s: int(np.sum(surf == s)) for s in names},
    }


def shots_figure(
    fs: dict[str, object], zs: dict[str, object], refs: dict[str, float], out_dir: Path
) -> None:
    """Test AP against labelled photos per surface: CLIP probes, zero-shot, and references."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ks = list(SHOTS)
    med = [fs[str(k)]["ap"]["median"] for k in ks]
    lo, hi = zip(*(fs[str(k)]["ap"]["range"] for k in ks), strict=True)
    ax.fill_between(ks, lo, hi, color=aml_style.PALETTE[0], alpha=0.15)
    ax.plot(
        ks,
        med,
        color=aml_style.PALETTE[0],
        marker="o",
        lw=2,
        label=f"CLIP probe, k photos per surface ({fs['draws']} draws; band: range)",
    )
    ax.axhline(
        zs["at_half"]["ap"],
        color=aml_style.PALETTE[1],
        ls="--",
        lw=2,
        label="CLIP zero-shot: no labels",
    )
    for (name, ap), mk, col in zip(
        refs.items(),
        ("s", "^", "D"),
        (aml_style.PALETTE[2], aml_style.PALETTE[3], aml_style.NAVY),
        strict=True,
    ):
        ax.plot([45], [ap], marker=mk, ms=10, ls="", color=col, label=name)
    ax.set_xscale("log")
    ax.set_xticks([*ks, 45], [*map(str, ks), "all"])
    ax.set_xlabel("k labelled training photos per surface (and k more for the threshold)")
    ax.set_ylabel("test average precision")
    ax.legend(fontsize=10, frameon=False, loc="lower right", bbox_to_anchor=(0.93, 0.07))
    aml_style.save(fig, out_dir, "shots")


def sam_figure(cr: dict[str, object], bl: dict[str, object], out_dir: Path) -> None:
    """A crack and a building with SAM's mask from clicks, and the IoU distributions."""
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, width_ratios=(1, 1, 1.3))
    fig.subplots_adjust(wspace=0.45)
    i, pts, m, _ = cr["example"]
    img, gt = cr["split"].x[i], cr["split"].y[i]
    axes[0].imshow(img, cmap="gray")
    axes[0].contour(gt, levels=[0.5], colors=[aml_style.PALETTE[2]], linewidths=0.8)
    axes[0].imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.5)
    axes[0].contour(m, levels=[0.5], colors=[aml_style.PALETTE[1]], linewidths=1.2)
    axes[0].scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        marker="*",
        s=90,
        color=aml_style.PALETTE[1],
        edgecolor="white",
        lw=0.6,
    )
    axes[0].set_title(f"crack: 3 clicks, IoU {cr['example_iou']:.2f}", fontsize=11)
    _, pt, mb, gb = bl["example"]
    t = bl["tile"]
    r0, c0 = max(pt[1] - 120, 0), max(pt[0] - 120, 0)
    win = (slice(r0, r0 + 240), slice(c0, c0 + 240))
    axes[1].imshow(t.pre[win])
    axes[1].contour(gb[win], levels=[0.5], colors=[aml_style.PALETTE[2]], linewidths=1.0)
    axes[1].imshow(np.ma.masked_where(~mb[win], mb[win]), cmap="autumn", alpha=0.45)
    axes[1].scatter(
        [pt[0] - c0],
        [pt[1] - r0],
        marker="*",
        s=90,
        color=aml_style.PALETTE[1],
        edgecolor="white",
        lw=0.6,
    )
    axes[1].set_title("building: 1 click", fontsize=11)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    data = [cr["iou"][str(k)] for k in CLICKS] + [bl["iou"]]
    labels = [f"crack, {k}" for k in CLICKS] + ["building, 1"]
    axes[2].boxplot(data, vert=False, widths=0.5, showfliers=False)
    axes[2].set_yticks(range(1, len(labels) + 1), labels, fontsize=10)
    axes[2].set_xlabel("IoU with the label")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("clicks", fontsize=11)
    aml_style.save(fig, out_dir, "sam")


def title_art_figure(out_dir: Path) -> None:
    """The title art: an image and a sentence meeting in one space, as two lights joining."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0d14", "#101722", "#22101a", "#08070b"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.05, 0.05, 0.55, 0.42, 4), soft=4), "#5a2216", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.50, 0.05, 0.45, 0.30, 3), soft=4), "#3a1912", 0.85)
    left, right, meet = (0.22, top * 0.55), (0.80, top * 0.62), (0.52, top * 0.80)
    c.glow(c.lines([[left, meet]], 2.4, soft=1.0), "#8fc7f0", 0.6, spread=10)
    c.glow(c.lines([[right, meet]], 2.4, soft=1.0), "#f3c98a", 0.6, spread=10)
    c.glow(c.disc(meet, 0.05, soft=3), "#eaa548", 0.55, spread=24)
    c.glaze(c.disc(meet, 0.035, soft=2), "#1d0a10", 0.95)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped rust forms at the bottom; a pale blue "
        "line of light from the left, an image, and a golden one from the right, a sentence, "
        "meet at a small dark sun with a glowing rim, the shared space where CLIP compares "
        "pictures with words.",
    )


def clip_numbers(
    c: Clip, img: np.ndarray, df: pd.DataFrame, part: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """CLIP's scale, the picture, the six captions on all test tiles, the loss, zero-shot.

    Args:
        c: CLIP.
        img: CLIP's image embeddings, one row per tile.
        df: The tile index.
        part: Lecture 15's split of every tile.
        out_dir: Where the figures go.

    Returns:
        The CLIP half of numbers.json, with zero-shot's chosen P(cracked) under ``"_p"``.
    """
    rng = np.random.default_rng(SEED)
    return {
        "scale": c.scale,
        "picture": picture(c, img, df, part, out_dir),
        "six_captions": six_captions(c, img, df, part),
        "contrastive": contrastive_check(c, img, df, list(rng.choice(len(df), 16, replace=False))),
        "zero_shot": zero_shot_screen(c, img, df, part),
    }


def adaptation_numbers(
    feats: dict[str, np.ndarray],
    df: pd.DataFrame,
    part: np.ndarray,
    zs: dict[str, object],
    out_dir: Path,
) -> dict[str, object]:
    """The few-label probes, the probes on all photos (CLIP, ViT-B/16, ResNet-18), and k-NN.

    Args:
        feats: CLIP's, ViT-B/16's, and ResNet-18's features ("clip", "vit", "resnet").
        df: The tile index.
        part: Lecture 15's split of every tile.
        zs: Zero-shot screening's numbers, with the chosen pair's P(cracked) under ``"_p"``.
        out_dir: Where the shots figure goes.

    Returns:
        ``few_shot``, ``full_probes``, ``knn``, the training photos per surface, and the photos
        per split.
    """
    y = df["cracked"].to_numpy()
    train, val, test = (part == r for r in ("train", "val", "test"))
    fs = few_shot(feats["clip"], df, part, zs["_p"])
    refs = {}
    for name, key in (
        ("CLIP probe, all photos", "clip"),
        ("ViT-B/16 probe (Lecture 17)", "vit"),
        ("ResNet-18 probe (Lecture 15)", "resnet"),
    ):
        pr = fit_probe(feats[key], y, train, val)
        p = pr.score(feats[key])
        refs[name] = scores(p[test], y[test], threshold_for(p[val], y[val])) | {"C": pr.C}
    shots_figure(fs, zs, {k: v["ap"] for k, v in refs.items()}, out_dir)
    return {
        "few_shot": fs,
        "full_probes": refs,
        "knn": knn_screen(feats["clip"], y, part),
        "photos_per_surface_train": {
            s: int(df.loc[train & (df["surface"] == s).to_numpy(), "photo"].nunique())
            for s in SURFACE_NOUN
        },
        "photos_per_split": {
            r: int(df.loc[part == r, "photo"].nunique()) for r in ("train", "val", "test")
        },
    }


def sam_numbers(out_dir: Path) -> dict[str, object]:
    """SAM 2.1 on 3DCrack's cracks and SpaceNet 8's buildings, and their figure.

    Args:
        out_dir: Where the figure goes.

    Returns:
        The SAM numbers: cracks and buildings, without per-mask arrays.
    """
    s = sam()
    cr = sam_cracks(s)
    bl = sam_buildings(s)
    sam_figure(cr, bl, out_dir)
    return {
        "cracks": {k: v for k, v in cr.items() if k not in ("iou", "example", "split")}
        | {"example_image": int(cr["example"][0])},
        "buildings": {k: v for k, v in bl.items() if k not in ("iou", "example", "tile")},
    }


def main() -> None:
    """Generate every figure and number for CEE Lecture 18."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.perf_counter()
    df, rfeat = load()
    part = photo_split(df, np.random.default_rng((0, 15)))  # Lecture 15's split, exactly
    feats = {
        "clip": load_cache(
            CLIP_FEATURES, CLIP_CACHE | {"shape": [len(df), 768]}, len(df),
            "scripts/derive_sdnet2018_clip.py",
        ),
        "vit": load_cache(
            VIT_FEATURES, VIT_CACHE | {"shape": [len(df), 768]}, len(df),
            "scripts/derive_sdnet2018_vit.py",
        ),
        "resnet": rfeat,
    }  # fmt: skip
    numbers = clip_numbers(clip(), feats["clip"], df, part, out_dir)
    numbers |= adaptation_numbers(feats, df, part, numbers["zero_shot"], out_dir)
    print(f"CLIP in {time.perf_counter() - t0:.0f} s", flush=True)
    numbers["sam"] = sam_numbers(out_dir)
    print(f"SAM in {time.perf_counter() - t0:.0f} s", flush=True)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_2", "shots", "sam", "picture_1"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
