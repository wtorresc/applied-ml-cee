"""3DCrack: laser range images of Georgia pavements, their crack masks, centrelines, and rulers.

3DCrack (Zhang, Wang, Hsieh, Yang, Yezzi & Tsai, 2025; doi:10.21227/2etq-xh36; CC BY 4.0), from
``scripts/fetch_3dcrack.sh``: 1,634 range images, 512 x 624 px at about 4 mm per pixel, split by
the authors into 1,139 / 245 / 250 training, validation, and test images, with crack masks and,
for every cracked image, crack centrelines that carry two widths per point: ``width_mm`` (from the
mask) and ``width_mm_rangeimg`` (from the range image).

Also here: the mask ruler (the distance transform), a thinning that finds a mask's ridge, the
ridge ruler and the skeleton length used on predicted masks, and the LTPP severity bands
(FHWA-HRT-13-092). Image names repeat across splits (``sample_000450`` is a different image in
Train and in Test), so per-image sets are keyed by (split, stem).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label, maximum_position

from common import DATA_DIR, require

RAW = DATA_DIR / "raw" / "3dcrack" / "3DCrack_v1"
SPLITS: tuple[str, ...] = ("Train", "Val", "Test")
H, W = 624, 512  # every range image: 624 rows (along the lane) x 512 columns (across)
MM_PER_PX: float = 4.0  # the record: "about 4 mm per pixel"
NATIVE_MM: float = 1.0  # the scanner's point spacing after interpolation (paper, Sec. 4.2)
MASK_CUT: int = 127  # a mask pixel is crack when its value exceeds this (see mask_facts)
LTPP_MM: tuple[float, float] = (6.0, 19.0)  # FHWA-HRT-13-092, ACP 4 and ACP 6: mean width
SEVERITIES: tuple[str, ...] = ("low", "moderate", "high")
EIGHT = np.ones((3, 3), bool)  # 8-connectivity


# ---------------------------------------------------------------------------------------------
# Images, masks, centrelines


@dataclass
class Split:
    """One of the dataset's three splits, as published."""

    name: str
    stems: list[str]
    x: np.ndarray  # (N, H, W) uint8 range image: darker = lower (a depression)
    y: np.ndarray  # (N, H, W) bool, mask > cut


def load_split(name: str, cut: int = MASK_CUT) -> Split:
    """Read every image and mask of one split.

    Args:
        name: ``"Train"``, ``"Val"``, or ``"Test"``.
        cut: A mask pixel is crack when its value exceeds ``cut``.

    Returns:
        The split, images and masks stacked in file-name order.
    """
    d = require(RAW / name, "fetch_3dcrack.sh")
    stems = sorted(p.stem for p in (d / "img").glob("*.png"))
    x = np.stack([np.asarray(Image.open(d / "img" / f"{s}.png")) for s in stems])
    y = np.stack([np.asarray(Image.open(d / "anno" / f"{s}.png")) > cut for s in stems])
    assert x.shape == y.shape == (len(stems), H, W), (x.shape, y.shape)
    return Split(name, stems, x, y)


@dataclass
class Branch:
    """One crack branch of a centreline file: its points and two widths at each point."""

    split: str
    stem: str
    rc: np.ndarray  # (n, 2) int pixel coordinates, [row, column]
    mask_mm: np.ndarray  # (n,) ``width_mm``, from the crack mask; 0 marks an end point
    range_mm: np.ndarray  # (n,) ``width_mm_rangeimg``, from the range image

    @property
    def key(self) -> tuple[str, str]:
        """(split, stem): stems alone repeat across splits."""
        return self.split, self.stem

    @property
    def length_m(self) -> float:
        """Length of the polyline in metres."""
        return float(np.linalg.norm(np.diff(self.rc, axis=0), axis=1).sum() * MM_PER_PX / 1000)

    @property
    def inner(self) -> np.ndarray:
        """Points that carry a width (end points carry 0, a placeholder)."""
        return self.mask_mm > 0


def load_branches(name: str) -> list[Branch]:
    """Read the centreline files of one split (crack-free samples have none).

    Args:
        name: The split.

    Returns:
        Every branch, in file order.
    """
    out = []
    for g in sorted((RAW / name / "CVM").glob("*.geojson")):
        for f in json.loads(g.read_text())["features"]:
            rc = np.asarray(f["geometry"]["coordinates"], dtype=float)
            p = f["properties"]
            a, b = np.asarray(p["width_mm"], float), np.asarray(p["width_mm_rangeimg"], float)
            assert rc.shape == (len(a), 2) and len(a) == len(b), (g, rc.shape, len(a), len(b))
            assert np.array_equal(rc, np.round(rc)), g  # pixel centres
            rc = rc.astype(int)
            assert (rc[:, 0] < H).all() and (rc[:, 1] < W).all() and (rc >= 0).all(), g
            out.append(Branch(name, g.stem, rc, a, b))
    return out


@dataclass
class MaskFacts:
    """What the masks hold besides 0 and 255, and the images published twice."""

    grey_masks: int  # masks with values strictly between 0 and 255
    faint: list[dict] = field(default_factory=list)  # masks with a value > 0 on most pixels
    twins: list[list[tuple[str, int]]] = field(default_factory=list)  # (split, index) groups

    @property
    def faint_keys(self) -> set[tuple[str, str]]:
        """(split, stem) of the faint masks."""
        return {(f["split"], f["stem"]) for f in self.faint}


def mask_facts(splits: dict[str, Split]) -> MaskFacts:
    """Find grey mask values, faint masks, and images published twice.

    Masks are documented as 0 / 255, but some carry grey values: most at a crack's edge; four
    (found here, not assumed) carry a faint value on almost every pixel, so ``mask > 0`` would
    call them nearly all crack. Two images appear in two splits with identical pixels and
    different masks: the same scan annotated twice.

    Args:
        splits: The three splits (masks cut at ``MASK_CUT``).

    Returns:
        The counts, the faint masks, and the groups of identical images.
    """
    facts, digests = MaskFacts(0), defaultdict(list)
    for s in splits.values():
        for i, stem in enumerate(s.stems):
            m = np.asarray(Image.open(RAW / s.name / "anno" / f"{stem}.png"))
            facts.grey_masks += bool(((m > 0) & (m < 255)).any())
            if (m > 0).mean() > 0.5:
                facts.faint.append({"split": s.name, "stem": stem, "any": float((m > 0).mean()),
                                    "cut": float(s.y[i].mean())})  # fmt: skip
            digests[hashlib.sha1(s.x[i].tobytes(), usedforsecurity=False).hexdigest()].append(
                (s.name, i)
            )
    facts.twins = [v for v in digests.values() if len(v) > 1]
    return facts


def coarsen(s: Split) -> tuple[np.ndarray, np.ndarray]:
    """The same survey at 8 mm per pixel.

    Average 2 x 2 blocks of the image; a block is crack if any of its pixels is. Rows are
    padded from 312 to 320 so the U-Net can halve 4 times.

    Args:
        s: A split at 4 mm.

    Returns:
        Images (N, 320, 256) uint8 and masks (N, 320, 256) bool.
    """
    n = len(s.x)
    x = s.x.reshape(n, H // 2, 2, W // 2, 2).astype(np.float32).mean(axis=(2, 4))
    y = s.y.reshape(n, H // 2, 2, W // 2, 2).any(axis=(2, 4))
    x = np.pad(np.round(x).astype(np.uint8), ((0, 0), (0, 8), (0, 0)), mode="edge")
    y = np.pad(y, ((0, 0), (0, 8), (0, 0)))
    assert x.shape == y.shape == (n, 320, W // 2), (x.shape, y.shape)
    return x, y


# ---------------------------------------------------------------------------------------------
# Rulers: width from the distance transform, at the centreline or at the mask's own ridge


def width_mm(d: np.ndarray | float) -> np.ndarray:
    """Width (2 d - 1) x pixel size, the dataset's convention; 0 where d is 0.

    d is the Euclidean distance from a crack pixel to the nearest background pixel. On a
    straight band w pixels wide the centre has d = ceil(w / 2), so the rule returns w for odd
    w and w - 1 for even w: one- and two-pixel lines both read 4 mm. Diagonal geometry gives
    d = sqrt(2) or sqrt(5), hence the 7.3 and 13.9 mm steps.

    Args:
        d: Distances, in pixels.

    Returns:
        Widths in mm.
    """
    d = np.asarray(d, float)
    return np.where(d > 0, (2 * d - 1) * MM_PER_PX, 0.0)


def ruler(mask: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """The mask ruler: width at the given points from the mask's distance transform.

    Args:
        mask: A crack mask, (H, W) bool.
        rc: Points, (n, 2) [row, column].

    Returns:
        (n,) widths in mm; 0 where a point is off the mask.
    """
    return width_mm(distance_transform_edt(mask)[rc[:, 0], rc[:, 1]])


def thin(mask: np.ndarray) -> np.ndarray:
    """A one-pixel skeleton of a mask (Zhang & Suen, 1984), keeping every component.

    Zhang-Suen deletes a 2 x 2 square entirely; a component that loses all its pixels keeps
    the one farthest from the background.

    Args:
        mask: (H, W) bool.

    Returns:
        (H, W) bool skeleton, a subset of ``mask``.
    """
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
    comps, n = label(mask, structure=EIGHT)
    lost = np.setdiff1d(np.arange(1, n + 1), np.unique(comps[skel]))
    if lost.size:
        d = distance_transform_edt(mask)
        for r, c in maximum_position(d, comps, lost):
            skel[int(r), int(c)] = True
    return skel


@dataclass
class Ridge:
    """A mask, its distance transform, and the nearest skeleton pixel to every pixel.

    The ridge ruler reads any mask the same way: a centreline point is found when the mask has
    a pixel within ``reach`` px (3 x 3 for reach 1), and its width is read at the mask's
    skeleton pixel nearest the point, where the distance transform is half the width. Not at
    the point itself (a predicted crack can sit a pixel off the true centreline), and not as
    the largest distance nearby (that reads wide next to every wider spot).
    """

    mask: np.ndarray  # (H, W) bool
    d: np.ndarray  # (H, W) distance to the background, px
    near: np.ndarray  # (2, H, W) row and column of the nearest skeleton pixel

    @classmethod
    def of(cls, mask: np.ndarray) -> Ridge:
        """Thin the mask and index its skeleton.

        Args:
            mask: (H, W) bool.

        Returns:
            The ridge of ``mask``.
        """
        skel = thin(mask)
        if not skel.any():
            return cls(mask, np.zeros(mask.shape), np.zeros((2, *mask.shape), int))
        _, near = distance_transform_edt(~skel, return_indices=True)
        return cls(mask, distance_transform_edt(mask), near)

    def width(self, rc: np.ndarray, reach: int = 1) -> np.ndarray:
        """Widths at centreline points by the ridge ruler.

        Args:
            rc: Points, (n, 2) [row, column].
            reach: Search radius for a mask pixel, px (Chebyshev).

        Returns:
            (n,) widths in mm; 0 where the mask has no pixel within ``reach``.
        """
        r, c = rc[:, 0], rc[:, 1]
        padded = np.pad(self.mask, reach)
        found = np.zeros(len(rc), bool)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                found |= padded[r + reach + dr, c + reach + dc]
        rr, cc = self.near[0][r, c], self.near[1][r, c]
        return np.where(found, width_mm(self.d[rr, cc]), 0.0)


def skeleton_length_m(mask: np.ndarray) -> float:
    """Crack length of a mask: its skeleton's 8-connected path length, in metres.

    Steps between 4-neighbours count 1 px; a diagonal step counts sqrt(2) px only where no
    4-neighbour path joins the two pixels already (so an L-corner is not counted twice).

    Args:
        mask: (H, W) bool.

    Returns:
        Length in metres.
    """
    s = thin(mask)
    orth = int((s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum())
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    return (orth + math.sqrt(2) * int(d1.sum() + d2.sum())) * MM_PER_PX / 1000


def band(width: np.ndarray | float) -> np.ndarray:
    """LTPP severity from a crack's mean width: 0 low (up to 6 mm), 1 moderate (up to 19), 2 high.

    The rule is the one for longitudinal and transverse cracks in asphalt (FHWA-HRT-13-092,
    ACP 4 and ACP 6; the adjacency rules for random cracking are left out). 3DCrack does not
    label crack types, so the lecture applies it to every branch, alligator cracking included,
    which LTPP rates by area instead (ACP 1): "all cracking treated as linear".
    """
    return np.digitize(width, LTPP_MM, right=True)
