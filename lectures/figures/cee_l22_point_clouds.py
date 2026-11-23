"""Figures and numbers for CEE Lecture 22: point clouds, PointNet, and registration.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l22_point_clouds.py \
        --out build/fig/cee_l22_point_clouds

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

The question: was the CRAS lab built as its BIM model says? A laser scan of four rooms and a
hallway (Abreu et al., 2023) is compared with the as-built model: label each point (part of
the building, or clutter), line the scan up with the model, and see which elements the scan
confirms and where it departs from them.

- Data: ``craslab.py``: the 2% sample of the 584,701,977 points, thinned to one point per
  3-cm voxel; the 34 labels grouped into ceiling, floor, wall, door, window, and clutter;
  rooms cut between wall faces; the IFC model meshed with IfcOpenShell.
- Labelling, each room scored by models trained on the other four (leave one room out): the
  commonest class of each 5-cm height band; covariance features of each point's 24 nearest
  neighbours (Lecture 5's eigenvalues; Weinmann et al., 2015) with gradient-boosted trees
  (Lecture 8) trained on every labelled point of the four training rooms, as PointNet is;
  PointNet (Qi et al., 2017) written from scratch, without its alignment
  networks, on 1.5-m columns of 4,096 points; and the same network without its max-pooled
  global feature. The network's settings were fixed before the leave-one-room-out runs, on one
  run that held out room 110 and validated on room 111.
- Checks: relabelling the points permutes the outputs; the critical points alone give the same
  global feature; the SVD alignment (Kabsch) agrees with SciPy's; ICP recovers a known motion.
- Registration: ICP from scratch (point to point, Besl & McKay, 1992), the scan's points to
  points spread over the model's surfaces, from a coarse start that matches bounding boxes, with
  a distance cap that shrinks from 1 m to 10 cm (a trimmed fit, Lecture 9).
- Scan vs BIM: each element's coverage, the share of its surface within 5 cm of a scan point;
  each scan point's distance to the plane of the nearest model surface. A wall's points are the
  scan points whose nearest model surface is that wall (within 30 cm): its labelled points (the
  dataset's hand labels wall, door, and window, not PointNet's) and its unassigned points (no
  label). A wall is seen when it has at least 100 of the two together; it is judged when at
  least 100 are labelled, and off when their median distance exceeds 5 cm, unless at least 100
  unassigned points sit within 5 cm (two surfaces: conflicting).
- Break it: a random split instead of rooms; a sparser scan; ICP from a turned start; plain ICP
  (no cap) with and without clutter.
"""

from __future__ import annotations

import copy
import itertools
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

import aml_style
import craslab
import title_art
from common import parse_out_dir, seed_all, write_numbers

SEED = 22
VOXEL = 0.03  # m: one point per voxel for labelling
BLOCK = 1.5  # m: the side of a column of points the network reads at once
NPTS = 4096  # points per column (Qi et al., 2017)
MIN_BLOCK = 200  # columns with fewer points are skipped in training
BATCH = 16
EPOCHS = 100  # cosine schedule to zero; the final network is used (no early stopping)
LR = 1e-3
WEIGHT_DECAY = 1e-2
K_EIG = 24  # neighbours for the covariance features
HEIGHT_BIN = 0.05  # m
# The trees train on every labelled point of the training rooms. They were first trained on a
# draw of 400,000 of them; that draw is still made (and not used), so the later draws from the
# same generator, the sparser scans and the random split's columns, stay the same points.
TREE_DRAW = 400_000
TREE_ROUNDS = 300  # boosting rounds allowed; early stopping must end every fit before them
SPARSE = 0.1  # share of the test room's points kept in the sparser-scan test
REG_VOXEL = 0.05  # m: the scan for the reference registration
BASIN_VOXEL = 0.10  # m: the scan for the start-angle and no-cap tests
BIM_SPACING = 0.05  # m between points spread over the model's surfaces (for ICP, coverage)
DEV_SPACING = 0.02  # m: finer, for the wall-to-plane distances
CAPS = (1.0, 0.5, 0.2, 0.1)  # m: ICP's shrinking distance cap
ICP_ITERS = 100  # per cap
TURNS = (0, 10, 20, 30, 45, 90, 180)  # degrees about z, added to the coarse start
CONVERGED = 0.01  # m: a start "converges" if it lands within this of the reference fit
COVER_D = 0.05  # m: a model surface point is covered if a scan point is this close
SEEN = 0.10  # the coverage figure's line: a tenth of an element's surface covered
ASSIGN_D = 0.30  # m: a scan point is compared with the nearest element within this distance
VERTICAL = (2, 3, 4)  # target indices of a wall's own points: wall, door, window
MIN_POINTS = 100  # the fewest points paired with a wall for each test of ``verdict``
MIN_POINTS_SWEEP = (25, 50, 100, 200, 400, 1000)  # that threshold, varied
OFF = 0.05  # m: a judged wall whose labelled points sit a median of more than this from it is off
VERDICTS = ("within", "off", "conflicting", "unlabelled", "hidden")  # ``verdict``'s answers
DEV_MAX = 0.10  # m: the deviation map's colour scale
LABEL_SHIFT = (0.5, 0.5)  # m: a wall's number, up and right of its centre
LABEL_GAP = 1.3  # m: numbers closer than this (in x and in y) are moved apart
FULL_SPAN = 1.2  # m: the shown column must span this much in x and y
SLICE = (0.1, 2.0)  # m above the floor: the plan views leave out floor and ceiling
KNOWN_TURN = (1.0, -2.0, 10.0)  # degrees about x, y, z: the known motion ICP must recover
KNOWN_SHIFT = (0.5, -0.3, 0.1)  # m
STRUCTURE = tuple(range(5))  # target indices of the building: ceiling to window
IFC_NAMES = {  # the IFC types as the slides name them
    "IfcWallStandardCase": "wall",
    "IfcSlab": "slab",
    "IfcCovering": "ceiling",
    "IfcColumn": "column",
    "IfcDoor": "door",
    "IfcWindow": "window",
    "IfcStairFlight": "stair",
}
assert set(IFC_NAMES) == set(craslab.BIM_TYPES)
# Ceiling blue, floor brown, wall navy, door orange, window green, clutter pink: each at least
# 3:1 against white (5.2, 6.4, 11.6, 3.9, 3.4, 3.1)
CLASS_COLORS = ("#0072B2", "#8C510A", "#1B3A5C", "#D55E00", "#009E73", "#CC79A7")
NETS = {  # name: (global feature, inputs per point)
    "PointNet, no global feature": (False, 7),
    "PointNet": (True, 7),
    "PointNet + covariance": (True, 13),
}
METHODS = ("height lookup", "covariance + trees", *NETS)
LEARNED = ("covariance + trees", "PointNet", "PointNet + covariance")  # the sparser-scan test


def log(msg: str) -> None:
    """Progress to stderr (the Makefile sends stdout to /dev/null)."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Cloud:
    """The thinned scan.

    Attributes:
        xyz: Coordinates in the scan's frame, m, (n, 3) float64.
        feat: Network inputs (n, 7) float32: x, y, height above the floor, r, g, b,
            intensity (colours and intensity in [-0.5, 0.5]).
        y: Target index into ``craslab.TARGETS``, or ``craslab.IGNORE``, (n,).
        room: Index into ``craslab.ROOMS``, (n,).
        eig: The six covariance features of ``eigen_features`` (n, 6), or None.
    """

    xyz: np.ndarray
    feat: np.ndarray
    y: np.ndarray
    room: np.ndarray
    eig: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> Cloud:
        """The points ``idx`` (their covariance features are not recomputed)."""
        eig = None if self.eig is None else self.eig[idx]
        return Cloud(self.xyz[idx], self.feat[idx], self.y[idx], self.room[idx], eig)

    def with_eig(self) -> Cloud:
        """The same points, with covariance features computed among themselves."""
        return Cloud(self.xyz, self.feat, self.y, self.room, eigen_features(self.xyz))


def voxel_first(xyz: np.ndarray, size: float) -> np.ndarray:
    """Indices of one point per voxel of side ``size`` (the first in file order), sorted."""
    key = np.floor(xyz / size).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


def prepare(d: dict[str, np.ndarray], voxel: float | None) -> Cloud:
    """The sample, thinned to one point per voxel of side ``voxel`` (``None``: every point)."""
    keep = (
        np.arange(len(d["label"]))
        if voxel is None
        else voxel_first(d["xyz"].astype(np.float64), voxel)
    )
    xyz = d["xyz"][keep].astype(np.float64)
    feat = np.column_stack(
        [
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2] - craslab.FLOOR_Z,
            d["rgb"][keep] / 255.0 - 0.5,
            d["intensity"][keep] - 0.5,
        ]
    ).astype(np.float32)
    assert feat.shape == (len(keep), 7), feat.shape
    return Cloud(xyz, feat, craslab.target_of(d["label"][keep]), craslab.room_of(xyz))


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Confusion counts over the labelled points, rows true, columns predicted."""
    k = len(craslab.TARGETS)
    m = y >= 0
    return np.bincount(y[m] * k + pred[m], minlength=k * k).reshape(k, k)


def scores(cm: np.ndarray) -> dict[str, object]:
    """Per-class IoU (Lecture 16), their mean, and accuracy, from confusion counts."""
    inter = np.diag(cm)
    iou = inter / np.maximum(cm.sum(0) + cm.sum(1) - inter, 1)
    return {
        "iou": dict(zip(craslab.TARGETS, iou.tolist(), strict=True)),
        "miou": float(iou.mean()),
        "accuracy": float(inter.sum() / cm.sum()),
    }


# ---------------------------------------------------------------- baselines


def eigen_features(xyz: np.ndarray, k: int = K_EIG) -> np.ndarray:
    """Covariance features of each point's ``k`` nearest neighbours (Weinmann et al., 2015).

    With the eigenvalues l1 >= l2 >= l3 of the neighbourhood's covariance and n the
    eigenvector of l3 (the local normal): linearity (l1 - l2)/l1, planarity (l2 - l3)/l1,
    scattering l3/l1, verticality 1 - |n_z|, size sqrt(l1), and the neighbourhood's height range.

    Returns:
        Features, shape (n, 6).
    """
    _, nb = cKDTree(xyz).query(xyz, k=k, workers=-1)  # (n, k)
    out = np.empty((len(xyz), 6))
    step = 200_000
    for s in range(0, len(xyz), step):
        p = xyz[nb[s : s + step]]  # (b, k, 3)
        c = p - p.mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum("bki,bkj->bij", c, c) / k)  # ascending; (b, 3), (b, 3, 3)
        l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2] + 1e-12
        out[s : s + step] = np.column_stack(
            [
                (l1 - l2) / l1,
                (l2 - l3) / l1,
                l3 / l1,
                1 - np.abs(v[:, 2, 0]),
                np.sqrt(l1),
                np.ptp(p[:, :, 2], axis=1),
            ]
        )
    return out


def tree_inputs(c: Cloud) -> np.ndarray:
    """The trees' inputs: height, the six covariance features, colour, intensity."""
    return np.column_stack([c.feat[:, 2], c.eig, c.feat[:, 3:]])


def height_lookup(train: Cloud, test: Cloud) -> np.ndarray:
    """Each 5-cm height band's commonest class in training, applied to the test points."""

    def band(c: Cloud) -> np.ndarray:
        return np.clip((c.feat[:, 2] / HEIGHT_BIN).astype(int), 0, 79)

    m = train.y >= 0
    table = np.zeros((80, len(craslab.TARGETS)), np.int64)
    np.add.at(table, (band(train)[m], train.y[m]), 1)
    return table.argmax(1)[band(test)]


def fit_trees(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees on every labelled point of the training set.

    Every defining parameter is stated, although each equals scikit-learn's default (with this
    many points, its "auto" early stopping is on): early stopping holds out 10% of the points
    (stratified, drawn with ``SEED``) and stops after 10 rounds without a lower loss on them.

    Args:
        x: Inputs, (n, 15).
        y: Targets, (n,), ``craslab.IGNORE`` for unlabelled points.
        rng: The labelling stage's generator; it makes the old ``TREE_DRAW`` draw (unused).

    Returns:
        The fitted trees.
    """
    lab = np.flatnonzero(y >= 0)
    rng.choice(len(lab), min(TREE_DRAW, len(lab)), replace=False)  # keeps later draws (above)
    trees = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=TREE_ROUNDS,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=SEED,
    )
    return trees.fit(x[lab], y[lab])


# ---------------------------------------------------------------- PointNet


class PointNet(nn.Module):
    """PointNet for segmentation (Qi et al., 2017), without the alignment networks.

    Each point is lifted by the same small MLP (a 1 x 1 convolution over the points) to 64
    numbers (its local feature), then to 1,024; the maximum of each of the 1,024 over the column's
    points is the global feature. Each point's label comes from its local feature and the global
    one. Without ``global_feature``, each point is labelled from its own 64 numbers alone.
    With ``n_in = 13``, each point also brings its six covariance features (its neighbourhood's
    shape), which plain PointNet does not see.
    """

    def __init__(self, n_in: int = 7, n_out: int = 6, global_feature: bool = True) -> None:
        """Layers for ``n_in`` inputs per point and ``n_out`` classes."""
        super().__init__()

        def mlp(*dims: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for a, b in itertools.pairwise(dims):
                layers += [nn.Conv1d(a, b, 1), nn.BatchNorm1d(b), nn.ReLU()]
            return nn.Sequential(*layers)

        self.global_feature = global_feature
        self.n_in = n_in
        self.local = mlp(n_in, 64, 64)
        self.lift = mlp(64, 128, 1024)
        head_in = 64 + (1024 if global_feature else 0)
        self.head = nn.Sequential(
            mlp(head_in, 512, 256, 128), nn.Dropout(0.3), nn.Conv1d(128, n_out, 1)
        )

    def pooled(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Local features, global feature, and the point that sets each global number.

        Args:
            x: Points, (B, N, n_in), with ``n_in`` 7 or 13.

        Returns:
            Local (B, 64, N), global (B, 1024), and argmax (B, 1024): for each global number,
            the index of the point holding the maximum.
        """
        loc = self.local(x.transpose(1, 2))  # (B, n_in, N) -> (B, 64, N)
        g, arg = self.lift(loc).max(dim=2)  # (B, 1024, N) -> (B, 1024)
        return loc, g, arg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class scores per point: (B, N, n_in) -> (B, N, 6), with ``n_in`` 7 or 13."""
        if self.global_feature:
            loc, g, _ = self.pooled(x)
            h = torch.cat([loc, g[:, :, None].expand(-1, -1, loc.shape[2])], dim=1)
        else:
            h = self.local(x.transpose(1, 2))
        return self.head(h).transpose(1, 2)


@dataclass(frozen=True)
class Block:
    """A column of points: their indices into the cloud, and the column's centre (x, y)."""

    idx: np.ndarray
    centre: np.ndarray


def blocks(c: Cloud, mask: np.ndarray, stride: float, min_points: int = 1) -> list[Block]:
    """Columns of side ``BLOCK`` over the points in ``mask``, every ``stride`` m.

    With ``stride = BLOCK`` the columns tile the points: each point is in exactly one.
    """
    idx = np.flatnonzero(mask)
    p = c.xyz[idx, :2]
    lo = p.min(0)
    cells = np.floor((p - lo) / stride).astype(np.int64)  # the cell each point starts in
    n_over = int(round(BLOCK / stride))  # columns overlapping a cell, per axis
    out = []
    for cx in range(cells[:, 0].max() + 1):
        for cy in range(cells[:, 1].max() + 1):
            m = (
                (cells[:, 0] >= cx)
                & (cells[:, 0] < cx + n_over)
                & (cells[:, 1] >= cy)
                & (cells[:, 1] < cy + n_over)
            )
            if m.sum() >= min_points:
                corner = lo + stride * np.array([cx, cy])
                out.append(Block(idx[m], corner + BLOCK / 2))
    return out


def columns(
    c: Cloud, bl: list[Block], picks: list[np.ndarray], device: str, n_in: int = 7
) -> torch.Tensor:
    """Network inputs for the given points of each column, x and y relative to its centre.

    With ``n_in = 13`` the six covariance features follow the seven of ``Cloud.feat``.
    """
    f = c.feat if n_in == 7 else np.hstack([c.feat, c.eig.astype(np.float32)])
    x = np.stack([f[p] for p in picks])  # (B, NPTS, n_in)
    assert x.shape[2] == n_in, x.shape
    x[:, :, :2] -= np.stack([b.centre for b in bl])[:, None].astype(np.float32)
    return torch.from_numpy(x).to(device)


def augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Turn each column by a random angle about z, and mirror half of them."""
    a = rng.uniform(0, 2 * np.pi, len(x))
    rot = np.stack([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).transpose(2, 0, 1)
    rot[rng.random(len(x)) < 0.5, :, 0] *= -1  # mirror x
    rot_t = torch.from_numpy(rot.astype(np.float32)).to(x.device)  # (B, 2, 2)
    xy = torch.einsum("bij,bnj->bni", rot_t, x[:, :, :2])  # (B, N, 2)
    return torch.cat([xy, x[:, :, 2:]], dim=2)


def train_pointnet(
    c: Cloud, train_blocks: list[Block], *, global_feature: bool, n_in: int, device: str
) -> tuple[PointNet, dict[str, object]]:
    """Train from scratch: AdamW, cosine schedule over ``EPOCHS``, random points each epoch."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = PointNet(n_in=n_in, global_feature=global_feature).to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    curve = []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(len(train_blocks))
        total, n = 0.0, 0
        for s in range(0, len(order), BATCH):
            bl = [train_blocks[i] for i in order[s : s + BATCH]]
            picks = [rng.choice(b.idx, NPTS, replace=len(b.idx) < NPTS) for b in bl]
            x = augment(columns(c, bl, picks, device, n_in), rng)
            y = torch.from_numpy(np.stack([c.y[p] for p in picks])).to(device)
            loss = nn.functional.cross_entropy(
                model(x).reshape(-1, len(craslab.TARGETS)),
                y.reshape(-1),
                ignore_index=craslab.IGNORE,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        sched.step()
        curve.append(total / n)
    info = {"curve": curve, "seconds": time.perf_counter() - t0}
    info["parameters"] = sum(p.numel() for p in model.parameters())
    return model, info


@torch.no_grad()
def predict_pointnet(
    model: PointNet, c: Cloud, bl: list[Block], device: str, seed: int = SEED
) -> np.ndarray:
    """Label every point of the tiling columns ``bl``.

    Each column's points, shuffled, go through in sets of ``NPTS``, the last set filled up with
    repeats of its own column's points.

    Returns:
        Predicted target per point of ``c`` (-1 where no column covers it).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    pred = np.full(len(c.y), -1)
    for b in bl:
        perm = rng.permutation(b.idx)
        n = -(-len(perm) // NPTS) * NPTS
        pad = np.concatenate([perm, rng.choice(perm, n - len(perm))])
        sets = pad.reshape(-1, NPTS)
        x = columns(c, [b] * len(sets), list(sets), device, model.n_in)
        out = model(x).argmax(-1).cpu().numpy()
        pred[sets.ravel()] = out.ravel()
    return pred


def pointnet_checks(model: PointNet, c: Cloud, b: Block, device: str) -> dict[str, object]:
    """Relabel the points: the outputs permute. Keep only the critical points: same global.

    Run on a float64 copy on the CPU: on the GPU, TF32 convolutions round differently for
    different numbers of points, which would hide whether the identities hold.
    """
    m64 = copy.deepcopy(model).double().cpu().eval()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(b.idx, NPTS, replace=len(b.idx) < NPTS)
    perm = rng.permutation(NPTS)
    with torch.no_grad():
        x = columns(c, [b], [pick], device).double().cpu()
        out, out_p = m64(x)[0], m64(x[:, perm])[0]
        _, g, arg = m64.pooled(x)
        crit = torch.unique(arg[0])
        _, g_crit, _ = m64.pooled(x[:, crit])
    return {
        "permutation_max_diff": float((out[perm] - out_p).abs().max()),
        "critical_points": len(crit),
        "points": NPTS,
        "critical_global_max_diff": float((g - g_crit).abs().max()),
    }


# ---------------------------------------------------------------- registration


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and shift that best move points ``p`` onto ``q`` (least squares, via SVD).

    Args:
        p: Points (n, 3).
        q: Their partners (n, 3).

    Returns:
        R (3, 3) and t (3,) minimising sum |R p_i + t - q_i|^2, with det R = +1.
    """
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)  # (3, 3) cross-covariance
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))  # -1 would be a reflection: flip the last axis
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - r @ pc


def icp(
    src: np.ndarray,
    tree: cKDTree,
    tgt: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    *,
    caps: tuple[float, ...] = CAPS,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Iterative closest point (Besl & McKay, 1992), with a shrinking distance cap.

    Pair each moved point with its nearest target point within the cap, solve for the motion
    (Kabsch), and repeat until the motion stops changing; then the next, smaller cap.

    Returns:
        R, t (the motion from the start), and per cap: iterations, share of points paired,
        median paired distance.
    """
    hist = []
    for cap in caps:
        it = 0
        while it < ICP_ITERS:
            it += 1
            moved = src @ r.T + t
            dist, idx = tree.query(moved, distance_upper_bound=cap, workers=-1)
            ok = np.isfinite(dist)
            dr, dt = kabsch(moved[ok], tgt[idx[ok]])
            r, t = dr @ r, dr @ t + dt
            angle = np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1))
            if np.linalg.norm(dt) < 1e-6 and angle < 1e-7:
                break
        hist.append(
            {
                "cap": cap,
                "iterations": it,
                "paired": float(ok.mean()),
                "median": float(np.median(dist[ok])),
            }
        )
    return r, t, hist


def motion_error(
    r: np.ndarray, t: np.ndarray, r_ref: np.ndarray, t_ref: np.ndarray, p: np.ndarray
) -> float:
    """RMS distance between the points moved by (r, t) and by the reference motion, m.

    The reference is itself a fit (capped ICP on the labelled structure), not the true pose:
    this is the distance from the reference fit, not an error against the truth.
    """
    return float(np.sqrt((((p @ r.T + t) - (p @ r_ref.T + t_ref)) ** 2).sum(1).mean()))


def registration_checks(scan: np.ndarray) -> dict[str, float]:
    """Kabsch against SciPy on known pairs; ICP recovering a known motion of the scan itself.

    The scan is centred first, so the known turn is about its middle (as the coarse start
    leaves it), not about a corner of the scanner's frame.
    """
    scan = scan - scan.mean(0)
    rng = np.random.default_rng(SEED)
    r0 = Rotation.from_euler("xyz", KNOWN_TURN, degrees=True).as_matrix()
    t0 = np.asarray(KNOWN_SHIFT)
    p = scan[rng.choice(len(scan), 1000, replace=False)]
    r, t = kabsch(p, p @ r0.T + t0)
    pc = p - p.mean(0)
    r_sp, _ = Rotation.align_vectors(pc @ r0.T, pc)  # SciPy: R minimising |a - R b|
    moved = scan @ r0.T + t0
    r_i, t_i, hist = icp(scan, cKDTree(moved), moved, np.eye(3), np.zeros(3))
    return {
        "kabsch_vs_scipy": float(np.abs(r - r_sp.as_matrix()).max()),
        "kabsch_vs_truth": float(max(np.abs(r - r0).max(), np.abs(t - t0).max())),
        "icp_known_rms": motion_error(r_i, t_i, r0, t0, scan),
        "icp_known_iterations": int(sum(h["iterations"] for h in hist)),
        "icp_points": len(scan),
    }


def bim_cloud(els: list[craslab.Element], spacing: float, seed: int) -> dict[str, np.ndarray]:
    """Points spread over every element's surfaces, with each point's element and normal."""
    rng = np.random.default_rng(seed)
    parts = [craslab.sample_surface(e, spacing, rng) for e in els]
    return {
        "xyz": np.concatenate([p for p, _ in parts]),
        "normal": np.concatenate([n for _, n in parts]),
        "owner": np.repeat(np.arange(len(els)), [len(p) for p, _ in parts]),
    }


def coarse_start(scan: np.ndarray, bim: np.ndarray) -> np.ndarray:
    """The shift that matches the centres of the two bounding boxes (no turn)."""
    return (bim.min(0) + bim.max(0)) / 2 - (scan.min(0) + scan.max(0)) / 2


# ---------------------------------------------------------------- figures


def _scatter3(
    ax: plt.Axes, p: np.ndarray, color: object, s: float = 1.0, alpha: float = 0.6
) -> None:
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color, s=s, alpha=alpha, linewidths=0, depthshade=False)


def picture(
    c: Cloud, b: Block, pick: np.ndarray, crit: np.ndarray, pred: np.ndarray, *, out_dir: Path
) -> None:
    """The one picture, three frames: a column's points; its critical points; its labels."""
    p = c.xyz[pick] - np.r_[b.centre, craslab.FLOOR_Z]  # x, y from the centre; z from the floor
    frames = [
        ("picture_1", [(p, "#7a7a7a", 1.0)]),
        ("picture_2", [(p, "#c8c8c8", 0.8), (p[crit], aml_style.PALETTE[1], 6.0)]),
        ("picture_3", [(p[pred == k], CLASS_COLORS[k], 1.5) for k in range(len(craslab.TARGETS))]),
    ]
    for name, layers in frames:
        fig = plt.figure(figsize=aml_style.FULL)
        ax = fig.add_subplot(projection="3d")
        for q, col, s in layers:
            _scatter3(ax, q, col, s)
        aml_style.style3d(ax, elev=22, azim=-60)
        ax.set_box_aspect((1, 1, np.ptp(p[:, 2]) / BLOCK))
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_zlabel("height, m")
        if name == "picture_3":
            for k, nm in enumerate(craslab.TARGETS):
                ax.scatter([], [], [], c=CLASS_COLORS[k], s=30, label=nm)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
        aml_style.save(fig, out_dir, name)


def attempt_figure(c: Cloud, b: Block, pick: np.ndarray, out_dir: Path) -> None:
    """The attempt: the column from above and from the side, every point grey."""
    p = c.xyz[pick] - np.r_[b.centre, craslab.FLOOR_Z]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=aml_style.FULL)
    a1.scatter(p[:, 0], p[:, 1], s=1, c="#555555", linewidths=0)
    a1.set(xlabel="x, m", ylabel="y, m", title="from above", aspect="equal")
    a2.scatter(p[:, 0], p[:, 2], s=1, c="#555555", linewidths=0)
    a2.set(xlabel="x, m", ylabel="height, m", title="from the side", aspect="equal")
    aml_style.save(fig, out_dir, "attempt")


def plan_labels(c: Cloud, room: int, pred: np.ndarray, out_dir: Path) -> float:
    """One room from above (floor and ceiling left out): the labels; where PointNet is wrong.

    Returns:
        The share of the slice's labelled points PointNet labels wrong.
    """
    h = c.feat[:, 2]
    m = (c.room == room) & (h > SLICE[0]) & (h < SLICE[1]) & (c.y >= 0)
    xy, lab, wrong = c.xyz[m, :2], c.y[m], pred[m] != c.y[m]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=aml_style.FULL)
    a1.scatter(*xy[lab <= 1].T, s=0.3, c="#dddddd", linewidths=0, rasterized=True)
    for k in range(2, len(craslab.TARGETS)):
        a1.scatter(*xy[lab == k].T, s=0.3, c=CLASS_COLORS[k], linewidths=0, rasterized=True)
        a1.scatter([], [], c=CLASS_COLORS[k], s=30, label=craslab.TARGETS[k])
    a1.legend(loc="center right", bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=10)
    a2.scatter(*xy[~wrong].T, s=0.3, c="#cccccc", linewidths=0, rasterized=True)
    a2.scatter(*xy[wrong].T, s=0.6, c=aml_style.PALETTE[1], linewidths=0, rasterized=True)
    a2.scatter([], [], c="#cccccc", s=30, label="right")
    a2.scatter([], [], c=aml_style.PALETTE[1], s=30, label="wrong")
    a2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    for ax, title in ((a1, "labels"), (a2, "PointNet: right or wrong")):
        ax.set(title=title, aspect="equal", xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
    aml_style.save(fig, out_dir, "labels")
    return float(wrong.mean())


def iou_figure(res: dict[str, dict], out_dir: Path) -> None:
    """Per-class IoU on the held-out rooms, pooled, one bar per method."""
    names = list(res)
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    w = 0.8 / len(names)
    x = np.arange(len(craslab.TARGETS))
    for i, n in enumerate(names):
        v = [res[n]["iou"][k] for k in craslab.TARGETS]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, v, w, color=aml_style.PALETTE[i % 4],
               label=f"{n} ({res[n]['miou']:.2f})", hatch=["", "//", "", ".."][i % 4])  # fmt: skip
    ax.set_xticks(x, craslab.TARGETS)
    ax.set_ylabel("IoU, rooms held out")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=10, ncols=2, loc="upper right")
    aml_style.save(fig, out_dir, "iou")


def registration_figure(
    scan: np.ndarray,
    h: np.ndarray,
    bim: np.ndarray,
    motions: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    """From above: the model's surfaces (grey) and the scan's walls (blue).

    Left, as given, each in its own frame; right, after ICP.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    bz = bim[:, 2]
    sel_b = (bz > 0.1 + SLICE[0]) & (bz < 0.1 + SLICE[1])  # the model's floor is at z = 0.1 m
    s = (h > SLICE[0]) & (h < SLICE[1])
    for ax, (r, t), title in zip(axes, motions, ("as given: two frames", "after ICP"), strict=True):
        ax.scatter(*bim[sel_b, :2].T, s=0.2, c="#9a9a9a", linewidths=0, rasterized=True)
        q = scan[s] @ r.T + t
        ax.scatter(*q[:, :2].T, s=0.2, c=aml_style.PALETTE[0], linewidths=0, rasterized=True)
        ax.set(title=title, aspect="equal", xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
    aml_style.save(fig, out_dir, "registration")


def basin_figure(turns: dict[int, float], out_dir: Path) -> None:
    """Where ICP lands (its distance from the reference fit), against the start's turn."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    a = np.array(list(turns))
    e = np.maximum(np.array(list(turns.values())), 1e-4)
    ax.plot(a, e, marker="o", color=aml_style.PALETTE[0])
    ax.axhline(CONVERGED, color=aml_style.MUTED, ls=":", lw=1)
    ax.text(
        a[-1],
        CONVERGED * 1.3,
        f"{CONVERGED * 100:.0f} cm",
        ha="right",
        color=aml_style.MUTED,
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xticks(a)
    ax.set_xlabel("start turned from the coarse start, degrees")
    ax.set_ylabel("from the reference fit, m (RMS)")
    aml_style.save(fig, out_dir, "basin")


def deviation_figure(p: np.ndarray, dev: np.ndarray, out_dir: Path) -> None:
    """From above: each wall, door, and window point coloured by its distance to the model."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    order = np.argsort(dev)
    sc = ax.scatter(*p[order, :2].T, c=dev[order], s=0.3, cmap="magma_r", vmin=0, vmax=DEV_MAX,
                    linewidths=0, rasterized=True)  # fmt: skip
    ax.set(aspect="equal", xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(sc, ax=ax, shrink=0.8, extend="max", label="distance to the model, m")
    aml_style.save(fig, out_dir, "deviation")


def coverage_figure(cov: dict[str, list[float]], out_dir: Path) -> None:
    """Each element's coverage, by IFC type."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    rng = np.random.default_rng(SEED)
    names = list(cov)
    for i, n in enumerate(names):
        v = np.asarray(cov[n])
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(v)), v, s=16, alpha=0.75, color=aml_style.PALETTE[0]
        )
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=aml_style.PALETTE[1], lw=3, ls="-")
    ax.axhline(SEEN, color=aml_style.MUTED, ls=":", lw=1)
    ax.text(  # outside the right spine: inside, it lands on a median bar
        1.01, SEEN, f"{SEEN:.0%}", transform=ax.get_yaxis_transform(), ha="left", va="center",
        color=aml_style.MUTED, fontsize=10, clip_on=False,
    )  # fmt: skip
    ax.set_xticks(range(len(names)), [f"{n}\n({len(cov[n])})" for n in names], fontsize=10)
    ax.set_ylabel(f"surface within {COVER_D * 100:.0f} cm of the scan")
    ax.set_ylim(-0.03, 1.03)
    aml_style.save(fig, out_dir, "coverage")


def verdict(row: dict, min_points: int = MIN_POINTS) -> str:
    """What the scan says about one wall or column (its points from ``compare_stage``).

    Labelled points (wall, door, window) are known to be the wall's surface or its openings;
    unassigned points carry no label, so off the model's plane they may be furniture in front
    of the wall, and only on it do they say that something stands where the model puts the wall.

    Args:
        row: The element's row: ``points`` and ``deviation`` (its labelled points, and their
            median distance to its plane, m), ``unassigned_points`` and ``unassigned_deviation``
            (the same for its unassigned points).
        min_points: The fewest points for each test.

    Returns:
        "hidden" (not seen): under ``min_points`` labelled and unassigned points together;
        "unlabelled": seen, but under ``min_points`` labelled points, so not judged;
        "within": the labelled points' median distance is at most ``OFF``;
        "conflicting": it is more, but at least ``min_points`` unassigned points sit a median
        of at most ``OFF`` from the plane (two surfaces);
        "off": otherwise.
    """
    if row["points"] + row["unassigned_points"] < min_points:
        return "hidden"
    if row["points"] < min_points:
        return "unlabelled"
    if row["deviation"] <= OFF:
        return "within"
    near = row["unassigned_deviation"] is not None and row["unassigned_deviation"] <= OFF
    return "conflicting" if row["unassigned_points"] >= min_points and near else "off"


def verdict_figure(
    els: list[craslab.Element], rows: list[dict], measure_k: list[int], footprint: np.ndarray,
    out_dir: Path,
) -> None:  # fmt: skip
    """From above: the model's walls and columns by verdict; the scan's points (light grey).

    The walls to measure on site (off or conflicting) are numbered as in ``measure_k`` (worst
    first). A wall's points are those of ``compare_stage``; the verdicts are ``verdict``'s.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.scatter(*footprint[:, :2].T, s=0.2, c="#e3e3e3", linewidths=0, rasterized=True)
    color = {
        "within": "#9a9a9a",
        "off": aml_style.PALETTE[1],
        "conflicting": aml_style.PALETTE[3],
        "unlabelled": aml_style.PALETTE[2],
        "hidden": aml_style.PALETTE[0],
    }
    label = {
        "within": f"within {OFF * 100:.0f} cm",
        "off": f"off by more than {OFF * 100:.0f} cm",
        "conflicting": "two surfaces: conflicting",
        "unlabelled": "seen, points unlabelled",
        "hidden": f"not seen: under {MIN_POINTS} points",
    }
    assert tuple(color) == VERDICTS and tuple(label) == VERDICTS
    for e, w in zip(els, rows, strict=True):
        if w["type"] in ("wall", "column"):
            xy = e.verts[:, :2]
            v = verdict(w)
            ax.add_patch(Polygon(xy[ConvexHull(xy).vertices], fc=color[v], ec=color[v], lw=1))
    for v, col in color.items():
        ax.scatter([], [], marker="s", s=40, c=col, label=label[v])
    shift, placed = np.asarray(LABEL_SHIFT), []  # label positions, m: one too close moves down
    for i, k in enumerate(measure_k, 1):
        anchor = els[k].verts[:, :2].mean(axis=0)
        pos = anchor + shift
        while any(np.abs(pos - q).max() < LABEL_GAP for q in placed):
            pos = pos - np.array([0.0, LABEL_GAP])
        placed.append(pos)
        moved = np.abs(pos - anchor - shift).max() > 0
        ink = color[verdict(rows[k])]
        ax.annotate(str(i), anchor, xytext=pos, textcoords="data", color=ink,
                    fontsize=11, fontweight="bold", va="center",
                    arrowprops={"arrowstyle": "-", "color": ink, "lw": 0.6}
                    if moved else None)  # fmt: skip
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    ax.set(aspect="equal", xticks=[], yticks=[])
    ax.autoscale_view()
    for side in ax.spines.values():
        side.set_visible(False)
    aml_style.save(fig, out_dir, "verdict")


def title_art_figure(out_dir: Path) -> None:
    """The title art: a dark room drawn in scattered points of light, one corner bright."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#08090d", "#10141c", "#151320", "#07070a"), grain=0.08)
    rng = np.random.default_rng(SEED)
    edges = [  # a room in two-point perspective: floor, walls, a doorway
        [(0.15, top * 0.25), (0.55, top * 0.12)], [(0.55, top * 0.12), (0.88, top * 0.28)],
        [(0.15, top * 0.25), (0.15, top * 0.80)], [(0.55, top * 0.12), (0.55, top * 0.92)],
        [(0.88, top * 0.28), (0.88, top * 0.82)], [(0.15, top * 0.80), (0.55, top * 0.92)],
        [(0.55, top * 0.92), (0.88, top * 0.82)], [(0.66, top * 0.17), (0.66, top * 0.62)],
        [(0.76, top * 0.22), (0.76, top * 0.66)], [(0.66, top * 0.62), (0.76, top * 0.66)],
    ]  # fmt: skip
    dots = []  # each scan point a short dash; one mask, one glow (a glow blurs the canvas)
    for (x0, y0), (x1, y1) in edges:
        for u in rng.random(120):
            x = x0 + u * (x1 - x0) + rng.normal(0, 0.004)
            y = y0 + u * (y1 - y0) + rng.normal(0, 0.004)
            dots.append([(x, y), (x + 0.0015, y)])
    c.glow(c.lines(dots, 2.5, soft=0.8), "#8fb3c9", 0.35, spread=3)
    c.glow(c.disc((0.55, top * 0.12), 0.03, soft=3), "#eaa548", 0.5, spread=22)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: the corner of a room, its floor, walls, and a doorway "
        "drawn only in faint blue points of light, as a laser scan sees it; one corner glows gold.",
    )


# ---------------------------------------------------------------- stages


def fold(
    c: Cloud, xt: np.ndarray, r: int, rng: np.random.Generator, device: str
) -> dict[str, object]:
    """One leave-one-room-out fold: room ``r`` labelled by every method, trained on the others.

    Then the sparser-scan test: ``SPARSE`` of the room's points, their covariance features
    recomputed among themselves, labelled by the same trained models.

    Args:
        c: The thinned scan, with covariance features.
        xt: The trees' inputs for every point (``tree_inputs``).
        r: The held-out room's index into ``craslab.ROOMS``.
        rng: The generator for the trees' draws and the sparser scan.
        device: Where the networks train ("cuda" or "cpu").

    Returns:
        ``preds`` (each method's labels of the room's points), ``sparse`` (confusion counts on
        the sparser scan), ``training`` (each network's seconds and final loss),
        ``tree_rounds``, ``sizes`` (points per column: the room's tiling at full density and
        sparser, and the training columns), ``model`` (PointNet), and ``test_bl`` (the tiling).
    """
    tr, te = c.room != r, c.room == r
    test_bl = blocks(c, te, BLOCK)
    train_bl = blocks(c, tr, BLOCK / 2, MIN_BLOCK)
    preds = {
        "height lookup": height_lookup(c.take(np.flatnonzero(tr)), c.take(np.flatnonzero(te))),
    }
    trees = fit_trees(xt[tr], c.y[tr], rng)
    preds["covariance + trees"] = trees.predict(xt[te])
    models: dict[str, PointNet] = {}
    training = {}
    for name, (glob, n_in) in NETS.items():
        models[name], info = train_pointnet(
            c, train_bl, global_feature=glob, n_in=n_in, device=device
        )
        preds[name] = predict_pointnet(models[name], c, test_bl, device)[te]
        assert (preds[name] >= 0).all()
        training[name] = {"seconds": info["seconds"], "final_loss": info["curve"][-1]}
    thin = np.sort(rng.choice(np.flatnonzero(te), int(SPARSE * te.sum()), replace=False))
    ct = c.take(thin).with_eig()
    ct_bl = blocks(ct, np.ones(len(thin), bool), BLOCK)
    sparse = {"covariance + trees": confusion(ct.y, trees.predict(tree_inputs(ct)))}
    for n in LEARNED[1:]:
        sparse[n] = confusion(ct.y, predict_pointnet(models[n], ct, ct_bl, device))
    return {
        "preds": preds,
        "sparse": sparse,
        "training": training,
        "tree_rounds": int(trees.n_iter_),
        "sizes": {
            "test": [len(b.idx) for b in test_bl],
            "sparse": [len(b.idx) for b in ct_bl],
            "train": [len(b.idx) for b in train_bl],
        },
        "model": models["PointNet"],
        "test_bl": test_bl,
    }


def point_median(sizes: list[int]) -> float:
    """The size of the column the median point sits in (each column weighted by its points).

    Args:
        sizes: Points per column.

    Returns:
        The median over points of their column's size: a float, since the median of an even
        number of points may fall between two sizes.
    """
    s = np.asarray(sizes)
    return float(np.median(np.repeat(s, s)))


def leave_one_room_out(
    c: Cloud, xt: np.ndarray, rng: np.random.Generator, device: str
) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Every fold, pooled: per-class IoU over the five held-out rooms, and per room.

    Args:
        c: The thinned scan, with covariance features.
        xt: The trees' inputs for every point (``tree_inputs``).
        rng: The generator shared by every fold's draws, in room order.
        device: Where the networks train ("cuda" or "cpu").

    Returns:
        The numbers (``methods``, ``sparse``, ``density``), PointNet's and the trees'
        out-of-room labels for every point, and room 110's network and tiling columns (for the
        picture).
    """
    t0 = time.perf_counter()
    pooled = {n: np.zeros((6, 6), np.int64) for n in METHODS}
    sparse = {n: np.zeros((6, 6), np.int64) for n in LEARNED}
    per_room: dict[str, dict[str, float]] = {n: {} for n in METHODS}
    training: dict[str, dict[str, dict]] = {n: {} for n in NETS}
    rounds: dict[str, int] = {}
    sizes: dict[str, list[int]] = {"test": [], "sparse": [], "train": []}
    out_of_room = {n: np.full(len(c.y), -1) for n in ("PointNet", "covariance + trees")}
    keep = {}
    for r, room in enumerate(craslab.ROOMS):
        f = fold(c, xt, r, rng, device)
        te = c.room == r
        for n, pred in out_of_room.items():
            pred[te] = f["preds"][n]
        for n, p in f["preds"].items():
            cm = confusion(c.y[te], p)
            pooled[n] += cm
            per_room[n][room] = scores(cm)["miou"]
        for n, cm in f["sparse"].items():
            sparse[n] += cm
        for n, info in f["training"].items():
            training[n][room] = info
        rounds[room] = f["tree_rounds"]
        for k in sizes:
            sizes[k] += f["sizes"][k]
        if room == "110":
            keep = {"model": f["model"], "test_bl": f["test_bl"], "room": r}
        log(f"  room {room}: " + ", ".join(f"{n} {per_room[n][room]:.2f}" for n in METHODS)
            + f" ({time.perf_counter() - t0:.0f} s)")  # fmt: skip
    methods = {n: scores(pooled[n]) | {"per_room": per_room[n]} for n in METHODS}
    for n in NETS:
        methods[n]["training"] = training[n]
    methods["covariance + trees"]["rounds"] = rounds
    assert all(k < TREE_ROUNDS for k in rounds.values()), rounds  # early stopping ended each
    full, thin = point_median(sizes["test"]), point_median(sizes["sparse"])
    assert thin < NPTS < full  # a set of NPTS samples a full column, and holds a thinned one
    numbers = {
        "methods": methods,
        "sparse": {n: scores(v) for n, v in sparse.items()} | {"keep": SPARSE},
        "density": {  # what each set of NPTS points the network reads covers
            "points_read": NPTS,
            "test_column_median": full,
            "sparse_column_median": thin,
            "train_column_median": float(np.median(sizes["train"])),
            "set_share_full": NPTS / full,  # a set's share of the median point's column
            "set_points_drop": NPTS / thin,  # points per set, full density over a tenth
        },
    }
    return numbers, out_of_room, keep


def random_split_test(
    c: Cloud, xt: np.ndarray, rng: np.random.Generator, rooms_pred: np.ndarray
) -> dict[str, object]:
    """The same features and trees, tested on random 1.5-m columns (one in five) instead.

    Also, like for like: the trees trained with each point's room held out (``rooms_pred``),
    scored on the same test points as the random split.

    Args:
        c: The thinned scan, with covariance features.
        xt: The trees' inputs for every point (``tree_inputs``).
        rng: The generator for the test columns and the trees.
        rooms_pred: The trees' out-of-room labels of every point (leave one room out).

    Returns:
        The random split's scores, its column counts and boosting rounds, and the out-of-room
        trees' scores on the same test points.
    """
    cols = blocks(c, np.ones(len(c.y), bool), BLOCK)
    test_cols = rng.permutation(len(cols))[: len(cols) // 5]
    te = np.zeros(len(c.y), bool)
    for i in test_cols:
        te[cols[i].idx] = True
    trees = fit_trees(xt[~te], c.y[~te], rng)
    assert trees.n_iter_ < TREE_ROUNDS, trees.n_iter_
    assert (rooms_pred[te] >= 0).all()
    return scores(confusion(c.y[te], trees.predict(xt[te]))) | {
        "columns": len(cols),
        "test_columns": len(test_cols),
        "rounds": int(trees.n_iter_),
        "rooms_held_out_same_points": scores(confusion(c.y[te], rooms_pred[te])),
    }


def room_110(c: Cloud, r: int, pn_pred: np.ndarray, out_dir: Path) -> dict[str, object]:
    """PointNet in the room of the picture: confusion counts, recall, commonest mistakes.

    Args:
        c: The thinned scan.
        r: Room 110's index into ``craslab.ROOMS``.
        pn_pred: PointNet's out-of-room labels of every point.
        out_dir: Where the plan view of its labels goes.

    Returns:
        The confusion counts (rows true), each class's recall and commonest wrong label, and
        the share of the plan view's points labelled wrong.
    """
    te = c.room == r
    cm = confusion(c.y[te], pn_pred[te])  # rows true
    off_diag = cm - np.diag(np.diag(cm))
    return {
        "confusion": cm.tolist(),
        "recall": dict(zip(craslab.TARGETS, (np.diag(cm) / cm.sum(1)).tolist(), strict=True)),
        "mistaken_for": {  # each class's commonest wrong label, and its share of that class
            k: [craslab.TARGETS[int(off_diag[i].argmax())], float(off_diag[i].max() / cm[i].sum())]
            for i, k in enumerate(craslab.TARGETS)
        },
        "slice_wrong": plan_labels(c, r, pn_pred, out_dir),
    }


def labelling_stage(c: Cloud, device: str, out_dir: Path) -> tuple[dict, np.ndarray, dict]:
    """Leave one room out (with the sparser-scan test), then the random split.

    The methods: the height lookup, the trees, and PointNet three ways (as published, without
    its global feature, and with each point's covariance features). One generator makes every
    draw, in this order: per fold, the trees' (unused) draw and the sparser scan; then the random
    split's columns.

    Returns:
        The numbers, PointNet's out-of-room labels for every point, and what the figures need
        (room 110's network and columns).
    """
    xt = tree_inputs(c)
    rng = np.random.default_rng(SEED)
    numbers, out_of_room, keep = leave_one_room_out(c, xt, rng, device)
    pn_pred = out_of_room["PointNet"]
    numbers["random_split"] = random_split_test(c, xt, rng, out_of_room["covariance + trees"])
    numbers["parameters"] = {
        n: sum(p.numel() for p in PointNet(n_in=k, global_feature=g).parameters())
        for n, (g, k) in NETS.items()
    }
    iou_figure({n: numbers["methods"][n] for n in ("height lookup", *LEARNED)}, out_dir)
    numbers["room_110"] = room_110(c, keep["room"], pn_pred, out_dir)
    return numbers, pn_pred, keep


def picture_stage(c: Cloud, keep: dict, device: str, out_dir: Path) -> dict[str, object]:
    """The column of the attempt and the picture, with the network's checks on it.

    The column: among room 110's columns whose points span at least ``FULL_SPAN`` in x and in y
    (not cut by the room's edge), the one holding the most classes (each at least 5% of its
    labelled points), the larger if tied.
    """

    def mix(b: Block) -> tuple[int, int]:
        y = c.y[b.idx]
        share = np.bincount(y[y >= 0], minlength=6) / max(1, (y >= 0).sum())
        return int((share >= 0.05).sum()), len(b.idx)

    full = [b for b in keep["test_bl"] if (np.ptp(c.xyz[b.idx, :2], axis=0) >= FULL_SPAN).all()]
    b = max(full, key=mix)
    checks = pointnet_checks(keep["model"], c, b, device)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(b.idx, NPTS, replace=len(b.idx) < NPTS)
    m64 = copy.deepcopy(keep["model"]).double().cpu().eval()  # as the checks: no TF32 near-ties
    with torch.no_grad():
        x = columns(c, [b], [pick], device).double().cpu()
        _, _, arg = m64.pooled(x)
        pred = m64(x)[0].argmax(-1).numpy()
    crit = np.unique(arg[0].numpy())
    assert len(crit) == checks["critical_points"], (len(crit), checks)
    attempt_figure(c, b, pick, out_dir)
    picture(c, b, pick, crit, pred, out_dir=out_dir)
    y = c.y[pick]
    return {
        "checks": checks,
        "column": {
            "centre": b.centre.tolist(),
            "points": len(b.idx),
            "share": {k: float(np.mean(y[y >= 0] == i)) for i, k in enumerate(craslab.TARGETS)},
            "critical": len(crit),
            "accuracy": float(np.mean(pred[y >= 0] == y[y >= 0])),
        },
    }


def registration_stage(c_full: Cloud, pn_pred: np.ndarray, c: Cloud, out_dir: Path) -> dict:
    """Scan to model: the reference registration, its checks, and the break-it tests.

    The tests: the start turned, no coarse start, no cap with and without clutter, the cap on
    the same thinning, and the registration from PointNet's structure points. Each is measured
    as its distance from the reference fit (``motion_error``).
    """
    t0 = time.perf_counter()
    els = craslab.bim_elements()
    bim = bim_cloud(els, BIM_SPACING, SEED)
    tree = cKDTree(bim["xyz"])
    n_bim = len(bim["xyz"])
    log(f"  model meshed: {len(els)} elements, {n_bim:,} points ({time.perf_counter() - t0:.0f} s)")
    out: dict[str, object] = {"elements": len(els), "bim_points": len(bim["xyz"])}
    # The reference: the labelled structure points, 5-cm voxels.
    sel = voxel_first(c_full.xyz, REG_VOXEL)
    ref = c_full.take(sel)
    struct = np.isin(ref.y, STRUCTURE)
    t_init = coarse_start(ref.xyz[struct], bim["xyz"])
    r_ref, t_ref, hist = icp(ref.xyz[struct], tree, bim["xyz"], np.eye(3), t_init)
    out["reference"] = {
        "points": int(struct.sum()),
        "coarse": t_init.tolist(),
        "shift": t_ref.tolist(),
        "turn_deg": Rotation.from_matrix(r_ref).as_euler("xyz", degrees=True).tolist(),
        "history": hist,
    }
    out["checks"] = registration_checks(voxel_first_points(ref.xyz[struct], BASIN_VOXEL))
    log(f"  reference registration and checks ({time.perf_counter() - t0:.0f} s)")
    # PointNet's structure (out-of-room predictions), and all points, with the cap.
    pn_struct = np.isin(pn_pred_on(ref, c, pn_pred), STRUCTURE)
    for name, m in (("PointNet's structure", pn_struct), ("all points", np.ones(len(ref.y), bool))):
        r, t, _ = icp(ref.xyz[m], tree, bim["xyz"], np.eye(3), t_init)
        out[name] = {
            "points": int(m.sum()),
            "error": motion_error(r, t, r_ref, t_ref, ref.xyz[struct]),
        }
    # Start-angle and no-cap tests on 10-cm voxels.
    bas = c_full.take(voxel_first(c_full.xyz, BASIN_VOXEL))
    bstruct = np.isin(bas.y, STRUCTURE)
    p = bas.xyz[bstruct]
    t_b = coarse_start(p, bim["xyz"])
    ctr = (p.min(0) + p.max(0)) / 2
    turns = {}
    for deg in TURNS:
        rz = Rotation.from_euler("z", deg, degrees=True).as_matrix()
        r, t, _ = icp(p, tree, bim["xyz"], rz, t_b + ctr - rz @ ctr)  # turned about the centre
        turns[deg] = motion_error(r, t, r_ref, t_ref, p)
    out["turns"] = turns
    out["basin_points"] = len(p)
    r, t, _ = icp(p, tree, bim["xyz"], np.eye(3), np.zeros(3))
    out["no_coarse_start"] = motion_error(r, t, r_ref, t_ref, p)
    for name, src in (("structure", p), ("all points", bas.xyz)):
        r, t, _ = icp(src, tree, bim["xyz"], np.eye(3), t_b, caps=(np.inf,))
        out[f"no cap, {name}"] = motion_error(r, t, r_ref, t_ref, p)
    # Like for like with the no-cap rows: the cap on the same 10-cm thinning, from the same start
    pn_bas = np.isin(pn_pred_on(bas, c, pn_pred), STRUCTURE)
    for name, m in (("all points", np.ones(len(bas.y), bool)), ("PointNet's structure", pn_bas)):
        r, t, _ = icp(bas.xyz[m], tree, bim["xyz"], np.eye(3), t_b)
        out[f"cap, 10 cm, {name}"] = motion_error(r, t, r_ref, t_ref, p)
    log(f"  start-angle and no-cap tests ({time.perf_counter() - t0:.0f} s)")
    registration_figure(
        ref.xyz[struct],
        ref.feat[struct, 2],
        bim["xyz"],
        [(np.eye(3), np.zeros(3)), (r_ref, t_ref)],
        out_dir,
    )
    basin_figure(turns, out_dir)
    out |= compare_stage(c_full, els, r_ref, t_ref, out_dir)
    return out


def voxel_first_points(p: np.ndarray, size: float) -> np.ndarray:
    """The points of ``p``, one per voxel of side ``size``."""
    return p[voxel_first(p, size)]


def pn_pred_on(target: Cloud, source: Cloud, pred: np.ndarray) -> np.ndarray:
    """PointNet's labels carried to another thinning of the same scan.

    Each point takes the label of the nearest point of ``source``.
    """
    _, idx = cKDTree(source.xyz).query(target.xyz, workers=-1)
    return pred[idx]


def compare_stage(
    c_full: Cloud, els: list[craslab.Element], r: np.ndarray, t: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Scan vs BIM after registration: coverage per element; each wall's verdict from its points.

    Coverage: the share of an element's surface within ``COVER_D`` of any scan point. Each scan
    point's distance to the plane of the nearest model surface; a wall's verdict from its
    labelled and unassigned points (``verdict``). The point fields of ``elements_table`` are
    filled for every element, but only walls and columns are judged: a ceiling's or a slab's
    wall, door, and window points measure nothing about it.

    Args:
        c_full: The whole 2% sample, unthinned, in the scan's frame.
        els: The model's elements, in file order.
        r: The reference registration's rotation (scan to model).
        t: Its shift, m.
        out_dir: Where the figures go.

    Returns:
        The walls' verdicts and their counts, coverage per IFC type, every element's row, and
        the distances' summaries.
    """
    t0 = time.perf_counter()
    scan = c_full.xyz @ r.T + t  # the whole 2% sample, in the model's frame
    scan_tree = cKDTree(scan)
    rng = np.random.default_rng(SEED)
    cov: dict[str, list[float]] = {}
    rows = []
    for e in els:
        p, _ = craslab.sample_surface(e, BIM_SPACING, rng)
        dist, _ = scan_tree.query(p, distance_upper_bound=COVER_D, workers=-1)
        share = float(np.isfinite(dist).mean())
        kind = IFC_NAMES[e.ifc_type]
        cov.setdefault(kind, []).append(share)
        rows.append(
            {"type": kind, "name": e.name, "coverage": share, "area": len(p) * BIM_SPACING**2}
        )
    log(f"  coverage ({time.perf_counter() - t0:.0f} s)")
    fine = bim_cloud(els, DEV_SPACING, SEED + 1)
    near, idx = cKDTree(fine["xyz"]).query(scan, workers=-1)
    dev = np.abs(
        np.einsum("ij,ij->i", scan - fine["xyz"][idx], fine["normal"][idx])
    )  # to the plane
    # Each element's points: the scan points whose nearest model surface (within ASSIGN_D) is on
    # that element. A wall is judged by its labelled points, the wall, door, and window points
    # (the ceiling and floor points next to its top and foot would measure the ceiling and the
    # floor), and checked against its unassigned points (``verdict``).
    vertical = np.isin(c_full.y, VERTICAL)
    unassigned = c_full.y == craslab.IGNORE
    paired = np.where(near < ASSIGN_D, fine["owner"][idx], -1)
    owner = np.where(vertical, paired, -1)
    owner_u = np.where(unassigned, paired, -1)
    for k in range(len(els)):
        for pre, o in (("", owner), ("unassigned_", owner_u)):
            m = o == k
            rows[k][f"{pre}points"] = int(m.sum())
            rows[k][f"{pre}deviation"] = float(np.median(dev[m])) if m.any() else None
    h = c_full.feat[:, 2]
    in_slice = vertical & (h > SLICE[0]) & (h < SLICE[1])
    deviation_figure(scan[in_slice], dev[in_slice], out_dir)
    coverage_figure(cov, out_dir)
    s = np.isin(c_full.y, STRUCTURE)
    wall_k = [k for k, w in enumerate(rows) if w["type"] == "wall"]
    by = {v: [k for k in wall_k if verdict(rows[k]) == v] for v in VERDICTS}
    judged = by["within"] + by["off"] + by["conflicting"]  # 100 or more labelled points
    measure_k = sorted(by["off"] + by["conflicting"], key=lambda k: -rows[k]["deviation"])
    low = [k for k in wall_k if rows[k]["coverage"] < SEEN]  # under 10% within 5 cm of the scan
    cols = [k for k, w in enumerate(rows) if w["type"] == "column"]
    assert all(verdict(rows[k]) == "hidden" for k in cols)
    verdict_figure(els, rows, measure_k, scan[(h > SLICE[0]) & (h < SLICE[1])][::10], out_dir)
    fields = ("name", "deviation", "points", "unassigned_deviation", "unassigned_points",
              "coverage", "area")  # fmt: skip
    return {
        "walls": {
            "n": len(wall_k),
            "min_points": MIN_POINTS,
            **{v: len(by[v]) for v in VERDICTS},
            "seen": len(wall_k) - len(by["hidden"]),
            "judged": len(judged),
            "measure": len(measure_k),
            "measure_list": [
                {"verdict": verdict(rows[k])} | {f: rows[k][f] for f in fields} for k in measure_k
            ],
            "fewest_points_measure": min((rows[k]["points"] for k in measure_k), default=None),
            "judged_median_deviation": (
                float(np.median([rows[k]["deviation"] for k in judged])) if judged else None
            ),
            # Within walls whose unassigned points sit farther: furniture in front, or a surface
            # the labels missed; their labelled points decide
            "within_unassigned_beyond": sum(
                rows[k]["unassigned_points"] >= MIN_POINTS and rows[k]["unassigned_deviation"] > OFF
                for k in by["within"]
            ),
            "by_min_points": {
                n: {v: sum(verdict(rows[k], n) == v for k in wall_k) for v in VERDICTS}
                for n in MIN_POINTS_SWEEP
            },
            # Coverage within 5 cm against the verdict: a wall far off has few points within 5 cm
            "low_cover": len(low),
            "low_cover_judged": sum(k in judged for k in low),
            "low_cover_measure": sum(k in measure_k for k in low),
            "covered_unlabelled": sum(rows[k]["coverage"] >= SEEN for k in by["unlabelled"]),
            "covered_hidden": sum(rows[k]["coverage"] >= SEEN for k in by["hidden"]),
            "column_points": sum(rows[k]["points"] for k in cols),
            "column_unassigned_points": sum(rows[k]["unassigned_points"] for k in cols),
        },
        "coverage": {
            k: {
                "n": len(v),
                "median": float(np.median(v)),
                "below_10pct": int(np.sum(np.asarray(v) < 0.1)),
            }
            for k, v in cov.items()
        },
        "elements_table": rows,
        "deviation": {
            "structure_median": float(np.median(dev[s])),
            "structure_p90": float(np.quantile(dev[s], 0.9)),
            "structure_within_5cm": float(np.mean(dev[s] < 0.05)),
            "walls_median": float(np.median(dev[vertical])),
            "clutter_median": float(np.median(dev[c_full.y == 5])),
        },
    }


def main() -> None:
    """Generate every figure and number for CEE Lecture 22."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before cuBLAS starts
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = craslab.load_sample()
    numbers: dict[str, object] = {"counts": craslab.check_counts()}
    numbers["settings"] = {
        "classes": len(craslab.TARGETS),
        "voxel": VOXEL,
        "block": BLOCK,
        "points": NPTS,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "k_eig": K_EIG,
        "sparse": SPARSE,
        "caps": list(CAPS),
        "cover_d": COVER_D,
        "seen": SEEN,
        "assign_d": ASSIGN_D,
        "min_points": MIN_POINTS,
        "tree_rounds": TREE_ROUNDS,
        "off": OFF,
        "dev_max": DEV_MAX,
        "converged": CONVERGED,
    }
    numbers["rooms_check"] = craslab.check_rooms(d["xyz"], d["label"])
    c_full = prepare(d, None)  # the whole 2% sample, for registration and comparison
    c = prepare(d, VOXEL).with_eig()
    numbers["data"] = {
        "sample": len(d["label"]),
        "voxel_points": len(c.y),
        "rooms": {room: int((c.room == r).sum()) for r, room in enumerate(craslab.ROOMS)},
        "share": {k: float(np.mean(c.y[c.y >= 0] == i)) for i, k in enumerate(craslab.TARGETS)},
        "ignored": float(np.mean(c.y < 0)),
    }
    log(f"data ready in {time.perf_counter() - t0:.0f} s")
    lab, pn_pred, keep = labelling_stage(c, device, out_dir)
    numbers["labelling"] = lab
    log(f"labelling in {time.perf_counter() - t0:.0f} s")
    numbers["picture"] = picture_stage(c, keep, device, out_dir)
    numbers["registration"] = registration_stage(c_full, pn_pred, c, out_dir)
    log(f"registration in {time.perf_counter() - t0:.0f} s")
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_2", "iou", "basin", "deviation"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    log(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
