"""SpaceNet 8, Germany: flooded buildings and roads from before-and-after satellite tiles.

SpaceNet 8 (Hänsch et al., 2022; CC BY-SA 4.0, Maxar imagery under a non-commercial licence),
from ``scripts/fetch_spacenet8.sh``: 202 tiles of the Ahr valley before and after the July 2021
flood, with OpenStreetMap building footprints and road centrelines, each marked flooded or not.

Here: reading and rasterizing the tiles (post-event images resampled onto the pre-event grid),
the spatial blocks, connected blobs, the SpaceNet building match, flood calls per known
footprint, detection boxes, non-maximum suppression from scratch, average precision, and
metres per degree. The networks that read the tiles are in the lecture's figure script.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects, label
from scipy.ndimage import mean as ndimage_mean
from torchvision.ops import box_iou, nms

from common import DATA_DIR, require

SN8 = DATA_DIR / "raw" / "spacenet8" / "Germany_Training_Public"
CLASSES: tuple[str, ...] = ("background", "building", "flooded building", "road", "flooded road")
ROAD_PX: int = 12  # roads are centrelines; drawn 12 px wide (3-5 m), a stated assumption
VAL_X, TEST_X = 35, 40  # tile columns: x < 35 train, 35-39 validation, >= 40 test
SIZE: int = 1300  # pre-event tiles; post-event tiles are resampled onto this grid
PAD: int = 1312  # a multiple of 32, for the five halvings of the ResNet-18 encoder
MIN_PX: int = 25  # blobs smaller than this (about 2.8 m2, smaller than any building) are noise


def blobs(mask: np.ndarray) -> np.ndarray:
    """Connected components of a mask, numbered 1..n, without those under ``MIN_PX``."""
    comps, n = label(mask)
    if n == 0:
        return comps
    size = np.bincount(comps.ravel(), minlength=n + 1)
    keep = size >= MIN_PX
    keep[0] = False
    new = np.zeros(n + 1, dtype=comps.dtype)
    new[keep] = np.arange(1, int(keep.sum()) + 1)
    return new[comps]


def wkt_rings(wkt: str) -> list[np.ndarray]:
    """The coordinate rings of a WKT POLYGON or LINESTRING, each (n, 2) [x, y] in pixels."""
    return [np.array([[float(v) for v in p.split()] for p in s.split(",")])
            for s in re.findall(r"\(([^()]+)\)", wkt)]  # fmt: skip


def geotransform(im: Image.Image) -> tuple[float, float, float, float]:
    """A GeoTIFF's west edge, north edge, and pixel size (x, y), all in degrees.

    Read from its ModelTiepoint (33922) and ModelPixelScale (33550) tags.
    """
    sx, sy, _ = im.tag_v2[33550]
    tie = im.tag_v2[33922]
    assert tie[:3] == (0.0, 0.0, 0.0), tie  # the tie point is the upper-left corner
    return tie[3], tie[4], sx, sy


@dataclass
class Tile:
    """One SpaceNet 8 tile: images on the pre-event grid and its labels."""

    name: str  # e.g. 0_41_58: grid column 41, row 58
    col: int
    pre: np.ndarray  # (1300, 1300, 3) uint8
    post: np.ndarray  # (1300, 1300, 3) uint8, resampled onto the pre-event grid
    lab: np.ndarray  # (1300, 1300) uint8 class index (CLASSES)
    inst: np.ndarray  # (1300, 1300) int32 building index, 0 = none
    buildings: list[dict]  # per building: flooded (bool), area_px
    roads: list[dict]  # per road segment: flooded (bool), length_m, pixel line
    lat: float  # latitude of the tile centre
    deg: tuple[float, float]  # pixel size in degrees of longitude and latitude


def post_on_pre(pre: Image.Image, post: Image.Image) -> Image.Image:
    """Resample the post-event image onto the pre-event grid (bilinear), by their geotransforms.

    Pillow's affine transform evaluates each output pixel at its centre, (c + 0.5, r + 0.5), in
    continuous pixel coordinates, and reads the input in the same convention: so input
    x = (lon0 - plon) / psx + (sx / psx) (c + 0.5), with no half-pixel terms.

    Args:
        pre: The pre-event GeoTIFF (with its tags).
        post: The post-event GeoTIFF (with its tags).

    Returns:
        The post-event image, RGB, on the pre-event grid.
    """
    lon0, lat0, sx, sy = geotransform(pre)
    plon, plat, psx, psy = geotransform(post)  # before convert(), which drops the tags
    coef = (sx / psx, 0, (lon0 - plon) / psx, 0, sy / psy, (plat - lat0) / psy)
    return post.convert("RGB").transform(
        pre.size, Image.Transform.AFFINE, coef, Image.Resampling.BILINEAR
    )


def tiles() -> list[Tile]:
    """Read and rasterize all 202 Germany tiles.

    Post-event images have a different pixel size (and some a different footprint); each is
    resampled onto the pre-event grid (``post_on_pre``). Labels come from the reference CSV's
    pixel geometries (``Wkt_Pix``), which are on the pre-event grid.

    Returns:
        The tiles, in label-file order.
    """
    d = require(SN8, "fetch_spacenet8.sh")
    with (d / "Germany_Training_Public_reference.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (d / "Germany_Training_Public_label_image_mapping.csv").open(newline="") as f:
        maps = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in rows:
        by[r["ImageId"]].append(r)
    out = []
    for m in maps:
        pre = Image.open(d / "PRE-event" / m["pre-event image"])
        post = Image.open(d / "POST-event" / m["post-event image 1"])
        _, lat0, sx, sy = geotransform(pre)
        assert pre.size == (SIZE, SIZE), pre.size
        post = post_on_pre(pre, post)
        name = m["pre-event image"][:-4]
        lab, inst, buildings, roads = rasterize(by[name])
        out.append(Tile(name.split("_", 1)[1], int(name.split("_")[2]),
                        np.asarray(pre.convert("RGB")), np.asarray(post), lab, inst,
                        buildings, roads, lat0 - sy * SIZE / 2, (sx, sy)))  # fmt: skip
    assert len(out) == 202, len(out)
    return out


def rasterize(objs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Draw a tile's roads (12 px lines) and then its buildings (polygons, on top).

    Args:
        objs: The tile's rows of the reference CSV.

    Returns:
        Class raster, building-index raster, building records, road records.
    """
    size = (SIZE, SIZE)
    lab, inst = Image.new("L", size, 0), Image.new("I", size, 0)
    dl, di = ImageDraw.Draw(lab), ImageDraw.Draw(inst)
    buildings, roads = [], []
    for r in objs:
        if r["Object"] != "Road" or "EMPTY" in r["Wkt_Pix"]:
            continue
        assert r["Flooded"] in ("True", "False"), r["Flooded"]
        line = [tuple(p) for p in wkt_rings(r["Wkt_Pix"])[0]]
        flooded = r["Flooded"] == "True"
        dl.line(line, fill=4 if flooded else 3, width=ROAD_PX, joint="curve")
        roads.append({"flooded": flooded, "length_m": float(r["length_m"]), "line": line})
    for r in objs:
        if r["Object"] != "Building" or "EMPTY" in r["Wkt_Pix"]:
            continue
        assert r["Flooded"] in ("True", "False"), r["Flooded"]
        rings = wkt_rings(r["Wkt_Pix"])
        assert len(rings) == 1, "no holes in the Germany footprints"
        flooded = r["Flooded"] == "True"
        poly = [tuple(p) for p in rings[0]]
        dl.polygon(poly, fill=2 if flooded else 1)
        di.polygon(poly, fill=len(buildings) + 1)
        buildings.append({"flooded": flooded})
    inst_a = np.asarray(inst).astype(np.int32)
    areas = np.bincount(inst_a.ravel(), minlength=len(buildings) + 1)
    for k, b in enumerate(buildings, start=1):
        b["area_px"] = int(areas[k])  # later footprints can cover earlier ones
    return np.asarray(lab), inst_a, buildings, roads


def split(all_tiles: list[Tile]) -> dict[str, list[Tile]]:
    """Spatial blocks along the valley: train x < 35, validation 35-39, test x >= 40.

    The blocks touch: a building on a block's edge can be cut into tiles of two blocks.
    """
    return {"train": [t for t in all_tiles if t.col < VAL_X],
            "val": [t for t in all_tiles if VAL_X <= t.col < TEST_X],
            "test": [t for t in all_tiles if t.col >= TEST_X]}  # fmt: skip


def stack(ts: list[Tile]) -> tuple[np.ndarray, np.ndarray]:
    """Before and after as six channels, padded to 1312 px; labels padded with background."""
    pad = PAD - SIZE
    x = np.stack([np.concatenate([t.pre, t.post], axis=2) for t in ts])
    x = np.pad(x, ((0, 0), (0, pad), (0, pad), (0, 0)), mode="reflect")
    y = np.pad(np.stack([t.lab for t in ts]), ((0, 0), (0, pad), (0, pad)))
    assert x.shape[1:] == (PAD, PAD, 6) and y.shape[1:] == (PAD, PAD), x.shape
    return x, y


# ---------------------------------------------------------------------------------------------
# Buildings as objects: matching, flood calls, boxes, NMS


@dataclass
class Match:
    """Predicted blobs matched one to one to true footprints at IoU >= 0.5."""

    hits: int
    true: int
    pred: int
    merged_ids: set[int] = field(default_factory=set)  # true buildings sharing a blob

    @property
    def merged(self) -> int:
        """True buildings that share a blob with another true building."""
        return len(self.merged_ids)


def match(gt: np.ndarray, pred: np.ndarray, iou_min: float = 0.5) -> Match:
    """Match predicted building components to true footprints, one to one, at IoU >= 0.5.

    This is the SpaceNet building metric.

    Args:
        gt: True building-index raster (0 = none).
        pred: Predicted building-index raster (connected components, 0 = none).
        iou_min: Smallest IoU that counts as a hit.

    Returns:
        Hits, counts, and the true buildings that share a component with another (merged).
    """
    ng, npred = int(gt.max()), int(pred.max())
    both = gt.astype(np.int64) * (npred + 1) + pred
    inter = np.bincount(both.ravel(), minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    area_g, area_p = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
    iou[0, :] = iou[:, 0] = 0
    pairs = sorted(((iou[g, p], g, p) for g, p in zip(*np.nonzero(iou >= iou_min), strict=True)),
                   reverse=True)  # fmt: skip
    used_g, used_p = set(), set()
    for _, g, p in pairs:
        if g not in used_g and p not in used_p:
            used_g.add(int(g))
            used_p.add(int(p))
    # A true building is merged if more than half of it lies in a component that also holds
    # more than half of another true building.
    owner = np.where(inter[1:, 1:] > area_g[1:, None] / 2)  # (building, component)
    per_comp = np.bincount(owner[1], minlength=npred) if npred else np.zeros(0, int)
    merged = {int(g) + 1 for g, p in zip(*owner, strict=True) if per_comp[p] > 1}
    return Match(len(used_g), ng, npred, merged)


def evaluate(ts: list[Tile], prob: np.ndarray) -> dict[str, object]:
    """Buildings found (detection), flooded or not (attribution), and flooded road length.

    Args:
        ts: Test tiles.
        prob: Their class probabilities, (N, 5, 1300, 1300).

    Returns:
        Detection counts, building flood calls, road kilometres, pixel F1 by class.
    """
    det = {"hits": 0, "true": 0, "pred": 0, "merged": 0}
    calls = np.zeros((2, 2), int)  # [true flooded, called flooded]
    km = np.zeros((2, 2))
    for t, p in zip(ts, prob, strict=True):
        cls = p.argmax(0)
        m = match(t.inst, blobs((cls == 1) | (cls == 2)))
        for k in det:
            det[k] += getattr(m, k)
        flood = p[2] - p[1]  # > 0 where "flooded building" beats "building"
        sums = np.bincount(t.inst.ravel(), weights=flood.ravel(), minlength=len(t.buildings) + 1)
        for k, b in enumerate(t.buildings, start=1):
            if b["area_px"]:
                calls[int(b["flooded"]), int(sums[k] > 0)] += 1
        for road in t.roads:
            img = Image.new("L", (SIZE, SIZE), 0)
            ImageDraw.Draw(img).line(road["line"], fill=1, width=ROAD_PX, joint="curve")
            on = np.asarray(img, dtype=bool)
            if on.any():
                km[int(road["flooded"]), int((p[4][on] - p[3][on]).sum() > 0)] += road["length_m"]
    y = np.stack([t.lab for t in ts])
    a = prob.argmax(1)
    f1 = [2 * ((a == c) & (y == c)).sum() / max((a == c).sum() + (y == c).sum(), 1)
          for c in range(1, 5)]  # fmt: skip
    return {"detection": det | {"precision": det["hits"] / max(det["pred"], 1),
                                "recall": det["hits"] / max(det["true"], 1)},
            "buildings": [[int(v) for v in r] for r in calls],
            "flood_recall": float(calls[1, 1] / max(calls[1].sum(), 1)),
            "flood_precision": float(calls[1, 1] / max(calls[:, 1].sum(), 1)),
            "road_km": [[float(v / 1000) for v in r] for r in km],
            "pixel_f1": dict(zip(CLASSES[1:], map(float, f1), strict=True))}  # fmt: skip


def building_scores(ts: list[Tile], prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each known footprint's flood score: the mean of P(flooded) - P(dry) over its pixels.

    Footprints come from OpenStreetMap before the event, so the question for each is only
    "flooded or not".

    Args:
        ts: Tiles.
        prob: Their class probabilities.

    Returns:
        Scores and true flooded flags, one per building with pixels.
    """
    score, flooded = [], []
    for t, p in zip(ts, prob, strict=True):
        n = len(t.buildings) + 1
        sums = np.bincount(t.inst.ravel(), weights=(p[2] - p[1]).ravel(), minlength=n)
        for k, b in enumerate(t.buildings, start=1):
            if b["area_px"]:
                score.append(sums[k] / b["area_px"])
                flooded.append(b["flooded"])
    return np.array(score), np.array(flooded)


def flood_target(val: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray],
                 recall: float) -> dict[str, float]:  # fmt: skip
    """Set the flood threshold on validation for a recall target; report it on the test block.

    The threshold is chosen to flag ``recall`` of the validation block's flooded buildings;
    the test block shows what it catches there, and what it costs in buildings flagged.

    Args:
        val: Validation building scores and flags.
        test: Test building scores and flags.
        recall: The target share of flooded buildings to flag.

    Returns:
        Threshold, test recall, precision, the share of all buildings flagged, prevalence.
    """
    s_val, y_val = val
    ranked = np.sort(s_val[y_val])[::-1]
    t = float(ranked[math.ceil(recall * len(ranked)) - 1])  # flag score >= t: recall >= target
    s, y = test
    flag = s >= t
    return {"recall_target": recall, "threshold": t, "test_recall": float(flag[y].mean()),
            "test_precision": float(y[flag].mean()), "flagged_share": float(flag.mean()),
            "prevalence": float(y.mean()), "buildings": len(y)}  # fmt: skip


def boxes_of(comps: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounding boxes (x0, y0, x1, y1) and mean scores of labelled components."""
    n = int(comps.max())
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0)
    sl = find_objects(comps)  # one (rows, cols) slice pair per label; None if absent
    present = [k for k in range(n) if sl[k] is not None]
    boxes = [[sl[k][1].start, sl[k][0].start, sl[k][1].stop, sl[k][0].stop] for k in present]
    s = ndimage_mean(score, comps, [k + 1 for k in present])
    return np.array(boxes, float).reshape(-1, 4), np.asarray(s, float)


def nms_scratch(boxes: np.ndarray, score: np.ndarray, iou_max: float = 0.5) -> list[int]:
    """Non-maximum suppression from scratch.

    Keep the best box, drop every box that overlaps it by more than ``iou_max``, repeat.

    Args:
        boxes: (n, 4) boxes (x0, y0, x1, y1), each with a positive area.
        score: (n,) scores.
        iou_max: Overlap above which a lower-scoring box is suppressed.

    Returns:
        Indices kept, best first.
    """
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
        iou = inter / (area[i] + area[rest] - inter + 1e-8)
        order = rest[iou <= iou_max]  # drop the boxes this one suppresses
    return keep


@dataclass
class TileBoxes:
    """One tile's proposals, the ones NMS kept, and the true footprints' boxes."""

    boxes: np.ndarray  # (n, 4) proposals
    scores: np.ndarray  # (n,)
    kept: list[int]  # indices into boxes, best first
    true: np.ndarray  # (m, 4)


@dataclass
class Detection:
    """Detection numbers for the slides, and every tile's boxes for the figures."""

    numbers: dict[str, object]
    per_tile: dict[str, TileBoxes]


def detection(ts: list[Tile], prob: np.ndarray,
              levels: tuple[float, ...] = (0.3, 0.5, 0.7)) -> Detection:  # fmt: skip
    """Boxes the way a detector makes them, cleaned by NMS and scored by AP at IoU 0.5.

    Proposals come from three thresholds of P(building), so they overlap; non-maximum
    suppression keeps one per object; average precision at box IoU >= 0.5 against the true
    footprints' boxes scores the result.

    Args:
        ts: Test tiles.
        prob: Their class probabilities.
        levels: Thresholds of P(building) + P(flooded building) that make proposals.

    Returns:
        Proposals and kept boxes, the from-scratch NMS check against torchvision, AP, and the
        boxes of every tile.
    """
    rows, n_true, n_prop, same = [], 0, 0, True
    per_tile = {}
    for t, p in zip(ts, prob, strict=True):
        pb = p[1] + p[2]
        props, sc = zip(*[boxes_of(blobs(pb > lv), pb) for lv in levels], strict=True)
        b, s = np.concatenate(props), np.concatenate(sc)
        n_prop += len(b)
        ours = nms_scratch(b, s)
        ref = nms(torch.from_numpy(b), torch.from_numpy(s), 0.5).tolist()
        same &= sorted(map(tuple, b[ours])) == sorted(map(tuple, b[ref]))  # ties: same boxes
        tb, _ = boxes_of(t.inst.astype(np.int64), np.ones((SIZE, SIZE)))
        n_true += len(tb)
        per_tile[t.name] = TileBoxes(b, s, ours, tb)
        if ours and len(tb):
            iou = box_iou(torch.from_numpy(b[ours]), torch.from_numpy(tb)).numpy()
        else:
            iou = np.zeros((len(ours), len(tb)))
        taken = set()
        for j in range(len(ours)):  # best first: a hit claims its best free true box
            free = [g for g in np.argsort(-iou[j]) if g not in taken and iou[j, g] >= 0.5]
            if free:
                taken.add(free[0])
            rows.append((s[ours[j]], bool(free)))
    rows.sort(key=lambda r: -r[0])
    hit = np.array([r[1] for r in rows], float)
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    rec = np.cumsum(hit) / n_true
    # Area under the precision envelope (all-point interpolation, as in VOC 2010+).
    env = np.maximum.accumulate(prec[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * env))
    numbers = {"proposals": n_prop, "kept": len(rows), "true": n_true,
               "nms_matches_torchvision": bool(same), "ap50": ap}  # fmt: skip
    return Detection(numbers, per_tile)


def metres_per_degree(lat: float) -> tuple[float, float]:
    """Metres per degree of longitude and of latitude at latitude ``lat`` (WGS 84 series)."""
    phi = math.radians(lat)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return m_lon, m_lat
