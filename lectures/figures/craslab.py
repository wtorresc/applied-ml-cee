"""The CRAS lab: a labelled indoor laser scan and its as-built BIM model (Lecture 22).

Abreu et al. (2023), *Data* 8(6), 101; Zenodo 7948116 (CC BY 4.0), fetched by
``scripts/fetch_craslab.sh``: four rooms and a hallway of INESC TEC's robotics laboratories at
the University of Porto, scanned in March 2023 from 21 Leica BLK360 G2 stations (584,701,977
points with colour, intensity, and one of 34 labels), and the IFC2X3 model of the same spaces,
drawn in Revit from tape and laser-distance measurements. ``scripts/derive_craslab.py`` streams
the scan once and keeps a 2% sample (``data/derived/craslab_sample.npz``), with per-label
statistics over every point (``craslab_stats.json``).

Labels: the file gives codes only; the names are the paper's Table 4. Its count for code 1
(ceiling) reads 288,024,661; the file holds 228,024,661, and only that value makes the 34 counts
sum to the total of its Table 3 and of the file, so the table has a typo there (``check_counts``).

Rooms: the file does not say which room a point is in, and the IFC model has no spaces. The rooms
here are rectangles between wall faces found in the wall-labelled points (``check_rooms``): the
hallway's faces at x = 7.95 and 10.3 m, room 108 to the left of x = 7.5 m, rooms 109-111 to the
right of x = 10.7 m, split at y = 3.15 and 11.1 m. Between the faces at y = -3.95, 3.15, 11.1,
and 19.95 m, rooms 111, 110, and 109 are 7.10, 7.95, and 8.85 m deep: within 10 cm of the
paper's floor plan (its Figure 2: 7.07, 7.85, and 8.79 m).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from common import DATA_DIR, require

SAMPLE = DATA_DIR / "derived" / "craslab_sample.npz"
STATS = DATA_DIR / "derived" / "craslab_stats.json"
IFC = DATA_DIR / "raw" / "craslab" / "craslabbim.ifc"
N_POINTS = 584_701_977
CLASSES: tuple[str, ...] = (  # code -> name, the paper's Table 4
    "unassigned", "ceiling", "floor", "wall", "door", "window", "desk", "chair", "cabinet",
    "mobile cabinet", "shelf", "vents", "water tank", "bin", "box", "board", "computer", "screen",
    "printer", "vest", "switch", "paper dispenser", "alcohol dispenser", "cable", "phone", "robot",
    "water kettle", "stairs", "ladder", "oil heater", "divider", "hanger", "fan",
    "water dispenser",
)  # fmt: skip
COUNTS: tuple[int, ...] = (  # Table 4, with code 1 corrected from 288,024,661 (module docstring)
    62_412_722, 228_024_661, 73_215_411, 89_783_056, 16_114_181, 12_037_086, 20_302_532,
    6_960_430, 20_399_593, 453_602, 2_834_452, 4_663_309, 4_858_587, 869_702, 18_243_987,
    5_266_606, 341_661, 1_864_965, 1_180_765, 67_128, 45_149, 397_825, 192_923, 281_862, 17_305,
    1_617_877, 17_205, 653_957, 156_425, 45_711, 8_779_277, 1_792_555, 100_189, 709_281,
)  # fmt: skip
PAPER_CEILING = 288_024_661
# The lecture's targets: the building's envelope, and everything else as clutter (furniture,
# equipment, the water tank, the stairs). Code 0 (unassigned) is left out of training and scoring.
TARGETS: tuple[str, ...] = ("ceiling", "floor", "wall", "door", "window", "clutter")
IGNORE = -1
ROOMS: tuple[str, ...] = ("108", "109", "110", "111", "175")  # 175 is the hallway
ROOM_NAMES: dict[str, str] = {
    "108": "water-tank lab",
    "109": "workshop",
    "110": "office",
    "111": "office",
    "175": "hallway",
}
X_SPLITS = (7.5, 10.7)  # m: room 108 | hallway | rooms 109-111
Y_SPLITS = (3.15, 11.1)  # m: room 111 | 110 | 109, right of the hallway
WALL_FACES = {"x": (0.4, 7.1, 7.95, 10.3, 11.1, 17.9), "y_right": (-3.95, 3.15, 11.1, 19.95)}
FLOOR_Z = -1.15  # m: the scan's floor level (the floor label's 1-cm histogram peak)
BIM_TYPES: tuple[str, ...] = (  # the IFC element types meshed for scan-vs-BIM
    "IfcWallStandardCase", "IfcSlab", "IfcCovering", "IfcColumn", "IfcDoor", "IfcWindow",
    "IfcStairFlight",
)  # fmt: skip


def load_sample() -> dict[str, np.ndarray]:
    """The 2% sample of the scan.

    Returns:
        ``xyz`` float32 (n, 3) in m, ``rgb`` uint8 (n, 3), ``intensity`` float32 (n,), and
        ``label`` uint8 (n,), the codes of ``CLASSES``.
    """
    hint = "fetch_craslab.sh; then scripts/derive_craslab.py"
    with np.load(require(SAMPLE, hint)) as d:
        return {k: d[k] for k in ("xyz", "rgb", "intensity", "label")}


def check_counts() -> dict[str, object]:
    """Compare the file's per-label counts with the paper's Table 4.

    Returns:
        The total, and the one code where the paper differs.

    Raises:
        ValueError: If any count other than the ceiling's differs, or the total is wrong.
    """
    stats = json.loads(require(STATS, "derive_craslab.py").read_text())
    have = [stats["classes"][str(c)]["count"] for c in range(len(CLASSES))]
    if have != list(COUNTS) or sum(have) != N_POINTS or stats["points"] != N_POINTS:
        raise ValueError(f"per-label counts differ from Table 4: {have}")
    return {
        "total": N_POINTS,
        "paper_ceiling": PAPER_CEILING,
        "file_ceiling": COUNTS[1],
        "paper_sum": sum(COUNTS) - COUNTS[1] + PAPER_CEILING,
    }


def target_of(label: np.ndarray) -> np.ndarray:
    """Map the 34 codes to ``TARGETS`` (0-5), and code 0 to ``IGNORE``.

    Args:
        label: Codes, shape (n,).

    Returns:
        Target indices, int64, shape (n,).
    """
    lut = np.full(len(CLASSES), TARGETS.index("clutter"), np.int64)
    lut[0] = IGNORE
    for name in TARGETS[:-1]:
        lut[CLASSES.index(name)] = TARGETS.index(name)
    return lut[label]


def room_of(xyz: np.ndarray) -> np.ndarray:
    """The room each point is in, as an index into ``ROOMS``.

    Args:
        xyz: Points in the scan's frame, shape (n, 3).

    Returns:
        Room indices, int64, shape (n,).
    """
    x, y = xyz[:, 0], xyz[:, 1]
    right = np.where(y < Y_SPLITS[0], 3, np.where(y < Y_SPLITS[1], 2, 1))  # 111, 110, 109
    return np.where(x < X_SPLITS[0], 0, np.where(x < X_SPLITS[1], 4, right)).astype(np.int64)


def check_rooms(xyz: np.ndarray, label: np.ndarray, tol: float = 0.15) -> dict[str, list]:
    """Check that the wall-labelled points pile up at the wall faces the rooms are cut between.

    Each expected face must be within ``tol`` of a local maximum of the 10-cm histogram of
    wall points, across x for all walls and across y for the walls right of the hallway.

    Args:
        xyz: Points, shape (n, 3).
        label: Codes, shape (n,).
        tol: Largest allowed distance from a face to a histogram peak, m.

    Returns:
        For each axis, the expected faces and the nearest peaks found.

    Raises:
        ValueError: If a face is farther than ``tol`` from every peak.
    """
    wall = xyz[label == CLASSES.index("wall")]
    out = {}
    for key, values in (
        ("x", wall[:, 0]),
        ("y_right", wall[wall[:, 0] > X_SPLITS[1] + 0.2, 1]),
    ):
        counts, edges = np.histogram(values, np.arange(-6.0, 22.0, 0.1))
        centres = edges[:-1] + 0.05
        top = counts >= np.quantile(counts, 0.9)
        peak = top & (counts >= np.roll(counts, 1)) & (counts >= np.roll(counts, -1))
        found = [float(centres[peak][np.argmin(abs(centres[peak] - f))]) for f in WALL_FACES[key]]
        miss = [f for f, p in zip(WALL_FACES[key], found, strict=True) if abs(f - p) > tol]
        if miss:
            raise ValueError(f"no wall-point peak near the {key} faces {miss}")
        out[key] = [list(WALL_FACES[key]), [round(p, 2) for p in found]]
    return out


@dataclass(frozen=True)
class Element:
    """One BIM element as a triangle mesh in the IFC model's world frame (m)."""

    guid: str
    ifc_type: str
    name: str
    verts: np.ndarray  # (v, 3) float64
    faces: np.ndarray  # (f, 3) int64, indices into verts


def bim_elements() -> list[Element]:
    """Mesh the IFC model's building elements of ``BIM_TYPES`` with IfcOpenShell.

    Returns:
        One ``Element`` per meshed element, in the order of ``BIM_TYPES`` and then of the file
        (the threaded iterator returns them in a different order each run).

    Raises:
        RuntimeError: If IfcOpenShell cannot start the geometry iterator.
    """
    model = ifcopenshell.open(str(require(IFC, "fetch_craslab.sh")))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    include = [e for t in BIM_TYPES for e in model.by_type(t)]
    it = ifcopenshell.geom.iterator(settings, model, os.cpu_count() or 1, include=include)
    if not it.initialize():
        raise RuntimeError("IfcOpenShell could not mesh the model")
    out = []
    while True:
        shape = it.get()
        e = model.by_guid(shape.guid)
        out.append(
            Element(
                guid=shape.guid,
                ifc_type=e.is_a(),
                name=e.Name or "",
                verts=np.asarray(shape.geometry.verts, np.float64).reshape(-1, 3),
                faces=np.asarray(shape.geometry.faces, np.int64).reshape(-1, 3),
            )
        )
        if not it.next():
            break
    order = {e.GlobalId: i for i, e in enumerate(include)}
    assert len(out) == len(order), (len(out), len(order))
    return sorted(out, key=lambda el: order[el.guid])


def sample_surface(
    el: Element, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Points spread uniformly over an element's surface, about one per ``spacing`` squared.

    Args:
        el: The element.
        spacing: Mean spacing, m.
        rng: Random generator.

    Returns:
        Points (k, 3) and the outward unit normal of the triangle each lies on (k, 3).
    """
    tri = el.verts[el.faces]  # (f, 3 corners, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # (f, 3)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    k = max(1, int(np.ceil(area.sum() / spacing**2)))
    pick = rng.choice(len(tri), size=k, p=area / area.sum())
    u, v = rng.random(k), rng.random(k)
    flip = u + v > 1  # reflect into the triangle: uniform over its area
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    normals = cross[pick] / (2 * area[pick, None] + 1e-12)
    return pts, normals
