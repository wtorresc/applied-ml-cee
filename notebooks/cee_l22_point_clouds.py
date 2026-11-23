"""Applied Machine Learning for CEE — Lecture 22: Point Clouds.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l22_point_clouds.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_craslab.sh`` once (Abreu et al., 2023; Zenodo 7948116, CC BY 4.0:
the labelled scan of the CRAS lab, 4.3 GB, and its IFC model), then
``PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_craslab.py`` (about 10 minutes) for
the 2% sample this notebook reads. The IFC model is meshed with IfcOpenShell (LGPL-3.0).

Set ``L22_SETTING=slides``, ``room110``, or ``reduced`` (the default) to choose the setting (see
the header cell).
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
    # Applied Machine Learning for CEE — Lecture 22: Point Clouds

    **Slides:** *Applied Machine Learning for CEE*, Lecture 22 (`lectures/cee_l22_point_clouds`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §4.1.4 and §4.3.1–4.3.3 (the covariance matrix, how mean and covariance move under a rotation and a shift, and diagonalizing a blob's covariance: the covariance features and the alignment), §6.1 (the SVD), §11.2 (robust regression: why far pairs are left out), §12.2.4–12.2.5 and §12.2.8 (gradient boosting), §16.3–16.4 (layers, training, dropout), and §17.1.2 (pooling). Point networks themselves: Qi et al. (2017), Zaheer et al. (2017); registration: Besl & McKay (1992), Kabsch (1976).

    **Data:** Abreu et al. (2023), *Data* 8(6), 101; Zenodo 7948116 (CC BY 4.0): the CRAS lab of INESC TEC at the University of Porto, four rooms and a hallway scanned in March 2023 from 21 Leica BLK360 G2 stations, 584,701,977 points each with colour, intensity, and one of 34 labels, and the as-built IFC model of the same spaces. The model is meshed with IfcOpenShell (LGPL-3.0). Fetch with `bash scripts/fetch_craslab.sh`, then derive the 2% sample with `scripts/derive_craslab.py`; this notebook reads only that sample and the IFC file, never the 4.3 GB archive.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. Three settings,
    chosen in the setup cell or by the environment variable `L22_SETTING`, whatever the device:

    - **`"slides"`**: PointNet three ways in each of five leave-one-room-out folds (15 networks
      of 100 epochs), as the slides' figure script: about 90 minutes on an RTX 3090 Ti alone,
      longer when the GPU is shared (123 minutes on November 20, 2026, when every assert passed).
    - **`"room110"`**: only the fold that holds out room 110 (the room of the picture), 100
      epochs: every method's score on room 110, the picture, and room 110's labels, as on the
      slides. About a fifth of the slides' training: 57 minutes on the RTX 3090 Ti with the
      machine heavily shared (November 20, 2026), when it reproduced every room-110 number of
      the slides.
    - **`"reduced"`** (the default): the same fold with 2 epochs, for a laptop CPU: 16 minutes on
      the CPU of the slides' machine (56 threads, no GPU) and 29 minutes on 8 of its cores (both
      November 20, 2026), with up to 12 GB of memory.

    In every setting, the parts that do not depend on training the networks are checked against
    the slides: the data, the rooms, the height lookup on all five rooms, the covariance
    features, the trees on room 110 and the random split, the derivation's checks, the IFC
    model, the registration and its four tests, coverage, and the walls' verdicts. The networks'
    numbers are checked on the reference GPU (an RTX 3090 Ti, deterministic kernels), where the
    slides were made; on any other device the training may take another path, and they are
    printed beside the slides' values instead.

    **Your tasks** (from the slides): label one column of the scan by eye; derive PointNet's
    max and check that it ignores the points' order; write the covariance features and the SVD
    alignment and check them against SciPy; train PointNet on four rooms and test it on the
    fifth, against a height lookup and gradient-boosted trees; line the scan up with the BIM model
    by ICP; measure each element's coverage and each wall's distance from the model; break it
    four ways (random columns, a sparser scan, a turned start, no cap); then decide which walls
    to measure on site and who labels the next scan.
    """)
    return


@app.cell
def _():
    import copy
    import itertools
    import json
    import os
    import random
    import time
    from dataclasses import dataclass
    from pathlib import Path

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts
    import ifcopenshell
    import ifcopenshell.geom
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull, cKDTree
    from scipy.spatial.transform import Rotation
    from sklearn.ensemble import HistGradientBoostingClassifier
    from torch import nn

    SEED = 22
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    torch.use_deterministic_algorithms(True)  # as the slides' figure script
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU = DEVICE == "cuda"

    # The slides' constants (lectures/figures/cee_l22_point_clouds.py)
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
    # The trees train on every labelled point of the training rooms. They were first trained on
    # a draw of 400,000 of them; that draw is still made (and not used), so the later draws from
    # the same generator (the sparser scans, the random split's columns) stay the same points.
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
    MIN_POINTS = 100  # the fewest points paired with a wall for each test of the verdict (§6)
    MIN_POINTS_SWEEP = (25, 50, 100, 200, 400, 1000)  # the seen rule's threshold, varied
    OFF = 0.05  # m: a judged wall whose labelled points sit a median of more than this is off
    DEV_MAX = 0.10  # m: the deviation map's colour scale
    LABEL_SHIFT = (0.5, 0.5)  # m: a wall's number on the verdict plan, up and right of its centre
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

    # The setting, chosen here or by the environment variable L22_SETTING, whatever the device:
    # - "slides": the five folds, 100 epochs each: every number on the slides;
    # - "room110": only the fold that holds out room 110 (the picture's room), 100 epochs: every
    #   method's score on room 110, the picture, and room 110's labels, as on the slides;
    # - "reduced": the same fold with 2 epochs, for a laptop CPU; the networks' numbers are
    #   printed, not checked.
    # Everything else (the data, the height lookup on all rooms, the covariance features, the
    # random split, the registration and its tests, coverage, the verdicts) runs in every setting.
    L22_SETTING = os.environ.get("L22_SETTING", "reduced").strip().lower()
    if L22_SETTING not in {"slides", "room110", "reduced"}:
        raise ValueError(f"L22_SETTING must be slides, room110, or reduced, not {L22_SETTING!r}")
    ALL_ROOMS = L22_SETTING == "slides"
    PICTURE_ROOM = "110"  # every setting trains this fold: the picture needs its network
    EPOCHS_RUN = 2 if L22_SETTING == "reduced" else EPOCHS
    # The slides' networks were trained on an RTX 3090 Ti with deterministic kernels; there, and
    # only there, the networks' numbers (and the round-off digits of the checks) are asserted.
    # Elsewhere the training may take another path: they are printed beside the slides' values.
    REFERENCE = GPU and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090 Ti"
    CHECK_TRAINED = L22_SETTING != "reduced" and REFERENCE

    # Tolerances. Counts, the thinning, the rooms, the height lookup, the covariance features,
    # the SVD alignment, the IFC meshes and their seeded samples, and ICP (on the CPU) are exact,
    # up to the order of floating-point sums.
    EXACT = 1e-9
    # Identities that hold in exact arithmetic, checked in float64: round-off only.
    ROUNDOFF = 1e-12
    # The trees sum gradients on several threads (OpenMP), so another machine may break a near
    # tie differently: they are checked to half a unit of the slide's last digit.
    TREE_TOL = 0.005
    # The networks, on the reference GPU (``REFERENCE``), where this notebook repeats the slides'
    # training: checked to half a unit of the slide's last digit and, digit by digit, against the
    # slides.
    NET_TOL = 0.005

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    MUTED = "#595959"
    # Ceiling blue, floor brown, wall navy, door orange, window green, clutter pink: each at
    # least 3:1 against white (5.2, 6.4, 11.6, 3.9, 3.4, 3.1)
    CLASS_COLORS = ("#0072B2", "#8C510A", "#1B3A5C", "#D55E00", "#009E73", "#CC79A7")

    def shown(x: float, text: str) -> bool:
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("0.61", "8e-15").

        Args:
            x: The unrounded value.
            text: The slide's digits, fixed-point or in e-notation.

        Returns:
            True if ``x`` rounds to ``text``.
        """
        if "e" in text:
            mantissa = text.split("e", maxsplit=1)[0]
            digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
            return f"{x:.{digits}e}" == text
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    def find_root() -> Path:
        """Locate the repository from its root, ``notebooks/``, or ``ipynb/``.

        Returns:
            The directory that holds ``data/derived/craslab_sample.npz``.

        Raises:
            FileNotFoundError: If the sample is missing (run ``bash scripts/fetch_craslab.sh``,
                then ``scripts/derive_craslab.py``).
        """
        rel = Path("data") / "derived" / "craslab_sample.npz"
        bases = [Path.cwd(), Path.cwd().parent]
        if "__file__" in globals():
            bases.insert(0, Path(__file__).resolve().parents[1])
        for base in bases:
            if (base / rel).exists():
                return base
        raise FileNotFoundError(
            f"{rel} not found; run `bash scripts/fetch_craslab.sh`, then "
            "`PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_craslab.py`"
        )

    ROOT = find_root()
    print(f"device: {DEVICE}; setting {L22_SETTING!r}; networks checked: {CHECK_TRAINED}")
    return (
        ALL_ROOMS,
        ASSIGN_D,
        BASIN_VOXEL,
        BATCH,
        BIM_SPACING,
        BLOCK,
        CAPS,
        CHECK_TRAINED,
        CLASS_COLORS,
        CONVERGED,
        COVER_D,
        ConvexHull,
        DEVICE,
        DEV_MAX,
        DEV_SPACING,
        EPOCHS,
        EPOCHS_RUN,
        EXACT,
        FULL_SPAN,
        HEIGHT_BIN,
        HistGradientBoostingClassifier,
        ICP_ITERS,
        IFC_NAMES,
        KNOWN_SHIFT,
        KNOWN_TURN,
        K_EIG,
        L22_SETTING,
        LABEL_GAP,
        LABEL_SHIFT,
        LR,
        MIN_BLOCK,
        MIN_POINTS,
        MIN_POINTS_SWEEP,
        MUTED,
        NET_TOL,
        NPTS,
        OFF,
        PALETTE,
        PICTURE_ROOM,
        Polygon,
        REFERENCE,
        REG_VOXEL,
        ROOT,
        ROUNDOFF,
        Rotation,
        SEED,
        SEEN,
        SLICE,
        SPARSE,
        STRUCTURE,
        TREE_DRAW,
        TREE_ROUNDS,
        TREE_TOL,
        TURNS,
        VERTICAL,
        VOXEL,
        WEIGHT_DECAY,
        cKDTree,
        copy,
        dataclass,
        ifcopenshell,
        itertools,
        json,
        nn,
        np,
        os,
        plt,
        shown,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 5–21

    On your own, no notes, two minutes:

    1. Lecture 5: which factorization solves least squares without forming the inverse?
    2. Lecture 9: how does RANSAC keep 30% bad points from moving a fitted line?
    3. Lecture 21: a graph layer sums its neighbours' messages. Why a sum, not a list?

    **Today:** a laser scan of a lab against its building model: label the points, line them
    up, and find what differs.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. QR or the SVD (Lecture 5). Today the SVD finds the rotation that best lines up two
           point sets (§3).
        2. It fits to the largest group that agrees within a tolerance and ignores the rest.
           Today registration ignores pairs farther apart than a cap, for the same reason:
           clutter (§5).
        3. A sum does not depend on the neighbours' order; a list does (checked below). A scan's
           points have no order at all; today a maximum plays the sum's part.
        """)
        }
    )
    return


@app.cell
def _(SEED, np):
    # Answer 1: least squares from the SVD, without the inverse, against NumPy's solver
    _rng = np.random.default_rng(SEED)
    _a, _b = _rng.normal(size=(20, 3)), _rng.normal(size=20)
    _u, _s, _vt = np.linalg.svd(_a, full_matrices=False)
    _x_svd = _vt.T @ ((_u.T @ _b) / _s)
    assert np.allclose(_x_svd, np.linalg.lstsq(_a, _b, rcond=None)[0], rtol=0, atol=1e-12)
    # Answer 3: five neighbours' messages in another order: the same sum and the same maximum,
    # but not the same list
    _m = _rng.normal(size=(5, 3))
    _perm = _rng.permutation(5)
    assert np.allclose(_m.sum(0), _m[_perm].sum(0), rtol=0, atol=1e-12)
    assert np.array_equal(_m.max(0), _m[_perm].max(0))
    assert not np.array_equal(_m.ravel(), _m[_perm].ravel())
    print(f"permutation {_perm}: sum and max unchanged, the concatenated list changed")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A facility manager plans a renovation of the CRAS lab from its BIM model. **Decide:** does
    the model match what was built, element by element? A wall drawn in the wrong place costs a
    redesign or a clash on site; a wall the scan never saw cannot be confirmed at all.

    The pipeline: **label** each point of a laser scan (part of the building, or clutter), **line
    up** the scan with the model, and **measure** where they differ. The scan (Abreu et al.,
    2023) labels every point with one of 34 codes; we group them into six classes: ceiling,
    floor, wall, door, window, and clutter (furniture, equipment, the water tank, the stairs).
    Code 0, "unassigned", is left out of training and scoring. The file has no room per point
    and the IFC model has no spaces, so the rooms are cut between wall faces found in the
    wall-labelled points (checked below).
    """)
    return


@app.cell
def _(ROOT, json, np):
    SAMPLE = ROOT / "data" / "derived" / "craslab_sample.npz"
    STATS = ROOT / "data" / "derived" / "craslab_stats.json"
    IFC = ROOT / "data" / "raw" / "craslab" / "craslabbim.ifc"
    N_POINTS = 584_701_977
    CLASSES = (  # code -> name, the paper's Table 4
        "unassigned", "ceiling", "floor", "wall", "door", "window", "desk", "chair", "cabinet",
        "mobile cabinet", "shelf", "vents", "water tank", "bin", "box", "board", "computer",
        "screen", "printer", "vest", "switch", "paper dispenser", "alcohol dispenser", "cable",
        "phone", "robot", "water kettle", "stairs", "ladder", "oil heater", "divider", "hanger",
        "fan", "water dispenser",
    )  # fmt: skip
    COUNTS = (  # Table 4, with code 1 (ceiling) corrected from 288,024,661 (checked below)
        62_412_722, 228_024_661, 73_215_411, 89_783_056, 16_114_181, 12_037_086, 20_302_532,
        6_960_430, 20_399_593, 453_602, 2_834_452, 4_663_309, 4_858_587, 869_702, 18_243_987,
        5_266_606, 341_661, 1_864_965, 1_180_765, 67_128, 45_149, 397_825, 192_923, 281_862,
        17_305, 1_617_877, 17_205, 653_957, 156_425, 45_711, 8_779_277, 1_792_555, 100_189,
        709_281,
    )  # fmt: skip
    PAPER_CEILING = 288_024_661
    TABLE3 = {  # the paper's Table 3: scans and points per space
        "108": (6, 183_787_505),
        "109": (3, 89_906_532),
        "110": (5, 132_629_460),
        "111": (3, 89_001_431),
        "175": (4, 89_377_049),
    }
    # Today's classes: the building's envelope, and everything else as clutter
    TARGETS = ("ceiling", "floor", "wall", "door", "window", "clutter")
    IGNORE = -1
    ROOMS = ("108", "109", "110", "111", "175")  # 175 is the hallway
    ROOM_NAMES = {
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
    BIM_TYPES = (  # the IFC element types meshed for scan-vs-BIM
        "IfcWallStandardCase", "IfcSlab", "IfcCovering", "IfcColumn", "IfcDoor", "IfcWindow",
        "IfcStairFlight",
    )  # fmt: skip

    def load_sample() -> dict[str, np.ndarray]:
        """The 2% sample of the scan (``scripts/derive_craslab.py``).

        Returns:
            ``xyz`` float32 (n, 3) in m, ``rgb`` uint8 (n, 3), ``intensity`` float32 (n,), and
            ``label`` uint8 (n,), the codes of ``CLASSES``.
        """
        with np.load(SAMPLE) as d:
            return {k: d[k] for k in ("xyz", "rgb", "intensity", "label")}

    def check_counts() -> dict[str, int]:
        """Compare the file's per-label counts (over every point) with the paper's Table 4.

        Returns:
            The total, the paper's and the file's ceiling counts, and what the paper's counts
            would sum to.

        Raises:
            ValueError: If any count other than the ceiling's differs, or the total is wrong.
        """
        stats = json.loads(STATS.read_text())
        have = [stats["classes"][str(k)]["count"] for k in range(len(CLASSES))]
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
        """The room each point is in, as an index into ``ROOMS``: rectangles between wall faces.

        Args:
            xyz: Points in the scan's frame, shape (n, 3).

        Returns:
            Room indices, int64, shape (n,).
        """
        x, y = xyz[:, 0], xyz[:, 1]
        right = np.where(y < Y_SPLITS[0], 3, np.where(y < Y_SPLITS[1], 2, 1))  # 111, 110, 109
        return np.where(x < X_SPLITS[0], 0, np.where(x < X_SPLITS[1], 4, right)).astype(np.int64)

    def check_rooms(xyz: np.ndarray, label: np.ndarray, tol: float = 0.15) -> dict[str, list]:
        """Check that the wall-labelled points pile up at the faces the rooms are cut between.

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
            found = [
                float(centres[peak][np.argmin(abs(centres[peak] - f))]) for f in WALL_FACES[key]
            ]
            miss = [f for f, p in zip(WALL_FACES[key], found, strict=True) if abs(f - p) > tol]
            if miss:
                raise ValueError(f"no wall-point peak near the {key} faces {miss}")
            out[key] = [list(WALL_FACES[key]), [round(p, 2) for p in found]]
        return out

    return (
        BIM_TYPES,
        CLASSES,
        FLOOR_Z,
        IFC,
        IGNORE,
        N_POINTS,
        ROOMS,
        ROOM_NAMES,
        TABLE3,
        TARGETS,
        check_counts,
        check_rooms,
        load_sample,
        room_of,
        target_of,
    )


@app.cell
def _(BLOCK, FLOOR_Z, K_EIG, TARGETS, cKDTree, dataclass, np, room_of, target_of):
    @dataclass(frozen=True)
    class Cloud:
        """The thinned scan.

        Attributes:
            xyz: Coordinates in the scan's frame, m, (n, 3) float64.
            feat: Network inputs (n, 7) float32: x, y, height above the floor, r, g, b,
                intensity (colours and intensity in [-0.5, 0.5]).
            y: Target index into ``TARGETS``, or ``IGNORE``, (n,).
            room: Index into ``ROOMS``, (n,).
            eig: The six covariance features of ``eigen_features`` (n, 6), or None.
        """

        xyz: np.ndarray
        feat: np.ndarray
        y: np.ndarray
        room: np.ndarray
        eig: np.ndarray | None = None

        def take(self, idx: np.ndarray) -> "Cloud":
            """The points ``idx`` (their covariance features are not recomputed)."""
            eig = None if self.eig is None else self.eig[idx]
            return Cloud(self.xyz[idx], self.feat[idx], self.y[idx], self.room[idx], eig)

        def with_eig(self) -> "Cloud":
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
                xyz[:, 2] - FLOOR_Z,
                d["rgb"][keep] / 255.0 - 0.5,
                d["intensity"][keep] - 0.5,
            ]
        ).astype(np.float32)
        assert feat.shape == (len(keep), 7), feat.shape
        return Cloud(xyz, feat, target_of(d["label"][keep]), room_of(xyz))

    def eigen_features(xyz: np.ndarray, k: int = K_EIG) -> np.ndarray:
        """Covariance features of each point's ``k`` nearest neighbours (Weinmann et al., 2015).

        With the eigenvalues l1 >= l2 >= l3 of the neighbourhood's covariance and n the
        eigenvector of l3 (the local normal): linearity (l1 - l2)/l1, planarity (l2 - l3)/l1,
        scattering l3/l1, verticality 1 - |n_z|, size sqrt(l1), and the neighbourhood's height
        range.

        Returns:
            Features, shape (n, 6).
        """
        _, nb = cKDTree(xyz).query(xyz, k=k, workers=-1)  # (n, k)
        out = np.empty((len(xyz), 6))
        step = 200_000
        for s in range(0, len(xyz), step):
            p = xyz[nb[s : s + step]]  # (b, k, 3)
            c = p - p.mean(1, keepdims=True)
            w, v = np.linalg.eigh(np.einsum("bki,bkj->bij", c, c) / k)  # ascending
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

    def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Confusion counts over the labelled points, rows true, columns predicted."""
        k = len(TARGETS)
        m = y >= 0
        return np.bincount(y[m] * k + pred[m], minlength=k * k).reshape(k, k)

    def scores(cm: np.ndarray) -> dict[str, object]:
        """Per-class IoU (Lecture 16), their mean, and accuracy, from confusion counts."""
        inter = np.diag(cm)
        iou = inter / np.maximum(cm.sum(0) + cm.sum(1) - inter, 1)
        return {
            "iou": dict(zip(TARGETS, iou.tolist(), strict=True)),
            "miou": float(iou.mean()),
            "accuracy": float(inter.sum() / cm.sum()),
        }

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

    return (
        blocks,
        confusion,
        eigen_features,
        prepare,
        scores,
        tree_inputs,
        voxel_first,
    )


@app.cell
def _(
    EXACT,
    N_POINTS,
    ROOMS,
    TABLE3,
    TARGETS,
    VOXEL,
    check_counts,
    check_rooms,
    load_sample,
    np,
    prepare,
    shown,
    time,
):
    _t0 = time.perf_counter()
    raw = load_sample()
    counts = check_counts()
    rooms_check = check_rooms(raw["xyz"], raw["label"])
    c_full = prepare(raw, None)  # every point of the 2% sample: registration and comparison
    c = prepare(raw, VOXEL).with_eig()  # one point per 3-cm voxel: labelling
    data_stats = {
        "sample": len(raw["label"]),
        "voxel_points": len(c.y),
        "rooms": {room: int((c.room == r).sum()) for r, room in enumerate(ROOMS)},
        "share": {k: float(np.mean(c.y[c.y >= 0] == i)) for i, k in enumerate(TARGETS)},
        "ignored": float(np.mean(c.y < 0)),
    }
    print(f"data ready in {time.perf_counter() - _t0:.0f} s: {data_stats}")
    # The paper's label table: its ceiling count makes the labels sum to 644,701,977; the file's
    # count (228,024,661) sums to the stated total, 584,701,977
    assert counts == {"total": 584_701_977, "paper_ceiling": 288_024_661,
                      "file_ceiling": 228_024_661, "paper_sum": 644_701_977}  # fmt: skip
    # The rooms: every wall face they are cut between sits on a peak of the wall points
    assert rooms_check == {
        "x": [[0.4, 7.1, 7.95, 10.3, 11.1, 17.9], [0.45, 7.05, 8.05, 10.35, 11.05, 17.85]],
        "y_right": [[-3.95, 3.15, 11.1, 19.95], [-3.95, 3.15, 11.15, 19.95]],
    }
    # The notes: a 2% sample, 1,577,190 points after 3-cm thinning, 17% of them unassigned
    assert data_stats["sample"] == 11_688_653 and data_stats["voxel_points"] == 1_577_190
    assert data_stats["rooms"] == {"108": 570_387, "109": 281_749, "110": 259_354,
                                   "111": 223_204, "175": 242_496}  # fmt: skip
    _share = [0.3168140092948397, 0.17013587995959562, 0.19273316694139914,
              0.02822351287526942, 0.03597226665363628, 0.2561211642752598]  # fmt: skip
    assert np.allclose(list(data_stats["share"].values()), _share, rtol=0, atol=EXACT)
    assert abs(data_stats["ignored"] - 0.16956168882633038) < EXACT
    assert shown(100 * data_stats["ignored"], "17")
    assert shown(100 * data_stats["sample"] / N_POINTS, "2")
    # The paper's Table 3: 21 scans whose points add up to the file's total; and the share of
    # the 2% sample in each of our rooms against each space's share in the table
    assert sum(_s for _s, _ in TABLE3.values()) == 21
    assert sum(_n for _, _n in TABLE3.values()) == N_POINTS
    table3_check = {
        _room: (
            float(np.mean(c_full.room == _r)),
            TABLE3[_room][1] / N_POINTS,
        )
        for _r, _room in enumerate(ROOMS)
    }
    print("share of points per room, the 2% sample against Table 3:",
          {_k: f"{_a:.3f} vs {_b:.3f}" for _k, (_a, _b) in table3_check.items()})  # fmt: skip
    # Our rectangles hold each space's share of the points to within 2 points of a percent
    # (the hallway gets the most extra: 17.0% of the sample against 15.3% in the table)
    assert max(abs(_a - _b) for _a, _b in table3_check.values()) < 0.02
    return c, c_full, counts, data_stats, raw, table3_check


@app.cell
def _(N_POINTS, TABLE3, counts, data_stats, mo):
    mo.md(f"""
    | CRAS lab, University of Porto (Abreu et al., 2023) | Value |
    |:---------------------------------------------------|------:|
    | Laser scans (Leica BLK360 G2, March 2023) | {sum(_s for _s, _ in TABLE3.values())} |
    | Points, each with colour, intensity, and a label | {counts["total"]:,} |
    | Rooms: a water-tank lab, a workshop, two offices, a hallway | {len(TABLE3)} |
    | Labels in the file → today's classes | 34 → 6 |
    | Elements of the as-built BIM model meshed (IFC; §4) | 135 |

    We use a 2% random sample ({data_stats["sample"]:,} points, {100 * data_stats["sample"] / N_POINTS:.0f}%),
    thinned to one point per 3-cm cube for labelling: {data_stats["voxel_points"]:,} points,
    {100 * data_stats["ignored"]:.0f}% of them unassigned. **A data check:** the paper's label
    table lists {counts["paper_ceiling"]:,} ceiling points, which would make the 34 labels sum to
    {counts["paper_sum"]:,}; the file has {counts["file_ceiling"]:,}, which sums to the stated
    total. A typo, caught by adding up.
    """)
    return


@app.cell
def _(
    BLOCK,
    EXACT,
    FULL_SPAN,
    NPTS,
    ROOMS,
    SEED,
    TARGETS,
    blocks,
    c,
    np,
    shown,
):
    # The attempt's column: of room 110's tiling columns that span at least 1.2 m in x and in y
    # (not cut by the room's edge), the one holding the most classes (each at least 5% of its
    # labelled points), the larger if tied. The same column carries the picture (§2, §4).
    test_bl_110 = blocks(c, c.room == ROOMS.index("110"), BLOCK)

    def mix(b) -> tuple[int, int]:
        """How many classes hold at least 5% of a column's labelled points, and its size."""
        y = c.y[b.idx]
        share = np.bincount(y[y >= 0], minlength=6) / max(1, (y >= 0).sum())
        return int((share >= 0.05).sum()), len(b.idx)

    _full = [_b for _b in test_bl_110 if (np.ptp(c.xyz[_b.idx, :2], axis=0) >= FULL_SPAN).all()]
    column = max(_full, key=mix)
    pick = np.random.default_rng(SEED).choice(column.idx, NPTS, replace=len(column.idx) < NPTS)
    _y = c.y[pick]
    column_share = {_k: float(np.mean(_y[_y >= 0] == _i)) for _i, _k in enumerate(TARGETS)}
    print(f"column at {column.centre}, {len(column.idx):,} points; shares {column_share}")
    # The slide: a 1.5-m column, 14,816 points (4,096 drawn); 55% clutter, 18% ceiling,
    # 14% wall, 10% floor, 3% window, no door
    assert np.allclose(column.centre, [12.950499534606934, 3.9005000591278076], rtol=0, atol=1e-6)
    assert len(column.idx) == 14_816 and len(pick) == 4096 and len(np.unique(pick)) == 4096
    _ref = [0.17995385798513203, 0.095872853114586, 0.14175852345552423, 0.0,
            0.033068443988720844, 0.5493463214560369]  # fmt: skip
    assert np.allclose(list(column_share.values()), _ref, rtol=0, atol=EXACT)
    assert all(shown(100 * column_share[_k], _t) for _k, _t in
               (("clutter", "55"), ("ceiling", "18"), ("wall", "14"), ("floor", "10"),
                ("window", "3")))  # fmt: skip
    return column, column_share, mix, pick, test_bl_110


@app.cell
def _(FLOOR_Z, c, column, mo, np, pick, plt):
    # The attempt: the column from above and from the side, every point grey
    _p = c.xyz[pick] - np.r_[column.centre, FLOOR_Z]  # x, y from the centre; z from the floor
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    _a1.scatter(_p[:, 0], _p[:, 1], s=1, c="#555555", linewidths=0)
    _a1.set(xlabel="x, m", ylabel="y, m", title="from above", aspect="equal")
    _a2.scatter(_p[:, 0], _p[:, 2], s=1, c="#555555", linewidths=0)
    _a2.set(xlabel="x, m", ylabel="height, m", title="from the side", aspect="equal")
    mo.vstack(
        [
            mo.md(f"""
    ### Your turn: wall, window, or clutter?

    One 1.5-m column of room 110: {len(column.idx):,} points, of which {len(pick):,} are drawn
    here (the number the network reads), every point grey. Two minutes, alone: mark the walls,
    the window, and the furniture. Then write down what you used to decide.
    """),
            _fig,
        ]
    )
    return


@app.cell
def _(column_share, mo):
    mo.accordion(
        {
            "The column's labels": mo.md(f"""
        {100 * column_share["clutter"]:.0f}% clutter, {100 * column_share["ceiling"]:.0f}% ceiling,
        {100 * column_share["wall"]:.0f}% wall, {100 * column_share["floor"]:.0f}% floor,
        {100 * column_share["window"]:.0f}% window, no door. The cues you used (height, flat and
        vertical, straight edges, what sits on what) are features; today a height lookup, the
        covariance features with trees, and PointNet learn them in three different ways.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: a scan is a set; a max reads it

    A point cloud is a **set**: listing its points in another order changes nothing. There is
    no grid (Lecture 15) and no edges (Lecture 21). What kind of function gives the same answer
    for every order of its inputs? A sum (Lecture 21's messages) or a **maximum**. PointNet
    lifts every point by the same small network to 1,024 numbers and keeps, for each number,
    the largest over the column. Each of the 1,024 maxima is set by one point; the points that
    set at least one are the **critical points**. Delete every other point and the 1,024 numbers
    do not change.

    Below: the column of the attempt as a set in 3-D, then the idea in NumPy with a random lift
    (no training): the maxima ignore the order, and the critical points alone give them back.
    The trained network's critical points and labels follow in §4.
    """)
    return


@app.cell
def _(BLOCK, CLASS_COLORS, FLOOR_Z, TARGETS, c, column, np, pick, plt):
    def column_3d(layers: list, legend: bool = False):
        """The column in 3-D; ``layers`` are (indices into ``pick``, colour, size) triples."""
        p = c.xyz[pick] - np.r_[column.centre, FLOOR_Z]
        fig = plt.figure(figsize=(9, 3.6), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        for sel, color, size in layers:
            q = p[sel]
            ax.scatter(*q.T, c=color, s=size, alpha=0.6, linewidths=0, depthshade=False)
        ax.view_init(elev=22, azim=-60)
        ax.set_box_aspect((1, 1, np.ptp(p[:, 2]) / BLOCK))
        ax.set(xlabel="x, m", ylabel="y, m", zlabel="height, m")
        if legend:
            for k, name in enumerate(TARGETS):
                ax.scatter([], [], [], c=CLASS_COLORS[k], s=30, label=name)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
        return fig

    column_3d([(np.arange(len(pick)), "#7a7a7a", 1.0)])
    return (column_3d,)


@app.cell
def _(NPTS, ROUNDOFF, SEED, c, column, np, pick):
    # The idea in NumPy: one random lift for every point (7 -> 64 -> 1,024, ReLU), then the
    # maximum of each of the 1,024 numbers over the points. No training.
    _x = c.feat[pick].astype(np.float64)
    _x[:, :2] -= column.centre  # x and y relative to the column's centre, as the network sees
    _rng = np.random.default_rng(SEED)
    _w1, _b1 = _rng.normal(0, 1 / np.sqrt(7), (7, 64)), _rng.normal(0, 0.1, 64)
    _w2, _b2 = _rng.normal(0, 1 / np.sqrt(64), (64, 1024)), _rng.normal(0, 0.1, 1024)

    def _lift(x: np.ndarray) -> np.ndarray:
        return np.maximum(np.maximum(x @ _w1 + _b1, 0) @ _w2 + _b2, 0)  # (n, 1024)

    _h = _lift(_x)
    assert _h.shape == (NPTS, 1024)
    _g, _arg = _h.max(0), _h.argmax(0)  # each number's maximum, and the point that sets it
    _crit = np.unique(_arg)
    _perm = _rng.permutation(NPTS)
    _g_perm = _lift(_x[_perm]).max(0)
    _g_crit = _lift(_x[_crit]).max(0)
    _sum_drop = np.abs(_lift(_x[_crit]).sum(0) - _h.sum(0)).max()
    random_lift = {
        "critical": len(_crit),
        "shuffled": float(np.abs(_g - _g_perm).max()),
        "critical_only": float(np.abs(_g - _g_crit).max()),
        "sum_change": float(_sum_drop),
    }
    print(f"random lift: {random_lift}")
    # The maxima ignore the order and need only the critical points (to round-off: the products
    # run in another order on another number of rows); a sum would need every point
    assert random_lift["shuffled"] < ROUNDOFF and random_lift["critical_only"] < ROUNDOFF
    assert random_lift["critical"] < NPTS and random_lift["sum_change"] > 1.0
    return (random_lift,)


@app.cell
def _(mo, random_lift):
    mo.md(f"""
    With random weights, {random_lift["critical"]} of the 4,096 points set the 1,024 maxima.
    Shuffling the points moves no maximum (largest change {random_lift["shuffled"]:.0g}), and
    the critical points alone give every maximum back ({random_lift["critical_only"]:.0g}); a
    sum over the critical points alone is off by up to {random_lift["sum_change"]:.0f}. Both
    properties come from the max, not from training.

    | Data | What is shared | Order-free by | Lecture |
    |:-----|:---------------|:--------------|:--------|
    | Pixels on a grid | the 3 × 3 weights | (the grid fixes the order) | 15, 16 |
    | Nodes and edges | message and update rules | a sum over neighbours | 21 |
    | Points in space | one network for every point | a max over the points | today |

    Each structure gets the network that respects it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive: lift each point, take the max

    **Assumption:** what a point is depends on its own coordinates, colour, and intensity, and on
    what its column contains, not on the order in which the scanner wrote the points down.

    A function of a set must give the same answer for every order of its points: it must be
    **symmetric**. PointNet (Qi et al., 2017) builds one from a network $h$ applied to each point
    alone, a maximum taken feature by feature, and another network $\gamma$:

    $$f(\{x_1, \dots, x_n\}) = \gamma\Big(\max_{i=1..n} h(x_i)\Big).$$

    Qi et al.'s Theorem 1: any continuous symmetric function of a set can be approximated this
    way, given enough features; Deep Sets (Zaheer et al., 2017) prove the same with a sum. For a
    label per point, each point keeps its own local feature $\ell(x_i)$ as well:

    $$\text{global} = \max_{i}\, h(x_i), \qquad \text{label}_i = g\big(\ell(x_i),\ \text{global}\big).$$

    - **Why the outputs permute with the points.** Shuffle the points by $\pi$: $h$ sees each point
      alone, so the maxima are the same numbers, and point $\pi(i)$'s label is computed from
      $\ell(x_{\pi(i)})$ and the same global feature. The labels are shuffled the same way, and
      nothing else changes.
    - **Why the critical points suffice.** For each of the 1,024 numbers, let $c_k$ be the point
      where it peaks. A subset $C$ that contains every $c_k$ has the same maxima, so the global
      feature, and every other point's label, rests on $C$.
    - **What PointNet cannot see.** Each point is lifted alone; it meets the others only through
      the column's 1,024 maxima. No point knows its neighbours: whether it sits on a flat patch or
      in a scattered one. PointNet++ (Qi et al., 2017b) builds neighbourhoods into the network;
      today we test the other route, giving each point its neighbourhood's covariance features.

    | Part | Size |
    |:-----|:-----|
    | Each point $x_i$ | 7 numbers: x, y in the column, height, colour (3), intensity |
    | $\ell$, then $h$: the same MLP for every point | 7 → 64 (local), → 128 → 1,024 |
    | $g$: label from local and global | 64 + 1,024 → 512 → 256 → 128 → 6 |
    | Training | 4 rooms, columns of 4,096 points, turned at random |

    Each layer is a 1 × 1 convolution over the points (the same weights for every point), batch
    normalization, and a ReLU; dropout 0.3 before the last layer. We leave out PointNet's small
    alignment networks. x and y are relative to the column's centre (where the column sits in
    the building does not matter); a turn changes x and y, so the network is not built to ignore
    turns: it is trained on turned columns.
    """)
    return


@app.cell
def _(itertools, nn, torch):
    class PointNet(nn.Module):
        """PointNet for segmentation (Qi et al., 2017), without the alignment networks.

        Each point is lifted by the same small MLP (a 1 x 1 convolution over the points) to 64
        numbers (its local feature), then to 1,024; the maximum of each of the 1,024 over the
        column's points is the global feature. Each point's label comes from its local feature and
        the global one. Without ``global_feature``, each point is labelled from its own 64 numbers
        alone. With ``n_in = 13``, each point also brings its six covariance features (its
        neighbourhood's shape), which plain PointNet does not see.
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
            """Local features (B, 64, N), global feature (B, 1024), and argmax (B, 1024).

            Args:
                x: Points, (B, N, n_in).

            Returns:
                For each global number, also the index of the point holding the maximum.
            """
            loc = self.local(x.transpose(1, 2))  # (B, n_in, N) -> (B, 64, N)
            g, arg = self.lift(loc).max(dim=2)  # (B, 1024, N) -> (B, 1024)
            return loc, g, arg

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Class scores per point: (B, N, n_in) -> (B, N, 6)."""
            if self.global_feature:
                loc, g, _ = self.pooled(x)
                h = torch.cat([loc, g[:, :, None].expand(-1, -1, loc.shape[2])], dim=1)
            else:
                h = self.local(x.transpose(1, 2))
            return self.head(h).transpose(1, 2)

    NETS = {  # name: (global feature, inputs per point)
        "PointNet, no global feature": (False, 7),
        "PointNet": (True, 7),
        "PointNet + covariance": (True, 13),
    }
    # A forward pass on dummy points, and the weights the notes quote (872,006)
    _m = PointNet().eval()
    with torch.no_grad():
        _loc, _g, _arg = _m.pooled(torch.zeros(2, 100, 7))
        _out = _m(torch.zeros(2, 100, 7))
    assert _loc.shape == (2, 64, 100) and _g.shape == _arg.shape == (2, 1024)
    assert _out.shape == (2, 100, 6)
    net_params = {
        _n: sum(_p.numel() for _p in PointNet(n_in=_k, global_feature=_gl).parameters())
        for _n, (_gl, _k) in NETS.items()
    }
    print(f"weights: {net_params}")
    assert net_params == {"PointNet, no global feature": 347_718, "PointNet": 872_006,
                          "PointNet + covariance": 872_390}  # fmt: skip
    return NETS, PointNet, net_params


@app.cell
def _(mo):
    mo.md(r"""
    ### Two baselines: height, and the shape of each neighbourhood

    **The height lookup.** Cut the height above the floor into 5-cm bands; each band gets the
    class most common in it in the training rooms. Ceilings and floors sit at known heights;
    walls, doors, and windows share theirs.

    **The covariance features** (Weinmann et al., 2015). For each point, take its $k = 24$
    nearest neighbours and their covariance (Forsyth §4.1.4):

    $$C = \frac{1}{k}\sum_{j=1}^{k} (p_j - \bar p)(p_j - \bar p)^\top .$$

    Diagonalizing $C$ rotates the neighbourhood's blob onto its own axes (Forsyth §4.3.3); the
    eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ are its spreads along them, and the
    eigenvector of $\lambda_3$ is the direction it is thinnest in: on a surface, the normal $n$.
    A patch of wall is a flat blob ($\lambda_3 \approx 0$, $n$ horizontal), an edge is a long
    one ($\lambda_1 \gg \lambda_2$), and a chair's legs and a desk's clutter are scattered
    ($\lambda_3 / \lambda_1$ large). Six features per point: linearity
    $(\lambda_1 - \lambda_2)/\lambda_1$, planarity $(\lambda_2 - \lambda_3)/\lambda_1$,
    scattering $\lambda_3/\lambda_1$, verticality $1 - |n_z|$, size $\sqrt{\lambda_1}$, and the
    neighbourhood's height range. Gradient-boosted trees (Lecture 8) read them with height,
    colour, and intensity; PointNet + covariance reads them as six more inputs per point.

    Below: the vectorized features checked against a direct computation for a few points, and
    worked on the attempt's column.
    """)
    return


@app.cell
def _(K_EIG, SEED, TARGETS, c, cKDTree, mo, np, pick):
    # The vectorized features against np.cov and eigh, one point at a time, for 20 points
    _rng = np.random.default_rng(SEED)
    _some = _rng.choice(len(c.y), 20, replace=False)
    _, _nb = cKDTree(c.xyz).query(c.xyz[_some], k=K_EIG, workers=-1)
    for _i, _j in enumerate(_some):
        _w, _v = np.linalg.eigh(np.cov(c.xyz[_nb[_i]].T, bias=True))
        _l3, _l2, _l1 = _w
        _direct = [(_l1 - _l2) / _l1, (_l2 - _l3) / _l1, _l3 / _l1, 1 - abs(_v[2, 0]),
                   np.sqrt(_l1), np.ptp(c.xyz[_nb[_i], 2])]  # fmt: skip
        assert np.allclose(c.eig[_j], _direct, rtol=0, atol=1e-8), (_j, c.eig[_j], _direct)
    # Worked on the attempt's column: the medians by class
    _names = ("linearity", "planarity", "scattering", "verticality", "size, m", "height range, m")
    _rows = []
    for _k in (0, 1, 2, 4, 5):  # the classes in the column (no door)
        _m = c.y[pick] == _k
        _med = np.median(c.eig[pick][_m], axis=0)
        _rows.append(
            f"| {TARGETS[_k]} | {_m.sum()} | " + " | ".join(f"{_x:.2f}" for _x in _med) + " |"
        )
    col_eig = {
        TARGETS[_k]: np.median(c.eig[pick][c.y[pick] == _k], axis=0) for _k in (0, 1, 2, 4, 5)
    }
    # Walls are vertical and flat; the ceiling and floor flat and level; clutter the least flat
    assert col_eig["wall"][3] > 0.9 and col_eig["ceiling"][3] < 0.1 and col_eig["floor"][3] < 0.1
    assert col_eig["clutter"][1] < min(col_eig[_k][1] for _k in ("ceiling", "floor", "wall"))
    mo.md(
        "The column's points, median covariance features by class (20 points also checked "
        "against `np.cov` and `eigh`, to 1e-8):\n\n"
        "| Class | Points | "
        + " | ".join(_names)
        + " |\n|:--|--:|"
        + "--:|" * 6
        + "\n"
        + "\n".join(_rows)
    )
    return (col_eig,)


@app.cell
def _(HEIGHT_BIN, HistGradientBoostingClassifier, SEED, TARGETS, TREE_DRAW, TREE_ROUNDS, np):
    def height_lookup(train, test) -> np.ndarray:
        """Each 5-cm height band's commonest class in training, applied to the test points."""

        def band(c) -> np.ndarray:
            return np.clip((c.feat[:, 2] / HEIGHT_BIN).astype(int), 0, 79)

        m = train.y >= 0
        table = np.zeros((80, len(TARGETS)), np.int64)
        np.add.at(table, (band(train)[m], train.y[m]), 1)
        return table.argmax(1)[band(test)]

    def fit_trees(
        x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> HistGradientBoostingClassifier:
        """Gradient-boosted trees on every labelled point of the training set.

        Every defining parameter is stated, although each equals scikit-learn's default (with
        this many points, its "auto" early stopping is on): early stopping holds out 10% of the
        points (stratified, drawn with ``SEED``) and stops after 10 rounds without a lower loss
        on them.

        Args:
            x: Inputs, (n, 15).
            y: Targets, (n,), -1 for unlabelled points.
            rng: The labelling's generator; it makes the old ``TREE_DRAW`` draw (unused).

        Returns:
            The fitted trees.
        """
        lab = np.flatnonzero(y >= 0)
        rng.choice(len(lab), min(TREE_DRAW, len(lab)), replace=False)  # keeps the later draws
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

    return fit_trees, height_lookup


@app.cell
def _(mo):
    mo.md(r"""
    ### Line the scan up with the model: the SVD, then ICP

    The scan and the model live in two frames: the scanner's and the model's. Given pairs of
    points $(p_i, q_i)$, find the rotation $R$ ($R^\top R = I$, $\det R = +1$) and shift $t$ that
    minimize

    $$E(R, t) = \sum_i \lVert R p_i + t - q_i \rVert^2 .$$

    1. **The shift.** $\partial E / \partial t = 0$ gives $t = \bar q - R \bar p$: the means must
       land on each other (Forsyth §4.3.1: the mean moves with the points).
    2. **Centre.** With $\tilde p_i = p_i - \bar p$ and $\tilde q_i = q_i - \bar q$,
       $E = \sum_i \lVert \tilde p_i \rVert^2 + \lVert \tilde q_i \rVert^2 - 2\,\tilde q_i^\top R \tilde p_i$,
       since a rotation keeps lengths (Forsyth's Useful Facts 4.4). Minimizing $E$ is maximizing
       $\sum_i \tilde q_i^\top R \tilde p_i = \operatorname{tr}(R H)$, with the cross-covariance
       $H = \sum_i \tilde p_i \tilde q_i^\top$.
    3. **The SVD** (Lecture 5; Forsyth §6.1). With $H = U \Sigma V^\top$,
       $\operatorname{tr}(R U \Sigma V^\top) = \operatorname{tr}(\Sigma M)$ for the orthogonal
       $M = V^\top R U$, whose diagonal entries are at most 1. So the trace is at most
       $\sum_k \sigma_k$, reached at $M = I$: $R = V U^\top$.
    4. **No mirror.** If $\det(V U^\top) = -1$ that is a reflection; the best rotation flips the
       axis of the smallest singular value: $R = V\,\mathrm{diag}(1, 1, d)\,U^\top$,
       $d = \operatorname{sign}\det(V U^\top)$ (Kabsch, 1976).

    **ICP** (Besl & McKay, 1992) does not know the pairs. It pairs each moved scan point with
    its nearest model point (a k-d tree), solves for $R, t$ as above, moves the scan, and repeats
    until the motion stops changing (a shift under 1 µm and a turn under $10^{-7}$ rad). Without a
    cap, each step lowers the mean squared distance (the SVD lowers it for the current pairs,
    and re-pairing to the nearest points lowers it again), so ICP converges, to a **local**
    minimum. Pairs farther apart than a cap are ignored, as RANSAC ignores outliers (Lecture 9)
    and a trimmed fit ignores large residuals (Forsyth §11.2); the cap shrinks from 1 m to 10 cm.
    """)
    return


@app.cell
def _(CAPS, ICP_ITERS, KNOWN_SHIFT, KNOWN_TURN, Rotation, SEED, cKDTree, np):
    def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rotation and shift that best move points ``p`` onto ``q`` (least squares, SVD).

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
        tree,
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

    # A reflection is never returned: mirrored partners give the nearest proper rotation
    _p = np.random.default_rng(SEED).normal(size=(50, 3))
    _r, _ = kabsch(_p, _p * np.array([1.0, 1.0, -1.0]))
    assert abs(np.linalg.det(_r) - 1) < 1e-12 and np.allclose(_r.T @ _r, np.eye(3), atol=1e-12)
    return icp, kabsch, motion_error, registration_checks


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    ```python
    out_p = model(x[perm])          # the same points, shuffled
    crit = unique(argmax over points of h(x))
    g_crit = max over crit of h(x)  # only the critical points
    R = kabsch(p, R0 @ p + t0)      # SVD; against SciPy
    R, t = icp(scan, R0 @ scan + t0) # recover a known motion
    ```

    How far is `out_p` from `out[perm]`? `g_crit` from `g`? `R` from `R0`? Two minutes,
    written. The known motion: turned (1°, −2°, 10°) about x, y, z and shifted (0.5, −0.3,
    0.1) m. The answers come in §4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### Leave one room out

    Each of the five rooms is labelled by models trained on the other four: what a new building
    asks of them. The network reads columns of 1.5 m: in training, columns every 0.75 m (they
    overlap; columns of fewer than 200 points are skipped), with 4,096 of each column's points
    drawn afresh every epoch, each column turned by a random angle about the vertical and
    mirrored half the time (walls run every way in a new building). In testing the columns tile
    the room, so each point is labelled once; a column's points go through in sets of 4,096, the
    last set filled with repeats of its own points. Training: AdamW (rate 0.001, weight decay
    0.01), batches of 16 columns, a cosine schedule to zero over 100 epochs, cross-entropy over
    the labelled points; the final network is used (no early stopping). The trees learn from
    every labelled point of the four training rooms, as the networks do, and stop early on a
    held-out tenth of them.

    PointNet's settings were fixed before these runs, on one earlier run that held out room 110
    and validated on room 111 (the figure script's record). Both rooms are test rooms here, so
    their scores reuse the rooms that chose the settings: a small reuse, stated rather than
    hidden.

    The scores: intersection over union per class (Lecture 16), pooled over the five held-out
    rooms (the confusion counts summed, then the IoU), and their mean.
    """)
    return


@app.cell
def _(
    BATCH,
    LR,
    NPTS,
    PointNet,
    SEED,
    TARGETS,
    WEIGHT_DECAY,
    copy,
    nn,
    np,
    time,
    torch,
):
    def columns(c, bl: list, picks: list[np.ndarray], device: str, n_in: int = 7) -> torch.Tensor:
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
        c, train_blocks: list, *, global_feature: bool, n_in: int, device: str, epochs: int
    ) -> tuple:
        """Train from scratch: AdamW, cosine schedule over ``epochs``, random points each epoch.

        Returns:
            The network and its record: the mean training loss per epoch, seconds, weights.
        """
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = PointNet(n_in=n_in, global_feature=global_feature).to(device)
        opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        curve = []
        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            order = rng.permutation(len(train_blocks))
            total, n = 0.0, 0
            for s in range(0, len(order), BATCH):
                bl = [train_blocks[i] for i in order[s : s + BATCH]]
                picks = [rng.choice(b.idx, NPTS, replace=len(b.idx) < NPTS) for b in bl]
                x = augment(columns(c, bl, picks, device, n_in), rng)
                y = torch.from_numpy(np.stack([c.y[p] for p in picks])).to(device)
                loss = nn.functional.cross_entropy(
                    model(x).reshape(-1, len(TARGETS)), y.reshape(-1), ignore_index=-1
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

    def predict_pointnet(model, c, bl: list, device: str, seed: int = SEED) -> np.ndarray:
        """Label every point of the tiling columns ``bl``.

        Each column's points, shuffled, go through in sets of ``NPTS``, the last set filled up
        with repeats of its own column's points.

        Returns:
            Predicted target per point of ``c`` (-1 where no column covers it).
        """
        model.eval()
        rng = np.random.default_rng(seed)
        pred = np.full(len(c.y), -1)
        with torch.no_grad():
            for b in bl:
                perm = rng.permutation(b.idx)
                n = -(-len(perm) // NPTS) * NPTS
                pad = np.concatenate([perm, rng.choice(perm, n - len(perm))])
                sets = pad.reshape(-1, NPTS)
                x = columns(c, [b] * len(sets), list(sets), device, model.n_in)
                out = model(x).argmax(-1).cpu().numpy()
                pred[sets.ravel()] = out.ravel()
        return pred

    def pointnet_checks(model, c, b, device: str) -> dict[str, object]:
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

    return columns, pointnet_checks, predict_pointnet, train_pointnet


@app.cell
def _(
    ALL_ROOMS,
    BLOCK,
    DEVICE,
    EPOCHS_RUN,
    MIN_BLOCK,
    NETS,
    NPTS,
    PICTURE_ROOM,
    ROOMS,
    SEED,
    SPARSE,
    TREE_DRAW,
    TREE_ROUNDS,
    blocks,
    c,
    confusion,
    fit_trees,
    height_lookup,
    np,
    predict_pointnet,
    scores,
    time,
    train_pointnet,
    tree_inputs,
):
    # The long cell: the folds; in each, the height lookup, the trees, and PointNet three ways
    # (the slides' setting: about 90 minutes on an RTX 3090 Ti, longer when the GPU is shared).
    # The other settings fit only room 110's models but make every other fold's random draws,
    # so room 110 and the random split see the slides' draws.
    _t0 = time.perf_counter()
    xt = tree_inputs(c)
    method_names = ("height lookup", "covariance + trees", *NETS)
    learned = ("covariance + trees", "PointNet", "PointNet + covariance")
    _pooled = {_n: np.zeros((6, 6), np.int64) for _n in method_names}
    _sparse = {_n: np.zeros((6, 6), np.int64) for _n in learned}
    per_room = {_n: {} for _n in method_names}  # mean IoU per held-out room
    train_info = {_n: {} for _n in NETS}  # per room: loss curve, seconds, weights
    tree_iters = {}
    _sizes = {"test": [], "sparse": [], "train": []}  # points per column
    pn_pred = np.full(len(c.y), -1)  # PointNet's out-of-room labels, every point
    tree_pred = np.full(len(c.y), -1)  # the trees' out-of-room labels, every point
    sparse_thin = {}
    rooms_run = []
    keep = {}
    _rng = np.random.default_rng(SEED)
    for _r, _room in enumerate(ROOMS):
        _tr, _te = c.room != _r, c.room == _r
        _preds = {
            "height lookup": height_lookup(c.take(np.flatnonzero(_tr)), c.take(np.flatnonzero(_te)))
        }
        _run = ALL_ROOMS or _room == PICTURE_ROOM
        if _run:
            _test_bl = blocks(c, _te, BLOCK)
            _train_bl = blocks(c, _tr, BLOCK / 2, MIN_BLOCK)
            _trees = fit_trees(xt[_tr], c.y[_tr], _rng)
            tree_iters[_room] = _trees.n_iter_
            _preds["covariance + trees"] = _trees.predict(xt[_te])
            tree_pred[_te] = _preds["covariance + trees"]
            _models = {}
            for _name, (_glob, _n_in) in NETS.items():
                _models[_name], _info = train_pointnet(
                    c, _train_bl, global_feature=_glob, n_in=_n_in, device=DEVICE, epochs=EPOCHS_RUN
                )
                _preds[_name] = predict_pointnet(_models[_name], c, _test_bl, DEVICE)[_te]
                assert (_preds[_name] >= 0).all()
                train_info[_name][_room] = _info
                print(f"  room {_room}, {_name}: final loss {_info['curve'][-1]:.4f}, "
                      f"{_info['seconds']:.0f} s", flush=True)  # fmt: skip
            pn_pred[_te] = _preds["PointNet"]
        else:  # the trees' unused draw, as fit_trees makes it, with nothing fitted
            _lab = np.flatnonzero(c.y[_tr] >= 0)
            _rng.choice(len(_lab), min(TREE_DRAW, len(_lab)), replace=False)
        for _n, _p in _preds.items():
            _cm = confusion(c.y[_te], _p)
            _pooled[_n] += _cm
            per_room[_n][_room] = scores(_cm)["miou"]
        # A sparser scan of the held-out room: SPARSE of its points, features recomputed
        _thin = np.sort(_rng.choice(np.flatnonzero(_te), int(SPARSE * _te.sum()), replace=False))
        if _run:
            sparse_thin[_room] = _thin
            _ct = c.take(_thin).with_eig()
            _sparse["covariance + trees"] += confusion(_ct.y, _trees.predict(tree_inputs(_ct)))
            _ct_bl = blocks(_ct, np.ones(len(_thin), bool), BLOCK)
            for _n in learned[1:]:
                _sparse[_n] += confusion(_ct.y, predict_pointnet(_models[_n], _ct, _ct_bl, DEVICE))
            _sizes["test"] += [len(_b.idx) for _b in _test_bl]
            _sizes["sparse"] += [len(_b.idx) for _b in _ct_bl]
            _sizes["train"] += [len(_b.idx) for _b in _train_bl]
            rooms_run.append(_room)
        if _room == PICTURE_ROOM:
            keep = {"model": _models["PointNet"], "test_bl": _test_bl, "room": _r}
        print(f"room {_room}: " + ", ".join(f"{_n} {_v[_room]:.3f}" for _n, _v in per_room.items()
              if _room in _v) + f" ({time.perf_counter() - _t0:.0f} s)", flush=True)  # fmt: skip
    results = {
        _n: scores(_pooled[_n]) | {"per_room": per_room[_n]}
        for _n in method_names
        if _pooled[_n].sum()
    }
    sparse_res = {_n: scores(_v) for _n, _v in _sparse.items() if _v.sum()}

    def _point_median(sizes: list[int]) -> float:
        """The size of the column the median point sits in (columns weighted by their points)."""
        _s = np.asarray(sizes)
        return float(np.median(np.repeat(_s, _s)))

    density = {  # what each set of NPTS points the network reads covers
        "points_read": NPTS,
        "test_column_median": _point_median(_sizes["test"]),
        "sparse_column_median": _point_median(_sizes["sparse"]),
        "train_column_median": float(np.median(_sizes["train"])),
    }
    # Random split: the same covariance features and trees, tested on random 1.5-m columns
    _cols = blocks(c, np.ones(len(c.y), bool), BLOCK)
    _test_cols = _rng.permutation(len(_cols))[: len(_cols) // 5]
    rand_te = np.zeros(len(c.y), bool)
    for _i in _test_cols:
        rand_te[_cols[_i].idx] = True
    _trees_rs = fit_trees(xt[~rand_te], c.y[~rand_te], _rng)
    tree_iters["random split"] = _trees_rs.n_iter_
    rand_pred = np.full(len(c.y), -1)
    rand_pred[rand_te] = _trees_rs.predict(xt[rand_te])
    random_split = scores(confusion(c.y[rand_te], rand_pred[rand_te])) | {
        "columns": len(_cols),
        "test_columns": len(_test_cols),
    }
    labelling_minutes = (time.perf_counter() - _t0) / 60
    print(f"labelling in {labelling_minutes:.1f} min; trees' rounds {tree_iters}; {density}")
    # Early stopping ended every fit of the trees before the rounds allowed
    assert all(_k < TREE_ROUNDS for _k in tree_iters.values()), tree_iters
    return (
        density,
        keep,
        labelling_minutes,
        learned,
        method_names,
        pn_pred,
        rand_pred,
        rand_te,
        random_split,
        results,
        rooms_run,
        sparse_res,
        sparse_thin,
        train_info,
        tree_iters,
        tree_pred,
        xt,
    )


@app.cell
def _(
    ALL_ROOMS,
    CHECK_TRAINED,
    EXACT,
    NET_TOL,
    PICTURE_ROOM,
    ROOMS,
    TREE_TOL,
    np,
    random_split,
    results,
    shown,
):
    _hl, _tr = results["height lookup"], results["covariance + trees"]
    for _n, _v in results.items():
        print(f"{_n:28s} mean IoU {_v['miou']:.4f}, accuracy {_v['accuracy']:.4f}; "
              + ", ".join(f"{_k} {_x:.3f}" for _k, _x in _v["iou"].items()))  # fmt: skip
    # The height lookup needs no training: exact, all five rooms, in every setting
    assert abs(_hl["miou"] - 0.4323447469238028) < EXACT
    assert abs(_hl["accuracy"] - 0.7412493443450283) < EXACT
    assert np.allclose(list(_hl["iou"].values()), [0.8144330124915956, 0.9411006872431287,
                       0.3042524120165843, 0.0, 0.0, 0.5342823697915082], rtol=0, atol=EXACT)  # fmt: skip
    assert np.allclose([_hl["per_room"][_r] for _r in ROOMS], [0.4415326567594458,
                       0.47417074363487927, 0.45853654657408294, 0.46744635693279907,
                       0.2688463617417466], rtol=0, atol=EXACT)  # fmt: skip
    assert shown(_hl["miou"], "0.43") and _hl["iou"]["door"] == _hl["iou"]["window"] == 0
    assert min(_hl["per_room"], key=_hl["per_room"].get) == "175"
    # numbers.json: pooled mean IoU, accuracy, IoU per class, mean IoU per room (108 to 175)
    _ref = {
        'covariance + trees': (0.6080456277744547, 0.8485416019282936,
            [0.9176636856494138, 0.9618111434909312, 0.5830060264443978, 0.3755349993096783, 0.1363509894619692, 0.673906922290338],
            [0.582454661381287, 0.6872523419147337, 0.7140233395778249, 0.731206794217058, 0.476906847266447]),
        'PointNet, no global feature': (0.5176017826312752, 0.7927236995508334,
            [0.8865930701732793, 0.9578222734738093, 0.4388984123154271, 0.11761221144922099, 0.11677079717690617, 0.5879139311990085],
            [0.504440773276592, 0.5614980067259117, 0.5986523830169136, 0.6078870979727756, 0.37634699222587237]),
        'PointNet': (0.5626743286952688, 0.8116348122059096,
            [0.883460951196064, 0.9464944729084026, 0.486783684924889, 0.2905488831518689, 0.1330942704597976, 0.635663709530591],
            [0.5382889212272337, 0.6272666048439451, 0.697413643288059, 0.7633684715523982, 0.40070607461852276]),
        'PointNet + covariance': (0.6265421007156154, 0.8536791883086888,
            [0.9189533330560216, 0.9641336692327274, 0.5905193475579962, 0.46794954662135957, 0.12847281550080306, 0.6892238923247845],
            [0.5760662687595149, 0.7084815509811239, 0.7748481168471298, 0.8168775769011636, 0.5060419663302244]),
    }  # fmt: skip
    # Room 110, in every setting: the trees (any machine, to TREE_TOL); the networks on the
    # reference GPU, and beside the slides' values elsewhere
    _i110 = ROOMS.index(PICTURE_ROOM)
    for _n, (_m, _a, _iou, _rooms) in _ref.items():
        _got = results[_n]["per_room"][PICTURE_ROOM]
        print(f"room 110, {_n}: {_got:.4f} (slides {_rooms[_i110]:.4f})")
        if _n == "covariance + trees":
            assert abs(_got - _rooms[_i110]) < TREE_TOL, (_n, _got)
        elif CHECK_TRAINED:
            assert abs(_got - _rooms[_i110]) < NET_TOL, (_n, _got)
    # The random split, in every setting (its draws are the slides' in every setting)
    assert random_split["columns"] == 206 and random_split["test_columns"] == 41
    assert abs(random_split["miou"] - 0.7057964903666584) < TREE_TOL
    assert abs(random_split["accuracy"] - 0.9103414116398373) < TREE_TOL
    assert shown(random_split["miou"], "0.71") and shown(100 * random_split["accuracy"], "91")
    # All five rooms (the slides' setting): the trees, and the networks on the reference GPU
    _checked = (["covariance + trees"] if ALL_ROOMS else []) + [
        _n for _n in _ref if _n != "covariance + trees" and CHECK_TRAINED and ALL_ROOMS
    ]
    for _n in _checked:
        _tol = TREE_TOL if _n == "covariance + trees" else NET_TOL
        _m, _a, _iou, _rooms = _ref[_n]
        _v = results[_n]
        assert abs(_v["miou"] - _m) < _tol and abs(_v["accuracy"] - _a) < _tol, (_n, _v)
        assert np.allclose(list(_v["iou"].values()), _iou, rtol=0, atol=2 * _tol), _n
        assert np.allclose([_v["per_room"][_r] for _r in ROOMS], _rooms, rtol=0, atol=2 * _tol), _n
        # Windows defeat every method: each method's lowest IoU; the hallway is every
        # method's hardest room
        assert min(_v["iou"], key=_v["iou"].get) == "window", _n
        assert min(_v["per_room"], key=_v["per_room"].get) == "175", _n
    if ALL_ROOMS:
        # The slide: trees 0.61 (85% of points right), windows 0.14, room 108 0.58
        assert shown(_tr["miou"], "0.61") and shown(100 * _tr["accuracy"], "85")
        assert shown(_tr["iou"]["window"], "0.14") and shown(_tr["per_room"]["108"], "0.58")
    if CHECK_TRAINED and ALL_ROOMS:
        _pn, _pc = results["PointNet"], results["PointNet + covariance"]
        _ng = results["PointNet, no global feature"]
        # The legend and notes: PointNet 0.56, + covariance 0.63, without the global feature 0.52
        assert shown(_pn["miou"], "0.56") and shown(_pc["miou"], "0.63")
        assert shown(_ng["miou"], "0.52")
        # + covariance against the trees (one seed each): 0.627 against 0.608, close; behind only
        # in room 108, 0.576 against 0.582
        assert [_r for _r in ROOMS if _pc["per_room"][_r] < _tr["per_room"][_r]] == ["108"]
        assert shown(_pc["per_room"]["108"], "0.576") and shown(_tr["per_room"]["108"], "0.582")
        assert abs(_pc["miou"] - _tr["miou"]) < 0.03
        # The global feature helps in every room, not only pooled
        assert all(_pn["per_room"][_r] > _ng["per_room"][_r] for _r in ROOMS)
        # Windows 0.13 for both networks; the hallway at best 0.51 (+ covariance)
        assert shown(_pn["iou"]["window"], "0.13") and shown(_pc["iou"]["window"], "0.13")
        _best175 = max(_v["per_room"]["175"] for _v in results.values())
        assert _best175 == _pc["per_room"]["175"] and shown(_best175, "0.51")
        # The vote: doors 0.47, ceilings 0.92, floors 0.96 (+ covariance)
        assert shown(_pc["iou"]["door"], "0.47") and shown(_pc["iou"]["ceiling"], "0.92")
        assert shown(_pc["iou"]["floor"], "0.96")
    if CHECK_TRAINED:  # Room 110 (the labels slide): PointNet 0.70
        assert shown(results["PointNet"]["per_room"]["110"], "0.70")
    return


@app.cell
def _(ALL_ROOMS, PALETTE, TARGETS, mo, np, plt, results):
    # Per-class IoU on the held-out rooms, one bar per method
    _names = [_n for _n in ("height lookup", "covariance + trees", "PointNet",
                            "PointNet + covariance") if _n in results]  # fmt: skip
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _w = 0.8 / len(_names)
    _x = np.arange(len(TARGETS))
    for _i, _n in enumerate(_names):
        _v = [results[_n]["iou"][_k] for _k in TARGETS]
        _ax.bar(_x + (_i - (len(_names) - 1) / 2) * _w, _v, _w, color=PALETTE[_i % 4],
                hatch=["", "//", "", ".."][_i % 4], label=f"{_n} ({results[_n]['miou']:.2f})")  # fmt: skip
    _ax.set_xticks(_x, TARGETS)
    _ax.set_ylabel("IoU, rooms held out")
    _ax.set_ylim(0, 1)
    _ax.legend(frameon=False, fontsize=9, ncols=2, loc="upper right")
    _scope = (
        "pooled over the five held-out rooms"
        if ALL_ROOMS
        else "room 110 only for the learned models (the height lookup: all five rooms)"
    )
    mo.vstack(
        [
            mo.md(f"""
    ### Neighbour features close PointNet's gap

    Intersection over union per class, {_scope}; mean IoU in the legend.
    """),
            _fig,
        ]
    )
    return


@app.cell
def _(ROOMS, mo, results):
    _rows = "\n    ".join(
        f"| {_n} | {_v['miou']:.2f} | "
        + " | ".join(f"{_v['per_room'][_r]:.2f}" if _r in _v["per_room"] else "–" for _r in ROOMS)
        + " |"
        for _n, _v in results.items()
    )
    _tr, _pn = results["covariance + trees"], results["PointNet"]
    _pc, _ng = results["PointNet + covariance"], results["PointNet, no global feature"]
    _run = [_r for _r in ROOMS if _r in _pn["per_room"]]
    _glob = sum(_pn["per_room"][_r] > _ng["per_room"][_r] for _r in _run)
    _ahead = [_r for _r in _run if _pc["per_room"][_r] > _tr["per_room"][_r]]
    mo.md(f"""
    | Method | Mean IoU | 108 | 109 | 110 | 111 | 175 (hallway) |
    |:-------|---------:|----:|----:|----:|----:|----:|
    {_rows}

    Plain PointNet sees each point alone plus the column's maxima: no point knows its
    neighbours. The max-pooled global feature earns its place: with it, PointNet is ahead in
    {_glob} of the {len(_run)} rooms trained. The covariance features say what each point's
    neighbourhood is like (flat or scattered, vertical or level, how large); given them as six
    more inputs, PointNet is ahead of the trees in {len(_ahead)} of {len(_run)}
    ({", ".join(_ahead) or "none"}), and the two differ by {_pc["miou"] - _tr["miou"]:+.3f} in
    mean IoU: close, with one training seed each; another seed would move both by an amount we
    did not measure. In the slides' run (all five rooms), windows are the lowest class of every
    method, and the hallway (175) the hardest room for all.
    """)
    return


@app.cell
def _(
    CHECK_TRAINED,
    CLASS_COLORS,
    NET_TOL,
    PALETTE,
    SLICE,
    TARGETS,
    c,
    confusion,
    keep,
    mo,
    np,
    plt,
    pn_pred,
    shown,
):
    # Room 110, which the network never saw: its confusion counts and the plan view
    _r = keep["room"]
    _te = c.room == _r
    cm_110 = confusion(c.y[_te], pn_pred[_te])
    _off = cm_110 - np.diag(np.diag(cm_110))
    mistaken_for = {  # each class's commonest wrong label, and its share of that class
        _k: (TARGETS[int(_off[_i].argmax())], float(_off[_i].max() / cm_110[_i].sum()))
        for _i, _k in enumerate(TARGETS)
    }
    _h = c.feat[:, 2]
    _m = _te & (_h > SLICE[0]) & (_h < SLICE[1]) & (c.y >= 0)
    _xy, _lab, _wrong = c.xyz[_m, :2], c.y[_m], pn_pred[_m] != c.y[_m]
    slice_wrong = float(_wrong.mean())
    print(f"room 110: {slice_wrong:.4f} of the slice wrong; mistaken for {mistaken_for}")
    _ref_cm = [[60448, 0, 2791, 0, 430, 238], [0, 49500, 42, 2, 0, 869],
               [1505, 131, 33691, 302, 234, 4475], [140, 120, 535, 2054, 0, 2835],
               [262, 0, 625, 0, 2772, 20], [0, 854, 10452, 319, 127, 58795]]  # fmt: skip
    # The labelled points per class do not depend on the network
    assert cm_110.sum(1).tolist() == [sum(_row) for _row in _ref_cm]
    if CHECK_TRAINED:
        print(f"confusion counts as in the slides' run: {cm_110.tolist() == _ref_cm}")
        # The notes: wall called clutter 11%, door called clutter 50%, window called wall 17%;
        # 18% of the slice wrong
        assert mistaken_for["wall"][0] == mistaken_for["door"][0] == "clutter"
        assert mistaken_for["window"][0] == "wall"
        assert abs(mistaken_for["wall"][1] - 0.11093757747037533) < NET_TOL
        assert abs(mistaken_for["door"][1] - 0.4987684729064039) < 2 * NET_TOL
        assert abs(mistaken_for["window"][1] - 0.16988312041315576) < NET_TOL
        assert shown(100 * mistaken_for["wall"][1], "11")
        assert shown(100 * mistaken_for["door"][1], "50")
        assert shown(100 * mistaken_for["window"][1], "17")
        assert abs(slice_wrong - 0.17787700995214162) < NET_TOL and shown(100 * slice_wrong, "18")
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    _a1.scatter(*_xy[_lab <= 1].T, s=0.3, c="#dddddd", linewidths=0, rasterized=True)
    for _k in range(2, len(TARGETS)):
        _a1.scatter(*_xy[_lab == _k].T, s=0.3, c=CLASS_COLORS[_k], linewidths=0, rasterized=True)
        _a1.scatter([], [], c=CLASS_COLORS[_k], s=30, label=TARGETS[_k])
    _a1.legend(loc="center right", bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=10)
    _a2.scatter(*_xy[~_wrong].T, s=0.3, c="#cccccc", linewidths=0, rasterized=True)
    _a2.scatter(*_xy[_wrong].T, s=0.6, c=PALETTE[1], linewidths=0, rasterized=True)
    _a2.scatter([], [], c="#cccccc", s=30, label="right")
    _a2.scatter([], [], c=PALETTE[1], s=30, label="wrong")
    _a2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    for _ax, _title in ((_a1, "labels"), (_a2, "PointNet: right or wrong")):
        _ax.set(title=_title, aspect="equal", xticks=[], yticks=[])
        for _side in _ax.spines.values():
            _side.set_visible(False)
    mo.vstack(
        [
            mo.md(f"""
    ### Room 110, a room it never saw

    From above, floor and ceiling left out: {100 * slice_wrong:.0f}% of these points wrong. The
    commonest mistakes: wall points called {mistaken_for["wall"][0]}
    ({100 * mistaken_for["wall"][1]:.0f}% of the wall points), door points called
    {mistaken_for["door"][0]} ({100 * mistaken_for["door"][1]:.0f}%), window points called
    {mistaken_for["window"][0]} ({100 * mistaken_for["window"][1]:.0f}%). Where does the orange
    gather, and what would a technician check first?
    """),
            _fig,
        ]
    )
    return cm_110, mistaken_for, slice_wrong


@app.cell
def _(CHECK_TRAINED, EPOCHS_RUN, PALETTE, ROOMS, mo, net_params, plt, train_info):
    # Training curves of room 110's networks; every network's loss fell
    _ref = {  # numbers.json: each network's final training loss, rooms 108 to 175
        "PointNet, no global feature": [0.2952497214078903, 0.3370584150776267,
                                        0.34722710868646933, 0.3627443929964846,
                                        0.3118028356915429],
        "PointNet": [0.09655032667020956, 0.0901287354528904, 0.09110225338575452,
                     0.0991144295443188, 0.08590540440664404],
        "PointNet + covariance": [0.06279280136028925, 0.0583134023938328, 0.05822169529490693,
                                  0.06803810909729112, 0.05550901653865973],
    }  # fmt: skip
    for _n, _rooms in train_info.items():
        for _room, _info in _rooms.items():
            assert _info["curve"][-1] < _info["curve"][0], (_n, _room)
        print(
            f"{_n}: final loss "
            + ", ".join(f"{_r} {_i['curve'][-1]:.4f}" for _r, _i in _rooms.items())
        )
    if CHECK_TRAINED:
        _same = all(
            abs(_info["curve"][-1] - _ref[_n][ROOMS.index(_room)]) < 1e-4
            for _n, _rooms in train_info.items()
            for _room, _info in _rooms.items()
        )
        print(f"final losses as in the slides' run, to 1e-4: {_same}")
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for (_n, _rooms), _c, _ls in zip(train_info.items(), PALETTE, ("-", "--", ":"), strict=False):
        _curve = _rooms["110"]["curve"]
        _ax.plot(range(1, len(_curve) + 1), _curve, color=_c, ls=_ls, lw=2, label=_n)
    _ax.set(xlabel="epoch", ylabel="training loss (cross-entropy)", yscale="log")
    _ax.legend(frameon=False, fontsize=9)
    _weights = ", ".join(f"{_n} {_k:,}" for _n, _k in net_params.items())
    mo.vstack(
        [
            mo.md(
                f"Training loss per epoch, the networks that held out room 110 ({EPOCHS_RUN} "
                "epochs; the cosine schedule ends at a rate of zero, so the last epochs barely "
                f"move). Weights: {_weights}."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(
    CHECK_TRAINED,
    CLASS_COLORS,
    FULL_SPAN,
    NPTS,
    PALETTE,
    ROUNDOFF,
    SEED,
    TARGETS,
    c,
    column,
    column_3d,
    columns,
    copy,
    keep,
    mix,
    mo,
    np,
    pick,
    pointnet_checks,
    shown,
    torch,
):
    # The picture with the trained network (room 110's, which never saw the room): its
    # critical points and its labels, on the attempt's column. Float64 on the CPU, as the checks.
    _b = max(
        [_x for _x in keep["test_bl"] if (np.ptp(c.xyz[_x.idx, :2], axis=0) >= FULL_SPAN).all()],
        key=mix,
    )
    assert np.array_equal(_b.idx, column.idx)  # the same column as the attempt
    net_checks = pointnet_checks(keep["model"], c, column, "cpu")
    _m64 = copy.deepcopy(keep["model"]).double().cpu().eval()
    with torch.no_grad():
        _x = columns(c, [column], [pick], "cpu").double()
        _, _g, _arg = _m64.pooled(_x)
        _pred = _m64(_x)[0].argmax(-1).numpy()
        # Repeats change no maximum: 1,000 of the points, padded to 4,096 with repeats of
        # themselves, give the same global feature as the 1,000 alone
        _few = _x[:, :1000]
        _rep = torch.from_numpy(np.random.default_rng(SEED).integers(0, 1000, NPTS - 1000))
        _pad = torch.cat([_few, _few[:, _rep]], 1)
        _repeats = float((_m64.pooled(_few)[1] - _m64.pooled(_pad)[1]).abs().max())
    crit = np.unique(_arg[0].numpy())
    _y = c.y[pick]
    column_accuracy = float(np.mean(_pred[_y >= 0] == _y[_y >= 0]))
    crit_share = {_k: int((_y[crit] == _i).sum()) for _i, _k in enumerate(TARGETS)}
    print(f"checks {net_checks}; repeats {_repeats:.1e}; column accuracy {column_accuracy:.4f}; "
          f"critical points by class {crit_share}")  # fmt: skip
    # The table: shuffled points change the class scores by round-off only; the critical points
    # alone give the global feature; and the padding of a short column changes nothing
    assert net_checks["permutation_max_diff"] < ROUNDOFF
    assert net_checks["critical_global_max_diff"] < ROUNDOFF and _repeats < ROUNDOFF
    assert len(crit) == net_checks["critical_points"] < NPTS
    if CHECK_TRAINED:
        # 576 of 4,096 critical points; 80% of the column's labelled points right; and the
        # table's round-off digits, 8e-15 and 2e-16 (this machine's float64 kernels)
        assert len(crit) == 576 and shown(100 * column_accuracy, "80")
        assert abs(column_accuracy - 0.8018456805947193) < 0.01
        assert shown(net_checks["permutation_max_diff"], "8e-15")
        assert shown(net_checks["critical_global_max_diff"], "2e-16")
        # They spread over every surface in the column: every class with at least 5% of the
        # column's points holds critical points
        assert all(crit_share[_k] > 0 for _k in ("ceiling", "floor", "wall", "window", "clutter"))
    _f2 = column_3d([(np.arange(NPTS), "#c8c8c8", 0.8), (crit, PALETTE[1], 6.0)])
    _f3 = column_3d([(_pred == _k, CLASS_COLORS[_k], 1.5) for _k in range(6)], legend=True)
    mo.vstack(
        [
            mo.md(f"""
    ### The picture, with the trained network

    **The critical points:** {len(crit)} of the 4,096 points (orange) set the largest value of
    at least one of the network's 1,024 features. By class: {", ".join(f"{_k} {_v}" for _k, _v in crit_share.items())}.
    On single objects Qi et al. found the critical points sketching the outline; in this
    cluttered column they fall on every surface it holds.
    """),
            _f2,
            mo.md(f"""
    **The labels:** each point coloured by the class PointNet gives it;
    {100 * column_accuracy:.0f}% of the column's labelled points right. Each label comes from
    the point's own features (where it is) and the column's 1,024 maxima (what the column holds).
    """),
            _f3,
        ]
    )
    return column_accuracy, crit, net_checks


@app.cell
def _(REG_VOXEL, STRUCTURE, c_full, np, voxel_first):
    # The scan for registration: one point per 5-cm voxel of the whole sample, and the points
    # labelled as the building (ceiling to window)
    ref = c_full.take(voxel_first(c_full.xyz, REG_VOXEL))
    struct = np.isin(ref.y, STRUCTURE)
    return ref, struct


@app.cell
def _(
    BASIN_VOXEL,
    REFERENCE,
    ROUNDOFF,
    mo,
    net_checks,
    ref,
    registration_checks,
    shown,
    struct,
    voxel_first,
):
    _p = ref.xyz[struct]
    reg_checks = registration_checks(_p[voxel_first(_p, BASIN_VOXEL)])
    print(f"registration checks: {reg_checks}")
    # The SVD rotation against SciPy's, and ICP recovering a known motion of 92,568 points of
    # the scan in 39 iterations: round-off only
    assert reg_checks["icp_points"] == 92_568 and reg_checks["icp_known_iterations"] == 39
    assert reg_checks["kabsch_vs_scipy"] < ROUNDOFF and reg_checks["kabsch_vs_truth"] < ROUNDOFF
    assert reg_checks["icp_known_rms"] < ROUNDOFF
    if REFERENCE:  # the table's round-off digits, 4e-16 and 4e-15, on the slides' machine
        assert shown(reg_checks["kabsch_vs_scipy"], "4e-16")
        assert shown(reg_checks["icp_known_rms"], "4e-15")
    mo.md(f"""
    ### Same answer, any order

    | Check | Value |
    |:------|------:|
    | Shuffled points: largest change in any class score | {net_checks["permutation_max_diff"]:.0e} |
    | Critical points of 4,096 | {net_checks["critical_points"]} |
    | Global feature from the critical points alone: largest change | {net_checks["critical_global_max_diff"]:.0e} |
    | Our SVD rotation against SciPy's | {reg_checks["kabsch_vs_scipy"]:.0e} |
    | ICP recovering the known motion: RMS point error, m | {reg_checks["icp_known_rms"]:.0e} |

    Shuffling permutes the outputs and changes nothing else, up to round-off (float64). Keeping
    only the critical points leaves every maximum where it was, so the global feature is
    unchanged: PointNet's answer rests on a few hundred points. The SVD solution matches SciPy's
    to round-off. ICP, given {reg_checks["icp_points"]:,} points of the scan and a moved copy of
    them, finds the motion again in {reg_checks["icp_known_iterations"]} iterations. These
    values are round-off: their digits change with the machine; their size does not.
    """)
    return (reg_checks,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The model: IfcOpenShell meshes, points spread over the surfaces

    IfcOpenShell turns each IFC element (walls, slabs, ceilings, columns, doors, windows,
    stairs) into a triangle mesh in the model's frame. Its geometry iterator runs on several
    threads and returns the elements in whatever order the threads finish, so they are sorted
    back into the file's order before anything random touches them: the seeded points spread
    over the surfaces then come out the same every run. Points are drawn uniformly over each
    element's area, about one per 5 × 5 cm (for ICP and coverage) or 2 × 2 cm (for the
    distances), each with its triangle's normal and its element.
    """)
    return


@app.cell
def _(BIM_TYPES, IFC, dataclass, ifcopenshell, np, os):
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
            One ``Element`` per meshed element, in the order of ``BIM_TYPES`` and then of the
            file (the threaded iterator returns them in a different order each run).

        Raises:
            RuntimeError: If IfcOpenShell cannot start the geometry iterator.
        """
        model = ifcopenshell.open(str(IFC))
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

    def bim_cloud(els: list[Element], spacing: float, seed: int) -> dict[str, np.ndarray]:
        """Points spread over every element's surfaces, with each point's element and normal."""
        rng = np.random.default_rng(seed)
        parts = [sample_surface(e, spacing, rng) for e in els]
        return {
            "xyz": np.concatenate([p for p, _ in parts]),
            "normal": np.concatenate([n for _, n in parts]),
            "owner": np.repeat(np.arange(len(els)), [len(p) for p, _ in parts]),
        }

    return bim_cloud, bim_elements, sample_surface


@app.cell
def _(BIM_SPACING, IFC_NAMES, SEED, bim_cloud, bim_elements, cKDTree, np, time):
    _t0 = time.perf_counter()
    els = bim_elements()
    bim = bim_cloud(els, BIM_SPACING, SEED)
    bim_tree = cKDTree(bim["xyz"])
    _types = {}
    for _e in els:
        _types[IFC_NAMES[_e.ifc_type]] = _types.get(IFC_NAMES[_e.ifc_type], 0) + 1
    print(f"model meshed in {time.perf_counter() - _t0:.0f} s: {len(els)} elements {_types}, "
          f"{len(bim['xyz']):,} points")  # fmt: skip
    # The slide: 135 elements; 1,297,494 points every 5 cm (seeded, in file order)
    assert len(els) == 135 and len(bim["xyz"]) == 1_297_494
    assert _types == {"wall": 61, "slab": 5, "ceiling": 7, "column": 10, "door": 6, "window": 44,
                      "stair": 2}  # fmt: skip
    assert np.allclose(np.linalg.norm(bim["normal"], axis=1), 1, atol=1e-6)
    return bim, bim_tree, els


@app.cell
def _(
    EXACT,
    PALETTE,
    Rotation,
    SLICE,
    bim,
    bim_tree,
    icp,
    mo,
    np,
    plt,
    ref,
    shown,
    struct,
):
    def coarse_start(scan: np.ndarray, model_pts: np.ndarray) -> np.ndarray:
        """The shift that matches the centres of the two bounding boxes (no turn)."""
        return (model_pts.min(0) + model_pts.max(0)) / 2 - (scan.min(0) + scan.max(0)) / 2

    # The reference registration: the labelled structure points, 5-cm voxels, the cap shrinking
    t_init = coarse_start(ref.xyz[struct], bim["xyz"])
    r_ref, t_ref, ref_hist = icp(ref.xyz[struct], bim_tree, bim["xyz"], np.eye(3), t_init)
    ref_turn = Rotation.from_matrix(r_ref).as_euler("xyz", degrees=True)
    print(f"coarse {t_init}; shift {t_ref}; turn {ref_turn} deg; {ref_hist}")
    assert int(struct.sum()) == 379_916
    assert np.allclose(t_init, [-9.935757772811243, -0.2855025437394936, 0.6304999971389773],
                       rtol=0, atol=EXACT)  # fmt: skip
    assert np.allclose(t_ref, [-9.848693884713795, -0.3890843704823051, 1.253512438788063],
                       rtol=0, atol=EXACT)  # fmt: skip
    assert np.allclose(ref_turn, [-0.01161146290040094, -0.010564327899567821, 0.7751460816695239],
                       rtol=0, atol=1e-7)  # fmt: skip
    assert [_h["iterations"] for _h in ref_hist] == [72, 31, 61, 62]
    assert np.allclose([_h["paired"] for _h in ref_hist], [0.9999894713568263, 0.9995999115593973,
                       0.9885685256740964, 0.9695327388159488], rtol=0, atol=EXACT)  # fmt: skip
    assert np.allclose([_h["median"] for _h in ref_hist], [0.026214407746912728,
                       0.026220143062509325, 0.026130366113998296, 0.02554147613825338],
                       rtol=0, atol=EXACT)  # fmt: skip
    # The notes: coarse start (-9.94, -0.29, 0.63) m; then (-9.85, -0.39, 1.25) m and 0.78°
    # about the vertical (the turns about x and y are hundredths of a degree)
    assert all(shown(_v, _t) for _v, _t in zip(t_init, ("-9.94", "-0.29", "0.63"), strict=True))
    assert all(shown(_v, _t) for _v, _t in zip(t_ref, ("-9.85", "-0.39", "1.25"), strict=True))
    assert shown(ref_turn[2], "0.78") and np.abs(ref_turn[:2]).max() < 0.02
    # From above: the model's surfaces (grey) and the scan's walls (blue), as given and after ICP
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    _bz = bim["xyz"][:, 2]
    _sel_b = (_bz > 0.1 + SLICE[0]) & (_bz < 0.1 + SLICE[1])  # the model's floor is at z = 0.1 m
    _h = ref.feat[struct, 2]
    _s = (_h > SLICE[0]) & (_h < SLICE[1])
    for _ax, (_r, _t), _title in zip(_axes, ((np.eye(3), np.zeros(3)), (r_ref, t_ref)),
                                     ("as given: two frames", "after ICP"), strict=True):  # fmt: skip
        _ax.scatter(*bim["xyz"][_sel_b, :2].T, s=0.2, c="#9a9a9a", linewidths=0, rasterized=True)
        _q = ref.xyz[struct][_s] @ _r.T + _t
        _ax.scatter(*_q[:, :2].T, s=0.2, c=PALETTE[0], linewidths=0, rasterized=True)
        _ax.set(title=_title, aspect="equal", xticks=[], yticks=[])
        for _side in _ax.spines.values():
            _side.set_visible(False)
    mo.vstack(
        [
            mo.md(f"""
    ### Line the scan up with the model

    A coarse start matches the centres of the two bounding boxes: a shift of
    ({t_init[0]:.2f}, {t_init[1]:.2f}, {t_init[2]:.2f}) m. Then ICP on the
    {int(struct.sum()):,} labelled structure points (5-cm voxels), the cap shrinking from 1 m to
    10 cm: a shift of ({t_ref[0]:.2f}, {t_ref[1]:.2f}, {t_ref[2]:.2f}) m and a turn of
    {ref_turn[2]:.2f}° about the vertical. Per cap: {", ".join(f"{_h['cap']:g} m: {_h['iterations']} iterations, {100 * _h['paired']:.1f}% paired, median {100 * _h['median']:.1f} cm" for _h in ref_hist)}.
    Below, a horizontal slice from above: the model's surfaces in grey, the scan's walls in blue.
    """),
            _fig,
        ]
    )
    return coarse_start, r_ref, ref_turn, t_init, t_ref


@app.cell
def _(
    ASSIGN_D,
    BIM_SPACING,
    COVER_D,
    DEV_SPACING,
    EXACT,
    IFC_NAMES,
    IGNORE,
    SEED,
    STRUCTURE,
    VERTICAL,
    bim_cloud,
    c_full,
    cKDTree,
    els,
    np,
    r_ref,
    sample_surface,
    shown,
    t_ref,
    time,
):
    # Scan vs model: each element's coverage, each scan point's distance to the model
    _t0 = time.perf_counter()
    scan_model = c_full.xyz @ r_ref.T + t_ref  # the whole 2% sample, in the model's frame
    _scan_tree = cKDTree(scan_model)
    _rng = np.random.default_rng(SEED)
    cov = {}
    rows = []
    for _e in els:
        _p, _ = sample_surface(_e, BIM_SPACING, _rng)
        _dist, _ = _scan_tree.query(_p, distance_upper_bound=COVER_D, workers=-1)
        _share = float(np.isfinite(_dist).mean())
        _kind = IFC_NAMES[_e.ifc_type]
        cov.setdefault(_kind, []).append(_share)
        rows.append(
            {"type": _kind, "name": _e.name, "coverage": _share, "area": len(_p) * BIM_SPACING**2}
        )
    _fine = bim_cloud(els, DEV_SPACING, SEED + 1)
    _near, _idx = cKDTree(_fine["xyz"]).query(scan_model, workers=-1)
    dev = np.abs(np.einsum("ij,ij->i", scan_model - _fine["xyz"][_idx], _fine["normal"][_idx]))
    # Each element's points: the scan points whose nearest model surface (within 0.30 m) is on
    # that element. A wall is judged by its labelled points, the wall, door, and window points
    # (the ceiling and floor points next to its top and foot would measure the ceiling and the
    # floor), and checked against its unassigned points (§6)
    _vertical = np.isin(c_full.y, VERTICAL)
    _paired = np.where(_near < ASSIGN_D, _fine["owner"][_idx], -1)
    for _pre, _kind_mask in (("", _vertical), ("unassigned_", c_full.y == IGNORE)):
        _owner = np.where(_kind_mask, _paired, -1)
        _order = np.argsort(_owner, kind="stable")
        _starts = np.searchsorted(_owner[_order], np.arange(len(els) + 1))
        for _k in range(len(els)):
            _sel = _order[_starts[_k] : _starts[_k + 1]]
            rows[_k][f"{_pre}points"] = len(_sel)
            rows[_k][f"{_pre}deviation"] = float(np.median(dev[_sel])) if len(_sel) else None
    _s = np.isin(c_full.y, STRUCTURE)
    deviation = {
        "structure_median": float(np.median(dev[_s])),
        "structure_p90": float(np.quantile(dev[_s], 0.9)),
        "structure_within_5cm": float(np.mean(dev[_s] < 0.05)),
        "walls_median": float(np.median(dev[_vertical])),
        "clutter_median": float(np.median(dev[c_full.y == 5])),
    }
    coverage = {
        _k: {
            "n": len(_v),
            "median": float(np.median(_v)),
            "below_10pct": int(np.sum(np.asarray(_v) < 0.1)),
        }
        for _k, _v in cov.items()
    }
    print(f"compared in {time.perf_counter() - _t0:.0f} s; {deviation}; {coverage}")
    # Every element in file order (lesson: a threaded mesher's order is sorted before sampling)
    assert rows[0]["name"] == "Basic Wall:(35cm) Tijolo cerâmico acabamento pintura branca:914875"
    assert rows[-1]["name"] == "Cast-In-Place Stair:Stair:975096 Run 2"
    assert abs(sum(_w["coverage"] for _w in rows) - 41.2866148500229) < EXACT
    assert abs(sum(_w["area"] for _w in rows) - 3243.7350000000006) < 1e-6
    # Each element's labelled and unassigned points and their median distances, summed over the
    # elements, as in the slides' run (numbers.json, elements_table)
    _sums = {
        _pre: (
            sum(_w[f"{_pre}points"] for _w in rows),
            sum(_w[f"{_pre}deviation"] is None for _w in rows),
            sum(_w[f"{_pre}deviation"] for _w in rows if _w[f"{_pre}deviation"] is not None),
        )
        for _pre in ("", "unassigned_")
    }
    _sums_ref = {"": (2334332, 36, 4.593419767021768),
                 "unassigned_": (768772, 22, 7.964805099091203)}  # fmt: skip
    for _pre, (_n, _none, _sum) in _sums_ref.items():
        assert _sums[_pre][:2] == (_n, _none) and abs(_sums[_pre][2] - _sum) < EXACT, _pre
    _dref = {"structure_median": 0.005834818786143144, "structure_p90": 0.04012028687817476,
             "structure_within_5cm": 0.9409461729748126, "walls_median": 0.023745098348864506,
             "clutter_median": 0.44237779409732253}  # fmt: skip
    assert all(abs(deviation[_k] - _v) < EXACT for _k, _v in _dref.items())
    _cref = {"wall": (61, 0.23204540493342066, 22), "slab": (5, 0.015011209669558436, 4),
             "ceiling": (7, 0.8238416892815403, 1), "column": (10, 0.00041597337770382697, 10),
             "door": (6, 0.40918084685397704, 0), "window": (44, 0.4393787575150301, 8),
             "stair": (2, 0.15952604096933995, 1)}  # fmt: skip
    for _k, (_n, _med, _below) in _cref.items():
        assert coverage[_k]["n"] == _n and coverage[_k]["below_10pct"] == _below, _k
        assert abs(coverage[_k]["median"] - _med) < EXACT, _k
    # One of the five IFC slabs is the stair's landing; the other four are floors
    _slabs = [_w["name"] for _w in rows if _w["type"] == "slab"]
    assert sum(_n.startswith("Floor:") for _n in _slabs) == 4
    assert [_n for _n in _slabs if not _n.startswith("Floor:")] == [
        "Cast-In-Place Stair:Stair:975096 Landing 1"
    ]
    # The notes: median 0.006 m, 94% within 5 cm, 90% within 0.040 m; walls, doors, windows
    # 0.024 m; clutter 0.44 m. Coverage medians: ceilings 82%, windows 44%, doors 41%, walls
    # 23%, slabs 2%, columns 0%; under 10%: 10 of 10 columns, 4 of 5 slabs, 22 of 61 walls
    assert shown(deviation["structure_median"], "0.006") and shown(
        100 * deviation["structure_within_5cm"], "94"
    )
    assert shown(deviation["structure_p90"], "0.040") and shown(deviation["walls_median"], "0.024")
    assert shown(deviation["clutter_median"], "0.44")
    assert all(shown(100 * coverage[_k]["median"], _t) for _k, _t in
               (("ceiling", "82"), ("window", "44"), ("door", "41"), ("wall", "23"),
                ("slab", "2"), ("column", "0")))  # fmt: skip
    return cov, coverage, dev, deviation, rows, scan_model


@app.cell
def _(
    COVER_D,
    DEV_MAX,
    MUTED,
    PALETTE,
    SEED,
    SEEN,
    SLICE,
    c_full,
    cov,
    coverage,
    dev,
    deviation,
    mo,
    np,
    plt,
    scan_model,
):
    # From above: each wall, door, and window point coloured by its distance to the model
    _walls = np.isin(c_full.y, (2, 3, 4))
    _h = c_full.feat[:, 2]
    _in = _walls & (_h > SLICE[0]) & (_h < SLICE[1])
    _p, _d = scan_model[_in], dev[_in]
    _order = np.argsort(_d)
    _f1, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _sc = _ax.scatter(*_p[_order, :2].T, c=_d[_order], s=0.3, cmap="magma_r", vmin=0, vmax=DEV_MAX,
                      linewidths=0, rasterized=True)  # fmt: skip
    _ax.set(aspect="equal", xticks=[], yticks=[])
    for _side in _ax.spines.values():
        _side.set_visible(False)
    _f1.colorbar(_sc, ax=_ax, shrink=0.8, extend="max", label="distance to the model, m")
    # Each element's coverage, by IFC type, with the median
    _f2, _ax2 = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _rng = np.random.default_rng(SEED)
    for _i, _n in enumerate(cov):
        _v = np.asarray(cov[_n])
        _ax2.scatter(
            _i + _rng.uniform(-0.18, 0.18, len(_v)), _v, s=16, alpha=0.75, color=PALETTE[0]
        )
        _ax2.plot([_i - 0.3, _i + 0.3], [np.median(_v)] * 2, color=PALETTE[1], lw=3)
    _ax2.axhline(SEEN, color=MUTED, ls=":", lw=1)
    _ax2.set_xticks(range(len(cov)), [f"{_n}\n({len(cov[_n])})" for _n in cov], fontsize=10)
    _ax2.set_ylabel(f"surface within {COVER_D * 100:.0f} cm of the scan")
    _ax2.set_ylim(-0.03, 1.03)
    mo.vstack(
        [
            mo.md(f"""
    ### Most of the lab is where the model says

    With the scan in the model's frame, each scan point's distance to the plane of the nearest
    model surface (the model's triangles, sampled every 2 cm). Over the building's points
    (ceiling, floor, wall, door, window): median {deviation["structure_median"]:.3f} m,
    {100 * deviation["structure_within_5cm"]:.0f}% within 5 cm, 90% within
    {deviation["structure_p90"]:.3f} m. The fit was made on (a thinning of) these same points,
    so a small median is expected; what matters is where the dark points gather. Walls, doors,
    and windows alone: median {deviation["walls_median"]:.3f} m; clutter, for comparison,
    {deviation["clutter_median"]:.2f} m. A dark stretch of wall: a wall built elsewhere, a model
    drawn wrong, or a poor fit there; the map alone cannot say which.
    """),
            _f1,
            mo.md(f"""
    ### The scan saw under 10% of every column

    Coverage: points every 5 cm over each element's surface, and the share with a scan point
    within 5 cm. Medians: ceilings {100 * coverage["ceiling"]["median"]:.0f}%, windows
    {100 * coverage["window"]["median"]:.0f}%, doors {100 * coverage["door"]["median"]:.0f}%,
    walls {100 * coverage["wall"]["median"]:.0f}%, slabs {100 * coverage["slab"]["median"]:.0f}%,
    columns {100 * coverage["column"]["median"]:.0f}%. Under 10%: {coverage["column"]["below_10pct"]}
    of {coverage["column"]["n"]} columns, {coverage["slab"]["below_10pct"]} of
    {coverage["slab"]["n"]} slabs (IFC slabs: the floors and the stair's landing),
    {coverage["wall"]["below_10pct"]} of {coverage["wall"]["n"]} walls. A scanner records only
    the surfaces in its line of sight; a missing column and a column hidden behind a finish leave
    the same trace: none. Low coverage within 5 cm is not the same as unseen, though: a wall
    displaced by more than 5 cm has little surface within 5 cm of the scan, however many points
    sit on it. §6 judges each wall by the points paired with it instead.
    """),
            _f2,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: predict four failures

    1. Score the trees on random 1.5-m columns instead of whole rooms.
    2. Scan the held-out room at a tenth of the density.
    3. Start ICP turned 30°, 45°, or 90° from the coarse start.
    4. Run ICP with no distance cap, with and without the clutter.

    Write what happens to each, then reveal them one by one.
    """)
    return


@app.cell
def _(
    ALL_ROOMS,
    ROOMS,
    TREE_TOL,
    c,
    confusion,
    mo,
    np,
    rand_pred,
    rand_te,
    random_split,
    results,
    scores,
    shown,
    tree_pred,
):
    # 1. Random columns. Like for like: the leave-one-room-out trees scored on the same
    # test points as the random split (in the reduced setting, the points of room 110 only)
    _tr = results["covariance + trees"]
    _same = rand_te & (tree_pred >= 0)
    lfl_rooms = scores(confusion(c.y[_same], tree_pred[_same]))
    lfl_random = scores(confusion(c.y[_same], rand_pred[_same]))
    print(f"on the random split's test points{'' if ALL_ROOMS else ' in room 110'}: "
          f"rooms held out {lfl_rooms['miou']:.4f}, random columns {lfl_random['miou']:.4f}")  # fmt: skip
    if ALL_ROOMS:
        assert np.array_equal(_same, rand_te)
        assert abs(lfl_random["miou"] - random_split["miou"]) < 1e-12
        # On the random split's own test points, the rooms-held-out trees score 0.63 (pooled over all points: 0.61)
        assert abs(lfl_rooms["miou"] - 0.6312035550736729) < TREE_TOL
        assert shown(lfl_rooms["miou"], "0.63")
        assert lfl_rooms["miou"] > _tr["miou"]
    # The random split flatters on the same points too
    assert lfl_random["miou"] > lfl_rooms["miou"] + 2 * TREE_TOL
    _rooms_label = "Whole rooms held out (five folds)" if ALL_ROOMS else "Room 110 held out"
    _test_rooms = {ROOMS[_r]: int((rand_te & (c.room == _r)).sum()) for _r in range(len(ROOMS))}
    mo.md(f"""
    ### Random columns flatter the score

    | Covariance + trees, the same features | Mean IoU | Accuracy |
    |:--------------------------------------|---------:|---------:|
    | {_rooms_label} | {_tr["miou"]:.2f} | {100 * _tr["accuracy"]:.0f}% |
    | Random 1.5-m columns held out (one in five) | {random_split["miou"]:.2f} | {100 * random_split["accuracy"]:.0f}% |

    The same features and trees, tested on {random_split["test_columns"]} random columns of
    {random_split["columns"]} after training on the rest. A held-out column's neighbours are in
    training: the other half of its wall, the rest of its desk, the same lighting. The score
    then measures memory of this lab, not labelling the next building.

    **Like for like.** The two rows score different points. On exactly the random split's test
    points ({", ".join(f"{_k}: {_v:,}" for _k, _v in _test_rooms.items())} points by room), the
    trees trained with each point's whole room held out reach {lfl_rooms["miou"]:.2f} against
    the random split's {lfl_random["miou"]:.2f}: of the gap between the table's rows
    ({random_split["miou"] - _tr["miou"]:.3f}), the split makes {lfl_random["miou"] - lfl_rooms["miou"]:.3f}
    and the choice of test points the rest. Hold out what you will meet: whole rooms here, whole
    buildings in practice (counties in Lecture 3, photographs in Lecture 15).
    """)
    return lfl_random, lfl_rooms


@app.cell
def _(
    ALL_ROOMS,
    CHECK_TRAINED,
    K_EIG,
    NET_TOL,
    NPTS,
    ROOMS,
    SPARSE,
    TREE_TOL,
    c,
    cKDTree,
    density,
    mo,
    np,
    results,
    shown,
    sparse_res,
    sparse_thin,
):
    # 2. A sparser scan. The 24 nearest neighbours now span a wider patch: by how much?
    _te = np.flatnonzero(c.room == ROOMS.index("110"))
    _thin = sparse_thin["110"]
    _r_full = cKDTree(c.xyz[_te]).query(c.xyz[_te], k=K_EIG, workers=-1)[0][:, -1]
    _r_thin = cKDTree(c.xyz[_thin]).query(c.xyz[_thin], k=K_EIG, workers=-1)[0][:, -1]
    span_ratio = float(np.median(_r_thin) / np.median(_r_full))
    print(f"room 110: {len(_thin):,} of {len(_te):,} points kept; median distance to the 24th "
          f"neighbour {np.median(_r_full):.3f} -> {np.median(_r_thin):.3f} m (x{span_ratio:.2f}; "
          f"sqrt(10) = {np.sqrt(1 / SPARSE):.2f})")  # fmt: skip
    assert len(_thin) == int(SPARSE * len(_te))
    # About three times wider, the square root of 10 (points on surfaces: area per point x10)
    assert abs(span_ratio - np.sqrt(1 / SPARSE)) < 0.3 and shown(span_ratio, "3")
    # What the network reads: sets of 4,096 points per column. The column the median test point
    # sits in, at full density and at a tenth; the points per set fall by far less than tenfold
    _set_full = NPTS / density["test_column_median"]  # share of a column's points in one set
    _set_drop = NPTS / min(density["sparse_column_median"], NPTS)  # points per set: full / tenth
    print(f"density: {density}; a set holds {100 * _set_full:.0f}% of a median column at full "
          f"density; points per set fall {_set_drop:.1f}-fold at a tenth")  # fmt: skip
    assert density["sparse_column_median"] < NPTS < density["test_column_median"]
    if ALL_ROOMS:  # the slides' run (numbers.json): every fold's columns
        assert density["test_column_median"] == 8999.0
        assert density["sparse_column_median"] == 895.0
        assert density["train_column_median"] == 7386.0
        # trees 0.61 -> 0.49, doors 0.000
        assert abs(sparse_res["covariance + trees"]["miou"] - 0.4874511893911369) < TREE_TOL
        assert shown(sparse_res["covariance + trees"]["miou"], "0.49")
        assert shown(sparse_res["covariance + trees"]["iou"]["door"], "0.000")
    if CHECK_TRAINED and ALL_ROOMS:  # PointNet 0.56 -> 0.56; + covariance 0.63 -> 0.52
        # the trees' door IoU to the slide's digit, on the reference machine: 0.0003
        assert shown(sparse_res["covariance + trees"]["iou"]["door"], "0.0003")
        assert abs(sparse_res["PointNet"]["miou"] - 0.56094823308949) < NET_TOL
        assert abs(sparse_res["PointNet + covariance"]["miou"] - 0.517657991183293) < NET_TOL
        assert shown(sparse_res["PointNet"]["miou"], "0.56")
        assert shown(sparse_res["PointNet + covariance"]["miou"], "0.52")
    _rows = "\n    ".join(
        f"| {_n} | {results[_n]['miou']:.2f} | {sparse_res[_n]['miou']:.2f} |" for _n in sparse_res
    )
    mo.md(f"""
    ### A tenth of the points: PointNet holds

    | Held-out {"rooms" if ALL_ROOMS else "room 110"}, mean IoU | Full scan | A tenth of the points |
    |:-------------------------|----------:|----------------------:|
    {_rows}

    Each held-out room thinned to a tenth of its points (a faster scan, or one from farther
    away), the features recomputed, the same trained models. The two kinds of input change by
    different amounts. PointNet reads each column in sets of 4,096 points: at full density the
    median point sits in a column of {density["test_column_median"]:,.0f} points, so one set
    holds about {100 * _set_full:.0f}% of them; at a tenth, in a column of
    {density["sparse_column_median"]:,.0f}, one set holds them all, filled with repeats (which
    change no maximum, §4). The points in a set fall {_set_drop:.1f}-fold, not tenfold. The
    covariance features are computed over the 24 nearest neighbours, which now span a patch
    {span_ratio:.1f} times wider (median distance to the 24th neighbour in room 110; the square
    root of 10 is 3.2). The trees fall, doors nearly vanish
    (IoU {sparse_res["covariance + trees"]["iou"]["door"]:.4f}), and so does the network fed
    the same features. Train at the density you will scan, or compute features within a fixed
    radius rather than over a fixed count.
    """)
    return (span_ratio,)


@app.cell
def _(
    BASIN_VOXEL,
    CONVERGED,
    EXACT,
    MUTED,
    PALETTE,
    Rotation,
    STRUCTURE,
    TURNS,
    bim,
    bim_tree,
    c_full,
    coarse_start,
    icp,
    mo,
    motion_error,
    np,
    plt,
    r_ref,
    shown,
    t_init,
    t_ref,
    voxel_first,
):
    # 3. The start turned about the scan's centre, 10-cm voxels
    bas = c_full.take(voxel_first(c_full.xyz, BASIN_VOXEL))
    bas_struct = np.isin(bas.y, STRUCTURE)
    p_bas = bas.xyz[bas_struct]
    t_b = coarse_start(p_bas, bim["xyz"])
    _ctr = (p_bas.min(0) + p_bas.max(0)) / 2
    # Which model element each point pairs with at the reference fit
    _owner_ref = bim["owner"][bim_tree.query(p_bas @ r_ref.T + t_ref, workers=-1)[1]]
    turns, wrong_pairs = {}, {}
    for _deg in TURNS:
        _rz = Rotation.from_euler("z", _deg, degrees=True).as_matrix()
        _t0 = t_b + _ctr - _rz @ _ctr  # turned about the centre
        _r, _t, _ = icp(p_bas, bim_tree, bim["xyz"], _rz, _t0)
        turns[_deg] = motion_error(_r, _t, r_ref, t_ref, p_bas)
        _start = bim["owner"][bim_tree.query(p_bas @ _rz.T + _t0, workers=-1)[1]]
        _end = bim["owner"][bim_tree.query(p_bas @ _r.T + _t, workers=-1)[1]]
        wrong_pairs[_deg] = (
            float(np.mean(_start != _owner_ref)),
            float(np.mean(_end != _owner_ref)),
        )
    _r, _t, _ = icp(p_bas, bim_tree, bim["xyz"], np.eye(3), np.zeros(3))
    no_coarse = motion_error(_r, _t, r_ref, t_ref, p_bas)
    print(f"{len(p_bas):,} points; error by start turn {turns}; no coarse start {no_coarse:.3f} m")
    print("points paired with another element than at the reference fit (start, end): "
          + ", ".join(f"{_k}°: {_a:.2f}, {_b:.2f}" for _k, (_a, _b) in wrong_pairs.items()))  # fmt: skip
    _ref = {0: 0.0012052977423788636, 10: 0.0012052977423774429, 20: 0.0012052977423860887,
            30: 1.1560198214983886, 45: 6.4483098497967735, 90: 12.652343232502647,
            180: 17.243289458825675}  # fmt: skip
    assert len(p_bas) == 85_467
    assert all(abs(turns[_k] - _v) < EXACT for _k, _v in _ref.items())
    assert abs(no_coarse - 7.046499715945134) < EXACT
    # The figure and notes: 0.001 m at 0° and 20° (0.0012 m, the same pose from 0° to 20°),
    # 1.16 m at 30°, 12.7 m at 90°; 7.0 m without the coarse start, whose shift is about 10 m
    assert max(turns[_k] for _k in (0, 10, 20)) - min(turns[_k] for _k in (0, 10, 20)) < 1e-9
    assert all(turns[_k] < CONVERGED for _k in (0, 10, 20)) and turns[30] > CONVERGED
    assert shown(turns[0], "0.001") and shown(turns[20], "0.0012") and shown(turns[30], "1.16")
    assert shown(turns[90], "12.7") and shown(no_coarse, "7.0")
    assert shown(float(np.linalg.norm(t_init)), "10")
    # Why it stops: up to 20° every point ends on the element it pairs with at the reference
    # pose, though at 20° half of them start on another (52%); from 30° on, 27% or more end on
    # another element. A share of wrong pairs at the start does not by itself predict failure.
    assert all(wrong_pairs[_k][1] < 0.005 for _k in (0, 10, 20))
    assert all(wrong_pairs[_k][1] > 0.25 for _k in (30, 45, 90, 180))
    assert shown(wrong_pairs[20][0], "0.52") and shown(wrong_pairs[30][0], "0.59")
    assert shown(wrong_pairs[30][1], "0.27")
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _a = np.array(list(turns))
    _ax.plot(_a, np.maximum(list(turns.values()), 1e-4), marker="o", color=PALETTE[0])
    _ax.axhline(CONVERGED, color=MUTED, ls=":", lw=1)
    _ax.text(
        _a[-1], CONVERGED * 1.3, f"{CONVERGED * 100:.0f} cm", ha="right", color=MUTED, fontsize=10
    )
    _ax.set(yscale="log", xticks=_a, xlabel="start turned from the coarse start, degrees",
            ylabel="from the reference fit, m (RMS)")  # fmt: skip
    mo.vstack(
        [
            mo.md(f"""
    ### ICP needs a start within 20°

    The same scan ({len(p_bas):,} structure points, 10-cm voxels), started turned about its centre
    by 0° to 180° from the coarse start; each result is measured as its distance from the
    reference fit (itself a fit, not the true pose). Up to 20°, ICP lands on the reference fit
    (within {turns[0]:.4f} m, the difference between 10-cm and 5-cm thinning: the same pose from every
    start up to 20°); from 30° it stops {turns[30]:.2f} m away, from 90° {turns[90]:.1f} m. With no
    coarse start at all (the two frames as given, {np.linalg.norm(t_init):.0f} m apart) it ends
    {no_coarse:.1f} m off. The coarse start is the engineer's job: matching bounding boxes here; in
    practice, picking a few common points or the building's main axes.
    """),
            _fig,
            mo.md(f"""
    **Why it stops.** Each step moves the scan to agree with its current nearest-point pairs, so
    it settles wherever the pairs and the pose agree: a local minimum. The share of points
    paired with another model element than at the reference fit, at the start → at the end:
    {", ".join(f"{_k}° {100 * _s:.0f}% → {100 * _e:.0f}%" for _k, (_s, _e) in wrong_pairs.items())}.
    Up to 20° the wrong pairs all get fixed, although at 20° about half the points start on
    another element; from 30° on, a quarter or more stay wrong. Wrong pairs at the start are
    normal (ICP fixes them step by step); what fails is a start from which the wrong pairs pull
    the pose toward another fit.
    """),
        ]
    )
    return bas, bas_struct, no_coarse, p_bas, t_b, turns, wrong_pairs


@app.cell
def _(
    ALL_ROOMS,
    CHECK_TRAINED,
    EXACT,
    STRUCTURE,
    bas,
    bim,
    bim_tree,
    c,
    cKDTree,
    icp,
    mo,
    motion_error,
    np,
    p_bas,
    pn_pred,
    r_ref,
    ref,
    shown,
    struct,
    t_b,
    t_init,
    t_ref,
    turns,
):
    def pn_pred_on(target, source, pred: np.ndarray) -> np.ndarray:
        """PointNet's labels carried to another thinning: each point takes its nearest's label."""
        _, idx = cKDTree(source.xyz).query(target.xyz, workers=-1)
        return pred[idx]

    # 4. No cap: every point (clutter too), and the labelled structure only, 10-cm voxels
    no_cap = {}
    for _name, _src in (("structure", p_bas), ("all points", bas.xyz)):
        _r, _t, _ = icp(_src, bim_tree, bim["xyz"], np.eye(3), t_b, caps=(np.inf,))
        no_cap[_name] = motion_error(_r, _t, r_ref, t_ref, p_bas)
    # With the cap, 5-cm voxels: every point, and PointNet's structure (labels from rooms it
    # never saw; the slides' setting only, where every room has them)
    capped = {}
    _sets = {"all points": np.ones(len(ref.y), bool)}
    if ALL_ROOMS:
        _sets["PointNet's structure"] = np.isin(pn_pred_on(ref, c, pn_pred), STRUCTURE)
    for _name, _m in _sets.items():
        _r, _t, _ = icp(ref.xyz[_m], bim_tree, bim["xyz"], np.eye(3), t_init)
        capped[_name] = (int(_m.sum()), motion_error(_r, _t, r_ref, t_ref, ref.xyz[struct]))
    # Like for like: the same four set-ups, all on the 10-cm voxels from the same start, the
    # error on the same points
    lfl_cap = {}
    _bas_sets = {"all points": np.ones(len(bas.y), bool)}
    if ALL_ROOMS:
        _bas_sets["PointNet's structure"] = np.isin(pn_pred_on(bas, c, pn_pred), STRUCTURE)
    for _name, _m in _bas_sets.items():
        _r, _t, _ = icp(bas.xyz[_m], bim_tree, bim["xyz"], np.eye(3), t_b)
        lfl_cap[_name] = motion_error(_r, _t, r_ref, t_ref, p_bas)
    print(f"no cap {no_cap}; capped (5 cm) {capped}; capped (10 cm) {lfl_cap}")
    assert abs(no_cap["structure"] - 0.010530464742037856) < EXACT
    assert abs(no_cap["all points"] - 0.23908555617728847) < EXACT
    assert capped["all points"][0] == 653_065
    assert abs(capped["all points"][1] - 0.005662080660366824) < EXACT
    # The table, every row on the 10-cm voxels: 0.24, 0.011, 0.009, 0.003 m (the last in the
    # slides' setting); the notes' 5-cm capped fit, 0.006 m
    assert shown(no_cap["all points"], "0.24") and shown(no_cap["structure"], "0.011")
    assert shown(lfl_cap["all points"], "0.009") and shown(capped["all points"][1], "0.006")
    # On one thinning (10 cm), the two cures are close: every point with the cap 0.0091 m,
    # the labelled structure without it 0.0105 m, a gap about the 0.0012 m by which the two
    # thinnings differ (the start-angle test at 0°); clutter without a cap stays far off
    assert abs(lfl_cap["all points"] - 0.009101514424953642) < EXACT
    assert shown(lfl_cap["all points"], "0.0091") and shown(no_cap["structure"], "0.0105")
    assert shown(turns[0], "0.0012")
    assert abs(no_cap["structure"] - lfl_cap["all points"]) < 2 * turns[0]
    assert max(no_cap["structure"], lfl_cap["all points"]) < no_cap["all points"] / 10
    if CHECK_TRAINED and ALL_ROOMS:
        assert abs(capped["PointNet's structure"][1] - 0.0022830714983727426) < 0.0005
        assert shown(capped["PointNet's structure"][1], "0.002")
        assert abs(lfl_cap["PointNet's structure"] - 0.0033588755471946085) < 0.0005
        assert shown(lfl_cap["PointNet's structure"], "0.003")
        assert lfl_cap["PointNet's structure"] < lfl_cap["all points"]
    _pn = lfl_cap.get("PointNet's structure")
    _pn_row = (
        f"| PointNet's structure points, cap 1 m → 10 cm | 10 cm | {_pn:.3f} |"
        if _pn is not None
        else "| PointNet's structure points, cap 1 m → 10 cm | 10 cm | (slides setting) |"
    )
    _pn5 = capped.get("PointNet's structure")
    _pn5 = "" if _pn5 is None else f", and on PointNet's structure {_pn5[1]:.3f} m"
    mo.md(f"""
    ### Clutter drags plain ICP off

    | ICP from the coarse start | Voxels | From the reference fit, m |
    |:--------------------------|:------:|--------------:|
    | Every point (clutter too), no cap | 10 cm | {no_cap["all points"]:.2f} |
    | Labelled structure points only, no cap | 10 cm | {no_cap["structure"]:.3f} |
    | Every point, cap 1 m → 10 cm | 10 cm | {lfl_cap["all points"]:.3f} |
    {_pn_row}

    Plain ICP pairs every scan point with the model, desks and boxes included, and the fit
    ends {no_cap["all points"]:.2f} m from the reference fit (RMS over the structure points;
    the reference is itself a fit, the capped ICP on the labelled structure, so these are
    distances from it, not errors against the truth). Two cures, each from a lecture: leave the
    clutter out (with the labels:
    {no_cap["structure"]:.3f} m), or ignore pairs farther apart than a shrinking cap, a trimmed
    fit (Lecture 9: {lfl_cap["all points"]:.3f} m). Today's pipeline does both, with PointNet's
    labels from rooms it never saw.

    **Like for like.** Every row uses the same 10-cm voxels and the same start, scored on the
    same points. The reference fit itself comes from the 5-cm voxels: there the capped fit on
    every point lands {capped["all points"][1]:.3f} m from it{_pn5}, small partly by construction,
    since the reference was fitted by the same method on overlapping points. On the 10-cm voxels
    the two cures are close: "structure only, no cap" {no_cap["structure"]:.4f} m against
    "every point, cap" {lfl_cap["all points"]:.4f} m, a gap about the {turns[0]:.4f} m by which
    the two thinnings differ even from the best start. Only clutter without a cap stands apart.
    """)
    return capped, lfl_cap, no_cap


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide: which walls to measure on site?

    A wall's points are the scan points whose nearest model surface is on that wall, within
    0.30 m, of two kinds: its **labelled points**, those the dataset's annotators labelled wall,
    door, or window (hand labels, not PointNet's; ceiling and floor points next to its top and
    foot would measure the ceiling and the floor), and its **unassigned points**, which carry no
    label. Five verdicts, each from at least 100 points:

    - **not seen**: fewer than 100 points of the two kinds together;
    - **seen, unlabelled**: seen, but fewer than 100 labelled points, so not judged: label those
      points first;
    - **within 5 cm** or **off by more**, by the median distance of the labelled points to the
      wall's plane;
    - **conflicting**: the labelled points sit off, while 100 or more unassigned points sit a
      median of 5 cm or less from the plane: two surfaces, and the scan cannot say which is the
      wall.

    Unassigned points farther than 5 cm do not overrule labelled ones: they may be furniture in
    front of the wall; only on the plane do they say that something stands where the model puts
    the wall. Points farther than 0.30 m from every element are not paired.

    Why not the coverage of §4 (the share of the surface within 5 cm of any scan point)? It
    uses the same 5 cm as "off": a wall displaced everywhere by more than 5 cm has almost no
    surface within 5 cm of the scan, and would be filed as unseen, however many points sit on it.
    """)
    return


@app.cell
def _(
    ConvexHull,
    LABEL_GAP,
    LABEL_SHIFT,
    MIN_POINTS,
    MIN_POINTS_SWEEP,
    OFF,
    PALETTE,
    Polygon,
    SEEN,
    SLICE,
    c_full,
    els,
    mo,
    np,
    plt,
    rows,
    scan_model,
    shown,
):
    VERDICTS = ("within", "off", "conflicting", "unlabelled", "hidden")

    def verdict(row: dict, min_points: int = MIN_POINTS) -> str:
        """What the scan says about one wall or column, as the slides' figure script decides.

        Args:
            row: The element's row: ``points`` and ``deviation`` (its labelled points and their
                median distance to its plane, m), ``unassigned_points`` and
                ``unassigned_deviation`` (the same for its unassigned points).
            min_points: The fewest points for each test.

        Returns:
            "hidden" (not seen) under ``min_points`` points of the two kinds together;
            "unlabelled" under ``min_points`` labelled points; "within" when the labelled
            points' median is at most ``OFF``; "conflicting" when it is more but at least
            ``min_points`` unassigned points sit a median of at most ``OFF``; else "off".
        """
        if row["points"] + row["unassigned_points"] < min_points:
            return "hidden"
        if row["points"] < min_points:
            return "unlabelled"
        if row["deviation"] <= OFF:
            return "within"
        _near = row["unassigned_deviation"] is not None and row["unassigned_deviation"] <= OFF
        return "conflicting" if row["unassigned_points"] >= min_points and _near else "off"

    _wall_k = [_k for _k, _w in enumerate(rows) if _w["type"] == "wall"]
    _by = {_v: [_k for _k in _wall_k if verdict(rows[_k]) == _v] for _v in VERDICTS}
    _judged = _by["within"] + _by["off"] + _by["conflicting"]  # 100 or more labelled points
    measure_k = sorted(_by["off"] + _by["conflicting"], key=lambda _k: -rows[_k]["deviation"])
    _low = [_k for _k in _wall_k if rows[_k]["coverage"] < SEEN]  # under 10% within 5 cm
    walls = {
        "n": len(_wall_k),
        **{_v: len(_by[_v]) for _v in VERDICTS},
        "seen": len(_wall_k) - len(_by["hidden"]),
        "judged": len(_judged),
        "measure": len(measure_k),
        "fewest_points_measure": min((rows[_k]["points"] for _k in measure_k), default=None),
        "judged_median_deviation": float(np.median([rows[_k]["deviation"] for _k in _judged])),
        "within_unassigned_beyond": sum(
            rows[_k]["unassigned_points"] >= MIN_POINTS and rows[_k]["unassigned_deviation"] > OFF
            for _k in _by["within"]
        ),
        "low_cover": len(_low),
        "low_cover_judged": sum(_k in _judged for _k in _low),
        "low_cover_measure": sum(_k in measure_k for _k in _low),
        "covered_unlabelled": sum(rows[_k]["coverage"] >= SEEN for _k in _by["unlabelled"]),
        "covered_hidden": sum(rows[_k]["coverage"] >= SEEN for _k in _by["hidden"]),
    }
    # How the verdicts move with the threshold, and which walls go to the tape measure
    by_min_points = {
        _n: {_v: sum(verdict(rows[_k], _n) == _v for _k in _wall_k) for _v in VERDICTS}
        for _n in MIN_POINTS_SWEEP
    }
    _measure_sets = {
        _n: {_k for _k in _wall_k if verdict(rows[_k], _n) in ("off", "conflicting")}
        for _n in MIN_POINTS_SWEEP
    }
    print(f"walls {walls}; by the threshold {by_min_points}")
    for _i, _k in enumerate(measure_k, 1):
        _w = rows[_k]
        print(
            f"  {_i}. {verdict(_w)}: {_w['name']}: labelled {_w['points']:,} at {_w['deviation']:.3f} m, "
            f"unassigned {_w['unassigned_points']:,} at {_w['unassigned_deviation'] or float('nan'):.3f} m, "
            f"{100 * _w['coverage']:.1f}% within 5 cm, {_w['area']:.1f} m2"
        )
    # The slides' run (numbers.json, registration.walls): the counts, the sweep, the walls to
    # measure, worst first
    _walls_ref = {  # copied from numbers.json by fill_refs.py
        "counts": {
            "n": 61,
            "within": 31,
            "off": 7,
            "conflicting": 2,
            "unlabelled": 14,
            "hidden": 7,
            "seen": 54,
            "judged": 40,
            "measure": 9,
            "fewest_points_measure": 1322,
            "within_unassigned_beyond": 20,
            "low_cover": 22,
            "low_cover_judged": 12,
            "low_cover_measure": 6,
            "covered_unlabelled": 10,
            "covered_hidden": 1,
        },
        "judged_median_deviation": 0.03967911938589326,
        "by_min_points": {
            25: {"within": 31, "off": 7, "conflicting": 2, "unlabelled": 14, "hidden": 7},
            50: {"within": 31, "off": 7, "conflicting": 2, "unlabelled": 14, "hidden": 7},
            100: {"within": 31, "off": 7, "conflicting": 2, "unlabelled": 14, "hidden": 7},
            200: {"within": 31, "off": 7, "conflicting": 2, "unlabelled": 14, "hidden": 7},
            400: {"within": 30, "off": 7, "conflicting": 2, "unlabelled": 15, "hidden": 7},
            1000: {"within": 26, "off": 7, "conflicting": 2, "unlabelled": 14, "hidden": 12},
        },
        "measure_list": [
            {
                "name": "Basic Wall:(15cm) Tijolo cerâmico acabamento pintura branca:942215",
                "verdict": "conflicting",
                "deviation": 0.28067992083500204,
                "points": 3270,
                "unassigned_deviation": 0.023405880076849413,
                "unassigned_points": 21407,
            },
            {
                "name": "Basic Wall:(10cm) Drywall:981441",
                "verdict": "off",
                "deviation": 0.196976284198531,
                "points": 6246,
                "unassigned_deviation": 0.19610904254947215,
                "unassigned_points": 1415,
            },
            {
                "name": "Basic Wall:(10cm) Drywall:976607",
                "verdict": "off",
                "deviation": 0.1689967314276823,
                "points": 25572,
                "unassigned_deviation": 0.08961573583393584,
                "unassigned_points": 3261,
            },
            {
                "name": "Basic Wall:(15cm) Tijolo cerâmico acabamento pintura branca:941802",
                "verdict": "off",
                "deviation": 0.1490833580417444,
                "points": 14758,
                "unassigned_deviation": 0.14881748552525317,
                "unassigned_points": 1104,
            },
            {
                "name": "Basic Wall:(10cm) Drywall:976859",
                "verdict": "off",
                "deviation": 0.09382893361778977,
                "points": 17059,
                "unassigned_deviation": 0.17173679826162047,
                "unassigned_points": 1964,
            },
            {
                "name": "Basic Wall:(35cm) Tijolo cerâmico acabamento pintura branca:1534316",
                "verdict": "off",
                "deviation": 0.09321686308444598,
                "points": 1557,
                "unassigned_deviation": 0.15841953387346114,
                "unassigned_points": 8,
            },
            {
                "name": "Basic Wall:(10cm) Drywall:976743",
                "verdict": "off",
                "deviation": 0.05837825821240421,
                "points": 60419,
                "unassigned_deviation": 0.15223008358717305,
                "unassigned_points": 1916,
            },
            {
                "name": "Basic Wall:(10cm) Drywall:981948",
                "verdict": "conflicting",
                "deviation": 0.05832671516727898,
                "points": 1322,
                "unassigned_deviation": 0.016537903415039412,
                "unassigned_points": 10682,
            },
            {
                "name": "Basic Wall:(15cm) Tijolo cerâmico acabamento pintura branca:921993",
                "verdict": "off",
                "deviation": 0.05045766693933532,
                "points": 10031,
                "unassigned_deviation": 0.07708372761136206,
                "unassigned_points": 48,
            },
        ],
    }  # end of the copy
    assert {_k: walls[_k] for _k in _walls_ref["counts"]} == _walls_ref["counts"]
    assert abs(walls["judged_median_deviation"] - _walls_ref["judged_median_deviation"]) < 1e-9
    assert by_min_points == _walls_ref["by_min_points"]
    assert [(rows[_k]["name"], verdict(rows[_k])) for _k in measure_k] == [
        (_w["name"], _w["verdict"]) for _w in _walls_ref["measure_list"]
    ]
    assert np.allclose(
        [(rows[_k]["deviation"], rows[_k]["points"], rows[_k]["unassigned_deviation"],
          rows[_k]["unassigned_points"]) for _k in measure_k],
        [(_w["deviation"], _w["points"], _w["unassigned_deviation"], _w["unassigned_points"])
         for _w in _walls_ref["measure_list"]],
        rtol=0,
        atol=1e-9,
    )  # fmt: skip
    assert sum(walls[_v] for _v in VERDICTS) == walls["n"]
    # The same walls go to the tape measure at every threshold from 25 to 1,000 points
    assert all(_m == set(measure_k) for _m in _measure_sets.values())
    # The slide: 61 walls; 7 not seen, 14 seen through unassigned points only; 40 judged,
    # typical 0.040 m; 9 to measure (7 off, 2 conflicting), each with at least 1,322 labelled
    # points; wall 1 0.28 m labelled against 0.023 m unassigned, wall 8 0.058 against 0.017,
    # wall 9 0.0505 m; 20 of the 31 within walls have unassigned points mostly farther
    _first, _eighth = rows[measure_k[0]], rows[measure_k[7]]
    assert shown(walls["judged_median_deviation"], "0.040")
    assert shown(_first["deviation"], "0.28") and shown(_first["unassigned_deviation"], "0.023")
    assert shown(_eighth["deviation"], "0.058") and shown(_eighth["unassigned_deviation"], "0.017")
    assert shown(rows[measure_k[-1]]["deviation"], "0.0505")
    # Nor do the 10 columns get a verdict: no labelled point is nearest to any, and fewer than
    # 100 unassigned points to each
    assert all(
        verdict(_w) == "hidden" and _w["points"] == 0 for _w in rows if _w["type"] == "column"
    )
    assert sum(_w["type"] == "column" for _w in rows) == 10
    # From above: the model's walls and columns by verdict, over the scan's points
    _h = c_full.feat[:, 2]
    _foot = scan_model[(_h > SLICE[0]) & (_h < SLICE[1])][::10]
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _ax.scatter(*_foot[:, :2].T, s=0.2, c="#e3e3e3", linewidths=0, rasterized=True)
    _color = {"within": "#9a9a9a", "off": PALETTE[1], "conflicting": PALETTE[3],
              "unlabelled": PALETTE[2], "hidden": PALETTE[0]}  # fmt: skip
    _label = {"within": f"within {OFF * 100:.0f} cm", "off": f"off by more than {OFF * 100:.0f} cm",
              "conflicting": "two surfaces: conflicting", "unlabelled": "seen, points unlabelled",
              "hidden": f"not seen: under {MIN_POINTS} points"}  # fmt: skip
    for _e, _w in zip(els, rows, strict=True):
        if _w["type"] in ("wall", "column"):
            _xy = _e.verts[:, :2]
            _v = verdict(_w)
            _ax.add_patch(
                Polygon(_xy[ConvexHull(_xy).vertices], fc=_color[_v], ec=_color[_v], lw=1)
            )
    for _v, _col in _color.items():
        _ax.scatter([], [], marker="s", s=40, c=_col, label=_label[_v])
    _placed = []  # label positions, m: a number too close to another moves down
    for _i, _k in enumerate(measure_k, 1):
        _anchor = els[_k].verts[:, :2].mean(axis=0)
        _pos = _anchor + np.asarray(LABEL_SHIFT)
        while any(np.abs(_pos - _q).max() < LABEL_GAP for _q in _placed):
            _pos = _pos - np.array([0.0, LABEL_GAP])
        _placed.append(_pos)
        _moved = np.abs(_pos - _anchor - np.asarray(LABEL_SHIFT)).max() > 0
        _ink = _color[verdict(rows[_k])]
        _ax.annotate(str(_i), _anchor, xytext=_pos, textcoords="data", color=_ink, fontsize=11,
                     fontweight="bold", va="center",
                     arrowprops={"arrowstyle": "-", "color": _ink, "lw": 0.6} if _moved else None)  # fmt: skip
    _ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    _ax.set(aspect="equal", xticks=[], yticks=[])
    _ax.autoscale_view()
    for _side in _ax.spines.values():
        _side.set_visible(False)
    mo.vstack(
        [
            mo.md(f"""
    ### Measure {walls["measure"]} walls; label {walls["unlabelled"]} more

    Of the model's {walls["n"]} walls, {walls["hidden"]} have fewer than {MIN_POINTS} points of
    either kind: not seen, no verdict; nor do the 10 columns. {walls["unlabelled"]} more are seen
    only through unassigned points: label those points before judging the walls. The
    {walls["judged"]} walls with {MIN_POINTS} or more labelled points are judged by them: the
    typical one sits {walls["judged_median_deviation"]:.3f} m from its labelled points (the median
    over walls of each wall's median distance); {walls["within"]} are within 5 cm, and
    {walls["measure"]} are farther, each with at least {walls["fewest_points_measure"]:,} labelled
    points. {walls["off"]} of those are off. {walls["conflicting"]} rest on two surfaces: wall 1's
    {_first["points"]:,} labelled points sit {_first["deviation"]:.2f} m from the model, but
    {_first["unassigned_points"]:,} unassigned points sit a median {_first["unassigned_deviation"]:.3f} m
    from it; wall 8's {_eighth["points"]:,} sit {_eighth["deviation"]:.3f} m off, its
    {_eighth["unassigned_points"]:,} unassigned points {_eighth["unassigned_deviation"]:.3f} m.
    Unassigned points farther than 5 cm do not overrule labelled ones: {walls["within_unassigned_beyond"]}
    of the {walls["within"]} walls within 5 cm have {MIN_POINTS} or more unassigned points sitting a
    median of more than 5 cm off (furniture in front, or a surface the labels missed). Wall
    {len(measure_k)} sits at {rows[measure_k[-1]]["deviation"]:.4f} m, just over the line. For a
    numbered wall the scan and the model disagree, and the scan cannot say which is wrong: a tape
    measure on site can.

    The verdicts do not lean on the threshold: the same {walls["measure"]} walls go to the tape
    measure at every threshold from {min(MIN_POINTS_SWEEP)} to {max(MIN_POINTS_SWEEP):,} points.
    They do lean on the labels: with the labelled points alone, the {walls["hidden"] + walls["unlabelled"]}
    walls under {MIN_POINTS} labelled points would all be "not seen", and walls 1 and 8 simply
    off. The coverage within 5 cm tells another story: of the {walls["low_cover"]} walls with
    under 10% of their surface within 5 cm of the scan, {walls["low_cover_judged"]} are judged
    and {walls["low_cover_measure"]} of those go to the tape measure; {walls["covered_unlabelled"]}
    of the {walls["unlabelled"]} walls seen only through unassigned points have 10% or more, and
    {walls["covered_hidden"]} of the {walls["hidden"]} unseen ones. Five centimetres is today's
    threshold; a contract's tolerance may be tighter, and the scan's noise and the fit's error
    both count against it.
    """),
            _fig,
        ]
    )
    return measure_k, verdict, walls


@app.cell
def _(MIN_POINTS, mo, results, sparse_res, turns, walls):
    _tr = results["covariance + trees"]
    mo.md(f"""
    | Do not use it alone when | Because, today |
    |:-------------------------|:---------------|
    | The scan did not see the element | {walls["hidden"]} of {walls["n"]} walls (under {MIN_POINTS} points) and every column: no verdict; {walls["unlabelled"]} more seen only through unlabelled points |
    | The new scan is sparser than training | the trees fell {_tr["miou"]:.2f} → {sparse_res["covariance + trees"]["miou"]:.2f} at a tenth of the points |
    | Openings matter | windows at IoU {_tr["iou"]["window"]:.2f} for the trees, the lowest class of every method |
    | ICP starts far from the answer | from a 30° turn it stops {turns[30]:.2f} m off (§5) |

    **Impact (economic, social).** A scan-to-BIM check replaces days of tape measurement, and a
    model that is wrong costs redesign and clashes on site; but its verdicts reach only what the
    scanner saw. Report the walls with no verdict as unknown, not as confirmed: a renovation
    planned on a hidden column assumes it is where the model says, and the people who will work
    and study in the renovated rooms carry that risk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: who labels the next scan?

    - **A.** A technician, by hand, point by point.
    - **B.** The covariance features and trees, trained on this lab.
    - **C.** PointNet, trained on this lab.
    - **D.** A model, then a technician who checks doors and windows.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(mo, results, sparse_res):
    _tr = results["covariance + trees"]
    _pn = results.get("PointNet", {"miou": float("nan")})
    _pc = results.get("PointNet + covariance", {"miou": float("nan"), "iou": {}})
    _sp = sparse_res.get("PointNet", {"miou": float("nan")})
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        A is how today's labels were made (Abreu et al., 2023), and every score here is measured
        against it. B: {_tr["miou"]:.2f} on rooms held out, {sparse_res["covariance + trees"]["miou"]:.2f}
        on a sparser scan. C: {_pn["miou"]:.2f}, and it holds on a sparser scan ({_sp["miou"]:.2f}).
        D: even the best network here (PointNet + covariance, {_pc["miou"]:.2f}) labels windows
        at IoU {_pc["iou"].get("window", float("nan")):.2f} and doors at
        {_pc["iou"].get("door", float("nan")):.2f}, the openings a building model needs. A
        defensible answer: D, the model doing the bulk (ceilings
        {_pc["iou"].get("ceiling", float("nan")):.2f}, floors {_pc["iou"].get("floor", float("nan")):.2f})
        and a person checking the openings, tested first on one hand-labelled room of the new
        building.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where point clouds live

    | Where | What the points are | Lecture |
    |:------|:--------------------|:--------|
    | Scan-to-BIM and progress checks | laser scans of a building, against its model | today |
    | Self-driving cars | lidar returns, sorted into cars, people, road | today |
    | Road and rail mapping | scans from a moving vehicle | today |
    | Phones and tablets with depth sensors | room scans for floor plans | today |

    Today's lab was scanned for the first row (Abreu et al., 2023); comparing scans with the
    model to track what was built goes back at least to Bosché et al. (2014). Self-driving cars
    find objects in lidar point clouds with networks that grew from PointNet (PointPillars, Lang
    et al., 2019). Road and rail operators scan from moving vehicles to map assets and
    clearances. Phones and tablets with depth sensors now draw a room's floor plan from a
    walk-through.

    ## 7. Check yourself

    ### Exit question

    Your firm scans a new building with a cheaper scanner, at a tenth of the density. Which part
    of today's pipeline would you check first, and how?
    """)
    return


@app.cell
def _(mo, results, sparse_res):
    _tr = results["covariance + trees"]
    _pn = results.get("PointNet", {"miou": float("nan")})
    _sp = sparse_res.get("PointNet", {"miou": float("nan")})
    mo.accordion(
        {
            "Answer": mo.md(f"""
        The labelling first: the covariance features depend on point density (the trees fell
        from {_tr["miou"]:.2f} to {sparse_res["covariance + trees"]["miou"]:.2f} at a tenth of
        the points); PointNet did not here ({_pn["miou"]:.2f} to {_sp["miou"]:.2f}). Label one room
        of the new scan by hand and score every model on it before trusting any; compute
        features within a fixed radius, or retrain. Then the comparison: a 5-cm coverage test
        needs points spaced well under 5 cm, so check the new scan's spacing.
        """)
        }
    )
    return


@app.cell
def _(labelling_minutes, mo):
    mo.md(f"""
    ## Gotchas

    - **Sort what a threaded producer returns.** IfcOpenShell's geometry iterator hands back the
      elements as its threads finish; the seeded surface points are drawn element by element,
      so an unsorted list gives a different model cloud every run. Sort by the file's order
      first (`bim_elements`).
    - **One room's points never in another's training.** Columns overlap in training; the test
      room's points must be cut out before the columns are made, not after.
    - **Columns centred, then turned.** x and y are relative to the column's centre before the
      random turn; turning raw coordinates would swing the column about the building's origin.
    - **Label every test point once.** Testing tiles the room (stride = column side); a column's
      points go through in sets of 4,096, the last filled with repeats, which change no maximum.
    - **Score the labelled points only.** "Unassigned" points (`IGNORE`) are left out of the
      loss (`ignore_index`) and of the confusion counts.
    - **Covariance features belong to a density.** Recompute them on the thinned scan, and
      expect them to shift: they are computed over a fixed count of neighbours.
    - **ICP needs a start, and a cap.** Match bounding boxes (or a few common points) first; a
      cap that shrinks keeps clutter and far pairs out of the fit.
    - **Identities in float64.** On the GPU, TF32 convolutions round differently for different
      numbers of points; check permutation and critical points on a float64 copy on the CPU.
    - **`model.eval()` and `torch.no_grad()`** for every prediction: dropout and batch
      normalization behave differently in training.
    - **Time.** The labelling took {labelling_minutes:.0f} minutes in this run.

    ## Going deeper (Forsyth)

    - §4.1.4 and §4.3.2–4.3.3: the covariance matrix and its diagonalization by a rotation onto
      the blob's own axes. The covariance features are exactly this, done on each point's 24
      neighbours: the eigenvalues are the blob's spreads, the last eigenvector its normal.
    - §4.3.1: how the mean and covariance of a dataset move under an affine map. The first step
      of the alignment (the shift that lands one mean on the other) is this rule; the rotation
      then acts on the centred points.
    - §6.1: the SVD. Forsyth uses it for low-rank approximation and PCA; Kabsch's alignment uses
      the SVD of a 3 × 3 cross-covariance instead, and the reflection fix is the only new step.
    - §11.2: M-estimators and iteratively reweighted least squares. ICP's cap is a crude robust
      loss (a weight of 1 inside the cap and 0 outside), and each ICP step is a reweighted fit.
    - §12.2.4–12.2.5, §12.2.8: gradient boosting, with stumps and then trees, and its software;
      the trees here stop early on a held-out tenth of their training points.
    - §17.1.2: pooling in convolutional networks, where a max summarizes a window of pixels.
      PointNet's max summarizes a whole set of points, in any order.

    **Exercises.** (1) Replace PointNet's max with a mean (`.mean(dim=2)`), train on rooms
    108–111, and test on room 175: does the order-free check still hold, and do the critical
    points still exist? (2) Compute the covariance features within a fixed radius (0.1 m)
    instead of over 24 neighbours, and rerun the sparser-scan test for the trees.

    ## Things to remember

    - A scan is a set: use order-free functions.
    - PointNet: a shared MLP, then a max.
    - Neighbourhood features lifted PointNet.
    - ICP: pair, SVD, repeat; start close.
    - No wall points, no verdict.
    - Hold out whole rooms.

    **Next:** physics-informed neural networks.
    """)
    return


if __name__ == "__main__":
    app.run()
