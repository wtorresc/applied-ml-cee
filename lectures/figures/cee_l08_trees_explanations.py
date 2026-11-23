"""Figures and numbers for CEE Lecture 8: trees, forests, boosting, and their explanations.

Run::

    pixi run -e aml-env python figures/cee_l08_trees_explanations.py \
        --out build/fig/cee_l08_trees_explanations

Writes the PNG figures, ``numbers.json`` (the data numbers the slides quote; the deck's constants,
such as regulatory limits and the lecture's settings, are not in it), and ``numbers.tex`` (one
section's prediction and Shapley values, for the TikZ diagram). Data:
``data/derived/ltpp_iri_sections.csv`` from ``scripts/fetch_ltpp.sh`` (FHWA LTPP SDR 40, public):
360 asphalt sections (GPS-1, GPS-2, SPS-1, original construction) with the growth rate of their
roughness (IRI, m/km per year) and their structure, traffic, and climate.

The question: which factors drive how fast a pavement gets rough, and can a model trained on the
national sections tell a state DOT? The lecture's answer is a caution: the trees fit, the
explanations look like engineering, and across held-out states the forest and boosting, each
tuned by an inner cross-validation over the training states, do no better than the mean. The
designed SPS-1 experiment, compared within sites, recovers the effect of asphalt thickness.
The nested cross-validation makes this the slowest step: the script takes about 50 minutes on
4 CPU cores (48 min in the deep-clean queue, 2026-11-21).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.tree import DecisionTreeRegressor

import aml_style
import title_art
from common import (
    DATA_DIR,
    parse_out_dir,
    require,
    rmse,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

CSV: Final[Path] = DATA_DIR / "derived" / "ltpp_iri_sections.csv"
SEED: Final[int] = 0
FOLDS: Final[int] = 5
N_JOBS: Final[int] = 4  # worker processes for the forests: a shared machine (brief, 2026-11-20)
FEATURES: Final[tuple[str, ...]] = (
    "ac_in",
    "granular_base_in",
    "treated_base_in",
    "esal_per_yr",
    "freeze_index",
    "precip_mm",
    "mean_temp_c",
    "iri_first",
    "age_first",
)
NAMES: Final[dict[str, str]] = {
    "ac_in": "asphalt thickness",
    "granular_base_in": "granular base",
    "treated_base_in": "treated base",
    "esal_per_yr": "trucks (ESAL/yr)",
    "freeze_index": "freezing index",
    "precip_mm": "precipitation",
    "mean_temp_c": "mean temperature",
    "iri_first": "first IRI",
    "age_first": "age at first visit",
}
ATTEMPT_SITE: Final[str] = "20-01"  # the SPS-1 project in Kansas
TRACE: Final[tuple[int, str]] = (20, "0105")
BACKGROUND: Final[int] = 100  # background sections for Shapley values
BOOT: Final[int] = 500
TREES: Final[int] = 500  # trees in the lecture's forest
MAX_FEATURES: Final[int] = 3  # inputs tried at each split: p/3 for regression (Breiman, 2001)
FOREST_LEAF: Final[int] = 3  # the lecture's forest: at least 3 sections in every leaf
SMALL_TREE: Final[dict[str, int]] = {"max_depth": 3, "min_samples_leaf": 5}  # the depth-3 tree
# Boosting: shrinkage 0.03, trees of depth 3; scikit-learn's default leaf size, set explicitly.
BOOST_RATE: Final[float] = 0.03
BOOST_DEPTH: Final[int] = 3
BOOST_LEAF: Final[int] = 20
BOOST_ROUNDS: Final[int] = 200  # the untuned setting, kept as a break-it row
# The tuned settings: each ladder runs from the model's no-fit limit (index 0) to
# its most complex end. Boosting with 0 rounds is the training mean itself, and 2,000 rounds is
# far past any useful setting. A forest whose leaves hold at least 160 sections cannot split any
# training set here (at most 288 sections, so at most 320 would be needed), so each tree is one
# leaf; leaves of one section are fully grown trees, the other limit. The search starts on the
# START values and widens one rung at a time while the best value sits at an edge.
BOOST_LADDER: Final[tuple[int, ...]] = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
BOOST_START: Final[tuple[int, ...]] = (10, 20, 50, 100, 200)
LEAF_LADDER: Final[tuple[int, ...]] = (160, 80, 40, 20, 10, 5, 3, 2, 1)  # min_samples_leaf
LEAF_START: Final[tuple[int, ...]] = (40, 20, 10, 5, 3)
INNER_FOLDS: Final[int] = 5
# Other assignments of sites and states to the five folds (GroupKFold shuffled by these seeds),
# for the spread of the tuned models' R^2 (q2 L08-D2: one partition is one draw).
PARTITION_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4)
BAG_TREES: Final[int] = 300  # the terraces figure's bag of depth-3 trees
# The within-site regression: the rate on the three layer thicknesses.
WITHIN_COLS: Final[tuple[str, ...]] = ("iri_rate", "ac_in", "treated_base_in", "granular_base_in")

type Regressor = (
    LinearRegression
    | DecisionTreeRegressor
    | RandomForestRegressor
    | HistGradientBoostingRegressor
    | DummyRegressor
)
type Splits = list[tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------------------------
# Data and models


def load() -> pd.DataFrame:
    """One row per section.

    Returns:
        The 360 sections with the inputs and the IRI growth rate.

    Raises:
        FileNotFoundError: If missing (run ``bash scripts/fetch_ltpp.sh``).
    """
    d = pd.read_csv(require(CSV, "fetch_ltpp.sh"), dtype={"SHRP_ID": str, "EXPERIMENT_NO": str})
    if len(d) != 360 or not d[[*FEATURES, "iri_rate"]].notna().all().all():
        raise ValueError(f"expected 360 complete sections in {CSV}, got {len(d)}")
    return d


def forest(min_leaf: int = FOREST_LEAF) -> RandomForestRegressor:
    """A random forest: 500 bootstrap trees, 3 of the 9 inputs tried at each split.

    Args:
        min_leaf: The fewest sections in a leaf; the lecture's forest uses ``FOREST_LEAF``.

    Returns:
        The unfitted forest.
    """
    return RandomForestRegressor(
        n_estimators=TREES,
        max_features=MAX_FEATURES,
        bootstrap=True,
        min_samples_leaf=min_leaf,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


def boosting(rounds: int = BOOST_ROUNDS) -> HistGradientBoostingRegressor | DummyRegressor:
    """Gradient boosting with shrinkage 0.03 and trees of depth 3.

    Args:
        rounds: Boosting rounds; 0 is the model before any tree, the training mean.

    Returns:
        The unfitted model.
    """
    if rounds == 0:
        return DummyRegressor(strategy="mean")
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=rounds,
        learning_rate=BOOST_RATE,
        max_depth=BOOST_DEPTH,
        min_samples_leaf=BOOST_LEAF,
        early_stopping=False,  # "auto" is off anyway below 10,000 rows (at most 288 here)
        random_state=SEED,
    )


def full_tree() -> DecisionTreeRegressor:
    """A tree grown until every leaf is pure or holds one section.

    Returns:
        The unfitted tree.
    """
    return DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=SEED)


def models() -> dict[str, Regressor]:
    """The lecture's untuned models, at fixed settings.

    Returns:
        The models by name; "mean" predicts the training mean. "random forest" is the lecture's
        forest (leaves of at least 3) and "boosting" runs 200 rounds: the break-it rows.
    """
    return {
        "mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "tree, depth 3": DecisionTreeRegressor(**SMALL_TREE, random_state=SEED),
        "tree, full depth": full_tree(),
        "random forest": forest(),
        "boosting": boosting(),
    }


def oof(make: Callable[[], Regressor], X: np.ndarray, y: np.ndarray, splits: Splits) -> np.ndarray:
    """Out-of-fold predictions of one model.

    Args:
        make: Builds a fresh, unfitted model.
        X: The inputs, shape ``(n, 9)``.
        y: The rates, shape ``(n,)``.
        splits: The (train, test) row indices of each fold; the tests partition the rows.

    Returns:
        The predictions, shape ``(n,)``.
    """
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    if np.isnan(pred).any():
        raise ValueError("the test folds do not cover every row")
    return pred


def cv_rmse(X: np.ndarray, y: np.ndarray, splits: Splits) -> dict[str, float]:
    """Out-of-fold RMSE of every untuned model.

    Args:
        X: The inputs, shape ``(n, 9)``.
        y: The rates, shape ``(n,)``.
        splits: The (train, test) row indices of each fold.

    Returns:
        The RMSE by model name (m/km/yr).
    """
    return {name: rmse(oof(lambda m=m: clone(m), X, y, splits), y) for name, m in models().items()}


def bracket(
    score: Callable[[int], float],
    ladder: Sequence[int],
    start: Sequence[int],
    *,
    last_is_limit: bool = False,
) -> tuple[int, dict[int, float]]:
    """A bracketing search: the setting with the lowest score, never at a grid edge.

    The search scores ``start`` (consecutive rungs of ``ladder``), then, while the best setting
    sits at an edge of the rungs scored so far, scores the next rung past that edge. Ties go to
    the simpler setting (the lower rung).

    Args:
        score: The inner cross-validated MSE of a setting.
        ladder: Settings from the model's no-fit limit (index 0: the training mean, or trees that
            cannot split) to a setting far past any useful one.
        start: The first settings scored, consecutive rungs of ``ladder``.
        last_is_limit: True when the last rung is also the setting's own limit, with nothing
            beyond it (a forest's leaves of one section: fully grown trees).

    Returns:
        The kept setting, and the score of every setting scored.

    Raises:
        ValueError: If the kept setting is the ladder's last rung and that rung is not a limit
            (the grid did not bracket the best setting), or ``start`` is not consecutive rungs
            of ``ladder``. The first rung, the no-fit limit, has nothing beyond it: keeping it
            means that no amount of fitting beat the mean on the inner folds, and it is
            reported as such.
    """
    lo, hi = ladder.index(start[0]), ladder.index(start[-1])
    if list(ladder[lo : hi + 1]) != list(start):
        raise ValueError(f"start {start} is not consecutive rungs of {ladder}")
    scores = {v: score(v) for v in start}
    while True:
        best = min(ladder[lo : hi + 1], key=lambda v: scores[v])  # first minimum: the simpler
        if best == ladder[lo] and lo > 0:
            lo -= 1
            scores[ladder[lo]] = score(ladder[lo])
        elif best == ladder[hi] and hi < len(ladder) - 1:
            hi += 1
            scores[ladder[hi]] = score(ladder[hi])
        else:
            break
    if best == ladder[-1] and not last_is_limit:
        raise ValueError(f"best setting {best} is the ladder's last rung: {scores}")
    return best, scores


def inner_splits(scheme: str, X: np.ndarray, groups: pd.DataFrame) -> Splits:
    """The inner folds within one outer training set, by the outer scheme's unit.

    Args:
        scheme: "random sections", "sites held out", or "states held out".
        X: The outer training inputs.
        groups: The outer training rows' ``site`` and ``STATE_CODE``.

    Returns:
        Five folds: random sections, or whole sites, or whole states (the training states only).
    """
    if scheme == "random sections":
        return list(KFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(X))
    unit = "site" if scheme == "sites held out" else "STATE_CODE"
    return list(GroupKFold(INNER_FOLDS).split(X, groups=groups[unit].to_numpy()))


def nested(
    scheme: str, model: str, d: pd.DataFrame, splits: Splits
) -> tuple[np.ndarray, list[int]]:
    """Nested cross-validation: each outer fold tunes the setting on its training rows only.

    Args:
        scheme: The outer scheme, which sets the inner folds' unit (:func:`inner_splits`).
        model: A key of ``TUNED``: the model builder, its ladder, and the first settings scored
            (:func:`bracket`).
        d: The sections.
        splits: The outer (train, test) row indices.

    Returns:
        The out-of-fold predictions, shape ``(n,)``, and the setting each outer fold kept.
    """
    make, ladder, start, last_is_limit = TUNED[model]
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    pred, kept = np.full(len(y), np.nan), []
    for tr, te in splits:
        Xt, yt = X[tr], y[tr]
        inner = inner_splits(scheme, Xt, d.iloc[tr])

        def score(v: int, Xt: np.ndarray = Xt, yt: np.ndarray = yt, inner: Splits = inner) -> float:
            return float(np.mean((oof(lambda: make(v), Xt, yt, inner) - yt) ** 2))

        best, _ = bracket(score, ladder, start, last_is_limit=last_is_limit)
        kept.append(int(best))
        pred[te] = make(best).fit(Xt, yt).predict(X[te])
    return pred, kept


def outer_splits(scheme: str, d: pd.DataFrame, seed: int | None = None) -> Splits:
    """The five outer folds of one scheme.

    Args:
        scheme: "random sections", "sites held out", or "states held out".
        d: The sections.
        seed: ``None`` for the lecture's folds (scikit-learn's ``GroupKFold`` without shuffling);
            otherwise the groups are shuffled into folds with this seed.

    Returns:
        The (train, test) row indices of each fold.
    """
    X = d[list(FEATURES)].to_numpy()
    if scheme == "random sections":
        return list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X))
    unit = d["site" if scheme == "sites held out" else "STATE_CODE"].to_numpy()
    kw = {} if seed is None else {"shuffle": True, "random_state": seed}
    return list(GroupKFold(FOLDS, **kw).split(X, groups=unit))


# Each tuned model: its builder, ladder, first settings, and whether the last rung is a limit.
# A forest's leaf size runs from no split (160) to fully grown trees (1): both ends are limits,
# so the whole range can be searched. Boosting's 2,000 rounds is only the grid's end.
TUNED: Final[
    dict[str, tuple[Callable[[int], Regressor], tuple[int, ...], tuple[int, ...], bool]]
] = {
    "random forest, tuned": (forest, LEAF_LADDER, LEAF_START, True),
    "boosting, tuned": (boosting, BOOST_LADDER, BOOST_START, False),
}


def best_split(X: np.ndarray, y: np.ndarray, min_leaf: int = 1) -> tuple[int, float, float]:
    """The split that most reduces the sum of squared errors (the slide's listing).

    Args:
        X: The inputs, shape ``(n, p)``.
        y: The targets, shape ``(n,)``.
        min_leaf: The fewest cases on each side.

    Returns:
        Feature index, threshold, and the reduction in SSE (feature -1 if no split helps).
    """
    best, n = (-1, 0.0, 0.0), len(y)
    sse0 = ((y - y.mean()) ** 2).sum()
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        xs, s, s2 = X[o, j], np.cumsum(y[o]), np.cumsum(y[o] ** 2)
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue  # no threshold between equals
            sse_l = s2[i - 1] - s[i - 1] ** 2 / i
            sse_r = s2[-1] - s2[i - 1] - (s[-1] - s[i - 1]) ** 2 / (n - i)
            gain = sse0 - sse_l - sse_r
            if gain > best[2] + 1e-12:
                best = (j, (xs[i - 1] + xs[i]) / 2, gain)
    return best


def grow(X: np.ndarray, y: np.ndarray, depth: int, min_leaf: int = 1) -> dict[str, object]:
    """A regression tree grown greedily to ``depth`` (CART).

    Args:
        X: The inputs, shape ``(n, p)``.
        y: The targets, shape ``(n,)``.
        depth: The depth left to grow.
        min_leaf: The fewest cases in a leaf.

    Returns:
        The tree as nested dicts: a leaf holds its mean (``leaf``) and size (``n``); a node holds
        ``feature``, ``threshold``, ``n``, and its ``left`` and ``right`` subtrees.
    """
    j, t, gain = best_split(X, y, min_leaf)
    if depth == 0 or j < 0 or gain <= 0:
        return {"leaf": float(y.mean()), "n": len(y)}
    left = X[:, j] <= t
    return {
        "feature": j,
        "threshold": t,
        "n": len(y),
        "left": grow(X[left], y[left], depth - 1, min_leaf),
        "right": grow(X[~left], y[~left], depth - 1, min_leaf),
    }


def predict_tree(node: dict[str, object], X: np.ndarray) -> np.ndarray:
    """Predictions of a tree from :func:`grow`.

    Args:
        node: The tree (or subtree).
        X: The inputs, shape ``(n, p)``.

    Returns:
        The predictions, shape ``(n,)``.
    """
    if "leaf" in node:
        return np.full(len(X), node["leaf"], dtype=float)
    go_left = X[:, node["feature"]] <= node["threshold"]  # type: ignore[index]
    out = np.empty(len(X))
    out[go_left] = predict_tree(node["left"], X[go_left])  # type: ignore[arg-type]
    out[~go_left] = predict_tree(node["right"], X[~go_left])  # type: ignore[arg-type]
    return out


def shapley(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray
) -> tuple[float, np.ndarray]:
    """Exact interventional Shapley values of ``f`` at ``x`` over all feature subsets.

    v(S) = mean over the background of f(x_S, b_rest). phi_j averages v(S + j) - v(S) with
    Shapley weights |S|! (d - |S| - 1)! / d!.

    Args:
        f: The model's prediction function, rows to predictions.
        x: The explained input, shape ``(d,)``.
        background: Rows that supply the inputs outside S, shape ``(m, d)``.

    Returns:
        v(empty set), the baseline, and the d Shapley values.
    """
    d = len(x)
    rows, keys = [], []
    for k in range(d + 1):
        for S in itertools.combinations(range(d), k):
            Z = background.copy()
            Z[:, list(S)] = x[list(S)]
            rows.append(Z)
            keys.append(S)
    preds = f(np.vstack(rows)).reshape(len(keys), len(background)).mean(axis=1)
    value = dict(zip(keys, preds, strict=True))
    phi = np.zeros(d)
    for j in range(d):
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for S in itertools.combinations([i for i in range(d) if i != j], k):
                phi[j] += w * (value[tuple(sorted((*S, j)))] - value[S])
    return float(value[()]), phi


# ---------------------------------------------------------------------------------------------
# Experiments


def summary(d: pd.DataFrame) -> dict[str, object]:
    """Counts for the question slide.

    Args:
        d: The sections.

    Returns:
        Sections, states, sites, sections by experiment, visits, and the rate's mean, SD, median.
    """
    return {
        "sections": int(len(d)),
        "states": int(d["STATE_CODE"].nunique()),
        "sites": int(d["site"].nunique()),
        "by_experiment": {
            f"{g}-{e}": int(n)
            for (g, e), n in d.groupby(["GPS_SPS", "EXPERIMENT_NO"]).size().items()
        },
        "visits": int(d["visits"].sum()),
        "rate_mean": rnd(d["iri_rate"].mean(), 4),
        "rate_sd": rnd(d["iri_rate"].std(), 4),
        "rate_median": rnd(d["iri_rate"].median(), 4),
    }


def attempt(d: pd.DataFrame) -> dict[str, object]:
    """The Kansas SPS-1 project: its eight sections, by asphalt thickness.

    Args:
        d: The sections.

    Returns:
        The site, one row per section (thicknesses to 0.1 in, rate to 3 decimals), the site's
        freezing index, and its median trucks per year to the thousand.
    """
    s = d[d["site"] == ATTEMPT_SITE].sort_values(["ac_in", "SHRP_ID"])
    rows = [
        {
            "id": r.SHRP_ID,
            "ac_in": rnd(r.ac_in, 1),
            "treated_base_in": rnd(r.treated_base_in, 1),
            "granular_base_in": rnd(r.granular_base_in, 1),
            "rate": rnd(r.iri_rate, 3),
        }
        for r in s.itertuples()
    ]
    return {
        "site": ATTEMPT_SITE,
        "rows": rows,
        "freeze_index": round(float(s["freeze_index"].iloc[0])),
        "esal_per_yr": int(round(float(s["esal_per_yr"].median()), -3)),
    }


SCHEMES: Final[tuple[str, ...]] = ("random sections", "sites held out", "states held out")
HELD_OUT: Final[tuple[str, ...]] = SCHEMES[1:]  # the schemes that hold out whole units


def r2_against_mean(pred: np.ndarray, base: np.ndarray, y: np.ndarray) -> float:
    """R^2 against the training-mean predictions of the same folds.

    Args:
        pred: A model's out-of-fold predictions.
        base: The mean model's out-of-fold predictions on the same folds.
        y: The rates.

    Returns:
        1 - MSE(model) / MSE(mean).
    """
    return float(1 - np.mean((pred - y) ** 2) / np.mean((base - y) ** 2))


def folds(d: pd.DataFrame, out_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    """Three ways to hold out sections, with the forest and boosting tuned inside each fold.

    The tuned models choose their setting by an inner five-fold cross-validation on each outer
    fold's training rows, grouped like the outer folds (random sections, sites, or states; with
    states held out, the inner folds hold out training states). The untuned models (the
    lecture's forest, 200 boosting rounds, both trees) stay as break-it rows. With sites or
    states held out, the tuned models are also run on four other assignments of the groups to
    the folds (``PARTITION_SEEDS``), for the spread one partition hides.

    Args:
        d: The sections.
        out_dir: The figure directory.

    Returns:
        The numbers (R^2 to 4 decimals), and the unrounded R^2 by scheme and model, for the
        figures (rounding twice would print -1.665 as -1.67).

    Raises:
        ValueError: If the forest's no-split rung could split some training set, or a search
            ends at its ladder's last rung (:func:`bracket`).
    """
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    schemes = {k: outer_splits(k, d) for k in SCHEMES}
    most = max(len(tr) for s in schemes.values() for tr, _ in s)
    if 2 * LEAF_LADDER[0] <= most:  # the no-split rung must not split any training set
        raise ValueError(f"leaves of {LEAF_LADDER[0]} could split {most} training sections")
    res = {k: cv_rmse(X, y, s) for k, s in schemes.items()}
    r2 = {k: {m: 1 - v**2 / r["mean"] ** 2 for m, v in r.items()} for k, r in res.items()}
    kept: dict[str, dict[str, list[int]]] = {k: {} for k in SCHEMES}
    for k, splits in schemes.items():
        base = oof(lambda: DummyRegressor(strategy="mean"), X, y, splits)
        for name in TUNED:
            pred, kept[k][name] = nested(k, name, d, splits)
            res[k][name] = rmse(pred, y)
            r2[k][name] = r2_against_mean(pred, base, y)
    # The spread over partitions: the lecture's folds first, then each shuffled assignment.
    spread: dict[str, dict[str, list[float]]] = {k: {} for k in HELD_OUT}
    for k in HELD_OUT:
        spread[k] = {m: [r2[k][m]] for m in ("linear", *TUNED)}
        for seed in PARTITION_SEEDS:
            splits = outer_splits(k, d, seed)
            base = oof(lambda: DummyRegressor(strategy="mean"), X, y, splits)
            spread[k]["linear"].append(
                r2_against_mean(oof(LinearRegression, X, y, splits), base, y)
            )
            for name in TUNED:
                pred, _ = nested(k, name, d, splits)
                spread[k][name].append(r2_against_mean(pred, base, y))
    folds_figure(r2, spread, out_dir)
    best_held = {
        k: max((m for m in r2[k] if m != "mean"), key=lambda m, k=k: r2[k][m]) for k in HELD_OUT
    }
    numbers = {
        "rmse": {k: {m: rnd(v, 4) for m, v in r.items()} for k, r in res.items()},
        "r2": {k: {m: rnd(v, 4) for m, v in r.items()} for k, r in r2.items()},
        "kept": kept,
        "kept_no_fit": {  # outer folds whose inner search kept the ladder's no-fit rung
            k: {m: sum(v == TUNED[m][1][0] for v in kept[k][m]) for m in TUNED} for k in SCHEMES
        },
        "kept_full_grown": {  # outer folds that kept the forest's fully grown trees
            k: sum(v == LEAF_LADDER[-1] for v in kept[k]["random forest, tuned"]) for k in SCHEMES
        },
        "ladders": {m: {"ladder": list(t[1]), "start": list(t[2])} for m, t in TUNED.items()},
        "best_held_out": {k: {"model": m, "r2": rnd(r2[k][m], 4)} for k, m in best_held.items()},
        "partitions": {
            k: {
                m: {
                    "r2": [rnd(v, 4) for v in vals],
                    "min": rnd(min(vals), 4),
                    "max": rnd(max(vals), 4),
                }
                for m, vals in spread[k].items()
            }
            for k in HELD_OUT
        },
        "partition_seeds": list(PARTITION_SEEDS),
        "train_rows": sorted({len(tr) for s in schemes.values() for tr, _ in s}),
    }
    return numbers, r2


def folds_figure(
    r2: dict[str, dict[str, float]], spread: dict[str, dict[str, list[float]]], out_dir: Path
) -> None:
    """R^2 against the mean by scheme: the tuned models, linear, and untuned boosting.

    Args:
        r2: Unrounded R^2 by scheme and model.
        spread: The tuned models' and linear's R^2 over partitions, for the held-out schemes.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    series = (
        ("random forest, tuned", "forest, tuned"),
        ("boosting, tuned", "boosting, tuned"),
        ("linear", "linear"),
        ("boosting", f"boosting, {BOOST_ROUNDS} rounds"),
    )
    for i, (m, label) in enumerate(series):
        xs = np.arange(len(SCHEMES)) + (i - 1.5) * 0.06
        ax.plot(
            xs,
            [r2[k][m] for k in SCHEMES],
            aml_style.MARKERS[i] + aml_style.LINESTYLES[i],
            color=aml_style.PALETTE[i],
            markersize=8,
            label=label,
        )
        for j, k in enumerate(SCHEMES):
            if k in spread and m in spread[k]:  # the range over partitions
                ax.plot([xs[j]] * 2, [min(spread[k][m]), max(spread[k][m])],
                        color=aml_style.PALETTE[i], lw=1.6, alpha=0.7)  # fmt: skip
    ax.axhline(0, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.text(2.12, 0.012, "the mean", color=aml_style.MUTED, fontsize=12)
    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels(["random\nsections", "sites\nheld out", "states\nheld out"])
    ax.set_ylabel("R² against the mean")
    ax.set_xlim(-0.3, 2.7)
    ax.legend(loc="lower left", fontsize=10)
    aml_style.save(fig, out_dir, "folds")


def noise_floor(d: pd.DataFrame) -> dict[str, object]:
    """A rough floor: the spread of rates among SPS-1 sections built alike at one site.

    "Alike" means the same asphalt, treated-base, and granular-base thickness when each is rounded
    to the inch, so a group can mix thicknesses up to an inch apart.

    Args:
        d: The sections.

    Returns:
        The number of such groups and the root mean square of their standard deviations.
    """
    s = d[d["GPS_SPS"] == "S"].copy()
    s["design"] = (
        s["ac_in"].round(0).astype(str)
        + "/"
        + s["treated_base_in"].round(0).astype(str)
        + "/"
        + s["granular_base_in"].round(0).astype(str)
    )
    g = s.groupby(["site", "design"])["iri_rate"]
    pairs = g.filter(lambda v: len(v) >= 2)
    sd = s.loc[pairs.index].groupby(["site", "design"])["iri_rate"].std()
    return {"twin_groups": int(len(sd)), "rms_sd": rnd(np.sqrt(np.mean(sd**2)), 4)}


def importance(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Permutation importance of the lecture's forest on training and held-out states.

    Args:
        d: The sections.
        out_dir: The figure directory.

    Returns:
        Importance by input on the training and held-out states (averaged over the five folds),
        the trucks' held-out importance by fold, and two correlations the notes quote.
    """
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    splits = outer_splits("states held out", d)
    tr_fold, te_fold = [], []
    for tr, te in splits:
        rf = forest().fit(X[tr], y[tr])
        tr_fold.append(
            permutation_importance(
                rf, X[tr], y[tr], n_repeats=10, random_state=SEED
            ).importances_mean
        )
        te_fold.append(
            permutation_importance(
                rf, X[te], y[te], n_repeats=10, random_state=SEED
            ).importances_mean
        )
    tr_imp, te_imp = np.mean(tr_fold, axis=0), np.mean(te_fold, axis=0)
    order = np.argsort(-tr_imp)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    yy = np.arange(len(FEATURES))
    ax.plot(
        tr_imp[order], yy, "o", color=aml_style.PALETTE[0], markersize=9, label="training states"
    )
    ax.plot(
        te_imp[order], yy, "s", color=aml_style.PALETTE[1], markersize=9, label="held-out states"
    )
    for i, k in enumerate(order):
        ax.plot([te_imp[k], tr_imp[k]], [i, i], color="#D9D9D9", lw=1.5, zorder=0)
    ax.axvline(0, color=aml_style.MUTED, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([NAMES[FEATURES[k]] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("permutation importance (drop in R²)")
    ax.legend(loc="lower right", fontsize=11)
    aml_style.save(fig, out_dir, "importance")
    trucks = np.array([f[FEATURES.index("esal_per_yr")] for f in te_fold])
    worst = int(np.argmin(trucks))
    return {
        "train": {NAMES[f]: rnd(v, 3) for f, v in zip(FEATURES, tr_imp, strict=True)},
        "held_out": {NAMES[f]: rnd(v, 3) for f, v in zip(FEATURES, te_imp, strict=True)},
        "train_top3": [NAMES[FEATURES[k]] for k in order[:3]],
        # Trucks on held-out states, fold by fold: one fold carries most of the mean.
        "trucks_held_out_by_fold": [rnd(v, 3) for v in trucks],
        "trucks_held_out_worst_fold": worst,
        "trucks_held_out_without_worst": rnd(np.delete(trucks, worst).mean(), 3),
        # Correlated inputs: shuffling one alone makes impossible climates (the notes).
        "corr_freeze_temp": rnd(np.corrcoef(d["freeze_index"], d["mean_temp_c"])[0, 1], 2),
        # Sections that start rougher roughen faster: what first IRI's held-out importance uses.
        "corr_first_iri_rate": rnd(np.corrcoef(d["iri_first"], d["iri_rate"])[0, 1], 2),
    }


def demean(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rates and thicknesses as deviations from their group's mean (group fixed effects).

    Args:
        frame: The sections.
        by: The column that names each section's group.

    Returns:
        The ``WITHIN_COLS`` columns, demeaned within groups.
    """
    return frame[list(WITHIN_COLS)] - frame.groupby(by)[list(WITHIN_COLS)].transform("mean")


def within(frame: pd.DataFrame, by: str) -> np.ndarray:
    """Within-group least squares: per inch of asphalt, treated base, and granular base.

    Args:
        frame: The sections.
        by: The column that names each section's group.

    Returns:
        The three slopes (m/km/yr per inch), shape ``(3,)``.
    """
    dm = demean(frame, by)
    X = dm[list(WITHIN_COLS[1:])].to_numpy()
    return np.linalg.lstsq(X, dm["iri_rate"].to_numpy(), rcond=None)[0]


def cr1_se(frame: pd.DataFrame, by: str, b: np.ndarray) -> np.ndarray:
    """Cluster-robust (CR1) standard errors of :func:`within`'s estimates, groups as clusters.

    The sandwich (X'X)^-1 (sum_g X_g' e_g e_g' X_g) (X'X)^-1 on the demeaned data, scaled by
    G/(G - 1) (n - 1)/(n - k), where k counts the slopes and the G absorbed group means: Stata's
    ``areg`` convention. With the group effects nested in the clusters, ``xtreg, fe`` counts
    only the slopes (Cameron & Miller, 2015, discuss both); here that would shrink the factor
    sqrt((n - 1)/(n - k)) from 1.054 to 1.006 (n = 172 sections). We keep the conservative count.

    Args:
        frame: The sections.
        by: The column that names each section's group (cluster).
        b: The within estimates, shape ``(3,)``.

    Returns:
        The standard errors, shape ``(3,)``.
    """
    dm = demean(frame, by)
    X = dm[list(WITHIN_COLS[1:])].to_numpy()
    e = dm["iri_rate"].to_numpy() - X @ b
    groups = frame[by].to_numpy()
    sites = np.unique(groups)
    scores = [X[groups == g].T @ e[groups == g] for g in sites]
    meat = sum(np.outer(u, u) for u in scores)
    xtx = X.T @ X
    V = np.linalg.solve(xtx, np.linalg.solve(xtx, meat).T)  # (X'X)^-1 meat (X'X)^-1
    G, n, k = len(sites), len(frame), X.shape[1] + len(sites)
    return np.sqrt(np.diag(V) * G / (G - 1) * (n - 1) / (n - k))


def site_bootstrap(s: pd.DataFrame) -> np.ndarray:
    """The asphalt estimate over ``BOOT`` resamples of whole sites.

    Args:
        s: The SPS-1 sections of the designed comparison.

    Returns:
        The within-site asphalt slope of each resample, shape ``(BOOT,)``.
    """
    rng = np.random.default_rng(SEED)
    sites = s["site"].unique()
    boot = []
    for _ in range(BOOT):
        pick = rng.choice(sites, len(sites))
        frame = pd.concat([s[s["site"] == p].assign(bsite=i) for i, p in enumerate(pick)])
        boot.append(within(frame, "bsite")[0])
    return np.array(boot)


def designed(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Asphalt thickness: across all sections against within SPS-1 sites (fixed effects).

    With 15 sites, the slide's 90% interval is cluster-robust (CR1) with t on G - 1 degrees of
    freedom; the percentile bootstrap of whole sites undercovers with so few clusters (Cameron,
    Gelbach & Miller, 2008), and is kept for the notes with the leave-one-site-out range. The
    treated base's CR1 interval supports the notes' "thicker treated bases slow roughening".
    The regression leaves out drainage, the SPS-1 factorial's other factor, which the data here
    do not carry. Two checks for the notes: how strongly thickness tracks traffic across
    sections, and the asphalt slope once all nine inputs enter a linear regression.

    Args:
        d: The sections.
        out_dir: The figure directory.

    Returns:
        The across-section and within-site slopes, their intervals, and the two checks.
    """
    obs = np.polyfit(d["ac_in"], d["iri_rate"], 1)[0]
    adjusted = LinearRegression().fit(d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy())
    s = (
        d[d["GPS_SPS"] == "S"]
        .groupby("site")
        .filter(lambda g: len(g) >= 4 and g["ac_in"].nunique() > 1)
    )
    sites = s["site"].unique()
    b = within(s, "site")
    se = cr1_se(s, "site", b)
    q = float(student_t.ppf(0.95, len(sites) - 1))
    lo, hi = b[0] - q * se[0], b[0] + q * se[0]
    lo_t, hi_t = b[1] - q * se[1], b[1] + q * se[1]  # treated base
    boot_lo, boot_hi = np.quantile(site_bootstrap(s), [0.05, 0.95])
    loso = [within(s[s["site"] != g], "site")[0] for g in sites]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.errorbar(
        [0],
        [b[0]],
        yerr=[[b[0] - lo], [hi - b[0]]],
        fmt="s",
        color=aml_style.PALETTE[1],
        markersize=10,
        capsize=6,
        label="within SPS-1 sites, 90% interval (CR1)",
    )
    ax.plot([1], [obs], "o", color=aml_style.PALETTE[0], markersize=10, label="across all sections")
    ax.axhline(0, color=aml_style.MUTED, lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"designed\n{len(sites)} sites", f"observational\n{len(d)} sections"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("change in IRI rate per inch\nof asphalt (m/km/yr)")
    ax.legend(loc="lower right", fontsize=10)
    aml_style.save(fig, out_dir, "designed")
    return {
        "observational_per_in": rnd(obs, 4),
        "corr_obs": rnd(np.corrcoef(d["ac_in"], d["iri_rate"])[0, 1], 3),
        "corr_asphalt_trucks": rnd(np.corrcoef(d["ac_in"], d["esal_per_yr"])[0, 1], 2),
        "adjusted_per_in": rnd(adjusted.coef_[FEATURES.index("ac_in")], 5),
        "within_per_in": rnd(b[0], 4),
        "within_treated_per_in": rnd(b[1], 4),
        "se_cr1": rnd(se[0], 4),
        "se_cr1_treated": rnd(se[1], 4),
        "t_df": len(sites) - 1,
        "ci90_cr1": [rnd(lo, 4), rnd(hi, 4)],
        "ci90_cr1_treated": [rnd(lo_t, 4), rnd(hi_t, 4)],
        "ci90_boot": [rnd(boot_lo, 4), rnd(boot_hi, 4)],
        "loso_range": [rnd(min(loso), 4), rnd(max(loso), 4)],
        "sites": len(sites),
        "sections": int(len(s)),
    }


def verify(d: pd.DataFrame) -> dict[str, object]:
    """CART from scratch against scikit-learn at depth 3 (:func:`trace` checks Shapley efficiency).

    Args:
        d: The sections.

    Returns:
        The largest prediction difference, the root split and the sections on its far side, the
        leaves, and the training R^2.

    Raises:
        ValueError: If the two trees' predictions differ by 1e-12 or more.
    """
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    ours = grow(X, y, depth=SMALL_TREE["max_depth"], min_leaf=SMALL_TREE["min_samples_leaf"])
    ref = DecisionTreeRegressor(**SMALL_TREE, random_state=SEED).fit(X, y)
    diff = float(np.max(np.abs(predict_tree(ours, X) - ref.predict(X))))
    if diff >= 1e-12:
        raise ValueError(f"CART from scratch differs from scikit-learn by {diff}")
    j, t = int(ours["feature"]), float(ours["threshold"])  # type: ignore[arg-type]
    return {
        "max_abs_diff": float(f"{diff:.0e}") if diff > 0 else 0.0,
        "root_feature": NAMES[FEATURES[j]],
        "root_threshold": round(t),
        "root_far_side": int((X[:, j] > t).sum()),  # sections above the first split
        "leaves": int(ref.get_n_leaves()),
        "train_r2_depth3": rnd(ref.score(X, y), 4),
    }


def overfit(d: pd.DataFrame) -> tuple[dict[str, object], dict[str, float]]:
    """Training fit of a full tree, the depth-3 tree, and the forest.

    Args:
        d: The sections.

    Returns:
        The numbers, and the unrounded training R^2 by model, for the figure.
    """
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    full = full_tree().fit(X, y)
    train = {
        "tree, full depth": float(full.score(X, y)),
        "tree, depth 3": float(
            DecisionTreeRegressor(**SMALL_TREE, random_state=SEED).fit(X, y).score(X, y)
        ),
        "random forest": float(forest().fit(X, y).score(X, y)),
    }
    numbers = {
        "full_tree_train_r2": rnd(train["tree, full depth"], 4),
        "forest_train_r2": rnd(train["random forest"], 4),
        "full_tree_leaves": int(full.get_n_leaves()),
    }
    return numbers, train


def trace(d: pd.DataFrame) -> dict[str, object]:
    """The forest's prediction for one section and its exact Shapley values.

    Args:
        d: The sections.

    Returns:
        The rounded trace for ``numbers.json`` and, under ``_raw``, the unrounded values for the
        figures and the TikZ macros.

    Raises:
        ValueError: If the Shapley values do not add up to the prediction (efficiency).
    """
    X, y = d[list(FEATURES)].to_numpy(), d["iri_rate"].to_numpy()
    i = int(np.flatnonzero((d["STATE_CODE"] == TRACE[0]) & (d["SHRP_ID"] == TRACE[1]))[0])
    tr = np.flatnonzero(d["STATE_CODE"] != TRACE[0])  # Kansas held out
    rf = forest().fit(X[tr], y[tr])
    bg = X[np.random.default_rng(SEED).choice(tr, BACKGROUND, replace=False)]
    base, phi = shapley(rf.predict, X[i], bg)
    pred = float(rf.predict(X[i : i + 1])[0])
    if abs(base + phi.sum() - pred) >= 1e-10:
        raise ValueError(f"Shapley values add to {base + phi.sum()}, the forest says {pred}")
    order = np.argsort(-np.abs(phi))
    top = [(NAMES[FEATURES[k]], float(X[i, k]), float(phi[k])) for k in order[:3]]
    rest = float(phi[order[3:]].sum())
    raw = {
        "phi": phi,
        "actual": float(y[i]),
        "pred": pred,
        "base": base,
        "top": top,
        "rest": rest,
        "x": dict(zip(FEATURES, X[i].astype(float), strict=True)),
    }
    return {
        "section": f"{TRACE[0]}-{TRACE[1]}",
        "actual": rnd(y[i], 4),
        "pred": rnd(pred, 4),
        "base": rnd(base, 4),
        "top": [{"input": n, "value": rnd(v, 1), "phi": rnd(p, 4)} for n, v, p in top],
        "rest": rnd(rest, 4),
        "inputs": {NAMES[f]: rnd(v, 1) for f, v in zip(FEATURES, X[i], strict=True)},
        "train_sections": int(len(tr)),
        "train_states": int(d["STATE_CODE"].iloc[tr].nunique()),
        "_raw": raw,
    }


def write_tex(tr: dict[str, object], out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, from the unrounded values.

    The section and state counts are those of the traced forest's training set (Kansas held
    out), the ones the diagram says train each tree.

    Args:
        tr: The trace from :func:`trace`.
        out_dir: The figure directory.

    Returns:
        The path of ``numbers.tex``.
    """
    raw = tr["_raw"]
    if not isinstance(raw, dict):
        raise TypeError("the trace has no unrounded values")
    macros = {
        "NSec": tex_num(tr["train_sections"]),
        "NStates": tex_num(tr["train_states"]),
        "Actual": tex_num(raw["actual"], ".3f"),
        "Pred": tex_num(raw["pred"], ".3f"),
        "Base": tex_num(raw["base"], ".3f"),
        "Rest": tex_num(raw["rest"], "+.3f"),
    }
    for k, (name, _val, phi) in zip("ABC", raw["top"], strict=True):
        macros |= {f"SName{k}": name, f"SPhi{k}": tex_num(phi, "+.3f")}
    x = raw["x"]
    macros |= {
        "NMore": tex_num(len(FEATURES) - 3),
        "XAsphalt": tex_num(x["ac_in"], ".1f"),
        "XAge": tex_num(x["age_first"], ".1f"),
        "XIri": tex_num(x["iri_first"], ".1f"),
        "XEsal": tex_num(round(x["esal_per_yr"], -3), ",.0f"),
    }  # to the thousand
    return write_tex_macros(macros, out_dir, "cee_l08_trees_explanations.py")


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

LTPP_TABLES: Final[Path] = DATA_DIR / "derived" / "ltpp"  # the tables fetch_ltpp.sh exported
PAIR: Final[tuple[str, str]] = ("freeze_index", "iri_first")  # the two inputs of the 2D pictures
BOOST_STAGES: Final[tuple[int, ...]] = (1, 10, 200)  # boosting rounds drawn, one input
KEY: Final[list[str]] = ["STATE_CODE", "SHRP_ID"]  # a list: pandas reads a tuple as one label


def _ltpp(table: str) -> pd.DataFrame:
    """One exported LTPP table (as ``fetch_ltpp.sh`` reads it).

    Args:
        table: The table's name.

    Returns:
        The table.

    Raises:
        FileNotFoundError: If missing (run ``bash scripts/fetch_ltpp.sh``).
    """
    path = require(LTPP_TABLES / f"{table}.csv", "fetch_ltpp.sh")
    return pd.read_csv(path, dtype={"SHRP_ID": str, "VISIT_NO": str})


def _date(s: pd.Series) -> pd.Series:
    """mdb-export's MM/DD/YY dates; years after 2030 are 19xx (as in ``fetch_ltpp.sh``).

    Args:
        s: The dates as text.

    Returns:
        The dates.
    """
    d = pd.to_datetime(s, format="%m/%d/%y %H:%M:%S", errors="raise")
    return d.where(d.dt.year <= 2030, d - pd.DateOffset(years=100))


def visits(d: pd.DataFrame, state: int, ids: list[str]) -> pd.DataFrame:
    """IRI at each roughness visit of the given sections, with age since opening.

    Rebuilds the visits behind ``iri_rate`` as ``fetch_ltpp.sh`` does (mean of both wheel paths,
    averaged over a visit's runs), and checks that their slopes reproduce the derived rates.

    Args:
        d: The sections.
        state: The state code.
        ids: The sections' SHRP IDs.

    Returns:
        One row per visit: the key, visit number, date, IRI (m/km), and age (years).

    Raises:
        ValueError: If a section's slope differs from its derived rate by 1e-4 or more.
    """
    prof, vis = _ltpp("MON_HSS_PROFILE_SECTION"), _ltpp("MON_HSS_VISIT_NO")
    p = prof.merge(vis[["VISIT_NO", *KEY, "CONSTRUCTION_NO", "VISIT_DATE"]], on="VISIT_NO")
    p = p[(p["CONSTRUCTION_NO"] == 1) & (p["STATE_CODE"] == state) & p["SHRP_ID"].isin(ids)]
    p["iri"] = p[["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"]].mean(axis=1)
    v = p.groupby([*KEY, "VISIT_NO"]).agg(date=("VISIT_DATE", "first"), iri=("iri", "mean"))
    v = v.reset_index()
    age = _ltpp("PROJECT_HIST_AGE_EXP")
    age["open"] = _date(age["TRAFFIC_OPEN_DATE"]).fillna(_date(age["CONSTRUCTION_DATE"]))
    v = v.merge(age[[*KEY, "open"]].drop_duplicates(KEY), on=KEY)
    v["age"] = (_date(v["date"]) - v["open"]).dt.days / 365.25
    v = v[(v["age"] >= 0) & v["iri"].notna()]
    for sid, g in v.groupby("SHRP_ID"):
        slope = np.polyfit(g["age"], g["iri"], 1)[0]
        want = d.loc[(d["STATE_CODE"] == state) & (d["SHRP_ID"] == sid), "iri_rate"].iloc[0]
        if abs(slope - want) >= 1e-4:
            raise ValueError(f"section {sid}: slope {slope} against the derived rate {want}")
    return v


def title_art_figure(site_visits: pd.DataFrame, out_dir: Path) -> None:
    """The title art: three stepped trees of light over a broken road, under a dark sun.

    The horizon is one section's roughness: Kansas 0105's IRI at each visit, joined.

    Args:
        site_visits: :func:`visits` of the attempt's site.
        out_dir: The figure directory.
    """
    v = site_visits[site_visits["SHRP_ID"] == TRACE[1]].sort_values("age")
    c = title_art.Canvas(seed=8)
    c.ground(("#16100c", "#1d0f12", "#0f0e1c", "#06060c"), grain=0.08)
    c.glaze(c.polygon([(0, 0), (1, 0), (1, 0.30), (0, 0.24)], soft=10), "#3a1a12", 0.8)
    for x0, w, h, col in ((0.06, 0.28, 0.80, "#7a2a18"), (0.36, 0.30, 1.02, "#a5481f"),
                          (0.68, 0.26, 0.70, "#5d2015")):  # fmt: skip
        c.glaze(c.polygon(title_art.stepped(x0, 0.22, w, h, 4), soft=3), col, 0.85)
        c.glow(c.lines([[(x0 + w / 2, 0.22), (x0 + w / 2, 0.22 + h)]], 2, soft=0.8), "#e9c27a",
               0.35, spread=4)  # fmt: skip
    rim = c.disc((0.80, 1.20), 0.11, soft=4) - c.disc((0.80, 1.20), 0.09, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.80, 1.20), 0.09, soft=3), "#1d0a10", 0.92)
    a, iri = v["age"].to_numpy(), v["iri"].to_numpy()
    xs = 0.06 + 0.88 * (a - a.min()) / np.ptp(a)
    ys = 0.04 + 0.16 * (iri - iri.min()) / np.ptp(iri)
    c.glow(c.lines([list(zip(xs, ys, strict=True))], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: three stepped Andean forms like trees, each with a "
        "faint golden trunk; a dark sun with a burning rim at the upper right; along the bottom a "
        "jagged golden line, one pavement section's roughness rising visit by visit.",
    )


def roughening_figure(
    d: pd.DataFrame, site_visits: pd.DataFrame, out_dir: Path
) -> dict[str, object]:
    """The Kansas SPS-1 sections: IRI at each visit against age, with their fitted rates.

    Args:
        d: The sections.
        site_visits: :func:`visits` of the attempt's site.
        out_dir: The figure directory.

    Returns:
        The number of sections and visits drawn.
    """
    site = d[d["site"] == ATTEMPT_SITE].sort_values("iri_rate")
    v = site_visits
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ids = site["SHRP_ID"]
    show = {ids.iloc[0]: aml_style.PALETTE[0], ids.iloc[-1]: aml_style.PALETTE[1]}  # slow, fast
    for sid, unsorted in v.groupby("SHRP_ID"):
        g = unsorted.sort_values("age")
        color = show.get(sid, aml_style.MUTED)
        ax.plot(g["age"], g["iri"], marker="o", ms=4, lw=2.2 if sid in show else 0.9, ls="-",
                color=color, alpha=1 if sid in show else 0.6)  # fmt: skip
        if sid in show:
            rate = float(site.loc[site["SHRP_ID"] == sid, "iri_rate"].iloc[0])
            ax.annotate(f"{sid}: {rate:.3f} m/km/yr", (g["age"].iloc[-1], g["iri"].iloc[-1]),
                        xytext=(8, -4), textcoords="offset points", ha="left", fontsize=12,
                        color=color, fontweight="bold", path_effects=aml_style.HALO)  # fmt: skip
    ax.set(xlabel="age (years)", ylabel="IRI (m/km)")
    if len(site) != 8:  # the title says so
        raise ValueError(f"{len(site)} sections at {ATTEMPT_SITE}, the title says eight")
    ax.set_title("Kansas SPS-1: eight sections, one site", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "roughening")
    return {"sections": int(v["SHRP_ID"].nunique()), "visits": int(len(v))}


def partition_figure(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """A depth-3 tree on two inputs: six or so boxes of the plane, each predicting its mean.

    Args:
        d: The sections.
        out_dir: The figure directory.

    Returns:
        The leaves, the splits on the freezing index, and each box's mean rate, sections, and
        states (sorted by freezing index, then first IRI).
    """
    X, y = d[list(PAIR)].to_numpy(), d["iri_rate"].to_numpy()
    tree = DecisionTreeRegressor(**SMALL_TREE, random_state=SEED).fit(X, y)
    t = tree.tree_
    pad = np.array([50.0, 0.05])
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    boxes = []

    def walk(node: int, box: list[list[float]]) -> None:
        if t.children_left[node] == -1:
            boxes.append((box, float(t.value[node][0, 0])))
            return
        j, thr = t.feature[node], t.threshold[node]
        left = [list(b) for b in box]
        right = [list(b) for b in box]
        left[j][1], right[j][0] = thr, thr
        walk(t.children_left[node], left)
        walk(t.children_right[node], right)

    walk(0, [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]])
    vals = np.array([v for _, v in boxes])
    cmap, norm = plt.get_cmap("cividis"), plt.Normalize(vals.min(), vals.max())
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for (bx, by), v in boxes:
        ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], color=cmap(norm(v)),
                               alpha=0.55, lw=1.2, ec="white"))  # fmt: skip
        if bx[1] - bx[0] < 0.06 * (hi[0] - lo[0]):
            continue  # too narrow to label
        ax.text((bx[0] + bx[1]) / 2, (by[0] + by[1]) / 2, f"{v:.3f}", ha="center", va="center",
                fontsize=11, color=aml_style.INK, zorder=4,
                path_effects=aml_style.HALO)  # fmt: skip
    ax.scatter(X[:, 0], X[:, 1], s=8, color=aml_style.INK, alpha=0.5, lw=0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="freezing index (°C-days)",
           ylabel="first IRI (m/km)")  # fmt: skip
    ax.set_title("depth 3: each box predicts its mean rate", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "partition")
    leaf = tree.apply(X)
    cells = []
    for node in np.unique(leaf):
        m = leaf == node
        cells.append(
            {
                "mean": rnd(y[m].mean(), 3),
                "sections": int(m.sum()),
                "states": int(d.loc[m, "STATE_CODE"].nunique()),
                "freeze_from": rnd(X[m, 0].min(), 0),
                "first_iri_from": rnd(X[m, 1].min(), 2),
            }
        )
    cells.sort(key=lambda c: (c["freeze_from"], c["first_iri_from"]))
    on_freeze = sorted(float(t.threshold[k]) for k in range(t.node_count) if t.feature[k] == 0)
    return {
        "leaves": len(boxes),
        "freeze_splits": [rnd(v, 1) for v in on_freeze],
        "boxes": cells,
        "mean_min": rnd(vals.min(), 3),
        "mean_max": rnd(vals.max(), 3),
    }


def terraces_figure(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """One depth-3 tree against the average of 300 such trees, in 3D: terraces against steps.

    Like for like (q2 L08-D4): the bag's trees are grown exactly as the single tree (depth 3,
    at least 5 sections in a leaf), each on a bootstrap sample of the 360 sections, with both
    inputs tried at every split (bagging: with two inputs there is little to sample).

    Args:
        d: The sections.
        out_dir: The figure directory.

    Returns:
        The highest prediction and the number of distinct levels of each surface on the grid.
    """
    X, y = d[list(PAIR)].to_numpy(), d["iri_rate"].to_numpy()
    tree = DecisionTreeRegressor(**SMALL_TREE, random_state=SEED).fit(X, y)
    bag = RandomForestRegressor(  # bagging: every input at every split
        n_estimators=BAG_TREES, max_features=1.0, bootstrap=True, **SMALL_TREE,
        random_state=SEED, n_jobs=N_JOBS,
    ).fit(X, y)  # fmt: skip
    f, i = np.meshgrid(np.linspace(*np.quantile(X[:, 0], [0.01, 0.99]), 71),
                       np.linspace(*np.quantile(X[:, 1], [0.01, 0.99]), 71))  # fmt: skip
    grid = np.column_stack([f.ravel(), i.ravel()])
    fig = plt.figure(figsize=aml_style.FULL)
    panels = (
        (tree, "one tree, depth 3: terraces"),
        (bag, f"{BAG_TREES} such trees, averaged: smaller steps"),
    )
    surfaces = {}
    for k, (model, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        z = model.predict(grid).reshape(f.shape)
        surfaces["tree" if k == 0 else "bag"] = z
        aml_style.surface3d(ax, f, i, z, cmap="cividis", vmin=0.0, vmax=0.2)
        ax.set_xlabel("freezing index", labelpad=8, fontsize=12)
        ax.set_ylabel("first IRI", labelpad=8, fontsize=12)
        ax.set(zlim=(0, 0.2), zticks=[0, 0.1, 0.2])
        ax.set_title(title, loc="left", fontsize=13)
        aml_style.style3d(ax, elev=30, azim=-130)
    fig.text(0.5, -0.06, "predicted IRI growth (m/km/yr), trained on all 360 sections", ha="center",
             fontsize=12, color=aml_style.MUTED)  # fmt: skip
    aml_style.save(fig, out_dir, "terraces")
    return {
        name: {"peak": rnd(z.max(), 3), "levels": int(len(np.unique(np.round(z, 6))))}
        for name, z in surfaces.items()
    } | {"bag_trees": BAG_TREES, "grid": int(f.size)}


def boosting_figure(d: pd.DataFrame, out_dir: Path) -> None:
    """Boosting on one input: after 1, 10, and 200 small trees, each fit to the residuals.

    Args:
        d: The sections.
        out_dir: The figure directory.
    """
    x, y = d[["iri_first"]].to_numpy(), d["iri_rate"].to_numpy()
    gb = GradientBoostingRegressor(n_estimators=max(BOOST_STAGES), learning_rate=0.1, max_depth=2,
                                   random_state=SEED).fit(x, y)  # fmt: skip
    grid = np.linspace(x.min(), x.max(), 400)[:, None]
    stages = list(gb.staged_predict(grid))
    fig, axes = plt.subplots(1, len(BOOST_STAGES), figsize=aml_style.FULL, sharey=True)
    for ax, m in zip(axes, BOOST_STAGES, strict=True):
        ax.scatter(x[:, 0], y, s=8, color=aml_style.MUTED, alpha=0.45, lw=0)
        ax.plot(grid[:, 0], stages[m - 1], color=aml_style.PALETTE[1], lw=2.6, ls="-")
        note = {1: "", max(BOOST_STAGES): ": chasing noise"}.get(m, "")
        ax.set_title(f"{m} tree{'s' if m > 1 else ''}{note}", loc="left")
        ax.set(xlabel="first IRI (m/km)", ylim=(-0.05, 0.3))
    axes[0].set_ylabel("IRI growth (m/km/yr)")
    aml_style.save(fig, out_dir, "boosting")


def waterfall_figure(tr: dict[str, object], out_dir: Path) -> None:
    """Exact Shapley values for Kansas 0105: from the average prediction to this one.

    Args:
        tr: The trace from :func:`trace`.
        out_dir: The figure directory.
    """
    raw = tr["_raw"]
    if not isinstance(raw, dict):
        raise TypeError("the trace has no unrounded values")
    phi, base, pred, actual = raw["phi"], raw["base"], raw["pred"], raw["actual"]
    order = np.argsort(-np.abs(phi))
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    level = base
    for row, k in enumerate(order):
        color = aml_style.PALETTE[1] if phi[k] > 0 else aml_style.PALETTE[0]
        ax.barh(row, phi[k], left=level, color=color, height=0.65)
        level += phi[k]
    ax.axvline(base, color=aml_style.MUTED, lw=1, ls=":")
    ax.axvline(pred, color=aml_style.INK, lw=1.6)
    lim = (min(base, pred, level) - 0.008, max(base, pred) + 0.012)
    ax.set_yticks(range(len(order)), [NAMES[FEATURES[k]] for k in order], fontsize=11)
    ax.invert_yaxis()
    # Offset from their lines by a few points, so neither label starts on its line.
    ax.annotate(f"average {base:.3f}", (base, -0.9), xytext=(-4, 0), textcoords="offset points",
                fontsize=11, ha="right", color=aml_style.MUTED)  # fmt: skip
    ax.annotate(f"prediction {pred:.3f}", (pred, len(order) - 0.2), xytext=(4, 0),
                textcoords="offset points", fontsize=11, ha="left")  # fmt: skip
    ax.annotate(rf"measured {actual:.3f} $\rightarrow$", (lim[1], len(order) / 2), ha="right",
                fontsize=12, color=aml_style.PALETTE[1], fontweight="bold")  # fmt: skip
    ax.set(xlabel="IRI growth (m/km/yr)", ylim=(len(order) + 0.2, -1.5), xlim=lim)
    aml_style.save(fig, out_dir, "waterfall")


def overfit_figure(
    train_r2: dict[str, float],
    held_r2: dict[str, float],
    out_dir: Path,
    *,
    held_out: bool = True,
) -> None:
    """Training R^2 against R^2 with states held out: the deep tree memorizes.

    Args:
        train_r2: Training R^2 by model.
        held_r2: R^2 with states held out, by model (the untuned models).
        out_dir: The figure directory.
        held_out: False draws the training bars only (the slide before the predictions).
    """
    names = ("tree, full depth", "tree, depth 3", "random forest")
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    yy = np.arange(len(names))
    train = [train_r2[name] for name in names]
    held = [held_r2[name] for name in names]
    ax.barh(yy - 0.2, train, height=0.38, color=aml_style.PALETTE[0], label="training")
    if held_out:
        ax.barh(yy + 0.2, held, height=0.38, color=aml_style.PALETTE[1], hatch="//", ec="white",
                label="states held out")  # fmt: skip
        for y0, v in zip(yy, held, strict=True):
            ax.text(v - 0.05, y0 + 0.2, f"{v:.2f}", ha="right", va="center", fontsize=12)
    for y0, v in zip(yy, train, strict=True):
        ax.text(v + 0.05, y0 - 0.2, f"{v:.2f}", ha="left", va="center", fontsize=12)
    ax.axvline(0, color=aml_style.INK, lw=1)
    ax.set_yticks(yy, list(names))
    ax.invert_yaxis()
    ax.set(xlabel="R² against the mean", xlim=(-2.1, 1.4) if held_out else (0, 1.4))
    ax.legend(loc="lower left" if held_out else "lower right", fontsize=11)  # clear of the bars
    aml_style.save(fig, out_dir, "overfit" if held_out else "overfit_train")


def main() -> None:
    """Generate every figure and number for CEE Lecture 8."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    d = load()
    tr = trace(d)
    fold_numbers, r2 = folds(d, out_dir)
    over, train_r2 = overfit(d)
    numbers = {
        "summary": summary(d),
        "attempt": attempt(d),
        "verify": verify(d),
        "folds": fold_numbers,
        "floor": noise_floor(d),
        "importance": importance(d, out_dir),
        "designed": designed(d, out_dir),
        "overfit": over,
        "trace": tr,
    }
    # the visuals that carry the slides
    ids = d.loc[d["site"] == ATTEMPT_SITE, "SHRP_ID"].tolist()
    site_visits = visits(d, TRACE[0], ids)
    title_art_figure(site_visits, out_dir)
    numbers["roughening"] = roughening_figure(d, site_visits, out_dir)
    numbers["partition"] = partition_figure(d, out_dir)
    numbers["terraces"] = terraces_figure(d, out_dir)
    boosting_figure(d, out_dir)
    waterfall_figure(tr, out_dir)
    for held in (False, True):
        overfit_figure(train_r2, r2["states held out"], out_dir, held_out=held)
    aml_style.thumbnails(out_dir, ("terraces", "folds", "importance", "designed"))
    write_numbers(numbers, out_dir)
    print(write_tex(tr, out_dir))


if __name__ == "__main__":
    main()
