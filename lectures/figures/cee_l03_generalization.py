"""Figures and numbers for CEE Lecture 3: generalization and validation (Illinois bridges).

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l03_generalization.py \
        --out build/fig/cee_l03_generalization

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one held-out bridge, for the TikZ diagram). Data: ``data/derived/nbi_il_bridges.csv``
from ``scripts/fetch_nbi.sh`` (FHWA National Bridge Inventory, Illinois, 2014-2024).

The setting: an engineer in 2019 has the 2014, 2016, and 2018 inventories (the training
records) and wants to know how wrong a k-NN model of the condition rating will be on the bridges
where it is used. The 2024 inventory is the test set: it scores the k that each validation
chose; its whole curve over k is drawn only as a post-hoc diagnostic, never used to choose.

Reproducibility: ties in the neighbor searches are resolved by the last bit of the distances,
which depends on the BLAS thread count. Rerun with the Makefile's default threads to reproduce
numbers.json exactly; with 1 or 4 threads the small-k curves move in the fourth decimal and the
tie disagreements of ``verify_listing`` change (2026-11-21 check).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from bridges import COUNTY, FEATURES, POOR, find_twins, load_bridges
from common import (
    parse_out_dir,
    pct,
    rmse,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

SEED: Final[int] = 0
FOLDS: Final[int] = 5
KS: Final[np.ndarray] = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100])
K: Final[int] = 50  # between the honest choices (30, 70); within 0.01 of both minima (main)
TRAIN_YEARS: Final[tuple[int, ...]] = (2014, 2016, 2018)  # what an engineer had in 2019
LATER_YEARS: Final[tuple[int, ...]] = (2020, 2022, 2024)
TEST_YEAR: Final[int] = 2024  # the test set: scores only the k that validation chose
FLAG_SHARE: Final[float] = 0.10  # screening: flag this share of bridges for inspection
DRAWS: Final[int] = 20  # training subsets for the bias-variance measurement
KS_BV: Final[np.ndarray] = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
KS.flags.writeable = False
KS_BV.flags.writeable = False
LEAKY: Final[tuple[str, ...]] = ("inspection_interval", "structural_eval")  # set from the rating
DESIGN: Final[dict[str, str]] = {
    "01": "slab",
    "02": "beams",
    "04": "tee beams",
    "05": "box beams",
    "07": "frame",
    "09": "truss",
    "10": "truss",
    "11": "arch",
    "22": "channel beams",
}
# The column that places a bridge in a county fold: the county of its first record. Ten bridges
# carry two county codes over the years (30 training records); with the records' own codes, two
# of them were split across county folds (D14 of the 2026-11-20 review).
COUNTY_FOLD: Final[str] = "county_fold"
Q_MIN_ADT: Final[int] = 50_000  # the traced bridge Q carries at least this many vehicles a day
MIN_COUNTY_RECORDS: Final[int] = 300  # counties ranked on the impact slide have this many
NULL_PERMUTATIONS: Final[int] = 500  # bridges dealt to counties at random: the map's noise floor
MIN_INTERVAL_RECORDS: Final[int] = 100  # inspection intervals quoted on the leakage slide
AGE_BINS: Final[tuple[int, ...]] = (0, 10, 20, 30, 40, 50, 60, 80, 200)  # drift figure, years
AGE_LABELS: Final[tuple[str, ...]] = (
    "0–9",
    "10–19",
    "20–29",
    "30–39",
    "40–49",
    "50–59",
    "60–79",
    "80+",
)
# Numbers quoted from verified sources (not computed here), as the slides and notes quote them.
CITED: Final[dict[str, dict[str, object]]] = {
    "moore_2001": {
        "within_one_point_pct": 68,
        "within_two_points_pct": 95,
        "sigma_points": 1,
        "inspectors": 49,
        "states": 25,
        "source": "Moore et al. (2001), Reliability of Visual Inspection for Highway Bridges, "
        "FHWA-RD-01-020",
    },
    "ploton_2020": {
        "trees_million": 11.8,
        "random_cv": "predicts more than half",
        "spatial_cv": "quasi-null predictive power",
        "source": "Ploton et al. (2020), Nature Communications 11, abstract",
    },
    "jakubik_2023": {
        "pretraining_tb": 1,
        "source": "Jakubik et al. (2023), Foundation Models for Generalist Geospatial "
        "Artificial Intelligence (Prithvi), arXiv:2310.18660, abstract",
    },
    "sainz_2023": {
        "quote": "contamination causes an overestimation of the performance",
        "source": "Sainz et al. (2023), Findings of EMNLP 2023",
    },
}
# Lecture 2's results that the warm-up recalls (its numbers.json keys in the comments).
RECALL_L02: Final[dict[str, float]] = {
    "water_distance_share_pct": 1.2,  # distance_share_pct.water
    "random_split_mpa": 5.7,  # representation_rmse["dimensionless groups"].random_mean, 5.653
    "designs_held_out_mpa": 6.9,  # representation_rmse["dimensionless groups"].grouped_mean
}


# ---------------------------------------------------------------------------------------------
# Data and the encoder

INSPECTION: Final[tuple[str, ...]] = ("structure", "inspection_year", "inspection_month")


def unique_inspections(records: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per inspection.

    A bridge on a 48-month interval reports the same inspection (same date, same ratings) in
    two consecutive inventories. Each inspection is kept once, from the first inventory that
    reports it.

    Args:
        records: Bridge-year records.

    Returns:
        The records with repeated inspections dropped, index reset.
    """
    first = records.sort_values("nbi_year", kind="stable").drop_duplicates(list(INSPECTION))
    return first.sort_index().reset_index(drop=True)


def add_county_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``COUNTY_FOLD``: the county of each bridge's first record, for the county folds.

    A bridge whose records carry two county codes (ten in training, 30 records) would otherwise
    have records in two county folds, so its other records would train the model that is
    scored on "its county". Every record of a bridge takes the code of its first record.

    Args:
        df: All bridge-year records, in inventory-year order (as ``load_bridges`` returns them).

    Returns:
        A copy with the ``COUNTY_FOLD`` column.

    Raises:
        ValueError: If the records are not in inventory-year order ("first" would not be the
            earliest record).
    """
    if not df["nbi_year"].is_monotonic_increasing:
        raise ValueError("records must be in inventory-year order")
    out = df.copy()
    out[COUNTY_FOLD] = out.groupby("structure")["county"].transform("first")
    return out


def later_records(df: pd.DataFrame, train: pd.DataFrame, year: int) -> pd.DataFrame:
    """The records of a later inventory whose inspection is not already in training.

    Args:
        df: All records.
        train: Training inspections.
        year: The later NBI year.

    Returns:
        Records of ``year`` that report an inspection not in ``train``, index reset.
    """
    rec = df[df["nbi_year"] == year]
    seen = pd.MultiIndex.from_frame(train[list(INSPECTION)])
    rec_key = pd.MultiIndex.from_frame(rec[list(INSPECTION)])
    return rec[~rec_key.isin(seen)].reset_index(drop=True)


def encode(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> np.ndarray:
    """The inputs as a float matrix (standardized later, inside each fold).

    Args:
        df: Bridge-year records from :func:`load_bridges`.
        extra: Further columns to append (the leakage experiment).

    Returns:
        Array of shape ``(len(df), len(FEATURES) + len(extra))``.
    """
    X = df[FEATURES + list(extra)].to_numpy(np.float64)
    assert X.shape == (len(df), len(FEATURES) + len(extra)), X.shape
    return X


# ---------------------------------------------------------------------------------------------
# Cross-validation (the code slide's listing) and held-out tests


def cv_predict(
    X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], ks: np.ndarray
) -> np.ndarray:
    """Out-of-fold k-NN predictions for every k in ``ks`` (the slide's listing).

    Args:
        X: Inputs, shape ``(N, D)``.
        y: Ratings, shape ``(N,)``.
        folds: ``(train, held_out)`` index pairs; every row is held out exactly once.
        ks: Numbers of neighbors, a NumPy integer array.

    Returns:
        Predictions, shape ``(N, len(ks))``.
    """
    pred = np.full((len(y), len(ks)), np.nan)
    for tr, te in folds:  # te: one held-out block
        sc = StandardScaler().fit(X[tr])  # training rows only
        Z_tr, Z_te = sc.transform(X[tr]), sc.transform(X[te])
        nn = NearestNeighbors(n_neighbors=ks.max(), algorithm="brute")
        _, idx = nn.fit(Z_tr).kneighbors(Z_te)  # nearest first
        total = np.cumsum(y[tr][idx], axis=1)  # sums of nearest 1..K
        pred[te] = total[:, ks - 1] / ks
    assert not np.isnan(pred).any()  # each row held out once
    return pred


def holdout_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, ks: np.ndarray = KS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on one set and predict another, for every k in ``ks``.

    Args:
        X_tr: Training inputs.
        y_tr: Training ratings.
        X_te: Inputs to predict.
        ks: Numbers of neighbors.

    Returns:
        Predictions ``(len(X_te), len(ks))``; neighbor indices into ``X_tr`` and their
        distances in standardized units, both ``(len(X_te), max(ks))``.
    """
    sc = StandardScaler().fit(X_tr)
    nn = NearestNeighbors(n_neighbors=int(ks.max()), algorithm="brute").fit(sc.transform(X_tr))
    dist, idx = nn.kneighbors(sc.transform(X_te))
    total = np.cumsum(y_tr[idx], axis=1)
    return total[:, ks - 1] / ks, idx, dist


def rmse_per_k(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RMSE of each column of ``pred`` against ``y``.

    Args:
        pred: Predictions, one column per k, shape ``(N, len(ks))``.
        y: Ratings, shape ``(N,)``.

    Returns:
        One RMSE per column.

    Raises:
        ValueError: If ``pred`` is not ``(N, len(ks))`` for a ``y`` of shape ``(N,)``: a
            ``(N, 1)`` target would broadcast to a meaningless ``(N, N, len(ks))`` difference.
    """
    if pred.ndim != 2 or y.ndim != 1 or pred.shape[0] != y.shape[0]:
        raise ValueError(f"rmse_per_k of shapes {pred.shape} and {y.shape}")
    return np.sqrt(((pred - y[:, None]) ** 2).mean(axis=0))


def county_fold_of(df: pd.DataFrame) -> dict[str, int]:
    """Deal the counties to ``FOLDS`` folds: ``GroupKFold`` over the records' own county codes.

    Args:
        df: Training records.

    Returns:
        County code to fold number, for every county code in ``df``.
    """
    fold_of: dict[str, int] = {}
    for j, (_, te) in enumerate(GroupKFold(FOLDS).split(df, groups=df["county"])):
        fold_of |= dict.fromkeys(df["county"].iloc[te].unique(), j)
    return fold_of


def county_fold_index(records: pd.DataFrame, fold_of: dict[str, int]) -> np.ndarray:
    """The county fold of each record: the fold of its bridge's first county.

    Args:
        records: Records with the ``COUNTY_FOLD`` column (``add_county_fold``).
        fold_of: Output of ``county_fold_of``.

    Returns:
        One fold number per record.

    Raises:
        KeyError: If a bridge's first county has no fold.
    """
    fold = records[COUNTY_FOLD].map(fold_of)
    if fold.isna().any():
        missing = sorted(set(records.loc[fold.isna(), COUNTY_FOLD]))
        raise KeyError(f"counties without a fold: {missing}")
    return fold.to_numpy(int)


def make_folds(df: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Five folds of the training records.

    The county folds deal the counties to folds exactly as ``GroupKFold`` does over the
    records' own county codes, then put every record in the fold of its bridge's first county
    (``add_county_fold``): only the records of bridges split across folds move. (Grouping
    ``GroupKFold`` by the first county directly would reshuffle four of the five folds, since
    it deals counties by size.)

    Args:
        df: Training records, with the ``COUNTY_FOLD`` column.
        scheme: ``"random"`` (records shuffled), ``"bridge"`` (every record of a bridge in one
            fold), or ``"county"`` (every record of a county, and of each bridge, in one fold).

    Returns:
        ``(train, held_out)`` index pairs.

    Raises:
        ValueError: On an unknown scheme.
    """
    if scheme == "random":
        return list(KFold(FOLDS, shuffle=True, random_state=SEED).split(df))
    if scheme == "bridge":
        return list(GroupKFold(FOLDS).split(df, groups=df["structure"]))
    if scheme != "county":
        raise ValueError(f"unknown scheme {scheme!r}")
    fold = county_fold_index(df, county_fold_of(df))
    rows = np.arange(len(df))
    return [(rows[fold != j], rows[fold == j]) for j in range(FOLDS)]


def k_index(k: int) -> int:
    """Column of ``k`` in the prediction matrices.

    Args:
        k: A number of neighbors in ``KS``.

    Returns:
        Its column.
    """
    return int(np.flatnonzero(k == KS)[0])


# ---------------------------------------------------------------------------------------------
# Experiments


def data_summary(df: pd.DataFrame, train: pd.DataFrame) -> dict[str, object]:
    """Counts for the data slide.

    Args:
        df: All records.
        train: Training inspections (one record each).

    Returns:
        Record, bridge, and county counts, the training and test sizes, the mean rating and
        poor share of each inventory, and the training bridges that carry two county codes
        (with how many of them the records' own codes would split across county folds).
    """
    test = later_records(df, train, TEST_YEAR)
    raw = df[df["nbi_year"].isin(TRAIN_YEARS)]
    by_year = df.groupby("nbi_year")["rating"]
    own_fold = train["county"].map(county_fold_of(train)).to_numpy(int)
    per_bridge = pd.DataFrame(
        {"bridge": train["structure"], "code": train["county"], "fold": own_fold}
    ).groupby("bridge")
    codes = per_bridge["code"].nunique()
    two_codes = codes[codes > 1].index
    return {
        "train_bridges_two_county_codes": int(len(two_codes)),
        "train_records_two_county_codes": int(train["structure"].isin(two_codes).sum()),
        "train_bridges_split_by_own_codes": int((per_bridge["fold"].nunique() > 1).sum()),
        "years": sorted(int(v) for v in df["nbi_year"].unique()),
        "records": int(len(df)),
        "bridges": int(df["structure"].nunique()),
        "counties": int(df["county"].nunique()),
        "bridges_per_year_min": int(df.groupby("nbi_year").size().min()),
        "bridges_per_year_max": int(df.groupby("nbi_year").size().max()),
        "train_inventory_records": int(len(raw)),
        "repeated_inspections": int(len(raw) - len(train)),
        "train_records": int(len(train)),
        "train_bridges": int(train["structure"].nunique()),
        "test_records": int(len(test)),
        "mean_rating": {str(y): rnd(v, 2) for y, v in by_year.mean().items()},
        "poor_pct": {str(y): pct((v <= POOR).mean(), 1) for y, v in by_year},
        "train_sd": rnd(train["rating"].std(ddof=0), 2),
        "n_features": len(FEATURES),
    }


@dataclass(frozen=True)
class Curves:
    """The validation curves: what goes in numbers.json and the arrays later steps reuse.

    Attributes:
        numbers: Curves (one RMSE per k in ``KS``, to 4 decimals), the k each validation picks,
            and the 2024 test at those k, for numbers.json.
        raw: Unrounded curves by name (``random``, ``bridge``, ``county``, ``test_<year>``,
            ``bias_<year>``, ``test_2024_new_counties``), one value per k in ``KS``.
        oof: Out-of-fold predictions of the training records by scheme, ``(N, len(KS))``.
        later: Predictions of each later inventory's new inspections by year (a string),
            ``(n_year, len(KS))``.
        best: The k each validation scheme picks.
    """

    numbers: dict[str, object]
    raw: dict[str, np.ndarray]
    oof: dict[str, np.ndarray]
    later: dict[str, np.ndarray]
    best: dict[str, int]


def interior_k(curve: np.ndarray, ks: np.ndarray, name: str) -> int:
    """The k at a curve's minimum, which must lie strictly inside the grid.

    Args:
        curve: One error per k in ``ks``.
        ks: The grid of k.
        name: The curve's name, for the error message.

    Returns:
        The k with the lowest error.

    Raises:
        RuntimeError: If the minimum sits at either end of the grid.
    """
    i = int(np.argmin(curve))
    if i in (0, len(ks) - 1):
        raise RuntimeError(f"{name}: the best k = {ks[i]} is at the edge of the grid {ks.tolist()}")
    return int(ks[i])


def validation_curves(train: pd.DataFrame, df: pd.DataFrame) -> Curves:
    """RMSE against k for three ways of folding the training records and for later years.

    Args:
        train: Records of ``TRAIN_YEARS``.
        df: All records.

    Returns:
        The curves, the out-of-fold and later-year predictions, and the k each validation picks
        (each strictly inside ``KS``).
    """
    X, y = encode(train), train["rating"].to_numpy(float)
    raw: dict[str, np.ndarray] = {}
    oof: dict[str, np.ndarray] = {}
    for scheme in ("random", "bridge", "county"):
        oof[scheme] = cv_predict(X, y, make_folds(train, scheme), KS)
        raw[scheme] = rmse_per_k(oof[scheme], y)
    later: dict[str, np.ndarray] = {}
    own_nn: dict[str, float] = {}
    st = train["structure"].to_numpy()
    for year in LATER_YEARS:
        test = later_records(df, train, year)
        pred, idx, _ = holdout_predict(X, y, encode(test))
        later[str(year)] = pred
        own_nn[f"own_nn_pct_{year}"] = pct(
            (st[idx[:, 0]] == test["structure"].to_numpy()).mean(), 0
        )
        y_te = test["rating"].to_numpy(float)
        raw[f"test_{year}"] = rmse_per_k(pred, y_te)
        raw[f"bias_{year}"] = (pred - y_te[:, None]).mean(axis=0)
    raw["test_2024_new_counties"] = new_county_test(train, df)
    out: dict[str, object] = {"k": KS.tolist()}
    for name, vals in raw.items():
        out[name] = [rnd(v, 4) for v in vals]
    out |= own_nn
    # The k each validation picks; the 2024 test scores those k only (choosing k on the 2024
    # curve would make it a validation set).
    best = {s: interior_k(raw[s], KS, s) for s in ("random", "bridge", "county")}
    test_raw = raw[f"test_{TEST_YEAR}"]
    out["best_k"] = best
    out["test_2024_at_chosen_k"] = {s: rnd(test_raw[k_index(k)], 2) for s, k in best.items()}
    out["test_2024_new_counties_at_county_k"] = rnd(
        raw["test_2024_new_counties"][k_index(best["county"])], 2
    )
    return Curves(numbers=out, raw=raw, oof=oof, later=later, best=best)


def new_county_test(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """2024 RMSE per k for counties new to the model.

    Each county block of the training folds is held out in turn: train on the other counties'
    2014-2018 records, test on the held-out counties' 2024 records. A 2024 record belongs to
    the fold of its bridge's first county (``COUNTY_FOLD``), as its training records do.

    Args:
        train: Records of ``TRAIN_YEARS``.
        df: All records, with the ``COUNTY_FOLD`` column.

    Returns:
        One RMSE per k in ``KS``.
    """
    X, y = encode(train), train["rating"].to_numpy(float)
    test = later_records(df, train, TEST_YEAR)
    test_fold = county_fold_index(test, county_fold_of(train))
    sq = []
    for j, (tr, _) in enumerate(make_folds(train, "county")):
        held = test[test_fold == j]
        pred, _, _ = holdout_predict(X[tr], y[tr], encode(held))
        sq.append((pred - held["rating"].to_numpy(float)[:, None]) ** 2)
    sq_all = np.concatenate(sq)
    assert sq_all.shape == (len(test), len(KS)), sq_all.shape  # every 2024 record once
    return np.sqrt(sq_all.mean(axis=0))


def baselines(train: pd.DataFrame) -> dict[str, float]:
    """Predict the training mean, counties held out (each fold's training mean).

    Args:
        train: Training records.

    Returns:
        Its RMSE.
    """
    y = train["rating"].to_numpy(float)
    pred = np.empty_like(y)
    for tr, te in make_folds(train, "county"):
        pred[te] = y[tr].mean()
    return {"mean_county_cv": rnd(rmse(pred, y), 2)}


def twin_floor(df: pd.DataFrame) -> dict[str, int | float]:
    """A noise floor from twin bridges in the 2018 inventory: one visit's disagreement.

    Two twins share age, design, traffic, and weather. If their true conditions were equal and
    their rating errors independent, with standard deviation sigma, the difference of their
    ratings would have variance 2 sigma^2, so sigma = RMS(difference) / sqrt(2). But most twins
    are rated in the same month, likely by the same crew, so their errors are positively
    correlated (rho): RMS / sqrt(2) then estimates sigma sqrt(1 - rho), the disagreement within
    one visit, and understates the noise between inspectors. A lower bound, not an upper one.
    The pairs come from the whole 2018 inventory, every bridge's record that year (not only
    the inspections new in 2018 that training keeps): the floor is a property of the ratings.

    Args:
        df: All bridge-year records.

    Returns:
        Pair counts, agreement shares, the RMS difference, the floor, the share of pairs
        inspected in the same month, and the floor of the other pairs.
    """
    s18 = df[df["nbi_year"] == 2018]
    tw = find_twins(s18)
    mutual = tw.merge(tw, left_on=["row", "twin_row"], right_on=["twin_row", "row"])
    pairs = mutual[mutual["row_x"] < mutual["twin_row_x"]][["row_x", "twin_row_x"]]
    a, b = s18.loc[pairs["row_x"]], s18.loc[pairs["twin_row_x"]]
    diff = a["rating"].to_numpy() - b["rating"].to_numpy()
    same_month = np.ones(len(pairs), dtype=bool)
    for col in ("inspection_year", "inspection_month"):
        same_month &= a[col].to_numpy() == b[col].to_numpy()
    rms = float(np.sqrt(np.mean(diff**2)))
    rms_other = float(np.sqrt(np.mean(diff[~same_month] ** 2)))
    return {
        "pairs": int(len(pairs)),
        "same_rating_pct": pct((diff == 0).mean(), 0),
        "within_one_pct": pct((np.abs(diff) <= 1).mean(), 0),
        "rms_difference": rnd(rms, 2),
        "floor": rnd(rms / np.sqrt(2), 2),
        "pairs_same_month": int(same_month.sum()),
        "same_month_pct": pct(same_month.mean(), 0),
        "pairs_other_months": int((~same_month).sum()),
        "floor_other_months": rnd(rms_other / np.sqrt(2), 2),
    }


def fold_overlap(train: pd.DataFrame) -> dict[str, float]:
    """How often a held-out record has its own bridge, or its twin, in training (random folds).

    Twins are found among the training records of 2018: the question is about training records.

    Args:
        train: Training records.

    Returns:
        The two shares, in percent.
    """
    twins = find_twins(train[train["nbi_year"] == 2018])
    st = train["structure"].to_numpy()
    twin_of = dict(zip(st[twins["row"]], st[twins["twin_row"]], strict=True))
    own, twin, n_twin = 0, 0, 0
    for tr, te in make_folds(train, "random"):
        in_train = set(st[tr])
        own += sum(s in in_train for s in st[te])
        held_twins = [twin_of[s] for s in st[te] if s in twin_of]
        n_twin += len(held_twins)
        twin += sum(t in in_train for t in held_twins)
    return {
        "own_bridge_in_train_pct": pct(own / len(train), 0),
        "twin_in_train_pct": pct(twin / n_twin, 0),
    }


def verify_listing(train: pd.DataFrame) -> dict[str, object]:
    """Compare ``cv_predict`` with scikit-learn's ``cross_val_predict`` (counties held out).

    The reference is a pipeline of ``StandardScaler`` and ``KNeighborsRegressor`` with the same
    folds. The two can differ only where the k-th and (k+1)-th nearest distances tie. Which tied
    record each code keeps depends on the last bit of the computed distances, so ``match``,
    ``differ`` and ``max_abs_diff`` change with the BLAS thread count (k = 50: 48 records and
    0.06 with MKL's 28 threads, 39 and 0.08 with 1 or 4); the tie counts do not. The slides
    quote this machine's counts as such.

    Args:
        train: Training records.

    Returns:
        The number of records and, for k = 1 and ``K``: the records that match to 1e-9, those
        that differ, those with a tie at the k-th neighbor, and the largest difference.

    Raises:
        RuntimeError: If the two codes differ on a record without a tie.
    """
    X, y = encode(train), train["rating"].to_numpy(float)
    folds = make_folds(train, "county")
    ours = cv_predict(X, y, folds, KS)
    out: dict[str, object] = {"records": int(len(y))}
    for k in (1, K):  # the slide quotes k = 50 and the notes k = 1
        ref = cross_val_predict(
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, algorithm="brute")),
            X,
            y,
            cv=folds,
        )
        agree = np.isclose(ours[:, k_index(k)], ref, rtol=0, atol=1e-9)
        tie = np.zeros(len(y), dtype=bool)
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            dist, _ = (
                NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
                .fit(sc.transform(X[tr]))
                .kneighbors(sc.transform(X[te]))
            )
            tie[te] = np.isclose(dist[:, k - 1], dist[:, k], rtol=0, atol=1e-9)
        if not agree[~tie].all():
            raise RuntimeError(f"k={k}: cv_predict disagrees with scikit-learn away from ties")
        out[f"k{k}"] = {
            "match": int(agree.sum()),
            "differ": int((~agree).sum()),
            "ties": int(tie.sum()),
            "max_abs_diff": rnd(np.abs(ours[:, k_index(k)] - ref).max(), 2),
        }
    return out


def bias_variance_draws(train: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Measure variance and bias^2 + noise on the first county block, over ``DRAWS`` subsets.

    Each draw keeps a random half of the training bridges (all their records) and predicts the
    held-out counties. For every held-out record, the mean squared error over draws splits
    exactly into the variance of the predictions and the squared distance of their mean to
    the rating (bias^2 + noise). Also measures the training error (fit on all records, predict
    the same records).

    Args:
        train: Training records.

    Returns:
        Per k in ``KS_BV``: ``mse``, ``bias2_noise``, ``variance``, and ``train_mse``; and the
        held-out rows.
    """
    X, y = encode(train), train["rating"].to_numpy(float)
    st = train["structure"].to_numpy()
    tr, te = make_folds(train, "county")[0]
    bridges = np.unique(st[tr])
    rng = np.random.default_rng(SEED)
    preds = np.empty((DRAWS, len(te), len(KS_BV)))
    for b in range(DRAWS):
        keep = rng.choice(bridges, size=len(bridges) // 2, replace=False)
        sub = tr[np.isin(st[tr], keep)]
        preds[b], _, _ = holdout_predict(X[sub], y[sub], X[te], ks=KS_BV)
    assert preds.shape == (DRAWS, len(te), len(KS_BV)), preds.shape
    variance = preds.var(axis=0).mean(axis=0)
    bias2_noise = ((preds.mean(axis=0) - y[te][:, None]) ** 2).mean(axis=0)
    mse = ((preds - y[te][None, :, None]) ** 2).mean(axis=(0, 1))
    assert np.allclose(mse, variance + bias2_noise, rtol=1e-10, atol=1e-12), "decomposition"
    train_pred, _, _ = holdout_predict(X, y, X, ks=KS_BV)
    train_mse = ((train_pred - y[:, None]) ** 2).mean(axis=0)
    curves = {"mse": mse, "bias2_noise": bias2_noise, "variance": variance, "train_mse": train_mse}
    return curves, te


def bias_variance_figure(curves: dict[str, np.ndarray], out_dir: Path) -> None:
    """Plot the held-out error, its two parts, and the training error against k.

    Args:
        curves: Output of ``bias_variance_draws``.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    labels = {
        "mse": "held-out error",
        "bias2_noise": "bias² + noise",
        "variance": "variance",
        "train_mse": "training error",
    }
    for c, (key, label) in enumerate(labels.items()):
        ax.plot(
            KS_BV,
            curves[key],
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            marker=aml_style.MARKERS[c],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("squared error (points²)")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper right", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "bias_variance")


def bias_variance(train: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """The bias-variance measurement (``bias_variance_draws``), its figure, and its numbers.

    Every error is squared (points²), as the figure plots it. The draws are random halves of
    one pool of training bridges, so ``variance`` is the spread over such halves, not over
    independent training sets from the population.

    Args:
        train: Training records.
        out_dir: Figure directory.

    Returns:
        The squared errors and their parts at the k the slide quotes.
    """
    curves, te = bias_variance_draws(train)
    bias_variance_figure(curves, out_dir)
    mse, b2n, var = curves["mse"], curves["bias2_noise"], curves["variance"]
    mse_best_k = interior_k(mse, KS_BV, "held-out error")
    b2n_min_k = interior_k(b2n, KS_BV, "bias² + noise")

    def at(k: int) -> int:
        """Column of ``k`` in the curves.

        Args:
            k: A number of neighbors in ``KS_BV``.

        Returns:
            Its column.
        """
        return int(np.flatnonzero(k == KS_BV)[0])

    i1, i10, i100, i1000 = at(1), at(10), at(100), at(1000)
    return {
        "held_out_records": int(len(te)),
        "draws": DRAWS,
        "k_max": int(KS_BV.max()),
        "held_out_counties": int(train[COUNTY_FOLD].iloc[te].nunique()),
        "variance_k1": rnd(var[i1], 2),
        "variance_k10": rnd(var[i10], 2),
        "variance_k100": rnd(var[i100], 2),
        "variance_k1000": rnd(var[i1000], 2),
        "bias2_noise_k1": rnd(b2n[i1], 2),
        "bias2_noise_k100": rnd(b2n[i100], 2),
        "bias2_noise_min": rnd(b2n.min(), 2),
        "bias2_noise_min_k": b2n_min_k,
        "bias2_noise_k1000": rnd(b2n[i1000], 2),
        "mse_k1": rnd(mse[i1], 2),
        "mse_best": rnd(mse.min(), 2),
        "mse_k1000": rnd(mse[i1000], 2),
        "mse_best_k": mse_best_k,
        "train_mse_k1": rnd(curves["train_mse"][i1], 3),
        # From the two rounded variances, so the notes' arithmetic reproduces (0.67 / 0.12).
        "variance_ratio_k1_k10": rnd(rnd(var[i1], 2) / rnd(var[i10], 2), 1),
    }


def bracket_figure(
    curves: Curves, base: dict[str, float], floor: dict[str, int | float], out_dir: Path
) -> None:
    """Dot plot: the honest error between predicting the mean and the twin-bridge floor.

    Args:
        curves: Output of ``validation_curves`` (its unrounded curves).
        base: Output of ``baselines``.
        floor: Output of ``twin_floor``.
        out_dir: Figure directory.
    """
    county = curves.raw["county"]
    rows = [
        ("predict the mean", base["mean_county_cv"], "o"),
        (f"k-NN, k = {K}", float(county[k_index(K)]), "s"),
        ("k-NN, k = 1", float(county[k_index(1)]), "^"),
        ("twins, one visit", floor["floor"], "D"),
    ]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    pos = np.arange(len(rows))[::-1]
    for p, (_, v, m) in zip(pos, rows, strict=True):
        ax.plot(v, p, marker=m, ms=10, ls="none", color=aml_style.PALETTE[0])
        ax.text(v, p + 0.25, f"{v:.2f}", ha="center", fontsize=12)
    ax.set_yticks(pos, [r[0] for r in rows])
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("RMSE (rating points), counties held out")
    aml_style.save(fig, out_dir, "bracket")


def folds_map(train: pd.DataFrame, out_dir: Path) -> dict[str, int]:
    """Two maps of the first held-out block: random folds vs counties held out.

    Args:
        train: Training records (the maps draw their 2018 ones).
        out_dir: Figure directory.

    Returns:
        The counties that each first held-out block touches.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.SIDE)
    s18 = train["nbi_year"].to_numpy() == 2018
    out: dict[str, int] = {}
    titles = {"random": "random folds", "county": "counties held out"}
    for ax, (scheme, title) in zip(axes, titles.items(), strict=True):
        _, te = make_folds(train, scheme)[0]
        held = np.zeros(len(train), dtype=bool)
        held[te] = True
        keep_tr, keep_te = s18 & ~held, s18 & held
        ax.scatter(
            train["x_km"][keep_tr],
            train["y_km"][keep_tr],
            s=0.6,
            color="#C8C8C8",
            marker="o",
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            train["x_km"][keep_te],
            train["y_km"][keep_te],
            s=1.6,
            color=aml_style.PALETTE[0],
            marker="s",
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, loc="center", fontsize=14)
        out[f"{scheme}_held_out_counties"] = int(train[COUNTY_FOLD].iloc[te].nunique())
    fig.tight_layout(w_pad=0.5)
    aml_style.save(fig, out_dir, "folds_map")
    return out


def k_curves_figure(curves: Curves, out_dir: Path) -> None:
    """The decisive figure: what each validation promises against the 2024 test.

    The 2024 curve is a post-hoc diagnostic, drawn after the choices: the test scores only the k
    that each validation picked.

    Args:
        curves: Output of ``validation_curves``.
        out_dir: Figure directory.
    """
    series = [
        ("random", "random folds"),
        ("bridge", "bridges held out"),
        ("county", "counties held out"),
        (f"test_{TEST_YEAR}", f"{TEST_YEAR} inventory (post hoc)"),
    ]
    raw, best = curves.raw, curves.best
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for c, (key, label) in enumerate(series):
        ax.plot(
            KS,
            raw[key],
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            marker=aml_style.MARKERS[c],
            ms=6,
            label=label,
            lw=3.0 if key.startswith("test") else 2.2,
        )
    random_curve = raw["random"]
    kb = best["random"]
    ax.annotate(
        f"random folds pick k = {kb}",
        (kb, random_curve[k_index(kb)]),
        xytext=(4.0, 0.80),
        fontsize=12,
        va="center",
        arrowprops={"arrowstyle": "->", "color": aml_style.INK},
    )
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("RMSE (rating points)")
    ax.set_ylim(0.75, 1.5)
    ax.legend(loc="upper right", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "k_curves")


def drift(df: pd.DataFrame, curves: Curves, out_dir: Path) -> dict[str, object]:
    """Mean rating by age in 2014 and 2024; error by forecast horizon at k = 1 and ``K``.

    The later inventories are scored at k = 1, a k no validation chose: a post-hoc diagnostic
    of how a copied inspection goes stale, not a model to use.

    Args:
        df: All records.
        curves: Output of ``validation_curves``.
        out_dir: Figure directory.

    Returns:
        Mean rating per age group in 2014 and 2024, and the RMSE and bias per later inventory.
    """
    labels = list(AGE_LABELS)
    mids = np.arange(len(labels))
    out: dict[str, object] = {"age_bins": labels}
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for c, year in enumerate((2014, TEST_YEAR)):
        rec = df[df["nbi_year"] == year]
        grp = pd.cut(rec["age"], list(AGE_BINS), right=False, labels=labels, include_lowest=True)
        mean = rec.groupby(grp, observed=False)["rating"].mean()
        out[f"mean_by_age_{year}"] = [rnd(v, 2) for v in mean]
        ax.plot(
            mids,
            mean.to_numpy(),
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            marker=aml_style.MARKERS[c],
            label=f"{year} inventory",
        )
    ax.set_xticks(mids, labels, rotation=30)
    ax.set_xlabel("age at inspection (years)")
    ax.set_ylabel("mean condition rating")
    ax.set_ylim(4.5, 8.2)
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "drift")
    i1, iK = k_index(1), k_index(K)
    raw = curves.raw
    for year in LATER_YEARS:
        t, b = raw[f"test_{year}"], raw[f"bias_{year}"]
        out[f"rmse_k1_{year}"] = rnd(t[i1], 2)
        out[f"rmse_k{K}_{year}"] = rnd(t[iK], 2)
        out[f"bias_k1_{year}"] = rnd(b[i1], 2)
        out[f"bias_k{K}_{year}"] = rnd(b[iK], 2)
    return out


def leakage(train: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Add columns that the rating itself decides; county-held-out and random CV at ``K``.

    Args:
        train: Training records.
        out_dir: Figure directory.

    Returns:
        RMSE for each set of columns (counties held out and random folds, k = 1 and ``K``), the
        mean rating per inspection interval, and how often the structural evaluation is at or
        below the lower of the superstructure and substructure ratings.
    """
    rec = train.dropna(subset=list(LEAKY)).reset_index(drop=True)  # both columns present
    y = rec["rating"].to_numpy(float)
    options = [
        ("inventory only", ()),
        ("+ inspection interval", ("inspection_interval",)),
        ("+ structural evaluation", ("structural_eval",)),
        ("+ both", tuple(LEAKY)),
    ]
    out: dict[str, object] = {"records": int(len(rec)), "dropped": int(len(train) - len(rec))}
    county_rmse, random_rmse = [], []
    for name, extra in options:
        X = encode(rec, extra)
        for scheme, store in (("county", county_rmse), ("random", random_rmse)):
            pred = cv_predict(X, y, make_folds(rec, scheme), np.array([1, K]))
            store.append(rmse_per_k(pred, y))
        out[name] = {
            "county_k50": rnd(county_rmse[-1][1], 2),
            "county_k1": rnd(county_rmse[-1][0], 2),
            "random_k1": rnd(random_rmse[-1][0], 2),
            "random_k50": rnd(random_rmse[-1][1], 2),
        }
    by_interval = rec.groupby("inspection_interval")["rating"].agg(["mean", "size"])
    out["mean_rating_by_interval"] = {  # 1 decimal, as the slide quotes it
        str(int(i)): rnd(m, 1)
        for i, m in by_interval["mean"].items()
        if by_interval.at[i, "size"] >= MIN_INTERVAL_RECORDS
    }
    lowest = rec[["superstructure", "substructure"]].min(axis=1)
    out["structural_eval_le_min_sup_sub_pct"] = pct((rec["structural_eval"] <= lowest).mean(), 0)

    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    pos = np.arange(len(options))[::-1]
    for p, v in zip(pos, county_rmse, strict=True):
        ax.plot(v[1], p, marker="o", ms=10, ls="none", color=aml_style.PALETTE[0])
        ax.text(v[1], p + 0.25, f"{v[1]:.2f}", ha="center", fontsize=12)
    ax.set_yticks(pos, [o[0] for o in options])
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.6, len(options) - 0.3)
    ax.set_xlabel(f"RMSE, counties held out, k = {K}")
    aml_style.save(fig, out_dir, "leakage")
    return out


def screening(
    train: pd.DataFrame, df: pd.DataFrame, curves: Curves, out_dir: Path
) -> dict[str, object]:
    """Flag the records predicted worst; how many poor ones does that catch?

    Ranks records by predicted rating (ties broken at random, seeded) and flags the lowest
    ``FLAG_SHARE``. Compares what random-fold validation promises for the k it picks (its
    out-of-fold predictions on the training records, several inspections per bridge) with what
    the 2024 test delivers (one record per bridge), for that k and for k = ``K``.

    Args:
        train: Training records.
        df: All records.
        curves: Output of ``validation_curves``.
        out_dir: Figure directory.

    Returns:
        The poor shares and counts, and the share of poor records caught at ``FLAG_SHARE``.
    """
    oof, later, best = curves.oof, curves.later, curves.best
    rng = np.random.default_rng(SEED)
    y_tr = train["rating"].to_numpy(float)
    y_te = later_records(df, train, TEST_YEAR)["rating"].to_numpy(float)
    kr = int(best["random"])
    cases = {
        "promised": (f"promised by random folds, k = {kr}", oof["random"][:, k_index(kr)], y_tr),
        "delivered": (f"{TEST_YEAR}, k = {kr}", later[str(TEST_YEAR)][:, k_index(kr)], y_te),
        "delivered_k50": (f"{TEST_YEAR}, k = {K}", later[str(TEST_YEAR)][:, k_index(K)], y_te),
    }
    out: dict[str, object] = {
        "flag_share_pct": round(100 * FLAG_SHARE),
        "poor_test_bridges": int((y_te <= POOR).sum()),
        "poor_train_pct": pct((y_tr <= POOR).mean(), 1),
    }
    curves_caught: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for key, (_, pred, y) in cases.items():
        order = np.lexsort((rng.random(len(pred)), pred))  # lowest prediction first
        caught = np.cumsum(y[order] <= POOR) / max(int((y <= POOR).sum()), 1)
        at = int(round(FLAG_SHARE * len(y))) - 1
        curves_caught[key] = (np.arange(1, len(y) + 1) / len(y), caught, at)
        out[f"{key}_pct"] = pct(caught[at], 0)
    plot_screening({cases[k][0]: v for k, v in curves_caught.items()}, out_dir)
    return out


def plot_screening(
    curves_caught: dict[str, tuple[np.ndarray, np.ndarray, int]], out_dir: Path
) -> None:
    """Share of poor records caught against the share flagged, one curve per case.

    Args:
        curves_caught: Label to (share flagged, share of poor caught, index at ``FLAG_SHARE``).
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for c, (label, (flagged, caught, at)) in enumerate(curves_caught.items()):
        ax.plot(
            100 * flagged,
            100 * caught,
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            label=label,
        )
        ax.plot(
            100 * flagged[at],
            100 * caught[at],
            marker=aml_style.MARKERS[c],
            ms=9,
            color=aml_style.PALETTE[c],
            ls="none",
        )
    ax.axvline(100 * FLAG_SHARE, color=aml_style.MUTED, ls=":", lw=1.2)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 100)
    ax.set_xlabel("records flagged for inspection (%)")
    ax.set_ylabel("poor records caught (%)")
    ax.legend(loc="lower right", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "screening")


@dataclass(frozen=True)
class CountyErrors:
    """The error per county, counties held out, k = ``K``.

    Attributes:
        numbers: The best and worst of the counties with ``MIN_COUNTY_RECORDS`` or more records,
            Cook County's error, their spread, and its noise floor, for numbers.json.
        stats: Per county (``COUNTY_FOLD`` code): ``rmse`` and the number of records ``n``.
    """

    numbers: dict[str, object]
    stats: pd.DataFrame


def county_noise_floor(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-county RMSE when bridges are dealt to counties at random.

    Each draw permutes the county labels over the bridges, so every county keeps its number of
    bridges, and every bridge keeps its records and their squared errors. What the per-county
    RMSE varies under it is the spread that chance alone gives.

    Args:
        frame: One row per record: ``county``, ``bridge``, and the squared error ``sq``.

    Returns:
        Per draw (``NULL_PERMUTATIONS`` of them): the lowest and highest RMSE and the standard
        deviation (ddof = 1) over the counties with at least ``MIN_COUNTY_RECORDS`` records.
    """
    per_bridge = frame.groupby("bridge", sort=True).agg(
        county=("county", "first"), sq=("sq", "sum"), n=("sq", "size")
    )
    codes, county_idx = np.unique(per_bridge["county"].to_numpy(), return_inverse=True)
    sq, n = per_bridge["sq"].to_numpy(), per_bridge["n"].to_numpy(float)
    rng = np.random.default_rng((SEED, 6))
    dealt = rng.permuted(np.tile(county_idx, (NULL_PERMUTATIONS, 1)), axis=1)
    assert dealt.shape == (NULL_PERMUTATIONS, len(per_bridge)), dealt.shape
    lo, hi, sd = (np.empty(NULL_PERMUTATIONS) for _ in range(3))
    for d, g in enumerate(dealt):
        n_c = np.bincount(g, weights=n, minlength=len(codes))
        r_c = np.sqrt(np.bincount(g, weights=sq, minlength=len(codes)) / np.maximum(n_c, 1))
        r_c = r_c[n_c >= MIN_COUNTY_RECORDS]
        lo[d], hi[d], sd[d] = r_c.min(), r_c.max(), r_c.std(ddof=1)
    return {"min": lo, "max": hi, "sd": sd}


def county_errors(train: pd.DataFrame, curves: Curves) -> CountyErrors:
    """Counties held out, k = ``K``: how the error varies from county to county.

    A bridge counts in the county of its fold (``COUNTY_FOLD``). The spread across counties
    is set against ``county_noise_floor``: the same errors with bridges dealt to counties at
    random.

    Args:
        train: Training records.
        curves: Output of ``validation_curves`` (its out-of-fold predictions).

    Returns:
        The per-county statistics (for the map) and the numbers for the slide.
    """
    err = (curves.oof["county"][:, k_index(K)] - train["rating"].to_numpy(float)) ** 2
    frame = pd.DataFrame(
        {
            "county": train[COUNTY_FOLD].to_numpy(),
            "bridge": train["structure"].to_numpy(),
            "sq": err,
        }
    )
    per = frame.groupby("county")["sq"]
    stats = pd.DataFrame({"rmse": np.sqrt(per.mean()), "n": per.size()})
    big = stats[stats["n"] >= MIN_COUNTY_RECORDS].sort_values("rmse")
    lo, hi = float(big["rmse"].iloc[0]), float(big["rmse"].iloc[-1])
    sd = float(big["rmse"].std(ddof=1))
    null = county_noise_floor(frame)
    null_sd = float(np.median(null["sd"]))
    numbers: dict[str, object] = {
        "counties_300_plus": int(len(big)),
        "best": {"county": COUNTY[big.index[0]], "rmse": rnd(lo, 2)},
        "worst": {"county": COUNTY[big.index[-1]], "rmse": rnd(hi, 2)},
        "cook": rnd(stats.at["031", "rmse"], 2),
        "rmse_sd": rnd(sd, 2),
        "null": {
            "permutations": NULL_PERMUTATIONS,
            "sd_median": rnd(null_sd, 2),
            "sd_p95": rnd(np.percentile(null["sd"], 95), 2),
            "min_median": rnd(np.median(null["min"]), 2),
            "min_interval": [rnd(v, 2) for v in np.percentile(null["min"], [2.5, 97.5])],
            "max_median": rnd(np.median(null["max"]), 2),
            "max_interval": [rnd(v, 2) for v in np.percentile(null["max"], [2.5, 97.5])],
            "draws_min_at_or_below_best_pct": pct(float((null["min"] <= lo).mean()), 1),
            "draws_max_at_or_above_worst_pct": pct(float((null["max"] >= hi).mean()), 1),
            "sd_ratio": rnd(sd / null_sd, 1),
        },
    }
    return CountyErrors(numbers=numbers, stats=stats)


# ---------------------------------------------------------------------------------------------
# The students' attempt, the traced bridge, and the tiny worked example


def describe(rec: pd.Series, q: pd.Series) -> dict[str, object]:
    """One table row for the attempt.

    Args:
        rec: A training record.
        q: The traced bridge Q's record (for the distance).

    Returns:
        The record's identity, inputs, distance to Q, and ratings.
    """
    return {
        "nbi_year": int(rec["nbi_year"]),
        "structure": rec["structure"],
        "county": COUNTY[rec["county"]],
        "facility": rec["facility"],
        "crosses": rec["crosses"],
        "built": int(rec["year_built"]),
        "rebuilt": None if pd.isna(rec["year_reconstructed"]) else int(rec["year_reconstructed"]),
        "design": f"{rec['material']} {DESIGN.get(rec['design'], 'other')}",
        "adt": int(rec["adt"]),
        "length_m": rnd(rec["length_m"], 0),
        "km_from_q": rnd(np.hypot(rec["x_km"] - q["x_km"], rec["y_km"] - q["y_km"]), 1),
        "rating": int(rec["rating"]),
        "deck": int(rec["deck"]),
        "superstructure": int(rec["superstructure"]),
        "substructure": int(rec["substructure"]),
    }


FoldNeighbors = Callable[[int, str, int], np.ndarray]


def fold_neighbors(
    train: pd.DataFrame,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict[str, np.ndarray], FoldNeighbors]:
    """Random and county folds, the fold of each record, and a neighbor search within them.

    Args:
        train: Training records.

    Returns:
        The folds by scheme, each record's fold by scheme, and ``neighbors(row, scheme, k)``:
        the ``k`` training rows nearest to ``row``, nearest first, with ``row``'s fold held out.
    """
    X, y = encode(train), train["rating"].to_numpy(float)
    folds = {s: make_folds(train, s) for s in ("random", "county")}
    fold_of = {s: np.empty(len(train), dtype=int) for s in folds}
    for s, fs in folds.items():
        for j, (_, te) in enumerate(fs):
            fold_of[s][te] = j

    def neighbors(row: int, scheme: str, k: int) -> np.ndarray:
        """The ``k`` training rows nearest to ``row``, with ``row``'s fold held out.

        Args:
            row: A training record's row.
            scheme: ``"random"`` or ``"county"``.
            k: Number of neighbors.

        Returns:
            Their rows, nearest first.
        """
        tr, _ = folds[scheme][fold_of[scheme][row]]
        _, idx, _ = holdout_predict(X[tr], y[tr], X[[row]], ks=np.array([k]))
        return tr[idx[0]]

    return folds, fold_of, neighbors


def pick_q(train: pd.DataFrame, neighbors: FoldNeighbors, held: set[int]) -> tuple[int, int, int]:
    """Choose the traced bridge Q by the stated rule (see ``attempt_and_trace``).

    Args:
        train: Training records.
        neighbors: The search from ``fold_neighbors``.
        held: Rows of the first county block.

    Returns:
        Q's row, its twin's row, and the number of candidates the rule left.
    """
    # find_twins returns index labels; they index the arrays below because train's index is
    # the RangeIndex that unique_inspections gives it (asserted in main).
    y = train["rating"].to_numpy(float)
    st, adt = train["structure"].to_numpy(), train["adt"].to_numpy()
    tw = find_twins(train[train["nbi_year"] == 2018])
    keep = tw["row"].isin(held) & (y[tw["row"]] <= POOR) & (adt[tw["row"]] >= Q_MIN_ADT)
    cand = tw[keep].copy()
    cand["r1"] = [int(neighbors(r, "random", 1)[0]) for r in cand["row"]]
    cand["c1"] = [int(neighbors(r, "county", 1)[0]) for r in cand["row"]]
    cand = cand[st[cand["r1"]] == st[cand["twin_row"]]]  # nearest is the twin
    cand["gap"] = np.abs(y[cand["r1"]] - y[cand["c1"]])
    cand["adt"] = adt[cand["row"]]
    q_row = int(cand.sort_values(["gap", "adt"], ascending=False)["row"].iloc[0])
    return q_row, int(tw.set_index("row").at[q_row, "twin_row"]), int(len(cand))


def trace_q(
    train: pd.DataFrame, q_row: int, twin_row: int, near_rand: np.ndarray, near_cty: np.ndarray
) -> dict[str, object]:
    """Q's history, inputs, and nearest records under random folds and with its county out.

    Args:
        train: Training records.
        q_row: Q's row.
        twin_row: Its twin's row.
        near_rand: Q's ``K`` nearest rows under random folds, nearest first.
        near_cty: The same with Q's county held out.

    Returns:
        The trace for the slides; ``_raw`` holds unrounded values for ``numbers.tex``.
    """
    y, st = train["rating"].to_numpy(float), train["structure"].to_numpy()
    q = train.loc[q_row]
    km = np.hypot(train["x_km"].to_numpy() - q["x_km"], train["y_km"].to_numpy() - q["y_km"])
    history = train[train["structure"] == q["structure"]].sort_values("nbi_year")

    def kind(r: int) -> str:
        """Whether a neighbor is Q's own record, its twin's, or another bridge's.

        Args:
            r: The neighbor's row.

        Returns:
            ``"own"``, ``"twin"``, or ``"other"``.
        """
        return "own" if st[r] == q["structure"] else "twin" if st[r] == st[twin_row] else "other"

    years = zip(history["nbi_year"], history["rating"], strict=True)
    return {
        "q": describe(q, q),
        "q_row": q_row,
        "q_history": {str(int(a)): int(b) for a, b in years},
        "q_inputs": {
            "age": int(q["age"]),
            "years_since_work": int(q["years_since_work"]),
            "log_adt": rnd(q["log_adt"], 1),
            "pct_truck": int(q["pct_truck"]),  # NBI Item 109: a whole percentage
        },
        "random_k1": int(y[near_rand[0]]),
        "county_k1": int(y[near_cty[0]]),
        "random_k1_m": int(round(1000 * float(km[near_rand[0]]), -1)),
        "county_k1_km": rnd(km[near_cty[0]], 1),
        "county_k1_county": COUNTY[train.at[int(near_cty[0]), "county"]],
        "random_k5": [kind(int(r)) for r in near_rand[:5]],
        "random_k5_mean": rnd(y[near_rand[:5]].mean(), 2),
        "county_k5_mean": rnd(y[near_cty[:5]].mean(), 2),
        "random_k50": rnd(y[near_rand].mean(), 2),
        "county_k50": rnd(y[near_cty].mean(), 2),
        "random_k50_own_or_twin": int(sum(kind(int(r)) != "other" for r in near_rand)),
        "county_k50_nearest_km": rnd(km[near_cty].min(), 2),
        "n_train": int(len(train)),
        "_raw": {"log_adt": float(q["log_adt"]), "county_k1_km": float(km[near_cty[0]])},
    }


def attempt_and_trace(train: pd.DataFrame) -> tuple[dict[str, object], int]:
    """Pick the traced bridge Q, its attempt table, and its trace through both splits.

    Q is chosen by a stated rule: among poor twin bridges (2018 records) with at least
    ``Q_MIN_ADT`` vehicles a day in the first county block whose nearest record under random
    folds is their twin, the one whose 1-NN predictions under the two splits differ most; ties
    go to the busier bridge. The attempt shows Q's two nearest training records under random
    folds and its two nearest with its county held out.

    Args:
        train: Training records.

    Returns:
        The attempt and trace, and Q's row in ``train``.
    """
    st = train["structure"].to_numpy()
    folds, fold_of, neighbors = fold_neighbors(train)
    q_row, twin_row, n_cand = pick_q(train, neighbors, set(folds["county"][0][1]))
    near_rand, near_cty = neighbors(q_row, "random", K), neighbors(q_row, "county", K)
    rows = [int(near_rand[0]), int(near_rand[1]), int(near_cty[0]), int(near_cty[1])]
    assert rows[0] == twin_row or st[rows[0]] == st[twin_row], "random 1-NN is not the twin"
    assert len(set(rows)) == 4, rows
    q = train.loc[q_row]
    table = [describe(train.loc[r], q) | {"letter": c} for r, c in zip(rows, "ABCD", strict=True)]
    trace = trace_q(train, q_row, twin_row, near_rand, near_cty) | {
        "n_train_random": int(len(folds["random"][fold_of["random"][q_row]][0])),
        "n_train_county": int(len(folds["county"][0][0])),
        "candidates": n_cand,
    }
    estimates = {
        "with_all": rnd(np.mean([t["rating"] for t in table]), 2),
        "without_ab": rnd(np.mean([t["rating"] for t in table[2:]]), 2),
    }
    return {"table": table, "trace": trace, "estimates": estimates}, q_row


def tiny_loo(train: pd.DataFrame, attempt: dict[str, object], q_row: int) -> dict[str, object]:
    """Leave-one-out 1-NN on the attempt's five records: by record, then by county.

    Distances use the scaler of all training records. Leaving out by county holds out every
    record of the left-out record's county at once (Q, A, and B share Cook County).

    Args:
        train: Training records.
        attempt: Output of ``attempt_and_trace`` (its table of A to D).
        q_row: Q's row.

    Returns:
        Each record's nearest and 1-NN prediction both ways, and the two RMSEs.
    """
    table = attempt["table"]
    assert isinstance(table, list)
    rows = [q_row] + [
        int(
            train.index[
                (train["structure"] == t["structure"]) & (train["nbi_year"] == t["nbi_year"])
            ][0]
        )
        for t in table
    ]
    names = ["Q"] + [t["letter"] for t in table]
    sc = StandardScaler().fit(encode(train))
    Z = sc.transform(encode(train.loc[rows]))
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    assert d.shape == (5, 5), d.shape
    y = train.loc[rows, "rating"].to_numpy(float)
    county = train.loc[rows, "county"].to_numpy()
    out_rows = []
    for i in range(5):
        by_rec = np.where(np.arange(5) == i, np.inf, d[i])
        by_cty = np.where(county == county[i], np.inf, d[i])
        j, m = int(np.argmin(by_rec)), int(np.argmin(by_cty))
        out_rows.append(
            {
                "record": names[i],
                "rating": int(y[i]),
                "rec_nearest": names[j],
                "rec_pred": int(y[j]),
                "cty_nearest": names[m],
                "cty_pred": int(y[m]),
            }
        )
    err_rec = np.array([r["rec_pred"] - r["rating"] for r in out_rows], dtype=float)
    err_cty = np.array([r["cty_pred"] - r["rating"] for r in out_rows], dtype=float)
    return {
        "rows": out_rows,
        "rmse_by_record": rnd(np.sqrt(np.mean(err_rec**2)), 2),
        "rmse_by_county": rnd(np.sqrt(np.mean(err_cty**2)), 2),
    }


def signed_error(err: int) -> str:
    """A prediction error with its sign: +2, −1, and ±0 for an exact prediction.

    Args:
        err: Prediction minus rating, whole points.

    Returns:
        The signed text for the diagram.
    """
    return "±0" if err == 0 else tex_num(err, "+d")


def write_tex(attempt: dict[str, object], out_dir: Path) -> Path:
    """Write the trace of Q as LaTeX macros for the TikZ diagram.

    Args:
        attempt: Output of ``attempt_and_trace``.
        out_dir: Figure directory.

    Returns:
        The path written.
    """
    tr = attempt["trace"]
    assert isinstance(tr, dict)
    q, inp, raw = tr["q"], tr["q_inputs"], tr["_raw"]
    heading = {"NB": "northbound", "SB": "southbound", "EB": "eastbound", "WB": "westbound"}
    words = []
    for w in re.sub(r"I-\s*(\d+)", r"I-\1", q["facility"]).split()[:3]:  # "IL  53 NB ..."
        words.append(heading.get(w, w))
        if w in heading:
            break
    rebuilt = "never rebuilt" if q["rebuilt"] is None else f"rebuilt {q['rebuilt']}"
    macros = {
        "QFacility": " ".join(words),
        "QCounty": q["county"],
        "QBuilt": tex_num(q["built"], "d"),
        "QRebuilt": rebuilt,
        "QADT": tex_num(q["adt"], ",d"),
        "QAge": tex_num(inp["age"], "d"),
        "QLogADT": tex_num(raw["log_adt"], ".1f"),
        "QRating": tex_num(q["rating"], "d"),
        "RandomPred": tex_num(tr["random_k1"], "d"),
        "RandomErr": signed_error(tr["random_k1"] - q["rating"]),
        "RandomMeters": tex_num(tr["random_k1_m"], "d"),
        "CountyPred": tex_num(tr["county_k1"], "d"),
        "CountyErr": signed_error(tr["county_k1"] - q["rating"]),
        "CountyKm": tex_num(raw["county_k1_km"], ".0f"),
        "CountyName": tr["county_k1_county"],
        "NTrain": tex_num(tr["n_train"], ",d"),
        "Poor": tex_num(POOR, "d"),
    }
    return write_tex_macros(macros, out_dir, "cee_l03_generalization.py")


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

RESAMPLE_N: Final[int] = 1_000  # bridges per training sample in the resampling picture
RESAMPLE_DRAWS: Final[int] = 6  # training samples drawn in it
RESAMPLE_KS: Final[tuple[int, ...]] = (1, 30, 1_000)  # k of its three panels
AGE_MAX: Final[int] = 110  # years shown in it
RESAMPLE_YLIM: Final[tuple[float, float]] = (-0.5, 9.5)  # every rating 0-9: k = 1 reaches both


def title_art_figure(curves: Curves, out_dir: Path) -> None:
    """The title art: a portal split by a crimson blade, points kept and held out.

    The horizon is the lecture's validation curve over k (counties held out): high at k = 1,
    lowest between, rising again: variance falling into bias.

    Args:
        curves: Output of ``validation_curves`` (its raw county curve).
        out_dir: Figure directory.
    """
    c = title_art.Canvas(seed=3)
    c.ground(("#18100a", "#1f0f12", "#0f0d1e", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.92, 0.95, 4), soft=4), "#5a2216", 0.82)
    c.glaze(c.polygon([(0.24, 0.06), (0.76, 0.06), (0.76, 0.86), (0.24, 0.86)], soft=3),
            "#0f0c1c", 0.92)  # fmt: skip
    rim = c.disc((0.50, 1.15), 0.12, soft=4) - c.disc((0.50, 1.15), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eba44a", 0.5, spread=36)
    c.glaze(c.disc((0.50, 1.15), 0.10, soft=3), "#200a10", 0.92)
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.28, 0.72, 60), rng.uniform(0.46, 0.82, 60)])
    left = pts[:, 0] < 0.5
    c.glow(np.clip(sum(c.disc(tuple(q), 0.006, soft=1.5) for q in pts[left]), 0, 1), "#9fb7c9",
           0.5, spread=6)  # fmt: skip
    c.glow(np.clip(sum(c.disc(tuple(q), 0.007, soft=1.5) for q in pts[~left]), 0, 1), "#f0a03c",
           0.6, spread=8)  # fmt: skip
    c.glaze(c.polygon([(0.494, 0.44), (0.506, 0.44), (0.512, 1.02), (0.488, 1.02)], soft=1.5),
            "#c2361f", 0.9)  # fmt: skip
    err = np.asarray(curves.raw["county"])
    xs = 0.28 + 0.44 * np.arange(len(err)) / (len(err) - 1)
    ys = 0.12 + 0.26 * (err - err.min()) / np.ptp(err)
    c.glow(c.lines([list(zip(xs, ys, strict=True))], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean portal in rust frames a dark "
        "doorway; inside, pale points on the left and amber points on the right are split by a "
        "thin crimson blade; a dark sun with a burning rim hangs above, and a golden curve, the "
        "lecture's validation error falling and rising with k, glows at the doorway's foot.",
    )


def ratings_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """How many bridges of the 2018 inventory hold each condition rating; poor in orange.

    The whole inventory (one record per bridge), not the training set: training keeps only the
    inspections new in 2018, which leaves out many bridges on a 48-month interval and so
    over-represents poor ones.

    Args:
        df: All bridge-year records.
        out_dir: Figure directory.

    Returns:
        The bridges, the count at each rating 0 to 9, and the poor share.
    """
    rec = df[df["nbi_year"] == max(TRAIN_YEARS)]
    assert rec["structure"].is_unique, "one record per bridge in an inventory"
    assert (rec["rating"] % 1 == 0).all(), "ratings are whole numbers"
    counts = rec["rating"].astype(int).value_counts().reindex(range(10), fill_value=0)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    poor = counts.index <= POOR
    ax.bar(counts.index[~poor], counts.values[~poor], color=aml_style.PALETTE[0])
    ax.bar(counts.index[poor], counts.values[poor], color=aml_style.PALETTE[1], hatch="//",
           edgecolor="white")  # fmt: skip
    ax.set(xlabel="lowest of deck, superstructure, substructure", ylabel="bridges, 2018",
           xticks=range(10))  # fmt: skip
    share = float((rec["rating"] <= POOR).mean())
    ax.set_title(f"{100 * share:.1f}% poor (rating 4 or less)", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "ratings")
    return {
        "bridges_2018": int(len(rec)),
        "count_by_rating": {str(r): int(n) for r, n in counts.items()},
        "poor_pct_2018": pct(share, 1),
    }


def resample_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """k-NN on one input, age, fitted to six samples of 1,000 bridges: k = 1, 30, 1,000.

    The samples come from the whole 2018 inventory; the dashed line is the mean rating at each
    age over all of its bridges (smoothed over five years). Small k follows each sample's noise
    (variance); k = 1,000 averages every bridge (bias).

    Args:
        df: All bridge-year records.
        out_dir: Figure directory.

    Returns:
        Per k: the variance of the six fits and their bias² against the dashed line, and the
        range of the fits; the dashed line at both ends of the age axis.
    """
    rec = df[df["nbi_year"] == max(TRAIN_YEARS)]
    age, rating = rec["age"].to_numpy(float), rec["rating"].to_numpy(float)
    grid = np.arange(0, AGE_MAX + 1, dtype=float)
    truth = pd.Series(rating).groupby(np.clip(age, 0, AGE_MAX).round()).mean()
    truth = truth.reindex(grid).interpolate().rolling(5, center=True, min_periods=1).mean()
    rng = np.random.default_rng((SEED, 3))
    samples = [rng.choice(len(rec), RESAMPLE_N, replace=False) for _ in range(RESAMPLE_DRAWS)]
    fig, axes = plt.subplots(1, len(RESAMPLE_KS), figsize=aml_style.FULL, sharey=True)
    spread: dict[str, float] = {}
    bias2_by_k: dict[str, float] = {}
    fit_range: dict[str, list[float]] = {}
    for ax, k in zip(axes, RESAMPLE_KS, strict=True):
        fits = np.array([
            KNeighborsRegressor(n_neighbors=k).fit(age[i, None], rating[i]).predict(grid[:, None])
            for i in samples
        ])  # fmt: skip
        assert fits.shape == (RESAMPLE_DRAWS, len(grid)), fits.shape
        for f in fits:
            ax.plot(grid, f, color=aml_style.PALETTE[0], lw=1.1, ls="-", alpha=0.55)
        ax.plot(grid, truth.to_numpy(), color=aml_style.INK, lw=2.2, ls="--")
        spread[str(k)] = rnd(float(fits.var(axis=0).mean()), 2)
        bias2 = float(((fits.mean(axis=0) - truth.to_numpy()) ** 2).mean())
        bias2_by_k[str(k)] = rnd(bias2, 2)
        fit_range[str(k)] = [rnd(fits.min(), 2), rnd(fits.max(), 2)]
        if fits.min() < RESAMPLE_YLIM[0] or fits.max() > RESAMPLE_YLIM[1]:
            raise RuntimeError(f"k = {k}: fits {fits.min()}-{fits.max()} leave the y-axis")
        # The numbers go above the panel: at k = 1 the fits reach every corner of it.
        ax.set_title(f"k = {k:,}", loc="left", pad=22)
        ax.text(0.0, 1.02, f"variance {fits.var(axis=0).mean():.2f}, bias² {bias2:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
                color=aml_style.MUTED)  # fmt: skip
        ax.set(xlabel="age (years)", xlim=(0, AGE_MAX), ylim=RESAMPLE_YLIM)
    axes[0].set_ylabel("predicted rating")
    aml_style.save(fig, out_dir, "resample")
    return {
        "variance_by_k": spread,
        "bias2_by_k": bias2_by_k,
        "fit_range_by_k": fit_range,
        "truth_age_0": rnd(truth.iloc[0], 2),
        "truth_age_max": rnd(truth.iloc[-1], 2),
        "age_max": AGE_MAX,
        "samples": RESAMPLE_DRAWS,
        "bridges_per_sample": RESAMPLE_N,
    }


def county_map(train: pd.DataFrame, county: CountyErrors, out_dir: Path) -> None:
    """Counties held out, k = 50: each county's RMSE at the centre of its bridges.

    Args:
        train: Training records (for the centres).
        county: Output of ``county_errors``.
        out_dir: Figure directory.
    """
    stats = county.stats
    centre = train.groupby(COUNTY_FOLD)[["lon", "lat"]].mean().loc[stats.index]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    sc = ax.scatter(centre["lon"], centre["lat"], c=stats["rmse"], s=12 + stats["n"] / 40,
                    cmap="cividis", edgecolor="white", lw=0.4)  # fmt: skip
    # Labels below the map with leaders: inside it they would sit on other counties' dots.
    for key, dx in (("best", -25), ("worst", 25)):
        info = county.numbers[key]
        assert isinstance(info, dict)
        code = next(c for c, name in COUNTY.items() if name == info["county"])
        x, y = centre.loc[code]
        ax.scatter([x], [y], s=12 + stats.at[code, "n"] / 40, facecolor="none",
                   edgecolor=aml_style.INK, lw=1.6)  # fmt: skip
        ax.annotate(f"{info['county']}: {info['rmse']:.2f}", (x, y),
                    xytext=(dx, -(y - centre["lat"].min()) * 40 - 30), textcoords="offset points",
                    fontsize=12, ha="right" if dx < 0 else "left", va="top",
                    arrowprops={"arrowstyle": "-", "color": aml_style.INK, "lw": 0.8})  # fmt: skip
    fig.colorbar(sc, ax=ax, label="RMSE (rating points)", shrink=0.85)
    ax.set(xticks=[], yticks=[], aspect=1 / np.cos(np.deg2rad(40.0)))
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    aml_style.save(fig, out_dir, "county_map")


def main() -> None:
    """Generate every figure and number for CEE Lecture 3.

    Raises:
        RuntimeError: If the chosen k do not order as the slides say, or ``K`` is not within
            0.01 of both honest minima.
    """
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = add_county_fold(load_bridges())
    train = unique_inspections(df[df["nbi_year"].isin(TRAIN_YEARS)])
    # Row labels double as positions in the numpy arrays below (find_twins returns labels).
    assert train.index.equals(pd.RangeIndex(len(train)))
    curves = validation_curves(train, df)
    best, raw = curves.best, curves.raw
    if not (best["random"] <= 3 and best["bridge"] <= K <= best["county"]):
        raise RuntimeError(f"the chosen k changed order: {best}")
    for scheme in ("bridge", "county"):  # K is within 0.01 of both honest minima
        if raw[scheme][k_index(K)] - raw[scheme].min() >= 0.01:
            raise RuntimeError(f"K = {K} is not within 0.01 of the {scheme} minimum")
    base, floor = baselines(train), twin_floor(df)
    attempt, q_row = attempt_and_trace(train)
    county = county_errors(train, curves)
    numbers: dict[str, object] = {
        "data": data_summary(df, train),
        "curves": curves.numbers,
        "baseline": base,
        "floor": floor,
        "overlap": fold_overlap(train),
        "verify_listing": verify_listing(train),
        "bias_variance": bias_variance(train, out_dir),
        "folds_map": folds_map(train, out_dir),
        "drift": drift(df, curves, out_dir),
        "leakage": leakage(train, out_dir),
        "screening": screening(train, df, curves, out_dir),
        "county_errors": county.numbers,
        "attempt": attempt,
        "tiny_loo": tiny_loo(train, attempt, q_row),
        "K": K,
        "cited": CITED,
        "recall_l02": RECALL_L02,
    }
    bracket_figure(curves, base, floor, out_dir)
    k_curves_figure(curves, out_dir)
    # the visuals that carry the slides
    title_art_figure(curves, out_dir)
    numbers["ratings"] = ratings_figure(df, out_dir)
    numbers["resample"] = resample_figure(df, out_dir)
    county_map(train, county, out_dir)
    aml_style.thumbnails(out_dir, ("resample", "folds_map", "k_curves", "leakage"))
    write_numbers(numbers, out_dir)
    print(write_tex(attempt, out_dir))


if __name__ == "__main__":
    main()
