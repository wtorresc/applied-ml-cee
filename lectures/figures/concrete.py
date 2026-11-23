"""The UCI concrete data and the conventions Lectures 1, 2, 5, and 10 share.

Data: ``data/derived/Concrete_Data.csv`` from ``scripts/fetch_concrete.sh`` (UCI Concrete
Compressive Strength; Yeh, 1998; CC BY 4.0). One loader, one encoder, and one set of splits, so
that "Lecture 2's splits" means the same 30 splits in every lecture that compares with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, require

CONCRETE_CSV: Path = DATA_DIR / "derived" / "Concrete_Data.csv"
CONCRETE_COLUMNS: list[str] = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
    "strength",
]
CONCRETE_INGREDIENTS: list[str] = CONCRETE_COLUMNS[:7]
RAW_INPUTS: list[str] = [*CONCRETE_INGREDIENTS, "age"]
GROUP_NAMES: list[str] = ["w/b", "slag / b", "fly ash / b", "SP / b", "aggregate / b", "ln age"]
FC: float = 35.0  # specified 28-day strength f'c, MPa (about 5,000 psi)
PROPOSED_ROW: int = 638  # the proposed design traced in Lectures 5 and 10

SPLIT_SEED: int = 0
REPEATS: int = 30  # repeated splits for every comparison (Lecture 2)
K: int = 5  # Lecture 2's k-NN
TEST_SIZE: float = 0.25


# ---------------------------------------------------------------------------------------------
# Data


def load_concrete(path: Path = CONCRETE_CSV, tol: float = 1.0) -> pd.DataFrame:
    """Load the UCI Concrete Compressive Strength data (Yeh, 1998).

    Ingredients are in kg per m^3 of concrete, age in days, strength in MPa. The same mix (same
    seven ingredients) is tested at several ages, and nearly identical designs recur, so ``mix``
    and ``design`` ids are added for grouped splits.

    Args:
        path: The CSV written by ``scripts/fetch_concrete.sh``.
        tol: Mixes whose seven proportions all agree within ``tol`` kg/m^3 share a ``design``.

    Returns:
        1,030 rows: the nine columns of ``CONCRETE_COLUMNS``, ``mix`` (identical proportions,
        428 ids), and ``design`` (proportions within ``tol``; 315 ids at 1 kg/m^3).

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_concrete.sh``).
    """
    df = pd.read_csv(require(path, "fetch_concrete.sh"))
    assert df.shape == (1030, 9), df.shape
    df.columns = CONCRETE_COLUMNS
    df["mix"] = df.groupby(CONCRETE_INGREDIENTS, sort=False).ngroup()
    assert df["mix"].nunique() == 428, df["mix"].nunique()
    df["design"] = _design_ids(df, tol)
    return df


def _design_ids(df: pd.DataFrame, tol: float) -> np.ndarray:
    """Merge mixes whose seven proportions all agree within ``tol`` kg/m^3 into one design.

    The data combine several laboratories' records, and the same design often appears twice,
    once rounded (e.g. fine aggregate 806.0 vs 806.9 kg/m^3). For "how well do we predict a new
    design?" those are one design, so grouped splits should hold them out together.
    """
    mixes = df.groupby("mix")[CONCRETE_INGREDIENTS].first().to_numpy()
    parent = np.arange(len(mixes))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    close = np.abs(mixes[:, None, :] - mixes[None, :, :]).max(axis=-1) <= tol
    for a, b in np.argwhere(np.triu(close, 1)):
        parent[find(int(a))] = find(int(b))
    roots = np.array([find(i) for i in range(len(mixes))])
    _, design_of_mix = np.unique(roots, return_inverse=True)
    return design_of_mix[df["mix"].to_numpy()]


def repeat_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat tests: one mix at one age, tested more than once with different strengths.

    Groups whose strengths are all identical are one record entered twice and are left out.
    Their standard deviations bracket a model's error from below (the noise floor). There are
    few such groups (9, from 3 mixes), so the floor is an indication, not a measurement.

    Args:
        df: The concrete data.

    Returns:
        One row per (mix, age) group: ``size``, ``nunique``, and ``std`` (MPa, ddof = 1).
    """
    stats = df.groupby(["mix", "age"])["strength"].agg(["size", "nunique", "std"])
    return stats[(stats["size"] > 1) & (stats["nunique"] > 1)]


# ---------------------------------------------------------------------------------------------
# Encoders


def raw_features(df: pd.DataFrame) -> np.ndarray:
    """Seven ingredients (kg/m^3) and age (days), as recorded; shape ``(N, 8)``."""
    return df[RAW_INPUTS].to_numpy(dtype=np.float64)


def raw_log_age(df: pd.DataFrame) -> np.ndarray:
    """Seven ingredients (kg/m^3) and the natural log of age; shape ``(N, 8)``."""
    return np.column_stack([df[CONCRETE_INGREDIENTS].to_numpy(np.float64), np.log(df["age"])])


def mix_groups(df: pd.DataFrame) -> np.ndarray:
    """Dimensionless mix descriptors plus log age (Lecture 2's encoder); shape ``(N, 6)``.

    Binder b = cement + slag + fly ash. Columns, in ``GROUP_NAMES`` order: water-binder ratio,
    slag and fly-ash fractions of the binder, superplasticizer-binder ratio, aggregate-binder
    ratio, ln(age).
    """
    binder = df["cement"] + df["slag"] + df["fly_ash"]
    z = np.column_stack(
        [
            df["water"] / binder,
            df["slag"] / binder,
            df["fly_ash"] / binder,
            df["superplasticizer"] / binder,
            (df["coarse_agg"] + df["fine_agg"]) / binder,
            np.log(df["age"]),
        ]
    )
    assert z.shape == (len(df), 6), z.shape
    return z


def abrams_constants(df: pd.DataFrame) -> tuple[float, float]:
    """Least-squares fit of ln f = ln A - (w/b) ln B on the 28-day tests; returns (A, B)."""
    binder = df["cement"] + df["slag"] + df["fly_ash"]
    d28 = df["age"] == 28
    slope, intercept = np.polyfit((df["water"] / binder)[d28], np.log(df["strength"][d28]), 1)
    return float(np.exp(intercept)), float(np.exp(-slope))


# ---------------------------------------------------------------------------------------------
# Splits


def grouped_splits(df: pd.DataFrame, by: str = "design") -> list[tuple[np.ndarray, np.ndarray]]:
    """``REPEATS`` splits that keep every group entirely in train or in test.

    A quarter of the *groups* go to test, so the test share of rows varies from split to
    split (21.3% to 29.8% of the tests for designs).

    Args:
        df: The concrete data.
        by: ``"design"`` (proportions within 1 kg/m^3, the default and the honest unit for a
            new design) or ``"mix"`` (identical proportions only).

    Returns:
        (train rows, test rows) for each split.
    """
    gss = GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    return list(gss.split(df, groups=df[by]))


def random_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """``REPEATS`` 75/25 splits of the 1,030 tests, ignoring which mix a test belongs to."""
    ss = ShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    return list(ss.split(df))


def knn_predict(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int = K, *, scale: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """k-NN regression fit on ``tr`` and predicting ``te``; the scaler sees ``tr`` only.

    Args:
        X: Inputs, shape ``(N, D)`` (columns already selected).
        y: Strengths, shape ``(N,)``.
        tr: Training row indices.
        te: Row indices to predict.
        k: Number of neighbours.
        scale: Standardize with the training rows' mean and standard deviation.

    Returns:
        Predictions for ``te``, shape ``(len(te),)``, and the rows of their neighbours (indices
        into ``X``), shape ``(len(te), k)``.
    """
    X_tr, X_te = X[tr], X[te]
    if scale:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    model = KNeighborsRegressor(n_neighbors=k).fit(X_tr, y[tr])
    _, idx = model.kneighbors(X_te)
    assert idx.shape == (len(te), k), idx.shape
    return model.predict(X_te), tr[idx]
