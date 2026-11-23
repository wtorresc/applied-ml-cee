"""Figures and numbers for CEE Lecture 1: framing, and training, validation, and test sets.

Run: ``pixi run -e aml-env python figures/cee_l01_framing.py --out build/fig/cee_l01_framing``

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one held-out design and the dataset counts, for the TikZ diagrams). Data, read
only: ``data/derived/Concrete_Data.csv`` (UCI Concrete Compressive Strength; Yeh, 1998),
``data/derived/nbi_il_bridges.csv`` (FHWA National Bridge Inventory, Illinois), and
``data/derived/bdg2_daily_electricity.npz`` (Building Data Genome 2; Miller et al., 2020).

The lecture's model is deliberately plain: 5-NN on the eight recorded inputs (seven ingredients in
kg/m^3 and the age in days), each standardized with training statistics. Lecture 2 improves it.
Conventions follow Lecture 2: the same 30 splits holding out 25% of the designs (``SEED`` 0),
the same 30 random splits, and scikit-learn's ``KNeighborsRegressor`` with its defaults, so the
numbers that Lecture 2 also quotes (9.5 and 8.9 MPa) are reproduced exactly. The approval
decision is about 28-day tests, so its honest margin is the held-out error on 28-day tests
(designs held out), not the error over all ages.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import integrate, stats
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from bridges import POOR, load_bridges
from buildings import Portfolio, build_portfolio, load_bdg2
from common import parse_out_dir, rmse, rnd, seed_all, tex_num, write_numbers, write_tex_macros
from concrete import (
    FC,
    RAW_INPUTS,
    REPEATS,
    TEST_SIZE,
    K,
    grouped_splits,
    knn_predict,
    load_concrete,
    random_splits,
    repeat_groups,
)

SEED: Final[int] = 0
FOLDS: Final[int] = 5  # validation folds inside the training designs
ORDERS: Final[int] = 20  # random orders of the input sets for the selection curve
# Worker processes for the 30 splits' searches. Each split is independent and seeded by its own
# index, so the results do not depend on this number (checked 2026-11-21: 4 against 30 workers).
N_JOBS: Final[int] = 4
M_GRID: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64, 128, 255)
INPUTS: Final[tuple[str, ...]] = tuple(RAW_INPUTS)
INPUT_NAMES: Final[tuple[str, ...]] = (
    "cement",
    "slag",
    "fly ash",
    "water",
    "SP",
    "coarse",
    "fine",
    "age",
)
SUBSETS: Final[tuple[tuple[int, ...], ...]] = tuple(
    c for r in range(1, len(INPUTS) + 1) for c in itertools.combinations(range(len(INPUTS)), r)
)
ALL8: Final[int] = len(SUBSETS) - 1  # index of the set with all eight inputs
# Rows that Lectures 2 and 5 use in their attempts and traces. Their designs are kept out of this
# lecture's examples so that later attempts are not spoiled. Lecture 2 raises if its rule stops
# picking 913 and (524, 341, 1018, 336); a shared concrete.EXAMPLE_ROWS is planned (q1 M1).
LATER_EXAMPLE_ROWS: Final[tuple[int, ...]] = (913, 524, 341, 1018, 336, 638, 825, 802, 779, 801)
BIN_MPA: Final[float] = 3.0  # histogram bin width in the margin picture
BIN_START_MPA: Final[float] = 10.0  # its first bin edge, below every strength in the band
# Numbers quoted from verified sources (not computed here), kept with the slide numbers.
CITED: Final[dict[str, dict[str, object]]] = {
    "flood_hub": {
        "countries": "over 80",
        "lead_days": 5,
        "source": "Nearing et al. (2024), Nature 627, abstract",
    },
    "aifs": {
        "operational": "25 February 2025",
        "source": "ECMWF news, 'ECMWF's AI forecasts become operational'",
    },
    "freeman": {"fail_ratio": 1.5, "source": "Freeman et al. (2014), PNAS 111(23), abstract"},
    "sinha_kapur": {"g": 0.36, "studies": 53, "source": "Sinha & Kapur (2021), RER 91(5)"},
}
# Course facts quoted on the slides, from ../course/cee_aml_course_design.tex (Grading and AI use).
# The design's grade adds 300 to the denominator for the best exam, the second-best exam and the
# final project; it does not state each part's points. The slide's 100 points each is that 300
# split evenly over the three parts (checked below), not a separate statement of the design.
EXAMS_COUNTED: Final[int] = 2
EXAM_POINTS: Final[int] = 100
PROJECT_POINTS: Final[int] = 100
DENOMINATOR_EXTRA: Final[int] = 300
COURSE: Final[dict[str, object]] = {
    "homeworks": 5,
    "core_points": 100,
    "exams": 3,
    "exams_counted": EXAMS_COUNTED,
    "exam_points": EXAM_POINTS,
    "project_points": PROJECT_POINTS,
    "xp_target_3_credits": 400,
    "xp_target_4_credits": 525,
    "denominator_extra": DENOMINATOR_EXTRA,
    "late_days": 10,
    "ai_levels": {
        "CBTF exams": "Restricted",
        "Homework cores": "Limited Use",
        "Critique-the-machine problems": "Use When Instructed",
        "Stretch goals and final project": "All Use, with a disclosure log",
    },
}
if len(SUBSETS) != 255 or SUBSETS[ALL8] != tuple(range(8)):
    raise ValueError(f"expected 255 input sets ending with all eight, got {len(SUBSETS)}")
if EXAM_POINTS * EXAMS_COUNTED + PROJECT_POINTS != DENOMINATOR_EXTRA:
    raise ValueError("the counted exams and the project must add up to the denominator's 300")


# ---------------------------------------------------------------------------------------------
# The model (Lecture 2 conventions): concrete.knn_predict, 5-NN on standardized inputs with the
# scaler and the neighbors from the training rows only; the splits are concrete.grouped_splits
# and random_splits.


def val_rmse(X: np.ndarray, y: np.ndarray, d: np.ndarray, k: int = K, folds: int = FOLDS) -> float:
    """Validation error over folds of whole designs, scaler fit inside (the slide listing).

    Args:
        X: Training inputs, columns already selected.
        y: Training strengths.
        d: Design id of each training row.
        k: Number of neighbors.
        folds: Number of folds of designs.

    Returns:
        RMSE of the out-of-fold predictions, MPa.
    """
    pred = np.empty(len(y))
    for fit, val in GroupKFold(folds).split(X, y, groups=d):
        sc = StandardScaler().fit(X[fit])  # fit rows only
        knn = KNeighborsRegressor(k).fit(sc.transform(X[fit]), y[fit])
        pred[val] = knn.predict(sc.transform(X[val]))
    return float(np.sqrt(np.mean((pred - y) ** 2)))  # written out, as on the slide


def search_one_split(
    X: np.ndarray, y: np.ndarray, d: np.ndarray, tr: np.ndarray, te: np.ndarray, *, halves_seed: int
) -> dict[str, np.ndarray]:
    """Score every input set on one split: validation, test, and the two halves of the test.

    The test designs are cut into two halves at random (``halves_seed``), so that one half can
    pick the best input set and the other half can check it.

    Args:
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.
        d: Design id of each row.
        tr: Training rows of the split.
        te: Test rows of the split.
        halves_seed: Seed of the cut of the test designs into two halves.

    Returns:
        ``val``, ``test``, ``half1``, ``half2``: RMSE per input set, each of shape ``(255,)``.
    """
    rng = np.random.default_rng(halves_seed)
    designs = np.unique(d[te])
    rng.shuffle(designs)
    in_half1 = np.isin(d[te], designs[: len(designs) // 2])
    out = {name: np.empty(len(SUBSETS)) for name in ("val", "test", "half1", "half2")}
    for i, cols in enumerate(SUBSETS):
        # Columns first, then rows, as cross_val_predict does with X[:, cols]. k-NN breaks ties
        # between equally distant tests (one mix at many ages, designs entered twice) by the
        # internals of its search tree, and a different copy order can change the last bit of
        # the standardized values and so which tied test is taken.
        Xc = X[:, list(cols)]
        out["val"][i] = val_rmse(Xc[tr], y[tr], d[tr])
        pred, _ = knn_predict(Xc, y, tr, te)
        out["test"][i] = rmse(pred, y[te])
        out["half1"][i] = rmse(pred[in_half1], y[te][in_half1])
        out["half2"][i] = rmse(pred[~in_half1], y[te][~in_half1])
    return out


def search_all_splits(
    X: np.ndarray, y: np.ndarray, d: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    """``search_one_split`` on every split, in ``N_JOBS`` parallel processes.

    Args:
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.
        d: Design id of each row.
        splits: The (train, test) rows of each split.

    Returns:
        ``val``, ``test``, ``half1``, ``half2``: RMSE per split and input set, ``(REPEATS, 255)``.
    """
    runs = Parallel(n_jobs=min(len(splits), N_JOBS))(
        delayed(search_one_split)(X, y, d, tr, te, halves_seed=SEED + s)
        for s, (tr, te) in enumerate(splits)
    )
    out = {name: np.stack([r[name] for r in runs]) for name in runs[0]}
    assert all(v.shape == (len(splits), len(SUBSETS)) for v in out.values())
    return out


# ---------------------------------------------------------------------------------------------
# Step 1: the attempt (four reported errors) and the bracket


@dataclass(frozen=True)
class Attempt:
    """The four teams' errors per split, and the scores of the input set chosen on validation.

    Attributes:
        train_err: A, all 8 inputs scored on the training tests, per design-held-out split.
        rand_err: B, all 8 inputs with random tests held out, per random split.
        test_err: D, all 8 inputs with whole designs held out, looked at once, per split.
        test28_err: D on the held-out 28-day tests only, per split: the decision's margin.
        best_test: C, the best of the 255 input sets on each split's test set.
        val_star: Validation score of the input set chosen on validation, per split.
        test_star: Test score of that input set, per split.
        chosen: Names of the input set chosen on validation, per split.
    """

    train_err: np.ndarray
    rand_err: np.ndarray
    test_err: np.ndarray
    test28_err: np.ndarray
    best_test: np.ndarray
    val_star: np.ndarray
    test_star: np.ndarray
    chosen: pd.Series

    def numbers(self) -> dict[str, object]:
        """The slide numbers, each rounded once from its raw value.

        Returns:
            Means over the splits of the four errors and of the chosen set's scores, Team C's
            optimism, and how often the most frequent set was chosen.
        """
        counts = self.chosen.value_counts()
        return {
            "A_train": rnd(self.train_err.mean(), 3),
            "B_random": rnd(self.rand_err.mean(), 3),
            "C_best_of_255_on_test": rnd(self.best_test.mean(), 3),
            "D_designs_once": rnd(self.test_err.mean(), 3),
            "D_28d": rnd(self.test28_err.mean(), 3),
            "honest_C_val": rnd(self.val_star.mean(), 3),
            "honest_C_test": rnd(self.test_star.mean(), 3),
            # Team C's optimism on the whole test set: its best of 255 against the set validation
            # chose, scored on the same test designs.
            "C_below_honest_C": rnd(self.test_star.mean() - self.best_test.mean(), 2),
            "honest_C_test_above_val_splits": int((self.test_star > self.val_star).sum()),
            "honest_C_gap": rnd((self.test_star - self.val_star).mean(), 3),
            "chosen_most_often": counts.index[0],
            "chosen_most_often_splits": int(counts.iloc[0]),
            "n_input_sets": len(SUBSETS),
        }


def team_errors(
    df: pd.DataFrame, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Errors of the model on all eight inputs, per split: Teams A, B and D.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.

    Returns:
        Per split: the training error (A, design-held-out splits), the random-split error (B),
        the design-held-out error (D), and D on the held-out 28-day tests only.
    """
    at28 = df["age"].to_numpy() == 28
    train_err, test_err, test28_err = [], [], []
    for tr, te in grouped_splits(df):
        pred_tr, _ = knn_predict(X, y, tr, tr)
        train_err.append(rmse(pred_tr, y[tr]))
        pred_te, _ = knn_predict(X, y, tr, te)
        test_err.append(rmse(pred_te, y[te]))
        test28_err.append(rmse(pred_te[at28[te]], y[te][at28[te]]))
    rand_err = [rmse(knn_predict(X, y, tr, te)[0], y[te]) for tr, te in random_splits(df)]
    return np.array(train_err), np.array(rand_err), np.array(test_err), np.array(test28_err)


def attempt_numbers(
    df: pd.DataFrame, X: np.ndarray, y: np.ndarray, search: dict[str, np.ndarray]
) -> Attempt:
    """The four teams' reported errors and the honest version of Team C's procedure.

    A: all 8 inputs, scored on the training tests it was fit to. B: all 8 inputs, random tests
    held out. C: the best of 255 input sets on the test set that picked it. D: all 8 inputs,
    whole designs held out, looked at once. D at 28 days is the same held-out error on the
    28-day tests only: the cases the approval decision faces, and so the decision's honest
    margin.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.
        search: Output of ``search_all_splits``.

    Returns:
        The per-split errors and the validation-chosen input set's scores.

    Raises:
        ValueError: If the search's all-eight test scores differ from Team D's.
    """
    train_err, rand_err, test_err, test28_err = team_errors(df, X, y)
    if not np.allclose(search["test"][:, ALL8], test_err, rtol=0, atol=1e-12):
        raise ValueError("the search's all-eight scores differ from Team D's")
    star = search["val"].argmin(axis=1)  # chosen on validation
    rows = np.arange(len(star))
    return Attempt(
        train_err=train_err,
        rand_err=rand_err,
        test_err=test_err,
        test28_err=test28_err,
        best_test=search["test"].min(axis=1),  # C: picked on the test set
        val_star=search["val"][rows, star],
        test_star=search["test"][rows, star],
        chosen=pd.Series([", ".join(INPUT_NAMES[j] for j in SUBSETS[i]) for i in star]),
    )


def baseline_and_floor(df: pd.DataFrame) -> dict[str, float | int]:
    """Lecture 2's bracket: predicting the training mean (above) and repeat tests (below).

    The floor rests on few repeat groups from few mixes (``concrete.repeat_groups``): an
    indication, not a measurement.

    Args:
        df: The concrete data.

    Returns:
        The mean predictor's RMSE (designs held out) and the repeat groups' counts and median SD.
    """
    y = df["strength"].to_numpy()
    base = [rmse(np.full(len(te), y[tr].mean()), y[te]) for tr, te in grouped_splits(df)]
    rep = repeat_groups(df)
    return {
        "predict_mean": rnd(np.mean(base), 2),
        "repeat_groups": int(len(rep)),
        "repeat_tests": int(rep["size"].sum()),
        "repeat_mixes": int(rep.index.get_level_values("mix").nunique()),
        "floor_median_sd": rnd(rep["std"].median(), 1),
    }


def bracket_figure(att: Attempt, bf: dict[str, float | int], out_dir: Path) -> None:
    """Dot plot: the honest errors between predicting the mean and the scatter of repeat tests.

    Args:
        att: Output of ``attempt_numbers``.
        bf: Output of ``baseline_and_floor``.
        out_dir: Figure directory.
    """
    labels = ["all 8 inputs", "chosen set:\nvalidation", "chosen set:\ntest, once"]
    vals = [float(att.test_err.mean()), float(att.val_star.mean()), float(att.test_star.mean())]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    pos = np.arange(len(labels))[::-1]
    for p, v, mk, fill in zip(pos, vals, ["o", "s", "o"], ["full", "none", "full"], strict=True):
        ax.plot(
            v,
            p,
            marker=mk,
            ms=10,
            ls="none",
            color=aml_style.PALETTE[0],
            fillstyle=fill,
            markeredgewidth=2,
        )
        ax.text(v, p + 0.25, f"{v:.1f}", ha="center", fontsize=12)
    ax.axvline(bf["predict_mean"], color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(
        bf["predict_mean"] - 0.3,
        pos[-1] - 0.5,
        "predict\nthe mean",
        ha="right",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.axvline(bf["floor_median_sd"], color=aml_style.MUTED, ls="--", lw=1.5)
    ax.text(
        bf["floor_median_sd"] + 0.3,
        pos[-1] - 0.5,
        "repeat tests\n(rough)",
        ha="left",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.set_yticks(pos, labels)
    ax.set_xlim(0, 18)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-1.0, len(labels) - 0.4)
    ax.set_xlabel("RMSE (MPa), designs held out")
    aml_style.save(fig, out_dir, "bracket")


# ---------------------------------------------------------------------------------------------
# Step 4: verify the listing against scikit-learn


def verify_listing(df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Compare ``val_rmse`` with ``cross_val_predict`` on split 0 for all 255 input sets.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.

    Returns:
        The number of input sets and the largest absolute difference, MPa.

    Raises:
        ValueError: If a design is on both sides of split 0, or if the two disagree by more
            than 1e-12 MPa.
    """
    d = df["design"].to_numpy()
    tr, te = grouped_splits(df)[0]
    if set(d[tr]) & set(d[te]):
        raise ValueError("a design is on both sides of split 0")
    worst = 0.0
    for cols in SUBSETS:
        Xc = X[:, list(cols)][tr]
        pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=K))
        ref = cross_val_predict(pipe, Xc, y[tr], groups=d[tr], cv=GroupKFold(FOLDS))
        ref_rmse = rmse(ref, y[tr])
        worst = max(worst, abs(val_rmse(Xc, y[tr], d[tr]) - ref_rmse))
    if worst > 1e-12:
        raise ValueError(f"val_rmse differs from cross_val_predict by {worst} MPa")
    return {"input_sets": len(SUBSETS), "max_abs_diff_mpa": worst, "designs_on_both_sides": 0}


# ---------------------------------------------------------------------------------------------
# Step 5: break it


def selection_curve(search: dict[str, np.ndarray], out_dir: Path) -> dict[str, object]:
    """The decisive figure: pick the best of M input sets on one half of the test designs.

    For every split, both ways round (half 1 picks and half 2 checks, then the reverse), and
    ``ORDERS`` random orders of the 255 input sets, the best of the first M on the picking half
    is scored on that half and on the other half. Means over all of them. A half holds fewer
    designs than the whole test set, so its scores are noisier and its winner's optimism larger
    than Team C's on the whole test set (``attempt_numbers``' ``C_below_honest_C``).

    Args:
        search: Output of ``search_all_splits``.
        out_dir: Figure directory.

    Returns:
        The winner's mean RMSE on the picking and on the fresh half, and their gap, per M.
    """
    rng = np.random.default_rng(SEED)
    picked = np.zeros(len(M_GRID))
    fresh = np.zeros(len(M_GRID))
    n = 0
    for s in range(REPEATS):
        for pick, check in (("half1", "half2"), ("half2", "half1")):
            for _ in range(ORDERS):
                order = rng.permutation(len(SUBSETS))
                for j, m in enumerate(M_GRID):
                    first = order[:m]
                    best = first[np.argmin(search[pick][s, first])]
                    picked[j] += search[pick][s, best]
                    fresh[j] += search[check][s, best]
                n += 1
    picked, fresh = picked / n, fresh / n
    gap = fresh - picked
    selection_figure(picked, fresh, out_dir)
    return {
        "M": M_GRID,
        "picked": [rnd(v, 2) for v in picked],
        "fresh": [rnd(v, 2) for v in fresh],
        "gap": [rnd(v, 2) for v in gap],
        "orders": ORDERS,
    }


def selection_figure(picked: np.ndarray, fresh: np.ndarray, out_dir: Path) -> None:
    """Line plot of the winner's score on the picking half and on the fresh half, against M.

    Args:
        picked: Mean winner's RMSE on the half that picked it, per M in ``M_GRID``.
        fresh: The same winners on the other half.
        out_dir: Figure directory.
    """
    gap = fresh - picked
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(
        M_GRID,
        fresh,
        color=aml_style.PALETTE[1],
        ls="--",
        marker="s",
        label="on the other half (fresh)",
    )
    ax.plot(
        M_GRID, picked, color=aml_style.PALETTE[0], marker="o", label="on the half that picked it"
    )
    ax.annotate(
        "",
        xy=(M_GRID[-1], picked[-1]),
        xytext=(M_GRID[-1], fresh[-1]),
        arrowprops={"arrowstyle": "<->", "color": aml_style.INK, "lw": 1.3},
    )
    ax.text(
        M_GRID[-1] * 0.9,
        picked[-1] - 0.25,
        f"{gap[-1]:.1f} MPa too good",
        ha="right",
        va="top",
        fontsize=12,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_GRID, [str(m) for m in M_GRID])
    ax.set_ylim(picked[-1] - 0.9, None)
    ax.set_xlabel("input sets tried, M")
    ax.set_ylabel("RMSE of the winner (MPa)")
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "selection_curve")


def random_split_reuse(df: pd.DataFrame) -> dict[str, object]:
    """How often a random split's test row has its mix, or its design, in training.

    The share for one split is the fraction of its test rows whose mix (or design) appears among
    its training rows; the result summarizes it over the 30 random splits, as Lecture 2's
    ``reuse_share`` does for the mix (so both lectures quote the same 74%). The design share is
    what the split picture rings, for split 0's drawn designs (``split_figures``).

    Args:
        df: The concrete data.

    Returns:
        For the mix and for the design: the mean, smallest, and largest share over the splits
        (2 dp, from the raw shares), and the number of splits.
    """
    splits = random_splits(df)
    out: dict[str, object] = {"splits": len(splits)}
    for by in ("mix", "design"):
        g = df[by].to_numpy()
        shares = np.array([np.isin(g[te], g[tr]).mean() for tr, te in splits])
        out[f"test_rows_with_{by}_in_train"] = {
            "mean": rnd(shares.mean(), 2),
            "min": rnd(shares.min(), 2),
            "max": rnd(shares.max(), 2),
        }
    return out


@dataclass(frozen=True)
class NewMaterial:
    """The model fit without fly ash: its validated promise and its error on fly-ash mixes.

    Attributes:
        promised: Mean validation RMSE over the splits of the no-fly-ash designs, MPa (raw).
        delivered: RMSE on the fly-ash tests, MPa (raw).
        pred: Predictions for the fly-ash rows.
        away: The fly-ash rows.
        numbers: The slide numbers.
    """

    promised: float
    delivered: float
    pred: np.ndarray
    away: np.ndarray
    numbers: dict[str, object]


def new_material(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, out_dir: Path) -> NewMaterial:
    """Train on mixes without fly ash; validate inside them; then predict the fly-ash mixes.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.
        out_dir: Figure directory.

    Returns:
        The raw promise and error (for the vote), the fly-ash predictions and rows, and the
        slide numbers.
    """
    d = df["design"].to_numpy()
    fly = df["fly_ash"].to_numpy() > 0
    home, away = np.flatnonzero(~fly), np.flatnonzero(fly)
    gss = GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SEED)
    promised = [
        rmse(knn_predict(X, y, home[a], home[b])[0], y[home[b]])
        for a, b in gss.split(home, groups=d[home])
    ]
    pred, _ = knn_predict(X, y, home, away)
    delivered = rmse(pred, y[away])
    mean_away = rmse(np.full(len(away), y[home].mean()), y[away])
    sc = StandardScaler().fit(X[home])
    fly_col = INPUTS.index("fly_ash")
    out: dict[str, object] = {
        "home_tests": int(len(home)),
        "home_designs": int(len(np.unique(d[home]))),
        "away_tests": int(len(away)),
        "away_designs": int(len(np.unique(d[away]))),
        "promised": rnd(np.mean(promised), 2),
        "promised_sd": rnd(np.std(promised), 2),
        "delivered": rnd(delivered, 2),
        "mean_predictor_on_away": rnd(mean_away, 2),
        "fly_ash_scale_in_training": float(sc.scale_[fly_col]),
        "fly_ash_sd_in_training": float(X[home][:, fly_col].std()),
    }
    new_material_figure(out, mean_away, out_dir)
    return NewMaterial(float(np.mean(promised)), delivered, pred, away, out)


def new_material_figure(out: dict[str, object], mean_away: float, out_dir: Path) -> None:
    """Dot plot: validation's promise without fly ash against the error on fly-ash mixes.

    Args:
        out: Output of ``new_material`` (rounded ``promised``, ``promised_sd``, ``delivered``).
        mean_away: RMSE of predicting the training mean on the fly-ash mixes, MPa.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    labels = ["validation:\nno fly ash", "new: mixes\nwith fly ash"]
    vals = [float(out["promised"]), float(out["delivered"])]
    pos = np.array([1, 0])
    ax.errorbar(
        vals[0],
        pos[0],
        xerr=out["promised_sd"],
        fmt="o",
        ms=10,
        capsize=4,
        color=aml_style.PALETTE[0],
    )
    ax.plot(vals[1], pos[1], marker="s", ms=10, ls="none", color=aml_style.PALETTE[1])
    ax.text(vals[0], pos[0] + 0.2, f"{vals[0]:.1f}", ha="center", fontsize=12)
    ax.text(vals[1] - 0.6, pos[1], f"{vals[1]:.1f}", ha="right", va="center", fontsize=12)
    ax.axvline(mean_away, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(
        mean_away - 0.3,
        -0.55,
        "predict the mean,\nfly-ash mixes",
        ha="right",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.set_yticks(pos, labels)
    ax.set_xlim(0, 18)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-0.8, 1.5)
    ax.set_xlabel("RMSE (MPa), 5-NN fit without fly ash")
    aml_style.save(fig, out_dir, "new_material")


# ---------------------------------------------------------------------------------------------
# Step 6: decide


def pooled_28d(
    df: pd.DataFrame, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out 28-day predictions of the 30 design-held-out splits (all 8 inputs), pooled.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.

    Returns:
        Predictions, measured strengths, and the rows they belong to (a row recurs when its design
        is held out in several splits).
    """
    age = df["age"].to_numpy()
    p28, y28, rows28 = [], [], []
    for tr, te in grouped_splits(df):
        pred, _ = knn_predict(X, y, tr, te)
        at28 = age[te] == 28
        p28.append(pred[at28])
        y28.append(y[te][at28])
        rows28.append(te[at28])
    p, t, rows = np.concatenate(p28), np.concatenate(y28), np.concatenate(rows28)
    assert p.shape == t.shape == rows.shape, (p.shape, t.shape, rows.shape)
    return p, t, rows


def decision_margins(
    p: np.ndarray, t: np.ndarray, rows: np.ndarray, honest: float
) -> dict[str, object]:
    """Approve a 28-day test when prediction - margin >= f'c; tally both kinds of error.

    Uses the pooled held-out 28-day predictions of the 30 design-held-out splits (all 8 inputs,
    ``pooled_28d``), as Lecture 2 does for its decision: the splits overlap, so a test is
    predicted in several of them. Weak: measured below f'c.

    Args:
        p: Pooled predictions.
        t: Their measured strengths.
        rows: Their rows.
        honest: The honest margin: the held-out error on 28-day tests, MPa (one decimal).

    Returns:
        Counts, the weak share, both error rates with no margin and with the honest one, and
        the RMSE of the pooled predictions (the honest margin is the mean of the 30 splits'
        RMSEs instead; both round to the same 9.9 MPa).
    """
    weak = t < FC
    out: dict[str, object] = {
        "predictions_28d": int(len(t)),
        "distinct_tests_28d": int(len(np.unique(rows))),
        "weak_share_28d": rnd(weak.mean(), 2),
        "honest_margin": honest,
        "pooled_rmse_28d": rnd(rmse(p, t), 2),
    }
    for name, margin in (("no_margin", 0.0), ("honest_margin", honest)):
        accept = p - margin >= FC
        out[f"{name}_weak_accepted"] = rnd(accept[weak].mean(), 3)
        out[f"{name}_strong_rejected"] = rnd((~accept)[~weak].mean(), 3)
    return out


def vote_mix(
    df: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    away: np.ndarray,
    *,
    promised: float,
    fly_rmse: float,
) -> dict[str, object]:
    """A fly-ash mix for the vote, chosen by a stated rule.

    Among 28-day fly-ash tests (designs not used as examples in later lectures) that the
    no-fly-ash model predicts high enough to pass with its validated margin (prediction -
    ``promised`` >= f'c) but that measured below f'c, the one whose error is closest to the
    model's RMSE on fly-ash mixes, ``fly_rmse``.

    Args:
        df: The concrete data.
        y: Strengths.
        pred: Predictions for the fly-ash rows ``away``.
        away: The fly-ash rows.
        promised: The validated margin as the slide quotes it (MPa, one decimal).
        fly_rmse: The raw RMSE on fly-ash mixes, MPa.

    Returns:
        The chosen test and the slide's arithmetic.
    """
    later = set(df["design"].iloc[list(LATER_EXAMPLE_ROWS)])
    cand = pd.DataFrame(
        {
            "row": away,
            "pred": pred,
            "y": y[away],
            "age": df["age"].to_numpy()[away],
            "design": df["design"].to_numpy()[away],
        }
    )
    cand = cand[(cand["age"] == 28) & ~cand["design"].isin(later)]
    cand = cand[(cand["pred"] - promised >= FC) & (cand["y"] < FC)]
    assert len(cand), "no fly-ash test fits the vote's rule"
    pick = cand.loc[((cand["pred"] - cand["y"]) - fly_rmse).abs().idxmin()]
    r = df.iloc[int(pick["row"])]
    binder = float(r["cement"] + r["slag"] + r["fly_ash"])
    return {
        "row": int(pick["row"]),
        "candidates": int(len(cand)),
        "pred": rnd(pick["pred"], 1),
        "y": rnd(pick["y"], 1),
        "fly_ash": rnd(r["fly_ash"], 1),
        "fly_ash_share_of_binder": rnd(float(r["fly_ash"]) / binder, 2),
        # Differences of the rounded values, as a reader would compute them on the slide.
        "pred_minus_promised": rnd(rnd(pick["pred"], 1) - promised, 1),
        "pred_minus_fly_rmse": rnd(rnd(pick["pred"], 1) - rnd(fly_rmse, 1), 1),
    }


# ---------------------------------------------------------------------------------------------
# Step 2: the traced design


@dataclass(frozen=True)
class Trace:
    """One held-out 28-day test followed through the pipeline.

    Attributes:
        x: Its eight inputs, raw (kg/m³ and days).
        z: Its standardized inputs, by input name.
        nbr: Its five neighbors' strengths, MPa.
        pred: The prediction, MPa.
        y: The measured strength, MPa.
        numbers: The slide numbers, rounded once from these raw values.
    """

    x: dict[str, float]
    z: dict[str, float]
    nbr: list[float]
    pred: float
    y: float
    numbers: dict[str, object]


def trace_design(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, honest: float) -> Trace:
    """Follow one held-out 28-day test of split 0 through the pipeline.

    Rule: among split 0's held-out 28-day tests whose design no later lecture uses as an
    example, take the wrong "yes" (prediction >= f'c > measured) whose error is closest to the
    honest error at 28 days, ``honest``: a costly error of typical size.

    Args:
        df: The concrete data.
        X: All eight inputs, shape ``(N, 8)``.
        y: Strengths, shape ``(N,)``.
        honest: The held-out error on 28-day tests, MPa (one decimal).

    Returns:
        The raw values for the diagram's macros and the slide numbers.
    """
    tr, te = grouped_splits(df)[0]
    pred, nbrs = knn_predict(X, y, tr, te)
    later = set(df["design"].iloc[list(LATER_EXAMPLE_ROWS)])
    d, age = df["design"].to_numpy(), df["age"].to_numpy()
    ok = (age[te] == 28) & ~np.isin(d[te], list(later)) & (pred >= FC) & (y[te] < FC)
    cand = np.flatnonzero(ok)
    i = int(cand[np.argmin(np.abs((pred[cand] - y[te][cand]) - honest))])
    row = int(te[i])
    sc = StandardScaler().fit(X[tr])
    z = sc.transform(X[[row]])[0]
    r = df.iloc[row]
    binder = float(r["cement"] + r["slag"] + r["fly_ash"])
    nb = nbrs[i]
    nb_binder = (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()[nb]
    same_design = int((d[te] == d[row]).sum())
    numbers: dict[str, object] = {
        "row": row,
        "candidates": int(len(cand)),
        "x": {c: rnd(r[c], 1) for c in INPUTS},
        "z": {n: rnd(v, 2) for n, v in zip(INPUT_NAMES, z, strict=True)},
        "wb": rnd(float(r["water"]) / binder, 2),
        "neighbor_strengths": [rnd(v, 1) for v in y[nb]],
        "neighbor_wb": [rnd(v, 2) for v in df["water"].to_numpy()[nb] / nb_binder],
        "neighbor_ages": [int(a) for a in age[nb]],
        "pred": rnd(pred[i], 1),
        "y": rnd(y[row], 1),
        "error": rnd(pred[i] - y[row], 1),
        "squared_error": rnd((pred[i] - y[row]) ** 2, 1),
        "records_of_its_design_in_test": same_design,
        "n_train": int(len(tr)),
        "n_train_designs": int(len(np.unique(d[tr]))),
    }
    return Trace(
        x={c: float(r[c]) for c in INPUTS},
        z={n: float(v) for n, v in zip(INPUT_NAMES, z, strict=True)},
        nbr=[float(v) for v in y[nb]],
        pred=float(pred[i]),
        y=float(y[row]),
        numbers=numbers,
    )


# ---------------------------------------------------------------------------------------------
# Tour of the datasets


def tour(
    df: pd.DataFrame,
    nbi: pd.DataFrame,
    data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame],
    portfolio: Portfolio,
) -> dict[str, object]:
    """Counts for the three datasets, read through the shared loaders.

    Bridges: ``bridges.load_bridges``, the bridge-year records with every input present, as
    Lectures 3 and 6 use them. Buildings: every metered building in the file, and the ones
    ``buildings.build_portfolio`` keeps for Lecture 4.

    Args:
        df: The concrete data.
        nbi: The bridge-year records.
        data: ``buildings.load_bdg2``'s kWh array, building ids, dates, and metadata.
        portfolio: ``buildings.build_portfolio``'s Lecture 4 buildings.

    Returns:
        The counts per dataset.
    """
    concrete = {
        "tests": int(len(df)),
        "mixes": int(df["mix"].nunique()),
        "designs": int(df["design"].nunique()),
        "ages": int(df["age"].nunique()),
        "age_min": int(df["age"].min()),
        "age_max": int(df["age"].max()),
        "tests_28d": int((df["age"] == 28).sum()),
        "strength_min": rnd(df["strength"].min(), 1),
        "strength_max": rnd(df["strength"].max(), 1),
    }

    per_year = nbi.groupby("nbi_year")["structure"].nunique()
    bridges = {
        "records": int(len(nbi)),
        "bridges": int(nbi["structure"].nunique()),
        "counties": int(nbi["county"].nunique()),
        "year_min": int(nbi["nbi_year"].min()),
        "year_max": int(nbi["nbi_year"].max()),
        "per_year_min": int(per_year.min()),
        "per_year_max": int(per_year.max()),
    }
    assert (bridges["records"], bridges["bridges"]) == (131_851, 23_464), bridges

    kwh, names, dates, meta = data
    buildings = {
        "buildings": int(len(names)),
        "sites": int(meta["site_id"].nunique()),
        "days": int(kwh.shape[1]),
        "hours_per_day": int(kwh.shape[2]),
        "year_min": int(dates.year.min()),
        "year_max": int(dates.year.max()),
        "lecture4_buildings": int(len(portfolio.name)),
        "lecture4_sites": int(len(set(portfolio.site))),
    }
    assert (buildings["lecture4_buildings"], buildings["lecture4_sites"]) == (1159, 15)
    return {"concrete": concrete, "nbi": bridges, "bdg2": buildings}


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

# Colour and marker of each role in the split pictures.
ROLE_STYLE: Final[dict[str, tuple[str, str]]] = {
    "training": (aml_style.PALETTE[0], "o"),
    "validation": (aml_style.PALETTE[2], "^"),
    "test": (aml_style.PALETTE[1], "s"),
}
ROLE_JOB: Final[dict[str, str]] = {
    "training": "fits",
    "validation": "chooses",
    "test": "reports, once",
}
NEAR_MPA: Final[float] = 2.0  # half-width of the band of predictions near the traced design's
M_SHOWN: Final[tuple[int, ...]] = (2, 16, 255)  # models compared in the winner's-curse picture
SPLIT_DESIGNS: Final[int] = 36  # designs drawn in the split pictures


def title_art_figure(out_dir: Path) -> None:
    """The title art: a stepped portal carrying a heavy lintel under a dark sun.

    The horizon is Abrams' law (strength falling as the water-cement ratio rises), the rule an
    engineer writes on the lecture's first diagram; the crimson wedge is the load.

    Args:
        out_dir: Figure directory.
    """
    c = title_art.Canvas(seed=1)
    c.ground(("#1d0b08", "#2a0d0c", "#120a16", "#07060d"), grain=0.08)
    c.glaze(c.polygon([(0, 0), (1, 0), (1, 0.36), (0, 0.46)], soft=30), "#4a1410", 0.55)
    rim = c.disc((0.30, 1.13), 0.15, soft=4) - c.disc((0.30, 1.13), 0.128, soft=4)
    c.glow(np.clip(rim, 0, 1), "#f2a33a", 0.55, spread=40)
    c.glaze(c.disc((0.30, 1.13), 0.128, soft=3), "#24090c", 0.92)
    for x0 in (0.12, 0.66):  # two stepped piers
        c.glaze(c.polygon(title_art.stepped(x0, 0.10, 0.24, 0.70, 3), soft=3), "#7a2418", 0.88)
        c.glaze(c.polygon([(x0 + 0.06, 0.10), (x0 + 0.18, 0.10), (x0 + 0.18, 0.74),
                           (x0 + 0.06, 0.74)], soft=2), "#a5481f", 0.75)  # fmt: skip
    lintel = [(0.06, 0.80), (0.94, 0.80), (0.94, 0.92), (0.06, 0.92)]
    c.glaze(c.polygon(lintel, soft=2), "#3a0f12", 0.95)
    c.glow(c.lines([[(0.06, 0.80), (0.94, 0.80)]], 3, soft=1), "#e6b35e", 0.5, spread=8)
    wedge = [(0.44, 1.30), (0.56, 1.30), (0.50, 0.86)]  # the load, pressing into the lintel
    c.glaze(c.polygon(wedge, soft=2), "#c2361f", 0.85)
    door = [(0.36, 0.10), (0.64, 0.10), (0.64, 0.78), (0.36, 0.78)]
    c.glaze(c.polygon(door, soft=3), "#120d1d", 0.9)
    wc = np.linspace(0.3, 0.9, 200)
    fc = 96.5 / 7.0**wc  # Abrams' law, f_c = A / B^(w/c), with A and B typical of the 1918 tests
    curve = [[(0.38 + 0.24 * (w - 0.3) / 0.6, 0.16 + 0.55 * (f - fc.min()) / np.ptp(fc))
              for w, f in zip(wc, fc, strict=True)]]  # fmt: skip
    c.glow(c.lines(curve, 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean piers in reds and ochres carry a "
        "dark lintel, pressed by a crimson wedge; a dark sun with a burning rim hangs above, and "
        "inside the doorway a golden curve falls, Abrams' law of strength against "
        "water-cement ratio.",
    )


def margin_figure(
    p: np.ndarray, t: np.ndarray, rows: np.ndarray, centre: float, out_dir: Path
) -> dict[str, object]:
    """Where mixes predicted near the traced design landed: measured strengths, f'c marked.

    The bins run from ``BIN_START_MPA`` past the largest strength in the band, so every
    prediction in the band is drawn.

    Args:
        p: Pooled held-out 28-day predictions (``pooled_28d``).
        t: Their measured strengths.
        rows: Their rows (a test recurs when its design is held out in several splits).
        centre: The traced design's prediction, MPa; the band is its whole MPa ± ``NEAR_MPA``.
        out_dir: Figure directory.

    Returns:
        The band, the predictions in it and their distinct tests, the range of their measured
        strengths, and the share of them measured weak.

    Raises:
        ValueError: If a strength in the band falls below the first bin, or the histogram
            does not count every prediction in the band.
    """
    lo, hi = round(centre) - NEAR_MPA, round(centre) + NEAR_MPA
    near = (p >= lo) & (p <= hi)
    landed = t[near]
    if landed.min() < BIN_START_MPA:
        raise ValueError(f"a strength in the band, {landed.min():.1f} MPa, is below the bins")
    bins = np.arange(BIN_START_MPA, landed.max() + BIN_MPA, BIN_MPA)
    if np.histogram(landed, bins=bins)[0].sum() != near.sum():
        raise ValueError("the margin histogram leaves out predictions in the band")
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    weak = landed < FC
    ax.hist(landed[~weak], bins=bins, color=aml_style.PALETTE[0], alpha=0.85, label="strong")
    ax.hist(landed[weak], bins=bins, color=aml_style.PALETTE[1], alpha=0.95, hatch="//",
            label="weak")  # fmt: skip
    ax.axvline(FC, color=aml_style.INK, lw=2.2)
    ax.text(FC - 1, ax.get_ylim()[1] * 0.92, rf"$f'_c$ = {FC:g}", ha="right", fontsize=13)
    ax.set(xlabel="measured strength at 28 days (MPa)", ylabel="held-out predictions")
    ax.set_title(f"predicted {lo:g}–{hi:g} MPa: {weak.mean():.0%} weak", loc="left")
    ax.legend(loc="upper right", fontsize=12)
    aml_style.save(fig, out_dir, "margin")
    return {
        "near_band_mpa": [lo, hi],
        "near_predictions": int(near.sum()),
        "near_distinct_tests": int(len(np.unique(rows[near]))),
        "near_measured_min_mpa": rnd(landed.min(), 1),
        "near_measured_max_mpa": rnd(landed.max(), 1),
        "near_weak_share": rnd(weak.mean(), 2),
    }


def _subtitle(ax: plt.Axes, title: str, line: str) -> None:
    """A bold panel title with a muted line under it.

    Args:
        ax: The panel.
        title: Its title.
        line: The muted line between the title and the panel.
    """
    ax.set_title(title, loc="left", pad=24)
    ax.annotate(line, (0, 1), xytext=(0, 6), xycoords="axes fraction", textcoords="offset points",
                fontsize=11.5, color=aml_style.MUTED, va="bottom")  # fmt: skip


def datasets_figure(
    df: pd.DataFrame, nbi: pd.DataFrame, portfolio: Portfolio, out_dir: Path
) -> dict[str, object]:
    """The three datasets, one panel each: tests, bridges on a map, typical weekdays.

    Args:
        df: The concrete data.
        nbi: The bridge-year records.
        portfolio: Lecture 4's buildings.
        out_dir: Figure directory.

    Returns:
        The bridges on the map and how many of them are poor.
    """
    # Wide gaps: each panel's subtitle is wider than the map panel between them.
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.6})
    ax = axes[0]
    weak = (df["strength"] < FC).to_numpy()
    for sel, color in ((~weak, aml_style.PALETTE[0]), (weak, aml_style.PALETTE[1])):
        ax.scatter(df["age"][sel], df["strength"][sel], s=7, color=color, alpha=0.55, lw=0)
    ax.axhline(FC, color=aml_style.INK, lw=1.4, ls="--")
    ax.set(xscale="log", xlabel="age (days)", ylabel="strength (MPa)", xticks=[1, 28, 365])
    ax.set_xticklabels(["1", "28", "365"])
    _subtitle(ax, "concrete", r"mix $\rightarrow$ strength $\rightarrow$ approve?")
    ax = axes[1]
    last = nbi[nbi["nbi_year"] == nbi["nbi_year"].max()]
    poor = last["rating"] <= POOR
    ax.scatter(last["lon"], last["lat"], s=1.0, color=aml_style.MUTED, alpha=0.3, lw=0)
    ax.scatter(last.loc[poor, "lon"], last.loc[poor, "lat"], s=1.6, color=aml_style.PALETTE[1],
               lw=0)  # fmt: skip
    ax.set(xticks=[], yticks=[], aspect=1 / np.cos(np.deg2rad(40.0)))
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    _subtitle(ax, "Illinois bridges", r"inventory $\rightarrow$ rating $\rightarrow$ inspect?")
    ax = axes[2]
    shape = portfolio.P16 / portfolio.P16.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(SEED)
    for i in rng.choice(len(shape), 40, replace=False):
        ax.plot(np.arange(24), shape[i], color=aml_style.MUTED, lw=0.6, ls="-", alpha=0.5)
    ax.plot(np.arange(24), np.median(shape, axis=0), color=aml_style.PALETTE[0], lw=2.6)
    ax.set(xlabel="hour", ylabel="load / daily mean", xticks=[0, 12, 23], ylim=(0, 3))
    _subtitle(ax, "buildings", r"meter $\rightarrow$ weekday $\rightarrow$ visit?")
    aml_style.save(fig, out_dir, "datasets")
    return {"poor_bridges_last_year": int(poor.sum()), "bridges_last_year": int(len(last))}


def split_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Two pictures of the same designs: whole designs held out, then random tests held out.

    Each column is one design and each dot one of its tests (by age). Left build: split 0 of
    the grouped splits, its validation fold the first of GroupKFold inside the training designs.
    Right build: split 0 of the random splits; ringed test dots have their design in training.
    The designs drawn are tested at least five times, so they show the reuse at its plainest.

    Args:
        df: The concrete data.
        out_dir: Figure directory.

    Returns:
        The random split's test rows among the drawn designs, and how many are ringed.
    """
    design, age = df["design"].to_numpy(), df["age"].to_numpy()
    counts = pd.Series(design).value_counts()
    many = counts[counts >= 5].index.to_numpy()
    shown = np.sort(np.random.default_rng(SEED).choice(many, SPLIT_DESIGNS, replace=False))
    col = {g: i for i, g in enumerate(shown)}
    rows = np.flatnonzero(np.isin(design, shown))
    tr, te = grouped_splits(df)[0]
    _, fold_val = next(GroupKFold(FOLDS).split(tr, groups=design[tr]))
    role = np.full(len(df), "training", dtype=object)
    role[tr[fold_val]] = "validation"
    role[te] = "test"
    rtr, rte = random_splits(df)[0]
    rrole = np.full(len(df), "training", dtype=object)
    rrole[rte] = "test"
    leaked = np.zeros(len(df), bool)
    leaked[rte] = np.isin(design[rte], design[rtr])
    out: dict[str, object] = {}
    for name, roles in (("split_designs", role), ("split_random", rrole)):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        for g, i in col.items():  # one stick per design: its role's colour if it has one role
            mine = rows[design[rows] == g]
            kinds = set(roles[mine])
            color = ROLE_STYLE[kinds.pop()][0] if len(kinds) == 1 else "#C8C8C8"
            ax.plot([i, i], [age[mine].min(), age[mine].max()], color=color, lw=5, ls="-",
                    alpha=0.35, solid_capstyle="round", zorder=1)  # fmt: skip
        x = np.array([col.get(g, -1) for g in design], dtype=float)
        for r, (color, marker) in ROLE_STYLE.items():
            m = rows[roles[rows] == r]
            if len(m):
                label = f"{r}: {ROLE_JOB[r]}" if name == "split_designs" else r
                ax.scatter(x[m], age[m], s=46, color=color, marker=marker, label=label,
                           edgecolor="white", lw=0.5, zorder=3)  # fmt: skip
        if name == "split_random":
            m = rows[leaked[rows]]
            ax.scatter(x[m], age[m], s=170, facecolor="none", edgecolor=aml_style.INK, lw=1.4,
                       zorder=4, label="design also in training")  # fmt: skip
            out["shown_test_rows_leaked"] = int(leaked[rows].sum())
            out["shown_test_rows"] = int((rrole[rows] == "test").sum())
        ax.set(yscale="log", yticks=[1, 7, 28, 90, 365], xticks=[], ylabel="age (days)",
               xlim=(-1, SPLIT_DESIGNS))  # fmt: skip
        ax.set_yticklabels(["1", "7", "28", "90", "365"])
        ax.set_xlabel(f"{SPLIT_DESIGNS} designs, one column each")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=12)
        aml_style.save(fig, out_dir, name)
    return out


def expected_min(m: int, mean: float, sd: float) -> float:
    """Expected smallest of ``m`` independent normal scores (an order statistic, by quadrature).

    Args:
        m: Number of scores.
        mean: Their common mean.
        sd: Their common standard deviation.

    Returns:
        E[min], from the density of the minimum, m phi(x) S(x)^(m-1), S the survival function.
    """

    def density_moment(x: float) -> float:
        """The integrand x f(x) of E[min] in standard units.

        Args:
            x: A standardized score.

        Returns:
            x times the density of the minimum of ``m`` standard normal scores at x.
        """
        return x * m * stats.norm.pdf(x) * stats.norm.sf(x) ** (m - 1)

    return mean + sd * integrate.quad(density_moment, -12, 12, limit=200)[0]


def winners_curse_figure(true_err: float, sd: float, out_dir: Path) -> dict[str, object]:
    """Equally good models, noisy scores: the best score falls further below the truth as M grows.

    One draw of M independent scores per panel (seeded), the true error dashed, the best score
    ringed, and the expected best (exact order statistic) as a solid line.

    Args:
        true_err: The models' common true error, MPa (Team D's score).
        sd: The spread of a score, MPa (Team D's spread over the 30 splits).
        out_dir: Figure directory.

    Returns:
        The true error, and for each M drawn the expected best score and its gap below the
        truth (the numbers printed in the panels).
    """
    rng = np.random.default_rng((SEED, 2))
    fig, axes = plt.subplots(1, len(M_SHOWN), figsize=aml_style.FULL, sharey=True)
    out: dict[str, float] = {}
    gaps: dict[str, float] = {}
    for ax, m in zip(axes, M_SHOWN, strict=True):
        scores = rng.normal(true_err, sd, m)
        x = rng.uniform(-1, 1, m)
        ax.scatter(x, scores, s=22 if m > 20 else 70, color=aml_style.PALETTE[0], alpha=0.7, lw=0)
        best = int(np.argmin(scores))
        ax.scatter([x[best]], [scores[best]], s=230, facecolor="none", edgecolor=aml_style.INK,
                   lw=2)  # fmt: skip
        e = expected_min(m, true_err, sd)
        out[str(m)] = rnd(e, 2)
        gaps[str(m)] = rnd(true_err - e, 1)
        ax.axhline(true_err, color=aml_style.MUTED, lw=1.6, ls="--", label="true error")
        ax.axhline(e, color=aml_style.PALETTE[1], lw=2.4, ls="-", label="expected best")
        ax.text(1.55, e - 0.12, f"{e - true_err:+.1f}", color=aml_style.PALETTE[1], va="top",
                ha="right", fontsize=13, fontweight="bold")  # fmt: skip
        ax.set(xlim=(-1.3, 1.6), xticks=[], ylim=(6.2, 12.6))
        ax.set_title(f"best of {m}", loc="left")
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("validation RMSE (MPa)")
    axes[0].legend(loc="upper left", fontsize=12)
    aml_style.save(fig, out_dir, "winners_curse")
    return {"true": rnd(true_err, 2), "expected_best": out, "expected_gap": gaps}


def costs_figure(p: np.ndarray, t: np.ndarray, honest: float, out_dir: Path) -> dict[str, object]:
    """Approve when prediction - margin >= f'c: weak mixes approved and strong ones rejected.

    The rates at no margin and at the honest one are ``decision_margins``'; this returns the
    curve's far end.

    Args:
        p: Pooled held-out 28-day predictions (``pooled_28d``).
        t: Their measured strengths.
        honest: The honest margin, MPa.
        out_dir: Figure directory.

    Returns:
        The largest margin drawn and both rates there.
    """
    margins = np.linspace(0, 16, 161)
    weak = t < FC
    approved_weak = np.array([((p - m >= FC) & weak).sum() / weak.sum() for m in margins])
    rejected_strong = np.array([((p - m < FC) & ~weak).sum() / (~weak).sum() for m in margins])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(margins, 100 * approved_weak, color=aml_style.PALETTE[1], lw=2.6,
            label="weak approved: the public's risk")  # fmt: skip
    ax.plot(margins, 100 * rejected_strong, color=aml_style.PALETTE[0], ls="--", lw=2.6,
            label="strong rejected: cost, cement")  # fmt: skip
    for m in (0.0, honest):
        ax.axvline(m, color=aml_style.MUTED, lw=1, ls=":")
    ax.text(honest + 0.3, 30, f"honest margin\n{honest:g} MPa", fontsize=12, va="top")
    ax.set(xlabel="margin (MPa)", ylabel="share (%)", ylim=(0, 100), xlim=(0, 16))
    ax.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)  # the honest-margin line runs behind it  # fmt: skip
    aml_style.save(fig, out_dir, "costs")
    return {
        "max_margin": float(margins[-1]),
        "weak_approved_at_max_margin": rnd(approved_weak[-1], 3),
        "strong_rejected_at_max_margin": rnd(rejected_strong[-1], 3),
    }


# ---------------------------------------------------------------------------------------------
# Macros for the diagrams


def tex_macros(trace: Trace, concrete: dict[str, object]) -> dict[str, str]:
    """The traced design and the concrete counts as LaTeX macros for the TikZ diagrams.

    Args:
        trace: ``trace_design``'s result.
        concrete: The concrete counts from ``tour``.

    Returns:
        Macro name to LaTeX text, every number formatted once from its raw value.
    """
    x, z, nbr = trace.x, trace.z, trace.nbr
    names = {
        "cement": "Cement",
        "slag": "Slag",
        "fly_ash": "FlyAsh",
        "water": "Water",
        "superplasticizer": "SP",
        "coarse_agg": "Coarse",
        "fine_agg": "Fine",
        "age": "Age",
    }
    macros = {f"X{name}": tex_num(x[key], ".0f") for key, name in names.items()}  # kg/m³, days
    macros |= {
        "ZWater": tex_num(z["water"], "+.2f"),
        "ZCement": tex_num(z["cement"], "+.2f"),
    }
    vals = [tex_num(v, ".1f") for v in nbr]
    pred, meas = trace.pred, trace.y
    macros |= {
        "K": str(K),
        "NbrStrengths": f"{', '.join(vals[:3])},\\\\{', '.join(vals[3:])}",
        "Pred": tex_num(pred, ".1f"),
        "Meas": tex_num(meas, ".1f"),
        "SqErr": tex_num((pred - meas) ** 2, ".1f"),
        "FC": tex_num(FC, "g"),
        "NTrain": str(trace.numbers["n_train"]),
        "NTests": tex_num(float(concrete["tests"]), ",.0f"),
        "NMixes": str(concrete["mixes"]),
    }
    return macros


def concrete_numbers(df: pd.DataFrame, out_dir: Path) -> tuple[dict[str, object], Trace]:
    """Every concrete number and figure of the lecture, in the order the steps need them.

    Args:
        df: The concrete data.
        out_dir: Figure directory.

    Returns:
        The concrete part of ``numbers.json``, and the traced design for the diagram macros.
    """
    X = df[list(INPUTS)].to_numpy(np.float64)
    y = df["strength"].to_numpy()
    assert X.shape == (1030, 8), X.shape
    search = search_all_splits(X, y, df["design"].to_numpy(), grouped_splits(df))
    att = attempt_numbers(df, X, y, search)
    bf = baseline_and_floor(df)
    bracket_figure(att, bf, out_dir)
    # The decision's margin: the held-out error on 28-day tests, the mean of the 30 splits' RMSEs.
    honest = rnd(att.test28_err.mean(), 1)
    shift = new_material(df, X, y, out_dir)
    trace = trace_design(df, X, y, honest)
    p28, t28, rows28 = pooled_28d(df, X, y)
    d_sd = float(att.test_err.std())
    numbers: dict[str, object] = {
        "attempt": att.numbers(),
        "bracket": bf,
        "trace": trace.numbers,
        "verify_listing": verify_listing(df, X, y),
        "selection_curve": selection_curve(search, out_dir),
        "random_split": random_split_reuse(df),
        "new_material": shift.numbers,
        "decision": decision_margins(p28, t28, rows28, honest),
        "vote": vote_mix(
            df,
            y,
            shift.pred,
            shift.away,
            promised=rnd(shift.promised, 1),
            fly_rmse=shift.delivered,
        ),
        # Two independent normal scores of spread sigma: the better is sigma / sqrt(pi) too
        # good; an upper bound when the scores are positively correlated (one test set).
        "two_models": {
            "sigma": rnd(d_sd, 1),
            "expected_gain": rnd(d_sd / np.sqrt(np.pi), 1),
        },
        "margin": margin_figure(p28, t28, rows28, trace.pred, out_dir),
        "splits": split_figures(df, out_dir),
        "winners_curse": winners_curse_figure(float(att.test_err.mean()), d_sd, out_dir),
        "costs": costs_figure(p28, t28, honest, out_dir),
    }
    return numbers, trace


def main() -> None:
    """Generate every figure and number for CEE Lecture 1."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = load_concrete()
    numbers, trace = concrete_numbers(df, out_dir)
    nbi = load_bridges()
    data = load_bdg2()
    portfolio = build_portfolio(data)
    counts = tour(df, nbi, data, portfolio)
    numbers |= {
        "tour": counts,
        "datasets": datasets_figure(df, nbi, portfolio, out_dir),
        "cited": CITED,
        "course": COURSE,
        "K": K,
        "fc": FC,
        "repeats": REPEATS,
        "folds": FOLDS,
    }
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("split_designs", "winners_curse", "new_material", "costs"))
    write_numbers(numbers, out_dir)
    print(write_tex_macros(tex_macros(trace, counts["concrete"]), out_dir, "cee_l01_framing.py"))


if __name__ == "__main__":
    main()
