"""Figures and numbers for CEE Lecture 2: nearest neighbors and representation (concrete strength).

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l02_nearest_neighbors.py \
        --out build/fig/cee_l02_nearest_neighbors

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one held-out mix, for the TikZ diagram). Data: ``data/derived/Concrete_Data.csv``
from ``scripts/fetch_concrete.sh`` (UCI Concrete Compressive Strength; Yeh, 1998).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from common import parse_out_dir, rmse, rnd, seed_all, tex_num, write_numbers, write_tex_macros
from concrete import (
    CONCRETE_INGREDIENTS,
    FC,
    GROUP_NAMES,
    RAW_INPUTS,
    K,
    abrams_constants,
    grouped_splits,
    knn_predict,
    load_concrete,
    mix_groups,
    random_splits,
    raw_features,
    raw_log_age,
    repeat_groups,
)

SEED: Final = 0
EARLY_AGE: Final = 28  # extrapolation experiment: train on ages up to this
LATE_AGE: Final = 56  # ... and predict the tests at this age or later
WB_SWEEP: Final = (0.30, 0.65)  # water-binder ratios of the water sweep
SWEEP_POINTS: Final = 141  # points of the water sweep
KS: Final = (1, 2, 3, 5, 7, 10, 15, 20)  # k of the post-hoc leakage curves
MIN_AGES_FOR_GAIN: Final = 4  # a mix's gain per ln(day) needs this many tested ages
WB_MATCH: Final = 0.02  # "matched on w/b" in the attempt: within this of the proposed mix
EXTRA_WATER: Final = 30.0  # kg/m³ of water added in the check question
MARGIN_FIXED: Final = 5.0  # MPa, the fixed margin marked on the trade-off
MARGIN_MAX: Final = 15.0  # MPa, the largest margin of the trade-off curve
MARGIN_CURVE_POINTS: Final = 61  # margins drawn on the curve, 0 to MARGIN_MAX
MARGIN_STEP: Final = 0.01  # MPa, the grid on which a fixed margin is matched to the spread rule
BINDER_MARGINS: Final = (0.0, 5.0, 10.0)  # MPa, margins marked on the binder curve
BOOT_DRAWS: Final = 400  # design bootstrap of "spread against fixed margin"
BOOT_SEED: Final = 0
# Rows chosen by rule here and frozen elsewhere: Lecture 1's LATER_EXAMPLE_ROWS excludes the
# traced mix and the attempt's four, so a change of rule must not pass silently.
TRACE_ROW_FROZEN: Final = 913
ATTEMPT_ROWS_FROZEN: Final = (524, 341, 1018, 336)
# Examples quoted on the leakage slide, checked in ``record_examples``.
ROUNDING_EXAMPLE_MIXES: Final = (14, 208)  # one design, fine aggregate 806.9 and 806.0 kg/m³
DUPLICATE_RECORD_ROWS: Final = (862, 1019)  # one 28-day test, cement 140 and 139.7 kg/m³
# The vote's hypothetical mix (not a row of the data).
VOTE_PREDICTION_MPA: Final = 37
VOTE_SPREAD_MPA: Final = 6


# ---------------------------------------------------------------------------------------------
# Representations (the encoders are in concrete.py)

# Slide label -> (encoder, standardize with training statistics?)
REPRESENTATIONS: Final[Mapping[str, tuple[Callable[[pd.DataFrame], np.ndarray], bool]]] = {
    "raw kg/m³ and days": (raw_features, False),
    "standardized": (raw_features, True),
    "standardized, ln age": (raw_log_age, True),
    "dimensionless groups": (mix_groups, True),
}


# ---------------------------------------------------------------------------------------------
# Fitting helpers


@dataclass(frozen=True)
class Fit:
    """Predictions of k-NN regression for one train/test split."""

    pred: np.ndarray  # mean of the k neighbors' strengths
    spread: np.ndarray  # sample standard deviation (ddof = 1) of the k neighbors' strengths
    neighbors: np.ndarray  # (n_test, k) training-row indices, nearest first


def knn_fit(
    X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int, *, scale: bool
) -> Fit:
    """``concrete.knn_predict`` plus the spread of the neighbors' strengths.

    Args:
        X: Features, shape (N, D).
        y: Strength, shape (N,).
        tr: Training row indices.
        te: Test row indices.
        k: Number of neighbors.
        scale: Standardize with training statistics.

    Returns:
        Predictions, the neighbors' sample standard deviation (NaN for k = 1, where it is
        undefined), and neighbor indices (into the full arrays).
    """
    pred, rows = knn_predict(X, y, tr, te, k, scale=scale)
    spread = y[rows].std(axis=1, ddof=1) if k > 1 else np.full(len(te), np.nan)
    return Fit(pred=pred, spread=spread, neighbors=rows)


def knn_regress(
    Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread of the k nearest training strengths (the code slide's listing).

    Args:
        Z_tr: Encoded training rows, shape (N, D).
        y_tr: Their strengths, shape (N,).
        Z_te: Encoded rows to predict, shape (M, D).
        k: Number of neighbors (at least 2, for the spread).

    Returns:
        The mean and the sample standard deviation (ddof = 1) of the k nearest strengths.
    """
    d2 = (Z_te**2).sum(1)[:, None] - 2 * Z_te @ Z_tr.T + (Z_tr**2).sum(1)[None, :]
    assert d2.shape == (len(Z_te), len(Z_tr))
    nn = np.argsort(d2, axis=1, kind="stable")[:, :k]  # k nearest
    return y_tr[nn].mean(axis=1), y_tr[nn].std(axis=1, ddof=1)


def verify_listing(df: pd.DataFrame) -> dict[str, object]:
    """Compare ``knn_regress`` with scikit-learn on the first design-held-out split (k = ``K``).

    The two can differ only where the k-th and (k+1)-th training distances tie. Each tie is
    classified by why its two records are equally far from the test: one design at two ages
    (``ties_same_design``), two designs whose groups are identical (``ties_identical_groups``),
    or neither (``ties_other``; in this split, a series stepping slag around the test's design).
    Each test whose means differ is described by its two tied records.

    Args:
        df: The concrete data.

    Returns:
        Test rows, rows that match, rows with a tie at rank k and their classes, rows whose
        means differ, and per differing test its row, the tied records' strengths (rank k,
        then k + 1) and their coarse and fine aggregate (kg/m³).

    Raises:
        ValueError: If the two codes disagree away from a tie, or on a tie that is not between
            identical groups (the slide says only that kind changes the mean).
    """
    y = df["strength"].to_numpy()
    tr, te = grouped_splits(df)[0]
    scaler = StandardScaler().fit(mix_groups(df.iloc[tr]))
    Z_tr = scaler.transform(mix_groups(df.iloc[tr]))
    Z_te = scaler.transform(mix_groups(df.iloc[te]))
    mean, _ = knn_regress(Z_tr, y[tr], Z_te, K)
    ref = KNeighborsRegressor(n_neighbors=K, algorithm="brute").fit(Z_tr, y[tr]).predict(Z_te)
    d2 = ((Z_te[:, None, :] - Z_tr[None, :, :]) ** 2).sum(-1)
    assert d2.shape == (len(te), len(tr)), d2.shape
    order = np.argsort(d2, axis=1, kind="stable")
    exact = np.take_along_axis(d2, order, axis=1)
    tie = np.isclose(exact[:, K - 1], exact[:, K], rtol=0, atol=1e-9)
    agree = np.isclose(mean, ref, rtol=0, atol=1e-9)
    if not agree[~tie].all():
        raise ValueError("knn_regress disagrees with scikit-learn away from ties")
    r_k, r_k1 = tr[order[:, K - 1]], tr[order[:, K]]  # the records at ranks k and k + 1
    groups = mix_groups(df)
    design = df["design"].to_numpy()
    same_design = tie & (design[r_k] == design[r_k1])
    same_groups = tie & np.isclose(groups[r_k], groups[r_k1]).all(axis=1)
    differing = np.flatnonzero(~agree)
    if not (same_groups[differing] & ~same_design[differing]).all():
        raise ValueError("a differing test is not a tie between two designs with equal groups")
    return {
        "test_rows": int(len(te)),
        "rows_matching": int(agree.sum()),
        "rows_with_tie": int(tie.sum()),
        "ties_same_design": int(same_design.sum()),
        "ties_identical_groups": int((same_groups & ~same_design).sum()),
        "ties_other": int((tie & ~same_design & ~same_groups).sum()),
        "rows_differing": int(len(differing)),
        "differing": [
            {
                "test_row": int(te[i]),
                "strengths": [rnd(y[r_k[i]], 1), rnd(y[r_k1[i]], 1)],
                "coarse_fine": [
                    [rnd(df["coarse_agg"].iat[r], 1), rnd(df["fine_agg"].iat[r], 1)]
                    for r in (r_k[i], r_k1[i])
                ],
            }
            for i in differing
        ],
        "agree_without_ties": True,
    }


# ---------------------------------------------------------------------------------------------
# Figures


def abrams_fit(df: pd.DataFrame) -> dict[str, float | int]:
    """Abrams' law on the 28-day tests, and the strength gain per ln(day) within mixes.

    Args:
        df: The concrete data.

    Returns:
        Abrams' constants, the R² of ln f on w/b and on w/c, the number of mixes tested at
        ``MIN_AGES_FOR_GAIN`` or more ages, and their median gain per ln(day).
    """
    binder = df["cement"] + df["slag"] + df["fly_ash"]
    wb = (df["water"] / binder).to_numpy()
    wc = (df["water"] / df["cement"]).to_numpy()
    d28 = (df["age"] == 28).to_numpy()
    log_f = np.log(df["strength"].to_numpy())
    a_const, b_const = abrams_constants(df)
    r_wb = float(np.corrcoef(wb[d28], log_f[d28])[0, 1])
    r_wc = float(np.corrcoef(wc[d28], log_f[d28])[0, 1])
    ages = df.groupby("mix")["age"].nunique()
    slopes = [
        np.polyfit(np.log(s["age"]), s["strength"], 1)[0]
        for m, s in df.groupby("mix")
        if ages[m] >= MIN_AGES_FOR_GAIN
    ]
    return {
        "abrams_A": rnd(a_const, 1),
        "abrams_B": rnd(b_const, 1),
        "r2_log_wb": rnd(r_wb**2, 2),
        "r2_log_wc": rnd(r_wc**2, 2),
        "n_mixes_4plus_ages": len(slopes),
        "median_gain_per_ln_day": rnd(np.median(slopes), 1),
    }


def plot_abrams_and_age(df: pd.DataFrame, out_dir: Path) -> None:
    """Left: 28-day strength vs water-binder ratio with Abrams' law. Right: strength vs age.

    The right panel draws the four mixes tested at the most ages; ties in that count keep the
    mixes' order in the data (a stable sort), so the legend cannot change with the library.

    Args:
        df: The concrete data.
        out_dir: Figure directory.
    """
    binder = df["cement"] + df["slag"] + df["fly_ash"]
    wb = (df["water"] / binder).to_numpy()
    d28 = (df["age"] == 28).to_numpy()
    a_const, b_const = abrams_constants(df)
    ages = df.groupby("mix")["age"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=aml_style.FULL)
    ax1.scatter(
        wb[d28],
        df["strength"][d28],
        s=18,
        alpha=0.6,
        color=aml_style.PALETTE[0],
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        label="28-day tests",
    )
    grid = np.linspace(wb[d28].min(), wb[d28].max(), 200)
    ax1.plot(
        grid,
        a_const / b_const**grid,
        color=aml_style.PALETTE[1],
        ls="--",
        label=rf"$f = {a_const:.0f}\,/\,{b_const:.1f}^{{w/b}}$",
    )
    ax1.set_xlabel("water–binder ratio $w/b$")
    ax1.set_ylabel("strength (MPa)")
    ax1.set_title("28 days: Abrams' law", loc="left")
    ax1.legend(loc="upper right", handletextpad=0.3)

    rich = ages.sort_values(ascending=False, kind="stable").index[:4]
    for c, m in enumerate(rich):
        s = df[df["mix"] == m].groupby("age")["strength"].mean()
        ax2.plot(
            s.index,
            s.values,
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            marker=aml_style.MARKERS[c],
            label=f"mix {c + 1}",
        )
    ax2.set_xscale("log")
    ax2.set_xticks([1, 3, 7, 28, 90, 365], ["1", "3", "7", "28", "90", "365"])
    ax2.set_xlabel("age (days, log scale)")
    ax2.set_title("Same mix, several ages", loc="left")
    ax2.legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    aml_style.save(fig, out_dir, "abrams_and_age")


def distance_share(df: pd.DataFrame) -> dict[str, float]:
    """Share (%) of the expected raw squared distance contributed by each input.

    For two independent tests, E[(x_j - x'_j)^2] = 2 Var(x_j), so feature j's share of
    E||x - x'||^2 is Var(x_j) / sum_l Var(x_l). The slide shows the attempt's four mixes as the
    worked example instead of a chart of these shares.

    Args:
        df: The concrete data.

    Returns:
        Each input's share, and the two aggregates' combined share (rounded once, from raw).
    """
    var = raw_features(df).var(axis=0)
    share = 100 * var / var.sum()
    names = [
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
    ]  # keys as published in numbers.json
    out = {n: rnd(s, 1) for n, s in zip(names, share, strict=True)}
    out["aggregates"] = rnd(share[5] + share[6], 1)
    return out


def representation_rmse(
    df: pd.DataFrame, out_dir: Path, floor: float
) -> dict[str, dict[str, float]]:
    """5-NN RMSE for each representation, grouped and random splits (``REPEATS`` each).

    The encoders are compared on the same 30 grouped splits, so each step is also measured
    paired, split by split: in how many splits the new encoder wins, and the mean and standard
    deviation of the paired difference. The four encoders were fixed in advance (engineering
    knowledge), not searched; the test splits only score them.

    Args:
        df: The concrete data.
        out_dir: Figure directory.
        floor: Median standard deviation of repeat tests, MPa (unrounded).

    Returns:
        Mean and standard deviation of the RMSE over the splits, per representation and split,
        and the paired steps.
    """
    y = df["strength"].to_numpy()
    per_split: dict[str, np.ndarray] = {}  # grouped RMSE per split, unrounded
    out: dict[str, dict[str, float]] = {}
    for name, (encode, scale) in REPRESENTATIONS.items():
        X = encode(df)
        res = {}
        for split_name, splits in (("grouped", grouped_splits(df)), ("random", random_splits(df))):
            errs = [rmse(knn_fit(X, y, tr, te, K, scale=scale).pred, y[te]) for tr, te in splits]
            res[f"{split_name}_mean"] = rnd(np.mean(errs), 3)
            res[f"{split_name}_sd"] = rnd(np.std(errs), 3)
            if split_name == "grouped":
                per_split[name] = np.asarray(errs)
        out[name] = res
    base = float(
        np.mean([rmse(np.full(len(te), y[tr].mean()), y[te]) for tr, te in grouped_splits(df)])
    )
    out["predict the mean"] = {"grouped_mean": rnd(base, 2)}
    names = list(REPRESENTATIONS)
    for before, after in itertools.pairwise(names):
        gain = per_split[before] - per_split[after]  # positive: the later encoder errs less
        out[f"paired: {after} vs {before}"] = {
            "wins": int((gain > 0).sum()),
            "splits": int(len(gain)),
            "mean_gain": rnd(gain.mean(), 2),
            "sd_gain": rnd(gain.std(ddof=1), 2),
        }
    plot_representation_rmse(per_split, base, floor, out_dir)
    return out


def plot_representation_rmse(
    per_split: dict[str, np.ndarray], base: float, floor: float, out_dir: Path
) -> None:
    """Dot plot of the grouped-split RMSE per representation, bracketed by two lines.

    Args:
        per_split: Grouped-split RMSE per split, per representation (``REPRESENTATIONS`` order).
        base: RMSE of predicting the training mean, MPa (the line above).
        floor: Median standard deviation of repeat tests, MPa (the line below).
        out_dir: Figure directory.
    """
    names = list(per_split)
    means = [float(per_split[n].mean()) for n in names]
    sds = [float(per_split[n].std()) for n in names]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    pos = np.arange(len(names))[::-1]
    ax.errorbar(means, pos, xerr=sds, fmt="o", color=aml_style.PALETTE[0], capsize=4, ms=8)
    for p, m in zip(pos, means, strict=True):
        ax.text(m, p + 0.22, f"{m:.1f}", ha="center", fontsize=12)
    ax.axvline(base, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(
        base - 0.2,
        pos[-1] - 0.45,
        "predict the mean",
        ha="right",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.axvline(floor, color=aml_style.MUTED, ls="--", lw=1.5)
    ax.text(
        floor + 0.3, pos[-1] - 0.45, "repeat tests", ha="left", fontsize=12, color=aml_style.MUTED
    )
    ax.set_yticks(pos, names)
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.8, len(names) - 0.4)
    ax.set_xlabel("test RMSE (MPa), 5-NN, designs held out")
    aml_style.save(fig, out_dir, "representation_rmse")


def leakage_curves(df: pd.DataFrame) -> dict[str, list[float]]:
    """Mean 5-NN test RMSE over the splits against k, for three ways of splitting (groups).

    Args:
        df: The concrete data.

    Returns:
        Per scheme (``random``, ``mix``, ``design``), the unrounded mean RMSE at each k of
        ``KS``.
    """
    y = df["strength"].to_numpy()
    X = mix_groups(df)
    schemes = (
        ("random", random_splits(df)),
        ("mix", grouped_splits(df, "mix")),
        ("design", grouped_splits(df, "design")),
    )
    return {
        name: [
            float(
                np.mean(
                    [rmse(knn_fit(X, y, tr, te, k, scale=True).pred, y[te]) for tr, te in splits]
                )
            )
            for k in KS
        ]
        for name, splits in schemes
    }


def reuse_share(
    df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], by: str
) -> dict[str, float]:
    """How often a test row's ``by`` group is also in training, over all the splits.

    The share for one split is the fraction of its test rows whose group appears among its
    training rows; the result summarizes that share over the splits (Lecture 1 computes the
    same share of mixes over the same random splits).

    Args:
        df: The concrete data.
        splits: (train, test) row indices.
        by: The group column (``"mix"`` or ``"design"``).

    Returns:
        The mean, smallest, and largest share over the splits (2 dp, from the raw shares).
    """
    g = df[by].to_numpy()
    shares = np.array([np.isin(g[te], g[tr]).mean() for tr, te in splits])
    return {"mean": rnd(shares.mean(), 2), "min": rnd(shares.min(), 2), "max": rnd(shares.max(), 2)}


def record_examples(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Check and describe the two records the leakage slide quotes.

    Args:
        df: The concrete data.

    Returns:
        ``rounding``: two mixes of one design that differ only in fine aggregate, by under
        1 kg/m³; ``duplicate``: one 28-day test recorded twice, cement rounded in one record,
        with the identical strength.

    Raises:
        ValueError: If either example no longer is what the slide says.
    """
    first = df.groupby("mix")[[*CONCRETE_INGREDIENTS, "design"]].first()
    a, b = (first.loc[m] for m in ROUNDING_EXAMPLE_MIXES)
    differs = [c for c in CONCRETE_INGREDIENTS if a[c] != b[c]]
    if a["design"] != b["design"] or differs != ["fine_agg"] or abs(a.fine_agg - b.fine_agg) >= 1:
        raise ValueError(f"mixes {ROUNDING_EXAMPLE_MIXES} are not a rounded copy: {differs}")
    twin = df.loc[list(DUPLICATE_RECORD_ROWS)]
    if twin["design"].nunique() != 1 or twin["strength"].nunique() != 1 or (twin.age != 28).any():
        raise ValueError(f"rows {DUPLICATE_RECORD_ROWS} are not one 28-day test recorded twice")
    return {
        "rounding": {
            "mixes": list(ROUNDING_EXAMPLE_MIXES),
            "design": int(a["design"]),
            "fine_agg": sorted(rnd(v, 1) for v in (a["fine_agg"], b["fine_agg"])),
        },
        "duplicate": {
            "rows": list(DUPLICATE_RECORD_ROWS),
            "cement": [rnd(v, 1) for v in twin["cement"]],
            "strength": rnd(twin["strength"].iloc[0], 1),
        },
    }


def leakage_vs_k(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """RMSE against k for three ways of splitting, dimensionless groups, and the reuse behind it.

    A post-hoc diagnostic: the curves score every k on the test splits, so they show how the
    split changes the error, not which k to use; the lecture's k = ``K`` was fixed in advance.

    Args:
        df: The concrete data.
        out_dir: Figure directory.

    Returns:
        The curves (2 dp), the gap between designs held out and a random split at k = ``K``
        (rounded once, from the unrounded curves), how often a test's mix (random splits) or
        design (mixes held out) is in training over the 30 splits, the repeated records, and
        the two quoted examples.
    """
    curves = leakage_curves(df)
    k5 = KS.index(K)
    dup = df[df.duplicated(RAW_INPUTS, keep=False)].groupby(RAW_INPUTS)["strength"]
    plot_leakage_vs_k(curves, out_dir)
    return {
        "k": list(KS),
        **{n: [rnd(v, 2) for v in c] for n, c in curves.items()},
        "gap_random_vs_design": rnd(curves["design"][k5] - curves["random"][k5], 1),
        "mix_in_train_random_splits": reuse_share(df, random_splits(df), "mix"),
        "design_in_train_mix_splits": reuse_share(df, grouped_splits(df, "mix"), "design"),
        "repeated_mix_age_pairs": int(dup.ngroups),
        "identical_strength_pairs": int((dup.nunique() == 1).sum()),
        "tests_per_mix_max": int(df.groupby("mix").size().max()),
        "mixes_tested_once": int((df.groupby("mix").size() == 1).sum()),
        **record_examples(df),
    }


def plot_leakage_vs_k(curves: dict[str, list[float]], out_dir: Path) -> None:
    """Test RMSE against k, one line per way of splitting.

    Args:
        curves: Output of ``leakage_curves``.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for c, (split_name, label) in enumerate(
        (
            ("random", "random split of tests"),
            ("mix", "identical mixes held out"),
            ("design", "designs held out"),
        )
    ):
        ax.plot(
            KS,
            curves[split_name],
            color=aml_style.PALETTE[c],
            ls=aml_style.LINESTYLES[c],
            marker=aml_style.MARKERS[c],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("test RMSE (MPa)")
    ax.set_title(f"post hoc: every k on the test splits; k = {K} fixed", loc="left", fontsize=12)
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "leakage_vs_k")


def plot_age_extrapolation(
    near_y: list[float],
    near_pred: list[float],
    far_y: np.ndarray,
    far_pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Predicted against measured strength of the late tests, with and without late ages.

    Args:
        near_y: Measured late strengths in the first grouped split (all ages in training).
        near_pred: Their predictions.
        far_y: Measured strengths of every late test.
        far_pred: Their predictions when training stops at ``EARLY_AGE`` days.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    lim = (0, 90)
    ax.plot(lim, lim, color=aml_style.MUTED, lw=1.2, ls=":")
    ax.scatter(
        near_y,
        near_pred,
        s=24,
        marker="o",
        color=aml_style.PALETTE[0],
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        label="all ages in training",
    )
    ax.scatter(
        far_y,
        far_pred,
        s=24,
        marker="s",
        color=aml_style.PALETTE[1],
        alpha=0.6,
        edgecolors="white",
        linewidths=0.4,
        label=f"trained on ≤ {EARLY_AGE} days",
    )
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"measured strength, age ≥ {LATE_AGE} days (MPa)")
    ax.set_ylabel("5-NN prediction (MPa)")
    ax.legend(loc="upper left", handletextpad=0.2)
    aml_style.save(fig, out_dir, "age_extrapolation")


def age_extrapolation(df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    """Train only on early ages, predict late ages; compare with late ages inside the range.

    The far run trains on every test up to ``EARLY_AGE`` days, so each late test's own mix is
    usually in training at an earlier age; the comparison holds whole designs out (Lecture 2's
    splits). The two runs therefore differ in two ways: the ages in training, and whether the
    mix itself is.

    Why the far run under-predicts: not a ceiling on strength (only one late test is stronger
    than the strongest training test). ln(age) of a late test lies beyond every training value,
    so its nearest neighbors are 28-day tests, mostly of its own mix: k-NN is flat beyond the
    training range of an input.

    Args:
        df: The concrete data.
        out_dir: Figure directory.

    Returns:
        Errors of both runs, and the shares that explain the far run's bias.
    """
    y = df["strength"].to_numpy()
    X = mix_groups(df)
    age = df["age"].to_numpy()
    mix = df["mix"].to_numpy()
    tr = np.where(age <= EARLY_AGE)[0]
    te = np.where(age >= LATE_AGE)[0]
    far = knn_fit(X, y, tr, te, K, scale=True)
    err_far = far.pred - y[te]
    nbr_age = age[far.neighbors]
    own_mix = (mix[far.neighbors] == mix[te][:, None]).any(axis=1)
    near_err, near_pred, near_y = [], [], []
    for tr2, te2 in grouped_splits(df):
        late = te2[age[te2] >= LATE_AGE]
        fit = knn_fit(X, y, tr2, late, K, scale=True)
        near_err.append(fit.pred - y[late])
        if not near_pred:
            near_pred, near_y = list(fit.pred), list(y[late])
    near = np.concatenate(near_err)

    plot_age_extrapolation(near_y, near_pred, y[te], far.pred, out_dir)
    return {
        "n_late": int(len(te)),
        "n_early_train": int(len(tr)),
        "late_mean_error": rnd(err_far.mean(), 1),
        "late_frac_under": rnd((err_far < 0).mean(), 2),
        "late_rmse": rnd(rmse(far.pred, y[te]), 1),
        "inrange_mean_error": rnd(near.mean(), 1),
        "inrange_rmse": rnd(np.sqrt(np.mean(near**2)), 1),
        "train_max_strength": rnd(y[tr].max(), 1),
        "late_above_train_max": int((y[te] > y[tr].max()).sum()),
        "late_neighbors_frac_28day": rnd((nbr_age == EARLY_AGE).mean(), 3),
        "late_frac_own_mix_in_neighbors": rnd(own_mix.mean(), 2),
    }


@dataclass(frozen=True)
class Sweep:
    """The water sweep of the traced mix: its w/b, the 5-NN predictions, and the numbers."""

    wb: np.ndarray  # water-binder ratio of each sweep point, shape (SWEEP_POINTS,)
    pred: np.ndarray  # 5-NN prediction at each point, MPa
    numbers: dict[str, float | int]


def water_sweep(df: pd.DataFrame, trace_row: int, train: np.ndarray) -> Sweep:
    """Sweep water for the traced mix at 28 days; does more water ever predict more strength?

    Where the prediction stops changing, the same five training tests stay nearest (checked).
    The sweep records how far the nearest test is there (standardized groups), against the
    proposed mix's own nearest test and the held-out tests' nearest distances, and how far the
    training tests wetter than that point are: from the wettest query, and from the proposed
    mix in the five groups other than w/b.

    Args:
        df: The concrete data.
        trace_row: The traced (proposed) mix.
        train: Training rows of the split the trace uses (the first design-held-out split).

    Returns:
        The sweep, with its end points, where it goes flat, how often and how much it rises,
        and the distances above.

    Raises:
        ValueError: If ``train`` is not the first design-held-out split's, or if the
            neighbor set still changes where the prediction is flat.
    """
    tr0, te0 = grouped_splits(df)[0]
    if not np.array_equal(tr0, train):
        raise ValueError("the sweep expects the first design-held-out split's training rows")
    y = df["strength"].to_numpy()
    X = mix_groups(df)
    base = df.iloc[[trace_row]].copy()
    base["age"] = 28
    binder = float(base["cement"].iloc[0] + base["slag"].iloc[0] + base["fly_ash"].iloc[0])
    water = binder * np.linspace(*WB_SWEEP, SWEEP_POINTS)
    sweep = pd.concat([base] * len(water), ignore_index=True)
    sweep["water"] = water
    Zs = mix_groups(sweep)
    scaler = StandardScaler().fit(X[train])
    Z_tr = scaler.transform(X[train])
    model = KNeighborsRegressor(n_neighbors=K).fit(Z_tr, y[train])
    Zs_std = scaler.transform(Zs)
    pred = model.predict(Zs_std)
    dist, nbr = model.kneighbors(Zs_std)
    wb = Zs[:, 0]
    steps = np.diff(pred)
    ups = int((steps > 1e-9).sum())
    changes = int((np.abs(steps) > 1e-9).sum())
    up_size = float(steps[steps > 1e-9].max()) if ups else 0.0
    flat = int(np.nonzero(np.abs(steps) > 1e-9)[0][-1] + 1)  # first point of the last terrace
    five = np.sort(train[nbr[flat]])
    if any(not np.array_equal(np.sort(train[n]), five) for n in nbr[flat:]):
        raise ValueError("the neighbors change on the sweep's last terrace")

    def nearest(Z: np.ndarray) -> np.ndarray:
        """Distance from each row of ``Z`` (standardized) to its nearest training test."""
        return model.kneighbors(Z, n_neighbors=1)[0][:, 0]

    held_out = nearest(scaler.transform(X[te0]))
    wet = train[X[train, 0] > wb[flat]]  # training tests wetter than the last terrace starts
    other = slice(1, None)  # the groups other than w/b
    z_row = scaler.transform(X[[trace_row]])
    d_other = np.sqrt(((scaler.transform(X[wet])[:, other] - z_row[:, other]) ** 2).sum(1))
    d_five = np.sqrt(((scaler.transform(X[five])[:, other] - z_row[:, other]) ** 2).sum(1))
    d_wet_end = np.sqrt(((scaler.transform(X[wet]) - Zs_std[-1]) ** 2).sum(1))
    a_const, b_const = abrams_constants(df)
    numbers: dict[str, float | int] = {
        "sweep_wb_min": WB_SWEEP[0],
        "sweep_wb_max": WB_SWEEP[1],
        "binder": rnd(binder, 1),
        "pred_at_min_wb": rnd(pred[0], 1),
        "pred_at_max_wb": rnd(pred[-1], 1),
        "flat_from_wb": rnd(wb[flat], 2),
        "abrams_at_max_wb": rnd(a_const / b_const ** WB_SWEEP[1], 1),
        "pred_changes": changes,
        "pred_increases": ups,
        "largest_increase": rnd(up_size, 1),
        "flat_neighbors_wb_max": rnd(X[five, 0].max(), 2),
        "nearest_distance_proposed": rnd(nearest(z_row)[0], 2),
        "nearest_distance_at_flat_from": rnd(dist[flat, 0], 2),
        "nearest_distance_at_max_wb": rnd(dist[-1, 0], 2),
        "heldout_nearest_distance_median": rnd(np.median(held_out), 2),
        "heldout_nearest_distance_p90": rnd(np.percentile(held_out, 90), 2),
        "wetter_train_tests": int(len(wet)),
        "wetter_nearest_distance_at_max_wb": rnd(d_wet_end.min(), 2),
        "wetter_other_groups_min_distance": rnd(d_other.min(), 2),
        "flat_neighbors_other_groups_max_distance": rnd(d_five.max(), 2),
    }
    return Sweep(wb=wb, pred=pred, numbers=numbers)


def plot_monotonicity(df: pd.DataFrame, sweep: Sweep, out_dir: Path) -> None:
    """The 5-NN prediction of the water sweep against Abrams' law.

    Args:
        df: The concrete data (for Abrams' constants).
        sweep: Output of ``water_sweep``.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.step(sweep.wb, sweep.pred, where="mid", color=aml_style.PALETTE[0], label="5-NN prediction")
    a_const, b_const = abrams_constants(df)
    ax.plot(
        sweep.wb,
        a_const / b_const**sweep.wb,
        color=aml_style.PALETTE[1],
        ls="--",
        label="Abrams' law (28-day fit)",
    )
    ax.set_xlabel("water–binder ratio $w/b$ (only water changes)")
    ax.set_ylabel("28-day strength (MPa)")
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "monotonicity")


def plot_tradeoff(curve: np.ndarray, points: dict[str, tuple[float, float]], out_dir: Path) -> None:
    """Weak mixes accepted against strong mixes rejected as the margin grows.

    Args:
        curve: (false accept, false reject) rates for margins 0 to 15 MPa, shape ``(M, 2)``.
        points: Rates for no margin (``"0"``), 5 MPa (``"5"``), and the neighbor spread
            (``"spread"``).
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(
        100 * curve[:, 1], 100 * curve[:, 0], color=aml_style.PALETTE[0], label="margin 0 to 15 MPa"
    )
    ax.plot(
        100 * points["0"][1],
        100 * points["0"][0],
        "o",
        ms=10,
        color=aml_style.PALETTE[1],
        label=r"accept if prediction $\geq f'_c$",
    )
    ax.plot(
        100 * points["5"][1],
        100 * points["5"][0],
        "s",
        ms=10,
        color=aml_style.PALETTE[2],
        label="margin 5 MPa",
    )
    ax.plot(
        100 * points["spread"][1],
        100 * points["spread"][0],
        "^",
        ms=11,
        color=aml_style.PALETTE[3],
        label="margin = neighbor spread",
    )
    ax.set_xlabel("strong mixes rejected (%)")
    ax.set_ylabel("weak mixes accepted (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.legend(loc="upper right", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "decision_tradeoff")


@dataclass(frozen=True)
class Pooled:
    """Held-out 28-day predictions pooled over the design-held-out splits."""

    rows: np.ndarray  # row of each prediction (a test is held out in several splits)
    pred: np.ndarray  # 5-NN prediction, MPa
    spread: np.ndarray  # the neighbors' sample standard deviation, MPa
    splits: int


def pooled_28d(df: pd.DataFrame) -> Pooled:
    """5-NN predictions of the 28-day tests in every design-held-out split, pooled.

    Args:
        df: The concrete data.

    Returns:
        The pooled rows, predictions, and spreads.
    """
    y = df["strength"].to_numpy()
    X = mix_groups(df)
    age = df["age"].to_numpy()
    rows, preds, spreads = [], [], []
    splits = grouped_splits(df)
    for tr, te in splits:
        t28 = te[age[te] == 28]
        fit = knn_fit(X, y, tr, t28, K, scale=True)
        rows.append(t28)
        preds.append(fit.pred)
        spreads.append(fit.spread)
    r, p, s = map(np.concatenate, (rows, preds, spreads))
    assert r.shape == p.shape == s.shape, (r.shape, p.shape, s.shape)
    return Pooled(rows=r, pred=p, spread=s, splits=len(splits))


def spread_vs_fixed(
    pred: np.ndarray, spread: np.ndarray, measured: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Strong mixes rejected by the spread rule minus by the fixed margin matched to it.

    For each row of ``weights`` (how often each prediction counts), the fixed margin is the
    smallest on a ``MARGIN_STEP`` grid that accepts no more weak mixes than the spread rule; the
    result is the spread rule's share of strong mixes rejected minus that margin's. With
    integer weights the counts are exact, so the comparison of accepted counts has no
    round-off.

    Args:
        pred: Pooled predictions, MPa, shape (n,).
        spread: The neighbors' spreads, MPa, shape (n,).
        measured: Measured strengths, MPa, shape (n,).
        weights: Non-negative integer weights, shape (B, n); a row of ones is the data itself.

    Returns:
        The difference in reject shares (fraction) and the matched margin (MPa), each shape
        (B,).

    Raises:
        ValueError: If no margin on the grid accepts as few weak mixes as the spread rule.
    """
    grid = np.round(np.arange(0, MARGIN_MAX + 0.001, MARGIN_STEP), 2)
    weak, strong = measured < FC, measured >= FC
    acc_spread = pred - spread >= FC
    acc_fixed = (pred[:, None] - grid[None, :]) >= FC  # (n, grid), as ``pred - m >= FC``
    w_weak, w_strong = weights[:, weak], weights[:, strong]
    ok = w_weak @ acc_fixed[weak] <= (w_weak @ acc_spread[weak])[:, None]
    if not ok.any(axis=1).all():
        raise ValueError("no fixed margin on the grid accepts as few weak mixes as the spread")
    j = ok.argmax(axis=1)  # the first, smallest, margin that does
    n_strong = w_strong.sum(axis=1)
    rej_spread = w_strong @ ~acc_spread[strong] / n_strong
    rej_fixed = (w_strong @ ~acc_fixed[strong])[np.arange(len(j)), j] / n_strong
    return rej_spread - rej_fixed, grid[j]


def design_bootstrap(pooled: Pooled, measured: np.ndarray, design: np.ndarray) -> np.ndarray:
    """``spread_vs_fixed`` over ``BOOT_DRAWS`` resamples of the designs (with replacement).

    A draw picks as many designs as the pool holds; every prediction of a picked design counts
    once per pick. The fixed margin is matched again in every draw.

    Args:
        pooled: Output of ``pooled_28d``.
        measured: Measured strength of each pooled prediction, MPa.
        design: Design of each pooled prediction.

    Returns:
        The difference in reject shares (fraction) of each draw, shape (``BOOT_DRAWS``,).
    """
    designs, which = np.unique(design, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    picks = rng.integers(0, len(designs), size=(BOOT_DRAWS, len(designs)))
    counts = np.stack([np.bincount(d, minlength=len(designs)) for d in picks]).astype(float)
    diff, _ = spread_vs_fixed(pooled.pred, pooled.spread, measured, counts[:, which])
    return diff


def decision_tradeoff(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Accept a mix when the prediction minus a margin reaches f'c; tally both kinds of error.

    Pools the 28-day tests of every grouped split: each of the 28-day tests is held out in
    several splits, so the pool counts predictions, not distinct tests. A false accept is a mix
    measured below f'c that the rule accepts; a false reject is a mix at or above f'c that the
    rule rejects. The spread margin is checked against the fixed margins: the smallest fixed
    margin that accepts no more weak mixes than the spread rule, what it rejects, and how much
    the difference moves when the designs are resampled.

    Args:
        df: The concrete data.
        out_dir: Figure directory.

    Returns:
        Pooled predictions and distinct tests, both error rates for three margins, the fixed
        margin that matches the spread rule's weak mixes accepted, and the spread's extra
        rejects (percentage points) with its design-bootstrap interval.
    """
    pooled = pooled_28d(df)
    p, s = pooled.pred, pooled.spread
    t = df["strength"].to_numpy()[pooled.rows]
    weak, strong = t < FC, t >= FC

    def rates(accept: np.ndarray) -> tuple[float, float]:
        """Weak mixes accepted and strong mixes rejected, as fractions.

        Args:
            accept: Whether the rule accepts each pooled prediction.

        Returns:
            The two error rates.
        """
        return float(accept[weak].mean()), float((~accept)[strong].mean())

    curve = np.array([rates(p - m >= FC) for m in np.linspace(0, MARGIN_MAX, MARGIN_CURVE_POINTS)])
    fa0, fr0 = rates(p >= FC)
    fa5, fr5 = rates(p - MARGIN_FIXED >= FC)
    fas, frs = rates(p - s >= FC)
    gap, m_eq = spread_vs_fixed(p, s, t, np.ones((1, len(t))))
    m_eq_mpa = float(m_eq[0])
    fa_eq, fr_eq = rates(p - m_eq_mpa >= FC)
    boot = 100 * design_bootstrap(pooled, t, df["design"].to_numpy()[pooled.rows])
    lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])

    plot_tradeoff(curve, {"0": (fa0, fr0), "5": (fa5, fr5), "spread": (fas, frs)}, out_dir)
    return {
        "n_tests": int(len(t)),
        "n_distinct_tests": int(len(np.unique(pooled.rows))),
        "splits": pooled.splits,
        "frac_weak": rnd(weak.mean(), 2),
        "false_accept_0": rnd(fa0, 3),
        "false_reject_0": rnd(fr0, 3),
        "false_accept_5": rnd(fa5, 3),
        "false_reject_5": rnd(fr5, 3),
        "false_accept_spread": rnd(fas, 3),
        "false_reject_spread": rnd(frs, 3),
        "median_spread": rnd(np.median(s), 1),
        "fixed_margin_same_accepts": m_eq_mpa,
        "fixed_margin_false_accept": rnd(fa_eq, 3),
        "fixed_margin_false_reject": rnd(fr_eq, 3),
        "spread_rejects_more_than_fixed": bool(fr_eq < frs),
        "spread_minus_fixed_reject_pts": rnd(100 * gap[0], 1),
        "bootstrap": {
            "draws": BOOT_DRAWS,
            "designs": int(len(np.unique(df["design"].to_numpy()[pooled.rows]))),
            "lo_pts": rnd(lo, 1),
            "median_pts": rnd(med, 1),
            "hi_pts": rnd(hi, 1),
            "share_at_or_below_zero": rnd((boot <= 0).mean(), 2),
        },
    }


# ---------------------------------------------------------------------------------------------
# One held-out mix, for the diagram and the hand-worked slide


@dataclass(frozen=True)
class Trace:
    """One held-out mix through the encoder and the decoder, for the diagram and the slides."""

    numbers: dict[str, object]  # rounded, for numbers.json
    x: dict[str, float]  # raw inputs, unrounded (numbers.tex)
    z: np.ndarray  # dimensionless groups, unrounded
    neighbor_strengths: np.ndarray  # the K nearest training strengths (groups), MPa
    pred: float  # 5-NN prediction (groups), MPa
    spread: float  # the neighbors' sample standard deviation, MPa
    y: float  # measured strength, MPa


def trace_mix(df: pd.DataFrame) -> tuple[Trace, int, np.ndarray]:
    """Follow one held-out 28-day test through the encoder and the 5-NN decoder.

    The traced test is the 28-day test in the first grouped split's test set whose measured
    strength is closest to f'c, so the example sits at the decision boundary.

    Args:
        df: The concrete data.

    Returns:
        The trace, the traced row index, and the training indices of that split.

    Raises:
        ValueError: If the rule picks another row than ``TRACE_ROW_FROZEN`` (Lecture 1 and
            Lecture 5 hard-code it).
    """
    y = df["strength"].to_numpy()
    tr, te = grouped_splits(df)[0]
    t28 = te[df["age"].to_numpy()[te] == 28]
    row = int(t28[np.argmin(np.abs(y[t28] - FC))])
    if row != TRACE_ROW_FROZEN:
        raise ValueError(f"traced row {row}, but Lectures 1 and 5 freeze {TRACE_ROW_FROZEN}")
    fit_groups = knn_fit(mix_groups(df), y, tr, np.array([row]), K, scale=True)
    fit_raw = knn_fit(raw_features(df), y, tr, np.array([row]), K, scale=False)
    z = mix_groups(df.iloc[[row]])[0]

    def describe(idx: np.ndarray) -> list[dict[str, float]]:
        """The w/b, age and strength of some rows, rounded as the slides show them.

        Args:
            idx: Row indices.

        Returns:
            One dictionary per row.
        """
        sub = df.iloc[idx]
        zz = mix_groups(sub)
        return [
            {
                "wb": rnd(zz[i, 0], 2),
                "age": int(sub["age"].iloc[i]),
                "strength": rnd(sub["strength"].iloc[i], 1),
            }
            for i in range(len(idx))
        ]

    r = df.iloc[row]
    numbers = {
        "row": row,
        "x": {c: rnd(r[c], 1) for c in RAW_INPUTS},
        "z": dict(zip(GROUP_NAMES, (rnd(v, 2) for v in z), strict=True)),
        "neighbors_groups": describe(fit_groups.neighbors[0]),
        "neighbors_raw": describe(fit_raw.neighbors[0]),
        "pred_groups": rnd(fit_groups.pred[0], 1),
        "spread_groups": rnd(fit_groups.spread[0], 1),
        "pred_raw": rnd(fit_raw.pred[0], 1),
        "y": rnd(y[row], 1),
        "error_groups": rnd(fit_groups.pred[0] - y[row], 1),
        "error_raw": rnd(fit_raw.pred[0] - y[row], 1),
        "n_train": int(len(tr)),
        "n_train_designs": int(df["design"].iloc[tr].nunique()),
    }
    trace = Trace(
        numbers=numbers,
        x={c: float(r[c]) for c in RAW_INPUTS},
        z=z,
        neighbor_strengths=y[fit_groups.neighbors[0]],
        pred=float(fit_groups.pred[0]),
        spread=float(fit_groups.spread[0]),
        y=float(y[row]),
    )
    return trace, row, tr


# ---------------------------------------------------------------------------------------------
# The students' attempt, the noise floor, and the check question


def attempt_rows(df: pd.DataFrame, row: int, train: np.ndarray) -> list[int]:
    """Pick the four tested mixes for the in-class attempt.

    They are the designs among the traced mix's 5 nearest training tests, under raw units or
    under the dimensionless groups, except the raw nearest design, which both encoders include
    (it would give the answer away). One row per design, the first met (raw neighbors first).

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.

    Returns:
        Four row indices, one per design.

    Raises:
        ValueError: If the groups' neighbors miss the raw nearest design, or the rule picks
            other rows than ``ATTEMPT_ROWS_FROZEN`` (Lecture 1 hard-codes them).
    """
    y = df["strength"].to_numpy()
    query = np.array([row])
    raw_nb = knn_fit(raw_features(df), y, train, query, K, scale=False).neighbors[0]
    grp_nb = knn_fit(mix_groups(df), y, train, query, K, scale=True).neighbors[0]
    design = df["design"].to_numpy()
    if design[raw_nb[0]] not in design[grp_nb]:
        raise ValueError("the groups' neighbors do not include the raw nearest design")
    chosen: dict[int, int] = {}  # design id -> first row met
    for i in [*raw_nb, *grp_nb]:
        if design[i] != design[raw_nb[0]] and design[i] not in chosen:
            chosen[int(design[i])] = int(i)
    if set(chosen.values()) != set(ATTEMPT_ROWS_FROZEN):
        raise ValueError(f"attempt rows {chosen}, but Lecture 1 freezes {ATTEMPT_ROWS_FROZEN}")
    return list(chosen.values())


def describe_attempt(
    df: pd.DataFrame, row: int, train: np.ndarray, rows: list[int]
) -> list[dict[str, object]]:
    """Describe the attempt mixes, lettered A to D by raw distance to the traced mix.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.
        rows: The attempt mixes (``attempt_rows``).

    Returns:
        Per mix, nearest first: proportions, its strengths in training at the same age, w/b, raw
        distance to the proposed mix, and the share of the squared raw distance due to water.
        Keys ``_wb`` and ``_distance`` hold unrounded values for the figure.

    Raises:
        ValueError: If a mix was tested at another age than the proposed mix.
    """
    y = df["strength"].to_numpy()
    X = raw_features(df)
    w = CONCRETE_INGREDIENTS.index("water")
    mix, age = df["mix"].to_numpy(), df["age"].to_numpy()
    wb = mix_groups(df)[:, 0]
    mixes: list[dict[str, object]] = []
    for i in rows:
        diff = X[i] - X[row]
        d2 = float(diff @ diff)
        same = train[(mix[train] == mix[i]) & (age[train] == age[i])]
        mixes.append(
            {
                "row": i,
                **{c: rnd(df[c].iat[i], 1) for c in RAW_INPUTS},
                "strengths": sorted(rnd(v, 1) for v in y[same]),
                "wb": rnd(wb[i], 2),
                "raw_distance": rnd(np.sqrt(d2), 1),
                "water_share_pct": rnd(100 * diff[w] ** 2 / d2, 1),
                "_wb": float(wb[i]),
                "_distance": float(np.sqrt(d2)),
            }
        )
    mixes.sort(key=lambda m: m["_distance"])
    for letter, m in zip("ABCD", mixes, strict=True):
        m["letter"] = letter
    if any(age[m["row"]] != age[row] for m in mixes):
        raise ValueError("the attempt mixes must share the proposed mix's age")
    return mixes


def plot_attempt(mixes: list[dict[str, object]], wb_row: float, out_dir: Path) -> None:
    """Raw distance against w/b for the attempt mixes: what raw units listen to.

    Args:
        mixes: Output of ``describe_attempt``.
        wb_row: The proposed mix's water-binder ratio.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.axhline(wb_row, color=aml_style.MUTED, ls="--", lw=1.2)
    ax.text(
        66,
        wb_row - 0.012,
        f"proposed mix, w/b = {wb_row:.2f}",
        fontsize=12,
        color=aml_style.MUTED,
        va="top",
    )
    for j, m in enumerate(mixes):
        ax.plot(
            m["_distance"],
            m["_wb"],
            marker=aml_style.MARKERS[j],
            ls="none",
            ms=10,
            color=aml_style.PALETTE[j],
        )
        mpa = ", ".join(f"{v:.1f}" for v in m["strengths"])
        below = m["_wb"] <= wb_row  # keep labels off the dashed line
        ax.annotate(
            f"{m['letter']}: {mpa} MPa",
            (m["_distance"], m["_wb"]),
            xytext=(9, -8 if below else 8),
            textcoords="offset points",
            fontsize=12,
            va="top" if below else "baseline",
        )
    ax.set_xlim(65, 125)
    ax.set_ylim(0.27, 0.45)
    ax.set_xlabel("raw distance to the proposed mix (kg/m³)")
    ax.set_ylabel("water–binder ratio")
    aml_style.save(fig, out_dir, "attempt_distance")


def attempt_mixes(
    df: pd.DataFrame, row: int, train: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Four tested mixes for the in-class attempt to estimate the traced mix, and the figure.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.
        out_dir: Figure directory.

    Returns:
        The four mixes (``describe_attempt``), the proposed mix, and the two estimates students
        tend to give: nearest in kilograms, or matched on w/b (within ``WB_MATCH``).
    """
    y = df["strength"].to_numpy()
    wb = mix_groups(df)[:, 0]
    mixes = describe_attempt(df, row, train, attempt_rows(df, row, train))
    plot_attempt(mixes, float(wb[row]), out_dir)
    proposed = {c: rnd(df[c].iat[row], 1) for c in RAW_INPUTS}
    proposed |= {"wb": rnd(wb[row], 2), "strength": rnd(y[row], 1)}
    matched = [m for m in mixes if abs(m["_wb"] - wb[row]) <= WB_MATCH]
    return {
        "mixes": mixes,
        "proposed": proposed,
        "mean_raw_nearest": rnd(np.mean(mixes[0]["strengths"]), 1),
        "mean_wb_matched": rnd(np.mean([v for m in matched for v in m["strengths"]]), 1),
        "letters_wb_matched": [m["letter"] for m in matched],
    }


def repeat_scatter(df: pd.DataFrame) -> tuple[dict[str, float | int], float]:
    """Scatter of repeat tests: one mix at one age, tested more than once with different results.

    The median standard deviation is a rough floor for the error of any model of these inputs:
    the same mix at the same age does not give the same strength twice. The groups come from few
    mixes (``concrete.repeat_groups``), so the floor is only an indication.

    Args:
        df: The concrete data.

    Returns:
        Groups, tests, mixes, the groups of the two mixes with the most, and the median and
        largest standard deviation (MPa); then the median standard deviation, unrounded, for
        the figure.
    """
    rep = repeat_groups(df)
    per_mix = rep.groupby(level="mix").size().sort_values(ascending=False)
    return {
        "groups": int(len(rep)),
        "tests": int(rep["size"].sum()),
        "mixes": int(len(per_mix)),
        "groups_top_two_mixes": int(per_mix.iloc[:2].sum()),
        "median_sd": rnd(rep["std"].median(), 1),
        "max_sd": rnd(rep["std"].max(), 1),
    }, float(rep["std"].median())


def water_check(
    df: pd.DataFrame, row: int, train: np.ndarray, extra_water: float = EXTRA_WATER
) -> dict[str, float]:
    """Numbers for the check question: add only ``extra_water`` kg/m^3 of water to the traced mix.

    Compares the raw distance of that change with the raw distance to the nearest tested mix,
    and the strength change that the fitted Abrams curve predicts.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.
        extra_water: Water added, kg/m^3.

    Returns:
        w/b and Abrams' strength before and after, and the raw distance to the nearest test.
    """
    a_const, b_const = abrams_constants(df)
    r = df.iloc[row]
    binder = float(r["cement"] + r["slag"] + r["fly_ash"])
    wb0, wb1 = float(r["water"]) / binder, (float(r["water"]) + extra_water) / binder
    X = raw_features(df)
    nearest = float(np.sqrt(((X[train] - X[row]) ** 2).sum(axis=1)).min())
    return {
        "extra_water": extra_water,
        "wb_before": rnd(wb0, 2),
        "wb_after": rnd(wb1, 2),
        "abrams_before": rnd(a_const / b_const**wb0, 1),
        "abrams_after": rnd(a_const / b_const**wb1, 1),
        "raw_nearest_tested": rnd(nearest, 1),
    }


# Diagram macro names for the raw inputs and the groups, in RAW_INPUTS and GROUP_NAMES order.
X_MACROS: Final = ("Cement", "Slag", "FlyAsh", "Water", "SP", "Coarse", "Fine", "Age")
Z_MACROS: Final[Mapping[str, int]] = {"Wb": 0, "LnAge": 5}  # the two groups the diagram shows


def write_tex(trace: Trace, out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, from unrounded values.

    Args:
        trace: Output of ``trace_mix``.
        out_dir: Figure directory.

    Returns:
        The path written.
    """
    x, z, nbr = trace.x, trace.z, trace.neighbor_strengths
    pred, spread, meas = trace.pred, trace.spread, trace.y
    macros = {f"X{m}": tex_num(x[c], "g") for c, m in zip(RAW_INPUTS, X_MACROS, strict=True)}
    macros |= {f"Z{m}": tex_num(z[j], ".2f") for m, j in Z_MACROS.items()}
    vals = [tex_num(v, ".1f") for v in nbr]
    macros |= {
        "K": str(K),
        "NbrStrengths": ", ".join(vals[:3]) + ",\\\\" + ", ".join(vals[3:]),  # two inset lines
        "Pred": tex_num(pred, ".1f"),
        "Spread": tex_num(spread, ".1f"),
        "Meas": tex_num(meas, ".1f"),
        "Err": tex_num(pred - meas, "+.1f"),
        "FC": tex_num(FC, "g"),
        "NTrain": str(trace.numbers["n_train"]),
        "NTrainDesigns": str(trace.numbers["n_train_designs"]),
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

NEAR_K: Final = 30  # neighbors drawn in the two-input picture of "near"
TERRACE_N: Final = 81  # grid points per side of the 3D prediction surface
WB_RANGE: Final = (0.25, 0.80)  # water-binder ratios of the surface


def title_art_figure(sweep: Sweep, out_dir: Path) -> None:
    """The title art: a constellation of tested mixes, one query joined to its five nearest.

    The horizon is the lecture's water sweep: the 5-NN prediction as only the water changes,
    a staircase that stops falling where the same five, drier, tests stay nearest.

    Args:
        sweep: Output of ``water_sweep``.
        out_dir: Figure directory.
    """
    c = title_art.Canvas(seed=2)
    c.ground(("#0b1a1c", "#0f1424", "#140b1d", "#06060d"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.05, 0.08, 0.90, 0.62, 5), soft=4), "#3b1d14", 0.8)
    c.glaze(c.polygon(title_art.stepped(0.20, 0.08, 0.60, 0.42, 3), soft=3), "#7c2f1a", 0.75)
    rim = c.disc((0.72, 1.18), 0.13, soft=4) - c.disc((0.72, 1.18), 0.11, soft=4)
    c.glow(np.clip(rim, 0, 1), "#e9a646", 0.5, spread=36)
    c.glaze(c.disc((0.72, 1.18), 0.11, soft=3), "#1c0c12", 0.9)
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(0.08, 0.92, 70), rng.uniform(0.72, 1.05, 70)])
    q = np.array([0.42, 0.88])
    near = pts[np.argsort(np.hypot(*(pts - q).T))[:5]]
    c.glow(c.lines([[tuple(q), tuple(n)] for n in near], 2.2, soft=0.8), "#e9c27a", 0.5, spread=6)
    dots = sum(c.disc(tuple(pt), 0.006, soft=1.5) for pt in pts)
    c.glow(np.clip(dots, 0, 1), "#9fb7c9", 0.45, spread=6)
    c.glow(c.disc(tuple(q), 0.014, soft=2), "#ffd98a", 1.0, spread=14)
    wb, pred = sweep.wb, sweep.pred
    xs = 0.08 + 0.84 * (wb - wb.min()) / np.ptp(wb)
    ys = 0.14 + 0.40 * (pred - pred.min()) / np.ptp(pred)
    stairs = [
        [(xs[i], ys[i]), (xs[i + 1], ys[i]), (xs[i + 1], ys[i + 1])] for i in range(len(xs) - 1)
    ]
    c.glow(c.lines(stairs, 4, soft=1.2), "#f3b45c", 0.8, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: stepped Andean terraces in umber and rust rise from "
        "the bottom; above them a constellation of pale points, one glowing point joined by golden "
        "threads to its five nearest; a dark sun with a burning rim hangs at the upper right, and "
        "a golden staircase, the lecture's water sweep, descends across the terraces.",
    )


def mixes_figure(df: pd.DataFrame, out_dir: Path) -> None:
    """Strength against age, one line per mix: a mix tested at several ages is several rows.

    The counts it shows (mixes, mixes tested once) are ``main``'s ``data`` and ``leakage``'s.

    Args:
        df: The concrete data.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    counts = df.groupby("mix")["age"].count()
    for m, tests in df.groupby("mix"):
        g = tests.groupby("age")["strength"].mean()
        if counts[m] > 1:
            ax.plot(g.index, g.values, color=aml_style.PALETTE[0], lw=0.8, ls="-", alpha=0.45)
        else:
            ax.plot(g.index, g.values, "o", color=aml_style.PALETTE[1], ms=3.5, alpha=0.7)
    ax.axhline(FC, color=aml_style.INK, lw=1.4, ls="--")
    ax.set(xscale="log", xlabel="age (days)", ylabel="strength (MPa)")
    ax.set_xticks([1, 7, 28, 90, 365], ["1", "7", "28", "90", "365"])
    ax.set_title("one line per mix; dots: tested once", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "mixes")


def near_figure(df: pd.DataFrame, row: int, train: np.ndarray, out_dir: Path) -> dict[str, float]:
    """The proposed mix's 30 nearest training tests, raw kilograms against the groups.

    Both panels plot the tests by water-binder ratio and age, the two things Abrams and curing
    say matter. Raw kilograms and days (all eight inputs, unscaled) pick tests of many ages;
    the dimensionless groups pick tests at the mix's own age, of similar w/b.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.
        out_dir: Figure directory.

    Returns:
        Per encoder: the w/b range of the 30 and how many are at the mix's age (28 days).
    """
    wb = mix_groups(df)[:, 0]
    age = df["age"].to_numpy()
    out: dict[str, float] = {}
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    for ax, name, X, scale in (
        (axes[0], "raw kg and days", raw_features(df), False),
        (axes[1], "dimensionless groups", mix_groups(df), True),
    ):
        nn = knn_fit(X, df["strength"].to_numpy(), train, np.array([row]), NEAR_K,
                     scale=scale).neighbors[0]  # fmt: skip
        ax.scatter(wb[train], age[train], s=8, color=aml_style.MUTED, alpha=0.25, lw=0)
        ax.scatter(wb[nn], age[nn], s=34, color=aml_style.PALETTE[0], lw=0.5, ec="white", zorder=3)
        ax.plot(wb[row], age[row], "*", ms=18, color=aml_style.PALETTE[1], mec="white", zorder=4)
        lo, hi = wb[nn].min(), wb[nn].max()
        ax.axvspan(lo, hi, color=aml_style.PALETTE[0], alpha=0.08, lw=0)
        key = "raw" if name.startswith("raw") else "groups"
        out[f"{key}_wb_min"], out[f"{key}_wb_max"] = rnd(lo, 2), rnd(hi, 2)
        out[f"{key}_at_28_days"] = int((age[nn] == age[row]).sum())
        ax.set_title(f"{name}: {out[key + '_at_28_days']} of {NEAR_K} at 28 days", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set(xlabel="water–binder ratio $w/b$", yscale="log", xlim=(0.2, 0.9))
    axes[0].set_ylabel("age (days)")
    axes[0].set_yticks([1, 7, 28, 90, 365], ["1", "7", "28", "90", "365"])
    aml_style.save(fig, out_dir, "near")
    return out


def terraces_figure(
    df: pd.DataFrame, row: int, train: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """5-NN's prediction over water-binder ratio and age, the proposed mix's other groups fixed.

    Between tested mixes the prediction is flat; where the neighbor set changes it steps: a
    terraced surface. Where the same neighbors stay nearest over a whole region (toward high
    w/b, for this mix's other groups) it stays on one terrace.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix.
        train: Training rows of the split the trace uses.
        out_dir: Figure directory.

    Returns:
        The share of grid steps along w/b that leave the prediction unchanged, and the w/b
        range of the surface.
    """
    y = df["strength"].to_numpy()
    X = mix_groups(df)
    scaler = StandardScaler().fit(X[train])
    model = KNeighborsRegressor(n_neighbors=K).fit(scaler.transform(X[train]), y[train])
    wb = np.linspace(*WB_RANGE, TERRACE_N)
    la = np.linspace(0.0, np.log(365.0), TERRACE_N)
    W, L = np.meshgrid(wb, la)
    Z = np.repeat(X[[row]], W.size, axis=0)
    Z[:, 0], Z[:, 5] = W.ravel(), L.ravel()
    pred = model.predict(scaler.transform(Z)).reshape(W.shape)
    assert pred.shape == (TERRACE_N, TERRACE_N), pred.shape
    fig = plt.figure(figsize=aml_style.SIDE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    aml_style.surface3d(ax, W, L, pred, cmap="cividis")
    ax.scatter([X[row, 0]], [X[row, 5]], [y[row]], color=aml_style.PALETTE[1], s=70,
               edgecolor="white", zorder=5)  # fmt: skip
    ax.set_xlabel("w/b", labelpad=-2, fontsize=12)
    ax.set_ylabel("age (days)", labelpad=-2, fontsize=12)
    ax.set_yticks(np.log([1, 7, 28, 365]), ["1", "7", "28", "365"])
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_zticks([20, 40, 60])
    ax.set_title("5-NN prediction (MPa)", loc="left", fontsize=13, pad=0)
    aml_style.style3d(ax, elev=32, azim=-50)
    aml_style.save(fig, out_dir, "terraces")
    steps = np.abs(np.diff(pred, axis=1)) > 1e-9
    return {"share_flat": rnd(1 - steps.mean(), 2), "wb_range": list(WB_RANGE)}


def binder_figure(df: pd.DataFrame, row: int, out_dir: Path) -> dict[str, int | float]:
    """Binder needed for f'c plus a margin, at the proposed mix's water, from Abrams' fit.

    Args:
        df: The concrete data.
        row: The traced (proposed) mix, whose water is held.
        out_dir: Figure directory.

    Returns:
        Binder (kg/m³, whole) at margins of 0, 5, 10, and 15 MPa, and the cost per MPa from 0
        to 10, from the rounded values as the slide shows them.
    """
    a_const, b_const = abrams_constants(df)
    water = float(df["water"].iloc[row])
    margin = np.linspace(0, MARGIN_MAX, 151)
    binder = water * np.log(b_const) / np.log(a_const / (FC + margin))
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(margin, binder, color=aml_style.PALETTE[0], lw=2.6)
    out: dict[str, int | float] = {"binder_at_15": round(float(binder[-1]))}
    for m in BINDER_MARGINS:
        b = float(np.interp(m, margin, binder))
        out[f"binder_at_{m:g}"] = round(b)
        ax.plot([m], [b], "o", color=aml_style.INK, ms=7)
        # Upper left of each point, except the first, whose label would cover the y ticks.
        left = m > 0
        ax.annotate(f"{b:.0f} kg", (m, b), xytext=(-10, 10) if left else (12, -2),
                    textcoords="offset points", fontsize=12, ha="right" if left else "left",
                    va="baseline" if left else "center")  # fmt: skip
    ax.set(xlabel="margin above 35 MPa (MPa)", ylabel="binder (kg/m³)", xticks=[0, 5, 10, 15],
           xlim=(-2.5, 15.5))  # fmt: skip
    ax.set_title(f"Abrams' fit, water {water:.0f} kg/m³", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "binder")
    out["per_mpa_0_to_10"] = rnd((out["binder_at_10"] - out["binder_at_0"]) / 10, 1)
    return out


# ---------------------------------------------------------------------------------------------
# Main


def main() -> None:
    """Generate every figure and number for CEE Lecture 2."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = load_concrete()
    trace, row, train = trace_mix(df)
    repeats, floor = repeat_scatter(df)
    sweep = water_sweep(df, row, train)
    plot_abrams_and_age(df, out_dir)
    plot_monotonicity(df, sweep, out_dir)
    numbers: dict[str, object] = {
        "data": {
            "tests": len(df),
            "mixes": int(df["mix"].nunique()),
            "ages": sorted(int(a) for a in df["age"].unique()),
            "n_28day": int((df["age"] == 28).sum()),
            "designs": int(df["design"].nunique()),
            "strength_min": rnd(df["strength"].min(), 1),
            "strength_max": rnd(df["strength"].max(), 1),
        },
        "abrams": abrams_fit(df),
        "distance_share_pct": distance_share(df),
        "representation_rmse": representation_rmse(df, out_dir, floor=floor),
        "repeat_scatter": repeats,
        "leakage": leakage_vs_k(df, out_dir),
        "attempt": attempt_mixes(df, row, train, out_dir),
        "water_check": water_check(df, row, train),
        "age_extrapolation": age_extrapolation(df, out_dir),
        "monotonicity": sweep.numbers,
        "decision": decision_tradeoff(df, out_dir),
        "trace": trace.numbers,
        "verify_listing": verify_listing(df),
        "vote": {"prediction_mpa": VOTE_PREDICTION_MPA, "spread_mpa": VOTE_SPREAD_MPA},
    }
    # the visuals that carry the slides
    title_art_figure(sweep, out_dir)
    mixes_figure(df, out_dir)
    numbers["near"] = near_figure(df, row, train, out_dir)
    numbers["terraces"] = terraces_figure(df, row, train, out_dir)
    numbers["binder"] = binder_figure(df, row, out_dir)
    aml_style.thumbnails(out_dir, ("near", "terraces", "leakage_vs_k", "monotonicity"))
    write_numbers(numbers, out_dir)
    print(write_tex(trace, out_dir))


if __name__ == "__main__":
    main()
