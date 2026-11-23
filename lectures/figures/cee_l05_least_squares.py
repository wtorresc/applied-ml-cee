"""Figures and numbers for CEE Lecture 5: least squares, ridge, and lasso (concrete strength).

Run::

    pixi run -e aml-env python figures/cee_l05_least_squares.py \
        --out build/fig/cee_l05_least_squares

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one held-out mix, for the TikZ diagram). Data: ``data/derived/Concrete_Data.csv``
from ``scripts/fetch_concrete.sh`` (UCI Concrete Compressive Strength; Yeh, 1998). Same loader,
design-grouped splits, and encoder as Lecture 2, so the numbers connect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.linear_model import Lasso, LinearRegression, Ridge, lasso_path
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from common import parse_out_dir, rmse, rnd, seed_all, tex_num, write_numbers, write_tex_macros
from concrete import (
    CONCRETE_INGREDIENTS,
    FC,
    GROUP_NAMES,
    PROPOSED_ROW,
    RAW_INPUTS,
    REPEATS,
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

SEED: Final[int] = 0
# The attempt: four 28-day tests from one laboratory series (water 185 kg/m^3, no slag, fly ash,
# or superplasticizer; only the cement changes), all in the first split's training set, and the
# proposed mix P, a held-out design of the same family.
ATTEMPT_ROWS: Final[Mapping[str, int]] = MappingProxyType({"A": 825, "B": 802, "C": 779, "D": 801})
ATTEMPT_WATER: Final[float] = 185.0  # kg/m^3, the attempt series' water (Abrams per-kg too)
# Extrapolation test: train on mixes with at most this much cement, test on mixes above the next.
CEMENT_TRAIN_MAX: Final[float] = 350.0
CEMENT_TEST_MIN: Final[float] = 400.0
# Abrams' strength gain per kg of cement is quoted at these cement contents (kg/m^3).
ABRAMS_CEMENT_LOW: Final[float] = 300.0
ABRAMS_CEMENT_HIGH: Final[float] = 500.0
# Intervals on the extrapolation test: whole designs resampled with replacement.
BOOT_DRAWS: Final[int] = 2000
BOOT_SEED: Final[int] = 0
CI_LEVEL: Final[float] = 0.95
# The mix the students ask an AI assistant about: more cement than any tested mix (max 540).
QUERY: Final[Mapping[str, float]] = MappingProxyType(
    {
        "cement": 600.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 165.0,
        "superplasticizer": 10.0,
        "coarse_agg": 1000.0,
        "fine_agg": 650.0,
        "age": 28.0,
    }
)
NEAR_WB_WINDOW: Final[float] = 0.03  # audit key: 28-day tests with w/b this close to the query's
DAYS_PER_YEAR: Final[float] = 365.0
RIDGE_UNITS_ALPHA: Final[float] = 100.0  # one penalty, two unit systems for age
RIDGE_COLLINEAR_ALPHA: Final[float] = 1.0  # standardized ridge on the collinear design
RIDGE_CV_ALPHAS: Final[tuple[float, ...]] = tuple(float(a) for a in np.logspace(-3, 3, 25))
# A penalty at the grid's lower edge counts as the lambda -> 0 limit (least squares) only if it
# shrinks every direction of the standardized training features by less than this share.
RIDGE_LIMIT_TOL: Final[float] = 1e-3
# Lasso's grid (LassoCV's default): LASSO_GRID_N penalties from the one that zeroes every weight
# down to LASSO_GRID_DECADES decades below it.
LASSO_GRID_N: Final[int] = 100
LASSO_GRID_DECADES: Final[float] = 3.0
LASSO_LADDER_DECADES: Final[float] = 6.0  # the grid grows down to here while the best is its end
CV_FOLDS: Final[int] = 5  # grouped (designs) inner folds for ridge's and lasso's penalties
MARGIN_QUANTILE: Final[float] = 0.90  # share of held-out 28-day predictions the margin covers
# The family's cement grid for the decision (kg/m^3): from a lean mix to the most ever tested.
CEMENT_GRID: Final[tuple[float, float, int]] = (250.0, 540.0, 2901)
# Prediction bands for the scatter of held-out errors: below the first, above the second (MPa).
SCATTER_BANDS_MPA: Final[tuple[float, float]] = (30.0, 50.0)
WATER_NEAR_TOL: Final[float] = 2.0  # kg/m^3: the decision figure's tests "with water near 185"

Encoder = Callable[[pd.DataFrame], np.ndarray]
Splits = list[tuple[np.ndarray, np.ndarray]]

# ---------------------------------------------------------------------------------------------
# Encoders (Lecture 2's, plus raw inputs with log age)

ENCODERS: Final[Mapping[str, Encoder]] = MappingProxyType(
    {
        "raw kg/m³ and days": raw_features,
        "raw, ln age": raw_log_age,
        "Lecture 2 groups": mix_groups,
    }
)


# ---------------------------------------------------------------------------------------------
# Solvers


def fit_least_squares(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights [w0, w] minimizing ||[1 Z] w - y||^2, via QR (the slide's listing).

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).

    Returns:
        Intercept followed by the D weights, shape (D + 1,).
    """
    A = np.column_stack([np.ones(len(Z)), Z])  # intercept column
    Q, R = np.linalg.qr(A)  # A = QR, Q^T Q = I
    assert Q.shape == A.shape and R.shape == (A.shape[1],) * 2
    return solve_triangular(R, Q.T @ y)  # R w = Q^T y


def fit_normal_inverse(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights [w0, w] from the explicit inverse (X^T X)^{-1} X^T y. For comparison only.

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).

    Returns:
        Intercept followed by the D weights, shape (D + 1,).

    Raises:
        numpy.linalg.LinAlgError: If X^T X is exactly singular in floating point.
    """
    A = np.column_stack([np.ones(len(Z)), Z])
    return np.linalg.inv(A.T @ A) @ A.T @ y


def fit_lstsq(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights [w0, w] from the SVD (minimum-norm solution when columns are dependent).

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).

    Returns:
        Intercept followed by the D weights, shape (D + 1,).
    """
    A = np.column_stack([np.ones(len(Z)), Z])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def predict(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Predictions of the weights [w0, w] on features ``Z``.

    Args:
        w: Intercept followed by the D weights.
        Z: Features, shape (N, D).

    Returns:
        Predictions, shape (N,).
    """
    return w[0] + Z @ w[1:]


def ridge_standardized(Z: np.ndarray, y: np.ndarray, alpha: float) -> tuple[Ridge, StandardScaler]:
    """Ridge on standardized features; scikit-learn leaves the intercept unpenalized.

    Args:
        Z: Training features, shape (N, D), in their own units.
        y: Training targets, shape (N,).
        alpha: The penalty on the standardized weights.

    Returns:
        The fitted ridge model and the scaler fitted on ``Z``.
    """
    scaler = StandardScaler().fit(Z)
    return Ridge(alpha=alpha).fit(scaler.transform(Z), y), scaler


def ridge_cv_alpha(
    Z: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float]
) -> float:
    """Penalty with the lowest grouped 5-fold CV RMSE (standardized ridge, designs held out).

    A choice at the grid's lower edge is accepted only as the lambda -> 0 limit (least
    squares): the smallest penalty must shrink every direction of the standardized training
    features by less than ``RIDGE_LIMIT_TOL``, so a smaller penalty could not change the fit.

    Args:
        Z: Training features, shape (N, D).
        y: Training targets, shape (N,).
        groups: Design id of each training row.
        alphas: Candidate penalties, increasing.

    Returns:
        The chosen penalty.

    Raises:
        ValueError: If the choice is the grid's largest penalty (the grid does not bracket the
            optimum), or the smallest one while that penalty still shrinks the fit.
    """
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(Z, groups=groups))
    cv_err = []
    for alpha in alphas:
        errs = []
        for a, b in folds:
            model, sc = ridge_standardized(Z[a], y[a], alpha)
            errs.append(rmse(model.predict(sc.transform(Z[b])), y[b]))
        cv_err.append(np.mean(errs))
    best = int(np.argmin(cv_err))
    if best == len(alphas) - 1:
        raise ValueError(f"ridge CV chose the grid's largest penalty, {alphas[-1]:g}")
    if best == 0:
        zs = StandardScaler().fit_transform(Z)
        ev_min = float(np.linalg.eigvalsh(zs.T @ zs).min())
        if alphas[0] / (ev_min + alphas[0]) >= RIDGE_LIMIT_TOL:
            raise ValueError(f"ridge CV chose {alphas[0]:g}, which still shrinks the fit")
    return float(alphas[best])


# ---------------------------------------------------------------------------------------------
# Test errors over a set of splits (one value per split)


def ls_errors(Z: np.ndarray, y: np.ndarray, splits: Splits) -> np.ndarray:
    """Test RMSE of least squares (QR) on features ``Z`` in each split.

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).
        splits: (train, test) row indices.

    Returns:
        One test RMSE per split, MPa.
    """
    return np.array(
        [rmse(predict(fit_least_squares(Z[tr], y[tr]), Z[te]), y[te]) for tr, te in splits]
    )


def knn_errors(Z: np.ndarray, y: np.ndarray, splits: Splits, scale: bool = True) -> np.ndarray:
    """Test RMSE of Lecture 2's 5-NN in each split, standardized on the training rows or not.

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).
        splits: (train, test) row indices.
        scale: Standardize on each split's training rows first.

    Returns:
        One test RMSE per split, MPa.
    """
    return np.array([rmse(knn_predict(Z, y, tr, te, scale=scale)[0], y[te]) for tr, te in splits])


def ridge_cv_errors(
    Z: np.ndarray, y: np.ndarray, designs: np.ndarray, splits: Splits
) -> tuple[np.ndarray, np.ndarray]:
    """Test RMSE of standardized ridge with lambda chosen inside each training split.

    Args:
        Z: Features, shape (N, D).
        y: Targets, shape (N,).
        designs: Design id of every row (the CV groups).
        splits: The outer splits.

    Returns:
        The test RMSE and the chosen penalty of each split.
    """
    errs, alphas = [], []
    for tr, te in splits:
        best = ridge_cv_alpha(Z[tr], y[tr], designs[tr], RIDGE_CV_ALPHAS)
        model, sc = ridge_standardized(Z[tr], y[tr], best)
        errs.append(rmse(model.predict(sc.transform(Z[te])), y[te]))
        alphas.append(best)
    return np.array(errs), np.array(alphas)


def mean_baseline_errors(y: np.ndarray, splits: Splits) -> np.ndarray:
    """Test RMSE of predicting the training mean in each split (the naive baseline).

    Args:
        y: Targets, shape (N,).
        splits: (train, test) row indices.

    Returns:
        One test RMSE per split, MPa.
    """
    return np.array([rmse(np.full(len(te), y[tr].mean()), y[te]) for tr, te in splits])


# ---------------------------------------------------------------------------------------------
# The attempt and the hand-solved example


def attempt_mixes(
    df: pd.DataFrame, x: np.ndarray, mpa: np.ndarray, lnf: np.ndarray
) -> list[dict[str, object]]:
    """The attempt's table rows for mixes A-D, as the slide shows them.

    Args:
        df: The concrete data.
        x: w/c of A-D, two decimals.
        mpa: Strength of A-D, one decimal.
        lnf: ln of the shown strength.

    Returns:
        One row per mix: proportions (kg/m^3), w/c, MPa, and ln f.
    """
    return [
        {
            "letter": letter,
            "row": r,
            "cement": round(float(df.at[r, "cement"]), 1),
            "water": round(float(df.at[r, "water"]), 1),
            "wc": float(xi),
            "mpa": float(fi),
            "lnf": round(float(li), 3),
            "coarse": round(float(df.at[r, "coarse_agg"]), 1),
            "fine": round(float(df.at[r, "fine_agg"]), 1),
        }
        for (letter, r), xi, fi, li in zip(ATTEMPT_ROWS.items(), x, mpa, lnf, strict=True)
    ]


def attempt_rows(
    df: pd.DataFrame, train: np.ndarray, test: np.ndarray
) -> tuple[pd.DataFrame, pd.Series]:
    """Mixes A-D and P, checked: one series (28 days, water 185, no admixtures), A-D in training.

    Args:
        df: The concrete data.
        train: The first split's training rows.
        test: Its test rows.

    Returns:
        The rows of A-D and the row of P.
    """
    rows = list(ATTEMPT_ROWS.values())
    sub = df.loc[rows]
    assert set(rows) <= set(train) and PROPOSED_ROW in set(test)
    assert (sub["age"] == 28).all() and np.allclose(sub["water"], ATTEMPT_WATER)
    assert (sub[["slag", "fly_ash", "superplasticizer"]] == 0).all().all()
    p = df.loc[PROPOSED_ROW]
    assert p["age"] == 28 and p[["slag", "fly_ash", "superplasticizer"]].eq(0).all()
    return sub, p


def attempt(
    df: pd.DataFrame, train: np.ndarray, test: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Four tested mixes and the proposed mix; Abrams' law fitted by hand on the four.

    The hand example uses the values as the slide shows them: w/c to two decimals and strength
    to one decimal, with ln f computed from the shown strength. Centering w/c makes the two
    columns of X orthogonal, so the normal equations decouple.

    Args:
        df: The concrete data.
        train: The first split's training rows.
        test: Its test rows.
        out_dir: The figure directory.

    Returns:
        The attempt's table and the hand fit; ``_`` keys hold raw arrays for later figures.
    """
    sub, p = attempt_rows(df, train, test)

    x = np.round((sub["water"] / sub["cement"]).to_numpy(), 2)  # w/c as shown
    mpa = np.round(sub["strength"].to_numpy(), 1)  # MPa as shown
    lnf = np.log(mpa)
    X = np.column_stack([np.ones(4), x])
    w = np.linalg.solve(X.T @ X, X.T @ lnf)  # tiny and well conditioned: fine here
    xbar, ybar = float(x.mean()), float(lnf.mean())
    dx, dy = x - xbar, lnf - ybar
    slope = float(dx @ dy / (dx @ dx))
    assert np.isclose(slope, w[1]) and np.isclose(ybar - slope * xbar, w[0])
    a_const, b_const = float(np.exp(w[0])), float(np.exp(-w[1]))
    wc_p = round(float(p["water"] / p["cement"]), 2)
    f_p = float(np.exp(w[0] + w[1] * wc_p))
    wc_35 = float((np.log(FC) - w[0]) / w[1])
    resid = lnf - X @ w
    attempt_figure(x, mpa, w, wc_p=wc_p, f_p=f_p, out_dir=out_dir)

    return {
        "mixes": attempt_mixes(df, x, mpa, lnf),
        "proposed": {
            "row": PROPOSED_ROW,
            "cement": float(p["cement"]),
            "water": float(p["water"]),
            "coarse": float(p["coarse_agg"]),
            "fine": float(p["fine_agg"]),
            "wc": wc_p,
            "mpa": round(float(p["strength"]), 1),
        },
        "xbar": round(xbar, 3),
        "ybar": round(ybar, 3),
        "dx": [round(float(v), 3) for v in dx],
        "sxx": round(float(dx @ dx), 4),
        "sxy": round(float(dx @ dy), 4),
        "slope": round(slope, 2),
        "intercept": round(float(w[0]), 2),
        "abrams_A": round(a_const, 1),
        "abrams_B": round(b_const, 1),
        "pred_P": round(f_p, 1),
        "err_P": round(f_p - float(p["strength"]), 1),
        # A student's straight-line interpolation between A and B at P's w/c.
        "interp_P": round(float(np.interp(wc_p, x[:2], mpa[:2])), 1),  # x increases A to D
        # The notes divide 185 by the three-decimal w/c: 185 / 0.511 = 362.
        "wc_for_fc": round(wc_35, 2),
        "wc_for_fc_3dp": round(wc_35, 3),
        "cement_for_fc": int(round(ATTEMPT_WATER / wc_35)),
        "max_abs_resid_ln": round(float(np.abs(resid).max()), 3),
        "_x": x,
        "_lnf": lnf,
        "_w": w,
    }


def attempt_figure(
    x: np.ndarray, mpa: np.ndarray, w: np.ndarray, *, wc_p: float, f_p: float, out_dir: Path
) -> None:
    """The four mixes, the fitted Abrams line on a log axis, and the prediction for P.

    Args:
        x: w/c of A-D as shown.
        mpa: Strength of A-D as shown, MPa.
        w: Fitted [a, b] of ln f = a + b (w/c).
        wc_p: P's w/c as shown.
        f_p: The line's prediction for P, MPa.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    grid = np.linspace(0.42, 0.78, 50)
    ax.plot(
        grid,
        np.exp(w[0] + w[1] * grid),
        color=aml_style.INK,
        lw=1.6,
        label=rf"$\ln f = {w[0]:.2f} {w[1]:+.2f}\,(w/c)$",
    )
    for j, (letter, xi, fi) in enumerate(zip(ATTEMPT_ROWS, x, mpa, strict=True)):
        fit_i = np.exp(w[0] + w[1] * xi)
        ax.plot([xi, xi], [fi, fit_i], color=aml_style.PALETTE[j], lw=1.5, ls=":")
        ax.plot(xi, fi, marker=aml_style.MARKERS[j], ms=10, ls="none", color=aml_style.PALETTE[j])
        ax.annotate(
            f"{letter}: {fi:.1f}", (xi, fi), xytext=(8, 2), textcoords="offset points", fontsize=12
        )
    ax.plot(wc_p, f_p, marker="o", ms=10, mfc="white", mec=aml_style.INK, ls="none")
    ax.annotate(
        f"P: {f_p:.1f}",
        (wc_p, f_p),
        xytext=(10, 4),
        textcoords="offset points",
        fontsize=12,
        ha="left",
    )
    ax.axhline(FC, color=aml_style.MUTED, ls="--", lw=1.2)
    ax.text(
        0.78,
        FC * 1.02,
        f"{FC:.0f} MPa",
        ha="right",
        va="bottom",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.set_yscale("log")
    ax.set_yticks([15, 20, 30, 40, 50], ["15", "20", "30", "40", "50"])
    ax.minorticks_off()
    ax.set_ylim(15, 52)
    ax.set_xlim(0.42, 0.80)
    ax.set_xlabel("water–cement ratio $w/c$")
    ax.set_ylabel("28-day strength (MPa, log scale)")
    ax.legend(loc="upper right", fontsize=12)
    aml_style.save(fig, out_dir, "attempt_fit")


# ---------------------------------------------------------------------------------------------
# Build and verify


def verify_listing(df: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> dict[str, object]:
    """Compare QR, the SVD, the explicit inverse, and scikit-learn on the first split.

    Relative differences are max |w - w_ref| / |w_ref| over the weights, against scikit-learn's
    ``LinearRegression`` (an SVD-based LAPACK solver).

    Args:
        df: The concrete data.
        train: The first split's training rows.
        test: Its test rows.

    Returns:
        Condition numbers and relative differences per encoder and solver, the six-group
        weights, the split's sizes, and its test RMSE.
    """
    y = df["strength"].to_numpy()
    out: dict[str, object] = {}
    for name, enc in (("groups", mix_groups), ("raw", raw_features)):
        Z = enc(df)
        ref = LinearRegression().fit(Z[train], y[train])
        r = np.r_[ref.intercept_, ref.coef_]
        A = np.column_stack([np.ones(len(train)), Z[train]])
        res = {
            "kappa_X": float(f"{np.linalg.cond(A):.2g}"),
            "kappa_XtX": float(f"{np.linalg.cond(A.T @ A):.2g}"),
        }
        for solver, fit in (
            ("qr", fit_least_squares),
            ("svd", fit_lstsq),
            ("inverse", fit_normal_inverse),
        ):
            w = fit(Z[train], y[train])
            rel = np.max(np.abs(w - r) / (np.abs(r) + 1e-12))  # no reference weight is 0
            res[f"{solver}_rel"] = float(f"{rel:.1g}")
            res[f"{solver}_allclose"] = bool(np.allclose(w, r, rtol=1e-8, atol=0))
        out[name] = res
    Z = mix_groups(df)
    w = fit_least_squares(Z[train], y[train])
    out["weights"] = {
        "intercept": round(float(w[0]), 1),
        **{n: round(float(v), 2) for n, v in zip(GROUP_NAMES, w[1:], strict=True)},
    }
    out["n_train"] = int(len(train))
    out["n_train_designs"] = int(df["design"].iloc[train].nunique())
    out["n_test"] = int(len(test))
    out["test_rmse_split0"] = round(rmse(predict(w, Z[test]), y[test]), 2)
    raw_w = fit_least_squares(raw_features(df)[train], y[train])
    out["raw_water_weight"] = round(float(raw_w[1 + RAW_INPUTS.index("water")]), 2)
    out["raw_cement_weight"] = round(float(raw_w[1 + RAW_INPUTS.index("cement")]), 2)
    assert w[1] < 0 and raw_w[1 + RAW_INPUTS.index("water")] < 0, "water should lower strength"
    return out


def trace(df: pd.DataFrame, train: np.ndarray) -> dict[str, object]:
    """Follow the proposed mix P through the six-group least-squares model of the first split.

    Args:
        df: The concrete data.
        train: The first split's training rows (P is not among them).

    Returns:
        P's groups, the intercept, each term w_j z_j, the prediction, the measurement, and the
        error. The ``_`` keys keep the raw values for the diagram's macros (not written to
        numbers.json).
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    w = fit_least_squares(Z[train], y[train])
    z = Z[PROPOSED_ROW]
    terms = w[1:] * z
    pred = float(w[0] + terms.sum())
    meas = float(y[PROPOSED_ROW])
    return {
        "z": dict(zip(GROUP_NAMES, (round(float(v), 2) for v in z), strict=True)),
        "intercept": round(float(w[0]), 1),
        "terms": dict(zip(GROUP_NAMES, (round(float(v), 1) for v in terms), strict=True)),
        "pred": round(pred, 1),
        "y": round(meas, 1),
        "err": round(pred - meas, 1),
        "_z": dict(zip(GROUP_NAMES, z, strict=True)),
        "_terms": dict(zip(GROUP_NAMES, terms, strict=True)),
        "_intercept": float(w[0]),
        "_pred": pred,
        "_y": meas,
    }


# ---------------------------------------------------------------------------------------------
# Break it


def features_rmse(df: pd.DataFrame, out_dir: Path, floor: dict[str, object]) -> dict[str, object]:
    """Least squares on three encoders, designs held out, bracketed by the mean and the floor.

    Also: ridge with lambda chosen inside each training split (grouped 5-fold CV), the random
    split for the groups model (leakage), the groups model's training error, a log target, and
    Lecture 2's 5-NN for reference. Error bars are the sample standard deviation (ddof = 1) of
    the 30 splits' RMSEs, as in Lecture 2.

    Args:
        df: The concrete data.
        out_dir: The figure directory.
        floor: ``repeat_floor``'s result.

    Returns:
        Mean and SD of the test RMSE per encoder, and the side results, MPa.
    """
    y = df["strength"].to_numpy()
    splits, rsplits = grouped_splits(df), random_splits(df)
    raw = {name: ls_errors(enc(df), y, splits) for name, enc in ENCODERS.items()}
    out: dict[str, object] = {
        name: {"mean": rnd(errs.mean(), 2), "sd": rnd(errs.std(ddof=1), 2)}
        for name, errs in raw.items()
    }
    Z = mix_groups(df)
    out["groups_random_split"] = rnd(ls_errors(Z, y, rsplits).mean(), 2)
    # Seven weights fit the training designs barely better than new ones (memorizing nothing).
    out["groups_train_rmse"] = rnd(
        np.mean([rmse(predict(fit_least_squares(Z[tr], y[tr]), Z[tr]), y[tr]) for tr, _ in splits]),
        2,
    )
    knn = float(knn_errors(Z, y, splits).mean())
    out["knn_groups"] = rnd(knn, 2)
    out["knn_groups_random_split"] = rnd(knn_errors(Z, y, rsplits).mean(), 2)
    # A log target (Abrams' form) on the same groups, back-transformed to MPa.
    out["groups_log_target"] = rnd(
        np.mean(
            [
                rmse(np.exp(predict(fit_least_squares(Z[tr], np.log(y[tr])), Z[te])), y[te])
                for tr, te in splits
            ]
        ),
        2,
    )
    ridge_err, ridge_alpha = ridge_cv_errors(Z, y, df["design"].to_numpy(), splits)
    alpha_med = float(np.median(ridge_alpha))
    out["ridge_groups"] = rnd(ridge_err.mean(), 2)
    out["ridge_alpha_median"] = rnd(alpha_med, 2)
    # Lower-edge picks are the lambda -> 0 limit (checked in ridge_cv_alpha); none at the top.
    out["ridge_alpha_lower_edge"] = int(np.sum(ridge_alpha == RIDGE_CV_ALPHAS[0]))
    # How much the median penalty shrinks the weakest direction of split 0's standardized groups.
    zs = StandardScaler().fit_transform(Z[splits[0][0]])
    ev_min = float(np.linalg.eigvalsh(zs.T @ zs).min())
    out["ridge_max_shrink_pct"] = rnd(100 * alpha_med / (ev_min + alpha_med), 1)
    base = float(mean_baseline_errors(y, splits).mean())
    out["predict_mean"] = rnd(base, 2)
    out["floor"] = floor["floor"]
    out["floor_groups"] = floor["groups"]
    out["floor_mixes"] = floor["mixes"]
    features_figure(raw, knn=knn, base=base, floor=float(floor["floor"]), out_dir=out_dir)
    return out


def features_figure(
    errs: dict[str, np.ndarray], *, knn: float, base: float, floor: float, out_dir: Path
) -> None:
    """Dot plot of the three encoders' RMSE with 5-NN, the mean baseline, and the floor.

    Args:
        errs: Each encoder's test RMSE per split, unrounded (labels round once, here).
        knn: Lecture 2's 5-NN on the groups, mean test RMSE.
        base: Predicting the training mean, mean test RMSE.
        floor: The scatter of repeat tests.
        out_dir: The figure directory.
    """
    names = list(ENCODERS)
    means = [float(errs[n].mean()) for n in names]
    sds = [float(errs[n].std(ddof=1)) for n in names]
    labels = ["LS: raw kg/m³, days", "LS: raw kg/m³, ln age", "LS: Lecture 2 groups"]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    pos = np.arange(len(names))[::-1] + 1
    ax.errorbar(means, pos, xerr=sds, fmt="o", color=aml_style.PALETTE[0], capsize=4, ms=8)
    for p_, m in zip(pos, means, strict=True):
        ax.text(m, p_ + 0.25, f"{m:.1f}", ha="center", fontsize=12)
    ax.plot(knn, 0, marker="s", ms=8, mfc="white", mec=aml_style.PALETTE[1], ls="none", mew=1.8)
    ax.text(knn, 0.25, f"{knn:.1f}", ha="center", fontsize=12)
    ax.axvline(base, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(base - 0.4, -0.75, "predict the mean", ha="right", fontsize=12, color=aml_style.MUTED)
    ax.axvline(floor, color=aml_style.MUTED, ls="--", lw=1.5)
    ax.text(floor + 0.3, -0.75, "repeat tests", ha="left", fontsize=12, color=aml_style.MUTED)
    ax.set_yticks([*pos, 0], [*labels, "5-NN: Lecture 2 groups"])
    ax.set_xlim(0, 18)
    ax.set_ylim(-1.1, len(names) + 0.6)
    ax.set_xlabel("test RMSE (MPa), designs held out")
    aml_style.save(fig, out_dir, "features_rmse")


def lasso_features(df: pd.DataFrame, rows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """The 14 features offered to lasso: 8 raw inputs, ln age, then the 5 ratios.

    Args:
        df: The concrete data.
        rows: The rows to return.

    Returns:
        The features of ``rows``, shape (len(rows), 14), and their names.
    """
    F = np.column_stack([raw_features(df), mix_groups(df)])[rows]
    names = [*RAW_INPUTS, *GROUP_NAMES[5:], *GROUP_NAMES[:5]]
    F = np.column_stack([F[:, :8], F[:, 13], F[:, 8:13]])  # raw 8, ln age, then 5 ratios
    if F.shape != (len(rows), 14) or len(names) != 14:
        raise ValueError(f"lasso features of shape {F.shape}, {len(names)} names")
    return F, names


def lasso_cv(F: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    """Lasso's penalty by grouped 5-fold CV over a bracketing grid, and the weights it keeps.

    The grid starts as LassoCV's default: ``LASSO_GRID_N`` penalties from the one that zeroes
    every weight down to ``LASSO_GRID_DECADES`` decades below it, at a fixed log spacing. While
    the best penalty is the smallest, the grid grows by one decade at that spacing, down to
    ``LASSO_LADDER_DECADES``; a choice still at that floor is the lambda -> 0 limit (least
    squares). The scaler is refitted inside each fold (a pipeline).

    Args:
        F: Training features, shape (N, D), in their own units.
        y: Training targets, shape (N,).
        groups: Design id of each training row.

    Returns:
        The chosen penalty relative to the largest (``alpha_rel``), its grid index and the
        grid's size, whether it is the lambda -> 0 limit, and the weights of the refitted model.

    Raises:
        ValueError: If the best penalty is the largest one (every weight zero).
    """
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(F, groups=groups))
    Fs = StandardScaler().fit_transform(F)
    alpha_max = float(np.abs(Fs.T @ (y - y.mean())).max() / len(y))
    step = LASSO_GRID_DECADES / (LASSO_GRID_N - 1)  # decades between grid points
    exps = np.linspace(0.0, -LASSO_GRID_DECADES, LASSO_GRID_N)  # as np.logspace builds it
    scores = np.empty(0)
    while True:
        new = exps[len(scores) :]
        search = GridSearchCV(
            make_pipeline(StandardScaler(), Lasso(max_iter=100_000)),
            {"lasso__alpha": alpha_max * 10.0**new},
            cv=folds,
            scoring="neg_mean_squared_error",
            refit=False,
        ).fit(F, y)
        scores = np.r_[scores, search.cv_results_["mean_test_score"]]
        best = int(np.argmax(scores))  # first of any ties, as GridSearchCV ranks them
        if best == 0:
            raise ValueError("lasso CV chose the penalty that zeroes every weight")
        if best < len(exps) - 1 or -exps[-1] >= LASSO_LADDER_DECADES - 1e-9:
            break
        exps = np.r_[exps, exps[-1] - step * np.arange(1, round(1 / step) + 1)]
    model = make_pipeline(
        StandardScaler(), Lasso(alpha=alpha_max * 10.0 ** exps[best], max_iter=100_000)
    )
    coef = model.fit(F, y)[-1].coef_
    return {
        "alpha_rel": float(10.0 ** exps[best]),
        "index": best,
        "grid_size": int(len(exps)),
        "limit": best == len(exps) - 1,
        "coef": coef,
    }


def lasso_order(df: pd.DataFrame, splits: Splits) -> dict[str, object]:
    """Order in which lasso admits 14 standardized features (8 raw, ln age, 5 groups).

    On the first split: the order along the path, and the features dropped at the penalty
    chosen by grouped 5-fold CV (``lasso_cv``). Over all 30 splits: which features the CV
    penalty drops, to show that with correlated features which one survives is unstable (Zou &
    Hastie, 2005).

    Args:
        df: The concrete data.
        splits: The 30 design-held-out splits; the first is the slides' split.

    Returns:
        The admission order and the first split's kept and dropped features; over the splits,
        how often w/b is dropped, the distinct dropped sets, and the kept counts.
    """
    designs = df["design"].to_numpy()
    strength = df["strength"].to_numpy()
    train = splits[0][0]
    F, names = lasso_features(df, train)
    y = strength[train]
    Fs = StandardScaler().fit_transform(F)
    alphas, coefs, _ = lasso_path(Fs, y - y.mean(), n_alphas=400, eps=1e-4)
    first = {
        n: float(alphas[np.flatnonzero(coefs[j])[0]])
        for j, n in enumerate(names)
        if np.any(coefs[j])
    }
    order = sorted(first, key=first.get, reverse=True)
    per_split = []
    for tr, _ in splits:
        Ft, _ = lasso_features(df, tr)
        res = lasso_cv(Ft, strength[tr], designs[tr])
        dropped = [n for n, c in zip(names, res["coef"], strict=True) if c == 0]
        per_split.append({**res, "dropped": dropped})
    s0 = per_split[0]
    if s0["index"] in (0, s0["grid_size"] - 1):
        raise ValueError("the first split's lasso penalty is not interior to its grid")
    sets: dict[tuple[str, ...], int] = {}
    for r in per_split:
        sets[tuple(r["dropped"])] = sets.get(tuple(r["dropped"]), 0) + 1
    kept = [len(names) - len(r["dropped"]) for r in per_split]
    return {
        "order": order,
        "first_two": order[:2],
        "cv_kept": kept[0],
        "cv_dropped": s0["dropped"],
        "cv_index": s0["index"],
        "cv_grid_size": s0["grid_size"],
        "splits": {
            "n": len(per_split),
            "wb_dropped": sum("w/b" in r["dropped"] for r in per_split),
            "distinct_sets": len(sets),
            "most_common_set_splits": max(sets.values()),
            "kept_min": min(kept),
            "kept_max": max(kept),
            "none_dropped": sum(not r["dropped"] for r in per_split),
            "grid_extended": sum(r["grid_size"] > LASSO_GRID_N for r in per_split),
            "lambda_to_zero": sum(bool(r["limit"]) for r in per_split),
        },
    }


def collinear_fits(
    X: np.ndarray, Xb: np.ndarray, y: np.ndarray, splits: Splits
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Cement weight and test RMSE of each solver in each split, with and without binder.

    Args:
        X: The 8 raw inputs, shape (N, 8); cement is column 0.
        Xb: The same plus binder content, shape (N, 9).
        y: Strengths, shape (N,).
        splits: The design-held-out splits.

    Returns:
        Cement weight (MPa per kg/m^3) and test RMSE, per fit, one value per split.
    """
    cem = RAW_INPUTS.index("cement")
    weights: dict[str, list[float]] = {k: [] for k in ("8 inputs", "inverse", "QR", "SVD", "ridge")}
    errs: dict[str, list[float]] = {k: [] for k in weights}
    for tr, te in splits:
        for name, fit, M in (
            ("8 inputs", fit_least_squares, X),
            ("inverse", fit_normal_inverse, Xb),
            ("QR", fit_least_squares, Xb),
            ("SVD", fit_lstsq, Xb),
        ):
            w = fit(M[tr], y[tr])
            weights[name].append(float(w[1 + cem]))
            errs[name].append(rmse(predict(w, M[te]), y[te]))
        model, sc = ridge_standardized(Xb[tr], y[tr], RIDGE_COLLINEAR_ALPHA)
        weights["ridge"].append(float(model.coef_[cem] / sc.scale_[cem]))
        errs["ridge"].append(rmse(model.predict(sc.transform(Xb[te])), y[te]))
    return weights, errs


def collinear(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Add binder content (cement + slag + fly ash) as a ninth input: an exact dependency.

    Compares the explicit inverse, QR, the SVD (``lstsq``), and standardized ridge across the
    30 design-held-out splits: the cement weight and the test RMSE. The inverse's and QR's
    digits are round-off (they change with the BLAS, the machine, or the column order), so the
    slides quote only the qualitative outcomes checked here, which hold whenever round-off leaves
    the smallest pivot nonzero (an exact zero would make ``inv`` and the triangular solve raise);
    the digits of this run are kept under ``roundoff`` for the record.

    Args:
        df: The concrete data.
        out_dir: The figure directory.

    Returns:
        Rank and size of the design, the cement weight's range per fit, the RMSEs, the checks,
        and this run's round-off digits.

    Raises:
        AssertionError: If a qualitative outcome the slides quote fails.
    """
    y = df["strength"].to_numpy()
    X = raw_features(df)
    binder = (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()
    Xb = np.column_stack([X, binder])
    assert Xb.shape == (len(df), len(RAW_INPUTS) + 1), Xb.shape
    splits = grouped_splits(df)
    A0 = np.column_stack([np.ones(len(splits[0][0])), Xb[splits[0][0]]])
    weights, errs = collinear_fits(X, Xb, y, splits)
    collinear_figure(weights, out_dir)

    kappa = float(np.linalg.cond(A0))
    qr_max = float(np.max(np.abs(fit_least_squares(Xb[splits[0][0]], y[splits[0][0]]))))
    inv_rmse = float(np.mean(errs["inverse"]))
    baseline = float(mean_baseline_errors(y, splits).mean())
    checks = {
        "kappa_X_above_1e15": kappa > 1e15,
        "inverse_weight_takes_both_signs": min(weights["inverse"]) < 0 < max(weights["inverse"]),
        "inverse_rmse_above_predict_mean": inv_rmse > baseline,
        "qr_max_abs_weight_above_1e10": qr_max > 1e10,
    }
    assert all(checks.values()), checks
    svd_equals_8 = bool(np.allclose(errs["SVD"], errs["8 inputs"], rtol=1e-6))
    assert svd_equals_8

    def span(v: list[float]) -> list[float]:
        """Smallest and largest value, two decimals."""
        return [round(min(v), 2), round(max(v), 2)]

    return {
        "columns": int(A0.shape[1]),
        "rank": int(np.linalg.matrix_rank(A0)),
        "cement_weight_8": span(weights["8 inputs"]),
        "cement_weight_svd": span(weights["SVD"]),
        "cement_weight_ridge": span(weights["ridge"]),
        "rmse": {k: round(float(np.mean(errs[k])), 2) for k in ("8 inputs", "SVD", "ridge")},
        "svd_equals_8_inputs": svd_equals_8,
        "checks": checks,
        "roundoff": {
            "kappa_X": float(f"{kappa:.2g}"),
            "cement_weight_inverse": span(weights["inverse"]),
            "rmse_inverse": round(inv_rmse, 2),
            "rmse_QR": round(float(np.mean(errs["QR"])), 2),
            "qr_max_abs_weight_split0": float(f"{qr_max:.1g}"),
            # Orders of magnitude the slides quote ("~1e16", "~1e12"); they depend on the BLAS.
            "kappa_X_order": float(10 ** np.floor(np.log10(kappa))),
            "qr_max_abs_weight_order": float(10 ** np.floor(np.log10(qr_max))),
        },
    }


def collinear_figure(weights: dict[str, list[float]], out_dir: Path) -> None:
    """Strip plot of the cement weight over the 30 splits, for four fits.

    Args:
        weights: Cement weight per split, by fit.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    shown = ["8 inputs", "inverse", "SVD", "ridge"]
    labels = ["8 inputs, QR", "+ binder, inverse", "+ binder, SVD", "+ binder, ridge"]
    rng = np.random.default_rng(SEED)
    for j, name in enumerate(shown):
        yy = len(shown) - 1 - j + rng.uniform(-0.15, 0.15, REPEATS)
        ax.plot(
            weights[name],
            yy,
            marker=aml_style.MARKERS[j],
            ls="none",
            ms=7,
            alpha=0.8,
            color=aml_style.PALETTE[j],
        )
    ax.axvline(0, color=aml_style.MUTED, lw=1)
    ax.set_yticks(range(len(shown))[::-1], labels)
    ax.set_xlabel("weight of cement (MPa per kg/m³), 30 splits")
    # The inverse's weights are round-off: widen the axis if they land farther out.
    lim = max(2.0, 1.05 * max(abs(v) for name in shown for v in weights[name]))
    ax.set_xlim(-lim, lim)
    aml_style.save(fig, out_dir, "collinear_weights")


def design_bootstrap(
    preds: dict[str, np.ndarray], y_te: np.ndarray, designs: np.ndarray
) -> dict[str, list[float]]:
    """Intervals that resample whole designs: the raw model's bias and its RMSE gap to groups.

    Tests of one design are not independent, so each draw takes the test set's designs with
    replacement (``BOOT_DRAWS`` draws, seed ``BOOT_SEED``) and keeps all of their tests.

    Args:
        preds: Each model's predictions for the test rows, MPa (needs "raw, ln age" and
            "Lecture 2 groups").
        y_te: Measured strength of the test rows, MPa.
        designs: Design id of each test row.

    Returns:
        ``bias_ci``: the raw-with-ln-age model's mean error; ``rmse_gap_ci``: its RMSE minus the
        groups model's; each as [low, high] at ``CI_LEVEL``, MPa.
    """
    ids, inv = np.unique(designs, return_inverse=True)
    n_rows = np.bincount(inv).astype(float)
    err_raw = preds["raw, ln age"] - y_te
    err_grp = preds["Lecture 2 groups"] - y_te
    sum_raw = np.bincount(inv, weights=err_raw)
    sse_raw = np.bincount(inv, weights=err_raw**2)
    sse_grp = np.bincount(inv, weights=err_grp**2)
    rng = np.random.default_rng(BOOT_SEED)
    picks = rng.integers(0, len(ids), size=(BOOT_DRAWS, len(ids)))
    counts = np.stack([np.bincount(row, minlength=len(ids)) for row in picks]).astype(float)
    assert counts.shape == (BOOT_DRAWS, len(ids))
    n = counts @ n_rows
    bias = counts @ sum_raw / n
    gap = np.sqrt(counts @ sse_raw / n) - np.sqrt(counts @ sse_grp / n)
    q = 100 * np.array([(1 - CI_LEVEL) / 2, (1 + CI_LEVEL) / 2])
    return {
        "bias_ci": [rnd(v, 1) for v in np.percentile(bias, q)],
        "rmse_gap_ci": [rnd(v, 1) for v in np.percentile(gap, q)],
    }


def extrapolation(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Train on mixes with cement <= ``CEMENT_TRAIN_MAX``; test those above ``CEMENT_TEST_MIN``.

    The decisive figure: raw kilograms (with ln age) over-predict the high-cement mixes, the
    Lecture 2 groups stay close, and 5-NN under-predicts: it averages five lower-cement
    neighbours, and many tests measure above all five (``knn``). The test set
    is a few dozen designs, so the raw model's bias and its RMSE gap to the groups get
    intervals that resample designs (``design_bootstrap``); the bias is also given at 28 days.

    Args:
        df: The concrete data.
        out_dir: The figure directory.

    Returns:
        Set sizes, each model's RMSE and mean error (all ages and 28 days), the intervals, the
        share of test mixes inside the training ratios, and Abrams' gain per kg of cement.
    """
    y = df["strength"].to_numpy()
    cement = df["cement"].to_numpy()
    tr = np.flatnonzero(cement <= CEMENT_TRAIN_MAX)
    te = np.flatnonzero(cement > CEMENT_TEST_MIN)
    preds: dict[str, np.ndarray] = {}
    cement_weight = 0.0
    for name, enc in (
        ("raw, ln age", raw_log_age),
        ("raw kg/m³ and days", raw_features),
        ("Lecture 2 groups", mix_groups),
    ):
        Z = enc(df)
        w = fit_least_squares(Z[tr], y[tr])
        preds[name] = predict(w, Z[te])
        if name == "raw, ln age":  # MPa per kg/m^3 of cement (cement is a raw input's column)
            cement_weight = float(w[1 + RAW_INPUTS.index("cement")])
    Z = mix_groups(df)
    preds["5-NN"], neighbours = knn_predict(Z, y, tr, te)
    above_all = y[te] > y[neighbours].max(axis=1)  # stronger than every one of its neighbours
    inside = (Z[te] >= Z[tr].min(axis=0)) & (Z[te] <= Z[tr].max(axis=0))
    assert inside.shape == (len(te), len(GROUP_NAMES))
    d28 = df["age"].to_numpy()[te] == 28
    # Abrams (28-day fit, Lecture 2): strength gained per kg of cement at fixed water.
    a_const, b_const = abrams_constants(df)

    def per_kg(c: float) -> float:
        """Abrams' d f / d cement at ``ATTEMPT_WATER`` of water and cement ``c`` (kg/m^3)."""
        r = ATTEMPT_WATER / c
        return float(a_const * np.log(b_const) * r / c * b_const ** (-r))

    extrapolation_figure(y[te], preds, out_dir)
    rmse_all = {k: rmse(v, y[te]) for k, v in preds.items()}
    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_test_designs": int(df["design"].iloc[te].nunique()),
        "n_test_mixes": int(df["mix"].iloc[te].nunique()),
        "n_test_28day": int(d28.sum()),
        "rmse": {k: rnd(v, 1) for k, v in rmse_all.items()},
        "bias": {k: rnd(np.mean(v - y[te]), 1) for k, v in preds.items()},
        "bias_28day": {k: rnd(np.mean(v[d28] - y[te][d28]), 1) for k, v in preds.items()},
        "rmse_gap": rnd(rmse_all["raw, ln age"] - rmse_all["Lecture 2 groups"], 1),
        **design_bootstrap(preds, y[te], df["design"].to_numpy()[te]),
        "knn": {
            "train_max": rnd(y[tr].max(), 1),
            "test_above_train_max": int(np.sum(y[te] > y[tr].max())),
            "test_above_all_neighbours": int(above_all.sum()),
            "test_within_neighbour_range": int(np.sum(~above_all)),
            "bias_within_neighbour_range": rnd(np.mean((preds["5-NN"] - y[te])[~above_all]), 1),
        },
        "wb_inside_share": rnd(inside[:, 0].mean(), 2),
        "all_ratios_inside_share": rnd(inside.all(axis=1).mean(), 2),
        "abrams_mpa_per_kg_300": rnd(per_kg(ABRAMS_CEMENT_LOW), 2),
        "abrams_mpa_per_kg_500": rnd(per_kg(ABRAMS_CEMENT_HIGH), 2),
        "raw_ln_age_cement_weight": rnd(cement_weight, 2),
        "cement_max": float(cement.max()),
    }


def extrapolation_figure(y_te: np.ndarray, preds: dict[str, np.ndarray], out_dir: Path) -> None:
    """Predicted against measured strength of the high-cement mixes, with each model's bias.

    Args:
        y_te: Measured strength of the test mixes, MPa.
        preds: Each model's predictions for them, MPa.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    lim = (0, 95)
    ax.plot(lim, lim, color=aml_style.MUTED, lw=1.2, ls=":")
    shown = (
        ("raw, ln age", "raw kg/m³, ln age", "s"),
        ("Lecture 2 groups", "Lecture 2 groups", "o"),
        ("5-NN", "5-NN, groups", "^"),
    )
    for j, (name, label, mk) in enumerate(shown):
        bias = float(np.mean(preds[name] - y_te))
        ax.scatter(
            y_te,
            preds[name],
            s=26,
            marker=mk,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            color=aml_style.PALETTE[[1, 0, 2][j]],
            label=f"{label}: {tex_num(bias, '+.1f')}",
        )
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"measured strength, cement > {CEMENT_TEST_MIN:.0f} (MPa)")
    ax.set_ylabel("prediction (MPa)")
    ax.legend(
        loc="upper left",
        fontsize=12,
        handletextpad=0.1,
        frameon=True,  # opaque: the points run up into the corner
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        title="mean error (MPa)",
        title_fontsize=12,
        alignment="left",
    )
    aml_style.save(fig, out_dir, "cement_extrapolation")


def ridge_units(df: pd.DataFrame) -> dict[str, float | int]:
    """One ridge penalty, age in days or in years, with and without standardizing.

    Fitted on all 1,030 tests, as an assistant asked for "a ridge regression" would; predicts
    the query mix. Also scikit-learn's default penalty, alpha = 1, unscaled in days.

    Args:
        df: The concrete data.

    Returns:
        The query mix's prediction (MPa) for least squares and for each ridge variant, and the
        share of age's least-squares weight that unscaled ridge keeps in years (percent).

    Raises:
        AssertionError: If the standardized fits depend on the units.
    """
    y = df["strength"].to_numpy()
    X = raw_features(df)
    q = np.array([[QUERY[c] for c in RAW_INPUTS]])
    age = RAW_INPUTS.index("age")
    Xy, qy = X.copy(), q.copy()
    Xy[:, age] /= DAYS_PER_YEAR
    qy[:, age] /= DAYS_PER_YEAR
    ols = LinearRegression().fit(X, y)
    a = RIDGE_UNITS_ALPHA
    days = Ridge(alpha=a).fit(X, y)
    years = Ridge(alpha=a).fit(Xy, y)
    s_days, sc_d = ridge_standardized(X, y, a)
    s_years, sc_y = ridge_standardized(Xy, y, a)
    out: dict[str, float | int] = {
        "alpha": a,
        "ols": round(float(ols.predict(q)[0]), 1),
        "unscaled_days": round(float(days.predict(q)[0]), 1),
        "unscaled_years": round(float(years.predict(qy)[0]), 1),
        "scaled_days": round(float(s_days.predict(sc_d.transform(q))[0]), 1),
        "scaled_years": round(float(s_years.predict(sc_y.transform(qy))[0]), 1),
        "unscaled_days_alpha1": round(float(Ridge(alpha=1.0).fit(X, y).predict(q)[0]), 1),
        "age_weight_kept_years_pct": int(
            round(100 * years.coef_[age] / DAYS_PER_YEAR / ols.coef_[age])
        ),
    }
    assert np.isclose(out["scaled_days"], out["scaled_years"])
    return out


def audit_key(df: pd.DataFrame) -> dict[str, object]:
    """Reference answers for the delegated question: the query mix, fitted on all tests.

    Args:
        df: The concrete data.

    Returns:
        The query, each model's prediction for it (MPa), its w/c and aggregate/b against the
        tested ranges, and the 28-day tests with w/b within ``NEAR_WB_WINDOW`` of it.
    """
    y = df["strength"].to_numpy()
    qdf = pd.DataFrame([dict(QUERY)])
    out: dict[str, object] = {"query": dict(QUERY)}
    for name, enc in ENCODERS.items():
        out[name] = round(float(predict(fit_least_squares(enc(df), y), enc(qdf))[0]), 1)
    Z = mix_groups(df)
    sc = StandardScaler().fit(Z)
    out["5-NN"] = round(
        float(
            KNeighborsRegressor(n_neighbors=K)
            .fit(sc.transform(Z), y)
            .predict(sc.transform(mix_groups(qdf)))[0]
        ),
        1,
    )
    zq = mix_groups(qdf)[0]
    out["query_wc"] = round(float(zq[0]), 3)
    out["wb_range"] = [round(float(Z[:, 0].min()), 2), round(float(Z[:, 0].max()), 2)]
    out["agg_b_query"] = round(float(zq[4]), 2)
    out["agg_b_range"] = [round(float(Z[:, 4].min()), 2), round(float(Z[:, 4].max()), 2)]
    out["cement_max"] = float(df["cement"].max())
    d28 = (df["age"] == 28).to_numpy()
    near = d28 & (np.abs(Z[:, 0] - zq[0]) <= NEAR_WB_WINDOW)
    out["tested_near_wc"] = {
        "n": int(near.sum()),
        "mixes": int(df.loc[near, "mix"].nunique()),
        "min": round(float(y[near].min()), 1),
        "max": round(float(y[near].max()), 1),
    }
    return out


# ---------------------------------------------------------------------------------------------
# Decide


def margin_errors(
    df: pd.DataFrame, log_target: bool = False
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Held-out errors of the six-group least squares on every 28-day test of the 30 splits.

    Args:
        df: The concrete data.
        log_target: Fit ln(strength) and back-transform with exp (a median) instead.

    Returns:
        Prediction minus measured strength (MPa) and the prediction, pooled over the splits,
        and each split's held-out 28-day rows.
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    age = df["age"].to_numpy()
    over, preds, tested = [], [], []
    for tr, te in grouped_splits(df):
        t28 = te[age[te] == 28]
        if log_target:
            p28 = np.exp(predict(fit_least_squares(Z[tr], np.log(y[tr])), Z[t28]))
        else:
            p28 = predict(fit_least_squares(Z[tr], y[tr]), Z[t28])
        over.append(p28 - y[t28])
        preds.append(p28)
        tested.append(t28)
    return np.concatenate(over), np.concatenate(preds), tested


def cement_margin(df: pd.DataFrame, train: np.ndarray, out_dir: Path) -> dict[str, float | int]:
    """Cement content for the proposed mix's family to reach f'c, with and without a margin.

    The model is the first split's six-group least squares. The margin is the 90th percentile
    of over-prediction (prediction minus measured) over every held-out prediction of a 28-day
    test in the 30 splits (each test is held out in several splits): if this mix errs like
    those, aiming the prediction at f'c + margin leaves about 10% of such mixes below f'c. Also
    each split's own 90th percentile, the share over-predicted per distinct test, and the
    scatter (sample SD) of the errors in two prediction bands, for this model and a log target.

    Args:
        df: The concrete data.
        train: The first split's training rows.
        out_dir: The figure directory.

    Returns:
        The margin (MPa), the cement for f'c with and without it (kg/m^3), the matching w/c,
        and the counts and checks behind them.

    Raises:
        AssertionError: If the family's prediction does not rise with cement, or a target lies
            off the curve (``np.interp`` would clamp).
    """
    y = df["strength"].to_numpy()
    over, preds, tested = margin_errors(df)
    margin = float(np.quantile(over, MARGIN_QUANTILE))
    split_q, start = [], 0
    for t28 in tested:  # each split's own 90th percentile
        split_q.append(float(np.quantile(over[start : start + len(t28)], MARGIN_QUANTILE)))
        start += len(t28)
    rows = np.concatenate(tested)
    per_test = pd.Series(over).groupby(rows).mean()  # each distinct test's mean error
    lo_band, hi_band = SCATTER_BANDS_MPA
    log_over, log_preds, _ = margin_errors(df, log_target=True)
    w = fit_least_squares(mix_groups(df)[train], y[train])
    base = df.loc[[PROPOSED_ROW]]
    cem = np.linspace(*CEMENT_GRID)
    fam = pd.concat([base] * len(cem), ignore_index=True)
    fam["cement"] = cem
    pred = predict(w, mix_groups(fam))
    assert np.all(np.diff(pred) > 0), "prediction should rise with cement"
    # np.interp clamps outside the curve: both targets must lie on it.
    assert pred[0] < FC < FC + margin < pred[-1], (pred[0], FC + margin, pred[-1])
    c50 = float(np.interp(FC, pred, cem))
    c90 = float(np.interp(FC + margin, pred, cem))
    water = float(base["water"].iloc[0])
    margin_figure(df, cem, pred, margin=margin, cement=(c50, c90), water=water, out_dir=out_dir)
    return {
        "margin": rnd(margin, 1),
        "n_tests": int(len(over)),
        "n_distinct_tests": int(len(np.unique(rows))),
        "held_out_per_test_min": int(np.bincount(rows)[np.unique(rows)].min()),
        "held_out_per_test_max": int(np.bincount(rows).max()),
        "margin_per_split_median": rnd(np.median(split_q), 1),
        "margin_per_split_min": rnd(min(split_q), 1),
        "margin_per_split_max": rnd(max(split_q), 1),
        "cement_mean": int(round(c50)),
        "cement_margin": int(round(c90)),
        "extra_cement": int(round(c90) - round(c50)),
        "extra_pct": int(round(100 * (round(c90) - round(c50)) / round(c50))),
        "target": rnd(FC + margin, 1),
        "share_overpredicted": rnd(np.mean(over > 0), 2),
        "share_overpredicted_per_test": rnd(np.mean(per_test.to_numpy() > 0), 2),
        "curve_at_250": rnd(pred[0], 0),
        "curve_at_540": rnd(pred[-1], 0),
        "wc_mean": rnd(water / round(c50), 2),
        "wc_margin": rnd(water / round(c90), 2),
        "wc_max_cement": rnd(water / float(df["cement"].max()), 2),
        "extra_cement_max": int(round(float(df["cement"].max())) - round(c50)),
        "resid_sd_pred_below_30": rnd(np.std(over[preds < lo_band], ddof=1), 1),
        "resid_sd_pred_above_50": rnd(np.std(over[preds > hi_band], ddof=1), 1),
        "log_target_resid_sd_pred_below_30": rnd(np.std(log_over[log_preds < lo_band], ddof=1), 1),
        "log_target_resid_sd_pred_above_50": rnd(np.std(log_over[log_preds > hi_band], ddof=1), 1),
    }


def margin_figure(
    df: pd.DataFrame,
    cem: np.ndarray,
    pred: np.ndarray,
    *,
    margin: float,
    cement: tuple[float, float],
    water: float,
    out_dir: Path,
) -> None:
    """The family's prediction against cement, shifted down by the margin, over tested mixes.

    Args:
        df: The concrete data (for the tested 28-day mixes with water near 185 kg/m^3).
        cem: Cement contents, kg/m^3.
        pred: Predicted 28-day strength at each, MPa.
        margin: The margin, MPa.
        cement: Cement where the line and the line minus the margin reach f'c.
        water: The family's water, kg/m^3 (the proposed mix's).
        out_dir: The figure directory.
    """
    y = df["strength"].to_numpy()
    plain = (
        (df["age"] == 28)
        & (df[["slag", "fly_ash", "superplasticizer"]].sum(axis=1) == 0)
        & (np.abs(df["water"] - ATTEMPT_WATER) <= WATER_NEAR_TOL)
    ).to_numpy()
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.scatter(
        df["cement"][plain],
        y[plain],
        s=22,
        color=aml_style.MUTED,
        alpha=0.6,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        label=f"28-day tests, water ≈ {ATTEMPT_WATER:.0f}",
    )
    ax.plot(cem, pred, color=aml_style.PALETTE[0], label="prediction")
    ax.plot(
        cem,
        pred - margin,
        color=aml_style.PALETTE[1],
        ls="--",
        label=f"prediction − {margin:.1f} MPa",
    )
    ax.axhline(FC, color=aml_style.INK, lw=1, ls=":")
    for c, mk, col in zip(cement, ("o", "s"), aml_style.PALETTE[:2], strict=True):
        ax.plot(c, FC, marker=mk, ms=10, color=col, ls="none")
        ax.annotate(
            f"{c:.0f}",
            (c, FC),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color=col,
            path_effects=aml_style.HALO,
        )
    ax.set_xlabel(f"cement (kg/m³), water {water:.0f}, no admixtures")
    ax.set_ylabel("28-day strength (MPa)")
    ax.set_xlim(CEMENT_GRID[0], CEMENT_GRID[1] + 5)
    ax.set_ylim(10, 70)
    ax.legend(loc="upper left", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "cement_margin")


# ---------------------------------------------------------------------------------------------
# Lecture 2's numbers, the floor, and the diagram's macros


def lecture2_reference(df: pd.DataFrame) -> dict[str, float]:
    """Lecture 2 numbers quoted in today's warm-up and notes, recomputed here.

    The w/b of Lecture 2's proposed mix (row 913) and of its raw-units neighbor A (row 524);
    water's share of the raw squared distance; 5-NN on raw kg/m^3 and days (designs held out);
    the median strength gain per unit of ln(age) over mixes tested at four or more ages; and
    the range of the seven ingredients' total mass (kg per m^3 of concrete).

    Args:
        df: The concrete data.

    Returns:
        The values Lecture 2 quoted, as recomputed here.
    """
    y = df["strength"].to_numpy()
    X = raw_features(df)
    var = X.var(axis=0)
    knn_raw = knn_errors(X, y, grouped_splits(df), scale=False)
    ages = df.groupby("mix")["age"].nunique()
    gains = [
        np.polyfit(np.log(s["age"]), s["strength"], 1)[0]
        for m, s in df.groupby("mix")
        if ages[m] >= 4
    ]
    mass = df[CONCRETE_INGREDIENTS].sum(axis=1)
    wb = mix_groups(df)[:, 0]
    return {
        "warmup_neighbor_wb": round(float(wb[524]), 2),  # Lecture 2's mix A (row 524)
        "warmup_proposed_wb": round(float(wb[913]), 2),  # Lecture 2's proposed mix
        "water_share_pct": round(float(100 * var[RAW_INPUTS.index("water")] / var.sum()), 1),
        "knn_raw": round(float(knn_raw.mean()), 2),
        "median_gain_per_ln_day": round(float(np.median(gains)), 1),
        "mass_min": round(float(mass.min()), 0),
        "mass_max": round(float(mass.max()), 0),
    }


def repeat_floor(df: pd.DataFrame) -> dict[str, object]:
    """Median standard deviation of repeat tests (one mix, one age, different strengths).

    Args:
        df: The concrete data.

    Returns:
        The floor (MPa) and how many groups and mixes it rests on.
    """
    groups = repeat_groups(df)
    return {
        "floor": round(float(groups["std"].median()), 1),
        "groups": int(len(groups)),
        "mixes": int(groups.index.get_level_values("mix").nunique()),
    }


def tex_macros(
    att: dict[str, object], tr: dict[str, object], ver: dict[str, object]
) -> dict[str, str]:
    """The trace of the proposed mix as LaTeX macros for the TikZ diagram.

    Args:
        att: ``attempt``'s result (P's proportions).
        tr: ``trace``'s result, with its raw ``_`` values.
        ver: ``verify_listing``'s result (training-set counts).

    Returns:
        Macro name to LaTeX text, every number formatted once from its raw value.
    """
    p, z, t = att["proposed"], tr["_z"], tr["_terms"]
    assert isinstance(p, dict) and isinstance(z, dict) and isinstance(t, dict)
    pred, meas = float(tr["_pred"]), float(tr["_y"])
    return {
        "XCement": tex_num(p["cement"], "g"),
        "XWater": tex_num(p["water"], "g"),
        "XCoarse": tex_num(p["coarse"], "g"),
        "XFine": tex_num(p["fine"], "g"),
        "ZWb": tex_num(z["w/b"], ".2f"),
        "ZAgg": tex_num(z["aggregate / b"], ".2f"),
        "ZLnAge": tex_num(z["ln age"], ".2f"),
        "WZero": tex_num(tr["_intercept"], ".1f"),
        "TWb": tex_num(t["w/b"], "+.1f"),
        "TAgg": tex_num(t["aggregate / b"], "+.1f"),
        "TLnAge": tex_num(t["ln age"], "+.1f"),
        "Pred": tex_num(pred, ".1f"),
        "Meas": tex_num(meas, ".1f"),
        "Err": tex_num(pred - meas, "+.1f"),
        "FC": tex_num(FC, "g"),
        "NTrain": str(ver["n_train"]),
        "NTrainDesigns": str(ver["n_train_designs"]),
        "NWeights": str(len(GROUP_NAMES) + 1),
    }


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

BOWL_N: int = 61  # grid points per side of the loss surfaces
TROUGH_LAMBDA_PER_TEST: float = 1.0  # ridge penalty of the trough picture, per 28-day test
WATER_BAND: tuple[float, float] = (175.0, 195.0)  # kg/m^3: tests like the slab mix's water


def title_art_figure(att: dict[str, object], out_dir: Path) -> None:
    """The title art: a sun of nested ellipses (the loss bowl's contours) over stepped forms.

    Below, the four mixes glow on a golden line, their least-squares fit, with residual threads.

    Args:
        att: ``attempt``'s result, with the raw ``_x``, ``_lnf``, and ``_w``.
        out_dir: The figure directory.
    """
    x, lnf, w = (np.asarray(att[k]) for k in ("_x", "_lnf", "_w"))
    c = title_art.Canvas(seed=5)
    c.ground(("#1c0a0c", "#21101a", "#0f0f1e", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.92, 0.80, 4), soft=4), "#5d2015", 0.85)
    centre = np.array([0.56, 1.12])
    t = np.linspace(0, 2 * np.pi, 200)
    for k, r in enumerate((0.20, 0.15, 0.10, 0.05)):  # the bowl's contours, tilted like X^T X
        ring = [tuple(centre + r * np.array([np.cos(a) * 1.0 + np.sin(a) * 0.55,
                                             np.sin(a) * 0.45])) for a in t]  # fmt: skip
        c.glow(c.lines([ring], 3, soft=1.0), "#eaa548", 0.25 + 0.12 * k, spread=10)
    c.glow(c.disc(tuple(centre), 0.012, soft=2), "#ffe2a0", 1.0, spread=14)
    xs = 0.14 + 0.72 * (x - x.min()) / np.ptp(x)
    fit = w[0] + w[1] * x

    def to_y(v: float) -> float:
        """Height on the canvas of a value of ln f."""
        return float(0.22 + 0.40 * (v - lnf.min()) / np.ptp(lnf))

    line = [(0.08, to_y(w[0] + w[1] * (x.min() - 0.08 * np.ptp(x) / 0.72))),
            (0.92, to_y(w[0] + w[1] * (x.max() + 0.06 * np.ptp(x) / 0.72)))]  # fmt: skip
    c.glow(c.lines([line], 4, soft=1.2), "#f3b45c", 0.8, spread=12)
    for xi, yi, fi in zip(xs, lnf, fit, strict=True):
        c.glow(c.lines([[(xi, to_y(yi)), (xi, to_y(fi))]], 2, soft=0.8), "#c2361f", 0.9, spread=4)
        c.glow(c.disc((xi, to_y(yi)), 0.011, soft=1.5), "#ffd98a", 0.9, spread=10)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a broad stepped Andean form in rust; above it a sun "
        "made of nested golden ellipses, the contours of a loss bowl, around a bright minimum; "
        "across the form a golden line passes near four glowing points, joined to it by short "
        "crimson threads, the lecture's least-squares fit and its residuals.",
    )


def cement_question_figure(
    df: pd.DataFrame, att: dict[str, object], out_dir: Path
) -> dict[str, object]:
    """28-day tests with water near 185 kg/m^3 and no slag or fly ash: strength against cement.

    Args:
        df: The concrete data.
        att: ``attempt``'s result (mixes A-D are ringed).
        out_dir: The figure directory.

    Returns:
        The number of tests drawn and their cement (kg/m^3) and strength (MPa) ranges.
    """
    band = (
        (df["age"] == 28)
        & df["water"].between(*WATER_BAND)
        & (df["slag"] == 0)
        & (df["fly_ash"] == 0)
    )
    sub = df[band]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.scatter(sub["cement"], sub["strength"], s=26, color=aml_style.PALETTE[0], alpha=0.7, lw=0)
    mixes = att["mixes"]
    assert isinstance(mixes, list)
    for m in mixes:
        ax.annotate(m["letter"], (m["cement"], m["mpa"]), xytext=(7, -2),
                    textcoords="offset points", fontsize=13, fontweight="bold",
                    path_effects=aml_style.HALO)  # fmt: skip
        ax.plot(m["cement"], m["mpa"], "o", ms=9, mfc="none", mec=aml_style.INK, mew=1.6)
    ax.axhline(FC, color=aml_style.INK, lw=1.4, ls="--")
    ax.text(ax.get_xlim()[0] + 5, FC + 1, f"{FC:.0f} MPa", fontsize=12)
    ax.set(xlabel="cement (kg/m³)", ylabel="28-day strength (MPa)")
    ax.set_title(f"water {WATER_BAND[0]:.0f}–{WATER_BAND[1]:.0f} kg/m³, no slag or fly ash",
                 loc="left", fontsize=13)  # fmt: skip
    aml_style.save(fig, out_dir, "cement_question")
    return {
        "tests": int(len(sub)),
        "cement_min": rnd(sub["cement"].min(), 0),
        "cement_max": rnd(sub["cement"].max(), 0),
        "strength_min": rnd(sub["strength"].min(), 1),
        "strength_max": rnd(sub["strength"].max(), 1),
    }


def _loss_grid(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, half: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squared error over a grid of two weights centred on ``w``, each +-``half``.

    Args:
        X: Design matrix with two columns, shape (N, 2).
        y: Targets, shape (N,).
        w: The grid's centre (two weights).
        half: Half-width of the grid in each weight.

    Returns:
        The two weight grids and the summed squared error on them, each (BOWL_N, BOWL_N).
    """
    a = np.linspace(w[0] - half[0], w[0] + half[0], BOWL_N)
    b = np.linspace(w[1] - half[1], w[1] + half[1], BOWL_N)
    A, B = np.meshgrid(a, b)
    W = np.stack([A.ravel(), B.ravel()])  # (2, n)
    L = ((X @ W - y[:, None]) ** 2).sum(axis=0).reshape(A.shape)
    return A, B, L


def bowl_figure(att: dict[str, object], out_dir: Path) -> dict[str, object]:
    """The four mixes' squared error over (intercept, slope), raw and with w/c centred.

    Raw columns (1, w/c) are nearly parallel: the bowl is a long valley, tilted. Centred, the
    columns are orthogonal: the same valley lies along the axes, so each weight is found alone.

    Args:
        att: ``attempt``'s result, with the raw ``_x``, ``_lnf``, and ``_w``.
        out_dir: The figure directory.

    Returns:
        The ratio of the bowl's curvatures (condition number of X^T X), raw and centred.
    """
    x, y, w = (np.asarray(att[k]) for k in ("_x", "_lnf", "_w"))
    xbar = float(x.mean())
    cases = (
        ("raw: (1, w/c)", np.column_stack([np.ones(4), x]), w),
        (f"centred: (1, w/c − {xbar:.3f})", np.column_stack([np.ones(4), x - xbar]),
         np.array([w[0] + w[1] * xbar, w[1]])),
    )  # fmt: skip
    half = np.array([1.2, 2.0])
    fig = plt.figure(figsize=aml_style.FULL)
    out = {}
    for k, (title, X, wk) in enumerate(cases):
        A, B, L = _loss_grid(X, y, wk, half)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        zmax = 3.0
        aml_style.surface3d(ax, A, B, np.minimum(L, zmax), cmap="cividis", vmax=zmax)
        ax.contour(A, B, L, levels=[0.05, 0.2, 0.6, 1.5], zdir="z", offset=0,
                   colors=aml_style.MUTED, linewidths=1.0, zorder=0)  # fmt: skip
        ax.scatter([wk[0]], [wk[1]], [float(((X @ wk - y) ** 2).sum())], s=60, color="white",
                   edgecolor=aml_style.INK, zorder=5)  # fmt: skip
        ax.set(zlim=(0, zmax), zticks=[0, 1, 2, 3])
        ax.set_xlabel("intercept", labelpad=6, fontsize=12)  # clear of the tick numbers
        ax.set_ylabel("slope b", labelpad=6, fontsize=12)
        ax.tick_params(labelsize=9)
        ax.set_title(title, loc="left", fontsize=13)
        aml_style.style3d(ax, elev=38, azim=-60)
        ev = np.linalg.eigvalsh(X.T @ X)
        out[("raw", "centred")[k]] = {"condition": round(float(ev.max() / ev.min()), 0)}
    fig.text(0.5, -0.05, "squared error in ln f, four mixes", ha="center", fontsize=12,
             color=aml_style.MUTED)  # fmt: skip
    fig.subplots_adjust(left=0, right=1, bottom=0.05, top=0.92, wspace=0.05)
    aml_style.save(fig, out_dir, "bowl")
    return out


def projection_figure(out_dir: Path) -> None:
    """Least squares as a projection, drawn in 3D: y, its shadow X w on the columns' plane.

    Args:
        out_dir: The figure directory.
    """
    c1, c2 = np.array([1.0, 0.15, 0.0]), np.array([0.25, 1.0, 0.0])  # a plane: z = 0
    y = np.array([0.95, 0.85, 0.75])
    yhat = np.array([y[0], y[1], 0.0])
    fig = plt.figure(figsize=aml_style.SIDE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    s, t = np.meshgrid(np.linspace(-0.15, 1.25, 2), np.linspace(-0.15, 1.15, 2))
    P = s[..., None] * c1 + t[..., None] * c2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=aml_style.PALETTE[0], alpha=0.15, lw=0)
    stroke = aml_style.HALO
    for v, color, label, off in (
        (y, aml_style.INK, r"$\mathbf{y}$", (0.02, 0.02, 0.04)),
        (yhat, aml_style.PALETTE[0], r"$X\hat{\mathbf{w}}$", (0.08, 0.0, -0.12)),
    ):  # below the tip, off the right angle
        ax.quiver(0, 0, 0, *v, color=color, lw=2.6, arrow_length_ratio=0.08)
        ax.text(*(v + off), label, fontsize=17, color=color, path_effects=stroke)
    ax.plot(*np.column_stack([yhat, y]), color=aml_style.PALETTE[1], lw=2.6, ls="--")
    ax.text2D(0.79, 0.50, r"$\mathbf{y}-X\hat{\mathbf{w}}$", transform=ax.transAxes,
              fontsize=15, color=aml_style.PALETTE[1], path_effects=stroke)  # fmt: skip
    sq = 0.07  # right-angle mark at the foot of the residual
    corner = yhat + np.array([[-sq, 0, 0], [-sq, 0, sq], [0, 0, sq]])
    ax.plot(*corner.T, color=aml_style.PALETTE[1], lw=1.4)
    for v, name in ((c1, "column 1"), (c2, "column 2")):
        ax.quiver(0, 0, 0, *(0.55 * v), color=aml_style.MUTED, lw=1.6, arrow_length_ratio=0.15)
        ax.text(*(0.6 * v), name, fontsize=11, color=aml_style.MUTED)
    ax.text2D(0.52, 0.12, "plane of the columns of X", transform=ax.transAxes, fontsize=12,
              color=aml_style.PALETTE[0])  # fmt: skip
    ax.set(xlim=(0, 1.2), ylim=(0, 1.2), zlim=(0, 1.0), xticks=[], yticks=[], zticks=[])
    aml_style.style3d(ax, elev=20, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    aml_style.save(fig, out_dir, "projection")


def trough_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """A column entered twice: the loss is a trough with a flat floor; ridge makes it a bowl.

    The 28-day tests, ln f against standardized w/b entered as two identical columns
    (intercept removed by centring). Every (w1, w2) with w1 + w2 = b fits equally well.

    Args:
        df: The concrete data.
        out_dir: The figure directory.

    Returns:
        The number of tests, the one-column slope b, each weight of the minimum-norm answer
        (b / 2, from the raw slope), the ridge weights, and ridge's shrinkage factor.
    """
    d28 = df[df["age"] == 28]
    wb = (d28["water"] / (d28["cement"] + d28["slag"] + d28["fly_ash"])).to_numpy()
    z = (wb - wb.mean()) / wb.std()
    y = np.log(d28["strength"].to_numpy())
    y = y - y.mean()
    n = len(y)
    b = float(z @ y / (z @ z))  # one column: the least-squares slope
    X = np.column_stack([z, z])
    lam = TROUGH_LAMBDA_PER_TEST * n
    w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)
    half = np.array([0.45, 0.45])
    centre = np.array([b / 2, b / 2])
    fig = plt.figure(figsize=aml_style.FULL)
    out = {
        "n": n,
        "slope": rnd(b, 3),
        "min_norm": rnd(b / 2, 3),  # each weight of the minimum-norm answer, from the raw slope
        "ridge": [rnd(v, 3) for v in w_ridge],
        "shrink": rnd(2 * n / (2 * n + lam), 3),  # ridge's factor along the trough's floor
    }
    panels = (("w/b entered twice: a trough", 0.0), (rf"ridge, $\lambda$ = {lam:.0f}: a bowl", lam))
    for k, (title, pen) in enumerate(panels):
        A, B, L = _loss_grid(X, y, centre, half)
        L = (L + pen * (A**2 + B**2)) / n
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        aml_style.surface3d(ax, A, B, L, cmap="cividis")
        if pen == 0:
            t = np.linspace(-half[0], half[0], 50)
            ax.plot(centre[0] + t, centre[1] - t, np.full_like(t, L.min()),
                    color=aml_style.PALETTE[1], lw=2.6, zorder=5)  # fmt: skip
            ax.scatter([b / 2], [b / 2], [L.min()], s=60, color="white", edgecolor=aml_style.INK,
                       zorder=6)  # fmt: skip
        else:
            Lr = (float(((X @ w_ridge - y) ** 2).sum()) + pen * float(w_ridge @ w_ridge)) / n
            ax.scatter(*w_ridge, [Lr], s=60, color="white", edgecolor=aml_style.INK, zorder=6)
        ax.set_xlabel("weight 1", labelpad=8, fontsize=12)  # clear of the tick numbers
        ax.set_ylabel("weight 2", labelpad=8, fontsize=12)
        ax.tick_params(labelsize=9)
        ax.set_title(title, loc="left", fontsize=13)
        aml_style.style3d(ax, elev=30, azim=-35)
        ax.set_box_aspect(None, zoom=0.85)  # smaller box: the labels stay inside the crop
    fig.subplots_adjust(left=0, right=1, bottom=0.03, top=0.92, wspace=0.05)
    aml_style.save(fig, out_dir, "trough")
    return out


def paths_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Ridge and lasso paths over the loss contours: standardized w/b and binder, all tests.

    The two inputs are strongly correlated (``corr`` in the result), so the contours are tilted
    ellipses. Least squares sits at their centre; as the penalty grows, ridge shrinks both
    weights together, while lasso drops binder to exactly zero and keeps w/b.

    Args:
        df: The concrete data.
        out_dir: The figure directory.

    Returns:
        The correlation of the two inputs, the least-squares weights, the input lasso admits
        first, and the number of tests.
    """
    binder = (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()
    wb = df["water"].to_numpy() / binder
    Z = StandardScaler().fit_transform(np.column_stack([wb, binder]))
    y = df["strength"].to_numpy() - df["strength"].mean()
    n = len(y)
    w_ls = np.linalg.lstsq(Z, y, rcond=None)[0]
    lams = np.logspace(-2, 5, 200)
    ridge = np.array([np.linalg.solve(Z.T @ Z + lam * np.eye(2), Z.T @ y) for lam in lams])
    _, coefs, _ = lasso_path(Z, y, n_alphas=200)
    lasso = coefs.T  # (alphas, 2), from zero to least squares
    g = np.linspace(-10, 2, 200), np.linspace(-3, 9, 200)
    A, B = np.meshgrid(*g)
    L = ((Z @ np.stack([A.ravel(), B.ravel()]) - y[:, None]) ** 2).mean(axis=0).reshape(A.shape)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    levels = np.quantile(L, [0.002, 0.01, 0.04, 0.12, 0.3])
    ax.contour(A, B, L, levels=np.sort(levels), colors=aml_style.MUTED, linewidths=1.0)
    ax.plot(ridge[:, 0], ridge[:, 1], color=aml_style.PALETTE[0], lw=2.6, ls="-", label="ridge")
    ax.plot(lasso[:, 0], lasso[:, 1], color=aml_style.PALETTE[1], lw=2.6, ls="--", label="lasso")
    ax.plot(*w_ls, "o", ms=9, color=aml_style.INK)
    ax.annotate("least squares", w_ls, xytext=(10, 10), textcoords="offset points", fontsize=12,
                ha="left", path_effects=aml_style.HALO)  # fmt: skip
    ax.plot(0, 0, "s", ms=7, color=aml_style.INK)
    ax.annotate(r"$\lambda$ large: 0", (0, 0), xytext=(8, -14), textcoords="offset points",
                fontsize=12)  # fmt: skip
    ax.axhline(0, color=aml_style.MUTED, lw=0.6)
    ax.axvline(0, color=aml_style.MUTED, lw=0.6)
    ax.set(xlabel="weight of w/b (MPa per sd)", ylabel="weight of binder (MPa per sd)",
           aspect="equal")  # fmt: skip
    ax.legend(loc="upper right", fontsize=12, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)  # fmt: skip
    aml_style.save(fig, out_dir, "paths")
    first = int(np.argmax(np.abs(lasso).sum(axis=1) > 0))
    return {
        "corr": round(float(np.corrcoef(Z.T)[0, 1]), 2),
        "w_ls": [round(float(v), 2) for v in w_ls],
        "lasso_first": ["w/b", "binder"][int(np.argmax(np.abs(lasso[first])))],
        "n": n,
    }


def main() -> None:
    """Generate every figure and number for CEE Lecture 5."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = load_concrete()
    tr0, te0 = grouped_splits(df)[0]
    att = attempt(df, tr0, te0, out_dir)
    ver = verify_listing(df, tr0, te0)
    tr = trace(df, tr0)
    a_const, b_const = abrams_constants(df)
    numbers: dict[str, object] = {
        "data": {
            "tests": len(df),
            "n_28day": int((df["age"] == 28).sum()),
            "designs": int(df["design"].nunique()),
        },
        "abrams_28day": {"A": round(a_const, 1), "B": round(b_const, 1)},
        "lecture2": lecture2_reference(df),
        "attempt": att,
        "verify": ver,
        "trace": tr,
        "features": features_rmse(df, out_dir, repeat_floor(df)),
        "lasso": lasso_order(df, grouped_splits(df)),
        "collinear": collinear(df, out_dir),
        "extrapolation": extrapolation(df, out_dir),
        "ridge_units": ridge_units(df),
        "audit": audit_key(df),
        "decision": cement_margin(df, tr0, out_dir),
    }
    # the visuals that carry the slides
    title_art_figure(att, out_dir)
    numbers["cement_question"] = cement_question_figure(df, att, out_dir)
    numbers["bowl"] = bowl_figure(att, out_dir)
    projection_figure(out_dir)
    numbers["trough"] = trough_figure(df, out_dir)
    numbers["paths"] = paths_figure(df, out_dir)
    aml_style.thumbnails(out_dir, ("bowl", "trough", "features_rmse", "cement_extrapolation"))
    write_numbers(numbers, out_dir)
    print(write_tex_macros(tex_macros(att, tr, ver), out_dir, "cee_l05_least_squares.py"))


if __name__ == "__main__":
    main()
