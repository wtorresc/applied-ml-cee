"""Figures and numbers for CEE Lecture 10: calibration and quantile regression (concrete).

Run: ``pixi run -e aml-env python figures/cee_l10_calibration_quantiles.py
--out build/fig/cee_l10_calibration_quantiles``

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one mix, for the TikZ diagram). Data: UCI Concrete Compressive Strength (Yeh,
1998), with the Lecture 2 encoder and the Lecture 5 splits (30 splits holding out whole designs).

The question: what strength can a mix be counted on to reach? Design codes count on a low
fractile, not the mean (the characteristic strength is the 5% fractile). Three ways to predict
the 5% quantile of a held-out test: the mean model minus 1.645 standard deviations (normal,
constant spread), linear quantile regression (pinball loss), and the 5% quantile of all training
strengths (no model). Each is judged by its coverage and by the pinball loss, both pooled over
the held-out predictions of the 30 splits. Each split's two models are fitted once
(:func:`fit_splits`); the design bootstrap resamples the held-out designs, not the fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor

import aml_style
import title_art
from common import parse_out_dir, rnd, seed_all, tex_num, write_numbers, write_tex_macros
from concrete import (
    FC,
    GROUP_NAMES,
    PROPOSED_ROW,
    grouped_splits,
    load_concrete,
    mix_groups,
    repeat_groups,
)

SEED: int = 0
BOOT: int = 2000  # design-clustered bootstrap resamples for the 95% intervals
TAU: float = 0.05  # the characteristic strength: 5% fractile
TAUS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BANDS: tuple[tuple[float, float], ...] = ((0.0, 30.0), (30.0, 50.0), (50.0, 200.0))
BAND_NAMES: tuple[str, ...] = ("below 30", "30 to 50", "above 50")
METHODS: tuple[str, ...] = ("no model", "mean − 1.645σ", "quantile regression")
VERIFY_ATOL: float = 1e-11  # the LP against scikit-learn, largest weight difference
CEMENT_GRID: tuple[float, float, int] = (250.0, 540.0, 2901)  # kg/m3, start, stop, points
SHEET_WB: tuple[float, float] = (0.30, 0.70)  # the margin sheets' w/b range
SHEET_AGE_DAYS: tuple[int, int] = (7, 365)  # and their age range
LABEL: Final[dict[str, str]] = {
    "no model": "no model",
    "mean − 1.645σ": "mean − 1.645$\\sigma$",
    "quantile regression": "quantile regression",
    "mean ± zσ": "mean ± z$\\sigma$",
}


# ---------------------------------------------------------------------------------------------
# The two models (the code slide's listing is fit_quantile)


def design_matrix(Z: np.ndarray) -> np.ndarray:
    """``[1 Z]``."""
    return np.column_stack([np.ones(len(Z)), Z])


def fit_mean(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Lecture 5's mean model: least squares and the SD of its training residuals.

    Args:
        A: Design matrix ``[1 Z]``, shape ``(n, d + 1)``.
        y: Targets, shape ``(n,)``.

    Returns:
        The weights, shape ``(d + 1,)``, and the residual SD with ``d + 1`` degrees of freedom
        removed.
    """
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w, float(np.std(y - A @ w, ddof=A.shape[1]))


def fit_quantile(Z: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Linear quantile regression as a linear program (pinball loss, no penalty).

    Minimize sum(tau * u_plus + (1 - tau) * u_minus) subject to A w + u_plus - u_minus = y,
    u_plus, u_minus >= 0, where A = [1 Z].

    Args:
        Z: Features, shape ``(n, d)``.
        y: Targets, shape ``(n,)``.
        tau: Quantile level in (0, 1).

    Returns:
        Weights ``[w0, w]``, shape ``(d + 1,)``.

    Raises:
        RuntimeError: If the solver fails.
    """
    A = design_matrix(Z)
    n, d = A.shape
    c = np.concatenate([np.zeros(d), np.full(n, tau), np.full(n, 1 - tau)])
    A_eq = np.hstack([A, np.eye(n), -np.eye(n)])  # A w + u+ - u- = y
    bounds = [(None, None)] * d + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed: {res.message}")
    return res.x[:d]


def pinball_each(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Pinball (quantile) loss of each predicted quantile ``q``."""
    r = y - q
    return np.maximum(tau * r, (tau - 1) * r)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss of predicted quantiles ``q``."""
    return float(np.mean(pinball_each(y, q, tau)))


# ---------------------------------------------------------------------------------------------
# Experiments


@dataclass(frozen=True)
class SplitFit:
    """One split's rows and its two fitted models.

    Attributes:
        tr: Training rows.
        te: Held-out rows.
        w: The mean model's weights ``[w0, w]``.
        sd: The mean model's training residual SD (MPa).
        wq: The 5% quantile regression's weights ``[w0, w]``.
    """

    tr: np.ndarray
    te: np.ndarray
    w: np.ndarray
    sd: float
    wq: np.ndarray


def fit_splits(df: pd.DataFrame) -> list[SplitFit]:
    """Fit the mean model and the 5% quantile regression once on each of the 30 splits.

    Args:
        df: The concrete tests.

    Returns:
        One :class:`SplitFit` per split, in split order.
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    A = design_matrix(Z)
    fits = []
    for tr, te in grouped_splits(df):
        w, sd = fit_mean(A[tr], y[tr])
        fits.append(SplitFit(tr, te, w, sd, fit_quantile(Z[tr], y[tr], TAU)))
    return fits


def held_out(df: pd.DataFrame, fits: list[SplitFit]) -> pd.DataFrame:
    """Predicted 5% quantiles of every held-out test of the 30 splits, three ways.

    Args:
        df: The concrete tests.
        fits: :func:`fit_splits`'s models.

    Returns:
        One row per held-out prediction: split, row, design, measured strength, the mean
        prediction and the training SD, and the three predicted 5% values.
    """
    y = df["strength"].to_numpy()
    A = design_matrix(mix_groups(df))
    rows = []
    for k, f in enumerate(fits):
        tr, te, sd = f.tr, f.te, f.sd
        mean = A[te] @ f.w
        q_qr = A[te] @ f.wq
        q_none = np.full(len(te), np.quantile(y[tr], TAU))
        rows.append(
            pd.DataFrame(
                {
                    "split": k,
                    "row": te,
                    "design": df["design"].to_numpy()[te],
                    "y": y[te],
                    "mean": mean,
                    "sd": sd,
                    "no model": q_none,
                    "mean − 1.645σ": mean + norm.ppf(TAU) * sd,
                    "quantile regression": q_qr,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def in_band(h: pd.DataFrame, band: tuple[float, float]) -> pd.Series:
    """Held-out predictions whose predicted mean lies in ``band`` (MPa)."""
    return (h["mean"] >= band[0]) & (h["mean"] < band[1])


def attempt(h: pd.DataFrame) -> dict[str, object]:
    """Held-out errors of the mean model, by predicted strength.

    Args:
        h: :func:`held_out`'s predictions.

    Returns:
        Per band and overall: the number of predictions, the error SD and the error's 5% point
        (MPa).
    """
    err = h["y"] - h["mean"]
    out = {}
    for band, name in zip(BANDS, BAND_NAMES, strict=True):
        m = in_band(h, band)
        out[name] = {
            "tests": int(m.sum()),
            "sd": round(float(err[m].std()), 1),
            "q05": round(float(np.quantile(err[m], TAU)), 1),
        }
    out["all"] = {
        "tests": len(h),
        "sd": round(float(err.std()), 1),
        "q05": round(float(np.quantile(err, TAU)), 1),
    }
    return out


def coverage(h: pd.DataFrame, out_dir: Path) -> tuple[dict[str, object], dict[str, float]]:
    """Share of held-out predictions above the measured strength, overall and by band.

    Shares and pinball losses are pooled over all held-out predictions of the 30 splits.

    Args:
        h: :func:`held_out`'s predictions.
        out_dir: Where the figure goes.

    Returns:
        The numbers, and each method's unrounded mean pinball loss.
    """
    out: dict[str, object] = {}
    by_band: dict[str, dict[str, float]] = {}
    losses: dict[str, float] = {}
    for mth in METHODS:
        below = h["y"] < h[mth]
        by_band[mth] = {
            name: rnd(100 * float(below[in_band(h, band)].mean()), 1)
            for band, name in zip(BANDS, BAND_NAMES, strict=True)
        }
        losses[mth] = pinball(h["y"].to_numpy(), h[mth].to_numpy(), TAU)
        out[mth] = {
            "all": rnd(100 * float(below.mean()), 1),
            "by_band": by_band[mth],
            "pinball": rnd(losses[mth], 2),
        }
    # The decisive figure: coverage by predicted strength, three methods.
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    x = np.arange(len(BAND_NAMES))
    for i, mth in enumerate(METHODS):
        vals = [by_band[mth][b] for b in BAND_NAMES]
        ax.plot(
            x + (i - 1) * 0.08,
            vals,
            aml_style.MARKERS[i] + aml_style.LINESTYLES[i],
            color=aml_style.PALETTE[i],
            markersize=10,
            label=LABEL[mth],
        )
    ax.axhline(100 * TAU, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.text(2.25, 100 * TAU + 0.4, "5%", color=aml_style.MUTED, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} MPa" for b in BAND_NAMES])
    ax.set_xlabel("predicted mean strength")
    ax.set_ylabel("held-out tests below\nthe 5% quantile (%)")
    ax.set_ylim(-0.8, 17)  # room for the markers at 0
    ax.legend(loc="upper center", fontsize=11)  # the first band's points sit in the upper left
    aml_style.save(fig, out_dir, "coverage_bands")
    return out, losses


def design_bootstrap(h: pd.DataFrame) -> dict[str, object]:
    """95% intervals that resample whole designs: band coverages and the paired pinball gap.

    Every held-out prediction of a design (its tests, in every split that held it out) moves
    together, so the intervals allow for the repeats across splits and for the tests of one
    design being alike. ``BOOT`` resamples, seed ``SEED``, percentile intervals. Designs are
    numbered by first appearance, so the notebook's copy draws the same resamples. The
    intervals cover which designs are tested; the models are not refitted.

    Args:
        h: :func:`held_out`'s predictions.

    Returns:
        The intervals (coverages in percent), the resample count, the number of designs, and
        the paired pinball gaps: normal minus quantile regression, and the constant minus each
        model, with each model's loss as a fraction of the constant's.

    Raises:
        ValueError: If a resample has no prediction in a band.
    """
    g = pd.factorize(h["design"])[0]
    G = int(g.max()) + 1
    W = np.random.default_rng(SEED).multinomial(G, np.full(G, 1 / G), size=BOOT).astype(float)

    def boot(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The pooled mean of ``values`` over ``mask``, in every resample."""
        num = np.bincount(g[mask], weights=values[mask], minlength=G)
        n = W @ np.bincount(g[mask], minlength=G).astype(float)
        if not np.all(n > 0):
            raise ValueError("a resample with no prediction in the band")
        return (W @ num) / n

    def ci(v: np.ndarray, scale: float, nd: int) -> list[float]:
        """The 2.5% and 97.5% points of the resampled values."""
        return [rnd(scale * q, nd) for q in np.quantile(v, [0.025, 0.975])]

    y = h["y"].to_numpy()
    every = np.ones(len(h), dtype=bool)
    out: dict[str, object] = {"resamples": BOOT, "designs": G}
    for mth in METHODS:
        below = (y < h[mth].to_numpy()).astype(float)
        out[mth] = {
            "all_ci95": ci(boot(below, every), 100, 1),
            "by_band_ci95": {
                name: ci(boot(below, in_band(h, band).to_numpy()), 100, 1)
                for band, name in zip(BANDS, BAND_NAMES, strict=True)
            },
        }
    each = {mth: pinball_each(y, h[mth].to_numpy(), TAU) for mth in METHODS}
    gap = each["mean − 1.645σ"] - each["quantile regression"]
    out["pinball_gap"] = {"normal_minus_qr": rnd(gap.mean(), 3), "ci95": ci(boot(gap, every), 1, 3)}
    # Slide "No model is calibrated too": the constant against each model.
    for key, mth in (
        ("none_minus_normal", "mean − 1.645σ"),
        ("none_minus_qr", "quantile regression"),
    ):
        d = each["no model"] - each[mth]
        out["pinball_gap"][key] = {
            "gap": rnd(d.mean(), 2),
            "ci95": ci(boot(d, every), 1, 2),
            "ratio": rnd(each["no model"].mean() / each[mth].mean(), 2),
        }
    return out


def reliability(
    df: pd.DataFrame, fits: list[SplitFit], out_dir: Path
) -> tuple[dict[str, object], float]:
    """Nominal against observed share below, at seven quantile levels, held-out designs.

    The 5% fits come from ``fits``; the other levels are fitted here.

    Args:
        df: The concrete tests.
        fits: :func:`fit_splits`'s models.
        out_dir: Where the figure goes.

    Returns:
        The pooled shares below (%) by method and level, and the largest gap to the diagonal
        (percentage points).
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    A = design_matrix(Z)
    below: dict[str, dict[float, list[np.ndarray]]] = {
        "mean ± zσ": {t: [] for t in TAUS},
        "quantile regression": {t: [] for t in TAUS},
    }
    for f in fits:
        tr, te = f.tr, f.te
        for t in TAUS:
            wq = f.wq if t == TAU else fit_quantile(Z[tr], y[tr], t)
            below["mean ± zσ"][t].append(y[te] < A[te] @ f.w + norm.ppf(t) * f.sd)
            below["quantile regression"][t].append(y[te] < A[te] @ wq)
    share = {
        k: {t: 100 * float(np.concatenate(v).mean()) for t, v in d.items()}
        for k, d in below.items()
    }
    gap = max(abs(v - 100 * t) for d in share.values() for t, v in d.items())
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, (k, d) in enumerate(share.items()):
        ax.plot(
            [100 * t for t in TAUS],
            [d[t] for t in TAUS],
            aml_style.MARKERS[i + 1] + aml_style.LINESTYLES[i + 1],
            color=aml_style.PALETTE[i + 1],
            markersize=9,
            label=LABEL[k],
        )
    ax.plot([0, 100], [0, 100], ":", color=aml_style.MUTED, lw=1.5)
    ax.set_xlabel("nominal level $\\tau$ (%)")
    ax.set_ylabel("held-out tests below (%)")
    ax.legend(loc="upper left", fontsize=12)
    aml_style.save(fig, out_dir, "reliability")
    rounded = {k: {str(t): round(v, 1) for t, v in d.items()} for k, d in share.items()}
    return rounded, gap


def bracket_figure(losses: dict[str, float], floor: float, out_dir: Path) -> None:
    """Dot plot of the mean pinball loss on held-out designs, with the repeat-test floor.

    Args:
        losses: Each method's mean pinball loss (MPa).
        floor: The expected pinball loss of repeat-test noise (MPa).
        out_dir: Where the figure goes.
    """
    rows = [(m, losses[m]) for m in METHODS]
    rows.append(("repeat tests (rough)", floor))
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, (_, v) in enumerate(rows):
        ax.plot([0, v], [i, i], color="#D9D9D9", lw=1.5, zorder=1)
        ax.plot(
            [v],
            [i],
            aml_style.MARKERS[i % 4],
            color=aml_style.PALETTE[i % 4],
            markersize=11,
            zorder=2,
        )
        ax.annotate(f"{v:.2f}", (v, i), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=13)  # clear of the marker  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(r[0], r[0]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.65)
    ax.set_xlabel("pinball loss at $\\tau$ = 0.05 (MPa)")
    aml_style.save(fig, out_dir, "bracket")


def verify(df: pd.DataFrame, fit: SplitFit) -> dict[str, object]:
    """The LP against scikit-learn's QuantileRegressor on the first split; the minimizer check.

    Args:
        df: The concrete tests.
        fit: The first split's models.

    Returns:
        The training size, the largest weight difference, the training tests below, on, and at
        or below the line, and the constant minimizer against the order statistic.

    Raises:
        RuntimeError: If the two fits differ by more than ``VERIFY_ATOL``.
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    tr, w = fit.tr, fit.wq
    ref = QuantileRegressor(quantile=TAU, alpha=0, solver="highs").fit(Z[tr], y[tr])
    w_ref = np.concatenate([[ref.intercept_], ref.coef_])
    diff = float(np.max(np.abs(w - w_ref)))
    if diff >= VERIFY_ATOL:
        raise RuntimeError(f"the LP and scikit-learn differ by {diff:.1e}")
    r = y[tr] - design_matrix(Z[tr]) @ w
    below, on = int(np.sum(r < -1e-9)), int(np.sum(np.abs(r) <= 1e-9))
    # The constant minimizing the mean pinball loss is the empirical 5% quantile.
    grid = np.linspace(y.min(), y.max(), 20001)
    best = grid[int(np.argmin([pinball(y[tr], np.full(len(tr), g), TAU) for g in grid]))]
    k = int(np.ceil(TAU * len(tr)))
    return {
        "train_tests": len(tr),
        "max_abs_diff": float(f"{diff:.0e}"),
        "train_below": below,
        "train_on_line": on,
        "train_at_or_below": below + on,
        "constant_minimizer": round(float(best), 2),
        "order_statistic": round(float(np.sort(y[tr])[k - 1]), 2),
        "k": k,
    }


def crossing(df: pd.DataFrame) -> tuple[dict[str, object], dict[float, np.ndarray]]:
    """Separate fits at 5%, 10%, and 50% on all tests: how often does q05 exceed q10?

    Also where the crossings sit: the predicted median below 30 MPa, and leverage (the
    diagonal of the hat matrix of ``[1 Z]``) in the top tenth of all tests'.

    Args:
        df: The concrete tests.

    Returns:
        The numbers, and each level's predicted quantile for every test (for the figure).
    """
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    A = design_matrix(Z)
    q = {t: A @ fit_quantile(Z, y, t) for t in (0.05, 0.10, 0.50)}
    cross = q[0.05] > q[0.10]
    low = q[0.50] < BANDS[0][1]
    lev = np.einsum("ij,jk,ik->i", A, np.linalg.pinv(A.T @ A), A)
    high = lev > np.quantile(lev, 0.9)
    numbers = {
        "tests": len(y),
        "crossing": int(cross.sum()),
        "crossing_pct": rnd(100 * float(cross.mean()), 1),
        "max_gap": rnd((q[0.05] - q[0.10]).max(), 1),
        "low_median": int((cross & low).sum()),
        "low_median_all_pct": rnd(100 * float(low.mean()), 0),
        "high_leverage": int((cross & high).sum()),
    }
    return numbers, q


def cement_for_fc(cem: np.ndarray, curve: np.ndarray, name: str) -> float:
    """The cement at which a rising strength curve reaches f'c, interpolated on the grid.

    Args:
        cem: The cement grid (kg/m3), increasing.
        curve: Predicted strength (MPa) at each grid point.
        name: The curve's name, for the error message.

    Returns:
        The cement content (kg/m3) where the curve equals ``FC``.

    Raises:
        ValueError: If the curve does not rise with cement, or does not reach ``FC`` inside the
            grid (``np.interp`` would clamp to the grid's end silently).
    """
    if not np.all(np.diff(curve) > 0):
        raise ValueError(f"the {name} curve should rise with cement")
    if not curve[0] <= FC <= curve[-1]:
        raise ValueError(
            f"the {name} curve spans {curve[0]:.1f}-{curve[-1]:.1f} MPa over "
            f"{cem[0]:g}-{cem[-1]:g} kg/m3 and does not reach {FC} MPa inside the grid"
        )
    return float(np.interp(FC, curve, cem))


def band_name(value: float) -> str:
    """The name of the ``BANDS`` band that holds a predicted mean strength (MPa).

    Args:
        value: A predicted mean strength (MPa).

    Returns:
        The band's name from ``BAND_NAMES``.

    Raises:
        ValueError: If no band holds it.
    """
    for (lo, hi), name in zip(BANDS, BAND_NAMES, strict=True):
        if lo <= value < hi:
            return name
    raise ValueError(f"{value} MPa lies in no band")


def decision(df: pd.DataFrame, fits: list[SplitFit], out_dir: Path) -> dict[str, object]:
    """Cement for the proposed mix's family to reach f'c = 35 MPa: mean, normal 5%, QR 5%.

    The slide's numbers come from the first split, which holds out mix P. The same answers in
    all 30 splits show how much of a difference between the two 5% curves is one split's fit.

    Args:
        df: The concrete tests.
        fits: :func:`fit_splits`'s models.
        out_dir: Where the figure goes.

    Returns:
        The first split's cement, w/c, margins and the cement where the two 5% curves cross;
        the predicted mean at the quantile line's cement and its band; and, over the 30
        splits, each curve's median and range of cement and the quantile line's extra cement
        over the normal margin.

    Raises:
        ValueError: If the two 5% curves of the first split never cross on the grid.
    """
    base = df.loc[[PROPOSED_ROW]]
    cem = np.linspace(*CEMENT_GRID)
    fam = pd.concat([base] * len(cem), ignore_index=True)
    fam["cement"] = cem
    Af = design_matrix(mix_groups(fam))

    def curves_of(f: SplitFit) -> dict[str, np.ndarray]:
        """The mean and the two 5% curves of one split's models over the cement grid."""
        return {"mean": Af @ f.w, "normal": Af @ f.w + norm.ppf(TAU) * f.sd, "qr": Af @ f.wq}

    every = [{k: cement_for_fc(cem, v, k) for k, v in curves_of(f).items()} for f in fits]
    curves, c = curves_of(fits[0]), every[0]
    cement_figure(cem, curves, c, out_dir)
    water = float(base["water"].iloc[0])
    gap = curves["qr"] - curves["normal"]
    changes = np.flatnonzero(np.diff(np.sign(gap)) != 0)
    if len(changes) == 0:
        raise ValueError("the two 5% curves do not cross between the grid's ends")
    cross = float(cem[changes[0]])
    mean_at_qr = float(np.interp(c["qr"], cem, curves["mean"]))
    table = pd.DataFrame(every)
    extra = table["qr"] - table["normal"]
    return {
        "curves_cross_at": round(cross),
        "margin_normal": round(-norm.ppf(TAU) * fits[0].sd, 1),
        "cement": {k: round(v) for k, v in c.items()},
        "wc": {k: round(water / round(v), 2) for k, v in c.items()},
        "qr_margin_at_c": round(float(np.interp(c["qr"], cem, curves["mean"] - curves["qr"])), 1),
        "mean_at_qr_cement": rnd(mean_at_qr, 1),
        "band_at_qr_cement": band_name(mean_at_qr),
        "cement_max_tested": float(df["cement"].max()),
        "splits": {
            "n": len(fits),
            "cement_median": {k: rnd(table[k].median(), 0) for k in table},
            "cement_range": {k: [rnd(table[k].min(), 0), rnd(table[k].max(), 0)] for k in table},
            "qr_minus_normal_median": rnd(extra.median(), 0),
            "qr_minus_normal_range": [rnd(extra.min(), 0), rnd(extra.max(), 0)],
            "qr_more": int((extra > 0).sum()),
        },
    }


def cement_figure(
    cem: np.ndarray, curves: dict[str, np.ndarray], c: dict[str, float], out_dir: Path
) -> None:
    """Strength against cement for mix P's family: the mean and the two 5% curves.

    Args:
        cem: The cement grid (kg/m3).
        curves: The mean, normal and quantile curves (MPa) over the grid.
        c: Each curve's cement for ``FC``.
        out_dir: Where the figure goes.
    """
    names = {"mean": "mean", "normal": "mean − 1.645$\\sigma$", "qr": "5% quantile regression"}
    offsets = {"mean": (0, -20), "normal": (-14, 12), "qr": (14, -20)}
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, (key, curve) in enumerate(curves.items()):
        ax.plot(cem, curve, aml_style.LINESTYLES[i], color=aml_style.PALETTE[i], label=names[key])
        ax.plot(c[key], FC, aml_style.MARKERS[i], color=aml_style.PALETTE[i], ms=10)
        ax.annotate(
            f"{c[key]:.0f}",
            (c[key], FC),
            xytext=offsets[key],
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color=aml_style.PALETTE[i],
        )
    ax.axhline(FC, color=aml_style.INK, lw=1, ls=":")
    ax.set_xlabel("cement (kg/m³), water 186, no admixtures")
    ax.set_ylabel("28-day strength (MPa)")
    ax.set_xlim(CEMENT_GRID[0], CEMENT_GRID[1] + 5)
    ax.set_ylim(5, 70)
    ax.legend(loc="upper left", fontsize=11)
    aml_style.save(fig, out_dir, "cement_quantile")


def trace(df: pd.DataFrame, fit: SplitFit) -> tuple[dict[str, object], dict[str, float]]:
    """The proposed mix P (held out in split 0): mean, both 5% quantiles, measured.

    Args:
        df: The concrete tests.
        fit: The first split's models.

    Returns:
        The values at the precision the diagram shows, and the unrounded ones (mean, SD, both
        5% values, measured strength, cement and water) for ``numbers.tex``.

    Raises:
        ValueError: If mix P is not held out in the first split.
    """
    y = df["strength"].to_numpy()
    A = design_matrix(mix_groups(df))
    if PROPOSED_ROW not in fit.te:
        raise ValueError("the proposed mix must be held out in the first split")
    a = A[PROPOSED_ROW]
    raw = {
        "mean": float(a @ fit.w),
        "sd": fit.sd,
        "q_normal": float(a @ fit.w + norm.ppf(TAU) * fit.sd),
        "q_qr": float(a @ fit.wq),
        "y": float(y[PROPOSED_ROW]),
    }
    numbers = {
        **{k: rnd(v, 2) for k, v in raw.items()},
        "cement": float(df.loc[PROPOSED_ROW, "cement"]),
        "water": float(df.loc[PROPOSED_ROW, "water"]),
        "train_tests": len(fit.tr),
    }
    return numbers, {**raw, "cement": numbers["cement"], "water": numbers["water"]}


def write_tex(raw: dict[str, float], n_train: int, out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, from the unrounded values.

    Args:
        raw: :func:`trace`'s unrounded values.
        n_train: The first split's training tests.
        out_dir: Where ``numbers.tex`` goes.

    Returns:
        The path of ``numbers.tex``.
    """
    macros = {
        "Mean": tex_num(raw["mean"], ".2f"),
        "SD": tex_num(raw["sd"], ".2f"),
        "QNorm": tex_num(raw["q_normal"], ".2f"),
        "QQR": tex_num(raw["q_qr"], ".2f"),
        "Meas": tex_num(raw["y"], ".2f"),
        "Cement": tex_num(raw["cement"], "g"),
        "Water": tex_num(raw["water"], "g"),
        "NTrain": tex_num(n_train, "d"),
    }
    return write_tex_macros(macros, out_dir, "cee_l10_calibration_quantiles.py")


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

FAN_TAUS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)  # the fan's quantile lines
SHEET_N: int = 41  # grid points per side of the margin sheets


def _wb(df: pd.DataFrame) -> np.ndarray:
    """Water-binder ratio."""
    return (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()


def fan_lines(df: pd.DataFrame) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray, float]:
    """28-day tests: quantile lines on w/b at ``FAN_TAUS``, and the mean line with its SD.

    Args:
        df: The concrete tests.

    Returns:
        The 28-day tests' w/b, each level's line ``[w0, w1]``, the mean line, and its SD.
    """
    d28 = df[df["age"] == 28]
    x, y = _wb(d28), d28["strength"].to_numpy()
    lines = {t: fit_quantile(x[:, None], y, t) for t in FAN_TAUS}
    w, sd = fit_mean(design_matrix(x[:, None]), y)
    return x, lines, w, sd


def title_art_figure(df: pd.DataFrame, out_dir: Path) -> None:
    """The title art: a fan of golden lines opening from a point, under a dark sun.

    The fan is the lecture's quantile lines of 28-day strength on w/b; below, the pinball
    loss's tilted V.

    Args:
        df: The concrete tests.
        out_dir: Where ``title_art.png`` goes.
    """
    x, lines, _, _ = fan_lines(df)
    c = title_art.Canvas(seed=10)
    c.ground(("#0f1418", "#18101c", "#1c0d0e", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.92, 0.62, 4), soft=4), "#521e15", 0.85)
    rim = c.disc((0.28, 1.18), 0.12, soft=4) - c.disc((0.28, 1.18), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.28, 1.18), 0.10, soft=3), "#1d0a10", 0.92)
    lo, hi = np.quantile(x, [0.02, 0.98])
    g = np.linspace(lo, hi, 50)
    vals = {t: w[0] + w[1] * g for t, w in lines.items()}
    vmin, vmax = min(v.min() for v in vals.values()), max(v.max() for v in vals.values())
    for t, v in vals.items():
        pts = list(zip(0.10 + 0.80 * (g - lo) / (hi - lo), 0.70 + 0.42 * (v - vmin) / (vmax - vmin),
                       strict=True))  # fmt: skip
        c.glow(c.lines([pts], 4 if t == TAU else 2, soft=1.0), "#f3b45c" if t == TAU else "#c9a45e",
               0.85 if t == TAU else 0.45, spread=10)  # fmt: skip
    r = np.linspace(-1, 1, 200)
    v = np.maximum(TAU * r, (TAU - 1) * r)
    c.glow(c.lines([list(zip(0.10 + 0.80 * (r + 1) / 2, 0.14 + 0.36 * v, strict=True))], 4,
                   soft=1.2), "#c2361f", 0.8, spread=10)  # fmt: skip
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean form in rust; above it a fan of "
        "golden lines opening to the left, the brightest at the bottom, the lecture's 5% line; "
        "a dark sun with a burning rim at the upper left, and along the foot a crimson tilted V, "
        "the pinball loss.",
    )


def fan_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """28-day strength against w/b: quantile lines fan out where strength is high.

    Args:
        df: The concrete tests.
        out_dir: Where the figure goes.

    Returns:
        The 90% band's width (MPa) at the 5th and 95th percentiles of w/b.
    """
    x, lines, w, sd = fan_lines(df)
    y = df.loc[df["age"] == 28, "strength"].to_numpy()
    g = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.scatter(x, y, s=10, color=aml_style.MUTED, alpha=0.45, lw=0)
    ax.fill_between(g, *(lines[t][0] + lines[t][1] * g for t in (0.05, 0.95)),
                    color=aml_style.PALETTE[0], alpha=0.10, lw=0)  # fmt: skip
    for t, wt in lines.items():
        ax.plot(g, wt[0] + wt[1] * g, color=aml_style.PALETTE[0], lw=2.6 if t == TAU else 1.2,
                ls="-")  # fmt: skip
    ax.plot(g, w[0] + w[1] * g + norm.ppf(TAU) * sd, color=aml_style.PALETTE[1], lw=2.2, ls="--",
            label=r"mean $-$ 1.645$\sigma$")  # fmt: skip
    ax.plot([], [], color=aml_style.PALETTE[0], lw=2.6, label="quantiles 5, 25, 50, 75, 95%")
    ax.set(xlabel="water–binder ratio $w/b$", ylabel="28-day strength (MPa)", ylim=(0, 90),
           xlim=(x.min() - 0.01, 0.75))  # fmt: skip
    ax.legend(loc="upper right", fontsize=11)
    aml_style.save(fig, out_dir, "fan")
    lo, hi = np.quantile(x, [0.05, 0.95])
    band = lines[0.95] - lines[0.05]  # weights of the 90% band's width
    width = [float(band[0] + band[1] * v) for v in (lo, hi)]
    return {"width_low_wb": round(width[0], 1), "width_high_wb": round(width[1], 1)}


def pinball_figure(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """The pinball loss, tilted; its average over 28-day strengths is lowest at the 5% point.

    Args:
        df: The concrete tests.
        out_dir: Where the figure goes.

    Returns:
        The number of 28-day tests, the best constant on a 0.1 MPa grid, and their 5% point.
    """
    y = df.loc[df["age"] == 28, "strength"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    r = np.linspace(-20, 20, 200)
    for t, ls in ((0.05, "-"), (0.5, ":"), (0.95, "--")):
        axes[0].plot(r, np.maximum(t * r, (t - 1) * r), lw=2.4, ls=ls,
                     label=rf"$\tau$ = {t:g}")  # fmt: skip
    axes[0].set(xlabel="residual $r = y - q$ (MPa)", ylabel=r"loss $\rho_\tau(r)$")
    axes[0].set_title("the tilt", loc="left")
    axes[0].legend(loc="upper center", fontsize=12)
    q = np.linspace(0, 80, 801)
    loss = np.array([pinball(y, np.full(len(y), v), TAU) for v in q])
    ax = axes[1]
    ax.hist(y, bins=np.arange(0, 85, 2.5), color="#DDDDDD", lw=0, density=True)
    ax2 = ax.twinx()
    ax2.plot(q, loss, color=aml_style.PALETTE[0], lw=2.6)
    best = float(q[np.argmin(loss)])
    ax2.axvline(best, color=aml_style.INK, lw=1.2, ls=":")
    ax2.annotate(f"lowest at {best:.1f} MPa\n= the 5% point", (best, loss.min()),
                 xytext=(40, 110), textcoords="offset points", fontsize=12,
                 bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 1.0},  # a callout
                 arrowprops={"arrowstyle": "->"})  # fmt: skip
    ax2.spines["right"].set_visible(True)
    ax.set(xlabel="a constant guess q (MPa)", yticks=[])
    ax2.set_ylabel(r"mean $\rho_{0.05}(y - q)$")
    ax.set_title(f"over the {len(y)} tests at 28 days", loc="left")
    fig.subplots_adjust(wspace=0.35)
    aml_style.save(fig, out_dir, "pinball")
    return {
        "tests": len(y),
        "argmin": round(best, 1),
        "q05": round(float(np.quantile(y, TAU)), 1),
    }


def sheets_figure(df: pd.DataFrame, fit: SplitFit, out_dir: Path) -> dict[str, object]:
    """The margin below the mean, mean - q05, over w/b and age for mix P's family, in 3D.

    Mean - 1.645 sigma subtracts the same margin everywhere: a flat sheet (its height is
    ``decision.margin_normal``). Quantile regression's margin grows where strength is high (low
    w/b, old age). Fits on split 0's training designs.

    Args:
        df: The concrete tests.
        fit: The first split's models.
        out_dir: Where the figure goes.

    Returns:
        The quantile margin's smallest and largest value over the sheet (MPa).
    """
    Z = mix_groups(df)
    wb = np.linspace(*SHEET_WB, SHEET_N)
    la = np.linspace(*np.log(SHEET_AGE_DAYS), SHEET_N)
    W, L = np.meshgrid(wb, la)
    G = np.repeat(Z[[PROPOSED_ROW]], W.size, axis=0)
    G[:, GROUP_NAMES.index("w/b")], G[:, GROUP_NAMES.index("ln age")] = W.ravel(), L.ravel()
    Ag = design_matrix(G)
    m_qr = (Ag @ fit.w - Ag @ fit.wq).reshape(W.shape)
    m_n = np.full(W.shape, -norm.ppf(TAU) * fit.sd)
    fig = plt.figure(figsize=aml_style.SIDE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(W, L, m_n, color=aml_style.PALETTE[1], alpha=0.35, lw=0, zorder=1)
    aml_style.surface3d(ax, W, L, m_qr, cmap="cividis", vmin=float(m_qr.min()),
                        vmax=float(m_qr.max()))  # fmt: skip
    ax.set_xlabel("w/b", labelpad=8, fontsize=12)
    ax.set_ylabel("age (days)", labelpad=8, fontsize=12)
    ticks = (SHEET_AGE_DAYS[0], 28, 90, SHEET_AGE_DAYS[1])
    ax.set_yticks(np.log(ticks), [str(t) for t in ticks])
    ax.set_zlabel("mean − 5% (MPa)", labelpad=6, fontsize=11)
    ax.set_title("margin below the mean", loc="left", fontsize=13)
    flat = r"mean $-$ 1.645$\sigma$: flat"
    ax.plot([], [], color=aml_style.PALETTE[1], lw=8, alpha=0.4, label=flat)
    ax.plot([], [], color=aml_style.PALETTE[0], lw=8, label="quantile regression: tilted")
    ax.legend(loc="center left", fontsize=10.5, bbox_to_anchor=(1.02, 0.5))  # beside the box
    aml_style.style3d(ax, elev=22, azim=-125)
    ax.set_box_aspect(None, zoom=0.85)  # smaller box: the moved labels stay inside the crop
    aml_style.save(fig, out_dir, "sheets")
    return {"qr_margin_range": [round(float(m_qr.min()), 1), round(float(m_qr.max()), 1)]}


def sharpness_figure(h: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Predicted 5% values on held-out designs: the constant is a spike; the model spreads.

    Args:
        h: :func:`held_out`'s predictions.
        out_dir: Where the figure goes.

    Returns:
        Quantile regression's lowest predicted 5% value, its 1st percentile, and how many of
        its predictions fall below zero (MPa; a strength cannot be negative).
    """
    q = h["quantile regression"].to_numpy()
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    bins = np.arange(-20, 75, 2.5)
    ax.hist(h["quantile regression"], bins=bins, color=aml_style.PALETTE[0], alpha=0.8,
            label="quantile regression")  # fmt: skip
    ax.hist(h["no model"], bins=bins, color=aml_style.PALETTE[1], alpha=0.85, hatch="//",
            ec="white", label="no model: one constant")  # fmt: skip
    ax.set(xlabel="predicted 5% strength (MPa)", ylabel="held-out predictions")
    ax.legend(loc="upper right", fontsize=11)
    aml_style.save(fig, out_dir, "sharpness")
    return {
        "qr_min": rnd(q.min(), 1),
        "qr_p01": rnd(np.quantile(q, 0.01), 1),
        "qr_negative": int((q < 0).sum()),
        "qr_negative_pct": rnd(100 * float((q < 0).mean()), 1),
    }


def crossing_figure(q: dict[float, np.ndarray], out_dir: Path) -> None:
    """Separate 5% and 10% fits on all tests: q10 - q05 against the predicted median.

    Args:
        q: :func:`crossing`'s predicted quantiles.
        out_dir: Where the figure goes.
    """
    q05, q10, q50 = q[0.05], q[0.10], q[0.50]
    gap = q10 - q05
    bad = gap < 0
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.scatter(q50[~bad], gap[~bad], s=8, color=aml_style.MUTED, alpha=0.5, lw=0)
    ax.scatter(q50[bad], gap[bad], s=22, color=aml_style.PALETTE[1], lw=0,
               label=f"{bad.sum()} tests: 5% line above the 10%")  # fmt: skip
    ax.axhline(0, color=aml_style.INK, lw=1.2)
    ax.set(xlabel="predicted median (MPa)", ylabel="10% value − 5% value (MPa)")
    ax.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)  # fmt: skip
    aml_style.save(fig, out_dir, "crossing")


def main() -> None:
    """Generate every figure and number for CEE Lecture 10."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = load_concrete()
    repeat_sd = float(repeat_groups(df)["std"].median())  # Lecture 5's noise floor, MPa
    if round(repeat_sd, 1) != 1.4:
        raise RuntimeError(f"the repeat-test SD moved: {repeat_sd}")
    fits = fit_splits(df)
    h = held_out(df, fits)
    cov, losses = coverage(h, out_dir)
    floor = rnd(repeat_sd * norm.pdf(norm.ppf(TAU)), 2)  # E pinball of N(q, sd)
    bracket_figure(losses, floor, out_dir)
    rel, rel_gap = reliability(df, fits, out_dir)
    tr, raw = trace(df, fits[0])
    cross, q_all = crossing(df)
    numbers = {
        "tests_held_out": len(h),
        "attempt": attempt(h),
        "coverage": cov,
        "bootstrap": design_bootstrap(h),
        "constant_mean": rnd(h.groupby("split")["no model"].first().mean(), 1),
        "repeat_sd": rnd(repeat_sd, 1),
        "floor": floor,
        "reliability": rel,
        "reliability_max_gap": rnd(rel_gap, 1),
        "verify": verify(df, fits[0]),
        "crossing": cross,
        "decision": decision(df, fits, out_dir),
        "trace": tr,
    }
    # the visuals that carry the slides
    title_art_figure(df, out_dir)
    numbers["fan"] = fan_figure(df, out_dir)
    numbers["pinball"] = pinball_figure(df, out_dir)
    numbers["sheets"] = sheets_figure(df, fits[0], out_dir)
    numbers["sharpness"] = sharpness_figure(h, out_dir)
    crossing_figure(q_all, out_dir)
    aml_style.thumbnails(out_dir, ("pinball", "sheets", "coverage_bands", "cement_quantile"))
    write_numbers(numbers, out_dir)
    print(write_tex(raw, len(fits[0].tr), out_dir))


if __name__ == "__main__":
    main()
