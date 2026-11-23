"""Figures and numbers for CEE Lecture 11: conformal prediction, covariate shift, nonstationarity.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l11_conformal_shift.py \
        --out build/fig/cee_l11_conformal_shift

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(one forecast traced through the interval, for the TikZ diagram). Data:
``data/derived/usgs_05572000_daily.csv`` from ``scripts/fetch_usgs.sh`` (USGS daily mean
discharge, Sangamon River at Monticello, Illinois; public domain).

The setting: in 1980 an engineer builds a 7-day-ahead forecast of the river's flow from its
flows up to today, fits it on 1950-1969, calibrates a 90% split-conformal interval on
1970-1979, and the interval is used ever since. How often does next week's flow fall inside
it, in later decades, and on the weeks that matter?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aml_style
import title_art
from common import (
    DATA_DIR,
    parse_out_dir,
    pct,
    require,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

CSV: Path = DATA_DIR / "derived" / "usgs_05572000_daily.csv"
SEED: int = 0
H: int = 7  # forecast horizon, days
ALPHA: float = 0.10  # 90% intervals
FIT: tuple[int, int] = (1950, 1969)
CAL: tuple[int, int] = (1970, 1979)
TEST: tuple[int, int] = (2000, 2025)
# Decades outside the fit years that the record covers from their first year to their last: the
# 1970s and the eight others (the 1910s lack the 567 missing days of 1912-1914).
OUT_OF_FIT: tuple[int, ...] = (1910, 1920, 1930, 1940, 1970, 1980, 1990, 2000, 2010)
GAMMA: float = 0.005  # adaptive conformal step size
RISE: float = 1.0  # a "rising week": log flow up by more than 1 (x2.7)
STEADY: float = 0.5  # a "steady week": log flow within 0.5 (x/÷ 1.6)
FLOOR_M3S: float = 0.01  # zero-flow days are floored here before the log
FLOORS_CHECKED_M3S: tuple[float, ...] = (FLOOR_M3S, 0.03, 0.1)  # the floor's effect, checked
MEAN_DAYS: int = 30  # window of the running-mean input
LOW_QUANTILE: float = 0.10  # "low flow now": today below the fit years' 10th percentile
N_HALVINGS: int = 200  # random halvings of the 1970s
LEVELS: tuple[float, ...] = (ALPHA, 0.05, 0.01)  # alphas of the 90%, 95% and 99% intervals
MIN_YEAR_ROWS: int = 300  # a year counts in the one-year reference with at least this many
TRACE_DAY: str = "2024-06-03"
EXAMPLE_YEAR: int = 2013
FEATURES: tuple[str, ...] = ("lag0", "lag1", "lag3", "lag7", "mean30", "sin", "cos")


# ---------------------------------------------------------------------------------------------
# Data and the forecast


def load() -> pd.Series:
    """Daily mean discharge on a complete daily index.

    Returns:
        Discharge in m^3/s by day; missing days are NaN.

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_usgs.sh``).
    """
    d = pd.read_csv(require(CSV, "fetch_usgs.sh"), parse_dates=["date"])
    d = d.set_index("date")["discharge_m3s"]
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))


def design(q: pd.Series, floor_m3s: float = FLOOR_M3S) -> pd.DataFrame:
    """Inputs known today and the target, log flow ``H`` days ahead; rows with gaps dropped.

    Args:
        q: Daily discharge (m^3/s).
        floor_m3s: Flows below this (the zero-flow days) are raised to it before the log.

    Returns:
        One row per issue day: the ``FEATURES`` and the target ``y``.
    """
    lq = np.log(q.clip(lower=floor_m3s))
    doy = q.index.dayofyear
    X = pd.DataFrame(
        {
            "lag0": lq,
            "lag1": lq.shift(1),
            "lag3": lq.shift(3),
            "lag7": lq.shift(7),
            "mean30": lq.rolling(MEAN_DAYS).mean(),
            "sin": np.sin(2 * np.pi * doy / 365.25),
            "cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=q.index,
    )
    X["y"] = lq.shift(-H)
    return X.dropna()


def design_matrix(d: pd.DataFrame) -> np.ndarray:
    """``[1, inputs]``."""
    return np.column_stack([np.ones(len(d)), d[list(FEATURES)].to_numpy()])


def period(d: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    """Rows whose forecast is issued in the given years.

    A row belongs to the period of its issue day, so the last seven fit rows (issued Dec 25-31,
    1969) have targets on Jan 1-7, 1970. No target is shared: :func:`setup` checks that the
    first calibration target follows the last fit target.
    """
    return d[(d.index.year >= years[0]) & (d.index.year <= years[1])]


def decade(index: pd.DatetimeIndex) -> np.ndarray:
    """The decade each forecast is issued in, e.g. 1970 for 1970-1979."""
    return np.asarray((index.year // 10) * 10)


def conformal_quantile(
    scores: np.ndarray, alpha: float = ALPHA, *, presorted: bool = False
) -> float:
    """Split conformal: the ceil((n + 1)(1 - alpha))-th smallest calibration score.

    Args:
        scores: The n calibration scores.
        alpha: Miscoverage level.
        presorted: Whether ``scores`` is already sorted ascending (skips the sort).

    Returns:
        The cutoff; ``inf`` (cover everything) when the rank exceeds n, which includes
        alpha <= 0, and ``-inf`` (an empty interval) when it is below 1, i.e. alpha >= 1: the
        two limits adaptive conformal needs.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float((scores if presorted else np.sort(scores))[k - 1])


@dataclass(frozen=True)
class Forecast:
    """The 1980 forecast: its rows, its least-squares weights, and its 1970s calibration.

    Attributes:
        d: Inputs and target for every day with complete data.
        w: Weights of ``[1, FEATURES]``, fit on 1950-1969.
        s_cal: The 1970s calibration scores, ``|ln y - ln y_hat|``.
        qhat: Their 90% conformal quantile.
    """

    d: pd.DataFrame
    w: np.ndarray
    s_cal: np.ndarray
    qhat: float

    def scores(self, rows: pd.DataFrame) -> np.ndarray:
        """Absolute errors in log flow, ``|ln y - ln y_hat|``, for the given rows."""
        return np.abs(rows["y"].to_numpy() - design_matrix(rows) @ self.w)


def setup(q: pd.Series, floor_m3s: float = FLOOR_M3S) -> Forecast:
    """Fit on 1950-1969, calibrate on 1970-1979.

    Args:
        q: Daily discharge (m^3/s).
        floor_m3s: The zero-flow floor (see :func:`design`).

    Returns:
        The fitted and calibrated forecast.

    Raises:
        ValueError: If a calibration target does not follow every fit target.
    """
    d = design(q, floor_m3s)
    fit, cal = period(d, FIT), period(d, CAL)
    lag = pd.Timedelta(days=H)
    if not cal.index[0] + lag > fit.index[-1] + lag:
        raise ValueError("the first calibration target must follow the last fit target")
    w = np.linalg.lstsq(design_matrix(fit), fit["y"].to_numpy(), rcond=None)[0]
    s_cal = np.abs(cal["y"].to_numpy() - design_matrix(cal) @ w)
    return Forecast(d=d, w=w, s_cal=s_cal, qhat=conformal_quantile(s_cal))


# ---------------------------------------------------------------------------------------------
# Experiments


def summary(q: pd.Series, f: Forecast) -> dict[str, object]:
    """Counts for the question slide and the guarantee.

    Args:
        q: Daily discharge (m^3/s), NaN on missing days.
        f: The forecast.

    Returns:
        The record's span and gaps, its range, the period sizes, the fit/calibration boundary,
        and the guarantee's upper bound.
    """
    fit, cal = period(f.d, FIT), period(f.d, CAL)
    missing_years = q.index[q.isna()].year
    n_cal = len(cal)
    return {
        "first": str(q.first_valid_index().date()),
        "last": str(q.last_valid_index().date()),
        "days": int(q.notna().sum()),
        "missing": int(q.isna().sum()),
        "missing_first_year": int(missing_years.min()),
        "missing_last_year": int(missing_years.max()),
        "max_m3s": rnd(q.max(), 1),
        "zero_days": int((q <= 0).sum()),
        "days_below_floor": int((q < FLOOR_M3S).sum()),
        "n_fit": len(fit),
        "n_cal": n_cal,
        "n_test": len(period(f.d, TEST)),
        "last_fit_target": str((fit.index[-1] + pd.Timedelta(days=H)).date()),
        "first_cal_target": str((cal.index[0] + pd.Timedelta(days=H)).date()),
        "guarantee_max_pct": pct(1 - ALPHA + 1 / (n_cal + 1), 2),
        "cal_scores_below_steady_pct": pct(np.mean(f.s_cal < STEADY)),
        "cal_scores_p995": rnd(np.quantile(f.s_cal, 0.995), 2),
    }


def halvings(r: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Calibrate on a random half of the groups and cover the other half, ``N_HALVINGS`` times.

    Args:
        r: Scores.
        groups: The group of each score (a week, or the day itself).
        rng: The generator that draws the halves.

    Returns:
        The coverage of each halving.
    """
    ids = np.unique(groups)
    covs = []
    for _ in range(N_HALVINGS):
        m = np.isin(groups, rng.permutation(ids)[: len(ids) // 2])
        covs.append(float(np.mean(r[~m] <= conformal_quantile(r[m]))))
    return np.asarray(covs)


def exchangeable_check(f: Forecast) -> tuple[dict[str, object], np.ndarray]:
    """Within 1970-1979: calibrate on half the weeks, test on the other half, 200 times.

    The same halvings by day, which ignore the order of the days, are the reference for
    exchangeable days: the spread they leave is what independence alone would give.

    Args:
        f: The forecast.

    Returns:
        The mean and the 5th and 95th percentiles of the coverages by week and by day, and the
        coverages by week.
    """
    cal = period(f.d, CAL)
    r = f.scores(cal)
    weeks = np.asarray((cal.index - cal.index[0]).days // 7)
    covs = halvings(r, weeks, np.random.default_rng(SEED))
    by_day = halvings(r, np.arange(len(r)), np.random.default_rng(SEED))
    numbers = {
        "mean_pct": pct(np.mean(covs)),
        "p5_pct": pct(np.quantile(covs, 0.05)),
        "p95_pct": pct(np.quantile(covs, 0.95)),
        "by_day_mean_pct": pct(np.mean(by_day)),
        "by_day_p5_pct": pct(np.quantile(by_day, 0.05)),
        "by_day_p95_pct": pct(np.quantile(by_day, 0.95)),
    }
    return numbers, covs


def adaptive(r: np.ndarray, index: pd.DatetimeIndex, s_cal: np.ndarray) -> pd.Series:
    """Adaptive conformal inference (Gibbs & Candès, 2021) from 1980 on: the slide's update.

    The level alpha_t moves after each forecast is judged, H days after it is issued:
    alpha_{t+1} = alpha_t + gamma (alpha - err), up after a hit, down after a miss. The cutoff is
    the conformal order statistic at level alpha_t of the fixed 1970s scores: infinite (cover
    everything) for alpha_t <= 0, empty for alpha_t >= 1. The forecast judged at row i is row
    i - H, so the rows from 1980 on must be consecutive days.

    Args:
        r: Scores of every forecast.
        index: Their issue days.
        s_cal: The 1970s calibration scores.

    Returns:
        Whether each forecast from 1980 on fell inside its interval.

    Raises:
        ValueError: If a day is missing from 1980 on.
    """
    start = int(np.flatnonzero(index.year >= 1980)[0])
    gaps = (index[start + 1 :] - index[start:-1]).days
    if not np.all(gaps == 1):
        raise ValueError("the H-day lag needs consecutive days")
    s_sorted = np.sort(s_cal)
    alpha_t, covered = ALPHA, []
    for i in range(start, len(r)):
        if i - H >= start:  # the forecast made H days ago is judged today
            err = not covered[i - H - start]
            alpha_t += GAMMA * (ALPHA - float(err))
        covered.append(bool(r[i] <= conformal_quantile(s_sorted, alpha_t, presorted=True)))
    return pd.Series(covered, index=index[start:])


def partial_decades(index: pd.DatetimeIndex) -> list[int]:
    """Decades that the record's first or last year cuts (the 1900s and the 2020s)."""
    first, last = int(index.year.min()), int(index.year.max())
    return [k for k in np.unique(decade(index)).tolist() if k < first or k + 9 > last]


def by_decade(f: Forecast) -> tuple[dict[str, object], pd.Series, pd.Series]:
    """Coverage of the fixed interval by decade, and of adaptive conformal after 1980.

    Args:
        f: The forecast.

    Returns:
        The numbers, the fixed interval's hits on every day, and adaptive conformal's from 1980.
    """
    r_all = f.scores(f.d)
    cov_fixed = pd.Series(r_all <= f.qhat, index=f.d.index)
    aci = adaptive(r_all, f.d.index, f.s_cal)
    dec_fixed = cov_fixed.groupby(decade(cov_fixed.index)).mean()
    dec_aci = aci.groupby(decade(aci.index)).mean()
    test = period(cov_fixed.to_frame("c"), TEST)["c"]
    aci_test = period(aci.to_frame("c"), TEST)["c"]
    rows = pd.Series(decade(f.d.index)).value_counts()
    earlier = [dec_fixed[k] for k in OUT_OF_FIT if k < FIT[0]]
    later = [dec_fixed[k] for k in OUT_OF_FIT if k > CAL[1]]
    numbers = {
        "fixed_by_decade": {str(k): pct(v) for k, v in dec_fixed.items()},
        "aci_by_decade": {str(k): pct(v) for k, v in dec_aci.items()},
        "fit_period_pct": pct(period(cov_fixed.to_frame("c"), FIT)["c"].mean()),
        "p1980_1999_pct": pct(period(cov_fixed.to_frame("c"), (1980, 1999))["c"].mean()),
        "test_pct": pct(test.mean()),
        "aci_test_pct": pct(aci_test.mean()),
        "earlier_min_pct": pct(min(earlier)),
        "earlier_max_pct": pct(max(earlier)),
        "later_min_pct": pct(min(later)),
        "later_max_pct": pct(max(later)),
        "partial_rows": {str(k): int(rows[k]) for k in partial_decades(f.d.index)},
    }
    return numbers, cov_fixed, aci


def decades_figure(cov_fixed: pd.Series, aci: pd.Series, out_dir: Path) -> None:
    """Coverage by decade: the fixed 1970s interval against adaptive conformal.

    Decades the record covers only in part (the 1900s and 2020s) get open markers.
    """
    partial = partial_decades(cov_fixed.index)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for cov, marker, ls, color, label in (
        (cov_fixed, "o", "-", aml_style.PALETTE[0], "fixed interval, calibrated 1970s"),
        (aci, "s", "--", aml_style.PALETTE[1], "adaptive conformal, from 1980"),
    ):
        dec = 100 * cov.groupby(decade(cov.index)).mean()
        ax.plot(dec.index, dec.to_numpy(), marker=marker, ls=ls, color=color, label=label)
        part = dec[dec.index.isin(partial)]
        ax.plot(part.index, part.to_numpy(), marker, color=color, mfc="white", mew=1.6, ms=8)
    ax.plot([], [], "o", color=aml_style.MUTED, mfc="white", mew=1.6, label="open: partial decade")
    ax.axhline(90, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.axvspan(1948, 1971, color="#EEEEEE", zorder=0)
    ax.axvspan(1968, 1981, color="#DDDDDD", zorder=0)
    ax.text(1952, 92.6, "fit", fontsize=11, color=aml_style.MUTED)
    ax.text(1970.5, 92.6, "calib.", fontsize=11, color=aml_style.MUTED)
    ax.set_xlabel("decade the forecast is issued")
    ax.set_ylabel("inside the 90% interval (%)")
    ax.set_ylim(84, 94)
    ax.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)  # the shaded bands run behind it  # fmt: skip
    aml_style.save(fig, out_dir, "coverage_decades")


def decade_scores(f: Forecast) -> dict[str, object]:
    """Why other decades fall short: each decade's own 90% score, and recalibration on each.

    The 1950s and 1960s are the fit data and the 1970s the calibration data, so their coverage
    is near 90% by construction. Each decade's 90% conformal score shows how large its errors
    are; calibrating on each decade of ``OUT_OF_FIT`` in turn, and covering the other eight,
    shows whether the 1970s are an unusual calibration decade.

    Args:
        f: The forecast.

    Returns:
        The 90% scores by decade, their ranges, and the recalibrated coverages.

    Raises:
        RuntimeError: If the 1970s' own 90% score is not the forecast's cutoff.
    """
    r = f.scores(f.d)
    dec = decade(f.d.index)
    q90 = {int(k): conformal_quantile(r[dec == k]) for k in np.unique(dec)}
    recal = {
        c: float(np.mean(r[np.isin(dec, OUT_OF_FIT) & (dec != c)] <= q90[c])) for c in OUT_OF_FIT
    }
    if not np.isclose(q90[CAL[0]], f.qhat):
        raise RuntimeError(f"1970s score {q90[CAL[0]]} != cutoff {f.qhat}")
    fit_cal = [q90[k] for k in (1950, 1960, 1970)]
    others = [v for k, v in q90.items() if k >= 1910 and k not in (1950, 1960, 1970)]
    other_recal = [v for c, v in recal.items() if c != CAL[0]]
    return {
        "q90_by_decade": {str(k): rnd(v, 3) for k, v in q90.items()},
        "q90_fit_cal_min": rnd(min(fit_cal), 2),
        "q90_fit_cal_max": rnd(max(fit_cal), 2),
        "q90_other_min": rnd(min(others), 2),
        "q90_other_max": rnd(max(others), 2),
        "recal_pct": {str(c): pct(v) for c, v in recal.items()},
        "recal_1970s_pct": pct(recal[CAL[0]]),
        "recal_other_min_pct": pct(min(other_recal)),
        "recal_other_max_pct": pct(max(other_recal)),
    }


STATES: Final[dict[str, str]] = {  # river-state group -> tick label
    "steady": "steady",
    "falling": "falling",
    "rising_mid": "rising ×1.6–2.7",
    "low flow": "low flow now",
    "rising": "rising ×2.7+",
}


def river_states(f: Forecast, t: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each forecast's river state: four change groups that split every forecast, and low flow.

    Args:
        f: The forecast (its fit years set the low-flow threshold).
        t: The rows to group.

    Returns:
        A mask per state of ``STATES``; "low flow" cuts across the four change groups.

    Raises:
        RuntimeError: If the four change groups do not split every forecast.
    """
    change = (t["y"] - t["lag0"]).to_numpy()
    groups = {
        "rising": change > RISE,
        "rising_mid": (change > STEADY) & (change <= RISE),
        "falling": change < -STEADY,
        "steady": np.abs(change) <= STEADY,
        "low flow": t["lag0"].to_numpy() < np.quantile(period(f.d, FIT)["lag0"], LOW_QUANTILE),
    }
    if sum(groups[k].sum() for k in ("rising", "rising_mid", "falling", "steady")) != len(t):
        raise RuntimeError("the change groups must split every forecast")
    return groups


def conditional(
    f: Forecast, aci: pd.Series
) -> tuple[dict[str, object], pd.Series, pd.Series, dict[str, tuple[float, float]]]:
    """Coverage in 2000-2025 by what the river does next week, by flow level, by month.

    The four change groups (falling, steady, rising x1.6-2.7, rising x2.7+) split every
    forecast; "low flow now" cuts across them. Wider intervals (95%, 99%) and adaptive
    conformal are scored on the rising weeks too.

    Args:
        f: The forecast.
        aci: Adaptive conformal's hits from 1980.

    Returns:
        The numbers; coverage by month for one interval and for one per month (Mondrian); and
        each river state's (coverage, share of forecasts), in percent, for the figure.
    """
    t = period(f.d, TEST)
    r = f.scores(t)
    groups = river_states(f, t)
    stats = {k: (pct(np.mean(r[m] <= f.qhat)), pct(m.mean())) for k, m in groups.items()}
    out: dict[str, object] = {
        k: {"share_pct": share, "coverage_pct": cov} for k, (cov, share) in stats.items()
    }
    cut = {f"{100 * (1 - a):.0f}": conformal_quantile(f.s_cal, a) for a in LEVELS}
    for name, m in (("all", np.ones(len(t), bool)), ("rising", groups["rising"]),
                    ("steady", groups["steady"])):  # fmt: skip
        out[f"{name}_by_level"] = {lv: pct(np.mean(r[m] <= q)) for lv, q in cut.items()}
    hits = aci[t.index].to_numpy()
    out["aci_rising_pct"] = pct(hits[groups["rising"]].mean())
    out["aci_not_rising_pct"] = pct(hits[~groups["rising"]].mean())
    months = pd.Series(r <= f.qhat, index=t.index).groupby(t.index.month).mean()
    out["month_min_pct"] = pct(months.min(), 0)
    out["month_min"] = int(months.idxmin())
    out["month_max_pct"] = pct(months.max(), 0)
    out["mondrian"], by_month = mondrian(f, t, r, groups["rising"])
    return out, months, by_month, stats


def mondrian(
    f: Forecast, t: pd.DataFrame, r: np.ndarray, rise: np.ndarray
) -> tuple[dict[str, object], pd.Series]:
    """Mondrian conformal by month: calibrate each month on its own 1970s scores.

    Args:
        f: The forecast.
        t: The test rows.
        r: Their scores.
        rise: Which of them are rising weeks.

    Returns:
        The numbers, and the coverage in each month.
    """
    cal = period(f.d, CAL)
    q_m = {m: conformal_quantile(f.s_cal[cal.index.month == m]) for m in range(1, 13)}
    cov_m = pd.Series(r <= np.array([q_m[m] for m in t.index.month]), index=t.index)
    by_m = cov_m.groupby(t.index.month).mean()
    numbers = {
        "overall_pct": pct(cov_m.mean()),
        "month_min_pct": pct(by_m.min(), 0),
        "month_min": int(by_m.idxmin()),
        "month_max_pct": pct(by_m.max(), 0),
        "rising_pct": pct(cov_m.to_numpy()[rise].mean()),
    }
    return numbers, by_m


def floor_sensitivity(q: pd.Series, f: Forecast) -> dict[str, object]:
    """Coverage since 2000 with the zero-flow floor at each of ``FLOORS_CHECKED_M3S``.

    Args:
        q: Daily discharge (m^3/s).
        f: The forecast at ``FLOOR_M3S``.

    Returns:
        Overall, low-flow, and rising-week coverage at each floor (keys are the floors).

    Raises:
        RuntimeError: If the default floor does not reproduce the forecast's cutoff.
    """
    out: dict[str, dict[str, float]] = {"test_pct": {}, "low_flow_pct": {}, "rising_pct": {}}
    for floor in FLOORS_CHECKED_M3S:
        g = setup(q, floor)
        if floor == FLOOR_M3S and g.qhat != f.qhat:
            raise RuntimeError("the default floor must reproduce the forecast")
        t = period(g.d, TEST)
        hit = g.scores(t) <= g.qhat
        groups = river_states(g, t)
        out["test_pct"][f"{floor:g}"] = pct(hit.mean())
        out["low_flow_pct"][f"{floor:g}"] = pct(hit[groups["low flow"]].mean())
        out["rising_pct"][f"{floor:g}"] = pct(hit[groups["rising"]].mean())
    return {"floor_m3s": FLOOR_M3S, **out}


def states_figure(stats: dict[str, tuple[float, float]], out_dir: Path) -> None:
    """Coverage by river state, with each state's share of the daily forecasts.

    Args:
        stats: :func:`conditional`'s (coverage, share) per state, in percent.
        out_dir: The figure directory.
    """
    vals = [stats[n][0] for n in STATES]
    shares = [stats[n][1] for n in STATES]
    colors = [aml_style.PALETTE[0]] * (len(STATES) - 1) + [aml_style.PALETTE[1]]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    bars = ax.barh(range(len(STATES)), vals, color=colors)
    for b, v, sh in zip(bars, vals, shares, strict=True):
        ax.text(
            v + 1,
            b.get_y() + b.get_height() / 2,
            f"{v:.1f}%  ({sh:.0f}% of forecasts)",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},  # 90% line passes behind
        )
    ax.axvline(90, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.text(90, -0.7, "90%", color=aml_style.MUTED, fontsize=12, ha="center", va="bottom")
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(list(STATES.values()))
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])  # 90 is labelled on its line
    ax.set_xlabel("inside the 90% interval, 2000–2025 (%)")
    ax.invert_yaxis()
    aml_style.save(fig, out_dir, "coverage_states")


def baseline_width(f: Forecast) -> dict[str, object]:
    """Interval width of the model against persistence (next week = today), both conformal.

    Args:
        f: The forecast.

    Returns:
        The times-or-divided-by factors of no model (climatology), persistence and the model,
        and the model's at 90%, 95% and 99%.
    """
    cal = period(f.d, CAL)
    q_pers = conformal_quantile(np.abs(cal["y"].to_numpy() - cal["lag0"].to_numpy()))
    q_clim = conformal_quantile(np.abs(cal["y"].to_numpy() - np.median(period(f.d, FIT)["y"])))
    return {
        "persistence_factor": rnd(np.exp(q_pers), 1),
        "climatology_factor": rnd(np.exp(q_clim), 1),
        "model_factor": rnd(np.exp(f.qhat), 1),
        "by_level": {
            f"{100 * (1 - a):.0f}": rnd(np.exp(conformal_quantile(f.s_cal, a)), 1) for a in LEVELS
        },
    }


def example_figure(f: Forecast, out_dir: Path) -> dict[str, object]:
    """One spring: the forecast interval issued 7 days earlier against the flow that came.

    Args:
        f: The forecast.
        out_dir: The figure directory.

    Returns:
        The spring's counts of misses and its peak.
    """
    t = f.d[(f.d.index >= f"{EXAMPLE_YEAR}-03-01") & (f.d.index <= f"{EXAMPLE_YEAR}-06-30")]
    pred = design_matrix(t) @ f.w
    target_day = t.index + pd.Timedelta(days=H)
    lo, hi = np.exp(pred - f.qhat), np.exp(pred + f.qhat)
    obs = np.exp(t["y"].to_numpy())
    miss = (obs < lo) | (obs > hi)
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.fill_between(
        target_day,
        lo,
        hi,
        color=aml_style.PALETTE[0],
        alpha=0.2,
        label="90% interval, issued 7 days before",
    )
    ax.plot(target_day, obs, "-", color=aml_style.INK, lw=1.6, label="flow that came")
    ax.plot(target_day[miss], obs[miss], "x", color=aml_style.PALETTE[1], ms=7, label="outside")
    ax.set_yscale("log")
    ax.set_ylabel("discharge (m³/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # Above the axes: inside, the band and the flood peaks reach every corner.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=12)
    aml_style.save(fig, out_dir, "example_spring")
    rises = (t["y"] - t["lag0"]).to_numpy() > RISE
    return {
        "year": EXAMPLE_YEAR,
        "days": len(t),
        "misses": int(miss.sum()),
        "misses_on_rises": int((miss & rises).sum()),
        "peak_m3s": int(round(float(obs.max()))),
        "peak_day": str(target_day[int(np.argmax(obs))].date()),
    }


@dataclass(frozen=True)
class Trace:
    """One forecast traced through the interval, unrounded, for the diagram's macros.

    Attributes:
        issued: The issue day.
        target: The day forecast.
        today: Today's flow (m^3/s).
        pred: The forecast (m^3/s).
        lo: The interval's lower end (m^3/s).
        hi: Its upper end (m^3/s).
        obs: The flow that came (m^3/s).
        qhat: The conformal cutoff, in log flow.
        factor: ``exp(qhat)``.
        n_cal: The number of calibration scores.
        k: The cutoff's rank among them.
    """

    issued: pd.Timestamp
    target: pd.Timestamp
    today: float
    pred: float
    lo: float
    hi: float
    obs: float
    qhat: float
    factor: float
    n_cal: int
    k: int


def trace(f: Forecast) -> tuple[dict[str, object], Trace]:
    """One forecast, issued on ``TRACE_DAY`` for ``H`` days later.

    Args:
        f: The forecast.

    Returns:
        The numbers, rounded as the slides quote them, and the unrounded trace.
    """
    issued = pd.Timestamp(TRACE_DAY)
    row = f.d.loc[[issued]]
    pred = float((design_matrix(row) @ f.w)[0])
    n = len(f.s_cal)
    raw = Trace(
        issued=issued,
        target=issued + pd.Timedelta(days=H),
        today=float(np.exp(row["lag0"].iloc[0])),
        pred=float(np.exp(pred)),
        lo=float(np.exp(pred - f.qhat)),
        hi=float(np.exp(pred + f.qhat)),
        obs=float(np.exp(row["y"].iloc[0])),
        qhat=f.qhat,
        factor=float(np.exp(f.qhat)),
        n_cal=n,
        k=int(np.ceil((n + 1) * (1 - ALPHA))),
    )
    numbers = {
        "day": TRACE_DAY,
        "target": str(raw.target.date()),
        "today_m3s": rnd(raw.today, 2),
        "pred_m3s": rnd(raw.pred, 2),
        "lo_m3s": rnd(raw.lo, 2),
        "hi_m3s": rnd(raw.hi, 1),
        "obs_m3s": rnd(raw.obs, 2),
        "qhat": rnd(raw.qhat, 3),
        "factor": rnd(raw.factor, 3),
        "n_cal": n,
        "k": raw.k,
    }
    return numbers, raw


def write_tex(raw: Trace, out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, formatted from raw values.

    The interval's ends are shown to the precision that the forecast (2 decimals) times or
    divided by the factor (3 decimals) reproduces: 2.60 and 33.4.

    Args:
        raw: The unrounded trace.
        out_dir: The figure directory.

    Returns:
        The path written.
    """
    macros = {
        "Today": tex_num(raw.today, ".2f"),
        "Pred": tex_num(raw.pred, ".2f"),
        "Lo": tex_num(raw.lo, ".2f"),
        "Hi": tex_num(raw.hi, ".1f"),
        "Obs": tex_num(raw.obs, ".2f"),
        "Qhat": tex_num(raw.qhat, ".3f"),
        "Factor": tex_num(raw.factor, ".3f"),
        "NCal": tex_num(raw.n_cal, ","),
        "Kth": tex_num(raw.k, ","),
        "DayIssued": f"{raw.issued:%B} {raw.issued.day}, {raw.issued.year}",
        "DayTarget": f"{raw.target:%B} {raw.target.day}",
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

ART_YEAR: int = 2013  # the year whose flow draws the title art
ROLL_YEARS: int = 3  # window of the rolling coverage, years


def title_art_figure(f: Forecast, out_dir: Path) -> None:
    """The title art: a river of light inside a pale band, under a dark sun.

    The river is the Sangamon's log flow through 2013; the band is the 90% interval around the
    forecast, as wide as the 1970s calibration made it.
    """
    t = f.d[f.d.index.year == ART_YEAR]
    pred = design_matrix(t) @ f.w
    y = t["y"].to_numpy()
    c = title_art.Canvas(seed=11)
    c.ground(("#0b1620", "#101826", "#1a0e16", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.02, 0.04, 0.40, 0.55, 3), soft=4), "#4d1c14", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.58, 0.04, 0.40, 0.70, 4), soft=4), "#6a2618", 0.85)
    rim = c.disc((0.70, 1.20), 0.11, soft=4) - c.disc((0.70, 1.20), 0.09, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.70, 1.20), 0.09, soft=3), "#1d0a10", 0.92)
    lo, hi = min(y.min(), (pred - f.qhat).min()), max(y.max(), (pred + f.qhat).max())
    xs = np.linspace(0.04, 0.96, len(y))

    def to_y(v: np.ndarray) -> np.ndarray:
        return 0.48 + 0.50 * (v - lo) / (hi - lo)

    upper = list(zip(xs, to_y(pred + f.qhat), strict=True))
    band = upper + list(zip(xs[::-1], to_y(pred - f.qhat)[::-1], strict=True))
    c.glaze(c.polygon(band, soft=6), "#9fb7c9", 0.22)
    c.glow(c.lines([list(zip(xs, to_y(y), strict=True))], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean banks in rust; between them a "
        "golden river, one year of the Sangamon's flow, winds inside a pale translucent band, "
        "the forecast's 90% range, and leaps out of it in the spring floods; a dark sun with a "
        "burning rim hangs above.",
    )


def record_figure(q: pd.Series, out_dir: Path) -> None:
    """The gauge's daily flow since 1908, floored as the forecast floors it, with its periods."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(q.index, q.clip(lower=FLOOR_M3S), color=aml_style.MUTED, lw=0.3, ls="-")
    for (a, b), name, color in ((FIT, "fit", aml_style.PALETTE[0]),
                                (CAL, "calibrate", aml_style.PALETTE[2]),
                                (TEST, "test", aml_style.PALETTE[1])):  # fmt: skip
        ax.axvspan(pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-31"), color=color,
                   alpha=0.18, lw=0)  # fmt: skip
        ax.text(pd.Timestamp(f"{a}-06-01"), 900, name, fontsize=12, color=color,
                fontweight="bold")  # fmt: skip
    ax.set(yscale="log", ylabel="daily flow (m³/s)", ylim=(FLOOR_M3S, 2500))
    ax.xaxis.set_major_locator(mdates.YearLocator(20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Sangamon River at Monticello, 1908–2025", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "record")


def scores_figure(f: Forecast, out_dir: Path) -> None:
    """The 1970s calibration scores and the conformal cutoff: 90% of them fall below it.

    The histogram stops at the scores' 99.5th percentile.
    """
    s = f.s_cal
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    bins = np.linspace(0, np.quantile(s, 0.995), 50)
    ax.hist(s[s <= f.qhat], bins=bins, color=aml_style.PALETTE[0], alpha=0.85, label="90% below")
    ax.hist(s[s > f.qhat], bins=bins, color=aml_style.PALETTE[1], alpha=0.9, hatch="//",
            ec="white", label="10% above")  # fmt: skip
    ax.axvline(f.qhat, color=aml_style.INK, lw=1.8)
    ax.annotate(rf"$\hat q$ = {f.qhat:.2f}: $\times/\div$ {np.exp(f.qhat):.1f}", (f.qhat, 0),
                xytext=(10, 140), textcoords="offset points", fontsize=12,
                arrowprops={"arrowstyle": "->"})  # fmt: skip
    ax.set(xlabel=r"score $|\ln y - \ln\hat y|$, 1970–1979", ylabel="forecasts")
    ax.legend(loc="upper right", fontsize=12)
    aml_style.save(fig, out_dir, "scores")


def halves_figure(covs: np.ndarray, out_dir: Path) -> None:
    """Coverage over 200 random halvings of the 1970s weeks: centred on 90%, spread by chance."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.hist(100 * covs, bins=np.arange(85.5, 94.6, 0.5), color=aml_style.PALETTE[0], alpha=0.85)
    ax.axvline(100 * covs.mean(), color=aml_style.INK, lw=2)
    ax.axvline(90, color=aml_style.PALETTE[1], lw=1.6, ls="--")
    ax.text(100 * covs.mean() + 0.15, ax.get_ylim()[1] * 0.92, f"mean {100 * covs.mean():.1f}%",
            fontsize=12)  # fmt: skip
    ax.set(xlabel="coverage on the other half (%)", ylabel="splits")
    aml_style.save(fig, out_dir, "halves")


def yearly(f: Forecast) -> pd.Series:
    """Coverage of the 1970s interval in each year of 2000-2025."""
    t = period(f.d, TEST)
    return pd.Series(f.scores(t) <= f.qhat, index=t.index).groupby(t.index.year).mean()


def one_year(f: Forecast, by_year: pd.Series) -> dict[str, object]:
    """One year at a time since 2000, against the one-year spread within a decade.

    The reference: within each decade of ``OUT_OF_FIT``, calibrate on nine years and cover the
    tenth, for every year with at least ``MIN_YEAR_ROWS`` forecasts.

    Args:
        f: The forecast.
        by_year: :func:`yearly`'s coverage by year.

    Returns:
        The worst and best years since 2000, the lowest of the others, and the reference's
        range in the 1970s and over all nine decades.
    """
    r = f.scores(f.d)
    yr = np.asarray(f.d.index.year)
    loyo: dict[int, list[float]] = {}
    for dec0 in OUT_OF_FIT:
        for y0 in range(dec0, dec0 + 10):
            out = yr == y0
            if out.sum() >= MIN_YEAR_ROWS:
                inside = (yr >= dec0) & (yr < dec0 + 10) & ~out
                cov = float(np.mean(r[out] <= conformal_quantile(r[inside])))
                loyo.setdefault(dec0, []).append(cov)
    pooled = [c for covs in loyo.values() for c in covs]
    worst = int(by_year.idxmin())
    return {
        "worst_year": worst,
        "worst_pct": pct(by_year.min(), 0),
        "best_pct": pct(by_year.max(), 0),
        "rest_min_pct": pct(by_year.drop(worst).min(), 0),
        "loyo_1970s_min_pct": pct(min(loyo[CAL[0]])),
        "loyo_1970s_max_pct": pct(max(loyo[CAL[0]])),
        "loyo_all_min_pct": pct(min(pooled)),
        "loyo_all_max_pct": pct(max(pooled)),
        "loyo_years": len(pooled),
        "worst_below_every_loyo": bool(by_year.min() < min(pooled)),
    }


def years_figure(by_year: pd.Series, out_dir: Path) -> None:
    """Coverage of the 1970s interval, year by year, 2000-2025."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    worst = int(by_year.idxmin())
    colors = [aml_style.PALETTE[1] if y == worst else aml_style.PALETTE[0] for y in by_year.index]
    ax.bar(by_year.index, 100 * by_year.to_numpy(), color=colors)
    ax.axhline(90, color=aml_style.INK, lw=1.4, ls="--")
    ax.annotate(f"{worst}: {100 * by_year.min():.0f}%", (worst, 100 * by_year.min()),
                xytext=(40, -45), textcoords="offset points", ha="left", fontsize=12,
                color=aml_style.PALETTE[1], fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none"},
                arrowprops={"arrowstyle": "->", "color": aml_style.PALETTE[1]})  # fmt: skip
    ax.set(ylim=(60, 100), xlabel="year", ylabel="inside the 90% interval (%)")
    aml_style.save(fig, out_dir, "years")


def months_figure(fixed: pd.Series, mond: pd.Series, out_dir: Path) -> None:
    """Coverage by month since 2000: one interval for all months against one per month."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    m = np.arange(1, 13)
    ax.plot(m, 100 * fixed.to_numpy(), "o-", color=aml_style.PALETTE[0], label="one interval")
    ax.plot(m, 100 * mond.to_numpy(), "s--", color=aml_style.PALETTE[1], label="one per month")
    ax.axhline(90, color=aml_style.INK, lw=1.2, ls=":")
    ax.set(xticks=m, xticklabels=list("JFMAMJJASOND"), ylim=(60, 100), xlabel="month",
           ylabel="inside, 2000–2025 (%)")  # fmt: skip
    ax.legend(loc="lower right", fontsize=12)
    aml_style.save(fig, out_dir, "months")


def rolling(cov_fixed: pd.Series, aci: pd.Series) -> tuple[dict[str, object], pd.DataFrame]:
    """Coverage over the last ``ROLL_YEARS`` years, day by day since 1980.

    Args:
        cov_fixed: The fixed interval's hits on every day.
        aci: Adaptive conformal's hits from 1980.

    Returns:
        The ranges of both rolling coverages and the fixed one's lowest windows before and
        after 2000; and the two rolling coverages (percent) for the figure.
    """
    win = 365 * ROLL_YEARS
    roll = pd.DataFrame(
        {
            "fixed": 100 * cov_fixed[aci.index].astype(float).rolling(win).mean(),
            "aci": 100 * aci.astype(float).rolling(win).mean(),
        }
    )
    full = roll.dropna()
    before, after = full[full.index.year < TEST[0]], full[full.index.year >= TEST[0]]
    return {
        "years": ROLL_YEARS,
        "first_day": str(full.index[0].date()),
        "fixed_min_pct": rnd(full["fixed"].min(), 1),
        "fixed_max_pct": rnd(full["fixed"].max(), 1),
        "aci_min_pct": rnd(full["aci"].min(), 1),
        "aci_max_pct": rnd(full["aci"].max(), 1),
        "fixed_dip_before_2000": int(before["fixed"].idxmin().year),
        "fixed_dip_since_2000": int(after["fixed"].idxmin().year),
    }, roll


def aci_figure(roll: pd.DataFrame, out_dir: Path) -> None:
    """Three-year rolling coverage since 1980: the fixed interval against adaptive conformal."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(roll.index, roll["fixed"], color=aml_style.PALETTE[0], lw=1.6, ls="-",
            label="fixed 1970s interval")  # fmt: skip
    ax.plot(roll.index, roll["aci"], color=aml_style.PALETTE[1], lw=1.6, ls="--",
            label="adaptive conformal")  # fmt: skip
    ax.axhline(90, color=aml_style.INK, lw=1.2, ls=":")
    ax.set(ylabel=f"inside, last {ROLL_YEARS} years (%)", ylim=(80, 96))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=12)  # above the dips
    aml_style.save(fig, out_dir, "aci")


def main() -> None:
    """Generate every figure and number for CEE Lecture 11."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    q = load()
    f = setup(q)
    tr, tr_raw = trace(f)
    exch, covs = exchangeable_check(f)
    decades, cov_fixed, aci = by_decade(f)
    cond, months_fixed, months_mondrian, state_stats = conditional(f, aci)
    by_year = yearly(f)
    roll_numbers, roll = rolling(cov_fixed, aci)
    numbers = {
        "summary": summary(q, f),
        "exchangeable": exch,
        "decades": decades,
        "decade_scores": decade_scores(f),
        "conditional": cond,
        "floor": floor_sensitivity(q, f),
        "widths": baseline_width(f),
        "example": example_figure(f, out_dir),
        "trace": tr,
        "weights": [rnd(v, 3) for v in f.w],
        "years": one_year(f, by_year),
        "rolling": roll_numbers,
    }
    decades_figure(cov_fixed, aci, out_dir)
    states_figure(state_stats, out_dir)
    # the visuals that carry the slides
    title_art_figure(f, out_dir)
    record_figure(q, out_dir)
    scores_figure(f, out_dir)
    halves_figure(covs, out_dir)
    years_figure(by_year, out_dir)
    months_figure(months_fixed, months_mondrian, out_dir)
    aci_figure(roll, out_dir)
    aml_style.thumbnails(out_dir, ("scores", "coverage_decades", "coverage_states", "aci"))
    write_numbers(numbers, out_dir)
    print(write_tex(tr_raw, out_dir))


if __name__ == "__main__":
    main()
