"""Figures and numbers for CEE Lecture 9: outliers and robust estimation (leak detection).

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l09_robust_estimation.py \
        --out build/fig/cee_l09_robust_estimation

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one reading, for the TikZ diagram). Data: ``data/raw/battledim/`` from
``scripts/fetch_battledim.sh`` (BattLeDIM, Vrachimis et al., 2022, CC BY 4.0): simulated SCADA
readings of the L-Town network, every 5 minutes, with the leaks known.

The question: pressure sensor n229 normally moves with its neighbor n636 along a line. A burst
nearby pulls n229 below that line. Fit the line on a window of readings that may contain the
burst (or a sensor fault), then alarm when n229 falls below it. The burst: pipe p673, from
2018-03-05 15:45 until its repair on 2018-03-23 10:25. The same alarm misses October's leak in
pipe p158, which lowers n229 and its neighbor together.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from matplotlib.collections import LineCollection
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

import aml_style
import title_art
from common import (
    DATA_DIR,
    parse_out_dir,
    require,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

RAW: Path = DATA_DIR / "raw" / "battledim"
SEED: int = 0
SENSOR: str = "n229"  # the sensor next to the burst (about 100 m away)
NEIGHBOR: str = "n636"  # its most correlated neighbor (about 370 m from the burst)
BURST_PIPE: str = "p673"
BURST_START: str = "2018-03-05 15:45"
BURST_END: str = "2018-03-23 10:20"  # last reading before the repair at 10:25
PRE_BURST_END: str = str(pd.Timestamp(BURST_START) - pd.Timedelta(minutes=5))  # last calm reading
CAL: tuple[str, str] = ("2018-02-20", "2018-02-27 23:55")  # a calm week: the noise scale
WINDOW: tuple[str, str] = ("2018-02-28", "2018-03-07 23:55")  # the fitting window
LONGER: tuple[str, ...] = (
    "2018-03-06 23:55",
    "2018-03-08 23:55",
    "2018-03-10 23:55",
    "2018-03-11 23:55",
    "2018-03-12 23:55",
    "2018-03-15 23:55",
)
ALARM_PLOT: tuple[str, str] = ("2018-03-03", "2018-03-08 23:55")  # the alarm figure's span
TITLE_SPAN: tuple[str, str] = ("2018-03-01", "2018-03-10 23:55")  # the title art's horizon
TRACE_TIME: str = "2018-03-05 17:00"  # the reading traced through the diagram
# The attempt's table: hourly means at 14:00 and 17:00, the day before the burst and the day of.
ATTEMPT_TIMES: tuple[str, ...] = (
    "2018-03-04 14:00",
    "2018-03-04 17:00",
    "2018-03-05 14:00",
    "2018-03-05 17:00",
)
CAL_LAST_HOUR: str = str(pd.Timestamp(CAL[1]).floor("h"))  # the calm week's last full hour
# The last full hour before the burst (15:45 falls in the 15:00 hour).
PRE_BURST_LAST_HOUR: str = str(pd.Timestamp(BURST_START).floor("h") - pd.Timedelta(hours=1))
AFTER_REPAIR: str = "2018-03-24"  # the first full day after the repair
INFLOW_PIPES: tuple[str, ...] = ("p227", "p235")  # the town's inflow: its two supply mains
MISS_PIPE: str = "p158"  # the leak nearest n229 after the repair (October 2018)
DAYS_AROUND: int = 10  # days before and after a leak's start, for its pressure drop
K_SIGMA: float = 3.0  # RANSAC inlier threshold, in noise standard deviations
ALARM_SIGMA: float = 5.0  # alarm when the hourly mean residual falls this far below
HUBER_T: float = 1.345  # Huber's tuning constant (95% efficiency under normal noise)
RLM_MATCH: float = 1e-12  # IRLS must match statsmodels' RLM to this relative difference
HAC_LAGS: int = 24  # Newey-West lags for the slope's standard error (2 hours of readings)
DROPOUTS: tuple[int, ...] = (100, 700, 1300)  # readings of the neighbor set to 0 (a fault)
TRIALS: int = 200  # RANSAC trials on the burst window and on the dropouts
TRIALS_LONG: int = 500  # RANSAC trials on the longer windows of the breakdown
CONSENSUS_TRIALS: int = 2000  # trials for each group's own best line (calm alone, burst alone)
SEEDS: int = 20
PER_HOUR: int = 12  # readings per hour
PER_DAY: int = 288  # readings per day
MAD_CONST: float = float(norm.ppf(0.75))  # 0.6745: MAD/0.6745 estimates sigma
SAME: Final[float] = 1e-9  # two computations of one slope must agree to this


# ---------------------------------------------------------------------------------------------
# Data


def read(name: str) -> pd.DataFrame:
    """One BattLeDIM CSV (semicolons, decimal commas, 5-minute timestamps).

    Args:
        name: File name under ``data/raw/battledim``.

    Returns:
        The readings, indexed by timestamp.

    Raises:
        FileNotFoundError: If the file is missing (run ``scripts/fetch_battledim.sh``).
    """
    path = require(RAW / name, "fetch_battledim.sh")
    return pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)


def town_inflow(F: pd.DataFrame) -> pd.Series:
    """The town's inflow (m³/h): the sum of its supply mains, ``INFLOW_PIPES``.

    Args:
        F: The SCADA flows.

    Returns:
        The inflow at each reading.
    """
    return sum((F[p] for p in INFLOW_PIPES), start=pd.Series(0.0, index=F.index))


def leak_table() -> pd.DataFrame:
    """Every leak of the dataset's configuration (BattLeDIM ground truth), both years.

    An incipient leak grows from its start time to full size at its peak time; an abrupt one is
    full size at once.

    Returns:
        One row per leak: pipe, start, end, diameter (m), kind, and peak; times as timestamps.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If a leak's kind is neither incipient nor abrupt.
    """
    config = yaml.safe_load(
        require(RAW / "dataset_configuration.yaml", "fetch_battledim.sh").read_text()
    )
    rows = [[f.strip() for f in str(e).split(",")] for e in config["leakages"] if e is not None]
    leaks = pd.DataFrame(rows, columns=["pipe", "start", "end", "diameter_m", "kind", "peak"])
    for col in ("start", "end", "peak"):
        leaks[col] = pd.to_datetime(leaks[col], errors="raise")
    if not set(leaks["kind"]) <= {"incipient", "abrupt"}:
        raise ValueError(f"unexpected leak kinds: {set(leaks['kind'])}")
    return leaks


@dataclass(frozen=True)
class Network:
    """The parts of the EPANET file ``L-TOWN.inp`` the lecture uses.

    Attributes:
        counts: Entries per section, e.g. ``counts["[PIPES]"]``.
        coords: Node name to (x, y) in meters.
        pipes: Pipe name to its two end nodes.
        length_m: Total pipe length in meters.
    """

    counts: dict[str, int]
    coords: dict[str, tuple[float, float]]
    pipes: dict[str, tuple[str, str]]
    length_m: float

    def distance(self, pipe: str, node: str) -> float:
        """Distance in meters from ``node`` to the midpoint of ``pipe``."""
        a, b = self.pipes[pipe]
        mid = (
            (self.coords[a][0] + self.coords[b][0]) / 2,
            (self.coords[a][1] + self.coords[b][1]) / 2,
        )
        return math.dist(self.coords[node], mid)


def read_network() -> Network:
    """Parse the junction, pipe, and coordinate sections of ``L-TOWN.inp``.

    Returns:
        The network's counts, node coordinates, pipe ends, and total pipe length.
    """
    counts: dict[str, int] = {}
    coords: dict[str, tuple[float, float]] = {}
    pipes: dict[str, tuple[str, str]] = {}
    length = 0.0
    section = ""
    text = require(RAW / "L-TOWN.inp", "fetch_battledim.sh").read_text(errors="replace")
    for line in text.splitlines():
        entry = line.split(";")[0].strip()
        if not entry:
            continue
        if entry.startswith("["):
            section = entry
            counts[section] = 0
            continue
        counts[section] += 1
        f = entry.split()
        if section == "[COORDINATES]":
            coords[f[0]] = (float(f[1]), float(f[2]))
        elif section == "[PIPES]":
            pipes[f[0]] = (f[1], f[2])
            length += float(f[3])
    return Network(counts=counts, coords=coords, pipes=pipes, length_m=length)


def design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``A = [1, neighbor]`` and ``y = sensor`` for a block of readings."""
    A = np.column_stack([np.ones(len(frame)), frame[NEIGHBOR].to_numpy()])
    y = frame[SENSOR].to_numpy()
    assert A.shape == (len(y), 2)
    return A, y


@dataclass(frozen=True)
class Window:
    """The fitting window: design, responses, and which readings follow the burst."""

    A: np.ndarray
    y: np.ndarray
    burst: np.ndarray


def fitting_window(P: pd.DataFrame) -> Window:
    """The readings of ``WINDOW``, with the burst's first days in it."""
    win = P.loc[WINDOW[0] : WINDOW[1]]
    A, y = design(win)
    return Window(A=A, y=y, burst=np.asarray(win.index >= BURST_START))


# ---------------------------------------------------------------------------------------------
# Estimators (the code slide's listings)


def lstsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares weights."""
    return np.linalg.lstsq(A, y, rcond=None)[0]


def huber_irls(
    A: np.ndarray, y: np.ndarray, t: float = HUBER_T, tol: float = 1e-12, max_steps: int = 500
) -> tuple[np.ndarray, int, float]:
    """Huber regression by iteratively reweighted least squares, scale from the MAD.

    Args:
        A: Design matrix with an intercept column.
        y: Responses.
        t: Huber's constant, in units of the scale.
        tol: Stop when no weight moves more than this.
        max_steps: Give up after this many steps.

    Returns:
        Weights, steps taken, and the final scale (MAD of the residuals about 0, over 0.6745,
        as statsmodels' RLM computes it).

    Raises:
        RuntimeError: If the iteration has not converged.
    """
    w = lstsq(A, y)  # start from least squares
    for step in range(1, max_steps + 1):
        r = y - A @ w
        s = max(np.median(np.abs(r)) / MAD_CONST, 1e-8)  # robust scale, never 0
        u = np.abs(r) / s
        wt = np.where(u <= t, 1.0, t / u)  # Huber weights psi(u)/u
        sw = np.sqrt(wt)
        w_new = lstsq(A * sw[:, None], y * sw)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new, step, float(s)
        w = w_new
    raise RuntimeError("Huber IRLS did not converge")


def ransac_line(
    A: np.ndarray, y: np.ndarray, tau: float, trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC for a line: fit pairs, keep the largest consensus, refit on it.

    Args:
        A: Design matrix ``[1, x]``.
        y: Responses.
        tau: Inlier threshold on the absolute residual.
        trials: Number of random pairs.
        rng: Random generator.

    Returns:
        Weights refit by least squares on the best consensus set, and its inlier mask.
    """
    best = np.zeros(len(y), dtype=bool)
    for _ in range(trials):
        i, j = rng.choice(len(y), size=2, replace=False)  # minimal sample: 2 points
        if A[i, 1] == A[j, 1]:
            continue  # no line through these two
        w = np.linalg.solve(A[[i, j]], y[[i, j]])
        inl = np.abs(y - A @ w) < tau
        if inl.sum() > best.sum():
            best = inl
    return lstsq(A[best], y[best]), best  # refit on the consensus


def trials_needed(eps: float, s: int = 2, p: float = 0.99) -> int:
    """RANSAC trials for probability ``p`` of one all-inlier sample (Fischler & Bolles, 1981)."""
    return math.ceil(math.log(1 - p) / math.log(1 - (1 - eps) ** s))


def autocorr(r: np.ndarray, lag: int) -> float:
    """Correlation of a residual series with itself ``lag`` readings later."""
    return float(np.corrcoef(r[:-lag], r[lag:])[0, 1])


# ---------------------------------------------------------------------------------------------
# Experiments


def network_summary(
    P: pd.DataFrame, K: pd.DataFrame, F: pd.DataFrame, net: Network
) -> dict[str, object]:
    """Counts for the question slide, from the EPANET file and the 2018 readings.

    Args:
        P: The 2018 pressures.
        K: The 2018 leak flows (m³/h), one column per leak.
        F: The 2018 SCADA flows.
        net: The parsed network.

    Returns:
        The network's size, the burst's flow, share of the inflow and duration, and the
        sensors' distances.

    Raises:
        RuntimeError: If the burst does not run at every reading from its start to the repair.
    """
    inflow = town_inflow(F)
    burst = K.loc[BURST_START:BURST_END, BURST_PIPE]
    if not (burst > 0).all():
        raise RuntimeError(f"{BURST_PIPE} stops before {BURST_END}")
    return {
        "junctions": net.counts["[JUNCTIONS]"],
        "pipes": net.counts["[PIPES]"],
        "pipe_km": round(net.length_m / 1000, 1),
        "pressure_sensors": int(P.shape[1]),
        "readings_per_day": PER_DAY,
        "leaks_2018": int(K.shape[1]),
        "burst_m3h": round(float(burst.mean()), 1),
        "burst_m3_per_day": int(round(24 * float(burst.mean()), -1)),
        "burst_share_of_inflow_pct": int(
            round(100 * float(burst.mean()) / float(inflow[burst.index].mean()))
        ),
        "burst_days": rnd(len(burst) / PER_DAY, 1),  # start to repair, 5 minutes a reading
        "sensor_m_from_burst": int(round(net.distance(BURST_PIPE, SENSOR), -1)),
        "neighbor_m_from_burst": int(round(net.distance(BURST_PIPE, NEIGHBOR), -1)),
        "sensors_apart_m": int(round(math.dist(net.coords[SENSOR], net.coords[NEIGHBOR]), -1)),
    }


def attempt(P: pd.DataFrame, F: pd.DataFrame, slope: float) -> dict[str, object]:
    """Hourly means of the two sensors at 14:00 and 17:00, the day before and the day of.

    Args:
        P: The 2018 pressures.
        F: The 2018 SCADA flows.
        slope: The calm line's slope (the clean fit), which predicts n229's drop from n636's.

    Returns:
        The table's rows, each day's drops, and the burst day's drop beyond the prediction.
    """
    H = P[[SENSOR, NEIGHBOR]].resample("h").mean()
    inflow = town_inflow(F).resample("h").mean()
    rows = {
        t: {
            "sensor": round(float(H.loc[t, SENSOR]), 2),
            "neighbor": round(float(H.loc[t, NEIGHBOR]), 2),
            "inflow": int(round(float(inflow[t]))),
        }
        for t in ATTEMPT_TIMES
    }
    # The drops are differences of the rounded table entries on purpose: the answers in the
    # notes must match what students compute from the table (unrounded, n636 fell 0.318 m on
    # March 5, not 0.31).
    drops = {}
    for day in ("2018-03-04", "2018-03-05"):
        a, b = rows[f"{day} 14:00"], rows[f"{day} 17:00"]
        drops[day] = {
            "sensor": round(b["sensor"] - a["sensor"], 2),
            "neighbor": round(b["neighbor"] - a["neighbor"], 2),
        }
    # The answer's arithmetic, from the numbers the notes show: the calm slope to 2 decimals
    # times n636's drop is what demand explains; n229 fell this much more.
    day = drops[ATTEMPT_TIMES[-1][:10]]
    predicted = round(round(slope, 2) * day["neighbor"], 2)
    excess = {
        "slope": round(slope, 2),
        "predicted": predicted,
        "beyond_prediction": round(day["sensor"] - predicted, 2),
        "beyond_neighbor": round(day["sensor"] - day["neighbor"], 2),
    }
    return {"rows": rows, "drops": drops, "excess": excess}


def calm_week(
    P: pd.DataFrame, K: pd.DataFrame, net: Network
) -> tuple[dict[str, object], float, np.ndarray]:
    """The line's noise on a calm week, which sets tau, and the leaks running during it.

    The week is calm near n229, not leak-free: three slow leaks run elsewhere in the town.

    Returns:
        The numbers, the noise SD (m), and the calm week's least-squares line, the trusted
        baseline the alarm uses.
    """
    A, y = design(P.loc[CAL[0] : CAL[1]])
    w = lstsq(A, y)
    r = y - A @ w
    sigma = float(np.std(r))
    leaks = K.loc[CAL[0] : CAL[1]]
    active = leaks.columns[(leaks > 0).any()].tolist()
    assert BURST_PIPE not in active, active
    nearest = min(net.distance(p, SENSOR) for p in active)
    numbers = {
        "sigma": rnd(sigma, 4),
        "tau": rnd(K_SIGMA * sigma, 3),
        "slope_cal": rnd(w[1], 4),
        "mad_scale_cal": rnd(np.median(np.abs(r)) / MAD_CONST, 4),
        "leaks_active_cal": len(active),
        "nearest_leak_cal_m": round(nearest),
    }
    return numbers, sigma, w


def clean_fit(win: Window) -> tuple[dict[str, object], np.ndarray]:
    """Least squares on the window's readings before the burst, with a Newey-West SE.

    Successive residuals are correlated, so the textbook standard error (independent errors)
    is too small; the HAC estimator (Newey & West, 1987) allows for it.

    Returns:
        The numbers and the clean line's weights.
    """
    A, y = win.A[~win.burst], win.y[~win.burst]
    ols = sm.OLS(y, A)
    hac = ols.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS, "use_correction": True})
    # The answer depends on the window: a one-day window spans the daily cycle, whose negative
    # correlations half a day apart offset the positive ones.
    hac_day = ols.fit(cov_type="HAC", cov_kwds={"maxlags": PER_DAY, "use_correction": True})
    iid = ols.fit()
    w = lstsq(A, y)
    assert np.allclose(hac.params, w, rtol=1e-9, atol=0), (hac.params, w)
    r = y - A @ w
    numbers = {
        "slope_clean": rnd(w[1], 4),
        "se_slope_hac": float(f"{hac.bse[1]:.1e}"),
        "se_slope_hac_day": float(f"{hac_day.bse[1]:.1e}"),
        "se_slope_iid": float(f"{iid.bse[1]:.1e}"),
        "resid_autocorr_5min": rnd(autocorr(r, 1), 2),
        "resid_autocorr_day": rnd(autocorr(r, PER_DAY), 2),
    }
    return numbers, w


def robust_fits(
    win: Window, tau: float, w_clean: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray], np.ndarray]:
    """Least squares, Huber (checked against statsmodels), and RANSAC on the burst window.

    Returns:
        The numbers, each method's line (``ols``, ``huber``, ``ransac``), and RANSAC's inliers.
    """
    A, y, burst = win.A, win.y, win.burst
    w_ols = lstsq(A, y)
    w_hub, steps, scale = huber_irls(A, y)
    rlm = sm.RLM(y, A, M=sm.robust.norms.HuberT(t=HUBER_T)).fit(
        scale_est="mad", tol=1e-12, maxiter=500, conv="coefs"
    )
    rel_rlm = float(np.max(np.abs(w_hub - rlm.params) / np.abs(rlm.params)))
    if not rel_rlm < RLM_MATCH:  # the slide quotes the bound, not the round-off digits
        raise RuntimeError(f"IRLS differs from statsmodels' RLM by {rel_rlm:.1e}")
    # scikit-learn's HuberRegressor estimates the scale jointly; alpha=0 removes its default
    # L2 penalty, so it is Huber's estimator as the slide names it.
    sk_huber = HuberRegressor(epsilon=HUBER_T, alpha=0.0, max_iter=2000).fit(A[:, 1:], y)
    w_ran, inl = ransac_line(A, y, tau, trials=TRIALS, rng=np.random.default_rng(SEED))
    shift = {}
    for name, w in (("clean", w_clean), ("ols", w_ols), ("huber", w_hub), ("ransac", w_ran)):
        r = y - A @ w
        shift[name] = rnd(r[burst].mean() - r[~burst].mean(), 4)
    numbers = {
        "window_readings": len(y),
        "burst_readings": int(burst.sum()),
        "burst_share_pct": round(100 * float(burst.mean())),
        "slope_ols": rnd(w_ols[1], 4),
        "slope_huber": rnd(w_hub[1], 4),
        "slope_ransac": rnd(w_ran[1], 4),
        "slope_sklearn_huber": rnd(sk_huber.coef_[0], 4),
        "huber_steps": steps,
        "huber_scale": rnd(scale, 4),
        "huber_rlm_rel_diff": float(f"{rel_rlm:.0e}"),
        "huber_rlm_bound": RLM_MATCH,
        # Half the gap between the bands: a threshold above it lets a line midway between the
        # bands hold readings of both.
        "half_gap_m": rnd(abs(shift["ransac"]) / 2, 2),
        "ransac_inliers": int(inl.sum()),
        "ransac_inliers_in_burst": int(inl[burst].sum()),
        # The burst's share plus 5 points of margin: the notes' "7 to 9 trials".
        "trials_for_window": trials_needed(float(burst.mean()) + 0.05),
        "trials_table": {str(e): trials_needed(e) for e in (0.3, 0.5, 0.7)},
        "shift": shift,
    }
    return numbers, {"ols": w_ols, "huber": w_hub, "ransac": w_ran}, inl


def ransac_seeds(win: Window, tau: float, slope_clean: float, out_dir: Path) -> dict[str, object]:
    """scikit-learn's RANSAC over 20 seeds, with its default threshold (MAD of y) and with tau."""
    X, y = win.A[:, 1:], win.y
    seeds_default = [
        float(RANSACRegressor(random_state=s).fit(X, y).estimator_.coef_[0]) for s in range(SEEDS)
    ]
    seeds_tau = [
        float(RANSACRegressor(residual_threshold=tau, random_state=s).fit(X, y).estimator_.coef_[0])
        for s in range(SEEDS)
    ]
    default_thr = float(np.median(np.abs(y - np.median(y))))
    labels = [
        f"default: MAD of y,\n{default_thr:.2f} m",
        f"3$\\sigma$ of a calm week,\n{tau:.2f} m",
    ]
    seeds_figure((seeds_default, seeds_tau), labels, slope_clean, out_dir)
    return {
        "default_threshold": round(default_thr, 2),
        "seeds_default_bad": sum(abs(v - slope_clean) > 0.05 for v in seeds_default),
        "seeds_tau_bad": sum(abs(v - slope_clean) > 0.05 for v in seeds_tau),
        "seeds_default_range": [round(min(seeds_default), 2), round(max(seeds_default), 2)],
        "seeds_tau_range": [round(min(seeds_tau), 3), round(max(seeds_tau), 3)],
    }


def seeds_figure(
    slopes: tuple[list[float], list[float]], labels: list[str], slope_clean: float, out_dir: Path
) -> None:
    """Strip plot of RANSAC's slope over the seeds, one column per threshold."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, vals in enumerate(slopes):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.plot(
            i + jitter,
            vals,
            aml_style.MARKERS[i],
            color=aml_style.PALETTE[i],
            ms=8,
            alpha=0.8,
            ls="none",
        )
    ax.axhline(slope_clean, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.text(
        0.5,  # between the two groups of markers
        slope_clean + 0.02,
        f"clean fit {slope_clean:.2f}",
        color=aml_style.MUTED,
        fontsize=12,
        ha="center",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("RANSAC slope, 20 seeds")
    aml_style.save(fig, out_dir, "ransac_seeds")


def burst_lines_figure(
    win: Window, lines: dict[str, np.ndarray], w_clean: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """The decisive figure: the window's readings and the three fitted lines.

    Args:
        win: The fitting window.
        lines: Each method's line, by its legend name.
        w_clean: The clean fit, on the readings before the burst.
        out_dir: The figure directory.

    Returns:
        For the alt text: the calm readings' range of n636, and the clean line at its ends.
    """
    A, y, burst = win.A, win.y, win.burst
    x_calm = A[~burst, 1]
    ends = np.array([x_calm.min(), x_calm.max()])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(
        A[~burst, 1], y[~burst], ".", ms=3, color="#9A9A9A", alpha=0.6, label="before the burst"
    )
    ax.plot(
        A[burst, 1],
        y[burst],
        "x",
        ms=4,
        color=aml_style.PALETTE[3],
        alpha=0.6,
        label="after 15:45, Mar 5",
    )
    xs = np.linspace(A[:, 1].min(), A[:, 1].max(), 50)
    styles = zip(aml_style.PALETTE[:3], aml_style.LINESTYLES[:3], strict=True)
    for (name, w), (c, ls) in zip(lines.items(), styles, strict=True):
        ax.plot(xs, w[0] + w[1] * xs, ls, color=c, lw=2.4, label=f"{name}: slope {w[1]:.2f}")
    ax.set_xlabel(f"{NEIGHBOR} pressure (m)")
    ax.set_ylabel(f"{SENSOR} pressure (m)")
    ax.legend(loc="upper left", fontsize=10, markerscale=2)
    aml_style.save(fig, out_dir, "burst_lines")
    return {
        "calm_x_range_m": [rnd(v, 2) for v in ends],
        "clean_line_at_ends_m": [rnd(w_clean[0] + w_clean[1] * v, 2) for v in ends],
    }


def bracket_figure(slopes: dict[str, float], clean: float, out_dir: Path) -> None:
    """Dot plot: each method's slope against the clean slope (the pre-burst fit)."""
    rows = [
        ("move 1:1 (no model)", 1.0, "o", aml_style.MUTED),
        ("least squares", slopes["ols"], "s", aml_style.PALETTE[0]),
        ("Huber", slopes["huber"], "^", aml_style.PALETTE[1]),
        ("RANSAC, $\\tau$ = 3$\\sigma$", slopes["ransac"], "D", aml_style.PALETTE[2]),
    ]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.axvline(clean, color=aml_style.MUTED, ls=":", lw=1.8)
    ax.text(clean + 0.01, -0.55, f"clean fit {clean:.2f}", color=aml_style.MUTED, fontsize=12)
    for i, (_, v, m, c) in enumerate(rows):
        ax.plot([clean, v], [i, i], color="#D9D9D9", lw=1.5, zorder=1)
        ax.plot([v], [i], m, color=c, markersize=11, zorder=2)
        ax.annotate(f"{v:.2f}", (v, i), xytext=(10, -8), textcoords="offset points", va="top",
                    fontsize=13)  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.9)
    ax.set_xlim(0.9, 1.85)
    ax.set_xlabel(f"slope of {SENSOR} on {NEIGHBOR}")
    aml_style.save(fig, out_dir, "bracket")


def dropouts(P: pd.DataFrame, tau: float) -> tuple[dict[str, object], dict[str, float]]:
    """Three readings of the neighbor recorded as 0 m (a fault) in the calm part of the window.

    Args:
        P: The 2018 pressures.
        tau: RANSAC's inlier threshold (m).

    Returns:
        The numbers (slopes, and the zeros' leverage, residuals and Huber weights), and the
        unrounded slopes, which :func:`dropout_figure` must reproduce.
    """
    A, y = design(P.loc[WINDOW[0] : PRE_BURST_END])
    ref = lstsq(A, y)
    Ad = A.copy()
    Ad[list(DROPOUTS), 1] = 0.0
    w_ols = lstsq(Ad, y)
    w_hub, _, scale = huber_irls(Ad, y)
    w_ran, _ = ransac_line(Ad, y, tau, trials=TRIALS, rng=np.random.default_rng(SEED))
    # Why Huber shrugs them off: each zero lies about mean(x) from the other readings in x, its
    # residual from Huber's line is huge, and IRLS gives it the weight t * scale / |r|.
    r0 = np.abs(y[list(DROPOUTS)] - Ad[list(DROPOUTS)] @ w_hub)
    wt0 = np.minimum(1.0, HUBER_T * scale / r0)
    numbers = {
        "readings": len(y),
        "faults": len(DROPOUTS),
        "slope_clean": rnd(ref[1], 4),
        "slope_ols": rnd(w_ols[1], 4),
        "slope_huber": rnd(w_hub[1], 4),
        "slope_ransac": rnd(w_ran[1], 4),
        "x_mean_m": rnd(A[:, 1].mean(), 2),  # each zero's distance from the others in x
        "zero_resid_m": [rnd(r0.min(), 2), rnd(r0.max(), 2)],
        "zero_weight_max": float(f"{wt0.max():.1e}"),
    }
    raw = {"clean": float(ref[1]), "ols": float(w_ols[1]), "huber": float(w_hub[1])}
    return numbers, raw


def breakdown(P: pd.DataFrame, tau: float, out_dir: Path) -> dict[str, object]:
    """Longer windows: the burst's share grows past one half; which line does RANSAC keep?

    Per window, also each group's own best line: RANSAC on the calm readings alone and on the
    burst readings alone, and how many of the group lie within tau of it. RANSAC keeps the
    group whose best line holds more readings, which is checked here.

    Args:
        P: The 2018 pressures.
        tau: RANSAC's inlier threshold (m).
        out_dir: The figure directory.

    Returns:
        One row per window.

    Raises:
        RuntimeError: If RANSAC keeps the line of the group whose best line holds fewer readings.
    """
    rows = []
    for end in LONGER:
        w = P.loc[WINDOW[0] : end]
        A, y = design(w)
        b = np.asarray(w.index >= BURST_START)
        w_ran, inl = ransac_line(A, y, tau, trials=TRIALS_LONG, rng=np.random.default_rng(SEED))
        w_ols = lstsq(A, y)
        r = y - A @ w_ran
        _, own_calm = ransac_line(A[~b], y[~b], tau, CONSENSUS_TRIALS, np.random.default_rng(SEED))
        _, own_burst = ransac_line(A[b], y[b], tau, CONSENSUS_TRIALS, np.random.default_rng(SEED))
        margin = int(own_calm.sum()) - int(own_burst.sum())
        kept_calm = bool(inl[~b].mean() > inl[b].mean())
        if kept_calm != (margin > 0):
            raise RuntimeError(f"window to {end}: RANSAC's choice disagrees with the counts")
        rows.append(
            {
                "end": end[:10],
                "burst_pct": round(100 * float(b.mean())),
                "slope_ols": rnd(w_ols[1], 4),
                "slope_ransac": rnd(w_ran[1], 4),
                "burst_inliers_pct": round(100 * float(inl[b].mean())),
                "calm_inliers_pct": round(100 * float(inl[~b].mean())),
                "calm_resid": round(float(r[~b].mean()), 2),
                "calm_readings": int((~b).sum()),
                "burst_readings": int(b.sum()),
                "calm_line_holds": int(own_calm.sum()),
                "burst_line_holds": int(own_burst.sum()),
                "calm_line_holds_pct": round(100 * float(own_calm.mean())),
                "burst_line_holds_pct": round(100 * float(own_burst.mean())),
                "calm_minus_burst": margin,
            }
        )
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    share = [r["burst_pct"] for r in rows]
    ax.plot(
        share,
        [r["slope_ols"] for r in rows],
        "s-",
        color=aml_style.PALETTE[0],
        label="least squares",
    )
    ax.plot(
        share,
        [r["slope_ransac"] for r in rows],
        "D--",
        color=aml_style.PALETTE[2],
        label="RANSAC, $\\tau$ = 3$\\sigma$",
    )
    ax.axvline(50, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(50.5, 1.27, "half", color=aml_style.MUTED, fontsize=12)
    ax.set_xlabel("burst readings in the window (%)")
    ax.set_ylabel(f"slope of {SENSOR} on {NEIGHBOR}")
    ax.legend(loc="center left", fontsize=12)
    aml_style.save(fig, out_dir, "breakdown")
    return {"rows": rows}


def alarm_counts(on: pd.Series) -> dict[str, object]:
    """When the alarm first sounds after the burst, and how often it sounds outside the burst."""
    start = pd.Timestamp(BURST_START)
    before = start - pd.Timedelta(minutes=5)
    first = on[start:].idxmax()
    assert on[first], "no alarm after the burst"
    week = on[start - pd.Timedelta(days=7) : before]
    assert len(week) == 7 * PER_DAY, len(week)
    return {
        "first_alarm": str(first),
        "minutes_after": int((first - start) / pd.Timedelta(minutes=1)),
        "readings_week_before": len(week),
        "false_alarms_week_before": int(week.sum()),
        "false_alarms_2018_before": int(on[:before].sum()),
        "alarms_after_repair": int(on[AFTER_REPAIR:].sum()),
    }


def alarm_miss(
    P: pd.DataFrame,
    K: pd.DataFrame,
    net: Network,
    *,
    r: pd.Series,
    hourly: pd.Series,
    thr: float,
) -> dict[str, object]:
    """The leak nearest n229 after the repair, and what the pair alarm saw of it.

    A pair alarm watches n229 against n636. A leak that lowers both moves the residual little.

    Args:
        P: The 2018 pressures.
        K: The 2018 leak flows (m³/h).
        net: The parsed network.
        r: n229's residual from the calm line, every reading of 2018.
        hourly: Its mean over the last hour.
        thr: The alarm threshold (m, negative).

    Returns:
        The leak's pipe, times, flow and distances; each sensor's mean drop over the
        ``DAYS_AROUND`` days after its start against the days before; the residual's mean before
        and during; its lowest hourly mean during the leak; and the alarms during it.

    Raises:
        RuntimeError: If the leak nearest n229 after the repair is not ``MISS_PIPE``.
    """
    after = K.loc[AFTER_REPAIR:]
    running = after.columns[(after > 0).any()].tolist()
    nearest = min(running, key=lambda p: net.distance(p, SENSOR))
    if nearest != MISS_PIPE:
        raise RuntimeError(f"nearest leak after the repair is {nearest}, not {MISS_PIPE}")
    flow = K[MISS_PIPE][K[MISS_PIPE] > 0]
    start, end = flow.index[0], flow.index[-1]
    days = pd.Timedelta(days=DAYS_AROUND)
    step = pd.Timedelta(minutes=5)
    before = P.loc[start - days : start - step, [SENSOR, NEIGHBOR]].mean()
    during = P.loc[start : start + days - step, [SENSOR, NEIGHBOR]].mean()
    first_full_hour = start + (PER_HOUR - 1) * step  # the first hourly mean wholly in the leak
    return {
        "pipe": MISS_PIPE,
        "start": str(start),
        "end": str(end),
        "flow_m3h": rnd(flow.mean(), 1),
        "sensor_m": int(round(net.distance(MISS_PIPE, SENSOR), -1)),
        "neighbor_m": int(round(net.distance(MISS_PIPE, NEIGHBOR), -1)),
        "sensor_drop_m": rnd(during[SENSOR] - before[SENSOR], 2),
        "neighbor_drop_m": rnd(during[NEIGHBOR] - before[NEIGHBOR], 2),
        "resid_before_m": rnd(r[start - days : start - step].mean(), 3),
        "resid_during_m": rnd(r[start:end].mean(), 3),
        "hourly_min_during_m": rnd(hourly[first_full_hour:end].min(), 3),
        "alarms_during": int((hourly[first_full_hour:end] < thr).sum()),
    }


def alarm(
    P: pd.DataFrame,
    K: pd.DataFrame,
    net: Network,
    *,
    w_cal: np.ndarray,
    w_window: np.ndarray,
    sigma: float,
    out_dir: Path,
) -> dict[str, object]:
    """Alarm when the hourly mean residual from the calm week's line falls below -5 sigma.

    The line is fitted once, on the trusted calm week (February 20-27), before the burst: a
    causal baseline for the burst and for every reading after it. RANSAC's line from the burst
    window (fitted through March 7) must give the same alarms, the check that the robust fit
    recovered the calm line. One residual series over all of 2018, so every hourly mean is over
    a full hour.

    Args:
        P: The 2018 pressures.
        K: The 2018 leak flows (m³/h).
        net: The parsed network.
        w_cal: The calm week's least-squares line.
        w_window: RANSAC's line on the burst window.
        sigma: The calm week's noise SD (m).
        out_dir: The figure directory.

    Returns:
        The alarm's numbers, with the leak it misses (:func:`alarm_miss`).
    """
    A, y = design(P)
    thr = -ALARM_SIGMA * sigma
    series = {}
    for name, w in (("calm", w_cal), ("window", w_window)):
        r = pd.Series(y - A @ w, index=P.index)
        hourly = r.rolling(PER_HOUR).mean()  # mean of the last 12 readings (1 hour)
        series[name] = (r, hourly, alarm_counts(hourly < thr))
    r, hourly, counts = series["calm"]
    assert counts == series["window"][2], (counts, series["window"][2])
    alarm_figure(
        r[ALARM_PLOT[0] : ALARM_PLOT[1]], hourly[ALARM_PLOT[0] : ALARM_PLOT[1]], thr, out_dir
    )
    return {
        **counts,
        "threshold_m": rnd(-thr, 3),
        "slope_line": rnd(w_cal[1], 4),
        # One reading in this many falls below -5 sigma when the noise is normal.
        "one_in_millions": rnd(1 / norm.cdf(-ALARM_SIGMA) / 1e6, 1),
        "miss": alarm_miss(P, K, net, r=r, hourly=hourly, thr=thr),
    }


def alarm_figure(r: pd.Series, hourly: pd.Series, thr: float, out_dir: Path) -> None:
    """Residual every 5 minutes, its hourly mean, and the alarm threshold, around the burst."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(r.index, r, color="#B0B0B0", lw=0.8, label="residual, every 5 min")
    ax.plot(
        hourly.index, hourly, "-", color=aml_style.PALETTE[0], lw=2.0, label="mean of the last hour"
    )
    ax.axhline(
        thr,
        color=aml_style.PALETTE[1],
        ls="--",
        lw=2,
        label=f"alarm: $-5\\sigma$ = {tex_num(thr, '.2f')} m",
    )
    ax.axvline(pd.Timestamp(BURST_START), color=aml_style.PALETTE[3], ls=":", lw=2)
    ax.text(
        pd.Timestamp(BURST_START), 0.12, " burst, 15:45", color=aml_style.PALETTE[3], fontsize=13
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel(f"{SENSOR} below its line (m)")
    ax.set_ylim(-1.0, 0.25)
    # In the gap between the alarm line and the residuals after the burst.
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=11)
    aml_style.save(fig, out_dir, "alarm")


def isolation(
    P: pd.DataFrame, K: pd.DataFrame, F: pd.DataFrame, leaks: pd.DataFrame
) -> dict[str, object]:
    """Isolation forest on hourly residual vectors of all 33 sensors, trained on the calm week.

    Beside its alarms by month, what changed in the network by month: the town's inflow and the
    number of leaks running. Both change together, so the counts cannot say which one the
    forest flags.

    Args:
        P: The 2018 pressures.
        K: The 2018 leak flows (m³/h).
        F: The 2018 SCADA flows.
        leaks: :func:`leak_table`.

    Returns:
        The share of hours flagged (the days before the burst; each month), the monthly inflow
        and leaks running, and the leaks that started or were still growing before the burst.
    """
    cal = P.loc[CAL[0] : CAL[1]]
    R = pd.DataFrame(index=P.index)
    for c in P.columns:
        others = [o for o in P.columns if o != c]
        m = LinearRegression().fit(cal[others], cal[c])
        R[c] = P[c] - m.predict(P[others])
    H = R.resample("h").mean()
    Hcal = H.loc[CAL[0] : CAL_LAST_HOUR]
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(Hcal)
    thr = np.quantile(-iso.score_samples(Hcal), 0.99)
    on = pd.Series(-iso.score_samples(H) > thr, index=H.index)
    monthly = on.resample("ME").mean()
    inflow = town_inflow(F)
    inflow_month = inflow.resample("ME").mean()
    inflow_cal = float(inflow[CAL[0] : CAL[1]].mean())
    running = (K > 0).resample("ME").max().sum(axis=1)
    late = slice("2018-05", None)  # May to December, where the forest flags most hours
    above = 100 * (inflow_month.loc[late] / inflow_cal - 1)
    # The days before the burst: leaks that started in them, and incipient leaks still growing.
    t0, t1 = pd.Timestamp(WINDOW[0]), pd.Timestamp(PRE_BURST_LAST_HOUR)
    started = leaks[(leaks["start"] >= t0) & (leaks["start"] <= t1)]
    growing = leaks[
        (leaks["kind"] == "incipient")
        & (leaks["start"] < t0)
        & (leaks["end"] > t0)
        & (leaks["peak"] > t1)
    ]
    return {
        "alarm_pct_next_week": round(100 * float(on.loc[WINDOW[0] : PRE_BURST_LAST_HOUR].mean())),
        "alarm_pct_by_month": {str(k.month): round(100 * float(v)) for k, v in monthly.items()},
        "alarm_pct_may_dec_min": round(100 * float(monthly.loc[late].min())),
        "alarm_pct_may_dec_max": round(100 * float(monthly.loc[late].max())),
        "inflow_cal_m3h": rnd(inflow_cal, 1),
        "inflow_by_month": {str(k.month): rnd(v, 1) for k, v in inflow_month.items()},
        "inflow_above_cal_pct_may_dec": [round(float(above.min())), round(float(above.max()))],
        "leaks_by_month": {str(k.month): int(v) for k, v in running.items()},
        "leaks_may_dec": [int(running.loc[late].min()), int(running.loc[late].max())],
        "leaks_started_next_week": len(started),
        "leaks_growing_next_week": sorted(growing["pipe"].tolist()),
    }


def trace(
    P: pd.DataFrame, w: np.ndarray, sigma: float
) -> tuple[dict[str, object], dict[str, float]]:
    """One reading, 17:00 on March 5, through the RANSAC line.

    The line's intercept lies far from the data (x near 45 m), so the diagram shows w0 to 3
    decimals and w1 to 4: enough for w0 + w1 x to reproduce y-hat to the centimeter.

    Args:
        P: The 2018 pressures.
        w: RANSAC's line on the burst window.
        sigma: The calm week's noise SD (m).

    Returns:
        The values as the slides quote them, and the unrounded ones for the diagram.

    Raises:
        RuntimeError: If the diagram's rounded terms do not reproduce y-hat to the centimeter.
    """
    t = pd.Timestamp(TRACE_TIME)
    x, y = float(P.loc[t, NEIGHBOR]), float(P.loc[t, SENSOR])
    pred = float(w[0] + w[1] * x)
    shown = round(float(w[0]), 3) + round(float(w[1]), 4) * round(x, 2)
    if round(shown, 2) != round(pred, 2):
        raise RuntimeError(f"the diagram's terms give {shown:.4f}, not {pred:.4f}")
    raw = {
        "x": x,
        "y": y,
        "w0": float(w[0]),
        "w1": float(w[1]),
        "pred": pred,
        "resid": y - pred,
        "in_sigma": (y - pred) / sigma,
    }
    numbers = {
        "time": str(t),
        "minutes_after_burst": int((t - pd.Timestamp(BURST_START)) / pd.Timedelta(minutes=1)),
        "x": rnd(x, 2),
        "y": rnd(y, 2),
        "w0": rnd(raw["w0"], 3),
        "w1": rnd(raw["w1"], 4),
        "pred": rnd(pred, 2),
        "resid": rnd(y - pred, 2),
        "in_sigma": round(raw["in_sigma"]),
    }
    return numbers, raw


def write_tex(
    raw: dict[str, float],
    sigma: float,
    *,
    window_readings: int,
    inliers: int,
    m_from_burst: float,
    out_dir: Path,
) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, from unrounded values.

    Args:
        raw: :func:`trace`'s unrounded values.
        sigma: The calm week's noise SD (m); tau is ``K_SIGMA`` times it.
        window_readings: Readings in the fitting window.
        inliers: RANSAC's inliers among them.
        m_from_burst: Distance from the sensor to the burst (m).
        out_dir: The figure directory.

    Returns:
        The path written.
    """
    macros = {
        "NWin": tex_num(window_readings, ","),
        "TauM": tex_num(K_SIGMA * sigma, ".2f"),
        "SigmaM": tex_num(sigma, ".3f"),
        "XNb": tex_num(raw["x"], ".2f"),
        "YSn": tex_num(raw["y"], ".2f"),
        "WZero": tex_num(raw["w0"], ".3f"),
        "WOne": tex_num(raw["w1"], ".4f"),
        "Pred": tex_num(raw["pred"], ".2f"),
        "Resid": tex_num(raw["resid"], ".2f"),
        "InSigma": tex_num(raw["in_sigma"], ".0f"),
        "Inliers": tex_num(inliers, ","),
        "Dist": tex_num(round(m_from_burst, -1), ".0f"),  # to the nearest 10 m, once
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


def leak_kinds(leaks: pd.DataFrame, year: int = 2018) -> dict[str, object]:
    """How the year's leaks began, from the dataset's configuration (BattLeDIM ground truth).

    Args:
        leaks: :func:`leak_table`.
        year: The year whose leaks (by start time) are counted.

    Returns:
        Counts by kind and the incipient leaks' growth time in days (start to peak).
    """
    leaks = leaks[leaks["start"].dt.year == year]
    grow = leaks[leaks["kind"] == "incipient"]
    days = (grow["peak"] - grow["start"]).dt.total_seconds() / 86400
    return {
        "leaks": len(leaks),
        "incipient": len(grow),
        "growth_days_min": round(float(days.min())),
        "growth_days_median": round(float(days.median())),
        "growth_days_max": round(float(days.max())),
    }


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

BUILD_TRIALS: int = 2  # RANSAC trials drawn before the best one
SURFACE_N: int = 61  # grid points per side of the dropout loss surfaces


def title_art_figure(P: pd.DataFrame, w_clean: np.ndarray, out_dir: Path) -> None:
    """The title art: points along a golden line, a few crimson strays, a pressure drop below.

    The horizon is n229's hourly residual from the calm line, March 1 to 10: flat, then the
    burst's drop on March 5.
    """
    A, y = design(P.loc[TITLE_SPAN[0] : TITLE_SPAN[1]])
    r = pd.Series(y - A @ w_clean).rolling(PER_HOUR).mean().to_numpy()[PER_HOUR::PER_HOUR]
    c = title_art.Canvas(seed=9)
    c.ground(("#0b1418", "#101420", "#190c14", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.92, 0.66, 4), soft=4), "#4a1c14", 0.85)
    rim = c.disc((0.76, 1.18), 0.12, soft=4) - c.disc((0.76, 1.18), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.76, 1.18), 0.10, soft=3), "#1d0a10", 0.92)
    rng = np.random.default_rng(9)
    t = rng.uniform(0.1, 0.9, 50)
    line = np.column_stack([t, 0.80 + 0.30 * (t - 0.1) + rng.normal(0, 0.012, 50)])
    c.glow(c.lines([[(0.06, 0.788), (0.94, 1.052)]], 3, soft=1), "#e9c27a", 0.5, spread=8)
    c.glow(np.clip(sum(c.disc(tuple(q), 0.006, soft=1.5) for q in line), 0, 1), "#9fb7c9", 0.55,
           spread=6)  # fmt: skip
    strays = [(0.30, 0.66), (0.55, 0.70), (0.62, 1.22), (0.20, 1.02)]
    c.glow(np.clip(sum(c.disc(q, 0.009, soft=1.5) for q in strays), 0, 1), "#e0452a", 0.9,
           spread=10)  # fmt: skip
    xs = np.linspace(0.06, 0.94, len(r))
    ys = 0.16 + 0.30 * (r - np.nanmin(r)) / np.nanmax(r - np.nanmin(r))
    c.glow(c.lines([list(zip(xs, ys, strict=True))], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean form in rust; above it pale points "
        "along a faint golden line, with four crimson points astray; a dark sun with a burning "
        "rim at the upper right, and a golden line along the foot that runs level, then drops: "
        "the pressure of the sensor nearest the burst.",
    )


def network_figure(net: Network, P: pd.DataFrame, out_dir: Path) -> None:
    """L-Town: the pipes, the 33 pressure sensors, the burst pipe, and the two sensors used."""
    segs = [[net.coords[a], net.coords[b]] for a, b in net.pipes.values()
            if a in net.coords and b in net.coords]  # fmt: skip
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.add_collection(LineCollection(segs, colors="#BBBBBB", linewidths=0.5))
    sx = [net.coords[n][0] for n in P.columns]
    sy = [net.coords[n][1] for n in P.columns]
    ax.scatter(sx, sy, s=14, color=aml_style.PALETTE[0], zorder=3, label="pressure sensors")
    a, b = net.pipes[BURST_PIPE]
    mid = np.mean([net.coords[a], net.coords[b]], axis=0)
    ax.plot(*mid, "*", ms=20, color=aml_style.PALETTE[1], mec="white", zorder=5,
            label=f"burst, pipe {BURST_PIPE}")  # fmt: skip
    for n, dx in ((SENSOR, 1), (NEIGHBOR, -1)):
        x, y = net.coords[n]
        ax.plot(x, y, "o", ms=9, mfc="none", mec=aml_style.INK, mew=1.6, zorder=4)
        ax.annotate(n, (x, y), xytext=(10 * dx, 6), textcoords="offset points", fontsize=12,
                    ha="left" if dx > 0 else "right", fontweight="bold")  # fmt: skip
    ax.autoscale()
    ax.set(aspect="equal", xticks=[], yticks=[])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=11, markerscale=0.8)
    aml_style.save(fig, out_dir, "network")


def losses_figure(out_dir: Path) -> None:
    """Squared, Huber, and absolute losses, and the weight IRLS gives each residual."""
    r = np.linspace(-4, 4, 400)
    d = 1.0
    huber = np.where(np.abs(r) <= d, r**2 / 2, d * np.abs(r) - d**2 / 2)
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    ax = axes[0]
    ax.plot(r, r**2 / 2, color=aml_style.MUTED, lw=2.2, ls=":", label=r"squared: $r^2/2$")
    ax.plot(r, huber, color=aml_style.PALETTE[0], lw=2.6, ls="-", label=r"Huber, $\delta$ = 1")
    ax.plot(r, np.abs(r), color=aml_style.PALETTE[1], lw=2.2, ls="--", label=r"absolute: $|r|$")
    ax.set(xlabel=r"residual $r$ (in units of $\delta$)", ylabel=r"loss $\rho(r)$", ylim=(0, 6))
    ax.set_title("the loss", loc="left")
    ax.legend(loc="upper center", fontsize=12)
    ax = axes[1]
    ax.plot(r, np.ones_like(r), color=aml_style.MUTED, lw=2.2, ls=":", label="squared: 1")
    ax.plot(r, np.minimum(1, d / np.maximum(np.abs(r), 1e-9)), color=aml_style.PALETTE[0],
            lw=2.6, ls="-", label=r"Huber: $\min(1, \delta/|r|)$")  # fmt: skip
    ax.set(xlabel=r"residual $r$ (in units of $\delta$)", ylabel="weight in IRLS", ylim=(0, 1.15))
    ax.set_title("each reading's weight", loc="left")
    ax.legend(loc="lower center", fontsize=12)
    fig.subplots_adjust(wspace=0.3)
    aml_style.save(fig, out_dir, "losses")


def dropout_surfaces(P: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The calm readings with three dropouts: centred design, responses, and the clean line."""
    A, y = design(P.loc[WINDOW[0] : PRE_BURST_END])
    xbar = float(A[:, 1].mean())
    clean = lstsq(np.column_stack([np.ones(len(y)), A[:, 1] - xbar]), y)
    Ad = A.copy()
    Ad[list(DROPOUTS), 1] = 0.0
    return np.column_stack([np.ones(len(y)), Ad[:, 1] - xbar]), y, clean


def dropout_figure(P: pd.DataFrame, slopes: Mapping[str, float], out_dir: Path) -> None:
    """The loss over (level, slope) with three dropouts: least squares' bowl moves, Huber's not.

    The design is centred on the neighbor's mean, so the level and the slope are nearly
    independent and each bowl sits square to the axes. Huber's scale is held at its fit's.

    Args:
        P: The 2018 pressures.
        slopes: :func:`dropouts`' unrounded slopes (``clean``, ``ols``, ``huber``), which the
            centred fits here must reproduce.
        out_dir: The figure directory.

    Raises:
        RuntimeError: If a centred fit's slope differs from :func:`dropouts`' slope.
    """
    A, y, clean = dropout_surfaces(P)
    w_ols = lstsq(A, y)
    w_hub, _, scale = huber_irls(A, y)
    delta = HUBER_T * scale
    a = np.linspace(clean[0] - 0.4, clean[0] + 0.4, SURFACE_N)
    b = np.linspace(-0.4, 1.8, SURFACE_N)
    Ag, Bg = np.meshgrid(a, b)
    R = y[:, None] - A @ np.stack([Ag.ravel(), Bg.ravel()])  # residuals at each grid point
    ls = (R**2).mean(axis=0).reshape(Ag.shape)
    ar = np.abs(R)
    hub = np.where(ar <= delta, R**2 / 2, delta * ar - delta**2 / 2).mean(axis=0)
    hub = hub.reshape(Ag.shape)
    for name, w in (("ols", w_ols), ("huber", w_hub), ("clean", clean)):
        if abs(float(w[1]) - slopes[name]) > SAME:
            raise RuntimeError(f"{name}: surface slope {w[1]} against {slopes[name]}")
    fig = plt.figure(figsize=aml_style.FULL)
    for k, (L, w, title) in enumerate(((ls, w_ols, "least squares"), (hub, w_hub, "Huber"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        Z = L / L.min()  # in multiples of the bottom: both bowls on one scale
        aml_style.surface3d(ax, Ag, Bg, np.minimum(Z, 6), cmap="cividis", vmin=1, vmax=6)
        ax.plot([clean[0]] * 2, [clean[1]] * 2, [1, 6], color=aml_style.PALETTE[2], lw=2.4,
                zorder=6)  # fmt: skip
        ax.scatter([w[0]], [w[1]], [1], s=70, color="white", edgecolor=aml_style.INK, zorder=7)
        ax.set_xlabel("level (m)", labelpad=8, fontsize=12)
        ax.set_ylabel("slope", labelpad=8, fontsize=12)
        ax.set(zlim=(1, 6), zticks=[1, 3, 5])
        ax.set_title(f"{title}: slope {w[1]:.2f}", loc="left", fontsize=13)
        aml_style.style3d(ax, elev=28, azim=-60)
    fig.text(0.5, -0.06, f"loss over {len(y):,} calm readings with {len(DROPOUTS)} dropouts, "
             f"relative to its minimum; green: the clean slope, {clean[1]:.2f}", ha="center",
             fontsize=11.5, color=aml_style.MUTED)  # fmt: skip
    aml_style.save(fig, out_dir, "dropout_surface")


def ransac_build_figure(
    win: Window, tau: float, w_best: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """RANSAC, trial by trial, on the burst window: two random pairs, then the best consensus."""
    A, y = win.A, win.y
    x = A[:, 1]
    rng = np.random.default_rng((SEED, 9))
    fig, axes = plt.subplots(1, BUILD_TRIALS + 1, figsize=aml_style.FULL, sharex=True, sharey=True)
    grid = np.linspace(x.min(), x.max(), 2)
    counts = []
    for k, ax in enumerate(axes):
        if k < BUILD_TRIALS:
            i, j = rng.choice(len(y), size=2, replace=False)
            w = np.linalg.solve(A[[i, j]], y[[i, j]])
            pair = [i, j]
            title = f"trial {k + 1}"
        else:
            w, pair, title = w_best, [], f"best of {TRIALS}, refit"
        inl = np.abs(y - A @ w) < tau
        counts.append(int(inl.sum()))
        ax.scatter(x[~inl], y[~inl], s=4, color=aml_style.MUTED, alpha=0.35, lw=0)
        ax.scatter(x[inl], y[inl], s=6, color=aml_style.PALETTE[0], alpha=0.8, lw=0)
        ax.plot(grid, w[0] + w[1] * grid, color=aml_style.PALETTE[1], lw=2.2, ls="-")
        if pair:
            ax.plot(x[pair], y[pair], "o", ms=11, mfc=aml_style.PALETTE[1], mec=aml_style.INK)
        ax.set_title(f"{title}: {inl.sum():,} inliers", loc="left", fontsize=12.5)
        ax.set_xlabel(f"{NEIGHBOR} (m)")
    axes[0].set_ylabel(f"{SENSOR} (m)")
    aml_style.save(fig, out_dir, "ransac_build")
    return {"inliers": counts}


def main() -> None:
    """Generate every figure and number for CEE Lecture 9."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    P = read("2018_SCADA_Pressures.csv")
    K, F = read("2018_Leakages.csv"), read("2018_SCADA_Flows.csv")
    assert P.shape == (105120, 33) and not P.isna().any().any(), P.shape
    net = read_network()

    calm, sigma, w_cal = calm_week(P, K, net)
    tau = K_SIGMA * sigma
    win = fitting_window(P)
    clean, w_clean = clean_fit(win)
    fit, w, inliers = robust_fits(win, tau, w_clean)
    seeds = ransac_seeds(win, tau, float(w_clean[1]), out_dir)
    lines = burst_lines_figure(
        win,
        {"least squares": w["ols"], "Huber": w["huber"], "RANSAC": w["ransac"]},
        w_clean,
        out_dir,
    )
    bracket_figure({k: float(v[1]) for k, v in w.items()}, float(w_clean[1]), out_dir)
    tr, raw = trace(P, w["ransac"], sigma)
    drop, drop_slopes = dropouts(P, tau)
    leaks = leak_table()

    numbers = {
        "network": network_summary(P, K, F, net),
        "attempt": attempt(P, F, float(w_clean[1])),
        "fits": {**calm, **clean, **fit, **seeds},
        "burst_lines": lines,
        "dropouts": drop,
        "breakdown": breakdown(P, tau, out_dir),
        "alarm": alarm(P, K, net, w_cal=w_cal, w_window=w["ransac"], sigma=sigma, out_dir=out_dir),
        "isolation": isolation(P, K, F, leaks),
        "leak_kinds": leak_kinds(leaks),
        "trace": tr,
    }
    # the visuals that carry the slides
    title_art_figure(P, w_clean, out_dir)
    network_figure(net, P, out_dir)
    losses_figure(out_dir)
    dropout_figure(P, drop_slopes, out_dir)  # checks its slopes against `dropouts`
    numbers["ransac_build"] = ransac_build_figure(win, tau, w["ransac"], out_dir)
    aml_style.thumbnails(out_dir, ("losses", "dropout_surface", "burst_lines", "breakdown"))
    write_numbers(numbers, out_dir)
    m_from_burst = net.distance(BURST_PIPE, SENSOR)
    print(
        write_tex(
            raw,
            sigma,
            window_readings=len(win.y),
            inliers=int(inliers.sum()),
            m_from_burst=m_from_burst,
            out_dir=out_dir,
        )
    )


if __name__ == "__main__":
    main()
