"""Figures and numbers for CEE Lecture 24: equations and operators learned from data.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l24_learning_dynamics.py \
        --out build/fig/cee_l24_learning_dynamics

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

Two questions. A detention pond's outlet is unknown (an orifice or a weir): from a year of inflow
and level records, find its law, and say whether the pond holds a design storm twice the
largest recorded. And a river reach routes floods: can a network learn the routing (a function
to a function) well enough, and fast enough, to stand in for the solver?

- Pond (assumed values, simulated records): A dh/dt = Q_in - C h^1.5 (a weir), level read every
  5 minutes with 5 mm of noise; 20 storms in year 1. SINDy (Brunton et al., 2016): regress dh/dt
  (central differences) on a library of candidate terms by sequentially thresholded least
  squares. Against a fit that assumes an orifice (sqrt h). Year 2: the weir partly blocked.
- Real rivers: recession days of the home basin (CAMELS; Lecture 20): -dQ/dt against Q, SINDy
  against the one-term power law of recession analysis (Brutsaert & Nieber, 1977).
- River reach (assumed values): a 60-km channel routed by variable-parameter Muskingum-Cunge
  (Cunge, 1969; Ponce & Yevjevich, 1978); 1,000 random inflow floods for training. A Fourier
  neural operator (Li et al., 2021) written from scratch, three seeds, against the unit
  hydrograph (Lecture 20's linear convolution, here from inflow to outflow and fitted by plain
  least squares where Lecture 20 used ridge on rain; its length chosen on validation over every
  length up to the record). Floods sampled every 30 minutes are routed with the same 900-s
  internal step as the hourly ones, so only the sampling of the input changes; the operator is
  also scored on the same floods sampled hourly, so its error on each sampling is measured on
  the same floods.
- Break it: a library without the true term; 5 cm of level noise; floods twice as large as any
  in training; the operator's water balance, in absolute terms and against the solver's flood by
  flood.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from torch import nn

import aml_style
import camels
import title_art
from common import parse_out_dir, rmse, seed_all, write_numbers

SEED = 24
# The pond (assumed values)
AREA = 2000.0  # m2, vertical walls
WEIR_C = 1.7  # m^0.5/s: Q_out = C h^1.5 (a 1-m broad weir, C_w = 1.7)
DEPTH = 2.5  # m from the outlet's crest to the rim
LEVEL_NOISE = 0.005  # m
READ_MIN = 5  # minutes between level readings
STORM_HOURS = 24
N_STORMS = 20
STORM_PEAK = (0.5, 3.0)  # m3/s
DESIGN = {"peak": 6.0, "tp_hours": 2.5, "shape": 3.0}  # twice the largest recorded peak
CLOG = 0.7  # year 2: the weir passes 70% of its year-1 flow at the same level
NAMES = ("Q_in", "1", "h^0.5", "h", "h^1.5", "h^2")
TRUE_TERM = "h^1.5"
LAMBDA = 0.05  # STLSQ threshold, a share of the largest scaled coefficient (fixed)
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3)  # the sweep shown on a slide
NOISES = (0.005, 0.02, 0.05)  # m, for the noise reveal
BOOT = 200  # bootstrap resamples of the storms
N_VAL_STORMS = 10
# The river reach (assumed values)
WIDTH_M, SLOPE, MANNING = 50.0, 5e-4, 0.035
REACH, LENGTH, SUB = 2000.0, 60_000.0, 4  # m per reach, m, sub-steps per output step
STEP_S = 3600 / SUB  # s: the solver's internal step for hourly floods
FINE_SUB = SUB // 2  # sub-steps for half-hourly floods: the same STEP_S
BASE = 50.0  # m3/s
HOURS = 72
FLOOD_PEAK, BIG_PEAK = (100.0, 600.0), (600.0, 1200.0)  # m3/s
N_TRAIN, N_VAL, N_TEST, N_BIG, N_FINE = 1000, 200, 500, 500, 200
Q_SCALE = 500.0  # m3/s: the network sees (Q - BASE) / Q_SCALE
# The operator (fixed before any test; Li et al., 2021, 1-D settings, smaller)
FNO_WIDTH, FNO_MODES, FNO_LAYERS = 32, 12, 4
EPOCHS, BATCH, LR = 300, 32, 1e-3
FNO_SEEDS = 3
THREADS, WORKERS = 4, 3
UH_LADDER = (6, 12, 24, 36, 48, 60, 72)  # hours of memory, up to the whole record
# Recession days (CAMELS)
RAIN_MM = 1.0  # a recession day has no more rain than this today and the two days before
REC_NAMES = ("1", "Q^0.5", "Q", "Q^1.5", "Q^2")

# --------------------------------------------------------------------------- the pond


def storm(t: np.ndarray, peak: float, tp: float, shape: float) -> np.ndarray:
    """A gamma-shaped inflow hydrograph.

    Args:
        t: Times, s.
        peak: Peak inflow, m3/s.
        tp: Time of the peak, s.
        shape: Gamma shape: larger is narrower.

    Returns:
        Inflow at ``t``, m3/s.
    """
    return peak * (t / tp) ** shape * np.exp(shape * (1 - t / tp))


def storms(n: int, rng: np.random.Generator, peak: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Random storms over STORM_HOURS, read every READ_MIN minutes.

    Args:
        n: Number of storms.
        rng: Generator for the peaks, peak times, and shapes.
        peak: Range of the peak inflow, m3/s.

    Returns:
        Times (s), and the inflows (n, times) in m3/s.
    """
    t = np.arange(0, STORM_HOURS * 60 + READ_MIN, READ_MIN) * 60.0
    q = [
        storm(t, rng.uniform(*peak), rng.uniform(1, 4) * 3600, rng.uniform(2, 5)) for _ in range(n)
    ]
    return t, np.array(q)


def pond_level(t: np.ndarray, q_in: np.ndarray, rhs: Callable[[float, float], float]) -> np.ndarray:
    """Integrate dh/dt = rhs(Q_in, h) from an empty pond (h = 0 at the crest).

    Args:
        t: Times, s.
        q_in: Inflow at ``t``, m3/s (linearly interpolated between them).
        rhs: dh/dt (m/s) from the inflow and the level.

    Returns:
        The level above the outlet at ``t``, m.

    Raises:
        RuntimeError: If the integration fails.
    """
    sol = solve_ivp(lambda tt, h: [rhs(float(np.interp(tt, t, q_in)), h[0])], (t[0], t[-1]),
                    [0.0], t_eval=t, rtol=1e-9, atol=1e-12, max_step=60.0)  # fmt: skip
    if not sol.success or sol.y.shape[1] != len(t):
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y[0]


def weir(c: float) -> Callable[[float, float], float]:
    """The true pond: A dh/dt = Q_in - c h^1.5.

    Args:
        c: Weir coefficient, m^0.5/s.

    Returns:
        dh/dt (m/s) as a function of the inflow (m3/s) and the level (m).
    """
    return lambda q, h: (q - c * max(h, 0.0) ** 1.5) / AREA


def library(q_in: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The candidate terms.

    Args:
        q_in: Inflow, (n,), m3/s.
        h: Level, (n,), m (negative values count as 0).

    Returns:
        One column per name in NAMES, (n, len(NAMES)).
    """
    hp = np.maximum(h, 0.0)
    return np.column_stack([q_in, np.ones_like(hp), hp**0.5, hp, hp**1.5, hp**2])


def stlsq(
    theta: np.ndarray, y: np.ndarray, lam: float, keep: np.ndarray | None = None, iters: int = 10
) -> np.ndarray:
    """Sequentially thresholded least squares (Brunton et al., 2016).

    Columns are scaled to unit norm; a coefficient below ``lam`` times the largest is set to
    zero and the rest refitted, ``iters`` times.

    Args:
        theta: The library, (n, terms).
        y: The derivative to explain, (n,).
        lam: Threshold, a share of the largest scaled coefficient; 0 is least squares.
        keep: Mask of the columns allowed at all (default: every column).
        iters: Rounds of thresholding and refitting.

    Returns:
        The coefficients in the library's own units, (terms,), zero where dropped.
    """  # fmt: skip
    keep = np.ones(theta.shape[1], bool) if keep is None else keep
    scale = np.linalg.norm(theta, axis=0)
    th = theta / scale
    xi = np.zeros(theta.shape[1])
    xi[keep] = np.linalg.lstsq(th[:, keep], y, rcond=None)[0]
    for _ in range(iters):
        active = keep & (np.abs(xi) >= lam * np.abs(xi).max())
        xi = np.zeros_like(xi)
        xi[active] = np.linalg.lstsq(th[:, active], y, rcond=None)[0]
    return xi / scale


def stack(t: np.ndarray, q: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The library rows and dh/dt (central differences) of every storm, stacked.

    Args:
        t: Times, s.
        q: Inflows, (storms, times), m3/s.
        h: Levels, (storms, times), m.

    Returns:
        The library (storms * times, terms) and dh/dt (storms * times,), m/s.
    """
    rows = np.vstack([library(qi, hi) for qi, hi in zip(q, h, strict=True)])
    dhdt = np.concatenate([np.gradient(hi, t) for hi in h])
    return rows, dhdt


def model_rhs(xi: np.ndarray) -> Callable[[float, float], float]:
    """dh/dt of a fitted model.

    Args:
        xi: Coefficients of the NAMES terms.

    Returns:
        dh/dt (m/s) as a function of the inflow (m3/s) and the level (m).
    """
    return lambda q, h: float((library(np.array([q]), np.array([h])) @ xi)[0])


def terms(xi: np.ndarray) -> dict[str, float]:
    """The kept terms of a model.

    Args:
        xi: Coefficients of the NAMES terms.

    Returns:
        The non-zero coefficients by term name.
    """
    return {n: float(x) for n, x in zip(NAMES, xi, strict=True) if x != 0}


def design_storm() -> tuple[np.ndarray, np.ndarray]:
    """The design storm, DESIGN.

    Returns:
        Times (s) and inflow (m3/s).
    """
    td = np.arange(0, STORM_HOURS * 60 + READ_MIN, READ_MIN) * 60.0
    return td, storm(td, DESIGN["peak"], DESIGN["tp_hours"] * 3600, DESIGN["shape"])


def pond_checks(t: np.ndarray, q: np.ndarray, h_clean: np.ndarray) -> dict[str, float]:
    """Exact derivatives recover the truth; a threshold of 0 is least squares.

    Args:
        t: Times, s.
        q: Year 1's inflows, (storms, times), m3/s.
        h_clean: Their levels without noise, m.

    Returns:
        The largest coefficient error relative to the largest true coefficient, the terms
        kept, and the gap between a threshold of 0 and least squares (relative).
    """
    truth = np.array([1 / AREA, 0, 0, 0, -WEIR_C / AREA, 0])
    rows = np.vstack([library(qi, hi) for qi, hi in zip(q, h_clean, strict=True)])
    rhs = weir(WEIR_C)
    exact = np.array([rhs(a, b) for a, b in zip(q.ravel(), h_clean.ravel(), strict=True)])
    xi = stlsq(rows, exact, LAMBDA)
    rows_n, y = stack(t, q, h_clean)
    ls = np.linalg.lstsq(rows_n, y, rcond=None)[0]
    return {
        "exact_rel": float(np.abs(xi - truth).max() / np.abs(truth).max()),
        "exact_terms": int((xi != 0).sum()),
        "lam0_rel": float(np.abs(stlsq(rows_n, y, 0.0) - ls).max() / np.abs(ls).max()),
    }


def clogging(t: np.ndarray, q: np.ndarray, h1: np.ndarray, xi1: np.ndarray) -> dict[str, object]:
    """Year 2 with the weir partly blocked: the h^1.5 coefficient's ratio, bootstrapped.

    Args:
        t: Times, s.
        q: The storms' inflows (the same storms both years), m3/s.
        h1: Year 1's noisy levels, m.
        xi1: SINDy's year-1 coefficients.

    Returns:
        Year 2's terms, the ratio of the h^1.5 coefficients, its 90% bootstrap interval over
        storms, the resamples that dropped the term, and the true ratio.
    """
    h2 = np.array([pond_level(t, qi, weir(CLOG * WEIR_C)) for qi in q])
    h2 = h2 + np.random.default_rng([SEED, 3]).normal(0, LEVEL_NOISE, h2.shape)
    xi2 = stlsq(*stack(t, q, h2), LAMBDA)
    k = NAMES.index(TRUE_TERM)
    rng = np.random.default_rng([SEED, 4])
    ratios = []
    for _ in range(BOOT):
        pick = rng.integers(0, N_STORMS, N_STORMS)
        a = stlsq(*stack(t, q[pick], h1[pick]), LAMBDA)
        b = stlsq(*stack(t, q[pick], h2[pick]), LAMBDA)
        ratios.append(b[k] / a[k] if a[k] != 0 and b[k] != 0 else np.nan)
    ratios = np.array(ratios)
    return {
        "year2": terms(xi2),
        "ratio": float(xi2[k] / xi1[k]),
        "ci90": np.nanpercentile(ratios, [5, 95]).tolist(),
        "boot_missing": int(np.isnan(ratios).sum()),
        "true": CLOG,
    }


def noise_runs(t: np.ndarray, q: np.ndarray, h_clean: np.ndarray) -> list[dict[str, object]]:
    """SINDy at more level noise: the terms kept and the design storm's peak.

    Args:
        t: Times, s.
        q: Year 1's inflows, m3/s.
        h_clean: Their levels without noise, m.

    Returns:
        Per noise level in NOISES: the terms kept, their count, and the design storm's peak (m).
    """
    td, qd = design_storm()
    out = []
    for k, sd in enumerate(NOISES):
        hn = h_clean + np.random.default_rng([SEED, 10 + k]).normal(0, sd, h_clean.shape)
        xn = stlsq(*stack(t, q, hn), LAMBDA)
        peak = float(pond_level(td, qd, model_rhs(xn)).max())
        out.append(
            {"sd_m": sd, "terms": terms(xn), "n_terms": int((xn != 0).sum()), "design_peak": peak}
        )
    return out


def pond_records() -> dict[str, np.ndarray]:
    """Year 1's storms and levels, and the held-out storms.

    Returns:
        ``t`` (s); ``q`` year 1's inflows (storms, times) in m3/s; ``h_clean`` their levels and
        ``h`` the same read with LEVEL_NOISE, in m; ``qv`` and ``hv`` the held-out storms and their
        true levels; ``hv_read`` those levels as the gauge reads them (LEVEL_NOISE added).
    """
    rng = np.random.default_rng([SEED, 1])
    t, q = storms(N_STORMS, rng, STORM_PEAK)
    h_clean = np.array([pond_level(t, qi, weir(WEIR_C)) for qi in q])
    h = h_clean + rng.normal(0, LEVEL_NOISE, h_clean.shape)
    _, qv = storms(N_VAL_STORMS, np.random.default_rng([SEED, 2]), STORM_PEAK)
    hv = np.array([pond_level(t, qi, weir(WEIR_C)) for qi in qv])
    hv_read = hv + np.random.default_rng([SEED, 5]).normal(0, LEVEL_NOISE, hv.shape)
    return {"t": t, "q": q, "h_clean": h_clean, "h": h, "qv": qv, "hv": hv, "hv_read": hv_read}


def held_out_levels(rec: dict[str, np.ndarray], xi: np.ndarray) -> np.ndarray:
    """A fitted model run through the held-out storms.

    Args:
        rec: ``pond_records``'s output.
        xi: The model's coefficients.

    Returns:
        Its levels, (storms, times), m.
    """
    return np.array([pond_level(rec["t"], q, model_rhs(xi)) for q in rec["qv"]])


def threshold_sweep(
    rec: dict[str, np.ndarray], rows: np.ndarray, y: np.ndarray
) -> list[dict[str, float]]:
    """Terms kept and held-out level error at every threshold in LAMBDAS.

    Args:
        rec: ``pond_records``'s output.
        rows: Year 1's library rows.
        y: Year 1's dh/dt.

    Returns:
        Per threshold: the terms kept, the level error against the true level, and against the
        gauge's readings (which carry LEVEL_NOISE), m.
    """
    out = []
    for lam in LAMBDAS:
        x = stlsq(rows, y, lam)
        hm = held_out_levels(rec, x)
        out.append({"lam": lam, "terms": int((x != 0).sum()), "val_rmse_m": rmse(hm, rec["hv"]),
                    "val_read_rmse_m": rmse(hm, rec["hv_read"])})  # fmt: skip
    return out


def pond_stage(out_dir: Path) -> dict[str, object]:
    """SINDy on the pond: checks, the fit, the sweep, the design storm, clogging, noise.

    Args:
        out_dir: Where the figures go.

    Returns:
        The truth, SINDy's and the other fits' terms and quality, the sweep, the design storm's
        peaks, clogging, and the noise runs.
    """
    rec = pond_records()
    t, q1, h1 = rec["t"], rec["q"], rec["h"]
    rows, y = stack(t, q1, h1)
    xi = stlsq(rows, y, LAMBDA)
    xi_o = stlsq(rows, y, 0.0, keep=np.array([n in ("Q_in", "h^0.5") for n in NAMES]))  # orifice
    xi_m = stlsq(rows, y, LAMBDA, keep=np.array([n != TRUE_TERM for n in NAMES]))  # term missing
    sweep = threshold_sweep(rec, rows, y)
    ls = np.linalg.lstsq(rows, y, rcond=None)[0]
    scale = np.linalg.norm(rows, axis=0)
    frames = {"least_squares": (ls * scale).tolist(), "final": (xi * scale).tolist(),
              "threshold": float(LAMBDA * np.abs(ls * scale).max())}  # fmt: skip
    td, qd = design_storm()
    levels = {
        "truth": pond_level(td, qd, weir(WEIR_C)),
        "SINDy": pond_level(td, qd, model_rhs(xi)),
        "orifice assumed": pond_level(td, qd, model_rhs(xi_o)),
        f"no {TRUE_TERM} in the library": pond_level(td, qd, model_rhs(xi_m)),
    }

    def r2(xi_: np.ndarray) -> float:  # share of the records' dh/dt the model explains
        return float(1 - np.sum((y - rows @ xi_) ** 2) / np.sum((y - y.mean()) ** 2))

    fits = {"SINDy": xi, "orifice assumed": xi_o, f"no {TRUE_TERM} in the library": xi_m}
    fit_quality = {
        k: {"r2": r2(v), "val_level_rmse_m": rmse(held_out_levels(rec, v), rec["hv"])}
        for k, v in fits.items()
    }
    attempt_figure(t, q1, h1, out_dir)
    picture_figures(frames, out_dir)
    sweep_figure(sweep, out_dir)
    design_figure(td, levels, out_dir)
    return {
        "truth": terms(np.array([1 / AREA, 0, 0, 0, -WEIR_C / AREA, 0])),
        "max_level": float(rec["h_clean"].max()),
        "recorded_peak_max": float(q1.max()),
        "checks": pond_checks(t, q1, rec["h_clean"]),
        "sindy": terms(xi),
        "area_from_fit": float(1 / xi[0]),
        "c_from_fit": float(-xi[NAMES.index(TRUE_TERM)] / xi[0]),
        "orifice": terms(xi_o) | {"area": float(1 / xi_o[0])},
        "missing": terms(xi_m),
        "sweep": sweep,
        "noise_floor_m": rmse(rec["hv_read"], rec["hv"]),  # the gauge's readings against the truth
        "frames": frames,
        "design_peak": {k: float(v.max()) for k, v in levels.items()},
        "fit_quality": fit_quality,
        "clog": clogging(t, q1, h1, xi),
        "noise": noise_runs(t, q1, rec["h_clean"]),
    }


# --------------------------------------------------------------------------- recessions


def recession_stage(out_dir: Path) -> dict[str, object]:
    """SINDy on the home basin's recession days, against the power law, train then test.

    Args:
        out_dir: Where the figure goes.

    Returns:
        The gauge, the days per period, the power law (fitted on the falls and on their
        logarithms), the linear reservoir, the library's largest correlation, and SINDy at five
        thresholds, each with its test RMSE (mm/day2).
    """
    cam = camels.load()
    b = cam.basin(camels.HOME)
    q = cam.flow[b].astype(float)
    p = cam.forcing[b, :, cam.channel("prcp_mm")].astype(float)
    dq = q[1:] - q[:-1]  # Q_t - Q_{t-1}
    qm = 0.5 * (q[1:] + q[:-1])
    dry = (p[1:] <= RAIN_MM) & (p[:-1] <= RAIN_MM) & (np.r_[0.0, p[:-2]] <= RAIN_MM)
    falling = (dq < 0) & (np.r_[0.0, dq[:-1]] < 0)
    ok = dry & falling & np.isfinite(dq) & (qm > 0)
    days = cam.dates[1:]
    split = {}
    for name in ("train", "test"):
        start, end = (np.datetime64(d) for d in camels.PERIODS[name])
        split[name] = ok & (days >= start) & (days <= end)
    x, y = qm[split["train"]], -dq[split["train"]]
    xt, yt = qm[split["test"]], -dq[split["test"]]

    def lib(v: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(v), v**0.5, v, v**1.5, v**2])

    slope, icpt = np.polyfit(np.log(x), np.log(y), 1)  # recession analysis's log-log fit, a start
    (a_pl, b_pl), _ = curve_fit(lambda v, a, b: a * v**b, x, y, p0=(np.exp(icpt), slope))

    out: dict[str, object] = {
        "gauge": camels.HOME,
        "days": {"train": int(split["train"].sum()), "test": int(split["test"].sum())},
        "power_law": {  # least squares on -dQ/dt itself, like every other model here
            "a": float(a_pl),
            "b": float(b_pl),
            "b_log_fit": float(slope),
            "test_rmse": rmse(a_pl * xt**b_pl, yt),
        },
        "linear": {"k": float(np.sum(x * y) / np.sum(x * x))},
        "corr_max": float(np.max(np.abs(np.corrcoef(lib(x)[:, 1:].T) - np.eye(4)))),
        "sindy": [],
    }
    out["linear"]["test_rmse"] = rmse(out["linear"]["k"] * xt, yt)
    for lam in (0.0, 0.01, LAMBDA, 0.1, 0.3):
        xi = stlsq(lib(x), y, lam)
        kept = {n: float(c) for n, c in zip(REC_NAMES, xi, strict=True) if c != 0}
        out["sindy"].append({"lam": lam, "terms": kept, "test_rmse": rmse(lib(xt) @ xi, yt)})
    recession_figure(x, y, out, out_dir)
    return out


# --------------------------------------------------------------------------- the river reach


def celerity(q: np.ndarray) -> np.ndarray:
    """Kinematic celerity (5/3) V of a wide channel by Manning's formula.

    Args:
        q: Flow, m3/s (below 1 m3/s counts as 1).

    Returns:
        Celerity, m/s.
    """
    q = np.maximum(q, 1.0)
    h = (q * MANNING / (WIDTH_M * np.sqrt(SLOPE))) ** 0.6
    return 5 / 3 * q / (WIDTH_M * h)


def route(q_in: np.ndarray, per_hour: int = 1, reach: float = REACH, sub: int = SUB) -> np.ndarray:
    """Variable-parameter Muskingum-Cunge (Ponce & Yevjevich, 1978), batched.

    Args:
        q_in: Inflow hydrographs, (batch, times), m3/s, ``per_hour`` values an hour.
        per_hour: Output steps an hour.
        reach: Reach length (m); the channel is ``LENGTH`` long.
        sub: Sub-steps per output step.

    Returns:
        Outflow at the downstream end, (batch, times).
    """
    nr = int(round(LENGTH / reach))
    dt = 3600 / per_hour / sub
    old = np.full((q_in.shape[0], nr + 1), BASE)
    out = np.empty_like(q_in)
    out[:, 0] = BASE
    for k in range(1, q_in.shape[1]):
        for s in range(1, sub + 1):
            w = s / sub
            new = np.empty_like(old)
            new[:, 0] = (1 - w) * q_in[:, k - 1] + w * q_in[:, k]
            for j in range(nr):
                qbar = (new[:, j] + old[:, j] + old[:, j + 1]) / 3  # three-point average
                c = celerity(qbar)
                cr = c * dt / reach  # Courant number
                dd = qbar / (WIDTH_M * SLOPE * c * reach)  # cell Reynolds number
                den = 1 + cr + dd
                new[:, j + 1] = ((-1 + cr + dd) * new[:, j] + (1 + cr - dd) * old[:, j]
                                 + (1 - cr + dd) * old[:, j + 1]) / den  # fmt: skip
            old = new
        out[:, k] = old[:, -1]
    return out


def floods(
    n: int, rng: np.random.Generator, peak: tuple[float, float], per_hour: int = 1
) -> np.ndarray:
    """Random gamma-shaped inflow floods on the base flow.

    Args:
        n: Number of floods.
        rng: Generator for the peaks, peak times, and shapes.
        peak: Range of the peak inflow, m3/s.
        per_hour: Samples an hour.

    Returns:
        Inflows, (n, HOURS * per_hour + 1), m3/s.
    """
    t = np.arange(HOURS * per_hour + 1) / per_hour  # hours
    out = np.empty((n, len(t)))
    for i in range(n):
        qp, tp, k = rng.uniform(*peak), rng.uniform(4, 16), rng.uniform(2, 6)
        out[i] = BASE + (qp - BASE) * (t / tp) ** k * np.exp(k * (1 - t / tp))
    return out


class SpectralConv(nn.Module):
    """Multiply the lowest ``modes`` Fourier modes by learned complex weights, drop the rest."""

    def __init__(self, width: int, modes: int) -> None:
        """``width`` channels in and out; ``modes`` kept."""
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, N) -> (B, C, N)."""
        xf = torch.fft.rfft(x, norm="forward")  # (B, C, N // 2 + 1): coefficients of the function
        yf = torch.zeros_like(xf)
        yf[:, :, : self.modes] = torch.einsum("bin,ion->bon", xf[:, :, : self.modes], self.w)
        return torch.fft.irfft(yf, n=x.shape[-1], norm="forward")


class FNO(nn.Module):
    """Lift, four Fourier layers (spectral + pointwise), project (Li et al., 2021)."""

    def __init__(self) -> None:
        """Sizes from the module constants."""
        super().__init__()
        self.lift = nn.Conv1d(2, FNO_WIDTH, 1)
        self.spectral = nn.ModuleList(SpectralConv(FNO_WIDTH, FNO_MODES) for _ in range(FNO_LAYERS))
        self.pointwise = nn.ModuleList(
            nn.Conv1d(FNO_WIDTH, FNO_WIDTH, 1) for _ in range(FNO_LAYERS)
        )
        self.project = nn.Sequential(nn.Conv1d(FNO_WIDTH, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Scaled inflow (B, N) -> scaled outflow (B, N), on any N."""
        n = q.shape[-1]
        grid = torch.linspace(0, 1, n).expand(q.shape[0], n)  # where each sample sits in time
        x = self.lift(torch.stack([q, grid], 1))  # (B, W, N)
        for i, (s, w) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            x = s(x) + w(x)
            if i < FNO_LAYERS - 1:
                x = nn.functional.gelu(x)
        return self.project(x).squeeze(1)


def to_net(q: np.ndarray) -> torch.Tensor:
    """Flows on the network's scale.

    Args:
        q: Flows, m3/s.

    Returns:
        (q - BASE) / Q_SCALE as float32.
    """
    return torch.tensor((q - BASE) / Q_SCALE, dtype=torch.float32)


def fno_job(seed: int, data: dict[str, np.ndarray]) -> dict[str, object]:
    """Train one FNO (CPU, fixed threads) and predict every evaluation set.

    Args:
        seed: Weights and batch-order seed.
        data: The flood sets, ``<set>_in`` and ``<set>_out`` in m3/s.

    Returns:
        The seed, training seconds, the validation curve (RMSE per epoch, m3/s), predictions for
        the test, big, fine, and fine_hourly floods (m3/s) and the test set's seconds, the
        number of real weights, and the share of validation error lost in the last 30 epochs.
    """
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    net = FNO()
    x, y = to_net(data["train_in"]), to_net(data["train_out"])
    xv, yv = to_net(data["val_in"]), to_net(data["val_out"])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    gen = torch.Generator().manual_seed(seed)
    val = []
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad(set_to_none=True)
            (net(x[b]) - y[b]).pow(2).mean().backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            val.append(Q_SCALE * float((net(xv) - yv).pow(2).mean().sqrt()))
    secs = time.perf_counter() - t0
    preds = {}
    with torch.no_grad():
        for k in ("test", "big", "fine", "fine_hourly"):
            t1 = time.perf_counter()
            preds[k] = BASE + Q_SCALE * net(to_net(data[f"{k}_in"])).numpy().astype(float)
            if k == "test":
                preds["test_seconds"] = time.perf_counter() - t1
    n_par = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())  # real numbers
    still_falling = (val[-31] - val[-1]) / val[-31]  # share of the error lost in the last 30 epochs
    return {"seed": seed, "train_seconds": secs, "val_curve": val, "preds": preds,
            "parameters": n_par, "val_drop_last30": still_falling}  # fmt: skip


def lagged(q: np.ndarray, k: int) -> np.ndarray:
    """The last k hours of inflow above BASE at every hour.

    Args:
        q: Inflows, (batch, times), m3/s.
        k: Hours of memory.

    Returns:
        (batch, times, k): lag 0 first, zero before the record starts.
    """
    e = q - BASE
    return np.stack([np.pad(e, ((0, 0), (j, 0)))[:, : e.shape[1]] for j in range(k)], -1)


def unit_hydrograph(q_in: np.ndarray, q_out: np.ndarray, k: int) -> np.ndarray:
    """A unit hydrograph of k hours, from inflow to outflow, by plain least squares.

    The same linear convolution as Lecture 20's, which went from rain to flow and was fitted by
    ridge; here the floods are noise-free and there is no penalty.

    Args:
        q_in: Inflows, (floods, times), m3/s.
        q_out: Outflows, the same shape, m3/s.
        k: Hours of memory.

    Returns:
        The k weights.
    """
    return np.linalg.lstsq(lagged(q_in, k).reshape(-1, k), (q_out - BASE).ravel(), rcond=None)[0]


def uh_search(data: dict[str, np.ndarray]) -> tuple[int, list[dict[str, float]]]:
    """Choose the unit hydrograph's length on validation, over every length up to the record.

    The ladder ends at the record's length, the longest memory a unit hydrograph can have, so
    the search covers the whole feasible range and the best may sit at its end (the
    interior rule is for grids that could be widened).

    Args:
        data: The flood sets.

    Returns:
        The chosen hours, and each length's validation RMSE (m3/s).
    """
    assert UH_LADDER[-1] == HOURS, (UH_LADDER, HOURS)
    tried = {}
    for k in UH_LADDER:
        u = unit_hydrograph(data["train_in"], data["train_out"], k)
        pred = BASE + lagged(data["val_in"], k) @ u
        tried[k] = rmse(pred, data["val_out"])
    best = min(tried, key=tried.get)
    return best, [{"hours": k, "val_rmse": v} for k, v in tried.items()]


def flood_sets() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Random inflow floods for every set, routed by the solver at one internal step, STEP_S.

    Hourly floods take SUB sub-steps an hour; the half-hourly ("fine") floods take FINE_SUB
    sub-steps per half hour, so the solver is the same and only the input's sampling changes.
    The same fine floods sampled hourly ("fine_hourly") are routed as hourly floods.

    Returns:
        ``<set>_in`` and ``<set>_out`` (m3/s) for train, val, test, big, fine, and
        fine_hourly; and the solver's seconds per hourly set.
    """
    assert 3600 / 2 / FINE_SUB == STEP_S, (FINE_SUB, STEP_S)
    rng = np.random.default_rng([SEED, 20])
    data = {f"{k}_in": floods(n, rng, peak) for k, n, peak in (
        ("train", N_TRAIN, FLOOD_PEAK), ("val", N_VAL, FLOOD_PEAK), ("test", N_TEST, FLOOD_PEAK),
        ("big", N_BIG, BIG_PEAK))}  # fmt: skip
    data["fine_in"] = floods(N_FINE, rng, FLOOD_PEAK, per_hour=2)
    secs = {}
    for k in ("train", "val", "test", "big"):
        t0 = time.perf_counter()
        data[f"{k}_out"] = route(data[f"{k}_in"])
        secs[k] = time.perf_counter() - t0
    data["fine_out"] = route(data["fine_in"], per_hour=2, sub=FINE_SUB)
    data["fine_hourly_in"] = np.ascontiguousarray(data["fine_in"][:, ::2])  # on the hour
    assert data["fine_hourly_in"].shape == data["test_in"][:N_FINE].shape
    data["fine_hourly_out"] = route(data["fine_hourly_in"])
    return data, secs


def solver_checks(data: dict[str, np.ndarray], secs: dict[str, float]) -> dict[str, object]:
    """The solver: its water balance, a finer grid, a finer step, lag and attenuation.

    Args:
        data: ``flood_sets``'s floods.
        secs: Its timings.

    Returns:
        The test set's seconds; the volume out / in after 200 more hours (first 50 test
        floods) and the most that arrives after the first 72 hours; the largest change on half
        the reach and step (m3/s); the RMSE of the fine floods at half the step (m3/s); median
        lags (h) and attenuation; the volume out / in within 72 hours, all test floods; and
        the RMSE between the solver's outflows for the fine floods sampled hourly and half
        hourly, at the hours (m3/s).
    """
    n_chk = min(50, N_TEST)
    test_in, test_out = data["test_in"][:n_chk], data["test_out"][:n_chk]
    long_in = np.concatenate([test_in, np.full((n_chk, 200), BASE)], axis=1)
    long_vol = (route(long_in) - BASE).sum(1) / (long_in - BASE).sum(1)
    vol_72h = (test_out - BASE).sum(1) / (test_in - BASE).sum(1)
    fine_grid = route(test_in, reach=REACH / 2, sub=2 * SUB)
    half_step = route(data["fine_in"], per_hour=2, sub=2 * FINE_SUB)  # 450 s instead of 900 s
    sv = (data["test_out"] - BASE).sum(1) / (data["test_in"] - BASE).sum(1)
    return {
        "test_seconds": secs["test"],
        "step_s": STEP_S,
        "long_volume": {"min": float(long_vol.min()), "max": float(long_vol.max())},
        "after_72h_max": float((long_vol - vol_72h).max()),  # share of the inflow, still to come
        "finer_grid_max_diff": float(np.abs(fine_grid - test_out).max()),
        "fine_half_step_rmse": rmse(half_step, data["fine_out"]),
        "lag_hours": {
            "test": float(np.median(data["test_out"].argmax(1) - data["test_in"].argmax(1))),
            "big": float(np.median(data["big_out"].argmax(1) - data["big_in"].argmax(1))),
        },
        "attenuation": float(np.median(data["test_out"].max(1) / data["test_in"].max(1))),
        "volume_72h": {"min": float(sv.min()), "max": float(sv.max())},
        "fine_sampling_rmse": rmse(data["fine_hourly_out"], data["fine_out"][:, ::2]),
    }


def flood_scores(pred: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> dict[str, float]:
    """Errors on the test floods and on the big ones.

    Args:
        pred: Predicted outflows for ``test`` and ``big``, m3/s.
        data: The flood sets.

    Returns:
        RMSE (m3/s), median relative peak error, and median peak-timing error (h), per set.
    """
    out = {}
    for k in ("test", "big"):
        p, o = pred[k], data[f"{k}_out"]
        out[f"{k}_rmse"] = rmse(p, o)
        out[f"{k}_peak"] = float(np.median(np.abs(p.max(1) - o.max(1)) / o.max(1)))
        out[f"{k}_timing_h"] = float(np.median(np.abs(p.argmax(1) - o.argmax(1))))
    return {k: out[k] for k in ("test_rmse", "big_rmse", "test_peak", "big_peak",
                                "test_timing_h", "big_timing_h")}  # fmt: skip


def water_balance(pred: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, object]:
    """An operator's volumes on the test floods: against the inflow, and against the solver.

    Volumes are sums of the hourly flow above BASE over the 72 hours (times 3,600 s for m3).

    Args:
        pred: Predicted outflows on the test floods, (floods, times), m3/s.
        data: The flood sets.

    Returns:
        Volume out / in and volume / the solver's (lowest, highest over floods); the mean flow
        above the solver's (m3/s, lowest and highest); and, for the flood with the most volume
        out / in, that ratio, its inflow peak (m3/s), its excess over the inflow in m3 and as a
        mean flow over the record (m3/s), and its volume and mean flow against the solver's.
    """
    q_in, q_sol = data["test_in"], data["test_out"]
    v_in, v_sol, v = ((a - BASE).sum(1) for a in (q_in, q_sol, pred))
    ratio, vs_sol = v / v_in, v / v_sol
    offset = (pred - q_sol).mean(1)  # m3/s over the record, per flood
    i = int(np.argmax(ratio))
    return {
        "volume": {"min": float(ratio.min()), "max": float(ratio.max())},
        "volume_vs_solver": {"min": float(vs_sol.min()), "max": float(vs_sol.max())},
        "offset_vs_solver_m3s": {"min": float(offset.min()), "max": float(offset.max())},
        "most_water": {
            "ratio": float(ratio[i]),
            "inflow_peak": float(q_in[i].max()),
            "excess_m3": float(3600 * (v[i] - v_in[i])),
            "excess_m3s": float((v[i] - v_in[i]) / q_in.shape[1]),
            "vs_solver": float(vs_sol[i]),
            "vs_solver_m3s": float(offset[i]),
        },
    }


def train_fnos(data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """FNO_SEEDS operators, one per worker process (spawned, so no state leaks between them).

    Args:
        data: The flood sets.

    Returns:
        ``fno_job``'s output per seed, in seed order.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(fno_job, SEED + s, data) for s in range(FNO_SEEDS)]
        return [f.result() for f in futs]


def river_stage(out_dir: Path) -> dict[str, object]:
    """Simulate floods, fit the unit hydrograph, train three FNOs, test them four ways.

    Args:
        out_dir: Where the figures go.

    Returns:
        The solver's checks; the unit hydrograph's length search and scores; per FNO seed its
        scores, fine-flood RMSE (half-hourly, and the same floods hourly, and their ratio),
        timings, water balance, and training progress; and the spans over seeds.
    """
    data, secs = flood_sets()
    out: dict[str, object] = {"solver": solver_checks(data, secs)}
    k_uh, uh_tried = uh_search(data)
    u = unit_hydrograph(data["train_in"], data["train_out"], k_uh)
    uh = {k: BASE + lagged(data[f"{k}_in"], k_uh) @ u for k in ("test", "big")}
    out["uh"] = {"hours": k_uh, "tried": uh_tried} | flood_scores(uh, data)
    fnos = train_fnos(data)
    out["fno"] = [
        flood_scores(f["preds"], data)
        | {
            "seed": f["seed"],
            "fine_rmse": rmse(f["preds"]["fine"], data["fine_out"]),
            "fine_hourly_rmse": rmse(f["preds"]["fine_hourly"], data["fine_hourly_out"]),
            "fine_ratio": rmse(f["preds"]["fine"], data["fine_out"])
            / rmse(f["preds"]["fine_hourly"], data["fine_hourly_out"]),
            "train_seconds": f["train_seconds"],
            "test_seconds": f["preds"]["test_seconds"],
            "val_final": f["val_curve"][-1],
        }
        | water_balance(f["preds"]["test"], data)
        | {"parameters": f["parameters"], "val_drop_last30": f["val_drop_last30"]}
        for f in fnos
    ]
    keys = ("test_rmse", "big_rmse", "test_peak", "big_peak", "fine_rmse", "fine_hourly_rmse",
            "fine_ratio", "train_seconds", "test_seconds", "val_drop_last30")  # fmt: skip
    span = {k: {"min": min(f[k] for f in out["fno"]), "max": max(f[k] for f in out["fno"])}
            for k in keys}  # fmt: skip
    for k in ("volume", "volume_vs_solver", "offset_vs_solver_m3s"):
        span[f"{k}_min"] = min(f[k]["min"] for f in out["fno"])
        span[f"{k}_max"] = max(f[k]["max"] for f in out["fno"])
    span["most_water"] = max((f["most_water"] for f in out["fno"]), key=lambda m: m["ratio"])
    out["fno_span"] = span
    fno_figure(data, fnos[0]["preds"], uh, out_dir)
    return out


# --------------------------------------------------------------------------- figures


def attempt_figure(t: np.ndarray, q: np.ndarray, h: np.ndarray, out_dir: Path) -> None:
    """One recorded storm: inflow and level."""
    i = int(np.argmax(q.max(1)))  # the largest recorded storm
    fig, (ax, bx) = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.3})
    ax.plot(t / 3600, q[i], color=aml_style.PALETTE[0], ls="-")
    ax.set(xlabel="hours", ylabel="inflow, m³/s", xlim=(0, STORM_HOURS))
    bx.plot(t / 3600, h[i], color=aml_style.PALETTE[1], ls="-", lw=1)
    bx.set(xlabel="hours", ylabel="level above the outlet, m", xlim=(0, STORM_HOURS))
    aml_style.save(fig, out_dir, "attempt")


def picture_figures(frames: dict, out_dir: Path) -> None:
    """The one picture: all six terms; the threshold; the two that survive."""
    ls, final, thr = (
        np.array(frames["least_squares"]),
        np.array(frames["final"]),
        frames["threshold"],
    )
    lim = 1.15 * np.abs(ls).max()
    for k, (vals, show_thr) in enumerate(((ls, False), (ls, True), (final, False)), start=1):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        colors = [aml_style.PALETTE[1] if (show_thr and abs(v) < thr) else aml_style.PALETTE[0]
                  for v in vals]  # fmt: skip
        ax.bar(range(len(NAMES)), vals, color=colors)
        if show_thr:
            for s in (-1, 1):
                ax.axhline(s * thr, color=aml_style.MUTED, ls=":", lw=1)
        ax.axhline(0, color=aml_style.INK, lw=0.8)
        ax.set(xticks=range(len(NAMES)), xticklabels=NAMES, ylim=(-lim, lim),
               ylabel="coefficient × column size")  # fmt: skip
        aml_style.save(fig, out_dir, f"picture_{k}")


def sweep_figure(sweep: list[dict], out_dir: Path) -> None:
    """Terms kept and validation error across the threshold."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    lam = [max(s["lam"], 3e-4) for s in sweep]
    ax.plot(
        lam, [100 * s["val_rmse_m"] for s in sweep], color=aml_style.PALETTE[0], ls="-", marker="o"
    )
    for x, s in zip(lam, sweep, strict=True):
        ax.annotate(f"{s['terms']} terms", (x, 100 * s["val_rmse_m"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)  # fmt: skip
    ax.axvline(LAMBDA, color=aml_style.MUTED, ls=":", lw=1)
    ax.set(xscale="log", ylabel="validation level error, cm",
           xlabel="threshold, share of the largest coefficient (0 drawn at left)")  # fmt: skip
    aml_style.save(fig, out_dir, "sweep")


def design_figure(t: np.ndarray, levels: dict[str, np.ndarray], out_dir: Path) -> None:
    """The design storm's level by each model, against the rim, in two figures.

    The decision slide's figure leaves out the model whose library lacks the true term; the
    reveal's figure adds it.
    """
    styles = {"truth": (aml_style.INK, "-", 3), "SINDy": (aml_style.PALETTE[0], "--", 2),
              "orifice assumed": (aml_style.PALETTE[1], "-", 2)}  # fmt: skip
    top = 1.08 * max([DEPTH, *(float(h.max()) for h in levels.values())])
    for name, shown in (("design", list(styles)), ("design_missing", list(levels))):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        for key in shown:
            color, ls, lw = styles.get(key, (aml_style.PALETTE[3], "-.", 2))
            ax.plot(t / 3600, levels[key], color=color, ls=ls, lw=lw, label=key)
        ax.axhline(DEPTH, color=aml_style.PALETTE[1], ls=":", lw=1.5)
        rim_y = DEPTH + 0.03 * top
        ax.text(
            STORM_HOURS - 0.3, rim_y, "rim", ha="right", color=aml_style.PALETTE[1], fontsize=10
        )
        ax.set(xlim=(0, STORM_HOURS), ylim=(0, top), xlabel="hours",
               ylabel="level above the outlet, m")  # fmt: skip
        ax.legend(loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.88))
        aml_style.save(fig, out_dir, name)


def recession_figure(x: np.ndarray, y: np.ndarray, out: dict, out_dir: Path) -> None:
    """-dQ/dt against Q on the training recession days, log axes, with the power law."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.scatter(x, y, s=5, color=aml_style.MUTED, alpha=0.4, linewidths=0, rasterized=True)
    qq = np.geomspace(x.min(), x.max(), 100)
    pl = out["power_law"]
    ax.plot(qq, pl["a"] * qq ** pl["b"], color=aml_style.PALETTE[1], ls="-", lw=2.5,
            label=f"power law: one term, b = {pl['b']:.2f}")  # fmt: skip
    power = {"1": 0.0, "Q^0.5": 0.5, "Q": 1.0, "Q^1.5": 1.5, "Q^2": 2.0}
    dense, sparse = out["sindy"][0], out["sindy"][-1]  # the smallest and largest thresholds
    for s, color, ls in ((dense, aml_style.PALETTE[0], "--"), (sparse, aml_style.PALETTE[2], "-.")):
        curve = sum(c * qq ** power[n] for n, c in s["terms"].items())
        ax.plot(qq, curve, color=color, ls=ls, lw=2, label=f"SINDy, {len(s['terms'])} terms")
    ax.set(xscale="log", yscale="log", xlabel="flow Q, mm/day", ylabel="fall −dQ/dt, mm/day²")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    aml_style.save(fig, out_dir, "recession")


def fno_figure(
    data: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    uh: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """A test flood and a flood twice as large as any in training: solver, FNO, unit hydrograph."""
    for name, key in (("fno", "test"), ("big", "big")):
        i = int(
            np.argsort(data[f"{key}_in"].max(1))[len(data[f"{key}_in"]) // 2]
        )  # the median peak
        hours = np.arange(data[f"{key}_in"].shape[1])
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        ax.plot(hours, data[f"{key}_in"][i], color=aml_style.MUTED, ls=":", lw=1.5, label="inflow")
        ax.plot(hours, data[f"{key}_out"][i], color=aml_style.INK, ls="-", lw=3, label="solver")
        ax.plot(hours, pred[key][i], color=aml_style.PALETTE[1], ls="--", lw=2, label="FNO")
        ax.plot(
            hours, uh[key][i], color=aml_style.PALETTE[0], ls="-.", lw=2, label="unit hydrograph"
        )
        ax.set(xlim=(0, 48), xlabel="hours", ylabel="flow, m³/s")
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        aml_style.save(fig, out_dir, name)


def title_art_figure(out_dir: Path) -> None:
    """The title art: a river's flood wave passing a dark pond, under a low sun."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0a0d14", "#101826", "#151a24", "#07080b"), grain=0.08)
    xs = np.linspace(0, 1, 200)
    wave = [(x, top * (0.62 - 0.12 * np.exp(-(((x - 0.45) / 0.12) ** 2)))) for x in xs]
    c.glaze(c.polygon([*wave, (1, top), (0, top)]), "#1c2a3d", 0.9)
    c.glow(c.lines([wave], 3.0, soft=1.0), "#8fb3c9", 0.45, spread=6)
    c.glaze(c.polygon(title_art.stepped(0.62, top * 0.70, 0.3, top * 0.12, 3)), "#2b2533", 0.9)
    c.glow(c.disc((0.25, top * 0.22), 0.045, soft=3), "#eaa548", 0.45, spread=24)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a flood wave rises as a single blue line of light "
        "across a dark river, beside a stepped basin; a gold sun glows low on the left.",
    )


# --------------------------------------------------------------------------- main


def log(msg: str) -> None:
    """Progress on stderr (stdout goes to /dev/null in the Makefile)."""
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    """Generate every figure and number for CEE Lecture 24."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.set_num_threads(THREADS)
    t0 = time.perf_counter()
    numbers: dict[str, object] = {
        "settings": {
            "area": AREA,
            "weir_c": WEIR_C,
            "depth": DEPTH,
            "level_noise": LEVEL_NOISE,
            "read_min": READ_MIN,
            "storms": N_STORMS,
            "storm_peak": list(STORM_PEAK),
            "design": DESIGN,
            "clog": CLOG,
            "lambda": LAMBDA,
            "boot": BOOT,
            "width_m": WIDTH_M,
            "slope": SLOPE,
            "manning": MANNING,
            "length_km": LENGTH / 1000,
            "reach_km": REACH / 1000,
            "base": BASE,
            "hours": HOURS,
            "flood_peak": list(FLOOD_PEAK),
            "big_peak": list(BIG_PEAK),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "fno": {
                "width": FNO_WIDTH,
                "modes": FNO_MODES,
                "layers": FNO_LAYERS,
                "epochs": EPOCHS,
                "seeds": FNO_SEEDS,
            },
            "rain_mm": RAIN_MM,
        },
    }
    numbers["pond"] = pond_stage(out_dir)
    log(f"pond in {time.perf_counter() - t0:.0f} s")
    numbers["recession"] = recession_stage(out_dir)
    log(f"recession in {time.perf_counter() - t0:.0f} s")
    numbers["river"] = river_stage(out_dir)
    log(f"river in {time.perf_counter() - t0:.0f} s")
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_3", "design", "fno", "big"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    log(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))  # read by the spawned workers
    main()
