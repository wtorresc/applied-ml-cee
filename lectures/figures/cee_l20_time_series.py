"""Figures and numbers for CEE Lecture 20: 1-D signals, 1-D CNNs, and LSTMs.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l20_time_series.py --out build/fig/cee_l20_time_series

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides). The lecture
follows CS 441's "Audio and 1D Signals" section by section, on two CEE records:

- A river: the North Fork Vermilion River near Bismarck, IL (USGS 03338780, 692 km2), and the
  other CAMELS basins (Newman et al., 2015; CC BY 4.0), from ``scripts/derive_camels.py`` (read
  by ``camels.py``). Can we warn, a day ahead, that the river will run above its 1-in-100-days
  flow? Windowed regressions, a unit hydrograph (a 1-D filter), SARIMA, a 1-D CNN and an LSTM
  trained here in seconds on one GPU, and Chronos-2 (Amazon, Apache-2.0) used without training.
- A bridge: KW51, a steel bowstring railway bridge in Leuven (Maes & Lombaert, 2021; Zenodo
  3745914, CC BY-NC-SA 4.0), from ``scripts/fetch_kw51.sh``: accelerations of train passages
  (825.8 Hz) for the Fourier series, spectra, and spectrograms, and the operator's hourly natural
  frequencies over 15 months, through a retrofit, for a temperature-confounded alarm.

Every comparison uses the same periods (``camels.PERIODS``: train WY1990-2004, validation
WY2005-2008, test WY2009-2014) and the same target days: a day-ahead forecast for day t is made on
day t-1 (``issued``), so no training row has a validation or test day as its target. Settings are
chosen on validation and the test period is scored once. Every network trains with early
stopping on validation, checked every ``CHECK_STEPS`` gradient steps whatever the amount of data,
at an Adam rate bracketed on validation (``bracket``, ``tune``); the kept runs and their
neighbours must stop by patience before the step cap, and every kept run's best check must come
after the first (a run that peaks before it was checked too coarsely to place its best).
Networks are trained with three or five seeds where a slide quotes their median; the forest
settings are scored on three seeds.
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, spectrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

import aml_style
import camels
import title_art
from common import DATA_DIR, parse_out_dir, require, seed_all, write_numbers

# Read when huggingface_hub is imported (by chronos): no usage reports from this script.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from chronos import BaseChronosPipeline

SEED = 20
KW51 = DATA_DIR / "raw" / "kw51"
# The attempt: two storms of similar rain in the test years, one in spring, one in summer.
STORMS = ("2013-04-10", "2012-08-26")  # first day of two rainy days each
STORM_DAYS = 3  # a storm's peak is read over its two rainy days and the day after
WINDOWS = (1, 2, 3, 5, 7, 14, 30)  # days of rain and flow given to the windowed regressions
KNN_K = 10
RF_TREES, RF_LEAF = 200, 5
RF_FEATURES = 1 / 3  # a random forest tries a third of the inputs at each split (Breiman, 2001)
FOREST_SEEDS = (SEED, SEED + 1, SEED + 2)  # the window sweep scores each forest setting on three
UH_LAGS = 60  # days of rain in the unit hydrograph (its weights are near 0 by then)
UH_ALPHAS = (0.0, 1e3, 1e4, 1e5, 1e6, 1e7)
SARIMA_ORDER, SARIMA_SEASONAL = (1, 0, 1), (0, 1, 1, 12)
LOG_EPS = 0.01  # mm/day added before logs (the home basin's smallest flows are near 0.02)
FLOOD_Q = 0.99  # alert when tomorrow's flow exceeds the training period's 99th percentile
SPLIT_DRAWS = 10  # random test sets (days, or whole water years) against the later years


def log(msg: str) -> None:
    """Progress on stderr (make sends stdout to /dev/null)."""
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# The river: windows, filters, SARIMA
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class River:
    """One basin's daily rain and flow, with the three periods' day indices."""

    rain: np.ndarray  # (days,) mm/day
    flow: np.ndarray  # (days,) mm/day, NaN where missing
    dates: np.ndarray  # (days,) datetime64[D]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def river(cam: camels.Camels, gauge: str = camels.HOME) -> River:
    """One basin's rain and flow."""
    b = cam.basin(gauge)
    return River(
        rain=cam.forcing[b, :, cam.channel("prcp_mm")].astype(float),
        flow=cam.flow[b].astype(float),
        dates=cam.dates,
        train=cam.period("train"),
        val=cam.period("val"),
        test=cam.period("test"),
    )


def window(
    r: River, days: np.ndarray, k: int, *, tomorrow_rain: bool = False, flow: bool = True
) -> np.ndarray:
    """Rain and flow of the last ``k`` days, ending on each day t, as one row per t.

    Columns: rain t, t-1, ..., t-k+1, then flow t, ..., t-k+1 (left out when ``flow`` is
    False), then rain t+1 if asked. The target of row t is the flow of day t+1.
    """
    lags = np.arange(k)
    cols = [r.rain[days[:, None] - lags]]
    if flow:
        cols.append(r.flow[days[:, None] - lags])
    if tomorrow_rain:
        cols.append(r.rain[days + 1, None])
    x = np.concatenate(cols, axis=1)
    assert x.shape == (len(days), k * (1 + flow) + tomorrow_rain), x.shape
    return x


def model_for(name: str, seed: int = SEED) -> object:
    """A windowed regression by name: linear, k-NN (standardized), or random forest."""
    if name == "linear":
        return LinearRegression()
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(KNN_K))
    if name == "forest":
        return RandomForestRegressor(
            RF_TREES,
            max_features=RF_FEATURES,
            min_samples_leaf=RF_LEAF,
            n_jobs=8,
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}")


def windowed_forecast(
    r: River,
    name: str,
    k: int,
    fit_days: np.ndarray,
    score_days: np.ndarray,
    *,
    target: str = "level",
    tomorrow_rain: bool = False,
    flow: bool = True,
    seed: int = SEED,
) -> np.ndarray:
    """Fit on ``fit_days`` and forecast tomorrow's flow on ``score_days``.

    Args:
        r: The basin.
        name: ``linear``, ``knn``, or ``forest``.
        k: Window length, days.
        fit_days: Days t whose (window, flow t+1) pairs train the model.
        score_days: Days t to forecast flow t+1 for.
        target: ``level`` (flow t+1) or ``change`` (flow t+1 - flow t, added back after).
        tomorrow_rain: Also give the model day t+1's rain (a perfect weather forecast).
        flow: Give the model the window's flow (False: rain only, which needs ``level``).
        seed: The forest's random state.

    Returns:
        Forecasts of flow t+1, one per score day (NaN where the window has a gap).
    """
    if not flow and target != "level":
        raise ValueError("a forecast of the change adds today's flow back: it needs the flow")
    xf, xs = (
        window(r, fit_days, k, tomorrow_rain=tomorrow_rain, flow=flow),
        window(r, score_days, k, tomorrow_rain=tomorrow_rain, flow=flow),
    )
    yf = r.flow[fit_days + 1] - (r.flow[fit_days] if target == "change" else 0.0)
    ok = np.isfinite(xf).all(axis=1) & np.isfinite(yf)
    model = model_for(name, seed)
    model.fit(xf[ok], yf[ok])
    good = np.isfinite(xs).all(axis=1)
    out = np.full(len(score_days), np.nan)
    out[good] = model.predict(xs[good])
    if target == "change":
        out += r.flow[score_days]
    return out


def issued(days: np.ndarray) -> np.ndarray:
    """The days a day-ahead forecast is made on, for target days ``days``: one day before.

    Every score and every fit is over target days inside a period, so a windowed model's last
    training row never has a validation or test day as its target.
    """
    return days - 1


@dataclass(frozen=True)
class Setting:
    """A windowed model's setting: the model, its target, and the window in days."""

    model: str  # linear, knn, or forest
    target: str  # level or change
    k: int


def setting_of(chosen: dict[str, object]) -> Setting:
    """The ``Setting`` in ``numbers["windows"]["chosen"]`` (which also holds its validation NSE)."""
    return Setting(**{f.name: chosen[f.name] for f in fields(Setting)})


def chosen_forecast(
    r: River, chosen: Setting, fit_days: np.ndarray, score_days: np.ndarray, **kw: bool
) -> np.ndarray:
    """The windowed model validation chose (``numbers["windows"]["chosen"]``), on other days.

    Args:
        r: The basin.
        chosen: The setting.
        fit_days: Days t whose (window, flow t+1) pairs train the model.
        score_days: Days t to forecast flow t+1 for.
        **kw: ``tomorrow_rain`` and ``flow``, as for ``windowed_forecast``.

    Returns:
        Forecasts of flow t+1, one per score day.
    """
    return windowed_forecast(
        r, chosen.model, chosen.k, fit_days, score_days, target=chosen.target, **kw
    )


def windows_sweep(r: River) -> dict[str, object]:
    """Validation NSE of each model, target, and window; the chosen setting scored on test.

    A forest's score moves by about 0.01 with its random state, as much as the gaps between
    its settings, so each forest setting is scored by the median over ``FOREST_SEEDS``;
    the forecasts after that use ``SEED``. The window chosen must be inside
    ``WINDOWS``. The test score refits on the training and validation years (19),
    and on the training years alone (15), the years the networks train on.
    """
    fit_tr, at_val, at_test = issued(r.train), issued(r.val), issued(r.test)
    val_nse: dict[str, dict[str, list[float]]] = {}
    forest_seeds: dict[str, list[list[float]]] = {}
    for name in ("linear", "knn", "forest"):
        for target in ("level", "change"):
            seeds = FOREST_SEEDS if name == "forest" else (SEED,)
            by_seed = [
                [
                    camels.nse(
                        r.flow[r.val],
                        windowed_forecast(r, name, k, fit_tr, at_val, target=target, seed=s),
                    )
                    for s in seeds
                ]
                for k in WINDOWS
            ]
            val_nse.setdefault(name, {})[target] = [float(np.median(v)) for v in by_seed]
            if name == "forest":
                forest_seeds[target] = by_seed
    best = max(
        ((n, t, k) for n in val_nse for t in val_nse[n] for k in range(len(WINDOWS))),
        key=lambda c: val_nse[c[0]][c[1]][c[2]],
    )
    name, target, ik = best
    runner = sorted(
        (
            (val_nse[n][t][k], n, t, WINDOWS[k])
            for n in val_nse
            for t in val_nse[n]
            for k in range(len(WINDOWS))
        ),
        reverse=True,
    )[1]
    runner_up = {"model": runner[1], "target": runner[2], "k": runner[3], "val_nse": runner[0]}
    if not 0 < ik < len(WINDOWS) - 1:
        raise ValueError(f"the chosen window {WINDOWS[ik]} is at the edge of WINDOWS; widen it")
    setting = Setting(name, target, WINDOWS[ik])
    chosen = asdict(setting) | {"val_nse": val_nse[name][target][ik]}
    fit19 = issued(np.concatenate([r.train, r.val]))
    test = {"persistence": camels.nse(r.flow[r.test], r.flow[at_test])}
    for years, fit in ((19, fit19), (15, fit_tr)):
        sfx = "" if years == 19 else "_15"
        test[f"chosen{sfx}"] = camels.nse(r.flow[r.test], chosen_forecast(r, setting, fit, at_test))
        test[f"chosen_tomorrow_rain{sfx}"] = camels.nse(
            r.flow[r.test], chosen_forecast(r, setting, fit, at_test, tomorrow_rain=True)
        )
    return {
        "windows": list(WINDOWS),
        "val_nse": val_nse,
        "val_persistence": camels.nse(r.flow[r.val], r.flow[at_val]),
        "forest_by_seed": forest_seeds,
        "forest_seeds": list(FOREST_SEEDS),
        "forest_seed_spread": float(
            max(max(v) - min(v) for by in forest_seeds.values() for v in by)
        ),
        "runner_up": runner_up,
        "chosen": chosen,
        "test": test,
        "linear_level_equals_change": bool(
            np.allclose(val_nse["linear"]["level"], val_nse["linear"]["change"], atol=1e-9)
        ),
    }


def windows_figure(sweep: dict[str, object], out_dir: Path) -> None:
    """Validation NSE against window length: linear, k-NN, forest; level and change targets."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    labels = {"linear": "linear", "knn": f"{KNN_K}-NN", "forest": "random forest"}
    for i, name in enumerate(("linear", "knn", "forest")):
        for target, ls in (("level", "-"), ("change", "--")):
            ax.plot(
                sweep["windows"],
                sweep["val_nse"][name][target],
                ls,
                color=aml_style.PALETTE[i],
                marker=aml_style.MARKERS[i],
                ms=5,
                label=f"{labels[name]}, {'flow' if target == 'level' else 'change'}",
            )
    ax.axhline(sweep["val_persistence"], color=aml_style.MUTED, lw=1, ls=":")
    ax.annotate(
        "persistence: tomorrow = today",
        (30, sweep["val_persistence"]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        color=aml_style.MUTED,
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_xticks(sweep["windows"], [str(k) for k in sweep["windows"]])
    ax.set_xlabel("window: days of rain and flow given to the model")
    ax.set_ylabel("validation NSE (WY2005-08)")
    ax.legend(ncol=3, fontsize=11, loc="lower left", frameon=False)
    aml_style.save(fig, out_dir, "windows")


def storm_days(r: River, day: str) -> np.ndarray:
    """A storm's three days: its two rainy days and the day after (where its peaks are read)."""
    t = int(np.flatnonzero(r.dates == np.datetime64(day))[0])
    return np.arange(t, t + STORM_DAYS)


def storms(r: River, uh: dict[str, object]) -> dict[str, object]:
    """The attempt: two two-day storms, their rain, the flow before, and the peak after.

    Also the unit hydrograph's peak for each (``uh`` from ``unit_hydrograph``): a filter on rain
    alone answers in proportion to the rain.
    """
    out = {}
    for day in STORMS:
        d = storm_days(r, day)
        out[day] = {
            "rain_2day": float(r.rain[d[0]] + r.rain[d[1]]),
            "flow_before": float(r.flow[d[0] - 1]),
            "peak_3day": float(np.nanmax(r.flow[d])),
            "uh_peak_3day": float(np.max(uh["sim"][d])),
            "month": str(r.dates[d[0]])[:7],
        }
    return out


def scales_figure(r: River, out_dir: Path) -> dict[str, float]:
    """Trends at several time scales (left) and smoothing and differencing a spring (right)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=aml_style.FULL, width_ratios=(1.5, 1))
    keep = r.dates >= np.datetime64("1989-10-01")
    d, q = r.dates[keep], r.flow[keep]
    a.plot(d, q + LOG_EPS, color=aml_style.PALETTE[0], lw=0.4, label="daily flow")
    year = pd.Series(q).rolling(365, center=True, min_periods=300).mean().to_numpy()
    a.plot(d, year, color=aml_style.PALETTE[1], lw=2, label="365-day mean")
    a.set_yscale("log")
    a.set_ylim(0.01, 80)
    a.set_ylabel("flow (mm/day, log)")
    a.legend(loc="lower left", fontsize=11, frameon=False, ncol=2)
    a.set_title(f"{int(keep.sum() // 365)} water years at Bismarck, IL", fontsize=13)
    z = (r.dates >= np.datetime64("2013-03-15")) & (r.dates <= np.datetime64("2013-06-15"))
    dz, qz = r.dates[z], r.flow[z]
    smooth = gaussian_filter1d(np.nan_to_num(qz), 3.0)
    change = np.diff(qz, prepend=np.nan)
    b.plot(dz, qz, color=aml_style.PALETTE[0], lw=1, marker=".", ms=3, label="daily")
    b.plot(
        dz, smooth, color=aml_style.PALETTE[1], lw=2, ls="--", label=r"Gaussian, $\sigma$ = 3 days"
    )
    b.plot(dz, change, color=aml_style.PALETTE[2], lw=1, ls="-.", label="change, [−1, 1]")
    b.axhline(0, color=aml_style.MUTED, lw=0.6)
    b.set_ylabel("mm/day")
    b.legend(loc="upper right", fontsize=11, frameon=False)
    b.set_title("Spring 2013: smooth and difference", fontsize=13)
    b.xaxis.set_major_locator(mdates.MonthLocator())
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    aml_style.save(fig, out_dir, "scales")
    month = pd.Series(q, index=pd.DatetimeIndex(d)).groupby(lambda x: x.month).mean()
    return {
        "wettest_month": int(month.idxmax()),
        "driest_month": int(month.idxmin()),
        "month_ratio": float(month.max() / month.min()),
        "largest_rise_2013": float(np.nanmax(change)),
        "years": int(keep.sum() // 365),
    }


def unit_hydrograph(r: River) -> dict[str, object]:
    """Flow today from the last ``UH_LAGS`` days of rain alone: one 1-D filter, fitted by ridge.

    The weights by lag are the unit hydrograph (Sherman, 1932): the flow a millimetre of rain
    adds 0, 1, 2, ... days later. Ridge's penalty is chosen on validation.
    """
    lags = np.arange(UH_LAGS)
    all_days = np.arange(UH_LAGS, len(r.rain))
    x_all = r.rain[all_days[:, None] - lags]

    def rows(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return r.rain[days[:, None] - lags], r.flow[days]

    xt, yt = rows(r.train)
    ok = np.isfinite(yt)
    val = {}
    for alpha in UH_ALPHAS:
        m = Ridge(alpha=alpha).fit(xt[ok], yt[ok])
        val[alpha] = camels.nse(r.flow[r.val], m.predict(rows(r.val)[0]))
    alpha = max(val, key=val.get)
    if alpha in (UH_ALPHAS[0], UH_ALPHAS[-1]):
        raise ValueError(f"ridge penalty {alpha} is at the edge of UH_ALPHAS; widen the grid")
    fit = np.concatenate([r.train, r.val])
    xf, yf = rows(fit)
    ok = np.isfinite(yf)
    m = Ridge(alpha=alpha).fit(xf[ok], yf[ok])
    sim = np.full(len(r.rain), np.nan)
    sim[all_days] = m.predict(x_all)
    return {
        "alpha": alpha,
        "val_nse": val[alpha],
        "test_nse": camels.nse(r.flow[r.test], sim[r.test]),
        "weights": m.coef_,
        "intercept": float(m.intercept_),
        "peak_lag": int(np.argmax(m.coef_)),
        "runoff_per_mm": float(m.coef_.sum()),
        "sim": sim,
    }


def unit_hydrograph_figure(r: River, uh: dict[str, object], out_dir: Path) -> None:
    """The filter's weights by lag (``uh_weights``) and a test year it simulates (``uh_year``)."""
    fig, a = plt.subplots(figsize=aml_style.SIDE)
    w = uh["weights"]
    a.bar(np.arange(UH_LAGS), w, color=aml_style.PALETTE[0], width=0.8)
    a.axhline(0, color=aml_style.MUTED, lw=0.6)
    a.set_xlim(-1, 30)
    a.set_xlabel("days since the rain")
    a.set_ylabel("mm/day of flow\nper mm of rain")
    a.set_title(f"{UH_LAGS} weights; they sum to {uh['runoff_per_mm']:.2f}", fontsize=13)
    aml_style.save(fig, out_dir, "uh_weights")
    fig, b = plt.subplots(figsize=aml_style.FULL)
    z = (r.dates >= np.datetime64("2012-06-01")) & (r.dates <= np.datetime64("2013-06-30"))
    b.plot(r.dates[z], r.flow[z], color=aml_style.INK, lw=1.2, label="observed")
    b.plot(
        r.dates[z],
        uh["sim"][z],
        color=aml_style.PALETTE[1],
        lw=1.2,
        ls="--",
        label=r"rain $\ast$ unit hydrograph",
    )
    b.set_ylabel("flow (mm/day)")
    b.legend(loc="upper left", fontsize=11, frameon=False)
    b.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for day, xy in zip(STORMS, ((-110, 45), (10, 40)), strict=True):
        d = storm_days(r, day)
        peak = d[0] + int(np.nanargmax(np.fmax(r.flow[d], uh["sim"][d])))
        rain = r.rain[d[0]] + r.rain[d[1]]
        b.annotate(
            f"{rain:.0f} mm of rain",
            (r.dates[peak], max(r.flow[peak], uh["sim"][peak])),
            xytext=xy,
            textcoords="offset points",
            fontsize=11,
            color=aml_style.INK,
            arrowprops={"arrowstyle": "->", "color": aml_style.MUTED},
        )
    b.set_title(f"Test NSE {uh['test_nse']:.2f}: the filter has no memory", fontsize=13)
    aml_style.save(fig, out_dir, "uh_year")


def monthly(r: River) -> pd.Series:
    """Monthly mean flow (mm/day), October 1989 to September 2014; a month with gaps is NaN."""
    s = pd.Series(r.flow, index=pd.DatetimeIndex(r.dates))["1989-10-01":"2014-09-30"]
    m = s.resample("MS").mean()
    m[s.isna().resample("MS").sum() > 3] = np.nan
    return m


def sarima(r: River) -> dict[str, object]:
    """SARIMA on log monthly flow: fit WY1990-2008, forecast WY2009-2014 (72 months at once).

    Also the one-month-ahead forecasts with the same parameters, and two baselines: the training
    months' mean for each calendar month (climatology), and last month (persistence).
    """
    m = np.log(monthly(r) + LOG_EPS)
    fit, test = m[:"2008-09-30"], m["2008-10-01":]
    res = SARIMAX(fit, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL).fit(disp=False)
    fc = res.get_forecast(len(test))
    band = fc.conf_int(alpha=0.10)
    one = res.apply(m).get_prediction(start=len(fit)).predicted_mean
    clim = fit.groupby(fit.index.month).mean()
    clim_test = np.array([clim[t.month] for t in test.index])

    def rmse(pred: np.ndarray) -> float:
        ok = np.isfinite(test.to_numpy()) & np.isfinite(pred)
        return float(np.sqrt(np.mean((test.to_numpy()[ok] - pred[ok]) ** 2)))

    inside = (test >= band.iloc[:, 0]) & (test <= band.iloc[:, 1])
    return {
        "params": {k: float(v) for k, v in res.params.items()},
        "rmse_log": {
            "sarima_72": rmse(fc.predicted_mean.to_numpy()),
            "sarima_1": rmse(one.to_numpy()),
            "climatology": rmse(clim_test),
            "persistence": rmse(m.shift(1)["2008-10-01":].to_numpy()),
        },
        "band90_coverage": float(inside[test.notna()].mean()),
        "series": m,
        "forecast": fc.predicted_mean,
        "band": band,
        "one_ahead": one,
    }


def sarima_figure(s: dict[str, object], out_dir: Path) -> None:
    """Log monthly flow, SARIMA's 72-month forecast with its 90% band, and its 1-month forecasts."""
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    m = s["series"]["2007-10-01":]
    ax.plot(
        m.index,
        np.exp(m) - LOG_EPS,
        color=aml_style.INK,
        lw=1.2,
        marker=".",
        ms=4,
        label="monthly mean flow",
    )
    f, band = s["forecast"], s["band"]
    ax.fill_between(
        band.index,
        np.exp(band.iloc[:, 0]) - LOG_EPS,
        np.exp(band.iloc[:, 1]) - LOG_EPS,
        color=aml_style.PALETTE[0],
        alpha=0.18,
        label="SARIMA, 72 months ahead: 90% band",
    )
    ax.plot(
        f.index,
        np.exp(f) - LOG_EPS,
        color=aml_style.PALETTE[0],
        lw=2,
        ls="--",
        label="SARIMA, 72 months ahead",
    )
    o = s["one_ahead"]
    ax.plot(
        o.index,
        np.exp(o) - LOG_EPS,
        color=aml_style.PALETTE[1],
        lw=1.2,
        ls="-.",
        marker="s",
        ms=3,
        label="SARIMA, 1 month ahead",
    )
    ax.axvline(np.datetime64("2008-10-01"), color=aml_style.MUTED, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("flow (mm/day, log)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10, frameon=False, ncol=1)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    aml_style.save(fig, out_dir, "sarima")


# --------------------------------------------------------------------------------------------
# The bridge: KW51's train passages and tracked natural frequencies
# --------------------------------------------------------------------------------------------

PASSAGES_ZIP = "traindata_201810.zip"
PASSAGE = "traindata_20181002_160753"  # 2 October 2018, 16:07: the first passage of the month
ACC_CHANNEL = "aBD17Az"  # vertical acceleration of the deck at cross girder 17 (mid-span)
STRAIN_CHANNEL = "sgBD1011A"  # deck strain between cross girders 10 and 11: the train's load
MODE = 13  # tracked mode 13 (about 6.3 Hz, a global vertical mode): cold and retrofit both move it
RETROFIT = ("2019-05-15", "2019-09-27")  # the retrofit works (Maes & Lombaert, 2021)
BASELINE = ("2018-10-01", "2018-11-30")  # autumn weeks, no frost: a first, naive reference
SINES = (1, 4, 16, 64)
SPEC3D_HZ = 8.0  # the landscape shows the modes below 8 Hz
FOURIER_SECONDS = 4.0  # length of the free-vibration piece the sums of sines are fitted to
MATLAB_EPOCH = 719_529  # datenum of 1970-01-01
MATCH_HZ = 0.1  # a spectral peak this close to an identified mode counts as that mode


@dataclass(frozen=True)
class Passage:
    """One train passage over KW51: deck acceleration and strain on one clock."""

    fs: float  # samples per second
    acc: np.ndarray  # (samples,) m/s2
    strain: np.ndarray  # (samples,) strain, the first second's median removed
    start: np.datetime64  # time of the first sample
    on: tuple[float, float]  # seconds when the train is on the bridge (from the strain)


def passage(name: str = PASSAGE) -> Passage:
    """Read one passage from ``traindata_201810.zip`` (no extraction to disk).

    The train is on the bridge while the deck strain is above 5% of its peak (from a sensor that
    the acceleration does not use).
    """
    archive = require(KW51 / PASSAGES_ZIP, "scripts/fetch_kw51.sh")
    with zipfile.ZipFile(archive) as zf, zf.open(f"traindata_201810/{name}.mat") as fh:
        m = sio.loadmat(fh, squeeze_me=True, struct_as_record=False)
    a, g = m["predat_a"], m["predat_sg"]
    acc = a.tdata[:, list(a.labels).index(ACC_CHANNEL)].astype(float)
    strain = g.tdata[:, list(g.labels).index(STRAIN_CHANNEL)].astype(float)
    fs = float(a.fs)
    strain = strain - np.median(strain[: int(fs)])
    loaded = np.flatnonzero(np.abs(strain) > 0.05 * np.abs(strain).max())
    start = np.datetime64(int(round((a.sdn[0] - MATLAB_EPOCH) * 86_400)), "s")
    return Passage(fs, acc, strain, start, (loaded[0] / fs, loaded[-1] / fs))


def tracked() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The operator's hourly natural frequencies (14 modes) and bridge temperature.

    Returns:
        Times (datetime64[s]), frequencies (hours x 14, Hz, NaN where a mode was not
        identified), and the bridge's surface temperature (deg C, sensor tBD31A).
    """
    archive = require(KW51 / "trackedmodes.zip", "scripts/fetch_kw51.sh")
    with zipfile.ZipFile(archive) as zf, zf.open("trackedmodes/trackedmodes.mat") as fh:
        m = sio.loadmat(fh, squeeze_me=True, struct_as_record=False)["modes"]
    times = np.round((m.sdn - MATLAB_EPOCH) * 86_400).astype("datetime64[s]")
    temp = m.env[:, list(m.labels_env).index("tBD31A")].astype(float)
    return times, m.f.astype(float), temp


def free_vibration(p: Passage) -> np.ndarray:
    """The acceleration from one second after the train leaves to the end of the record."""
    return p.acc[int((p.on[1] + 1.0) * p.fs) :]


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a Hann-windowed record (frequencies in Hz)."""
    w = np.hanning(len(x))
    amp = np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()
    return np.fft.rfftfreq(len(x), 1 / fs), amp


def fourier_terms(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest terms of a record's Fourier series, and the record rebuilt from them.

    Each term is one sine at a multiple of 1/T (T the record's length), with its own amplitude
    and phase; all N/2 terms rebuild the record exactly.

    Returns:
        The rebuilt record and the terms' frequency indices (multiples of 1/T), largest first.
    """
    c = np.fft.rfft(x - x.mean())
    order = np.argsort(np.abs(c))[::-1][:k]
    kept = np.zeros_like(c)
    kept[order] = c[order]
    return np.fft.irfft(kept, n=len(x)) + x.mean(), order


def peaks(f: np.ndarray, amp: np.ndarray, n: int, fmax: float = 20.0) -> np.ndarray:
    """Frequencies of the ``n`` tallest spectral peaks below ``fmax``, tallest first."""
    keep = (f > 0.3) & (f < fmax)
    idx, props = find_peaks(amp[keep], height=0)
    order = np.argsort(props["peak_heights"])[::-1][:n]
    return f[keep][idx[order]]


def fourier_figure(p: Passage, out_dir: Path) -> dict[str, object]:
    """Four seconds of free vibration rebuilt from its 1, 4, 16, and 64 largest Fourier terms."""
    piece = free_vibration(p)[: int(FOURIER_SECONDS * p.fs)]
    t = np.arange(len(piece)) / p.fs
    fig, axes = plt.subplots(len(SINES), 1, figsize=aml_style.FULL, sharex=True)
    explained, first = {}, None
    for ax, n in zip(axes, SINES, strict=True):
        fit, order = fourier_terms(piece, n)
        first = order if first is None else first
        explained[n] = float(1 - np.var(piece - fit) / np.var(piece))
        ax.plot(t, piece, color="#B0B0B0", lw=1.8)
        ax.plot(t, fit, color=aml_style.PALETTE[0], lw=1.1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(
            1.01,
            0.5,
            f"{n} sine{'s' if n > 1 else ''}: {100 * explained[n]:.0f}%",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12,
        )
    axes[-1].set_xlabel("seconds after the train has left")
    fig.subplots_adjust(hspace=0.15, right=0.84)
    aml_style.save(fig, out_dir, "fourier")
    return {
        "explained": explained,
        "first_hz": float(first[0] / FOURIER_SECONDS),
        "terms_total": len(piece) // 2,
        "seconds": FOURIER_SECONDS,
    }


def spectrum_figure(
    p: Passage, times: np.ndarray, freqs: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """The free vibration's spectrum, with the operator's modes identified in the same hour."""
    tail = free_vibration(p)
    f, amp = spectrum(tail, p.fs)
    hour = int(np.argmin(np.abs(times - p.start)))
    modes = freqs[hour][np.isfinite(freqs[hour])]
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    keep = f < 12
    ax.semilogy(f[keep], amp[keep], color=aml_style.PALETTE[0], lw=1.2)
    for fm in modes:
        ax.axvline(fm, color=aml_style.PALETTE[1], lw=1, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"amplitude (m/s$^2$)")
    ax.set_xlim(0, 12)
    matched = []
    idx, _ = find_peaks(amp)
    for fm in modes:  # the tallest local peak within MATCH_HZ of each identified mode
        near = np.flatnonzero(np.abs(f - fm) <= MATCH_HZ)
        cand = np.intersect1d(near, idx)
        if len(cand):
            k = cand[np.argmax(amp[cand])]
            matched.append({"mode": float(fm), "peak": float(f[k]), "diff": float(f[k] - fm)})
    ax.plot(
        [], [], color=aml_style.PALETTE[1], lw=1, ls="--", label="the operator's modes, same hour"
    )
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    aml_style.save(fig, out_dir, "spectrum")
    return {
        "resolution_hz": float(f[1] - f[0]),
        "modes_that_hour": modes.tolist(),
        "hour": str(times[hour]),
        "matched": matched,
        "within_bin": sum(abs(m["diff"]) <= f[1] - f[0] for m in matched),
        "within_bin_max": max(abs(m["diff"]) for m in matched if abs(m["diff"]) <= f[1] - f[0]),
        "tallest_hz": float(peaks(f, amp, 1)[0]),
        "nyquist_hz": p.fs / 2,
    }


def spectrogram_figure(p: Passage, out_dir: Path) -> dict[str, object]:
    """Strain (when the train is on) above the acceleration's spectrogram, in dB."""
    nper = int(4 * p.fs)  # 4-s windows: 0.25 Hz apart
    f, t, sxx = spectrogram(p.acc, p.fs, window="hann", nperseg=nper, noverlap=nper * 7 // 8)
    db = 10 * np.log10(sxx + 1e-12)
    keep = f <= 20
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.0, 4.2), sharex=True, height_ratios=(1, 3))
    ts = np.arange(len(p.strain)) / p.fs
    a.plot(ts, 1e6 * p.strain, color=aml_style.INK, lw=1)
    a.axvspan(*p.on, color=aml_style.PALETTE[1], alpha=0.12)
    a.set_ylabel("strain\n" + r"($\mu\varepsilon$)", fontsize=11)
    mesh = b.pcolormesh(
        t,
        f[keep],
        db[keep],
        shading="nearest",
        cmap="cividis",
        vmin=np.percentile(db[keep], 5),
        vmax=db[keep].max(),
    )
    b.set_ylabel("frequency (Hz)")
    b.set_xlabel("seconds")
    cb = fig.colorbar(mesh, ax=(a, b), shrink=0.7, pad=0.01)
    cb.set_label("power (dB)", fontsize=11)
    aml_style.save(fig, out_dir, "spectrogram")
    return {
        "window_s": nper / p.fs,
        "df_hz": float(f[1] - f[0]),
        "on_bridge_s": list(p.on),
        "record_s": len(p.acc) / p.fs,
    }


def spectrogram3d_figure(
    p: Passage, out_dir: Path, *, elev: float = 38.0, azim: float = -58.0
) -> None:
    """The spectrogram as a landscape: the train's broad hill, then ridges at the modes."""
    nper = int(4 * p.fs)
    f, t, sxx = spectrogram(p.acc, p.fs, window="hann", nperseg=nper, noverlap=nper * 7 // 8)
    keep = f <= SPEC3D_HZ
    db = gaussian_filter(10 * np.log10(sxx[keep] + 1e-12), (1.0, 1.5))
    db = np.maximum(db, np.percentile(db, 20))
    tt, ff = np.meshgrid(t, f[keep])
    fig = plt.figure(figsize=aml_style.FULL)
    ax = fig.add_subplot(projection="3d")
    aml_style.surface3d(ax, tt, ff, db)
    aml_style.style3d(ax, elev=elev, azim=azim)
    ax.set_xlabel("seconds", fontsize=11, labelpad=4)
    ax.set_ylabel("Hz", fontsize=11, labelpad=4)
    ax.set_zlabel("dB", fontsize=11, labelpad=2)
    ax.set_box_aspect((2.2, 1.4, 0.7), zoom=1.05)
    aml_style.save(fig, out_dir, "spectrogram3d")


def daily(times: np.ndarray, x: np.ndarray) -> pd.Series:
    """Daily medians of an hourly series (NaN hours ignored; days with none are dropped)."""
    return pd.Series(x, index=pd.DatetimeIndex(times)).resample("D").median().dropna()


def hinge(temp: np.ndarray) -> np.ndarray:
    """Design matrix of a line in temperature that bends at 0 deg C (frozen vs not)."""
    return np.stack([np.ones_like(temp), temp, np.minimum(temp, 0.0)], axis=1)


def alarm(times: np.ndarray, freqs: np.ndarray, temp: np.ndarray) -> dict[str, object]:
    """A frequency alarm before and after removing temperature, judged against the retrofit.

    Both alarms take the same reference rows: the pre-retrofit days with a temperature reading,
    a winter included. Naive: the daily median frequency outside those days' mean +- 3 SD.
    Corrected: the residual of a hinge regression on bridge temperature, fitted on the same
    days, outside +- 3 SD of its residuals there. False alarms are counted on the reference rows
    (in sample, for both); detection on the days after the retrofit.

    Also the autumn weeks as the reference instead (a first, naive design): each alarm's false
    alarms on the healthy days after them, the residual's band from the autumn days that have a
    temperature reading; and the residual alarm with a band as wide as the raw one's.
    """
    f = daily(times, freqs[:, MODE - 1])
    temp_d = daily(times, temp).reindex(f.index)
    days = f.index
    pre = days < RETROFIT[0]
    post = days > RETROFIT[1]
    ok = temp_d.notna().to_numpy()
    ref = pre & ok  # the reference rows of both alarms
    after = post & ok
    mu, sd = f[ref].mean(), f[ref].std(ddof=1)
    naive = ((f - mu).abs() > 3 * sd).to_numpy()
    coef, *_ = np.linalg.lstsq(hinge(temp_d[ref].to_numpy()), f[ref].to_numpy(), rcond=None)
    resid = pd.Series(np.nan, index=days)
    resid[ok] = f[ok].to_numpy() - hinge(temp_d[ok].to_numpy()) @ coef
    rsd = resid[ref].std(ddof=1)
    corrected = (resid.abs() > 3 * rsd).to_numpy()
    frozen = ref & (temp_d < 0).to_numpy()
    # The autumn weeks as the reference: each alarm's own band, and a residual band as wide as
    # the naive one's (its own rests on the few autumn days with a temperature reading)
    base = (days >= BASELINE[0]) & (days <= BASELINE[1])
    later = pre & ~base  # the healthy days after the autumn weeks: all have a temperature
    assert (later <= ok).all()
    a_mu, a_sd = f[base].mean(), f[base].std(ddof=1)
    a_rsd = resid[base & ok].std(ddof=1)
    a_naive = ((f - a_mu).abs() > 3 * a_sd).to_numpy()
    a_corrected = (resid.abs() > 3 * a_rsd).to_numpy()
    a_same_width = (resid.abs() > 3 * a_sd).to_numpy()
    winter = days.month.isin((12, 1, 2))
    return {
        "mode_hz_autumn": float(a_mu),
        "frozen_days": int(frozen.sum()),
        "frozen_shift_pct": float(100 * (f[frozen].mean() / a_mu - 1)),
        "retrofit_shift_pct": float(100 * (f[post].median() / f[pre].median() - 1)),
        "retrofit_shift_hz": float(f[post].median() - f[pre].median()),
        "frost_peak_hz": float(f[pre].max()),
        "frost_peak_month": f[pre].idxmax().strftime("%Y-%m"),
        "post_median_hz": float(f[post].median()),
        "resid_post_median_hz": float(resid[post].median()),
        "resid_pre_median_hz": float(resid[pre].median()),
        "hinge": coef.tolist(),
        "ref_days": int(ref.sum()),
        "post_days": int(after.sum()),
        "band_hz": {"naive": float(3 * sd), "corrected": float(3 * rsd)},
        "naive": {
            "false": int((naive & ref).sum()),
            "false_frozen": int((naive & frozen).sum()),
            "caught": int((naive & after).sum()),
        },
        "corrected": {
            "false": int((corrected & ref).sum()),
            "caught": int((corrected & after).sum()),
        },
        "autumn": {
            "band_hz": float(3 * a_sd),
            "later_days": int(later.sum()),
            "naive_false": int((a_naive & later).sum()),
            "naive_false_winter": int((a_naive & later & winter).sum()),
            "corrected_band_hz": float(3 * a_rsd),
            "corrected_days": int((base & ok).sum()),
            "corrected_false": int((a_corrected & later).sum()),
            "same_width_false": int((a_same_width & later).sum()),
        },
        "series": f,
        "temp": temp_d,
        "resid": resid,
        "limits": (mu, sd, rsd),
    }


def alarm_figure(al: dict[str, object], out_dir: Path) -> None:
    """Mode 13 by day with the naive alarm band; against temperature; the corrected residual.

    Top: daily medians with the naive band (the pre-retrofit days' mean +- 3 SD); bottom left:
    frequency against bridge temperature with the hinge fit; bottom right: the residual from
    that fit, with its band from the same days.
    """
    f, temp, resid = al["series"], al["temp"], al["resid"]
    mu, sd, rsd = al["limits"]
    fig = plt.figure(figsize=(9.0, 4.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1), hspace=0.55, wspace=0.28)
    a = fig.add_subplot(grid[0, :])
    post = f.index > RETROFIT[1]
    pre = f.index < RETROFIT[0]
    a.axvspan(np.datetime64(RETROFIT[0]), np.datetime64(RETROFIT[1]), color="#E6E6E6")
    a.plot(f.index, f, color=aml_style.PALETTE[0], lw=0, marker=".", ms=4)
    a.axhspan(mu - 3 * sd, mu + 3 * sd, color=aml_style.PALETTE[2], alpha=0.18)
    a.set_ylabel("Hz", fontsize=11)
    a.set_title(
        f"Mode {MODE} by day; band: before the retrofit ± 3 SD; grey: retrofit", fontsize=12
    )
    a.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    b = fig.add_subplot(grid[1, 0])
    b.plot(temp[pre], f[pre], ".", ms=4, color=aml_style.PALETTE[0], label="before")
    b.plot(temp[post], f[post], "^", ms=3, color=aml_style.PALETTE[1], label="after")
    tt = np.linspace(np.nanmin(temp), np.nanmax(temp), 100)
    b.plot(tt, hinge(tt) @ al["hinge"], color=aml_style.INK, lw=1.5)
    b.set_xlabel("bridge temperature (°C)", fontsize=11)
    b.set_ylabel("Hz", fontsize=11)
    b.legend(fontsize=10, frameon=False, loc="upper right")
    c = fig.add_subplot(grid[1, 1])
    c.axvspan(np.datetime64(RETROFIT[0]), np.datetime64(RETROFIT[1]), color="#E6E6E6")
    c.plot(resid.index, resid, color=aml_style.PALETTE[0], lw=0, marker=".", ms=3)
    c.axhspan(-3 * rsd, 3 * rsd, color=aml_style.PALETTE[2], alpha=0.18)
    c.set_ylabel("Hz above the fit", fontsize=11)
    c.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    c.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for ax_ in (a, b, c):
        ax_.tick_params(labelsize=10)
    aml_style.save(fig, out_dir, "alarm")


def series_figure(
    r: River, p: Passage, times: np.ndarray, freqs: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Three 1-D records: a river's daily flow, a bridge's acceleration, its hourly frequency."""
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 4.4))
    z = (r.dates >= np.datetime64("2012-10-01")) & (r.dates <= np.datetime64("2013-09-30"))
    axes[0].plot(r.dates[z], r.flow[z], color=aml_style.PALETTE[0], lw=1)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_ylabel("mm/day", fontsize=11)
    t = np.arange(len(p.acc)) / p.fs
    axes[1].plot(t, p.acc, color=aml_style.PALETTE[1], lw=0.4)
    axes[1].set_ylabel(r"m/s$^2$", fontsize=11)
    f = daily(times, freqs[:, MODE - 1])
    axes[2].plot(f.index, f, color=aml_style.PALETTE[2], lw=0, marker=".", ms=3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axes[2].set_ylabel("Hz", fontsize=11)
    labels = (
        "River flow at Bismarck, IL: one value a day, WY2013",
        f"KW51 deck acceleration: {p.fs:.1f} values a second, one train",
        f"KW51 mode {MODE}: one value an hour (daily medians), 15 months",
    )
    for ax, label in zip(axes, labels, strict=True):
        ax.set_title(label, fontsize=11, loc="left")
        ax.tick_params(labelsize=10)
    fig.subplots_adjust(hspace=0.75)
    aml_style.save(fig, out_dir, "series")
    return {"fs": p.fs, "passage_start": str(p.start)}


# --------------------------------------------------------------------------------------------
# Networks along time: a 1-D CNN and an LSTM, trained here in seconds on one GPU
# --------------------------------------------------------------------------------------------

INPUTS = ("prcp_mm", "tmax_c", "tmin_c", "srad_wm2", "vp_pa", "dayl_s")  # Daymet, per day
SEQ = 365  # days of input a network sees before each prediction
HIDDEN = 64
BATCH = 256
PREDICT_BATCH = 1024  # windows per forward pass when predicting (GPU memory, shared)
DROPOUT = 0.2
CLIP = 1.0  # gradient-norm clipping: rare storms give rare, large gradients
FIRST_KERNEL = 31  # days in the CNN's first filters (compare with the unit hydrograph)
DILATIONS = (1, 2, 4, 8, 16, 32, 64)  # receptive field 31 + 2 x 127 = 285 days
NET_SEEDS = (0, 1, 2, 3, 4)
FEW_SEEDS = (0, 1, 2)
SEED_LADDER = 0  # the seed of the check that the ladder's last rate fails
YEARS = (1, 2, 5, 15)  # training years for the data-size test: the last ones before WY2005
# Training budget, in gradient steps so that every amount of data gets the same:
# validation is checked every CHECK_STEPS steps, a run stops PATIENCE_STEPS after its best check
# and keeps that check's weights, and MAX_STEPS caps it. Kept runs must stop by patience. A
# 1-year run peaks near step 40: checked every 20 steps, its kept validation NSE moved by up to
# 0.06 against checks every 5 (r5 review, 2026-11-20), so checks come every 5 steps.
CHECK_STEPS = 5
PATIENCE_STEPS = 600
MAX_STEPS = 12_000
# Adam's learning rate, bracketed on validation for every network: start
# from LR_START and widen along LR_LADDER until the best rate is interior. The ladder ends at a
# rate where training fails (checked in ``ladder_end``: with gradients clipped, the LSTMs still
# train at 1e-1), so a search can never run off it.
LR_LADDER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
LR_START = (1e-3, 3e-3, 1e-2)


@dataclass(frozen=True)
class Task:
    """What a network sees and predicts.

    Attributes:
        use_flow: Past flow is an input (forecasting) or not (simulating from weather alone).
        horizon: Days from the window's last day to the target day.
        tomorrow_rain: One more input channel, the rain of the day after each window day (on
            the window's last day, the target day's rain: a perfect rain forecast).
    """

    use_flow: bool
    horizon: int
    tomorrow_rain: bool


SIMULATE = Task(use_flow=False, horizon=0, tomorrow_rain=False)
FORECAST = Task(use_flow=True, horizon=1, tomorrow_rain=False)
FORECAST_RAIN = Task(use_flow=True, horizon=1, tomorrow_rain=True)


@dataclass(frozen=True)
class Basins:
    """Inputs and targets of one or more basins on the GPU, standardized on training days."""

    x: torch.Tensor  # (basins, days, channels); NaN where an input is unknown
    y: torch.Tensor  # (basins, days) flow / the basin's training SD; NaN where missing
    scale: np.ndarray  # (basins,) training SD of flow, mm/day
    flow: np.ndarray  # (basins, days) mm/day
    weather_sd: np.ndarray  # (len(INPUTS),) training SD of each weather input, float64
    unknown: np.ndarray  # (basins, days + 1) running count of days with an unknown input
    task: Task
    gauges: tuple[str, ...]


def device() -> torch.device:
    """The GPU if there is one."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def basins(cam: camels.Camels, gauges: list[str], task: Task) -> Basins:
    """Stack the basins' weather (and past flow, and tomorrow's rain) as network inputs.

    Weather is standardized with the training days' mean and SD pooled over the basins; flow is
    divided by each basin's own training SD, so the squared error weighs every basin alike (the
    NSE's denominator). Missing past flow is carried forward from the last observed day, never
    back from a later one; days before a basin's first reading stay unknown, and no window that
    contains an unknown input is used.
    """
    rows = [cam.basin(g) for g in gauges]
    tr = cam.period("train")
    chans = [cam.channel(c) for c in INPUTS]
    w = cam.forcing[rows][:, :, chans].astype(np.float64)
    mu, sd = w[:, tr].reshape(-1, len(chans)).mean(0), w[:, tr].reshape(-1, len(chans)).std(0)
    w = (w - mu) / sd
    flow = cam.flow[rows].astype(np.float64)
    scale = np.nanstd(flow[:, tr], axis=1)
    y = flow / scale[:, None]
    parts = [w]
    if task.use_flow:
        parts.append(pd.DataFrame(y.T).ffill().to_numpy().T[:, :, None])
    if task.tomorrow_rain:
        rain = w[:, :, INPUTS.index("prcp_mm")]
        nxt = np.concatenate([rain[:, 1:], np.full((len(rows), 1), np.nan)], axis=1)
        parts.append(nxt[:, :, None])
    x = np.concatenate(parts, axis=2)
    n_in = len(INPUTS) + task.use_flow + task.tomorrow_rain
    assert x.shape == (len(rows), len(cam.dates), n_in), x.shape
    bad = ~np.isfinite(x).all(axis=2)
    dev = device()
    return Basins(
        x=torch.tensor(x, dtype=torch.float32, device=dev),
        y=torch.tensor(y, dtype=torch.float32, device=dev),
        scale=scale,
        flow=flow,
        weather_sd=sd,
        unknown=np.concatenate([np.zeros((len(rows), 1), int), np.cumsum(bad, axis=1)], axis=1),
        task=task,
        gauges=tuple(gauges),
    )


def samples(data: Basins, days: np.ndarray, seq: int) -> torch.Tensor:
    """(basin, last input day) pairs whose target day is in ``days`` and observed.

    A window qualifies only if every one of its inputs is known.
    """
    h = data.task.horizon
    last = days - h
    last = last[last - seq + 1 >= 0]
    b, t = np.meshgrid(np.arange(data.x.shape[0]), last, indexing="ij")
    b, t = b.ravel(), t.ravel()
    known = data.unknown[b, t + 1] - data.unknown[b, t - seq + 1] == 0
    ok = known & np.isfinite(data.y[b, t + h].cpu().numpy())
    return torch.tensor(np.stack([b[ok], t[ok]], axis=1), device=data.x.device)


def gather(data: Basins, pairs: torch.Tensor, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Input windows (batch, seq, channels) and targets (batch,) for (basin, last day) pairs."""
    offsets = torch.arange(seq - 1, -1, -1, device=pairs.device)
    days = pairs[:, 1:2] - offsets  # (batch, seq): oldest day first
    x = data.x[pairs[:, :1], days]
    y = data.y[pairs[:, 0], pairs[:, 1] + data.task.horizon]
    return x, y


class LSTMNet(nn.Module):
    """An LSTM over the window, then a linear read-out of its last hidden state."""

    def __init__(self, n_in: int, hidden: int = HIDDEN) -> None:
        """One LSTM layer of ``hidden`` cells over ``n_in`` input channels."""
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, channels) -> (batch,)."""
        h, _ = self.lstm(x)
        return self.head(self.drop(h[:, -1])).squeeze(-1)


class CNN1d(nn.Module):
    """A 1-D CNN: 16 filters a month long, then dilated filters; the read-out at the last day."""

    def __init__(self, n_in: int, width: int = 32) -> None:
        """The layers, for ``n_in`` input channels and ``width`` filters after the first."""
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(n_in, 16, FIRST_KERNEL), nn.ReLU()]
        c = 16
        for d in DILATIONS:
            layers += [nn.Conv1d(c, width, 3, dilation=d), nn.ReLU()]
            c = width
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, channels) -> (batch,): channels first for Conv1d, the last position out."""
        z = self.net(x.transpose(1, 2))  # (batch, width, seq - 285 + 1)
        return self.head(self.drop(z[:, :, -1])).squeeze(-1)


@dataclass
class Run:
    """A trained network and its scores."""

    model: nn.Module
    lr: float
    val_nse: float  # median over the basins, at the kept check
    test_nse: np.ndarray  # one per basin
    best_step: int  # the gradient step whose weights were kept (validation's choice)
    steps: int  # gradient steps taken
    stopped: bool  # stopped by patience (True) or by MAX_STEPS
    seconds: float


def predict(model: nn.Module, data: Basins, days: np.ndarray, seq: int) -> np.ndarray:
    """Predicted flow (mm/day), basins x len(days); NaN where a day cannot be predicted.

    A day can be predicted when its target is observed and its window's inputs are known.
    """
    model.eval()
    out = np.full((data.x.shape[0], len(days)), np.nan)
    pairs = samples(data, days, seq)
    pos = {int(d): i for i, d in enumerate(days)}
    with torch.no_grad():
        for i in range(0, len(pairs), PREDICT_BATCH):
            p = pairs[i : i + PREDICT_BATCH]
            x, _ = gather(data, p, seq)
            yh = model(x).cpu().numpy()
            pb, pt = p[:, 0].cpu().numpy(), p[:, 1].cpu().numpy() + data.task.horizon
            out[pb, [pos[int(t)] for t in pt]] = yh * data.scale[pb]
    return out


def scores(model: nn.Module, data: Basins, days: np.ndarray, seq: int) -> np.ndarray:
    """NSE of each basin over ``days``; -inf for a basin whose model output is not finite.

    A diverged network outputs NaN or huge values: it scores -inf, so it is never kept.
    """
    pred = predict(model, data, days, seq)
    expected = np.bincount(samples(data, days, seq)[:, 0].cpu().numpy(), minlength=len(pred))
    out = []
    for b in range(len(pred)):
        good = np.isfinite(pred[b]) & (np.abs(np.nan_to_num(pred[b])) < 1e6)
        ok = good.sum() == expected[b] and expected[b] >= 2
        out.append(camels.nse(data.flow[b, days], pred[b]) if ok else -np.inf)
    return np.array(out)


def train(
    data: Basins,
    kind: str,
    seed: int,
    lr: float,
    *,
    days: tuple[np.ndarray, np.ndarray, np.ndarray],
    seq: int = SEQ,
) -> Run:
    """Train a network with Adam on the squared error of scaled flow, with early stopping.

    Validation NSE (median over the basins) is checked every ``CHECK_STEPS`` gradient steps,
    the same for any amount of training data; the run stops ``PATIENCE_STEPS`` after its best
    check, or at ``MAX_STEPS``, and keeps the best check's weights.

    Args:
        data: The basins.
        kind: ``lstm`` or ``cnn``.
        seed: Seeds the weights, dropout, and batch order.
        lr: Adam's learning rate.
        days: Target days of training, validation, and test.
        seq: Days of input per prediction.
    """
    train_days, val_days, test_days = days
    seed_all(seed)
    t0 = time.perf_counter()
    n_in = data.x.shape[2]
    model = (LSTMNet(n_in) if kind == "lstm" else CNN1d(n_in)).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pairs = samples(data, train_days, seq)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_step, step, stopped = -np.inf, 0, 0, False
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    while not stopped and step < MAX_STEPS:
        order = torch.randperm(len(pairs), generator=gen).to(pairs.device)
        for i in range(0, len(pairs), BATCH):
            model.train()
            x, y = gather(data, pairs[order[i : i + BATCH]], seq)
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            step += 1
            if step % CHECK_STEPS == 0:
                val = float(np.median(scores(model, data, val_days, seq)))
                if val > best:
                    best, best_step = val, step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                elif step - best_step >= PATIENCE_STEPS:
                    stopped = True
            if stopped or step >= MAX_STEPS:
                break
    model.load_state_dict(best_state)
    test = scores(model, data, test_days, seq)
    return Run(model, lr, best, test, best_step, step, stopped, time.perf_counter() - t0)


_RUNS: dict[tuple[object, ...], Run] = {}  # every run of this script, by what defines it


def run(
    data: Basins, kind: str, seed: int, lr: float, *, cam: camels.Camels, train_days: np.ndarray
) -> Run:
    """One training run on the standard validation and test years, trained once per setting.

    The same setting (basins, task, network, seed, rate, training days) always gives the same
    run, so a run that two comparisons share (the 15-year forecaster of the data-size test, the
    home basin among the fifteen) is trained once and reused.
    """
    key = (data.gauges, data.task, kind, seed, lr, int(train_days[0]), len(train_days))
    if key not in _RUNS:
        days = (train_days, cam.period("val"), cam.period("test"))
        _RUNS[key] = train(data, kind, seed, lr, days=days)
        r = _RUNS[key]
        log(
            f"  {kind} {','.join(data.gauges) if len(data.gauges) < 3 else len(data.gauges)} "
            f"seed {seed} lr {lr:g}: val {r.val_nse:.4f} at step {r.best_step} of {r.steps}"
            f"{'' if r.stopped else ' (cap)'}, {r.seconds:.0f} s"
        )
    return _RUNS[key]


def bracket(score: Callable[[float], float]) -> tuple[float, dict[float, float]]:
    """The learning rate with the highest ``score``, bracketed along ``LR_LADDER``.

    Starts from ``LR_START``; while the best rate is the lowest or the highest tried, the next
    rate of the ladder past it is tried too. The kept rate is asserted interior, so
    no caller can skip the check.

    Raises:
        ValueError: If the best rate is at an end of the ladder.
    """
    tried: dict[float, float] = {}
    todo = list(LR_START)
    while todo:
        lr = todo.pop(0)
        tried[lr] = score(lr)
        best = max(tried, key=tried.get)
        if not todo and best in (min(tried), max(tried)):
            i = LR_LADDER.index(best) + (-1 if best == min(tried) else 1)
            if not 0 <= i < len(LR_LADDER):
                raise ValueError(f"the best rate {best:g} ends LR_LADDER; widen it")
            todo.append(LR_LADDER[i])
    best = max(tried, key=tried.get)
    assert min(tried) < best < max(tried), (best, tried)  # interior
    return best, tried


@dataclass
class Tuned:
    """The runs at the learning rate validation chose, and the record of the search."""

    runs: list[Run]  # one per seed (several basins: their order in ``gauges``, seeds inside)
    record: dict[str, object]


def tune(
    datas: list[Basins],
    kind: str,
    seeds: tuple[int, ...],
    cam: camels.Camels,
    *,
    train_days: np.ndarray | None = None,
) -> Tuned:
    """Bracket one learning rate for a set of models on validation; return its runs.

    ``datas`` are trained separately with a shared rate (one model per basin) or hold one
    pooled model. A rate's score is the median, over ``datas``, of the median over seeds of the
    validation NSE. Every run at the kept rate and at both neighbours must stop by patience,
    so the bracket is not an artefact of the step cap, and every run at
    the kept rate must have its best check after the first, so the check spacing did not decide
    its score. A rate whose runs diverged scores -inf; its record stores None (JSON has no
    infinity).

    Args:
        datas: The basins, one model each, or one pooled model.
        kind: ``lstm`` or ``cnn``.
        seeds: The seeds of each rate's runs.
        cam: CAMELS.
        train_days: Target days of training (default: the training period).

    Returns:
        The runs at the kept rate and the search's record.
    """
    tr = cam.period("train") if train_days is None else train_days
    by_rate: dict[float, list[Run]] = {}

    def score(lr: float) -> float:
        by_rate[lr] = [run(d, kind, s, lr, cam=cam, train_days=tr) for d in datas for s in seeds]
        per_data = np.reshape([r.val_nse for r in by_rate[lr]], (len(datas), len(seeds)))
        return float(np.median(np.median(per_data, axis=1)))

    lr, tried = bracket(score)
    rates = sorted(tried)
    near = rates[rates.index(lr) - 1 : rates.index(lr) + 2]
    capped = [(r.lr, r.best_step) for q in near for r in by_rate[q] if not r.stopped]
    assert not capped, f"runs that hit MAX_STEPS near the kept rate {lr:g}: {capped}"
    first = [r.best_step for r in by_rate[lr] if r.best_step <= CHECK_STEPS]
    assert not first, f"kept runs at rate {lr:g} whose best check is the first: {first}"
    record = {
        "lr": lr,
        "tried": {
            f"{q:g}": {
                "val": tried[q] if np.isfinite(tried[q]) else None,  # a diverged rate: -inf
                "best_steps": [r.best_step for r in by_rate[q]],
                "steps": [r.steps for r in by_rate[q]],
            }
            for q in rates
        },
    }
    return Tuned(by_rate[lr], record)


def ladder_end(cam: camels.Camels) -> dict[str, object]:
    """Training at the ladder's last rate fails: validation NSE below 0 (worse than the mean).

    One seed of each network on the home basin (the forecaster with past flow too); checks
    the premise that the ladder ends at a rate the method cannot use. A run whose
    output is not finite scores -inf (stored as None).
    """
    out = {}
    for name, task, kind in (
        ("lstm_sim", SIMULATE, "lstm"),
        ("cnn_sim", SIMULATE, "cnn"),
        ("lstm_forecast", FORECAST, "lstm"),
    ):
        data = basins(cam, [camels.HOME], task)
        r = run(data, kind, SEED_LADDER, LR_LADDER[-1], cam=cam, train_days=cam.period("train"))
        assert r.val_nse < 0, (name, r.val_nse)
        out[name] = r.val_nse if np.isfinite(r.val_nse) else None
    return {"lr": LR_LADDER[-1], "val_nse": out}


def lstm_scratch(x: torch.Tensor, lstm: nn.LSTM) -> tuple[torch.Tensor, torch.Tensor]:
    """An LSTM layer written out, with nn.LSTM's weights (gate order i, f, g, o).

    Args:
        x: (seq, channels) one window.
        lstm: The layer whose weights to use (one layer, batch_first).

    Returns:
        Hidden states (seq, hidden) and cell states (seq, hidden).
    """
    w_ih, w_hh = lstm.weight_ih_l0, lstm.weight_hh_l0  # (4H, C), (4H, H)
    b = lstm.bias_ih_l0 + lstm.bias_hh_l0  # (4H,)
    n = lstm.hidden_size
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    c = torch.zeros_like(h)
    hs, cs = [], []
    for x_t in x:
        z = w_ih @ x_t + w_hh @ h + b  # (4H,)
        i, f = torch.sigmoid(z[:n]), torch.sigmoid(z[n : 2 * n])
        g, o = torch.tanh(z[2 * n : 3 * n]), torch.sigmoid(z[3 * n :])
        c = f * c + i * g  # keep part of the memory, add part of the new candidate
        h = o * torch.tanh(c)  # show part of it
        hs.append(h)
        cs.append(c)
    return torch.stack(hs), torch.stack(cs)


def verify_lstm(model: LSTMNet, data: Basins, day: int) -> dict[str, float]:
    """The written-out LSTM against nn.LSTM, both in float64 on the CPU, on one real window."""
    lstm = nn.LSTM(model.lstm.input_size, model.lstm.hidden_size, batch_first=True)
    lstm.load_state_dict(model.lstm.state_dict())
    lstm = lstm.double().cpu()
    x = data.x[0, day - SEQ + 1 : day + 1].double().cpu()  # (SEQ, channels)
    with torch.no_grad():
        h_ref, (_, c_ref) = lstm(x[None])
        h, c = lstm_scratch(x, lstm)
    return {
        "max_abs_h": float((h - h_ref[0]).abs().max()),
        "max_abs_c": float((c[-1] - c_ref[0, 0]).abs().max()),
        "weights": int(sum(p.numel() for p in model.parameters())),
    }


def summary(t: Tuned) -> dict[str, object]:
    """Median and range of test NSE over seeds (each run's median over its basins), the rate."""
    rs = t.runs
    s = np.array([float(np.median(r.test_nse)) for r in rs])
    return {
        "test_median": float(np.median(s)),
        "test_min": float(s.min()),
        "test_max": float(s.max()),
        "test": s.tolist(),
        "best_steps": [r.best_step for r in rs],
        "seconds": float(np.mean([r.seconds for r in rs])),
        **t.record,
    }


def median_run(rs: list[Run]) -> Run:
    """The run whose test score is the median of the seeds' (for the figures)."""
    t = [float(np.median(r.test_nse)) for r in rs]
    return rs[int(np.argsort(t)[len(t) // 2])]


def data_size(cam: camels.Camels, r: River, chosen: Setting, full: Tuned) -> dict[str, object]:
    """Test NSE of tomorrow's flow against years of training record: LSTM vs forest.

    Each uses the last ``n`` water years before WY2005 for training, the same validation years,
    and the same test years. The LSTM's rate is bracketed again at every length; the forest
    keeps the setting validation chose on 15 years. The 15-year LSTM is the forecaster's own
    runs (``full``). At every length the LSTM's inputs are standardized with the 15 training
    years' statistics (``basins``): a little information a 1-year model would not have.
    """
    data = basins(cam, [camels.HOME], FORECAST)
    tr = cam.period("train")
    out: dict[str, dict[int, object]] = {"lstm": {}, "forest": {}}
    for n in YEARS:
        days = tr[cam.dates[tr] >= np.datetime64(f"{2004 - n}-10-01")]
        assert (n == 15) == (len(days) == len(tr))
        t = full if n == 15 else tune([data], "lstm", FEW_SEEDS, cam, train_days=days)
        out["lstm"][n] = summary(t)
        out["forest"][n] = camels.nse(
            r.flow[r.test], chosen_forecast(r, chosen, issued(days), issued(r.test))
        )
    return out


def regional(cam: camels.Camels) -> dict[str, object]:
    """One LSTM for the 15 Illinois basins against one LSTM per basin (simulating flow).

    The same seeds on both sides; each side's rate bracketed on validation (one shared rate
    for the fifteen single-basin models).
    """
    gauges = camels.illinois(cam)
    pooled = tune([basins(cam, gauges, SIMULATE)], "lstm", FEW_SEEDS, cam)
    single = tune([basins(cam, [g], SIMULATE) for g in gauges], "lstm", FEW_SEEDS, cam)
    pooled_nse = np.median(np.stack([r.test_nse for r in pooled.runs]), axis=0)  # per basin
    each = np.reshape([r.test_nse[0] for r in single.runs], (len(gauges), len(FEW_SEEDS)))
    each = np.median(each, axis=1)  # per basin, over the same seeds
    return {
        "gauges": gauges,
        "pooled_per_basin": pooled_nse.tolist(),
        "single_per_basin": each.tolist(),
        "pooled_median": float(np.median(pooled_nse)),
        "single_median": float(np.median(each)),
        "pooled_better": int((pooled_nse > each).sum()),
        "pooled_seconds": float(np.mean([r.seconds for r in pooled.runs])),
        "pooled_rate": pooled.record,
        "single_rate": single.record,
    }


def water_year(dates: np.ndarray) -> np.ndarray:
    """The water year of each date: 1 October to 30 September, named by the year it ends."""
    year = dates.astype("datetime64[Y]").astype(int) + 1970
    month = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return year + (month >= 10)


def random_split(r: River, chosen: Setting) -> dict[str, object]:
    """The forest scored on random days, and on random whole years, of the same 25 years.

    Ten draws of random test days (as many as the test years have), and ten draws of random
    whole water years (as many as the test years: no test day has a neighbour in training),
    against the later years (WY2009-2014); for the chosen forest (it sees today's flow) and for
    the same forest on rain alone, to tomorrow (no flow input: it simulates). Every split is by
    target day. If random years score like random days, the gap to the later years is the later
    years themselves (harder: the 2012 drought), not neighbours leaking into training.
    """
    targets = np.concatenate([r.train, r.val, r.test])  # every target day of WY1990-2014
    days = issued(targets)
    fit_t, at_test = issued(np.concatenate([r.train, r.val])), issued(r.test)
    wy = water_year(r.dates[targets])
    years = np.unique(wy)
    n_test_years = len(np.unique(water_year(r.dates[r.test])))
    assert (len(years), n_test_years) == (25, 6), (len(years), n_test_years)
    out: dict[str, object] = {"years": len(years), "test_years": n_test_years}
    for name, setting, kw in (
        ("with_flow", chosen, {}),
        ("rain_only", replace(chosen, target="level"), {"flow": False, "tomorrow_rain": True}),
    ):
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(SPLIT_DRAWS):
            test = np.sort(rng.choice(days, size=len(r.test), replace=False))
            fit = np.setdiff1d(days, test)
            draws.append(camels.nse(r.flow[test + 1], chosen_forecast(r, setting, fit, test, **kw)))
        rng = np.random.default_rng(SEED)
        year_draws = []
        for _ in range(SPLIT_DRAWS):
            held = np.isin(wy, rng.choice(years, size=n_test_years, replace=False))
            test_t, fit_t_years = targets[held], targets[~held]
            f = chosen_forecast(r, setting, issued(fit_t_years), issued(test_t), **kw)
            year_draws.append(camels.nse(r.flow[test_t], f))
        out[name] = {
            "random_days": draws,
            "random_median": float(np.median(draws)),
            "random_min": float(min(draws)),
            "random_max": float(max(draws)),
            "random_years": year_draws,
            "random_years_median": float(np.median(year_draws)),
            "random_years_min": float(min(year_draws)),
            "random_years_max": float(max(year_draws)),
            "later_years": camels.nse(
                r.flow[r.test], chosen_forecast(r, setting, fit_t, at_test, **kw)
            ),
        }
    return out


def rain_response(
    run_: Run, data: Basins, cam: camels.Camels, day: str, seq: int = SEQ
) -> np.ndarray:
    """The LSTM's own unit hydrograph on one day: d(flow that day) / d(rain k days before).

    Gradients of the simulated flow (mm/day) with respect to each past day's rain (mm), lag 0
    first: how much a millimetre more rain k days ago would have added, given everything else
    the network saw. The rain's training SD (float64) undoes its standardization.
    """
    t = int(np.flatnonzero(cam.dates == np.datetime64(day))[0])
    k = INPUTS.index("prcp_mm")
    x = data.x[0, t - seq + 1 : t + 1].clone()[None].requires_grad_(True)  # (1, seq, channels)
    run_.model.eval()
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN's RNN has no backward in eval mode
        y = run_.model(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    return grad[0, :, k].cpu().numpy()[::-1] * data.scale[0] / data.weather_sd[k]  # lag 0 first


RESPONSE_DAYS = {"wet": "2013-04-11", "dry": "2012-08-27"}  # the LSTM's response, two days


def response_figure(uh: dict[str, object], resp: dict[str, np.ndarray], out_dir: Path) -> None:
    """The fixed filter's weights against the LSTM's response on a wet and a dry day.

    ``resp`` maps each day of ``RESPONSE_DAYS`` to its response; the labels come from there.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    lags = np.arange(31)
    ax.plot(
        lags,
        uh["weights"][:31],
        color=aml_style.MUTED,
        lw=2,
        ls=":",
        marker="o",
        ms=4,
        label="the filter (unit hydrograph): every day",
    )
    for i, (tag, day) in enumerate(RESPONSE_DAYS.items()):
        g, label = resp[day], f"{tag}: {pd.Timestamp(day):%-d %b %Y}"
        ax.plot(
            lags,
            g[:31],
            color=aml_style.PALETTE[i],
            lw=2.2,
            ls=aml_style.LINESTYLES[i],
            marker=aml_style.MARKERS[i],
            ms=4,
            label=f"LSTM, {label}",
        )
    ax.axhline(0, color=aml_style.MUTED, lw=0.6)
    ax.set_xlabel("days since the rain")
    ax.set_ylabel("mm/day of flow\nper mm of rain")
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    aml_style.save(fig, out_dir, "response")


def hydrograph_figure(
    r: River,
    uh: dict[str, object],
    sims: dict[str, np.ndarray],
    nses: dict[str, float],
    out_dir: Path,
) -> None:
    """A test year simulated from weather alone: the filter, the 1-D CNN, and the LSTM."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    z = (r.dates >= np.datetime64("2012-10-01")) & (r.dates <= np.datetime64("2013-09-30"))
    ax.plot(r.dates[z], r.flow[z], color=aml_style.INK, lw=1.6, label="observed")
    series = {"unit hydrograph": uh["sim"], **sims}
    for i, (name, sim) in enumerate(series.items()):
        ax.plot(
            r.dates[z],
            sim[z],
            color=aml_style.PALETTE[i + 1],
            lw=1.2,
            ls=aml_style.LINESTYLES[i + 1],
            label=f"{name}: test NSE {nses[name]:.2f}",
        )
    ax.xaxis.set_major_locator(mdates.MonthLocator((10, 1, 4, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("flow (mm/day)")
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    aml_style.save(fig, out_dir, "hydrograph")


# --------------------------------------------------------------------------------------------
# A pretrained forecaster: Chronos-2, used as is
# --------------------------------------------------------------------------------------------

CHRONOS = "amazon/chronos-2"  # Ansari et al. (2025); Apache-2.0
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CONTEXTS = (256, 512, 1024, 2048)  # days of history given to it, chosen on validation
EMBED_DAYS = 2048  # each basin's flow before 1 October 2004, embedded once
CLUSTERS = 5
PROBE_ALPHAS = tuple(10.0**k for k in range(-2, 6))


def chronos() -> BaseChronosPipeline:
    """Chronos-2 from the Hugging Face Hub at a fixed revision (cached after the first run)."""
    return BaseChronosPipeline.from_pretrained(
        CHRONOS, revision=CHRONOS_REVISION, device_map=str(device())
    )


def chronos_forecast(
    pipe: BaseChronosPipeline,
    r: River,
    days: np.ndarray,
    context: int,
    *,
    rain: bool,
    tomorrow_rain: bool = False,
    levels: tuple[float, ...] = (0.5,),
) -> np.ndarray:
    """Tomorrow's flow from the last ``context`` days of flow (and rain), no training.

    Returns:
        Quantile forecasts of flow t+1, shape (len(days), len(levels)).
    """
    inputs = []
    for t in days:
        item: dict[str, object] = {"target": r.flow[t - context + 1 : t + 1].astype(np.float32)}
        if rain:
            item["past_covariates"] = {"rain": r.rain[t - context + 1 : t + 1].astype(np.float32)}
        if tomorrow_rain:
            item["future_covariates"] = {"rain": r.rain[t + 1 : t + 2].astype(np.float32)}
        inputs.append(item)
    with torch.no_grad():
        q, _ = pipe.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(levels))
    return np.stack([qi[0, 0].float().cpu().numpy() for qi in q])


def chronos_scores(pipe: BaseChronosPipeline, r: River) -> dict[str, object]:
    """Zero-shot NSE: context chosen on validation; test with past rain, then tomorrow's too.

    Raises:
        ValueError: If the chosen context is at the edge of ``CONTEXTS``.
    """
    val = {
        c: camels.nse(r.flow[r.val], chronos_forecast(pipe, r, issued(r.val), c, rain=True)[:, 0])
        for c in CONTEXTS
    }
    best = max(val, key=val.get)
    if best in (CONTEXTS[0], CONTEXTS[-1]):
        raise ValueError(f"the chosen context {best} is at the edge of CONTEXTS; widen it")
    out = {"context": best, "val_nse": val, "test": {}}
    for name, kw in (
        ("flow", {"rain": False}),
        ("rain", {"rain": True}),
        ("tomorrow_rain", {"rain": True, "tomorrow_rain": True}),
    ):
        out["test"][name] = camels.nse(
            r.flow[r.test], chronos_forecast(pipe, r, issued(r.test), best, **kw)[:, 0]
        )
    return out


def chronos_event_figure(
    pipe: BaseChronosPipeline, r: River, context: int, forest: np.ndarray, out_dir: Path
) -> dict[str, float]:
    """The April 2013 flood, forecast a day ahead: Chronos-2's 10-90% band and median."""
    z = np.flatnonzero(
        (r.dates >= np.datetime64("2013-03-25")) & (r.dates <= np.datetime64("2013-05-10"))
    )
    q = chronos_forecast(pipe, r, issued(z), context, rain=True, levels=(0.1, 0.5, 0.9))
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    d = r.dates[z]
    ax.fill_between(
        d,
        q[:, 0],
        q[:, 2],
        color=aml_style.PALETTE[0],
        alpha=0.2,
        label="Chronos-2, 10-90%: no training",
    )
    ax.plot(d, q[:, 1], color=aml_style.PALETTE[0], lw=2, ls="--", label="Chronos-2, median")
    ax.plot(
        d,
        forest,
        color=aml_style.PALETTE[2],
        lw=1.6,
        ls="-.",
        marker="^",
        ms=4,
        label="forest, trained on 15 years",
    )
    ax.plot(d, r.flow[z], color=aml_style.INK, lw=1.6, marker=".", ms=5, label="observed")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("flow (mm/day)")
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    aml_style.save(fig, out_dir, "chronos_event")
    inside = (r.flow[z] >= q[:, 0]) & (r.flow[z] <= q[:, 2])
    return {"band_covers": float(inside.mean()), "days": len(z)}


def embeddings(pipe: BaseChronosPipeline, cam: camels.Camels) -> np.ndarray:
    """Each basin's flow, embedded by Chronos-2: the mean of its patch tokens (basins x 768)."""
    end = int(np.flatnonzero(cam.dates == np.datetime64("2004-09-30"))[0])
    series = [
        cam.flow[b, end - EMBED_DAYS + 1 : end + 1].astype(np.float32)
        for b in range(len(cam.gauge_id))
    ]
    with torch.no_grad():
        emb, _ = pipe.embed(series, batch_size=64)
    patches = EMBED_DAYS // pipe.model.chronos_config.input_patch_size
    return np.stack([e[0, :patches].float().mean(0).cpu().numpy() for e in emb])


def signatures(cam: camels.Camels) -> np.ndarray:
    """Four flow signatures a hydrologist would compute from the same days (the probe baseline).

    When in the year the flow comes (sine and cosine of its centroid), its variability, and its
    day-to-day persistence.
    """
    end = int(np.flatnonzero(cam.dates == np.datetime64("2004-09-30"))[0])
    days = np.arange(end - EMBED_DAYS + 1, end + 1)
    doy = 2 * np.pi * (cam.dates[days].astype("datetime64[D]").astype(int) % 365.25) / 365.25
    rows = []
    for b in range(len(cam.gauge_id)):
        q = cam.flow[b, days].astype(float)
        ok = np.isfinite(q)
        w = q[ok] / q[ok].sum()
        lq = np.log(q[ok] + LOG_EPS)
        rows.append(
            [
                np.sum(w * np.sin(doy[ok])),
                np.sum(w * np.cos(doy[ok])),
                q[ok].std() / q[ok].mean(),
                np.corrcoef(lq[:-1], lq[1:])[0, 1],
            ]
        )
    return np.array(rows)


def probe(features: np.ndarray, target: np.ndarray, regions: np.ndarray) -> float:
    """R2 of a ridge probe, each HUC-02 region held out in turn (neighbours never in training)."""
    pred = np.empty_like(target)
    for reg in np.unique(regions):
        held = regions == reg
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=PROBE_ALPHAS))
        model.fit(features[~held], target[~held])
        pred[held] = model.predict(features[held])
    return float(1 - np.sum((pred - target) ** 2) / np.sum((target - target.mean()) ** 2))


def basin_map(emb: np.ndarray, cam: camels.Camels, out_dir: Path) -> dict[str, object]:
    """k-means on the embeddings (Lecture 4), mapped; probes for snow and aridity."""
    z = PCA(20, random_state=SEED).fit_transform(StandardScaler().fit_transform(emb))
    labels = KMeans(CLUSTERS, n_init=20, random_state=SEED).fit_predict(z)
    a = cam.attrs
    order = np.argsort([a["frac_snow"][labels == k].median() for k in range(CLUSTERS)])
    rank = np.empty(CLUSTERS, int)
    rank[order] = np.arange(CLUSTERS)  # clusters numbered by snow, least first
    labels = rank[labels]
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    for k in range(CLUSTERS):
        m = labels == k
        ax.scatter(
            a["gauge_lon"][m],
            a["gauge_lat"][m],
            s=14,
            marker=aml_style.MARKERS[k % 4] if k < 4 else "v",
            color=[*aml_style.PALETTE, aml_style.NAVY][k],
            label=f"group {k + 1}: {m.sum()} basins",
        )
    home = cam.basin(camels.HOME)
    ax.scatter(
        a["gauge_lon"][home],
        a["gauge_lat"][home],
        s=160,
        facecolor="none",
        edgecolor=aml_style.INK,
        lw=1.5,
    )
    ax.set_aspect(1.25)
    ax.set_axis_off()
    ax.legend(
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    aml_style.save(fig, out_dir, "basin_map")
    regions = a["huc_02"].to_numpy()
    sig = signatures(cam)
    groups = {
        k + 1: {
            "basins": int((labels == k).sum()),
            "frac_snow": float(a["frac_snow"][labels == k].median()),
            "aridity": float(a["aridity"][labels == k].median()),
            "baseflow_index": float(a["baseflow_index"][labels == k].median()),
        }
        for k in range(CLUSTERS)
    }
    return {
        "groups": groups,
        "home_group": int(labels[home] + 1),
        "probe": {
            t: {
                "embedding": probe(emb, a[t].to_numpy(), regions),
                "signatures": probe(sig, a[t].to_numpy(), regions),
            }
            for t in ("frac_snow", "aridity")
        },
        "basins": len(labels),
        "dims": emb.shape[1],
    }


# --------------------------------------------------------------------------------------------
# The one picture, the decision, the title art
# --------------------------------------------------------------------------------------------

PICTURE_DAY = "2013-04-11"  # the window ends on the storm's big day; forecast for 12 April
PICTURE_K = 7


def window_build(r: River, out_dir: Path) -> dict[str, object]:
    """Seven days of rain and flow, fourteen weights, one forecast: two frames.

    The linear model (window 7, flow as the target) is fitted on WY1990-2008.
    """
    t = int(np.flatnonzero(r.dates == np.datetime64(PICTURE_DAY))[0])
    fit = issued(np.concatenate([r.train, r.val]))
    xf = window(r, fit, PICTURE_K)
    ok = np.isfinite(xf).all(axis=1) & np.isfinite(r.flow[fit + 1])
    m = LinearRegression().fit(xf[ok], r.flow[fit + 1][ok])
    x = window(r, np.array([t]), PICTURE_K)[0]
    contrib = m.coef_ * x  # rain t..t-6, then flow t..t-6
    forecast = float(contrib.sum() + m.intercept_)
    days = r.dates[t - PICTURE_K + 1 : t + 2]
    for frame in (1, 2):
        fig, (a, b) = plt.subplots(
            2, 1, figsize=aml_style.FULL, sharex=True, height_ratios=(1, 1.4)
        )
        a.bar(days[:-1], r.rain[t - PICTURE_K + 1 : t + 1], color=aml_style.PALETTE[0], width=0.7)
        a.set_ylabel("rain\n(mm)", fontsize=11)
        b.plot(days[:-1], r.flow[t - PICTURE_K + 1 : t + 1], color=aml_style.INK, marker="o", lw=2)
        b.set_ylabel("flow\n(mm/day)", fontsize=11)
        b.annotate(
            "?",
            (days[-1], forecast),
            fontsize=26,
            color=aml_style.PALETTE[1],
            ha="center",
            va="center",
            fontweight="bold",
        )
        for ax in (a, b):
            ax.axvspan(
                days[0] - np.timedelta64(12, "h"),
                days[-2] + np.timedelta64(12, "h"),
                color=aml_style.PALETTE[0],
                alpha=0.07,
            )
        if frame == 2:
            for k in range(PICTURE_K):
                d = days[PICTURE_K - 1 - k]
                a.annotate(
                    f"{m.coef_[k]:+.3f}",
                    (d, r.rain[t - k]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=aml_style.PALETTE[0],
                )
                b.annotate(
                    f"{m.coef_[PICTURE_K + k]:+.2f}",
                    (d, r.flow[t - k]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=aml_style.INK,
                )
            b.plot([days[-1]], [r.flow[t + 1]], marker="*", ms=16, color=aml_style.PALETTE[1])
            b.text(
                0.03,
                0.88,
                f"forecast {forecast:.1f} (?)\nobserved {r.flow[t + 1]:.1f} (star)",
                transform=b.transAxes,
                va="top",
                fontsize=11,
            )
        b.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        aml_style.save(fig, out_dir, f"window_{frame}")
    return {
        "rain": x[:PICTURE_K].tolist(),
        "flow": x[PICTURE_K:].tolist(),
        "weights_rain": m.coef_[:PICTURE_K].tolist(),
        "weights_flow": m.coef_[PICTURE_K:].tolist(),
        "intercept": float(m.intercept_),
        "forecast": forecast,
        "observed": float(r.flow[t + 1]),
        "persistence": float(r.flow[t]),
    }


def data_size_figure(
    ds: dict[str, object], chronos_nse: float, chosen: Setting, out_dir: Path
) -> None:
    """Test NSE against years of training record: LSTM from scratch, forest, Chronos-2 (none)."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    yrs = list(YEARS)
    med = [ds["lstm"][n]["test_median"] for n in yrs]
    lo = [ds["lstm"][n]["test_median"] - ds["lstm"][n]["test_min"] for n in yrs]
    hi = [ds["lstm"][n]["test_max"] - ds["lstm"][n]["test_median"] for n in yrs]
    ax.errorbar(
        yrs,
        med,
        yerr=[lo, hi],
        color=aml_style.PALETTE[0],
        marker="o",
        lw=2,
        capsize=4,
        label=f"LSTM from scratch ({len(FEW_SEEDS)} seeds)",
    )
    ax.plot(
        yrs,
        [ds["forest"][n] for n in yrs],
        color=aml_style.PALETTE[2],
        marker="^",
        lw=2,
        ls="--",
        label=f"random forest on {chosen.k}-day windows",
    )
    ax.axhline(
        chronos_nse,
        color=aml_style.PALETTE[1],
        lw=2,
        ls="-.",
        label="Chronos-2, no training at all",
    )
    ax.set_xscale("log")
    ax.set_xticks(yrs, [str(n) for n in yrs])
    ax.set_xlabel("years of record to train on")
    ax.set_ylabel("test NSE, tomorrow's flow")
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    aml_style.save(fig, out_dir, "data_size")


def alerts(r: River, forecasts: dict[str, np.ndarray], area_km2: float) -> dict[str, object]:
    """Day-ahead alerts above the training years' 99th-percentile flow, on the test years.

    ``forecasts`` hold one forecast per test day (the target day), each made the day before.
    """
    thr = float(np.nanquantile(r.flow[r.train], FLOOD_Q))
    obs = r.flow[r.test]
    seen = np.isfinite(obs)
    event = seen & (obs > thr)
    out = {
        "threshold": thr,
        "threshold_m3s": thr * area_km2 / 86.4,  # mm/day over km2 -> m3/s
        "events": int(event.sum()),
        "days": int(seen.sum()),
        "by": {},
    }
    for name, f in forecasts.items():
        assert f.shape == obs.shape, (name, f.shape)
        alert = seen & (f > thr)
        out["by"][name] = {
            "hits": int((alert & event).sum()),
            "misses": int((~alert & event).sum()),
            "false": int((alert & ~event).sum()),
        }
    return out


def decide_figure(al: dict[str, object], out_dir: Path) -> None:
    """Hits and false alarms of each forecaster at the flood threshold."""
    names = list(al["by"])
    hits = [al["by"][n]["hits"] for n in names]
    false = [al["by"][n]["false"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.barh(
        y - 0.2,
        hits,
        height=0.38,
        color=aml_style.PALETTE[2],
        label=f"floods caught (of {al['events']} days)",
    )
    ax.barh(
        y + 0.2, false, height=0.38, color=aml_style.PALETTE[1], hatch="//", label="false alarms"
    )
    for yi, h, f in zip(y, hits, false, strict=True):
        ax.text(h + 0.3, yi - 0.2, str(h), va="center", fontsize=11)
        ax.text(f + 0.3, yi + 0.2, str(f), va="center", fontsize=11)
    ax.set_yticks(y, names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("days in the six test years")
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    aml_style.save(fig, out_dir, "decide")


def title_art_figure(r: River, uh: dict[str, object], modes: np.ndarray, out_dir: Path) -> None:
    """The title art: the April 2013 flood, the filter's weights, the bridge's frequencies.

    The flood is a line of light over the unit hydrograph's stepped weights; a dark sun is ringed
    at the bridge's natural frequencies.
    """
    c = title_art.Canvas(seed=20)
    top = c.top
    c.ground(("#0b0d14", "#101722", "#1d1420", "#08070b"), grain=0.08)
    w = np.asarray(uh["weights"][:12])
    for k, wk in enumerate(w):  # the unit hydrograph as stepped Andean blocks
        h = 0.05 + 0.9 * wk / w.max() * 0.45
        c.glaze(
            c.polygon(
                [
                    (0.06 + 0.075 * k, 0.05),
                    (0.13 + 0.075 * k, 0.05),
                    (0.13 + 0.075 * k, 0.05 + h),
                    (0.06 + 0.075 * k, 0.05 + h),
                ],
                soft=4,
            ),
            ("#5a2216", "#3a1912")[k % 2],
            0.85,
        )
    z = (r.dates >= np.datetime64("2013-03-01")) & (r.dates <= np.datetime64("2013-06-30"))
    q = np.nan_to_num(r.flow[z])
    xs = np.linspace(0.04, 0.96, len(q))
    ys = top * 0.42 + 0.55 * top * q / q.max() * 0.5
    c.glow(c.lines([list(zip(xs, ys, strict=True))], 2.5, soft=1.0), "#8fc7f0", 0.7, spread=10)
    centre = (0.70, top * 0.80)
    for f in sorted(modes):
        rad = 0.025 * f
        ring = c.disc(centre, rad + 0.004, soft=2) - c.disc(centre, rad - 0.004, soft=2)
        c.glow(np.clip(ring, 0, 1), "#eaa548", 0.45, spread=8)
    c.glaze(c.disc(centre, 0.035, soft=2), "#1d0a10", 0.95)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: rust stepped blocks along the bottom, falling from "
        "left to right like a unit hydrograph's weights; across the middle a pale blue line of "
        "light, the real flow of an Illinois river through its April 2013 flood; at the upper "
        "right a small dark sun inside thin golden rings spaced as a railway bridge's natural "
        "frequencies.",
    )


def river_stage(r: River, out_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """The windows, the filter, SARIMA, the random-day split: numbers, and the filter."""
    numbers: dict[str, object] = {"scales": scales_figure(r, out_dir)}
    numbers["picture"] = window_build(r, out_dir)
    numbers["windows"] = sweep = windows_sweep(r)
    windows_figure(sweep, out_dir)
    uh = unit_hydrograph(r)
    unit_hydrograph_figure(r, uh, out_dir)
    numbers["uh"] = {k: v for k, v in uh.items() if k not in ("sim", "weights")}
    numbers["uh"]["weights"] = uh["weights"].tolist()
    s = sarima(r)
    sarima_figure(s, out_dir)
    numbers["sarima"] = {k: s[k] for k in ("params", "rmse_log", "band90_coverage")}
    numbers["split"] = random_split(r, setting_of(sweep["chosen"]))
    return numbers, uh


def bridge_stage(r: River, out_dir: Path) -> dict[str, object]:
    """KW51: the records, Fourier series, spectrum, spectrogram, and the frequency alarm."""
    p = passage()
    times, freqs, temp = tracked()
    numbers = {"series": series_figure(r, p, times, freqs, out_dir)}
    numbers["fourier"] = fourier_figure(p, out_dir)
    numbers["spectrum"] = spectrum_figure(p, times, freqs, out_dir)
    numbers["spectrogram"] = spectrogram_figure(p, out_dir)
    spectrogram3d_figure(p, out_dir, elev=30.0, azim=-110.0)  # chosen from six views
    al = alarm(times, freqs, temp)
    alarm_figure(al, out_dir)
    numbers["alarm"] = {k: v for k, v in al.items() if k not in ("series", "temp", "resid")}
    return numbers


def simulate_stage(
    cam: camels.Camels, r: River, uh: dict[str, object], out_dir: Path
) -> dict[str, object]:
    """The 1-D CNN and the LSTM from weather alone: scores, check, hydrograph, rain response."""
    sim = basins(cam, [camels.HOME], SIMULATE)
    lstm_t, cnn_t = (tune([sim], kind, NET_SEEDS, cam) for kind in ("lstm", "cnn"))
    numbers = {"lstm_sim": summary(lstm_t), "cnn_sim": summary(cnn_t)}
    lstm, cnn = median_run(lstm_t.runs), median_run(cnn_t.runs)
    numbers["verify"] = verify_lstm(lstm.model, sim, int(cam.period("test")[400]))
    numbers["cnn_weights"] = int(sum(q.numel() for q in cnn.model.parameters()))
    convs = [m for m in cnn.model.net if isinstance(m, nn.Conv1d)]
    numbers["cnn_layers"] = {
        "first": sum(q.numel() for q in convs[0].parameters()),
        "dilated": sum(q.numel() for m in convs[1:] for q in m.parameters()),
        "head": sum(q.numel() for q in cnn.model.head.parameters()),
    }
    days = np.arange(SEQ, len(cam.dates))
    sims = {}
    for name, kept in (("1-D CNN", cnn), ("LSTM", lstm)):
        sims[name] = np.full(len(cam.dates), np.nan)
        sims[name][days] = predict(kept.model, sim, days, SEQ)[0]
    nses = {
        "unit hydrograph": uh["test_nse"],
        "1-D CNN": float(cnn.test_nse[0]),
        "LSTM": float(lstm.test_nse[0]),
    }
    hydrograph_figure(r, uh, sims, nses, out_dir)
    resp = {day: rain_response(lstm, sim, cam, day) for day in RESPONSE_DAYS.values()}
    response_figure(uh, resp, out_dir)
    numbers["response"] = {
        d: {"sum30": float(g[:30].sum()), "lag0": float(g[0]), "lags1_29": float(g[1:30].sum())}
        for d, g in resp.items()
    }
    numbers["storms"] = storms(r, uh) | {
        "lstm_peaks": {d: float(np.nanmax(sims["LSTM"][storm_days(r, d)])) for d in STORMS},
        "lowest_training_flow": float(np.nanmin(r.flow[r.train])),
    }
    return numbers


def forecast_stage(cam: camels.Camels) -> tuple[dict[str, object], dict[Task, Tuned]]:
    """The LSTM given past flow (and tomorrow's rain): rates bracketed, three seeds each."""
    numbers, fc = {}, {}
    for task, name in ((FORECAST, "lstm_forecast"), (FORECAST_RAIN, "lstm_rain")):
        fc[task] = tune([basins(cam, [camels.HOME], task)], "lstm", FEW_SEEDS, cam)
        numbers[name] = summary(fc[task])
    return numbers, fc


def chronos_stage(
    cam: camels.Camels,
    r: River,
    ds: dict[str, object],
    chosen: Setting,
    out_dir: Path,
) -> tuple[dict[str, object], BaseChronosPipeline, int]:
    """Chronos-2 zero-shot: scores, the April 2013 flood, data size, the embedding map."""
    pipe = chronos()
    cs = chronos_scores(pipe, r)
    cs["weights"] = int(sum(q.numel() for q in pipe.model.parameters()))
    numbers: dict[str, object] = {"chronos": cs}
    z = np.flatnonzero(
        (r.dates >= np.datetime64("2013-03-25")) & (r.dates <= np.datetime64("2013-05-10"))
    )
    forest_z = chosen_forecast(r, chosen, issued(r.train), issued(z))  # 15 years, as the LSTM
    numbers["chronos_event"] = chronos_event_figure(pipe, r, cs["context"], forest_z, out_dir)
    data_size_figure(ds, cs["test"]["rain"], chosen, out_dir)
    numbers["map"] = basin_map(embeddings(pipe, cam), cam, out_dir)
    return numbers, pipe, cs["context"]


def decide_stage(
    cam: camels.Camels,
    r: River,
    fc: dict[Task, Tuned],
    chosen: Setting,
    *,
    pipe: BaseChronosPipeline,
    context: int,
    out_dir: Path,
) -> dict[str, object]:
    """Day-ahead flood alerts on the test years, each forecaster at the same threshold.

    The forest and the LSTMs trained on the same 15 years (WY1990-2004); each LSTM is its
    median seed's run.
    """
    te, at = r.test, issued(r.test)
    fit = issued(r.train)

    def lstm(task: Task) -> np.ndarray:
        kept = median_run(fc[task].runs)
        return predict(kept.model, basins(cam, [camels.HOME], task), te, SEQ)[0]

    forecasts = {
        "persistence": r.flow[at],
        "forest": chosen_forecast(r, chosen, fit, at),
        "LSTM": lstm(FORECAST),
        "Chronos-2": chronos_forecast(pipe, r, at, context, rain=True)[:, 0],
        "forest + tomorrow's rain": chosen_forecast(r, chosen, fit, at, tomorrow_rain=True),
        "LSTM + tomorrow's rain": lstm(FORECAST_RAIN),
        "Chronos-2 + tomorrow's rain": chronos_forecast(
            pipe, r, at, context, rain=True, tomorrow_rain=True
        )[:, 0],
    }
    al = alerts(r, forecasts, float(cam.attrs["area_gages2"][cam.basin(camels.HOME)]))
    decide_figure(al, out_dir)
    i = storm_days(r, STORMS[1]) - te[0]  # forecasts[i] are for the storm's days
    al["storm_b_peaks"] = {k: float(np.nanmax(forecasts[k][i])) for k in ("forest", "LSTM")}
    al["forest_years"] = int(round(len(fit) / 365.25))
    return al


def main() -> None:
    """Generate every figure and number for CEE Lecture 20."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before cuBLAS starts
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    t0 = time.perf_counter()
    cam = camels.load()
    r = river(cam)
    cols = ["gauge_name", "area_gages2", "frac_snow", "aridity"]
    numbers: dict[str, object] = {"basin": cam.attrs.loc[cam.basin(camels.HOME), cols].to_dict()}
    river_numbers, uh = river_stage(r, out_dir)
    numbers |= river_numbers
    chosen = setting_of(numbers["windows"]["chosen"])
    assert chosen.model == "forest", chosen  # the slides name the forest
    numbers |= bridge_stage(r, out_dir)
    log(f"river and bridge in {time.perf_counter() - t0:.0f} s")
    numbers["ladder_end"] = ladder_end(cam)
    numbers |= simulate_stage(cam, r, uh, out_dir)
    log(f"simulators in {time.perf_counter() - t0:.0f} s")
    fc_numbers, fc = forecast_stage(cam)
    numbers |= fc_numbers
    numbers["data_size"] = data_size(cam, r, chosen, fc[FORECAST])
    log(f"forecasters and data size in {time.perf_counter() - t0:.0f} s")
    numbers["regional"] = regional(cam)
    numbers["runs"] = len(_RUNS)
    log(f"networks ({len(_RUNS)} runs) in {time.perf_counter() - t0:.0f} s")
    chronos_numbers, pipe, context = chronos_stage(cam, r, numbers["data_size"], chosen, out_dir)
    numbers |= chronos_numbers
    numbers["alerts"] = decide_stage(
        cam, r, fc, chosen, pipe=pipe, context=context, out_dir=out_dir
    )
    title_art_figure(r, uh, np.asarray(numbers["spectrum"]["modes_that_hour"]), out_dir)
    aml_style.thumbnails(out_dir, ("windows", "response", "spectrogram", "data_size"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    log(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
