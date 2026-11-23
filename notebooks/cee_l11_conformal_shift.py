"""Applied Machine Learning for CEE — Lecture 11: Conformal Prediction and Shift.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l11_conformal_shift.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_usgs.sh`` once (USGS daily discharge, public domain).
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Applied Machine Learning for CEE — Lecture 11: Conformal Prediction and Shift

    **Slides:** *Applied Machine Learning for CEE*, Lecture 11 (`lectures/cee_l11_conformal_shift`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Angelopoulos & Bates (2023), *A Gentle Introduction to Conformal Prediction*, §1–§3 and §7 (shift); Gibbs & Candès (2021) for adaptive conformal inference; Milly et al. (2008) on nonstationarity.
    **Data:** USGS daily mean discharge, Sangamon River at Monticello, Illinois (05572000), 1908–2025; public domain. Fetch with `bash scripts/fetch_usgs.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. It runs in
    about six seconds on a CPU.

    **Your tasks** (from the slides): build the interval by hand, write split conformal and
    check its guarantee on exchangeable halves, then measure coverage by decade, year, and
    river state, and fix what can be fixed.
    """)
    return


@app.cell
def _():
    import os
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    H = 7  # forecast horizon, days
    ALPHA = 0.10  # 90% intervals
    FIT, CAL, TEST = (1950, 1969), (1970, 1979), (2000, 2025)
    GAMMA = 0.005  # adaptive conformal step size
    RISE = 1.0  # rising week: log flow up by more than 1 (x2.7)
    STEADY = 0.5  # steady week: log flow within 0.5 (x/÷ 1.6)
    FLOOR_M3S = 0.01  # zero-flow days are raised to this before the log
    MEAN_DAYS = 30  # window of the running-mean input
    LOW_QUANTILE = 0.10  # "low flow now": below the fit years' 10th percentile
    N_HALVINGS = 200  # random halvings of the 1970s
    LEVELS = (0.10, 0.05, 0.01)  # the 90%, 95% and 99% intervals
    # Decades outside the fit years that the record covers from first to last year.
    OUT_OF_FIT = (1910, 1920, 1930, 1940, 1970, 1980, 1990, 2000, 2010)
    MIN_YEAR_ROWS = 300  # a year enters the one-year reference with this many forecasts
    ROLL_YEARS = 3  # window of the rolling coverage
    FEATURES = ("lag0", "lag1", "lag3", "lag7", "mean30", "sin", "cos")
    PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    return (
        ALPHA,
        CAL,
        FEATURES,
        FIT,
        FLOOR_M3S,
        GAMMA,
        H,
        LEVELS,
        LOW_QUANTILE,
        MEAN_DAYS,
        MIN_YEAR_ROWS,
        N_HALVINGS,
        OUT_OF_FIT,
        PALETTE,
        Path,
        RISE,
        ROLL_YEARS,
        SEED,
        STEADY,
        TEST,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 7–10

    On your own, no notes, two minutes:

    1. The 5% strength line was calibrated over all mixes, yet 13.5% of strong mixes fell below
       it. What is the general lesson?
    2. RANSAC kept the calm line until the burst made up 65% of the window. What did it assume
       about the readings?
    3. Glisan's breakdown threshold misfired at Stark/Washington. What differed between the data
       it learned from and the data it was used on?

    **Today:** how sure is next week's river forecast, decades after it was built?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. Calibration on average can hide miscalibration by subgroup (Lecture 10).
                2. That the calm readings were the largest tight group (Lecture 9).
                3. The station's traffic and its occupancy–congestion relation (Lecture 7): the
                   use was not like the training. Today: shift.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    The Sangamon River feeds Lake Decatur, Decatur's water supply. **The decision:** plan
    withdrawals and releases for a 90% range of next week's flow. **The setting:** the forecast
    was built in 1980, fit on 1950–1969, calibrated on 1970–1979, and used ever since. The
    forecast: log flow seven days ahead from the log flows of today, 1, 3, and 7 days ago,
    their 30-day mean, and the season, by least squares (Lecture 5).
    """)
    return


@app.cell
def _(CAL, FEATURES, FIT, FLOOR_M3S, H, MEAN_DAYS, Path, np, pd):
    def find_csv() -> Path:
        """Locate ``data/derived/usgs_05572000_daily.csv`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_usgs.sh``).
        """
        rel = Path("data") / "derived" / "usgs_05572000_daily.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_usgs.sh`")

    _raw = pd.read_csv(find_csv(), parse_dates=["date"]).set_index("date")["discharge_m3s"]
    q = _raw.reindex(pd.date_range(_raw.index.min(), _raw.index.max(), freq="D"))

    def make_rows(flow: pd.Series, floor_m3s: float) -> pd.DataFrame:
        """Inputs known today and the target, log flow ``H`` days ahead; gaps dropped.

        Args:
            flow: Daily discharge (m3/s).
            floor_m3s: Flows below this (the zero-flow days) are raised to it before the log.

        Returns:
            One row per issue day: the ``FEATURES`` and the target ``y``.
        """
        _lq = np.log(flow.clip(lower=floor_m3s))
        _doy = flow.index.dayofyear
        _rows = pd.DataFrame(
            {
                "lag0": _lq,
                "lag1": _lq.shift(1),
                "lag3": _lq.shift(3),
                "lag7": _lq.shift(7),
                "mean30": _lq.rolling(MEAN_DAYS).mean(),
                "sin": np.sin(2 * np.pi * _doy / 365.25),
                "cos": np.cos(2 * np.pi * _doy / 365.25),
            },
            index=flow.index,
        )
        _rows["y"] = _lq.shift(-H)
        return _rows.dropna()

    d = make_rows(q, FLOOR_M3S)

    def design(frame: pd.DataFrame) -> np.ndarray:
        """``[1, inputs]``."""
        _A = np.column_stack([np.ones(len(frame)), frame[list(FEATURES)].to_numpy()])
        assert _A.shape == (len(frame), len(FEATURES) + 1)
        return _A

    def period(frame: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
        """Rows whose forecast is issued in the given years (by issue day, not target day)."""
        return frame[(frame.index.year >= years[0]) & (frame.index.year <= years[1])]

    # Question slide: 42,499 days since 1908-02-04; 567 missing, all in 1912-1914; a peak of
    # 529.5 m3/s; 59 zero-flow days and 135 below the 0.01 m3/s floor.
    assert (int(q.notna().sum()), int(q.isna().sum()), str(q.index[0].date())) == (
        42499,
        567,
        "1908-02-04",
    )
    _gap_years = q.index[q.isna()].year
    assert (_gap_years.min(), _gap_years.max()) == (1912, 1914)
    assert round(float(q.max()), 1) == 529.5
    assert (int((q <= 0).sum()), int((q < FLOOR_M3S).sum())) == (59, 135)
    # Rows belong to the period of their issue day. The last fit target (Jan 7, 1970) comes
    # before the first calibration target (Jan 8, 1970): no target is shared.
    _lag = pd.Timedelta(days=H)
    _last_fit, _first_cal = period(d, FIT).index[-1] + _lag, period(d, CAL).index[0] + _lag
    assert _first_cal > _last_fit
    assert (str(_last_fit.date()), str(_first_cal.date())) == ("1970-01-07", "1970-01-08")
    return d, design, make_rows, period, q


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: build the interval

    Nine forecasts from a calibration year, each with its error factor (how many times too high
    or too low the flow came): 1.2, 1.5, 1.1, 2.0, 1.3, 3.1, 1.4, 1.8, 2.4. **A tenth forecast is
    made. Which factor gives a 90% range? And 80%?**
    """)
    return


@app.cell
def _(ALPHA, np):
    def conformal_quantile(
        scores: np.ndarray, alpha: float = ALPHA, *, presorted: bool = False
    ) -> float:
        """Split conformal: the ceil((n + 1)(1 - alpha))-th smallest calibration score.

        Args:
            scores: Calibration scores.
            alpha: Miscoverage level.
            presorted: Whether ``scores`` is already sorted ascending (skips the sort).

        Returns:
            The cutoff; infinity when there are too few scores for this level (or alpha <= 0),
            minus infinity, an empty interval, for alpha >= 1.
        """
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))  # rank of the cutoff
        if k > n:
            return float("inf")  # too few scores
        if k < 1:
            return float("-inf")  # alpha >= 1: cover nothing
        return float((scores if presorted else np.sort(scores))[k - 1])

    _f = np.array([1.2, 1.5, 1.1, 2.0, 1.3, 3.1, 1.4, 1.8, 2.4])
    assert conformal_quantile(np.log(_f), 0.10) == np.log(3.1)
    assert conformal_quantile(np.log(_f), 0.20) == np.log(2.4)
    assert conformal_quantile(np.log(_f[:8]), 0.10) == float("inf")
    print("90%: ×/÷ 3.1 (the 9th of 9); 80%: ×/÷ 2.4 (the 8th); with 8 scores, no 90% interval")
    return (conformal_quantile,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with **conformal calibration** as this lecture's step: past flows →
    encoder (lags, season) → $\mathbf{z}$ → decoder (a line, fit on 1950–1969) → the forecast
    $\ln\hat y$ → conformal: $\ln\hat y \pm \hat q$, with $\hat q$ from the 1970s scores → a 90%
    interval. §4 recomputes the slide's trace.

    ## 3. Derive

    **Assumption:** the calibration cases and the new case are **exchangeable**: every order of
    them is equally likely (i.i.d. is a special case).

    With scores $s_i = |\ln y_i - \ln\hat y_i|$ on $n$ calibration cases and
    $\hat q$ = the $\lceil (n+1)(1-\alpha)\rceil$-th smallest,

    $$1-\alpha \le P(s_{n+1} \le \hat q) \le 1-\alpha+\frac{1}{n+1}.$$

    **Proof.** By exchangeability, $s_{n+1}$ is equally likely to hold each of the $n+1$ ranks
    (ties aside). It is at most $\hat q$ exactly when its rank is at most
    $k = \lceil (n+1)(1-\alpha)\rceil$, with probability $k/(n+1) \in [1-\alpha,\ 1-\alpha+1/(n+1))$.
    The model must be fixed before calibration (split conformal), and the guarantee is
    **marginal**: averaged over calibration sets and new cases.

    **Shift.** Under covariate shift, weight the calibration scores by
    $w(x) = p_{\text{new}}(x)/p_{\text{cal}}(x)$ (Tibshirani et al., 2019). Under drift, adapt
    the level online (Gibbs & Candès, 2021): $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \mathrm{err}_t)$,
    which guarantees $\big|\frac1T\sum_t \mathrm{err}_t - \alpha\big| \le
    \frac{\max(\alpha_1, 1-\alpha_1)+\gamma}{\gamma T}$ whatever the data do.
    """)
    return


@app.cell
def _(CAL, FIT, conformal_quantile, d, design, np, period):
    _fit, _cal = period(d, FIT), period(d, CAL)
    w = np.linalg.lstsq(design(_fit), _fit["y"].to_numpy(), rcond=None)[0]
    s_cal = np.abs(_cal["y"].to_numpy() - design(_cal) @ w)
    qhat = conformal_quantile(s_cal)
    _k = int(np.ceil((len(s_cal) + 1) * 0.9))
    # Slides: fit 7,305 days, calibrate 3,652; qhat = the 3,288th score = 1.28, a factor of 3.6.
    assert (len(_fit), len(_cal), _k) == (7305, 3652, 3288)
    assert round(qhat, 2) == 1.28 and round(float(np.exp(qhat)), 1) == 3.6
    # Guarantee: 90.0% to 1 - alpha + 1/(n + 1) = 90.03%.
    assert round(100 * (1 - 0.10 + 1 / (len(s_cal) + 1)), 2) == 90.03
    # Scores figure: 59.3% below 0.5; the plot stops at the 99.5th percentile, 3.44.
    assert round(100 * float(np.mean(s_cal < 0.5)), 1) == 59.3
    assert round(float(np.quantile(s_cal, 0.995)), 2) == 3.44
    _row = d.loc["2024-06-03"]
    _pred = float((design(d.loc[["2024-06-03"]]) @ w)[0])
    _today, _fc, _lo, _hi, _obs = (
        float(np.exp(_v)) for _v in (_row["lag0"], _pred, _pred - qhat, _pred + qhat, _row["y"])
    )
    # Diagram: 9.31 m3/s x/÷ 3.586 (qhat 1.277) -> 2.60 to 33.4; today 10.48, came 10.31.
    _shown = (round(_today, 2), round(_fc, 2), round(_lo, 2), round(_hi, 1), round(_obs, 2))
    assert _shown == (10.48, 9.31, 2.6, 33.4, 10.31), _shown
    assert round(qhat, 3) == 1.277 and round(float(np.exp(qhat)), 3) == 3.586
    # The terms the diagram shows reproduce its interval.
    assert round(9.31 / 3.586, 2) == 2.6 and round(9.31 * 3.586, 1) == 33.4
    print(
        f"qhat = {qhat:.3f} (x/÷ {np.exp(qhat):.3f}); June 3, 2024: forecast {_fc:.2f}, "
        f"interval {_lo:.2f} to {_hi:.1f}, came {_obs:.2f} m3/s"
    )
    return qhat, s_cal, w


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Predict:** split the 1970s into two halves by week, 200 times; calibrate on one half and
    test on the other. Mean coverage? Spread? Weeks are held out together because the days of
    a week share a storm.
    """)
    return


@app.cell
def _(CAL, N_HALVINGS, SEED, conformal_quantile, d, design, np, period, w):
    _cal = period(d, CAL)
    _r = np.abs(_cal["y"].to_numpy() - design(_cal) @ w)

    def halvings(groups: np.ndarray) -> np.ndarray:
        """Calibrate on a random half of the groups, cover the rest, ``N_HALVINGS`` times.

        Args:
            groups: The group of each 1970s score (its week, or the day itself).

        Returns:
            The coverage of each halving.
        """
        _rng = np.random.default_rng(SEED)
        _ids = np.unique(groups)
        _covs = []
        for _ in range(N_HALVINGS):
            _m = np.isin(groups, _rng.permutation(_ids)[: len(_ids) // 2])
            _covs.append(float(np.mean(_r[~_m] <= conformal_quantile(_r[_m]))))
        return np.asarray(_covs)

    _by_week = halvings(np.asarray((_cal.index - _cal.index[0]).days // 7))
    _by_day = halvings(np.arange(len(_r)))  # ignores the days' order: exchangeable days
    _summ = {
        _name: (
            round(100 * float(np.mean(_c)), 1),
            round(100 * float(np.quantile(_c, 0.05)), 1),
            round(100 * float(np.quantile(_c, 0.95)), 1),
        )
        for _name, _c in (("by week", _by_week), ("by day", _by_day))
    }
    # Slide "Exchangeable halves: 90.1%": 5th to 95th percentile, 86.2% to 93.2% by week;
    # by day, 88.5% to 91.6%: correlated days widen the spread.
    assert _summ == {"by week": (90.1, 86.2, 93.2), "by day": (90.1, 88.5, 91.6)}, _summ
    print({_k: f"mean {_v[0]}%, 5th-95th {_v[1]}% to {_v[2]}%" for _k, _v in _summ.items()})
    return


@app.cell
def _(CAL, FIT, LEVELS, conformal_quantile, d, np, period, qhat, s_cal):
    _cal = period(d, CAL)
    _pers = conformal_quantile(np.abs(_cal["y"].to_numpy() - _cal["lag0"].to_numpy()))
    _clim = conformal_quantile(np.abs(_cal["y"].to_numpy() - np.median(period(d, FIT)["y"])))
    # Bracket: no model x/÷ 13.8; persistence 4.4; the model 3.6.
    assert (
        round(float(np.exp(_clim)), 1),
        round(float(np.exp(_pers)), 1),
        round(float(np.exp(qhat)), 1),
    ) == (13.8, 4.4, 3.6)
    print(
        f"90% interval widths: climatology x/÷ {np.exp(_clim):.1f}, persistence "
        f"{np.exp(_pers):.1f}, model {np.exp(qhat):.1f}"
    )
    # Exit and vote: the model's 90%, 95% and 99% intervals are x/÷ 3.6, 6.5 and 15.5; the
    # 99% one is wider than no model at all (13.8).
    _by_level = [round(float(np.exp(conformal_quantile(s_cal, _a))), 1) for _a in LEVELS]
    assert _by_level == [3.6, 6.5, 15.5] and _by_level[-1] > round(float(np.exp(_clim)), 1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **Later decades.** The 1970s interval, used from 1980 to 2025. Coverage by decade?
    2. **One year at a time.** The worst year since 2000?
    3. **Rising weeks.** Weeks when the flow rises 2.7-fold or more: coverage?
    4. **By month.** Calibrate a separate interval for each month (Mondrian). Fixed?
    """)
    return


@app.cell
def _(
    ALPHA,
    GAMMA,
    H,
    MIN_YEAR_ROWS,
    OUT_OF_FIT,
    PALETTE,
    ROLL_YEARS,
    TEST,
    conformal_quantile,
    d,
    design,
    np,
    pd,
    plt,
    qhat,
    s_cal,
    w,
):
    r_all = np.abs(d["y"].to_numpy() - design(d) @ w)
    cov_fixed = pd.Series(r_all <= qhat, index=d.index)
    # Adaptive conformal from 1980, the slide's update: a forecast is judged H days after it is
    # issued, and alpha_t moves by GAMMA (ALPHA - err). The cutoff is the conformal order
    # statistic at level alpha_t of the fixed 1970s scores (infinite for alpha_t <= 0; empty for
    # alpha_t >= 1), not np.quantile's interpolation. Row i - H is H days back: no gaps after 1980.
    _start = int(np.flatnonzero(d.index.year >= 1980)[0])
    _gaps = (d.index[_start + 1 :] - d.index[_start:-1]).days
    assert np.all(_gaps == 1), "the H-day lag needs consecutive days"
    _s_sorted = np.sort(s_cal)
    _alpha_t, _covered = ALPHA, []
    for _i in range(_start, len(r_all)):
        if _i - H >= _start:  # the forecast made H days ago is judged today
            _err = not _covered[_i - H - _start]
            _alpha_t += GAMMA * (ALPHA - float(_err))
        _covered.append(bool(r_all[_i] <= conformal_quantile(_s_sorted, _alpha_t, presorted=True)))
    aci = pd.Series(_covered, index=d.index[_start:])
    _fixed_dec = (100 * cov_fixed.groupby((cov_fixed.index.year // 10) * 10).mean()).round(1)
    _aci_dec = (100 * aci.groupby((aci.index.year // 10) * 10).mean()).round(1)
    _test = cov_fixed[(cov_fixed.index.year >= TEST[0]) & (cov_fixed.index.year <= TEST[1])]
    _aci_test = aci[(aci.index.year >= TEST[0]) & (aci.index.year <= TEST[1])]
    _yearly = (100 * _test.groupby(_test.index.year).mean()).round(0)
    # Slides "Later decades fall short" and "One year: 75% inside".
    assert (_fixed_dec[1980], _fixed_dec[2010], round(100 * float(_test.mean()), 1)) == (
        87.5,
        87.0,
        87.9,
    )
    assert [_fixed_dec[_k] for _k in (1910, 1920, 1930, 1940)] == [88.9, 88.9, 88.8, 89.0]
    assert round(100 * float(_aci_test.mean()), 1) == 89.8 and (_aci_dec.min(), _aci_dec.max()) == (
        89.2,
        90.6,
    )
    assert _aci_dec.to_dict() == {1980: 89.9, 1990: 90.6, 2000: 89.2, 2010: 90.2, 2020: 90.1}
    assert (_yearly.min(), int(_yearly.idxmin()), _yearly.max()) == (75.0, 2014, 93.0)
    assert _yearly.drop(2014).min() == 83.0  # the other years: 83% to 93%
    # Partial decades (open markers on the slide): 1908-1909, 668 forecasts, 91.8%; 2020-2025,
    # 2,185 forecasts, 89.2% fixed and 90.1% adaptive. Full decades since 1980: 87.0-89.3%.
    _rows = pd.Series((d.index.year // 10) * 10).value_counts()
    assert (_rows[1900], _rows[2020], _fixed_dec[1900], _fixed_dec[2020]) == (668, 2185, 91.8, 89.2)
    assert [_fixed_dec[_k] for _k in (1980, 1990, 2000, 2010)] == [87.5, 89.3, 88.2, 87.0]

    # The one-year yardstick: within each full decade outside the fit years, a year with at
    # least MIN_YEAR_ROWS forecasts, covered by the interval calibrated on the other nine.
    _yr = d.index.year.to_numpy()
    _loyo = {}
    for _dec in OUT_OF_FIT:
        for _y in range(_dec, _dec + 10):
            _out = _yr == _y
            if _out.sum() >= MIN_YEAR_ROWS:
                _in = (_yr >= _dec) & (_yr < _dec + 10) & ~_out
                _cov = 100 * float(np.mean(r_all[_out] <= conformal_quantile(r_all[_in])))
                _loyo.setdefault(_dec, []).append(_cov)
    _pooled = [_c for _v in _loyo.values() for _c in _v]
    # Slide "One year: 75% inside": 84.4-96.2% in the 1970s, 79.7-98.4% over 87 years in nine
    # decades; 2014 (74.8%) is below every one of them.
    assert (round(min(_loyo[1970]), 1), round(max(_loyo[1970]), 1)) == (84.4, 96.2)
    assert (round(min(_pooled), 1), round(max(_pooled), 1), len(_pooled)) == (79.7, 98.4, 87)
    assert 100 * float(_test[_test.index.year == 2014].mean()) < min(_pooled)

    # Slide "Adapt the level, add the rain": three-year rolling coverage from Dec 30, 1982.
    _win = 365 * ROLL_YEARS
    _roll = pd.DataFrame(
        {
            "fixed": 100 * cov_fixed[aci.index].astype(float).rolling(_win).mean(),
            "aci": 100 * aci.astype(float).rolling(_win).mean(),
        }
    ).dropna()
    assert str(_roll.index[0].date()) == "1982-12-30"
    assert (round(_roll["fixed"].min(), 1), round(_roll["fixed"].max(), 1)) == (81.4, 93.2)
    assert (round(_roll["aci"].min(), 1), round(_roll["aci"].max(), 1)) == (87.9, 92.1)
    _early = _roll[_roll.index.year < 2000]["fixed"]
    _late = _roll[_roll.index.year >= 2000]["fixed"]
    assert (_early.idxmin().year, _late.idxmin().year) == (1991, 2015)
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.plot(_fixed_dec.index, _fixed_dec.to_numpy(), "o-", color=PALETTE[0], label="fixed, 1970s")
    _ax.plot(
        _aci_dec.index, _aci_dec.to_numpy(), "s--", color=PALETTE[1], label="adaptive, from 1980"
    )
    _ax.axhline(90, color="grey", ls=":")
    _ax.set_xlabel("decade")
    _ax.set_ylabel("inside the 90% interval (%)")
    _ax.legend()
    _fig
    return aci, r_all


@app.cell
def _(CAL, conformal_quantile, d, np, pd, r_all):
    # Why the other decades fall short: each decade's own 90% score, and recalibration on each
    # complete decade outside the fit years in turn, covering the other eight.
    _dec = (d.index.year.to_numpy() // 10) * 10
    _q90 = {int(_k): conformal_quantile(r_all[_dec == _k]) for _k in np.unique(_dec)}
    _full = [1910, 1920, 1930, 1940, 1970, 1980, 1990, 2000, 2010]
    _recal = {
        _c: 100 * float(np.mean(r_all[np.isin(_dec, _full) & (_dec != _c)] <= _q90[_c]))
        for _c in _full
    }
    _fitcal = [_q90[_k] for _k in (1950, 1960, 1970)]
    _other = [_v for _k, _v in _q90.items() if 1910 <= _k <= 1940 or _k >= 1980]
    _rest = [_v for _c, _v in _recal.items() if _c != CAL[0]]
    # Notes of "Later decades fall short": 90% scores 1.26-1.28 in 1950-1979, 1.33-1.47 in
    # 1910-1949 and since 1980; calibrated on another full decade 89.3-91.2%, on the 1970s 88.4%.
    assert (round(min(_fitcal), 2), round(max(_fitcal), 2)) == (1.26, 1.28)
    assert (round(min(_other), 2), round(max(_other), 2)) == (1.33, 1.47)
    assert (round(min(_rest), 1), round(max(_rest), 1)) == (89.3, 91.2)
    assert round(_recal[CAL[0]], 1) == 88.4 and _recal[CAL[0]] < min(_rest)
    pd.DataFrame(
        {
            "90% score": pd.Series(_q90).round(3),
            "covers the other full decades (%)": pd.Series(_recal).round(1),
        }
    )
    return


@app.cell
def _(LOW_QUANTILE, RISE, STEADY, np, pd):
    def river_states(rows: pd.DataFrame, fit_rows: pd.DataFrame) -> dict[str, np.ndarray]:
        """Each forecast's river state: four change groups, and low flow across them.

        The four change groups (falling, steady, rising x1.6-2.7, rising x2.7+) split every
        forecast; "low flow now" cuts across them.

        Args:
            rows: The forecasts to group (inputs and target).
            fit_rows: The fit years' rows, which set the low-flow threshold.

        Returns:
            A mask per state.
        """
        _change = (rows["y"] - rows["lag0"]).to_numpy()
        _states = {
            "rising ×2.7+": _change > RISE,
            "rising ×1.6–2.7": (_change > STEADY) & (_change <= RISE),
            "falling": _change < -STEADY,
            "steady": np.abs(_change) <= STEADY,
            "low flow now": rows["lag0"].to_numpy() < np.quantile(fit_rows["lag0"], LOW_QUANTILE),
        }
        _split = sum(int(_m.sum()) for _k, _m in _states.items() if _k != "low flow now")
        assert _split == len(rows)
        return _states

    return (river_states,)


@app.cell
def _(
    CAL,
    FIT,
    LEVELS,
    TEST,
    aci,
    conformal_quantile,
    d,
    design,
    np,
    pd,
    period,
    qhat,
    river_states,
    s_cal,
    w,
):
    _t = period(d, TEST)
    _r = np.abs(_t["y"].to_numpy() - design(_t) @ w)
    _groups = river_states(_t, period(d, FIT))
    _tab = pd.DataFrame(
        {
            _k: {
                "share of daily forecasts (%)": round(100 * float(_m.mean()), 1),
                "inside (%)": round(100 * float(np.mean(_r[_m] <= qhat)), 1),
            }
            for _k, _m in _groups.items()
        }
    ).T
    # Slide "90% on average, 28% on rises".
    assert _tab["inside (%)"].tolist() == [28.1, 93.6, 96.1, 99.7, 81.1]
    assert _tab["share of daily forecasts (%)"].tolist() == [14.3, 8.3, 28.9, 48.6, 7.6]
    _months = (100 * pd.Series(_r <= qhat, index=_t.index).groupby(_t.index.month).mean()).round(0)
    assert (_months.min(), int(_months.idxmin())) == (78.0, 10)
    # Mondrian conformal by month: each month's own 1970s scores.
    _cal = period(d, CAL)
    _s = np.abs(_cal["y"].to_numpy() - design(_cal) @ w)
    _qm = {_m: conformal_quantile(_s[_cal.index.month == _m]) for _m in range(1, 13)}
    _cov_m = pd.Series(_r <= np.array([_qm[_m] for _m in _t.index.month]), index=_t.index)
    _by = (100 * _cov_m.groupby(_t.index.month).mean()).round(0)
    # Slide "Calibrating by month does not help": 87.6% overall, months 74-95%, rises 29.0%.
    assert round(100 * float(_cov_m.mean()), 1) == 87.6 and (_by.min(), _by.max()) == (74.0, 95.0)
    assert round(100 * float(_cov_m.to_numpy()[_groups["rising ×2.7+"]].mean()), 1) == 29.0
    assert int(_by.idxmin()) == 10  # October is lowest for both
    # Exit and vote: wider intervals. 95% covers 62.8% of rises (94.5% of all weeks); 99%
    # covers 89.3% of rises (98.4%); steady weeks 99.7% at 90%.
    _rise = _groups["rising ×2.7+"]
    _wide = {}
    for _a in LEVELS:
        _cut = conformal_quantile(s_cal, _a)
        _wide[round(100 * (1 - _a))] = (
            round(100 * float(np.mean(_r[_rise] <= _cut)), 1),
            round(100 * float(np.mean(_r <= _cut)), 1),
        )
    assert _wide == {90: (28.1, 87.9), 95: (62.8, 94.5), 99: (89.3, 98.4)}, _wide
    # Vote B: adaptive conformal covers 39.0% of rises since 2000.
    assert round(100 * float(aci[_t.index].to_numpy()[_rise].mean()), 1) == 39.0
    _tab
    return


@app.cell
def _(CAL, FIT, TEST, conformal_quantile, design, make_rows, np, period, q, river_states):
    # "Low flow now" rests on the 0.01 m3/s floor for zero flows: refit and recalibrate with the
    # floor at 0.03 and 0.1 m3/s. Low flow moves 81.1 -> 84.0 -> 89.7%; all weeks and rises
    # barely move.
    _by_floor = {}
    for _floor in (0.01, 0.03, 0.1):
        _rows = make_rows(q, _floor)
        _fit, _cal, _t = period(_rows, FIT), period(_rows, CAL), period(_rows, TEST)
        _w = np.linalg.lstsq(design(_fit), _fit["y"].to_numpy(), rcond=None)[0]
        _q = conformal_quantile(np.abs(_cal["y"].to_numpy() - design(_cal) @ _w))
        _hit = np.abs(_t["y"].to_numpy() - design(_t) @ _w) <= _q
        _st = river_states(_t, _fit)
        _by_floor[_floor] = tuple(
            round(100 * float(_hit[_m].mean()), 1)
            for _m in (np.ones(len(_t), bool), _st["low flow now"], _st["rising ×2.7+"])
        )
    assert _by_floor == {
        0.01: (87.9, 81.1, 28.1),
        0.03: (88.1, 84.0, 28.1),
        0.1: (88.5, 89.7, 28.0),
    }, _by_floor
    print("floor: (all weeks, low flow now, rises) %:", _by_floor)
    return


@app.cell
def _(H, PALETTE, RISE, d, design, np, plt, qhat, w):
    _t = d[(d.index >= "2013-03-01") & (d.index <= "2013-06-30")]
    _pred = design(_t) @ w
    _target = _t.index + np.timedelta64(H, "D")
    _lo, _hi, _obs = np.exp(_pred - qhat), np.exp(_pred + qhat), np.exp(_t["y"].to_numpy())
    _miss = (_obs < _lo) | (_obs > _hi)
    _rises = (_t["y"] - _t["lag0"]).to_numpy() > RISE
    # Slide "The interval chases the flood": 19 of 122 missed, 18 on rises; peak 203 on Apr 19.
    assert (len(_t), int(_miss.sum()), int((_miss & _rises).sum())) == (122, 19, 18)
    assert (
        round(float(_obs.max())) == 203
        and str(_target[int(np.argmax(_obs))].date()) == "2013-04-19"
    )
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.fill_between(
        _target, _lo, _hi, color=PALETTE[0], alpha=0.2, label="90% interval, 7 days before"
    )
    _ax.plot(_target, _obs, color="black", label="flow that came")
    _ax.plot(_target[_miss], _obs[_miss], "x", color=PALETTE[1], label="outside")
    _ax.set_yscale("log")
    _ax.set_ylabel("discharge (m³/s)")
    _ax.legend(fontsize=9)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the breaks show.** The fit decades (1950s, 1960s) and the calibration decade (1970s)
    sit near 90% by construction; every other decade falls 1–3 points short. The per-decade
    scores show why: the 1970s were calm (90% score 1.28, against 1.33–1.47 in 1910–1949
    and since 1980), and calibrating on any other full decade covers 89.3–91.2% of
    the rest, the 1970s only 88.4%. The shortfall says as much about the calm calibration decade
    as about a changing river. Coverage over a single year swings from 75% to 93%; within a
    decade, a year covered by an interval calibrated on the other nine years ranges from 79.7%
    to 98.4%, so 2014's 75% is below every one of them. On rising weeks the interval covers
    28%: a rise is caused by rain the model never sees, and no calibration on these inputs can
    single out the rises. A wider interval only buys them with width: 95% covers 62.8% of
    rises (×/÷ 6.5), 99% covers 89.3% (×/÷ 15.5, wider than no model at all). "Low flow now"
    (81.1%) rests on the 0.01 m³/s floor for zero flows: 84.0% at 0.03, 89.7% at 0.1.
    Mondrian conformal by month fixes coverage within groups only when the groups do not drift.

    ## 6. Decide

    Adaptive conformal keeps the long-run 90% (89.8% since 2000). For rises, use a forecast
    with rain in its inputs (the National Weather Service's river forecasts) and calibrate its
    intervals the same way. Report coverage by decade, by year, and by river state.

    **When not to use it:** after the world changes (land use, drainage, climate, dams) without
    recalibrating or adapting; for subgroups it was not calibrated for, and never for groups
    defined by the outcome; for a single year; with few calibration cases (with $n < 9$, no 90%
    interval exists).

    **Impact.** Water infrastructure is designed from past records; "stationarity is dead"
    (Milly et al., 2008). A miss on a rising week falls on the people downstream and the
    utility's customers: state which coverage you promise, and for whom.

    ### Vote: which range for Decatur?

    **A.** The 1970s interval. **B.** Adaptive conformal on the same forecast. **C.** Adaptive
    conformal, plus a rain-aware forecast when storms come. **D.** A 99% interval instead.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                C. B restores the long-run 90% (89.8%) but covers 39.0% of rises; A fell to
                87.9%; D covers 89.3% of rises only with a range of ×/÷ 15.5, wider than no
                model at all (×/÷ 13.8). What would change the answer: a dam upstream, or a
                decade of drought.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where conformal prediction lives today

    - **Language models:** conformal sets of answers with a guaranteed chance of containing a
      correct one (Quach et al., 2024); Lecture 19.
    - **Quantile networks:** conformalized quantile regression adapts the width to each case
      (Romano, Patterson & Candès, 2019); Lecture 10's quantiles, with a guarantee.
    - **Deployed models under shift:** weighted and adaptive conformal (Tibshirani et al., 2019;
      Gibbs & Candès, 2021).

    ## 7. Check yourself

    **Exit question.** Your 90% interval covered 90% of the 1970s and 88% since. A colleague
    proposes widening it to 95% to "fix" the shortfall. What does that fix, what does it not, and
    what would you do instead?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                A 95% interval lifts rises from 28.1% to 62.8%, and all weeks to 94.5%, by
                widening every week's range from ×/÷ 3.6 to ×/÷ 6.5; steady weeks were already
                at 99.7%. Rises stay far below 95% (a 99% interval reaches 89.3%, at ×/÷ 15.5),
                and every week's range is less useful. Instead: adaptive conformal for the
                long-run rate, a rain-aware forecast for rises, and coverage reported by river
                state.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Calibrating on the training data.** The model must be fixed before calibration.
    - **`np.quantile(scores, 0.9)` instead of the $\lceil(n+1)(1-\alpha)\rceil$-th score.** It
      under-covers slightly, badly for small $n$.
    - **Random splits of a time series** for calibration and testing: neighbors share storms;
      hold out blocks.
    - **Reading 90% as conditional.** Report coverage by the groups that matter.
    - **Zero flows on a log scale.** Floor them (here 0.01 m³/s, 135 days) and say so: the
      floor alone moves "low flow now" from 81.1% to 89.7% (at 0.1 m³/s).
    - **The legacy USGS service** (waterservices.usgs.gov) shuts down in February 2027; use the
      Water Data API.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Angelopoulos & Bates (2023), §1–§3:** split conformal, the coverage proof, and
      conformalized quantile regression; §7 covers shift and adaptive methods.
    - **Lei et al. (2018):** distribution-free predictive inference for regression, with the
      upper bound $1-\alpha+1/(n+1)$.
    - **Exercise.** Show that under covariate shift with known density ratio $w(x)$, weighting
      each calibration score by $w(x_i)$ and the test point by $w(x_{n+1})$ restores coverage
      (Tibshirani et al., 2019, Corollary 1; §2.2 states its split-conformal form). Their
      Lemma 1 is the unweighted quantile lemma behind §3's proof.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Split conformal: the $\lceil(n+1)(1-\alpha)\rceil$-th calibration score gives $1-\alpha$
      coverage, if exchangeable.
    - The guarantee is marginal: 90% on average, 28% on rising weeks.
    - Calibrated on the calm 1970s, the interval covered 87.9% after 2000.
    - Adaptive conformal restores the long-run rate; only better inputs fix what the model
      cannot see.

    **Next:** Gaussian processes and surrogates: emulating a water network.
    """)
    return


if __name__ == "__main__":
    app.run()
