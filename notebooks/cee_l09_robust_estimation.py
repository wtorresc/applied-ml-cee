"""Applied Machine Learning for CEE — Lecture 9: Outliers and Robust Estimation.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l09_robust_estimation.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_battledim.sh`` once (BattLeDIM, Vrachimis et al., 2022, CC BY 4.0).
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
    # Applied Machine Learning for CEE — Lecture 9: Outliers and Robust Estimation

    **Slides:** *Applied Machine Learning for CEE*, Lecture 9 (`lectures/cee_l09_robust_estimation`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §11.2 (robust regression: M-estimators and iteratively reweighted least squares).
    **Data:** BattLeDIM (Battle of the Leakage Detection and Isolation Methods; Vrachimis et al., 2022, CC BY 4.0): simulated SCADA readings of the L-Town water network, every 5 minutes, with every leak known. Fetch it with `bash scripts/fetch_battledim.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`, except
    figure readings in the alt text (a range read off a plot). It runs in about 13 seconds on a CPU.

    **Your tasks** (from the slides): write Huber IRLS and check it against statsmodels, write
    RANSAC and the trials formula, run the four breaks, and set the alarm.
    """)
    return


@app.cell
def _():
    import math
    import os
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import torch
    from scipy.stats import norm
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    SENSOR, NEIGHBOR = "n229", "n636"  # the sensor near the burst, and its neighbor
    BURST_PIPE, BURST_START = "p673", "2018-03-05 15:45"
    CAL = ("2018-02-20", "2018-02-27 23:55")  # a calm week near n229: the noise scale
    WINDOW = ("2018-02-28", "2018-03-07 23:55")  # the fitting window
    K_SIGMA = 3.0  # RANSAC threshold, in noise standard deviations
    ALARM_SIGMA = 5.0  # alarm threshold on the hourly mean residual
    HUBER_T = 1.345  # Huber's constant (95% efficiency, normal noise)
    MAD_CONST = float(norm.ppf(0.75))  # 0.6745: MAD / 0.6745 estimates sigma
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        ALARM_SIGMA,
        BURST_PIPE,
        BURST_START,
        CAL,
        HUBER_T,
        HuberRegressor,
        IsolationForest,
        K_SIGMA,
        LinearRegression,
        MAD_CONST,
        MARKERS,
        NEIGHBOR,
        PALETTE,
        Path,
        RANSACRegressor,
        SEED,
        SENSOR,
        WINDOW,
        math,
        norm,
        np,
        pd,
        plt,
        sm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 4–6

    On your own, no notes, two minutes:

    1. A screen flags bridges when $p > 0.5$ and scores 96% accuracy. Is it a good screen?
       What would you report instead?
    2. Each Newton step for logistic regression solves a weighted least squares. What are the
       weights, and which bridges get the largest?
    3. $k$-means spent a center on four buildings that run all night. Error or finding?

    **Today:** when did the pipe burst, and can a fit trust readings that contain the burst?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. No: in 2024, $p > 0.5$ flagged no bridge, and flagging nobody also scores
                   96%. Report what the screen catches and what it flags, or the cost with
                   $r$, and flag above $1/(1+r)$ (Lecture 6).
                2. $p(1-p)$: largest for bridges near $p = 0.5$ (Lecture 6). Today the weights
                   come from the residuals instead, and shrink the points that do not fit.
                3. It depends on the question: an outlier for typical days, the finding for an
                   energy audit (Lecture 4). Today: is a reading an error to ignore, or the
                   event we want?
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A water network's pressure sensors move together with demand; a burst nearby pulls one
    below its neighbors. **The decision:** send a leak crew to the district now, or wait. A
    false alarm costs a crew day; a miss loses water. **The catch:** the readings we fit a
    baseline on may already contain a burst, or a faulty sensor.

    **Data.** L-Town, the BattLeDIM network, was simulated from its hydraulic model with real
    demand patterns and sensor noise, so every leak's pipe, start, and size is published. The
    next cell reads the 2018 readings and the network file.
    """)
    return


@app.cell
def _(Path, pd):
    def find_raw() -> Path:
        """Locate ``data/raw/battledim`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_battledim.sh``).
        """
        rel = Path("data") / "raw" / "battledim"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel / "README.txt").exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_battledim.sh`")

    RAW = find_raw()

    def read(name: str) -> pd.DataFrame:
        """One BattLeDIM CSV: semicolons, decimal commas, 5-minute timestamps."""
        return pd.read_csv(RAW / name, sep=";", decimal=",", index_col=0, parse_dates=True)

    P = read("2018_SCADA_Pressures.csv")
    K = read("2018_Leakages.csv")
    F = read("2018_SCADA_Flows.csv")
    assert P.shape == (105120, 33) and not P.isna().any().any(), P.shape
    return F, K, P, RAW


@app.cell
def _(BURST_PIPE, BURST_START, CAL, F, K, NEIGHBOR, P, RAW, SENSOR, math, pd):
    _counts, _coords, _pipes, _length, _section = {}, {}, {}, 0.0, ""
    for _line in (RAW / "L-TOWN.inp").read_text(errors="replace").splitlines():
        _text = _line.split(";")[0].strip()
        if not _text:
            continue
        if _text.startswith("["):
            _section = _text
            _counts[_section] = 0
            continue
        _counts[_section] += 1
        _f = _text.split()
        if _section == "[COORDINATES]":
            _coords[_f[0]] = (float(_f[1]), float(_f[2]))
        elif _section == "[PIPES]":
            _pipes[_f[0]] = (_f[1], _f[2])
            _length += float(_f[3])

    def m_from(pipe: str, node: str) -> float:
        """Distance (m) from a node to a pipe's midpoint.

        Args:
            pipe: A pipe of ``L-TOWN.inp``.
            node: A junction with coordinates.

        Returns:
            The straight-line distance in meters.
        """
        _a, _b = _pipes[pipe]
        _mid = ((_coords[_a][0] + _coords[_b][0]) / 2, (_coords[_a][1] + _coords[_b][1]) / 2)
        return math.dist(_coords[node], _mid)

    _burst = K.loc[BURST_START:"2018-03-23 10:20", BURST_PIPE]
    _inflow = F["p227"] + F["p235"]
    # The calm week is calm near n229, not leak-free: which leaks run during it, and how far away?
    _calm = K.loc[CAL[0] : CAL[1]]
    _running = _calm.columns[(_calm > 0).any()].tolist()
    _net = {
        "junctions": _counts["[JUNCTIONS]"],
        "pipes": _counts["[PIPES]"],
        "pipe km": round(_length / 1000, 1),
        "pressure sensors": P.shape[1],
        "leaks in 2018": K.shape[1],
        "burst m3/h": round(float(_burst.mean()), 1),
        "burst m3/day": int(round(24 * float(_burst.mean()), -1)),
        "burst % of inflow": round(
            100 * float(_burst.mean()) / float(_inflow[_burst.index].mean())
        ),
        f"{SENSOR} m from burst": int(round(m_from(BURST_PIPE, SENSOR), -1)),
        f"{NEIGHBOR} m from burst": int(round(m_from(BURST_PIPE, NEIGHBOR), -1)),
        "sensors apart (m)": int(round(math.dist(_coords[SENSOR], _coords[NEIGHBOR]), -1)),
        "leaks running in the calm week": len(_running),
        f"nearest of them to {SENSOR} (m)": round(min(m_from(_p, SENSOR) for _p in _running)),
        "burst days": round(len(_burst) / 288, 1),  # 5-minute readings, start to repair
    }
    # Question slide and notes: 782 junctions, 905 pipes (43 km), 33 sensors, 14 leaks; the burst
    # leaked 28.4 m3/h, 680 m3 a day, 13% of the inflow; n229 100 m away, n636 370 m, 340 m apart.
    assert (_net["junctions"], _net["pipes"], _net["pipe km"]) == (782, 905, 43.2)
    assert (_net["leaks in 2018"], _net["burst m3/h"], _net["burst m3/day"]) == (14, 28.4, 680)
    assert _net["burst % of inflow"] == 13
    assert (_net[f"{SENSOR} m from burst"], _net[f"{NEIGHBOR} m from burst"]) == (100, 370)
    assert _net["sensors apart (m)"] == 340
    # Notes of "Lost water is lost money and energy": 680 m3 a day for 18 days (17.8).
    assert (_burst > 0).all() and _net["burst days"] == 17.8
    # Notes ("How many trials?", "Not for slow leaks"): three slow leaks run during the calm
    # week, none of them the burst, the nearest 566 m from n229.
    assert BURST_PIPE not in _running and _net["leaks running in the calm week"] == 3
    assert _net[f"nearest of them to {SENSOR} (m)"] == 566
    pd.Series(_net, name="L-Town").to_frame()
    return (m_from,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: did a pipe burst?

    Hourly mean pressures at sensor n229 and its neighbor n636, and the town's inflow, at
    14:00 and 17:00 on two days. **Did a pipe burst? If so, near which sensor, and when?**
    Write down the evidence you used.
    """)
    return


@app.cell
def _(F, NEIGHBOR, P, SENSOR, pd, w_clean):
    _H = P[[SENSOR, NEIGHBOR]].resample("h").mean()
    _inflow = (F["p227"] + F["p235"]).resample("h").mean()
    _times = ["2018-03-04 14:00", "2018-03-04 17:00", "2018-03-05 14:00", "2018-03-05 17:00"]
    _table = pd.DataFrame(
        {
            _t: [
                round(float(_H.loc[_t, SENSOR]), 2),
                round(float(_H.loc[_t, NEIGHBOR]), 2),
                round(float(_inflow[_t])),
            ]
            for _t in _times
        },
        index=[f"sensor {SENSOR} (m)", f"neighbor {NEIGHBOR} (m)", "town inflow (m3/h)"],
    )
    assert _table.iloc[0].tolist() == [52.48, 51.98, 51.8, 50.64]
    assert _table.iloc[1].tolist() == [45.47, 45.07, 44.91, 44.6]
    assert _table.iloc[2].tolist() == [188, 225, 242, 257]
    # The answer's arithmetic, from the table: n229 fell 1.16 m, n636 0.31 m; the calm slope 1.24
    # predicts 0.38 m of n229's drop, so 0.78 m is left unexplained (0.85 m beyond n636's drop).
    _d_sensor = round(_table.iloc[0, 3] - _table.iloc[0, 2], 2)
    _d_neighbor = round(_table.iloc[1, 3] - _table.iloc[1, 2], 2)
    _predicted = round(round(float(w_clean[1]), 2) * _d_neighbor, 2)
    assert (_d_sensor, _d_neighbor, _predicted) == (-1.16, -0.31, -0.38)
    assert round(_d_sensor - _predicted, 2) == -0.78 and round(_d_sensor - _d_neighbor, 2) == -0.85
    _table
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                From 14:00 to 17:00 on March 4, n229 fell 0.50 m and its neighbor 0.40 m: the
                usual evening. On March 5, n229 fell 1.16 m and its neighbor only 0.31 m. The
                calm line predicts $1.24 \times 0.31 = 0.38$ m for n229, so n229 lost 0.78 m more
                than demand explains (0.85 m more than its neighbor). A burst near n229 between
                14:00 and 17:00; it began at 15:45, about 100 m from n229. You compared n229 with what its neighbor predicts: that is the
                model. The question is how to fit it when the data contain the burst.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with the **loss** as this lecture's step: the neighbor's pressure $x$ →
    decoder, a line $w_0 + w_1 x$ → $\hat y$; the residual $r = y - \hat y$ against sensor
    n229's reading $y$; an alarm when the hourly mean of $r$ falls below $-5\sigma$. The loss
    that fits the line, Huber weights or a RANSAC consensus, decides which readings count as
    normal. §4 recomputes the slide's trace at 17:00 on March 5.

    ## 3. Derive

    **Assumption:** most readings follow one line; a minority, the outliers, follow anything.

    **Least squares** minimizes $\sum_i r_i^2$. Its estimating equation is
    $\sum_i r_i\,\mathbf{a}_i = \mathbf{0}$: each reading pulls in proportion to its residual,
    so one reading can move the line anywhere (breakdown point 0).

    **Huber** (1964) replaces $r^2/2$ by

    $$\rho(r) = \begin{cases} r^2/2 & |r| \le \delta \\ \delta|r| - \delta^2/2 & |r| > \delta
      \end{cases}, \qquad \psi(r) = \rho'(r) = \mathrm{clip}(r, -\delta, \delta),$$

    convex, quadratic near zero and linear in the tails, so a far reading pulls with at most
    $\delta$. The scale: $\delta = 1.345\,\hat\sigma$, with
    $\hat\sigma = \mathrm{median}\,|r_i| / 0.6745$ (the MAD estimates $\sigma$ for normal noise
    and ignores up to half the points); 1.345 keeps 95% of least squares' efficiency under
    normal noise.

    **IRLS.** The estimating equation $\sum_i \psi(r_i)\mathbf{a}_i = \mathbf{0}$ with
    $\psi(r) = w(r)\,r$ reads

    $$\sum_i w_i\, r_i\, \mathbf{a}_i = \mathbf{0}, \qquad w_i = \min\!\Big(1, \frac{\delta}{|r_i|}\Big):$$

    the normal equations of a **weighted least squares**. Fix the weights, solve, update the
    weights from the new residuals, repeat: the loop of Lecture 6's Newton steps, with other
    weights (Forsyth §11.2.1).

    **RANSAC** (Fischler & Bolles, 1981): draw 2 readings, fit the line through them, count the
    readings within $\tau$ of it, keep the largest set, refit on it by least squares. The
    number of trials for probability $p$ of at least one all-inlier pair, with outlier share
    $\varepsilon$ and sample size $s$:

    $$N = \frac{\ln(1-p)}{\ln\big(1-(1-\varepsilon)^s\big)},$$

    because one trial is clean with probability $(1-\varepsilon)^s$, and all $N$ fail with
    probability $(1-(1-\varepsilon)^s)^N = 1 - p$.
    """)
    return


@app.cell
def _(math):
    def trials_needed(eps: float, s: int = 2, p: float = 0.99) -> int:
        """RANSAC trials for probability ``p`` of one all-inlier sample of size ``s``."""
        return math.ceil(math.log(1 - p) / math.log(1 - (1 - eps) ** s))

    # Your turn: 7, 17, and 49 trials at 30%, 50%, and 70% outliers.
    assert [trials_needed(_e) for _e in (0.3, 0.5, 0.7)] == [7, 17, 49]
    print({_e: trials_needed(_e) for _e in (0.3, 0.5, 0.7)})
    return (trials_needed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    Huber by IRLS (the slide's listing) and RANSAC for a line, from scratch.
    """)
    return


@app.cell
def _(HUBER_T, MAD_CONST, np):
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
            Weights, steps taken, and the final scale (MAD of the residuals about 0, over
            0.6745, as statsmodels' RLM computes it).

        Raises:
            RuntimeError: If the iteration has not converged.
        """
        w = np.linalg.lstsq(A, y, rcond=None)[0]  # start: least squares
        for step in range(1, max_steps + 1):
            r = y - A @ w
            s = max(np.median(np.abs(r)) / MAD_CONST, 1e-8)  # robust scale, never 0
            u = np.abs(r) / s
            wt = np.where(u <= t, 1.0, t / u)  # Huber weights
            sw = np.sqrt(wt)
            w_new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
            if np.max(np.abs(w_new - w)) < tol:
                return w_new, step, float(s)
            w = w_new
        raise RuntimeError("IRLS did not converge")

    def ransac_line(
        A: np.ndarray, y: np.ndarray, tau: float, trials: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """RANSAC for a line: fit pairs, keep the largest consensus, refit on it.

        Returns:
            Weights refit by least squares on the best consensus, and its inlier mask.
        """
        best = np.zeros(len(y), dtype=bool)
        for _ in range(trials):
            i, j = rng.choice(len(y), size=2, replace=False)  # minimal sample: 2 points
            if A[i, 1] == A[j, 1]:
                continue
            w = np.linalg.solve(A[[i, j]], y[[i, j]])
            inl = np.abs(y - A @ w) < tau
            if inl.sum() > best.sum():
                best = inl
        return np.linalg.lstsq(A[best], y[best], rcond=None)[0], best

    return huber_irls, ransac_line


@app.cell
def _(BURST_START, CAL, K_SIGMA, MAD_CONST, NEIGHBOR, P, SENSOR, WINDOW, np, trials_needed):
    def design(frame):
        """``A = [1, neighbor]`` and ``y = sensor``."""
        _A = np.column_stack([np.ones(len(frame)), frame[NEIGHBOR].to_numpy()])
        return _A, frame[SENSOR].to_numpy()

    _A_cal, _y_cal = design(P.loc[CAL[0] : CAL[1]])
    w_cal = np.linalg.lstsq(_A_cal, _y_cal, rcond=None)[0]  # the trusted baseline for the alarm
    _r_cal = _y_cal - _A_cal @ w_cal
    sigma = float(np.std(_r_cal))
    _mad_scale_cal = float(np.median(np.abs(_r_cal)) / MAD_CONST)
    tau = K_SIGMA * sigma
    win = P.loc[WINDOW[0] : WINDOW[1]]
    A, yw = design(win)
    burst = win.index >= BURST_START
    w_clean = np.linalg.lstsq(A[~burst], yw[~burst], rcond=None)[0]
    # Notes: sigma 0.017 m from the calm week (its MAD scale: 0.016 m), tau 0.05 m; 2,304
    # readings, 675 of the burst (29%), so 7 to 9 RANSAC trials suffice.
    assert round(sigma, 4) == 0.0167 and round(tau, 2) == 0.05 and round(_mad_scale_cal, 3) == 0.016
    assert (len(yw), int(burst.sum()), round(100 * float(burst.mean()))) == (2304, 675, 29)
    assert (trials_needed(float(burst.mean())), trials_needed(float(burst.mean()) + 0.05)) == (7, 9)
    assert round(float(w_clean[1]), 4) == 1.2377
    print(
        f"sigma {sigma:.4f} m (MAD scale {_mad_scale_cal:.4f} m), tau {tau:.3f} m; "
        f"clean slope {w_clean[1]:.4f}"
    )
    return A, burst, design, sigma, tau, w_cal, w_clean, yw


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    On the burst window, will Huber's slope be near the calm 1.24, or near least squares' 1.63?
    The reference is statsmodels' `RLM` with the `HuberT` norm and the same MAD scale.
    """)
    return


@app.cell
def _(A, HUBER_T, HuberRegressor, burst, huber_irls, np, ransac_line, sm, tau, w_clean, yw):
    w_ols = np.linalg.lstsq(A, yw, rcond=None)[0]
    w_hub, _steps, _scale = huber_irls(A, yw)
    _rlm = sm.RLM(yw, A, M=sm.robust.norms.HuberT(t=HUBER_T)).fit(
        scale_est="mad", tol=1e-12, maxiter=500, conv="coefs"
    )
    _rel = float(np.max(np.abs(w_hub - _rlm.params) / np.abs(_rlm.params)))
    assert np.allclose(w_hub, _rlm.params, rtol=1e-5, atol=0) and _rel < 1e-12, _rel
    # alpha=0: scikit-learn's default adds an L2 penalty, which is not Huber's estimator.
    _sk = HuberRegressor(epsilon=HUBER_T, alpha=0.0, max_iter=2000).fit(A[:, 1:], yw)
    w_ran, _inl = ransac_line(A, yw, tau, trials=200, rng=np.random.default_rng(0))
    # The calm fit's standard error. Successive residuals are correlated, so the formula for
    # independent errors is too small; Newey-West (HAC, 24 lags = 2 hours) allows for it.
    _ols = sm.OLS(yw[~burst], A[~burst])
    _hac = _ols.fit(cov_type="HAC", cov_kwds={"maxlags": 24, "use_correction": True})
    # A one-day window spans the daily cycle, whose negative correlations offset the positive.
    _hac_day = _ols.fit(cov_type="HAC", cov_kwds={"maxlags": 288, "use_correction": True})
    _r_clean = yw[~burst] - A[~burst] @ w_clean
    _rho = [float(np.corrcoef(_r_clean[:-_k], _r_clean[_k:])[0, 1]) for _k in (1, 288)]
    # Slide "The check passes; Huber does not": RLM to 1e-12 in 17 steps; Huber 1.65 (sklearn's
    # 1.66), scale 0.33 m; least squares 1.63; RANSAC 1.24 with 1,626 inliers, none in the burst;
    # notes: the calm fit 1.238, standard error 0.0023 (HAC, 2-hour window; 0.0009 with a
    # one-day window; 0.0012 assuming independence), residual autocorrelation 0.44 at 5 minutes
    # and 0.33 at a day.
    assert _steps == 17 and round(_scale, 2) == 0.33
    assert (
        round(float(w_ols[1]), 2),
        round(float(w_hub[1]), 2),
        round(float(_sk.coef_[0]), 2),
    ) == (1.63, 1.65, 1.66)
    assert (
        round(float(w_ran[1]), 2) == 1.24
        and int(_inl.sum()) == 1626
        and int(_inl[burst].sum()) == 0
    )
    assert round(float(w_clean[1]), 3) == 1.238 and np.allclose(
        _hac.params, w_clean, rtol=1e-9, atol=0
    )
    assert (
        round(float(_hac.bse[1]), 4),
        round(float(_hac_day.bse[1]), 4),
        round(float(_ols.fit().bse[1]), 4),
    ) == (0.0023, 0.0009, 0.0012)
    assert [round(_v, 2) for _v in _rho] == [0.44, 0.33]
    print(
        f"IRLS vs RLM: max relative difference {_rel:.0e}, {_steps} steps; slopes: least squares "
        f"{w_ols[1]:.3f}, Huber {w_hub[1]:.3f} (sklearn {_sk.coef_[0]:.3f}), RANSAC {w_ran[1]:.3f}"
    )
    return w_hub, w_ols, w_ran


@app.cell
def _(A, np, tau, yw):
    # Slide "RANSAC, trial by trial": two random pairs from the burst window, then the best.
    _rng = np.random.default_rng((0, 9))
    _counts = []
    for _ in range(2):
        _i, _j = _rng.choice(len(yw), size=2, replace=False)
        _w = np.linalg.solve(A[[_i, _j]], yw[[_i, _j]])
        _counts.append(int((np.abs(yw - A @ _w) < tau).sum()))
    assert _counts == [956, 324]
    _counts
    return


@app.cell
def _(BURST_START, NEIGHBOR, P, SENSOR, pd, sigma, w_ran):
    # The slide's trace: the single reading at 17:00 on March 5, 75 minutes after the burst (the
    # attempt's table shows that hour's mean, 50.64 m).
    _x, _y = float(P.loc["2018-03-05 17:00", NEIGHBOR]), float(P.loc["2018-03-05 17:00", SENSOR])
    assert (pd.Timestamp("2018-03-05 17:00") - pd.Timestamp(BURST_START)).seconds == 75 * 60
    _pred = float(w_ran[0] + w_ran[1] * _x)
    assert (round(_x, 2), round(_y, 2), round(_pred, 2), round(_y - _pred, 2)) == (
        44.64,
        50.75,
        51.46,
        -0.71,
    )
    assert round((_y - _pred) / sigma) == -43 and (
        round(float(w_ran[0]), 3),
        round(float(w_ran[1]), 4),
    ) == (-3.786, 1.2376)
    # The diagram's arithmetic reproduces from the terms it shows: -3.786 + 1.2376 x 44.64.
    assert round(-3.786 + 1.2376 * 44.64, 2) == 51.46
    print(
        f"x = {_x:.2f}, y-hat = {_pred:.2f}, y = {_y:.2f}, r = {_y - _pred:.2f} m = {(_y - _pred) / sigma:.0f} sigma"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The code is right; the method is wrong for this window.** The burst is 29% of the
    readings: not a few isolated bad points but a second line, parallel to the first and 0.73 m
    below it, and it inflated the MAD scale from the calm week's 0.016 m to 0.33 m, so most burst
    readings look ordinary to Huber.

    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **A fault.** Three readings of n636 recorded as 0 m among 1,629 calm ones. Slopes of
       least squares, Huber, RANSAC?
    2. **The burst.** Which line does each method fit through the burst window, and how big a
       burst does each report?
    3. **The threshold.** RANSAC with scikit-learn's default threshold, over 20 seeds?
    4. **The majority.** Grow the window until the burst is 65% of it. Which line does RANSAC
       keep?
    """)
    return


@app.cell
def _(BURST_START, HUBER_T, P, WINDOW, design, huber_irls, np, pd, ransac_line, tau):
    # Break 1: a telemetry dropout, injected into the calm part of the window (up to the last
    # reading before the burst).
    _calm_end = pd.Timestamp(BURST_START) - pd.Timedelta(minutes=5)
    _A, _y = design(P.loc[WINDOW[0] : _calm_end])
    assert (P.loc[WINDOW[0] : _calm_end].index < BURST_START).all()
    _ref = np.linalg.lstsq(_A, _y, rcond=None)[0]
    _Ad = _A.copy()
    _Ad[[100, 700, 1300], 1] = 0.0
    _ols = np.linalg.lstsq(_Ad, _y, rcond=None)[0]
    _hub, _, _hub_scale = huber_irls(_Ad, _y)
    _ran, _ = ransac_line(_Ad, _y, tau, trials=200, rng=np.random.default_rng(0))
    # Notes: each zero lies about 45 m from the others in x (they average 45.44 m); its residual
    # from Huber's line is about 55 m (55.18 to 55.39), so its weight is about 0.0005.
    _zeros = [100, 700, 1300]
    _r0 = np.abs(_y[_zeros] - _Ad[_zeros] @ _hub)
    _wt0 = np.minimum(1.0, HUBER_T * _hub_scale / _r0)
    assert round(float(_A[:, 1].mean()), 2) == 45.44
    assert (round(float(_r0.min()), 2), round(float(_r0.max()), 2)) == (55.18, 55.39)
    assert float(f"{_wt0.max():.1e}") == 4.8e-4 and round(float(_wt0.max()), 4) == 0.0005
    # Slide "One dropout breaks least squares": 1.24 -> 0.05; Huber 1.22; RANSAC 1.24.
    assert len(_y) == 1629 and round(float(_ref[1]), 2) == 1.24
    assert (round(float(_ols[1]), 2), round(float(_hub[1]), 2), round(float(_ran[1]), 2)) == (
        0.05,
        1.22,
        1.24,
    )
    print(
        f"dropouts: least squares {_ols[1]:.3f}, Huber {_hub[1]:.3f}, RANSAC {_ran[1]:.3f} (clean {_ref[1]:.3f})"
    )
    return


@app.cell
def _(A, MARKERS, PALETTE, burst, np, plt, w_clean, w_hub, w_ols, w_ran, yw):
    _shift = {}
    for _name, _w in (
        ("calm fit", w_clean),
        ("least squares", w_ols),
        ("Huber", w_hub),
        ("RANSAC", w_ran),
    ):
        _r = yw - A @ _w
        _shift[_name] = round(float(_r[burst].mean() - _r[~burst].mean()), 2)
    # Slide "A burst is a second line": 0.53 and 0.52 m reported, 0.73 m by RANSAC and the calm fit.
    assert _shift == {"calm fit": -0.73, "least squares": -0.53, "Huber": -0.52, "RANSAC": -0.73}
    # Notes of "RANSAC is only as good as tau": half the gap between the bands, 0.36 m (from the
    # unrounded gap), is below the default threshold, 0.44 m.
    _r_ran = yw - A @ w_ran
    assert round(abs(float(_r_ran[burst].mean() - _r_ran[~burst].mean())) / 2, 2) == 0.36
    # Alt text: the calm readings span n636 = 44.63 to 46.16 m, where the clean line gives 51.45
    # and 53.34 m.
    _ends = np.array([A[~burst, 1].min(), A[~burst, 1].max()])
    assert [round(float(_v), 2) for _v in _ends] == [44.63, 46.16]
    assert [round(float(w_clean[0] + w_clean[1] * _v), 2) for _v in _ends] == [51.45, 53.34]

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(A[~burst, 1], yw[~burst], ".", ms=3, color="grey", alpha=0.5, label="before the burst")
    _ax.plot(
        A[burst, 1], yw[burst], "x", ms=3, color=PALETTE[3], alpha=0.5, label="after 15:45, Mar 5"
    )
    _xs = np.linspace(A[:, 1].min(), A[:, 1].max(), 50)
    for _k, (_name, _w) in enumerate(
        (("least squares", w_ols), ("Huber", w_hub), ("RANSAC", w_ran))
    ):
        _ax.plot(
            _xs,
            _w[0] + _w[1] * _xs,
            color=PALETTE[_k],
            marker=MARKERS[_k],
            markevery=10,
            label=f"{_name}: slope {_w[1]:.2f}",
        )
    _ax.set_xlabel("n636 pressure (m)")
    _ax.set_ylabel("n229 pressure (m)")
    _ax.legend(fontsize=9)
    print("burst size reported (m):", _shift)
    _fig
    return


@app.cell
def _(A, RANSACRegressor, np, tau, w_clean, yw):
    _default = [
        float(RANSACRegressor(random_state=_s).fit(A[:, 1:], yw).estimator_.coef_[0])
        for _s in range(20)
    ]
    _with_tau = [
        float(
            RANSACRegressor(residual_threshold=tau, random_state=_s)
            .fit(A[:, 1:], yw)
            .estimator_.coef_[0]
        )
        for _s in range(20)
    ]
    _mad_y = float(np.median(np.abs(yw - np.median(yw))))
    _bad = sum(abs(_v - w_clean[1]) > 0.05 for _v in _default)
    # Slide "RANSAC is only as good as tau": default 0.44 m; 13 of 20 seeds off (1.15 to 1.99);
    # with tau all 20 agree, 1.230 to 1.242.
    assert round(_mad_y, 2) == 0.44 and _bad == 13
    assert (round(min(_default), 2), round(max(_default), 2)) == (1.15, 1.99)
    assert (round(min(_with_tau), 3), round(max(_with_tau), 3)) == (1.23, 1.242)
    print(
        f"default threshold {_mad_y:.2f} m: {_bad} of 20 seeds off; with tau: "
        f"{min(_with_tau):.3f} to {max(_with_tau):.3f}"
    )
    return


@app.cell
def _(BURST_START, P, WINDOW, design, np, pd, ransac_line, tau):
    _rows = []
    for _end in (
        "2018-03-06 23:55",
        "2018-03-08 23:55",
        "2018-03-10 23:55",
        "2018-03-11 23:55",
        "2018-03-12 23:55",
        "2018-03-15 23:55",
    ):
        _w = P.loc[WINDOW[0] : _end]
        _A, _y = design(_w)
        _b = _w.index >= BURST_START
        _ran, _inl = ransac_line(_A, _y, tau, trials=500, rng=np.random.default_rng(0))
        _ols = np.linalg.lstsq(_A, _y, rcond=None)[0]
        # Each group's own best line (2,000 trials on the calm readings alone, then on the burst
        # readings alone): how many of the group lie within tau of it.
        _, _own_calm = ransac_line(_A[~_b], _y[~_b], tau, 2000, np.random.default_rng(0))
        _, _own_burst = ransac_line(_A[_b], _y[_b], tau, 2000, np.random.default_rng(0))
        _rows.append(
            {
                "window ends": _end[:10],
                "burst %": round(100 * float(_b.mean())),
                "least squares": round(float(_ols[1]), 2),
                "RANSAC": round(float(_ran[1]), 2),
                "calm residual (m)": round(float((_y - _A @ _ran)[~_b].mean()), 2),
                "calm line holds": int(_own_calm.sum()),
                "burst line holds": int(_own_burst.sum()),
                "burst line holds %": round(100 * float(_own_burst.mean())),
            }
        )
    _tab = pd.DataFrame(_rows).set_index("window ends")
    # Slide "The biggest tight group wins": RANSAC keeps 1.24 through 19, 37, 49, 53, 56%; at 65%
    # it jumps to 1.35 and the calm days look 0.68 m high; least squares 1.58 to 1.66.
    assert _tab["burst %"].tolist() == [19, 37, 49, 53, 56, 65]
    assert _tab["RANSAC"].tolist() == [1.24] * 5 + [1.35]
    assert _tab["calm residual (m)"].iloc[-1] == 0.68
    assert (_tab["least squares"].min(), _tab["least squares"].max()) == (1.58, 1.66)
    # Notes: the calm line holds 1,627 of the 1,629 calm readings; the burst's own best line only
    # 76 to 78% of the burst readings. At 56% the calm line wins by 26 (1,627 against 1,601); at
    # 65% the burst line holds 2,281 and wins. RANSAC keeps whichever holds more.
    assert _tab["calm line holds"].tolist() == [1627] * 6
    assert _tab["burst line holds"].tolist() == [301, 752, 1173, 1383, 1601, 2281]
    assert _tab["burst line holds %"].tolist() == [78, 78, 76, 76, 76, 77]
    _kept_calm = _tab["RANSAC"] == 1.24
    assert (_kept_calm == (_tab["calm line holds"] > _tab["burst line holds"])).all()
    _tab
    return


@app.cell
def _(mo):
    mo.md(r"""
    **RANSAC keeps the line with the most readings within $\tau$, not the majority's line.** The
    burst readings spread more about any one line: the best line through them alone holds 76 to
    78% of them within 0.05 m, the best through the calm readings 1,627 of 1,629. So the calm
    line survives past one half, narrowly: at 56% burst it holds 1,627 readings against the
    burst line's 1,601. At 65% the burst line holds 2,281 and wins. Once the burst dominates, a
    model refit on recent data learns the leak as normal.

    ## 6. Decide

    **The alarm:** the calm week's line, fitted once on a trusted week before the burst (a
    causal baseline), and an alarm when the mean residual of the last hour (12 readings) falls
    below $-5\sigma$. RANSAC's line from the burst window, fitted through March 7, gives the same
    alarms: the robust fit recovered the calm line.

    **And one miss.** After the repair, March 24 to December, the alarm never sounds: no false
    alarm, but also nothing during October's leak in pipe p158 (24.4 m³/h, nearly the burst's
    28.4), 220 m from n229 and 140 m from n636. Both sensors fell, so the residual between them
    barely moved. A pair alarm watches the gap between two sensors; a leak that lowers both
    leaves the gap alone.
    """)
    return


@app.cell
def _(
    ALARM_SIGMA,
    BURST_START,
    K,
    NEIGHBOR,
    P,
    PALETTE,
    SENSOR,
    design,
    m_from,
    norm,
    pd,
    plt,
    sigma,
    w_cal,
    w_ran,
):
    # One residual series over all of 2018, so every hourly mean covers a full hour.
    _A, _y = design(P)
    _start = pd.Timestamp(BURST_START)

    def _alarms(w):
        """Residuals, their hourly mean, and the alarm counts for one baseline line."""
        _res = pd.Series(_y - _A @ w, index=P.index)
        _hour = _res.rolling(12).mean()
        _on = _hour < -ALARM_SIGMA * sigma
        _counts = (
            _on[_start:].idxmax(),  # the first alarm after the burst
            int(_on[_start - pd.Timedelta(days=7) : _start - pd.Timedelta(minutes=5)].sum()),
            len(_on[_start - pd.Timedelta(days=7) : _start - pd.Timedelta(minutes=5)]),
            int(_on[: _start - pd.Timedelta(minutes=5)].sum()),
            int(_on["2018-03-24":].sum()),
        )
        return _res, _hour, _counts

    # The calm week's line is the alarm's baseline; RANSAC's window line must agree.
    _r_all, _hourly_all, _counts = _alarms(w_cal)
    assert _counts == _alarms(w_ran)[2], (_counts, _alarms(w_ran)[2])
    _first = _counts[0]
    # Slide "Alarm in five minutes": -5 sigma = -0.08 m; first alarm 15:50; none in the week
    # before (2,016 readings) nor earlier in 2018; none from March 24 to December.
    assert round(ALARM_SIGMA * sigma, 2) == 0.08 and str(_first) == "2018-03-05 15:50:00"
    assert _counts[1:] == (0, 2016, 0, 0), _counts
    # Notes: with normal noise, one reading in 3.5 million falls below -5 sigma.
    assert round(1 / norm.cdf(-ALARM_SIGMA) / 1e6, 1) == 3.5

    # The miss. The leak nearest n229 after the repair is October's p158 (24.4 m3/h), 220 m from
    # n229 and 140 m from n636. Over its first 10 days both sensors fell (0.31 and 0.24 m against
    # the 10 days before), so the residual moved only from -0.013 to -0.019 m, and its lowest
    # hourly mean (-0.066 m) stayed above the threshold: no alarm.
    _after = K.loc["2018-03-24":]
    _running = _after.columns[(_after > 0).any()].tolist()
    assert min(_running, key=lambda _p: m_from(_p, SENSOR)) == "p158"
    _flow = K["p158"][K["p158"] > 0]
    _t0, _t1 = _flow.index[0], _flow.index[-1]
    _days, _step = pd.Timedelta(days=10), pd.Timedelta(minutes=5)
    _pair = [SENSOR, NEIGHBOR]
    _drop = (
        P.loc[_t0 : _t0 + _days - _step, _pair].mean()
        - P.loc[_t0 - _days : _t0 - _step, _pair].mean()
    )
    _hour_in = _hourly_all[_t0 + 11 * _step : _t1]  # hourly means wholly inside the leak
    assert round(float(_flow.mean()), 1) == 24.4
    assert (round(m_from("p158", SENSOR), -1), round(m_from("p158", NEIGHBOR), -1)) == (220, 140)
    assert [round(float(_v), 2) for _v in _drop] == [-0.31, -0.24]
    assert round(float(_r_all[_t0 - _days : _t0 - _step].mean()), 3) == -0.013
    assert round(float(_r_all[_t0:_t1].mean()), 3) == -0.019
    assert round(float(_hour_in.min()), 3) == -0.066
    assert not (_hour_in < -ALARM_SIGMA * sigma).any()
    print(
        f"p158, {_t0} to {_t1}: n229 fell {-_drop[SENSOR]:.2f} m, n636 {-_drop[NEIGHBOR]:.2f} m; "
        f"lowest hourly residual {_hour_in.min():.3f} m against {-ALARM_SIGMA * sigma:.3f} m"
    )

    _r = _r_all["2018-03-03":"2018-03-08 23:55"]
    _hourly = _hourly_all["2018-03-03":"2018-03-08 23:55"]
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.plot(_r.index, _r, color="#B0B0B0", lw=0.8, label="residual, every 5 min")
    _ax.plot(_hourly.index, _hourly, "-", color=PALETTE[0], lw=2, label="mean of the last hour")
    _ax.axhline(-ALARM_SIGMA * sigma, color=PALETTE[1], ls="--", label="alarm")
    _ax.axvline(pd.Timestamp(BURST_START), color=PALETTE[3], ls=":")
    _ax.set_ylabel("n229 below its line (m)")
    _ax.legend(loc="lower left", fontsize=9)
    print(
        f"first alarm {_first}, "
        f"{int((_first - _start).total_seconds() // 60)} minutes after the burst"
    )
    _fig
    return


@app.cell
def _(CAL, F, IsolationForest, K, LinearRegression, P, SEED, WINDOW, np, pd):
    _cal = P.loc[CAL[0] : CAL[1]]
    _R = pd.DataFrame(index=P.index)
    for _c in P.columns:
        _others = [_o for _o in P.columns if _o != _c]
        _m = LinearRegression().fit(_cal[_others], _cal[_c])
        _R[_c] = P[_c] - _m.predict(P[_others])
    _H = _R.resample("h").mean()
    _Hcal = _H.loc[CAL[0] : "2018-02-27 23:00"]
    _iso = IsolationForest(n_estimators=300, random_state=SEED).fit(_Hcal)
    _thr = np.quantile(-_iso.score_samples(_Hcal), 0.99)
    _on = pd.Series(-_iso.score_samples(_H) > _thr, index=_H.index)
    _monthly = (100 * _on.resample("ME").mean()).round()
    # Slide "Not for slow leaks or new normals": 13% of the hours before the burst; 92 to 100% of
    # each month from May.
    assert round(100 * float(_on.loc[WINDOW[0] : "2018-03-05 14:00"].mean())) == 13
    assert (_monthly.loc["2018-05":].min(), _monthly.loc["2018-05":].max()) == (92, 100)
    # What changed meanwhile: from May the inflow ran 7 to 30% above the training week's 175.2
    # m3/h, and 4 to 6 leaks were running. Both change, so the counts cannot say which it flags.
    _inflow = F["p227"] + F["p235"]
    _inflow_cal = float(_inflow[CAL[0] : CAL[1]].mean())
    _above = 100 * (_inflow.resample("ME").mean().loc["2018-05":] / _inflow_cal - 1)
    _leaks_month = (K > 0).resample("ME").max().sum(axis=1).loc["2018-05":]
    assert round(_inflow_cal, 1) == 175.2
    assert (round(float(_above.min())), round(float(_above.max()))) == (7, 30)
    assert (int(_leaks_month.min()), int(_leaks_month.max())) == (4, 6)
    _monthly.rename("isolation-forest alarms, % of hours").to_frame().T
    return


@app.cell
def _(mo):
    mo.md(r"""
    **An anomaly detector flags whatever differs from its training week.** The isolation forest
    (Liu, Ting & Zhou, 2008) isolates unusual points with few random splits. Trained on one calm
    week, it flagged 92 to 100% of hours in every month from May. Meanwhile the inflow ran 7 to
    30% above the training week's and 4 to 6 leaks were running: both changed, so the counts
    cannot say which the forest flags. Eight of 2018's 14 leaks were incipient, growing for 4 to
    98 days (median 50) before reaching full size (checked below from the dataset's
    configuration). "Calm" means calm near n229: three slow leaks were already running elsewhere
    in the town that week, the nearest 566 m from n229, and in the days before the burst two of
    them (p461, p427) were still growing.

    **When not to use these methods.** Not when the outliers are a majority; not a pair alarm
    alone, which missed a leak both sensors felt; not without separating faults (one sensor
    jumps) from leaks (a sensor departs from its neighbors). Advice the lecture does not
    measure: do not refit a baseline on recent data during a slow leak.

    **Impact.** Worldwide, utilities lose about 126 billion m³ of treated water a year to leaks,
    theft, and metering errors, worth about US\$39 billion (Liemberger & Wyatt, 2019); every lost
    cubic meter was pumped and treated, and customers pay for it through their bills.

    ### Vote: which baseline for the alarm?

    Your fitting week may contain an unknown event. **A.** Least squares on the last week.
    **B.** Huber on the last week. **C.** RANSAC with $\tau$ from a calm, checked week.
    **D.** RANSAC with the library's default threshold.
    """)
    return


@app.cell
def _(RAW, WINDOW, pd):
    import yaml

    # The ground truth: each leak's start, kind, and peak (full size) time.
    _config = yaml.safe_load((RAW / "dataset_configuration.yaml").read_text())
    _rows = [[_f.strip() for _f in str(_e).split(",")] for _e in _config["leakages"] if _e]
    _leaks = pd.DataFrame(_rows, columns=["pipe", "start", "end", "diameter_m", "kind", "peak"])
    _leaks = _leaks[pd.to_datetime(_leaks["start"]).dt.year == 2018]
    _grow = _leaks[_leaks["kind"] == "incipient"]
    _days = (
        pd.to_datetime(_grow["peak"]) - pd.to_datetime(_grow["start"])
    ).dt.total_seconds() / 86400
    # Notes of "Not for slow leaks or new normals": 8 of 14 incipient, 4 to 98 days (median 50).
    assert (len(_leaks), len(_grow)) == (14, 8)
    assert [round(float(_v)) for _v in (_days.min(), _days.median(), _days.max())] == [4, 50, 98]
    # In the days before the burst (the isolation forest's 13%) no leak started, but two incipient
    # leaks were still growing: p461 (to March 27) and p427 (to May 14).
    _t0, _t1 = pd.Timestamp(WINDOW[0]), pd.Timestamp("2018-03-05 14:00")
    _start, _end, _peak = (pd.to_datetime(_grow[_c]) for _c in ("start", "end", "peak"))
    _growing = _grow[(_start < _t0) & (_end > _t0) & (_peak > _t1)]
    _started = pd.to_datetime(_leaks["start"]).between(_t0, _t1)
    assert sorted(_growing["pipe"]) == ["p427", "p461"] and not _started.any()
    _grow[["pipe", "start", "peak"]]
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                C. On the burst window, A and B fitted 1.63 and 1.65 and shrank the burst to
                0.53 and 0.52 m; D was off for 13 of 20 seeds; C recovered the calm line and the
                full 0.73 m. What would change the answer: a window that is mostly burst (C fails
                too: fit on a trusted period), or isolated dropouts only (B is enough).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where robust losses live today

    - **Object detectors** regress boxes with smooth L1, the Huber loss with $\delta = 1$
      (Fast R-CNN; Girshick, 2015); Lectures 15–16.
    - **Deep Q-networks** clip the temporal-difference error to $[-1, 1]$, a Huber loss
      (Mnih et al., 2015); Lecture 26.
    - **RANSAC** aligns point clouds and estimates camera poses from feature matches full of
      outliers, as in COLMAP (Schönberger & Frahm, 2016); Lecture 22.
    - **Gradient boosting** with the Huber loss resists bad labels (Friedman, 2001); Lecture 8.

    ## 7. Check yourself

    **Exit question.** Your SCADA vendor proposes refitting each sensor's baseline every night
    on the last seven days, with least squares. Name two ways it will miss a leak, and one
    change you would make.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                A burst in the last week pulls least squares toward it (on the burst window it
                reported 0.53 m for 0.73 m); a slow leak becomes part of each new baseline; a
                dropout tilts the line. Changes: fit on a trusted, checked period; use RANSAC
                with $\tau$ from the noise; alarm on departures from neighbors.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Least squares on raw SCADA.** One dropout recorded as 0 moved the slope from 1.24 to
      0.05.
    - **A robust loss with a scale estimated on contaminated data.** The burst inflated Huber's
      MAD scale from 0.016 m (the calm week's) to 0.33 m.
    - **RANSAC with the default threshold.** scikit-learn uses the MAD of $y$, which here measured
      the demand cycle (0.44 m), not the noise.
    - **A single RANSAC seed.** Check the answer's spread over seeds.
    - **Refitting baselines on recent data** during a burst: at 65% burst readings RANSAC takes
      the burst line as normal.
    - **One pair of sensors as the only alarm.** October's p158 leak lowered both n229 and its
      neighbor, and the pair alarm stayed silent.
    - **Reading decimal commas as text.** BattLeDIM's files use `;` and `,`: pass `sep=";"` and
      `decimal=","`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §11.2:** why removing outliers by hand fails in high dimensions, and robust
      regression as the alternative.
    - **Forsyth §11.2.1:** M-estimators, the Huber loss, and iteratively reweighted least
      squares, with a worked regression of weight on height.
    - **Forsyth §11.2.2:** choosing the scale parameter, the step that failed on our burst
      window.
    - **Exercise.** Show that IRLS with weights $\min(1, \delta/|r|)$ decreases the Huber
      objective at every step (the weights define a quadratic upper bound), then rerun §4 with
      the scale fixed at the calm week's $\sigma$. Does Huber recover the calm slope?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Least squares lets one reading move the line; Huber bounds each reading's pull.
    - Huber is IRLS: least squares with weights $\min(1, \delta/|r|)$.
    - RANSAC keeps the largest set within $\tau$; set $\tau$ from the noise, not the default.
    - A coherent minority, such as a burst, defeats Huber; a majority defeats RANSAC.
    - Fit baselines on trusted periods; a pair alarm misses a leak both sensors feel.
    - An anomaly detector flags change; it cannot say whether the change is a leak.

    **Next:** calibration and quantile regression: the strength we can count on.
    """)
    return


if __name__ == "__main__":
    app.run()
