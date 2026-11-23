"""Applied Machine Learning for CEE — Lecture 10: Calibration and Quantile Regression.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l10_calibration_quantiles.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_concrete.sh`` once (UCI Concrete Compressive Strength, Yeh 1998).
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
    # Applied Machine Learning for CEE — Lecture 10: Calibration and Quantile Regression

    **Slides:** *Applied Machine Learning for CEE*, Lecture 10 (`lectures/cee_l10_calibration_quantiles`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §10.2 and §11.2 (regression losses); Koenker & Bassett (1978) for quantile regression; Gneiting & Raftery (2007) for proper scoring rules.
    **Data:** UCI Concrete Compressive Strength (Yeh, 1998; CC BY 4.0), with Lecture 2's encoder and Lecture 5's 30 splits that hold out whole designs.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. It runs in
    about half a minute (about 220 small linear programs).

    **Your tasks** (from the slides): derive why the pinball loss finds quantiles, write
    quantile regression as a linear program and check it against scikit-learn, measure
    calibration overall and by strength, and choose the cement.
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
    from scipy.optimize import linprog
    from scipy.stats import norm
    from sklearn.linear_model import QuantileRegressor
    from sklearn.model_selection import GroupShuffleSplit

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    TAU = 0.05  # the characteristic strength: 5% fractile
    TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    BANDS = {"below 30": (0.0, 30.0), "30 to 50": (30.0, 50.0), "above 50": (50.0, 200.0)}
    REPEATS, TEST_SIZE = 30, 0.25  # Lecture 5's splits
    FC = 35.0  # specified 28-day strength, MPa
    PROPOSED_ROW = 638  # Lecture 5's proposed mix P
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        BANDS,
        FC,
        GroupShuffleSplit,
        MARKERS,
        PALETTE,
        PROPOSED_ROW,
        Path,
        QuantileRegressor,
        REPEATS,
        SEED,
        TAU,
        TAUS,
        TEST_SIZE,
        linprog,
        norm,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 5–9

    On your own, no notes, two minutes:

    1. In Lecture 5 we aimed the mix at $f'_c$ plus 9.0 MPa. Where did 9.0 come from, and what
       did it buy?
    2. In 2024 the bridge model predicted that 5.0% of bridges would turn poor; 4.0% did. Which
       assumption of the cost threshold failed?
    3. On the burst window, why did Huber's scale stop protecting the fit?

    **Today:** what strength can a mix be counted on to reach?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. The 90th percentile of over-prediction on held-out 28-day tests: aiming at
                   $f'_c$ + 9.0 leaves about 10% of such mixes below $f'_c$ (Lecture 5).
                2. Calibration: $p > 1/(1+r)$ assumes that of bridges given $p$, a share $p$
                   turns poor (Lecture 6).
                3. The burst inflated the MAD scale from the calm week's 0.016 m to 0.33 m
                   (Lecture 9). Today the spread itself changes with strength.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A mean prediction is exceeded half the time (in Lecture 5, 57% of the 3,179 held-out 28-day
    predictions were above the measured strength; 58% of the 425 tests). Codes count on a **low fractile**: the characteristic strength is the 5%
    fractile (Eurocode 2, EN 1992-1-1), and resistance factors $\phi < 1$ reduce it further.
    **The decision:** how much cement makes 95% of held-out designs like P (Lecture 5's proposed
    mix) reach 35 MPa? For a new design, that 5% mostly reflects the model's error (held-out SD
    7.6 MPa), not batch-to-batch scatter (repeat tests, SD 1.4 MPa): it is 95% of designs like
    P, not 95% of batches of one design. Both SDs are computed below.
    """)
    return


@app.cell
def _(GroupShuffleSplit, Path, REPEATS, SEED, TEST_SIZE, np, pd):
    COLUMNS = [
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_agg",
        "fine_agg",
        "age",
        "strength",
    ]

    def find_concrete() -> Path:
        """Locate ``data/derived/Concrete_Data.csv`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_concrete.sh``).
        """
        rel = Path("data") / "derived" / "Concrete_Data.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_concrete.sh`")

    def load_concrete(tol: float = 1.0) -> pd.DataFrame:
        """The 1,030 tests with ``design`` ids: mixes within ``tol`` kg/m3 are one design."""
        frame = pd.read_csv(find_concrete())
        assert frame.shape == (1030, 9), frame.shape
        frame.columns = COLUMNS
        ingredients = COLUMNS[:7]
        frame["mix"] = frame.groupby(ingredients, sort=False).ngroup()
        mixes = frame.groupby("mix")[ingredients].first().to_numpy()
        parent = np.arange(len(mixes))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = int(parent[a])
            return a

        close = np.abs(mixes[:, None, :] - mixes[None, :, :]).max(axis=-1) <= tol
        for _a, _b in np.argwhere(np.triu(close, 1)):
            parent[find(int(_a))] = find(int(_b))
        _, design_of_mix = np.unique([find(_i) for _i in range(len(mixes))], return_inverse=True)
        frame["design"] = design_of_mix[frame["mix"].to_numpy()]
        return frame

    def mix_groups(frame: pd.DataFrame) -> np.ndarray:
        """Lecture 2's encoder: w/b, slag/b, fly ash/b, SP/b, aggregate/b, ln(age)."""
        binder = frame["cement"] + frame["slag"] + frame["fly_ash"]
        z = np.column_stack(
            [
                frame["water"] / binder,
                frame["slag"] / binder,
                frame["fly_ash"] / binder,
                frame["superplasticizer"] / binder,
                (frame["coarse_agg"] + frame["fine_agg"]) / binder,
                np.log(frame["age"]),
            ]
        )
        assert z.shape == (len(frame), 6), z.shape
        return z

    def fit_mean(A_: np.ndarray, y_: np.ndarray) -> tuple[np.ndarray, float]:
        """Lecture 5's mean model: least-squares weights and the SD of its training residuals."""
        w = np.linalg.lstsq(A_, y_, rcond=None)[0]
        return w, float(np.std(y_ - A_ @ w, ddof=A_.shape[1]))

    df = load_concrete()
    y = df["strength"].to_numpy()
    Z = mix_groups(df)
    A = np.column_stack([np.ones(len(Z)), Z])
    splits = list(
        GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SEED).split(
            df, groups=df["design"]
        )
    )
    assert df["design"].nunique() == 315 and len(splits) == 30
    return A, Z, df, fit_mean, mix_groups, splits, y


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: how much margin?

    The mean model (Lecture 5's least squares on the six groups) errs on held-out designs. The
    next cell tabulates its errors (measured − predicted) by predicted strength. **A mix is
    predicted at 60 MPa: what strength would you sign for, so that 95% of designs predicted there
    reach it? And at 25 MPa? Is one margin enough?** The 30 splits give 7,689 held-out
    predictions of the 1,030 tests.
    """)
    return


@app.cell
def _(A, BANDS, TAU, fit_mean, norm, np, pd, splits, y):
    # Each split's mean model, fitted once and reused below.
    mean_fits = [fit_mean(A[_tr], y[_tr]) for _tr, _ in splits]
    _rows = []
    for _k, ((_tr, _te), (_w, _sd)) in enumerate(zip(splits, mean_fits, strict=True)):
        _rows.append(
            pd.DataFrame({"split": _k, "row": _te, "y": y[_te], "mean": A[_te] @ _w, "sd": _sd})
        )
    held = pd.concat(_rows, ignore_index=True)
    held["mean − 1.645σ"] = held["mean"] + norm.ppf(TAU) * held["sd"]
    _err = held["y"] - held["mean"]
    _tab = {}
    for _name, (_lo, _hi) in BANDS.items():
        _m = (held["mean"] >= _lo) & (held["mean"] < _hi)
        _tab[_name] = (
            int(_m.sum()),
            round(float(_err[_m].std()), 1),
            round(float(np.quantile(_err[_m], TAU)), 1),
        )
    _tab["all"] = (len(held), round(float(_err.std()), 1), round(float(np.quantile(_err, TAU)), 1))
    assert _tab == {
        "below 30": (2537, 5.8, -10.3),
        "30 to 50": (3724, 7.1, -11.4),
        "above 50": (1357, 10.7, -15.6),
        "all": (7689, 7.6, -12.2),
    }, _tab
    pd.DataFrame(
        _tab, index=["held-out predictions", "error SD (MPa)", "5% of errors below (MPa)"]
    ).T
    return held, mean_fits


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with the **pinball loss** as this lecture's step: the mix $\mathbf{x}$ →
    encoder (dimensionless groups) → $\mathbf{z}$ → decoder, a line $\mathbf{w}_\tau^\top
    \mathbf{z}$ → the predicted quantile $\hat q_\tau$ → the pinball loss against the measured
    strength. Training is a linear program. §4 recomputes the slide's trace of mix P.

    ## 3. Derive

    **Assumption:** the $\tau$-quantile of strength is a weighted sum of the groups; the spread
    may change with the mix. The **pinball loss** of a residual $r = y - q$:

    $$\rho_\tau(r) = \begin{cases} \tau\, r & r \ge 0 \\ (\tau - 1)\, r & r < 0 \end{cases}.$$

    **Why it finds the quantile.** For a random strength $Y$ with distribution $F$,

    $$\mathbb{E}\,\rho_\tau(Y-q) = \tau\!\int_{y>q}(y-q)\,dF(y) + (1-\tau)\!\int_{y<q}(q-y)\,dF(y),$$

    and differentiating in $q$ gives $-\tau(1-F(q)) + (1-\tau)F(q) = F(q) - \tau$: zero where
    $F(q) = \tau$. The loss is minimized by the true quantile, so it is a **proper scoring rule**
    for quantiles (Gneiting & Raftery, 2007). At $\tau = 0.05$, a reading below $q$ costs 0.95
    per MPa and one above costs 0.05: Lecture 6's cost ratio in another form.

    **The mean model's alternative** assumes normal errors of one size:
    $q = \hat y - 1.645\,\sigma$.

    **A linear program** (Koenker & Bassett, 1978). Split each residual into
    $y_i - \mathbf{a}_i^\top\mathbf{w} = u_i^+ - u_i^-$ with $u^\pm \ge 0$:

    $$\min_{\mathbf{w},\,\mathbf{u}^\pm}\ \sum_i \big(\tau u_i^+ + (1-\tau) u_i^-\big)
      \quad \text{s.t.} \quad \mathbf{A}\mathbf{w} + \mathbf{u}^+ - \mathbf{u}^- = \mathbf{y}.$$

    At the optimum $u_i^+$ and $u_i^-$ are not both positive, so the objective is the pinball
    loss. A vertex solution passes **exactly** through $d+1$ tests (7 here).
    """)
    return


@app.cell
def _(linprog, np):
    def fit_quantile(Z_: np.ndarray, y_: np.ndarray, tau: float) -> np.ndarray:
        """Linear quantile regression as a linear program (pinball loss, no penalty).

        Args:
            Z_: Features, shape ``(n, d)``.
            y_: Targets, shape ``(n,)``.
            tau: Quantile level in (0, 1).

        Returns:
            Weights ``[w0, w]``.

        Raises:
            RuntimeError: If the solver fails.
        """
        A_ = np.column_stack([np.ones(len(Z_)), Z_])
        n, d = A_.shape
        # variables: w (free), u+ >= 0, u- >= 0 with A w + u+ - u- = y
        c = np.concatenate([np.zeros(d), np.full(n, tau), np.full(n, 1 - tau)])
        A_eq = np.hstack([A_, np.eye(n), -np.eye(n)])
        bounds = [(None, None)] * d + [(0, None)] * (2 * n)
        res = linprog(c, A_eq=A_eq, b_eq=y_, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        return res.x[:d]

    def pinball(y_: np.ndarray, q: np.ndarray, tau: float) -> float:
        """Mean pinball loss of predicted quantiles ``q``."""
        r = y_ - q
        return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))

    return fit_quantile, pinball


@app.cell
def _(TAU, np, pinball, splits, y):
    # The constant that minimizes the mean pinball loss is an order statistic.
    _tr = splits[0][0]
    _grid = np.linspace(y.min(), y.max(), 20001)
    _best = _grid[int(np.argmin([pinball(y[_tr], np.full(len(_tr), _g), TAU) for _g in _grid]))]
    _k = int(np.ceil(TAU * len(_tr)))
    assert (_k, round(float(_best), 2), round(float(np.sort(y[_tr])[_k - 1]), 2)) == (
        38,
        10.76,
        10.76,
    )
    print(
        f"the {_k}th smallest of {len(_tr)} strengths, {np.sort(y[_tr])[_k - 1]:.2f} MPa, "
        f"minimizes the mean pinball loss over constants"
    )
    return


@app.cell
def _(PROPOSED_ROW, TAU, Z, df, fit_quantile, mean_fits, norm, np, pinball, wq_splits, y):
    # Slides "Sign for the strength you count on", "The tilt moves the minimum", and "One
    # margin is flat; the quantile's tilts".
    _d28 = (df["age"] == 28).to_numpy()
    _wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()[_d28]
    _y28 = y[_d28]
    assert len(_y28) == 425, len(_y28)
    _band = fit_quantile(_wb[:, None], _y28, 0.95) - fit_quantile(_wb[:, None], _y28, 0.05)
    _lo, _hi = np.quantile(_wb, [0.05, 0.95])
    assert [round(float(_band[0] + _band[1] * _v), 1) for _v in (_lo, _hi)] == [34.0, 25.2]

    _q = np.linspace(0, 80, 801)
    _best = _q[int(np.argmin([pinball(_y28, np.full(len(_y28), _v), TAU) for _v in _q]))]
    assert round(float(_best), 1) == round(float(np.quantile(_y28, TAU)), 1) == 15.4

    # Split 0's models, as fitted in the held-out cells.
    (_w, _sd), _wq = mean_fits[0], wq_splits[0]
    _W, _L = np.meshgrid(np.linspace(0.30, 0.70, 41), np.linspace(np.log(7), np.log(365), 41))
    _G = np.repeat(Z[[PROPOSED_ROW]], _W.size, axis=0)
    _G[:, 0], _G[:, 5] = _W.ravel(), _L.ravel()
    _Ag = np.column_stack([np.ones(len(_G)), _G])
    _margin = _Ag @ _w - _Ag @ _wq
    assert round(float(-norm.ppf(TAU) * _sd), 1) == 12.1
    assert (round(float(_margin.min()), 1), round(float(_margin.max()), 1)) == (2.9, 20.7)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: is calibration enough?

    A forecaster quotes the same value for every mix: the 5% quantile of all training strengths
    (about 11.0 MPa). What share of held-out tests falls below it? Is it calibrated? Would you
    design with it? How would you score it against a model that changes with the mix?

    ## 4. Build and verify

    **Predict:** of the 746 training tests of the first split, how many fall below the fitted
    5% line?
    """)
    return


@app.cell
def _(QuantileRegressor, TAU, Z, fit_quantile, np, splits, y):
    _tr = splits[0][0]
    _w = fit_quantile(Z[_tr], y[_tr], TAU)
    _ref = QuantileRegressor(quantile=TAU, alpha=0, solver="highs").fit(Z[_tr], y[_tr])
    _w_ref = np.concatenate([[_ref.intercept_], _ref.coef_])
    _diff = float(np.max(np.abs(_w - _w_ref)))
    assert np.allclose(_w, _w_ref, rtol=0, atol=1e-8), _diff
    _r = y[_tr] - np.column_stack([np.ones(len(_tr)), Z[_tr]]) @ _w
    _below, _on = int(np.sum(_r < -1e-9)), int(np.sum(np.abs(_r) <= 1e-9))
    # Slide "Sharp and calibrated beats calibrated": 3e-13; 34 below, 7 on the line.
    assert len(_tr) == 746 and _diff < 1e-11 and (_below, _on) == (34, 7)
    assert _below + _on == 41  # "41 at or below": 37.3 = 0.05 x 746 lies between 34 and 41
    print(f"LP vs scikit-learn: {_diff:.0e}; {_below} tests below the line, {_on} on it")
    return


@app.cell
def _(A, PROPOSED_ROW, TAU, Z, df, fit_mean, fit_quantile, norm, splits, y):
    # The slide's trace: mix P, held out in the first split.
    _tr, _te = splits[0]
    assert PROPOSED_ROW in _te
    _w, _sd = fit_mean(A[_tr], y[_tr])
    _a = A[PROPOSED_ROW]
    _trace = tuple(
        round(float(_v), 2)
        for _v in (
            _a @ _w,
            _sd,
            _a @ _w + norm.ppf(TAU) * _sd,
            _a @ fit_quantile(Z[_tr], y[_tr], TAU),
            y[PROPOSED_ROW],
        )
    )
    # The diagram's inset: 39.77 - 1.645 x 7.34 = 27.70; the 5% line, 28.88; measured, 38.21.
    assert _trace == (39.77, 7.34, 27.7, 28.88, 38.21), _trace
    assert round(39.77 - 1.645 * 7.34, 2) == 27.70
    assert df.loc[PROPOSED_ROW, ["cement", "water", "age"]].tolist() == [375.0, 186.0, 28.0]
    print("mix P: mean, sd, mean - 1.645 sd, quantile line, measured =", _trace)
    return


@app.cell
def _(A, TAU, Z, fit_quantile, held, np, splits, y):
    # Each split's 5% quantile regression, fitted once and reused below.
    wq_splits = [fit_quantile(Z[_tr], y[_tr], TAU) for _tr, _ in splits]
    _qr, _const = [], []
    for (_tr, _te), _wq in zip(splits, wq_splits, strict=True):
        _qr.append(A[_te] @ _wq)
        _const.append(np.full(len(_te), np.quantile(y[_tr], TAU)))
    held_q = held.assign(
        **{"quantile regression": np.concatenate(_qr), "no model": np.concatenate(_const)}
    )
    assert round(float(held_q.groupby("split")["no model"].first().mean()), 1) == 11.0
    # Slides "No model is calibrated too" and "Use quantiles inside the data": a straight line
    # extrapolates, and 139 held-out 5% values (1.8%) are negative strengths, down to -14.4 MPa.
    _q = held_q["quantile regression"].to_numpy()
    assert (round(float(_q.min()), 1), round(float(np.quantile(_q, 0.01)), 1)) == (-14.4, -3.3)
    assert (int((_q < 0).sum()), round(100 * float((_q < 0).mean()), 1)) == (139, 1.8)
    return held_q, wq_splits


@app.cell
def _(TAU, df, held_q, norm, pd, pinball):
    _methods = ["no model", "mean − 1.645σ", "quantile regression"]
    # Pooled over all 7,689 held-out predictions, like the coverage shares.
    _pin = {
        _m: round(pinball(held_q["y"].to_numpy(), held_q[_m].to_numpy(), TAU), 2) for _m in _methods
    }
    # Repeat tests: one mix at one age tested more than once, with different strengths
    # (identical strengths are one record entered twice). Lecture 5's noise floor.
    _rep = df.groupby(["mix", "age"])["strength"].agg(["size", "nunique", "std"])
    _repeat_sd = float(_rep[(_rep["size"] > 1) & (_rep["nunique"] > 1)]["std"].median())
    _floor = round(float(_repeat_sd * norm.pdf(norm.ppf(TAU))), 2)
    # Bracket: 1.40, 0.77, 0.75; floor 0.14 (the expected pinball loss of repeat-test noise).
    assert _pin == {"no model": 1.4, "mean − 1.645σ": 0.77, "quantile regression": 0.75}
    assert round(_repeat_sd, 1) == 1.4 and _floor == 0.14, (_repeat_sd, _floor)
    pd.Series({**_pin, "repeat tests (rough)": _floor}, name="pinball loss at τ = 0.05").to_frame()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **Reliability.** For $\tau$ from 5% to 95%, what share of held-out tests falls below each
       predicted quantile, for both methods?
    2. **By strength.** Below 30 MPa and above 50 MPa: what share falls below mean − 1.645σ?
    3. **No model.** Same question for the constant.
    4. **Crossing.** Fit the 5% and the 10% lines separately. Can the 5% line lie above the 10%?
    """)
    return


@app.cell
def _(
    A,
    MARKERS,
    PALETTE,
    TAU,
    TAUS,
    Z,
    fit_quantile,
    mean_fits,
    norm,
    np,
    plt,
    splits,
    wq_splits,
    y,
):
    _below = {"mean ± zσ": {_t: [] for _t in TAUS}, "quantile regression": {_t: [] for _t in TAUS}}
    for (_tr, _te), (_w, _sd), _wq05 in zip(splits, mean_fits, wq_splits, strict=True):
        for _t in TAUS:
            _wq = _wq05 if _t == TAU else fit_quantile(Z[_tr], y[_tr], _t)
            _below["mean ± zσ"][_t].append(y[_te] < A[_te] @ _w + norm.ppf(_t) * _sd)
            _below["quantile regression"][_t].append(y[_te] < A[_te] @ _wq)
    # Shares pooled over all held-out predictions, as in the coverage cell.
    _share = {
        _k: [100 * float(np.concatenate(_d[_t]).mean()) for _t in TAUS] for _k, _d in _below.items()
    }
    _res = {_k: [round(_v, 1) for _v in _s] for _k, _s in _share.items()}
    _gap = max(
        abs(_v - 100 * _t) for _s in _share.values() for _v, _t in zip(_s, TAUS, strict=True)
    )
    # Slide "Both look calibrated overall": within 2.3 points; 5.0 and 5.5 at 5%.
    assert _res["mean ± zσ"] == [5.0, 9.8, 22.7, 51.3, 75.7, 88.8, 93.8], _res
    assert _res["quantile regression"] == [5.5, 10.3, 24.9, 48.9, 74.0, 88.6, 93.4], _res
    assert round(_gap, 1) == 2.3, _gap
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _i, (_k, _v) in enumerate(_share.items()):
        _ax.plot(
            [100 * _t for _t in TAUS], _v, marker=MARKERS[_i + 1], color=PALETTE[_i + 1], label=_k
        )
    _ax.plot([0, 100], [0, 100], ":", color="grey")
    _ax.set_xlabel("nominal level (%)")
    _ax.set_ylabel("held-out tests below (%)")
    _ax.legend()
    _fig
    return


@app.cell
def _(BANDS, MARKERS, PALETTE, held_q, plt):
    _methods = ["no model", "mean − 1.645σ", "quantile regression"]
    _cov = {}
    for _m in _methods:
        _below = held_q["y"] < held_q[_m]
        _cov[_m] = {"all": round(100 * float(_below.mean()), 1)}
        for _name, (_lo, _hi) in BANDS.items():
            _sel = (held_q["mean"] >= _lo) & (held_q["mean"] < _hi)
            _cov[_m][_name] = round(100 * float(_below[_sel].mean()), 1)
    # Slides "Calibrated overall, unsafe when strong" and "No model is calibrated too".
    assert _cov["mean − 1.645σ"] == {"all": 5.0, "below 30": 2.9, "30 to 50": 3.4, "above 50": 13.5}
    assert _cov["quantile regression"] == {
        "all": 5.5,
        "below 30": 7.6,
        "30 to 50": 3.9,
        "above 50": 6.4,
    }
    assert _cov["no model"] == {"all": 5.2, "below 30": 12.9, "30 to 50": 0.0, "above 50": 0.0}
    _fig, _ax = plt.subplots(figsize=(5.5, 3.6))
    for _i, _m in enumerate(_methods):
        _ax.plot(
            range(3),
            [_cov[_m][_b] for _b in BANDS],
            marker=MARKERS[_i],
            color=PALETTE[_i],
            label=_m,
        )
    _ax.axhline(5, color="grey", ls=":")
    _ax.set_xticks(range(3), [f"{_b} MPa" for _b in BANDS])
    _ax.set_ylabel("held-out tests below the 5% quantile (%)")
    _ax.legend(fontsize=9)
    print(_cov)
    _fig
    return


@app.cell
def _(BANDS, SEED, TAU, df, held_q, np, pd):
    # 95% intervals that resample whole designs: every held-out prediction of a design (its
    # tests, in every split that held it out) moves together. 2,000 resamples, seed 0,
    # percentile intervals; designs numbered by first appearance, as the figure script does.
    _g = pd.factorize(df["design"].to_numpy()[held_q["row"].to_numpy()])[0]
    _G = int(_g.max()) + 1
    _W = np.random.default_rng(SEED).multinomial(_G, np.full(_G, 1 / _G), size=2000)
    _W = _W.astype(float)

    def _boot(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The pooled mean of ``values`` over ``mask``, in every resample."""
        _num = np.bincount(_g[mask], weights=values[mask], minlength=_G)
        _n = _W @ np.bincount(_g[mask], minlength=_G).astype(float)
        assert np.all(_n > 0), "a resample with no prediction in the band"
        return (_W @ _num) / _n

    def _ci(v: np.ndarray, scale: float, nd: int) -> list[float]:
        """The 2.5% and 97.5% points, rounded once."""
        return [round(float(scale * _q), nd) + 0.0 for _q in np.quantile(v, [0.025, 0.975])]

    _y = held_q["y"].to_numpy()
    _ci_band = {
        (_m, _name): _ci(
            _boot(
                (_y < held_q[_m].to_numpy()).astype(float),
                ((held_q["mean"] >= _lo) & (held_q["mean"] < _hi)).to_numpy(),
            ),
            100,
            1,
        )
        for _m in ("mean − 1.645σ", "quantile regression")
        for _name, (_lo, _hi) in BANDS.items()
    }
    _ci_strong = {_m: _ci_band[(_m, "above 50")] for _m in ("mean − 1.645σ", "quantile regression")}
    # Quantile regression below 30 MPa, 7.6% (4.7 to 10.7): not separated from 5%. In the
    # 30-50 band, where mix P's family lands at 468 kg, both cover more than 95%: 3.4% (2.0 to
    # 5.0) and 3.9% (2.2 to 5.7) below.
    assert _ci_band[("quantile regression", "below 30")] == [4.7, 10.7]
    assert _ci_band[("mean − 1.645σ", "30 to 50")] == [2.0, 5.0]
    assert _ci_band[("quantile regression", "30 to 50")] == [2.2, 5.7]
    _rn = _y - held_q["mean − 1.645σ"].to_numpy()
    _rq = _y - held_q["quantile regression"].to_numpy()
    _gap = np.maximum(TAU * _rn, (TAU - 1) * _rn) - np.maximum(TAU * _rq, (TAU - 1) * _rq)
    _ci_gap = _ci(_boot(_gap, np.ones(len(_y), dtype=bool)), 1, 3)
    # Slide "Calibrated overall, unsafe when strong": 13.5% (6.8% to 21.5%) and 6.4% (2.7% to
    # 11.1%); slide "Sharp and calibrated beats calibrated": the pinball gap 0.023, 95% interval
    # -0.019 to 0.066, which includes zero.
    assert _G == 315
    assert _ci_strong == {"mean − 1.645σ": [6.8, 21.5], "quantile regression": [2.7, 11.1]}, (
        _ci_strong
    )
    assert round(float(_gap.mean()), 3) == 0.023 and _ci_gap == [-0.019, 0.066], _ci_gap
    # Slide "No model is calibrated too": the constant loses 0.63 (0.53 to 0.73) more than
    # mean - 1.645 sigma and 0.65 (0.54 to 0.75) more than quantile regression, 1.82 and 1.88
    # times their losses: "nearly twice", and separated from both.
    _rc = _y - held_q["no model"].to_numpy()
    _lc = np.maximum(TAU * _rc, (TAU - 1) * _rc)
    _vs_const = {}
    for _name, _r in (("mean − 1.645σ", _rn), ("quantile regression", _rq)):
        _lm = np.maximum(TAU * _r, (TAU - 1) * _r)
        _vs_const[_name] = (
            round(float((_lc - _lm).mean()), 2),
            _ci(_boot(_lc - _lm, np.ones(len(_y), dtype=bool)), 1, 2),
            round(float(_lc.mean() / _lm.mean()), 2),
        )
    assert _vs_const == {
        "mean − 1.645σ": (0.63, [0.53, 0.73], 1.82),
        "quantile regression": (0.65, [0.54, 0.75], 1.88),
    }, _vs_const
    print(
        f"strong mixes below the 5% value, 95% intervals: {_ci_strong}; "
        f"pinball gap {_gap.mean():.3f} ({_ci_gap[0]} to {_ci_gap[1]}); "
        f"the constant's extra loss, interval and ratio: {_vs_const}"
    )
    return


@app.cell
def _(A, Z, fit_quantile, np, y):
    _q05 = A @ fit_quantile(Z, y, 0.05)
    _q10 = A @ fit_quantile(Z, y, 0.10)
    _q50 = A @ fit_quantile(Z, y, 0.50)
    _cross = _q05 > _q10
    # "Near the edge of the data": 31 of the 32 have a predicted median below 30 MPa (34% of
    # all tests do), and 16 are among the tenth of tests with the highest leverage.
    _lev = np.einsum("ij,jk,ik->i", A, np.linalg.pinv(A.T @ A), A)
    _high = _lev > np.quantile(_lev, 0.9)
    assert int((_cross & (_q50 < 30)).sum()) == 31 and round(100 * float((_q50 < 30).mean())) == 34
    assert int((_cross & _high).sum()) == 16
    # Slide "Separate quantiles can cross": 32 of 1,030 tests (3.1%), up to 1.5 MPa.
    assert (
        int(_cross.sum()),
        round(100 * float(_cross.mean()), 1),
        round(float((_q05 - _q10).max()), 1),
    ) == (32, 3.1, 1.5)
    _sorted = np.sort(np.column_stack([_q05, _q10]), axis=1)
    assert np.all(_sorted[:, 0] <= _sorted[:, 1])
    print(f"5% line above 10% line for {_cross.sum()} tests; sorting each test's pair removes it")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Calibration overall is not calibration where it matters.** The error SD grows from 5.8 to
    10.7 MPa across the bands, so one margin covers 97% of weak mixes and 86.5% of strong ones.
    The constant forecaster is calibrated too, and useless: only a proper score, here the pinball
    loss, rewards calibration and sharpness together (Gneiting, Balabdaoui & Raftery, 2007).
    Separate quantile fits can cross; sorting each test's predicted quantiles (rearrangement;
    Chernozhukov, Fernández-Val & Galichon, 2010) removes the contradiction.

    ## 6. Decide

    How much cement does the family of mix P (water 186 kg/m³, no admixtures) need for 95% of
    held-out designs like P to reach 35 MPa?
    """)
    return


@app.cell
def _(
    BANDS,
    FC,
    MARKERS,
    PALETTE,
    PROPOSED_ROW,
    TAU,
    df,
    mean_fits,
    mix_groups,
    norm,
    np,
    pd,
    plt,
    wq_splits,
):
    _cem = np.linspace(250, 540, 2901)
    _fam = pd.concat([df.loc[[PROPOSED_ROW]]] * len(_cem), ignore_index=True)
    _fam["cement"] = _cem
    _Af = np.column_stack([np.ones(len(_cem)), mix_groups(_fam)])

    def _cement_for_fc(curve: np.ndarray) -> float:
        """Cement where a rising curve reaches f'c; refuses to clamp at the grid's ends."""
        assert np.all(np.diff(curve) > 0), "the curve should rise with cement"
        assert curve[0] <= FC <= curve[-1], "f'c lies outside the cement grid"
        return float(np.interp(FC, curve, _cem))

    def _curves_of(fit: tuple[np.ndarray, float], wq: np.ndarray) -> dict[str, np.ndarray]:
        """The mean and the two 5% curves of one split's models over the cement grid."""
        return {
            "mean": _Af @ fit[0],
            "mean − 1.645σ": _Af @ fit[0] + norm.ppf(TAU) * fit[1],
            "5% quantile line": _Af @ wq,
        }

    _all = [
        {_k: _cement_for_fc(_v) for _k, _v in _curves_of(_f, _wq).items()}
        for _f, _wq in zip(mean_fits, wq_splits, strict=True)
    ]
    _curves = _curves_of(mean_fits[0], wq_splits[0])  # split 0 holds out P: the slide's fits
    _c = {_k: int(round(_v)) for _k, _v in _all[0].items()}
    # Slide "Count on the 5% line": 337, 454, 468 kg/m3; w/c 0.55, 0.41, 0.40.
    assert _c == {"mean": 337, "mean − 1.645σ": 454, "5% quantile line": 468}, _c
    assert [round(186.0 / _v, 2) for _v in _c.values()] == [0.55, 0.41, 0.4]
    assert df["cement"].max() == 540.0  # 468 kg lies inside the tested range
    # The two 5% curves cross near 420 kg; at 468 kg the quantile line's margin is 13.1 MPa,
    # against the constant 12.1, and the predicted mean, 48.1 MPa, lies in the 30-50 band.
    _gap = _curves["5% quantile line"] - _curves["mean − 1.645σ"]
    assert round(float(_cem[np.flatnonzero(np.diff(np.sign(_gap)) != 0)[0]])) == 419
    _at = _all[0]["5% quantile line"]
    _mean_at = float(np.interp(_at, _cem, _curves["mean"]))
    _margin_at = float(np.interp(_at, _cem, _curves["mean"] - _curves["5% quantile line"]))
    assert (round(_margin_at, 1), round(_mean_at, 1)) == (13.1, 48.1)
    assert BANDS["30 to 50"][0] <= _mean_at < BANDS["30 to 50"][1]
    # The same answer on all 30 splits: 468 is one split's fit.
    _t = pd.DataFrame(_all)
    _extra = _t["5% quantile line"] - _t["mean − 1.645σ"]
    _summary = {_k: (round(_t[_k].median()), round(_t[_k].min()), round(_t[_k].max())) for _k in _t}
    assert _summary == {
        "mean": (338, 334, 344),
        "mean − 1.645σ": (456, 444, 468),
        "5% quantile line": (468, 441, 491),
    }, _summary
    assert (round(_extra.median()), round(_extra.min()), round(_extra.max())) == (13, -12, 31)
    assert int((_extra > 0).sum()) == 26
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _i, (_k, _v) in enumerate(_curves.items()):
        _ax.plot(_cem, _v, color=PALETTE[_i], label=_k)
        _ax.plot(_c[_k], FC, MARKERS[_i], color=PALETTE[_i], ms=9)
    _ax.axhline(FC, color="black", lw=1, ls=":")
    _ax.set_xlabel("cement (kg/m³)")
    _ax.set_ylabel("28-day strength (MPa)")
    _ax.legend(fontsize=9)
    print("cement for 35 MPa, split 0:", _c)
    print("median (min, max) over 30 splits:", _summary)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the 468 kg rests on.** It is one split's fit, the split that holds out P; over the 30
    splits the quantile line needs 441 to 491 kg (median 468) and the normal margin 444 to 468
    (median 456). At 468 kg the predicted mean is 48.1 MPa, in the 30 to 50 band, where both
    margins already cover more than 95% (3.4% and 3.9% below) and the error SD, 7.1 MPa, is
    below the overall 7.6. The extra cement comes from the quantile line's tilt (a 13.1 MPa
    margin there, against the constant 12.1), not from this band's spread: a median 13 kg more
    than the normal margin, −12 to +31 across splits, more in 26 of 30.

    **When not to use it.** Not in the far tails (about 37 tests per split define the 5% line;
    a 1% line would rest on about 8); not beyond the tested ratios or for new materials (a
    straight line extrapolates: 139 held-out 5% values, 1.8%, are negative strengths, down to
    −14.4 MPa); not with one overall check; not instead of acceptance testing.

    **Impact.** A 5% fractile accepts that one batch in 20 falls short of what is counted on;
    resistance and load factors cover the rest. A model calibrated only on average moves that
    risk onto strong mixes, used where loads are high; more cement for more assurance has a
    carbon cost.

    ### Vote: which 5% value do you sign?

    A strong mix, predicted mean 60 MPa; specified $f'_c$ = 45 MPa. **A.** Mean − 1.645σ, 47.9
    MPa. **B.** Quantile regression's 5% value. **C.** The mean. **D.** Trial batches first; the
    model only plans.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                A covers only 86.5% of strong mixes (13.5% below); C is the mean; B follows the
                spread (6.4% below in this band); D is always right in practice. B and D
                together. What would change the answer: a producer's own records with less
                scatter, or a mix outside the tested ratios.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where quantile losses live today

    - **Forecasting:** the Temporal Fusion Transformer trains on the pinball loss at the 10th,
      50th, and 90th percentiles (Lim et al., 2021); Lecture 20.
    - **Weather:** GenCast is scored with the CRPS (Price et al., 2025), which equals twice the
      integral of the pinball loss over all levels.
    - **Guarantees:** conformalized quantile regression adds finite-sample coverage (Romano,
      Patterson & Candès, 2019); Lecture 11.
    - **Language models** can be reasonably calibrated about their own answers (Kadavath et al.,
      2022); Lecture 19.

    ## 7. Check yourself

    **Exit question.** A producer's quantile model is calibrated over all its mixes: 5.0% fall
    below the 5% line. Why is that not enough to sign a high-strength mix? What would you ask
    for?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Overall calibration can hide miscalibration by subgroup (13.5% below for strong
                mixes with a constant margin) and is achieved even by a constant. Ask for
                coverage by strength band and a proper score (pinball loss) on held-out designs,
                and for the share below the 5% line among mixes like this one.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **A margin of $1.645\sigma$ with one $\sigma$** when the spread changes with the mean.
    - **Checking calibration only overall.** Split by the quantity that sets the risk.
    - **Scoring quantiles by coverage alone.** A constant reaches 5%; use the pinball loss.
    - **Penalized quantile regression by default.** scikit-learn's `QuantileRegressor` has
      `alpha=1.0` by default, an $L_1$ penalty; pass `alpha=0` for the plain LP.
    - **Separate fits at several levels** without sorting: quantiles can cross.
    - **Quantiles in the far tails** from a few hundred tests.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Koenker & Bassett (1978):** regression quantiles, their LP form, and why they are
      robust to outliers in $y$ (the median is the case $\tau = 0.5$).
    - **Gneiting & Raftery (2007):** proper scoring rules (pinball, CRPS, log, Brier) and why
      they reward honest forecasts.
    - **Forsyth §11.2:** robust losses; the absolute loss is the pinball loss at $\tau = 0.5$,
      times two.
    - **Exercise.** Show that the expected pinball loss of a normal forecast at its true
      $\tau$-quantile is $\sigma\,\varphi(\Phi^{-1}(\tau))$, the floor used in §4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Count on a quantile, not the mean: the pinball loss finds it; the LP fits it.
    - Calibrated overall is not calibrated where it matters: check by subgroup.
    - A constant can be calibrated: compare forecasts with a proper score.
    - A constant margin fails when the spread changes: 13.5% below for strong mixes.
    - Separate quantile fits can cross; sort them or fit them jointly.

    **Next:** conformal prediction and shift, at Illinois stream gauges.
    """)
    return


if __name__ == "__main__":
    app.run()
