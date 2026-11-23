"""Applied Machine Learning for CEE — Lecture 6: Linear Classifiers and Costs.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l06_linear_classifiers.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_nbi.sh`` once (FHWA National Bridge Inventory, Illinois).
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
    # Applied Machine Learning for CEE — Lecture 6: Linear Classifiers and Costs

    **Slides:** *Applied Machine Learning for CEE*, Lecture 6 (`lectures/cee_l06_linear_classifiers`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §2.1 (the support vector machine and the hinge loss), §11.3.1–11.3.2 (logistic and multiclass logistic regression).
    **Data:** FHWA National Bridge Inventory (NBI), Illinois, 2014–2024 (every two years); fetch it with `bash scripts/fetch_nbi.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    The slides' results are recomputed here and checked with an `assert`; a few readings of
    the figures in the slides' descriptions (the sheet, the cost curve, the dial, Newton's
    steps) are only in the figure script's `numbers.json`. The whole notebook takes about a
    minute to run on a CPU.

    **Your tasks** (from the slides): write Newton's method for logistic regression and check
    it against scikit-learn, derive the cost threshold and apply it, fit the SVM and the
    ordinal model, and check the probabilities on a later period.
    """)
    return


@app.cell
def _():
    import os
    import random
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.optimize import minimize
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)  # child processes only: this kernel's is fixed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    def fit_unpenalized(X: np.ndarray, y: np.ndarray, **kw: object) -> LogisticRegression:
        """scikit-learn's unpenalized logistic regression (C=np.inf), Newton-Cholesky by default.

        scikit-learn 1.8 deprecates penalty=None in favor of C=np.inf, then warns that C=np.inf
        sets penalty=None. The fit is unpenalized either way, so that one message is silenced,
        for this fit only.
        """
        kw = {"solver": "newton-cholesky"} | kw
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Setting penalty=None will ignore", category=UserWarning
            )
            return LogisticRegression(C=np.inf, **kw).fit(X, y)

    POOR = 4  # a lowest rating of 4 or less is poor
    HORIZON = 4  # years from the decision to the judging inventory
    START = {"train": 2014, "val": 2016, "test": 2020}
    R = 20  # a miss costs R false alarms (working value)
    RATIOS = (10, 20, 50)
    FLAG_SHARE = 0.10  # a fixed budget: flag this share of bridges
    OUTCOMES = np.array([POOR, 5, 6, 7, 8])  # ordinal cuts: P(later rating <= j)
    TWIN_KM = 0.2  # parallel structures closer than this are twins
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        FLAG_SHARE,
        HORIZON,
        LinearSVC,
        LogisticRegression,
        MARKERS,
        NearestNeighbors,
        OUTCOMES,
        PALETTE,
        POOR,
        Path,
        R,
        RATIOS,
        SEED,
        START,
        StandardScaler,
        TWIN_KM,
        expit,
        fit_unpenalized,
        log_loss,
        logit,
        minimize,
        np,
        pd,
        plt,
        roc_auc_score,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 3–5

    On your own, no notes, two minutes:

    1. You will forecast the next inspection of the same bridges. Which split do you validate
       with, and which columns do you drop?
    2. Why solve least squares with QR or the SVD, and never with $(X^{\top}X)^{-1}$?
    3. Why divide each building's daily profile by its mean before clustering? What do you
       lose?

    **Today:** which bridges that are not poor today will be poor in four years?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. Train on earlier inventories and validate on a later one, at the horizon of
                   the use; keep each inspection once; drop the columns the target sets, such
                   as the inspection interval of the record being predicted (Lecture 3).
                   Today's split is exactly that.
                2. Forming $X^{\top}X$ squares the condition number, so the inverse loses twice
                   as many digits; QR and the SVD work on $X$ itself (Lecture 5). Today each
                   Newton step is a weighted least squares, solved the same way.
                3. To compare schedules, not sizes; you lose how much is at stake, so keep the
                   mean as a number of its own (Lecture 4).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Inspectors rate each bridge's deck, superstructure, and substructure from 0 to 9; the
    lowest at 4 or below is **poor** (Lecture 3). **The decision:** which bridges rated 5 or
    more get preventive work and a closer look now, before they turn poor. A **false alarm**
    is work on a bridge that would have held; a **miss** is a bridge that turns poor: repair,
    load posting, detours. Today a miss costs $r = 20$ false alarms; the owner sets $r$ (§6).

    **The setting.** An engineer in 2021 has the inventories through 2020, so the last
    four-year outcome she can see is 2016 → 2020. She trains on the 2014 → 2018 transitions,
    validates on 2016 → 2020, and flags the 2020 bridges; the 2024 inventory is the test,
    looked at once. The label is the lowest rating in the inventory four years later, so a
    bridge poor in 2022 and repaired by 2024 counts as not poor. About 1% of bridges leave the
    inventory in four years (replaced or removed) and have no label; §6 shows they are the
    riskier ones.

    **The inputs** (16): the 12 inventory inputs of Lecture 3 (age, years since work, traffic,
    trucks, length, width, material, location) and the deck, superstructure, and substructure
    ratings and the lowest of them, all known when the decision is made. In Lecture 3 the
    ratings were the target; today they come from the earlier record, so they are not leakage.
    """)
    return


@app.cell
def _(HORIZON, POOR, Path, np, pd):
    FEATURES = [
        "age",
        "years_since_work",
        "log_adt",
        "pct_truck",
        "log_length",
        "deck_width_m",
        "concrete",
        "steel",
        "prestressed",
        "timber",
        "lat",
        "lon",
    ]
    COMPONENTS = ["deck", "superstructure", "substructure"]
    INPUTS = FEATURES + COMPONENTS + ["rating"]  # 12 inventory inputs + 4 ratings
    MATERIAL = {
        "1": "concrete",
        "2": "concrete",
        "3": "steel",
        "4": "steel",
        "5": "prestressed",
        "6": "prestressed",
        "7": "timber",
    }

    def find_nbi_csv() -> Path:
        """Locate ``data/derived/nbi_il_bridges.csv`` from the repo root, notebooks/, or ipynb/.

        Returns:
            Path of the CSV.

        Raises:
            FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_nbi.sh``).
        """
        rel = Path("data") / "derived" / "nbi_il_bridges.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_nbi.sh`")

    def load_bridges(path: Path) -> pd.DataFrame:
        """Load the bridge-year records and add the inputs (the Lecture 3 loader).

        Args:
            path: The CSV written by ``scripts/fetch_nbi.sh``.

        Returns:
            Records with complete inventory inputs, ``material``, and a planar grid in km.
        """
        text = ["structure", "county", "district", "owner", "kind", "design", "parallel"]
        frame = pd.read_csv(path, dtype=dict.fromkeys(text, str))
        frame["age"] = (frame["inspection_year"] - frame["year_built"]).clip(lower=0)
        _last = np.fmax(frame["year_built"], frame["year_reconstructed"].fillna(0))
        frame["years_since_work"] = (frame["inspection_year"] - _last).clip(lower=0)
        frame["log_adt"] = np.log1p(frame["adt"])
        frame["log_length"] = np.log(frame["length_m"].clip(lower=1.0))
        frame["material"] = frame["kind"].map(MATERIAL).fillna("other")
        for _m in ("concrete", "steel", "prestressed", "timber"):
            frame[_m] = (frame["material"] == _m).astype(float)
        frame = frame.dropna(subset=FEATURES).reset_index(drop=True)
        frame["x_km"] = (frame["lon"] + 89.0) * 111.32 * np.cos(np.deg2rad(40.0))
        frame["y_km"] = (frame["lat"] - 40.0) * 110.57
        return frame

    def transitions(frame: pd.DataFrame, start: int) -> pd.DataFrame:
        """Bridges not poor in ``start`` and their lowest rating ``HORIZON`` years later.

        Args:
            frame: Bridge-year records from :func:`load_bridges`.
            start: The NBI year of the decision.

        Returns:
            One row per bridge rated above ``POOR`` in ``start`` that is still in the inventory
            ``HORIZON`` years later, with ``rating_next``, ``poor_next`` (0/1), and
            ``deck_missing``. A missing deck rating (no deck) gets the lower of the other two.
        """
        key = ["structure", "nbi_year"]
        later = frame[[*key, "rating"]].rename(columns={"rating": "rating_next"})
        later["nbi_year"] -= HORIZON
        out = frame[frame["nbi_year"] == start].merge(
            later, on=key, how="inner", validate="one_to_one"
        )
        out = out[out["rating"] > POOR].reset_index(drop=True)
        out["deck_missing"] = out["deck"].isna()
        out["deck"] = out["deck"].fillna(out[["superstructure", "substructure"]].min(axis=1))
        out["poor_next"] = (out["rating_next"] <= POOR).astype(int)
        assert out[INPUTS].notna().all().all(), "missing inputs"
        return out

    return INPUTS, find_nbi_csv, load_bridges, transitions


@app.cell
def _(POOR, START, find_nbi_csv, load_bridges, pd, transitions):
    df = load_bridges(find_nbi_csv())
    sets = {_name: transitions(df, _start) for _name, _start in START.items()}
    tr, va, te = sets["train"], sets["val"], sets["test"]
    _rows = []
    for _name, _d in sets.items():
        _now = df[(df["nbi_year"] == START[_name]) & (df["rating"] > POOR)]
        _rows.append(
            {
                "period": f"{START[_name]} → {START[_name] + 4}",
                "bridges": len(_d),
                "turned poor": int(_d["poor_next"].sum()),
                "poor %": round(100 * _d["poor_next"].mean(), 1),
                "left the inventory %": round(100 * (1 - len(_d) / len(_now)), 1),
                "no deck rating": int(_d["deck_missing"].sum()),
            }
        )
    _summary = pd.DataFrame(_rows).set_index("period")
    # Question slide: 19,748 bridges, 841 turned poor (train); 19,637 and 793 (test); about 1%
    # leave the inventory and about 1% have no deck rating (notes).
    assert _summary["bridges"].tolist() == [19748, 19777, 19637]
    assert _summary["turned poor"].tolist() == [841, 898, 793]
    assert _summary["poor %"].tolist() == [4.3, 4.5, 4.0]
    assert _summary["left the inventory %"].tolist() == [1.1, 0.9, 0.8]
    assert _summary["no deck rating"].tolist() == [195, 205, 227]
    _summary
    return df, te, tr, va


@app.cell
def _(INPUTS, StandardScaler, np, pd, te, tr, va):
    scaler = StandardScaler().fit(tr[INPUTS].to_numpy(np.float64))  # training bridges only

    def encode(records: pd.DataFrame) -> np.ndarray:
        """The 16 inputs, standardized with the training scaler.

        Args:
            records: Transitions with every column of ``INPUTS``.

        Returns:
            The standardized inputs, shape ``(len(records), len(INPUTS))``.
        """
        _Z = scaler.transform(records[INPUTS].to_numpy(np.float64))
        assert _Z.shape == (len(records), len(INPUTS)), _Z.shape
        return _Z

    Z, Zv, Zt = encode(tr), encode(va), encode(te)
    y, yv, yt = (_d["poor_next"].to_numpy() for _d in (tr, va, te))
    return Z, Zt, Zv, encode, y, yt, yv


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: which bridges do you flag?

    Six city bridges in Champaign–Urbana, 2020. **The budget allows two flags. Which two?**
    Then: if a miss costs 20 false alarms, how many would you flag, and which? Write a chance
    for each. The six are the busiest bridges in Champaign County owned by a city (NBI Item 22
    = 04) whose lowest rating in 2020 was 5 or 6; all are prestressed concrete.
    """)
    return


@app.cell
def _(pd, te):
    _names = {
        "010314700000000": "Duncan Rd over Phinney Branch",
        "010601000000000": "Duncan Rd over Copper Slough",
        "010435100000000": "FAU 7154 over Copper Slough",
        "010448400000000": "Staley Rd over Phinney Branch",
        "010620200000000": "Mattis Ave over Phinney Branch",
        "010612800000000": "Broadway over Saline Branch",
    }
    _pick = te[(te["county"] == "019") & (te["owner"] == "04") & (te["rating"] <= 6)]
    _pick = _pick.sort_values(["adt", "structure"], ascending=[False, True], kind="stable")
    six = _pick.head(6).assign(id=list("ABCDEF"))
    assert six["structure"].tolist() == list(_names), six["structure"].tolist()
    assert (six["material"] == "prestressed").all()
    pd.DataFrame(
        {
            "id": six["id"],
            "road, over": six["structure"].map(_names),
            "built": six["year_built"].astype(int),
            "rebuilt": six["year_reconstructed"].fillna(0).astype(int).replace(0, ""),
            "vehicles a day": six["adt"].astype(int),
            "deck, super, sub": [
                f"{int(_a)}, {int(_b)}, {int(_c)}"
                for _a, _b, _c in zip(
                    six["deck"], six["superstructure"], six["substructure"], strict=True
                )
            ],
        }
    ).set_index("id")
    return (six,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture, with the **decoder** as this lecture's step: a bridge $\mathbf{x}$ →
    encoder (16 inputs, standardized) → $\mathbf{z}$ → decoder
    $p = \sigma(w_0 + \mathbf{w}^\top\mathbf{z})$ → a **decision**, flag if $p > 1/(1+r)$.
    Training fixes $\mathbf{w}$ with the cross-entropy loss against the label four years later,
    by Newton's method (§3–§4). The slide traces bridge B of the attempt, Duncan Rd over
    Copper Slough; §4 recomputes every number of that trace once the model is fitted.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** the log-odds of turning poor are a weighted sum of the inputs,

    $$\ln\frac{p}{1-p} = w_0 + \mathbf{w}^{\top}\mathbf{z} = s,\qquad
      p = \sigma(s) = \frac{1}{1+e^{-s}}.$$

    The odds $p/(1-p)$ range over $(0,\infty)$ and the log-odds over the whole line, the range
    of a weighted sum. A line fitted to 0/1 labels predicts below 0 and above 1; $\sigma$ maps
    any score into $(0, 1)$. The next cell checks the assumption for one input, the lowest
    rating: are the empirical log-odds near a line?
    """)
    return


@app.cell
def _(fit_unpenalized, np, pd, tr):
    _by = tr.groupby("rating")["poor_next"].agg(["sum", "size"])
    _by["share %"] = (100 * _by["sum"] / _by["size"]).round(1)
    _rate = (_by["sum"] / _by["size"]).to_numpy()
    with np.errstate(divide="ignore"):
        _by["log-odds"] = np.round(np.log(_rate / (1 - _rate)), 3)
    _line = fit_unpenalized(tr[["rating"]].to_numpy(float), tr["poor_next"].to_numpy())
    _slope = round(float(_line.coef_[0, 0]), 2)
    # Log-odds slide: 18.2%, 4.1%, 1.3%, 0.4%; none of 746 rated 9; -1.44 per point.
    assert _by["share %"].tolist() == [18.2, 4.1, 1.3, 0.4, 0.0]
    assert _by["log-odds"].tolist()[:4] == [-1.506, -3.159, -4.295, -5.526]
    assert int(_by.loc[9.0, "size"]) == 746 and int(_by.loc[9.0, "sum"]) == 0
    assert _slope == -1.44
    pd.concat([_by, pd.Series({"slope per point": _slope}, name="fit").to_frame().T])
    return


@app.cell
def _(expit, fit_unpenalized, np, plt, tr):
    # Slide "The probability is an S-shaped sheet": two inputs, rating and age. The rating does
    # the work (-1.39 per point); age adds 0.005 per year.
    _X = tr[["rating", "age"]].to_numpy(float)
    _y = tr["poor_next"].to_numpy()
    _two = fit_unpenalized(_X, _y)
    _w = [round(float(_v), 3) for _v in (_two.intercept_[0], *_two.coef_[0])]
    assert _w == [5.113, -1.388, 0.005]

    # Slide "Newton walks down one bowl": the one-input model, rating standardized, from zero.
    _z = (_X[:, 0] - _X[:, 0].mean()) / _X[:, 0].std()
    _A = np.column_stack([np.ones(len(_z)), _z])
    _path = [np.zeros(2)]
    for _ in range(12):
        _p = expit(_A @ _path[-1])
        _step = np.linalg.solve(_A.T @ (_A * (_p * (1 - _p))[:, None]), _A.T @ (_p - _y))
        _path.append(_path[-1] - _step)
        if np.abs(_step).max() < 1e-10:
            break
    assert len(_path) - 1 == 9 and [round(float(_v), 3) for _v in _path[-1]] == [-4.139, -1.612]

    _r, _a = np.meshgrid(np.linspace(4.6, 9.4, 41), np.linspace(0, 110, 41))
    _P = _two.predict_proba(np.column_stack([_r.ravel(), _a.ravel()]))[:, 1].reshape(_r.shape)
    _fig = plt.figure(figsize=(6, 4.5))
    _ax = _fig.add_subplot(projection="3d")
    _ax.plot_surface(_r, _a, _P, cmap="cividis")
    _ax.contour(_r, _a, _P, levels=[1 / 21], colors="#D55E00")
    _ax.set(xlabel="lowest rating", ylabel="age", zlabel="P(poor in 4 years)")
    _ax.view_init(elev=22, azim=-25)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Likelihood gives cross-entropy.** Each label is a coin flip with chance $p_i$. The
    likelihood is $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$; its negative log is

    $$\mathcal{L}(\mathbf{w}) = -\sum_i \big[y_i \ln p_i + (1-y_i)\ln(1-p_i)\big].$$

    With $\mathbf{A} = [\mathbf{1}\ \mathbf{Z}]$ and $p_i = \sigma(\mathbf{a}_i^\top\mathbf{w})$,
    use $\sigma'(s) = \sigma(s)(1-\sigma(s))$:

    $$\frac{\partial}{\partial \mathbf{w}}\Big[-y\ln\sigma(s) - (1-y)\ln(1-\sigma(s))\Big]
      = \Big[-y(1-\sigma) + (1-y)\sigma\Big]\mathbf{a} = (\sigma - y)\,\mathbf{a},$$

    so $\nabla\mathcal{L} = \mathbf{A}^\top(\mathbf{p}-\mathbf{y})$ and
    $\nabla^2\mathcal{L} = \sum_i p_i(1-p_i)\,\mathbf{a}_i\mathbf{a}_i^\top
    = \mathbf{A}^\top\mathbf{S}\mathbf{A}$, $\mathbf{S} = \mathrm{diag}\,p_i(1-p_i)$: a sum of
    positive semidefinite matrices. **Convex**, one minimum, but no closed form.

    **Newton's method** minimizes the local quadratic model at each step:

    $$\mathbf{w} \leftarrow \mathbf{w} - (\mathbf{A}^\top\mathbf{S}\mathbf{A})^{-1}
      \mathbf{A}^\top(\mathbf{p}-\mathbf{y}),$$

    a weighted least squares at each step (iteratively reweighted least squares), solved with
    a factorization, never an inverse (Lecture 5). If a hyperplane separates the classes
    perfectly, the likelihood has no maximum and the weights grow without bound; a ridge
    penalty fixes that.

    **Check the gradient before you trust it.** The next cell compares the analytic gradient
    with central differences at a random $\mathbf{w}$.
    """)
    return


@app.cell
def _(Z, expit, np, y):
    def nll_and_grad(w: np.ndarray, Z_: np.ndarray, y_: np.ndarray) -> tuple[float, np.ndarray]:
        """Cross-entropy of logistic regression and its gradient (weights ``[w0, w]``)."""
        _A = np.column_stack([np.ones(len(Z_)), Z_])
        _s = _A @ w
        # log(1 + e^s) - y s, computed stably with logaddexp.
        _loss = float(np.sum(np.logaddexp(0.0, _s) - y_ * _s))
        return _loss, _A.T @ (expit(_s) - y_)

    _rng = np.random.default_rng(0)
    _w = _rng.normal(scale=0.3, size=Z.shape[1] + 1)
    _, _g = nll_and_grad(_w, Z, y)
    _h = 1e-5
    _num = np.array(
        [
            (nll_and_grad(_w + _h * _e, Z, y)[0] - nll_and_grad(_w - _h * _e, Z, y)[0]) / (2 * _h)
            for _e in np.eye(len(_w))
        ]
    )
    _rel = float(np.max(np.abs(_num - _g)) / np.max(np.abs(_g)))
    assert _rel < 1e-6, _rel
    print(f"gradient check: max relative difference {_rel:.1e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: where to draw the line

    A false alarm costs 1. A miss costs $r$. Correct decisions cost 0. The model says $p$.

    1. Write the expected cost of flagging a bridge, and of letting it wait.
    2. Find the threshold on $p$ above which you flag. Evaluate it at $r = 20$.
    3. Apply it to the six bridges of the attempt (their $p$ is in §4, after the fit).
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer: costs set the threshold": mo.md(
                r"""
                Flag: expected cost $(1-p)\cdot 1$ (it may hold anyway). Wait: $p \cdot r$. Flag
                when $p\,r > 1 - p$, that is, when

                $$p > \frac{1}{1+r}.$$

                At $r = 20$: $p > 1/21 = 0.048$, not 0.5 (Elkan, 2001). Costs are counted
                relative to the correct decision, so only the two errors carry a cost. A fixed
                budget, flag the top $k$, gives the same rule with an implied $r$. The rule
                needs **calibrated** $p$: of the bridges given 0.05, about 5 in 100 must turn
                poor (§5 checks it).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The SVM keeps a margin, not odds** (Forsyth §2.1; Cortes & Vapnik, 1995). With labels
    $y_i = \pm 1$ and score $s_i = w_0 + \mathbf{w}^\top\mathbf{z}_i$, Forsyth's soft-margin
    SVM uses the hinge, $\max(0,\, 1 - y_i s_i)$. We fit its square, the L2-SVM, which
    minimizes

    $$\frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_i \max(0,\, 1 - y_i s_i)^2.$$

    This is scikit-learn's `LinearSVC` with `loss="squared_hinge"` (its default); its solver,
    liblinear, also penalizes $w_0$, as the weight of a constant input. Either hinge is zero
    beyond the margin, so only the bridges near the boundary (the support vectors,
    $1 - y_i s_i > 0$) shape it; the margin is $2/\|\mathbf{w}\|$ wide. Logistic regression's
    loss, $\ln(1 + e^{-y s})$, never reaches zero and decays like the hinge for large $ys$
    (Forsyth §11.3.1 makes the comparison). The SVM's score is not a
    probability; Platt scaling fits $\sigma(a s + b)$ to it on held-out data (Platt, 1999).

    **Ratings are ordered: cut once.** The proportional-odds (cumulative-logit) model
    (McCullagh, 1980) uses every later rating, not only poor or not:

    $$P(y \le j \mid \mathbf{z}) = \sigma(\theta_j - \mathbf{w}^\top\mathbf{z}),\qquad
      \theta_1 < \theta_2 < \dots < \theta_5,$$

    with six outcomes: poor (4 or less), 5, 6, 7, 8, 9. One weight vector, five cutpoints;
    the chance of each outcome is the difference of two adjacent cumulative probabilities.
    **Assumption:** what moves a bridge from 8 to 7 also moves it from 5 to 4. §5 tests it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    Newton's method from §3, ten lines of NumPy (the slide's listing).
    """)
    return


@app.cell
def _(expit, np):
    def fit_logistic(
        Z: np.ndarray, y: np.ndarray, tol: float = 1e-10, max_steps: int = 50
    ) -> tuple[np.ndarray, int]:
        """Maximum-likelihood logistic regression by Newton's method.

        Args:
            Z: Standardized inputs, shape ``(n, d)``.
            y: Labels in {0, 1}, shape ``(n,)``.
            tol: Stop when no weight moves more than this.
            max_steps: Give up after this many steps.

        Returns:
            Weights ``[w0, w]`` of shape ``(d + 1,)``, and the number of steps taken.

        Raises:
            RuntimeError: If Newton's method has not converged after ``max_steps``, as on
                (quasi-)separated data, whose likelihood has no maximum (§6 meets one).
        """
        A = np.column_stack([np.ones(len(Z)), Z])  # intercept column
        w = np.zeros(A.shape[1])
        for step in range(1, max_steps + 1):
            p = expit(A @ w)  # sigma(s), s = A w
            g = A.T @ (p - y)  # gradient of the NLL
            H = A.T @ (A * (p * (1 - p))[:, None])  # Hessian A^T S A
            delta = np.linalg.solve(H, g)  # Newton step
            w -= delta
            if np.abs(delta).max() < tol:
                return w, step
        raise RuntimeError("Newton did not converge")

    def predict_proba(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """P(poor in four years) for standardized inputs.

        Args:
            w: Weights ``[w0, w]``, shape ``(d + 1,)``.
            Z: Standardized inputs, shape ``(n, d)``.

        Returns:
            ``sigma(w0 + Z w)``, shape ``(n,)``.
        """
        return expit(w[0] + Z @ w[1:])

    return fit_logistic, predict_proba


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    1. How many Newton steps from $\mathbf{w} = \mathbf{0}$?
    2. Against scikit-learn's `LogisticRegression(C=np.inf)` with its **default** solver, at
       `rtol=1e-6`: pass or fail? And with `solver="newton-cholesky"`?
    """)
    return


@app.cell
def _(Z, fit_logistic, fit_unpenalized, log_loss, np, predict_proba, tr, y):
    w, steps = fit_logistic(Z, y)
    _nc = fit_unpenalized(Z, y, tol=1e-12, max_iter=200)
    _nc_w = np.concatenate([_nc.intercept_, _nc.coef_[0]])
    _rel = float(np.max(np.abs(w - _nc_w) / (np.abs(_nc_w) + 1e-12)))
    assert np.allclose(w, _nc_w, rtol=1e-6, atol=0) and _rel < 1e-10, _rel
    _lb = fit_unpenalized(Z, y, solver="lbfgs", max_iter=10_000)  # scikit-learn's default
    _lb_w = np.concatenate([_lb.intercept_, _lb.coef_[0]])
    _lb_pass = bool(np.allclose(w, _lb_w, rtol=1e-6, atol=0))
    _loss_gap = log_loss(y, _lb.predict_proba(Z)[:, 1]) - log_loss(y, predict_proba(w, Z))
    _p_gap = float(np.max(np.abs(predict_proba(w, Z) - _lb.predict_proba(Z)[:, 1])))
    _tight = fit_unpenalized(Z, y, solver="lbfgs", max_iter=100_000, tol=1e-8)
    _tight_gap = float(np.max(np.abs(w - np.concatenate([_tight.intercept_, _tight.coef_[0]]))))
    _other = int((tr["material"] == "other").sum())
    # Bar slide and notes: 9 steps; Newton solver to 1e-12; the default solver fails, 0.4 off,
    # after 19 iterations; losses differ by 1e-5, a probability by 0.13; 14 bridges of "other"
    # material; at tol 1e-8 it agrees to 4e-6.
    assert steps == 9 and not _lb_pass and int(_lb.n_iter_[0]) == 19
    assert round(float(np.max(np.abs(w - _lb_w))), 1) == 0.4 and round(_p_gap, 2) == 0.13
    assert 0 < _loss_gap < 2e-5 and _other == 14 and _tight_gap < 1e-5
    print(f"Newton: {steps} steps; newton-cholesky max relative difference {_rel:.1e}")
    print(
        f"default L-BFGS: pass={_lb_pass}, {_lb.n_iter_[0]} iterations, "
        f"max weight gap {np.max(np.abs(w - _lb_w)):.2f}, loss gap {_loss_gap:.0e}, "
        f"max probability gap {_p_gap:.2f}; at tol=1e-8: {_tight_gap:.0e}"
    )
    return (w,)


@app.cell
def _(mo):
    mo.md(r"""
    **Why the default solver failed.** L-BFGS stops when its tolerance (`tol=1e-4`) is met.
    Only 14 training bridges are of "other" material, so the four material indicators nearly
    sum to one: the loss is almost flat along one direction, and L-BFGS stopped on it, 0.4
    from the optimum, with the loss only $10^{-5}$ higher. With `tol=1e-8` it agrees to
    $4\times10^{-6}$. **A check needs a reference whose tolerance you know**, and a tolerance
    stated for the quantity you use (weights, losses, or probabilities).

    **A ridge penalty?** The next cell fits $L_2$-penalized models on the training bridges and
    scores them on 2016 → 2020.
    """)
    return


@app.cell
def _(LogisticRegression, Z, Zv, log_loss, predict_proba, w, y, yv):
    _none = round(float(log_loss(yv, predict_proba(w, Zv))), 5)
    _l2 = {}
    _grid = (0.01, 0.1, 1.0, 10.0)
    for _C in _grid:
        _pv = LogisticRegression(C=_C, solver="newton-cholesky").fit(Z, y).predict_proba(Zv)[:, 1]
        _l2[_C] = round(float(log_loss(yv, _pv)), 5)
    _best = min(_l2, key=_l2.get)
    if _best in (_grid[0], _grid[-1]):  # a best value at the edge says: widen the grid
        raise RuntimeError(f"best C {_best} is at an edge of {_grid}")
    # Notes: the best C improves the validation log loss by 5e-5.
    assert _none == 0.14664 and round(_none - min(_l2.values()), 5) == 5e-05
    print(f"validation log loss, no penalty {_none}; with L2: {_l2}")
    return


@app.cell
def _(INPUTS, StandardScaler, Zt, expit, fit_logistic, np, pd, predict_proba, te, tr, w):
    # The trace of bridge B (slide "The decoder outputs a probability"). The material is one
    # input coded as four 0/1 columns against the 14 "other" bridges, so one column alone is
    # an artefact of that small baseline. The slide reads material with prestressed, the most
    # common, as the baseline: the same model, refitted with an "other material" column in
    # place of the prestressed one, gives the same p; only the attribution moves.
    _k = INPUTS.index("prestressed")
    _names = ["other material" if _n == "prestressed" else _n for _n in INPUTS]

    def _recode(records: pd.DataFrame) -> np.ndarray:
        """The inputs with the prestressed column replaced by an "other material" column.

        Args:
            records: Transitions with every column of ``INPUTS`` and ``material``.

        Returns:
            The recoded inputs, unscaled, shape ``(len(records), len(INPUTS))``.
        """
        _x = records[INPUTS].to_numpy(np.float64).copy()
        _x[:, _k] = (records["material"] == "other").to_numpy(np.float64)
        return _x

    _sc2 = StandardScaler().fit(_recode(tr))
    _w2, _steps2 = fit_logistic(_sc2.transform(_recode(tr)), tr["poor_next"].to_numpy())
    _Zt2 = _sc2.transform(_recode(te))
    assert np.abs(predict_proba(_w2, _Zt2) - predict_proba(w, Zt)).max() < 1e-9  # same model
    _i = int(np.flatnonzero(te["structure"].to_numpy() == "010601000000000")[0])
    _terms, _own = _w2[1:] * _Zt2[_i], w[1:] * Zt[_i]
    _mat = [INPUTS.index(_m) for _m in ("concrete", "steel", "prestressed", "timber")]
    assert abs(_terms[_mat].sum() - _own[_mat].sum()) < 1e-9  # material, read whole: the same
    _top = np.argsort(-np.abs(_terms))[:3]
    _s = float(_w2[0] + _terms.sum())
    trace = {
        "top": [
            (_names[_j], round(float(_Zt2[_i, _j]), 2), round(float(_terms[_j]), 2)) for _j in _top
        ],
        "rest": round(float(_terms.sum() - _terms[_top].sum()), 2),
        "w0": round(float(_w2[0]), 2),
        "score": round(_s, 2),
        "p": round(float(expit(_s)), 2),
        "material terms": round(float(_terms[_mat].sum()), 2),
        "prestressed column, own coding": round(float(_own[_k]), 2),
    }
    # Values from the figure script's numbers.json (trace), copied by script.
    assert trace["top"] == [
        ("rating", -1.6, 1.67),
        ("superstructure", -1.98, 0.62),
        ("deck", -2.08, 0.28),
    ], trace["top"]
    assert (trace["rest"], trace["w0"], trace["score"], trace["p"]) == (0.12, -4.21, -1.52, 0.18)
    assert (trace["material terms"], trace["prestressed column, own coding"]) == (-0.15, -0.49)
    assert int(te.loc[_i, "rating_next"]) == 2
    print("bridge B:", trace)
    return


@app.cell
def _(R, Zt, expit, six, w):
    # The six bridges: p, the ranking, and the flags at r = 20.
    _p6 = expit(w[0] + Zt[six.index.to_numpy()] @ w[1:])
    six_p = {_k: round(float(_v), 3) for _k, _v in zip(six["id"], _p6, strict=True)}
    assert six_p == {"A": 0.049, "B": 0.179, "C": 0.044, "D": 0.038, "E": 0.134, "F": 0.2}
    assert sorted(six_p, key=lambda _k: -six_p[_k]) == list("FBEACD")
    assert [_k for _k in six_p if six_p[_k] > 1 / (1 + R)] == list("ABEF")
    assert six.set_index("id")["rating_next"].le(4).loc[lambda _s: _s].index.tolist() == ["B", "F"]
    print("six bridges, p:", six_p, "| flagged at r = 20: A, B, E, F | turned poor: B, F")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Bracket the result.** The error that matters is the cost per bridge at $r = 20$:
    $(\text{false alarms} + r\cdot\text{misses})/n$. Above: flag none, flag all, and the
    engineer's rule, **flag every bridge rated 5**, which uses the same inspection. Below: a
    rough floor from **twin bridges** (parallel structures built the same year, side by
    side): flag a twin if its twin turned poor. Validation period, 2016 → 2020.

    Twins turn poor more often than the average bridge (5.4% against 4.5%), and the cost per
    bridge grows with that share, so the bracket costs every row on the **same twin records**
    as the floor; the first column is all bridges, for reference. Validation also shares
    2016 to 2018 and the same bridges with training; §5 checks, in hindsight, whether that is
    why it ranks better than the test period.
    """)
    return


@app.cell
def _(NearestNeighbors, R, TWIN_KM, np, pd):
    def expected_cost(flag: np.ndarray, y_: np.ndarray, r: float = R) -> float:
        """Cost per bridge: 1 per false alarm, ``r`` per miss."""
        assert flag.dtype == bool, flag.dtype  # ~ on integers would be a bitwise not
        return float((r * ((~flag) & (y_ == 1)).sum() + (flag & (y_ == 0)).sum()) / len(y_))

    def caught_at_share(score: np.ndarray, y_: np.ndarray, share: float) -> float:
        """Share of the bridges that turned poor among the top ``share`` by score."""
        _top = np.argsort(-score, kind="stable")[: int(round(share * len(y_)))]
        return float(y_[_top].sum() / y_.sum())

    def find_twins(records: pd.DataFrame) -> pd.DataFrame:
        """Parallel structures (Item 101 L or R) built the same year, within ``TWIN_KM``."""
        _par = records[records["parallel"].isin(["R", "L"])]
        _xy = _par[["x_km", "y_km"]].to_numpy()
        _dist, _idx = NearestNeighbors(n_neighbors=3).fit(_xy).kneighbors(_xy)
        _rows, _own = _par.index.to_numpy(), np.arange(len(_par))
        _first = np.where(_idx[:, 0] == _own, 1, 0)  # coincident points: skip the record
        _out = pd.DataFrame(
            {"row": _rows, "twin_row": _rows[_idx[_own, _first]], "km": _dist[_own, _first]}
        )
        _built = records["year_built"]
        _same = _built.loc[_out["row"]].to_numpy() == _built.loc[_out["twin_row"]].to_numpy()
        return _out[(_out["km"] < TWIN_KM) & _same].reset_index(drop=True)

    return caught_at_share, expected_cost, find_twins


@app.cell
def _(R, Zv, expected_cost, find_twins, np, pd, predict_proba, va, w, yv):
    p_val = predict_proba(w, Zv)
    _flag = p_val > 1 / (1 + R)
    _r5 = va["rating"].to_numpy() == 5
    _tw = find_twins(va)
    assert va.index.equals(pd.RangeIndex(len(va)))  # twin rows index p_val by position
    _t = _tw["row"].to_numpy()
    _a = va["poor_next"].to_numpy()[_t]
    _b = va["poor_next"].to_numpy()[_tw["twin_row"].to_numpy()]
    _rows = {
        "flag all": (np.ones_like(_flag), np.ones(len(_t), bool)),
        "flag none": (np.zeros_like(_flag), np.zeros(len(_t), bool)),
        "rule: flag every 5": (_r5, _r5[_t]),
        "model, p > 1/21": (_flag, _flag[_t]),
        "twin's outcome (rough)": (None, _b == 1),
    }
    _bracket = pd.DataFrame(
        {
            "all bridges": [
                np.nan if _f is None else expected_cost(_f, yv) for _f, _ in _rows.values()
            ],
            "twin records": [expected_cost(_ft, _a) for _, _ft in _rows.values()],
        },
        index=list(_rows),
    ).round(3)
    # Bar slide (twin records): 0.95, 1.07, 0.61, 0.52, 0.37; body and notes (all bridges):
    # model 0.43, rule 0.45, flag none 0.91; the model flags 28% and catches 80%, the rule
    # flags 16% and catches 65%; 1,546 twin records, 96.4% agree, 5.4% turned poor.
    assert _bracket["twin records"].tolist() == [0.946, 1.074, 0.609, 0.52, 0.367]
    assert _bracket["all bridges"].tolist()[:4] == [0.955, 0.908, 0.453, 0.426]
    assert round(100 * _flag.mean(), 1) == 28.3
    assert round(100 * (_flag & (yv == 1)).sum() / yv.sum(), 1) == 80.3
    assert (round(100 * _r5.mean(), 1), round(100 * (_r5 & (yv == 1)).sum() / yv.sum(), 1)) == (
        16.3,
        64.8,
    )
    assert (len(_tw), round(100 * (_a == _b).mean(), 1), round(100 * _a.mean(), 1)) == (
        1546,
        96.4,
        5.4,
    )
    _bracket
    return (p_val,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **0.5.** Flag when $p > 0.5$. How many of the 793 bridges that turned poor by 2024 does
       it catch? What is its accuracy?
    2. **Costs.** Flag when $p > 1/21$. Validation promised 0.43 against the rule's 0.45. In
       2024?
    3. **Ordinal.** The model that sees every later rating: will it catch more of the poor?
    4. **2024.** The model learned 2014 → 2018. Will its probabilities hold for 2020 → 2024?
    """)
    return


@app.cell
def _(Zt, predict_proba, w, yt):
    p_test = predict_proba(w, Zt)
    _half = p_test > 0.5
    _acc_half = round(100 * float((_half == yt).mean()), 2)
    _acc_none = round(100 * float(1 - yt.mean()), 2)
    # Slide "0.5 flags no bridge": highest p 0.497; none flagged; 95.96% for both.
    assert round(float(p_test.max()), 3) == 0.497 and int(_half.sum()) == 0
    assert _acc_half == _acc_none == 95.96
    print(
        f"highest p {p_test.max():.3f}; p > 0.5 flags {_half.sum()}; accuracy {_acc_half}% "
        f"= flagging nobody {_acc_none}%"
    )
    return (p_test,)


@app.cell
def _(MARKERS, PALETTE, R, expected_cost, np, p_test, plt, roc_auc_score, te, yt):
    _t = 1 / (1 + R)
    _flag = p_test > _t
    _rule = te["rating"].to_numpy() == 5
    _grid = np.unique(np.concatenate([np.linspace(0.002, 0.6, 600), [_t]]))
    _costs = np.array([expected_cost(p_test > _s, yt) for _s in _grid])
    _near = _grid[_costs <= _costs.min() + 0.02]
    # Slide "Costs, not 0.5, set the threshold": flags 31.5%, catches 631 of 793 (80%); cost
    # 0.45 against 0.81 at 0.5; the rule 0.44; best in hindsight 0.074 at 0.44; within 0.02 of
    # the best from 0.045 to 0.12 (about: the band has gaps, up to 0.0218 above); flag all 0.96;
    # 162 missed; ranked by the lowest rating alone, AUC 0.796 against the model's 0.815.
    assert round(100 * _flag.mean(), 1) == 31.5 and int((_flag & (yt == 1)).sum()) == 631
    assert int(((~_flag) & (yt == 1)).sum()) == 162
    assert round(roc_auc_score(yt, -te["rating"].to_numpy()), 3) == 0.796
    assert round(expected_cost(_flag, yt), 3) == 0.447
    assert round(expected_cost(p_test > 0.5, yt), 3) == 0.808
    assert round(expected_cost(_rule, yt), 3) == 0.436
    assert round(float(_grid[np.argmin(_costs)]), 3) == 0.074 and round(_costs.min(), 3) == 0.438
    assert (round(float(_near.min()), 3), round(float(_near.max()), 3)) == (0.045, 0.117)
    assert round(expected_cost(np.ones_like(_flag), yt), 2) == 0.96

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _g = np.geomspace(0.005, 0.7, 300)
    _ax.plot(
        _g,
        [expected_cost(p_test > _s, yt) for _s in _g],
        color=PALETTE[0],
        label="model: flag if p > threshold",
    )
    _ax.axhline(expected_cost(np.zeros_like(_flag), yt), color="grey", ls=":", label="flag none")
    _ax.axhline(expected_cost(_rule, yt), color=PALETTE[2], ls="--", label="rule: flag every 5")
    _ax.plot([_t], [expected_cost(_flag, yt)], MARKERS[1], color=PALETTE[1], ms=10, label="1/21")
    _ax.plot(
        [0.5], [expected_cost(p_test > 0.5, yt)], MARKERS[3], color=PALETTE[3], ms=10, label="0.5"
    )
    _ax.set_xscale("log")
    _ax.set_xlabel(f"threshold on p (r = {R})")
    _ax.set_ylabel("cost per bridge, 2020 → 2024")
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **In 2024 the model only ties the engineer's rule** (0.45 against 0.44), though validation
    promised 0.43 against 0.45. The bottom of the curve is wide, so the threshold is not the
    problem: the best threshold in hindsight costs 0.44, the same as the rule. At $r = 20$ the
    lowest rating does most of the work: ranked by it alone, the 2020 bridges reach an AUC of
    0.796, against the model's 0.815. A model must beat the engineer's rule, not only the
    coin flip. The next two tests say why the promise shrank.

    **The ordinal model, from scratch.** Six outcomes, five cutpoints kept in order by
    writing $\theta_1 = a_1$ and $\theta_j = \theta_{j-1} + e^{a_j}$.
    """)
    return


@app.cell
def _(OUTCOMES, expit, minimize, np):
    def _thresholds(a: np.ndarray) -> np.ndarray:
        """Increasing cutpoints from unconstrained parameters (first, then log-gaps)."""
        return np.concatenate([[a[0]], a[0] + np.cumsum(np.exp(a[1:]))])

    def fit_ordinal(Z_: np.ndarray, r_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative-logit model: ``P(r <= OUTCOMES[j]) = sigma(theta_j - w.z)``.

        Args:
            Z_: Standardized inputs, shape ``(n, d)``.
            r_: Later ratings, integers 0-9.

        Returns:
            Weights ``w`` of shape ``(d,)`` and increasing cutpoints of shape
            ``(len(OUTCOMES),)``.

        Raises:
            RuntimeError: If the optimizer fails.
        """
        _cls = np.searchsorted(OUTCOMES, r_, side="left")  # 0 = poor, ..., 5 = rated 9
        _d = Z_.shape[1]

        def _nll(params: np.ndarray) -> float:
            _w, _theta = params[:_d], _thresholds(params[_d:])
            _cdf = expit(_theta[None, :] - (Z_ @ _w)[:, None])
            _cdf = np.column_stack([np.zeros(len(Z_)), _cdf, np.ones(len(Z_))])
            _prob = _cdf[np.arange(len(Z_)), _cls + 1] - _cdf[np.arange(len(Z_)), _cls]
            return float(-np.log(np.clip(_prob, 1e-300, None)).sum())

        _x0 = np.concatenate([np.zeros(_d), [-3.0], np.zeros(len(OUTCOMES) - 1)])
        # Tight tolerances: the default ones stop on a flat direction, 2e-3 from the optimum.
        _res = minimize(
            _nll, _x0, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-8}
        )
        if not _res.success:
            raise RuntimeError(f"ordinal fit failed: {_res.message}")
        return _res.x[:_d], _thresholds(_res.x[_d:])

    return (fit_ordinal,)


@app.cell
def _(OUTCOMES, Z, fit_ordinal, np, tr):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    w_ord, theta_ord = fit_ordinal(Z, tr["rating_next"].to_numpy(int))
    _cls = np.searchsorted(OUTCOMES, tr["rating_next"].to_numpy(int), side="left")
    _ref = OrderedModel(_cls, Z, distr="logit").fit(
        method="bfgs", disp=False, maxiter=5000, gtol=1e-8
    )
    _ref_w = _ref.params[: Z.shape[1]]
    _ref_theta = _ref.model.transform_threshold_params(_ref.params)[1:-1]
    _gap_w = float(np.max(np.abs(w_ord - _ref_w)))
    _gap_t = float(np.max(np.abs(theta_ord - _ref_theta)))
    assert _gap_w < 1e-4 and _gap_t < 1e-4, (_gap_w, _gap_t)
    print(
        f"ordinal vs statsmodels OrderedModel: weights to {_gap_w:.0e}, cutpoints to {_gap_t:.0e}"
    )
    return theta_ord, w_ord


@app.cell
def _(
    FLAG_SHARE,
    INPUTS,
    OUTCOMES,
    PALETTE,
    R,
    Z,
    Zt,
    caught_at_share,
    expected_cost,
    expit,
    fit_unpenalized,
    p_test,
    plt,
    roc_auc_score,
    theta_ord,
    tr,
    w_ord,
    yt,
):
    _p_ord = expit(theta_ord[0] - Zt @ w_ord)  # P(later rating <= 4)
    _show = ["age", "superstructure"]
    _r_next = tr["rating_next"].to_numpy(int)
    _per_cut = {}
    for _j in OUTCOMES:
        _fit = fit_unpenalized(Z, (_r_next <= _j).astype(int))
        _per_cut[int(_j)] = [round(float(_fit.coef_[0, INPUTS.index(_n)]), 3) for _n in _show]
    _shared = [round(float(-w_ord[INPUTS.index(_n)]), 3) for _n in _show]
    # Slide "One weight vector misfits every cut": 39% against 43% caught at a 10% budget;
    # costs 0.44 and 0.45; AUC 0.811 and 0.815; age 0.01 for poor, 0.26 for 7 or less.
    assert round(100 * caught_at_share(_p_ord, yt, FLAG_SHARE)) == 39
    assert round(100 * caught_at_share(p_test, yt, FLAG_SHARE)) == 43
    assert round(expected_cost(_p_ord > 1 / (1 + R), yt), 3) == 0.437
    assert (round(roc_auc_score(yt, _p_ord), 3), round(roc_auc_score(yt, p_test), 3)) == (
        0.811,
        0.815,
    )
    assert [_per_cut[_j][0] for _j in (4, 5, 6, 7, 8)] == [0.007, 0.037, 0.093, 0.263, -0.267]
    assert [_per_cut[_j][1] for _j in (4, 5, 6, 7, 8)] == [-0.312, -0.214, -0.441, 0.332, -0.272]
    assert _shared == [0.103, -0.223]

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for _k, (_ax, _n) in enumerate(zip(_axes, _show, strict=True)):
        _ax.plot(
            range(len(OUTCOMES)),
            [_per_cut[int(_j)][_k] for _j in OUTCOMES],
            "o-",
            color=PALETTE[_k],
            label="separate fit per cut",
        )
        _ax.axhline(_shared[_k], color="grey", ls="--", label="ordinal: one weight")
        _ax.set_xticks(range(len(OUTCOMES)), [f"≤{int(_j)}" for _j in OUTCOMES])
        _ax.set_title(_n)
        _ax.set_xlabel("later rating")
    _axes[0].set_ylabel("weight toward a lower rating")
    _axes[0].legend(fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **More labels did not help.** What moves a bridge from 8 to 7 (age, size) is not what moves
    it from 5 to 4 (the condition of the superstructure), so one weight vector for every cut
    misfits the cut we care about. Separate fits, one per cut, are the check of the
    proportional-odds assumption.

    **Calibration over time.** The cost threshold needs calibrated probabilities. Compare the
    mean predicted chance with the observed share, overall and in tenths of predicted risk.
    """)
    return


@app.cell
def _(MARKERS, PALETTE, R, expected_cost, p_test, p_val, pd, plt, roc_auc_score, yt, yv):
    _fig, _ax = plt.subplots(figsize=(5, 3.6))
    _out = {}
    for _k, (_name, _p, _y) in enumerate(
        (("2016 → 2020 (validation)", p_val, yv), ("2020 → 2024 (test)", p_test, yt))
    ):
        _tab = pd.DataFrame({"p": _p, "y": _y, "q": pd.qcut(_p, 10, labels=False)})
        _tab = _tab.groupby("q").mean()
        _ax.plot(
            100 * _tab["p"], 100 * _tab["y"], marker=MARKERS[_k], color=PALETTE[_k], label=_name
        )
        _out[_name] = (
            round(100 * _p.mean(), 1),
            round(100 * _y.mean(), 1),
            round(100 * _tab["p"].iloc[-1]),
            round(100 * _tab["y"].iloc[-1]),
        )
    _ax.plot([0, 26], [0, 26], ":", color="grey")
    _ax.set_xlabel("predicted, % turning poor")
    _ax.set_ylabel("observed, %")
    _ax.legend(fontsize=9)
    # Slide "2024 probabilities run high": 4.5% predicted and observed in 2016 → 2020, top tenth
    # 20% and 22%; 5.0% against 4.0% in 2020 → 2024, top tenth 21% and 17%; AUC 0.847, 0.815.
    assert _out["2016 → 2020 (validation)"] == (4.5, 4.5, 20, 22)
    assert _out["2020 → 2024 (test)"] == (5.0, 4.0, 21, 17)
    assert (round(roc_auc_score(yv, p_val), 3), round(roc_auc_score(yt, p_test), 3)) == (
        0.847,
        0.815,
    )
    assert round(expected_cost(p_test > 1 / (1 + R), yt), 2) == 0.45
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    INPUTS,
    R,
    StandardScaler,
    df,
    encode,
    expected_cost,
    fit_logistic,
    fit_unpenalized,
    logit,
    np,
    p_test,
    p_val,
    predict_proba,
    roc_auc_score,
    te,
    transitions,
    va,
    w,
    yt,
    yv,
):
    # Checks in hindsight (post hoc), never used for a decision: could the 2021 engineer have
    # fixed the 2024 calibration with 2016 -> 2020, the latest period she had? And is the
    # overlap with training why validation ranks better? A later period, 2018 -> 2022, answers.
    _t = 1 / (1 + R)
    _platt = fit_unpenalized(logit(p_val)[:, None], yv)  # Platt scaling of the log-odds
    _pp = _platt.predict_proba(logit(p_test)[:, None])[:, 1]
    _x = va[INPUTS].to_numpy(np.float64)
    _sc = StandardScaler().fit(_x)
    try:  # quasi-separation: the likelihood has no maximum
        fit_logistic(_sc.transform(_x), yv)
        _converges = True
    except RuntimeError:
        _converges = False
    _refit = fit_unpenalized(_sc.transform(_x), yv, tol=1e-12, max_iter=200)
    _pr = _refit.predict_proba(_sc.transform(te[INPUTS].to_numpy(np.float64)))[:, 1]
    _other = (va["material"] == "other").to_numpy()
    _later = transitions(df, 2018)
    _yl, _pl = _later["poor_next"].to_numpy(), predict_proba(w, encode(_later))
    post_hoc = {
        "2024: mean p %, observed %": (round(100 * p_test.mean(), 2), round(100 * yt.mean(), 2)),
        "Platt on 2016-20: 2024 mean p %, cost": (
            round(100 * _pp.mean(), 2),
            round(expected_cost(_pp > _t, yt), 3),
        ),
        "refit on 2016-20: 2024 mean p %, cost": (
            round(100 * _pr.mean(), 2),
            round(expected_cost(_pr > _t, yt), 3),
        ),
        "Newton listing converges on 2016-20": _converges,
        "2016-20: other-material bridges, turned poor": (int(_other.sum()), int(yv[_other].sum())),
        "2018-22: mean p %, observed %": (round(100 * _pl.mean(), 2), round(100 * _yl.mean(), 2)),
        "2018-22: AUC, rating alone": (
            round(roc_auc_score(_yl, _pl), 3),
            round(roc_auc_score(_yl, -_later["rating"].to_numpy()), 3),
        ),
        "rating alone, AUC: 2016-20, 2020-24": (
            round(roc_auc_score(yv, -va["rating"].to_numpy()), 3),
            round(roc_auc_score(yt, -te["rating"].to_numpy()), 3),
        ),
    }
    # Values from the figure script's numbers.json (post_hoc, ranking), copied by script.
    assert post_hoc["2024: mean p %, observed %"] == (5.03, 4.04)
    assert post_hoc["Platt on 2016-20: 2024 mean p %, cost"] == (5.06, 0.447)
    assert post_hoc["refit on 2016-20: 2024 mean p %, cost"] == (5.07, 0.445)
    assert post_hoc["Newton listing converges on 2016-20"] is False
    assert post_hoc["2016-20: other-material bridges, turned poor"] == (13, 0)
    assert post_hoc["2018-22: mean p %, observed %"] == (4.79, 4.09)
    assert post_hoc["2018-22: AUC, rating alone"] == (0.843, 0.824)
    assert post_hoc["rating alone, AUC: 2016-20, 2020-24"] == (0.823, 0.796)
    post_hoc
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The ranking held; the level moved.** Fewer bridges turned poor from 2020 to 2024 than
    the 2014–2018 rates predicted, so $p > 1/21$ flags too many: 0.45, against 0.44 at the
    best threshold in hindsight.

    **Could the 2021 engineer have fixed it?** Not with 2016 → 2020, the latest period she had
    (the cell above, in hindsight): Platt scaling on it still predicts 5.06% for 2024, and a
    refit on it 5.07%, against 4.04% observed. The Newton listing does not even converge there:
    none of the 13 "other"-material bridges of 2016 → 2020 turned poor, so the likelihood has
    no maximum (a ridge penalty fixes that). The drift shows only after 2020: 2018 → 2022,
    unknown in 2021, already ran high (4.79% predicted, 4.09% observed). So recheck
    calibration as each new period arrives.

    **Validation ranks better, but not because it overlaps training.** 2018 → 2022 ranks like
    validation (AUC 0.843 against 0.847), and the rating alone drops the same way in
    2020 → 2024 (0.823 and 0.824, then 0.796). Checking calibration: Lecture 10; coverage
    guarantees under shift, Lecture 11.

    **The SVM on the same inputs**, then Platt scaling on the validation scores.
    """)
    return


@app.cell
def _(
    FLAG_SHARE,
    LinearSVC,
    R,
    SEED,
    Z,
    Zt,
    Zv,
    caught_at_share,
    expected_cost,
    fit_unpenalized,
    p_test,
    roc_auc_score,
    y,
    yt,
    yv,
):
    # The squared hinge (L2-SVM), LinearSVC's default, set explicitly; liblinear also
    # penalizes the intercept, as the weight of a constant input (intercept_scaling = 1).
    _kw = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "max_iter": 100_000}
    _svm = LinearSVC(**_kw, random_state=SEED).fit(Z, y)
    _s_val, _s = _svm.decision_function(Zv), _svm.decision_function(Zt)
    _platt = fit_unpenalized(_s_val[:, None], yv)
    _p_platt = _platt.predict_proba(_s[:, None])[:, 1]
    # SVM slide: AUC 0.815 for both; no positive score, in 2020 or in training; Platt 0.44.
    assert round(roc_auc_score(yt, _s), 3) == round(roc_auc_score(yt, p_test), 3) == 0.815
    assert int((_s > 0).sum()) == 0 and int((_svm.decision_function(Z) > 0).sum()) == 0
    assert round(100 * caught_at_share(_s, yt, FLAG_SHARE)) == 43
    assert round(expected_cost(_p_platt > 1 / (1 + R), yt), 3) == 0.441
    # Notes: no positive score is a matter of costs. Weight the classes equally and the same
    # loss gives 6,221 training bridges a positive score.
    _balanced = LinearSVC(**_kw, class_weight="balanced", random_state=SEED).fit(Z, y)
    assert int((_balanced.decision_function(Z) > 0).sum()) == 6221
    print(
        f"SVM: AUC {roc_auc_score(yt, _s):.3f}; positive scores {(_s > 0).sum()}; "
        f"Platt-scaled cost at r = {R}: {expected_cost(_p_platt > 1 / (1 + R), yt):.3f}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    With 4% positives and the same cost on both sides of the hinge, the boundary that
    minimizes the training loss flags nobody, even among the training bridges; with the
    classes weighted equally (`class_weight="balanced"`), 6,221 of them get a positive score.
    The SVM ranks as well as logistic regression; only a calibrated score turns a rank into a
    decision.

    ## 6. Decide

    **Flag above $1/(1+r)$; the owner sets $r$.** The model gives one dial for every $r$; the
    rules give two points.
    """)
    return


@app.cell
def _(SEED, np):
    BOOT_DRAWS = 2000  # county-cluster bootstrap draws, as in the figure script
    CI = (2.5, 97.5)  # percentiles of a 95% interval

    def county_draws(
        num: np.ndarray, den: np.ndarray, county: np.ndarray, stream: int
    ) -> np.ndarray:
        """Bootstrap draws of ratios of sums, resampling whole counties.

        Bridges in one county share inspectors, budgets, and weather, so each draw resamples
        the counties with replacement and keeps all of a county's bridges.

        Args:
            num: Per-bridge numerators, shape ``(n, k)``.
            den: Per-bridge denominators, shape ``(n, k)``.
            county: Each bridge's county code, shape ``(n,)``.
            stream: The generator's stream (with ``SEED``), the figure script's for each interval.

        Returns:
            ``sum(num) / sum(den)`` per column for each draw, shape ``(BOOT_DRAWS, k)``.

        Raises:
            RuntimeError: If a draw has a zero denominator.
        """
        _codes, _inv = np.unique(county, return_inverse=True)
        _S = np.zeros((len(_codes), num.shape[1]))
        _D = np.zeros_like(_S)
        np.add.at(_S, _inv, num)
        np.add.at(_D, _inv, den)
        _rng = np.random.default_rng((SEED, stream))
        _idx = _rng.integers(0, len(_codes), (BOOT_DRAWS, len(_codes)))
        _num, _den = _S[_idx].sum(axis=1), _D[_idx].sum(axis=1)
        assert _num.shape == (BOOT_DRAWS, num.shape[1]), _num.shape
        if (_den == 0).any():
            raise RuntimeError("a bootstrap draw has no bridge behind a ratio")
        return _num / _den

    return CI, county_draws


@app.cell
def _(
    CI,
    MARKERS,
    PALETTE,
    R,
    RATIOS,
    caught_at_share,
    county_draws,
    expected_cost,
    np,
    p_test,
    plt,
    te,
    yt,
):
    _rating = te["rating"].to_numpy()
    _rows = {}
    for _r in RATIOS:
        _f = p_test > 1 / (1 + _r)
        _rows[_r] = (
            round(100 * _f.mean(), 1),
            round(100 * (_f & (yt == 1)).sum() / yt.sum(), 1),
            round(expected_cost(_f, yt, _r), 3),
            round(expected_cost(_rating == 5, yt, _r), 3),
        )
    _r56 = _rating <= 6
    _rule56 = (round(100 * _r56.mean(), 1), round(100 * (_r56 & (yt == 1)).sum() / yt.sum(), 1))
    _r5 = _rating == 5
    _rule5 = (round(100 * _r5.mean(), 1), round(100 * (_r5 & (yt == 1)).sum() / yt.sum(), 1))
    _same = round(100 * caught_at_share(p_test, yt, float(_r5.mean())), 1)
    # Dial slide and notes: r = 10, 18% and 64%, 0.300 against the rule's 0.302; r = 20, 31.5%
    # and 80%; r = 50, 49.5% and 89%, 0.68 against 0.84; every 5 or 6: 49% and 89%; at the
    # rule's budget (19.4% flagged) the model catches 66.3% and the rule 66.7%.
    assert _rows[10] == (18.1, 64.3, 0.3, 0.302) and _rows[20][:2] == (31.5, 79.6)
    assert _rows[50] == (49.5, 89.3, 0.675, 0.84) and _rule56 == (49.1, 89.2)
    assert _same == 66.3 and _rule5 == (19.4, 66.7)
    # Model minus rule, cost per bridge, with an interval that resamples whole counties: a tie
    # at r = 20; at r = 50 the model's lead holds.
    _ties = {}
    for _r in (R, max(RATIOS)):
        _f = p_test > 1 / (1 + _r)
        _cm = _r * (~_f & (yt == 1)) + (_f & (yt == 0))
        _cr = _r * (~_r5 & (yt == 1)) + (_r5 & (yt == 0))
        _d = (_cm - _cr).astype(np.float64)
        _draws = county_draws(_d[:, None], np.ones((len(yt), 1)), te["county"].to_numpy(), _r)
        _ties[_r] = (
            round(float(_d.mean()), 3),
            [round(float(_v), 3) for _v in np.percentile(_draws[:, 0], CI)],
        )
    # From the figure script's numbers.json (thresholds.model_minus_rule), copied by script.
    assert _ties == {20: (0.011, [-0.019, 0.042]), 50: (-0.164, [-0.265, -0.071])}, _ties
    print("model minus rule, cost per bridge (95% county interval):", _ties)

    _order = np.argsort(-p_test, kind="stable")
    _fig, _ax = plt.subplots(figsize=(5, 3.6))
    _ax.plot(
        100 * np.arange(len(yt) + 1) / len(yt),
        100 * np.concatenate([[0], np.cumsum(yt[_order])]) / yt.sum(),
        color=PALETTE[0],
        label="model",
    )
    for _k, _r in enumerate(RATIOS):
        _ax.plot(*_rows[_r][:2], MARKERS[1 + _k], color=PALETTE[1], ms=9, label=f"r = {_r}")
    _ax.plot(
        [_rule5[0], _rule56[0]],
        [_rule5[1], _rule56[1]],
        "D",
        mfc="none",
        color=PALETTE[2],
        ms=12,
        label="rules: every 5; every 5 or 6",
    )
    _ax.set_xlabel("bridges flagged (%)")
    _ax.set_ylabel("poor by 2024, caught (%)")
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(CI, INPUTS, R, Zt, county_draws, np, p_test, pd, te, w, yt):
    _flag = p_test > 1 / (1 + R)
    _owners = {"01": "state", "02": "county", "03": "township", "04": "city"}
    _masks = {_name: (te["owner"] == _code).to_numpy() for _code, _name in _owners.items()}
    _masks["other owners"] = ~te["owner"].isin(list(_owners)).to_numpy()
    _rows = []
    for _name, _m in _masks.items():
        _rows.append(
            {
                "owner": _name,
                "bridges": int(_m.sum()),
                "turned poor": int(yt[_m].sum()),
                "poor %": round(100 * yt[_m].mean(), 1),
                "mean p %": round(100 * p_test[_m].mean(), 1),
                "rated 5 %": round(100 * (te["rating"].to_numpy()[_m] == 5).mean(), 1),
                "flagged %": round(100 * _flag[_m].mean(), 1),
                "caught %": round(100 * (_flag[_m] & (yt[_m] == 1)).sum() / yt[_m].sum()),
                "missed": int((~_flag[_m] & (yt[_m] == 1)).sum()),
            }
        )
    _by = pd.DataFrame(_rows).set_index("owner")
    # Slide "Who carries a missed bridge?": 84% of state bridges, 75% of township bridges;
    # township bridges are 48% of the inventory and 86 of the 162 misses; the 544 bridges of
    # other owners (toll authority, parks, railroads, agencies): 4 missed of 9.
    assert _by["caught %"].tolist() == [84, 82, 75, 87, 56]
    assert _by["missed"].tolist() == [33, 26, 86, 13, 4] and _by["missed"].sum() == 162
    assert _by["missed"].sum() == int(((~_flag) & (yt == 1)).sum())
    assert _by["flagged %"].tolist() == [45.6, 27.2, 23.3, 39.6, 35.3]
    assert round(100 * _by.loc["township", "bridges"] / len(te)) == 48
    assert _by["poor %"].tolist() == [3.7, 5.0, 3.6, 7.9, 1.7]
    assert _by.loc["other owners", "bridges"] == 544
    # Why state bridges get a higher p than township bridges: the mean score gap, by input.
    _st, _tw = (te["owner"] == "01").to_numpy(), (te["owner"] == "03").to_numpy()
    _gap = (Zt[_st] * w[1:]).mean(axis=0) - (Zt[_tw] * w[1:]).mean(axis=0)
    _groups = {
        "ratings": ["deck", "superstructure", "substructure", "rating"],
        "material": ["concrete", "steel", "prestressed", "timber"],
        "traffic_and_size": ["log_adt", "pct_truck", "log_length", "deck_width_m"],
        "age_and_work": ["age", "years_since_work"],
        "location": ["lat", "lon"],
    }
    assert sorted(_c for _cols in _groups.values() for _c in _cols) == sorted(INPUTS)
    _by_group = {
        _g: round(float(sum(_gap[INPUTS.index(_c)] for _c in _cols)), 2)
        for _g, _cols in _groups.items()
    }
    # The catch gap, state minus township, with an interval that resamples whole counties.
    _y = yt == 1
    _hit = _flag & _y
    _num = np.column_stack([_hit & _st, _hit & _tw]).astype(np.float64)
    _den = np.column_stack([_y & _st, _y & _tw]).astype(np.float64)
    _draws = county_draws(_num, _den, te["county"].to_numpy(), 3)
    _caught = _num.sum(axis=0) / _den.sum(axis=0)
    _ci = [round(float(_v), 1) for _v in np.percentile(100 * (_draws[:, 0] - _draws[:, 1]), CI)]
    # From the figure script's numbers.json (by_owner, owner_gap), copied by script.
    assert _by["mean p %"].tolist() == [6.9, 4.5, 3.9, 6.5, 5.2]
    assert _by["rated 5 %"].tolist() == [25.6, 17.2, 16.3, 24.0, 14.2]
    assert round(float(_gap.sum()), 2) == 0.93 and _by_group == {
        "ratings": 0.7,
        "material": 0.19,
        "traffic_and_size": 0.02,
        "age_and_work": -0.01,
        "location": 0.04,
    }
    assert round(float(100 * (_caught[0] - _caught[1])), 1) == 9.1
    assert _ci == [-3.2, 19.7], _ci
    print(f"state minus township: score {_gap.sum():.2f} by group {_by_group}; caught gap")
    print(f"{100 * (_caught[0] - _caught[1]):.1f} points, 95% county interval {_ci}")
    _by
    return


@app.cell
def _(INPUTS, POOR, df, encode, predict_proba, te, w):
    # The bridges with no label: in the 2020 inventory, rated above poor, gone by 2024.
    _now = df[(df["nbi_year"] == 2020) & (df["rating"] > POOR)]
    _gone_mask = ~_now["structure"].isin(set(te["structure"]))
    _gone, _kept = _now[_gone_mask].copy(), _now[~_gone_mask]
    _gone["deck"] = _gone["deck"].fillna(_gone[["superstructure", "substructure"]].min(axis=1))
    assert _gone[INPUTS].notna().all().all(), "missing inputs"
    _pg = predict_proba(w, encode(_gone))
    dropped = {
        "dropped, of": (len(_gone), len(_now)),
        "rated 5 %: dropped, kept": (
            round(100 * (_gone["rating"] == 5).mean()),
            round(100 * (_kept["rating"] == 5).mean()),
        ),
        "dropped: mean p %, expected poor": (round(100 * _pg.mean()), round(float(_pg.sum()))),
    }
    # From the figure script's numbers.json (dropped_test), copied by script.
    assert dropped["dropped, of"] == (150, 19787)
    assert dropped["rated 5 %: dropped, kept"] == (65.0, 19.0)
    assert dropped["dropped: mean p %, expected poor"] == (15.0, 23)
    dropped
    return


@app.cell
def _(mo):
    mo.md(r"""
    **One threshold allocates attention unevenly.** State bridges get a higher $p$ than
    township bridges (mean 6.9% against 3.9%): their scores are higher by 0.93, mostly through
    the ratings (+0.70; 26% of state bridges are rated 5, 16% of township bridges) and material
    (+0.19); traffic and size add only 0.02. The model runs high for state bridges (6.9%
    predicted, 3.7% turned poor) and is close for townships (3.9% against 3.6%). The gap in the
    share caught, 84% against 75%, is 9.1 points, but resampling whole counties puts its 95%
    interval at −3.2 to +19.7 points: within county-to-county variation. Township owners still
    carry 86 of the 162 misses. $r$ prices risk to the people who cross the bridge: the owner
    sets it, in the open, not the modeler. Report misses by owner.

    **The bridges with no label are the riskier ones.** Of the 19,787 bridges of 2020 above
    poor, 150 left the inventory by 2024; 65% of them were rated 5, against 19% of the rest,
    and the model gives them a mean $p$ of 15%, about 23 expected poor beside the 793 counted.
    The test costs leave them out (informative censoring).

    **When not to use it.** Not instead of inspection: the 2022 National Bridge Inspection
    Standards require routine inspections at most every 24 months by default, every 12 months
    when a component is rated 3 or less, and allow up to 48 months only when every component is
    rated 6 or more and other criteria hold (23 CFR 650.311). Not with 0.5 or accuracy for rare
    events. Not where the rule does as well. Not without rechecking calibration each new period,
    and not for bridges unlike these (culverts, new materials, other states). Not for one bridge's
    fate: $p = 0.18$ means about one in five bridges like it.

    ### Vote: what do you recommend?

    The county will act on your list. A miss costs 20 false alarms. **A.** Flag $p > 0.5$, the
    highest accuracy. **B.** Flag every bridge rated 5: no model. **C.** Flag $p > 1/21$, the
    cost threshold. **D.** Flag every bridge.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                2024 costs: B 0.44, C 0.45, A 0.81, D 0.96. B is as good here, and anyone can
                audit it; C adapts to $r$ and states a probability per bridge, but its
                calibration needs rechecking each period; A is the accuracy trap; D spends everything. What would change
                the answer: $r = 50$ (B costs 0.84, C 0.68), a cap on the budget, or a decade
                with drift.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where logistic regression lives today

    - **The last layer.** A classifier network ends in a softmax regression trained on
      cross-entropy (Forsyth §11.3.2); a language model's next token is a softmax over its
      vocabulary, trained with today's loss on tens of thousands of classes (Lectures 17, 19).
    - **CLIP** trains with a cross-entropy over the image–text pairs in a batch (Radford et al.,
      2021); Lecture 18.
    - **Reward models for RLHF** score pairs of answers that people ranked, with
      $P(A \succ B) = \sigma(r_A - r_B)$, the Bradley–Terry form of logistic regression
      (Ouyang et al., 2022); Lecture 26.
    - **Calibration.** Large networks are overconfident; temperature scaling divides the scores
      by one number fitted on validation data, Platt scaling with one parameter (Guo et al.,
      2017). No later lecture covers it; Lecture 10 checks calibration.

    The decoder and the loss you derived today are inside the largest models, and so is today's
    question: what does a probability cost?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question.** The owner raises $r$ from 20 to 50. What changes: the threshold, the
    list, and the rule the model matches? What would you check before using the model in 2026?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                The threshold falls to $1/51 = 0.020$; the list grows from 31.5% to 49.5% of the
                bridges and catches 89% instead of 80% of those that turn poor; it matches the
                rule "flag every 5 or 6" (49% flagged, 89% caught). Before 2026: check the
                calibration on the latest four-year period, 2020 → 2024, and recalibrate or
                refit.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **`predict()` thresholds at 0.5.** For a rare event with unequal costs, use
      `predict_proba` and $1/(1+r)$, or a budget.
    - **Accuracy on imbalanced labels.** 95.96% was the score of flagging nobody.
    - **Comparing solvers without stating the tolerance.** scikit-learn's default L-BFGS stopped
      0.4 from the optimum on a nearly flat direction; compare losses and probabilities, not
      only weights.
    - **`penalty=None` is deprecated in scikit-learn 1.8:** write `C=np.inf`.
    - **Treating SVM scores as probabilities.** Calibrate them (Platt) on held-out data first.
    - **`LinearSVC()` is not the textbook SVM.** Its defaults minimize the squared hinge and
      penalize the intercept (liblinear treats $w_0$ as the weight of a constant input,
      `intercept_scaling=1`); set `loss=` explicitly and say which one the slide shows. On
      these bridges `loss="hinge"` collapses: the fit puts every bridge at $s \approx -1$
      with $\|\mathbf{w}\| < 10^{-4}$, and what ranking is left depends on where the solver
      stopped: test AUC 0.46 with the default intercept handling, 0.78 with
      `intercept_scaling=100` (checked below).
    - **Trusting calibration to hold.** The 2024 probabilities ran high, and nothing seen by
      2021 showed it: recheck calibration as each new period arrives.
    - **Using the current inspection as an input to predict the same inspection.** Today's
      ratings come from the earlier record; Lecture 3's leak came from the record being
      predicted.
    """)
    return


@app.cell
def _(LinearSVC, SEED, Z, Zt, np, roc_auc_score, y, yt):
    # Gotcha check: the plain hinge, with liblinear's penalized intercept at two scalings.
    for _scaling, _auc in ((1.0, 0.46), (100.0, 0.78)):
        _hinge = LinearSVC(
            C=1.0,
            loss="hinge",
            penalty="l2",
            dual=True,
            max_iter=100_000,
            intercept_scaling=_scaling,
            random_state=SEED,
        ).fit(Z, y)
        _s = _hinge.decision_function(Zt)
        assert np.linalg.norm(_hinge.coef_) < 1e-4 and np.abs(_s + 1).max() < 1e-4
        assert round(roc_auc_score(yt, _s), 2) == _auc
        print(
            f"hinge, intercept_scaling = {_scaling:g}: |w| = {np.linalg.norm(_hinge.coef_):.1e}, "
            f"test AUC {roc_auc_score(yt, _s):.2f}"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §2.1.1–2.1.2:** the hinge loss and the regularized SVM objective; Forsyth
      writes $\mathbf{a}$ for our $\mathbf{w}$ and $b$ for $w_0$, and $\lambda$ for the penalty.
    - **Forsyth §2.1.3:** training the SVM with stochastic gradient descent, the route
      Lecture 13 takes at scale.
    - **Forsyth §11.3.1:** logistic regression as a generalized linear model, and its
      comparison with the hinge loss.
    - **Forsyth §11.3.2:** multiclass logistic regression (softmax), the last layer of every
      network classifier.
    - **Exercise.** Derive the Hessian of the cross-entropy and show it is positive
      semidefinite; then show that with perfectly separable data the likelihood has no maximum.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Model the log-odds; the likelihood gives cross-entropy; Newton steps are least squares.
    - Flag when $p > 1/(1+r)$, not 0.5. Accuracy hides misses.
    - The cost threshold needs calibrated probabilities: check the latest period.
    - Beat the engineer's rule: here the rating did most of the work.
    - SVMs give margins, not odds; ordinal models assume one weight for every cut.

    **Next:** naive Bayes, mixtures, and EM: free-flow and congested traffic.
    """)
    return


if __name__ == "__main__":
    app.run()
