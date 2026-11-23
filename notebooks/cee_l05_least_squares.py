"""Applied Machine Learning for CEE — Lecture 5: Least Squares and Regularization.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l05_least_squares.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_concrete.sh`` once (UCI Concrete Compressive Strength).
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
    # Applied Machine Learning for CEE — Lecture 5: Least Squares and Regularization

    **Slides:** *Applied Machine Learning for CEE*, Lecture 5
    (`lectures/cee_l05_least_squares`, PowerPoint).<br>
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning*
    (University of Illinois Urbana-Champaign, Fall 2026).<br>
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §10.2 (least squares),
    §10.4.2 (regularizing linear regressions), §11.4 (L1 regularization and sparse models).<br>
    **Data:** UCI Concrete Compressive Strength (Yeh, 1998), 1,030 laboratory tests; fetch it
    with `bash scripts/fetch_concrete.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    It uses the slides' data, splits, and seeds, and every number on the slides is recomputed
    here and checked with an `assert`.

    **The in-class exercise** (*Derive, implement, delegate: how strong is this concrete?*) runs
    through the lecture as a group worksheet; its five parts are marked **Worksheet** below.
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
    from scipy.linalg import solve_triangular
    from sklearn.linear_model import Lasso, LinearRegression, Ridge, lasso_path
    from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, ShuffleSplit
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)  # child processes only: this kernel's is fixed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    K = 5  # neighbors of the Lecture 2 reference model
    REPEATS = 30  # repeated 75/25 splits for every comparison, as on the slides
    FC = 35.0  # specified 28-day strength f'c, MPa
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        FC,
        GridSearchCV,
        GroupKFold,
        GroupShuffleSplit,
        K,
        KNeighborsRegressor,
        Lasso,
        LinearRegression,
        MARKERS,
        PALETTE,
        Path,
        REPEATS,
        Ridge,
        SEED,
        ShuffleSplit,
        StandardScaler,
        lasso_path,
        np,
        pd,
        plt,
        make_pipeline,
        solve_triangular,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 2–4

    On your own, no notes, two minutes:

    1. 5-NN on raw kg/m³ chose a mix with $w/b = 0.30$ as a neighbor of one with $w/b = 0.40$.
       Why, and what fixed it?
    2. Bridges in one county share climate, traffic, and inspectors. Why does a random split of
       bridge records overstate a model's accuracy? How would you split instead?
    3. Stack buildings' daily load profiles as rows and center the matrix. What do its right
       singular vectors and its singular values tell you?

    **Today:** how much cement does a 35 MPa mix need?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. In raw units the squared distance is dominated by the inputs with the largest
                   variance, cement and the aggregates; water carried 1.2% of it. The Lecture 2
                   encoder ($w/b$, binder fractions, $\ln$ age) cut the 5-NN error from 10.0 to
                   6.9 MPa (recomputed in §5).
                2. Bridges near each other are alike, so a random split puts near-copies of each
                   test bridge in training. Hold out whole spatial blocks, such as counties.
                3. The right singular vectors are the principal directions: the shapes in which
                   daily profiles differ most from the average day. The singular values give the
                   spread along each; $\sigma_i^2/\sum_j\sigma_j^2$ is the share of variance.
                   Today the SVD returns, as a way to solve least squares.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A producer designs a slab mix for $f'_c = 35$ MPa at 28 days with 185 kg/m³ of water, fixed
    by workability. **How much cement?** Too little puts weak concrete in place, followed by
    cores, repair, or removal; the owner and the public carry that risk. Too much costs money and
    carbon in every cubic meter (Lecture 2). Lecture 2 asked whether one proposed mix would pass;
    today we *design* the mix, so we need a model we can solve for an input. A weighted sum of
    features, fitted by least squares, is the simplest one.

    **The data** are Lecture 2's: 1,030 tests, seven ingredients in kg/m³, age in days, strength
    in MPa, with the same `mix` and `design` ids and the same 30 design-held-out splits.

    ### The in-class exercise: derive, implement, delegate (25 minutes)

    Groups of three, one laptop per group. The five parts sit where they belong in the lecture:

    | Part | Minutes | How | Section |
    |:--|--:|:--|:--|
    | 1. Predict | 3 | alone, then pairs | §1 |
    | 2. Derive | 5 | groups of three | §3 |
    | 3. Implement | 7 | one laptop per group | §4 |
    | 4. Delegate | 7 | groups of three | §6 |
    | 5. Debrief | 3 | alone, on paper | §7 |

    Write each answer in the box under its part **before** opening the solution.
    """)
    return


@app.cell
def _(Path, np, pd):
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
    INGREDIENTS = COLUMNS[:7]

    def find_concrete_csv() -> Path:
        """Locate ``data/derived/Concrete_Data.csv`` from the repo root, notebooks/, or ipynb/.

        Returns:
            Path of the CSV.

        Raises:
            FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_concrete.sh``).
        """
        rel = Path("data") / "derived" / "Concrete_Data.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_concrete.sh`")

    def design_ids(df: pd.DataFrame, tol: float) -> np.ndarray:
        """Merge mixes whose seven proportions all agree within ``tol`` kg/m³ (union-find).

        Args:
            df: Concrete data with a ``mix`` column.
            tol: Largest difference, in kg/m³, between two mixes of one design.

        Returns:
            Design id for every row.
        """
        mixes = df.groupby("mix")[INGREDIENTS].first().to_numpy()
        parent = np.arange(len(mixes))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = int(parent[a])
            return a

        close = np.abs(mixes[:, None, :] - mixes[None, :, :]).max(axis=-1) <= tol
        for a, b in np.argwhere(np.triu(close, 1)):
            parent[find(int(a))] = find(int(b))
        roots = np.array([find(i) for i in range(len(mixes))])
        _, design_of_mix = np.unique(roots, return_inverse=True)
        return design_of_mix[df["mix"].to_numpy()]

    def load_concrete(path: Path, tol: float = 1.0) -> pd.DataFrame:
        """Load the concrete tests with ``mix`` and ``design`` ids (same as the slides' loader).

        Args:
            path: The CSV written by ``scripts/fetch_concrete.sh``.
            tol: Design tolerance in kg/m³.

        Returns:
            1,030 rows: the nine data columns plus ``mix`` and ``design``.
        """
        frame = pd.read_csv(path)
        assert frame.shape == (1030, 9), frame.shape
        frame.columns = COLUMNS
        frame["mix"] = frame.groupby(INGREDIENTS, sort=False).ngroup()
        frame["design"] = design_ids(frame, tol)
        return frame

    return INGREDIENTS, find_concrete_csv, load_concrete


@app.cell
def _(find_concrete_csv, load_concrete):
    df = load_concrete(find_concrete_csv())
    y_all = df["strength"].to_numpy()
    # Slide "Set the cement for 35 MPa": 1,030 tests, 425 at 28 days (315 designs).
    assert (len(df), int((df["age"] == 28).sum()), df["design"].nunique()) == (1030, 425, 315)
    return df, y_all


@app.cell
def _(mo):
    mo.md(r"""
    **Helpers used in every section:** the three encoders (raw inputs, raw inputs with
    $\ln$ age, and Lecture 2's dimensionless groups), the two ways of splitting, and a few
    one-liners. Skip them on a first read.
    """)
    return


@app.cell
def _(GroupShuffleSplit, INGREDIENTS, REPEATS, SEED, ShuffleSplit, np, pd):
    RAW_NAMES = [*INGREDIENTS, "age"]
    GROUP_NAMES = ["w/b", "slag / b", "fly ash / b", "SP / b", "aggregate / b", "ln age"]

    def raw_features(df: pd.DataFrame) -> np.ndarray:
        """Seven ingredients (kg/m³) and age (days), as recorded."""
        return df[RAW_NAMES].to_numpy(dtype=np.float64)

    def raw_log_age(df: pd.DataFrame) -> np.ndarray:
        """Seven ingredients (kg/m³) and the natural log of age."""
        return np.column_stack([df[INGREDIENTS].to_numpy(np.float64), np.log(df["age"])])

    def mix_groups(df: pd.DataFrame) -> np.ndarray:
        """Lecture 2's encoder: w/b, slag/b, fly ash/b, SP/b, aggregate/b, ln age.

        Args:
            df: Concrete data with the ingredient columns and ``age``.

        Returns:
            Array of shape ``(len(df), 6)``; binder b = cement + slag + fly ash.
        """
        b = df.cement + df.slag + df.fly_ash
        z = np.column_stack(
            [
                df.water / b,
                df.slag / b,
                df.fly_ash / b,
                df.superplasticizer / b,
                (df.coarse_agg + df.fine_agg) / b,
                np.log(df.age),
            ]
        )
        assert z.shape == (len(df), 6), z.shape
        return z

    def grouped_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """``REPEATS`` 75/25 splits that hold out whole designs (the slides' splits)."""
        gss = GroupShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(gss.split(df, groups=df["design"]))

    def random_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """``REPEATS`` 75/25 splits of the tests, ignoring designs."""
        ss = ShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(ss.split(df))

    def predict(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Predictions of weights ``[w0, w]`` (intercept first) on features ``Z``."""
        return w[0] + Z @ w[1:]

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square difference."""
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return (
        GROUP_NAMES,
        RAW_NAMES,
        grouped_splits,
        mix_groups,
        predict,
        random_splits,
        raw_features,
        raw_log_age,
        rmse,
    )


@app.cell
def _(df, grouped_splits, random_splits):
    splits = grouped_splits(df)
    rsplits = random_splits(df)
    tr0, te0 = splits[0]
    # The first design-held-out split: 746 training tests of 236 designs, 284 held out.
    assert (len(tr0), df["design"].iloc[tr0].nunique(), len(te0)) == (746, 236, 284)
    return rsplits, splits, te0, tr0


@app.cell
def _(mo):
    mo.md(r"""
    ### Worksheet, part 1: Predict (3 minutes, alone, then pairs)

    Four tested mixes from one laboratory series: 185 kg/m³ of water, no slag, fly ash, or
    superplasticizer; only the cement changes. All tested at 28 days. P is the proposed mix.

    1. In a model of strength from the ingredients, what sign will **water's** weight have? Why?
    2. Estimate P's 28-day strength. Does it reach 35 MPa?

    *Your answers:* 1. ________ 2. ________ MPa
    """)
    return


@app.cell
def _(df, np, pd, te0, tr0):
    ATTEMPT_ROWS = {"A": 825, "B": 802, "C": 779, "D": 801}
    P_ROW = 638
    _rows = list(ATTEMPT_ROWS.values())
    _sub = df.loc[_rows]
    # A–D: training tests of the first split, 28 days, water 185, no admixtures. P: held out.
    assert set(_rows) <= set(tr0) and P_ROW in set(te0)
    assert (_sub["age"] == 28).all() and np.allclose(_sub["water"], 185.0)
    assert (_sub[["slag", "fly_ash", "superplasticizer"]] == 0).all().all()
    assert (
        df.at[P_ROW, "age"] == 28
        and (df.loc[P_ROW, ["slag", "fly_ash", "superplasticizer"]] == 0).all()
    )
    _all = df.loc[[*_rows, P_ROW]]
    attempt_table = pd.DataFrame(
        {
            "cement (kg/m³)": _all["cement"].round(0).astype(int).to_numpy(),
            "water (kg/m³)": _all["water"].round(0).astype(int).to_numpy(),
            "w/c": (_all["water"] / _all["cement"]).round(2).to_numpy(),
            "MPa at 28 days": [f"{_v:.1f}" for _v in _sub["strength"]] + ["?"],
        },
        index=pd.Index([*ATTEMPT_ROWS, "P"], name="mix"),
    )
    # Slide "Your turn: predict the water effect".
    assert list(attempt_table["cement (kg/m³)"]) == [397, 339, 295, 252, 375]
    assert list(attempt_table["w/c"]) == [0.47, 0.55, 0.63, 0.73, 0.50]
    assert list(attempt_table["MPa at 28 days"][:4]) == ["39.1", "31.6", "25.2", "19.7"]
    attempt_table
    return ATTEMPT_ROWS, P_ROW


@app.cell
def _(mo):
    mo.accordion(
        {
            "Solution, part 1 (open after you commit)": mo.md(
                r"""
                1. **Negative.** More water at the same binder raises $w/c$, and Abrams' law
                   (1918) says strength falls exponentially as $w/c$ rises. Water is constant in
                   this table, so the sign comes from physics, not from these rows; §4 confirms
                   it in the fitted model.
                2. Between A (0.47, 39.1 MPa) and B (0.55, 31.6 MPa), straight-line interpolation
                   at $w/c = 0.50$ gives **36.3 MPa**, just above 35. Measured: **38.2 MPa**. §3
                   fits Abrams' law to A–D by hand; §4 traces P through the full model.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture from Lecture 1, with today's step highlighted:

    > mix $\mathbf{x}$ → **encoder** (Lecture 2's dimensionless groups) → $\mathbf{z}$ →
    > **decoder** (weighted sum $w_0 + \mathbf{w}^\top\mathbf{z}$) → $\hat{y}$ →
    > loss $(\hat{y}-y)^2$

    The encoder is Lecture 2's; the decoder is new. k-NN stored 746 tests; least squares fits
    7 numbers once (QR, §3–4) and then forgets the training set. Each prediction is a sum of
    terms $w_j z_j$, so you can read where it comes from. For P (cement 375, water 186, coarse
    1,038, fine 758 kg/m³, 28 days): $w_0 = 54.0$, the $w/b$ term $-48.0$, the aggregate term
    $+6.0$, the age term $+27.8$, sum **39.8 MPa**; measured 38.2 (error $+1.6$). §4 recomputes
    every term.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** strength is a weighted sum of the features, plus noise that is independent
    from test to test and has the same spread for every mix. If that noise is Gaussian,
    maximizing the likelihood is exactly least squares (Forsyth §10.2.2).

    **The model.** Stack the $N$ training tests as rows of $X \in \mathbb{R}^{N\times D}$, with
    a first column of ones for the intercept $w_0$, and the targets in $\mathbf{y}$. The model is
    *linear in the weights*, not in the inputs: Abrams' law $f_c = A/B^{\,w/c}$ becomes
    $\ln f_c = a + b\,(w/c)$ with $A = e^{a}$, $B = e^{-b}$, a weighted sum of the features
    $(1, w/c)$.

    ### Worksheet, part 2: Derive (5 minutes, groups of three)

    Write $\mathcal{L}(\mathbf{w}) = \lVert X\mathbf{w}-\mathbf{y}\rVert^2$ and set its gradient
    to zero. Why do we solve with QR or the SVD rather than forming $(X^\top X)^{-1}$?

    *Your derivation:* ________
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Solution, part 2": mo.md(
                r"""
                $$\mathcal{L}(\mathbf{w})=(X\mathbf{w}-\mathbf{y})^\top(X\mathbf{w}-\mathbf{y}),
                \qquad \nabla_{\mathbf{w}}\mathcal{L}=2X^\top(X\mathbf{w}-\mathbf{y})=\mathbf{0}
                \;\Longrightarrow\; \boxed{X^\top X\,\mathbf{w}=X^\top\mathbf{y}}$$

                the **normal equations**. The Hessian $2X^\top X$ is positive semidefinite, so
                $\mathcal{L}$ is convex and the solution is a global minimum. Geometrically, the
                residual $\mathbf{y}-X\mathbf{w}$ is orthogonal to every column of $X$: the fit is
                the projection of $\mathbf{y}$ onto the column space.

                **Why not the inverse.** (1) Forming $X^\top X$ squares the condition number,
                $\kappa(X^\top X)=\kappa(X)^2$, so a solve can lose twice as many digits. (2) When
                a column depends on the others, $X^\top X$ is singular and the inverse does not
                exist. QR and the SVD work on $X$ itself and are backward stable; the SVD also
                handles dependent columns (Trefethen & Bau, 1997, Parts II–III: QR
                factorization and least squares; conditioning and stability).
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked example: Abrams' law on the four mixes, by hand

    Use the values as the table shows them: $x = w/c$ to two decimals, $y = \ln f_c$ with $f_c$
    to one decimal. Center $x$: $\Delta x = x - \bar{x}$. Then the columns $(1, \Delta x)$ are
    orthogonal, $X^\top X$ is diagonal, and the normal equations decouple:
    $$b=\frac{\sum\Delta x\,\Delta y}{\sum\Delta x^2},\qquad a=\bar{y}-b\,\bar{x}.$$
    Centering did by hand what QR does for any number of columns. Compute $b$, $a$, $A$, $B$,
    the prediction for P, and the $w/c$ (and cement, at 185 kg/m³ of water) for 35 MPa; then run
    the cell.
    """)
    return


@app.cell
def _(ATTEMPT_ROWS, FC, MARKERS, PALETTE, P_ROW, df, np, plt):
    _sub = df.loc[list(ATTEMPT_ROWS.values())]
    _x = np.round((_sub["water"] / _sub["cement"]).to_numpy(), 2)
    _mpa = np.round(_sub["strength"].to_numpy(), 1)
    _y = np.log(_mpa)
    _dx, _dy = _x - _x.mean(), _y - _y.mean()
    _b = float(_dx @ _dy / (_dx @ _dx))
    _a = float(_y.mean() - _b * _x.mean())
    # The same answer from the full normal equations (tiny and well conditioned: fine here).
    _X = np.column_stack([np.ones(4), _x])
    assert _X.shape == (4, 2)
    assert np.allclose(np.linalg.solve(_X.T @ _X, _X.T @ _y), [_a, _b], rtol=0, atol=1e-12)
    _A, _B = np.exp(_a), np.exp(-_b)
    _wc_p = round(float(df.at[P_ROW, "water"] / df.at[P_ROW, "cement"]), 2)
    _f_p = float(np.exp(_a + _b * _wc_p))
    _wc35 = float((np.log(FC) - _a) / _b)
    # Slide "Four mixes, solved by hand" and its notes.
    assert (round(float(_x.mean()), 3), round(float(_y.mean()), 3)) == (0.595, 3.332)
    assert (round(float(_dx @ _dy), 4), round(float(_dx @ _dx), 4)) == (-0.0983, 0.0371)
    assert (round(_b, 2), round(_a, 2), round(_A, 1), round(_B, 1)) == (-2.65, 4.91, 135.5, 14.2)
    assert (round(_f_p, 1), round(_wc35, 2), round(185.0 / _wc35)) == (36.0, 0.51, 362)
    # Notes: 185 / 0.511 = 362, from the three-decimal w/c (the shown a and b give it too).
    assert round(_wc35, 3) == 0.511 and round((np.log(FC) - 4.91) / -2.65, 3) == 0.511
    assert round(185.0 / 0.511) == 362
    assert round(float(np.abs(_y - _X @ [_a, _b]).max()), 3) == 0.012
    assert round(float(np.interp(_wc_p, _x[:2], _mpa[:2])), 1) == 36.3  # the attempt

    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    _grid = np.linspace(0.42, 0.78, 50)
    _ax.plot(
        _grid,
        np.exp(_a + _b * _grid),
        color="#1A1A1A",
        lw=1.5,
        label=f"ln f = {_a:.2f} {_b:+.2f} (w/c)",
    )
    for _j, (_l, _xi, _fi) in enumerate(zip(ATTEMPT_ROWS, _x, _mpa, strict=True)):
        _ax.plot(_xi, _fi, marker=MARKERS[_j], ms=9, ls="none", color=PALETTE[_j])
        _ax.annotate(f"{_l}: {_fi:.1f}", (_xi, _fi), xytext=(8, 2), textcoords="offset points")
    _ax.plot(_wc_p, _f_p, "o", ms=9, mfc="white", mec="#1A1A1A")
    _ax.annotate(f"P: {_f_p:.1f}", (_wc_p, _f_p), xytext=(10, 4), textcoords="offset points")
    _ax.axhline(FC, ls="--", color="#595959", lw=1)
    _ax.set(yscale="log", xlabel="w/c", ylabel="28-day strength (MPa, log scale)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ATTEMPT_ROWS, StandardScaler, df, lasso_path, np, plt):
    # The loss as a surface (slides "The fit sits at the bottom of a bowl", "A copy of a column
    # flattens the bowl", "Ridge shrinks the shaky directions", "Set the cement for 35 MPa").
    _sub = df.loc[list(ATTEMPT_ROWS.values())]
    _x = np.round((_sub["water"] / _sub["cement"]).to_numpy(), 2)
    _conds = []
    for _cols in (np.column_stack([np.ones(4), _x]), np.column_stack([np.ones(4), _x - _x.mean()])):
        _ev = np.linalg.eigvalsh(_cols.T @ _cols)  # the bowl's curvatures
        _conds.append(round(float(_ev.max() / _ev.min())))
    assert _conds == [198, 108]  # raw: a tilted valley; centred: along the axes

    # A column entered twice: every (w1, w2) with w1 + w2 = b fits; ridge picks one point.
    _d28 = df[df["age"] == 28]
    _wb = (_d28["water"] / (_d28["cement"] + _d28["slag"] + _d28["fly_ash"])).to_numpy()
    _z = (_wb - _wb.mean()) / _wb.std()
    _y = np.log(_d28["strength"].to_numpy())
    _y = _y - _y.mean()
    _X2 = np.column_stack([_z, _z])
    assert np.linalg.matrix_rank(_X2) == 1
    _w_ridge = np.linalg.solve(_X2.T @ _X2 + len(_y) * np.eye(2), _X2.T @ _y)
    assert round(float(_z @ _y / (_z @ _z)), 3) == -0.329
    # The minimum-norm answer: each weight b / 2, rounded once from the raw slope.
    assert round(float(_z @ _y / (_z @ _z)) / 2, 3) == -0.164
    assert [round(float(_v), 2) for _v in _w_ridge] == [-0.11, -0.11]

    # Ridge against lasso on two correlated inputs, w/b and binder (all tests, standardized).
    _binder = (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()
    _Z = StandardScaler().fit_transform(
        np.column_stack([df["water"].to_numpy() / _binder, _binder])
    )
    _yy = df["strength"].to_numpy() - df["strength"].mean()
    _w_ls = np.linalg.lstsq(_Z, _yy, rcond=None)[0]
    _, _coefs, _ = lasso_path(_Z, _yy, n_alphas=200)
    _first = int(np.argmax(np.abs(_coefs).sum(axis=0) > 0))
    assert round(float(np.corrcoef(_Z.T)[0, 1]), 2) == -0.89
    assert [round(float(_v), 2) for _v in _w_ls] == [-6.21, 4.72]
    assert int(np.argmax(np.abs(_coefs[:, _first]))) == 0  # lasso admits w/b first

    _band = (df["age"] == 28) & df["water"].between(175, 195) & (df["slag"] == 0)
    _band = _band & (df["fly_ash"] == 0)
    assert int(_band.sum()) == 58
    # Its alt text: cement 200 to 528 kg/m³, strength 12.2 to 56.8 MPa.
    assert (
        round(float(df.loc[_band, "cement"].min())),
        round(float(df.loc[_band, "cement"].max())),
    ) == (
        200,
        528,
    )
    assert (
        round(float(df.loc[_band, "strength"].min()), 1),
        round(float(df.loc[_band, "strength"].max()), 1),
    ) == (12.2, 56.8)

    # The trough in 3D: rotate the view (view_init) to see its flat floor.
    _a = np.linspace(-0.6, 0.3, 61)
    _A, _B = np.meshgrid(_a, _a)
    _L = ((_X2 @ np.stack([_A.ravel(), _B.ravel()]) - _y[:, None]) ** 2).mean(axis=0)
    _fig = plt.figure(figsize=(6, 4.5))
    _ax = _fig.add_subplot(projection="3d")
    _ax.plot_surface(_A, _B, _L.reshape(_A.shape), cmap="cividis")
    _ax.set(xlabel="weight 1", ylabel="weight 2", title="w/b entered twice: a trough")
    _ax.view_init(elev=30, azim=-35)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Result.** $b = -0.0983/0.0371 = -2.65$ and $a = 3.332 + 2.65 \times 0.595 = 4.91$, so
    $A = 135.5$ MPa and $B = 14.2$ (Lecture 2's fit to all 425 tests at 28 days: 123.8 and
    16.5). P: **36.0 MPa** predicted, 38.2 measured. 35 MPa needs
    $w/c \le (\ln 35 - 4.91)/(-2.65) = 0.511$, that is $185/0.511 =$ **362 kg/m³** of cement
    at 185 kg/m³ of water. The largest residual in $\ln f$ is 0.012: on
    one laboratory's series, Abrams' law is nearly exact.

    ### Why QR or the SVD, measured

    The cell computes the condition numbers the slide quotes, for the design matrix with an
    intercept column, in raw kg/m³ and in Lecture 2's groups.
    """)
    return


@app.cell
def _(mix_groups, np, raw_features, df, tr0):
    _out = {}
    for _name, _enc in (("raw kg/m³ and days", raw_features), ("Lecture 2 groups", mix_groups)):
        _A = np.column_stack([np.ones(len(tr0)), _enc(df)[tr0]])
        _out[_name] = (float(np.linalg.cond(_A)), float(np.linalg.cond(_A.T @ _A)))
        print(
            f"{_name:>20s}: kappa(X) = {_out[_name][0]:9.3g}   kappa(X^T X) = {_out[_name][1]:9.3g}"
        )
    # Slide "Solve with QR or SVD, not the inverse" and its notes.
    assert [float(f"{_v:.2g}") for _v in _out["raw kg/m³ and days"]] == [1.0e5, 1.0e10]
    assert [float(f"{_v:.2g}") for _v in _out["Lecture 2 groups"]] == [640.0, 4.0e5]
    assert np.isclose(_out["raw kg/m³ and days"][1], _out["raw kg/m³ and days"][0] ** 2, rtol=1e-3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **QR.** $X = QR$ with $Q^\top Q = I$ and $R$ upper triangular. Substituting into the normal
    equations, $R^\top Q^\top Q R\,\mathbf{w} = R^\top Q^\top\mathbf{y}$, so
    $R\mathbf{w} = Q^\top\mathbf{y}$: one triangular solve, with $\kappa(R) = \kappa(X)$.

    **SVD.** $X = U\Sigma V^\top$ gives $\mathbf{w} = V\Sigma^{+}U^\top\mathbf{y}$, where
    $\Sigma^{+}$ inverts only the singular values above a numerical threshold. With a dependent
    column some $\sigma_i = 0$; $\Sigma^{+}$ skips it and returns the **minimum-norm** solution.
    `np.linalg.lstsq` and scikit-learn's `LinearRegression` work this way.

    ### Ridge: penalize large weights

    $$\mathcal{L}_\lambda(\mathbf{w})=\lVert X\mathbf{w}-\mathbf{y}\rVert^2
    +\lambda\lVert\mathbf{w}\rVert^2
    \;\Longrightarrow\; 2X^\top(X\mathbf{w}-\mathbf{y})+2\lambda\mathbf{w}=\mathbf{0}
    \;\Longrightarrow\; \mathbf{w}_\lambda=(X^\top X+\lambda I)^{-1}X^\top\mathbf{y}.$$

    Adding $\lambda I$ lifts every eigenvalue of $X^\top X$ by $\lambda$, so the solution is
    unique for every $\lambda > 0$, even with a dependent column (Forsyth, Exercise 10.7). With
    $X = U\Sigma V^\top$,
    $$\mathbf{w}_\lambda=\sum_i \mathbf{v}_i\,\frac{\sigma_i}{\sigma_i^2+\lambda}\,
    \mathbf{u}_i^\top\mathbf{y},
    \qquad \hat{\mathbf{y}}_\lambda=\sum_i\frac{\sigma_i^2}{\sigma_i^2+\lambda}\,
    \mathbf{u}_i\mathbf{u}_i^\top\mathbf{y}:$$
    each singular direction shrinks by $\sigma_i^2/(\sigma_i^2+\lambda)$, so the directions the
    data barely span shrink most (Hoerl & Kennard, 1970). With a Gaussian prior
    $\mathbf{w}\sim\mathcal{N}(0,\tau^2 I)$ and noise variance $\sigma^2$, ridge is the
    posterior mode with $\lambda=\sigma^2/\tau^2$.

    Two rules: **standardize first**, because the penalty treats every weight alike and a
    weight's size depends on its feature's units (§5 shows it); and **leave the intercept out of
    the penalty**: center $X$ and $\mathbf{y}$, fit, then set
    $w_0 = \bar{y} - \bar{\mathbf{x}}^\top\mathbf{w}$.

    ### Lasso: the $L_1$ penalty sets weights to zero

    $\lVert X\mathbf{w}-\mathbf{y}\rVert^2+\lambda\lVert\mathbf{w}\rVert_1$ is convex but not
    differentiable at zero, so there is no closed form; coordinate descent solves it. With
    orthonormal columns ($X^\top X = I$) and $z_j=\mathbf{x}_j^\top\mathbf{y}$ the least-squares
    weight, the problem decouples: ridge gives $w_j = z_j/(1+\lambda)$ (every weight divided by
    the same factor), lasso gives
    $w_j=\operatorname{sign}(z_j)\max(\lvert z_j\rvert-\lambda/2,\,0)$ (a constant subtracted,
    then clipped at zero). That clipping is why lasso selects features (Tibshirani, 1996).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### Worksheet, part 3: Implement (7 minutes, one laptop per group)

    Fit the six-group model to the 746 training tests of the first split in about ten lines of
    NumPy, then compare with scikit-learn's `LinearRegression` using `np.allclose` at a
    tolerance you state. *The check is part of the answer.* Skeleton:

    ```python
    def fit_least_squares(Z, y):
        A = ...            # add a column of ones
        Q, R = ...         # QR factorization
        return ...         # solve R w = Q^T y
    ```

    **Predict first**, and write it down:

    1. Your QR solution against scikit-learn at `rtol=1e-8`: pass or fail?
    2. The textbook formula, `inv(X.T @ X) @ X.T @ y`, on raw kg/m³, where
       $\kappa(X^\top X) = 10^{10}$: pass or fail?

    *Your predictions:* 1. ________ 2. ________

    The next cells are the solution: the slide's listing, an SVD solver, and ridge.
    """)
    return


@app.cell
def _(np, solve_triangular):
    def fit_least_squares(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Weights [w0, w] minimizing ||[1 Z] w - y||^2, via QR (the slide's listing).

        Args:
            Z: Features, shape ``(N, D)``.
            y: Targets, shape ``(N,)``.

        Returns:
            Intercept followed by the ``D`` weights, shape ``(D + 1,)``.
        """
        A = np.column_stack([np.ones(len(Z)), Z])  # intercept column
        Q, R = np.linalg.qr(A)  # A = QR, Q^T Q = I
        assert Q.shape == A.shape and R.shape == (A.shape[1],) * 2
        return solve_triangular(R, Q.T @ y)  # R w = Q^T y

    def fit_svd(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Weights [w0, w] from the SVD of the centered features (minimum norm if rank-deficient).

        Centering removes the intercept from the SVD; ``w0 = mean(y) - mean(Z) @ w``.

        Args:
            Z: Features, shape ``(N, D)``.
            y: Targets, shape ``(N,)``.

        Returns:
            Intercept followed by the ``D`` weights.
        """
        zbar, ybar = Z.mean(axis=0), float(y.mean())
        U, sig, Vt = np.linalg.svd(Z - zbar, full_matrices=False)
        assert U.shape == (len(Z), len(sig)) and Vt.shape == (len(sig), Z.shape[1])
        keep = sig > max(Z.shape) * np.finfo(np.float64).eps * sig[0]  # numerical rank
        w = (Vt[keep].T / sig[keep]) @ (U[:, keep].T @ (y - ybar))  # V S^+ U^T y
        return np.r_[ybar - zbar @ w, w]

    def fit_ridge(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        """Ridge on standardized features via the SVD shrinkage formula; intercept unpenalized.

        Args:
            Z: Features, shape ``(N, D)``, in their own units.
            y: Targets, shape ``(N,)``.
            lam: Penalty on the standardized weights.

        Returns:
            Intercept and weights **in the original units** of ``Z``.
        """
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        Zs = (Z - mu) / sd
        U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
        ws = Vt.T @ (S / (S**2 + lam) * (U.T @ (y - y.mean())))  # sigma / (sigma^2 + lam)
        w = ws / sd  # back to original units
        assert w.shape == (Z.shape[1],)
        return np.r_[y.mean() - mu @ w, w]

    return fit_least_squares, fit_ridge, fit_svd


@app.cell
def _(
    GROUP_NAMES,
    LinearRegression,
    RAW_NAMES,
    df,
    fit_least_squares,
    fit_svd,
    mix_groups,
    np,
    pd,
    predict,
    raw_features,
    rmse,
    te0,
    tr0,
    y_all,
):
    _rows, _rel = [], {}
    for _name, _enc in (("Lecture 2 groups", mix_groups), ("raw kg/m³ and days", raw_features)):
        _Z = _enc(df)
        _ref = LinearRegression().fit(_Z[tr0], y_all[tr0])
        _r = np.r_[_ref.intercept_, _ref.coef_]
        _A = np.column_stack([np.ones(len(tr0)), _Z[tr0]])
        for _solver, _w in (
            ("QR (ours)", fit_least_squares(_Z[tr0], y_all[tr0])),
            ("SVD (ours)", fit_svd(_Z[tr0], y_all[tr0])),
            ("inverse", np.linalg.inv(_A.T @ _A) @ _A.T @ y_all[tr0]),
        ):
            _rel[(_name, _solver)] = float(np.max(np.abs(_w - _r) / (np.abs(_r) + 1e-12)))
            _rows.append(
                {
                    "features": _name,
                    "solver": _solver,
                    "max relative difference": _rel[(_name, _solver)],
                    "allclose(rtol=1e-8)": bool(np.allclose(_w, _r, rtol=1e-8, atol=0)),
                }
            )
    # Slide "The check is part of the answer": both pass; QR to ~4e-14 (groups), inverse to
    # ~6e-12 (raw kg/m³). Exact round-off digits depend on the BLAS, so we pin the order.
    assert all(_row["allclose(rtol=1e-8)"] for _row in _rows)
    assert 1e-14 <= _rel[("Lecture 2 groups", "QR (ours)")] < 1e-13
    assert 1e-12 <= _rel[("raw kg/m³ and days", "inverse")] < 1e-11

    w_groups = fit_least_squares(mix_groups(df)[tr0], y_all[tr0])
    _w_raw = fit_least_squares(raw_features(df)[tr0], y_all[tr0])
    weights_table = pd.Series(w_groups, index=["intercept w0", *GROUP_NAMES]).round(2)
    # Water's weights are negative, as predicted in part 1: -96.83 per unit w/b, -0.16 per kg/m³.
    assert list(weights_table.round(2)[["w/b", "ln age"]]) == [-96.83, 8.34]
    assert round(float(w_groups[0]), 1) == 54.0
    assert round(float(_w_raw[1 + RAW_NAMES.index("water")]), 2) == -0.16
    assert round(rmse(predict(w_groups, mix_groups(df)[te0]), y_all[te0]), 2) == 7.82
    (pd.DataFrame(_rows), weights_table.to_frame("weight"))
    return (w_groups,)


@app.cell
def _(mo):
    mo.md(r"""
    **Result: both pass.** QR agrees with scikit-learn to about $4\times10^{-14}$, the explicit
    inverse to about $6\times10^{-12}$ on raw kg/m³, although $\kappa(X^\top X) = 10^{10}$. The
    squared condition number is a *bound*, a worst case; here it cost the inverse about two
    digits, not ten. §5 shows where the inverse does fail: a column that depends on the others.

    **Read the weights.** $w/b$: $-96.83$ MPa per unit (about $-1$ MPa per 0.01 of $w/b$);
    $\ln$ age: $+8.34$ MPa per unit (Lecture 2 measured a median gain of 8.7 within mixes). In raw
    units water's weight is $-0.16$ MPa per kg/m³. Your part 1 sign was right. Test RMSE on this
    split: 7.82 MPa.

    ### Worked example: P through the model, term by term
    """)
    return


@app.cell
def _(GROUP_NAMES, P_ROW, df, mix_groups, pd, w_groups, y_all):
    _z = mix_groups(df.loc[[P_ROW]])[0]
    _terms = w_groups[1:] * _z
    _pred = float(w_groups[0] + _terms.sum())
    _trace = pd.DataFrame(
        {"z": _z, "weight": w_groups[1:], "term w·z": _terms}, index=GROUP_NAMES
    ).round(2)
    # Diagram "A weighted sum, fitted once": 54.0 - 48.0 + 6.0 + 27.8 = 39.8; measured 38.2.
    assert [round(float(_v), 2) for _v in _z[[0, 4, 5]]] == [0.50, 4.79, 3.33]
    assert [round(float(_v), 1) for _v in _terms[[0, 4, 5]]] == [-48.0, 6.0, 27.8]
    assert (round(_pred, 1), round(float(y_all[P_ROW]), 1)) == (39.8, 38.2)
    assert round(_pred - float(y_all[P_ROW]), 1) == 1.6
    _trace
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Modify.** Check `fit_ridge` against scikit-learn's `Ridge` (on standardized features, it
    leaves the intercept unpenalized too), then look at the ridge and lasso paths when both raw
    inputs and Lecture 2's groups are offered: 14 standardized features.
    """)
    return


@app.cell
def _(Ridge, StandardScaler, df, fit_ridge, mix_groups, np, raw_features, tr0, y_all):
    _Z = np.column_stack([raw_features(df), mix_groups(df)])[tr0]  # 14 correlated columns
    _sc = StandardScaler().fit(_Z)
    for _lam in (1e-3, 1.0, 100.0):
        _ours = fit_ridge(_Z, y_all[tr0], _lam)
        _sk = Ridge(alpha=_lam, solver="svd").fit(_sc.transform(_Z), y_all[tr0])
        # scikit-learn's weights are on the standardized scale; ours are in original units.
        assert np.allclose(_ours[1:] * _sc.scale_, _sk.coef_, rtol=1e-8, atol=1e-8), _lam
        assert np.isclose(_ours[0] + _sc.mean_ @ _ours[1:], _sk.intercept_, rtol=1e-10)
        print(f"lambda = {_lam:g}: ridge matches scikit-learn (rtol 1e-8)")
    return


@app.cell
def _(
    GROUP_NAMES,
    GridSearchCV,
    GroupKFold,
    Lasso,
    MARKERS,
    PALETTE,
    RAW_NAMES,
    StandardScaler,
    df,
    fit_ridge,
    lasso_path,
    make_pipeline,
    mix_groups,
    np,
    plt,
    raw_features,
    splits,
    tr0,
    y_all,
):
    # Same 14 features as the slide: 8 raw inputs, ln age, then the 5 ratios.
    _F = np.column_stack([raw_features(df), mix_groups(df)])[tr0]
    _F = np.column_stack([_F[:, :8], _F[:, 13], _F[:, 8:13]])
    _names = RAW_NAMES + GROUP_NAMES[5:] + GROUP_NAMES[:5]
    assert _F.shape == (len(tr0), 14) and len(_names) == 14
    _Fs = StandardScaler().fit_transform(_F)
    _yc = y_all[tr0] - y_all[tr0].mean()
    _alphas, _coefs, _ = lasso_path(_Fs, _yc, n_alphas=400, eps=1e-4)
    _first = {
        _n: float(_alphas[np.flatnonzero(_coefs[_j])[0]])
        for _j, _n in enumerate(_names)
        if np.any(_coefs[_j])
    }
    _order = sorted(_first, key=_first.get, reverse=True)

    # Choose the penalty by grouped 5-fold CV, refitting the scaler inside each fold (the
    # pipeline), over LassoCV's default grid: 100 penalties from the one that zeroes every
    # weight down to 1e-3 of it. While the best penalty is the grid's smallest, the grid grows by
    # one decade at the same spacing, down to 1e-6 (the choice must be bracketed);
    # a choice still at 1e-6 is the lambda -> 0 limit, least squares.
    def _lasso_cv(rows: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Lasso weights at the CV penalty on training rows ``rows``, its index, the grid size."""
        _Ft = np.column_stack([raw_features(df), mix_groups(df)])[rows]
        _Ft = np.column_stack([_Ft[:, :8], _Ft[:, 13], _Ft[:, 8:13]])
        _yt = y_all[rows]
        _folds = list(GroupKFold(n_splits=5).split(_Ft, groups=df["design"].to_numpy()[rows]))
        _Fts = StandardScaler().fit_transform(_Ft)
        _amax = float(np.abs(_Fts.T @ (_yt - _yt.mean())).max() / len(_yt))
        _step = 3.0 / 99
        _exps = np.linspace(0.0, -3.0, 100)
        _scores = np.empty(0)
        while True:
            _search = GridSearchCV(
                make_pipeline(StandardScaler(), Lasso(max_iter=100_000)),
                {"lasso__alpha": _amax * 10.0 ** _exps[len(_scores) :]},
                cv=_folds,
                scoring="neg_mean_squared_error",
                refit=False,
            ).fit(_Ft, _yt)
            _scores = np.r_[_scores, _search.cv_results_["mean_test_score"]]
            _best = int(np.argmax(_scores))
            assert _best > 0, "the penalty that zeroes every weight cannot be the best"
            if _best < len(_exps) - 1 or -_exps[-1] >= 6.0 - 1e-9:
                break
            _exps = np.r_[_exps, _exps[-1] - _step * np.arange(1, 34)]
        _model = make_pipeline(
            StandardScaler(), Lasso(alpha=_amax * 10.0 ** _exps[_best], max_iter=100_000)
        )
        return _model.fit(_Ft, _yt)[-1].coef_, _best, len(_exps)

    _per_split = []
    for _a, _ in splits:  # about a minute: 30 splits x 100 penalties x 5 folds
        _c, _i, _n = _lasso_cv(_a)
        _per_split.append(([_nm for _nm, _w in zip(_names, _c, strict=True) if _w == 0], _i, _n))
    _dropped, _i0, _n0 = _per_split[0]
    _sets = {tuple(_r[0]) for _r in _per_split}
    _kept = [14 - len(_r[0]) for _r in _per_split]
    # Slide "Ridge shrinks the shaky directions" and notes: w/b enters first, then ln age. At the
    # cross-validated penalty lasso drops w/b on the first split (keeps 13 of 14; the penalty is
    # interior, index 92 of 100) but on only 12 of 30 splits, and the dropped set changes.
    assert _order[:2] == ["w/b", "ln age"] and _order[2] == "cement"
    assert (14 - len(_dropped), _dropped, _i0, _n0) == (13, ["w/b"], 92, 100)
    assert sum("w/b" in _r[0] for _r in _per_split) == 12
    assert (len(_sets), min(_kept), max(_kept)) == (22, 8, 14)
    # The grid grew on 3 splits; on 2 the best penalty stayed at 1e-6 (least squares).
    assert sum(_r[2] > 100 for _r in _per_split) == 3
    assert sum(_r[1] == _r[2] - 1 for _r in _per_split) == 2
    print(
        f"lasso at its CV penalty drops w/b on {sum('w/b' in _r[0] for _r in _per_split)} of "
        f"{len(_per_split)} splits; {len(_sets)} different dropped sets; keeps "
        f"{min(_kept)} to {max(_kept)} of 14"
    )

    _lams = np.logspace(-2, 5, 60)
    _ridge = np.array([fit_ridge(_F, y_all[tr0], _l)[1:] * _F.std(axis=0) for _l in _lams])
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _hl = {"w/b": 0, "ln age": 1, "cement": 2, "water": 3}
    for _j, _n in enumerate(_names):
        _kw = (
            {
                "color": PALETTE[_hl[_n]],
                "lw": 2,
                "label": _n,
                "marker": MARKERS[_hl[_n]],
                "markevery": 12,
            }
            if _n in _hl
            else {"color": "#BBBBBB", "lw": 0.8}
        )
        _ax1.plot(_lams, _ridge[:, _j], **_kw)
        _ax2.plot(_alphas, _coefs[_j], **_kw)
    _ax1.set(
        xscale="log",
        xlabel="ridge λ (standardized)",
        ylabel="weight (MPa per SD)",
        title="Ridge: every weight shrinks",
    )
    _ax2.set(
        xscale="log", xlabel="lasso α (scikit-learn scale)", title="Lasso: weights reach exactly 0"
    )
    _ax2.invert_xaxis()
    _ax1.invert_xaxis()
    _ax2.legend(loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the paths.** Read each panel from left (strong penalty) to right (weak). Lasso
    admits $w/b$ first, then $\ln$ age, then cement: under a strong penalty it picks the physics.
    At the penalty chosen by cross-validation with designs held out, though, it drops $w/b$ on
    the first split (it keeps 13 of the 14), and on only 12 of the 30 splits: the dropped set
    changes from split to split (22 different sets, 8 to 14 features kept). With correlated
    features, *which* one lasso keeps is unstable (Zou & Hastie, 2005); don't read a dropped
    feature as "irrelevant". Ridge never sets a weight to zero; it shrinks correlated weights
    together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    Each test probes a piece of the assumption from §3, *a weighted sum of the features, plus
    independent noise of constant spread*.

    **Predict before we test.** Write down four answers.

    1. **Features.** Least squares on raw kg/m³ and days, on raw kg/m³ with $\ln$ age, and on
       the Lecture 2 groups. Rank the errors, designs held out.
    2. **A dependent column.** Add binder content (cement + slag + fly ash) as a ninth input.
       What happens to the weights, and to the predictions?
    3. **Beyond the data.** Train on mixes with at most 350 kg/m³ of cement; predict those
       above 400. Too high or too low?
    4. **Units.** Ridge with one $\lambda$, age in days or in years. Same prediction?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Features: inside the data

    Same 30 design-held-out splits as Lecture 2. Above the errors: predicting the mean. Below:
    the scatter of repeat tests. The cell also refits Lecture 2's 5-NN for reference, ridge with
    $\lambda$ chosen inside each training split, a random split (leakage), and a log target.
    """)
    return


@app.cell
def _(
    GroupKFold,
    K,
    KNeighborsRegressor,
    MARKERS,
    PALETTE,
    StandardScaler,
    df,
    fit_least_squares,
    fit_ridge,
    mix_groups,
    np,
    pd,
    plt,
    predict,
    raw_features,
    raw_log_age,
    rmse,
    rsplits,
    splits,
    y_all,
):
    _enc = {
        "raw kg/m³ and days": raw_features,
        "raw, ln age": raw_log_age,
        "Lecture 2 groups": mix_groups,
    }
    _res = {}
    for _n, _e in _enc.items():
        _Z = _e(df)
        _err = [
            rmse(predict(fit_least_squares(_Z[_a], y_all[_a]), _Z[_t]), y_all[_t])
            for _a, _t in splits
        ]
        # Error bars: the sample standard deviation over the 30 splits (ddof = 1, as Lecture 2).
        _res[_n] = (round(float(np.mean(_err)), 2), round(float(np.std(_err, ddof=1)), 2))
    _Z = mix_groups(df)
    _raw = raw_features(df)

    def _knn(X: np.ndarray, sp: list, scale: bool) -> float:
        """Mean test RMSE of 5-NN over the splits, standardized on the training rows or not."""
        _e2 = []
        for _a, _t in sp:
            _Xa, _Xt = X[_a], X[_t]
            if scale:
                _sc = StandardScaler().fit(_Xa)
                _Xa, _Xt = _sc.transform(_Xa), _sc.transform(_Xt)
            _m = KNeighborsRegressor(n_neighbors=K).fit(_Xa, y_all[_a])
            _e2.append(rmse(_m.predict(_Xt), y_all[_t]))
        return float(np.mean(_e2))

    _knn_groups, _knn_random = _knn(_Z, splits, True), _knn(_Z, rsplits, True)
    _knn_raw = _knn(_raw, splits, False)
    _random = float(
        np.mean(
            [
                rmse(predict(fit_least_squares(_Z[_a], y_all[_a]), _Z[_t]), y_all[_t])
                for _a, _t in rsplits
            ]
        )
    )
    # A log target, back-transformed with exp (a median) and with Duan's smearing factor, the
    # mean of exp(training residual), which turns it into a mean.
    _log, _smear = [], []
    for _a, _t in splits:
        _wl = fit_least_squares(_Z[_a], np.log(y_all[_a]))
        _factor = float(np.mean(np.exp(np.log(y_all[_a]) - predict(_wl, _Z[_a]))))
        _p = np.exp(predict(_wl, _Z[_t]))
        _log.append(rmse(_p, y_all[_t]))
        _smear.append(rmse(_factor * _p, y_all[_t]))
    _mean = float(
        np.mean([rmse(np.full(len(_t), y_all[_a].mean()), y_all[_t]) for _a, _t in splits])
    )
    # Least squares on the groups, scored on its own training rows: little to memorize.
    _train = float(
        np.mean(
            [
                rmse(predict(fit_least_squares(_Z[_a], y_all[_a]), _Z[_a]), y_all[_a])
                for _a, _ in splits
            ]
        )
    )
    # Ridge with lambda chosen by grouped 5-fold CV inside each training split. The grid must
    # bracket the choice: never its largest penalty, and its smallest only as the
    # lambda -> 0 limit, where it shrinks no direction of the standardized features by 0.1%.
    _grid = np.logspace(-3, 3, 25)
    _design = df["design"].to_numpy()
    _ridge_err, _ridge_lam = [], []
    for _a, _t in splits:
        _folds = list(GroupKFold(5).split(_Z[_a], groups=_design[_a]))
        _cv = [
            np.mean(
                [
                    rmse(
                        predict(fit_ridge(_Z[_a][_i], y_all[_a][_i], _l), _Z[_a][_v]), y_all[_a][_v]
                    )
                    for _i, _v in _folds
                ]
            )
            for _l in _grid
        ]
        _best = float(_grid[int(np.argmin(_cv))])
        assert _best < _grid[-1]
        _zs = (_Z[_a] - _Z[_a].mean(axis=0)) / _Z[_a].std(axis=0)
        _ev_min = float(np.linalg.eigvalsh(_zs.T @ _zs).min())
        assert _best > _grid[0] or _grid[0] / (_ev_min + _grid[0]) < 1e-3
        _ridge_lam.append(_best)
        _ridge_err.append(rmse(predict(fit_ridge(_Z[_a], y_all[_a], _best), _Z[_t]), y_all[_t]))
    # The floor: repeat tests, one mix at one age tested more than once with different strengths.
    _stats = df.groupby(["mix", "age"])["strength"].agg(["size", "nunique", "std"])
    _repeat = _stats[(_stats["size"] > 1) & (_stats["nunique"] > 1)]
    _floor = float(_repeat["std"].median())

    # Slide "Inside the data, log age does the work" and its notes.
    assert [_res[_n][0] for _n in _enc] == [10.62, 7.41, 7.55]
    assert (round(_knn_groups, 2), round(_mean, 1), round(_floor, 1)) == (6.87, 17.0, 1.4)
    # The floor rests on 9 groups from 3 mixes.
    assert (len(_repeat), _repeat.index.get_level_values("mix").nunique()) == (9, 3)
    assert (round(float(np.mean(_ridge_err)), 2), round(float(np.median(_ridge_lam)), 2)) == (
        7.55,
        3.16,
    )
    assert (round(_random, 2), round(_knn_random, 2)) == (7.38, 5.65)
    assert round(_train, 2) == 7.38  # its own training rows, against 7.55 on new designs
    # 2 splits pick the smallest penalty (the least-squares limit); at the median lambda,
    # ridge shrinks the weakest direction of split 0's standardized groups by 4.8%.
    assert sum(_l == _grid[0] for _l in _ridge_lam) == 2
    _zs0 = (_Z[splits[0][0]] - _Z[splits[0][0]].mean(axis=0)) / _Z[splits[0][0]].std(axis=0)
    _ev0 = float(np.linalg.eigvalsh(_zs0.T @ _zs0).min())
    _lam_med = float(np.median(_ridge_lam))
    assert round(100 * _lam_med / (_ev0 + _lam_med), 1) == 4.8
    # The log target: 9.01 MPa with exp, 9.17 with smearing; both above 7.55 in MPa.
    assert (round(float(np.mean(_log)), 2), round(float(np.mean(_smear)), 2)) == (9.01, 9.17)
    # Lecture 2's numbers quoted in the warm-up: 5-NN on raw kg/m³ and days 9.95 MPa; water
    # carries 1.2% of the raw squared distance (its share of the summed variances).
    _var = _raw.var(axis=0)
    assert round(_knn_raw, 2) == 9.95 and round(float(100 * _var[3] / _var.sum()), 1) == 1.2
    # Warm-up question 1: Lecture 2's proposed mix (row 913) has w/b 0.40, its raw-units
    # neighbor A (row 524) 0.30.
    assert (round(float(_Z[913, 0]), 2), round(float(_Z[524, 0]), 2)) == (0.40, 0.30)

    _fig, _ax = plt.subplots(figsize=(6.5, 3.3))
    _pos = np.arange(3)[::-1] + 1
    _ax.errorbar(
        [_res[_n][0] for _n in _enc],
        _pos,
        xerr=[_res[_n][1] for _n in _enc],
        fmt=MARKERS[0],
        color=PALETTE[0],
        capsize=4,
    )
    _ax.plot(_knn_groups, 0, MARKERS[1], mfc="white", mec=PALETTE[1], mew=1.8)
    _ax.axvline(_mean, ls=":", color="#595959", label=f"predict the mean ({_mean:.1f})")
    _ax.axvline(_floor, ls="--", color="#595959", label=f"repeat tests, rough ({_floor:.1f})")
    _ax.set_yticks([*_pos, 0], [f"LS: {_n}" for _n in _enc] + ["5-NN: Lecture 2 groups"])
    _ax.set(xlabel="test RMSE (MPa), designs held out", xlim=(0, 18))
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    (pd.DataFrame(_res, index=["RMSE (MPa)", "SD over splits"]).T, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 1: inside the data, log age does the work.** Raw kg/m³ and days 10.6, with
    $\ln$ age 7.4, Lecture 2 groups 7.5 MPa; predicting the mean 17.0 above, repeat tests 1.4
    below (rough: 9 groups from 3 mixes); 5-NN on the groups 6.9. Strength grows linearly in $\ln(\text{age})$, not in
    days, so a weighted sum of days cannot follow it. Inside the data the ratios add nothing to
    $\ln$ age; reveal 3 tests them beyond it.

    Three side results. **Ridge** with $\lambda$ chosen inside each split (median 3.16) gives
    7.55, no gain: at $\lambda = 3.16$ it shrinks even the weakest direction of the six
    standardized groups by only 4.8%. Two splits pick the grid's smallest penalty, which is the
    least-squares limit (it shrinks no direction by 0.1%). **Leakage** matters less than for
    k-NN: a random split flatters least squares by 0.2 MPa (7.38 against 7.55) but 5-NN by 1.2
    (5.65 against 6.87). Least squares has little to memorize: on its own training rows its
    error is 7.38, barely below the 7.55 on new designs. **A log target**
    (Abrams' form, back-transformed) raises the error to 9.01 MPa here: across ages, strength
    in MPa, not in log MPa, is what grows linearly in $\ln(\text{age})$. It is not the
    back-transform: `exp` of a log prediction is a median, and Duan's (1983) smearing factor
    turns it into a mean, yet the error rises to 9.17. The functional form is what costs.

    ### A dependent column
    """)
    return


@app.cell
def _(
    INGREDIENTS,
    MARKERS,
    PALETTE,
    Ridge,
    SEED,
    StandardScaler,
    df,
    fit_least_squares,
    np,
    plt,
    predict,
    raw_features,
    rmse,
    splits,
    tr0,
    y_all,
):
    _X = raw_features(df)
    _Xb = np.column_stack([_X, (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()])
    _A0 = np.column_stack([np.ones(len(tr0)), _Xb[tr0]])
    _w = {"8 inputs, QR": [], "inverse": [], "QR": [], "SVD": [], "ridge": []}
    _e = {_k: [] for _k in _w}
    for _a, _t in splits:
        _Aa = np.column_stack([np.ones(len(_a)), _Xb[_a]])
        _fits = {
            "8 inputs, QR": (fit_least_squares(_X[_a], y_all[_a]), _X),
            "inverse": (np.linalg.inv(_Aa.T @ _Aa) @ _Aa.T @ y_all[_a], _Xb),
            "QR": (fit_least_squares(_Xb[_a], y_all[_a]), _Xb),
            "SVD": (np.linalg.lstsq(_Aa, y_all[_a], rcond=None)[0], _Xb),
        }
        for _k, (_wk, _M) in _fits.items():
            _w[_k].append(float(_wk[1]))
            _e[_k].append(rmse(predict(_wk, _M[_t]), y_all[_t]))
        _sc = StandardScaler().fit(_Xb[_a])
        _m = Ridge(alpha=1.0).fit(_sc.transform(_Xb[_a]), y_all[_a])
        _w["ridge"].append(float(_m.coef_[0] / _sc.scale_[0]))
        _e["ridge"].append(rmse(_m.predict(_sc.transform(_Xb[_t])), y_all[_t]))
    _span = {_k: (round(min(_v), 2), round(max(_v), 2)) for _k, _v in _w.items()}
    _qr0 = fit_least_squares(_Xb[tr0], y_all[tr0])
    _baseline = float(
        np.mean([rmse(np.full(len(_t), y_all[_a].mean()), y_all[_t]) for _a, _t in splits])
    )
    _mass = df[INGREDIENTS].sum(axis=1)
    # Slide "A dependent column breaks the inverse" and its notes. The inverse's and QR's digits
    # are round-off (they change with the BLAS, the machine, or the column order), so we check
    # only what holds on any machine: kappa ~ 1e16, the inverse's cement weight takes both
    # signs and its RMSE is worse than predicting the mean, QR's weights reach ~1e12.
    assert (np.linalg.matrix_rank(_A0), _A0.shape[1]) == (9, 10)
    assert np.linalg.cond(_A0) > 1e15
    assert min(_w["inverse"]) < 0 < max(_w["inverse"])
    assert float(np.mean(_e["inverse"])) > _baseline
    assert float(np.max(np.abs(_qr0))) > 1e10
    assert _span["8 inputs, QR"] == (0.10, 0.15) and _span["SVD"] == (0.04, 0.05)
    assert np.allclose(_e["SVD"], _e["8 inputs, QR"], rtol=1e-6)
    assert round(float(np.mean(_e["SVD"])), 1) == 10.6
    assert round(float(np.mean(_e["ridge"])), 1) == 10.6
    # Notes: the seven ingredients add up to 2,195 to 2,551 kg per cubic meter.
    assert (round(float(_mass.min())), round(float(_mass.max()))) == (2195, 2551)
    print(
        f"this machine: kappa = {np.linalg.cond(_A0):.1e}; inverse cement weight "
        f"{min(_w['inverse']):+.2f} to {max(_w['inverse']):+.2f}, RMSE "
        f"{np.mean(_e['inverse']):.1f} MPa (mean baseline {_baseline:.1f}); "
        f"max |QR weight| {np.max(np.abs(_qr0)):.0e}"
    )

    _fig, _ax = plt.subplots(figsize=(6.5, 3.3))
    _rng = np.random.default_rng(SEED)
    _show = ["8 inputs, QR", "inverse", "SVD", "ridge"]
    for _j, _k in enumerate(_show):
        _ax.plot(
            _w[_k],
            3 - _j + _rng.uniform(-0.15, 0.15, len(_w[_k])),
            MARKERS[_j],
            color=PALETTE[_j],
            alpha=0.8,
        )
    _ax.set_yticks(
        [3, 2, 1, 0], ["8 inputs, QR", "+ binder, inverse", "+ binder, SVD", "+ binder, ridge"]
    )
    _ax.axvline(0, color="#595959", lw=1)
    _lim = max(2.0, 1.05 * max(abs(_v) for _k in _show for _v in _w[_k]))  # round-off: widen
    _ax.set(xlabel="weight of cement (MPa per kg/m³), 30 splits", xlim=(-_lim, _lim))
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 2: a dependent column breaks the inverse.** Binder = cement + slag + fly ash holds
    exactly in every row, so the design matrix has rank 9 of 10 and $\kappa(X)\sim10^{16}$: its
    smallest singular value sits at round-off. Any multiple of $(1, 1, 1, -1)$ can be added to
    the weights of cement, slag, fly ash, and binder without changing a single prediction.

    The digits printed above are round-off: they change with the BLAS, the machine, even the
    order of the columns. Only their size is a result, and it holds on any machine:

    - **Inverse:** the cement weight flips sign from split to split, and the test RMSE (tens of
      MPa) is worse than predicting the mean.
    - **QR without pivoting:** weights of order $10^{12}$ that cancel only because the test rows
      obey the same identity.
    - **SVD** (minimum norm) **and ridge:** stable weights and 10.6 MPa, exactly the 8-input
      model's error. Cement's effect is now split between cement (0.04) and binder.

    Real data have near-dependencies too: the seven ingredients add up to the mass of a cubic
    meter of concrete, 2,195 to 2,551 kg. When inputs depend on each other, report predictions,
    not single weights (Forsyth, Exercise 10.6, proves why).

    ### Beyond the data: more cement than in training (the decisive check)
    """)
    return


@app.cell
def _(
    K,
    KNeighborsRegressor,
    MARKERS,
    PALETTE,
    RAW_NAMES,
    StandardScaler,
    df,
    fit_least_squares,
    mix_groups,
    np,
    plt,
    predict,
    raw_features,
    raw_log_age,
    rmse,
    y_all,
):
    _cem = df["cement"].to_numpy()
    _tr, _te = np.flatnonzero(_cem <= 350.0), np.flatnonzero(_cem > 400.0)
    _pred = {}
    for _n, _e in (
        ("raw, ln age", raw_log_age),
        ("raw kg/m³ and days", raw_features),
        ("Lecture 2 groups", mix_groups),
    ):
        _w = fit_least_squares(_e(df)[_tr], y_all[_tr])
        _pred[_n] = predict(_w, _e(df)[_te])
        if _n == "raw, ln age":
            _cement_w = float(_w[1 + RAW_NAMES.index("cement")])
    _Z = mix_groups(df)
    _sc = StandardScaler().fit(_Z[_tr])
    _knn = KNeighborsRegressor(n_neighbors=K).fit(_sc.transform(_Z[_tr]), y_all[_tr])
    _pred["5-NN"] = _knn.predict(_sc.transform(_Z[_te]))
    _nbr = _tr[_knn.kneighbors(_sc.transform(_Z[_te]))[1]]  # rows of each test's 5 neighbours
    assert _nbr.shape == (len(_te), K)
    _above = y_all[_te] > y_all[_nbr].max(axis=1)  # stronger than all five of its neighbours
    _bias = {_n: round(float(np.mean(_p - y_all[_te])), 1) for _n, _p in _pred.items()}
    _err = {_n: round(rmse(_p, y_all[_te]), 1) for _n, _p in _pred.items()}
    _in = (_Z[_te] >= _Z[_tr].min(axis=0)) & (_Z[_te] <= _Z[_tr].max(axis=0))  # per ratio
    assert _in.shape == (len(_te), 6)
    _inside = float(_in[:, 0].mean())  # w/b
    _te28 = df["age"].to_numpy()[_te] == 28
    _bias28 = {
        _n: round(float(np.mean(_p[_te28] - y_all[_te][_te28])), 1) for _n, _p in _pred.items()
    }
    # The test set is 136 tests of only a few dozen designs, and tests of one design are not
    # independent: resample whole designs (2,000 draws) for intervals on the raw model's bias and
    # on its RMSE gap to the groups.
    _ids, _inv = np.unique(df["design"].to_numpy()[_te], return_inverse=True)
    _er, _eg = _pred["raw, ln age"] - y_all[_te], _pred["Lecture 2 groups"] - y_all[_te]
    _cnt = np.stack(
        [
            np.bincount(_row, minlength=len(_ids))
            for _row in np.random.default_rng(0).integers(0, len(_ids), size=(2000, len(_ids)))
        ]
    ).astype(float)
    _nrow = _cnt @ np.bincount(_inv)
    _bias_draws = _cnt @ np.bincount(_inv, weights=_er) / _nrow
    _gap_draws = np.sqrt(_cnt @ np.bincount(_inv, weights=_er**2) / _nrow) - np.sqrt(
        _cnt @ np.bincount(_inv, weights=_eg**2) / _nrow
    )
    _bias_ci = [round(float(_v), 1) for _v in np.percentile(_bias_draws, [2.5, 97.5])]
    _gap_ci = [round(float(_v), 1) for _v in np.percentile(_gap_draws, [2.5, 97.5])]
    # Abrams' law, 28-day fit (Lecture 2: A = 123.8, B = 16.5): MPa gained per kg of cement at
    # fixed water 185 kg/m³, d/dc [A B^(-185/c)] = A ln B (185/c²) B^(-185/c).
    _b = df["cement"] + df["slag"] + df["fly_ash"]
    _d28 = df["age"] == 28
    _s, _i = np.polyfit((df["water"] / _b)[_d28], np.log(df["strength"][_d28]), 1)
    _A, _B = float(np.exp(_i)), float(np.exp(-_s))

    def _per_kg(c: float) -> float:
        return _A * np.log(_B) * 185.0 / c**2 * _B ** (-185.0 / c)

    # Slide "Ratios hold up beyond the data" (the decisive figure) and its notes.
    assert (len(_tr), len(_te)) == (774, 136)
    assert _bias == {
        "raw, ln age": 4.7,
        "raw kg/m³ and days": 6.1,
        "Lecture 2 groups": -2.3,
        "5-NN": -8.5,
    }
    assert _err == {
        "raw, ln age": 9.9,
        "raw kg/m³ and days": 14.6,
        "Lecture 2 groups": 8.9,
        "5-NN": 12.6,
    }
    assert round(_inside, 2) == 0.92 and round(_cement_w, 2) == 0.16
    # 5-NN averages lower-cement neighbours: 64 of the tests measured above all five; the
    # 72 within their neighbours' range err by only -2.4. No test exceeds the training
    # maximum (81.8 MPa), so the cap of Lecture 2 is not the cause.
    assert (int(_above.sum()), int((~_above).sum())) == (64, 72)
    assert round(float(np.mean((_pred["5-NN"] - y_all[_te])[~_above])), 1) == -2.4
    assert round(float(y_all[_tr].max()), 1) == 81.8
    assert int(np.sum(y_all[_te] > y_all[_tr].max())) == 0
    # Every ratio inside its training range: 82% of the test mixes.
    assert round(float(_in.all(axis=1).mean()), 2) == 0.82
    # 136 tests of 39 designs and 41 mixes; 43 of them at 28 days.
    assert (
        df["design"].iloc[_te].nunique(),
        df["mix"].iloc[_te].nunique(),
        int(_te28.sum()),
    ) == (39, 41, 43)
    # The over-prediction comes from the other ages: at 28 days the raw model is nearly unbiased.
    assert _bias28 == {
        "raw, ln age": 0.4,
        "raw kg/m³ and days": -3.0,
        "Lecture 2 groups": -5.5,
        "5-NN": -7.7,
    }
    # Intervals over designs: the bias is robust, the RMSE gap (1.1) is not resolved.
    _gap = rmse(_pred["raw, ln age"], y_all[_te]) - rmse(_pred["Lecture 2 groups"], y_all[_te])
    assert round(_gap, 1) == 1.1  # from the raw RMSEs, not 9.9 - 8.9
    assert (_bias_ci, _gap_ci) == ([1.9, 6.9], [-1.2, 3.1])
    assert (round(_A, 1), round(_B, 1)) == (123.8, 16.5)
    assert (round(_per_kg(300.0), 2), round(_per_kg(500.0), 2)) == (0.13, 0.09)

    _fig, _ax = plt.subplots(figsize=(5.5, 4.2))
    _ax.plot([0, 95], [0, 95], ":", color="#595959")
    for _j, (_n, _c) in enumerate((("raw, ln age", 1), ("Lecture 2 groups", 0), ("5-NN", 2))):
        _ax.scatter(
            y_all[_te],
            _pred[_n],
            s=18,
            marker=MARKERS[[1, 0, 2][_j]],
            alpha=0.7,
            color=PALETTE[_c],
            label=f"{_n}: {_bias[_n]:+.1f} MPa",
        )
    _ax.set(
        xlim=(0, 95),
        ylim=(0, 95),
        xlabel="measured strength, cement > 400 (MPa)",
        ylabel="prediction (MPa)",
    )
    _ax.legend(title="mean error", loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 3: ratios hold up beyond the data.** Trained on the 774 tests with at most
    350 kg/m³ of cement and tested on the 136 above 400, least squares on raw kg/m³ with
    $\ln$ age **over-predicts by 4.7 MPa** on average over all ages (RMSE 9.9), the unsafe
    side; resampling the 39 test designs gives a 95% interval of 1.9 to 6.9 MPa, so the sign is
    robust. The Lecture 2 groups err by $-2.3$ (RMSE 8.9), on the safe side; 5-NN by $-8.5$
    (12.6): it averages five lower-cement neighbours, and 64 of the 136 tests measured above
    all five (the 72 within their neighbours' range err by only $-2.4$; no test exceeds the
    training maximum, 81.8 MPa). Inside the data the two
    least-squares models were equal (7.4 and 7.5).

    Two limits on this result. The RMSE gap between the raw and the groups model is within
    design resampling (95%: $-1.2$ to $+3.1$ MPa): their RMSEs are similar; what differs is the
    side their errors fall on. And the over-prediction comes from the other ages: on the 43
    tests at 28 days, the acceptance age, the raw model errs $+0.4$ and the groups $-5.5$.

    **Why.** Extrapolation depends on the features. In kilograms these mixes are new; in ratios
    most are not: 92% of their $w/b$ values were tested, and 82% lie inside every ratio's
    training range. Abrams' law says each kg of cement adds 0.13 MPa at
    300 kg/m³ but only 0.09 at 500 (fixed water 185 kg/m³); the raw line keeps the 0.16 MPa per
    kg it learned from low-cement mixes. The assumption *a weighted sum of these features* holds
    beyond the data only if the features carry the physics.

    ### Units: ridge without standardizing
    """)
    return


@app.cell
def _(LinearRegression, RAW_NAMES, Ridge, StandardScaler, df, np, pd, raw_features, y_all):
    QUERY = {
        "cement": 600.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 165.0,
        "superplasticizer": 10.0,
        "coarse_agg": 1000.0,
        "fine_agg": 650.0,
        "age": 28.0,
    }
    _X = raw_features(df)
    _q = np.array([[QUERY[_c] for _c in RAW_NAMES]])
    _Xy, _qy = _X.copy(), _q.copy()
    _age = RAW_NAMES.index("age")
    _Xy[:, _age] /= 365.0  # age in years
    _qy[:, _age] /= 365.0
    _lam = 100.0
    _ols = LinearRegression().fit(_X, y_all)
    _days, _years = Ridge(alpha=_lam).fit(_X, y_all), Ridge(alpha=_lam).fit(_Xy, y_all)
    _scd, _scy = StandardScaler().fit(_X), StandardScaler().fit(_Xy)
    _sd = Ridge(alpha=_lam).fit(_scd.transform(_X), y_all)
    _sy = Ridge(alpha=_lam).fit(_scy.transform(_Xy), y_all)
    _table = pd.DataFrame(
        {
            "age in days": [_days.predict(_q)[0], _sd.predict(_scd.transform(_q))[0]],
            "age in years": [_years.predict(_qy)[0], _sy.predict(_scy.transform(_qy))[0]],
        },
        index=["ridge, not standardized", "ridge, standardized"],
    ).round(1)
    _kept = 100 * _years.coef_[_age] / 365.0 / _ols.coef_[_age]
    # Slide "Unscaled ridge depends on units": 61.1 / 63.9 unscaled, 60.4 both scaled; least
    # squares 61.1; ridge keeps 22% of age's weight in years.
    assert _table.to_numpy().tolist() == [[61.1, 63.9], [60.4, 60.4]]
    assert round(float(_ols.predict(_q)[0]), 1) == 61.1 and round(float(_kept)) == 22
    # Notes and the audit key: unscaled ridge at scikit-learn's default lambda = 1 gives 61.1.
    assert round(float(Ridge(alpha=1.0).fit(_X, y_all).predict(_q)[0]), 1) == 61.1
    _table
    return (QUERY,)


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 4: unscaled ridge depends on units.** All 1,030 tests, one $\lambda = 100$, and the
    600 kg/m³ mix of the delegate exercise. With age in days the penalty is negligible
    (61.1 MPa, the least-squares answer); in years, age's weight is 365 times larger, so the
    penalty bites and ridge keeps only 22% of it (63.9 MPa). Standardized, both give 60.4.
    The penalty acts on a weight's size, and a weight's size depends on its feature's units;
    standardizing makes $\lambda$ mean the same for every feature. With scikit-learn's default
    $\lambda = 1$, unscaled ridge on kg/m³ and days is least squares in disguise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Worksheet, part 4: Delegate (7 minutes, groups of three)

    Ask an AI assistant:

    > "With the UCI concrete data, fit a ridge regression and predict the 28-day strength of:
    > cement 600 kg/m³, water 165, superplasticizer 10, coarse 1,000, fine 650, no slag or fly
    > ash."

    Paste its code and its answer into your worksheet, then **audit** it:

    | Audit question | Assistant's answer | Your verdict |
    |:--|:--|:--|
    | 1. Did it warn about extrapolation? (Tested cement stops at 540 kg/m³.) | | |
    | 2. Did it standardize before penalizing, and leave the intercept out? | | |
    | 3. Is the prediction physically plausible, given part 1? | | |
    """)
    return


@app.cell
def _(
    K,
    KNeighborsRegressor,
    QUERY,
    StandardScaler,
    df,
    fit_least_squares,
    mix_groups,
    mo,
    np,
    pd,
    predict,
    raw_features,
    raw_log_age,
    y_all,
):
    _qdf = pd.DataFrame([QUERY])
    _key = {}
    for _n, _e in (
        ("least squares, raw kg/m³ and days", raw_features),
        ("least squares, raw kg/m³ with ln age", raw_log_age),
        ("least squares, Lecture 2 groups", mix_groups),
    ):
        _key[_n] = float(predict(fit_least_squares(_e(df), y_all), _e(_qdf))[0])
    _Z = mix_groups(df)
    _sc = StandardScaler().fit(_Z)
    _key["5-NN, Lecture 2 groups"] = float(
        KNeighborsRegressor(n_neighbors=K)
        .fit(_sc.transform(_Z), y_all)
        .predict(_sc.transform(mix_groups(_qdf)))[0]
    )
    _zq = mix_groups(_qdf)[0]
    _d28 = (df["age"] == 28).to_numpy()
    _near = _d28 & (np.abs(_Z[:, 0] - _zq[0]) <= 0.03)
    # Slide notes, audit key: 61.1, 65.5, 59.9, 53.9 MPa; w/c 0.275 inside 0.24–0.90; aggregate/b
    # 2.75 inside 2.38–9.85; 42 28-day tests (35 mixes) with w/b within 0.03 reached 33.7 to 80.
    assert [round(_v, 1) for _v in _key.values()] == [61.1, 65.5, 59.9, 53.9]
    assert round(float(_zq[0]), 3) == 0.275 and df["cement"].max() == 540.0
    assert (round(float(_Z[:, 0].min()), 2), round(float(_Z[:, 0].max()), 2)) == (0.24, 0.90)
    assert (
        round(float(_zq[4]), 2),
        round(float(_Z[:, 4].min()), 2),
        round(float(_Z[:, 4].max()), 2),
    ) == (2.75, 2.38, 9.85)
    assert (
        int(_near.sum()),
        round(float(y_all[_near].min()), 1),
        round(float(y_all[_near].max()), 1),
    ) == (42, 33.7, 80.0)
    assert df.loc[_near, "mix"].nunique() == 35  # 42 tests of 35 mixes
    _rows = "\n".join(f"| {_n} | {_v:.1f} |" for _n, _v in _key.items())
    _notes = "\n".join(
        [
            "1. **Extrapolation.** In kilograms the mix is outside the data (600 > 540 kg/m³). In",
            "   ratios it is inside: $w/c = 0.275$ (tested $w/b$: 0.24 to 0.90), aggregate$/b$ =",
            "   2.75 (tested 2.38 to 9.85). A sound answer says which, and checks every feature.",
            "2. **Scaling.** Unscaled ridge at the default $\\lambda = 1$ returns the least-squares",
            "   answer, 61.1 MPa: the penalty did nothing (§5, reveal 4).",
            "3. **Plausibility.** 42 28-day tests (35 mixes) with $w/b$ within 0.03 of 0.275 reached",
            "   33.7 to 80.0 MPa, so any answer in that band is plausible. An answer that falls as",
            "   cement rises, or a positive weight on water, is not.",
        ]
    )
    _table = "| Model, all 1,030 tests | Prediction (MPa) |\n|:--|--:|\n" + _rows
    mo.accordion({"Audit key (open after your audit)": mo.md(_table + "\n\n" + _notes)})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Uncertainty for the decision: how much cement for 35 MPa?

    Use the verified model of §4 (the first split's least squares on the groups) for P's family:
    186 kg/m³ of water, P's aggregates, no admixtures, 28 days. Solve for the cement where the
    prediction reaches 35 MPa. Then add a margin: the 90th percentile of *prediction minus
    measured* over every held-out prediction of a 28-day test in the 30 splits: 3,179
    predictions of the 425 distinct 28-day tests, since each test is held out in several splits.
    If this mix errs like those, aiming the prediction at $35 + \text{margin}$ leaves about 10%
    of such mixes below 35 MPa.
    """)
    return


@app.cell
def _(
    FC,
    MARKERS,
    PALETTE,
    P_ROW,
    df,
    fit_least_squares,
    mix_groups,
    np,
    pd,
    plt,
    predict,
    splits,
    w_groups,
    y_all,
):
    _Z = mix_groups(df)
    _age = df["age"].to_numpy()
    _over, _preds, _tested, _split_q, _log_over, _log_preds = [], [], [], [], [], []
    for _a, _t in splits:
        _t28 = _t[_age[_t] == 28]
        _p = predict(fit_least_squares(_Z[_a], y_all[_a]), _Z[_t28])
        _over.append(_p - y_all[_t28])
        _preds.append(_p)
        _tested.append(_t28)
        _split_q.append(float(np.quantile(_p - y_all[_t28], 0.90)))  # this split's own margin
        # The same model on a log target, back-transformed with exp, for its scatter by band.
        _pl = np.exp(predict(fit_least_squares(_Z[_a], np.log(y_all[_a])), _Z[_t28]))
        _log_over.append(_pl - y_all[_t28])
        _log_preds.append(_pl)
    _over, _preds = np.concatenate(_over), np.concatenate(_preds)
    _log_over, _log_preds = np.concatenate(_log_over), np.concatenate(_log_preds)
    _rows = np.concatenate(_tested)
    _per_test = pd.Series(_over).groupby(_rows).mean()  # each distinct test's mean error
    _margin = float(np.quantile(_over, 0.90))
    _cem = np.linspace(250, 540, 2901)
    _fam = pd.concat([df.loc[[P_ROW]]] * len(_cem), ignore_index=True)
    _fam["cement"] = _cem
    _curve = predict(w_groups, mix_groups(_fam))
    assert np.all(np.diff(_curve) > 0), "prediction should rise with cement"
    # np.interp clamps outside the curve: both targets must lie on it.
    assert _curve[0] < FC < FC + _margin < _curve[-1]
    _c50, _c90 = float(np.interp(FC, _curve, _cem)), float(np.interp(FC + _margin, _curve, _cem))
    _water = float(df.at[P_ROW, "water"])
    # Slide "Uncertainty costs cement", the vote, and "Use the line inside the data only".
    assert (len(_over), round(_margin, 1), round(FC + _margin, 1)) == (3179, 9.0, 44.0)
    # Notes: the 3,179 held-out predictions are of 425 distinct 28-day tests (each test is held
    # out in several of the 30 splits).
    assert len(np.unique(_rows)) == 425
    _held = np.bincount(_rows)[np.unique(_rows)]
    assert (int(_held.min()), int(_held.max())) == (1, 14)  # times each test is held out
    # Per test: 58% of the 425 tests were over-predicted on average.
    assert round(float(np.mean(_per_test.to_numpy() > 0)), 2) == 0.58
    # The margin is itself uncertain: each split's own 90th percentile.
    assert (
        round(float(np.median(_split_q)), 1),
        round(min(_split_q), 1),
        round(max(_split_q), 1),
    ) == (8.8, 5.0, 11.6)
    assert (round(_c50), round(_c90), round(_c90) - round(_c50)) == (337, 418, 81)
    assert round(100 * (round(_c90) - round(_c50)) / round(_c50)) == 24
    assert round(float(np.mean(_over > 0)), 2) == 0.57
    assert (
        round(_water / round(_c50), 2),
        round(_water / round(_c90), 2),
        round(_water / 540.0, 2),
        540 - round(_c50),
    ) == (0.55, 0.44, 0.34, 203)
    # Scatter (sample SD) by prediction band; a log target does not even it out.
    assert (
        round(float(np.std(_over[_preds < 30], ddof=1)), 1),
        round(float(np.std(_over[_preds > 50], ddof=1)), 1),
    ) == (6.5, 11.1)
    assert (
        round(float(np.std(_log_over[_log_preds < 30], ddof=1)), 1),
        round(float(np.std(_log_over[_log_preds > 50], ddof=1)), 1),
    ) == (6.4, 11.6)

    _plain = (
        (df["age"] == 28)
        & (df[["slag", "fly_ash", "superplasticizer"]].sum(axis=1) == 0)
        & (np.abs(df["water"] - 185.0) <= 2.0)
    ).to_numpy()
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    _ax.scatter(
        df["cement"][_plain],
        y_all[_plain],
        s=16,
        color="#595959",
        alpha=0.6,
        label="28-day tests, water ≈ 185",
    )
    _ax.plot(_cem, _curve, color=PALETTE[0], label="prediction")
    _ax.plot(_cem, _curve - _margin, "--", color=PALETTE[1], label=f"prediction − {_margin:.1f}")
    _ax.axhline(FC, ls=":", color="#1A1A1A", lw=1)
    _ax.plot([_c50, _c90], [FC, FC], ls="none", marker=MARKERS[0], color=PALETTE[0])
    _ax.plot(_c90, FC, MARKERS[1], color=PALETTE[1])
    _ax.set(
        xlabel=f"cement (kg/m³), water {_water:.0f}", ylabel="28-day strength (MPa)", ylim=(10, 70)
    )
    _ax.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The cost of uncertainty, in cement.** The line reaches 35 MPa at **337 kg/m³**, but 57%
    of the 3,179 held-out 28-day predictions were above the measured strength (58% of the 425
    tests, averaging each test's predictions). 90% were at most 9.0 MPa above it, so aim the
    prediction at 44.0 MPa: **418 kg/m³**, 81 kg/m³ more (+24%). The margin is itself
    uncertain: each split's own 90th percentile runs from 5.0 to 11.6 MPa (median 8.8);
    Lecture 10 returns to it. The four-mix fit of §3 said 362 kg/m³: one laboratory's series,
    with almost no scatter. Design codes already make a mix target an average strength above
    $f'_c$ (Lecture 2), and in practice that margin comes from the standard deviation of the
    producer's own test records (ACI 301), not from many laboratories' mixes. This is the same
    idea, priced in cement.

    ### When not to use the line

    - **Not beyond the tested ratios and ages:** check every feature's range, not only the
      inputs'.
    - **Not for new materials:** other cements, admixtures, or aggregates are not in these
      tests.
    - **Not to read single weights** when inputs depend on each other: report predictions.
    - **Not with one margin for all mixes:** the scatter grows with strength (standard
      deviation 6.5 MPa for predictions below 30, 11.1 above 50), against the constant-spread
      assumption of §3. A log target does not even it out here (6.4 and 11.6 MPa).
    - **Not for acceptance:** cylinder tests accept concrete (Lecture 2). Calibrated intervals
      come in Lectures 10–11.

    ### Impact: the engineer signs, not the assistant

    A confident number arrives in seconds; whether it checked range, units, and sign is what
    your audit found out. The Campus AI Curriculum Task Force (University of Illinois, Office of
    the Provost) defines AI fluency as using AI tools "effectively, reflectively, and
    responsibly within disciplinary and professional contexts." In concrete, responsibly means
    that the engineer of record signs the mix design, and the audit is part of the engineering.
    An unaudited answer can cut cement below what is safe, or add cement no one needs; both
    costs fall on people who never saw the model.

    ### Vote: how much cement for 35 MPa?

    $f'_c = 35$ MPa at 28 days; 186 kg/m³ of water, no admixtures; materials like those tested.

    - **A.** 337 kg/m³: where the fitted line reaches 35 MPa.
    - **B.** 418 kg/m³: where the line minus 9.0 MPa, its 90th-percentile over-prediction,
      reaches 35 MPa.
    - **C.** 540 kg/m³: the most cement ever tested, to be safe.

    Vote alone. Discuss with a neighbor for two minutes. Vote again. Then open the explanation.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Explanation": mo.md(
                r"""
                **A** ignores the scatter: 57% of held-out 28-day predictions were above the
                measured strength. **C** is safe but not free: 203 kg/m³ more cement than A
                ($w/c = 0.34$), with its cost and carbon, and it sits at the edge of the tested
                range. **B** prices the uncertainty the data show ($w/c = 0.44$); trial batches
                confirm it before production.

                What would change the answer: the producer's own records (ACI 301 sets the
                margin from them), a cheaper wrong "yes", or new materials.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Least squares inside modern ML

    The closed form of §3 keeps reappearing inside the largest models.

    - **The last layer.** A network for regression ends in a weighted sum trained on squared
      error. Fix the layers below it, and fitting the last layer is today's problem
      (Lecture 14). The next cell freezes a random hidden layer and fits only the last one, with
      our `fit_ridge`.
    - **Linear probes.** Freeze a foundation model and fit a linear model on its embeddings, as
      CLIP does with logistic regression to report its representations (Radford et al., 2021).
      For a regression target the probe is ridge, solved exactly as today. Lectures 15 and
      17–19 fit such probes on frozen features.
    - **Weight decay** is the ridge penalty in network training (Krogh & Hertz, 1991). For SGD,
      $L_2$ regularization and weight decay are the same update; for Adam they differ, which is
      why AdamW decouples the decay (Loshchilov & Hutter, 2019). The cell after next checks both
      claims. At scale, stochastic gradient descent replaces QR on the same loss (Lecture 13).
    """)
    return


@app.cell
def _(SEED, df, fit_ridge, mix_groups, np, predict, rmse, splits, y_all):
    _rng = np.random.default_rng(SEED)
    _Z = mix_groups(df)
    _W1 = _rng.normal(size=(6, 200))  # frozen first layer: 200 random ReLU units
    _b1 = _rng.normal(size=200)

    def _hidden(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        _H = np.maximum(((Z - mu) / sd) @ _W1 + _b1, 0.0)
        assert _H.shape == (len(Z), 200)
        return _H

    _err = []
    for _a, _t in splits:
        _mu, _sd = _Z[_a].mean(axis=0), _Z[_a].std(axis=0)
        _H = _hidden(_Z[_a], _mu, _sd)
        _keep = _H.std(axis=0) > 0  # drop units that never fire on training rows
        _w = fit_ridge(_H[:, _keep], y_all[_a], 10.0)  # the last layer, in closed form
        _err.append(rmse(predict(_w, _hidden(_Z[_t], _mu, _sd)[:, _keep]), y_all[_t]))
    assert round(float(np.mean(_err)), 2) == 7.22  # seeded; this notebook only
    print(
        f"random hidden layer + ridge last layer: test RMSE {np.mean(_err):.2f} MPa "
        f"(designs held out; least squares on the 6 groups: 7.55)"
    )
    return


@app.cell
def _(df, mix_groups, np, torch, tr0, y_all):
    # One optimizer step from the same start on the least-squares loss of the six groups.
    _Zt = torch.tensor(
        (mix_groups(df)[tr0] - mix_groups(df)[tr0].mean(0)) / mix_groups(df)[tr0].std(0),
        dtype=torch.float64,
    )
    _yt = torch.tensor(y_all[tr0] - y_all[tr0].mean(), dtype=torch.float64)
    _lam, _lr = 0.1, 0.01

    def _step(opt_name: str, wd: float, l2: float) -> np.ndarray:
        _w = torch.ones(6, dtype=torch.float64, requires_grad=True)
        _opt = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[
            opt_name
        ]([_w], lr=_lr, weight_decay=wd)
        _loss = ((_Zt @ _w - _yt) ** 2).mean() + 0.5 * l2 * (_w**2).sum()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        return _w.detach().numpy().copy()

    # SGD: weight decay and an L2 penalty in the loss give the same step.
    _sgd_wd, _sgd_l2 = _step("SGD", _lam, 0.0), _step("SGD", 0.0, _lam)
    assert np.allclose(_sgd_wd, _sgd_l2, rtol=1e-12, atol=1e-12)
    # Adam: the L2 penalty (Adam's weight_decay) and decoupled decay (AdamW) differ.
    _adam_l2, _adamw = _step("Adam", _lam, 0.0), _step("AdamW", _lam, 0.0)
    assert not np.allclose(_adam_l2, _adamw, rtol=1e-6, atol=1e-9)
    print("SGD: weight decay == L2 penalty:", np.allclose(_sgd_wd, _sgd_l2))
    print(
        "Adam with L2 vs AdamW, max |difference| after one step:",
        f"{np.max(np.abs(_adam_l2 - _adamw)):.2e}",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    ### Worksheet, part 5: Debrief (3 minutes, alone, on paper)

    One-minute paper. Complete: **"One thing the AI did that I would not have caught without
    step 3."** (Step 3 is Implement.) Responses seed the next lecture's warm-up.

    *Your answer:* ________
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Common answers": mo.md(
                r"""
                No warning that 600 kg/m³ is beyond every tested mix; ridge without
                standardizing (so the penalty did nothing, or depended on units); a prediction
                quoted to a decimal with no error bar; a model on raw kilograms, which
                over-predicts high-cement mixes (§5, reveal 3).
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **More drills.** Answer each question before you open it.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Q1. Why does centering the feature make the 2×2 normal equations decouple?": mo.md(
                r"""
                With $\Delta x = x - \bar{x}$, the columns $\mathbf{1}$ and $\Delta\mathbf{x}$ are
                orthogonal: $\mathbf{1}^\top\Delta\mathbf{x} = 0$. So $X^\top X$ is diagonal,
                $\operatorname{diag}(N, \sum\Delta x^2)$, and each weight is solved alone. QR does
                the same for any number of columns: $Q$ has orthonormal columns.
                """
            ),
            "Q2. When is X^T X singular, and what do the SVD and ridge do then?": mo.md(
                r"""
                When a column is a linear combination of others (binder = cement + slag + fly
                ash), or when $D > N$. The SVD inverts only nonzero singular values and returns the
                minimum-norm weights; ridge adds $\lambda I$ and returns a unique solution for any
                $\lambda > 0$. Predictions on data that obey the same identity are unaffected;
                single weights are not identifiable.
                """
            ),
            "Q3. Why standardize before ridge or lasso, but not before plain least squares?": mo.md(
                r"""
                Least squares is equivariant to rescaling a feature: the weight rescales and the
                predictions do not change. The penalty is not: it acts on the weight's size, which
                depends on the units (§5: age in days or in years changed the ridge prediction from
                61.1 to 63.9 MPa). Standardizing makes $\lambda$ mean the same for every feature.
                """
            ),
            "Q4. Ridge did not lower the in-range error here. Why use it at all?": mo.md(
                r"""
                With 746 tests and 7 weights there is little variance to remove (7.55 MPa with or
                without it). Ridge pays off when features are many or correlated: it makes the
                solution unique and stable (§5, reveal 2), and it is the weight decay that trains
                large networks.
                """
            ),
            "Q5. Raw kg/m³ with ln age, or the groups: which would you deploy?": mo.md(
                r"""
                The groups. Inside the data they tie (7.4 and 7.5 MPa). Beyond the tested cement
                their RMSEs are similar (the gap is within design resampling), but the raw model
                over-predicts by 4.7 MPa over all ages (95%: 1.9 to 6.9, designs resampled), the
                unsafe side, while the groups err by $-2.3$, the safe side. In ratios, 92% of the
                high-cement mixes' $w/b$ values were inside the tested range.
                """
            ),
            "Q6. Lasso at its cross-validated penalty dropped w/b. Is w/b irrelevant?": mo.md(
                r"""
                No. Under a strong penalty lasso admits $w/b$ first. At the CV penalty it drops
                $w/b$ on 12 of the 30 splits, and the dropped set changes from split to split
                (22 different sets): with correlated features which one lasso keeps is unstable
                (Zou & Hastie, 2005). A zero weight means "not needed given the others, in this
                sample", not "irrelevant".
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **`np.linalg.inv(X.T @ X)`.** Squares the condition number and fails silently on a
      dependent column (worse than predicting the mean here, with no error raised). Use
      `lstsq`, QR, or a library.
    - **Unpivoted QR on rank-deficient data.** `np.linalg.qr` does not pivot; a round-off-sized
      pivot on $R$'s diagonal gives weights of order $10^{12}$. Check the rank first.
    - **Quoting round-off.** On a rank-deficient problem the inverse's and QR's digits change
      from machine to machine; report and test their size, never their digits.
    - **Interpreting single weights** when inputs are collinear, or when a ratio and its parts
      are both in the model.
    - **Ridge or lasso without standardizing**, or with a scaler fit on all data instead of the
      training rows. Also: penalizing the intercept.
    - **Choosing $\lambda$ on the test set.** Cross-validate inside the training split, with the
      same grouping (designs) as the outer split.
    - **Back-transforming a log target** with `exp` gives a median, not a mean; correct it
      with Duan's smearing factor. Here the log target lost for another reason, its functional
      form: 9.01 MPa with `exp`, 9.17 with smearing, against 7.55 in MPa.
    - **Trusting a prediction outside every feature's range**, or reporting it without a margin.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §10.2.1–10.2.2:** the linear regression model and two routes to least squares,
      a probabilistic one (normal noise, maximum likelihood) and a direct one (make the
      unmodelled effects small); both end at the normal equations. Forsyth writes $\beta$ for
      our $\mathbf{w}$ and averages the squared error over $N$.
    - **Forsyth §10.3.2:** the hat matrix $H = X(X^\top X)^{-1}X^\top$ and leverage; its
      Exercise 10.8 shows $H = UU^\top$ from the SVD, the projection view we skipped on the
      slides.
    - **Forsyth §10.4.2:** why correlated explanatory variables give $X^\top X$ small
      eigenvalues, and ridge as the fix, with $\lambda$ chosen by cross-validation. Because his
      error is averaged over $N$, his $\lambda$ corresponds to our $\lambda/N$.
    - **Forsyth §11.4–11.4.1:** $L_1$ regularization and sparse models: why a quadratic penalty
      never forces a weight to exactly zero and an $L_1$ penalty does.
    - **Exercise.** Forsyth 10.6 (correlation forces a small eigenvalue of $X^\top X$ and a poor
      estimate) and 10.7 (ridge lifts every eigenvalue by $\lambda$): work both with the binder
      column of §5.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Least squares: set the gradient to zero, then solve with QR or the SVD, never the inverse.
    - $\ln$ age cut the error from 10.6 to 7.4 MPa; beyond the tested cement, Abrams' ratio kept
      the errors on the safe side.
    - A dependent column makes single weights meaningless; the SVD and ridge still answer.
    - Standardize before ridge or lasso; leave the intercept unpenalized.
    - Delegate only what you can check: range, units, and sign.

    **Next:** linear classifiers, and screening bridges when a miss costs more than a false
    alarm.
    """)
    return


if __name__ == "__main__":
    app.run()
