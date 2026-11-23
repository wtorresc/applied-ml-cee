"""Applied Machine Learning for CEE — Lecture 2: Nearest Neighbors and Representation.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l02_nearest_neighbors.py

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
    # Applied Machine Learning for CEE — Lecture 2: Nearest Neighbors and Representation

    **Slides:** *Applied Machine Learning for CEE*, Lecture 2 (`lectures/cee_l02_nearest_neighbors`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §1.2 (nearest neighbors) and §1.1.3 (overfitting and cross-validation).
    **Data:** UCI Concrete Compressive Strength (Yeh, 1998), 1,030 laboratory tests; fetch it with `bash scripts/fetch_concrete.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`.

    **Your tasks** (from the slides): build the encoder and `knn_regress`, match scikit-learn,
    work the four mixes by hand, and reproduce the leakage and extrapolation tests.
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
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    K = 5  # neighbors used throughout, as on the slides
    REPEATS = 30  # repeated 75/25 splits for every comparison
    FC = 35.0  # specified 28-day strength f'c, MPa
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        FC,
        GroupShuffleSplit,
        K,
        KNeighborsRegressor,
        MARKERS,
        PALETTE,
        Path,
        REPEATS,
        SEED,
        ShuffleSplit,
        StandardScaler,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions from Lecture 1

    On your own, no notes, two minutes:

    1. Which part of the data may you look at only once, and why only once?
    2. A model ranks bridges for inspection from their records. Name its input, its output, and
       the decision it feeds.
    3. Name one thing you would need before signing a decision based on a model's output.

    **Today:** will a proposed concrete mix reach its specified strength?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. The test set, once, at the end. Every look that changes the model makes its
                   score optimistic.
                2. Input: age, materials, traffic, past condition ratings. Output: a condition
                   rating, or the chance of poor condition. Decision: which bridges to inspect or
                   repair first.
                3. Any of: the error on cases like this one; whether this case lies inside the
                   data; what an error costs, and who carries it; who is accountable for the
                   decision.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A producer proposes a mix for a slab specified at $f'_c = 35$ MPa at 28 days (about
    5,000 psi). Test cylinders take 28 days to answer. **Can the tests we already have answer
    today?**

    The two errors are not equal. A wrong **"yes"** puts weak concrete in place, followed by
    cores, strengthening, or removal; the owner and the public carry that risk. A wrong **"no"**
    means more cement than needed (money and carbon) or a redesign. We will come back to this in
    §6, after we know how good the predictions are.

    **The data.** 1,030 laboratory tests: seven ingredients in kg per m³ of concrete (cement,
    blast-furnace slag, fly ash, water, superplasticizer, coarse and fine aggregate), the age in
    days, and the compressive strength in MPa. One test is a row, but one mix can be many rows:
    the same mix is tested at several ages, and the compiled records repeat some designs with
    rounded proportions. We give every row two ids:

    - `mix`: identical proportions (428 mixes);
    - `design`: all seven proportions within 1 kg/m³ of each other (315 designs).
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
def _(find_concrete_csv, load_concrete, pd):
    df = load_concrete(find_concrete_csv())
    _per_mix = df.groupby("mix").size()
    _summary = {
        "tests": len(df),
        "mixes (identical proportions)": df["mix"].nunique(),
        "designs (within 1 kg/m³)": df["design"].nunique(),
        "distinct ages": df["age"].nunique(),
        "tests at 28 days": int((df["age"] == 28).sum()),
        "mixes tested once": int((_per_mix == 1).sum()),
        "strength range (MPa)": f"{df['strength'].min():.1f} to {df['strength'].max():.1f}",
    }
    # Slide "The data": 1,030 tests, 428 mixes, 14 ages, 425 at 28 days, 246 tested once.
    assert (_summary["tests"], _summary["mixes (identical proportions)"]) == (1030, 428)
    assert _summary["designs (within 1 kg/m³)"] == 315
    assert (_summary["distinct ages"], _summary["tests at 28 days"]) == (14, 425)
    assert _summary["mixes tested once"] == 246
    pd.Series(_summary, name="value").to_frame()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    **Helpers used in every section.** The next two cells define the encoder `mix_groups`
    (derived in §3, explained in §4), the two ways of splitting the tests, and the scikit-learn
    fit. They come this early because the attempt below already uses them; skip them on a first
    read.
    """)
    return


@app.cell
def _(np, pd):
    def mix_groups(df: pd.DataFrame) -> np.ndarray:
        """Dimensionless descriptors of each mix, plus log age (the slide's listing).

        Args:
            df: Concrete data with the ingredient columns and ``age``.

        Returns:
            Array of shape ``(len(df), 6)``: w/b, slag/b, fly ash/b, SP/b, aggregate/b, ln age.
        """
        b = df.cement + df.slag + df.fly_ash  # binder, kg/m³
        z = np.column_stack(
            [
                df.water / b,  # Abrams: w/b
                df.slag / b,
                df.fly_ash / b,  # binder makeup
                df.superplasticizer / b,
                (df.coarse_agg + df.fine_agg) / b,  # aggregate-binder
                np.log(df.age),  # gain ~ log time
            ]
        )
        assert z.shape == (len(df), 6), z.shape
        return z

    return (mix_groups,)


@app.cell
def _(GroupShuffleSplit, KNeighborsRegressor, REPEATS, SEED, ShuffleSplit, StandardScaler, np, pd):
    def grouped_splits(df: pd.DataFrame, by: str = "design") -> list[tuple[np.ndarray, np.ndarray]]:
        """``REPEATS`` 75/25 splits that keep every ``by`` group entirely in train or in test."""
        gss = GroupShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(gss.split(df, groups=df[by]))

    def random_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """``REPEATS`` 75/25 splits of the tests, ignoring which mix each test belongs to."""
        ss = ShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(ss.split(df))

    def knn_fit(
        X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int, scale: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """scikit-learn k-NN regression; scaler fit on ``tr`` only.

        Args:
            X: Features, shape ``(N, D)``.
            y: Strengths, shape ``(N,)``.
            tr: Training rows.
            te: Test rows.
            k: Number of neighbors.
            scale: Standardize with training statistics.

        Returns:
            Predictions, the neighbors' spread (sample standard deviation, ddof = 1; NaN for
            k = 1, where it is undefined), and neighbor row indices ``(len(te), k)``.
        """
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        _, idx = KNeighborsRegressor(n_neighbors=k).fit(Xtr, y[tr]).kneighbors(Xte)
        nbr = y[tr][idx]
        spread = nbr.std(axis=1, ddof=1) if k > 1 else np.full(len(te), np.nan)
        return nbr.mean(axis=1), spread, tr[idx]

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square difference of two arrays of the same shape.

        Args:
            a: Predictions.
            b: Targets, the same shape as ``a``.

        Returns:
            sqrt(mean((a - b)^2)).

        Raises:
            ValueError: If the shapes differ: (n,) against (n, 1) would broadcast to n x n.
        """
        if np.shape(a) != np.shape(b):
            raise ValueError(f"rmse of different shapes: {np.shape(a)} and {np.shape(b)}")
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return grouped_splits, knn_fit, random_splits, rmse


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: estimate the proposed mix

    Four tested mixes, A to D, and the proposed mix, all at 28 days. On your own, two minutes:
    **estimate the proposed mix's 28-day strength, and write down which mixes you relied on, and
    why.** Do it before reading on; later sections return to your answer.
    """)
    return


@app.cell(hide_code=True)
def _(FC, INGREDIENTS, K, df, grouped_splits, knn_fit, mix_groups, np, pd):
    # Same selection as the slides (figure script, `attempt_mixes`). The code is hidden in the
    # editor because it uses the neighbors, which would give the answer away.
    _y = df["strength"].to_numpy()
    _tr, _te = grouped_splits(df)[0]
    _t28 = _te[df["age"].to_numpy()[_te] == 28]
    # The proposed mix: the 28-day test of the first design-held-out split closest to f'c.
    trace_row = int(_t28[np.argmin(np.abs(_y[_t28] - FC))])
    _keys = [*INGREDIENTS, "age"]
    _raw = df[_keys].to_numpy(np.float64)
    _q = np.array([trace_row])
    _raw_nb = knn_fit(_raw, _y, _tr, _q, K, False)[2][0]
    _grp_nb = knn_fit(mix_groups(df), _y, _tr, _q, K, True)[2][0]

    # One row per design among the 5 nearest under either encoder, except the raw nearest
    # design: both encoders find it, so it would give the answer away.
    _design = df["design"].to_numpy()
    assert _design[_raw_nb[0]] in _design[_grp_nb], "groups miss the raw nearest design"
    _chosen: dict[int, int] = {}  # design id -> first row met (raw neighbors first)
    for _i in [*_raw_nb, *_grp_nb]:
        if _design[_i] != _design[_raw_nb[0]] and _design[_i] not in _chosen:
            _chosen[int(_design[_i])] = int(_i)
    assert len(_chosen) == 4, _chosen

    # Letter the four by raw distance to the proposed mix, nearest first.
    _rows = np.array(list(_chosen.values()))
    _dist = np.sqrt(((_raw[_rows] - _raw[trace_row]) ** 2).sum(axis=1))
    _rows = _rows[np.argsort(_dist, kind="stable")]
    _mix, _age = df["mix"].to_numpy(), df["age"].to_numpy()
    _wb = mix_groups(df)[:, 0]
    attempt = (
        df.iloc[_rows][_keys]
        .assign(
            **{
                "row": _rows,
                # every training test of the same mix at the same age
                "strengths": [
                    sorted(_y[_tr[(_mix[_tr] == _mix[_i]) & (_age[_tr] == _age[_i])]])
                    for _i in _rows
                ],
                "w/b": _wb[_rows],
            }
        )
        .set_axis(pd.Index(list("ABCD"), name="mix"))
    )

    # Slide "Your turn" and its notes: rows 524, 341, 1018, 336 (A–D), proposed row 913;
    # nearest in kilograms A, 61.2 MPa; matched on w/b B, C, D, 45.3 MPa; measured 35.9 MPa.
    _matched = [_l for _l in attempt.index if abs(attempt.at[_l, "w/b"] - _wb[trace_row]) <= 0.02]
    _mean_raw = float(np.mean(attempt.at["A", "strengths"]))
    _mean_wb = float(np.mean([_v for _l in _matched for _v in attempt.at[_l, "strengths"]]))
    assert list(attempt["row"]) == [524, 341, 1018, 336] and trace_row == 913
    assert (attempt["age"] == 28).all() and df.at[trace_row, "age"] == 28
    assert _matched == ["B", "C", "D"]
    assert [round(float(_v), 1) for _v in attempt.at["A", "strengths"]] == [59.5, 62.9]
    assert (round(_mean_raw, 1), round(_mean_wb, 1)) == (61.2, 45.3)
    assert round(float(_y[trace_row]), 1) == 35.9
    # Slide "One picture": training stores 746 tests of 236 designs.
    assert (len(_tr), df["design"].iloc[_tr].nunique()) == (746, 236)

    _names = {
        "cement": "Cement",
        "slag": "Slag",
        "fly_ash": "Fly ash",
        "water": "Water",
        "superplasticizer": "SP",
        "coarse_agg": "Coarse",
        "fine_agg": "Fine",
    }
    _table = pd.concat(
        [attempt[INGREDIENTS], df.iloc[[trace_row]][INGREDIENTS].set_axis(["Proposed"])]
    )
    _table = _table.apply(lambda _col: _col.map("{:.1f}".format)).rename(columns=_names)
    _table["MPa at 28 days"] = [
        ", ".join(f"{_v:.1f}" for _v in _s) for _s in attempt["strengths"]
    ] + ["?"]
    _table.index.name = "Mix (kg/m³)"
    _table
    return attempt, trace_row


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer: two common estimates, and the measured strength": mo.md(
                r"""
                - **Matched on kilograms:** A is the nearest mix in raw units, tested twice
                  (59.5 and 62.9 MPa). Estimate: **61.2 MPa**.
                - **Matched on the water–binder ratio**, as Abrams' law suggests: B, C, and D
                  have $w/b$ 0.40 to 0.42, like the proposed mix (0.40). Estimate: the mean of
                  47.4, 37.3, and 51.3, **45.3 MPa**.
                - **Measured: 35.9 MPa**, just above $f'_c = 35$ MPa.

                Keep your own estimate and the mixes you used: §3 asks why the kilograms point
                to A, and §4 shows which mixes k-NN picks.
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

    > mix $\mathbf{x}$ → **encoder** (dimensionless groups) → $\mathbf{z}$ → **decoder**
    > (average strength of the $k$ nearest training tests) → $\hat{y}$ → loss $(\hat{y}-y)^2$

    The decoder has no weights to fit: "training" stores the tests (746 tests of 236 designs in
    the first design-held-out split). All the engineering is in the encoder, that is, in what we
    decide "similar" means. We follow the proposed mix from your attempt, a held-out design:
    cement 298, fly ash 107, water 164, superplasticizer 13, coarse 953, fine 784 kg/m³, tested
    at 28 days, measured 35.9 MPa. In §4 we trace it through this pipeline by hand.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** similar mixes, tested at similar ages, reach similar strengths.

    **The rule.** For an encoded mix $\mathbf{z}$, average the strengths of its $k$ nearest
    training tests:
    $$\hat{y}(\mathbf{z})=\frac{1}{k}\sum_{i\in\mathcal{N}_k(\mathbf{z})}y_i,
    \qquad \mathcal{N}_k(\mathbf{z}) = \text{the } k \text{ training } \mathbf{z}_i
    \text{ closest to } \mathbf{z} \text{ in } \lVert\mathbf{z}-\mathbf{z}_i\rVert_2 .$$
    "Training" only stores the tests; each prediction costs $O(ND)$ by brute force. Everything
    rests on $\mathbf{z}$: what makes two mixes similar?

    **What a raw distance listens to.** For two independent tests $\mathbf{x}$ and $\mathbf{u}$
    from the same distribution,
    $$\mathbb{E}\,(x_j-u_j)^2 = \mathrm{Var}(x_j)+\mathrm{Var}(u_j)-2\,\mathrm{Cov}(x_j,u_j)
    = 2\,\mathrm{Var}(x_j),$$
    because independence makes the covariance zero. So input $j$ contributes the share
    $\mathrm{Var}(x_j)/\sum_l \mathrm{Var}(x_l)$ of the expected squared distance. The cell
    below computes the shares and checks the identity on random pairs of tests.
    """)
    return


@app.cell
def _(INGREDIENTS, df, np, pd):
    _X = df[[*INGREDIENTS, "age"]].to_numpy(np.float64)
    _var = _X.var(axis=0)
    _share = pd.Series(100 * _var / _var.sum(), index=[*INGREDIENTS, "age"])  # unrounded

    # Monte Carlo check of E(x_j - u_j)^2 = 2 Var(x_j) on 200,000 random pairs of tests.
    _rng = np.random.default_rng(0)
    _i, _j = _rng.integers(0, len(_X), size=(2, 200_000))
    _mc = ((_X[_i] - _X[_j]) ** 2).mean(axis=0)
    assert np.allclose(_mc, 2 * _var, rtol=0.05), _mc / (2 * _var)
    # Slide "Raw kilograms hide the water": water 1.2%, cement 27.7%; notes: aggregates 31.6%.
    assert (round(_share["water"], 1), round(_share["cement"], 1)) == (1.2, 27.7)
    assert round(_share["coarse_agg"] + _share["fine_agg"], 1) == 31.6
    _share.sort_values(ascending=False).round(1).to_frame("share of raw squared distance (%)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked example: the four mixes of your attempt

    Water is 1.2% of the raw distance on average. Check it on A to D: compute the raw distance
    from each to the proposed mix (all at 28 days, so age adds nothing) and the share of its
    squared distance that comes from water,
    $$d=\lVert\mathbf{x}-\mathbf{x}_{\text{proposed}}\rVert_2,\qquad
    \text{water share}=\frac{(x_{\text{water}}-x_{\text{water, proposed}})^2}{d^2}.$$
    Then compare the order in kilograms with the order in $w/b$.
    """)
    return


@app.cell
def _(INGREDIENTS, MARKERS, PALETTE, attempt, df, mix_groups, np, pd, plt, trace_row):
    _keys = [*INGREDIENTS, "age"]
    _diff = attempt[_keys].to_numpy(np.float64) - df.iloc[trace_row][_keys].to_numpy(np.float64)
    assert _diff.shape == (4, len(_keys)), _diff.shape
    _d2 = (_diff**2).sum(axis=1)
    _wb0 = float(mix_groups(df.iloc[[trace_row]])[0, 0])
    _example = pd.DataFrame(
        {
            "raw distance (kg/m³)": np.sqrt(_d2),
            "water share of d² (%)": 100 * _diff[:, _keys.index("water")] ** 2 / _d2,
            "w/b": attempt["w/b"],
            "|w/b − 0.40|": (attempt["w/b"] - _wb0).abs(),
        },
        index=attempt.index,
    )
    # Slide "Raw kilograms hide the water" and its notes: A is the nearest in
    # kilograms but the farthest in w/b; water is 1.7% of its squared distance.
    _dist = [round(float(_v), 1) for _v in _example["raw distance (kg/m³)"]]
    assert _dist == [75.8, 77.8, 90.7, 104.5], _dist
    assert [round(float(_v), 1) for _v in _example["water share of d² (%)"]] == [1.7, 1.9, 4.2, 0.2]
    assert [round(float(_v), 2) for _v in _example["w/b"]] == [0.30, 0.42, 0.41, 0.40]
    assert _example["|w/b − 0.40|"].idxmax() == "A"

    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.axhline(_wb0, color="#595959", ls="--", lw=1.2)
    _ax.text(66, _wb0 - 0.012, f"proposed mix, w/b = {_wb0:.2f}", color="#595959", va="top")
    for _j, _l in enumerate(attempt.index):
        _x, _w = float(_example.at[_l, "raw distance (kg/m³)"]), float(attempt.at[_l, "w/b"])
        _ax.plot(_x, _w, marker=MARKERS[_j], ls="none", ms=9, color=PALETTE[_j])
        _mpa = ", ".join(f"{_v:.1f}" for _v in attempt.at[_l, "strengths"])
        _below = _w <= _wb0  # keep labels off the dashed line
        _ax.annotate(
            f"{_l}: {_mpa} MPa",
            (_x, _w),
            xytext=(9, -8 if _below else 4),
            textcoords="offset points",
            va="top" if _below else "baseline",
        )
    _ax.set(
        xlim=(65, 125),
        ylim=(0.27, 0.45),
        xlabel="raw distance to the proposed mix (kg/m³)",
        ylabel="water–binder ratio",
    )
    _fig.tight_layout()
    (
        _example.round(
            {"raw distance (kg/m³)": 1, "water share of d² (%)": 1, "w/b": 2, "|w/b − 0.40|": 2}
        ),
        _fig,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Check:** add 30 kg/m³ of water to the proposed mix. **Near or far?** Answer twice, once in
    raw units and once as an engineer, and write both down. The answer comes after Abrams' law,
    below.

    **Standardizing fixes the units, not the meaning.** With $z_j=(x_j-\mu_j)/\sigma_j$ from
    the training split, every input counts equally. That is still arbitrary: strength depends on
    *proportions*, and a day of curing matters far more at age 3 than at age 300.

    **What an engineer already knows.** Abrams (1918) found that strength falls exponentially as
    the water–cement ratio rises, $f_c = A/B^{\,w/c}$, so $\ln f_c = \ln A - (w/c)\ln B$ is linear
    in the ratio. With slag and fly ash in the binder $b$ = cement + slag + fly ash, the
    water–binder ratio $w/b$ is the natural group. And strength gain is roughly linear in
    $\ln(\text{age})$. The next cell fits Abrams' law to the 28-day tests and measures the gain per
    unit of log age within each mix tested at four or more ages.
    """)
    return


@app.cell
def _(MARKERS, PALETTE, df, np, plt):
    _b = df["cement"] + df["slag"] + df["fly_ash"]
    _wb = (df["water"] / _b).to_numpy()
    _wc = (df["water"] / df["cement"]).to_numpy()
    _d28 = (df["age"] == 28).to_numpy()
    _logf = np.log(df["strength"].to_numpy())
    _slope, _icpt = np.polyfit(_wb[_d28], _logf[_d28], 1)
    abrams_A, abrams_B = float(np.exp(_icpt)), float(np.exp(-_slope))
    _r2_wb = np.corrcoef(_wb[_d28], _logf[_d28])[0, 1] ** 2
    _r2_wc = np.corrcoef(_wc[_d28], _logf[_d28])[0, 1] ** 2
    _ages = df.groupby("mix")["age"].nunique()
    _gain = [
        np.polyfit(np.log(_s["age"]), _s["strength"], 1)[0]
        for _m, _s in df.groupby("mix")
        if _ages[_m] >= 4
    ]
    # Slide "Strength follows w/b and log age" (notes): A = 123.8, B = 16.5, R^2 0.60 vs 0.42,
    # 133 mixes, 8.7.
    assert (round(abrams_A, 1), round(abrams_B, 1)) == (123.8, 16.5)
    assert (round(_r2_wb, 2), round(_r2_wc, 2)) == (0.60, 0.42)
    assert (len(_gain), round(float(np.median(_gain)), 1)) == (133, 8.7)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax1.scatter(
        _wb[_d28],
        df["strength"][_d28],
        s=14,
        alpha=0.6,
        color=PALETTE[0],
        marker=MARKERS[0],
        label="28-day tests",
    )
    _grid = np.linspace(_wb[_d28].min(), _wb[_d28].max(), 200)
    _ax1.plot(
        _grid,
        abrams_A / abrams_B**_grid,
        "--",
        color=PALETTE[1],
        label=f"$f = {abrams_A:.0f} / {abrams_B:.1f}^{{w/b}}$",
    )
    _ax1.set(
        xlabel="water–binder ratio w/b",
        ylabel="strength (MPa)",
        title=f"28 days: R² of ln f on w/b = {_r2_wb:.2f} (w/c: {_r2_wc:.2f})",
    )
    _ax1.legend()
    # A stable sort: three mixes tie at 7 ages, and the legend must not depend on the library.
    for _c, _m in enumerate(_ages.sort_values(ascending=False, kind="stable").index[:4]):
        _s = df[df["mix"] == _m].groupby("age")["strength"].mean()
        _ax2.plot(_s.index, _s.values, marker=MARKERS[_c], color=PALETTE[_c], label=f"mix {_c + 1}")
    _ax2.set(xscale="log", xlabel="age (days, log scale)", title="Same mix, several ages")
    _ax2.legend(ncol=2)
    _fig.tight_layout()
    _fig
    return abrams_A, abrams_B


@app.cell(hide_code=True)
def _(FC, INGREDIENTS, abrams_A, abrams_B, df, grouped_splits, mo, np, trace_row):
    # Numbers for the check question (figure script, `water_check`). The code is hidden in the
    # editor because its asserts give the answer away.
    _extra = 30.0  # kg/m³ of water added to the proposed mix, nothing else changes
    _r = df.iloc[trace_row]
    _b = float(_r.cement + _r.slag + _r.fly_ash)
    _wb0, _wb1 = float(_r.water) / _b, (float(_r.water) + _extra) / _b
    _f0, _f1 = abrams_A / abrams_B**_wb0, abrams_A / abrams_B**_wb1
    # Raw distance from the proposed mix to the nearest training test of the first split.
    _tr, _ = grouped_splits(df)[0]
    _raw = df[[*INGREDIENTS, "age"]].to_numpy(np.float64)
    _nearest = float(np.sqrt(((_raw[_tr] - _raw[trace_row]) ** 2).sum(axis=1)).min())
    # Slide notes: nearest tested mix 33.1 kg/m³ away; w/b 0.40 to 0.48; 39.8 to 32.4 MPa,
    # from above f'c to below it.
    assert _extra < _nearest and round(_nearest, 1) == 33.1
    assert (round(_wb0, 2), round(_wb1, 2)) == (0.40, 0.48)
    assert (round(_f0, 1), round(_f1, 1)) == (39.8, 32.4) and _f1 < FC < _f0
    mo.accordion(
        {
            "Check: add 30 kg/m³ of water to the proposed mix. Near or far?": mo.md(
                r"""
                **Near, in raw units.** Only the water changes, so the changed mix is
                30 kg/m³ from the proposed one, nearer than any tested mix (the nearest is
                33.1 kg/m³ away). To a raw distance the two look almost identical.

                **Far, to an engineer.** $w/b$ goes from 0.40 to 0.48, and the fitted Abrams
                curve drops from 39.8 to 32.4 MPa: from above $f'_c = 35$ MPa to below it. The
                encoder has to see what the raw distance cannot.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **The encoder.** Describe each mix by ratios to the binder: $w/b$ (Abrams' ratio), slag$/b$
    and fly ash$/b$ (what the binder is made of), superplasticizer$/b$, aggregate$/b$, and
    $\ln(\text{age})$. Eight inputs with units become six dimensionless numbers; we standardize
    them with training statistics. This is dimensional analysis used as an encoder (Buckingham,
    1914): similar groups, similar behavior. The listing is `mix_groups`, in the helper cell of
    §1, where the attempt already used it.

    **The decoder** is a distance, a sort, and a mean: `knn_regress`, from scratch.
    """)
    return


@app.cell
def _(np):
    def knn_regress(
        Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and spread of the k nearest training strengths (the slide's listing).

        Args:
            Z_tr: Encoded training tests, shape ``(N, D)``.
            y_tr: Training strengths, shape ``(N,)``.
            Z_te: Encoded queries, shape ``(M, D)``.
            k: Number of neighbors (at least 2, for the spread).

        Returns:
            Mean and sample standard deviation (ddof = 1) of the k neighbors' strengths, each
            shape ``(M,)``.
        """
        d2 = (Z_te**2).sum(1)[:, None] - 2 * Z_te @ Z_tr.T + (Z_tr**2).sum(1)[None, :]
        assert d2.shape == (len(Z_te), len(Z_tr))
        nn = np.argsort(d2, axis=1, kind="stable")[:, :k]  # k nearest
        assert nn.shape == (len(Z_te), k)
        return y_tr[nn].mean(axis=1), y_tr[nn].std(axis=1, ddof=1)

    return (knn_regress,)


@app.cell
def _(mo):
    mo.md(r"""
    **Predict.** On the 284 held-out tests of the first design-held-out split, with $k = 5$, will
    `knn_regress` match scikit-learn's `KNeighborsRegressor` on every test? On almost every test?
    Or will they often differ? Write your answer down, then **run** the next cell. It compares the
    two to $10^{-9}$ at $k = 2, 3, 5, 9, 15$ (from 2: the spread, a sample standard deviation,
    needs two neighbors).
    """)
    return


@app.cell
def _(K, KNeighborsRegressor, StandardScaler, df, grouped_splits, knn_regress, mix_groups, np, pd):
    _y = df["strength"].to_numpy()
    tr0, te0 = grouped_splits(df)[0]
    _sc = StandardScaler().fit(mix_groups(df.iloc[tr0]))
    Z_tr0, Z_te0 = _sc.transform(mix_groups(df.iloc[tr0])), _sc.transform(mix_groups(df.iloc[te0]))
    _rows, _agree_at = [], {}
    for _k in (2, 3, 5, 9, 15):
        _mean, _ = knn_regress(Z_tr0, _y[tr0], Z_te0, _k)
        _ref = (
            KNeighborsRegressor(n_neighbors=_k, algorithm="brute")
            .fit(Z_tr0, _y[tr0])
            .predict(Z_te0)
        )
        _exact = np.sort(((Z_te0[:, None, :] - Z_tr0[None, :, :]) ** 2).sum(-1), axis=1)
        _tie = np.isclose(_exact[:, _k - 1], _exact[:, _k], rtol=0, atol=1e-9)
        _agree_at[_k] = np.isclose(_mean, _ref, rtol=0, atol=1e-9)
        assert _agree_at[_k][~_tie].all(), f"k={_k}: disagreement away from ties"
        _rows.append(
            {
                "k": _k,
                "test rows": len(te0),
                "rows with a tie": int(_tie.sum()),
                "rows that differ": int((~_agree_at[_k]).sum()),
            }
        )
    differs_k5 = ~_agree_at[K]
    _check = pd.DataFrame(_rows).set_index("k")
    # Slide "Raw units pick the wrong neighbors": 284 tests, 8 with a tie, 1 differs.
    assert (len(te0), _check.at[K, "rows with a tie"], int(differs_k5.sum())) == (284, 8, 1)
    _check
    return Z_te0, Z_tr0, differs_k5, te0, tr0


@app.cell
def _(mo):
    mo.md(r"""
    **Result.** At $k = 5$, 283 of the 284 tests match to $10^{-9}$; the other has a tie for the
    5th neighbor: its 5th and 6th nearest training records are exactly equally far away. 8 of
    the 284 tests have such a tie, and which tied record counts as the 5th is then arbitrary in
    both codes. For one test the two codes pick differently, and the mean changes.

    **Investigate.** Why would two records be exactly equally far from a test? The next cell
    lists the 8 ties, then the seven nearest training records of the test that differs.
    """)
    return


@app.cell
def _(INGREDIENTS, K, Z_te0, Z_tr0, df, differs_k5, mix_groups, np, pd, te0, tr0):
    _d2 = ((Z_te0[:, None, :] - Z_tr0[None, :, :]) ** 2).sum(-1)
    assert _d2.shape == (len(te0), len(tr0)), _d2.shape
    _order = np.argsort(_d2, axis=1, kind="stable")
    _sorted = np.take_along_axis(_d2, _order, axis=1)
    _tie = np.isclose(_sorted[:, K - 1], _sorted[:, K], rtol=0, atol=1e-9)
    _tests, _r5, _r6 = te0[_tie], tr0[_order[_tie, K - 1]], tr0[_order[_tie, K]]
    _g = mix_groups(df)  # unscaled groups: w/b, slag/b, fly ash/b, SP/b, aggregate/b, ln age
    _design = df["design"].to_numpy()
    _same_design = _design[_r5] == _design[_r6]
    _same_groups = np.isclose(_g[_r5], _g[_r6]).all(axis=1)
    _series = ~_same_design & ~_same_groups
    _mid = np.isclose(_g[_r5] + _g[_r6], 2 * _g[_tests])  # test halfway between the two, per group
    # 4 ties: one design at two ages equally far in ln age; 3: a series stepping slag by 10% of
    # the binder, the test's design in the middle; 1: two designs with identical groups (the
    # test where the codes differ). None is one design and age entered twice.
    assert (_same_design.sum(), _series.sum(), _same_groups.sum()) == (4, 3, 1)
    assert _mid[_same_design, 5].all() and _mid[_series, 1].all()
    assert np.allclose(np.abs(_g[_r5, 1] - _g[_r6, 1])[_series], 0.2)
    assert list(_tests[_same_groups]) == list(te0[differs_k5])
    _a, _b = int(_r5[_same_groups][0]), int(_r6[_same_groups][0])
    assert df.at[_a, "coarse_agg"] != df.at[_b, "coarse_agg"]
    # Slide notes: test row 910; its two tied records measured 25.6 and 26.2 MPa, with coarse/fine
    # aggregate 1,002/689 against 867/824 kg/m³.
    assert list(_tests[_same_groups]) == [910]
    assert [round(float(df.at[_x, "strength"]), 1) for _x in (_a, _b)] == [25.6, 26.2]
    assert [df.at[_x, _c] for _x in (_a, _b) for _c in ("coarse_agg", "fine_agg")] == [
        1002.0,
        689.0,
        867.0,
        824.0,
    ]
    assert np.isclose(
        df.at[_a, "coarse_agg"] + df.at[_a, "fine_agg"],
        df.at[_b, "coarse_agg"] + df.at[_b, "fine_agg"],
    )

    _ties = pd.DataFrame(
        {
            "test row": _tests,
            "test age": df["age"].to_numpy()[_tests],
            "5th: design": _design[_r5],
            "5th: age": df["age"].to_numpy()[_r5],
            "6th: design": _design[_r6],
            "6th: age": df["age"].to_numpy()[_r6],
            "why equally far": np.select(
                [_same_design, _series],
                ["same design, ages equally far in ln age", "slag series, equal steps in slag/b"],
                "identical groups: only coarse/fine differ",
            ),
        }
    )
    _q = int(np.flatnonzero(differs_k5)[0])  # position of the differing test in te0
    _nbrs = (
        df.iloc[tr0[_order[_q, : K + 2]]][["design", *INGREDIENTS, "age", "strength"]]
        .assign(distance=np.sqrt(_sorted[_q, : K + 2]))
        .set_axis(pd.Index(np.arange(1, K + 3), name="rank"))
    )
    (_ties, _nbrs.round({"strength": 1, "distance": 4}))
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the ties are.** Each comes from structure in the data or in the encoder, and none is
    one design and age entered twice:

    - 4 are one design tested at two ages equally far in log age from the test: 14 and 56 days
      around a 28-day test ($\ln 2$ either way), or 7 and 28 around 14.
    - 3 come from a series that steps slag by 10% of the binder: the test's design sits in the
      middle, with one design on each side.
    - 1 is two designs the encoder cannot tell apart: same binder, water, superplasticizer, and
      total aggregate, split differently between coarse and fine (1,002/689 against
      867/824 kg/m³). Aggregate$/b$ sums the two, so their groups are identical, yet they measured
      25.6 and 26.2 MPa. This is the test where the two codes differ (ranks 5 and 6 above).

    "Matches except at ties" is only a check once you have counted the ties and looked at them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked example: the proposed mix, by hand

    This is the proposed mix from your attempt: the 28-day test in the first design-held-out
    split whose measured strength is closest to $f'_c = 35$ MPa. Compute its groups by hand
    first:

    - binder $b = 298 + 0 + 107 = 405$ kg/m³
    - $w/b = 164/405 = 0.405$; fly ash$/b = 107/405 = 0.264$; superplasticizer$/b = 13/405 = 0.032$
    - aggregate$/b = (953+784)/405 = 4.289$; $\ln 28 = 3.332$

    Then find its 5 nearest training tests with each encoder and average their strengths. The
    `mix` columns say which of your mixes A to D each neighbor is.
    """)
    return


@app.cell
def _(K, attempt, df, grouped_splits, knn_fit, mix_groups, np, pd, trace_row):
    _y = df["strength"].to_numpy()
    _tr, _ = grouped_splits(df)[0]
    _r = df.iloc[trace_row]
    _b = _r.cement + _r.slag + _r.fly_ash
    assert np.isclose(_b, 405.0) and np.isclose(_r.water / _b, 164 / 405)
    assert np.allclose(
        mix_groups(df.iloc[[trace_row]])[0],
        [164 / 405, 0, 107 / 405, 13 / 405, (953 + 784) / 405, np.log(28)],
    )

    _raw = df[
        ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
    ].to_numpy(np.float64)
    _pg, _sg, _ng = knn_fit(mix_groups(df), _y, _tr, np.array([trace_row]), K, True)
    _pr, _, _nr = knn_fit(_raw, _y, _tr, np.array([trace_row]), K, False)
    _wb = (df["water"] / (df["cement"] + df["slag"] + df["fly_ash"])).to_numpy()
    # Which attempt mix (A–D) each neighbor belongs to, by design.
    _letter = {int(df.at[_i, "design"]): _l for _l, _i in attempt["row"].items()}
    _design = df["design"].to_numpy()
    _raw_mix = [_letter.get(int(_d), "left out") for _d in _design[_nr[0]]]
    _grp_mix = [_letter.get(int(_d), "left out") for _d in _design[_ng[0]]]
    _trace_table = pd.DataFrame(
        {
            "raw: mix": _raw_mix,
            "raw: w/b": _wb[_nr[0]].round(2),
            "raw: MPa": _y[_nr[0]].round(1),
            "groups: mix": _grp_mix,
            "groups: w/b": _wb[_ng[0]].round(2),
            "groups: MPa": _y[_ng[0]].round(1),
        },
        index=pd.Index(range(1, K + 1), name="rank"),
    )
    # Slide "Raw units pick the wrong neighbors": raw 49.4 (+13.5), groups 42.6 ± 6.4 (+6.7; the
    # spread is the sample standard deviation), measured 35.9; raw picks A twice and B, the
    # groups pick D, C, and B.
    assert (round(float(_pr[0]), 1), round(float(_pg[0]), 1)) == (49.4, 42.6)
    assert (round(float(_sg[0]), 1), round(float(_y[trace_row]), 1)) == (6.4, 35.9)
    _err = (round(float(_pr[0] - _y[trace_row]), 1), round(float(_pg[0] - _y[trace_row]), 1))
    assert _err == (13.5, 6.7), _err
    assert list(_trace_table["raw: w/b"]) == [0.40, 0.40, 0.30, 0.30, 0.42]
    assert list(_trace_table["groups: w/b"]) == [0.40, 0.40, 0.40, 0.41, 0.42]
    assert _raw_mix == ["left out", "left out", "A", "A", "B"]
    assert _grp_mix == ["D", "left out", "left out", "C", "B"]
    _trace_table
    return


@app.cell
def _(
    K,
    KNeighborsRegressor,
    StandardScaler,
    abrams_A,
    abrams_B,
    df,
    grouped_splits,
    knn_fit,
    mix_groups,
    np,
    trace_row,
):
    # Slide "Groups find tests like the mix": of the proposed mix's 30 nearest training tests,
    # 15 are 28-day tests with raw units and 24 with the groups.
    _y, _age = df["strength"].to_numpy(), df["age"].to_numpy()
    _tr, _ = grouped_splits(df)[0]
    _raw = df[
        ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
    ].to_numpy(np.float64)
    _, _, _nr = knn_fit(_raw, _y, _tr, np.array([trace_row]), 30, False)
    _, _, _ng = knn_fit(mix_groups(df), _y, _tr, np.array([trace_row]), 30, True)
    assert (int((_age[_nr[0]] == 28).sum()), int((_age[_ng[0]] == 28).sum())) == (15, 24)

    # Slide "Similar mixes, similar strengths": 5-NN over w/b and log age, the proposed mix's
    # other groups fixed, is a terraced surface: 82% of the steps along w/b change nothing.
    _X = mix_groups(df)
    _sc = StandardScaler().fit(_X[_tr])
    _knn = KNeighborsRegressor(n_neighbors=K).fit(_sc.transform(_X[_tr]), _y[_tr])
    _W, _L = np.meshgrid(np.linspace(0.25, 0.80, 81), np.linspace(0.0, np.log(365.0), 81))
    _Z = np.repeat(_X[[trace_row]], _W.size, axis=0)
    _Z[:, 0], _Z[:, 5] = _W.ravel(), _L.ravel()
    _pred = _knn.predict(_sc.transform(_Z)).reshape(_W.shape)
    assert _pred.shape == (81, 81), _pred.shape
    assert round(float(1 - (np.abs(np.diff(_pred, axis=1)) > 1e-9).mean()), 2) == 0.82

    # Slide "The margin costs carbon or safety": binder for 35 MPa plus a margin from Abrams'
    # fit, water held at 164 kg/m³: 364, 407, 454, 507 kg/m³ at 0, 5, 10, 15 MPa, about 9 kg
    # per MPa.
    _binder = 164 * np.log(abrams_B) / np.log(abrams_A / (35 + np.array([0.0, 5.0, 10.0, 15.0])))
    assert [round(float(_b)) for _b in _binder] == [364, 407, 454, 507]
    assert round(float((_binder[2] - _binder[0]) / 10), 0) == 9
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Back to your attempt.** The raw distance cannot see water (1.2% of it), so it picks A twice
    ($w/b = 0.30$, much stronger) and B: the raw prediction is 49.4 MPa (error +13.5). The groups
    pick D, C, and B ($w/b$ 0.40 to 0.42) and predict 42.6 MPa (error +6.7); measured 35.9.
    Matching on $w/b$, as most engineers do, is what the groups encode. The 38.5 MPa records at
    ranks 1 and 2 (raw) and 2 and 3 (groups) are one more design, entered twice in the data and
    left out of the attempt because both encoders find it.

    **Bracket the error.** An error means little on its own. Above it: predicting the mean
    strength of the training tests. Below it: the scatter of the test itself. Some mixes were
    tested more than once at the same age and gave different strengths, and no model of these
    inputs can beat that scatter. Groups whose strengths are all identical are one record entered
    twice, so they are left out.
    """)
    return


@app.cell
def _(df):
    _stats = df.groupby(["mix", "age"])["strength"].agg(["size", "nunique", "std"])
    _rep = _stats[(_stats["size"] > 1) & (_stats["nunique"] > 1)]
    repeat_floor = float(_rep["std"].median())
    _per_mix = _rep.groupby(level="mix").size().sort_values(ascending=False)
    # Slide "Representation did most of the work": 9 groups from 3 mixes (two supply 8),
    # 20 tests, median SD 1.4 MPa (largest 16.5).
    assert (len(_rep), int(_rep["size"].sum())) == (9, 20)
    assert (len(_per_mix), int(_per_mix.iloc[:2].sum())) == (3, 8)
    assert (round(repeat_floor, 1), round(float(_rep["std"].max()), 1)) == (1.4, 16.5)
    _rep.drop(columns="nunique").rename(columns={"size": "tests", "std": "SD (MPa)"}).round(1)
    return (repeat_floor,)


@app.cell
def _(mo):
    mo.md(r"""
    The 9 groups come from only 3 mixes, and two of those mixes supply 8 of them, so the floor
    is an indication, not a measurement.

    **Delegate.** Our `knn_regress` matches scikit-learn away from ties, so the experiments below
    use `KNeighborsRegressor`, exactly as the figure script does. Same 5-NN, same 30 splits, whole
    designs held out. In the figure, the dotted line is predicting the mean, the dashed line the
    scatter of repeat tests.

    **Modify.** Add an encoder of your own to `_encoders` in the next cell, for example the raw
    inputs with $\ln(\text{age})$ but not standardized, or the groups without aggregate$/b$, and
    rerun. Where does it land between the two lines?
    """)
    return


@app.cell
def _(
    INGREDIENTS,
    K,
    MARKERS,
    PALETTE,
    df,
    grouped_splits,
    knn_fit,
    mix_groups,
    np,
    pd,
    plt,
    random_splits,
    repeat_floor,
    rmse,
):
    _y = df["strength"].to_numpy()
    _encoders = {
        "raw kg/m³ and days": (df[[*INGREDIENTS, "age"]].to_numpy(np.float64), False),
        "standardized": (df[[*INGREDIENTS, "age"]].to_numpy(np.float64), True),
        "standardized, ln age": (
            np.column_stack([df[INGREDIENTS].to_numpy(np.float64), np.log(df["age"])]),
            True,
        ),
        "dimensionless groups": (mix_groups(df), True),
    }
    _rows, _per_split = {}, {}
    for _name, (_X, _scale) in _encoders.items():
        _g = [rmse(knn_fit(_X, _y, _a, _t, K, _scale)[0], _y[_t]) for _a, _t in grouped_splits(df)]
        _s = [rmse(knn_fit(_X, _y, _a, _t, K, _scale)[0], _y[_t]) for _a, _t in random_splits(df)]
        _per_split[_name] = np.asarray(_g)
        _rows[_name] = {
            "designs held out": np.mean(_g),
            "sd": np.std(_g),
            "random split": np.mean(_s),
        }
    # Each step, paired split by split (same 30 splits): wins, and the mean and SD of the gain.
    _names = list(_encoders)
    _steps = {}
    for _i in range(1, len(_names)):
        _diff = _per_split[_names[_i - 1]] - _per_split[_names[_i]]  # > 0: the new one errs less
        _steps[f"{_names[_i - 1]} → {_names[_i]}"] = {
            "wins of 30": int((_diff > 0).sum()),
            "mean gain": float(_diff.mean()),
            "sd of gain": float(_diff.std(ddof=1)),
        }
    _paired = pd.DataFrame(_steps).T
    _mean_only = np.mean(
        [rmse(np.full(len(_t), _y[_a].mean()), _y[_t]) for _a, _t in grouped_splits(df)]
    )
    _raw_table = pd.DataFrame(_rows).T  # unrounded; round once, for display or a check
    _rep_table = _raw_table.round(2)
    # Slide "Representation did most of the work": 10.0, 9.5, 8.0, 6.9 (9.95, 9.51, 7.97, 6.87);
    # predicting the mean 17.0; the biggest step is ln age (9.5 to 8.0), then the groups.
    _deck = ["raw kg/m³ and days", "standardized", "standardized, ln age", "dimensionless groups"]
    _held = _raw_table.loc[_deck, "designs held out"]
    assert [round(float(_v), 2) for _v in _held] == [9.95, 9.51, 7.97, 6.87]
    assert [round(float(_v), 1) for _v in _held] == [10.0, 9.5, 8.0, 6.9]
    assert round(float(_mean_only), 1) == 17.0
    _gain = -_held.diff()
    assert list(_gain.sort_values(ascending=False).index[:2]) == _deck[2:]
    # Slide notes: paired, ln age beats standardized in 30 of 30 splits (1.54, SD 0.42); the
    # groups beat ln age in 29 of 30 (1.10, SD 0.57).
    _p_ln, _p_grp = _paired.iloc[1], _paired.iloc[2]
    assert (int(_p_ln["wins of 30"]), round(_p_ln["mean gain"], 2)) == (30, 1.54)
    assert round(_p_ln["sd of gain"], 2) == 0.42
    assert (int(_p_grp["wins of 30"]), round(_p_grp["mean gain"], 2)) == (29, 1.10)
    assert round(_p_grp["sd of gain"], 2) == 0.57

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _pos = np.arange(len(_rep_table))[::-1]
    _ax.errorbar(
        _rep_table["designs held out"],
        _pos,
        xerr=_rep_table["sd"],
        fmt=MARKERS[0],
        color=PALETTE[0],
        capsize=4,
    )
    _ax.axvline(_mean_only, ls=":", color="#595959", label=f"predict the mean ({_mean_only:.1f})")
    _ax.axvline(
        repeat_floor,
        ls="--",
        color="#595959",
        label=f"repeat tests, an indication ({repeat_floor:.1f})",
    )
    _ax.set_yticks(_pos, _rep_table.index)
    _ax.set(xlabel="test RMSE (MPa), 5-NN, designs held out", xlim=(0, 18))
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    (_rep_table, _paired.round(2), _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Representation did most of the work.** Raw 10.0, standardized 9.5, with
    $\ln(\text{age})$ 8.0, groups 6.9 MPa, between predicting the mean (17.0) above and the
    scatter of repeat tests (a median 1.4 MPa, 9 groups from 3 mixes; an indication) below.
    Both big steps came from engineering knowledge, not from the algorithm: log time (9.5 to
    8.0) and Abrams' ratio (8.0 to 6.9). 6.9 MPa leaves room for better models.

    The error bars overlap, but every encoder is scored on the same 30 splits, so compare them
    split by split (the second table): $\ln(\text{age})$ beats standardized in 30 of 30 splits,
    and the groups beat $\ln(\text{age})$ in 29 of 30. The four encoders were fixed in advance
    from engineering knowledge, not searched on these splits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    Each test below probes the assumption from §3, *similar mixes, tested at similar ages, reach
    similar strengths*: are the neighbors really similar, and is any tested mix similar at all?

    **Predict before we test.** Write down three answers.

    1. **Splits.** Random tests; identical mixes held out; whole designs held out. Rank their
       errors, lowest first. How far apart?
    2. **Ages.** Train only on tests up to 28 days, then predict the tests at 56 to 365 days. Too
       high, too low, or both?
    3. **Physics.** Raise only the water of one mix, step by step. Does the prediction always
       fall?

    Commit in writing; each subsection reveals one answer.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Splits: the same design in train and test

    A random split of the 1,030 tests puts most test rows next to their own mix, tested at another
    age. Holding out identical mixes is better but not enough: designs recur with rounded
    proportions (fine aggregate 806.0 vs 806.9 kg/m³). The cell measures how often a test's mix or
    design is already in training, and the error under three ways of splitting.
    """)
    return


@app.cell
def _(INGREDIENTS, K, df, grouped_splits, knn_fit, mix_groups, np, pd, random_splits, rmse):
    _y = df["strength"].to_numpy()
    _Z = mix_groups(df)
    # How often a test row's mix (random splits) or design (identical mixes held out) is also in
    # training: the share for each of the 30 splits, then its mean and range over the splits.
    _mix, _design = df["mix"].to_numpy(), df["design"].to_numpy()
    _seen_mix = np.array([np.isin(_mix[_t], _mix[_a]).mean() for _a, _t in random_splits(df)])
    _seen_design = np.array(
        [np.isin(_design[_t], _design[_a]).mean() for _a, _t in grouped_splits(df, "mix")]
    )
    _schemes = {
        "random split of tests": random_splits(df),
        "identical mixes held out": grouped_splits(df, "mix"),
        "designs held out": grouped_splits(df, "design"),
    }
    _leak = pd.Series(
        {
            _n: np.mean([rmse(knn_fit(_Z, _y, _a, _t, K, True)[0], _y[_t]) for _a, _t in _s])
            for _n, _s in _schemes.items()
        }
    )  # unrounded
    _gap = round(float(_leak["designs held out"] - _leak["random split of tests"]), 1)

    _keys = [*INGREDIENTS, "age"]
    _dup = df[df.duplicated(_keys, keep=False)].groupby(_keys)["strength"]
    # The same 28-day test recorded twice, rounded and unrounded (different `mix`, same design).
    _twin = df.loc[
        [862, 1019], ["cement", "slag", "fly_ash", "water", "age", "strength", "mix", "design"]
    ]
    assert _twin["strength"].nunique() == 1 and _twin["design"].nunique() == 1
    assert list(_twin["cement"]) == [140.0, 139.7]
    assert round(float(_twin["strength"].iloc[0]), 1) == 35.2
    # One design recorded with rounded and unrounded proportions: fine aggregate 806.0 and 806.9.
    _rounded = df[(df["cement"] == 349.0) & df["fine_agg"].round(1).isin([806.0, 806.9])]
    assert _rounded["mix"].nunique() == 2 and _rounded["design"].nunique() == 1
    assert sorted(_rounded["fine_agg"].round(1).unique()) == [806.0, 806.9]
    # Slide "Random splits flatter the model" (notes): over the 30 splits, 74% (67% to 80%) and
    # 22% (15% to 28%); 5.7 / 6.3 / 6.9 (5.65 / 6.27 / 6.87); gap 1.2; 19 repeated mix–age
    # pairs, 10 with identical strength.
    _shares = [
        round(float(_f(_v)), 2)
        for _v in (_seen_mix, _seen_design)
        for _f in (np.mean, np.min, np.max)
    ]
    assert _shares == [0.74, 0.67, 0.80, 0.22, 0.15, 0.28], _shares
    assert [round(float(_v), 2) for _v in _leak] == [5.65, 6.27, 6.87]
    assert [round(float(_v), 1) for _v in _leak] == [5.7, 6.3, 6.9] and _gap == 1.2
    assert (int(_dup.ngroups), int((_dup.nunique() == 1).sum())) == (19, 10)
    (_leak.round(2).to_frame("5-NN test RMSE (MPa)"), _twin)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 1: a random split flatters the model by 1.2 MPa.** Random 5.7, identical mixes held
    out 6.3, designs held out 6.9 MPa. Over the 30 random splits, on average 74% of test rows
    (67% to 80% by split) have their own mix, at another age, in training; with identical mixes
    held out, 22% (15% to 28%) still have a near-identical design there. The two rows above are
    one 28-day test entered twice, with 140 and 139.7 kg of
    cement and the identical strength 35.2 MPa. What will the model face in practice? A new
    design. So hold out designs (Kaufman et al., 2012, on leakage; Roberts et al., 2017, on
    structured cross-validation).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Ages: beyond the tested ages

    Train only on ages up to 28 days and predict the tests at 56 days or later. $\hat{y}$ is an
    average of strengths already seen, so $\min_i y_i \le \hat{y} \le \max_i y_i$ over the
    training set. Is that ceiling what goes wrong? The cell also counts the late tests stronger
    than every training test, and looks at the ages and mixes of their neighbors.
    """)
    return


@app.cell
def _(K, MARKERS, PALETTE, df, grouped_splits, knn_fit, mix_groups, np, plt):
    _y = df["strength"].to_numpy()
    _Z = mix_groups(df)
    _age = df["age"].to_numpy()
    _mix = df["mix"].to_numpy()
    _tr = np.where(_age <= 28)[0]
    _te = np.where(_age >= 56)[0]
    _pred_far, _, _nbr_far = knn_fit(_Z, _y, _tr, _te, K, True)
    _far = _pred_far - _y[_te]
    _near = np.concatenate(
        [
            knn_fit(_Z, _y, _a, _t[_age[_t] >= 56], K, True)[0] - _y[_t[_age[_t] >= 56]]
            for _a, _t in grouped_splits(df)
        ]
    )
    # Slide "k-NN cannot see past tested ages": 749 / 281 tests; −9.0 MPa, 92% under, RMSE 10.7; −1.4, 7.4.
    assert (len(_tr), len(_te)) == (749, 281)
    assert (round(float(_far.mean()), 1), round(float((_far < 0).mean()), 2)) == (-9.0, 0.92)
    assert round(float(np.sqrt(np.mean(_far**2))), 1) == 10.7
    assert (round(float(_near.mean()), 1), round(float(np.sqrt(np.mean(_near**2))), 1)) == (
        -1.4,
        7.4,
    )
    # Why (slide and notes): not a ceiling, only 1 of 281 late tests beats the strongest training
    # test (81.8 MPa); 99.9% of the neighbors are 28-day tests; 95% of the late tests have their
    # own mix among their 5 neighbors.
    _top = float(_y[_tr].max())
    assert (round(_top, 1), int((_y[_te] > _top).sum())) == (81.8, 1)
    assert round(float((_age[_nbr_far] == 28).mean()), 3) == 0.999
    assert round(float((_mix[_nbr_far] == _mix[_te][:, None]).any(axis=1).mean()), 2) == 0.95

    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.scatter(
        _y[_te],
        _y[_te] + _far,
        s=14,
        marker=MARKERS[1],
        color=PALETTE[1],
        alpha=0.6,
        label="trained on ≤ 28 days",
    )
    _ax.plot([0, 90], [0, 90], ":", color="#595959")
    _ax.set(
        xlabel="measured strength, age ≥ 56 days (MPa)",
        ylabel="5-NN prediction (MPa)",
        xlim=(0, 90),
        ylim=(0, 90),
    )
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 2: too low.** Trained on the 749 tests up to 28 days, 5-NN under-predicts 92% of the
    281 tests at 56 days or later, by 9.0 MPa on average (RMSE 10.7 MPa). With all ages in
    training the mean error on the same late tests is −1.4 MPa (RMSE 7.4).

    **Why: extrapolation in the inputs, not a ceiling on the output.** Only 1 of the 281 late
    tests is stronger than the strongest training test (81.8 MPa), so the ceiling explains almost
    nothing. A late test's $\ln(\text{age})$ lies beyond every training value, so its nearest
    neighbors are the tests closest in age, at 28 days (99.9% of the neighbors), and 95% of the
    late tests have their own mix among their five. The prediction is an average of 28-day
    strengths, whatever the age: k-NN is flat beyond the training range of an input.

    **Two differences at once.** The far run keeps each mix's early tests in training; the
    −1.4 MPa comparison holds whole designs out. The two runs differ in the ages in training and
    in whether the mix itself is there, so −9.0 against −1.4 is not a like-for-like comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Physics, the decisive check: more water should never mean more strength

    Keep the proposed mix and change only its water, from $w/b = 0.30$ to $0.65$, at 28 days.
    Abrams' law says strength only falls. Does the 5-NN prediction agree? This is the lecture's
    decisive check: it tests the assumption from §3 directly, at points where physics, not a
    held-out test, gives the answer.
    """)
    return


@app.cell
def _(
    K,
    KNeighborsRegressor,
    MARKERS,
    PALETTE,
    StandardScaler,
    abrams_A,
    abrams_B,
    df,
    grouped_splits,
    mix_groups,
    np,
    pd,
    plt,
    trace_row,
):
    _y = df["strength"].to_numpy()
    _tr, _te = grouped_splits(df)[0]
    _base = df.iloc[[trace_row]].copy()
    _base["age"] = 28
    _b = float(_base["cement"].iloc[0] + _base["slag"].iloc[0] + _base["fly_ash"].iloc[0])
    _sweep = pd.concat([_base] * 141, ignore_index=True)
    _sweep["water"] = _b * np.linspace(0.30, 0.65, 141)
    _sc = StandardScaler().fit(mix_groups(df.iloc[_tr]))
    _model = KNeighborsRegressor(n_neighbors=K).fit(
        _sc.transform(mix_groups(df.iloc[_tr])), _y[_tr]
    )
    _wb = mix_groups(_sweep)[:, 0]
    _pred = _model.predict(_sc.transform(mix_groups(_sweep)))
    _steps = np.diff(_pred)
    _ups = _steps[_steps > 1e-9]
    _flat_from = float(_wb[np.flatnonzero(np.abs(_steps) > 1e-9)[-1] + 1])
    # Slide "k-NN can contradict physics": binder 405; 48.7 down to 30.1 in 13 steps, 3 up (max 2.6);
    # flat from w/b 0.57; Abrams 20.0 at w/b 0.65.
    assert round(_b, 1) == 405.0
    assert (int((np.abs(_steps) > 1e-9).sum()), len(_ups), round(float(_ups.max()), 1)) == (
        13,
        3,
        2.6,
    )
    assert (round(float(_pred[0]), 1), round(float(_pred[-1]), 1)) == (48.7, 30.1)
    assert round(_flat_from, 2) == 0.57
    assert round(abrams_A / abrams_B**0.65, 1) == 20.0

    # Why it sticks (notes). From the last step on, the same five training tests stay nearest,
    # all of them drier than the sweep (w/b at most 0.55).
    _Ztr = _sc.transform(mix_groups(df.iloc[_tr]))
    _Zs = _sc.transform(mix_groups(_sweep))
    _dist, _nbr = _model.kneighbors(_Zs)
    _flat = int(np.flatnonzero(np.abs(_steps) > 1e-9)[-1] + 1)
    _five = np.sort(_tr[_nbr[_flat]])
    assert all(np.array_equal(np.sort(_tr[_n]), _five) for _n in _nbr[_flat:])
    _X = mix_groups(df)
    assert round(float(_X[_five, 0].max()), 2) == 0.55
    # Is the query far from every tested mix? Not at first: at w/b 0.57 the nearest test is 0.48
    # standardized units away, nearer than the proposed mix's own nearest (0.64); held-out tests'
    # nearest distances have median 0.37 and 90th percentile 0.74; at w/b 0.65 it is 0.91.
    _z_row = _sc.transform(_X[[trace_row]])
    _own = float(np.sqrt(((_Ztr - _z_row) ** 2).sum(1)).min())
    _held = _model.kneighbors(_sc.transform(_X[_te]), n_neighbors=1)[0][:, 0]
    assert (round(float(_dist[_flat, 0]), 2), round(_own, 2), round(float(_dist[-1, 0]), 2)) == (
        0.48,
        0.64,
        0.91,
    )
    assert (round(float(np.median(_held)), 2), round(float(np.percentile(_held, 90)), 2)) == (
        0.37,
        0.74,
    )
    # Why no wetter test takes over: training holds 159 tests wetter than w/b 0.57, but each is
    # at least 1.65 units from the proposed mix in the five other groups (the five: at most 0.73),
    # and the nearest of them is 1.72 from the w/b 0.65 query (the five: 0.91).
    _wet = _tr[_X[_tr, 0] > _wb[_flat]]
    _other = slice(1, None)
    _d_wet = np.sqrt(((_sc.transform(_X[_wet])[:, _other] - _z_row[:, _other]) ** 2).sum(1))
    _d_five = np.sqrt(((_sc.transform(_X[_five])[:, _other] - _z_row[:, _other]) ** 2).sum(1))
    _d_end = np.sqrt(((_sc.transform(_X[_wet]) - _Zs[-1]) ** 2).sum(1))
    assert len(_wet) == 159
    assert (round(float(_d_wet.min()), 2), round(float(_d_five.max()), 2)) == (1.65, 0.73)
    assert round(float(_d_end.min()), 2) == 1.72

    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _ax.step(_wb, _pred, where="mid", color=PALETTE[0], label="5-NN prediction")
    _ax.plot(
        _wb, abrams_A / abrams_B**_wb, "--", color=PALETTE[1], label="Abrams' law (28-day fit)"
    )
    _ax.plot(
        _wb[1:][_steps > 1e-9],
        _pred[1:][_steps > 1e-9],
        MARKERS[3],
        color=PALETTE[3],
        label="prediction goes up",
    )
    _ax.set(xlabel="w/b (only water changes)", ylabel="28-day strength (MPa)")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 3: no.** The prediction falls from 48.7 to 30.1 MPa, but it goes **up** in 3 of its
    13 steps, by up to 2.6 MPa, and past $w/b \approx 0.57$ it sticks at 30.1 MPa while Abrams'
    law keeps falling, to 20.0 MPa at $w/b = 0.65$.

    **The assumption fails twice.** First the neighbors are not similar enough: a step up means a
    little more water swapped one neighbor for a stronger one. Then, from $w/b$ 0.57 on, the same
    five tests stay nearest, and all five are drier ($w/b$ at most 0.55). Training holds 159
    wetter tests, but none is like this mix in its other groups (each at least 1.65 standardized
    units away in them; the five, at most 0.73), so none ever comes nearer. The distance does not
    warn: at $w/b$ 0.57 the nearest test is 0.48 away, nearer than the proposed mix's own nearest
    (0.64), and a held-out test's nearest is typically 0.37 (90th percentile 0.74). A model can
    fit the held-out tests well and still contradict physics between and beyond them. Check
    models against physics before trusting them; Lecture 23 builds the physics into the model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    Accept a mix when its prediction minus a margin reaches $f'_c$. Pool the held-out 28-day
    tests of all 30 design-held-out splits (425 tests, each held out in several splits: 3,179
    predictions) and count both errors: a **weak mix accepted** (measured below 35 MPa,
    accepted) and a **strong mix rejected**. The margin can be fixed, or the spread of the $k$
    neighbors' strengths.
    """)
    return


@app.cell
def _(FC, K, MARKERS, PALETTE, df, grouped_splits, knn_fit, mix_groups, np, pd, plt):
    _y = df["strength"].to_numpy()
    _Z = mix_groups(df)
    _age = df["age"].to_numpy()
    _r, _p, _s = [], [], []
    for _a, _te in grouped_splits(df):
        _t28 = _te[_age[_te] == 28]
        _pred, _spread, _ = knn_fit(_Z, _y, _a, _t28, K, True)
        _r.append(_t28)
        _p.append(_pred)
        _s.append(_spread)
    _r, _p, _s = map(np.concatenate, (_r, _p, _s))
    assert _r.shape == _p.shape == _s.shape, (_r.shape, _p.shape, _s.shape)
    _t = _y[_r]
    _weak, _strong = _t < FC, _t >= FC

    def _rates(accept: np.ndarray) -> tuple[float, float]:
        return float(accept[_weak].mean()), float((~accept)[_strong].mean())

    # The check on the spread: the smallest fixed margin (0.01 MPa grid) that accepts no more
    # weak mixes than the spread rule, and how many strong mixes each rejects.
    _fa_spread = _rates(_p - _s >= FC)[0]
    _m_eq = float(
        next(
            _x
            for _x in np.round(np.arange(0, 15.001, 0.01), 2)
            if _rates(_p - _x >= FC)[0] <= _fa_spread
        )
    )
    _decision = pd.DataFrame(
        {
            "no margin": _rates(_p >= FC),
            "margin 5 MPa": _rates(_p - 5 >= FC),
            "margin = neighbor spread": _rates(_p - _s >= FC),
            f"margin {_m_eq:.2f} MPa": _rates(_p - _m_eq >= FC),
        },
        index=["weak mixes accepted", "strong mixes rejected"],
    ).round(3)
    # Slide "Every margin is a trade-off": 425 tests at 28 days, 3,179 held-out predictions
    # over 30 splits (a test is held out in several splits); 13.9 / 14.9; 4.4 / 36.5;
    # spread (sample SD) 4.8 / 37.0, median spread 4.7; a fixed 4.62 MPa accepts the same 4.8%
    # and rejects 34.8%; notes: 51% of the predictions are of weak mixes.
    assert (len(np.unique(_r)), len(_t), len(grouped_splits(df))) == (425, 3179, 30)
    assert round(float(np.median(_s)), 1) == 4.7 and round(float(_weak.mean()), 2) == 0.51
    assert list(_decision["no margin"]) == [0.139, 0.149]
    assert list(_decision["margin 5 MPa"]) == [0.044, 0.365]
    assert list(_decision["margin = neighbor spread"]) == [0.048, 0.37]
    assert _m_eq == 4.62 and list(_decision["margin 4.62 MPa"]) == [0.048, 0.348]

    # Is the 2.2-point gap more than the noise? Resample the designs (with replacement, 400
    # draws), match the fixed margin again in each draw, and recompute the gap. Weights count
    # how often each prediction's design was drawn; with integer weights the counts are exact.
    _grid = np.round(np.arange(0, 15.001, 0.01), 2)
    _acc_s = _p - _s >= FC
    _acc_f = (_p[:, None] - _grid[None, :]) >= FC  # (predictions, margins)

    def _gap(weights: np.ndarray) -> np.ndarray:
        """Spread's strong-mix reject share minus the matched fixed margin's, per weight row.

        Args:
            weights: How often each pooled prediction counts, shape (draws, predictions).

        Returns:
            The difference in reject shares, shape (draws,).
        """
        _ww, _ws = weights[:, _weak], weights[:, _strong]
        _ok = _ww @ _acc_f[_weak] <= (_ww @ _acc_s[_weak])[:, None]
        assert _ok.any(axis=1).all()
        _j = _ok.argmax(axis=1)  # the smallest margin that accepts as few weak mixes
        _n = _ws.sum(axis=1)
        _fixed = (_ws @ ~_acc_f[_strong])[np.arange(len(_j)), _j]
        return (_ws @ ~_acc_s[_strong] - _fixed) / _n

    _designs, _which = np.unique(df["design"].to_numpy()[_r], return_inverse=True)
    _picks = np.random.default_rng(0).integers(0, len(_designs), size=(400, len(_designs)))
    _counts = np.stack([np.bincount(_d, minlength=len(_designs)) for _d in _picks]).astype(float)
    _boot = 100 * _gap(_counts[:, _which])
    _lo, _hi = np.percentile(_boot, [2.5, 97.5])
    # Notes: 2.2 points; 308 designs; 95% of draws between −5.2 and +8.9; 29% at or below zero.
    assert round(float(100 * _gap(np.ones((1, len(_t))))[0]), 1) == 2.2 and len(_designs) == 308
    assert (round(float(_lo), 1), round(float(_hi), 1)) == (-5.2, 8.9)
    assert round(float((_boot <= 0).mean()), 2) == 0.29

    _m = np.linspace(0, 15, 61)
    _curve = np.array([_rates(_p - _x >= FC) for _x in _m])
    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _ax.plot(100 * _curve[:, 1], 100 * _curve[:, 0], color=PALETTE[0], label="margin 0 to 15 MPa")
    for _c, (_name, _col) in enumerate(_decision.iloc[:, :3].items(), start=1):
        _ax.plot(
            100 * _col.iloc[1], 100 * _col.iloc[0], MARKERS[_c], color=PALETTE[_c], label=_name
        )
    _fixed = _decision.iloc[:, 3]  # on the curve: hollow, in the curve's colour
    _ax.plot(100 * _fixed.iloc[1], 100 * _fixed.iloc[0], "D", mfc="none", color=PALETTE[0],
             label=_fixed.name)  # fmt: skip
    _ax.set(xlabel="strong mixes rejected (%)", ylabel="weak mixes accepted (%)", ylim=(0, 40))
    _ax.legend()
    (_decision, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Which point on the trade-off is right depends on what a wrong "yes" costs. Design codes
    already make a mix target an average strength above $f'_c$ (ACI 318); a margin applies the
    same idea to a prediction.

    ### When not to use k-NN here: it can screen a mix; it cannot accept one

    - **Rough uncertainty.** The neighbors' spread (proposed mix: 42.6 ± 6.4 MPa, measured 35.9)
      is no better than a fixed margin: a fixed 4.62 MPa accepts as few weak mixes (4.8%) and
      rejects 34.8% of strong ones against the spread's 37.0%. That 2.2-point gap is within the
      noise: resampling the 308 designs puts it between −5.2 and +8.9 points, at or below zero
      in 29% of the draws. With $k = 5$ the spread is noisy itself. Duplicated records make it
      too small: two copies of one test agree perfectly.
    - **Not beyond the data:** ages, $w/b$, or binders that no tested mix shares (§5).
    - **Not for new materials:** other cements, admixtures, or aggregates are not in these tests.
    - **Not for acceptance:** codes accept concrete on cylinder strength tests (ACI 318), and the
      engineer of record signs the design, not the model.

    Uncertainty with a guarantee comes later: calibration (Lecture 10), conformal prediction
    (Lecture 11).

    ### Impact: the margin costs carbon or safety

    Cement production emits roughly 8% of global CO₂ (Lehne & Preston, 2018): a wide margin means
    more cement in every mix, a narrow one shifts risk to the owner and the public. The engineer
    of record signs the design, not the model. A model that lets a producer cut cement must show
    that it does not raise the share of weak mixes accepted.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: would you accept this mix?

    $f'_c = 35$ MPa. The 5-NN prediction is 37 MPa; the neighbors' spread is 6 MPa.

    - **A.** Accept: the prediction is above 35 MPa.
    - **B.** Reject: 37 − 35 is smaller than the spread.
    - **C.** Not yet: require a trial batch and cylinder tests first.

    Vote alone. Discuss with a neighbor for two minutes. Vote again. Then open the explanation.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Explanation": mo.md(
                r"""
                **A** ignores the uncertainty: with no margin, 13.9% of weak mixes were accepted.
                **B** treats a rough spread as if it were calibrated. **C** is the engineer's
                answer when a wrong "yes" is costly: codes already ask a mix to target an average
                strength above $f'_c$ (ACI 318), and a prediction 2 MPa above $f'_c$ with a 6 MPa
                spread does not do that.

                What would change the answer: a calibrated interval (Lectures 10–11), a cheaper
                wrong "yes", or more tested mixes like this one.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where k-NN lives today: retrieval

    **Retrieval-augmented generation** (RAG; Lewis et al., 2020, NeurIPS) answers a question with a
    language model that reads the $k$ passages nearest to the question in an embedding space: the
    encoder–decoder picture of §2, with a learned encoder and a language model as the decoder.
    Today's questions carry over unchanged. The encoder decides what "near" means, as dimensionless
    groups did for concrete. When no tested mix is similar, k-NN still returns an average (§5);
    when no passage answers the question, RAG still returns an answer, and it can sound sure.
    At scale the search is approximate nearest-neighbor (ANN) search: an index returns most of the
    true $k$ nearest, measured by recall@$k$, far faster than an exact scan, on GPUs over a billion
    vectors (Johnson, Douze & Jégou, 2021, *IEEE Transactions on Big Data* 7(3), the FAISS paper).
    In theory, a $(1+\varepsilon)$-approximate neighbor is within $1+\varepsilon$ times the
    nearest distance; practical indexes such as HNSW and IVF-PQ carry no such guarantee and are
    judged by recall. Lecture 19 builds a RAG system over the MUTCD (FHWA, 11th Edition) and
    evaluates its retrieval.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question** (one minute, no notes): which change cut the error most, and which split
    would you use to report the error to a client? Why?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Log age (9.5 to 8.0 MPa), then Abrams' ratio (8.0 to 6.9 MPa), both from
                engineering knowledge, not from the algorithm. Report the error with whole designs
                held out (6.9 MPa, not the 5.7 of a random split), because the client's mix is a
                new design.
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
            "Q1. Why does water, the ingredient that controls strength, carry only 1.2% of the raw distance?": mo.md(
                r"""
                In expectation, input $j$ contributes $2\,\mathrm{Var}(x_j)$ to the squared distance
                between two tests. Water varies by about 21 kg/m³ across the data, cement by about
                105 and the aggregates by about 80 each, so their variances are 13 to 24 times
                larger. Units, not importance, decide the raw distance. Standardizing equalizes the
                inputs; dimensionless groups go further and use the ratio that matters.
                """
            ),
            "Q2. Why fit the scaler on the training split only?": mo.md(
                r"""
                The means and standard deviations are part of the encoder. Computing them on all
                data lets test-set statistics shape the training representation, a small but real
                leak. Fit on training rows, then transform both.
                """
            ),
            "Q3. Name two ways k-NN can violate physics, and how to detect them.": mo.md(
                r"""
                Non-monotone responses (more water predicting more strength: 3 of 13 steps in the
                sweep) and flat extrapolation (stuck at 30.1 MPa past $w/b \approx 0.57$, while
                Abrams' law falls to 20.0 MPa at $w/b = 0.65$). Detect them with sweeps along one
                physical variable, holding the others fixed, and compare against known laws.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Raw units in a distance.** Any distance-based method (k-NN, k-means, RBF kernels) on
      kg/m³ and days is dominated by the largest variances, not the most important inputs.
    - **Scaler fit on all data.** Fit `StandardScaler` on training rows only.
    - **Random splits of repeated measurements.** Tests of one design at several ages, or one
      design recorded twice, must be held out together (`GroupShuffleSplit`, `GroupKFold`).
    - **Grouping by exact equality.** Compiled data repeat designs with rounded values; group with
      a tolerance and check how many groups merge.
    - **Duplicate records.** They count twice in a vote and shrink the neighbor spread.
    - **Trusting predictions outside the data.** Check the distance to the nearest neighbor and
      the range of the training inputs (ages, ratios) before using a prediction.
    - **Ties in `argsort`/`argpartition`.** Implementations may break distance ties differently;
      compare against a reference away from ties, and count the ties.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §1.2 and §1.2.1** — nearest-neighbor classification and its practical issues:
      enough labeled data, a sensible distance (scale each feature, or whiten so the covariance is
      the identity), and the cost of finding neighbors. Our dimensionless groups are a third answer
      to the distance question: choose coordinates from physics.
    - **Forsyth §1.1.3** — overfitting and cross-validation. Next lecture builds on it with grouped,
      spatial, and temporal splits.
    - **Exercise.** Replace standardization by whitening (Forsyth §1.2.1) on the raw inputs and on
      the dimensionless groups. Does whitening close the gap to the groups? Hold out designs.
    - **Exercise.** Add `water / cement` as a seventh group, or drop `aggregate / b`. Which change
      matters more, and does the answer agree with Abrams?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - k-NN predicts from the most similar tested mixes; the engineering is in what "similar"
      means.
    - Raw units let cement and aggregates dominate; log age and Abrams' ratio cut the error from
      10.0 to 6.9 MPa.
    - Hold out whole designs, not tests: near-duplicates hide in compiled data.
    - k-NN cannot extrapolate and need not respect physics. Check both.
    - A prediction becomes a decision only with a margin chosen by the cost of a wrong "yes".

    **Next:** generalization, and validation the way infrastructure data fail.
    """)
    return


if __name__ == "__main__":
    app.run()
