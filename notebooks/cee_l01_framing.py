"""Applied Machine Learning for CEE — Lecture 1: When Predictions Carry Loads.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l01_framing.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data, read only, already in ``data/derived``: ``Concrete_Data.csv`` (``bash
scripts/fetch_concrete.sh``), ``nbi_il_bridges.csv`` (``bash scripts/fetch_nbi.sh``), and
``bdg2_daily_electricity.npz`` (``bash scripts/fetch_bdg2.sh``).
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
    # Applied Machine Learning for CEE — Lecture 1: When Predictions Carry Loads

    **Slides:** *Applied Machine Learning for CEE*, Lecture 1 (`lectures/cee_l01_framing`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §1.1.3 (overfitting and cross-validation), §1.3.1 (cross-validation to choose a model), and §3.1 (held-out loss predicts test loss).
    **Data:** UCI Concrete Compressive Strength (Yeh, 1998), 1,030 laboratory tests; the Illinois records of the FHWA National Bridge Inventory; Building Data Genome 2 (Miller et al., 2020). All three are read from `data/derived`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`.

    **Your tasks** (from the slides): rank the four teams yourself, write `val_rmse` and match
    scikit-learn, reproduce the winner's-curse curve, and break the model with fly ash.
    """)
    return


@app.cell
def _():
    import itertools
    import os
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from joblib import Parallel, delayed
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        ShuffleSplit,
        cross_val_predict,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    K = 5  # neighbors, as in Lecture 2
    REPEATS = 30  # Lecture 2's repeated 75/25 splits
    FOLDS = 5  # validation folds inside the training designs
    FC = 35.0  # specified 28-day strength f'c, MPa
    ORDERS = 20  # random orders of the input sets for the winner's-curse curve
    M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 255]
    N_JOBS = 4  # worker processes for the 30 splits; the results do not depend on it
    NEAR_MPA = 2.0  # half-width of the band of predictions near the traced design's
    # Rows that Lectures 2 and 5 use in their attempts and traces; their designs stay out of
    # this lecture's examples.
    LATER_EXAMPLE_ROWS = [913, 524, 341, 1018, 336, 638, 825, 802, 779, 801]
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        FC,
        FOLDS,
        GroupKFold,
        GroupShuffleSplit,
        K,
        KNeighborsRegressor,
        LATER_EXAMPLE_ROWS,
        MARKERS,
        M_GRID,
        NEAR_MPA,
        N_JOBS,
        ORDERS,
        PALETTE,
        Parallel,
        Path,
        REPEATS,
        SEED,
        ShuffleSplit,
        StandardScaler,
        cross_val_predict,
        delayed,
        itertools,
        make_pipeline,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions to guess

    On your own, two minutes. Write a guess; nobody expects the answers yet.

    1. A model's error on its test set is 8.1 MPa. Would you sign off on a mix design that rests
       on its prediction? What else would you ask?
    2. You try 255 sets of inputs and keep the one with the best test score. As an estimate of
       its error on new mixes, is that score too low, too high, or about right?
    3. Do students learn more in lectures where they solve problems or in lectures where they
       listen? In which do they *feel* they learn more?

    These are *prequestions*: asked before the lesson, answered during it. Guessing first helps
    learning even when most first guesses are wrong, provided the correct answers follow (Pan &
    Carpenter, 2023, *Educational Psychology Review* 35, 97). Keep your guesses; the answers are
    in §6, §3 and §5, and "How the course runs".
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers (open after §6)": mo.md(
                r"""
                1. Not on that number alone. Ask how it was measured: on cases like this one
                   (whole new designs), looked at once? Is this mix inside the data? What does an
                   error cost, and who carries it? Who signs? In §5 the 8.1 MPa turns out to be
                   the best of 255 scores on one test set; chosen honestly, the same search
                   scores 8.7.
                2. Too low. The best of many noisy scores is biased low (§3); on fresh designs
                   the winner of 255 is 1.0 MPa worse than on the half that picked it (§5).
                3. Students learn more in active classes but feel they learn less (Deslauriers et
                   al., 2019, *PNAS* 116(39)).
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A producer wants to approve new mix designs from a model's predicted strength, before any
    cylinder is tested; the slab is specified at $f'_c = 35$ MPa at 28 days. **Frame it:** the
    input is the mix (seven ingredients in kg/m³, and the age), the output is the compressive
    strength, and the decision is to approve the design or test it first.

    The model's reported error sets the safety margin. If the reported error is **too small**,
    weak concrete is approved, and the owner and the public carry the risk. If it is **too
    large**, cement and money are spent for nothing. So before any accuracy number we ask how
    it was measured.

    **The data.** 1,030 laboratory tests: seven ingredients in kg per m³ of concrete, the age in
    days, and the compressive strength in MPa. One test is a row, but one mix is often tested at
    several ages, and the compiled records repeat some designs with rounded proportions. Each
    row gets two ids, as in Lecture 2: `mix` (identical proportions) and `design` (all seven
    proportions within 1 kg/m³).
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
    INPUTS = [*INGREDIENTS, "age"]
    INPUT_NAMES = ["cement", "slag", "fly ash", "water", "SP", "coarse", "fine", "age"]

    def find_data(name: str) -> Path:
        """Locate ``data/derived/<name>`` from the repo root, notebooks/, or ipynb/.

        Args:
            name: File name inside ``data/derived``.

        Returns:
            Path of the file.

        Raises:
            FileNotFoundError: If the file is missing (run the matching ``scripts/fetch_*.sh``).
        """
        rel = Path("data") / "derived" / name
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run the matching scripts/fetch_*.sh")

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

    return INPUTS, INPUT_NAMES, find_data, load_concrete


@app.cell
def _(INPUTS, find_data, load_concrete, np, pd):
    df = load_concrete(find_data("Concrete_Data.csv"))
    X = df[INPUTS].to_numpy(np.float64)  # seven ingredients (kg/m³) and age (days)
    y = df["strength"].to_numpy()
    d = df["design"].to_numpy()
    assert X.shape == (1030, 8) and y.shape == d.shape == (1030,), (X.shape, y.shape)
    _summary = {
        "tests": len(df),
        "mixes (identical proportions)": df["mix"].nunique(),
        "designs (within 1 kg/m³)": df["design"].nunique(),
        "distinct ages": df["age"].nunique(),
        "age range (days)": f"{df['age'].min()} to {df['age'].max()}",
        "tests at 28 days": int((df["age"] == 28).sum()),
        "strength range (MPa)": f"{df['strength'].min():.1f} to {df['strength'].max():.1f}",
    }
    # Slide "A model's error sets the margin" and its notes.
    assert (_summary["tests"], _summary["mixes (identical proportions)"]) == (1030, 428)
    assert (_summary["designs (within 1 kg/m³)"], _summary["distinct ages"]) == (315, 14)
    assert (df["age"].min(), df["age"].max(), _summary["tests at 28 days"]) == (1, 365, 425)
    assert _summary["strength range (MPa)"] == "2.3 to 82.6"
    pd.Series(_summary, name="value").to_frame()
    return X, d, df, y


@app.cell
def _(mo):
    mo.md(r"""
    ### Tour of the course datasets

    Three public datasets are already in `data/derived`, and the first five lectures use them.
    For each, name the input, the output, and the decision; then ask what an error costs and
    who carries it.

    | Data | Input → output | Decision | Lectures |
    |:--|:--|:--|:--|
    | Concrete tests (Yeh, 1998) | mix and age → strength | approve a mix design | 2, 5, 10 |
    | Illinois bridges, FHWA National Bridge Inventory, 2014–2024 | inventory → condition rating | which bridges to inspect first | 3, 6 |
    | Building Data Genome 2 (Miller et al., 2020) | a year of meter data → typical weekday | which buildings get an after-hours visit | 4 |

    Later units add, among others: loop detectors on Portland's I-205, pavement performance
    (FHWA LTPP), a water network with leaks (BattLeDIM), USGS stream gauges in Illinois, crack
    images (SDNET2018, 3DCrack), flooded buildings seen from satellites (SpaceNet 8), a laser
    scan of a building (CRAS lab), OSHA injury reports, and the traffic-control manual (FHWA
    MUTCD). The cell
    counts the records. The bridge counts use Lecture 3's cleaning (every input present); the
    building counts use Lecture 4's rules, which need only the meter data.
    """)
    return


@app.cell
def _(find_data, np, pd):
    _nbi = pd.read_csv(find_data("nbi_il_bridges.csv"), dtype={"structure": str, "county": str})
    _needed = [
        "year_built",
        "inspection_year",
        "adt",
        "pct_truck",
        "length_m",
        "deck_width_m",
        "lat",
        "lon",
    ]
    _nbi = _nbi.dropna(subset=_needed)
    _per_year = _nbi.groupby("nbi_year")["structure"].nunique()

    _bdg = np.load(find_data("bdg2_daily_electricity.npz"))
    _kwh, _names = _bdg["kwh"], _bdg["building_id"]
    _dates = pd.DatetimeIndex(pd.to_datetime(_bdg["date"]))
    _site = np.array([_n.split("_")[0] for _n in _names])  # BDG2 ids are site_use_name
    _weekday, _year = np.asarray(_dates.dayofweek < 5), np.asarray(_dates.year)

    def _typical(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean kW per hour over the kept days with all 24 hours metered, and the day count."""
        _sub = _kwh[:, keep, :]
        _ok = ~np.isnan(_sub).any(axis=2)
        _n = _ok.sum(axis=1)
        return np.where(_ok[..., None], _sub, 0.0).sum(axis=1) / np.maximum(_n, 1)[:, None], _n

    _p16, _n16 = _typical(_weekday & (_year == 2016))
    _p17, _n17 = _typical(_weekday & (_year == 2017))
    _enough = (_n16 >= 0.8 * (_weekday & (_year == 2016)).sum()) & (
        _n17 >= 0.8 * (_weekday & (_year == 2017)).sum()
    )
    _positive = (_p16.min(axis=1) > 0) & (_p17.min(axis=1) > 0)
    _complete = ~np.isnan(_kwh).any(axis=2)
    _flat = np.ptp(np.where(_complete[..., None], _kwh, 0.0), axis=2) == 0
    _hourly = (_flat & _complete).sum(axis=1) / np.maximum(_complete.sum(axis=1), 1) < 0.5
    _keep = _enough & _positive & _hourly

    _tour = pd.DataFrame(
        {
            "records": [len(_nbi), int(_kwh.shape[0])],
            "units": [_nbi["structure"].nunique(), len(set(_site))],
            "unit": ["bridges", "sites"],
        },
        index=["NBI Illinois (bridge-year records)", "BDG2 (buildings)"],
    )
    # Slide "Three datasets open the course" and its notes.
    assert len(_nbi) == 131_851
    assert (_nbi["structure"].nunique(), _nbi["county"].nunique()) == (23_464, 102)
    assert (_nbi["nbi_year"].min(), _nbi["nbi_year"].max()) == (2014, 2024)
    assert (_per_year.min(), _per_year.max()) == (21_935, 22_015)
    assert (_kwh.shape, len(set(_site))) == ((1578, 731, 24), 19)
    assert (_dates.year.min(), _dates.year.max()) == (2016, 2017)
    assert (int(_keep.sum()), len(set(_site[_keep]))) == (1159, 15)
    # the map on the slide: 2024's bridges, poor = lowest rating 4 or less (FHWA)
    _poor = 4
    _last = _nbi[_nbi["nbi_year"] == 2024]
    assert (len(_last), int((_last["rating"] <= _poor).sum())) == (21_966, 2_308)
    _tour
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: which error would you quote?

    A model predicts a new design at 38.1 MPa; $f'_c = 35$ MPa. Four teams measured its error.
    The model averages the strengths of the five most similar tested mixes, each input
    standardized (Lecture 2 builds it); RMSE is the root-mean-square error.

    | Team | Error measured on | RMSE (MPa) |
    |:--|:--|--:|
    | A | the tests it was fit to | 7.0 |
    | B | random tests held out | 8.9 |
    | C | the best of 255 input sets, on the test set that picked it | 8.1 |
    | D | whole designs held out, looked at once | 9.5 |

    **Two minutes, alone:** rank A–D, most trustworthy first. Would you approve the design?
    Write your answer down before running the cells below; §3 to §6 explain each row.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Helpers used in every section.** Lecture 2's 30 splits that hold out a quarter of the
    designs, its 30 random splits, and the plain model: 5-NN on the eight inputs, standardized
    with training statistics. One detail matters for reproducing the slides to the last digit:
    always select columns first, then rows (`X[:, cols][rows]`), as `cross_val_predict` does.
    The data have many exact ties between equally distant tests (one mix at many ages, designs
    entered twice), k-NN breaks them by the internals of its search tree, and a different copy
    order can change the last bit of the standardized values and so which tied test is taken.
    """)
    return


@app.cell
def _(
    GroupShuffleSplit,
    K,
    KNeighborsRegressor,
    REPEATS,
    SEED,
    ShuffleSplit,
    StandardScaler,
    np,
    pd,
):
    def grouped_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Lecture 2's ``REPEATS`` 75/25 splits that keep every design in train or in test."""
        gss = GroupShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(gss.split(df, groups=df["design"]))

    def random_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Lecture 2's ``REPEATS`` 75/25 splits of the tests, ignoring mixes and designs."""
        ss = ShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
        return list(ss.split(df))

    def knn_predict(
        X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """5-NN on standardized inputs; scaler and neighbors from ``tr`` only.

        Args:
            X: Inputs, shape ``(N, D)``, columns already selected.
            y: Strengths, shape ``(N,)``.
            tr: Training rows.
            te: Rows to predict.

        Returns:
            Predictions for ``te`` and the training rows of their neighbors, ``(len(te), K)``.
        """
        sc = StandardScaler().fit(X[tr])
        knn = KNeighborsRegressor(n_neighbors=K).fit(sc.transform(X[tr]), y[tr])
        Z_te = sc.transform(X[te])
        _, idx = knn.kneighbors(Z_te)
        assert idx.shape == (len(te), K), idx.shape
        return knn.predict(Z_te), tr[idx]

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square difference of two arrays of one shape.

        Args:
            a: Predictions.
            b: Measured values, the same shape as ``a``.

        Returns:
            The RMSE, in the units of ``b``.

        Raises:
            ValueError: If the shapes differ (broadcasting would silently compare the wrong
                pairs).
        """
        if np.shape(a) != np.shape(b):
            raise ValueError(f"shapes differ: {np.shape(a)} against {np.shape(b)}")
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return grouped_splits, knn_predict, random_splits, rmse


@app.cell
def _(X, df, grouped_splits, knn_predict, np, random_splits, rmse, y):
    splits = grouped_splits(df)
    _at28 = df["age"].to_numpy() == 28
    _train_err, _test_err, _test28_err = [], [], []
    for _tr, _te in splits:
        _train_err.append(rmse(knn_predict(X, y, _tr, _tr)[0], y[_tr]))  # scored on itself
        _pred_te = knn_predict(X, y, _tr, _te)[0]
        _test_err.append(rmse(_pred_te, y[_te]))  # designs held out
        _test28_err.append(rmse(_pred_te[_at28[_te]], y[_te][_at28[_te]]))  # their 28-day tests
    _rand_err = [rmse(knn_predict(X, y, _tr, _te)[0], y[_te]) for _tr, _te in random_splits(df)]
    _team_A, _team_B, team_D = (
        float(np.mean(_train_err)),
        float(np.mean(_rand_err)),
        float(np.mean(_test_err)),
    )
    team_D_sd = float(np.std(_test_err))
    # The approval decision is about 28-day tests: its honest margin is the same held-out error
    # on the 28-day tests only.
    honest_28 = float(np.mean(_test28_err))
    # Slide "Your turn": A 7.0, B 8.9, D 9.5; B and D are Lecture 2's numbers (8.886, 9.511).
    # Slide "Before you sign": 9.9 MPa on the held-out 28-day tests.
    assert (round(_team_A, 1), round(_team_B, 3), round(team_D, 3)) == (7.0, 8.886, 9.511)
    assert round(honest_28, 1) == 9.9
    {
        "A: training tests": round(_team_A, 2),
        "B: random tests": round(_team_B, 2),
        "D: designs held out": round(team_D, 2),
        "D: spread over 30 splits": round(team_D_sd, 2),
        "D: 28-day tests only": round(honest_28, 2),
    }
    return honest_28, splits, team_D, team_D_sd


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The picture that every later lecture reuses, with one step highlighted:

    > input $\mathbf{x}$ → **encoder** $\mathbf{z}=f_\phi(\mathbf{x})$ → $\mathbf{z}$ →
    > **decoder** $\hat{y}=g_\theta(\mathbf{z})$ → $\hat{y}$ → **loss** $\mathcal{L}(\hat{y}, y)$
    > ← target $y$

    The encoder and the decoder form the model; training adjusts $\phi$ and $\theta$ to lower
    the loss on the training set. For today's model the encoder standardizes each input with
    the training means and spreads ($\phi$), the decoder averages the strengths of the 5 nearest
    training tests ($\theta$ is the stored tests), and the loss is the squared error.

    The cell follows the design from your attempt through the pipeline. How it was chosen: among
    split 0's held-out 28-day tests (excluding designs that later lectures use as examples), the
    wrong "yes" (prediction ≥ $f'_c$ > measured) whose error is closest to the held-out error
    on 28-day tests, 9.9 MPa: a costly error of typical size.
    """)
    return


@app.cell
def _(
    FC,
    INPUTS,
    INPUT_NAMES,
    LATER_EXAMPLE_ROWS,
    StandardScaler,
    X,
    d,
    df,
    honest_28,
    knn_predict,
    np,
    pd,
    splits,
    y,
):
    # Designs that Lectures 2 and 5 use in their attempts and traces stay out.
    _later = set(d[LATER_EXAMPLE_ROWS])
    _tr, _te = splits[0]
    _pred, _nbrs = knn_predict(X, y, _tr, _te)
    _age = df["age"].to_numpy()
    _ok = (_age[_te] == 28) & ~np.isin(d[_te], list(_later)) & (_pred >= FC) & (y[_te] < FC)
    _cand = np.flatnonzero(_ok)
    _i = int(_cand[np.argmin(np.abs((_pred[_cand] - y[_te][_cand]) - round(honest_28, 1)))])
    _row = int(_te[_i])
    _z = StandardScaler().fit(X[_tr]).transform(X[[_row]])[0]
    _binder = (df["cement"] + df["slag"] + df["fly_ash"]).to_numpy()
    _trace = {
        "pred": float(_pred[_i]),
        "y": float(y[_row]),
        "nbr_strengths": [round(float(_v), 1) for _v in y[_nbrs[_i]]],
        "nbr_wb": [
            round(float(_v), 2) for _v in df["water"].to_numpy()[_nbrs[_i]] / _binder[_nbrs[_i]]
        ],
        "wb": float(df["water"].iloc[_row] / _binder[_row]),
    }
    _err = _trace["pred"] - _trace["y"]
    # Slide "Encoder, decoder, loss: every model" (diagram and notes).
    assert _row == 873 and (len(_tr), len(np.unique(d[_tr]))) == (746, 236)
    assert (round(_trace["pred"], 1), round(_trace["y"], 1)) == (38.1, 28.6)
    assert (round(_err, 1), round(_err**2, 1)) == (9.5, 90.3)
    assert _trace["nbr_strengths"] == [35.2, 35.2, 37.2, 37.2, 45.9]
    assert _trace["nbr_wb"] == [0.55, 0.55, 0.46, 0.46, 0.60] and round(_trace["wb"], 2) == 0.62
    assert (round(_z[3], 2), round(_z[0], 2)) == (3.13, -0.41)  # water, cement (SDs)
    assert int((d[_te] == d[_row]).sum()) == 2  # the design, entered twice
    pd.DataFrame(
        {
            "x (kg/m³, days)": df.loc[_row, INPUTS].to_numpy(float).round(1),
            "z (standardized)": _z.round(2),
        },
        index=INPUT_NAMES,
    )
    trace_pred = _trace["pred"]  # the centre of the margin picture's band
    return (trace_pred,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Goal.** The error on the cases the model will face, $L=\mathbb{E}\,(\hat{y}-y)^2$. We
    estimate it from $n$ held-out cases and report its square root, the RMSE, in MPa:
    $$\hat{L}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2 .$$

    **Assumption.** Training, validation, and test cases are drawn like the future cases, and
    the test cases were never used to fit or to choose. Then each held-out squared error has
    expectation $L$, and by linearity $\mathbb{E}\,\hat{L}=L$ (Forsyth §3.1 adds Chebyshev's
    inequality: the estimate is rarely far off when $n$ is large).

    **Three roles.**
    - The **training set** fits $\phi$ and $\theta$. Its error flatters: each training test is
      one of its own five neighbors (Team A's 7.0).
    - The **validation set** chooses among models: which inputs, how many neighbors. It can be
      a separate set or cross-validation, folds of the training designs held out in turn.
    - The **test set** scores the final choice once, at the end.

    **Why only once.** Each model's score is unbiased, $\mathbb{E}\,\hat{L}_m=L_m$. The best of
    several is not: $\min_m\hat{L}_m\le\hat{L}_j$ for every $j$, so taking expectations,
    $$\mathbb{E}\,\min_m\hat{L}_m\le\min_m L_m .$$
    Choosing among models on a set uses up that set's honesty (Cawley & Talbot, 2010, *JMLR* 11).

    ### Worked example: two equally good models

    Let the two scores be independent, with mean $L$ and spread $\sigma$. Since
    $\min(a,b)=\tfrac{a+b}{2}-\tfrac{|a-b|}{2}$ and $a-b$ has spread $\sigma\sqrt{2}$, so that
    $\mathbb{E}|a-b| = \sigma\sqrt{2}\sqrt{2/\pi} = 2\sigma/\sqrt{\pi}$:
    $$\mathbb{E}\,\min(a,b) = L-\frac{\sigma}{\sqrt{\pi}} .$$
    Take $\sigma$ as the spread of Team D's score over the 30 splits.

    **Two caveats.** Two models scored on one test set are not independent: they err on the same
    hard cases, so their scores are positively correlated, with correlation $\rho$. Then $a-b$
    has spread $\sigma\sqrt{2(1-\rho)}$ and the optimism is $\sigma\sqrt{1-\rho}/\sqrt{\pi}$, so
    $\sigma/\sqrt{\pi}$ is an upper bound. And "unbiased" holds for $\hat{L}$, the mean squared
    error; its square root, the RMSE we quote, is slightly low on average (Jensen's inequality).
    The cell checks both formulas by simulation (normal scores, $\rho = 0$ and $\rho = 0.5$).
    """)
    return


@app.cell
def _(np, team_D, team_D_sd):
    _rng = np.random.default_rng(0)
    _gains = {}
    for _rho in (0.0, 0.5):
        _cov = team_D_sd**2 * np.array([[1.0, _rho], [_rho, 1.0]])
        _scores = _rng.multivariate_normal([team_D, team_D], _cov, size=200_000)
        _gain_mc = team_D - _scores.min(axis=1).mean()
        _gain = team_D_sd * np.sqrt(1 - _rho) / np.sqrt(np.pi)
        assert abs(_gain_mc - _gain) < 0.01, (_rho, _gain_mc, _gain)
        _gains[f"rho = {_rho}"] = {"formula": round(_gain, 3), "simulated": round(_gain_mc, 3)}
    # Slide "The best of many scores is too good": sigma 0.9 MPa, 0.5 MPa too good (rho = 0).
    assert (round(team_D_sd, 1), round(team_D_sd / np.sqrt(np.pi), 1)) == (0.9, 0.5)
    _gains


@app.cell
def _(np, team_D, team_D_sd):
    # The slide's picture: the expected best of M equally good, independent scores (true error
    # 9.5 MPa, spread 0.9) sits 0.5, 1.5, and 2.5 MPa below the truth for M = 2, 16, 255.
    _rng = np.random.default_rng(1)
    _best = {
        _m: float(_rng.normal(team_D, team_D_sd, size=(20_000, _m)).min(axis=1).mean())
        for _m in (2, 16, 255)
    }
    # Exactly, as an order statistic: E min = mean + sd * integral of x M phi(x) S(x)^(M-1),
    # S the normal survival function; the figure script quotes 9.02, 7.96, 7.04 MPa.
    _x = np.linspace(-12, 12, 240_001)
    _phi = np.exp(-(_x**2) / 2) / np.sqrt(2 * np.pi)
    _sf = 1 - np.cumsum(_phi) * (_x[1] - _x[0])
    _exact = {
        _m: team_D + team_D_sd * float(np.trapezoid(_x * _m * _phi * _sf ** (_m - 1), _x))
        for _m in _best
    }
    assert [round(_e, 2) for _e in _exact.values()] == [9.02, 7.96, 7.04]
    assert all(abs(_best[_m] - _exact[_m]) < 0.01 for _m in _best)  # simulation agrees
    _best, _exact


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Predict first** (write it down): (1) will `val_rmse` match scikit-learn's
    `cross_val_predict` on all 255 input sets? (2) Over 30 splits, will the chosen set's test
    score be higher or lower than its validation score?

    `val_rmse` is the slide's listing: five folds of whole designs (`d` holds each test's design
    id), the scaler fit inside each fold because it is part of the model. **Run** the next cells;
    then **investigate**: why must the scaler be fit inside the fold?
    """)
    return


@app.cell
def _(FOLDS, GroupKFold, KNeighborsRegressor, StandardScaler, np):
    def val_rmse(
        X: np.ndarray, y: np.ndarray, d: np.ndarray, k: int = 5, folds: int = FOLDS
    ) -> float:
        """RMSE over folds of whole designs; scaler fit inside."""
        pred = np.empty(len(y))
        for fit, val in GroupKFold(folds).split(X, y, groups=d):
            sc = StandardScaler().fit(X[fit])  # fit rows only
            knn = KNeighborsRegressor(k).fit(sc.transform(X[fit]), y[fit])
            pred[val] = knn.predict(sc.transform(X[val]))
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    return (val_rmse,)


@app.cell
def _(itertools):
    # Every nonempty set of the eight inputs: 255 candidate models.
    subsets = [list(_c) for _r in range(1, 9) for _c in itertools.combinations(range(8), _r)]
    assert len(subsets) == 255 and subsets[-1] == list(range(8))
    return (subsets,)


@app.cell
def _(
    FOLDS,
    GroupKFold,
    K,
    KNeighborsRegressor,
    StandardScaler,
    X,
    cross_val_predict,
    d,
    make_pipeline,
    np,
    splits,
    subsets,
    val_rmse,
    y,
):
    _tr, _te = splits[0]
    assert not set(d[_tr]) & set(d[_te]), "a design is on both sides"
    _worst = 0.0
    for _c in subsets:
        _Xc = X[:, _c][_tr]
        _pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=K))
        _ref = cross_val_predict(_pipe, _Xc, y[_tr], groups=d[_tr], cv=GroupKFold(FOLDS))
        _ref_rmse = float(np.sqrt(np.mean((_ref - y[_tr]) ** 2)))
        _worst = max(_worst, abs(float(val_rmse(_Xc, y[_tr], d[_tr])) - _ref_rmse))
    # Slide "Validation flatters a little; test once": within 1e-12 MPa.
    assert _worst <= 1e-12, _worst
    {"input sets checked": len(subsets), "largest |difference| (MPa)": _worst}
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The search.** For each of the 30 splits, score all 255 input sets four ways: by
    validation inside the training designs, on the whole test set, and on each half of the test
    designs (cut at random; §5 uses the halves). The splits run in parallel processes (about
    5,000 model fits per split); expect well under a minute on a multicore machine.
    """)
    return


@app.cell
def _(N_JOBS, Parallel, SEED, X, d, delayed, knn_predict, np, rmse, splits, subsets, val_rmse, y):
    def _search_one_split(
        tr: np.ndarray, te: np.ndarray, halves_seed: int
    ) -> dict[str, np.ndarray]:
        """Score every input set on one split: validation, test, and the two test halves.

        Args:
            tr: Training rows.
            te: Test rows.
            halves_seed: Seed that cuts the test designs into two halves.

        Returns:
            ``val``, ``test``, ``half1``, ``half2``: RMSE per input set, shape ``(255,)``.
        """
        rng = np.random.default_rng(halves_seed)
        designs = np.unique(d[te])
        rng.shuffle(designs)
        in_half1 = np.isin(d[te], designs[: len(designs) // 2])
        out = {name: np.empty(len(subsets)) for name in ("val", "test", "half1", "half2")}
        for i, cols in enumerate(subsets):
            Xc = X[:, cols]  # columns first, then rows
            out["val"][i] = val_rmse(Xc[tr], y[tr], d[tr])
            pred, _ = knn_predict(Xc, y, tr, te)
            out["test"][i] = rmse(pred, y[te])
            out["half1"][i] = rmse(pred[in_half1], y[te][in_half1])
            out["half2"][i] = rmse(pred[~in_half1], y[te][~in_half1])
        return out

    _runs = Parallel(n_jobs=min(len(splits), N_JOBS))(
        delayed(_search_one_split)(_tr, _te, SEED + _s) for _s, (_tr, _te) in enumerate(splits)
    )
    search = {_name: np.stack([_r[_name] for _r in _runs]) for _name in _runs[0]}
    assert search["test"].shape == (len(splits), len(subsets)), search["test"].shape
    # Q4 in "Check yourself": validation compares the sets on all of a split's training tests,
    # about 770 on average; a test half holds about 130 (the same cut as above).
    _halves = []
    for _s, (_tr, _te) in enumerate(splits):
        _designs = np.unique(d[_te])
        np.random.default_rng(SEED + _s).shuffle(_designs)
        _in_half1 = int(np.isin(d[_te], _designs[: len(_designs) // 2]).sum())
        _halves += [_in_half1, len(_te) - _in_half1]
    assert round(float(np.mean([len(_tr) for _tr, _ in splits])), -1) == 770
    assert round(float(np.mean(_halves)), -1) == 130
    return (search,)


@app.cell
def _(FOLDS, INPUT_NAMES, np, pd, search, subsets, team_D):
    _star = search["val"].argmin(axis=1)  # chosen on validation
    _rows = np.arange(len(_star))
    _val_star, _test_star = search["val"][_rows, _star], search["test"][_rows, _star]
    team_C = float(search["test"].min(axis=1).mean())  # picked on the test set
    honest_C = float(_test_star.mean())
    _chosen = pd.Series([", ".join(INPUT_NAMES[_j] for _j in subsets[_i]) for _i in _star])
    assert np.allclose(search["test"][:, -1].mean(), team_D, rtol=0, atol=1e-12)
    # Slides "Your turn" (C 8.1) and "Validation flatters a little; test once".
    assert round(team_C, 1) == 8.1
    if FOLDS == 5:  # the slides' setting; the Modify exercise below changes it
        assert (round(float(_val_star.mean()), 1), round(honest_C, 1)) == (8.5, 8.7)
        assert int((_test_star > _val_star).sum()) == 19
        assert round(float((_test_star - _val_star).mean()), 1) == 0.2
        # Slide "The winner's score is partly luck" (notes): on the whole test set, Team C's
        # best of 255 sits 0.6 MPa below the validation-chosen set's test score.
        assert round(honest_C - team_C, 1) == 0.6
        assert (_chosen.value_counts().index[0], int(_chosen.value_counts().iloc[0])) == (
            "cement, slag, water, SP, age",
            11,
        )
    {
        "C: best of 255 on the test set": round(team_C, 3),
        "chosen on validation: validation score": round(float(_val_star.mean()), 3),
        "chosen on validation: test, once": round(honest_C, 3),
        "test above validation (splits of 30)": int((_test_star > _val_star).sum()),
        "set chosen most often": _chosen.value_counts().index[0],
    }
    return honest_C, team_C


@app.cell
def _(mo):
    mo.md(r"""
    **Bracket the error.** Above: predict the training mean for every test. Below: the same mix
    at the same age, tested more than once, does not give the same strength twice (Lecture 2's
    rough floor; groups whose strengths are all identical are one record entered twice and are
    left out). Only 9 such groups exist, from 3 mixes, so the floor is an indication, not a
    measurement.
    """)
    return


@app.cell
def _(PALETTE, df, honest_C, np, plt, rmse, search, splits, team_D):
    _y = df["strength"].to_numpy()
    _predict_mean = float(
        np.mean([rmse(np.full(len(_te), _y[_tr].mean()), _y[_te]) for _tr, _te in splits])
    )
    _stats = df.groupby(["mix", "age"])["strength"].agg(["size", "nunique", "std"])
    _rep = _stats[(_stats["size"] > 1) & (_stats["nunique"] > 1)]
    _floor = float(_rep["std"].median())
    _val_chosen = float(search["val"].min(axis=1).mean())
    # Slide "Validation flatters a little; test once": 17.0 above, 1.4 below (notes: 9 groups,
    # 20 tests, 3 mixes).
    assert round(_predict_mean, 1) == 17.0 and round(_floor, 1) == 1.4
    assert (len(_rep), int(_rep["size"].sum())) == (9, 20)
    assert _rep.index.get_level_values("mix").nunique() == 3

    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _labels = ["all 8 inputs", "chosen set: validation", "chosen set: test, once"]
    for _p, _v, _mk, _fill in zip(
        [2, 1, 0],
        [team_D, _val_chosen, honest_C],
        ["o", "s", "o"],
        ["full", "none", "full"],
        strict=True,
    ):
        _ax.plot(_v, _p, marker=_mk, ms=9, ls="none", color=PALETTE[0], fillstyle=_fill)
        _ax.text(_v, _p + 0.25, f"{_v:.1f}", ha="center")
    _ax.axvline(_predict_mean, color="#595959", ls=":", lw=1.5)
    _ax.axvline(_floor, color="#595959", ls="--", lw=1.5)
    _ax.text(_predict_mean - 0.3, -0.6, "predict the mean", ha="right", color="#595959")
    _ax.text(_floor + 0.3, -0.6, "repeat tests (rough)", color="#595959")
    _ax.set(
        yticks=[2, 1, 0],
        yticklabels=_labels,
        xlim=(0, 18),
        ylim=(-0.9, 2.6),
        xlabel="RMSE (MPa), designs held out",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Modify:** change `FOLDS` to 10 in the imports cell (the asserts that pin the slides'
    five-fold numbers then stand aside). Does the gap between the chosen set's validation and
    test scores shrink or grow? **Make:** score the 255 sets with $k=1$ and with
    $k=10$ as well (765 models). How does the gap change with three times as many models?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (one minute, in writing):

    1. **Many models.** Half the test designs pick the best of 255 input sets; the other half
       checks it. How far apart are the two scores?
    2. **Random split.** Hold out random tests instead of whole designs. Higher or lower error?
       Why?
    3. **New material.** Fit the model to mixes without fly ash, then predict mixes with fly
       ash. Does validation's promise hold?

    ### Many models: the winner's score is partly luck (the decisive figure)

    For every split, both ways round (half 1 picks and half 2 checks, then the reverse), and 20
    random orders of the 255 input sets, the best of the first $M$ on the picking half is scored
    on that half and on the other, fresh, half.
    """)
    return


@app.cell
def _(MARKERS, M_GRID, ORDERS, PALETTE, SEED, np, pd, plt, search, splits, subsets, team_C):
    _rng = np.random.default_rng(SEED)
    _picked, _fresh, _n = np.zeros(len(M_GRID)), np.zeros(len(M_GRID)), 0
    for _s in range(len(splits)):
        for _pick, _check in (("half1", "half2"), ("half2", "half1")):
            for _ in range(ORDERS):
                _order = _rng.permutation(len(subsets))
                for _j, _m in enumerate(M_GRID):
                    _first = _order[:_m]
                    _best = _first[np.argmin(search[_pick][_s, _first])]
                    _picked[_j] += search[_pick][_s, _best]
                    _fresh[_j] += search[_check][_s, _best]
                _n += 1
    _picked, _fresh = _picked / _n, _fresh / _n
    _gap = _fresh - _picked
    # Slide "The winner's score is partly luck": 12.95 and 13.00 at M = 1; 7.87 picked, 8.84
    # fresh, so 1.0 MPa too good (8.84 - 7.87 = 0.97).
    assert (round(_picked[0], 2), round(_fresh[0], 2)) == (12.95, 13.0)
    assert (round(_picked[-1], 2), round(_fresh[-1], 2), round(_gap[-1], 1)) == (7.87, 8.84, 1.0)
    assert np.all(np.diff(_gap[2:]) > 0)  # the gap only grows from M = 4 on
    assert round(team_C, 1) == 8.1

    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.plot(
        M_GRID,
        _fresh,
        color=PALETTE[1],
        ls="--",
        marker=MARKERS[1],
        label="on the other half (fresh)",
    )
    _ax.plot(
        M_GRID, _picked, color=PALETTE[0], marker=MARKERS[0], label="on the half that picked it"
    )
    _ax.set_xscale("log", base=2)
    _ax.set_xticks(M_GRID, [str(_m) for _m in M_GRID])
    _ax.set(xlabel="input sets tried, M", ylabel="RMSE of the winner (MPa)")
    _ax.legend()
    _fig.tight_layout()
    (
        pd.DataFrame(
            {
                "M": M_GRID,
                "picked": _picked.round(2),
                "fresh": _fresh.round(2),
                "gap": _gap.round(2),
            }
        ),
        _fig,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    With one model the two halves agree: a score used once is honest. Every extra model is
    another lottery ticket for the picking half, and the gap grows to 1.0 MPa. Both curves fall
    because trying more sets does find better models; only the fresh half says how much better.
    Team C's 8.1 MPa was a best-of-255 score on its test set; chosen on validation and tested
    once, the same search scores 8.7 MPa. That gap, 0.6 MPa, is smaller than the halves' 1.0: a
    half holds fewer designs than the whole test set, so its scores are noisier and its winner
    luckier.

    ### Random split: the same mixes on both sides
    """)
    return


@app.cell
def _(df, np, random_splits):
    # For each of the 30 random splits, the share of its test rows whose mix (or design) is
    # also among its training rows; then the mean and the range over the splits.
    _shares = {}
    for _by in ("mix", "design"):
        _g = df[_by].to_numpy()
        _shares[_by] = np.array(
            [np.isin(_g[_te], _g[_tr]).mean() for _tr, _te in random_splits(df)]
        )
    _summary = {
        _by: tuple(round(float(_f(_v)), 2) for _f in (np.mean, np.min, np.max))
        for _by, _v in _shares.items()
    }
    # Slide "Random splits reuse mixes: 8.9, not 9.5": on average 74% of test rows have their
    # identical mix in training (67-80% by split; Lecture 2's number), 88% their design (83-93%).
    assert _summary == {"mix": (0.74, 0.67, 0.8), "design": (0.88, 0.83, 0.93)}
    {
        f"random-split test rows whose {_by} is in training": (
            f"{100 * _m:.0f}% on average ({100 * _lo:.0f}% to {100 * _hi:.0f}%)"
        )
        for _by, (_m, _lo, _hi) in _summary.items()
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    The rows are tests; the independent units are designs. A random split puts the same mix at
    7, 28, and 90 days on both sides, and designs entered twice with rounded proportions, so the
    model recognizes them instead of predicting: 8.9 MPa against 9.5 with whole designs held out
    (Roberts et al., 2017, *Ecography* 40(8), on validation with dependent data). Lectures 2 and
    3 measure this on concrete and on bridges.

    ### New material: fly ash
    """)
    return


@app.cell
def _(
    GroupShuffleSplit,
    INPUTS,
    PALETTE,
    REPEATS,
    SEED,
    StandardScaler,
    X,
    d,
    df,
    knn_predict,
    np,
    plt,
    rmse,
    y,
):
    _fly = df["fly_ash"].to_numpy() > 0
    _home, away = np.flatnonzero(~_fly), np.flatnonzero(_fly)
    _gss = GroupShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
    _promised = [
        rmse(knn_predict(X, y, _home[_a], _home[_b])[0], y[_home[_b]])
        for _a, _b in _gss.split(_home, groups=d[_home])
    ]
    pred_away, _ = knn_predict(X, y, _home, away)
    promised, delivered = float(np.mean(_promised)), rmse(pred_away, y[away])
    _mean_away = rmse(np.full(len(away), y[_home].mean()), y[away])
    _scale = StandardScaler().fit(X[_home]).scale_[INPUTS.index("fly_ash")]
    # Slide "New materials void the promise".
    assert (len(_home), len(np.unique(d[_home])), len(away), len(np.unique(d[away]))) == (
        566,
        167,
        464,
        148,
    )
    assert (round(promised, 1), round(float(np.std(_promised)), 1)) == (9.8, 1.4)
    assert (round(delivered, 1), round(_mean_away, 1)) == (13.7, 14.4)
    assert X[_home][:, INPUTS.index("fly_ash")].std() == 0 and _scale == 1.0

    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    _ax.errorbar(promised, 1, xerr=np.std(_promised), fmt="o", ms=9, capsize=4, color=PALETTE[0])
    _ax.plot(delivered, 0, marker="s", ms=9, ls="none", color=PALETTE[1])
    _ax.axvline(_mean_away, color="#595959", ls=":", lw=1.5)
    _ax.text(_mean_away - 0.3, -0.5, "predict the mean", ha="right", color="#595959")
    _ax.set(
        yticks=[1, 0],
        yticklabels=["validation: no fly ash", "new: with fly ash"],
        xlim=(0, 18),
        ylim=(-0.8, 1.5),
        xlabel="RMSE (MPa), 5-NN fit without fly ash",
    )
    _fig.tight_layout()
    _fig
    return away, delivered, pred_away, promised


@app.cell
def _(mo):
    mo.md(r"""
    Fly ash is zero in every training test, so its column has zero spread; `StandardScaler`
    then leaves it unscaled (scale 1), and every fly-ash query sits the same distance above every
    training test in that coordinate. The column adds a constant to all distances and never
    changes which tests are nearest: the model answers as if the fly ash were not there. No
    split of the training data could warn us, because every validation case was drawn from mixes
    without fly ash.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Before you sign: four questions

    1. **The error on cases like this one?** Whole designs held out, tested once: 9.5 MPa over
       all ages, not 7.0, 8.1, or 8.9; and on the 28-day tests the decision is about, 9.9 MPa.
    2. **Is this case inside the data?** Fly-ash mixes: 13.7, not 9.8.
    3. **What does an error cost, and who carries it?** The traced design: predicted 38.1,
       measured 28.6, a wrong "yes".
    4. **Who is accountable?** The engineer who signs, not the model.

    What the honest error means for the decision: approve a 28-day test when its prediction
    minus a margin reaches $f'_c$, and tally both kinds of error over the held-out 28-day
    predictions of the 30 splits. The honest margin is the held-out error on those 28-day
    tests, 9.9 MPa: the mean of the 30 splits' RMSEs (pooled over the predictions, 9.92). The
    splits overlap, so the 3,179 predictions are of 425 distinct tests, each held out in
    several splits.
    """)
    return


@app.cell
def _(FC, NEAR_MPA, X, df, honest_28, knn_predict, np, rmse, splits, trace_pred, y):
    _age = df["age"].to_numpy()
    _p, _t, _rows = [], [], []
    for _tr, _te in splits:
        _pred, _ = knn_predict(X, y, _tr, _te)
        _p.append(_pred[_age[_te] == 28])
        _t.append(y[_te][_age[_te] == 28])
        _rows.append(_te[_age[_te] == 28])
    _p, _t, _rows = np.concatenate(_p), np.concatenate(_t), np.concatenate(_rows)
    assert _p.shape == _t.shape == _rows.shape, (_p.shape, _t.shape, _rows.shape)
    _weak = _t < FC
    # The honest margin is the mean of the splits' RMSEs; pooled, the predictions give 9.92.
    assert round(rmse(_p, _t), 2) == 9.92
    _rates = {}
    _margins = {
        "no margin": 0.0,
        "margin = 9.9 MPa": round(honest_28, 1),  # the honest margin, 28-day tests
        "margin = 16 MPa": 16.0,  # the far end of the slide's curves
    }
    for _name, _margin in _margins.items():
        _accept = _p - _margin >= FC
        _rates[_name] = {
            "weak approved": round(float(_accept[_weak].mean()), 3),
            "strong rejected": round(float((~_accept)[~_weak].mean()), 3),
        }
    # Slide "Before you sign" notes and "Someone else carries the error" (the curve's far end,
    # 16 MPa: 77.7% of strong mixes rejected).
    assert (len(_t), round(float(_weak.mean()), 2)) == (3179, 0.51)
    assert len(np.unique(_rows)) == 425  # distinct 28-day tests
    assert _rates["no margin"] == {"weak approved": 0.13, "strong rejected": 0.305}
    assert _rates["margin = 9.9 MPa"] == {"weak approved": 0.011, "strong rejected": 0.592}
    assert _rates["margin = 16 MPa"]["strong rejected"] == 0.777
    # Slide "A model's error sets the margin": the band is the traced design's prediction, to
    # the whole MPa, plus or minus NEAR_MPA (the figure script's rule): 36-40 MPa. Its 330
    # predictions are of 111 tests; 37% measured weak, the strongest 72.1 MPa.
    _lo, _hi = round(trace_pred) - NEAR_MPA, round(trace_pred) + NEAR_MPA
    assert (_lo, _hi) == (36.0, 40.0)
    _near = (_p >= _lo) & (_p <= _hi)
    assert (int(_near.sum()), round(float(_weak[_near].mean()), 2)) == (330, 0.37)
    assert len(np.unique(_rows[_near])) == 111
    assert round(float(_t[_near].max()), 1) == 72.1
    _rates
    return


@app.cell
def _(mo):
    mo.md(r"""
    An honest error can say that a model is not ready. With the honest margin, 9.9 MPa, 1.1% of
    weak predictions are approved but 59.2% of strong ones are rejected: this model could screen
    designs, not approve them. Lecture 2 cuts the error with engineering knowledge.

    ### When not to use ML

    - A trusted rule already answers, inside its range: a code provision, Abrams' law, a
      mechanics model.
    - The case lies outside the data: new materials, ages, climates, or regions.
    - The error cannot be measured honestly: no held-out cases like the future.
    - Nobody can check the answer before it matters, and a wrong "yes" is costly. Test first;
      approval of concrete rests on cylinder tests.

    A model can still help in these cases as a screen, a first guess, or a check on another
    method, with the decision resting on something else.

    ### Impact: someone else carries the error

    An optimistic error moves risk from the modeler to the owner and the public (ethical). A
    model is honest only for cases like its data: producers using fly ash would meet 13.7 MPa,
    not 9.8 (social). The honest margin rejects 59.2% of strong mixes: redesigns, more tests,
    more cement (economic and environmental). The Campus AI Curriculum Task Force asks that
    these four considerations be integrated throughout AI curricula rather than isolated in
    stand-alone courses; this course gives each unit one such note.

    ### Where ML lives today in engineering practice

    - **Flood warnings.** An AI model forecasts extreme floods in ungauged basins up to five days
      ahead, with reliability similar to or better than current nowcasts; it runs in an
      operational system with public forecasts in over 80 countries (Nearing et al., 2024,
      *Nature* 627). Lectures 11 and 20.
    - **Weather.** ECMWF's machine-learned forecasting system (AIFS) became operational on
      25 February 2025, running side by side with its physics-based model (ECMWF news, 2025).
      Lecture 17.

    Today's questions apply to both: were they tested on cases like the ones they face (later
    years for a forecast, whole basins for ungauged basins)? Who signs the decision?

    ### Vote: would you sign this mix?

    A model fit to mixes without fly ash predicts 47.3 MPa for a new design with 77 kg/m³ of fly
    ash. $f'_c = 35$ MPa. Validation, designs held out: 9.8 MPa.

    - **A.** Sign: 47.3 − 9.8 = 37.5 MPa clears $f'_c$.
    - **B.** Sign if a second model, fit to the same tests, agrees.
    - **C.** Not yet: no training test had fly ash. Cast cylinders first.

    Vote alone, discuss with a neighbor, vote again; then run the cell.
    """)
    return


@app.cell
def _(FC, LATER_EXAMPLE_ROWS, away, d, delivered, df, pd, pred_away, promised, y):
    _cand = pd.DataFrame(
        {
            "row": away,
            "pred": pred_away,
            "y": y[away],
            "age": df["age"].to_numpy()[away],
            "design": d[away],
        }
    )
    _cand = _cand[(_cand["age"] == 28) & ~_cand["design"].isin(set(d[LATER_EXAMPLE_ROWS]))]
    _cand = _cand[(_cand["pred"] - round(promised, 1) >= FC) & (_cand["y"] < FC)]
    _pick = _cand.loc[((_cand["pred"] - _cand["y"]) - delivered).abs().idxmin()]
    _row = int(_pick["row"])
    _pred = round(float(_pick["pred"]), 1)
    # Slide "Vote: would you sign this mix?" and its notes.
    assert (_row, len(_cand)) == (922, 8)
    assert (_pred, round(float(df["fly_ash"].iloc[_row]))) == (47.3, 77)
    assert round(_pred - round(promised, 1), 1) == 37.5
    assert round(_pred - round(delivered, 1), 1) == 33.6
    assert round(float(_pick["y"]), 1) == 33.8 and _pick["y"] < FC
    _binder = float(df["cement"].iloc[_row] + df["slag"].iloc[_row] + df["fly_ash"].iloc[_row])
    assert round(float(df["fly_ash"].iloc[_row]) / _binder, 2) == 0.18
    {
        "predicted (MPa)": _pred,
        "minus validated error": round(_pred - round(promised, 1), 1),
        "minus error on fly-ash mixes": round(_pred - round(delivered, 1), 1),
        "measured (MPa)": round(float(_pick["y"]), 1),
    }
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Explanation of the vote": mo.md(
                r"""
                **C.** The 9.8 MPa was measured on mixes like the training mixes; this one is
                outside the data. On fly-ash mixes the same model errs by 13.7 MPa, and
                47.3 − 13.7 = 33.6 MPa does not clear 35. **B** fails because a second model fit
                to the same tests shares the blind spot. The mix is real: it measured 33.8 MPa,
                below $f'_c$. It was chosen by a stated rule: among 28-day fly-ash tests that
                option A would approve but that measured below 35 MPa, the one whose error is
                closest to 13.7 MPa. What would change the answer: tests of fly-ash mixes in
                training, or a cheaper wrong "yes".
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the course runs

    **Every lecture runs the same steps:** 0 warm-up (recall earlier lectures), 1 the question
    and your attempt, 2 one picture, 3 derive from the assumption, 4 build and verify (predict
    first), 5 break it (predict, then reveal), 6 decide (vote, discuss, revote), 7 check
    yourself. Why:

    - Recall beats rereading at a delay (Roediger & Karpicke, 2006, *Psychological Science*
      17(3)).
    - Problem solving before instruction beats instruction first, Hedges' $g = 0.36$ over 53
      studies (Sinha & Kapur, 2021, *Review of Educational Research* 91(5)).
    - Students who only watch a demonstration understand it no better than students who never
      see it; predicting the outcome first helps (Crouch, Fagen, Callan & Mazur, 2004,
      *American Journal of Physics* 72(6)).
    - Students in lecture-only classes were 1.5 times as likely to fail (Freeman et al., 2014,
      *PNAS* 111(23)).
    - Students in active classes learn more but feel they learn less, partly because the effort
      feels like confusion (Deslauriers et al., 2019, *PNAS* 116(39)). Expect that feeling; it
      is the work of learning.

    **Grades.** Five homeworks, each a 100-point core plus stretch goals, earning experience
    points (XP); three CBTF exams, the best two counting; a final project on custom CEE data with
    a five-minute oral defense:
    $$\text{Grade}=\frac{\text{XP}+\text{best exam}+\text{second-best exam}+\text{final project}}
    {\max(\text{XP},\,\text{XP}_{\text{target}})+300},$$
    with $\text{XP}_{\text{target}} = 400$ for 3 credits and 525 for 4.

    **AI use is declared on every task**, with the Grainger College of Engineering's four levels
    (AI Guidance, 2026):

    | Component | AI-use level | Why |
    |:--|:--|:--|
    | CBTF exams | Restricted | They measure unaided mastery. |
    | Homework cores: from-scratch code and analysis | Limited Use | AI may explain and help debug; producing the code and the analysis is how you learn. |
    | Critique-the-machine problems | Use When Instructed | You work with AI output by design. |
    | Stretch goals and final project | All Use, with a disclosure log | The grade rests on your data, your verification, and your oral defense. |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question** (one minute, no notes): a city wants a model that flags water mains likely
    to break next year. Name its input, its output, and the decision it feeds. How would you
    split the records so the test set looks like next year?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "One answer": mo.md(
                r"""
                Input: material, diameter, age, soil, pressure, past breaks of each main.
                Output: the chance it breaks next year. Decision: which mains to inspect or
                replace first, within a budget. Split by time: fit and validate on the years up
                to the last one, test once on the last year; and keep each main, or each
                neighborhood, on one side. A random split of main-years would put a main's own
                history on both sides.
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
            "Q1. Why is Team A's training error 7.0 MPa and not zero?": mo.md(
                r"""
                Scored on its own training tests, each test is one of its own five neighbors,
                so one fifth of every prediction is the answer itself; the other four neighbors
                are other tests. With one neighbor the training error would be close to zero
                (only tests with identical inputs and different strengths would err). Either
                way it measures memory, not prediction.
                """
            ),
            "Q2. Why fit the scaler inside each validation fold?": mo.md(
                r"""
                The means and spreads are parameters of the encoder ($\phi$). Fitting them on
                all training rows lets the validation fold shape the representation it is
                scored with, a small leak. A pipeline that is refit inside each fold, as
                `val_rmse` and `cross_val_predict` do, avoids it.
                """
            ),
            "Q3. You looked at the test set, changed the model, and looked again. Now what?": mo.md(
                r"""
                The test set has become a validation set: its score is now optimistic, by an
                amount you cannot know. Report it as such, and get an honest number from data
                that played no part in any choice: a fresh set of new designs, or nested
                cross-validation, where an outer loop holds out a test fold and an inner loop
                does all the choosing.
                """
            ),
            "Q4. Why 0.2 MPa of optimism from validation, but 1.0 MPa from a test half?": mo.md(
                r"""
                Optimism grows with the number of models compared and with the noise of each
                score. Five folds over all training designs compare the 255 sets on about 770
                tests; a test half has about 130. Each score on a half is noisier, so the best
                of 255 is further from its true error (Cawley & Talbot, 2010).
                """
            ),
            "Q5. What does an honest test look like for a weather or flood forecast?": mo.md(
                r"""
                Later years, held out after every choice was made, at the lead time of the use;
                for warnings in ungauged basins, whole basins held out as well. A random split
                of days would put the same storm on both sides.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Choosing on the test set.** Any choice (inputs, $k$, scaling, features, stopping) made on
      test scores makes them optimistic. One test set, looked at once, after every choice.
    - **Reporting the best validation score as the final error.** It is optimistic too; report a
      test score, or nested cross-validation.
    - **Scaler fit on all data.** Fit preprocessing inside each training fold (a pipeline).
    - **Random splits of repeated measurements.** Tests of one design at several ages, or one
      design entered twice, must be held out together (`GroupKFold`, `GroupShuffleSplit`).
    - **Inputs that never varied in training.** A constant column carries no information; a
      model cannot learn what fly ash does from mixes without fly ash, and it will not warn you.
    - **Ties decide k-NN predictions.** With duplicate records, which tied neighbor is taken
      depends on the search tree and even on the memory layout of the array. Fix the order of
      operations (columns, then rows) when results must reproduce, and count the ties.
    - **Training error as evidence.** Scored on its own training data, a model reports memory.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §1.1.3** — training error versus test error, why a model chosen to do well on its
      training data looks better there, and cross-validation as repeated splits into training
      and validation sets. Forsyth notes that a validation set is sometimes, confusingly, called
      a test set; in this course the test set is the one looked at once.
    - **Forsyth §1.3.1 and Procedure 1.1** — choosing a model by cross-validation inside the
      training set, refitting on all of it, and evaluating once on a held-out test set: exactly
      the protocol of §4, with a random split where we hold out whole designs.
    - **Forsyth §3.1** — held-out loss predicts test loss: the held-out loss is a sample mean,
      and Chebyshev's inequality bounds how far it strays from the expected loss when the
      held-out items are independent draws from the test distribution and were not used to
      build the predictor. Both conditions are the assumption of §3.
    - **Forsyth §11.1.3** — choosing a regression model by cross-validation (Lecture 5 returns to
      it with ridge and lasso).
    - **Exercise.** Repeat the winner's-curse curve with test halves of 20 designs instead of
      about 40. Predict first: does the gap at $M = 255$ grow or shrink?
    - **Exercise.** Replace the single test split by nested cross-validation (five outer folds
      of designs, five inner folds for choosing). Compare its estimate with the 8.7 MPa of §4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Machine learning picks a rule from examples; it is judged on cases it has not seen.
    - Frame every model by its input, its output, and the decision it feeds.
    - Training fits, validation chooses, the test set reports once: the best of many scores is
      too good.
    - A held-out error is honest only for cases like the held-out ones: whole designs, no new
      materials.
    - Before you sign: the error on cases like this one, inside the data, the cost of an error
      and who carries it, who is accountable.

    **Next:** nearest neighbors, and what makes two concrete mixes similar.
    """)
    return


if __name__ == "__main__":
    app.run()
