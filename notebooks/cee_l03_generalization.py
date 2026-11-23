"""Applied Machine Learning for CEE — Lecture 3: Generalization and Validation.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l03_generalization.py

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
    # Applied Machine Learning for CEE — Lecture 3: Generalization and Validation

    **Slides:** *Applied Machine Learning for CEE*, Lecture 3 (`lectures/cee_l03_generalization`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §1.1.3 (overfitting and cross-validation), §3.1 (held-out loss predicts test loss), §11.1.1 (bias and variance).
    **Data:** FHWA National Bridge Inventory (NBI), Illinois, 2014–2024 (every two years); fetch it with `bash scripts/fetch_nbi.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. The whole
    notebook takes about four minutes to run on a CPU (3.6 min measured; 4.4 min on four BLAS
    threads): each cross-validation searches the neighbors of tens of thousands of records.

    **Your tasks** (from the slides): write `cv_predict`, match scikit-learn, work the five
    records by hand, and reproduce the random, county, and 2024 errors and the leak.
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
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    FOLDS = 5  # five folds in every cross-validation
    KS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100])
    K = 50  # between the honest choices, k = 30 and 70
    TRAIN_YEARS = (2014, 2016, 2018)  # what an engineer had in 2019
    LATER_YEARS = (2020, 2022, 2024)
    TEST_YEAR = 2024  # the test set: scores the k validation chose
    POOR = 4  # a lowest rating of 4 or less is poor
    TWIN_KM = 0.2  # parallel structures closer than this
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    LINESTYLES = ["-", "--", "-.", ":"]
    return (
        FOLDS,
        GroupKFold,
        K,
        KFold,
        KNeighborsRegressor,
        KS,
        LATER_YEARS,
        LINESTYLES,
        MARKERS,
        NearestNeighbors,
        PALETTE,
        POOR,
        Path,
        SEED,
        StandardScaler,
        TEST_YEAR,
        TRAIN_YEARS,
        TWIN_KM,
        cross_val_predict,
        make_pipeline,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: questions from Lectures 1–2

    On your own, no notes, two minutes:

    1. You tried ten values of $k$ and kept the one with the best test score. What went wrong?
    2. Why did water carry only 1.2% of the raw distance between two concrete mixes?
    3. A random split of the concrete tests gave 5.7 MPa; holding out whole designs gave
       6.9 MPa. Which do you report to a client, and why?

    **Today:** how wrong will a model of bridge condition be on the next bridge?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. The test set chose $k$, so it is no longer a test: the best of ten scores is
                   optimistic. Choose $k$ on validation data and look at the test set once, at
                   the end (Lecture 1).
                2. A raw distance weighs each input by its variance, and water varies little in
                   kg/m³ (only superplasticizer varies less); cement and the aggregates
                   dominate. Units, not importance, decided the distance (Lecture 2).
                3. Designs held out: the client's mix is a new design, and the random split put
                   the same design, tested at other ages or entered twice, in training
                   (Lecture 2). Question 3 is today's lecture in miniature.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Inspectors rate every highway bridge's deck, superstructure, and substructure from 0 to 9,
    usually every two years (FHWA, 1995: 7 good, 5 fair, 4 poor, 3 serious). The lowest of the
    three at 4 or below means **poor**. Owners must decide which bridges to inspect or
    rehabilitate first, and a model could predict the rating from the inventory: between
    inspections, or where records are missing.

    The two errors are not equal. A wrong **"fair"** lets a poor bridge wait, perhaps until
    load posting or closure, and the public carries the risk. A wrong **"poor"** sends a crew
    and a budget to a sound bridge. Before anyone uses such a model, one number matters: **how
    wrong will it be on the bridges where we use it?**

    **The data.** The NBI for Illinois, 2014–2024, every two years: one record per bridge and
    inventory year, culverts aside. **The inputs** (12): age, years since the last
    reconstruction, ln traffic (ADT), percent trucks, ln length, deck width, four material flags
    (concrete, steel, prestressed, timber), and latitude and longitude. **The target:** the
    lowest of the three ratings.

    **The setting.** We stand in 2019 with the 2014, 2016, and 2018 inventories: the training
    records. The 2024 inventory is the test set: it scores the choices and makes none. One
    subtlety: a bridge on a 48-month inspection interval reports the *same* inspection in two
    consecutive inventories, so we keep each inspection once.
    """)
    return


@app.cell
def _(Path, np, pd):
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
    LEAKY = ["inspection_interval", "structural_eval"]
    INSPECTION = ["structure", "inspection_year", "inspection_month"]
    MATERIAL = {
        "1": "concrete",
        "2": "concrete",
        "3": "steel",
        "4": "steel",
        "5": "prestressed",
        "6": "prestressed",
        "7": "timber",
    }
    DESIGN = {
        "01": "slab",
        "02": "beams",
        "04": "tee beams",
        "05": "box beams",
        "07": "frame",
        "09": "truss",
        "10": "truss",
        "11": "arch",
        "22": "channel beams",
    }
    # Illinois counties in alphabetical order; FIPS codes are 001, 003, ..., 203 in this order.
    COUNTY_NAMES = [
        "Adams",
        "Alexander",
        "Bond",
        "Boone",
        "Brown",
        "Bureau",
        "Calhoun",
        "Carroll",
        "Cass",
        "Champaign",
        "Christian",
        "Clark",
        "Clay",
        "Clinton",
        "Coles",
        "Cook",
        "Crawford",
        "Cumberland",
        "DeKalb",
        "De Witt",
        "Douglas",
        "DuPage",
        "Edgar",
        "Edwards",
        "Effingham",
        "Fayette",
        "Ford",
        "Franklin",
        "Fulton",
        "Gallatin",
        "Greene",
        "Grundy",
        "Hamilton",
        "Hancock",
        "Hardin",
        "Henderson",
        "Henry",
        "Iroquois",
        "Jackson",
        "Jasper",
        "Jefferson",
        "Jersey",
        "Jo Daviess",
        "Johnson",
        "Kane",
        "Kankakee",
        "Kendall",
        "Knox",
        "Lake",
        "LaSalle",
        "Lawrence",
        "Lee",
        "Livingston",
        "Logan",
        "McDonough",
        "McHenry",
        "McLean",
        "Macon",
        "Macoupin",
        "Madison",
        "Marion",
        "Marshall",
        "Mason",
        "Massac",
        "Menard",
        "Mercer",
        "Monroe",
        "Montgomery",
        "Morgan",
        "Moultrie",
        "Ogle",
        "Peoria",
        "Perry",
        "Piatt",
        "Pike",
        "Pope",
        "Pulaski",
        "Putnam",
        "Randolph",
        "Richland",
        "Rock Island",
        "St. Clair",
        "Saline",
        "Sangamon",
        "Schuyler",
        "Scott",
        "Shelby",
        "Stark",
        "Stephenson",
        "Tazewell",
        "Union",
        "Vermilion",
        "Wabash",
        "Warren",
        "Washington",
        "Wayne",
        "White",
        "Whiteside",
        "Will",
        "Williamson",
        "Winnebago",
        "Woodford",
    ]
    COUNTY = {f"{2 * _i + 1:03d}": _n for _i, _n in enumerate(COUNTY_NAMES)}

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
        """Load the bridge-year records and add the inputs (same as the slides' loader).

        Args:
            path: The CSV written by ``scripts/fetch_nbi.sh``.

        Returns:
            Records with complete inputs, with ``FEATURES``, ``material``, and a planar grid
            in km (``x_km``, ``y_km``).
        """
        text = ["structure", "county", "district", "owner", "kind", "design", "parallel"]
        frame = pd.read_csv(path, dtype=dict.fromkeys(text, str))
        assert set(frame["county"]) <= set(COUNTY), "unknown county code"
        frame["age"] = (frame["inspection_year"] - frame["year_built"]).clip(lower=0)
        last_work = np.fmax(frame["year_built"], frame["year_reconstructed"].fillna(0))
        frame["years_since_work"] = (frame["inspection_year"] - last_work).clip(lower=0)
        frame["log_adt"] = np.log1p(frame["adt"])
        frame["log_length"] = np.log(frame["length_m"].clip(lower=1.0))
        frame["material"] = frame["kind"].map(MATERIAL).fillna("other")
        for _m in ("concrete", "steel", "prestressed", "timber"):
            frame[_m] = (frame["material"] == _m).astype(float)
        frame = frame.dropna(subset=FEATURES).reset_index(drop=True)  # complete inputs
        frame["x_km"] = (frame["lon"] + 89.0) * 111.32 * np.cos(np.deg2rad(40.0))
        frame["y_km"] = (frame["lat"] - 40.0) * 110.57
        return frame

    def unique_inspections(records: pd.DataFrame) -> pd.DataFrame:
        """Keep one record per inspection, from the first inventory that reports it."""
        first = records.sort_values("nbi_year", kind="stable").drop_duplicates(INSPECTION)
        return first.sort_index().reset_index(drop=True)

    def later_records(frame: pd.DataFrame, train: pd.DataFrame, year: int) -> pd.DataFrame:
        """Records of a later inventory whose inspection is not already in training."""
        rec = frame[frame["nbi_year"] == year]
        seen = pd.MultiIndex.from_frame(train[INSPECTION])
        return rec[~pd.MultiIndex.from_frame(rec[INSPECTION]).isin(seen)].reset_index(drop=True)

    def encode(records: pd.DataFrame, extra: tuple[str, ...] = ()) -> np.ndarray:
        """The inputs as a float matrix (scaled later, inside each fold).

        Args:
            records: Bridge records.
            extra: Further columns to append (the leakage experiment).

        Returns:
            Array of shape ``(len(records), 12 + len(extra))``.
        """
        X = records[FEATURES + list(extra)].to_numpy(np.float64)
        assert X.shape == (len(records), len(FEATURES) + len(extra)), X.shape
        return X

    return (
        COUNTY,
        DESIGN,
        LEAKY,
        encode,
        find_nbi_csv,
        later_records,
        load_bridges,
        unique_inspections,
    )


@app.cell
def _(TEST_YEAR, TRAIN_YEARS, find_nbi_csv, later_records, load_bridges, pd, unique_inspections):
    df = load_bridges(find_nbi_csv())
    # A few bridges carry two county codes over the years (10 in training, 30 records). The
    # county folds place every record of a bridge in the fold of its first record's county.
    if not df["nbi_year"].is_monotonic_increasing:
        raise ValueError("records must be in inventory-year order")
    df["county_fold"] = df.groupby("structure")["county"].transform("first")
    _raw = df[df["nbi_year"].isin(TRAIN_YEARS)]
    train = unique_inspections(_raw)
    y_train = train["rating"].to_numpy(float)
    _test = later_records(df, train, TEST_YEAR)
    _summary = {
        "bridge-year records, 2014–2024": len(df),
        "bridges": df["structure"].nunique(),
        "counties": df["county"].nunique(),
        "training inventory records (2014–2018)": len(_raw),
        "of which repeat an earlier inspection": len(_raw) - len(train),
        "training inspections": len(train),
        "training bridges": train["structure"].nunique(),
        "2024 test records": len(_test),
        "mean rating, 2014": round(df.loc[df["nbi_year"] == 2014, "rating"].mean(), 2),
        "mean rating, 2024": round(df.loc[df["nbi_year"] == 2024, "rating"].mean(), 2),
        "poor share 2024 (%)": round(
            100 * (df.loc[df["nbi_year"] == 2024, "rating"] <= 4).mean(), 1
        ),
    }
    # Data slide and notes: 131,851 records of 23,464 bridges in 102 counties; training: 65,928
    # inventory records, 10,773 repeated inspections, 55,155 inspections of 22,720 bridges;
    # 21,964 test records; mean rating 6.48 in 2014 and 6.13 in 2024; 10.5% poor in 2024.
    assert (len(df), df["structure"].nunique(), df["county"].nunique()) == (131851, 23464, 102)
    assert (len(_raw), len(_raw) - len(train), len(train)) == (65928, 10773, 55155)
    assert train["structure"].nunique() == 22720 and len(_test) == 21964
    assert (_summary["mean rating, 2014"], _summary["mean rating, 2024"]) == (6.48, 6.13)
    assert _summary["poor share 2024 (%)"] == 10.5
    pd.Series(_summary, name="value").to_frame()
    return df, train, y_train


@app.cell
def _(KNeighborsRegressor, PALETTE, SEED, df, np, pd, plt):
    # Slide "Rank bridges before they fail": the whole 2018 inventory, one record per bridge,
    # 9.6% poor. (The training set's 2018 records are only the inspections new in 2018, which
    # leave out many bridges on a 48-month interval, and so over-represent poor ones.)
    _rec = df[df["nbi_year"] == 2018]
    assert _rec["structure"].is_unique and (_rec["rating"] % 1 == 0).all()
    _counts = _rec["rating"].astype(int).value_counts().sort_index()
    assert len(_rec) == 22_015 and round(100 * float((_rec["rating"] <= 4).mean()), 1) == 9.6
    assert _counts.to_dict() == {
        0: 29,
        1: 12,
        2: 145,
        3: 477,
        4: 1444,
        5: 3655,
        6: 5861,
        7: 5349,
        8: 4707,
        9: 336,
    }

    # Slide "Small k follows noise; huge k, the mean": k-NN on age alone, six samples of 1,000
    # bridges of the 2018 inventory; variance across samples and squared bias from the mean
    # rating at each age over all its bridges.
    _age, _y = _rec["age"].to_numpy(float), _rec["rating"].to_numpy(float)
    _grid = np.arange(0, 111, dtype=float)
    _truth = pd.Series(_y).groupby(np.clip(_age, 0, 110).round()).mean()
    _truth = _truth.reindex(_grid).interpolate().rolling(5, center=True, min_periods=1).mean()
    _rng = np.random.default_rng((SEED, 3))
    _samples = [_rng.choice(len(_rec), 1_000, replace=False) for _ in range(6)]
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    _parts = {}
    _ranges = {}
    for _ax, _k in zip(_axes, (1, 30, 1_000), strict=True):
        _fits = np.array(
            [
                KNeighborsRegressor(n_neighbors=_k)
                .fit(_age[_i, None], _y[_i])
                .predict(_grid[:, None])
                for _i in _samples
            ]
        )
        _parts[_k] = (
            round(float(_fits.var(axis=0).mean()), 2),
            round(float(((_fits.mean(axis=0) - _truth.to_numpy()) ** 2).mean()), 2),
        )
        _ranges[_k] = (round(float(_fits.min()), 2), round(float(_fits.max()), 2))
        for _f in _fits:
            _ax.plot(_grid, _f, color=PALETTE[0], lw=1, alpha=0.5)
        _ax.plot(_grid, _truth.to_numpy(), "k--", lw=2)
        _ax.set(
            title=f"k = {_k:,}: variance {_parts[_k][0]}, bias² {_parts[_k][1]}",
            xlabel="age",
            ylim=(-0.5, 9.5),
        )
    assert _parts == {1: (0.99, 0.22), 30: (0.04, 0.01), 1_000: (0.0, 1.09)}
    # The k = 1 fits reach every rating from 0 to 9 (the slide's axis shows all of them); k =
    # 1,000 sits near 6.3; the dashed mean falls from 8.39 at age 0 to 4.77 at 110 years.
    assert _ranges[1] == (0.0, 9.0)
    assert _ranges[1_000] == (6.27, 6.38)
    assert (round(float(_truth.iloc[0]), 2), round(float(_truth.iloc[-1]), 2)) == (8.39, 4.77)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Helpers used in every section.** The next cell defines the folds, a fit-and-predict
    for every $k$ at once (one neighbor search, then running sums), and the twin finder. The
    cross-validation loop itself is §4's `cv_predict`; it comes first here because the attempt
    and §3 already use it. Skip this cell on a first read.
    """)
    return


@app.cell
def _(FOLDS, GroupKFold, KFold, KS, NearestNeighbors, SEED, StandardScaler, TWIN_KM, np, pd):
    def county_fold_of(records: pd.DataFrame) -> dict[str, int]:
        """County code to fold: ``GroupKFold`` deals the counties by their records' own codes.

        Args:
            records: Training records.

        Returns:
            The fold of every county code in ``records``.
        """
        _fold_of = {}
        for _j, (_, _te) in enumerate(GroupKFold(FOLDS).split(records, groups=records["county"])):
            _fold_of |= dict.fromkeys(records["county"].iloc[_te].unique(), _j)
        return _fold_of

    def county_fold_index(records: pd.DataFrame, fold_of: dict[str, int]) -> np.ndarray:
        """Each record's county fold: the fold of its bridge's first county (``county_fold``).

        Args:
            records: Records with the ``county_fold`` column.
            fold_of: Output of ``county_fold_of``.

        Returns:
            One fold number per record.

        Raises:
            KeyError: If a bridge's first county has no fold.
        """
        _fold = records["county_fold"].map(fold_of)
        if _fold.isna().any():
            _missing = sorted(set(records.loc[_fold.isna(), "county_fold"]))
            raise KeyError(f"counties without a fold: {_missing}")
        return _fold.to_numpy(int)

    def make_folds(records: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Five folds: ``"random"`` records, whole ``"bridge"``s, or whole ``"county"``s.

        County folds: ``GroupKFold`` deals the counties to folds by the records' own county
        codes, then every record goes to the fold of its bridge's first county, so a bridge
        recorded under two codes is never split across folds.

        Args:
            records: Training records, with the ``county_fold`` column.
            scheme: ``"random"``, ``"bridge"``, or ``"county"``.

        Returns:
            ``(train, held_out)`` index pairs.

        Raises:
            ValueError: On an unknown scheme.
        """
        if scheme == "random":
            return list(KFold(FOLDS, shuffle=True, random_state=SEED).split(records))
        if scheme == "bridge":
            return list(GroupKFold(FOLDS).split(records, groups=records["structure"]))
        if scheme != "county":
            raise ValueError(f"unknown scheme {scheme!r}")
        _fold = county_fold_index(records, county_fold_of(records))
        _rows = np.arange(len(records))
        return [(_rows[_fold != _j], _rows[_fold == _j]) for _j in range(FOLDS)]

    def holdout_predict(
        X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, ks: np.ndarray = KS
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit on one set, predict another, for every k in ``ks``.

        Returns:
            Predictions ``(len(X_te), len(ks))``; neighbor indices and distances,
            ``(len(X_te), max(ks))``.
        """
        sc = StandardScaler().fit(X_tr)
        nn = NearestNeighbors(n_neighbors=int(ks.max()), algorithm="brute")
        dist, idx = nn.fit(sc.transform(X_tr)).kneighbors(sc.transform(X_te))
        total = np.cumsum(y_tr[idx], axis=1)
        return total[:, ks - 1] / ks, idx, dist

    def rmse_per_k(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RMSE of each column of ``pred`` against ``y``.

        Args:
            pred: Predictions, one column per k, shape ``(N, len(ks))``.
            y: Ratings, shape ``(N,)``.

        Returns:
            One RMSE per column.

        Raises:
            ValueError: On any other shapes: a ``(N, 1)`` target would broadcast silently.
        """
        if pred.ndim != 2 or y.ndim != 1 or pred.shape[0] != y.shape[0]:
            raise ValueError(f"rmse_per_k of shapes {pred.shape} and {y.shape}")
        return np.sqrt(((pred - y[:, None]) ** 2).mean(axis=0))

    def k_index(k: int) -> int:
        """Column of ``k`` in the prediction matrices."""
        return int(np.flatnonzero(k == KS)[0])

    def find_twins(records: pd.DataFrame) -> pd.DataFrame:
        """Parallel structures (Item 101 L or R) built the same year, within ``TWIN_KM``."""
        par = records[records["parallel"].isin(["R", "L"])]
        xy = par[["x_km", "y_km"]].to_numpy()
        dist, idx = NearestNeighbors(n_neighbors=3).fit(xy).kneighbors(xy)
        rows, own = par.index.to_numpy(), np.arange(len(par))
        first = np.where(idx[:, 0] == own, 1, 0)  # coincident points: skip the record itself
        out = pd.DataFrame({"row": rows, "twin_row": rows[idx[own, first]], "km": dist[own, first]})
        built = records["year_built"]
        same = built.loc[out["row"]].to_numpy() == built.loc[out["twin_row"]].to_numpy()
        return out[(out["km"] < TWIN_KM) & same].reset_index(drop=True)

    return (
        county_fold_index,
        county_fold_of,
        find_twins,
        holdout_predict,
        k_index,
        make_folds,
        rmse_per_k,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: rate bridge Q

    Q is a bridge in the 2018 inventory. Four training records follow it, all with the same 12
    inputs. On your own, two minutes: **estimate Q's 2018 rating, and write down which rows you
    relied on.** Then: if Q's county had never shared its records, which rows would not exist?
    Cover them and estimate again.
    """)
    return


@app.cell(hide_code=True)
def _(
    COUNTY, DESIGN, K, POOR, encode, find_twins, holdout_predict, make_folds, np, pd, train, y_train
):
    # Same selection as the slides (figure script, `attempt_and_trace`). The code is hidden in
    # the editor because it uses the neighbors, which would give the answer away.
    _X = encode(train)
    _st = train["structure"].to_numpy()
    _folds = {_s: make_folds(train, _s) for _s in ("random", "county")}
    _fold_of = {}
    for _s, _fs in _folds.items():
        _fold_of[_s] = np.empty(len(train), dtype=int)
        for _j, (_, _te) in enumerate(_fs):
            _fold_of[_s][_te] = _j

    def _neighbors(row: int, scheme: str, k: int) -> np.ndarray:
        _tr, _ = _folds[scheme][_fold_of[scheme][row]]
        _, _idx, _ = holdout_predict(_X[_tr], y_train[_tr], _X[[row]], ks=np.array([k]))
        return _tr[_idx[0]]

    # Q: among poor twin bridges (2018, >= 50,000 vehicles a day) in the first county block whose
    # nearest record under random folds is their twin, the one where the two splits' 1-NN
    # predictions differ most; ties go to the busier bridge.
    _tw = find_twins(train[train["nbi_year"] == 2018])
    _held0 = set(_folds["county"][0][1])
    _cand = _tw[
        _tw["row"].isin(_held0)
        & (y_train[_tw["row"]] <= POOR)
        & (train["adt"].to_numpy()[_tw["row"]] >= 50_000)
    ].copy()
    _cand["r1"] = [int(_neighbors(_r, "random", 1)[0]) for _r in _cand["row"]]
    _cand["c1"] = [int(_neighbors(_r, "county", 1)[0]) for _r in _cand["row"]]
    _cand = _cand[_st[_cand["r1"]] == _st[_cand["twin_row"]]]
    _cand["gap"] = np.abs(y_train[_cand["r1"]] - y_train[_cand["c1"]])
    _cand["adt"] = train["adt"].to_numpy()[_cand["row"]]
    q_row = int(_cand.sort_values(["gap", "adt"], ascending=False)["row"].iloc[0])
    _nbr_random = _neighbors(q_row, "random", K)
    _nbr_county = _neighbors(q_row, "county", K)
    attempt_rows = [
        int(_nbr_random[0]),
        int(_nbr_random[1]),
        int(_nbr_county[0]),
        int(_nbr_county[1]),
    ]
    _q = train.loc[q_row]
    _km = np.hypot(train["x_km"] - _q["x_km"], train["y_km"] - _q["y_km"])
    _table = pd.DataFrame(
        {
            "inventory": train["nbi_year"],
            "facility": train["facility"].str.replace(r"\s+", " ", regex=True),
            "county": train["county"].map(COUNTY),
            "built": train["year_built"],
            "design": train["material"] + " " + train["design"].map(DESIGN).fillna("other"),
            "vehicles a day": train["adt"],
            "km from Q": _km.round(1),
            "rating": train["rating"].astype(int).astype(str),
        }
    ).loc[[q_row, *attempt_rows]]
    _table.loc[q_row, "rating"] = "?"
    _table.index = pd.Index(["Q", "A", "B", "C", "D"], name="record")
    # Slide "Your turn: rate bridge Q": Q is IL 53 northbound over Kirchoff Road (Cook County,
    # built 1963, 157,100 vehicles a day), rated 4; A, its twin (4); B, Q in 2016 (4); C and D,
    # the I-290 twins over Addison Road in DuPage County, 15.9 km away (6 and 6).
    _all = [q_row, *attempt_rows]
    assert list(train["structure"].to_numpy()[_all]) == [
        "016037600000000",
        "016112100000000",
        "016037600000000",
        "022009500000000",
        "022010400000000",
    ]
    assert list(train["nbi_year"].to_numpy()[_all]) == [2018, 2018, 2016, 2018, 2018]
    assert [int(_v) for _v in y_train[_all]] == [4, 4, 4, 6, 6]
    assert int(train.at[q_row, "adt"]) == 157100 and int(train.at[q_row, "year_built"]) == 1963
    assert round(1000 * float(_km.loc[attempt_rows[0]]), -1) == 10
    assert [round(float(_km.loc[_r]), 1) for _r in attempt_rows[2:]] == [15.9, 15.9]
    # Slide "The split decides who is near": 1-NN 4 (the twin) vs 6 (DuPage, 043);
    # k = 50: 5.54 vs 5.20.
    assert (int(y_train[_nbr_random[0]]), int(y_train[_nbr_county[0]])) == (4, 6)
    assert train.at[int(_nbr_county[0]), "county"] == "043"
    assert (
        round(float(y_train[_nbr_random].mean()), 2),
        round(float(y_train[_nbr_county].mean()), 2),
    ) == (5.54, 5.2)
    _table
    return attempt_rows, q_row


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer: what the rows say, and the measured rating": mo.md(
                r"""
                - **A** (Q's twin, 10 m away) and **B** (Q itself, two years earlier) both
                  say **4**. The mean of all four rows is 5.0.
                - **Without Cook County's records**, only C and D remain: the I-290 twins over
                  Addison Road in DuPage County, 16 km away. Estimate: **6**.
                - **Measured: 4, poor.** With the county's own records you would flag it;
                  without them, you would call a poor bridge satisfactory.

                Keep both estimates: §2 traces Q through the pipeline, and §5 shows that random
                folds score the model as if every county had shared its records.
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

    > bridge $\mathbf{x}$ → **encoder** (12 inputs, scaled) → $\mathbf{z}$ → **decoder**
    > (average rating of the $k$ nearest training records) → $\hat{y}$ → **held-out error**
    > $\sqrt{\mathrm{mean}\,(\hat{y}-y)^2}$, honest only if the held-out bridges look like the
    > next ones.

    The decoder has no weights to fit: "training" stores the 55,155 inspections of
    2014–2018. Trace Q with $k=1$. Under **random folds**, Q's nearest training record is its
    twin, 10 m away, rated 4: prediction 4, error 0. With **its county held out**, the nearest
    record is in DuPage County, 16 km away, rated 6: prediction 6, error +2, a poor bridge
    called satisfactory. The model is the same in both rows; only what was held out changed.
    With $k=50$, the two predictions are 5.5 and 5.2: averaging 50 records dilutes the twin.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** the held-out cases are exchangeable with the cases the model will face.

    **Why validation works when it does.** Let $L=\mathbb{E}\,\ell(\hat{y}(\mathbf{x}),y)$ be
    the expected loss on the cases the model will face, and $\hat{L}=\frac1n\sum_{i=1}^{n}\ell_i$
    the mean loss on $n$ held-out cases. If each held-out $(\mathbf{x}_i,y_i)$ is drawn like a
    future case and the model never saw it, each $\ell_i$ has expectation $L$, so by linearity
    $$\mathbb{E}\,\hat{L}=L .$$
    If the losses are also independent, $\mathrm{Var}\,\hat{L}=\mathrm{Var}\,\ell/n$, and
    Chebyshev's inequality bounds how far $\hat{L}$ strays (Forsyth §3.1):
    $$P\big(|\hat{L}-L|\ge\epsilon\big)\le\frac{\mathrm{Var}\,\ell}{n\,\epsilon^2}.$$
    Dependence breaks both statements. A held-out case with a near-copy in training (a twin, or
    the same bridge two years earlier) has a smaller loss than a future case, so
    $\mathbb{E}\,\hat{L}<L$; and correlated losses shrink the effective $n$. Choosing $k$ by the
    smallest of several validation errors also biases that error low,
    $\mathbb{E}[\min_k \hat{L}_k]\le\min_k\mathbb{E}[\hat{L}_k]$ (Cawley & Talbot, 2010): hence
    a test set that only scores the choices validation made, and makes none.

    ### Worked example: the five records of your attempt

    Leave one record out and predict it with 1-NN from the other four; then leave its whole
    county out. Distances use the 12 scaled inputs, with the scaler of all training records.
    Compute them by hand from the table the next cell prints, then compare.
    """)
    return


@app.cell
def _(attempt_rows, encode, np, pd, q_row, StandardScaler, train):
    _rows = [q_row, *attempt_rows]
    _names = ["Q", "A", "B", "C", "D"]
    _sc = StandardScaler().fit(encode(train))
    _Z = _sc.transform(encode(train.loc[_rows]))
    _d = np.sqrt(((_Z[:, None, :] - _Z[None, :, :]) ** 2).sum(-1))
    assert _d.shape == (5, 5), _d.shape
    _y = train.loc[_rows, "rating"].to_numpy(float)
    _county = train.loc[_rows, "county"].to_numpy()
    _out = []
    for _i in range(5):
        _by_rec = np.where(np.arange(5) == _i, np.inf, _d[_i])
        _by_cty = np.where(_county == _county[_i], np.inf, _d[_i])
        _j, _m = int(np.argmin(_by_rec)), int(np.argmin(_by_cty))
        _out.append(
            {
                "record": _names[_i],
                "rating": int(_y[_i]),
                "one out: nearest": _names[_j],
                "predicted": int(_y[_j]),
                "county out: nearest": _names[_m],
                "predicted ": int(_y[_m]),
            }
        )
    _loo = pd.DataFrame(_out).set_index("record")
    _err_rec = _loo["predicted"] - _loo["rating"]
    _err_cty = _loo["predicted "] - _loo["rating"]
    _rmse_rec = float(np.sqrt((_err_rec**2).mean()))
    _rmse_cty = float(np.sqrt((_err_cty**2).mean()))
    # Slide "Twins make leave-one-out lie": one record out, RMSE 0.0; county out, RMSE 2.0.
    assert list(_loo["one out: nearest"]) == ["A", "Q", "Q", "D", "C"]
    assert list(_loo["county out: nearest"]) == ["C", "C", "C", "B", "B"]
    assert (round(_rmse_rec, 1), round(_rmse_cty, 1)) == (0.0, 2.0)
    (pd.DataFrame(_d, index=_names, columns=_names).round(2), _loo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Same model, same five records; only what was held out changed.** Leaving one record out,
    every record finds its twin or its own other inspection, and the RMSE is 0.0. Leaving
    its county out, Q, A, and B fall back on the DuPage pair, and C and D on B: RMSE
    2.0. Which number describes a bridge in a county with no records?

    ### Where the error comes from: noise, bias, and variance

    **Setup.** $y=f(\mathbf{x})+\varepsilon$ with $\mathbb{E}\,\varepsilon=0$,
    $\mathrm{Var}\,\varepsilon=\sigma^2$, and $\varepsilon$ independent of the training set
    $\mathcal{D}$. The prediction $\hat f_{\mathcal{D}}(\mathbf{x})$ depends on which training
    set we drew; its average over training sets is $\bar f=\mathbb{E}_{\mathcal{D}}\hat
    f_{\mathcal{D}}$. At a fixed $\mathbf{x}$ (dropped from the notation), add and subtract $f$
    and $\bar f$:

    $$
    \begin{aligned}
    \mathbb{E}\big[(y-\hat f_{\mathcal{D}})^2\big]
    &= \mathbb{E}\big[(\varepsilon + (f-\bar f) + (\bar f-\hat f_{\mathcal{D}}))^2\big] \\
    &= \underbrace{\sigma^2}_{\text{noise}}
     + \underbrace{(f-\bar f)^2}_{\text{bias}^2}
     + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat f_{\mathcal{D}}-\bar f)^2\big]}_{\text{variance}}.
    \end{aligned}
    $$

    The cross terms vanish: $\varepsilon$ has mean 0 and is independent of $\mathcal{D}$;
    $\hat f_{\mathcal{D}}-\bar f$ has mean 0 over $\mathcal{D}$; $f-\bar f$ is a constant. For
    bridges, $\sigma^2$ is the rating noise between inspectors: no model of these inputs
    removes it. (Forsyth §11.1.1 derives the same result and writes the noise as $\xi$.)

    **For $k$-NN**, hold the neighbors fixed; only their noise is random:
    $$\hat f(\mathbf{x})=\frac1k\sum_{i\in\mathcal N_k(\mathbf{x})}\big(f(\mathbf{x}_i)+\varepsilon_i\big)
    \;\Rightarrow\;\mathrm{Var}=\frac{\sigma^2}{k},\qquad
    \mathrm{Bias}=\frac1k\sum_{i\in\mathcal N_k(\mathbf{x})}f(\mathbf{x}_i)-f(\mathbf{x}).$$

    **Measuring it on bridges.** We do not know $f$, but we can draw training sets: keep a
    random half of the training bridges (all their records), 20 times, and predict the first
    block of held-out counties. For each held-out record, the mean squared error over the draws
    splits *exactly* into the variance of the 20 predictions and the squared distance of their
    mean to the rating, which is bias² + noise. The cell checks that identity.
    """)
    return


@app.cell
def _(
    LINESTYLES, MARKERS, PALETTE, SEED, encode, holdout_predict, make_folds, np, plt, train, y_train
):
    _ks = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
    _X = encode(train)
    _st = train["structure"].to_numpy()
    _tr, _te = make_folds(train, "county")[0]
    _bridges = np.unique(_st[_tr])
    _rng = np.random.default_rng(SEED)
    _preds = np.empty((20, len(_te), len(_ks)))
    for _b in range(20):
        _keep = _rng.choice(_bridges, size=len(_bridges) // 2, replace=False)
        _sub = _tr[np.isin(_st[_tr], _keep)]
        _preds[_b], _, _ = holdout_predict(_X[_sub], y_train[_sub], _X[_te], ks=_ks)
    assert _preds.shape == (20, len(_te), len(_ks)), _preds.shape
    _var = _preds.var(axis=0).mean(axis=0)
    _b2n = ((_preds.mean(axis=0) - y_train[_te][:, None]) ** 2).mean(axis=0)
    _mse = ((_preds - y_train[_te][None, :, None]) ** 2).mean(axis=(0, 1))
    assert np.allclose(_mse, _var + _b2n, rtol=1e-10, atol=1e-12)  # the identity, checked
    _train_pred, _, _ = holdout_predict(_X, y_train, _X, ks=_ks)
    _train_mse = ((_train_pred - y_train[:, None]) ** 2).mean(axis=0)
    _at = {int(_k): _i for _i, _k in enumerate(_ks)}
    # Slide "k trades variance for bias": 18 counties, 11,052 records, 20 draws; variance 0.67
    # (k = 1) and 0.12 (k = 10); held-out error lowest at k = 100 (1.13), 1.24 at k = 1000;
    # bias² + noise lowest at k = 50; training error 0.001 points² at k = 1 (all squared).
    assert len(_te) == 11052
    _v1, _v10 = round(float(_var[_at[1]]), 2), round(float(_var[_at[10]]), 2)
    assert (_v1, _v10) == (0.67, 0.12)
    assert int(_ks[int(np.argmin(_mse))]) == 100 and round(float(_mse.min()), 2) == 1.13
    assert round(float(_mse[_at[1000]]), 2) == 1.24
    assert int(_ks[int(np.argmin(_b2n))]) == 50
    assert round(float(_train_mse[_at[1]]), 3) == 0.001
    assert round(_v1 / _v10, 1) == 5.6  # "a factor of about 5.6 (0.67 / 0.12)"

    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _c, (_vals, _label) in enumerate(
        [
            (_mse, "held-out error"),
            (_b2n, "bias² + noise"),
            (_var, "variance"),
            (_train_mse, "training error"),
        ]
    ):
        _ax.plot(_ks, _vals, color=PALETTE[_c], ls=LINESTYLES[_c], marker=MARKERS[_c], label=_label)
    _ax.set(
        xscale="log",
        xlabel="k (number of neighbors)",
        ylabel="squared error (points²)",
        ylim=(0, 2.2),
    )
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Read the curves** (every error here is squared, in points²). Variance falls from 0.67 at
    $k=1$ to 0.12 at $k=10$, a factor of about 5.6 rather than the 10 of $\sigma^2/k$: the
    neighbors themselves change from one training set to the next. (The training sets are
    random halves of one pool of bridges, so this is the spread over such halves, not over
    independent samples from the population.) Bias² + noise is lowest near $k=50$ and rises beyond it, as the neighbors
    spread to bridges unlike the one predicted. At small $k$ it is also high, because the
    average prediction still carries the few neighbors' own rating noise. The held-out error is
    lowest near $k=100$ (1.13 points²) and reaches 1.24 by $k=1000$. The training error says
    nothing: 0.001 points² at $k=1$, since each record is its own nearest neighbor.

    ## 4. Build and verify

    **Cross-validation is a loop.** For each fold: fit the scaler on the training rows only
    (it is part of the model), search the neighbors once, and get every $k$ at once from
    running sums, since the mean of the nearest $k$ is the sum of the nearest $k$ divided by
    $k$. The folds: `KFold(5, shuffle=True)` for random records; `GroupKFold(5)` by bridge or by
    county.
    """)
    return


@app.cell
def _(NearestNeighbors, StandardScaler, np):
    def cv_predict(
        X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], ks: np.ndarray
    ) -> np.ndarray:
        """Out-of-fold k-NN predictions for every k in ks (the slide's listing).

        Args:
            X: Inputs, shape ``(N, D)``.
            y: Ratings, shape ``(N,)``.
            folds: ``(train, held_out)`` index pairs; every row is held out exactly once.
            ks: Numbers of neighbors, a NumPy integer array.

        Returns:
            Predictions, shape ``(N, len(ks))``.
        """
        pred = np.full((len(y), len(ks)), np.nan)
        for tr, te in folds:  # te: one held-out block
            sc = StandardScaler().fit(X[tr])  # training rows only
            Z_tr, Z_te = sc.transform(X[tr]), sc.transform(X[te])
            nn = NearestNeighbors(n_neighbors=ks.max(), algorithm="brute")
            _, idx = nn.fit(Z_tr).kneighbors(Z_te)  # nearest first
            total = np.cumsum(y[tr][idx], axis=1)  # sums of nearest 1..K
            pred[te] = total[:, ks - 1] / ks
        assert not np.isnan(pred).any()  # each row held out once
        return pred

    return (cv_predict,)


@app.cell
def _(mo):
    mo.md(r"""
    **Predict.** Counties held out, $k = 50$. Will `cv_predict` match scikit-learn's
    `cross_val_predict` (a pipeline of `StandardScaler` and `KNeighborsRegressor`, same folds)
    on every one of the 55,155 training records? On almost every record? Or will they often
    differ? Write your answer down, then **run** the next cell. It compares the two to
    $10^{-9}$ at $k = 1, 5, 50$ and counts the records whose $k$-th and $(k+1)$-th nearest
    distances tie.
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    KS,
    NearestNeighbors,
    StandardScaler,
    cross_val_predict,
    cv_predict,
    encode,
    k_index,
    make_folds,
    make_pipeline,
    np,
    pd,
    train,
    y_train,
):
    _X = encode(train)
    _folds = make_folds(train, "county")
    oof_county = cv_predict(_X, y_train, _folds, KS)  # reused by the bracket and §5
    _rows = []
    for _k in (1, 5, 50):
        _ref = cross_val_predict(
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=_k, algorithm="brute")),
            _X,
            y_train,
            cv=_folds,
        )
        _agree = np.isclose(oof_county[:, k_index(_k)], _ref, rtol=0, atol=1e-9)
        _tie = np.zeros(len(y_train), dtype=bool)
        for _tr, _te in _folds:
            _sc = StandardScaler().fit(_X[_tr])
            _dist, _ = (
                NearestNeighbors(n_neighbors=_k + 1, algorithm="brute")
                .fit(_sc.transform(_X[_tr]))
                .kneighbors(_sc.transform(_X[_te]))
            )
            _tie[_te] = np.isclose(_dist[:, _k - 1], _dist[:, _k], rtol=0, atol=1e-9)
        assert _agree[~_tie].all(), f"k={_k}: disagreement away from ties"
        _rows.append(
            {
                "k": _k,
                "records": len(y_train),
                "with a tie": int(_tie.sum()),
                "differ": int((~_agree).sum()),
                "largest difference": float(np.abs(oof_county[:, k_index(_k)] - _ref).max()),
            }
        )
    _check = pd.DataFrame(_rows).set_index("k")
    # Slide "Between the mean and the twins": the two codes differ only at ties (the loop
    # above stops on any other difference); 865 records tie for the 50th neighbor, 2,092 for
    # the first. How many tied records the codes resolve differently depends on the last bit
    # of the distances, so on the machine and its BLAS threads: at k = 50, 48 (by at most 0.06)
    # on the lecture's machine with MKL's 28 threads, 39 (0.08) with 1 or 4 threads. Not
    # asserted for that reason.
    assert (_check.at[50, "with a tie"], _check.at[1, "with a tie"]) == (865, 2092)
    assert (_check["differ"] <= _check["with a tie"]).all()
    _check
    return (oof_county,)


@app.cell
def _(mo):
    mo.md(r"""
    **Result.** At $k=50$, every record without a tie matches to $10^{-9}$. 865 records have a
    tie for the 50th neighbor: the 50th and 51st nearest training records are equally far away,
    and which one counts is arbitrary in both codes. For some of them the two codes pick
    differently: 48 on the lecture's machine, each moving the mean by at most 0.06. Which tied
    record wins depends on the last bit of the computed distances, so your count may differ
    (with one or four BLAS threads it is 39); the check that matters is that every difference
    sits at a tie. At $k=1$, a tie can move a prediction by whole rating points. "Matches
    except at ties" is a check only once you have counted the ties.

    **Bracket the error.** An error means little on its own. Above it: predicting the
    training mean. Below it: the noise of the rating itself. Twin bridges, side by side and
    built the same year, share age, design, traffic, and weather. If their true conditions were
    equal and their rating errors independent, with standard deviation $\sigma$, the difference
    of their ratings would have variance $2\sigma^2$, and $\mathrm{RMS}/\sqrt2$ would estimate
    $\sigma$. But twins are mostly rated on the same visit, likely by the same crew, so their
    errors are positively correlated ($\rho>0$): the difference has variance
    $2\sigma^2(1-\rho)$, and $\mathrm{RMS}/\sqrt2$ estimates $\sigma\sqrt{1-\rho}$, the
    disagreement within one visit. Real differences between twins push it up, correlated errors
    push it down; the cell checks which wins by comparing pairs rated in the same month with
    the others.
    """)
    return


@app.cell
def _(K, df, find_twins, k_index, make_folds, np, oof_county, train, y_train):
    _folds = make_folds(train, "county")
    _cty = np.sqrt(((oof_county - y_train[:, None]) ** 2).mean(axis=0))
    _mean_pred = np.empty_like(y_train)
    for _tr, _te in _folds:
        _mean_pred[_te] = y_train[_tr].mean()
    _base = float(np.sqrt(np.mean((_mean_pred - y_train) ** 2)))
    # Twins of the whole 2018 inventory, every bridge's record that year: the floor is a
    # property of the ratings, not of the training set.
    _s18 = df[df["nbi_year"] == 2018]
    _tw = find_twins(_s18)
    _mutual = _tw.merge(_tw, left_on=["row", "twin_row"], right_on=["twin_row", "row"])
    _pairs = _mutual[_mutual["row_x"] < _mutual["twin_row_x"]]
    _a, _b = _s18.loc[_pairs["row_x"]], _s18.loc[_pairs["twin_row_x"]]
    _diff = _a["rating"].to_numpy() - _b["rating"].to_numpy()
    _same_month = (_a["inspection_year"].to_numpy() == _b["inspection_year"].to_numpy()) & (
        _a["inspection_month"].to_numpy() == _b["inspection_month"].to_numpy()
    )
    _rms = float(np.sqrt(np.mean(_diff**2)))
    _rms_other = float(np.sqrt(np.mean(_diff[~_same_month] ** 2)))
    _bracket = {
        "predict the mean": _base,
        f"k-NN, k = {K}": float(_cty[k_index(K)]),
        "k-NN, k = 1": float(_cty[k_index(1)]),
        "twins, one visit": _rms / np.sqrt(2),
    }
    # Slide "Between the mean and the twins": the mean 1.44; k = 50, 1.09; k = 1, 1.45; twins:
    # 872 pairs, RMS difference 0.58, floor 0.41, 77% equal, 97% within one point; 772 pairs
    # (89%) inspected in the same month; the other 100 pairs give 0.66.
    assert {_n: round(_v, 2) for _n, _v in _bracket.items()} == {
        "predict the mean": 1.44,
        f"k-NN, k = {K}": 1.09,
        "k-NN, k = 1": 1.45,
        "twins, one visit": 0.41,
    }
    assert (len(_pairs), round(_rms, 2)) == (872, 0.58)
    assert (
        round(100 * float((_diff == 0).mean())),
        round(100 * float((np.abs(_diff) <= 1).mean())),
    ) == (77, 97)
    assert (int(_same_month.sum()), round(100 * float(_same_month.mean()))) == (772, 89)
    assert (int((~_same_month).sum()), round(_rms_other / np.sqrt(2), 2)) == (100, 0.66)
    (
        {_n: round(_v, 2) for _n, _v in _bracket.items()},
        {
            "pairs inspected in the same month": int(_same_month.sum()),
            "floor of the other pairs": round(_rms_other / np.sqrt(2), 2),
        },
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Counties held out, $k=50$: 1.09 rating points**, between predicting the
    mean (1.44) above and the twins (0.41) below; $k=1$ (1.45) is no better than the mean.
    Twins agree exactly 77% of the time and within one point 97% of the time. But 772 of the
    872 pairs (89%) were inspected in the same month, and the other 100 pairs disagree more
    (0.66): correlated errors win, so 0.41 is the disagreement within one visit, a lower bound
    on the noise between inspectors, not an upper one. A national study found more spread
    between inspectors from different states: 68% of ratings within one point of the average
    and 95% within two (Moore et al., 2001; 49 inspectors from 25 states), about what a normal
    spread with a standard deviation of one point gives: $\sigma\approx1$ between inspectors.

    ## 5. Break it

    Each test below probes the assumption from §3, *the held-out cases are exchangeable with
    the cases the model will face*.

    **Predict before we test.** Write down three answers.

    1. **Random folds.** Hold out random records instead of whole counties. Higher or lower
       error? Which $k$ will it pick?
    2. **The future.** Train on 2014–2018, then test on the 2024 inventory. Nearer the
       random-fold number or the county number?
    3. **Leaky columns.** Add the inspection interval and the structural evaluation from the
       same record. What happens to the error? Would they help in 2024?

    Commit in writing; each subsection reveals one answer.

    ### Random folds, bridges held out, counties held out
    """)
    return


@app.cell
def _(
    KS,
    MARKERS,
    PALETTE,
    cv_predict,
    encode,
    find_twins,
    make_folds,
    np,
    oof_county,
    pd,
    plt,
    rmse_per_k,
    train,
    y_train,
):
    _X = encode(train)
    oof = {_s: cv_predict(_X, y_train, make_folds(train, _s), KS) for _s in ("random", "bridge")}
    oof["county"] = oof_county  # computed once, in §4's check
    cv_curves = pd.DataFrame(
        {_s: rmse_per_k(_p, y_train) for _s, _p in oof.items()}, index=pd.Index(KS, name="k")
    )
    _best = cv_curves.idxmin()
    # How often a held-out record has its own bridge, or its twin, in training (random folds).
    _st = train["structure"].to_numpy()
    _tw = find_twins(train[train["nbi_year"] == 2018])
    _twin_of = dict(zip(_st[_tw["row"]], _st[_tw["twin_row"]], strict=True))
    _own = _twin_in = _n_twin = 0
    for _tr, _te in make_folds(train, "random"):
        _in = set(_st[_tr])
        _own += sum(_s in _in for _s in _st[_te])
        _held = [_twin_of[_s] for _s in _st[_te] if _s in _twin_of]
        _n_twin += len(_held)
        _twin_in += sum(_t in _in for _t in _held)
    _own_pct, _twin_pct = 100 * _own / len(train), 100 * _twin_in / _n_twin
    # Slide "Random folds reuse the same bridges": random folds 0.90 at k = 2; bridges held out
    # 1.03 at k = 30; counties held out 1.09 at k = 70; 88% own bridge, 99% twin in training.
    assert {_s: int(_k) for _s, _k in _best.items()} == {"random": 2, "bridge": 30, "county": 70}
    assert [round(float(cv_curves.at[int(_k), _s]), 2) for _s, _k in _best.items()] == [
        0.90,
        1.03,
        1.09,
    ]
    assert (round(_own_pct), round(_twin_pct)) == (88, 99)

    _fig, _axes = plt.subplots(1, 2, figsize=(6, 4))
    _s18 = train["nbi_year"].to_numpy() == 2018
    for _ax, _s, _title in zip(
        _axes, ("random", "county"), ("random folds", "counties held out"), strict=True
    ):
        _h = np.zeros(len(train), dtype=bool)
        _h[make_folds(train, _s)[0][1]] = True
        _ax.scatter(
            train["x_km"][_s18 & ~_h],
            train["y_km"][_s18 & ~_h],
            s=0.5,
            color="#C8C8C8",
            marker="o",
            linewidths=0,
        )
        _ax.scatter(
            train["x_km"][_s18 & _h],
            train["y_km"][_s18 & _h],
            s=1.5,
            color=PALETTE[0],
            marker=MARKERS[1],
            linewidths=0,
        )
        _ax.set_aspect("equal")
        _ax.set_axis_off()
        _ax.set_title(_title)
    _fig.tight_layout()
    (cv_curves.round(3).T, _fig)
    return cv_curves, oof


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 1: random folds reuse the same bridges.** Random folds: 0.90 at $k=2$. Bridges held
    out: 1.03 at $k=30$. Counties held out: 1.09 at $k=70$. With random folds, 88% of held-out
    records have another inspection of the same bridge in training, and 99% of held-out twins
    have their twin there. Holding out whole bridges removes the first leak; holding out
    counties also removes the twins and most neighbors on the same corridor, inspected by the
    same crews, though a neighbor across a county border stays in training (Roberts et al.,
    2017, on blocked cross-validation).

    ### The future: train on 2014–2018, test on later inventories

    Fit on all training records, then predict the 2020, 2022, and 2024 inventories, keeping
    only inspections not already in training. The 2024 inventory is the test set: it scores
    the $k$ that each validation chose, and nothing is chosen from it. The cell also tests
    counties new to the model: train on the other counties' 2014–2018 records, test on the
    held-out counties' 2024 records.
    """)
    return


@app.cell
def _(
    KS,
    LATER_YEARS,
    LINESTYLES,
    MARKERS,
    PALETTE,
    TEST_YEAR,
    county_fold_index,
    county_fold_of,
    cv_curves,
    df,
    encode,
    holdout_predict,
    later_records,
    make_folds,
    np,
    pd,
    plt,
    rmse_per_k,
    train,
    y_train,
):
    _X = encode(train)
    _st = train["structure"].to_numpy()
    later_pred, later_y = {}, {}
    _rows = {}
    for _year in LATER_YEARS:
        _test = later_records(df, train, _year)
        _pred, _idx, _ = holdout_predict(_X, y_train, encode(_test))
        later_pred[_year], later_y[_year] = _pred, _test["rating"].to_numpy(float)
        _rows[f"test {_year}"] = rmse_per_k(_pred, later_y[_year])
        _rows[f"bias {_year}"] = (_pred - later_y[_year][:, None]).mean(axis=0)
        _rows[f"own 1-NN {_year} (%)"] = np.full(
            len(KS), 100 * (_st[_idx[:, 0]] == _test["structure"].to_numpy()).mean()
        )
    _test = later_records(df, train, TEST_YEAR)
    # A 2024 record belongs to the fold of its bridge's first county, as its training records do.
    _test_fold = county_fold_index(_test, county_fold_of(train))
    _sq = []
    for _j, (_tr, _) in enumerate(make_folds(train, "county")):
        _held = _test[_test_fold == _j]
        _p, _, _ = holdout_predict(_X[_tr], y_train[_tr], encode(_held))
        _sq.append((_p - _held["rating"].to_numpy(float)[:, None]) ** 2)
    _sq_all = np.concatenate(_sq)
    assert _sq_all.shape == (len(_test), len(KS)), _sq_all.shape  # every 2024 record once
    _rows["test 2024, new counties"] = np.sqrt(_sq_all.mean(axis=0))
    later_table = pd.DataFrame(_rows, index=pd.Index(KS, name="k"))
    # The test scores only the k that each validation chose; nothing is chosen on it.
    _chosen = {_s: int(cv_curves[_s].idxmin()) for _s in ("random", "bridge", "county")}
    _t24 = later_table[f"test {TEST_YEAR}"]
    _at_chosen = {_s: round(float(_t24.at[_k]), 2) for _s, _k in _chosen.items()}
    # Slide "Random folds pick the wrong k": random folds pick k = 2 (0.90), which scores 1.12
    # on 2024; bridges held out pick 30 and counties 70, both 1.01 on 2024; new counties in
    # 2024: 1.06 at the counties' k = 70.
    assert _chosen == {"random": 2, "bridge": 30, "county": 70}
    assert _at_chosen == {"random": 1.12, "bridge": 1.01, "county": 1.01}
    _new = later_table["test 2024, new counties"]
    assert round(float(_new.at[_chosen["county"]]), 2) == 1.06
    # Slide "Old ratings go stale": 1-NN errs 0.93, 1.07, 1.20 in 2020, 2022, 2024; bias +0.20
    # (k = 1) and +0.11 (k = 50) in 2024; the 1-NN is the bridge's own inspection 70, 56, 42%;
    # 2020 at k = 2: 0.91.
    assert [round(float(later_table.at[1, f"test {_y}"]), 2) for _y in LATER_YEARS] == [
        0.93,
        1.07,
        1.20,
    ]
    assert (
        round(float(later_table.at[1, "bias 2024"]), 2),
        round(float(later_table.at[50, "bias 2024"]), 2),
    ) == (0.20, 0.11)
    assert [round(float(later_table.at[1, f"own 1-NN {_y} (%)"])) for _y in LATER_YEARS] == [
        70,
        56,
        42,
    ]
    assert round(float(later_table.at[2, "test 2020"]), 2) == 0.91

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _series = [
        (cv_curves["random"], "random folds"),
        (cv_curves["bridge"], "bridges held out"),
        (cv_curves["county"], "counties held out"),
        (later_table[f"test {TEST_YEAR}"], f"{TEST_YEAR} inventory (post hoc)"),
    ]
    for _c, (_vals, _label) in enumerate(_series):
        _ax.plot(
            KS,
            _vals,
            color=PALETTE[_c],
            ls=LINESTYLES[_c],
            marker=MARKERS[_c],
            label=_label,
            lw=3 if _c == 3 else 2,
        )
    _ax.set(
        xscale="log",
        xlabel="k (number of neighbors)",
        ylabel="RMSE (rating points)",
        ylim=(0.75, 1.5),
    )
    _ax.legend()
    _fig.tight_layout()
    (later_table.round(3).T, _fig)
    return later_pred, later_y


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 2: random folds pick the wrong $k$.** Random folds pick $k=2$ and promise 0.90. On the
    2024 inventory, that model scores 1.12. The honest folds pick $k=30$ (bridges held out)
    and $k=70$ (counties held out), and the 2024 inventory scores both 1.01. Counties new to the
    model in 2024 score 1.06 at $k=70$, close to what counties held out promised (1.09). Random
    folds are not only optimistic; they pick the wrong model: their curve rises with $k$ while
    every honest curve falls. This is the lecture's decisive figure. The full 2024 curve is a
    post-hoc diagnostic, drawn after the choices: reading $k$ off it would turn the test set
    into a validation set.

    ### Why the future is harder: old ratings go stale
    """)
    return


@app.cell
def _(LINESTYLES, MARKERS, PALETTE, df, np, pd, plt):
    _bins = [0, 10, 20, 30, 40, 50, 60, 80, 200]
    _labels = ["0–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60–79", "80+"]
    _by_age = {}
    for _year in (2014, 2024):
        _rec = df[df["nbi_year"] == _year]
        _grp = pd.cut(_rec["age"], _bins, right=False, labels=_labels)
        _by_age[f"{_year} inventory"] = _rec.groupby(_grp, observed=False)["rating"].mean()
    _by_age = pd.DataFrame(_by_age)
    # Slide "Old ratings go stale": bridges under 10 averaged 7.96 in 2014 and 7.58 in 2024;
    # at 30 to 39 years, 6.37 and 6.35.
    assert [round(float(_v), 2) for _v in _by_age.iloc[0]] == [7.96, 7.58]
    assert [round(float(_v), 2) for _v in _by_age.iloc[3]] == [6.37, 6.35]

    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _c, _col in enumerate(_by_age.columns):
        _ax.plot(
            np.arange(len(_labels)),
            _by_age[_col],
            color=PALETTE[_c],
            ls=LINESTYLES[_c],
            marker=MARKERS[_c],
            label=_col,
        )
    _ax.set_xticks(np.arange(len(_labels)), _labels, rotation=30)
    _ax.set(xlabel="age at inspection (years)", ylabel="mean condition rating")
    _ax.legend()
    _fig.tight_layout()
    (_by_age.round(2).T, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 2, continued.** With 1-NN, the nearest record of a 2020 bridge is its own
    earlier inspection 70% of the time; by 2024, 42%. The copy goes stale: 1-NN errs by 0.93
    two years after the 2018 inventory (2020), 1.07 at four (2022), and 1.20 at six (2024).
    (A record's own last inspection in training can be older: intervals reach 48 months.) These scores
    are a post-hoc diagnostic: no validation chose $k = 1$, so they describe how a copied
    inspection ages, not a model to use. The model is
    optimistic about condition: it overrates 2024 bridges by 0.20 at $k=1$ and 0.11 at $k=50$.
    Part of the reason shows above: at the same age, bridges under 30 rated lower in 2024 than
    in 2014 (7.58 against 7.96 under 10). Why is a question for the inventory's owners: rating
    practice, the bridges built, or both. Validate forward in time, at the horizon of the use:
    the 2020 inventory, two years out, scores 0.91 at $k=2$, about what random folds promised
    (0.90), and says nothing about six years out.

    ### Leaky columns

    Two columns sit in the same record as the rating. The **inspection interval** (Item 91) is
    12, 24, or 48 months; the Coding Guide says it "could vary from inspection to inspection
    depending on the condition of the bridge at the time of inspection" (FHWA, 1995). The
    **structural evaluation** (Item 67) is computed from the lowest of the superstructure
    rating, the substructure rating, and a load-based code. Add them as inputs.
    """)
    return


@app.cell
def _(K, LEAKY, cv_predict, encode, make_folds, np, pd, rmse_per_k, train):
    _rec = train.dropna(subset=LEAKY).reset_index(drop=True)  # both columns present
    _y = _rec["rating"].to_numpy(float)
    _options = {
        "inventory only": (),
        "+ inspection interval": ("inspection_interval",),
        "+ structural evaluation": ("structural_eval",),
        "+ both": ("inspection_interval", "structural_eval"),
    }
    _rows = {}
    for _name, _extra in _options.items():
        _X = encode(_rec, _extra)
        _r = {}
        for _s in ("county", "random"):
            _p = cv_predict(_X, _y, make_folds(_rec, _s), np.array([1, K]))
            _e = rmse_per_k(_p, _y)
            _r[f"{_s}, k = 1"], _r[f"{_s}, k = {K}"] = _e[0], _e[1]
        _rows[_name] = _r
    _leak = pd.DataFrame(_rows).T
    _by_interval = _rec.groupby("inspection_interval")["rating"].agg(["mean", "size"])
    _by_interval = _by_interval[_by_interval["size"] >= 100]
    _computed = (
        _rec["structural_eval"] <= _rec[["superstructure", "substructure"]].min(axis=1)
    ).mean()
    # Slide "Some columns already know the answer": 55,045 records with both columns; counties
    # held out, k = 50: 1.10, 0.94, 0.61, 0.61; random folds with both: 0.57; mean rating by
    # interval 3.8, 5.7, 7.6; every structural evaluation is at or below min(Items 59, 60).
    assert len(_rec) == 55045
    assert [round(float(_v), 2) for _v in _leak[f"county, k = {K}"]] == [1.10, 0.94, 0.61, 0.61]
    assert round(float(_leak.at["+ both", f"random, k = {K}"]), 2) == 0.57
    assert [round(float(_v), 1) for _v in _by_interval["mean"]] == [3.8, 5.7, 7.6]
    assert _computed == 1.0
    (_leak.round(2), _by_interval.round(2))
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 3: some columns already know the answer.** Counties held out, $k=50$: 1.10 with the
    inventory alone, 0.94 with the interval, 0.61 with the structural evaluation (55,045 records
    have both columns; 110 lack one). The interval's mean rating, 3.8 at 12 months, 5.7 at 24,
    and 7.6 at 48, shows how it was set, and every structural evaluation is at or below the
    lower of the superstructure and substructure ratings. Neither column exists before the
    inspection that sets the rating: this is **leakage** (Kaufman et al., 2012). Ask of every
    column: would I have it when I make the prediction?

    ## 6. Decide

    **Screening.** Flag the 10% of records predicted worst for inspection first (ties broken
    at random, with a fixed seed), and count the poor records caught. Compare what random
    folds promise for the model they pick, on the training records (several inspections per
    bridge), with what the 2024 inventory delivers (one record per bridge).
    """)
    return


@app.cell
def _(
    K,
    LINESTYLES,
    MARKERS,
    PALETTE,
    POOR,
    SEED,
    TEST_YEAR,
    cv_curves,
    k_index,
    later_pred,
    later_y,
    np,
    oof,
    plt,
    y_train,
):
    _rng = np.random.default_rng(SEED)
    _kr = int(cv_curves["random"].idxmin())

    def _gains(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _order = np.lexsort((_rng.random(len(pred)), pred))  # lowest prediction first
        _caught = np.cumsum(y[_order] <= POOR) / max(int((y <= POOR).sum()), 1)
        return np.arange(1, len(y) + 1) / len(y), _caught

    _cases = [
        (f"promised by random folds, k = {_kr}", oof["random"][:, k_index(_kr)], y_train),
        (f"{TEST_YEAR}, k = {_kr}", later_pred[TEST_YEAR][:, k_index(_kr)], later_y[TEST_YEAR]),
        (f"{TEST_YEAR}, k = {K}", later_pred[TEST_YEAR][:, k_index(K)], later_y[TEST_YEAR]),
    ]
    _caught_at = {}
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _c, (_label, _pred, _y) in enumerate(_cases):
        _f, _g = _gains(_pred, _y)
        _at = int(round(0.10 * len(_y))) - 1
        _caught_at[_label] = round(100 * float(_g[_at]))
        _ax.plot(100 * _f, 100 * _g, color=PALETTE[_c], ls=LINESTYLES[_c], label=_label)
        _ax.plot(100 * _f[_at], 100 * _g[_at], MARKERS[_c], color=PALETTE[_c])
    _ax.axvline(10, color="#595959", ls=":")
    _ax.set(
        xlim=(0, 40),
        ylim=(0, 100),
        xlabel="records flagged for inspection (%)",
        ylabel="poor records caught (%)",
    )
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    # Slide "Optimism misses poor bridges": flag 10%; promised 60%; delivered 43% (k = 2) and
    # 37% (k = 50); 2,307 poor bridges in the 2024 test.
    assert list(_caught_at.values()) == [60, 43, 37]
    assert int((later_y[TEST_YEAR] <= POOR).sum()) == 2307
    (_caught_at, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Random folds promise that the $k=2$ model they pick catches 60% of the poor training records
    when 10% of records are flagged. In 2024, one record per bridge, it caught 43% of the poor
    bridges; the $k=50$ model, 37%. Two
    lessons. Optimistic validation costs missed poor bridges, the expensive error. And the $k$
    with the lowest RMSE is not the best screen: copying a bridge's own last rating ranks poor
    bridges better than averaging 50 neighbors, so choose the validation metric by the decision
    (Lecture 6).

    ### When not to use this model: rank bridges, and inspect them anyway

    - **Uncertainty.** An honest error of about one rating point means a predicted 6 can be a
      poor 4. The model ranks; it does not rate.
    - **Not instead of inspection.** The inventory rests on routine inspections, on a 24-month
      standard (FHWA, 1995), and an inspector, not a model, assigns the rating.
    - **Not beyond the data:** new materials or designs, other states, or a later decade,
      without validating again.
    - **Not with leaky columns**, and not with an error from random folds.

    Uncertainty with a guarantee comes later: calibration (Lecture 10), conformal prediction
    (Lecture 11).

    ### Impact: whose bridges are in the data?

    The NBI is a complete inventory: every public highway bridge, inspected on a schedule.
    Crowd-reported data are not: Boston's Street Bump app found potholes through drivers'
    smartphones, so its map followed who owned one (Crawford, 2013). Measurement bias becomes
    allocation bias. But a complete inventory is still not served equally, as the next cell
    shows: the error, with counties held out, differs from county to county.
    """)
    return


@app.cell
def _(COUNTY, K, SEED, k_index, np, oof, pd, train, y_train):
    _err = (oof["county"][:, k_index(K)] - y_train) ** 2
    _frame = pd.DataFrame(
        {
            "county": train["county_fold"].to_numpy(),
            "bridge": train["structure"].to_numpy(),
            "sq": _err,
        }
    )
    _per = _frame.groupby("county")["sq"]
    _stats = pd.DataFrame({"rmse": np.sqrt(_per.mean()), "records": _per.size()})
    _stats.index = _stats.index.map(COUNTY)
    _big = _stats[_stats["records"] >= 300].sort_values("rmse")
    _sd = float(_big["rmse"].std(ddof=1))

    # Noise floor: deal the bridges to counties at random (each county keeps its number of
    # bridges, each bridge its records and errors), 500 times, as the figure script does.
    _bridges = _frame.groupby("bridge", sort=True).agg(
        county=("county", "first"), sq=("sq", "sum"), n=("sq", "size")
    )
    _codes, _idx = np.unique(_bridges["county"].to_numpy(), return_inverse=True)
    _dealt = np.random.default_rng((SEED, 6)).permuted(np.tile(_idx, (500, 1)), axis=1)
    _null = {"min": [], "max": [], "sd": []}
    for _g in _dealt:
        _n = np.bincount(_g, weights=_bridges["n"].to_numpy(float), minlength=len(_codes))
        _r = np.sqrt(np.bincount(_g, weights=_bridges["sq"].to_numpy(), minlength=len(_codes)) / _n)
        _r = _r[_n >= 300]
        _null["min"].append(_r.min())
        _null["max"].append(_r.max())
        _null["sd"].append(_r.std(ddof=1))
    _null = {_key: np.array(_v) for _key, _v in _null.items()}
    _floor = {
        "observed sd": round(_sd, 2),
        "chance sd (median)": round(float(np.median(_null["sd"])), 2),
        "chance best (median)": round(float(np.median(_null["min"])), 2),
        "chance worst (median)": round(float(np.median(_null["max"])), 2),
    }
    # Slide "Whose bridges are in the data?": 79 counties with 300+ records; Franklin 0.76,
    # White 1.39; Cook 1.06. Notes: the spread is 0.13 against chance's 0.06; chance alone gives
    # a best and worst of about 0.95 and 1.25; Franklin and White lie beyond all 500 deals.
    assert _floor == {
        "observed sd": 0.13,
        "chance sd (median)": 0.06,
        "chance best (median)": 0.95,
        "chance worst (median)": 1.25,
    }
    assert (_null["min"] > _big["rmse"].iloc[0]).all() and (
        _null["max"] < _big["rmse"].iloc[-1]
    ).all()
    assert len(_big) == 79
    assert (_big.index[0], round(float(_big["rmse"].iloc[0]), 2)) == ("Franklin", 0.76)
    assert (_big.index[-1], round(float(_big["rmse"].iloc[-1]), 2)) == ("White", 1.39)
    assert round(float(_stats.at["Cook", "rmse"]), 2) == 1.06
    (pd.concat([_big.head(5), _big.tail(5)]).round(2), _floor)
    return


@app.cell
def _(mo):
    mo.md(r"""
    With counties held out and $k=50$, the error is 0.76 in Franklin County
    and 1.39 in White County (79 counties with at least 300 records; Cook County, 1.06). Is the
    spread real? Dealing the bridges to counties at random, 500 times, spreads the counties by
    0.06 around the state's 1.09, with a best and worst of about 0.95 and 1.25: chance alone.
    The real spread is twice that, 0.13, and Franklin and White lie beyond every random deal:
    the unequal error is real. But part of each extreme is chance, as in Lecture 1's winner's
    curse: chance alone moves the best and worst of 79 counties that far from 1.09. A ranking
    that is less accurate in some counties moves inspections and money between them. Report
    the error per county, with its noise floor, not only the state average.

    ### Current systems: foundation models face the same test

    - **Language models** train on web text that can include benchmark test sets.
      "Contamination causes an overestimation of the performance" (Sainz et al., 2023). The fix
      is today's: test on data newer than the training data, and check for overlap.
    - **Mapping from satellite and survey data.** For forest biomass in central Africa (11.8
      million trees), random validation said a model "predicts more than half" of the
      variation; spatial validation revealed "quasi-null predictive power" (Ploton et al., 2020).
    - **Geospatial foundation models** pretrain on over 1 TB of satellite imagery (Prithvi;
      Jakubik et al., 2023). Before trusting a benchmark score, ask which places and dates the
      test tiles share with pretraining.

    Lecture 18 mentions geospatial foundation models (Prithvi); Lecture 19 shows how to evaluate
    language models.

    ### Vote: which error do you quote?

    A county engineer will rank her bridges with your model. Her county has never shared its
    records. Which error do you quote?

    - **A.** 0.90: random folds ($k=2$)
    - **B.** 1.03: bridges held out ($k=30$)
    - **C.** 1.09: counties held out ($k=70$)
    - **D.** 1.01: the 2024 inventory, at B's $k=30$

    Vote alone. Discuss with a neighbor for two minutes. Vote again. Then open the explanation.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Explanation": mo.md(
                r"""
                **C** matches her use: her county is new to the model. **A**
                reuses her neighbors' bridges and her own bridges' other inspections. **B** and
                **D** still saw her county's other bridges, twins included; **D**, the 2024
                score of the model B chose (not a $k$ picked on the test set), is the right test
                for a different use, the next inventory of the same bridges. The check: counties
                new to the model in 2024 scored 1.06 at $k=70$, close to C.

                What would change the answer: her records joining the training data, or a
                forecast for bridges the model has already seen.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question** (one minute, no notes): your model will forecast the next inspection of
    the same bridges. Which split do you validate with, and which columns do you drop? Why?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Train on earlier inventories and validate on a later one, at the same horizon as
                the use, keeping each inspection once. Drop the inspection interval and the
                structural evaluation: the rating sets both.
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
            "Q1. Why is the training error of 1-NN (almost) zero, and why does that say nothing?": mo.md(
                r"""
                Each training record is its own nearest neighbor, at distance 0, so it predicts
                itself. The few exceptions share all 12 inputs with another record of a
                different rating. The training error measures memory, not generalization.
                """
            ),
            "Q2. With the neighbors fixed and noise variance σ² = 0.16, what is the variance of a 4-NN prediction?": mo.md(
                r"""
                $\sigma^2/k = 0.16/4 = 0.04$, a standard deviation of 0.2 rating points. On
                bridges the measured variance falls more slowly, because the neighbors
                themselves change with the training set.
                """
            ),
            "Q3. You pick k by the lowest of 12 validation errors. Is that error an unbiased estimate of the chosen model's error?": mo.md(
                r"""
                No. Each validation error is unbiased for its own $k$, but the minimum is
                biased low, $\mathbb{E}[\min_k \hat L_k] \le \min_k \mathbb{E}[\hat L_k]$
                (Cawley & Talbot, 2010). Report a test error measured once, after every
                choice.
                """
            ),
            "Q4. Why fit the scaler inside each fold?": mo.md(
                r"""
                The means and standard deviations are part of the encoder. Computing them on
                all records lets held-out statistics shape the training representation, a
                small but real leak, and it would not be possible for a future bridge.
                """
            ),
            "Q5. Name a column you would drop from an NBI model of condition, and say why.": mo.md(
                r"""
                The inspection interval (set from the condition at the inspection), the
                structural evaluation (computed from the superstructure and substructure
                ratings), the bridge's good/fair/poor class or lowest rating (the target
                itself), and the work proposed or its cost (decided because of the
                condition). Ask of every column: would I have it when I make the prediction?
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Shuffled K-fold on dependent records.** Repeated inspections of one bridge, twin
      bridges, and bridges on one corridor must be held out together (`GroupKFold`), by the
      unit the model will face: bridge, county, or time.
    - **The same record twice.** A bridge on a 48-month interval reports one inspection in two
      inventories; deduplicate by (structure, inspection date) before splitting.
    - **Test records already in training.** A later inventory can repeat an inspection that is
      in training; drop it from the test set.
    - **Columns set from the target**: inspection interval, structural evaluation,
      good/fair/poor class, proposed work and its cost.
    - **Selecting on the test set.** Choose $k$ by validation; look at the test set once.
    - **Scaler fit on all data.** Fit `StandardScaler` inside each fold.
    - **RMSE-optimal is not decision-optimal.** The $k$ with the lowest RMSE did not catch the
      most poor bridges; evaluate with the decision's own metric.
    - **Ties in neighbor search.** Two codes may break distance ties differently; compare away
      from ties, and count them.
    - **One bridge, two county codes.** Ten training bridges change county code between
      inventories; group the folds by a code fixed per bridge (here its first county), or a
      bridge's other records train the model scored on "its county".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §1.1.3** — overfitting and cross-validation: why training error is not test
      error, and why an error estimate from held-out data is unbiased. That argument assumes
      the held-out data are drawn like future data, which is exactly what §5 tests.
    - **Forsyth §3.1** — held-out loss predicts test loss, through Chebyshev's inequality and a
      simple generalization bound. Forsyth writes the held-out loss as $L^{(N)}$; we write
      $\hat L$.
    - **Forsyth §11.1.1 and §11.1.3** — bias and variance, and choosing a model by
      cross-validation (Worked Example 11.2 uses leave-one-out).
    - **Exercise.** Repeat §5 with a forward split at the deployment horizon: train on 2014,
      validate on 2018 (four years), and compare with training on 2014–2018 and testing on
      2022. Does the forward split predict the later error better than random folds?
    - **Exercise.** Replace the four material flags by the full design type (Item 43B, one-hot)
      and rerun the county-held-out curve. Does the bias at large $k$ change?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - Validation is honest only for cases like those the model will face.
    - Random folds reused bridges, twins, and repeated inspections: they
      promised 0.90 and picked $k=2$; the 2024 inventory gave 1.12.
    - Hold out what the use holds out: whole counties for a new county, later inventories for
      the future.
    - Expected error = noise + bias² + variance; $k$ trades variance for bias.
    - Drop columns the target decides; keep one copy of each inspection.

    **Next:** retrieval, clustering, and PCA, on building load profiles.
    """)
    return


if __name__ == "__main__":
    app.run()
