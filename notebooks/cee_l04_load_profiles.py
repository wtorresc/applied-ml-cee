"""Applied Machine Learning for CEE — Lecture 4: Retrieval, Clustering, and Modes.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l04_load_profiles.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_bdg2.sh`` once (Building Data Genome 2 electricity meters).
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
    # Applied Machine Learning for CEE — Lecture 4: Retrieval, Clustering, and Modes

    **Slides:** *Applied Machine Learning for CEE*, Lecture 4 (`lectures/cee_l04_load_profiles`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §8.2 ($k$-means), §5.1 (principal components), §6.1 (the SVD).
    **Data:** Building Data Genome 2 (Miller et al., 2020, *Scientific Data* 7, 368), hourly electricity meters of 1,578 buildings, 2016–2017; fetch it with `bash scripts/fetch_bdg2.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`.

    **Your tasks** (from the slides): build the shapes, `pca_svd`, and Lloyd's algorithm; match
    scikit-learn and FAISS; work Q and A–D by hand; and reproduce the size, leakage, and outlier
    tests.
    """)
    return


@app.cell
def _():
    import os
    import random
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)  # child processes only: this kernel's is fixed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    REPEATS = 30  # repeated site-held-out splits
    TEST_SIZE = 0.25  # share of sites held out in each
    MIN_COMPLETE = 0.8  # share of a year's weekdays with all 24 hours metered
    R = 3  # modes kept in the picture and the trace
    K = 4  # typical days
    NIGHT, DAY = 0, 3  # four-hour blocks: 0-4 h and 12-16 h
    NIGHT_HOURS = 8  # rough annualization of excess night load (midnight to 8 am)
    HOURS = np.arange(24)
    # Same colors and markers as the slides (Okabe–Ito; color is never the only cue).
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        DAY,
        GroupShuffleSplit,
        HOURS,
        K,
        KMeans,
        MARKERS,
        MIN_COMPLETE,
        NIGHT,
        NIGHT_HOURS,
        PALETTE,
        PCA,
        Path,
        R,
        REPEATS,
        SEED,
        StandardScaler,
        TEST_SIZE,
        dataclass,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 1–3

    On your own, no notes, two minutes:

    1. Lecture 2: a raw distance between concrete mixes barely noticed the water. Why, and what
       fixed it?
    2. Lecture 3: bridge records were split into blocks of counties or years, not at random. Why?
    3. Lecture 1: a model's test error is low. Name one more thing you need before you sign a
       decision that rests on it.

    **Today:** which campus buildings should get an after-hours visit?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. A Euclidean distance is dominated by the inputs with the largest spread, so
                   kilograms of cement and aggregate drowned out water. Engineering fixed it:
                   dimensionless groups, above all the water–binder ratio. Today asks the same
                   question of buildings, where size plays the part of the big units.
                2. Nearby bridges share climate, traffic, materials, and inspectors, and records
                   of one bridge in different years are near-copies. A random split puts
                   near-copies on both sides and flatters the model; holding out blocks of
                   counties or years imitates a new county or a future year.
                3. Any of: the error on cases like this one; whether this case lies inside the
                   data; what an error costs and who carries it; that the test set was used only
                   once; who is accountable.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A campus energy manager can send an engineer to a few buildings a month to find
    **after-hours waste**: lights, air handlers, and equipment left running overnight. Hourly
    electricity meters already record every building. **Can they choose the visits?**

    The two errors are not equal. A **wasted visit** costs an engineer's day, spent on a building
    that was fine. A **missed building** keeps wasting energy every night, in money and in carbon.
    We return to this in §6, with the ranking of a real campus.

    **The data.** Building Data Genome 2 (BDG2): hourly electricity (kWh in each hour, so the mean
    kW) of 1,578 buildings at 19 sites in North America and Europe, 2016–2017, in local time.
    Buildings carry code names (`Hog_office_Elnora`: site Hog, an office). One object runs through
    the lecture, a building's **typical weekday**: the mean kW in each hour of the day over a
    year's weekdays that have all 24 hours metered, a vector $\mathbf{x}\in\mathbb{R}^{24}$.
    """)
    return


@app.cell
def _(MIN_COMPLETE, Path, dataclass, mo, np, pd):
    @dataclass(frozen=True)
    class Portfolio:
        """Buildings with a usable typical weekday in 2016 and 2017.

        Attributes:
            name: Building code names, shape ``(N,)``.
            site: Site code names.
            use: Primary use (``"Unknown"`` when missing).
            sqm: Gross floor area, m².
            P16: Typical weekday 2016, kW per hour of the day, shape ``(N, 24)``.
            P17: Typical weekday 2017, same layout.
            counts: How many buildings each eligibility rule removed.
        """

        name: np.ndarray
        site: np.ndarray
        use: np.ndarray
        sqm: np.ndarray
        P16: np.ndarray
        P17: np.ndarray
        counts: dict

    def find_data() -> tuple[Path, Path]:
        """Locate the derived meter file and the metadata from the repo root or notebooks/.

        Returns:
            Paths of ``bdg2_daily_electricity.npz`` and ``metadata.csv``.

        Raises:
            FileNotFoundError: If they are missing (run ``bash scripts/fetch_bdg2.sh``).
        """
        rel_npz = Path("data") / "derived" / "bdg2_daily_electricity.npz"
        rel_meta = Path("data") / "raw" / "bdg2" / "metadata.csv"
        bases = [Path.cwd(), Path.cwd().parent]
        if mo.notebook_dir() is not None:
            bases.append(Path(mo.notebook_dir()).parent)
        for base in bases:
            if (base / rel_npz).exists() and (base / rel_meta).exists():
                return base / rel_npz, base / rel_meta
        raise FileNotFoundError(f"{rel_npz} not found; run `bash scripts/fetch_bdg2.sh`")

    def load_bdg2(
        npz: Path, meta_csv: Path
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
        """Hourly kWh as ``(building, day, hour)``, building ids, dates, and metadata.

        Args:
            npz: File written by ``scripts/fetch_bdg2.sh``.
            meta_csv: BDG2 ``metadata.csv``.

        Returns:
            kWh (NaN where missing), ids, dates, and metadata rows in the same order.
        """
        d = np.load(npz)
        kwh, names = d["kwh"], d["building_id"]
        dates = pd.DatetimeIndex(pd.to_datetime(d["date"]))
        assert kwh.shape == (len(names), len(dates), 24), kwh.shape
        meta = pd.read_csv(meta_csv).set_index("building_id").loc[names]
        return kwh, names, dates, meta

    def typical_weekday(kwh: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean kW in each hour over the kept days that have all 24 hours metered.

        Args:
            kwh: Hourly kWh, shape ``(B, D, 24)``.
            keep: Boolean mask of days, shape ``(D,)``.

        Returns:
            Profiles ``(B, 24)`` and the number of days averaged ``(B,)``.
        """
        sub = kwh[:, keep, :].astype(np.float64)  # the file stores float32: sum in float64
        ok = ~np.isnan(sub).any(axis=2)
        n = ok.sum(axis=1)
        prof = np.where(ok[..., None], sub, 0.0).sum(axis=1) / np.maximum(n, 1)[:, None]
        assert prof.shape == (len(kwh), 24), prof.shape
        return prof, n

    def build_portfolio(
        kwh: np.ndarray, names: np.ndarray, dates: pd.DatetimeIndex, meta: pd.DataFrame
    ) -> Portfolio:
        """Keep buildings with complete, positive, truly hourly typical weekdays in both years.

        Rules: (1) at least ``MIN_COMPLETE`` of each year's weekdays fully metered; (2) every hour
        of both typical weekdays positive; (3) fewer than half of the complete days perfectly flat
        (rule 3 removes meters that report one value per day, spread over the 24 hours).
        """
        weekday = np.asarray(dates.dayofweek < 5)
        year = np.asarray(dates.year)
        P16, n16 = typical_weekday(kwh, weekday & (year == 2016))
        P17, n17 = typical_weekday(kwh, weekday & (year == 2017))
        enough = (n16 >= MIN_COMPLETE * (weekday & (year == 2016)).sum()) & (
            n17 >= MIN_COMPLETE * (weekday & (year == 2017)).sum()
        )
        positive = (P16.min(axis=1) > 0) & (P17.min(axis=1) > 0)
        complete = ~np.isnan(kwh).any(axis=2)
        flat_day = np.ptp(np.where(complete[..., None], kwh, 0.0), axis=2) == 0
        flat_share = (flat_day & complete).sum(axis=1) / np.maximum(complete.sum(axis=1), 1)
        hourly = flat_share < 0.5
        keep = enough & positive & hourly
        flat_sites = meta["site_id"].to_numpy()[enough & positive & ~hourly]
        counts = {
            "buildings_in_file": int(len(names)),
            "sites_in_file": int(meta["site_id"].nunique()),
            "incomplete_or_nonpositive": int((~(enough & positive)).sum()),
            "daily_meters": int((enough & positive & ~hourly).sum()),
            "daily_meters_top_site_count": int(pd.Series(flat_sites).value_counts().iloc[0]),
            "eligible": int(keep.sum()),
        }
        m = meta[keep]
        return Portfolio(
            name=names[keep],
            site=m["site_id"].to_numpy(),
            use=m["primaryspaceusage"].fillna("Unknown").to_numpy(),
            sqm=m["sqm"].to_numpy(np.float64),
            P16=P16[keep],
            P17=P17[keep],
            counts=counts,
        )

    return build_portfolio, find_data, load_bdg2, typical_weekday


@app.cell
def _(build_portfolio, find_data, load_bdg2, np, pd):
    kwh, names, dates, meta = load_bdg2(*find_data())
    pf = build_portfolio(kwh, names, dates, meta)
    # Slide "1,159 buildings, one typical weekday": 1,578 buildings at 19 sites; 356 dropped for
    # missing data; 63 daily meters, 61 at one site; 1,159 kept at 15 sites.
    assert (pf.counts["buildings_in_file"], pf.counts["sites_in_file"]) == (1578, 19)
    assert (pf.counts["incomplete_or_nonpositive"], pf.counts["daily_meters"]) == (356, 63)
    assert pf.counts["daily_meters_top_site_count"] == 61
    assert (pf.counts["eligible"], len(set(pf.site))) == (1159, 15)
    pd.Series(
        {
            **pf.counts,
            "sites kept": len(set(pf.site)),
            "median size (kW)": round(float(np.median(pf.P16.mean(axis=1))), 1),
        },
        name="value",
    ).to_frame()
    return dates, kwh, names, pf


@app.cell
def _(mo):
    mo.md(r"""
    **Why 63 meters were dropped.** Their every day is perfectly flat: the meter reports one value
    per day, which the data spread evenly over the 24 hours. To any method in this lecture they
    would look like buildings that never sleep. Across the whole file 81 meters behave this way,
    71 of them at one site (Gator); 63 of them would otherwise have passed the completeness rules.
    The next cell counts them by site and prints one day of one of them.
    """)
    return


@app.cell
def _(kwh, names, np, pd, pf):
    # Every perfectly flat day of every building in the file, against the eligible buildings.
    _complete = ~np.isnan(kwh).any(axis=2)
    _flat = np.ptp(np.where(_complete[..., None], kwh, 0.0), axis=2) == 0
    _share = (_flat & _complete).sum(axis=1) / np.maximum(_complete.sum(axis=1), 1)
    _daily = pd.Series(names[_share >= 0.5]).str.split("_").str[0].value_counts()
    _i = int(np.flatnonzero(_share >= 0.5)[0])
    _first_day = int(np.flatnonzero(_complete[_i])[0])
    # In the whole file: 81 daily meters, 71 of them at Gator; 63 pass rules 1-2 and are
    # dropped by rule 3 (the load cell). None is among the kept buildings.
    assert (int(_daily.sum()), _daily.index[0], int(_daily.iloc[0])) == (81, "Gator", 71)
    assert not np.isin(names[_share >= 0.5], pf.name).any()
    (
        _daily.rename("buildings with daily meters, by site").to_frame(),
        pd.Series(kwh[_i, _first_day].round(2), name=f"{names[_i]}, one day (kWh per hour)"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Helpers used in every section.** Shape, four-hour blocks, the night/day ratio, the two kinds
    of split, and a root-mean-square error. The held-out **campus** is the site with the most
    offices: it plays the energy manager's campus, new to every model.
    """)
    return


@app.cell
def _(DAY, GroupShuffleSplit, NIGHT, REPEATS, SEED, TEST_SIZE, np, pd):
    def shape(P: np.ndarray) -> np.ndarray:
        """Divide each profile by its own mean: the shape, unitless with mean 1."""
        return P / P.mean(axis=1, keepdims=True)

    def blocks(P: np.ndarray) -> np.ndarray:
        """Mean of each four-hour block of the day, shape ``(N, 6)``."""
        out = P.reshape(len(P), 6, 4).mean(axis=2)
        assert out.shape == (len(P), 6), out.shape
        return out

    def night_day(P: np.ndarray) -> np.ndarray:
        """Night (0-4 h) over afternoon (12-16 h) load."""
        b = blocks(P)
        return b[:, NIGHT] / b[:, DAY]

    def campus_split(site: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hold out the site with the most offices; train on every other site."""
        campus = pd.Series(site[use == "Office"]).value_counts().index[0]
        return np.flatnonzero(site != campus), np.flatnonzero(site == campus)

    def site_splits(site: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """``REPEATS`` splits holding out a quarter of the sites, each site kept whole."""
        gss = GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SEED)
        return list(gss.split(np.zeros(len(site)), groups=site))

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square difference over every building and hour.

        Raises ValueError if the shapes differ: (n,) against (n, 1) would broadcast to an n x n
        difference and return a meaningless number.
        """
        if np.shape(a) != np.shape(b):
            raise ValueError(f"rms of arrays of different shapes: {np.shape(a)} and {np.shape(b)}")
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return blocks, campus_split, night_day, rms, shape, site_splits


@app.cell
def _(campus_split, pf):
    tr0, te0 = campus_split(pf.site, pf.use)
    # Slide "1,159 buildings": one campus (Hog) held out, 136 buildings; 14 campuses, 1,023 kept.
    assert set(pf.site[te0]) == {"Hog"} and (len(te0), len(tr0)) == (136, 1023)
    assert len(set(pf.site[tr0])) == 14 and int((pf.use[te0] == "Office").sum()) == 67
    return te0, tr0


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: which building runs like Q?

    Mean kW on a 2016 weekday, by four-hour block. Q, A, and B are offices on one campus; C and D
    are at other sites. On your own, two minutes: **which of A–D runs most like Q? Would you send
    an engineer to look at Q's nights?** Write down your answer and your reason before reading
    on.
    """)
    return


@app.cell(hide_code=True)
def _(DAY, NIGHT, blocks, night_day, np, pd, pf, shape, te0, tr0):
    # Same selection as the slides (figure script, `pick_attempt`). The code is hidden in the
    # editor because it uses the distances, which would give the answer away.
    _rho = night_day(pf.P16)
    _b = blocks(pf.P16)
    # The office yardstick: the median night/day ratio of the offices on the 14 other campuses,
    # 0.658, used as 0.66 so that every excess on the slides can be checked by hand.
    _ref_exact = float(np.median(_rho[tr0][pf.use[tr0] == "Office"]))
    office_ref = round(_ref_exact, 2)
    _excess = _b[:, NIGHT] - office_ref * _b[:, DAY]
    _offices = te0[pf.use[te0] == "Office"]
    # Q: the held-out office with the most night load above the office yardstick.
    _q = int(_offices[np.argmax(_excess[_offices])])
    _size = pf.P16.mean(axis=1)
    _peers = np.flatnonzero((pf.site == pf.site[_q]) & (pf.use == "Office"))
    _peers = _peers[_peers != _q]
    _by_size = _peers[np.argsort(np.abs(_size[_peers] - _size[_q]), kind="stable")[:2]]
    _S = shape(pf.P16)
    _by_shape = tr0[np.argsort(np.sqrt(((_S[tr0] - _S[_q]) ** 2).sum(axis=1)), kind="stable")[:2]]
    _four = np.concatenate([_by_size, _by_shape])
    # Letter A-D by raw distance to Q, nearest first.
    _four = _four[
        np.argsort(np.sqrt(((pf.P16[_four] - pf.P16[_q]) ** 2).sum(axis=1)), kind="stable")
    ]
    attempt_rows = pd.Series([_q, *_four], index=list("QABCD"))

    _table = pd.DataFrame(
        np.round(_b[attempt_rows]).astype(int),
        index=attempt_rows.index,
        columns=["0–4 h", "4–8 h", "8–12 h", "12–16 h", "16–20 h", "20–24 h"],
    )
    _table.insert(0, "Site", pf.site[attempt_rows])
    _table.insert(0, "Use", pf.use[attempt_rows])
    # Slide "Your turn": the table, Q = Hog_office_Elnora; A, B its campus peers by size; C, D
    # its nearest training buildings by shape; yardstick 0.66 from 157 training offices.
    assert pf.name[_q] == "Hog_office_Elnora"
    assert list(pf.name[_four]) == [
        "Hog_office_Terry",
        "Hog_office_Betsy",
        "Rat_education_Jacob",
        "Fox_public_Belle",
    ]
    assert _table.loc["Q", "0–4 h":].tolist() == [690, 689, 708, 722, 709, 674]
    assert _table.loc["D", "0–4 h":].tolist() == [56, 56, 59, 60, 58, 57]
    assert office_ref == 0.66 and round(_ref_exact, 4) == 0.658
    assert int((pf.use[tr0] == "Office").sum()) == 157
    # Slide "Close shapes, buildings run alike": A and B dip to 0.75 and 0.73 and rise to 1.26;
    # Q, C and D stay between 0.96 and 1.04.
    _lo = {_L: round(float(_S[_i].min()), 2) for _L, _i in attempt_rows.items()}
    _hi = {_L: round(float(_S[_i].max()), 2) for _L, _i in attempt_rows.items()}
    assert (_lo["A"], _lo["B"], _hi["A"], _hi["B"]) == (0.75, 0.73, 1.26, 1.26)
    assert min(_lo[_L] for _L in "QCD") == 0.96 and max(_hi[_L] for _L in "QCD") == 1.04
    _table
    return attempt_rows, office_ref


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer: two common answers, and what the data say": mo.md(
                r"""
                - **By kilowatts:** A and B, offices of the same size on the same campus.
                - **By schedule:** C and D. Their nights stay level with their afternoons, as Q's
                  do, at a tenth of Q's size.
                - Q's 0–4 h load is **96%** of its 12–16 h load; A's and B's nights drop to 66%
                  and 65%. An office whose nights look like its afternoons is worth a look (§6).

                Keep your answer: §3 measures both distances, and §4 shows which buildings the
                methods pick.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture from Lecture 1, with today's step highlighted. There is no label: the
    target is the input's own shape, as in every unsupervised method.

    > day $\mathbf{x}$ (24 h, kW) → **encoder** ($\mathbf{p}=\mathbf{x}/\bar{x}$, then
    > $\mathbf{z}=V_r^\top(\mathbf{p}-\boldsymbol{\mu})$) → $\mathbf{z}\in\mathbb{R}^3$ →
    > **decoder** ($\hat{\mathbf{p}}=\boldsymbol{\mu}+V_r\mathbf{z}$) → loss
    > $\lVert\mathbf{p}-\hat{\mathbf{p}}\rVert^2$

    The fitting is an SVD of the 2016 shapes on the 14 other campuses (1,023 buildings). We follow
    Q, the office from your attempt: size 699 kW, night 690 kW, afternoon 722 kW. §4 traces it by
    hand: three numbers rebuild its day to 2.2% of its mean, its nearest buildings are C and D, and
    §5 finds its nearest typical day.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** buildings that run alike have daily shapes that are close in Euclidean
    distance.

    **Size and shape.** Write the typical weekday as size times shape,
    $$\mathbf{x}=\bar{x}\,\mathbf{p},\qquad \bar{x}=\tfrac{1}{24}\textstyle\sum_h x_h\ \text{(kW)},
    \qquad \mathbf{p}=\mathbf{x}/\bar{x}\ \text{(unitless, mean 1)}.$$
    Every shape has mean 1, so $\mathbf{p}-\mathbf{1}$ is orthogonal to the all-ones vector
    $\mathbf{1}$: $\mathbf{1}^\top(\mathbf{p}-\mathbf{1})=24-24=0$. For two buildings, split
    $\bar{x}\,\mathbf{p}-\bar{x}'\mathbf{p}'=(\bar{x}-\bar{x}')\,\mathbf{1}+\big[\bar{x}
    (\mathbf{p}-\mathbf{1})-\bar{x}'(\mathbf{p}'-\mathbf{1})\big]$ into those two orthogonal
    parts; Pythagoras gives
    $$\lVert\mathbf{x}-\mathbf{x}'\rVert^2=24\,(\bar{x}-\bar{x}')^2
    +\big\lVert\bar{x}(\mathbf{p}-\mathbf{1})-\bar{x}'(\mathbf{p}'-\mathbf{1})\big\rVert^2
    \;\ge\;24\,\Delta^2,\qquad \Delta=\bar{x}-\bar{x}' .$$
    A difference in size of $\Delta$ kW costs at least $24\Delta^2$, whatever the schedules;
    the schedule term compares each building's deviations from a flat day, in kW, so it too
    scales with size. Across buildings from 0.2 kW to 3.5 MW the size term wins.
    **Retrieval** returns the stored buildings nearest to $\mathbf{p}$ in
    $\lVert\mathbf{p}-\mathbf{p}_i\rVert_2$: Lecture 2's search, without the average.

    ### Worked example: Q and A–D, in kW and in shape
    """)
    return


@app.cell
def _(HOURS, MARKERS, PALETTE, attempt_rows, np, pd, pf, plt, shape):
    _x = pf.P16[attempt_rows]
    _S = shape(_x)
    _d_raw = np.sqrt(((_x[1:] - _x[0]) ** 2).sum(axis=1))
    _d_shape = np.sqrt(((_S[1:] - _S[0]) ** 2).sum(axis=1))
    # The derivation's orthogonal split, checked for Q against A-D:
    # ||x - x'||^2 = 24 (xbar - xbar')^2 + ||xbar (p - 1) - xbar' (p' - 1)||^2 >= 24 Delta^2.
    _xbar = _x.mean(axis=1)
    _dev = _xbar[:, None] * (_S - 1.0)  # kW above or below a flat day
    _split = 24 * (_xbar[1:] - _xbar[0]) ** 2 + ((_dev[1:] - _dev[0]) ** 2).sum(axis=1)
    assert np.allclose(_d_raw**2, _split, rtol=1e-12, atol=0)
    assert np.all(_d_raw**2 >= 24 * (_xbar[1:] - _xbar[0]) ** 2)
    # Slide "Close shapes, buildings run alike": raw 536, 596, 3,114, 3,141 kW;
    # shape 0.77, 0.84, 0.07, 0.07; sizes from 0.2 kW to 3.5 MW.
    assert [round(float(_v)) for _v in _d_raw] == [536, 596, 3114, 3141]
    assert [round(float(_v), 2) for _v in _d_shape] == [0.77, 0.84, 0.07, 0.07]
    _sizes = pf.P16.mean(axis=1)
    assert (round(float(_sizes.min()), 1), round(float(_sizes.max()) / 1000, 1)) == (0.2, 3.5)

    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _j, _l in enumerate("ABCD"):
        _ax.plot(
            HOURS,
            _S[_j + 1],
            color=PALETTE[_j],
            marker=MARKERS[_j],
            markevery=4,
            label=f"{_l}: {_x[_j + 1].mean():.0f} kW",
        )
    _ax.plot(HOURS, _S[0], color="#1A1A1A", lw=3, label=f"Q: {_x[0].mean():.0f} kW")
    _ax.set(xlabel="hour of the day", ylabel="load / daily mean")
    _ax.legend(ncol=2, fontsize=8)
    _fig.tight_layout()
    (
        pd.DataFrame(
            {
                "raw distance to Q (kW)": _d_raw.round(0),
                "shape distance": _d_shape.round(3),
                "night/day": pd.Series(_S[1:, 0:4].mean(1) / _S[1:, 12:16].mean(1))
                .round(2)
                .to_numpy(),
            },
            index=list("ABCD"),
        ),
        _fig,
    )
    return


@app.cell(hide_code=True)
def _(attempt_rows, mo, np, pf, shape):
    # Numbers for the check question. Hidden in the editor because the asserts give it away.
    _x = pf.P16[attempt_rows["Q"]]
    _doubled = 2 * _x
    _raw = float(np.sqrt(((_doubled - _x) ** 2).sum()))
    _others = np.sqrt(((pf.P16[attempt_rows.iloc[1:]] - _x) ** 2).sum(axis=1))
    _shape = float(np.abs(shape(_doubled[None])[0] - shape(_x[None])[0]).max())
    # Slide notes: the doubled Q is 3,423 kW away in raw kW, farther than any of A-D; 0 in shape.
    assert round(_raw) == 3423 and _raw > _others.max() and _shape < 1e-12
    mo.accordion(
        {
            "Check: double every hour of Q's load. Near or far?": mo.md(
                r"""
                **Far, in raw kW:** the doubled Q is 3,423 kW away from Q, farther than any of
                A–D. **Identical, in shape:** the distance is exactly 0. Same schedule, twice the
                size, which is what an engineer would say too.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $k$-means: a few typical days

    Choose $K$ centers $\mathbf{c}_k$ (typical days) and assignments $z_n\in\{1,\dots,K\}$ to
    minimize
    $$J(\mathbf{z},\mathbf{c})=\sum_{n=1}^{N}\lVert\mathbf{p}_n-\mathbf{c}_{z_n}\rVert^2 .$$

    1. **Assign.** With the centers fixed, each term depends only on its own $z_n$, so
       $z_n=\arg\min_k\lVert\mathbf{p}_n-\mathbf{c}_k\rVert^2$ minimizes every term.
    2. **Update.** With the assignments fixed, $J$ separates over clusters. The gradient of
       $\sum_{n\in S_k}\lVert\mathbf{p}_n-\mathbf{c}\rVert^2$ is $-2\sum_{n\in S_k}(\mathbf{p}_n-
       \mathbf{c})$; setting it to zero gives the mean of the cluster.

    Neither step can raise $J$, and there are finitely many assignments, so Lloyd's algorithm
    (Lloyd, 1982) stops, at a **local** minimum that depends on the start. Each iteration costs
    $O(NKD)$. Use several starts, seeded with $k$-means++ (Arthur & Vassilvitskii, 2007).

    **By hand.** Use the block shapes of the five buildings in the attempt table (each
    building's six four-hour blocks divided by their mean), $K=2$, started at A and C.
    """)
    return


@app.cell
def _(attempt_rows, blocks, np, pd, pf):
    _B = blocks(pf.P16[attempt_rows])
    _B = _B / _B.mean(axis=1, keepdims=True)  # block shapes, mean 1
    _C0 = _B[[1, 3]]  # start at A and C
    _d2 = ((_B[:, None, :] - _C0[None, :, :]) ** 2).sum(-1)  # (5, 2)
    _z = _d2.argmin(axis=1)  # assign
    _J0 = float(_d2[np.arange(5), _z].sum())
    _C1 = np.stack([_B[_z == _k].mean(axis=0) for _k in range(2)])  # update
    _J1 = float(((_B - _C1[_z]) ** 2).sum())
    _z2 = ((_B[:, None, :] - _C1[None, :, :]) ** 2).sum(-1).argmin(axis=1)
    _letters = np.array(list("QABCD"))
    # Slide "k-means: a few typical days": {A, B} and {Q, C, D}; notes: J 0.0048 -> 0.0026,
    # and the next assignment does not change.
    assert [list(_letters[_z == _k]) for _k in range(2)] == [["A", "B"], ["Q", "C", "D"]]
    assert (round(_J0, 4), round(_J1, 4)) == (0.0048, 0.0026) and np.array_equal(_z, _z2)
    (
        pd.DataFrame(
            _B.round(2),
            index=list("QABCD"),
            columns=["0–4", "4–8", "8–12", "12–16", "16–20", "20–24"],
        ).assign(cluster=_z),
        pd.DataFrame(_C1.round(2), index=["center 1", "center 2"]),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### PCA: the best $r$-number summary

    Stack the centered shapes as the rows of $\tilde{P}$ ($N\times 24$) and let
    $C=\tilde{P}^\top\tilde{P}/(N-1)$.

    1. **Variance along a direction.** For a unit vector $\mathbf{v}$, the scores
       $\tilde{\mathbf{p}}_n^\top\mathbf{v}$ have variance $\mathbf{v}^\top C\mathbf{v}$.
    2. **Maximize it.** The Lagrangian $\mathbf{v}^\top C\mathbf{v}-\lambda(\mathbf{v}^\top
       \mathbf{v}-1)$ has zero gradient when $C\mathbf{v}=\lambda\mathbf{v}$, and then the
       variance is $\lambda$: take the top eigenvector. The next mode maximizes variance
       orthogonal to the first, and so on.
    3. **Keep $r$ modes.** $\mathbf{z}=V_r^\top(\mathbf{p}-\boldsymbol{\mu})$,
       $\hat{\mathbf{p}}=\boldsymbol{\mu}+V_r\mathbf{z}$. The modes are orthonormal, so the squared
       length of a centered shape splits into its squared scores, and the dropped modes carry the
       error:
       $$\frac{1}{N-1}\sum_n\lVert\mathbf{p}_n-\hat{\mathbf{p}}_n\rVert^2=\sum_{k>r}\lambda_k .$$
       Keeping the most variance is the same as making the least error; no rank-$r$ summary
       does better in squared error (Eckart & Young, 1936).
    4. **SVD.** $\tilde{P}=U\Sigma V^\top$ gives $\tilde{P}^\top\tilde{P}=V\Sigma^2V^\top$: the
       columns of $V$ are the modes and $\lambda_k=\sigma_k^2/(N-1)$, without forming $C$, which
       would square the condition number.

    The next cell checks step 3 on the 1,023 training shapes.
    """)
    return


@app.cell
def _(R, np, pf, shape, tr0):
    _P = shape(pf.P16[tr0])
    _Pc = _P - _P.mean(axis=0)
    _, _sig, _Vt = np.linalg.svd(_Pc, full_matrices=False)
    _lam = _sig**2 / (len(_P) - 1)
    _V = _Vt[:R].T
    _err = ((_Pc - _Pc @ _V @ _V.T) ** 2).sum() / (len(_P) - 1)
    assert np.isclose(_err, _lam[R:].sum(), rtol=1e-10, atol=0)
    # The eigenvectors of C are the right singular vectors (up to sign).
    _w, _U = np.linalg.eigh(_Pc.T @ _Pc / (len(_P) - 1))
    assert np.allclose(_w[::-1][:R], _lam[:R], rtol=1e-10, atol=0)
    assert np.allclose(np.abs(_U[:, ::-1][:, :R].T @ _V), np.eye(R), atol=1e-8)
    {"error (r = 3)": round(float(_err), 6), "sum of dropped λ": round(float(_lam[R:].sum()), 6)}
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **PCA in seven lines of NumPy** (the slide's listing). Shapes, not kilowatts, go in.
    """)
    return


@app.cell
def _(np):
    def pca_svd(P: np.ndarray, r: int) -> tuple[np.ndarray, ...]:
        """Mean shape, top-r modes, their variances, and the scores."""
        mu = P.mean(axis=0)  # mean shape, (24,)
        _, sig, Vt = np.linalg.svd(P - mu, full_matrices=False)
        lam = sig**2 / (len(P) - 1)  # variance along each mode
        V = Vt[:r].T  # modes as columns, (24, r)
        Z = (P - mu) @ V  # scores, (N, r)
        assert V.shape == (P.shape[1], r) and Z.shape == (len(P), r)
        return mu, V, lam, Z

    return (pca_svd,)


@app.cell
def _(mo):
    mo.md(r"""
    **Predict.** Fit on the 1,023 training shapes with $r=3$. Will `pca_svd` match
    scikit-learn's `PCA` exactly, up to rounding, or not at all? What could differ? Write your
    answer down, then **run** the next cell.
    """)
    return


@app.cell
def _(PCA, R, np, pca_svd, pf, shape, tr0):
    _S = shape(pf.P16[tr0])
    _mu, _V, _lam, _Z = pca_svd(_S, R)
    _ref = PCA(n_components=R, svd_solver="full").fit(_S)
    _sign = np.sign((_V.T * _ref.components_).sum(axis=1))
    _checks = {
        "max |variance difference|": float(np.abs(_lam[:R] - _ref.explained_variance_).max()),
        "max |mode difference|, signs aligned": float(
            np.abs(_V.T * _sign[:, None] - _ref.components_).max()
        ),
        "max |score difference|, signs aligned": float(
            np.abs(_Z * _sign - _ref.transform(_S)).max()
        ),
    }
    assert np.allclose(_lam[:R], _ref.explained_variance_, rtol=0, atol=1e-10)
    assert np.allclose(_V.T * _sign[:, None], _ref.components_, rtol=0, atol=1e-10)
    assert np.allclose(_Z * _sign, _ref.transform(_S), rtol=0, atol=1e-10)
    # Slide "The first mode is the working day": agreement to round-off, below 1e-13. How many
    # modes flip sign is the linear-algebra library's choice, so it is shown, not asserted.
    assert max(_checks.values()) < 1e-13
    _checks | {"modes with flipped sign": int((_sign < 0).sum())}
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Result.** Everything matches to round-off (below $10^{-13}$), once signs are aligned. Some
    modes come out with the opposite sign (the cell counts them; which ones depends on the
    linear-algebra library, not on the data). Both signs are valid, since $\mathbf{v}$ and
    $-\mathbf{v}$ span the same line; compare modes only after aligning signs.

    To make the scores the same on every machine, we fix each sign by a rule (next cell): mode 1
    has more load in the afternoon (12–16 h) than at night (0–4 h), mode 2 more in the evening
    (16–20 h) than in the early morning (4–8 h), mode 3 more in the afternoon than in the morning
    (8–12 h). Then the modes themselves: the mean shape plus and minus two standard deviations
    along each.
    """)
    return


@app.cell
def _(np):
    # Sign rules, mode by mode: (hours with more load, hours with less) when the score is positive.
    _SHAPE_RULES = (
        (slice(12, 16), slice(0, 4)),  # mode 1: afternoon above night
        (slice(16, 20), slice(4, 8)),  # mode 2: evening above early morning
        (slice(12, 16), slice(8, 12)),
    )  # mode 3: afternoon above morning
    KW_SIGN_RULES = ((slice(0, 24), slice(0, 0)),)  # raw kW: mode 1 sums positive

    def orient(V: np.ndarray, rules: tuple = _SHAPE_RULES) -> np.ndarray:
        """Fix the arbitrary sign of the leading modes by ``rules``; later modes keep theirs.

        Args:
            V: Modes as columns, shape ``(24, r)``.
            rules: ``(pos, neg)`` hours for modes 1, 2, ...

        Returns:
            ``V`` with each ruled mode flipped where needed.
        """
        _signs = np.ones(V.shape[1])
        for _k, (_pos, _neg) in enumerate(rules[: V.shape[1]]):
            _signs[_k] = np.sign(V[_pos, _k].sum() - V[_neg, _k].sum())
        assert np.all(_signs != 0), "a sign rule is undecided"
        return V * _signs

    return KW_SIGN_RULES, orient


@app.cell
def _(HOURS, PALETTE, night_day, np, orient, pca_svd, pf, plt, shape, tr0):
    _S = shape(pf.P16[tr0])
    _mu, _V, _lam, _ = pca_svd(_S, 24)
    _V = orient(_V)
    _share = _lam / _lam.sum()
    _corr = float(np.corrcoef((_S - _mu) @ _V[:, 0], night_day(pf.P16[tr0]))[0, 1])
    # Slide "The first mode is the working day": 65%, 19%, 7%; three modes 91%; r = -0.77;
    # mode 2 peaks near 20 h.
    assert [round(100 * float(_v), 1) for _v in _share[:3]] == [65.2, 18.6, 7.2]
    assert round(100 * float(_share[:3].sum()), 1) == 91.0
    assert round(_corr, 2) == -0.77 and int(np.argmax(_V[:, 1])) == 20
    # Its alt text: the mean shape runs from 0.77 to 1.24 (at 13 h); mode 1's +2 SD shape from
    # 0.45 at night to 1.79 (at 11 h).
    _plus = _mu + 2 * np.sqrt(_lam[0]) * _V[:, 0]
    assert (round(float(_mu.min()), 2), round(float(_mu.max()), 2), int(np.argmax(_mu))) == (
        0.77,
        1.24,
        13,
    )
    assert (round(float(_plus.min()), 2), round(float(_plus.max()), 2)) == (0.45, 1.79)

    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for _k, _ax in enumerate(_axes):
        _sd = np.sqrt(_lam[_k])
        _ax.plot(HOURS, _mu, ":", color="#595959", label="mean shape")
        _ax.plot(HOURS, _mu + 2 * _sd * _V[:, _k], "-", color=PALETTE[0], label="+2 SD")
        _ax.plot(HOURS, _mu - 2 * _sd * _V[:, _k], "--", color=PALETTE[1], label="−2 SD")
        _ax.set(title=f"mode {_k + 1}: {100 * _share[_k]:.0f}%", xlabel="hour")
    _axes[0].set_ylabel("load / daily mean")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(attempt_rows, dates, kmeans, kwh, names, np, orient, pca_svd, pf, rms, shape, tr0):
    # Slides "Hourly meters can choose the visits", "PCA: the line and plane of most spread",
    # "Lloyd's steps, drawn in 2D", and "Mean plus three modes rebuild Q's day".
    _q = int(attempt_rows["Q"])
    # All 2016 days, weekends included (the attempt's table uses typical weekdays).
    _hourly = kwh[list(names).index(pf.name[_q]), np.asarray(dates.year == 2016)].astype(float)
    assert (round(np.nanmean(_hourly[:, 0:4])), round(np.nanmean(_hourly[:, 12:16]))) == (679, 707)
    # The carpet's colour bar spans Q's lowest and highest hour of 2016: 293 to 994 kW.
    assert (round(np.nanmin(_hourly)), round(np.nanmax(_hourly))) == (293, 994)

    _S = shape(pf.P16[tr0])
    _, _, _lam3, _ = pca_svd(_S[:, [3, 9, 15]], 3)  # three hours of each building's day
    _share3 = _lam3 / _lam3.sum()
    # Shares over all 1,023 training buildings (the slide draws 400 of them).
    assert len(_S) == 1023
    assert (round(100 * float(_share3[0])), round(100 * float(_share3[:2].sum()))) == (79, 94)

    _, _V2, _, _ = pca_svd(_S, 2)
    _X2 = (_S - _S.mean(axis=0)) @ orient(_V2)  # scores on modes 1 and 2
    _C0 = _X2[np.random.default_rng(4).choice(len(_X2), 4, replace=False)]
    _, _, _hist = kmeans(_X2, _C0)
    assert (len(_hist), round(_hist[0], 1), round(_hist[-1], 1)) == (12, 471.9, 356.2)

    _Sall = shape(pf.P16)
    _mu, _V, _, _ = pca_svd(_S, 3)
    _V = orient(_V)
    _z = (_Sall[_q] - _mu) @ _V
    _err = [round(100 * rms(_mu + _V[:, :_r] @ _z[:_r], _Sall[_q]), 1) for _r in range(4)]
    assert [round(float(_v), 2) for _v in _z] == [-0.62, -0.38, -0.01]
    assert _err == [15.1, 8.1, 2.2, 2.2]
    _err
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Mode 1 (65%) is the working day**: day against night. Its score follows the engineer's
    night/day ratio ($r=-0.77$), found from data alone. **Mode 2 (19%)** is an evening peak near
    20 h against the early morning; **mode 3 (7%)** moves the day earlier or later.

    **$k$-means from scratch.** Lloyd's algorithm, checked against scikit-learn from the same
    four starting shapes.
    """)
    return


@app.cell
def _(np):
    def kmeans(
        X: np.ndarray, C0: np.ndarray, max_iter: int = 300
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Lloyd's algorithm from the given centers; stops when no assignment changes.

        Args:
            X: Points, shape ``(N, D)``.
            C0: Starting centers, shape ``(K, D)``.
            max_iter: Iteration cap.

        Returns:
            Centers ``(K, D)``, assignments ``(N,)``, and J after each assignment step.
        """
        C, z, history = C0.copy(), None, []
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # (N, K)
            z_new = d2.argmin(axis=1)  # assign
            history.append(float(d2[np.arange(len(X)), z_new].sum()))
            if z is not None and np.array_equal(z_new, z):
                break
            z = z_new
            # An empty cluster would get a NaN center, and argmin then sends every point to it.
            assert np.bincount(z, minlength=len(C)).min() > 0, "a cluster lost all its points"
            C = np.stack([X[z == k].mean(axis=0) for k in range(len(C))])  # update
        assert z is not None and C.shape == C0.shape
        return C, z, history

    return (kmeans,)


@app.cell
def _(K, KMeans, SEED, kmeans, np, pf, shape, tr0):
    _S = shape(pf.P16[tr0])
    _C0 = _S[np.random.default_rng(SEED).choice(len(_S), size=K, replace=False)]
    _C, _z, _hist = kmeans(_S, _C0)
    _ref = KMeans(n_clusters=K, init=_C0, n_init=1, algorithm="lloyd", tol=0.0, max_iter=300).fit(
        _S
    )
    _rel = abs(_hist[-1] - _ref.inertia_) / _ref.inertia_
    # Slide notes: identical labels, J within 1e-15 (scikit-learn sums J in a thread-dependent
    # order, so the difference is 0 or a few 1e-16), 14 iterations, J never rose.
    assert np.array_equal(_z, _ref.labels_) and _rel < 1e-15
    assert len(_hist) == 14 and np.all(np.diff(_hist) <= 1e-12)
    assert np.allclose(_C, _ref.cluster_centers_, rtol=0, atol=1e-12)
    {
        "same labels": True,
        "J": round(_hist[-1], 2),
        "relative J difference": f"{_rel:.0e}",
        "iterations": len(_hist),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Retrieval, exactly.** Brute force is a distance, a sort, and a cut, $O(ND)$ per query. FAISS's
    `IndexFlatL2` does the same exact search, in float32: it computes
    $\lVert\mathbf{q}\rVert^2-2\mathbf{q}^\top\mathbf{x}+\lVert\mathbf{x}\rVert^2$, which is
    exact only to a few float32 epsilons ($\approx10^{-7}$) times the squared norms. Two neighbors
    closer than that are a **tie**, and either order is right. So the check is: for every building
    of the held-out campus, FAISS's five neighbors must be, rank by rank, as close as NumPy's,
    up to that rounding. Count how many queries come back in a different order.
    """)
    return


@app.cell
def _(attempt_rows, np, pf, shape, te0, tr0):
    import faiss as _faiss

    _S = shape(pf.P16)
    _sq = (_S**2).sum(axis=1)
    _d2 = _sq[te0][:, None] - 2 * _S[te0] @ _S[tr0].T + _sq[tr0][None, :]  # float64
    assert _d2.shape == (len(te0), len(tr0))
    _nn_np = np.argsort(_d2, axis=1, kind="stable")[:, :5]
    _index = _faiss.IndexFlatL2(24)
    _index.add(np.ascontiguousarray(_S[tr0].astype(np.float32)))
    _D_fa, _nn_fa = _index.search(np.ascontiguousarray(_S[te0].astype(np.float32)), 5)
    # Exact (float64) distances of each method's neighbors, and FAISS's float32 rounding bound.
    _d_np = np.take_along_axis(_d2, _nn_np, axis=1)
    _d_fa = np.take_along_axis(_d2, _nn_fa, axis=1)
    _eps = float(np.finfo(np.float32).eps)
    _bound_fa = 4 * _eps * (_sq[te0][:, None] + _sq[tr0][_nn_fa])
    _bound_np = 4 * _eps * (_sq[te0][:, None] + _sq[tr0][_nn_np])
    assert np.all(np.abs(_D_fa - _d_fa) <= _bound_fa)  # FAISS's distances: float32-exact
    _same = (_nn_np == _nn_fa).all(axis=1)
    _ties = (np.abs(_d_fa - _d_np) <= _bound_fa + _bound_np).all(axis=1)
    _q_pos = int(np.flatnonzero(te0 == attempt_rows["Q"])[0])
    # All 136 held-out buildings get the same top 5 up to ties (one comes back with two
    # near-tied neighbors swapped); Q's two nearest are C and D.
    assert _ties.all() and int((~_same).sum()) == 1
    assert list(tr0[_nn_np[_q_pos, :2]]) == list(attempt_rows[["C", "D"]])
    {
        "queries": len(te0),
        "database": len(tr0),
        "same top 5, same order (%)": round(100 * float(_same.mean()), 1),
        "same top 5 up to ties (%)": round(100 * float(_ties.mean()), 1),
        "Q's nearest two": list(pf.name[tr0[_nn_np[_q_pos, :2]]]),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked example: Q through the encoder and decoder

    Fit three modes on the training shapes, then encode Q: divide by its mean (699 kW), subtract
    the mean shape, and project. Decode, and compare with Q's own shape.
    """)
    return


@app.cell
def _(R, attempt_rows, orient, pca_svd, pf, rms, shape, tr0):
    _S = shape(pf.P16)
    _q = int(attempt_rows["Q"])
    _mu, _V, _, _ = pca_svd(_S[tr0], R)
    _V = orient(_V)  # all three modes, so z does not depend on the machine
    _z = (_S[_q] - _mu) @ _V
    _p_hat = _mu + _V @ _z
    _err, _dev = rms(_S[_q], _p_hat), rms(_S[_q], _mu)
    _offices = pf.use[tr0] == "Office"
    _flatter = float((((_S[tr0] - _mu) @ _V)[_offices, 0] > _z[0]).mean())
    # Slide "The encoder keeps shape": 699 kW; z = (-0.62, -0.38, -0.01); error 2.2% of the
    # mean; 98% of Q's departure from the mean shape captured; swing smaller than in 80% of the
    # 157 training offices.
    assert round(float(pf.P16[_q].mean())) == 699
    assert [round(float(_v), 2) for _v in _z] == [-0.62, -0.38, -0.01]
    assert round(100 * _err, 1) == 2.2 and round(100 * (1 - _err**2 / _dev**2)) == 98
    assert round(100 * _flatter) == 80 and int(_offices.sum()) == 157
    {
        "size (kW)": round(float(pf.P16[_q].mean())),
        "z": _z.round(2).tolist(),
        "error (% of mean)": round(100 * _err, 1),
        "swing smaller than in (% of offices)": round(100 * _flatter),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Bracket the error.** An error means little on its own. Over 30 splits that each hold out a
    quarter of the sites, fit on 2016 shapes at the training sites and rebuild the 2017 shapes at
    the held-out sites.

    - **Above:** the mean shape ($r=0$), the naive baseline.
    - **Below (rough):** the year-to-year change, each building's 2016 shape against its own 2017
      shape. A building's typical weekday changes from one year to the next; a summary closer
      than that describes this year's details, not the building. It is a forecast error, while
      the modes' error is a reconstruction (the 2017 shape projected on the modes), so it is a
      rough marker, not a strict floor.

    Each split's error is one RMS over its held-out buildings and hours; the numbers are the mean
    over the 30 splits. The typical days here keep the best of 10 $k$-means++ starts in each
    split.

    **Modify.** Change the list of `_ks` (typical days) in the next cell, or the largest $r$, and
    rerun. Where do 8 typical days land?
    """)
    return


@app.cell
def _(KMeans, PALETTE, SEED, np, pca_svd, pf, plt, rms, shape, site_splits):
    _S16, _S17 = shape(pf.P16), shape(pf.P17)
    _rs = list(range(9))
    _ks = [4, 16]
    _err = {_r: [] for _r in _rs}
    _km = {_k: [] for _k in _ks}
    _floor = []
    for _tr, _te in site_splits(pf.site):
        _mu, _V, _, _ = pca_svd(_S16[_tr], 24)
        for _r in _rs:
            _Zt = (_S17[_te] - _mu) @ _V[:, :_r]
            _err[_r].append(rms(_S17[_te], _mu + _Zt @ _V[:, :_r].T))
        _floor.append(rms(_S17[_te], _S16[_te]))
        for _k in _ks:
            _C = (
                KMeans(
                    n_clusters=_k,
                    init="k-means++",
                    n_init=10,
                    algorithm="lloyd",
                    random_state=SEED,
                )
                .fit(_S16[_tr])
                .cluster_centers_
            )
            _d2 = ((_S17[_te][:, None, :] - _C[None]) ** 2).sum(-1).min(axis=1)
            _km[_k].append(float(np.sqrt(_d2.mean() / 24)))
    _mean = [round(100 * float(np.mean(_err[_r])), 1) for _r in _rs]
    _fl = round(100 * float(np.mean(_floor)), 1)
    # One RMS pooled over all 1,159 buildings, instead of the mean of 30 per-split values: 7.4%.
    assert round(100 * rms(_S17, _S16), 1) == 7.4
    # Slide "Four modes reach the yearly change": 22.1 / 13.3 / 9.9 / 7.5 / 6.0; floor 6.9 ± 2.4;
    # 4 typical days 16.0, 16 typical days 12.2.
    assert _mean[:5] == [22.1, 13.3, 9.9, 7.5, 6.0] and _mean[8] == 3.5
    assert (_fl, round(100 * float(np.std(_floor)), 1)) == (6.9, 2.4)
    assert [round(100 * float(np.mean(_km[_k])), 1) for _k in _ks] == [16.0, 12.2]
    # Paired, split by split: 4 modes beat the year-to-year change in 21 of 30 splits, 3 in 12.
    _below = {_r: int((np.array(_err[_r]) < np.array(_floor)).sum()) for _r in _rs}
    assert (_below[4], _below[3]) == (21, 12)

    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.errorbar(
        _rs,
        _mean,
        yerr=[100 * np.std(_err[_r]) for _r in _rs],
        marker="o",
        color=PALETTE[0],
        capsize=3,
        label="r modes",
    )
    _ax.axhline(_mean[0], ls=":", color="#595959", label="mean shape (r = 0)")
    _ax.axhline(_fl, ls="--", color="#595959", label="year-to-year change (rough)")
    for _j, _k in enumerate(_ks):
        _ax.axhline(
            100 * np.mean(_km[_k]), ls="-.", color=PALETTE[2 + _j], label=f"{_k} typical days"
        )
    _ax.set(xlabel="modes kept, r", ylabel="2017 error (% of daily mean)", ylim=(0, 30))
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Four modes reach the yearly change.** The mean shape misses by 22.1% of the daily mean;
    3 modes by 7.5%, 4 modes by 6.0%, inside a building's own year-to-year change (6.9%, rough)
    on average and in 21 of the 30 splits.
    Typical days are a much coarser summary: 4 of them miss by 16.0%, and even 16 (12.2%) do worse
    than 2 modes (9.9%). A typical day is one integer per building, good for talking to people
    and poor for measuring.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    Each test below probes the assumption from §3, *buildings that run alike have close daily
    shapes*, and the "few typical days or modes describe them all" that comes with it.

    **Predict before we test.** Write down three answers.

    1. **Size.** Run PCA on raw kW, or standardize each hour first as in Lecture 2. What does
       mode 1 capture, and how much of the variance?
    2. **Days.** Retrieve single days, with every other building-day in the database. How often is
       the nearest day the same building?
    3. **Typical days.** $k$-means with $K=4$ on the 1,023 training buildings. About how many
       buildings in each typical day?

    Commit in writing; each subsection reveals one answer.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Size, the decisive check: PCA on kilowatts

    Fit PCA three ways on the same 1,023 training buildings: raw kW, each hour standardized
    (`StandardScaler`, as in Lecture 2), and shapes. Compare mode 1 and its score with each
    building's mean load.
    """)
    return


@app.cell
def _(
    HOURS,
    KW_SIGN_RULES,
    MARKERS,
    PALETTE,
    StandardScaler,
    np,
    orient,
    pca_svd,
    pf,
    plt,
    shape,
    tr0,
):
    _X = pf.P16[tr0]
    _size = _X.mean(axis=1)
    _rows, _loads = {}, {}
    for _name, _M in (
        ("raw kW", _X),
        ("hours standardized", StandardScaler().fit_transform(_X)),
        ("shape", shape(_X)),
    ):
        _mu, _V, _lam, _ = pca_svd(_M, 24)
        _v1 = (orient(_V) if _name == "shape" else orient(_V, KW_SIGN_RULES))[:, 0]
        _z1 = (_M - _mu) @ _v1
        _loads[_name] = _v1
        _rows[_name] = {
            "mode 1 (% of variance)": round(100 * float(_lam[0] / _lam.sum()), 1),
            "corr(score, mean kW)": round(float(np.corrcoef(_z1, _size)[0, 1]), 5),
        }
    # Slide "Unscaled, the first mode is size": 98.0% (r = 0.99995); standardized 97.9%
    # (r = 0.99996); shapes 65.2%.
    assert _rows["raw kW"] == {"mode 1 (% of variance)": 98.0, "corr(score, mean kW)": 0.99995}
    assert _rows["hours standardized"]["mode 1 (% of variance)"] == 97.9
    assert _rows["hours standardized"]["corr(score, mean kW)"] == 0.99996
    assert _rows["shape"]["mode 1 (% of variance)"] == 65.2

    _fig, _ax = plt.subplots(figsize=(5, 3))
    for _j, (_name, _v1) in enumerate(_loads.items()):
        _ax.plot(HOURS, _v1, color=PALETTE[_j], marker=MARKERS[_j], markevery=6, label=_name)
    _ax.axhline(0, color="#595959", lw=0.8)
    _ax.set(xlabel="hour", ylabel="mode-1 loading")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    (_rows, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 1: mode 1 is size.** On raw kW it holds 98.0% of the variance, and its score tracks
    each building's mean load with $r=0.99995$; its loading is almost the mean profile, flat across
    the hours. Standardizing each hour, the Lecture 2 fix, changes nothing (97.9%,
    $r=0.99996$): it rescales columns, while size multiplies a whole row, $\mathbf{x}=\bar{x}\,
    \mathbf{p}$. Dividing each row by its own mean removes it. The assumption fails completely in
    kW: "close" means "the same size", and a screen built on the raw mode would visit buildings in
    order of size (§6 prices that: 50% of the excess with 10 visits, against 90%).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Days: the same building in the database

    Now retrieve single days. Building-days: every 2016 weekday of an eligible building with all
    24 hours metered and positive, divided by its own mean. Queries: 2,000 random building-days of
    the held-out campus. Two databases: every other building-day (a random split of days), or only
    the days of the 14 training campuses.
    """)
    return


@app.cell
def _(PALETTE, SEED, dates, kwh, names, np, pd, pf, plt, te0, tr0):
    import faiss as _faiss

    _rows = pd.Index(names).get_indexer(pf.name)
    assert (_rows >= 0).all(), "an eligible building is missing from the hourly file"
    _keep = np.asarray((dates.dayofweek < 5) & (dates.year == 2016))
    _days = kwh[_rows][:, _keep, :]
    _ok = ~np.isnan(_days).any(axis=2) & (np.nan_to_num(_days, nan=0.0).min(axis=2) > 0)
    _b, _d = np.nonzero(_ok)
    _X = _days[_b, _d].astype(np.float32)
    _X /= _X.mean(axis=1, keepdims=True)
    _q = np.random.default_rng(SEED).choice(
        np.flatnonzero(np.isin(_b, te0)), size=2000, replace=False
    )
    _dbs = {
        "random split of days": np.setdiff1d(np.arange(len(_X)), _q),
        "sites held out": np.flatnonzero(np.isin(_b, tr0)),
    }
    _res = {}
    for _name, _db in _dbs.items():
        _index = _faiss.IndexFlatL2(24)
        _index.add(np.ascontiguousarray(_X[_db]))
        _, _nn = _index.search(np.ascontiguousarray(_X[_q]), 1)
        _nb, _qb = _b[_db][_nn[:, 0]], _b[_q]
        _res[_name] = {
            "same building": round(100 * float((_nb == _qb).mean()), 1),
            "same site": round(100 * float((pf.site[_nb] == pf.site[_qb]).mean()), 1),
            "same use": round(100 * float((pf.use[_nb] == pf.use[_qb]).mean()), 1),
            "database days": len(_db),
        }
    _share = pd.Series(pf.use[_b[_dbs["sites held out"]]]).value_counts(normalize=True)
    _chance = round(100 * float(np.mean([_share.get(_u, 0.0) for _u in pf.use[_b[_q]]])), 1)
    # Slide "The nearest day is the same building": 288,999 other days; random 56 / 66 / 68%;
    # sites held out: same use 25%; chance 16%.
    assert len(_X) == 290999 and _res["random split of days"]["database days"] == 288999
    assert [
        _res["random split of days"][_c] for _c in ("same building", "same site", "same use")
    ] == [55.9, 66.3, 68.2]
    assert _res["sites held out"]["same use"] == 25.1 and _chance == 16.0

    _fig, _ax = plt.subplots(figsize=(5, 3))
    _cats = ["same building", "same site", "same use"]
    for _j, _name in enumerate(_dbs):
        _ax.bar(
            np.arange(3) + (_j - 0.5) * 0.38,
            [_res[_name][_c] for _c in _cats],
            width=0.36,
            color=PALETTE[_j],
            hatch="" if _j == 0 else "//",
            label=_name,
        )
    _ax.axhline(_chance, ls=":", color="#595959", label="chance, same use")
    _ax.set_xticks(np.arange(3), _cats)
    _ax.set(ylabel="nearest day is ... (%)", ylim=(0, 100))
    _ax.legend(fontsize=8)
    (pd.DataFrame(_res).T, _fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 2: a random split measures how well a building recognizes itself.** With every other
    day in the database, the nearest day is the same building 56% of the time and the same site
    66%, and "same use" reaches 68%. Hold out the campus and same use falls to 25%, against 16% by
    chance. A building's other days are near-copies of today, as a bridge's other inspections were
    in Lecture 3: leakage (Kaufman et al., 2012), and the reason for structured validation
    (Roberts et al., 2017). Use labels are coarse ("Education" spans classrooms and
    laboratories), which may be part of why 25% is low; we have not measured it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Typical days: $k$-means with $K=4$

    The best of 20 $k$-means++ starts on the 1,023 training shapes. Name each typical day by its
    shape and count its buildings.
    """)
    return


@app.cell
def _(HOURS, K, KMeans, MARKERS, PALETTE, SEED, attempt_rows, np, pf, plt, shape, tr0):
    _S = shape(pf.P16[tr0])
    _km = KMeans(
        n_clusters=K, init="k-means++", n_init=20, algorithm="lloyd", random_state=SEED
    ).fit(_S)
    typical_centers = _km.cluster_centers_
    _counts = np.bincount(_km.labels_, minlength=K)
    # Name the centers: the evening peak, then by night/day ratio (strong day, moderate, always on).
    _rho = typical_centers[:, 0:4].mean(1) / typical_centers[:, 12:16].mean(1)
    _evening = int(np.argmax(typical_centers[:, 18:22].mean(1)))
    _rest = sorted([_k for _k in range(K) if _k != _evening], key=lambda _k: _rho[_k])
    _names = [""] * K
    _names[_evening] = "evening peak"
    for _k, _n in zip(_rest, ["strong day", "moderate day", "always on"], strict=True):
        _names[_k] = _n
    _small = int(np.argmin(_counts))
    _members = tr0[_km.labels_ == _small]
    _peaks = [round(float(shape(pf.P16[[_i]])[0].max()), 1) for _i in _members]
    _q_shape = shape(pf.P16[[attempt_rows["Q"]]])[0]
    _q_center = _names[int(((typical_centers - _q_shape) ** 2).sum(1).argmin())]
    # Slide "Four buildings claim a typical day": 518, 394, 107, and 4 buildings; they peak near
    # 20 h, up to 8.6 times their daily mean (center 4.7); three assembly venues and a
    # warehouse. Diagram: Q's nearest typical day is "always on".
    assert sorted(_counts.tolist(), reverse=True) == [518, 394, 107, 4]
    assert _names[_small] == "evening peak" and int(np.argmax(typical_centers[_small])) == 20
    assert max(_peaks) == 8.6 and round(float(typical_centers[_small].max()), 1) == 4.7
    assert sorted(pf.use[_members]) == ["Entertainment/public assembly"] * 3 + ["Warehouse/storage"]
    assert _q_center == "always on"
    # Its alt text: always on 0.89-1.09; moderate day 0.69 at night to 1.31 at noon; strong day
    # 0.43 to 1.87; evening peak 0.37 at its lowest.
    _range = {
        _names[_k]: (
            round(float(typical_centers[_k].min()), 2),
            round(float(typical_centers[_k].max()), 2),
        )
        for _k in range(K)
    }
    assert _range["always on"] == (0.89, 1.09) and _range["moderate day"] == (0.69, 1.31)
    assert _range["strong day"] == (0.43, 1.87) and _range["evening peak"][0] == 0.37

    _fig, _ax = plt.subplots(figsize=(5.5, 3.2))
    for _j, _k in enumerate(np.argsort(-_counts)):
        _ax.plot(
            HOURS,
            typical_centers[_k],
            color=PALETTE[_j],
            marker=MARKERS[_j],
            markevery=4,
            label=f"{_names[_k]} ({_counts[_k]})",
        )
    _ax.set(xlabel="hour of the day", ylabel="load / daily mean")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    (list(pf.name[_members]), _fig)
    return (typical_centers,)


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal 3: four buildings claim a typical day.** The groups are 518 (always on), 394 (moderate
    day), 107 (strong day), and **4**: three assembly venues and a warehouse that peak near 20 h,
    up to 8.6 times their daily mean. Squared error charges far points heavily, so $k$-means spends
    a whole center on them; Forsyth (§8.2.6) describes this habit of single-element or
    straggling clusters. A typical day is not a building type: read the counts, flag outliers,
    then cluster. Remedies: cap or remove outliers first, more clusters, a "junk" cluster, or
    $k$-medoids, whose centers are real buildings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    Back to the energy manager on Q's campus, 67 offices. **Excess** night load is the 0–4 h load
    minus the office yardstick (0.66, the median night/day ratio of the offices on the other 14
    campuses) times the 12–16 h load. Rank the offices on 2016 data three ways, then count how much
    of the **2017** excess each ranking would find with a given number of visits.
    """)
    return


@app.cell
def _(
    DAY,
    MARKERS,
    NIGHT,
    NIGHT_HOURS,
    PALETTE,
    SEED,
    attempt_rows,
    blocks,
    dates,
    night_day,
    np,
    office_ref,
    pf,
    plt,
):
    _q = int(attempt_rows["Q"])
    _o = np.flatnonzero((pf.site == pf.site[_q]) & (pf.use == "Office"))
    _b16, _b17 = blocks(pf.P16[_o]), blocks(pf.P17[_o])
    _ex16 = _b16[:, NIGHT] - office_ref * _b16[:, DAY]
    _ex17_raw = _b17[:, NIGHT] - office_ref * _b17[:, DAY]
    _ex17 = np.clip(_ex17_raw, 0, None)
    _scores = {
        "size × shape": _ex16,
        "shape only": night_day(pf.P16[_o]),
        "size only": pf.P16[_o].mean(axis=1),
    }
    _curves = {
        _n: np.concatenate([[0.0], np.cumsum(_ex17[np.argsort(-_s, kind="stable")]) / _ex17.sum()])
        for _n, _s in _scores.items()
    }
    _at10 = {_n: round(100 * float(_c[10])) for _n, _c in _curves.items()}
    _at5 = {_n: round(100 * float(_c[5])) for _n, _c in _curves.items()}
    _top16, _top17 = set(np.argsort(-_ex16)[:10]), set(np.argsort(-_ex17_raw)[:10])
    _q_ex = float(blocks(pf.P16[[_q]])[0, NIGHT] - office_ref * blocks(pf.P16[[_q]])[0, DAY])
    _moved = sorted(
        zip(np.round(_ex16), np.round(_ex17_raw), strict=True), key=lambda _t: -abs(_t[1] - _t[0])
    )
    # Slide "Shape says whether; size says how much": 67 offices; 10 visits find 90 / 75 / 50%,
    # random 15%; 5 visits 63 / 60 / 9%; 9 of the top 10 repeat; Q 214 kW above, about
    # 214 x 8 x 261 / 1000 = 447 MWh a year (8 night hours, the 261 weekdays of 2016, counted
    # from the dates); one office 39 -> 100 kW, one -29 -> -97.
    assert len(_o) == 67 and _at10 == {"size × shape": 90, "shape only": 75, "size only": 50}
    assert _at5 == {"size × shape": 63, "shape only": 60, "size only": 9}
    assert round(100 * 10 / len(_o)) == 15 and len(_top16 & _top17) == 9
    _weekdays = int(((dates.dayofweek < 5) & (dates.year == 2016)).sum())
    assert round(_q_ex) == 214 and _weekdays == 261
    assert round(round(_q_ex) * NIGHT_HOURS * _weekdays / 1000) == 447  # from the shown 214 kW
    assert {(float(_a), float(_b)) for _a, _b in _moved[:2]} == {(39.0, 100.0), (-29.0, -97.0)}

    # One campus of 67 offices: how much would another draw of offices move these shares?
    # Resample the offices with replacement 1,000 times; rank each draw by each score (paired).
    _idx = np.random.default_rng((SEED, 4)).integers(0, len(_o), size=(1000, len(_o)))
    _e17 = _ex17[_idx]
    _found = {
        _n: np.take_along_axis(
            _e17, np.argsort(-_s[_idx], axis=1, kind="stable")[:, :10], axis=1
        ).sum(axis=1)
        / _e17.sum(axis=1)
        for _n, _s in _scores.items()
    }

    def _ci(_v):
        return [round(100 * float(np.percentile(_v, _p))) for _p in (2.5, 97.5)]

    _boot = {_n: _ci(_v) for _n, _v in _found.items()}
    _lead_shape = _found["size × shape"] - _found["shape only"]
    _lead_size = _found["size × shape"] - _found["size only"]
    # Slide notes: 95% intervals over the offices, size × shape 73-97%, shape alone 56-95%, size
    # alone 14-76%; lead over shape alone, median +8 points, 0 to +28; over size alone, 15 to 76.
    assert _boot == {"size × shape": [73, 97], "shape only": [56, 95], "size only": [14, 76]}
    assert round(100 * float(np.median(_lead_shape))) == 8
    assert _ci(_lead_shape) == [0, 28] and _ci(_lead_size) == [15, 76]

    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    for _j, (_n, _c) in enumerate(_curves.items()):
        _ax.plot(
            np.arange(16),
            100 * _c[:16],
            color=PALETTE[_j],
            marker=MARKERS[_j],
            markevery=5,
            label=_n,
        )
    _ax.plot(np.arange(16), 100 * np.arange(16) / len(_o), ":", color="#595959", label="random")
    _ax.set(xlabel="visits, ranked on 2016 data", ylabel="2017 excess covered (%)", ylim=(0, 102))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    (
        {
            "10 visits (%)": _at10,
            "5 visits (%)": _at5,
            "10 visits, 95% interval over the offices (%)": _boot,
            "Q excess (kW)": round(_q_ex),
        },
        _fig,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Shape says whether; size says how much.** Ranked on 2016, ten visits find 90% of 2017's
    excess by size × shape, 75% by shape alone, 50% by size alone, and 15% in random order. This
    is one campus of 67 offices: resampling its offices 1,000 times gives 95% intervals of
    73–97% for size × shape, 56–95% for shape alone, and 14–76% for size alone. Paired draw by
    draw, size × shape leads size alone by 15 to 76 points; its lead over shape alone (median +8
    points, interval 0 to +28) touches zero, so on this one campus it is at least as good as shape
    alone, not shown to be better. The ranking is fairly stable (9 of the top 10 repeat), but individual offices move: one went from
    39 to 100 kW of excess, another from −29 to −97 kW. Re-rank every year. Q itself is 214 kW above
    an office's typical night; held for the 8 hours from midnight on the 261 weekdays of 2016,
    that is $214\times8\times261$ kWh, about 447 MWh a year (rough). Excess above a yardstick is
    not proven waste.

    ### When not to use the profile screen: it can choose a visit; it cannot diagnose

    - **Flat can be normal:** laboratories, data rooms, hospitals, and residences run at night.
      Use the right yardstick; "Education" includes laboratories. Q uses 71 W/m², more than twice
      its peers A (28) and B (31): a server room or a laboratory would explain its flat day.
    - **Flat can be the meter:** 63 meters gave one value per day, spread over the hours (§1).
      Check the time resolution before the shape.
    - **Shapes move with the season** (next cell).
    - **Not a diagnosis:** a profile says when energy is used, not by what. The visit finds out.
    """)
    return


@app.cell
def _(attempt_rows, dates, kwh, names, np, pd, pf, rms, shape, typical_centers, typical_weekday):
    _rows = pd.Index(names).get_indexer(pf.name)
    assert (_rows >= 0).all(), "an eligible building is missing from the hourly file"
    _wk, _year = np.asarray(dates.dayofweek < 5), np.asarray(dates.year)
    _month = np.asarray(dates.month)
    # Typical weekdays of each season (3 months) in each year, and whether each has enough days.
    _win, _enough = {}, {}
    for _y in (2016, 2017):
        for _name, _months in (("summer", [6, 7, 8]), ("winter", [1, 2, 12])):
            _P, _n = typical_weekday(kwh[_rows], _wk & (_year == _y) & np.isin(_month, _months))
            _win[_y, _name] = _P
            _enough[_y, _name] = (_n >= 40) & (_P.min(axis=1) > 0)
    _ok = _enough[2016, "summer"] & _enough[2016, "winter"]
    _Ss, _Sw = shape(_win[2016, "summer"][_ok]), shape(_win[2016, "winter"][_ok])
    _ls = ((_Ss[:, None, :] - typical_centers[None]) ** 2).sum(-1).argmin(axis=1)
    _lw = ((_Sw[:, None, :] - typical_centers[None]) ** 2).sum(-1).argmin(axis=1)
    _season = {
        "buildings": int(_ok.sum()),
        "change typical day, summer to winter (%)": round(100 * float((_ls != _lw).mean())),
        "summer vs winter shape (%)": round(100 * rms(_Ss, _Sw), 1),
        "2016 vs 2017 shape (%)": round(100 * rms(shape(pf.P16[_ok]), shape(pf.P17[_ok])), 1),
    }
    _wpm2 = (
        1000
        * pf.P16[attempt_rows[["Q", "A", "B"]]].mean(axis=1)
        / pf.sqm[attempt_rows[["Q", "A", "B"]]]
    ).round()
    # Slide "Screen with meters; confirm on site": 17% change typical day (1,144 buildings);
    # notes: Q 71 W/m², A 28, B 31.
    assert list(_season.values()) == [1144, 17, 16.1, 7.4]
    assert _wpm2.tolist() == [71.0, 28.0, 31.0]

    # 7.4% (2016 against 2017) compares 12-month windows, not 3-month ones. Like for like, on the
    # buildings with enough days in all four windows: the same season a year apart.
    _both = _ok & _enough[2017, "summer"] & _enough[2017, "winter"]

    def _sh(_y, _name):
        return shape(_win[_y, _name][_both])

    _like = {
        "buildings": int(_both.sum()),
        "summer vs winter 2016 (%)": round(100 * rms(_sh(2016, "summer"), _sh(2016, "winter")), 1),
        "summer 2016 vs 2017 (%)": round(100 * rms(_sh(2016, "summer"), _sh(2017, "summer")), 1),
        "winter 2016 vs 2017 (%)": round(100 * rms(_sh(2016, "winter"), _sh(2017, "winter")), 1),
    }
    # Slide notes: 16.1% against 12.8% (summer) and 9.2% (winter) a year apart, 1,138 buildings.
    assert list(_like.values()) == [1138, 16.1, 12.8, 9.2]
    (_season, _like)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Shapes move with the season.** 17% of buildings change typical day between summer
    (June–August) and winter (January, February, December) weekdays of 2016. A building's summer
    and winter shapes differ by 16.1% of its mean. The yardstick is the same season a year apart,
    windows of the same length: 12.8% for summer 2016 against summer 2017, and 9.2% for winter
    (1,138 buildings with enough weekdays in all four). The full-year change (7.4%) compares
    12-month windows, not 3-month ones, so it is not the yardstick. Compare buildings with the same
    season's typical days.

    ### Impact: a load profile is a fingerprint

    Retrieval doubles as re-identification. Query each building's 2017 weekday shape against the
    2016 shapes of all 1,159 buildings: how often is the nearest one itself?
    """)
    return


@app.cell
def _(np, pf, shape):
    import faiss as _faiss

    _out = {}
    for _name, (_a, _b) in {
        "shape": (shape(pf.P17), shape(pf.P16)),
        "raw kW": (pf.P17, pf.P16),
    }.items():
        _index = _faiss.IndexFlatL2(24)
        _index.add(np.ascontiguousarray(_b.astype(np.float32)))
        _, _nn = _index.search(np.ascontiguousarray(_a.astype(np.float32)), 5)
        _own = np.arange(len(_a))[:, None]
        _out[_name] = {
            "top 1 (%)": round(100 * float((_nn[:, 0] == _own[:, 0]).mean())),
            "top 5 (%)": round(100 * float((_nn == _own).any(axis=1).mean())),
        }
    # Slide "A load profile is a fingerprint": 64% (81% in the top 5) among 1,159; raw kW 50%.
    assert (
        _out
        == {
            "shape": {"top 1 (%)": 64, "top 5 (%)": 81},
            "raw kW": {"top 1 (%)": 50, "top 5 (%)": 65},
        }
        and len(pf.P16) == 1159
    )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    A building's 2017 shape finds its own 2016 shape **64%** of the time among 1,159 buildings
    (81% in the top 5). Code names do not anonymize a meter: BDG2 already replaces names and gives
    no location for some sites (Miller et al., 2020), yet a profile can be matched across releases.
    Hourly profiles show when people arrive, leave, and sleep (Molina-Markham et al., 2010,
    *BuildSys '10*). For residences that is personal data: share aggregates, coarser time steps, or data with
    consent. Lecture 27 returns to who is measured and who decides.

    ### Vote: which building is next?

    Q got this month's visit. The screen lists three more on Q's campus, by night load above the
    office yardstick (next cell):

    - **A.** Education, 3,494 kW, nights 80% of afternoon, 540 kW above; 131 W/m².
    - **B.** Office, 1,307 kW, nights 74%, 123 kW above; 42 W/m².
    - **C.** Residence, 362 kW, nights 94%, 103 kW above; 21 W/m².

    Vote alone. Discuss with a neighbor for two minutes. Vote again. Then open the explanation.
    """)
    return


@app.cell
def _(DAY, NIGHT, attempt_rows, blocks, night_day, np, office_ref, pd, pf):
    _q = int(attempt_rows["Q"])
    _b = blocks(pf.P16)
    _ex = _b[:, NIGHT] - office_ref * _b[:, DAY]
    _b17 = blocks(pf.P17)
    _ex17 = _b17[:, NIGHT] - office_ref * _b17[:, DAY]
    _campus = np.flatnonzero((pf.site == pf.site[_q]) & (np.arange(len(pf.site)) != _q))
    _picks = []
    for _allowed in (None, "Office", "Lodging/residential"):
        _pool = _campus if _allowed is None else _campus[pf.use[_campus] == _allowed]
        _picks.append(int(_pool[np.argmax(_ex[_pool])]))
    _kw = pf.P16[_picks].mean(axis=1)
    _cards = pd.DataFrame(
        {
            "building": pf.name[_picks],
            "use": pf.use[_picks],
            "kW": _kw.round(),
            "nights (% of afternoon)": (100 * night_day(pf.P16[_picks])).round(),
            "kW above yardstick": _ex[_picks].round(),
            "kW above yardstick, 2017": _ex17[_picks].round(),
            "W/m²": (1000 * _kw / pf.sqm[_picks]).round(),
        },
        index=list("ABC"),
    )
    # Slide "Vote: which building is next?"
    assert _cards["kW"].tolist() == [3494, 1307, 362]
    assert _cards["nights (% of afternoon)"].tolist() == [80, 74, 94]
    assert _cards["kW above yardstick"].tolist() == [540, 123, 103]
    assert _cards["W/m²"].tolist() == [131, 42, 21]
    # Explanation: B's excess fell from 123 kW (2016) to 71 kW (2017); it did not vanish.
    assert _cards.loc["B", "kW above yardstick, 2017"] == 71
    _cards
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Explanation": mo.md(
                r"""
                **B** is the after-hours visit: the office yardstick fits an office, and 123 kW
                above a typical office night on a 1.3 MW building is the largest real candidate.
                **A** has the most kilowatts above the yardstick, but the yardstick is wrong for
                it: 131 W/m², over four times offices A (28) and B (31) of the attempt, says
                something runs there day and night, by design or not. It may deserve a visit of a
                different kind, with a different yardstick. **C** is a residence: people are home
                at night.

                What would change the answer: A's records say classrooms only; B's excess
                vanishing (in 2017 it fell from 123 to 71 kW, but stayed); a cheaper visit.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where these methods live today: embeddings and vector search

    PCA's $\mathbf{z}$ is an **embedding**: 24 numbers become 3 that keep what varies, and
    retrieval searches the embeddings. Foundation models learn embeddings of text, images, and
    time series, and the search is the same nearest-neighbor step, run at billion scale on GPUs
    (Johnson, Douze & Jégou, 2021, *IEEE Transactions on Big Data* 7(3), the FAISS paper).
    **Retrieval-augmented generation** (RAG; Lewis et al., 2020, NeurIPS) retrieves the passages
    nearest a question and lets a language model answer from them; Lecture 19 builds one over
    public specifications and evaluates its retrieval. For buildings, BuildingsBench (Emami, Sahu &
    Graf, 2023, NeurIPS Datasets and Benchmarks) pretrains load forecasters on 900,000 simulated
    buildings and tests them on real meters, BDG2 among them; Lecture 20 covers pretrained
    forecasters for time series.

    **k-means builds the index.** Vector-search indexes such as FAISS's IVF split the vectors into
    cells with $k$-means and search only the cells nearest the query. e-means (Bessa et al., 2026,
    FAIR at Meta, arXiv:2611.09501) keeps the objective $J$ of §3 and changes the optimizer: each
    center is stored as a sum $S_k$ over a count $N_k$, both updated by an exponential moving
    average over mini-batches, $N_k \leftarrow \gamma N_k + (1-\gamma)\,n_k$ and
    $S_k \leftarrow \gamma S_k + (1-\gamma)\,s_k$, with $\mathbf{c}_k = S_k / N_k$; the step
    $1-\gamma$ decays on a cosine schedule, empty clusters are revived by splitting large ones,
    and step and batch sizes come from scaling laws fitted on embedding data. It beats Lloyd's
    loss after 25 passes in 1.4 to 2.6 passes, and partitions 10 billion image embeddings into
    16 million clusters in 2.5 passes, below Lloyd's loss after 25 (the paper's abstract). The authors note that their scaling laws are not tuned for
    low-dimensional, small datasets, which our profiles are; the exercise below tests it.
    Code: `faiss/contrib/e_means` (MIT license).

    Today's failures carry over unchanged: an embedding can encode size, or any nuisance, instead
    of what matters (§5, reveal 1), and a random split of documents or days flatters retrieval
    (reveal 2).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exercise: does e-means help on our profiles?

    e-means was built for billions of embeddings and thousands of centers. Ours are 1,023
    training profiles and four typical days. **Predict first, in writing:**

    1. After 5 passes over the data, will e-means reach a lower training error than Lloyd run
       until no assignment changes?
    2. Will it keep a cluster for the four evening-peak buildings of reveal 3?
    3. On held-out campuses, whose centers do best: k-means++ (best of 20), Lloyd, or e-means?

    **Step 1: the settings.** The paper sets the batch size $B$ and the peak step $\alpha_0$ by
    scaling laws fitted on embedding data, so nothing is tuned by hand (its Eqs. 6–7):
    $B = b_c\,k^{b_k} n^{b_n} T^{b_T}$, rounded to a power of two and capped near $n/2$, and
    $\alpha_0 = \min(a_c\,k^{a_k} B^{a_B} n^{a_n} T^{a_T},\,1)$, for $k$ clusters, $n$ points, and
    $T$ passes. The constants are those shipped in the reference code (the paper's Table 2).
    """)
    return


@app.cell
def _(K, np, pf, shape, tr0):
    def emeans_settings(k: int, n: int, T: int) -> tuple[int, float]:
        """Batch size and peak step of e-means from the paper's scaling laws (Eqs. 6-7).

        Args:
            k: Number of clusters.
            n: Number of training points.
            T: Number of passes (epochs).

        Returns:
            The batch size ``B`` and the peak step ``alpha0``.
        """

        def pow2(v: float) -> int:
            lo = 2 ** int(np.floor(np.log2(v)))
            return lo if v / lo < 2 * lo / v else 2 * lo  # nearest power of two

        bc, bk, bn, bT = 0.035134, 0.134749, 0.876473, 0.813344  # batch law (d > 16)
        B = int(min(max(min(pow2(bc * k**bk * n**bn * T**bT), pow2(n / 2)), 1), n))
        ac, ak, aB, an, aT = 0.8244, -0.4617, 0.9003, -0.4498, -0.1057  # step law
        alpha0 = float(min(max(ac * k**ak * B**aB * n**an * T**aT, 1e-4), 1.0))
        return B, alpha0

    _B, _a0 = emeans_settings(K, len(tr0), 5)
    # Our size: 1,023 profiles, k = 4, T = 5 passes -> batches of 64 and a peak step of 0.686.
    assert len(shape(pf.P16[tr0])) == 1023 and _B == 64 and round(_a0, 3) == 0.686
    {"n": len(tr0), "B": _B, "alpha0": round(_a0, 3), "batches per pass": -(-len(tr0) // _B)}
    return (emeans_settings,)


@app.cell
def _(mo):
    mo.md(r"""
    **Step 2: the algorithm** (the paper's Algorithm 1). Start from $k$ random profiles with a
    tiny count $N_j$ each. For every batch of $B$ profiles from a fresh shuffle:

    1. Assign each profile to the nearest center $\mathbf{c}_j = S_j / N_j$.
    2. Count $n_j$ and sum $\mathbf{s}_j$ the batch's profiles in each cluster.
    3. Update with the step $\alpha = 1-\gamma$:
       $N_j \leftarrow \gamma N_j + \alpha\, n_j$, $S_j \leftarrow \gamma S_j + \alpha\,\mathbf{s}_j$.
       The step decays on a cosine from $\alpha_0$ to $0.1\,\alpha_0$ over all batches.
    4. Every 5 batches, **revive**: a cluster whose count is below 10% of the mean count is
       dead; a large cluster, drawn with probability proportional to its count, is split in two
       ($\mathbf{c}\odot[1\pm\delta, 1\mp\delta, \dots]$, $\delta = 10^{-4}$), each half taking half
       its count.

    At the end, each center is the mean of the profiles assigned to it during the last pass.
    Where the paper and the reference code differ, we follow the code: starting count $2^{-16}$
    (paper: $10^{-10}$), $\delta = 10^{-4}$, and a cluster that gets no profile in a batch keeps
    its position (its $N_j$ and $S_j$ shrink together).
    """)
    return


@app.cell
def _(np):
    def e_means(
        X: np.ndarray,
        C0: np.ndarray,
        T: int,
        B: int,
        alpha0: float,
        rng: np.random.Generator,
        tau: float = 0.1,
        floor: float = 0.1,
        every: int = 5,
        delta: float = 1e-4,
    ) -> np.ndarray:
        """e-means (Bessa et al., 2026, Algorithm 1), from scratch.

        Args:
            X: Training points, shape ``(n, d)``.
            C0: Starting centers, shape ``(k, d)``.
            T: Passes over the data.
            B: Batch size.
            alpha0: Peak step of the cosine schedule.
            rng: Generator for the shuffles and the choice of donors.
            tau: Revival threshold, a share of the mean count; 0 turns revival off.
            floor: Final step as a share of ``alpha0``.
            every: Batches between revival checks.
            delta: Relative offset that splits a donor.

        Returns:
            Centers, shape ``(k, d)``.
        """
        n, d = X.shape
        k = len(C0)
        N = np.full(k, 2.0**-16)  # counts
        S = N[:, None] * C0  # sums: S / N is the center
        M = T * int(np.ceil(n / B))  # batches in all
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        t = 0
        for _epoch in range(T):
            last_sum, last_count = np.zeros((k, d)), np.zeros(k)
            order = rng.permutation(n)
            for start in range(0, n, B):
                xb = X[order[start : start + B]]
                alpha = alpha0 * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t / M)))
                gamma = 1.0 - alpha
                # 1-2. assign to the current centers; count and sum per cluster
                C = S / N[:, None]
                a = ((xb[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(axis=1)
                nj = np.bincount(a, minlength=k).astype(float)
                sj = np.zeros((k, d))
                np.add.at(sj, a, xb)
                assert sj.shape == (k, d)
                last_sum += sj
                last_count += nj
                # 3. moving averages; a cluster with no point keeps its center
                hit = nj > 0
                N[hit] = gamma * N[hit] + alpha * nj[hit]
                S[hit] = gamma * S[hit] + alpha * sj[hit]
                if gamma > 0:
                    N[~hit] *= gamma
                    S[~hit] *= gamma
                # 4. revive dead clusters by splitting donors drawn in proportion to count
                if tau > 0 and (t + 1) % every == 0:
                    dead = N < tau * N.mean()  # noqa: SIM300 (N: the paper's counts)
                    donors = ~dead & (N > 0)
                    m = int(min(dead.sum(), donors.sum()))
                    if m:
                        keys = np.full(k, -np.inf)  # Gumbel keys: weighted, no repeats
                        keys[donors] = np.log(N[donors]) - np.log(-np.log(rng.random(k)[donors]))
                        for js, jt in zip(
                            np.argsort(keys)[-m:][::-1], np.flatnonzero(dead)[:m], strict=True
                        ):
                            c, half = S[js] / N[js], N[js] / 2
                            N[js] = N[jt] = half
                            S[js] = c * (1 - delta * sign) * half
                            S[jt] = c * (1 + delta * sign) * half
                t += 1
        # finalize: the mean of each cluster's points over the last pass
        C = S / N[:, None]
        seen = last_count > 0
        C[seen] = last_sum[seen] / last_count[seen, None]
        return C

    return (e_means,)


@app.cell
def _(mo):
    mo.md(r"""
    **Step 3: verify.** With the whole training set as one batch ($B = n$), a constant step
    $\alpha = 1$ ($\gamma = 0$, so no memory), and no revival, each pass assigns every profile and
    replaces each center by the mean of its profiles: exactly Lloyd's two steps. So $T$ passes of
    `e_means` must give the centers of $T$ iterations of our `kmeans`. **Predict:** will they match
    to rounding, or differ because e-means shuffles the profiles?
    """)
    return


@app.cell
def _(K, SEED, e_means, kmeans, np, pf, shape, tr0):
    _S = shape(pf.P16[tr0])
    _C0 = _S[np.random.default_rng(SEED).choice(len(_S), size=K, replace=False)]
    _worst = 0.0
    for _T in range(1, 7):
        _Ce = e_means(_S, _C0, _T, len(_S), 1.0, np.random.default_rng(SEED), tau=0.0, floor=1.0)
        _Cl, _, _ = kmeans(_S, _C0, max_iter=_T)
        _worst = max(_worst, float(np.abs(_Ce - _Cl).max()))
    # The shuffle only reorders the sums: the centers agree to rounding (about 2e-15).
    assert _worst < 1e-12, _worst
    {"passes checked": "1 to 6", "largest |difference|": f"{_worst:.0e}"}
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 4: compare**, on the 30 campus splits of §5, from 10 random starts per split. For each
    start, Lloyd runs until no assignment changes, and e-means runs 5 passes with and without
    revival, from the same starting centers. k-means++ keeps the best of 20 starts, as for the
    typical days. Errors are the mean squared distance from each profile to its nearest center,
    on the training campuses and on the held-out ones; each split counts once (its mean over
    the starts).
    """)
    return


@app.cell
def _(K, KMeans, SEED, e_means, emeans_settings, kmeans, np, pd, pf, shape, site_splits):
    def _mse(X, C):
        _d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return float(_d2.min(axis=1).mean()), int(np.bincount(_d2.argmin(1), minlength=K).min())

    _rows = []
    for _i, (_tr, _te) in enumerate(site_splits(pf.site)):
        _S, _St = shape(pf.P16[_tr]), shape(pf.P16[_te])
        _B, _a0 = emeans_settings(K, len(_S), 5)
        _runs = [
            (
                "k-means++, best of 20",
                0,
                np.nan,
                KMeans(
                    n_clusters=K, init="k-means++", n_init=20, algorithm="lloyd", random_state=SEED
                )
                .fit(_S)
                .cluster_centers_,
            )
        ]
        for _j in range(10):
            _C0 = _S[np.random.default_rng([SEED, _i, _j]).choice(len(_S), size=K, replace=False)]
            _Cl, _, _hist = kmeans(_S, _C0)
            _runs += [("Lloyd", _j, len(_hist), _Cl)]
            for _name, _tau in (("e-means", 0.1), ("e-means, no revival", 0.0)):
                _Ce = e_means(
                    _S, _C0, 5, _B, _a0, np.random.default_rng([SEED, _i, _j, 1]), tau=_tau
                )
                _runs += [(_name, _j, 5, _Ce)]
        for _name, _j, _passes, _C in _runs:
            _train, _smallest = _mse(_S, _C)
            _rows.append(
                {
                    "split": _i,
                    "start": _j,
                    "method": _name,
                    "passes": _passes,
                    "train MSE": _train,
                    "held-out MSE": _mse(_St, _C)[0],
                    "smallest cluster": _smallest,
                }
            )
    _runs_df = pd.DataFrame(_rows)
    _per = _runs_df.groupby(["method", "split"], sort=False)[
        ["passes", "train MSE", "held-out MSE"]
    ].mean()
    _table = _per.groupby("method", sort=False).median()
    _w, _v = _per["train MSE"].unstack(0), _per["held-out MSE"].unstack(0)
    _small = _runs_df[_runs_df["smallest cluster"] <= 10].groupby("method").size()
    _lloyd = _runs_df[_runs_df.method == "Lloyd"]
    _r = _table.round(3)

    # Medians over the 30 splits (train, held out): k-means++ 0.493, 0.576; Lloyd 0.521, 0.523
    # (median 18 passes, 7 to 72); e-means 0.540, 0.518; e-means without revival 0.511, 0.541.
    assert _r.loc["k-means++, best of 20"].tolist()[1:] == [0.493, 0.576]
    assert _r.loc["Lloyd"].tolist()[1:] == [0.521, 0.523]
    assert _r.loc["e-means"].tolist()[1:] == [0.540, 0.518]
    assert _r.loc["e-means, no revival"].tolist()[1:] == [0.511, 0.541]
    assert (_lloyd.passes.median(), _lloyd.passes.min(), _lloyd.passes.max()) == (18, 7, 72)
    # Without revival, 5 passes beat Lloyd's converged training error in 19 of 30 splits.
    assert int((_w["e-means, no revival"] < _w["Lloyd"]).sum()) == 19
    # With revival, training error is higher than Lloyd's in 22 of 30 and than k-means++ in all
    # 30; held out, it is lower than Lloyd's in 25 of 30 and than k-means++ in 26 of 30.
    assert int((_w["e-means"] > _w["Lloyd"]).sum()) == 22
    assert int((_w["e-means"] > _w["k-means++, best of 20"]).sum()) == 30
    assert int((_v["e-means"] < _v["Lloyd"]).sum()) == 25
    assert int((_v["e-means"] < _v["k-means++, best of 20"]).sum()) == 26
    # A cluster of 10 buildings or fewer: k-means++ in 22 of 30 splits; Lloyd in 128 and e-means
    # without revival in 180 of 300 runs; e-means in none.
    assert _small.to_dict() == {
        "Lloyd": 128,
        "e-means, no revival": 180,
        "k-means++, best of 20": 22,
    }
    _r
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reveal.**

    1. **The speed claim holds once revival is off.** Without revival, 5 passes of e-means beat
       Lloyd's converged training error in 19 of 30 splits; Lloyd needs a median of 18 passes
       (7 to 72).
    2. **Revival changes the answer.** It treats any cluster under 10% of the mean count as dead.
       That suits the paper's setting, thousands of centers over embeddings, where a nearly empty
       center is wasted capacity. With four typical days a small cluster can be real: in 22 of 30
       splits the best k-means++ solution spends a center on 10
       buildings or fewer, such as the four evening-peak buildings of reveal 3; e-means never keeps one (0 of 300
       runs). So its training error is the highest of the four (0.540 against 0.521 for Lloyd
       and 0.493 for k-means++).
    3. **Held out, that trade wins.** On campuses the centers never saw, e-means does best (0.518
       against 0.523 for Lloyd, lower in 25 of 30 splits) and k-means++ does worst (0.576): a
       center spent on four evening-peak buildings fits no held-out building. The lowest training $J$ is not
       the goal; the held-out error is (Lectures 1 and 3).

    **When not to use it:** e-means pays off on millions of points and thousands of centers. At
    our size Lloyd takes milliseconds, and the revival rule silently drops small clusters, which
    is right for typical days but wrong when the small cluster is the finding (a leak, a
    failing sensor, a building that runs all night).

    **Stretch:** set $K = 8$ and $\tau = 0.01$. Does e-means keep the four evening-peak buildings now? What happens to
    the held-out error?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question** (one minute, no notes): why does dividing each profile by its own mean fix
    what standardizing each hour cannot? Which split would you use to report how well retrieval
    finds similar buildings on a new campus?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Size multiplies every hour of a building, a factor on the row,
                $\mathbf{x}=\bar{x}\,\mathbf{p}$. Standardizing rescales each column and leaves
                that factor in place, so mode 1 stays size (97.9% of the variance). Report
                retrieval with whole sites held out (25% same use, against 68% with a random split
                of days), because a new campus shares no buildings with the database.
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
            "Q1. Why can a Lloyd iteration never raise J, and why can it still end badly?": mo.md(
                r"""
                The assignment step picks, for each point, the center that minimizes its own term;
                the update step replaces each center by the mean of its points, which minimizes the
                cluster's sum of squared distances. Neither can raise $J$, and there are finitely
                many assignments, so it stops. But it stops at a local minimum that depends on the
                start, and a far point can capture a center (the four evening-peak buildings). Use several
                $k$-means++ starts and read the cluster sizes.
                """
            ),
            "Q2. Three modes keep 91% of the variance. Is 91% the number to report?": mo.md(
                r"""
                Not alone. Report the reconstruction error on held-out buildings and a later year
                (7.5% of the daily mean with 3 modes), bracketed by the mean shape (22.1%) and by a
                building's own year-to-year change (6.9%, rough). Variance explained on the
                training set says nothing about new campuses or the next year.
                """
            ),
            "Q3. Retrieval finds the same use 68% of the time. What do you check first?": mo.md(
                r"""
                Whether the database contains the query's own building, or its site: with a random
                split of days, 56% of nearest days were the same building. Hold out whole buildings
                or sites and compare with chance (25% against 16% here).
                """
            ),
            "Q4. Some modes come out with the opposite sign from scikit-learn. Bug?": mo.md(
                r"""
                No. An eigenvector is defined up to sign; $\mathbf{v}$ and $-\mathbf{v}$ give the
                same subspace and the same reconstruction, and which one an SVD returns depends on
                the linear-algebra library. Fix signs by a rule (for example, make the daytime
                part of mode 1 positive) before comparing or interpreting modes or scores.
                """
            ),
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Raw kW in any squared distance.** Retrieval, $k$-means, and PCA on kilowatts all measure
      size. Divide each profile by its own mean; keep the mean as a separate feature.
    - **Column standardization is not row normalization.** `StandardScaler` rescales hours; it
      cannot remove a factor that multiplies a whole row.
    - **Random splits of days or rows.** A building's other days are near-copies; hold out
      buildings, and sites when the model will meet a new campus.
    - **Fitting PCA or $k$-means on all data before evaluating.** Fit the modes and centers on the
      training buildings only, as the scaler in Lecture 2.
    - **Comparing modes without aligning signs.** $\mathbf{v}$ and $-\mathbf{v}$ are the same mode.
    - **Trusting cluster sizes you did not read.** One center can go to a handful of outliers.
    - **Flat profiles from coarse meters.** Check the time resolution (daily values spread over 24
      hours) and constant or stuck meters before reading shapes.
    - **Averaging days with gaps.** A typical day built from incomplete days mixes hours from
      different days; average only fully metered days, or handle missing values explicitly.
    - **FAISS in float32.** Distances differ from NumPy's float64 in the last digits; compare the
      neighbors, and count ties, not raw distances.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §8.2 and §8.2.1** — $k$-means as the "go-to" recipe, and choosing $k$ from a knee
      in the cost or, better, from how the clusters will be used (here: which buildings to visit).
    - **Forsyth §8.2.6** — why $k$-means tends to leave single-element or straggling clusters, and a
      "junk cluster" for points far from every center: compare it with the four evening-peak buildings.
    - **Forsyth §5.1.3–5.1.4 and §6.1.1–6.1.2** — the low-dimensional representation, its error as
      the sum of the dropped eigenvalues, and PCA read off the SVD of the data matrix. Forsyth
      divides by $N$ where we (and scikit-learn) divide by $N-1$; the modes are the same.
    - **Forsyth §5.1.6** — principal components with missing values (NIPALS): an alternative to
      dropping days with gaps, as we did.
    - **Modes of a sensor array.** The same SVD, applied to snapshots of many sensors (strain gauges
      on a girder, say), gives spatial modes, known as proper orthogonal decomposition; Brunton &
      Kutz, *Data-Driven Science and Engineering* (2nd ed., Cambridge, 2022), ch. 1, works it
      through. Lecture 24 returns to data-driven dynamics.
    - **Exercise (Forsyth 5.2).** Prove that the mean squared error of the $r$-mode representation
      is the sum of the dropped eigenvalues; then check it in §3's cell.
    - **Exercise (Forsyth 6.1).** With noisy data $X=\tilde{X}+\Xi$, argue that a low-rank
      approximation lies closer to $\tilde{X}$ than $X$ does. Relate it to the rough floor in §4:
      what are modes beyond the floor fitting?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A typical weekday is a vector of 24; divide by its mean to compare schedules, and keep the
      mean for how much is at stake.
    - Retrieval, $k$-means, and PCA all use squared distance: in raw kW all three see only size.
    - PCA's first mode, the working day, rediscovers the night/day ratio; 4 modes reach the
      year-to-year change.
    - Hold out buildings and sites: a building's other days are near-copies.
    - $k$-means spends centers on outliers; read the counts before naming types.

    **Next:** least squares through QR and the SVD, on Abrams' law.
    """)
    return


if __name__ == "__main__":
    app.run()
