"""Applied Machine Learning for CEE — Lecture 20: 1-D Signals, CNNs, and LSTMs.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l20_time_series.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_camels.sh`` (CAMELS, Newman et al., 2015, and Addor et al., 2017,
CC BY 4.0, about 3.5 GB), then ``pixi run -e aml-env python scripts/derive_camels.py``, and
``bash scripts/fetch_kw51.sh`` (KW51, Maes & Lombaert, 2021, CC BY-NC-SA 4.0, about 200 MB) once.
Chronos-2 (Ansari et al., 2025, Apache-2.0) comes from the Hugging Face Hub on the first run.
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
    # Applied Machine Learning for CEE — Lecture 20: 1-D Signals, CNNs, and LSTMs

    **Slides:** *Applied Machine Learning for CEE*, Lecture 20 (`lectures/cee_l20_time_series`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §8.3 (describing repetition with vector quantization: features for signals of any length), §10.4.2 (regularizing linear regressions), §13.1–13.2 (Markov chains and hidden Markov models: a state carried along time), and §17.1.1 (pattern detection by convolution).

    **Data:**
    - CAMELS v1.2 (Newman et al., 2015; Addor et al., 2017; doi:10.5065/D6MW2F4D; CC BY 4.0):
      Daymet weather averaged over each of 671 US basins and USGS daily flow, 1980–2014, with
      the basins' attributes. Fetch with `bash scripts/fetch_camels.sh`, then
      `python scripts/derive_camels.py` (two tables in `data/derived/`).
    - KW51 (Maes & Lombaert, 2021; doi:10.5281/zenodo.3745914; CC BY-NC-SA 4.0, teaching and
      research use): a steel bowstring railway bridge in Leuven, Belgium. Accelerations and
      strains of the October 2018 train passages, and the operator's hourly natural frequencies
      over 15 months, through a retrofit. Fetch with `bash scripts/fetch_kw51.sh`; the zips
      are read in place.
    - Chronos-2 (Ansari et al., 2025; Amazon; Apache-2.0): a pretrained forecaster,
      `amazon/chronos-2` at a pinned revision, downloaded once from the Hugging Face Hub
      (about 0.5 GB).

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert` against the
    slides' `numbers.json` (read from `lectures/build/fig/cee_l20_time_series/`). Two settings,
    chosen in the setup cell or by the environment variable `L20_SETTING`, whatever the device:

    - **`"slides"`**: every network the slides trained, each Adam rate bracketed on validation
      over three or five seeds, the fifteen Illinois basins, and every Chronos-2 forecast:
      244 training runs, about an hour on an RTX 3090 Ti (the slides' figure script took 64
      minutes there, alone on the GPU, on November 21, 2026). Only for a machine with a GPU.
    - **`"reduced"`** (the default): one seed per network at the rate the slides' search kept,
      a step cap of 200 for an LSTM and 800 for the 1-D CNN, record lengths of 1 and 15 years,
      two Illinois basins, and Chronos-2 at the slides' context on every sixth test day: 37
      minutes on 4 CPU cores of a busy machine, with 3 GB of memory at its peak (November 21,
      2026); not timed on a GPU. Its networks are undertrained and differ from the slides';
      that is what the full setting is for.

    In both settings, everything that does not train a network is checked against the slides
    exactly: the windows and the forest, the filter, SARIMA, the bridge, the alarm, the
    derivations' checks. Chronos-2 is checked in both, to a stated tolerance off the reference
    GPU. The networks' numbers are checked in the `"slides"` setting: exactly on the reference
    GPU (an RTX 3090 Ti, deterministic kernels), within stated tolerances elsewhere; in the
    `"reduced"` setting they are printed beside the slides'.

    **Your tasks** (from the slides): call two storms yourself, weigh a week of rain and flow,
    fit the unit hydrograph as a 1-D filter, read a bridge's modes from its spectrum, write an
    LSTM step and match PyTorch, train a 1-D CNN and an LSTM, then break the forecaster four ways
    and decide what runs next year's flood alert.
    """)
    return


@app.cell
def _():
    import json
    import os
    import random
    import time
    import zipfile
    from dataclasses import dataclass
    from pathlib import Path

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub loads
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.io as sio
    import torch
    import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
    from chronos import BaseChronosPipeline
    from scipy.ndimage import gaussian_filter1d
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

    SEED = 20
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed):
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"
    T_START = time.perf_counter()

    # The setting, chosen here or by the environment variable L20_SETTING, whatever the device:
    # - "slides": every network of the slides, each rate bracketed on validation, every seed,
    #   the fifteen Illinois basins, and every Chronos-2 forecast;
    # - "reduced": one seed per network at the rate the slides' search kept, MAX_STEPS_REDUCED
    #   gradient steps at most, record lengths of 1 and 15 years, two Illinois basins, and
    #   Chronos-2 at the slides' context on every CHRONOS_STRIDE_REDUCED-th test day.
    L20_SETTING = os.environ.get("L20_SETTING", "reduced").strip().lower()
    if L20_SETTING not in {"slides", "reduced"}:
        raise ValueError(f"L20_SETTING must be 'slides' or 'reduced', not {L20_SETTING!r}")
    SLIDES = L20_SETTING == "slides"
    # The slides' networks and Chronos-2 ran on an RTX 3090 Ti with deterministic kernels; there,
    # and only there, this notebook repeats them exactly.
    REFERENCE = GPU and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090 Ti"
    CHECK_NETS = SLIDES  # the networks' numbers are checked against the slides' in "slides"

    # Tolerances. Everything fitted without a network (windows, forests, the filter, SARIMA, the
    # bridge, the alarm) repeats the slides' run to the order of floating-point sums: EXACT.
    EXACT = 1e-6
    # Identities that hold in exact arithmetic, checked in float64: round-off only.
    ROUNDOFF = 1e-13
    # SARIMA's optimizer takes a slightly different path with another BLAS thread count (its
    # coefficients move by about 1e-5): checked to 1e-4, which keeps every digit on the slides.
    SARIMA_TOL = 1e-4
    # The networks and Chronos-2: exact on the reference GPU; elsewhere another device's kernels
    # take another path, so the checks allow a stated difference.
    NET_TOL = EXACT if REFERENCE else 0.05  # NSE of a trained network
    NET_REL = EXACT if REFERENCE else 0.25  # flows and gradients of a trained network, relative
    CHRONOS_TOL = EXACT if REFERENCE else 0.02  # NSE, band coverage, and R² that use Chronos-2
    COUNT_TOL = 0 if REFERENCE else 2  # alert days of a network or Chronos-2

    def shown(x, text):
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("0.66")."""
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    def find_root():
        """The repository, from its root, ``notebooks/``, or ``ipynb/`` (Jupyter's start).

        Raises:
            FileNotFoundError: If CAMELS' derived tables are missing.
        """
        rel = Path("data") / "derived" / "camels_daily.npz"
        bases = [Path.cwd(), Path.cwd().parent]
        if "__file__" in globals():
            bases.insert(0, Path(__file__).resolve().parents[1])
        for base in bases:
            if (base / rel).exists():
                return base
        raise FileNotFoundError(
            f"{rel} not found: run `bash scripts/fetch_camels.sh`, then "
            "`pixi run -e aml-env python scripts/derive_camels.py`"
        )

    ROOT = find_root()
    DAILY = ROOT / "data" / "derived" / "camels_daily.npz"
    ATTRS = ROOT / "data" / "derived" / "camels_attributes.csv"
    KW51 = ROOT / "data" / "raw" / "kw51"
    # The slides' numbers, copied from here by every check (never typed)
    _numbers = ROOT / "lectures" / "build" / "fig" / "cee_l20_time_series" / "numbers.json"
    REF = json.loads(_numbers.read_text(encoding="utf-8")) if _numbers.exists() else None
    if REF is None and SLIDES:
        raise FileNotFoundError(f"{_numbers} not found: build the slides' figures first")

    def ref(path):
        """The slides' value at ``path`` ("windows/test/chosen"), or None without numbers.json."""
        if REF is None:
            return None
        node = REF
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def check(x, path, tol=EXACT, *, rel=False):
        """Assert that ``x`` equals the slides' value at ``path`` within ``tol``.

        Absolute unless ``rel``; skipped (True) when numbers.json is not there.
        """
        if REF is None:
            return True
        r = ref(path)
        r = -np.inf if r is None else r  # numbers.json stores -inf (a diverged run) as null
        err = abs(x - r) / abs(r) if rel and r != 0 else abs(x - r)
        assert err <= tol, f"{path}: {x!r} vs the slides' {r!r}"
        return True

    # The river: North Fork Vermilion River near Bismarck, IL (USGS 03338780)
    HOME = "03338780"
    PERIODS = {  # water years, in time order: no test day precedes a training day
        "train": ("1989-10-01", "2004-09-30"),
        "val": ("2004-10-01", "2008-09-30"),
        "test": ("2008-10-01", "2014-09-30"),
    }
    STORMS = ("2013-04-10", "2012-08-26")  # storms A and B: the first of two rainy days each
    STORM_DAYS = 3  # a storm's peak is read over its two rainy days and the day after
    WINDOWS = (1, 2, 3, 5, 7, 14, 30)  # days of rain and flow given to the windowed models
    KNN_K, RF_TREES, RF_LEAF = 10, 200, 5
    RF_FEATURES = 1 / 3  # a random forest tries a third of the inputs at each split (Breiman, 2001)
    FOREST_SEEDS = (SEED, SEED + 1, SEED + 2)  # each forest setting is scored on three seeds
    UH_LAGS = 60  # days of rain in the unit hydrograph
    UH_ALPHAS = (0.0, 1e3, 1e4, 1e5, 1e6, 1e7)  # ridge penalties tried on validation
    SARIMA_ORDER, SARIMA_SEASONAL = (1, 0, 1), (0, 1, 1, 12)
    LOG_EPS = 0.01  # mm/day added before logs (the smallest flows are near 0.02)
    FLOOD_Q = 0.99  # alert above the training years' 99th-percentile flow
    SPLIT_DRAWS = 10  # random-day test sets
    PICTURE_DAY, PICTURE_K = "2013-04-11", 7  # the one picture: a week to the storm's big day
    # The bridge: KW51
    PASSAGE = "traindata_20181002_160753"  # 2 October 2018, 16:07, the month's first passage
    ACC_CHANNEL = "aBD17Az"  # vertical deck acceleration at cross girder 17 (mid-span)
    STRAIN_CHANNEL = "sgBD1011A"  # deck strain between cross girders 10 and 11
    MODE = 13  # tracked mode 13, about 6.3 Hz, a global vertical mode
    RETROFIT = ("2019-05-15", "2019-09-27")  # the retrofit works (Maes & Lombaert, 2021)
    BASELINE = ("2018-10-01", "2018-11-30")  # autumn weeks, no frost: a first, naive reference
    SINES = (1, 4, 16, 64)
    FOURIER_SECONDS = 4.0
    MATLAB_EPOCH = 719_529  # MATLAB's datenum of 1970-01-01
    MATCH_HZ = 0.1  # a spectral peak this close to an identified mode counts as that mode
    # The networks
    INPUTS = ("prcp_mm", "tmax_c", "tmin_c", "srad_wm2", "vp_pa", "dayl_s")  # Daymet, per day
    SEQ, HIDDEN, BATCH, DROPOUT, CLIP = 365, 64, 256, 0.2, 1.0
    PREDICT_BATCH = 1024  # windows per forward pass when predicting
    FIRST_KERNEL = 31  # days in the 1-D CNN's first filters
    DILATIONS = (1, 2, 4, 8, 16, 32, 64)
    NET_SEEDS, FEW_SEEDS = (0, 1, 2, 3, 4), (0, 1, 2)
    YEARS = (1, 2, 5, 15)  # training years for the data-size test, the last before WY2005
    # Training budget, in gradient steps, the same for every amount of data: validation every
    # CHECK_STEPS steps (a 1-year run peaks near step 40, so every 5); stop PATIENCE_STEPS after
    # the best check, keep its weights; MAX_STEPS caps a run. The reduced setting caps each
    # network lower (an LSTM step takes about a second on a laptop CPU, a 1-D CNN step a tenth
    # of that).
    CHECK_STEPS, PATIENCE_STEPS, MAX_STEPS = 5, 600, 12_000
    MAX_STEPS_REDUCED = {"lstm": 200, "cnn": 800}
    # Adam's rate, bracketed on validation: start from LR_START, widen along LR_LADDER until the
    # best rate is interior. The ladder ends where training fails (checked in §4).
    LR_LADDER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    LR_START = (1e-3, 3e-3, 1e-2)
    SEED_LADDER = 0
    # Chronos-2
    CHRONOS = "amazon/chronos-2"
    CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    CONTEXTS = (256, 512, 1024, 2048)  # days of history given to it, chosen on validation
    EMBED_DAYS, CLUSTERS = 2048, 5
    CHRONOS_STRIDE_REDUCED = 6  # the reduced setting forecasts every 6th test day
    PROBE_ALPHAS = tuple(10.0**k for k in range(-2, 6))
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    # What each setting runs
    SIM_SEEDS = NET_SEEDS if SLIDES else NET_SEEDS[:1]
    FC_SEEDS = FEW_SEEDS if SLIDES else FEW_SEEDS[:1]
    REGIONAL_BASINS = 15 if SLIDES else 2
    YEARS_RUN = YEARS if SLIDES else (1, 15)  # the data-size test's lengths
    print(
        f"device: {DEVICE}; setting {L20_SETTING!r}; reference GPU: {REFERENCE}; "
        f"slides' numbers.json: {'found' if REF is not None else 'not found, checks skipped'}"
    )
    return (
        ACC_CHANNEL,
        ATTRS,
        BASELINE,
        BATCH,
        BaseChronosPipeline,
        CHECK_NETS,
        CHECK_STEPS,
        CHRONOS,
        CHRONOS_REVISION,
        CHRONOS_STRIDE_REDUCED,
        CHRONOS_TOL,
        CLIP,
        CLUSTERS,
        CONTEXTS,
        COUNT_TOL,
        DAILY,
        DEVICE,
        DILATIONS,
        DROPOUT,
        EMBED_DAYS,
        EXACT,
        F,
        FC_SEEDS,
        FEW_SEEDS,
        FIRST_KERNEL,
        FLOOD_Q,
        FOREST_SEEDS,
        FOURIER_SECONDS,
        GPU,
        HIDDEN,
        HOME,
        INPUTS,
        KMeans,
        KNN_K,
        KNeighborsRegressor,
        KW51,
        LOG_EPS,
        LR_LADDER,
        LR_START,
        LinearRegression,
        MATCH_HZ,
        MATLAB_EPOCH,
        MAX_STEPS,
        MAX_STEPS_REDUCED,
        MODE,
        NET_REL,
        NET_SEEDS,
        NET_TOL,
        PALETTE,
        PASSAGE,
        PATIENCE_STEPS,
        PCA,
        PERIODS,
        PICTURE_DAY,
        PICTURE_K,
        PREDICT_BATCH,
        PROBE_ALPHAS,
        REF,
        REFERENCE,
        REGIONAL_BASINS,
        RETROFIT,
        RF_FEATURES,
        RF_LEAF,
        RF_TREES,
        ROUNDOFF,
        RandomForestRegressor,
        Ridge,
        RidgeCV,
        SARIMAX,
        SARIMA_ORDER,
        SARIMA_SEASONAL,
        SARIMA_TOL,
        SEED,
        SEED_LADDER,
        SEQ,
        SIM_SEEDS,
        SINES,
        SLIDES,
        SPLIT_DRAWS,
        STORMS,
        STORM_DAYS,
        STRAIN_CHANNEL,
        StandardScaler,
        T_START,
        UH_ALPHAS,
        UH_LAGS,
        WINDOWS,
        YEARS,
        YEARS_RUN,
        check,
        dataclass,
        find_peaks,
        gaussian_filter1d,
        make_pipeline,
        nn,
        np,
        pd,
        plt,
        ref,
        seed_all,
        shown,
        sio,
        spectrogram,
        time,
        torch,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 8–19

    On your own, no notes, two minutes:

    1. Lecture 15: a 3 × 3 filter slid over a tile's pixels. What would a filter that slides
       over days of rain compute?
    2. Lecture 8: the forest scored $R^2$ +0.16 on random folds, 0.00 with whole sites held out
       (both checked in Lecture 8's notebook). What will random *days* do to a river forecast?
    3. Lecture 19: a report's embedding lands near reports that mean the same. What would the
       embedding of a river's flow record land near?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. A weighted sum of the last few days of rain: the flow they add today. Hydrologists
           have used that filter since 1932 (the unit hydrograph); §3 fits it, and a 1-D CNN
           learns many of them.
        2. Usually inflate it: neighbouring days share their weather and their river, and a
           random split puts them on both sides. §5 tests that twice, on a forecaster that sees
           today's flow and on one that sees rain alone, and against random whole years, which
           leave no neighbour in training: here the score becomes a lottery, and random splits
           hide that the later years are harder; no leak between days shows.
        3. Rivers that behave alike: snowmelt rivers near snowmelt rivers, flashy ones near
           flashy ones. In §6 a pretrained forecaster embeds all 671 CAMELS rivers, and its
           groups line up with climate it was never shown.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Three 1-D records from CEE practice: a river's daily flow (USGS 03338780, the North Fork
    Vermilion River near Bismarck, IL), a steel railway bridge's deck acceleration (KW51,
    Leuven) at 825.8 values a second, and the same bridge's 13th natural frequency, identified
    every hour by its operator. What they share: order matters, neighbours are alike, and a
    record can be any length. CAMELS gives the river's weather as basin averages from Daymet
    and its flow divided by the basin's area, both in mm/day.
    """)
    return


@app.cell
def _(ATTRS, DAILY, PERIODS, dataclass, np, pd):
    @dataclass(frozen=True)
    class Camels:
        """CAMELS in memory, one row per basin: ids, dates, weather, flow, attributes."""

        gauge_id: np.ndarray  # 8-digit USGS ids, one per basin
        dates: np.ndarray  # datetime64[D], 1980-01-01 to 2014-12-31
        forcing: np.ndarray  # float32, basins x days x 6 Daymet basin means
        forcing_names: tuple
        flow: np.ndarray  # float32 mm/day, basins x days; NaN where the record is missing
        attrs: pd.DataFrame

        def basin(self, gauge):
            """Row of a gauge id."""
            hits = np.flatnonzero(self.gauge_id == gauge)
            if len(hits) != 1:
                raise KeyError(f"gauge {gauge} is not in CAMELS")
            return int(hits[0])

        def period(self, name):
            """Day indices of a period in PERIODS."""
            start, end = (np.datetime64(d) for d in PERIODS[name])
            return np.flatnonzero((self.dates >= start) & (self.dates <= end))

        def channel(self, name):
            """Index of a forcing variable, e.g. "prcp_mm"."""
            return self.forcing_names.index(name)

    def load_camels():
        """Read the two tables scripts/derive_camels.py writes to data/derived/."""
        for path in (DAILY, ATTRS):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found: run `bash scripts/fetch_camels.sh`, then "
                    "`pixi run -e aml-env python scripts/derive_camels.py`"
                )
        z = np.load(DAILY)
        attrs = pd.read_csv(ATTRS, dtype={"gauge_id": str})
        if not np.array_equal(attrs["gauge_id"].to_numpy(), z["gauge_id"]):
            raise ValueError("camels_attributes.csv and camels_daily.npz list different basins")
        return Camels(
            gauge_id=z["gauge_id"],
            dates=z["dates"],
            forcing=z["forcing"],
            forcing_names=tuple(z["forcing_names"].tolist()),
            flow=z["flow_mm"],
            attrs=attrs,
        )

    def nse(obs, sim):
        """Nash-Sutcliffe efficiency over the days both are finite: 1 - SSE / SS about obs' mean.

        1 is a perfect fit; 0 is no better than the observed mean of the same days.
        """
        obs, sim = np.asarray(obs, float), np.asarray(sim, float)
        if obs.shape != sim.shape:
            raise ValueError(f"shapes differ: {obs.shape} vs {sim.shape}")
        ok = np.isfinite(obs) & np.isfinite(sim)
        if ok.sum() < 2:
            raise ValueError("fewer than two days to score")
        o, s = obs[ok], sim[ok]
        return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))

    return load_camels, nse


@app.cell
def _(HOME, PERIODS, load_camels, np):
    cam = load_camels()
    home = cam.basin(HOME)
    rain = cam.forcing[home, :, cam.channel("prcp_mm")].astype(float)  # mm/day
    flow = cam.flow[home].astype(float)  # mm/day, NaN where missing
    dates = cam.dates
    tr, va, te = (cam.period(_name) for _name in PERIODS)
    area_km2 = float(cam.attrs["area_gages2"][home])
    # Straight-line (haversine) distance from the Main Quad, 40.1074 N, 88.2272 W
    _lat = np.radians([40.1074, cam.attrs["gauge_lat"][home]])
    _dlon = np.radians(cam.attrs["gauge_lon"][home] + 88.2272)
    _h = (
        np.sin((_lat[1] - _lat[0]) / 2) ** 2
        + np.cos(_lat[0]) * np.cos(_lat[1]) * np.sin(_dlon / 2) ** 2
    )
    _miles = 2 * 6371.0 * np.arcsin(np.sqrt(_h)) / 1.609344
    print(
        f"{cam.attrs['gauge_name'][home].strip()}: {area_km2:.0f} km², {_miles:.0f} miles from "
        f"campus in a straight line; {len(cam.gauge_id)} basins in "
        f"{cam.attrs['huc_02'].nunique()} water regions"
    )
    print(f"days: train {len(tr)}, validation {len(va)}, test {len(te)}")
    assert cam.forcing.shape == (671, len(dates), 6) and cam.flow.shape == (671, len(dates))
    assert cam.attrs["huc_02"].nunique() == 18
    assert cam.attrs["gauge_name"][home].strip() == "NORTH FORK VERMILION RIVER NEAR BISMARCK, IL"
    assert round(area_km2) == 692
    assert 30 < _miles < 36  # about 33 miles east of campus
    assert (len(tr), len(va), len(te)) == (5479, 1461, 2191)  # 15, 4, and 6 water years
    assert str(dates[tr[0]]) == "1989-10-01" and str(dates[te[-1]]) == "2014-09-30"
    return area_km2, cam, dates, flow, home, rain, te, tr, va


@app.cell
def _(
    ACC_CHANNEL,
    KW51,
    MATLAB_EPOCH,
    PASSAGE,
    STRAIN_CHANNEL,
    dataclass,
    np,
    pd,
    sio,
    zipfile,
):
    @dataclass(frozen=True)
    class Passage:
        """One train passage over KW51: deck acceleration and strain on one clock."""

        fs: float  # samples per second
        acc: np.ndarray  # m/s², one value per sample
        strain: np.ndarray  # strain, its first second's median removed
        start: np.datetime64  # time of the first sample
        on: tuple  # seconds when the train is on the bridge, from the strain

    def kw51_zip(name):
        """A KW51 archive in data/raw/kw51 (read in place, never extracted)."""
        path = KW51 / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `bash scripts/fetch_kw51.sh`")
        return path

    def read_passage(name=PASSAGE):
        """One passage from traindata_201810.zip.

        The train is on the bridge while the deck strain is above 5% of its peak (a sensor
        that the acceleration does not use).
        """
        with (
            zipfile.ZipFile(kw51_zip("traindata_201810.zip")) as zf,
            zf.open(f"traindata_201810/{name}.mat") as fh,
        ):
            m = sio.loadmat(fh, squeeze_me=True, struct_as_record=False)
        a, g = m["predat_a"], m["predat_sg"]
        acc = a.tdata[:, list(a.labels).index(ACC_CHANNEL)].astype(float)
        strain = g.tdata[:, list(g.labels).index(STRAIN_CHANNEL)].astype(float)
        fs = float(a.fs)
        strain = strain - np.median(strain[: int(fs)])
        loaded = np.flatnonzero(np.abs(strain) > 0.05 * np.abs(strain).max())
        start = np.datetime64(int(round((a.sdn[0] - MATLAB_EPOCH) * 86_400)), "s")
        return Passage(fs, acc, strain, start, (loaded[0] / fs, loaded[-1] / fs))

    def tracked():
        """The operator's hourly natural frequencies and the bridge's temperature.

        Returns times (datetime64[s]), frequencies (hours x 14 modes, Hz, NaN where a mode was
        not identified), and the deck's surface temperature (°C, sensor tBD31A).
        """
        with (
            zipfile.ZipFile(kw51_zip("trackedmodes.zip")) as zf,
            zf.open("trackedmodes/trackedmodes.mat") as fh,
        ):
            m = sio.loadmat(fh, squeeze_me=True, struct_as_record=False)["modes"]
        times = np.round((m.sdn - MATLAB_EPOCH) * 86_400).astype("datetime64[s]")
        temp = m.env[:, list(m.labels_env).index("tBD31A")].astype(float)
        return times, m.f.astype(float), temp

    def daily(times, x):
        """Daily medians of an hourly series (NaN hours ignored; days with none dropped)."""
        return pd.Series(x, index=pd.DatetimeIndex(times)).resample("D").median().dropna()

    def free_vibration(p):
        """The acceleration from one second after the train leaves to the end of the record."""
        return p.acc[int((p.on[1] + 1.0) * p.fs) :]

    return daily, free_vibration, read_passage, tracked


@app.cell
def _(MODE, daily, np, read_passage, tracked):
    passage = read_passage()
    times, freqs, temp = tracked()
    mode13 = daily(times, freqs[:, MODE - 1])  # one value a day: the median of its hours
    _seconds = len(passage.acc) / passage.fs
    _months = (mode13.index[-1] - mode13.index[0]).days / 30.44
    _orders = np.log10(passage.fs * 86_400)  # 825.8 values a second against one a day
    print(
        f"passage at {passage.start}: {passage.fs:.1f} values a second for {_seconds:.1f} s, "
        f"train on the bridge {passage.on[0]:.1f}-{passage.on[1]:.1f} s"
    )
    print(
        f"tracked modes: {freqs.shape[0]} hours x {freqs.shape[1]} modes; mode 13 on "
        f"{len(mode13)} days over {_months:.1f} months; sampling rates {_orders:.2f} orders "
        "of magnitude apart"
    )
    assert f"{passage.fs:.1f}" == "825.8" and round(_seconds) == 52
    assert [round(float(_s)) for _s in passage.on] == [11, 22]  # "from about 11 to 22 seconds"
    assert str(passage.start) == "2018-10-02T16:07:44"
    assert freqs.shape == (11328, 14) and len(mode13) == 465 and round(_months) == 15
    assert round(float(_orders), 2) == 7.85  # about eight orders of magnitude
    return freqs, mode13, passage, temp, times


@app.cell
def _(PALETTE, dates, flow, mode13, np, passage, plt):
    _fig, _axes = plt.subplots(3, 1, figsize=(9, 5.4), layout="constrained")
    _z = (dates >= np.datetime64("2012-10-01")) & (dates <= np.datetime64("2013-09-30"))
    _axes[0].plot(dates[_z], flow[_z], color=PALETTE[0], lw=1)
    _axes[0].set_ylabel("mm/day")
    _axes[0].set_title("River flow at Bismarck, IL: one value a day, WY2013", loc="left")
    _axes[1].plot(np.arange(len(passage.acc)) / passage.fs, passage.acc, color=PALETTE[1], lw=0.4)
    _axes[1].set_ylabel("m/s²")
    _axes[1].set_xlabel("seconds")
    _axes[1].set_title("KW51 deck acceleration: 825.8 values a second, one train", loc="left")
    _axes[2].plot(mode13.index, mode13, ".", ms=3, color=PALETTE[2])
    _axes[2].set_ylabel("Hz")
    _axes[2].set_title("KW51 mode 13: daily medians of an hourly value, 15 months", loc="left")
    _fig
    return


@app.cell
def _(PALETTE, dates, flow, gaussian_filter1d, np, pd, plt, shown):
    # Smoothing shows the slow part (wet years, droughts, the season); differencing shows the
    # fast part (the change from yesterday). Both are filters.
    _keep = dates >= np.datetime64("1989-10-01")
    _q = flow[_keep]
    _year = pd.Series(_q).rolling(365, center=True, min_periods=300).mean().to_numpy()
    _month = pd.Series(_q, index=pd.DatetimeIndex(dates[_keep])).groupby(lambda d: d.month).mean()
    _z = (dates >= np.datetime64("2013-03-15")) & (dates <= np.datetime64("2013-06-15"))
    _smooth = gaussian_filter1d(np.nan_to_num(flow[_z]), 3.0)  # Gaussian, sigma = 3 days
    _change = np.diff(flow[_z], prepend=np.nan)  # the filter [-1, 1]: today minus yesterday
    _rise_day = dates[_z][np.nanargmax(_change)]
    _ratio = float(_month.max() / _month.min())
    print(
        f"{_keep.sum() // 365} water years; 365-day mean {np.nanmin(_year):.2f} to "
        f"{np.nanmax(_year):.2f} mm/day; wettest month {_month.idxmax()}, driest "
        f"{_month.idxmin()}, ratio {_ratio:.2f}; largest one-day rise in spring 2013 "
        f"{np.nanmax(_change):+.1f} mm/day on {_rise_day}"
    )
    assert _keep.sum() // 365 == 25
    assert round(float(np.nanmax(_q)), -1) == 50 and 0.01 <= np.nanmin(_q) + 0.01 < 0.02  # log axis
    assert shown(np.nanmin(_year), "0.2") and shown(np.nanmax(_year), "2.6")  # wet and dry years
    assert (_month.idxmax(), _month.idxmin()) == (6, 9) and shown(_ratio, "5.2")  # June, Sept.
    assert shown(float(np.nanmax(_change)), "20.3") and str(_rise_day) == "2013-04-19"
    _fig, (_a, _b) = plt.subplots(
        1, 2, figsize=(9, 3.4), width_ratios=(1.5, 1), layout="constrained"
    )
    _a.plot(dates[_keep], _q + 0.01, color=PALETTE[0], lw=0.4, label="daily flow")
    _a.plot(dates[_keep], _year, color=PALETTE[1], lw=2, label="365-day mean")
    _a.set_yscale("log")
    _a.set_ylabel("flow (mm/day, log)")
    _a.legend(loc="lower left", frameon=False, ncol=2)
    _b.plot(dates[_z], flow[_z], ".-", color=PALETTE[0], lw=1, ms=3, label="daily")
    _b.plot(dates[_z], _smooth, "--", color=PALETTE[1], lw=2, label="Gaussian, σ = 3 days")
    _b.plot(dates[_z], _change, "-.", color=PALETTE[2], lw=1, label="change, [−1, 1]")
    _b.set_ylabel("mm/day")
    _b.legend(frameon=False, fontsize=8)
    _b.tick_params(axis="x", labelrotation=30)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Will it flood tomorrow?

    The county closes low-water crossings and moves equipment a day ahead when tomorrow's flow
    at Bismarck will beat the **1-in-100-days flow**, the training years' 99th percentile of
    daily flow. A missed flood risks lives at the crossings; a false alarm costs a closure and
    trust in the next alert.

    | Tomorrow | Floods | Doesn't |
    |:---------|:-------|:--------|
    | Alert | roads closed in time | a needless closure |
    | No alert | cars in the water | nothing |

    Periods in time order: train on water years 1990–2004, validate on 2005–2008, test on
    2009–2014, scored once at the settings validation chose.
    """)
    return


@app.cell
def _(FLOOD_Q, ROUNDOFF, area_km2, check, flow, np, tr):
    threshold = float(np.nanquantile(flow[tr], FLOOD_Q))
    _m3s = threshold * area_km2 / 86.4  # mm/day over km² -> m³/s
    print(f"alert threshold: {threshold:.2f} mm/day over the basin, {_m3s:.1f} m³/s")
    check(threshold, "alerts/threshold", ROUNDOFF)
    check(_m3s, "alerts/threshold_m3s", 1e-9)
    assert 50 < _m3s < 200  # a flood flow for a 692-km² basin, in m³/s (one conversion only)
    return (threshold,)


@app.cell
def _(ROUNDOFF, STORMS, STORM_DAYS, check, dates, flow, mo, np, rain):
    # Each storm's three days: its two rainy days and the day after (where peaks are read)
    storm_days = {
        _d: int(np.flatnonzero(dates == np.datetime64(_d))[0]) + np.arange(STORM_DAYS)
        for _d in STORMS
    }
    storms = {
        _d: {
            "rain_2day": float(rain[_days[0]] + rain[_days[1]]),
            "flow_before": float(flow[_days[0] - 1]),
            "peak_3day": float(np.nanmax(flow[_days])),
            "month": str(dates[_days[0]])[:7],
        }
        for _d, _days in storm_days.items()
    }
    _a, _b = (storms[_d] for _d in STORMS)
    for _d in STORMS:
        for _k in ("rain_2day", "flow_before", "peak_3day"):
            check(storms[_d][_k], f"storms/{_d}/{_k}", ROUNDOFF)
    assert (_a["month"], _b["month"]) == ("2013-04", "2012-08")
    mo.md(rf"""
    ### Your turn: same rain, two floods?

    | Storm | Rain, two days | Flow the day before | Month |
    |:------|---------------:|--------------------:|:------|
    | A, 10–11 April 2013 | {_a["rain_2day"]:.0f} mm | {_a["flow_before"]:.2f} mm/day | April |
    | B, 26–27 August 2012 | {_b["rain_2day"]:.0f} mm | {_b["flow_before"]:.3f} mm/day | August |

    Which storm raises the river more, and by how much? Two minutes alone, then a neighbour.
    """)
    return storm_days, storms


@app.cell
def _(STORMS, mo, storms):
    _a, _b = (storms[_d] for _d in STORMS)
    _ratio = _a["peak_3day"] / _b["peak_3day"]
    assert round(_ratio, -1) == 30  # "about thirty times less from more rain"
    mo.accordion(
        {
            "Reveal": mo.md(rf"""
        Storm A peaked at {_a["peak_3day"]:.1f} mm/day within three days; storm B at
        {_b["peak_3day"]:.2f}, {_ratio:.0f} times less from more rain. Storm A fell on a wet
        spring, storm B at the end of the 2012 drought. What did the river know that the rain
        gauge did not? §3 (the filter), §4 (the LSTM), and §5 (the drought) return to it.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: seven days, fourteen numbers

    Cut the record into windows of the last $k$ days of rain $P$ and flow $q$, and predict the
    next day's flow. Each window becomes one row of a table, and every method of Unit 1
    applies. With $k = 7$ and a linear model, fitted on water years 1990–2008:

    $$\hat q_{t+1} = b + \sum_{k=0}^{6} a_k\, P_{t-k} + \sum_{k=0}^{6} c_k\, q_{t-k}$$

    The picture is the week to 11 April 2013, storm A's big day, and the forecast for 12 April.
    The same arithmetic, with learned weights, slid along the whole record, is a 1-D filter.
    """)
    return


@app.cell
def _(
    KNN_K,
    KNeighborsRegressor,
    LinearRegression,
    RF_FEATURES,
    RF_LEAF,
    RF_TREES,
    RandomForestRegressor,
    SEED,
    StandardScaler,
    flow,
    make_pipeline,
    np,
    rain,
):
    def issued(days):
        """The days a day-ahead forecast is made on, for target days ``days``: one day before.

        Every fit and every score is over target days inside a period, so no training row has
        a validation or test day as its target.
        """
        return days - 1

    def window(days, k, *, tomorrow_rain=False, with_flow=True):
        """Rain and flow of the k days ending on each day t, one row per t.

        Columns: rain t, t-1, ..., t-k+1, then flow t, ..., t-k+1 (left out without
        with_flow), then rain t+1 if asked. The target of row t is the flow of day t+1.
        """
        lags = np.arange(k)
        cols = [rain[days[:, None] - lags]]
        if with_flow:
            cols.append(flow[days[:, None] - lags])
        if tomorrow_rain:
            cols.append(rain[days + 1, None])
        x = np.concatenate(cols, axis=1)
        assert x.shape == (len(days), k * (1 + with_flow) + tomorrow_rain), x.shape
        return x

    def model_for(name, seed=SEED):
        """A windowed regression by name: linear, k-NN (standardized), or random forest.

        The random forest tries a random third of the inputs at each split (Breiman, 2001).
        scikit-learn's default, max_features=1.0, would try them all: bagging, not a forest.
        """
        if name == "linear":
            return LinearRegression(fit_intercept=True)
        if name == "knn":
            return make_pipeline(
                StandardScaler(), KNeighborsRegressor(n_neighbors=KNN_K, weights="uniform", p=2)
            )
        if name == "forest":
            return RandomForestRegressor(
                n_estimators=RF_TREES,
                max_features=RF_FEATURES,
                min_samples_leaf=RF_LEAF,
                n_jobs=8,
                random_state=seed,
            )
        raise ValueError(f"unknown model {name!r}")

    def windowed_forecast(
        name,
        k,
        fit_days,
        score_days,
        *,
        target="level",
        tomorrow_rain=False,
        with_flow=True,
        seed=SEED,
    ):
        """Fit on fit_days' windows and forecast tomorrow's flow on score_days.

        target "level" predicts flow t+1; "change" predicts flow t+1 - flow t and adds today's
        flow back (so it needs the flow). tomorrow_rain also gives the model day t+1's rain: a
        perfect weather forecast. Returns one forecast per score day, NaN where its window has
        a gap.
        """
        if not with_flow and target != "level":
            raise ValueError("a forecast of the change adds today's flow back: it needs the flow")
        xf = window(fit_days, k, tomorrow_rain=tomorrow_rain, with_flow=with_flow)
        xs = window(score_days, k, tomorrow_rain=tomorrow_rain, with_flow=with_flow)
        yf = flow[fit_days + 1] - (flow[fit_days] if target == "change" else 0.0)
        ok = np.isfinite(xf).all(axis=1) & np.isfinite(yf)
        model = model_for(name, seed)
        model.fit(xf[ok], yf[ok])
        good = np.isfinite(xs).all(axis=1)
        out = np.full(len(score_days), np.nan)
        out[good] = model.predict(xs[good])
        if target == "change":
            out += flow[score_days]
        return out

    def chosen_forecast(chosen, fit_days, score_days, **kw):
        """The windowed model validation chose, (model, target, k), fitted and used elsewhere."""
        name, target, k = chosen
        return windowed_forecast(name, k, fit_days, score_days, target=target, **kw)

    return chosen_forecast, issued, window, windowed_forecast


@app.cell
def _(
    LinearRegression,
    PALETTE,
    PICTURE_DAY,
    PICTURE_K,
    ROUNDOFF,
    check,
    dates,
    flow,
    issued,
    np,
    plt,
    rain,
    tr,
    va,
    window,
):
    _t = int(np.flatnonzero(dates == np.datetime64(PICTURE_DAY))[0])
    _fit = issued(np.concatenate([tr, va]))  # targets: water years 1990-2008
    _xf = window(_fit, PICTURE_K)
    _ok = np.isfinite(_xf).all(axis=1) & np.isfinite(flow[_fit + 1])
    _m = LinearRegression(fit_intercept=True).fit(_xf[_ok], flow[_fit + 1][_ok])
    _x = window(np.array([_t]), PICTURE_K)[0]  # rain t..t-6, then flow t..t-6
    _contrib = _m.coef_ * _x  # each weight times its day
    _forecast = float(_contrib.sum() + _m.intercept_)
    _a, _c = _m.coef_[:PICTURE_K], _m.coef_[PICTURE_K:]
    print("rain (mm), 11 April back to 5 April:", np.round(_x[:PICTURE_K], 2))
    print("flow (mm/day):                       ", np.round(_x[PICTURE_K:], 2))
    print("weights on rain:", np.round(_a, 3), "\nweights on flow:", np.round(_c, 2))
    print(
        f"forecast {_forecast:.2f}, observed {flow[_t + 1]:.2f}, persistence {flow[_t]:.2f} mm/day"
    )
    assert len(_m.coef_) == 14  # seven days, fourteen numbers
    assert np.all(_x[3:PICTURE_K] < 0.5)  # almost no rain before the 9th
    assert np.all(np.abs(_x[PICTURE_K + 2 :] - 0.7) < 0.15)  # steady near 0.7 before
    assert np.argmax(_a) == 0 and np.argmax(_c) == 0  # largest on the last day's rain and flow
    assert np.isclose(_forecast, _m.predict(_x[None])[0], rtol=1e-12)  # the sum is the model
    for _i in range(PICTURE_K):
        check(_x[_i], f"picture/rain/{_i}", ROUNDOFF)
        check(_x[PICTURE_K + _i], f"picture/flow/{_i}", ROUNDOFF)
        check(_a[_i], f"picture/weights_rain/{_i}")
        check(_c[_i], f"picture/weights_flow/{_i}")
    check(_forecast, "picture/forecast")
    check(flow[_t + 1], "picture/observed", ROUNDOFF)
    check(flow[_t], "picture/persistence", ROUNDOFF)
    _days = dates[_t - PICTURE_K + 1 : _t + 2]
    _fig, (_top, _bot) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True, height_ratios=(1, 1.4), layout="constrained"
    )
    _top.bar(_days[:-1], rain[_t - PICTURE_K + 1 : _t + 1], color=PALETTE[0], width=0.7)
    _top.set_ylabel("rain (mm)")
    _bot.plot(_days[:-1], flow[_t - PICTURE_K + 1 : _t + 1], "o-", color="black", lw=2)
    _bot.plot([_days[-1]], [_forecast], "s", ms=10, color=PALETTE[1], label="forecast")
    _bot.plot([_days[-1]], [flow[_t + 1]], "*", ms=14, color=PALETTE[2], label="observed")
    for _k in range(PICTURE_K):
        _d = _days[PICTURE_K - 1 - _k]
        _top.annotate(f"{_a[_k]:+.3f}", (_d, rain[_t - _k]), ha="center", va="bottom", fontsize=8)
        _bot.annotate(f"{_c[_k]:+.2f}", (_d, flow[_t - _k]), ha="center", va="bottom", fontsize=8)
    _bot.set_ylabel("flow (mm/day)")
    _bot.legend(frameon=False, loc="upper left")
    _bot.tick_params(axis="x", labelrotation=30)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    ### Windows: which model, which target, how long?

    **Assume: tomorrow depends on the last $k$ days only.** Three models from Unit 1 on the same
    windows (linear, 10 nearest neighbours on standardized inputs, and a random forest of 200
    trees whose splits each choose among a random third of the inputs, with leaves of at least
    5 days; Breiman, 2001), each predicting either tomorrow's flow or its change from today,
    for windows of 1 to 30 days. Score: the Nash–Sutcliffe efficiency
    (Nash & Sutcliffe, 1970),

    $$\mathrm{NSE} = 1 - \frac{\sum_t (\hat q_t - q_t)^2}{\sum_t (q_t - \bar q)^2},$$

    1 for a perfect forecast, 0 for no better than the observed mean. The setting is chosen on
    validation (water years 2005–2008), refitted on 1990–2008, and scored once on the test
    years. This takes about a minute.
    """)
    return


@app.cell
def _(
    FOREST_SEEDS,
    SEED,
    WINDOWS,
    chosen_forecast,
    flow,
    issued,
    np,
    nse,
    te,
    time,
    tr,
    va,
    windowed_forecast,
):
    _t0 = time.perf_counter()
    # A forest's validation NSE moves by about 0.01 with its random state, as much as the gaps
    # between its settings: each forest setting is scored by its median over three seeds.
    forest_by_seed, val_nse = {}, {}
    for _name in ("linear", "knn", "forest"):
        for _target in ("level", "change"):
            _seeds = FOREST_SEEDS if _name == "forest" else (SEED,)
            _by = [
                [
                    nse(
                        flow[va],
                        windowed_forecast(
                            _name, _k, issued(tr), issued(va), target=_target, seed=_s
                        ),
                    )
                    for _s in _seeds
                ]
                for _k in WINDOWS
            ]
            val_nse.setdefault(_name, {})[_target] = [float(np.median(_v)) for _v in _by]
            if _name == "forest":
                forest_by_seed[_target] = _by
    val_persistence = nse(flow[va], flow[issued(va)])  # tomorrow = today
    _best = max(
        ((_n, _g, _i) for _n in val_nse for _g in val_nse[_n] for _i in range(len(WINDOWS))),
        key=lambda c: val_nse[c[0]][c[1]][c[2]],
    )
    assert 0 < _best[2] < len(WINDOWS) - 1, "the chosen window is at the edge of the grid"
    chosen = (_best[0], _best[1], WINDOWS[_best[2]])
    _fit19 = issued(np.concatenate([tr, va]))  # refit on 19 years after the choice
    forest_test = chosen_forecast(chosen, _fit19, issued(te))
    forest_rain_test = chosen_forecast(chosen, _fit19, issued(te), tomorrow_rain=True)
    # The same forest on the 15 training years only: the years the networks train on
    forest15_test = chosen_forecast(chosen, issued(tr), issued(te))
    forest15_rain_test = chosen_forecast(chosen, issued(tr), issued(te), tomorrow_rain=True)
    windows_test = {
        "persistence": nse(flow[te], flow[issued(te)]),
        "chosen": nse(flow[te], forest_test),
        "chosen_tomorrow_rain": nse(flow[te], forest_rain_test),
        "chosen_15": nse(flow[te], forest15_test),
        "chosen_tomorrow_rain_15": nse(flow[te], forest15_rain_test),
    }
    print(f"chosen on validation: {chosen}, {time.perf_counter() - _t0:.0f} s")
    print({_k: round(_v, 3) for _k, _v in windows_test.items()})
    return (
        chosen,
        forest15_rain_test,
        forest15_test,
        forest_by_seed,
        forest_test,
        val_nse,
        val_persistence,
        windows_test,
    )


@app.cell
def _(
    WINDOWS,
    check,
    chosen,
    flow,
    forest_by_seed,
    issued,
    np,
    ref,
    tr,
    va,
    val_nse,
    val_persistence,
    window,
    windowed_forecast,
    windows_test,
):
    for _n in val_nse:
        for _g in val_nse[_n]:
            for _i in range(len(WINDOWS)):
                check(val_nse[_n][_g][_i], f"windows/val_nse/{_n}/{_g}/{_i}")
    for _g, _by in forest_by_seed.items():
        for _i, _v in enumerate(_by):
            for _j, _x in enumerate(_v):
                check(_x, f"windows/forest_by_seed/{_g}/{_i}/{_j}")
    if ref("windows") is not None:
        _c = ref("windows/chosen")
        assert chosen == (_c["model"], _c["target"], _c["k"]), chosen
    check(val_nse[chosen[0]][chosen[1]][WINDOWS.index(chosen[2])], "windows/chosen/val_nse")
    for _k, _v in windows_test.items():
        check(_v, f"windows/test/{_k}")
    check(val_persistence, "windows/val_persistence")
    # The forest's seeds spread about as widely as its settings differ
    _spread = max(max(_v) - min(_v) for _by in forest_by_seed.values() for _v in _by)
    print(f"largest spread of a forest setting over its three seeds: {_spread:.4f}")
    # Linear: the same for both targets, because today's flow is an input (moving it across the
    # equals sign changes nothing): the forecasts themselves agree.
    _lev = windowed_forecast("linear", 7, issued(tr), issued(va), target="level")
    _chg = windowed_forecast("linear", 7, issued(tr), issued(va), target="change")
    assert np.allclose(_lev, _chg, rtol=0, atol=1e-8, equal_nan=True)
    # k-NN on flow falls as the window grows; on changes it does not, with the same inputs
    _kl, _kc = np.array(val_nse["knn"]["level"]), np.array(val_nse["knn"]["change"])
    assert np.all(np.diff(_kl) < 0) and _kc[-1] - _kl[-1] > 0.2
    assert window(va[:1], 30).shape[1] == 60
    # A tree's forecast of a level is an average of training levels, so it cannot exceed the
    # largest; a forecast of the change adds today's flow and reaches further up a big flood.
    _tl = windowed_forecast("forest", 14, issued(tr), issued(va), target="level")
    _tc = windowed_forecast("forest", 14, issued(tr), issued(va), target="change")
    _big = int(np.nanargmax(flow[va]))
    assert np.nanmax(_tl) <= np.nanmax(flow[tr])
    assert _tc[_big] > _tl[_big]
    print(
        f"largest validation flood {flow[va][_big]:.1f} mm/day: forecast as a level "
        f"{_tl[_big]:.1f}, as a change {_tc[_big]:.1f}"
    )
    return


@app.cell
def _(KNN_K, PALETTE, WINDOWS, plt, val_nse, val_persistence):
    _fig, _ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    _labels = {"linear": "linear", "knn": f"{KNN_K}-NN", "forest": "forest"}
    for _i, _n in enumerate(_labels):
        for _g, _ls in (("level", "-"), ("change", "--")):
            _ax.plot(
                WINDOWS,
                val_nse[_n][_g],
                _ls,
                marker="osv"[_i],
                color=PALETTE[_i],
                label=f"{_labels[_n]}, {'flow' if _g == 'level' else 'change'}",
            )
    _ax.axhline(val_persistence, color="grey", ls=":", label="persistence")
    _ax.set_xscale("log")
    _ax.set_xticks(WINDOWS, [str(_k) for _k in WINDOWS])
    _ax.set_xlabel("window: days of rain and flow")
    _ax.set_ylabel("validation NSE")
    _ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower left")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Assume: rain adds flow the same way every day

    One filter $w$ on rain alone, the same on every day of the year:

    $$\hat q_t = b + \sum_{k=0}^{59} w_k\, P_{t-k},$$

    a 1-D convolution: Sherman's (1932) unit hydrograph, the flow a basin gives from one unit of
    rain, shifted and added for every day's rain. The 60 weights are fitted by ridge regression
    (Forsyth §10.4.2), the penalty chosen on validation, then refitted on 1990–2008.
    """)
    return


@app.cell
def _(Ridge, UH_ALPHAS, UH_LAGS, flow, np, nse, rain, te, tr, va):
    _lags = np.arange(UH_LAGS)

    def _rows(days):
        """The last 60 days of rain (lag 0 first) and the day's flow, one row per day."""
        return rain[days[:, None] - _lags], flow[days]

    _xt, _yt = _rows(tr)
    _ok = np.isfinite(_yt)
    _val = {}
    for _alpha in UH_ALPHAS:
        _m = Ridge(alpha=_alpha).fit(_xt[_ok], _yt[_ok])
        _val[_alpha] = nse(flow[va], _m.predict(_rows(va)[0]))
    _alpha = max(_val, key=_val.get)
    if _alpha in (UH_ALPHAS[0], UH_ALPHAS[-1]):
        raise ValueError(f"ridge penalty {_alpha} is at the edge of UH_ALPHAS; widen the grid")
    _xf, _yf = _rows(np.concatenate([tr, va]))
    _ok = np.isfinite(_yf)
    _m = Ridge(alpha=_alpha).fit(_xf[_ok], _yf[_ok])
    _all = np.arange(UH_LAGS, len(rain))
    _sim = np.full(len(rain), np.nan)
    _sim[_all] = _m.predict(rain[_all[:, None] - _lags])
    uh = {
        "alpha": _alpha,
        "alphas": UH_ALPHAS,
        "val_nse": _val[_alpha],
        "test_nse": nse(flow[te], _sim[te]),
        "weights": _m.coef_,
        "intercept": float(_m.intercept_),
        "sim": _sim,
    }
    print(
        f"ridge penalty {_alpha:g}; validation NSE {uh['val_nse']:.3f}, test {uh['test_nse']:.3f}; "
        f"weights sum to {_m.coef_.sum():.3f}, peak at lag {np.argmax(_m.coef_)}"
    )
    return (uh,)


@app.cell
def _(STORMS, UH_LAGS, check, np, rain, storm_days, uh):
    _w = uh["weights"]
    assert 0 < list(uh["alphas"]).index(uh["alpha"]) < len(uh["alphas"]) - 1  # inside the grid
    check(uh["alpha"], "uh/alpha")
    check(uh["val_nse"], "uh/val_nse")
    check(uh["test_nse"], "uh/test_nse")
    check(uh["intercept"], "uh/intercept")
    for _i in range(UH_LAGS):
        check(_w[_i], f"uh/weights/{_i}")
    check(float(_w.sum()), "uh/runoff_per_mm")
    check(int(np.argmax(_w)), "uh/peak_lag", 0)
    assert np.max(np.abs(_w[21:])) < 0.1 * _w.max()  # near zero by three weeks
    # The fitted filter is a convolution: b + (w * P)(t), checked against np.convolve
    _conv = np.convolve(rain, _w)[: len(rain)]
    assert np.allclose(uh["sim"][UH_LAGS:], uh["intercept"] + _conv[UH_LAGS:], rtol=0, atol=1e-9)
    # The attempt, first reveal: the filter answers in proportion to the rain
    _peak = {_d: float(np.max(uh["sim"][_days])) for _d, _days in storm_days.items()}
    _a, _b = (_peak[_d] for _d in STORMS)
    print(f"the filter's peaks: storm A {_a:.2f}, storm B {_b:.2f} mm/day")
    assert _b > _a
    for _d in STORMS:
        check(_peak[_d], f"storms/{_d}/uh_peak_3day")
    return


@app.cell
def _(PALETTE, UH_LAGS, dates, flow, np, plt, uh):
    _fig, (_a, _b) = plt.subplots(
        1, 2, figsize=(10, 3.4), width_ratios=(1, 2), layout="constrained"
    )
    _a.bar(np.arange(UH_LAGS), uh["weights"], color=PALETTE[0], width=0.8)
    _a.set_xlim(-1, 30)
    _a.set_xlabel("days since the rain")
    _a.set_ylabel("mm/day of flow per mm of rain")
    _a.set_title(f"sum {uh['weights'].sum():.2f}")
    _z = (dates >= np.datetime64("2012-06-01")) & (dates <= np.datetime64("2013-06-30"))
    _b.plot(dates[_z], flow[_z], color="black", lw=1.2, label="observed")
    _b.plot(dates[_z], uh["sim"][_z], "--", color=PALETTE[1], lw=1.2, label="rain * filter")
    _b.set_ylabel("flow (mm/day)")
    _b.set_title(f"test NSE {uh['test_nse']:.2f}: the filter has no memory")
    _b.legend(frameon=False, loc="upper left")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The filter makes floods out of the late-summer 2012 storms that the river never ran, and
    reaches only part of the April 2013 flood: more rain, more flow, every day of the year. The
    assumption fails because the basin keeps water: wet soil sends rain to the river, dry soil
    drinks it. A model needs memory.

    ### SARIMA: seasons first, then noise

    **Assume: the record repeats its season, and the rest is a short memory of recent changes
    and errors.** ARIMA (Box & Jenkins, 1970) predicts a value from recent values and recent
    errors; SARIMA does the same at a lag of 12 months. Here: statsmodels' SARIMAX,
    $(1, 0, 1)(0, 1, 1)_{12}$, on log monthly flow $y$, with $B$ the shift one month back:

    $$(1 - \phi B)(1 - B^{12})\, y_t = (1 + \theta B)(1 + \Theta B^{12})\, \varepsilon_t.$$

    The seasonal difference $(1 - B^{12})$ is the change from the same month a year before; the
    order $(1, 0, 1)$ adds one autoregressive term and one moving-average term on top, with no
    month-to-month difference. Fitted on water years 1990–2008 and asked for the 72 test months
    at once, then one month ahead with the same parameters. (statsmodels writes the
    moving-average polynomials with a plus sign, as here.)
    """)
    return


@app.cell
def _(
    LOG_EPS,
    SARIMAX,
    SARIMA_ORDER,
    SARIMA_SEASONAL,
    SARIMA_TOL,
    check,
    dates,
    flow,
    np,
    pd,
):
    _s = pd.Series(flow, index=pd.DatetimeIndex(dates))["1989-10-01":"2014-09-30"]
    _monthly = _s.resample("MS").mean()
    _monthly[_s.isna().resample("MS").sum() > 3] = np.nan  # a month with gaps is missing
    _m = np.log(_monthly + LOG_EPS)
    _fit, _test = _m[:"2008-09-30"], _m["2008-10-01":]
    _res = SARIMAX(_fit, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL).fit(disp=False)
    _fc = _res.get_forecast(len(_test))
    _band = _fc.conf_int(alpha=0.10)
    _one = _res.apply(_m).get_prediction(start=len(_fit)).predicted_mean  # one month ahead
    _clim = _fit.groupby(_fit.index.month).mean()  # each calendar month's training mean
    _clim_test = np.array([_clim[_t.month] for _t in _test.index])

    def _rmse(pred):
        ok = np.isfinite(_test.to_numpy()) & np.isfinite(pred)
        return float(np.sqrt(np.mean((_test.to_numpy()[ok] - pred[ok]) ** 2)))

    _rmse_log = {
        "SARIMA, 72 months ahead": _rmse(_fc.predicted_mean.to_numpy()),
        "SARIMA, 1 month ahead": _rmse(_one.to_numpy()),
        "climatology": _rmse(_clim_test),
        "persistence (last month)": _rmse(_m.shift(1)["2008-10-01":].to_numpy()),
    }
    _inside = (_test >= _band.iloc[:, 0]) & (_test <= _band.iloc[:, 1])
    _coverage = float(_inside[_test.notna()].mean())
    _gap = float(np.sqrt(np.mean((_fc.predicted_mean.to_numpy() - _clim_test) ** 2)))
    print({_k: round(float(_v), 4) for _k, _v in _res.params.items()})
    print({_k: round(_v, 3) for _k, _v in _rmse_log.items()})
    print(
        f"90% band covered {_coverage:.1%} of {len(_test)} months; 72-month forecast vs "
        f"climatology: RMS gap {_gap:.2f}"
    )
    assert len(_fit) == 19 * 12 and len(_test) == 72 and _m.notna().all()
    for _k, _v in _res.params.items():
        check(float(_v), f"sarima/params/{_k}", SARIMA_TOL)
    _keys = {
        "SARIMA, 72 months ahead": "sarima_72",
        "SARIMA, 1 month ahead": "sarima_1",
        "climatology": "climatology",
        "persistence (last month)": "persistence",
    }
    for _k, _key in _keys.items():
        check(_rmse_log[_k], f"sarima/rmse_log/{_key}", SARIMA_TOL)
    check(_coverage, "sarima/band90_coverage", SARIMA_TOL)
    # The seasonal MA coefficient nearly cancels the seasonal difference: (1 + Θ B¹²) ≈ (1 − B¹²)
    assert abs(_res.params["ma.S.L12"] + 1) < 0.01
    assert _gap < 0.15  # 72 months ahead it is the climatology (the seasonal MA cancels)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Trees and ARIMA, or deep networks?

    | Practitioners weigh | Trees, (S)ARIMA | Deep networks |
    |:--------------------|:----------------|:--------------|
    | Accuracy | often enough | often better with many series |
    | Explain a forecast | yes | hard |
    | Stable, controllable | yes | needs care |
    | Expert knowledge in | easy | hard |
    | Cost to run and watch | low | higher |

    Hydrology is the counterexample to keep in mind (cited, not recomputed here): Kratzert et
    al. (2019) trained one LSTM on 531 basins; over the 447 basins every benchmark model
    covered, its median NSE was 0.76 against 0.60 for a calibrated conceptual model (SAC-SMA).
    Google's flood forecasts, in over 80 countries, run on LSTMs (Nearing et al., 2024). The
    difference: thousands of similar series to learn from.

    ### Any record is a sum of sines

    **Assume: the record is a weighted sum of sines and cosines** (Fourier, 1807). Over $T$
    seconds the sines are the multiples of $1/T$; all $N/2$ of them rebuild a record of $N$
    samples exactly. A bridge is the friendliest case: after the load leaves, it rings at a
    few natural frequencies, each a sine that decays. Four seconds after a train left KW51:
    """)
    return


@app.cell
def _(FOURIER_SECONDS, SINES, check, free_vibration, np, passage, shown):
    _piece = free_vibration(passage)[: int(FOURIER_SECONDS * passage.fs)]
    _c = np.fft.rfft(_piece - _piece.mean())  # one coefficient per multiple of 1/T

    def _rebuild(k):
        """The piece from its k largest Fourier terms, and their indices, largest first."""
        order = np.argsort(np.abs(_c))[::-1][:k]
        kept = np.zeros_like(_c)
        kept[order] = _c[order]
        return np.fft.irfft(kept, n=len(_piece)) + _piece.mean(), order

    _explained = {}
    for _n in SINES:
        _fit, _order = _rebuild(_n)
        _explained[_n] = float(1 - np.var(_piece - _fit) / np.var(_piece))
    _first_hz = float(_rebuild(1)[1][0] / FOURIER_SECONDS)
    _all = np.fft.irfft(_c, n=len(_piece)) + _piece.mean()
    print(
        f"{len(_piece)} samples, {len(_piece) // 2} sines {passage.fs / len(_piece):.4f} Hz "
        f"apart; largest term at {_first_hz} Hz; variance explained:",
        {_n: f"{_v:.1%}" for _n, _v in _explained.items()},
    )
    assert len(_piece) == 3303 and shown(passage.fs / len(_piece), "0.25")
    check(len(_piece) // 2, "fourier/terms_total", 0)
    check(_first_hz, "fourier/first_hz")
    for _n in SINES:
        check(_explained[_n], f"fourier/explained/{_n}")
    assert np.allclose(_all, _piece, rtol=0, atol=1e-12)  # all 1,651 rebuild it exactly
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The spectrum finds the bridge's modes

    The amplitude of each sine over the 29 seconds of free vibration after the train, with a
    Hann window, against the natural frequencies the operator identified in the same hour from
    ambient vibration (Maes & Lombaert, 2021). Frequencies are $1/T$ apart; the highest a
    record sampled at $f_s$ can show is $f_s / 2$ (Nyquist).
    """)
    return


@app.cell
def _(
    MATCH_HZ,
    check,
    find_peaks,
    free_vibration,
    freqs,
    np,
    passage,
    ref,
    times,
):
    def spectrum(x, fs):
        """One-sided amplitude spectrum of a Hann-windowed record (frequencies in Hz)."""
        w = np.hanning(len(x))
        amp = np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()
        return np.fft.rfftfreq(len(x), 1 / fs), amp

    _tail = free_vibration(passage)
    spec_f, spec_amp = spectrum(_tail, passage.fs)
    _hour = int(np.argmin(np.abs(times - passage.start)))
    modes_hour = freqs[_hour][np.isfinite(freqs[_hour])]
    _peak_idx, _ = find_peaks(spec_amp)
    _df = spec_f[1] - spec_f[0]
    _matched = []
    for _fm in modes_hour:  # the tallest local peak within MATCH_HZ of each identified mode
        _cand = np.intersect1d(np.flatnonzero(np.abs(spec_f - _fm) <= MATCH_HZ), _peak_idx)
        if len(_cand):
            _k = _cand[np.argmax(spec_amp[_cand])]
            _matched.append((float(_fm), float(spec_f[_k]), float(spec_f[_k] - _fm)))
    _within = [_m for _m in _matched if abs(_m[2]) <= _df]
    _keep = (spec_f > 0.3) & (spec_f < 20.0)
    _pk, _props = find_peaks(spec_amp[_keep], height=0)
    _tallest = float(spec_f[_keep][_pk[np.argmax(_props["peak_heights"])]])
    _missed = [_fm for _fm in modes_hour if all(abs(_m[0] - _fm) > 1e-12 for _m in _within)]
    _amp_at = {float(_fm): spec_amp[np.argmin(np.abs(spec_f - _fm))] for _fm in modes_hour}
    print(f"{len(_tail) / passage.fs:.1f} s of ringing; {_df:.4f} Hz between frequencies")
    print("modes that hour (Hz):", np.round(modes_hour, 3))
    print("mode, nearest peak, difference:", [tuple(round(_v, 3) for _v in _m) for _m in _matched])
    assert round(len(_tail) / passage.fs) == 29
    if ref("spectrum") is not None:
        assert str(times[_hour]) == ref("spectrum/hour")
        assert len(modes_hour) == len(ref("spectrum/modes_that_hour"))
        assert len(_within) == ref("spectrum/within_bin")
    check(_df, "spectrum/resolution_hz")
    for _i, _fm in enumerate(modes_hour):
        check(_fm, f"spectrum/modes_that_hour/{_i}")
    check(max(abs(_m[2]) for _m in _within), "spectrum/within_bin_max")
    assert len(_missed) == 1 and len(_within) == len(modes_hour) - 1  # the seventh
    assert _amp_at[float(_missed[0])] < 0.1 * min(
        _v for _f, _v in _amp_at.items() if _f != _missed[0]
    )
    check(_tallest, "spectrum/tallest_hz")
    assert abs(_tallest - modes_hour[3]) < _df  # the operator's mode the tallest peak matches
    check(passage.fs / 2, "spectrum/nyquist_hz")
    # Peaks above 7 Hz are modes the operator does not track: all 14 tracked modes sit below
    assert np.any(spec_f[_keep][_pk] > 7.0) and np.all(np.nanmedian(freqs, axis=0) < 7.0)
    return modes_hour, spec_amp, spec_f


@app.cell
def _(PALETTE, modes_hour, plt, spec_amp, spec_f):
    _fig, _ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    _k = spec_f < 12
    _ax.semilogy(spec_f[_k], spec_amp[_k], color=PALETTE[0], lw=1)
    for _fm in modes_hour:
        _ax.axvline(_fm, color=PALETTE[1], ls="--", lw=1)
    _ax.set_xlabel("frequency (Hz); dashed: the operator's modes, same hour")
    _ax.set_ylabel("amplitude (m/s²)")
    _fig
    return


@app.cell
def _(find_peaks, np, passage, shown, spectrogram):
    # The spectrum of each 4-second window, slid every half second: time against frequency,
    # power in dB. Longer windows separate frequencies better and blur time.
    _nper = int(4 * passage.fs)
    _f, _t, _sxx = spectrogram(
        passage.acc, passage.fs, window="hann", nperseg=_nper, noverlap=_nper * 7 // 8
    )
    _db = 10 * np.log10(_sxx + 1e-12)
    _k = _f <= 20
    _before = _t + 2 < passage.on[0]  # whole windows before the train arrives
    _during = (_t - 2 > passage.on[0]) & (_t + 2 < passage.on[1])
    _after = _t - 2 > passage.on[1]
    _brighter = float(np.mean(_db[_k][:, _during].mean(1) > _db[_k][:, _before].mean(1)))
    _first, _last = _db[_k][:, _after][:, :4].mean(1), _db[_k][:, _after][:, -4:].mean(1)
    _ring = _f[_k][np.argmax(_first)]  # the strongest line once the train has left
    # The strain gauge (another sensor) shows the train: a row of humps as its axles pass
    _s = np.abs(passage.strain)
    _humps, _ = find_peaks(_s, prominence=0.2 * _s.max(), distance=int(0.2 * passage.fs))
    print(
        f"{_nper / passage.fs:.2f}-s windows {_t[1] - _t[0]:.2f} s apart, "
        f"{_f[1] - _f[0]:.3f} Hz between frequencies; {len(_humps)} strain humps from "
        f"{_humps[0] / passage.fs:.1f} to {_humps[-1] / passage.fs:.1f} s"
    )
    print(
        f"brighter while the train crosses at {_brighter:.0%} of frequencies up to 20 Hz; "
        f"strongest line after it leaves: {_ring:.2f} Hz, fading "
        f"{_first[np.argmax(_first)]:.0f} -> {_last[np.argmax(_first)]:.0f} dB"
    )
    assert shown(_nper / passage.fs, "4.0") and shown(_f[1] - _f[0], "0.25")
    assert shown(_t[1] - _t[0], "0.5") and round(len(passage.acc) / passage.fs) == 52
    assert len(_humps) >= 5 and passage.on[0] < _humps[0] / passage.fs  # a row of humps, all
    assert _humps[-1] / passage.fs < passage.on[1]  # while the train is on (about 11 to 22 s)
    assert _brighter == 1.0  # bright across all frequencies while it crosses
    assert 4.0 <= _ring <= 4.5 and _last[np.argmax(_first)] < _first.max()  # lines near 4 Hz fade
    return


@app.cell
def _(mo):
    mo.md(r"""
    A spectrogram computed with dB for the power and a linear frequency axis: speech systems
    use a mel (logarithmic) frequency axis because human hearing is logarithmic in frequency; a
    bridge does not hear. The same two building blocks carry audio's models over to our
    records:

    | Audio | The recipe | On CEE records |
    |:------|:-----------|:---------------|
    | Sound classes | spectrogram → image CNN | bridge spectrograms → Lecture 15's ResNet |
    | Speech to text | CNN features → LSTM over time (Hannun et al., 2014) | weather → 1-D CNN, LSTM → river flow |
    | Foundation models | pretrained, used as is | Chronos-2 on the river (§5–§6) |

    ### 1-D CNN: Lecture 15's filter, along time

    **Assume: the same small pattern means the same thing on any day.** One filter applied at
    every day, summed over the input channels $c$:

    $$z_t = b + \sum_{c} \sum_{k=0}^{K-1} w_{c,k}\, x_{c,\,t-k}.$$

    Causal: the output at day $t$ sees only days up to $t$. PyTorch's `Conv1d` slides the
    filter forward (a cross-correlation), so it computes the formula with the filter flipped,
    and its output $j$ belongs to day $j + K - 1$. The check, on a real window of the six
    Daymet inputs standardized on training days:
    """)
    return


@app.cell
def _(F, FIRST_KERNEL, INPUTS, SEED, SEQ, cam, home, np, te, torch, tr):
    _w = cam.forcing[home][:, [cam.channel(_c) for _c in INPUTS]].astype(np.float64)
    _w = (_w - _w[tr].mean(0)) / _w[tr].std(0)
    _x = _w[te[0] - SEQ + 1 : te[0] + 1].T  # channels first: (6, 365)
    _rng = np.random.default_rng(SEED)
    _filt = _rng.normal(size=(len(INPUTS), FIRST_KERNEL)) / np.sqrt(len(INPUTS) * FIRST_KERNEL)
    _bias = 0.1
    _lags = np.arange(FIRST_KERNEL)
    # From scratch: z_t = b + sum_c sum_k w[c, k] x[c, t - k], for every day with a full window
    _z = np.array(
        [_bias + np.sum(_filt * _x[:, _t - _lags]) for _t in range(FIRST_KERNEL - 1, SEQ)]
    )
    _ref = F.conv1d(
        torch.from_numpy(_x)[None],
        torch.from_numpy(_filt[:, ::-1].copy())[None],  # flipped: PyTorch slides forward
        torch.tensor([_bias], dtype=torch.float64),
    )[0, 0].numpy()
    assert _x.shape == (len(INPUTS), SEQ)
    assert _z.shape == _ref.shape == (SEQ - FIRST_KERNEL + 1,)
    assert np.allclose(_z, _ref, rtol=0, atol=1e-12)
    print(f"from scratch vs conv1d: max difference {np.max(np.abs(_z - _ref)):.1e}")
    return


@app.cell
def _(DILATIONS, DROPOUT, FIRST_KERNEL, HIDDEN, nn):
    class LSTMNet(nn.Module):
        """An LSTM over the window, then a linear read-out of its last hidden state."""

        def __init__(self, n_in, hidden=HIDDEN):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            """(batch, seq, channels) -> (batch,)."""
            h, _ = self.lstm(x)
            return self.head(self.drop(h[:, -1])).squeeze(-1)

    class CNN1d(nn.Module):
        """16 filters a month long, then dilated filters; the read-out at the last day."""

        def __init__(self, n_in, width=32):
            super().__init__()
            layers = [nn.Conv1d(n_in, 16, FIRST_KERNEL), nn.ReLU()]
            c = 16
            for d in DILATIONS:
                layers += [nn.Conv1d(c, width, 3, dilation=d), nn.ReLU()]
                c = width
            self.net = nn.Sequential(*layers)
            self.drop = nn.Dropout(DROPOUT)
            self.head = nn.Linear(width, 1)

        def forward(self, x):
            """(batch, seq, channels) -> (batch,): channels first for Conv1d, the last day out."""
            z = self.net(x.transpose(1, 2))  # (batch, width, seq - 285 + 1)
            return self.head(self.drop(z[:, :, -1])).squeeze(-1)

    return CNN1d, LSTMNet


@app.cell
def _(CNN1d, DILATIONS, FIRST_KERNEL, HIDDEN, INPUTS, LSTMNet, SEQ, nn, torch):
    _cnn = CNN1d(len(INPUTS))
    _convs = [_m for _m in _cnn.net if isinstance(_m, nn.Conv1d)]
    _first = sum(_p.numel() for _p in _convs[0].parameters())
    _dilated = sum(_p.numel() for _m in _convs[1:] for _p in _m.parameters())
    _head = sum(_p.numel() for _p in _cnn.head.parameters())
    _field = FIRST_KERNEL + 2 * sum(DILATIONS)  # each dilated layer adds 2 x its spacing
    with torch.no_grad():
        _out = _cnn.net(torch.zeros(1, len(INPUTS), SEQ))
    _lstm = LSTMNet(len(INPUTS))
    _layer = sum(_p.numel() for _p in _lstm.lstm.parameters())
    print(
        f"1-D CNN: first layer {_first:,}, dilated {_dilated:,}, read-out {_head}, total "
        f"{_first + _dilated + _head:,}; sees {_field} days; LSTM layer {_layer:,} + read-out "
        f"{HIDDEN + 1} = {sum(_p.numel() for _p in _lstm.parameters()):,}"
    )
    assert (_first, _dilated, _head) == (2_992, 20_192, 33) and _first + _dilated + _head == 23_217
    assert _field == 285 and _field - FIRST_KERNEL == 254  # 31 days, then +254
    assert _out.shape == (1, 32, SEQ - _field + 1)  # one output per day with 285 days in view
    assert _layer == 4 * HIDDEN * (HIDDEN + len(INPUTS) + 2) == 18_432
    assert sum(_p.numel() for _p in _lstm.parameters()) == 18_432 + 65 == 18_497
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Layer | Sees | Weights |
    |:------|:-----|--------:|
    | 16 filters, 6 inputs × 31 days | 31 days | 2,992 |
    | 7 dilated layers, 3 taps, 1–64 apart | +254 days | 20,192 |
    | Read-out of the last day | | 33 |

    Dilation (van den Oord et al., 2016) spaces a filter's taps 1, 2, 4, ..., 64 days apart, so
    seven small layers see 285 days with 23,217 weights in all.

    ### An LSTM keeps a store

    **Assume: a few numbers carried from day to day can summarize the past.** Each day the LSTM
    (Hochreiter & Schmidhuber, 1997) reads yesterday's hidden state $h_{t-1}$ and today's
    weather $x_t$ into four gates, $\sigma$ for input $i$, forget $f$, and output $o$, $\tanh$
    for the candidate $g$, each $W[h_{t-1}, x_t] + b$; then

    $$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t).$$

    The cell state $c$ is a store: the forget gate keeps a share of it, the input gate adds a
    share of a new candidate, and the output gate decides how much shows. Like a basin's soil
    and groundwater, it keeps water from earlier rain and releases it later. Because $c$ is
    updated by addition, gradients flow back over long spans (Lecture 15's residual idea, in
    time). With 64 cells over 6 inputs: $4 \times 64 \times (64 + 6 + 2) = 18{,}432$ weights,
    plus a read-out of 65: 18,497 (checked above).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### Your turn: predict the check

    The loop below is the two equations, with PyTorch's gate order (input, forget, candidate,
    output) and its two bias vectors added. Run against `nn.LSTM` with the same weights, in
    float64, on one real 365-day window: how close? Write a number first.
    """)
    return


@app.cell
def _(torch):
    def lstm_scratch(x, lstm):
        """An LSTM layer written out, with nn.LSTM's weights (gate order i, f, g, o).

        x is one window, (seq, channels); returns the hidden and cell states, (seq, hidden).
        """
        w_ih, w_hh = lstm.weight_ih_l0, lstm.weight_hh_l0  # (4H, C), (4H, H)
        b = lstm.bias_ih_l0 + lstm.bias_hh_l0  # (4H,)
        n = lstm.hidden_size
        h = torch.zeros(n, dtype=x.dtype, device=x.device)
        c = torch.zeros_like(h)
        hs, cs = [], []
        for x_t in x:
            z = w_ih @ x_t + w_hh @ h + b  # four gates at once
            i, f = torch.sigmoid(z[:n]), torch.sigmoid(z[n : 2 * n])
            g, o = torch.tanh(z[2 * n : 3 * n]), torch.sigmoid(z[3 * n :])
            c = f * c + i * g  # keep some, add some
            h = o * torch.tanh(c)  # show some
            hs.append(h)
            cs.append(c)
        return torch.stack(hs), torch.stack(cs)

    return (lstm_scratch,)


@app.cell
def _(
    BATCH,
    CHECK_STEPS,
    CLIP,
    CNN1d,
    DEVICE,
    INPUTS,
    LR_LADDER,
    LR_START,
    LSTMNet,
    MAX_STEPS,
    MAX_STEPS_REDUCED,
    PATIENCE_STEPS,
    PREDICT_BATCH,
    SEQ,
    SLIDES,
    cam,
    dataclass,
    nn,
    np,
    nse,
    pd,
    ref,
    seed_all,
    te,
    time,
    torch,
    tr,
    va,
):
    @dataclass(frozen=True)
    class Task:
        """What a network sees and predicts.

        use_flow: past flow is an input (forecasting) or not (simulating from weather alone);
        horizon: days from the window's last day to the target; tomorrow_rain: one more input,
        the rain of the day after each window day (on the last day, the target day's rain: a
        perfect rain forecast).
        """

        use_flow: bool
        horizon: int
        tomorrow_rain: bool

    SIMULATE = Task(use_flow=False, horizon=0, tomorrow_rain=False)
    FORECAST = Task(use_flow=True, horizon=1, tomorrow_rain=False)
    FORECAST_RAIN = Task(use_flow=True, horizon=1, tomorrow_rain=True)

    @dataclass(frozen=True)
    class Basins:
        """Inputs and targets of one or more basins on the device."""

        x: torch.Tensor  # basins x days x channels, standardized; NaN where unknown
        y: torch.Tensor  # basins x days: flow / the basin's training SD; NaN where missing
        scale: np.ndarray  # each basin's training SD of flow, mm/day
        flow: np.ndarray  # basins x days, mm/day
        weather_sd: np.ndarray  # training SD of each weather input (float64)
        unknown: np.ndarray  # basins x (days + 1): running count of days with an unknown input
        task: Task
        gauges: tuple

    def basins(gauges, task):
        """Stack the basins' weather (and past flow, and tomorrow's rain) as network inputs.

        Weather is standardized with the training days' mean and SD pooled over the basins;
        flow is divided by each basin's own training SD, so the squared error weighs every
        basin alike (the NSE's denominator). Missing past flow is carried forward, never back
        from a later day; no window with an unknown input is used.
        """
        rows = [cam.basin(g) for g in gauges]
        chans = [cam.channel(c) for c in INPUTS]
        w = cam.forcing[rows][:, :, chans].astype(np.float64)
        mu = w[:, tr].reshape(-1, len(chans)).mean(0)
        sd = w[:, tr].reshape(-1, len(chans)).std(0)
        w = (w - mu) / sd
        flow_b = cam.flow[rows].astype(np.float64)
        scale = np.nanstd(flow_b[:, tr], axis=1)
        y = flow_b / scale[:, None]
        parts = [w]
        if task.use_flow:
            parts.append(pd.DataFrame(y.T).ffill().to_numpy().T[:, :, None])
        if task.tomorrow_rain:
            _rain = w[:, :, INPUTS.index("prcp_mm")]
            _next = np.concatenate([_rain[:, 1:], np.full((len(rows), 1), np.nan)], axis=1)
            parts.append(_next[:, :, None])
        x = np.concatenate(parts, axis=2)
        n_in = len(INPUTS) + task.use_flow + task.tomorrow_rain
        assert x.shape == (len(rows), len(cam.dates), n_in), x.shape
        bad = ~np.isfinite(x).all(axis=2)
        return Basins(
            x=torch.tensor(x, dtype=torch.float32, device=DEVICE),
            y=torch.tensor(y, dtype=torch.float32, device=DEVICE),
            scale=scale,
            flow=flow_b,
            weather_sd=sd,
            unknown=np.concatenate([np.zeros((len(rows), 1), int), np.cumsum(bad, axis=1)], axis=1),
            task=task,
            gauges=tuple(gauges),
        )

    def samples(data, days, seq):
        """(basin, last input day) pairs whose target day is in days, observed, inputs known."""
        h = data.task.horizon
        last = days - h
        last = last[last - seq + 1 >= 0]
        b, t = np.meshgrid(np.arange(data.x.shape[0]), last, indexing="ij")
        b, t = b.ravel(), t.ravel()
        known = data.unknown[b, t + 1] - data.unknown[b, t - seq + 1] == 0
        ok = known & np.isfinite(data.y[b, t + h].cpu().numpy())
        return torch.tensor(np.stack([b[ok], t[ok]], axis=1), device=data.x.device)

    def gather(data, pairs, seq):
        """Input windows (batch, seq, channels), oldest day first, and targets (batch,)."""
        offsets = torch.arange(seq - 1, -1, -1, device=pairs.device)
        days = pairs[:, 1:2] - offsets  # (batch, seq)
        x = data.x[pairs[:, :1], days]
        y = data.y[pairs[:, 0], pairs[:, 1] + data.task.horizon]
        return x, y

    @dataclass
    class Run:
        """A trained network and its scores."""

        model: nn.Module
        lr: float
        val_nse: float  # median over the basins, at the kept check
        test_nse: np.ndarray  # one per basin
        best_step: int  # the gradient step whose weights were kept
        steps: int  # gradient steps taken
        stopped: bool  # stopped by patience (True) or by the step cap
        seconds: float

    def predict(model, data, days, seq=SEQ):
        """Predicted flow (mm/day), basins x len(days); NaN where a day cannot be predicted."""
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

    def scores(model, data, days, seq=SEQ):
        """NSE of each basin over days; -inf for a basin whose output is not finite (diverged)."""
        pred = predict(model, data, days, seq)
        expected = np.bincount(samples(data, days, seq)[:, 0].cpu().numpy(), minlength=len(pred))
        out = []
        for b in range(len(pred)):
            good = np.isfinite(pred[b]) & (np.abs(np.nan_to_num(pred[b])) < 1e6)
            ok = good.sum() == expected[b] and expected[b] >= 2
            out.append(nse(data.flow[b, days], pred[b]) if ok else -np.inf)
        return np.array(out)

    def train(data, kind, seed, lr, *, days, max_steps, seq=SEQ):
        """Adam on the squared error of scaled flow, with early stopping on validation.

        Validation NSE (median over the basins) every CHECK_STEPS gradient steps, the same for
        any amount of data; stop PATIENCE_STEPS after the best check or at max_steps, and keep
        the best check's weights. seed seeds the weights, dropout, and batch order; days are
        the target days of training, validation, and test.
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
        while not stopped and step < max_steps:
            order = torch.randperm(len(pairs), generator=gen).to(pairs.device)
            for i in range(0, len(pairs), BATCH):
                model.train()
                x, y = gather(data, pairs[order[i : i + BATCH]], seq)
                loss = torch.mean((model(x) - y) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # rare storms, rare big steps
                opt.step()
                step += 1
                if step % CHECK_STEPS == 0:
                    val = float(np.median(scores(model, data, val_days, seq)))
                    if val > best:
                        best, best_step = val, step
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    elif step - best_step >= PATIENCE_STEPS:
                        stopped = True
                if stopped or step >= max_steps:
                    break
        model.load_state_dict(best_state)
        test = scores(model, data, test_days, seq)
        return Run(model, lr, best, test, best_step, step, stopped, time.perf_counter() - t0)

    _runs = {}  # every run of this notebook, by what defines it: a shared run trains once

    def run(data, kind, seed, lr, *, train_days):
        """One run on the standard validation and test years, trained once per setting."""
        max_steps = MAX_STEPS if SLIDES else MAX_STEPS_REDUCED[kind]
        key = (data.gauges, data.task, kind, seed, lr, int(train_days[0]), len(train_days))
        if key not in _runs:
            _runs[key] = train(data, kind, seed, lr, days=(train_days, va, te), max_steps=max_steps)
        return _runs[key]

    def bracket(score):
        """The learning rate with the highest score, bracketed along LR_LADDER.

        Starts from LR_START; while the best rate is the lowest or the highest tried, the next
        rate of the ladder past it is tried too. Asserts the kept rate is interior.
        """
        tried = {}
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
        """The runs at the kept learning rate, and the record of the search."""

        runs: list  # one per data and seed (data-major)
        record: dict

    def tune(datas, kind, seeds, *, rate_path, train_days=None):
        """One learning rate for a set of models, then its runs.

        "slides": bracketed on validation (median over datas of the median over seeds); every
        run at the kept rate and its neighbours must stop by patience, and every run at the
        kept rate must have its best check after the first. "reduced": the rate the slides'
        search kept (numbers.json at rate_path), or LR_START's middle.
        """
        days = tr if train_days is None else train_days
        by_rate = {}

        def score(lr):
            by_rate[lr] = [run(d, kind, s, lr, train_days=days) for d in datas for s in seeds]
            per = np.reshape([r.val_nse for r in by_rate[lr]], (len(datas), len(seeds)))
            return float(np.median(np.median(per, axis=1)))

        if not SLIDES:
            lr = ref(rate_path) if ref(rate_path) is not None else LR_START[1]
            score(lr)
            return Tuned(by_rate[lr], {"lr": lr})
        lr, tried = bracket(score)
        rates = sorted(tried)
        near = rates[rates.index(lr) - 1 : rates.index(lr) + 2]
        capped = [(r.lr, r.best_step) for q in near for r in by_rate[q] if not r.stopped]
        assert not capped, f"runs that hit MAX_STEPS near the kept rate {lr:g}: {capped}"
        _first = [r.best_step for r in by_rate[lr] if r.best_step <= CHECK_STEPS]
        assert not _first, f"kept runs at rate {lr:g} whose best check is the first: {_first}"
        record = {
            "lr": lr,
            "tried": {
                f"{q:g}": {
                    "val": tried[q],
                    "best_steps": [r.best_step for r in by_rate[q]],
                    "steps": [r.steps for r in by_rate[q]],
                }
                for q in rates
            },
        }
        return Tuned(by_rate[lr], record)

    def summary(t):
        """Median and range over seeds of each run's test NSE (its median over basins)."""
        s = np.array([float(np.median(r.test_nse)) for r in t.runs])
        return {
            "test_median": float(np.median(s)),
            "test_min": float(s.min()),
            "test_max": float(s.max()),
            "test": s.tolist(),
            "best_steps": [r.best_step for r in t.runs],
            "seconds": float(np.mean([r.seconds for r in t.runs])),
            **t.record,
        }

    def median_run(rs):
        """The run whose test score is the median of the seeds'."""
        t = [float(np.median(r.test_nse)) for r in rs]
        return rs[int(np.argsort(t)[len(t) // 2])]

    return (
        FORECAST,
        FORECAST_RAIN,
        SIMULATE,
        basins,
        median_run,
        predict,
        run,
        summary,
        tune,
    )


@app.cell
def _(HIDDEN, HOME, INPUTS, SEED, SEQ, SIMULATE, basins, lstm_scratch, nn, seed_all, te, torch):
    sim = basins([HOME], SIMULATE)  # weather alone: six inputs, no flow
    verify_day = int(te[400])  # one real window, ending on a test day
    # First on fresh weights: the loop against nn.LSTM, float64, on the CPU
    seed_all(SEED)
    _lstm = nn.LSTM(len(INPUTS), HIDDEN, batch_first=True).double()
    _x = sim.x[0, verify_day - SEQ + 1 : verify_day + 1].double().cpu()  # (365, 6)
    with torch.no_grad():
        _h_ref, (_, _c_ref) = _lstm(_x[None])
        _h, _c = lstm_scratch(_x, _lstm)
    _dh, _dc = float((_h - _h_ref[0]).abs().max()), float((_c[-1] - _c_ref[0, 0]).abs().max())
    print(f"fresh weights: hidden states differ by {_dh:.1e}, last cell state by {_dc:.1e}")
    assert _x.shape == (SEQ, len(INPUTS)) and _h.shape == (SEQ, HIDDEN)
    assert _dh < 1e-13 and _dc < 1e-13
    return sim, verify_day


@app.cell
def _(mo):
    mo.md(r"""
    ### Train the 1-D CNN and the LSTM on weather alone

    Inputs: the six Daymet variables, standardized on training days; target: the day's flow
    divided by its training standard deviation, from weather alone (no flow input). Adam,
    batches of 256 windows of 365 days, gradient norm clipped at 1, dropout 0.2 before the
    read-out. **The training budget is counted in gradient steps**, the same whatever the
    amount of data: validation NSE every 5 steps, the run stops 600 steps after its best
    check and keeps that check's weights. **The learning rate is chosen on validation by a
    bracketing search**: start from $10^{-3}$, $3 \times 10^{-3}$, and $10^{-2}$; while the best
    rate is the lowest or the highest tried, try the next rate of a fixed ladder past it, so
    the kept rate always has a worse rate on each side. The ladder ends at 1, where training
    fails (checked below). In the `"slides"` setting every run at the kept rate and at both its
    neighbours must stop by patience, not by the 12,000-step cap, and every run at the kept
    rate must have its best check after the first. Five seeds each.
    """)
    return


@app.cell
def _(CHECK_NETS, NET_TOL, REFERENCE, check, ref):
    def check_net(s, path):
        """A network's summary against the slides' (numbers.json at path), or printed beside it.

        The kept rate and the runs' kept steps only on the reference GPU; test NSE within
        NET_TOL in the "slides" setting.
        """
        _slides = ref(f"{path}/test_median")
        print(
            f"{path}: rate {s['lr']:g}; test NSE median {s['test_median']:.4f} "
            f"({s['test_min']:.4f} to {s['test_max']:.4f}), steps kept {s['best_steps']}"
            + (f"; the slides: {_slides:.4f} at rate {ref(path + '/lr'):g}" if _slides else "")
        )
        if not CHECK_NETS:
            return
        for _k in ("test_median", "test_min", "test_max"):
            check(s[_k], f"{path}/{_k}", NET_TOL)
        if REFERENCE:
            assert s["lr"] == ref(f"{path}/lr") and s["best_steps"] == ref(f"{path}/best_steps")
            for _i, _v in enumerate(s["test"]):
                check(_v, f"{path}/test/{_i}", NET_TOL)

    return (check_net,)


@app.cell
def _(SIM_SEEDS, sim, summary, time, tune):
    _t0 = time.perf_counter()
    lstm_t = tune([sim], "lstm", SIM_SEEDS, rate_path="lstm_sim/lr")
    cnn_t = tune([sim], "cnn", SIM_SEEDS, rate_path="cnn_sim/lr")
    lstm_runs, cnn_runs = lstm_t.runs, cnn_t.runs
    lstm_sim, cnn_sim = summary(lstm_t), summary(cnn_t)
    for _name, _s in (("LSTM", lstm_sim), ("1-D CNN", cnn_sim)):
        _tried = {_k: round(_v["val"], 4) for _k, _v in _s.get("tried", {}).items()}
        print(f"{_name}: validation NSE by rate {_tried}; {_s['seconds']:.1f} s a run")
    print(f"{time.perf_counter() - _t0:.0f} s")
    return cnn_runs, cnn_sim, lstm_runs, lstm_sim


@app.cell
def _(
    CHECK_NETS,
    FORECAST,
    HOME,
    LR_LADDER,
    REFERENCE,
    SEED_LADDER,
    SIMULATE,
    SLIDES,
    basins,
    check,
    np,
    ref,
    run,
    tr,
):
    # The ladder's last rate fails: validation NSE below 0 (worse than the mean), or an output
    # that is not finite (-inf here, null in numbers.json). Only in the "slides" setting.
    if SLIDES:
        for _name, _task, _kind in (
            ("lstm_sim", SIMULATE, "lstm"),
            ("cnn_sim", SIMULATE, "cnn"),
            ("lstm_forecast", FORECAST, "lstm"),
        ):
            _r = run(basins([HOME], _task), _kind, SEED_LADDER, LR_LADDER[-1], train_days=tr)
            print(f"{_name} at rate {LR_LADDER[-1]:g}: validation NSE {_r.val_nse:.3f}")
            assert _r.val_nse < 0
            if CHECK_NETS and REFERENCE:
                if np.isfinite(_r.val_nse):
                    check(_r.val_nse, f"ladder_end/val_nse/{_name}")
                else:
                    assert ref(f"ladder_end/val_nse/{_name}") is None
    return


@app.cell
def _(CHECK_NETS, check_net, cnn_sim, flow, lstm_runs, lstm_sim, np, te, uh, va):
    check_net(lstm_sim, "lstm_sim")
    check_net(cnn_sim, "cnn_sim")
    if CHECK_NETS:  # both networks beat the filter; test scores sit below validation's
        assert min(lstm_sim["test_median"], cnn_sim["test_median"]) > uh["test_nse"]
        _val = np.median([_r.val_nse for _r in lstm_runs])
        assert _val > lstm_sim["test_median"]  # the medians over seeds
    # The test years are drier (they include the 2012 drought)
    assert np.nanmean(flow[te]) < np.nanmean(flow[va])
    return


@app.cell
def _(SEQ, check, lstm_runs, lstm_scratch, median_run, nn, sim, torch, verify_day):
    # The deck's check: the median run's trained weights, float64, on the CPU (both sides)
    lstm_best = median_run(lstm_runs)
    _m = lstm_best.model.lstm
    _lstm = nn.LSTM(_m.input_size, _m.hidden_size, batch_first=True)
    _lstm.load_state_dict(_m.state_dict())
    _lstm = _lstm.double().cpu()
    _x = sim.x[0, verify_day - SEQ + 1 : verify_day + 1].double().cpu()
    with torch.no_grad():
        _h_ref, (_, _c_ref) = _lstm(_x[None])
        _h, _c = lstm_scratch(_x, _lstm)
    _dh, _dc = float((_h - _h_ref[0]).abs().max()), float((_c[-1] - _c_ref[0, 0]).abs().max())
    _weights = sum(_p.numel() for _p in lstm_best.model.parameters())
    print(f"hidden states, all days, 64 cells: {_dh:.1e}; last cell state: {_dc:.1e}")
    print(f"weights: {_weights:,}")
    # Round-off: the digits depend on the order of additions, so assert their size only
    assert _dh < 1e-14 and _dc < 1e-14
    check(_weights, "verify/weights", 0)
    verify_diffs = (_dh, _dc)
    return lstm_best, verify_diffs


@app.cell
def _(mo, verify_diffs):
    mo.accordion(
        {
            "Answer": mo.md(rf"""
        Float64 round-off over 365 steps, about $10^{{-16}}$ (here {verify_diffs[0]:.1e} for
        the hidden states and {verify_diffs[1]:.1e} for the last cell state). Not exactly zero:
        PyTorch adds the terms in another order, so the last bit can differ. Both sides ran in
        float64 on the CPU, with the weights the network learned in float32 on the GPU.
        """)
        }
    )
    return


@app.cell
def _(
    CHECK_NETS,
    NET_TOL,
    PALETTE,
    SEQ,
    check,
    cnn_runs,
    dates,
    flow,
    lstm_best,
    median_run,
    np,
    nse,
    plt,
    predict,
    sim,
    uh,
):
    # Water year 2013 from weather alone: the filter, the 1-D CNN, and the LSTM
    _cnn_best = median_run(cnn_runs)
    _days = np.arange(SEQ, len(dates))
    sims = {"unit hydrograph": uh["sim"]}
    for _name, _run in (("1-D CNN", _cnn_best), ("LSTM", lstm_best)):
        sims[_name] = np.full(len(dates), np.nan)
        sims[_name][_days] = predict(_run.model, sim, _days, SEQ)[0]
    _wy = (dates >= np.datetime64("2012-10-01")) & (dates <= np.datetime64("2013-09-30"))
    _apr = (dates >= np.datetime64("2013-04-01")) & (dates <= np.datetime64("2013-04-30"))
    _wy_nse = {_k: nse(flow[_wy], _s[_wy]) for _k, _s in sims.items()}
    print("NSE over water year 2013:", {_k: round(_v, 2) for _k, _v in _wy_nse.items()})
    print(
        "test NSE of the median runs:",
        round(_cnn_best.test_nse[0], 3),
        round(lstm_best.test_nse[0], 3),
    )
    if CHECK_NETS:  # the figure's runs are the median seeds'
        check(float(_cnn_best.test_nse[0]), "cnn_sim/test_median", NET_TOL)
        check(float(lstm_best.test_nse[0]), "lstm_sim/test_median", NET_TOL)
        # The networks follow the year more closely than the filter; all three miss part of
        # the April peak
        assert min(_wy_nse["LSTM"], _wy_nse["1-D CNN"]) > _wy_nse["unit hydrograph"]
        assert all(np.nanmax(_s[_apr]) < np.nanmax(flow[_apr]) for _s in sims.values())
    _fig, _ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    _ax.plot(dates[_wy], flow[_wy], color="black", lw=1.6, label="observed")
    for _i, (_k, _s) in enumerate(sims.items()):
        _ax.plot(dates[_wy], _s[_wy], ("--", "-.", ":")[_i], color=PALETTE[_i + 1], label=_k)
    _ax.set_ylabel("flow (mm/day)")
    _ax.legend(frameon=False, loc="upper left")
    _fig
    return (sims,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The LSTM's own unit hydrograph, on a wet and a dry day

    The gradient of the LSTM's simulated flow on one day with respect to each past day's rain,
    in mm/day of flow per mm of rain: how much a millimetre more rain $k$ days ago would have
    added, given everything else the network saw. cuDNN's LSTM has no backward pass in
    evaluation mode, so cuDNN is switched off for this one computation.
    """)
    return


@app.cell
def _(
    CHECK_NETS,
    INPUTS,
    NET_REL,
    PALETTE,
    SEQ,
    check,
    dates,
    lstm_best,
    np,
    pd,
    plt,
    sim,
    torch,
    uh,
):
    def rain_response(run_, day):
        """d(simulated flow that day) / d(rain k days before), in mm/day per mm, lag 0 first.

        The rain's training SD (float64, the one that standardized it) undoes its scaling.
        """
        t = int(np.flatnonzero(dates == np.datetime64(day))[0])
        k = INPUTS.index("prcp_mm")
        x = sim.x[0, t - SEQ + 1 : t + 1].clone()[None].requires_grad_(True)  # (1, 365, 6)
        run_.model.eval()
        with torch.backends.cudnn.flags(enabled=False):  # cuDNN's RNN: no backward in eval
            y = run_.model(x)
        (grad,) = torch.autograd.grad(y.sum(), x)
        return grad[0, :, k].cpu().numpy()[::-1] * sim.scale[0] / sim.weather_sd[k]

    response_days = {"wet": "2013-04-11", "dry": "2012-08-27"}
    response = {_tag: rain_response(lstm_best, _d) for _tag, _d in response_days.items()}
    for _tag, _g in response.items():
        print(
            f"{_tag}, {response_days[_tag]}: lags 0-9 {np.round(_g[:10], 3)}; today "
            f"{_g[0]:.3f}, lags 1-29 {_g[1:30].sum():.2f}, 30-day sum {_g[:30].sum():.2f}"
        )
        if CHECK_NETS:
            _d = response_days[_tag]
            check(float(_g[:30].sum()), f"response/{_d}/sum30", NET_REL, rel=True)
            check(float(_g[0]), f"response/{_d}/lag0", NET_REL, rel=True)
            check(float(_g[1:30].sum()), f"response/{_d}/lags1_29", NET_REL, rel=True)
    if CHECK_NETS:  # today's rain counts on both days; the month before only when wet
        _wet, _dry = response["wet"], response["dry"]
        assert min(_wet[0], _dry[0]) > 0.1
        assert _wet[1:30].sum() > 0.5 and abs(_dry[1:30].sum()) < 0.1
    _fig, _ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    _lags = np.arange(31)
    _ax.plot(_lags, uh["weights"][:31], ":o", color="grey", ms=3, label="the filter: every day")
    for _i, (_tag, _g) in enumerate(response.items()):
        _label = f"LSTM, {_tag}: {pd.Timestamp(response_days[_tag]):%-d %b %Y}"
        _ax.plot(_lags, _g[:31], ("-o", "--s")[_i], color=PALETTE[_i], ms=3, label=_label)
    _ax.axhline(0, color="grey", lw=0.6)
    _ax.set_xlabel("days since the rain")
    _ax.set_ylabel("mm/day of flow per mm of rain")
    _ax.legend(frameon=False)
    _fig
    return (response,)


@app.cell
def _(mo, response, uh):
    _wet, _dry = response["wet"], response["dry"]
    mo.md(rf"""
    The filter's weights sum to {uh["weights"].sum():.2f} on every day of the year. The LSTM's
    response to today's rain is {_wet[0]:.2f} on the wet day and {_dry[0]:.2f} on the dry one;
    to the rain of the 29 days before, {_wet[1:30].sum():.2f} against {_dry[1:30].sum():.2f}
    mm/day per mm. Today's rain counts on both days; the rain of the month before reaches the
    river on the wet day and nets to almost nothing on the dry one. The same network,
    with the same weights, answers the attempt: what it carries from day to day (its store)
    differs between a wet basin and a dry one.
    """)
    return


@app.cell
def _(chosen, mo):
    mo.md(rf"""
    ## 5. Break it

    ### Your turn: predict four failures

    1. Score the forest on random days instead of later years.
    2. Storm B: the river was lower than on any training day.
    3. Train with one year of record instead of fifteen.
    4. Alarm on the bridge's frequency through a winter.

    Write what happens to each, and why. Each subsection below reveals one.

    ### 5.1 Random splits: a lottery, not a leak

    The same forest and settings ({chosen[2]}-day windows, the {chosen[1]} as the target); only
    the split changes: ten draws of random test days from water years 1990–2014, as many as the
    test years have; ten draws of random whole water years, 6 of the 25 (no test day then has a
    neighbouring day in training); and the later years, 2009–2014. Then the same forest on rain
    alone (to tomorrow's), with no flow input: it simulates the river rather than forecasting it.
    """)
    return


@app.cell
def _(
    SEED,
    SPLIT_DRAWS,
    check,
    chosen,
    chosen_forecast,
    dates,
    flow,
    issued,
    np,
    nse,
    te,
    tr,
    va,
):
    def water_year(d):
        """The water year of each date (1 October to 30 September, named by its end)."""
        _year = d.astype("datetime64[Y]").astype(int) + 1970
        return _year + (d.astype("datetime64[M]").astype(int) % 12 + 1 >= 10)

    _targets = np.concatenate([tr, va, te])  # every target day of WY1990-2014
    _days = issued(_targets)  # the days the forecasts are made on
    _fit_t, _at_test = issued(np.concatenate([tr, va])), issued(te)
    _wy = water_year(dates[_targets])
    _years = np.unique(_wy)
    _n_years = len(np.unique(water_year(dates[te])))  # as many as the later years: 6 of 25
    assert (len(_years), _n_years) == (25, 6)
    split = {}
    for _name, _setting, _kw in (
        ("with_flow", chosen, {}),  # the chosen forest: it sees today's flow
        ("rain_only", (chosen[0], "level", chosen[2]), {"with_flow": False, "tomorrow_rain": True}),
    ):
        _rng = np.random.default_rng(SEED)
        _draws = []
        for _ in range(SPLIT_DRAWS):  # random days
            _test = np.sort(_rng.choice(_days, size=len(te), replace=False))
            _fit = np.setdiff1d(_days, _test)
            _draws.append(nse(flow[_test + 1], chosen_forecast(_setting, _fit, _test, **_kw)))
        _rng = np.random.default_rng(SEED)
        _year_draws = []
        for _ in range(SPLIT_DRAWS):  # random whole years: no test day's neighbour in training
            _held = np.isin(_wy, _rng.choice(_years, size=_n_years, replace=False))
            _f = chosen_forecast(_setting, issued(_targets[~_held]), issued(_targets[_held]), **_kw)
            _year_draws.append(nse(flow[_targets[_held]], _f))
        split[_name] = {
            "random_days": _draws,
            "random_years": _year_draws,
            "later_years": nse(flow[te], chosen_forecast(_setting, _fit_t, _at_test, **_kw)),
        }
        _s = split[_name]
        print(
            f"{_name}: later years {_s['later_years']:.3f}; random days: median "
            f"{np.median(_draws):.3f}, {min(_draws):.3f} to {max(_draws):.3f}; random years: "
            f"median {np.median(_year_draws):.3f}, {min(_year_draws):.3f} to {max(_year_draws):.3f}"
        )
        for _key in ("random_days", "random_years"):
            for _i, _v in enumerate(_s[_key]):
                check(_v, f"split/{_name}/{_key}/{_i}")
        check(_s["later_years"], f"split/{_name}/later_years")
    _w, _r = split["with_flow"], split["rain_only"]
    # With today's flow, random days score like the later years; the draw moves it widely
    assert abs(np.median(_w["random_days"]) - _w["later_years"]) < 0.02
    assert max(_w["random_days"]) - min(_w["random_days"]) > 0.15
    # Rain alone, no flow: every random draw, of days or of whole years, scores above the later
    # years, and random whole years (no neighbour of a test day in training) score about the
    # same as random days: the gap is the later years, which are harder, not a leak
    assert min(_r["random_days"] + _r["random_years"]) > _r["later_years"]
    assert abs(np.median(_r["random_years"]) - np.median(_r["random_days"])) < 0.02
    assert np.median(_r["random_years"]) > _r["later_years"] + 0.05
    # "Drier": the later years' mean flow is below the earlier years' (0.89 against 1.17 mm/day)
    assert np.nanmean(flow[te]) < 0.8 * np.nanmean(flow[np.concatenate([tr, va])])
    return (split,)


@app.cell
def _(mo, np, split):
    _w, _r = split["with_flow"], split["rain_only"]
    mo.md(rf"""
    With today's flow as an input, random days score like the later years (median
    {np.median(_w["random_days"]):.2f} against {_w["later_years"]:.2f}; random years
    {np.median(_w["random_years"]):.2f}), but one draw reports {min(_w["random_days"]):.2f} and
    another {max(_w["random_days"]):.2f}: which floods land in the test set decides the score.
    On rain alone, every random-day draw scores above the later years
    ({min(_r["random_days"]):.2f} to {max(_r["random_days"]):.2f} against
    {_r["later_years"]:.2f}). Is that neighbouring days, sharing a storm, on both sides of the
    split? Random whole years leave no neighbour of a test day in training, and they score the
    same (median {np.median(_r["random_years"]):.2f} against {np.median(_r["random_days"]):.2f}
    for random days): no leak shows. The gap is the later years themselves, drier (the 2012
    drought) and harder for this model, and a random split hides them among easier years. A
    forecast is used on days after its training: test on later years, and report how much the
    score moves.

    ### 5.2 The drought no training day saw

    Storm B, August 2012, against models that also see today's flow, all trained on the same
    15 years (1990–2004): the forest from §3, refitted on them, and an LSTM given past flow as a
    seventh input (three seeds); then the same LSTM given tomorrow's rain as an eighth.
    """)
    return


@app.cell
def _(
    FC_SEEDS,
    FORECAST,
    FORECAST_RAIN,
    HOME,
    basins,
    check_net,
    median_run,
    predict,
    summary,
    te,
    time,
    tune,
):
    _t0 = time.perf_counter()
    fc_data = {_task: basins([HOME], _task) for _task in (FORECAST, FORECAST_RAIN)}
    _paths = {FORECAST: "lstm_forecast", FORECAST_RAIN: "lstm_rain"}
    fc_tuned = {
        _task: tune([_d], "lstm", FC_SEEDS, rate_path=f"{_paths[_task]}/lr")
        for _task, _d in fc_data.items()
    }
    lstm_forecast, lstm_rain = summary(fc_tuned[FORECAST]), summary(fc_tuned[FORECAST_RAIN])
    # Day-ahead forecasts of the median runs for every test day (made the day before)
    lstm_test, lstm_rain_test = (
        predict(median_run(fc_tuned[_task].runs).model, fc_data[_task], te)[0]
        for _task in (FORECAST, FORECAST_RAIN)
    )
    check_net(lstm_forecast, "lstm_forecast")
    check_net(lstm_rain, "lstm_rain")
    print(f"{time.perf_counter() - _t0:.0f} s")
    return fc_data, fc_tuned, lstm_forecast, lstm_rain, lstm_rain_test, lstm_test


@app.cell
def _(
    CHECK_NETS,
    NET_REL,
    ROUNDOFF,
    STORMS,
    check,
    flow,
    forest15_test,
    lstm_test,
    np,
    sims,
    storm_days,
    te,
    tr,
    uh,
):
    _days = storm_days[STORMS[1]]
    _i = _days - te[0]  # positions of the storm's days among the test days
    _peaks = {
        "observed": float(np.nanmax(flow[_days])),
        "filter on rain": float(np.max(uh["sim"][_days])),
        "LSTM, weather alone": float(np.nanmax(sims["LSTM"][_days])),
        "forest, with today's flow": float(np.nanmax(forest15_test[_i])),
        "LSTM, with today's flow": float(np.nanmax(lstm_test[_i])),
    }
    _lowest = float(np.nanmin(flow[tr]))
    print({_k: round(_v, 3) for _k, _v in _peaks.items()})
    print(f"flow before storm B {flow[_days[0] - 1]:.4f} mm/day; lowest training day {_lowest:.4f}")
    assert flow[_days[0] - 1] < _lowest
    check(_lowest, "storms/lowest_training_flow", ROUNDOFF)
    check(_peaks["forest, with today's flow"], "alerts/storm_b_peaks/forest")
    if CHECK_NETS:
        check(_peaks["LSTM, weather alone"], f"storms/lstm_peaks/{STORMS[1]}", NET_REL, rel=True)
        check(_peaks["LSTM, with today's flow"], "alerts/storm_b_peaks/LSTM", NET_REL, rel=True)
        # The models that see only the weather overshoot more than tenfold; those that see
        # today's flow come within a quarter of their peaks (the forest even falls short)
        _blind = min(_peaks["filter on rain"], _peaks["LSTM, weather alone"])
        _seeing = max(_peaks["forest, with today's flow"], _peaks["LSTM, with today's flow"])
        assert _blind > 10 * _peaks["observed"] and _seeing < 0.25 * _blind
    return


@app.cell
def _(mo):
    mo.md(r"""
    The models that see only the weather make a flood of storm B, more than ten times the
    river's small rise; those that see today's flow peak at under a quarter of the blind
    models' peaks (the forest even falls short of the river). Storm B began below every
    training day's flow: Lecture
    14's lesson in time, a model asked about conditions it never saw, and a drought and a
    flood are exactly where an alert matters.

    ### 5.3 One year of record: borrow a model

    Every model forecasts tomorrow from past flow and rain, trained on the last 1, 2, 5, or 15
    water years before 2005, with the same validation and test years. The LSTM's learning rate
    is bracketed again at each length, with the same stopping rule, counted in gradient steps
    (validation every 5, stop 600 after the best), and its inputs standardized with the 15
    training years' statistics at every length; the forest keeps the setting chosen on 15
    years. Chronos-2 (Ansari et al., 2025), a transformer pretrained on many public time
    series, is used as is: given the last days of flow and rain (how many, chosen on
    validation) and asked for tomorrow's quantiles. No training at all. Loading it and the
    forecasts take a few minutes on a GPU, longer on a CPU.
    """)
    return


@app.cell
def _(BaseChronosPipeline, CHRONOS, CHRONOS_REVISION, DEVICE, check, flow, np, rain, torch):
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS, revision=CHRONOS_REVISION, device_map=str(DEVICE)
    )

    def chronos_forecast(days, context, *, with_rain, tomorrow_rain=False, levels=(0.5,)):
        """Tomorrow's flow from the last context days of flow (and rain), no training.

        Returns quantile forecasts of flow t + 1 for each day t, shape (len(days), len(levels)).
        """
        inputs = []
        for t in days:
            item = {"target": flow[t - context + 1 : t + 1].astype(np.float32)}
            if with_rain:
                item["past_covariates"] = {"rain": rain[t - context + 1 : t + 1].astype(np.float32)}
            if tomorrow_rain:
                item["future_covariates"] = {"rain": rain[t + 1 : t + 2].astype(np.float32)}
            inputs.append(item)
        with torch.no_grad():
            q, _ = pipe.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(levels))
        return np.stack([qi[0, 0].float().cpu().numpy() for qi in q])

    chronos_weights = sum(_p.numel() for _p in pipe.model.parameters())
    print(f"Chronos-2: {chronos_weights / 1e6:.1f} million weights")
    check(chronos_weights, "chronos/weights", 0)
    return chronos_forecast, pipe


@app.cell
def _(
    CHRONOS_TOL,
    CHRONOS_STRIDE_REDUCED,
    CONTEXTS,
    SLIDES,
    check,
    chronos_forecast,
    flow,
    issued,
    np,
    nse,
    ref,
    te,
    time,
    va,
):
    _t0 = time.perf_counter()
    # In the slides' setting, the context is chosen on validation among CONTEXTS (inside the
    # grid), on every test day, three variants. In the reduced setting: the slides' context, every
    # CHRONOS_STRIDE_REDUCED-th test day, and no forecast from flow alone (Chronos-2 is slow on
    # a CPU); its test NSE is then over that sample of days, printed beside the slides'.
    if SLIDES or ref("chronos/context") is None:
        _val = {
            _c: nse(flow[va], chronos_forecast(issued(va), _c, with_rain=True)[:, 0])
            for _c in CONTEXTS
        }
        context = max(_val, key=_val.get)
        assert context not in (CONTEXTS[0], CONTEXTS[-1]), "the context is at the grid's edge"
    else:
        _val, context = {}, ref("chronos/context")
    _variants = {
        "flow": {"with_rain": False},
        "rain": {"with_rain": True},
        "tomorrow_rain": {"with_rain": True, "tomorrow_rain": True},
    }
    if not SLIDES:
        del _variants["flow"]
    _pos = np.arange(0, len(te), 1 if SLIDES else CHRONOS_STRIDE_REDUCED)
    chronos_test = {}  # one forecast per test day; NaN on the days not forecast
    for _k, _kw in _variants.items():
        chronos_test[_k] = np.full(len(te), np.nan)
        chronos_test[_k][_pos] = chronos_forecast(issued(te[_pos]), context, **_kw)[:, 0]
    chronos_nse = {_k: nse(flow[te[_pos]], _f[_pos]) for _k, _f in chronos_test.items()}
    print(f"validation NSE by context: { {_c: round(_v, 4) for _c, _v in _val.items()} }")
    print(
        f"context {context} days; test NSE on {len(_pos)} days:",
        {_k: round(_v, 4) for _k, _v in chronos_nse.items()},
    )
    print(f"{time.perf_counter() - _t0:.0f} s")
    check(context, "chronos/context", 0)
    for _c, _v in _val.items():
        check(_v, f"chronos/val_nse/{_c}", CHRONOS_TOL)
    if SLIDES:
        for _k, _v in chronos_nse.items():
            check(_v, f"chronos/test/{_k}", CHRONOS_TOL)
    elif ref("chronos/test/rain") is not None:
        print(
            "the slides, every test day:",
            {_k: round(ref(f"chronos/test/{_k}"), 4) for _k in chronos_nse},
        )
    return chronos_nse, chronos_test, context


@app.cell
def _(
    FC_SEEDS,
    FORECAST,
    YEARS_RUN,
    check,
    check_net,
    chosen,
    chosen_forecast,
    dates,
    fc_data,
    fc_tuned,
    flow,
    issued,
    lstm_forecast,
    np,
    nse,
    summary,
    te,
    time,
    tr,
    tune,
):
    _t0 = time.perf_counter()
    data_size = {"lstm": {}, "forest": {}}
    for _n in YEARS_RUN:
        _days = tr[dates[tr] >= np.datetime64(f"{2004 - _n}-10-01")]  # the last n water years
        if _n == 15:  # all of the training years: the forecaster of §5.2, reused
            assert len(_days) == len(tr)
            data_size["lstm"][_n] = lstm_forecast
        else:  # the LSTM's rate bracketed again at each length; the forest keeps its setting
            _t = tune(
                [fc_data[FORECAST]],
                "lstm",
                FC_SEEDS,
                rate_path=f"data_size/lstm/{_n}/lr",
                train_days=_days,
            )
            data_size["lstm"][_n] = summary(_t)
            check_net(data_size["lstm"][_n], f"data_size/lstm/{_n}")
        data_size["forest"][_n] = nse(flow[te], chosen_forecast(chosen, issued(_days), issued(te)))
        check(data_size["forest"][_n], f"data_size/forest/{_n}")
    assert data_size["lstm"][15] is lstm_forecast and fc_tuned[FORECAST].runs
    print("LSTM:", {_n: round(_v["test_median"], 4) for _n, _v in data_size["lstm"].items()})
    print("forest:", {_n: round(_v, 4) for _n, _v in data_size["forest"].items()})
    print(f"{time.perf_counter() - _t0:.0f} s")
    return (data_size,)


@app.cell
def _(CHECK_NETS, PALETTE, YEARS_RUN, chosen, chronos_nse, data_size, plt):
    _c = chronos_nse["rain"]
    _lstm = [data_size["lstm"][_n]["test_median"] for _n in YEARS_RUN]
    if CHECK_NETS:
        # With one or two years, Chronos-2 (no training) is above both models trained here
        assert all(_c > data_size[_m][_n] for _m in ("forest",) for _n in (1, 2))
        assert all(_c > data_size["lstm"][_n]["test_median"] for _n in (1, 2))
        # Trained well, the LSTM stays close to the forest at every length; "close" is a gap
        # under 0.03, the rule stated below the figure
        assert all(
            abs(data_size["lstm"][_n]["test_median"] - data_size["forest"][_n]) < 0.03
            for _n in YEARS_RUN
        )
    _fig, _ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    _ax.plot(
        YEARS_RUN,
        _lstm,
        "o-",
        color=PALETTE[0],
        label="LSTM from scratch",
    )
    _ax.plot(
        YEARS_RUN,
        [data_size["forest"][_n] for _n in YEARS_RUN],
        "^--",
        color=PALETTE[2],
        label=f"forest, {chosen[2]}-day windows",
    )
    _ax.axhline(_c, color=PALETTE[1], ls="-.", label="Chronos-2, no training at all")
    _ax.set_xscale("log")
    _ax.set_xticks(YEARS_RUN, [str(_n) for _n in YEARS_RUN])
    _ax.set_xlabel("years of record to train on")
    _ax.set_ylabel("test NSE, tomorrow's flow")
    _ax.legend(frameon=False, loc="lower right")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    "Close" here means a gap under 0.03 in test NSE: the forest is one
    seed, and the LSTM's three seeds spread by a few hundredths, so smaller gaps say nothing
    about which model is better. With very little data, use a pretrained model as it is; with
    moderate data, fine-tune it (Lecture 15's rule, for time series). A new gauge has one year.

    ### 5.4 Frost hides the retrofit

    KW51 was retrofitted from May to September 2019 (Maes & Lombaert, 2021). Both alarms take
    the same reference rows: the pre-retrofit days with a temperature reading, a winter
    included. The naive alarm fires when the daily median of mode 13 leaves those days' mean
    ± 3 standard deviations. The corrected alarm first regresses the frequency on the bridge's
    temperature with a bend at 0 °C (Lecture 5), fitted on the same days, and alarms on the
    residual, outside ± 3 standard deviations of the residuals there. False alarms are counted
    on the reference rows (in sample, for both); detection on the days after the retrofit.
    """)
    return


@app.cell
def _(BASELINE, PALETTE, RETROFIT, ROUNDOFF, check, daily, mode13, np, pd, plt, ref, temp, times):
    def _hinge(t):
        """A line in temperature that bends at 0 °C (frozen or not)."""
        return np.stack([np.ones_like(t), t, np.minimum(t, 0.0)], axis=1)

    _f = mode13
    _temp = daily(times, temp).reindex(_f.index)
    _days = _f.index
    _pre, _post = _days < RETROFIT[0], _days > RETROFIT[1]
    _ok = _temp.notna().to_numpy()
    _ref = _pre & _ok  # the reference rows of both alarms
    _after = _post & _ok
    _mu, _sd = _f[_ref].mean(), _f[_ref].std(ddof=1)
    _naive = ((_f - _mu).abs() > 3 * _sd).to_numpy()
    _coef, *_ = np.linalg.lstsq(_hinge(_temp[_ref].to_numpy()), _f[_ref].to_numpy(), rcond=None)
    _resid = pd.Series(np.nan, index=_days)
    _resid[_ok] = _f[_ok].to_numpy() - _hinge(_temp[_ok].to_numpy()) @ _coef
    _rsd = _resid[_ref].std(ddof=1)
    _corrected = (_resid.abs() > 3 * _rsd).to_numpy()
    _frozen = _ref & (_temp < 0).to_numpy()
    # The autumn weeks as the reference instead: each alarm's band from them (the residual's
    # from the autumn days with a temperature reading), and a residual band as wide as the raw
    _base = (_days >= BASELINE[0]) & (_days <= BASELINE[1])
    _later = _pre & ~_base  # the healthy days after the autumn weeks: all have a temperature
    assert (_later <= _ok).all()
    _a_mu, _a_sd = _f[_base].mean(), _f[_base].std(ddof=1)
    _a_rsd = _resid[_base & _ok].std(ddof=1)
    _a_naive = ((_f - _a_mu).abs() > 3 * _a_sd).to_numpy()
    _a_corrected = (_resid.abs() > 3 * _a_rsd).to_numpy()
    _a_same_width = (_resid.abs() > 3 * _a_sd).to_numpy()
    _winter = _days.month.isin((12, 1, 2))
    alarm = {
        "mode_hz_autumn": float(_a_mu),
        "frozen_days": int(_frozen.sum()),
        "frozen_shift_pct": float(100 * (_f[_frozen].mean() / _a_mu - 1)),
        "retrofit_shift_pct": float(100 * (_f[_post].median() / _f[_pre].median() - 1)),
        "retrofit_shift_hz": float(_f[_post].median() - _f[_pre].median()),
        "frost_peak_hz": float(_f[_pre].max()),
        "post_median_hz": float(_f[_post].median()),
        "resid_post_median_hz": float(_resid[_post].median()),
        "resid_pre_median_hz": float(_resid[_pre].median()),
        "ref_days": int(_ref.sum()),
        "post_days": int(_after.sum()),
        "band_hz": {"naive": float(3 * _sd), "corrected": float(3 * _rsd)},
        "naive": {
            "false": int((_naive & _ref).sum()),
            "false_frozen": int((_naive & _frozen).sum()),
            "caught": int((_naive & _after).sum()),
        },
        "corrected": {
            "false": int((_corrected & _ref).sum()),
            "caught": int((_corrected & _after).sum()),
        },
        "autumn": {
            "band_hz": float(3 * _a_sd),
            "later_days": int(_later.sum()),
            "naive_false": int((_a_naive & _later).sum()),
            "naive_false_winter": int((_a_naive & _later & _winter).sum()),
            "corrected_band_hz": float(3 * _a_rsd),
            "corrected_days": int((_base & _ok).sum()),
            "corrected_false": int((_a_corrected & _later).sum()),
            "same_width_false": int((_a_same_width & _later).sum()),
        },
    }

    def _walk(d, path):
        for _k, _v in d.items():
            if isinstance(_v, dict):
                _walk(_v, f"{path}/{_k}")
            else:
                check(_v, f"{path}/{_k}", ROUNDOFF if isinstance(_v, float) else 0)

    _walk(alarm, "alarm")
    for _i, _c in enumerate(_coef):
        check(float(_c), f"alarm/hinge/{_i}")
    for _k, _v in alarm.items():
        print(f"{_k}: {_v}")
    # Same rows: the raw band must hold the frost, so it misses every day after the retrofit;
    # the residual's band catches them all
    assert alarm["naive"]["caught"] == 0 and alarm["corrected"]["caught"] == alarm["post_days"]
    # The frost: the frequency rises more when frozen than after the retrofit
    assert alarm["frozen_shift_pct"] > 2 * alarm["retrofit_shift_pct"] > 0
    # The picture: a jump to 6.6 Hz in the January 2019 frost; after the retrofit, 0.1 Hz higher
    # and inside the raw band; flat and slightly falling above 0 °C, climbing steeply below 0;
    # the residual steps 0.1 Hz
    if ref("alarm") is not None:
        assert _f[_pre].idxmax().strftime("%Y-%m") == ref("alarm/frost_peak_month")
    assert _f[_pre].max() > _mu + 3 * _sd and abs(_f[_post].median() - _mu) < 3 * _sd
    assert -0.01 < _coef[1] < 0 and _coef[1] + _coef[2] < -0.2  # Hz per °C above and below 0
    assert _resid[_post].median() > 3 * _rsd and abs(_resid[_pre].median()) < 0.01
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for _ax, _y, _band, _title in (
        (_a, _f, (_mu - 3 * _sd, _mu + 3 * _sd), "mode 13 (Hz); band: before the retrofit ± 3 SD"),
        (_b, _resid, (-3 * _rsd, 3 * _rsd), "after removing temperature (Hz)"),
    ):
        _ax.axvspan(np.datetime64(RETROFIT[0]), np.datetime64(RETROFIT[1]), color="#E6E6E6")
        _ax.axhspan(*_band, color=PALETTE[2], alpha=0.2)
        _ax.plot(_y.index, _y, ".", ms=3, color=PALETTE[0])
        _ax.set_title(_title, fontsize=10)
        _ax.tick_params(axis="x", labelrotation=30)
    _fig
    return (alarm,)


@app.cell
def _(alarm, mo):
    _n, _c, _a = alarm["naive"], alarm["corrected"], alarm["autumn"]
    mo.md(rf"""
    A frozen structure is stiffer, a known effect in bridge monitoring (Peeters & De Roeck,
    2001, on the Z24 bridge): on the {alarm["frozen_days"]} days below 0 °C mode 13 sat
    {alarm["frozen_shift_pct"]:.1f}% above autumn, while the retrofit moved it
    {alarm["retrofit_shift_pct"]:.1f}%. To hold the frost, the raw band must be
    ±{alarm["band_hz"]["naive"]:.2f} Hz wide: it fires on {_n["false"]} of the
    {alarm["ref_days"]} reference days ({_n["false_frozen"]} of them frozen) and on
    {_n["caught"]} of the {alarm["post_days"]} days after the retrofit. The residual's band is
    ±{alarm["band_hz"]["corrected"]:.3f} Hz: {_c["false"]} of the {alarm["ref_days"]}, and
    {_c["caught"]} of the {alarm["post_days"]}.

    With the autumn weeks as the reference instead, each alarm with its own band, the raw alarm
    fires on {_a["naive_false"]} of the {_a["later_days"]} healthy days after them
    ({_a["naive_false_winter"]} in December–February) and the residual alarm on
    {_a["corrected_false"]}; but the residual's band then rests on only {_a["corrected_days"]}
    autumn days with a temperature reading. Given the raw band's width, the residual alarm
    fires on {_a["same_width_false"]}. What the correction surely changes is detection: the
    retrofit becomes visible. The confounder is the season, again.

    ## 6. Decide

    ### Chronos-2: no training, close behind

    The April 2013 flood forecast one day ahead, with Chronos-2's 10–90% band, against the
    forest trained on the same 15 years as the LSTM (1990–2004).
    """)
    return


@app.cell
def _(
    CHRONOS_TOL,
    check,
    chosen,
    chosen_forecast,
    chronos_forecast,
    context,
    dates,
    flow,
    issued,
    np,
    tr,
):
    _z = np.flatnonzero(
        (dates >= np.datetime64("2013-03-25")) & (dates <= np.datetime64("2013-05-10"))
    )
    _q = chronos_forecast(issued(_z), context, with_rain=True, levels=(0.1, 0.5, 0.9))
    _forest = chosen_forecast(chosen, issued(tr), issued(_z))  # 15 years, as the LSTM
    _obs = flow[_z]
    _covers = float(np.mean((_obs >= _q[:, 0]) & (_obs <= _q[:, 2])))

    def _corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    # A day late on every rise: each forecast tracks yesterday's flow better than today's
    _late = {
        _name: (_corr(_f, _obs), _corr(_f[1:], _obs[:-1]))
        for _name, _f in (("Chronos-2", _q[:, 1]), ("forest", _forest))
    }
    print(f"{len(_z)} days; the 10-90% band covered {_covers:.1%}")
    print(
        "correlation with the same day, with the day before:",
        {_k: tuple(round(_v, 2) for _v in _p) for _k, _p in _late.items()},
    )
    check(len(_z), "chronos_event/days", 0)
    check(_covers, "chronos_event/band_covers", CHRONOS_TOL)
    assert len(tr) == 15 * 365 + 4  # water years 1990-2004: 15 years, 4 leap days
    assert all(_before > _same for _same, _before in _late.values())
    # The forest, a day late too, overshoots the day after each flood's peak (it stays below
    # the peak itself)
    _peaks = [
        _i
        for _i in range(1, len(_z) - 1)
        if _obs[_i] > 10 and _obs[_i] >= _obs[_i - 1] and _obs[_i] >= _obs[_i + 1]
    ]
    assert len(_peaks) == 2 and all(_forest[_i + 1] > _obs[_i + 1] for _i in _peaks)
    assert np.max(_forest) < np.max(_obs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Like the forest, it is a day late on a rise it cannot see coming without tomorrow's
    rain. Its training corpora list no CAMELS or USGS series, but one of them does not list
    its datasets: contamination cannot be ruled out (Lecture 3's warning).

    ### Embeddings group 671 rivers

    Lecture 19's move, on rivers: stop training and use a pretrained model's embeddings. Each
    basin's flow over the 2,048 days to 30 September 2004, embedded by Chronos-2: the mean of
    its 768-number patch tokens (128 patches of 16 days). k-means (Lecture 4) on 20 principal
    components finds five groups, numbered by snow, least first. Then a linear probe (ridge,
    its penalty chosen by leave-one-out), with each of the 18 water regions (HUC-02) held out
    in turn, reads snow fraction and aridity from the embedding, against four flow signatures
    a hydrologist would compute from the same days: when in the year the flow comes (sine and
    cosine of its centroid), its variability, and its day-to-day persistence.
    """)
    return


@app.cell
def _(
    CHRONOS_TOL,
    CLUSTERS,
    EMBED_DAYS,
    KMeans,
    LOG_EPS,
    PCA,
    PROBE_ALPHAS,
    REFERENCE,
    RidgeCV,
    SEED,
    StandardScaler,
    cam,
    check,
    home,
    make_pipeline,
    np,
    pipe,
    time,
    torch,
):
    _t0 = time.perf_counter()
    _end = int(np.flatnonzero(cam.dates == np.datetime64("2004-09-30"))[0])
    _series = [
        cam.flow[_b, _end - EMBED_DAYS + 1 : _end + 1].astype(np.float32)
        for _b in range(len(cam.gauge_id))
    ]
    with torch.no_grad():
        _tokens, _ = pipe.embed(_series, batch_size=64)
    _patches = EMBED_DAYS // pipe.model.chronos_config.input_patch_size
    _emb = np.stack([_e[0, :_patches].float().mean(0).cpu().numpy() for _e in _tokens])
    _z = PCA(20, random_state=SEED).fit_transform(StandardScaler().fit_transform(_emb))
    _labels = KMeans(CLUSTERS, init="k-means++", n_init=20, random_state=SEED).fit_predict(_z)
    _a = cam.attrs
    _rank = np.empty(CLUSTERS, int)
    _rank[np.argsort([_a["frac_snow"][_labels == _k].median() for _k in range(CLUSTERS)])] = (
        np.arange(CLUSTERS)
    )
    groups = _rank[_labels]  # 0 = least snow

    # Four flow signatures from the same days
    _days = np.arange(_end - EMBED_DAYS + 1, _end + 1)
    _doy = 2 * np.pi * (cam.dates[_days].astype("datetime64[D]").astype(int) % 365.25) / 365.25

    def _signature(q):
        ok = np.isfinite(q)
        w = q[ok] / q[ok].sum()
        lq = np.log(q[ok] + LOG_EPS)
        return [
            np.sum(w * np.sin(_doy[ok])),
            np.sum(w * np.cos(_doy[ok])),
            q[ok].std() / q[ok].mean(),
            np.corrcoef(lq[:-1], lq[1:])[0, 1],
        ]

    _sig = np.array(
        [_signature(cam.flow[_b, _days].astype(float)) for _b in range(len(cam.gauge_id))]
    )
    _regions = _a["huc_02"].to_numpy()

    def _probe(features, target):
        """R² of a ridge probe, each HUC-02 region held out in turn."""
        pred = np.empty_like(target)
        for reg in np.unique(_regions):
            held = _regions == reg
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=PROBE_ALPHAS))
            model.fit(features[~held], target[~held])
            pred[held] = model.predict(features[held])
        return float(1 - np.sum((pred - target) ** 2) / np.sum((target - target.mean()) ** 2))

    _r2 = {
        _t: {
            "embedding": _probe(_emb, _a[_t].to_numpy()),
            "signatures": _probe(_sig, _a[_t].to_numpy()),
        }
        for _t in ("frac_snow", "aridity")
    }
    _stats = {
        _k + 1: (
            int((groups == _k).sum()),
            float(_a["frac_snow"][groups == _k].median()),
            float(_a["aridity"][groups == _k].median()),
            float(np.mean(_a["gauge_lon"][groups == _k] < -100)),
        )
        for _k in range(CLUSTERS)
    }
    print(f"embeddings {_emb.shape}, {_patches} patches; {time.perf_counter() - _t0:.0f} s")
    print("group: basins, median snow fraction, median aridity, share west of 100 W")
    for _k, _v in _stats.items():
        print(_k, _v[0], round(_v[1], 3), round(_v[2], 3), round(_v[3], 2))
    print("probe R²:", {_t: {_k: round(_v, 3) for _k, _v in _d.items()} for _t, _d in _r2.items()})
    assert _emb.shape == (671, 768) and _patches == 128 and len(np.unique(_regions)) == 18
    assert _sig.shape == (671, 4)
    if REFERENCE:  # exact on the slides' GPU; elsewhere the groups can trade a few basins
        for _k, _v in _stats.items():
            check(_v[0], f"map/groups/{_k}/basins", 0)
            check(_v[1], f"map/groups/{_k}/frac_snow")
            check(_v[2], f"map/groups/{_k}/aridity")
    # Where the groups lie, by longitude and latitude boxes (the slides' markers in brackets)
    _lon, _lat = _a["gauge_lon"].to_numpy(), _a["gauge_lat"].to_numpy()
    _box = {
        "west of 100 W": _lon < -100,
        "Pacific Northwest (W of 116 W, N of 42 N)": (_lon < -116) & (_lat > 42),
        "Northeast (E of 80 W, N of 40 N)": (_lon > -80) & (_lat > 40),
        "Appalachians and Southeast (E of 90 W, S of 39 N)": (_lon > -90) & (_lat < 39),
        "Midwest and South (100 W to 80 W)": (_lon > -100) & (_lon < -80),
    }
    _share = {
        (_g, _r): float(_m[groups == _g - 1].mean()) for _g in _stats for _r, _m in _box.items()
    }
    for _g in _stats:
        _in = groups == _g - 1
        print(
            f"group {_g}: median gauge {np.median(_lon[_in]):.1f}, {np.median(_lat[_in]):.1f}; "
            f"median elevation {np.median(_a['elev_mean'][_in]):.0f} m;",
            {_r.split(" (")[0]: round(_share[_g, _r], 2) for _r in _box},
        )
    _arid = [_v[2] for _v in _stats.values()]
    # Snowiest (downward triangles): almost entirely in the western mountains
    assert _share[5, "west of 100 W"] > 0.85 and np.median(_a["elev_mean"][groups == 4]) > 1500
    # Driest (circles): the most arid median, centred on the southern Plains and Texas
    assert _arid[0] == max(_arid) and -104 < np.median(_lon[groups == 0]) < -94
    assert 30 < np.median(_lat[groups == 0]) < 38
    # Diamonds: the most humid, shared by the Northeast and the Pacific Northwest
    _pnw, _ne = (_share[4, _r] for _r in list(_box)[1:3])
    assert _arid[3] == min(_arid) and _pnw > 0.25 and _ne > 0.25 and _pnw + _ne > 0.6
    # Upward triangles: the Appalachians and the Southeast
    assert _share[3, "Appalachians and Southeast (E of 90 W, S of 39 N)"] > 0.6
    # Squares: the home basin's group, the Midwest and South
    assert groups[home] + 1 == 2 and _share[2, "Midwest and South (100 W to 80 W)"] > 0.7
    for _t in ("frac_snow", "aridity"):
        check(_r2[_t]["embedding"], f"map/probe/{_t}/embedding", CHRONOS_TOL)
        check(_r2[_t]["signatures"], f"map/probe/{_t}/signatures")  # no network: exact
        assert _r2[_t]["embedding"] > _r2[_t]["signatures"]
    return (groups,)


@app.cell
def _(PALETTE, cam, groups, home, plt):
    _a = cam.attrs
    _fig, _ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for _k in range(int(groups.max()) + 1):
        _m = groups == _k
        _ax.scatter(
            _a["gauge_lon"][_m],
            _a["gauge_lat"][_m],
            s=12,
            marker="os^Dv"[_k],
            color=[*PALETTE, "#1B3A5C"][_k],  # the slides' colours and markers
            label=f"group {_k + 1}: {_m.sum()} basins",
        )
    _ax.scatter(
        _a["gauge_lon"][home], _a["gauge_lat"][home], s=160, facecolor="none", edgecolor="black"
    )
    _ax.set_aspect(1.25)
    _ax.set_axis_off()
    _ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    The groups line up with climate the model never saw. The snowiest group (downward
    triangles, median snow fraction 0.48) lies almost entirely in the western mountains; the
    driest (circles) centres on the southern Plains and Texas; humid basins of the Northeast
    and the Pacific Northwest share the diamonds; the Appalachians and the Southeast the upward
    triangles; and the home basin, circled, sits in the Midwest and South group (squares).

    ### Day-ahead alerts at Bismarck

    Each forecaster alerts when its forecast of tomorrow's flow beats the 12.0 mm/day threshold,
    on the six test years. Persistence, the forest, the LSTM, and Chronos-2 from past flow and
    rain; then the three given tomorrow's rain as well.
    """)
    return


@app.cell
def _(
    CHECK_NETS,
    COUNT_TOL,
    SLIDES,
    check,
    chronos_test,
    flow,
    forest15_rain_test,
    forest15_test,
    issued,
    lstm_rain_test,
    lstm_test,
    mo,
    np,
    te,
    threshold,
):
    # One forecast per test day (the target day), made the day before; the forest and the
    # LSTMs trained on the same 15 years
    _forecasts = {
        "persistence": flow[issued(te)],
        "forest": forest15_test,
        "LSTM": lstm_test,
        "Chronos-2": chronos_test["rain"],
        "forest + tomorrow's rain": forest15_rain_test,
        "LSTM + tomorrow's rain": lstm_rain_test,
        "Chronos-2 + tomorrow's rain": chronos_test["tomorrow_rain"],
    }
    if not SLIDES:  # the reduced setting forecast only a sample of days with Chronos-2
        _forecasts = {_k: _v for _k, _v in _forecasts.items() if not _k.startswith("Chronos")}
    _obs = flow[te]
    _seen = np.isfinite(_obs)
    _event = _seen & (_obs > threshold)
    alerts = {}
    for _name, _f in _forecasts.items():
        assert _f.shape == _obs.shape, (_name, _f.shape)
        _alert = _seen & (_f > threshold)
        alerts[_name] = {
            "hits": int((_alert & _event).sum()),
            "misses": int((~_alert & _event).sum()),
            "false": int((_alert & ~_event).sum()),
        }
    check(int(_event.sum()), "alerts/events", 0)
    check(int(_seen.sum()), "alerts/days", 0)
    _nets = {"LSTM", "LSTM + tomorrow's rain"}
    for _name, _v in alerts.items():
        if _name in _nets and not CHECK_NETS:
            continue
        _tol = 0 if _name in {"persistence", "forest", "forest + tomorrow's rain"} else COUNT_TOL
        for _k, _x in _v.items():
            check(_x, f"alerts/by/{_name}/{_k}", _tol)
    # Given tomorrow's rain, each catches more floods; persistence raises the most false alarms
    for _m in ("forest",) + (("Chronos-2", "LSTM") if CHECK_NETS else ()):
        assert alerts[f"{_m} + tomorrow's rain"]["hits"] > alerts[_m]["hits"]
    # Persistence raises the most false alarms; the LSTMs count only when trained as the
    # slides' (the reduced setting's are undertrained)
    _compared = [_k for _k in alerts if CHECK_NETS or _k not in _nets]
    assert alerts["persistence"]["false"] == max(alerts[_k]["false"] for _k in _compared)
    # Ten flood days: one binomial SD of a hit count near half of ten is 1.6 days, so a
    # difference of one or two is noise
    _sd = float(np.sqrt(int(_event.sum()) * 0.5 * 0.5))
    assert 1 < _sd < 2
    _rows = "\n".join(
        f"| {_k} | {_v['hits']} | {_v['misses']} | {_v['false']} |" for _k, _v in alerts.items()
    )
    mo.md(
        "| Forecaster | Floods caught | Missed | False alarms |\n"
        "|:-----------|-----:|-----:|-----:|\n" + _rows + "\n\n"
        f"{int(_event.sum())} flood days in {int(_seen.sum()):,}; one binomial standard "
        f"deviation of a hit count is {_sd:.1f} days."
    )
    return (alerts,)


@app.cell
def _(uh):
    # "A storm's rain reaches this river the same day": the filter's weight on today's rain is
    # nearly its peak, so a forecast made from yesterday's record cannot see most floods coming
    _w = uh["weights"]
    assert _w[0] > 0.9 * _w.max()
    return


@app.cell
def _(mo, uh):
    mo.md(rf"""
    A storm's rain reaches this river the same day (the filter's weight on today's rain is
    {uh["weights"][0]:.3f}, next to its peak of {uh["weights"].max():.3f} a day later), so a
    forecast made from yesterday's record cannot
    see most floods coming: the alert needs the weather forecast. In service that rain is a
    forecast, not the observation used here, so the counts with tomorrow's rain are an upper
    bound.

    **When not to use these models:** at a gauge without years of record (the forest and the
    LSTM need them), after a change to the basin or the gauge, and for days outside anything
    in training (storm B).

    ### Who answers for a missed flood?

    | Question | A gauge reading | A model's alert |
    |:---------|:----------------|:----------------|
    | Issued by | the weather service, by rule | an agency, a vendor, a county |
    | A miss | a known limit of the rule | whose model, whose threshold? |
    | Weakest | where there is no gauge | where training had no such year |
    | Gains most | gauged, wealthier basins | ungauged ones, if it transfers |

    In the US the National Weather Service issues flood warnings; a county acting on a vendor's
    model inherits a decision with no clear author. Google's model extended reliable warnings to
    ungauged rivers in over 80 countries (Nearing et al., 2024), where people have had none: a
    real gain in equity. Its errors fall where data are scarcest, the same places. Who must
    re-check the threshold after a drought like 2012?

    ### Vote: what runs the alert next year?

    The county wants a day-ahead alert at Bismarck.

    - **A.** Persistence: alert if today is above the threshold.
    - **B.** The forest on 14-day windows, with the rain forecast.
    - **C.** An LSTM trained here, with the rain forecast.
    - **D.** Chronos-2 as is: no training, no upkeep.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(alerts, mo):
    _f, _l = alerts["forest + tomorrow's rain"], alerts["LSTM + tomorrow's rain"]
    _p, _n = alerts["persistence"], sum(alerts["persistence"][_k] for _k in ("hits", "misses"))
    mo.accordion(
        {
            "Discussion": mo.md(rf"""
        **B, with a check.** Trained on the same 15 years and given tomorrow's rain, the forest
        caught {_f["hits"]} flood days with {_f["false"]} false alarm(s) and the LSTM
        {_l["hits"]} with {_l["false"]}: close, on {_n} flood days, and the forest is cheaper to
        run, explain, and monitor. D is the right start at a new gauge with one year of record
        (§5.3). A catches only floods already under way ({_p["hits"]} of {_n}) and raises the
        most false alarms ({_p["false"]}). C becomes the answer at scale: many basins, many
        years, one model (Kratzert et al., 2019; Nearing et al., 2024). Either way: re-check the
        threshold after a drought, and log every alert against what happened.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where 1-D models live today

    | Today | What it is | Lecture |
    |:------|:-----------|:--------|
    | Google Flood Hub | LSTMs on thousands of gauges | here |
    | Chronos-2, TimesFM, Moirai | pretrained transformers, used as is | 17, here |
    | Whisper and speech models | spectrogram encoder + transformer | 17 |
    | Bridge monitoring | hourly modes, alarms on residuals | here |

    Pretrained forecasters: Chronos-2 (Ansari et al., 2025), TimesFM (Das et al., 2024), Moirai
    (Woo et al., 2024). Speech recognition moved from recurrent layers to transformers on
    spectrograms (Whisper; Radford et al., 2023). One network for many rivers needs many
    rivers. With our 15 Illinois basins (those of Kratzert et al.'s 531 in Illinois), one LSTM
    for all of them against one LSTM per basin, each simulating flow from weather alone, with
    the same three seeds on both sides and each side's learning rate bracketed on validation
    (one shared rate for the fifteen single-basin models). The reduced setting runs three of
    the basins with one seed:
    """)
    return


@app.cell
def _(
    CHECK_NETS,
    FC_SEEDS,
    NET_TOL,
    REFERENCE,
    REGIONAL_BASINS,
    SIMULATE,
    basins,
    cam,
    check,
    np,
    ref,
    time,
    tune,
):
    _t0 = time.perf_counter()
    _a = cam.attrs
    _il = _a["gauge_name"].str.upper().str.rstrip().str.endswith(", IL") & (_a["kratzert531"] == 1)
    _all = _a.loc[_il, "gauge_id"].tolist()
    assert len(_all) == 15 and _all[0] == "03338780"
    _gauges = _all[:REGIONAL_BASINS]
    # The same seeds on both sides; each side's rate bracketed on validation (one shared rate for
    # the single-basin models)
    _pooled = tune(
        [basins(_gauges, SIMULATE)], "lstm", FC_SEEDS, rate_path="regional/pooled_rate/lr"
    )
    _single = tune(
        [basins([_g], SIMULATE) for _g in _gauges],
        "lstm",
        FC_SEEDS,
        rate_path="regional/single_rate/lr",
    )
    _pooled_nse = np.median(np.stack([_r.test_nse for _r in _pooled.runs]), axis=0)  # per basin
    _each = np.reshape([_r.test_nse[0] for _r in _single.runs], (len(_gauges), len(FC_SEEDS)))
    _each = np.median(_each, axis=1)  # per basin, over the same seeds
    _better = int((_pooled_nse > _each).sum())
    print(
        f"{len(_gauges)} basins: one LSTM for all (rate {_pooled.record['lr']:g}), median NSE "
        f"{np.median(_pooled_nse):.4f}; one per basin (rate {_single.record['lr']:g}) "
        f"{np.median(_each):.4f}; shared better at {_better}; {time.perf_counter() - _t0:.0f} s"
    )
    if ref("regional") is not None:
        print(
            f"the slides (15 basins): {ref('regional/pooled_median'):.4f} vs "
            f"{ref('regional/single_median'):.4f}, shared better at {ref('regional/pooled_better')}"
        )
    if CHECK_NETS:
        check(float(np.median(_pooled_nse)), "regional/pooled_median", NET_TOL)
        check(float(np.median(_each)), "regional/single_median", NET_TOL)
        check(_better, "regional/pooled_better", 0 if REFERENCE else 3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question.** A vendor sells a river forecaster: "NSE 0.92, tested on a random 20% of
    days from 30 years." Name two things you would ask before believing it, and the number you
    would want instead.
    """)
    return


@app.cell
def _(mo, split):
    _d = split["with_flow"]["random_days"]
    mo.accordion(
        {
            "Answer": mo.md(rf"""
        The split first: random days put each test day's neighbours in training, and one random
        draw can land anywhere (here {min(_d):.2f} to {max(_d):.2f}); ask for later years held
        out, and for a drought or flood year the training lacked. Then the inputs: did it use
        tomorrow's observed rain? That is a perfect weather forecast. The number to want: hits
        and false alarms at the county's threshold on held-out years, with the rain forecast it
        will have in service.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas

    - **Off by one day.** Row $t$ predicts day $t + 1$; an input that reaches day $t + 1$ is a
      leak, unless it is meant as a perfect forecast (tomorrow's rain here) and labelled as one.
    - **Random days are not held-out days.** Neighbouring days share storms; test on later
      years and report the spread of the score.
    - **Standardize with training days only**, and carry missing past flow forward as an
      input, but never score a missing target.
    - **PyTorch's `Conv1d` is a cross-correlation** on (batch, channels, time); `nn.LSTM` with
      `batch_first=True` takes (batch, time, channels). Transpose between them, and flip a
      filter to match the convolution formula.
    - **cuDNN's LSTM has no backward pass in evaluation mode**: switch cuDNN off
      (`torch.backends.cudnn.flags(enabled=False)`) to take gradients with respect to inputs.
    - **Determinism on a GPU.** `torch.use_deterministic_algorithms(True)` needs
      `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA starts.
    - **A spectrum's resolution is one over the record's length**, and nothing above half the
      sampling rate (Nyquist) can be seen.
    - **A random forest needs `max_features`.** scikit-learn's `RandomForestRegressor` tries
      every input at every split by default (`max_features=1.0`), which is bagging. Set it, as
      here (a third of the inputs, Breiman, 2001), so the code runs the method it names.

    ## Going deeper (Forsyth)

    - §8.3.1–8.3.2: vector quantization turns a signal of any length into a fixed-length
      feature: cut it into short pieces, cluster the pieces with k-means, and describe the
      signal by how often each cluster appears; Forsyth classifies activities from
      accelerometer records this way. It is the bag-of-pieces cousin of today's windows, and
      it ignores the order that the LSTM keeps.
    - §13.1: Markov chains, where the next state depends only on the current one: the
      assumption behind persistence and ARIMA's short memory.
    - §13.2–13.3: hidden Markov models, a hidden state observed only through what it emits,
      inferred by dynamic programming and learned with EM. The LSTM's store plays the hidden
      state (soil moisture, seen only through the flow), but it is continuous and its update
      is learned by gradient descent rather than a transition matrix.
    - §17.1.1: convolution as pattern detection, the same weights at every location; here
      along one axis, days, instead of two. Forsyth writes it for images; PyTorch's layers
      compute a cross-correlation, the same thing with the kernel flipped.
    - §10.4.2: the ridge penalty that keeps the unit hydrograph's 60 weights from chasing noise.

    **Exercises.** (1) Fit the unit hydrograph with a smoothness penalty on second differences
    of the weights instead of ridge, and compare the weights and the test NSE. (2) Describe
    each KW51 day by vector quantization of its hourly mode-13 values (§8.3) and check whether
    the frost days form their own cluster.

    ## Things to remember

    - Windows turn a record into a table.
    - A filter on time is a 1-D convolution.
    - Sines and spectrograms read vibration.
    - An LSTM's store is its memory.
    - Few years? Borrow a pretrained model.
    """)
    return


if __name__ == "__main__":
    app.run()
