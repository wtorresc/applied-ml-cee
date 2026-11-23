"""Applied Machine Learning for CEE — Lecture 24: Learning Dynamics from Data.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l24_learning_dynamics.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: the recessions use CAMELS (Newman et al., 2015; Addor et al., 2017; CC BY 4.0): run
``bash scripts/fetch_camels.sh`` once, then ``pixi run -e aml-env python scripts/derive_camels.py``.
The pond and the river are simulated with assumed values. Set ``L24_SETTING=slides`` or
``L24_SETTING=reduced`` (the default) to choose the setting (see the header cell); the choice
never depends on the machine.
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
    # Applied Machine Learning for CEE — Lecture 24: Learning Dynamics from Data

    **Slides:** *Applied Machine Learning for CEE*, Lecture 24 (`lectures/cee_l24_learning_dynamics`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §10.4.1 (functions of one measurement as extra columns: today's library of candidate terms), §11.1.3–11.1.4 (choosing a model by held-out error; greedy stagewise regression, a cousin of thresholded least squares), §11.4.1 (L1 regularization, sparse models, and correlated columns), §10.2.5 (transforming variables: a power law fitted on logarithms), and §17.1.1–17.1.2 (convolution, and maps applied at every position). SINDy itself: Brunton et al. (2016); the Fourier neural operator: Li et al. (2021); the routing: Cunge (1969), Ponce & Yevjevich (1978).

    **Data:** CAMELS v1.2 (Newman et al., 2015; Addor et al., 2017; doi:10.5065/D6MW2F4D; CC BY 4.0) for the recessions: the daily flow of the North Fork Vermilion River near Bismarck, IL, and its basin's Daymet rain (Lecture 20's home basin). Fetch with `bash scripts/fetch_camels.sh`, then `python scripts/derive_camels.py`. The pond and the river are simulated with assumed values, so the truth is known.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`, against the
    slides' `numbers.json` (read from `lectures/build/fig/cee_l24_learning_dynamics/`). The
    notebook integrates the pond's equation about 100 times (SciPy's `solve_ivp` at the slides'
    tolerances) and trains Fourier neural operators (FNOs) on the CPU, each in its own worker
    process with 4 threads, as the slides' figure script does. Two settings, chosen by the
    environment variable `L24_SETTING` (or by editing `SETTING` below), never by the machine:

    - **`reduced`** (the default): one FNO seed of 30 epochs. It runs on a laptop CPU: 3
      minutes on 8 cores of the slides' machine (November 21, 2026), 12 minutes on 8 cores when
      that machine was heavily loaded (November 19), with about 1 GB of memory.
      Everything that does not depend on training is checked against the slides exactly (the
      pond and SINDy, the sweep, the design storm, the bootstrap, the routing solver, the unit
      hydrograph, the recessions, the FNO's Fourier layer and its weight count); the trained
      network's numbers are printed, not checked.
    - **`slides`**: three FNOs of 300 epochs, as in the slides: 26 minutes on the 56-thread
      machine that made the slides (heavily loaded, November 19, 2026), where every number but
      the timings must equal the slides' run (259 numbers, the FNOs included, that day; the
      same-floods scores of §5 were added on November 20).

    Timings are never reproducible, in either setting: only their order of magnitude is checked.

    **Your tasks** (from the slides): decide, from a year of inflow and level records, whether
    a pond holds a storm twice the largest recorded; derive SINDy (a library, least squares, a
    threshold) and check it on exact derivatives; find the pond's outlet law, sweep the
    threshold, run the design storm, and track a clogging weir with a bootstrap; route floods
    down a river with Muskingum–Cunge, build a Fourier neural operator from scratch, check its
    Fourier layer against a hand-written transform, and compare it with the unit hydrograph;
    break it four ways (a missing term, 5 cm of level noise, bigger floods, real recessions)
    and check the operator's water balance; then decide what routes tomorrow's forecasts.
    """)
    return


@app.cell
def _():
    import json
    import math
    import os
    import random
    import time
    from collections.abc import Callable
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor
    from scipy.integrate import solve_ivp
    from scipy.optimize import curve_fit
    from torch import nn

    T_START = time.perf_counter()
    SEED = 24
    os.environ["PYTHONHASHSEED"] = "0"  # for the worker processes; this one needs it at start

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # The slides' constants (lectures/figures/cee_l24_learning_dynamics.py)
    # The pond (assumed values)
    AREA = 2000.0  # m2, vertical walls
    WEIR_C = 1.7  # m^0.5/s: Q_out = C h^1.5 (a 1-m broad weir)
    DEPTH = 2.5  # m from the outlet's crest to the rim
    LEVEL_NOISE = 0.005  # m, one standard deviation
    READ_MIN = 5  # minutes between level readings
    STORM_HOURS = 24
    N_STORMS = 20  # year 1
    STORM_PEAK = (0.5, 3.0)  # m3/s
    DESIGN = {"peak": 6.0, "tp_hours": 2.5, "shape": 3.0}  # twice the largest recorded peak
    CLOG = 0.7  # year 2: the weir passes 70% of its year-1 flow at the same level
    NAMES = ("Q_in", "1", "h^0.5", "h", "h^1.5", "h^2")
    TRUE_TERM = "h^1.5"
    LAMBDA = 0.05  # STLSQ threshold, a share of the largest scaled coefficient (fixed)
    LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3)  # the sweep
    NOISES = (0.005, 0.02, 0.05)  # m, for the noise reveal
    BOOT = 200  # bootstrap resamples of the storms
    N_VAL_STORMS = 10
    # The river reach (assumed values)
    WIDTH_M, SLOPE, MANNING = 50.0, 5e-4, 0.035
    REACH, LENGTH, SUB = 2000.0, 60_000.0, 4  # m per reach, m, sub-steps per output step
    STEP_S = 3600 / SUB  # s: the solver's internal step for hourly floods
    FINE_SUB = SUB // 2  # sub-steps for half-hourly floods: the same STEP_S
    BASE = 50.0  # m3/s
    HOURS = 72
    FLOOD_PEAK, BIG_PEAK = (100.0, 600.0), (600.0, 1200.0)  # m3/s
    N_TRAIN, N_VAL, N_TEST, N_BIG, N_FINE = 1000, 200, 500, 500, 200
    Q_SCALE = 500.0  # m3/s: the network sees (Q - BASE) / Q_SCALE
    # The operator (fixed before any test; Li et al., 2021, 1-D settings, smaller)
    FNO_WIDTH, FNO_MODES, FNO_LAYERS = 32, 12, 4
    EPOCHS, BATCH, LR = 300, 32, 1e-3
    FNO_SEEDS = 3  # seeds 24, 25, 26
    THREADS, WORKERS = 4, 3  # torch threads per worker process; worker processes
    UH_LADDER = (6, 12, 24, 36, 48, 60, 72)  # hours of memory, up to the whole record
    # Recession days (CAMELS; Lecture 20's home basin and periods)
    RAIN_MM = 1.0  # a recession day has no more rain than this today and the two days before
    REC_NAMES = ("1", "Q^0.5", "Q", "Q^1.5", "Q^2")
    REC_LAMBDAS = (0.0, 0.01, LAMBDA, 0.1, 0.3)
    HOME = "03338780"  # North Fork Vermilion River near Bismarck, IL
    PERIODS = {"train": ("1989-10-01", "2004-09-30"), "test": ("2008-10-01", "2014-09-30")}

    seed_all(SEED)
    torch.set_num_threads(THREADS)  # this process too, as in the slides' script

    # The setting: "reduced" (one FNO of 30 epochs, the default) or "slides" (three FNOs of 300
    # epochs, every number checked). Never chosen by the machine: only the worker counts are
    # The CPUs this process may use (a laptop's 4 to 8 cores; a free Colab instance's 2)
    _cpus = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    ) or 1
    SETTING = os.environ.get("L24_SETTING", "reduced").strip().lower()
    if SETTING not in {"slides", "reduced"}:
        raise ValueError(f"L24_SETTING must be 'slides' or 'reduced', not {SETTING!r}")
    FULL = SETTING == "slides"
    SEEDS_RUN = FNO_SEEDS if FULL else 1
    EPOCHS_RUN = EPOCHS if FULL else 30
    POOL = min(SEEDS_RUN, WORKERS, max(1, _cpus // THREADS))  # worker processes
    # Worker processes for the pond's forward runs on the held-out storms (the orifice model
    # takes tens of seconds a storm); each run is the same in any process
    POND_WORKERS = min(10, max(1, _cpus // 2))

    # Tolerances. The pond (its ODE at the script's tolerances), STLSQ, the sweep, the bootstrap,
    # the recession fits, the routing, and the unit hydrograph repeat the slides' run to the
    # order of floating-point sums: relative EXACT.
    EXACT = 1e-9
    # Identities that hold in exact arithmetic, checked in float64 (round-off only). The slides
    # quote two of them (6e-16, 2e-15); their digits are round-off, so only the bound is checked.
    ROUNDOFF = 1e-14
    # The FNOs: CPU, 4 threads per process, deterministic algorithms, as the slides' run. On the
    # machine that made the slides they reproduce it exactly (two runs, every FNO number equal
    # to numbers.json); on another CPU training may take another path, and the slide's digits
    # are then the check (shown).
    FNO_TOL = 1e-9

    def near(x: float, ref: float, tol: float = EXACT) -> bool:
        """Whether ``x`` equals ``ref`` to a relative ``tol`` (absolute where ``ref`` is 0).

        Args:
            x: The computed value.
            ref: The slides' run's value (numbers.json, full precision).
            tol: The tolerance.

        Returns:
            True if they agree.
        """
        return abs(x - ref) <= tol * abs(ref) if ref != 0 else abs(x) <= tol

    def shown(x: float, text: str) -> bool:
        """Whether ``x`` reads as the slide's ``text`` at its precision.

        Handles "0.28", "2,015", "4.96e-04", "+8.37e-05", and "6e-16".

        Args:
            x: The unrounded value.
            text: The slide's digits (a percentage is passed without its % sign).

        Returns:
            True if ``x`` rounds to ``text``.
        """
        plain = text.replace(",", "")
        sign = "+" if plain.startswith("+") else ""
        if "e" in plain:
            mantissa = plain.split("e", maxsplit=1)[0]
            digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
            return f"{x:{sign}.{digits}e}" == plain
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        return f"{x:{sign}.{decimals}f}" == plain

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square difference of two arrays of the same shape (as ``common.rmse``).

        Args:
            a: Predictions.
            b: Targets, the same shape.

        Returns:
            sqrt(mean((a - b)^2)).
        """
        assert np.shape(a) == np.shape(b), (np.shape(a), np.shape(b))
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def same_magnitude(x: float, slide: float) -> bool:
        """Whether a timing is within one power of ten of the slide's (timings vary by machine).

        Args:
            x: Seconds measured in this run.
            slide: Seconds in the slides' run.

        Returns:
            True if the two powers of ten differ by at most one.
        """
        return abs(math.floor(math.log10(x)) - math.floor(math.log10(slide))) <= 1

    def find_root() -> Path:
        """The repository's root, from the repo root, ``notebooks/``, or ``ipynb/``.

        Returns:
            The first of the working directory and its parent that holds ``data/``.

        Raises:
            FileNotFoundError: If neither does (run from inside the repository).
        """
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / "data").is_dir():
                return base
        raise FileNotFoundError("data/ not found: run from the repository, notebooks/, or ipynb/")

    ROOT = find_root()

    def find_daily() -> Path:
        """CAMELS' daily table under ``ROOT``.

        Returns:
            The path of ``data/derived/camels_daily.npz``.

        Raises:
            FileNotFoundError: If it is missing (run ``bash scripts/fetch_camels.sh``, then
                ``scripts/derive_camels.py``).
        """
        path = ROOT / "data" / "derived" / "camels_daily.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `bash scripts/fetch_camels.sh`, then "
                "`pixi run -e aml-env python scripts/derive_camels.py`"
            )
        return path

    NUMBERS = ROOT / "lectures" / "build" / "fig" / "cee_l24_learning_dynamics" / "numbers.json"
    # The slides' run, every value at full precision: the reference for every check (lesson
    # #32: copied programmatically, never typed). Build it with `make` in lectures/ if missing.
    REF = json.loads(NUMBERS.read_text(encoding="utf-8")) if NUMBERS.exists() else None
    if REF is None:
        print(f"{NUMBERS} not found: the comparisons with the slides are skipped")

    def ref(path: str) -> object:
        """The slides' value at ``path`` in numbers.json, e.g. "pond/area_from_fit".

        Args:
            path: Keys separated by "/"; list indices as integers.

        Returns:
            The value, or None when numbers.json is missing.
        """
        if REF is None:
            return None
        node = REF
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def timing(key: object) -> bool:
        """Whether a numbers.json key holds a timing, which no run reproduces.

        Args:
            key: The key.

        Returns:
            True for "seconds" keys, "minutes", and ratios of timings ("time_ratio_...").
        """
        return "seconds" in str(key) or str(key) == "minutes" or "time_ratio" in str(key)

    def close(a: object, b: object, tol: float = EXACT) -> bool:
        """Whether two JSON-like values agree: numbers to a relative ``tol``, timings skipped.

        Args:
            a: The computed value: a number, None, a string, or dicts and lists of them.
            b: The reference, the same structure.
            tol: Relative tolerance (absolute where the reference is 0).

        Returns:
            True if every pair agrees.
        """
        if isinstance(a, dict):  # every key of the reference; the notebook may add its own
            return (isinstance(b, dict) and set(b) <= set(a)
                    and all(timing(k) or close(a[k], b[k], tol) for k in b))  # fmt: skip
        if isinstance(a, list | tuple | np.ndarray):
            a = a.tolist() if isinstance(a, np.ndarray) else list(a)
            return (isinstance(b, list) and len(a) == len(b)
                    and all(close(x, y, tol) for x, y in zip(a, b, strict=True)))  # fmt: skip
        if a is None or b is None or isinstance(a, str):
            return a == b
        return near(float(a), float(b), tol)

    def same(x: object, path: str, tol: float = EXACT) -> bool:
        """Whether ``x`` equals the slides' value at ``path`` (see ``close``).

        Args:
            x: The computed value.
            path: Its key in numbers.json (see ``ref``).
            tol: Relative tolerance.

        Returns:
            True if they agree, or if numbers.json is missing (nothing to compare).
        """
        return REF is None or close(x, ref(path), tol)

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    INK, MUTED = "#1A1A1A", "#595959"
    FIG = (9.0, 3.6)
    print(
        f"setting {SETTING!r}: {SEEDS_RUN} FNO seed(s) of {EPOCHS_RUN} epochs, {POOL} worker(s) "
        f"x {THREADS} threads; root {ROOT}"
    )
    return (
        AREA,
        BASE,
        BATCH,
        BIG_PEAK,
        BOOT,
        CLOG,
        Callable,
        DEPTH,
        DESIGN,
        EPOCHS_RUN,
        EXACT,
        FIG,
        FINE_SUB,
        FLOOD_PEAK,
        FNO_LAYERS,
        FNO_MODES,
        FNO_SEEDS,
        FNO_TOL,
        FNO_WIDTH,
        FULL,
        HOME,
        HOURS,
        INK,
        LAMBDA,
        LAMBDAS,
        LENGTH,
        LEVEL_NOISE,
        LR,
        MANNING,
        MUTED,
        NAMES,
        NOISES,
        N_BIG,
        N_FINE,
        N_STORMS,
        N_TEST,
        N_TRAIN,
        N_VAL,
        N_VAL_STORMS,
        PALETTE,
        PERIODS,
        POND_WORKERS,
        POOL,
        Parallel,
        Q_SCALE,
        RAIN_MM,
        REACH,
        READ_MIN,
        REC_LAMBDAS,
        REC_NAMES,
        REF,
        ROUNDOFF,
        SEED,
        SEEDS_RUN,
        SLOPE,
        STEP_S,
        STORM_HOURS,
        STORM_PEAK,
        SUB,
        THREADS,
        TRUE_TERM,
        T_START,
        UH_LADDER,
        WEIR_C,
        WIDTH_M,
        curve_fit,
        delayed,
        find_daily,
        get_reusable_executor,
        near,
        nn,
        np,
        plt,
        ref,
        rmse,
        same,
        same_magnitude,
        shown,
        solve_ivp,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 5–23

    On your own, no notes, two minutes:

    1. Lecture 5: ridge shrinks every weight. Which penalty sets some weights to exactly zero?
    2. Lecture 20: a unit hydrograph turns rain into flow by a convolution. What must be true of
       the catchment for that to hold?
    3. Lecture 23: a PINN needed the PDE. What if we do not know the equation?

    **Today:** learn the equation itself from data, or learn the solver's input-to-output map.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. The lasso's absolute-value penalty (Tibshirani, 1996; checked below on a toy
           regression). Today a simpler cousin: least squares, then drop the small coefficients
           and refit (§3).
        2. Linear and time-invariant: twice the rain gives twice the flow, at the same delay.
           Today a river whose big floods travel faster breaks that (§4: the routing solver's
           median lag is 7 h for floods in the training range and 5 h for bigger ones; the unit
           hydrograph's is 7 h for both).
        3. Two answers today: search a library of candidate terms for a sparse equation (SINDy),
           or learn the map from an input function to an output function (a neural operator).
        """)
        }
    )
    return


@app.cell
def _(SEED, np):
    # Answer 1, on a toy regression where 2 of 6 inputs matter: the lasso's penalty sets the
    # other four weights to exactly zero; ridge only shrinks them
    from sklearn.linear_model import Lasso as _Lasso
    from sklearn.linear_model import Ridge as _Ridge

    _rng = np.random.default_rng(SEED)
    _x = _rng.normal(size=(200, 6))
    _y = 2.0 * _x[:, 0] - 1.5 * _x[:, 4] + 0.1 * _rng.normal(size=200)
    _lasso = _Lasso(alpha=0.1, fit_intercept=False, max_iter=100_000, tol=1e-12).fit(_x, _y)
    _ridge = _Ridge(alpha=10.0, fit_intercept=False, solver="cholesky").fit(_x, _y)
    print(f"lasso {np.round(_lasso.coef_, 4)}; ridge {np.round(_ridge.coef_, 4)}")
    assert np.flatnonzero(_lasso.coef_).tolist() == [0, 4]
    assert np.count_nonzero(_ridge.coef_) == 6
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A detention pond stores a storm's runoff and lets it out through an outlet whose law is
    unknown: an orifice ($Q_{out} \propto \sqrt h$) or a weir ($Q_{out} \propto h^{1.5}$)? The
    city has a year of records: the inflow of 20 storms, and the pond's level read every 5
    minutes with 5 mm of noise. **Decide:** for a design storm with a peak inflow of 6 m³/s,
    about twice the largest recorded, will the pond hold under its rim, 2.5 m above the outlet,
    or must the city enlarge it? A level predicted too low risks overtopping; too high, a
    needless enlargement.

    The physics is a water balance, $A\,dh/dt = Q_{in} - Q_{out}(h)$, with $h$ the level above
    the outlet's crest and $A$ the pond's area; the outlet's law $Q_{out}(h)$ is the unknown. The
    records here are simulated from a weir, $Q_{out} = C\,h^{1.5}$ with $C$ = 1.7 m$^{0.5}$/s,
    and nobody is told: the truth is known, so every fit can be scored against it.

    | The pond (assumed values; records simulated, so the truth is known) | Value |
    |:----------------------------------------|-----------------:|
    | Area (vertical walls) | 2,000 m² |
    | Outlet: an orifice (Q ∝ √h) or a weir (Q ∝ h^1.5)? | unknown |
    | Storms recorded in year 1 (inflow measured) | 20 |
    | Level read every | 5 min |
    | Level noise (one standard deviation) | 0.005 m |
    | Design storm: peak inflow | 6 m³/s |
    | Rim above the outlet | 2.5 m |
    """)
    return


@app.cell
def _(AREA, Callable, READ_MIN, STORM_HOURS, np, solve_ivp):
    def storm(t: np.ndarray, peak: float, tp: float, shape: float) -> np.ndarray:
        """A gamma-shaped inflow hydrograph (m3/s) at times ``t`` (s), peaking at ``tp`` (s)."""
        return peak * (t / tp) ** shape * np.exp(shape * (1 - t / tp))

    def storms(
        n: int, rng: np.random.Generator, peak: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Times (s) and ``n`` random storms, (n, times): peak, 1-4 h to the peak, shape 2-5."""
        t = np.arange(0, STORM_HOURS * 60 + READ_MIN, READ_MIN) * 60.0
        q = [
            storm(t, rng.uniform(*peak), rng.uniform(1, 4) * 3600, rng.uniform(2, 5))
            for _ in range(n)
        ]
        return t, np.array(q)

    def pond_level(
        t: np.ndarray, q_in: np.ndarray, rhs: Callable[[float, float], float]
    ) -> np.ndarray:
        """Integrate dh/dt = rhs(Q_in, h) from an empty pond (h = 0 at the crest).

        Explicit Runge-Kutta 4(5) with the slides' tolerances; the inflow is interpolated
        linearly between readings.

        Raises:
            RuntimeError: If the integration fails.
        """
        sol = solve_ivp(lambda tt, h: [rhs(float(np.interp(tt, t, q_in)), h[0])],
                        (t[0], t[-1]), [0.0], method="RK45", t_eval=t, rtol=1e-9,
                        atol=1e-12, max_step=60.0)  # fmt: skip
        if not sol.success or sol.y.shape[1] != len(t):
            raise RuntimeError(f"integration failed: {sol.message}")
        return sol.y[0]

    def weir(c: float) -> Callable[[float, float], float]:
        """The true pond: A dh/dt = Q_in - c h^1.5."""
        return lambda q, h: (q - c * max(h, 0.0) ** 1.5) / AREA

    return pond_level, storm, storms, weir


@app.cell
def _(
    DESIGN,
    LEVEL_NOISE,
    N_STORMS,
    N_VAL_STORMS,
    SEED,
    STORM_PEAK,
    WEIR_C,
    np,
    pond_level,
    same,
    shown,
    storms,
    weir,
):
    # Year 1: 20 storms, the true levels, and the readings (5 mm of noise); 10 more storms held
    # out for validation (their true levels). The same generators, in the same order, as the
    # slides' script.
    _rng = np.random.default_rng([SEED, 1])
    times, q_rec = storms(N_STORMS, _rng, STORM_PEAK)
    h_true = np.array([pond_level(times, _q, weir(WEIR_C)) for _q in q_rec])
    h_rec = h_true + _rng.normal(0, LEVEL_NOISE, h_true.shape)
    _, q_val = storms(N_VAL_STORMS, np.random.default_rng([SEED, 2]), STORM_PEAK)
    h_val = np.array([pond_level(times, _q, weir(WEIR_C)) for _q in q_val])
    # ... and as the gauge reads them (5 mm of noise, a generator of its own): the noise floor
    h_val_read = h_val + np.random.default_rng([SEED, 5]).normal(0, LEVEL_NOISE, h_val.shape)
    assert times.shape == (289,) and q_rec.shape == h_true.shape == h_rec.shape == (20, 289)
    assert q_val.shape == h_val.shape == h_val_read.shape == (10, 289)
    print(f"largest recorded inflow {q_rec.max():.4f} m3/s; highest true level {h_true.max():.4f} m;"
          f" design peak / largest recorded {DESIGN['peak'] / q_rec.max():.3f}")  # fmt: skip
    assert same(q_rec.max(), "pond/recorded_peak_max") and same(h_true.max(), "pond/max_level")
    # The notes: about twice the largest recorded (3.0 m3/s); the records reach 1.44 m (§5)
    assert shown(q_rec.max(), "3.0") and shown(h_true.max(), "1.44")
    assert 1.95 < DESIGN["peak"] / q_rec.max() < 2.05
    return h_rec, h_true, h_val, h_val_read, q_rec, q_val, times


@app.cell
def _(FIG, PALETTE, STORM_HOURS, h_rec, mo, np, plt, q_rec, times):
    # The attempt: the largest recorded storm, its inflow and its level readings
    _i = int(np.argmax(q_rec.max(1)))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=FIG, layout="constrained")
    _a1.plot(times / 3600, q_rec[_i], color=PALETTE[0], ls="-")
    _a1.set(xlabel="hours", ylabel="inflow, m³/s", xlim=(0, STORM_HOURS))
    _a2.plot(times / 3600, h_rec[_i], color=PALETTE[1], ls="-", lw=1)
    _a2.set(xlabel="hours", ylabel="level above the outlet, m", xlim=(0, STORM_HOURS))
    mo.vstack(
        [
            mo.md(f"""
    ### Your turn: orifice or weir?

    Storm {_i + 1} of 20, the largest recorded: its inflow peaks at {q_rec[_i].max():.1f} m³/s
    and is over within hours; the level, read every five minutes, rises and falls more slowly.
    Two minutes, alone. Hint: when the inflow has stopped, the level falls at $Q_{{out}}(h)/A$.
    How would the fall look if $Q_{{out}}$ grew like $\\sqrt h$? Like $h^{{1.5}}$?
    """),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion": mo.md(r"""
        With no inflow, $A\,dh/dt = -Q_{out}(h)$, so for $Q_{out} = C h^p$ the fall rate is
        $-dh/dt = (C/A)\,h^p$: on log axes, a line of slope $p$ (1/2 for an orifice, 3/2 for a
        weir). Plot the fall rate against the level on the storms' tails and read the slope
        (done below). SINDy generalizes this: it uses every reading, the inflow included, and
        tries several powers at once.
        """)
        }
    )
    return


@app.cell
def _(FIG, MUTED, PALETTE, h_rec, h_true, mo, np, plt, q_rec, times):
    # The students' check: on the storms' tails (inflow under 1% of the largest recorded peak,
    # level above 5 cm), the log-log slope of the fall rate against the level. On the true
    # levels it reads close to the weir's 1.5 (the inflow is small, not zero); on the readings,
    # the noise in the differenced levels pulls it to about halfway between 0.5 and 1.5
    _tail = (q_rec < 0.01 * q_rec.max()) & (h_true > 0.05)
    _fall_true = -np.gradient(h_true, times, axis=1)[_tail]
    _fall_rec = -np.gradient(h_rec, times, axis=1)[_tail]
    _level_true, _level_rec = h_true[_tail], h_rec[_tail]
    tail_slope = float(np.polyfit(np.log(_level_true), np.log(_fall_true), 1)[0])
    _pos = _fall_rec > 0
    tail_slope_rec = float(np.polyfit(np.log(_level_rec[_pos]), np.log(_fall_rec[_pos]), 1)[0])
    print(f"tail readings {_tail.sum()}: slope on true levels {tail_slope:.4f}; on the readings "
          f"{tail_slope_rec:.3f} ({(~_pos).sum()} of {_pos.size} falls not positive)")  # fmt: skip
    assert abs(tail_slope - 1.5) < 0.05 and 0.9 < tail_slope_rec < 1.2
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _ax.scatter(_level_rec[_pos], _fall_rec[_pos], s=4, color=MUTED, alpha=0.4, linewidths=0,
                label="from the readings (5 mm noise)")  # fmt: skip
    _ax.scatter(_level_true, _fall_true, s=4, color=PALETTE[0], marker="s", linewidths=0,
                label="from the true levels")  # fmt: skip
    _ax.set(xscale="log", yscale="log", xlabel="level h, m", ylabel="fall −dh/dt, m/s")
    _ax.legend(loc="upper left", frameon=False, fontsize=10)
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    On the storms' tails ({_tail.sum()} readings), the true levels fall with a slope of
    {tail_slope:.2f}: a weir. The readings, differenced, give {tail_slope_rec:.2f} on the
    {_pos.sum()} positive falls ({(~_pos).sum()} come out negative), about halfway between an
    orifice's 0.5 and a weir's 1.5: five millimetres of noise is too much for this check on the
    tails alone.
    """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: least squares, a threshold, two terms left

    Regress the level's rate of change $dh/dt$ (central differences of the readings) on six
    candidate terms: the inflow, a constant, $h^{0.5}$, $h$, $h^{1.5}$, $h^2$. **Frame 1:** least
    squares gives every term some weight. **Frame 2:** drop every coefficient smaller than a
    threshold (5% of the largest, on columns scaled to the same size), refit on the terms left,
    and repeat. **Frame 3:** what survives. The code is derived in §3.
    """)
    return


@app.cell
def _(
    AREA,
    INK,
    LAMBDA,
    MUTED,
    NAMES,
    PALETTE,
    WEIR_C,
    h_rec,
    mo,
    np,
    plt,
    q_rec,
    same,
    shown,
    stack,
    stlsq,
    times,
):
    rows, dhdt = stack(times, q_rec, h_rec)
    assert rows.shape == (20 * 289, 6) and dhdt.shape == (20 * 289,)
    col_size = np.linalg.norm(rows, axis=0)
    ls_scaled = np.linalg.lstsq(rows, dhdt, rcond=None)[0] * col_size  # frames 1 and 2
    _xi = stlsq(rows, dhdt, LAMBDA)
    final_scaled = _xi * col_size  # frame 3
    threshold = LAMBDA * float(np.abs(ls_scaled).max())
    assert same(ls_scaled, "pond/frames/least_squares")
    assert same(final_scaled, "pond/frames/final") and same(threshold, "pond/frames/threshold")
    # Frame 1: every term gets a non-zero weight; the inflow and h^1.5 dominate. Frame 2: the
    # other four fall under the threshold (5% of the largest). Frame 3: two terms left.
    _big = [NAMES.index("Q_in"), NAMES.index("h^1.5")]
    _small = [_k for _k in range(6) if _k not in _big]
    assert np.all(ls_scaled != 0)
    assert np.all(np.abs(ls_scaled[_small]) < threshold)
    assert threshold < np.abs(ls_scaled[_big]).min()
    assert np.flatnonzero(final_scaled).tolist() == _big
    # The notes of frame 3: area 2,015 m2 (true 2,000), weir coefficient 1.70 (true 1.70)
    _area, _c = 1 / _xi[0], -_xi[NAMES.index("h^1.5")] / _xi[0]
    assert shown(_area, "2,015") and shown(_c, "1.70") and shown(WEIR_C, "1.70")
    assert AREA == 2000
    _fig, _axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained", sharey=True)
    _lim = 1.15 * np.abs(ls_scaled).max()
    _titles = ("1. least squares", "2. the threshold (5%)", "3. refit: two terms")
    for _j, (_ax, _vals, _thr) in enumerate(
        zip(_axes, (ls_scaled, ls_scaled, final_scaled), (False, True, False), strict=True)
    ):
        _cols = [PALETTE[1] if (_thr and abs(_v) < threshold) else PALETTE[0] for _v in _vals]
        _ax.bar(range(6), _vals, color=_cols)
        if _thr:
            for _s in (-1, 1):
                _ax.axhline(_s * threshold, color=MUTED, ls=":", lw=1)
        _ax.axhline(0, color=INK, lw=0.8)
        _ax.set(xticks=range(6), ylim=(-_lim, _lim), title=_titles[_j])
        _ax.set_xticklabels(NAMES, rotation=45)
    _axes[0].set_ylabel("coefficient × column size")
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    Frame 1: least squares gives all six terms a weight; the four small ones are at most
    {100 * np.abs(ls_scaled[_small]).max() / np.abs(ls_scaled).max():.1f}% of the largest.
    Frame 2: the threshold, {threshold:.2e} on the scaled coefficients, catches exactly those
    four (orange). Frame 3: two terms survive, the inflow ($1/A$) and $h^{{1.5}}$: a weir. Read
    off the pond: area {_area:,.0f} m² (true 2,000), weir coefficient {_c:.2f} (true 1.70). An
    equation, not a black box: it can be checked, compared with a handbook, and run beyond the
    data.
    """),
        ]
    )
    return dhdt, final_scaled, ls_scaled, rows, threshold


@app.cell
def _(
    NAMES,
    NOISES,
    SEED,
    WEIR_C,
    h_true,
    library,
    ls_scaled,
    mo,
    np,
    q_rec,
    stack,
    times,
    weir,
):
    # The notes of frame 1: least squares gives every term some weight, "small here, because
    # these records are clean; with noisier records the small weights grow". The four small
    # weights' largest share of the largest weight: on the exact derivative (no noise), on the
    # records (5 mm), and on the noise reveal's draws (5 mm, 2 cm, 5 cm)
    _small = [NAMES.index(_n) for _n in ("1", "h^0.5", "h", "h^2")]

    def _share(r: np.ndarray, y: np.ndarray) -> float:
        _s = np.linalg.lstsq(r, y, rcond=None)[0] * np.linalg.norm(r, axis=0)
        return float(np.abs(_s[_small]).max() / np.abs(_s).max())

    _rows_true = np.vstack([library(_q, _h) for _q, _h in zip(q_rec, h_true, strict=True)])
    _exact = np.array([weir(WEIR_C)(_a, _b) for _a, _b in
                       zip(q_rec.ravel(), h_true.ravel(), strict=True)])  # fmt: skip
    small_share = {"exact derivative": _share(_rows_true, _exact)}
    small_share["the records (5 mm)"] = float(
        np.abs(ls_scaled[_small]).max() / np.abs(ls_scaled).max()
    )
    for _k, _sd in enumerate(NOISES):
        _h = h_true + np.random.default_rng([SEED, 10 + _k]).normal(0, _sd, h_true.shape)
        small_share[f"noise draw, {_sd:g} m"] = _share(*stack(times, q_rec, _h))
    print({_k: f"{_v:.2e}" for _k, _v in small_share.items()})
    _v = list(small_share.values())
    assert _v[0] < 1e-12  # exact derivatives: the small weights are round-off
    assert _v[2] < _v[3] < _v[4]  # 5 mm < 2 cm < 5 cm on the noise reveal's draws
    mo.md(f"""
    **Why the small weights are small (checked).** On the exact derivative the four small
    weights are round-off ({small_share["exact derivative"]:.0e} of the largest). On the
    records they are {100 * small_share["the records (5 mm)"]:.1f}% of it, and on the noise
    reveal's draws {100 * small_share["noise draw, 0.005 m"]:.1f}% at 5 mm,
    {100 * small_share["noise draw, 0.02 m"]:.1f}% at 2 cm, and
    {100 * small_share["noise draw, 0.05 m"]:.1f}% at 5 cm: they come from the noise and grow
    with it (§5).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive: a library and a threshold

    **Assumption.** The true law is a sum of a few terms from a library we write down; the
    derivative can be estimated from the data; and the data exercise the terms enough to tell
    them apart. Each is broken in §5.

    **The regression.** Divide the water balance by the area:

    $$\frac{dh}{dt} = \frac{1}{A}\,Q_{in} - \frac{1}{A}\,Q_{out}(h).$$

    Write $Q_{out}$ as a sum of candidate terms, and the right side is linear in their
    coefficients. With one row per reading $i$ and one column per term,

    $$\dot h_i \approx \Theta(Q_{in,i}, h_i)\,\xi, \qquad
      \Theta = [\,Q_{in},\ 1,\ h^{0.5},\ h,\ h^{1.5},\ h^{2}\,].$$

    For a weir the true $\xi$ is $(1/A,\ 0,\ 0,\ 0,\ -C/A,\ 0)$: two non-zero entries. The pond
    is read off a fit: $A = 1/\xi_{Q}$ and $C = -\xi_{h^{1.5}}/\xi_{Q}$.

    **The derivative.** Only levels are recorded. The central difference
    $\dot h_i \approx (h_{i+1} - h_{i-1})/(2\Delta t)$ errs by about $\Delta t^2\,h'''/6$ on a
    smooth level. Independent noise of standard deviation $\sigma$ on each reading becomes
    noise of standard deviation

    $$\operatorname{sd}\!\left(\frac{\varepsilon_{i+1} - \varepsilon_{i-1}}{2\Delta t}\right)
      = \frac{\sqrt{2}\,\sigma}{2\Delta t} = \frac{\sigma}{\sqrt{2}\,\Delta t}$$

    on the rate: with $\sigma$ = 5 mm and $\Delta t$ = 300 s, $1.2 \times 10^{-5}$ m/s (checked
    in §5). The division by $\Delta t$ is the danger: readings closer together make the rate's
    noise larger, not smaller.

    **Least squares, then a threshold.** The columns differ in size (an inflow in m³/s, levels
    in metres, a constant), so a coefficient's size means nothing until every column is scaled
    to unit length: $\tilde\Theta = \Theta D^{-1}$ and $\tilde\xi = D\xi$, with $D$ the
    diagonal of column norms. Sequentially thresholded least squares (STLSQ; Brunton et al.,
    2016):

    1. $\tilde\xi \leftarrow \arg\min_{\tilde\xi} \lVert \dot h - \tilde\Theta\tilde\xi \rVert^2$ on all columns;
    2. set every $\tilde\xi_j$ with $|\tilde\xi_j| < \lambda \max_k |\tilde\xi_k|$ to zero;
    3. refit least squares on the columns left; repeat steps 2 and 3 (ten times here), and
       return $\xi = D^{-1}\tilde\xi$.

    With $\lambda = 0$ nothing is dropped: plain least squares (checked below). The refit
    matters: dropping a column changes the best weights of the others, as in backward
    stagewise regression (Forsyth §11.1.4), but here every small term goes at once. The lasso
    (warm-up question 1) shrinks and selects in one penalized fit; STLSQ selects by a threshold
    and leaves the kept weights unshrunk. The threshold, 5%, is fixed before any test; §4
    sweeps it on storms the fit never saw.
    """)
    return


@app.cell
def _(Callable, NAMES, ROUNDOFF, SEED, np):
    def library(q_in: np.ndarray, h: np.ndarray) -> np.ndarray:
        """The candidate terms, (n, len(NAMES)): Q_in, 1, h^0.5, h, h^1.5, h^2."""
        hp = np.maximum(h, 0.0)
        return np.column_stack([q_in, np.ones_like(hp), hp**0.5, hp, hp**1.5, hp**2])

    def central_diff(h: np.ndarray, t: np.ndarray) -> np.ndarray:
        """dh/dt from scratch: centred inside, one-sided at the two ends (as ``np.gradient``)."""
        d = np.empty_like(h)
        d[1:-1] = (h[2:] - h[:-2]) / (t[2:] - t[:-2])
        d[0] = (h[1] - h[0]) / (t[1] - t[0])
        d[-1] = (h[-1] - h[-2]) / (t[-1] - t[-2])
        return d

    def stack(t: np.ndarray, q: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The library rows and dh/dt (central differences) of every storm, stacked."""
        rows = np.vstack([library(qi, hi) for qi, hi in zip(q, h, strict=True)])
        dhdt = np.concatenate([np.gradient(hi, t) for hi in h])  # = central_diff (checked)
        return rows, dhdt

    def stlsq(
        theta: np.ndarray,
        y: np.ndarray,
        lam: float,
        keep: np.ndarray | None = None,
        iters: int = 10,
    ) -> np.ndarray:
        """Sequentially thresholded least squares (Brunton et al., 2016).

        Columns are scaled to unit norm; a coefficient below ``lam`` times the largest is set
        to zero and the rest refitted, ``iters`` times. ``keep`` masks the columns allowed.

        Returns:
            The coefficients in the columns' own units, zeros where dropped.
        """
        keep = np.ones(theta.shape[1], bool) if keep is None else keep
        scale = np.linalg.norm(theta, axis=0)
        th = theta / scale
        xi = np.zeros(theta.shape[1])
        xi[keep] = np.linalg.lstsq(th[:, keep], y, rcond=None)[0]
        for _ in range(iters):
            active = keep & (np.abs(xi) >= lam * np.abs(xi).max())
            xi = np.zeros_like(xi)
            xi[active] = np.linalg.lstsq(th[:, active], y, rcond=None)[0]
        return xi / scale

    def model_rhs(xi: np.ndarray) -> Callable[[float, float], float]:
        """dh/dt of a fitted model, to integrate."""
        return lambda q, h: float((library(np.array([q]), np.array([h])) @ xi)[0])

    def terms(xi: np.ndarray) -> dict[str, float]:
        """The kept terms of a model and their coefficients."""
        return {n: float(x) for n, x in zip(NAMES, xi, strict=True) if x != 0}

    # The hand-written central differences against np.gradient on uneven random levels
    _rng = np.random.default_rng(SEED)
    _t = np.arange(0, 3600.0, 300.0)
    _h = np.cumsum(_rng.normal(size=_t.size))
    _d1, _d2 = central_diff(_h, _t), np.gradient(_h, _t)
    assert np.allclose(_d1, _d2, rtol=0, atol=ROUNDOFF * np.abs(_d2).max())
    # A threshold of zero drops nothing: STLSQ is least squares (here on a random problem; on
    # the records in the checks below)
    _a, _b = _rng.normal(size=(50, 6)) * [1, 10, 0.1, 1, 100, 1], _rng.normal(size=50)
    _ls = np.linalg.lstsq(_a, _b, rcond=None)[0]
    assert np.allclose(stlsq(_a, _b, 0.0), _ls, rtol=0, atol=ROUNDOFF * np.abs(_ls).max())
    print("central differences = np.gradient; STLSQ at threshold 0 = np.linalg.lstsq")
    return library, model_rhs, stack, stlsq, terms


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    ```python
    xi = stlsq(library(q, h), exact_dhdt, lam)  # no noise
    ls = lstsq(library(q, h), dhdt)              # all terms
    d  = stlsq(library(q, h), dhdt, lam=0) - ls
    v  = (route(long) - base).sum() / (long - base).sum()  # solver
    ```

    How far is `xi` from the truth? `d` from 0? `v` from 1? Two minutes, written. The third
    line: with a threshold of zero, nothing is dropped. The fourth: the river's routing solver
    (next), given a flood and 200 more hours of base flow to let it pass; the volumes are those
    above the 50-m³/s base flow.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The river's routing solver (used from §4 on, checked now)

    The lecture's second question is a 60-km river reach that routes floods. Its reference is
    variable-parameter Muskingum–Cunge (Cunge, 1969; Ponce & Yevjevich, 1978): 30 reaches of
    2 km, four sub-steps per hour, and at each reach

    $$Q_{j+1}^{n+1} = C_1 Q_j^{n+1} + C_2 Q_j^{n} + C_3 Q_{j+1}^{n}, \qquad
      C_1, C_2, C_3 = \frac{-1 + C_r + D,\ \ 1 + C_r - D,\ \ 1 - C_r + D}{1 + C_r + D},$$

    with the Courant number $C_r = c\,\Delta t/\Delta x$ and the cell Reynolds number
    $D = Q/(B S_0 c\,\Delta x)$. The celerity $c = \tfrac53 V$ comes from Manning's formula on a
    wide channel (width $B$ = 50 m, slope $S_0$ = 0.0005, $n$ = 0.035), evaluated at a
    three-point average of the flow: bigger floods travel faster. The three weights sum to 1,
    so a steady flow, once established, stays steady (checked at the base flow).
    """)
    return


@app.cell
def _(BASE, HOURS, LENGTH, MANNING, REACH, SLOPE, SUB, WIDTH_M, np):
    def celerity(q: np.ndarray) -> np.ndarray:
        """Kinematic celerity (5/3) V of a wide channel by Manning's formula, m/s."""
        q = np.maximum(q, 1.0)
        h = (q * MANNING / (WIDTH_M * np.sqrt(SLOPE))) ** 0.6
        return 5 / 3 * q / (WIDTH_M * h)

    def route(
        q_in: np.ndarray, per_hour: int = 1, reach: float = REACH, sub: int = SUB
    ) -> np.ndarray:
        """Variable-parameter Muskingum-Cunge (Ponce & Yevjevich, 1978), batched.

        Args:
            q_in: Inflow hydrographs, (batch, times), m3/s, ``per_hour`` values an hour.
            per_hour: Output steps an hour.
            reach: Reach length (m); the channel is ``LENGTH`` long.
            sub: Sub-steps per output step.

        Returns:
            Outflow at the downstream end, (batch, times).
        """
        nr = int(round(LENGTH / reach))
        dt = 3600 / per_hour / sub
        old = np.full((q_in.shape[0], nr + 1), BASE)
        out = np.empty_like(q_in)
        out[:, 0] = BASE
        for k in range(1, q_in.shape[1]):
            for s in range(1, sub + 1):
                w = s / sub
                new = np.empty_like(old)
                new[:, 0] = (1 - w) * q_in[:, k - 1] + w * q_in[:, k]
                for j in range(nr):
                    qbar = (new[:, j] + old[:, j] + old[:, j + 1]) / 3  # three-point average
                    c = celerity(qbar)
                    cr = c * dt / reach  # Courant number
                    dd = qbar / (WIDTH_M * SLOPE * c * reach)  # cell Reynolds number
                    den = 1 + cr + dd
                    new[:, j + 1] = ((-1 + cr + dd) * new[:, j] + (1 + cr - dd) * old[:, j]
                                     + (1 - cr + dd) * old[:, j + 1]) / den  # fmt: skip
                old = new
            out[:, k] = old[:, -1]
        return out

    def floods(
        n: int, rng: np.random.Generator, peak: tuple[float, float], per_hour: int = 1
    ) -> np.ndarray:
        """Random inflow floods, (n, HOURS * per_hour + 1), m3/s: BASE plus a gamma shape."""
        t = np.arange(HOURS * per_hour + 1) / per_hour  # hours
        out = np.empty((n, len(t)))
        for i in range(n):
            qp, tp, k = rng.uniform(*peak), rng.uniform(4, 16), rng.uniform(2, 6)
            out[i] = BASE + (qp - BASE) * (t / tp) ** k * np.exp(k * (1 - t / tp))
        return out

    return floods, route


@app.cell
def _(
    BASE,
    BIG_PEAK,
    EXACT,
    FINE_SUB,
    FLOOD_PEAK,
    N_BIG,
    N_FINE,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    STEP_S,
    floods,
    np,
    route,
    time,
):
    # The floods, drawn in the slides' order from one generator: 1,000 to train, 200 to
    # validate, 500 to test (peaks 100-600 m3/s), 500 bigger (600-1,200), and 200 in the
    # training range read every 30 minutes; each routed by the solver at one internal step,
    # STEP_S = 900 s: 4 sub-steps an hour for hourly floods, 2 per half hour for the fine ones,
    # so only the input's sampling changes, not the solver
    assert 3600 / 2 / FINE_SUB == STEP_S == 900.0
    _rng = np.random.default_rng([SEED, 20])
    flood_sets = {}
    for _k, _n, _peak in (("train", N_TRAIN, FLOOD_PEAK), ("val", N_VAL, FLOOD_PEAK),
                          ("test", N_TEST, FLOOD_PEAK), ("big", N_BIG, BIG_PEAK)):  # fmt: skip
        flood_sets[f"{_k}_in"] = floods(_n, _rng, _peak)
    flood_sets["fine_in"] = floods(N_FINE, _rng, FLOOD_PEAK, per_hour=2)
    solver_seconds = {}
    for _k in ("train", "val", "test", "big"):
        _t0 = time.perf_counter()
        flood_sets[f"{_k}_out"] = route(flood_sets[f"{_k}_in"])
        solver_seconds[_k] = time.perf_counter() - _t0
    flood_sets["fine_out"] = route(flood_sets["fine_in"], per_hour=2, sub=FINE_SUB)
    # The same fine floods on the hour, routed as hourly floods: the FNO is scored on both
    flood_sets["fine_hourly_in"] = np.ascontiguousarray(flood_sets["fine_in"][:, ::2])
    flood_sets["fine_hourly_out"] = route(flood_sets["fine_hourly_in"])
    for _k, _n in (
        ("train", 1000),
        ("val", 200),
        ("test", 500),
        ("big", 500),
        ("fine_hourly", 200),
    ):
        assert flood_sets[f"{_k}_in"].shape == flood_sets[f"{_k}_out"].shape == (_n, 73)
    assert flood_sets["fine_in"].shape == flood_sets["fine_out"].shape == (200, 145)
    # The three weights sum to 1: the base flow the channel starts with passes unchanged
    _steady = route(np.full((2, 73), BASE))
    assert np.allclose(_steady, BASE, rtol=EXACT, atol=0)
    assert np.all(flood_sets["train_in"][:, 0] == BASE)
    print(f"routed in {solver_seconds['test']:.2f} s for the 500 test floods")
    return flood_sets, solver_seconds


@app.cell
def _(
    AREA,
    BASE,
    FINE_SUB,
    LAMBDA,
    N_TEST,
    REACH,
    ROUNDOFF,
    SUB,
    WEIR_C,
    flood_sets,
    h_true,
    library,
    mo,
    np,
    q_rec,
    rmse,
    route,
    same,
    shown,
    stack,
    stlsq,
    times,
    weir,
):
    # 1. SINDy on the exact derivative (the true right side at the true levels): the truth
    _truth = np.array([1 / AREA, 0, 0, 0, -WEIR_C / AREA, 0])
    _rows_true = np.vstack([library(_q, _h) for _q, _h in zip(q_rec, h_true, strict=True)])
    _rhs = weir(WEIR_C)
    _exact = np.array([_rhs(_a, _b) for _a, _b in zip(q_rec.ravel(), h_true.ravel(), strict=True)])
    _xi = stlsq(_rows_true, _exact, LAMBDA)
    # 2. A threshold of zero against least squares, on the true levels' central differences
    _rows_n, _y = stack(times, q_rec, h_true)
    _ls = np.linalg.lstsq(_rows_n, _y, rcond=None)[0]
    checks_out = {
        "exact_rel": float(np.abs(_xi - _truth).max() / np.abs(_truth).max()),
        "exact_terms": int((_xi != 0).sum()),
        "lam0_rel": float(np.abs(stlsq(_rows_n, _y, 0.0) - _ls).max() / np.abs(_ls).max()),
    }
    # One more round of thresholding changes nothing: ten is enough on these records
    assert np.array_equal(stlsq(_rows_n, _y, LAMBDA, iters=11), stlsq(_rows_n, _y, LAMBDA))
    # 3. The solver: 50 test floods followed by 200 hours of base flow; volume out / in
    _n = min(50, N_TEST)
    _long_in = np.concatenate([flood_sets["test_in"][:_n], np.full((_n, 200), BASE)], axis=1)
    _long_out = route(_long_in)
    long_volume = (_long_out - BASE).sum(1) / (_long_in - BASE).sum(1)
    # ... and the share still to come after the first 72 hours (the same floods, causal solver)
    _vol_72h = (flood_sets["test_out"][:_n] - BASE).sum(1) / (
        flood_sets["test_in"][:_n] - BASE
    ).sum(1)
    after_72h_max = float((long_volume - _vol_72h).max())
    # 4. Half the reach and half the time step
    _fine_grid = route(flood_sets["test_in"][:_n], reach=REACH / 2, sub=2 * SUB)
    finer_grid_max_diff = float(np.abs(_fine_grid - flood_sets["test_out"][:_n]).max())
    # 5. The fine floods at half the step (450 s): how far the target would move if the
    # 30-minute test also changed the solver's step
    fine_half_step_rmse = rmse(
        route(flood_sets["fine_in"], per_hour=2, sub=2 * FINE_SUB), flood_sets["fine_out"]
    )
    # 6. The same floods sampled hourly and half-hourly: how far the sampling alone moves the
    # solver's outflow, at the hours
    fine_sampling_rmse = rmse(flood_sets["fine_hourly_out"], flood_sets["fine_out"][:, ::2])
    print(checks_out, f"volume {long_volume.min():.6f} to {long_volume.max():.6f}; still to "
          f"come after 72 h at most {after_72h_max:.2e}; finer grid {finer_grid_max_diff:.4f} "
          f"m3/s; fine floods at 450 s {fine_half_step_rmse:.3f} m3/s; hourly against "
          f"half-hourly input {fine_sampling_rmse:.3f} m3/s")  # fmt: skip
    # The slide's 6e-16 and 2e-15 are round-off: checked as bounds
    assert checks_out["exact_rel"] < ROUNDOFF and checks_out["lam0_rel"] < ROUNDOFF
    assert checks_out["exact_terms"] == 2
    assert same({"min": float(long_volume.min()), "max": float(long_volume.max())},
                "river/solver/long_volume")  # fmt: skip
    assert same(after_72h_max, "river/solver/after_72h_max")
    assert same(finer_grid_max_diff, "river/solver/finer_grid_max_diff")
    assert same(fine_half_step_rmse, "river/solver/fine_half_step_rmse")
    assert same(fine_sampling_rmse, "river/solver/fine_sampling_rmse")
    # The table and notes: 0.947 to 0.998 (94.7% to 99.8%), 3.3 m3/s
    assert shown(long_volume.min(), "0.947") and shown(long_volume.max(), "0.998")
    assert shown(100 * long_volume.min(), "94.7") and shown(100 * long_volume.max(), "99.8")
    assert shown(finer_grid_max_diff, "3.3")
    assert shown(fine_sampling_rmse, "0.23")  # the notes: the sampling alone moves the solver 0.23
    mo.md(f"""
    ### SINDy recovers an exact law exactly

    | Check | Value |
    |:------|------:|
    | SINDy on exact derivatives: largest coefficient error (relative) | {checks_out["exact_rel"]:.0e} |
    | … terms kept | {checks_out["exact_terms"]} |
    | Threshold 0 against least squares (relative) | {checks_out["lam0_rel"]:.0e} |
    | Solver, {len(long_volume)} floods, 200 h more: volume out ÷ in | {long_volume.min():.3f} to {long_volume.max():.3f} |
    | Solver, half the reach and step: largest change, m³/s | {finer_grid_max_diff:.1f} |

    Given the exact derivative, the regression returns the true equation, 2 terms, to
    round-off: the regression code is right, so later errors come from the data or the
    library. With a threshold of zero it is plain least squares. (Both errors are round-off,
    6e-16 and 2e-15 in the slides' run; round-off digits are not a result, so the checks here
    are bounds, under {ROUNDOFF:.0e}.) The routing solver: even 200 hours after a flood, only
    {100 * long_volume.min():.1f}% to {100 * long_volume.max():.1f}% of the water has come out
    (the first {len(long_volume)} test floods). That method is known to lose some volume (up to about 8% in Tang et al., 1999; a citation,
    not checked here); it is the reference anyway, as in practice, and the network is scored
    against it. Halving the reach and the time step moves the outflow by at most
    {finer_grid_max_diff:.1f} m³/s.
    """)
    return (
        after_72h_max,
        checks_out,
        fine_half_step_rmse,
        fine_sampling_rmse,
        finer_grid_max_diff,
        long_volume,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### SINDy finds the weir

    SINDy on the year-1 records (central differences of the noisy readings, the six-term
    library, the threshold fixed at 5%), against a fit that assumes an orifice: least squares
    on the inflow and $h^{0.5}$ only.
    """)
    return


@app.cell
def _(
    AREA,
    LAMBDA,
    NAMES,
    TRUE_TERM,
    WEIR_C,
    dhdt,
    mo,
    np,
    rows,
    same,
    shown,
    stlsq,
    terms,
):
    xi_sindy = stlsq(rows, dhdt, LAMBDA)
    xi_orifice = stlsq(rows, dhdt, 0.0, keep=np.array([_n in ("Q_in", "h^0.5") for _n in NAMES]))
    xi_missing = stlsq(rows, dhdt, LAMBDA, keep=np.array([_n != TRUE_TERM for _n in NAMES]))  # §5
    _k = NAMES.index(TRUE_TERM)
    pond_fit = {
        "sindy": terms(xi_sindy),
        "area_from_fit": float(1 / xi_sindy[0]),
        "c_from_fit": float(-xi_sindy[_k] / xi_sindy[0]),
        "orifice": terms(xi_orifice) | {"area": float(1 / xi_orifice[0])},
        "missing": terms(xi_missing),
    }
    print(pond_fit)
    _s, _o = pond_fit["sindy"], pond_fit["orifice"]
    assert list(_s) == ["Q_in", "h^1.5"] and list(_o) == ["Q_in", "h^0.5", "area"]
    assert same(_s, "pond/sindy") and same(_o, "pond/orifice")
    assert same(pond_fit["area_from_fit"], "pond/area_from_fit")
    assert same(pond_fit["c_from_fit"], "pond/c_from_fit")
    # The table, to the slide's digits
    _truth = {"Q_in": 1 / AREA, "h^1.5": -WEIR_C / AREA}
    assert shown(_truth["Q_in"], "5.00e-04") and shown(_truth["h^1.5"], "-8.50e-04")
    assert shown(_s["Q_in"], "4.96e-04") and shown(_s["h^1.5"], "-8.44e-04")
    assert shown(_o["Q_in"], "5.80e-05") and shown(_o["h^0.5"], "-6.82e-05")
    assert xi_sindy[NAMES.index("h^0.5")] == 0 and xi_orifice[_k] == 0
    # The notes: area 2,015 m2 and weir coefficient 1.70, both within 1% of the truth; the
    # orifice fit's area 17,255 m2
    _area_err = abs(pond_fit["area_from_fit"] / AREA - 1)
    _c_err = abs(pond_fit["c_from_fit"] / WEIR_C - 1)
    assert _area_err < 0.01 and _c_err < 0.01
    assert shown(pond_fit["area_from_fit"], "2,015") and shown(pond_fit["c_from_fit"], "1.70")
    assert shown(_o["area"], "17,255")
    mo.md(f"""
    | Coefficient of | Truth | SINDy | Orifice assumed |
    |:---------------|------:|------:|----------------:|
    | Q_in (1/A) | {_truth["Q_in"]:.2e} | {_s["Q_in"]:.2e} | {_o["Q_in"]:.2e} |
    | h^0.5 | 0 | 0 | {_o["h^0.5"]:.2e} |
    | h^1.5 | {_truth["h^1.5"]:.2e} | {_s["h^1.5"]:.2e} | 0 |

    SINDy keeps two terms: the inflow (an area of {pond_fit["area_from_fit"]:,.0f} m²) and
    $h^{{1.5}}$ (a weir coefficient of {pond_fit["c_from_fit"]:.2f}), within
    {100 * _area_err:.2f}% and {100 * _c_err:.2f}% of the truth. Assume an orifice instead, and
    least squares bends both coefficients to fit the records: an area of
    {_o["area"]:,.0f} m² against the true 2,000.
    """)
    return pond_fit, xi_missing, xi_orifice, xi_sindy


@app.cell
def _(
    POND_WORKERS,
    Parallel,
    delayed,
    dhdt,
    h_val,
    mo,
    model_rhs,
    np,
    pond_level,
    q_val,
    rows,
    same,
    shown,
    times,
    xi_missing,
    xi_orifice,
    xi_sindy,
):
    # Does a wrong form show in the fit? For each fitted equation, as the slides' script: the
    # share of the records' dh/dt variance it explains (R2), and its level error run forward on
    # the 10 held-out storms; here also its RMS residual, against the rate's own noise
    # (sigma / (sqrt(2) dt) = 1.2e-5 m/s at 5 mm, §3). The orifice model takes tens of seconds a
    # storm, so the 30 forward runs go to worker processes (each run is the same in any process)
    _fits = {"SINDy": xi_sindy, "orifice assumed": xi_orifice,
             "no h^1.5 in the library": xi_missing}  # fmt: skip
    _jobs = [(_n, _i) for _n in _fits for _i in range(len(q_val))]
    _levels = Parallel(n_jobs=POND_WORKERS, backend="loky")(
        delayed(pond_level)(times, q_val[_i], model_rhs(_fits[_n])) for _n, _i in _jobs
    )
    fit_quality = {}
    for _n, _xi in _fits.items():
        _hm = np.array([_l for (_m, _), _l in zip(_jobs, _levels, strict=True) if _m == _n])
        assert _hm.shape == h_val.shape
        _res = dhdt - rows @ _xi
        fit_quality[_n] = {
            "r2": float(1 - np.sum(_res**2) / np.sum((dhdt - dhdt.mean()) ** 2)),
            "rms_residual": float(np.sqrt(np.mean(_res**2))),
            "val_level_rmse_m": float(np.sqrt(np.mean((_hm - h_val) ** 2))),
        }
    print({_n: {_k: f"{_v:.4g}" for _k, _v in _f.items()} for _n, _f in fit_quality.items()})
    assert same(fit_quality, "pond/fit_quality")  # the notebook adds rms_residual
    _s, _o = fit_quality["SINDy"], fit_quality["orifice assumed"]
    _m = fit_quality["no h^1.5 in the library"]
    # The notes: R2 0.11 against SINDy's 0.94; without h^1.5, 0.934 against 0.940, and a
    # held-out level error of 0.015 m against SINDy's 0.0004; the orifice's 0.214 m
    assert shown(_o["r2"], "0.11") and shown(_s["r2"], "0.94") and shown(_s["r2"], "0.940")
    assert shown(_m["r2"], "0.934") and shown(_o["r2"], "0.110")
    assert shown(_s["val_level_rmse_m"], "0.0004") and shown(_m["val_level_rmse_m"], "0.015")
    assert shown(_m["val_level_rmse_m"], "0.0150") and shown(_o["val_level_rmse_m"], "0.214")
    # The orifice's residuals are about four times the rate's noise; SINDy's are the noise; the
    # missing-term model's residuals do not give it away (within 10% of SINDy's), "the first
    # reveal shows one that does not"
    assert _o["rms_residual"] > 3 * _s["rms_residual"]
    assert abs(_s["rms_residual"] / (0.005 / (np.sqrt(2) * 300)) - 1) < 0.05
    assert _m["rms_residual"] < 1.1 * _s["rms_residual"]
    mo.md(f"""
    **Does the wrong form show in the fit? (checked)** Here it does. SINDy's equation explains
    R² = {_s["r2"]:.2f} of the variance of the records' $dh/dt$, with residuals of
    {_s["rms_residual"]:.2e} m/s, the rate's own noise ($\\sigma/(\\sqrt2\\,\\Delta t)$ =
    1.18e-05 m/s, §3); run forward on the 10 held-out storms, its level errs
    {_s["val_level_rmse_m"]:.4f} m. The orifice fit explains R² = {_o["r2"]:.2f}, with
    residuals {_o["rms_residual"] / _s["rms_residual"]:.1f} times as large, and its level errs
    {_o["val_level_rmse_m"]:.3f} m on the same storms: its area of 17,255 m² is least squares
    doing what it can with a form that cannot fit. Look at the fit's residuals before its
    coefficients. The first reveal (§5) shows a wrong library whose residuals do not give it
    away: without $h^{{1.5}}$, R² = {_m["r2"]:.3f} and residuals
    {100 * (_m["rms_residual"] / _s["rms_residual"] - 1):.0f}% above SINDy's, though its level
    errs {_m["val_level_rmse_m"]:.3f} m on the held-out storms.
    """)
    return (fit_quality,)


@app.cell
def _(
    FIG,
    LAMBDA,
    LAMBDAS,
    MUTED,
    PALETTE,
    dhdt,
    h_val,
    h_val_read,
    mo,
    model_rhs,
    np,
    plt,
    pond_level,
    q_val,
    rmse,
    rows,
    same,
    shown,
    stlsq,
    times,
):
    # The sweep: terms kept and the held-out storms' level error (RMS over 10 storms x 289
    # readings, against the true levels) across the threshold. Models that come out identical
    # (bit for bit) are integrated once.
    sweep = []
    sweep_levels = {}
    for _lam in LAMBDAS:
        _x = stlsq(rows, dhdt, _lam)
        _key = _x.tobytes()
        if _key not in sweep_levels:
            sweep_levels[_key] = np.array([pond_level(times, _q, model_rhs(_x)) for _q in q_val])
        sweep.append({"lam": _lam, "terms": int((_x != 0).sum()),
                      "val_rmse_m": rmse(sweep_levels[_key], h_val),
                      "val_read_rmse_m": rmse(sweep_levels[_key], h_val_read)})  # fmt: skip
    # The gauge's own error: its readings (5 mm of noise) against the true level
    noise_floor = rmse(h_val_read, h_val)
    assert same(sweep, "pond/sweep") and same(noise_floor, "pond/noise_floor_m")
    print([(_s["lam"], _s["terms"], f"{_s['val_rmse_m']:.6f}", f"{_s['val_read_rmse_m']:.6f}")
           for _s in sweep], f"noise floor {noise_floor:.6f} m")  # fmt: skip
    # The alt text and notes: six terms 0.00106 m at 0; five at 0.001 and 0.003, 0.00125 m; two
    # from 0.01 up, 0.00041 m
    assert shown(sweep[0]["val_rmse_m"], "0.00106") and shown(sweep[1]["val_rmse_m"], "0.00125")
    assert shown(sweep[3]["val_rmse_m"], "0.00041")
    assert next(_s["lam"] for _s in sweep if _s["terms"] == 2) == 0.01
    assert min(sweep, key=lambda _s: _s["val_rmse_m"])["terms"] == 2
    # Against the gauge's readings (the noise floor about 5 mm) every model sits within 0.2 mm
    # of the floor: the ranking needs the true level, which a real pond never gives
    _read = [_s["val_read_rmse_m"] for _s in sweep]
    assert all(0 <= _r - noise_floor < 2e-4 for _r in _read)
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _lx = [max(_s["lam"], 3e-4) for _s in sweep]
    _ax.plot(_lx, [100 * _s["val_rmse_m"] for _s in sweep], color=PALETTE[0], ls="-", marker="o")
    for _x, _s in zip(_lx, sweep, strict=True):
        _ax.annotate(f"{_s['terms']} terms", (_x, 100 * _s["val_rmse_m"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)  # fmt: skip
    _ax.axvline(LAMBDA, color=MUTED, ls=":", lw=1)
    _ax.set(xscale="log", ylabel="validation level error, cm",
            xlabel="threshold, share of the largest coefficient (0 drawn at left)")  # fmt: skip
    mo.vstack(
        [
            mo.md("### Two terms track the true level best"),
            _fig,
            mo.md(f"""
    The threshold trades terms against fit. Against the true level of 10 storms it never saw,
    the sparse model is also the accurate one: two terms give a level error of
    {sweep[3]["val_rmse_m"]:.5f} m, all six {sweep[0]["val_rmse_m"]:.5f} m, five
    {sweep[1]["val_rmse_m"]:.5f} m. All three are far below the gauge's own error: scored
    against the gauge's readings (5 mm of noise; the readings against the truth,
    {noise_floor:.4f} m), they give {min(_read):.4f} to {max(_read):.4f} m, all within
    {1000 * (max(_read) - noise_floor):.1f} mm of that noise floor. The clear ranking needs the
    true level, which only a simulation gives.
    """),
        ]
    )
    return noise_floor, sweep, sweep_levels


@app.cell
def _(
    LAMBDA,
    LAMBDAS,
    WEIR_C,
    dhdt,
    h_val,
    library,
    mo,
    np,
    q_val,
    rows,
    stlsq,
    sweep_levels,
    weir,
):
    # The notes: "Extra terms fit noise in the derivative and drift when the model is run
    # forward." Measured on six terms (threshold 0) against two (5%): (a) the records' rate,
    # which six terms must fit at least as well (two terms are one of their choices); (b) the
    # held-out storms' exact rate; (c) the held-out level error in each quarter of the day
    _models = {"six": stlsq(rows, dhdt, LAMBDAS[0]), "two": stlsq(rows, dhdt, LAMBDA)}
    _rows_v = np.vstack([library(_q, _h) for _q, _h in zip(q_val, h_val, strict=True)])
    _exact_v = np.array([weir(WEIR_C)(_a, _b) for _a, _b in
                         zip(q_val.ravel(), h_val.ravel(), strict=True)])  # fmt: skip
    drift = {}
    for _n, _x in _models.items():
        _err = sweep_levels[_x.tobytes()] - h_val
        _q4 = np.array_split(np.arange(_err.shape[1]), 4)
        drift[_n] = {
            "records_rate_rms": float(np.sqrt(np.mean((dhdt - rows @ _x) ** 2))),
            "heldout_rate_rms": float(np.sqrt(np.mean((_rows_v @ _x - _exact_v) ** 2))),
            "level_rms_by_quarter_mm": [1000 * float(np.sqrt(np.mean(_err[:, _i] ** 2))) for _i in _q4],
            "end_level_error_mm": 1000 * float(np.mean(_err[:, -1])),
        }  # fmt: skip
    print(drift)
    _six, _two = drift["six"], drift["two"]
    assert _six["records_rate_rms"] <= _two["records_rate_rms"]  # (a)
    assert _six["heldout_rate_rms"] > _two["heldout_rate_rms"]  # (b)
    _s4, _t4 = _six["level_rms_by_quarter_mm"], _two["level_rms_by_quarter_mm"]
    assert _s4[3] > _s4[0] and _t4[3] < _t4[0] / 10  # (c): six terms' error grows, two's fades
    assert _six["end_level_error_mm"] > 0  # the six-term pond drains too slowly at the end
    mo.md(f"""
    **Why six terms do worse (checked).** On the records' rate, six terms fit a hair better
    than two (RMS residual {_six["records_rate_rms"]:.5e} against {_two["records_rate_rms"]:.5e}
    m/s): the extra weights fit the noise. Against the held-out storms' exact rate they are
    worse ({_six["heldout_rate_rms"]:.2e} against {_two["heldout_rate_rms"]:.2e} m/s). Run
    forward, their level error grows through the day (RMS by quarter
    {", ".join(f"{_v:.2f}" for _v in _s4)} mm) while two terms' fades
    ({", ".join(f"{_v:.2f}" for _v in _t4)} mm); at the end of the day the six-term pond
    stands {_six["end_level_error_mm"]:.1f} mm too high on average: it drains too slowly. Real
    data are rarely this clean; the fourth reveal (§5) shows a case.
    """)
    return


@app.cell
def _(
    DEPTH,
    DESIGN,
    FIG,
    INK,
    PALETTE,
    READ_MIN,
    STORM_HOURS,
    TRUE_TERM,
    WEIR_C,
    fit_quality,
    mo,
    model_rhs,
    np,
    plt,
    pond_level,
    same,
    shown,
    storm,
    weir,
    xi_missing,
    xi_orifice,
    xi_sindy,
):
    # The design storm (peak 6 m3/s at 2.5 h), run through the truth and each fitted equation
    design_t = np.arange(0, STORM_HOURS * 60 + READ_MIN, READ_MIN) * 60.0
    design_q = storm(design_t, DESIGN["peak"], DESIGN["tp_hours"] * 3600, DESIGN["shape"])
    design_levels = {
        "truth": pond_level(design_t, design_q, weir(WEIR_C)),
        "SINDy": pond_level(design_t, design_q, model_rhs(xi_sindy)),
        "orifice assumed": pond_level(design_t, design_q, model_rhs(xi_orifice)),
        f"no {TRUE_TERM} in the library": pond_level(design_t, design_q, model_rhs(xi_missing)),
    }
    design_peak = {_k: float(_v.max()) for _k, _v in design_levels.items()}
    print({_k: f"{_v:.6f}" for _k, _v in design_peak.items()})
    assert same(design_peak, "pond/design_peak")
    # The alt text: truth 2.31 m under the 2.5-m rim; SINDy 2.31; orifice 2.74, over the rim.
    # The notes: SINDy 2.309 against the truth's 2.309
    assert shown(design_peak["truth"], "2.31") and shown(design_peak["SINDy"], "2.31")
    assert shown(design_peak["truth"], "2.309") and shown(design_peak["SINDy"], "2.309")
    assert shown(design_peak["orifice assumed"], "2.74")
    assert design_peak["truth"] < DEPTH < design_peak["orifice assumed"]
    assert design_peak["SINDy"] < DEPTH
    _styles = {"truth": (INK, "-", 3), "SINDy": (PALETTE[0], "--", 2),
               "orifice assumed": (PALETTE[1], "-", 2)}  # fmt: skip
    _top = 1.08 * max([DEPTH, *design_peak.values()])
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    for _key, (_c, _ls, _lw) in _styles.items():
        _ax.plot(design_t / 3600, design_levels[_key], color=_c, ls=_ls, lw=_lw, label=_key)
    _ax.axhline(DEPTH, color=PALETTE[1], ls=":", lw=1.5)
    _ax.text(STORM_HOURS - 0.3, DEPTH + 0.03 * _top, "rim", ha="right", color=PALETTE[1])
    _ax.set(xlim=(0, STORM_HOURS), ylim=(0, _top), xlabel="hours",
            ylabel="level above the outlet, m")  # fmt: skip
    _ax.legend(loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.88))
    mo.vstack(
        [
            mo.md("### The pond holds; the orifice fit says no"),
            _fig,
            mo.md(f"""
    The decision. Run each fitted equation through the design storm, about twice the largest
    recorded peak. SINDy's model gives {design_peak["SINDy"]:.3f} m against the truth's
    {design_peak["truth"]:.3f} m: the pond holds under its {DEPTH}-m rim. The orifice model
    fitted the same records badly (R² = {fit_quality["orifice assumed"]["r2"]:.2f}) and says
    {design_peak["orifice assumed"]:.2f} m: overtopping, and a needless enlargement. A poor fit
    is a warning to heed before the design storm. Without the truth, how would the city know
    which to trust?
    """),
        ]
    )
    return design_levels, design_peak, design_q, design_t


@app.cell
def _(
    BOOT,
    CLOG,
    LAMBDA,
    LEVEL_NOISE,
    NAMES,
    N_STORMS,
    SEED,
    TRUE_TERM,
    WEIR_C,
    h_rec,
    h_true,
    mo,
    np,
    pond_level,
    q_rec,
    same,
    shown,
    stack,
    stlsq,
    terms,
    times,
    weir,
    xi_sindy,
):
    # Year 2: the same storms with the weir partly blocked (70% of its flow at a level); new
    # noise. Then a bootstrap over storms: resample the 20 (with replacement), refit both years,
    # and take the ratio of the h^1.5 coefficients
    _h2_true = np.array([pond_level(times, _q, weir(CLOG * WEIR_C)) for _q in q_rec])
    _h2 = _h2_true + np.random.default_rng([SEED, 3]).normal(0, LEVEL_NOISE, _h2_true.shape)
    _xi2 = stlsq(*stack(times, q_rec, _h2), LAMBDA)
    _k = NAMES.index(TRUE_TERM)
    # The same fit on the true levels of both years (no noise; central differences)
    _ratio_true = float(
        stlsq(*stack(times, q_rec, _h2_true), LAMBDA)[_k]
        / stlsq(*stack(times, q_rec, h_true), LAMBDA)[_k]
    )
    _rng = np.random.default_rng([SEED, 4])
    _ratios = []
    for _ in range(BOOT):
        _pick = _rng.integers(0, N_STORMS, N_STORMS)
        _a = stlsq(*stack(times, q_rec[_pick], h_rec[_pick]), LAMBDA)
        _b = stlsq(*stack(times, q_rec[_pick], _h2[_pick]), LAMBDA)
        _ratios.append(_b[_k] / _a[_k] if _a[_k] != 0 and _b[_k] != 0 else np.nan)
    _ratios = np.array(_ratios)
    clog = {
        "year2": terms(_xi2),
        "ratio": float(_xi2[_k] / xi_sindy[_k]),
        "ci90": np.nanpercentile(_ratios, [5, 95]).tolist(),
        "boot_missing": int(np.isnan(_ratios).sum()),
        "true": CLOG,
    }
    print(clog, f"on the true levels: {_ratio_true:.5f}")
    assert list(clog["year2"]) == ["Q_in", "h^1.5"]
    assert same(clog, "pond/clog") and clog["boot_missing"] == 0
    # The table: 0.702; 0.7004 to 0.7038 (200 resamples of the 20 storms); the truth 0.70
    assert shown(clog["ratio"], "0.702") and shown(CLOG, "0.70")
    assert shown(clog["ci90"][0], "0.7004") and shown(clog["ci90"][1], "0.7038")
    # The notes: "just above the truth (0.70): the level noise biases it slightly". Without the
    # noise the same fit's offset is under a fifth of it (0.0002 against 0.0020)
    assert CLOG < clog["ci90"][0] < clog["ratio"] < clog["ci90"][1] and BOOT == 200
    assert abs(_ratio_true - CLOG) < 0.2 * (clog["ratio"] - CLOG)
    mo.md(f"""
    ### Year 2: the weir passes 70% as much

    | Year 2 | h^1.5 coefficient ÷ year 1 |
    |:-------|---------------------------:|
    | SINDy | {clog["ratio"]:.3f} |
    | 90% bootstrap interval ({BOOT} resamples of the storms) | {clog["ci90"][0]:.4f} to {clog["ci90"][1]:.4f} |
    | Truth (the weir partly blocked) | {CLOG:.2f} |

    The same storms a year later, with the weir partly blocked (debris, sediment). SINDy's
    $h^{{1.5}}$ coefficient falls to {clog["ratio"]:.3f} of year 1's; a bootstrap over storms
    (resample the 20, refit both years) gives {clog["ci90"][0]:.4f} to {clog["ci90"][1]:.4f},
    just above the truth (0.70): the level noise biases it slightly. Resampling storms shows how
    much the estimate moves from storm to storm, not its offset; on the true levels (no noise)
    the same fit gives {_ratio_true:.4f}, so most of the offset ({clog["ratio"] - CLOG:.4f})
    comes with the noise. A
    coefficient that means something physical can be tracked over time: a maintenance trigger.
    What would a black-box model of the level have told the city?
    """)
    return (clog,)


@app.cell
def _(mo):
    mo.md(r"""
    ### A river reach that routes floods

    | The reach (assumed values) | Value |
    |:----------------------------------------|-----------------:|
    | Length, in reaches of 2 km | 60 km |
    | Width, slope, Manning's n | 50 m, 0.0005, 0.035 |
    | Base flow | 50 m³/s |
    | Training floods (random peaks 100–600 m³/s) | 1,000 |
    | Solver | variable-parameter Muskingum–Cunge |

    The task: route a 1,000-member forecast of inflow floods every few hours. Training data:
    1,000 random floods routed by the solver of §3.
    """)
    return


@app.cell
def _(
    BASE,
    BIG_PEAK,
    FLOOD_PEAK,
    LENGTH,
    MANNING,
    N_TRAIN,
    REACH,
    SLOPE,
    WIDTH_M,
    flood_sets,
    mo,
    np,
    same,
):
    # The slide's table, and the notes: floods travel faster when they are bigger (the solver's
    # median lag of the peak, 7 h in the training range, 5 h for bigger floods): nonlinear
    assert (LENGTH, REACH, WIDTH_M, SLOPE, MANNING, BASE) == (60_000, 2000, 50, 5e-4, 0.035, 50)
    assert N_TRAIN == 1000 and FLOOD_PEAK == (100, 600) and BIG_PEAK == (600, 1200)
    lag_hours = {
        _k: float(np.median(flood_sets[f"{_k}_out"].argmax(1) - flood_sets[f"{_k}_in"].argmax(1)))
        for _k in ("test", "big")
    }
    attenuation = float(np.median(flood_sets["test_out"].max(1) / flood_sets["test_in"].max(1)))
    print(f"median lag {lag_hours}; median peak out / in {attenuation:.4f}")
    assert lag_hours == {"test": 7.0, "big": 5.0}
    assert same(attenuation, "river/solver/attenuation")
    mo.md(f"""
    The solver's floods travel faster when they are bigger: the median peak arrives
    {lag_hours["test"]:.0f} h after the inflow's for floods in the training range and
    {lag_hours["big"]:.0f} h after it for the bigger floods of §5 (peaks 600–1,200 m³/s). The
    routing is nonlinear. A median peak leaves the reach at {100 * attenuation:.1f}% of its
    inflow.
    """)
    return attenuation, lag_hours


@app.cell
def _(mo):
    mo.md(r"""
    ### The operator: a Fourier layer, transform, keep, mix

    A neural operator learns a map from a function (the inflow over time) to a function (the
    outflow), not from a fixed list of numbers to another (Li et al., 2021). The Fourier neural
    operator (FNO):

    1. **Lift:** at every time step, the scaled inflow $(Q - 50)/500$ and the time
       $t/72\text{ h} \in [0, 1]$ go to 32 channels by one linear map (a 1 × 1 convolution).
    2. **Four Fourier layers:**

       $$v \leftarrow \sigma\big(W v + \mathcal{F}^{-1}(R \cdot \mathcal{F} v)\big)$$

       $\mathcal{F}$ is the discrete Fourier transform over time; $R$ multiplies each of the
       lowest 12 modes by a learned complex 32 × 32 matrix and drops the rest; $W$ is a
       pointwise linear map; $\sigma$ is the GELU (not after the last layer).
    3. **Project:** 32 → 64 → 1 channels at every time step, back to a scaled outflow.

    The weights act on frequencies and on channels, never on a time step's index, so the
    same network can read a hydrograph sampled hourly or every half hour (checked in §4's
    scores). Training: mean squared error on the scaled outflow, Adam at $10^{-3}$ with a cosine
    schedule to zero, batches of 32, 300 epochs, 3 seeds. The settings are fixed before any
    test, not tuned.
    """)
    return


@app.cell
def _(BASE, FNO_LAYERS, FNO_MODES, FNO_WIDTH, Q_SCALE, nn, np, torch):
    class SpectralConv(nn.Module):
        """Multiply the lowest ``modes`` Fourier modes by learned complex weights, drop the rest."""

        def __init__(self, width: int, modes: int) -> None:
            """``width`` channels in and out; ``modes`` kept."""
            super().__init__()
            self.modes = modes
            self.w = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width**2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """(B, C, N) -> (B, C, N)."""
            xf = torch.fft.rfft(x, norm="forward")  # (B, C, N // 2 + 1): the coefficients
            yf = torch.zeros_like(xf)
            yf[:, :, : self.modes] = torch.einsum("bin,ion->bon", xf[:, :, : self.modes], self.w)
            return torch.fft.irfft(yf, n=x.shape[-1], norm="forward")

    class FNO(nn.Module):
        """Lift, four Fourier layers (spectral + pointwise), project (Li et al., 2021)."""

        def __init__(self) -> None:
            """Sizes from the slides' constants."""
            super().__init__()
            self.lift = nn.Conv1d(2, FNO_WIDTH, 1)
            self.spectral = nn.ModuleList(
                SpectralConv(FNO_WIDTH, FNO_MODES) for _ in range(FNO_LAYERS)
            )
            self.pointwise = nn.ModuleList(
                nn.Conv1d(FNO_WIDTH, FNO_WIDTH, 1) for _ in range(FNO_LAYERS)
            )
            self.project = nn.Sequential(
                nn.Conv1d(FNO_WIDTH, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1)
            )

        def forward(self, q: torch.Tensor) -> torch.Tensor:
            """Scaled inflow (B, N) -> scaled outflow (B, N), on any N."""
            n = q.shape[-1]
            grid = torch.linspace(0, 1, n).expand(q.shape[0], n)  # where each sample sits
            x = self.lift(torch.stack([q, grid], 1))  # (B, W, N)
            for i, (s, w) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
                x = s(x) + w(x)
                if i < FNO_LAYERS - 1:
                    x = nn.functional.gelu(x)
            return self.project(x).squeeze(1)

    def to_net(q: np.ndarray) -> torch.Tensor:
        """m3/s -> the network's scale, float32."""
        return torch.tensor((q - BASE) / Q_SCALE, dtype=torch.float32)

    def count_weights(net: nn.Module) -> int:
        """Real numbers in a network's parameters (a complex weight counts twice)."""
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())

    return FNO, SpectralConv, count_weights, to_net


@app.cell
def _(FNO, ROUNDOFF, SEED, SpectralConv, count_weights, mo, np, torch):
    # 1. The spectral layer against a hand-written transform, in 64-bit floats: the lowest
    # modes' coefficients X_k = (1/N) sum_n x_n exp(-2 pi i k n / N), mixed by R, then
    # y_n = Re(Y_0) + 2 sum_{k>=1} Re(Y_k exp(2 pi i k n / N)), on an odd and an even length
    _g = torch.Generator().manual_seed(SEED)
    _layer = SpectralConv(5, 4)
    _layer.w.data = torch.randn(5, 5, 4, dtype=torch.cdouble, generator=_g)
    _dft_err = []
    for _n in (73, 144):
        _x = torch.randn(2, 5, _n, dtype=torch.float64, generator=_g)
        _xn = _x.numpy()
        _k, _t = np.arange(4), np.arange(_n)
        _basis = np.exp(-2j * np.pi * np.outer(_k, _t) / _n)  # (modes, N)
        _xf = _xn @ _basis.T / _n  # (B, C, modes)
        _yf = np.einsum("bik,iok->bok", _xf, _layer.w.data.numpy())
        _c = np.where(_k == 0, 1.0, 2.0)
        _y = np.real((_yf * _c) @ np.conj(_basis))  # (B, C, N)
        with torch.no_grad():
            _out = _layer(_x).numpy()
        assert _out.shape == _y.shape == (2, 5, _n)
        _dft_err.append(float(np.abs(_out - _y).max() / np.abs(_y).max()))
    assert max(_dft_err) < ROUNDOFF
    # 2. Modes above the kept ones are dropped: a cosine of 20 cycles over the record maps to 0
    _cos = torch.cos(2 * np.pi * 20 * torch.arange(73, dtype=torch.float64) / 73)
    with torch.no_grad():
        _zero = _layer(_cos.expand(1, 5, 73).clone())
    assert float(_zero.abs().max()) < ROUNDOFF
    # 3. The FNO: a forward pass on dummy inflows of two lengths, and its weights
    _net = FNO()
    with torch.no_grad():
        _o73, _o145 = _net(torch.zeros(3, 73)), _net(torch.zeros(3, 145))
    assert _o73.shape == (3, 73) and _o145.shape == (3, 145)
    n_weights = count_weights(_net)
    _parts = {
        "lift": 2 * 32 + 32,
        "spectral": 4 * 2 * 32 * 32 * 12,
        "pointwise": 4 * (32 * 32 + 32),
        "project": 32 * 64 + 64 + 64 + 1,
    }
    assert n_weights == sum(_parts.values()) == 104_801
    print(f"spectral layer vs the hand-written transform: {max(_dft_err):.1e}; weights {n_weights:,}"
          f" = {_parts}")  # fmt: skip
    mo.md(f"""
    | Part | Size |
    |:-----|:-----|
    | Lift | inflow and time → 32 channels |
    | Fourier layers | 4, each 12 modes |
    | Project | 32 → 64 → 1 |
    | Weights | {n_weights:,} |

    The spectral layer matches a hand-written Fourier transform to {max(_dft_err):.0e}
    (relative, 64-bit floats, on 73 and 144 samples), drops a cosine of 20 cycles entirely, and
    the FNO maps 73 or 145 samples to as many. Of the {n_weights:,} weights,
    {_parts["spectral"]:,} are the spectral layers' (complex numbers count twice).
    """)
    return (n_weights,)


@app.cell
def _(
    BASE,
    BATCH,
    EPOCHS_RUN,
    FNO,
    LR,
    POOL,
    Q_SCALE,
    SEED,
    SEEDS_RUN,
    THREADS,
    count_weights,
    flood_sets,
    get_reusable_executor,
    time,
    to_net,
    torch,
):
    def fno_job(seed: int, data: dict, epochs: int) -> dict:
        """Train one FNO (CPU, fixed threads, deterministic) and predict the evaluation sets.

        Runs in a worker process. Each epoch: a random order of the 1,000 training floods
        (from a generator seeded with ``seed``), batches of 32, mean squared error on the
        scaled outflow, Adam; a cosine schedule to zero over ``epochs``; the validation
        error (m3/s) after each epoch.
        """
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        net = FNO()
        x, y = to_net(data["train_in"]), to_net(data["train_out"])
        xv, yv = to_net(data["val_in"]), to_net(data["val_out"])
        opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0.0)  # fmt: skip
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        gen = torch.Generator().manual_seed(seed)
        val = []
        for _ in range(epochs):
            perm = torch.randperm(len(x), generator=gen)
            for i in range(0, len(x), BATCH):
                b = perm[i : i + BATCH]
                opt.zero_grad(set_to_none=True)
                (net(x[b]) - y[b]).pow(2).mean().backward()
                opt.step()
            sched.step()
            with torch.no_grad():
                val.append(Q_SCALE * float((net(xv) - yv).pow(2).mean().sqrt()))
        secs = time.perf_counter() - t0
        preds = {}
        with torch.no_grad():
            for k in ("test", "big", "fine", "fine_hourly"):
                t1 = time.perf_counter()
                preds[k] = BASE + Q_SCALE * net(to_net(data[f"{k}_in"])).numpy().astype(float)
                if k == "test":
                    preds["test_seconds"] = time.perf_counter() - t1
        return {"seed": seed, "train_seconds": secs, "val_curve": val, "preds": preds,
                "parameters": count_weights(net)}  # fmt: skip

    # loky's executor pickles the cell's functions and classes by value (cloudpickle), and
    # sets OMP_NUM_THREADS=4 in each worker, as the slides' script does before spawning its own
    _data = {_k: _v for _k, _v in flood_sets.items() if _k not in ("fine_out", "fine_hourly_out")}
    executor = get_reusable_executor(max_workers=POOL, env={"OMP_NUM_THREADS": str(THREADS)})
    fno_futures = [
        executor.submit(fno_job, SEED + _s, _data, EPOCHS_RUN) for _s in range(SEEDS_RUN)
    ]
    print(f"{len(fno_futures)} FNO(s) submitted: seeds {[SEED + _s for _s in range(SEEDS_RUN)]}, "
          f"{EPOCHS_RUN} epochs")  # fmt: skip
    return (fno_futures,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The unit hydrograph, for comparison

    Lecture 20's linear convolution, here from inflow to outflow: Lecture 20's unit hydrograph
    turned daily rain into flow and was fitted by ridge (penalty chosen on validation); this one
    turns the inflow above base flow into the outflow above it, $Q_{out}(t) - 50 = \sum_{j<k} u_j\,
    (Q_{in}(t - j) - 50)$, its $k$ weights fitted by plain least squares on the 1,000 training floods
    and its length $k$ chosen on the 200 validation floods over every length up to the record
    (6 to 72 hours). The search ends at the record's length, the longest memory a unit
    hydrograph can have, so the whole feasible range is covered, and the best may sit at its
    end. Being linear, it cannot make big floods travel faster (checked).
    """)
    return


@app.cell
def _(BASE, HOURS, UH_LADDER, flood_sets, mo, np, same):
    def lagged(q: np.ndarray, k: int) -> np.ndarray:
        """(batch, times) -> (batch, times, k): the last k hours of inflow above BASE."""
        e = q - BASE
        return np.stack([np.pad(e, ((0, 0), (j, 0)))[:, : e.shape[1]] for j in range(k)], -1)

    def unit_hydrograph(q_in: np.ndarray, q_out: np.ndarray, k: int) -> np.ndarray:
        """A unit hydrograph of k hours, inflow to outflow, by plain least squares (no ridge)."""
        x = lagged(q_in, k).reshape(-1, k)
        assert x.shape == (q_in.size, k)
        return np.linalg.lstsq(x, (q_out - BASE).ravel(), rcond=None)[0]

    assert UH_LADDER[-1] == HOURS
    uh_tried = []
    for _k in UH_LADDER:
        _u = unit_hydrograph(flood_sets["train_in"], flood_sets["train_out"], _k)
        _pred = BASE + lagged(flood_sets["val_in"], _k) @ _u
        uh_tried.append(
            {"hours": _k, "val_rmse": float(np.sqrt(np.mean((_pred - flood_sets["val_out"]) ** 2)))}
        )
    k_uh = min(uh_tried, key=lambda _r: _r["val_rmse"])["hours"]
    uh_weights = unit_hydrograph(flood_sets["train_in"], flood_sets["train_out"], k_uh)
    uh_pred = {
        _k: BASE + lagged(flood_sets[f"{_k}_in"], k_uh) @ uh_weights for _k in ("test", "big")
    }
    assert same(uh_tried, "river/uh/tried")
    assert k_uh == 72  # the notes: 72 hours, the whole record
    # Each longer memory helps, less and less: under 0.2% from 36 hours on
    _v = [_r["val_rmse"] for _r in uh_tried]
    assert np.all(np.diff(_v) < 0)
    assert _v[3] / _v[-1] - 1 < 0.002
    # Linear: twice the inflow above base gives twice the outflow above base, at the same hour,
    # so its peaks travel no faster for bigger floods
    _double = BASE + 2 * (flood_sets["test_in"] - BASE)
    _pred2 = BASE + lagged(_double, k_uh) @ uh_weights
    assert np.allclose(_pred2 - BASE, 2 * (uh_pred["test"] - BASE), rtol=1e-12, atol=1e-9)
    assert np.array_equal(_pred2.argmax(1), uh_pred["test"].argmax(1))
    uh_lag = {_k: float(np.median(uh_pred[_k].argmax(1) - flood_sets[f"{_k}_in"].argmax(1)))
              for _k in ("test", "big")}  # fmt: skip
    assert uh_lag == {"test": 7.0, "big": 7.0}
    print(f"validation RMSE by length {[(_r['hours'], round(_r['val_rmse'], 4)) for _r in uh_tried]}; "
          f"median lag {uh_lag}")  # fmt: skip
    mo.md(f"""
    The validation error falls with every longer memory and flattens past 36 hours
    ({uh_tried[3]["val_rmse"]:.4f} m³/s at 36, {uh_tried[-1]["val_rmse"]:.4f} at 72): the search
    keeps {k_uh} hours. The unit hydrograph's median peak lag is {uh_lag["test"]:.0f} h in the
    training range and {uh_lag["big"]:.0f} h for the bigger floods, where the solver's is
    5 h: twice the inflow gives exactly twice the outflow, at the same hour.
    """)
    return k_uh, uh_pred, uh_tried


@app.cell
def _(
    BASE,
    FNO_SEEDS,
    FNO_TOL,
    FULL,
    REF,
    SEEDS_RUN,
    fine_half_step_rmse,
    fine_sampling_rmse,
    flood_sets,
    fno_futures,
    k_uh,
    mo,
    n_weights,
    np,
    ref,
    same,
    same_magnitude,
    shown,
    solver_seconds,
    uh_pred,
):
    fnos = [_f.result() for _f in fno_futures]  # in submission order: seeds 24, 25, 26

    def _rmse(p: np.ndarray, o: np.ndarray) -> float:
        assert p.shape == o.shape
        return float(np.sqrt(np.mean((p - o) ** 2)))

    def _peak_err(p: np.ndarray, o: np.ndarray, how: str = "median") -> float:
        _e = np.abs(p.max(1) - o.max(1)) / o.max(1)
        return float(np.median(_e) if how == "median" else _e.max())

    def _timing_err(p: np.ndarray, o: np.ndarray) -> float:
        return float(np.median(np.abs(p.argmax(1) - o.argmax(1))))

    def scores(pred: dict) -> dict[str, float]:
        """RMSE, median (and largest) relative peak error, median peak timing error."""
        _d = flood_sets
        return {"test_rmse": _rmse(pred["test"], _d["test_out"]),
                "big_rmse": _rmse(pred["big"], _d["big_out"]),
                "test_peak": _peak_err(pred["test"], _d["test_out"]),
                "big_peak": _peak_err(pred["big"], _d["big_out"]),
                "test_peak_max": _peak_err(pred["test"], _d["test_out"], "max"),
                "big_peak_max": _peak_err(pred["big"], _d["big_out"], "max"),
                "test_timing_h": _timing_err(pred["test"], _d["test_out"]),
                "big_timing_h": _timing_err(pred["big"], _d["big_out"])}  # fmt: skip

    def water_balance(pred: np.ndarray) -> dict[str, object]:
        """An operator's volumes on the test floods: against the inflow and the solver's.

        Volumes are sums of the hourly flow above base flow over the 72 hours (times 3,600 s for
        m³), as in the slides' script.

        Returns volume out / in and volume / the solver's (lowest, highest over floods); the
        mean flow above the solver's (m³/s); and, for the flood with the most volume out / in,
        its ratio, inflow peak, excess over the inflow (m³, and as a mean flow over the
        record), and its volume and mean flow against the solver's.
        """
        _q_in, _q_sol = flood_sets["test_in"], flood_sets["test_out"]
        _v_in, _v_sol, _v = ((_a - BASE).sum(1) for _a in (_q_in, _q_sol, pred))
        _ratio, _vs_sol = _v / _v_in, _v / _v_sol
        _offset = (pred - _q_sol).mean(1)  # m3/s over the record, per flood
        _i = int(np.argmax(_ratio))
        return {
            "volume": {"min": float(_ratio.min()), "max": float(_ratio.max())},
            "volume_vs_solver": {"min": float(_vs_sol.min()), "max": float(_vs_sol.max())},
            "offset_vs_solver_m3s": {"min": float(_offset.min()), "max": float(_offset.max())},
            "most_water": {
                "ratio": float(_ratio[_i]),
                "inflow_peak": float(_q_in[_i].max()),
                "excess_m3": float(3600 * (_v[_i] - _v_in[_i])),
                "excess_m3s": float((_v[_i] - _v_in[_i]) / _q_in.shape[1]),
                "vs_solver": float(_vs_sol[_i]),
                "vs_solver_m3s": float(_offset[_i]),
            },
        }

    uh_scores = scores(uh_pred)
    fno_scores = [
        scores(_f["preds"]) | {
            "seed": _f["seed"],
            "fine_rmse": _rmse(_f["preds"]["fine"], flood_sets["fine_out"]),
            "fine_hourly_rmse": _rmse(_f["preds"]["fine_hourly"], flood_sets["fine_hourly_out"]),
            "fine_ratio": _rmse(_f["preds"]["fine"], flood_sets["fine_out"])
            / _rmse(_f["preds"]["fine_hourly"], flood_sets["fine_hourly_out"]),
            "train_seconds": _f["train_seconds"],
            "test_seconds": _f["preds"]["test_seconds"],
            "val_final": _f["val_curve"][-1],
            **water_balance(_f["preds"]["test"]),
            "parameters": _f["parameters"],
            # the share of the validation error lost over the last 30 epochs (all of them in
            # the reduced setting's 30), as the slides' script computes it
            "val_drop_last30": (_f["val_curve"][-min(31, len(_f["val_curve"]))] - _f["val_curve"][-1])
            / _f["val_curve"][-min(31, len(_f["val_curve"]))],
        }
        for _f in fnos
    ]  # fmt: skip
    _keys = ("test_rmse", "big_rmse", "test_peak", "big_peak", "fine_rmse", "fine_hourly_rmse",
             "fine_ratio", "train_seconds", "test_seconds", "val_drop_last30")  # fmt: skip
    fno_span = {_k: {"min": min(_s[_k] for _s in fno_scores), "max": max(_s[_k] for _s in fno_scores)}
                for _k in _keys}  # fmt: skip
    for _k in ("volume", "volume_vs_solver", "offset_vs_solver_m3s"):
        fno_span[f"{_k}_min"] = min(_s[_k]["min"] for _s in fno_scores)
        fno_span[f"{_k}_max"] = max(_s[_k]["max"] for _s in fno_scores)
    fno_span["most_water"] = max(
        (_s["most_water"] for _s in fno_scores), key=lambda _m: _m["ratio"]
    )
    for _s in fno_scores:
        print({_k: (round(_v, 6) if isinstance(_v, float) else _v) for _k, _v in _s.items()})
    print("unit hydrograph", {_k: round(_v, 6) for _k, _v in uh_scores.items()})
    # The unit hydrograph (deterministic): 17.1 m3/s, 4.2%, 1 h; bigger floods 81 m3/s
    assert all(same(uh_scores[_k], f"river/uh/{_k}") for _k in (
        "test_rmse", "big_rmse", "test_peak", "big_peak", "test_timing_h", "big_timing_h"))  # fmt: skip
    assert shown(uh_scores["test_rmse"], "17.1") and shown(100 * uh_scores["test_peak"], "4.2")
    assert shown(uh_scores["test_rmse"], "17") and shown(uh_scores["big_rmse"], "81")
    assert all(_s["parameters"] == n_weights == 104_801 for _s in fno_scores)
    # Timings (never reproducible): the slides' run's solver, FNO, and training seconds, read
    # from numbers.json and checked within a power of ten
    _t_ref = {"solver": ref("river/solver/test_seconds"),
              "test_min": ref("river/fno_span/test_seconds/min"),
              "test_max": ref("river/fno_span/test_seconds/max"),
              "train_max": ref("river/fno_span/train_seconds/max")}  # fmt: skip
    if REF is not None:
        assert same_magnitude(solver_seconds["test"], _t_ref["solver"])
    # The FNOs: the slides' run, seed by seed (the slides setting only: reduced trains less)
    if FULL:
        assert SEEDS_RUN == FNO_SEEDS == len(fno_scores) == 3
        assert all(same(_s, f"river/fno/{_i}", FNO_TOL) for _i, _s in enumerate(fno_scores))
        assert same(fno_span, "river/fno_span", FNO_TOL)
        assert all(_s["test_timing_h"] == 0 and _s["big_timing_h"] == 0 for _s in fno_scores)
        # The table and notes: 0.25-0.28 m3/s, 0.07%-0.09%, timing 0, 1.09 to 1.10 m3/s
        _sp = fno_span
        assert shown(_sp["test_rmse"]["min"], "0.25") and shown(_sp["test_rmse"]["max"], "0.28")
        assert shown(100 * _sp["test_peak"]["min"], "0.07")
        assert shown(100 * _sp["test_peak"]["max"], "0.09")
        assert shown(_sp["fine_rmse"]["min"], "1.09") and shown(_sp["fine_rmse"]["max"], "1.10")
        # ... and on the same 200 floods sampled hourly, 0.25 to 0.29: 3.8 to 4.3 times less
        assert shown(_sp["fine_hourly_rmse"]["min"], "0.25")
        assert shown(_sp["fine_hourly_rmse"]["max"], "0.29")
        assert shown(_sp["fine_ratio"]["min"], "3.8") and shown(_sp["fine_ratio"]["max"], "4.3")
        # The Fourier layer's notes: validation error still falling, 2.1% to 6.9% over the last
        # 30 epochs
        assert shown(100 * _sp["val_drop_last30"]["min"], "2.1")
        assert shown(100 * _sp["val_drop_last30"]["max"], "6.9")
        assert all(_s["val_drop_last30"] > 0.02 for _s in fno_scores)
        # Timings: the FNO's seconds for the 500 floods, and training per seed
        assert same_magnitude(_sp["test_seconds"]["min"], _t_ref["test_min"])
        assert same_magnitude(_sp["test_seconds"]["max"], _t_ref["test_max"])
        assert same_magnitude(_sp["train_seconds"]["max"], _t_ref["train_max"])
        # The title: three seeds, each far ahead of the unit hydrograph
        assert _sp["test_rmse"]["max"] < uh_scores["test_rmse"] / 10
        assert _sp["test_peak"]["max"] < uh_scores["test_peak"] / 10
    _fs = fno_span
    _secs = f"{_fs['test_seconds']['min']:.2f}–{_fs['test_seconds']['max']:.2f}"
    _slides_run = "" if REF is None else (
        f"the slides' run: {_t_ref['test_min']:.2f}–{_t_ref['test_max']:.2f} s, "
        f"{_t_ref['solver']:.2f} s, and {_t_ref['train_max']:.0f} s; ")  # fmt: skip
    mo.md(f"""
    ### In range, the FNO beats the UH

    | {len(flood_sets["test_in"])} test floods | Error, m³/s | Peak error (median) | Peak timing, h (median) | Seconds |
    |:----------------|------------:|-----------:|---------------:|--------:|
    | Unit hydrograph ({k_uh} h) | {uh_scores["test_rmse"]:.1f} | {100 * uh_scores["test_peak"]:.1f}% | {uh_scores["test_timing_h"]:.0f} | |
    | FNO, {len(fno_scores)} seed(s) | {_fs["test_rmse"]["min"]:.2f}–{_fs["test_rmse"]["max"]:.2f} | {100 * _fs["test_peak"]["min"]:.2f}%–{100 * _fs["test_peak"]["max"]:.2f}% | {max(_s["test_timing_h"] for _s in fno_scores):.0f} | {_secs} |
    | Solver | 0 | 0 | 0 | {solver_seconds["test"]:.2f} |

    On the test floods in the training range, the FNO errs {_fs["test_rmse"]["min"]:.2f} to
    {_fs["test_rmse"]["max"]:.2f} m³/s across its seeds, the unit hydrograph
    {uh_scores["test_rmse"]:.1f}; the median peak is off by at most
    {100 * _fs["test_peak"]["max"]:.2f}% (over seeds) against {100 * uh_scores["test_peak"]:.1f}%,
    and the worst of the 500 peaks by {100 * max(_s["test_peak_max"] for _s in fno_scores):.2f}%
    against {100 * uh_scores["test_peak_max"]:.1f}%. Trained on hourly floods, it routes floods sampled
    every 30 minutes with an error of {_fs["fine_rmse"]["min"]:.2f} to {_fs["fine_rmse"]["max"]:.2f}
    m³/s, without retraining; on the same floods sampled hourly it errs
    {_fs["fine_hourly_rmse"]["min"]:.2f} to {_fs["fine_hourly_rmse"]["max"]:.2f} m³/s, so the
    finer sampling multiplies its error by {_fs["fine_ratio"]["min"]:.1f} to
    {_fs["fine_ratio"]["max"]:.1f}. The solver routes those floods at the same 900-s internal step
    as the hourly ones, so only the input's sampling changes; that change alone moves the
    solver's own outflow by {fine_sampling_rmse:.2f} m³/s (RMSE at the hours), and had the step
    been halved too, the target would have moved by {fine_half_step_rmse:.2f} m³/s. Speed in this run: {_secs} s for the
    {len(flood_sets["test_in"])} floods (across seeds), the solver {solver_seconds["test"]:.2f} s;
    training took up to {_fs["train_seconds"]["max"]:.0f} s per seed ({_slides_run}timings vary on a
    shared CPU). Validation error was still
    falling at the end: by {100 * _fs["val_drop_last30"]["min"]:.1f}% to
    {100 * _fs["val_drop_last30"]["max"]:.1f}% over the last 30 epochs. This solver is fast
    already: a surrogate pays when the solver is slow and the runs are many.
    """)
    return fno_scores, fno_span, fnos, uh_scores


@app.cell
def _(flood_sets, np):
    def median_flood(key: str) -> int:
        """The flood of median peak in a set (the slides' figure picks the same)."""
        return int(np.argsort(flood_sets[f"{key}_in"].max(1))[len(flood_sets[f"{key}_in"]) // 2])

    return (median_flood,)


@app.cell
def _(
    INK,
    MUTED,
    PALETTE,
    flood_sets,
    fno_scores,
    fnos,
    median_flood,
    mo,
    np,
    plt,
    uh_pred,
):
    # The median-peak test flood: the solver, the FNO (seed 24, as the slide), the unit
    # hydrograph; and every seed's validation curve
    _i = median_flood("test")
    _hours = np.arange(flood_sets["test_in"].shape[1])
    _fno = fnos[0]["preds"]["test"][_i]
    _gap = float(np.abs(_fno - flood_sets["test_out"][_i]).max())
    _peak = float(flood_sets["test_out"][_i].max())
    _t_solver = int(flood_sets["test_out"][_i].argmax())
    _uh_peak, _t_uh = float(uh_pred["test"][_i].max()), int(uh_pred["test"][_i].argmax())
    # The alt text: "the FNO's outflow on top of it" (seed 24, as the slide's figure)
    print(f"median test flood {_i}: solver peak {_peak:.1f} m3/s at hour {_t_solver}; FNO's "
          f"largest gap {_gap:.3f} m3/s; unit hydrograph {_uh_peak:.1f} at hour {_t_uh}")  # fmt: skip
    if len(fnos[0]["val_curve"]) == 300:  # the slides' setting
        assert _gap < 0.01 * _peak
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained",
                                    gridspec_kw={"width_ratios": [1.6, 1]})  # fmt: skip
    _a1.plot(_hours, flood_sets["test_in"][_i], color=MUTED, ls=":", lw=1.5, label="inflow")
    _a1.plot(_hours, flood_sets["test_out"][_i], color=INK, ls="-", lw=3, label="solver")
    _a1.plot(_hours, _fno, color=PALETTE[1], ls="--", lw=2, label="FNO")
    _a1.plot(_hours, uh_pred["test"][_i], color=PALETTE[0], ls="-.", lw=2, label="unit hydrograph")
    _a1.set(xlim=(0, 48), xlabel="hours", ylabel="flow, m³/s")
    _a1.legend(loc="upper right", frameon=False, fontsize=10)
    for _f, _ls in zip(fnos, ("-", "--", ":"), strict=False):
        _a2.plot(np.arange(1, len(_f["val_curve"]) + 1), _f["val_curve"], color=PALETTE[0],
                 ls=_ls, label=f"seed {_f['seed']}")  # fmt: skip
    _a2.set(yscale="log", xlabel="epoch", ylabel="validation error, m³/s")
    _a2.legend(frameon=False, fontsize=10)
    mo.vstack(
        [
            mo.md("### The operator learns the routing"),
            _fig,
            mo.md(f"""
    Left: a test flood of median size (the solver's peak {_peak:.0f} m³/s at hour
    {_t_solver}). The FNO's outflow lies on the solver's, at most {_gap:.2f} m³/s off; the unit
    hydrograph peaks at {_uh_peak:.0f} m³/s at hour {_t_uh}. Right: each seed's validation
    error after every epoch; over the last 30 epochs it fell by
    {", ".join(f"{100 * _s['val_drop_last30']:.1f}%" for _s in fno_scores)}: still falling,
    though the cosine schedule brings the step to zero at the end.
    """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: predict four failures

    1. Pond: leave $h^{1.5}$ out of the library.
    2. Pond: level noise of 5 cm instead of 5 mm.
    3. River: floods up to twice as large as any in training.
    4. A real river's recessions (CAMELS, Lecture 20): find the law.

    Write what happens to each. Then a fifth check: the operator's water balance.
    """)
    return


@app.cell
def _(
    DEPTH,
    FIG,
    INK,
    PALETTE,
    STORM_HOURS,
    design_levels,
    design_peak,
    design_t,
    fit_quality,
    h_true,
    h_val,
    mo,
    near,
    plt,
    pond_fit,
    same,
    shown,
    sweep,
):
    # 1. A library without h^1.5: SINDy builds the weir from the terms it has
    _m = pond_fit["missing"]
    # "Over the recorded levels the combination fits the rate almost as well as SINDy": its R2
    # on the records' rate, and its level error on the 10 held-out storms (§4; their true
    # levels reach 1.31 m, within the records' 1.44 m)
    _fq, _fs = fit_quality["no h^1.5 in the library"], fit_quality["SINDy"]
    missing_val_rmse = _fq["val_level_rmse_m"]
    _sindy_val = _fs["val_level_rmse_m"]
    # SINDy's held-out level error is the sweep's at the same threshold (the same runs)
    assert near(_sindy_val, next(_s for _s in sweep if _s["lam"] == 0.05)["val_rmse_m"], 1e-12)
    print(f"no h^1.5: R2 {_fq['r2']:.4f} (SINDy {_fs['r2']:.4f}); held-out level error "
          f"{missing_val_rmse:.4f} m (SINDy {_sindy_val:.5f} m); held-out levels up to {h_val.max():.2f} m")  # fmt: skip
    assert h_val.max() < h_true.max() and _fq["r2"] > 0.9 and missing_val_rmse < 0.02
    assert list(_m) == ["Q_in", "h^0.5", "h", "h^2"]
    assert same(_m, "pond/missing")
    # The notes: +8.37e-05 h^0.5, -5.11e-04 h, -4.04e-04 h^2; records up to 1.44 m; the design
    # peak 2.19 m against 2.31: low
    assert shown(_m["h^0.5"], "+8.37e-05") and shown(_m["h"], "-5.11e-04")
    assert shown(_m["h^2"], "-4.04e-04") and shown(h_true.max(), "1.44")
    _key = "no h^1.5 in the library"
    assert shown(design_peak[_key], "2.19") and design_peak[_key] < design_peak["truth"]
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _styles = {"truth": (INK, "-", 3), "SINDy": (PALETTE[0], "--", 2),
               "orifice assumed": (PALETTE[1], "-", 2), _key: (PALETTE[3], "-.", 2)}  # fmt: skip
    for _k, (_c, _ls, _lw) in _styles.items():
        _ax.plot(design_t / 3600, design_levels[_k], color=_c, ls=_ls, lw=_lw, label=_k)
    _ax.axhline(DEPTH, color=PALETTE[1], ls=":", lw=1.5)
    _ax.axhline(h_true.max(), color=INK, ls=(0, (1, 3)), lw=1)
    _ax.text(
        STORM_HOURS - 0.3, h_true.max() + 0.04, "highest recorded level", ha="right", fontsize=9
    )
    _ax.set(
        xlim=(0, STORM_HOURS), ylim=(0, 3.0), xlabel="hours", ylabel="level above the outlet, m"
    )
    _ax.legend(loc="upper right", frameon=False, fontsize=10)
    mo.vstack(
        [
            mo.md("### A missing term fits, then misleads"),
            _fig,
            mo.md(f"""
    First reveal. Without $h^{{1.5}}$ in the library, SINDy builds the weir from what it has:
    {_m["h^0.5"]:+.2e} $h^{{0.5}}$, {_m["h"]:+.2e} $h$, {_m["h^2"]:+.2e} $h^2$. Over the
    recorded levels (up to {h_true.max():.2f} m) the combination fits the rate almost as well as
    SINDy (R² = {_fq["r2"]:.3f} against {_fs["r2"]:.3f}), though its held-out level error is
    {missing_val_rmse:.3f} m against SINDy's {_sindy_val:.4f} ({missing_val_rmse / _sindy_val:.0f}
    times as large). At the design storm's levels it peaks at {design_peak[_key]:.2f} m against
    {design_peak["truth"]:.2f}: low, the unsafe direction. The library is an assumption: put in
    every term the physics allows (orifice, weir), and check the model beyond the recorded range
    before trusting it there.
    """),
        ]
    )
    return


@app.cell
def _(
    LAMBDA,
    LEVEL_NOISE,
    NOISES,
    SEED,
    design_q,
    design_t,
    h_true,
    mo,
    model_rhs,
    np,
    pond_level,
    q_rec,
    same,
    shown,
    stack,
    stlsq,
    terms,
    times,
):
    # 2. More level noise: the terms SINDy keeps and the design storm's peak
    noise_runs = []
    _rates = {}
    _, _clean = stack(times, q_rec, h_true)
    for _k, _sd in enumerate(NOISES):
        _h = h_true + np.random.default_rng([SEED, 10 + _k]).normal(0, _sd, h_true.shape)
        _rows, _y = stack(times, q_rec, _h)
        _x = stlsq(_rows, _y, LAMBDA)
        _peak = float(pond_level(design_t, design_q, model_rhs(_x)).max())
        noise_runs.append({"sd_m": _sd, "terms": terms(_x), "n_terms": int((_x != 0).sum()),
                           "design_peak": _peak})  # fmt: skip
        # The rate's noise on interior readings, against sigma / (sqrt(2) dt)
        _d = (_y - _clean).reshape(h_true.shape)[:, 1:-1]
        _rates[_sd] = (float(_d.std()), _sd / (np.sqrt(2) * 300.0))
    for _r in noise_runs:
        print(_r)
    assert same(noise_runs, "pond/noise")
    # The table: 2 terms, 2.31 m; 2, 2.31; 5, 2.39
    assert [_r["n_terms"] for _r in noise_runs] == [2, 2, 5]
    assert [f"{_r['design_peak']:.2f}" for _r in noise_runs] == ["2.31", "2.31", "2.39"]
    assert shown(noise_runs[0]["sd_m"], "0.005") and shown(noise_runs[2]["sd_m"], "0.050")
    # The derivation (§3): the rate's noise is sigma / (sqrt(2) dt), 1.2e-5 m/s at 5 mm
    for _sd, (_meas, _theory) in _rates.items():
        assert abs(_meas / _theory - 1) < 0.05, (_sd, _meas, _theory)
    assert shown(_rates[LEVEL_NOISE][1], "1.2e-05")
    # The notes: "noise in the level becomes larger noise in dh/dt" (NOISE_CHECK)
    _rms_h, _rms_rate = float(np.sqrt(np.mean(h_true**2))), float(np.sqrt(np.mean(_clean**2)))
    rel_noise = {_sd: (_sd / _rms_h, _rates[_sd][0] / _rms_rate) for _sd in NOISES}
    print({_sd: (f"{_a:.3f}", f"{_b:.3f}") for _sd, (_a, _b) in rel_noise.items()})
    assert all(_b > _a for _a, _b in rel_noise.values())
    mo.md(f"""
    ### 5 cm of level noise adds false terms

    | Level noise | Terms kept | Design peak, m |
    |:------------|-----------:|---------------:|
    | {noise_runs[0]["sd_m"]:.3f} m | {noise_runs[0]["n_terms"]} | {noise_runs[0]["design_peak"]:.2f} |
    | {noise_runs[1]["sd_m"]:.3f} m | {noise_runs[1]["n_terms"]} | {noise_runs[1]["design_peak"]:.2f} |
    | {noise_runs[2]["sd_m"]:.3f} m | {noise_runs[2]["n_terms"]} | {noise_runs[2]["design_peak"]:.2f} |

    Second reveal. The derivative comes from differences of noisy levels, so noise in the
    level becomes larger noise in $dh/dt$: relative to each signal's RMS, 5 mm is
    {100 * rel_noise[0.005][0]:.1f}% of the level and {100 * rel_noise[0.005][1]:.1f}% of its
    rate; 5 cm, {100 * rel_noise[0.05][0]:.1f}% and {100 * rel_noise[0.05][1]:.0f}%. (Measured
    rate noise {_rates[0.005][0]:.2e} m/s at 5 mm, against $\\sigma/(\\sqrt2\\,\\Delta t)$ =
    {_rates[0.005][1]:.2e}.) At 2 cm SINDy still keeps the two true terms; at 5 cm it keeps
    {noise_runs[2]["n_terms"]}, and its design peak moves to {noise_runs[2]["design_peak"]:.2f} m.
    Remedies: better sensors, smoothing before differencing, or fitting the integrated equation
    instead of the derivative.
    """)
    return (noise_runs,)


@app.cell
def _(
    BIG_PEAK,
    FIG,
    FULL,
    INK,
    MUTED,
    PALETTE,
    flood_sets,
    fno_scores,
    fno_span,
    fnos,
    median_flood,
    mo,
    np,
    plt,
    shown,
    uh_pred,
    uh_scores,
):
    # 3. Floods up to twice as large as any in training: 500 with peaks of 600-1,200 m3/s
    _train_max = float(flood_sets["train_in"].max())
    _big = flood_sets["big_in"].max(1)
    big_range = {"train_max": _train_max, "big_min": float(_big.min()),
                 "big_median": float(np.median(_big)), "big_max": float(_big.max())}  # fmt: skip
    print({_k: round(_v, 1) for _k, _v in big_range.items()},
          f"ratio to the largest training peak: {_big.min() / _train_max:.3f} to {_big.max() / _train_max:.3f}"
          f" (median {np.median(_big) / _train_max:.2f})")  # fmt: skip
    # The notes: "beyond every training flood"; the alt text: 600 to 1200 m3/s at the peak
    assert _big.min() > _train_max and BIG_PEAK == (600, 1200)
    # (peaks drawn from 600 to 1,200 m3/s; the hourly samples fall just under a drawn peak)
    assert _big.min() > 0.99 * 600 and _big.max() <= 1200
    # "Up to twice as large as any in training": the largest is 2.00 times the largest training
    # peak (to two digits)
    assert f"{_big.max() / _train_max:.2f}" == "2.00"
    if FULL:  # the notes: from at most 0.28 to 13.5-16.0 m3/s; peaks off by up to 1.5%
        _fs = fno_span
        assert shown(_fs["test_rmse"]["max"], "0.28")
        assert shown(_fs["big_rmse"]["min"], "13.5") and shown(_fs["big_rmse"]["max"], "16.0")
        assert shown(100 * _fs["big_peak"]["max"], "1.5") and shown(_fs["big_rmse"]["max"], "16")
        assert all(_s["big_rmse"] > 10 * _s["test_rmse"] for _s in fno_scores)
        assert uh_scores["big_rmse"] > _fs["big_rmse"]["max"]
    _i = median_flood("big")
    _hours = np.arange(flood_sets["big_in"].shape[1])
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _ax.plot(_hours, flood_sets["big_in"][_i], color=MUTED, ls=":", lw=1.5, label="inflow")
    _ax.plot(_hours, flood_sets["big_out"][_i], color=INK, ls="-", lw=3, label="solver")
    _ax.plot(_hours, fnos[0]["preds"]["big"][_i], color=PALETTE[1], ls="--", lw=2, label="FNO")
    _ax.plot(_hours, uh_pred["big"][_i], color=PALETTE[0], ls="-.", lw=2, label="unit hydrograph")
    _ax.set(xlim=(0, 48), xlabel="hours", ylabel="flow, m³/s")
    _ax.legend(loc="upper right", frameon=False, fontsize=10)
    _fs = fno_span
    mo.vstack(
        [
            mo.md("### Past the training range, errors grow"),
            _fig,
            mo.md(f"""
    Third reveal. On 500 floods of 600 to 1,200 m³/s at the peak, beyond every training flood
    (the largest training peak is {_train_max:.0f} m³/s, the smallest of these {_big.min():.0f}:
    1.0 to {_big.max() / _train_max:.1f} times it, {np.median(_big) / _train_max:.1f} in the
    median), the FNO's error rises from at most {_fs["test_rmse"]["max"]:.2f} to
    {_fs["big_rmse"]["min"]:.1f}–{_fs["big_rmse"]["max"]:.1f} m³/s; its median peak is off by up
    to {100 * _fs["big_peak"]["max"]:.1f}% over the seeds, and the worst peak by
    {100 * max(_s["big_peak_max"] for _s in fno_scores):.1f}%. The unit hydrograph is worse still
    ({uh_scores["big_rmse"]:.0f} m³/s). Nothing in the network says it is extrapolating: flag
    members outside the training range and send them to the solver.
    """),
        ]
    )
    return


@app.cell
def _(
    FIG,
    HOME,
    LAMBDA,
    MUTED,
    PALETTE,
    PERIODS,
    RAIN_MM,
    REC_LAMBDAS,
    REC_NAMES,
    REF,
    curve_fit,
    find_daily,
    mo,
    np,
    plt,
    ref,
    same,
    shown,
    stlsq,
):
    # 4. Real recessions. CAMELS' home basin (Lecture 20): daily flow Q (mm/day) and rain. A
    # recession day: no more than 1 mm of rain today and the two days before, the flow falling
    # today and yesterday. The fall -dQ/dt = Q_{t-1} - Q_t against the mean flow of the two days
    with np.load(find_daily()) as _z:
        _b = int(np.flatnonzero(_z["gauge_id"] == HOME)[0])
        _names = _z["forcing_names"].tolist()
        _q = _z["flow_mm"][_b].astype(float)
        _p = _z["forcing"][_b, :, _names.index("prcp_mm")].astype(float)
        _dates = _z["dates"]
    _dq = _q[1:] - _q[:-1]
    _qm = 0.5 * (_q[1:] + _q[:-1])
    _dry = (_p[1:] <= RAIN_MM) & (_p[:-1] <= RAIN_MM) & (np.r_[0.0, _p[:-2]] <= RAIN_MM)
    _falling = (_dq < 0) & (np.r_[0.0, _dq[:-1]] < 0)
    _ok = _dry & _falling & np.isfinite(_dq) & (_qm > 0)
    _days = _dates[1:]
    _split = {}
    for _name in ("train", "test"):
        _start, _end = (np.datetime64(_d) for _d in PERIODS[_name])
        _split[_name] = _ok & (_days >= _start) & (_days <= _end)
    rec_x, rec_y = _qm[_split["train"]], -_dq[_split["train"]]
    rec_xt, rec_yt = _qm[_split["test"]], -_dq[_split["test"]]

    def rec_library(v: np.ndarray) -> np.ndarray:
        """The recession library: 1, Q^0.5, Q, Q^1.5, Q^2."""
        return np.column_stack([np.ones_like(v), v**0.5, v, v**1.5, v**2])

    def _rmse(pred: np.ndarray) -> float:
        assert pred.shape == rec_yt.shape
        return float(np.sqrt(np.mean((pred - rec_yt) ** 2)))

    # The power law -dQ/dt = a Q^b: the log-log line (recession analysis's fit) as the start,
    # then least squares on -dQ/dt itself (Levenberg-Marquardt), like every other model here
    _slope, _icpt = np.polyfit(np.log(rec_x), np.log(rec_y), 1)
    (_a, _bb), _ = curve_fit(lambda v, a, b: a * v**b, rec_x, rec_y,
                             p0=(np.exp(_icpt), _slope), method="lm")  # fmt: skip
    _k_lin = float(np.sum(rec_x * rec_y) / np.sum(rec_x * rec_x))  # a linear reservoir
    recession = {
        "days": {"train": int(_split["train"].sum()), "test": int(_split["test"].sum())},
        "power_law": {"a": float(_a), "b": float(_bb), "b_log_fit": float(_slope),
                      "test_rmse": _rmse(_a * rec_xt**_bb)},
        "linear": {"k": _k_lin, "test_rmse": _rmse(_k_lin * rec_xt)},
        "corr_max": float(np.max(np.abs(np.corrcoef(rec_library(rec_x)[:, 1:].T) - np.eye(4)))),
        "sindy": [],
    }  # fmt: skip
    for _lam in REC_LAMBDAS:
        _xi = stlsq(rec_library(rec_x), rec_y, _lam)
        _kept = {_n: float(_c) for _n, _c in zip(REC_NAMES, _xi, strict=True) if _c != 0}
        recession["sindy"].append(
            {"lam": _lam, "terms": _kept, "test_rmse": _rmse(rec_library(rec_xt) @ _xi)}
        )
    print(recession)
    _r = recession
    assert _r["days"] == {"train": 1699, "test": 604}
    assert all(same(_r[_k], f"recession/{_k}") for _k in ("days", "power_law", "linear",
                                                          "corr_max", "sindy"))  # fmt: skip
    if REF is not None:  # the same terms kept, in the same order
        assert [list(_s["terms"]) for _s in _r["sindy"]] == [
            list(_s["terms"]) for _s in ref("recession/sindy")
        ]
    # The notes: 1,699 and 604 days; b = 1.73; 0.0915; five terms 0.0932, three 0.0928, two
    # 0.0931; linear 0.1524; Q^0.5 -0.123; correlation up to 0.97; 1.73 against 1.26
    _pl, _sd = _r["power_law"], _r["sindy"]
    assert shown(_pl["b"], "1.73") and shown(_pl["b_log_fit"], "1.26")
    assert shown(_pl["test_rmse"], "0.0915") and shown(_r["linear"]["test_rmse"], "0.1524")
    assert [len(_s["terms"]) for _s in _sd] == [5, 5, 5, 3, 2]
    assert shown(_sd[0]["test_rmse"], "0.0932") and shown(_sd[3]["test_rmse"], "0.0928")
    assert shown(_sd[4]["test_rmse"], "0.0931") and shown(_sd[0]["terms"]["Q^0.5"], "-0.123")
    assert shown(_r["corr_max"], "0.97") and _sd[2]["lam"] == LAMBDA
    # The alt text: the training days run from WY1990 to WY2004 (October 1989 to September 2004)
    assert PERIODS["train"] == ("1989-10-01", "2004-09-30")
    _xi5 = stlsq(rec_library(rec_x), rec_y, REC_LAMBDAS[0])  # the smallest threshold
    _xi2 = stlsq(rec_library(rec_x), rec_y, REC_LAMBDAS[-1])  # the largest
    rec_fits = {"power law": lambda v: _a * v**_bb,
                "SINDy, 5 terms": lambda v: rec_library(v) @ _xi5,
                "SINDy, 2 terms": lambda v: rec_library(v) @ _xi2}  # fmt: skip
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _ax.scatter(rec_x, rec_y, s=5, color=MUTED, alpha=0.4, linewidths=0, rasterized=True)
    _qq = np.geomspace(rec_x.min(), rec_x.max(), 100)
    for (_label, _f), _c, _ls in zip(rec_fits.items(), (PALETTE[1], PALETTE[0], PALETTE[2]),
                                     ("-", "--", "-."), strict=True):  # fmt: skip
        _ax.plot(_qq, _f(_qq), color=_c, ls=_ls, lw=2, label=_label)
    _ax.set(xscale="log", yscale="log", xlabel="flow Q, mm/day", ylabel="fall −dQ/dt, mm/day²")
    _ax.legend(loc="upper left", frameon=False, fontsize=10)
    mo.vstack([mo.md("### Real rivers: many laws fit alike"), _fig])
    return rec_fits, rec_x, rec_xt, rec_yt, recession


@app.cell
def _(mo, np, rec_fits, rec_x, rec_xt, rec_yt, recession, shown):
    # The alt text: the laws are "close at the highest flows and apart below them"; the notes:
    # "the curved laws tie, and their coefficients disagree, even in sign (Q^1.5: -0.106 ...
    # +0.016)".
    # Measured: the ratio of the largest to the smallest of the three curves at quantiles of the
    # training days' flow (RECESSION_CHECK)
    _qs = {"lowest": rec_x.min(), "5th percentile": np.percentile(rec_x, 5),
           "median": np.median(rec_x), "75th percentile": np.percentile(rec_x, 75),
           "95th percentile": np.percentile(rec_x, 95), "highest": rec_x.max()}  # fmt: skip
    rec_spread = {}
    for _name, _v in _qs.items():
        _vals = np.array([float(_f(np.array([_v]))[0]) for _f in rec_fits.values()])
        rec_spread[_name] = (float(_v), float(_vals.max() / _vals.min()))
    print({_k: (round(_a, 4), round(_b, 3)) for _k, (_a, _b) in rec_spread.items()})
    # Close (within 15%) from the 75th percentile of the flows up; a factor of 1.4 or more at
    # the median flow; over 100 at the lowest flow
    assert all(rec_spread[_k][1] < 1.15 for _k in ("75th percentile", "95th percentile", "highest"))
    assert rec_spread["median"][1] > 1.4 and rec_spread["lowest"][1] > 100
    # Where the test error is made: the share of each law's squared test error on the test days
    # above the training days' 75th percentile of flow
    _hi = rec_xt > np.percentile(rec_x, 75)
    high_share = {}
    for _name, _f in rec_fits.items():
        _se = (_f(rec_xt) - rec_yt) ** 2
        high_share[_name] = float(_se[_hi].sum() / _se.sum())
    print(f"{_hi.mean():.3f} of the test days lie above the 75th percentile; their share of the "
          f"squared test error {high_share}")  # fmt: skip
    assert all(_v > 0.9 for _v in high_share.values()) and _hi.mean() < 0.2
    _sd = recession["sindy"]
    _curved = [recession["power_law"]["test_rmse"], *(_s["test_rmse"] for _s in _sd)]
    _gap = max(_curved) - min(_curved)
    # The curved laws' test errors lie within 0.002 mm/day2 of each other; the linear reservoir's
    # is 0.06 more
    assert _gap < 0.002 < recession["linear"]["test_rmse"] - max(_curved)
    # Q^1.5's coefficient: negative in the five-term law, positive in the three-term one
    assert shown(_sd[0]["terms"]["Q^1.5"], "-0.106") and shown(_sd[3]["terms"]["Q^1.5"], "+0.016")
    assert _sd[0]["terms"]["Q^1.5"] < 0 < _sd[3]["terms"]["Q^1.5"]
    assert "Q^0.5" not in _sd[3]["terms"] and "Q^0.5" not in _sd[4]["terms"]
    mo.md(f"""
    Fourth reveal. North Fork Vermilion River (Lecture 20's home basin),
    {recession["days"]["train"]:,} dry falling days to fit (water years 1990–2004),
    {recession["days"]["test"]} to test (water years 2009–2014). On the test days: the one-term power law
    $-dQ/dt = a Q^b$ ($b$ = {recession["power_law"]["b"]:.2f}, the form of recession analysis;
    Brutsaert & Nieber, 1977) errs {recession["power_law"]["test_rmse"]:.4f} mm/day²; SINDy
    with five terms {_sd[0]["test_rmse"]:.4f}, three {_sd[3]["test_rmse"]:.4f}, two
    {_sd[4]["test_rmse"]:.4f}; a linear reservoir {recession["linear"]["test_rmse"]:.4f}. The
    curved laws tie (within {_gap:.4f}), and their coefficients disagree, even in sign
    ($Q^{{1.5}}$: {_sd[0]["terms"]["Q^1.5"]:+.3f} in the five-term law,
    {_sd[3]["terms"]["Q^1.5"]:+.3f} in the three-term law; $Q^{{0.5}}$:
    {_sd[0]["terms"]["Q^0.5"]:.3f} in the five-term law, absent from the others). The library's
    columns are nearly collinear (correlation up to {recession["corr_max"]:.2f}). The data
    cannot pick the law; even the exponent depends on how it is fitted:
    {recession["power_law"]["b"]:.2f} on the falls themselves, {recession["power_law"]["b_log_fit"]:.2f}
    on their logarithms.

    **Where the laws agree (checked).** The largest of the three curves over the smallest, at
    the training days' flows: {", ".join(f"{_k} ({_q:.3g} mm/day) {_r:.2f}" for _k, (_q, _r) in rec_spread.items())}.
    They agree within 15% from the 75th percentile of the flows up, where the falls are large
    and the test error is made ({100 * _hi.mean():.0f}% of the test days, over
    {100 * min(high_share.values()):.0f}% of each law's squared error); at the median flow,
    the middle of the cloud, they differ by a
    factor of {rec_spread["median"][1]:.1f}, and at the lowest flow by
    {rec_spread["lowest"][1]:.0f}: close at the highest flows and apart below them.
    """)
    return


@app.cell
def _(
    BASE,
    FULL,
    flood_sets,
    fno_scores,
    fno_span,
    long_volume,
    mo,
    np,
    same,
    shown,
):
    # 5. The water balance after 72 hours: volume out / in for the 500 test floods
    _vin = (flood_sets["test_in"] - BASE).sum(1)
    _sv = (flood_sets["test_out"] - BASE).sum(1) / _vin
    volume_72h = {"min": float(_sv.min()), "max": float(_sv.max())}
    assert same(volume_72h, "river/solver/volume_72h")
    assert shown(volume_72h["min"], "0.947") and shown(volume_72h["max"], "0.998")
    assert shown(100 * volume_72h["min"], "94.7") and shown(100 * volume_72h["max"], "99.8")
    # The notes: the shortfall is mostly the method's, not water still on its way. On the 50
    # floods routed for 200 hours more (§3): the share of each flood's shortfall at hour 72 that
    # comes out later
    _n = len(long_volume)
    _extra = long_volume - _sv[:_n]
    _later = np.clip(_extra, 0, None) / (1 - _sv[:_n])  # (round-off can make 0 a tiny negative)
    still_coming = {"extra_max": float(_extra.max()), "later_share_min": float(_later.min()),
                    "later_share_median": float(np.median(_later)),
                    "later_share_max": float(_later.max()),
                    "lost_max": float(1 - long_volume.min())}  # fmt: skip
    print(f"72 h: {volume_72h}; after 200 h more: {still_coming}")
    # Both happen: one flood's shortfall is all lost (the lowest volume does not move), another's
    # still partly on its way
    assert _extra.min() > -1e-12 and _later.min() < 1e-9 and _extra.max() > 0.005
    assert np.median(_later) < 0.01  # for the median flood, the shortfall is lost, not late
    assert same(still_coming["extra_max"], "river/solver/after_72h_max")  # 1.1% at most
    assert shown(100 * still_coming["extra_max"], "1.1")
    _mw = fno_span["most_water"]  # the flood with the most volume out / in, over seeds
    if FULL:  # the table and notes (D17): absolute terms, and the FNO against the solver
        assert shown(fno_span["volume_min"], "0.946") and shown(fno_span["volume_max"], "1.070")
        assert shown(100 * fno_span["volume_max"], "107.0") and fno_span["volume_max"] > 1
        assert all(_s["volume"]["max"] > 1 for _s in fno_scores)
        assert shown(fno_span["volume_vs_solver_min"], "0.982")
        assert shown(fno_span["volume_vs_solver_max"], "1.085")
        assert shown(fno_span["offset_vs_solver_m3s_min"], "-0.30")
        assert shown(fno_span["offset_vs_solver_m3s_max"], "0.31")
        assert shown(_mw["inflow_peak"], "105") and shown(_mw["excess_m3"], "68011")
        assert shown(_mw["excess_m3s"], "0.26")
        # "the size of the FNO's in-range error": the mean excess within its RMSE's range
        assert fno_span["test_rmse"]["min"] <= _mw["excess_m3s"] <= fno_span["test_rmse"]["max"]
    _rows = "\n".join(f"| FNO, seed {_s['seed']}: volume out ÷ in | {_s['volume']['min']:.3f} | {_s['volume']['max']:.3f} |"
                      for _s in fno_scores)  # fmt: skip
    mo.md(f"""
    ### The operator does not conserve water

    | {len(flood_sets["test_in"])} test floods, after 72 h | Lowest | Highest |
    |:--------------------------------------------|-------:|--------:|
    | Solver: volume out ÷ in | {volume_72h["min"]:.3f} | {volume_72h["max"]:.3f} |
    | FNO, {len(fno_scores)} seed(s): volume out ÷ in | {fno_span["volume_min"]:.3f} | {fno_span["volume_max"]:.3f} |
    {_rows}
    | FNO ÷ solver, flood by flood | {fno_span["volume_vs_solver_min"]:.3f} | {fno_span["volume_vs_solver_max"]:.3f} |
    | FNO − solver, mean flow, m³/s | {fno_span["offset_vs_solver_m3s_min"]:.2f} | {fno_span["offset_vs_solver_m3s_max"]:.2f} |

    After 72 hours the solver has passed {100 * volume_72h["min"]:.1f}% to
    {100 * volume_72h["max"]:.1f}% of each flood's volume. The shortfall is mostly the
    method's, not water still on its way. Routed for 200 hours more (the 50 floods of §3), a flood
    passes at most {100 * still_coming["extra_max"]:.1f} points more; of each flood's
    shortfall at hour 72, between {100 * still_coming["later_share_min"]:.0f}% and
    {100 * still_coming["later_share_max"]:.0f}% comes out later (median
    {100 * still_coming["later_share_median"]:.0f}%), and the rest, up to
    {100 * still_coming["lost_max"]:.1f}% of a flood, is lost. The FNO, on the same floods, passes up to
    {100 * fno_span["volume_max"]:.1f}%: more water out than came in. In absolute terms that is
    small: the flood with the most (an inflow peak of {_mw["inflow_peak"]:.0f} m³/s) gains
    {_mw["excess_m3"]:,.0f} m³ over the 72 hours, a mean of {_mw["excess_m3s"]:.2f} m³/s, the size of
    the FNO's in-range error ({fno_span["test_rmse"]["min"]:.2f}–{fno_span["test_rmse"]["max"]:.2f} m³/s). Flood by flood, the FNO's volume is
    {fno_span["volume_vs_solver_min"]:.3f} to {fno_span["volume_vs_solver_max"]:.3f} times the solver's, a mean flow from
    {fno_span["offset_vs_solver_m3s_min"]:.2f} to {fno_span["offset_vs_solver_m3s_max"]:.2f} m³/s off. Nothing in its architecture
    conserves volume; a water-balance check catches it, as in Lecture 23, and a percentage on a
    small flood makes a small offset look large.
    """)
    return (volume_72h,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Vote: what routes tomorrow's forecasts?

    - **A.** The solver, every member.
    - **B.** The FNO, every member.
    - **C.** The unit hydrograph.
    - **D.** The FNO, and the solver for members beyond the training floods.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(fno_span, mo, ref, same_magnitude, solver_seconds, uh_scores):
    # The notes: the solver's seconds for 500 floods, so 1,000 members take seconds
    _slides = ref("river/solver/test_seconds")  # the slides' run (numbers.json)
    _members = 2 * solver_seconds["test"]
    assert _slides is None or same_magnitude(_members, 2 * _slides)
    _in_slides = "" if _slides is None else f" ({_slides:.2f} s in the slides')"
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        A: the solver took {solver_seconds["test"]:.2f} s for 500 floods in this run{_in_slides},
        so 1,000 members take about {_members:.0f} s: fast
        enough every hour.
        B: {fno_span["test_rmse"]["max"]:.2f} m³/s in range, but up to
        {fno_span["big_rmse"]["max"]:.0f} on bigger floods and {fno_span["fine_rmse"]["max"]:.2f} on half-hourly ones, and its
        water balance is off the solver's by up to {fno_span["offset_vs_solver_m3s_max"]:.2f} m³/s. C:
        {uh_scores["test_rmse"]:.0f} m³/s even in range. D is the general answer when the
        solver is slow; with this solver, A is the defensible one: use the model you can check
        when it is cheap enough.
        """)
        }
    )
    return


@app.cell
def _(design_peak, mo):
    mo.md(f"""
    ### When to learn the dynamics

    | You have | Reach for |
    |:---------|:----------|
    | The equation and a fast solver | the solver |
    | The equation, unknown parameters | least squares on the solver (Lecture 23) |
    | Candidate terms and clean data | SINDy, then check it beyond the data |
    | A slow solver and many runs | a neural operator, checked against the solver |

    The first row is today's river: the solver is cheap. The second is Lecture 23. The third is
    the pond: candidate terms from the physics, clean data, and a check beyond the data. The
    fourth is where neural operators are proposed: solvers that take long per run, many runs,
    inputs inside the training range; score the operator against the solver on new cases.

    **Impact (economic, social).** A learned equation inherits its library and its data. The
    orifice fit would have the city enlarge a pond that holds ({design_peak["orifice assumed"]:.2f} m
    predicted against a true {design_peak["truth"]:.2f} m): money spent for nothing. The fit
    without $h^{{1.5}}$ errs the other way ({design_peak["no h^1.5 in the library"]:.2f} m):
    a pond judged safe on it would be judged on a level lower than the truth, and the people
    downstream carry that risk. Report the library and the range of the records with any
    learned equation used for design.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    ### Exit question

    The city's consultant fits a pond's outlet law with SINDy from one year of records and uses
    it to size a spillway for a storm twice as large. What would you check before approving?
    """)
    return


@app.cell
def _(design_peak, mo):
    mo.accordion(
        {
            "Answer": mo.md(f"""
        Any two of: the library holds every outlet law the physics allows; the fitted model is
        checked on storms it did not see and beyond the recorded levels (the missing-term model
        peaked at {design_peak["no h^1.5 in the library"]:.2f} m against
        {design_peak["truth"]:.2f}); the level noise and the derivative's quality; a bootstrap
        over storms for the coefficients; the model's water balance.
        """)
        }
    )
    return


@app.cell
def _(
    FNO_TOL,
    FULL,
    REF,
    STEP_S,
    after_72h_max,
    attenuation,
    checks_out,
    clog,
    design_peak,
    final_scaled,
    fine_half_step_rmse,
    fine_sampling_rmse,
    finer_grid_max_diff,
    fit_quality,
    fno_scores,
    fno_span,
    h_true,
    k_uh,
    lag_hours,
    long_volume,
    ls_scaled,
    noise_floor,
    noise_runs,
    pond_fit,
    q_rec,
    recession,
    sweep,
    threshold,
    uh_scores,
    uh_tried,
    volume_72h,
):
    # Every number of this run against the slides' numbers.json (timings left out; the two
    # round-off checks left out, bounded in §3; the FNOs only in the slides' setting)
    numbers = {
        "pond": {
            "max_level": float(h_true.max()), "recorded_peak_max": float(q_rec.max()),
            "checks": {"exact_terms": checks_out["exact_terms"]},
            **pond_fit, "sweep": sweep, "noise_floor_m": noise_floor,
            "frames": {"least_squares": ls_scaled.tolist(), "final": final_scaled.tolist(),
                       "threshold": threshold},
            "design_peak": design_peak, "clog": clog, "noise": noise_runs,
            "fit_quality": fit_quality,
        },
        "recession": recession,
        "river": {
            "solver": {"step_s": STEP_S,
                       "long_volume": {"min": float(long_volume.min()), "max": float(long_volume.max())},
                       "after_72h_max": after_72h_max,
                       "finer_grid_max_diff": finer_grid_max_diff,
                       "fine_half_step_rmse": fine_half_step_rmse,
                       "fine_sampling_rmse": fine_sampling_rmse, "lag_hours": lag_hours,
                       "attenuation": attenuation, "volume_72h": volume_72h},
            "uh": {"hours": k_uh, "tried": uh_tried, **uh_scores},
        },
    }  # fmt: skip
    if FULL:
        numbers["river"]["fno"] = fno_scores
        numbers["river"]["fno_span"] = fno_span

    def pairs(mine: object, ref: object, path: str = ""):
        """Yield (path, mine, ref) for every number in both; timings left out."""
        if isinstance(mine, dict) and isinstance(ref, dict):
            for key in sorted(set(mine) & set(ref), key=str):
                if "seconds" not in str(key):
                    yield from pairs(mine[key], ref[key], f"{path}/{key}")
        elif isinstance(mine, list | tuple) and isinstance(ref, list):
            assert len(mine) == len(ref), path
            for i, (a, b) in enumerate(zip(mine, ref, strict=True)):
                yield from pairs(a, b, f"{path}[{i}]")
        elif isinstance(mine, int | float):
            yield path, mine, ref

    if REF is None:
        print("numbers.json not found (build the slides' figures): comparison skipped")
        compared = {}
    else:
        _ref = {_k: REF[_k] for _k in ("pond", "recession", "river")}
        _all = list(pairs(numbers, _ref))
        _rel = [(abs(_a - _b) / abs(_b) if _b != 0 else abs(_a), _p) for _p, _a, _b in _all]
        _fno = [_r for _r in _rel if "/fno" in _r[1]]
        _rest = [_r for _r in _rel if "/fno" not in _r[1]]
        compared = {
            "numbers": len(_all),
            "worst": max(_rest),
            "worst_fno": max(_fno, default=(0.0, "")),
        }
        print(f"compared {len(_all)} numbers with the slides' run (timings left out): largest "
              f"relative difference {compared['worst'][0]:.1e} at {compared['worst'][1]}; FNOs "
              f"{compared['worst_fno'][0]:.1e} at {compared['worst_fno'][1]}")  # fmt: skip
        assert compared["worst"][0] <= 1e-9
        if FULL:  # the slides setting: the FNOs too, every number
            assert compared["worst_fno"][0] <= FNO_TOL
    return (compared,)


@app.cell
def _(T_START, compared, mo, time):
    _minutes = (time.perf_counter() - T_START) / 60
    _n = compared.get("numbers", 0)
    mo.md(f"""
    ## Gotchas

    - **Scale the columns before thresholding.** The inflow is in m³/s and the levels in
      metres; a threshold on raw coefficients compares units, not importance.
    - **Refit after dropping.** The kept weights change when a column goes; one more round of
      thresholding should change nothing (checked in §3).
    - **Differencing amplifies noise by $1/\\Delta t$.** Smooth first, or fit the integrated
      equation; reading more often makes the rate noisier.
    - **Run the fitted equation forward,** on storms it did not see and beyond the recorded
      levels: a good fit of the derivative is not a good model of the level.
    - **The library is an assumption.** Put in every term the physics allows; a missing term
      is replaced by a combination that fits the records and fails beyond them (§5).
    - **Scale the operator's inputs and outputs** ($(Q - 50)/500$ here), and give it the time
      as a channel; predictions come back in m³/s.
    - **Fixed threads per process.** The order of floating-point sums can depend on the thread
      count; each FNO trains in its own worker process with 4 threads, deterministic algorithms,
      and its own seed, as in the slides' run.
    - **`torch.no_grad()`** for every prediction.
    - **Time.** This run took {_minutes:.0f} minutes; {_n} numbers were compared with the
      slides' run below the asserts.

    ## Going deeper (Forsyth)

    - §10.4.1: new explanatory variables made from functions of one measurement, and Forsyth's
      warning that the training error never rises as such columns are added. Today's library is
      this idea with two measurements (the inflow and the level), and the warning is why the
      threshold was swept on held-out storms.
    - §11.1.3–11.1.4: choosing among models by cross-validation, and greedy forward and
      backward stagewise searches over sets of variables. STLSQ is a backward search that drops
      every column under the threshold at once and refits.
    - §11.4.1: L1 regularization (the lasso) and sparse models, and the lasso's habit of
      keeping one of several correlated variables and dropping the others. The recession
      library's columns correlate up to 0.97: there, which terms survive depends on the
      threshold.
    - §10.2.5: transforming variables. A power law becomes a line on log axes; fitting that
      line minimizes errors in the logarithm (relative errors), fitting the curve itself
      minimizes errors in the falls: two questions, two exponents (1.26 and 1.73).
    - §17.1.1–17.1.2: convolution. The unit hydrograph is a 1-D convolution with 72 weights;
      the FNO's lift, pointwise maps, and projection are 1 × 1 convolutions (the same map at
      every time step), and its spectral layer is, by the convolution theorem, a convolution
      whose kernel spans the whole record.

    **Exercises.** (1) Fit the integrated equation instead of the derivative: regress each
    30-minute change of the level on the candidate terms integrated over the same 30 minutes
    (the trapezoid rule), and redo the 5-cm noise reveal. (2) Make the FNO conserve volume:
    rescale each predicted outflow above base flow so its volume over 72 hours equals the
    solver's share (or the inflow's), and redo the water-balance table and the in-range scores.

    ## Things to remember

    - SINDy: a library, least squares, a threshold.
    - The library must hold the true term.
    - Noise in the derivative adds false terms.
    - An operator maps a function to a function.
    - Check learned models where you will use them.

    **Next:** agents that plan, call tools, and check.
    """)
    return


if __name__ == "__main__":
    app.run()
