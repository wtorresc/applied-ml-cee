"""Applied Machine Learning for CEE — Lecture 23: Physics-Informed Networks.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l23_physics_informed.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: none to fetch. The site's values are assumed and its piezometer readings simulated from
Terzaghi's (1943) series solution; the river uses Ogata and Banks's (1961) exact solution.
Set ``L23_SETTING=slides`` or ``L23_SETTING=reduced`` (the default) to choose the setting (see
the header cell); the choice never depends on the machine.
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
    # Applied Machine Learning for CEE — Lecture 23: Physics-Informed Networks

    **Slides:** *Applied Machine Learning for CEE*, Lecture 23 (`lectures/cee_l23_physics_informed`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §16.3.2–16.3.4 (Jacobians, the chain rule, and backpropagation: the same machinery differentiates a network with respect to its inputs), §16.4.4 (gradient scaling, and why Newton's method is out of reach for large networks), §10.2.3 and §10.2.6 (residuals, and checking whether a regression can be trusted), and §10.2.5 (transforming variables: the series fit works on log c_v). Physics-informed networks themselves: Raissi et al. (2019), Lu et al. (2021), Krishnapriyan et al. (2021).

    **Data:** none downloaded: the site's values are assumed and the readings simulated from Terzaghi's solution; the river uses Ogata and Banks's (1961) exact solution.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`, against the
    slides' `numbers.json` (read from `lectures/build/fig/cee_l23_physics_informed/`). Two
    settings, chosen by the environment variable `L23_SETTING` (or by editing `SETTING` below),
    never by the machine:

    - **`reduced`** (the default): one network seed, one noise draw, and one Péclet number
      (1,000): 8 networks, the same training as the slides'. It runs on a laptop CPU: 7
      minutes on 8 cores (2 workers of 4 threads) of the slides' machine (November 21, 2026),
      36 minutes on 8 cores when that machine was heavily loaded (November 19), with about 1 GB
      of memory; a machine with fewer than 8 cores runs fewer workers (one on a 2-core Colab
      instance). Only the parts that do not depend on training are checked
      against the slides exactly (Terzaghi's series, the readings, the finite-difference
      solvers, Ogata and Banks's solution, the checks of §3, the series fits); the trained
      networks' numbers are printed, and checked only for plausibility.
    - **`slides`**: the slides' 83 small networks (physics-informed neural networks, PINNs) in
      64-bit floats on the CPU, 4 threads each, in parallel worker processes, as the slides'
      figure script does: 60 minutes with 10 workers on the 56-thread machine that made the
      slides, heavily loaded, where every number but the timings must equal the slides' run (§7;
      1,008 numbers on November 19, 2026).

    Timings are never reproducible, in either setting: only their order of magnitude is checked.

    **Your tasks** (from the slides): give a paving date from six months of readings; build a
    PINN for Terzaghi's equation (a residual by automatic differentiation, a no-flow base, a start
    condition, a drained top built into the network) and check its pieces against finite
    differences and the exact solutions; compare it with finite differences on the same grid,
    and, for an unknown $c_v$, with a one-parameter fit of Terzaghi's series; break it (no start
    condition, residual points only in the first six months, the settlement read two ways, a
    spill in a river at Péclet numbers 1 to 10,000, and residual-based refinement against as
    many random points); then decide
    who sets the paving date.
    """)
    return


@app.cell
def _():
    import itertools
    import json
    import os
    import random
    import time
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from joblib.externals.loky import get_reusable_executor
    from matplotlib.colors import LogNorm
    from matplotlib.patheffects import withStroke
    from scipy.linalg import solve_banded
    from scipy.optimize import brentq, least_squares
    from scipy.special import erfc, erfcx
    from torch import nn

    T_START = time.perf_counter()
    SEED = 23
    os.environ["PYTHONHASHSEED"] = (
        "0"  # for the worker processes; this one needs it before Python starts
    )

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # The slides' constants (lectures/figures/cee_l23_physics_informed.py)
    # The site (assumed values)
    H = 4.0  # m: clay thickness, drained at the top, impermeable base
    CV = 4.0  # m2/year
    FILL_M, FILL_WEIGHT = 2.5, 20.0  # m of fill, kN/m3
    U0 = 50.0  # kPa: excess pore pressure under 2.5 m of fill at 20 kN/m3
    SIGMA = 1.0  # kPa: piezometer noise
    DEPTHS = (1.0, 2.0, 3.0)  # m
    MONTHS = (1, 2, 3, 4, 5, 6)  # readings
    T_MAX = 5.0  # years: the PINN's time window
    TARGET = 0.9  # pave at 90% average consolidation
    N_DRAWS = 10  # noise draws for the inverse problem
    N_SEEDS = 5  # network seeds for the forward and transport studies
    N_NO_IC = 3  # seeds without the start condition
    # The network and its training (fixed before any comparison)
    WIDTH, LAYERS = 32, 4  # tanh MLP, 32 wide, 4 hidden layers
    ADAM_STEPS, ADAM_LR = 3000, 1e-3
    LBFGS_ITERS = 1000
    N_RES, N_IC, N_BC = 2000, 200, 200  # residual, initial-condition, boundary points
    THREADS = 4  # torch threads per worker process
    WORKERS = 10
    # The river (dimensionless: X = x / L, tau = v t / L)
    TAU = 0.75
    PECLETS = (1, 10, 100, 1000, 10000)
    RAR_PECLETS = (1000, 10000)
    RAR_POOL, RAR_ADD = 20000, 1000  # residual-based refinement: add the worst of a fresh pool
    INTAKE = 0.5  # X of the intake
    SHOW_PE = 1000  # the Peclet number drawn in the profiles figure
    SHOW_TAU = 0.1  # ... at this time: early, while the front is sharp
    FD_TERMS = 200  # series terms

    def setup_worker() -> None:
        """Each process: float64, fixed threads, deterministic kernels (as the slides' script)."""
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True)

    seed_all(SEED)
    setup_worker()  # this process too: the checks of §3 run here, as in the script

    # The setting: "reduced" (8 networks, the default) or "slides" (83 networks, every number
    # checked against the slides' run). Never chosen by the machine: only the worker count is
    SETTING = os.environ.get("L23_SETTING", "reduced").strip().lower()
    if SETTING not in {"slides", "reduced"}:
        raise ValueError(f"L23_SETTING must be 'slides' or 'reduced', not {SETTING!r}")
    FULL = SETTING == "slides"
    # Worker processes: as many 4-thread jobs as the CPUs this process may use fit (a laptop's
    # 4 to 8 cores: 1 or 2; a free Colab instance's 2: 1), at most the script's 10
    _cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    POOL = min(WORKERS, max(1, (_cpus or 1) // THREADS))
    FWD_SEEDS = N_SEEDS if FULL else 1  # forward networks, seeds 23, 24, ...
    NO_IC_SEEDS = N_NO_IC if FULL else 1
    DRAWS = N_DRAWS if FULL else 1  # noise draws trained (the series fit always uses all 10)
    RIVER_SEEDS = N_SEEDS if FULL else 1
    PINN_PECLETS = PECLETS if FULL else (SHOW_PE,)
    RAR_RUN = RAR_PECLETS if FULL else (SHOW_PE,)
    FD_PECLETS = PECLETS if FULL else tuple(p for p in PECLETS if p <= SHOW_PE)

    # Tolerances. Terzaghi's series, the readings (seeded noise), the finite-difference solvers,
    # Ogata-Banks, and the series fits are exact up to the order of floating-point sums.
    EXACT = 1e-9

    def near(x: float, ref: float, tol: float = EXACT) -> bool:
        """Whether ``x`` equals ``ref`` to ``tol``: absolute below 1, relative above.

        Args:
            x: The computed value.
            ref: The slides' run's value (numbers.json, full precision).
            tol: The tolerance.

        Returns:
            True if they agree.
        """
        return abs(x - ref) <= tol * max(1.0, abs(ref))

    def find_root() -> Path:
        """The repository's root, from the repo root, ``notebooks/``, or ``ipynb/``.

        Returns:
            The first of the working directory and its parent that holds ``lectures/``.

        Raises:
            FileNotFoundError: If neither does (run from inside the repository).
        """
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / "lectures").is_dir():
                return base
        raise FileNotFoundError(
            "lectures/ not found: run from the repository, notebooks/, or ipynb/"
        )

    ROOT = find_root()
    NUMBERS = ROOT / "lectures" / "build" / "fig" / "cee_l23_physics_informed" / "numbers.json"
    # The slides' run, every value at full precision: the reference for every check (lesson
    # #32: copied programmatically, never typed). Build it with `make` in lectures/ if missing.
    REF = json.loads(NUMBERS.read_text(encoding="utf-8")) if NUMBERS.exists() else None
    if REF is None:
        print(f"{NUMBERS} not found: the comparisons with the slides are skipped")

    def ref(path: str) -> object:
        """The slides' value at ``path`` in numbers.json, e.g. "consolidation/site/t90".

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
            True for "seconds" keys (a ratio of timings is named "seconds_ratio_...") and
            "minutes".
        """
        return "seconds" in str(key) or str(key) == "minutes"

    def close(a: object, b: object, tol: float = EXACT) -> bool:
        """Whether two JSON-like values agree: numbers to ``tol``, timings ("seconds") skipped.

        Args:
            a: The computed value: a number, None, a string, or dicts and lists of them.
            b: The reference, the same structure.
            tol: Absolute below 1, relative above.

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
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

    def same(x: object, path: str, tol: float = EXACT) -> bool:
        """Whether ``x`` equals the slides' value at ``path`` (see ``close``).

        Args:
            x: The computed value.
            path: Its key in numbers.json (see ``ref``).
            tol: Absolute below 1, relative above.

        Returns:
            True if they agree, or if numbers.json is missing (nothing to compare).
        """
        return REF is None or close(x, ref(path), tol)

    def shown(x: float, text: str) -> bool:
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("3.9", "6e-08").

        The trained networks' numbers are checked this way, to half a unit of the slide's last
        digit: on the machine that made the slides they reproduce exactly (float64, 4 threads,
        deterministic kernels); on another CPU training may take another path.

        Args:
            x: The unrounded value.
            text: The slide's digits, fixed-point or in e-notation, without thousands commas.

        Returns:
            True if ``x`` rounds to ``text``.
        """
        if "e" in text:
            mantissa = text.split("e", maxsplit=1)[0]
            digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
            return f"{x:.{digits}e}" == text
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    MARKERS = ["o", "s", "^", "D"]
    INK, MUTED, NAVY = "#1A1A1A", "#595959", "#1B3A5C"
    HALO = [withStroke(linewidth=3, foreground="white")]
    FIG = (9.0, 3.6)
    print(f"setting {SETTING!r}: {POOL} workers x {THREADS} threads; root {ROOT}")
    return (
        ADAM_LR,
        ADAM_STEPS,
        CV,
        Callable,
        DEPTHS,
        DRAWS,
        FD_PECLETS,
        FD_TERMS,
        FIG,
        FILL_M,
        FILL_WEIGHT,
        FULL,
        FWD_SEEDS,
        H,
        HALO,
        Iterator,
        INK,
        INTAKE,
        LAYERS,
        LBFGS_ITERS,
        LogNorm,
        MARKERS,
        MONTHS,
        MUTED,
        NAVY,
        NO_IC_SEEDS,
        N_BC,
        N_DRAWS,
        N_IC,
        N_NO_IC,
        N_RES,
        N_SEEDS,
        PALETTE,
        PECLETS,
        PINN_PECLETS,
        POOL,
        RAR_ADD,
        RAR_POOL,
        RAR_RUN,
        REF,
        RIVER_SEEDS,
        SEED,
        SHOW_PE,
        SHOW_TAU,
        SIGMA,
        TARGET,
        TAU,
        THREADS,
        T_MAX,
        T_START,
        U0,
        WIDTH,
        brentq,
        erfc,
        erfcx,
        get_reusable_executor,
        itertools,
        least_squares,
        near,
        nn,
        np,
        plt,
        same,
        setup_worker,
        shown,
        solve_banded,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 5–22

    On your own, no notes, two minutes:

    1. Lecture 5: least squares fits a line's slope. Could it fit a soil's coefficient instead?
    2. Lecture 14: backpropagation differentiates the loss with respect to the weights. What else
       can it differentiate?
    3. Lecture 20: a 1-D filter with weights [1, −2, 1] / h² slides along a signal. What does it
       compute?

    **Today:** a network trained to obey a physical law, not just to fit data, and when it pays.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. Yes: least squares fits a model's parameters through its predictions, not only a
           line's slope. Today it fits a clay's coefficient of consolidation $c_v$ through
           Terzaghi's solution, one unknown (§4).
        2. Its inputs. The same automatic differentiation gives $\partial u/\partial t$ and
           $\partial^2 u/\partial z^2$ of a network $u(z, t)$, which is what the PDE's residual
           needs (checked below on $\sin x$, and in §3 on the network).
        3. The second derivative, by finite differences (checked below on a cubic, where it is
           exact up to round-off): the classical way to solve today's equation, and today's
           baseline.
        """)
        }
    )
    return


@app.cell
def _(np, torch):
    # Q3: [1, -2, 1] / h^2 is the second derivative. Its truncation error is h^2/12 times the
    # fourth derivative, zero for a cubic, so only round-off is left
    _h = 0.01
    _z = np.arange(0.0, 1.0 + _h / 2, _h)
    _f = _z**3 - 2 * _z**2
    _second = (_f[2:] - 2 * _f[1:-1] + _f[:-2]) / _h**2
    _err_fd = float(np.abs(_second - (6 * _z[1:-1] - 4)).max())
    # Q2: the machinery of backpropagation differentiates with respect to inputs too
    _x = torch.linspace(0.1, 1.0, 10, requires_grad=True)
    (_dy,) = torch.autograd.grad(torch.sin(_x).sum(), _x)
    _err_ad = float((_dy - torch.cos(_x.detach())).abs().max())
    print(
        f"[1, -2, 1]/h^2 on z^3 - 2z^2: {_err_fd:.1e}; autograd of sin against cos: {_err_ad:.1e}"
    )
    assert _err_fd < 1e-9 and _err_ad < 1e-15
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question: when can the road be paved?

    An embankment's fill squeezes water out of the soft clay below it; the clay settles as the
    water drains out through its top. Pave too early and the road keeps settling and cracks.
    **Decide:** a paving date, set when the clay has reached 90% of its consolidation settlement,
    from three piezometers read monthly for the first six months.

    The physics is Terzaghi's (1943): the excess pore pressure $u$ diffuses out through the
    drained top, $\partial u/\partial t = c_v\,\partial^2 u/\partial z^2$. The site's values are
    assumed and its readings simulated from Terzaghi's exact solution with 1-kPa noise, so every
    method can be scored against the truth. The exact solution is a series (derived in §3):
    """)
    return


@app.cell
def _(
    CV,
    DEPTHS,
    FD_TERMS,
    H,
    MONTHS,
    SEED,
    SIGMA,
    U0,
    brentq,
    least_squares,
    np,
):
    def series_u(zeta: np.ndarray, tv: np.ndarray, terms: int = FD_TERMS) -> np.ndarray:
        """Terzaghi's series: u / u0 at depth zeta = z / H and time factor T = c_v t / H^2."""
        m = np.pi * (2 * np.arange(terms) + 1) / 2  # (K,)
        zz, tt = np.broadcast_arrays(np.asarray(zeta, float), np.asarray(tv, float))
        terms_ = (2 / m) * np.sin(m * zz[..., None]) * np.exp(-(m**2) * tt[..., None])  # (..., K)
        return terms_.sum(-1)

    def series_degree(tv: np.ndarray, terms: int = FD_TERMS) -> np.ndarray:
        """Average degree of consolidation U(T) = 1 - mean over depth of u / u0."""
        m = np.pi * (2 * np.arange(terms) + 1) / 2
        return 1 - (2 / m**2 * np.exp(-(m**2) * np.asarray(tv, float)[..., None])).sum(-1)

    def time_factor_at(degree: float) -> float:
        """The time factor T at which the average degree of consolidation reaches ``degree``."""
        return float(brentq(lambda tv: series_degree(tv) - degree, 1e-6, 5.0, xtol=1e-14))

    def readings(draw: int) -> dict[str, np.ndarray]:
        """The piezometer readings of one noise draw (draw 0 is the site's)."""
        rng = np.random.default_rng([SEED, draw])
        zz, tt = np.meshgrid(np.asarray(DEPTHS), np.asarray(MONTHS) / 12)  # (months, depths)
        u = U0 * series_u(zz / H, CV * tt / H**2) + rng.normal(0, SIGMA, zz.shape)
        return {"z": zz.ravel(), "t": tt.ravel(), "u": u.ravel()}

    def series_fit(data: dict[str, np.ndarray]) -> float:
        """c_v by least squares on Terzaghi's series (one parameter, fitted on log c_v)."""

        def resid(p: np.ndarray) -> np.ndarray:
            return U0 * series_u(data["z"] / H, np.exp(p[0]) * data["t"] / H**2) - data["u"]

        return float(np.exp(least_squares(resid, [0.0], xtol=1e-14, ftol=1e-14).x[0]))

    return (
        readings,
        series_degree,
        series_fit,
        series_u,
        time_factor_at,
    )


@app.cell
def _(
    CV,
    DEPTHS,
    FILL_M,
    FILL_WEIGHT,
    H,
    MONTHS,
    SIGMA,
    TARGET,
    U0,
    mo,
    np,
    readings,
    same,
    series_degree,
    series_u,
    shown,
    time_factor_at,
):
    T90 = time_factor_at(TARGET)
    t90_true = T90 * H**2 / CV
    site = readings(0)
    u_3m_at_t90 = float(U0 * series_u(DEPTHS[2] / H, T90))
    _six = float(series_degree(CV * 0.5 / H**2))
    _last = site["u"].reshape(len(MONTHS), len(DEPTHS))[-1]  # month 6 at 1, 2, 3 m
    print(f"t90 = {t90_true:.4f} years (T90 = {T90:.4f}); 3-m piezometer at t90: "
          f"{u_3m_at_t90:.2f} kPa; average consolidation after 6 months {_six:.3f}")  # fmt: skip
    print("month-6 readings at 1, 2, 3 m:", np.round(_last, 2))
    assert FILL_M * FILL_WEIGHT == U0  # 2.5 m of fill at 20 kN/m3 -> 50 kPa
    # The site's table: 4 m, 4 m2/year, 1, 2, 3 m read monthly for 6 months, 1 kPa, 90%
    assert (H, CV, DEPTHS, MONTHS, SIGMA, TARGET) == (
        4.0,
        4.0,
        (1.0, 2.0, 3.0),
        (1, 2, 3, 4, 5, 6),
        1.0,
        0.9,
    )
    # Against the slides' run (numbers.json); the readings are draw 0 of
    # np.random.default_rng([23, draw]), as in the slides' script
    assert same(t90_true, "consolidation/site/t90") and same(T90, "consolidation/site/T90")
    assert same(u_3m_at_t90, "consolidation/site/u_3m_at_t90")
    assert same(_six, "consolidation/site/degree_6_months")
    assert site["u"].shape == (18,) and same(site["u"], "consolidation/site/readings/u")
    assert same(site["z"], "consolidation/site/readings/z")
    assert same(site["t"], "consolidation/site/readings/t")
    # Terzaghi's textbook time factor for 90%: T90 = 0.848
    assert abs(T90 - 0.848) < 1e-3
    # The slides: 3.39 years; 7.3 kPa at 3 m; month 6 reads 19, 33, 42 kPa; 18 readings
    assert shown(t90_true, "3.39") and shown(u_3m_at_t90, "7.3")
    assert [f"{_u:.0f}" for _u in _last] == ["19", "33", "42"]
    mo.md(f"""
    | The site (assumed values; readings simulated, so the truth is known) | Value |
    |:----------------------------------------|-----------------:|
    | Soft clay, drained at the top, rock below | {H:.0f} m |
    | Coefficient of consolidation, $c_v$ (to be learned) | {CV:.0f} m²/year |
    | Excess pore pressure under {FILL_M} m of fill ({FILL_WEIGHT:.0f} kN/m³) | {U0:.0f} kPa |
    | Piezometers, read monthly for {len(MONTHS)} months | {", ".join(f"{_z:.0f}" for _z in DEPTHS)} m deep |
    | Reading noise (one standard deviation) | {SIGMA:.0f} kPa |
    | Pave when the clay has consolidated | {100 * TARGET:.0f}% |

    The truth, from the series: 90% consolidation at **{t90_true:.2f} years** (time factor
    $T_{{90}}$ = {T90:.3f}). Average consolidation, the share of the final settlement, is not any
    one piezometer: at the true t90 the 3-m piezometer still reads {u_3m_at_t90:.1f} kPa, so that
    depth is {100 * (1 - u_3m_at_t90 / U0):.1f}% consolidated while the layer is 90%.
    """)
    return T90, site, t90_true


@app.cell
def _(DEPTHS, FIG, MARKERS, MONTHS, MUTED, PALETTE, U0, np, plt, site):
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _months = np.asarray(MONTHS)
    for _i, _z in enumerate(DEPTHS):
        _u = site["u"].reshape(len(MONTHS), len(DEPTHS))[:, _i]
        _ax.plot(np.r_[0, _months], np.r_[U0, _u], marker=MARKERS[_i], ls="-",
                 color=PALETTE[_i], label=f"{_z:.0f} m deep")  # fmt: skip
    _ax.axvspan(6, 48, color="#f2f2f2", zorder=0)
    _ax.text(27, 45, "?", fontsize=40, color=MUTED, ha="center", va="center")
    _ax.set(xlim=(0, 48), ylim=(0, 55), xlabel="months since the fill was placed",
            ylabel="excess pore pressure, kPa", xticks=range(0, 49, 6))  # fmt: skip
    _ax.legend(loc="lower left", frameon=False)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: when is the clay 90% done?

    Six months of readings, from 50 kPa down to 19, 33, and 42 kPa at 1, 2, and 3 m. Write a
    paving date, in months. Two minutes, alone. How did you get it: a ruler along the curves, a
    curve through the points, Terzaghi's chart from the geotechnics course? The answer is above;
    §4 scores three methods on it.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(r"""
        The readings fall fastest near the drained top. A straight line through the 1-m
        readings reaches 0 within a year; the 3-m readings have barely started. Terzaghi's
        chart turns a time factor into a degree of consolidation, but it needs $c_v$, and $c_v$
        is what the readings have to tell us.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: a network that must obey the PDE

    Three frames. **Frame 1:** the truth, Terzaghi's excess pore pressure over depth and time,
    with the 18 readings, all in the first six months. What does a network fitted to the stars
    alone know about year three? **Frame 2:** a physics-informed neural network (Raissi et al.,
    2019) is a network $u(z, t)$ trained on a loss read at these points: the PDE's residual at
    2,000 random points (grey), the start condition $u$ = 50 kPa at $t = 0$ (orange), no flow
    through the rock at the base (green), and the readings when we have them (stars). The
    drained top needs no points: the network's output is multiplied by $z$, so $u = 0$ there
    exactly. The residual points need no data: they cost only computing. **Frame 3** (after the
    networks are trained, below): the trained network's error.
    """)
    return


@app.cell
def _(CV, H, N_BC, N_IC, N_RES, T_MAX, U0, np, series_u):
    def field_grid() -> tuple[np.ndarray, np.ndarray]:
        """The evaluation grid: depths (81,) in m, times (500,) in years, t > 0."""
        return np.linspace(0, H, 81), np.linspace(0, T_MAX, 501)[1:]

    def exact_grid() -> np.ndarray:
        """Terzaghi's u (kPa) on ``field_grid``: (times, depths)."""
        zs, ts = field_grid()
        zz, tt = np.meshgrid(zs / H, CV * ts / H**2)
        return U0 * series_u(zz, tt)

    def sample_points(seed: int, window: float = T_MAX) -> dict[str, np.ndarray]:
        """The layer's training points on the unit square (zeta = z / H, s = t / T_MAX)."""
        rng = np.random.default_rng(seed)
        return {
            "z_res": rng.uniform(0, 1, N_RES),
            "s_res": rng.uniform(0, window / T_MAX, N_RES),
            "z_ic": rng.uniform(0, 1, N_IC),
            "s_base": rng.uniform(0, 1, N_BC),
        }

    return exact_grid, field_grid, sample_points


@app.cell
def _(
    H,
    HALO,
    INK,
    MUTED,
    NAVY,
    N_BC,
    N_IC,
    N_RES,
    PALETTE,
    SEED,
    T_MAX,
    U0,
    exact_grid,
    field_grid,
    np,
    plt,
    sample_points,
    site,
):
    _zs, _ts = field_grid()
    _exact = exact_grid()
    assert _exact.shape == (len(_ts), len(_zs)) == (500, 81)
    # "The pressure falls first near the drained top and last at the base": at every time it
    # grows with depth
    assert np.all(np.diff(_exact, axis=1) >= -1e-9)
    _p = sample_points(SEED)  # the points of the seed-23 forward network of frame 3
    assert len(_p["z_res"]) == N_RES == 2000 and N_IC == N_BC == 200 and len(site["t"]) == 18
    assert site["t"].max() == 0.5  # all readings within the first six months
    _fig, (_a, _b) = plt.subplots(2, 1, figsize=(9, 7.4), layout="constrained", sharex=True)
    _mesh = _a.pcolormesh(_ts, _zs, _exact.T, cmap="Blues", vmin=0, vmax=U0, shading="auto",
                          rasterized=True)  # fmt: skip
    _a.scatter(site["t"], site["z"], marker="*", s=70, color=PALETTE[1], zorder=3)
    _fig.colorbar(_mesh, ax=_a, label="excess pore pressure, kPa")
    _a.set_title("Frame 1: the exact field and the 18 readings", fontsize=11)
    _b.scatter(T_MAX * _p["s_res"], H * _p["z_res"], s=2, color=MUTED, alpha=0.5, linewidths=0,
               rasterized=True)  # fmt: skip
    _b.scatter(np.zeros(N_IC), H * _p["z_ic"], s=8, color=PALETTE[1], clip_on=False, zorder=3)
    _b.scatter(T_MAX * _p["s_base"], np.full(N_BC, H), s=8, color=PALETTE[2], clip_on=False,
               zorder=3)  # fmt: skip
    _b.scatter(site["t"], site["z"], marker="*", s=70, color=NAVY, zorder=4)
    _b.axhline(0, color=PALETTE[0], lw=4, clip_on=False)
    for _x, _y, _text in [
        (2.5, 1.9, "PDE residual at random points"),
        (0.12, 3.5, "u = 50 kPa at t = 0"),
        (2.5, 3.75, "no flow through the base"),
        (2.5, 0.25, "drained top: u = 0, built into the network"),
        (0.6, 1.0, "readings"),
    ]:
        _b.text(_x, _y, _text, fontsize=11, color=INK, path_effects=HALO,
                ha="left" if _x < 1 else "center", va="center")  # fmt: skip
    _b.set_title("Frame 2: where each term of the loss is read", fontsize=11)
    _fig.colorbar(_mesh, ax=_b).ax.set_visible(False)  # keep the two frames the same width
    for _ax in (_a, _b):
        _ax.set(xlim=(0, T_MAX), ylim=(H, 0), ylabel="depth, m")
    _b.set_xlabel("years")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The network

    A tanh MLP of two inputs, depth and time scaled to the unit square ($\zeta = z/H$,
    $s = t/T$ with $T$ = 5 years) and then to $[-1, 1]$: 4 hidden layers of 32. With the
    **hard top**, its output is multiplied by $\zeta$: $u = \zeta\,N(\zeta, s)$ is 0 at the top
    for any weights. `d` differentiates a pointwise output with respect to an input and keeps the
    graph, so `d(d(u, z), z)` is $\partial^2 u/\partial \zeta^2$.
    """)
    return


@app.cell
def _(LAYERS, WIDTH, nn, torch):
    class Net(nn.Module):
        """A tanh MLP of (zeta, s) on the unit square; ``hard_top`` builds in u(0, s) = 0."""

        def __init__(self, hard_top: bool) -> None:
            """Four tanh layers of ``WIDTH``; ``hard_top`` multiplies the output by zeta."""
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(2, WIDTH), nn.Tanh()]
            for _ in range(LAYERS - 1):
                layers += [nn.Linear(WIDTH, WIDTH), nn.Tanh()]
            layers.append(nn.Linear(WIDTH, 1))
            self.mlp = nn.Sequential(*layers)
            self.hard_top = hard_top

        def forward(self, zeta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            """(N,), (N,) -> (N,)."""
            x = torch.stack([2 * zeta - 1, 2 * s - 1], -1)  # (N, 2), inputs scaled to [-1, 1]
            out = self.mlp(x).squeeze(-1)  # (N,)
            return zeta * out if self.hard_top else out

    def d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """dy/dx for pointwise y(x), keeping the graph for higher derivatives."""
        return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

    return Net, d


@app.cell
def _(ADAM_LR, ADAM_STEPS, LAYERS, LBFGS_ITERS, Net, SEED, WIDTH, np, torch):
    # The weights the slides quote, and the hard top: u(0, s) = 0 exactly, for any weights
    n_weights = sum(_p.numel() for _p in Net(hard_top=True).parameters())
    _rng = np.random.default_rng(SEED)
    _worst_top = 0.0
    for _trial in range(5):
        torch.manual_seed(SEED + 1000 + _trial)  # five random networks
        _net = Net(hard_top=True)
        with torch.no_grad():
            _s = torch.tensor(_rng.uniform(0, 1, 1000))
            _worst_top = max(_worst_top, float(_net(torch.zeros(1000), _s).abs().max()))
    print(
        f"{n_weights:,} weights; |u(0, s)| over 5 random networks, 1,000 times each: {_worst_top}"
    )
    assert n_weights == 3297 and _worst_top == 0.0
    # The loss slide's notes: 4 tanh layers of 32; Adam 3,000 steps, then up to 1,000 L-BFGS
    assert (LAYERS, WIDTH, ADAM_STEPS, ADAM_LR, LBFGS_ITERS) == (4, 32, 3000, 1e-3, 1000)
    return (n_weights,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The loss in code, and its training

    `fit_consolidation` is the slides' PINN, line for line (§3 derives each term): the residual
    $\partial u/\partial s - k\,\partial^2 u/\partial\zeta^2$ at the residual points, the slope at
    the base, the start condition (weighted by `w_ic`), and the readings (scaled by $u_0$). With
    `learn_k`, $\log k$ is one more parameter. `physics=False` fits the readings and the start
    alone: the same start term at the same 200 depths, but no residual, no base term, and no
    built-in top. Training is Adam (3,000 steps at 10⁻³),
    then L-BFGS (up to 1,000 iterations, strong Wolfe line search), in 64-bit floats.
    """)
    return


@app.cell
def _(
    ADAM_LR,
    ADAM_STEPS,
    CV,
    Callable,
    H,
    LBFGS_ITERS,
    Net,
    T_MAX,
    U0,
    d,
    nn,
    np,
    sample_points,
    time,
    torch,
):
    def k_true() -> float:
        """The scaled diffusivity c_v T_MAX / H^2 of u_s = k u_zz on the unit square."""
        return CV * T_MAX / H**2

    def train(
        params: list[torch.Tensor],
        loss_fn: Callable[[], torch.Tensor],
        after_adam: Callable[[], None] | None = None,
    ) -> float:
        """Adam, then L-BFGS; returns the final loss."""
        opt = torch.optim.Adam(params, lr=ADAM_LR)
        for _ in range(ADAM_STEPS):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()
        if after_adam is not None:
            after_adam()
        lb = torch.optim.LBFGS(
            params,
            max_iter=LBFGS_ITERS,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            history_size=50,
        )

        def closure() -> torch.Tensor:
            lb.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            return loss

        lb.step(closure)
        return float(loss_fn().detach())

    def fit_consolidation(
        seed: int,
        *,
        physics: bool = True,
        w_ic: float = 1.0,
        data: dict[str, np.ndarray] | None = None,
        learn_k: bool = False,
        window: float = T_MAX,
    ) -> tuple[Net, float, float, float]:
        """Train a network for the clay layer.

        Args:
            seed: Network and point seed.
            physics: False fits the readings and the start alone (no residual, no boundary
                terms, no built-in top): the series fit and the PINN both know the start (the
                fill's load), so the network fitted to the readings gets the PINN's start term.
            w_ic: Weight of the initial-condition term (with or without physics).
            data: Readings to fit (z in m, t in years, u in kPa), or None.
            learn_k: Learn the diffusivity (the inverse problem); else it is the true value.
            window: Residual points cover 0 <= t <= window (years).

        Returns:
            The network, the diffusivity k it used, its final loss, and the seconds taken.
        """
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        net = Net(hard_top=physics)
        log_k = nn.Parameter(torch.tensor(0.0 if learn_k else np.log(k_true())))
        p = sample_points(seed, window)
        zr = torch.tensor(p["z_res"], requires_grad=True)
        sr = torch.tensor(p["s_res"], requires_grad=True)
        z0, sb = torch.tensor(p["z_ic"]), torch.tensor(p["s_base"])
        zd = sd = ud = None
        if data is not None:
            zd, sd, ud = (
                torch.tensor(a) for a in (data["z"] / H, data["t"] / T_MAX, data["u"] / U0)
            )

        def loss_fn() -> torch.Tensor:
            loss = torch.zeros(())
            if physics:
                u = net(zr, sr)
                loss = loss + (d(u, sr) - log_k.exp() * d(d(u, zr), zr)).pow(2).mean()
                zb = torch.ones_like(sb, requires_grad=True)
                loss = loss + d(net(zb, sb), zb).pow(2).mean()  # no flow through the base
            loss = loss + w_ic * (net(z0, torch.zeros_like(z0)) - 1).pow(2).mean()
            if zd is not None:
                loss = loss + (net(zd, sd) - ud).pow(2).mean()
            return loss

        params = [*net.parameters(), *([log_k] if learn_k else [])]
        loss = train(params, loss_fn)
        return net, float(log_k.detach().exp()), loss, time.perf_counter() - t0

    return fit_consolidation, k_true, train


@app.cell
def _(H, T_MAX, U0, brentq, d, exact_grid, field_grid, np, torch):
    def degree_field(net: torch.nn.Module, t_years: np.ndarray) -> np.ndarray:
        """U(t) = 1 - mean over depth of the network's u / u0 (trapezoid, 401 depths)."""
        zq = torch.linspace(0, 1, 401)
        out = []
        with torch.no_grad():
            for tv in np.atleast_1d(t_years):
                u = net(zq, torch.full_like(zq, tv / T_MAX))
                out.append(1 - float(torch.trapezoid(u, zq)))
        return np.array(out)

    def drained(u_fn, k: float, s1: float, s2: float) -> float:
        """Water out of the drain between s1 and s2, as a share of the layer's excess water.

        From u_s = k u_zz and no flow through the base, d/ds of the integral of u over depth is
        -k u_z(0, s): k times the integral of du/dz at the top (trapezoid, 4,001 times).
        """
        s = torch.linspace(s1, s2, 4001)
        z = torch.zeros_like(s, requires_grad=True)
        return k * float(torch.trapezoid(d(u_fn(z, s), z).detach(), s))

    def degree_drained(net: torch.nn.Module, k: float, t_years: np.ndarray) -> np.ndarray:
        """U(t) read from the water drained since t = 0."""
        return np.array([drained(net, k, 0.0, tv / T_MAX) for tv in np.atleast_1d(t_years)])

    def time_to(net: torch.nn.Module, degree: float) -> float | None:
        """When the network's own field reaches ``degree`` (years), or None within the window."""
        ts = np.linspace(0.01, T_MAX, 500)
        u = degree_field(net, ts)
        above = np.nonzero(u >= degree)[0]
        if len(above) == 0:
            return None
        i = above[0]
        if i == 0:
            return float(ts[0])
        return float(
            brentq(lambda tv: degree_field(net, np.array([tv]))[0] - degree, ts[i - 1], ts[i])
        )

    def predict(net: torch.nn.Module) -> np.ndarray:
        """The network's u (kPa) on ``field_grid``: (times, depths)."""
        zs, ts = field_grid()
        zz, tt = np.meshgrid(zs / H, ts / T_MAX)
        with torch.no_grad():
            u = net(torch.tensor(zz.ravel()), torch.tensor(tt.ravel())).numpy()
        return U0 * u.reshape(zz.shape)

    def max_location(field: np.ndarray) -> dict[str, float]:
        """Depth (m) and time (years) of a field's largest value on ``field_grid``."""
        zs, ts = field_grid()
        it, iz = np.unravel_index(int(field.argmax()), field.shape)
        return {"depth": float(zs[iz]), "years": float(ts[it])}

    def residual_stats(
        net: torch.nn.Module, k: float, rng: np.random.Generator, n: int = 20_000
    ) -> dict[str, float]:
        """Notebook only: a trained network's PDE residual at ``n`` fresh points of the square.

        Returns the RMS and the largest |residual| (where it is), and the RMS within the first
        six months (s <= 0.1) and after them.
        """
        zq = torch.tensor(rng.uniform(0, 1, n), requires_grad=True)
        sq = torch.tensor(rng.uniform(0, 1, n), requires_grad=True)
        u = net(zq, sq)
        r = (d(u, sq) - k * d(d(u, zq), zq)).detach().numpy()
        z, s = zq.detach().numpy(), sq.detach().numpy()
        i = int(np.abs(r).argmax())
        early = s <= 0.5 / T_MAX
        return {
            "rms": float(np.sqrt(np.mean(r**2))),
            "max": float(np.abs(r).max()),
            "max_at_depth": float(H * z[i]),
            "max_at_years": float(T_MAX * s[i]),
            "rms_first_6_months": float(np.sqrt(np.mean(r[early] ** 2))),
            "rms_after": float(np.sqrt(np.mean(r[~early] ** 2))),
        }

    def summarize_forward(f: dict) -> dict[str, object]:
        """Errors of one forward run against the series, in kPa."""
        zs, ts = field_grid()
        err = np.abs(f["field"] - exact_grid())  # (times, depths)
        big = err > 1.0
        it, iz = np.unravel_index(int(err.argmax()), err.shape)
        return {
            "seed": f["seed"],
            "seconds": f["seconds"],
            "loss": f["loss"],
            "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(exact_grid())),
            "max_kpa": float(err.max()),
            "max_at": {"depth": float(zs[iz]), "years": float(ts[it])},
            "over_1kpa": {
                "share": float(big.mean()),
                "deepest": float(zs[big.any(0)].max()) if big.any() else None,
                "latest": float(ts[big.any(1)].max()) if big.any() else None,
            },
            "after_3_months_max_kpa": float(err[ts >= 0.25].max()),
            "degree_field": dict(
                zip(("3 months", "1 year", "t90"), f["degree_field"].tolist(), strict=True)
            ),
            "degree_drained": dict(
                zip(("3 months", "1 year", "t90"), f["degree_drained"].tolist(), strict=True)
            ),
        }

    return (
        degree_drained,
        degree_field,
        drained,
        max_location,
        predict,
        residual_stats,
        summarize_forward,
        time_to,
    )


@app.cell
def _(
    CV,
    H,
    SEED,
    TARGET,
    T_MAX,
    degree_drained,
    degree_field,
    fit_consolidation,
    max_location,
    np,
    predict,
    readings,
    residual_stats,
    setup_worker,
    time_factor_at,
    time_to,
):
    def forward_job(seed: int, w_ic: float) -> dict:
        """The forward problem: c_v known, no readings."""
        setup_worker()
        net, k, loss, secs = fit_consolidation(seed, w_ic=w_ic)
        t90 = time_factor_at(TARGET) * H**2 / CV
        probe = np.array([0.25, 1.0, t90])
        out = {
            "seed": seed,
            "loss": loss,
            "seconds": secs,
            "field": predict(net),
            "probe_t": probe,
            "degree_field": degree_field(net, probe),
            "degree_drained": degree_drained(net, k, probe),
        }
        # Notebook only, after every number the slides use: the residual at fresh points, from a
        # generator of its own, so nothing above changes
        out["residual"] = residual_stats(net, k, np.random.default_rng([SEED, 1, seed]))
        return out

    def inverse_job(draw: int, physics: bool, window: float) -> dict:
        """The inverse problem on one noise draw: learn c_v (physics) or fit the readings alone.

        Without physics the network gets the readings and the known start (the PINN's start
        term, at the same 200 depths).
        """
        setup_worker()
        net, k, loss, secs = fit_consolidation(
            SEED + draw, physics=physics, data=readings(draw), learn_k=physics, window=window
        )
        curve_t = np.linspace(0, T_MAX, 251)
        field_kpa = predict(net)  # kPa already: no second U0
        out = {
            "draw": draw,
            "cv": k * H**2 / T_MAX if physics else None,
            "loss": loss,
            "seconds": secs,
            "t90_field": time_to(net, TARGET),
            "curve_t": curve_t,
            "curve": degree_field(net, curve_t),
            "min_u_kpa": float(field_kpa.min()),
            "max_u_kpa": float(field_kpa.max()),
            "max_at": max_location(field_kpa),
        }
        # Notebook only: the field itself (kPa), and the residual with the learned k
        out["field_kpa"] = field_kpa
        out["residual"] = (
            residual_stats(net, k, np.random.default_rng([SEED, 2, draw])) if physics else None
        )
        return out

    return forward_job, inverse_job


@app.cell
def _(mo):
    mo.md(r"""
    ### Every consolidation network, in parallel

    The jobs of the slides' figure script, with its seeds: the forward problem (seeds 23–27),
    the forward problem without the start condition (seeds 23–25), and on noise draws 0–9 the
    inverse problem, the readings alone, and the residual only in the first six months. Each job
    runs in its own worker process with 4 threads, so it reproduces whatever else runs beside it.
    They are submitted now and collected where they are used; frame 3 waits only for the first.
    """)
    return


@app.cell
def _(
    DRAWS,
    FWD_SEEDS,
    NO_IC_SEEDS,
    POOL,
    SEED,
    THREADS,
    T_MAX,
    forward_job,
    get_reusable_executor,
    inverse_job,
):
    # loky's executor pickles the cell's functions by value (cloudpickle); OMP_NUM_THREADS=4 in
    # each worker, as the figure script sets before spawning its own. Idle workers stay 5
    # minutes, so they are still there when §5 submits the river's networks
    executor = get_reusable_executor(
        max_workers=POOL, timeout=300, env={"OMP_NUM_THREADS": str(THREADS)}
    )
    consolidation_jobs = {
        "forward": [executor.submit(forward_job, SEED + s, 1.0) for s in range(FWD_SEEDS)],
        "no_ic": [executor.submit(forward_job, SEED + s, 0.0) for s in range(NO_IC_SEEDS)],
        "pinn": [executor.submit(inverse_job, dr, True, T_MAX) for dr in range(DRAWS)],
        "readings_only": [executor.submit(inverse_job, dr, False, T_MAX) for dr in range(DRAWS)],
        "window": [executor.submit(inverse_job, dr, True, 0.5) for dr in range(DRAWS)],
    }
    print(f"{sum(len(v) for v in consolidation_jobs.values())} consolidation networks submitted")
    return consolidation_jobs, executor


@app.cell
def _(
    FULL,
    H,
    LogNorm,
    T_MAX,
    consolidation_jobs,
    exact_grid,
    field_grid,
    mo,
    np,
    plt,
    same,
    shown,
    summarize_forward,
):
    _f = consolidation_jobs["forward"][0].result()  # seed 23
    first_forward = summarize_forward(_f)
    _zs, _ts = field_grid()
    _err = np.abs(_f["field"] - exact_grid())
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _mesh = _ax.pcolormesh(_ts, _zs, np.maximum(_err.T, 1e-3), cmap="magma_r",
                           norm=LogNorm(1e-2, 10), shading="auto", rasterized=True)  # fmt: skip
    _ax.set(xlim=(0, T_MAX), ylim=(H, 0), xlabel="years", ylabel="depth, m")
    _fig.colorbar(_mesh, ax=_ax, extend="both", label="|PINN − exact|, kPa")
    _ax.set_title("Frame 3: the trained network's error (seed 23, c_v known, no readings)",
                  fontsize=11)  # fmt: skip
    _o = first_forward["over_1kpa"]
    print(f"seed 23: relative error {100 * first_forward['rel_l2']:.2f}%, max "
          f"{first_forward['max_kpa']:.3f} kPa at {first_forward['max_at']}; over 1 kPa: "
          f"{100 * _o['share']:.2f}% of the grid, to {_o['deepest']} m and {_o['latest']} years; "
          f"after 3 months {first_forward['after_3_months_max_kpa']:.4f} kPa")  # fmt: skip
    if FULL:  # the slide's digits: 1.0%, 3.9 kPa, in the top 1.0 m, the first 0.03 years, 0.80 kPa
        # Over 1 kPa only in the top 1.0 m and the first 0.03 years (both bounds reached), at the
        # corner where the drained top meets the start; the largest error at the earliest time
        assert same(first_forward, "consolidation/forward/0")
        assert first_forward["max_at"]["years"] == _ts[0]
        assert shown(100 * first_forward["rel_l2"], "1.0") and shown(
            first_forward["max_kpa"], "3.9"
        )
        assert shown(_o["deepest"], "1.0") and shown(_o["latest"], "0.03")
        assert shown(first_forward["after_3_months_max_kpa"], "0.80")
    mo.vstack([_fig, mo.md(f"""
    The network after training on the PDE alone ($c_v$ known, no readings) reproduces the exact
    solution to a relative error of {100 * first_forward["rel_l2"]:.1f}%, except at the corner where the drained
    top ($u = 0$) meets the start ($u$ = 50 kPa): over 1 kPa only in the top {_o["deepest"]:.1f} m and the
    first {_o["latest"]:.2f} years ({100 * _o["share"]:.1f}% of the grid). There the exact solution jumps, and the water's flow
    out of the top is infinite at $t = 0$ (checked in §3); a smooth network cannot follow a jump. Its
    error peaks there at {first_forward["max_kpa"]:.1f} kPa and is at most {first_forward["after_3_months_max_kpa"]:.2f} kPa after three months (seed 23; across five seeds §4 gives the range). Which part of the field matters for a paving date years later?
    """)])  # fmt: skip
    return


@app.cell
def _(FULL, consolidation_jobs, mo):
    _r = consolidation_jobs["forward"][0].result()["residual"]
    print(f"seed 23, residual at 20,000 fresh points: {_r}")
    if FULL:  # seed 23: the residual is concentrated at the start, by the corner
        assert _r["rms_first_6_months"] > 100 * _r["rms_after"] and _r["max_at_years"] < 0.05
    mo.md(f"""
    ### Grids, graphs, sets, and now physics

    | Data | What the model is told | How | Lecture |
    |:------------|:-----------------------|:-------------------|:-------|
    | Pixels on a grid | a pattern means the same anywhere | shared 3 × 3 weights | 15, 16 |
    | Nodes and edges | the same rules at every node | messages summed | 21 |
    | Points in space | order does not matter | a max over points | 22 |
    | A field u(z, t) | a PDE holds everywhere | a penalty in the loss | today |

    Each structure gets the model that respects it. The first three build the structure into
    the architecture: it holds exactly, for any weights. A PINN mostly asks for it through the
    loss: it holds only as well as training drives the residual down. The drained top is the
    exception today: built in, $u(0, s)$ = 0 to the last bit on five random networks (above). The
    PDE is not: on 20,000 fresh points the trained network's residual has an RMS of
    {_r["rms"]:.1e} and reaches {_r["max"]:.0f} at {_r["max_at_depth"]:.2f} m and {_r["max_at_years"]:.4f} years, by the corner.
    Most of it is in the first six months (RMS {_r["rms_first_6_months"]:.1e} there, {_r["rms_after"]:.1e} after them).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive: a loss that is zero only if the physics holds

    **Assumption:** the excess pore pressure obeys Terzaghi's equation everywhere in the layer,
    with one coefficient $c_v$ (saturated clay, water flowing only vertically, small strains, the
    load applied at once).

    **Scale first.** With depth $\zeta = z/H$, time $s = t/T$ ($T$ = 5 years, the window the
    network covers), and $u$ in units of $u_0$ = 50 kPa, Terzaghi's equation on the unit square is

    $$\frac{\partial u}{\partial s} = k\,\frac{\partial^2 u}{\partial \zeta^2}, \qquad k = \frac{c_v T}{H^2},$$

    with three conditions: the top drains, $u(0, s) = 0$; no water enters the rock,
    $\partial u/\partial\zeta\,(1, s) = 0$; and the load's excess pressure at the start,
    $u(\zeta, 0) = 1$.

    **The residual.** For a network $u_\theta(\zeta, s)$,

    $$r(\zeta, s) = \frac{\partial u_\theta}{\partial s} - k\,\frac{\partial^2 u_\theta}{\partial \zeta^2}.$$

    Both derivatives come from automatic differentiation with respect to the inputs, the chain
    rule of backpropagation (Lecture 14; Forsyth §16.3) applied to $\zeta$ and $s$ instead of the
    weights. $r = 0$ everywhere *is* the PDE.

    **The loss.** Each term is the mean of a squared mismatch over its own points:

    $$L(\theta) = \frac{1}{N_r}\sum_{j} r(\zeta_j, s_j)^2 + \frac{1}{N_b}\sum_{j} \Big(\frac{\partial u_\theta}{\partial \zeta}(1, s_j)\Big)^2 + \frac{1}{N_0}\sum_{j} \big(u_\theta(\zeta_j, 0) - 1\big)^2 + \frac{1}{18}\sum_{i} \Big(u_\theta(\zeta_i, s_i) - \frac{\tilde u_i}{u_0}\Big)^2.$$

    $L = 0$ only if the network obeys the PDE at the residual points, the base and start
    conditions at theirs, and the readings $\tilde u_i$. Equal weights on the four terms;
    choosing them is a known difficulty (Wang et al., 2021).

    **Build in what you can.** The top condition needs no term: $u_\theta = \zeta\,N_\theta(\zeta, s)$
    is 0 at $\zeta = 0$ for any weights (a hard constraint). The other conditions are soft: they
    hold only as well as training drives their terms down.

    **The inverse problem.** $\log k$ becomes a trainable parameter (it starts at 0, $k = 1$)
    and the readings term is added: the same gradient steps fit $c_v$ and the field together.

    **Why tanh, not ReLU.** The residual needs $\partial^2 u/\partial\zeta^2$; a ReLU network is
    piecewise linear in its inputs, so its second derivative is zero almost everywhere (checked
    below). tanh is smooth.

    **The exact answer, to score against.** Separation of variables: the functions
    $\sin(M\zeta)$ with $M = \pi(2m+1)/2$ vanish at the top and have zero slope at the base, and
    expanding the start, $u = 1$, in them gives

    $$\frac{u}{u_0} = \sum_{m=0}^{\infty} \frac{2}{M}\,\sin(M\zeta)\,e^{-M^2 T_v}, \qquad U(T_v) = 1 - \sum_{m=0}^{\infty} \frac{2}{M^2}\,e^{-M^2 T_v},$$

    with the time factor $T_v = c_v t / H^2$ and $U$ the average degree of consolidation (200
    terms here). The top's slope is $\sum 2e^{-M^2 T_v}$, which grows without bound as
    $T_v \to 0$: the water's flow out of the top is infinite at the start (checked below).

    **Water drained, a second reading of $U$.** Integrate the PDE over depth:
    $\frac{d}{ds}\int_0^1 u\,d\zeta = k\big[\partial_\zeta u\big]_0^1 = -k\,\partial_\zeta u(0, s)$,
    because the base has no flow. So the water out of the top between $s_1$ and $s_2$,
    $k\int_{s_1}^{s_2} \partial_\zeta u(0, s)\,ds$, must equal the rise of $U$ over that time:
    an identity for the exact solution, true for a network only as far as its residual is zero.
    """)
    return


@app.cell
def _(CV, H, T90, T_MAX, k_true, np, shown, t90_true):
    # The numbers in the derivation: k = 1.25; the inverse problem starts at c_v = 3.2 m2/year;
    # T90 = 0.848, 3.39 years
    _cv_start = float(np.exp(0.0)) * H**2 / T_MAX
    assert k_true() == CV * T_MAX / H**2 == 1.25 and _cv_start == 3.2
    assert shown(T90, "0.848") and shown(t90_true, "3.39")
    # The top's slope sum(2 exp(-M^2 T)) grows like 1/sqrt(pi T) as T -> 0: infinite at the start
    _m = np.pi * (2 * np.arange(200_000) + 1) / 2
    _flux = {_tv: float((2 * np.exp(-(_m**2) * _tv)).sum()) for _tv in (1e-2, 1e-4, 1e-6)}
    _scaled = {_tv: _f * np.sqrt(np.pi * _tv) for _tv, _f in _flux.items()}
    print(f"top slope at T = 1e-2, 1e-4, 1e-6: {list(_flux.values())}; times sqrt(pi T): "
          f"{list(_scaled.values())}")  # fmt: skip
    assert all(abs(_v - 1) < 1e-3 for _v in _scaled.values())  # 1 / sqrt(pi T): unbounded
    return


@app.cell
def _(SEED, d, nn, np, torch):
    # A ReLU network's second derivative is zero almost everywhere: it is piecewise linear
    torch.manual_seed(SEED)
    _relu = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
    )
    _z = torch.tensor(np.random.default_rng(SEED).uniform(0, 1, 1000), requires_grad=True)
    _u = _relu(torch.stack([_z, torch.full_like(_z, 0.5)], -1)).squeeze(-1)
    _uz = d(_u, _z)
    _uzz = torch.autograd.grad(_uz, _z, torch.ones_like(_uz), allow_unused=True)[0]
    _uzz = torch.zeros_like(_z) if _uzz is None else _uzz.detach()
    _uz_max, _uzz_max = float(_uz.detach().abs().max()), float(_uzz.abs().max())
    print(f"ReLU network: |du/dz| up to {_uz_max:.3f}, |d2u/dz2| up to {_uzz_max}")
    assert _uz_max > 0 and _uzz_max == 0.0
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    ```python
    u_zz = grad(grad(net(z, t), z), z)    # autograd, twice
    fd   = (net(z+h,t) - 2*net(z,t) + net(z-h,t)) / h**2
    r    = residual(terzaghi_series, z, t) # same code
    U1   = drained(terzaghi_series, 3 mo, t90)  # flux, top
    cv   = series_fit(readings_without_noise)
    ```

    How far is `u_zz` from `fd`? `r` from 0? `U1` from the change in U? `cv` from 4? Two
    minutes, written. $h = 10^{-4}$ in 64-bit floats. The drained route: the water leaving through
    the top, the integral of $k\,\partial u/\partial z$ there, must equal the drop in the layer's
    average pressure. The check code below is the slides' `checks()`; it runs here, in this
    process, with the same seeds. First, the river's exact solution, which it also checks (the
    river comes back in §5):
    """)
    return


@app.cell
def _(erfc, erfcx, np):
    def ogata_banks(x: np.ndarray, tau: np.ndarray, pe: float) -> np.ndarray:
        """Concentration for a continuous release at X = 0 (Ogata & Banks, 1961), overflow-safe."""
        x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
        s = 2 * np.sqrt(np.maximum(tau, 1e-300) / pe)
        a = (x + tau) / s
        out = 0.5 * (erfc((x - tau) / s) + np.exp(pe * x - a**2) * erfcx(a))
        return np.where(tau > 0, out, (x == 0).astype(float))

    return (ogata_banks,)


@app.cell
def _(
    CV,
    DEPTHS,
    FD_TERMS,
    H,
    MONTHS,
    Net,
    SEED,
    TARGET,
    TAU,
    T_MAX,
    U0,
    d,
    drained,
    k_true,
    np,
    ogata_banks,
    series_degree,
    series_fit,
    series_u,
    setup_worker,
    time_factor_at,
    torch,
):
    def checks() -> dict[str, float]:
        """Autograd, exact residuals, the drained-water identity, and a noise-free series fit."""
        setup_worker()
        torch.manual_seed(SEED)
        net = Net(hard_top=True)
        rng = np.random.default_rng(SEED)
        # Autograd's second derivative of an untrained network against its central difference
        z = torch.tensor(rng.uniform(0.05, 0.95, 1000), requires_grad=True)
        s = torch.tensor(rng.uniform(0.05, 0.95, 1000))
        u_zz = d(d(net(z, s), z), z).detach()
        h = 1e-4
        with torch.no_grad():
            zd = z.detach()
            fd2 = (net(zd + h, s) - 2 * net(zd, s) + net(zd - h, s)) / h**2
        out = {"autograd_vs_fd": float((u_zz - fd2).abs().max())}
        # Terzaghi's series in torch: its residual by the same autograd code is round-off
        m = torch.pi * (2 * torch.arange(FD_TERMS) + 1) / 2
        zt = torch.tensor(rng.uniform(0, 1, 1000), requires_grad=True)
        tt = torch.tensor(rng.uniform(0.01, 1.25, 1000), requires_grad=True)
        u = ((2 / m) * torch.sin(m * zt[:, None]) * torch.exp(-(m**2) * tt[:, None])).sum(-1)
        out["series_residual"] = float((d(u, tt) - d(d(u, zt), zt)).detach().abs().max())
        # Ogata-Banks: the river's exact solution, residual by autograd
        for pe in (10.0, 1000.0):
            x = torch.tensor(rng.uniform(0.05, 0.95, 1000), requires_grad=True)
            ta = torch.tensor(rng.uniform(0.05, TAU, 1000), requires_grad=True)
            sd = 2 * torch.sqrt(ta / pe)
            a = (x + ta) / sd
            c = 0.5 * (torch.special.erfc((x - ta) / sd)
                       + torch.exp(pe * x - a**2) * torch.special.erfcx(a))  # fmt: skip
            cx = d(c, x)
            res = (d(c, ta) + cx - d(cx, x) / pe).detach()
            out[f"ogata_banks_residual_{int(pe)}"] = float(res.abs().max())
            ref = ogata_banks(x.detach().numpy(), ta.detach().numpy(), pe)
            out[f"ogata_banks_torch_vs_numpy_{int(pe)}"] = float(
                np.abs(c.detach().numpy() - ref).max()
            )
        # The drained-water route on the exact solution, from 3 months to t90 (the flux is
        # singular at t = 0): it must equal the change in U from the pore pressures
        k = k_true()

        def exact_u(zeta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            tf = (k * s)[:, None]  # time factor T = k s
            return ((2 / m) * torch.sin(m * zeta[:, None]) * torch.exp(-(m**2) * tf)).sum(-1)

        s1, s2 = 0.25 / T_MAX, time_factor_at(TARGET) / k
        change = float(series_degree(k * s2) - series_degree(k * s1))
        out["drained_vs_field_exact"] = abs(drained(exact_u, k, s1, s2) - change)
        # A noise-free fit recovers c_v
        zz, tt_ = np.meshgrid(np.asarray(DEPTHS), np.asarray(MONTHS) / 12)
        clean = {
            "z": zz.ravel(),
            "t": tt_.ravel(),
            "u": U0 * series_u(zz / H, CV * tt_ / H**2).ravel(),
        }
        out["series_fit_noise_free"] = abs(series_fit(clean) - CV)
        return out

    checks_out = checks()
    print(checks_out)
    return (checks_out,)


@app.cell
def _(
    CV,
    H,
    TARGET,
    U0,
    exact_grid,
    field_grid,
    itertools,
    np,
    series_u,
    time,
    time_factor_at,
):
    def fd_step(nz: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """One Crank-Nicolson step of u_T = u_ZZ as a matrix on nodes 1..nz, and the nodes 0..nz."""
        dz = 1 / nz
        n = nz  # unknowns at nodes 1..nz (node 0 is drained)
        lap = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
        lap[n - 1, n - 2] = 2  # ghost node: u[nz + 1] = u[nz - 1]
        lap /= dz**2
        a = np.eye(n) - 0.5 * dt * lap
        b = np.eye(n) + 0.5 * dt * lap
        return np.linalg.solve(a, b), np.linspace(0, 1, nz + 1)

    def fd_consolidation(nz: int, nt: int, t_factor: float) -> tuple[np.ndarray, np.ndarray]:
        """Crank-Nicolson for u_T = u_ZZ on nz cells; a ghost node mirrors the impermeable base.

        Returns:
            The nodes Z (nz + 1,) and u / u0 at T = t_factor, (nz + 1,).
        """
        step, z = fd_step(nz, t_factor / nt)
        u = np.linalg.matrix_power(step, nt) @ np.ones(nz)
        return z, np.r_[0.0, u]

    def fd_convergence() -> dict[str, object]:
        """The FD solver against the series at T = 0.05, 0.2, and T90, three grids.

        Returns:
            Per grid the largest error over the three times (kPa) and the seconds for T90; the
            observed orders; and the three times, as time factors and in years.
        """
        t90 = time_factor_at(TARGET)
        factors = (0.05, 0.2, t90)
        rows = []
        for nz in (20, 40, 80):
            err = 0.0
            for tf in factors:
                z, u = fd_consolidation(nz, 4 * nz * nz // 10, tf)  # dt proportional to dz^2
                err = max(err, float(np.abs(u - series_u(z, tf)).max()))
            t0 = time.perf_counter()
            fd_consolidation(nz, 4 * nz * nz // 10, t90)
            rows.append(
                {"cells": nz, "max_error_kpa": U0 * err, "seconds": time.perf_counter() - t0}
            )
        order = [
            np.log2(a["max_error_kpa"] / b["max_error_kpa"]) for a, b in itertools.pairwise(rows)
        ]
        return {
            "grids": rows,
            "order": order,
            "time_factors": list(factors),
            "times_years": [tf * H**2 / CV for tf in factors],
        }

    def fd_on_grid(nz: int = 80, per_output: int = 16) -> dict[str, object]:
        """Finite differences scored like the PINN: every time and depth of ``field_grid``.

        Crank-Nicolson on ``nz`` cells (the grid's 81 depths are its nodes), ``per_output`` steps
        between output times (dt / dz^2 = 1 at nz = 80).
        """
        zs, ts = field_grid()
        assert nz + 1 == len(zs), (nz, len(zs))
        t0 = time.perf_counter()
        dt_out = CV * (ts[1] - ts[0]) / H**2  # time-factor step between outputs
        _, u_first = fd_consolidation(nz, per_output, CV * ts[0] / H**2)
        step_z, _ = fd_step(nz, dt_out / per_output)
        step = np.linalg.matrix_power(step_z, per_output)
        rows, u = [u_first], u_first[1:]
        for _ in ts[1:]:
            u = step @ u
            rows.append(np.r_[0.0, u])
        secs = time.perf_counter() - t0
        err = np.abs(U0 * np.array(rows) - exact_grid())  # (times, depths)
        assert err.shape == (len(ts), len(zs))
        it, iz = np.unravel_index(int(err.argmax()), err.shape)
        return {
            "cells": nz,
            "steps": per_output * len(ts),  # time steps to the last output
            "outputs": len(ts),
            "max_kpa": float(err.max()),
            "max_at": {"depth": float(zs[iz]), "years": float(ts[it])},
            "after_3_months_max_kpa": float(err[ts >= 0.25].max()),
            "seconds": secs,
        }

    fd = fd_convergence() | {"on_grid": fd_on_grid()}
    return (fd,)


@app.cell
def _(FULL, checks_out, fd, mo, same, shown):
    _c, _g = checks_out, fd["grids"]
    assert same(_c, "checks")  # the slides' run (numbers.json), every check
    # Bounds that hold on any machine: round-off for the exact solutions and the noise-free fit;
    # about h^2 for the network's second difference; the trapezoid's error for the drained water
    assert _c["autograd_vs_fd"] < 1e-6 and _c["drained_vs_field_exact"] < 1e-6
    assert (
        max(_c["series_residual"], _c["ogata_banks_residual_10"], _c["ogata_banks_residual_1000"])
        < 1e-12
    )
    assert (
        _c["ogata_banks_torch_vs_numpy_10"] < 1e-12
        and _c["ogata_banks_torch_vs_numpy_1000"] < 1e-12
    )
    assert _c["series_fit_noise_free"] < 1e-12
    if FULL:  # round-off digits, as on the slides' machine (another CPU may round differently)
        assert shown(_c["autograd_vs_fd"], "1e-08") and shown(_c["series_residual"], "2e-15")
        assert shown(_c["ogata_banks_residual_1000"], "5e-15")
        assert shown(_c["drained_vs_field_exact"], "6e-08") and shown(
            _c["series_fit_noise_free"], "9e-16"
        )
    # Finite differences, 20 -> 40 -> 80 cells: 0.034 -> 0.009 -> 0.0021 kPa; orders 2.00, 2.00;
    # scored at 0.2, 0.8, and 3.39 years (T = 0.05, 0.2, T90)
    assert all(same(fd[_k], f"fd/{_k}") for _k in ("grids", "order", "time_factors", "times_years"))
    assert [_r["cells"] for _r in _g] == [20, 40, 80]
    _years = fd["times_years"]
    assert [f"{_y:.1f}" for _y in _years[:2]] == ["0.2", "0.8"] and shown(_years[2], "3.39")
    assert shown(_g[0]["max_error_kpa"], "0.034") and shown(_g[1]["max_error_kpa"], "0.009")
    assert shown(_g[2]["max_error_kpa"], "0.0021")
    assert shown(fd["order"][0], "2.00") and shown(fd["order"][1], "2.00")
    mo.md(f"""
    ### The code passes every check

    | Check | Value |
    |:------|------:|
    | Autograd ∂²u/∂z² against finite differences of the network | {_c["autograd_vs_fd"]:.0e} |
    | Residual of Terzaghi's exact series | {_c["series_residual"]:.0e} |
    | Residual of the river's exact solution (Péclet 10; 1,000) | {_c["ogata_banks_residual_10"]:.0e}; {_c["ogata_banks_residual_1000"]:.0e} |
    | The river's solution in torch against NumPy (Péclet 10; 1,000) | {_c["ogata_banks_torch_vs_numpy_10"]:.0e}; {_c["ogata_banks_torch_vs_numpy_1000"]:.0e} |
    | Water drained against the drop in pressure, exact solution | {_c["drained_vs_field_exact"]:.0e} |
    | $c_v$ from noise-free readings, error in m²/year | {_c["series_fit_noise_free"]:.0e} |
    | Finite differences at {_years[0]:.1f}, {_years[1]:.1f}, {_years[2]:.2f} years, 20 → 40 → 80 cells: max error, kPa | {_g[0]["max_error_kpa"]:.3f} → {_g[1]["max_error_kpa"]:.3f} → {_g[2]["max_error_kpa"]:.4f} |

    The residual code scores the exact solutions at round-off, so it computes the right PDE. The
    drained route agrees with the pressures on the exact solution, so a disagreement later is
    the network's, not the bookkeeping's. The finite-difference solver (Crank–Nicolson, a ghost
    node mirroring the impermeable base) quarters its error when the cells halve: second order
    (observed orders {fd["order"][0]:.2f} and {fd["order"][1]:.2f}). It is scored at three times
    (T = 0.05, 0.2, and $T_{{90}}$); §4 scores it at every time of the PINN's grid, from 0.01
    years, where the corner makes its error larger.
    """)
    return


@app.cell
def _(FULL, Net, SEED, mo, np, torch):
    # The slides: "the second difference matches autograd to about h^2". Which error is it? The
    # truncation error falls like h^2; round-off in the three evaluations grows like eps / h^2.
    torch.manual_seed(SEED)
    _net = Net(hard_top=True)  # the network of checks(): same seed, same first draws
    _rng = np.random.default_rng(SEED)
    _z = torch.tensor(_rng.uniform(0.05, 0.95, 1000), requires_grad=True)
    _s = torch.tensor(_rng.uniform(0.05, 0.95, 1000))
    _u = _net(_z, _s)
    _uz = torch.autograd.grad(_u, _z, torch.ones_like(_u), create_graph=True)[0]
    _uzz = torch.autograd.grad(_uz, _z, torch.ones_like(_uz))[0].detach()
    h_sweep = {}
    with torch.no_grad():
        _zd = _z.detach()
        for _h in (1e-2, 1e-3, 1e-4, 1e-5):
            _fd2 = (_net(_zd + _h, _s) - 2 * _net(_zd, _s) + _net(_zd - _h, _s)) / _h**2
            h_sweep[_h] = float((_uzz - _fd2).abs().max())
    # The same check in float32: round-off of order eps32 * |u| / h^2 at h = 1e-4
    _net32 = Net(hard_top=True)
    _net32.load_state_dict(_net.state_dict())
    _net32 = _net32.float()
    with torch.no_grad():
        _z32, _s32 = _zd.float(), _s.float()
        _fd32 = (
            _net32(_z32 + 1e-4, _s32) - 2 * _net32(_z32, _s32) + _net32(_z32 - 1e-4, _s32)
        ) / 1e-8
        fd32_gap = float((_uzz - _fd32.double()).abs().max())
        _u_max = float(_net(_zd, _s).abs().max())
    print(
        {f"{_h:.0e}": f"{_e:.1e}" for _h, _e in h_sweep.items()},
        f"float32 at 1e-4: {fd32_gap:.1e}; max |u| {_u_max:.3f}",
    )
    _trunc_1e4 = h_sweep[1e-3] / 100  # the truncation error, extrapolated as h^2 from h = 1e-3
    if FULL:  # round-off-level values: their digits are this machine's
        # From 1e-2 to 1e-3 the gap falls a hundredfold (truncation, h^2); at 1e-4 it is round-off,
        # far above the extrapolated truncation error; at 1e-5 round-off grows (eps / h^2)
        assert 0.009 < h_sweep[1e-3] / h_sweep[1e-2] < 0.011
        assert h_sweep[1e-4] > 10 * _trunc_1e4 and h_sweep[1e-5] > 10 * h_sweep[1e-4]
        assert fd32_gap > 1e4 * h_sweep[1e-4]
    mo.md(
        "Autograd against the central difference, largest gap over 1,000 points: "
        + ", ".join(f"h = {_h:.0e}: {_e:.1e}" for _h, _e in h_sweep.items())
        + f"; in float32 at h = 1e-04: {fd32_gap:.1e}. From 10⁻² to 10⁻³ the gap falls a hundredfold:"
        f" truncation, of order h². At the slides' h = 10⁻⁴ truncation alone would leave about"
        f" {_trunc_1e4:.0e}; the gap is {h_sweep[1e-4]:.0e}, round-off in the three evaluations"
        f" (of order ε|u|/h², with |u| up to {_u_max:.2f} here), which grows again at 10⁻⁵. The"
        " slides' check is at round-off: agreement to that level is all it can show."
    )
    return fd32_gap, h_sweep


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify: the classical solver on the same grid

    The network is built (§2) and its loss derived (§3). The baseline is the classical solver of
    the same equation: Crank–Nicolson finite differences on 80 cells (the 81 depths of the
    scoring grid are its nodes), 16 steps between output times. Both are scored at the same 81
    depths and 500 times, against the series.

    **Predict:** which errs less, and by how much? Which takes longer?
    """)
    return


@app.cell
def _(np):
    def span(vals: list[float | None]) -> dict[str, float | int | None]:
        """Min, median, and max of the values that exist, and how many are missing."""
        v = np.array([x for x in vals if x is not None], dtype=float)
        if len(v) == 0:
            return {"min": None, "median": None, "max": None, "missing": len(vals)}
        return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
                "missing": len(vals) - len(v)}  # fmt: skip

    def worst_abs(vals: list[float | None]) -> float | None:
        """The largest magnitude among the values that exist (None if none do)."""
        have = [abs(x) for x in vals if x is not None]
        return max(have) if have else None

    def consolidation_summary(c: dict, fd: dict) -> dict[str, object]:
        """The ranges and counts the slides quote, from the per-run numbers."""
        fwd = c["forward"]
        fd80 = fd["on_grid"]  # scored on the same grid as the PINN
        gaps = [f["degree_drained"]["t90"] - f["degree_field"]["t90"] for f in fwd]
        fit_rel, pinn_rel = c["series_fit"]["t90_rel"], c["pinn"]["t90_rel"]
        fit_rel = fit_rel[: len(pinn_rel)]  # the draws trained (all 10 in the slides' setting)
        only = c["readings_only"]
        low = min(fwd, key=lambda f: f["loss"])  # the seed a lowest-loss rule would keep
        worst_first = sorted(fwd, key=lambda f: -f["max_kpa"])
        return {
            "forward": {
                "loss": span([f["loss"] for f in fwd]),
                "lowest_loss": {
                    "seed": low["seed"],
                    "loss": low["loss"],
                    "max_kpa": low["max_kpa"],
                    "error_rank": 1 + worst_first.index(low),  # 1: the largest error of the seeds
                    "drained_t90": low["degree_drained"]["t90"],
                    "other_losses": span([f["loss"] for f in fwd if f is not low]),
                },
                "rel_l2": span([f["rel_l2"] for f in fwd]),
                "max_kpa": span([f["max_kpa"] for f in fwd]),
                "max_at_years": span([f["max_at"]["years"] for f in fwd]),
                "max_at_depth": span([f["max_at"]["depth"] for f in fwd]),
                "after_3_months_max_kpa": span([f["after_3_months_max_kpa"] for f in fwd]),
                "seconds": span([f["seconds"] for f in fwd]),
                "error_ratio_to_fd": min(f["max_kpa"] for f in fwd) / fd80["max_kpa"],
                "late_error_ratio_to_fd": min(f["after_3_months_max_kpa"] for f in fwd)
                / fd80["after_3_months_max_kpa"],
                "seconds_ratio_to_fd": float(np.median([f["seconds"] for f in fwd]))
                / fd80["seconds"],
            },
            "conservation": {
                "field_t90": span([f["degree_field"]["t90"] for f in fwd]),
                "drained_t90": span([f["degree_drained"]["t90"] for f in fwd]),
                "worst_gap": worst_abs(gaps),
                "worst_seed": fwd[int(np.argmax(np.abs(gaps)))]["seed"],
                "worst_drained_t90": fwd[int(np.argmax(np.abs(gaps)))]["degree_drained"]["t90"],
                "gaps_over_1pct": int(sum(abs(g) > 0.01 for g in gaps)),
            },
            "no_ic": {
                "degree_min": min(min(f["degree_field"]) for f in c["no_ic"]),
                "max_abs_kpa": max(f["max_abs_kpa"] for f in c["no_ic"]),
            },
            "series_fit": {
                "t90": span(c["series_fit"]["t90"]),
                "cv": span(c["series_fit"]["cv"]),
                "worst_rel": worst_abs(c["series_fit"]["t90_rel"]),
                "seconds": span(c["series_fit"]["seconds"]),
            },
            "pinn": {
                "t90": span(c["pinn"]["t90"]),
                "cv": span(c["pinn"]["cv"]),
                "worst_rel": worst_abs(pinn_rel),
                "seconds": span(c["pinn"]["seconds"]),
                "worse_than_fit": int(
                    sum(
                        p is None or abs(p) > abs(f) for p, f in zip(pinn_rel, fit_rel, strict=True)
                    )
                ),
            },
            "readings_only": {
                "t90": span(only["t90"]),
                "above_1": int(sum(m > 1 for m in only["max_degree"])),
                "max_degree": max(only["max_degree"]),
                "min_u_kpa": min(only["min_u_kpa"]),
                "max_u_kpa": max(only["max_u_kpa"]),
            },
            "window": {
                "cv": span(c["window"]["cv"]),
                "t90": span(c["window"]["t90"]),
                "degree_5_years": span(c["window"]["degree_at"]["5 years"]),
            },
        }

    def river_summary(r: dict) -> dict[str, object]:
        """Per Peclet number: spans over seeds, and how often refinement helped."""
        out: dict[str, object] = {
            "fd_max_error": max(v["max_error"] for v in r["fd"].values()),
            "fd_seconds_max": max(v["seconds"] for v in r["fd"].values()),
        }
        for pe, v in r["pinn"].items():
            out[pe] = {
                "max_error": span(v["max_error"]),
                "mass_error_worst": worst_abs(v["mass_error"]),
                "arrival_5pct_worst_rel": worst_abs(
                    [(a - r["exact_arrival"][pe]["5pct"]) / r["exact_arrival"][pe]["5pct"]
                     for a in v["arrival_5pct"] if a is not None]
                ),
                "over_0.1": int(sum(e > 0.1 for e in v["max_error"])),
                "front_gap_max": worst_abs(
                    [x - tau for x, tau in zip(v["max_at_x"], v["max_at_tau"], strict=True)]
                ),
                "max_at_tau": span(v["max_at_tau"]),
                "shown_error": span(v["shown_error"]),
                "seconds": span(v["seconds"]),
            }  # fmt: skip
        for pe, v in r["refined"].items():
            plain = r["pinn"][pe]["max_error"]
            rand = r["random_added"][pe]["max_error"]
            out[f"{pe} random"] = {
                "max_error": span(rand),
                "mass_error_worst": worst_abs(r["random_added"][pe]["mass_error"]),
            }
            out[f"{pe} refined"] = {
                "max_error": span(v["max_error"]),
                "better": int(sum(a < b for a, b in zip(v["max_error"], plain, strict=True))),
                "beats_random": int(sum(a < b for a, b in zip(v["max_error"], rand, strict=True))),
                "mass_error_worst": worst_abs(v["mass_error"]),
                "ratio_to_fd": min(v["max_error"]) / r["fd"][pe]["max_error"],
            }
        return out

    return consolidation_summary, river_summary


@app.cell
def _(
    CV,
    H,
    N_DRAWS,
    N_IC,
    T90,
    TARGET,
    U0,
    consolidation_jobs,
    consolidation_summary,
    fd,
    np,
    readings,
    series_degree,
    series_fit,
    series_u,
    site,
    summarize_forward,
    t90_true,
    time,
    time_factor_at,
):
    # Collect every consolidation network (submitted in §2), in submission order
    runs = {_k: [_f.result() for _f in _v] for _k, _v in consolidation_jobs.items()}
    # The series fit on all 10 noise draws (milliseconds each), timed
    fits, _fit_secs = [], []
    for _dr in range(N_DRAWS):
        _t0 = time.perf_counter()
        fits.append(series_fit(readings(_dr)))
        _fit_secs.append(time.perf_counter() - _t0)

    def _t90s(rows: list[dict]) -> list[float | None]:
        return [r["t90_field"] for r in rows]

    def _rel(vals: list[float | None]) -> list[float | None]:
        return [None if v is None else (v - t90_true) / t90_true for v in vals]

    _fit_t90 = [time_factor_at(TARGET) * H**2 / _cv for _cv in fits]
    _only = runs["readings_only"]
    _win = runs["window"]
    consolidation = {  # the slides' script's structure, so it compares key by key (§7)
        "site": {
            "readings": {_k: _v.tolist() for _k, _v in site.items()},
            "t90": t90_true,
            "T90": T90,
            "degree_6_months": float(series_degree(CV * 0.5 / H**2)),
            "u_3m_at_t90": float(U0 * series_u(0.75, T90)),
            "degree_exact": dict(
                zip(
                    ("3 months", "1 year", "t90"),
                    series_degree(CV * np.array([0.25, 1.0, t90_true]) / H**2).tolist(),
                    strict=True,
                )
            ),
        },
        "forward": [summarize_forward(_f) for _f in runs["forward"]],
        "no_ic": [
            {
                "seed": _f["seed"],
                "max_abs_kpa": float(np.abs(_f["field"]).max()),
                "degree_field": _f["degree_field"].tolist(),
            }
            for _f in runs["no_ic"]
        ],
        "series_fit": {
            "cv": fits,
            "t90": _fit_t90,
            "t90_rel": _rel(_fit_t90),
            "seconds": _fit_secs,
        },
        "pinn": {
            "cv": [_r["cv"] for _r in runs["pinn"]],
            "t90": _t90s(runs["pinn"]),
            "t90_rel": _rel(_t90s(runs["pinn"])),
            "seconds": [_r["seconds"] for _r in runs["pinn"]],
        },
        "readings_only": {
            "t90": _t90s(_only),
            "max_degree": [float(_r["curve"].max()) for _r in _only],
            "start_points": N_IC,  # u = U0 at t = 0, at the PINN's start depths
            "min_u_kpa": [_r["min_u_kpa"] for _r in _only],
            "max_u_kpa": [_r["max_u_kpa"] for _r in _only],
            "max_u_at": [_r["max_at"] for _r in _only],
            "min_degree": [float(_r["curve"].min()) for _r in _only],
            "degree_at": {
                f"{_y} years": [float(np.interp(_y, _r["curve_t"], _r["curve"])) for _r in _only]
                for _y in (1, 2, 3)
            },
        },
        "window": {
            "cv": [_r["cv"] for _r in _win],
            "t90": _t90s(_win),
            "degree_at": {
                f"{_y} years": [float(np.interp(_y, _r["curve_t"], _r["curve"])) for _r in _win]
                for _y in (1, 3, 5)
            },
        },
    }
    consolidation["summary"] = consolidation_summary(consolidation, fd)
    return consolidation, fits, runs


@app.cell
def _(FULL, consolidation, fd, mo, same, shown):
    _s = consolidation["summary"]["forward"]
    _g = fd["on_grid"]
    _n = len(consolidation["forward"])
    # Finite differences on the PINN's grid (exact): 0.038 kPa, 0.0019 after three months
    assert same(_g, "fd/on_grid")
    assert _g["steps"] == 8000 and _g["outputs"] == 500
    assert shown(_g["max_kpa"], "0.038") and shown(_g["after_3_months_max_kpa"], "0.0019")
    if FULL:  # 2.3-25.5 kPa; 0.35-1.53 after three months; 59 and 184 times FD's
        assert _n == 5
        # Timings vary from run to run (a shared CPU): only their order of magnitude is
        # checked. Finite differences take milliseconds; a PINN minutes, thousands of times longer
        assert 1e-4 < _g["seconds"] < 1.0
        assert 10 < _s["seconds"]["median"] < 3000 and _s["seconds_ratio_to_fd"] > 1e3
        # Both err most at the earliest time scored, 0.01 years, near the drained top
        assert _s["max_at_years"]["min"] == _s["max_at_years"]["max"] == _g["max_at"]["years"]
        assert _s["max_at_depth"]["max"] <= _g["max_at"]["depth"] + 1e-12
        assert shown(_s["max_kpa"]["min"], "2.3") and shown(_s["max_kpa"]["max"], "25.5")
        assert shown(_s["after_3_months_max_kpa"]["min"], "0.35")
        assert shown(_s["after_3_months_max_kpa"]["max"], "1.53")
        assert shown(_s["error_ratio_to_fd"], "59") and shown(_s["late_error_ratio_to_fd"], "184")
        # The lowest training loss kept the worst seed: seed 26, loss 8.8e-05 (the others
        # 0.013 to 0.031), the largest error of the five (25.5 kPa)
        _low = _s["lowest_loss"]
        assert _low["seed"] == 26 and _low["error_rank"] == 1 and shown(_low["loss"], "8.8e-05")
        assert shown(_low["other_losses"]["min"], "0.013")
        assert shown(_low["other_losses"]["max"], "0.031")
        assert same(_s, "consolidation/summary/forward")
    print(f"FD on the grid: {_g}; PINN: {_s}")
    _low = _s["lowest_loss"]
    _loss_note = "." if _n < 2 else (
        f", and the training loss does not tell which seed to keep. The lowest loss, "
        f"{_low['loss']:.1e} (seed {_low['seed']}; the others {_low['other_losses']['min']:.3f} to "
        f"{_low['other_losses']['max']:.3f}), belongs to the run whose error ranks "
        f"{_low['error_rank']} of {_n} from the worst ({_low['max_kpa']:.1f} kPa), and whose water "
        f"balance fails in §5. Choose among runs by a physics check (conservation, a classical "
        f"solution), not by the loss.")  # fmt: skip
    mo.md(f"""
    ### The classical solver wins, by far

    | Forward problem, $c_v$ known | Max error, kPa | After 3 months | Seconds |
    |:---------------------------|---------------:|---------------:|--------:|
    | Finite differences, {_g["cells"]} cells, {_g["steps"]:,} steps | {_g["max_kpa"]:.3f} | {_g["after_3_months_max_kpa"]:.4f} | {_g["seconds"]:.2f} |
    | PINN, {_n} seed{"s" if _n > 1 else ""} | {_s["max_kpa"]["min"]:.1f}–{_s["max_kpa"]["max"]:.1f} | {_s["after_3_months_max_kpa"]["min"]:.2f}–{_s["after_3_months_max_kpa"]["max"]:.2f} | {_s["seconds"]["median"]:.0f} (median) |

    Both are scored at the same 81 depths and 500 times, and both err most in the first days
    near the drained top, where the exact solution jumps: at the earliest time scored
    ({_g["max_at"]["years"]:.2f} years, {365 * _g["max_at"]["years"]:.1f} days), finite differences at {_g["max_at"]["depth"]:.2f} m and the PINNs at
    {_s["max_at_depth"]["min"]:.2f}–{_s["max_at_depth"]["max"]:.2f} m. The best PINN
    errs {_s["error_ratio_to_fd"]:.0f} times more than finite differences there and {_s["late_error_ratio_to_fd"]:.0f} times more after three
    months, and takes minutes, not milliseconds ({_s["seconds"]["median"]:.0f} s against {_g["seconds"]:.3f} s in this run,
    4 threads per fit; timings vary from run to run). Across seeds the PINN's worst error ranges
    {_s["max_kpa"]["min"]:.1f} to {_s["max_kpa"]["max"]:.1f} kPa: the seed matters{_loss_note} For diffusion in 1-D with known
    coefficients, use the classical solver. Grossmann et al. (2024) found the same in a
    systematic comparison with the finite element method; McGreivy and Hakim (2024) found that
    most papers claiming otherwise compared against weak baselines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Learning $c_v$ from six months of readings

    The inverse problem: $c_v$ unknown, the 18 readings given. Three ways to a paving date, on
    10 noise draws of the readings (draw 0 is the site's):

    - **Series fit:** least squares on Terzaghi's series, one unknown, fitted as $\log c_v$ so it
      stays positive (Forsyth §10.2.5 transforms variables for the same kind of reason).
    - **PINN:** the network of §2 with $\log k$ trainable and the readings term added; t90 from
      its own field.
    - **Readings only:** the same network fitted to the readings alone: no residual, no base
      term, no built-in top. It gets the start as the other two have it, through the PINN's own
      start term: 50 kPa at the same 200 depths at $t = 0$.

    **Predict:** which is closest to the truth, 3.39 years, and which is most spread?
    """)
    return


@app.cell
def _(
    CV,
    DEPTHS,
    FIG,
    FULL,
    H,
    INK,
    MARKERS,
    MONTHS,
    MUTED,
    PALETTE,
    TARGET,
    T_MAX,
    U0,
    consolidation,
    field_grid,
    fits,
    mo,
    near,
    np,
    plt,
    runs,
    same,
    series_degree,
    shown,
    t90_true,
    time_factor_at,
):
    _c, _sum = consolidation, consolidation["summary"]
    _fit, _pinn, _only = _sum["series_fit"], _sum["pinn"], _sum["readings_only"]
    _inv0, _only0 = runs["pinn"][0], runs["readings_only"][0]
    _tq = _inv0["curve_t"]
    _exact = series_degree(CV * _tq / H**2)
    # The series fit (exact, all 10 draws in both settings), against the slides' run
    assert same(fits, "consolidation/series_fit/cv")
    assert shown(_fit["t90"]["min"], "3.23") and shown(_fit["t90"]["max"], "3.52")
    assert shown(100 * _fit["worst_rel"], "4.8")
    assert shown(_fit["cv"]["min"], "3.85") and shown(_fit["cv"]["max"], "4.20")
    # "Terzaghi's series with c_v fitted, on top of [the exact curve]"; the PINN "close to it"
    _fit_gap = float(np.abs(series_degree(fits[0] * _tq / H**2) - _exact).max())
    _pinn_gap = float(np.abs(_inv0["curve"] - _exact).max())
    assert _fit_gap < 1e-3
    # Where the readings-only network leaves the physics (draw 0): pore pressures above the
    # load, and below zero, by depth and time (the field is already in kPa)
    _zs, _ts = field_grid()
    _field = _only0["field_kpa"]
    _above, _below = _field > U0, _field < 0
    _depths_above = _zs[_above.any(0)]
    _piezo = [int(np.argmin(np.abs(_zs - _z))) for _z in DEPTHS]  # grid columns at 1, 2, 3 m
    _read = (_ts >= MONTHS[0] / 12) & (_ts <= MONTHS[-1] / 12)  # first to last reading
    _above_read = bool(_above[np.ix_(_read, _piezo)].any())  # at a piezometer, between readings
    _above_piezo = [float(DEPTHS[_j]) for _j, _c_ in enumerate(_piezo) if _above[:, _c_].any()]
    _peak = np.unravel_index(int(_field.argmax()), _field.shape)
    _peak_read = float(_field[np.ix_(_read, _piezo)].max())  # at the piezometers, between readings
    _t_min_u = float(_only0["curve_t"][int(np.argmin(_only0["curve"]))])
    print(f"readings only, draw 0: above the load at the piezometers' depths {_above_piezo} m; "
          f"from the first to the last reading: {_above_read} (up to {_peak_read:.1f} kPa); largest "
          f"{_field.max():.1f} kPa at {_zs[_peak[1]]:.2f} m, {_ts[_peak[0]]:.2f} years; U lowest "
          f"at {_t_min_u:.2f} years")  # fmt: skip
    _min_u = float(_field.min())
    # The stored values are the field's own, in kPa (predict returns kPa): no second u0
    assert near(_c["readings_only"]["min_u_kpa"][0], _min_u)
    assert near(_c["readings_only"]["max_u_kpa"][0], float(_field.max()))
    assert _c["readings_only"]["max_u_at"][0] == {
        "depth": float(_zs[_peak[1]]),
        "years": float(_ts[_peak[0]]),
    }
    # Plausibility, as in the slides' script: a network fitted to readings of 15 to
    # 50 kPa reaches hundreds of kPa at most, not tens of thousands (a second U0 would give that)
    assert all(-50 * U0 < _r["min_u_kpa"] < U0 for _r in runs["readings_only"])
    assert all(0 < _r["max_u_kpa"] < 50 * U0 for _r in runs["readings_only"])
    _true_min_u = _only["min_u_kpa"]  # kPa, over the draws trained
    print(f"draw 0: fit gap {_fit_gap:.1e}, PINN gap {_pinn_gap:.3f}; readings-only u > 50 kPa at "
          f"depths {_depths_above.min() if len(_depths_above) else None}-"
          f"{_depths_above.max() if len(_depths_above) else None} m, times "
          f"{_ts[_above.any(1)].min() if _above.any() else None}-"
          f"{_ts[_above.any(1)].max() if _above.any() else None} years; min u {_min_u:.1f} kPa; "
          f"lowest over draws {_true_min_u:.1f} kPa, highest {_only['max_u_kpa']:.1f} kPa")  # fmt: skip
    if FULL:
        assert shown(_pinn["t90"]["min"], "3.14") and shown(_pinn["t90"]["max"], "3.84")
        assert shown(100 * _pinn["worst_rel"], "13.2") and _pinn["worse_than_fit"] == 10
        assert shown(_pinn["cv"]["min"], "3.56") and shown(_pinn["cv"]["max"], "4.36")
        assert 10 < _pinn["seconds"]["median"] < 3000 and _fit["seconds"]["max"] < 1.0  # timing
        # The readings alone (with the start), every number against the slides' run
        assert same(_c["readings_only"], "consolidation/readings_only")
        assert same(_only, "consolidation/summary/readings_only")
        # The slides: on the site's readings U first passes 90% at 1.16 years, rises to 1.17 at
        # two years and falls back to 0.88 at three; over the draws years to 90% from 0.58 to
        # 4.15, never on 3, U above 1 on 6, pore pressures from -204 to 120 kPa
        _ro = _c["readings_only"]
        assert shown(_ro["t90"][0], "1.16") and shown(_ro["max_degree"][0], "1.17")
        assert shown(_ro["degree_at"]["2 years"][0], "1.17")
        assert shown(_ro["degree_at"]["3 years"][0], "0.88")
        assert shown(_only["t90"]["min"], "0.58") and shown(_only["t90"]["max"], "4.15")
        assert _only["t90"]["missing"] == 3 and _only["above_1"] == 6
        assert shown(_only["min_u_kpa"], "-204") and shown(_only["max_u_kpa"], "120")
    # Left: U(t) on the site's readings; right: years to 90% over the draws
    _fig, (_ax, _bx) = plt.subplots(1, 2, figsize=FIG, width_ratios=(2.2, 1), layout="constrained")
    _ax.axvspan(0, 0.5, color="#f2f2f2", zorder=0)
    _ax.plot(_tq, _exact, color=INK, lw=2.5, ls="-", label="exact")
    _ax.plot(
        _tq, series_degree(fits[0] * _tq / H**2), color=PALETTE[0], ls="--", label="series fit"
    )
    _ax.plot(_tq, _inv0["curve"], color=PALETTE[1], ls="-", label="PINN")
    _ax.plot(_tq, _only0["curve"], color=PALETTE[3], ls="-.", label="readings only")
    _ax.axhline(TARGET, color=MUTED, ls=":", lw=1)
    _lo = min(0.0, float(_only0["curve"].min()))
    _hi = max(1.05, float(_only0["curve"].max()))
    _ax.set(xlim=(0, T_MAX), ylim=(_lo - 0.05 * (_hi - _lo), _hi + 0.05 * (_hi - _lo)),
            xlabel="years", ylabel="average consolidation")  # fmt: skip
    _ax.legend(loc="lower right", frameon=False, fontsize=9)
    for _i, _vals in enumerate(
        (_c["series_fit"]["t90"], [_v for _v in _c["pinn"]["t90"] if _v is not None])
    ):
        _x = _i + np.linspace(-0.15, 0.15, len(_vals))
        _bx.scatter(_x, _vals, s=18, color=PALETTE[_i], marker=MARKERS[_i])
    _bx.axhline(t90_true, color=INK, ls="-", lw=1.5)
    _bx.set(
        xticks=[0, 1], xticklabels=["series\nfit", "PINN"], xlim=(-0.5, 1.5), ylabel="years to 90%"
    )
    _n = len(runs["pinn"])
    _t90_fit0 = time_factor_at(TARGET) * H**2 / fits[0]
    _t90_range = (
        "never reached"
        if _only["t90"]["min"] is None
        else f"from {_only['t90']['min']:.2f} to {_only['t90']['max']:.2f}"
    )
    mo.vstack([_fig, mo.md(f"""
    On the site's readings (left) both physics-based methods land near the truth: the series
    fit's t90 is {_t90_fit0:.2f} years ({100 * (_t90_fit0 / t90_true - 1):+.2f}%), its curve within {_fit_gap:.0e} of the exact one;
    the PINN's {_c["pinn"]["t90"][0]:.2f} years ({100 * _c["pinn"]["t90_rel"][0]:+.1f}%), its curve within {_pinn_gap:.3f}. The network fitted to the
    readings alone gets the start as the PINN has it (the same start term: 50 kPa at the same
    200 depths at $t = 0$), and nothing holds it beyond its readings: its average
    consolidation swings from {_c["readings_only"]["min_degree"][0]:.2f} (at {_t_min_u:.2f} years) to {_c["readings_only"]["max_degree"][0]:.2f}
    (above 1: negative pore pressures, down to {_min_u:,.0f} kPa) and first reaches 90% at
    {_c["readings_only"]["t90"][0] if _c["readings_only"]["t90"][0] is not None else float("nan"):.2f} years. Its pore pressure is largest, {_c["readings_only"]["max_u_kpa"][0]:.0f} kPa, at {_c["readings_only"]["max_u_at"][0]["years"]:.2f} years and
    {_c["readings_only"]["max_u_at"][0]["depth"]:.2f} m{"" if not len(_depths_above) else f"; it exceeds the 50-kPa load at depths from {_depths_above.min():.2f} to {_depths_above.max():.2f} m"},
    {f"and even at a piezometer between its first and last readings (up to {_peak_read:.1f} kPa)" if _above_read else "but never at a piezometer between its first and last readings"}.

    Over the noise draws (right): the series fit's worst t90 is off by {100 * _fit["worst_rel"]:.1f}% (all 10 draws):
    that spread is the readings' noise, since a noise-free fit recovers $c_v$ to round-off (§3).
    The PINN's worst is off by {100 * _pinn["worst_rel"]:.1f}%, and it is further off than the fit on
    {_pinn["worse_than_fit"]} of {_n} draws: the noise plus its own approximation error. Its $c_v$ ranges
    {_pinn["cv"]["min"]:.2f} to {_pinn["cv"]["max"]:.2f} m²/year, the fit's {_fit["cv"]["min"]:.2f} to {_fit["cv"]["max"]:.2f}. Each PINN took about
    {_pinn["seconds"]["median"]:.0f} s; each fit at most {_fit["seconds"]["max"]:.3f} s. The readings alone, over the same draws: years to
    90% {_t90_range}, and never within five years on {_only["t90"]["missing"]};
    average consolidation above 1 on {_only["above_1"]}; pore pressures from {_true_min_u:,.0f} kPa up to
    {_only["max_u_kpa"]:,.0f} kPa.

    The physics is what makes extrapolation possible; the series fit and the PINN both carry
    it.
    """)])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it: four failures

    ### Your turn: predict four failures

    1. Train the PINN without the initial-condition term.
    2. Score the residual only where the readings are: the first six months.
    3. Read the settlement from the water drained, not from the pressures.
    4. A spill in a river: the same method at Péclet numbers 1 to 10,000.

    Write what happens to each. Two minutes alone; then four reveals.
    """)
    return


@app.cell
def _(
    FULL,
    Net,
    SEED,
    consolidation,
    d,
    k_true,
    mo,
    same,
    sample_points,
    shown,
    torch,
):
    _c = consolidation
    _ex, _with = _c["site"]["degree_exact"], _c["forward"][0]["degree_field"]
    # "u = 0 everywhere satisfies everything left in the loss": a network whose last layer is
    # zero outputs 0; the forward loss of fit_consolidation on seed 23's points, term by term
    _net = Net(hard_top=True)
    with torch.no_grad():
        _net.mlp[-1].weight.zero_()
        _net.mlp[-1].bias.zero_()
    _p = sample_points(SEED)
    _zr = torch.tensor(_p["z_res"], requires_grad=True)
    _sr = torch.tensor(_p["s_res"], requires_grad=True)
    _z0, _sb = torch.tensor(_p["z_ic"]), torch.tensor(_p["s_base"])
    _u = _net(_zr, _sr)
    _res = float((d(_u, _sr) - k_true() * d(d(_u, _zr), _zr)).pow(2).mean().detach())
    _zb = torch.ones_like(_sb, requires_grad=True)
    _base = float(d(_net(_zb, _sb), _zb).pow(2).mean().detach())
    _start = float((_net(_z0, torch.zeros_like(_z0)) - 1).pow(2).mean().detach())
    print(f"u = 0: residual {_res}, base {_base}, start {_start}")
    assert _res == 0.0 and _base == 0.0 and _start == 1.0  # zero loss without the start term
    # The exact row (exact): 0.282, 0.562, 0.900
    assert same(_ex, "consolidation/site/degree_exact")
    assert [f"{_v:.3f}" for _v in _ex.values()] == ["0.282", "0.562", "0.900"]
    _rows = "\n".join(
        f"    | PINN without it, seed {_f['seed']} | "
        + " | ".join(f"{_v:.3f}" for _v in _f["degree_field"])
        + " |"
        for _f in _c["no_ic"]
    )
    _ns = _c["summary"]["no_ic"]
    if FULL:  # 0.286, 0.566, 0.898 with the start term; 1.000 without, seeds 23-25; 0.041 kPa
        assert [f"{_v:.3f}" for _v in _with.values()] == ["0.286", "0.566", "0.898"]
        assert [_f["seed"] for _f in _c["no_ic"]] == [23, 24, 25]
        assert all(f"{_v:.3f}" == "1.000" for _f in _c["no_ic"] for _v in _f["degree_field"])
        assert shown(_ns["max_abs_kpa"], "0.041")
    mo.md(f"""
    ### No start condition: settled at once

    | Average consolidation | 3 months | 1 year | t90 |
    |:----------------------|---------:|-------:|----:|
    | Exact | {_ex["3 months"]:.3f} | {_ex["1 year"]:.3f} | {_ex["t90"]:.3f} |
    | PINN with the start term (seed {_c["forward"][0]["seed"]}) | {_with["3 months"]:.3f} | {_with["1 year"]:.3f} | {_with["t90"]:.3f} |
    {_rows}

    First reveal. Without the start term, $u = 0$ everywhere satisfies everything left in the
    loss: the PDE, the no-flow base, the drained top (computed above: a network that outputs 0
    scores {_res:g} on the residual and {_base:g} on the base, and {_start:g} on the start term it no longer
    has). Training finds it: the network's pore pressure is at most {_ns["max_abs_kpa"]:.3f} kPa anywhere
    on the grid, so the clay is "consolidated" from the first day. A loss that a trivial answer
    satisfies will be satisfied by it. Every condition the physics needs must be in the loss or
    built into the network; then check the answer against something known, here 50 kPa at the
    start.
    """)  # fmt: skip
    return


@app.cell
def _(
    CV,
    FIG,
    FULL,
    H,
    INK,
    MUTED,
    PALETTE,
    TARGET,
    T_MAX,
    consolidation,
    mo,
    np,
    plt,
    runs,
    series_degree,
    shown,
):
    _w = consolidation["summary"]["window"]
    _full0, _win0 = runs["pinn"][0], runs["window"][0]
    _tq = _full0["curve_t"]
    _exact = series_degree(CV * _tq / H**2)
    _in = _tq <= 0.5
    _late = _tq >= 1.0
    _gap_in = float(np.abs(_win0["curve"][_in] - _exact[_in]).max())
    _gap_late = float(np.abs(_win0["curve"][_late] - _exact[_late]).max())
    _rw, _rf = _win0["residual"], _full0["residual"]
    _d5 = consolidation["window"]["degree_at"]["5 years"]
    print(f"draw 0, window PINN: |U - exact| up to {_gap_in:.3f} in the first 6 months, "
          f"{_gap_late:.3f} after 1 year; residual RMS {_rw['rms_first_6_months']:.1e} within "
          f"the window, {_rw['rms_after']:.1e} after it (5-year PINN: {_rf['rms_first_6_months']:.1e}, "
          f"{_rf['rms_after']:.1e})")  # fmt: skip
    if FULL:  # 0.36 at five years (draw 0); never 90% on 8 draws; 0.12-1.36; c_v 2.59-3.89
        # Beyond its residual points the PDE is not enforced: after six months its residual is
        # far above the five-year network's there, and its curve departs from the truth
        assert _rw["rms_after"] > 10 * _rf["rms_after"]
        assert _gap_late > 5 * _gap_in
        assert shown(_d5[0], "0.36") and _w["t90"]["missing"] == 8
        assert shown(_w["degree_5_years"]["min"], "0.12") and shown(
            _w["degree_5_years"]["max"], "1.36"
        )
        assert shown(_w["cv"]["min"], "2.59") and shown(_w["cv"]["max"], "3.89")
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    _ax.axvspan(0, 0.5, color="#f2f2f2", zorder=0)
    _ax.plot(_tq, _exact, color=INK, lw=2.5, ls="-", label="exact")
    _ax.plot(_tq, _full0["curve"], color=PALETTE[1], ls="-", label="residual over 5 years")
    _ax.plot(
        _tq, _win0["curve"], color=PALETTE[0], ls="--", label="residual over the first 6 months"
    )
    _ax.axhline(TARGET, color=MUTED, ls=":", lw=1)
    _ax.set(xlim=(0, T_MAX), ylim=(min(0.0, float(_win0["curve"].min())), max(1.05, float(_win0["curve"].max()))),
            xlabel="years", ylabel="average consolidation")  # fmt: skip
    _ax.legend(loc="lower right", frameon=False, fontsize=10)
    _n = len(runs["window"])
    mo.vstack([_fig, mo.md(f"""
    ### No residual points, no physics

    Second reveal. The residual is scored only where its points are. Beyond six months the
    network still meets the drained top (built in) and the no-flow base (scored over all five
    years), but nothing ties how it changes in time to the PDE. On the site's readings, the network with points only in
    the first six months follows the truth there (within {_gap_in:.3f}) and then departs (up to
    {_gap_late:.2f} after the first year; {_d5[0]:.2f} at five years). After six months its PDE residual has
    an RMS of {_rw["rms_after"]:.1e}, against {_rf["rms_after"]:.1e} for the network with points over five years
    (within the first six months both are dominated by the corner: {_rw["rms_first_6_months"]:.1e} and {_rf["rms_first_6_months"]:.1e}).
    Over {_n} draws: the field never reaches 90% within five years on {_w["t90"]["missing"]} draws, and its
    average consolidation at five years ranges from {_w["degree_5_years"]["min"]:.2f} to {_w["degree_5_years"]["max"]:.2f}. The learned
    $c_v$ is off too: {_w["cv"]["min"]:.2f} to {_w["cv"]["max"]:.2f} m²/year against 4. Residual points cost no data: put
    them everywhere the network will be asked a question.
    """)])  # fmt: skip
    return


@app.cell
def _(
    FIG,
    FULL,
    INK,
    MUTED,
    PALETTE,
    TARGET,
    checks_out,
    consolidation,
    mo,
    np,
    plt,
    runs,
    shown,
):
    _fwd = consolidation["forward"]
    # "The PDE holds only approximately, worst near the drained top at the start": each seed's
    # residual at 20,000 fresh points (§2)
    _res = [_f["residual"] for _f in runs["forward"]]
    _res_days = 365 * max(_r["max_at_years"] for _r in _res)
    _res_depth = max(_r["max_at_depth"] for _r in _res)
    _res_ratio = min(_r["rms_first_6_months"] / _r["rms_after"] for _r in _res)
    print(f"residual's largest value, over seeds: within {_res_days:.1f} days and {_res_depth:.2f} m; "
          f"RMS in the first six months at least {_res_ratio:.0f} times the RMS after")  # fmt: skip
    _s = consolidation["summary"]["conservation"]
    _gap_t90 = np.array([_f["degree_drained"]["t90"] - _f["degree_field"]["t90"] for _f in _fwd])
    _gap_3m = np.array(
        [_f["degree_drained"]["3 months"] - _f["degree_field"]["3 months"] for _f in _fwd]
    )
    _big = np.abs(_gap_t90) > 0.01
    # Where does the disagreement come from? The drained water is integrated from t = 0, so a
    # gap already there at 3 months was made near the corner, where the network errs most
    print("gap at 3 months:", np.round(_gap_3m, 4), "; at t90:", np.round(_gap_t90, 4))
    _share = _gap_3m[_big] / _gap_t90[_big]
    print("share of each >1% gap already there at 3 months:", np.round(_share, 3))
    if FULL:  # 0.898-0.901 and 0.878-1.429; 3 of 5 over 1%; seed 26 at 1.43
        assert shown(_s["field_t90"]["min"], "0.898") and shown(_s["field_t90"]["max"], "0.901")
        assert shown(_s["drained_t90"]["min"], "0.878") and shown(_s["drained_t90"]["max"], "1.429")
        assert _s["gaps_over_1pct"] == 3 and _s["worst_seed"] == 26
        assert shown(_s["worst_drained_t90"], "1.43") and _s["worst_drained_t90"] > 1
        assert np.all(_share > 0.9) and shown(checks_out["drained_vs_field_exact"], "6e-08")
        assert _res_days < 3.65 and _res_depth < 0.5 and _res_ratio > 100
    _fig, _ax = plt.subplots(figsize=FIG, layout="constrained")
    for _i, _f in enumerate(_fwd):
        _a, _b = _f["degree_field"]["t90"], _f["degree_drained"]["t90"]
        _ax.plot([_i, _i], [_a, _b], color=MUTED, lw=1, ls="-")
        _ax.scatter([_i], [_a], color=PALETTE[0], marker="o", s=40, zorder=3,
                    label="from the pore pressures" if _i == 0 else None)  # fmt: skip
        _ax.scatter([_i], [_b], color=PALETTE[1], marker="s", s=40, zorder=3,
                    label="from the water drained" if _i == 0 else None)  # fmt: skip
    _ax.axhline(TARGET, color=INK, ls="-", lw=1)
    _ax.set(xticks=range(len(_fwd)), xticklabels=[f"seed {_f['seed']}" for _f in _fwd],
            ylabel="consolidation at the true t90")  # fmt: skip
    _ax.legend(loc="upper left", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 1.0))
    mo.vstack([_fig, mo.md(f"""
    ### Two readings of one settlement disagree

    Third reveal. Water is conserved: what leaves through the drained top must equal the drop
    in the layer's pore pressures, and on the exact solution the two agree to
    {checks_out["drained_vs_field_exact"]:.0e} (§3). For each seed, at the true t90: from the pore pressures
    {_s["field_t90"]["min"]:.3f} to {_s["field_t90"]["max"]:.3f} (the truth is 0.90); from the water drained {_s["drained_t90"]["min"]:.3f} to
    {_s["drained_t90"]["max"]:.3f}. The two readings differ by more than 1% (of the final settlement) on {_s["gaps_over_1pct"]}
    of {len(_fwd)} seeds; on seed {_s["worst_seed"]} the water drained says {_s["worst_drained_t90"]:.2f}{", more water out than the layer held" if _s["worst_drained_t90"] > 1 else ""}. The PDE holds only approximately, worst near the drained top at the start: on every seed the
    residual's largest value lies within the first {_res_days:.1f} days and the top {_res_depth:.2f} m, and its RMS in the
    first six months is at least {_res_ratio:.0f} times its RMS after them. The drained water is integrated from $t = 0$: of each gap over 1%, {100 * _share.min() if len(_share) else float("nan"):.0f}% or more is already there at three
    months, so it was made early, at the corner where the network errs most. A model that
    conserves only approximately can be right on one output and wrong on another: check every
    quantity you report, and prefer a method that conserves by construction (finite volumes)
    when a balance matters.
    """)])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A spill in a river: the same method, a sharper front

    Fourth failure. A continuous release enters a river reach at $X = 0$; the concentration $c$
    follows the advection–dispersion equation, in units of the reach's length $L$ and of the
    time the water takes to cross it ($X = x/L$, $\tau = vt/L$):

    $$\frac{\partial c}{\partial \tau} + \frac{\partial c}{\partial X} = \frac{1}{\mathrm{Pe}}\,\frac{\partial^2 c}{\partial X^2}, \qquad \mathrm{Pe} = \frac{vL}{D},$$

    with $c = 1$ at the inlet, $c = 0$ at the start, and Ogata and Banks's (1961) exact solution
    at the outlet $X = 1$ (their solution is for a semi-infinite river). The Péclet number
    compares carrying with spreading: 5 km of river at 0.5 m/s with $D$ = 2.5 m²/s (assumed)
    gives 1,000. The PINN is the same network and training as the clay's, on $(X, s = \tau/0.75)$,
    without the hard top; five seeds at each Péclet number from 1 to 10,000, scored at 1,001
    points and 71 times ($\tau$ = 0.05 to 0.75). The classical baseline: Crank–Nicolson with
    central differences and at least 4 cells per unit of Pe (cell Péclet number 0.25).

    **Residual-based refinement** (Lu et al., 2021) at Pe 1,000 and 10,000: after Adam, score
    the residual at 20,000 fresh random points, add the 1,000 worst to the training points, then
    finish with L-BFGS. One round here; Lu et al. repeat the round until the residual is small
    enough, so this is the method's first step, not all of it.
    """)
    return


@app.cell
def _(
    INTAKE,
    N_BC,
    N_IC,
    N_RES,
    Net,
    RAR_ADD,
    RAR_POOL,
    TAU,
    d,
    np,
    ogata_banks,
    setup_worker,
    solve_banded,
    time,
    torch,
    train,
):
    def fd_transport(pe: float, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Crank-Nicolson with central differences, cell Peclet 0.25, for the river.

        Args:
            pe: Peclet number.
            taus: Output times, multiples of 0.01.

        Returns:
            Nodes X (nx + 1,), concentrations at ``taus`` (len(taus), nx + 1), seconds.
        """
        t0 = time.perf_counter()
        nx = max(400, 4 * int(pe))
        dx = 1 / nx
        per = max(10, int(np.ceil(0.01 / dx)))  # steps per 0.01 of tau: Courant at most 1
        dt = 0.01 / per
        x = np.linspace(0, 1, nx + 1)
        lo = 1 / (2 * dx) + 1 / (pe * dx**2)  # coefficient of c[i-1] in dc/dtau
        di = -2 / (pe * dx**2)
        up = -1 / (2 * dx) + 1 / (pe * dx**2)  # coefficient of c[i+1]
        n = nx - 1
        ab = np.zeros((3, n))  # banded (I - dt/2 A)
        ab[0, 1:] = -0.5 * dt * up
        ab[1, :] = 1 - 0.5 * dt * di
        ab[2, :-1] = -0.5 * dt * lo
        c = np.zeros(nx + 1)
        out, k = [], 0
        targets = np.rint(np.asarray(taus) / dt).astype(int)
        for step in range(1, targets.max() + 1):
            tau = step * dt
            right_new = float(ogata_banks(1.0, tau, pe))
            inner = c[1:-1]
            rhs = inner + 0.5 * dt * (lo * c[:-2] + di * inner + up * c[2:])
            rhs[0] += 0.5 * dt * lo * 1.0  # inlet held at 1 for tau > 0
            rhs[-1] += 0.5 * dt * up * right_new
            c = np.r_[1.0, solve_banded((1, 1), ab, rhs), right_new]
            if step == targets[k]:
                out.append(c.copy())
                k += 1
                if k == len(targets):
                    break
        return x, np.array(out), time.perf_counter() - t0

    def transport_grid() -> tuple[np.ndarray, np.ndarray]:
        """X (1001,) and tau (71,) for the river's errors: tau from 0.05 to 0.75 by 0.01."""
        return np.linspace(0, 1, 1001), np.round(np.arange(5, 76) / 100, 2)

    def first_crossing(ts: np.ndarray, c: np.ndarray, th: float) -> float | None:
        """The first time ``c`` reaches ``th``, or None."""
        hit = np.nonzero(c >= th)[0]
        return float(ts[hit[0]]) if len(hit) else None

    def mass_balance_net(net: torch.nn.Module, pe: float) -> dict[str, float]:
        """Mass gained in the reach over [0, TAU] against the net inflow through its two ends."""
        xq = torch.linspace(0, 1, 4001)
        with torch.no_grad():
            gain = float(torch.trapezoid(net(xq, torch.ones_like(xq)), xq)) - float(
                torch.trapezoid(net(xq, torch.zeros_like(xq)), xq)
            )
        sq = torch.linspace(0, 1, 4001)

        def flux(x_end: float) -> torch.Tensor:  # advective + dispersive flux, per unit tau
            x = torch.full_like(sq, x_end).requires_grad_(True)
            c = net(x, sq)
            return (c - d(c, x) / pe).detach()

        inflow = TAU * float(torch.trapezoid(flux(0.0) - flux(1.0), sq))
        return {"gain": gain, "inflow": inflow, "error": (gain - inflow) / inflow}

    def transport_job(pe: float, seed: int, mode: str) -> dict:
        """The river: a PINN for c_tau + c_X = c_XX / Pe; inputs (X, s = tau / TAU)."""
        setup_worker()
        t0 = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.default_rng([seed, int(pe)])
        net = Net(hard_top=False)
        pts = {
            "x": torch.tensor(rng.uniform(0, 1, N_RES), requires_grad=True),
            "s": torch.tensor(rng.uniform(0, 1, N_RES), requires_grad=True),
        }
        x0 = torch.tensor(rng.uniform(0, 1, N_IC))
        sb = torch.tensor(rng.uniform(0, 1, N_BC))
        right = torch.tensor(ogata_banks(1.0, TAU * sb.numpy(), pe))

        def residual(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            c = net(x, s)
            cx = d(c, x)
            return d(c, s) / TAU + cx - d(cx, x) / pe

        def loss_fn() -> torch.Tensor:
            res = residual(pts["x"], pts["s"]).pow(2).mean()
            inlet = (net(torch.zeros_like(sb), sb) - 1).pow(2).mean()
            outlet = (net(torch.ones_like(sb), sb) - right).pow(2).mean()
            start = net(x0, torch.zeros_like(x0)).pow(2).mean()
            return res + inlet + outlet + start

        def add_points() -> None:
            """After Adam: add RAR_ADD points from a fresh pool of RAR_POOL."""
            xc = torch.tensor(rng.uniform(0, 1, RAR_POOL), requires_grad=True)
            sc = torch.tensor(rng.uniform(0, 1, RAR_POOL), requires_grad=True)
            if mode == "refine":  # Lu et al. (2021): the points with the largest residual
                pick = residual(xc, sc).detach().abs().argsort()[-RAR_ADD:]
            else:  # the control: as many points from the same pool, at random (it is random)
                pick = torch.arange(RAR_ADD)
            pts["x"] = torch.cat([pts["x"].detach(), xc.detach()[pick]]).requires_grad_(True)
            pts["s"] = torch.cat([pts["s"].detach(), sc.detach()[pick]]).requires_grad_(True)

        assert mode in ("plain", "refine", "random"), mode
        loss = train(list(net.parameters()), loss_fn, None if mode == "plain" else add_points)
        secs = time.perf_counter() - t0
        xs, taus = transport_grid()
        xx, tt = np.meshgrid(xs, taus)
        with torch.no_grad():
            c = (
                net(torch.tensor(xx.ravel()), torch.tensor(tt.ravel() / TAU))
                .numpy()
                .reshape(xx.shape)
            )
            ta = np.linspace(0, TAU, 3001)
            at_intake = net(torch.full((3001,), INTAKE), torch.tensor(ta / TAU)).numpy()
        return {
            "pe": pe,
            "seed": seed,
            "mode": mode,
            "loss": loss,
            "seconds": secs,
            "field": c,
            "arrival": {th: first_crossing(ta, at_intake, th) for th in (0.05, 0.5)},
            "mass": mass_balance_net(net, pe),
        }

    def fd_job(pe: float) -> dict[str, float]:
        """Finite differences for the river, interpolated to the scoring grid, against the exact."""
        xs, taus = transport_grid()
        xg, cg, secs = fd_transport(pe, taus)
        cf = np.array([np.interp(xs, xg, row) for row in cg])
        exact = ogata_banks(xs[None, :], taus[:, None], pe)
        assert cf.shape == exact.shape == (len(taus), len(xs))
        return {"max_error": float(np.abs(cf - exact).max()), "seconds": secs, "cells": len(xg) - 1}

    return fd_job, first_crossing, transport_grid, transport_job


@app.cell
def _(
    FD_PECLETS,
    PINN_PECLETS,
    RAR_RUN,
    RIVER_SEEDS,
    SEED,
    executor,
    fd_job,
    transport_job,
):
    # The finite-difference runs go to the workers too (Pe 10,000 first: 40,000 cells, the
    # longest), then the PINNs in the script's order
    river_jobs = {
        "fd": {_pe: executor.submit(fd_job, _pe) for _pe in sorted(FD_PECLETS, reverse=True)},
        "pinn": [executor.submit(transport_job, _pe, SEED + _s, "plain")
                 for _pe in PINN_PECLETS for _s in range(RIVER_SEEDS)],
        "refined": [executor.submit(transport_job, _pe, SEED + _s, "refine")
                    for _pe in RAR_RUN for _s in range(RIVER_SEEDS)],
        "random": [executor.submit(transport_job, _pe, SEED + _s, "random")
                   for _pe in RAR_RUN for _s in range(RIVER_SEEDS)],
    }  # fmt: skip
    print(f"{sum(len(river_jobs[_k]) for _k in ('pinn', 'refined', 'random'))} river networks and "
          f"{len(river_jobs['fd'])} finite-difference runs submitted")  # fmt: skip
    return (river_jobs,)


@app.cell
def _(SHOW_TAU, brentq, mo, np, ogata_banks):
    # "The front's width shrinks like 1/sqrt(Pe)": the exact profile's 10%-90% width at
    # tau = 0.1, and how it grows with time (sharpest at the earliest time scored, 0.05)
    def front_width(pe: float, tau: float) -> float:
        """Distance between c = 0.9 and c = 0.1 on the exact profile at ``tau``."""

        def at(level: float) -> float:
            return brentq(lambda x: float(ogata_banks(x, tau, pe)) - level, 0.0, 1.0, xtol=1e-14)

        return at(0.1) - at(0.9)

    widths = {_pe: front_width(_pe, SHOW_TAU) for _pe in (100, 1000, 10000)}
    _w = list(widths.values())
    _slopes = [float(np.log10(_w[_i + 1] / _w[_i])) for _i in range(2)]  # per decade of Pe
    _in_time = [front_width(1000, _t) for _t in (0.05, SHOW_TAU, 0.75)]
    print(f"front width at tau = 0.1: {widths}; slopes per decade {_slopes}; "
          f"at Pe 1,000, tau = 0.05, 0.1, 0.75: {np.round(_in_time, 5)}")  # fmt: skip
    # 1 / sqrt(Pe) is the large-Pe limit (width ~ sqrt(tau / Pe)): slope -0.5 per decade
    assert abs(_slopes[1] + 0.5) < 0.01 and abs(_slopes[0] + 0.5) < 0.05
    assert _in_time[0] < _in_time[1] < _in_time[2]
    assert 0.5 * 5000 / 2.5 == 1000  # the notes' river: v L / D, 0.5 m/s, 5 km, 2.5 m2/s
    mo.md(
        "The exact front's width (10% to 90%) at τ = 0.1: "
        + ", ".join(f"{_w:.4f} at Pe {_pe:,}" for _pe, _w in widths.items())
        + f"; on log axes its slope is {_slopes[0]:.3f} from 100 to 1,000 and {_slopes[1]:.3f} from"
        " 1,000 to 10,000: the width shrinks like 1/√Pe, the more closely the larger Pe. At Pe 1,000"
        f" it grows from {_in_time[0]:.4f} at τ = 0.05 to {_in_time[2]:.4f} at 0.75: the front is"
        " sharpest at the earliest time scored."
    )
    return


@app.cell
def _(
    FD_PECLETS,
    INTAKE,
    PECLETS,
    PINN_PECLETS,
    RAR_RUN,
    SHOW_TAU,
    TAU,
    first_crossing,
    np,
    ogata_banks,
    river_jobs,
    river_summary,
    transport_grid,
):
    _xs, _taus = transport_grid()
    _k_show = int(np.argmin(np.abs(_taus - SHOW_TAU)))

    def _score(r: dict) -> dict:
        """The script's errors of one river run, on the scoring grid."""
        exact = ogata_banks(_xs[None, :], _taus[:, None], r["pe"])
        err = np.abs(r["field"] - exact)
        assert err.shape == (len(_taus), len(_xs))
        it, ix = np.unravel_index(int(err.argmax()), err.shape)
        return r | {
            "max_error": float(err.max()),
            "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(exact)),
            "max_at": {"x": float(_xs[ix]), "tau": float(_taus[it])},  # the front sits at X = tau
            "shown_error": float(err[_k_show].max()),
        }

    river_runs = [
        _score(_f.result())
        for _f in river_jobs["pinn"] + river_jobs["refined"] + river_jobs["random"]
    ]
    _fd = {_pe: river_jobs["fd"][_pe].result() for _pe in FD_PECLETS}

    def _table(mode: str) -> dict[str, dict[str, list]]:
        out = {}
        for pe in PINN_PECLETS if mode == "plain" else RAR_RUN:
            sel = [r for r in river_runs if r["pe"] == pe and r["mode"] == mode]
            out[str(pe)] = {
                "max_error": [r["max_error"] for r in sel],
                "rel_l2": [r["rel_l2"] for r in sel],
                "mass_error": [r["mass"]["error"] for r in sel],
                "arrival_5pct": [r["arrival"][0.05] for r in sel],
                "arrival_50pct": [r["arrival"][0.5] for r in sel],
                "max_at_x": [r["max_at"]["x"] for r in sel],
                "max_at_tau": [r["max_at"]["tau"] for r in sel],
                "shown_error": [r["shown_error"] for r in sel],
                "seconds": [r["seconds"] for r in sel],
            }
        return out

    _ta = np.linspace(0, TAU, 3001)
    river = {
        "fd": {str(_k): _v for _k, _v in _fd.items()},
        "exact_arrival": {
            str(_pe): {
                "5pct": first_crossing(_ta, ogata_banks(INTAKE, _ta, _pe), 0.05),
                "50pct": first_crossing(_ta, ogata_banks(INTAKE, _ta, _pe), 0.5),
            }
            for _pe in PECLETS
        },
        "pinn": _table("plain"),
        "refined": _table("refine"),
        "random_added": _table("random"),
    }
    river["summary"] = river_summary(river)
    return river, river_runs


@app.cell
def _(
    FD_PECLETS,
    FIG,
    FULL,
    INK,
    PALETTE,
    PECLETS,
    SHOW_PE,
    SHOW_TAU,
    mo,
    np,
    ogata_banks,
    plt,
    river,
    river_runs,
    same,
    shown,
    transport_grid,
):
    _fd, _sum = river["fd"], river["summary"]
    # Finite differences (exact): errors and cells at each Peclet number run, as in the slides
    for _pe, _v in _fd.items():
        assert same(_v["max_error"], f"river/fd/{_pe}/max_error")
        assert same(_v["cells"], f"river/fd/{_pe}/cells") and _v["cells"] == max(400, 4 * int(_pe))
    assert _sum["fd_max_error"] <= 0.008 and shown(_sum["fd_max_error"], "0.008")
    _p1000 = _sum[str(SHOW_PE)]
    if FULL:
        assert len(_fd) == 5 and shown(_fd["10000"]["max_error"], "0.002")
        assert 10 < _fd["10000"]["seconds"] < 1000  # 106 s on the slides' run (timing)
        assert shown(_p1000["shown_error"]["max"], "0.31")  # five seeds miss by up to 0.31
        assert shown(_sum["1"]["max_error"]["median"], "0.016")
        assert shown(_sum["10000"]["max_error"]["median"], "0.48")
        # At Pe 1,000 every seed's largest error is within 0.006 of the front, at tau = 0.05
        assert _p1000["front_gap_max"] <= 0.006 + 1e-12 and shown(_p1000["front_gap_max"], "0.006")
        assert _p1000["max_at_tau"]["min"] == _p1000["max_at_tau"]["max"] == 0.05
        # From Pe 1,000 every seed's largest error exceeds 0.1; the mass balance: 0.7% and 6.3%
        assert _sum["1000"]["over_0.1"] == 5 and _sum["10000"]["over_0.1"] == 5
        assert shown(100 * _sum["1000"]["mass_error_worst"], "0.7")
        assert shown(100 * _sum["10"]["mass_error_worst"], "6.3")
    _medians = {_pe: _sum[str(_pe)]["max_error"]["median"] for _pe in PECLETS if str(_pe) in _sum}
    print(f"PINN medians of the largest error: {_medians}")
    # Left: the profiles at tau = 0.1 for Pe 1,000; right: the largest error by Peclet number
    _xs, _taus = transport_grid()
    _k = int(np.argmin(np.abs(_taus - SHOW_TAU)))
    _fig, (_ax, _bx) = plt.subplots(1, 2, figsize=FIG, layout="constrained")
    _ax.plot(_xs, ogata_banks(_xs, _taus[_k], SHOW_PE), color=INK, lw=3, ls="-", label="exact")
    for _j, _r in enumerate(
        _x for _x in river_runs if _x["pe"] == SHOW_PE and _x["mode"] == "plain"
    ):
        _ax.plot(_xs, _r["field"][_k], color=PALETTE[1], lw=1.2, ls="-",
                 label="PINN, each seed" if _j == 0 else None)  # fmt: skip
    _ax.set(xlim=(0, 2.5 * SHOW_TAU), xlabel="distance down the river, X", ylabel="concentration")
    _ax.legend(loc="upper right", frameon=False, fontsize=10)
    for _mode, _color, _marker, _dx, _label in (
        ("plain", PALETTE[1], "o", -0.08, "PINN"),
        ("refine", PALETTE[2], "^", 0.0, "PINN + refinement"),
        ("random", PALETTE[3], "D", 0.08, "PINN + random points"),
    ):
        _sel = [_r for _r in river_runs if _r["mode"] == _mode]
        _bx.scatter([_r["pe"] * 10**_dx for _r in _sel], [_r["max_error"] for _r in _sel], s=20,
                    color=_color, marker=_marker, label=_label)  # fmt: skip
    _bx.plot(FD_PECLETS, [_fd[str(_pe)]["max_error"] for _pe in FD_PECLETS], color=PALETTE[0],
             ls="--", marker="s", label="finite differences")  # fmt: skip
    _bx.set(xscale="log", yscale="log", xlabel="Péclet number", ylabel="max error")
    _bx.legend(loc="upper left", frameon=False, fontsize=9)
    _lines = " ".join(f"{_med:.3f} at {_pe:,};" for _pe, _med in _medians.items())
    mo.vstack([_fig, mo.md(f"""
    ### Sharp fronts break the PINN

    The PINN's median largest error, by Péclet number: {_lines} finite differences stay at or
    below {_sum["fd_max_error"]:.3f}. At Pe {SHOW_PE:,} the seeds miss the front at τ = 0.1 by up to
    {_p1000["shown_error"]["max"]:.2f}, every seed's largest error sits within {_p1000["front_gap_max"]:.3f} of the front's centre
    ($X = \\tau$), at the earliest time scored (τ = {_p1000["max_at_tau"]["min"]:.2f}), when the front is sharpest
    (its width is computed above). Krishnapriyan et al. (2021) report the same failure for
    convection and trace it to a hard optimization problem, not to the network's size. The mass
    balance does not raise this alarm (worst {100 * _sum[str(SHOW_PE)]["mass_error_worst"]:.1f}% at {SHOW_PE:,}){"" if "10" not in _sum else f", but it flags low-Pe runs (worst {100 * _sum['10']['mass_error_worst']:.1f}% at 10)"}:
    this check missed this failure.
    """)])  # fmt: skip
    return


@app.cell
def _(FULL, RAR_RUN, mo, river, shown):
    _sum, _fd = river["summary"], river["fd"]
    _cells = []
    for _pe in RAR_RUN:
        _p, _r = _sum[str(_pe)]["max_error"], _sum[f"{_pe} refined"]
        _cells.append((_pe, _p, _r, _fd[str(_pe)]["max_error"], _sum[f"{_pe} random"]))
    assert shown(_fd["1000"]["max_error"], "0.005")
    if FULL:  # the table, and the notes' 5 of 5, 4 of 5, 0.10%, 1.5 and 54 times; random points
        _qa, _qb = _sum["1000 random"], _sum["10000 random"]
        assert shown(_qa["max_error"]["min"], "0.099") and shown(_qa["max_error"]["max"], "0.42")
        assert shown(_qb["max_error"]["min"], "0.17") and shown(_qb["max_error"]["max"], "0.74")
        assert (
            _sum["1000 refined"]["beats_random"] == 5 and _sum["10000 refined"]["beats_random"] == 3
        )
        assert shown(100 * _qa["mass_error_worst"], "0.61")
        _a, _b = _sum["1000"]["max_error"], _sum["10000"]["max_error"]
        _ra, _rb = _sum["1000 refined"], _sum["10000 refined"]
        assert shown(_a["min"], "0.128") and shown(_a["max"], "0.43")
        assert shown(_b["min"], "0.20") and shown(_b["max"], "0.77")
        assert shown(_ra["max_error"]["min"], "0.008") and shown(_ra["max_error"]["max"], "0.35")
        assert shown(_rb["max_error"]["min"], "0.09") and shown(_rb["max_error"]["max"], "0.22")
        assert _ra["better"] == 5 and _rb["better"] == 4
        assert shown(100 * _ra["mass_error_worst"], "0.10")
        assert shown(_ra["ratio_to_fd"], "1.5") and shown(_rb["ratio_to_fd"], "54")
    _head = " | ".join(f"Péclet {_pe:,}" for _pe in RAR_RUN)
    _rule = " | ".join("-------------:" for _ in RAR_RUN)
    _plain = " | ".join(f"{_p['min']:.3f}–{_p['max']:.2f}" for _, _p, _, _, _ in _cells)
    _ref = " | ".join(
        f"{_r['max_error']['min']:.3f}–{_r['max_error']['max']:.2f}" for _, _, _r, _, _ in _cells
    )
    _rnd = " | ".join(
        f"{_q['max_error']['min']:.3f}–{_q['max_error']['max']:.2f}" for *_, _q in _cells
    )
    _fdrow = " | ".join(f"{_f:.3f}" for _, _, _, _f, _ in _cells)
    _n_of = {_pe: len(river["refined"][str(_pe)]["max_error"]) for _pe in RAR_RUN}
    _better = "; ".join(
        f"{_r['better']} of {_n_of[_pe]} seeds at {_pe:,}" for _pe, _, _r, _, _ in _cells
    )
    _ratio = "; ".join(
        f"{_r['ratio_to_fd']:.1f} times smaller at {_pe:,}" for _pe, _, _r, _, _ in _cells
    )
    _vs_random = "; ".join(
        f"{_r['beats_random']} of {_n_of[_pe]} seeds at {_pe:,} (medians {_r['max_error']['median']:.3f}"
        f" against {_q['max_error']['median']:.3f})"
        for _pe, _, _r, _, _q in _cells
    )
    _rnd_mass = "; ".join(
        f"{100 * _q['mass_error_worst']:.2f}% at {_pe:,} (refined {100 * _r['mass_error_worst']:.2f}%)"
        for _pe, _, _r, _, _q in _cells
    )
    mo.md(f"""
    ### Refinement helps; FD still wins

    | Max error, over seeds | {_head} |
    |:----------------------|{_rule}|
    | PINN | {_plain} |
    | PINN, refined | {_ref} |
    | PINN, 1,000 random points added | {_rnd} |
    | Finite differences | {_fdrow} |

    Refinement lowered the error on {_better}, and nearly closed the mass balance (worst
    {100 * _sum[f"{RAR_RUN[0]} refined"]["mass_error_worst"]:.2f}% at {RAR_RUN[0]:,}). Finite differences still beat the best refined seed: an error
    {_ratio}.

    Is it the extra points or where they go? The refined runs train their L-BFGS phase on 3,000
    points, the plain ones on 2,000. Adding 1,000 points from the same fresh pool at random
    instead, refinement still has the lower error on {_vs_random}. The random points' worst
    mass error: {_rnd_mass}. On five paired seeds a 3-to-2 split is close: read it together
    with the medians and the mass balance.
    """)  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide: who sets the paving date?

    ### Vote

    - **A.** A network fitted to the readings alone.
    - **B.** Terzaghi's solution with $c_v$ fitted to the readings.
    - **C.** The PINN, with $c_v$ learned.
    - **D.** B now, refitted as each month's readings arrive.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(consolidation, mo, runs):
    _s = consolidation["summary"]
    _only, _fit, _pinn = _s["readings_only"], _s["series_fit"], _s["pinn"]
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        A: a network fitted to the readings alone gives years to 90% {"never" if _only["t90"]["min"] is None else f"from {_only['t90']['min']:.2f}"} to never
        ({_only["t90"]["missing"]} of {len(runs["readings_only"])} draws), and average consolidation above 1 on {_only["above_1"]}. B: the worst of 10 noise draws is off by
        {100 * _fit["worst_rel"]:.1f}%, in milliseconds. C: further off than B on {_pinn["worse_than_fit"]} of {len(runs["pinn"])} draws,
        minutes per fit, and the conservation and corner checks still to pass. D: B, checked
        against the readings as they arrive, and the defensible answer: the site's $c_v$ is an
        assumption until the readings confirm it, and refitting each month (below) is Peck's
        (1969) observational method: the prediction is checked against the site, not trusted.
        """)
        }
    )
    return


@app.cell
def _(
    H,
    MONTHS,
    SIGMA,
    TARGET,
    U0,
    fits,
    mo,
    near,
    np,
    series_fit,
    series_u,
    site,
    t90_true,
    time_factor_at,
):
    # Option D on the site's readings: refit c_v each month, from the readings so far
    _rows = []
    for _k in range(2, len(MONTHS) + 1):
        _keep = site["t"] <= MONTHS[_k - 1] / 12 + 1e-12
        _data = {_key: _v[_keep] for _key, _v in site.items()}
        _cv = series_fit(_data)
        _fit_u = U0 * series_u(_data["z"] / H, _cv * _data["t"] / H**2)
        _rms = float(np.sqrt(np.mean((_fit_u - _data["u"]) ** 2)))
        _rows.append((_k, int(_keep.sum()), _cv, time_factor_at(TARGET) * H**2 / _cv, _rms))
    assert near(_rows[-1][2], fits[0])  # all six months: the series fit of §4
    _table = "\n".join(
        f"    | {_k} | {_n} | {_cv:.3f} | {_t:.2f} | {_r:.2f} |" for _k, _n, _cv, _t, _r in _rows
    )
    mo.md(f"""
    | Months read | Readings | $c_v$, m²/year | Years to 90% | RMS misfit, kPa |
    |---:|---:|---:|---:|---:|
    {_table}

    The truth is 4 m²/year and {t90_true:.2f} years; the readings' noise is {SIGMA:.0f} kPa. These readings
    were simulated from Terzaghi's model, so the model fits them by construction and the misfit
    can only reflect the noise. On a real clay (layered, or with creep after the pore pressures
    have dissipated) a misfit that grows month after month would be the sign that the model
    does not fit; this simulated site cannot show that.
    """)  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### When a physics-informed network pays

    | Situation | Reach for |
    |:----------|:----------|
    | PDE known, parameters known | a classical solver: finite differences, finite elements |
    | PDE known, a few unknown parameters | least squares on the solver |
    | PDE known, an unknown field, scattered data | a PINN, against a classical inverse method |
    | PDE partly unknown | learn the missing terms: Lecture 24 |

    The first two rows are today's measurements (§4). The third is where PINNs are proposed for
    inverse problems (Raissi et al., 2019, recover unknown coefficients and hidden fields from scattered
    data), and where the classical alternatives (adjoint and regularized inversion) are the
    baseline to beat. The fourth is next lecture: sparse regression on data finds the equation
    itself.

    **Impact (accountability):** when a network replaces an analysis that a design code
    sanctions, who signs for the result? The engineer of record still does, so the engineer
    needs today's checks: a classical solution of the same problem, the error where the solution
    is steep, conservation read two ways, several seeds, and residual points everywhere a
    prediction is used.

    ## 7. Check yourself

    ### Exit question

    A consultant's report predicts an embankment's settlement with a PINN. Which two checks
    would you ask for before signing off, and why?
    """)
    return


@app.cell
def _(consolidation, mo):
    _s = consolidation["summary"]["forward"]["max_kpa"]
    mo.accordion(
        {
            "Answer": mo.md(f"""
        Any two of: the same problem solved by a classical method (finite differences, or the
        series fitted) and compared; the error where the solution is steep, at boundaries and
        corners; a conservation check read two ways (settlement from pore pressures and from
        water drained); several seeds, since one can be badly off ({_s["min"]:.1f} to {_s["max"]:.1f} kPa worst
        error across seeds here); residual points everywhere the prediction is used. Who is
        accountable: the engineer who signs, so the method's failure modes are theirs to check.
        """)
        }
    )
    return


@app.cell
def _(
    ADAM_LR,
    ADAM_STEPS,
    CV,
    DEPTHS,
    FULL,
    H,
    INTAKE,
    Iterator,
    LAYERS,
    LBFGS_ITERS,
    MONTHS,
    N_BC,
    N_DRAWS,
    N_IC,
    N_NO_IC,
    N_RES,
    N_SEEDS,
    PECLETS,
    RAR_ADD,
    RAR_POOL,
    REF,
    SIGMA,
    TARGET,
    TAU,
    THREADS,
    T_MAX,
    U0,
    WIDTH,
    checks_out,
    consolidation,
    fd,
    n_weights,
    near,
    river,
):
    numbers = {
        "settings": {
            "H": H, "cv": CV, "u0": U0, "sigma": SIGMA, "depths": list(DEPTHS),
            "months": list(MONTHS), "t_max": T_MAX, "target": TARGET, "draws": N_DRAWS,
            "seeds": N_SEEDS, "no_ic_seeds": N_NO_IC, "width": WIDTH, "layers": LAYERS, "adam_steps": ADAM_STEPS,
            "adam_lr": ADAM_LR, "lbfgs_iters": LBFGS_ITERS, "n_res": N_RES, "n_ic": N_IC,
            "n_bc": N_BC, "tau": TAU, "peclets": list(PECLETS), "rar_pool": RAR_POOL,
            "rar_add": RAR_ADD, "intake": INTAKE, "threads": THREADS, "parameters": n_weights,
        },
        "checks": checks_out,
        "fd": fd,
        "consolidation": consolidation,
        "river": river,
    }  # fmt: skip

    def leaf_paths(obj: object, path: str = "") -> Iterator[tuple[str, object]]:
        """Yield (path, value) for every number or None in a JSON value, as ``pairs`` names them.

        Args:
            obj: A JSON value (dicts, lists, numbers, None).
            path: The path of ``obj`` itself.

        Yields:
            (path, value) for each number or None, keys sorted.
        """
        if isinstance(obj, dict):
            for key in sorted(obj, key=str):
                yield from leaf_paths(obj[key], f"{path}/{key}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from leaf_paths(v, f"{path}[{i}]")
        elif isinstance(obj, int | float) or obj is None:
            yield path, obj

    def plausible(n: dict) -> None:
        """The slides' script's plausibility bounds, on this run's numbers."""
        c, site = n["consolidation"], n["consolidation"]["site"]
        assert abs(site["T90"] - 0.848) < 1e-3 and abs(site["t90"] - 0.848 * H**2 / CV) < 0.01
        assert 0 < site["degree_6_months"] < site["degree_exact"]["t90"] < TARGET + 1e-9
        assert 0 < site["u_3m_at_t90"] < U0
        assert all(0 < u < U0 + 5 * SIGMA for u in site["readings"]["u"])
        assert all(0 <= v < 1e-6 for v in n["checks"].values()), n["checks"]
        assert all(1.9 < o < 2.1 for o in n["fd"]["order"])
        assert all(0 < g["max_error_kpa"] < SIGMA for g in n["fd"]["grids"])
        assert 0 < n["fd"]["on_grid"]["max_kpa"] < SIGMA
        for f in c["forward"]:
            assert 0 < f["max_kpa"] < U0 and 0 < f["after_3_months_max_kpa"] < U0, f
            assert 0 < f["rel_l2"] < 1 and f["loss"] > 0, f
            assert all(0 <= v <= 1.05 for v in f["degree_field"].values()), f
            assert all(-1 < v < 2 for v in f["degree_drained"].values()), f
        for f in c["no_ic"]:
            assert f["max_abs_kpa"] < U0 and all(-1 < v < 2 for v in f["degree_field"]), f
        for key in ("series_fit", "pinn", "window"):
            assert all(CV / 2 < v < 2 * CV for v in c[key]["cv"]), key
        for key in ("series_fit", "pinn", "readings_only", "window"):
            assert all(t is None or 0 < t <= T_MAX for t in c[key]["t90"]), key
        only = c["readings_only"]
        assert all(-50 * U0 < v < U0 for v in only["min_u_kpa"]), only["min_u_kpa"]
        assert all(0 < v < 50 * U0 for v in only["max_u_kpa"]), only["max_u_kpa"]
        assert all(abs(v) < 51 for v in only["min_degree"] + only["max_degree"])
        r = n["river"]
        assert all(0 < v["max_error"] < 0.05 for v in r["fd"].values())
        for table in ("pinn", "refined", "random_added"):
            for v in r[table].values():
                assert all(0 < e < 2 for e in v["max_error"]) and all(
                    abs(m) < 1 for m in v["mass_error"]
                )
        assert (
            n["settings"]["parameters"]
            == 3 * WIDTH + (LAYERS - 1) * (WIDTH + 1) * WIDTH + WIDTH + 1
        )

    def pairs(mine: object, ref: object, path: str = ""):
        """Yield (path, mine, ref) for every number in both; timings and, reduced, summaries out."""
        if isinstance(mine, dict) and isinstance(ref, dict):
            for key in sorted(set(mine) & set(ref), key=str):
                skip = "seconds" in str(key) or key == "minutes"
                if not skip and not (key == "summary" and not FULL):
                    yield from pairs(mine[key], ref[key], f"{path}/{key}")
        elif isinstance(mine, list | tuple) and isinstance(ref, list):
            for i, (a, b) in enumerate(zip(mine, ref, strict=False)):  # reduced: the first runs
                yield from pairs(a, b, f"{path}[{i}]")
        elif isinstance(mine, int | float) or mine is None:
            yield path, mine, ref

    plausible(numbers)  # the slides' script's bounds, on this run's numbers
    if REF is None:
        print("numbers.json not found (build the slides' figures): comparison skipped")
    else:
        _all = list(pairs(numbers, REF))
        _none = [_p for _p, _a, _b in _all if (_a is None) != (_b is None)]
        _diffs = [(abs(_a - _b), _p) for _p, _a, _b in _all if _a is not None and _b is not None]
        _worst = max(_diffs) if _diffs else (0.0, "")
        _worst = _worst if _worst[0] > 0 else (0.0, "none: every number equal")
        _rel = max((abs(_a - _b) / max(abs(_b), 1e-300), _p) for _p, _a, _b in _all
                   if _a is not None and _b is not None and _b != 0)  # fmt: skip
        _rel = _rel if _rel[0] > 0 else (0.0, "none")
        print(f"compared {len(_all)} numbers with the slides' run (timings left out): largest "
              f"difference {_worst[0]:.2e} at {_worst[1]}; largest relative {_rel[0]:.2e} at "
              f"{_rel[1]}; missing on one side only: {_none}")  # fmt: skip
        if FULL:  # the slides setting: every number but the timings equals the slides' run
            assert not _none and all(near(_a, _b) for _p, _a, _b in _all if _a is not None), _worst
            # ... and every number of the slides' run was recomputed (the notebook adds its own)
            _seen = {_q for _q, *_ in _all}
            _missing = [_p for _p, _r in leaf_paths(REF) if "seconds" not in _p and _p not in _seen
                        and _p != "/minutes"]  # fmt: skip
            assert not _missing, _missing[:10]
    return (numbers,)


@app.cell
def _(T_START, fd32_gap, h_sweep, mo, numbers, river_runs, runs, time):
    _minutes = (time.perf_counter() - T_START) / 60
    _n = sum(len(_v) for _v in runs.values()) + len(river_runs)
    assert numbers["settings"]["parameters"] == 3297
    mo.md(f"""
    ## Gotchas

    - **The second derivative needs `create_graph=True`** on the first `torch.autograd.grad`,
      or the graph is gone before the second. And the inputs need `requires_grad=True`.
    - **tanh, not ReLU:** a ReLU network's second derivative is zero almost everywhere (§3).
    - **Scale the inputs and the output.** The network sees $(2\\zeta - 1, 2s - 1)$ on
      $[-1, 1]$ and returns $u/u_0$; the units live in $k = c_v T/H^2$ and in $u_0$.
    - **A loss that a trivial answer satisfies will be satisfied by it** (no start condition,
      §5): list every condition the solution needs and check that each is in the loss or built in.
    - **The physics holds only where the residual is scored.** Put residual points everywhere
      the network will be asked a question (§5).
    - **Score against a classical solver on the same grid,** and at the steep places (the
      corner here, the front in the river), not by one average.
    - **64-bit floats.** The check of §3 divides by $h^2 = 10^{{-8}}$: run in float32 (machine
      epsilon about $10^{{-7}}$), the same check's gap is {fd32_gap:.0e} instead of {h_sweep[1e-4]:.0e}.
    - **Fixed threads per process.** The order of floating-point sums can depend on the thread
      count; each job here sets its own threads (4, as in the slides' run), dtype, and seed in
      its own worker process, and §7 compares this run with the slides' number by number.
    - **Units.** `predict` returns kPa; convert in one place only, and check the order of
      magnitude of every quantity you quote (§4 checks the pore pressures this way).
    - **Time.** This run took {_minutes:.0f} minutes ({_n} networks).

    ## Going deeper (Forsyth)

    - §16.3.2–16.3.4: the Jacobians and the chain rule that backpropagation applies layer by
      layer. Automatic differentiation with respect to the inputs is the same product of
      Jacobians, ending at the input instead of the weights; the second derivative is a second
      pass through the graph the first one built. Forsyth's $\\theta^{{(l)}}$ are our weights; his
      $x$ is our $(\\zeta, s)$.
    - §16.4.4: why steps straight down the gradient are slow in a narrow valley, and why
      Newton's method, which would fix it, needs a matrix of second derivatives too large for
      big networks. L-BFGS builds a low-rank stand-in for that matrix from the last 50 steps
      (`history_size=50`), practical for our 3,297 weights.
    - §10.2.3 and §10.2.6: residuals, and the checks that tell whether a regression can be
      trusted. The series fit's misfit (§6) is that check, and a PINN's PDE residual is the same
      idea applied to the physics instead of the data.
    - §10.2.5: transforming variables. The series fit works on $\\log c_v$ (and the PINN on
      $\\log k$), so the estimate stays positive and the search is better scaled.

    **Exercises.** (1) Weight the start term by 10 and by 0.1 (`w_ic`) and rerun the forward
    problem for seeds 23–27: does the corner error shrink, and what happens to the conservation
    gap of §5? (2) At Pe 10,000 refinement had a lower error than as many random points on 3 of
    5 seeds. Run seeds 28–32 as well (`transport_job(10000, seed, mode)` for both modes): with
    10 paired seeds, on how many does it win, and how far apart are the medians?

    ## Things to remember

    - A PINN: a network, its PDE residual by autograd, a loss.
    - The physics holds only where the residual is scored.
    - Build in what you can: the drained top was exact.
    - Check conservation two ways.
    - Beat the classical solver, or use it.

    **Next:** SINDy and neural operators.
    """)
    return


if __name__ == "__main__":
    app.run()
