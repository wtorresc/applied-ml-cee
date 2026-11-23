"""Applied Machine Learning for CEE — Lecture 14: Neural Networks and Backpropagation.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l14_neural_networks.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_battledim.sh`` once (BattLeDIM, Vrachimis et al., 2022, CC BY 4.0).
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
    # Applied Machine Learning for CEE — Lecture 14: Neural Networks and Backpropagation

    **Slides:** *Applied Machine Learning for CEE*, Lecture 14 (`lectures/cee_l14_neural_networks`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §16.3 (layers, Jacobians, backpropagation) and §16.4 (training multilayer networks); Griewank & Walther (2008), *Evaluating Derivatives*, Ch. 3 (reverse mode).
    **Data:** BattLeDIM's L-Town EPANET model (Vrachimis et al., 2022, CC BY 4.0), run with WNTR (Klise et al., 2017). Fetch with `bash scripts/fetch_battledim.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides from this lecture's runs is recomputed here and checked with an
    `assert`. It runs the simulator 4,834 times (and up to 74 more to check the picks) and trains
    94 networks and 24 GPs in parallel processes (`L14_WORKERS`, default up to 16); with 4
    workers it took about 19 minutes on the course's machine, with 2.8 GB of RAM.

    **Your tasks** (from the slides): fit the attempt's slice with ramps by hand, write the
    autograd and match PyTorch, train the networks and compare them with Lecture 12's Gaussian
    process, then break them.
    """)
    return


@app.cell
def _():
    import contextlib
    import os
    import random
    import tempfile
    import warnings
    from dataclasses import dataclass
    from itertools import pairwise
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import torch
    import wntr
    from joblib import Parallel as JobParallel
    from joblib import delayed
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from threadpoolctl import threadpool_limits

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # One BLAS thread, as in the slides' figure script: the same sums in the same order.
    threadpool_limits(1)

    TODAY = (40.0, 50.0)  # PRV-1, PRV-2 settings in L-TOWN.inp, m
    P_SERVICE = 35 * 6894.757 / (1000.0 * 9.80665)  # 35 psi in metres of water, 24.6 m
    LO, HI = 25.0, 60.0  # the box the runs cover, m (Lecture 12's)
    BUDGETS = (16, 64, 256, 1024)  # runs per training set
    DESIGNS = range(3)  # scrambled Sobol sequences
    VAL_SEED, VAL_RUNS = 3, 256  # a fourth Sobol design: validation runs for the training length
    ENSEMBLE = 5  # networks per fit
    HIDDEN = (64, 64)
    STEPS, BATCH, LR = 20_000, 128, 3e-3  # Adam steps, rows per step, peak step size
    STEP_CHOICES = (2_500, 5_000, 10_000, 20_000, 40_000)  # lengths checked on validation runs
    ACCURATE = 1e-4  # EPANET accuracy that re-checks the picks (the file's is 0.01)
    BETAS, ADAM_EPS = (0.9, 0.999), 1e-8
    VERIFY_STEPS, FD_STEP = 1_000, 1e-6
    SLICE_S1, ATTEMPT_S2 = 36.0, (25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
    TRACE, EXTRAP_HI = (36.0, 47.0), 90.0
    CAND_STEP, OPT_WINDOW = 0.2, ((35.0, 37.2), (45.8, 48.0))
    ALPHA, CAL_RUNS, BREAK_BUDGET = 0.05, 256, 256
    GP_ELL_BOUNDS, GP_VAR_BOUNDS, GP_JITTER, GP_RESTARTS = (1e-2, 10.0), (1e-2, 1e2), 1e-6, 5
    # Worker processes (L14_WORKERS; default up to 16): every run and fit is seeded and computed
    # in its own process with one BLAS thread, so the numbers do not depend on the count.
    WORKERS = int(os.environ.get("L14_WORKERS", min(16, os.cpu_count() or 1)))
    LR_CHECK = (1e-3, 1e-2)  # peak steps either side of LR: a check, never a choice
    NEAR_LINE_M = 2.0  # "near a switch": within 2 m (along PRV-2) of a fitted shut-off line
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    return (
        ADAM_EPS,
        ALPHA,
        ATTEMPT_S2,
        BATCH,
        BETAS,
        BREAK_BUDGET,
        BUDGETS,
        CAL_RUNS,
        CAND_STEP,
        ConstantKernel,
        DESIGNS,
        ENSEMBLE,
        EXTRAP_HI,
        FD_STEP,
        GP_ELL_BOUNDS,
        GP_JITTER,
        GP_RESTARTS,
        GP_VAR_BOUNDS,
        GaussianProcessRegressor,
        HI,
        HIDDEN,
        JobParallel,
        LO,
        LR,
        LR_CHECK,
        Matern,
        OPT_WINDOW,
        PALETTE,
        NEAR_LINE_M,
        P_SERVICE,
        Path,
        SEED,
        SLICE_S1,
        STEPS,
        STEP_CHOICES,
        ACCURATE,
        TODAY,
        TRACE,
        VAL_RUNS,
        VAL_SEED,
        VERIFY_STEPS,
        WORKERS,
        contextlib,
        dataclass,
        delayed,
        np,
        nx,
        pairwise,
        pd,
        plt,
        qmc,
        tempfile,
        threadpool_limits,
        torch,
        warnings,
        wntr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–13

    On your own, no notes, two minutes:

    1. Lecture 6's logistic regression is $\sigma(\mathbf{w}^\top\mathbf{z})$. Drawn as a
       network, how many layers and units? Can its boundary bend?
    2. Past the box, Lecture 12's GP band on leakage widened: sd 0.77 m³/h at 60 m, 5.67 at
       90 m (asserted in Lecture 12's notebook). Why?
    3. Adam needs the gradient of every weight (Lecture 13). By finite differences, how many
       forward passes for 4,482 weights?

    **Today:** all 4,482 gradients from one backward pass.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. One unit and no hidden layer: its boundary is a plane in $\mathbf{z}$. A
                   hidden layer lets it bend.
                2. Far from every run, the kernel's correlation with the runs fades, so the
                   posterior variance climbs back toward the prior's. Networks have no such
                   mechanism (§5).
                3. Central differences take two forward passes per weight: 8,964.
                   Backpropagation takes one forward and one backward pass (§3).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Lecture 12 set L-Town's two pressure-reducing valves with 15 simulator runs. A control room
    wants more: for **any** pair of settings in the box (25–60 m), leakage and the lowest
    customer pressure, instantly. Runs made in a batch (16 to 1,024) train a surrogate.
    **Decide:** which surrogate, how many runs, and where to trust it. The constraint is
    Lecture 12's: every customer of Areas A and B keeps 35 psi (24.6 m).

    The simulator is Lecture 12's, unchanged: BattLeDIM's L-Town, the peak day, background
    leakage in Area A calibrated to BattLeDIM's 2018 mean leak flow. One run takes under half a
    second on the course's machine (0.43 s in the slides' run), so a 1-m map of the whole box (1,296 runs) checks every surrogate.
    """)
    return


@app.cell
def _(P_SERVICE, Path, TODAY, contextlib, dataclass, np, nx, pd, tempfile, wntr):
    def find_raw() -> Path:
        """Locate ``data/raw/battledim`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_battledim.sh``).
        """
        rel = Path("data") / "raw" / "battledim"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel / "L-TOWN.inp").exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_battledim.sh`")

    RAW = find_raw()
    STEP_S, DAY_S = 300, 86_400

    @dataclass(frozen=True)
    class Run:
        """One simulated day: Area A leakage (m3/h), lowest customer pressure in A and B (m)."""

        leak: float
        pmin: float
        node: str
        closed_steps: tuple  # reported steps at which PRV-1, PRV-2 pass no water

    def mean_leak_2018() -> float:
        """Mean total leak flow of BattLeDIM's 2018 leaks, m3/h."""
        d = pd.read_csv(RAW / "2018_Leakages.csv", sep=";", decimal=",", index_col=0)
        assert d.shape == (105_120, 14), d.shape
        return float(d.sum(axis=1).mean())

    class Town:
        """L-Town on its peak day with calibrated background leakage in Area A (Lecture 12)."""

        def __init__(self, accuracy: float | None = None) -> None:
            """Load the model, find the zones and the peak day, calibrate the leakage.

            ``accuracy`` overrides EPANET's hydraulic accuracy (the file's is 0.01); the
            leakage is calibrated at the same accuracy.
            """
            self.wn = wntr.network.WaterNetworkModel(str(RAW / "L-TOWN.inp"))
            graph = nx.Graph()
            graph.add_nodes_from(self.wn.junction_name_list)
            graph.add_edges_from((p.start_node_name, p.end_node_name) for _, p in self.wn.pipes())
            parts = sorted(nx.connected_components(graph), key=len, reverse=True)
            junctions = set(self.wn.junction_name_list)
            self.area_a, self.area_c, self.area_b = (sorted(set(p) & junctions) for p in parts[:3])
            self.customers = [
                n
                for n in self.area_a + self.area_b
                if any(d.base_value > 0 for d in self.wn.get_node(n).demand_timeseries_list)
            ]
            self.day = self._peak_day()
            opts = self.wn.options
            opts.time.duration = DAY_S
            opts.time.pattern_start = self.day * DAY_S
            opts.time.report_timestep = STEP_S
            opts.hydraulic.emitter_exponent = 1.0
            if accuracy is not None:
                opts.hydraulic.accuracy = accuracy
            index = {n: i for i, n in enumerate(self.area_a)}
            self.length = np.zeros(len(self.area_a))
            for _, p in self.wn.pipes():
                for end in (p.start_node_name, p.end_node_name):
                    if end in index:
                        self.length[index[end]] += p.length / 2
            self.coef = np.zeros(len(self.area_a))
            self.target = mean_leak_2018()
            self._calibrate()

        def _peak_day(self) -> int:
            """Day of the one-week demand pattern with the largest total demand."""
            steps = DAY_S // STEP_S
            total = np.zeros(7 * steps)
            for _, j in self.wn.junctions():
                for d in j.demand_timeseries_list:
                    mult = np.asarray(self.wn.get_pattern(d.pattern_name).multipliers)
                    total += d.base_value * np.resize(mult, total.size)
            return int(total.reshape(7, steps).sum(axis=1).argmax())

        def _simulate(self, s1: float, s2: float):
            """Run EPANET with the valve settings and the emitter coefficients."""
            self.wn.get_link("PRV-1").initial_setting = float(s1)
            self.wn.get_link("PRV-2").initial_setting = float(s2)
            for name, c in zip(self.area_a, self.coef, strict=True):
                self.wn.get_node(name).emitter_coefficient = float(c) if c > 0 else None
            sim = wntr.sim.EpanetSimulator(self.wn)
            # Each run in its own temporary directory, removed when the run ends (worker
            # processes exit without cleaning up after themselves). EPANET also writes a scratch
            # file to the working directory, so the run works inside it.
            with tempfile.TemporaryDirectory(prefix="ltown-") as tmp, contextlib.chdir(tmp):
                res = sim.run_sim(file_prefix=str(Path(tmp) / "ltown"), convergence_error=True)
            # A run EPANET could not balance raises, rather than returning unconverged pressures.
            failed = [
                w
                for w in sim.enData.errcodelist
                if "hydraulically unbalanced" in w or "hydraulically unstable" in w
            ]
            if failed:
                raise RuntimeError(f"EPANET did not solve ({s1}, {s2}): {failed}")
            return res

        def _leak(self, res) -> np.ndarray:
            """Area A leakage at each step, m3/s: sum of C_i p_i."""
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            return (p * self.coef).sum(axis=1)

        def _calibrate(self) -> None:
            """Scale the coefficients so today's settings leak the 2018 mean leak flow.

            The last step checks the formula's leakage against the emitter flow EPANET itself
            reports (the demand with emitters minus the demand without).

            Raises:
                RuntimeError: If the steps do not converge, or the two leakages disagree.
            """
            res = self._simulate(*TODAY)
            demand = res.node["demand"][self.area_a].to_numpy()  # no emitters yet
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            weight = self.length / self.length.sum()
            total = self.target / 3600 / float((p * weight).sum(axis=1).mean())
            for _ in range(20):
                self.coef = total * weight
                res = self._simulate(*TODAY)
                leak = float(self._leak(res).mean()) * 3600
                if abs(leak / self.target - 1) < 1e-6:
                    break
                total *= self.target / leak
            else:
                raise RuntimeError("leakage calibration did not converge")
            reported = (res.node["demand"][self.area_a].to_numpy() - demand).sum(axis=1).mean()
            if abs(reported * 3600 - leak) >= 1e-3 * self.target:
                raise RuntimeError(f"EPANET reports {reported * 3600} m3/h of leakage, not {leak}")

        def run(self, s1: float, s2: float) -> Run:
            """Simulate the peak day at PRV-1 = s1, PRV-2 = s2 (m)."""
            res = self._simulate(round(float(s1), 6), round(float(s2), 6))
            p = res.node["pressure"][self.customers]
            low = p.min()
            status = res.link["status"][["PRV-1", "PRV-2"]].to_numpy()
            return Run(
                leak=float(self._leak(res).mean()) * 3600,
                pmin=float(low.min()),
                node=str(low.idxmin()),
                closed_steps=tuple(int(v) for v in (status == 0).sum(axis=0)),
            )

        def customer_pressure(self, s1: float, s2: float):
            """Each customer's lowest pressure over the day (m), indexed by junction."""
            res = self._simulate(round(float(s1), 6), round(float(s2), 6))
            return res.node["pressure"][self.customers].min()

    town = Town()
    _today = town.run(*TODAY)
    assert (len(town.area_a), len(town.area_b), len(town.area_c), town.day) == (657, 31, 92, 3)
    assert (round(_today.leak, 2), round(_today.pmin, 2), round(P_SERVICE, 1)) == (
        25.52,
        28.19,
        24.6,
    )
    return Town, town


@app.cell
def _(JobParallel, Town, WORKERS, delayed, np, threadpool_limits):
    def _sim_batch(pairs, accuracy):
        """One worker process: build its own calibrated model, run a batch of settings.

        Processes, not threads: WNTR's EPANET library is not safe to call from several threads.
        """
        threadpool_limits(1)
        t = Town(accuracy)
        return [t.run(*p) for p in pairs]

    def simulate(pairs, accuracy=None) -> list:
        """Run independent settings in ``WORKERS`` processes; results in the given order."""
        pairs = [(float(a), float(b)) for a, b in pairs]
        batches = [pairs[i::WORKERS] for i in range(WORKERS)]
        done = JobParallel(n_jobs=WORKERS)(delayed(_sim_batch)(b, accuracy) for b in batches if b)
        out = [None] * len(pairs)
        for i, batch in enumerate(done):
            for j, r in enumerate(batch):
                out[i + j * WORKERS] = r
        return out

    def outputs(runs) -> np.ndarray:
        """Leakage and lowest customer pressure of each run, shape ``(n, 2)``."""
        return np.array([(r.leak, r.pmin) for r in runs])

    return outputs, simulate


@app.cell
def _(
    BUDGETS,
    CAND_STEP,
    DESIGNS,
    EXTRAP_HI,
    HI,
    LO,
    OPT_WINDOW,
    SLICE_S1,
    VAL_RUNS,
    VAL_SEED,
    np,
    outputs,
    qmc,
    simulate,
):
    def lattice(lo1, hi1, lo2, hi2) -> np.ndarray:
        """Candidate settings every ``CAND_STEP`` m, shape ``(n, 2)``."""
        a = np.round(np.arange(lo1, hi1 + 1e-9, CAND_STEP), 6)
        b = np.round(np.arange(lo2, hi2 + 1e-9, CAND_STEP), 6)
        return np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)

    _g = np.arange(LO, HI + 1e-9, 1.0)
    grid = np.stack(np.meshgrid(_g, _g, indexing="ij"), axis=-1).reshape(-1, 2)  # (1296, 2)
    design = (
        np.stack([qmc.Sobol(2, scramble=True, seed=_s).random(max(BUDGETS)) for _s in DESIGNS])
        * (HI - LO)
        + LO
    )  # (3, 1024, 2)
    slice_s2 = np.arange(LO, EXTRAP_HI + 1e-9, 1.0)
    window = lattice(*OPT_WINDOW[0], *OPT_WINDOW[1])
    # Validation runs that choose the training length: a fourth Sobol design, never the map
    val = qmc.Sobol(2, scramble=True, seed=VAL_SEED).random(VAL_RUNS) * (HI - LO) + LO
    _parts = [grid, design.reshape(-1, 2), np.c_[np.full_like(slice_s2, SLICE_S1), slice_s2]]
    _parts += [window, val]
    _runs = simulate(np.concatenate(_parts))
    _y = outputs(_runs)
    _i = np.cumsum([0, *(len(q) for q in _parts)])
    grid_y, grid_runs = _y[: _i[1]], _runs[: _i[1]]
    design_y = _y[_i[1] : _i[2]].reshape(design.shape)
    slice_y, window_y, val_y = _y[_i[2] : _i[3]], _y[_i[3] : _i[4]], _y[_i[4] :]
    assert grid.shape == (1296, 2) and design_y.shape == (3, 1024, 2) and len(window) == 144
    assert val_y.shape == (VAL_RUNS, 2)
    return (
        design,
        design_y,
        grid,
        grid_runs,
        grid_y,
        lattice,
        slice_s2,
        slice_y,
        val,
        val_y,
        window,
        window_y,
    )


@app.cell
def _(HI, LO, grid, grid_runs, grid_y, np, slice_y, town):
    # The valves' regimes, from the simulator's valve states: closed all day (289 reported steps).
    _closed = np.array([_r.closed_steps for _r in grid_runs])
    _full = _closed.max()
    _regimes = {
        "prv2_shut": int((_closed[:, 1] == _full).sum()),
        "prv1_shut": int((_closed[:, 0] == _full).sum()),
    }
    assert _full == 289 and _regimes == {"prv2_shut": 538, "prv1_shut": 105}
    assert len(grid) - sum(_regimes.values()) == 653

    def _fit_line(valve, pick):
        pts = []
        for _s1 in np.unique(grid[:, 0]):
            _row = grid[:, 0] == _s1
            _shut = grid[_row][_closed[_row, valve] == _full, 1]
            if len(_shut):
                pts.append((_s1, pick(_shut)))
        _p = np.array(pts)
        slope, icpt = np.polyfit(_p[:, 0], _p[:, 1], 1)
        centre = 0.5 * (LO + HI)
        return {"slope": slope, "intercept": icpt, "offset": slope * centre + icpt - centre}

    lines = {"prv2": _fit_line(1, np.max), "prv1": _fit_line(0, np.min)}
    assert (round(lines["prv2"]["offset"], 1), round(lines["prv2"]["slope"], 2)) == (-3.6, 0.94)
    assert (round(lines["prv1"]["offset"], 1), round(lines["prv1"]["slope"], 2)) == (22.0, 1.0)

    # The lowest customer, and the ceiling PRV-3 sets in Area B
    _nodes = np.array([_r.node for _r in grid_runs])
    _u, _c = np.unique(_nodes, return_counts=True)
    assert dict(zip(_u.tolist(), _c.tolist(), strict=True)) == {"n206": 265, "n50": 1031}
    cap = float(grid_y[:, 1].max())
    assert round(cap, 2) == 33.19 and set(_nodes[grid_y[:, 1] == cap]) <= set(town.area_b)
    assert slice_y[:, 1].max() == cap
    return cap, lines


@app.cell
def _(HI, LO, P_SERVICE, PALETTE, grid_y, lines, np, plt):
    _g = np.arange(LO, HI + 1e-9, 1.0)
    _P = grid_y[:, 1].reshape(36, 36).T  # rows PRV-2, columns PRV-1
    _fig, _ax = plt.subplots(figsize=(6.4, 4.2))
    _cs = _ax.contour(_g, _g, _P, levels=np.arange(8, 33, 4), colors="#595959", linewidths=1.0)
    _ax.clabel(_cs, fmt="%d", fontsize=9)
    _ax.contour(_g, _g, _P, levels=[P_SERVICE], colors="#1A1A1A", linewidths=2.6)
    _s1 = np.linspace(LO, HI, 50)
    for _k, _color, _ls, _lab in (
        ("prv2", PALETTE[0], "--", "PRV-2 shut below"),
        ("prv1", PALETTE[1], "-.", "PRV-1 shut above"),
    ):
        _s2 = lines[_k]["slope"] * _s1 + lines[_k]["intercept"]
        _keep = (_s2 >= LO) & (_s2 <= HI)
        _ax.plot(_s1[_keep], _s2[_keep], color=_color, ls=_ls, lw=2.4, label=_lab)
    _ax.plot([], [], color="#1A1A1A", lw=2.6, label="24.6 m (35 psi)")
    _ax.set(
        xlim=(LO, HI),
        ylim=(LO, HI),
        xlabel="PRV-1 setting (m)",
        ylabel="PRV-2 setting (m)",
        aspect="equal",
    )
    _ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _ax.set_title("Lowest customer pressure (m), 1,296 runs")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    A valve passes no water when the other valve's water reaches its outlet with more head than
    its own setting: PRV-2 below the dashed line, PRV-1 above the dash-dot line. The contours
    bend at those lines. The lowest customer is n50 at 1,031 settings and n206 (Area B) at the
    other 265, where PRV-3 holds the pressure at 33.19 m: the surface's ceiling.

    ### Your turn: draw it with ramps

    PRV-1 held at 36 m. The lowest customer pressure as PRV-2 rises:

    | PRV-2 setting (m) | 25 | 30 | 35 | 40 | 45 | 50 |
    |:------------------|---:|---:|---:|---:|---:|---:|
    | Lowest pressure (m) | 14.89 | 14.89 | 16.12 | 20.11 | 23.69 | 25.73 |

    Sketch it. Where does it bend, and why there? Fit $a+b\max(0,s_2-c)$. What would one more
    ramp fix?
    """)
    return


@app.cell
def _(ATTEMPT_S2, HI, np, slice_s2, slice_y):
    _inside = slice_s2 <= HI
    _s2, _p = slice_s2[_inside], slice_y[_inside, 1]
    assert [round(float(_p[_s2 == _v][0]), 2) for _v in ATTEMPT_S2] == [
        14.89,
        14.89,
        16.12,
        20.11,
        23.69,
        25.73,
    ]
    # the best two ramps over the whole slice: least squares for each pair of kinks, and the pair
    # whose fit misses least at its worst point is kept; the straight line is least squares too
    _best = None
    for _c1 in _s2[1:-1]:
        for _c2 in _s2[(_s2 > _c1) & (_s2 < _s2[-1])]:
            _A = np.c_[np.ones_like(_s2), np.maximum(0, _s2 - _c1), np.maximum(0, _s2 - _c2)]
            _coef, *_ = np.linalg.lstsq(_A, _p, rcond=None)
            _worst = float(np.abs(_A @ _coef - _p).max())
            if _best is None or _worst < _best[0]:
                _best = (_worst, _c1, _c2, _coef)
    _worst, _c1, _c2, _coef = _best
    assert (_c1, _c2) == (33.0, 43.0) and [round(float(_x), 2) for _x in _coef] == [
        14.76,
        0.81,
        -0.46,
    ]
    assert round(_worst, 2) == 0.64
    _line, *_ = np.linalg.lstsq(np.c_[np.ones_like(_s2), _s2], _p, rcond=None)
    assert round(float(np.abs(np.c_[np.ones_like(_s2), _s2] @ _line - _p).max()), 2) == 1.90
    # the hand fit in the answer: a = 14.89, b = 0.8, c = 33, evaluated at 50 m
    assert round(14.89 + 0.8 * (50 - 33), 2) == 28.49 and round(20.11 - 16.12, 2) == 3.99
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Flat at 14.89 m until about 33 m: below that PRV-2 passes no water (the dashed
                line), so its setting changes nothing. Then about 0.8 m per metre (from 35 to 40 m
                the pressure rises 3.99 m). With $a$ = 14.89, $b$ = 0.8, $c$ = 33 the ramp gives
                28.49 m at 50 m against 25.73: a second ramp, with a negative slope from about
                43 m, bends it back. Least squares for each pair of kinks over the whole slice,
                keeping the pair that misses least at its worst point: kinks at 33 and 43 m,
                $a$ = 14.76, slopes 0.81 and −0.46, at most 0.64 m off; the least-squares
                straight line misses by 1.90 m. Two ramps are a hidden layer of two units.
                """
            )
        }
    )
    return


@app.cell
def _(ATTEMPT_S2, HI, PALETTE, np, plt, slice_s2, slice_y):
    # The attempt, drawn: the six runs, one ramp at 33 m, and the best two ramps (33 and 43 m)
    _inside = slice_s2 <= HI
    _s2, _p = slice_s2[_inside], slice_y[_inside, 1]
    _A = np.c_[np.ones_like(_s2), np.maximum(0, _s2 - 33), np.maximum(0, _s2 - 43)]
    _coef = np.linalg.lstsq(_A, _p, rcond=None)[0]
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _ax.plot(_s2, _p, color="#595959", lw=1.2, label="simulator, every metre")
    _six = np.isin(_s2, ATTEMPT_S2)
    _ax.plot(_s2[_six], _p[_six], "o", color="#1A1A1A", ms=8, label="the six runs")
    _ax.plot(_s2, _A[:, :2] @ _coef[:2], ":", color="#595959", label="one ramp")
    _ax.plot(_s2, _A @ _coef, color=PALETTE[1], lw=2.4, label="two ramps")
    _ax.set(xlabel="PRV-2 setting (m); PRV-1 at 36 m", ylabel="Lowest pressure (m)", ylim=(12, 34))
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with the **network** as the focus step: the encoder standardizes the two
    settings; the network (2 → 64 ReLU units → 64 → 2) predicts leakage and the lowest pressure;
    the loss compares them with the simulator's runs; **backpropagation** sends the gradient of all
    4,482 weights back in one pass, and Adam (Lecture 13) steps. §4 traces one run, (36, 47) m,
    through a trained network.

    ## 3. Derive

    **Assumption:** the runs pin every ramp where the network will be used.

    **A hidden unit is a ramp.** $h=\max(0,\mathbf{w}^\top\mathbf{z}+b)$ is zero on one side of
    the line $\mathbf{w}^\top\mathbf{z}+b=0$ and rises on the other. A layer sums ramps,
    $\hat y=\sum_k v_kh_k+c$: a surface folded along lines; a second layer folds the folds.
    Networks like this can approximate any continuous function on a box (Cybenko, 1989; Leshno et
    al., 1993): a statement about existence, not about training from 16 runs, and nothing about
    settings outside the runs.

    **Backpropagation.** Forward, with $Z$ the $n\times2$ standardized settings:
    $A_1=ZW_1+\mathbf{b}_1$, $H_1=\max(0,A_1)$, $A_2=H_1W_2+\mathbf{b}_2$, $H_2=\max(0,A_2)$,
    $\hat Y=H_2W_3+\mathbf{b}_3$, and $L=\frac{1}{nm}\sum(\hat Y-Y)^2$ with $m=2$ outputs. Backward:

    $$G=\frac{\partial L}{\partial\hat Y}=\frac{2(\hat Y-Y)}{nm},\quad\frac{\partial L}{\partial W_3}=H_2^\top G,\quad\frac{\partial L}{\partial\mathbf{b}_3}=\mathbf{1}^\top G,\quad\frac{\partial L}{\partial H_2}=GW_3^\top,\quad\frac{\partial L}{\partial A_2}=\frac{\partial L}{\partial H_2}\odot[A_2>0],$$

    and so on down. Each operation needs only its own local derivative applied to the gradient
    arriving from above (a vector–Jacobian product); the forward values it needs were stored on the
    way up. One output and many weights make reverse mode cheap: every gradient for about three
    forward passes' work (each product's backward step is two products of the same size).

    **Start.** Random weights and biases, uniform in $\pm1/\sqrt{\text{fan-in}}$ (PyTorch's
    default for `nn.Linear`); zeros give zero gradients, equal weights stay copies. Standardize
    the settings on the training runs, and the outputs too (here that matters little, §5).

    ## 4. Build and verify
    """)
    return


@app.cell
def _(np):
    class Var:
        """A value in the computation graph, its gradient, and how to send gradients back."""

        def __init__(self, value, parents=(), back=None):
            self.value = np.asarray(value, dtype=np.float64)
            self.grad = np.zeros_like(self.value)
            self.parents, self.back = parents, back

    def matmul(a, b):  # g @ b.T flows to a; a.T @ g flows to b
        return Var(a.value @ b.value, (a, b), lambda g: (g @ b.value.T, a.value.T @ g))

    def add(a, b):  # b is a bias row broadcast over the batch: its gradient sums the rows
        return Var(a.value + b.value, (a, b), lambda g: (g, g.sum(axis=0)))

    def relu(a):  # the gradient passes where the unit is on
        return Var(np.maximum(a.value, 0.0), (a,), lambda g: (g * (a.value > 0),))

    def mse(a, y):  # mean squared error against fixed targets of the same shape (no broadcasting)
        assert a.value.shape == np.shape(y), (a.value.shape, np.shape(y))
        r = a.value - y
        return Var((r**2).mean(), (a,), lambda g: (g * 2.0 * r / r.size,))

    def topo_order(out):
        """Every Var that ``out`` depends on, each after its parents (depth-first)."""
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)

        visit(out)
        return order

    def backward(out):
        out.grad = np.ones_like(out.value)
        for v in reversed(topo_order(out)):  # users before inputs
            if v.back:
                for p, g in zip(v.parents, v.back(v.grad), strict=True):
                    p.grad = p.grad + g

    return Var, add, backward, matmul, mse, relu


@app.cell
def _(
    ADAM_EPS, BATCH, BETAS, HIDDEN, LR, STEPS, Var, add, backward, matmul, mse, np, pairwise, relu
):
    def init_params(seed, sizes, scheme="uniform"):
        """``[W1, b1, W2, b2, ...]``; weights ``(fan_in, fan_out)``.

        ``uniform``: weights and biases U(+-1/sqrt(fan-in)), PyTorch's default for nn.Linear;
        ``he0``: weights N(0, 2/fan-in), biases 0; ``zero``; ``const``: weights 0.1, biases 0.
        """
        rng = np.random.default_rng(seed)
        params = []
        for fan_in, fan_out in pairwise(sizes):
            bound, b = 1.0 / np.sqrt(fan_in), np.zeros(fan_out)
            if scheme == "uniform":
                w = rng.uniform(-bound, bound, (fan_in, fan_out))
                b = rng.uniform(-bound, bound, fan_out)
            elif scheme == "he0":
                w = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            elif scheme == "zero":
                w = np.zeros((fan_in, fan_out))
            elif scheme == "const":
                w = np.full((fan_in, fan_out), 0.1)
            else:
                raise ValueError(f"unknown initialization {scheme!r}")
            params += [w, b]
        return params

    def forward(params, Z):
        """ReLU hidden layers, a linear output layer; ``params`` are Vars."""
        h = Var(Z)
        for i in range(0, len(params) - 2, 2):
            h = relu(add(matmul(h, params[i]), params[i + 1]))
        return add(matmul(h, params[-2]), params[-1])

    def predict(params, Z):
        """Forward pass only, shape ``(n, outputs)``."""
        return forward([Var(p) for p in params], Z).value

    def batches(n, seed):
        """Rows for each step: all of them if n <= BATCH, else shuffled batches of BATCH."""
        if n <= BATCH:
            while True:
                yield slice(None)
        assert n % BATCH == 0
        rng = np.random.default_rng((seed, 1))
        while True:
            yield from rng.permutation(n).reshape(-1, BATCH)

    def train(Z, Y, seed, scheme="uniform", steps=STEPS, lr=LR, *, probe=None):
        """Adam on minibatches, step decayed from ``lr`` to 0 by a cosine; returns (params, loss).

        ``probe(t, params)`` is called every 250 steps, at steps 10 and 50, and after the last
        (training curves and the snapshots of the training sheet).
        """
        params = init_params(seed, (Z.shape[1], *HIDDEN, Y.shape[1]), scheme)
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2 = BETAS
        rows = batches(len(Z), seed)
        for t in range(1, steps + 1):
            eta = lr * 0.5 * (1.0 + np.cos(np.pi * (t - 1) / steps))
            r = next(rows)
            leaves = [Var(p) for p in params]
            backward(mse(forward(leaves, Z[r]), Y[r]))
            for i, w in enumerate(leaves):
                m[i] = b1 * m[i] + (1 - b1) * w.grad
                v[i] = b2 * v[i] + (1 - b2) * w.grad**2
                step = (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + ADAM_EPS)
                params[i] = params[i] - eta * step
            if probe is not None and (t % 250 == 0 or t == steps or t in (10, 50)):
                probe(t, params)
        return params, float(((predict(params, Z) - Y) ** 2).mean())

    class Scaler:
        """Standardize columns with the training rows' mean and standard deviation."""

        def __init__(self, mean, sd):
            self.mean, self.sd = np.asarray(mean, float), np.asarray(sd, float)

        @classmethod
        def fit(cls, A):
            """Fit on the training rows; a constant column cannot be standardized."""
            sd = A.std(axis=0)
            if not (sd > 0).all():
                raise ValueError(f"a constant column cannot be standardized: sd {sd}")
            return cls(A.mean(axis=0), sd)

        def __call__(self, A):
            return (A - self.mean) / self.sd

        def inverse(self, A):
            return A * self.sd + self.mean

    return Scaler, forward, init_params, predict, train


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    The slide shows `Var`, `matmul`, and `backward` from the cell above. Before running the next
    cell: how many digits will our gradients share with PyTorch's autograd? With central finite
    differences? After 1,000 Adam steps, will our weights still match `torch.optim.Adam`'s?
    """)
    return


@app.cell
def _(
    ADAM_EPS,
    BETAS,
    FD_STEP,
    HIDDEN,
    LR,
    SEED,
    Scaler,
    VERIFY_STEPS,
    Var,
    backward,
    design,
    design_y,
    forward,
    init_params,
    mse,
    np,
    torch,
    train,
):
    # 16 runs of design 0 (the same runs the slides' check uses), 4,482 weights, float64
    _S, _Y = design[0, :16], design_y[0, :16]
    Z16, T16 = Scaler.fit(_S)(_S), Scaler.fit(_Y)(_Y)
    _params16 = init_params(SEED, (2, *HIDDEN, 2))
    assert sum(p.size for p in _params16) == 4482 == 2 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
    _leaves = [Var(p) for p in _params16]
    backward(mse(forward(_leaves, Z16), T16))
    _ours = [w.grad for w in _leaves]

    def _torch_net(tp, Z):
        """The same network in PyTorch, on a list of tensors."""
        h = torch.tensor(Z)
        for i in range(0, len(tp) - 2, 2):
            h = torch.relu(h @ tp[i] + tp[i + 1])
        return h @ tp[-2] + tp[-1]

    _tp = [torch.tensor(p, requires_grad=True) for p in _params16]
    ((_torch_net(_tp, Z16) - torch.tensor(T16)) ** 2).mean().backward()
    _grad_diff = max(
        float(np.abs(g - t.grad.numpy()).max()) for g, t in zip(_ours, _tp, strict=True)
    )

    def _loss(ps):
        return float(mse(forward([Var(p) for p in ps], Z16), T16).value)

    _fd_err = 0.0
    for _i, _p in enumerate(_params16):
        for _idx in np.ndindex(_p.shape):
            _up = [q.copy() for q in _params16]
            _dn = [q.copy() for q in _params16]
            _up[_i][_idx] += FD_STEP
            _dn[_i][_idx] -= FD_STEP
            _fd_err = max(_fd_err, abs((_loss(_up) - _loss(_dn)) / (2 * FD_STEP) - _ours[_i][_idx]))

    def torch_adam(steps):
        """torch.optim.Adam from the same start, with the same cosine-decayed steps."""
        tp = [torch.tensor(p, requires_grad=True) for p in _params16]
        opt = torch.optim.Adam(tp, lr=LR, betas=BETAS, eps=ADAM_EPS)
        for t in range(1, steps + 1):
            for group in opt.param_groups:
                group["lr"] = LR * 0.5 * (1.0 + np.cos(np.pi * (t - 1) / steps))
            opt.zero_grad()
            ((_torch_net(tp, Z16) - torch.tensor(T16)) ** 2).mean().backward()
            opt.step()
        return [t.detach().numpy() for t in tp]

    _trained, _ = train(Z16, T16, SEED, steps=VERIFY_STEPS)
    _adam_diff = max(
        float(np.abs(a - b).max()) for a, b in zip(_trained, torch_adam(VERIFY_STEPS), strict=True)
    )
    assert f"{_grad_diff:.0e}" == "6e-17" and f"{_fd_err:.0e}" == "2e-10"
    assert round(max(float(np.abs(g).max()) for g in _ours), 2) == 0.23
    assert f"{_adam_diff:.0e}" == "2e-14"
    print(
        f"gradients vs PyTorch {_grad_diff:.1e}; vs finite differences {_fd_err:.1e}; Adam, {VERIFY_STEPS} steps, {_adam_diff:.1e}"
    )
    return T16, Z16, torch_adam


@app.cell
def _(mo):
    mo.md(r"""
    The gradients agree with PyTorch's to the last bit or two: it computes the same products in the
    same order. Finite differences take 8,964 forward passes and agree to $2\times10^{-10}$, the
    error of the difference quotient itself. Our Adam and PyTorch's take the same steps.

    ### Train the surrogates

    Five networks (seeds 0–4) and Lecture 12's GP (scikit-learn, Matérn 5/2 with one lengthscale
    per valve, settings scaled to the unit box, five restarts of the evidence search) on the first
    16, 64, 256, and 1,024 runs of three Sobol designs; plus five networks per bad start (§5), and
    network 0 on 1,024 runs trained to each length from 2,500 to 40,000 steps, and at peak steps
    0.001 and 0.01 either side of 0.003, all checked on the validation runs. Every fit runs in its
    own process with one BLAS thread, as in the slides' script.
    """)
    return


@app.cell
def _(
    BREAK_BUDGET,
    BUDGETS,
    ConstantKernel,
    DESIGNS,
    ENSEMBLE,
    GP_ELL_BOUNDS,
    GP_JITTER,
    GP_RESTARTS,
    GP_VAR_BOUNDS,
    GaussianProcessRegressor,
    HI,
    HIDDEN,
    JobParallel,
    LO,
    LR,
    LR_CHECK,
    Matern,
    SEED,
    STEP_CHOICES,
    Scaler,
    WORKERS,
    delayed,
    STEPS,
    design,
    design_y,
    init_params,
    np,
    predict,
    threadpool_limits,
    train,
    val,
    val_y,
    warnings,
):
    def unit_box(S):
        """Valve settings scaled to the unit box, as in Lecture 12."""
        return (S - LO) / (HI - LO)

    class Networks:
        """An ensemble on the same runs, with its input and output scaling."""

        def __init__(self, members, xs, ys):
            self.members, self.xs, self.ys = members, xs, ys

        def each(self, S):
            return np.stack([self.ys.inverse(predict(p, self.xs(S))) for p in self.members])

        def __call__(self, S):
            return self.each(S).mean(axis=0)

    class GPs:
        """One GP per output, and whether a hyperparameter sits at one of Lecture 12's bounds."""

        def __init__(self, fitted):
            self.models = [g for g, _ in fitted]
            self.at_bound = [b for _, b in fitted]

        def __call__(self, S, sd=False):
            out = [g.predict(unit_box(S), return_std=sd) for g in self.models]
            if not sd:
                return np.stack(out, axis=1)
            return np.stack([m for m, _ in out], axis=1), np.stack([s for _, s in out], axis=1)

    def _job(kind, args):
        threadpool_limits(1)
        if kind == "net":
            return train(*args)[0]
        if kind == "curve":  # (step, training RMS, validation RMS) of the lowest pressure
            Z, Y, seed, steps, Zp, Yp = args
            history = []
            snaps = {0: init_params(seed, (2, *HIDDEN, 2))}  # the weights at steps 0, 10, 50, last

            def probe(t, params):
                if t in (10, 50, steps):
                    snaps[t] = [p.copy() for p in params]
                if t % 250 and t != steps:
                    return
                tr = predict(params, Z)[:, 1] - Y[:, 1]
                te = predict(params, Zp)[:, 1] - Yp[:, 1]
                history.append((t, np.sqrt((tr**2).mean()), np.sqrt((te**2).mean())))

            train(Z, Y, seed, steps=steps, probe=probe)
            return history, snaps
        X, y = args
        kernel = ConstantKernel(1.0, GP_VAR_BOUNDS) * Matern(np.ones(2), GP_ELL_BOUNDS, nu=2.5)
        gp = GaussianProcessRegressor(
            kernel,
            alpha=GP_JITTER,
            normalize_y=True,
            n_restarts_optimizer=GP_RESTARTS,
            random_state=SEED,
        )
        # The evidence may prefer a hyperparameter at Lecture 12's bounds; scikit-learn's warning
        # about it becomes a flag we count. Every other warning is issued again.
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            gp.fit(unit_box(X), y)
        for w in _caught:
            if not str(w.message).startswith("The optimal value found"):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        at_bound = bool(np.any(np.isclose(gp.kernel_.theta[:, None], gp.kernel_.bounds, atol=1e-2)))
        return gp, at_bound

    _keys = [(_d, _n) for _n in sorted(BUDGETS, reverse=True) for _d in DESIGNS]
    _jobs, _meta = [], []
    for _d, _n in _keys:  # the GPs on most runs first: the longest jobs
        _jobs += [("gp", (design[_d, :_n], design_y[_d, :_n, _j])) for _j in range(2)]
    for _d, _n in _keys:
        _xs, _ys = Scaler.fit(design[_d, :_n]), Scaler.fit(design_y[_d, :_n])
        _meta.append(("net", _d, _n, _xs, _ys))
        _jobs += [
            ("net", (_xs(design[_d, :_n]), _ys(design_y[_d, :_n]), _seed))
            for _seed in range(ENSEMBLE)
        ]
    _S, _Y = design[0, :BREAK_BUDGET], design_y[0, :BREAK_BUDGET]
    _xs, _ys = Scaler.fit(_S), Scaler.fit(_Y)
    _starts = ("zero", "const", "he0")
    _jobs += [
        ("net", (_xs(_S), _ys(_Y), _seed, _name)) for _name in _starts for _seed in range(ENSEMBLE)
    ]
    _raw = Scaler(np.zeros(2), np.ones(2))  # no scaling
    _jobs += [("net", (_S, _ys(_Y), _seed)) for _seed in range(ENSEMBLE)]  # settings in metres
    _jobs += [("net", (_xs(_S), _Y, _seed)) for _seed in range(ENSEMBLE)]  # outputs unscaled
    # network 0 on 1,024 runs: each training length, then the curves (20,000 steps and 4,000),
    # all judged on the validation runs, never on the map
    _xc, _yc = Scaler.fit(design[0, :1024]), Scaler.fit(design_y[0, :1024])
    _curve = (_xc(design[0, :1024]), _yc(design_y[0, :1024]), SEED)
    _jobs += [("net", (*_curve, "uniform", _steps)) for _steps in STEP_CHOICES]
    _jobs += [("net", (*_curve, "uniform", STEPS, _lr)) for _lr in LR_CHECK]  # the step check
    for _steps in (STEPS, 4_000):
        _jobs.append(("curve", (*_curve, _steps, _xc(val), _yc(val_y))))
    _done = JobParallel(n_jobs=WORKERS)(delayed(_job)(*_a) for _a in _jobs)

    gps = {_d: {} for _d in DESIGNS}
    nets = {_d: {} for _d in DESIGNS}
    _k = 0
    for _d, _n in _keys:
        gps[_d][_n] = GPs(_done[_k : _k + 2])
        _k += 2
    for _, _d, _n, _xs_n, _ys_n in _meta:
        nets[_d][_n] = Networks(_done[_k : _k + ENSEMBLE], _xs_n, _ys_n)
        _k += ENSEMBLE
    starts = {}
    for _name, _sc_x, _sc_y in (
        *((n, _xs, _ys) for n in _starts),
        ("raw", _raw, _ys),
        ("raw_out", _xs, _raw),
    ):
        starts[_name] = Networks(_done[_k : _k + ENSEMBLE], _sc_x, _sc_y)
        _k += ENSEMBLE
    lengths = {}
    for _steps in STEP_CHOICES:
        lengths[_steps] = Networks([_done[_k]], _xc, _yc)
        _k += 1
    rates = {LR: lengths[STEPS]}
    for _lr in LR_CHECK:
        rates[_lr] = Networks([_done[_k]], _xc, _yc)
        _k += 1
    curves = {STEPS: np.array(_done[_k][0]), 4_000: np.array(_done[_k + 1][0])}
    curves = {_s: _c * np.array([1, _yc.sd[1], _yc.sd[1]]) for _s, _c in curves.items()}  # metres
    snaps = _done[_k][1]  # network 0's weights at steps 0, 10, 50, and 20,000
    assert _k + 2 == len(_done)
    return Networks, curves, gps, lengths, nets, rates, snaps, starts


@app.cell
def _(TRACE, grid, grid_y, nets, np):
    # One run through network 0 of design 0 with 1,024 runs: the diagram's inset
    _net = nets[0][1024]
    _p = _net.members[0]
    _z = _net.xs(np.array([TRACE]))
    _h1 = np.maximum(0.0, _z @ _p[0] + _p[1])
    _h2 = np.maximum(0.0, _h1 @ _p[2] + _p[3])
    _y = _net.ys.inverse(_h2 @ _p[4] + _p[5])[0]
    _truth = grid_y[np.flatnonzero((grid == TRACE).all(axis=1))[0]]
    assert [round(float(_v), 2) for _v in _z[0]] == [-0.64, 0.45]
    assert (int((_h1 > 0).sum()), int((_h2 > 0).sum())) == (36, 32)
    assert [round(float(_v), 2) for _v in _y] == [23.66, 24.69]
    assert [round(float(_v), 2) for _v in _net(np.array([TRACE]))[0]] == [23.67, 24.71]
    assert [round(float(_v), 2) for _v in _truth] == [23.66, 24.72]
    return


@app.cell
def _(slice_s2, slice_y, np):
    # The backpropagation build on the attempt's hand-made network, at PRV-2 = 47 m. The target is
    # the simulator's value to the centimetre, so every number follows from the numbers shown.
    _a, _b, _d, _c1, _c2, _s = 14.89, 0.8, -0.4, 33.0, 43.0, 47.0
    _y = round(float(slice_y[np.flatnonzero(slice_s2 == _s)[0], 1]), 2)
    _h1, _h2 = max(0.0, _s - _c1), max(0.0, _s - _c2)
    _yhat = _a + _b * _h1 + _d * _h2
    _g = 2 * (_yhat - _y)
    _bp = [_y, _h1, _h2, _yhat, (_yhat - _y) ** 2, _g, _g * _h1, _g * _h2, -_g * _b, -_g * _d]
    assert [round(_v, 4) for _v in _bp] == [
        24.72,
        14,
        4,
        24.49,
        0.0529,
        -0.46,
        -6.44,
        -1.84,
        0.368,
        -0.184,
    ]
    return


@app.cell
def _(LR, PALETTE, STEPS, STEP_CHOICES, curves, lengths, np, pairwise, plt, rates, val, val_y):
    # Training curves: one network on 1,024 runs, 20,000 steps against 4,000 (same decay,
    # compressed), on its training runs and on the validation runs (never the map)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _steps, _ls in ((STEPS, "-"), (4_000, "--")):
        _h = curves[_steps]
        _ax.plot(_h[:, 0], _h[:, 1], color=PALETTE[0], ls=_ls, label=f"training, {_steps:,} steps")
        _ax.plot(
            _h[:, 0], _h[:, 2], color=PALETTE[1], ls=_ls, label=f"validation, {_steps:,} steps"
        )
    _ax.set(xlabel="Adam step", ylabel="RMS error, lowest pressure (m)", yscale="log")
    _ax.legend(fontsize=9)
    assert [round(float(curves[STEPS][-1, _j]), 3) for _j in (1, 2)] == [0.012, 0.014]
    assert [round(float(curves[4_000][-1, _j]), 3) for _j in (1, 2)] == [0.041, 0.044]
    # The length: trained to each, the validation error still falls with every doubling (no
    # overfitting on a deterministic simulator), so 20,000 steps is a budget, not a choice.
    _val = {
        k: float(np.sqrt(np.mean((lengths[k](val)[:, 1] - val_y[:, 1]) ** 2))) for k in STEP_CHOICES
    }
    assert [round(_val[k], 3) for k in STEP_CHOICES] == [0.057, 0.038, 0.022, 0.014, 0.011], _val
    assert all(_val[b] < _val[a] for a, b in pairwise(STEP_CHOICES))
    # The recipe was not tuned. Checked afterwards, never used to choose: peak steps either side.
    _by_lr = {
        k: float(np.sqrt(np.mean((rates[k](val)[:, 1] - val_y[:, 1]) ** 2))) for k in sorted(rates)
    }
    assert [round(_v, 3) for _v in _by_lr.values()] == [0.021, 0.014, 0.011], _by_lr
    assert min(_by_lr, key=_by_lr.get) != LR  # 0.01 does better: the networks' errors are high
    _fig
    return


@app.cell
def _(HI, LO, grid, grid_y, nets, np, plt, predict, snaps):
    # Training folds the sheet: network 0's surface of the lowest pressure at four steps, in 3D,
    # over the simulator's surface (grey mesh)
    _net = nets[0][1024]
    _n = int(round(HI - LO)) + 1
    _g = np.arange(LO, HI + 1e-9)
    _G1, _G2 = np.meshgrid(_g, _g)
    _truth = grid_y[:, 1].reshape(_n, _n).T
    _fig = plt.figure(figsize=(12, 3.2))
    _rms = {}
    for _k, _step in enumerate(sorted(snaps)):
        _pred = _net.ys.inverse(predict(snaps[_step], _net.xs(grid)))[:, 1]
        _rms[_step] = round(float(np.sqrt(np.mean((_pred - grid_y[:, 1]) ** 2))), 2)
        _ax = _fig.add_subplot(1, 4, _k + 1, projection="3d")
        _ax.plot_wireframe(_G1, _G2, _truth, rstride=5, cstride=5, color="grey", lw=0.5)
        _ax.plot_surface(_G1, _G2, np.clip(_pred.reshape(_n, _n).T, 5, 35), cmap="cividis",
                         vmin=5, vmax=35, alpha=0.9)  # fmt: skip
        _ax.set(zlim=(5, 35), title=f"step {_step:,}: {_rms[_step]} m")
        _ax.view_init(elev=24, azim=-135)
    # the training network is network 0 of the ensemble: same runs, seed, and schedule
    assert all(
        np.array_equal(_a, _b) for _a, _b in zip(snaps[20_000], _net.members[0], strict=True)
    )
    assert _rms == {0: 7.33, 10: 3.2, 50: 0.94, 20_000: 0.02}
    _fig
    return


@app.cell
def _(PALETTE, design, design_y, nets, np, plt, predict):
    # Two valleys: the error along the straight line between two trained networks' weights
    _net = nets[0][1024]
    _Z, _Y = _net.xs(design[0, :1024]), design_y[0, :1024, 1]
    _t = np.linspace(-0.25, 1.25, 61)
    _p0, _p1 = _net.members[0], _net.members[1]
    _rms = np.array(
        [
            np.sqrt(
                np.mean(
                    (
                        _net.ys.inverse(
                            predict(
                                [(1 - _u) * _a + _u * _b for _a, _b in zip(_p0, _p1, strict=True)],
                                _Z,
                            )
                        )[:, 1]
                        - _Y
                    )
                    ** 2
                )
            )
            for _u in _t
        ]
    )
    _i0, _i1, _im = np.argmin(np.abs(_t)), np.argmin(np.abs(_t - 1)), np.argmin(np.abs(_t - 0.5))
    assert [round(float(_rms[_i]), 3) for _i in (_i0, _i1)] == [0.012, 0.008]
    assert round(float(_rms[_im]), 2) == 6.31 and _rms[_im] == _rms[(_t > 0) & (_t < 1)].max()
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.plot(_t, _rms, color=PALETTE[1])
    _ax.set(
        xlabel="weights: (1 - t) network 1 + t network 2",
        ylabel="RMS error on its runs (m)",
        yscale="log",
    )
    _fig
    return


@app.cell
def _(SEED, design, design_y, nets, np, plt, predict):
    # The loss as a landscape, in 3D. Left: around network 1 along two random directions, each
    # scaled unit by unit to that unit's weights (Li et al., 2018). Right: over the plane through
    # networks 1, 2, 3, w1 + a (w2 - w1) + b (w3 - w1) (Garipov et al., 2018).
    _net = nets[0][1024]
    _Z, _Y = _net.xs(design[0, :1024]), design_y[0, :1024, 1]
    _w1, _w2, _w3 = _net.members[:3]

    def _mse(params):
        return float(np.mean((_net.ys.inverse(predict(params, _Z))[:, 1] - _Y) ** 2))

    def _direction(params, rng):
        out = []
        for _i, _p in enumerate(params):
            if _i % 2:  # a bias: left alone
                out.append(np.zeros_like(_p))
                continue
            _r = rng.normal(size=_p.shape)
            out.append(_r * np.linalg.norm(_p, axis=0) / (np.linalg.norm(_r, axis=0) + 1e-12))
        return out

    _rng = np.random.default_rng((SEED, 7))
    _d1, _d2 = _direction(_w1, _rng), _direction(_w1, _rng)
    _u = np.linspace(-0.4, 0.4, 61)

    def _around(a, b):
        return [_w + a * _x + b * _y for _w, _x, _y in zip(_w1, _d1, _d2, strict=True)]

    _bowl = np.array([[_mse(_around(_a, _b)) for _a in _u] for _b in _u])

    def _plane(a, b):
        return [
            _p + a * (_q - _p) + b * (_r - _p) for _p, _q, _r in zip(_w1, _w2, _w3, strict=True)
        ]

    _v = np.linspace(-0.25, 1.25, 91)
    _flat = np.array([[_mse(_plane(_a, _b)) for _a in _v] for _b in _v])
    _line = np.linspace(-0.25, 1.25, 121)
    _ridge = np.array([_mse(_plane(_a, 0.0)) for _a in _line])
    _edge = np.median(np.r_[_bowl[0], _bowl[-1], _bowl[:, 0], _bowl[:, -1]])
    assert round(float(_edge), 1) == 6.4  # m^2 at the edge of the close-up
    assert [round(_mse(_w), 5) for _w in (_w1, _w2, _w3)] == [0.00015, 0.00006, 0.00015]
    assert round(float(_ridge[(_line > 0) & (_line < 1)].max()), 1) == 39.8  # 6.31 m squared
    # the other two straight lines: network 1 to 3, and network 2 to 3
    _r13 = max(_mse(_plane(0.0, _t)) for _t in _line if 0 < _t < 1)
    _r23 = max(_mse(_plane(1.0 - _t, _t)) for _t in _line if 0 < _t < 1)
    assert (round(_r13, 1), round(_r23, 1)) == (25.4, 30.1)

    _fig = plt.figure(figsize=(11, 4.2))
    for _k, (_s, _h, _title) in enumerate(
        ((_u, _bowl, "around network 1: a bowl"), (_v, _flat, "networks 1, 2, 3: valleys"))
    ):
        _ax = _fig.add_subplot(1, 2, _k + 1, projection="3d")
        _A, _B = np.meshgrid(_s, _s)
        _ax.plot_surface(_A, _B, np.minimum(_h, 45), cmap="cividis", vmin=0, vmax=45)
        _ax.set(zlim=(0, 45), title=_title, zlabel="squared error (m²)")
        _ax.view_init(elev=32, azim=-60)
    _fig
    return


@app.cell
def _(HI, LO, Networks, PALETTE, init_params, lines, nets, np, plt):
    def kink_lines(net, member):
        """First-layer kink lines in settings units: s2 = slope * s1 + intercept."""
        W, b = net.members[member][0], net.members[member][1]
        a = W / net.xs.sd[:, None]
        c = -b + (W * (net.xs.mean / net.xs.sd)[:, None]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):  # no weight on PRV-2: vertical
            return -a[0] / a[1], c / a[1]

    _net = nets[0][1024]
    _centre = 0.5 * (LO + HI)
    _offsets, _nearest = [], {"prv2": [], "prv1": []}
    for _i in range(len(_net.members)):
        _slope, _icpt = kink_lines(_net, _i)
        _off = _slope * _centre + _icpt - _centre
        _off = _off[(_slope > 0.8) & (_slope < 1.25) & (np.abs(_off) < HI - LO)]
        _offsets.append(_off)
        for _k, _list in _nearest.items():
            _list.append(float(np.abs(_off - lines[_k]["offset"]).min()))
    assert [len(_o) for _o in _offsets] == [10, 11, 12, 16, 10]
    # the slide's one ramp: network 0's parallel unit nearest PRV-2's shut-off line
    _slope0, _icpt0 = kink_lines(_net, 0)
    _off0 = _slope0 * _centre + _icpt0 - _centre
    _par0 = (_slope0 > 0.8) & (_slope0 < 1.25)
    assert (
        round(float(_off0[_par0][np.argmin(np.abs(_off0[_par0] - lines["prv2"]["offset"]))]), 2)
        == -3.91
    )
    assert round(max(_nearest["prv2"]), 2) == 0.59
    assert [round(_v, 2) for _v in _nearest["prv1"]] == [0.13, 2.22, 0.23, 0.21, 0.13]
    assert (round(min(_o.min() for _o in _offsets)), round(max(_o.max() for _o in _offsets))) == (
        -6,
        32,
    )
    # the baseline: the same seeds at the start, scaled as trained; training moved the kinks
    _start = Networks([init_params(_i, (2, 64, 64, 2)) for _i in range(5)], _net.xs, _net.ys)
    _start_near = {"prv2": [], "prv1": []}
    for _i in range(5):
        _slope, _icpt = kink_lines(_start, _i)
        _off = _slope * _centre + _icpt - _centre
        _off = _off[(_slope > 0.8) & (_slope < 1.25) & (np.abs(_off) < HI - LO)]
        for _k, _list in _start_near.items():
            _list.append(round(float(np.abs(_off - lines[_k]["offset"]).min()), 2))
    assert _start_near == {
        "prv2": [1.68, 1.4, 0.41, 1.81, 3.04],
        "prv1": [14.04, 2.63, 8.42, 14.36, 7.31],
    }, _start_near

    _fig, _ax = plt.subplots(figsize=(6.4, 3.6))
    for _i, _off in enumerate(_offsets):
        _ax.plot(_off, np.full_like(_off, _i + 1), "o", color="#1A1A1A", mfc="white")
    _ax.axvline(lines["prv2"]["offset"], color=PALETTE[0], ls="--", lw=2.4, label="PRV-2 shuts")
    _ax.axvline(lines["prv1"]["offset"], color=PALETTE[1], ls="-.", lw=2.4, label="PRV-1 shuts")
    _ax.set(xlim=(-(HI - LO), HI - LO), xlabel="Kink line: PRV-2 minus PRV-1 (m)", ylabel="Network")
    _ax.legend(loc="lower left")
    _fig.tight_layout()
    _fig
    return (kink_lines,)


@app.cell
def _(mo):
    mo.md(r"""
    **Hidden units found the valves.** Each network has 10 to 16 first-layer kinks parallel to the
    shut-off lines (slope 0.8 to 1.25), and some sit right on them: PRV-2's line has a kink within
    0.59 m in all five networks; PRV-1's, within 0.23 m in four and 2.22 m in the fifth. At the
    start the same seeds put their nearest kinks 0.41 to 3.04 m from PRV-2's line and 2.63 to
    14.36 m from PRV-1's: training moved them there. The shut-off lines come from the simulator's
    valve states, which the networks never saw.
    """)
    return


@app.cell
def _(BUDGETS, DESIGNS, PALETTE, gps, grid, grid_y, nets, np, plt):
    def errors(pred, truth):
        """RMS and worst absolute error of each output over the map."""
        e = pred - truth
        return {
            o: (float(np.sqrt((e[:, j] ** 2).mean())), float(np.abs(e[:, j]).max()))
            for j, o in enumerate(("leak", "pmin"))
        }

    _table, _per = {}, {}
    for _n in BUDGETS:
        for _name, _get in (
            ("gp", lambda d, n: gps[d][n](grid)),
            ("networks", lambda d, n: nets[d][n](grid)),
            ("one", lambda d, n: nets[d][n].each(grid)[0]),
        ):
            _rows = [errors(_get(_d, _n), grid_y) for _d in DESIGNS]
            _per[_name, _n] = _rows  # every design's errors, design 0 first
            _table[_name, _n] = {
                o: tuple(float(np.median([r[o][k] for r in _rows])) for k in range(2))
                for o in ("leak", "pmin")
            }
    # not only the medians: on pressure the networks beat the GP for every budget and design
    assert all(
        _per["networks", _n][_d]["pmin"][0] < _per["gp", _n][_d]["pmin"][0]
        for _n in BUDGETS
        for _d in DESIGNS
    )
    # and no GP fit ended at one of Lecture 12's bounds
    assert not any(any(gps[_d][_n].at_bound) for _d in DESIGNS for _n in BUDGETS)

    def _r(name, n, o, k, d):
        return round(_table[name, n][o][k], d)

    # lowest pressure: RMS and worst, networks against the GP
    assert (_r("networks", 16, "pmin", 0, 2), _r("gp", 16, "pmin", 0, 2)) == (0.99, 1.26)
    assert (_r("networks", 64, "pmin", 0, 2), _r("gp", 64, "pmin", 0, 2)) == (0.19, 0.25)
    assert (_r("networks", 256, "pmin", 0, 3), _r("gp", 256, "pmin", 0, 3)) == (0.028, 0.081)
    assert (_r("networks", 1024, "pmin", 0, 3), _r("gp", 1024, "pmin", 0, 3)) == (0.011, 0.026)
    assert (_r("networks", 16, "pmin", 1, 1), _r("gp", 16, "pmin", 1, 1)) == (3.6, 7.8)
    assert (_r("networks", 1024, "pmin", 1, 2), _r("gp", 1024, "pmin", 1, 2)) == (0.17, 0.34)
    assert _r("one", 1024, "pmin", 1, 2) == 0.21
    # leakage: close at every budget; designs where the GP's RMS error is the lower, by budget
    assert (_r("networks", 1024, "leak", 0, 3), _r("gp", 1024, "leak", 0, 3)) == (0.004, 0.004)
    assert (_r("networks", 1024, "leak", 1, 3), _r("gp", 1024, "leak", 1, 3)) == (0.034, 0.023)
    assert [
        sum(
            1
            for _g, _q in zip(_per["gp", _n], _per["networks", _n], strict=True)
            if _g["leak"][0] < _q["leak"][0]
        )
        for _n in BUDGETS
    ] == [0, 2, 2, 3]
    # one network instead of five: the 15 single networks' worst pressure errors at 1,024 runs
    _single = [errors(_p, grid_y)["pmin"][1] for _d in DESIGNS for _p in nets[_d][1024].each(grid)]
    assert (
        round(float(np.median(_single)), 2),
        round(min(_single), 2),
        round(max(_single), 2),
    ) == (
        0.15,
        0.07,
        0.25,
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for _ax, _o, _title in zip(
        _axes, ("leak", "pmin"), ("Leakage (m³/h)", "Lowest pressure (m)"), strict=True
    ):
        for _name, _c, _mk in (("gp", PALETTE[0], "o"), ("networks", PALETTE[1], "s")):
            for _k, _ls, _mfc, _lab in ((1, "-", _c, "worst"), (0, "--", "white", "RMS")):
                _med = np.array([_table[_name, _n][_o][_k] for _n in BUDGETS])
                _lo = np.array([min(r[_o][_k] for r in _per[_name, _n]) for _n in BUDGETS])
                _hi = np.array([max(r[_o][_k] for r in _per[_name, _n]) for _n in BUDGETS])
                _ax.errorbar(BUDGETS, _med, yerr=[_med - _lo, _hi - _med], color=_c, marker=_mk,
                             ls=_ls, mfc=_mfc, capsize=3, label=f"{_name}, {_lab}")  # fmt: skip
        _ax.set(xscale="log", yscale="log", xlabel="Runs used for training", title=_title)
        _ax.set_xticks(BUDGETS, [str(_n) for _n in BUDGETS])
        _ax.minorticks_off()
    _axes[0].set_ylabel("Error over the 1-m map")
    _axes[0].legend(fontsize=9)
    _fig.tight_layout()
    _fig
    return (errors,)


@app.cell
def _(BUDGETS, DESIGNS, NEAR_LINE_M, gps, grid, grid_y, lines, nets, np):
    # Where does the networks' lead on pressure come from? The settings within 2 m (along PRV-2)
    # of either shut-off line, where a valve switches, against the rest of the map.
    _dist = [
        np.abs(grid[:, 1] - (_l["slope"] * grid[:, 0] + _l["intercept"])) for _l in lines.values()
    ]
    _near = np.minimum(*_dist) <= NEAR_LINE_M
    assert int(_near.sum()) == 192

    def _rms(e, m):
        return float(np.sqrt(np.mean(e[m] ** 2)))

    _rmse, _ratio = {}, {}
    for _n in BUDGETS:
        _rows = []
        for _d in DESIGNS:
            _eg = gps[_d][_n](grid)[:, 1] - grid_y[:, 1]
            _en = nets[_d][_n](grid)[:, 1] - grid_y[:, 1]
            _rows.append([_rms(_eg, _near), _rms(_eg, ~_near), _rms(_en, _near), _rms(_en, ~_near)])
        _rows = np.array(_rows)  # (designs, [GP near, GP away, networks near, networks away])
        _rmse[_n] = np.median(_rows, axis=0)
        _ratio[_n] = (_rows[:, 0] / _rows[:, 2], _rows[:, 1] / _rows[:, 3])  # GP over networks
    # both models err more near the switches than away (medians of three designs), every budget
    assert all(_rmse[_n][0] > _rmse[_n][1] and _rmse[_n][2] > _rmse[_n][3] for _n in BUDGETS)
    # 1,024 runs: the lead is larger near the switches, in every design
    assert np.all(_ratio[1024][0] > _ratio[1024][1])
    assert [round(float(np.median(_r)), 1) for _r in _ratio[1024]] == [3.5, 1.9]
    assert [round(float(_v), 3) for _v in _rmse[1024]] == [0.053, 0.019, 0.015, 0.01]
    # 16 to 256 runs: about the same lead near and away (GP's error 1.3 to 3.1 times the networks')
    assert [round(float(np.median(_r)), 1) for _n in (16, 64, 256) for _r in _ratio[_n]] == [
        1.4,
        1.3,
        1.5,
        1.6,
        3.1,
        2.9,
    ]
    print({_n: [round(float(np.median(_r)), 2) for _r in _ratio[_n]] for _n in BUDGETS})
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Networks lead on pressure.** On the lowest pressure, five networks beat the GP at every
    budget: at 1,024 runs the worst error over the map is 0.17 m, half the GP's 0.34. Averaging
    five gave no clear gain on the worst error: the 15 single networks range from 0.07 to 0.25 m
    (median 0.15). Leakage is smooth and close at every budget: the networks lower in all three
    designs at 16 runs, the GP in two of three at 64 and 256 and in all three at 1,024. The
    bracket: a plane fitted to 256 runs misses the pressures by 2.44 m RMS (§5); the floor is
    zero, because the simulator is deterministic.

    **Where the lead comes from** (the cell above). Both models' errors are larger within 2 m of a
    valve's shut-off line (192 of the 1,296 settings) than elsewhere (medians of three designs). At 16 to 256 runs the networks' lead is about the
    same there as elsewhere; at 1,024 runs it is larger there, in every design: the GP's RMS
    error is 3.5 times the networks' near the lines (0.053 against 0.015 m) and 1.9 times away
    (0.019 against 0.010 m).

    ## 5. Break it

    **Predict first.** (1) Rank four bad starts against today's (0.04 m RMS on 256 runs, the
    median of five seeds): all
    weights 0; all weights equal; He's normal weights with zero biases; settings in metres. (2) At
    PRV-2 = 90 m, 30 m past the runs, how far apart are the five networks, and how far from the
    simulator? (3) How often is a surrogate's best pick really short of 24.6 m?
    """)
    return


@app.cell
def _(
    BREAK_BUDGET,
    HI,
    LO,
    Networks,
    SEED,
    design,
    design_y,
    errors,
    grid,
    grid_y,
    init_params,
    nets,
    np,
    starts,
):
    def kink_mask(net, member=0):
        """First-layer units on at some corners of the box and off at others."""
        corners = np.array([(LO, LO), (LO, HI), (HI, LO), (HI, HI)])
        a = net.xs(corners) @ net.members[member][0] + net.members[member][1]
        return (a > 0).any(axis=0) & (a < 0).any(axis=0)

    def kinks_in_box(net, member=0):
        return int(kink_mask(net, member).sum())

    # Five networks per start (seeds 0-4; "uniform" is the 256-run ensemble): median (range)
    _all = {"uniform": nets[0][BREAK_BUDGET]} | starts
    _bad = {}
    for _k, _n in _all.items():
        _e = [errors(_p, grid_y)["pmin"] for _p in _n.each(grid)]
        assert len(_e) == 5
        _bad[_k] = tuple(
            (round(float(np.median(_v)), 2), round(min(_v), 2), round(max(_v), 2))
            for _v in zip(*_e, strict=True)
        )  # (RMS: median, lo, hi), (worst: median, lo, hi)
    assert _bad == {
        "uniform": ((0.04, 0.03, 0.05), (0.31, 0.18, 0.33)),
        "he0": ((0.09, 0.09, 0.12), (0.70, 0.63, 0.79)),
        "raw": ((0.15, 0.13, 0.18), (1.23, 0.76, 1.73)),
        "raw_out": ((0.04, 0.03, 0.1), (0.31, 0.24, 0.63)),
        "const": ((2.49, 2.49, 2.49), (8.88, 8.88, 8.88)),
        "zero": ((6.98, 6.98, 6.98), (19.24, 19.24, 19.24)),
    }, _bad
    # zeros: no weight but the output bias moved, so every prediction is the training mean
    _p = starts["zero"].members[0]
    assert not any(np.any(_q != 0) for _q in _p[:-1])
    for _pred in starts["zero"].each(grid):
        assert np.allclose(_pred, design_y[0, :BREAK_BUDGET].mean(axis=0), atol=1e-9)
    # equal weights: the units stay copies, to round-off
    _q = starts["const"].members[0]
    assert (
        f"{np.abs(_q[0] - _q[0][:, :1]).max():.0e}" == "6e-16"
        and np.abs(_q[2] - _q[2][:, :1]).max() == 0
    )
    # settings in metres: 17 of 64 kinks cross the box at the start, 18 after training (not the
    # same units); 60 of 64 scaled
    _raw0 = Networks([init_params(SEED, (2, 64, 64, 2))], starts["raw"].xs, starts["raw"].ys)
    assert (
        kinks_in_box(_raw0),
        kinks_in_box(starts["raw"]),
        kinks_in_box(nets[0][BREAK_BUDGET]),
    ) == (17, 18, 60)
    assert not np.array_equal(kink_mask(_raw0), kink_mask(starts["raw"]))
    _he0 = Networks(
        [init_params(SEED, (2, 64, 64, 2), "he0")],
        nets[0][BREAK_BUDGET].xs,
        nets[0][BREAK_BUDGET].ys,
    )
    assert kinks_in_box(_he0) == 64  # zero biases: every kink through the centre of the runs
    # the plane: the baseline above every error
    _S, _Y = design[0, :BREAK_BUDGET], design_y[0, :BREAK_BUDGET]
    _plane, *_ = np.linalg.lstsq(np.c_[np.ones(len(_S)), _S], _Y[:, 1], rcond=None)
    assert (
        round(
            float(np.sqrt(np.mean((np.c_[np.ones(len(grid)), grid] @ _plane - grid_y[:, 1]) ** 2))),
            2,
        )
        == 2.44
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A bad start never recovers** (five networks per start, seeds 0–4, 256 runs, 20,000 steps;
    lowest pressure over the map, median and range):

    | Start | RMS (m) | Worst (m) |
    |:------|--------:|----------:|
    | Uniform, scaled (today's) | 0.04 (0.03–0.05) | 0.31 (0.18–0.33) |
    | Outputs left unscaled | 0.04 (0.03–0.10) | 0.31 (0.24–0.63) |
    | He's weights, zero biases | 0.09 (0.09–0.12) | 0.70 (0.63–0.79) |
    | Settings in metres | 0.15 (0.13–0.18) | 1.23 (0.76–1.73) |
    | Every weight 0.1 | 2.49 | 8.88 |
    | Every weight 0 | 6.98 | 19.24 |

    Zeros: every hidden unit outputs 0, so no gradient reaches the weights and the network
    predicts the mean. Equal weights: identical units get identical gradients; they stay copies
    (to $6\times10^{-16}$) and the network is no better than a plane (2.44 m). He's start puts
    every first-layer kink through the centre of the runs. Settings in metres (outputs still
    scaled): settings near 40 m and starting weights of at most 0.71 put the kinks near the
    origin; 17 of 64 cross the box at the start and 18 after training. Outputs left in m and m³/h
    (settings scaled) give the same median, 0.04 m, with a wider range.
    """)
    return


@app.cell
def _(EXTRAP_HI, HI, PALETTE, SLICE_S1, cap, gps, grid, nets, np, plt, slice_s2, slice_y):
    _S = np.c_[np.full_like(slice_s2, SLICE_S1), slice_s2]
    _each = nets[0][1024].each(_S)  # (5, points, 2)
    _mu, _sd = gps[0][1024](_S, sd=True)
    _inside = slice_s2 <= HI
    _spread = _each.max(axis=0) - _each.min(axis=0)
    assert (
        round(float(_spread[_inside, 1].max()), 2),
        round(float(_spread[_inside, 0].max()), 2),
    ) == (0.08, 0.05)
    _i90 = int(np.flatnonzero(slice_s2 == EXTRAP_HI)[0])
    assert [
        round(float(_v), 2)
        for _v in (_each[:, _i90, 1].min(), _each[:, _i90, 1].max(), slice_y[_i90, 1])
    ] == [45.56, 51.06, 33.19]
    assert [
        round(float(_v), 2)
        for _v in (_each[:, _i90, 0].min(), _each[:, _i90, 0].max(), slice_y[_i90, 0])
    ] == [38.51, 42.22, 35.65]
    assert [
        round(float(_v), 2) for _v in (_mu[_i90, 1], _sd[_i90, 1], _mu[_i90, 0], _sd[_i90, 0])
    ] == [25.40, 4.24, 35.10, 4.70]
    assert np.all(np.abs(_mu[_i90] - slice_y[_i90]) <= 2 * _sd[_i90])  # the GP's band covers both
    assert (
        slice_s2[slice_y[:, 0] >= slice_y[:, 0].max() - 1e-6][0] == 75.0
    )  # leakage flat from 75 m
    assert slice_s2[slice_y[:, 1] == cap][0] == 65.0  # the pressure stops at the cap from 65 m
    # over the whole map, not only this slice, the five agree within 0.15 m (0.06 m3/h)
    _map = nets[0][1024].each(grid)
    assert [round(float(_v), 2) for _v in (_map.max(0) - _map.min(0)).max(0)] == [0.06, 0.15]
    # the other two designs' networks go wrong together too
    _at90 = [nets[_d][1024].each(_S[_i90 : _i90 + 1])[:, 0, 1] for _d in (1, 2)]
    assert [(round(float(_e.min()), 2), round(float(_e.max()), 2)) for _e in _at90] == [
        (44.01, 51.43),
        (42.55, 50.69),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for _j, (_ax, _title) in enumerate(
        zip(_axes, ("Leakage (m³/h)", "Lowest pressure (m)"), strict=True)
    ):
        _ax.axvspan(HI, EXTRAP_HI, color="#F0F0F0", zorder=0)
        _ax.fill_between(
            slice_s2,
            _mu[:, _j] - 2 * _sd[:, _j],
            _mu[:, _j] + 2 * _sd[:, _j],
            color=PALETTE[0],
            alpha=0.2,
            lw=0,
            label="GP, ±2 sd",
        )
        for _k, _row in enumerate(_each[:, :, _j]):
            _ax.plot(
                slice_s2, _row, color=PALETTE[1], lw=1.2, label="five networks" if _k == 0 else None
            )
        _ax.plot(slice_s2, slice_y[:, _j], color="#1A1A1A", ls="--", lw=2.2, label="simulator")
        _ax.set(title=_title, xlabel=f"PRV-2 setting (m); PRV-1 at {SLICE_S1:.0f} m")
    _axes[0].legend(loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(HI, PALETTE, SLICE_S1, nets, np, plt, slice_s2, slice_y):
    # Past the runs, the slopes stay up: the slope of the predicted lowest pressure along the slice
    _S = np.c_[np.full_like(slice_s2, SLICE_S1), slice_s2]
    _each = nets[0][1024].each(_S)[:, :, 1]
    _mid = 0.5 * (slice_s2[1:] + slice_s2[:-1])
    _slope = np.diff(_each, axis=1)
    _truth = np.diff(slice_y[:, 1])
    _last = [float(_mid[np.flatnonzero(np.abs(np.diff(_r)) > 1e-9)].max()) for _r in _slope]
    assert [round(_v, 2) for _v in _last] == [84.5, 87.5, 87.5, 87.5, 82.5]
    assert [round(float(_v), 2) for _v in np.sort(_slope[:, -1])] == [0.4, 0.43, 0.49, 0.5, 0.63]
    assert round(float(_truth[-1]), 2) == 0.0
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.axvspan(HI, slice_s2.max(), color="#F0F0F0")
    for _r in _slope:
        _ax.plot(_mid, _r, color=PALETTE[1], lw=1.2)
    _ax.plot(_mid, _truth, "--", color="#1A1A1A", lw=2)
    _ax.set(xlabel="PRV-2 setting (m); PRV-1 at 36 m", ylabel="Slope (m per m)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Five networks agree, all wrong** (the decisive test). Inside the box the five networks agree
    within 0.08 m on pressure and 0.05 m³/h on leakage along this slice (0.15 m and 0.06 m³/h over
    the whole map). Past it, each keeps its last ramps: at
    90 m they predict 45.56 to 51.06 m of lowest pressure where the simulator gives 33.19 m (PRV-3
    holds Area B; no setting of these two valves can raise it), and 38.51 to 42.22 m³/h of leakage
    against 35.65 (flat from 75 m, where PRV-2 is fully open). The other two designs' networks
    predict 44.01 to 51.43 and 42.55 to 50.69 m. The GP's band widens and covers both.
    **The assumption failed:** no run pinned the ramps past 60 m. Agreement is not accuracy: five
    networks trained on the same runs share the same blind spot.

    ## 6. Decide
    """)
    return


@app.cell
def _(
    ACCURATE,
    ALPHA,
    BUDGETS,
    CAL_RUNS,
    DESIGNS,
    HI,
    LO,
    OPT_WINDOW,
    P_SERVICE,
    design,
    design_y,
    gps,
    grid,
    grid_y,
    lattice,
    nets,
    np,
    outputs,
    simulate,
    window,
    window_y,
):
    def choose(pred, cand, margin=0.0):
        """Least predicted leakage among candidates predicted to keep 24.6 m, less the margin."""
        ok = pred[:, 1] - margin >= P_SERVICE
        if not ok.any():
            raise ValueError("no candidate is predicted feasible")
        i = np.flatnonzero(ok)[np.argmin(pred[ok, 0])]
        return float(cand[i, 0]), float(cand[i, 1])

    _cand = lattice(LO, HI, LO, HI)
    assert len(_cand) == 30_976
    _feas = window_y[:, 1] >= P_SERVICE
    _i = np.flatnonzero(_feas)[np.argmin(window_y[_feas, 0])]
    _best, _best_leak = window[_i], float(window_y[_i, 0])
    assert (
        OPT_WINDOW[0][0] < _best[0] < OPT_WINDOW[0][1]
        and OPT_WINDOW[1][0] < _best[1] < OPT_WINDOW[1][1]
    )
    assert (round(float(_best[0]), 2), round(float(_best[1]), 2), round(_best_leak, 2)) == (
        36.0,
        46.8,
        23.61,
    )

    _picks = {}
    for _d in DESIGNS:
        for _n in BUDGETS:
            _picks["networks", _d, _n] = choose(nets[_d][_n](_cand), _cand)
            _picks["gp", _d, _n] = choose(gps[_d][_n](_cand), _cand)
    # The margin: the 95% split-conformal quantile of the networks' overprediction on 256 runs
    # they never saw, those of the next design; never negative. One per design and budget.
    _margins = {}
    for _d in DESIGNS:
        _c = (_d + 1) % len(DESIGNS)
        for _n in BUDGETS:
            _res = np.sort(nets[_d][_n](design[_c, :CAL_RUNS])[:, 1] - design_y[_c, :CAL_RUNS, 1])
            _margins[_d, _n] = max(0.0, float(_res[int(np.ceil((CAL_RUNS + 1) * (1 - ALPHA))) - 1]))
            _picks["margin", _d, _n] = choose(nets[_d][_n](_cand), _cand, _margins[_d, _n])
    _net, _margin = nets[0][1024], _margins[0, 1024]
    _mu, _sd = gps[0][1024](_cand, sd=True)
    _picks["gp_2sd", 0, 1024] = choose(np.c_[_mu[:, 0], _mu[:, 1] - 2 * _sd[:, 1]], _cand)
    _pairs = sorted(set(_picks.values()))
    _truth = dict(zip(_pairs, outputs(simulate(_pairs)), strict=True))
    # Feasibility re-checked 100 times more tightly than the model file's accuracy (0.01)
    _exact = dict(zip(_pairs, outputs(simulate(_pairs, accuracy=ACCURATE)), strict=True))
    _shift = max(abs(_exact[q][1] - _truth[q][1]) for q in _pairs)
    assert round(1000 * _shift, 1) == 0.2 and _shift < 2e-4  # m: under 0.2 mm

    _plain = [k for k in _picks if k[0] in ("networks", "gp")]
    _short = [P_SERVICE - _exact[_picks[k]][1] for k in _plain]
    assert (len(_plain), sum(_x > 0 for _x in _short), round(max(_short), 2)) == (24, 7, 0.09)
    # like for like: each surrogate's own 12 picks (the margin is the networks')
    assert [
        sum(1 for _k, _x in zip(_plain, _short, strict=True) if _k[0] == _m and _x > 0)
        for _m in ("networks", "gp")
    ] == [4, 3]
    assert sorted(round(1000 * _x, 1) for _x in _short if _x > 0)[:3] == [1.7, 1.7, 1.7]  # mm
    _with = [k for k in _picks if k[0] == "margin"]
    assert len(_with) == 12 and all(_exact[_picks[k]][1] >= P_SERVICE for k in _with)
    assert (round(min(_margins.values()), 2), round(max(_margins.values()), 2)) == (0.01, 2.15)
    assert round(max(_truth[_picks[k]][0] - _best_leak for k in _with), 2) == 1.35
    assert round(_margin, 2) == 0.02
    assert round(float(np.mean(_net(grid)[:, 1] - grid_y[:, 1] <= _margin)), 3) == 0.962
    _mg, _sg = gps[0][1024](grid, sd=True)
    assert round(float(np.mean(np.abs(_mg[:, 1] - grid_y[:, 1]) <= 2 * _sg[:, 1])), 3) == 0.919
    # one-sided, the event the GP's own band pick relies on (nominal 97.7% for a Gaussian)
    assert round(float(np.mean(_mg[:, 1] - 2 * _sg[:, 1] <= grid_y[:, 1])), 3) == 0.951
    _m = _picks["margin", 0, 1024]
    assert _m == (36.0, 46.8) and [round(float(_v), 2) for _v in _truth[_m]] == [23.61, 24.63]
    print("margins (m):", {k: round(v, 2) for k, v in _margins.items()})
    return


@app.cell
def _(P_SERVICE, TODAY, town):
    # Who sits at the margin: customers within 3 m of the limit at the chosen settings and today
    _near = {}
    for _name, _s in (("pick", (36.0, 46.8)), ("today", TODAY)):
        _low = town.customer_pressure(*_s)
        assert (_low >= P_SERVICE - 1e-6).all()
        _near[_name] = int((_low < P_SERVICE + 3.0).sum())
    assert _near == {"pick": 7, "today": 0}
    return


@app.cell
def _(mo):
    mo.md(r"""
    **A pick on the line needs a margin.** Each surrogate's best pick (on a 0.2-m grid of 30,976
    candidates) sits on its own 24.6 m line, so it is short whenever the surrogate overpredicts the
    pressure there: 7 of 24 picks were (4 of the networks' 12, 3 of the GP's 12), by up to 0.09 m
    (three by only 1.7 mm; each pick re-checked at EPANET accuracy $10^{-4}$, 100 times the
    file's, which moves pressures by under 0.2 mm). A margin from 256 held-out runs of another
    design (split conformal, Lecture 11) is 0.02 m for five networks on 1,024 runs and covers
    96.2% of the map (one-sided, nominal 95%), a guarantee over the box, not at the boundary. The
    GP's 2-sd band covers 91.9% of the map both ways, and 95.1% for the one-sided bound its own
    pick uses (nominal 97.7%). With it they pick (36.0, 46.8) m: 23.61 m³/h at
    24.63 m, the best on the grid. With a margin of its own for every design and budget (0.01 to
    2.15 m), none of the 12 picks is short; the price is up to 1.35 m³/h of leakage at 16 runs.
    Lecture 12's lesson still holds: this margin covers the surrogate's error against the
    simulator only; nothing here measures the simulator's error against the town, which for a
    city model is likely larger than 0.02 m.

    **When a network is the wrong tool.** Few runs and a smooth response: Lecture 12's GP found
    the optimum in 15 runs and comes with a band. Past the runs: ramps extrapolate linearly and nothing warns you. A
    calibrated error bar: a network has none, and five networks' spread is not one. Its home: many
    runs, switches and kinks, many inputs and outputs.

    **Impact.** A surrogate answers in milliseconds, so operators will ask it more, and further out;
    its errors land on the customers at the margin (n50, on high ground at the end of the line).
    Log every query, flag those outside the runs, retrain when the city model is recalibrated.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: what goes in the control room?

    - **A.** Lecture 12's GP on 15 runs: it found the optimum.
    - **B.** Five networks on 1,024 runs: half the GP's worst pressure error.
    - **C.** B, with a margin from held-out runs and a refusal outside 25–60 m.
    - **D.** The GP on 1,024 runs: its band covers past the box.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                **C.** A: a search's runs pin the optimum, not the box; even 16 runs spread over
                the box leave the GP a worst pressure error of 7.8 m. B: without a margin, picks
                fall short (4 of the networks' 12), and past the box it answers with pressures no
                setting can reach. D is defensible: its one-sided band covers 95.1% of the map,
                about as much as C's margin (96.2%); its weakness is the worst pressure error,
                0.34 m against the networks' 0.17. What would change the answer: a gauge at n50
                reading 1.5 m below the model (the margin must come from the model's error against
                the town), or a request for PRV-2 at 70 m during a main break (refuse; run the
                simulator).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Inside every LLM: MLPs and autograd.** A transformer layer is attention plus an MLP like
    today's; with the usual 4× width the MLP holds two-thirds of the layer's weights (attention's
    four $d\times d$ projections, $4d^2$, against $8d^2$; Vaswani et al., 2017). PyTorch's autograd
    runs today's backward pass over billions of weights (Paszke et al., 2019). Neural surrogates
    replace simulators: GraphCast forecasts 10 days in under a minute (Lam et al., 2023); in water
    systems, Broad, Dandy & Maier (2005) replaced EPANET with networks inside a genetic algorithm.

    ## 7. Check yourself

    **Exit question.** A colleague trains five networks on 1,024 runs of a flood model with
    rainfall up to 100 mm. For a 150 mm storm the five agree within 1 cm, and the colleague calls
    the prediction certain. What do you say, and what do you check first?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Agreement is not accuracy. 150 mm is outside the runs, where every ramp keeps its
                last slope and every member shares the same blind spot, as at 90 m in §5. Check
                the training range first; outside it, run the model, and add runs there if such
                storms matter.
                """
            )
        }
    )
    return


@app.cell
def _(SEED, T16, Z16, np, torch_adam, train):
    # Gotcha, computed: once the loss reaches round-off, Adam divides noise by noise.
    _ours, _loss = train(Z16, T16, SEED, steps=20_000)
    _drift = max(float(np.abs(a - b).max()) for a, b in zip(_ours, torch_adam(20_000), strict=True))
    assert _loss < 1e-30 and _drift > 0.01
    print(
        f"after 20,000 steps: training loss {_loss:.1e}; weights differ from PyTorch's by {_drift:.2f}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Comparing trajectories after the loss is zero.** On 16 runs both our Adam and PyTorch's fit
      exactly (loss below $10^{-30}$); after that, Adam divides round-off by round-off, and the two
      runs' weights drift apart (the cell above). Compare optimizers while the loss is still large;
      compare predictions, not weights, after.
    - **Zero biases with low-dimensional inputs.** Every first-layer kink starts through one point;
      with two inputs that wastes most units. PyTorch's default draws the biases at random too.
    - **Unscaled inputs.** Kinks near the origin, far from the data (§5). Standardize on the
      training rows.
    - **Trusting an ensemble's spread.** It measures disagreement, not error; outside the data the
      members agree and are wrong together.
    - **Checking a surrogate at its training runs.** Every network here fits its runs almost
      exactly; check on runs it never saw (the 1-m map).
    - **Threads.** WNTR's EPANET is not thread-safe: simulate in processes. One BLAS thread per
      process keeps the networks' sums, and so their weights, reproducible.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Forsyth §16.3.1–16.3.4:** layers as functions $o^{(r)}(u^{(r)},\theta^{(r)})$ and the
      backpropagation recursion written with Jacobians, $v^{(i-1)}=v^{(i)}J_{o^{(i)};u^{(i)}}$.
      His gradients are row vectors and his "layer" counts the linear map and the ReLU separately;
      our `back` functions compute the same products $v^{(i)}J$ without forming any Jacobian.
    - **Forsyth §16.4.1:** the software environments that build the backward pass from a
      description of the forward one: today's `Var` is a minimal version of that idea.
    - **Forsyth §16.4.4:** why plain gradient steps crawl along narrow valleys, the motivation for
      momentum and Adam (Lecture 13).
    - **Griewank & Walther (2008), Ch. 3–4:** reverse mode and its cost bound, a small constant
      times the cost of the function, whatever the number of inputs.
    - **Exercise.** Add `tanh` to the autograd (its derivative is $1-\tanh^2$), retrain the
      networks on 256 runs, and compare the worst pressure error with ReLU's. Where do smooth units
      lose to ramps, and why?
    - **Exercise.** Write the forward-mode alternative: one pass per weight, carrying a tangent.
      Count the passes for 4,482 weights and time both on 1,024 runs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A hidden unit is a ramp. A network sums and folds ramps, and puts them where the runs bend.
    - Backpropagation gives every gradient in one backward pass; check it against finite
      differences.
    - Start random and small, with scaled inputs: zeros never learn, metres barely do.
    - Agreement is not accuracy: past the runs, networks go wrong together.

    **Next:** convolutional networks: the same units, with weights shared across an image.
    """)
    return


if __name__ == "__main__":
    app.run()
