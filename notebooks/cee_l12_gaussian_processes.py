"""Applied Machine Learning for CEE — Lecture 12: Gaussian Processes for Optimization.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l12_gaussian_processes.py

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
    # Applied Machine Learning for CEE — Lecture 12: Gaussian Processes for Optimization

    **Slides:** *Applied Machine Learning for CEE*, Lecture 12 (`lectures/cee_l12_gaussian_processes`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, Ch. 2 and §5.4; Frazier (2018), *A Tutorial on Bayesian Optimization*; Forsyth, *Applied Machine Learning* (Springer 2019), §4.4 (the multivariate normal).
    **Data:** BattLeDIM's L-Town EPANET model (Vrachimis et al., 2022, CC BY 4.0), run with WNTR (Klise et al., 2017). Fetch with `bash scripts/fetch_battledim.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`, except the time
    of one run, which depends on the machine. It runs the simulator a few thousand times, in 4
    parallel processes: about 12 minutes on the CPU.

    **Your tasks** (from the slides): reason about four runs by hand, write the Gaussian process
    and match scikit-learn, run Bayesian optimization on the simulator, then break it.
    """)
    return


@app.cell
def _():
    import contextlib
    import os
    import random
    import tempfile
    import time
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import torch
    import wntr
    from joblib import Parallel as JobParallel
    from joblib import delayed
    from scipy.linalg import cho_solve, cholesky, solve_triangular
    from scipy.optimize import minimize
    from scipy.stats import norm, qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from threadpoolctl import threadpool_limits

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # One BLAS thread, as in the slides' figure script: the same sums in the same order, so every
    # search picks the same candidates.
    threadpool_limits(1)

    TODAY = (40.0, 50.0)  # PRV-1, PRV-2 settings in L-TOWN.inp, m
    P_SERVICE = 35 * 6894.757 / (1000.0 * 9.80665)  # 35 psi in metres of water, 24.6 m
    LO, HI = 25.0, 60.0  # search box for both settings, m
    BUDGET, N_INIT, SEEDS = 15, 5, range(10)
    CAND_STEP, MAP_STEP = 0.5, 2.5
    JITTER = 1e-6  # added to the standardized kernel diagonal: the simulator is deterministic
    SLICE_S1, SLICE_RUNS = 36.0, (30.0, 40.0, 50.0, 60.0)
    # At most 4 processes. No number depends on the count: each task runs in a worker with its
    # own calibrated model and one BLAS thread, and the results are put back in task order.
    WORKERS = min(4, os.cpu_count() or 1)
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    return (
        BUDGET,
        CAND_STEP,
        ConstantKernel,
        GaussianProcessRegressor,
        HI,
        JITTER,
        JobParallel,
        LO,
        MAP_STEP,
        Matern,
        N_INIT,
        PALETTE,
        P_SERVICE,
        Path,
        SEED,
        SEEDS,
        SLICE_RUNS,
        SLICE_S1,
        TODAY,
        WORKERS,
        cho_solve,
        cholesky,
        contextlib,
        dataclass,
        delayed,
        minimize,
        norm,
        np,
        nx,
        pd,
        plt,
        qmc,
        solve_triangular,
        tempfile,
        threadpool_limits,
        time,
        wntr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 2–11

    On your own, no notes, two minutes:

    1. k-NN predicted a new mix from its five nearest tests (Lecture 2). What did it tell you
       about how sure it was?
    2. Ridge regression adds $\lambda\lVert\mathbf{w}\rVert^2$ to the loss (Lecture 5). What does
       $\lambda$ trade?
    3. How many calibration scores does a 90% split-conformal interval need (Lecture 11)?

    **Today:** fifteen simulator runs to set two valves.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. Nothing: the spread of the five neighbors is a hint, not an interval. Today's
                   predictor also weighs past cases by similarity, and reports how sure it is.
                2. Fit against the size of the weights: a little bias for much less variance.
                   Today's mean is kernel ridge regression.
                3. At least 9. With 5 runs there is no 90% conformal interval; today's interval
                   comes from a model of the function, and is only as good as that model.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    L-Town's Area A (657 junctions) is fed through two pressure-reducing valves, PRV-1 at 40 m
    and PRV-2 at 50 m. **The decision:** lower settings leak less, but every customer must keep
    35 psi (24.6 m), the normal working pressure of the Ten States Standards. **The simulator:**
    BattLeDIM's EPANET model of L-Town, one peak day, run with WNTR. **The budget:** 15 runs; a
    calibrated city model can take minutes to hours per run.

    **The assumption about leakage**, stated on the slide: background leakage at every Area A
    junction, $q_i=C_i\,p_i^{N_1}$ with $N_1=1$ (an EPANET emitter), $C_i$ proportional to the
    pipe length the junction serves, scaled so that today's settings leak BattLeDIM's mean 2018
    leak flow. Here one run takes well under a second (the slides' figure script timed 0.38 s a run;
    the cell after the model times a few on your machine), so every answer is checked against
    the simulator itself.
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
    STEPS = DAY_S // STEP_S + 1  # reported steps of one day, 00:00 to 24:00

    @dataclass(frozen=True)
    class Run:
        """One simulated day: Area A leakage (m3/h), lowest customer pressure in A and B (m)."""

        leak: float
        pmin: float
        node: str
        hour: float
        open_steps: tuple
        pmin_c: float

    def mean_leak_2018() -> float:
        """Mean total leak flow of BattLeDIM's 2018 leaks, m3/h."""
        d = pd.read_csv(RAW / "2018_Leakages.csv", sep=";", decimal=",", index_col=0)
        assert d.shape == (105_120, 14), d.shape
        return float(d.sum(axis=1).mean())

    class Town:
        """L-Town on its peak day with calibrated background leakage in Area A."""

        def __init__(self, n1: float = 1.0) -> None:
            """Load the model, find the zones and the peak day, calibrate the leakage."""
            self.n1 = n1
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
            opts.hydraulic.emitter_exponent = n1
            index = {n: i for i, n in enumerate(self.area_a)}
            self.length = np.zeros(len(self.area_a))
            for _, p in self.wn.pipes():
                for end in (p.start_node_name, p.end_node_name):
                    if end in index:
                        self.length[index[end]] += p.length / 2
            self.coef = np.zeros(len(self.area_a))
            self.cache = {}
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
            """Area A leakage at each step, m3/s: sum of C_i p_i^N1."""
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            return (p**self.n1 * self.coef).sum(axis=1)

        def _calibrate(self) -> None:
            """Scale the coefficients so today's settings leak the 2018 mean leak flow."""
            res = self._simulate(*TODAY)
            demand = res.node["demand"][self.area_a].to_numpy()
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            weight = self.length / self.length.sum()
            total = self.target / 3600 / float((p**self.n1 * weight).sum(axis=1).mean())
            for _ in range(20):
                self.coef = total * weight
                res = self._simulate(*TODAY)
                leak = float(self._leak(res).mean()) * 3600
                if abs(leak / self.target - 1) < 1e-6:
                    break
                total *= self.target / leak
            else:
                raise RuntimeError("leakage calibration did not converge")
            # The formula against the emitter flow EPANET itself reports:
            reported = (res.node["demand"][self.area_a].to_numpy() - demand).sum(axis=1).mean()
            assert abs(reported * 3600 - leak) < 1e-3 * self.target

        def run(self, s1: float, s2: float) -> Run:
            """Simulate the peak day at PRV-1 = s1, PRV-2 = s2 (m); results are cached."""
            key = (round(float(s1), 6), round(float(s2), 6))
            if key not in self.cache:
                res = self._simulate(*key)
                p = res.node["pressure"][self.customers]
                low = p.min()
                node = str(low.idxmin())
                status = res.link["status"][["PRV-1", "PRV-2"]].to_numpy()
                self.cache[key] = Run(
                    leak=float(self._leak(res).mean()) * 3600,
                    pmin=float(low.min()),
                    node=node,
                    hour=float(p[node].idxmin()) / 3600,
                    open_steps=tuple(int(v) for v in (status == 1).sum(axis=0)),
                    pmin_c=float(res.node["pressure"][self.area_c].to_numpy().min()),
                )
            return self.cache[key]

    town = Town()
    _today = town.run(*TODAY)
    assert (len(town.area_a), len(town.area_b), len(town.area_c), town.day) == (657, 31, 92, 3)
    assert (round(town.target, 2), round(_today.leak, 2), round(_today.pmin, 2)) == (
        25.52,
        25.52,
        28.19,
    )
    assert (_today.node, round(_today.hour, 2), round(P_SERVICE, 1)) == ("n50", 19.83, 24.6)
    _c = [town.run(*_s).pmin_c for _s in ((25.0, 25.0), TODAY, (60.0, 60.0))]
    assert max(_c) - min(_c) < 0.01 and round(min(_c), 2) == 24.82  # Area C: the tank's
    # n50, where the pressure is lowest, is the highest customer junction of Areas A and B:
    _elev = {n: town.wn.get_node(n).elevation for n in town.customers}
    assert sum(v > _elev["n50"] for v in _elev.values()) == 0 and round(_elev["n50"], 1) == 44.7
    return STEPS, Town, town


@app.cell
def _(TODAY, np, time, town):
    # How long one run takes here (uncached, one at a time). The slides quote the figure script's
    # timing; this machine's may differ, so it is shown, not asserted.
    _secs = []
    for _pair in (TODAY, (36.0, 47.0), (25.0, 25.0)):
        _t0 = time.perf_counter()
        town._simulate(*_pair)
        _secs.append(time.perf_counter() - _t0)
    f"one simulator run: median {np.median(_secs):.2f} s over {len(_secs)} runs"
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: where does it cross 24.6 m?

    PRV-1 held at 36 m. Four runs of the simulator, peak day:

    | PRV-2 setting (m) | 30 | 40 | 50 | 60 |
    |:------------------|---:|---:|---:|---:|
    | Lowest pressure (m) | 14.89 | 20.11 | 25.73 | 29.33 |

    At what PRV-2 setting does the lowest pressure cross 24.6 m? How sure are you? Where would
    you spend a fifth run?
    """)
    return


@app.cell
def _(P_SERVICE, SLICE_RUNS, SLICE_S1, town):
    attempt = [town.run(SLICE_S1, _s2) for _s2 in SLICE_RUNS]
    assert [round(_r.pmin, 2) for _r in attempt] == [14.89, 20.11, 25.73, 29.33]
    _y40, _y50 = attempt[1].pmin, attempt[2].pmin
    line_cross = 40.0 + (P_SERVICE - _y40) * 10.0 / (_y50 - _y40)  # a straight line
    assert round(line_cross, 1) == 48.0
    return (attempt,)


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                A straight line between the 40 m and 50 m runs crosses at 48.0 m; the simulator
                says 46.77 m (§4 finds it by bisection). The fifth run belongs where the answer
                is uncertain and matters: between 45 and 48 m.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture, with the Gaussian process as the focus step inside a loop: settings are
    scaled to the box (the encoder); the process gives a mean and a band; a decision rule picks
    the next run; the simulator answers; the runs condition the process, and the evidence fits its
    lengthscale. §4 traces the attempt's four runs through it.

    ## 3. Derive

    **Assumption:** outputs at nearby settings are strongly correlated, and "nearby" means the
    same everywhere in the box (a stationary kernel). A Gaussian process says any finite set of
    outputs is jointly Gaussian with covariance $k$; here Matérn 5/2 with one lengthscale per
    setting,

    $$k(r)=\sigma^2\Big(1+\sqrt{5}\,r+\tfrac{5}{3}r^2\Big)e^{-\sqrt{5}\,r},\qquad r^2=\sum_d\frac{(z_d-z'_d)^2}{\ell_d^2}.$$

    **Conditioning** on runs $\mathbf{y}$ at $Z$ (the conditional of a joint Gaussian, Forsyth
    §4.4; Rasmussen & Williams §2.2):

    $$\mu(\mathbf{z}_*)=\mathbf{k}_*^\top(K+\epsilon I)^{-1}\mathbf{y},\qquad\sigma^2(\mathbf{z}_*)=k_{**}-\mathbf{k}_*^\top(K+\epsilon I)^{-1}\mathbf{k}_*.$$

    The mean weighs the runs by $\mathbf{w}=(K+\epsilon I)^{-1}\mathbf{k}_*$; weights can be
    negative. $\epsilon=10^{-6}$ (standardized units) is jitter for a deterministic simulator.

    **The evidence** picks $\ell$ and $\sigma^2$:

    $$\log p(\mathbf{y}\mid\ell,\sigma^2)=-\tfrac12\mathbf{y}^\top K^{-1}\mathbf{y}-\tfrac12\log|K|-\tfrac n2\log2\pi.$$

    **Expected improvement** over the least feasible leakage $f^+$, times the probability that
    the lowest pressure holds (Jones et al., 1998; Gardner et al., 2014):

    $$\mathrm{EI}=(f^+-\mu)\Phi(u)+\sigma\phi(u),\quad u=\frac{f^+-\mu}{\sigma};\qquad P(\text{lowest}\ge24.6)=\Phi\Big(\frac{\mu_p-24.6}{\sigma_p}\Big).$$

    ## 4. Build and verify
    """)
    return


@app.cell
def _(
    HI,
    JITTER,
    LO,
    cho_solve,
    cholesky,
    dataclass,
    minimize,
    np,
    solve_triangular,
):
    def matern52(A, B, ell, var):
        """Matérn 5/2 covariance, one lengthscale per input; shape (len(A), len(B))."""
        d = (A[:, None, :] - B[None, :, :]) / ell  # (n, m, D)
        s = np.sqrt(5.0 * (d**2).sum(axis=-1))  # sqrt(5) r
        return var * (1.0 + s + s**2 / 3.0) * np.exp(-s)

    def gp_posterior(Z, y, Zs, ell, var, eps=1e-6):
        """The slide's listing: posterior mean and sd for standardized outputs y."""
        K = matern52(Z, Z, ell, var) + eps * np.eye(len(Z))
        L = np.linalg.cholesky(K)  # K = L L^T
        alpha = cho_solve((L, True), y)  # K^-1 y
        ks = matern52(Zs, Z, ell, var)  # (m, n)
        v = solve_triangular(L, ks.T, lower=True)  # (n, m)
        sd = np.sqrt(np.clip(var - (v**2).sum(0), 0, None))
        return ks @ alpha, sd

    @dataclass(frozen=True)
    class GP:
        """A fitted process on standardized outputs (mean mu and scale sd removed)."""

        X: np.ndarray
        ell: np.ndarray
        var: float
        mu: float
        sd: float
        chol: np.ndarray
        alpha: np.ndarray
        nlml: float

        def predict(self, Xs):
            """Posterior mean and sd at Xs, in output units."""
            ks = matern52(Xs, self.X, self.ell, self.var)
            v = solve_triangular(self.chol, ks.T, lower=True)
            var = np.clip(self.var - (v**2).sum(axis=0), 0.0, None)
            return self.mu + self.sd * (ks @ self.alpha), self.sd * np.sqrt(var)

        def weights(self, x):
            """Weights w on the standardized outputs at one setting x."""
            ks = matern52(x[None, :], self.X, self.ell, self.var)[0]
            return cho_solve((self.chol, True), ks)

    def condition(X, z, ell, var):
        """Cholesky factor, weights, negative log marginal likelihood."""
        K = matern52(X, X, ell, var) + JITTER * np.eye(len(X))
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), z)
        nlml = 0.5 * z @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(X) * np.log(2 * np.pi)
        return L, alpha, float(nlml)

    ELL_BOUNDS, VAR_BOUNDS = (1e-2, 10.0), (1e-2, 1e2)

    def fit_gp(X, y, ell=None):
        """Maximize the evidence from three starts, or fit only the variance with ell fixed."""
        D = X.shape[1]
        mu, sd = float(y.mean()), float(y.std())
        sd = sd if sd > 10 * np.finfo(float).eps else 1.0  # constant outputs, as scikit-learn
        z = (y - mu) / sd

        def objective(theta):
            ells = np.full(D, ell) if ell is not None else np.exp(theta[:D])
            return condition(X, z, ells, float(np.exp(theta[-1])))[2]

        n_ell = 0 if ell is not None else D
        bounds = [tuple(np.log(ELL_BOUNDS))] * n_ell + [tuple(np.log(VAR_BOUNDS))]
        best = None
        for start in (0.1, 0.5, 2.0) if ell is None else (1.0,):
            res = minimize(
                objective, np.log([start] * n_ell + [1.0]), method="L-BFGS-B", bounds=bounds
            )
            if best is None or res.fun < best.fun:
                best = res
        ells = np.full(D, ell) if ell is not None else np.exp(best.x[:D])
        var = float(np.exp(best.x[-1]))
        L, alpha, nlml = condition(X, z, ells, var)
        return GP(X, ells, var, mu, sd, L, alpha, nlml)

    def unit(x):
        """Valve settings (m) scaled to the unit box."""
        return (np.asarray(x, dtype=np.float64) - LO) / (HI - LO)

    return GP, ELL_BOUNDS, VAR_BOUNDS, fit_gp, gp_posterior, matern52, unit


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    scikit-learn's `GaussianProcessRegressor`, same kernel and hyperparameters: the largest
    difference in the mean? And when both choose the lengthscale themselves?
    """)
    return


@app.cell
def _(
    BUDGET,
    ConstantKernel,
    ELL_BOUNDS,
    GaussianProcessRegressor,
    HI,
    JITTER,
    LO,
    Matern,
    N_INIT,
    SEED,
    SLICE_RUNS,
    VAR_BOUNDS,
    attempt,
    fit_gp,
    gp_posterior,
    np,
    qmc,
    town,
    unit,
):
    def check_sklearn(X, y, gp, Xs):
        """Our process against scikit-learn's: fixed hyperparameters, then each optimizing."""
        fixed = ConstantKernel(gp.var, "fixed") * Matern(gp.ell, "fixed", nu=2.5)
        free = ConstantKernel(1.0, VAR_BOUNDS) * Matern(np.ones(X.shape[1]), ELL_BOUNDS, nu=2.5)
        ref = GaussianProcessRegressor(fixed, alpha=JITTER, normalize_y=True, optimizer=None)
        ref.fit(X, y)
        m_ref, s_ref = ref.predict(Xs, return_std=True)
        m, s = gp.predict(Xs)
        opt = GaussianProcessRegressor(
            free, alpha=JITTER, normalize_y=True, n_restarts_optimizer=5, random_state=SEED
        ).fit(X, y)
        out = {
            "mean": float(np.abs(m - m_ref).max()),
            "sd": float(np.abs(s - s_ref).max()),
            "lml": float(abs(-gp.nlml - ref.log_marginal_likelihood_value_)),
            "opt": float(-gp.nlml - opt.log_marginal_likelihood_value_),
        }
        # The same optimum both ways: a much better optimum than scikit-learn's is not "the same".
        assert max(out["mean"], out["sd"], out["lml"]) < 1e-11 and abs(out["opt"]) < 1e-8, out
        return out

    def initial_design(seed):
        """Five space-filling settings (Latin hypercube), rounded to 0.1 m."""
        pts = qmc.scale(qmc.LatinHypercube(d=2, seed=seed).random(N_INIT), [LO, LO], [HI, HI])
        return [(round(float(a), 1), round(float(b), 1)) for a, b in pts]

    def random_search(seed):
        """Fifteen settings drawn uniformly over the box, rounded to 0.1 m."""
        pts = np.random.default_rng(seed).uniform(LO, HI, (BUDGET, 2))
        return [(round(float(a), 1), round(float(b), 1)) for a, b in pts]

    # The attempt's slice, then fifteen runs in two settings.
    _X1 = unit(np.array(SLICE_RUNS)[:, None])
    _y1 = np.array([r.pmin for r in attempt])
    gp_slice = fit_gp(_X1, _y1)
    _xs = unit(np.arange(LO, HI + 1e-9, 0.5)[:, None])
    checks = {"slice": check_sklearn(_X1, _y1, gp_slice, _xs)}
    # The slide's listing gives the same posterior (standardized outputs in, then rescaled):
    _m, _s = gp_posterior(_X1, (_y1 - gp_slice.mu) / gp_slice.sd, _xs, gp_slice.ell, gp_slice.var)
    assert np.allclose(gp_slice.mu + gp_slice.sd * _m, gp_slice.predict(_xs)[0], atol=1e-10)
    _X2 = initial_design(SEED) + random_search(SEED)[: BUDGET - N_INIT]
    _runs = [town.run(*x) for x in _X2]
    _g = np.arange(LO, HI + 1e-9, 0.5)
    _cand = np.array([(a, b) for a in _g for b in _g])
    ells = {}
    for _name, _y in (("leak", [r.leak for r in _runs]), ("pmin", [r.pmin for r in _runs])):
        _gp = fit_gp(unit(_X2), np.array(_y))
        checks[_name] = check_sklearn(unit(_X2), np.array(_y), _gp, unit(_cand))
        ells[_name] = [round(float(v) * (HI - LO)) for v in _gp.ell]
    assert ells == {"leak": [37, 45], "pmin": [29, 22]}, ells
    checks
    return gp_slice, initial_design, random_search


@app.cell
def _(HI, LO, P_SERVICE, attempt, gp_slice, norm, np, unit):
    # The diagram's trace: the four runs, one prediction at PRV-2 = 47 m.
    _xs = unit(np.array([[47.0]]))
    _m, _s = gp_slice.predict(_xs)
    _w = gp_slice.weights(_xs[0])
    _y = np.array([round(r.pmin, 2) for r in attempt])
    _ybar = round(gp_slice.mu, 2)
    assert [round(float(v), 3) for v in _w] == [-0.054, 0.33, 0.8, -0.078]
    assert (round(float(_m[0]), 2), round(float(_s[0]), 2)) == (24.17, 0.09)
    assert (round(float(_m[0] - 2 * _s[0]), 1), round(float(_m[0] + 2 * _s[0]), 1)) == (24.0, 24.4)
    assert round(float(gp_slice.ell[0]) * (HI - LO)) == 48 and _ybar == 22.51
    # The diagram's arithmetic reproduces from its rounded numbers:
    _shown = _ybar + sum(round(float(w), 3) * (y - _ybar) for w, y in zip(_w, _y, strict=True))
    assert round(_shown, 1) == round(float(_m[0]), 1) == 24.2
    assert round(float(norm.cdf((_m[0] - P_SERVICE) / _s[0])), 3) == 0.0  # "sure" it fails
    return


@app.cell
def _(
    BUDGET,
    CAND_STEP,
    HI,
    LO,
    P_SERVICE,
    fit_gp,
    initial_design,
    norm,
    np,
    unit,
):
    def candidates():
        """Every pair on a 0.5 m lattice over the box (5,041)."""
        g = np.arange(LO, HI + 1e-9, CAND_STEP)
        return np.array([(a, b) for a in g for b in g])

    def fit_both(town, X):
        """Processes for leakage and for the lowest pressure."""
        runs = [town.run(*x) for x in X]
        U = unit(X)
        return (
            fit_gp(U, np.array([r.leak for r in runs])),
            fit_gp(U, np.array([r.pmin for r in runs])),
        )

    def probability_feasible(g_p, U):
        """P(lowest pressure >= 24.6 m)."""
        m, s = g_p.predict(U)
        return norm.cdf((m - P_SERVICE) / np.maximum(s, 1e-12))

    def bo_score(g_leak, g_p, U, best):
        """Expected improvement in leakage times the probability of feasibility."""
        pof = probability_feasible(g_p, U)
        if best is None:
            return pof
        m, s = g_leak.predict(U)
        s = np.maximum(s, 1e-12)
        u = (best - m) / s
        return ((best - m) * norm.cdf(u) + s * norm.pdf(u)) * pof

    def straddle_score(g_leak, g_p, U, best):
        """Bryan et al. (2005): 1.96 sd minus the distance to the boundary."""
        del g_leak, best  # unused: the same signature as bo_score
        m, s = g_p.predict(U)
        return 1.96 * s - np.abs(m - P_SERVICE)

    STRATEGIES = {"bo": bo_score, "straddle": straddle_score}

    def best_feasible(runs, p_req=P_SERVICE):
        """Least leakage among runs meeting p_req, or None."""
        ok = [r.leak for r in runs if r.pmin >= p_req]
        return min(ok) if ok else None

    def sequential(town, seed, name):
        """Five space-filling runs, then ten chosen by the strategy's score."""
        cand = candidates()
        U = unit(cand)
        X = initial_design(seed)
        while len(X) < BUDGET:
            g_leak, g_p = fit_both(town, X)
            score = STRATEGIES[name](g_leak, g_p, U, best_feasible([town.run(*x) for x in X]))
            done = {(round(a, 6), round(b, 6)) for a, b in X}
            score[[(round(a, 6), round(b, 6)) in done for a, b in cand]] = -np.inf
            a, b = cand[int(np.argmax(score))]
            X.append((float(a), float(b)))
        return X

    def boundary_s2(town, s1, p_req=P_SERVICE, tol=0.02):
        """Lowest PRV-2 setting in the box that meets p_req (bisection), or None."""
        if town.run(s1, HI).pmin < p_req:
            return None
        if town.run(s1, LO).pmin >= p_req:
            return LO
        a, b = LO, HI
        while b - a > tol:
            m = 0.5 * (a + b)
            a, b = (a, m) if town.run(s1, m).pmin >= p_req else (m, b)
        return b

    def lattice_boundary(town, s1, p_req=P_SERVICE):
        """Lowest PRV-2 setting on the candidates' 0.5 m lattice that meets p_req, or None.

        The bisection lands within its tolerance above the boundary, so the answer is the
        lattice point at or above it, or the one just below.
        """
        b = boundary_s2(town, s1, p_req)
        if b is None:
            return None
        up = round(LO + np.ceil((b - LO) / CAND_STEP - 1e-9) * CAND_STEP, 6)
        for s2 in (round(up - CAND_STEP, 6), up):
            if s2 >= LO and town.run(s1, s2).pmin >= p_req:
                return s2
        raise RuntimeError(f"PRV-2 = {up} m infeasible above the boundary at PRV-1 = {s1} m")

    return (
        STRATEGIES,
        best_feasible,
        bo_score,
        boundary_s2,
        candidates,
        fit_both,
        lattice_boundary,
        probability_feasible,
        sequential,
    )


@app.cell
def _(
    JobParallel,
    P_SERVICE,
    Town,
    WORKERS,
    boundary_s2,
    delayed,
    lattice_boundary,
    np,
    sequential,
    threadpool_limits,
):
    def _key(p):
        return (round(float(p[0]), 6), round(float(p[1]), 6))

    def _work(n1, tasks):
        """One worker process: build its own model, run a batch of tasks, return the results.

        Processes, not threads: WNTR's EPANET library is not safe to call from several threads.
        """
        threadpool_limits(1)  # the same sums as the slides' script, in every worker
        t = Town(n1=n1)
        out = []
        for kind, args in tasks:
            if kind == "run":
                out.append((None, {_key(args): t.run(*args)}))
            elif kind in ("boundary", "lattice"):
                s1, p_req = args
                s2 = (boundary_s2 if kind == "boundary" else lattice_boundary)(t, s1, p_req)
                out.append((s2, {} if s2 is None else {_key((s1, s2)): t.run(s1, s2)}))
            else:
                name, seed = args
                X = sequential(t, seed, name)
                out.append((X, {_key(x): t.run(*x) for x in X}))
        return out

    class Workers:
        """Parallel simulation in ``WORKERS`` processes; their runs join ``town``'s cache."""

        COARSE = np.arange(32.0, 44.01, 1.0)

        def __init__(self, town):
            """Keep the model whose cache receives the results."""
            self.town = town

        def map(self, tasks):
            """Run tasks in parallel (dealt round-robin); return their values in order."""
            batches = [tasks[i::WORKERS] for i in range(WORKERS)]
            done = JobParallel(n_jobs=WORKERS)(
                delayed(_work)(self.town.n1, b) for b in batches if b
            )
            values = [None] * len(tasks)
            for i, batch in enumerate(done):
                for j, (value, runs) in enumerate(batch):
                    self.town.cache.update(runs)
                    values[i + j * WORKERS] = value
            return values

        def runs(self, pairs):
            """Simulate independent settings (cached ones are not run again)."""
            pairs = [(float(a), float(b)) for a, b in pairs]
            todo = sorted({_key(p) for p in pairs} - set(self.town.cache))
            self.map([("run", p) for p in todo])
            return [self.town.run(*p) for p in pairs]

        def optimum(self, p_req=P_SERVICE):
            """Least leakage on the boundary: PRV-1 every 1 m, then every 0.1 m around the best."""

            def scan(s1s):
                s1s = [round(float(v), 6) for v in s1s]
                s2s = self.map([("boundary", (s1, p_req)) for s1 in s1s])
                leak = [
                    np.inf if s2 is None else self.town.run(s1, s2).leak
                    for s1, s2 in zip(s1s, s2s, strict=True)
                ]
                i = int(np.argmin(leak))
                if not np.isfinite(leak[i]):
                    raise RuntimeError(f"no feasible PRV-2 setting for p_req = {p_req}")
                if not 0 < i < len(s1s) - 1:  # the best must lie inside each scan
                    raise RuntimeError(f"optimum PRV-1 {s1s[i]} m at the edge of {s1s}")
                return s1s[i], float(s2s[i])

            s1, _ = scan(self.COARSE)
            return scan(np.arange(s1 - 0.9, s1 + 0.9 + 1e-9, 0.1))

        def lattice_best(self, p_req=P_SERVICE):
            """Least leakage among feasible settings on the 0.5 m lattice (what BO can pick)."""
            s1s = [round(float(v), 6) for v in np.arange(32.0, 44.0 + 1e-9, 0.5)]
            s2s = self.map([("lattice", (s1, p_req)) for s1 in s1s])
            leak = [
                np.inf if s2 is None else self.town.run(s1, s2).leak
                for s1, s2 in zip(s1s, s2s, strict=True)
            ]
            i = int(np.argmin(leak))
            if not 0 < i < len(s1s) - 1:
                raise RuntimeError(f"lattice optimum PRV-1 {s1s[i]} m at the edge of the scan")
            return s1s[i], float(s2s[i])

    return (Workers,)


@app.cell
def _(HI, LO, MAP_STEP, TODAY, Workers, np, town):
    pool = Workers(town)
    _g = np.arange(LO, HI + 1e-9, MAP_STEP)
    grid_pts = np.stack(np.meshgrid(_g, _g, indexing="ij"), axis=-1)  # (15, 15, 2)
    truth = np.array([[r.leak, r.pmin] for r in pool.runs(grid_pts.reshape(-1, 2))])
    truth = truth.reshape(*grid_pts.shape[:2], 2)
    opt = pool.optimum()
    best_true = town.run(*opt).leak
    _today = town.run(*TODAY).leak
    assert (round(opt[0], 2), round(opt[1], 2), round(best_true, 2)) == (35.9, 46.89, 23.6)
    assert round(100 * (1 - best_true / _today), 1) == 7.5 and round(_today - best_true, 2) == 1.92
    # A lengthscale is a correlation distance, not a rate. The rate, from the map: the mean change
    # in the lowest pressure between neighbouring map points, per metre of each setting.
    _slope = [
        round(float(np.abs(np.diff(truth[..., 1], axis=_a)).mean()) / MAP_STEP, 3) for _a in (0, 1)
    ]
    assert _slope == [0.471, 0.308], _slope
    return best_true, grid_pts, opt, pool, truth


@app.cell
def _(
    P_SERVICE,
    STRATEGIES,
    SEEDS,
    best_feasible,
    best_true,
    candidates,
    np,
    pool,
    random_search,
    town,
):
    _tasks = [("search", (name, seed)) for name in STRATEGIES for seed in SEEDS]
    found = {name: [] for name in STRATEGIES}
    for (_, (_name, _)), _X in zip(_tasks, pool.map(_tasks), strict=True):
        found[_name].append(_X)
    found["random"] = [random_search(seed) for seed in SEEDS]
    _g = [25.0, 25.0 + 35 / 3, 25.0 + 70 / 3, 60.0]
    found["grid"] = [(a, b) for a in _g for b in _g]
    pool.runs([x for X in found["random"] for x in X] + found["grid"])

    def _regret(X):
        """Least feasible leakage among X's runs, minus the best possible (inf if none)."""
        _b = best_feasible([town.run(*x) for x in X])
        return np.inf if _b is None else _b - best_true

    assert len(candidates()) == 5041  # the 0.5 m lattice, 71 x 71

    _bo = sorted(_regret(X) for X in found["bo"])
    _rnd = sorted(_regret(X) for X in found["random"])
    _grid_best = best_feasible([town.run(*x) for x in found["grid"]])
    assert (round(float(_bo[4] / 2 + _bo[5] / 2), 2), round(_bo[-1], 2)) == (0.05, 0.16)
    assert round(float(_rnd[4] / 2 + _rnd[5] / 2), 2) == 1.57
    assert (round(_grid_best, 2), round(_grid_best - best_true, 2)) == (24.18, 0.58)
    # The bracket below: the 0.5 m lattice's best feasible setting is BO's best, 0.05 above
    _lat = pool.lattice_best()
    assert _lat == (35.5, 47.5) and round(town.run(*_lat).leak - best_true, 2) == 0.05
    _X0 = found["bo"][0]
    _r0 = [town.run(*x) for x in _X0]
    _j = min(range(len(_r0)), key=lambda i: _r0[i].leak if _r0[i].pmin >= P_SERVICE else 1e9)
    assert (_X0[_j], round(_r0[_j].leak, 2), round(_r0[_j].pmin - P_SERVICE, 2)) == (
        (35.5, 47.5),
        23.65,
        0.02,
    )
    return (found,)


@app.cell
def _(
    ELL_BOUNDS,
    HI,
    JITTER,
    LO,
    N_INIT,
    P_SERVICE,
    SLICE_RUNS,
    SLICE_S1,
    attempt,
    best_feasible,
    bo_score,
    cho_solve,
    cholesky,
    fit_both,
    fit_gp,
    found,
    gp_slice,
    matern52,
    norm,
    np,
    town,
    unit,
):
    # The pictures behind the slides: the evidence's two terms, the leakage process after 5 and
    # 15 runs of one search, and the acquisition surface that picks run 6.
    _y = np.array([r.pmin for r in attempt])
    _X = unit(np.array(SLICE_RUNS)[:, None])
    _z = (_y - gp_slice.mu) / gp_slice.sd
    _ells = np.linspace(5.0, 120.0, 116)  # m
    _nlml = []
    for _e in _ells:  # signal variance held at the fitted one, so the two terms trade visibly
        _K = matern52(_X, _X, np.array([_e / (HI - LO)]), gp_slice.var) + JITTER * np.eye(4)
        _L = cholesky(_K, lower=True)
        _nlml.append(0.5 * _z @ cho_solve((_L, True), _z) + np.log(np.diag(_L)).sum())
    assert round(float(_ells[int(np.argmin(_nlml))])) == 48 and SLICE_S1 == 36.0

    _X0 = found["bo"][0]
    _g = np.linspace(LO, HI, 41)
    _U = unit(np.array([(a, b) for b in _g for a in _g]))
    _med = []
    for _n in (N_INIT, len(_X0)):
        _gp = fit_gp(unit(_X0[:_n]), np.array([town.run(*x).leak for x in _X0[:_n]]))
        _med.append(round(float(np.median(_gp.predict(_U)[1])), 2))
        if _n == N_INIT:  # five runs cannot see PRV-2's effect: its lengthscale hits the bound
            assert np.isclose(_gp.ell[1], ELL_BOUNDS[1]) and round(_gp.ell[0] * (HI - LO), 1) == 6.5
    assert _med == [1.57, 0.4], _med

    _gl, _gp_p = fit_both(town, _X0[:N_INIT])
    _best5 = best_feasible([town.run(*x) for x in _X0[:N_INIT]])
    _g71 = np.linspace(LO, HI, 71)
    _P = np.array([(a, b) for b in _g71 for a in _g71])
    _score = bo_score(_gl, _gp_p, unit(_P), _best5)
    _i = int(np.argmax(_score))
    assert tuple(_P[_i]) == _X0[N_INIT] == (39.5, 25.0)
    assert round(_best5, 2) == 25.95 and round(float(_score.max()), 2) == 1.22
    _m5, _s5 = _gl.predict(unit(_P[_i : _i + 1]))
    _u = (_best5 - _m5) / _s5
    _ei = float(((_best5 - _m5) * norm.cdf(_u) + _s5 * norm.pdf(_u))[0])
    _mp, _sp = _gp_p.predict(unit(_P[_i : _i + 1]))
    _pof = float(norm.cdf((_mp - P_SERVICE) / _sp)[0])
    assert (round(_ei, 2), round(_pof, 2)) == (2.55, 0.48)
    _col = _score[_P[:, 0] == 39.5]  # the peak's PRV-1 setting, every PRV-2 setting
    assert (round(float(_col.min()), 2), round(float(_col.max()), 2)) == (1.21, 1.22)
    return


@app.cell
def _(BUDGET, PALETTE, TODAY, best_feasible, best_true, found, np, plt, town):
    # Least feasible leakage after each run: Bayesian optimization against random settings.
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _n = np.arange(1, BUDGET + 1)
    for _name, _c, _mk in (("bo", PALETTE[0], "o"), ("random", PALETTE[1], "s")):
        _curves = np.array(
            [
                [
                    np.inf
                    if (b := best_feasible([town.run(*x) for x in X[: i + 1]])) is None
                    else b
                    for i in range(BUDGET)
                ]
                for X in found[_name]
            ]
        )
        _med = np.median(_curves, axis=0)
        _ax.plot(_n, np.where(np.isfinite(_med), _med, np.nan), marker=_mk, color=_c, label=_name)
    _ax.axhline(town.run(*TODAY).leak, color="k", ls="--", label="today's settings")
    _ax.axhline(best_true, color="k", ls=":", label="best possible")
    _ax.set(xlabel="Simulator runs", ylabel="Least feasible leakage (m³/h)")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict first:** (1) With four runs, is the ±2σ band at 47 m honest? (2) What does the
    process predict for PRV-2 settings up to 90 m, past the box? (3) Spend 15 runs mapping the
    24.6 m boundary instead: a better map? Better settings? (4) With a leakage exponent of 0.5 or
    1.5 instead of 1, do the best settings move?
    """)
    return


@app.cell
def _(HI, LO, PALETTE, P_SERVICE, SLICE_RUNS, SLICE_S1, fit_gp, np, plt, pool, town, unit):
    # (1) Four runs cannot pin the lengthscale.
    _s2 = np.arange(LO, HI + 1e-9, 1.0)
    _truth = np.array([r.pmin for r in pool.runs([(SLICE_S1, v) for v in _s2])])
    _more = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
    band, _inside_n = {}, {}
    for _ax, (_name, _runs, _ell) in zip(
        _axes,
        (
            ("short", SLICE_RUNS, 2.0 / (HI - LO)),
            ("fitted", SLICE_RUNS, None),
            ("more", _more, None),
        ),
        strict=True,
    ):
        _y = np.array([town.run(SLICE_S1, v).pmin for v in _runs])
        _gp = fit_gp(unit(np.array(_runs)[:, None]), _y, ell=_ell)
        _m, _s = _gp.predict(unit(_s2[:, None]))
        _off = ~np.isin(_s2, _runs)  # between the runs: at a run the process repeats it
        assert _off.sum() == len(_s2) - len(_runs)
        _inside = np.abs(_m[_off] - _truth[_off]) <= 2 * _s[_off]
        band[_name] = (
            round(float(_gp.ell[0]) * (HI - LO)),
            round(float(np.sqrt(np.mean((_m[_off] - _truth[_off]) ** 2))), 2),
            round(float(np.mean(_inside)), 3),
        )
        _inside_n[_name] = (int(_inside.sum()), int(_off.sum()))
        _ax.fill_between(_s2, _m - 2 * _s, _m + 2 * _s, color=PALETTE[0], alpha=0.2)
        _ax.plot(_s2, _m, color=PALETTE[0])
        _ax.plot(_s2, _truth, "k--")
        _ax.plot(_runs, _y, "o", color=PALETTE[1])
        _ax.axhline(P_SERVICE, color="grey", ls=":")
        _ax.set_title(f"{_name}: ℓ = {band[_name][0]} m")
    assert band == {"short": (2, 3.37, 1.0), "fitted": (48, 0.94, 0.0), "more": (17, 0.3, 0.828)}
    assert _inside_n["more"] == (24, 29)  # seven runs: the band holds the truth at 24 of 29
    _flat = _s2[np.abs(_truth - _truth[0]) < 0.01]
    assert _flat.max() == 33.0  # flat up to 33 m: a feature four runs cannot reveal
    _fig
    return


@app.cell
def _(HI, LO, PALETTE, SLICE_S1, STEPS, fit_gp, found, np, plt, pool, town, unit):
    # (2) Beyond the runs: PRV-2 up to 90 m.
    _s2 = np.arange(LO, 90.0 + 1e-9, 1.0)
    _truth = pool.runs([(SLICE_S1, v) for v in _s2])
    _X = found["bo"][0]
    _g = fit_gp(unit(_X), np.array([town.run(*x).leak for x in _X]))
    _m, _s = _g.predict(unit(np.column_stack([np.full_like(_s2, SLICE_S1), _s2])))
    _leak = np.array([r.leak for r in _truth])
    _open = _s2[np.array([r.open_steps[1] for r in _truth]) == STEPS]
    _i = int(np.argmin(np.abs(_s2 - 90.0)))
    assert (_open.min(), round(_leak[_i], 2), round(_m[_i], 2), round(_s[_i], 2)) == (
        75.0,
        35.65,
        32.86,
        5.67,
    )
    assert abs(_leak[_i] - _m[_i]) <= 2 * _s[_i]  # the band is honest; the mean is not
    _fig, _ax = plt.subplots(figsize=(5.2, 3.6))
    _ax.axvspan(HI, 90, color="grey", alpha=0.1)
    _ax.fill_between(_s2, _m - 2 * _s, _m + 2 * _s, color=PALETTE[0], alpha=0.2)
    _ax.plot(_s2, _m, color=PALETTE[0], label="GP mean, ±2 sd")
    _ax.plot(_s2, _leak, "k--", label="simulator")
    _ax.set(xlabel="PRV-2 setting (m); PRV-1 at 36 m", ylabel="Leakage (m³/h)")
    _ax.legend()
    _fig
    return


@app.cell
def _(P_SERVICE, best_feasible, best_true, fit_gp, found, grid_pts, np, pool, town, truth, unit):
    # (3) The question picks the runs: optimum (BO) or boundary (straddle). Each search's map
    # errors count only the map points it did not run: at a run the process repeats the
    # simulator, so that point would count as right whatever the process does elsewhere.
    _P = grid_pts.reshape(-1, 2)
    _feasible = truth[..., 1].ravel() >= P_SERVICE
    assert int(_feasible.sum()) == 134
    table = {}
    for _name in ("bo", "straddle", "random"):
        _reg, _wrong = [], []
        for _X in found[_name]:
            _runs = [town.run(*x) for x in _X]
            _b = best_feasible(_runs)
            _reg.append(np.inf if _b is None else _b - best_true)
            _ran = {(round(a, 6), round(b, 6)) for a, b in _X}
            _off = np.array([(round(a, 6), round(b, 6)) not in _ran for a, b in _P])
            _m = fit_gp(unit(_X), np.array([r.pmin for r in _runs])).predict(unit(_P[_off]))[0]
            _wrong.append(100 * float(np.mean((_m >= P_SERVICE) != _feasible[_off])))
        table[_name] = (
            round(float(np.median(_reg)), 2),
            (round(float(min(_reg)), 2), round(float(max(_reg)), 2)),
            round(float(np.median(_wrong)), 1),
            (round(min(_wrong), 1), round(max(_wrong), 1)),
        )
    assert table == {
        "bo": (0.05, (0.05, 0.16), 2.9, (0.9, 8.1)),
        "straddle": (1.25, (0.44, 2.47), 1.6, (0.9, 3.2)),
        "random": (1.57, (0.68, 2.75), 3.1, (1.3, 6.7)),
    }, table
    # The boundary's corner: from some PRV-1 setting on, any PRV-2 setting keeps 24.6 m.
    _s1s = [44.0, 45.0, 46.0, 47.0, 48.0, 49.0]
    _s2s = pool.map([("boundary", (s1, P_SERVICE)) for s1 in _s1s])
    corner = next((s1 for s1, s2 in zip(_s1s, _s2s, strict=True) if s2 == 25.0), None)
    if corner is None:
        raise RuntimeError(f"no corner in PRV-1 {_s1s}: boundary {_s2s}")
    assert corner == 47.0 and round(_s2s[2], 2) == 43.17  # from 47 m on; at 46 m, 43.17 m
    # Beyond the corner the least feasible leakage on the map is far above the optimum, and BO's
    # first search spends none of its ten guided runs past PRV-1 = 46 m:
    _beyond = (grid_pts[..., 0] >= corner) & (truth[..., 1] >= P_SERVICE)
    assert round(float(truth[..., 0][_beyond].min()), 2) == 26.59
    assert sum(x[0] > 46.0 for x in found["bo"][0][5:]) == 0
    table
    return


@app.cell
def _(P_SERVICE, TODAY, fit_gp, found, np, pool, probability_feasible, town, unit):
    # The GP is sure; the town is not. Then: what does a margin cost?
    _X = found["bo"][0]
    _g_p = fit_gp(unit(_X), np.array([town.run(*x).pmin for x in _X]))
    _x = (35.5, 47.5)
    _s = float(_g_p.predict(unit([_x]))[1][0])
    assert float(f"{_s:.1g}") == 0.005
    assert round(float(probability_feasible(_g_p, unit([_x]))[0]), 2) == 1.0
    _today = town.run(*TODAY)
    margin = {}
    for _extra in (0.0, 1.0, 2.0):
        _o = pool.optimum(P_SERVICE + _extra)
        margin[_extra] = (round(_o[0], 2), round(_o[1], 2), round(town.run(*_o).leak, 2))
    assert margin == {
        0.0: (35.9, 46.89, 23.6),
        1.0: (36.9, 47.92, 24.14),
        2.0: (37.9, 48.94, 24.67),
    }
    assert round(_today.pmin - P_SERVICE, 2) == 3.58
    _same = pool.optimum(_today.pmin)
    _same_leak = town.run(*_same).leak
    assert (round(_same[0], 2), round(_same[1], 2), round(_same_leak, 2)) == (39.5, 50.55, 25.51)
    assert round(_today.leak - _same_leak, 2) == 0.01  # today's settings: already on the frontier
    # ... to the search's resolution: 0.02 m less PRV-2 (one bisection bracket) changes leakage by
    _res = _same_leak - town.run(_same[0], _same[1] - 0.02).leak
    assert round(_res, 3) == 0.005, _res
    _bo_best = town.run(*_x).leak
    _m1 = pool.optimum(P_SERVICE + 1.0)
    _saves = [round(100 * (1 - town.run(*o).leak / _today.leak), 1) for o in (_x, _m1)]
    assert _saves == [7.3, 5.4] and round(_bo_best, 2) == 23.65, _saves
    margin
    return


@app.cell
def _(TODAY, Town, Workers):
    # (4) The leak model sets the prize; the settings move by under 0.1 m.
    exponent = {}
    for _n1 in (0.5, 1.5):
        _t = Town(n1=_n1)
        _o = Workers(_t).optimum()
        _today, _best = _t.run(*TODAY).leak, _t.run(*_o).leak
        exponent[_n1] = (round(_o[0], 2), round(_o[1], 2), round(100 * (1 - _best / _today), 1))
    assert exponent == {0.5: (35.9, 46.93, 3.9), 1.5: (35.9, 46.84, 11.0)}, exponent
    # Across N1 = 0.5, 1, 1.5 the best PRV-2 setting moves by under 0.1 m (46.89 m at N1 = 1):
    assert max(_v[1] for _v in exponent.values()) - min(_v[1] for _v in exponent.values()) < 0.1
    exponent
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    At BO's best run the process is sure: 0.02 m to spare, σ = 0.005 m, P(feasible) = 1.00. That
    certainty is about the simulator, not the town. The least leakage with 0, 1, and 2 m to spare
    is 23.60, 24.14, and 24.67 m³/h; today's settings keep 3.58 m and leak 25.52, and the best
    settings with that same margin leak 25.51. **Today's settings were already on the frontier:
    the saving is the margin given up.** Set the margin from the model's error against the town's
    gauges, check a week and fire flows, then lower in steps with the pressure at n50 logged.

    **When not to use a GP:** many settings (beyond about 20; Frazier, 2018); many runs ($O(n^3)$);
    regimes and jumps (a valve opens fully); a cheap simulator, where a grid is simpler and exact.

    **Impact.** Lower pressure saves water and the energy that treated and pumped it; the margin
    is thinnest at n50, the highest customer junction (44.7 m), and those customers carry the
    risk of a model error.

    ### Vote: which settings for tomorrow?

    **A.** BO's best run, (35.5, 47.5) m. **B.** The best settings with 1 m to spare. **C.** Keep
    today's (40, 50) m. **D.** B, after a week-long run with fire flows and a logged trial at n50.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                D. A trusts the simulator to 2 cm; C keeps a margin nobody chose; B is the right
                kind of answer, but its margin must come from the model's error against the
                town. What would change the answer: a gauge at n50 that reads 1.5 m below the
                model, or a planned fire-flow test.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where Bayesian optimization lives today

    - **Hyperparameters** of learning systems (Snoek, Larochelle & Adams, 2012); Google Vizier
      (Golovin et al., 2017); BoTorch (Balandat et al., 2020).
    - **Infinitely wide networks are Gaussian processes** (Neal, 1996; Lee et al., 2018).
      Lecture 14 builds a neural surrogate of this simulator.
    - **Self-driving labs** choose the next experiment with the same loop.

    ## 7. Check yourself

    **Exit question.** After 15 runs, the GP says settings (35.5, 47.5) m keep 24.6 m with
    probability 1.00. A colleague proposes deploying them tonight. What does that probability
    mean, what does it not, and what would you do first?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                It is the process's belief about the simulator at a setting it has already run;
                it says nothing about the model's error against the town, which is larger than
                the 0.02 m to spare. First: a margin from the model's error against the gauges,
                a week-long run with fire flows, then lower in steps with logging at n50.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Unscaled inputs or outputs.** Lengthscale bounds and the jitter assume the box is scaled
      to [0, 1] and the outputs standardized.
    - **Inverting $K$.** Use a Cholesky factor and triangular solves; add jitter for a
      deterministic simulator.
    - **Trusting the band with few runs.** Check it: leave one run out, or make a confirming run.
    - **Optimizing past the box.** A stationary kernel cannot know a valve opened.
    - **Simulator scratch files.** WNTR writes `.inp`, `.rpt`, and `.bin` files; give them a
      temporary directory, not the working directory.
    - **BLAS threads and reproducibility.** Ties in the acquisition can depend on summation order;
      fix the thread count.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Rasmussen & Williams (2006), Ch. 2 and §5.4:** Gaussian process regression and the
      marginal likelihood; §4.2 for the Matérn family.
    - **Forsyth §4.4:** the multivariate normal, whose conditional gives the posterior.
    - **Frazier (2018):** expected improvement, knowledge gradient, and entropy search.
    - **Exercise.** Show that the posterior mean is kernel ridge regression with $\lambda=\epsilon$,
      and that the posterior variance at a run is about $\epsilon$ when the jitter is small.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A GP's mean weighs the runs by kernel similarity; its band shrinks near runs.
    - With few runs, the evidence can pick a lengthscale that makes the band confidently wrong.
    - EI times P(feasible) came within 0.05 m³/h of the best in 15 runs; straddle maps the
      boundary.
    - On the frontier, the decision is the margin: 7.5% saved at none, nothing at today's 3.58 m.

    **Next:** stochastic gradient descent: fitting at the scale of the national inventory.
    """)
    return


if __name__ == "__main__":
    app.run()
