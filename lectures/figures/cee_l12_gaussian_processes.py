"""Figures and numbers for CEE Lecture 12: Gaussian Processes for Optimization.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l12_gaussian_processes.py \
        --out build/fig/cee_l12_gaussian_processes

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(one prediction traced through the Gaussian process, for the TikZ diagram). Data: BattLeDIM's
L-Town model from ``scripts/fetch_battledim.sh``, run with WNTR (see ``ltown.py``).

The setting: two pressure-reducing valves feed Area A of L-Town. Lower settings leak less, but
every customer needs 35 psi (24.6 m). A calibrated city model can take minutes to hours per run,
so the engineer has 15 runs to choose the pair. A Gaussian process fitted to the runs so far
predicts leakage and the lowest pressure with their uncertainty; expected improvement times the
probability of feasibility picks the next run. Here a run takes well under a second (timed in
``numbers.json``'s ``run_time``), so every answer is checked against the simulator itself, run in
``WORKERS`` parallel processes.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.contour import ContourSet
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits

import aml_style
import title_art
from common import (
    parse_out_dir,
    pct,
    rmse,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)
from ltown import (
    BOX,
    GP_ELL_BOUNDS,
    GP_JITTER,
    GP_VAR_BOUNDS,
    P_SERVICE,
    STEPS,
    TODAY,
    LTown,
    Pair,
    Run,
    init_worker,
    key,
    worker,
)

SEED: Final[int] = 0
LO, HI = BOX  # search box for both valve settings, m
BUDGET: Final[int] = 15  # simulator runs per search
N_INIT: Final[int] = 5  # space-filling runs before the first model-guided one
SEEDS: Final[range] = range(10)  # repeats of every randomized search
CAND_STEP: Final[float] = 0.5  # candidate settings for the acquisition, m
MAP_STEP: Final[float] = 2.5  # the exhaustive map used to check, m (15 x 15 runs)
JITTER: Final[float] = GP_JITTER  # on the standardized kernel diagonal: deterministic simulator
SLICE_S1: Final[float] = 36.0  # PRV-1 setting of the one-dimensional slice (the attempt)
SLICE_RUNS: Final[tuple[float, ...]] = (30.0, 40.0, 50.0, 60.0)  # the attempt's four PRV-2 runs
MORE_RUNS: Final[tuple[float, ...]] = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)  # more runs
TRACE_S2: Final[float] = 47.0  # the setting traced through the diagram
SHORT_ELL: Final[float] = 2.0  # a fixed lengthscale far too short for the slice, m
EXTRAP_HI: Final[float] = 90.0  # the extrapolation slice runs past the box to here, m
EXTRAP_AT: Final[tuple[float, ...]] = (60.0, 75.0, 90.0)  # PRV-2 settings read off that slice, m
MARGINS_M: Final[tuple[float, ...]] = (0.0, 1.0, 2.0)  # extra metres above 35 psi
EXPONENTS: Final[tuple[float, ...]] = (0.5, 1.0, 1.5)  # leakage exponents N1 compared
# PRV-1 settings scanned for the optimum (every 1 m), then every FINE_STEP within FINE_HALF of
# the best; a scan's answer must lie strictly inside it.
OPT_COARSE: Final[tuple[float, ...]] = tuple(np.arange(32.0, 44.01, 1.0).tolist())
FINE_STEP: Final[float] = 0.1
FINE_HALF: Final[float] = 0.9
BISECT_TOL: Final[float] = 0.02  # bisection tolerance on the PRV-2 boundary, m
CORNER_S1: Final[tuple[float, ...]] = (44.0, 45.0, 46.0, 47.0, 48.0, 49.0)  # the corner scan, m
STRADDLE_Z: Final[float] = 1.96  # the straddle score's multiple of the sd (Bryan et al., 2005)
SKLEARN_TOL: Final[float] = 1e-11  # our process against scikit-learn's at fixed hyperparameters
OPT_LML_TOL: Final[float] = 1e-8  # the two optimizers' log evidence, either way
LEAK_CONTOURS: Final[tuple[float, ...]] = tuple(np.arange(14.0, 34.0, 2.0).tolist())  # m^3/h
ACQ_N: Final[int] = 71  # acquisition surface points per side: the 0.5 m lattice over the box
# Settings timed one at a time in the main process, uncached: "a run takes under half a second".
TIMED_RUNS: Final[tuple[Pair, ...]] = (
    TODAY,
    (SLICE_S1, TRACE_S2),
    (LO, LO),
    (HI, HI),
    (35.9, 46.9),
)
# At most 4 processes: the machine is shared (deep clean, 2026-11-20). No number depends on the
# count: each task runs alone in a worker with its own calibrated model and one BLAS thread.
WORKERS: Final[int] = min(4, os.cpu_count() or 1)

TaskKind = Literal["run", "boundary", "lattice", "search"]
Task = tuple[TaskKind, tuple[float, float] | tuple[str, int]]
TaskValue = float | list[Pair] | None


# ---------------------------------------------------------------------------------------------
# A Gaussian process from scratch


def matern52(A: np.ndarray, B: np.ndarray, ell: np.ndarray, var: float) -> np.ndarray:
    """Matérn 5/2 covariance with one lengthscale per input; shape ``(len(A), len(B))``."""
    d = (A[:, None, :] - B[None, :, :]) / ell  # (n, m, D)
    s = np.sqrt(5.0 * (d**2).sum(axis=-1))  # sqrt(5) r
    return var * (1.0 + s + s**2 / 3.0) * np.exp(-s)


@dataclass(frozen=True)
class GP:
    """A fitted Gaussian process on standardized outputs.

    Attributes:
        X: Training inputs, shape ``(n, D)``.
        ell: Lengthscales, shape ``(D,)``, in input units.
        var: Signal variance of the standardized output.
        mu: Output mean removed before fitting.
        sd: Output standard deviation divided out before fitting.
        chol: Lower Cholesky factor of ``K + JITTER I``.
        alpha: ``(K + JITTER I)^-1 z`` for the standardized outputs ``z``.
        nlml: Negative log marginal likelihood at these hyperparameters.
    """

    X: np.ndarray
    ell: np.ndarray
    var: float
    mu: float
    sd: float
    chol: np.ndarray
    alpha: np.ndarray
    nlml: float

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at ``Xs``, in output units.

        Args:
            Xs: Inputs scaled to the unit box, shape ``(m, D)``.

        Returns:
            The mean and the standard deviation, each of shape ``(m,)``.
        """
        ks = matern52(Xs, self.X, self.ell, self.var)  # (m, n)
        v = solve_triangular(self.chol, ks.T, lower=True)  # (n, m)
        var = np.clip(self.var - (v**2).sum(axis=0), 0.0, None)
        return self.mu + self.sd * (ks @ self.alpha), self.sd * np.sqrt(var)

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Weights on the standardized outputs: mean = mu + sd * weights @ z.

        Args:
            x: One input scaled to the unit box, shape ``(D,)``.

        Returns:
            One weight per run, shape ``(n,)``.
        """
        ks = matern52(x[None, :], self.X, self.ell, self.var)[0]
        return cho_solve((self.chol, True), ks)


def _condition(
    X: np.ndarray, z: np.ndarray, ell: np.ndarray, var: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cholesky factor, weights, and negative log marginal likelihood (R&W 2006, eq. 2.30)."""
    K = matern52(X, X, ell, var) + JITTER * np.eye(len(X))
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), z)
    nlml = 0.5 * z @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(X) * np.log(2 * np.pi)
    return L, alpha, float(nlml)


ELL_BOUNDS: Final[tuple[float, float]] = GP_ELL_BOUNDS  # lengthscales, in box widths
VAR_BOUNDS: Final[tuple[float, float]] = GP_VAR_BOUNDS


def fit_gp(X: np.ndarray, y: np.ndarray, ell: float | None = None) -> GP:
    """Fit by maximizing the marginal likelihood, from three starts; or with ``ell`` fixed.

    Args:
        X: Inputs scaled to the unit box, shape ``(n, D)``.
        y: Outputs, shape ``(n,)``.
        ell: If given, every lengthscale is fixed at this value (box widths) and only the signal
            variance is fitted.

    Returns:
        The fitted process.
    """
    D = X.shape[1]
    mu, sd = float(y.mean()), float(y.std())  # as scikit-learn's normalize_y
    sd = sd if sd > 10 * np.finfo(float).eps else 1.0  # constant outputs: as scikit-learn too
    z = (y - mu) / sd

    def objective(theta: np.ndarray) -> float:
        ells = np.full(D, ell) if ell is not None else np.exp(theta[:D])
        return _condition(X, z, ells, float(np.exp(theta[-1])))[2]

    n_ell = 0 if ell is not None else D
    bounds = [tuple(np.log(ELL_BOUNDS))] * n_ell + [tuple(np.log(VAR_BOUNDS))]
    best = None
    for start in (0.1, 0.5, 2.0) if ell is None else (1.0,):
        res = minimize(objective, np.log([start] * n_ell + [1.0]), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    ells = np.full(D, ell) if ell is not None else np.exp(best.x[:D])
    var = float(np.exp(best.x[-1]))
    L, alpha, nlml = _condition(X, z, ells, var)
    return GP(X, ells, var, mu, sd, L, alpha, nlml)


def unit(x: Iterable[Pair] | np.ndarray) -> np.ndarray:
    """Valve settings (m) scaled to the unit box."""
    return (np.asarray(x, dtype=np.float64) - LO) / (HI - LO)


def check_sklearn(X: np.ndarray, y: np.ndarray, gp: GP, Xs: np.ndarray) -> dict[str, float]:
    """Compare with scikit-learn: same kernel and hyperparameters, then its own optimizer.

    Args:
        X: Training inputs scaled to the unit box, shape ``(n, D)``.
        y: Training outputs, shape ``(n,)``.
        gp: Our process, fitted to ``X`` and ``y``.
        Xs: Inputs to compare the predictions at, shape ``(m, D)``.

    Returns:
        Largest differences in mean, standard deviation, and log marginal likelihood at fixed
        hyperparameters, and ours minus scikit-learn's optimized log marginal likelihood.

    Raises:
        RuntimeError: If the predictions or the evidence differ by ``SKLEARN_TOL`` or more, or
            the two optimized log evidences by ``OPT_LML_TOL`` or more, in either direction.
    """
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
        "max_mean_diff": float(np.abs(m - m_ref).max()),
        "max_sd_diff": float(np.abs(s - s_ref).max()),
        "lml_diff": float(abs(-gp.nlml - ref.log_marginal_likelihood_value_)),
        "ours_minus_sklearn_opt_lml": float(-gp.nlml - opt.log_marginal_likelihood_value_),
    }
    # The slides quote agreement within 1e-11 and the same optimum within 1e-8: both ways, since
    # an optimum much better than scikit-learn's would not be "the same" either.
    if max(out["max_mean_diff"], out["max_sd_diff"], out["lml_diff"]) >= SKLEARN_TOL:
        raise RuntimeError(f"ours and scikit-learn's process differ: {out}")
    if abs(out["ours_minus_sklearn_opt_lml"]) >= OPT_LML_TOL:
        raise RuntimeError(f"ours and scikit-learn's optimized evidence differ: {out}")
    return out


# ---------------------------------------------------------------------------------------------
# Searches: Bayesian optimization, straddle (active learning), random, grid


def candidates() -> np.ndarray:
    """Every pair of settings on a ``CAND_STEP`` lattice over the box, shape ``(5041, 2)``."""
    g = np.arange(LO, HI + 1e-9, CAND_STEP)
    return np.array([(a, b) for a in g for b in g])


def initial_design(seed: int) -> list[Pair]:
    """``N_INIT`` space-filling settings (Latin hypercube), rounded to 0.1 m like a valve pilot.

    Args:
        seed: Seed of the Latin hypercube.

    Returns:
        The settings (PRV-1, PRV-2), m.
    """
    pts = qmc.scale(qmc.LatinHypercube(d=2, seed=seed).random(N_INIT), [LO, LO], [HI, HI])
    return [(round(float(a), 1), round(float(b), 1)) for a, b in pts]


def fit_both(town: LTown, X: list[Pair]) -> tuple[GP, GP]:
    """Processes for leakage and for the lowest pressure, from the runs so far.

    Args:
        town: The simulator (its cache holds the runs).
        X: The settings run so far, m.

    Returns:
        The process for leakage and the process for the lowest pressure.
    """
    runs = [town.run(*x) for x in X]
    U = unit(X)
    return (
        fit_gp(U, np.array([r.leak for r in runs])),
        fit_gp(U, np.array([r.pmin for r in runs])),
    )


def probability_feasible(g_p: GP, U: np.ndarray, p_req: float = P_SERVICE) -> np.ndarray:
    """P(lowest pressure >= ``p_req``) under the process for the lowest pressure.

    Args:
        g_p: The process for the lowest pressure.
        U: Settings scaled to the unit box, shape ``(m, 2)``.
        p_req: The pressure to keep, m.

    Returns:
        The probability at each setting, shape ``(m,)``.
    """
    m, s = g_p.predict(U)
    return norm.cdf((m - p_req) / np.maximum(s, 1e-12))


def expected_improvement(m: np.ndarray, s: np.ndarray, best: float) -> np.ndarray:
    """E[max(best - f, 0)] for f ~ N(m, s^2): the closed form of Jones et al. (1998).

    Args:
        m: Predicted means.
        s: Predicted standard deviations (floored at 1e-12).
        best: The least feasible value so far.

    Returns:
        The expected improvement, the shape of ``m``.
    """
    s = np.maximum(s, 1e-12)
    z = (best - m) / s
    return (best - m) * norm.cdf(z) + s * norm.pdf(z)


def bo_score(g_leak: GP, g_p: GP, U: np.ndarray, best: float | None) -> np.ndarray:
    """Expected improvement in leakage times the probability of feasibility (Gardner et al., 2014).

    Args:
        g_leak: The process for leakage.
        g_p: The process for the lowest pressure.
        U: Candidate settings scaled to the unit box, shape ``(m, 2)``.
        best: The least feasible leakage so far, or None before any feasible run.

    Returns:
        The score at each candidate; before any feasible run, the probability of feasibility.
    """
    pof = probability_feasible(g_p, U)
    if best is None:
        return pof
    return expected_improvement(*g_leak.predict(U), best) * pof


def straddle_score(g_leak: GP, g_p: GP, U: np.ndarray, best: float | None) -> np.ndarray:
    """Active learning of the 24.6 m boundary: 1.96 sd minus distance to it (Bryan et al., 2005).

    Args:
        g_leak: Unused (the signature of ``bo_score``).
        g_p: The process for the lowest pressure.
        U: Candidate settings scaled to the unit box, shape ``(m, 2)``.
        best: Unused (the signature of ``bo_score``).

    Returns:
        The score at each candidate.
    """
    del g_leak, best  # unused: the same signature as bo_score
    m, s = g_p.predict(U)
    return STRADDLE_Z * s - np.abs(m - P_SERVICE)


STRATEGIES: Final[Mapping[str, Callable[[GP, GP, np.ndarray, float | None], np.ndarray]]] = {
    "bo": bo_score,
    "straddle": straddle_score,
}


def best_feasible(runs: list[Run], p_req: float = P_SERVICE) -> float | None:
    """Least leakage among runs that meet ``p_req``, or None.

    Args:
        runs: Simulated runs.
        p_req: The pressure to keep, m.

    Returns:
        The least feasible leakage (m^3/h), or None if no run is feasible.
    """
    ok = [r.leak for r in runs if r.pmin >= p_req]
    return min(ok) if ok else None


def sequential(town: LTown, seed: int, name: str) -> list[Pair]:
    """Start from the initial design, then add the candidate that maximizes the strategy's score.

    Args:
        town: The simulator.
        seed: Seed of the initial design.
        name: The strategy, a key of ``STRATEGIES``.

    Returns:
        The ``BUDGET`` settings run, in order, m.
    """
    cand = candidates()
    U = unit(cand)
    X = initial_design(seed)
    while len(X) < BUDGET:
        g_leak, g_p = fit_both(town, X)
        score = STRATEGIES[name](g_leak, g_p, U, best_feasible([town.run(*x) for x in X]))
        done = {key(a, b) for a, b in X}
        score[[key(a, b) in done for a, b in cand]] = -np.inf
        a, b = cand[int(np.argmax(score))]
        X.append((float(a), float(b)))
    return X


def random_search(seed: int) -> list[Pair]:
    """``BUDGET`` settings drawn uniformly over the box, rounded to 0.1 m.

    Args:
        seed: Seed of the draws.

    Returns:
        The settings, m.
    """
    pts = np.random.default_rng(seed).uniform(LO, HI, (BUDGET, 2))
    return [(round(float(a), 1), round(float(b), 1)) for a, b in pts]


def grid_search() -> list[Pair]:
    """A 4 x 4 grid over the box (16 runs)."""
    g = np.linspace(LO, HI, 4)
    return [(float(a), float(b)) for a in g for b in g]


def running_best(runs: list[Run]) -> list[float | None]:
    """Least feasible leakage after each run.

    Args:
        runs: The runs, in the order they were made.

    Returns:
        After each run, the least feasible leakage so far, or None before any feasible run.
    """
    return [best_feasible(runs[: i + 1]) for i in range(len(runs))]


# ---------------------------------------------------------------------------------------------
# The truth: boundary bisection and an exhaustive map


def boundary_s2(
    town: LTown, s1: float, p_req: float = P_SERVICE, tol: float = BISECT_TOL
) -> float | None:
    """Lowest PRV-2 setting in the box that meets ``p_req`` (bisection), or None.

    The lowest pressure rises with the PRV-2 setting, and so does leakage, so for a given PRV-1
    setting the best feasible PRV-2 setting is this boundary.

    Args:
        town: The simulator.
        s1: The PRV-1 setting, m.
        p_req: The pressure to keep, m.
        tol: Width of the final bracket, m: the answer lies at most this far above the boundary.

    Returns:
        The PRV-2 setting (m), ``LO`` if every setting is feasible, or None if none is.
    """
    if town.run(s1, HI).pmin < p_req:
        return None
    if town.run(s1, LO).pmin >= p_req:
        return LO
    a, b = LO, HI
    while b - a > tol:
        m = 0.5 * (a + b)
        a, b = (a, m) if town.run(s1, m).pmin >= p_req else (m, b)
    return b


def lattice_boundary(town: LTown, s1: float, p_req: float = P_SERVICE) -> float | None:
    """Lowest PRV-2 setting on the candidates' ``CAND_STEP`` lattice that meets ``p_req``, or None.

    The bisection's answer lies within its tolerance above the true boundary, so the lattice's
    lowest feasible setting is the lattice point at or above it, or the one just below.

    Args:
        town: The simulator.
        s1: The PRV-1 setting, m.
        p_req: The pressure to keep, m.

    Returns:
        The PRV-2 setting on the lattice (m), or None if no setting in the box is feasible.

    Raises:
        RuntimeError: If the lattice point above a feasible setting is infeasible (the lowest
            pressure would then not rise with the PRV-2 setting, which the search assumes).
    """
    b = boundary_s2(town, s1, p_req)
    if b is None:
        return None
    up = round(LO + math.ceil((b - LO) / CAND_STEP - 1e-9) * CAND_STEP, 6)
    for s2 in (round(up - CAND_STEP, 6), up):
        if s2 >= LO and town.run(s1, s2).pmin >= p_req:
            return s2
    raise RuntimeError(f"PRV-2 = {up} m is infeasible above the boundary {b} m at PRV-1 = {s1} m")


def map_grid() -> np.ndarray:
    """Settings of the exhaustive map, shape ``(15, 15, 2)``: [i, j] = (PRV-1 g[i], PRV-2 g[j])."""
    g = np.arange(LO, HI + 1e-9, MAP_STEP)
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)


# ---------------------------------------------------------------------------------------------
# Parallel simulation: each worker process holds its own calibrated copy of the model (ltown's
# init_worker), with one BLAS thread, so workers do not oversubscribe and sums are reproducible.


def _task(task: Task) -> tuple[TaskValue, dict[Pair, Run]]:
    """Run one task in a worker; return its value and the runs the caller needs.

    Args:
        task: ``("run", (s1, s2))``, ``("boundary", (s1, p_req))``, ``("lattice", (s1,
            p_req))``, or ``("search", (strategy, seed))``.

    Returns:
        The task's value (None, a PRV-2 setting or None, or a search's settings) and the runs
        whose results the caller's cache needs.

    Raises:
        ValueError: On an unknown kind of task.
    """
    town = worker()
    kind, args = task
    if kind == "run":
        return None, {key(*args): town.run(*args)}
    if kind == "boundary":
        s1, p_req = args
        s2 = boundary_s2(town, s1, p_req)
        return s2, ({} if s2 is None else {key(s1, s2): town.run(s1, s2)})
    if kind == "lattice":  # the lowest feasible PRV-2 setting on the candidate lattice
        s1, p_req = args
        s2 = lattice_boundary(town, s1, p_req)
        return s2, ({} if s2 is None else {key(s1, s2): town.run(s1, s2)})
    if kind == "search":
        name, seed = args
        X = sequential(town, seed, name)
        return X, {key(*x): town.run(*x) for x in X}
    raise ValueError(f"unknown task {kind!r}")


class Pool:
    """A process pool whose workers simulate with the same leakage exponent as ``town``."""

    def __init__(self, town: LTown) -> None:
        """Start ``WORKERS`` processes.

        Args:
            town: The model whose leakage exponent the workers use and whose cache gets the runs.
        """
        self.town = town
        self.ex = ProcessPoolExecutor(WORKERS, initializer=init_worker, initargs=(town.n1,))

    def __enter__(self) -> Pool:
        """Return the pool."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Shut the workers down (the exception, if any, propagates)."""
        del exc
        self.ex.shutdown()

    def map(self, tasks: list[Task]) -> list[TaskValue]:
        """Run tasks in parallel; their runs join ``town``'s cache.

        Args:
            tasks: The tasks (see ``_task``).

        Returns:
            Their values, in the order of ``tasks``.
        """
        values = []
        for value, runs in self.ex.map(_task, tasks):
            self.town.cache.update(runs)
            values.append(value)
        return values

    def runs(self, pairs: Iterable[Pair]) -> list[Run]:
        """Simulate independent settings; cached ones are not run again.

        Args:
            pairs: Settings (PRV-1, PRV-2), m.

        Returns:
            One run per pair, in order.
        """
        pairs = [(float(a), float(b)) for a, b in pairs]
        todo = sorted({key(*p) for p in pairs} - set(self.town.cache))
        self.map([("run", p) for p in todo])
        return [self.town.run(*p) for p in pairs]

    def optimum(self, p_req: float = P_SERVICE) -> Pair:
        """Least leakage on the boundary: PRV-1 every 1 m, then every 0.1 m around the best.

        Args:
            p_req: The pressure to keep, m.

        Returns:
            The best settings (PRV-1, PRV-2), m.

        Raises:
            RuntimeError: If no PRV-1 setting of a scan has a feasible PRV-2 setting, or the best
                lies at either end of the coarse or the fine scan.
        """

        def scan(s1s: Iterable[float]) -> Pair:
            s1s = [round(float(v), 6) for v in s1s]
            s2s = self.map([("boundary", (s1, p_req)) for s1 in s1s])
            leak = [
                np.inf if s2 is None else self.town.run(s1, s2).leak
                for s1, s2 in zip(s1s, s2s, strict=True)
            ]
            i = int(np.argmin(leak))
            if not np.isfinite(leak[i]):
                raise RuntimeError(f"no feasible PRV-2 setting in the box for p_req = {p_req}")
            if not 0 < i < len(s1s) - 1:
                raise RuntimeError(f"optimum PRV-1 {s1s[i]} m at the edge of the scan {s1s}")
            return s1s[i], float(s2s[i])

        s1, _ = scan(OPT_COARSE)
        return scan(np.arange(s1 - FINE_HALF, s1 + FINE_HALF + 1e-9, FINE_STEP))

    def lattice_best(self, p_req: float = P_SERVICE) -> Pair:
        """Least leakage among feasible settings on the candidates' lattice (what BO can pick).

        Args:
            p_req: The pressure to keep, m.

        Returns:
            The best lattice settings (PRV-1, PRV-2), m.

        Raises:
            RuntimeError: If the best lies at either end of the scan.
        """
        s1s = [
            round(float(v), 6) for v in np.arange(OPT_COARSE[0], OPT_COARSE[-1] + 1e-9, CAND_STEP)
        ]
        s2s = self.map([("lattice", (s1, p_req)) for s1 in s1s])
        leak = [
            np.inf if s2 is None else self.town.run(s1, s2).leak
            for s1, s2 in zip(s1s, s2s, strict=True)
        ]
        i = int(np.argmin(leak))
        if not 0 < i < len(s1s) - 1:
            raise RuntimeError(f"lattice optimum PRV-1 {s1s[i]} m at the edge of the scan")
        return s1s[i], float(s2s[i])


# ---------------------------------------------------------------------------------------------
# Number formatting


def run_dict(r: Run, s: Pair) -> dict[str, object]:
    """One run as numbers for the slides.

    Args:
        r: The run.
        s: Its settings, m.

    Returns:
        Settings, leakage, lowest pressure, and the junction where it occurs.
    """
    return {
        "s1": rnd(s[0], 2),
        "s2": rnd(s[1], 2),
        "leak": rnd(r.leak, 2),
        "pmin": rnd(r.pmin, 2),
        "node": r.node,
    }


# ---------------------------------------------------------------------------------------------
# Lecture sections


def network_figure(town: LTown, out_dir: Path) -> dict[str, object]:
    """Map of L-Town by area, with the two valves and the junction with the lowest pressure.

    Args:
        town: The simulator.
        out_dir: Where the figure goes.

    Returns:
        The junction with today's lowest pressure, and the hour it occurs.
    """
    sizes = {"A": len(town.area_a), "B": len(town.area_b), "C": len(town.area_c)}
    styles = {
        "A": (aml_style.PALETTE[0], "-"),
        "B": (aml_style.PALETTE[1], "--"),
        "C": (aml_style.PALETTE[2], ":"),
    }
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for a, b, zone in town.segments():
        color, ls = styles.get(zone, (aml_style.MUTED, "-"))
        ax.plot(*zip(a, b, strict=True), color=color, ls=ls, lw=1.0)
    for zone, (color, ls) in styles.items():
        ax.plot([], [], color=color, ls=ls, lw=2, label=f"Area {zone} ({sizes[zone]})")
    for name, mk in (("PRV-1", "s"), ("PRV-2", "D")):
        x, y = town.outlet_xy(name)
        ax.plot(x, y, mk, ms=11, color=aml_style.INK, mfc="white", mew=2, zorder=5)
        ax.annotate(name, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=12,
                    path_effects=aml_style.HALO)  # fmt: skip
    run = town.run(*TODAY)
    x, y = town.wn.get_node(run.node).coordinates
    ax.plot(x, y, "^", ms=12, color=aml_style.PALETTE[3], zorder=5)
    ax.annotate(
        f"{run.node}, lowest",
        (x, y),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=12,
        path_effects=aml_style.HALO,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=12,
        handlelength=2.2,
        title="Junctions by area",
        title_fontsize=12,
    )
    aml_style.save(fig, out_dir, "ltown_map")
    return {"critical_node": run.node, "critical_hour": rnd(run.hour, 2)}


def setting_summary(town: LTown, pool: Pool) -> dict[str, object]:
    """Today's settings, the network, and the area the valves do not reach.

    Args:
        town: The simulator.
        pool: Its worker processes.

    Returns:
        Today's run, the service pressure, the junction counts, and Area C's lowest pressure;
        the elevation of the junction with the lowest pressure and how many customers lie higher.

    Raises:
        RuntimeError: If any customer junction lies higher than the one with the lowest pressure
            (the slides say its customers are on the highest ground).
    """
    today = town.run(*TODAY)
    # Area C is fed by the tank: its lowest pressure does not move with the two valves.
    c_low = [r.pmin_c for r in pool.runs([(LO, LO), TODAY, (HI, HI)])]
    assert max(c_low) - min(c_low) < 0.01, c_low
    elevation = {n: float(town.wn.get_node(n).elevation) for n in town.customers}
    higher = sum(v > elevation[today.node] for v in elevation.values())
    if higher:
        raise RuntimeError(f"{higher} customers lie higher than {today.node}")
    return {
        "today": run_dict(today, TODAY),
        "p_service": rnd(P_SERVICE, 2),
        "leak_target": rnd(town.target, 2),
        "junctions": {"A": len(town.area_a), "B": len(town.area_b), "C": len(town.area_c)},
        "customers_ab": len(town.customers),
        "peak_day": town.day,
        "area_c_min": rnd(min(c_low), 2),
        "critical_elevation_m": rnd(elevation[today.node], 1),
        "customers_higher": higher,
    }


@dataclass(frozen=True)
class Trace:
    """The attempt's trace at full precision: the diagram's macros are formatted from these.

    Attributes:
        gp: The process fitted to the attempt's four runs.
        pmin: The four runs' lowest pressures, m.
        k: Kernel values between the traced setting and the runs.
        weights: The weights on the standardized runs.
        mean: Predicted lowest pressure at the traced setting, m.
        sd: Its standard deviation, m.
        truth: The simulator's lowest pressure there, m.
        pof: Probability of feasibility there.
    """

    gp: GP
    pmin: tuple[float, ...]
    k: np.ndarray
    weights: np.ndarray
    mean: float
    sd: float
    truth: float
    pof: float


def attempt_and_trace(town: LTown) -> tuple[dict[str, object], Trace]:
    """The attempt's four runs on the slice PRV-1 = 36 m, the fitted process, and the trace.

    Args:
        town: The simulator.

    Returns:
        The numbers for the slides, and the same trace unrounded (with the fitted process).
    """
    runs = [town.run(SLICE_S1, s2) for s2 in SLICE_RUNS]
    X = unit(np.array(SLICE_RUNS)[:, None])
    gp = fit_gp(X, np.array([r.pmin for r in runs]))
    xs = unit(np.array([[TRACE_S2]]))
    m, s = gp.predict(xs)
    truth = town.run(SLICE_S1, TRACE_S2)
    # Where a straight line between the 40 m and 50 m runs crosses 35 psi:
    (x0, y0), (x1, y1) = ((v, town.run(SLICE_S1, v).pmin) for v in (40.0, 50.0))
    raw = Trace(
        gp=gp,
        pmin=tuple(r.pmin for r in runs),
        k=matern52(xs, X, gp.ell, gp.var)[0],
        weights=gp.weights(xs[0]),
        mean=float(m[0]),
        sd=float(s[0]),
        truth=truth.pmin,
        pof=float(norm.cdf((m[0] - P_SERVICE) / s[0])),
    )
    trace = {
        "s1": SLICE_S1,
        "s2": TRACE_S2,
        "k": [rnd(v, 3) for v in raw.k],
        "weights": [rnd(v, 3) for v in raw.weights],
        "mean": rnd(raw.mean, 2),
        "sd": rnd(raw.sd, 2),
        "lo": rnd(raw.mean - 2 * raw.sd, 2),
        "hi": rnd(raw.mean + 2 * raw.sd, 2),
        "pof": rnd(raw.pof, 3),
        "truth": rnd(raw.truth, 2),
        "ell_m": rnd(gp.ell[0] * (HI - LO), 2),
        "var": rnd(gp.var, 2),
        "z_mean": rnd(gp.mu, 2),
        "z_sd": rnd(gp.sd, 2),
        "line_cross": rnd(x0 + (P_SERVICE - y0) * (x1 - x0) / (y1 - y0), 2),
        "true_cross": rnd(boundary_s2(town, SLICE_S1), 2),
    }
    attempt = {
        "runs": [run_dict(r, (SLICE_S1, s2)) for r, s2 in zip(runs, SLICE_RUNS, strict=True)],
        "trace": trace,
    }
    return attempt, raw


def _slice_band(
    gp: GP, s2: np.ndarray, truth: np.ndarray, runs: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Mean, sd, and how well they describe the simulator along the slice, between the runs.

    The settings that were run are left out: there the process interpolates the simulator, so
    they would count as covered and error-free whatever the band does elsewhere.

    Args:
        gp: The process on the slice.
        s2: PRV-2 settings along the slice, m.
        truth: The simulator's lowest pressure at ``s2``, m.
        runs: The PRV-2 settings run.

    Returns:
        The mean and sd at ``s2``, and the lengthscale, error, and coverage of the +-2 sd band
        between the runs (as a share and as a count of settings).
    """
    m, s = gp.predict(unit(s2[:, None]))
    off = ~np.isin(s2, runs)
    assert off.sum() == len(s2) - len(runs), "a run is not on the evaluation grid"
    inside = np.abs(m[off] - truth[off]) <= 2 * s[off]
    stats = {
        "ell_m": rnd(gp.ell[0] * (HI - LO), 2),
        "rmse": rnd(rmse(m[off], truth[off]), 2),
        "inside_2sd": rnd(np.mean(inside), 3),
        "inside_n": int(inside.sum()),
        "points": int(off.sum()),
    }
    return m, s, stats


def lengthscale_figure(town: LTown, pool: Pool, out_dir: Path) -> dict[str, object]:
    """Four runs with a short lengthscale, four with the fitted one, seven with the fitted one.

    Args:
        town: The simulator.
        pool: Its worker processes.
        out_dir: Where the figure goes.

    Returns:
        Each panel's band statistics (``_slice_band``), and where the simulator's flat start ends.
    """
    s2 = np.arange(LO, HI + 1e-9, 1.0)
    truth = np.array([r.pmin for r in pool.runs([(SLICE_S1, v) for v in s2])])
    titles = {
        "short": "4 runs, ℓ = {:.0f} m",
        "fitted": "4 runs, ℓ = {:.0f} m (fitted)",
        "more": "7 runs, ℓ = {:.0f} m (fitted)",
    }
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, sharey=True)
    out = {}
    for ax, (name, runs, ell) in zip(
        axes,
        (
            ("short", SLICE_RUNS, SHORT_ELL / (HI - LO)),
            ("fitted", SLICE_RUNS, None),
            ("more", MORE_RUNS, None),
        ),
        strict=True,
    ):
        y = np.array([town.run(SLICE_S1, v).pmin for v in runs])
        gp = fit_gp(unit(np.array(runs)[:, None]), y, ell=ell)
        m, s, out[name] = _slice_band(gp, s2, truth, runs)
        ax.fill_between(s2, m - 2 * s, m + 2 * s, color=aml_style.PALETTE[0], alpha=0.18, lw=0)
        ax.plot(s2, m, color=aml_style.PALETTE[0], label="GP mean, ±2 sd")
        ax.plot(s2, truth, color=aml_style.INK, ls="--", lw=1.6, label="simulator")
        ax.plot(runs, y, "o", color=aml_style.PALETTE[1], ms=8, zorder=4, label="runs")
        ax.axhline(P_SERVICE, color=aml_style.MUTED, ls=":", lw=1.4)
        ax.set_title(titles[name].format(out[name]["ell_m"]), fontsize=13)
        ax.set_xlabel("PRV-2 setting (m)")
    axes[0].set_ylabel("Lowest pressure (m)")
    # In the middle panel: the first panel's wide band fills its corners.
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper left", fontsize=10)
    aml_style.save(fig, out_dir, "lengthscale")
    # The simulator's curve is flat at the low end, a feature four runs cannot reveal.
    flat = s2[np.abs(truth - truth[0]) < 0.01]
    assert np.all(np.diff(flat) == 1.0), flat  # one flat stretch from the box's edge
    out["flat_until"] = rnd(flat.max(), 2)
    return out


def verify(town: LTown, attempt_gp: GP) -> dict[str, object]:
    """From-scratch processes against scikit-learn, on the attempt slice and on 15 runs in 2-D.

    Args:
        town: The simulator.
        attempt_gp: The process fitted to the attempt's four runs.

    Returns:
        ``check_sklearn``'s differences for each, and the 2-D processes' lengthscales (m).
    """
    y1 = np.array([town.run(SLICE_S1, v).pmin for v in SLICE_RUNS])
    out = {
        "slice": check_sklearn(
            attempt_gp.X, y1, attempt_gp, unit(np.arange(LO, HI + 1e-9, 0.5)[:, None])
        )
    }
    X2 = initial_design(SEED) + random_search(SEED)[: BUDGET - N_INIT]
    runs = [town.run(*x) for x in X2]
    for name, y in (("leak", [r.leak for r in runs]), ("pmin", [r.pmin for r in runs])):
        gp = fit_gp(unit(X2), np.array(y))
        out[name] = check_sklearn(unit(X2), np.array(y), gp, unit(candidates()))
        out[name]["ell_m"] = [rnd(v * (HI - LO), 2) for v in gp.ell]
    return out


@dataclass(frozen=True)
class Found:
    """The settings each search ran: one list per seed, and the grid's one list.

    Attributes:
        bo: Bayesian optimization, one list of ``BUDGET`` settings per seed in ``SEEDS``.
        straddle: Straddle (active learning of the boundary), the same.
        random: Random settings, the same.
        grid: The 4 x 4 grid.
    """

    bo: tuple[list[Pair], ...]
    straddle: tuple[list[Pair], ...]
    random: tuple[list[Pair], ...]
    grid: list[Pair]

    def seeds(self, name: str) -> tuple[list[Pair], ...]:
        """One randomized search's settings, one list per seed.

        Args:
            name: ``"bo"``, ``"straddle"``, or ``"random"``.

        Returns:
            The lists of settings.
        """
        return {"bo": self.bo, "straddle": self.straddle, "random": self.random}[name]


def searches(pool: Pool) -> Found:
    """Every seed of Bayesian optimization and straddle in parallel, plus random and a grid.

    Args:
        pool: The worker processes; every run joins the main model's cache.

    Returns:
        The settings each search ran.
    """
    tasks: list[Task] = [("search", (name, seed)) for name in STRATEGIES for seed in SEEDS]
    by_name: dict[str, list[list[Pair]]] = {name: [] for name in STRATEGIES}
    for (_, (name, _)), X in zip(tasks, pool.map(tasks), strict=True):
        assert isinstance(X, list)
        by_name[str(name)].append(X)
    found = Found(
        bo=tuple(by_name["bo"]),
        straddle=tuple(by_name["straddle"]),
        random=tuple(random_search(seed) for seed in SEEDS),
        grid=grid_search(),
    )
    pool.runs([x for X in found.random for x in X] + found.grid)
    return found


def search_numbers(
    town: LTown, pool: Pool, found: Found, opt: Pair, truth: np.ndarray
) -> dict[str, object]:
    """Regret and map errors of each search, against the true optimum and the exhaustive map.

    The bracket below the regret: the best feasible setting on the candidates' lattice, the
    least regret a model-guided run can reach (the initial design is not on the lattice).

    A search's map errors count only the map points it did not run: at a run the process repeats
    the simulator, so such a point would count as right whatever the process does elsewhere.
    Medians over the seeds, with the range across them.

    Args:
        town: The simulator (its cache holds every run).
        pool: Its worker processes.
        found: The settings each search ran.
        opt: The true optimum, m.
        truth: The exhaustive map, shape ``(15, 15, 2)``: leakage and lowest pressure.

    Returns:
        Per search: regret and misclassified map share (median, range), map RMSE, and how many
        map points each seed ran; the grid's best; BO seed 0's runs; the lattice's best.
    """
    best_true = town.run(*opt).leak
    lat = pool.lattice_best()
    P = map_grid().reshape(-1, 2)
    p_map = truth[..., 1].ravel()
    feasible = p_map >= P_SERVICE
    out: dict[str, object] = {
        "map_feasible": int(feasible.sum()),
        "map_points": int(P.shape[0]),
        "candidates": len(candidates()),
        "lattice": run_dict(town.run(*lat), lat)
        | {"regret": rnd(town.run(*lat).leak - best_true, 2)},
    }
    for name in ("bo", "straddle", "random"):
        regrets, wrong, rmses, on_map = [], [], [], []
        for X in found.seeds(name):
            runs = [town.run(*x) for x in X]
            b = best_feasible(runs)
            regrets.append(np.inf if b is None else b - best_true)
            ran = {key(*x) for x in X}
            off = np.array([key(*q) not in ran for q in P])
            m = fit_gp(unit(X), np.array([r.pmin for r in runs])).predict(unit(P[off]))[0]
            wrong.append(float(np.mean((m >= P_SERVICE) != feasible[off])))
            rmses.append(rmse(m, p_map[off]))
            on_map.append(int((~off).sum()))
        out[name] = {
            "regret_median": rnd(np.median(regrets), 2),
            "regret_min": rnd(np.min(regrets), 2),
            "regret_max": rnd(np.max(regrets), 2),
            "misclassified_median_pct": pct(np.median(wrong), 1),
            "misclassified_min_pct": pct(np.min(wrong), 1),
            "misclassified_max_pct": pct(np.max(wrong), 1),
            "map_rmse_median": rnd(np.median(rmses), 2),
            "map_points_run": on_map,
        }
    grid_best = best_feasible([town.run(*x) for x in found.grid])
    assert grid_best is not None
    out["grid"] = {
        "runs": len(found.grid),
        "best": rnd(grid_best, 2),
        "regret": rnd(grid_best - best_true, 2),
    }
    bo0 = found.bo[SEED]
    runs0 = [town.run(*x) for x in bo0]
    j = int(np.argmin([r.leak if r.pmin >= P_SERVICE else np.inf for r in runs0]))
    today = town.run(*TODAY).leak
    out["bo_seed0"] = {
        "best": run_dict(runs0[j], bo0[j]),
        "saving_pct": pct(1 - runs0[j].leak / today, 1),
        "best_at_run": j + 1,
        "runs": [run_dict(r, x) for r, x in zip(runs0, bo0, strict=True)],
    }
    return out


def map_numbers(truth: np.ndarray) -> dict[str, object]:
    """How fast the lowest pressure changes along each valve's axis, from the exhaustive map.

    A lengthscale is a correlation distance, not a rate; this is the rate: the mean absolute
    change in the lowest pressure between neighbouring map points, per metre of setting.

    Args:
        truth: The exhaustive map, shape ``(15, 15, 2)``: [i, j] = (PRV-1 g[i], PRV-2 g[j]).

    Returns:
        Mean absolute change of the lowest pressure per metre of PRV-1 and of PRV-2 (m/m).
    """
    p = truth[..., 1]
    return {
        "pmin_per_m": {
            "prv1": rnd(np.abs(np.diff(p, axis=0)).mean() / MAP_STEP, 3),
            "prv2": rnd(np.abs(np.diff(p, axis=1)).mean() / MAP_STEP, 3),
        }
    }


def _label_spots(
    cs: ContourSet, keep: np.ndarray, busy: np.ndarray, box: tuple[float, float]
) -> list[tuple[float, float]]:
    """Where to label each kept contour: its point farthest from markers and other curves.

    Args:
        cs: The contours.
        keep: Levels to label.
        busy: (n, 2) points to stay away from (runs, markers, boundary vertices).
        box: The axes range, the same on both axes.

    Returns:
        One position per labelled level, inside the box with a margin of 6%.
    """
    lo, hi = box
    margin = 0.06 * (hi - lo)
    spots = []
    for level, path in zip(cs.levels, cs.get_paths(), strict=True):
        v = path.vertices
        v = v[np.all((v > lo + margin) & (v < hi - margin), axis=1)] if len(v) else v
        if level not in keep or len(v) == 0:
            continue
        d = np.linalg.norm(v[:, None, :] - busy[None, :, :], axis=2).min(axis=1)
        spots.append((float(v[np.argmax(d), 0]), float(v[np.argmax(d), 1])))
    return spots


def _bo_map_panel(
    ax: plt.Axes, town: LTown, X: list[Pair], *, opt: Pair, truth: np.ndarray, title: str
) -> None:
    """One search's map: leakage contours, the true and the GP's boundary, and its runs."""
    G = map_grid()
    S1, S2 = G[..., 0], G[..., 1]
    cs = ax.contour(
        S1, S2, truth[..., 0], levels=LEAK_CONTOURS, colors=aml_style.MUTED, linewidths=0.8
    )
    ax.contourf(S1, S2, truth[..., 1], levels=[-1e9, P_SERVICE], colors="none", hatches=["///"])
    true_b = ax.contour(
        S1, S2, truth[..., 1], levels=[P_SERVICE], colors=aml_style.INK, linewidths=2.2
    )
    g_p = fit_gp(unit(X), np.array([town.run(*x).pmin for x in X]))
    m = g_p.predict(unit(G.reshape(-1, 2)))[0].reshape(G.shape[:2])
    gp_b = ax.contour(
        S1, S2, m, levels=[P_SERVICE], colors=aml_style.PALETTE[0], linewidths=2.2, linestyles="--"
    )
    bounds = [c.get_paths()[0].vertices for c in (true_b, gp_b)]
    busy = np.vstack([np.array(X), [opt], [TODAY], *bounds])
    ax.clabel(
        cs,  # no levels=: with manual spots, matplotlib indexes the full level list
        fmt="%d",
        fontsize=12,
        manual=_label_spots(cs, cs.levels[::2], busy, (LO, HI)),
    )
    for t in cs.labelTexts:  # legible over the hatching
        t.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    xs = np.array(X)
    first = {"color": aml_style.PALETTE[1], "ms": 8, "mfc": "white", "mew": 2}
    ax.plot(*xs[:N_INIT].T, "o", clip_on=False, zorder=5, **first)
    ax.plot(*xs[N_INIT:].T, "s", color=aml_style.PALETTE[1], ms=7, clip_on=False, zorder=5)
    ax.plot(*opt, "*", color=aml_style.PALETTE[3], ms=18, mec=aml_style.INK, zorder=6)
    ax.plot(*TODAY, "D", color=aml_style.INK, ms=8, mfc="white", mew=2, zorder=6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PRV-1 setting (m)")


def _bo_map_legend(fig: plt.Figure) -> None:
    """The shared legend under both maps."""
    marker = {"ls": "", "color": aml_style.PALETTE[1]}
    handles = [
        plt.Line2D([], [], color=aml_style.INK, lw=2.2, label="true 24.6 m boundary"),
        plt.Line2D([], [], color=aml_style.PALETTE[0], lw=2.2, ls="--", label="GP's boundary"),
        plt.Line2D([], [], color=aml_style.MUTED, lw=0.8, label="leakage (m³/h)"),
        plt.Line2D([], [], marker="o", mfc="white", mew=2, label="first 5 runs", **marker),
        plt.Line2D([], [], marker="s", label="next 10", **marker),
        plt.Line2D([], [], marker="*", ls="", color=aml_style.PALETTE[3], ms=14, label="optimum"),
        plt.Line2D([], [], marker="D", ls="", color=aml_style.INK, mfc="white", label="today"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.05)
    )


def bo_map_figure(town: LTown, found: Found, opt: Pair, truth: np.ndarray, out_dir: Path) -> None:
    """Leakage over the box with the true boundary, and where each strategy spent its runs.

    Args:
        town: The simulator.
        found: The settings each search ran (seed ``SEED`` is drawn).
        opt: The true optimum, m.
        truth: The exhaustive map, shape ``(15, 15, 2)``.
        out_dir: Where the figure goes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(aml_style.FULL[0], 3.0), sharey=True)
    for ax, name, title in zip(
        axes,
        ("bo", "straddle"),
        ("Bayesian optimization", "Straddle (map the boundary)"),
        strict=True,
    ):
        _bo_map_panel(ax, town, found.seeds(name)[SEED], opt=opt, truth=truth, title=title)
    axes[0].set_ylabel("PRV-2 setting (m)")
    _bo_map_legend(fig)
    aml_style.save(fig, out_dir, "bo_map")


def regret_figure(town: LTown, found: Found, opt: Pair, out_dir: Path) -> dict[str, object]:
    """Least feasible leakage found after each run: BO and random (median, range), the grid.

    Args:
        town: The simulator.
        found: The settings each search ran.
        opt: The true optimum, m.
        out_dir: Where the figure goes.

    Returns:
        BO's and random's median curves (None before every seed has a feasible run).
    """
    best_true = town.run(*opt).leak
    today = town.run(*TODAY).leak
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    n = np.arange(1, BUDGET + 1)
    out = {}
    for name, label, color, mk in (
        ("bo", "BO", aml_style.PALETTE[0], "o"),
        ("random", "random", aml_style.PALETTE[1], "s"),
    ):
        curves = np.array(
            [
                [np.inf if b is None else b for b in running_best([town.run(*x) for x in X])]
                for X in found.seeds(name)
            ]
        )  # (seeds, runs); inf until a feasible run is found
        med = np.median(curves, axis=0)
        med = np.where(np.isfinite(med), med, np.nan)
        ok = np.isfinite(curves).all(axis=0)  # range drawn once every seed has a feasible run
        ax.fill_between(
            n[ok], curves[:, ok].min(0), curves[:, ok].max(0), color=color, alpha=0.15, lw=0
        )
        ax.plot(n, med, marker=mk, color=color, ms=5, label=f"{label}, median of 10")
        out[name] = [None if np.isnan(v) else rnd(v, 2) for v in med]
    grid_best = best_feasible([town.run(*x) for x in found.grid])
    ax.plot([16], [grid_best], "^", ms=11, color=aml_style.PALETTE[2], label="4 × 4 grid, 16 runs")
    ax.axhline(today, color=aml_style.INK, ls="--", lw=1.4, label=f"today, {today:.2f}")
    ax.axhline(best_true, color=aml_style.INK, ls=":", lw=1.6, label=f"best, {best_true:.2f}")
    ax.set_xlabel("Simulator runs")
    ax.set_ylabel("Least feasible leakage (m³/h)")
    ax.set_xticks([1, 5, 10, 15])
    ax.legend(loc="upper right", fontsize=11)
    aml_style.save(fig, out_dir, "regret")
    return out


def extrapolation_figure(town: LTown, pool: Pool, found: Found, out_dir: Path) -> dict[str, object]:
    """BO's process for leakage, followed along the slice past the box, against the simulator.

    Args:
        town: The simulator.
        pool: Its worker processes.
        found: The settings each search ran (BO seed ``SEED``'s process is followed).
        out_dir: Where the figure goes.

    Returns:
        Where PRV-2 opens fully, and the truth, mean, and sd at ``EXTRAP_AT``.
    """
    s2 = np.arange(LO, EXTRAP_HI + 1e-9, 1.0)
    truth = pool.runs([(SLICE_S1, v) for v in s2])
    X = found.bo[SEED]
    g_leak = fit_gp(unit(X), np.array([town.run(*x).leak for x in X]))
    m, s = g_leak.predict(unit(np.column_stack([np.full_like(s2, SLICE_S1), s2])))
    leak = np.array([r.leak for r in truth])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.axvspan(HI, EXTRAP_HI, color=aml_style.MUTED, alpha=0.08, lw=0)
    ax.fill_between(s2, m - 2 * s, m + 2 * s, color=aml_style.PALETTE[0], alpha=0.18, lw=0)
    ax.plot(s2, m, color=aml_style.PALETTE[0], label="GP mean, ±2 sd")
    ax.plot(s2, leak, color=aml_style.INK, ls="--", lw=1.6, label="simulator")
    ax.set_xlabel("PRV-2 setting (m); PRV-1 at 36 m")
    ax.set_ylabel("Leakage (m³/h)")
    ax.text(
        HI + 1.5, float(np.min(m - 2 * s)), "beyond\nthe runs", fontsize=11, color=aml_style.MUTED
    )
    ax.legend(loc="upper left", fontsize=11)
    aml_style.save(fig, out_dir, "extrapolation")
    full_open = s2[np.array([r.open_steps[1] for r in truth]) == STEPS]
    assert full_open.size, "PRV-2 never fully open on the slice"
    at = {v: int(np.argmin(np.abs(s2 - v))) for v in EXTRAP_AT}
    return {
        "full_open_from": rnd(full_open.min(), 2),
        "leak_open": rnd(leak[at[EXTRAP_HI]], 2),
        **{
            f"at{int(v)}": {"truth": rnd(leak[i], 2), "mean": rnd(m[i], 2), "sd": rnd(s[i], 2)}
            for v, i in at.items()
        },
    }


def at_run_certainty(town: LTown, found: Found) -> dict[str, object]:
    """At BO's best run the process is sure it is feasible, with almost no pressure to spare.

    Args:
        town: The simulator.
        found: The settings each search ran (BO seed ``SEED``).

    Returns:
        The pressure to spare (m), the process's sd there (m), and its probability of feasibility.
    """
    X = found.bo[SEED]
    runs = [town.run(*x) for x in X]
    g_p = fit_gp(unit(X), np.array([r.pmin for r in runs]))
    j = int(np.argmin([r.leak if r.pmin >= P_SERVICE else np.inf for r in runs]))
    s = g_p.predict(unit([X[j]]))[1][0]
    return {
        "spare_m": rnd(runs[j].pmin - P_SERVICE, 2),
        "sd_m": float(f"{s:.1g}"),
        "pof": rnd(probability_feasible(g_p, unit([X[j]]))[0], 3),
    }


def boundary_corner(pool: Pool, found: Found, truth: np.ndarray) -> dict[str, object]:
    """The boundary's corner: from some PRV-1 setting on, any PRV-2 setting keeps 35 psi.

    Args:
        pool: The worker processes.
        found: The settings each search ran (BO seed ``SEED``'s model-guided runs are counted).
        truth: The exhaustive map, shape ``(15, 15, 2)``.

    Returns:
        The boundary at each ``CORNER_S1`` setting, the first setting from which any PRV-2
        setting is feasible, the least feasible leakage on the map from there on, and how many
        of BO seed ``SEED``'s model-guided runs went beyond the setting before it.

    Raises:
        RuntimeError: If no ``CORNER_S1`` setting is feasible at every PRV-2 setting.
    """
    s2s = pool.map([("boundary", (s1, P_SERVICE)) for s1 in CORNER_S1])
    first = next((s1 for s1, s2 in zip(CORNER_S1, s2s, strict=True) if s2 == LO), None)
    if first is None:
        raise RuntimeError(f"no corner in PRV-1 {CORNER_S1}: boundary {s2s}")
    G = map_grid()
    beyond = (G[..., 0] >= first) & (truth[..., 1] >= P_SERVICE)
    before = CORNER_S1[CORNER_S1.index(first) - 1]
    return {
        "s1": list(CORNER_S1),
        "s2": [None if v is None else rnd(v, 2) for v in s2s],
        "any_s2_from": first,
        "least_leak_beyond": rnd(truth[..., 0][beyond].min(), 2),
        "bo_seed0_guided_runs_beyond": sum(x[0] > before for x in found.bo[SEED][N_INIT:]),
    }


def margin_price(town: LTown, pool: Pool, opt: Pair) -> dict[str, object]:
    """The optimum when the lowest pressure must clear 35 psi by 0, 1, and 2 m, and by today's.

    The last answers whether today's settings are already the least leakage for the margin they
    keep: if so, the whole saving is the choice of margin, not a better pair of settings. The
    search resolves leakage only to about what ``BISECT_TOL`` of PRV-2 changes it by, so a
    difference of that size means "on the frontier", and nothing finer.

    Args:
        town: The simulator.
        pool: Its worker processes.
        opt: The optimum with no margin, m.

    Returns:
        Each margin's best settings and saving, the cost per metre, today's margin, the best
        settings for today's margin, today's excess leakage, and the search's resolution in
        leakage at those settings (m^3/h).
    """
    today = town.run(*TODAY)
    out, leaks = [], []
    for extra in MARGINS_M:
        o = opt if extra == 0 else pool.optimum(P_SERVICE + extra)
        r = town.run(*o)
        leaks.append(r.leak)
        saving = pct(1 - r.leak / today.leak, 1)
        out.append({"extra_m": extra, **run_dict(r, o), "saving_pct": saving})
    o = pool.optimum(today.pmin)
    best = town.run(*o)
    # Leakage per bisection bracket at the same-margin optimum: the search's resolution.
    below = pool.runs([(o[0], o[1] - BISECT_TOL)])[0]
    return {
        "levels": out,
        "per_m": rnd((leaks[-1] - leaks[0]) / (MARGINS_M[-1] - MARGINS_M[0]), 2),
        "two_m": rnd(leaks[-1] - leaks[0], 2),
        "today_spare": rnd(today.pmin - P_SERVICE, 2),
        "same_margin": run_dict(best, o),
        "today_excess": rnd(today.leak - best.leak, 2),
        "resolution": rnd(best.leak - below.leak, 3),
    }


def leak_exponent(opt: Pair) -> tuple[list[dict[str, object]], dict[str, float]]:
    """The prize under other leakage exponents: recalibrate, then find the optimum again.

    Args:
        opt: The optimum at N1 = 1, m.

    Returns:
        Per exponent: the best settings, today's and the best leakage, and the saving; and how far
        each setting moves across the exponents (max minus min, m).
    """
    out, best_settings = [], []
    for n1 in EXPONENTS:
        town = LTown(n1=n1)
        with Pool(town) as pool:
            o = opt if n1 == 1.0 else pool.optimum()
            today, best = town.run(*TODAY).leak, town.run(*o).leak
            best_settings.append(o)
            out.append(
                {
                    "n1": n1,
                    "s1": rnd(o[0], 2),
                    "s2": rnd(o[1], 2),
                    "today": rnd(today, 2),
                    "best": rnd(best, 2),
                    "saving_pct": pct(1 - best / today, 1),
                }
            )
    s1s, s2s = zip(*best_settings, strict=True)
    spread = {"s1_m": rnd(max(s1s) - min(s1s), 2), "s2_m": rnd(max(s2s) - min(s2s), 2)}
    return out, spread


def write_tex(raw: Trace, out_dir: Path) -> Path:
    """Macros for the diagram: one prediction traced through the process.

    Each number is formatted once, from its unrounded value (``lectures/README.md``).

    Args:
        raw: The attempt's trace at full precision.
        out_dir: Where ``numbers.tex`` goes.

    Returns:
        The path written.
    """
    macros = {
        "SliceS": tex_num(SLICE_S1, ".0f"),
        "TraceS": tex_num(TRACE_S2, ".0f"),
        "Ell": tex_num(raw.gp.ell[0] * (HI - LO), ".0f"),
        "Mean": tex_num(raw.mean, ".1f"),
        "Sd": tex_num(raw.sd, ".1f"),
        "Lo": tex_num(raw.mean - 2 * raw.sd, ".1f"),
        "Hi": tex_num(raw.mean + 2 * raw.sd, ".1f"),
        "Truth": tex_num(raw.truth, ".1f"),
        "Pserv": tex_num(P_SERVICE, ".1f"),
        "RunRows": r"\\".join(
            f"{tex_num(s2, '.0f')} $\\to$ {tex_num(p, '.2f')} m"
            for s2, p in zip(SLICE_RUNS, raw.pmin, strict=True)
        ),
        "Kvals": ", ".join(tex_num(v, ".2f") for v in raw.k),
        "Wvals": ", ".join(tex_num(v, ".3f") for v in raw.weights),
        "Ybar": tex_num(raw.gp.mu, ".2f"),
        "Pof": tex_num(raw.pof, ".3f"),
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


def run_time(town: LTown) -> dict[str, object]:
    """Seconds per simulator run, timed one at a time in this process (uncached runs).

    The key ``seconds`` is a timing, which ``tools/diff_numbers.py`` ignores.

    Args:
        town: The calibrated simulator.

    Returns:
        ``{"seconds": {"median", "max", "runs"}}`` over ``TIMED_RUNS``.
    """
    secs = []
    for pair in TIMED_RUNS:
        t0 = time.perf_counter()
        town.customer_pressure(*pair)  # one EPANET run, not cached
        secs.append(time.perf_counter() - t0)
    return {
        "seconds": {"median": rnd(np.median(secs), 2), "max": rnd(max(secs), 2), "runs": len(secs)}
    }


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

SURF_N: Final[int] = 41  # grid points per side of the 3D posterior surfaces
SD_SCALE: Final[float] = 2.0  # m^3/h: the posterior sd at the top of the colour scale
SAMPLES: Final[int] = 6  # functions drawn from the prior and from the posterior
# Lengthscales scanned for the evidence, m.
ELL_GRID: Final[tuple[float, ...]] = tuple(np.linspace(5.0, 120.0, 116).tolist())


def title_art_figure(town: LTown, gp: GP, out_dir: Path) -> None:
    """The title art: four glowing runs and the GP's golden mean through them.

    The attempt's slice (PRV-1 at 36 m), from four runs.

    Args:
        town: The simulator.
        gp: The process fitted to the attempt's four runs.
        out_dir: Where the art goes.
    """
    s2 = np.linspace(LO, HI, 200)
    m = gp.predict(unit(s2[:, None]))[0]
    runs = np.array([town.run(SLICE_S1, v).pmin for v in SLICE_RUNS])
    lo, hi = m.min(), m.max()

    def to_xy(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
        return list(zip(0.06 + 0.88 * (x - LO) / (HI - LO), 0.62 + 0.46 * (y - lo) / (hi - lo),
                        strict=True))  # fmt: skip

    c = title_art.Canvas(seed=12)
    c.ground(("#0d1418", "#131424", "#1d0d13", "#06060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.40, 0.50, 3), soft=4), "#4f1c14", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.54, 0.06, 0.42, 0.62, 3), soft=4), "#6e2818", 0.85)
    rim = c.disc((0.50, 1.24), 0.10, soft=4) - c.disc((0.50, 1.24), 0.085, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.50, 1.24), 0.085, soft=3), "#1d0a10", 0.92)
    c.glow(c.lines([to_xy(s2, m)], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    pts = to_xy(np.array(SLICE_RUNS), runs)
    c.glow(np.clip(sum(c.disc(q, 0.012, soft=1.5) for q in pts), 0, 1), "#ffe2a0", 1.0, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean forms in rust; above them a golden "
        "curve rises through four glowing points, the lecture's Gaussian process through its four "
        "simulator runs; a dark sun with a burning rim hangs at the top.",
    )


def samples_figure(town: LTown, gp: GP, out_dir: Path) -> dict[str, object]:
    """Functions from the prior and from the posterior after the four runs, on the slice.

    Args:
        town: The simulator.
        gp: The process fitted to the attempt's four runs.
        out_dir: Where the figure goes.

    Returns:
        How many functions each panel draws.
    """
    s2 = np.linspace(LO, HI, 141)
    U = unit(s2[:, None])
    X = gp.X
    Kss = matern52(U, U, gp.ell, gp.var)
    rng = np.random.default_rng((SEED, 12))
    prior = rng.multivariate_normal(np.zeros(len(U)), Kss + 1e-9 * np.eye(len(U)), SAMPLES)
    ks = matern52(U, X, gp.ell, gp.var)
    v = solve_triangular(gp.chol, ks.T, lower=True)
    post_cov = Kss - v.T @ v
    post_mean = ks @ gp.alpha
    post = rng.multivariate_normal(post_mean, post_cov + 1e-9 * np.eye(len(U)), SAMPLES)
    runs = np.array([town.run(SLICE_S1, v_).pmin for v_ in SLICE_RUNS])
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    for ax, draws, title in ((axes[0], prior, "before the runs: the kernel's curves"),
                             (axes[1], post, "after four runs: curves through them")):  # fmt: skip
        for f in draws:
            ax.plot(s2, gp.mu + gp.sd * f, color=aml_style.PALETTE[0], lw=1.2, ls="-", alpha=0.7)
        ax.axhline(P_SERVICE, color=aml_style.MUTED, ls=":", lw=1.4)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("PRV-2 setting (m)")
    m, sd = gp.predict(U)
    axes[1].fill_between(s2, m - 2 * sd, m + 2 * sd, color=aml_style.PALETTE[0], alpha=0.12, lw=0)
    axes[1].plot(SLICE_RUNS, runs, "o", ms=9, color=aml_style.PALETTE[1], zorder=5)
    axes[0].set_ylabel("lowest pressure (m)")
    aml_style.save(fig, out_dir, "samples")
    return {"samples": SAMPLES}


def evidence_figure(town: LTown, gp: GP, out_dir: Path) -> dict[str, object]:
    """The four runs' evidence against the lengthscale: fit term, complexity term, and total.

    The signal variance is held at the fitted process's, so the two terms trade visibly (refit
    at each lengthscale, the variance would absorb the misfit).

    Args:
        town: The simulator.
        gp: The process fitted to the attempt's four runs.
        out_dir: Where the figure goes.

    Returns:
        The lengthscale with the least negative log evidence on ``ELL_GRID`` (m).
    """
    y = np.array([town.run(SLICE_S1, v).pmin for v in SLICE_RUNS])
    X = unit(np.array(SLICE_RUNS)[:, None])
    z = (y - gp.mu) / gp.sd
    fit, comp, total = [], [], []
    for ell_m in ELL_GRID:
        L, alpha, nlml = _condition(X, z, np.array([ell_m / (HI - LO)]), gp.var)
        fit.append(0.5 * float(z @ alpha))
        comp.append(float(np.log(np.diag(L)).sum()))
        total.append(nlml)
    best = float(ELL_GRID[int(np.argmin(total))])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    curves = (
        (fit, aml_style.PALETTE[0], "--", r"misfit: $\frac{1}{2}z^\top K^{-1}z$"),
        (comp, aml_style.PALETTE[1], ":", r"complexity: $\frac{1}{2}\log|K|$"),
        (total, aml_style.INK, "-", r"$-$log evidence (+ const)"),
    )
    for vals, color, ls, label in curves:
        ax.plot(ELL_GRID, vals, color=color, lw=2.4, ls=ls, label=label)
    ax.axvline(best, color=aml_style.MUTED, lw=1, ls=":")
    ax.annotate(f"best: {best:.0f} m", (best, min(total)), xytext=(25, 35),
                textcoords="offset points", fontsize=12,
                arrowprops={"arrowstyle": "->"})  # fmt: skip
    ax.set(xlabel=r"lengthscale $\ell$ (m)", ylabel="nats")
    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    )  # opaque: the dotted line at the best lengthscale runs behind it
    aml_style.save(fig, out_dir, "evidence")
    return {"best_ell_m": round(best)}


def posterior3d_figure(town: LTown, found: Found, out_dir: Path) -> dict[str, object]:
    """Leakage GP over the box after 5 and after 15 runs of BO (seed 0), coloured by its sd.

    Args:
        town: The simulator.
        found: The settings each search ran (BO seed ``SEED``).
        out_dir: Where the figure goes.

    Returns:
        The median posterior sd over the box after 5 and after 15 runs (m^3/h).
    """
    X = found.bo[SEED]
    g = np.linspace(LO, HI, SURF_N)
    A, B = np.meshgrid(g, g)
    Ug = unit(np.column_stack([A.ravel(), B.ravel()]))
    fig = plt.figure(figsize=aml_style.FULL)
    out = {}
    for k, n in enumerate((N_INIT, BUDGET)):
        runs = [town.run(*x) for x in X[:n]]
        gp = fit_gp(unit(X[:n]), np.array([r.leak for r in runs]))
        m, sd = (v.reshape(A.shape) for v in gp.predict(Ug))
        out[f"sd_median_{n}"] = rnd(np.median(sd), 2)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        colors = plt.get_cmap("cividis")(np.clip(sd / SD_SCALE, 0, 1))  # dark: sure
        ax.plot_surface(A, B, m, facecolors=colors, rstride=1, cstride=1, lw=0, shade=False,
                        antialiased=False, zorder=1)  # fmt: skip
        for (a, b), r in zip(X[:n], runs, strict=True):
            ax.plot([a, a], [b, b], [15, r.leak], color=aml_style.PALETTE[1], lw=1.4, ls="-",
                    zorder=5)  # fmt: skip
            ax.scatter([a], [b], [r.leak], s=28, color=aml_style.PALETTE[1], edgecolor="white",
                       zorder=6)  # fmt: skip
        ax.set(zlim=(15, 45), xticks=[30, 45, 60], yticks=[30, 45, 60], zticks=[20, 30, 40])
        ax.set_xlabel("PRV-1 (m)", labelpad=8, fontsize=12)
        ax.set_ylabel("PRV-2 (m)", labelpad=8, fontsize=12)
        ax.set_title(f"{n} runs: median sd {np.median(sd):.1f} m³/h", loc="left", fontsize=13)
        aml_style.style3d(ax, elev=24, azim=-58)
        ax.set_box_aspect(None, zoom=0.85)  # smaller box: the moved labels stay inside the crop
    fig.text(0.5, -0.04, f"leakage (m³/h): GP mean, coloured by its sd (dark: sure; yellow: "
             f"{SD_SCALE:g} m³/h or more); runs as orange pins",
             ha="center", fontsize=11.5, color=aml_style.MUTED)  # fmt: skip
    aml_style.save(fig, out_dir, "posterior3d")
    return out


def acquisition_figure(town: LTown, found: Found, out_dir: Path) -> dict[str, object]:
    """EI x P(feasible) over the box after the 5 initial runs of BO (seed 0), and its peak.

    Args:
        town: The simulator.
        found: The settings each search ran (BO seed ``SEED``).
        out_dir: Where the figure goes.

    Returns:
        The next run, the best feasible leakage after five runs, the score at the peak and its
        two factors, the score's range along PRV-2 at the peak's PRV-1 setting, and the leakage
        process's lengthscales.
    """
    X = found.bo[SEED]
    init = X[:N_INIT]
    g_leak, g_p = fit_both(town, init)
    g = np.linspace(LO, HI, ACQ_N)
    A, B = np.meshgrid(g, g)
    U = unit(np.column_stack([A.ravel(), B.ravel()]))
    score = bo_score(g_leak, g_p, U, best_feasible([town.run(*x) for x in init])).reshape(A.shape)
    fig = plt.figure(figsize=aml_style.SIDE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    aml_style.surface3d(ax, A, B, score, cmap="cividis")
    nxt = X[N_INIT]
    i = np.unravel_index(int(np.argmax(score)), score.shape)
    assert key(A[i], B[i]) == key(*nxt), (A[i], B[i], nxt)  # the peak drawn is the run taken
    ax.scatter([A[i]], [B[i]], [score[i]], s=60, color=aml_style.PALETTE[1], edgecolor="white",
               zorder=6)  # fmt: skip
    for a, b in init:
        ax.scatter([a], [b], [0], s=22, color=aml_style.INK, zorder=5)
    ax.set_xlabel("PRV-1 (m)", labelpad=8, fontsize=12)
    ax.set_ylabel("PRV-2 (m)", labelpad=8, fontsize=12)
    ax.set(xticks=[30, 45, 60], yticks=[30, 45, 60])
    ax.set_title(f"next run: ({nxt[0]:.1f}, {nxt[1]:.1f}) m", loc="left", fontsize=13)
    aml_style.style3d(ax, elev=30, azim=-60)
    ax.set_box_aspect(None, zoom=0.85)  # smaller box: the moved labels stay inside the crop
    ax.tick_params(axis="z", pad=5)  # tick numbers clear of their tick marks
    aml_style.save(fig, out_dir, "acquisition")
    best5 = best_feasible([town.run(*x) for x in init])
    assert best5 is not None
    ei = expected_improvement(*g_leak.predict(U), best5).reshape(A.shape)
    pof = probability_feasible(g_p, U).reshape(A.shape)
    col = A[0] == A[i]  # the peak's PRV-1 setting, every PRV-2 setting
    return {
        "next": [rnd(nxt[0], 2), rnd(nxt[1], 2)],
        "best5": rnd(best5, 2),
        "peak": rnd(score[i], 2),
        "peak_ei": rnd(ei[i], 2),
        "peak_pof": rnd(pof[i], 2),
        "along_prv2": [rnd(score[:, col].min(), 2), rnd(score[:, col].max(), 2)],
        "ell1_m": rnd(g_leak.ell[0] * (HI - LO), 1),
        "ell2_at_bound": bool(np.isclose(g_leak.ell[1], ELL_BOUNDS[1])),
    }


def margin_figure(margins: dict[str, object], today: dict[str, object], out_dir: Path) -> None:
    """Least leakage against the pressure kept above 35 psi; today's settings on that curve.

    Args:
        margins: ``margin_price``'s numbers.
        today: Today's run, as ``run_dict`` gives it.
        out_dir: Where the figure goes.
    """
    levels = margins["levels"]
    same = margins["same_margin"]
    assert isinstance(levels, list) and isinstance(same, dict)
    xs = [lv["extra_m"] for lv in levels] + [margins["today_spare"]]
    ys = [lv["leak"] for lv in levels] + [same["leak"]]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(xs, ys, "o-", color=aml_style.PALETTE[0], lw=2.2, ms=8, label="best settings")
    ax.plot([margins["today_spare"]], [today["leak"]], "s", ms=12, mfc="none",
            mec=aml_style.PALETTE[1], mew=2.2, label="today's (40, 50) m")  # fmt: skip
    for x_, y_ in zip(xs, ys, strict=True):
        first = x_ == min(xs)
        ax.annotate(f"{y_:.2f}", (x_, y_), xytext=(8, -12) if first else (-8, 10),
                    textcoords="offset points", ha="left" if first else "right",
                    fontsize=11)  # the first one right: left of it is the y axis  # fmt: skip
    ax.set(xlabel="pressure kept above 35 psi (m)", ylabel="least leakage (m³/h)")
    ax.set_ylim(bottom=min(ys) - 0.3)  # room for the first label under its point
    ax.legend(loc="lower right", fontsize=11)
    aml_style.save(fig, out_dir, "margin")


def main() -> None:
    """Generate every figure and number for CEE Lecture 12."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    # One BLAS thread here as in the workers: the same sums in the same order everywhere, so
    # every search picks the same candidates in this script, its workers, and the notebook.
    threadpool_limits(1)
    town = LTown()
    with Pool(town) as pool:
        G = map_grid()
        truth = np.array([[r.leak, r.pmin] for r in pool.runs(G.reshape(-1, 2))])
        truth = truth.reshape(*G.shape[:2], 2)
        opt = pool.optimum()
        attempt, trace = attempt_and_trace(town)
        found = searches(pool)
        best = town.run(*opt)
        today = town.run(*TODAY)
        numbers = {
            "setting": setting_summary(town, pool) | network_figure(town, out_dir),
            "optimum": run_dict(best, opt)
            | {
                "saving": rnd(today.leak - best.leak, 2),
                "saving_pct": pct(1 - best.leak / today.leak, 1),
            },
            "attempt": attempt,
            "verify": verify(town, trace.gp),
            "lengthscale": lengthscale_figure(town, pool, out_dir),
            "searches": search_numbers(town, pool, found, opt, truth),
            "map": map_numbers(truth),
            "curves": regret_figure(town, found, opt, out_dir),
            "extrapolation": extrapolation_figure(town, pool, found, out_dir),
            "at_run": at_run_certainty(town, found),
            "margins": margin_price(town, pool, opt),
            "corner": boundary_corner(pool, found, truth),
        }
        bo_map_figure(town, found, opt, truth, out_dir)
        # the visuals that carry the slides
        title_art_figure(town, trace.gp, out_dir)
        numbers["samples"] = samples_figure(town, trace.gp, out_dir)
        numbers["evidence"] = evidence_figure(town, trace.gp, out_dir)
        numbers["posterior3d"] = posterior3d_figure(town, found, out_dir)
        numbers["acquisition"] = acquisition_figure(town, found, out_dir)
        margin_figure(numbers["margins"], run_dict(today, TODAY), out_dir)
        numbers["runs_main"] = town.runs
    numbers["run_time"] = run_time(town)
    numbers["exponent"], numbers["exponent_spread"] = leak_exponent(opt)
    aml_style.thumbnails(out_dir, ("samples", "posterior3d", "regret", "extrapolation"))
    write_numbers(numbers, out_dir)
    print(write_tex(trace, out_dir))


if __name__ == "__main__":
    main()
