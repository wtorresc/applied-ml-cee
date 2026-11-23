"""Figures and numbers for CEE Lecture 14: neural networks and backpropagation.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l14_neural_networks.py \
        --out build/fig/cee_l14_neural_networks

Writes the PNG figures, ``numbers.json`` (the numbers the slides quote from this lecture's runs;
quotes of Lecture 12 and the slides' own arithmetic are not in it), and ``numbers.tex`` (one run
traced through a trained network, for the TikZ diagram). Data: BattLeDIM's L-Town model from
``scripts/fetch_battledim.sh``, run with WNTR (see ``ltown.py``). Worker processes:
``L14_WORKERS`` (default: up to 16); every network, GP and run is seeded and computed in its own
process with one BLAS thread, so the numbers do not depend on the count.

The setting is Lecture 12's: two pressure-reducing valves feed Area A of L-Town; lower settings
leak less, but every customer needs 35 psi (24.6 m). Lecture 12 searched the settings with a
Gaussian process and 15 runs. Here a control room wants a surrogate that answers for any pair of
settings in the box, trained on runs made overnight. A network of ReLU units (two hidden layers
of 64) is trained from scratch, with a minimal reverse-mode autograd checked against PyTorch,
and compared with Lecture 12's Gaussian process on the same runs. A run takes about half a
second (``setting.run.seconds``), so every answer is checked against the simulator, on a 1-m
map of the whole box that no choice is made on: the training length is checked on a separate
validation design, and the decision's margin comes from held-out runs of another design.
"""

from __future__ import annotations

import itertools
import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits

import aml_style
import title_art
from common import parse_out_dir, rmse, rnd, seed_all, tex_num, write_numbers, write_tex_macros
from ltown import (
    BOX,
    GP_ELL_BOUNDS,
    GP_JITTER,
    GP_VAR_BOUNDS,
    P_SERVICE,
    TODAY,
    LTown,
    Run,
    run_many,
)
from ltown import STEPS as DAY_STEPS  # reported steps of one simulated day (289)

SEED: int = 0
LO, HI = BOX  # the box of valve settings the runs cover, m (Lecture 12's)
GRID_STEP: float = 1.0  # the map that checks every surrogate: 36 x 36 runs
BUDGETS: tuple[int, ...] = (16, 64, 256, 1024)  # runs per training set
DESIGNS: range = range(3)  # scrambled Sobol sequences; a budget is the first n points of one
VAL_SEED: int = 3  # a fourth Sobol design: validation runs that choose the training length
VAL_RUNS: int = 256
ENSEMBLE: int = 5  # networks per fit, one per initialization seed (Lakshminarayanan et al., 2017)
HIDDEN: tuple[int, ...] = (64, 64)  # units per hidden layer
STEPS: int = 20_000  # Adam steps: a compute budget, checked on validation runs (training_length)
STEP_CHOICES: tuple[int, ...] = (2_500, 5_000, 10_000, 20_000, 40_000)  # each doubles the last
BATCH: int = 128  # rows per step, reshuffled every pass (Lecture 13); all rows if fewer
INIT: str = "uniform"  # PyTorch's default for a linear layer: U(+-1/sqrt(fan-in))
VERIFY_STEPS: int = 1_000  # our Adam against torch.optim.Adam: a whole fit of 16 runs
PROBE_EVERY: int = 250  # steps between points of the training curve
SHORT_STEPS: int = 4_000  # a run stopped early, for the training-curve slide
SNAPSHOTS: tuple[int, ...] = (0, 10, 50, STEPS)  # steps at which the training sheet is drawn
LR: float = 3e-3  # peak step size, decayed to 0 by a cosine schedule (Loshchilov & Hutter, 2017)
BETAS: tuple[float, float] = (0.9, 0.999)  # Adam's averaging rates (Kingma & Ba, 2015)
ADAM_EPS: float = 1e-8
FD_STEP: float = 1e-6  # central finite differences for the gradient check
SLICE_S1: float = 36.0  # PRV-1 setting of the one-dimensional slice (Lecture 12's attempt)
ATTEMPT_S2: tuple[float, ...] = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0)  # the attempt's rows
TRACE: tuple[float, float] = (36.0, 47.0)  # the run traced through the diagram
EXTRAP_HI: float = 90.0  # the slice runs past the box to here, m
CAND_STEP: float = 0.2  # lattice of candidate settings for the decision, m
OPT_WINDOW: tuple[tuple[float, float], tuple[float, float]] = ((35.0, 37.2), (45.8, 48.0))
ALPHA: float = 0.05  # the margin covers 95% of held-out errors (split conformal, Lecture 11)
CAL_RUNS: int = 256  # held-out runs that set design d's margin: the first of design (d + 1) % 3
ACCURATE: float = 1e-4  # EPANET accuracy that re-checks the picks (the file's is 0.01)
GP_RESTARTS: int = 5
WORKERS: int = int(os.environ.get("L14_WORKERS", min(16, os.cpu_count() or 1)))
OUTPUTS: tuple[str, str] = ("leak", "pmin")  # leakage (m^3/h), lowest customer pressure (m)
BREAK_BUDGET: int = 256  # runs for the initialization and units tests (design 0, seeds 0-4)
STARTS: tuple[str, ...] = ("zero", "const", "he0")  # bad starts, plus the two unscaled ones
LR_CHECK: tuple[float, ...] = (1e-3, 1e-2)  # peak steps either side of LR: a post-hoc check
NEAR_LINE_M: float = 2.0  # "near a switch": within this of a fitted shut-off line, PRV-2 metres
TIMING_RUNS: int = 5  # simulator runs timed for the slides' "about half a second"
NEAR_M: float = 3.0  # band above the limit for the impact map (today's settings keep 3.58 m)
# The attempt's hand-made network, traced through the backpropagation build: two ramps with kinks
# at 33 and 43 m, output a + b*ramp1 + d*ramp2, at PRV-2 = 47 m (PRV-1 at 36 m).
HAND: Final[Mapping[str, float]] = MappingProxyType(
    {"a": 14.89, "b": 0.8, "d": -0.4, "c1": 33.0, "c2": 43.0, "s2": 47.0}
)
CMAP: str = "cividis"  # sequential and readable with colour-vision deficiency
BOWL_RADIUS: float = 0.4  # half-width of the close-up, in filter-normalized steps
BOWL_N: int = 61  # grid points per side of the close-up
PLANE_RANGE: tuple[float, float] = (-0.25, 1.25)  # plane coordinates around the three networks
PLANE_N: int = 91  # grid points per side of the plane
LINE_N: int = 121  # points on each straight line between two networks
LOSS_CAP: float = 45.0  # m^2: heights above it are drawn flat, so the valleys keep the scale

Params = list[np.ndarray]


# ---------------------------------------------------------------------------------------------
# A minimal reverse-mode autograd (the slide's listing)


class Var:
    """A value in the computation graph, its gradient, and how to send gradients back.

    Attributes:
        value: The array computed in the forward pass.
        grad: Derivative of the final output with respect to ``value``; filled by ``backward``.
        parents: The Vars this one was computed from.
        back: Maps this Var's gradient to one gradient per parent (vector-Jacobian products).
    """

    def __init__(
        self,
        value: np.ndarray | float,
        parents: tuple[Var, ...] = (),
        back: Callable[[np.ndarray], tuple[np.ndarray, ...]] | None = None,
    ) -> None:
        """Wrap ``value``; leaves have no parents."""
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)
        self.parents = parents
        self.back = back


def matmul(a: Var, b: Var) -> Var:
    """``a @ b``; the gradient flows back as ``g @ b.T`` and ``a.T @ g``."""
    return Var(a.value @ b.value, (a, b), lambda g: (g @ b.value.T, a.value.T @ g))


def add(a: Var, b: Var) -> Var:
    """``a + b`` with ``b`` a bias row broadcast over the batch; its gradient sums the rows."""
    return Var(a.value + b.value, (a, b), lambda g: (g, g.sum(axis=0)))


def relu(a: Var) -> Var:
    """``max(0, a)``; the gradient passes where the unit is on."""
    return Var(np.maximum(a.value, 0.0), (a,), lambda g: (g * (a.value > 0),))


def mse(a: Var, y: np.ndarray) -> Var:
    """Mean squared error against fixed targets ``y`` of the same shape (no broadcasting)."""
    assert a.value.shape == np.shape(y), (a.value.shape, np.shape(y))
    r = a.value - y
    return Var((r**2).mean(), (a,), lambda g: (g * 2.0 * r / r.size,))


def topo_order(out: Var) -> list[Var]:
    """Every Var that ``out`` depends on, each after its parents (depth-first)."""
    order: list[Var] = []
    seen: set[int] = set()

    def visit(v: Var) -> None:
        if id(v) not in seen:
            seen.add(id(v))
            for p in v.parents:
                visit(p)
            order.append(v)

    visit(out)
    return order


def backward(out: Var) -> None:
    """Fill ``grad`` for every Var that ``out`` depends on."""
    out.grad = np.ones_like(out.value)
    for v in reversed(topo_order(out)):  # users before inputs
        if v.back:
            for p, g in zip(v.parents, v.back(v.grad), strict=True):
                p.grad = p.grad + g


# ---------------------------------------------------------------------------------------------
# The network


def init_params(seed: int, sizes: Sequence[int], scheme: str = INIT) -> Params:
    """Weights and biases of a ReLU network: ``[W1, b1, W2, b2, ...]``.

    Args:
        seed: Seed of the generator.
        sizes: Units per layer, inputs first, outputs last.
        scheme: ``"uniform"`` draws weights and biases from U(-1/sqrt(fan-in), 1/sqrt(fan-in)),
            PyTorch's default for ``nn.Linear``; ``"he0"`` draws weights from N(0, 2 / fan-in)
            (He et al., 2015) with biases 0; ``"zero"`` sets everything to 0; ``"const"`` sets
            every weight to 0.1 and every bias to 0.

    Returns:
        The parameters, alternating weight matrices ``(fan_in, fan_out)`` and bias rows.

    Raises:
        ValueError: On an unknown scheme.
    """
    rng = np.random.default_rng(seed)
    params: Params = []
    for fan_in, fan_out in pairwise(sizes):
        bound = 1.0 / np.sqrt(fan_in)
        b = np.zeros(fan_out)
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


def forward(params: Sequence[Var], Z: np.ndarray) -> Var:
    """The network on inputs ``Z``: ReLU hidden layers, a linear output layer.

    Args:
        params: The weights and biases as Vars, ``[W1, b1, W2, b2, ...]``.
        Z: Inputs, shape ``(n, D)``.

    Returns:
        The output Var, shape ``(n, outputs)``.
    """
    h = Var(Z)
    for i in range(0, len(params) - 2, 2):
        h = relu(add(matmul(h, params[i]), params[i + 1]))
    return add(matmul(h, params[-2]), params[-1])


def predict(params: Params, Z: np.ndarray) -> np.ndarray:
    """Forward pass only.

    Args:
        params: The weights and biases (arrays).
        Z: Inputs, shape ``(n, D)``.

    Returns:
        The outputs, shape ``(n, outputs)``.
    """
    return forward([Var(p) for p in params], Z).value


def batches(n: int, seed: int) -> Iterator[np.ndarray | slice]:
    """Rows for each step: all of them if ``n <= BATCH``, else shuffled batches of ``BATCH``.

    Args:
        n: Training rows.
        seed: Seed of the shuffles (with the stream id 1, apart from the initialization).

    Yields:
        A slice of every row, or one batch's row indices; each pass is a fresh shuffle.

    Raises:
        ValueError: If ``n > BATCH`` does not split into whole batches.
    """
    if n <= BATCH:
        while True:
            yield slice(None)
    if n % BATCH:
        raise ValueError(f"{n} rows do not split into batches of {BATCH}")
    rng = np.random.default_rng((seed, 1))
    while True:
        order = rng.permutation(n)
        yield from order.reshape(-1, BATCH)


def train(
    Z: np.ndarray,
    Y: np.ndarray,
    seed: int,
    scheme: str = INIT,
    steps: int = STEPS,
    *,
    lr: float = LR,
    probe: Callable[[int, Params], None] | None = None,
) -> tuple[Params, float]:
    """Adam on minibatches with a cosine-decayed step, from scratch.

    Args:
        Z: Inputs, shape ``(n, D)``.
        Y: Targets, shape ``(n, outputs)``.
        seed: Seed of the initialization and of the shuffles.
        scheme: Initialization (see ``init_params``).
        steps: Adam steps.
        lr: Peak step size, decayed to 0 by a cosine.
        probe: Called as ``probe(t, params)`` every ``PROBE_EVERY`` steps, at the steps of
            ``SNAPSHOTS``, and after the last.

    Returns:
        The trained parameters and their loss on all training rows.
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
        if probe is not None and (t % PROBE_EVERY == 0 or t == steps or t in SNAPSHOTS):
            probe(t, params)
    return params, float(((predict(params, Z) - Y) ** 2).mean())


def _curve_job(
    job: tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray],
) -> tuple[list[tuple[int, float, float]], dict[int, Params]]:
    """Train one network; record (step, training RMS, validation RMS) of the lowest pressure.

    Args:
        job: Training inputs and targets, seed, steps, and the validation inputs and targets,
            all standardized.

    Returns:
        The history, in standardized output units (the caller converts it to metres), and the
        parameters at each step of ``SNAPSHOTS`` that the run reaches (step 0: the start).
    """
    Z, Y, seed, steps, Zp, Yp = job  # Zp, Yp: the validation runs
    history: list[tuple[int, float, float]] = []
    snaps = {0: init_params(seed, (Z.shape[1], *HIDDEN, Y.shape[1]))}

    def probe(t: int, params: Params) -> None:
        if t in SNAPSHOTS:
            snaps[t] = [p.copy() for p in params]
        if t % PROBE_EVERY and t != steps:
            return
        tr = predict(params, Z)[:, 1] - Y[:, 1]
        te = predict(params, Zp)[:, 1] - Yp[:, 1]
        history.append((t, float(np.sqrt((tr**2).mean())), float(np.sqrt((te**2).mean()))))

    with threadpool_limits(1):
        train(Z, Y, seed, steps=steps, probe=probe)
    return history, snaps


TrainJob = tuple[np.ndarray, np.ndarray, int, str, int, float]


def _train_job(job: TrainJob) -> tuple[Params, float, float]:
    """Train one network in a worker process, with one BLAS thread.

    Args:
        job: Inputs, targets, seed, initialization, steps, and peak step size.

    Returns:
        The parameters, their training loss, and the wall time, s.
    """
    Z, Y, seed, scheme, steps, lr = job
    with threadpool_limits(1):
        t0 = time.perf_counter()
        params, loss = train(Z, Y, seed, scheme, steps, lr=lr)
        return params, loss, time.perf_counter() - t0


@dataclass(frozen=True)
class Scaler:
    """Standardize columns with the training rows' mean and standard deviation."""

    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, A: np.ndarray) -> Scaler:
        """Fit on the training rows.

        Args:
            A: Training rows, shape ``(n, columns)``.

        Returns:
            The scaler.

        Raises:
            ValueError: If a column is constant (its standard deviation would divide by 0).
        """
        sd = A.std(axis=0)
        if not (sd > 0).all():
            raise ValueError(f"a constant column cannot be standardized: sd {sd}")
        return cls(A.mean(axis=0), sd)

    @classmethod
    def identity(cls, columns: int) -> Scaler:
        """A scaler that leaves ``columns`` columns in their own units (the break-it tests)."""
        return cls(np.zeros(columns), np.ones(columns))

    def __call__(self, A: np.ndarray) -> np.ndarray:
        """Standardize ``A`` (rows by columns)."""
        return (A - self.mean) / self.sd

    def inverse(self, A: np.ndarray) -> np.ndarray:
        """Undo the standardization of ``A`` (rows by columns)."""
        return A * self.sd + self.mean


@dataclass(frozen=True)
class Networks:
    """An ensemble of networks trained on the same runs, with its encoder and decoder scaling.

    Attributes:
        members: Parameters of each network.
        xs: Scaler of the settings (the encoder).
        ys: Scaler of the outputs.
        seconds: Wall time to train each network, s.
    """

    members: list[Params]
    xs: Scaler
    ys: Scaler
    seconds: list[float]

    def each(self, S: np.ndarray) -> np.ndarray:
        """Each member's predictions in output units, shape ``(members, n, outputs)``."""
        return np.stack([self.ys.inverse(predict(p, self.xs(S))) for p in self.members])

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """The ensemble's mean prediction, shape ``(n, outputs)``."""
        return self.each(S).mean(axis=0)


def submit_networks(
    pool: ProcessPoolExecutor,
    S: np.ndarray,
    Y: np.ndarray,
    seeds: Sequence[int],
    *,
    scheme: str = INIT,
    raw_settings: bool = False,
    raw_outputs: bool = False,
    steps: int = STEPS,
    lr: float = LR,
) -> Callable[[], Networks]:
    """Start training an ensemble in the pool; call the result to wait for it.

    Args:
        pool: Worker processes.
        S: Settings, shape ``(n, 2)``.
        Y: Outputs, shape ``(n, 2)``.
        seeds: One network per seed.
        scheme: Initialization (see ``init_params``).
        raw_settings: Train on the settings in metres (a break-it test).
        raw_outputs: Train on the outputs in their own units (a break-it test).
        steps: Adam steps.
        lr: Peak step size.

    Returns:
        A function that waits for the networks and returns them.
    """
    xs = Scaler.identity(S.shape[1]) if raw_settings else Scaler.fit(S)
    ys = Scaler.identity(Y.shape[1]) if raw_outputs else Scaler.fit(Y)
    futures = [pool.submit(_train_job, (xs(S), ys(Y), seed, scheme, steps, lr)) for seed in seeds]

    def collect() -> Networks:
        done = [f.result() for f in futures]
        return Networks([p for p, _, _ in done], xs, ys, [t for _, _, t in done])

    return collect


# ---------------------------------------------------------------------------------------------
# Lecture 12's Gaussian process (scikit-learn, which matched Lecture 12's from-scratch GP)


def unit_box(S: np.ndarray) -> np.ndarray:
    """Valve settings scaled to the unit box, as in Lecture 12.

    Args:
        S: Settings in metres, shape ``(n, 2)``.

    Returns:
        The settings in [0, 1], same shape.
    """
    return (S - LO) / (HI - LO)


def _fit_gp(job: tuple[np.ndarray, np.ndarray]) -> tuple[GaussianProcessRegressor, float, bool]:
    """Fit one output's GP in a worker (Matérn 5/2, one lengthscale per setting, evidence).

    The evidence may prefer a hyperparameter at Lecture 12's bounds; the fit is then the maximum
    within them, which is what the lecture compares. scikit-learn's warning about it is replaced
    by the returned flag (and counted on the slides' numbers); any other warning is re-issued.

    Args:
        job: Settings in metres, shape ``(n, 2)``, and one output, shape ``(n,)``.

    Returns:
        The fitted process, its wall time (s), and whether a hyperparameter sits at a bound.
    """
    X, y = job
    kernel = ConstantKernel(1.0, GP_VAR_BOUNDS) * Matern(np.ones(2), GP_ELL_BOUNDS, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel,
        alpha=GP_JITTER,
        normalize_y=True,
        n_restarts_optimizer=GP_RESTARTS,
        random_state=SEED,
    )
    with threadpool_limits(1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        gp.fit(unit_box(X), y)
        seconds = time.perf_counter() - t0
    for w in caught:
        bound = str(w.message).startswith("The optimal value found")
        if not (issubclass(w.category, ConvergenceWarning) and bound):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    theta, bounds = gp.kernel_.theta, gp.kernel_.bounds  # log scale
    at_bound = bool(np.any(np.isclose(theta[:, None], bounds, atol=1e-2)))
    return gp, seconds, at_bound


@dataclass(frozen=True)
class GPs:
    """One Gaussian process per output, fitted to the same runs."""

    models: list[GaussianProcessRegressor]
    seconds: list[float]
    at_bound: list[bool]

    def mean(self, S: np.ndarray) -> np.ndarray:
        """Posterior means at settings ``S`` (m), shape ``(n, outputs)``."""
        return np.stack([g.predict(unit_box(S)) for g in self.models], axis=1)

    def mean_sd(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior means and standard deviations at settings ``S``, each ``(n, outputs)``."""
        out = [g.predict(unit_box(S), return_std=True) for g in self.models]
        return np.stack([m for m, _ in out], axis=1), np.stack([s for _, s in out], axis=1)


def submit_gps(pool: ProcessPoolExecutor, S: np.ndarray, Y: np.ndarray) -> Callable[[], GPs]:
    """Start fitting one GP per output in the pool; call the result to wait for them.

    Args:
        pool: Worker processes.
        S: Settings, shape ``(n, 2)``.
        Y: Outputs, shape ``(n, 2)``.

    Returns:
        A function that waits for the fits and returns them.
    """
    futures = [pool.submit(_fit_gp, (S, Y[:, j])) for j in range(len(OUTPUTS))]

    def collect() -> GPs:
        done = [f.result() for f in futures]
        return GPs([g for g, _, _ in done], [t for _, t, _ in done], [b for _, _, b in done])

    return collect


# ---------------------------------------------------------------------------------------------
# Runs of the simulator


def map_grid() -> np.ndarray:
    """Every 1-m pair of settings in the box, PRV-2 varying fastest.

    Returns:
        Settings in metres, shape ``(1296, 2)``.
    """
    g = np.arange(LO, HI + 1e-9, GRID_STEP)
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)


def lattice(lo1: float, hi1: float, lo2: float, hi2: float) -> np.ndarray:
    """Candidate settings every ``CAND_STEP`` m over a rectangle.

    Args:
        lo1: Lowest PRV-1 setting, m.
        hi1: Highest PRV-1 setting, m.
        lo2: Lowest PRV-2 setting, m.
        hi2: Highest PRV-2 setting, m.

    Returns:
        The candidates, shape ``(n, 2)``.
    """
    a = np.round(np.arange(lo1, hi1 + 1e-9, CAND_STEP), 6)
    b = np.round(np.arange(lo2, hi2 + 1e-9, CAND_STEP), 6)
    return np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)


def designs() -> np.ndarray:
    """Scrambled Sobol sequences over the box, one per seed of ``DESIGNS``.

    Returns:
        Settings in metres, shape ``(len(DESIGNS), max(BUDGETS), 2)``.
    """
    return (
        np.stack([qmc.Sobol(2, scramble=True, seed=s).random(max(BUDGETS)) for s in DESIGNS])
        * (HI - LO)
        + LO
    )


def validation_design() -> np.ndarray:
    """The validation runs' settings: the first ``VAL_RUNS`` of a fourth Sobol sequence."""
    return qmc.Sobol(2, scramble=True, seed=VAL_SEED).random(VAL_RUNS) * (HI - LO) + LO


def outputs(runs: Sequence[Run]) -> np.ndarray:
    """Leakage and lowest customer pressure of each run.

    Args:
        runs: Simulated days.

    Returns:
        Leakage (m^3/h) and lowest pressure (m), shape ``(len(runs), 2)``.
    """
    return np.array([(r.leak, r.pmin) for r in runs])


@dataclass(frozen=True)
class Data:
    """Every run the lecture uses.

    Attributes:
        grid: The 1-m map's settings (1296, 2), and ``grid_y`` its outputs.
        grid_runs: Its runs (valve states, critical node).
        design: Training settings (designs, runs, 2), and ``design_y`` their outputs.
        slice_s2: PRV-2 settings of the slice at PRV-1 = ``SLICE_S1``, and ``slice_y``.
        window: The fine lattice around the optimum, and ``window_y``.
        val: The validation runs' settings (``VAL_RUNS``, 2), and ``val_y``.
    """

    grid: np.ndarray
    grid_y: np.ndarray
    grid_runs: list[Run]
    design: np.ndarray
    design_y: np.ndarray
    slice_s2: np.ndarray
    slice_y: np.ndarray
    window: np.ndarray
    window_y: np.ndarray
    val: np.ndarray
    val_y: np.ndarray


def simulate() -> Data:
    """Run the simulator for the map, the designs, the slice, the window, and the validation.

    Returns:
        Every run, in ``WORKERS`` processes (the results do not depend on the count).
    """
    grid, design, val = map_grid(), designs(), validation_design()
    slice_s2 = np.arange(LO, EXTRAP_HI + 1e-9, GRID_STEP)
    window = lattice(*OPT_WINDOW[0], *OPT_WINDOW[1])
    parts = [grid, design.reshape(-1, 2), np.c_[np.full_like(slice_s2, SLICE_S1), slice_s2]]
    parts += [window, val]
    pairs = np.concatenate(parts)
    runs = run_many([tuple(p) for p in pairs], WORKERS)
    y = outputs(runs)
    i = np.cumsum([0, *(len(q) for q in parts)])
    assert i[-1] == len(pairs)
    return Data(
        grid=grid,
        grid_y=y[: i[1]],
        grid_runs=runs[: i[1]],
        design=design,
        design_y=y[i[1] : i[2]].reshape(design.shape),
        slice_s2=slice_s2,
        slice_y=y[i[2] : i[3]],
        window=window,
        window_y=y[i[3] : i[4]],
        val=val,
        val_y=y[i[4] :],
    )


# ---------------------------------------------------------------------------------------------
# Number formatting


def errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, float]]:
    """RMSE and worst absolute error of each output over the map.

    Args:
        pred: Predictions, shape ``(n, 2)``.
        truth: The simulator's outputs, same shape.

    Returns:
        For each output, its ``rmse`` and ``worst`` error.
    """
    e = pred - truth
    return {
        name: {"rmse": rmse(pred[:, j], truth[:, j]), "worst": float(np.abs(e[:, j]).max())}
        for j, name in enumerate(OUTPUTS)
    }


# ---------------------------------------------------------------------------------------------
# Lecture sections


def shut_lines(data: Data) -> dict[str, dict[str, float]]:
    """Where each valve passes no water all day, from the simulator's valve states.

    PRV-2 is closed all day below a line (PRV-1's water reaches its outlet with more head than
    its setting); PRV-1 is closed all day above another. Each line is fitted, by least squares,
    to the 1-m map's boundary: for each PRV-1 setting, the highest PRV-2 setting with PRV-2
    closed all day, and the lowest with PRV-1 closed all day.

    Returns:
        For each valve, the line ``s2 = slope * s1 + intercept`` and its offset ``s2 - s1`` at
        the box's centre.

    Raises:
        ValueError: If no run of the map has a valve closed all day (``DAY_STEPS`` steps).
    """
    closed = np.array([r.closed_steps for r in data.grid_runs])
    if closed.max() != DAY_STEPS:
        raise ValueError(f"no map run has a valve closed all {DAY_STEPS} steps: {closed.max()}")
    s1s = np.unique(data.grid[:, 0])
    out = {}
    for name, valve, pick in (("prv2", 1, np.max), ("prv1", 0, np.min)):
        pts = []
        for s1 in s1s:
            row = data.grid[:, 0] == s1
            shut = data.grid[row][closed[row, valve] == DAY_STEPS, 1]
            if len(shut):
                pts.append((s1, pick(shut)))
        p = np.array(pts)
        slope, icpt = np.polyfit(p[:, 0], p[:, 1], 1)
        centre = 0.5 * (LO + HI)
        out[name] = {
            "slope": float(slope),
            "intercept": float(icpt),
            "offset": float(slope * centre + icpt - centre),
            "points": len(p),
        }
    return out


def setting(data: Data, town: LTown, lines: dict, out_dir: Path) -> dict[str, object]:
    """The 1-m map: regimes of the two valves and the lowest pressure; the regimes figure.

    Args:
        data: Every run.
        town: The calibrated model (for Area B's junctions).
        lines: The fitted shut-off lines (``shut_lines``).
        out_dir: Figure directory.

    Returns:
        The map's size, the valves' regimes (closed all day), the critical customers, and the
        ceiling Area B sets.

    Raises:
        ValueError: If the highest pressure is not set in Area B, or the slice exceeds it.
    """
    runs = data.grid_runs
    closed = np.array([r.closed_steps for r in runs])
    full = DAY_STEPS  # closed at every reported step of the day (shut_lines checks some are)
    nodes = np.array([r.node for r in runs])
    pmin = data.grid_y[:, 1]
    cap = float(pmin.max())
    at_cap = nodes[pmin == cap]
    if not set(at_cap) <= set(town.area_b):
        raise ValueError(f"the cap is not set in Area B: {set(at_cap)}")
    if pmin.max() != data.slice_y[:, 1].max():
        raise ValueError("the slice's highest pressure is not the map's cap")
    regimes = {
        "prv2_shut": int((closed[:, 1] == full).sum()),
        "prv1_shut": int((closed[:, 0] == full).sum()),
    }
    regimes["both"] = len(runs) - regimes["prv2_shut"] - regimes["prv1_shut"]
    critical = {str(n): int(c) for n, c in zip(*np.unique(nodes, return_counts=True), strict=True)}
    regimes_figure(data, lines, out_dir)
    return {
        "map_runs": len(runs),
        "regimes": regimes,
        "critical": critical,
        "cap_m": rnd(cap, 2),
        "cap_node": str(at_cap[0]),
        "lines": {
            k: {"offset": rnd(v["offset"], 2), "slope": rnd(v["slope"], 2)}
            for k, v in lines.items()
        },
        "leak_range": [rnd(data.grid_y[:, 0].min(), 2), rnd(data.grid_y[:, 0].max(), 2)],
        "pmin_range": [rnd(pmin.min(), 2), rnd(cap, 2)],
    }


def run_seconds(town: LTown, data: Data) -> dict[str, float]:
    """Time single simulator runs in this process: the slides' "about half a second".

    Args:
        town: A calibrated model with no cached runs of the validation settings.
        data: Every run (the first ``TIMING_RUNS`` validation settings are timed again).

    Returns:
        The median wall time of one run, s (a timing: ``make verify`` ignores it).
    """
    times = []
    for s in data.val[:TIMING_RUNS]:
        t0 = time.perf_counter()
        town.run(*s)
        times.append(time.perf_counter() - t0)
    return {"seconds": float(np.median(times))}


def _line_s2(line: dict, s1: np.ndarray) -> np.ndarray:
    """PRV-2 setting (m) on a fitted shut-off line at PRV-1 settings ``s1`` (m)."""
    return line["slope"] * s1 + line["intercept"]


def regimes_figure(data: Data, lines: dict, out_dir: Path) -> None:
    """Contours of the lowest pressure over the box, with the valves' shut-off lines.

    Args:
        data: Every run (the map).
        lines: The fitted shut-off lines.
        out_dir: Figure directory.
    """
    n = int(round((HI - LO) / GRID_STEP)) + 1
    g = np.arange(LO, HI + 1e-9, GRID_STEP)
    P = data.grid_y[:, 1].reshape(n, n).T  # rows PRV-2, columns PRV-1
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    cs = ax.contour(g, g, P, levels=np.arange(8, 33, 4), colors=aml_style.MUTED, linewidths=1.0)
    for t in ax.clabel(cs, fmt="%d", fontsize=11):  # above the bold 24.6 m line
        t.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        t.set_zorder(5)
    ax.contour(g, g, P, levels=[P_SERVICE], colors=aml_style.INK, linewidths=2.6)
    s1 = np.linspace(LO, HI, 50)
    styles = {"prv2": (aml_style.PALETTE[0], "--"), "prv1": (aml_style.PALETTE[1], "-.")}
    labels = {"prv2": "PRV-2 shut below", "prv1": "PRV-1 shut above"}
    for k, (color, ls) in styles.items():
        s2 = _line_s2(lines[k], s1)
        keep = (s2 >= LO) & (s2 <= HI)
        ax.plot(s1[keep], s2[keep], color=color, ls=ls, lw=2.6, label=labels[k])
    ax.plot([], [], color=aml_style.INK, lw=2.6, label="24.6 m (35 psi)")
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xlabel("PRV-1 setting (m)")
    ax.set_ylabel("PRV-2 setting (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=12, handlelength=2.4)
    aml_style.save(fig, out_dir, "regimes")


@dataclass(frozen=True)
class RampFit:
    """The attempt's best two ramps: kinks ``c1`` and ``c2`` (m) and coefficients (a, b, d)."""

    c1: float
    c2: float
    coef: np.ndarray


def attempt(data: Data) -> tuple[dict[str, object], RampFit]:
    """The attempt's table, and the best two-ramp fit of the whole slice inside the box.

    For each pair of kinks on the 1-m slice, the rest is fitted by least squares; the pair whose
    least-squares fit has the smallest worst error is kept. The straight line is least squares.

    Args:
        data: Every run (the slice).

    Returns:
        The numbers for the slides, and the kept fit (for the ramp build).
    """
    inside = data.slice_s2 <= HI
    s2, p = data.slice_s2[inside], data.slice_y[inside, 1]
    rows = [{"s2": rnd(v, 2), "pmin": rnd(p[s2 == v][0], 2)} for v in ATTEMPT_S2]
    best = None
    for c1 in s2[1:-1]:
        for c2 in s2[(s2 > c1) & (s2 < s2[-1])]:
            A = np.c_[np.ones_like(s2), np.maximum(0, s2 - c1), np.maximum(0, s2 - c2)]
            coef, *_ = np.linalg.lstsq(A, p, rcond=None)
            worst = float(np.abs(A @ coef - p).max())
            if best is None or worst < best[0]:
                best = (worst, c1, c2, coef)
    assert best is not None
    worst, c1, c2, coef = best
    one = np.c_[np.ones_like(s2), s2]
    line, *_ = np.linalg.lstsq(one, p, rcond=None)
    return {
        "s1": SLICE_S1,
        "rows": rows,
        "ramps": {
            "c1": rnd(c1, 2),
            "c2": rnd(c2, 2),
            "a": rnd(coef[0], 2),
            "b": rnd(coef[1], 2),
            "d": rnd(coef[2], 2),
            "worst": rnd(worst, 2),
        },
        "line_worst": rnd(np.abs(one @ line - p).max(), 2),
    }, RampFit(float(c1), float(c2), coef)


def _torch_forward(tp: list[torch.Tensor], Z: np.ndarray) -> torch.Tensor:
    """The same network in PyTorch, float64: ReLU hidden layers, a linear output layer.

    Args:
        tp: Weights and biases as tensors.
        Z: Inputs, shape ``(n, D)``.

    Returns:
        The outputs, shape ``(n, outputs)``.
    """
    h = torch.tensor(Z)
    for i in range(0, len(tp) - 2, 2):
        h = torch.relu(h @ tp[i] + tp[i + 1])
    return h @ tp[-2] + tp[-1]


def verify(data: Data) -> dict[str, float | int]:
    """Our gradients against finite differences and PyTorch; our Adam against torch.optim.Adam.

    Uses the first 16 runs of design 0, standardized, and the lecture's initialization (uniform,
    PyTorch's default) of seed 0.

    Args:
        data: Every run (design 0's first 16).

    Returns:
        The weight count, the largest differences from PyTorch and from finite differences, the
        largest gradient, and the Adam comparison.
    """
    S, Y = data.design[0, :16], data.design_y[0, :16]
    Z, T = Scaler.fit(S)(S), Scaler.fit(Y)(Y)
    params = init_params(SEED, (2, *HIDDEN, 2))
    return gradient_checks(Z, T, params) | adam_check(Z, T, params)


def gradient_checks(Z: np.ndarray, T: np.ndarray, params: Params) -> dict[str, float | int]:
    """Our backward pass against PyTorch's autograd and central finite differences, float64.

    Args:
        Z: Standardized inputs.
        T: Standardized targets.
        params: The starting weights.

    Returns:
        The weight count, the starting loss, both largest differences, and the largest gradient.
    """
    leaves = [Var(p) for p in params]
    out = mse(forward(leaves, Z), T)
    backward(out)
    ours = [w.grad for w in leaves]

    # PyTorch's autograd on the same graph, float64
    tp = [torch.tensor(p, requires_grad=True) for p in params]
    ((_torch_forward(tp, Z) - torch.tensor(T)) ** 2).mean().backward()
    torch_diff = max(float(np.abs(g - t.grad.numpy()).max()) for g, t in zip(ours, tp, strict=True))

    # central finite differences, every weight
    def loss_at(ps: Params) -> float:
        return float(mse(forward([Var(p) for p in ps], Z), T).value)

    fd_err = 0.0
    for i, p in enumerate(params):
        for idx in np.ndindex(p.shape):
            up = [q.copy() for q in params]
            dn = [q.copy() for q in params]
            up[i][idx] += FD_STEP
            dn[i][idx] -= FD_STEP
            fd = (loss_at(up) - loss_at(dn)) / (2 * FD_STEP)
            fd_err = max(fd_err, abs(fd - ours[i][idx]))
    return {
        "weights": int(sum(p.size for p in params)),
        "loss0": float(out.value),
        "torch_grad_diff": torch_diff,
        "fd_max_err": fd_err,
        "max_grad": float(max(np.abs(g).max() for g in ours)),
    }


def adam_check(Z: np.ndarray, T: np.ndarray, params: Params) -> dict[str, float | int]:
    """Our training loop against torch.optim.Adam: same start, same cosine-decayed steps.

    Args:
        Z: Standardized inputs.
        T: Standardized targets.
        params: The starting weights (``init_params(SEED, ...)``, as ``train`` draws them).

    Returns:
        The largest weight difference after ``VERIFY_STEPS`` steps, the loss difference, our
        final loss, and the step count.
    """
    trained, our_loss = train(Z, T, SEED, steps=VERIFY_STEPS)
    tp = [torch.tensor(p, requires_grad=True) for p in params]
    opt = torch.optim.Adam(tp, lr=LR, betas=BETAS, eps=ADAM_EPS)
    for t in range(1, VERIFY_STEPS + 1):
        for group in opt.param_groups:
            group["lr"] = LR * 0.5 * (1.0 + np.cos(np.pi * (t - 1) / VERIFY_STEPS))
        opt.zero_grad()
        ((_torch_forward(tp, Z) - torch.tensor(T)) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        torch_loss = float(((_torch_forward(tp, Z) - torch.tensor(T)) ** 2).mean())
        weight_diff = max(
            float(np.abs(a - b.numpy()).max()) for a, b in zip(trained, tp, strict=True)
        )
    return {
        "torch_weight_diff": weight_diff,
        "loss_diff": abs(our_loss - torch_loss),
        "final_loss": our_loss,
        "steps": VERIFY_STEPS,
    }


@dataclass(frozen=True)
class Fits:
    """Every surrogate the lecture trains.

    Attributes:
        nets: ``nets[d][n]``, five networks on the first n runs of design d.
        gps: ``gps[d][n]``, one GP per output on the same runs.
        starts: Five networks (seeds 0-4) on ``BREAK_BUDGET`` runs of design 0 per bad start:
            ``STARTS``, plus ``raw`` (settings in metres) and ``raw_out`` (outputs in their own
            units).
        curves: ``curves[steps]``, the training curve of network 0 on 1,024 runs of design 0,
            against the validation runs.
        snaps: Its parameters at each step of ``SNAPSHOTS``, for the full schedule.
        lengths: Network 0 on 1,024 runs trained to each length in ``STEP_CHOICES``.
        rates: Network 0 on 1,024 runs at each peak step of ``LR_CHECK`` (``STEPS`` steps).
    """

    nets: dict[int, dict[int, Networks]]
    gps: dict[int, dict[int, GPs]]
    starts: dict[str, Networks]
    curves: dict[int, list[tuple[int, float, float]]]
    snaps: dict[int, Params]
    lengths: dict[int, Networks]
    rates: dict[float, Networks]


def fit_all(pool: ProcessPoolExecutor, data: Data) -> Fits:
    """Submit every fit at once, the longest first, so the pool stays busy; then wait.

    Args:
        pool: Worker processes.
        data: Every run.

    Returns:
        Every fitted surrogate.
    """
    keys = [(d, n) for n in sorted(BUDGETS, reverse=True) for d in DESIGNS]
    gps = {
        k: submit_gps(pool, data.design[k[0], : k[1]], data.design_y[k[0], : k[1]]) for k in keys
    }
    nets = {
        k: submit_networks(
            pool, data.design[k[0], : k[1]], data.design_y[k[0], : k[1]], range(ENSEMBLE)
        )
        for k in keys
    }
    S, Y = data.design[0, :BREAK_BUDGET], data.design_y[0, :BREAK_BUDGET]
    seeds = range(ENSEMBLE)
    starts = {name: submit_networks(pool, S, Y, seeds, scheme=name) for name in STARTS}
    starts["raw"] = submit_networks(pool, S, Y, seeds, raw_settings=True)
    starts["raw_out"] = submit_networks(pool, S, Y, seeds, raw_outputs=True)
    # the training length: network 0 on 1,024 runs, each schedule run to its end
    n = max(BUDGETS)
    S, Y = data.design[0, :n], data.design_y[0, :n]
    lengths = {k: submit_networks(pool, S, Y, [SEED], steps=k) for k in STEP_CHOICES}
    rates = {lr: submit_networks(pool, S, Y, [SEED], lr=lr) for lr in LR_CHECK}
    # training curves: the same network, as trained and stopped at SHORT_STEPS
    xs, ys = Scaler.fit(S), Scaler.fit(Y)
    job = (xs(S), ys(Y), SEED)
    probe = (xs(data.val), ys(data.val_y))
    curves = {k: pool.submit(_curve_job, (*job, k, *probe)) for k in (STEPS, SHORT_STEPS)}
    done = {k: f.result() for k, f in curves.items()}
    return Fits(
        {d: {n: nets[d, n]() for n in BUDGETS} for d in DESIGNS},
        {d: {n: gps[d, n]() for n in BUDGETS} for d in DESIGNS},
        {name: wait() for name, wait in starts.items()},
        {k: history for k, (history, _) in done.items()},
        done[STEPS][1],
        {k: wait() for k, wait in lengths.items()},
        {lr: wait() for lr, wait in rates.items()},
    )


def training_length(fits: Fits, data: Data) -> dict[str, object]:
    """The training length against the validation runs (a fourth design; the map is not used).

    Network 0 on 1,024 runs is trained to each length in ``STEP_CHOICES``. On a deterministic
    simulator its validation error keeps falling with every doubling (no overfitting), so no
    length is "where the curve flattens": ``STEPS`` is a compute budget, and longer training
    could only lower the networks' errors in the comparison with the GP.

    The peak step size was not searched: ``LR`` is a standard Adam rate. As a post-hoc check
    (never used to choose), the same network is trained at the steps of ``LR_CHECK`` either side,
    and the validation errors are recorded with whether ``LR``'s is the lowest of the three.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the validation runs).

    Returns:
        The validation RMS error of the lowest pressure at each length (m), the share each
        doubling removes, and the step-size check.

    Raises:
        ValueError: If a longer schedule ends with a larger validation error (overfitting would
            make the length a choice, not a budget), or if ``STEPS`` is not among the lengths.
    """
    if STEPS not in STEP_CHOICES:
        raise ValueError(f"{STEPS} steps is not one of the checked lengths {STEP_CHOICES}")
    val = {k: rmse(net(data.val)[:, 1], data.val_y[:, 1]) for k, net in fits.lengths.items()}
    gains = [1 - val[b] / val[a] for a, b in itertools.pairwise(STEP_CHOICES)]
    if min(gains) <= 0:
        raise ValueError(f"validation error rose with a longer schedule: {val}")
    by_lr = {lr: rmse(net(data.val)[:, 1], data.val_y[:, 1]) for lr, net in fits.rates.items()}
    by_lr[LR] = val[STEPS]
    return {
        "val_rmse": {str(k): v for k, v in val.items()},
        "gain_per_doubling": gains,
        "steps": STEPS,
        "val_runs": VAL_RUNS,
        "lr_check": {
            "val_rmse": {f"{lr:g}": by_lr[lr] for lr in sorted(by_lr)},
            "lr_lowest": bool(min(by_lr, key=by_lr.get) == LR),
        },
    }


def near_switch(grid: np.ndarray, lines: dict) -> np.ndarray:
    """Which settings lie within ``NEAR_LINE_M`` of either fitted shut-off line.

    Args:
        grid: Settings in metres, shape ``(n, 2)``.
        lines: The fitted shut-off lines (``shut_lines``).

    Returns:
        A boolean mask over the settings; the distance is along PRV-2, in metres.
    """
    dist = [np.abs(grid[:, 1] - _line_s2(line, grid[:, 0])) for line in lines.values()]
    return np.minimum(*dist) <= NEAR_LINE_M


def fit_seconds(fits: Fits) -> dict[str, list[float]]:
    """Wall time per fit by budget, median over designs (printed to the log, never quoted).

    Args:
        fits: Every fitted surrogate.

    Returns:
        The GP's (both outputs) and one network's seconds at each budget.
    """
    return {
        "gp_fit_s": [
            float(np.median([sum(fits.gps[d][n].seconds) for d in DESIGNS])) for n in BUDGETS
        ],
        "net_fit_s": [
            float(np.median([np.mean(fits.nets[d][n].seconds) for d in DESIGNS])) for n in BUDGETS
        ],
    }


def compare(fits: Fits, data: Data, lines: dict, out_dir: Path) -> dict[str, object]:
    """Errors over the 1-m map, median over designs, by budget; the learning-curve figure.

    Also splits the lowest pressure's error between the settings near a valve's shut-off line
    (``near_switch``) and the rest, to test where the networks' lead comes from.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the map).
        lines: The fitted shut-off lines.
        out_dir: Figure directory.

    Returns:
        Unrounded errors by model, budget and design (the slides round each value once), counts of designs where one model beats the other, and the split.
    """
    models = ("gp", "networks", "one_network")
    table: dict[str, dict[int, dict]] = {k: {} for k in models}
    spread: dict[str, dict[int, dict]] = {k: {} for k in models}
    near = near_switch(data.grid, lines)
    split: dict[str, dict[str, dict[str, list[float]]]] = {"gp": {}, "networks": {}}
    members_worst = []  # each member's worst pressure error at the largest budget
    big = max(BUDGETS)
    for n in BUDGETS:
        per: dict[str, list] = {k: [] for k in models}
        for by_budget in split.values():
            by_budget[str(n)] = {"near": [], "away": []}
        for d in DESIGNS:
            net = fits.nets[d][n]
            each = net.each(data.grid)
            preds = {"networks": each.mean(axis=0), "gp": fits.gps[d][n].mean(data.grid)}
            per["networks"].append(errors(preds["networks"], data.grid_y))
            per["one_network"].append(errors(each[0], data.grid_y))
            per["gp"].append(errors(preds["gp"], data.grid_y))
            for k, pred in preds.items():
                e = pred[:, 1] - data.grid_y[:, 1]
                split[k][str(n)]["near"].append(float(np.sqrt(np.mean(e[near] ** 2))))
                split[k][str(n)]["away"].append(float(np.sqrt(np.mean(e[~near] ** 2))))
            if n == big:
                members_worst += [errors(p, data.grid_y)["pmin"]["worst"] for p in each]
        for k, rows in per.items():
            table[k][n] = {
                o: {m: float(np.median([r[o][m] for r in rows])) for m in ("rmse", "worst")}
                for o in OUTPUTS
            }
            spread[k][n] = {  # every design's value, design 0 first
                o: {m: [r[o][m] for r in rows] for m in ("rmse", "worst")} for o in OUTPUTS
            }
    curves_figure(table, spread, out_dir)
    pmin_rmse = {k: np.array([spread[k][n]["pmin"]["rmse"] for n in BUDGETS]) for k in spread}
    # the GP's lead over the networks near the switches and away from them, per design
    lead = {
        str(n): {
            where: [
                g / q
                for g, q in zip(
                    split["gp"][str(n)][where], split["networks"][str(n)][where], strict=True
                )
            ]
            for where in ("near", "away")
        }
        for n in BUDGETS
    }
    return {
        "budgets": list(BUDGETS),
        "designs": len(DESIGNS),
        "ensemble": ENSEMBLE,
        "errors": {k: {str(n): row for n, row in t.items()} for k, t in table.items()},
        "per_design": {k: {str(n): row for n, row in t.items()} for k, t in spread.items()},
        # in how many (budget, design) pairs the networks beat the GP on pressure, RMS error
        "nets_better_pmin": int(np.count_nonzero(pmin_rmse["networks"] < pmin_rmse["gp"])),
        "pairs": len(BUDGETS) * len(DESIGNS),
        # leakage: in how many designs the GP's error is the lower, at each budget
        "gp_better_leak": {
            str(n): {
                m: sum(
                    1
                    for g, q in zip(
                        spread["gp"][n]["leak"][m], spread["networks"][n]["leak"][m], strict=True
                    )
                    if g < q
                )
                for m in ("rmse", "worst")
            }
            for n in BUDGETS
        },
        "gp_at_bound": {
            str(n): sum(1 for d in DESIGNS for b in fits.gps[d][n].at_bound if b) for n in BUDGETS
        },
        "ratio_worst_pmin_big": rnd(
            table["gp"][big]["pmin"]["worst"] / table["networks"][big]["pmin"]["worst"], 2
        ),
        # one network instead of five: each member's worst pressure error, every design, 1,024 runs
        "members_worst_pmin_big": {
            "median": float(np.median(members_worst)),
            "range": [float(min(members_worst)), float(max(members_worst))],
        },
        "switches": {
            "near_m": NEAR_LINE_M,
            "near_settings": int(np.count_nonzero(near)),
            "rmse_pmin": split,  # every design, design 0 first
            "gp_over_networks": lead,  # the GP's RMS over the networks', near and away
            "median_ratio": {
                n: {w: float(np.median(v)) for w, v in r.items()} for n, r in lead.items()
            },
        },
    }


def curves_figure(table: dict, spread: dict, out_dir: Path) -> None:
    """Worst and RMS error over the map against the number of runs: median and range of designs.

    Args:
        table: Median errors by model, budget, output and measure.
        spread: Every design's errors, same keys.
        out_dir: Figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    titles = {"leak": "Leakage (m³/h)", "pmin": "Lowest pressure (m)"}
    styles = {
        ("gp", "worst"): {"color": aml_style.PALETTE[0], "ls": "-", "marker": "o"},
        ("networks", "worst"): {"color": aml_style.PALETTE[1], "ls": "-", "marker": "s"},
        ("gp", "rmse"): {"color": aml_style.PALETTE[0], "ls": "--", "marker": "o", "mfc": "white"},
        ("networks", "rmse"): {
            "color": aml_style.PALETTE[1],
            "ls": "--",
            "marker": "s",
            "mfc": "white",
        },
    }
    names = {"gp": "GP", "networks": "five networks"}
    labels = {"worst": "worst", "rmse": "RMS"}
    for ax, o in zip(axes, OUTPUTS, strict=True):
        for (k, m), st in styles.items():
            ys = np.array([table[k][n][o][m] for n in BUDGETS])
            lo = np.array([min(spread[k][n][o][m]) for n in BUDGETS])
            hi = np.array([max(spread[k][n][o][m]) for n in BUDGETS])
            ax.errorbar(BUDGETS, ys, yerr=[ys - lo, hi - ys], ms=7, lw=2.2, capsize=4,
                        label=f"{names[k]}, {labels[m]}", **st)  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(BUDGETS, [str(n) for n in BUDGETS])
        ax.minorticks_off()
        ax.set_xlabel("Runs used for training")
        ax.set_title(titles[o])
    axes[0].set_ylabel("Error over the 1-m map")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=12)
    fig.tight_layout()
    aml_style.save(fig, out_dir, "curves")


def kink_lines(net: Networks, member: int) -> tuple[np.ndarray, np.ndarray]:
    """Each first-layer unit's kink line in settings units: ``s2 = slope * s1 + intercept``.

    A unit with no weight on PRV-2 has a vertical kink line (slope infinite).

    Args:
        net: An ensemble (its settings scaler places the lines in metres).
        member: Which network.

    Returns:
        The slopes and intercepts (m), one per first-layer unit.
    """
    W, b = net.members[member][0], net.members[member][1]  # (2, H), (H,)
    a = W / net.xs.sd[:, None]  # coefficients on s1, s2 in metres
    c = -b + (W * (net.xs.mean / net.xs.sd)[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -a[0] / a[1], c / a[1]


def parallel_offsets(net: Networks, member: int) -> np.ndarray:
    """Offsets of a network's first-layer kinks that run parallel to the shut-off lines.

    A unit whose kink has slope 0.8 to 1.25 is placed by its offset s2 - s1 at the box's centre;
    only offsets inside the box (s2 - s1 spans -35 to 35 m there) are kept.

    Args:
        net: An ensemble.
        member: Which network.

    Returns:
        The offsets, m.
    """
    centre = 0.5 * (LO + HI)
    slope, icpt = kink_lines(net, member)
    off = slope * centre + icpt - centre
    return off[(slope > 0.8) & (slope < 1.25) & (np.abs(off) < HI - LO)]


def nearest_to(offsets: np.ndarray, line: dict) -> float | None:
    """Distance (m) from a shut-off line's offset to the nearest kink; None if there is none."""
    return float(np.abs(offsets - line["offset"]).min()) if len(offsets) else None


def kinks(net: Networks, lines: dict, out_dir: Path) -> dict[str, object]:
    """Do the first-layer kinks of each network sit on the valves' shut-off lines?

    For each line, the nearest parallel kink in each network after training, and the same at
    the start (each member's seed, scaled as trained): the baseline the trained kinks beat.

    Args:
        net: The five networks on 1,024 runs of design 0 (members are seeds 0-4).
        lines: The fitted shut-off lines.
        out_dir: Figure directory.

    Returns:
        Counts and offsets of the parallel kinks, and the nearest kink to each line, trained
        and at the start (None for a start with no parallel kink).

    Raises:
        ValueError: If a trained network has no kink parallel to the shut-off lines.
    """
    offsets = [parallel_offsets(net, i) for i in range(len(net.members))]
    if not all(len(o) for o in offsets):
        raise ValueError(f"a trained network has no parallel kink: {[len(o) for o in offsets]}")
    nearest = {k: [nearest_to(off, line) for off in offsets] for k, line in lines.items()}
    sizes = (2, *HIDDEN, 2)
    start = Networks([init_params(i, sizes) for i in range(len(net.members))], net.xs, net.ys, [])
    at_start = [parallel_offsets(start, i) for i in range(len(net.members))]
    kinks_figure(offsets, lines, out_dir)
    return {
        "units": HIDDEN[0],
        "parallel": [len(o) for o in offsets],
        "parallel_start": [len(o) for o in at_start],
        "start_nearest": {
            k: [None if (x := nearest_to(off, line)) is None else rnd(x, 2) for off in at_start]
            for k, line in lines.items()
        },
        "offset_range": [
            rnd(min(o.min() for o in offsets), 2),
            rnd(max(o.max() for o in offsets), 2),
        ],
        "nearest_max": {k: rnd(max(v), 2) for k, v in nearest.items()},
        "nearest": {k: [rnd(x, 2) for x in v] for k, v in nearest.items()},
    }


def kinks_figure(offsets: list[np.ndarray], lines: dict, out_dir: Path) -> None:
    """Strip plot: each network's parallel kinks by offset, with the simulator's shut-off lines.

    Args:
        offsets: Each network's parallel kink offsets, m.
        lines: The fitted shut-off lines.
        out_dir: Figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, off in enumerate(offsets):
        ax.plot(off, np.full_like(off, i + 1), "o", color=aml_style.INK, ms=8, mfc="white", mew=1.6)
    styles = {"prv2": (aml_style.PALETTE[0], "--"), "prv1": (aml_style.PALETTE[1], "-.")}
    names = {"prv2": "PRV-2 shuts", "prv1": "PRV-1 shuts"}
    for k, (color, ls) in styles.items():
        ax.axvline(lines[k]["offset"], color=color, ls=ls, lw=2.6, label=names[k])
    ax.set_yticks(range(1, len(offsets) + 1))
    ax.set_ylim(0.4, len(offsets) + 0.6)
    ax.set_ylabel("Network")
    ax.set_xlim(-(HI - LO), HI - LO)
    ax.set_xlabel("Kink line: PRV-2 minus PRV-1 (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=12)
    aml_style.save(fig, out_dir, "kinks")


def kinks_crossing(net: Networks, member: int = 0) -> np.ndarray:
    """Which first-layer units have a kink crossing the box: on at some corners, off at others.

    A unit's input is linear in the settings, so it changes sign inside the box exactly when it
    has different signs at two of the box's corners.

    Returns:
        A boolean mask over the units.
    """
    corners = np.array([(LO, LO), (LO, HI), (HI, LO), (HI, HI)])
    W, b = net.members[member][0], net.members[member][1]
    a = net.xs(corners) @ W + b  # (4, units)
    return (a > 0).any(axis=0) & (a < 0).any(axis=0)


def kinks_in_box(net: Networks, member: int = 0) -> int:
    """How many first-layer kinks of network ``member`` cross the box (``kinks_crossing``)."""
    return int(kinks_crossing(net, member).sum())


def breaks(data: Data, fits: Fits) -> dict[str, object]:
    """Bad starts against the lecture's: zeros, equal weights, zero biases, unscaled units.

    Every start trains five networks (seeds 0-4; "uniform" is the 256-run ensemble), reported
    as the median and range of their lowest-pressure errors over the map. "raw" leaves only the
    settings in metres (outputs scaled); "raw_out" leaves only the outputs in their own units.
    The mechanisms (what moved, how alike the units stay, which kinks cross the box) are shown
    on seed 0.

    Args:
        data: Every run.
        fits: Every fitted surrogate.

    Returns:
        Each start's errors (median and range over seeds), and the mechanisms' checks.

    Raises:
        ValueError: If a start has the wrong number of networks, or the all-zero start
            predicts anything but the training mean.
    """
    S, Y = data.design[0, :BREAK_BUDGET], data.design_y[0, :BREAK_BUDGET]
    nets = {"uniform": fits.nets[0][BREAK_BUDGET]} | fits.starts
    out: dict[str, object] = {}
    for name, net in nets.items():
        errs = [errors(pred, data.grid_y) for pred in net.each(data.grid)]
        if len(errs) != ENSEMBLE:
            raise ValueError(f"{name}: {len(errs)} networks, not {ENSEMBLE}")
        row: dict[str, object] = {}
        for label, o, m in (("pmin", "pmin", "rmse"), ("worst_pmin", "pmin", "worst"),
                            ("leak", "leak", "rmse")):  # fmt: skip
            v = [e[o][m] for e in errs]
            row[label] = rnd(np.median(v), 2)
            row[f"{label}_range"] = [rnd(min(v), 2), rnd(max(v), 2)]
        row["kinks_in_box"] = kinks_in_box(net)
        out[name] = row
    # all-zero start: nothing but the output bias moves, so every prediction is the training mean
    p = fits.starts["zero"].members[0]
    out["zero"]["moved"] = int(sum(np.any(q != 0) for q in p[:-1]))
    for pred in fits.starts["zero"].each(data.grid):
        if not np.allclose(pred, Y.mean(axis=0), atol=1e-9):
            raise ValueError("the zero start did not predict the training mean")
    # equal start: each layer's units start identical and get identical gradients; only the
    # rounding of the matrix products can tell them apart
    q = fits.starts["const"].members[0]
    out["const"]["unit_spread"] = [float(np.abs(q[i] - q[i][:, :1]).max()) for i in (0, 2)]
    # unscaled: where the first layer's kinks are at the start (scaled vs metres)
    raw = fits.starts["raw"]
    start = Networks([init_params(SEED, (2, *HIDDEN, 2))], raw.xs, raw.ys, [])
    out["raw"]["kinks_in_box_start"] = kinks_in_box(start)
    out["raw"]["same_kinks"] = bool(np.array_equal(kinks_crossing(start), kinks_crossing(raw)))
    out["raw_out"]["kinks_in_box_start"] = kinks_in_box(
        Networks([init_params(SEED, (2, *HIDDEN, 2))], fits.starts["raw_out"].xs, raw.ys, [])
    )
    good = fits.nets[0][BREAK_BUDGET]
    start = Networks([init_params(SEED, (2, *HIDDEN, 2))], good.xs, good.ys, [])
    out["uniform"]["kinks_in_box_start"] = kinks_in_box(start)
    one = np.c_[np.ones(len(S)), S]
    plane, *_ = np.linalg.lstsq(one, Y[:, 1], rcond=None)
    grid_pred = np.c_[np.ones(len(data.grid)), data.grid] @ plane
    out["plane_rmse_pmin"] = rnd(rmse(grid_pred, data.grid_y[:, 1]), 2)
    out["budget"] = BREAK_BUDGET
    out["seeds"] = ENSEMBLE
    return out


def beyond(fits: Fits, data: Data, out_dir: Path) -> dict[str, object]:
    """The slice at PRV-1 = 36 m, past the box to 90 m: five networks and the GP, 1,024 runs.

    Design 0's ensemble draws the figure; the other two designs' ensembles are checked at 90 m.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the slice and the map).
        out_dir: Figure directory.

    Returns:
        The networks' spread inside the box (on the slice and over the map), their predictions
        and the GP's at 75 and 90 m against the simulator, and where the simulator flattens.
    """
    S = np.c_[np.full_like(data.slice_s2, SLICE_S1), data.slice_s2]
    n = max(BUDGETS)
    nets = fits.nets[0][n].each(S)  # (members, points, outputs)
    mu, sd = fits.gps[0][n].mean_sd(S)
    beyond_figure(data.slice_s2, data.slice_y, nets, gp=(mu, sd), out_dir=out_dir)
    inside = data.slice_s2 <= HI
    open_from = float(data.slice_s2[data.slice_y[:, 0] >= data.slice_y[:, 0].max() - 1e-6][0])

    def at(v: float) -> dict[str, dict[str, object]]:
        i = int(np.flatnonzero(data.slice_s2 == v)[0])
        return {
            o: {
                "truth": rnd(data.slice_y[i, j], 2),
                "nets": [rnd(x, 2) for x in np.sort(nets[:, i, j])],
                "gp_mean": rnd(mu[i, j], 2),
                "gp_sd": rnd(sd[i, j], 2),
                "gp_covers": bool(abs(mu[i, j] - data.slice_y[i, j]) <= 2 * sd[i, j]),
            }
            for j, o in enumerate(OUTPUTS)
        }

    spread = nets.max(axis=0) - nets.min(axis=0)
    grid_each = fits.nets[0][n].each(data.grid)
    spread_map = grid_each.max(axis=0) - grid_each.min(axis=0)
    cap = data.slice_y[:, 1].max()
    i90 = int(np.flatnonzero(data.slice_s2 == EXTRAP_HI)[0])
    by_design = {}
    for d in DESIGNS:
        e = fits.nets[d][n].each(S[i90 : i90 + 1])[:, 0]  # (members, outputs)
        by_design[str(d)] = {
            o: {
                "nets": [rnd(e[:, j].min(), 2), rnd(e[:, j].max(), 2)],
                "spread": rnd(np.ptp(e[:, j]), 2),
                "min_error": rnd(np.abs(e[:, j] - data.slice_y[i90, j]).min(), 2),
            }
            for j, o in enumerate(OUTPUTS)
        }
    return {
        "runs": n,
        "spread_inside": {o: rnd(spread[inside, j].max(), 2) for j, o in enumerate(OUTPUTS)},
        "spread_map": {o: rnd(spread_map[:, j].max(), 2) for j, o in enumerate(OUTPUTS)},
        "at75": at(75.0),
        "at90": at(EXTRAP_HI),
        "at90_by_design": by_design,
        "cap": rnd(cap, 2),
        "cap_from": rnd(float(data.slice_s2[data.slice_y[:, 1] == cap][0]), 2),
        "leak_flat_from": rnd(open_from, 2),
    }


def beyond_figure(
    s2: np.ndarray,
    truth: np.ndarray,
    nets: np.ndarray,
    *,
    gp: tuple[np.ndarray, np.ndarray],
    out_dir: Path,
) -> None:
    """Two panels on the slice past the box: leakage and lowest pressure.

    Args:
        s2: PRV-2 settings of the slice, m.
        truth: The simulator's outputs on the slice.
        nets: Each network's predictions, shape ``(members, points, outputs)``.
        gp: The GP's means and standard deviations on the slice.
        out_dir: Figure directory.
    """
    mu, sd = gp
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    titles = ("Leakage (m³/h)", "Lowest pressure (m)")
    for j, ax in enumerate(axes):
        ax.axvspan(HI, s2.max(), color="#F0F0F0", zorder=0)
        ax.fill_between(
            s2,
            mu[:, j] - 2 * sd[:, j],
            mu[:, j] + 2 * sd[:, j],
            color=aml_style.PALETTE[0],
            alpha=0.2,
            lw=0,
            label="GP, ±2 sd",
        )
        for i, row in enumerate(nets[:, :, j]):
            ax.plot(
                s2,
                row,
                color=aml_style.PALETTE[1],
                ls="-",
                lw=1.4,
                label="five networks" if i == 0 else None,
            )
        ax.plot(s2, truth[:, j], color=aml_style.INK, ls="--", lw=2.4, label="simulator")
        ax.set_title(titles[j])
        ax.set_xlabel(f"PRV-2 setting (m); PRV-1 at {SLICE_S1:.0f} m")
        ax.text(
            HI + 1.5,
            0.05,
            "beyond\nthe runs",
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            color=aml_style.MUTED,
        )
    axes[0].legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    aml_style.save(fig, out_dir, "beyond")


Pick = tuple[float, float]  # PRV-1, PRV-2 settings, m
PickKey = tuple[str, int, int]  # (surrogate, design, budget)


def choose(pred: np.ndarray, cand: np.ndarray, margin: float = 0.0) -> Pick:
    """Least predicted leakage among candidates predicted to keep 24.6 m, less the margin.

    Args:
        pred: Predicted leakage and lowest pressure, shape ``(len(cand), 2)``.
        cand: Candidate settings, shape ``(n, 2)``.
        margin: Metres subtracted from the predicted lowest pressure.

    Returns:
        The chosen PRV-1, PRV-2 settings.

    Raises:
        ValueError: If no candidate is predicted to keep 24.6 m.
    """
    ok = pred[:, 1] - margin >= P_SERVICE
    if not ok.any():
        raise ValueError("no candidate is predicted feasible")
    i = np.flatnonzero(ok)[np.argmin(pred[ok, 0])]
    return float(cand[i, 0]), float(cand[i, 1])


def conformal_margin(residual: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal one-sided margin: the ceil((n+1)(1-alpha))-th smallest residual.

    Args:
        residual: Held-out overpredictions (prediction minus truth), m.
        alpha: Miscoverage level.

    Returns:
        The margin, m.

    Raises:
        ValueError: If there are too few held-out runs for ``alpha``.
    """
    k = int(np.ceil((len(residual) + 1) * (1 - alpha)))
    if k > len(residual):
        raise ValueError(f"{len(residual)} held-out runs are too few for alpha {alpha}")
    return float(np.sort(residual)[k - 1])


def window_optimum(data: Data) -> tuple[np.ndarray, float]:
    """The least-leakage feasible setting on the fine lattice around the optimum, and its leak.

    Args:
        data: Every run (the window).

    Returns:
        The best settings (m) and their leakage (m^3/h).

    Raises:
        ValueError: If the optimum sits on the window's edge (the window would be too small).
    """
    feas = data.window_y[:, 1] >= P_SERVICE
    i = np.flatnonzero(feas)[np.argmin(data.window_y[feas, 0])]
    best = data.window[i]
    (a1, b1), (a2, b2) = OPT_WINDOW
    if not (a1 < best[0] < b1 and a2 < best[1] < b2):
        raise ValueError(f"optimum {best} on the window's edge")
    return best, float(data.window_y[i, 0])


def all_picks(
    fits: Fits, data: Data, cand: np.ndarray
) -> tuple[dict[PickKey, Pick], dict[tuple[int, int], float]]:
    """Every surrogate's pick: networks and GPs without a margin, networks with one, the GP's band.

    Design d's networks are calibrated on the first ``CAL_RUNS`` runs of design (d + 1) % 3,
    which they never saw. A margin is never negative: a network that underpredicts keeps 0. The
    GP's band pick (design 0, 1,024 runs) keeps 24.6 m at its mean minus 2 sd.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the calibration designs).
        cand: Candidate settings.

    Returns:
        The picks, keyed (surrogate, design, budget), and the margins (m), keyed (d, n).
    """
    picks: dict[PickKey, Pick] = {}
    margins: dict[tuple[int, int], float] = {}
    for d in DESIGNS:
        c = (d + 1) % len(DESIGNS)
        S_cal, Y_cal = data.design[c, :CAL_RUNS], data.design_y[c, :CAL_RUNS]
        for n in BUDGETS:
            net = fits.nets[d][n]
            pred = net(cand)
            picks["networks", d, n] = choose(pred, cand)
            picks["gp", d, n] = choose(fits.gps[d][n].mean(cand), cand)
            margins[d, n] = max(0.0, conformal_margin(net(S_cal)[:, 1] - Y_cal[:, 1]))
            picks["networks_margin", d, n] = choose(pred, cand, margins[d, n])
    big = max(BUDGETS)
    mu, sd = fits.gps[0][big].mean_sd(cand)
    picks["gp_2sd", 0, big] = choose(np.c_[mu[:, 0], mu[:, 1] - 2 * sd[:, 1]], cand)
    return picks, margins


def _counts(checks: list[dict[str, object]]) -> dict[str, object]:
    """Feasible picks, the largest shortfall, and the largest regret among some checked picks."""
    return {
        "feasible": sum(1 for c in checks if c["feasible"]),
        "short_max": max(c["short_m"] for c in checks),
        "regret_max": max(c["regret"] for c in checks),
    }


@dataclass(frozen=True)
class Decision:
    """The decision's numbers for the slides, and what the margin figure draws.

    Attributes:
        numbers: The checked picks and the coverage of the margin and of the GP's band.
        picks: Every pick, keyed (surrogate, design, budget).
        margin: Design 0's margin at 1,024 runs, m.
    """

    numbers: dict[str, object]
    picks: dict[PickKey, Pick]
    margin: float


def decide(fits: Fits, data: Data) -> Decision:
    """Each surrogate picks settings on a 0.2-m lattice; simulator runs check each pick.

    Feasibility is checked at EPANET accuracy ``ACCURATE``, 100 times tighter than the file's,
    so that a shortfall of millimetres is not the solver's; regret uses the lecture's runs.

    Args:
        fits: Every fitted surrogate.
        data: Every run.

    Returns:
        The decision: picks checked by the simulator, shortfalls by surrogate, the margins, and
        the coverage of the margin and of the GP's band over the map, two- and one-sided.
    """
    cand = lattice(LO, HI, LO, HI)
    best, best_leak = window_optimum(data)
    picks, margins = all_picks(fits, data, cand)
    big = max(BUDGETS)
    pairs = sorted(set(picks.values()))
    truth = dict(zip(pairs, outputs(run_many(pairs, WORKERS)), strict=True))
    exact = dict(zip(pairs, outputs(run_many(pairs, WORKERS, accuracy=ACCURATE)), strict=True))

    def check(k: PickKey) -> dict[str, object]:
        s = picks[k]
        pmin = exact[s][1]
        return {
            "s1": rnd(s[0], 2),
            "s2": rnd(s[1], 2),
            "leak": rnd(truth[s][0], 2),
            "pmin": rnd(pmin, 2),
            "short_m": rnd(max(0.0, P_SERVICE - pmin), 2),
            "regret": rnd(truth[s][0] - best_leak, 2),
            "feasible": bool(pmin >= P_SERVICE),
        }

    keys = {k: [(k, d, n) for d in DESIGNS for n in BUDGETS] for k in ("networks", "gp")}
    keys["networks_margin"] = [("networks_margin", d, n) for d in DESIGNS for n in BUDGETS]
    by_budget = {
        k: {str(n): _counts([check((k, d, n)) for d in DESIGNS]) for n in BUDGETS} for k in keys
    }
    plain = keys["networks"] + keys["gp"]
    short = {p: P_SERVICE - exact[picks[p]][1] for p in plain}
    short_default = [P_SERVICE - truth[picks[p]][1] for p in plain]
    with_margin = [check(k) for k in keys["networks_margin"]]
    numbers = {
        "optimum": {"s1": rnd(best[0], 2), "s2": rnd(best[1], 2), "leak": rnd(best_leak, 2)},
        "candidates": len(cand),
        "by_budget": by_budget,
        "picks": len(plain),
        "accuracy": ACCURATE,
        "short": sum(1 for x in short.values() if x > 0),
        # like for like: each surrogate's own 12 picks (the margin applies to the networks')
        "short_by_model": {
            m: sum(1 for p, x in short.items() if p[0] == m and x > 0) for m in ("networks", "gp")
        },
        "short_default": sum(1 for x in short_default if x > 0),
        "short_max": rnd(max(short.values()), 2),
        "short_mm": sorted(1000 * x for x in short.values() if x > 0),  # unrounded
        "pmin_shift_mm": 1000 * max(abs(exact[q][1] - truth[q][1]) for q in pairs),  # unrounded
        "design0": {
            f"{k}_{n}": check((k, 0, n)) for k in ("networks", "gp") for n in (min(BUDGETS), big)
        },
        "margin": margin_numbers(fits, data, margins, with_margin)
        | {"networks": check(("networks_margin", 0, big))},
        "gp_band": gp_band(fits, data) | {"pick": check(("gp_2sd", 0, big))},
    }
    return Decision(numbers, picks, margins[0, big])


def margin_numbers(
    fits: Fits, data: Data, margins: dict[tuple[int, int], float], checked: list[dict]
) -> dict[str, object]:
    """The networks' margins: design 0's coverage over the map, and every design's picks.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the map).
        margins: Each design's and budget's margin, m.
        checked: The checked picks with a margin.

    Returns:
        Design 0's margin at 1,024 runs and the share of the map it covers (one-sided, the
        event the decision needs), and the 12 margined picks' shortfalls and regret.
    """
    big = max(BUDGETS)
    resid = fits.nets[0][big](data.grid)[:, 1] - data.grid_y[:, 1]
    return {
        "m": rnd(margins[0, big], 2),
        "cal_runs": CAL_RUNS,
        "alpha": ALPHA,
        "covered_on_map": rnd(np.mean(resid <= margins[0, big]), 3),
        "all": {
            "short": sum(1 for c in checked if not c["feasible"]),
            "of": len(checked),
            "regret_max": max(c["regret"] for c in checked),
            "range_m": [rnd(min(margins.values()), 2), rnd(max(margins.values()), 2)],
        },
        "by_design": {f"{d}_{n}": rnd(m, 2) for (d, n), m in margins.items()},
    }


def gp_band(fits: Fits, data: Data) -> dict[str, object]:
    """Coverage of the GP's 2-sd band of the lowest pressure over the map (design 0, 1,024 runs).

    Two-sided, ``|mu - y| <= 2 sd``; one-sided, ``mu - 2 sd <= y``, the event its band pick
    relies on (nominal 97.7% for a Gaussian, against the margin's 95%).

    Args:
        fits: Every fitted surrogate.
        data: Every run (the map).

    Returns:
        Both coverages and the one-sided nominal level.
    """
    mu, sd = fits.gps[0][max(BUDGETS)].mean_sd(data.grid)
    y = data.grid_y[:, 1]
    return {
        "covered_on_map": rnd(np.mean(np.abs(mu[:, 1] - y) <= 2 * sd[:, 1]), 3),
        "covered_one_sided": rnd(np.mean(mu[:, 1] - 2 * sd[:, 1] <= y), 3),
        "nominal_one_sided": rnd(norm.cdf(2.0), 3),
    }


def trace(fits: Fits, data: Data) -> dict[str, object]:
    """One run traced through network 0 of design 0 with the most runs: the diagram's inset.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the map holds the traced setting).

    Returns:
        The standardized settings, the units on in each layer, the prediction, the ensemble's,
        and the simulator's.

    Raises:
        ValueError: If the hand-written forward pass disagrees with ``Networks.each``.
    """
    net = fits.nets[0][max(BUDGETS)]
    p = net.members[0]
    z = net.xs(np.array([TRACE]))
    h1 = np.maximum(0.0, z @ p[0] + p[1])
    h2 = np.maximum(0.0, h1 @ p[2] + p[3])
    y = net.ys.inverse(h2 @ p[4] + p[5])[0]
    if not np.allclose(y, net.each(np.array([TRACE]))[0, 0]):
        raise ValueError("the traced forward pass differs from the network's")
    i = int(np.flatnonzero((data.grid == TRACE).all(axis=1))[0])
    return {
        "s": list(TRACE),
        "z": [rnd(v, 2) for v in z[0]],
        "on": [int((h1 > 0).sum()), int((h2 > 0).sum())],
        "pred": [rnd(v, 2) for v in y],
        "ensemble": [rnd(v, 2) for v in net(np.array([TRACE]))[0]],
        "truth": [rnd(v, 2) for v in data.grid_y[i]],
        "weights": int(sum(q.size for q in p)),
        "runs": max(BUDGETS),
    }


# ---------------------------------------------------------------------------------------------
# Visuals that carry the slides: builds, concept pictures, the title art


def title_art_figure(out_dir: Path) -> None:
    """The title art: a stepped portal holding a glowing network, a dark sun, and a blade.

    The attempt's two-ramp curve is the horizon; the crimson blade is the backward pass.
    """
    c = title_art.Canvas(seed=14)
    c.ground(("#2a0710", "#1a0816", "#0d0a1a", "#07060f"), grain=0.08)
    c.glaze(c.polygon([(0, 0), (1, 0), (1, 0.42), (0, 0.30)], soft=30), "#5c1018", 0.55)
    rim = c.disc((0.70, 1.08), 0.17, soft=4) - c.disc((0.70, 1.08), 0.145, soft=4)
    c.glow(np.clip(rim, 0, 1), "#f0a03c", 0.55, spread=40)
    c.glaze(c.disc((0.70, 1.08), 0.145, soft=3), "#2b0b0e", 0.92)
    c.glaze(c.polygon(title_art.stepped(0.08, 0.10, 0.56, 1.02, 4), soft=3), "#6e1a1c", 0.88)
    c.glaze(c.polygon(title_art.stepped(0.15, 0.16, 0.42, 0.80, 3), soft=3), "#a8401e", 0.70)
    door = [(0.24, 0.20), (0.48, 0.20), (0.48, 0.74), (0.24, 0.74)]
    c.glaze(c.polygon(door, soft=3), "#140f22", 0.92)
    cols = [0.28, 0.33, 0.39, 0.44]
    ys = [np.linspace(0.32, 0.62, n) for n in (2, 5, 5, 2)]
    edges = [
        [(xa, y1), (xb, y2)]
        for xa, xb, ya, yb in zip(cols[:-1], cols[1:], ys[:-1], ys[1:], strict=True)
        for y1 in ya
        for y2 in yb
    ]
    c.glow(c.lines(edges, 2.0, soft=0.8), "#e9c27a", 0.45, spread=6)
    nodes = sum(c.disc((x, y), 0.008, soft=1.5) for x, yy in zip(cols, ys, strict=True) for y in yy)
    c.glow(np.clip(nodes, 0, 1), "#ffd98a", 0.9, spread=10)
    blade = [(0.62, 0.95), (0.66, 0.97), (0.47, 0.50), (0.45, 0.52)]
    c.glaze(c.polygon(blade, soft=2), "#c2361f", 0.85)
    s2 = np.linspace(LO, HI, 300)
    ramp = HAND["a"] + HAND["b"] * np.maximum(0, s2 - HAND["c1"])
    ramp += HAND["d"] * np.maximum(0, s2 - HAND["c2"])
    horizon = [
        [
            (0.03 + 0.94 * (x - LO) / (HI - LO), 0.035 + (y - 14.5) / 50)
            for x, y in zip(s2, ramp, strict=True)
        ]
    ]
    c.glow(c.lines(horizon, 4, soft=1.2), "#f3b45c", 0.8, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean portal in reds and ochres holds a "
        "glowing network of four layers; a dark sun with a burning rim hangs to the right, a "
        "crimson blade runs from it back into the network, and a golden line with two bends, "
        "the lecture's two ramps, crosses the bottom as a horizon.",
    )


def ltown_map(town: LTown, out_dir: Path, name: str, marked: Sequence[str] = ()) -> None:
    """L-Town's pipes by area, the two valves, n50, and optionally some customers marked.

    Args:
        town: The calibrated model.
        out_dir: Figure directory.
        name: File name of the figure, without extension.
        marked: Junctions to mark.
    """
    wn = town.wn
    styles = {"A": (aml_style.MUTED, "-"), "B": (aml_style.PALETTE[2], "--"), "C": ("#9a9a9a", ":")}
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for a, b, zone in town.segments():
        color, ls = styles.get(zone, (aml_style.MUTED, "-"))
        ax.plot(*zip(a, b, strict=True), color=color, ls=ls, lw=0.9)
    if marked:
        xy = np.array([wn.get_node(n).coordinates for n in marked])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "^",
            color=aml_style.PALETTE[1],
            ms=7,
            zorder=5,
            label=f"{len(marked)} customers within {NEAR_M:.0f} m of the limit",
        )
    for valve, mk in (("PRV-1", "s"), ("PRV-2", "D")):
        x, y = town.outlet_xy(valve)
        ax.plot(x, y, mk, ms=11, color=aml_style.INK, mfc="white", mew=2, zorder=6)
        ax.annotate(valve, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=12,
                    path_effects=aml_style.HALO)  # fmt: skip
    x, y = wn.get_node("n50").coordinates
    ax.plot(x, y, "o", ms=12, color=aml_style.PALETTE[0], zorder=7)
    # Up and left of the node: the marked customers sit beside it on the right.
    ax.annotate("n50", (x, y), xytext=(-12, 6), textcoords="offset points", fontsize=12,
                ha="right", path_effects=aml_style.HALO)  # fmt: skip
    for zone, (color, ls) in styles.items():
        ax.plot([], [], color=color, ls=ls, lw=2, label=f"Area {zone}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=12)
    aml_style.save(fig, out_dir, name)


def ramp_build(data: Data, fit: RampFit, out_dir: Path) -> None:
    """Three frames on the attempt's slice: the six runs, one ramp, two ramps and the simulator.

    Args:
        data: Every run (the slice).
        fit: The attempt's best two ramps.
        out_dir: Figure directory.
    """
    c1, c2, (a, b, d) = fit.c1, fit.c2, fit.coef
    inside = data.slice_s2 <= HI
    s2, p = data.slice_s2[inside], data.slice_y[inside, 1]
    six = np.isin(s2, ATTEMPT_S2)
    one = a + b * np.maximum(0, s2 - c1)
    two = one + d * np.maximum(0, s2 - c2)
    for k in range(3):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        if k == 2:
            ax.plot(s2, p, color=aml_style.MUTED, lw=1.4, label="simulator, every metre")
        ax.plot(s2[six], p[six], "o", color=aml_style.INK, ms=10, zorder=5, label="the six runs")
        if k >= 1:
            style = (
                {"color": aml_style.PALETTE[1], "ls": "--", "lw": 2.8}
                if k == 1
                else {"color": aml_style.MUTED, "ls": ":", "lw": 2.0}
            )
            ax.plot(s2, one, label=rf"one ramp: $a+b\;\max(0,\,s_2-{c1:.0f})$", **style)
            ax.axvline(c1, color=aml_style.PALETTE[0], lw=1.2, ls=":")
            ax.text(  # left of the kink, above the flat part: the ramp climbs on the right
                c1 - 0.6,
                17.0,
                f"kink at {c1:.0f} m:\nPRV-2 opens",
                fontsize=13,
                color=aml_style.PALETTE[0],
                va="bottom",
                ha="right",
            )
        if k == 2:
            ax.plot(s2, two, color=aml_style.PALETTE[1], lw=2.8,
                    label=rf"$+\;d\;\max(0,\,s_2-{c2:.0f})$")  # fmt: skip
            ax.axvline(c2, color=aml_style.PALETTE[2], lw=1.2, ls=":")
            ax.text(
                c2 + 0.6,
                12.6,
                f"kink at {c2:.0f} m:\nthe slope drops",
                fontsize=13,
                color=aml_style.PALETTE[2],
                va="bottom",
            )
        ax.set(
            xlabel=f"PRV-2 setting (m); PRV-1 at {SLICE_S1:.0f} m",
            ylabel="Lowest pressure (m)",
            xlim=(LO - 1, HI + 1),
            ylim=(12, 34),
        )
        # Opaque: the kink line at 33 m runs up behind it.
        ax.legend(loc="upper left", fontsize=12, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=1)  # fmt: skip
        aml_style.save(fig, out_dir, f"ramp{k}")


def relu_figure(out_dir: Path) -> None:
    """A ramp unit and its derivative: it passes the gradient where on, blocks it where off.

    Args:
        out_dir: Figure directory.
    """
    x = np.linspace(LO, HI, 400)
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    axes[0].plot(x, np.maximum(0, x - HAND["c1"]), color=aml_style.PALETTE[1], lw=2.8)
    c1 = HAND["c1"]
    axes[0].set(title=rf"$h=\max(0,\,s_2-{c1:.0f})$", xlabel="PRV-2 setting (m)", ylabel="h")
    axes[1].step(
        x, (x > HAND["c1"]).astype(float), color=aml_style.PALETTE[0], lw=2.8, where="post"
    )
    axes[1].set(
        title=rf"$\partial h/\partial s_2=[\,s_2>{c1:.0f}\,]$",
        xlabel="PRV-2 setting (m)",
        ylim=(-0.15, 1.3),
    )
    # The note sits right of the step, with an arrow to the flat part it describes: on the
    # left, between the axis and the step, it did not fit.
    axes[1].annotate("off: blocks\nthe gradient", (29.0, 0.0), xytext=(38.0, 0.3), fontsize=13,
                     color=aml_style.MUTED, arrowprops={"arrowstyle": "->",
                                                        "color": aml_style.MUTED})  # fmt: skip
    axes[1].text(46, 1.05, "on: passes it", fontsize=13, color=aml_style.MUTED)
    for ax in axes:
        ax.axvline(HAND["c1"], color=aml_style.MUTED, lw=1, ls=":")
    fig.tight_layout()
    aml_style.save(fig, out_dir, "relu")


def unit_maps(net: Networks, lines: dict, data: Data, out_dir: Path) -> dict[str, object]:
    """3D: one first-layer ramp folds the plane; the networks' surface; the simulator's.

    Args:
        net: The five networks on 1,024 runs of design 0.
        lines: The fitted shut-off lines.
        data: Every run (the map).
        out_dir: Figure directory.

    Returns:
        The shown unit's offset (the one nearest PRV-2's line) and network 1's kinks in the box.
    """
    n = int(round((HI - LO) / GRID_STEP)) + 1
    g = np.arange(LO, HI + 1e-9, GRID_STEP)
    G1, G2 = np.meshgrid(g, g)
    p = net.members[0]
    a1 = net.xs(data.grid) @ p[0] + p[1]  # (cells, units): first-layer inputs
    slope, icpt = kink_lines(net, 0)
    centre = 0.5 * (LO + HI)
    off = slope * centre + icpt - centre
    par = (slope > 0.8) & (slope < 1.25)
    unit = int(np.flatnonzero(par)[np.argmin(np.abs(off[par] - lines["prv2"]["offset"]))])
    fig = plt.figure(figsize=aml_style.FULL)
    zlim = (5.0, 35.0)
    surfaces = (
        (np.maximum(0, a1[:, unit]).reshape(n, n).T, "one ramp unit"),
        (np.clip(net(data.grid)[:, 1].reshape(n, n).T, *zlim), "five networks"),
        (data.grid_y[:, 1].reshape(n, n).T, "simulator"),
    )
    for k, (z, title) in enumerate(surfaces):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if k == 0:
            aml_style.surface3d(ax, G1, G2, z, cmap=CMAP)
            s1 = np.linspace(LO, HI, 50)
            s2 = slope[unit] * s1 + icpt[unit]
            keep = (s2 >= LO) & (s2 <= HI)
            ax.computed_zorder = False  # the crease is drawn over the surface
            ax.plot(s1[keep], s2[keep], 0 * s1[keep], color=aml_style.PALETTE[1], lw=2.8, zorder=5)
            ax.set_zticks([])
        else:
            aml_style.surface3d(ax, G1, G2, z, cmap=CMAP, vmin=zlim[0], vmax=zlim[1])
            ax.set(zlim=zlim, zticks=[10, 30])
        ax.set(xlim=(LO, HI), ylim=(LO, HI), xticks=[30, 60], yticks=[30, 60])
        ax.set_xlabel("PRV-1", labelpad=-8, fontsize=11)
        ax.set_ylabel("PRV-2", labelpad=-8, fontsize=11)
        aml_style.style3d(ax, elev=26, azim=-130)
        ax.set_box_aspect((1, 1, 0.75), zoom=1.05)
        ax.set_title(title, pad=-2)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.92, wspace=0.12)
    aml_style.save(fig, out_dir, "unit_maps")
    return {"unit_offset": rnd(off[unit], 2), "kinks_in_box": kinks_in_box(net, 0)}


def training_figure(fits: Fits, out_dir: Path) -> dict[str, object]:
    """Training and validation error of the lowest pressure against the step, two schedules.

    Args:
        fits: Every fitted surrogate (the curves).
        out_dir: Figure directory.

    Returns:
        Each schedule's last training and validation RMS error, m.
    """
    sd = fits.nets[0][max(BUDGETS)].ys.sd[1]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    out = {}
    for steps, ls in ((STEPS, "-"), (SHORT_STEPS, "--")):
        h = np.array(fits.curves[steps])
        ax.plot(
            h[:, 0],
            h[:, 1] * sd,
            color=aml_style.PALETTE[0],
            ls=ls,
            lw=2.2,
            label=f"training runs, {steps:,} steps",
        )
        ax.plot(
            h[:, 0],
            h[:, 2] * sd,
            color=aml_style.PALETTE[1],
            ls=ls,
            lw=2.2,
            label=f"validation runs, {steps:,} steps",
        )
        out[str(steps)] = {"train": float(h[-1, 1] * sd), "val": float(h[-1, 2] * sd)}
    ax.set(xlabel="Adam step", ylabel="RMS error, lowest pressure (m)", yscale="log")
    ax.legend(fontsize=11, loc="upper right")
    aml_style.save(fig, out_dir, "training")
    return out


def barrier_figure(net: Networks, data: Data, out_dir: Path) -> dict[str, object]:
    """Error along the straight line between two trained networks: two valleys and a ridge.

    Args:
        net: The five networks on 1,024 runs of design 0.
        data: Every run (the training runs).
        out_dir: Figure directory.

    Returns:
        The RMS error (m) at both ends, at the midpoint, and the line's highest point.
    """
    n = max(BUDGETS)
    Z, Y = net.xs(data.design[0, :n]), data.design_y[0, :n, 1]
    t = np.linspace(-0.25, 1.25, 61)
    p0, p1 = net.members[0], net.members[1]
    rms = np.array(
        [
            _rms_pmin(net, [(1 - u) * a + u * b for a, b in zip(p0, p1, strict=True)], Z, Y)
            for u in t
        ]
    )
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(t, rms, color=aml_style.PALETTE[1], lw=2.6)
    ax.plot([0, 1], [rms[t == 0][0], rms[np.isclose(t, 1)][0]], "o", color=aml_style.INK, ms=9)
    # Beside each minimum, where the valleys are narrow: higher up, the curve fills the width.
    ax.annotate("network 1", (0, rms[t == 0][0]), xytext=(12, 0), textcoords="offset points",
                fontsize=13, va="center")  # fmt: skip
    ax.annotate("network 2", (1, rms[np.isclose(t, 1)][0]), xytext=(-12, 0),
                textcoords="offset points", fontsize=13, va="center", ha="right")  # fmt: skip
    ax.set(
        xlabel="weights: (1 - t) network 1 + t network 2",
        ylabel="RMS error on its runs (m)",
        yscale="log",
    )
    aml_style.save(fig, out_dir, "barrier")
    mid = int(np.argmin(np.abs(t - 0.5)))
    return {
        "end0": float(rms[t == 0][0]),
        "end1": float(rms[np.isclose(t, 1)][0]),
        "mid": float(rms[mid]),
        "peak": float(rms[(t > 0) & (t < 1)].max()),
    }


def _rms_pmin(net: Networks, params: Params, Z: np.ndarray, Y: np.ndarray) -> float:
    """RMS error of the lowest pressure of the network with ``params`` on some runs.

    Args:
        net: The ensemble whose output scaler converts the predictions to metres.
        params: The weights to evaluate.
        Z: Standardized settings of the runs.
        Y: Their lowest pressures, m.

    Returns:
        The RMS error, m.
    """
    return float(np.sqrt(np.mean((net.ys.inverse(predict(params, Z))[:, 1] - Y) ** 2)))


def filter_direction(params: Params, rng: np.random.Generator) -> Params:
    """A random direction in weight space, scaled unit by unit to the unit's weights.

    Each unit's incoming weights (a column of a weight matrix) get a Gaussian direction of the
    same norm; biases get none. Li et al. (2018) draw loss surfaces along two such directions,
    so a step of 1 changes every unit by about its own size.

    Args:
        params: The network's weights.
        rng: The generator of the direction.

    Returns:
        The direction, shaped like ``params``.
    """
    out = []
    for i, p in enumerate(params):
        if i % 2:  # a bias
            out.append(np.zeros_like(p))
            continue
        r = rng.normal(size=p.shape)
        out.append(r * np.linalg.norm(p, axis=0) / (np.linalg.norm(r, axis=0) + 1e-12))
    return out


@dataclass(frozen=True)
class Landscape:
    """The squared error of the lowest pressure on two slices of weight space.

    Attributes:
        u: Coordinates of the close-up around network 1, and ``bowl`` the errors on them.
        v: Coordinates of the plane through networks 1, 2, 3, and ``flat`` the errors on them.
        line: Coordinates along each straight line between two networks.
        ridges: Errors along the lines from network 1 to 2, 1 to 3, and 2 to 3.
        valleys: The three networks' own errors.
        centroid: The error at the plane's centroid, the three networks' mean weights.
    """

    u: np.ndarray
    bowl: np.ndarray
    v: np.ndarray
    flat: np.ndarray
    line: np.ndarray
    ridges: dict[str, np.ndarray]
    valleys: list[float]
    centroid: float


def landscape(net: Networks, data: Data) -> Landscape:
    """The training loss around network 1 and over the plane through networks 1, 2, 3.

    Around network 1: along two filter-normalized random directions (Li et al., 2018). The
    plane: w1 + a (w2 - w1) + b (w3 - w1) (as Garipov et al., 2018, draw it).

    Args:
        net: The five networks on 1,024 runs of design 0.
        data: Every run (the training runs).

    Returns:
        The two slices, the three straight lines between the networks, and the valleys.
    """
    n = max(BUDGETS)
    Z, Y = net.xs(data.design[0, :n]), data.design_y[0, :n, 1]
    w1, w2, w3 = net.members[:3]
    rng = np.random.default_rng((SEED, 7))
    d1, d2 = filter_direction(w1, rng), filter_direction(w1, rng)

    def mse(params: Params) -> float:
        return _rms_pmin(net, params, Z, Y) ** 2

    def around(a: float, b: float) -> Params:
        return [w + a * x + b * y for w, x, y in zip(w1, d1, d2, strict=True)]

    def plane(a: float, b: float) -> Params:
        return [p + a * (q - p) + b * (r - p) for p, q, r in zip(w1, w2, w3, strict=True)]

    u = np.linspace(-BOWL_RADIUS, BOWL_RADIUS, BOWL_N)
    v = np.linspace(*PLANE_RANGE, PLANE_N)
    line = np.linspace(*PLANE_RANGE, LINE_N)
    return Landscape(
        u=u,
        bowl=np.array([[mse(around(a, b)) for a in u] for b in u]),
        v=v,
        flat=np.array([[mse(plane(a, b)) for a in v] for b in v]),
        line=line,
        ridges={
            "12": np.array([mse(plane(t, 0.0)) for t in line]),
            "13": np.array([mse(plane(0.0, t)) for t in line]),
            "23": np.array([mse(plane(1.0 - t, t)) for t in line]),
        },
        valleys=[mse(w) for w in (w1, w2, w3)],
        centroid=mse(plane(1 / 3, 1 / 3)),
    )


def landscape_figure(net: Networks, data: Data, out_dir: Path) -> dict[str, object]:
    """The training loss in 3D: close to one network, a bowl; through three networks, valleys.

    Left: the squared error of the lowest pressure on the 1,024 training runs around network 1
    (``landscape``). Right: over the plane through networks 1, 2, 3 of the ensemble, with the
    straight line from network 1 to network 2 (next slide) in orange.

    Args:
        net: The five networks on 1,024 runs of design 0.
        data: Every run (the training runs).
        out_dir: Figure directory.

    Returns:
        The bowl's bottom and edge, the valleys, the highest point of each straight line
        between two networks, and the centroid, m^2.
    """
    land = landscape(net, data)
    u, bowl, v, flat, line = land.u, land.bowl, land.v, land.flat, land.line
    ridge = land.ridges["12"]
    label = {
        "fontsize": 14,
        "fontweight": "bold",
        "color": aml_style.INK,
        "ha": "center",
        "va": "bottom",
        "path_effects": aml_style.HALO,
    }
    fig = plt.figure(figsize=aml_style.FULL)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.computed_zorder = False  # the marker and labels are drawn over the surface
    A, B = np.meshgrid(u, u)
    aml_style.surface3d(ax, A, B, np.minimum(bowl, LOSS_CAP), cmap=CMAP, vmax=LOSS_CAP)
    ax.scatter([0], [0], [land.valleys[0]], color="white", edgecolor=aml_style.INK, s=45, zorder=5)
    ax.text(0, 0, 2.0, "1", zorder=6, **label)
    ax.set(xticks=[], yticks=[], zlim=(0, LOSS_CAP), zticks=[0, 20, 40])
    ax.set_xlabel("random direction 1", labelpad=-12, fontsize=11)
    ax.set_ylabel("random direction 2", labelpad=-12, fontsize=11)
    ax.set_title("around network 1: a bowl", pad=0)
    aml_style.style3d(ax, elev=30, azim=-60)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.computed_zorder = False  # the path from network 1 to 2 is drawn over the ridge
    A, B = np.meshgrid(v, v)
    aml_style.surface3d(ax, A, B, np.minimum(flat, LOSS_CAP), cmap=CMAP, vmax=LOSS_CAP)
    ax.plot(line, 0 * line, np.minimum(ridge, LOSS_CAP) + 0.6, color=aml_style.PALETTE[1], lw=2.4,
            zorder=5)  # fmt: skip
    for k, (a, b) in enumerate(((0, 0), (1, 0), (0, 1))):
        ax.text(a, b, 2.0, str(k + 1), zorder=6, **label)
    ax.set(xticks=[], yticks=[], zlim=(0, LOSS_CAP), zticks=[0, 20, 40])
    ax.set_xlabel("toward network 2", labelpad=-12, fontsize=11)
    ax.set_ylabel("toward network 3", labelpad=-12, fontsize=11)
    ax.set_title("through networks 1, 2, 3: valleys", pad=14)  # above the box's top corner
    aml_style.style3d(ax, elev=62, azim=-48)  # from above: valley 3 is not hidden by a ridge
    fig.text(0.5, 0.02, "squared error on the 1,024 runs (m²)", ha="center", fontsize=12,
             color=aml_style.MUTED)  # fmt: skip
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.92, wspace=0.0)
    aml_style.save(fig, out_dir, "landscape")
    inner = (line > 0) & (line < 1)
    return {
        "bowl_min": float(bowl.min()),
        "bowl_edge_median": float(np.median(np.r_[bowl[0], bowl[-1], bowl[:, 0], bowl[:, -1]])),
        "valleys": land.valleys,
        "ridge_12": float(ridge[inner].max()),
        "ridge_13": float(land.ridges["13"][inner].max()),
        "ridge_23": float(land.ridges["23"][inner].max()),
        "centroid": land.centroid,
    }


def sheet_figure(fits: Fits, data: Data, out_dir: Path) -> dict[str, object]:
    """The network's surface of the lowest pressure at four steps of training, over the box.

    Network 1 of the 1,024-run ensemble (the training-curve run) at each step of ``SNAPSHOTS``,
    coloured; the simulator's surface as a grey mesh. Training folds a nearly flat sheet onto it.

    Args:
        fits: Every fitted surrogate (the snapshots).
        data: Every run (the map).
        out_dir: Figure directory.

    Returns:
        The RMS error over the map at each snapshot, m.
    """
    net = fits.nets[0][max(BUDGETS)]
    n = int(round((HI - LO) / GRID_STEP)) + 1
    g = np.arange(LO, HI + 1e-9, GRID_STEP)
    G1, G2 = np.meshgrid(g, g)
    truth = data.grid_y[:, 1].reshape(n, n).T
    zlim = (5.0, 35.0)
    fig = plt.figure(figsize=(aml_style.FULL[0], 3.0))
    rms = {}
    for k, step in enumerate(SNAPSHOTS):
        pred = net.ys.inverse(predict(fits.snaps[step], net.xs(data.grid)))[:, 1]
        rms[step] = float(np.sqrt(np.mean((pred - data.grid_y[:, 1]) ** 2)))
        ax = fig.add_subplot(1, len(SNAPSHOTS), k + 1, projection="3d")
        ax.plot_wireframe(G1, G2, truth, rstride=5, cstride=5, color=aml_style.MUTED, lw=0.5)
        aml_style.surface3d(ax, G1, G2, np.clip(pred.reshape(n, n).T, *zlim), cmap=CMAP,
                            vmin=zlim[0], vmax=zlim[1], alpha=0.95)  # fmt: skip
        ax.set(xlim=(LO, HI), ylim=(LO, HI), zlim=zlim, xticks=[], yticks=[], zticks=[])
        aml_style.style3d(ax, elev=24, azim=-135)
        ax.set_box_aspect((1, 1, 0.7), zoom=0.95)  # at 1.15 neighbouring boxes overlapped
        ax.set_title(f"step {step:,}\nerror {rms[step]:.2f} m", fontsize=13, pad=-4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0)
    aml_style.save(fig, out_dir, "sheet")
    return {"rms": rms}


def slope_figure(fits: Fits, data: Data, out_dir: Path) -> dict[str, object]:
    """Slope of the predicted lowest pressure along the slice: past the last kink it freezes.

    Args:
        fits: Every fitted surrogate.
        data: Every run (the slice).
        out_dir: Figure directory.

    Returns:
        Each network's last kink on the slice (m), the slopes at 90 m, and the simulator's.
    """
    S = np.c_[np.full_like(data.slice_s2, SLICE_S1), data.slice_s2]
    each = fits.nets[0][max(BUDGETS)].each(S)[:, :, 1]
    mid = 0.5 * (data.slice_s2[1:] + data.slice_s2[:-1])
    slope = np.diff(each, axis=1) / GRID_STEP
    truth = np.diff(data.slice_y[:, 1]) / GRID_STEP
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.axvspan(HI, data.slice_s2.max(), color="#F0F0F0", zorder=0)
    for i, row in enumerate(slope):
        ax.plot(
            mid, row, color=aml_style.PALETTE[1], lw=1.4, label="five networks" if i == 0 else None
        )
    ax.plot(mid, truth, color=aml_style.INK, ls="--", lw=2.2, label="simulator")
    ax.set(xlabel=f"PRV-2 setting (m); PRV-1 at {SLICE_S1:.0f} m", ylabel="Slope (m per m)")
    ax.text(
        HI + 12,
        0.30,
        "beyond\nthe runs",
        transform=ax.get_xaxis_transform(),
        fontsize=12,
        color=aml_style.MUTED,
        va="top",
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=12)  # above the curves
    aml_style.save(fig, out_dir, "slopes")
    last = [float(mid[np.flatnonzero(np.abs(np.diff(row)) > 1e-9)].max()) for row in slope]
    return {
        "last_kink": [rnd(v, 2) for v in last],
        "slope90": [rnd(v, 2) for v in np.sort(slope[:, -1])],
        "truth90": rnd(truth[-1], 2),
    }


def margin_figure(data: Data, fits: Fits, decided: Decision, out_dir: Path) -> None:
    """Zoom on the optimum: the true 24.6 m line, the networks' line, the margin, and the picks.

    Args:
        data: Every run (the window).
        fits: Every fitted surrogate.
        decided: The decision (picks, design 0's margin, the counts for the note).
        out_dir: Figure directory.
    """
    w = data.window
    a = np.unique(w[:, 0])
    b = np.unique(w[:, 1])
    shape = (len(a), len(b))
    truth = data.window_y[:, 1].reshape(shape).T
    leak = data.window_y[:, 0].reshape(shape).T
    net = fits.nets[0][max(BUDGETS)]
    pred = net(w)[:, 1].reshape(shape).T
    m = decided.margin
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    cs = ax.contour(
        a, b, leak, levels=np.arange(23.4, 24.21, 0.2), colors="#b0b0b0", linewidths=0.9
    )
    ax.clabel(cs, fmt="%.1f", fontsize=10)
    ax.contour(a, b, truth, levels=[P_SERVICE], colors=aml_style.INK, linewidths=2.6)
    ax.contour(
        a, b, pred, levels=[P_SERVICE], colors=aml_style.PALETTE[1], linestyles="--", linewidths=2.2
    )
    ax.contour(
        a,
        b,
        pred,
        levels=[P_SERVICE + m],
        colors=aml_style.PALETTE[0],
        linestyles="-.",
        linewidths=2.2,
    )
    picks, big = decided.picks, max(BUDGETS)
    ax.plot(
        *picks["networks", 0, big],
        "s",
        ms=9,
        color=aml_style.INK,
        mfc="white",
        label="pick, no margin",
        zorder=5,
    )
    ax.plot(
        *picks["networks_margin", 0, big],
        "*",
        ms=15,
        color=aml_style.PALETTE[3],
        mec=aml_style.INK,
        label="pick with the margin",
        zorder=5,
    )
    ax.plot([], [], color=aml_style.INK, lw=2.6, label="24.6 m, simulator")
    ax.plot([], [], color=aml_style.PALETTE[1], ls="--", lw=2.2, label="24.6 m, networks")
    ax.plot([], [], color=aml_style.PALETTE[0], ls="-.", lw=2.2, label="with the margin")
    nums = decided.numbers
    ax.text(
        1.04,
        0.04,
        f"Networks: {nums['short_by_model']['networks']} of {len(DESIGNS) * len(BUDGETS)} picks\n"
        f"truly short; with margins,\n{nums['margin']['all']['short']} of "
        f"{nums['margin']['all']['of']}.\n"
        f"Margin here: {m:.2f} m, from\n{CAL_RUNS} held-out runs.",
        transform=ax.transAxes,
        fontsize=13,
        va="bottom",
    )
    ax.set(xlabel="PRV-1 (m)", ylabel="PRV-2 (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=12)
    aml_style.save(fig, out_dir, "margin")


def impact(town: LTown, pick: tuple[float, float], out_dir: Path) -> dict[str, object]:
    """Customers within ``NEAR_M`` of the 24.6 m limit at the chosen settings and today's.

    Args:
        town: The calibrated model.
        pick: The chosen PRV-1, PRV-2 settings, m.
        out_dir: Figure directory.

    Returns:
        The customer count and how many sit near the limit at each setting.

    Raises:
        ValueError: If a customer is below the limit at either setting.
    """
    near = {}
    for name, s in (("pick", pick), ("today", TODAY)):
        low = town.customer_pressure(*s)  # each customer's lowest, which Run does not keep
        if not (low >= P_SERVICE - 1e-6).all():
            raise ValueError(f"a customer is below the limit at {s}")
        near[name] = low[low < P_SERVICE + NEAR_M]
    ltown_map(town, out_dir, "near_limit", list(near["pick"].index))
    return {
        "customers": len(town.customers),
        "near_m": NEAR_M,
        "near_pick": len(near["pick"]),
        "near_today": len(near["today"]),
        "nodes": sorted(near["pick"].index),
    }


def kinks_start_figure(data: Data, out_dir: Path) -> dict[str, int]:
    """Where the 64 first-layer kinks start, for three starts (design 0's first 256 runs).

    Args:
        data: Every run (design 0's settings scale the starts).
        out_dir: Figure directory.

    Returns:
        Kinks crossing the box at the start, keyed by a slug of each panel.
    """
    S = data.design[0, :BREAK_BUDGET]
    xs, raw = Scaler.fit(S), Scaler.identity(2)
    starts = (  # (key, panel title, initialization, settings scaler)
        ("he0", "Zero biases (He)", "he0", xs),
        ("uniform", "Uniform ±1/√fan-in", "uniform", xs),
        ("raw", "Settings in metres", "uniform", raw),
    )
    g = np.linspace(-5, 95, 200)
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL)
    out = {}
    for ax, (slug, title, scheme, sc) in zip(axes, starts, strict=True):
        net = Networks([init_params(SEED, (2, *HIDDEN, 2), scheme)], sc, sc, [])
        slope, icpt = kink_lines(net, 0)
        for k in np.flatnonzero(np.isfinite(slope)):
            ax.plot(
                g, slope[k] * g + icpt[k], color=aml_style.PALETTE[1], ls="-", lw=0.8, alpha=0.7
            )
        n_in = kinks_in_box(net)
        out[slug] = n_in
        ax.add_patch(
            plt.Rectangle((LO, LO), HI - LO, HI - LO, fill=False, ec=aml_style.INK, lw=2.4)
        )
        ax.set(
            xlim=(-5, 95),
            ylim=(-5, 95),
            aspect="equal",
            xticks=[0, 25, 60, 90],
            yticks=[0, 25, 60, 90],
        )
        ax.set_title(f"{title}\n{n_in} of {HIDDEN[0]} kinks cross the box")
        ax.set_xlabel("PRV-1 (m)")
    axes[0].set_ylabel("PRV-2 (m)")
    fig.tight_layout()
    aml_style.save(fig, out_dir, "kinks_start")
    return out


def backprop_example(data: Data) -> dict[str, float]:
    """The hand-made network's forward and backward pass at PRV-2 = 47 m, as the build shows it.

    The target is the simulator's lowest pressure rounded to the centimetre, so every number on
    the slides follows from the numbers shown.

    Args:
        data: Every run (the slice).

    Returns:
        The forward values, the loss, and every gradient, to 6 decimals.
    """
    i = int(np.flatnonzero(data.slice_s2 == HAND["s2"])[0])
    y = round(float(data.slice_y[i, 1]), 2)
    h1, h2 = max(0.0, HAND["s2"] - HAND["c1"]), max(0.0, HAND["s2"] - HAND["c2"])
    yhat = HAND["a"] + HAND["b"] * h1 + HAND["d"] * h2
    g = 2 * (yhat - y)
    out = {
        "y": y,
        "h1": h1,
        "h2": h2,
        "bh1": HAND["b"] * h1,
        "dh2": HAND["d"] * h2,
        "yhat": yhat,
        "loss": (yhat - y) ** 2,
        "g": g,
        "db": g * h1,
        "dd": g * h2,
        "da": g,
        "dc1": -g * HAND["b"],
        "dc2": -g * HAND["d"],
    }
    return {k: round(v, 6) for k, v in out.items()}


def write_tex(n: dict, out_dir: Path) -> Path:
    """Macros for the diagrams: the traced run and the backpropagation build.

    Args:
        n: The lecture's numbers (``trace`` and ``backprop``).
        out_dir: Figure directory.

    Returns:
        The path of ``numbers.tex``.
    """
    t = n["trace"]
    macros = {
        "TraceSone": tex_num(t["s"][0], ".0f"),
        "TraceStwo": tex_num(t["s"][1], ".0f"),
        "Zone": tex_num(t["z"][0], ".2f"),
        "Ztwo": tex_num(t["z"][1], ".2f"),
        "OnOne": str(t["on"][0]),
        "OnTwo": str(t["on"][1]),
        "Units": str(HIDDEN[0]),
        "PredLeak": tex_num(t["pred"][0], ".2f"),
        "PredP": tex_num(t["pred"][1], ".2f"),
        "TrueLeak": tex_num(t["truth"][0], ".2f"),
        "TrueP": tex_num(t["truth"][1], ".2f"),
        "NWeights": f"{t['weights']:,}",
        "NRuns": f"{t['runs']:,}",
    }
    bp = n["backprop"]
    macros |= {
        "BpA": tex_num(HAND["a"], ".2f"),
        "BpB": tex_num(HAND["b"], ".1f"),
        "BpD": tex_num(HAND["d"], ".1f"),
        "BpCOne": tex_num(HAND["c1"], ".0f"),
        "BpCTwo": tex_num(HAND["c2"], ".0f"),
        "BpS": tex_num(HAND["s2"], ".0f"),
        "BpHOne": tex_num(bp["h1"], ".0f"),
        "BpHTwo": tex_num(bp["h2"], ".0f"),
        "BpBH": tex_num(bp["bh1"], ".1f"),
        "BpDH": tex_num(bp["dh2"], ".1f"),
        "BpDHabs": tex_num(abs(bp["dh2"]), ".1f"),
        "BpYhat": tex_num(bp["yhat"], ".2f"),
        "BpY": tex_num(bp["y"], ".2f"),
        "BpLoss": tex_num(bp["loss"], ".4f"),
        "BpG": tex_num(bp["g"], ".2f"),
        "BpDb": tex_num(bp["db"], ".2f"),
        "BpDd": tex_num(bp["dd"], ".2f"),
        "BpDcOne": tex_num(bp["dc1"], "+.3f"),
        "BpDcTwo": tex_num(bp["dc2"], "+.3f"),
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


def main() -> None:
    """Generate every figure and number for CEE Lecture 14.

    Raises:
        ValueError: If the hand-made network's kinks are not the attempt's fitted ones.
    """
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    # One BLAS thread: the same sums in the same order here and in the notebook.
    threadpool_limits(1)
    print(f"{WORKERS} worker processes")
    t0 = time.perf_counter()
    data = simulate()
    print(f"simulated in {time.perf_counter() - t0:.0f} s")
    lines = shut_lines(data)
    town = LTown()
    numbers: dict[str, object] = {"setting": setting(data, town, lines, out_dir)}
    numbers["setting"]["run"] = run_seconds(town, data)  # alone, before the pool starts
    numbers["attempt"], ramps = attempt(data)
    if (HAND["c1"], HAND["c2"]) != (ramps.c1, ramps.c2):
        raise ValueError(f"the hand-made kinks must be the fitted ones: {ramps.c1}, {ramps.c2}")
    with ProcessPoolExecutor(WORKERS) as pool:
        fits = fit_all(pool, data)
    print(f"fitted in {time.perf_counter() - t0:.0f} s")
    print("seconds per fit (median over designs):", fit_seconds(fits))
    numbers["steps"] = training_length(fits, data)
    numbers["breaks"] = breaks(data, fits)
    numbers["compare"] = compare(fits, data, lines, out_dir)
    numbers["kinks"] = kinks(fits.nets[0][max(BUDGETS)], lines, out_dir)
    numbers["beyond"] = beyond(fits, data, out_dir)
    decided = decide(fits, data)
    numbers["decide"] = decided.numbers
    numbers["trace"] = trace(fits, data)
    numbers["backprop"] = backprop_example(data)
    numbers["verify"] = verify(data)
    # the visuals that carry the slides
    title_art_figure(out_dir)
    ramp_build(data, ramps, out_dir)
    relu_figure(out_dir)
    numbers["unit_maps"] = unit_maps(fits.nets[0][max(BUDGETS)], lines, data, out_dir)
    numbers["training"] = training_figure(fits, out_dir)
    numbers["barrier"] = barrier_figure(fits.nets[0][max(BUDGETS)], data, out_dir)
    numbers["landscape"] = landscape_figure(fits.nets[0][max(BUDGETS)], data, out_dir)
    numbers["sheet"] = sheet_figure(fits, data, out_dir)
    numbers["slopes"] = slope_figure(fits, data, out_dir)
    margin_figure(data, fits, decided, out_dir)
    ltown_map(town, out_dir, "ltown")
    pick = decided.picks["networks_margin", 0, max(BUDGETS)]
    numbers["impact"] = impact(town, pick, out_dir)
    numbers["kinks_start"] = kinks_start_figure(data, out_dir)
    aml_style.thumbnails(out_dir, ("ramp2", "kinks_start", "kinks", "beyond"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    write_tex(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
