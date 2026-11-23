"""Figures and numbers for CEE Lecture 23: physics-informed neural networks.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l23_physics_informed.py \
        --out build/fig/cee_l23_physics_informed

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

The question: an embankment loads 4 m of soft clay; the road can be paved once the clay has
reached 90% of its consolidation settlement. Three piezometers are read monthly for six months.
When will the clay be 90% consolidated?

- Physics: Terzaghi's 1-D consolidation, ``u_t = c_v u_zz``, drained at the top, no flow through
  the base, a uniform initial excess pore pressure (Terzaghi, 1943). Its series solution is
  exact; a Crank-Nicolson finite-difference solver is the classical numerical baseline. The
  layer's properties are assumed, not measured: c_v = 4 m2/year, 50 kPa of excess pressure
  (2.5 m of fill), piezometer noise 1 kPa.
- PINN (Raissi et al., 2019), written from scratch: a tanh MLP u(z, t) trained on the PDE's
  residual at random points (by automatic differentiation), the base's no-flow condition, the
  initial condition, and, for the inverse problem, the readings; the drained top is built into
  the network (u = z N(z, t)). Adam, then L-BFGS, in float64 on the CPU.
- Inverse problem: learn c_v from the readings, ten noise draws; against a one-parameter
  least-squares fit of Terzaghi's series and a network fitted to the readings alone (with the
  PINN's own start term, 50 kPa at 200 depths at t = 0: the known start, as the other two
  methods have it).
- Checks: autograd against finite differences of the network; the exact solutions give zero
  residual; the drained-water identity holds for the series; the FD solver converges.
- Break it: no initial condition; residual points only where the readings are; settlement read
  two ways (pore pressures, water out of the drain); a spill in a river (advection-dispersion
  against Ogata & Banks, 1961) at Peclet numbers 1 to 10,000, five seeds each, with one round
  of residual-based refinement (Lu et al., 2021) as the remedy, and as many random points added
  instead as its control.
- Plausibility: every quantity the slides quote is bounded by the physics before it is written,
  so a unit converted twice cannot reach a slide.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from scipy.linalg import solve_banded
from scipy.optimize import brentq, least_squares
from scipy.special import erfc, erfcx
from torch import nn

import aml_style
import title_art
from common import MINUS, parse_out_dir, seed_all, write_numbers

SEED = 23
# The site (assumed values)
H = 4.0  # m: clay thickness, drained at the top, impermeable base
CV = 4.0  # m2/year
U0 = 50.0  # kPa: excess pore pressure under 2.5 m of fill at 20 kN/m3
SIGMA = 1.0  # kPa: piezometer noise
DEPTHS = (1.0, 2.0, 3.0)  # m
MONTHS = (1, 2, 3, 4, 5, 6)  # readings
T_MAX = 5.0  # years: the PINN's time window
TARGET = 0.9  # pave at 90% average consolidation
N_DRAWS = 10  # noise draws for the inverse problem
N_SEEDS = 5  # network seeds for the forward and transport studies
N_NO_IC = 3  # seeds trained without the start term
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

# --------------------------------------------------------------------------- exact solutions


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


def ogata_banks(x: np.ndarray, tau: np.ndarray, pe: float) -> np.ndarray:
    """Concentration for a continuous release at X = 0 (Ogata & Banks, 1961), overflow-safe."""
    x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    s = 2 * np.sqrt(np.maximum(tau, 1e-300) / pe)
    a = (x + tau) / s
    out = 0.5 * (erfc((x - tau) / s) + np.exp(pe * x - a**2) * erfcx(a))
    return np.where(tau > 0, out, (x == 0).astype(float))


# --------------------------------------------------------------------------- classical solvers


def fd_consolidation(nz: int, nt: int, t_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Crank-Nicolson for u_T = u_ZZ on nz cells; a ghost node mirrors the impermeable base.

    Returns:
        The nodes Z (nz + 1,) and u / u0 at T = t_factor, (nz + 1,).
    """
    step, z = fd_step(nz, t_factor / nt)
    u = np.linalg.matrix_power(step, nt) @ np.ones(nz)
    return z, np.r_[0.0, u]


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


# --------------------------------------------------------------------------- the network


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


def setup_worker() -> None:
    """Each worker: float64, fixed threads, deterministic kernels."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def k_true() -> float:
    """The scaled diffusivity c_v T_MAX / H^2 of u_s = k u_zz on the unit square."""
    return CV * T_MAX / H**2


def readings(draw: int) -> dict[str, np.ndarray]:
    """The piezometer readings of one noise draw (draw 0 is the site's)."""
    rng = np.random.default_rng([SEED, draw])
    zz, tt = np.meshgrid(np.asarray(DEPTHS), np.asarray(MONTHS) / 12)  # (months, depths)
    u = U0 * series_u(zz / H, CV * tt / H**2) + rng.normal(0, SIGMA, zz.shape)
    return {"z": zz.ravel(), "t": tt.ravel(), "u": u.ravel()}


def sample_points(seed: int, window: float = T_MAX) -> dict[str, np.ndarray]:
    """The layer's training points on the unit square (zeta = z / H, s = t / T_MAX)."""
    rng = np.random.default_rng(seed)
    return {
        "z_res": rng.uniform(0, 1, N_RES),
        "s_res": rng.uniform(0, window / T_MAX, N_RES),
        "z_ic": rng.uniform(0, 1, N_IC),
        "s_base": rng.uniform(0, 1, N_BC),
    }


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
        zd, sd, ud = (torch.tensor(a) for a in (data["z"] / H, data["t"] / T_MAX, data["u"] / U0))

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


def degree_field(net: Net, t_years: np.ndarray) -> np.ndarray:
    """U(t) = 1 - mean over depth of the network's u / u0 (trapezoid, 401 depths)."""
    zq = torch.linspace(0, 1, 401)
    out = []
    with torch.no_grad():
        for tv in np.atleast_1d(t_years):
            u = net(zq, torch.full_like(zq, tv / T_MAX))
            out.append(1 - float(torch.trapezoid(u, zq)))
    return np.array(out)


def drained(
    u_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], k: float, s1: float, s2: float
) -> float:
    """Water out of the drain between s1 and s2, as a share of the layer's excess water.

    From u_s = k u_zz and no flow through the base, d/ds of the integral of u over depth is
    -k u_z(0, s): k times the integral of du/dz at the top (trapezoid, 4,001 times).
    """  # fmt: skip
    s = torch.linspace(s1, s2, 4001)
    z = torch.zeros_like(s, requires_grad=True)
    return k * float(torch.trapezoid(d(u_fn(z, s), z).detach(), s))


def degree_drained(net: Net, k: float, t_years: np.ndarray) -> np.ndarray:
    """U(t) read from the water drained since t = 0."""
    return np.array([drained(net, k, 0.0, tv / T_MAX) for tv in np.atleast_1d(t_years)])


def time_to(net: Net, degree: float) -> float | None:
    """When the network's own field reaches ``degree`` (years), or None within the window."""
    ts = np.linspace(0.01, T_MAX, 500)
    u = degree_field(net, ts)
    above = np.nonzero(u >= degree)[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(ts[0])
    return float(brentq(lambda tv: degree_field(net, np.array([tv]))[0] - degree, ts[i - 1], ts[i]))


def field_grid() -> tuple[np.ndarray, np.ndarray]:
    """The evaluation grid: depths (81,) in m, times (500,) in years, t > 0."""
    return np.linspace(0, H, 81), np.linspace(0, T_MAX, 501)[1:]


def predict(net: Net) -> np.ndarray:
    """The network's u (kPa) on ``field_grid``: (times, depths)."""
    zs, ts = field_grid()
    zz, tt = np.meshgrid(zs / H, ts / T_MAX)
    with torch.no_grad():
        u = net(torch.tensor(zz.ravel()), torch.tensor(tt.ravel())).numpy()
    return U0 * u.reshape(zz.shape)


def exact_grid() -> np.ndarray:
    """Terzaghi's u (kPa) on ``field_grid``: (times, depths)."""
    zs, ts = field_grid()
    zz, tt = np.meshgrid(zs / H, CV * ts / H**2)
    return U0 * series_u(zz, tt)


# --------------------------------------------------------------------------- jobs (worker side)


def forward_job(seed: int, w_ic: float) -> dict:
    """The forward problem: c_v known, no readings."""
    setup_worker()
    net, k, loss, secs = fit_consolidation(seed, w_ic=w_ic)
    t90 = time_factor_at(TARGET) * H**2 / CV
    probe = np.array([0.25, 1.0, t90])
    return {
        "seed": seed,
        "loss": loss,
        "seconds": secs,
        "field": predict(net),
        "probe_t": probe,
        "degree_field": degree_field(net, probe),
        "degree_drained": degree_drained(net, k, probe),
    }


def inverse_job(draw: int, physics: bool, window: float) -> dict:
    """The inverse problem on one noise draw: learn c_v (physics) or fit the readings alone.

    Args:
        draw: Noise draw of the readings (0 is the site's).
        physics: True learns c_v with the PDE; False fits the readings and the start alone
            (the PINN's start term, at ``N_IC`` depths).
        window: Residual points cover 0 <= t <= window (years).

    Returns:
        The learned c_v (m2/year, None without physics), the final loss, seconds, t90 from the
        network's field (years, None if not reached), U(t) on 251 times, and the field's extremes
        in kPa with where the largest sits.
    """
    setup_worker()
    net, k, loss, secs = fit_consolidation(
        SEED + draw, physics=physics, data=readings(draw), learn_k=physics, window=window
    )
    curve_t = np.linspace(0, T_MAX, 251)
    field_kpa = predict(net)  # kPa already: no second U0
    return {
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


def max_location(field: np.ndarray) -> dict[str, float]:
    """Depth (m) and time (years) of a field's largest value on ``field_grid``."""
    zs, ts = field_grid()
    it, iz = np.unravel_index(int(field.argmax()), field.shape)
    return {"depth": float(zs[iz]), "years": float(ts[it])}


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
        c = net(torch.tensor(xx.ravel()), torch.tensor(tt.ravel() / TAU)).numpy().reshape(xx.shape)
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


def first_crossing(ts: np.ndarray, c: np.ndarray, th: float) -> float | None:
    """The first time ``c`` reaches ``th``, or None."""
    hit = np.nonzero(c >= th)[0]
    return float(ts[hit[0]]) if len(hit) else None


def mass_balance_net(net: Net, pe: float) -> dict[str, float]:
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


def transport_grid() -> tuple[np.ndarray, np.ndarray]:
    """X (1001,) and tau (71,) for the river's errors: tau from 0.05 to 0.75 by 0.01."""
    return np.linspace(0, 1, 1001), np.round(np.arange(5, 76) / 100, 2)


def run_jobs(jobs: list[tuple[Callable[..., dict], tuple]]) -> list[dict]:
    """Run the jobs in worker processes; results in submission order."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(f, *args) for f, args in jobs]
        return [f.result() for f in futs]


# --------------------------------------------------------------------------- checks


def checks() -> dict[str, float]:
    """Autograd, exact residuals, the drained-water identity, and the FD solver's convergence."""
    setup_worker()
    torch.manual_seed(SEED)
    net = Net(hard_top=True)
    rng = np.random.default_rng(SEED)
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
        out[f"ogata_banks_torch_vs_numpy_{int(pe)}"] = float(np.abs(c.detach().numpy() - ref).max())
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
    clean = {"z": zz.ravel(), "t": tt_.ravel(), "u": U0 * series_u(zz / H, CV * tt_ / H**2).ravel()}
    out["series_fit_noise_free"] = abs(series_fit(clean) - CV)
    return out


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
        rows.append({"cells": nz, "max_error_kpa": U0 * err, "seconds": time.perf_counter() - t0})
    order = [np.log2(a["max_error_kpa"] / b["max_error_kpa"]) for a, b in itertools.pairwise(rows)]
    return {
        "grids": rows,
        "order": order,
        "time_factors": list(factors),
        "times_years": [tf * H**2 / CV for tf in factors],
    }


def series_fit(data: dict[str, np.ndarray]) -> float:
    """c_v by least squares on Terzaghi's series (one parameter, fitted on log c_v)."""

    def resid(p: np.ndarray) -> np.ndarray:
        return U0 * series_u(data["z"] / H, np.exp(p[0]) * data["t"] / H**2) - data["u"]

    return float(np.exp(least_squares(resid, [0.0], xtol=1e-14, ftol=1e-14).x[0]))


# --------------------------------------------------------------------------- figures


def attempt_figure(r: dict[str, np.ndarray], out_dir: Path) -> None:
    """The six months of readings, on an axis that runs to four years."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    months = np.asarray(MONTHS)
    for i, z in enumerate(DEPTHS):
        u = r["u"].reshape(len(MONTHS), len(DEPTHS))[:, i]
        ax.plot(np.r_[0, months], np.r_[U0, u], marker=aml_style.MARKERS[i], ls="-",
                color=aml_style.PALETTE[i], label=f"{z:.0f} m deep")  # fmt: skip
    ax.axvspan(6, 48, color="#f2f2f2", zorder=0)
    ax.text(27, 45, "?", fontsize=40, color=aml_style.MUTED, ha="center", va="center")
    ax.set(xlim=(0, 48), ylim=(0, 55), xlabel="months since the fill was placed",
           ylabel="excess pore pressure, kPa", xticks=range(0, 49, 6))  # fmt: skip
    ax.legend(loc="lower left", frameon=False)
    aml_style.save(fig, out_dir, "attempt")


def plane_axes(ax: plt.Axes) -> None:
    """Time across, depth down."""
    ax.set(xlim=(0, T_MAX), ylim=(H, 0), xlabel="years", ylabel="depth, m")


def picture_figures(fwd: dict, r: dict[str, np.ndarray], out_dir: Path) -> None:
    """The one picture: the exact field, the points the loss reads, the PINN's error."""
    zs, ts = field_grid()
    exact = exact_grid()
    # 1: the field, with the piezometers' readings
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    mesh = ax.pcolormesh(ts, zs, exact.T, cmap="Blues", vmin=0, vmax=U0, shading="auto",
                         rasterized=True)  # fmt: skip
    ax.scatter(r["t"], r["z"], marker="*", s=70, color=aml_style.PALETTE[1], zorder=3)
    plane_axes(ax)
    fig.colorbar(mesh, ax=ax, label="excess pore pressure, kPa")
    aml_style.save(fig, out_dir, "picture_1")
    # 2: where each loss term is evaluated: the points the network of picture 3 trained on
    p = sample_points(fwd["seed"])
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.scatter(T_MAX * p["s_res"], H * p["z_res"], s=2, color=aml_style.MUTED, alpha=0.5,
               linewidths=0, rasterized=True)  # fmt: skip
    ax.scatter(np.zeros(N_IC), H * p["z_ic"], s=8, color=aml_style.PALETTE[1], clip_on=False,
               zorder=3)  # fmt: skip
    ax.scatter(T_MAX * p["s_base"], np.full(N_BC, H), s=8, color=aml_style.PALETTE[2],
               clip_on=False, zorder=3)  # fmt: skip
    ax.scatter(r["t"], r["z"], marker="*", s=70, color=aml_style.NAVY, zorder=4)
    ax.axhline(0, color=aml_style.PALETTE[0], lw=4, clip_on=False)
    labels = [
        (2.5, 1.9, "PDE residual at random points"),
        (0.12, 3.5, "u = 50 kPa at t = 0"),
        (2.5, 3.75, "no flow through the base"),
        (2.5, 0.25, "drained top: u = 0, built into the network"),
        (0.6, 1.0, "readings"),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, fontsize=11, color=aml_style.INK, path_effects=aml_style.HALO,
                ha="left" if x < 1 else "center", va="center")  # fmt: skip
    plane_axes(ax)
    aml_style.save(fig, out_dir, "picture_2")
    # 3: the trained network's error
    err = np.abs(fwd["field"] - exact)
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    mesh = ax.pcolormesh(ts, zs, np.maximum(err.T, 1e-3), cmap="magma_r",
                         norm=LogNorm(1e-2, 10), shading="auto", rasterized=True)  # fmt: skip
    plane_axes(ax)
    fig.colorbar(mesh, ax=ax, extend="both", label=f"|PINN {MINUS} exact|, kPa")
    aml_style.save(fig, out_dir, "picture_3")


def inverse_figure(
    inv: list[dict], data_only: list[dict], fits: list[float], out_dir: Path
) -> None:
    """Left: U(t) by each method on the site's readings. Right: t90 over the noise draws."""
    t90_true = time_factor_at(TARGET) * H**2 / CV
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=aml_style.FULL, width_ratios=(2.2, 1), gridspec_kw={"wspace": 0.35}
    )
    tq = inv[0]["curve_t"]
    ax.axvspan(0, 0.5, color="#f2f2f2", zorder=0)
    ax.plot(tq, series_degree(CV * tq / H**2), color=aml_style.INK, lw=2.5, ls="-", label="exact")
    ax.plot(tq, series_degree(fits[0] * tq / H**2), color=aml_style.PALETTE[0], ls="--",
            label="series fit")  # fmt: skip
    ax.plot(tq, inv[0]["curve"], color=aml_style.PALETTE[1], ls="-", label="PINN")
    ax.plot(tq, data_only[0]["curve"], color=aml_style.PALETTE[3], ls="-.",
            label="readings only")  # fmt: skip
    ax.axhline(TARGET, color=aml_style.MUTED, ls=":", lw=1)
    lo = min(0.0, float(data_only[0]["curve"].min()))  # nothing clipped: the misfit shows
    hi = max(1.05, float(data_only[0]["curve"].max()))
    ax.set(xlim=(0, T_MAX), ylim=(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)), xlabel="years",
           ylabel="average consolidation")  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    t_fit = [time_factor_at(TARGET) * H**2 / cv for cv in fits]
    t_pinn = [r["t90_field"] for r in inv]
    for i, vals in enumerate((t_fit, t_pinn)):
        x = i + np.linspace(-0.15, 0.15, len(vals))
        bx.scatter(x, vals, s=18, color=aml_style.PALETTE[i], marker=aml_style.MARKERS[i])
    bx.axhline(t90_true, color=aml_style.INK, ls="-", lw=1.5)
    bx.set(xticks=[0, 1], xticklabels=["series\nfit", "PINN"], xlim=(-0.5, 1.5),
           ylabel="years to 90%")  # fmt: skip
    aml_style.save(fig, out_dir, "inverse")


def conservation_figure(fwds: list[dict], out_dir: Path) -> None:
    """At t90, each seed's settlement from its pore pressures and from the water drained."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    for i, f in enumerate(fwds):
        a, b = f["degree_field"][-1], f["degree_drained"][-1]
        ax.plot([i, i], [a, b], color=aml_style.MUTED, lw=1, ls="-")
        ax.scatter([i], [a], color=aml_style.PALETTE[0], marker="o", s=40, zorder=3,
                   label="from the pore pressures" if i == 0 else None)  # fmt: skip
        ax.scatter([i], [b], color=aml_style.PALETTE[1], marker="s", s=40, zorder=3,
                   label="from the water drained" if i == 0 else None)  # fmt: skip
    ax.axhline(TARGET, color=aml_style.INK, ls="-", lw=1)
    ax.set(xticks=range(len(fwds)), xticklabels=[f"seed {f['seed']}" for f in fwds],
           ylabel="consolidation at the true t90")  # fmt: skip
    ax.legend(loc="upper left", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 1.0))
    aml_style.save(fig, out_dir, "conservation")


def peclet_figure(runs: list[dict], fd: dict[float, dict], out_dir: Path) -> None:
    """Left: profiles at SHOW_TAU for SHOW_PE. Right: max error by Peclet number."""
    xs, taus = transport_grid()
    k = int(np.argmin(np.abs(taus - SHOW_TAU)))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"wspace": 0.3})
    ax.plot(xs, ogata_banks(xs, taus[k], SHOW_PE), color=aml_style.INK, lw=3, ls="-",
            label="exact")  # fmt: skip
    for j, r in enumerate(x for x in runs if x["pe"] == SHOW_PE and x["mode"] == "plain"):
        ax.plot(xs, r["field"][k], color=aml_style.PALETTE[1], lw=1.2, ls="-",
                label="PINN, five seeds" if j == 0 else None)  # fmt: skip
    ax.set(xlim=(0, 2.5 * SHOW_TAU), xlabel="distance down the river, X", ylabel="concentration")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    for rar, color, marker, dx in ((False, aml_style.PALETTE[1], "o", -0.06),
                                   (True, aml_style.PALETTE[2], "^", 0.06)):  # fmt: skip
        sel = [r for r in runs if r["mode"] == ("refine" if rar else "plain")]
        bx.scatter([r["pe"] * 10**dx for r in sel], [r["max_error"] for r in sel], s=20,
                   color=color, marker=marker,
                   label="PINN + refinement" if rar else "PINN")  # fmt: skip
    bx.plot(PECLETS, [fd[pe]["max_error"] for pe in PECLETS], color=aml_style.PALETTE[0],
            ls="--", marker="s", label="finite differences")  # fmt: skip
    bx.set(xscale="log", yscale="log", xlabel="Péclet number", ylabel="max error")
    bx.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    aml_style.save(fig, out_dir, "peclet")


def window_figure(full: dict, window: dict, out_dir: Path) -> None:
    """U(t) with residual points over five years against points only in the first six months."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    tq = full["curve_t"]
    ax.axvspan(0, 0.5, color="#f2f2f2", zorder=0)
    ax.plot(tq, series_degree(CV * tq / H**2), color=aml_style.INK, lw=2.5, ls="-", label="exact")
    ax.plot(tq, full["curve"], color=aml_style.PALETTE[1], ls="-", label="residual over 5 years")
    ax.plot(tq, window["curve"], color=aml_style.PALETTE[0], ls="--",
            label="residual over the first 6 months")  # fmt: skip
    ax.axhline(TARGET, color=aml_style.MUTED, ls=":", lw=1)
    lo = min(0.0, float(window["curve"].min()))
    hi = max(1.05, float(window["curve"].max()))
    ax.set(xlim=(0, T_MAX), ylim=(lo, hi), xlabel="years", ylabel="average consolidation")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    aml_style.save(fig, out_dir, "window")


def title_art_figure(out_dir: Path) -> None:
    """The title art: an embankment on layered clay, water rising out of it as light."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0d14", "#131826", "#1a1420", "#08080b"), grain=0.08)
    for y0, color in ((0.55, "#2a2233"), (0.68, "#2f2530"), (0.81, "#251d26")):
        c.glaze(c.polygon([(0, top * y0), (1, top * (y0 - 0.02)), (1, top * (y0 + 0.13)),
                           (0, top * (y0 + 0.13))]), color, 0.85)  # fmt: skip
    c.glaze(c.polygon(title_art.stepped(0.18, top * 0.55, 0.64, top * 0.16, 4)), "#4a2f2a", 0.9)
    rng = np.random.default_rng(SEED)
    drops = []
    for _ in range(140):
        x = rng.uniform(0.05, 0.95)
        y = top * (0.55 - rng.exponential(0.12))
        drops.append([(x, y), (x, y + 0.004)])
    c.glow(c.lines(drops, 2.0, soft=0.8), "#8fb3c9", 0.35, spread=3)
    c.glow(c.disc((0.72, top * 0.18), 0.05, soft=3), "#eaa548", 0.45, spread=24)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped embankment rests on bands of dark clay, "
        "and faint blue points of light rise out of the ground into the night; a gold sun glows "
        "above.",
    )


# --------------------------------------------------------------------------- stages


def log(msg: str) -> None:
    """Progress on stderr (stdout goes to /dev/null in the Makefile)."""
    print(msg, file=sys.stderr, flush=True)


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


def consolidation_runs() -> dict[str, list[dict]]:
    """Train every network of the clay layer in worker processes, in one pool.

    Returns:
        The runs by group, each in seed or draw order: ``forward`` (N_SEEDS seeds), ``no_ic``
        (N_NO_IC seeds, no start term), ``pinn`` (c_v learned), ``readings_only``, and ``window``
        (residual points in the first six months), the last three over N_DRAWS noise draws.
    """
    groups = {
        "forward": [(forward_job, (SEED + s, 1.0)) for s in range(N_SEEDS)],
        "no_ic": [(forward_job, (SEED + s, 0.0)) for s in range(N_NO_IC)],
        "pinn": [(inverse_job, (dr, True, T_MAX)) for dr in range(N_DRAWS)],
        "readings_only": [(inverse_job, (dr, False, T_MAX)) for dr in range(N_DRAWS)],
        "window": [(inverse_job, (dr, True, 0.5)) for dr in range(N_DRAWS)],
    }
    res = run_jobs([job for jobs in groups.values() for job in jobs])
    out, i = {}, 0
    for name, jobs in groups.items():
        out[name] = res[i : i + len(jobs)]
        i += len(jobs)
    assert i == len(res), (i, len(res))
    return out


def series_fits() -> tuple[list[float], list[float]]:
    """The one-parameter series fit on every noise draw.

    Returns:
        c_v (m2/year) per draw, and the seconds each fit took.
    """
    fits, secs = [], []
    for dr in range(N_DRAWS):
        t0 = time.perf_counter()
        fits.append(series_fit(readings(dr)))
        secs.append(time.perf_counter() - t0)
    return fits, secs


def site_numbers() -> dict[str, object]:
    """The site's readings and Terzaghi's exact answers: t90, U at the probe times.

    Returns:
        The readings of draw 0, t90 (years) and T90, U at six months, the 3-m piezometer's
        pressure at t90 (kPa), and the exact U at ``forward_job``'s probe times.
    """
    t90_true = time_factor_at(TARGET) * H**2 / CV
    return {
        "readings": {k: v.tolist() for k, v in readings(0).items()},
        "t90": t90_true,
        "T90": time_factor_at(TARGET),
        "degree_6_months": float(series_degree(CV * 0.5 / H**2)),
        "u_3m_at_t90": float(U0 * series_u(0.75, time_factor_at(TARGET))),
        "degree_exact": dict(  # at forward_job's probe times
            zip(
                ("3 months", "1 year", "t90"),
                series_degree(CV * np.array([0.25, 1.0, t90_true]) / H**2).tolist(),
                strict=True,
            )
        ),
    }


def inverse_numbers(
    runs: dict[str, list[dict]], fits: list[float], fit_secs: list[float]
) -> dict[str, object]:
    """Per draw: the series fit, the PINN, the readings alone, and the six-month window.

    Args:
        runs: ``consolidation_runs``'s output.
        fits: The series fit's c_v per draw (m2/year).
        fit_secs: Seconds per series fit.

    Returns:
        c_v, t90 (years) and t90's error relative to the truth per method and draw; for the
        readings alone also the extremes of U and of the pore pressure (kPa).
    """
    t90_true = time_factor_at(TARGET) * H**2 / CV
    inv, only, win = runs["pinn"], runs["readings_only"], runs["window"]
    fit_t90 = [time_factor_at(TARGET) * H**2 / cv for cv in fits]

    def t90s(rows: list[dict]) -> list[float | None]:
        return [r["t90_field"] for r in rows]

    def rel(vals: list[float | None]) -> list[float | None]:
        return [None if v is None else (v - t90_true) / t90_true for v in vals]

    def degree_at(rows: list[dict], years: tuple[int, ...]) -> dict[str, list[float]]:
        return {
            f"{y} years": [float(np.interp(y, r["curve_t"], r["curve"])) for r in rows]
            for y in years
        }

    return {
        "series_fit": {"cv": fits, "t90": fit_t90, "t90_rel": rel(fit_t90), "seconds": fit_secs},
        "pinn": {
            "cv": [r["cv"] for r in inv],
            "t90": t90s(inv),
            "t90_rel": rel(t90s(inv)),
            "seconds": [r["seconds"] for r in inv],
        },
        "readings_only": {
            "start_points": N_IC,  # u = U0 at t = 0, at the PINN's start depths
            "t90": t90s(only),
            "max_degree": [float(r["curve"].max()) for r in only],
            "min_u_kpa": [r["min_u_kpa"] for r in only],
            "max_u_kpa": [r["max_u_kpa"] for r in only],
            "max_u_at": [r["max_at"] for r in only],
            "min_degree": [float(r["curve"].min()) for r in only],
            "degree_at": degree_at(only, (1, 2, 3)),
        },
        "window": {
            "cv": [r["cv"] for r in win],
            "t90": t90s(win),
            "degree_at": degree_at(win, (1, 3, 5)),
        },
    }


def consolidation_stage(out_dir: Path) -> dict[str, object]:
    """Forward, inverse, readings only, no initial condition, the six-month window.

    Args:
        out_dir: Where the figures go.

    Returns:
        The site's exact answers, the forward runs' errors, the no-start runs, and the inverse
        problem's numbers (``inverse_numbers``).
    """
    runs = consolidation_runs()
    fits, fit_secs = series_fits()
    r0 = readings(0)
    attempt_figure(r0, out_dir)
    picture_figures(runs["forward"][0], r0, out_dir)
    inverse_figure(runs["pinn"], runs["readings_only"], fits, out_dir)
    conservation_figure(runs["forward"], out_dir)
    window_figure(runs["pinn"][0], runs["window"][0], out_dir)
    return {
        "site": site_numbers(),
        "forward": [summarize_forward(f) for f in runs["forward"]],
        "no_ic": [
            {
                "seed": f["seed"],
                "max_abs_kpa": float(np.abs(f["field"]).max()),
                "degree_field": f["degree_field"].tolist(),
            }
            for f in runs["no_ic"]
        ],
        **inverse_numbers(runs, fits, fit_secs),
    }


def transport_stage(out_dir: Path) -> dict[str, object]:
    """The river: finite differences, then PINNs at each Peclet number, five seeds each."""
    xs, taus = transport_grid()
    fd = {}
    for pe in PECLETS:
        xg, cg, secs = fd_transport(pe, taus)
        cf = np.array([np.interp(xs, xg, row) for row in cg])
        exact = ogata_banks(xs[None, :], taus[:, None], pe)
        fd[pe] = {"max_error": float(np.abs(cf - exact).max()), "seconds": secs,
                  "cells": len(xg) - 1}  # fmt: skip
        log(f"  FD Pe {pe}: max error {fd[pe]['max_error']:.2e}, {secs:.1f} s")
    jobs = [(transport_job, (pe, SEED + s, "plain")) for pe in PECLETS for s in range(N_SEEDS)]
    jobs += [(transport_job, (pe, SEED + s, m)) for m in ("refine", "random") for pe in RAR_PECLETS
             for s in range(N_SEEDS)]  # fmt: skip
    runs = run_jobs(jobs)
    ta = np.linspace(0, TAU, 3001)
    for r in runs:
        exact = ogata_banks(xs[None, :], taus[:, None], r["pe"])
        err = np.abs(r["field"] - exact)
        r["max_error"] = float(err.max())
        r["rel_l2"] = float(np.linalg.norm(err) / np.linalg.norm(exact))
        it, ix = np.unravel_index(int(err.argmax()), err.shape)
        r["max_at"] = {"x": float(xs[ix]), "tau": float(taus[it])}  # the front sits at X = tau
        r["shown_error"] = float(err[int(np.argmin(np.abs(taus - SHOW_TAU)))].max())
    peclet_figure(runs, fd, out_dir)

    def table(mode: str) -> dict[str, dict[str, list]]:
        out = {}
        for pe in PECLETS if mode == "plain" else RAR_PECLETS:
            sel = [r for r in runs if r["pe"] == pe and r["mode"] == mode]
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

    return {
        "fd": {str(k): v for k, v in fd.items()},
        "exact_arrival": {
            str(pe): {
                "5pct": first_crossing(ta, ogata_banks(INTAKE, ta, pe), 0.05),
                "50pct": first_crossing(ta, ogata_banks(INTAKE, ta, pe), 0.5),
            }
            for pe in PECLETS
        },
        "pinn": table("plain"),
        "refined": table("refine"),
        "random_added": table("random"),
    }


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
                "error_rank": 1 + [f["seed"] for f in worst_first].index(low["seed"]),  # 1: worst
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
            "seconds_ratio_to_fd": float(np.median([f["seconds"] for f in fwd])) / fd80["seconds"],
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
            "worst_rel": worst_abs(fit_rel),
            "seconds": span(c["series_fit"]["seconds"]),
        },
        "pinn": {
            "t90": span(c["pinn"]["t90"]),
            "cv": span(c["pinn"]["cv"]),
            "worst_rel": worst_abs(pinn_rel),
            "seconds": span(c["pinn"]["seconds"]),
            "worse_than_fit": int(
                sum(p is None or abs(p) > abs(f) for p, f in zip(pinn_rel, fit_rel, strict=True))
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


def plausible(n: dict) -> None:
    """Bound every quantity the slides quote by the physics, to its order of magnitude.

    A value converted twice ("-34,914 kPa") looked no more absurd than the right one
    (-698 kPa), so each quantity gets a bound from the physics, even when the expected value is
    itself absurd.

    Args:
        n: The numbers about to be written.

    Raises:
        AssertionError: If a value lies outside what the physics allows.
    """
    c, s = n["consolidation"], n["consolidation"]["summary"]
    site = c["site"]
    # Terzaghi: T90 = 0.848 (the textbook value), so t90 = 0.848 H^2 / c_v = 3.39 years
    assert abs(site["T90"] - 0.848) < 1e-3 and abs(site["t90"] - 0.848 * H**2 / CV) < 0.01, site
    assert 0 < site["degree_6_months"] < site["degree_exact"]["t90"] < TARGET + 1e-9, site
    assert 0 < site["u_3m_at_t90"] < U0, site  # kPa, under the load
    assert all(0 < u < U0 + 5 * SIGMA for u in site["readings"]["u"]), site["readings"]
    # The checks are round-off, or the second difference's truncation at h = 1e-4
    assert all(0 <= v < 1e-6 for v in n["checks"].values()), n["checks"]
    # Finite differences: second order; errors far below the 1-kPa reading noise
    assert all(1.9 < o < 2.1 for o in n["fd"]["order"]), n["fd"]["order"]
    assert all(0 < g["max_error_kpa"] < SIGMA for g in n["fd"]["grids"]), n["fd"]["grids"]
    assert 0 < n["fd"]["on_grid"]["max_kpa"] < SIGMA, n["fd"]["on_grid"]
    assert np.allclose(n["fd"]["times_years"][-1], site["t90"]), n["fd"]["times_years"]
    # The forward PINN: errors in kPa below the load itself; U from the pressures in [0, 1]
    # up to a network's slack; U from the water drained is not conserved (1.43 seen), so only
    # its order of magnitude
    for f in c["forward"]:
        assert 0 < f["max_kpa"] < U0 and 0 < f["after_3_months_max_kpa"] < U0, f
        assert 0 < f["rel_l2"] < 1 and f["loss"] > 0, f
        assert all(0 <= v <= 1.05 for v in f["degree_field"].values()), f
        assert all(-1 < v < 2 for v in f["degree_drained"].values()), f
    for f in c["no_ic"]:  # whatever it learns, a pressure in kPa below the load
        assert f["max_abs_kpa"] < U0 and all(-1 < v < 2 for v in f["degree_field"]), f
    # c_v within a factor of two for the physics-based methods; t90 inside the window
    for key in ("series_fit", "pinn", "window"):
        assert all(CV / 2 < v < 2 * CV for v in c[key]["cv"]), (key, c[key]["cv"])
    for key in ("series_fit", "pinn", "readings_only", "window"):
        assert all(t is None or 0 < t <= T_MAX for t in c[key]["t90"]), (key, c[key]["t90"])
    # The readings alone have no physics, but a network fitted to readings of 15 to 50 kPa
    # reaches hundreds of kPa at most, not tens of thousands (a second U0 would give that)
    only = c["readings_only"]
    assert all(-50 * U0 < v < U0 for v in only["min_u_kpa"]), only["min_u_kpa"]
    assert all(0 < v < 50 * U0 for v in only["max_u_kpa"]), only["max_u_kpa"]
    assert all(abs(v) < 51 for v in only["min_degree"] + only["max_degree"]), only
    assert s["readings_only"]["min_u_kpa"] == min(only["min_u_kpa"]), s["readings_only"]
    # The river: concentrations in [0, 1]; a network may overshoot, not by orders of magnitude
    r = n["river"]
    assert all(0 < v["max_error"] < 0.05 for v in r["fd"].values()), r["fd"]
    for table in ("pinn", "refined", "random_added"):
        for pe, v in r[table].items():
            assert all(0 < e < 2 for e in v["max_error"]), (table, pe, v["max_error"])
            assert all(abs(m) < 1 for m in v["mass_error"]), (table, pe, v["mass_error"])
            arrivals = v["arrival_5pct"] + v["arrival_50pct"]
            assert all(a is None or 0 <= a <= TAU for a in arrivals), (table, pe, arrivals)
    assert n["settings"]["parameters"] == 3 * WIDTH + (LAYERS - 1) * (WIDTH + 1) * WIDTH + WIDTH + 1


def main() -> None:
    """Generate every figure and number for CEE Lecture 23."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    setup_worker()
    t0 = time.perf_counter()
    numbers: dict[str, object] = {
        "settings": {
            "H": H,
            "cv": CV,
            "u0": U0,
            "sigma": SIGMA,
            "depths": list(DEPTHS),
            "months": list(MONTHS),
            "t_max": T_MAX,
            "target": TARGET,
            "draws": N_DRAWS,
            "seeds": N_SEEDS,
            "no_ic_seeds": N_NO_IC,
            "width": WIDTH,
            "layers": LAYERS,
            "adam_steps": ADAM_STEPS,
            "adam_lr": ADAM_LR,
            "lbfgs_iters": LBFGS_ITERS,
            "n_res": N_RES,
            "n_ic": N_IC,
            "n_bc": N_BC,
            "tau": TAU,
            "peclets": list(PECLETS),
            "rar_pool": RAR_POOL,
            "rar_add": RAR_ADD,
            "intake": INTAKE,
            "threads": THREADS,
            "parameters": sum(p.numel() for p in Net(hard_top=True).parameters()),
        },
        "checks": checks(),
        "fd": fd_convergence() | {"on_grid": fd_on_grid()},
    }
    log(f"checks in {time.perf_counter() - t0:.0f} s")
    numbers["consolidation"] = consolidation_stage(out_dir)
    log(f"consolidation in {time.perf_counter() - t0:.0f} s")
    numbers["river"] = transport_stage(out_dir)
    log(f"river in {time.perf_counter() - t0:.0f} s")
    numbers["consolidation"]["summary"] = consolidation_summary(
        numbers["consolidation"], numbers["fd"]
    )
    numbers["river"]["summary"] = river_summary(numbers["river"])
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_3", "inverse", "conservation", "peclet"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    plausible(numbers)
    write_numbers(numbers, out_dir)
    log(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))  # read by the spawned workers
    main()
