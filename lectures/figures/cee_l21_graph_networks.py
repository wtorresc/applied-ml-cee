"""Figures and numbers for CEE Lecture 21: graph neural networks, on a water network.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l21_graph_networks.py \
        --out build/fig/cee_l21_graph_networks

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

The question: L-Town's utility reads 33 pressure sensors every 5 minutes; 749 junctions are
not measured. Estimate the hydraulic head (pressure + elevation) at every junction from the
sensors, the tank level, and the valve settings, and flag customers below 35 psi.

- Data: L-Town (BattLeDIM; Vrachimis et al., 2022; CC BY 4.0) simulated with WNTR/EPANET through
  ``ltown.py`` (its calibrated background leakage), over days that vary the weekday, the town's
  demand (x0.7 to x1.3), each junction's demand (lognormal noise), and the two PRV settings.
  Scenarios are split by day: training, validation (early stopping), test.
- Models, all trained on the same snapshots with the same protocol (each sensor hidden with
  probability 0.2 per snapshot; Adam; early stopping on the validation days):
  a message-passing network written from scratch (``index_add_``; Gilmer et al., 2017), with
  12 layers, one more than the most pipes between a junction and its nearest sensor; the same
  network with 2 and 5 layers; and a multilayer perceptron on the 33 readings (no graph).
  Baselines with no learning: each junction's training mean, and the head of the nearest
  observed node in the same pressure zone (valves and the pump cut the graph into zones).
- Real data: BattLeDIM's 2019 SCADA pressures (first four weeks, hourly), one sensor hidden at a
  time and estimated from the other 32, against the hydraulic model alone (EPANET with the
  file's demand patterns, aligned by weekday on 2018 data, no sensors).
- Break it: too few layers, a closed pipe (the graph changes), demand beyond training, and
  simulation against reality.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import wntr
from matplotlib.collections import LineCollection
from torch import nn

import aml_style
import ltown
import title_art
from common import parse_out_dir, require, seed_all, write_numbers

SEED = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCADA = {year: ltown.RAW / f"{year}_SCADA_Pressures.csv" for year in (2018, 2019)}
LEVELS = {year: ltown.RAW / f"{year}_SCADA_Levels.csv" for year in (2018, 2019)}
STRIDE = 12  # one snapshot an hour of a simulated day (the file's step is 5 min)
SCENARIOS = {"train": 240, "val": 30, "test": 30}  # simulated days
SCALE = (0.7, 1.3)  # the town's demand multiplier, per day
FAR_SCALE = (1.5, 1.8)  # beyond training: the extrapolation test
NOISE = 0.3  # each junction's demand multiplier: lognormal, mean 1, this sigma
P_HIDE = 0.2  # each sensor hidden with this probability in a training snapshot
LAYERS = 12  # one more than the most pipes between a junction and its nearest sensor (11)
FEW_LAYERS = (2, 5)
HIDDEN = 64
MLP_HIDDEN = 512
BATCH = 64
LRS = (3e-4, 1e-3)  # every model's first Adam rates; each keeps its best on validation
LR_LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # widens along this
MAX_EPOCHS = 200
PATIENCE = 15  # epochs without a better validation error before stopping
REQUIRE_CONVERGED = True  # off only in quick tests of the code paths
REAL_DAYS = 28  # the first four weeks of 2019, hourly
WORKERS = min(32, os.cpu_count() or 1)
NODE_F = 6  # observed head, observed flag, elevation, junction / reservoir / tank
EDGE_F = 7  # log resistance, pipe, PRV forward / backward, PRV head, pump forward / backward


def log(msg: str) -> None:
    """Progress to stderr (the Makefile discards stdout)."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- the network as a graph


@dataclass
class Graph:
    """L-Town as a directed graph (each link both ways) with node and edge features.

    Attributes:
        nodes: Node names, in WNTR's order (junctions, reservoirs, tank).
        kind: 0 junction, 1 reservoir, 2 tank, per node.
        elev: Elevation, m (a reservoir's head, the tank's bottom).
        sensors: Node indices of the 33 pressure sensors.
        src, dst: Edge ends, (E,) each.
        edge: Static edge features (E, EDGE_F); the PRV heads are filled per snapshot.
        prv: For PRV-1, PRV-2, PRV-3: the forward and backward edge indices and the outlet.
        link_of: The link name of each edge.
        hops: Pipes between each node and its nearest sensor (any link).
        xy: Map coordinates (N, 2).
        area: 0, 1, 2 for a junction of Area A, B, C (``ltown.zones``); -1 for the others.
    """

    nodes: list[str]
    kind: np.ndarray
    elev: np.ndarray
    sensors: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    edge: np.ndarray
    prv: dict[str, tuple[int, int, int]]
    link_of: list[str]
    hops: np.ndarray
    xy: np.ndarray
    area: np.ndarray

    @property
    def fixed(self) -> np.ndarray:
        """Reservoirs and the tank: their heads are always known."""
        return np.flatnonzero(self.kind > 0)

    @property
    def junctions(self) -> np.ndarray:
        """Junction indices."""
        return np.flatnonzero(self.kind == 0)

    def without(self, link: str) -> Graph:
        """The same graph with one link removed (both directions), for a closed pipe."""
        keep = np.array([lk != link for lk in self.link_of])
        old = np.cumsum(keep) - 1  # new index of each kept edge
        prv = {k: (int(old[f]), int(old[b]), o) for k, (f, b, o) in self.prv.items()}
        return Graph(
            self.nodes,
            self.kind,
            self.elev,
            self.sensors,
            self.src[keep],
            self.dst[keep],
            self.edge[keep],
            prv,
            [lk for lk, k in zip(self.link_of, keep, strict=True) if k],
            self.hops,
            self.xy,
            self.area,
        )


def sensor_names() -> list[str]:
    """The 33 pressure sensors, as BattLeDIM's SCADA files name them (junction names)."""
    cols = pd.read_csv(require(SCADA[2019], "scripts/fetch_battledim.sh"), sep=";", nrows=0)
    return [c for c in cols.columns if c != "Timestamp"]


def build_graph(wn: wntr.network.WaterNetworkModel) -> Graph:
    """Nodes, edges, and features of the network (open links only)."""
    nodes = list(wn.node_name_list)
    index = {n: i for i, n in enumerate(nodes)}
    kind = np.array(
        [{"Junction": 0, "Reservoir": 1, "Tank": 2}[wn.get_node(n).node_type] for n in nodes]
    )
    elev = np.array(
        [
            wn.get_node(n).base_head if k == 1 else wn.get_node(n).elevation
            for n, k in zip(nodes, kind, strict=True)
        ]
    )
    src, dst, feats, link_of, prv = [], [], [], [], {}
    log_r = {}
    for name, p in wn.pipes():  # Hazen-Williams resistance, h = r q^1.852 (SI)
        log_r[name] = np.log10(10.67 * p.length / (p.roughness**1.852 * p.diameter**4.87))
    mu, sd = np.mean(list(log_r.values())), np.std(list(log_r.values()))
    for name, link in wn.links():
        if link.initial_status == wntr.network.LinkStatus.Closed:
            continue
        a, b = index[link.start_node_name], index[link.end_node_name]
        f = np.zeros((2, EDGE_F), dtype=np.float32)  # forward a->b, backward b->a
        if link.link_type == "Pipe":
            f[:, 0] = (log_r[name] - mu) / sd
            f[:, 1] = 1
        elif link.link_type == "Valve":
            f[0, 2] = f[1, 3] = 1
            prv[name] = (len(src), len(src) + 1, b)
        else:  # the pump
            f[0, 5] = f[1, 6] = 1
        src += [a, b]
        dst += [b, a]
        feats += [f[0], f[1]]
        link_of += [name, name]
    sensors = np.array([index[s] for s in sensor_names()])
    g = nx.Graph()
    g.add_edges_from(zip(src, dst, strict=True))
    d = nx.multi_source_dijkstra_path_length(g, set(sensors.tolist()), weight=None)
    hops = np.array([d.get(i, -1) for i in range(len(nodes))])
    xy = np.array([wn.get_node(n).coordinates for n in nodes], dtype=float)
    area = np.full(len(nodes), -1)
    for i, names in enumerate(ltown.zones(wn)):
        area[[index[n] for n in names]] = i
    return Graph(
        nodes,
        kind,
        elev,
        sensors,
        np.array(src),
        np.array(dst),
        np.array(feats),
        prv,
        link_of,
        hops,
        xy,
        area,
    )


# ---------------------------------------------------------------- simulated days


@dataclass(frozen=True)
class Scenario:
    """One simulated day: weekday, PRV-1 and PRV-2 settings (m), demand scale, noise seed."""

    day: int
    s1: float
    s2: float
    scale: float
    seed: int
    closed: str | None = None


def draw_scenarios(
    n: int, rng: np.random.Generator, scale: tuple[float, float], closed: str | None = None
) -> list[Scenario]:
    """Days with random weekday, valve settings in Lecture 12's box, and demand scale."""
    return [
        Scenario(
            int(rng.integers(7)),
            float(rng.uniform(*ltown.BOX)),
            float(rng.uniform(*ltown.BOX)),
            float(rng.uniform(*scale)),
            int(rng.integers(2**31)),
            closed,
        )
        for _ in range(n)
    ]


def simulate(sc: Scenario) -> np.ndarray:
    """Heads (snapshots, nodes) of one day, every ``STRIDE``-th step, in a pool worker.

    Each junction's demands are scaled by ``scale`` times its own lognormal factor, the
    weekday sets the demand patterns' start, and a ``closed`` pipe is shut; all are restored
    after the run, so the worker's model is unchanged. ``LTown._simulate`` is used directly:
    it keeps ``ltown``'s calibrated leakage and solver checks without changing that module
    (Lectures 12 and 14 depend on it).
    """
    town = ltown.worker()
    wn = town.wn
    rng = np.random.default_rng(sc.seed)
    factor = sc.scale * rng.lognormal(-(NOISE**2) / 2, NOISE, len(wn.junction_name_list))
    saved = []
    for f, name in zip(factor, wn.junction_name_list, strict=True):
        for d in wn.get_node(name).demand_timeseries_list:
            saved.append((d, d.base_value))
            d.base_value *= f
    start = wn.options.time.pattern_start
    link = wn.get_link(sc.closed) if sc.closed else None
    try:
        wn.options.time.pattern_start = sc.day * ltown.DAY_S
        if link is not None:
            link.initial_status = wntr.network.LinkStatus.Closed
        res = town._simulate(sc.s1, sc.s2)
    finally:
        for d, v in saved:
            d.base_value = v
        wn.options.time.pattern_start = start
        if link is not None:
            link.initial_status = wntr.network.LinkStatus.Open
    return res.node["head"][list(wn.node_name_list)].to_numpy()[::STRIDE].astype(np.float32)


def simulate_many(scs: list[Scenario]) -> np.ndarray:
    """Heads (days, snapshots, nodes) for many days, in parallel processes."""
    with ProcessPoolExecutor(WORKERS, initializer=ltown.init_worker) as ex:
        return np.stack(list(ex.map(simulate, scs, chunksize=max(1, len(scs) // WORKERS))))


@dataclass
class Snapshots:
    """Flattened snapshots: heads (n, N) and the PRV-1, PRV-2 settings (n, 2)."""

    heads: np.ndarray
    settings: np.ndarray
    hour: np.ndarray

    @classmethod
    def of(cls, scs: list[Scenario], heads: np.ndarray) -> Snapshots:
        """From simulated days: every snapshot of every day."""
        t = heads.shape[1]
        settings = np.repeat([[s.s1, s.s2] for s in scs], t, axis=0).astype(np.float32)
        hour = np.tile(np.arange(t) * STRIDE * ltown.STEP_S / 3600, len(scs))
        return cls(heads.reshape(-1, heads.shape[2]), settings, hour)


# ---------------------------------------------------------------- inputs


@dataclass
class Scale:
    """Heads and elevations share one standardization, so pressure stays a difference."""

    mu: float
    sd: float

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Standardize."""
        return (h - self.mu) / self.sd


def inputs(
    g: Graph, sc: Scale, heads: torch.Tensor, observed: torch.Tensor, settings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Node features (B, N, NODE_F) and edge features (B, E, EDGE_F) of a batch, on its device.

    Args:
        g: The graph.
        sc: The standardization.
        heads: Heads (B, N); only the observed ones are used (others may be NaN).
        observed: Which nodes are observed (B, N): visible sensors, reservoirs, the tank.
        settings: PRV-1 and PRV-2 settings (B, 2), m; PRV-3 keeps the file's setting.
    """
    b, n = heads.shape
    dev = heads.device
    x = torch.zeros(b, n, NODE_F, device=dev)
    x[..., 0] = torch.where(observed, (heads - sc.mu) / sc.sd, 0.0)
    x[..., 1] = observed.float()
    x[..., 2] = torch.as_tensor((g.elev - sc.mu) / sc.sd, dtype=torch.float32, device=dev)
    x[:, torch.arange(n, device=dev), 3 + torch.as_tensor(g.kind, device=dev)] = 1.0
    e = torch.as_tensor(g.edge, device=dev).expand(b, -1, -1).clone()
    for j, name in enumerate(("PRV-1", "PRV-2", "PRV-3")):
        fwd, bwd, outlet = g.prv[name]
        s_ = settings[:, j] if j < 2 else torch.full((b,), PRV3_SETTING, device=dev)
        e[:, [fwd, bwd], 4] = ((s_ + float(g.elev[outlet]) - sc.mu) / sc.sd)[:, None]
    return x, e


PRV3_SETTING = 35.0  # m, L-TOWN.inp's setting of PRV-3 (Area B), never changed here


def observe(
    g: Graph, n: int, rng: np.random.Generator | None, hide: int | None = None
) -> np.ndarray:
    """Observed mask (n, N): the fixed nodes and the sensors, some hidden.

    ``rng`` hides each sensor with probability ``P_HIDE`` (training, validation); ``hide``
    hides one given sensor (leave one out); neither shows all 33.
    """
    obs = np.zeros((n, len(g.nodes)), dtype=bool)
    obs[:, g.fixed] = True
    obs[:, g.sensors] = True
    if rng is not None:
        obs[:, g.sensors] = rng.random((n, len(g.sensors))) >= P_HIDE
    if hide is not None:
        obs[:, hide] = False
    return obs


# ---------------------------------------------------------------- models


class MessagePassing(nn.Module):
    """One layer: every edge sends a message; each node sums what arrives and updates itself.

    h_i <- h_i + U(h_i, sum over edges j->i of M(h_j, h_i, e_ji), mean over nodes of h), with M
    and U small MLPs shared by every node and edge (Gilmer et al., 2017). The mean over nodes is
    the graph-level attribute of Battaglia et al. (2018): the town-wide state (how much water
    the town draws now) that sets every pipe's friction loss, which no K-link neighbourhood
    sees. ``global_mean=False`` drops it: a purely local layer.
    """

    def __init__(self, d: int, global_mean: bool = True) -> None:
        """Message and update networks of width ``d``."""
        super().__init__()
        self.global_mean = global_mean
        self.norm = nn.LayerNorm(d)  # pre-norm: keeps 12 residual layers stable
        self.message = nn.Sequential(nn.Linear(2 * d + EDGE_F, d), nn.ReLU(), nn.Linear(d, d))
        self.update = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """Node states h (B, N, d) and edge features e (B, E, EDGE_F) -> new states (B, N, d)."""
        z = self.norm(h)
        m = self.message(torch.cat([z[:, src], z[:, dst], e], dim=-1))  # (B, E, d)
        agg = torch.zeros_like(z).index_add_(1, dst, m)  # (B, N, d): sum of incoming messages
        u = z.mean(dim=1, keepdim=True).expand_as(z) if self.global_mean else torch.zeros_like(z)
        return h + self.update(torch.cat([z, agg, u], dim=-1))


class GNN(nn.Module):
    """Encode each node, pass messages ``layers`` times, decode a head per node."""

    def __init__(self, layers: int, d: int = HIDDEN, global_mean: bool = True) -> None:
        """A network with ``layers`` message-passing layers of width ``d``."""
        super().__init__()
        self.encode = nn.Linear(NODE_F, d)
        self.layers = nn.ModuleList(MessagePassing(d, global_mean) for _ in range(layers))
        self.decode = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, e: torch.Tensor, g: Graph) -> torch.Tensor:
        """Node features x (B, N, NODE_F), edge features e (B, E, EDGE_F) -> heads (B, N).

        The heads are standardized (``Scale``).
        """
        src = torch.as_tensor(g.src, device=x.device)
        dst = torch.as_tensor(g.dst, device=x.device)
        h = self.encode(x)
        for layer in self.layers:
            h = layer(h, e, src, dst)
        return self.decode(h)[..., 0]


class MLP(nn.Module):
    """No graph: the 33 readings in, every node's head out (Lecture 14's network, wider).

    Inputs: each sensor's reading and whether it is hidden, the tank's head, the PRV heads.
    """

    def __init__(self, g: Graph, d: int = MLP_HIDDEN) -> None:
        """Input and output sizes fixed by this network's sensors and nodes."""
        super().__init__()
        self.sensors = torch.as_tensor(g.sensors)
        self.tank = int(np.flatnonzero(g.kind == 2)[0])
        self.prv = [g.prv[k][0] for k in ("PRV-1", "PRV-2")]
        n_in = 2 * len(g.sensors) + 1 + 2
        self.net = nn.Sequential(
            nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, len(g.nodes))
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor, g: Graph) -> torch.Tensor:  # noqa: ARG002 (GNN's signature)
        """The same inputs as the GNN, read at fixed positions -> standardized heads (B, N)."""
        s = x[:, self.sensors.to(x.device), :2].flatten(1)  # (B, 66): value, observed
        z = torch.cat([s, x[:, self.tank, :1], e[:, self.prv, 4]], dim=1)  # (B, 69)
        return self.net(z)


def predict(
    model: nn.Module, g: Graph, sc: Scale, snaps: Snapshots, obs: np.ndarray, *, batch: int = 256
) -> np.ndarray:
    """Heads (n, N), m, for snapshots with the given observed masks."""
    model.eval()
    dev = next(model.parameters()).device
    heads = torch.as_tensor(snaps.heads, device=dev)
    settings = torch.as_tensor(snaps.settings, device=dev)
    observed = torch.as_tensor(obs, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(heads), batch):
            sl = slice(i, i + batch)
            x, e = inputs(g, sc, heads[sl], observed[sl], settings[sl])
            out.append(model(x, e, g).cpu().numpy() * sc.sd + sc.mu)
    return np.concatenate(out)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference, m."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train(
    model: nn.Module, g: Graph, sc: Scale, tr: Snapshots, va: Snapshots, *, seed: int, lr: float
) -> dict[str, object]:
    """Adam on squared error at the junctions, sensors hidden at random; early stopping.

    The validation snapshots hide sensors with a fixed draw, and the score is the head error
    at unobserved junctions; the rate halves after 3 epochs without improvement, and the best
    epoch's weights are kept. Gradients clipped at norm 1.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    heads = torch.as_tensor(tr.heads, device=DEVICE)
    settings = torch.as_tensor(tr.settings, device=DEVICE)
    va_obs = observe(g, len(va.heads), np.random.default_rng((SEED, 999)))
    junc = torch.as_tensor(g.junctions, device=DEVICE)
    best, best_state, curve, wait = np.inf, None, [], 0
    t0 = time.perf_counter()
    for _ in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(tr.heads))
        for i in range(0, len(order), BATCH):
            k = order[i : i + BATCH]
            kt = torch.as_tensor(k, device=DEVICE)
            obs = torch.as_tensor(observe(g, len(k), rng), device=DEVICE)
            x, e = inputs(g, sc, heads[kt], obs, settings[kt])
            y = (heads[kt] - sc.mu) / sc.sd
            loss = ((model(x, e, g) - y)[:, junc] ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = predict(model, g, sc, va, va_obs)
        unobs = ~va_obs & (g.kind == 0)[None]
        err = rmse(pred[unobs], va.heads[unobs])
        curve.append(err)
        plateau.step(err)
        if err < best:
            best, wait = err, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {
        "val_rmse": best,
        "epochs": len(curve),
        "best_epoch": int(np.argmin(curve)) + 1,
        "curve": curve,
        "seconds": time.perf_counter() - t0,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------- baselines


def zones(g: Graph) -> nx.Graph:
    """The graph without valves and the pump: its components are the pressure zones."""
    z = nx.Graph()
    z.add_nodes_from(range(len(g.nodes)))
    z.add_edges_from((a, b) for a, b, f in zip(g.src, g.dst, g.edge, strict=True) if f[1] == 1)
    return z


def nearest_observed(g: Graph, z: nx.Graph, observed: np.ndarray) -> np.ndarray:
    """Each node's estimate: the mean head of its nearest observed nodes in the same zone.

    Returns a matrix W (N, N) with ``heads @ W.T`` the estimates; a node with no observed
    node in its zone gets NaN.
    """
    obs = set(np.flatnonzero(observed).tolist())
    w = np.full((len(g.nodes), len(g.nodes)), np.nan)
    for comp in nx.connected_components(z):
        seeds = obs & comp
        if not seeds:
            continue
        dist = {s: nx.single_source_shortest_path_length(z, s) for s in seeds}
        for i in comp:
            d = {s: dist[s][i] for s in seeds}
            m = min(d.values())
            near = [s for s, v in d.items() if v == m]
            w[i] = 0
            w[i, near] = 1 / len(near)
    return w


# ---------------------------------------------------------------- checks of the derivation


def aggregation_check(g: Graph) -> dict[str, float]:
    """Summing incoming messages with index_add_ equals a product with the incidence matrix.

    B[i, k] = 1 if edge k ends at node i: sum_{k: dst(k)=i} m_k = (B m)_i.
    """
    rng = np.random.default_rng(SEED)
    m = torch.from_numpy(rng.normal(size=(len(g.src), 8))).double()  # (E, 8) messages
    ours = torch.zeros(len(g.nodes), 8, dtype=torch.float64).index_add_(
        0, torch.as_tensor(g.dst), m
    )
    inc = np.zeros((len(g.nodes), len(g.src)))
    inc[g.dst, np.arange(len(g.src))] = 1
    dense = torch.from_numpy(inc) @ m
    return {
        "max_abs_diff": float((ours - dense).abs().max()),
        "edges": len(g.src),
        "nodes": len(g.nodes),
    }


def equivariance_check(model: GNN, g: Graph, sc: Scale, snaps: Snapshots) -> float:
    """Relabel the nodes at random: the outputs must be relabelled the same way."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(g.nodes))  # new position -> old node
    inv = np.argsort(perm)
    pg = Graph(
        [g.nodes[i] for i in perm],
        g.kind[perm],
        g.elev[perm],
        inv[g.sensors],
        inv[g.src],
        inv[g.dst],
        g.edge,
        {k: (f, b, int(inv[o])) for k, (f, b, o) in g.prv.items()},
        g.link_of,
        g.hops[perm],
        g.xy[perm],
        g.area[perm],
    )
    obs = observe(g, 4, None)
    a = predict(model, g, sc, Snapshots(snaps.heads[:4], snaps.settings[:4], snaps.hour[:4]), obs)
    b = predict(
        model,
        pg,
        sc,
        Snapshots(snaps.heads[:4][:, perm], snaps.settings[:4], snaps.hour[:4]),
        obs[:, perm],
    )
    return float(np.abs(a[:, perm] - b).max())


def reach_check(model: GNN, g: Graph, sc: Scale, snaps: Snapshots, layers: int) -> dict:
    """Change one sensor's reading: only nodes within ``layers`` links may change."""
    s = int(g.sensors[0])
    obs = observe(g, 1, None)
    one = Snapshots(snaps.heads[:1].copy(), snaps.settings[:1], snaps.hour[:1])
    a = predict(model, g, sc, one, obs)
    one.heads[0, s] += 5.0
    b = predict(model, g, sc, one, obs)
    gr = nx.Graph()
    gr.add_edges_from(zip(g.src, g.dst, strict=True))
    d = nx.single_source_shortest_path_length(gr, s)
    dist = np.array([d.get(i, 10**6) for i in range(len(g.nodes))])
    moved = np.abs(b - a)[0] > 0
    return {
        "sensor": g.nodes[s],
        "layers": layers,
        "moved": int(moved.sum()),
        "within": int((dist <= layers).sum()),
        "farthest_moved": int(dist[moved].max()),
        "max_change_beyond": float(np.abs(b - a)[0][dist > layers].max()),
    }


# ---------------------------------------------------------------- real data and the model alone


def scada(g: Graph, year: int, days: int, every: int) -> tuple[pd.DatetimeIndex, Snapshots]:
    """BattLeDIM's measured heads: the sensors (pressure + elevation) and the tank.

    The first ``days`` days of the year, every ``every``-th 5-min reading. Reservoir heads
    are the file's; PRV-1 and PRV-2 at the file's settings. Unmeasured nodes are NaN.
    """
    read = {"sep": ";", "decimal": ",", "parse_dates": ["Timestamp"], "index_col": "Timestamp"}
    p = pd.read_csv(require(SCADA[year], "scripts/fetch_battledim.sh"), **read)
    lv = pd.read_csv(require(LEVELS[year], "scripts/fetch_battledim.sh"), **read)
    assert list(p.columns) == [g.nodes[i] for i in g.sensors]
    end = p.index[0] + pd.Timedelta(days=days)
    p = p[p.index < end].iloc[::every]
    lv = lv.loc[p.index]
    heads = np.full((len(p), len(g.nodes)), np.nan, dtype=np.float32)
    heads[:, g.sensors] = p.to_numpy() + g.elev[g.sensors]
    tank = int(np.flatnonzero(g.kind == 2)[0])
    heads[:, tank] = lv["T1"].to_numpy() + g.elev[tank]
    res = np.flatnonzero(g.kind == 1)
    heads[:, res] = g.elev[res]
    settings = np.tile(np.array(ltown.TODAY, dtype=np.float32), (len(p), 1))
    hour = (p.index.hour + p.index.minute / 60).to_numpy()
    return p.index, Snapshots(heads, settings, hour)


def nominal_week() -> np.ndarray:
    """The hydraulic model alone: heads (steps, N) over the file's week, every 5 min.

    The file's demand patterns, the PRVs at the file's settings, ``ltown``'s calibrated
    background leakage; no measurement enters.
    """
    town = ltown.LTown()
    opts = town.wn.options.time
    keep = opts.duration, opts.pattern_start
    try:
        opts.duration, opts.pattern_start = 7 * ltown.DAY_S, 0
        res = town._simulate(*ltown.TODAY)
    finally:
        opts.duration, opts.pattern_start = keep
    return res.node["head"][list(town.wn.node_name_list)].to_numpy()


def model_alone(week: np.ndarray, times: pd.DatetimeIndex, first_day: int) -> np.ndarray:
    """The nominal week's heads at the given times: pattern day = (weekday - first_day) mod 7."""
    day = (times.weekday.to_numpy() - first_day) % 7
    step = (times.hour.to_numpy() * 3600 + times.minute.to_numpy() * 60) // ltown.STEP_S
    return week[day * (ltown.DAY_S // ltown.STEP_S) + step]


def align_weekday(g: Graph, week: np.ndarray) -> dict[str, object]:
    """Which weekday the file's patterns start on, fitted on 2018's first eight weeks.

    The shift whose nominal sensor heads correlate best with 2018's measured ones; 2019, the
    test period, is not used.
    """
    times, real = scada(g, 2018, 56, 1)
    corr = []
    for k in range(7):
        sim = model_alone(week, times, k)[:, g.sensors]
        a = sim - sim.mean(0)
        b = real.heads[:, g.sensors] - real.heads[:, g.sensors].mean(0)
        corr.append(float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())))
    return {"first_day": int(np.argmax(corr)), "corr": corr}


def leave_one_out(
    g: Graph,
    sc: Scale,
    snaps: Snapshots,
    models: dict[str, nn.Module],
    z: nx.Graph,
    *,
    alone: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """Hide each sensor in turn; estimate it from the other 32. Errors (m) per method.

    Returns, per method: each sensor's RMSE over all snapshots (None where the method gives no
    estimate: the nearest-in-zone rule has none for a sensor alone in its zone), their median,
    the RMSE over every estimate, and how many sensors got none.
    """
    out: dict[str, list[np.ndarray]] = {k: [] for k in [*models, "nearest in zone"]}
    if alone is not None:
        out["model alone"] = []
    for s in g.sensors:
        obs = observe(g, len(snaps.heads), None, hide=int(s))
        truth = snaps.heads[:, s]
        for name, m in models.items():
            out[name].append(predict(m, g, sc, snaps, obs)[:, s] - truth)
        w = nearest_observed(g, z, obs[0])[s]  # NaN row: no observed node in the zone
        est = np.nan_to_num(snaps.heads) @ np.nan_to_num(w) if np.isfinite(w).all() else np.nan
        out["nearest in zone"].append(est - truth)
        if alone is not None:
            out["model alone"].append(alone[:, s] - truth)
    res = {}
    for k, v in out.items():
        per = [rmse(e, 0) if np.isfinite(e).all() else None for e in v]
        ok = [e for e in v if np.isfinite(e).all()]
        res[k] = {
            "rmse": rmse(np.stack(ok), 0),
            "per_sensor": per,
            "median": float(np.median([p for p in per if p is not None])),
            "no_estimate": sum(p is None for p in per),
        }
    return res


def pick_closure(
    g: Graph, wn: wntr.network.WaterNetworkModel, scs: list[Scenario], open_heads: np.ndarray
) -> dict[str, object]:
    """The pipe whose closing changes the heads most, among Area A's busiest non-bridge pipes.

    Candidates: the 10 pipes inside Area A with the largest mean flow on a nominal day whose
    removal leaves the network connected. Each is closed on the given validation days; the one
    that moves junction heads most (mean absolute change over those days) is the test's
    closed pipe. The test days are not used.
    """
    town = ltown.LTown()
    flow = town._simulate(*ltown.TODAY).link["flowrate"].abs().mean()
    area_a = set(town.area_a)
    gr = nx.Graph()
    gr.add_edges_from(zip(g.src, g.dst, strict=True))
    bridges = {frozenset(e) for e in nx.bridges(gr)}
    index = {n: i for i, n in enumerate(g.nodes)}
    cand = [
        n
        for n in flow.sort_values(ascending=False).index
        if n in wn.pipe_name_list
        and {wn.get_link(n).start_node_name, wn.get_link(n).end_node_name} <= area_a
        and frozenset((index[wn.get_link(n).start_node_name], index[wn.get_link(n).end_node_name]))
        not in bridges
    ][:10]
    closed = simulate_many(
        [Scenario(s.day, s.s1, s.s2, s.scale, s.seed, c) for c in cand for s in scs]
    ).reshape(len(cand), len(scs), *open_heads.shape[1:])  # (candidates, days, steps, N)
    change = np.abs(closed - open_heads[None])[..., g.junctions].mean(axis=(1, 2, 3))
    best = int(np.argmax(change))
    return {
        "pipe": cand[best],
        "candidates": cand,
        "mean_change": change.tolist(),
        "diameter_mm": wn.get_link(cand[best]).diameter * 1000,
        "length_m": wn.get_link(cand[best]).length,
    }


# ---------------------------------------------------------------- figures


def _pipes(ax: plt.Axes, g: Graph, color: str = "#c9c9c9", lw: float = 0.6) -> None:
    """Every link as a grey segment."""
    seg = [(g.xy[a], g.xy[b]) for a, b in zip(g.src[::2], g.dst[::2], strict=True)]
    ax.add_collection(LineCollection(seg, colors=color, linewidths=lw, zorder=1))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()


def picture(g: Graph, out_dir: Path) -> dict[str, int]:
    """The one picture, as frames: the sensors; what 2 layers reach; what 12 layers reach.

    A junction is reached after k layers when it is within k links of a visible sensor:
    only then can a sensor's reading enter its estimate.
    """
    reach = {}
    frames = ((1, None), (2, 2), (3, LAYERS))
    for frame, k in frames:
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        _pipes(ax, g)
        j = g.junctions
        if k is not None:
            near = j[(g.hops[j] <= k) & (g.hops[j] >= 0)]
            reach[str(k)] = len(near)
            ax.scatter(
                *g.xy[near].T,
                s=5,
                color=aml_style.PALETTE[0],
                zorder=2,
                label=f"within {k} links of a sensor: {len(near)} of {len(j)}",
            )
        else:  # an empty entry, so all three frames have the same legend and size
            ax.scatter([], [], s=5, color="none", label=" ")
        ax.scatter(
            *g.xy[g.sensors].T,
            s=60,
            marker="*",
            color=aml_style.PALETTE[1],
            zorder=3,
            label=f"{len(g.sensors)} pressure sensors",
        )
        ax.scatter(
            *g.xy[g.fixed].T,
            s=50,
            marker="s",
            color=aml_style.NAVY,
            zorder=3,
            label="reservoirs and tank: heads known",
        )
        # Below the map, one row: inside, it would cover the south-west junctions.
        ax.legend(
            frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3
        )
        aml_style.save(fig, out_dir, f"picture_{frame}")
    return reach


def neighborhood_figure(g: Graph, heads: np.ndarray, j: int, out_dir: Path, radius: int = 3) -> int:
    """The attempt's junction and everything within ``radius`` links, coloured by pressure.

    Arrows mark the messages that reach the junction in one layer (from its neighbours). The
    junction and the sensors are labelled with head and elevation. Returns the nodes drawn.
    """
    gr = nx.Graph()
    gr.add_edges_from(zip(g.src, g.dst, strict=True))
    near = nx.single_source_shortest_path_length(gr, j, cutoff=radius)
    nodes = list(near)
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    seg = [(g.xy[a], g.xy[b]) for a, b in gr.subgraph(nodes).edges]
    ax.add_collection(LineCollection(seg, colors="#9a9a9a", linewidths=1.5, zorder=1))
    pts = ax.scatter(
        *g.xy[nodes].T,
        c=heads[nodes] - g.elev[nodes],
        cmap="viridis",
        s=70,
        zorder=2,
        edgecolors="white",
        linewidths=0.8,
    )
    fig.colorbar(pts, ax=ax, shrink=0.8, label="pressure, m (head - elevation)")
    for nb in gr.neighbors(j):
        ax.annotate(
            "",
            xy=g.xy[j],
            xytext=g.xy[nb],
            zorder=3,
            arrowprops={"arrowstyle": "-|>", "color": aml_style.PALETTE[1], "lw": 2},
        )
    sens = [n for n in nodes if n in set(g.sensors.tolist())]
    ax.scatter(*g.xy[sens].T, s=260, marker="*", color=aml_style.PALETTE[1], zorder=4)
    for n in [j, *sens]:
        ax.annotate(
            f"{g.nodes[n]}\nhead {heads[n]:.2f} m\nelevation {g.elev[n]:.2f} m",
            g.xy[n],
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            path_effects=aml_style.HALO,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.5 * (x1 - x0))  # room for the labels right of the nodes
    aml_style.save(fig, out_dir, "neighborhood")
    return len(nodes)


def hops_figure(by_hops: dict[str, list[float]], counts: list[int], out_dir: Path) -> None:
    """Test error at unobserved junctions against the links to the nearest sensor."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    styles = {
        "nearest in zone": (aml_style.MUTED, "s", "--"),
        "MLP (no graph)": (aml_style.PALETTE[1], "D", "-."),
        "GNN, 2 layers": (aml_style.PALETTE[3], "v", ":"),
        "GNN, 5 layers": (aml_style.PALETTE[2], "^", ":"),
        f"GNN, {LAYERS} layers": (aml_style.PALETTE[0], "o", "-"),
    }
    hops = np.arange(1, len(counts))
    for name, (col, mk, ls) in styles.items():
        ax.plot(hops, by_hops[name][1:], color=col, marker=mk, ls=ls, lw=2, label=name)
    for k in FEW_LAYERS:
        ax.axvline(k + 0.5, color=aml_style.MUTED, lw=0.8, ls=":")
    ax.set_xticks(hops, [f"{h}\n({c})" for h, c in zip(hops, counts[1:], strict=True)], fontsize=9)
    ax.set_xlabel("links to the nearest sensor (junctions)")
    ax.set_ylabel("head error, RMSE (m)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    aml_style.save(fig, out_dir, "hops")


def error_map(g: Graph, err: np.ndarray, out_dir: Path) -> None:
    """Each unobserved junction's test RMSE on the map: where the estimate is weakest."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    _pipes(ax, g)
    j = np.setdiff1d(g.junctions, g.sensors)
    sc_ = ax.scatter(
        *g.xy[j].T,
        c=err[j],
        s=8,
        cmap="magma_r",
        vmin=0,
        vmax=float(np.percentile(err[j], 99)),
        zorder=2,
    )
    ax.scatter(*g.xy[g.sensors].T, s=50, marker="*", color=aml_style.PALETTE[0], zorder=3)
    fig.colorbar(  # the arrow marks the 1% of junctions above the colour scale
        sc_, ax=ax, shrink=0.8, extend="max", label=f"GNN ({LAYERS} layers) head RMSE, m"
    )
    aml_style.save(fig, out_dir, "error_map")


def real_figure(real: dict[str, dict], out_dir: Path) -> None:
    """Each hidden sensor's error on 2019's readings, per method, with the median."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    names = ["model alone", "nearest in zone", "MLP (no graph)", f"GNN, {LAYERS} layers"]
    rng = np.random.default_rng(SEED)
    for i, name in enumerate(names):
        per = np.array([p for p in real[name]["per_sensor"] if p is not None])
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, len(per)),
            per,
            s=18,
            alpha=0.7,
            color=aml_style.PALETTE[0],
            zorder=2,
        )
        med = real[name]["median"]
        ax.plot([i - 0.3, i + 0.3], [med, med], color=aml_style.PALETTE[1], lw=3, zorder=3)
        ax.text(i + 0.33, med, f"{med:.2f}", va="center", fontsize=10)
    labels = [
        n
        if not real[n]["no_estimate"]
        else f"{n}\n({real[n]['no_estimate']} sensor"
        f"{'s' if real[n]['no_estimate'] > 1 else ''}: no estimate)"
        for n in names
    ]
    ax.set_xticks(range(len(names)), labels, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("hidden sensor's head error,\nRMSE over 4 weeks (m)")
    ax.set_xlim(-0.5, len(names) - 0.2)
    aml_style.save(fig, out_dir, "real")


def title_art_figure(out_dir: Path) -> None:
    """The title art: a lattice of pipes, a few lit nodes spreading light along the links."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0d14", "#0f1a20", "#14121c", "#08070b"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.05, 0.05, 0.50, 0.36, 4), soft=4), "#2c3a2a", 0.8)
    c.glaze(c.polygon(title_art.stepped(0.52, 0.05, 0.43, 0.24, 3), soft=4), "#1f2a24", 0.8)
    rng = np.random.default_rng(SEED)
    pts = [(0.12 + 0.11 * i, top * (0.45 + 0.13 * j)) for i in range(7) for j in range(3)]
    links = [[pts[k], pts[k + 3]] for k in range(len(pts) - 3)]
    links += [[pts[k], pts[k + 1]] for k in range(len(pts) - 1) if (k + 1) % 3]
    c.glow(c.lines(links, 1.0, soft=1.0), "#7fa3b8", 0.25, spread=4)
    for k in rng.choice(len(pts), 3, replace=False):
        c.glow(c.disc(pts[k], 0.03, soft=3), "#eaa548", 0.55, spread=20)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: low stepped green forms at the bottom; above them a "
        "faint blue lattice of pipes joining small nodes, three of which glow golden, the "
        "sensors whose light spreads along the links.",
    )


# ---------------------------------------------------------------- evaluation


def fit_best(
    make: object,
    g: Graph,
    sc: Scale,
    tr: Snapshots,
    va: Snapshots,
    *,
    seed: int,
    rates: tuple[float, ...],
) -> tuple[nn.Module, dict[str, object]]:
    """Train one fresh model per rate; keep the one with the lowest validation error.

    Every model starts from ``rates``; whenever the best rate sits at an edge of those tried,
    the next rate of ``LR_LADDER`` past that edge is tried too, so the kept rate is interior.
    The same protocol for every model (Lecture 13: tune both sides alike).
    """
    best = None
    runs = []  # every rate's run, kept for the record (the slides cite the others too)
    todo = list(rates)
    while todo:
        lr = todo.pop(0)
        model = make()
        info = train(model, g, sc, tr, va, seed=seed, lr=lr) | {"lr": lr}
        runs.append({k: info[k] for k in ("lr", "val_rmse", "epochs", "best_epoch")})
        log(
            f"  {type(model).__name__} lr {lr:g}: val {info['val_rmse']:.3f} m, "
            f"{info['epochs']} epochs (best {info['best_epoch']}), {info['seconds']:.0f} s"
        )
        if best is None or info["val_rmse"] < best[1]["val_rmse"]:
            best = (model, info)
        tried = sorted(r["lr"] for r in runs)
        i = LR_LADDER.index(best[1]["lr"])
        if not todo and best[1]["lr"] in (tried[0], tried[-1]):
            step = -1 if best[1]["lr"] == tried[0] else 1
            assert 0 <= i + step < len(LR_LADDER), (
                f"the best rate {best[1]['lr']:g} ends the ladder"
            )
            todo.append(LR_LADDER[i + step])
    tried = sorted(r["lr"] for r in runs)
    assert tried[0] < best[1]["lr"] < tried[-1], (best[1]["lr"], tried)  # interior
    if REQUIRE_CONVERGED:  # the kept run's best epoch must not be the last allowed
        assert best[1]["best_epoch"] < MAX_EPOCHS or best[1]["epochs"] < MAX_EPOCHS, best[1]
    return best[0], best[1] | {"runs": sorted(runs, key=lambda r: r["lr"])}


def scores(g: Graph, pred: np.ndarray, truth: np.ndarray, obs: np.ndarray) -> dict[str, object]:
    """Head error at unobserved junctions, by links to the nearest sensor, and low-pressure flags.

    A flag: estimated pressure (head - elevation) below 35 psi (24.6 m) at an unobserved
    junction in a snapshot; recall and precision against the simulated truth.
    """
    unobs = ~obs & (g.kind == 0)[None]
    err = pred - truth
    by_hops = [  # None where no unobserved junction is that far (0 links: the sensors)
        rmse(err[unobs & (g.hops == h)[None]], 0) if (unobs & (g.hops == h)[None]).any() else None
        for h in range(int(g.hops.max()) + 1)
    ]
    by_area = {  # the three pressure zones: A (PRV-1/2), B (PRV-3), C (behind the pump)
        name: rmse(err[unobs & (g.area == i)[None]], 0) for i, name in enumerate("ABC")
    }
    low_true = (truth - g.elev) < ltown.P_SERVICE
    low_pred = (pred - g.elev) < ltown.P_SERVICE
    tp = (low_true & low_pred & unobs).sum()
    by_area_low = {}  # per zone: the junction-hours truly below 35 psi, and the share flagged
    for i, name in enumerate("ABC"):
        low = low_true & unobs & (g.area == i)[None]
        n = int(low.sum())
        by_area_low[name] = {"low": n, "found": float((low & low_pred).sum() / n) if n else None}
    return {
        "rmse": rmse(err[unobs], 0),
        "by_hops": by_hops,
        "by_area": by_area,
        "low_share": float(low_true[unobs].mean()),
        "low_recall": float(tp / max(1, (low_true & unobs).sum())),
        "low_precision": float(tp / max(1, (low_pred & unobs).sum())),
        "flags": int((low_pred & unobs).sum()),
        "by_area_low": by_area_low,
    }


def area_c_spread(g: Graph, heads: np.ndarray) -> dict[str, float]:
    """Area C (behind the pump): how far its heads spread in one hour, and over the hours, m."""
    hc = heads[:, g.area == 2]
    return {
        "within_sd": float(hc.std(1).mean()),  # across its junctions in one hour, mean over hours
        "over_time_sd": float(hc.std(0).mean()),  # over the hours, mean over its junctions
    }


def pipe_drop(g: Graph, heads: np.ndarray) -> float:
    """Mean head drop along a pipe, m, over pipes and snapshots (the friction losses)."""
    p = g.edge[:, 1] == 1  # pipes only: valves and the pump hold or add head
    return float(np.abs(heads[:, g.src[p]] - heads[:, g.dst[p]]).mean())


def nearest_predict(g: Graph, z: nx.Graph, snaps: Snapshots, obs: np.ndarray) -> np.ndarray:
    """The no-learning baseline for snapshots that share one observed mask."""
    assert (obs == obs[0]).all()
    w = np.nan_to_num(nearest_observed(g, z, obs[0]))
    return np.nan_to_num(np.where(obs, snaps.heads, 0.0)) @ w.T


def attempt(g: Graph, snaps: Snapshots, est: dict[str, np.ndarray], t: int) -> dict:
    """The attempt: one unmeasured junction 3 links from its nearest sensors, at one snapshot.

    Drawn at random (seed ``SEED``) among junctions exactly 3 links from a sensor; the table
    gives its elevation and the nearest sensors' heads; the answers come later.
    """
    rng = np.random.default_rng(SEED)
    cand = np.flatnonzero((g.kind == 0) & (g.hops == 3))
    j = int(rng.choice(cand))
    gr = nx.Graph()
    gr.add_edges_from(zip(g.src, g.dst, strict=True))
    d = nx.single_source_shortest_path_length(gr, j)
    near = sorted((d[int(s)], int(s)) for s in g.sensors if int(s) in d)[:2]
    return {
        "junction": g.nodes[j],
        "elevation": float(g.elev[j]),
        "hour": float(snaps.hour[t]),
        "true_head": float(snaps.heads[t, j]),
        "true_pressure": float(snaps.heads[t, j] - g.elev[j]),
        "sensors": [
            {
                "name": g.nodes[s],
                "links": int(k),
                "head": float(snaps.heads[t, s]),
                "elevation": float(g.elev[s]),
                "pressure": float(snaps.heads[t, s] - g.elev[s]),
            }
            for k, s in near
        ],
        "estimates": {k: float(v[t, j] - g.elev[j]) for k, v in est.items()},
    }


def simulate_days() -> tuple[dict[str, list[Scenario]], dict[str, Snapshots], int]:
    """Training, validation, test, and beyond-training days, simulated; snapshots per day."""
    rng = np.random.default_rng((SEED, 0))
    scs = {k: draw_scenarios(n, rng, SCALE) for k, n in SCENARIOS.items()}
    scs["far"] = draw_scenarios(SCENARIOS["test"], np.random.default_rng((SEED, 1)), FAR_SCALE)
    heads = {k: simulate_many(v) for k, v in scs.items()}
    return scs, {k: Snapshots.of(scs[k], heads[k]) for k in scs}, int(heads["test"].shape[1])


def train_all(g: Graph, sc: Scale, tr: Snapshots, va: Snapshots) -> tuple[dict, dict]:
    """Train every model; each chooses its rate in ``LRS`` on the validation days.

    The same protocol for all: the GNNs of 12, 5, and 2 layers and the MLP.
    """
    makes = {f"GNN, {k} layers": (lambda k=k: GNN(k)) for k in (LAYERS, *FEW_LAYERS)}
    makes["MLP (no graph)"] = lambda: MLP(g)
    models, info = {}, {}
    for name, make in makes.items():
        models[name], info[name] = fit_best(make, g, sc, tr, va, seed=SEED, rates=LRS)
    return models, info


def closure_test(
    g: Graph,
    wn: wntr.network.WaterNetworkModel,
    scs: dict,
    snaps: dict,
    models: dict,
    *,
    obs: np.ndarray,
) -> dict[str, object]:
    """The test days again with one pipe closed: the GNN told (the new graph) or not."""
    days = scs["val"][:3]
    closure = pick_closure(g, wn, days, simulate_many(days))
    shut = [Scenario(s.day, s.s1, s.s2, s.scale, s.seed, closure["pipe"]) for s in scs["test"]]
    cl = Snapshots.of(shut, simulate_many(shut))
    g_cut = g.without(closure["pipe"])
    gnn = models[f"GNN, {LAYERS} layers"]
    sc = models["scale"]
    return closure | {
        "GNN, told": scores(g, predict(gnn, g_cut, sc, cl, obs), cl.heads, obs),
        "GNN, not told": scores(g, predict(gnn, g, sc, cl, obs), cl.heads, obs),
        "MLP (no graph)": scores(
            g, predict(models["MLP (no graph)"], g, sc, cl, obs), cl.heads, obs
        ),
        "nearest in zone": scores(g, nearest_predict(g, zones(g_cut), cl, obs), cl.heads, obs),
        "head_change": rmse(cl.heads[:, g.junctions], snaps["test"].heads[:, g.junctions]),
    }


def real_test(g: Graph, sc: Scale, te: Snapshots, models: dict, z: nx.Graph) -> dict:
    """Hide each sensor in turn, on simulated test days and on 2019's measurements."""
    week = nominal_week()
    wk = align_weekday(g, week)
    times, real = scada(g, 2019, REAL_DAYS, 12)
    pick = {k: models[k] for k in ("MLP (no graph)", f"GNN, {LAYERS} layers")}
    sim = leave_one_out(g, sc, Snapshots(te.heads[::4], te.settings[::4], te.hour[::4]), pick, z)
    real_loo = leave_one_out(g, sc, real, pick, z, alone=model_alone(week, times, wk["first_day"]))
    return {
        "weekday": wk,
        "sim": sim,
        "real": real_loo,
        "real_snapshots": len(times),
        "real_start": str(times[0]),
        "real_end": str(times[-1]),
    }


def main() -> None:
    """Generate every figure and number for CEE Lecture 21."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.perf_counter()
    wn = wntr.network.WaterNetworkModel(str(require(ltown.INP, "scripts/fetch_battledim.sh")))
    g = build_graph(wn)
    z = zones(g)
    j = g.junctions
    numbers: dict[str, object] = {
        "graph": {
            "junctions": len(j),
            "sensors": len(g.sensors),
            "links": len(g.src) // 2,
            "zones": nx.number_connected_components(z),
            "zone_junctions": sorted(
                (sum(int(g.kind[n] == 0) for n in c) for c in nx.connected_components(z)),
                reverse=True,
            ),
            "hops_max": int(g.hops.max()),
            "hops_median": float(np.median(g.hops[j])),
            "hops_count": np.bincount(g.hops[j]).tolist(),
            "area_junctions": {a: int((g.area == i).sum()) for i, a in enumerate("ABC")},
            "area_sensors": {a: int((g.area[g.sensors] == i).sum()) for i, a in enumerate("ABC")},
            "area_hops_median": {
                a: float(np.median(g.hops[g.area == i])) for i, a in enumerate("ABC")
            },
        }
    }
    scs, snaps, per_day = simulate_days()
    tr, va, te = snaps["train"], snaps["val"], snaps["test"]
    sc = Scale(float(tr.heads[:, j].mean()), float(tr.heads[:, j].std()))
    numbers["data"] = {
        "snapshots": {k: len(v.heads) for k, v in snaps.items()},
        "per_day": per_day,
        "mu": sc.mu,
        "sd": sc.sd,
    }
    numbers["aggregation_check"] = aggregation_check(g)
    log(f"simulated in {time.perf_counter() - t0:.0f} s")

    models, numbers["training"] = train_all(g, sc, tr, va)
    gnn = models[f"GNN, {LAYERS} layers"]
    log(f"trained in {time.perf_counter() - t0:.0f} s")
    numbers["equivariance_max_diff"] = equivariance_check(gnn, g, sc, te)
    torch.manual_seed(SEED)
    local = GNN(FEW_LAYERS[0], global_mean=False).to(DEVICE)  # untrained: locality is structural
    # Purely local layers: nothing beyond k links can move, whatever the weights. The trained
    # networks add the graph-level mean, which carries a town-wide average everywhere.
    numbers["reach"] = {"local": reach_check(local, g, sc, te, FEW_LAYERS[0])} | {
        str(k): reach_check(models[f"GNN, {k} layers"], g, sc, te, k) for k in FEW_LAYERS
    }

    obs = observe(g, len(te.heads), None)
    preds = {name: predict(m, g, sc, te, obs) for name, m in models.items()}
    preds["nearest in zone"] = nearest_predict(g, z, te, obs)
    preds["training mean"] = np.broadcast_to(tr.heads.mean(0), te.heads.shape)
    numbers["test"] = {k: scores(g, v, te.heads, obs) for k, v in preds.items()}
    numbers["area_c"] = area_c_spread(g, te.heads)
    t_peak = int(np.argmin(te.heads[:per_day, j].mean(1)))  # the first test day's lowest head
    numbers["attempt"] = attempt(
        g,
        te,
        {k: preds[k] for k in ("nearest in zone", "MLP (no graph)", f"GNN, {LAYERS} layers")},
        t_peak,
    )
    jt = g.nodes.index(numbers["attempt"]["junction"])
    numbers["neighborhood_nodes"] = neighborhood_figure(g, te.heads[t_peak], jt, out_dir)
    error_map(g, np.sqrt(((preds[f"GNN, {LAYERS} layers"] - te.heads) ** 2).mean(0)), out_dir)
    hops_figure(
        {k: v["by_hops"] for k, v in numbers["test"].items()},
        numbers["graph"]["hops_count"],
        out_dir,
    )

    numbers["closure"] = closure_test(g, wn, scs, snaps, models | {"scale": sc}, obs=obs)
    fobs = observe(g, len(snaps["far"].heads), None)
    numbers["far"] = {
        name: scores(g, predict(m, g, sc, snaps["far"], fobs), snaps["far"].heads, fobs)
        for name, m in models.items()
    }
    numbers["far"]["nearest in zone"] = scores(
        g, nearest_predict(g, z, snaps["far"], fobs), snaps["far"].heads, fobs
    )
    numbers["pipe_drop"] = {k: pipe_drop(g, snaps[k].heads) for k in ("test", "far")}
    log(f"break-it tests in {time.perf_counter() - t0:.0f} s")

    numbers["loo"] = loo = real_test(g, sc, te, models, z)
    real_figure(loo["real"], out_dir)
    numbers["picture_reach"] = picture(g, out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_3", "hops", "error_map", "real"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    log(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
