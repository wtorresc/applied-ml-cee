"""Applied Machine Learning for CEE — Lecture 21: Graph Neural Networks.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l21_graph_networks.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_battledim.sh`` once (BattLeDIM, Vrachimis et al., 2022, CC BY 4.0):
L-Town's EPANET model, its 2018 leak flows, and its 2018/2019 SCADA pressures and tank levels.

Set ``L21_SETTING=reduced`` (the default, for a laptop CPU: 10 to 47 minutes, see below) or
``L21_SETTING=slides`` (the slides' training, about two hours on an RTX 3090 Ti); see the header
cell.
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
    # Applied Machine Learning for CEE — Lecture 21: Graph Neural Networks

    **Slides:** *Applied Machine Learning for CEE*, Lecture 21 (`lectures/cee_l21_graph_networks`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §17.1.2 (the receptive field of stacked convolutional layers), §14.1.1–14.1.2 (inference on a chain or a tree by passing costs from vertex to vertex), §15.1.2 and §15.3 (Markov random fields, and mean-field updates of each node from its neighbours), §16.3 (layers and backpropagation), and §18.1.9 (residual networks). Graph networks themselves: Gilmer et al. (2017), Battaglia et al. (2018), Kipf & Welling (2017).

    **Data:** BattLeDIM's L-Town (Vrachimis et al., 2022; CC BY 4.0): the EPANET model `L-TOWN.inp`, the 2018 leak flows (they calibrate the background leakage, as in Lectures 12 and 14), and the 2018/2019 SCADA readings of 33 pressure sensors and the tank level, every 5 minutes. Simulated with WNTR (Klise et al., 2017) and EPANET (Rossman, 2000). Fetch with `bash scripts/fetch_battledim.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. It simulates 390
    days of L-Town in parallel processes (under a minute on 32 cores; EPANET is not thread-safe)
    and trains four networks. Each searches its Adam learning rate the same way: start from 3e-4
    and 1e-3, and while the best on the validation days is at an edge of the rates tried, add
    the next rate past that edge, so the rate kept is inside the range tried.

    **Two settings, whatever the device** (the environment variable `L21_SETTING`, or the line
    that reads it in the next cell):

    - `reduced` (the default) runs on a laptop CPU or a free Colab T4: 10 minutes on the CPU of
      the slides' machine (56 threads, no GPU, November 20, 2026) and 47 minutes on four of its
      cores (November 19), with up to 4 GB of memory. The same simulated days, but the
      networks learn from 20 of the 240 training days for at most 8 epochs, and each search
      stops after 3 rates. Every number that does not depend on training
      is checked against the slides exactly (the graph, the simulated days, the baselines without
      learning, the hydraulic model alone, and the checks of the derivation); the trained
      networks' numbers are printed, not checked, and are far from the slides'.
    - `slides` repeats the slides' training: about 2 hours on an RTX 3090 Ti (2.3 hours when the
      GPU is shared), many hours on a CPU. On that GPU it reproduces the slides' numbers and
      checks them; on another GPU the training may take another path and those checks may fail.

    **Your tasks** (from the slides): estimate the pressure at junction n213 from two sensors;
    count what $k$ layers can reach; write a message-passing layer with `index_add_` and check
    it against the incidence matrix, under relabelling, and for its reach; train the graph network
    and an MLP on the same simulated days and compare them with the nearest sensor in the zone;
    break them (too few layers, a closed pipe, more demand than trained, 2019's real readings);
    then decide who runs the low-pressure flags.
    """)
    return


@app.cell
def _():
    import contextlib
    import json
    import os
    import random
    import tempfile
    import time
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
    from matplotlib.collections import LineCollection
    from matplotlib.patheffects import withStroke
    from threadpoolctl import threadpool_limits
    from torch import nn

    SEED = 21
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    # As the slides' figure script: deterministic kernels wherever PyTorch has one
    torch.use_deterministic_algorithms(True, warn_only=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"

    # The slides' constants (lectures/figures/cee_l21_graph_networks.py)
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
    LR_LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # the search widens along it
    MAX_EPOCHS = 200
    PATIENCE = 15  # epochs without a better validation error before stopping
    REAL_DAYS = 28  # the first four weeks of 2019, hourly
    NODE_F = 6  # observed head, observed flag, elevation, junction / reservoir / tank
    EDGE_F = 7  # log resistance, pipe, PRV forward / backward, PRV head, pump forward / backward
    WORKERS = min(32, os.cpu_count() or 1)

    # L-Town (lectures/figures/ltown.py, Lectures 12 and 14)
    TODAY = (40.0, 50.0)  # PRV-1, PRV-2 settings in L-TOWN.inp, m
    BOX = (25.0, 60.0)  # the box of settings Lectures 12 and 14 search, m
    PSI_M = 6894.757 / (1000.0 * 9.80665)  # metres of water per psi (0.7031)
    P_SERVICE = 35 * PSI_M  # 35 psi normal working pressure (Ten States Standards), m
    STEP_S, DAY_S = 300, 86_400  # the file's hydraulic and pattern step; one day, s
    N1 = 1.0  # leakage exponent: background leakage proportional to pressure
    SOLVER_WARNINGS = ("hydraulically unbalanced", "hydraulically unstable")

    # The setting, chosen here or by the environment, never by the device: "reduced" (a laptop
    # CPU or a free Colab T4; 10 to 47 minutes, see the header) or "slides" (the slides'
    # training, 2 hours on an RTX 3090 Ti). Reduced: the same simulated days, fewer to learn
    # from, fewer epochs, and a search that stops after MAX_RATES rates.
    L21_SETTING = os.environ.get("L21_SETTING", "reduced").strip().lower()
    if L21_SETTING not in {"slides", "reduced"}:
        raise ValueError(f"L21_SETTING must be 'slides' or 'reduced', not {L21_SETTING!r}")
    SLIDES = L21_SETTING == "slides"
    TRAIN_USED = SCENARIOS["train"] if SLIDES else 20  # training days the networks learn from
    VAL_USED = SCENARIOS["val"] if SLIDES else 5  # validation days for early stopping
    EPOCHS = MAX_EPOCHS if SLIDES else 8
    MAX_RATES = None if SLIDES else 3  # None: widen until the best rate is inside the range
    CHECK_TRAINED = SLIDES  # the trained networks' numbers are checked in the slides' setting

    # Tolerances. The graph, the simulated days (EPANET is deterministic), and every estimate
    # without learning are exact, up to the order of floating-point sums.
    EXACT = 1e-9
    # The networks were trained on a GPU (an RTX 3090 Ti) with deterministic kernels, where this
    # notebook repeats the slides' training. Their errors are checked to half a unit of the
    # slide's last digit (0.005 m for errors shown as 0.19 m) and, digit by digit, against the
    # slides; on another GPU the training may take another path, and these checks may fail.
    NET_TOL = 0.005  # m
    PCT_TOL = 0.005  # shares of flags (0.5 percentage points)

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    MUTED = "#595959"
    NAVY = "#1B3A5C"
    HALO = [withStroke(linewidth=3, foreground="white")]

    def shown(x: float, text: str) -> bool:
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("0.19", "8e-06").

        Args:
            x: The unrounded value.
            text: The slide's digits, fixed-point or in e-notation.

        Returns:
            True if ``x`` rounds to ``text``.
        """
        if "e" in text:
            mantissa = text.split("e", maxsplit=1)[0]
            digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
            return f"{x:.{digits}e}" == text
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    def find_root() -> Path:
        """Locate the repository from its root, ``notebooks/``, or ``ipynb/`` (Jupyter's cwd).

        Returns:
            The directory that holds ``data/raw/battledim/L-TOWN.inp``.

        Raises:
            FileNotFoundError: If the data are missing (run ``bash scripts/fetch_battledim.sh``).
        """
        rel = Path("data") / "raw" / "battledim" / "L-TOWN.inp"
        bases = [Path.cwd(), Path.cwd().parent]
        if "__file__" in globals():
            bases.insert(0, Path(__file__).resolve().parents[1])
        for base in bases:
            if (base / rel).exists():
                return base
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_battledim.sh`")

    ROOT = find_root()
    RAW = ROOT / "data" / "raw" / "battledim"
    print(f"device: {DEVICE}; setting: {L21_SETTING}")
    if SLIDES and not GPU:
        print("the slides setting without a GPU: the training takes many hours")
    return (
        BATCH,
        BOX,
        CHECK_TRAINED,
        DAY_S,
        DEVICE,
        EDGE_F,
        EPOCHS,
        EXACT,
        FAR_SCALE,
        FEW_LAYERS,
        HALO,
        HIDDEN,
        JobParallel,
        LAYERS,
        LRS,
        LR_LADDER,
        LineCollection,
        MAX_EPOCHS,
        MAX_RATES,
        MLP_HIDDEN,
        MUTED,
        N1,
        NAVY,
        NET_TOL,
        NODE_F,
        NOISE,
        PALETTE,
        PATIENCE,
        PCT_TOL,
        PSI_M,
        P_HIDE,
        P_SERVICE,
        RAW,
        ROOT,
        REAL_DAYS,
        SCALE,
        SCENARIOS,
        SEED,
        SLIDES,
        SOLVER_WARNINGS,
        STEP_S,
        STRIDE,
        TODAY,
        TRAIN_USED,
        VAL_USED,
        WORKERS,
        contextlib,
        dataclass,
        delayed,
        json,
        nn,
        np,
        nx,
        pairwise,
        pd,
        plt,
        shown,
        tempfile,
        threadpool_limits,
        time,
        torch,
        wntr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 12–20

    On your own, no notes, two minutes:

    1. Lecture 12: what pressure must every L-Town customer keep, in psi and in metres of water?
    2. Lecture 16: after two 3 × 3 convolutions, which input pixels can change one output pixel?
    3. Lecture 17: in attention, which tokens does each token look at?

    **Today:** the pressure at every junction from 33 sensors, by passing messages along the pipes.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. 35 psi, 24.6 m of water (Ten States Standards, Lecture 12). Today's flag: a junction
           estimated below it.
        2. A 5 × 5 patch: each layer widens the receptive field by one pixel on each side
           (checked below with a gradient). Today the same holds on a graph: each layer reaches
           one more pipe away.
        3. All of them. A graph says which nodes may talk: only those joined by a pipe.
        """)
        }
    )
    return


@app.cell
def _(P_SERVICE, SEED, shown, torch):
    # Two 3 x 3 convolutions: which input pixels move the centre output pixel? The gradient's
    # support is the receptive field.
    torch.manual_seed(SEED)
    _x = torch.zeros(1, 1, 11, 11, requires_grad=True)
    _convs = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.Conv2d(4, 1, 3, padding=1)
    )
    _convs(_x)[0, 0, 5, 5].backward()
    _support = (_x.grad[0, 0] != 0).nonzero()
    _rows, _cols = _support[:, 0], _support[:, 1]
    print(f"35 psi = {P_SERVICE:.4f} m; receptive field of two 3 x 3 layers: "
          f"rows {int(_rows.min())}-{int(_rows.max())}, columns {int(_cols.min())}-{int(_cols.max())}")  # fmt: skip
    assert shown(P_SERVICE, "24.6")
    assert len(_support) == 25 and int(_rows.max() - _rows.min()) == 4 == int(
        _cols.max() - _cols.min()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    L-Town (BattLeDIM; Lectures 9, 12, and 14) reads 33 pressure sensors every 5 minutes; its
    other junctions are not measured. **Decide:** every 5 minutes, flag the junctions whose
    pressure is below 35 psi (24.6 m) though no sensor reads them. A missed flag is a customer
    without service, or a backflow risk; a false flag sends a crew.

    We need an estimate of the **head** (pressure + elevation) at every junction from the
    sensors, the tank level, and the valve settings. We simulate days with L-Town's EPANET model
    to learn from, and test on simulated days and on 2019's real readings. First the model,
    exactly as Lectures 12 and 14 run it: the file's hydraulics, with background leakage in
    Area A calibrated to BattLeDIM's mean 2018 leak flow.
    """)
    return


@app.cell
def _(DAY_S, N1, Path, RAW, SOLVER_WARNINGS, STEP_S, TODAY, contextlib, np, nx, pd, tempfile, wntr):
    def areas(wn) -> tuple[list[str], list[str], list[str]]:
        """Areas A, B, C: the junctions joined by pipes alone (valves and the pump cut them apart).

        Args:
            wn: The water network model.

        Returns:
            Junction names of Area A (fed by PRV-1/2), Area B (PRV-3), and Area C (the tank T1).
        """
        graph = nx.Graph()
        graph.add_nodes_from(wn.junction_name_list)
        graph.add_edges_from((p.start_node_name, p.end_node_name) for _, p in wn.pipes())
        parts = sorted(nx.connected_components(graph), key=len, reverse=True)
        junctions = set(wn.junction_name_list)
        area_a, area_c, area_b = (sorted(set(p) & junctions) for p in parts[:3])
        assert (len(area_a), len(area_b), len(area_c)) == (657, 31, 92)
        assert wn.get_link("PRV-3").end_node_name in area_b and "T1" in parts[1]
        return area_a, area_b, area_c

    def mean_leak_2018() -> float:
        """Mean total leak flow of BattLeDIM's 2018 leaks, m3/h (14 leaks, 5-minute readings)."""
        d = pd.read_csv(RAW / "2018_Leakages.csv", sep=";", decimal=",", index_col=0)
        assert d.shape == (105_120, 14), d.shape
        return float(d.sum(axis=1).mean())

    class Town:
        """L-Town on its peak day with calibrated background leakage in Area A (``ltown.py``).

        Each run writes EPANET's scratch files to its own temporary directory, removed when the
        run ends (pool workers exit without cleaning up after themselves).
        """

        def __init__(self) -> None:
            """Load the model, find the areas and the peak day, and calibrate the leakage."""
            self.wn = wntr.network.WaterNetworkModel(str(RAW / "L-TOWN.inp"))
            self.area_a, self.area_b, self.area_c = areas(self.wn)
            self.day = self._peak_day()
            assert self.day == 3, self.day
            opts = self.wn.options
            opts.time.duration = DAY_S
            opts.time.pattern_start = self.day * DAY_S
            opts.time.report_timestep = STEP_S
            opts.hydraulic.emitter_exponent = N1
            index = {n: i for i, n in enumerate(self.area_a)}
            self.length = np.zeros(len(self.area_a))  # pipe length each junction serves
            for _, p in self.wn.pipes():
                for end in (p.start_node_name, p.end_node_name):
                    if end in index:
                        self.length[index[end]] += p.length / 2
            self.coef = np.zeros(len(self.area_a))
            self.target = mean_leak_2018()
            self._calibrate()

        def weekly_demand(self) -> np.ndarray:
            """The town's nominal total demand at each 5-minute step of the pattern week, m3/s."""
            steps = DAY_S // STEP_S
            total = np.zeros(7 * steps)
            for _, j in self.wn.junctions():
                for d in j.demand_timeseries_list:
                    mult = np.asarray(
                        self.wn.get_pattern(d.pattern_name).multipliers, dtype=np.float64
                    )
                    total += d.base_value * np.resize(
                        mult, total.size
                    )  # a 1-step pattern is constant
            return total

        def _peak_day(self) -> int:
            """Day (0-6) of the one-week demand pattern with the largest total demand."""
            return int(self.weekly_demand().reshape(7, DAY_S // STEP_S).sum(axis=1).argmax())

        def run_settings(self, s1: float, s2: float):
            """Run EPANET with PRV-1 = ``s1``, PRV-2 = ``s2`` (m) and the emitter coefficients.

            Raises:
                RuntimeError: If EPANET reports the hydraulics unbalanced or unstable.
            """
            for name, s in zip(("PRV-1", "PRV-2"), (s1, s2), strict=True):
                self.wn.get_link(name).initial_setting = float(s)
            for name, c in zip(self.area_a, self.coef, strict=True):
                self.wn.get_node(name).emitter_coefficient = float(c) if c > 0 else None
            sim = wntr.sim.EpanetSimulator(self.wn)
            # EPANET also writes a scratch file to the working directory: run inside the
            # temporary directory (chdir is per process; parallel runs use processes)
            with tempfile.TemporaryDirectory(prefix="ltown-") as tmp, contextlib.chdir(tmp):
                res = sim.run_sim(file_prefix=str(Path(tmp) / "ltown"), convergence_error=True)
            failed = [w for w in sim.enData.errcodelist if any(s in w for s in SOLVER_WARNINGS)]
            if failed:
                raise RuntimeError(f"EPANET did not solve the hydraulics at ({s1}, {s2}): {failed}")
            return res

        def leak(self, res) -> np.ndarray:
            """Background leakage of Area A at each step, m3/s: the sum of C_i p_i."""
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            return (p**N1 * self.coef).sum(axis=1)

        def _calibrate(self) -> None:
            """Scale the coefficients so leakage at today's settings equals the 2018 mean leak flow.

            Raises:
                RuntimeError: If the fixed-point steps do not converge.
            """
            res = self.run_settings(*TODAY)
            demand = res.node["demand"][self.area_a].to_numpy()  # without emitters
            p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
            weight = self.length / self.length.sum()
            total = self.target / 3600 / float((p**N1 * weight).sum(axis=1).mean())
            for _ in range(20):
                self.coef = total * weight
                res = self.run_settings(*TODAY)
                leak = float(self.leak(res).mean()) * 3600
                if abs(leak / self.target - 1) < 1e-6:
                    break
                total *= self.target / leak
            else:
                raise RuntimeError(f"leakage calibration did not converge: {leak} vs {self.target}")
            # The formula against the emitter flow EPANET itself reports
            reported = (res.node["demand"][self.area_a].to_numpy() - demand).sum(axis=1).mean()
            assert abs(reported * 3600 - leak) < 1e-3 * self.target, (reported * 3600, leak)

    return Town, areas


@app.cell
def _(Town):
    # One calibrated model in this process: the hydraulic model alone (§5), the closed pipe's
    # candidates (§5), and the attempt's branch (§3). The simulated days run in worker processes.
    town = Town()
    print(f"calibrated background leakage: {town.target:.2f} m3/h at today's settings")
    assert round(town.target, 2) == 25.52
    return (town,)


@app.cell
def _(EDGE_F, RAW, areas, dataclass, np, nx, pd, wntr):
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
            hops: Links between each node and its nearest sensor (any link).
            xy: Map coordinates (N, 2).
            area: 0, 1, 2 for a junction of Area A, B, C; -1 for the others.
        """

        nodes: list[str]
        kind: np.ndarray
        elev: np.ndarray
        sensors: np.ndarray
        src: np.ndarray
        dst: np.ndarray
        edge: np.ndarray
        prv: dict
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

        def without(self, link: str) -> "Graph":
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
        cols = pd.read_csv(RAW / "2019_SCADA_Pressures.csv", sep=";", nrows=0)
        return [c for c in cols.columns if c != "Timestamp"]

    def build_graph(wn) -> Graph:
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
        gx = nx.Graph()
        gx.add_edges_from(zip(src, dst, strict=True))
        d = nx.multi_source_dijkstra_path_length(gx, set(sensors.tolist()), weight=None)
        hops = np.array([d.get(i, -1) for i in range(len(nodes))])
        xy = np.array([wn.get_node(n).coordinates for n in nodes], dtype=float)
        area = np.full(len(nodes), -1)
        for i, names in enumerate(areas(wn)):
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

    def zone_graph(g: Graph) -> nx.Graph:
        """The graph without valves and the pump: its components are the pressure zones."""
        z = nx.Graph()
        z.add_nodes_from(range(len(g.nodes)))
        z.add_edges_from((a, b) for a, b, f in zip(g.src, g.dst, g.edge, strict=True) if f[1] == 1)
        return z

    wn = wntr.network.WaterNetworkModel(str(RAW / "L-TOWN.inp"))
    g = build_graph(wn)
    zg = zone_graph(g)
    # PRV-3 (Area B) keeps the file's setting on every simulated day: read it from the file
    PRV3_SETTING = float(wn.get_link("PRV-3").initial_setting)  # m
    assert PRV3_SETTING == 35.0  # the slides' run
    return Graph, PRV3_SETTING, g, wn, zg, zone_graph


@app.cell
def _(g, mo, np, nx, shown, wn, zg):
    _j = g.junctions
    _types = {
        t: sum(1 for _, _l in wn.links() if _l.link_type == t) for t in ("Pipe", "Valve", "Pump")
    }
    _zones = sorted(nx.connected_components(zg), key=len, reverse=True)
    _small = [sorted(g.nodes[_i] for _i in _c) for _c in _zones[3:]]
    print(
        f"links by type: {_types}; zone sizes: {[len(_c) for _c in _zones]}; the small ones: {_small}"
    )
    # The slide's table
    assert len(_j) == 782 and len(g.sensors) == 33 and len(g.nodes) == 785
    assert len(g.src) // 2 == 909 == sum(_types.values()) and _types == {
        "Pipe": 905,
        "Valve": 3,
        "Pump": 1,
    }
    assert len(_zones) == 5 and [len(_c) for _c in _zones] == [657, 93, 31, 2, 2]
    # The slide's row: junctions per pressure zone (numbers.json "zone_junctions")
    _zone_j = sorted((sum(int(g.kind[_n] == 0) for _n in _c) for _c in _zones), reverse=True)
    assert _zone_j == [657, 92, 31, 1, 1]
    # Two of the five zones are a reservoir and the one junction between it and its valve
    assert sorted(_small) == [["R1", "n303"], ["R2", "n336"]]
    assert int(g.hops.max()) == 11 and float(np.median(g.hops[_j])) == 4.0
    assert np.bincount(g.hops[_j]).tolist() == [33, 69, 110, 134, 139, 104, 74, 56, 32, 21, 9, 1]
    # Areas A, B, C (the error map's notes): junctions, sensors, median links to a sensor
    _area_j = {_a: int((g.area == _i).sum()) for _i, _a in enumerate("ABC")}
    _area_s = {_a: int((g.area[g.sensors] == _i).sum()) for _i, _a in enumerate("ABC")}
    _area_h = {_a: float(np.median(g.hops[g.area == _i])) for _i, _a in enumerate("ABC")}
    assert _area_j == {"A": 657, "B": 31, "C": 92} and _area_s == {"A": 29, "B": 1, "C": 3}
    assert _area_h == {"A": 4.0, "B": 3.0, "C": 5.0}
    assert (g.kind == 1).sum() == 2 and (g.kind == 2).sum() == 1  # 2 reservoirs and the tank
    assert shown(float(np.median(g.hops[_j])), "4") and shown(float(g.hops.max()), "11")
    mo.md(f"""
    | L-Town (BattLeDIM) | Value |
    |:-------------------|------:|
    | Junctions | {len(_j)} |
    | Pressure sensors, read every 5 minutes | {len(g.sensors)} |
    | Links: pipes, 3 pressure-reducing valves, 1 pump | {len(g.src) // 2} |
    | Junctions per pressure zone (valves and the pump separate them) | {", ".join(map(str, _zone_j))} |
    | Pipes from a junction to its nearest sensor: median, most | {np.median(g.hops[_j]):.0f}, {g.hops.max()} |

    Of the five zones, three are the areas of Lectures 12 and 14: Area A ({_area_j["A"]} junctions,
    {_area_s["A"]} sensors, fed by PRV-1 and PRV-2), Area B ({_area_j["B"]} junctions, {_area_s["B"]}
    sensor, behind PRV-3), and Area C ({_area_j["C"]} junctions and the tank, {_area_s["C"]} sensors,
    behind the pump). The other two are each a reservoir and the one junction between it and its
    valve ({", ".join("–".join(_s) for _s in _small)}).
    """)
    return


@app.cell
def _(
    BOX,
    DAY_S,
    JobParallel,
    NOISE,
    STEP_S,
    STRIDE,
    Town,
    WORKERS,
    dataclass,
    delayed,
    np,
    threadpool_limits,
    wntr,
):
    @dataclass(frozen=True)
    class Scenario:
        """One simulated day: weekday, PRV-1 and PRV-2 settings (m), demand scale, noise seed."""

        day: int
        s1: float
        s2: float
        scale: float
        seed: int
        closed: str | None = None

    def draw_scenarios(n: int, rng, scale: tuple, closed: str | None = None) -> list:
        """Days with random weekday, valve settings in Lecture 12's box, and demand scale."""
        return [
            Scenario(
                int(rng.integers(7)),
                float(rng.uniform(*BOX)),
                float(rng.uniform(*BOX)),
                float(rng.uniform(*scale)),
                int(rng.integers(2**31)),
                closed,
            )
            for _ in range(n)
        ]

    def run_day(town, sc: Scenario):
        """One simulated day on ``town``: WNTR's results, every 5 minutes.

        Each junction's demands are scaled by ``scale`` times its own lognormal factor, the
        weekday sets the demand patterns' start, and a ``closed`` pipe is shut; all are restored
        after the run, so the model is unchanged.
        """
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
        status = link.initial_status if link is not None else None  # restored after the run
        try:
            wn.options.time.pattern_start = sc.day * DAY_S
            if link is not None:
                link.initial_status = wntr.network.LinkStatus.Closed
            res = town.run_settings(sc.s1, sc.s2)
        finally:
            for d, v in saved:
                d.base_value = v
            wn.options.time.pattern_start = start
            if link is not None:
                link.initial_status = status
        return res

    def day_heads(res, wn) -> np.ndarray:
        """Heads (snapshots, nodes) of one day, every ``STRIDE``-th 5-minute step, float32."""
        return res.node["head"][list(wn.node_name_list)].to_numpy()[::STRIDE].astype(np.float32)

    def _sim_batch(scs: list) -> list:
        """One worker process: build its own calibrated model, simulate a batch of days.

        Processes, not threads: WNTR's EPANET library is not safe to call from several threads.
        """
        threadpool_limits(1)
        t = Town()
        return [day_heads(run_day(t, sc), t.wn) for sc in scs]

    def simulate_many(scs: list) -> np.ndarray:
        """Heads (days, snapshots, nodes) for many days, in ``WORKERS`` processes, in order."""
        batches = [scs[i::WORKERS] for i in range(WORKERS)]
        done = JobParallel(n_jobs=WORKERS)(delayed(_sim_batch)(b) for b in batches if b)
        out = [None] * len(scs)
        for i, batch in enumerate(done):
            for k, h in enumerate(batch):
                out[i + k * WORKERS] = h
        return np.stack(out)

    @dataclass
    class Snapshots:
        """Flattened snapshots: heads (n, N), the PRV-1, PRV-2 settings (n, 2), hour of the day."""

        heads: np.ndarray
        settings: np.ndarray
        hour: np.ndarray

        @classmethod
        def of(cls, scs: list, heads: np.ndarray) -> "Snapshots":
            """From simulated days: every snapshot of every day."""
            t = heads.shape[1]
            settings = np.repeat([[s.s1, s.s2] for s in scs], t, axis=0).astype(np.float32)
            hour = np.tile(np.arange(t) * STRIDE * STEP_S / 3600, len(scs))
            out = cls(heads.reshape(-1, heads.shape[2]), settings, hour)
            assert out.heads.shape == (len(scs) * t, heads.shape[2])
            return out

        def first(self, n: int) -> "Snapshots":
            """The first ``n`` snapshots."""
            return Snapshots(self.heads[:n], self.settings[:n], self.hour[:n])

    @dataclass
    class Scale:
        """Heads and elevations share one standardization, so pressure stays a difference."""

        mu: float
        sd: float

    return Scale, Scenario, Snapshots, day_heads, draw_scenarios, run_day, simulate_many


@app.cell
def _(
    BOX,
    EXACT,
    FAR_SCALE,
    SCALE,
    SCENARIOS,
    SEED,
    Scale,
    Snapshots,
    draw_scenarios,
    g,
    np,
    shown,
    simulate_many,
    time,
):
    _t0 = time.perf_counter()
    # The slides' draws, in the slides' order: training, validation, test from one generator,
    # then 30 days beyond the training demand from another
    _rng = np.random.default_rng((SEED, 0))
    scs = {k: draw_scenarios(n, _rng, SCALE) for k, n in SCENARIOS.items()}
    scs["far"] = draw_scenarios(SCENARIOS["test"], np.random.default_rng((SEED, 1)), FAR_SCALE)
    _all = [s for k in scs for s in scs[k]]
    _heads = simulate_many(_all)
    _cut = np.cumsum([0] + [len(scs[k]) for k in scs])
    sim_heads = {k: _heads[_a:_b] for k, _a, _b in zip(scs, _cut[:-1], _cut[1:], strict=True)}
    per_day = int(_heads.shape[1])
    snaps = {k: Snapshots.of(scs[k], sim_heads[k]) for k in scs}
    _j = g.junctions
    scale = Scale(
        float(snaps["train"].heads[:, _j].mean()), float(snaps["train"].heads[:, _j].std())
    )
    print(f"simulated {len(_all)} days in {time.perf_counter() - _t0:.0f} s: "
          f"{ {k: len(v.heads) for k, v in snaps.items()} } snapshots; mu {scale.mu:.4f} m, sd {scale.sd:.4f} m")  # fmt: skip
    # The draws are the slides' (first and last of each set, from the figure script)
    assert (scs["train"][0].day, scs["train"][0].s1, scs["train"][0].scale, scs["train"][0].seed) == (
        2, 46.20464603498388, 0.7534587237533253, 1677437249)  # fmt: skip
    assert (scs["train"][-1].day, scs["train"][-1].s2) == (1, 59.99335748926765)
    assert (scs["val"][0].day, scs["val"][0].s1, scs["val"][0].seed) == (
        5,
        58.179671602429266,
        135991290,
    )
    assert (scs["test"][0].day, scs["test"][0].s1, scs["test"][0].s2, scs["test"][0].scale) == (
        6, 58.89266117534497, 36.08364918737439, 1.2363823808260745)  # fmt: skip
    assert (scs["far"][0].day, scs["far"][0].scale, scs["far"][0].seed) == (
        4,
        1.6238251484281074,
        1706594403,
    )
    assert all(BOX[0] <= s.s1 <= BOX[1] and BOX[0] <= s.s2 <= BOX[1] for s in _all)
    assert all(SCALE[0] <= s.scale <= SCALE[1] for k in ("train", "val", "test") for s in scs[k])
    assert all(FAR_SCALE[0] <= s.scale <= FAR_SCALE[1] for s in scs["far"])
    # The simulated heads are the slides' (EPANET is deterministic): the test days' sum, the
    # training days' standardization
    assert per_day == 25 and {k: len(v.heads) for k, v in snaps.items()} == {
        "train": 6000, "val": 750, "test": 750, "far": 750}  # fmt: skip
    assert abs(float(sim_heads["test"].astype(np.float64).sum()) - 46581881.72102356) < 1e-6
    assert abs(scale.mu - 76.85603332519531) < EXACT and abs(scale.sd - 13.72947883605957) < EXACT
    assert shown(SCALE[0], "0.7") and shown(FAR_SCALE[1], "1.8")
    return per_day, scale, scs, sim_heads, snaps


@app.cell
def _(EXACT, PRV3_SETTING, SEED, g, np, nx, per_day, shown, snaps):
    def attempt_table(snap, t: int) -> dict:
        """One unmeasured junction 3 links from its nearest sensor, at one snapshot.

        Drawn at random (seed 21) among junctions exactly 3 links from a sensor; the table gives
        its elevation and the two nearest sensors' heads.
        """
        rng = np.random.default_rng(SEED)
        cand = np.flatnonzero((g.kind == 0) & (g.hops == 3))
        j = int(rng.choice(cand))
        gx = nx.Graph()
        gx.add_edges_from(zip(g.src, g.dst, strict=True))
        d = nx.single_source_shortest_path_length(gx, j)
        near = sorted((d[int(s)], int(s)) for s in g.sensors if int(s) in d)[:2]
        return {
            "j": j,
            "junction": g.nodes[j],
            "elevation": float(g.elev[j]),
            "hour": float(snap.hour[t]),
            "true_head": float(snap.heads[t, j]),
            "true_pressure": float(snap.heads[t, j] - g.elev[j]),
            "sensors": [
                {
                    "name": g.nodes[s],
                    "links": int(k),
                    "head": float(snap.heads[t, s]),
                    "elevation": float(g.elev[s]),
                    "pressure": float(snap.heads[t, s] - g.elev[s]),
                }
                for k, s in near
            ],
        }

    # The first test day's lowest-head moment (mean over the junctions)
    t_peak = int(np.argmin(snaps["test"].heads[:per_day, g.junctions].mean(1)))
    attempt = attempt_table(snaps["test"], t_peak)
    _s = attempt["sensors"]
    assert attempt["junction"] == "n213" and attempt["hour"] == 21.0 and t_peak == 21
    assert abs(attempt["true_head"] - 41.091896057128906) < EXACT
    assert [(_x["name"], _x["links"]) for _x in _s] == [("n215", 3), ("n229", 8)]
    assert (
        abs(_s[0]["head"] - 41.090476989746094) < EXACT
        and abs(_s[1]["head"] - 81.52720642089844) < EXACT
    )
    # The slide's table: elevations 3.24, 2.01, 21.58; pressures 39.08, 59.95
    assert (
        shown(attempt["elevation"], "3.24")
        and shown(_s[0]["elevation"], "2.01")
        and shown(_s[1]["elevation"], "21.58")
    )
    assert shown(_s[0]["pressure"], "39.08") and shown(_s[1]["pressure"], "59.95")
    # The hint: PRV-3 holds Area B's head near its setting plus its outlet's elevation; Area B's
    # sensor n215 reads within centimetres of it
    _held = PRV3_SETTING + float(g.elev[g.prv["PRV-3"][2]])
    print(f"PRV-3 holds {_held:.2f} m; n215 reads {_s[0]['head']:.2f} m")
    assert (
        g.area[g.nodes.index("n215")] == 1
        and g.area[attempt["j"]] == 1
        and abs(_s[0]["head"] - _held) < 0.1
    )
    assert g.area[g.nodes.index("n229")] == 0
    return attempt, t_peak


@app.cell
def _(attempt, mo):
    _s = attempt["sensors"]
    mo.md(f"""
    ### Your turn: estimate the pressure at {attempt["junction"]}

    | At {attempt["hour"]:02.0f}:00 on a test day | Links away | Elevation, m | Pressure, m |
    |:---------------------------------------|:----------:|-------------:|------------:|
    | Junction {attempt["junction"]} | 0 | {attempt["elevation"]:.2f} | ? |
    | Sensor {_s[0]["name"]} | {_s[0]["links"]} | {_s[0]["elevation"]:.2f} | {_s[0]["pressure"]:.2f} |
    | Sensor {_s[1]["name"]} | {_s[1]["links"]} | {_s[1]["elevation"]:.2f} | {_s[1]["pressure"]:.2f} |

    Which sensor do you trust, and why? Two minutes, alone. The junction was drawn at random
    (seed 21) among junctions exactly 3 links from their nearest sensor; the hour is the first
    test day's lowest-head moment. The answers come in §4.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(r"""
        Head = pressure + elevation, and water flows from high head to low head. Sensor n215
        reads a head of 41.09 m; n229, across PRV-3 in Area A, 81.53 m. The valve between them
        holds the head of n215's zone (Area B) near its setting.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: information travels one pipe per layer

    A graph: junctions, reservoirs, and the tank are nodes; pipes, valves, and the pump are
    edges. Each layer of a graph network lets every node read its neighbours, so after $k$
    layers a node has heard from everything within $k$ links. The picture: what a reading at a
    sensor can reach, with no layer, after 2, and after 12.
    """)
    return


@app.cell
def _(LAYERS, LineCollection, NAVY, PALETTE, g, np, plt):
    def draw_pipes(ax, color: str = "#c9c9c9", lw: float = 0.6) -> None:
        """Every link of L-Town as a grey segment on ``ax``."""
        seg = [(g.xy[a], g.xy[b]) for a, b in zip(g.src[::2], g.dst[::2], strict=True)]
        ax.add_collection(LineCollection(seg, colors=color, linewidths=lw, zorder=1))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale()

    _j = g.junctions
    reach = {k: int(((g.hops[_j] <= k) & (g.hops[_j] >= 0)).sum()) for k in range(LAYERS + 1)}
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for _ax, _k in zip(_axes, (None, 2, LAYERS), strict=True):
        draw_pipes(_ax)
        if _k is not None:
            _near = _j[(g.hops[_j] <= _k) & (g.hops[_j] >= 0)]
            _ax.scatter(*g.xy[_near].T, s=4, color=PALETTE[0], zorder=2)
            _ax.set_title(f"after {_k} layers: {len(_near)} of {len(_j)} junctions", fontsize=11)
        else:
            _ax.set_title(
                f"{len(g.sensors)} sensors (stars); reservoirs and tank (squares)", fontsize=11
            )
        _ax.scatter(*g.xy[g.sensors].T, s=50, marker="*", color=PALETTE[1], zorder=3)
        _ax.scatter(*g.xy[g.fixed].T, s=40, marker="s", color=NAVY, zorder=3)
    print({k: reach[k] for k in (2, 10, 11, 12)})
    # The frames' counts: 212 of 782 junctions within 2 links of a sensor; all 782 from 11 on
    assert reach[2] == 212 and reach[11] == reach[LAYERS] == 782 and reach[10] == 781
    assert reach[2] == int(np.bincount(g.hops[_j])[:3].sum())
    _fig
    return draw_pipes, reach


@app.cell
def _(mo, reach):
    mo.md(f"""
    After 2 layers, {reach[2]} of 782 junctions have heard from a sensor; the rest have heard
    only the town-wide mean of every node (§3), whatever the weights. After 11 layers every
    junction can (the most links to a sensor is 11:
    {782 - reach[10]} junction is 11 links away); the network uses 12. This is Lecture 16's
    receptive field on a graph: a convolution widens it by a pixel per layer, a graph layer by a
    link.

    | Data | Neighbours | Lecture |
    |:-----|:-----------|:--------|
    | Pixels on a grid | the 3 × 3 around each pixel, same weights everywhere | 15, 16 |
    | Tokens in a sequence | every token (attention) | 17 |
    | A fixed list of readings | none: one weight per input | 14 (MLP) |
    | Nodes and edges | the nodes a link joins, same weights for every node and link | today |

    The graph is the structure the network is told about. A convolution assumes a grid;
    attention assumes nothing; a graph network assumes that what matters at a node reaches it
    along the edges. For a water network the edges are the pipes (and the valves and the pump).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive: send, sum, update

    **Assumption:** a node's state depends on its neighbours through the same rules at every
    node and every link. One layer updates every node's state $h_i$ at once:

    $$h_i \leftarrow h_i + U\Big(h_i,\ \sum_{j \to i} M(h_j, h_i, e_{ji}),\ \bar h\Big)$$

    - **Send.** Every edge $j \to i$ computes a message $M(h_j, h_i, e_{ji})$, with $M$ a small
      MLP shared by all edges. The edge features $e_{ji}$: a pipe's Hazen-Williams resistance
      ($\log_{10}$ of $10.67\,L / (C^{1.852} D^{4.87})$, standardized); for a valve, its direction
      and the head it holds (setting + outlet elevation); for the pump, its direction.
    - **Sum.** Node $i$ adds what arrives. A sum does not depend on the order of the neighbours,
      so the layer cannot depend on how the nodes are numbered (checked below). In code, the sum
      is one `index_add_` over the edges' destination indices: a product with the incidence
      matrix $B$, $B_{ik} = 1$ if edge $k$ ends at node $i$.
    - **Update.** $U$, another shared MLP, maps the node's state, its sum, and $\bar h$, the mean
      of all nodes' states, to a change of state (a residual step, Lecture 15). The mean is the
      graph-level attribute of Battaglia et al. (2018): the only path for a town-wide quantity
      to reach a node beyond $k$ links.

    A graph convolution (Kipf & Welling, 2017) is the special case with a linear message and a
    degree-scaled sum. Each node starts from six features: its observed head (0 if not
    observed), whether it is observed, its elevation, and its type (junction, reservoir, tank).
    Heads and elevations share one standardization, so the network can form pressure as a
    difference. A decoder maps each node's final state to its head. Sensors are hidden at random
    in training (probability 0.2 each), so the network learns to fill in what it does not see.
    """)
    return


@app.cell
def _(DEVICE, EDGE_F, HIDDEN, MLP_HIDDEN, NODE_F, PRV3_SETTING, P_HIDE, nn, np, torch):
    def observe(g, n: int, rng, hide: int | None = None) -> np.ndarray:
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

    def inputs(g, sc, heads: torch.Tensor, observed: torch.Tensor, settings: torch.Tensor):
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
        assert x.shape == (b, n, NODE_F) and e.shape == (b, len(g.src), EDGE_F)
        return x, e

    class MessagePassing(nn.Module):
        """One layer: every edge sends a message; each node sums what arrives and updates itself.

        With z = LayerNorm(h) (pre-norm), h_i <- h_i + U(z_i, sum over edges j->i of
        M(z_j, z_i, e_ji), mean over nodes of z): the slides' formula, with the messages and the
        update reading the normalized states. M and U are small MLPs shared by every node and
        edge (Gilmer et al., 2017). The mean over nodes is the graph-level attribute of
        Battaglia et al. (2018); ``global_mean=False`` drops it: a purely local layer.
        """

        def __init__(self, d: int, global_mean: bool = True) -> None:
            """Message and update networks of width ``d``."""
            super().__init__()
            self.global_mean = global_mean
            self.norm = nn.LayerNorm(d)  # pre-norm: keeps 12 residual layers stable
            self.message = nn.Sequential(nn.Linear(2 * d + EDGE_F, d), nn.ReLU(), nn.Linear(d, d))
            self.update = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))

        def forward(self, h, e, src, dst):
            """Node states h (B, N, d) and edge features e (B, E, EDGE_F) -> new states (B, N, d)."""
            z = self.norm(h)
            m = self.message(torch.cat([z[:, src], z[:, dst], e], dim=-1))  # (B, E, d)
            agg = torch.zeros_like(z).index_add_(1, dst, m)  # (B, N, d): sum of incoming messages
            u = (
                z.mean(dim=1, keepdim=True).expand_as(z)
                if self.global_mean
                else torch.zeros_like(z)
            )
            return h + self.update(torch.cat([z, agg, u], dim=-1))

    class GNN(nn.Module):
        """Encode each node, pass messages ``layers`` times, decode a head per node."""

        def __init__(self, layers: int, d: int = HIDDEN, global_mean: bool = True) -> None:
            """A network with ``layers`` message-passing layers of width ``d``."""
            super().__init__()
            self.encode = nn.Linear(NODE_F, d)
            self.layers = nn.ModuleList(MessagePassing(d, global_mean) for _ in range(layers))
            self.decode = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
            )

        def forward(self, x, e, g):
            """Node features x (B, N, NODE_F), edge features e (B, E, EDGE_F) -> standardized heads (B, N)."""
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

        def __init__(self, g, d: int = MLP_HIDDEN) -> None:
            """Input and output sizes fixed by this network's sensors and nodes."""
            super().__init__()
            self.sensors = torch.as_tensor(g.sensors)
            self.tank = int(np.flatnonzero(g.kind == 2)[0])
            self.prv = [g.prv[k][0] for k in ("PRV-1", "PRV-2")]
            n_in = 2 * len(g.sensors) + 1 + 2
            self.net = nn.Sequential(
                nn.Linear(n_in, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, len(g.nodes)),
            )

        def forward(self, x, e, g):  # noqa: ARG002 (the GNN's signature)
            """The same inputs as the GNN, read at fixed positions -> standardized heads (B, N)."""
            s = x[:, self.sensors.to(x.device), :2].flatten(1)  # (B, 66): value, observed
            z = torch.cat([s, x[:, self.tank, :1], e[:, self.prv, 4]], dim=1)  # (B, 69)
            assert z.shape == (x.shape[0], 2 * len(self.sensors) + 3)
            return self.net(z)

    def predict(model, g, sc, snaps, obs: np.ndarray, *, batch: int = 256) -> np.ndarray:
        """Heads (n, N), m, for snapshots with the given observed masks (no gradients)."""
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

    def rmse(a: np.ndarray, b) -> float:
        """Root-mean-square difference, m."""
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"training device: {DEVICE}")
    return GNN, MLP, inputs, observe, predict, rmse


@app.cell
def _(HALO, LineCollection, PALETTE, attempt, g, mo, nx, plt, snaps, t_peak):
    # The attempt's junction and everything within 3 links, coloured by pressure
    _gx = nx.Graph()
    _gx.add_edges_from(zip(g.src, g.dst, strict=True))
    _j = attempt["j"]
    _near = nx.single_source_shortest_path_length(_gx, _j, cutoff=3)
    _nodes = list(_near)
    _heads = snaps["test"].heads[t_peak]
    _sens = [_n for _n in _nodes if _n in set(g.sensors.tolist())]
    _nbrs = list(_gx.neighbors(_j))
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    _seg = [(g.xy[_a], g.xy[_b]) for _a, _b in _gx.subgraph(_nodes).edges]
    _ax.add_collection(LineCollection(_seg, colors="#9a9a9a", linewidths=1.5, zorder=1))
    _pts = _ax.scatter(*g.xy[_nodes].T, c=_heads[_nodes] - g.elev[_nodes], cmap="viridis", s=70, zorder=2,
                       edgecolors="white", linewidths=0.8)  # fmt: skip
    _fig.colorbar(_pts, ax=_ax, shrink=0.8, label="pressure, m (head - elevation)")
    for _nb in _nbrs:
        _ax.annotate("", xy=g.xy[_j], xytext=g.xy[_nb], zorder=3,
                     arrowprops={"arrowstyle": "-|>", "color": PALETTE[1], "lw": 2})  # fmt: skip
    _ax.scatter(*g.xy[_sens].T, s=260, marker="*", color=PALETTE[1], zorder=4)
    for _n in [_j, *_sens]:
        _ax.annotate(f"{g.nodes[_n]}\nhead {_heads[_n]:.2f} m\nelevation {g.elev[_n]:.2f} m", g.xy[_n],
                     xytext=(8, 8), textcoords="offset points", fontsize=10, path_effects=HALO)  # fmt: skip
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.autoscale()
    _x0, _x1 = _ax.get_xlim()
    _ax.set_xlim(_x0, _x1 + 0.5 * (_x1 - _x0))
    # The slide's picture: 10 nodes within 3 links; one neighbour (n213 ends a branch); one
    # sensor among them, n215, whose head reaches n213 after 3 layers
    assert len(_nodes) == 10 and len(_nbrs) == 1 and [g.nodes[_n] for _n in _sens] == ["n215"]
    assert _near[_sens[0]] == 3
    assert f"{_heads[_j]:.2f}" == f"{_heads[_sens[0]]:.2f}" == "41.09"
    mo.vstack([_fig, mo.md(f"Junction {attempt['junction']} ends a branch: its one neighbour, {g.nodes[_nbrs[0]]}, "
                             "passes it what the branch knows, and the sensor's head reaches it after 3 layers.")])  # fmt: skip
    return


@app.cell
def _(STRIDE, attempt, day_heads, g, mo, np, nx, pairwise, run_day, scs, sim_heads, t_peak, town):
    # The heads in the attempt's branch agree to the millimetre. Is the water at rest? Run the
    # first test day again in this process and read the flows on the path n215 -> n213.
    _res = run_day(town, scs["test"][0])
    assert np.array_equal(
        day_heads(_res, town.wn), sim_heads["test"][0]
    )  # the workers' run, exactly
    _gx = nx.Graph()
    _gx.add_edges_from(zip(g.src, g.dst, strict=True))
    _path = [g.nodes[_n] for _n in nx.shortest_path(_gx, g.nodes.index("n215"), attempt["j"])]
    _step = t_peak * STRIDE
    _pipes = {}
    for _a, _b in pairwise(_path):
        _name = next(
            _n for _n, _l in town.wn.links() if {_l.start_node_name, _l.end_node_name} == {_a, _b}
        )
        _pipes[_name] = abs(float(_res.link["flowrate"][_name].iloc[_step])) * 1000  # L/s
    _h = [float(_res.node["head"][_n].iloc[_step]) for _n in _path]
    _spread = max(_h) - min(_h)
    print(f"path {_path}; flows (L/s) {_pipes}; head spread {_spread * 1000:.1f} mm")
    assert _path == ["n215", "n212", "n210", "n213"]
    # Not at rest: every pipe carries its customers' demand, all under 0.3 L/s, so friction
    # losses along the branch are millimetres (and within EPANET's solver tolerance)
    assert all(0 < _q < 0.3 for _q in _pipes.values()) and _spread < 0.005
    assert (
        abs(_h[0] - attempt["sensors"][0]["head"]) < 1e-4
        and abs(_h[-1] - attempt["true_head"]) < 1e-4
    )
    mo.md(f"""
    The heads in the branch agree to the centimetre: {min(_h):.4f} to {max(_h):.4f} m, a spread of
    {_spread * 1000:.1f} mm. The water is not at rest (the three pipes carry
    {", ".join(f"{_q:.2f}" for _q in _pipes.values())} L/s to the customers on them), but such
    small flows lose almost no head to friction, so the pressures differ by the elevations:
    {attempt["sensors"][0]["pressure"]:.2f} − ({attempt["elevation"]:.2f} − {attempt["sensors"][0]["elevation"]:.2f})
    ≈ {attempt["sensors"][0]["pressure"] - attempt["elevation"] + attempt["sensors"][0]["elevation"]:.2f} m at
    {attempt["junction"]}; the truth is {attempt["true_pressure"]:.2f} m.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    ```python
    agg = zeros(N, d).index_add_(0, dst, m)  # each node's incoming sum
    dense = B @ m                     # B[i, k] = 1 if edge k ends at i
    out = model(relabel(x))           # shuffle the node numbers
    # Two purely local layers; add 5 m to one sensor's reading:
    moved = nodes whose estimate changes
    ```

    How close are `agg` and `dense`? Does relabelling change the answers? Which nodes move?
    Two minutes, written.
    """)
    return


@app.cell
def _(DEVICE, GNN, Graph, LAYERS, SEED, Snapshots, nx, np, observe, predict, torch):
    def aggregation_check(g) -> dict:
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
        assert ours.shape == dense.shape == (len(g.nodes), 8)
        return {
            "max_abs_diff": float((ours - dense).abs().max()),
            "rows": inc.shape[0],
            "columns": inc.shape[1],
        }

    def equivariance_check(model, g, sc, snaps) -> float:
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
        a = predict(
            model, g, sc, Snapshots(snaps.heads[:4], snaps.settings[:4], snaps.hour[:4]), obs
        )
        b = predict(
            model,
            pg,
            sc,
            Snapshots(snaps.heads[:4][:, perm], snaps.settings[:4], snaps.hour[:4]),
            obs[:, perm],
        )
        return float(np.abs(a[:, perm] - b).max())

    def reach_check(model, g, sc, snaps, layers: int) -> dict:
        """Change one sensor's reading by 5 m: only nodes within ``layers`` links may change."""
        s = int(g.sensors[0])
        obs = observe(g, 1, None)
        one = Snapshots(snaps.heads[:1].copy(), snaps.settings[:1], snaps.hour[:1])
        a = predict(model, g, sc, one, obs)
        one.heads[0, s] += 5.0
        b = predict(model, g, sc, one, obs)
        gx = nx.Graph()
        gx.add_edges_from(zip(g.src, g.dst, strict=True))
        d = nx.single_source_shortest_path_length(gx, s)
        dist = np.array([d.get(i, 10**6) for i in range(len(g.nodes))])
        moved = np.abs(b - a)[0] > 0
        return {
            "sensor": g.nodes[s],
            "moved": int(moved.sum()),
            "within": int((dist <= layers).sum()),
            "farthest_moved": int(dist[moved].max()),
            "max_change_beyond": float(np.abs(b - a)[0][dist > layers].max()),
        }

    def untrained(layers: int, global_mean: bool = True):
        """A network with the slides' starting weights (seed 21), untrained."""
        torch.manual_seed(SEED)
        return GNN(layers, global_mean=global_mean).to(DEVICE)

    assert LAYERS == 12
    return aggregation_check, equivariance_check, reach_check, untrained


@app.cell
def _(
    FEW_LAYERS,
    LAYERS,
    SLIDES,
    aggregation_check,
    equivariance_check,
    g,
    mo,
    reach_check,
    scale,
    shown,
    snaps,
    untrained,
):
    agg_check = aggregation_check(g)
    # Relabelling and reach are properties of the layer, whatever the weights: checked here on
    # untrained networks, and in §4 again on the trained ones
    _eq0 = equivariance_check(untrained(LAYERS), g, scale, snaps["test"])
    reach_local = reach_check(
        untrained(FEW_LAYERS[0], global_mean=False), g, scale, snaps["test"], FEW_LAYERS[0]
    )
    print(
        f"aggregation {agg_check}; relabelled, untrained 12 layers: {_eq0:.2e} m; local reach {reach_local}"
    )
    assert agg_check["rows"] == 785 and agg_check["columns"] == 1818
    assert agg_check["max_abs_diff"] < 1e-14  # float64 round-off
    if SLIDES:  # the slide's digit, 9e-16 (8.881784197001252e-16): the slides' machine's BLAS
        assert shown(agg_check["max_abs_diff"], "9e-16")
    assert _eq0 < 1e-4  # float32 round-off on heads of tens of metres
    # Two purely local layers: one sensor nudged moves exactly the 3 nodes within 2 links
    assert reach_local == {
        "sensor": "n1",
        "moved": 3,
        "within": 3,
        "farthest_moved": 2,
        "max_change_beyond": 0.0,
    }
    mo.md(f"""
    | Check (untrained networks) | Value |
    |:------|------:|
    | `index_add_` against the incidence product ({agg_check["rows"]} × {agg_check["columns"]:,}), largest difference | {agg_check["max_abs_diff"]:.0e} |
    | Nodes relabelled at random (12 layers): largest change in any estimate, m | {_eq0:.0e} |
    | 2 local layers, sensor {reach_local["sensor"]} nudged by 5 m: nodes that move | {reach_local["moved"]} |
    | … nodes within 2 links of that sensor | {reach_local["within"]} |
    | … largest change beyond 2 links, m | {reach_local["max_change_beyond"]:g} |

    The sum is a sparse matrix product, to float64 round-off. Relabelling permutes the answers
    and changes nothing else, to float32 round-off: the network has no notion of a node's
    number, only of its neighbourhood (permutation equivariance). With purely local layers,
    nothing beyond 2 links can move, whatever the weights.
    """)
    return (agg_check,)


@app.cell
def _(DAY_S, STEP_S, STRIDE, g, mo, np, per_day, scs, snaps, town):
    # What must the town-wide mean carry? Area A's mean head in each training snapshot, against
    # the two valve settings, and against the town's demand at that hour (the nominal weekly
    # demand, times the day's scale)
    _tr = snaps["train"]
    _y = _tr.heads[:, g.area == 0].mean(1).astype(np.float64)
    _n = len(scs["train"])
    _step = (np.repeat([s.day for s in scs["train"]], per_day) * (DAY_S // STEP_S)
             + np.tile(np.arange(per_day) * STRIDE, _n)) % (7 * DAY_S // STEP_S)  # fmt: skip
    _demand = np.repeat([s.scale for s in scs["train"]], per_day) * town.weekly_demand()[_step]
    _one = np.ones_like(_y)

    def _r2(X: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(X, _y, rcond=None)
        return float(1 - np.var(_y - X @ coef) / np.var(_y))

    mean_r2 = {
        "valves": _r2(np.c_[_one, _tr.settings]),
        "demand": _r2(np.c_[_one, _demand, _demand**2]),
        "both": _r2(
            np.c_[_one, _tr.settings, _demand, _demand**2, _tr.settings * _demand[:, None]]
        ),
    }
    print(f"Area A mean head, R^2: {mean_r2}")
    # The valve settings explain ten times what the demand does (deterministic: the simulated days)
    assert abs(mean_r2["valves"] - 0.7985209628632347) < 1e-6
    assert abs(mean_r2["demand"] - 0.06974012122610074) < 1e-6
    assert mean_r2["valves"] > 10 * mean_r2["demand"] and mean_r2["both"] >= mean_r2["valves"]
    mo.md(f"""
    **What the town-wide mean has to carry.** Area A's mean head over a snapshot varies by
    {_y.std():.1f} m (sd) across the training snapshots. The two valve settings explain
    R² = {mean_r2["valves"]:.2f} of it; the town's demand at that hour alone, {mean_r2["demand"]:.2f}; both,
    {mean_r2["both"]:.2f}. A junction more than $k$ links from every sensor and from both valves can
    learn the settings only through $\\bar h$. (The valve settings dominate Area A's heads because PRV-1 and PRV-2 set the
    head where water enters the zone; demand sets the friction losses below them.)
    """)
    return (mean_r2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    The four networks, trained on the same simulated snapshots with the same protocol:

    - the message-passing network above with 12 layers (one more than the most links between a
      junction and its nearest sensor), and the same network with 2 and with 5 layers;
    - an MLP on the 33 readings (each sensor's value and whether it is hidden), the tank's head,
      and the two PRV heads: Lecture 14's network, wider, with no graph.

    Adam on the squared error at the junctions, each sensor hidden with probability 0.2 in every
    training snapshot, gradients clipped at norm 1; the rate halves after 3 epochs without
    improvement; training stops after 15 epochs without a better validation error, at most 200,
    and keeps the best epoch's weights. Validation: the head error at unobserved junctions on the
    30 validation days, sensors hidden with one fixed draw. Every model searches its learning
    rate the same way: Adam at 3e-4 and 1e-3, then, while the best on validation is at an edge of
    the rates tried, the next rate past that edge (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    1e-1), until the best lies inside the range tried: the same protocol for every model (Lecture
    13). A grid of two rates cannot show that the best is inside it. One training seed per model:
    a gap of a few centimetres between two networks is "close", not a win. Two baselines with no
    learning: each junction's training mean, and the head of the nearest observed node in the same
    pressure zone.
    """)
    return


@app.cell
def _(BATCH, DEVICE, LR_LADDER, PATIENCE, SEED, inputs, np, observe, predict, rmse, time, torch):
    def train_model(model, g, sc, tr, va, *, seed: int, lr: float, epochs: int) -> dict:
        """Adam on squared error at the junctions, sensors hidden at random; early stopping.

        The validation snapshots hide sensors with a fixed draw, and the score is the head error
        at unobserved junctions; the rate halves after 3 epochs without improvement, and the
        best epoch's weights are kept. Gradients clipped at norm 1.
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
        for _ in range(epochs):
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

    def fit_best(
        make, g, sc, tr, va, *, rates: tuple, epochs: int, converged: bool, max_rates: int | None
    ) -> tuple:
        """Train one fresh model per rate; keep the one with the lowest validation error.

        A bracketing search: start from ``rates``; whenever the best rate sits at an edge of the
        rates tried, train at the next rate of ``LR_LADDER`` past that edge too, so the rate kept
        is inside the range tried (asserted here, so no caller can skip it). Every model starts
        from the same draw (seed 21), as in the slides' script. With ``converged``, the kept
        run's best epoch must not be the last allowed. ``max_rates`` (the reduced setting) stops
        the search after that many rates; the kept rate may then sit at an edge, which is
        printed, not asserted: the stopped search cannot guarantee an interior rate.

        Returns:
            The kept model, and its training record with every rate's run, sorted by rate.
        """
        best, runs, todo = None, [], list(rates)
        while todo:
            lr = todo.pop(0)
            torch.manual_seed(SEED)
            model = make()
            info = train_model(model, g, sc, tr, va, seed=SEED, lr=lr, epochs=epochs) | {"lr": lr}
            runs.append({k: info[k] for k in ("lr", "val_rmse", "epochs", "best_epoch")})
            print(f"  {type(model).__name__} lr {lr:g}: val {info['val_rmse']:.3f} m, {info['epochs']} epochs "
                  f"(best {info['best_epoch']}), {info['seconds']:.0f} s", flush=True)  # fmt: skip
            if best is None or info["val_rmse"] < best[1]["val_rmse"]:
                best = (model, info)
            tried = sorted(r["lr"] for r in runs)
            room = max_rates is None or len(runs) < max_rates
            if not todo and room and best[1]["lr"] in (tried[0], tried[-1]):  # at an edge: widen
                i = LR_LADDER.index(best[1]["lr"]) + (-1 if best[1]["lr"] == tried[0] else 1)
                assert 0 <= i < len(LR_LADDER), f"the best rate {best[1]['lr']:g} ends the ladder"
                todo.append(LR_LADDER[i])
        tried = sorted(r["lr"] for r in runs)
        interior = tried[0] < best[1]["lr"] < tried[-1]
        if max_rates is None:
            assert interior, (best[1]["lr"], tried)
        elif not interior:
            print(f"  kept rate {best[1]['lr']:g} at an edge of {tried}: the reduced search "
                  f"stops after {max_rates} rates", flush=True)  # fmt: skip
        if converged:
            assert best[1]["best_epoch"] < epochs or best[1]["epochs"] < epochs, best[1]
        return best[0], best[1] | {"runs": sorted(runs, key=lambda r: r["lr"])}

    return (fit_best,)


@app.cell
def _(
    CHECK_TRAINED,
    EPOCHS,
    FEW_LAYERS,
    GNN,
    LAYERS,
    LRS,
    MAX_RATES,
    MLP,
    TRAIN_USED,
    VAL_USED,
    fit_best,
    g,
    per_day,
    scale,
    snaps,
    time,
):
    # The long cell: four networks, three to five rates each (2.3 hours on a shared RTX 3090 Ti)
    _t0 = time.perf_counter()
    _tr = snaps["train"].first(TRAIN_USED * per_day)
    _va = snaps["val"].first(VAL_USED * per_day)
    _makes = {f"GNN, {k} layers": (lambda k=k: GNN(k)) for k in (LAYERS, *FEW_LAYERS)}
    _makes["MLP (no graph)"] = lambda: MLP(g)
    models, training = {}, {}
    for _name, _make in _makes.items():
        print(_name, flush=True)
        models[_name], training[_name] = fit_best(
            _make,
            g,
            scale,
            _tr,
            _va,
            rates=LRS,
            epochs=EPOCHS,
            converged=CHECK_TRAINED,
            max_rates=MAX_RATES,
        )
    train_minutes = (time.perf_counter() - _t0) / 60
    print(f"trained in {train_minutes:.0f} min on {len(_tr.heads)} snapshots")
    return models, train_minutes, training


@app.cell
def _(
    CHECK_TRAINED,
    EPOCHS,
    LRS,
    MAX_EPOCHS,
    MAX_RATES,
    NET_TOL,
    PALETTE,
    PATIENCE,
    mo,
    models,
    plt,
    shown,
    training,
):
    _ref = {  # numbers.json: every rate tried, (rate, validation error, epochs, best epoch)
        "GNN, 12 layers": [(3e-4, 0.27364569902420044, 200, 197), (1e-3, 0.22666296362876892, 193, 178),
                           (3e-3, 0.2473532259464264, 200, 199)],
        "GNN, 2 layers": [(3e-4, 1.8654804229736328, 163, 148), (1e-3, 1.7563250064849854, 138, 123),
                          (3e-3, 1.7152228355407715, 56, 41), (1e-2, 1.8270761966705322, 116, 101)],
        "GNN, 5 layers": [(1e-4, 2.1332435607910156, 111, 96), (3e-4, 0.44732311367988586, 200, 197),
                          (1e-3, 0.46449095010757446, 69, 54)],
        "MLP (no graph)": [(3e-4, 0.1870162934064865, 200, 199), (1e-3, 0.17550954222679138, 147, 132),
                           (3e-3, 0.16787052154541016, 106, 91), (1e-2, 0.15613552927970886, 141, 126),
                           (3e-2, 0.7781669497489929, 69, 54)],
    }  # fmt: skip
    _params = {_k: sum(_p.numel() for _p in _m.parameters()) for _k, _m in models.items()}
    print({_k: [(_r["lr"], round(_r["val_rmse"], 4), _r["epochs"], _r["best_epoch"]) for _r in _v["runs"]]
           for _k, _v in training.items()})  # fmt: skip
    # The notes' weights: the MLP 701,201, the 12-layer GNN 358,849 (exact, whatever the training)
    assert _params == {"GNN, 12 layers": 358_849, "GNN, 2 layers": 63_809, "GNN, 5 layers": 152_321,
                       "MLP (no graph)": 701_201}  # fmt: skip
    # Every search started from 3e-4 and 1e-3 and, in the slides setting, kept a rate inside the
    # range it tried. The reduced search stops after MAX_RATES rates, so its kept rate may sit at
    # an edge: there the check is printed, not asserted.
    _interior = {}
    for _k, _v in training.items():
        _tried = [_r["lr"] for _r in _v["runs"]]
        assert set(LRS) <= set(_tried)
        _interior[_k] = min(_tried) < _v["lr"] < max(_tried)
    print(f"kept rate inside the range tried: {_interior}")
    if MAX_RATES is None:
        assert all(_interior.values()), _interior
    # Runs that reached the epoch budget (best epoch within PATIENCE of the last): a longer budget
    # could only lower their best validation errors (the best epoch is kept), not the others'
    _capped = {
        _k: [_r for _r in _v["runs"] if _r["epochs"] == EPOCHS] for _k, _v in training.items()
    }
    if CHECK_TRAINED:
        for _k, _runs in _ref.items():
            assert [_r["lr"] for _r in training[_k]["runs"]] == [_x[0] for _x in _runs], _k
            for _run, (_, _v, _e, _b) in zip(training[_k]["runs"], _runs, strict=True):
                assert abs(_run["val_rmse"] - _v) < NET_TOL, (_k, _run)
                assert abs(_run["epochs"] - _e) <= 10 and abs(_run["best_epoch"] - _b) <= 10, (
                    _k,
                    _run,
                )
        # The rates kept: GNN-12 1e-3, GNN-2 3e-3, GNN-5 3e-4, MLP 1e-2
        assert {_k: _v["lr"] for _k, _v in training.items()} == {
            "GNN, 12 layers": 1e-3, "GNN, 2 layers": 3e-3, "GNN, 5 layers": 3e-4, "MLP (no graph)": 1e-2}  # fmt: skip
        # The notes: the two-layer network stopped early (best at epoch 41), at every rate tried;
        # the five-layer network's kept run was still improving at epoch 197 of 200
        _two, _five = training["GNN, 2 layers"], training["GNN, 5 layers"]
        assert _two["best_epoch"] == 41 and all(_r["epochs"] < MAX_EPOCHS for _r in _two["runs"])
        assert _five["epochs"] == MAX_EPOCHS and _five["best_epoch"] == 197
        assert shown(training["GNN, 12 layers"]["val_rmse"], "0.23")
        # The slides' disclosure: GNN-12's kept run (1e-3) stopped early (best 178 of 193), but
        # both of its neighbours, and GNN-5's kept run, reached the 200-epoch budget (best epochs
        # 197-199), so the GNNs' errors are budget-limited. The MLP's kept run and both neighbours
        # stopped early (its 3e-4 run, two steps below, reached the budget too).
        _twelve = training["GNN, 12 layers"]
        assert (_twelve["epochs"], _twelve["best_epoch"]) == (193, 178)
        assert [(_r["lr"], _r["best_epoch"]) for _r in _capped["GNN, 12 layers"]] == [
            (3e-4, 197),
            (3e-3, 199),
        ]
        assert [(_r["lr"], _r["best_epoch"]) for _r in _capped["GNN, 5 layers"]] == [(3e-4, 197)]
        assert [_r["lr"] for _r in _capped["MLP (no graph)"]] == [3e-4] and not _capped[
            "GNN, 2 layers"
        ]
        _mlp_lrs = [_r["lr"] for _r in training["MLP (no graph)"]["runs"]]
        _i = _mlp_lrs.index(training["MLP (no graph)"]["lr"])
        assert all(
            _r["epochs"] < MAX_EPOCHS for _r in training["MLP (no graph)"]["runs"][_i - 1 : _i + 2]
        )
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for (_k, _v), _c, _ls in zip(training.items(), PALETTE, ("-", ":", "--", "-."), strict=True):
        _ax.plot(
            range(1, len(_v["curve"]) + 1),
            _v["curve"],
            color=_c,
            ls=_ls,
            lw=2,
            label=f"{_k} (rate {_v['lr']:g})",
        )
    _ax.set_yscale("log")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("validation head error, RMSE (m)")
    _ax.legend(frameon=False, fontsize=9)
    _rows = "\n    ".join(
        f"| {_k} | {_v['lr']:g} | {_v['epochs']} (best {_v['best_epoch']}) | "
        + (", ".join(f"{_r['lr']:g} (best {_r['best_epoch']})" for _r in _capped[_k]) or "none")
        + " |"
        for _k, _v in training.items()
    )
    mo.vstack([
        _fig,
        mo.md(f"""
    | Model | Rate kept | Its epochs | Runs that reached the {EPOCHS}-epoch budget |
    |:--|--:|--:|:--|
    {_rows}

    **Read the budget.** A run that reaches the last epoch allowed had its best epoch within
    {PATIENCE} epochs of the end, so it may still have been improving. A longer budget can only
    lower each run's best validation error (the best epoch's weights are kept), so it might move
    the search to another rate or a later epoch; the kept model's other errors (test, by area,
    closed pipe, higher demand, 2019) could then move either way. In the slides' run, the
    12-layer GNN kept 0.001, which stopped early, but both of its neighbours (0.0003 and 0.003)
    were still improving at the 200-epoch budget; the 5-layer GNN's kept run reached the budget
    itself. So the 12- and 5-layer GNNs' errors on the slides are budget-limited: they may be
    lower with a longer budget (only the validation error is guaranteed not to rise). The
    2-layer GNN's kept rate stopped early, and so did the MLP's and both of its neighbours.
    """),
    ])  # fmt: skip
    return


@app.cell
def _(
    CHECK_TRAINED,
    FEW_LAYERS,
    LAYERS,
    equivariance_check,
    g,
    mo,
    models,
    reach_check,
    scale,
    shown,
    snaps,
):
    # The checks again, on the trained networks (the slide's table)
    eq_trained = equivariance_check(models[f"GNN, {LAYERS} layers"], g, scale, snaps["test"])
    reach_trained = {
        k: reach_check(models[f"GNN, {k} layers"], g, scale, snaps["test"], k) for k in FEW_LAYERS
    }
    print(f"relabelled, trained 12 layers: {eq_trained:.2e} m; trained reach {reach_trained}")
    assert eq_trained < 1e-4
    # With the town-wide mean, a nudge at one sensor moves nodes beyond 2 links
    assert reach_trained[2]["moved"] > reach_trained[2]["within"] == 3
    # The slide: 8e-06; 783 of 785 nodes move; largest change beyond 2 links 2e-03 m
    if CHECK_TRAINED:
        assert shown(eq_trained, "8e-06") and reach_trained[2]["moved"] == 783
        assert shown(reach_trained[2]["max_change_beyond"], "2e-03")
        assert abs(reach_trained[2]["max_change_beyond"] - 0.00214385986328125) < 5e-4
    _two = reach_trained[2]
    mo.md(f"""
    On the trained networks: relabelling the nodes changes the 12-layer network's estimates by at
    most {eq_trained:.0e} m. Nudging sensor {_two["sensor"]} by 5 m moves {_two["moved"]} of 785 estimates of the trained
    2-layer network, though only {_two["within"]} nodes are within 2 links: the town-wide mean carries the nudge
    to the others, and only through that one average (largest change beyond 2 links:
    {_two["max_change_beyond"]:.0e} m).
    """)
    return


@app.cell
def _(g, np, nx, predict, scale, snaps, zone_graph):
    def nearest_observed(g, z, observed: np.ndarray) -> np.ndarray:
        """Each node's estimate: the mean head of its nearest observed nodes in the same zone.

        Returns a matrix W (N, N), the estimates being ``heads @ W.T``; a node with no observed
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

    def nearest_predict(g, z, snap, obs: np.ndarray) -> np.ndarray:
        """The no-learning baseline for snapshots that share one observed mask."""
        assert (obs == obs[0]).all()
        w = np.nan_to_num(nearest_observed(g, z, obs[0]))
        return np.nan_to_num(np.where(obs, snap.heads, 0.0)) @ w.T

    def all_predictions(models: dict, snap, obs: np.ndarray, graph=None) -> dict:
        """Every network's heads, then the nearest observed node in the zone."""
        graph = g if graph is None else graph
        out = {name: predict(m, graph, scale, snap, obs) for name, m in models.items()}
        out["nearest in zone"] = nearest_predict(g, zone_graph(graph), snap, obs)
        return out

    assert snaps["train"].heads.shape[1] == len(g.nodes)
    return all_predictions, nearest_observed, nearest_predict


@app.cell
def _(P_SERVICE, g, np, rmse):
    def scores(pred: np.ndarray, truth: np.ndarray, obs: np.ndarray) -> dict:
        """Head error at unobserved junctions, by links to the nearest sensor and by area; flags.

        A flag: estimated pressure (head - elevation) below 35 psi (24.6 m) at an unobserved
        junction in a snapshot; recall and precision against the simulated truth.
        """
        unobs = ~obs & (g.kind == 0)[None]
        err = pred - truth
        by_hops = [  # None where no unobserved junction is that far (0 links: the sensors)
            rmse(err[unobs & (g.hops == h)[None]], 0)
            if (unobs & (g.hops == h)[None]).any()
            else None
            for h in range(int(g.hops.max()) + 1)
        ]
        by_area = {name: rmse(err[unobs & (g.area == i)[None]], 0) for i, name in enumerate("ABC")}
        low_true = (truth - g.elev) < P_SERVICE
        low_pred = (pred - g.elev) < P_SERVICE
        tp = (low_true & low_pred & unobs).sum()
        by_area_low = {}  # per zone: the junction-hours truly below 35 psi, and the share flagged
        for i, name in enumerate("ABC"):
            low = low_true & unobs & (g.area == i)[None]
            n = int(low.sum())
            by_area_low[name] = {
                "low": n,
                "found": float((low & low_pred).sum() / n) if n else None,
            }
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

    return (scores,)


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    PCT_TOL,
    all_predictions,
    g,
    models,
    np,
    observe,
    scores,
    shown,
    snaps,
):
    te = snaps["test"]
    test_obs = observe(g, len(te.heads), None)  # every sensor read
    test_preds = all_predictions(models, te, test_obs)
    test_preds["training mean"] = np.broadcast_to(snaps["train"].heads.mean(0), te.heads.shape)
    test_scores = {k: scores(v, te.heads, test_obs) for k, v in test_preds.items()}
    _gnn, _mlp = test_scores[f"GNN, {LAYERS} layers"], test_scores["MLP (no graph)"]
    _near, _mean = test_scores["nearest in zone"], test_scores["training mean"]
    for _k, _v in test_scores.items():
        print(
            f"{_k:16s} rmse {_v['rmse']:.4f} m, flags found {_v['low_recall']:.4f}, right {_v['low_precision']:.4f}"
        )
    # The baselines with no learning: exact (numbers.json)
    assert (
        abs(_mean["rmse"] - 8.849180221557617) < EXACT
        and _mean["low_recall"] == _mean["low_precision"] == 0
    )
    assert (
        _mean["flags"] == 0
    )  # the training means flag nothing: "flags right" has no flags to be right
    assert abs(_near["rmse"] - 0.2468549095300912) < EXACT
    assert (
        abs(_near["low_recall"] - 0.9980822873082287) < EXACT
        and abs(_near["low_precision"] - 0.990484429065744) < EXACT
    )
    assert abs(_near["low_share"] - 0.010210947930574098) < EXACT
    _hops_near = [0.07197022404177203, 0.12054097869422177, 0.1725913592797331, 0.23927764906786111,
                  0.29971584384011896, 0.3248985300446993, 0.2993609598360723, 0.3518889758557218,
                  0.34398888777126935, 0.4676485228672438, 1.0717936179913592]  # fmt: skip
    assert _near["by_hops"][0] is None and np.allclose(
        _near["by_hops"][1:], _hops_near, rtol=0, atol=EXACT
    )
    # The slide's table: 8.85, 0%, 0%; 0.25, 100%, 99%; and 1.0% of junction-hours below 35 psi
    assert shown(_mean["rmse"], "8.85") and shown(100 * _mean["low_recall"], "0")
    assert (
        shown(_near["rmse"], "0.25")
        and shown(100 * _near["low_recall"], "100")
        and shown(100 * _near["low_precision"], "99")
    )
    assert shown(100 * _near["low_share"], "1.0")
    # The notes: the training means raise 0 flags ("none raised"); 5,736 junction-hours are low,
    # all in Area A
    assert _near["by_area_low"] == {"A": {"low": 5736, "found": _near["low_recall"]}, "B": {"low": 0, "found": None},
                                    "C": {"low": 0, "found": None}}  # fmt: skip
    assert _near["flags"] == 5780
    if CHECK_TRAINED:  # MLP 0.18, 99%, 97%; GNN 0.19, 99%, 99% (numbers.json)
        assert (
            abs(_mlp["rmse"] - 0.1762000024318695) < NET_TOL
            and abs(_gnn["rmse"] - 0.19186528027057648) < NET_TOL
        )
        assert (
            abs(_mlp["low_recall"] - 0.9902370990237099) < PCT_TOL
            and abs(_gnn["low_recall"] - 0.9886680613668062) < PCT_TOL
        )
        assert abs(_mlp["low_precision"] - 0.9676320272572402) < PCT_TOL
        assert abs(_gnn["low_precision"] - 0.9916069242874629) < PCT_TOL
        assert _mlp["flags"] == 5870 and _gnn["flags"] == 5719
        assert shown(_mlp["rmse"], "0.18") and shown(_gnn["rmse"], "0.19")
        assert shown(100 * _mlp["low_recall"], "99") and shown(100 * _mlp["low_precision"], "97")
        assert shown(100 * _gnn["low_recall"], "99") and shown(100 * _gnn["low_precision"], "99")
        # "close": one seed each, a gap of 1.6 cm (under 3 cm with one seed is close)
        assert 0 < _gnn["rmse"] - _mlp["rmse"] < 0.03
    return te, test_obs, test_preds, test_scores


@app.cell
def _(LAYERS, mo, test_scores):
    def flags(s: dict) -> str:
        """Flags found and flags right, in percent ("none raised" when there are no flags)."""
        right = f"{100 * s['low_precision']:.0f}%" if s["flags"] else "none raised"
        return f"{100 * s['low_recall']:.0f}% | {right}"

    _rows = [("Each junction's training mean", "training mean"), ("Nearest observed node in the zone", "nearest in zone"),
             ("MLP on the 33 readings", "MLP (no graph)"), (f"GNN, {LAYERS} layers", f"GNN, {LAYERS} layers")]  # fmt: skip
    mo.md(
        "### Fixed network: MLP and GNN are close\n\n"
        "| Test days (30, simulated), all 33 sensors | Head error, m | Low-pressure flags found | Flags right |\n"
        "|:------------------------------------------|--------------:|-------------------------:|------------:|\n"
        + "\n".join(
            f"| {_t} | {test_scores[_k]['rmse']:.2f} | {flags(test_scores[_k])} |"
            for _t, _k in _rows
        )
        + f"\n\n{100 * test_scores['nearest in zone']['low_share']:.1f}% of unmeasured junction-hours are below "
        f"35 psi; each junction's training mean raises {test_scores['training mean']['flags']} flags."
        " The MLP and the GNN are close: one training seed each, a gap of "
        f"{100 * (test_scores[f'GNN, {LAYERS} layers']['rmse'] - test_scores['MLP (no graph)']['rmse']):.1f} cm."
        " In the slides' run the GNN's errors here and below are budget-limited: both of the"
        " rates around its kept one were still improving at the 200-epoch budget (the table under"
        " the training curves), so they may be lower with a longer budget."
    )
    return (flags,)


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    attempt,
    g,
    mo,
    t_peak,
    te,
    test_obs,
    test_preds,
    test_scores,
):
    # Zone by zone: where does the overall gap between the MLP and the GNN come from?
    _unobs = ~test_obs & (g.kind == 0)[None]
    _name = f"GNN, {LAYERS} layers"
    _sse = {
        _k: {
            _a: float(((test_preds[_k] - te.heads)[_unobs & (g.area == _i)[None]] ** 2).sum())
            for _i, _a in enumerate("ABC")
        }
        for _k in (_name, "MLP (no graph)")
    }
    _n_unobs = int(_unobs.sum())
    _gap = {
        _a: (_sse[_name][_a] - _sse["MLP (no graph)"][_a]) / _n_unobs for _a in "ABC"
    }  # MSE, m^2
    _total_gap = test_scores[_name]["rmse"] ** 2 - test_scores["MLP (no graph)"]["rmse"] ** 2
    print(f"GNN minus MLP, mean squared error by area (m^2): {_gap}; overall {_total_gap:.5f}")
    # The answers to the attempt
    _est = {
        _k: float(test_preds[_k][t_peak, attempt["j"]] - attempt["elevation"])
        for _k in ("nearest in zone", "MLP (no graph)", _name)
    }
    print(f"attempt n213: truth {attempt['true_pressure']:.2f} m; estimates {_est}")
    assert abs(_est["nearest in zone"] - 37.847776989746095) < EXACT
    assert abs(test_scores["training mean"]["by_area"]["B"] - 0.006217288784682751) < 1e-6
    if CHECK_TRAINED:
        assert (
            abs(_est["MLP (no graph)"] - 37.377710919189454) < 10 * NET_TOL
            and abs(_est[_name] - 37.82784901123047) < 10 * NET_TOL
        )
        # The GNN is ahead in Areas A and B; Area C alone accounts for more than the whole gap
        assert _gap["A"] < 0 and _gap["B"] < 0 < _total_gap < _gap["C"]
    _w = {"MLP (no graph)": 701_201, _name: 358_849}
    mo.md(f"""
    **The answers to the attempt** (n213 at 21:00 on the first test day; truth {attempt["true_pressure"]:.2f} m):
    nearest observed node in the zone (n215, across nothing but pipes) {_est["nearest in zone"]:.2f} m,
    the GNN {_est[_name]:.2f} m, the MLP {_est["MLP (no graph)"]:.2f} m. Sensor n229 sits in Area A, across
    PRV-3, and Area B's heads barely move whatever happens upstream (their training means err
    {test_scores["training mean"]["by_area"]["B"]:.3f} m), so the head across the valve says little about them.

    **Where the small gap comes from.** Overall, the MLP ({_w["MLP (no graph)"]:,} weights) errs
    {test_scores["MLP (no graph)"]["rmse"]:.2f} m and the GNN ({_w[_name]:,} weights, shared by every node and
    link) {test_scores[_name]["rmse"]:.2f} m: close, with one training seed each. Split the mean squared error
    over the areas: the GNN minus the MLP is {_gap["A"]:+.4f} m² from Area A, {_gap["B"]:+.4f} m² from Area B,
    and {_gap["C"]:+.4f} m² from Area C (total {_total_gap:+.4f} m²). The GNN is ahead in A and B; Area C, behind
    the pump, decides the overall ranking. Next: zone by zone.
    """)
    return


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    draw_pipes,
    g,
    mo,
    np,
    pairwise,
    plt,
    shown,
    te,
    test_preds,
    test_scores,
):
    _name = f"GNN, {LAYERS} layers"
    _err = np.sqrt(
        ((test_preds[_name] - te.heads) ** 2).mean(0)
    )  # per node, over the test snapshots
    _jm = np.setdiff1d(g.junctions, g.sensors)
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    draw_pipes(_ax)
    _sc = _ax.scatter(
        *g.xy[_jm].T,
        c=_err[_jm],
        s=8,
        cmap="magma_r",
        vmin=0,
        vmax=float(np.percentile(_err[_jm], 99)),
        zorder=2,
    )
    _ax.scatter(*g.xy[g.sensors].T, s=50, marker="*", color="#0072B2", zorder=3)
    _fig.colorbar(
        _sc, ax=_ax, shrink=0.8, extend="max", label=f"GNN ({LAYERS} layers) head RMSE, m"
    )
    # Area C on the map: west and north of the rest (y grows northward)
    _c, _rest = g.xy[g.area == 2].mean(0), g.xy[(g.area == 0) | (g.area == 1)].mean(0)
    _med = {_a: float(np.median(_err[_jm][g.area[_jm] == _i])) for _i, _a in enumerate("ABC")}
    _top = _jm[_err[_jm] > np.percentile(_err[_jm], 99)]
    # The tank: Area C's heads move with it, and together
    _tank = int(np.flatnonzero(g.kind == 2)[0])
    _hc = te.heads[:, g.area == 2].astype(np.float64)
    _corr = [float(np.corrcoef(_hc[:, _k], te.heads[:, _tank])[0, 1]) for _k in range(_hc.shape[1])]
    # Area C's spread (numbers.json "area_c", float32 as the figure script): across its junctions
    # in one hour, mean over the hours; over the hours, mean over its junctions
    _c32 = te.heads[:, g.area == 2]
    _within, _over = float(_c32.std(1).mean()), float(_c32.std(0).mean())
    _s = {
        _k: test_scores[_k]["by_area"]
        for _k in (_name, "MLP (no graph)", "nearest in zone", "training mean")
    }
    print(f"median per-junction error by area {_med}; top 1% by area {np.bincount(g.area[_top] + 1, minlength=4)}; "
          f"Area C: corr with tank min {min(_corr):.4f}, spread within a snapshot {_within:.3f} m, over time {_over:.3f} m")  # fmt: skip
    assert _c[0] < _rest[0] and _c[1] > _rest[1]  # north-west
    assert min(_corr) > 0.99 and _within < 0.1 * _over
    assert abs(_within - 0.02740766853094101) < EXACT and abs(_over - 0.5220845937728882) < EXACT
    assert shown(_within, "0.027") and shown(_over, "0.52")  # the notes: 0.027 m in an hour, 0.52 m
    # The notes' by-area numbers: nearest sensor 0.269, 0.006, 0.033 m (exact); Area C heads
    # 0.52 m from their training means
    _near = _s["nearest in zone"]
    assert np.allclose([_near["A"], _near["B"], _near["C"]], [0.26891195679392804, 0.0064234235671018665, 0.03277254152307396],
                       rtol=0, atol=EXACT)  # fmt: skip
    assert shown(_near["A"], "0.269") and shown(_near["B"], "0.006") and shown(_near["C"], "0.033")
    assert shown(_s["training mean"]["C"], "0.52")
    _hops = test_scores[_name]["by_hops"]
    _mlp_hops = test_scores["MLP (no graph)"]["by_hops"][1:]
    # GNN 0.125, 0.021, 0.445 (alt text: 0.45 against 0.13); MLP 0.170, 0.223, 0.198; along the
    # pipes the GNN's error grows 0.14 -> 0.31 m, the MLP's stays at 0.17-0.19 m
    if CHECK_TRAINED:
        _ref = {_name: (0.12527620792388916, 0.020890511572360992, 0.4454874098300934),
                "MLP (no graph)": (0.17041343450546265, 0.22265096008777618, 0.19799402356147766)}  # fmt: skip
        for _k, _v in _ref.items():
            assert np.allclose([_s[_k]["A"], _s[_k]["B"], _s[_k]["C"]], _v, rtol=0, atol=NET_TOL)
        assert (
            shown(_s[_name]["A"], "0.125")
            and shown(_s[_name]["B"], "0.021")
            and shown(_s[_name]["C"], "0.445")
        )
        assert shown(_s[_name]["C"], "0.45") and shown(_s[_name]["A"], "0.13")
        assert shown(_s["MLP (no graph)"]["A"], "0.170") and shown(
            _s["MLP (no graph)"]["B"], "0.223"
        )
        assert shown(_s["MLP (no graph)"]["C"], "0.198")
        # Zone by zone the GNN is ahead in A and B, the MLP in C; both err more in C than the
        # nearest sensor does
        assert (
            _s[_name]["A"] < _s["MLP (no graph)"]["A"]
            and _s[_name]["B"] < _s["MLP (no graph)"]["B"]
        )
        assert _near["C"] < _s["MLP (no graph)"]["C"] < _s[_name]["C"]
        assert shown(_hops[1], "0.14") and shown(_hops[10], "0.31")
        assert shown(min(_mlp_hops), "0.17") and shown(max(_mlp_hops), "0.19")
        assert _med["C"] > max(_med["A"], _med["B"])  # darkest: Area C
        # The GNN in Area C: barely better than the training means (0.445 against 0.522 m)
        assert 0.75 * _s["training mean"]["C"] < _s[_name]["C"] < _s["training mean"]["C"]
        assert all(_a <= _b + 0.01 for _a, _b in pairwise(_hops[1:11]))  # grows with distance
    mo.vstack([
        _fig,
        mo.md(f"""
    ### The GNN errs most behind the pump

    | Head error at unmeasured junctions, m | Area A (29 sensors, 657 junctions) | Area B (1, 31) | Area C (3, 92) |
    |:--|--:|--:|--:|
    | GNN, {LAYERS} layers | {_s[_name]["A"]:.3f} | {_s[_name]["B"]:.3f} | {_s[_name]["C"]:.3f} |
    | MLP on the 33 readings | {_s["MLP (no graph)"]["A"]:.3f} | {_s["MLP (no graph)"]["B"]:.3f} | {_s["MLP (no graph)"]["C"]:.3f} |
    | Nearest observed node in the zone | {_near["A"]:.3f} | {_near["B"]:.3f} | {_near["C"]:.3f} |
    | Each junction's training mean | {_s["training mean"]["A"]:.3f} | {_s["training mean"]["B"]:.3f} | {_s["training mean"]["C"]:.3f} |

    Area C lies north-west of the rest, behind the pump and the tank; its junctions have the
    largest median error on the map ({_med["C"]:.2f} m, against {_med["A"]:.2f} in A and {_med["B"]:.2f} in B).
    Heads in Area C swing with the tank: every Area C junction's head correlates with the tank's
    at {min(_corr):.3f} or more, and they move together (spread within a snapshot {_within:.3f} m, over
    time {_over:.2f} m; standard deviations). So the nearest reading in the zone is nearly right
    ({_near["C"]:.3f} m); both networks err more there, the GNN most, barely better than the training
    means. Along the pipes the
    GNN's error grows with the distance to a sensor, {_hops[1]:.2f} m at 1 link to {_hops[10]:.2f} m at 10,
    while the MLP's stays at {min(_mlp_hops):.2f}–{max(_mlp_hops):.2f} m at every distance (§5). One town-wide
    number hides a zone served worse: report errors zone by zone.
    """),
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: predict four failures

    1. Use 2 message-passing layers instead of 12.
    2. Close one pipe (p182, 200 mm) on the test days.
    3. Raise the town's demand to ×1.5–1.8, beyond training's ×0.7–1.3.
    4. Estimate 2019's measured sensors, one hidden at a time, from the other 32.

    Write what happens to each, for the GNN and for the MLP. Two minutes alone; then four
    reveals.
    """)
    return


@app.cell
def _(
    CHECK_TRAINED,
    FEW_LAYERS,
    LAYERS,
    MUTED,
    NET_TOL,
    PALETTE,
    g,
    mo,
    np,
    plt,
    shown,
    test_scores,
    training,
):
    _counts = np.bincount(g.hops[g.junctions]).tolist()
    _styles = {
        "nearest in zone": (MUTED, "s", "--"),
        "MLP (no graph)": (PALETTE[1], "D", "-."),
        "GNN, 2 layers": (PALETTE[3], "v", ":"),
        "GNN, 5 layers": (PALETTE[2], "^", ":"),
        f"GNN, {LAYERS} layers": (PALETTE[0], "o", "-"),
    }
    _x = np.arange(1, len(_counts))
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for _k, (_c, _m, _ls) in _styles.items():
        _ax.plot(_x, test_scores[_k]["by_hops"][1:], color=_c, marker=_m, ls=_ls, lw=2, label=_k)
    for _k in FEW_LAYERS:
        _ax.axvline(_k + 0.5, color=MUTED, lw=0.8, ls=":")
    _ax.set_xticks(
        _x, [f"{_h}\n({_n})" for _h, _n in zip(_x, _counts[1:], strict=True)], fontsize=9
    )
    _ax.set_xlabel("links to the nearest sensor (junctions)")
    _ax.set_ylabel("head error, RMSE (m)")
    _ax.set_ylim(0, None)
    _ax.legend(frameon=False, fontsize=9, loc="upper left")
    _two, _five = test_scores["GNN, 2 layers"], test_scores["GNN, 5 layers"]
    _mlp, _mean = test_scores["MLP (no graph)"], test_scores["training mean"]
    # Beyond 2 links a 2-layer network hears no sensor, only the town-wide mean: compare it with
    # each junction's training mean at the same distances
    _beyond = [(_a, _b) for _a, _b in zip(_two["by_hops"][3:], _mean["by_hops"][3:], strict=True)]
    _curve5 = training["GNN, 5 layers"]["curve"]
    # Improvement over the last 25 epochs (none to measure in the reduced setting)
    _slow = min(_curve5[:-25]) - min(_curve5[-25:]) if len(_curve5) > 25 else float("nan")
    print(f"2 layers beyond 2 links vs training mean: {[(round(_a, 2), round(_b, 2)) for _a, _b in _beyond]}; "
          f"5 layers' kept run, last 25 epochs: {_slow:.4f} m")  # fmt: skip
    assert _counts[1:] == [69, 110, 134, 139, 104, 74, 56, 32, 21, 9, 1]
    assert np.allclose(_mean["by_hops"][3], 8.966877937316895, rtol=0, atol=1e-6)  # exact baseline
    assert shown(_mean["by_hops"][3], "8.97")  # the notes: each junction's training mean at 3 links
    _lr5 = {_r["lr"]: _r for _r in training["GNN, 5 layers"]["runs"]}
    _two_fit = training["GNN, 2 layers"]
    if CHECK_TRAINED:
        # The alt text and notes: 2 layers 0.48 m at 2 links, 1.69 at 3, overall 1.43; 5 layers 0.27
        # at 3, 0.46 at 5, then flat, overall 0.38; MLP near 0.18 at every distance
        assert shown(_two["by_hops"][2], "0.48") and shown(_two["by_hops"][3], "1.69")
        assert shown(_two["rmse"], "1.43") and abs(_two["rmse"] - 1.4323933124542236) < NET_TOL
        assert (
            shown(_five["by_hops"][3], "0.27")
            and shown(_five["by_hops"][5], "0.46")
            and shown(_five["rmse"], "0.38")
        )
        assert (
            max(_five["by_hops"][5:11]) - min(_five["by_hops"][5:11]) < 0.05
        )  # flat from 5 to 10 links
        assert all(abs(_v - 0.18) < 0.025 for _v in _mlp["by_hops"][1:])
        # Beyond its reach the 2-layer network has only the town-wide mean: far junctions at
        # 1.1-1.7 m, against 8.0-9.0 m for their training means
        assert all(_a < 0.25 * _b for _a, _b in _beyond[:-1])
        # The 5-layer network's kept run still improved at the cap, by under 0.01 m in 25 epochs;
        # it was 0.22 m above 12 layers' validation error
        assert (
            0 < _slow < 0.01
            and training["GNN, 5 layers"]["val_rmse"]
            - training[f"GNN, {LAYERS} layers"]["val_rmse"]
            > 0.2
        )
    mo.vstack([
        _fig,
        mo.md(f"""
    ### Two layers cannot see three links

    The jump sits exactly at the reach: beyond 2 links, a 2-layer network hears no sensor, only
    the town-wide mean, so its error leaps from {_two["by_hops"][2]:.2f} to {_two["by_hops"][3]:.2f} m (overall
    {_two["rmse"]:.2f} m). That mean is not nothing: at 3 to 10 links the 2-layer network errs
    {min(_a for _a, _ in _beyond[:-1]):.1f}–{max(_a for _a, _ in _beyond[:-1]):.1f} m, the training means
    {min(_b for _, _b in _beyond[:-1]):.1f}–{max(_b for _, _b in _beyond[:-1]):.1f} m. Five layers move the step out:
    {_five["by_hops"][3]:.2f} m at 3 links, {_five["by_hops"][5]:.2f} m at 5, then flat (overall {_five["rmse"]:.2f} m). The
    two-layer network stopped early at every rate it tried (its kept run, rate {_two_fit["lr"]:g}, at its
    best at epoch {_two_fit["best_epoch"]}), so its step is not a short training budget. The five-layer network's kept
    run (rate 0.0003) was still improving at epoch {training["GNN, 5 layers"]["best_epoch"]} of 200, by {_slow:.4f} m over the last
    25 epochs, so its errors may be lower with a longer budget (as may 12 layers', §4); its run at 0.001 stopped early at
    {_lr5.get(1e-3, {}).get("val_rmse", float("nan")):.2f} m on validation, against 12 layers' {training[f"GNN, {LAYERS} layers"]["val_rmse"]:.2f} m. The MLP sees all 33 readings at once:
    distance means nothing to it. Rule: depth at least the most links to a sensor (11).
    """),
    ])  # fmt: skip
    return


@app.cell
def _(Scenario, TODAY, g, np, nx, simulate_many, sim_heads, scs, town, wn):
    def pick_closure(days: list, open_heads: np.ndarray) -> dict:
        """The pipe whose closing changes the heads most, among Area A's busiest non-bridge pipes.

        Candidates: the 10 pipes inside Area A with the largest mean flow on a nominal day whose
        removal leaves the network connected. Each is closed on the given validation days; the
        one that moves junction heads most (mean absolute change over those days) is the test's
        closed pipe. The test days are not used.
        """
        flow = town.run_settings(*TODAY).link["flowrate"].abs().mean()
        area_a = set(town.area_a)
        gx = nx.Graph()
        gx.add_edges_from(zip(g.src, g.dst, strict=True))
        bridges = {frozenset(e) for e in nx.bridges(gx)}
        index = {n: i for i, n in enumerate(g.nodes)}
        cand = [
            n
            for n in flow.sort_values(ascending=False).index
            if n in wn.pipe_name_list
            and {wn.get_link(n).start_node_name, wn.get_link(n).end_node_name} <= area_a
            and frozenset(
                (index[wn.get_link(n).start_node_name], index[wn.get_link(n).end_node_name])
            )
            not in bridges
        ][:10]
        closed = simulate_many(
            [Scenario(s.day, s.s1, s.s2, s.scale, s.seed, c) for c in cand for s in days]
        )
        closed = closed.reshape(
            len(cand), len(days), *open_heads.shape[1:]
        )  # (candidates, days, steps, N)
        change = np.abs(closed - open_heads[None])[..., g.junctions].mean(axis=(1, 2, 3))
        best = int(np.argmax(change))
        return {
            "pipe": cand[best],
            "candidates": cand,
            "mean_change": change.tolist(),
            "diameter_mm": wn.get_link(cand[best]).diameter * 1000,
            "length_m": wn.get_link(cand[best]).length,
        }

    # Three validation days choose the pipe; the 30 test days are then simulated with it closed
    closure = pick_closure(scs["val"][:3], sim_heads["val"][:3])
    shut = [Scenario(s.day, s.s1, s.s2, s.scale, s.seed, closure["pipe"]) for s in scs["test"]]
    shut_heads = simulate_many(shut)
    print({k: v for k, v in closure.items() if k != "mean_change"})
    assert (
        closure["pipe"] == "p182"
        and closure["diameter_mm"] == 200.0
        and abs(closure["length_m"] - 37.7973) < 1e-9
    )
    assert closure["candidates"] == [
        "p477",
        "p478",
        "p182",
        "p781",
        "p780",
        "p779",
        "p778",
        "p777",
        "p776",
        "p744",
    ]
    _ref = [0.018706517294049263, 0.018421949818730354, 5.406692981719971, 4.579799175262451, 4.565614700317383,
            4.546415328979492, 4.526033878326416, 4.506036281585693, 4.487488269805908, 3.991203546524048]  # fmt: skip
    assert np.allclose(closure["mean_change"], _ref, rtol=0, atol=1e-6)
    return closure, shut, shut_heads


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    PCT_TOL,
    Snapshots,
    all_predictions,
    closure,
    flags,
    g,
    mo,
    models,
    rmse,
    scores,
    shown,
    shut,
    shut_heads,
    te,
    test_obs,
    test_scores,
):
    cl = Snapshots.of(shut, shut_heads)
    _cut = g.without(closure["pipe"])
    assert len(_cut.src) == len(g.src) - 2
    _name = f"GNN, {LAYERS} layers"
    _told = all_predictions(
        {_name: models[_name]}, cl, test_obs, graph=_cut
    )  # the GNN and nearest, told
    _not = all_predictions(
        {_name: models[_name], "MLP (no graph)": models["MLP (no graph)"]}, cl, test_obs
    )
    closure_scores = {
        "GNN, told": scores(_told[_name], cl.heads, test_obs),
        "GNN, not told": scores(_not[_name], cl.heads, test_obs),
        "MLP (no graph)": scores(_not["MLP (no graph)"], cl.heads, test_obs),
        "nearest in zone": scores(_told["nearest in zone"], cl.heads, test_obs),
    }
    head_change = rmse(cl.heads[:, g.junctions], te.heads[:, g.junctions])
    _s = closure_scores
    print(
        {_k: (round(_v["rmse"], 4), round(_v["low_precision"], 4)) for _k, _v in _s.items()},
        f"head change {head_change:.4f} m",
    )
    # Exact: the heads move by 9.06 m (RMSE over the junctions and test days); 3.8% of
    # junction-hours below 35 psi; the nearest node in the (new) zone 0.83 m, 98% of flags right
    assert abs(head_change - 9.060975074768066) < 1e-6 and shown(head_change, "9.06")
    assert abs(_s["nearest in zone"]["rmse"] - 0.8315426342363479) < EXACT and shown(
        _s["nearest in zone"]["rmse"], "0.83"
    )
    assert abs(_s["nearest in zone"]["low_precision"] - 0.9800280636108513) < EXACT
    assert shown(100 * _s["nearest in zone"]["low_precision"], "98") and shown(
        100 * _s["nearest in zone"]["low_share"], "3.8"
    )
    if CHECK_TRAINED:  # told 0.45, 98%; not told 0.68, 98%; MLP 2.34, 79%
        _ref = {"GNN, told": (0.44827592372894287, 0.9822237049039699, "0.45", "98"),
                "GNN, not told": (0.6796607375144958, 0.9792022792022792, "0.68", "98"),
                "MLP (no graph)": (2.3394908905029297, 0.7943265209500513, "2.34", "79")}  # fmt: skip
        for _k, (_e, _p, _te, _tp) in _ref.items():
            assert (
                abs(_s[_k]["rmse"] - _e) < NET_TOL and abs(_s[_k]["low_precision"] - _p) < PCT_TOL
            ), _k
            assert shown(_s[_k]["rmse"], _te) and shown(100 * _s[_k]["low_precision"], _tp), _k
        # Told, the GNN errs least of the four; its error rises from 0.19 to 0.45 m
        assert (
            _s["GNN, told"]["rmse"]
            < _s["GNN, not told"]["rmse"]
            < _s["nearest in zone"]["rmse"]
            < _s["MLP (no graph)"]["rmse"]
        )
        assert shown(test_scores[_name]["rmse"], "0.19")
    _rows = [("GNN, told (the pipe removed from its graph)", "GNN, told"), ("GNN, not told", "GNN, not told"),
             ("MLP on the 33 readings", "MLP (no graph)"), ("Nearest observed node in the zone", "nearest in zone")]  # fmt: skip
    _open = {_k: test_scores[_k]["rmse"] for _k in (_name, "MLP (no graph)")}
    _table = "\n    ".join(f"| {_t} | {_s[_k]['rmse']:.2f} | {flags(_s[_k])} |" for _t, _k in _rows)
    mo.md(f"""
    ### A closed pipe: told, the GNN errs least

    | Pipe {closure["pipe"]} closed ({closure["diameter_mm"]:.0f} mm), 30 test days | Head error, m | Flags found | Flags right |
    |:-----------------------------------------------|--------------:|------------:|------------:|
    {_table}

    Closing the pipe moves the heads by {head_change:.2f} m (RMSE over the junctions and test days), and
    {100 * _s["nearest in zone"]["low_share"]:.1f}% of junction-hours are below 35 psi (1.0% with it open). No model saw a
    closed pipe in training. Given the new graph, the GNN's error rises from {_open[_name]:.2f} to
    {_s["GNN, told"]["rmse"]:.2f} m, the least of the four; not told, to {_s["GNN, not told"]["rmse"]:.2f} m: telling it the change
    of neighbours is worth {_s["GNN, not told"]["rmse"] - _s["GNN, told"]["rmse"]:.2f} m. The MLP has no way to be told; its error grows from
    {_open["MLP (no graph)"]:.2f} to {_s["MLP (no graph)"]["rmse"]:.2f} m, and {100 * _s["MLP (no graph)"]["low_precision"]:.0f}% of its flags are right
    (the rest send crews for nothing). The pipe: of Area A's ten busiest pipes whose loss keeps
    the network connected, the one whose closing moved the heads most on 3 validation days. (The
    GNN's errors are budget-limited in the slides' run, §4.)
    """)
    return closure_scores, head_change


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    PCT_TOL,
    all_predictions,
    flags,
    g,
    mo,
    models,
    np,
    observe,
    scores,
    shown,
    snaps,
    te,
    test_scores,
):
    _far = snaps["far"]
    _obs = observe(g, len(_far.heads), None)
    far_scores = {
        k: scores(v, _far.heads, _obs) for k, v in all_predictions(models, _far, _obs).items()
    }
    _name = f"GNN, {LAYERS} layers"
    _ratio = {
        _k: far_scores[_k]["rmse"] / test_scores[_k]["rmse"]
        for _k in (_name, "MLP (no graph)", "nearest in zone")
    }
    # Every estimator's error grows with demand, the no-learning one's too. The head lost along a
    # pipe, mean over pipes (both directions) and snapshots (numbers.json "pipe_drop"):
    _pipe = g.edge[:, 1] == 1  # pipes only: valves and the pump hold or add head
    pipe_drop = {
        _k: float(np.abs(_s.heads[:, g.src[_pipe]] - _s.heads[:, g.dst[_pipe]]).mean())
        for _k, _s in (("test", te), ("far", _far))
    }
    print(
        {_k: round(_v["rmse"], 4) for _k, _v in far_scores.items()},
        f"ratios {_ratio}; head drop per pipe {pipe_drop}",
    )
    _near = far_scores["nearest in zone"]
    assert abs(_near["rmse"] - 0.4695237216647936) < EXACT and shown(_near["rmse"], "0.47")
    assert (
        abs(_near["low_recall"] - 0.9899519440803844) < EXACT
        and abs(_near["low_precision"] - 0.98369670075246) < EXACT
    )
    assert shown(100 * _near["low_recall"], "99") and shown(100 * _near["low_precision"], "98")
    assert shown(100 * _near["low_share"], "3.7")
    assert _ratio["nearest in zone"] > 1.8 and shown(test_scores["nearest in zone"]["rmse"], "0.25")
    # The head lost along a pipe doubles: 0.029 to 0.058 m (exact: the simulated days)
    assert (
        abs(pipe_drop["test"] - 0.029310710728168488) < EXACT
        and abs(pipe_drop["far"] - 0.05832022428512573) < EXACT
    )
    assert (
        shown(pipe_drop["test"], "0.029")
        and shown(pipe_drop["far"], "0.058")
        and shown(pipe_drop["far"] / pipe_drop["test"], "2.0")
    )
    # Area C: 174 junction-hours below 35 psi; the nearest sensor in the zone finds them all
    assert (
        _near["by_area_low"]["C"] == {"low": 174, "found": 1.0}
        and _near["by_area_low"]["B"]["low"] == 0
    )
    if CHECK_TRAINED:  # GNN 0.31, 97%, 99%; MLP 0.31, 98%, 95%
        _ref = {_name: (0.30568355321884155, 0.9743701762050386, 0.9930246363906203, "97", "99"),
                "MLP (no graph)": (0.3123057782649994, 0.9841269841269841, 0.9507597073719752, "98", "95")}  # fmt: skip
        for _k, (_e, _r, _p, _tr, _tp) in _ref.items():
            _v = far_scores[_k]
            assert (
                abs(_v["rmse"] - _e) < NET_TOL
                and abs(_v["low_recall"] - _r) < PCT_TOL
                and abs(_v["low_precision"] - _p) < PCT_TOL
            )
            assert (
                shown(_v["rmse"], "0.31")
                and shown(100 * _v["low_recall"], _tr)
                and shown(100 * _v["low_precision"], _tp)
            )
        # The notes: GNN 0.19 -> 0.31, MLP 0.18 -> 0.31; both networks' errors grow by less than the
        # baseline's; the two stay close (one seed each); in Area C the GNN flags 0% of the 174
        # low junction-hours, the MLP 6%
        assert shown(test_scores["MLP (no graph)"]["rmse"], "0.18") and shown(
            test_scores[_name]["rmse"], "0.19"
        )
        assert (
            _ratio[_name] < _ratio["nearest in zone"]
            and _ratio["MLP (no graph)"] < _ratio["nearest in zone"]
        )
        assert abs(far_scores[_name]["rmse"] - far_scores["MLP (no graph)"]["rmse"]) < 0.03
        assert far_scores[_name]["by_area_low"]["C"] == {"low": 174, "found": 0.0}
        assert shown(100 * far_scores["MLP (no graph)"]["by_area_low"]["C"]["found"], "6")
    _rows = [
        (f"GNN, {LAYERS} layers", _name),
        ("MLP on the 33 readings", "MLP (no graph)"),
        ("Nearest observed node in the zone", "nearest in zone"),
    ]
    _table = "\n    ".join(
        f"| {_t} | {far_scores[_k]['rmse']:.2f} | {flags(far_scores[_k])} |" for _t, _k in _rows
    )
    mo.md(f"""
    ### Higher demand: every method errs more

    | Demand ×1.5–1.8, 30 test days | Head error, m | Flags found | Flags right |
    |:------------------------------------------|--------------:|------------:|------------:|
    {_table}

    Thirty test days with the town's demand at ×1.5–1.8, beyond anything in training. Every
    estimator loses accuracy: the GNN goes from {test_scores[_name]["rmse"]:.2f} to {far_scores[_name]["rmse"]:.2f} m
    (×{_ratio[_name]:.2f}), the MLP from {test_scores["MLP (no graph)"]["rmse"]:.2f} to {far_scores["MLP (no graph)"]["rmse"]:.2f} m
    (×{_ratio["MLP (no graph)"]:.2f}), and the nearest node in the zone, which learns nothing and so
    extrapolates nothing, from {test_scores["nearest in zone"]["rmse"]:.2f} to {_near["rmse"]:.2f} m (×{_ratio["nearest in zone"]:.2f}).
    Even the method that learns nothing errs more, so these days are harder, not only unfamiliar:
    the head lost along a pipe goes from {pipe_drop["test"]:.3f} to {pipe_drop["far"]:.3f} m on average. Both networks'
    errors grow by less than the baseline's, so these days do not show a failure to extrapolate;
    to measure one, train a network on days at ×1.5–1.8 and compare (exercise 2). The GNN and the
    MLP stay close (one seed each; the GNN's error budget-limited in the slides' run, §4). More junctions fall below 35 psi
    ({100 * _near["low_share"]:.1f}% of junction-hours). One zone breaks the pattern: in Area C,
    {_near["by_area_low"]["C"]["low"]} junction-hours fall below 35 psi; the GNN flags
    {100 * far_scores[_name]["by_area_low"]["C"]["found"]:.0f}% of them, the MLP
    {100 * far_scores["MLP (no graph)"]["by_area_low"]["C"]["found"]:.0f}%, the nearest sensor
    {100 * _near["by_area_low"]["C"]["found"]:.0f}%.
    """)
    return far_scores, pipe_drop


@app.cell
def _(
    DAY_S, RAW, STEP_S, TODAY, Snapshots, g, nearest_observed, np, observe, pd, predict, rmse, town
):
    def scada(year: int, days: int, every: int) -> tuple:
        """BattLeDIM's measured heads: the sensors (pressure + elevation) and the tank.

        The first ``days`` days of the year, every ``every``-th 5-min reading. Reservoir heads
        are the file's; PRV-1 and PRV-2 at the file's settings. Unmeasured nodes are NaN.
        """
        read = {"sep": ";", "decimal": ",", "parse_dates": ["Timestamp"], "index_col": "Timestamp"}
        p = pd.read_csv(RAW / f"{year}_SCADA_Pressures.csv", **read)
        lv = pd.read_csv(RAW / f"{year}_SCADA_Levels.csv", **read)
        assert list(p.columns) == [g.nodes[i] for i in g.sensors]
        assert (
            p.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)
        ).all()  # every 5 minutes
        end = p.index[0] + pd.Timedelta(days=days)
        p = p[p.index < end].iloc[::every]
        lv = lv.loc[p.index]
        heads = np.full((len(p), len(g.nodes)), np.nan, dtype=np.float32)
        heads[:, g.sensors] = p.to_numpy() + g.elev[g.sensors]
        tank = int(np.flatnonzero(g.kind == 2)[0])
        heads[:, tank] = lv["T1"].to_numpy() + g.elev[tank]
        res = np.flatnonzero(g.kind == 1)
        heads[:, res] = g.elev[res]
        settings = np.tile(np.array(TODAY, dtype=np.float32), (len(p), 1))
        hour = (p.index.hour + p.index.minute / 60).to_numpy()
        return p.index, Snapshots(heads, settings, hour)

    def nominal_week() -> np.ndarray:
        """The hydraulic model alone: heads (steps, N) over the file's week, every 5 min.

        The file's demand patterns, the PRVs at the file's settings, the calibrated background
        leakage; no measurement enters.
        """
        opts = town.wn.options.time
        keep = opts.duration, opts.pattern_start
        try:
            opts.duration, opts.pattern_start = 7 * DAY_S, 0
            res = town.run_settings(*TODAY)
        finally:
            opts.duration, opts.pattern_start = keep
        return res.node["head"][list(town.wn.node_name_list)].to_numpy()

    def model_alone(week: np.ndarray, times, first_day: int) -> np.ndarray:
        """The nominal week's heads at the given times: pattern day = (weekday - first_day) mod 7."""
        day = (times.weekday.to_numpy() - first_day) % 7
        step = (times.hour.to_numpy() * 3600 + times.minute.to_numpy() * 60) // STEP_S
        return week[day * (DAY_S // STEP_S) + step]

    def align_weekday(week: np.ndarray) -> dict:
        """Which weekday the file's patterns start on, fitted on 2018's first eight weeks.

        The shift whose nominal sensor heads correlate best with 2018's measured ones; 2019, the
        test period, is not used.
        """
        times, real = scada(2018, 56, 1)
        corr = []
        for k in range(7):
            sim = model_alone(week, times, k)[:, g.sensors]
            a = sim - sim.mean(0)
            b = real.heads[:, g.sensors] - real.heads[:, g.sensors].mean(0)
            corr.append(float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())))
        return {"first_day": int(np.argmax(corr)), "corr": corr}

    def leave_one_out(sc, snap, models: dict, z, *, alone: np.ndarray | None = None) -> dict:
        """Hide each sensor in turn; estimate it from the other 32. Errors (m) per method.

        Returns, per method: each sensor's RMSE over all snapshots (None where the method gives
        no estimate: the nearest-in-zone rule has none for a sensor alone in its zone), their
        median, the RMSE over every estimate, and how many sensors got none.
        """
        out = {k: [] for k in [*models, "nearest in zone"]}
        if alone is not None:
            out["model alone"] = []
        for s in g.sensors:
            obs = observe(g, len(snap.heads), None, hide=int(s))
            truth = snap.heads[:, s]
            for name, m in models.items():
                out[name].append(predict(m, g, sc, snap, obs)[:, s] - truth)
            w = nearest_observed(g, z, obs[0])[s]  # NaN row: no observed node in the zone
            est = np.nan_to_num(snap.heads) @ np.nan_to_num(w) if np.isfinite(w).all() else np.nan
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

    return align_weekday, leave_one_out, model_alone, nominal_week, scada


@app.cell
def _(
    CHECK_TRAINED,
    EXACT,
    LAYERS,
    NET_TOL,
    PALETTE,
    RAW,
    REAL_DAYS,
    Snapshots,
    align_weekday,
    g,
    leave_one_out,
    mo,
    model_alone,
    models,
    nominal_week,
    np,
    pd,
    plt,
    scada,
    scale,
    shown,
    te,
    zg,
):
    _week = nominal_week()
    weekday = align_weekday(_week)
    _times, _real = scada(2019, REAL_DAYS, 12)
    # BattLeDIM's 2019 leaks in the same four weeks (the notes: "2019 also holds BattLeDIM's leaks")
    _leaks = pd.read_csv(
        RAW / "2019_Leakages.csv", sep=";", decimal=",", index_col=0, parse_dates=True
    )
    _leaks = _leaks[
        _leaks.index < _leaks.index[0] + pd.Timedelta(days=REAL_DAYS)
    ]  # every 5 minutes
    _active = sorted(_leaks.columns[(_leaks > 0).any()])
    _leak_flow = float(_leaks.sum(axis=1).mean())  # m3/h
    print(f"2019 leaks active in the window: {_active}, {_leak_flow:.1f} m3/h in all on average")
    assert (
        _active == ["p257", "p427", "p523", "p654", "p810", "p827"]
        and abs(_leak_flow - 41.175) < 0.01
    )
    _name = f"GNN, {LAYERS} layers"
    _pick = {_k: models[_k] for _k in ("MLP (no graph)", _name)}
    loo_sim = leave_one_out(
        scale, Snapshots(te.heads[::4], te.settings[::4], te.hour[::4]), _pick, zg
    )
    loo_real = leave_one_out(
        scale, _real, _pick, zg, alone=model_alone(_week, _times, weekday["first_day"])
    )
    _none = [
        g.nodes[_s]
        for _s, _p in zip(g.sensors, loo_real["nearest in zone"]["per_sensor"], strict=True)
        if _p is None
    ]
    print(
        f"weekday {weekday}; {len(_times)} hours, {_times[0]} to {_times[-1]}; no estimate for {_none}"
    )
    for _k, _v in loo_real.items():
        print(
            f"  2019 {_k:16s} median {_v['median']:.4f} m, rmse {_v['rmse']:.4f} m, no estimate {_v['no_estimate']}"
        )
    # Exact: the alignment, the hours, the model alone and the nearest node in the zone
    assert weekday["first_day"] == 0 and np.allclose(
        weekday["corr"], [0.7626447677612305, 0.7366626262664795, 0.7031451463699341, 0.6936308145523071,
                          0.6827698945999146, 0.674225926399231, 0.7126825451850891], rtol=0, atol=1e-6)  # fmt: skip
    assert (
        len(_times) == 672
        and str(_times[0]) == "2019-01-01 00:00:00"
        and str(_times[-1]) == "2019-01-28 23:00:00"
    )
    _alone, _near = loo_real["model alone"], loo_real["nearest in zone"]
    # The medians are over all sensors each method estimates: 33 for the networks and the
    # model alone, 32 for the nearest rule. The same 32 for every method:
    _same = {
        _k: float(np.median([_p for _p, _q in zip(loo_real[_k]["per_sensor"], _near["per_sensor"], strict=True)
                             if _q is not None]))
        for _k in ("model alone", "MLP (no graph)", _name)
    }  # fmt: skip
    print(f"medians on the 32 sensors the nearest rule estimates: {_same}")
    assert (
        abs(_alone["median"] - 0.2892395853996277) < 1e-6
        and abs(_alone["rmse"] - 0.31028276681900024) < 1e-6
    )
    assert (
        abs(_near["median"] - 0.10294749210899083) < EXACT
        and abs(_near["rmse"] - 0.179022276677614) < EXACT
    )
    assert _near["no_estimate"] == 1 and _none == ["n215"]  # the one sensor of Area B
    assert (
        abs(loo_sim["nearest in zone"]["rmse"] - 0.5018274740069988) < EXACT
        and loo_sim["nearest in zone"]["no_estimate"] == 1
    )
    # The alt text and notes: medians 0.29, 0.10; RMSE 0.31, 0.18
    assert shown(_alone["median"], "0.29") and shown(_near["median"], "0.10")
    assert shown(_alone["rmse"], "0.31") and shown(_near["rmse"], "0.18")
    assert shown(_near["median"], "0.103") and shown(_near["rmse"], "0.179")
    if CHECK_TRAINED:  # MLP 0.11 and 0.13, GNN 0.12 and 0.24; simulated hold-outs 0.17 and 0.25
        _ref = {"MLP (no graph)": (0.11171530932188034, 0.12983255088329315, 0.16592130064964294, "0.11", "0.13", "0.17"),
                _name: (0.12438133358955383, 0.23501810431480408, 0.25234660506248474, "0.12", "0.24", "0.25")}  # fmt: skip
        for _k, (_m, _r, _sm, _tm, _tr, _ts) in _ref.items():
            assert (
                abs(loo_real[_k]["median"] - _m) < NET_TOL
                and abs(loo_real[_k]["rmse"] - _r) < NET_TOL
            ), _k
            assert abs(loo_sim[_k]["rmse"] - _sm) < NET_TOL, _k
            assert (
                shown(loo_real[_k]["median"], _tm)
                and shown(loo_real[_k]["rmse"], _tr)
                and shown(loo_sim[_k]["rmse"], _ts)
            ), _k
        # The networks held up on real readings (real RMSE no worse than on simulated hold-outs),
        # and the no-training baseline matches them by median
        assert all(loo_real[_k]["rmse"] <= loo_sim[_k]["rmse"] for _k in ("MLP (no graph)", _name))
        assert _near["median"] <= min(loo_real[_k]["median"] for _k in ("MLP (no graph)", _name))
        # By median every method that reads the other sensors more than halves the model's error
        assert all(
            loo_real[_k]["median"] < 0.5 * _alone["median"]
            for _k in ("nearest in zone", "MLP (no graph)", _name)
        )
        # The notes' three decimals: medians 0.112 and 0.124 (within centimetres of the nearest
        # sensor's 0.103); by RMSE the MLP is lowest (0.130, nearest 0.179, GNN 0.235)
        _mlp_r = loo_real["MLP (no graph)"]
        assert shown(_mlp_r["median"], "0.112") and shown(loo_real[_name]["median"], "0.124")
        assert max(_mlp_r["median"], loo_real[_name]["median"]) - _near["median"] < 0.03
        assert shown(_mlp_r["rmse"], "0.130") and shown(loo_real[_name]["rmse"], "0.235")
        assert _mlp_r["rmse"] < _near["rmse"] < loo_real[_name]["rmse"]
        # Like for like, on the 32 sensors the nearest rule can estimate (without n215): MLP
        # 0.108, GNN 0.129; the nearest sensor's 0.103 is still within centimetres of both
        assert shown(_same["MLP (no graph)"], "0.108") and shown(_same[_name], "0.129")
        assert (
            _near["median"] <= min(_same.values()) and max(_same.values()) - _near["median"] < 0.03
        )
    _names = ["model alone", "nearest in zone", "MLP (no graph)", _name]
    _rng = np.random.default_rng(21)
    _fig, _ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for _i, _k in enumerate(_names):
        _per = np.array([_p for _p in loo_real[_k]["per_sensor"] if _p is not None])
        _ax.scatter(
            _i + _rng.uniform(-0.15, 0.15, len(_per)),
            _per,
            s=18,
            alpha=0.7,
            color=PALETTE[0],
            zorder=2,
        )
        _med = loo_real[_k]["median"]
        _ax.plot([_i - 0.3, _i + 0.3], [_med, _med], color=PALETTE[1], lw=3, zorder=3)
        _ax.text(_i + 0.33, _med, f"{_med:.2f}", va="center", fontsize=10)
    _ax.set_xticks(
        range(len(_names)),
        [
            _k if not loo_real[_k]["no_estimate"] else f"{_k}\n(1 sensor: no estimate)"
            for _k in _names
        ],
        fontsize=10,
    )
    _ax.set_yscale("log")
    _ax.set_ylabel("hidden sensor's head error,\nRMSE over 4 weeks (m)")
    _ax.set_xlim(-0.5, len(_names) - 0.2)
    mo.vstack([
        _fig,
        mo.md(f"""
    ### 2019: sensors beat the model alone

    Each of the 33 sensors hidden in turn, its head estimated hourly from the other 32 over
    2019's first four weeks ({len(_times)} hours). The hydraulic model alone uses no measurement:
    the file's demand patterns, aligned by weekday on 2018 (correlation {max(weekday["corr"]):.3f} at the
    best shift). Medians per sensor: the model alone {_alone["median"]:.2f} m; nearest in zone
    {_near["median"]:.2f} m; MLP {loo_real["MLP (no graph)"]["median"]:.2f} m; GNN {loo_real[_name]["median"]:.2f} m. Over all
    sensors and hours (RMSE): {_alone["rmse"]:.2f}, {_near["rmse"]:.2f} (without {_none[0]}, alone in Area B),
    {loo_real["MLP (no graph)"]["rmse"]:.2f}, and {loo_real[_name]["rmse"]:.2f} m. By median, reading the other sensors
    more than halves the model's error; by RMSE the GNN cuts it by
    {100 * (1 - loo_real[_name]["rmse"] / _alone["rmse"]):.0f}%. 2019 also holds BattLeDIM's leaks ({len(_active)} active in these four
    weeks, {_leak_flow:.0f} m³/h in all on average), which no method here knows about. The networks learned
    only from simulated days and held up on real ones (simulated hold-outs: MLP
    {loo_sim["MLP (no graph)"]["rmse"]:.2f}, GNN {loo_sim[_name]["rmse"]:.2f} m). By median the nearest sensor in the zone
    ({_near["median"]:.3f} m, on the 32 sensors it can estimate) is as good as both networks, within centimetres
    and with one seed per network (MLP {loo_real["MLP (no graph)"]["median"]:.3f}, GNN {loo_real[_name]["median"]:.3f} on all 33 sensors;
    {_same["MLP (no graph)"]:.3f} and {_same[_name]:.3f} on the same 32); by RMSE the MLP is lowest
    ({loo_real["MLP (no graph)"]["rmse"]:.3f}, nearest {_near["rmse"]:.3f}, GNN {loo_real[_name]["rmse"]:.3f}; the GNN's budget-limited in
    the slides' run, §4). On real data, test against a careful baseline.
    """),
    ])  # fmt: skip
    loo_hours = [str(_t) for _t in _times]  # the hours estimated, for the comparison below
    return loo_hours, loo_real, loo_sim, weekday


@app.cell
def _(CHECK_TRAINED, LAYERS, closure_scores, far_scores, loo_real, mo, shown, test_scores):
    _name = f"GNN, {LAYERS} layers"
    _low = {_k: {_a: _v["low"] for _a, _v in _s[_m]["by_area_low"].items()}
            for _k, _s, _m in (("test", test_scores, _name), ("closed pipe", closure_scores, "GNN, told"),
                               ("demand ×1.5–1.8", far_scores, _name))}  # fmt: skip
    _found = {
        _k: far_scores[_k]["by_area_low"] for _k in (_name, "MLP (no graph)", "nearest in zone")
    }
    print(
        f"junction-hours below 35 psi, by area: {_low}; demand x1.5-1.8, flags found by area: {_found}"
    )
    # On the test and closed-pipe days every junction-hour below 35 psi is in Area A; beyond the
    # training demand, 174 are in Area C (deterministic: the simulated truth)
    assert _low["test"] == {"A": 5736, "B": 0, "C": 0} and _low["closed pipe"] == {
        "A": 21575,
        "B": 0,
        "C": 0,
    }
    assert _low["demand ×1.5–1.8"] == {"A": 20427, "B": 0, "C": 174}
    assert loo_real["nearest in zone"]["no_estimate"] == 1
    _t, _c, _f = test_scores, closure_scores, far_scores

    def _pct(x: float | None) -> str:
        return "none to find" if x is None else f"{100 * x:.0f}%"

    # The table: 79% (98%); 0% of 174, 100%; 0.45 m against 0.03 m. The notes: 0.19 overall,
    # 0.45 in Area C, the MLP's 0.22 in Area B
    assert shown(100 * _f["nearest in zone"]["by_area_low"]["C"]["found"], "100")
    if CHECK_TRAINED:
        assert shown(100 * _c["MLP (no graph)"]["low_precision"], "79") and shown(
            100 * _c["GNN, told"]["low_precision"], "98"
        )
        assert shown(100 * _f[_name]["by_area_low"]["C"]["found"], "0")
        assert shown(_t[_name]["by_area"]["C"], "0.45") and shown(
            _t["nearest in zone"]["by_area"]["C"], "0.03"
        )
        assert shown(_t[_name]["rmse"], "0.19") and shown(
            _t["MLP (no graph)"]["by_area"]["B"], "0.22"
        )
    mo.md(f"""
    ## 6. Decide: who runs the low-pressure flags?

    | Do not use it alone when | Because, today |
    |:-------------------------|:---------------|
    | A pipe was closed and the MLP not retrained | {100 * _c["MLP (no graph)"]["low_precision"]:.0f}% of its flags right (GNN told: {100 * _c["GNN, told"]["low_precision"]:.0f}%) |
    | Demand beyond training, in Area C | GNN found {_pct(_found[_name]["C"]["found"])} of {_low["demand ×1.5–1.8"]["C"]} low junction-hours; the nearest sensor {_pct(_found["nearest in zone"]["C"]["found"])} |
    | A zone is fed by one tank (Area C) | GNN: {_t[_name]["by_area"]["C"]:.2f} m; the nearest sensor: {_t["nearest in zone"]["by_area"]["C"]:.2f} m |
    | A zone has one sensor | nearest in zone: no estimate for that sensor |

    The flags send crews: a false flag costs a crew's hours; a missed one leaves a customer below
    35 psi, where backflow can draw contaminated water into the mains. **Impact (social,
    equity):** a town-wide error hides the zones served worse (the GNN's {_t[_name]["rmse"]:.2f} m overall,
    budget-limited in the slides' run, is {_t[_name]["by_area"]["C"]:.2f} m in Area C; the MLP's is {_t["MLP (no graph)"]["by_area"]["B"]:.2f} m in Area B). On the test
    days every junction-hour below 35 psi is in Area A ({_low["test"]["A"]:,} of them, none in B or C), and
    so on the closed-pipe days; there the zones served worse lose accuracy, not flags. At higher
    demand, {_low["demand ×1.5–1.8"]["C"]} junction-hours of Area C drop below 35 psi: the GNN finds
    {_pct(_found[_name]["C"]["found"])} of them (Area A: {_pct(_found[_name]["A"]["found"])}), the MLP {_pct(_found["MLP (no graph)"]["C"]["found"])}
    ({_pct(_found["MLP (no graph)"]["A"]["found"])}), the nearest sensor in the zone {_pct(_found["nearest in zone"]["C"]["found"])}
    ({_pct(_found["nearest in zone"]["A"]["found"])}). Report and audit errors and flags zone by zone. The
    utility's operations log (which valves and pipes are closed) is part of the model: the GNN is
    only as good as the graph it is told.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: what estimates the pressures?

    - **A.** The MLP: lowest overall error (RMSE), simulated and real.
    - **B.** The GNN, told of every valve and pipe closure.
    - **C.** The nearest sensor in the zone: no training at all.
    - **D.** The hydraulic model alone, as today.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(LAYERS, closure_scores, loo_real, mo, test_scores):
    _name = f"GNN, {LAYERS} layers"
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        A has the lowest RMSE on a fixed network ({test_scores["MLP (no graph)"]["rmse"]:.2f} m, close to the GNN's
        {test_scores[_name]["rmse"]:.2f} m with one seed each) and on real readings ({loo_real["MLP (no graph)"]["rmse"]:.2f} m); with one
        pipe closed its error grows to {closure_scores["MLP (no graph)"]["rmse"]:.2f} m. B needs every closure recorded and
        fed to the graph; then it errs least of the four ({closure_scores["GNN, told"]["rmse"]:.2f} m, budget-limited: its
        training search hit the epoch budget, §4). C needs no training and
        matches the networks by median on real data ({loo_real["nearest in zone"]["median"]:.2f} m), but gives nothing for
        a sensor alone in its zone and {closure_scores["nearest in zone"]["rmse"]:.2f} m with the pipe closed. D errs most
        ({loo_real["model alone"]["median"]:.2f} m median). A defensible answer: B in the main zone where the closure
        log is reliable, C in the small zones (Area C: {test_scores["nearest in zone"]["by_area"]["C"]:.3f} m) and as a
        running cross-check; A only if retrained after every change to the network. The vote is
        about the utility's records as much as about the model.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where graph networks live

    | Today | What the graph is | Lecture |
    |:------|:------------------|:--------|
    | GraphCast weather forecasts | points on the globe, joined to their neighbours | today |
    | Google Maps arrival times | road segments, joined where they meet | today |
    | Traffic forecasting (DCRNN) | freeway sensors, joined by road distance | today, 20 |
    | Molecules and materials | atoms, joined by bonds | today |
    | Language models | tokens, every one joined to every other | 17, 19 |

    GraphCast (Lam et al., 2023) passes messages on a mesh over the globe and forecasts ten days
    of weather. Google Maps' arrival times come from a graph network over road segments
    (Derrow-Pinion et al., 2021). DCRNN (Li et al., 2018) forecasts freeway speeds with diffusion
    on the sensor graph, Lecture 20's sequences on a graph. Message passing began in chemistry,
    predicting molecules' properties (Gilmer et al., 2017). A transformer is a graph network on
    the complete graph: attention is a learned message from every token to every other.

    ## 7. Check yourself

    ### Exit question

    A consultant offers a graph network trained on another utility's network. What would you
    check before using it on yours, and what must be true of the two networks for it to transfer?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        That the other network's graph is built the same way (node and pipe features in the same
        units, valves and pumps as edges); that its depth reaches your farthest junction from a
        sensor (here 11 links, so at least 11 layers); and its error on simulated days of your own
        network, zone by zone, before any real use. The MLP cannot transfer at all: its first
        layer is wired to this town's 33 sensors. The GNN's rules are local (plus one town-wide
        mean), so they can run on any graph; whether they are right there is an empirical
        question, answered on your network's simulations.
        """)
        }
    )
    return


@app.cell
def _(
    CHECK_TRAINED,
    FEW_LAYERS,
    LAYERS,
    NET_TOL,
    Path,
    ROOT,
    agg_check,
    attempt,
    closure,
    closure_scores,
    eq_trained,
    far_scores,
    g,
    head_change,
    json,
    loo_hours,
    loo_real,
    loo_sim,
    np,
    nx,
    per_day,
    pipe_drop,
    reach,
    reach_local,
    reach_trained,
    scale,
    snaps,
    t_peak,
    te,
    test_preds,
    test_scores,
    training,
    weekday,
    zg,
):
    # Every number of the slides' numbers.json, against this run, read from the file (lesson:
    # never type a reference value). Those that do not depend on training must agree in either
    # setting; the trained networks' only in the slides setting.
    _j = g.junctions
    _zones = list(nx.connected_components(zg))
    _gx = nx.Graph()
    _gx.add_edges_from(zip(g.src, g.dst, strict=True))
    _c32 = te.heads[:, g.area == 2]
    numbers = {
        "graph": {
            "junctions": len(_j), "sensors": len(g.sensors), "links": len(g.src) // 2,
            "zones": len(_zones),
            "zone_junctions": sorted((sum(int(g.kind[_n] == 0) for _n in _c) for _c in _zones), reverse=True),
            "hops_max": int(g.hops.max()), "hops_median": float(np.median(g.hops[_j])),
            "hops_count": np.bincount(g.hops[_j]).tolist(),
            "area_junctions": {_a: int((g.area == _i).sum()) for _i, _a in enumerate("ABC")},
            "area_sensors": {_a: int((g.area[g.sensors] == _i).sum()) for _i, _a in enumerate("ABC")},
            "area_hops_median": {_a: float(np.median(g.hops[g.area == _i])) for _i, _a in enumerate("ABC")},
        },
        "data": {"snapshots": {_k: len(_v.heads) for _k, _v in snaps.items()}, "per_day": per_day,
                 "mu": scale.mu, "sd": scale.sd},
        "aggregation_check": {"max_abs_diff": agg_check["max_abs_diff"], "edges": agg_check["columns"],
                              "nodes": agg_check["rows"]},
        "training": training,
        "equivariance_max_diff": eq_trained,
        "reach": {"local": reach_local | {"layers": FEW_LAYERS[0]}}
        | {str(_k): reach_trained[_k] | {"layers": _k} for _k in FEW_LAYERS},
        "test": test_scores,
        "area_c": {"within_sd": float(_c32.std(1).mean()), "over_time_sd": float(_c32.std(0).mean())},
        "attempt": {_k: _v for _k, _v in attempt.items() if _k != "j"} | {
            "estimates": {_k: float(test_preds[_k][t_peak, attempt["j"]] - attempt["elevation"])
                          for _k in ("nearest in zone", "MLP (no graph)", f"GNN, {LAYERS} layers")}},
        "neighborhood_nodes": len(nx.single_source_shortest_path_length(_gx, attempt["j"], cutoff=3)),
        "closure": closure | closure_scores | {"head_change": head_change},
        "far": far_scores,
        "pipe_drop": pipe_drop,
        "loo": {"weekday": weekday, "sim": loo_sim, "real": loo_real, "real_snapshots": len(loo_hours),
                "real_start": loo_hours[0], "real_end": loo_hours[-1]},
        "picture_reach": {"2": reach[2], str(LAYERS): reach[LAYERS]},
    }  # fmt: skip

    def trained(path: str) -> bool:
        """Whether a numbers.json path depends on the trained networks."""
        top = ("/training/", "/equivariance_max_diff", "/reach/2/", "/reach/5/")
        return path.startswith(top) or "GNN" in path or "MLP" in path

    def pairs(mine: object, ref: object, path: str = ""):
        """Yield (path, mine, ref) for every leaf of the reference, timings left out.

        A key or list position missing from this run yields ``mine`` as the string "missing".
        """
        if isinstance(ref, dict):
            for key in ref:
                if "seconds" in key or key == "minutes":
                    continue
                sub = mine.get(key, "missing") if isinstance(mine, dict) else "missing"
                yield from pairs(sub, ref[key], f"{path}/{key}")
        elif isinstance(ref, list):
            seq = mine if isinstance(mine, list | tuple) else []
            for i, r in enumerate(ref):
                yield from pairs(seq[i] if i < len(seq) else "missing", r, f"{path}[{i}]")
        else:
            yield path, mine, ref

    def agrees(mine: object, ref: object, tol: float) -> bool:
        """Equal strings and None; numbers within ``tol`` (absolute)."""
        if isinstance(ref, str) or ref is None or mine is None or isinstance(mine, str):
            return mine == ref
        return abs(float(mine) - float(ref)) <= tol

    _path = ROOT / Path("lectures/build/fig/cee_l21_graph_networks/numbers.json")
    if not _path.exists():
        print(f"{_path} not found (build the slides' figures): comparison skipped")
    else:
        _ref = json.loads(_path.read_text(encoding="utf-8"))
        _all = list(pairs(numbers, _ref))
        _trained = [_x for _x in _all if trained(_x[0])]
        _fixed = [_x for _x in _all if not trained(_x[0])]
        # Fixed: the graph, EPANET's days, the baselines without learning, the real readings, the
        # untrained checks (float32 sums of simulated heads agree to 1e-6)
        _bad = [_x for _x in _fixed if not agrees(_x[1], _x[2], 1e-6)]
        print(
            f"compared {len(_fixed)} numbers that do not depend on training: {len(_bad)} differ {_bad[:5]}"
        )
        assert not _bad
        # Trained: within half a unit of the slides' last digit (epoch counts exactly, since the
        # slides quote best epochs), in the slides setting on the slides' GPU
        _tol = [(_p, _a, _b) for _p, _a, _b in _trained
                if not agrees(_a, _b, 0 if _p.endswith(("epochs", "epoch", "flags", "parameters", "moved",
                                                        "within", "farthest_moved")) else NET_TOL)]  # fmt: skip
        print(f"compared {len(_trained)} numbers of the trained networks: {len(_tol)} differ "
              f"(checked in the slides setting only) {_tol[:3]}")  # fmt: skip
        # How close, number by number: the count equal to the last bit, and the largest gap
        for _name, _xs in (("not depending on training", _fixed), ("trained", _trained)):
            _num = [(abs(float(_a) - float(_b)), _p) for _p, _a, _b in _xs
                    if isinstance(_b, int | float) and isinstance(_a, int | float)]  # fmt: skip
            _same = sum(_d == 0 for _d, _ in _num)
            print(f"  {_name}: {_same} of {len(_num)} numbers identical; largest gap {max(_num)}")
        if CHECK_TRAINED:
            assert not _tol, _tol[:10]
    return


@app.cell
def _(mo, train_minutes):
    mo.md(f"""
    ## Gotchas

    - **Each link both ways.** A pipe is two directed edges, so `index_add_` over `dst` sums the
      messages arriving from both sides; forgetting the reverse edges makes water information
      flow one way only.
    - **`index_add_` sums into its first argument.** Start from zeros of the nodes' shape and
      add along the node dimension (`dim=1` for a batch), indexed by each edge's destination.
    - **One standardization for heads and elevations,** so pressure (head − elevation) stays a
      difference the network can form. Standardize with the training days only.
    - **Score where nothing is observed.** Errors are computed at unobserved junctions; a sensor
      that is read is not an estimate.
    - **Depth is reach.** Fewer layers than the most links to a sensor leave junctions that hear
      nothing but the town-wide mean (§5).
    - **A closed pipe is a new graph:** remove both of its edges and re-index the valves' edges
      (`Graph.without`), and the nearest-in-zone baseline must use the new zones too.
    - **EPANET in processes, not threads,** each run in its own temporary directory (Lecture 12's
      lesson; joblib's workers here).
    - **`model.eval()` and `torch.no_grad()`** in every prediction; the best epoch's weights are
      copied (`state_dict` clones), not referenced.
    - **Time.** Training took {train_minutes:.0f} minutes in this run.

    ## Going deeper (Forsyth)

    - §17.1.2: stacked convolutional layers, each seeing a larger window of pixels than the one
      below: the receptive field. A message-passing layer does the same on a graph, one link per
      layer, which is why the depth must match the most links to a sensor.
    - §14.1.1–14.1.2: inference on a chain or a tree by dynamic programming, where each vertex
      passes a cost-to-go to its neighbour and the answer is exact when the graph is a forest.
      Graph networks keep the passing but learn the messages, and run a fixed number of rounds
      on a graph with loops (L-Town has many).
    - §15.1.2 and §15.3: a Markov random field couples each node's variable to its neighbours',
      and mean-field inference updates each node from its neighbours' current estimates, sweep
      after sweep. A GNN layer is one such sweep with learned update rules; Forsyth's
      $\\theta_{{ij}}$ coupling plays the part of our message function.
    - §16.3: layers and backpropagation; §18.1.9: residual networks, which add each layer's
      output to its input, as every layer here does ($h_i \\leftarrow h_i + U(\\cdot)$).

    **Exercises.** (1) Train the 12-layer network without the town-wide mean
    (`global_mean=False`) and compare its error by area and by links with the slides': what does
    the mean carry? (2) Simulate 240 training days at ×1.0–1.8 and train again: on the ×1.5–1.8
    test days, how much of the error is extrapolation, and how much is the harder hydraulics the
    no-learning baseline also faces?

    ## Things to remember

    - A graph layer: send, sum, update; shared rules.
    - $k$ layers reach $k$ links; plan depth by distance.
    - Beyond its reach a layer sees only the town-wide mean.
    - On a fixed network an MLP is as good; graphs pay when the network changes.
    - Check errors zone by zone; a baseline can win in one.

    **Next:** point clouds and 3-D scans.
    """)
    return


if __name__ == "__main__":
    app.run()
