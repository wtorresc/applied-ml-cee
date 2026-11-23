"""L-Town (BattLeDIM) as a hydraulic simulator: the two-valve pressure decision of Lecture 12.

Lecture 12 searches it with a Gaussian process; Lecture 14 trains a neural surrogate of it.

Data: ``data/raw/battledim/L-TOWN.inp`` (the EPANET model) and ``2018_Leakages.csv`` from
``scripts/fetch_battledim.sh`` (Vrachimis et al., 2022; CC BY 4.0). Runs use WNTR's
``EpanetSimulator`` (Klise et al., 2017); each run writes its scratch files to its own temporary
directory, removed when the run ends. WNTR's EPANET library is not thread-safe: run simulations
in parallel with processes, not threads (``run_many``, or ``init_worker`` and ``worker`` for a
pool of one's own).

The network: two reservoirs feed Area A (657 junctions) through pressure-reducing valves PRV-1
and PRV-2, set to 40 and 50 m in the file. Area A feeds Area B (31 junctions) through PRV-3 and
Area C (92 junctions) through a pump and tank T1. The decision is the pair of PRV-1/PRV-2
settings; one run simulates the peak-demand day of the file's one-week demand pattern.

Assumption, stated on the slides: background leakage at every Area A junction,
``q_i = C_i p_i**n1`` (an EPANET emitter), with ``C_i`` proportional to the pipe length the
junction serves (half of each pipe it touches) and the total scaled so that leakage at today's
settings equals BattLeDIM's mean 2018 leak flow. BattLeDIM's own leaks are bursts at given
pipes; this spreads the same volume as pressure-driven background leakage.

Solver: the file's hydraulic options are kept (accuracy 0.01, "Unbalanced Continue 10"), but a
run that EPANET reports as hydraulically unbalanced or unstable, or that stops early, raises
instead of returning partial or unconverged results. ``LTown(accuracy=1e-4)`` solves the same
model to a tighter tolerance, to check results that sit within centimetres of the limit.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
from threadpoolctl import threadpool_limits

from common import DATA_DIR, require

RAW: Path = DATA_DIR / "raw" / "battledim"
INP: Path = RAW / "L-TOWN.inp"
LEAKS_2018: Path = RAW / "2018_Leakages.csv"
VALVES: tuple[str, str] = ("PRV-1", "PRV-2")
TODAY: tuple[float, float] = (40.0, 50.0)  # PRV-1, PRV-2 settings in L-TOWN.inp, m
BOX: tuple[float, float] = (25.0, 60.0)  # the box of settings Lectures 12 and 14 search, m
PSI_M: float = 6894.757 / (1000.0 * 9.80665)  # metres of water per psi (0.7031)
P_SERVICE: float = 35 * PSI_M  # 35 psi normal working pressure (Ten States Standards), m
STEP_S: int = 300  # the file's hydraulic and pattern step, s
DAY_S: int = 86_400
STEPS: int = DAY_S // STEP_S + 1  # reported steps of one day, 00:00 to 24:00
# The Gaussian process both lectures fit: bounds in box widths (settings scaled to [0, 1]) and on
# the standardized output's variance; jitter on the standardized kernel's diagonal, not noise.
GP_ELL_BOUNDS: tuple[float, float] = (1e-2, 10.0)
GP_VAR_BOUNDS: tuple[float, float] = (1e-2, 1e2)
GP_JITTER: float = 1e-6
# EPANET warnings that mean the hydraulic solution itself failed (wntr.epanet.exceptions, 1 and 2).
SOLVER_WARNINGS: tuple[str, ...] = ("hydraulically unbalanced", "hydraulically unstable")

Pair = tuple[float, float]


@dataclass(frozen=True)
class Run:
    """One simulated day at one pair of valve settings.

    Attributes:
        leak: Mean background leakage in Area A, m^3/h.
        pmin: Lowest pressure at any customer junction of Areas A and B, m.
        node: The junction where ``pmin`` occurs.
        hour: Hour of the day when it occurs.
        open_steps: Reported steps (of ``STEPS``) at which PRV-1, PRV-2 are fully open.
        pmin_c: Lowest pressure in Area C (fed by the tank), m.
        closed_steps: Reported steps at which PRV-1, PRV-2 are closed: the other valve's head
            downstream is above this valve's setting, so it passes no water.
    """

    leak: float
    pmin: float
    node: str
    hour: float
    open_steps: tuple[int, int]
    pmin_c: float
    closed_steps: tuple[int, int]


def key(s1: float, s2: float) -> Pair:
    """The cache key of a pair of settings: each rounded to the micrometre."""
    return (round(float(s1), 6), round(float(s2), 6))


def mean_leak_2018() -> float:
    """Mean total leak flow of BattLeDIM's 2018 leaks, m^3/h (14 leaks, 5-minute readings)."""
    d = pd.read_csv(require(LEAKS_2018, "fetch_battledim.sh"), sep=";", decimal=",", index_col=0)
    assert d.shape == (105_120, 14), d.shape
    return float(d.sum(axis=1).mean())


def zones(wn: wntr.network.WaterNetworkModel) -> tuple[list[str], list[str], list[str]]:
    """Areas A, B, C: the parts joined by pipes alone (valves and the pump cut them apart).

    Returns:
        Junction names of Area A (fed by PRV-1/2), Area B (fed by PRV-3), Area C (fed by T1).
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


def peak_day(wn: wntr.network.WaterNetworkModel) -> int:
    """Day (0-6) of the one-week demand pattern with the largest total demand."""
    steps = DAY_S // STEP_S
    total = np.zeros(7 * steps)
    for _, j in wn.junctions():
        for d in j.demand_timeseries_list:
            mult = np.asarray(wn.get_pattern(d.pattern_name).multipliers, dtype=np.float64)
            total += d.base_value * np.resize(mult, total.size)  # a 1-step pattern is constant
    return int(total.reshape(7, steps).sum(axis=1).argmax())


def _customers(wn: wntr.network.WaterNetworkModel, names: list[str]) -> list[str]:
    """Junctions with any positive demand."""
    return [
        n for n in names if any(d.base_value > 0 for d in wn.get_node(n).demand_timeseries_list)
    ]


@dataclass
class LTown:
    """The L-Town model on its peak day, with calibrated background leakage in Area A.

    Usage::

        town = LTown(n1=1.0)
        run = town.run(36.0, 47.0)

    Attributes:
        n1: Leakage exponent (EPANET emitter exponent): 1 means leakage proportional to pressure.
        leak_target: Leakage at today's settings, m^3/h (default: the 2018 mean leak flow).
        accuracy: EPANET's hydraulic accuracy; None keeps the file's (0.01). The leakage is
            calibrated at the same accuracy, so a tighter one solves the same model more exactly.
    """

    n1: float = 1.0
    leak_target: float | None = None
    accuracy: float | None = None
    wn: wntr.network.WaterNetworkModel = field(init=False, repr=False)
    cache: dict[Pair, Run] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Load the model, find the zones and the peak day, and calibrate the leakage."""
        self.wn = wntr.network.WaterNetworkModel(str(require(INP, "fetch_battledim.sh")))
        self.area_a, self.area_b, self.area_c = zones(self.wn)
        self.customers = _customers(self.wn, self.area_a + self.area_b)
        self.day = peak_day(self.wn)
        assert self.day == 3, self.day
        opts = self.wn.options
        opts.time.duration = DAY_S
        opts.time.pattern_start = self.day * DAY_S
        opts.time.report_timestep = STEP_S
        opts.hydraulic.emitter_exponent = self.n1
        if self.accuracy is not None:
            opts.hydraulic.accuracy = self.accuracy
        self._length = self._served_length()
        self.coef = np.zeros(len(self.area_a))
        self.target = mean_leak_2018() if self.leak_target is None else self.leak_target
        self._calibrate()

    def _served_length(self) -> np.ndarray:
        """Pipe length each Area A junction serves (half of each pipe it touches), m."""
        index = {n: i for i, n in enumerate(self.area_a)}
        length = np.zeros(len(self.area_a))
        for _, p in self.wn.pipes():
            for end in (p.start_node_name, p.end_node_name):
                if end in index:
                    length[index[end]] += p.length / 2
        return length

    def _simulate(self, s1: float, s2: float) -> wntr.sim.SimulationResults:
        """Run EPANET with the valve settings and the current emitter coefficients.

        Raises:
            RuntimeError: If EPANET reports the hydraulics unbalanced or unstable, or the run
                stops before the end of the day.
        """
        for name, s in zip(VALVES, (s1, s2), strict=True):
            self.wn.get_link(name).initial_setting = float(s)
        for name, c in zip(self.area_a, self.coef, strict=True):
            self.wn.get_node(name).emitter_coefficient = float(c) if c > 0 else None
        sim = wntr.sim.EpanetSimulator(self.wn)
        # One temporary directory per run, removed when it ends (also in pool workers, which exit
        # without running finalizers). EPANET also writes a scratch file (enXXXXXX) to the working
        # directory: running inside the temporary directory keeps it there. chdir is per
        # process: parallel runs use processes.
        with tempfile.TemporaryDirectory(prefix="ltown-") as tmp, contextlib.chdir(tmp):
            res = sim.run_sim(file_prefix=str(Path(tmp) / "ltown"), convergence_error=True)
        failed = [w for w in sim.enData.errcodelist if any(s in w for s in SOLVER_WARNINGS)]
        if failed:
            raise RuntimeError(f"EPANET did not solve the hydraulics at ({s1}, {s2}): {failed}")
        return res

    def _leak(self, res: wntr.sim.SimulationResults) -> np.ndarray:
        """Background leakage of Area A at each step, m^3/s."""
        p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
        return (p**self.n1 * self.coef).sum(axis=1)

    def _calibrate(self) -> None:
        """Scale the emitter coefficients so leakage at today's settings equals the target.

        Leakage is nearly proportional to the total coefficient (pressure falls a little as
        leakage grows), so a few fixed-point steps suffice. The last step checks the formula
        against the emitter flow EPANET itself reports.

        Raises:
            RuntimeError: If the fixed-point steps do not converge.
        """
        res = self._simulate(*TODAY)
        demand = res.node["demand"][self.area_a].to_numpy()  # without emitters
        p = res.node["pressure"][self.area_a].to_numpy().clip(min=0.0)
        weight = self._length / self._length.sum()
        total = self.target / 3600 / float((p**self.n1 * weight).sum(axis=1).mean())
        for _ in range(20):
            self.coef = total * weight
            res = self._simulate(*TODAY)
            leak = float(self._leak(res).mean()) * 3600
            if abs(leak / self.target - 1) < 1e-6:
                break
            total *= self.target / leak
        else:
            raise RuntimeError(f"leakage calibration did not converge: {leak} vs {self.target}")
        reported = (res.node["demand"][self.area_a].to_numpy() - demand).sum(axis=1).mean()
        assert abs(reported * 3600 - leak) < 1e-3 * self.target, (reported * 3600, leak)
        self.cache.clear()

    def run(self, s1: float, s2: float) -> Run:
        """Simulate the peak day at PRV-1 = ``s1`` and PRV-2 = ``s2`` (m); results are cached."""
        k = key(s1, s2)
        if k not in self.cache:
            res = self._simulate(*k)
            p = res.node["pressure"][self.customers]
            low = p.min()
            node = str(low.idxmin())
            status = res.link["status"][list(VALVES)].to_numpy()
            self.cache[k] = Run(
                leak=float(self._leak(res).mean()) * 3600,
                pmin=float(low.min()),
                node=node,
                hour=float(p[node].idxmin()) / 3600,
                open_steps=tuple(int(v) for v in (status == 1).sum(axis=0)),
                pmin_c=float(res.node["pressure"][self.area_c].to_numpy().min()),
                closed_steps=tuple(int(v) for v in (status == 0).sum(axis=0)),
            )
        return self.cache[k]

    def customer_pressure(self, s1: float, s2: float) -> pd.Series:
        """Each customer's lowest pressure over the day (m), indexed by junction; not cached."""
        return self._simulate(*key(s1, s2)).node["pressure"][self.customers].min()

    def segments(self) -> list[tuple[tuple[float, float], tuple[float, float], str | None]]:
        """Every pipe as (start, end) coordinates and the area ("A", "B", "C") it belongs to."""
        area = dict.fromkeys(self.area_a, "A") | dict.fromkeys(self.area_b, "B")
        area |= dict.fromkeys(self.area_c, "C")
        out = []
        for _, p in self.wn.pipes():
            a, b = self.wn.get_node(p.start_node_name), self.wn.get_node(p.end_node_name)
            out.append((a.coordinates, b.coordinates, area.get(a.name) or area.get(b.name)))
        return out

    def outlet_xy(self, valve: str) -> tuple[float, float]:
        """Map coordinates of a valve's outlet (the junction it feeds)."""
        return self.wn.get_node(self.wn.get_link(valve).end_node_name).coordinates

    @property
    def runs(self) -> int:
        """Distinct settings simulated so far."""
        return len(self.cache)


# ---------------------------------------------------------------------------------------------
# Many independent runs in parallel processes (EPANET is not thread-safe)

_WORKER: LTown | None = None


def init_worker(n1: float = 1.0, accuracy: float | None = None) -> None:
    """Pool initializer: build the worker's calibrated model; one BLAS thread per worker."""
    global _WORKER  # noqa: PLW0603 (one model per worker process)
    threadpool_limits(1)
    _WORKER = LTown(n1=n1, accuracy=accuracy)


def worker() -> LTown:
    """The model ``init_worker`` built in this worker process.

    Raises:
        RuntimeError: If called outside a pool started with ``init_worker``.
    """
    if _WORKER is None:
        raise RuntimeError("no worker model: start the pool with initializer=init_worker")
    return _WORKER


def _run_pair(pair: Pair) -> Run:
    """Simulate one pair of settings in a worker."""
    return worker().run(*pair)


def run_many(
    pairs: Sequence[Pair], workers: int, n1: float = 1.0, accuracy: float | None = None
) -> list[Run]:
    """Simulate independent pairs of settings in ``workers`` processes, in the given order.

    Each worker calibrates its own copy of the model, so every run equals
    ``LTown(n1, accuracy=accuracy).run``.

    Args:
        pairs: PRV-1, PRV-2 settings, m.
        workers: Number of processes.
        n1: Leakage exponent.
        accuracy: EPANET's hydraulic accuracy (None: the file's).

    Returns:
        One ``Run`` per pair.
    """
    pairs = [(float(a), float(b)) for a, b in pairs]
    chunk = max(1, len(pairs) // (4 * workers))
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(n1, accuracy)) as ex:
        return list(ex.map(_run_pair, pairs, chunksize=chunk))
