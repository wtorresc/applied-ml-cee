"""Applied Machine Learning for CEE — Lecture 26: Decisions Over Time.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l26_reinforcement_learning.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: the Illinois National Bridge Inventory (FHWA; U.S. Government work): run
``bash scripts/fetch_nbi.sh`` once. The intersection runs in SUMO 1.28.0 (Eclipse SUMO, EPL-2.0)
through libsumo, both installed in the pixi environment from PyPI; its demand is assumed. The
slides' numbers come from ``lectures/build/fig/cee_l26_reinforcement_learning/`` (``make`` in
``lectures/``). Set ``L26_SETTING=reduced`` (the default), ``slides``, or ``smoke`` to choose how
much of the signal training to run (see the header cell); the choice never depends on the machine.
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
    # Applied Machine Learning for CEE — Lecture 26: Decisions Over Time

    **Slides:** *Applied Machine Learning for CEE*, Lecture 26 (`lectures/cee_l26_reinforcement_learning`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth's *Applied Machine Learning* (Springer 2019) has no chapter on reinforcement learning. Its §13.1 (Markov chains: transition matrices, absorbing states, recurrences) and §13.2.3 (dynamic programming as a cost-to-go) are the background. For the rest, Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., chapters 3, 4, 6, and 13; Mnih et al. (2015) on the deep Q-network; Schulman et al. (2017) on PPO (see Going deeper).

    **Data:** the National Bridge Inventory (FHWA; Illinois, 2014–2024, every two years; U.S. Government work), the records of Lectures 1, 3, and 6: `bash scripts/fetch_nbi.sh` writes `data/derived/nbi_il_bridges.csv`. The intersection: SUMO 1.28.0 (Eclipse SUMO, EPL-2.0; Lopez et al., 2018), driven through libsumo; the `eclipse-sumo` and `libsumo` wheels are in the pixi environment. Its demand is assumed, not counted. The slides' numbers: `lectures/build/fig/cee_l26_reinforcement_learning/numbers.json`, written by the figure script `lectures/figures/cee_l26_reinforcement_learning.py`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Everything is written here from scratch: the bridge MDP from the inventory's records (the
    two-year pairs, the do-nothing matrix, the replacement row), value iteration, policy
    evaluation, the critical cost of being poor, tabular Q-learning, the Markov test, the
    age-aware MDP, and the first-passage times; then the intersection in SUMO (the network with
    a fixed and an actuated program, the demand and the same demand reversed in time, an
    environment that decides every five seconds, the saturation flow, Webster's plans and a
    fixed plan tuned on training hours, the trip summary), a deep Q-network with replay and a
    target network, the same network trained online, and PPO. Each piece is checked three ways:

    - **Against the figure script's own functions**, loaded from its file as `ref`: the same
      matrices and values, the same SUMO network, the same observations, rewards, and lights for
      a fixed sequence of actions, the same saturation flow and plans, and, weight for weight,
      the same trained networks after a few training hours (one thread, deterministic
      algorithms, the same seeds).
    - **Against the slides' `numbers.json`**, read programmatically: every bridge
      number, the saturation flow, the plans, the tuning, and the untrained controllers' delays
      (Webster's plans, the tuned plan, actuated control) always; every signal number in the
      slides setting.
    - **Against the deck's own words:** every number the deck states is rendered here from this
      notebook's values and found word for word in the deck.

    **How much to run** is chosen by the environment variable `L26_SETTING` (or by editing
    `SETTING` below), never by the machine. Everything runs on the CPU; no GPU is used.

    - **`reduced`** (the default): one training run per method (seed 0, all 200 simulated
      hours), the learned controllers on three held-out hours, the untrained ones on all ten.
      It asserts what does not depend on how long the networks train: the bridge numbers in
      full, the network, the saturation flow, the plans and the tuning, the untrained
      controllers' delays on both demands, and the trained weights against the script's. The
      learned controllers' delays are reported beside the slides', not asserted; the snapshot
      cell says whether seed 0's DQN came out equal to the slides' run. Three worker
      processes; the three training runs go in parallel, so a laptop with four cores is
      enough.
    - **`slides`**: every run of the slides, as the figure script does them: three training runs
      per method, ten held-out hours on both demands, the dead-detector test, in a process pool
      of nine workers. Every signal number must equal `numbers.json`, the dead detectors
      included.
    - **`smoke`**: four training hours and one held-out hour, to check that the notebook runs
      end to end (a few minutes); its learned numbers mean nothing.

    Each run prints its own times at the end (Gotchas, "Time"): the bridge part, the SUMO
    checks, the training, the fixed plan's tuning, and the evaluation. Memory at the peak, over
    all processes (November 21, 2026): 1.9 GB for the reduced setting, 3.9 GB for the slides
    setting's nine workers.

    **Your tasks** (from the slides): decide whether to replace a poor bridge now or in two years;
    choose a discount rate; derive the Bellman equation and solve the MDP by value iteration;
    predict and check the policy; learn the same policy by Q-learning without the transition
    matrix; build a signal controller in SUMO, train a DQN, an online DQN, and PPO, and compare
    them with Webster's plans, a tuned fixed plan, and actuated control, on the demand they
    trained on and on that demand reversed; break the bridge MDP (is the rating Markov?) and the
    DQN (dead detectors); and decide what the city should install.
    """)
    return


@app.cell
def _():
    import importlib.metadata
    import importlib.util
    import json
    import os
    import random
    import re
    import subprocess
    import sys
    import tempfile
    import time
    import xml.etree.ElementTree as ET
    from bisect import bisect_right
    from itertools import pairwise
    from pathlib import Path

    # PYTHONHASHSEED only reaches processes started from here (the pool's workers)
    os.environ["PYTHONHASHSEED"] = "0"

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from joblib import Parallel, delayed
    from matplotlib.patches import FancyBboxPatch
    from torch import nn

    T_START = time.perf_counter()
    SEED = 26

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)

    def find_root() -> Path:
        """The repository's root, from the repo root, ``notebooks/``, or ``ipynb/``.

        Returns:
            The first of the working directory and its parent that holds ``data/`` and
            ``lectures/``.

        Raises:
            FileNotFoundError: If neither does (run from inside the repository).
        """
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / "data").is_dir() and (base / "lectures").is_dir():
                return base
        raise FileNotFoundError(
            "data/ and lectures/ not found: run from the repository, notebooks/, or ipynb/"
        )

    ROOT = find_root()
    # The figure scripts' shared modules on the path, and the slides' figure script itself, whose
    # functions are the reference for this notebook's. It has this notebook's file name, so it is
    # loaded from its path under another module name (an import of the same name fails marimo's
    # check).
    FIGURES = ROOT / "lectures" / "figures"
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    _spec = importlib.util.spec_from_file_location(
        "l26_figures", FIGURES / "cee_l26_reinforcement_learning.py"
    )
    ref = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = ref
    _spec.loader.exec_module(ref)

    # ------------------------------------------------ the slides' constants, written out here
    # The bridge MDP. A state is a bridge's lowest rating (Items 58-60; 3 stands for 3 or
    # worse); a step is one two-year inspection cycle; every cost is in replacements.
    RATINGS = np.arange(3, 10)
    N_S = len(RATINGS)
    YEARS_PER_STEP = 2
    GAP_MONTHS = (18, 30)  # a two-year pair: its two inspections 18 to 30 months apart
    RATES = {"2%": 0.02, "7%": 0.07}  # OMB Circular A-94: App. C (2026), and the 1992 base case
    REPLACE_COST = 1.0
    PENALTY = 0.10  # two years spent poor (a judgement, so it is swept)
    PENALTY_GRID = np.round(np.arange(0.0, 0.3001, 0.005), 3)
    POOR = 4  # FHWA: a lowest rating of 4 or less is poor
    VI_TOL = 1e-10
    NEW_AGE = 2  # a bridge inspected within two years of being built is new
    AGE_BANDS = (0, 20, 40, 60, 200)
    Q_STEPS = 2_000_000
    Q_EPSILON = 0.2
    Q_POWER = 0.7
    Q_CHECKPOINTS = (1_000, 10_000, 100_000, 1_000_000, 2_000_000)
    Z95 = 1.959963984540054  # the standard normal's 97.5% quantile (95% Wilson intervals)
    # The intersection. Four single-lane approaches of 300 m at 50 km/h, through traffic only.
    SPEED_MS = 13.89
    APPROACH_M = 300.0
    APPROACHES = ("N_in", "S_in", "E_in", "W_in")
    EXITS = {"N_in": "C_S", "S_in": "C_N", "E_in": "C_W", "W_in": "C_E"}
    NS, EW = 0, 1  # the two greens, which are the actions
    GREEN_PHASE = {NS: 0, EW: 3}  # SUMO's phases: green, then its yellow (+1) and all-red (+2)
    YELLOW_S, ALL_RED_S, DECISION_S = 3, 2, 5
    LOST_S = 2 * (YELLOW_S + ALL_RED_S)
    HOUR_S = 3600
    CLEAR_S = 900
    END_S = HOUR_S + CLEAR_S  # SUMO's --end: no run steps past it
    # (begin s, end s, north-south veh/h per approach, east-west veh/h per approach): assumed
    DEMAND = (
        (0, 900, 600, 250),
        (900, 1800, 700, 300),
        (1800, 2700, 400, 500),
        (2700, 3600, 300, 600),
    )
    # The shifted-demand test: the same quarter hours in reverse order (east-west's peak first)
    REVERSED = tuple(
        (b, e, ns, ew) for (b, e, _, _), (_, _, ns, ew) in zip(DEMAND, DEMAND[::-1], strict=True)
    )
    ROUTES = ("demand", "reversed")
    OBS_SCALE = 40.0
    TRAIN_SEEDS = (0, 1, 2)
    EVAL_SEEDS = tuple(range(1000, 1010))
    EPISODES = 200
    HIDDEN = 64
    GAMMA_RL = 0.99
    DQN_LR, DQN_BATCH, DQN_BUFFER, DQN_TARGET_EVERY = 1e-3, 64, 50_000, 500
    EPS_END, EPS_DECISIONS = 0.05, 30_000
    PPO_CLIP, PPO_LAMBDA, PPO_EPOCHS, PPO_MINIBATCH = 0.2, 0.95, 10, 256
    PPO_LR_ACTOR, PPO_LR_CRITIC, PPO_ENTROPY, PPO_EPISODES_PER_UPDATE = 3e-4, 1e-3, 0.01, 2
    REWARD_SCALE = 100.0
    # The tuned fixed-time plan: Webster's 15-minute greens times one factor, chosen on the
    # first ten training hours of run 0 by a bracketing search along the ladder
    TUNE_HOURS = 10
    TUNE_LADDER = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0)
    TUNE_START = (1.0, 1.2, 1.4, 1.6)
    STUCK = ("E_in", "W_in")
    FAULT_S = 300
    VIEW_M = 160  # the intersection's picture shows the 160 m nearest the junction
    _mine = {
        "SEED": SEED, "RATINGS": RATINGS, "N_S": N_S, "YEARS_PER_STEP": YEARS_PER_STEP,
        "GAP_MONTHS": GAP_MONTHS, "Z95": Z95, "END_S": END_S, "REVERSED": REVERSED,
        "ROUTES": ROUTES, "TUNE_HOURS": TUNE_HOURS, "TUNE_LADDER": TUNE_LADDER,
        "TUNE_START": TUNE_START, "VIEW_M": VIEW_M, "RATES": RATES, "REPLACE_COST": REPLACE_COST, "PENALTY": PENALTY,
        "PENALTY_GRID": PENALTY_GRID, "POOR": POOR, "VI_TOL": VI_TOL, "NEW_AGE": NEW_AGE,
        "AGE_BANDS": AGE_BANDS, "Q_STEPS": Q_STEPS, "Q_EPSILON": Q_EPSILON, "Q_POWER": Q_POWER,
        "Q_CHECKPOINTS": Q_CHECKPOINTS, "SPEED_MS": SPEED_MS, "APPROACH_M": APPROACH_M,
        "APPROACHES": APPROACHES, "EXITS": EXITS, "NS": NS, "EW": EW,
        "GREEN_PHASE": GREEN_PHASE, "YELLOW_S": YELLOW_S, "ALL_RED_S": ALL_RED_S,
        "DECISION_S": DECISION_S, "LOST_S": LOST_S, "HOUR_S": HOUR_S, "CLEAR_S": CLEAR_S,
        "DEMAND": DEMAND, "OBS_SCALE": OBS_SCALE, "TRAIN_SEEDS": TRAIN_SEEDS,
        "EVAL_SEEDS": EVAL_SEEDS, "EPISODES": EPISODES, "HIDDEN": HIDDEN, "GAMMA_RL": GAMMA_RL,
        "DQN_LR": DQN_LR, "DQN_BATCH": DQN_BATCH, "DQN_BUFFER": DQN_BUFFER,
        "DQN_TARGET_EVERY": DQN_TARGET_EVERY, "EPS_END": EPS_END,
        "EPS_DECISIONS": EPS_DECISIONS, "PPO_CLIP": PPO_CLIP, "PPO_LAMBDA": PPO_LAMBDA,
        "PPO_EPOCHS": PPO_EPOCHS, "PPO_MINIBATCH": PPO_MINIBATCH,
        "PPO_LR_ACTOR": PPO_LR_ACTOR, "PPO_LR_CRITIC": PPO_LR_CRITIC,
        "PPO_ENTROPY": PPO_ENTROPY, "PPO_EPISODES_PER_UPDATE": PPO_EPISODES_PER_UPDATE,
        "REWARD_SCALE": REWARD_SCALE, "STUCK": STUCK, "FAULT_S": FAULT_S,
    }  # fmt: skip
    for _name, _value in _mine.items():  # the same settings as the slides
        _theirs = getattr(ref, _name)
        if isinstance(_value, np.ndarray):
            assert np.array_equal(_value, _theirs), _name
        else:
            assert _value == _theirs, _name

    # The setting: "reduced" (the default), "slides", or "smoke"; never chosen by the machine
    SETTING = os.environ.get("L26_SETTING", "reduced").strip().lower()
    if SETTING not in {"reduced", "slides", "smoke"}:
        raise ValueError(f"L26_SETTING must be 'reduced', 'slides', or 'smoke', not {SETTING!r}")
    SLIDES = SETTING == "slides"
    RUN_TRAIN_SEEDS = TRAIN_SEEDS if SLIDES else TRAIN_SEEDS[:1]
    RUN_EVAL_SEEDS = {"slides": EVAL_SEEDS, "reduced": EVAL_SEEDS[:3], "smoke": EVAL_SEEDS[:1]}[
        SETTING
    ]
    RUN_EPISODES = 4 if SETTING == "smoke" else EPISODES
    # Training hours for the weight-for-weight check against the script (PPO updates every two)
    CHECK_EPISODES = {"dqn": 2, "online": 2, "ppo": 4}
    Q_CHECK_STEPS = 100_000  # Q-learning steps for the check against the script's function
    # Worker processes, each run on one thread; about 0.75 GB each (PyTorch). The count never
    # changes a result: every run is deterministic and independent of the others.
    WORKERS = min(9 if SLIDES else 3, os.cpu_count() or 1)
    KINDS = {"dqn": "DQN", "online": "DQN, online", "ppo": "PPO"}  # slug -> the slides' label
    # Every controller on the slides, untrained first, and the slug of each label
    CONTROLLERS = ("Webster, hour", "Webster, 15 min", "Fixed, tuned", "Actuated", *KINDS.values())
    UNTRAINED = CONTROLLERS[:4]
    CTRL_SLUG = {"Webster, hour": "webster_hour", "Webster, 15 min": "webster_15min",
                 "Webster, 15 min, retimed": "webster_15min_retimed", "Fixed, tuned": "fixed_tuned",
                 "Actuated": "actuated", **{v: k for k, v in KINDS.items()}}  # fmt: skip
    assert all(CTRL_SLUG[k] == ref.SLUG[k] for k in CTRL_SLUG)
    assert CONTROLLERS == ref.CONTROLLERS and KINDS == ref.KIND_LABEL
    SLUG = re.compile(r"[A-Za-z0-9_]+")  # SUMO splits option values at commas: slugs only

    # The slides' run, every value as the figure script wrote it: the reference
    # (read programmatically, never typed)
    BUILD = ROOT / "lectures" / "build" / "fig" / "cee_l26_reinforcement_learning"
    NUMBERS = BUILD / "numbers.json"
    if not NUMBERS.exists():
        raise FileNotFoundError(
            f"{NUMBERS} not found: build the slides' figures with `make` in lectures/ (the "
            "figure script takes about 7 minutes on a CPU)"
        )
    REF = json.loads(NUMBERS.read_text(encoding="utf-8"))
    DECK = re.sub(
        r"\s+",
        " ",
        (ROOT / "lectures" / "cee_l26_reinforcement_learning.md").read_text(encoding="utf-8"),
    )

    def rnd(v: float, nd: int) -> float:
        """Round once, as the figure script does (no signed zero).

        Args:
            v: The value.
            nd: Decimal places.

        Returns:
            A Python float.
        """
        return round(float(v), nd) + 0.0

    def pct(v: float, nd: int = 1) -> float:
        """A share as a rounded percentage.

        Args:
            v: A share in [0, 1].
            nd: Decimal places of the percentage.

        Returns:
            100 v, rounded.
        """
        return rnd(100 * float(v), nd)

    def slide(path: str) -> object:
        """The slides' value at ``path`` in numbers.json, e.g. "bridge/vi/2%/sweeps".

        Args:
            path: Keys separated by "/"; list indices as integers.

        Returns:
            The value.
        """
        node = REF
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def as_json(x: object) -> object:
        """``x`` as it reads back from JSON: string keys, lists, Python numbers.

        Args:
            x: Numbers, strings, None, NumPy scalars, and dicts, lists, and tuples of them.

        Returns:
            Its JSON round trip.
        """
        return json.loads(json.dumps(x, default=lambda o: o.item()))

    def agree(a: object, b: object, rel: float = 1e-12) -> bool:
        """Whether two JSON-like values are equal: the same keys, numbers to a relative ``rel``.

        Args:
            a: The computed value.
            b: The reference, the same structure.
            rel: Relative tolerance for floats (exact where the reference is 0).

        Returns:
            True if every pair agrees.
        """
        if isinstance(a, dict):
            return (isinstance(b, dict) and set(a) == set(b)
                    and all(agree(a[k], b[k], rel) for k in b))  # fmt: skip
        if isinstance(a, list):
            return (isinstance(b, list) and len(a) == len(b)
                    and all(agree(x, y, rel) for x, y in zip(a, b, strict=True)))  # fmt: skip
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return a == b
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return abs(a - b) <= rel * abs(b) if b != 0 else a == 0

    def first_difference(a: object, b: object, path: str = "") -> str | None:
        """Where two JSON-like values first differ, for an assert's message.

        Args:
            a: One value.
            b: The other.
            path: The path so far.

        Returns:
            The path and the two values, or None if they agree.
        """
        if isinstance(a, dict) and isinstance(b, dict):
            gone = "(missing)"
            pairs = [(f"{path}/{k}", a.get(k, gone), b.get(k, gone))
                     for k in sorted(set(a) | set(b), key=str)]  # fmt: skip
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            pairs = [(f"{path}[{i}]", x, y) for i, (x, y) in enumerate(zip(a, b, strict=True))]
        else:
            return None if agree(a, b) else f"{path}: {a!r:.200} against {b!r:.200}"
        for p, x, y in pairs:
            if (d := first_difference(x, y, p)) is not None:
                return d
        return None

    def check(x: object, path: str) -> None:
        """Assert that ``x`` equals the slides' value at ``path``.

        Args:
            x: The computed value.
            path: Its key in numbers.json.

        Raises:
            AssertionError: Saying where the two first differ.
        """
        a = as_json(x)
        assert agree(a, slide(path)), first_difference(a, slide(path), path)

    def label(r: int) -> str:
        """A rating as the slides write it ("≤3" for 3 or worse)."""
        return "≤3" if r == RATINGS[0] else str(int(r))

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    INK, MUTED = "#1A1A1A", "#595959"
    FIG = (9.0, 3.6)
    print(
        f"setting {SETTING!r}: train seeds {RUN_TRAIN_SEEDS}, {len(RUN_EVAL_SEEDS)} held-out "
        f"hours, {RUN_EPISODES} training hours a run, {WORKERS} workers; root {ROOT}"
    )
    return (
        AGE_BANDS,
        ALL_RED_S,
        APPROACHES,
        APPROACH_M,
        CHECK_EPISODES,
        CLEAR_S,
        CONTROLLERS,
        CTRL_SLUG,
        DECISION_S,
        DECK,
        DEMAND,
        DQN_BATCH,
        DQN_BUFFER,
        DQN_LR,
        DQN_TARGET_EVERY,
        EPISODES,
        EPS_DECISIONS,
        END_S,
        EPS_END,
        ET,
        EVAL_SEEDS,
        EW,
        EXITS,
        FAULT_S,
        FIG,
        FancyBboxPatch,
        GAMMA_RL,
        GAP_MONTHS,
        GREEN_PHASE,
        HIDDEN,
        HOUR_S,
        INK,
        KINDS,
        LOST_S,
        MUTED,
        NEW_AGE,
        NS,
        N_S,
        OBS_SCALE,
        PALETTE,
        PENALTY,
        PENALTY_GRID,
        POOR,
        PPO_CLIP,
        PPO_ENTROPY,
        PPO_EPISODES_PER_UPDATE,
        PPO_EPOCHS,
        PPO_LAMBDA,
        PPO_LR_ACTOR,
        PPO_LR_CRITIC,
        PPO_MINIBATCH,
        Parallel,
        Path,
        Q_CHECKPOINTS,
        Q_CHECK_STEPS,
        Q_EPSILON,
        Q_POWER,
        Q_STEPS,
        RATES,
        RATINGS,
        REPLACE_COST,
        REVERSED,
        REWARD_SCALE,
        ROOT,
        ROUTES,
        RUN_EPISODES,
        RUN_EVAL_SEEDS,
        RUN_TRAIN_SEEDS,
        SEED,
        SETTING,
        SLIDES,
        SLUG,
        SPEED_MS,
        STUCK,
        TRAIN_SEEDS,
        TUNE_HOURS,
        TUNE_LADDER,
        TUNE_START,
        T_START,
        UNTRAINED,
        VIEW_M,
        VI_TOL,
        WORKERS,
        YEARS_PER_STEP,
        YELLOW_S,
        Z95,
        agree,
        as_json,
        bisect_right,
        check,
        delayed,
        first_difference,
        importlib,
        label,
        nn,
        np,
        pairwise,
        pct,
        pd,
        plt,
        ref,
        rnd,
        slide,
        subprocess,
        tempfile,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–25

    On your own, no notes, two minutes:

    1. Lecture 6: a miss costs 20 false alarms. Where does the flagging threshold go?
    2. Lecture 13: what did the logistic regression predict about bridge decks?
    3. Lecture 25: what stood between the agent's answer and a report?

    **Today:** decisions whose consequences arrive years, or minutes, later.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. Flag when $p > 1/(1 + 20)$, about 0.048: the threshold that minimizes the expected cost
           of one decision. Today the decision repeats, and today's action changes the state the
           next decision faces.
        2. The chance that a deck loses a rating point by its next inspection. Today those
           chances become a transition matrix, and the question becomes what to do about them.
        3. Five checks and an engineer. Today's learned controller needs a check of its own.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    **When should Illinois replace a bridge?** Every Illinois highway bridge is inspected every
    two years and rated 0 to 9 for its deck, superstructure, and substructure; FHWA calls a
    bridge poor when its lowest rating is 4 or less. A replacement is expensive; a poor bridge
    costs its users (weight limits, detours, risk). The decision repeats at every inspection,
    and today's choice changes the bridge the next inspector sees. An error costs either way:
    replace too early and the money buys years a bridge still had; too late and people use a
    poor bridge for longer.

    The records: the National Bridge Inventory for Illinois, every two years from 2014 to 2024,
    one row per bridge and file, with its lowest rating (the `rating` column), the year it was
    built and last reconstructed (Item 106), and the year of the inspection.
    """)
    return


@app.cell
def _(GAP_MONTHS, NEW_AGE, N_S, RATINGS, ROOT, YEARS_PER_STEP, np, pd, ref, time):
    _t0 = time.perf_counter()
    NBI_CSV = ROOT / "data" / "derived" / "nbi_il_bridges.csv"
    if not NBI_CSV.exists():
        raise FileNotFoundError(f"{NBI_CSV} not found: run `bash scripts/fetch_nbi.sh`")

    def load_records(path: object) -> pd.DataFrame:
        """Bridge-year records with a lowest rating, sorted by bridge, then year.

        Args:
            path: The derived NBI table.

        Returns:
            One row per bridge and file; ``s`` is the lowest rating, 3 standing for 3 or worse.
        """
        df = pd.read_csv(path, dtype={"structure": str}).dropna(subset=["rating"])
        df = df.sort_values(["structure", "nbi_year"], kind="stable").reset_index(drop=True)
        df["s"] = df["rating"].clip(lower=RATINGS[0]).astype(int)
        assert df["s"].between(RATINGS[0], RATINGS[-1]).all()
        return df

    def rating_pairs(df: pd.DataFrame) -> pd.DataFrame:
        """Each record paired with the same bridge's record in the next file, one cycle on.

        The files are two years apart, but a bridge inspected every four years appears in two
        files with the same inspection, and then across a new inspection four years after the
        last. A pair counts as one two-year step only if its two inspections are 18 to 30
        months apart; that also leaves out the pairs of one inspection.

        Args:
            df: The records, sorted by bridge and year.

        Returns:
            One row per pair: the bridge, the year, the rating ``s`` and the next ``s2``, the
            months between the inspections, whether Item 106 (year reconstructed) changed
            between them (``work``), and the age.
        """
        nxt = df.groupby("structure", sort=False).shift(-1)  # the same bridge's next record
        months = (nxt["inspection_year"] - df["inspection_year"]) * 12 + (
            nxt["inspection_month"] - df["inspection_month"]
        )
        keep = (nxt["nbi_year"] == df["nbi_year"] + YEARS_PER_STEP) & months.between(*GAP_MONTHS)
        out = pd.DataFrame(
            {
                "structure": df["structure"],
                "nbi_year": df["nbi_year"],
                "s": df["s"],
                "s2": nxt["s"],
                "months": months,
                "work": nxt["year_reconstructed"].fillna(0) != df["year_reconstructed"].fillna(0),
                "age": df["inspection_year"] - df["year_built"],
            }
        )[keep]
        out["s2"] = out["s2"].astype(int)
        out["months"] = out["months"].astype(int)
        assert out["months"].between(*GAP_MONTHS).all()
        return out.reset_index(drop=True)

    def count_transitions(pairs: pd.DataFrame) -> np.ndarray:
        """Counts of (rating now, rating two years later).

        Args:
            pairs: Rows with ``s`` and ``s2``.

        Returns:
            An ``(N_S, N_S)`` float array, rows the rating now.
        """
        C = np.zeros((N_S, N_S))
        np.add.at(C, (pairs["s"].to_numpy() - RATINGS[0], pairs["s2"].to_numpy() - RATINGS[0]), 1)
        return C

    def bridge_mdp(df: pd.DataFrame, pairs: pd.DataFrame, rises: str = "drop") -> dict:
        """The two transition matrices: doing nothing (counted) and replacing (new bridges).

        Args:
            df: The records.
            pairs: The two-year pairs.
            rises: A rating that rose with no recorded work (unrecorded work, or inspectors
                disagreeing): "drop" the pair (the slides' MDP) or count it as a "stay".

        Returns:
            ``P`` (actions x states x states), the do-nothing counts ``C0``, the new bridges'
            first ratings ``p_new``, and the counts behind each choice.
        """
        no_work = pairs[~pairs["work"]]
        rose = no_work["s2"] > no_work["s"]  # a rise with no recorded work
        assert rises in {"drop", "stay"}
        if rises == "drop":
            kept = no_work[~rose]
        else:
            kept = no_work.assign(s2=no_work["s2"].where(~rose, no_work["s"]))
        C0 = count_transitions(kept)
        assert (C0.sum(1) > 0).all()
        P0 = C0 / C0.sum(1, keepdims=True)
        new = df[(df["inspection_year"] - df["year_built"]).between(0, NEW_AGE)]
        new = new.drop_duplicates("structure", keep="first")  # each new bridge once, first seen
        p_new = np.bincount(new["s"] - RATINGS[0], minlength=N_S).astype(float)
        p_new /= p_new.sum()
        P = np.stack([P0, np.tile(p_new, (N_S, 1))])
        assert np.allclose(P.sum(2), 1.0, rtol=0, atol=1e-12), "a row does not sum to one"
        assert np.array_equal(np.tril(P0), P0), "doing nothing never raises a rating"
        return {
            "P": P,
            "C0": C0,
            "p_new": p_new,
            "pairs": len(pairs),
            "work": int(pairs["work"].sum()),
            "no_work": len(no_work),
            "rose": int(rose.sum()),
            "used": int(C0.sum()),
            "new": len(new),
        }

    records = load_records(NBI_CSV)
    pairs = rating_pairs(records)
    mdp = bridge_mdp(records, pairs)
    P = mdp["P"]
    assert P.shape == (2, N_S, N_S)

    # The same records, pairs, and matrices as the figure script's own functions
    _r = ref.load_records()
    _p = ref.rating_pairs(_r)
    _m = ref.bridge_mdp(_r, _p)
    _cols = ["structure", "nbi_year", "s", "s2", "months", "work", "age"]
    pd.testing.assert_frame_equal(records[_r.columns], _r)
    pd.testing.assert_frame_equal(pairs[_cols], _p[_cols])
    assert np.array_equal(P, _m["P"]) and np.array_equal(mdp["C0"], _m["C0"])
    assert all(mdp[_k] == _m[_k] for _k in ("pairs", "work", "no_work", "rose", "used", "new"))
    _stay, _stay_ref = bridge_mdp(records, pairs, "stay"), ref.bridge_mdp(_r, _p, rises="stay")
    assert np.array_equal(_stay["P"], _stay_ref["P"])
    t_bridge_data = time.perf_counter() - _t0
    print(
        f"{len(records):,} records, {mdp['pairs']:,} two-year pairs, {mdp['used']:,} used; "
        f"equal to the script's ({t_bridge_data:.1f} s)"
    )
    return P, bridge_mdp, mdp, pairs, records, t_bridge_data


@app.cell
def _(FIG, PALETTE, POOR, RATINGS, check, label, np, plt, records):
    _last = records[records["nbi_year"] == records["nbi_year"].max()]
    _n = _last["s"].value_counts().reindex(RATINGS, fill_value=0)
    ratings_num = {
        "bridges": int(_n.sum()),
        "poor": int(_n[RATINGS <= POOR].sum()),
        "by_rating": {int(r): int(v) for r, v in _n.items()},
    }
    check(ratings_num, "bridge/ratings")
    # Illinois's 2024 ratings: where Q-learning's simulated inspections start
    start_2024 = _n.to_numpy(float) / _n.sum()
    _fig, _ax = plt.subplots(figsize=(FIG[0] * 0.6, FIG[1]))
    _ax.bar([label(r) for r in RATINGS], _n.to_numpy(),
            color=[PALETTE[1] if r <= POOR else PALETTE[0] for r in RATINGS])  # fmt: skip
    _ax.set(xlabel="lowest rating (Items 58-60)", ylabel="bridges, 2024",
            title=f"{ratings_num['bridges']:,} bridges; poor: {ratings_num['poor']:,}")  # fmt: skip
    assert np.isclose(start_2024.sum(), 1.0, rtol=0, atol=1e-12)
    _fig
    return ratings_num, start_2024


@app.cell
def _(PENALTY, RATES, YEARS_PER_STEP, mo):
    def gamma_of(rate: float) -> float:
        """The discount factor of one step: a real rate per year, compounded over two years.

        Args:
            rate: The real discount rate per year.

        Returns:
            (1 + rate) ** -2.
        """
        return (1.0 + rate) ** -YEARS_PER_STEP

    GAMMAS = {k: gamma_of(r) for k, r in RATES.items()}
    mo.md(f"""
    ### Your turn: replace now or wait?

    | A poor bridge: replace now, or in two years? | 2% a year | 7% a year |
    |:----------------------------------------------|----------:|----------:|
    | γ, the weight of a cost two years away | {GAMMAS["2%"]:.3f} | {GAMMAS["7%"]:.3f} |
    | Replace now: 1, plus {PENALTY:.2f} for this poor step | {1 + PENALTY:.2f} | {1 + PENALTY:.2f} |
    | Wait: {PENALTY:.2f} now, then {1 + PENALTY:.2f} in two years | ? | ? |

    Every cost is in units of one replacement, and a two-year step that starts poor costs
    {PENALTY:.2f}, whatever is done about it (the MDP's convention below). Two minutes, with a
    calculator: fill in the last row, and say which plan is cheaper at each rate.
    """)
    return GAMMAS, gamma_of


@app.cell
def _(GAMMAS, PENALTY, REPLACE_COST, check, mo, rnd):
    _now = REPLACE_COST + PENALTY  # replacing a poor bridge: the replacement and the poor step
    attempt = {k: {"now": rnd(_now, 3), "wait": rnd(PENALTY + g * _now, 3)}
               for k, g in GAMMAS.items()}  # fmt: skip
    check(attempt, "bridge/attempt")
    _w2, _w7 = (PENALTY + GAMMAS[k] * _now for k in ("2%", "7%"))
    assert _w2 > _now > _w7  # replace now at 2%, wait at 7%
    # The arithmetic as shown, from γ to four places, gives the same three decimals
    assert all(round(PENALTY + round(GAMMAS[k], 4) * _now, 3) == attempt[k]["wait"]
               for k in GAMMAS)  # fmt: skip
    mo.accordion(
        {
            "Answer": mo.md(f"""
        Waiting costs {PENALTY:.2f} now, plus replacing in two years, {_now:.2f} then (the
        replacement, and that step starts poor too), worth γ × {_now:.2f} today:
        {PENALTY:.2f} + {GAMMAS["2%"]:.4f} × {_now:.2f} = {_w2:.3f} at 2%, more than
        {_now:.2f}, so replace now; {PENALTY:.2f} + {GAMMAS["7%"]:.4f} × {_now:.2f} =
        {_w7:.3f} at 7%, so wait. The same bridge, the same costs, opposite answers: the
        discount rate decides. And this compares only two plans; waiting four years, or
        replacing the new bridge sooner in its own life, are plans too. A Markov decision
        process compares all of them at once.
        """)
        }
    )
    return (attempt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: an agent, a world, and later costs

    A **Markov decision process** (MDP): the agency (the agent) sees the bridge's **state**, its
    lowest rating; it chooses an **action**, do nothing or replace; the bridge and its next
    inspection (the environment) return a **cost** and the **next state**, two years later,
    drawn from $P(s' \mid s, a)$. "Markov": the next state depends only on the current state and
    action, not on how the bridge got there. CS 441 states the problem with rewards; an agency
    minimizes costs, which are rewards with the sign flipped.

    The objective is the expected **discounted** sum of costs, $\mathbb{E}\big[\sum_k \gamma^k
    c_k\big]$: a cost $k$ steps ahead weighs $\gamma^k$. A **policy** $\pi$ maps each state to an
    action; its **value** $V^\pi(s)$ is the expected discounted cost of following it from $s$.
    The goal: the policy with the smallest value in every state.
    """)
    return


@app.cell
def _(FancyBboxPatch, GAMMAS, INK, MUTED, PALETTE, np, plt):
    _fig, _ax = plt.subplots(figsize=(9.0, 3.6))
    _ax.set(xlim=(0, 1.6), ylim=(-0.05, 1.0), xticks=[], yticks=[])
    for _s in _ax.spines.values():
        _s.set_visible(False)

    def _box(xy: tuple[float, float], text: str, color: str, w: float) -> None:
        _ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - 0.09), w, 0.18,
                                     boxstyle="round,pad=0.01", facecolor=color,
                                     edgecolor=INK, lw=1))  # fmt: skip
        _ax.text(*xy, text, ha="center", va="center", fontsize=10, color=INK)

    def _arrow(a: tuple[float, float], b: tuple[float, float]) -> None:
        _ax.annotate("", b, a, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2})

    _box((0.32, 0.78), "agency (agent)\npolicy π(rating)", "#e3f1e8", 0.34)
    _box((0.32, 0.22), "bridge + inspection\n(environment)", "#e8f0f7", 0.34)
    _arrow((0.22, 0.31), (0.22, 0.69))
    _ax.text(0.20, 0.50, "state s:\nrating", ha="right", va="center", fontsize=9, color=MUTED)
    _arrow((0.42, 0.69), (0.42, 0.31))
    _ax.text(0.44, 0.50, "action a: do nothing\nor replace", ha="left", va="center",
             fontsize=9, color=MUTED)  # fmt: skip
    _arrow((0.49, 0.22), (0.72, 0.22))
    _ax.text(0.605, 0.27, "cost c(s, a); s' ~ P(s' | s, a)", ha="center", fontsize=9,
             color=MUTED)  # fmt: skip
    _box((0.86, 0.22), "two years later", "#e8f0f7", 0.26)
    # The weight of a cost k steps (2k years) ahead, at 2% a year
    _k = np.arange(6)
    _w = GAMMAS["2%"] ** _k
    _x = 1.12 + 0.075 * _k
    _ax.bar(_x, 0.35 * _w, width=0.055, bottom=0.45, color="#fbe3d4", edgecolor=INK)
    for _xi, _wi, _ki in zip(_x, _w, _k, strict=True):
        _ax.text(_xi, 0.45 + 0.35 * _wi + 0.02, f"{_wi:.2f}", ha="center", fontsize=8)
        _ax.text(_xi, 0.39, f"{2 * _ki}", ha="center", fontsize=8, color=MUTED)
    _ax.text(1.31, 0.31, "years from now", ha="center", fontsize=9, color=MUTED)
    _ax.text(1.31, 0.92, f"a cost's weight γ^k, γ = {GAMMAS['2%']:.3f}", ha="center",
             fontsize=10, color=PALETTE[1])  # fmt: skip
    _ax.text(0.32, 0.98, "V(s) = expected discounted cost of following π from s",
             ha="center", fontsize=9, color=PALETTE[2])  # fmt: skip
    _fig
    return


@app.cell
def _(FIG, P, POOR, RATINGS, check, label, mdp, mo, np, pct, plt, rnd):
    mdp_num = {
        "pairs": mdp["pairs"],
        "pairs_work": mdp["work"],
        "pairs_rose": mdp["rose"],
        "pairs_rose_pct": pct(mdp["rose"] / mdp["no_work"], 1),
        "pairs_used": mdp["used"],
        "new_bridges": mdp["new"],
        "p_stay": {int(r): rnd(P[0][i, i], 3) for i, r in enumerate(RATINGS)},
        "p_new": {int(r): rnd(v, 3) for r, v in zip(RATINGS, mdp["p_new"], strict=True)},
    }
    for _k, _v in mdp_num.items():
        check(_v, f"bridge/{_k}")
    # How far a rating falls in two years: the share of each row more than two ratings down
    _i, _j = np.indices(P[0].shape)
    far_drop = np.where(_i - _j > 2, P[0], 0).sum(1)  # per rating, the share falling 3 or more
    _fig, _axes = plt.subplots(1, 2, figsize=FIG, gridspec_kw={"width_ratios": [7, 1.6]})
    for _ax, _m, _title in ((_axes[0], P[0], "do nothing: rating in two years"),
                            (_axes[1], mdp["p_new"][:, None], "replace")):  # fmt: skip
        _ax.imshow(_m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (_a, _b), _p in np.ndenumerate(_m):
            if _p >= 0.005:
                _ax.text(_b, _a, f"{100 * _p:.0f}", ha="center", va="center", fontsize=8,
                         color="white" if _p > 0.5 else "black")  # fmt: skip
        _ax.set_title(_title, fontsize=10)
    _labels = [label(r) for r in RATINGS]
    _axes[0].set(xticks=range(len(RATINGS)), xticklabels=_labels, yticks=range(len(RATINGS)),
                 yticklabels=_labels, xlabel="rating two years later", ylabel="rating now")  # fmt: skip
    _axes[1].set(xticks=[], yticks=range(len(RATINGS)), yticklabels=_labels,
                 ylabel="first rating, %")  # fmt: skip
    mo.vstack([
        mo.md(f"""
    ### Two years of Illinois bridges

    Counted, not assumed. {mdp["pairs"]:,} pairs of a bridge's records in consecutive files,
    2014 to 2024, whose inspections were 18 to 30 months apart (a bridge inspected every four
    years would otherwise count four years of change as one step). Pairs with recorded work
    (Item 106, {mdp["work"]:,} of them) are not "do nothing". A rating that rose with no
    recorded work ({mdp["rose"]:,} pairs, {mdp_num["pairs_rose_pct"]}% of the rest) is
    unrecorded work or two inspectors disagreeing; those pairs are left out:
    {mdp["used"]:,} remain. The matching disagreements downward stay in, a bias toward
    deterioration; §3 measures how little it moves the policy. A replacement leads to the
    ratings that {mdp["new"]:,} new bridges received at their first inspection.

    Most of each row's mass is on the diagonal ({100 * P[0][1, 1]:.0f}% of 4s are 4s two years
    later, {100 * P[0][3, 3]:.0f}% of 6s, {100 * P[0][5, 5]:.0f}% of 8s); the rest falls one or
    two ratings (at ratings 4 to 8, at most {100 * far_drop[1:6].max():.1f}% falls further; at 9,
    {100 * far_drop[6]:.1f}%). Rating 9 rarely lasts:
    {100 * P[0][6, 6]:.0f}% of nines are still nines. The poor ratings, {POOR} and below, are
    where the decision will be made. Forsyth §13.1.1 writes the same object, a stochastic matrix
    with rows that sum to one, with the same convention: row = from, column = to.
    """),
        _fig,
    ])  # fmt: skip
    return far_drop, mdp_num


@app.cell
def _(GAMMAS, RATES, mo):
    mo.md(f"""
    ### Your turn: which discount rate?

    | OMB Circular A-94 | Real rate | γ, two years |
    |:------------------|----------:|-------------:|
    | App. C (2026): cost-effectiveness | {100 * RATES["2%"]:.1f}% | {GAMMAS["2%"]:.3f} |
    | 1992 base case: benefit-cost | {100 * RATES["7%"]:.0f}% | {GAMMAS["7%"]:.3f} |

    Which fits a maintenance policy, and what does each say about the future?
    """)
    return


@app.cell
def _(GAMMAS, RATES, YEARS_PER_STEP, check, mo, rnd):
    gamma_num = {k: rnd(g, 4) for k, g in GAMMAS.items()}
    one_minus_gamma_num = {k: rnd(1 - g, 4) for k, g in GAMMAS.items()}
    weight_20_num = {k: rnd((1 + r) ** -20, 2) for k, r in RATES.items()}
    check(gamma_num, "bridge/gamma")
    check(one_minus_gamma_num, "bridge/one_minus_gamma")
    check(weight_20_num, "bridge/weight_20_years")
    _w = {k: (1 + r) ** -20 for k, r in RATES.items()}
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        Appendix C (March 2026) gives Treasury borrowing rates for cost-effectiveness analysis:
        choosing the cheapest way to deliver a fixed service, which a maintenance policy is. The
        1992 Circular, reinstated in 2025, sets 7% for benefit-cost analysis of public
        investment, the return capital would earn elsewhere. At 7%, a cost 20 years away weighs
        {_w["7%"]:.2f}; at 2%, {_w["2%"]:.2f}. A high rate makes the future cheap. Per step, γ =
        (1 + r)^−{YEARS_PER_STEP}: one step is two years. CS 441 asks the same of Atari, where γ
        = 0.99 per frame.
        """)
        }
    )
    return gamma_num, one_minus_gamma_num, weight_20_num


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive: Bellman, cost now plus what follows

    **The assumption, in one sentence:** the next rating depends only on the rating now and the
    action (Markov), with the same probabilities every cycle.

    Then the best expected discounted cost from a state splits into the cost of the first action
    and the best expected cost from wherever it leads:

    $$V^*(s) = \min_a \Big[\, c(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s') \Big].$$

    Step by step: the cost of following a plan from $s$ is $c_0 + \gamma c_1 + \gamma^2 c_2 +
    \dots$; take $c_0$ out and the rest is $\gamma$ times the same kind of sum, started from
    $s'$; its expected value under the best plan from $s'$ is $V^*(s')$, averaged over where the
    action leads. The **action value** $Q^*(s,a)$ is the same quantity for one action, then the
    best:

    $$Q^*(s,a) = c(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, \min_{a'} Q^*(s', a'), \qquad
    \pi^*(s) = \arg\min_a Q^*(s,a).$$

    **Value iteration** applies the right-hand side to any guess until nothing moves. It
    converges, from any start, to one answer, because the update is a **contraction**: for two
    guesses $V$ and $W$, $\max_s |(TV)(s) - (TW)(s)| \le \gamma \max_s |V(s) - W(s)|$ (the
    min of a difference is at most the max of the differences, and the probabilities in a row
    sum to one). So each sweep shrinks the largest change by at least $\gamma$.

    **Policy evaluation** asks the same of one fixed policy $\pi$: $V^\pi = c_\pi + \gamma
    P_\pi V^\pi$, a linear system, $(I - \gamma P_\pi) V^\pi = c_\pi$. At value iteration's
    answer, evaluating its own greedy policy must give back its values: our check.

    **The costs.** Replacing costs 1 (a replacement); a step that starts poor (rating 4 or
    less) costs the poor-condition penalty, whatever is done about it. Forsyth's §13.2.3 builds
    the same recursion for a path through a trellis (a cost-to-go: the best cost from each node
    is a min over the next node of the step's cost plus that node's cost-to-go), without the
    discount and over a finite horizon.
    """)
    return


@app.cell
def _(GAMMAS, N_S, P, PENALTY, POOR, RATINGS, REPLACE_COST, VI_TOL, np, ref, time):
    _t0 = time.perf_counter()

    def costs(penalty: float) -> np.ndarray:
        """One step's cost of each action in each state, in replacements.

        Args:
            penalty: The cost of two years spent poor.

        Returns:
            An ``(N_S, 2)`` array: column 0 do nothing, column 1 replace.
        """
        C = np.zeros((N_S, 2))
        C[:, 1] = REPLACE_COST
        C[RATINGS <= POOR, :] += penalty  # the step starts poor, whatever is done about it
        return C

    def value_iteration(P: np.ndarray, C: np.ndarray, gamma: float) -> dict:
        """Bellman backups from V = 0 until no state's value moves by ``VI_TOL`` or more.

        Args:
            P: Transitions, actions x states x states.
            C: Costs, states x actions.
            gamma: Discount factor per step.

        Returns:
            ``V``, ``Q``, the greedy ``policy``, every sweep's values (``trace``), and each
            sweep's largest change (``deltas``).
        """
        V = np.zeros(C.shape[0])
        trace, deltas = [V], []
        while True:
            Q = C + gamma * np.einsum("ast,t->sa", P, V)  # every state and action at once
            V_new = Q.min(1)
            deltas.append(float(np.abs(V_new - V).max()))
            V = V_new
            trace.append(V)
            if deltas[-1] < VI_TOL:
                break
        return {"V": V, "Q": Q, "policy": Q.argmin(1), "trace": np.array(trace), "deltas": deltas}

    def evaluate_policy(
        P: np.ndarray, C: np.ndarray, gamma: float, policy: np.ndarray
    ) -> np.ndarray:
        """A policy's expected discounted cost from every state: (I - gamma P_pi) V = c_pi.

        Args:
            P: Transitions, actions x states x states.
            C: Costs, states x actions.
            gamma: Discount factor per step.
            policy: One action per state.

        Returns:
            The value of each state under the policy.
        """
        rows = np.arange(C.shape[0])
        P_pi = P[policy, rows]  # each state's row under its own action
        assert P_pi.shape == (C.shape[0], C.shape[0])
        return np.linalg.solve(np.eye(len(rows)) - gamma * P_pi, C[rows, policy])

    vi = {k: value_iteration(P, costs(PENALTY), g) for k, g in GAMMAS.items()}
    for _k, _g in GAMMAS.items():
        _v = vi[_k]
        # Policy evaluation of value iteration's own policy gives back its values
        assert np.allclose(evaluate_policy(P, costs(PENALTY), _g, _v["policy"]), _v["V"],
                           rtol=0, atol=1e-8)  # fmt: skip
        # Each sweep shrinks the largest change by at least gamma (the contraction)
        _d = np.array(_v["deltas"])
        assert (_d[1:] <= _g * _d[:-1] + 1e-12).all()
        # The same as the figure script's functions, to the last bit
        _theirs = ref.value_iteration(P, ref.costs(PENALTY), _g)
        assert all(np.array_equal(_v[_n], _theirs[_n]) for _n in ("V", "Q", "policy", "trace"))
        assert _v["deltas"] == _theirs["deltas"]
        assert np.array_equal(costs(PENALTY), ref.costs(PENALTY))
        assert np.array_equal(
            evaluate_policy(P, costs(PENALTY), _g, _v["policy"]),
            ref.evaluate_policy(P, ref.costs(PENALTY), _g, _v["policy"]),
        )
    t_vi = time.perf_counter() - _t0
    print({k: f"{len(v['deltas'])} sweeps, replace at {RATINGS[v['policy'] == 1].tolist()}"
           for k, v in vi.items()})  # fmt: skip
    return costs, evaluate_policy, t_vi, value_iteration, vi


@app.cell
def _(FIG, GAMMAS, PALETTE, RATINGS, VI_TOL, check, label, mo, np, plt, rnd, vi):
    vi_num = {
        k: {
            "sweeps": len(v["deltas"]),
            "replace_at": [int(r) for r in RATINGS[v["policy"] == 1]],
            "V": {int(r): rnd(x, 3) for r, x in zip(RATINGS, v["V"], strict=True)},
            "Q": {int(r): [rnd(x, 3) for x in q] for r, q in zip(RATINGS, v["Q"], strict=True)},
        }
        for k, v in vi.items()
    }
    check(vi_num, "bridge/vi")
    _v = vi["2%"]
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=FIG)
    for _i, _r in enumerate(RATINGS):
        if _r in (3, 4, 6, 9):
            _a.plot(_v["trace"][:, _i], lw=1.6, label=f"rating {label(_r)}")
    _a.set(xlabel="sweep", ylabel="V (replacements)")
    _a.legend(fontsize=8, frameon=False)
    _d = np.array(_v["deltas"])
    _k = np.arange(1, len(_d) + 1)
    _b.semilogy(_k, _d, color=PALETTE[0], lw=1.6, label="largest change")
    _b.semilogy(_k, _d[0] * GAMMAS["2%"] ** (_k - 1), ls="--", color="grey", lw=1,
                label=f"γ^k, γ = {GAMMAS['2%']:.3f}")  # fmt: skip
    _b.axhline(VI_TOL, color="grey", lw=0.6, ls=":")
    _b.set(xlabel="sweep", ylabel="max |V_k+1 − V_k|")
    _b.legend(fontsize=8, frameon=False)
    mo.vstack([
        mo.md(f"""
    ### Value iteration converges like γ^k

    At 2% a year, {vi_num["2%"]["sweeps"]} sweeps at γ = {GAMMAS["2%"]:.3f} reach changes
    under 1e-10 (at 7%, {vi_num["7%"]["sweeps"]}: a smaller γ contracts faster). The values
    are in replacements: a new bridge (rating 9) will cost its owner {_v["V"][-1]:.2f} of a
    replacement, discounted, over all its future; a poor one, {_v["V"][0]:.2f}. Checked: the
    policy's own evaluation, a linear solve, gives back the same values (to 1e-8), every sweep
    contracts by γ, and the figure script's `value_iteration` returns the same arrays bit for
    bit.
    """),
        _fig,
    ])  # fmt: skip
    return (vi_num,)


@app.cell
def _(PENALTY, mo):
    mo.md(f"""
    ### Your turn: predict the policy

    A poor bridge costs {PENALTY:.2f} per two years. At which ratings does the optimal policy
    replace?

    1. At 2% a year.
    2. At 7% a year.
    3. With no cost for being poor.

    Write a rating, or "never", for each. Use the attempt: at 2% replacing now beat waiting two
    years; at 7% it did not.
    """)
    return


@app.cell
def _(GAMMAS, PENALTY_GRID, POOR, RATINGS, costs, np, ref, value_iteration):
    def critical_penalty(P: np.ndarray, gamma: float) -> float:
        """The smallest poor-condition cost at which replacing is optimal anywhere (bisection).

        Args:
            P: Transitions.
            gamma: Discount factor per step.

        Returns:
            The critical penalty, to 1e-6.
        """
        lo, hi = 0.0, 5.0
        assert value_iteration(P, costs(hi), gamma)["policy"].any(), "no replacement even at 5"
        while hi - lo > 1e-6:
            mid = (lo + hi) / 2
            if value_iteration(P, costs(mid), gamma)["policy"].any():
                hi = mid
            else:
                lo = mid
        return hi

    def policy_sweep(P: np.ndarray, gamma: float) -> np.ndarray:
        """The optimal policy at every poor-condition cost of the grid.

        Args:
            P: Transitions.
            gamma: Discount factor per step.

        Returns:
            A ``(len(PENALTY_GRID), N_S)`` array of actions.
        """
        return np.array([value_iteration(P, costs(c), gamma)["policy"] for c in PENALTY_GRID])

    def make_crit(P: np.ndarray) -> tuple[dict, dict]:
        """Critical penalties and policy maps at both rates, checked.

        Args:
            P: Transitions.

        Returns:
            The critical penalty and the policy map at each rate.
        """
        crit_, sweep_ = {}, {}
        for k, g in GAMMAS.items():
            crit_[k] = critical_penalty(P, g)
            assert crit_[k] == ref.critical_penalty(P, g)  # the script's bisection agrees
            # Just below it nothing is replaced; at it, the poor ratings are
            below = value_iteration(P, costs(crit_[k] - 2e-6), g)
            at = value_iteration(P, costs(crit_[k]), g)
            assert not below["policy"].any() and at["policy"].any()
            assert set(RATINGS[at["policy"] == 1]) <= set(RATINGS[RATINGS <= POOR])
            sweep_[k] = policy_sweep(P, g)
        return crit_, sweep_

    return critical_penalty, make_crit, policy_sweep


@app.cell
def _(
    FIG,
    GAMMAS,
    P,
    PALETTE,
    PENALTY,
    PENALTY_GRID,
    RATINGS,
    bridge_mdp,
    check,
    costs,
    critical_penalty,
    make_crit,
    np,
    pairs,
    plt,
    records,
    rnd,
    time,
    value_iteration,
):
    _t0 = time.perf_counter()
    crit, sweep = make_crit(P)
    check({k: rnd(c, 3) for k, c in crit.items()}, "bridge/critical_penalty")
    no_penalty_replaces = bool(sweep["2%"][0].any() or sweep["7%"][0].any())
    check(no_penalty_replaces, "bridge/no_penalty_replaces")
    # The rises with no recorded work counted as no change, instead of left out
    _stay = bridge_mdp(records, pairs, "stay")["P"]
    rises_as_stays = {
        "critical_penalty": {k: rnd(critical_penalty(_stay, g), 3) for k, g in GAMMAS.items()},
        "V_poor_2%": rnd(value_iteration(_stay, costs(PENALTY), GAMMAS["2%"])["V"][0], 3),
    }
    check(rises_as_stays, "bridge/rises_as_stays")
    # Q at rating 4 for both actions over the grid, at 2%: the two lines cross at the boundary
    _q4 = np.array([value_iteration(P, costs(c), GAMMAS["2%"])["Q"][1] for c in PENALTY_GRID])
    t_crit = time.perf_counter() - _t0
    _fig, _axes = plt.subplots(1, 3, figsize=(FIG[0], FIG[1]), sharey=False)
    for _ax, (_name, _pol) in zip(_axes[:2], sweep.items(), strict=True):
        _ax.imshow(_pol.T, origin="lower", aspect="auto", cmap="Blues", vmin=-0.3, vmax=1.3,
                   extent=(PENALTY_GRID[0], PENALTY_GRID[-1], RATINGS[0] - 0.5,
                           RATINGS[-1] + 0.5))  # fmt: skip
        _ax.axvline(crit[_name], color=PALETTE[1], lw=1.5)
        _ax.plot([PENALTY], [6.5], marker="v", color="black")
        _ax.set(
            title=f"{_name} a year: replace above {crit[_name]:.3f}",
            xlabel="cost of two years poor",
        )
        _ax.title.set_fontsize(9)
    _axes[0].set(ylabel="lowest rating", yticks=RATINGS)
    _axes[2].plot(PENALTY_GRID, _q4[:, 0], color=PALETTE[0], lw=1.6, label="Q(4, do nothing)")
    _axes[2].plot(PENALTY_GRID, _q4[:, 1], color=PALETTE[1], lw=1.6, ls="--", label="Q(4, replace)")
    _axes[2].axvline(crit["2%"], color="grey", lw=0.8, ls=":")
    _axes[2].set(
        xlabel="cost of two years poor", ylabel="Q (replacements)", title="rating 4, at 2%"
    )
    _axes[2].title.set_fontsize(9)
    _axes[2].legend(fontsize=7, frameon=False)
    _fig.tight_layout()
    _fig
    return crit, no_penalty_replaces, rises_as_stays, sweep, t_crit


@app.cell
def _(PENALTY, RATINGS, crit, mo, no_penalty_replaces, rises_as_stays, sweep, vi):
    _r2 = RATINGS[vi["2%"]["policy"] == 1].tolist()
    _r7 = RATINGS[vi["7%"]["policy"] == 1].tolist()
    assert _r2 == [3, 4] and _r7 == [] and not no_penalty_replaces
    assert sweep["2%"].shape == (len(sweep["2%"]), len(RATINGS))
    mo.accordion(
        {
            "Answer": mo.md(f"""
        1. At 2%, with {PENALTY:.2f} per two years poor: replace at {max(_r2)} and below, once
           that cost exceeds {crit["2%"]:.3f}.
        2. At 7%: never. The bridge stays poor forever, at {PENALTY:.2f} every two years,
           worth {vi["7%"]["V"][0]:.2f} today, less than a replacement; replacing pays only
           above {crit["7%"]:.3f}.
        3. With no cost for being poor, no rate ever replaces: the cheapest policy lets every
           bridge decay.

        The cost of being poor is a judgement (user delays, detours, risk), not data. The policy
        is only as good as that judgement. At rating 4 and a cost of {PENALTY:.2f}, the two
        Q-values are {vi["2%"]["Q"][1, 0]:.3f} (do nothing) and {vi["2%"]["Q"][1, 1]:.3f}
        (replace): close, so a small change in the costs flips the decision. The replace line
        is nearly flat: after a replacement, every bridge starts over the same way.

        The rises with no recorded work were left out of the counts. Counted as no change
        instead, they move the critical cost from {crit["2%"]:.3f} to
        {rises_as_stays["critical_penalty"]["2%"]:.3f} at 2% and from {crit["7%"]:.3f} to
        {rises_as_stays["critical_penalty"]["7%"]:.3f} at 7%: the choice barely matters.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    ### Q-learning: learn Q by trying

    Value iteration needed $P$, which we counted. **Q-learning** never reads it: it learns from
    experience, one simulated inspection at a time. After each step $(s, a, c, s')$ it moves
    $Q(s,a)$ toward a target made from one step of hindsight:

    $$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, c + \gamma \min_{a'} Q(s', a') - Q(s,a) \Big].$$

    The bracket is the **temporal-difference error**: the cost now plus the estimate at the
    next state, minus the current estimate. Its expected value is zero exactly at $Q^*$ (the
    Bellman equation), so $Q^*$ is where the updates stop moving on average. Two conditions
    make it converge (Watkins and Dayan, 1992): every pair $(s,a)$ is visited infinitely often,
    here by exploring (a random action one time in five), and the step sizes shrink slowly
    enough, $\sum \alpha = \infty$ and $\sum \alpha^2 < \infty$: here $\alpha = 1/n(s,a)^{0.7}$.

    **Predict** before running: after 2 million simulated inspections, will Q-learning's greedy
    policy equal value iteration's? Will its Q values be within 0.01?

    The world below draws the next rating from the counted matrix, as the inventory would; the
    learner sees only the draws. The loop uses Python floats and lists, not NumPy scalars, which
    is several times faster and gives the same bits (checked against the figure script's
    NumPy loop).
    """)
    return


@app.cell
def _(N_S, Q_CHECKPOINTS, Q_EPSILON, Q_POWER, Q_STEPS, RATINGS, SEED, bisect_right, np):
    def q_learning(
        P: np.ndarray,
        C: np.ndarray,
        gamma: float,
        start: np.ndarray,
        Q_star: np.ndarray,
        steps: int = Q_STEPS,
    ) -> dict:
        """Tabular Q-learning from simulated inspections; ``P`` only draws the next state.

        Args:
            P: Transitions, used only to simulate the world.
            C: Costs, states x actions.
            gamma: Discount factor per step.
            start: The first state's distribution (Illinois's 2024 ratings).
            Q_star: Value iteration's Q, to measure the error at checkpoints.
            steps: Simulated inspections.

        Returns:
            The final ``Q`` and ``visits``; at each checkpoint the largest error, whether the
            greedy policy equals value iteration's, the fewest visits of any pair, and the
            ratings where the policies differ; the last step at which the greedy policy
            differed (checked after every step); and, per rating, the steps it differed on.
        """
        rng = np.random.default_rng(SEED)
        cum = np.cumsum(P, axis=2)
        cum[:, :, -1] = 1.0  # guard the last bin against round-off
        cum_rows = cum.tolist()  # cum_rows[a][s]: the cumulative distribution of s'
        cost = C.tolist()
        Q = [[0.0, 0.0] for _ in range(N_S)]
        visits = [[0.0, 0.0] for _ in range(N_S)]
        s = int(rng.choice(len(start), p=start))
        marks = set(Q_CHECKPOINTS) | set(
            np.unique(np.geomspace(100, steps, 31).astype(int)).tolist()
        )
        # Every random number drawn up front, in the order the figure script draws them
        explore = (rng.random(steps) < Q_EPSILON).tolist()
        random_a = rng.integers(0, 2, steps).tolist()
        u = rng.random(steps).tolist()
        best_star = Q_star.argmin(1).tolist()
        # An update changes one row of Q, so only that rating's comparison can change: keep
        # each rating's verdict and their count, and check the policy after every step
        differ = [b != 0 for b in best_star]  # Q = 0: every argmin is 0, "do nothing"
        n_differ = sum(differ)
        differ_steps = [0] * N_S
        last_differs_at = 0
        log = []
        for t in range(steps):
            q_s = Q[s]
            a = random_a[t] if explore[t] else (0 if q_s[0] <= q_s[1] else 1)  # argmin, ties: 0
            s2 = bisect_right(cum_rows[a][s], u[t])  # the world draws the next rating
            visits[s][a] += 1.0
            alpha = visits[s][a] ** -Q_POWER
            q2 = Q[s2]
            Q[s][a] += alpha * (cost[s][a] + gamma * min(q2[0], q2[1]) - Q[s][a])
            now = (0 if q_s[0] <= q_s[1] else 1) != best_star[s]
            n_differ += now - differ[s]
            differ[s] = now
            if n_differ:
                last_differs_at = t + 1
                for i in range(N_S):
                    differ_steps[i] += differ[i]
            s = s2
            if t + 1 in marks:
                Qa = np.array(Q)
                wrong = Qa.argmin(1) != np.array(best_star)
                assert wrong.tolist() == differ  # the running comparison, checked here too
                log.append(
                    {
                        "steps": t + 1,
                        "max_error": float(np.abs(Qa - Q_star).max()),
                        "same_policy": bool(not wrong.any()),
                        "fewest_visits": int(np.min(visits)),
                        "differ_at": [int(r) for r in RATINGS[wrong]],
                    }
                )
        assert differ == (np.array(Q).argmin(1) != Q_star.argmin(1)).tolist()
        return {"Q": np.array(Q), "visits": np.array(visits), "log": log,
                "last_differs_at": last_differs_at, "differ_steps": differ_steps}  # fmt: skip

    return (q_learning,)


@app.cell
def _(
    GAMMAS,
    P,
    PENALTY,
    Q_CHECKPOINTS,
    Q_CHECK_STEPS,
    Q_EPSILON,
    Q_STEPS,
    RATINGS,
    check,
    costs,
    q_learning,
    ref,
    rnd,
    start_2024,
    time,
    vi,
):
    _t0 = time.perf_counter()
    _C, _g, _Qs = costs(PENALTY), GAMMAS["2%"], vi["2%"]["Q"]
    # The check against the figure script's NumPy loop, on a shorter run (both draw their random
    # numbers for the run's length, so a shorter run is not a prefix of the long one)
    _saved = ref.Q_STEPS
    ref.Q_STEPS = Q_CHECK_STEPS
    try:
        _theirs = ref.q_learning(P, _C, _g, start_2024, _Qs)
    finally:
        ref.Q_STEPS = _saved
    _short = q_learning(P, _C, _g, start_2024, _Qs, steps=Q_CHECK_STEPS)
    assert ref.Q_STEPS == Q_STEPS
    assert (_short["Q"] == _theirs["Q"]).all() and (_short["visits"] == _theirs["visits"]).all()
    assert [{k: r[k] for k in _theirs["log"][0]} for r in _short["log"]] == _theirs["log"]
    assert _short["last_differs_at"] == _theirs["last_differs_at"]
    assert _short["differ_steps"] == _theirs["differ_steps"].tolist()
    # The slides' run: two million simulated inspections
    ql = q_learning(P, _C, _g, start_2024, _Qs)
    _log = ql["log"]
    q_num = {
        "steps": Q_STEPS,
        "epsilon": Q_EPSILON,
        "log": [
            {k: (rnd(r[k], 4) if k == "max_error" else r[k]) for k in
             ("steps", "max_error", "same_policy", "fewest_visits")}
            for r in _log if r["steps"] in Q_CHECKPOINTS
        ],
        "last_differs_at": ql["last_differs_at"],
        "differ_steps": {int(r): n for r, n in zip(RATINGS, ql["differ_steps"], strict=True)},
        "fewest_visits": int(ql["visits"].min()),
    }  # fmt: skip
    check(q_num, "bridge/q_learning")
    # Where the checkpoints alone would have put "for good": after the last disagreeing one
    _late = [r["steps"] for r in _log if not r["same_policy"]]
    checkpoint_for_good = next(r["steps"] for r in _log if r["steps"] > max(_late))
    assert ql["last_differs_at"] >= max(_late)  # every step is checked, the checkpoints too
    # Where the greedy policy flickered after it first agreed with value iteration's
    _first = next(r["steps"] for r in _log if r["same_policy"])
    flicker = [(r["steps"], r["differ_at"]) for r in _log
               if r["steps"] > _first and not r["same_policy"]]  # fmt: skip
    t_ql = time.perf_counter() - _t0
    print(f"Q-learning: equal to the script's at {Q_CHECK_STEPS:,} steps; slides' run in "
          f"{t_ql:.1f} s; flicker after {_first:,}: {flicker}; last difference at step "
          f"{ql['last_differs_at']:,} (the checkpoints alone say {checkpoint_for_good:,})")  # fmt: skip
    return checkpoint_for_good, flicker, q_num, ql, t_ql


@app.cell
def _(FIG, PALETTE, checkpoint_for_good, flicker, mo, np, plt, q_num, ql, vi):
    _t = np.array([r["steps"] for r in ql["log"]])
    _e = np.array([r["max_error"] for r in ql["log"]])
    _same = np.array([r["same_policy"] for r in ql["log"]])
    _fig, _ax = plt.subplots(figsize=(FIG[0] * 0.6, FIG[1]))
    _ax.loglog(_t, _e, color=PALETTE[0], lw=1.6)
    _ax.scatter(_t[_same], _e[_same], color=PALETTE[2], s=14, zorder=3, label="same policy as VI")
    _ax.scatter(_t[~_same], _e[~_same], color=PALETTE[1], marker="x", s=18, zorder=3,
                label="a different policy")  # fmt: skip
    _ax.axvline(q_num["last_differs_at"], color="grey", ls=":", lw=1, label="last difference")
    _ax.set(xlabel="simulated inspections", ylabel="max |Q − Q*|")
    _ax.legend(fontsize=8, frameon=False)
    _log = {r["steps"]: r for r in q_num["log"]}
    _rates = sorted({r for _, d in flicker for r in d})
    _where = {r: n for r, n in q_num["differ_steps"].items() if n}
    mo.vstack([
        mo.md(f"""
    ### Q-learning finds the same policy

    Q-learning never saw $P$, value iteration did, and they agree. The largest error falls from
    {_log[1000]["max_error"]:.2f} after a thousand simulated inspections to
    {_log[1_000_000]["max_error"]:.4f} by a million and {_log[2_000_000]["max_error"]:.4f} by
    two million; the rarest state-action pair was visited {q_num["fewest_visits"]:,} times.
    Compared after every step, the greedy policy last differs from value iteration's at step
    {q_num["last_differs_at"]:,} and matches it for good after that (steps on which each
    rating's greedy action differed: {", ".join(f"{n:,} at {r}" for r, n in _where.items())}).
    Before, it flickers: at the logged checkpoints after its first agreement it disagreed
    {len(flicker)} times, always at rating(s) {_rates}, where the two Q values are close
    ({vi["2%"]["Q"][1, 0]:.3f} against {vi["2%"]["Q"][1, 1]:.3f}; rating 3 has the same
    pair). The {len(ql["log"])} logged checkpoints alone would have said "for good from
    {checkpoint_for_good:,}": a grid of checkpoints can miss a late flip. Checked: the figure
    script's own loop gives the same Q, visit counts, and last difference, bit for bit, on a
    run of 100,000.
    """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Too many states for a table

    CS 441's Breakout has a screen for a state; a signal has queues. One intersection in SUMO, a
    microscopic simulator in which each car follows the one ahead: four single-lane approaches
    of 300 m at 50 km/h, through traffic only. Every five seconds the controller picks a green,
    north-south or east-west; a switch costs a 3 s yellow and a 2 s all-red first. Its state:
    each approach's stopped and total vehicles, the green, and how long it has run. Ten numbers,
    eight of them counts up to about 40 (a 300 m lane holds about 300 / 7.5 cars): a table over
    them is out of reach, so a network will estimate Q.

    **The network.** Five nodes (the junction C, with a traffic light, and the four far ends),
    eight one-lane edges, and the four through movements, converted by SUMO's `netconvert` with
    a 3 s yellow and a 2 s all-red. **The demand.** Through traffic on each approach, a
    Bernoulli arrival every second (Poisson-like), at assumed volumes that swing from
    north-south to east-west over the hour. Drivers are SUMO's default car (Krauss model;
    `sigma` 0.5 is the drivers' imperfection, drawn from SUMO's seed). A second route file
    offers a car every second on the north approach only, to measure the saturation flow.

    SUMO reads a comma in an option's value as a list separator, so every file name below is
    built from a slug of letters, digits, and underscores, never from a label such as "DQN,
    online".
    """)
    return


@app.cell
def _(
    ALL_RED_S,
    APPROACHES,
    APPROACH_M,
    DEMAND,
    ET,
    EW,
    EXITS,
    GREEN_PHASE,
    NS,
    Path,
    REVERSED,
    SEED,
    SPEED_MS,
    YELLOW_S,
    check,
    ref,
    subprocess,
    tempfile,
    time,
):
    _t0 = time.perf_counter()

    def sumo_tool(name: str) -> Path:
        """A SUMO program from the eclipse-sumo wheel.

        Args:
            name: The program, e.g. "netconvert".

        Returns:
            Its path.
        """
        import sumo as _sumo  # the wheel's package only says where its binaries are

        return Path(_sumo.SUMO_HOME) / "bin" / name

    def write_routes(
        path: Path,
        profile: tuple[tuple[int, int, float, float], ...],
        approaches: tuple[str, ...] = APPROACHES,
    ) -> None:
        """A route file: through traffic, a Bernoulli arrival each second on each approach.

        Args:
            path: The file to write.
            profile: (begin s, end s, north-south veh/h, east-west veh/h), per approach.
            approaches: The approaches that receive traffic.
        """
        lines = [
            "<routes>",
            '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5"'
            f' maxSpeed="{SPEED_MS}"/>',
        ]
        lines += [f'  <route id="r_{a}" edges="{a} {b}"/>' for a, b in EXITS.items()]
        for i, (begin, end, ns, ew) in enumerate(profile):
            for a in approaches:
                q = ns if a in ("N_in", "S_in") else ew
                if q <= 0:
                    continue
                lines.append(
                    f'  <flow id="f{i}_{a}" type="car" route="r_{a}" begin="{begin}" end="{end}"'
                    f' probability="{min(q, 3600) / 3600:.6f}" departLane="best" departSpeed="max"/>'
                )
        lines.append("</routes>")
        path.write_text("\n".join(lines) + "\n")

    def write_network(out: Path) -> dict[str, Path]:
        """Write the nodes, edges, and movements, run netconvert twice, and write the demands.

        The network is converted with a static program, whose phases the controllers hold, and
        again with SUMO's gap-based actuated program at its defaults (the same phases).

        Args:
            out: A new directory for the SUMO inputs.

        Returns:
            Paths of the network ("net"), the actuated network ("net_actuated"), the hour's
            demand ("demand"), the same demand reversed in time ("reversed"), and the
            saturation test's demand ("saturation").

        Raises:
            RuntimeError: If netconvert fails.
        """
        out.mkdir(parents=True)
        ends = {"N": (0, APPROACH_M), "S": (0, -APPROACH_M), "E": (APPROACH_M, 0),
                "W": (-APPROACH_M, 0)}  # fmt: skip
        nodes = ['<node id="C" x="0" y="0" type="traffic_light" tl="C"/>']
        nodes += [f'<node id="{k}" x="{x}" y="{y}" type="priority"/>' for k, (x, y) in ends.items()]
        edges = []
        for k in ends:
            edges.append(f'<edge id="{k}_in" from="{k}" to="C" numLanes="1" speed="{SPEED_MS}"/>')
            edges.append(f'<edge id="C_{k}" from="C" to="{k}" numLanes="1" speed="{SPEED_MS}"/>')
        moves = [f'<connection from="{a}" to="{b}"/>' for a, b in EXITS.items()]
        (out / "net.nod.xml").write_text("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n")
        (out / "net.edg.xml").write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n")
        (out / "net.con.xml").write_text(
            "<connections>\n" + "\n".join(moves) + "\n</connections>\n"
        )
        nets = {"net": out / "net.net.xml", "net_actuated": out / "net_actuated.net.xml"}
        for key, extra in (("net", []), ("net_actuated", ["--tls.default-type", "actuated"])):
            cmd = [
                str(sumo_tool("netconvert")),
                "-n", str(out / "net.nod.xml"),
                "-e", str(out / "net.edg.xml"),
                "-x", str(out / "net.con.xml"),
                "--no-turnarounds",
                "--tls.yellow.time", str(YELLOW_S),
                "--tls.allred.time", str(ALL_RED_S),
                *extra,
                "--seed", str(SEED),
                "-o", str(nets[key]),
            ]  # fmt: skip
            done = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"netconvert failed: {done.stderr}")
        write_routes(out / "demand.rou.xml", DEMAND)
        write_routes(out / "demand_reversed.rou.xml", REVERSED)
        write_routes(out / "saturation.rou.xml", ((0, 150, 3600, 0),), ("N_in",))
        return {**nets, "demand": out / "demand.rou.xml",
                "reversed": out / "demand_reversed.rou.xml",
                "saturation": out / "saturation.rou.xml"}  # fmt: skip

    # SUMO's inputs and trip files live in one temporary directory, removed when Python exits.
    # Pool workers only write trip files into it; each trip file is deleted once it is read.
    SUMO_TMP = tempfile.TemporaryDirectory(prefix="l26_sumo_")
    WORK = Path(SUMO_TMP.name)
    assert "," not in str(WORK)
    files = write_network(WORK / "net")
    TRIPS = WORK / "trips"
    TRIPS.mkdir()

    def _xml(path: Path) -> bytes:
        return ET.tostring(ET.parse(path).getroot())  # our own SUMO files; comments dropped

    # The same network and demand files as the figure script's (its header comment, which
    # records the paths and the time, aside)
    _theirs = ref.build_network(WORK / "ref_net")
    assert set(files) == set(_theirs)
    for _k in ("net", "net_actuated"):
        assert _xml(files[_k]) == _xml(_theirs[_k]), _k
    for _k in ("demand", "reversed", "saturation"):
        assert files[_k].read_text() == _theirs[_k].read_text(), _k
    check([list(_d) for _d in DEMAND], "signal/demand")
    check([list(_d) for _d in REVERSED], "signal/reversed_demand")
    _junctions = ET.parse(files["net"]).getroot().findall("junction")
    _tl = {k: ET.parse(files[k]).getroot().find("tlLogic") for k in ("net", "net_actuated")}
    _phases = {k: v.findall("phase") for k, v in _tl.items()}
    sumo_net = {
        "junctions": sorted(j.attrib["id"] for j in _junctions if j.attrib["type"] != "internal"),
        "phases": [(int(p.attrib["duration"]), p.attrib["state"]) for p in _phases["net"]],
    }
    # The actuated program: the same phases, each green between a minimum and a maximum
    assert (
        _tl["net_actuated"].attrib["type"] == "actuated" and _tl["net"].attrib["type"] == "static"
    )
    assert [p.attrib["state"] for p in _phases["net_actuated"]] == [
        s for _, s in sumo_net["phases"]
    ]
    _greens = [_phases["net_actuated"][GREEN_PHASE[g]] for g in (NS, EW)]
    actuated_num = {"min_green_s": [float(p.attrib["minDur"]) for p in _greens],
                    "max_green_s": [float(p.attrib["maxDur"]) for p in _greens]}  # fmt: skip
    check(actuated_num, "signal/actuated")
    t_net = time.perf_counter() - _t0
    print(f"networks equal to the script's; signal phases {sumo_net['phases']}; actuated "
          f"greens {actuated_num}")  # fmt: skip
    return SUMO_TMP, TRIPS, WORK, actuated_num, files, sumo_net, t_net


@app.cell
def _(
    ALL_RED_S,
    APPROACHES,
    DECISION_S,
    END_S,
    EVAL_SEEDS,
    EW,
    GREEN_PHASE,
    HOUR_S,
    NS,
    OBS_SCALE,
    SEED,
    STUCK,
    YELLOW_S,
    files,
    np,
    ref,
    time,
):
    _t0 = time.perf_counter()

    class SignalEnv:
        """One SUMO run through libsumo; a controller picks a green every five seconds.

        The observation: each approach's halted and total vehicles (divided by ``OBS_SCALE``),
        the green phase, and its seconds so far (capped at 60, divided by 60). A switch runs the
        yellow and the all-red first. The reward: minus the halted vehicles summed over every
        simulated second, all approaches, as SUMO counts them (whatever a detector says). Used
        in a ``with`` block, the simulation is closed however the block ends.
        """

        def __init__(
            self,
            files: dict,
            seed: int,
            *,
            routes: str = "demand",
            stuck: tuple[str, ...] = (),
            tripinfo: object = None,
            actuated: bool = False,
        ) -> None:
            """Start SUMO.

            Args:
                files: From ``write_network``.
                seed: SUMO's seed (the arrivals and the drivers' imperfection).
                routes: "demand", "reversed", or "saturation".
                stuck: Approaches whose detectors read zero.
                tripinfo: Where to write every vehicle's trip summary, if anywhere.
                actuated: Let SUMO's actuated program run the light, untouched.
            """
            import libsumo  # one simulation per process: imported where it runs

            self.sumo = libsumo
            args = [
                "sumo", "-n", str(files["net_actuated" if actuated else "net"]),
                "-r", str(files[routes]),
                "--seed", str(seed), "--no-step-log", "--no-warnings",
                "--time-to-teleport", "-1",  # never remove a stuck car: a starved queue counts
                "--end", str(END_S),
            ]  # fmt: skip
            if tripinfo is not None:  # every vehicle: also those unfinished, or never let in
                assert "," not in str(tripinfo), "SUMO splits option values at commas"
                args += [
                    "--tripinfo-output", str(tripinfo),
                    "--tripinfo-output.write-unfinished", "true",
                    "--tripinfo-output.write-undeparted", "true",
                ]  # fmt: skip
            self.sumo.start(args)
            self.closed, self.left = False, {}
            self.stuck = tuple(stuck)
            self.phase, self.since = NS, 0
            self.lights: list[int] = []  # each second: NS or EW green, -1 in a change interval
            if not actuated:  # our controllers hold SUMO's phases one by one
                self.hold(GREEN_PHASE[NS])

        def __enter__(self) -> "SignalEnv":
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

        def hold(self, index: int) -> None:
            """Show one of SUMO's phases until told otherwise."""
            self.sumo.trafficlight.setPhase("C", index)
            self.sumo.trafficlight.setPhaseDuration("C", 1e6)

        def detector(self, approach: str) -> tuple[int, int]:
            """What an approach's detector reports: halted (under 0.1 m/s) and total vehicles."""
            if approach in self.stuck:
                return 0, 0
            edge = self.sumo.edge
            return edge.getLastStepHaltingNumber(approach), edge.getLastStepVehicleNumber(approach)

        def observe(self) -> np.ndarray:
            """The controller's view, ten numbers in float32."""
            seen = [self.detector(a) for a in APPROACHES]
            counts = np.array([h for h, _ in seen] + [n for _, n in seen], float) / OBS_SCALE
            return np.array([*counts, float(self.phase), min(self.since, 60) / 60.0], np.float32)

        def advance(self, seconds: int) -> float:
            """Simulate whole seconds; return the halted vehicles summed over them."""
            total = 0.0
            green = {GREEN_PHASE[NS]: NS, GREEN_PHASE[EW]: EW}
            for _ in range(seconds):
                self.sumo.simulationStep()
                total += sum(self.sumo.edge.getLastStepHaltingNumber(a) for a in APPROACHES)
                self.lights.append(green.get(self.sumo.trafficlight.getPhase("C"), -1))
            return total

        def step(self, action: int) -> tuple[np.ndarray, float, bool]:
            """One decision: change the green if asked (yellow, all-red), then hold five seconds.

            Args:
                action: ``NS`` or ``EW``.

            Returns:
                The next observation, the reward (minus halted vehicle-seconds), and whether
                the hour is over.
            """
            halted = 0.0
            if action != self.phase:
                self.hold(GREEN_PHASE[self.phase] + 1)  # yellow
                halted += self.advance(YELLOW_S)
                self.hold(GREEN_PHASE[self.phase] + 2)  # all-red
                halted += self.advance(ALL_RED_S)
                self.phase, self.since = action, 0
                self.hold(GREEN_PHASE[action])
            halted += self.advance(DECISION_S)
            self.since += DECISION_S
            return self.observe(), -halted, self.sumo.simulation.getTime() >= HOUR_S

        def close(self) -> dict:
            """Stop SUMO once; return the vehicles still waiting to enter and still inside."""
            if not self.closed:
                self.closed = True
                try:
                    self.left = {
                        "pending": len(self.sumo.simulation.getPendingVehicles()),
                        "inside": self.sumo.vehicle.getIDCount(),
                    }
                finally:
                    self.sumo.close()
            return self.left

    def _drive(env: object, step_name: str, obs_name: str, actions: np.ndarray) -> dict:
        """Apply a fixed action sequence until the hour ends (or the actions run out)."""
        out = {"obs": [], "rewards": [], "done": [], "switches": 0}
        try:
            out["obs"].append(getattr(env, obs_name)())
            for a in actions:
                out["switches"] += int(a) != env.phase
                o, r, d = getattr(env, step_name)(int(a))
                out["obs"].append(o)
                out["rewards"].append(r)
                out["done"].append(d)
                if d:
                    break
            out["lights"] = list(env.lights)
        finally:
            out["left"] = env.close()
        return out

    # A fixed action sequence: hold, or switch with probability 0.3, one draw per decision
    _rng = np.random.default_rng(SEED)
    _actions = np.cumsum(_rng.random(800) < 0.3) % 2
    _cases = {"working": ((), 800), "dead side street": (STUCK, 100)}
    for _name, (_stuck, _n) in _cases.items():
        _seed = EVAL_SEEDS[0] if not _stuck else EVAL_SEEDS[1]
        _mine = _drive(SignalEnv(files, _seed, stuck=_stuck), "step", "observe", _actions[:_n])
        _ref = _drive(ref.Intersection(files, _seed, stuck=_stuck), "step", "obs", _actions[:_n])
        assert np.array_equal(np.array(_mine["obs"]), np.array(_ref["obs"])), _name
        assert _mine["obs"][0].dtype == np.float32 and _mine["obs"][0].shape == (10,)
        assert _mine["rewards"] == _ref["rewards"] and _mine["done"] == _ref["done"], _name
        assert _mine["lights"] == _ref["lights"] and _mine["left"] == _ref["left"], _name
        # A decision every five seconds, ten when the green changes
        _n, _sw = len(_mine["rewards"]), _mine["switches"]
        assert len(_mine["lights"]) == DECISION_S * _n + (YELLOW_S + ALL_RED_S) * _sw, _name
        if not _stuck:
            env_check = {"decisions": _n, "switches": _sw, "seconds": len(_mine["lights"]),
                         "halted_s": -sum(_mine["rewards"])}  # fmt: skip
    t_env = time.perf_counter() - _t0
    print(f"environment equal to the script's on a fixed action sequence: {env_check}")
    return SignalEnv, env_check, t_env


@app.cell
def _(EW, GREEN_PHASE, NS, SEED, SignalEnv, check, files, np, ref, rnd):
    def measure_saturation(files: dict) -> dict:
        """Discharge a standing queue at a green and time it: SUMO's saturation flow.

        North-south is held red for 150 s while a car is offered every second; then it turns
        green. Each vehicle leaving the approach is timed. The headways from the fifth vehicle
        to the 25th (past the start-up loss, as the Highway Capacity Manual measures them)
        give the saturation flow.

        Args:
            files: From ``write_network``.

        Returns:
            The queue at the green, the mean headway, and the saturation flow in vehicles per
            hour of green.
        """
        with SignalEnv(files, SEED, routes="saturation") as env:
            env.hold(GREEN_PHASE[EW])  # north is red while its queue builds
            env.advance(150)
            sim, edge = env.sumo.simulation, env.sumo.edge
            queue = edge.getLastStepHaltingNumber("N_in")
            env.hold(GREEN_PHASE[NS])
            t0 = sim.getTime()
            before = set(edge.getLastStepVehicleIDs("N_in"))
            times: list[float] = []
            for _ in range(90):
                env.sumo.simulationStep()
                now = set(edge.getLastStepVehicleIDs("N_in"))
                times += [sim.getTime() - t0] * len(before - now)  # each car that left this second
                before = now
        t = np.array(times)
        headway = (t[24] - t[4]) / 20  # the 5th to the 25th departure: 20 headways
        return {"queue": int(queue), "headway_s": float(headway), "veh_per_h": 3600.0 / headway}

    sat = measure_saturation(files)
    assert sat == ref.saturation_flow(files)  # the script's measurement, to the last bit
    check({"queue": sat["queue"], "headway_s": rnd(sat["headway_s"], 2),
           "veh_per_h": int(round(sat["veh_per_h"]))}, "signal/saturation")  # fmt: skip
    print(sat)
    return (sat,)


@app.cell
def _(DEMAND, LOST_S, REVERSED, check, np, ref, rnd, sat):
    def webster(q_ns: float, q_ew: float, s: float) -> dict:
        """Webster's (1958) optimal fixed cycle and the greens that share it by flow ratio.

        Args:
            q_ns: Flow on a north-south approach, veh/h.
            q_ew: Flow on an east-west approach, veh/h.
            s: Saturation flow, veh/h of green.

        Returns:
            The flow ratios y = q / s, their sum Y, the cycle C0 = (1.5 L + 5) / (1 - Y), and
            each phase's green (C0 - L) y / Y.
        """
        y = np.array([q_ns, q_ew]) / s
        Y = float(y.sum())
        assert Y < 1, "demand above capacity: no fixed cycle serves it"
        C = (1.5 * LOST_S + 5) / (1 - Y)
        g = (C - LOST_S) * y / Y
        return {"y": y.tolist(), "Y": Y, "cycle_s": C, "green_s": g.tolist()}

    def webster_plans(
        s: float, profile: tuple = DEMAND, factor: float = 1.0
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Webster's fixed-time plans as (start s, north-south green s, east-west green s).

        Args:
            s: The measured saturation flow.
            profile: The demand the plans are computed from.
            factor: Every 15-minute green is multiplied by this before rounding.

        Returns:
            "hour": one plan from the hour's mean flows; "15 min": one per quarter hour, which
            knows the demand schedule in advance.
        """
        q_ns = float(np.mean([d[2] for d in profile]))
        q_ew = float(np.mean([d[3] for d in profile]))
        hour = webster(q_ns, q_ew, s)
        return {
            "hour": [(0, *(round(g) for g in hour["green_s"]))],
            "15 min": [(b, *(round(factor * g) for g in webster(ns, ew, s)["green_s"]))
                       for b, _, ns, ew in profile],
        }  # fmt: skip

    _s = sat["veh_per_h"]
    plans = webster_plans(_s)
    webster_hour = webster(np.mean([d[2] for d in DEMAND]), np.mean([d[3] for d in DEMAND]), _s)
    webster_15 = [webster(ns, ew, _s) for _, _, ns, ew in DEMAND]
    assert plans == ref.webster_plans(_s)
    # Reversing the quarter hours leaves the hour's mean flows, so its plan, as they were; the
    # plans per 15 minutes retimed for the reversed demand are the same plans in reverse order
    plans["retimed"] = webster_plans(_s, REVERSED)["15 min"]
    assert webster_plans(_s, REVERSED)["hour"] == plans["hour"]
    assert plans["retimed"] == ref.webster_plans(_s, REVERSED)["15 min"]
    assert webster_hour == ref.webster(float(np.mean([d[2] for d in DEMAND])),
                                       float(np.mean([d[3] for d in DEMAND])), _s)  # fmt: skip
    for _k in ("hour", "15 min", "retimed"):
        check([list(p) for p in plans[_k]], f"signal/plans/{_k}")
    check({"Y": rnd(webster_hour["Y"], 3), "cycle_s": rnd(webster_hour["cycle_s"], 1)},
          "signal/webster_hour")  # fmt: skip
    check([{"Y": rnd(w["Y"], 3), "cycle_s": rnd(w["cycle_s"], 1)} for w in webster_15],
          "signal/webster_15")  # fmt: skip
    return plans, webster, webster_15, webster_hour, webster_plans


@app.cell
def _(DEMAND, FIG, HOUR_S, LOST_S, PALETTE, mo, np, plans, plt, sat, webster_hour):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=FIG)
    _t = np.array([d[0] for d in DEMAND] + [HOUR_S]) / 60
    _a.stairs([d[2] for d in DEMAND], _t, color=PALETTE[0], lw=2, label="north-south")
    _a.stairs([d[3] for d in DEMAND], _t, color=PALETTE[1], lw=2, ls="--", label="east-west")
    _a.set(xlabel="minute", ylabel="veh/h per approach", ylim=(0, 800), xticks=[0, 15, 30, 45, 60])
    _a.legend(fontsize=8, frameon=False)
    _rows = [plans["hour"][0], *plans["15 min"]]
    _x = np.arange(len(_rows))
    _ns = np.array([r[1] for r in _rows])
    _ew = np.array([r[2] for r in _rows])
    _b.bar(_x, _ns, color=PALETTE[0], label="north-south green")
    _b.bar(_x, _ew, bottom=_ns, color=PALETTE[1], hatch="//", label="east-west green")
    _b.bar(_x, LOST_S, bottom=_ns + _ew, color="#cccccc", label="yellow + all-red")
    _b.set(xticks=_x, xticklabels=["hour", "0-15", "15-30", "30-45", "45-60"],
           ylabel="seconds per cycle")  # fmt: skip
    _b.legend(fontsize=7, frameon=False)
    _h = plans["hour"][0]
    mo.vstack([
        mo.md(f"""
    ### Webster's plan: a first baseline

    The baseline engineers use: Webster (1958). The cycle $C_0 = (1.5L + 5)/(1 - Y)$, with $L$
    the lost time ({LOST_S} s: two changes of yellow and all-red) and $Y$ the sum of the
    critical flow ratios $q/s$; the greens share the rest of the cycle by flow ratio. The
    saturation flow $s$ is measured in the simulator, as the cell above did: a standing queue
    of {sat["queue"]} cars discharged at {sat["headway_s"]:.1f} s headways,
    {sat["veh_per_h"]:,.0f} vehicles per hour of green (checked against the script's
    measurement, bit for bit). One plan for the hour uses its mean flows (Y =
    {webster_hour["Y"]:.3f}, a {webster_hour["cycle_s"]:.1f} s cycle: greens of {_h[1]} s and
    {_h[2]} s). The plans per 15 minutes know the schedule in advance. The formula counts every
    yellow and all-red as lost time and the displayed green as effective green, so its cycle is
    a starting point, not the best fixed plan: §4 tunes it on training hours. The demand is
    assumed, not counted.
    """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(
    ALL_RED_S,
    APPROACHES,
    END_S,
    ET,
    EVAL_SEEDS,
    HOUR_S,
    STUCK,
    SignalEnv,
    WORK,
    YELLOW_S,
    files,
    np,
    plans,
    ref,
):
    def fixed_cycle(g_ns: int, g_ew: int) -> tuple[tuple[int, int], ...]:
        """One fixed-time cycle as (SUMO phase, seconds): each green, its yellow, its all-red."""
        return ((0, g_ns), (1, YELLOW_S), (2, ALL_RED_S), (3, g_ew), (4, YELLOW_S), (5, ALL_RED_S))

    def trip_summary(path: object, left: dict) -> dict:
        """Mean delay per vehicle, and the side street's service, from a trip file; then delete it.

        The delay is the time lost while driving plus the wait to enter the network. Vehicles
        still inside, or never let in, when the run stops are written too (SUMO's
        write-unfinished and write-undeparted), with the delay they had accrued by then: a
        starved approach counts, as a lower bound.

        Args:
            path: The tripinfo file (our own SUMO output), deleted once read.
            left: The vehicles still waiting and inside at the stop, from ``close``.

        Returns:
            The vehicles, the mean delay (time loss plus depart delay) in seconds, and the side
            street's vehicles and how many of them finished their trip.
        """
        trips = ET.parse(path).getroot().findall("tripinfo")
        path.unlink()
        loss = np.array([float(t.attrib["timeLoss"]) + float(t.attrib["departDelay"])
                         for t in trips])  # fmt: skip
        # Finished: arrived, not cut off at the end ("vaporized" is set only for those)
        done = np.array([t.attrib.get("vaporized", "") == "" and float(t.attrib["arrival"]) >= 0
                         for t in trips])  # fmt: skip
        # A vehicle's id is its flow's, "f<quarter>_<approach>", then ".<index>"
        origin = [t.attrib["id"].rsplit(".", 1)[0].split("_", 1)[1] for t in trips]
        assert set(origin) <= set(APPROACHES)
        side = np.array([o in STUCK for o in origin])
        assert len(trips) - done.sum() == left["pending"] + left["inside"]  # every vehicle
        return {"vehicles": len(trips), "delay_s": float(loss.mean()),
                "side_vehicles": int(side.sum()), "side_finished": int((side & done).sum())}  # fmt: skip

    def run_fixed(files: dict, seed: int, plan: list, trip: object, routes: str = "demand") -> dict:
        """A fixed-time plan, second by second, through the hour and then until the network clears.

        Args:
            files: From ``write_network``.
            seed: The demand draw.
            plan: (start s, NS green s, EW green s) entries; each takes over at a cycle start.
            trip: The trip file to write (read and deleted here).
            routes: Which demand.

        Returns:
            The trip summary, the halted vehicle-seconds of the hour, the vehicles left at the
            cap, and the light each second of the hour.
        """
        with SignalEnv(files, seed, routes=routes, tripinfo=trip) as env:
            sim = env.sumo.simulation
            halted, t = 0.0, 0
            while sim.getTime() < HOUR_S or sim.getMinExpectedNumber() > 0:
                _, g_ns, g_ew = next(p for p in reversed(plan) if p[0] <= t)
                for phase, dur in fixed_cycle(g_ns, g_ew):
                    env.hold(phase)
                    for _ in range(dur):
                        if t >= END_S:
                            break
                        h = env.advance(1)
                        halted += h if t < HOUR_S else 0.0
                        t += 1
                if t >= END_S:
                    break
            lights = np.array(env.lights[:HOUR_S])
            left = env.close()
        return {**trip_summary(trip, left), "halted": halted, **left, "_lights": lights}

    def run_actuated(files: dict, seed: int, trip: object, routes: str = "demand") -> dict:
        """SUMO's actuated program, untouched, through the hour and until the network clears.

        Args:
            files: From ``write_network``.
            seed: The demand draw.
            trip: The trip file to write (read and deleted here).
            routes: Which demand.

        Returns:
            The trip summary, the halted vehicle-seconds of the hour, the vehicles left at the
            cap, and the light each second of the hour.
        """
        with SignalEnv(files, seed, routes=routes, tripinfo=trip, actuated=True) as env:
            sim = env.sumo.simulation
            halted, t = 0.0, 0
            while t < END_S and (t < HOUR_S or sim.getMinExpectedNumber() > 0):
                h = env.advance(1)
                halted += h if t < HOUR_S else 0.0
                t += 1
            lights = np.array(env.lights[:HOUR_S])
            left = env.close()
        return {**trip_summary(trip, left), "halted": halted, **left, "_lights": lights}

    # The same runs as the figure script's, on one held-out hour
    for _name, _mine_run, _ref_run in (
        ("fixed", lambda t: run_fixed(files, EVAL_SEEDS[0], plans["hour"], t),
         lambda t: ref.run_fixed(files, EVAL_SEEDS[0], plans["hour"], t)),
        ("actuated", lambda t: run_actuated(files, EVAL_SEEDS[0], t),
         lambda t: ref.run_actuated(files, EVAL_SEEDS[0], t)),
    ):  # fmt: skip
        _mine = _mine_run(WORK / f"check_{_name}.xml")
        _ref = _ref_run(WORK / f"ref_{_name}.xml")
        assert np.array_equal(_mine.pop("_lights"), _ref.pop("_lights")), _name
        assert _mine == _ref, (_name, _mine, _ref)
        print(f"{_name} on hour {EVAL_SEEDS[0]}: {_mine}")
    return fixed_cycle, run_actuated, run_fixed, trip_summary


@app.cell
def _(mo):
    mo.md(r"""
    ### A deep Q-network

    Replace the table by a network $Q_\theta(s, \cdot)$: ten inputs, two hidden layers of 64
    ReLU units, two outputs, one Q per green. The reward is minus the vehicles stopped, each
    second, so rewards sum to minus the vehicle-seconds of waiting, and the controller takes
    the green with the **larger** Q (rewards, not costs). Q-learning's update becomes a loss on
    a batch of past decisions $(s, a, r, s')$:

    $$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a'), \qquad
    \mathcal{L}(\theta) = \operatorname{mean}\, \text{Huber}\big(Q_\theta(s, a) - y\big).$$

    **No state here is terminal.** A training hour stops at 3,600 s, but the queues are still
    there: the hour's end is a time limit. The observation has no clock, so a state stored as
    terminal at the end of one hour looks like states that are not terminal in all the others.
    Every target therefore bootstraps from the next state, the last decision of an hour too
    (Pardo et al., 2018, "Time limits in reinforcement learning").

    | Part | Setting | Why |
    |:-----|:--------|:----|
    | Reward | minus the vehicles stopped, each second (divided by 100) | delay |
    | γ per five-second decision | 0.99 | a horizon of about 100 decisions, 8.3 minutes |
    | Exploration | ε from 1 to 0.05 over 30,000 decisions | try both greens |
    | Replay buffer | 50,000 decisions, batches of 64 | break the correlation |
    | Target network $\theta^-$ | copied every 500 decisions | a target that holds still |
    | Loss, step | Huber; Adam at 1e-3; gradients clipped at norm 10 | outliers, stability |

    Mnih et al. (2015), as CS 441 presents it, scaled down. Each training run is 200 simulated
    hours on demand draws never used to evaluate (a decision every five seconds, ten when the
    green changes), one CPU thread, deterministic kernels.
    **Online**, the same network learns from each decision once, as it happens, with no replay
    and no target network (the target uses $Q_\theta$ itself).
    """)
    return


@app.cell
def _(
    DQN_BATCH,
    DQN_BUFFER,
    DQN_LR,
    DQN_TARGET_EVERY,
    EPISODES,
    EPS_DECISIONS,
    EPS_END,
    EVAL_SEEDS,
    GAMMA_RL,
    HIDDEN,
    REWARD_SCALE,
    SEED,
    SignalEnv,
    nn,
    np,
    time,
    torch,
):
    def mlp(n_out: int) -> nn.Sequential:
        """Ten inputs, two ReLU layers of ``HIDDEN`` units, ``n_out`` outputs."""
        return nn.Sequential(
            nn.Linear(10, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_out),
        )  # fmt: skip

    def torch_setup(seed: int) -> None:
        """One CPU thread and deterministic kernels, so a run repeats exactly."""
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)

    def episode_seeds(seed: int, episodes: int = EPISODES) -> np.ndarray:
        """The demand draws a training run sees, never one of the held-out hours.

        Args:
            seed: The training run's seed.
            episodes: Simulated hours.

        Returns:
            One SUMO seed per hour.
        """
        rng = np.random.default_rng([SEED, seed])
        draws = rng.choice(np.arange(10_000, 1_000_000), size=episodes, replace=False)
        assert not set(draws.tolist()) & set(EVAL_SEEDS)
        return draws

    def train_dqn(files: dict, seed: int, replay: bool, episodes: int = EPISODES) -> dict:
        """Deep Q-learning, with replay and a target network, or online (each decision once).

        Args:
            files: From ``write_network``.
            seed: The run's seed (initial weights, exploration, replay draws, demand draws).
            replay: True for the DQN; False learns from each decision once, as it happens.
            episodes: Simulated training hours.

        Returns:
            The trained weights, each hour's halted vehicle-seconds and decisions, and the time.
        """
        torch_setup(seed)
        rng = np.random.default_rng([SEED, seed, 1])
        q = mlp(2)
        target = mlp(2) if replay else None
        if replay:
            target.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=DQN_LR)
        # The replay buffer: the last DQN_BUFFER decisions, a ring (no "done" flag: no state
        # is terminal, the hour's end is a time limit)
        S = np.zeros((DQN_BUFFER, 10), np.float32)
        A = np.zeros(DQN_BUFFER, np.int64)
        R = np.zeros(DQN_BUFFER, np.float32)
        S2 = np.zeros((DQN_BUFFER, 10), np.float32)
        n, curve, decisions, t0 = 0, [], [], time.perf_counter()
        batch = DQN_BATCH if replay else 1
        for ep_seed in episode_seeds(seed, episodes):
            with SignalEnv(files, int(ep_seed)) as env:
                obs, hour_over, hour, k = env.observe(), False, 0.0, 0
                while not hour_over:
                    eps = max(EPS_END, 1.0 - n / EPS_DECISIONS)  # linear decay, then flat
                    if rng.random() < eps:
                        a = int(rng.integers(2))
                    else:
                        with torch.no_grad():
                            a = int(q(torch.from_numpy(obs)).argmax())
                    obs2, r, hour_over = env.step(a)
                    hour -= r
                    k += 1
                    i = n % DQN_BUFFER
                    S[i], A[i], R[i], S2[i] = obs, a, r / REWARD_SCALE, obs2
                    n += 1
                    obs = obs2
                    if n < batch:
                        continue
                    # A random batch of past decisions (replay), or the one that just happened
                    idx = rng.integers(0, min(n, DQN_BUFFER), batch) if replay else np.array([i])
                    s, a_, r_, s2 = (torch.from_numpy(X[idx]) for X in (S, A, R, S2))
                    with torch.no_grad():  # the target: one step of hindsight, held still
                        nxt = (target if replay else q)(s2).max(1).values
                    y = r_ + GAMMA_RL * nxt  # always bootstrapped: the hour's end is a time limit
                    pred = q(s).gather(1, a_[:, None]).squeeze(1)
                    assert pred.shape == y.shape == (batch,)
                    loss = nn.functional.smooth_l1_loss(pred, y)  # Huber, beta = 1
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()
                    if replay and n % DQN_TARGET_EVERY == 0:
                        target.load_state_dict(q.state_dict())
            curve.append(hour)
            decisions.append(k)
        return {"state": q.state_dict(), "curve": curve, "decisions": decisions,
                "seconds": time.perf_counter() - t0}  # fmt: skip

    # The Huber loss by hand against PyTorch's smooth L1 (beta = 1): quadratic inside |x| < 1
    _g = torch.Generator().manual_seed(SEED)
    _x, _y = 3 * torch.randn(1000, generator=_g), torch.randn(1000, generator=_g)
    _e = _x - _y
    _huber = torch.where(_e.abs() < 1, 0.5 * _e**2, _e.abs() - 0.5).mean()
    assert torch.allclose(_huber, nn.functional.smooth_l1_loss(_x, _y), rtol=1e-6, atol=1e-7)
    return episode_seeds, mlp, torch_setup, train_dqn


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: learn from each step?

    CS 441 on learning from each step of one game at a time: "may not be efficient", "may be
    unstable".

    **Predict:** the same network, learning from each decision once, as it happens: no replay,
    no target network. Better or worse than Webster's plans? Than the DQN? Two minutes,
    written.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### PPO: improve, but not too far

    **Policy learning** skips Q: a network outputs the policy itself, $\pi_\theta(a \mid s)$, a
    probability for each green, and a second network, the critic, predicts the state's value
    $V_\phi(s)$. The policy-gradient theorem says the expected return rises along
    $\mathbb{E}\big[\nabla_\theta \log \pi_\theta(a \mid s)\, A(s, a)\big]$, where the
    **advantage** $A$ is how much better the action did than the critic expected. Here $A$ is
    estimated over the following decisions by GAE (Schulman et al., 2016):

    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
    A_t = \sum_{k \ge 0} (\gamma\lambda)^k \delta_{t+k}, \quad \lambda = 0.95,$$

    the sum running to the hour's last decision, where $V(s_{t+1})$ is the critic's value of
    the state the hour ends in (a time limit, as for the DQN: the queues do not vanish).

    PPO (Schulman et al., 2017) reuses each batch for several epochs, so the new policy drifts
    from the one that collected the data. With the ratio $r_t = \pi_\text{new}(a_t \mid s_t) /
    \pi_\text{old}(a_t \mid s_t)$ it maximizes

    $$\operatorname{mean}_t \min\big(r_t A_t,\ \operatorname{clip}(r_t, 1 - 0.2, 1 + 0.2)\, A_t\big)
    + 0.01\, \text{entropy}.$$

    Past $1 \pm 0.2$ the objective stops paying, so one batch has no incentive to drag the
    policy far; at $r_t = 1$ its gradient is the policy gradient above. The entropy bonus keeps
    the policy from becoming sure too early. Here the actor and the critic are separate networks
    (tanh layers of 64), each with its own Adam (3e-4 and 1e-3); every two simulated hours, ten
    epochs of minibatches of 256. In a trial run with one shared body, PPO learned to hold one
    green; a likely reason, not measured, is that the critic's large early errors dominate the
    shared gradients.
    """)
    return


@app.cell
def _(
    EPISODES,
    GAMMA_RL,
    HIDDEN,
    PPO_CLIP,
    PPO_ENTROPY,
    PPO_EPISODES_PER_UPDATE,
    PPO_EPOCHS,
    PPO_LAMBDA,
    PPO_LR_ACTOR,
    PPO_LR_CRITIC,
    PPO_MINIBATCH,
    REWARD_SCALE,
    SEED,
    SignalEnv,
    episode_seeds,
    nn,
    np,
    ref,
    time,
    torch,
    torch_setup,
):
    def tanh_mlp(n_out: int) -> nn.Sequential:
        """PPO's networks: ten inputs, two tanh layers of ``HIDDEN`` units, ``n_out`` outputs."""
        return nn.Sequential(
            nn.Linear(10, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, n_out),
        )  # fmt: skip

    class ActorCritic(nn.Module):
        """A policy over the two greens (the actor) and a value of the state (the critic)."""

        def __init__(self) -> None:
            """Two separate networks (a trial with one shared body learned to hold one green)."""
            super().__init__()
            self.actor = tanh_mlp(2)
            self.critic = tanh_mlp(1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Action logits and the state's value."""
            return self.actor(x), self.critic(x).squeeze(-1)

    def gae(
        rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, ends: np.ndarray
    ) -> np.ndarray:
        """Generalized advantage estimates, backwards through the decisions, reset at each hour's end.

        Args:
            rewards: One per decision (float32).
            values: The critic's value of each decision's state (float32).
            next_values: The critic's value of each decision's next state; at an hour's last
                decision, of the state the hour ends in (float32).
            ends: True at an hour's last decision.

        Returns:
            A_t = delta_t + gamma lambda A_{t+1} within the hour, with
            delta_t = r_t + gamma V(s_{t+1}) - V(s_t), in float32.
        """
        adv = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + GAMMA_RL * next_values[t] - values[t]  # bootstrapped always
            running = delta + GAMMA_RL * PPO_LAMBDA * (0.0 if ends[t] else running)
            adv[t] = running
        return adv

    def ppo_rollout(ac: ActorCritic, files: dict, seeds: np.ndarray) -> dict:
        """Run the current stochastic policy for a few simulated hours; keep every decision.

        Args:
            ac: The actor-critic.
            files: From ``write_network``.
            seeds: One demand draw per hour.

        Returns:
            Observations, actions, log-probabilities, scaled rewards, values, hour ends, the
            critic's value of each hour's last state, each hour's halted vehicle-seconds and its
            decisions.
        """
        keys = ("obs", "act", "logp", "rew", "val", "end", "v_end", "hours", "n")
        out: dict = {k: [] for k in keys}
        for ep_seed in seeds:
            with SignalEnv(files, int(ep_seed)) as env:
                obs, hour_over, hour, k = env.observe(), False, 0.0, 0
                while not hour_over:
                    with torch.no_grad():
                        logits, v = ac(torch.from_numpy(obs))
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()  # PPO explores by sampling its own policy
                    obs2, r, hour_over = env.step(int(a))
                    hour -= r
                    k += 1
                    out["obs"].append(obs)
                    out["act"].append(int(a))
                    out["logp"].append(float(dist.log_prob(a)))
                    out["rew"].append(r / REWARD_SCALE)
                    out["val"].append(float(v))
                    out["end"].append(hour_over)
                    obs = obs2
                with torch.no_grad():  # the hour stops, the state does not end: bootstrap from it
                    out["v_end"].append(float(ac(torch.from_numpy(obs))[1]))
            out["hours"].append(hour)
            out["n"].append(k)
        return out

    def ppo_update(ac: ActorCritic, opts: tuple, roll: dict, rng: np.random.Generator) -> None:
        """Ten epochs of minibatch steps on the clipped objective and on the critic's error.

        Args:
            ac: The actor-critic.
            opts: The actor's and the critic's optimizers.
            roll: From ``ppo_rollout``.
            rng: Shuffles the minibatches.
        """
        opt_pi, opt_v = opts
        values = np.array(roll["val"], np.float32)
        ends = np.array(roll["end"], bool)
        next_values = np.empty_like(values)
        next_values[:-1] = values[1:]  # the next decision's state, within an hour
        assert ends[-1] and ends.sum() == len(roll["v_end"])
        next_values[ends] = np.array(roll["v_end"], np.float32)  # each hour's last state
        adv = gae(np.array(roll["rew"], np.float32), values, next_values, ends)
        ret = torch.from_numpy(adv + values)  # the critic's target
        adv_t = torch.from_numpy(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        obs_t = torch.from_numpy(np.array(roll["obs"]))
        act_t, logp_t = torch.tensor(roll["act"]), torch.tensor(roll["logp"])
        n = len(roll["act"])
        assert obs_t.shape == (n, 10) and adv_t.shape == ret.shape == (n,)
        for _ in range(PPO_EPOCHS):
            perm = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, PPO_MINIBATCH):
                j = perm[i : i + PPO_MINIBATCH]
                logits, v = ac(obs_t[j])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_t[j]) - logp_t[j])
                clipped = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP)
                loss_pi = -torch.min(ratio * adv_t[j], clipped * adv_t[j]).mean()
                loss_pi = loss_pi - PPO_ENTROPY * dist.entropy().mean()
                opt_pi.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(ac.actor.parameters(), 0.5)
                opt_pi.step()
                loss_v = ((v - ret[j]) ** 2).mean()
                opt_v.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(ac.critic.parameters(), 10.0)
                opt_v.step()

    def train_ppo(files: dict, seed: int, episodes: int = EPISODES) -> dict:
        """PPO: roll out two simulated hours, update, repeat.

        Args:
            files: From ``write_network``.
            seed: The run's seed.
            episodes: Simulated training hours.

        Returns:
            The trained weights, each hour's halted vehicle-seconds and decisions, and the time.
        """
        torch_setup(seed)
        rng = np.random.default_rng([SEED, seed, 2])
        ac = ActorCritic()
        opts = (torch.optim.Adam(ac.actor.parameters(), lr=PPO_LR_ACTOR),
                torch.optim.Adam(ac.critic.parameters(), lr=PPO_LR_CRITIC))  # fmt: skip
        seeds = episode_seeds(seed, episodes)
        curve, decisions, t0 = [], [], time.perf_counter()
        for start in range(0, episodes, PPO_EPISODES_PER_UPDATE):
            roll = ppo_rollout(ac, files, seeds[start : start + PPO_EPISODES_PER_UPDATE])
            curve += roll["hours"]
            decisions += roll["n"]
            ppo_update(ac, opts, roll, rng)
        return {"state": ac.state_dict(), "curve": curve, "decisions": decisions,
                "seconds": time.perf_counter() - t0}  # fmt: skip

    # GAE checked twice: against the script's function (bit for bit) and against the
    # definition, a discounted sum of TD errors within each hour, each hour's last one
    # bootstrapped from the value of the state it ends in (to float32 round-off)
    _rng = np.random.default_rng(SEED)
    _r = _rng.normal(size=50).astype(np.float32)
    _v = _rng.normal(size=50).astype(np.float32)
    _v_end = _rng.normal(size=2).astype(np.float32)  # the states hours 1 and 2 end in
    _ends = np.zeros(50, bool)
    _ends[[19, 49]] = True
    _nv = np.empty_like(_v)
    _nv[:-1] = _v[1:]
    _nv[_ends] = _v_end
    _adv = gae(_r, _v, _nv, _ends)
    assert np.array_equal(_adv, ref.gae(_r, _v, _nv, _ends))
    _next = [_v_end[0] if t == 19 else _v_end[1] if t == 49 else _v[t + 1] for t in range(50)]
    _delta = [_r[t] + GAMMA_RL * _next[t] - _v[t] for t in range(50)]
    _direct = [
        sum((GAMMA_RL * PPO_LAMBDA) ** (k - t) * _delta[k] for k in range(t, 20 if t < 20 else 50))
        for t in range(50)
    ]
    assert np.allclose(_adv, _direct, rtol=1e-5, atol=1e-5)
    return ActorCritic, gae, train_ppo


@app.cell
def _(FIG, PALETTE, PPO_CLIP, np, plt):
    _r = np.linspace(0.0, 2.0, 400)
    _fig, _axes = plt.subplots(1, 2, figsize=FIG)
    for _ax, _A, _title in ((_axes[0], 1.0, "advantage > 0"), (_axes[1], -1.0, "advantage < 0")):
        _ax.plot(_r, _r * _A, color="grey", ls="--", lw=1.2, label="r·A (unclipped)")
        _obj = np.minimum(_r * _A, np.clip(_r, 1 - PPO_CLIP, 1 + PPO_CLIP) * _A)
        _ax.plot(_r, _obj, color=PALETTE[2], lw=2.2, label="PPO: min(r·A, clip(r)·A)")
        _ax.axvspan(1 - PPO_CLIP, 1 + PPO_CLIP, color=PALETTE[2], alpha=0.08, lw=0)
        _ax.set(title=_title, xlabel="r = π_new(a|s) / π_old(a|s)")
        _ax.legend(fontsize=8, frameon=False)
    _axes[0].set_ylabel("objective")
    _fig
    return


@app.cell
def _(CHECK_EPISODES, KINDS, files, mo, ref, time, torch, train_dqn, train_ppo):
    # Weight for weight: this notebook's training against the figure script's, for one seed and a
    # few simulated hours each (the script's run length is its module constant EPISODES, set
    # here for the check and restored)
    _t0 = time.perf_counter()
    weight_check = {}
    _saved = ref.EPISODES
    try:
        for _kind, _n in CHECK_EPISODES.items():
            ref.EPISODES = _n
            if _kind == "ppo":
                _mine, _theirs = train_ppo(files, 0, _n), ref.train_ppo(files, 0)
            else:
                _replay = _kind == "dqn"
                _mine = train_dqn(files, 0, _replay, _n)
                _theirs = ref.train_dqn(files, 0, replay=_replay)
            _sm, _st = _mine["state"], _theirs["state"]
            assert list(_sm) == list(_st) and _mine["curve"] == _theirs["curve"], _kind
            assert all(torch.equal(_sm[_k], _st[_k]) for _k in _sm), _kind
            weight_check[KINDS[_kind]] = {
                "hours": _n,
                "tensors": len(_sm),
                "weights": sum(int(_v.numel()) for _v in _sm.values()),
                "max_abs_difference": max(float((_sm[_k] - _st[_k]).abs().max()) for _k in _sm),
            }
    finally:
        ref.EPISODES = _saved
    assert ref.EPISODES == 200
    t_weights = time.perf_counter() - _t0
    _rows = "\n".join(
        f"| {k} | {v['hours']} | {v['tensors']} | {v['weights']:,} | {v['max_abs_difference']:g} |"
        for k, v in weight_check.items()
    )
    mo.md(f"""
    **Weight for weight.** Each method trained here and by the figure script's own function,
    seed 0, one thread, deterministic kernels ({t_weights:.0f} s):

    | Method | Training hours | Tensors | Weights | Largest difference |
    |:-------|---------------:|--------:|--------:|-------------------:|
    {_rows}

    The training curves (halted vehicle-seconds per hour) are equal too. So a run here repeats
    the slides' run of the same seed; in the slides setting every delay below must equal
    `numbers.json`.
    """)
    return t_weights, weight_check


@app.cell
def _(
    APPROACHES,
    DECISION_S,
    END_S,
    FAULT_S,
    GREEN_PHASE,
    HOUR_S,
    NS,
    OBS_SCALE,
    Parallel,
    SEED,
    SLUG,
    STUCK,
    SignalEnv,
    WORKERS,
    ActorCritic,
    delayed,
    fixed_cycle,
    mlp,
    np,
    run_actuated,
    run_fixed,
    torch,
    torch_setup,
    train_dqn,
    train_ppo,
    trip_summary,
):
    def greedy(kind: str, state: dict) -> object:
        """A trained network as a controller: the green with the larger Q (or probability).

        PPO trains a stochastic policy; here it is run greedily, by its most likely green.

        Args:
            kind: "dqn", "online", or "ppo".
            state: The trained weights.

        Returns:
            A function from an observation to a green.
        """
        if kind == "ppo":
            model = ActorCritic()
            model.load_state_dict(state)
            net = model.actor
        else:
            net = mlp(2)
            net.load_state_dict(state)
        net.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                return int(net(torch.from_numpy(obs)).argmax())

        return act

    def run_learned(
        files: dict,
        seed: int,
        act: object,
        trip: object,
        *,
        routes: str = "demand",
        stuck: tuple[str, ...] = (),
        fallback: list | None = None,
    ) -> dict:
        """A learned controller on one held-out hour, then until the network clears (or the cap).

        Args:
            files: From ``write_network``.
            seed: The demand draw.
            act: The controller.
            trip: The trip file to write (read and deleted here).
            routes: Which demand.
            stuck: Approaches whose detectors read zero.
            fallback: A fixed-time plan to switch to once an approach has read no vehicle for
                ``FAULT_S`` (a detector fault), or None. The plan's entry is chosen by the time.

        Returns:
            The trip summary, the vehicles left, when a fault was declared, the light each
            second of the hour, and the side street's queue after each decision.
        """
        with SignalEnv(files, seed, routes=routes, stuck=stuck, tripinfo=trip) as env:
            sim, edge = env.sumo.simulation, env.sumo.edge
            obs = env.observe()
            silent = dict.fromkeys(APPROACHES, 0.0)
            fault_at, side_queue = None, []
            cap = END_S - 2 * DECISION_S  # a decision with a change of green takes 10 s
            while sim.getTime() < cap and (
                sim.getTime() < HOUR_S or sim.getMinExpectedNumber() > 0
            ):
                t0 = int(sim.getTime())
                if fallback is not None and fault_at is None:
                    counts = obs[4:8] * OBS_SCALE  # what each approach's detector reports
                    for k, a in enumerate(APPROACHES):
                        silent[a] = silent[a] + DECISION_S if counts[k] == 0 else 0.0
                    if max(silent.values()) >= FAULT_S:
                        fault_at = t0
                if fault_at is not None:  # one cycle of the fixed plan, then look again
                    _, g_ns, g_ew = next(p for p in reversed(fallback) if p[0] <= t0)
                    for phase, dur in fixed_cycle(g_ns, g_ew):
                        env.hold(phase)
                        env.advance(min(dur, END_S - int(sim.getTime())))  # never past --end
                    env.phase, env.since = NS, 0
                    env.hold(GREEN_PHASE[NS])
                    obs = env.observe()
                else:
                    obs, _, _ = env.step(act(obs))
                waiting = sum(edge.getLastStepHaltingNumber(s) for s in STUCK)  # the truth
                side_queue.append((int(sim.getTime()), waiting))
            lights = np.array(env.lights[:HOUR_S])
            left = env.close()
        return {
            **trip_summary(trip, left),
            **left,
            "fault_at_s": fault_at,
            "_lights": lights,
            "_side_queue": side_queue,
        }

    def green_share(lights: np.ndarray, window_s: int = 300) -> np.ndarray:
        """North-south's share of the green seconds in each window (change intervals left out).

        Args:
            lights: The light each second of the hour.
            window_s: Window length.

        Returns:
            One share per window.
        """
        w = lights[:HOUR_S].reshape(-1, window_s)
        ns, ew = (w == 0).sum(1), (w == 1).sum(1)
        return ns / np.maximum(ns + ew, 1)

    def to_numpy(state: dict) -> dict:
        """Weights as NumPy arrays, to cross process boundaries by plain pickling."""
        return {k: v.detach().cpu().numpy().copy() for k, v in state.items()}

    def to_torch(state: dict) -> dict:
        """The inverse of ``to_numpy``."""
        return {k: torch.from_numpy(v) for k, v in state.items()}

    def run_job(spec: dict) -> dict:
        """One unit of SUMO work, in its own process (libsumo runs one simulation per process).

        Args:
            spec: ``task`` "train" (``kind``, ``seed``, ``episodes``), "fixed" (``plan``),
                "actuated", or "eval" (``kind``, ``state``, ``stuck``, ``fallback``). Every
                evaluation carries ``name``, ``slug``, ``seed``, ``train_seed``, and ``routes``;
                all carry ``files`` and ``tmp``.

        Returns:
            The task's result, with the spec's identifying fields.
        """
        torch_setup(spec.get("seed", SEED))
        files, tmp, task = spec["files"], spec["tmp"], spec["task"]
        if task == "train":
            if spec["kind"] == "ppo":
                out = train_ppo(files, spec["seed"], spec["episodes"])
            else:
                out = train_dqn(files, spec["seed"], spec["kind"] == "dqn", spec["episodes"])
            out["state"] = to_numpy(out["state"])
            return {**out, "kind": spec["kind"], "seed": spec["seed"]}
        assert SLUG.fullmatch(spec["slug"]), f"not a slug: {spec['slug']!r}"
        trip = tmp / f"{spec['slug']}_{spec['train_seed']}_{spec['seed']}_{spec['routes']}.xml"
        if task == "fixed":
            out = run_fixed(files, spec["seed"], spec["plan"], trip, spec["routes"])
        elif task == "actuated":
            out = run_actuated(files, spec["seed"], trip, spec["routes"])
        else:
            assert task == "eval", task
            act = greedy(spec["kind"], to_torch(spec["state"]))
            out = run_learned(files, spec["seed"], act, trip, routes=spec["routes"],
                              stuck=spec["stuck"], fallback=spec["fallback"])  # fmt: skip
        keep = ("name", "kind", "seed", "train_seed", "routes")
        return {**out, **{k: spec[k] for k in keep if k in spec}}

    def run_pool(specs: list[dict]) -> list[dict]:
        """Run jobs in worker processes (joblib's loky, spawned), results in submission order.

        Args:
            specs: The jobs.

        Returns:
            Their results.
        """
        return Parallel(n_jobs=min(WORKERS, len(specs)))(delayed(run_job)(s) for s in specs)

    return green_share, greedy, run_pool, to_torch


@app.cell
def _(KINDS, RUN_EPISODES, RUN_TRAIN_SEEDS, TRIPS, files, mo, np, run_pool, time):
    _t0 = time.perf_counter()
    _specs = [{"task": "train", "kind": _k, "seed": _s, "episodes": RUN_EPISODES, "files": files,
               "tmp": TRIPS} for _k in KINDS for _s in RUN_TRAIN_SEEDS]  # fmt: skip
    trained = run_pool(_specs)
    t_train = time.perf_counter() - _t0
    assert [(t["kind"], t["seed"]) for t in trained] == [(s["kind"], s["seed"]) for s in _specs]
    assert all(len(t["curve"]) == RUN_EPISODES for t in trained)
    _rows = "\n".join(
        f"| {KINDS[t['kind']]} | {t['seed']} | {t['seconds'] / 60:.1f} | "
        f"{np.mean(t['decisions']):.0f} | {np.mean(t['curve'][-20:]) / 3600:.2f} |"
        for t in trained
    )
    mo.md(f"""
    **Training** ({len(trained)} runs of {RUN_EPISODES} simulated hours, in
    {t_train / 60:.1f} min of wall time):

    | Method | Seed | Minutes | Decisions per hour | Waiting, last 20 hours (veh-h per h) |
    |:-------|-----:|--------:|-------------------:|-------------------------------------:|
    {_rows}
    """)
    return t_train, trained


@app.cell
def _(
    EVAL_SEEDS,
    TRIPS,
    TUNE_HOURS,
    TUNE_LADDER,
    TUNE_START,
    check,
    episode_seeds,
    files,
    mo,
    np,
    plans,
    rnd,
    run_pool,
    sat,
    time,
    webster_plans,
):
    # The tuned fixed-time plan: Webster's 15-minute greens times one factor, chosen as the
    # DQN's weights were, on training hours (here the first ten of training run 0), never on
    # the held-out ones. A bracketing search: while the best factor sits at an edge of those
    # tried, try the next one of the ladder past it; the kept factor must be interior.
    _t0 = time.perf_counter()
    _seeds = [int(x) for x in episode_seeds(0)[:TUNE_HOURS]]
    assert not set(_seeds) & set(EVAL_SEEDS)
    tune_delay, _grid = {}, list(TUNE_START)
    while _grid:
        _res = run_pool([
            {"task": "fixed", "name": "tuning", "slug": f"tune_{round(100 * _f)}",
             "plan": webster_plans(sat["veh_per_h"], factor=_f)["15 min"], "seed": _e,
             "routes": "demand", "train_seed": 0, "files": files, "tmp": TRIPS}
            for _f in _grid
            for _e in _seeds
        ])  # fmt: skip
        for _k, _f in enumerate(_grid):
            _rs = _res[_k * len(_seeds) : (_k + 1) * len(_seeds)]
            tune_delay[_f] = float(np.mean([r["delay_s"] for r in _rs]))
        _best = min(tune_delay, key=tune_delay.__getitem__)
        _i = TUNE_LADDER.index(_best)
        if _best == min(tune_delay) and _i > 0:
            _grid = [TUNE_LADDER[_i - 1]]
        elif _best == max(tune_delay) and _i < len(TUNE_LADDER) - 1:
            _grid = [TUNE_LADDER[_i + 1]]
        else:
            _grid = []
    assert min(tune_delay) < _best < max(tune_delay), f"not interior: {tune_delay}"
    tune_factor = _best
    tuned_plan = webster_plans(sat["veh_per_h"], factor=tune_factor)["15 min"]
    assert webster_plans(sat["veh_per_h"], factor=1.0)["15 min"] == plans["15 min"]
    tuning_num = {"hours": len(_seeds), "factor": tune_factor,
                  "delay_s": {f"{f:.1f}": rnd(d, 2) for f, d in sorted(tune_delay.items())}}  # fmt: skip
    check(tuning_num, "signal/tuning")
    check([list(p) for p in tuned_plan], "signal/plans/tuned")
    t_tune = time.perf_counter() - _t0
    _rows = "\n".join(f"| {f:.1f} | {d:.2f} |" for f, d in sorted(tune_delay.items()))
    mo.md(f"""
    **A fixed plan tuned like the DQN.** Webster's formula takes the whole intergreen as lost
    time and the displayed green as effective green, so its cycle is only a starting point.
    One factor on the 15-minute greens, chosen on {len(_seeds)} training hours (never the
    held-out ones; {t_tune:.0f} s): ×{tune_factor:.1f}, an interior point of what was tried.

    | Factor on Webster's greens | Mean delay on the training hours, s |
    |---------------------------:|------------------------------------:|
    {_rows}

    The plan: {", ".join(f"{g1} s and {g2} s from minute {b // 60}" for b, g1, g2 in tuned_plan)}
    (north-south, east-west).
    """)
    return t_tune, tune_factor, tuned_plan, tuning_num


@app.cell
def _(
    EVAL_SEEDS,
    KINDS,
    ROUTES,
    RUN_EVAL_SEEDS,
    STUCK,
    TRIPS,
    CTRL_SLUG,
    files,
    plans,
    run_pool,
    time,
    trained,
    tuned_plan,
):
    _t0 = time.perf_counter()
    _base = {"files": files, "tmp": TRIPS, "train_seed": 0}
    _plan = {"Webster, hour": plans["hour"], "Webster, 15 min": plans["15 min"],
             "Fixed, tuned": tuned_plan, "Webster, 15 min, retimed": plans["retimed"]}  # fmt: skip
    # The untrained controllers on all ten held-out hours (cheap), in every setting, on both
    # demands; on the reversed demand also the 15-minute plans an engineer retimed for it
    _untrained = []
    for _routes in ROUTES:
        _names = [*_plan][:3] + (["Webster, 15 min, retimed"] if _routes == "reversed" else [])
        _untrained += [
            {**_base, "task": "fixed", "plan": _plan[_n], "seed": _e, "name": _n,
             "slug": CTRL_SLUG[_n], "routes": _routes}
            for _n in _names
            for _e in EVAL_SEEDS
        ]  # fmt: skip
        _untrained += [
            {**_base, "task": "actuated", "seed": _e, "name": "Actuated", "slug": "actuated",
             "routes": _routes}
            for _e in EVAL_SEEDS
        ]  # fmt: skip
    _learned = [
        {**_base, "task": "eval", "kind": _t["kind"], "state": _t["state"], "seed": _e,
         "train_seed": _t["seed"], "stuck": (), "fallback": None, "name": KINDS[_t["kind"]],
         "slug": _t["kind"], "routes": _routes}
        for _routes in ROUTES
        for _t in trained
        for _e in RUN_EVAL_SEEDS
    ]  # fmt: skip
    # The DQNs with the side street's detectors dead: alone, with the slides' fallback to the
    # hourly plan, and (not on the slides) with a fallback to the plans per 15 minutes
    _cases = (("dead", "dead", None), ("dead, fallback", "dead_fallback", plans["hour"]),
              ("dead, fallback 15 min", "dead_fallback_15min", plans["15 min"]))  # fmt: skip
    _stuck = [
        {**_base, "task": "eval", "kind": "dqn", "state": _t["state"], "seed": _e,
         "train_seed": _t["seed"], "stuck": STUCK, "fallback": _fb, "name": _n, "slug": _slug,
         "routes": "demand"}
        for _n, _slug, _fb in _cases
        for _t in trained
        if _t["kind"] == "dqn"
        for _e in RUN_EVAL_SEEDS
    ]  # fmt: skip
    _results = run_pool(_untrained + _learned + _stuck)
    fixed_runs = _results[: len(_untrained)]  # the untrained controllers, fixed and actuated
    learned_runs = _results[len(_untrained) : len(_untrained) + len(_learned)]
    stuck_runs = _results[len(_untrained) + len(_learned) :]
    assert len(stuck_runs) == len(_stuck)
    assert not list(TRIPS.iterdir()), "a trip file was left behind"
    t_eval = time.perf_counter() - _t0
    print(f"{len(_results)} evaluation runs in {t_eval:.0f} s")
    return fixed_runs, learned_runs, stuck_runs, t_eval


@app.cell
def _(
    APPROACHES,
    EVAL_SEEDS,
    NS,
    PALETTE,
    SignalEnv,
    VIEW_M,
    files,
    greedy,
    mo,
    np,
    plt,
    slide,
    time,
    to_torch,
    trained,
):
    _t0 = time.perf_counter()

    def busiest_decision(files: dict, act: object) -> int:
        """The end of the decision with the most stopped vehicles in the first held-out hour.

        Args:
            files: From ``write_network``.
            act: The controller.

        Returns:
            The simulated second at which that decision ended.
        """
        with SignalEnv(files, EVAL_SEEDS[0]) as env:
            obs, hour_over, best, best_t = env.observe(), False, -1, 0
            while not hour_over:
                obs, _, hour_over = env.step(act(obs))
                stopped = sum(env.sumo.edge.getLastStepHaltingNumber(a) for a in APPROACHES)
                if stopped > best:
                    best, best_t = stopped, int(env.sumo.simulation.getTime())
        return best_t

    def take_snapshot(files: dict, act: object) -> dict:
        """Every car's place at the busiest moment: a second, deterministic run stops there.

        Args:
            files: From ``write_network``.
            act: The controller.

        Returns:
            The second, the green, and each car's position relative to the junction, whether it
            is stopped, and whether it is on an approach (not an exit road or the junction).
        """
        at_s = busiest_decision(files, act)
        with SignalEnv(files, EVAL_SEEDS[0]) as env:
            obs = env.observe()
            while env.sumo.simulation.getTime() < at_s:
                obs, _, _ = env.step(act(obs))
            assert int(env.sumo.simulation.getTime()) == at_s, (
                "the rerun did not stop where the first did"
            )
            v = env.sumo.vehicle
            ids = v.getIDList()
            cx, cy = env.sumo.junction.getPosition("C")  # netconvert moves the origin to a corner
            return {"t": int(env.sumo.simulation.getTime()), "phase": env.phase,
                    "xy": [(v.getPosition(i)[0] - cx, v.getPosition(i)[1] - cy) for i in ids],
                    "halted": [v.getSpeed(i) < 0.1 for i in ids],
                    "approach": [v.getRoadID(i) in APPROACHES for i in ids]}  # fmt: skip

    _dqn0 = next(t for t in trained if t["kind"] == "dqn" and t["seed"] == 0)
    snap = take_snapshot(files, greedy("dqn", to_torch(_dqn0["state"])))
    _on, _halted = np.array(snap["approach"]), np.array(snap["halted"])
    assert not (_halted & ~_on).any(), "a vehicle stopped off the approaches"
    snapshot_num = {
        "t": snap["t"],
        "minute": snap["t"] // 60,
        "in_network": len(snap["xy"]),
        "on_approaches": int(_on.sum()),
        "halted": int((_halted & _on).sum()),
        "drawn": int(sum(max(abs(x), abs(y)) <= VIEW_M for x, y in snap["xy"])),
    }
    t_snapshot = time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(5.2, 4.2))
    _ax.set_aspect("equal")
    _ax.set(xlim=(-VIEW_M, VIEW_M), ylim=(-VIEW_M, VIEW_M), xticks=[], yticks=[])
    for _s in _ax.spines.values():
        _s.set_visible(False)
    _ax.add_patch(plt.Rectangle((-VIEW_M, -6), 2 * VIEW_M, 12, color="#d9d9d9", lw=0))
    _ax.add_patch(plt.Rectangle((-6, -VIEW_M), 12, 2 * VIEW_M, color="#d9d9d9", lw=0))
    for (_x, _y), _h in zip(snap["xy"], snap["halted"], strict=True):
        _ax.add_patch(plt.Circle((_x, _y), 3.5, color=PALETTE[1] if _h else PALETTE[0], lw=0))
    _green = "north-south" if snap["phase"] == NS else "east-west"
    _ax.set_title(
        f"t = {snap['t']} s, green: {_green}; orange = stopped, blue = moving", fontsize=9
    )
    mo.vstack([
        mo.md(f"""
    **The intersection** at the busiest moment of the first held-out hour under the trained DQN
    (seed 0): minute {snapshot_num["minute"]}, {snapshot_num["on_approaches"]} vehicles on the
    four approaches, {snapshot_num["halted"]} of them stopped ({snapshot_num["in_network"]} in
    the whole network, exit roads and the junction included; the picture shows the
    {VIEW_M} m nearest the junction, {snapshot_num["drawn"]} vehicles). The slides' run: minute
    {slide("signal/snapshot/minute")}, {slide("signal/snapshot/on_approaches")} and
    {slide("signal/snapshot/halted")}: {"the same" if snapshot_num == slide("signal/snapshot") else "not the same (this setting's DQN trained for fewer hours, or on another CPU)"}.
    """),
        _fig,
    ])  # fmt: skip
    return snapshot_num, t_snapshot


@app.cell
def _(
    CONTROLLERS,
    EVAL_SEEDS,
    KINDS,
    UNTRAINED,
    check,
    fixed_runs,
    green_share,
    learned_runs,
    np,
    pct,
    rnd,
    trained,
):
    # The summary the figure script writes, from this notebook's runs
    def runs_of(name: str, routes: str = "demand") -> list[dict]:
        """One controller's held-out runs on one demand, untrained or learned."""
        rs = [r for r in fixed_runs + learned_runs if r["name"] == name and r["routes"] == routes]
        assert rs, (name, routes)
        return rs

    def by_run(rs: list[dict]) -> np.ndarray:
        """The mean delay over the held-out hours, per training run (one if untrained)."""
        seeds = sorted({r["train_seed"] for r in rs})
        return np.array(
            [np.mean([r["delay_s"] for r in rs if r["train_seed"] == s]) for s in seeds]
        )

    def paired(rs: list[dict], base: list[dict]) -> dict:
        """A trained controller against an untrained one on the same held-out hours."""
        ref_delay = {r["seed"]: r["delay_s"] for r in base}
        seeds = sorted({r["train_seed"] for r in rs})
        diff = [
            np.array([r["delay_s"] - ref_delay[r["seed"]] for r in rs if r["train_seed"] == s])
            for s in seeds
        ]
        every = np.concatenate(diff)
        return {
            "better_by_run": [int((d < 0).sum()) for d in diff],
            "mean_diff_s_by_run": [rnd(d.mean(), 2) for d in diff],
            "better": int((every < 0).sum()),
            "of": len(every),
            "mean_diff_s": rnd(every.mean(), 2),
        }

    def unfinished(rs: list[dict]) -> int:
        """Vehicles still waiting or inside at the cap, summed over runs."""
        return int(sum(r["pending"] + r["inside"] for r in rs))

    for _name in UNTRAINED:
        assert [r["seed"] for r in runs_of(_name)] == list(EVAL_SEEDS)
        assert unfinished(runs_of(_name)) == 0, f"{_name} did not clear"
    delay = {_n: by_run(runs_of(_n)) for _n in CONTROLLERS}
    hours_halted = {_n: float(np.mean([r["halted"] for r in runs_of(_n)])) for _n in UNTRAINED}
    train_curves = {
        _lab: [t["curve"] for t in trained if t["kind"] == _k] for _k, _lab in KINDS.items()
    }
    _rev_names = [*CONTROLLERS[:2], "Webster, 15 min, retimed", *CONTROLLERS[2:]]
    _rev = {_n: by_run(runs_of(_n, "reversed")) for _n in _rev_names}
    signal_results = {
        "delay_s": {k: [rnd(x, 1) for x in v] for k, v in delay.items()},
        "delay_mean_s": {k: rnd(v.mean(), 1) for k, v in delay.items()},
        "unfinished": {_n: unfinished(runs_of(_n)) for _n in CONTROLLERS},
        "paired": {_lab: {_b: paired(runs_of(_lab), runs_of(_b))
                          for _b in ("Webster, 15 min", "Fixed, tuned", "Actuated")}
                   for _lab in KINDS.values()},
        "reversed": {
            "delay_s": {k: [rnd(x, 1) for x in v] for k, v in _rev.items()},
            "delay_mean_s": {k: rnd(v.mean(), 1) for k, v in _rev.items()},
            "unfinished": {_n: unfinished(runs_of(_n, "reversed")) for _n in _rev_names},
        },
        "last20_vehicle_hours": {k: rnd(np.mean([np.mean(c[-20:]) for c in v]) / 3600, 2)
                                 for k, v in train_curves.items()},
    }  # fmt: skip
    share = {
        _n: np.mean([green_share(r["_lights"]) for r in runs_of(_n)], axis=0)
        for _n in ("Webster, hour", "Webster, 15 min", "Actuated", "DQN")
    }
    signal_results["green_share_ns_pct"] = {k: [pct(x, 0) for x in v] for k, v in share.items()}
    train_seconds = {_lab: int(round(np.mean([t["seconds"] for t in trained if t["kind"] == _k])))
                     for _k, _lab in KINDS.items()}  # fmt: skip
    # The untrained controllers need no training: equal to the slides' in every setting, on
    # both demands
    for _name in UNTRAINED:
        for _key in ("delay_s", "delay_mean_s", "unfinished"):
            check(signal_results[_key][_name], f"signal/{_key}/{_name}")
            check(signal_results["reversed"][_key][_name], f"signal/reversed/{_key}/{_name}")
        if _name != "Fixed, tuned":
            check(signal_results["green_share_ns_pct"][_name], f"signal/green_share_ns_pct/{_name}")
    for _key in ("delay_s", "delay_mean_s", "unfinished"):
        _n = "Webster, 15 min, retimed"
        check(signal_results["reversed"][_key][_n], f"signal/reversed/{_key}/{_n}")
    return delay, hours_halted, share, signal_results, train_curves, train_seconds


@app.cell
def _(FIG, KINDS, PALETTE, hours_halted, mo, np, plt, signal_results, slide, train_curves):
    _colors = {"DQN": PALETTE[0], "DQN, online": PALETTE[3], "PPO": PALETTE[2]}
    _styles = {"DQN": "-", "DQN, online": "--", "PPO": "-."}
    _fig, _ax = plt.subplots(figsize=FIG)
    for _lab, _runs in train_curves.items():
        _c = np.array(_runs) / 3600  # vehicle-hours of waiting in each training hour
        _k = min(10, _c.shape[1])  # a ten-hour moving average
        _sm = np.array([np.convolve(r, np.ones(_k) / _k, mode="valid") for r in _c])
        _x = np.arange(_k, _c.shape[1] + 1)
        _ax.plot(_x, _sm.mean(0), color=_colors[_lab], ls=_styles[_lab], lw=1.8, label=_lab)
        _ax.fill_between(_x, _sm.min(0), _sm.max(0), color=_colors[_lab], alpha=0.18, lw=0)
    for _n, _c, _ls in (("Webster, hour", "#9a9a9a", ":"), ("Webster, 15 min", "#6f6f6f", ":"),
                        ("Actuated", PALETTE[1], "--")):  # fmt: skip
        _ax.axhline(hours_halted[_n] / 3600, color=_c, ls=_ls, lw=1.2, label=_n)
    _ax.set(
        xlabel="simulated hours of training",
        ylabel="waiting, vehicle-hours per hour",
        ylim=(0, None),
    )
    _ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    _l20 = signal_results["last20_vehicle_hours"]
    _rows = "\n".join(
        f"| {lab} | {_l20[lab]:.2f} | {slide('signal/last20_vehicle_hours')[lab]:.2f} |"
        for lab in KINDS.values()
    )
    mo.vstack([
        mo.md(f"""
    ### Replay and a target net learn more

    The same network, the same 200 hours. Waiting per training hour, vehicle-hours, averaged
    over the last 20 training hours (exploration still on: ε = 0.05 for the DQNs, and PPO
    samples its policy, so these are not the held-out scores that follow). The comparison
    changes three things at once: the online network has no replay, no target network, and a
    batch of one decision instead of 64, at the same Adam rate; which of them matters most was
    not tested. The lines: the baselines' waiting on the held-out hours.

    | Method | This run | Slides (3 seeds) |
    |:-------|---------:|-----------------:|
    {_rows}
    """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(CONTROLLERS, FIG, PALETTE, SETTING, delay, mo, np, plt, signal_results, slide):
    _names = CONTROLLERS
    _colors = ["#9a9a9a", "#6f6f6f", "#3d3d3d", PALETTE[1], PALETTE[0], PALETTE[3], PALETTE[2]]
    _fig, _ax = plt.subplots(figsize=FIG)
    for _i, (_n, _c) in enumerate(zip(_names, _colors, strict=True)):
        _vals = delay[_n]
        _ax.bar(_i, _vals.mean(), color=_c, alpha=0.85)
        _ax.scatter(np.full(len(_vals), _i), _vals, color="black", s=12, zorder=3)
        _ax.scatter([_i + 0.25], [slide("signal/delay_mean_s")[_n]], marker="x", color="black",
                    s=30, zorder=3, label="slides" if _i == 0 else None)  # fmt: skip
        _ax.text(_i, _vals.mean() + 0.6, f"{_vals.mean():.1f}", ha="center", fontsize=9)
    _ax.set(xticks=range(len(_names)), xticklabels=[n.replace(", ", ",\n") for n in _names],
            ylabel="delay per vehicle, s")  # fmt: skip
    _ax.legend(fontsize=8, frameon=False)
    _dm, _ref = signal_results["delay_mean_s"], slide("signal/delay_mean_s")
    for _n in _names:
        print(f"delay {_n}: this run {signal_results['delay_s'][_n]} (mean {_dm[_n]}); "
              f"slides {slide('signal/delay_s')[_n]} (mean {_ref[_n]})")  # fmt: skip
    _rows = "\n".join(
        f"| {n} | {', '.join(f'{x:.1f}' for x in signal_results['delay_s'][n])} | {_dm[n]:.1f} | "
        f"{', '.join(f'{x:.1f}' for x in slide('signal/delay_s')[n])} | {_ref[n]:.1f} |"
        for n in _names
    )
    mo.vstack([
        mo.md(f"""
    ### Actuated control is the bar

    Mean delay per vehicle on held-out demand draws never seen in training; every vehicle
    counts, including any still waiting when the network had to clear. Two baselines an
    engineer tries first: the fixed plan tuned on training hours (above), and SUMO's own
    actuated control, untuned (each green runs 5 to 50 s, extended while its detectors see
    cars; no schedule, no training). The untrained controllers are always run on all ten
    held-out hours. This setting (`{SETTING}`) ran the learned controllers {"on every training run and held-out hour of the slides" if SETTING == "slides" else "on fewer training runs and held-out hours than the slides"}
    (crosses: the slides' means; dots: this run's training runs). PPO is run greedily, by its
    most likely green.

    | Controller | This run, by training run | Mean | Slides, by training run | Mean |
    |:-----------|:--------------------------|-----:|:------------------------|-----:|
    {_rows}
    """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(DEMAND, FIG, HOUR_S, PALETTE, mo, np, plt, share):
    _t = (np.arange(HOUR_S // 300) + 0.5) * 5
    _fig, _ax = plt.subplots(figsize=FIG)
    for (_n, _v), _c, _m in zip(
        share.items(), ["#9a9a9a", "#6f6f6f", PALETTE[1], PALETTE[0]], "s^Do", strict=True
    ):
        _ax.plot(_t, 100 * _v, marker=_m, ms=4, lw=1.6, color=_c, label=_n)
    demand_share = np.array([d[2] / (d[2] + d[3]) for d in DEMAND])
    _ax.stairs(100 * demand_share, np.array([0, 15, 30, 45, 60]), color="grey", lw=1.2, ls=":",
               label="north-south share of demand")  # fmt: skip
    _ax.set(xlabel="minute", ylabel="north-south share of green, %", xticks=[0, 15, 30, 45, 60])
    _ax.legend(fontsize=8, frameon=False)
    mo.vstack([
        mo.md(r"""
    ### The DQN follows the demand

    North-south's share of the green seconds, by five minutes (change intervals left out). Nobody
    told the DQN the schedule: it reads only the queues, and its split follows the demand's
    (measured by quarter hour in §7's table). Actuated control, which reads only its detectors,
    follows it too. Webster's 15-minute plans follow the demand more closely, but only because
    they were computed from it.
    """),
        _fig,
    ])  # fmt: skip
    return (demand_share,)


@app.cell
def _(mo, signal_results, slide, train_seconds):
    _dm = signal_results["delay_mean_s"]
    _ts = slide("signal/train_seconds")
    mo.md(f"""
    ### DQN or PPO?

    | | DQN | PPO |
    |:--|:--|:--|
    | Learns | Q for each action | a policy and a value |
    | Actions | discrete, one output each | discrete or continuous |
    | Data | off-policy: replays old decisions | on-policy: each batch, then discarded |
    | Exploration | ε-greedy | sampling its own policy |
    | Here: delay; training | {_dm["DQN"]:.1f} s; {train_seconds["DQN"] / 60:.1f} min a run | {_dm["PPO"]:.1f} s; {train_seconds["PPO"] / 60:.1f} min a run |
    | The slides | {slide("signal/delay_mean_s/DQN"):.1f} s; {_ts["DQN"] / 60:.1f} min a run | {slide("signal/delay_mean_s/PPO"):.1f} s; {_ts["PPO"] / 60:.1f} min a run |

    Training times are on one CPU thread each, several runs at once on a shared machine. PPO's
    on-policy data is used for ten epochs and then thrown away, while the DQN replays each decision many times; whether that explains the gap here was
    not tested, on the slides or in this notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Four ways to learn a policy

    | CS 441's approach | Learns | Here |
    |:------------------|:-------|:-----|
    | Behavioural cloning | the expert's action, given the state | what Illinois did, by rating |
    | Q-learning | the value of each state and action | Q-learning; the DQN |
    | Policy learning | the policy, and a value to judge it | PPO |
    | Model-based | the world's transitions; then plan | value iteration on counted P |

    The first three are model-free: they never estimate $P$. Value iteration on the inventory's
    transition matrix is model-based: count the world, then plan. When the model can be
    counted, as with bridges, planning is exact and cheap; when it cannot (a signal's queues in
    continuous time), learn from a simulator.
    """)
    return


@app.cell
def _(FIG, N_S, PALETTE, RATINGS, check, label, mo, np, pairs, pct, plt, records, ref, vi):
    def agency_behaviour(df: object, pairs: object) -> dict:
        """What Illinois did, by rating: recorded work, and bridges that left the inventory.

        Args:
            df: The records.
            pairs: The two-year pairs.

        Returns:
            Per rating: the share of pairs with recorded work (Item 106); the share of records
            before 2024 whose bridge is absent from the next file (replaced, removed, or
            renumbered); and the records counted.
        """
        work = pairs.groupby("s")["work"].mean().reindex(RATINGS)
        last = df["nbi_year"].max()
        nxt_year = df.groupby("structure", sort=False)["nbi_year"].shift(-1)
        before = df[df["nbi_year"] < last].assign(gone=nxt_year[df["nbi_year"] < last].isna())
        gone = before.groupby("s")["gone"].mean().reindex(RATINGS)
        n = before.groupby("s").size().reindex(RATINGS)
        return {"work": work.to_numpy(float), "gone": gone.to_numpy(float), "n": n.to_numpy(int)}

    agency = agency_behaviour(records, pairs)
    _theirs = ref.agency_behaviour(ref.load_records(), pairs)
    assert all(np.array_equal(agency[k], _theirs[k]) for k in agency)
    agency_num = {
        int(r): {"gone_pct": pct(g, 1), "work_pct": pct(w, 1), "n": int(n)}
        for r, g, w, n in zip(RATINGS, agency["gone"], agency["work"], agency["n"], strict=True)
    }
    check(agency_num, "bridge/agency")
    _x = np.arange(N_S)
    _fig, _ax = plt.subplots(figsize=FIG)
    _ax.bar(_x - 0.2, 100 * agency["gone"], 0.4, color=PALETTE[1], label="left the inventory")
    _ax.bar(_x + 0.2, 100 * agency["work"], 0.4, color=PALETTE[0], hatch="//",
            label="recorded work (Item 106)")  # fmt: skip
    for _i in np.flatnonzero(vi["2%"]["policy"] == 1):
        _ax.axvspan(_i - 0.5, _i + 0.5, color=PALETTE[2], alpha=0.12, lw=0)
    _ax.set(xticks=_x, xticklabels=[label(r) for r in RATINGS], xlabel="lowest rating",
            ylabel="within two years, %", title="shaded: the MDP at 2% replaces")  # fmt: skip
    _ax.legend(fontsize=8, frameon=False)
    mo.vstack([
        mo.md(f"""
    ### What Illinois did, by rating

    Behavioural cloning would learn this: act at 3 and 4, rarely above. Within two years,
    {agency_num[3]["gone_pct"]}% of bridges rated 3 or worse left the inventory (replaced,
    removed, or renumbered), {agency_num[4]["gone_pct"]}% at 4, {agency_num[5]["gone_pct"]}% at
    5; recorded work stays under {np.ceil(100 * np.nanmax(agency["work"])):.0f}% at every rating.
    The MDP agrees on where (at 4 and below), not on how often: it replaces every poor bridge at
    once. Budgets may explain the gap; the MDP has no budget. A clone of the agency inherits its
    budget and its habits; the MDP inherits our cost judgement.
    """),
        _fig,
    ])  # fmt: skip
    return agency, agency_num


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: predict three failures

    1. Is a bridge's rating Markov? Does a rating-6 bridge drop faster when it is old?
    2. If it does, does the replacement policy change?
    3. The side street's detector cabinet loses power: both east-west detectors read zero. What
       does the DQN do? What would you add?

    Each question tests an assumption: the MDP assumes the rating is the whole state; the DQN
    assumes its sensors tell the truth. Two minutes, written; then three reveals.
    """)
    return


@app.cell
def _(
    AGE_BANDS,
    FIG,
    N_S,
    PALETTE,
    RATINGS,
    Z95,
    check,
    mo,
    np,
    pairs,
    pairwise,
    pct,
    pd,
    plt,
    ref,
    time,
):
    _t0 = time.perf_counter()

    def age_band(age: object) -> object:
        """Each age's band: [0, 20] (a bridge in its first year included), (20, 40], (40, 60], over 60."""
        assert (age >= 0).all() and (age <= AGE_BANDS[-1]).all()
        band = pd.cut(age, AGE_BANDS, right=True, include_lowest=True, labels=False)
        assert band.notna().all()
        return band.astype(int)

    def wilson(k: int, n: int) -> tuple[float, float]:
        """The 95% Wilson score interval of k successes in n: never below 0 or above 1."""
        p = k / n
        den = 1.0 + Z95**2 / n
        centre = (p + Z95**2 / (2 * n)) / den
        half = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / den
        return float(centre - half), float(centre + half)

    def markov_test(pairs: object) -> dict:
        """Does the chance of a drop depend on age, given the rating? Flat lines if Markov.

        Args:
            pairs: The two-year pairs.

        Returns:
            For ratings 4 to 8 and each age band, the pairs and the share that dropped within
            two years, with a 95% Wilson interval (a bridge's pairs treated as independent, so
            somewhat too narrow); the do-nothing counts of each band; and the pairs left out
            because the bridge was built after the inspection (data errors).
        """
        data = pairs[~pairs["work"] & (pairs["s2"] <= pairs["s"])]
        negative = int((data["age"] < 0).sum())
        data = data[data["age"] >= 0].copy()
        data["band"] = age_band(data["age"])
        from_rating = {}
        for r in range(4, 9):
            rows = []
            for b in range(len(AGE_BANDS) - 1):
                g = data[(data["s"] == r) & (data["band"] == b)]
                k = int((g["s2"] < r).sum())
                lo, hi = wilson(k, len(g))
                rows.append({"band": b, "n": len(g), "p": k / len(g), "lo": lo, "hi": hi})
            from_rating[r] = rows
        counts = [count_band(data[data["band"] == b]) for b in range(len(AGE_BANDS) - 1)]
        return {"by_rating": from_rating, "counts": counts, "negative_ages": negative}

    def count_band(rows: object) -> np.ndarray:
        """Transition counts of one age band's pairs, rows the rating now."""
        C = np.zeros((N_S, N_S))
        np.add.at(C, (rows["s"].to_numpy() - RATINGS[0], rows["s2"].to_numpy() - RATINGS[0]), 1)
        return C

    markov = markov_test(pairs)
    _theirs = ref.markov_test(pairs)
    assert markov["by_rating"] == _theirs["by_rating"]
    assert markov["negative_ages"] == _theirs["negative_ages"]
    assert all(0 <= x["lo"] <= x["p"] <= x["hi"] <= 1 for v in markov["by_rating"].values()
               for x in v)  # fmt: skip
    assert all(
        np.array_equal(a, b) for a, b in zip(markov["counts"], _theirs["counts"], strict=True)
    )
    markov_num = {int(r): [{"n": x["n"], "pct": pct(x["p"], 1)} for x in rows]
                  for r, rows in markov["by_rating"].items()}  # fmt: skip
    check(markov_num, "bridge/markov")
    t_markov = time.perf_counter() - _t0
    _names = [f"{a}-{b}" if b < 200 else f"over {a}" for a, b in pairwise(AGE_BANDS)]
    _fig, _ax = plt.subplots(figsize=FIG)
    _colors = [*PALETTE, "#E69F00"]
    for _k, (_r, _rows) in enumerate(markov["by_rating"].items()):
        _p = np.array([x["p"] for x in _rows])
        _err = 100 * np.vstack([_p - [x["lo"] for x in _rows], [x["hi"] for x in _rows] - _p])
        _ax.errorbar(np.arange(4) + 0.06 * (_k - 2), 100 * _p, yerr=_err, marker="osD^v"[_k],
                     capsize=3, lw=1.4, color=_colors[_k], label=f"rating {_r}")  # fmt: skip
    _ax.set(xticks=range(4), xticklabels=_names, xlabel="age (years since built)",
            ylabel="dropped within two years, %")  # fmt: skip
    _ax.legend(fontsize=8, frameon=False, ncol=5)
    _m6 = markov_num[6]
    mo.vstack([
        mo.md(f"""
    ### Age matters: the rating is not Markov

    A Markov chain on the rating alone predicts flat lines. The data: at rating 6, a bridge 40
    to 60 years old dropped within two years {_m6[2]["pct"]}% of the time ({_m6[2]["n"]:,}
    pairs), one under 20 {_m6[0]["pct"]}% ({_m6[0]["n"]:,} pairs). The rating does not carry
    the whole state. CS 441's DQN fed four stacked frames for the same reason: one frame does
    not say which way the ball moves. The fix is the same: add what is missing to the state.
    """),
        _fig,
    ])  # fmt: skip
    return age_band, markov, markov_num, t_markov


@app.cell
def _(
    AGE_BANDS,
    GAMMAS,
    N_S,
    PENALTY,
    POOR,
    RATINGS,
    YEARS_PER_STEP,
    age_band,
    check,
    costs,
    evaluate_policy,
    markov,
    mdp,
    np,
    pct,
    records,
    ref,
    rnd,
    time,
    value_iteration,
):
    _t0 = time.perf_counter()

    def age_mdp(
        mdp: dict, band_counts: list, gamma: float, penalty: float, start: np.ndarray
    ) -> dict:
        """The MDP with age in the state (rating x 20-year band), solved and compared.

        A bridge moves into the next band with probability 2/20 per step (its next band's first
        two years); the oldest band keeps it. A replacement starts a new bridge in the youngest
        band. A band's rating with no pairs borrows the pooled counts.

        Args:
            mdp: The rating-only MDP.
            band_counts: Do-nothing counts per band.
            gamma: Discount factor per step.
            penalty: The poor-condition cost.
            start: The 2024 distribution over (band, rating), band-major.

        Returns:
            The do-nothing matrix, each band's policy, the rating-only policy, both policies'
            expected costs from the 2024 bridges, and the rating-only policy's extra cost.
        """
        nb = len(band_counts)
        P_band = []
        for raw in band_counts:
            Cb = raw.copy()
            empty = Cb.sum(1) == 0
            Cb[empty] = mdp["C0"][empty]
            P_band.append(Cb / Cb.sum(1, keepdims=True))
        move = YEARS_PER_STEP / (AGE_BANDS[1] - AGE_BANDS[0])
        K = N_S * nb
        P = np.zeros((2, K, K))
        for b in range(nb):
            nxt = [(b, 1 - move), (b + 1, move)] if b < nb - 1 else [(b, 1.0)]
            for s in range(N_S):
                for b2, w in nxt:
                    P[0, b * N_S + s, b2 * N_S : (b2 + 1) * N_S] += w * P_band[b][s]
                P[1, b * N_S + s, :N_S] = mdp["p_new"]  # a new bridge, youngest band
        assert np.allclose(P.sum(2), 1.0, rtol=0, atol=1e-12)
        C = np.tile(costs(penalty), (nb, 1))
        best = value_iteration(P, C, gamma)
        rating_only = value_iteration(mdp["P"], costs(penalty), gamma)["policy"]
        V_ro = evaluate_policy(P, C, gamma, np.tile(rating_only, nb))  # ignore age, act by rating
        cost_best, cost_ro = float(start @ best["V"]), float(start @ V_ro)
        return {"P0": P[0], "policy": best["policy"].reshape(nb, N_S), "rating_only": rating_only,
                "cost_best": cost_best, "cost_rating_only": cost_ro,
                "regret": cost_ro / cost_best - 1.0}  # fmt: skip

    def first_passage(P0: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Expected years until the chain first enters a state outside ``up``, doing nothing.

        Solves t = 1 + P0[up, up] t (Forsyth's recurrences, §13.1, for a hitting time).

        Args:
            P0: Do-nothing transitions.
            up: The states not yet arrived (rating above poor).

        Returns:
            Years from each state (nan where already arrived).
        """
        A = np.eye(up.sum()) - P0[np.ix_(up, up)]
        out = np.full(len(up), np.nan)
        out[up] = YEARS_PER_STEP * np.linalg.solve(A, np.ones(up.sum()))
        return out

    _nb = len(AGE_BANDS) - 1
    _last = records[records["nbi_year"] == records["nbi_year"].max()]
    _age = _last["inspection_year"] - _last["year_built"]
    negative_age = {"pairs": markov["negative_ages"], "bridges_2024": int((_age < 0).sum())}
    check(negative_age, "bridge/negative_age")
    _last = _last[_age >= 0].assign(b=age_band(_age[_age >= 0]))  # age 0 in the youngest band
    start_aged = np.zeros(N_S * _nb)
    _grp = _last.groupby(["s", "b"]).size()
    for (_s, _b), _m in _grp.items():
        start_aged[int(_b) * N_S + int(_s) - RATINGS[0]] += _m
    start_aged /= start_aged.sum()
    aged = age_mdp(mdp, markov["counts"], GAMMAS["2%"], PENALTY, start_aged)
    _theirs = ref.age_mdp(mdp, markov["counts"], GAMMAS["2%"], PENALTY, start_aged)
    assert all(np.array_equal(aged[k], _theirs[k]) for k in ("P0", "policy", "rating_only"))
    assert all(aged[k] == _theirs[k] for k in ("cost_best", "cost_rating_only", "regret"))
    aged_num = {
        "replace_at_by_band": [[int(r) for r in RATINGS[row == 1]] for row in aged["policy"]],
        "same_policy": bool((aged["policy"] == aged["rating_only"]).all()),
        "regret_pct": pct(aged["regret"], 2),
    }
    check(aged_num, "bridge/age_mdp")
    _i6 = int(np.flatnonzero(RATINGS == 6)[0])
    _pooled = first_passage(mdp["P"][0], RATINGS > POOR)
    _by_band = first_passage(aged["P0"], np.tile(RATINGS > POOR, _nb))
    assert np.allclose(
        _pooled, ref.first_passage_years(mdp["P"][0]), rtol=0, atol=0, equal_nan=True
    )
    assert np.allclose(
        _by_band, ref.first_passage_aged(aged["P0"], _nb), rtol=0, atol=0, equal_nan=True
    )
    years_num = {"rating only": rnd(_pooled[_i6], 1)}
    for _k in range(_nb):
        years_num[f"band{_k}"] = rnd(_by_band[_k * N_S + _i6], 1)
    check(years_num, "bridge/years_6_to_poor")
    t_aged = time.perf_counter() - _t0
    print(f"age-aware MDP: {aged_num}; years from 6 to poor {years_num}; left out for a "
          f"negative age: {negative_age}")  # fmt: skip
    return aged, aged_num, negative_age, years_num, t_aged


@app.cell
def _(PENALTY, aged_num, mo, years_num):
    def _at(rs: list[int]) -> str:
        return f"{max(rs)} and below" if rs else "never"

    _b = aged_num["replace_at_by_band"]
    mo.md(f"""
    ### The policy survives the missing state

    With age in the state (four 20-year bands), a rating-6 bridge under 20 reaches poor in
    {years_num["band0"]} years, not the {years_num["rating only"]} the rating-only chain
    predicts (40 to 60 years old: {years_num["band2"]}). The forecast is wrong. The decision:

    | At 2%, poor costing {PENALTY:.2f} | Replaces at |
    |:-------------------------|:------------|
    | Rating only | {_at([3, 4])} |
    | Rating and age, 0-20 years | {_at(_b[0])} |
    | Rating and age, 20-40 years | {_at(_b[1])} |
    | Rating and age, 40-60 years | {_at(_b[2])} |
    | Rating and age, over 60 years | {_at(_b[3])} |
    | Extra cost of ignoring age | {aged_num["regret_pct"]:.2f}% |

    Replacing pays only once a bridge is poor, and a poor bridge costs the same at 4 as at 3,
    so how fast it drops does not change the choice; above 4, no age band drops fast enough to
    make an early replacement pay. So the rating-only policy costs nothing extra in the
    age-aware world (the extra cost is measured from Illinois's 2024 bridges, by rating and
    age). That is a finding about this decision, not a licence: a policy that acted earlier
    (preventive work at 5 or 6) would depend on age.
    """)
    return


@app.cell
def _(
    EVAL_SEEDS,
    FIG,
    HOUR_S,
    PALETTE,
    learned_runs,
    mo,
    np,
    plt,
    rnd,
    slide,
    stuck_runs,
):
    def stuck_summary(learned: list[dict], stuck: list[dict]) -> dict:
        """The DQN with working detectors, dead side-street detectors, and dead with a fallback.

        Args:
            learned: The learned controllers' runs (working detectors).
            stuck: The DQN's runs with the side street's detectors reading zero.

        Returns:
            Per case: the cases' mean delay, in all and per training run; the vehicles
            unfinished at the cap (all, and per hour); per training run, the side street's
            vehicles and how many got through; when the fault was declared; and the runs.
        """
        cases = {"working": [r for r in learned if r["kind"] == "dqn" and r["routes"] == "demand"]}
        for name in ("dead", "dead, fallback", "dead, fallback 15 min"):
            cases[name] = [r for r in stuck if r["name"] == name]
        out = {}
        for name, rs in cases.items():
            seeds = sorted({r["train_seed"] for r in rs})
            per = {s: [r for r in rs if r["train_seed"] == s] for s in seeds}
            n_left = int(sum(r["pending"] + r["inside"] for r in rs))
            out[name] = {
                "delay_s": rnd(np.mean([r["delay_s"] for r in rs]), 1),
                "delay_s_by_run": [
                    rnd(np.mean([r["delay_s"] for r in v]), 1) for v in per.values()
                ],
                "unfinished": n_left,
                "unfinished_per_hour": int(round(n_left / len(rs))),
                "side_vehicles_by_run": [sum(r["side_vehicles"] for r in v) for v in per.values()],
                "side_served_by_run": [sum(r["side_finished"] for r in v) for v in per.values()],
                "fault_at_s": sorted({r["fault_at_s"] for r in rs if r["fault_at_s"] is not None}),
                "runs": len(rs),
            }
        return out, cases

    stuck_num, _cases = stuck_summary(learned_runs, stuck_runs)
    # The queues of the first training run on the first held-out hour, as the slides draw them
    _first = {n: next(r for r in rs if r["train_seed"] == 0 and r["seed"] == EVAL_SEEDS[0])
              for n, rs in _cases.items()}  # fmt: skip
    _queues = {n: np.array(r["_side_queue"]) for n, r in _first.items()}
    stuck_num["dead, fallback"]["fault_at_s_drawn"] = _first["dead, fallback"]["fault_at_s"]
    stuck_extra = {
        n: {
            "max_side_queue": int(q[q[:, 0] <= HOUR_S, 1].max()),
            "side_queue_at_60_min": int(q[q[:, 0] <= HOUR_S, 1][-1]),
            "east_west_green_pct": rnd(100 * np.mean([np.sum(r["_lights"] == 1) / np.sum(r["_lights"] >= 0) for r in _cases[n]]), 1),
        }
        for n, q in _queues.items()
    }  # fmt: skip
    _fig, _ax = plt.subplots(figsize=FIG)
    _styles = {"working": (PALETTE[0], "-"), "dead": (PALETTE[1], "--"),
               "dead, fallback": (PALETTE[2], "-"), "dead, fallback 15 min": (PALETTE[3], ":")}  # fmt: skip
    for _n, _q in _queues.items():
        _keep = _q[:, 0] <= HOUR_S
        _ax.plot(_q[_keep, 0] / 60, _q[_keep, 1], color=_styles[_n][0], ls=_styles[_n][1], lw=1.4,
                 label=_n)  # fmt: skip
    _ax.axvline(_first["dead, fallback"]["fault_at_s"] / 60, color="grey", ls=":", lw=1)
    _ax.set(xlabel="minute", ylabel="side street, vehicles waiting", xticks=[0, 15, 30, 45, 60])
    _ax.legend(fontsize=8, frameon=False, title="east-west detectors", title_fontsize=8)
    _rows = "\n".join(
        f"| {n} | {v['runs']} | {v['delay_s']:.1f} | {', '.join(f'{x:.1f}' for x in v['delay_s_by_run'])} | "
        f"{', '.join(f'{a:,} of {b:,}' for a, b in zip(v['side_served_by_run'], v['side_vehicles_by_run'], strict=True))} | "
        f"{v['unfinished_per_hour']:.0f} | {', '.join(map(str, v['fault_at_s'])) or '—'} | "
        + (f"{slide('signal/stuck')[n]['delay_s']:.1f} | {slide('signal/stuck')[n]['unfinished_per_hour']:.0f} |"
           if n in slide("signal/stuck") else "not run | |")
        for n, v in stuck_num.items()
    )  # fmt: skip
    _d = stuck_num["dead"]
    _never = sum(x == 0 for x in _d["side_served_by_run"])
    mo.vstack([
        mo.md(f"""
    ### Dead detectors starve the side street

    The DQN trusts its inputs. With both east-west detectors reading zero, {_never} of this
    run's {len(_d["delay_s_by_run"])} trained DQNs never turned the side street green (none of
    its vehicles got through); the rest served it, slowly (the table, run by run). Averaging
    the runs would hide two different behaviours. The fix is the one signal controllers already
    carry: an approach whose detector reads no vehicle for 60 decisions in a row is declared
    faulty (five minutes or a little more; the first reading is at t = 0, so with no change of
    green the 60th comes at 295 s), and control falls back to a fixed plan. The slides' test
    falls back to the hourly plan; this notebook also falls back to the plans per 15 minutes
    (not on the slides).

    | East-west detectors | Runs | Delay, s | Delay by training run, s | Side street served, by run | Unfinished per hour | Fault declared, s | Slides: delay | Slides: unfinished |
    |:--------------------|-----:|---------:|:-------------------------|:---------------------------|--------------------:|:------------------|--------------:|-------------------:|
    {_rows}

    Unfinished vehicles are counted at the end of the clearing time, so the delay of a starved
    street is a lower bound.
    """),
        _fig,
    ])  # fmt: skip
    return stuck_extra, stuck_num


@app.cell
def _(PENALTY, mo, no_penalty_replaces, vi):
    assert not no_penalty_replaces and not vi["7%"]["policy"].any()
    mo.md(f"""
    ### Rewards get what they pay for

    | Reward | What the optimizer did |
    |:-------|:-----------------------|
    | Agency cost only, nothing for poor bridges | never replace |
    | The same at 7% a year, poor at {PENALTY:.2f} | never replace |
    | People's approval of a chatbot's answer | agree with the user (sycophancy) |

    The optimizer is not wrong in any row; the reward is. The first two rows are today's bridge
    MDP (computed above: no replacement at any rate without a poor-condition cost, none at 7%
    with it). The third is RLHF's known failure: people prefer answers that agree with them, so
    a reward model learned from their choices rewards agreement (Sharma et al., 2024). The
    engineer's job is the reward, and checking what the optimizer does with it.
    """)
    return


@app.cell
def _(
    GAMMAS,
    GAP_MONTHS,
    agency_num,
    aged_num,
    attempt,
    check,
    crit,
    gamma_num,
    markov_num,
    mdp_num,
    negative_age,
    no_penalty_replaces,
    one_minus_gamma_num,
    q_num,
    ratings_num,
    records,
    rises_as_stays,
    rnd,
    t_aged,
    t_bridge_data,
    t_crit,
    t_markov,
    t_ql,
    t_vi,
    vi_num,
    weight_20_num,
    years_num,
):
    # Every bridge number of the slides, rebuilt here and compared as one object
    bridge_num = {
        "records": len(records),
        "files": sorted(int(y) for y in records["nbi_year"].unique()),
        "gap_months": list(GAP_MONTHS),
        **mdp_num,
        "ratings": ratings_num,
        "gamma": gamma_num,
        "gamma_powers": [rnd(GAMMAS["2%"] ** k, 2) for k in range(6)],
        "one_minus_gamma": one_minus_gamma_num,
        "weight_20_years": weight_20_num,
        "attempt": attempt,
        "vi": vi_num,
        "critical_penalty": {k: rnd(c, 3) for k, c in crit.items()},
        "no_penalty_replaces": no_penalty_replaces,
        "rises_as_stays": rises_as_stays,
        "q_learning": q_num,
        "markov": markov_num,
        "age_mdp": aged_num,
        "years_6_to_poor": years_num,
        "negative_age": negative_age,
        "agency": agency_num,
    }
    check(bridge_num, "bridge")
    bridge_seconds = t_bridge_data + t_vi + t_crit + t_ql + t_markov + t_aged
    print(f"every bridge number equals numbers.json; the bridge part's computing took "
          f"{bridge_seconds:.0f} s")  # fmt: skip
    return bridge_num, bridge_seconds


@app.cell
def _(
    DECISION_S,
    DEMAND,
    DQN_BATCH,
    DQN_BUFFER,
    DQN_TARGET_EVERY,
    EPISODES,
    EPS_DECISIONS,
    EVAL_SEEDS,
    FAULT_S,
    GAMMA_RL,
    HOUR_S,
    PENALTY,
    PPO_CLIP,
    Q_STEPS,
    RATES,
    REVERSED,
    SEED,
    SETTING,
    SLIDES,
    TRAIN_SEEDS,
    UNTRAINED,
    YEARS_PER_STEP,
    actuated_num,
    agree,
    as_json,
    check,
    first_difference,
    importlib,
    plans,
    rnd,
    sat,
    signal_results,
    slide,
    snapshot_num,
    stuck_num,
    torch,
    tuned_plan,
    tuning_num,
    webster_15,
    webster_hour,
):
    # What every setting rebuilds and asserts: the network's numbers, the plans, the settings
    always = {
        "saturation": {"queue": sat["queue"], "headway_s": rnd(sat["headway_s"], 2),
                       "veh_per_h": int(round(sat["veh_per_h"]))},
        "plans": {k: [list(p) for p in v] for k, v in {**plans, "tuned": tuned_plan}.items()},
        "demand": [list(d) for d in DEMAND],
        "reversed_demand": [list(d) for d in REVERSED],
        "actuated": actuated_num,
        "tuning": tuning_num,
        "webster_hour": {"Y": rnd(webster_hour["Y"], 3), "cycle_s": rnd(webster_hour["cycle_s"], 1)},
        "webster_15": [{"Y": rnd(w["Y"], 3), "cycle_s": rnd(w["cycle_s"], 1)} for w in webster_15],
        "settings": {
            "episodes": EPISODES, "decisions_per_hour": HOUR_S // DECISION_S,
            "train_seeds": len(TRAIN_SEEDS), "eval_hours": len(EVAL_SEEDS), "gamma": GAMMA_RL,
            "horizon_minutes": rnd(DECISION_S / (1 - GAMMA_RL) / 60, 1), "buffer": DQN_BUFFER,
            "batch": DQN_BATCH, "target_every": DQN_TARGET_EVERY, "eps_decisions": EPS_DECISIONS,
            "clip": PPO_CLIP, "fault_s": FAULT_S,
        },
    }  # fmt: skip
    for _k, _v in always.items():
        check(_v, f"signal/{_k}")
    _top = slide("settings")
    _mine = {"seed": SEED, "penalty": PENALTY, "rates": RATES, "years_per_step": YEARS_PER_STEP,
             "q_steps": Q_STEPS, "sumo": importlib.metadata.version("libsumo"),
             "eclipse_sumo": importlib.metadata.version("eclipse-sumo"),
             "torch": torch.__version__, "threads_per_run": 1}  # fmt: skip
    assert agree(as_json(_mine), {k: _top[k] for k in _mine}), first_difference(
        as_json(_mine), {k: _top[k] for k in _mine}
    )
    # The learned controllers' numbers: rebuilt here; asserted in the slides setting only
    learned = {
        **signal_results,
        "stuck": {k: v for k, v in stuck_num.items() if k != "dead, fallback 15 min"},
        "snapshot": snapshot_num,
    }
    signal_num = {**always, **learned}
    _ref = {k: v for k, v in slide("signal").items() if k != "train_seconds"}  # timings aside
    # The same structure as the slides' in every setting, so the claims below read either
    assert set(as_json(signal_num)) == set(_ref)
    assert all(
        set(as_json(signal_num)[k]) == set(_ref[k]) for k in _ref if isinstance(_ref[k], dict)
    )
    if SLIDES:
        assert agree(as_json(signal_num), _ref), first_difference(as_json(signal_num), _ref)
        signal_for_claims = signal_num
        _note = "every signal number equals numbers.json (training times aside)"
    else:  # the claims render the slides' learned numbers, which this setting does not rerun
        signal_for_claims = {**slide("signal"), **always}
        _same = sorted(k for k, v in learned.items() if agree(as_json(v), slide(f"signal/{k}")))
        _note = (f"setting {SETTING!r}: network, saturation, plans, tuning, settings, and the "
                 f"untrained controllers ({', '.join(UNTRAINED)}) equal; learned numbers not "
                 f"asserted (equal anyway: {_same or 'none'})")  # fmt: skip
    print(_note)
    return learned, signal_for_claims, signal_num


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Where RL lives today

    | Where | What runs |
    |:------|:----------|
    | Bridge management (AASHTOWare BrM, once Pontis) | Markov deterioration, optimized actions |
    | Chat models: RLHF | a reward learned from people's choices, PPO |
    | Signal control in the field | actuated, or adaptive (SCOOT, SCATS); RL mostly simulated |
    | Games and robots | DQN, PPO, and their successors |

    Pontis (Golabi and Shepard, 1997) brought Markov deterioration models and optimized
    maintenance to US bridge networks; its successor is AASHTOWare BrM. It models deterioration
    as a Markov chain on condition states, the assumption tested in §5. RLHF (Ouyang et al.,
    2022): people rank a model's answers, a reward model learns their preferences, and PPO
    tunes the language model to it. Signal control: many field signals are actuated, extending
    each green while detectors see cars; the adaptive systems some cities run (SCOOT, SCATS)
    retune designed plans. Fixed and actuated control are the baselines RL studies compare
    against, and RL has beaten them mostly in simulation (Wei et al., 2019).
    Lecture 27 turns to fairness and accountability: who bears a policy's errors.

    ### Vote: what should the city install?

    - **A.** Actuated control, as SUMO ships it.
    - **B.** Fixed plans per 15 minutes, tuned and retimed by an engineer.
    - **C.** The DQN.
    - **D.** The DQN, with a detector-fault check that falls back to a fixed plan.

    Vote alone. Discuss with a neighbor for two minutes. Vote again.
    """)
    return


@app.cell
def _(SETTING, mo, signal_for_claims, stuck_num):
    _s = signal_for_claims
    _dm, _st, _rv = _s["delay_mean_s"], _s["stuck"], _s["reversed"]["delay_mean_s"]
    _b = stuck_num["dead, fallback 15 min"]
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        The evidence, in simulation with assumed demand (the slides' run). A:
        {_dm["Actuated"]:.1f} s per vehicle, with no schedule and no training. B:
        {_dm["Fixed, tuned"]:.1f} s tuned ({_dm["Webster, 15 min"]:.1f} s as Webster's formula
        gives it), and it needs the demand schedule, which changes: with the same quarter hours
        in reverse order, the 15-minute plans reach {_rv["Fixed, tuned"]:.1f} s tuned and
        {_rv["Webster, 15 min"]:.1f} s untuned, until an engineer retimes them
        ({_rv["Webster, 15 min, retimed"]:.1f} s). C: {_dm["DQN"]:.1f} s, and
        {_rv["DQN"]:.1f} s on the reversed demand too, but behind A in every run, and at least
        {_st["dead"]["delay_s"]:.1f} s when the side street's detectors die. D: C's delay with
        the fault covered; the slides' test fell back to the hourly plan
        ({_st["dead, fallback"]["delay_s"]:.1f} s). This notebook also fell back to the plans
        per 15 minutes (setting `{SETTING}`, {_b["runs"]} runs): {_b["delay_s"]:.1f} s against
        {stuck_num["dead, fallback"]["delay_s"]:.1f} s on the same runs. A is the bar: no DQN
        run beat it, on the demand it trained on or the reversed one ({_rv["Actuated"]:.1f} s).
        The DQN beats Webster's plans, and the tuned plan in two runs of three, and it does not
        need the schedule; before any purchase it must beat actuated control, in a field trial
        with the fallback in place. Actuated control relies on its detectors too; how it fails
        when they die was not tested here.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### When to use reinforcement learning

    | You have | Reach for |
    |:---------|:----------|
    | A model you can count (P) and few states | value iteration: no learning |
    | A signal with detectors | actuated control first: the bar to beat |
    | A simulator, not P; discrete actions | Q-learning, a DQN |
    | Continuous actions, or a stochastic policy | PPO |
    | Records of what experts did | behavioural cloning (their habits too) |
    | A controller in the field | a fault check and a fixed fallback |

    The first row is the reminder: if you can write $P$ down, plan; nothing needs learning. The
    second is today's result: a learned controller is worth its training only if it beats what
    the field already runs; here SUMO's untuned actuated control beat every trained DQN.
    Learning from a simulator inherits the simulator's assumptions (today, assumed demand on one
    intersection). And a learned controller fails in ways a fixed plan cannot: it trusts its
    sensors.

    **Impact (economic, social).** The discount rate is a value judgement with a budget
    attached: at 7% the cheapest policy leaves every poor bridge in service forever, so the
    users of poor bridges carry the cost a high rate makes cheap. At the intersection, the dead-detector test shows who pays for a sensor fault: the
    side street's drivers, not the main road's. Both choices belong in the open, next to the
    numbers they produce.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    ### Exit question

    A vendor's RL signal controller "cut delay 30% in simulation". What would you ask before the
    city buys it?

    **In the notebook:** value iteration, Q-learning, the intersection in SUMO, a DQN, and PPO,
    each checked against the slides' figure script.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        Any two of: 30% against which baseline (a stale plan, a tuned one, or actuated
        control?); on demand it was trained on, or held out, or with the peak moved?; how many
        training runs, and their spread?; what happens when a detector fails?; was the
        simulator calibrated to the corridor's counts?; what is the fallback, and who decides
        when it takes over?
        """)
        }
    )
    return


@app.cell
def _(
    ALL_RED_S,
    DECISION_S,
    DECK,
    DEMAND,
    DQN_BATCH,
    DQN_BUFFER,
    DQN_TARGET_EVERY,
    EPS_DECISIONS,
    EPS_END,
    EVAL_SEEDS,
    FAULT_S,
    GAMMAS,
    GAMMA_RL,
    GAP_MONTHS,
    HIDDEN,
    LOST_S,
    OBS_SCALE,
    P,
    PENALTY,
    PPO_CLIP,
    Q_EPSILON,
    Q_POWER,
    Q_STEPS,
    RATES,
    REPLACE_COST,
    TRAIN_SEEDS,
    VIEW_M,
    YELLOW_S,
    bridge_num,
    crit,
    mdp,
    mo,
    ql,
    sat,
    signal_for_claims,
    signal_num,
    slide,
    vi,
    webster_hour,
):
    _b = bridge_num
    _words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 8: "eight",
              10: "ten"}  # fmt: skip
    _dshare = [d[2] / (d[2] + d[3]) for d in DEMAND]  # north-south's share of the demand
    _g2, _g7 = GAMMAS["2%"], GAMMAS["7%"]
    _now = REPLACE_COST + PENALTY  # replacing a poor bridge: the replacement and the poor step
    _V2, _Q2, _V7 = vi["2%"]["V"], vi["2%"]["Q"], vi["7%"]["V"]
    _P0, _pn = P[0], mdp["p_new"]
    _w20 = {k: (1 + r) ** -20 for k, r in RATES.items()}
    _e = {r["steps"]: r["max_error"] for r in ql["log"]}
    _ag, _mk, _yr = _b["agency"], _b["markov"], _b["years_6_to_poor"]
    _ql = _b["q_learning"]
    _max_work = max(v["work_pct"] for v in _ag.values())
    _ratio6 = _mk[6][2]["pct"] / _mk[6][0]["pct"]
    _band_at = [
        "4 and below" if r == [3, 4] else str(r) for r in _b["age_mdp"]["replace_at_by_band"]
    ]
    _ts = slide("signal/train_seconds")  # the slides' timings: a timing is not reproducible

    def _render(_s: dict) -> list[str]:
        """The deck's statements, from the bridge numbers and the signal numbers ``_s``."""
        _dm, _ds, _st = _s["delay_mean_s"], _s["delay_s"], _s["stuck"]
        _l20, _gs = _s["last20_vehicle_hours"], _s["green_share_ns_pct"]
        _snap, _h = _s["snapshot"], _s["plans"]["hour"][0]
        _pd, _rv = _s["paired"], _s["reversed"]["delay_mean_s"]
        _tune, _act = _s["tuning"], _s["actuated"]
        _faults = _st["dead, fallback"]["fault_at_s"] or [0]
        _served = _st["dead"]["side_served_by_run"]
        _never = sum(x == 0 for x in _served)
        _trail = _pd["DQN"]["Actuated"]["mean_diff_s_by_run"]
        _ppo15 = _pd["PPO"]["Webster, 15 min"]
        _tried = [float(f) for f in _tune["delay_s"]]
        return [
        f"Illinois's {_b['ratings']['bridges']:,} highway bridges in 2024 (culverts left out)",
        f"{_b['ratings']['poor']:,} are poor (4 or less), highlighted",
        f"| γ, the weight of a cost two years away | {_g2:.3f} | {_g7:.3f} |",
        f"| Replace now: 1, plus {PENALTY:.2f} for this poor step | {_now:.2f} | {_now:.2f} |",
        f"| Wait: {PENALTY:.2f} now, then {_now:.2f} in two years | ? | ? |",
        f"a two-year step that starts poor costs {PENALTY:.2f}, whatever is done about it, so "
        f"replacing a poor bridge now costs {_now:.2f}.",
        f"{PENALTY:.2f} + {_g2:.4f} × {_now:.2f} = {_b['attempt']['2%']['wait']:.3f} at 2%, more "
        f"than {_now:.2f}, so replace now; {PENALTY:.2f} + {_g7:.4f} × {_now:.2f} = "
        f"{_b['attempt']['7%']['wait']:.3f} at 7%, so wait",
        f"γ = {_g2:.3f} per two years: 1.00 now, "
        + ", ".join(f"{w:.2f}" for w in _b["gamma_powers"][1:]) + " over ten years",
        f"({100 * _P0[1, 1]:.0f}% at 4, {100 * _P0[3, 3]:.0f}% at 6, {100 * _P0[5, 5]:.0f}% at 8)",
        f"{100 * _pn[6]:.0f}% of new bridges are first rated 9, {100 * _pn[5]:.0f}% are rated 8",
        f"{_b['pairs']:,} pairs of a bridge's records in consecutive files, 2014 to 2024, whose "
        f"inspections were {GAP_MONTHS[0]} to {GAP_MONTHS[1]} months apart",
        f"(Item 106, {_b['pairs_work']:,} of them)",
        f"({_b['pairs_rose']:,} pairs, {_b['pairs_rose_pct']}% of the rest)",
        f"those pairs are left out: {_b['pairs_used']:,} remain",
        f"moves the critical cost of being poor (two slides on) from {crit['2%']:.3f} to "
        f"{_b['rises_as_stays']['critical_penalty']['2%']:.3f}",
        f"the ratings that {_b['new_bridges']:,} new bridges received at their first inspection",
        f"{100 * _P0[6, 6]:.0f}% of nines are still nines two years later",
        f"with {_b['pairs_used']:,} pairs, counting is enough",
        f"| App. C (2026): cost-effectiveness | {100 * RATES['2%']:.1f}% | {_g2:.3f} |",
        f"| 1992 base case: benefit-cost | {100 * RATES['7%']:.0f}% | {_g7:.3f} |",
        f"At 7%, a cost 20 years away weighs {_w20['7%']:.2f}; at 2%, {_w20['2%']:.2f}.",
        f"(rating 3 or worse and rating 4 at {_V2[0]:.2f} replacements, 6 at {_V2[3]:.2f}, 9 at "
        f"{_V2[6]:.2f})",
        f"below 1e-10 after {_b['vi']['2%']['sweeps']} sweeps",
        f"{_b['vi']['2%']['sweeps']} sweeps at γ = {_g2:.3f} reach changes under 1e-10",
        f"a new bridge (rating 9) will cost its owner {_V2[6]:.2f} of a replacement",
        f"a poor one, {_V2[0]:.2f}.",
        f"At 2% a year, replace at ratings 4 and below once that cost exceeds {crit['2%']:.3f}; "
        f"at 7%, only above {crit['7%']:.3f}.",
        f"at a cost of {PENALTY:.2f} (the marker), 2% replaces and 7% does not",
        f"At 2%, with {PENALTY:.2f} per two years poor: replace at 4 and 3. At 7%: never;",
        f"forever, at {PENALTY:.2f} every two years, worth {_V7[0]:.2f} today, less than a "
        "replacement",
        f"crosses the replace surface at {crit['2%']:.3f}",
        f"the two Q values are {_Q2[1, 0]:.3f} (do nothing) and {_Q2[1, 1]:.3f} (replace)",
        f"for t in range({Q_STEPS:_}):",
        f"rng.random() < {Q_EPSILON} else Q[s].argmin()",
        f"n[s, a] ** -{Q_POWER}",
        f"falling from {_e[1000]:.2f} after a thousand simulated inspections to "
        f"{_e[2_000_000]:.2f} after two million",
        f"the greedy policy last differs from value iteration's at step "
        f"{_ql['last_differs_at']:,} and matches it for good after that",
        f"flickers almost only at ratings 3 and 4 (on {_ql['differ_steps'][3]:,} and "
        f"{_ql['differ_steps'][4]:,} steps; a few dozen at 5 and 9), where the two Q values are "
        f"close ({_Q2[1, 0]:.3f} against {_Q2[1, 1]:.3f})",
        f"The largest error falls to {_e[1_000_000]:.4f} by a million and {_e[2_000_000]:.4f} by "
        f"two million; the rarest state-action pair was visited {_ql['fewest_visits']:,} times.",
        f"exploration tries a random action one time in {_words[round(1 / Q_EPSILON)]}",
        f"at the busiest moment (minute {_snap['minute']}) of a held-out hour under the trained DQN",
        f"the {VIEW_M} m nearest the junction drawn",
        f"{_snap['on_approaches']} vehicles on the approaches, {_snap['halted']} of them stopped",
        f"At the busiest moment of a held-out hour (minute {_snap['minute']}), "
        f"{_snap['on_approaches']} vehicles were on the approaches and {_snap['halted']} were "
        f"stopped; with the exit roads, {_snap['in_network']} were in the network.",
        f"a switch costs a {YELLOW_S} s yellow and a {ALL_RED_S} s all-red",
        f"{_words[10].capitalize()} numbers, {_words[8]} of them counts up to about "
        f"{OBS_SCALE:.0f}",
        "north-south " + ", ".join(str(d[2]) for d in _s["demand"])
        + " vehicles per hour per approach; east-west " + ", ".join(str(d[3]) for d in _s["demand"]),
        f"one plan for the hour ({_h[1]} s and {_h[2]} s)",
        f"({LOST_S} s: two changes)",
        f"a standing queue of {sat['queue']} cars discharged at {sat['headway_s']:.1f} s headways, "
        f"{sat['veh_per_h']:,.0f} vehicles per hour of green",
        f"(Y = {webster_hour['Y']:.3f}, a {webster_hour['cycle_s']:.1f} s cycle)",
        f"| Q-network | 10 inputs, 2 × {HIDDEN} ReLU, 2 outputs | one Q per green |",
        f"| γ per five-second decision | {GAMMA_RL} | about "
        f"{_s['settings']['horizon_minutes']} minutes ahead |",
        f"| Exploration | ε from 1 to {EPS_END} over {EPS_DECISIONS:,} decisions | try both greens |",
        f"| Replay buffer | {DQN_BUFFER:,} decisions, batches of {DQN_BATCH} | break the correlation |",
        f"| Target network | copied every {DQN_TARGET_EVERY} decisions | a target that holds still |",
        f"refreshed only every {DQN_TARGET_EVERY} decisions",
        f"Each run trains on {_s['settings']['episodes']} simulated hours (a decision every "
        f"{_words[DECISION_S]} seconds, {_words[DECISION_S + YELLOW_S + ALL_RED_S]} when the "
        f"green changes), {_words[_s['settings']['train_seeds']]} runs per method, one CPU thread "
        "each.",
        f"over the last 20 hours the DQN averages {_l20['DQN']:.2f}, the online network "
        f"{_l20['DQN, online']:.2f}, PPO {_l20['PPO']:.2f}",
        f"{_l20['DQN']:.2f} vehicle-hours of waiting per hour over the last 20 training hours; "
        f"online, each decision once: {_l20['DQN, online']:.2f}",
        f"a batch of one decision instead of {DQN_BATCH}, at the same Adam rate",
        f"(ε = {EPS_END} for the DQNs; PPO samples its policy)",
        f"min(r A, clip(r, {1 - PPO_CLIP:.1f}, {1 + PPO_CLIP:.1f}) A)",
        f"past r = {1 + PPO_CLIP:.1f}",
        f"below r = {1 - PPO_CLIP:.1f}",
        f"clip(r_t, 1 − {PPO_CLIP}, 1 + {PPO_CLIP})",
        f"past 1 ± {PPO_CLIP}",
        f"Webster for the hour {_dm['Webster, hour']:.1f} s; Webster per 15 minutes "
        f"{_dm['Webster, 15 min']:.1f} s; the same plans with every green × {_tune['factor']:.1f}, "
        f"tuned on training hours, {_dm['Fixed, tuned']:.1f} s; SUMO's actuated control "
        f"{_dm['Actuated']:.1f} s; DQN {_dm['DQN']:.1f} s; DQN online {_dm['DQN, online']:.1f} s; "
        f"PPO {_dm['PPO']:.1f} s",
        f"one factor on its 15-minute greens was chosen on {_words[_tune['hours']]} training "
        f"hours, ×{_tune['factor']:.1f} of {min(_tried):.1f} to {max(_tried):.1f}: "
        f"{_dm['Fixed, tuned']:.1f} s.",
        f"each green runs {_act['min_green_s'][0]:.0f} to {_act['max_green_s'][0]:.0f} s, "
        f"extended while its detectors see cars; no schedule, no training: "
        f"{_dm['Actuated']:.1f} s.",
        f"The DQN's {_words[len(_ds['DQN'])]} runs ({min(_ds['DQN']):.1f} to "
        f"{max(_ds['DQN']):.1f} s) beat Webster per 15 minutes on "
        f"{_pd['DQN']['Webster, 15 min']['better']} of {_pd['DQN']['Webster, 15 min']['of']} "
        f"run-hours and the tuned plan on {_pd['DQN']['Fixed, tuned']['better']} of "
        f"{_pd['DQN']['Fixed, tuned']['of']}, but beat actuated control on only "
        f"{_pd['DQN']['Actuated']['better']} of {_pd['DQN']['Actuated']['of']}: every run trails "
        f"it, by {min(_trail):.1f} to {max(_trail):.1f} s.",
        f"the hourly plan ({_dm['Webster, hour']:.1f} s) above; the spread of the "
        f"{_words[len(_ds['DQN'])]} DQN runs ({min(_ds['DQN']):.1f} to {max(_ds['DQN']):.1f} s) "
        "below.",
        f"PPO ({min(_ds['PPO']):.1f} to {max(_ds['PPO']):.1f} s) is slightly behind Webster per "
        f"15 minutes ({_dm['Webster, 15 min']:.1f} s), worse on "
        f"{_ppo15['of'] - _ppo15['better']} of {_ppo15['of']} run-hours, and left "
        f"{_s['unfinished']['PPO']} vehicles waiting at the end of the "
        f"{len(TRAIN_SEEDS) * len(EVAL_SEEDS)} runs' clearing time",
        f"The online network: {_dm['DQN, online']:.1f} s, one run at {max(_ds['DQN, online']):.1f} s.",
        f"the hourly plan stays near {sum(_gs['Webster, hour']) / len(_gs['Webster, hour']):.0f}%; "
        f"the plan per 15 minutes steps from {_gs['Webster, 15 min'][0]:.0f}% to "
        f"{_gs['Webster, 15 min'][-1]:.0f}%; the DQN goes from {_gs['DQN'][0]:.0f}% to about "
        f"{sum(_gs['DQN'][-3:]) / 3:.0f}%, actuated control from {_gs['Actuated'][0]:.0f}% to "
        f"about {sum(_gs['Actuated'][-3:]) / 3:.0f}%, in the direction of the demand's own share "
        f"(dotted), which falls from {100 * _dshare[0]:.0f}% to {100 * _dshare[-1]:.0f}%.",
        f"| Here: delay; training | {_dm['DQN']:.1f} s; about {round(_ts['DQN'] / 60)} min a run | "
        f"{_dm['PPO']:.1f} s; about {round(_ts['PPO'] / 60)} min a run |",
        f"{_ag[3]['gone_pct']}% at 3 or worse, {_ag[4]['gone_pct']}% at 4, {_ag[5]['gone_pct']}% "
        "at 5, near zero above",
        f"the share with recorded work within two years, under {-(-_max_work // 1):.0f}% "
        "everywhere",
        f"while {_ag[4]['gone_pct']:.0f}% of Illinois's rating-4 bridges leave the rated "
        "inventory by the next file",
        "The MDP agrees on where (at 4 and below)",
        f"at rating 6, from {_mk[6][0]['pct']}% to {_mk[6][2]['pct']}%",
        f"drops twice as often as one under 20 ({_mk[6][2]['pct']}% against {_mk[6][0]['pct']}% in "
        "two years)",
        f"(four 20-year bands), a rating-6 bridge under 20 reaches poor in {_yr['band0']} years, "
        f"not the {_yr['rating only']} the rating-only chain predicts",
        f"| At 2%, poor costing {PENALTY:.2f} | Replaces at |",
        "| Rating only | 4 and below |",
        f"| Rating and age, 0-20 years | {_band_at[0]} |",
        f"| Rating and age, over 60 years | {_band_at[3]} |",
        f"| Extra cost of ignoring age | {_b['age_mdp']['regret_pct']:.2f}% |",
        f"{_words[_never]} of the {_words[len(_served)]} trained DQNs never turned the side street "
        f"green: none of its {_st['dead']['side_vehicles_by_run'][0]:,} vehicles over the "
        f"{_words[len(EVAL_SEEDS)]} hours got through. The third served all of them, with a mean "
        f"delay of {max((d for d, x in zip(_st['dead']['delay_s_by_run'], _served, strict=True) if x), default=float('nan')):.1f} "
        "s per vehicle.",
        f"delay rises from {_st['working']['delay_s']:.1f} s to at least "
        f"{_st['dead']['delay_s']:.1f} s per vehicle, and {_st['dead']['unfinished_per_hour']} "
        "vehicles an hour never get through",
        f"a detector that reads no vehicle for {FAULT_S // DECISION_S} decisions in a row "
        f"({_words[FAULT_S // 60]} minutes or a little more; the first reading is at t = 0) is "
        f"declared faulty (here between {min(_faults)} and {max(_faults)} s)",
        f"control falls back to a fixed plan: {_st['dead, fallback']['delay_s']:.1f} s, worse than "
        f"the DQN's {_st['working']['delay_s']:.1f} s, far better than {_st['dead']['delay_s']:.1f} s.",
        "| Agency cost only, nothing for poor bridges | never replace |",
        f"| The same at 7% a year, poor at {PENALTY:.2f} | never replace |",
        f"A: {_dm['Actuated']:.1f} s per vehicle, with no schedule and no training. B: "
        f"{_dm['Fixed, tuned']:.1f} s tuned ({_dm['Webster, 15 min']:.1f} s as Webster's formula "
        "gives it)",
        f"the 15-minute plans reach {_rv['Fixed, tuned']:.1f} s tuned and "
        f"{_rv['Webster, 15 min']:.1f} s untuned, until an engineer retimes them "
        f"({_rv['Webster, 15 min, retimed']:.1f} s)",
        f"C: {_dm['DQN']:.1f} s, and {_rv['DQN']:.1f} s on the reversed demand too, but behind A "
        f"in every run, and at least {_st['dead']['delay_s']:.1f} s when the side street's "
        "detectors die",
        f"our test fell back to the hourly plan ({_st['dead, fallback']['delay_s']:.1f} s)",
        f"no DQN run beat it, on the demand it trained on or the reversed one "
        f"({_rv['Actuated']:.1f} s)",
        "Every five seconds the controller picks a green",
        f"10 points now and 10 three steps later, at γ = 1/2, are worth 10 + 10/8 = "
        f"{10 + 10 * 0.5**3:g}.",
    ]  # fmt: skip

    # The words the slides attach to these numbers, each a fact of the notebook's values
    _s = signal_for_claims
    _dm, _ds, _st, _pd = _s["delay_mean_s"], _s["delay_s"], _s["stuck"], _s["paired"]
    _rvd = _s["reversed"]["delay_s"]
    assert _b["vi"]["2%"]["replace_at"] == [3, 4] and _b["vi"]["7%"]["replace_at"] == []
    assert not _b["no_penalty_replaces"] and round(_ratio6) == 2  # "never", "twice as often"
    assert all(v[2]["pct"] > v[0]["pct"] for v in _mk.values())  # "every line rises" to 40-60
    assert round(_V2[0], 2) == round(_V2[1], 2)  # "rating 3 or worse and rating 4 at 1.89"
    assert all(v["gone_pct"] < 0.5 for k, v in _ag.items() if k >= 6)  # "near zero above"
    # Q-learning flickers "almost only at ratings 3 and 4", "a few dozen at 5 and 9"
    _dif = _ql["differ_steps"]
    assert _dif[6] == _dif[7] == _dif[8] == 0 and 12 <= _dif[5] + _dif[9] < 100
    assert _dif[3] + _dif[4] > 1000 * (_dif[5] + _dif[9])
    # The comparison: actuated control beats every DQN run, on both demands; the DQN beats
    # Webster's plans in every run, and the tuned plan in two of three (the third ties it)
    assert all(d > 0 for d in _pd["DQN"]["Actuated"]["mean_diff_s_by_run"])  # "every run trails"
    assert min(_rvd["DQN"]) > _rvd["Actuated"][0]  # "no DQN run beat it ... the reversed one"
    assert all(d < 0 for d in _pd["DQN"]["Webster, 15 min"]["mean_diff_s_by_run"])
    assert max(_ds["DQN"]) < _dm["Webster, hour"]
    _vs_tuned = sorted(_pd["DQN"]["Fixed, tuned"]["mean_diff_s_by_run"])
    assert _vs_tuned[0] < _vs_tuned[1] < 0 < _vs_tuned[2] < 0.2  # two better, one tie
    assert 0 < _pd["PPO"]["Webster, 15 min"]["mean_diff_s"] < 1.5  # "slightly behind"
    assert len(_ds["DQN"]) == len(TRAIN_SEEDS) and len(_ds["DQN, online"]) == len(TRAIN_SEEDS)
    # The tuned factor is interior to the factors tried
    _tried = sorted(float(f) for f in _s["tuning"]["delay_s"])
    assert _tried[0] < _s["tuning"]["factor"] < _tried[-1]
    # Dead detectors: two runs never serve the side street, the third serves every vehicle; the
    # drawn run (the first) is one that never serves it
    _served, _tot = _st["dead"]["side_served_by_run"], _st["dead"]["side_vehicles_by_run"]
    assert sorted(_served) == [0, 0, _tot[0]] and len(set(_tot)) == 1 and _served[0] == 0
    assert _dm["DQN"] < _st["dead, fallback"]["delay_s"] < _st["dead"]["delay_s"]
    assert _st["dead, fallback"]["fault_at_s_drawn"] == min(_st["dead, fallback"]["fault_at_s"])
    _own = _render(signal_num)  # this run's own signal numbers render too (any setting)
    claims = _render(signal_for_claims)
    assert len(_own) == len(claims)
    assert DECISION_S == 5 and FAULT_S == 300
    _missing = [c for c in claims if c not in DECK]
    assert not _missing, _missing
    mo.md(
        f"**Every number on the slides:** {len(claims)} statements, each rendered from this "
        "notebook's values (the bridge numbers and the untrained controllers' always rebuilt "
        "here; the learned controllers' rebuilt in the slides setting, the slides' own in the "
        "others) and found word for word in the deck's text."
    )
    return (claims,)


@app.cell
def _(
    APPROACH_M,
    DECISION_S,
    DECK,
    EVAL_SEEDS,
    FAULT_S,
    HOUR_S,
    SETTING,
    demand_share,
    env_check,
    far_drop,
    fixed_runs,
    flicker,
    hours_halted,
    learned_runs,
    mo,
    np,
    q_num,
    share,
    stuck_extra,
    stuck_num,
    train_curves,
    trained,
):
    # Statements in the deck that numbers.json does not carry, each measured here (the deck as
    # revised after the first audit). In the reduced and smoke settings the learned
    # controllers' checks use this setting's runs.
    findings = []

    def finding(statement: str, measured: str, verdict: str) -> None:
        assert statement in DECK, statement
        findings.append((statement, measured, verdict))

    # 1. A decision every five seconds, ten when the green changes (the environment check above
    # asserts seconds = 5 x decisions + 5 x changes); how many decisions a training hour holds
    _dec = np.array([d for t in trained for d in t["decisions"]])
    finding(
        "(a decision every five seconds, ten when the green changes)",
        f"on the fixed action sequence, {env_check['seconds']:,} s = {DECISION_S} × "
        f"{env_check['decisions']} decisions + {DECISION_S} × {env_check['switches']} changes; "
        f"training hours hold {_dec.mean():.0f} decisions on average ({_dec.min()} to "
        f"{_dec.max()}, {len(trained)} runs), not 3,600 / {DECISION_S}",
        "supported",
    )
    # 2. The rest of a row falls one or two ratings, mostly
    finding("the rest mostly one or two ratings lower",
            f"three or more ratings down in two years: at most {100 * far_drop[1:6].max():.1f}% "
            f"of a row at ratings 4 to 8, {100 * far_drop[6]:.1f}% at 9",
            "supported" if far_drop[1:6].max() < 0.02 else "check")  # fmt: skip
    # 4. Q-learning's flicker
    _rates = sorted({r for _, d in flicker for r in d})
    finding("flickers almost only at ratings 3 and 4",
            f"after the first agreement the greedy policy differed at {len(flicker)} logged "
            f"checkpoints, at ratings {_rates}; over all steps, per rating: "
            f"{q_num['differ_steps']}", "supported (rating 3 has the same two Q values as 4)")  # fmt: skip
    # 5. All three methods fall below the hourly plan during training
    _below = {}
    for _lab, _runs in train_curves.items():
        _c = np.array(_runs)
        _k = min(10, _c.shape[1])
        _sm = np.array([np.convolve(r, np.ones(_k) / _k, mode="valid") for r in _c]).mean(0)
        _below[_lab] = bool(_sm.min() < hours_halted["Webster, hour"])
    finding("All three fall below the hourly plan's level during training",
            f"ten-hour average below the hourly plan's {hours_halted['Webster, hour'] / 3600:.2f} "
            f"vehicle-hours at some point: {_below}",
            "supported" if all(_below.values()) else "not in this run")  # fmt: skip
    # 6. Replay against online, run by run
    _l20 = {lab: sorted(round(float(np.mean(c[-20:])) / 3600, 2) for c in runs)
            for lab, runs in train_curves.items()}  # fmt: skip
    finding("Replay and a target net learn more",
            f"last 20 training hours, vehicle-hours per hour, by run: {_l20}",
            "supported" if max(_l20["DQN"]) < min(_l20["DQN, online"]) else "overlapping runs")  # fmt: skip
    finding("which of them matters most was not tested",
            "replay, the target network, and the batch were not ablated one at a time",
            "stated as untested")  # fmt: skip
    # 7. Hour by hour in this setting: the DQN against actuated control
    _base = {r["seed"]: r["delay_s"] for r in fixed_runs
             if r["name"] == "Actuated" and r["routes"] == "demand"}  # fmt: skip
    _dqn = [(r["delay_s"], _base[r["seed"]]) for r in learned_runs
            if r["kind"] == "dqn" and r["routes"] == "demand"]  # fmt: skip
    _wins = sum(a < b for a, b in _dqn)
    finding("but beat actuated control on only",
            f"this setting: the DQN's delay was lower than actuated control's on {_wins} of "
            f"{len(_dqn)} (run, hour) pairs", "supported" if _wins <= len(_dqn) / 4 else "check")  # fmt: skip
    # 8. The DQN's split follows the demand's, within 6 points by quarter hour; so does actuated
    # control's; the 15-minute plans follow it more closely
    _q = {n: 100 * share[n].reshape(4, 3).mean(1) for n in ("DQN", "Actuated", "Webster, 15 min")}
    _dem = 100 * demand_share
    _gap = {n: float(np.abs(v - _dem).max()) for n, v in _q.items()}
    _with = {n: bool((np.diff(v) <= 0.5).all()) for n, v in _q.items()}  # never against it
    finding("its split follows the demand's, within 6 points in every quarter hour",
            "north-south share of green by quarter hour: DQN "
            + ", ".join(f"{x:.0f}%" for x in _q["DQN"]) + "; demand "
            + ", ".join(f"{x:.0f}%" for x in _dem) + f"; the DQN within {_gap['DQN']:.1f} points, "
            f"the 15-minute plans within {_gap['Webster, 15 min']:.1f}",
            "supported" if _with["DQN"] and _gap["DQN"] < 6 and _gap["Webster, 15 min"] < _gap["DQN"]
            else "not in this run")  # fmt: skip
    finding("Actuated control, which reads only its detectors, follows it too.",
            "actuated control by quarter hour: " + ", ".join(f"{x:.0f}%" for x in _q["Actuated"])
            + f"; within {_gap['Actuated']:.1f} points of the demand",
            "supported" if _with["Actuated"] and _gap["Actuated"] < 6 else "check")  # fmt: skip
    # 10. The side street with dead detectors
    _cap = 2 * APPROACH_M / 7.5  # two approaches of 300 m, 5 m cars with 2.5 m gaps
    _d = stuck_extra["dead"]
    finding("its queue grows until both approaches are full",
            f"most vehicles waiting on the side street in the hour: {_d['max_side_queue']} (two "
            f"approaches hold about {_cap:.0f}); east-west got {_d['east_west_green_pct']}% of "
            "the green seconds", "supported" if _d["max_side_queue"] >= 0.9 * _cap else "partly")  # fmt: skip
    _f = stuck_extra["dead, fallback"]
    finding("the queue clears",
            f"with the fallback, {_f['side_queue_at_60_min']} vehicles waiting at minute 60 "
            f"(working detectors: {stuck_extra['working']['side_queue_at_60_min']})", "supported"
            if _f["side_queue_at_60_min"] <= 2 * stuck_extra["working"]["max_side_queue"] else "check")  # fmt: skip
    # 11. The fault rule: 60 zero readings in a row, the first at t = 0
    _faults = stuck_num["dead, fallback"]["fault_at_s"]
    finding("(five minutes or a little more;",
            f"faults declared at {_faults} s: the 60th reading in a row comes at "
            f"{(FAULT_S // DECISION_S - 1) * DECISION_S} s when no green changes (the first is "
            "at t = 0), later when greens change",
            f"supported, but the earliest fault came {FAULT_S - min(_faults)} s short of five "
            "minutes")  # fmt: skip
    # 12. The fallback: the hourly plan on the slides; the plans per 15 minutes here too
    _b, _a = stuck_num["dead, fallback 15 min"], stuck_num["dead, fallback"]
    finding("our test fell back to the hourly plan",
            f"a fallback to the plans per 15 minutes: {_b['delay_s']:.1f} s against "
            f"{_a['delay_s']:.1f} s to the hourly plan ({_b['runs']} runs each, this notebook)",
            "the 15-minute plans are the better fallback" if _b["delay_s"] < _a["delay_s"]
            else "the hourly plan is no worse")  # fmt: skip
    # 13. What the slides say was not measured, or comes from a trial
    finding("whether that explains the gap here was not tested",
            "not tested here either", "stated as untested")  # fmt: skip
    finding("in a trial run with one shared body, PPO learned to hold one green",
            "a prototype run (about 258,000 halted vehicle-seconds per "
            "training hour, against about 20,000 with separate networks), not "
            "rerun by the script or here",
            "a prototype's anecdote, said to come from a trial; its reason is labelled not measured")  # fmt: skip
    for _f in findings:
        print("finding:", " | ".join(_f))
    _rows = "\n".join(f"| {s} | {m} | {v} |" for s, m, v in findings)
    _n_learned = len(learned_runs)
    mo.md(f"""
    **Statements the slides make without a numbers.json key of their own,** each measured here
    (setting `{SETTING}`: {len(trained)} training runs, {_n_learned} learned-controller runs
    on both demands, the untrained controllers on {len(EVAL_SEEDS)} hours; {HOUR_S // 60}-minute
    hours):

    | The deck says | Measured | Verdict |
    |:--------------|:---------|:--------|
    {_rows}
    """)
    return (findings,)


@app.cell
def _(
    KINDS,
    SETTING,
    T_START,
    bridge_seconds,
    claims,
    findings,
    mo,
    t_env,
    t_eval,
    t_net,
    t_snapshot,
    t_train,
    t_tune,
    t_weights,
    time,
    train_seconds,
    weight_check,
):
    _minutes = (time.perf_counter() - T_START) / 60
    _ts = ", ".join(f"{lab} {train_seconds[lab] / 60:.1f} min" for lab in KINDS.values())
    print(f"setting {SETTING!r}: done in {_minutes:.1f} min (bridge {bridge_seconds:.0f} s, "
          f"network and checks {t_net + t_env:.0f} s, weights {t_weights:.0f} s, training "
          f"{t_train / 60:.1f} min, tuning {t_tune:.0f} s, evaluation {t_eval:.0f} s, snapshot "
          f"{t_snapshot:.0f} s); "
          f"{len(claims)} claims found; {len(findings)} statements measured; training "
          f"seconds per run {train_seconds}")  # fmt: skip
    mo.md(f"""
    ## Gotchas

    - **SUMO splits option values at commas.** A trip file named after the label "DQN, online"
      silently became two file names. Every file name and option value here is
      built from a slug of letters, digits, and underscores, and the environment asserts that
      no path holds a comma.
    - **libsumo runs one simulation per process.** Training and evaluation run in worker
      processes (joblib's loky, which starts fresh interpreters rather than forking after
      PyTorch has started threads); each worker closes its simulation before the next job.
    - **Deterministic means one thread, deterministic kernels, and the same order of float32
      operations.** That is what makes a run here equal to the script's weight for weight
      ({", ".join(f"{k}: {v['weights']:,} weights" for k, v in weight_check.items())}). Even
      the order of a product matters: GAE's `GAMMA_RL * PPO_LAMBDA * running` multiplies the two
      Python floats first, then rounds once to float32.
    - **A random sequence must be drawn the same way, not just seeded the same way.** The
      training hours come from `rng.choice(..., replace=False)`, which for a different count
      draws different hours, not a prefix; Q-learning draws all its random numbers up front,
      so a shorter run is not the start of a longer one.
    - **Teleporting is off** (`--time-to-teleport -1`): by default SUMO removes a car stuck for
      300 s, which would make a starved side street look served.
    - **Count every vehicle,** including those still inside or never let in when the run stops
      (`write-unfinished`, `write-undeparted`); otherwise the dead-detector delay looks small.
      Each trip file is deleted once it is read (the script's runs once left 119 MB of them).
    - **A time limit is not a terminal state.** The hour stops at 3,600 s with queues still
      there, and the observation has no clock; storing that step as terminal tells the
      networks that a state like all the others is worth nothing, once a simulated hour. Every
      target bootstraps, the hour's last one from the state it ends in (Pardo et al., 2018).
    - **A comparison is only as good as its baseline.** Webster's formula is a starting point,
      not the best fixed plan: tuned on training hours like the networks, it does better. And
      the baseline the field runs is actuated control, which needs neither a schedule nor
      training.
    - **A checkpoint grid can miss a late flip.** Whether Q-learning's policy has settled is
      checked after every step, not only at the logged checkpoints.
    - **Training curves include exploration.** They are not held-out scores; the held-out hours
      are demand draws that no training run sees (asserted).
    - **A discount factor has a time unit.** γ = (1 + r)^−2 per two-year step; a per-year γ
      would halve the horizon.
    - **A rise in rating without recorded work is work the inventory missed, or two inspectors
      disagreeing,** not improvement: such pairs are left out of the do-nothing matrix (and the
      matching disagreements downward stay in; §3 shows it barely moves the policy).
    - **Two files two years apart are not always two years of change.** A bridge on a 48-month
      cycle carries one inspection into two files; the pairs are kept only when their
      inspections are 18 to 30 months apart.
    - **`torch.no_grad()`** for every action chosen and every target computed.
    - **Time.** This run (`{SETTING}`) took {_minutes:.1f} minutes: the bridge part
      {bridge_seconds:.0f} s; the SUMO network and the environment checks {t_net + t_env:.0f}
      s; the weight-for-weight checks {t_weights:.0f} s; training {t_train / 60:.1f} min of
      wall time ({_ts} per run); the fixed plan's tuning {t_tune:.0f} s; evaluation
      {t_eval:.0f} s; the snapshot {t_snapshot:.0f} s.

    ## Going deeper

    Forsyth's *Applied Machine Learning* stops before reinforcement learning; its §13.1 on
    Markov chains and §13.2.3 on dynamic programming are the background. The primary sources,
    paraphrased:

    - **Sutton and Barto (2018), chapter 3, finite MDPs.** The agent–environment loop, returns
      and discounting, value functions and the Bellman equations for a policy and for the
      optimum. The Markov property belongs to the state *signal* the designer chooses, which is
      why adding age to the bridge's state (§5) is a design decision, not a new algorithm. They
      write rewards and maxima; costs and minima are the same with the sign flipped.
    - **Chapter 4, dynamic programming.** Iterative policy evaluation, the policy-improvement
      theorem, policy iteration, and value iteration, which truncates evaluation to one sweep;
      both converge because the Bellman operators are γ-contractions. Their "generalized policy
      iteration" frames every method in this lecture as evaluation and improvement interleaved.
    - **Chapter 6, temporal-difference learning.** TD(0), SARSA (on-policy), and Q-learning
      (off-policy: it learns the greedy policy's values while behaving ε-greedily). Their
      cliff-walking example shows the cost: Q-learning learns the optimal path but, exploring,
      falls off more often than SARSA. They also show maximization bias, the reason for double
      Q-learning, a fix this lecture's DQN does not use.
    - **Chapter 13, policy-gradient methods.** The policy-gradient theorem, REINFORCE, a
      baseline to cut the variance, and actor–critic methods: PPO's critic and advantage come
      from here.
    - **Mnih et al. (2015), *Nature*.** One network and one set of hyperparameters learned 49
      Atari games from pixels; experience replay and a target network refreshed every so often
      were both needed (their ablations removed each and scores fell sharply), with rewards
      clipped and four frames stacked as the state.
    - **Schulman et al. (2017), PPO.** The clipped surrogate lets several epochs of minibatch
      steps reuse each batch without a trust-region constraint (TRPO's); compared on continuous
      control and on Atari against A2C, ACER, and TRPO. The advantage estimator, GAE, is
      Schulman et al. (2016).

    **Exercises.** (1) Policy iteration: start from "never replace", evaluate with
    `evaluate_policy`, improve greedily, and repeat; count the iterations it needs against value
    iteration's sweeps (§3), and check that it ends at the same policy. (2) Make the fault
    check's timer count simulated seconds, not decisions (a change of green makes a decision 10
    s long), rerun the dead-detector test, and report when the faults are now declared and what
    the delay becomes.

    ## Things to remember

    - RL: decisions whose costs come later.
    - Bellman: cost now, plus discounted cost next.
    - Known P: value iteration. Experience: Q-learning.
    - DQN: replay and a target net; PPO: clipped steps.
    - Beat the field's baseline; test reward and state.

    **Next:** fairness and accountability.
    """)
    return


if __name__ == "__main__":
    app.run()
