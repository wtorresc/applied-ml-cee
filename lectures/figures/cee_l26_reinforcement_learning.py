"""Figures and numbers for CEE Lecture 26: Markov decision processes and reinforcement learning.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l26_reinforcement_learning.py \
        --out build/fig/cee_l26_reinforcement_learning

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

Two decisions that unfold over time, in CS 441's order (problem statement, Bellman equation,
Q-learning, experience replay and a Q-network, PPO, four approaches, rewards that go wrong):

- **When should Illinois replace a bridge?** A Markov decision process on the National Bridge
  Inventory (Illinois, 2014-2024, every two years; Lectures 1, 3, 6). The state is a bridge's
  lowest rating (Items 58-60, 3 or worse merged), one step is one two-year inspection cycle, and
  the actions are to do nothing or to replace. Do-nothing transitions are counted from pairs of
  a bridge's records in consecutive files whose inspections are 18 to 30 months apart, with no
  recorded work (Item 106). A rating that rose without recorded work is unrecorded work or a
  disagreement between inspectors; those pairs are left out, and counting them as no change
  instead is reported beside. A replacement leads to the ratings that new bridges receive at
  their first inspections. Costs are in units of one replacement, plus a cost for every two
  years spent poor (rating 4 or less, FHWA), which is a judgement and is therefore swept. The
  discount rates are OMB's: Circular A-94 Appendix C (March 2026) gives 2.0% real for a 30-year
  cost-effectiveness analysis; the reinstated 1992 Circular gives 7% for the base case of
  benefit-cost analysis. Value iteration solves the MDP; Q-learning learns the same policy from
  simulated inspections, without the transition matrix. Break it: is the rating Markov? The
  chance of a drop depends on the bridge's age.
- **When should a signal switch?** One intersection in SUMO (Lopez et al., 2018), four
  single-lane approaches with demand that swings from north-south to east-west over the hour
  (assumed volumes). A deep Q-network (Mnih et al., 2015) with experience replay and a target
  network, the same network trained one transition at a time, and PPO (Schulman et al., 2017).
  The hour's end is a time limit, not a terminal state, so every target bootstraps from the
  state the hour ends in (Pardo et al., 2018). Baselines on the same held-out hours: Webster's
  (1958) fixed-time plan for the hour and for each 15 minutes, Webster's 15-minute greens
  scaled by a factor chosen on training hours, and SUMO's gap-based actuated controller with its
  default settings. Every controller is also run with the demand profile reversed in time.
  Break it: the side street's detectors die and read zero.
"""

from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from torch import nn

import aml_style
import title_art
from bridges import NBI_CSV, POOR
from common import parse_out_dir, pct, require, rnd, seed_all, write_numbers

SEED = 26

# ------------------------------------------------------------------ the bridge MDP: settings
RATINGS = np.arange(3, 10)  # the lowest of Items 58-60; 3 stands for "3 or worse"
N_S = len(RATINGS)
YEARS_PER_STEP = 2  # one step is one routine inspection cycle (NBI files every two years)
# A pair of records is one two-year step when its two inspections are 18 to 30 months apart:
# a bridge on a 48-month cycle otherwise contributes four years of change as one step
GAP_MONTHS = (18, 30)
# Real discount rates, per year. OMB Circular A-94, Appendix C (revised March 6, 2026): 2.0% for
# a 30-year cost-effectiveness analysis. The 1992 Circular, reinstated by M-25-23 (2025): 7% for
# the base case of a benefit-cost analysis of public investment.
RATES = {"2%": 0.02, "7%": 0.07}
REPLACE_COST = 1.0  # every cost is in units of one replacement
PENALTY = 0.10  # the cost of two years in poor condition (a judgement, hence swept below)
PENALTY_GRID = np.round(np.arange(0.0, 0.3001, 0.005), 3)
VI_TOL = 1e-10  # value iteration stops when no state value moves by more than this
NEW_AGE = 2  # a bridge inspected within two years of being built is new
AGE_BANDS = (0, 20, 40, 60, 200)  # years since built, for the Markov test (age 0 in the first)
Q_STEPS = 2_000_000  # simulated inspections for tabular Q-learning
Q_EPSILON = 0.2  # exploration: a random action this often
Q_POWER = 0.7  # step size 1 / n(s, a)^0.7 (Robbins-Monro: sum infinite, sum of squares finite)
Q_CHECKPOINTS = (1_000, 10_000, 100_000, 1_000_000, 2_000_000)
Z95 = 1.959963984540054  # the standard normal's 97.5% quantile, for 95% Wilson intervals


# --------------------------------------------------------------------------- bridge data


def load_records() -> pd.DataFrame:
    """The Illinois bridge-year records with a lowest rating, sorted by bridge and year.

    Returns:
        One row per bridge and NBI file, with ``s`` the lowest rating clipped below at 3.

    Raises:
        FileNotFoundError: If the derived CSV is missing (run ``bash scripts/fetch_nbi.sh``).
    """
    df = pd.read_csv(require(NBI_CSV, "fetch_nbi.sh"), dtype={"structure": str})
    df = df.dropna(subset=["rating"]).sort_values(["structure", "nbi_year"], kind="stable")
    df["s"] = df["rating"].clip(lower=RATINGS[0]).astype(int)
    assert df["s"].between(RATINGS[0], RATINGS[-1]).all(), "a rating outside 0-9"
    assert df[["inspection_year", "inspection_month", "year_built"]].notna().all().all()
    return df.reset_index(drop=True)


def rating_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs of a bridge's records in consecutive files whose inspections are one cycle apart.

    The files are two years apart, but a bridge on a 48-month cycle appears in two files with
    the same inspection, and then across a new inspection four years after the last. A pair
    counts as one two-year step only when its two inspections are 18 to 30 months apart
    (``GAP_MONTHS``); this also leaves out pairs of the same inspection.

    Args:
        df: The records from ``load_records``.

    Returns:
        One row per pair: the first record's columns, plus ``s2`` (the next lowest rating),
        ``months`` (between the two inspections), ``work`` (Item 106, year reconstructed,
        changed between the two), and ``age`` (years since built at the first inspection).
    """
    nxt = df.groupby("structure", sort=False).shift(-1)
    months = (nxt["inspection_year"] - df["inspection_year"]) * 12 + (
        nxt["inspection_month"] - df["inspection_month"]
    )
    pair = df.assign(
        s2=nxt["s"], next_year=nxt["nbi_year"], next_rec=nxt["year_reconstructed"], months=months
    )
    keep = (pair["next_year"] == pair["nbi_year"] + YEARS_PER_STEP) & pair["months"].between(
        *GAP_MONTHS
    )
    pair = pair[keep].copy()
    assert pair["months"].between(*GAP_MONTHS).all(), "a pair outside 18-30 months"
    pair["s2"] = pair["s2"].astype(int)
    pair["months"] = pair["months"].astype(int)
    pair["work"] = pair["next_rec"].fillna(0) != pair["year_reconstructed"].fillna(0)
    pair["age"] = pair["inspection_year"] - pair["year_built"]
    return pair.reset_index(drop=True)


def counts(pairs: pd.DataFrame) -> np.ndarray:
    """Transition counts, rows the rating now and columns the rating two years later.

    Args:
        pairs: Rows with ``s`` and ``s2``.

    Returns:
        A ``(N_S, N_S)`` float array.
    """
    tab = pd.crosstab(pairs["s"], pairs["s2"]).reindex(index=RATINGS, columns=RATINGS)
    return np.array(tab.fillna(0), dtype=float)


def bridge_mdp(df: pd.DataFrame, pairs: pd.DataFrame, rises: str = "drop") -> dict:
    """The MDP's transition matrices, estimated from the inventory.

    Args:
        df: The records.
        pairs: The two-year pairs.
        rises: What to do with a rating that rose with no recorded work (unrecorded work, or two
            inspectors disagreeing): "drop" the pair (the slides' MDP), or count it as a "stay".

    Returns:
        ``P`` (actions x states x states), the do-nothing counts, the new-bridge ratings, and
        the counts behind each choice.

    Raises:
        ValueError: If ``rises`` is neither "drop" nor "stay".
    """
    no_work = pairs[~pairs["work"]]
    rose = no_work["s2"] > no_work["s"]
    if rises == "drop":
        kept = no_work[~rose]
    elif rises == "stay":
        kept = no_work.assign(s2=no_work["s2"].where(~rose, no_work["s"]))
    else:
        raise ValueError(f"rises must be 'drop' or 'stay', not {rises!r}")
    C0 = counts(kept)
    assert (C0.sum(1) > 0).all(), "a rating with no do-nothing pairs"
    P0 = C0 / C0.sum(1, keepdims=True)
    new = df[(df["inspection_year"] - df["year_built"]).between(0, NEW_AGE)]
    new = new.drop_duplicates("structure", keep="first")  # each new bridge once, when first seen
    p_new = new["s"].value_counts().reindex(RATINGS, fill_value=0).to_numpy(float)
    p_new /= p_new.sum()
    P = np.stack([P0, np.tile(p_new, (N_S, 1))])
    assert np.allclose(P.sum(2), 1.0), "a transition row does not sum to one"
    assert np.allclose(np.tril(P0, 0), P0), "do nothing must never raise a rating"
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


def costs(penalty: float) -> np.ndarray:
    """One step's cost of each action in each state, in replacements.

    Args:
        penalty: The cost of two years spent poor (rating ``POOR`` or less).

    Returns:
        A ``(N_S, 2)`` array: column 0 do nothing, column 1 replace.
    """
    C = np.zeros((N_S, 2))
    C[:, 1] = REPLACE_COST
    C[RATINGS <= POOR, :] += penalty  # the step starts poor, whatever is done about it
    return C


def gamma_of(rate: float) -> float:
    """The discount factor of one step: a real rate per year, compounded over the step.

    Args:
        rate: The real discount rate per year.

    Returns:
        ``(1 + rate) ** -YEARS_PER_STEP``.
    """
    return (1.0 + rate) ** -YEARS_PER_STEP


# --------------------------------------------------------------------------- dynamic programming


def value_iteration(P: np.ndarray, C: np.ndarray, gamma: float) -> dict:
    """Bellman backups until no state value moves by more than ``VI_TOL``.

    V_{k+1}(s) = min_a [ C(s, a) + gamma * sum_s' P(s' | s, a) V_k(s') ].

    Args:
        P: Transitions, actions x states x states.
        C: Costs, states x actions.
        gamma: Discount factor per step.

    Returns:
        ``V`` and ``Q`` at convergence, the greedy ``policy``, every sweep's values (``trace``),
        and the largest change of each sweep (``deltas``).
    """
    V = np.zeros(C.shape[0])
    trace, deltas = [V], []
    while True:
        Q = C + gamma * np.einsum("ast,t->sa", P, V)
        V_new = Q.min(1)
        deltas.append(float(np.abs(V_new - V).max()))
        V = V_new
        trace.append(V)
        if deltas[-1] < VI_TOL:
            break
    return {"V": V, "Q": Q, "policy": Q.argmin(1), "trace": np.array(trace), "deltas": deltas}


def evaluate_policy(P: np.ndarray, C: np.ndarray, gamma: float, policy: np.ndarray) -> np.ndarray:
    """A policy's expected discounted cost from each state: solve (I - gamma P_pi) V = c_pi.

    Args:
        P: Transitions, actions x states x states.
        C: Costs, states x actions.
        gamma: Discount factor per step.
        policy: One action per state.

    Returns:
        The value of each state under the policy.
    """
    idx = np.arange(C.shape[0])
    return np.linalg.solve(np.eye(len(idx)) - gamma * P[policy, idx], C[idx, policy])


def critical_penalty(P: np.ndarray, gamma: float) -> float:
    """The smallest poor-condition cost at which replacing is optimal somewhere (bisection).

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


def q_learning(
    P: np.ndarray, C: np.ndarray, gamma: float, start: np.ndarray, Q_star: np.ndarray
) -> dict:
    """Tabular Q-learning from simulated inspections; it never sees ``P``, only samples of it.

    Q(s, a) <- Q(s, a) + alpha [ c + gamma min_a' Q(s', a') - Q(s, a) ], with an
    epsilon-greedy behaviour policy and a step size that shrinks with each pair's visits. The
    greedy policy is compared with value iteration's after every step (an update changes one
    row of Q, so only that state's comparison can change).

    Args:
        P: Transitions (used only to draw the next state, as the world would).
        C: Costs.
        gamma: Discount factor per step.
        start: The distribution of the first state (Illinois's ratings in 2024).
        Q_star: Value iteration's answer, to measure the error at the checkpoints.

    Returns:
        The final ``Q``; the visits per pair; at each checkpoint the largest error, whether the
        greedy policy equals value iteration's, and the fewest visits; the last step at which
        the greedy policy differed from value iteration's (``last_differs_at``); and, per state,
        the steps at which its greedy action differed (``differ_steps``).
    """
    rng = np.random.default_rng(SEED)
    cum = np.cumsum(P, axis=2)
    cum[:, :, -1] = 1.0  # guard the last bin against rounding
    Q = np.zeros_like(C)
    visits = np.zeros_like(C)
    pi_star = Q_star.argmin(1)
    differ = Q.argmin(1) != pi_star
    differ_steps = np.zeros(N_S, dtype=np.int64)
    last_differs_at = 0
    log = []
    s = int(rng.choice(len(start), p=start))
    marks = set(Q_CHECKPOINTS) | set(np.unique(np.geomspace(100, Q_STEPS, 31).astype(int)).tolist())
    explore = rng.random(Q_STEPS) < Q_EPSILON
    random_a = rng.integers(0, 2, Q_STEPS)
    u = rng.random(Q_STEPS)
    for t in range(1, Q_STEPS + 1):
        a = int(random_a[t - 1]) if explore[t - 1] else int(Q[s].argmin())
        s2 = int(np.searchsorted(cum[a, s], u[t - 1], side="right"))
        visits[s, a] += 1
        alpha = visits[s, a] ** -Q_POWER
        Q[s, a] += alpha * (C[s, a] + gamma * Q[s2].min() - Q[s, a])
        differ[s] = Q[s].argmin() != pi_star[s]
        if differ.any():
            last_differs_at = t
            differ_steps += differ
        s = s2
        if t in marks:
            log.append(
                {
                    "steps": t,
                    "max_error": float(np.abs(Q - Q_star).max()),
                    "same_policy": bool(not differ.any()),
                    "fewest_visits": int(visits.min()),
                }
            )
    assert np.array_equal(differ, Q.argmin(1) != pi_star), "the running comparison drifted"
    return {
        "Q": Q,
        "visits": visits,
        "log": log,
        "last_differs_at": last_differs_at,
        "differ_steps": differ_steps,
    }


def first_passage_years(P0: np.ndarray, threshold: int = POOR) -> np.ndarray:
    """Expected years until the rating first reaches ``threshold`` or less, doing nothing.

    Args:
        P0: The do-nothing transitions.
        threshold: The rating counted as arrived.

    Returns:
        Years from each rating above the threshold (``nan`` at or below it).
    """
    up = threshold < RATINGS
    A = np.eye(up.sum()) - P0[np.ix_(up, up)]
    steps = np.linalg.solve(A, np.ones(up.sum()))
    out = np.full(N_S, np.nan)
    out[up] = YEARS_PER_STEP * steps
    return out


def first_passage_aged(P0: np.ndarray, n_bands: int, threshold: int = POOR) -> np.ndarray:
    """Expected years to poor on the age-aware chain (bridges age as they go).

    Args:
        P0: The age-aware do-nothing transitions, states ordered band-major.
        n_bands: The number of age bands.
        threshold: The rating counted as arrived.

    Returns:
        Years from each (band, rating) state (``nan`` where already poor).
    """
    up = np.tile(threshold < RATINGS, n_bands)
    A = np.eye(up.sum()) - P0[np.ix_(up, up)]
    out = np.full(len(up), np.nan)
    out[up] = YEARS_PER_STEP * np.linalg.solve(A, np.ones(up.sum()))
    return out


def age_band(age: pd.Series) -> pd.Series:
    """The age band of each age in years: (0, 20] with 0 included, (20, 40], (40, 60], over 60.

    Args:
        age: Years since built, none negative.

    Returns:
        The band index of each age (an integer Series).
    """
    assert (age >= 0).all() and (age <= AGE_BANDS[-1]).all(), "an age outside the bands"
    band = pd.cut(age, AGE_BANDS, right=True, include_lowest=True, labels=False)
    assert band.notna().all()
    return band.astype(int)


def wilson(k: int, n: int) -> tuple[float, float]:
    """The 95% Wilson score interval of a binomial share.

    Args:
        k: Successes.
        n: Trials (at least one).

    Returns:
        The interval's lower and upper ends, inside [0, 1].
    """
    p = k / n
    den = 1.0 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / den
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / den
    return float(centre - half), float(centre + half)


def markov_test(pairs: pd.DataFrame) -> dict:
    """Test the Markov property: does the chance of a drop depend on age, given the rating?

    Pairs with a negative age (built after the inspection: data errors) are left out and
    counted.

    Args:
        pairs: The two-year pairs.

    Returns:
        For each rating 4-8 and age band, the pairs and the share that dropped, with a 95%
        Wilson interval (pairs treated as independent); the do-nothing matrix of each band; and
        the pairs left out for a negative age.
    """
    data = pairs[~pairs["work"] & (pairs["s2"] <= pairs["s"])]
    negative = int((data["age"] < 0).sum())
    data = data[data["age"] >= 0].copy()
    data["band"] = age_band(data["age"])
    mats = [counts(data[data["band"] == b]) for b in range(len(AGE_BANDS) - 1)]
    out = {}
    for r in range(4, 9):
        rows = []
        for b in range(len(AGE_BANDS) - 1):
            g = data[(data["s"] == r) & (data["band"] == b)]
            k = int((g["s2"] < r).sum())
            lo, hi = wilson(k, len(g))
            rows.append({"band": b, "n": len(g), "p": k / len(g), "lo": lo, "hi": hi})
        out[int(r)] = rows
    return {"by_rating": out, "counts": mats, "negative_ages": negative}


def age_mdp(
    mdp: dict, band_counts: list[np.ndarray], gamma: float, penalty: float, start: np.ndarray
) -> dict:
    """The MDP with age added to the state (rating x age band), solved and compared.

    Age moves into the next 20-year band with probability 2/20 per step (the next band's
    first two years), the oldest band keeps it; a replacement starts a new bridge in the
    youngest band. A band's row with no pairs borrows the pooled row.

    Args:
        mdp: The rating-only MDP from ``bridge_mdp``.
        band_counts: Do-nothing counts per age band.
        gamma: Discount factor per step.
        penalty: The poor-condition cost.
        start: The 2024 distribution over (rating, band), flattened band-major.

    Returns:
        The policy per band, the rating-only policy's expected cost in the age-aware world,
        the age-aware optimum's, and their ratio minus one (the regret).
    """
    nb = len(band_counts)
    P_band = []
    for raw in band_counts:
        Cb = raw.copy()
        empty = Cb.sum(1) == 0
        Cb[empty] = mdp["C0"][empty]
        P_band.append(Cb / Cb.sum(1, keepdims=True))
    width = AGE_BANDS[1] - AGE_BANDS[0]
    move = YEARS_PER_STEP / width
    K = N_S * nb
    P = np.zeros((2, K, K))
    for b in range(nb):
        nxt = [(b, 1 - move), (b + 1, move)] if b < nb - 1 else [(b, 1.0)]
        for s in range(N_S):
            for b2, w in nxt:
                P[0, b * N_S + s, b2 * N_S : (b2 + 1) * N_S] += w * P_band[b][s]
            P[1, b * N_S + s, :N_S] = mdp["p_new"]
    assert np.allclose(P.sum(2), 1.0)
    C = np.tile(costs(penalty), (nb, 1))
    best = value_iteration(P, C, gamma)
    rating_only = value_iteration(mdp["P"], costs(penalty), gamma)["policy"]
    V_ro = evaluate_policy(P, C, gamma, np.tile(rating_only, nb))
    cost_best, cost_ro = float(start @ best["V"]), float(start @ V_ro)
    return {
        "P0": P[0],
        "policy": best["policy"].reshape(nb, N_S),
        "rating_only": rating_only,
        "cost_best": cost_best,
        "cost_rating_only": cost_ro,
        "regret": cost_ro / cost_best - 1.0,
    }


def agency_behaviour(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """What Illinois did, by rating: recorded work, and records never followed by another.

    Args:
        df: The records.
        pairs: The two-year pairs.

    Returns:
        Per rating: the share of pairs with recorded work (Item 106), and the share of records
        before 2024 whose bridge has no rated record in any later file (replaced, removed,
        renumbered, or dropped from the rated bridges).
    """
    work = pairs.groupby("s")["work"].mean().reindex(RATINGS)
    nxt_year = df.groupby("structure", sort=False)["nbi_year"].shift(-1)
    before = df[df["nbi_year"] < df["nbi_year"].max()].assign(
        gone=nxt_year[df["nbi_year"] < df["nbi_year"].max()].isna()
    )
    gone = before.groupby("s")["gone"].mean().reindex(RATINGS)
    n = before.groupby("s").size().reindex(RATINGS)
    return {"work": work.to_numpy(float), "gone": gone.to_numpy(float), "n": n.to_numpy(int)}


# ------------------------------------------------------------------ the intersection: settings
SPEED_MS = 13.89  # 50 km/h
APPROACH_M = 300.0
APPROACHES = ("N_in", "S_in", "E_in", "W_in")
EXITS = {"N_in": "C_S", "S_in": "C_N", "E_in": "C_W", "W_in": "C_E"}
NS, EW = 0, 1  # the two green phases (actions)
GREEN_PHASE = {NS: 0, EW: 3}  # SUMO's phase indices: green, then yellow (+1), then all-red (+2)
YELLOW_S, ALL_RED_S, DECISION_S = 3, 2, 5
LOST_S = 2 * (YELLOW_S + ALL_RED_S)  # Webster's lost time per cycle: every intergreen
HOUR_S = 3600
CLEAR_S = 900  # after the hour no one arrives; the controller runs until the network is empty
END_S = HOUR_S + CLEAR_S  # SUMO's --end: no run steps past it
# Assumed demand, vehicles per hour on each approach of a direction, per 15 minutes: the morning
# flow runs north-south, then the afternoon's east-west (Poisson arrivals, one vehicle type).
DEMAND = ((0, 900, 600, 250), (900, 1800, 700, 300), (1800, 2700, 400, 500), (2700, 3600, 300, 600))
# The shifted-demand test: the same quarter hours in reverse order (east-west's peak first)
REVERSED = tuple(
    (b, e, ns, ew) for (b, e, _, _), (_, _, ns, ew) in zip(DEMAND, DEMAND[::-1], strict=True)
)
ROUTES = ("demand", "reversed")
OBS_SCALE = 40.0  # vehicles a 300 m lane holds, about: 300 / (5 m + 2.5 m gap)
TRAIN_SEEDS = (0, 1, 2)  # each method is trained three times
EVAL_SEEDS = tuple(range(1000, 1010))  # held-out demand draws, never trained on
EPISODES = 200  # simulated hours of training per run, the same for every method
HIDDEN = 64
GAMMA_RL = 0.99  # per five-second decision: a horizon of about 100 decisions, 8 minutes
# DQN (Mnih et al., 2015): replay, a target network, epsilon-greedy exploration, Huber loss
DQN_LR, DQN_BATCH, DQN_BUFFER, DQN_TARGET_EVERY = 1e-3, 64, 50_000, 500
EPS_END, EPS_DECISIONS = 0.05, 30_000
# PPO (Schulman et al., 2017): clipped ratio, GAE, an entropy bonus; actor and critic apart
PPO_CLIP, PPO_LAMBDA, PPO_EPOCHS, PPO_MINIBATCH = 0.2, 0.95, 10, 256
PPO_LR_ACTOR, PPO_LR_CRITIC, PPO_ENTROPY, PPO_EPISODES_PER_UPDATE = 3e-4, 1e-3, 0.01, 2
REWARD_SCALE = 100.0  # rewards (halted vehicles x seconds per decision) divided by this
# The tuned fixed-time plan: Webster's 15-minute greens times one factor, chosen like the DQN's
# training, on training hours (the first TUNE_HOURS demand draws of training run 0), never on the
# held-out ones, by a bracketing search along the ladder
TUNE_HOURS = 10
TUNE_LADDER = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0)
TUNE_START = (1.0, 1.2, 1.4, 1.6)
STUCK = ("E_in", "W_in")  # the side street's detectors, which read zero in the break-it test
FAULT_S = 300  # an approach that reads no vehicle for FAULT_S / DECISION_S decisions in a row
# (60: five minutes, a little more when greens change) is a detector fault: fall back to the
# fixed-time plan, as signal controllers do (with Poisson arrivals at 250 veh/h, five minutes
# without a vehicle has a chance of about e^-20)
WORKERS = 9
KIND_LABEL = {"dqn": "DQN", "online": "DQN, online", "ppo": "PPO"}  # trained kinds: slug, label
CONTROLLERS = (
    "Webster, hour", "Webster, 15 min", "Fixed, tuned", "Actuated", "DQN", "DQN, online", "PPO",
)  # fmt: skip
FIXED = ("Webster, hour", "Webster, 15 min", "Fixed, tuned")  # each with its plan key
PLAN_KEY = {"Webster, hour": "hour", "Webster, 15 min": "15 min", "Fixed, tuned": "tuned",
            "Webster, 15 min, retimed": "retimed"}  # fmt: skip
# File names and command-line values are built from slugs of [A-Za-z0-9_], never from a label:
# SUMO reads a comma in an option's value as a list separator
SLUG = {
    "Webster, hour": "webster_hour",
    "Webster, 15 min": "webster_15min",
    "Webster, 15 min, retimed": "webster_15min_retimed",
    "Fixed, tuned": "fixed_tuned",
    "Actuated": "actuated",
    "DQN": "dqn",
    "DQN, online": "online",
    "PPO": "ppo",
    "dead": "dead",
    "dead, fallback": "dead_fallback",
}
SLUG_RE = re.compile(r"[A-Za-z0-9_]+")
VIEW_M = 160  # the intersection figure shows the 160 m nearest the junction


# --------------------------------------------------------------------------- SUMO


def sumo_bin(tool: str) -> Path:
    """A SUMO executable from the eclipse-sumo wheel (SUMO_HOME is the installed package).

    Args:
        tool: The program's name, e.g. "netconvert".

    Returns:
        Its path.
    """
    import sumo  # noqa: PLC0415 (the wheel's package only locates the binaries)

    return Path(sumo.SUMO_HOME) / "bin" / tool


def netconvert(sumo_dir: Path, out: Path, *extra: str) -> None:
    """Convert the plain nodes, edges, and connections in ``sumo_dir`` into a SUMO network.

    Args:
        sumo_dir: Where ``net.nod.xml``, ``net.edg.xml``, and ``net.con.xml`` are.
        out: The network file to write.
        *extra: More netconvert options.

    Raises:
        RuntimeError: If netconvert fails.
    """
    cmd = [
        str(sumo_bin("netconvert")),
        "-n", str(sumo_dir / "net.nod.xml"),
        "-e", str(sumo_dir / "net.edg.xml"),
        "-x", str(sumo_dir / "net.con.xml"),
        "--no-turnarounds",
        "--tls.yellow.time", str(YELLOW_S),
        "--tls.allred.time", str(ALL_RED_S),
        *extra,
        "--seed", str(SEED),
        "-o", str(out),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"netconvert failed: {done.stderr}")


def build_network(sumo_dir: Path) -> dict[str, Path]:
    """Write the intersection (nodes, edges, through movements), its two programs, and demands.

    The network is converted twice: once with a static program, whose phases the controllers
    below hold one by one, and once with SUMO's gap-based actuated program at its defaults (the
    same phases; detectors placed by netconvert).

    Args:
        sumo_dir: Where the SUMO inputs are written.

    Returns:
        Paths of the network, the actuated network, the demand, the reversed demand, and the
        saturation-flow demand.
    """
    sumo_dir.mkdir(parents=True, exist_ok=True)
    far = {"N": (0, APPROACH_M), "S": (0, -APPROACH_M), "E": (APPROACH_M, 0), "W": (-APPROACH_M, 0)}
    nodes = ['<node id="C" x="0" y="0" type="traffic_light" tl="C"/>']
    nodes += [f'<node id="{k}" x="{x}" y="{y}" type="priority"/>' for k, (x, y) in far.items()]
    edges = []
    for k in far:
        edges.append(f'<edge id="{k}_in" from="{k}" to="C" numLanes="1" speed="{SPEED_MS}"/>')
        edges.append(f'<edge id="C_{k}" from="C" to="{k}" numLanes="1" speed="{SPEED_MS}"/>')
    cons = [f'<connection from="{a}" to="{b}"/>' for a, b in EXITS.items()]
    (sumo_dir / "net.nod.xml").write_text("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n")
    (sumo_dir / "net.edg.xml").write_text("<edges>\n" + "\n".join(edges) + "\n</edges>\n")
    (sumo_dir / "net.con.xml").write_text(
        "<connections>\n" + "\n".join(cons) + "\n</connections>\n"
    )
    files = {
        "net": sumo_dir / "net.net.xml",
        "net_actuated": sumo_dir / "net_actuated.net.xml",
        "demand": sumo_dir / "demand.rou.xml",
        "reversed": sumo_dir / "demand_reversed.rou.xml",
        "saturation": sumo_dir / "saturation.rou.xml",
    }
    netconvert(sumo_dir, files["net"])
    netconvert(sumo_dir, files["net_actuated"], "--tls.default-type", "actuated")
    write_demand(files["demand"], DEMAND)
    write_demand(files["reversed"], REVERSED)
    write_demand(files["saturation"], ((0, 150, 3600, 0),), ("N_in",))  # a car a second, north
    return files


def tl_program(net: Path) -> dict:
    """The traffic light's program in a network file.

    Args:
        net: A network written by ``build_network``.

    Returns:
        The program's type and its phases' states, durations, and minimum and maximum
        durations (None where the program has none).
    """
    tl = ET.parse(net).getroot().find("tlLogic")
    assert tl is not None, f"no traffic light in {net}"
    phases = tl.findall("phase")

    def opt(p: ET.Element, key: str) -> float | None:
        return float(p.attrib[key]) if key in p.attrib else None

    return {
        "type": tl.attrib["type"],
        "states": [p.attrib["state"] for p in phases],
        "durations": [float(p.attrib["duration"]) for p in phases],
        "min_dur": [opt(p, "minDur") for p in phases],
        "max_dur": [opt(p, "maxDur") for p in phases],
    }


def write_demand(
    path: Path,
    profile: tuple[tuple[int, int, float, float], ...],
    approaches: tuple[str, ...] = APPROACHES,
) -> None:
    """Route file: through traffic, Bernoulli arrivals each second (Poisson-like).

    Args:
        path: The route file to write.
        profile: (begin s, end s, north-south veh/h per approach, east-west veh/h per approach).
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


class Intersection:
    """One SUMO run through libsumo: observe queues, choose a phase every five seconds.

    The observation is each approach's halted and total vehicles (scaled by ``OBS_SCALE``), the
    green phase, and the seconds since it started (capped at 60). A switch runs the yellow and
    the all-red first. The reward is minus the halted vehicles summed over every second. Use it
    as a context manager: the simulation is closed however the block ends (one per process).
    """

    def __init__(
        self,
        files: dict[str, Path],
        seed: int,
        *,
        routes: str = "demand",
        stuck: tuple[str, ...] = (),
        tripinfo: Path | None = None,
        actuated: bool = False,
    ) -> None:
        """Start SUMO.

        Args:
            files: From ``build_network``.
            seed: SUMO's random seed (the arrivals and the drivers' imperfection).
            routes: Which route file: "demand", "reversed", or "saturation".
            stuck: Approaches whose detectors read zero.
            tripinfo: Where to write each vehicle's trip summary, if anywhere.
            actuated: Run SUMO's actuated program untouched, instead of holding phases.
        """
        import libsumo  # noqa: PLC0415 (one SUMO per process: imported where it runs)

        self.ts = libsumo
        net = files["net_actuated" if actuated else "net"]
        cmd = [
            "sumo", "-n", str(net), "-r", str(files[routes]),
            "--seed", str(seed), "--no-step-log", "--no-warnings",
            "--time-to-teleport", "-1", "--end", str(END_S),
        ]  # fmt: skip
        if tripinfo is not None:  # every vehicle, also those unfinished or never let in
            assert "," not in str(tripinfo), "SUMO splits option values at commas"
            cmd += [
                "--tripinfo-output", str(tripinfo),
                "--tripinfo-output.write-unfinished", "true",
                "--tripinfo-output.write-undeparted", "true",
            ]  # fmt: skip
        self.ts.start(cmd)
        self.closed = False
        self.left: dict[str, int] = {}
        self.stuck = stuck
        self.phase, self.since = NS, 0
        self.lights: list[int] = []  # each second: NS or EW with green, -1 in a change interval
        if not actuated:
            self.set_phase(GREEN_PHASE[NS])

    def __enter__(self) -> Intersection:
        """Enter a ``with`` block.

        Returns:
            The running simulation.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the simulation, whether or not the block raised."""
        self.close()

    def set_phase(self, index: int) -> None:
        """Hold one of SUMO's phases until told otherwise.

        Args:
            index: The phase's index in the static program.
        """
        self.ts.trafficlight.setPhase("C", index)
        self.ts.trafficlight.setPhaseDuration("C", 1e6)

    def halted(self, approach: str) -> int:
        """Vehicles at a standstill on an approach (speed under 0.1 m/s), as its detector says.

        Args:
            approach: One of ``APPROACHES``.

        Returns:
            The count, zero if the approach's detector is dead.
        """
        return 0 if approach in self.stuck else self.ts.edge.getLastStepHaltingNumber(approach)

    def obs(self) -> np.ndarray:
        """The agent's view: halted and total vehicles per approach, phase, time in phase.

        Returns:
            Ten numbers in float32.
        """
        q = [self.halted(a) for a in APPROACHES]
        n = [0 if a in self.stuck else self.ts.edge.getLastStepVehicleNumber(a) for a in APPROACHES]
        x = np.array([*q, *n], float) / OBS_SCALE
        return np.array([*x, float(self.phase), min(self.since, 60) / 60.0], np.float32)

    def run(self, seconds: int) -> float:
        """Advance whole seconds, recording the light each second.

        Args:
            seconds: How many.

        Returns:
            The halted vehicles summed over them, all approaches (the truth, not the detectors).
        """
        total = 0.0
        green = {GREEN_PHASE[NS]: NS, GREEN_PHASE[EW]: EW}
        for _ in range(seconds):
            self.ts.simulationStep()
            total += sum(self.ts.edge.getLastStepHaltingNumber(a) for a in APPROACHES)
            self.lights.append(green.get(self.ts.trafficlight.getPhase("C"), -1))
        return total

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Apply a phase for one decision (with the change interval first, if it switches).

        Args:
            action: ``NS`` or ``EW``.

        Returns:
            The next observation, the reward (minus halted vehicle-seconds), and whether the
            hour is over: a time limit, not a terminal state (the queues are still there).
        """
        halted = 0.0
        if action != self.phase:
            self.set_phase(GREEN_PHASE[self.phase] + 1)
            halted += self.run(YELLOW_S)
            self.set_phase(GREEN_PHASE[self.phase] + 2)
            halted += self.run(ALL_RED_S)
            self.phase, self.since = action, 0
            self.set_phase(GREEN_PHASE[action])
        halted += self.run(DECISION_S)
        self.since += DECISION_S
        return self.obs(), -halted, self.ts.simulation.getTime() >= HOUR_S

    def close(self) -> dict[str, int]:
        """Stop SUMO once (later calls return the same counts).

        Returns:
            The vehicles still waiting to enter and still inside when it stopped.
        """
        if not self.closed:
            self.closed = True
            try:
                self.left = {
                    "pending": len(self.ts.simulation.getPendingVehicles()),
                    "inside": self.ts.vehicle.getIDCount(),
                }
            finally:
                self.ts.close()
        return self.left


Controller = Callable[[np.ndarray, Intersection], int]


def saturation_flow(files: dict[str, Path]) -> dict:
    """Measure SUMO's saturation flow: discharge a standing queue at a green light.

    North-south is held red for 150 s while a car is offered every second, then turns green.
    Each vehicle leaving the approach is timed; the headways from the fifth vehicle on (past the
    start-up loss, as the HCM measures it) give the saturation flow.

    Args:
        files: From ``build_network``.

    Returns:
        The queue at the green, the mean saturation headway, and the saturation flow in vehicles
        per hour of green.
    """
    with Intersection(files, SEED, routes="saturation") as env:
        env.set_phase(GREEN_PHASE[EW])
        env.run(150)
        queue = env.ts.edge.getLastStepHaltingNumber("N_in")
        env.set_phase(GREEN_PHASE[NS])
        t0 = env.ts.simulation.getTime()
        before = set(env.ts.edge.getLastStepVehicleIDs("N_in"))
        times: list[float] = []
        for _ in range(90):
            env.ts.simulationStep()
            now = set(env.ts.edge.getLastStepVehicleIDs("N_in"))
            times += [env.ts.simulation.getTime() - t0] * len(before - now)
            before = now
    t = np.array(times)
    headway = (t[24] - t[4]) / 20  # vehicles 5 to 25
    return {"queue": int(queue), "headway_s": float(headway), "veh_per_h": 3600.0 / headway}


def webster(q_ns: float, q_ew: float, s: float) -> dict:
    """Webster's (1958) optimal fixed cycle and the greens that share it by flow ratio.

    C0 = (1.5 L + 5) / (1 - Y), with Y the sum of the critical flow ratios y = q / s, and each
    phase's effective green (C0 - L) y_i / Y. The whole intergreen counts as lost time and the
    displayed green as effective green (no start-up loss).

    Args:
        q_ns: Flow on a north-south approach, veh/h.
        q_ew: Flow on an east-west approach, veh/h.
        s: Saturation flow, veh/h of green.

    Returns:
        The flow ratios, the cycle, and the two greens in seconds.
    """
    y = np.array([q_ns, q_ew]) / s
    Y = float(y.sum())
    assert Y < 1, "demand above capacity: no fixed cycle serves it"
    C = (1.5 * LOST_S + 5) / (1 - Y)
    g = (C - LOST_S) * y / Y
    return {"y": y.tolist(), "Y": Y, "cycle_s": C, "green_s": g.tolist()}


def webster_plans(
    s: float, profile: tuple[tuple[int, int, float, float], ...] = DEMAND, factor: float = 1.0
) -> dict[str, list[tuple[int, int, int]]]:
    """Webster's fixed-time plans for a demand profile, as (start s, NS green s, EW green s).

    Args:
        s: The measured saturation flow.
        profile: The demand the plans are computed from.
        factor: Every 15-minute green is multiplied by this before rounding (1 is Webster's).

    Returns:
        "hour": one plan from the hour's mean flows; "15 min": one per quarter hour, which knows
        the demand schedule in advance.
    """
    q_ns = float(np.mean([d[2] for d in profile]))
    q_ew = float(np.mean([d[3] for d in profile]))
    hour = webster(q_ns, q_ew, s)
    plans = {"hour": [(0, *(round(g) for g in hour["green_s"]))]}
    plans["15 min"] = [
        (b, *(round(factor * g) for g in webster(ns, ew, s)["green_s"])) for b, _, ns, ew in profile
    ]
    return plans


def cycle(g_ns: int, g_ew: int) -> tuple[tuple[int, int], ...]:
    """One fixed-time cycle as (SUMO phase, seconds): each green, its yellow, its all-red.

    Args:
        g_ns: North-south green, seconds.
        g_ew: East-west green, seconds.

    Returns:
        The six phases in order.
    """
    return ((0, g_ns), (1, YELLOW_S), (2, ALL_RED_S), (3, g_ew), (4, YELLOW_S), (5, ALL_RED_S))


def approach_of(vehicle_id: str) -> str:
    """The approach a vehicle came from, read from its flow's id ("f2_E_in.17" -> "E_in").

    Args:
        vehicle_id: SUMO's id of a vehicle from ``write_demand``'s flows.

    Returns:
        One of ``APPROACHES``.
    """
    approach = vehicle_id.rsplit(".", 1)[0].split("_", 1)[1]
    assert approach in APPROACHES, vehicle_id
    return approach


def trip_summary(path: Path, left: dict[str, int]) -> dict:
    """Mean delay per vehicle and the side street's service, from a trip file; then delete it.

    The delay is the time lost while driving plus the wait to enter the network. Vehicles still
    inside, or never let in, when the run stops are written too (SUMO's write-unfinished and
    write-undeparted), with the delay they had accrued by then: a starved approach counts, as a
    lower bound.

    Args:
        path: The tripinfo file (our own SUMO output), deleted once read.
        left: The vehicles still waiting and inside at the stop, from ``Intersection.close``.

    Returns:
        The vehicles, the mean delay in seconds, and the side street's (``STUCK``) vehicles and
        how many of them finished their trip.
    """
    trips = ET.parse(path).getroot().findall("tripinfo")
    path.unlink()
    loss = np.array([float(t.attrib["timeLoss"]) + float(t.attrib["departDelay"]) for t in trips])
    done = np.array(
        [t.attrib.get("vaporized", "") == "" and float(t.attrib["arrival"]) >= 0 for t in trips]
    )
    side = np.array([approach_of(t.attrib["id"]) in STUCK for t in trips])
    assert len(trips) - done.sum() == left["pending"] + left["inside"], "a vehicle unaccounted"
    return {
        "vehicles": len(trips),
        "delay_s": float(loss.mean()),
        "side_vehicles": int(side.sum()),
        "side_finished": int((side & done).sum()),
    }


def run_fixed(
    files: dict[str, Path],
    seed: int,
    plan: list[tuple[int, int, int]],
    trip: Path,
    *,
    routes: str = "demand",
) -> dict:
    """Evaluate a fixed-time plan, second by second, then clear the network.

    Args:
        files: From ``build_network``.
        seed: The demand draw.
        plan: (start s, NS green s, EW green s); a plan takes over at the next cycle start.
        trip: The trip file to write (read and deleted here).
        routes: Which demand.

    Returns:
        The trip summary, the halted vehicle-seconds in the hour, what was left at the cap, and
        the light each second of the hour.
    """
    with Intersection(files, seed, routes=routes, tripinfo=trip) as env:
        halted, t = 0.0, 0
        while env.ts.simulation.getTime() < HOUR_S or env.ts.simulation.getMinExpectedNumber() > 0:
            _, g_ns, g_ew = next(p for p in reversed(plan) if p[0] <= t)
            for phase, dur in cycle(g_ns, g_ew):
                env.set_phase(phase)
                for _ in range(dur):
                    if t >= END_S:
                        break
                    h = env.run(1)
                    halted += h if t < HOUR_S else 0.0
                    t += 1
            if t >= END_S:
                break
        lights = np.array(env.lights[:HOUR_S])
        left = env.close()
    return {**trip_summary(trip, left), "halted": halted, **left, "_lights": lights}


def run_actuated(files: dict[str, Path], seed: int, trip: Path, *, routes: str = "demand") -> dict:
    """Evaluate SUMO's actuated program, as netconvert wrote it, then clear the network.

    Args:
        files: From ``build_network``.
        seed: The demand draw.
        trip: The trip file to write (read and deleted here).
        routes: Which demand.

    Returns:
        The trip summary, the halted vehicle-seconds in the hour, what was left at the cap, and
        the light each second of the hour.
    """
    with Intersection(files, seed, routes=routes, tripinfo=trip, actuated=True) as env:
        sim = env.ts.simulation
        halted, t = 0.0, 0
        while t < END_S and (t < HOUR_S or sim.getMinExpectedNumber() > 0):
            h = env.run(1)
            halted += h if t < HOUR_S else 0.0
            t += 1
        lights = np.array(env.lights[:HOUR_S])
        left = env.close()
    return {**trip_summary(trip, left), "halted": halted, **left, "_lights": lights}


# --------------------------------------------------------------------------- learning to switch


def mlp(n_out: int) -> nn.Sequential:
    """The Q-network: ten inputs, two ReLU layers of ``HIDDEN`` units.

    Args:
        n_out: Outputs.

    Returns:
        The network.
    """
    return nn.Sequential(
        nn.Linear(10, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, n_out),
    )


def torch_setup(seed: int) -> None:
    """One CPU thread and deterministic kernels, so a run repeats exactly on any machine.

    Args:
        seed: PyTorch's global seed.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def episode_seeds(seed: int) -> np.ndarray:
    """The demand draws a training run sees, disjoint from ``EVAL_SEEDS``.

    Args:
        seed: The training run's seed.

    Returns:
        ``EPISODES`` SUMO seeds.
    """
    rng = np.random.default_rng([SEED, seed])
    draws = rng.choice(np.arange(10_000, 1_000_000), size=EPISODES, replace=False)
    assert not set(draws.tolist()) & set(EVAL_SEEDS)
    return draws


def train_dqn(files: dict[str, Path], seed: int, replay: bool) -> dict:
    """Deep Q-learning: with replay and a target network, or online (each step once, as it comes).

    The hour's end is a time limit, not a terminal state: no transition is stored as terminal,
    and every target bootstraps from the next state.

    Args:
        files: From ``build_network``.
        seed: The run's seed (network weights, exploration, demand draws).
        replay: True for DQN (Mnih et al., 2015); False learns from each transition once, as it
            happens, with no replay buffer and no target network.

    Returns:
        The trained weights, the halted vehicle-seconds of each training hour, and the time.
    """
    torch_setup(seed)
    rng = np.random.default_rng([SEED, seed, 1])
    q, target = mlp(2), mlp(2)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=DQN_LR)
    buf = {
        "s": np.zeros((DQN_BUFFER, 10), np.float32),
        "a": np.zeros(DQN_BUFFER, np.int64),
        "r": np.zeros(DQN_BUFFER, np.float32),
        "s2": np.zeros((DQN_BUFFER, 10), np.float32),
    }
    n, curve, t0 = 0, [], time.perf_counter()
    batch = DQN_BATCH if replay else 1
    for ep_seed in episode_seeds(seed):
        with Intersection(files, int(ep_seed)) as env:
            obs, hour_over, hour = env.obs(), False, 0.0
            while not hour_over:
                eps = max(EPS_END, 1.0 - n / EPS_DECISIONS)
                if rng.random() < eps:
                    a = int(rng.integers(2))
                else:
                    with torch.no_grad():
                        a = int(q(torch.from_numpy(obs)).argmax())
                obs2, r, hour_over = env.step(a)
                hour -= r
                i = n % DQN_BUFFER
                buf["s"][i], buf["a"][i], buf["r"][i] = obs, a, r / REWARD_SCALE
                buf["s2"][i] = obs2
                n += 1
                obs = obs2
                if n < batch:
                    continue
                idx = rng.integers(0, min(n, DQN_BUFFER), batch) if replay else np.array([i])
                s, a_, r_, s2 = (torch.from_numpy(buf[k][idx]) for k in ("s", "a", "r", "s2"))
                with torch.no_grad():
                    nxt = (target if replay else q)(s2).max(1).values
                y = r_ + GAMMA_RL * nxt  # bootstrap always: no state here is terminal
                pred = q(s).gather(1, a_[:, None]).squeeze(1)
                assert pred.shape == y.shape == (batch,)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                if replay and n % DQN_TARGET_EVERY == 0:
                    target.load_state_dict(q.state_dict())
        curve.append(hour)
    return {"state": q.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}


def tanh_mlp(n_out: int) -> nn.Sequential:
    """PPO's networks: ten inputs, two tanh layers of ``HIDDEN`` units.

    Args:
        n_out: Outputs.

    Returns:
        The network.
    """
    return nn.Sequential(
        nn.Linear(10, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.Tanh(),
        nn.Linear(HIDDEN, n_out),
    )


class ActorCritic(nn.Module):
    """PPO's two networks: a policy over the two phases and a value of the state."""

    def __init__(self) -> None:
        """Separate actor and critic: the critic's large errors cannot swamp the policy's."""
        super().__init__()
        self.actor = tanh_mlp(2)
        self.critic = tanh_mlp(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Action logits and the state's value.

        Args:
            x: Observations, ``(..., 10)``.

        Returns:
            The logits ``(..., 2)`` and the values ``(...)``.
        """
        return self.actor(x), self.critic(x).squeeze(-1)


def gae(
    rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Generalized advantage estimates (Schulman et al., 2016), hour by hour, in float32.

    delta_t = r_t + gamma V(s_{t+1}) - V(s_t), where at an hour's last decision V(s_{t+1}) is the
    critic's value of the state the hour ends in (a time limit, not a terminal state); the sum
    of deltas stops at each hour's end. The arrays are float32 and the constants Python floats,
    which NumPy does not let promote the result (NEP 50).

    Args:
        rewards: One per decision (float32).
        values: The critic's value of each decision's state (float32).
        next_values: The critic's value of each decision's next state (float32).
        ends: True at an hour's last decision.

    Returns:
        A_t = sum_k (gamma lambda)^k delta_{t+k}, within the hour.
    """
    adv = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + GAMMA_RL * next_values[t] - values[t]
        running = delta + GAMMA_RL * PPO_LAMBDA * (0.0 if ends[t] else running)
        adv[t] = running
    return adv


def ppo_rollout(ac: ActorCritic, files: dict[str, Path], seeds: np.ndarray) -> dict:
    """Run the current stochastic policy for a few simulated hours and keep every decision.

    Args:
        ac: The actor-critic.
        files: From ``build_network``.
        seeds: One demand draw per hour.

    Returns:
        Observations, actions, their log-probabilities, scaled rewards, values, hour ends, the
        critic's value of each hour's last state, and each hour's halted vehicle-seconds.
    """
    keys = ("obs", "act", "logp", "rew", "val", "end", "v_end", "hours")
    out: dict[str, list] = {k: [] for k in keys}
    for ep_seed in seeds:
        with Intersection(files, int(ep_seed)) as env:
            obs, hour_over, hour = env.obs(), False, 0.0
            while not hour_over:
                with torch.no_grad():
                    logits, v = ac(torch.from_numpy(obs))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                obs2, r, hour_over = env.step(int(a))
                hour -= r
                out["obs"].append(obs)
                out["act"].append(int(a))
                out["logp"].append(float(dist.log_prob(a)))
                out["rew"].append(r / REWARD_SCALE)
                out["val"].append(float(v))
                out["end"].append(hour_over)
                obs = obs2
            with torch.no_grad():  # the hour stops here, the state does not end: bootstrap
                out["v_end"].append(float(ac(torch.from_numpy(obs))[1]))
        out["hours"].append(hour)
    return out


def ppo_update(
    ac: ActorCritic, opts: tuple[torch.optim.Optimizer, ...], roll: dict, rng: np.random.Generator
) -> None:
    """Several epochs of minibatch steps on PPO's clipped objective and the critic's error.

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
    next_values[:-1] = values[1:]
    assert ends[-1] and ends.sum() == len(roll["v_end"])
    next_values[ends] = np.array(roll["v_end"], np.float32)
    adv = gae(np.array(roll["rew"], np.float32), values, next_values, ends)
    ret = torch.from_numpy(adv + values)
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


def train_ppo(files: dict[str, Path], seed: int) -> dict:
    """Proximal policy optimization with a clipped ratio, an entropy bonus, and GAE.

    Args:
        files: From ``build_network``.
        seed: The run's seed.

    Returns:
        The trained weights, the halted vehicle-seconds of each training hour, and the time.
    """
    torch_setup(seed)
    rng = np.random.default_rng([SEED, seed, 2])
    ac = ActorCritic()
    opts = (
        torch.optim.Adam(ac.actor.parameters(), lr=PPO_LR_ACTOR),
        torch.optim.Adam(ac.critic.parameters(), lr=PPO_LR_CRITIC),
    )
    seeds = episode_seeds(seed)
    curve, t0 = [], time.perf_counter()
    for start in range(0, EPISODES, PPO_EPISODES_PER_UPDATE):
        roll = ppo_rollout(ac, files, seeds[start : start + PPO_EPISODES_PER_UPDATE])
        curve += roll["hours"]
        ppo_update(ac, opts, roll, rng)
    return {"state": ac.state_dict(), "curve": curve, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- evaluation


def greedy(kind: str, state: dict[str, torch.Tensor]) -> Controller:
    """The trained network as a controller: the action with the best value (or probability).

    PPO trains a stochastic policy and is evaluated greedily, by its most likely green.

    Args:
        kind: "dqn", "online", or "ppo".
        state: The trained weights.

    Returns:
        A function from an observation (and the simulation) to a green.
    """
    if kind == "ppo":
        model = ActorCritic()
        model.load_state_dict(state)
        net = model.actor
    else:
        net = mlp(2)
        net.load_state_dict(state)
    net.eval()

    def act(obs: np.ndarray, env: Intersection) -> int:  # noqa: ARG001 (the Controller type)
        with torch.no_grad():
            return int(net(torch.from_numpy(obs)).argmax())

    return act


def run_learned(
    files: dict[str, Path],
    seed: int,
    act: Controller,
    trip: Path,
    *,
    routes: str = "demand",
    stuck: tuple[str, ...] = (),
    fallback: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Evaluate a learned controller on one demand draw, then clear the network.

    Args:
        files: From ``build_network``.
        seed: The demand draw (one of ``EVAL_SEEDS``).
        act: The controller.
        trip: The trip file to write (read and deleted here).
        routes: Which demand.
        stuck: Approaches whose detectors read zero.
        fallback: A fixed-time plan to switch to on a detector fault (``FAULT_S``), or None; the
            plan's entry in force is chosen by the time.

    Returns:
        The trip summary, the vehicles left at the cap, when a fault was declared, the light
        each second of the hour, and the side street's queue after each decision.
    """
    with Intersection(files, seed, routes=routes, stuck=stuck, tripinfo=trip) as env:
        sim = env.ts.simulation
        obs = env.obs()
        silent = dict.fromkeys(APPROACHES, 0)
        fault_at: int | None = None
        side_queue: list[tuple[int, int]] = []
        cap = END_S - 2 * DECISION_S  # a decision with a change of green takes 10 s
        while sim.getTime() < cap and (sim.getTime() < HOUR_S or sim.getMinExpectedNumber() > 0):
            t0 = int(sim.getTime())
            if fallback is not None and fault_at is None:
                n_seen = obs[4:8] * OBS_SCALE
                for k, a in enumerate(APPROACHES):
                    silent[a] = silent[a] + DECISION_S if n_seen[k] == 0 else 0
                if max(silent.values()) >= FAULT_S:
                    fault_at = t0
            if fallback is not None and fault_at is not None:
                _, g_ns, g_ew = next(p for p in reversed(fallback) if p[0] <= t0)
                for phase, dur in cycle(g_ns, g_ew):  # one cycle, never past SUMO's end
                    env.set_phase(phase)
                    env.run(min(dur, END_S - int(sim.getTime())))
                env.phase, env.since = NS, 0
                env.set_phase(GREEN_PHASE[NS])
                obs = env.obs()
            else:
                obs, _, _ = env.step(act(obs, env))
            waiting = sum(env.ts.edge.getLastStepHaltingNumber(a) for a in STUCK)
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
        lights: The light each second (``NS``, ``EW``, or -1), at least ``HOUR_S`` long.
        window_s: Window length.

    Returns:
        One share per window of the hour.
    """
    w = lights[:HOUR_S].reshape(-1, window_s)
    ns, ew = (w == NS).sum(1), (w == EW).sum(1)
    return ns / np.maximum(ns + ew, 1)


# --------------------------------------------------------------------------- jobs


def to_numpy(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """A state dict as NumPy arrays, to cross process boundaries by plain pickling.

    Args:
        state: PyTorch weights.

    Returns:
        The same weights as arrays.
    """
    return {k: v.detach().cpu().numpy().copy() for k, v in state.items()}


def to_torch(state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """The inverse of ``to_numpy``.

    Args:
        state: Weights as arrays.

    Returns:
        The same weights as tensors.
    """
    return {k: torch.from_numpy(v) for k, v in state.items()}


def job(spec: dict) -> dict:
    """One unit of SUMO work, run in its own process (libsumo holds one simulation per process).

    Args:
        spec: ``task`` is "train" (``kind``: dqn, online, ppo; ``seed``), "fixed" (``plan``),
            "actuated", or "eval" (``kind``, ``state``, ``stuck``, ``fallback``). Every
            evaluation carries ``name``, ``slug``, ``seed``, ``train_seed``, and ``routes``;
            every spec carries ``files`` and ``tmp``.

    Returns:
        The task's result, with the spec's identifying fields.

    Raises:
        ValueError: If the task is unknown.
    """
    torch_setup(spec.get("seed", SEED))
    files, tmp, task = spec["files"], spec["tmp"], spec["task"]
    if task == "train":
        if spec["kind"] == "ppo":
            out = train_ppo(files, spec["seed"])
        else:
            out = train_dqn(files, spec["seed"], replay=spec["kind"] == "dqn")
        out["state"] = to_numpy(out["state"])
        return {**out, "kind": spec["kind"], "seed": spec["seed"]}
    assert SLUG_RE.fullmatch(spec["slug"]), f"not a slug: {spec['slug']!r}"
    trip = tmp / f"{spec['slug']}_{spec['train_seed']}_{spec['seed']}_{spec['routes']}.xml"
    if task == "fixed":
        out = run_fixed(files, spec["seed"], spec["plan"], trip, routes=spec["routes"])
    elif task == "actuated":
        out = run_actuated(files, spec["seed"], trip, routes=spec["routes"])
    elif task == "eval":
        act = greedy(spec["kind"], to_torch(spec["state"]))
        out = run_learned(files, spec["seed"], act, trip, routes=spec["routes"],
                          stuck=spec["stuck"], fallback=spec["fallback"])  # fmt: skip
    else:
        raise ValueError(f"unknown task {task!r}")
    keep = ("name", "seed", "train_seed", "routes")
    return {**out, **{k: spec[k] for k in keep}}


def run_jobs(specs: list[dict]) -> list[dict]:
    """Run jobs in spawned processes (never forked after torch), results in submission order.

    Args:
        specs: The jobs, for ``job``.

    Returns:
        Their results.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(WORKERS, len(specs)), mp_context=ctx) as pool:
        return list(pool.map(job, specs))


def eval_spec(base: dict, name: str, seed: int, **fields: object) -> dict:
    """An evaluation job of a named controller on one held-out hour.

    Args:
        base: ``files`` and ``tmp``.
        name: The controller's label (a key of ``SLUG``).
        seed: The demand draw.
        **fields: The task and its own fields; ``routes`` and ``train_seed`` default to the
            training demand and 0.

    Returns:
        The job's spec.
    """
    return {**base, "name": name, "slug": SLUG[name], "seed": seed, "routes": "demand",
            "train_seed": 0, **fields}  # fmt: skip


# --------------------------------------------------------------------------- figures: bridges

BLUE, ORANGE, GREEN, PINK = aml_style.PALETTE
POOR_COLOR = ORANGE


def bare(ax: Axes) -> None:
    """Hide an axes' frame and ticks (for drawn pictures).

    Args:
        ax: The axes.
    """
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def ratings_figure(df: pd.DataFrame, out_dir: Path) -> dict:
    """Illinois's bridges in 2024 by lowest rating, the poor ones highlighted.

    Args:
        df: The records.
        out_dir: The figure directory.

    Returns:
        The bridges, the poor ones, and the count per rating.
    """
    last = df[df["nbi_year"] == df["nbi_year"].max()]
    n = last["s"].value_counts().reindex(RATINGS, fill_value=0)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    colors = [POOR_COLOR if r <= POOR else BLUE for r in RATINGS]
    ax.bar([str(r) if r > 3 else "≤3" for r in RATINGS], n.to_numpy(), color=colors)
    ax.set(xlabel="lowest rating (Items 58-60)", ylabel="bridges, 2024")
    poor = int(n[RATINGS <= POOR].sum())
    ax.text(0.5, n.iloc[:2].max() + 0.06 * n.max(), f"poor: {poor:,}", color=POOR_COLOR,
            fontsize=12, ha="center")  # fmt: skip
    aml_style.save(fig, out_dir, "ratings")
    return {"bridges": int(n.sum()), "poor": poor, "by_rating": n.to_dict()}


def transitions_figure(mdp: dict, out_dir: Path) -> None:
    """The two transition matrices: doing nothing (counted) and replacing (new bridges).

    Args:
        mdp: From ``bridge_mdp``.
        out_dir: The figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, gridspec_kw={"width_ratios": [N_S, 1.6]})
    labels = ["≤3", *[str(r) for r in RATINGS[1:]]]
    for ax, P, title in (
        (axes[0], mdp["P"][0], "do nothing: rating in two years"),
        (axes[1], mdp["p_new"][None, :].T, "replace"),
    ):
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(P):
            if v >= 0.005:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.5 else aml_style.INK)  # fmt: skip
        ax.set_title(title, fontsize=12)
    axes[0].set(xticks=range(N_S), xticklabels=labels, yticks=range(N_S), yticklabels=labels,
                xlabel="rating two years later", ylabel="rating now")  # fmt: skip
    axes[1].set(xticks=[], yticks=range(N_S), yticklabels=labels, ylabel="first rating, %")
    aml_style.save(fig, out_dir, "transitions")


def picture_figures(out_dir: Path, gamma: float) -> None:
    """The one picture, in three builds: the loop; discounted costs; the policy and its value.

    Args:
        out_dir: The figure directory.
        gamma: The discount factor per step whose powers are drawn.
    """
    light, cost_c, pol_c = "#e8f0f7", "#fbe3d4", "#e3f1e8"

    def box(ax: Axes, xy: tuple[float, float], text: str, color: str, w: float) -> None:
        ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - 0.08), w, 0.16,
                                    boxstyle="round,pad=0.01", facecolor=color,
                                    edgecolor=aml_style.INK, lw=1))  # fmt: skip
        ax.text(*xy, text, ha="center", va="center", fontsize=11, color=aml_style.INK)

    def arrow(ax: Axes, a: tuple, b: tuple, text: str, dy: float = 0.05) -> None:
        ax.annotate("", b, a, arrowprops={"arrowstyle": "-|>", "color": aml_style.INK, "lw": 1.2})
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", fontsize=10,
                color=aml_style.MUTED)  # fmt: skip

    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        ax.set(xlim=(0, 1.6), ylim=(-0.05, 1))
        bare(ax)
        box(ax, (0.35, 0.78), "agency (agent)", pol_c if k == 3 else light, 0.34)
        box(ax, (0.35, 0.30), "bridge + inspection\n(environment)", light, 0.34)
        arrow(ax, (0.22, 0.38), (0.22, 0.70), "")
        ax.text(0.20, 0.54, "state:\nrating 6", ha="right", va="center", fontsize=10,
                color=aml_style.MUTED)  # fmt: skip
        arrow(ax, (0.48, 0.70), (0.48, 0.38), "")
        ax.text(0.50, 0.54, "action: do nothing\nor replace", ha="left", va="center",
                fontsize=10, color=aml_style.MUTED)  # fmt: skip
        arrow(ax, (0.52, 0.30), (0.75, 0.30), "cost, next state", 0.04)
        box(ax, (0.92, 0.30), "two years later", light, 0.30)
        if k >= 2:
            t = np.arange(6)
            x = 1.15 + 0.07 * t
            w = gamma**t
            ax.bar(x, 0.35 * w, width=0.05, bottom=0.55, color=cost_c, edgecolor=aml_style.INK)
            for xi, wi, ti in zip(x, w, t, strict=True):
                ax.text(xi, 0.55 + 0.35 * wi + 0.02, f"{wi:.2f}", ha="center", fontsize=8)
                ax.text(xi, 0.50, f"{2 * ti}", ha="center", fontsize=8, color=aml_style.MUTED)
            ax.text(1.325, 0.43, "years from now", ha="center", fontsize=9,
                    color=aml_style.MUTED)  # fmt: skip
            weight = rf"a cost's weight: $\gamma^k$, $\gamma$ = {gamma:.3f}"
            ax.text(1.325, 0.98, weight, ha="center", fontsize=10)
        if k == 3:
            policy = (
                r"policy $\pi$(rating) gives the action;  value V(rating) = expected"
                "\n"
                r"discounted cost of following $\pi$ from here"
            )
            ax.text(0.35, 0.97, policy, ha="center", va="center", fontsize=10, color=GREEN)
        aml_style.save(fig, out_dir, f"picture_{k}")


def vi_figure(vi: dict, gamma: float, out_dir: Path) -> None:
    """Value iteration: each rating's value by sweep, and the largest change per sweep.

    Args:
        vi: From ``value_iteration``.
        gamma: Its discount factor.
        out_dir: The figure directory.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=aml_style.FULL)
    tr = vi["trace"]
    for i, r in enumerate(RATINGS):
        if r in (3, 4, 6, 9):
            a.plot(tr[:, i], label=f"rating {r if r > 3 else '≤3'}", lw=1.6)
    a.set(xlabel="sweep", ylabel="V (replacements)")
    a.legend(fontsize=9, frameon=False)
    d = np.array(vi["deltas"])
    b.semilogy(np.arange(1, len(d) + 1), d, color=BLUE, lw=1.6, label="largest change")
    k = np.arange(1, len(d) + 1)
    b.semilogy(k, d[0] * gamma ** (k - 1), ls="--", color=aml_style.MUTED, lw=1,
               label=rf"$\gamma^k$ ($\gamma$ = {gamma:.3f})")  # fmt: skip
    b.set(xlabel="sweep", ylabel="max |V_k+1 − V_k|")
    b.legend(fontsize=9, frameon=False)
    aml_style.save(fig, out_dir, "value_iteration")


def policy_figure(sweep: dict, crit: dict, out_dir: Path) -> None:
    """The optimal action by rating and poor-condition cost, at each discount rate.

    Args:
        sweep: Per rate, the policy at each cost of ``PENALTY_GRID``.
        crit: Per rate, the critical cost.
        out_dir: The figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    for ax, (name, pol) in zip(axes, sweep.items(), strict=True):
        ax.imshow(pol.T, origin="lower", aspect="auto", cmap="Blues", vmin=-0.3, vmax=1.3,
                  extent=(PENALTY_GRID[0], PENALTY_GRID[-1], RATINGS[0] - 0.5,
                          RATINGS[-1] + 0.5))  # fmt: skip
        ax.axvline(crit[name], color=ORANGE, lw=1.5)
        ax.text(crit[name] + 0.005, 8.6, f"{crit[name]:.3f}", color=ORANGE, fontsize=10)
        ax.plot([PENALTY], [6.5], marker="v", color=aml_style.INK)
        ax.set(title=f"discount {name} a year", xlabel="cost of two years poor (replacements)")
        ax.text(crit[name] + 0.01, 3.5, "replace", color="white", fontsize=11, va="center")
        ax.text(0.01, 7.5, "do nothing", color=aml_style.INK, fontsize=11, va="center")
    axes[0].set(ylabel="lowest rating", yticks=RATINGS, yticklabels=["≤3", *RATINGS[1:]])
    aml_style.save(fig, out_dir, "policy_map")


def q_surface_figure(surfaces: dict, out_dir: Path) -> None:
    """Q(rating, do nothing) and Q(rating, replace) over the cost of being poor.

    Where the two surfaces cross is the decision boundary of the policy map.

    Args:
        surfaces: ``penalty`` (the grid) and ``Q`` (penalty x state x action).
        out_dir: The figure directory.
    """
    fig = plt.figure(figsize=(7.0, 4.4))
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(surfaces["penalty"], RATINGS)
    aml_style.surface3d(ax, X, Y, surfaces["Q"][:, :, 0].T, cmap="Blues", alpha=0.9)
    aml_style.surface3d(ax, X, Y, surfaces["Q"][:, :, 1].T, cmap="Oranges", alpha=0.7)
    aml_style.style3d(ax, elev=24, azim=-52)
    ax.set_xlabel("cost of two years poor", fontsize=11, labelpad=6)
    ax.set_ylabel("lowest rating", fontsize=11, labelpad=6)
    ax.set_zlabel("Q (replacements)", fontsize=11, labelpad=6)
    ax.set_box_aspect(None, zoom=0.85)
    ax.text2D(0.02, 0.92, "blue: do nothing", color=BLUE, transform=ax.transAxes, fontsize=11)
    ax.text2D(0.02, 0.85, "orange: replace", color=ORANGE, transform=ax.transAxes, fontsize=11)
    aml_style.save(fig, out_dir, "q_surfaces")


def qlearning_figure(ql: dict, out_dir: Path) -> None:
    """Q-learning's largest error against value iteration, by simulated inspections.

    Args:
        ql: From ``q_learning``.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    t = np.array([r["steps"] for r in ql["log"]])
    e = np.array([r["max_error"] for r in ql["log"]])
    same = np.array([r["same_policy"] for r in ql["log"]])
    ax.loglog(t, e, color=BLUE, lw=1.6)
    ax.scatter(t[same], e[same], color=GREEN, s=14, zorder=3, label="same policy as VI")
    ax.scatter(t[~same], e[~same], color=ORANGE, marker="x", s=18, zorder=3,
               label="a different policy")  # fmt: skip
    ax.axvline(ql["last_differs_at"], color=aml_style.MUTED, ls=":", lw=1)
    ax.text(ql["last_differs_at"] * 0.85, e.max() * 0.5, "last\ndifference", ha="right",
            fontsize=9, color=aml_style.MUTED)  # fmt: skip
    ax.set(xlabel="simulated inspections", ylabel="max |Q − Q*|")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    aml_style.save(fig, out_dir, "q_learning")


def markov_figure(test: dict, out_dir: Path) -> None:
    """The chance of a drop in two years, by age, for each rating: flat if Markov.

    Args:
        test: From ``markov_test``.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    centers = np.arange(len(AGE_BANDS) - 1)
    names = [f"{a}-{b}" if b < 200 else f"over {a}" for a, b in pairwise(AGE_BANDS)]
    colors = [*aml_style.PALETTE, "#E69F00"]
    for k, (r, rows) in enumerate(test["by_rating"].items()):
        p = np.array([x["p"] for x in rows])
        lo = np.array([x["lo"] for x in rows])
        hi = np.array([x["hi"] for x in rows])
        err = 100 * np.vstack([p - lo, hi - p])
        ax.errorbar(centers + 0.06 * (k - 2), 100 * p, yerr=err, marker="osD^v"[k], capsize=3,
                    lw=1.4, color=colors[k], label=f"rating {r}")  # fmt: skip
    ax.set(xticks=centers, xticklabels=names, xlabel="age (years since built)",
           ylabel="dropped within two years, %")  # fmt: skip
    ax.legend(fontsize=9, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    aml_style.save(fig, out_dir, "markov")


def agency_figure(agency: dict, policy: np.ndarray, out_dir: Path) -> None:
    """What Illinois did by rating, beside the MDP's policy (shaded where it replaces).

    Args:
        agency: From ``agency_behaviour``.
        policy: The MDP's policy at 2%.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    x = np.arange(N_S)
    ax.bar(x - 0.2, 100 * agency["gone"], 0.4, color=ORANGE, label="no later rated record")
    ax.bar(x + 0.2, 100 * agency["work"], 0.4, color=BLUE, hatch="//",
           label="recorded work (Item 106)")  # fmt: skip
    for i in np.flatnonzero(policy == 1):
        ax.axvspan(i - 0.5, i + 0.5, color=GREEN, alpha=0.12, lw=0)
    ax.text(0.0, ax.get_ylim()[1] * 0.92, "MDP: replace", color=GREEN, fontsize=10)
    ax.set(xticks=x, xticklabels=["≤3", *RATINGS[1:]], xlabel="lowest rating",
           ylabel="within two years, %")  # fmt: skip
    ax.legend(fontsize=9, frameon=False)
    aml_style.save(fig, out_dir, "agency")


# --------------------------------------------------------------------------- figures: signals

CTRL_COLOR = dict(
    zip(CONTROLLERS, ["#9a9a9a", "#6f6f6f", "#3d3d3d", ORANGE, BLUE, PINK, GREEN], strict=True)
)


def intersection_figure(snap: dict, out_dir: Path) -> None:
    """The intersection at one moment: the roads near the junction, and every car on them.

    Args:
        snap: From ``snapshot``.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.set_aspect("equal")
    lim = VIEW_M  # the queues sit near the stop lines; the rest of each approach is moving
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    bare(ax)
    for horizontal in (True, False):
        xy = (-lim, -6) if horizontal else (-6, -lim)
        wh = (2 * lim, 12) if horizontal else (12, 2 * lim)
        ax.add_patch(Rectangle(xy, *wh, color="#d9d9d9", lw=0))
    for (x, y), halted in zip(snap["xy"], snap["halted"], strict=True):
        ax.add_patch(Circle((x, y), 3.5, color=ORANGE if halted else BLUE, lw=0))
    green = "north-south" if snap["phase"] == NS else "east-west"
    ax.set_title(f"t = {snap['t']} s, green: {green}", fontsize=11)
    ax.text(-lim + 10, lim - 30, "waiting", color=ORANGE, fontsize=12)
    ax.text(-lim + 10, lim - 70, "moving", color=BLUE, fontsize=12)
    aml_style.save(fig, out_dir, "intersection")


def demand_figure(plans: dict, sat: float, out_dir: Path) -> None:
    """The assumed demand by quarter hour, and Webster's greens for the hour and per 15 min.

    Args:
        plans: From ``webster_plans``.
        sat: The saturation flow, veh/h of green.
        out_dir: The figure directory.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=aml_style.FULL)
    t = np.array([d[0] for d in DEMAND] + [HOUR_S]) / 60
    a.stairs([d[2] for d in DEMAND], t, color=BLUE, lw=2, label="north-south")
    a.stairs([d[3] for d in DEMAND], t, color=ORANGE, lw=2, ls="--", label="east-west")
    a.set(xlabel="minute", ylabel="veh/h per approach", ylim=(0, 800), xticks=[0, 15, 30, 45, 60])
    a.legend(fontsize=9, frameon=False)
    a.set_title(f"saturation flow {sat:,.0f} veh/h of green", fontsize=11)
    names = ["hour", "0-15", "15-30", "30-45", "45-60"]
    rows = [plans["hour"][0], *plans["15 min"]]
    x = np.arange(len(rows))
    ns = np.array([r[1] for r in rows])
    ew = np.array([r[2] for r in rows])
    b.bar(x, ns, color=BLUE, label="north-south green")
    b.bar(x, ew, bottom=ns, color=ORANGE, hatch="//", label="east-west green")
    b.bar(x, LOST_S, bottom=ns + ew, color="#cccccc", label="yellow + all-red")
    b.set(xticks=x, xticklabels=names, ylabel="seconds per cycle")
    b.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    aml_style.save(fig, out_dir, "webster")


def curves_figure(train: dict, halted: dict, out_dir: Path) -> None:
    """Training: waiting in each simulated hour, three runs per method, against the baselines.

    Args:
        train: Per method, each run's halted vehicle-seconds per training hour.
        halted: Per baseline, the mean halted vehicle-seconds per held-out hour.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    styles = {"DQN": "-", "DQN, online": "--", "PPO": "-."}
    for kind, ls in styles.items():
        c = np.array(train[kind]) / 3600  # vehicle-hours
        k = min(10, c.shape[1])  # a ten-hour moving average
        sm = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in c])
        x = np.arange(k, c.shape[1] + 1)
        ax.plot(x, sm.mean(0), color=CTRL_COLOR[kind], ls=ls, lw=1.8, label=kind)
        ax.fill_between(x, sm.min(0), sm.max(0), color=CTRL_COLOR[kind], alpha=0.18, lw=0)
    for name, ls in (("Webster, hour", ":"), ("Webster, 15 min", ":"), ("Actuated", "--")):
        ax.axhline(halted[name] / 3600, color=CTRL_COLOR[name], ls=ls, lw=1.2, label=name)
    ax.set(xlabel="simulated hours of training", ylabel="waiting, vehicle-hours per hour",
           ylim=(0, None))  # fmt: skip
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
    aml_style.save(fig, out_dir, "curves")


def results_figure(delay: dict, out_dir: Path) -> None:
    """Mean delay per vehicle on ten held-out hours; dots are the three trained runs.

    Args:
        delay: Per controller, the mean delay of each run (one for an untrained controller).
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    for i, name in enumerate(CONTROLLERS):
        vals = np.atleast_1d(delay[name])
        ax.bar(i, vals.mean(), color=CTRL_COLOR[name], alpha=0.85,
               hatch="//" if name == "Actuated" else None)  # fmt: skip
        ax.scatter(np.full(len(vals), i), vals, color=aml_style.INK, s=12, zorder=3)
        ax.text(i, vals.mean() + 0.6, f"{vals.mean():.1f}", ha="center", fontsize=10)
    labels = [c.replace(", ", ",\n") for c in CONTROLLERS]
    ax.set(xticks=range(len(CONTROLLERS)), xticklabels=labels, ylabel="delay per vehicle, s")
    aml_style.save(fig, out_dir, "results")


def adapt_figure(share: dict, out_dir: Path) -> None:
    """North-south's share of green by five minutes, beside the demand's own share.

    Args:
        share: Per controller, the share in each five-minute window.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    t = (np.arange(HOUR_S // 300) + 0.5) * 5
    for name, marker in zip(share, "s^Do", strict=True):
        ax.plot(t, 100 * share[name], marker=marker, ms=4, lw=1.6, color=CTRL_COLOR[name],
                label=name)  # fmt: skip
    ratio = [d[2] / (d[2] + d[3]) for d in DEMAND]
    ax.stairs(100 * np.array(ratio), np.array([0, 15, 30, 45, 60]), color=aml_style.MUTED, lw=1.2,
              ls=":", label="north-south share of demand")  # fmt: skip
    ax.set(xlabel="minute", ylabel="north-south share of green, %", xticks=[0, 15, 30, 45, 60])
    ax.legend(fontsize=9, frameon=False)
    aml_style.save(fig, out_dir, "adapt")


def ppo_clip_figure(out_dir: Path) -> None:
    """PPO's clipped objective against the probability ratio, for a good and a bad action.

    Args:
        out_dir: The figure directory.
    """
    r = np.linspace(0.0, 2.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=False)
    for ax, A, title in ((axes[0], 1.0, "advantage > 0"), (axes[1], -1.0, "advantage < 0")):
        ax.plot(r, r * A, color=aml_style.MUTED, ls="--", lw=1.2, label="r·A (unclipped)")
        obj = np.minimum(r * A, np.clip(r, 1 - PPO_CLIP, 1 + PPO_CLIP) * A)
        ax.plot(r, obj, color=GREEN, lw=2.2, label="PPO: min(r·A, clip(r)·A)")
        ax.axvspan(1 - PPO_CLIP, 1 + PPO_CLIP, color=GREEN, alpha=0.08, lw=0)
        ax.axvline(1.0, color=aml_style.INK, lw=0.8)
        ax.set(title=title, xlabel=r"$r = \pi_{new}(a|s)\,/\,\pi_{old}(a|s)$")
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("objective")
    lo, hi = 1 - PPO_CLIP, 1 + PPO_CLIP
    fig.suptitle(rf"PPO: $\min\,(r A,\ \mathrm{{clip}}(r,\ {lo:.1f},\ {hi:.1f})\, A)$",
                 fontsize=14, y=1.08)  # fmt: skip
    aml_style.save(fig, out_dir, "ppo_clip")


def stuck_figure(queues: dict, fault_at: int, out_dir: Path) -> None:
    """The side street's queue over the hour: detectors working, dead, dead with a fallback.

    Args:
        queues: Per case, (second, vehicles waiting) after each decision of one run.
        fault_at: When that run's fault check declared the fault, seconds.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    styles = {"working": (BLUE, "-"), "dead": (ORANGE, "--"), "dead, fallback": (GREEN, "-")}
    for name, series in queues.items():
        q = np.array(series)
        keep = q[:, 0] <= HOUR_S
        ax.plot(q[keep, 0] / 60, q[keep, 1], color=styles[name][0], ls=styles[name][1], lw=1.4,
                label=name)  # fmt: skip
    ax.axvline(fault_at / 60, color=aml_style.MUTED, ls=":", lw=1)
    ax.text(fault_at / 60 + 0.5, ax.get_ylim()[1] * 0.55, "fault declared", fontsize=9,
            color=aml_style.MUTED)  # fmt: skip
    ax.set(xlabel="minute", ylabel="side street, vehicles waiting", xticks=[0, 15, 30, 45, 60])
    ax.legend(fontsize=9, frameon=False, title="east-west detectors", title_fontsize=9)
    aml_style.save(fig, out_dir, "stuck")


def title_art_figure(out_dir: Path) -> None:
    """Title art: a descending stair of states and one gold arc that lifts the lowest back up.

    Args:
        out_dir: The figure directory.
    """
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0c12", "#141824", "#191520", "#08080b"), grain=0.08)
    for i in range(6):
        x0 = 0.12 + 0.11 * i
        h = top * (0.55 - 0.07 * i)
        c.glaze(c.polygon([(x0, top * 0.12), (x0 + 0.09, top * 0.12), (x0 + 0.09, h), (x0, h)]),
                "#8fb3c9" if i < 4 else "#5a2216", 0.55)  # fmt: skip
    arc = [(0.12 + 0.62 * u, top * (0.32 + 0.5 * np.sin(np.pi * u))) for u in np.linspace(0, 1, 80)]
    c.glow(c.lines([arc[::-1]], 3.0, soft=1.0), "#eaa548", 0.45, spread=8)
    c.glow(c.disc((0.85, top * 0.75), 0.03, soft=3), "#eaa548", 0.5, spread=20)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: six pale-blue blocks step down from left to right, "
        "the last two dark red; a gold arc rises from the lowest block back over the highest, "
        "beside a small gold sun.",
    )


# --------------------------------------------------------------------------- stages: bridges


def bridge_stage(out_dir: Path) -> dict:
    """The bridge MDP: data, value iteration, policies, Q-learning, the Markov test.

    Args:
        out_dir: The figure directory.

    Returns:
        The bridge numbers for ``numbers.json``.
    """
    df = load_records()
    pairs = rating_pairs(df)
    mdp = bridge_mdp(df, pairs)
    P = mdp["P"]
    num: dict[str, object] = {
        "records": len(df),
        "files": sorted(int(y) for y in df["nbi_year"].unique()),
        "gap_months": list(GAP_MONTHS),
        "pairs": mdp["pairs"],
        "pairs_work": mdp["work"],
        "pairs_rose": mdp["rose"],
        "pairs_rose_pct": pct(mdp["rose"] / mdp["no_work"], 1),
        "pairs_used": mdp["used"],
        "new_bridges": mdp["new"],
        "p_stay": {int(r): rnd(P[0][i, i], 3) for i, r in enumerate(RATINGS)},
        "p_new": {int(r): rnd(v, 3) for r, v in zip(RATINGS, mdp["p_new"], strict=True)},
    }
    num["ratings"] = ratings_figure(df, out_dir)
    transitions_figure(mdp, out_dir)
    gammas = {k: gamma_of(r) for k, r in RATES.items()}
    num.update(planning_stage(mdp, gammas, out_dir))
    # Rises with no recorded work counted as no change, instead of left out
    stay = bridge_mdp(df, pairs, rises="stay")
    num["rises_as_stays"] = {
        "critical_penalty": {k: rnd(critical_penalty(stay["P"], g), 3) for k, g in gammas.items()},
        "V_poor_2%": rnd(value_iteration(stay["P"], costs(PENALTY), gammas["2%"])["V"][0], 3),
    }
    # Q-learning on the 2% MDP from Illinois's 2024 ratings
    start = num["ratings"]["by_rating"]
    start = np.array([start[int(r)] for r in RATINGS], float)
    start /= start.sum()
    vi2 = value_iteration(P, costs(PENALTY), gammas["2%"])
    ql = q_learning(P, costs(PENALTY), gammas["2%"], start, vi2["Q"])
    num["q_learning"] = {
        "steps": Q_STEPS,
        "epsilon": Q_EPSILON,
        "log": [
            {**r, "max_error": rnd(r["max_error"], 4)}
            for r in ql["log"]
            if r["steps"] in Q_CHECKPOINTS
        ],
        "last_differs_at": ql["last_differs_at"],
        "differ_steps": {int(r): int(n) for r, n in zip(RATINGS, ql["differ_steps"], strict=True)},
        "fewest_visits": int(ql["visits"].min()),
    }
    qlearning_figure(ql, out_dir)
    num["markov"], num["age_mdp"], num["years_6_to_poor"], num["negative_age"] = markov_stage(
        df, pairs, mdp, gammas["2%"], out_dir
    )
    agency = agency_behaviour(df, pairs)
    num["agency"] = {
        int(r): {"gone_pct": pct(g, 1), "work_pct": pct(w, 1), "n": int(n)}
        for r, g, w, n in zip(RATINGS, agency["gone"], agency["work"], agency["n"], strict=True)
    }
    agency_figure(agency, vi2["policy"], out_dir)
    return num


def planning_stage(mdp: dict, gammas: dict[str, float], out_dir: Path) -> dict:
    """The discount rates, the attempt, value iteration, the critical costs, the policy maps.

    Args:
        mdp: From ``bridge_mdp``.
        gammas: The discount factor per step at each rate.
        out_dir: The figure directory.

    Returns:
        Their numbers for ``numbers.json``.
    """
    P = mdp["P"]
    num: dict[str, object] = {
        "gamma": {k: rnd(g, 4) for k, g in gammas.items()},
        "gamma_powers": [rnd(gammas["2%"] ** k, 2) for k in range(6)],
        "one_minus_gamma": {k: rnd(1 - g, 4) for k, g in gammas.items()},
        "weight_20_years": {k: rnd((1 + r) ** -20, 2) for k, r in RATES.items()},
    }
    # The attempt: a poor bridge, replace now or wait one cycle and then replace, under the MDP's
    # convention (a step that starts poor costs PENALTY, whatever is done about it)
    now = REPLACE_COST + PENALTY
    num["attempt"] = {k: {"now": rnd(now, 3), "wait": rnd(PENALTY + g * now, 3)}
                      for k, g in gammas.items()}  # fmt: skip
    assert now < PENALTY + gammas["2%"] * now and now > PENALTY + gammas["7%"] * now
    vi = {k: value_iteration(P, costs(PENALTY), g) for k, g in gammas.items()}
    num["vi"] = {
        k: {
            "sweeps": len(v["deltas"]),
            "replace_at": [int(r) for r in RATINGS[v["policy"] == 1]],
            "V": {int(r): rnd(x, 3) for r, x in zip(RATINGS, v["V"], strict=True)},
            "Q": {int(r): [rnd(x, 3) for x in q] for r, q in zip(RATINGS, v["Q"], strict=True)},
        }
        for k, v in vi.items()
    }
    # the contraction: each sweep shrinks the largest change by at least gamma
    d = np.array(vi["2%"]["deltas"])
    assert (d[1:] <= gammas["2%"] * d[:-1] + 1e-12).all(), "a sweep did not contract by gamma"
    picture_figures(out_dir, gammas["2%"])
    vi_figure(vi["2%"], gammas["2%"], out_dir)
    crit = {k: critical_penalty(P, g) for k, g in gammas.items()}
    num["critical_penalty"] = {k: rnd(c, 3) for k, c in crit.items()}
    sweep = {
        k: np.array([value_iteration(P, costs(c), g)["policy"] for c in PENALTY_GRID])
        for k, g in gammas.items()
    }
    surfaces = {
        "penalty": PENALTY_GRID,
        "Q": np.array([value_iteration(P, costs(c), gammas["2%"])["Q"] for c in PENALTY_GRID]),
    }
    # Kept although always false: Lecture 27's notebook reads it (its warm-up, question 3)
    num["no_penalty_replaces"] = bool(sweep["2%"][0].any() or sweep["7%"][0].any())
    assert not num["no_penalty_replaces"], "without a poor-condition cost nothing is replaced"
    policy_figure(sweep, crit, out_dir)
    q_surface_figure(surfaces, out_dir)
    return num


def markov_stage(
    df: pd.DataFrame, pairs: pd.DataFrame, mdp: dict, gamma: float, out_dir: Path
) -> tuple[dict, dict, dict, dict]:
    """The Markov test, the age-aware MDP, and the years from rating 6 to poor.

    Args:
        df: The records.
        pairs: The two-year pairs.
        mdp: The rating-only MDP.
        gamma: Discount factor per step.
        out_dir: The figure directory.

    Returns:
        The drop shares by rating and age band; the age-aware policy and the rating-only
        policy's regret; the expected years from rating 6 to poor, pooled and by age band; and
        the pairs and 2024 bridges left out for a negative age.
    """
    P = mdp["P"]
    test = markov_test(pairs)
    markov = {
        int(r): [{"n": x["n"], "pct": pct(x["p"], 1)} for x in rows]
        for r, rows in test["by_rating"].items()
    }
    markov_figure(test, out_dir)
    last = df[df["nbi_year"] == df["nbi_year"].max()]
    age = last["inspection_year"] - last["year_built"]
    negative = {"pairs": test["negative_ages"], "bridges_2024": int((age < 0).sum())}
    last = last[age >= 0].assign(b=age_band(age[age >= 0]))
    start_a = np.zeros(N_S * (len(AGE_BANDS) - 1))
    for (s, b), m in last.groupby(["s", "b"]).size().items():
        start_a[int(b) * N_S + int(s) - RATINGS[0]] += m
    start_a /= start_a.sum()
    aged = age_mdp(mdp, test["counts"], gamma, PENALTY, start_a)
    aged_num = {
        "replace_at_by_band": [[int(r) for r in RATINGS[row == 1]] for row in aged["policy"]],
        "same_policy": bool((aged["policy"] == aged["rating_only"]).all()),
        "regret_pct": pct(aged["regret"], 2),
    }
    i6 = int(np.flatnonzero(RATINGS == 6)[0])
    years = {"rating only": rnd(first_passage_years(P[0])[i6], 1)}
    aged_years = first_passage_aged(aged["P0"], len(AGE_BANDS) - 1)
    for k in range(len(AGE_BANDS) - 1):
        years[f"band{k}"] = rnd(aged_years[k * N_S + i6], 1)
    return markov, aged_num, years, negative


# --------------------------------------------------------------------------- stages: signals


def tune_fixed(base: dict, sat: float) -> dict:
    """Choose the factor on Webster's 15-minute greens on training hours (a bracketing search).

    The search starts from ``TUNE_START``; while the best factor sits at an edge of what was
    tried, the next ladder value past that edge is added. The held-out hours are never used.

    Args:
        base: ``files`` and ``tmp``.
        sat: The saturation flow.

    Returns:
        The chosen factor, each factor's mean delay on the tuning hours, and the hours.
    """
    seeds = [int(x) for x in episode_seeds(0)[:TUNE_HOURS]]
    assert not set(seeds) & set(EVAL_SEEDS)
    tried: dict[float, float] = {}
    grid = list(TUNE_START)
    while grid:
        specs = [
            {**base, "task": "fixed", "name": "tuning", "slug": f"tune_{round(100 * f)}",
             "plan": webster_plans(sat, factor=f)["15 min"], "seed": e, "routes": "demand",
             "train_seed": 0}
            for f in grid
            for e in seeds
        ]  # fmt: skip
        res = run_jobs(specs)
        for k, f in enumerate(grid):
            tried[f] = float(
                np.mean([r["delay_s"] for r in res[k * len(seeds) : (k + 1) * len(seeds)]])
            )
        best = min(tried, key=tried.__getitem__)
        i = TUNE_LADDER.index(best)
        if best == min(tried) and i > 0:
            grid = [TUNE_LADDER[i - 1]]
        elif best == max(tried) and i < len(TUNE_LADDER) - 1:
            grid = [TUNE_LADDER[i + 1]]
        else:
            grid = []
    assert min(tried) < best < max(tried), f"the chosen factor {best} is not interior: {tried}"
    return {"factor": best, "tried": dict(sorted(tried.items())), "seeds": seeds}


def evaluation_specs(base: dict, plans: dict, trained: list[dict]) -> list[dict]:
    """Every held-out evaluation: each controller on both demands, and the dead detectors.

    Args:
        base: ``files`` and ``tmp``.
        plans: Webster's plans for the hour and per 15 minutes, the tuned plan, and the plan
            retimed for the reversed demand.
        trained: The training runs.

    Returns:
        The jobs' specs.
    """
    specs = []
    for routes in ROUTES:
        fixed = [*FIXED, "Webster, 15 min, retimed"] if routes == "reversed" else FIXED
        for name in fixed:
            specs += [eval_spec(base, name, e, task="fixed", plan=plans[PLAN_KEY[name]],
                                routes=routes) for e in EVAL_SEEDS]  # fmt: skip
        specs += [eval_spec(base, "Actuated", e, task="actuated", routes=routes)
                  for e in EVAL_SEEDS]  # fmt: skip
        specs += [
            eval_spec(base, KIND_LABEL[t["kind"]], e, task="eval", kind=t["kind"],
                      state=t["state"], train_seed=t["seed"], stuck=(), fallback=None,
                      routes=routes)
            for t in trained
            for e in EVAL_SEEDS
        ]  # fmt: skip
    specs += [
        eval_spec(base, name, e, task="eval", kind="dqn", state=t["state"], train_seed=t["seed"],
                  stuck=STUCK, fallback=fb)
        for name, fb in (("dead", None), ("dead, fallback", plans["hour"]))
        for t in trained
        if t["kind"] == "dqn"
        for e in EVAL_SEEDS
    ]  # fmt: skip
    return specs


def runs_of(results: list[dict], name: str, routes: str = "demand") -> list[dict]:
    """One controller's evaluations on one demand.

    Args:
        results: Every evaluation.
        name: The controller's label.
        routes: The demand.

    Returns:
        Its runs, in submission order.
    """
    rs = [r for r in results if r["name"] == name and r["routes"] == routes]
    assert rs, f"no runs of {name} on {routes}"
    return rs


def by_run(rs: list[dict], key: str = "delay_s") -> np.ndarray:
    """Per training run (one value for an untrained controller), the mean over held-out hours.

    Args:
        rs: One controller's runs.
        key: The quantity.

    Returns:
        One mean per training seed, in seed order.
    """
    seeds = sorted({r["train_seed"] for r in rs})
    return np.array([np.mean([r[key] for r in rs if r["train_seed"] == s]) for s in seeds])


def paired(rs: list[dict], base: list[dict]) -> dict:
    """A trained controller against an untrained one, hour by hour (the same demand draws).

    Args:
        rs: The trained controller's runs.
        base: The untrained controller's runs.

    Returns:
        Per training run and in all, the hours on which the trained one's delay was lower,
        and the mean difference in seconds (negative: the trained one is better).
    """
    ref = {r["seed"]: r["delay_s"] for r in base}
    seeds = sorted({r["train_seed"] for r in rs})
    diff = {s: np.array([r["delay_s"] - ref[r["seed"]] for r in rs if r["train_seed"] == s])
            for s in seeds}  # fmt: skip
    every = np.concatenate(list(diff.values()))
    return {
        "better_by_run": [int((d < 0).sum()) for d in diff.values()],
        "mean_diff_s_by_run": [rnd(d.mean(), 2) for d in diff.values()],
        "better": int((every < 0).sum()),
        "of": len(every),
        "mean_diff_s": rnd(every.mean(), 2),
    }


def unfinished(rs: list[dict]) -> int:
    """Vehicles still waiting or inside at the cap, summed over runs.

    Args:
        rs: Runs.

    Returns:
        The count.
    """
    return int(sum(r["pending"] + r["inside"] for r in rs))


def summarize(results: list[dict], trained: list[dict], out_dir: Path) -> dict:
    """The held-out comparison: delays, hour-by-hour pairs, the reversed demand, the figures.

    Args:
        results: Every evaluation.
        trained: The training runs.
        out_dir: The figure directory.

    Returns:
        The signal numbers of the comparison for ``numbers.json``.
    """
    delay = {name: by_run(runs_of(results, name)) for name in CONTROLLERS}
    for name in ("Webster, hour", "Webster, 15 min", "Fixed, tuned", "Actuated"):
        assert unfinished(runs_of(results, name)) == 0, f"{name} did not clear"
    num: dict[str, object] = {
        "delay_s": {k: [rnd(x, 1) for x in v] for k, v in delay.items()},
        "delay_mean_s": {k: rnd(v.mean(), 1) for k, v in delay.items()},
        "unfinished": {k: unfinished(runs_of(results, k)) for k in CONTROLLERS},
        "paired": {
            kind: {
                base: paired(runs_of(results, kind), runs_of(results, base))
                for base in ("Webster, 15 min", "Fixed, tuned", "Actuated")
            }
            for kind in KIND_LABEL.values()
        },
    }
    names = [*CONTROLLERS[:2], "Webster, 15 min, retimed", *CONTROLLERS[2:]]
    rev = {name: by_run(runs_of(results, name, "reversed")) for name in names}
    num["reversed"] = {
        "delay_s": {k: [rnd(x, 1) for x in v] for k, v in rev.items()},
        "delay_mean_s": {k: rnd(v.mean(), 1) for k, v in rev.items()},
        "unfinished": {k: unfinished(runs_of(results, k, "reversed")) for k in names},
    }
    num["train_seconds"] = {
        label: int(round(np.mean([t["seconds"] for t in trained if t["kind"] == kind])))
        for kind, label in KIND_LABEL.items()
    }
    curves = {label: [t["curve"] for t in trained if t["kind"] == kind]
              for kind, label in KIND_LABEL.items()}  # fmt: skip
    num["last20_vehicle_hours"] = {
        k: rnd(np.mean([np.mean(c[-20:]) for c in v]) / 3600, 2) for k, v in curves.items()
    }
    halted = {name: float(np.mean([r["halted"] for r in runs_of(results, name)]))
              for name in ("Webster, hour", "Webster, 15 min", "Actuated")}  # fmt: skip
    curves_figure(curves, halted, out_dir)
    results_figure(delay, out_dir)
    share = {name: np.mean([green_share(r["_lights"]) for r in runs_of(results, name)], axis=0)
             for name in ("Webster, hour", "Webster, 15 min", "Actuated", "DQN")}  # fmt: skip
    num["green_share_ns_pct"] = {k: [pct(x, 0) for x in v] for k, v in share.items()}
    adapt_figure(share, out_dir)
    return num


def stuck_stage(results: list[dict], out_dir: Path) -> dict:
    """Summarize the dead side-street detectors: working, dead, dead with the fallback.

    Args:
        results: Every evaluation; the dead-detector cases are named "dead" and "dead,
            fallback", the DQN with working detectors "DQN".
        out_dir: The figure directory.

    Returns:
        Per case: mean delay (in all and per training run), vehicles unfinished at the cap,
        the side street's vehicles and how many got through per run, and when the fault was
        declared.
    """
    cases = {"working": "DQN", "dead": "dead", "dead, fallback": "dead, fallback"}
    out = {}
    for case, name in cases.items():
        rs = runs_of(results, name)
        seeds = sorted({r["train_seed"] for r in rs})

        def total(key: str, rs: list[dict] = rs, seeds: list[int] = seeds) -> list[int]:
            return [int(sum(r[key] for r in rs if r["train_seed"] == s)) for s in seeds]

        out[case] = {
            "delay_s": rnd(np.mean([r["delay_s"] for r in rs]), 1),
            "delay_s_by_run": [rnd(x, 1) for x in by_run(rs)],
            "unfinished": unfinished(rs),
            "unfinished_per_hour": int(round(unfinished(rs) / len(rs))),
            "side_vehicles_by_run": total("side_vehicles"),
            "side_served_by_run": total("side_finished"),
            "fault_at_s": sorted({r["fault_at_s"] for r in rs if r["fault_at_s"] is not None}),
            "runs": len(rs),
        }
    first = {case: next(r for r in runs_of(results, name)
                        if r["train_seed"] == TRAIN_SEEDS[0] and r["seed"] == EVAL_SEEDS[0])
             for case, name in cases.items()}  # fmt: skip
    out["dead, fallback"]["fault_at_s_drawn"] = first["dead, fallback"]["fault_at_s"]
    stuck_figure({c: r["_side_queue"] for c, r in first.items()},
                 first["dead, fallback"]["fault_at_s"], out_dir)  # fmt: skip
    return out


def busiest_decision(files: dict[str, Path], act: Controller) -> int:
    """The end of the decision with the most stopped vehicles in the first held-out hour.

    Args:
        files: From ``build_network``.
        act: The controller.

    Returns:
        The simulated second at which that decision ended.
    """
    with Intersection(files, EVAL_SEEDS[0]) as env:
        obs, hour_over, best, best_t = env.obs(), False, -1, 0
        while not hour_over:
            obs, _, hour_over = env.step(act(obs, env))
            stopped = sum(env.ts.edge.getLastStepHaltingNumber(a) for a in APPROACHES)
            if stopped > best:
                best, best_t = stopped, int(env.ts.simulation.getTime())
    return best_t


def snapshot(files: dict[str, Path], state: dict[str, np.ndarray], out_dir: Path) -> dict:
    """Draw the intersection under the first DQN at the busiest moment of the first held-out hour.

    The run is deterministic, so a second run stops at the same moment as the first saw it.

    Args:
        files: From ``build_network``.
        state: The first DQN's weights.
        out_dir: The figure directory.

    Returns:
        The moment, the vehicles in the network, on the approaches, and in the drawing, and the
        stopped vehicles on the approaches.
    """
    act = greedy("dqn", to_torch(state))
    at_s = busiest_decision(files, act)
    with Intersection(files, EVAL_SEEDS[0]) as env:
        obs = env.obs()
        while env.ts.simulation.getTime() < at_s:
            obs, _, _ = env.step(act(obs, env))
        assert int(env.ts.simulation.getTime()) == at_s, (
            "the rerun did not stop where the first did"
        )
        veh = env.ts.vehicle
        ids = veh.getIDList()
        cx, cy = env.ts.junction.getPosition("C")  # netconvert moves the origin to a corner
        snap = {
            "t": int(env.ts.simulation.getTime()),
            "phase": env.phase,
            "xy": [tuple(np.subtract(veh.getPosition(v), (cx, cy))) for v in ids],
            "halted": [veh.getSpeed(v) < 0.1 for v in ids],
            "approach": [veh.getRoadID(v) in APPROACHES for v in ids],
        }
    intersection_figure(snap, out_dir)
    on = np.array(snap["approach"], bool)
    halted = np.array(snap["halted"], bool)
    assert not (halted & ~on).any(), "a vehicle stopped off the approaches"
    return {
        "t": snap["t"],
        "minute": snap["t"] // 60,
        "in_network": len(ids),
        "on_approaches": int(on.sum()),
        "halted": int((halted & on).sum()),
        "drawn": int(sum(max(abs(x), abs(y)) <= VIEW_M for x, y in snap["xy"])),
    }


def signal_stage(out_dir: Path) -> dict:
    """The intersection: the baselines, DQN, online DQN, PPO, and the dead-detector test.

    Args:
        out_dir: The figure directory.

    Returns:
        The signal numbers for ``numbers.json``.
    """
    files = build_network(out_dir / "sumo")
    tmp = out_dir / "sumo" / "trips"
    tmp.mkdir(parents=True, exist_ok=True)
    base = {"files": files, "tmp": tmp}
    static, actuated = tl_program(files["net"]), tl_program(files["net_actuated"])
    assert static["states"] == actuated["states"] and actuated["type"] == "actuated"
    greens = [GREEN_PHASE[NS], GREEN_PHASE[EW]]
    sat = saturation_flow(files)
    s = sat["veh_per_h"]
    plans = webster_plans(s)
    assert webster_plans(s, REVERSED)["hour"] == plans["hour"], "the hour's mean flows differ"
    q_ns = float(np.mean([d[2] for d in DEMAND]))
    q_ew = float(np.mean([d[3] for d in DEMAND]))
    w = webster(q_ns, q_ew, s)
    num: dict[str, object] = {
        "saturation": {"queue": sat["queue"], "headway_s": rnd(sat["headway_s"], 2),
                       "veh_per_h": int(round(s))},
        "demand": [list(d) for d in DEMAND],
        "reversed_demand": [list(d) for d in REVERSED],
        "webster_hour": {"Y": rnd(w["Y"], 3), "cycle_s": rnd(w["cycle_s"], 1)},
        "webster_15": [{"Y": rnd(x["Y"], 3), "cycle_s": rnd(x["cycle_s"], 1)}
                       for x in (webster(ns, ew, s) for _, _, ns, ew in DEMAND)],
        "actuated": {"min_green_s": [actuated["min_dur"][i] for i in greens],
                     "max_green_s": [actuated["max_dur"][i] for i in greens]},
    }  # fmt: skip
    demand_figure(plans, s, out_dir)
    trained = run_jobs([{**base, "task": "train", "kind": kind, "seed": seed}
                        for kind in KIND_LABEL for seed in TRAIN_SEEDS])  # fmt: skip
    tuning = tune_fixed(base, s)
    plans["tuned"] = webster_plans(s, factor=tuning["factor"])["15 min"]
    plans["retimed"] = webster_plans(s, REVERSED)["15 min"]
    num["plans"] = {k: [list(p) for p in v] for k, v in plans.items()}
    num["tuning"] = {
        "hours": len(tuning["seeds"]),
        "factor": tuning["factor"],
        "delay_s": {f"{f:.1f}": rnd(d, 2) for f, d in tuning["tried"].items()},
    }
    results = run_jobs(evaluation_specs(base, plans, trained))
    assert not any(tmp.iterdir()), "a trip file was left behind"
    tmp.rmdir()
    num.update(summarize(results, trained, out_dir))
    num["stuck"] = stuck_stage(results, out_dir)
    ppo_clip_figure(out_dir)
    dqn0 = next(t for t in trained if t["kind"] == "dqn" and t["seed"] == TRAIN_SEEDS[0])
    num["snapshot"] = snapshot(files, dqn0["state"], out_dir)
    num["settings"] = {
        "episodes": EPISODES,
        "decisions_per_hour": HOUR_S // DECISION_S,
        "train_seeds": len(TRAIN_SEEDS),
        "eval_hours": len(EVAL_SEEDS),
        "gamma": GAMMA_RL,
        "horizon_minutes": rnd(DECISION_S / (1 - GAMMA_RL) / 60, 1),
        "buffer": DQN_BUFFER,
        "batch": DQN_BATCH,
        "target_every": DQN_TARGET_EVERY,
        "eps_decisions": EPS_DECISIONS,
        "clip": PPO_CLIP,
        "fault_s": FAULT_S,
    }
    return num


def main() -> None:
    """Run both stages, draw the remaining figures, and write ``numbers.json``."""
    out_dir = parse_out_dir((__doc__ or "").splitlines()[0])
    seed_all(SEED)
    aml_style.apply()
    t0 = time.perf_counter()
    numbers: dict[str, object] = {"bridge": bridge_stage(out_dir)}
    numbers["bridge_seconds"] = rnd(time.perf_counter() - t0, 0)
    numbers["signal"] = signal_stage(out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("policy_map", "q_learning", "results", "stuck"))
    numbers["settings"] = {
        "seed": SEED,
        "penalty": PENALTY,
        "rates": RATES,
        "years_per_step": YEARS_PER_STEP,
        "q_steps": Q_STEPS,
        "workers": WORKERS,
        "sumo": importlib.metadata.version("libsumo"),
        "eclipse_sumo": importlib.metadata.version("eclipse-sumo"),
        "torch": torch.__version__,
        "threads_per_run": 1,
        "cpu_count": os.cpu_count(),
    }
    numbers["minutes"] = rnd((time.perf_counter() - t0) / 60, 1)
    write_numbers(numbers, out_dir)


if __name__ == "__main__":
    main()
