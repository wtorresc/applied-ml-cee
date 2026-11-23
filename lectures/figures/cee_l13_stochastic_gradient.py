"""Figures and numbers for CEE Lecture 13: stochastic gradient descent.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l13_stochastic_gradient.py \
        --out build/fig/cee_l13_stochastic_gradient

Writes the PNG figures, ``numbers.json`` (the data numbers the slides quote; constants such as
the batch size and the step grids live in this module), and ``numbers.tex`` (the first SGD step,
for the TikZ diagram). Data: ``scripts/fetch_nbi_us.sh`` (FHWA National Bridge Inventory, all
states, 2018-2024; U.S. Government work).

The setting: will a bridge deck lose at least one rating point by its next inspection? A
logistic regression with 108 weights is fitted to 864,669 national transitions (2018 to 2020
and 2020 to 2022) and tested on the next cycle, 434,373 transitions from 2022 to 2024. Three
2018 records carry sentinel values (an ADT of 999,999 and two deck widths of 999.9 m) and are
dropped on loading; decks wider than 200 m are otherwise used as published.
Newton's method gives the exact optimum (Lecture 6). Gradient descent, SGD, SGD with averaged
iterates, momentum, and Adam are written from scratch, checked against scikit-learn and
PyTorch, and compared with the optimum in passes over the data. The break-it test streams the
records in FHWA's published order (state by state) instead of shuffled.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import aml_style
import title_art
from common import (
    DATA_DIR,
    parse_out_dir,
    require,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

CSV: Final[Path] = DATA_DIR / "derived" / "nbi_us_deck_transitions.csv.gz"
N_TRANSITIONS: Final = 1_299_045  # rows in the file, before the sentinel rows are dropped
# Sentinel values that pass the fetch script's numeric checks: one ADT of 999,999
# vehicles a day and two deck widths of 999.9 m, all in 2018 pairs. These rows are dropped; the
# (state, structure number) of each is asserted, so a new release cannot drop others silently.
SENTINEL_ADT: Final = 999_999.0
SENTINEL_WIDTH_M: Final = 999.9
SENTINEL_ROWS: Final = frozenset(
    {("39", "2504235"), ("27", "82527"), ("51", "000000000003802")}
)  # Ohio (ADT), Minnesota and Virginia (deck width)
WIDE_DECK_M: Final = 200.0  # decks wider than this are kept as published: a stated limitation
CYCLE_MONTHS: Final = 24  # the NBI's routine inspection interval
SEED = 0
# Every optimizer's step size is chosen on seeds 0-4 and its results are reported on seeds 5-9, so
# no reported number is the best of the draws that chose it. Single runs use seed 5.
SELECT_SEEDS = range(5)
REPORT_SEEDS = range(5, 10)
REPORT_SEED = REPORT_SEEDS[0]
BATCH = 256  # rows per SGD step; 3,377 steps make one pass
TEST_FROM = 2022  # transitions 2022 -> 2024 are the test cycle
ILLINOIS = "17"
GOOD = 8  # the attempt's input: a deck rated 8 or 9
# Passes over the training data at which a run's weights are kept (log-spaced, for the plot).
CHECKS: tuple[float, ...] = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 5, 10)
LONG = 10  # passes in the long runs
GD_PASSES: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
# Gradient descent's step sizes, in units of 1/L. Near the optimum it is stable below
# 2 / lambda_max(H*), about 5.8/L here, so the grid is finer there (q4 review, L13-D4).
GD_STEPS: tuple[float, ...] = (1.0, 2.0, 4.0, 5.0, 6.0, 8.0)
TRACE_POINTS = 40  # checkpoints within the one file-order pass
BETA = 0.9  # momentum
ADAM_BETAS = (0.9, 0.999)  # Kingma & Ba (2015) defaults, as PyTorch's
ADAM_EPS = 1e-8
# Step sizes swept for each method; the one with the smallest one-pass gap is used.
STEPS: Final[dict[str, tuple[float, ...]]] = {
    "sgd": (0.1, 0.3, 1.0, 3.0, 10.0),
    "average": (0.1, 0.3, 1.0, 3.0, 10.0),
    "momentum": (0.03, 0.1, 0.3, 1.0),
    "adam": (0.001, 0.003, 0.01, 0.03),
}
# Step sizes on inputs in raw units; each sweep must find its best step inside the grid.
RAW_STEPS: tuple[float, ...] = (1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5)
RAW_ADAM: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
TORCH_STEPS = 200  # steps compared with torch.optim
NOISE_BAND_PASSES: tuple[float, float] = (1.0, 10.0)  # passes over which the noise ball is read
LABELS: Final[dict[str, str]] = {
    "gd": "gradient descent",
    "sgd": "SGD",
    "average": "SGD, averaged",
    "adam": "Adam",
}
# FIPS state code (NBI Item 1) to postal abbreviation, for the figures' state ticks.
STATE_ABBR: Final[dict[str, str]] = {
    "01": "AL", "06": "CA", "17": "IL", "23": "ME", "49": "UT", "72": "PR",
}  # fmt: skip

# Inputs: numbers (standardized on the training rows) and coded items (one-hot, the most common
# training level as reference). Rare codes are pooled with "other" (Item 107: 7 aluminum into 9;
# Item 108C: 4 cathodic protection into 9; Item 43A: 8 masonry, 9 aluminum into 0).
NUMERIC: tuple[str, ...] = (
    "age", "age2", "since_work", "log_adt", "pct_truck", "truck_missing", "log_length",
    "deck_width", "months",
)  # fmt: skip
CODED: tuple[str, ...] = (
    "deck_now", "deck_type", "surface", "membrane", "protection", "kind", "owner_group", "state",
)  # fmt: skip
POOL: Final[dict[str, tuple[frozenset[str], str]]] = {
    "deck_type": (frozenset({"7"}), "9"),
    "protection": (frozenset({"4"}), "9"),
    "kind": (frozenset({"8", "9"}), "0"),
}


# ---------------------------------------------------------------------------------------------
# Data and model


def drop_sentinels(t: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows whose ADT or deck width is a sentinel value, not a measurement.

    Args:
        t: The transitions as read from ``CSV``.

    Returns:
        ``t`` without those rows, the index reset (positions stay in published order).

    Raises:
        ValueError: If the rows dropped are not exactly ``SENTINEL_ROWS``.
    """
    bad = (t["adt"] >= SENTINEL_ADT) | (t["deck_width_m"] >= SENTINEL_WIDTH_M)
    found = set(zip(t.loc[bad, "state"], t.loc[bad, "structure"], strict=True))
    if found != SENTINEL_ROWS or int(bad.sum()) != len(SENTINEL_ROWS):
        raise ValueError(f"sentinel rows changed: {sorted(found)} ({int(bad.sum())} rows)")
    return t[~bad].reset_index(drop=True)


def load() -> pd.DataFrame:
    """The national deck transitions, sentinel rows dropped, coded items pooled and grouped.

    Returns:
        One row per transition, in FHWA's published order (file, state, structure number).

    Raises:
        ValueError: If the file's row count, its sentinel rows, or its codes have changed.
    """
    codes = ["state", "structure", "facility", "crosses", "owner", "kind", "deck_type",
             "surface", "membrane", "protection"]  # fmt: skip
    t = pd.read_csv(require(CSV, "fetch_nbi_us.sh"), dtype=dict.fromkeys(codes, str))
    if len(t) != N_TRANSITIONS:
        raise ValueError(f"{len(t)} rows, expected {N_TRANSITIONS}")
    t = drop_sentinels(t)
    for c, (small, to) in POOL.items():
        t[c] = t[c].where(~t[c].isin(small), to)
    if not pd.api.types.is_integer_dtype(t["deck"]):  # a float rating would give "8.0" below
        raise ValueError(f"deck ratings read as {t['deck'].dtype}")
    t["deck_now"] = t["deck"].clip(3, 9).astype(str)  # ratings 0-3 are rare: one level
    own = pd.to_numeric(t["owner"], errors="raise")  # every owner code is a number: no "other"
    t["owner_group"] = np.select([own == 1, own.isin([2, 3, 4])], ["state", "local"], "other")
    return t


def numeric_inputs(t: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    """The numeric inputs, in the model's units or in the inventory's own units.

    Args:
        t: Transitions from ``load``.
        raw: Years, vehicles per day (linear), percent, meters, and months, unscaled.

    Returns:
        One column per name in ``NUMERIC``.
    """
    age = t["nbi_year"] - t["year_built"]
    since = t["nbi_year"] - np.maximum(t["year_built"], t["year_reconstructed"])
    trucks = t["pct_truck"].fillna(0.0)
    if raw:
        cols = {"age": age, "age2": age**2, "since_work": since, "log_adt": t["adt"],
                "pct_truck": trucks, "log_length": t["length_m"],
                "deck_width": t["deck_width_m"], "months": t["months"]}  # fmt: skip
    else:
        cols = {"age": age / 10, "age2": (age / 10) ** 2, "since_work": since / 10,
                "log_adt": np.log10(t["adt"] + 1), "pct_truck": trucks / 10,
                "log_length": np.log10(t["length_m"]), "deck_width": t["deck_width_m"] / 10,
                "months": t["months"] / 12}  # fmt: skip
    cols["truck_missing"] = t["pct_truck"].isna().astype(float)
    return pd.DataFrame(cols)[list(NUMERIC)].astype(float)


def design(
    t: pd.DataFrame, train: np.ndarray, rows: np.ndarray, raw: bool = False, scale: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Inputs matrix with an intercept column: numeric inputs, then one-hot codes.

    The matrix is allocated once, for ``rows`` only, and filled column by column: the same
    values as stacking the blocks for every row and then selecting, at a third of the memory.

    Args:
        t: Transitions from ``load``.
        train: Boolean mask of the training rows (scaling and reference levels come from them).
        rows: Boolean mask of the rows to build (the training rows or the test rows).
        raw: Numeric inputs in the inventory's units (see ``numeric_inputs``).
        scale: Standardize the numeric inputs with the training mean and standard deviation.

    Returns:
        ``A`` of shape (rows.sum(), D) and the D column names.

    Raises:
        ValueError: If a numeric input is constant on the training rows.
    """
    Z = numeric_inputs(t, raw).to_numpy()
    if scale:
        sd = Z[train].std(0)
        if not (sd > 0).all():
            raise ValueError(f"constant inputs: {np.asarray(NUMERIC)[sd == 0].tolist()}")
        Z = (Z - Z[train].mean(0)) / sd
    levels = {}
    for c in CODED:
        counts = t.loc[train, c].value_counts()
        levels[c] = sorted(set(counts.index) - {counts.index[0]})
    names = ["intercept", *NUMERIC, *(f"{c}={lv}" for c in CODED for lv in levels[c])]
    A = np.empty((int(rows.sum()), len(names)))
    A[:, 0] = 1.0
    A[:, 1 : 1 + len(NUMERIC)] = Z[rows]
    j = 1 + len(NUMERIC)
    for c in CODED:
        col = t.loc[rows, c].to_numpy()
        for lv in levels[c]:
            A[:, j] = col == lv
            j += 1
    if j != len(names):
        raise ValueError(f"filled {j} columns of {len(names)}")
    return A, names


def loss(w: np.ndarray, A: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of a logistic model, computed stably (log1p of exp, via logaddexp).

    Args:
        w: Weights, shape (D,).
        A: Inputs with an intercept column, shape (n, D).
        y: Labels in {0, 1}, shape (n,).

    Returns:
        The mean loss in nats.
    """
    z = A @ w
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def gradient(w: np.ndarray, A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of ``loss``: the inputs weighted by the errors, A^T (p - y) / n.

    Args:
        w: Weights, shape (D,).
        A: Inputs with an intercept column, shape (n, D).
        y: Labels in {0, 1}, shape (n,).

    Returns:
        The gradient, shape (D,).
    """
    return A.T @ (expit(A @ w) - y) / len(y)


def hessian(w: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Hessian of ``loss``: A^T S A / n with S = diag(p (1 - p)).

    Args:
        w: Weights, shape (D,).
        A: Inputs with an intercept column, shape (n, D).

    Returns:
        The Hessian, shape (D, D).
    """
    p = expit(A @ w)
    return (A * (p * (1 - p))[:, None]).T @ A / len(A)


def newton(
    A: np.ndarray, y: np.ndarray, tol: float = 1e-10, max_steps: int = 50
) -> tuple[np.ndarray, list[float]]:
    """Newton's method for the logistic loss (Lecture 6), from w = 0.

    Args:
        A: Inputs with an intercept column, shape (n, D).
        y: Labels in {0, 1}, shape (n,).
        tol: Stop when no weight moves by more than this.
        max_steps: Give up after this many steps.

    Returns:
        The optimum and the loss before each step and after the last.

    Raises:
        RuntimeError: If the steps have not converged after ``max_steps``.
    """
    w = np.zeros(A.shape[1])
    losses = [loss(w, A, y)]
    for _ in range(max_steps):
        delta = np.linalg.solve(hessian(w, A), gradient(w, A, y))
        w = w - delta
        losses.append(loss(w, A, y))
        if np.abs(delta).max() < tol:
            return w, losses
    raise RuntimeError(f"Newton did not converge in {max_steps} steps")


# ---------------------------------------------------------------------------------------------
# Optimizers, from scratch


@dataclass(frozen=True)
class Run:
    """One stochastic run: the update rule, its step size, and how the rows are visited."""

    kind: str  # "sgd", "momentum", or "adam"
    lr: float
    average: bool = False  # return the running mean of the iterates (Polyak-Ruppert)
    order: str = "shuffle"  # "shuffle" each pass, or "file": FHWA's published order
    seed: int = SEED

    @property
    def method(self) -> str:
        """Key into ``STEPS`` and ``LABELS``.

        Returns:
            "average" for averaged SGD, else the update rule.
        """
        return "average" if self.average else self.kind


def fit(
    A: np.ndarray, y: np.ndarray, run: Run, passes: float, checks: Iterable[float]
) -> dict[float, np.ndarray]:
    """Minibatch stochastic optimization of ``loss``, keeping the weights at checkpoints.

    Each pass visits ``N // BATCH`` batches of ``BATCH`` rows (the last ``N % BATCH`` rows of
    the visiting order are skipped), shuffled anew each pass or in file order.

    Args:
        A: Inputs with an intercept column, shape (N, D).
        y: Labels in {0, 1}, shape (N,).
        run: The update rule and step size.
        passes: How many passes over the rows.
        checks: Passes (fractions allowed) at which to keep the weights.

    Returns:
        Pass -> weights (the running mean if ``run.average``).
    """
    rng = np.random.default_rng(run.seed)
    N, D = A.shape
    per_pass = N // BATCH
    total = round(passes * per_pass)
    at = {round(c * per_pass): c for c in checks if c <= passes}
    w, m, v, w_bar = (np.zeros(D) for _ in range(4))
    b1, b2 = ADAM_BETAS
    kept: dict[float, np.ndarray] = {}
    for s in range(total):
        j = s % per_pass
        if j == 0:
            visit = rng.permutation(N) if run.order == "shuffle" else np.arange(N)
        rows = visit[j * BATCH : (j + 1) * BATCH]
        g = gradient(w, A[rows], y[rows])
        if run.kind == "sgd":
            w -= run.lr * g
        elif run.kind == "momentum":
            m = BETA * m + g  # heavy ball, as torch.optim.SGD(momentum=0.9)
            w -= run.lr * m
        elif run.kind == "adam":
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat, v_hat = m / (1 - b1 ** (s + 1)), v / (1 - b2 ** (s + 1))
            w -= run.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        else:
            raise ValueError(f"unknown update rule {run.kind!r}")
        if run.average:
            w_bar += (w - w_bar) / (s + 1)
        if s + 1 in at:
            kept[at[s + 1]] = (w_bar if run.average else w).copy()
    return kept


def gradient_descent(
    A: np.ndarray, y: np.ndarray, lr: float, passes: Sequence[int]
) -> dict[int, np.ndarray]:
    """Full-batch gradient descent from w = 0; one step is one pass over the rows.

    Args:
        A: Inputs with an intercept column.
        y: Labels in {0, 1}.
        lr: Step size.
        passes: Steps at which to keep the weights.

    Returns:
        Step -> weights.
    """
    w = np.zeros(A.shape[1])
    kept = {}
    for s in range(1, max(passes) + 1):
        w = w - lr * gradient(w, A, y)
        if s in passes:
            kept[s] = w.copy()
    return kept


# ---------------------------------------------------------------------------------------------
# Checks against finite differences, scikit-learn, and PyTorch


def check_gradient(A: np.ndarray, y: np.ndarray, w_star: np.ndarray) -> float:
    """Largest relative error of ``gradient`` against central differences of ``loss``.

    Checked at a point away from the optimum (w* plus noise), on five coordinates.

    Args:
        A: Training inputs with an intercept column.
        y: Training labels.
        w_star: The exact optimum.

    Returns:
        max |analytic - numeric| / max |analytic| over the coordinates checked.
    """
    rng = np.random.default_rng(SEED)
    w = w_star + rng.normal(0, 0.1, w_star.shape)
    g = gradient(w, A, y)
    h, errs = 1e-5, []
    for i in rng.choice(len(w), 5, replace=False):
        e = np.zeros_like(w)
        e[i] = h
        errs.append(abs((loss(w + e, A, y) - loss(w - e, A, y)) / (2 * h) - g[i]))
    return float(max(errs) / np.abs(g).max())


def check_sklearn(A: np.ndarray, y: np.ndarray, w_star: np.ndarray) -> float:
    """Largest weight difference between our Newton optimum and scikit-learn's.

    scikit-learn's ``newton-cholesky`` solver, unpenalized (``C=np.inf``), on the same inputs.

    Args:
        A: Training inputs with an intercept column (scikit-learn fits its own intercept).
        y: Training labels.
        w_star: Our optimum.

    Returns:
        The largest absolute difference over the weights.
    """
    sk = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=100)
    with warnings.catch_warnings():
        # scikit-learn 1.8 asks for C=np.inf instead of the deprecated penalty=None, then warns
        # that the resulting penalty=None ignores C. The fit is unpenalized, as intended.
        warnings.filterwarnings("ignore", "Setting penalty=None", UserWarning)
        sk.fit(A[:, 1:], y)
    return float(np.abs(np.r_[sk.intercept_, sk.coef_.ravel()] - w_star).max())


def check_torch(A: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Our gradient, momentum, and Adam against PyTorch's autograd and ``torch.optim``.

    Both sides take ``TORCH_STEPS`` steps on the same shuffled batches, in float64.

    Args:
        A: Training inputs with an intercept column.
        y: Training labels.

    Returns:
        Largest absolute difference in the first batch's gradient, and in the weights after
        the steps, for momentum and for Adam.
    """
    torch.manual_seed(SEED)
    out: dict[str, float] = {}
    for run in (Run("momentum", 0.1), Run("adam", 0.003)):
        ours = fit(A, y, run, TORCH_STEPS / (len(A) // BATCH), [TORCH_STEPS / (len(A) // BATCH)])
        (w_ours,) = ours.values()
        rng = np.random.default_rng(run.seed)
        visit = rng.permutation(len(A))  # the same rows fit() visits in its first pass
        p = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)
        opt = (
            torch.optim.SGD([p], lr=run.lr, momentum=BETA)
            if run.kind == "momentum"
            else torch.optim.Adam([p], lr=run.lr, betas=ADAM_BETAS, eps=ADAM_EPS)
        )
        for s in range(TORCH_STEPS):
            rows = visit[s * BATCH : (s + 1) * BATCH]
            Ab, yb = torch.from_numpy(A[rows]), torch.from_numpy(y[rows])
            opt.zero_grad()
            z = Ab @ p
            torch.mean(torch.nn.functional.softplus(z) - yb * z).backward()
            if s == 0:
                g0 = gradient(np.zeros(A.shape[1]), A[rows], y[rows])
                out["gradient"] = float(np.abs(p.grad.numpy() - g0).max())
            opt.step()
        out[run.kind] = float(np.abs(p.detach().numpy() - w_ours).max())
    return out


# ---------------------------------------------------------------------------------------------
# The setting, the attempt, and the first step


def setting(t: pd.DataFrame, train: np.ndarray, names: list[str]) -> dict[str, object]:
    """Counts for the setting slide.

    Args:
        t: Transitions from ``load`` (sentinel rows already dropped).
        train: Boolean mask of the training rows.
        names: The design's column names.

    Returns:
        Counts, drop shares, and the data limitations the notes state.
    """
    bridges = t.groupby(["state", "structure"]).ngroups
    rows = t[train].groupby("state").size()
    adt = t.loc[train, "adt"]
    return {
        "transitions": len(t),
        "sentinels_dropped": len(SENTINEL_ROWS),
        "train": int(train.sum()),
        "test": int((~train).sum()),
        "bridges": bridges,
        "states": t["state"].nunique(),
        "weights": len(names),
        "drop_train": float(t.loc[train, "drop"].mean()),
        "drop_test": float(t.loc[~train, "drop"].mean()),
        "drop_by_cycle": {
            int(k): float(v) for k, v in t.groupby("nbi_year")["drop"].mean().items()
        },
        "steps_per_pass": int(train.sum()) // BATCH,
        "months_24": float((t.loc[train, "months"] == CYCLE_MONTHS).mean()),
        "drop_by_state": {
            str(k): float(v) for k, v in t[train].groupby("state")["drop"].mean().items()
        },
        "rows_by_state": {str(k): int(v) for k, v in rows.items()},
        "smallest_states": [str(k) for k in rows.nsmallest(5).index],
        "smallest_share": float(rows.min() / train.sum()),
        "wide_decks": {  # wider than WIDE_DECK_M, kept as published
            "train": int((t.loc[train, "deck_width_m"] > WIDE_DECK_M).sum()),
            "all": int((t["deck_width_m"] > WIDE_DECK_M).sum()),
        },
        "adt_max": float(adt.max()),
        "adt_p99": float(adt.quantile(0.99)),
    }


def drop_by_rating_figure(t: pd.DataFrame, train: np.ndarray, out_dir: Path) -> dict:
    """Bar chart: share of decks that dropped, by the rating at the first inspection.

    Args:
        t: Transitions from ``load``.
        train: Boolean mask of the training rows.
        out_dir: The figure directory.

    Returns:
        Rating -> the share that dropped and the number of training pairs.
    """
    g = t[train].groupby("deck_now")["drop"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    x = np.arange(len(g))
    ax.bar(x, 100 * g["mean"], color=aml_style.PALETTE[0], width=0.7)
    for xi, (share, n) in zip(x, g.itertuples(index=False), strict=True):
        ax.text(xi, 100 * share + 0.6, f"{100 * share:.0f}%", ha="center", fontsize=12)
        ax.text(xi, -3.2, f"{n / 1000:.0f}k", ha="center", fontsize=11, color=aml_style.MUTED)
    # The row label goes at the right end: at the left it ran into the y axis's 0.
    ax.text(x[-1] + 0.75, -3.2, "n", ha="center", fontsize=11, color=aml_style.MUTED,
            style="italic")  # fmt: skip
    ax.set_xticks(x, ["≤3" if r == "3" else r for r in g.index])
    ax.set_xlabel("deck rating at the first inspection")
    ax.set_ylabel("decks that dropped (%)")
    ax.set_ylim(-5, 100 * g["mean"].max() + 6)
    ax.axhline(0, color=aml_style.MUTED, lw=0.8)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    aml_style.save(fig, out_dir, "drop_by_rating")
    return {r: {"share": float(s), "n": int(n)} for r, (s, n) in g.iterrows()}


def attempt(t: pd.DataFrame) -> dict:
    """Four Illinois inspections, 2018 -> 2020, and one SGD step by hand.

    Input x = 1 if the deck was rated 8 or 9; the label y = 1 if it dropped. From w = 0 every
    p is 0.5. Batch 1 = rows A, B; batch 2 = rows C, D; step size 1.

    Args:
        t: Transitions from ``load``.

    Returns:
        The four rows, the two batches' gradients and the full one, and the weights after
        each step.
    """
    il = t[(t["state"] == ILLINOIS) & (t["nbi_year"] == 2018)]
    il = il[il["facility"].notna() & il["crosses"].notna()]
    good = il["deck"] >= GOOD
    picks = {
        "A": il[good & (il["drop"] == 1)].iloc[0],
        "B": il[~good & (il["drop"] == 0)].iloc[0],
        "C": il[good & (il["drop"] == 0)].iloc[0],
        "D": il[~good & (il["drop"] == 0)].iloc[1],
    }
    rows = {}
    for k, r in picks.items():
        rows[k] = {
            "name": f"{r['facility'].title()} over {r['crosses'].title()}",
            "deck": int(r["deck"]),
            "deck_next": int(r["deck_next"]),
            "x": int(r["deck"] >= GOOD),
            "y": int(r["drop"]),
        }

    def step(keys: str) -> list[float]:
        g = np.mean([(0.5 - rows[k]["y"]) * np.array([1.0, rows[k]["x"]]) for k in keys], 0)
        return [float(v) for v in g]

    g1, g2, g_all = step("AB"), step("CD"), step("ABCD")
    assert np.allclose(np.add(g1, g2) / 2, g_all)
    steps = {"w1": [-v for v in g1], "w2": [-v for v in g2], "w_all": [-v for v in g_all]}
    return {"rows": rows, "g1": g1, "g2": g2, "g_all": g_all} | steps


def first_step(A: np.ndarray, y: np.ndarray, run: Run) -> dict[str, float]:
    """The first step of ``run``, for the diagram's inset: from w = 0, p = 0.5 everywhere.

    Args:
        A: Training inputs with an intercept column.
        y: Training labels.
        run: The plain SGD run whose first batch is shown.

    Returns:
        The batch's drops, the intercept's gradient and new value, and the training loss
        before, after, and with only the intercept moved.
    """
    rows = np.random.default_rng(run.seed).permutation(len(A))[:BATCH]
    g = gradient(np.zeros(A.shape[1]), A[rows], y[rows])
    w1 = -run.lr * g
    w_int = np.zeros_like(w1)
    w_int[0] = w1[0]
    return {
        "drops": int(y[rows].sum()),
        "g0": float(g[0]),
        "w0": float(w1[0]),
        "loss_before": loss(np.zeros(A.shape[1]), A, y),
        "loss_after": loss(w1, A, y),
        "loss_intercept_only": loss(w_int, A, y),
    }


def choose_step(scores: Mapping[float, float], what: str) -> float:
    """The step with the smallest score, which must lie strictly inside its grid.

    A search whose best value sits at a grid edge has not found it.

    Args:
        scores: Step -> score (a gap; ``inf`` for a diverged run), over the whole grid.
        what: The search's name, for the error message.

    Returns:
        The best step.

    Raises:
        ValueError: If the best step is the smallest or the largest of the grid.
    """
    grid = sorted(scores)
    best = min(grid, key=lambda k: scores[k])
    if best in (grid[0], grid[-1]):
        raise ValueError(f"{what}: best step {best:g} at the edge of {grid}")
    return best


# ---------------------------------------------------------------------------------------------
# Experiments


@dataclass
class Fits:
    """The training data, the exact optimum, and the noise floors every gap is compared to."""

    A: np.ndarray
    y: np.ndarray
    w_star: np.ndarray
    F_star: float
    A_il: np.ndarray  # Illinois's test-cycle inputs, for its forecast

    def gap(self, w: np.ndarray) -> float:
        """Optimization error: training loss above the exact optimum.

        Args:
            w: Weights.

        Returns:
            The gap in nats.
        """
        return loss(w, self.A, self.y) - self.F_star

    def illinois(self, w: np.ndarray) -> float:
        """Expected number of Illinois decks that drop in the test cycle.

        Args:
            w: Weights.

        Returns:
            The sum of Illinois's test-cycle probabilities.
        """
        return float(expit(self.A_il @ w).sum())


def one_pass(fits: Fits, run: Run) -> tuple[float, float]:
    """Gap and Illinois forecast after one pass of ``run``.

    Args:
        fits: The training data and the exact optimum.
        run: The update rule, step size, and seed.

    Returns:
        The training loss above the optimum and Illinois's expected drops.
    """
    (w,) = fit(fits.A, fits.y, run, 1, [1]).values()
    return fits.gap(w), fits.illinois(w)


def sweep(fits: Fits, method: str, seeds: Iterable[int]) -> dict[float, list[tuple[float, float]]]:
    """One-pass (gap, Illinois forecast) over the step sizes in ``STEPS[method]``, per seed.

    Args:
        fits: The training data and the exact optimum.
        method: A key of ``STEPS``.
        seeds: The seeds to run.

    Returns:
        Step -> one (gap, forecast) pair per seed.
    """
    kind = "sgd" if method == "average" else method
    return {
        lr: [one_pass(fits, Run(kind, lr, average=method == "average", seed=s)) for s in seeds]
        for lr in STEPS[method]
    }


def best_runs(
    fits: Fits,
) -> tuple[dict[str, Run], dict[str, dict[str, object]], dict[str, dict[float, list[float]]]]:
    """Each method's step size: smallest median one-pass gap on ``SELECT_SEEDS``.

    Every method gets the same protocol: its whole grid of steps on the selection seeds, then
    the chosen step alone on ``REPORT_SEEDS``, which give every number reported.

    Args:
        fits: The training data and the exact optimum.

    Returns:
        The chosen run per method (seeded for the single runs); the table: the sweep's median
        per step (selection seeds), the chosen step, and the reported gaps and Illinois
        forecasts (report seeds); and each method's one-pass gaps per step and selection seed.

    Raises:
        ValueError: If a method's best step is at the edge of its grid.
    """
    sweeps = {method: sweep(fits, method, SELECT_SEEDS) for method in STEPS}
    runs, table = {}, {}
    for method, sw in sweeps.items():
        med = {lr: float(np.median([g for g, _ in r])) for lr, r in sw.items()}
        lr = choose_step(med, method)
        run = Run(
            "sgd" if method == "average" else method, lr, method == "average", seed=REPORT_SEED
        )
        runs[method] = run
        gaps, il = zip(*[one_pass(fits, replace(run, seed=s)) for s in REPORT_SEEDS], strict=True)
        table[method] = {
            "steps": {str(k): v for k, v in med.items()},
            "best": lr,
            "median": float(np.median(gaps)),
            "lo": float(min(gaps)),
            "hi": float(max(gaps)),
            "illinois": [float(v) for v in il],
            "illinois_median": float(np.median(il)),
        }
    gaps = {m: {lr: [g for g, _ in r] for lr, r in sw.items()} for m, sw in sweeps.items()}
    return runs, table, gaps


def long_runs(
    fits: Fits, runs: dict[str, Run], L: float
) -> tuple[dict[str, dict[float, float]], dict[float, np.ndarray], dict]:
    """Gaps along ``LONG`` passes for each method, and gradient descent at its best step.

    Gradient descent is tuned like the stochastic methods: each step in ``GD_STEPS`` (units of
    1/L, L the smoothness bound), the one with the least loss after its last pass kept.

    Args:
        fits: The training data and the exact optimum.
        runs: Each stochastic method's chosen run.
        L: The smoothness bound.

    Returns:
        Method -> pass -> gap, averaged SGD's weights at each checkpoint, and gradient descent's
        gap after ``max(GD_PASSES)`` passes at each step with the step chosen.

    Raises:
        ValueError: If gradient descent's best step is at the edge of ``GD_STEPS``.
    """
    curves, kept_average = {}, {}
    for method, run in runs.items():
        kept = fit(fits.A, fits.y, run, LONG, CHECKS)
        curves[method] = {c: fits.gap(w) for c, w in kept.items()}
        if method == "average":
            kept_average = kept
    last = max(GD_PASSES)
    with np.errstate(over="ignore", invalid="ignore"):  # a step too large may diverge
        gd = {k: gradient_descent(fits.A, fits.y, k / L, GD_PASSES) for k in GD_STEPS}
        ends = {k: fits.gap(w[last]) for k, w in gd.items()}
    ends = {k: (v if np.isfinite(v) else np.inf) for k, v in ends.items()}
    best = choose_step(ends, "gradient descent")
    curves["gd"] = {s: fits.gap(w) for s, w in gd[best].items()}
    finite = {f"{k:g}": (v if np.isfinite(v) else None) for k, v in ends.items()}
    gd_table = {"steps": finite, "best": best}
    return curves, kept_average, gd_table


def convergence_figure(
    curves: dict[str, dict[float, float]],
    newton_gaps: list[float],
    fl: dict,
    gd_step: float,
    out_dir: Path,
) -> None:
    """Log-log plot: optimization error against passes over the training data.

    Args:
        curves: Method -> pass -> gap, gradient descent's included.
        newton_gaps: Newton's gap before each step and after the last.
        fl: The bracket from ``floors``.
        gd_step: Gradient descent's chosen step, in units of 1/L.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    styles = {
        "gd": {"color": aml_style.MUTED, "ls": "-", "marker": "o"},
        "sgd": {"color": aml_style.PALETTE[0], "ls": "--", "marker": None},
        "average": {"color": aml_style.PALETTE[1], "ls": "-", "marker": None},
        "adam": {"color": aml_style.PALETTE[2], "ls": "-.", "marker": None},
    }
    for method, st in styles.items():
        c = curves[method]
        label = f"{LABELS[method]}, {gd_step:g}/L" if method == "gd" else LABELS[method]
        ax.plot(list(c), list(c.values()), label=label, ms=5, **st)
    shown = [(i, g) for i, g in enumerate(newton_gaps) if g > 1e-6 and i > 0]
    ax.plot([i for i, _ in shown], [g for _, g in shown], "kD", ms=6, label="Newton, per step")
    ax.axhline(fl["base_gap"], color=aml_style.INK, ls=":", lw=1.4)
    ax.text(0.3, fl["base_gap"] * 1.25, "base rate only", ha="center", fontsize=12)
    ax.axhline(fl["estimation"], color=aml_style.INK, ls=":", lw=1.4)
    ax.text(1 / 70, fl["estimation"] * 1.25, "estimation floor", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1 / 80, 130)
    ax.set_ylim(3e-5, 0.5)
    ax.set_xticks([1 / 64, 1 / 8, 1, 10, 100], ["1/64", "1/8", "1", "10", "100"])
    ax.set_xlabel("passes over the training inspections")
    ax.set_ylabel("loss above the\noptimum (nats)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=12)
    aml_style.save(fig, out_dir, "convergence")


def step_size_figure(gaps: dict[str, dict[float, list[float]]], out_dir: Path) -> None:
    """One-pass gap against the step size, SGD and averaged SGD: median and range of seeds 0-4.

    Args:
        gaps: Method -> step -> one-pass gaps on the selection seeds (from ``best_runs``).
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, method in enumerate(("sgd", "average")):
        sw = gaps[method]
        lrs = list(sw)
        med = [np.median(sw[k]) for k in lrs]
        ax.fill_between(lrs, [min(sw[k]) for k in lrs], [max(sw[k]) for k in lrs],
                        color=aml_style.PALETTE[i], alpha=0.2, lw=0)  # fmt: skip
        ax.plot(lrs, med, color=aml_style.PALETTE[i], ls=aml_style.LINESTYLES[i],
                marker=aml_style.MARKERS[i], label=LABELS[method])  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(STEPS["sgd"], [f"{v:g}" for v in STEPS["sgd"]])
    ax.minorticks_off()
    ax.set_xlabel(r"step size $\eta$")
    ax.set_ylabel("loss above the optimum\nafter one pass (nats)")
    ax.legend(loc="upper left", fontsize=12)
    aml_style.save(fig, out_dir, "step_size")


def floors(fits: Fits, H: np.ndarray, cov: np.ndarray) -> dict:
    """The bracket for every gap: the base-rate model above, estimation error below.

    The estimation floor is the expected excess loss of the exact fit itself, (1/2) tr(H Cov)
    with Cov the bridge-clustered sandwich covariance; for independent rows and a correct model
    Cov = H^-1 / N and it reduces to D / 2N, also reported.

    Args:
        fits: The training data and the exact optimum.
        H: The Hessian at the optimum.
        cov: The clustered sandwich covariance.

    Returns:
        The base-rate model's gap, the clustered and independent estimation floors, and F*.
    """
    ybar = fits.y.mean()
    base = -(ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))
    N, D = fits.A.shape
    return {
        "base_gap": float(base - fits.F_star),
        "estimation": float(0.5 * np.trace(H @ cov)),
        "estimation_iid": D / (2 * N),
        "F_star": fits.F_star,
    }


def clustered_covariance(fits: Fits, bridge: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Sandwich covariance of the optimum, scores summed by bridge (two transitions each).

    Args:
        fits: The training data and the optimum.
        bridge: Each training row's bridge (state and structure number).
        H: The Hessian of the mean loss at the optimum.

    Returns:
        H^-1 G H^-1 / N, G the mean outer product of the bridges' summed scores.
    """
    A, y, w = fits.A, fits.y, fits.w_star
    H_inv = np.linalg.inv(H)
    scores = A * (expit(A @ w) - y)[:, None]
    by_bridge = pd.DataFrame(scores).groupby(bridge, sort=False).sum().to_numpy()
    G = by_bridge.T @ by_bridge / len(A)
    return H_inv @ G @ H_inv / len(A)


def forecasts(
    w: np.ndarray, A_te: np.ndarray, state_te: np.ndarray
) -> tuple[dict[str, float], pd.Series]:
    """Test-cycle forecasts of one fit: the national share and Illinois's expected drops.

    Args:
        w: The weights.
        A_te: Test-cycle inputs.
        state_te: Each test row's state code.

    Returns:
        The share of test decks expected to drop and Illinois's expected number of drops; and
        each state's expected share (a Series indexed by state code, in file order).
    """
    p = expit(A_te @ w)
    out = {"share": float(p.mean()), "illinois": float(p[state_te == ILLINOIS].sum())}
    return out, pd.Series(p).groupby(state_te).mean()


def one_pass_gap(fits: Fits, run: Run) -> float:
    """Gap after one pass of ``run``; a diverged run counts as the worst (``inf``), not missing.

    Args:
        fits: The training data and the exact optimum.
        run: The update rule, step size, and seed.

    Returns:
        The training loss above the optimum, or ``inf``.
    """
    g = fits.gap(fit(fits.A, fits.y, run, 1, [1])[1])
    return g if np.isfinite(g) else np.inf


def raw_units(t: pd.DataFrame, train: np.ndarray, runs: dict[str, Run]) -> dict[str, object]:
    """The same model with ADT linear, in raw units and standardized: one-pass gaps.

    Raw units: SGD and Adam each sweep their grid on ``SELECT_SEEDS`` (the best step must lie
    inside it), then report the median gap of ``REPORT_SEEDS``; raw SGD's chosen step is also
    reported seed by seed (selection and report seeds): its gaps scatter by an order of
    magnitude.
    Scaled: the main model's steps (no search), reported on the same seeds.

    Args:
        t: Transitions from ``load``.
        train: Boolean mask of the training rows.
        runs: The main model's chosen runs (the scaled row's steps).

    Returns:
        Per unit system: the optimum's loss, the gap at the start, each method's step, gaps,
        and sweep, the largest input, and ADT's weight.

    Raises:
        ValueError: If a raw-unit search's best step is at the edge of its grid.
    """
    y = t.loc[train, "drop"].to_numpy(float)
    out: dict[str, object] = {}
    for label, scale in (("raw", False), ("scaled", True)):
        A, _ = design(t, train, train, raw=True, scale=scale)
        w_star, _ = newton(A, y)
        fits = Fits(A, y, w_star, loss(w_star, A, y), A[:0])  # no forecast needed
        grids = {"sgd": RAW_STEPS, "adam": RAW_ADAM}
        if scale:
            grids = {"sgd": (runs["sgd"].lr,), "adam": (runs["adam"].lr,)}
        res: dict[str, object] = {}
        with np.errstate(over="ignore", invalid="ignore"):  # large steps overflow exp()
            for kind, grid in grids.items():
                per_seed = {
                    lr: [one_pass_gap(fits, Run(kind, lr, seed=s)) for s in SELECT_SEEDS]
                    for lr in grid
                }
                med = {lr: float(np.median(v)) for lr, v in per_seed.items()}
                lr = choose_step(med, f"{label} {kind}") if len(grid) > 1 else grid[0]
                rest = [one_pass_gap(fits, Run(kind, lr, seed=s)) for s in REPORT_SEEDS]
                res[kind] = float(np.median(rest))
                res[f"{kind}_lr"] = lr
                res[f"{kind}_steps"] = {
                    f"{k:g}": (v if np.isfinite(v) else None) for k, v in med.items()
                }
                if not scale:
                    seeds = per_seed[lr] + rest  # selection then report seeds: 0-9
                    res[f"{kind}_seeds"] = [(v if np.isfinite(v) else None) for v in seeds]
        out[label] = {
            "F_star": fits.F_star,
            "start": float(np.log(2) - fits.F_star),  # the gap at w = 0
            **res,
            "max_input": float(np.abs(A).max()),
            "adt_weight": float(w_star[NUMERIC.index("log_adt") + 1]),  # ADT, linear here
        }
        del A, fits
    raw, scaled = out["raw"], out["scaled"]
    seeds = np.array([np.inf if v is None else v for v in raw["sgd_seeds"]])
    out["sgd_raw_above_start"] = int((seeds > raw["start"]).sum())
    out["adam_raw_over_scaled"] = raw["adam"] / scaled["adam"]
    return out


def worst(errors: pd.Series, k: int = 3) -> list[dict]:
    """The ``k`` states with the largest forecast errors (points), largest first.

    Args:
        errors: Forecast error per state code, in points.
        k: How many states.

    Returns:
        One record per state: its code and its error.
    """
    top = errors.abs().sort_values(ascending=False).index[:k]
    return [{"state": str(s), "pp": float(errors[s])} for s in top]


def file_order(
    fits: Fits,
    runs: dict[str, Run],
    state_tr: np.ndarray,
    *,
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    exact_by_state: pd.Series,
    drop_by_state: Mapping[str, float],
    out_dir: Path,
) -> tuple[dict[str, object], pd.Series]:
    """The break-it test: one pass in FHWA's published order against one shuffled pass.

    The published-order step is chosen on the very pass it reports (it is deterministic),
    while the shuffled step was chosen on other seeds: the comparison favours published order.

    Args:
        fits: The training data and the exact optimum.
        runs: The shuffled methods' chosen runs.
        state_tr: Each training row's state code, in published order.
        test: Test-cycle inputs, labels, and state codes.
        exact_by_state: Newton's expected share of drops per state (test cycle).
        drop_by_state: Each state's training drop share.
        out_dir: The figure directory.

    Returns:
        The numbers for the slides, and each state's forecast error after the shuffled
        averaged pass (points, Newton's subtracted).

    Raises:
        ValueError: If a published-order best step is at the edge of its grid.
    """
    A_te, _, state_te = test
    out: dict[str, object] = {}
    for method in ("sgd", "average"):
        kind = "sgd"
        gaps = {}
        for lr in STEPS[method]:
            (w,) = fit(fits.A, fits.y, Run(kind, lr, method == "average", "file"), 1, [1]).values()
            gaps[lr] = fits.gap(w)
        best = choose_step(gaps, f"published order, {method}")
        out[method] = {
            "best": best,
            "gap": gaps[best],
            "steps": {str(k): v for k, v in gaps.items()},
        }
    # Trajectories at each order's best step, averaged SGD.
    marks = [k / TRACE_POINTS for k in range(1, TRACE_POINTS + 1)]
    shuffled = fit(fits.A, fits.y, runs["average"], 1, marks)
    in_file = fit(fits.A, fits.y, Run("sgd", out["average"]["best"], True, "file"), 1, marks)
    per_pass = len(fits.A) // BATCH
    trace = {
        "pass": marks,
        "shuffled": [fits.gap(shuffled[c]) for c in marks],
        "file": [fits.gap(in_file[c]) for c in marks],
        # The state of the last row of the last full batch at each mark: ``fit`` skips the
        # ``N % BATCH`` tail, so the step count times BATCH is the rows visited (capped at N).
        "state": [
            str(state_tr[min(round(c * per_pass) * BATCH, len(state_tr)) - 1]) for c in marks
        ],
    }
    f_shuf, by_shuf = forecasts(shuffled[1], A_te, state_te)
    f_file, by_file = forecasts(in_file[1], A_te, state_te)
    err_shuf = 100 * (by_shuf - exact_by_state)
    err_file = 100 * (by_file - exact_by_state)
    # What the published-order errors follow (q4 review, L13-D2): each state's training drop
    # share minus the national one, or the state's place in the file (codes sort as the file).
    order = list(err_file.index)
    national = float(fits.y.mean())
    excess = np.array([drop_by_state[s] - national for s in order])
    position = np.arange(len(order), dtype=float)
    half = len(order) // 2
    out |= {
        "trace": trace,
        "shuffled": f_shuf,
        "file": f_file,
        "worst_file": worst(err_file),
        "worst_shuffled": worst(err_shuf),
        "mean_abs_file": float(err_file.abs().mean()),
        "most_above_file": float(err_file.max()),
        "mean_abs_shuffled": float(err_shuf.abs().mean()),
        "corr_drop_rate": float(np.corrcoef(err_file[order], excess)[0, 1]),
        "corr_drop_rate_shuffled": float(np.corrcoef(err_shuf[order], excess)[0, 1]),
        "corr_position": float(np.corrcoef(err_file[order], position)[0, 1]),
        "corr_abs_position": float(np.corrcoef(err_file[order].abs(), position)[0, 1]),
        "mean_abs_first_half": float(err_file[order[:half]].abs().mean()),
        "mean_abs_last_half": float(err_file[order[half:]].abs().mean()),
    }
    order_figure(trace, err_shuf, err_file, out_dir)
    return out, err_shuf


def order_figure(trace: dict, err_shuf: pd.Series, err_file: pd.Series, out_dir: Path) -> None:
    """Two panels: the gap along the pass, and each state's forecast error after it.

    Args:
        trace: The gaps along the pass, from ``file_order``.
        err_shuf: Each state's forecast error after the shuffled pass (points).
        err_file: The same after the published-order pass.
        out_dir: The figure directory.
    """
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=aml_style.FULL, gridspec_kw={"width_ratios": [1, 1.3]}
    )
    shuf = {"color": aml_style.PALETTE[0], "label": "shuffled"}
    pub = {"color": aml_style.PALETTE[1], "label": "published order"}
    x = np.array(trace["pass"])
    ax.plot(x, trace["shuffled"], ls="--", **shuf)
    ax.plot(x, trace["file"], ls="-", **pub)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.3)
    ax.set_xlabel("fraction of the pass")
    ax.set_ylabel("loss above the optimum (nats)")
    ax.axvline(0.5, color=aml_style.MUTED, lw=0.8, ls=":")
    for x0, year in ((0.25, 2018), (0.75, 2020)):
        ax.text(x0, 0.25, f"{year} file\nAL to PR", ha="center", va="top", fontsize=11)
    order = list(err_file.index)  # state codes, in file order
    xs = np.arange(len(order))
    bx.bar(xs - 0.2, err_shuf[order], width=0.4, **shuf)
    bx.bar(xs + 0.2, err_file[order], width=0.4, **pub)
    bx.axhline(0, color=aml_style.MUTED, lw=0.8)
    show = [i for i, s in enumerate(order) if s in STATE_ABBR]
    bx.set_xticks(show, [STATE_ABBR[order[i]] for i in show])
    bx.set_xlabel("state, in file order")
    bx.set_ylabel("state forecast minus\nNewton's (points)")
    handles, labels = bx.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    aml_style.save(fig, out_dir, "file_order")


def coefficients(
    fits: Fits,
    names: list[str],
    cov: np.ndarray,
    kept: dict[float, np.ndarray],
    *,
    flattest: list[str],
    err_shuffled: pd.Series,
) -> dict[str, object]:
    """How far one and ten passes of averaged SGD leave each weight, in standard errors.

    Args:
        fits: The training data and the exact optimum.
        names: The design's column names.
        cov: The clustered sandwich covariance of the optimum.
        kept: Averaged SGD's weights by pass.
        flattest: The names with the largest components of the Hessian's flattest direction.
        err_shuffled: Each state's forecast error after the shuffled averaged pass (points).

    Returns:
        Summary z's after 1 and ``LONG`` passes, the epoxy weight, and for each state of the
        flattest direction: its weight's z, rank among the state weights by |z| (1 = furthest
        from Newton's), distance moved from the start, Newton's value, and its forecast error
        and rank among the states by |error| (1 = worst).
    """
    se = np.sqrt(np.diag(cov))
    out = {}
    for p in (1, LONG):
        w = kept[p]
        z = (w - fits.w_star) / se
        j_max = int(np.argmax(np.abs(z)))
        out[str(p)] = {"median_z": float(np.median(np.abs(z))), "max_z": float(np.abs(z[j_max])),
                       "worst": names[j_max], "gap": fits.gap(w),
                       "share_over_2": float(np.mean(np.abs(z) > 2))}  # fmt: skip
    i = names.index("protection=1")  # epoxy-coated reinforcing (Item 108C = 1)
    w1 = kept[1]
    out["epoxy"] = {"exact": float(fits.w_star[i]), "se": float(se[i]), "one_pass": float(w1[i]),
                    "z": float((w1[i] - fits.w_star[i]) / se[i]),
                    "odds": float(np.exp(fits.w_star[i]))}  # fmt: skip
    st = [j for j, n in enumerate(names) if n.startswith("state=")]
    out["states_median_z"] = float(np.median(np.abs((w1[st] - fits.w_star[st]) / se[st])))
    other = [j for j in range(len(names)) if j not in st and j != 0]
    out["others_median_z"] = float(np.median(np.abs((w1[other] - fits.w_star[other]) / se[other])))
    # The flattest direction's states (q4 review, L13-D1): where their weights and forecasts sit.
    z_st = pd.Series(np.abs((w1[st] - fits.w_star[st]) / se[st]), [names[j] for j in st])
    z_rank = z_st.rank(ascending=False, method="min")
    err_rank = err_shuffled.abs().rank(ascending=False, method="min")
    flat = {}
    for n in flattest:
        if not n.startswith("state="):
            continue
        j, code = names.index(n), n.removeprefix("state=")
        flat[code] = {
            "z": float(z_st[n]),
            "z_rank": int(z_rank[n]),
            "moved": float(abs(w1[j])),
            "exact": float(fits.w_star[j]),
            "err": float(err_shuffled[code]),
            "err_rank": int(err_rank[code]),
        }
    out["flat_states"] = flat
    out["state_weights"] = len(st)
    return out


def budget(
    fits: Fits,
    cov: np.ndarray,
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    w_average: np.ndarray,
    names: list[str],
    sgd_forecasts: Sequence[float],
    file_forecast: float,
) -> dict[str, object]:
    """Error budget for Illinois's 2022 -> 2024 forecast: what each source of error is worth.

    Args:
        fits: The training data and the exact optimum.
        cov: The clustered sandwich covariance of the optimum.
        test: Test-cycle inputs, labels, and state codes.
        w_average: Averaged SGD's weights after one shuffled pass (report seed).
        names: The design's column names.
        sgd_forecasts: Illinois forecasts of one plain SGD pass, report seeds.
        file_forecast: Illinois forecast of one averaged pass in published order.

    Returns:
        Newton's forecast, the outcome, its standard error and the chance spread, the miss in
        those units, the averaged pass's shortfall, and that shortfall restored weight by
        weight: Illinois's weight alone, the intercept alone, and both.
    """
    A_te, y_te, state_te = test
    il = state_te == ILLINOIS
    p = expit(A_te[il] @ fits.w_star)
    g = (A_te[il] * (p * (1 - p))[:, None]).sum(0)  # d(sum p)/dw
    exact, actual = float(p.sum()), float(y_te[il].sum())
    se, sd = float(np.sqrt(g @ cov @ g)), float(np.sqrt((p * (1 - p)).sum()))
    one_pass = fits.illinois(w_average)
    restored = {}
    il_col = names.index(f"state={ILLINOIS}")
    for key, cols in (("illinois_weight", [il_col]), ("intercept", [0]), ("both", [0, il_col])):
        w = w_average.copy()
        w[cols] = fits.w_star[cols]
        restored[key] = fits.illinois(w)
    return {
        "exact": exact,
        "actual": actual,
        "inspections": int(il.sum()),
        "estimation_se": se,
        "outcome_sd": sd,
        "miss": exact - actual,
        "miss_over_chance": (exact - actual) / sd,
        "miss_over_both": (exact - actual) / float(np.hypot(se, sd)),
        "one_pass": one_pass,
        "one_pass_short": exact - one_pass,
        "one_pass_short_se": (exact - one_pass) / se,
        "restored": restored,
        "sgd_max_over_exact": max(sgd_forecasts) / exact - 1,
        "file": file_forecast,
    }


# ---------------------------------------------------------------------------------------------
# Two-weight pictures: the attempt's model on every training row, where the loss is exact


PIC_STEPS: int = 300  # SGD steps drawn in the step-size picture
BALL_STEPS: int = 1000  # SGD steps in the noise-ball picture (256,000 rows, under one pass)
PIC_LRS: tuple[float, ...] = (0.1, 3.0, 30.0)  # too small, about right, too large
BALL_LR: float = 3.0
ARROWS: int = 30  # batch gradients drawn at each point
NEAR_OFFSET: tuple[float, float] = (0.12, -0.18)  # the near point, from the optimum
BALL_ZOOM: float = 0.35  # half-width of the noise-ball picture around the optimum
RAVINE_STEPS: int = 30
RAVINE_W1_TICKS: dict[str, list[float]] = {"decades": [-0.4, 0.0], "scaled": [-1.0, 0.0, 1.0]}
RAVINE_ZOOM: dict[str, float] = {"decades": 0.75, "scaled": 0.85}
RAVINE_VIEW: dict[str, dict[str, float]] = {
    "decades": {"elev": 35.0, "azim": -150.0},  # chosen from six angles: trough and crawl show
    "scaled": {"elev": 35.0, "azim": -150.0},
}
RAVINE_SAFE: float = 1.9  # each ravine run steps 1.9 / L, L the largest curvature at w = 0
RAVINE_CAP: float = 0.75  # nats: the ravine surfaces are cut flat above this, so the trough reads


@dataclass(frozen=True)
class Pair:
    """A two-weight logistic model, p = sigmoid(w0 + w1 v), on the training rows.

    Attributes:
        v: The input of every training row, shape (N,).
        y: Labels in {0, 1}, shape (N,).
        levels: The distinct input values, shape (K,).
        n0: Rows with y = 0 at each level, shape (K,).
        n1: Rows with y = 1 at each level, shape (K,).
    """

    v: np.ndarray
    y: np.ndarray
    levels: np.ndarray
    n0: np.ndarray
    n1: np.ndarray

    @classmethod
    def of(cls, v: np.ndarray, y: np.ndarray) -> Pair:
        """Count the labels at each distinct input value.

        Args:
            v: The input of every training row.
            y: Labels in {0, 1}.

        Returns:
            The model.
        """
        levels, inv = np.unique(v, return_inverse=True)
        n1 = np.bincount(inv, weights=y, minlength=len(levels))
        n0 = np.bincount(inv, minlength=len(levels)) - n1
        return cls(v, y, levels, n0, n1)

    def loss(self, W0: np.ndarray, W1: np.ndarray) -> np.ndarray:
        """Exact mean cross-entropy at every (w0, w1), from the counts at each input level.

        Args:
            W0: Intercepts, any shape.
            W1: Slopes, the same shape.

        Returns:
            The loss at each (w0, w1), that shape.
        """
        z = np.asarray(W0)[..., None] + np.asarray(W1)[..., None] * self.levels  # (..., K)
        out = (self.n0 + self.n1) * np.logaddexp(0.0, z) - self.n1 * z
        return out.sum(-1) / len(self.y)

    def design(self) -> np.ndarray:
        """Inputs with an intercept column.

        Returns:
            Shape (N, 2).
        """
        return np.column_stack([np.ones_like(self.v), self.v])  # (N, 2)

    def optimum(self) -> np.ndarray:
        """The exact minimizer, by Newton's method.

        Returns:
            (w0, w1).
        """
        w, _ = newton(self.design(), self.y)
        return w

    def kappa(self, w: np.ndarray) -> float:
        """Condition number of the Hessian at ``w``.

        Args:
            w: Weights.

        Returns:
            Its largest eigenvalue over its smallest.
        """
        ev = np.linalg.eigvalsh(hessian(w, self.design()))
        return float(ev.max() / ev.min())


def sgd_path(pair: Pair, lr: float, steps: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """SGD from w = 0 on shuffled batches of ``BATCH`` rows; the iterates and their running mean.

    Args:
        pair: The two-weight model.
        lr: Step size.
        steps: Number of steps (they must fit in one pass).
        seed: Seed of the shuffle.

    Returns:
        Iterates and averaged iterates, each of shape (steps + 1, 2), starting at w = 0.
    """
    A = pair.design()
    rows = np.random.default_rng(seed).permutation(len(A))[: steps * BATCH]
    assert len(rows) == steps * BATCH, "the path must fit in one pass"
    w, w_bar = np.zeros(2), np.zeros(2)
    path, avg = [w.copy()], [w_bar.copy()]
    for s in range(steps):
        r = rows[s * BATCH : (s + 1) * BATCH]
        w = w - lr * gradient(w, A[r], pair.y[r])
        w_bar = w_bar + (w - w_bar) / (s + 1)
        path.append(w.copy())
        avg.append(w_bar.copy())
    return np.array(path), np.array(avg)


def gd_path(pair: Pair, lr: float, steps: int) -> np.ndarray:
    """Full-batch gradient descent from w = 0.

    Args:
        pair: The two-weight model.
        lr: Step size.
        steps: Number of steps.

    Returns:
        The iterates, shape (steps + 1, 2).
    """
    A, w = pair.design(), np.zeros(2)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * gradient(w, A, pair.y)
        path.append(w.copy())
    return np.array(path)


def two_weight_pairs(t: pd.DataFrame, train: np.ndarray) -> dict[str, Pair]:
    """The attempt's model (x = rated 8 or 9), and deck age two ways: decades from 0, scaled.

    Args:
        t: Transitions from ``load``.
        train: Boolean mask of the training rows.

    Returns:
        Name -> the two-weight model on every training row.
    """
    tr = t[train]
    y = tr["drop"].to_numpy(float)
    age = (tr["nbi_year"] - tr["year_built"]).to_numpy(float)
    return {
        "good": Pair.of((tr["deck"] >= GOOD).to_numpy(float), y),
        "decades": Pair.of(age / 10, y),
        "scaled": Pair.of((age - age.mean()) / age.std(), y),
    }


def _contour(
    ax: plt.Axes, pair: Pair, box: tuple[float, float, float, float], n: int = 121
) -> None:
    """Filled loss contours over ``box`` = (w0_lo, w0_hi, w1_lo, w1_hi).

    Args:
        ax: The axes to draw on.
        pair: The two-weight model.
        box: The weights' ranges.
        n: Grid points per side.
    """
    W0, W1 = np.meshgrid(np.linspace(box[0], box[1], n), np.linspace(box[2], box[3], n))
    F = pair.loss(W0, W1)
    levels = np.quantile(F, np.linspace(0, 1, 16) ** 2)
    ax.contourf(W0, W1, F, levels=levels, cmap="cividis", alpha=0.55)
    ax.contour(W0, W1, F, levels=levels, colors="white", linewidths=0.6)
    ax.set(xlim=box[:2], ylim=box[2:], xlabel="$w_0$ (intercept)", ylabel="$w_1$ (rated 8 or 9)")


def batches_figure(pair: Pair, w_star: np.ndarray, out_dir: Path) -> dict[str, object]:
    """Steps from ``ARROWS`` random batches: far from the optimum they agree; near it, they scatter.

    Every arrow is one step, -BALL_LR times a batch gradient, from the same point.

    Args:
        pair: The attempt's two-weight model.
        w_star: Its optimum.
        out_dir: The figure directory.

    Returns:
        For the far and the near point: the full step's length, the batch steps' spread in
        degrees, and the distance from their mean to the full step.
    """
    A = pair.design()
    rng = np.random.default_rng((SEED, 13))
    near = w_star + np.array(NEAR_OFFSET)
    out: dict[str, object] = {}
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL)
    zoom = (near[0] - 0.3, near[0] + 0.3, near[1] - 0.3, near[1] + 0.3)
    boxes = ((-3.4, 0.4, -0.6, 2.6), zoom)
    for ax, w, box, key in zip(axes, (np.zeros(2), near), boxes, ("far", "near"), strict=True):
        _contour(ax, pair, box)
        steps = np.array([
            -BALL_LR * gradient(w, A[r], pair.y[r])
            for r in (rng.choice(len(A), BATCH, replace=False) for _ in range(ARROWS))
        ])  # fmt: skip
        full = -BALL_LR * gradient(w, A, pair.y)
        for d in steps:
            thin = {"arrowstyle": "-|>", "color": aml_style.PALETTE[1], "lw": 1.0, "alpha": 0.6,
                    "mutation_scale": 9}  # fmt: skip
            ax.annotate("", w + d, w, arrowprops=thin)
        ax.annotate("", w + full, w, arrowprops={"arrowstyle": "-|>", "color": aml_style.INK,
                    "lw": 2.4, "mutation_scale": 16})  # fmt: skip
        ax.plot(*w_star, marker="*", ms=16, color="white", mec=aml_style.INK, ls="none")
        ang = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
        out[key] = {
            "full_len": float(np.linalg.norm(full)),
            "spread_deg": float(np.ptp(ang)),
            "mean_err": float(np.linalg.norm(steps.mean(0) - full)),
        }
    axes[0].add_patch(plt.Rectangle((zoom[0], zoom[2]), 0.6, 0.6, fill=False, lw=1.4,
                                    ec=aml_style.INK))  # fmt: skip
    axes[0].set_title(f"far off: {ARROWS} batches agree", loc="left", fontsize=13)
    axes[1].set_title("near the bottom: they scatter", loc="left", fontsize=13)
    axes[1].set_ylabel("")
    aml_style.save(fig, out_dir, "batch_arrows")
    return out


def _surface_with_paths(
    ax: plt.Axes,
    pair: Pair,
    box: tuple[float, float, float, float],
    paths: list[tuple],
    cap: float = np.inf,
) -> None:
    """The loss surface over ``box``, cut flat above ``cap``, with each path drawn on it.

    Args:
        ax: 3D axes.
        pair: The two-weight model.
        box: (w0_lo, w0_hi, w1_lo, w1_hi).
        paths: (iterates, colour, label) per path.
        cap: Height above which the surface is cut flat.
    """
    W0, W1 = np.meshgrid(np.linspace(box[0], box[1], 81), np.linspace(box[2], box[3], 81))
    F = np.minimum(pair.loss(W0, W1), cap)
    ax.computed_zorder = False
    aml_style.surface3d(ax, W0, W1, F, cmap="cividis")
    lift = 0.01 * np.ptp(F)
    for path, color, label in paths:
        inside = ((path[:, 0] >= box[0]) & (path[:, 0] <= box[1])
                  & (path[:, 1] >= box[2]) & (path[:, 1] <= box[3]))  # fmt: skip
        z = np.where(inside, np.minimum(pair.loss(*path.T), cap) + lift, np.nan)
        P = np.where(inside[:, None], path, np.nan)
        ax.plot(P[:, 0], P[:, 1], z, color=color, lw=1.6, ls="-", label=label, zorder=5)
        ax.scatter(*P[-1], z[-1], s=30, color=color, edgecolor="white", zorder=6)
    ax.set_xlabel("$w_0$", labelpad=6)
    ax.set_ylabel("$w_1$", labelpad=6)
    ax.set_zticks([])


def ravine_figure(pairs: dict[str, Pair], out_dir: Path) -> dict[str, object]:
    """Gradient descent on deck age in decades from 0 (a ravine), and centered and scaled.

    Args:
        pairs: The models from ``two_weight_pairs``.
        out_dir: The figure directory.

    Returns:
        Per unit choice: the condition number, the step, the optimum, where the run ends, its
        final gap, and the steps it took to come within 1e-3 (None if it did not).
    """
    fig = plt.figure(figsize=aml_style.FULL)
    out: dict[str, object] = {}
    boxes = {"decades": (-2.2, 0.3, -0.55, 0.15), "scaled": (-3.3, 0.3, -1.0, 1.0)}
    titles = {"decades": "age in decades from 0", "scaled": "age centered and scaled"}
    for k, name in enumerate(("decades", "scaled")):
        pair = pairs[name]
        w_star = pair.optimum()
        lr = RAVINE_SAFE / float(np.linalg.eigvalsh(hessian(np.zeros(2), pair.design())).max())
        path = gd_path(pair, lr, RAVINE_STEPS)
        gap = pair.loss(*path.T) - pair.loss(*w_star)
        reach = int(np.argmax(gap < 1e-3)) if (gap < 1e-3).any() else None
        out[name] = {
            "kappa": pair.kappa(w_star),
            "lr": lr,
            "w_star": [float(v) for v in w_star],
            "w_end": [float(v) for v in path[-1]],
            "gap_end": float(gap[-1]),
            "steps_to_1e-3": reach,
        }
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        b = boxes[name]
        _surface_with_paths(ax, pair, b, [(path, aml_style.PALETTE[1], "gradient descent")],
                            cap=RAVINE_CAP)  # fmt: skip
        z = np.minimum(pair.loss(*path.T), RAVINE_CAP)
        ax.plot(path[:, 0], path[:, 1], z, "o", ms=3, color=aml_style.PALETTE[1], zorder=6)
        ax.scatter(*w_star, pair.loss(*w_star), marker="*", s=160, color="white",
                   edgecolor=aml_style.INK, zorder=7)  # fmt: skip
        # Shrunk so w1's label stays inside the axes box, which the tight crop keeps; the long,
        # narrow decades box needs more.
        ax.set_box_aspect((b[1] - b[0], b[3] - b[2], 1.0), zoom=RAVINE_ZOOM[name])
        ax.set_xlabel("$w_0$", labelpad=12)  # clear of the -1 tick label
        ax.set_ylabel("$w_1$", labelpad=6)
        ax.set(xticks=[-2, -1, 0], yticks=RAVINE_W1_TICKS[name])
        steps = f"{RAVINE_STEPS} steps" if reach is None else f"there in {reach} steps"
        ax.set_title(f"{titles[name]}: $\\kappa$ = {out[name]['kappa']:.2g}, {steps}",
                     loc="left", fontsize=12.5)  # fmt: skip
        aml_style.style3d(ax, **RAVINE_VIEW[name])
    aml_style.save(fig, out_dir, "ravine3d")
    return out


def steps_figure(pair: Pair, w_star: np.ndarray, out_dir: Path) -> dict[str, object]:
    """SGD at three step sizes on the two-weight bowl, and the loss above the optimum per step.

    Args:
        pair: The attempt's two-weight model.
        w_star: Its optimum.
        out_dir: The figure directory.

    Returns:
        Per step size: the gap after 100 steps and at the end, and the median and largest gap
        over the last 100 steps.
    """
    F_star = float(pair.loss(*w_star))
    fig = plt.figure(figsize=aml_style.FULL)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    box = (-3.6, 0.4, -0.8, 3.2)
    paths, out = [], {}
    colors = (aml_style.PALETTE[0], aml_style.PALETTE[2], aml_style.PALETTE[1])
    for lr, color in zip(PIC_LRS, colors, strict=True):
        path, _ = sgd_path(pair, lr, PIC_STEPS)
        gap = pair.loss(*path.T) - F_star
        paths.append((path, color, f"$\\eta$ = {lr:g}"))
        ax2.plot(np.arange(len(gap)), np.maximum(gap, 1e-6), color=color, lw=1.6, ls="-",
                 label=f"$\\eta$ = {lr:g}")  # fmt: skip
        out[f"{lr:g}"] = {
            "gap_100": float(gap[100]),
            "gap_end": float(gap[-1]),
            "median_gap_last100": float(np.median(gap[-100:])),
            "max_gap_last100": float(gap[-100:].max()),
        }
    _surface_with_paths(ax3, pair, box, paths)
    ax3.scatter(*w_star, F_star, marker="*", s=160, color="white", edgecolor=aml_style.INK,
                zorder=7)  # fmt: skip
    ax3.set_box_aspect(None, zoom=1.15)
    aml_style.style3d(ax3, elev=42, azim=-120)
    ax2.set(yscale="log", xlabel="SGD step (256 decks each)", ylabel="loss above the optimum")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=11)
    aml_style.save(fig, out_dir, "steps3d")
    return out


def noise_ball_figure(pair: Pair, w_star: np.ndarray, out_dir: Path) -> dict[str, object]:
    """SGD at a constant step hops around the optimum; the running average settles toward it.

    Args:
        pair: The attempt's two-weight model.
        w_star: Its optimum.
        out_dir: The figure directory.

    Returns:
        The step, the steps taken, the last iterate's median gap after step 100, the average's
        final gap, and the iterate's standard deviation per weight after step 100.
    """
    path, avg = sgd_path(pair, BALL_LR, BALL_STEPS)
    F_star = float(pair.loss(*w_star))
    tail = path[100:]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    r = BALL_ZOOM
    _contour(ax, pair, (w_star[0] - r, w_star[0] + r, w_star[1] - r, w_star[1] + r))
    ax.plot(tail[:, 0], tail[:, 1], color=aml_style.PALETTE[1], lw=0.7, ls="-", alpha=0.7,
            label=f"SGD, steps 100–{BALL_STEPS}")  # fmt: skip
    ax.plot(avg[100:, 0], avg[100:, 1], color=aml_style.INK, lw=2.6, ls="-", label="its average")
    ax.plot(*w_star, marker="*", ms=16, color="white", mec=aml_style.INK, ls="none")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9, frameon=True)
    aml_style.save(fig, out_dir, "noise_ball")
    gap = pair.loss(*path.T) - F_star
    gap_avg = pair.loss(*avg.T) - F_star
    return {
        "lr": BALL_LR,
        "steps": BALL_STEPS,
        "last_median_gap": float(np.median(gap[100:])),
        "avg_gap_end": float(gap_avg[-1]),
        "ball_sd": [float(v) for v in tail.std(0)],
    }


def title_art_figure(pair: Pair, w_star: np.ndarray, out_dir: Path) -> None:
    """The title art: SGD's golden, jittering path from w = 0 down into the dark sun at w*.

    Args:
        pair: The attempt's two-weight model.
        w_star: Its optimum.
        out_dir: The figure directory.
    """
    path, _ = sgd_path(pair, BALL_LR, PIC_STEPS)
    lo, hi = np.array([-3.0, -0.3]), np.array([0.3, 1.9])
    sun = (0.30, 0.95)

    def to_xy(w: np.ndarray) -> tuple[float, float]:
        u = (w - lo) / (hi - lo)
        # w* lands on the sun's centre; w = 0 at the upper right
        return (sun[0] + (u[0] - (w_star[0] - lo[0]) / (hi - lo)[0]) * 0.8,
                sun[1] + (u[1] - (w_star[1] - lo[1]) / (hi - lo)[1]) * 0.6)  # fmt: skip

    c = title_art.Canvas(seed=13)
    c.ground(("#0e1318", "#16131f", "#1f0f12", "#07070c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.05, 0.06, 0.55, 0.42, 4), soft=4), "#5a2216", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.58, 0.06, 0.38, 0.30, 3), soft=4), "#3e1a14", 0.85)
    rim = c.disc(sun, 0.11, soft=4) - c.disc(sun, 0.095, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc(sun, 0.095, soft=3), "#1d0a10", 0.92)
    c.glow(c.lines([[to_xy(w) for w in path]], 2.5, soft=1.0), "#f3b45c", 0.9, spread=10)
    c.glow(c.disc(to_xy(path[0]), 0.012, soft=1.5), "#ffe2a0", 1.0, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean forms in rust; from a glowing "
        "point at the upper right a golden line falls in a long jittering stroke and then circles, "
        "trembling, inside a dark sun with a burning rim: the lecture's stochastic gradient "
        "descent, three hundred batches of bridge decks, settling at its optimum.",
    )


def two_weight_pictures(t: pd.DataFrame, train: np.ndarray, out_dir: Path) -> dict[str, object]:
    """Every two-weight picture: the batches, the ravine, three step sizes, the noise ball.

    Args:
        t: Transitions from ``load``.
        train: Boolean mask of the training rows.
        out_dir: The figure directory.

    Returns:
        The attempt model's optimum and drop shares, and each picture's numbers.
    """
    pairs = two_weight_pairs(t, train)
    good = pairs["good"]
    w_star = good.optimum()
    title_art_figure(good, w_star, out_dir)
    return {
        "good": {
            "w_star": [float(v) for v in w_star],
            "kappa": good.kappa(w_star),
            "loss_star": float(good.loss(*w_star)),
            "drop_good_pct": 100 * float(good.y[good.v == 1].mean()),
            "drop_other_pct": 100 * float(good.y[good.v == 0].mean()),
        },
        "batches": batches_figure(good, w_star, out_dir),
        "ravine": ravine_figure(pairs, out_dir),
        "steps": steps_figure(good, w_star, out_dir),
        "ball": noise_ball_figure(good, w_star, out_dir),
    }


def write_tex(n: dict, out_dir: Path) -> Path:
    """Macros for the diagram: the setting and the first SGD step.

    Args:
        n: The lecture's numbers.
        out_dir: The figure directory.

    Returns:
        The path of ``numbers.tex``.
    """
    s, f = n["setting"], n["first_step"]
    macros = {
        "NTrain": f"{s['train']:,}",
        "NWeights": str(s["weights"]),
        "Batch": str(BATCH),
        "StepsPerPass": f"{s['steps_per_pass']:,}",
        "Drops": str(f["drops"]),
        "GZero": tex_num(f["g0"], ".3f"),
        "WZero": tex_num(f["w0"], ".3f"),
        "LossBefore": tex_num(f["loss_before"], ".3f"),
        "LossAfter": tex_num(f["loss_after"], ".3f"),
        "Eta": tex_num(n["sweep"]["sgd"]["best"], "g"),
    }
    return write_tex_macros(macros, out_dir, Path(__file__).name)


def kappa(H: np.ndarray) -> float:
    """Condition number of a symmetric positive definite matrix.

    Args:
        H: A Hessian.

    Returns:
        Its largest eigenvalue over its smallest.
    """
    ev = np.linalg.eigvalsh(H)
    return float(ev.max() / ev.min())


def exact_fit(
    A: np.ndarray, y: np.ndarray, names: list[str]
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Newton's optimum, its Hessian, the condition number, and the smoothness constant L.

    The condition number is also given with the 0/1 code columns rescaled, through the
    Hessian alone: inputs A M have the Hessian M^T H M in their own weights (q4 review, L13-D6).

    Args:
        A: Training inputs with an intercept column.
        y: Training labels.
        names: The design's column names.

    Returns:
        The optimum, the numbers for the slides, and the Hessian at the optimum.
    """
    w_star, newton_losses = newton(A, y)
    F_star = loss(w_star, A, y)
    H = hessian(w_star, A)
    ev, V = np.linalg.eigh(H)
    flat = np.argsort(-np.abs(V[:, 0]))[:5]  # the flattest direction's largest components
    L = float(np.linalg.eigvalsh(A.T @ A / len(A)).max() / 4)  # smoothness: sigma' <= 1/4
    codes = [j for j, n in enumerate(names) if "=" in n]
    mean, sd = A[:, codes].mean(0), A[:, codes].std(0)
    scaled = np.eye(len(names))
    scaled[codes, codes] = 1 / sd  # each code column to standard deviation 1
    centred = scaled.copy()
    centred[0, codes] = -mean / sd  # and centred: the intercept absorbs the means
    return (
        w_star,
        {
            "steps": len(newton_losses) - 1,
            "gaps": [v - F_star for v in newton_losses],
            "kappa": float(ev.max() / ev.min()),
            "kappa_codes_scaled": kappa(scaled.T @ H @ scaled),
            "kappa_codes_standardized": kappa(centred.T @ H @ centred),
            "L": L,
            "safe_step": 1 / L,
            "lambda_max": float(ev.max()),
            "lambda_max_over_L": float(ev.max()) / L,
            "flattest": [names[i] for i in flat],
        },
        H,
    )


def main() -> None:
    """Generate every figure and number for CEE Lecture 13."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    # One BLAS thread: the same sums in the same order here and in the notebook.
    threadpool_limits(1)
    t = load()
    train = (t["nbi_year"] < TEST_FROM).to_numpy()
    A, names = design(t, train, train)
    A_te, _ = design(t, train, ~train)
    y_all = t["drop"].to_numpy(float)
    state = t["state"].to_numpy()
    y = y_all[train]
    test = (A_te, y_all[~train], state[~train])
    w_star, newton_nums, H = exact_fit(A, y, names)
    fits = Fits(A, y, w_star, loss(w_star, A, y), A_te[test[2] == ILLINOIS])
    exact, exact_by_state = forecasts(w_star, A_te, test[2])
    runs, sweep_table, sweep_gaps = best_runs(fits)
    after_one = {m: forecasts(fit(A, y, r, 1, [1])[1], A_te, test[2])[0] for m, r in runs.items()}
    bridge = (t.loc[train, "state"] + "|" + t.loc[train, "structure"]).to_numpy()
    cov = clustered_covariance(fits, bridge, H)
    setting_nums = setting(t, train, names)
    file_res, err_shuffled = file_order(
        fits,
        runs,
        state[train],
        test=test,
        exact_by_state=exact_by_state,
        drop_by_state=setting_nums["drop_by_state"],
        out_dir=out_dir,
    )
    curves, kept_average, gd_table = long_runs(fits, runs, newton_nums["L"])
    for m in ("sgd", "average"):
        file_res[f"ratio_{m}"] = file_res[m]["gap"] / sweep_table[m]["median"]
    fl = floors(fits, H, cov)
    gd_table["vs_average_1"] = curves["gd"][max(GD_PASSES)] / curves["average"][1]
    band = [g for m in ("sgd", "momentum", "adam") for c, g in curves[m].items()
            if NOISE_BAND_PASSES[0] <= c <= NOISE_BAND_PASSES[1]]  # fmt: skip
    numbers = {
        "setting": setting_nums,
        "drop_by_rating": drop_by_rating_figure(t, train, out_dir),
        "attempt": attempt(t),
        "first_step": first_step(A, y, replace(runs["sgd"], seed=SEED)),
        "newton": newton_nums,
        "verify": {
            "finite_difference": check_gradient(A, y, w_star),
            "sklearn": check_sklearn(A, y, w_star),
            "torch": check_torch(A, y),
        },
        "floors": fl,
        "sweep": sweep_table,
        "gd": gd_table,
        "curves": {m: {str(k): v for k, v in c.items()} for m, c in curves.items()},
        "noise_band": {"lo": min(band), "hi": max(band)},
        "exact": exact,
        "one_pass": after_one,
        "raw_units": raw_units(t, train, runs),
        "file_order": file_res,
        "coefficients": coefficients(
            fits,
            names,
            cov,
            kept_average,
            flattest=newton_nums["flattest"],
            err_shuffled=err_shuffled,
        ),
    }
    numbers["budget"] = budget(
        fits,
        cov,
        test,
        w_average=kept_average[1],
        names=names,
        sgd_forecasts=sweep_table["sgd"]["illinois"],
        file_forecast=file_res["file"]["illinois"],
    )
    if abs(numbers["budget"]["one_pass"] - after_one["average"]["illinois"]) > 1e-6:
        raise ValueError("the averaged pass's weights differ between the long and one-pass runs")
    convergence_figure(curves, numbers["newton"]["gaps"], fl, gd_table["best"], out_dir)
    step_size_figure(sweep_gaps, out_dir)
    numbers["two_weight"] = two_weight_pictures(t, train, out_dir)
    aml_style.thumbnails(out_dir, ("batch_arrows", "ravine3d", "steps3d", "file_order"))
    write_numbers(numbers, out_dir)
    print(write_tex(numbers, out_dir))


if __name__ == "__main__":
    main()
