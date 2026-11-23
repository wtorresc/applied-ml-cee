"""Applied Machine Learning for CEE — Lecture 13: Stochastic Gradient Descent.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l13_stochastic_gradient.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_nbi_us.sh`` once (FHWA National Bridge Inventory, all states,
2018-2024; U.S. Government work).
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
    # Applied Machine Learning for CEE — Lecture 13: Stochastic Gradient Descent

    **Slides:** *Applied Machine Learning for CEE*, Lecture 13 (`lectures/cee_l13_stochastic_gradient`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §2.1.3 (stochastic gradient descent and steplength schedules) and §16.4.4 (gradient scaling: momentum, Adagrad, RMSprop, Adam); Bottou, Curtis & Nocedal (2018), *Optimization Methods for Large-Scale Machine Learning*, §4.
    **Data:** FHWA National Bridge Inventory, all-states files 2018, 2020, 2022, 2024 (U.S. Government work, public domain). Fetch with `bash scripts/fetch_nbi_us.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked with an `assert`. It holds 1.3
    million rows of 108 inputs in memory (1.1 GB as float64; the run peaks near 6 GB of RAM) and
    runs a few hundred passes of stochastic optimization on one CPU thread: about 7 minutes
    (measured 2026-11-21).

    **Your tasks** (from the slides): take one SGD step by hand, write SGD, momentum, and Adam
    and match PyTorch and Newton's method, sweep the step size, then break the method by
    streaming the inventory in the order FHWA publishes it.
    """)
    return


@app.cell
def _():
    import os
    import random
    import warnings
    from dataclasses import dataclass, replace
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # One BLAS thread, as in the slides' figure script: the same sums in the same order, so every
    # run reproduces the slides' numbers.
    threadpool_limits(1)

    # Every optimizer's step is chosen on seeds 0-4 and reported on seeds 5-9; single runs use 5.
    SELECT_SEEDS, REPORT_SEEDS = range(5), range(5, 10)
    REPORT_SEED = REPORT_SEEDS[0]
    BATCH = 256  # rows per SGD step
    TEST_FROM = 2022  # pairs 2022 -> 2024 are the test cycle
    ILLINOIS = "17"  # FIPS state code (NBI Item 1)
    GOOD = 8  # the attempt's input: a deck rated 8 or 9
    CHECKS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 5, 10)  # passes kept
    LONG = 10  # passes in the long runs
    GD_PASSES = (1, 2, 5, 10, 20, 50, 100)
    # Gradient descent's steps, in units of 1/L: finer near its stability limit (about 5.8/L).
    GD_STEPS = (1.0, 2.0, 4.0, 5.0, 6.0, 8.0)
    TRACE_POINTS = 40  # checkpoints within the one published-order pass
    BETA = 0.9  # momentum
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    STEPS = {
        "sgd": (0.1, 0.3, 1.0, 3.0, 10.0),
        "average": (0.1, 0.3, 1.0, 3.0, 10.0),
        "momentum": (0.03, 0.1, 0.3, 1.0),
        "adam": (0.001, 0.003, 0.01, 0.03),
    }
    # Steps on inputs in raw units; each sweep must find its best step inside the grid.
    RAW_STEPS = (1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5)
    RAW_ADAM = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
    TORCH_STEPS = 200
    NOISE_BAND_PASSES = (1.0, 10.0)  # passes over which the noise ball is read
    # Sentinel values, not measurements: one ADT of 999,999 and two deck widths of 999.9 m,
    # all in 2018 pairs; those rows are dropped on loading (and checked to be exactly these).
    SENTINEL_ADT, SENTINEL_WIDTH_M = 999_999.0, 999.9
    SENTINEL_ROWS = {("39", "2504235"), ("27", "82527"), ("51", "000000000003802")}
    WIDE_DECK_M = 200.0  # decks wider than this are kept as published: a stated limitation
    CYCLE_MONTHS = 24  # the NBI's routine inspection interval
    NUMERIC = ("age", "age2", "since_work", "log_adt", "pct_truck", "truck_missing",
               "log_length", "deck_width", "months")  # fmt: skip
    CODED = ("deck_now", "deck_type", "surface", "membrane", "protection", "kind",
             "owner_group", "state")  # fmt: skip
    # Rare codes pooled with "other": Item 107 code 7 (aluminum) into 9; Item 108C code 4
    # (cathodic protection) into 9; Item 43A codes 8 and 9 (masonry, aluminum) into 0.
    POOL = {"deck_type": ({"7"}, "9"), "protection": ({"4"}, "9"), "kind": ({"8", "9"}, "0")}
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    def choose_step(scores: dict, what: str) -> float:
        """The step with the smallest score, which must lie strictly inside its grid.

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

    return (
        ADAM_BETAS,
        ADAM_EPS,
        BATCH,
        BETA,
        CHECKS,
        CODED,
        CYCLE_MONTHS,
        GD_PASSES,
        GD_STEPS,
        GOOD,
        ILLINOIS,
        LONG,
        LogisticRegression,
        NOISE_BAND_PASSES,
        NUMERIC,
        PALETTE,
        POOL,
        Path,
        RAW_ADAM,
        RAW_STEPS,
        REPORT_SEED,
        REPORT_SEEDS,
        SEED,
        SELECT_SEEDS,
        SENTINEL_ADT,
        SENTINEL_ROWS,
        SENTINEL_WIDTH_M,
        STEPS,
        TEST_FROM,
        TORCH_STEPS,
        TRACE_POINTS,
        WIDE_DECK_M,
        choose_step,
        dataclass,
        expit,
        np,
        pd,
        plt,
        replace,
        torch,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 5–12

    On your own, no notes, two minutes:

    1. QR solves least squares in about $ND^2$ operations (Lecture 5). With $N$ = 864,669 rows
       and $D$ = 108 weights, is that seconds or hours? With a million weights?
    2. What system does one Newton step for logistic regression solve (Lecture 6)?
    3. L-BFGS chose the GP's lengthscale (Lecture 12). What did it need from us at each step?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. About $10^{10}$ multiply-adds: seconds. With a million weights, $10^{18}$:
                   never, and the $D\times D$ matrix alone would not fit in memory.
                2. $(A^{\top}SA)\,\boldsymbol{\delta}=A^{\top}(\mathbf{p}-\mathbf{y})$ with
                   $S=\mathrm{diag}(p_i(1-p_i))$: a weighted least squares with a $D\times D$
                   matrix.
                3. The objective and its gradient. Today's methods need only the gradient, and
                   only on a few rows at a time.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Will a bridge deck lose at least one rating point (NBI Item 58) by its next inspection? A
    *pair* joins a bridge's record in one national file to its record two years later. The
    inputs are the deck's rating now, its age, the years since its last reconstruction, traffic,
    trucks, size, the months between the two inspections, and coded items: deck type, wearing
    surface, membrane, deck protection, structure kind, owner, and state. With the intercept,
    108 weights: Lecture 6's logistic regression, fitted to the whole country.

    **Data check.** Three 2018 records carry sentinel codes, not measurements: an ADT of 999,999
    vehicles a day (Ohio, structure 2504235) and deck widths of 999.9 m (Minnesota 82527,
    Virginia 3802). `load` drops exactly those three and stops if a new release has others.
    Decks reported wider than 200 m are otherwise kept as published, a limitation.

    **Decide:** how many decks to expect to drop in each state next cycle, for the budget.
    Train on pairs starting in 2018 and 2020; test on 2022 → 2024.
    """)
    return


@app.cell
def _(
    CODED,
    NUMERIC,
    POOL,
    Path,
    SENTINEL_ADT,
    SENTINEL_ROWS,
    SENTINEL_WIDTH_M,
    np,
    pd,
):
    def find_csv() -> Path:
        """Locate the national pairs table from the repo root, notebooks/, or ipynb/.

        Returns:
            Path of ``data/derived/nbi_us_deck_transitions.csv.gz``.

        Raises:
            FileNotFoundError: If it is missing (run ``bash scripts/fetch_nbi_us.sh``).
        """
        rel = Path("data") / "derived" / "nbi_us_deck_transitions.csv.gz"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_nbi_us.sh`")

    def load(path: Path) -> pd.DataFrame:
        """The national deck pairs, sentinel rows dropped, rare codes pooled, owners grouped.

        Args:
            path: The table written by ``scripts/fetch_nbi_us.sh``.

        Returns:
            One row per pair, in FHWA's published order (file, state, structure number).

        Raises:
            ValueError: If the sentinel rows are not exactly the three expected, or a code
                is not what the inputs assume.
        """
        codes = ["state", "structure", "facility", "crosses", "owner", "kind", "deck_type",
                 "surface", "membrane", "protection"]  # fmt: skip
        t = pd.read_csv(path, dtype=dict.fromkeys(codes, str))
        # An ADT of 999,999 and deck widths of 999.9 m are codes for "unknown", not values.
        bad = (t["adt"] >= SENTINEL_ADT) | (t["deck_width_m"] >= SENTINEL_WIDTH_M)
        found = set(zip(t.loc[bad, "state"], t.loc[bad, "structure"], strict=True))
        if found != SENTINEL_ROWS or int(bad.sum()) != len(SENTINEL_ROWS):
            raise ValueError(f"sentinel rows changed: {sorted(found)}")
        t = t[~bad].reset_index(drop=True)
        for c, (small, to) in POOL.items():
            t[c] = t[c].where(~t[c].isin(small), to)
        if not pd.api.types.is_integer_dtype(t["deck"]):  # a float rating would give "8.0"
            raise ValueError(f"deck ratings read as {t['deck'].dtype}")
        t["deck_now"] = t["deck"].clip(3, 9).astype(str)  # ratings 0-3 are rare: one level
        own = pd.to_numeric(t["owner"], errors="raise")  # every owner code is a number
        t["owner_group"] = np.select([own == 1, own.isin([2, 3, 4])], ["state", "local"], "other")
        return t

    def numeric_inputs(t: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
        """Numeric inputs in the model's units, or in the inventory's own units (``raw``).

        Args:
            t: Pairs from ``load``.
            raw: Years, vehicles per day (linear), percent, meters, months; unscaled.

        Returns:
            One column per name in ``NUMERIC``.
        """
        age = t["nbi_year"] - t["year_built"]
        since = t["nbi_year"] - np.maximum(t["year_built"], t["year_reconstructed"])
        trucks = t["pct_truck"].fillna(0.0)  # Item 109 is blank for 70% of Minnesota's pairs
        if raw:
            cols = {"age": age, "age2": age**2, "since_work": since, "log_adt": t["adt"],
                    "pct_truck": trucks, "log_length": t["length_m"],
                    "deck_width": t["deck_width_m"], "months": t["months"]}  # fmt: skip
        else:
            cols = {"age": age / 10, "age2": (age / 10) ** 2, "since_work": since / 10,
                    "log_adt": np.log10(t["adt"] + 1), "pct_truck": trucks / 10,
                    "log_length": np.log10(t["length_m"]),
                    "deck_width": t["deck_width_m"] / 10, "months": t["months"] / 12}  # fmt: skip
        cols["truck_missing"] = t["pct_truck"].isna().astype(float)
        return pd.DataFrame(cols)[list(NUMERIC)].astype(float)

    def design(
        t: pd.DataFrame,
        train: np.ndarray,
        rows: np.ndarray,
        raw: bool = False,
        scale: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Inputs with an intercept column: numeric inputs, then one-hot codes.

        Scaling and the reference level of each code (its most common training level) come
        from the training rows only. The matrix is allocated once, for ``rows`` only, and
        filled column by column, which keeps the peak memory near the matrix itself.

        Args:
            t: Pairs from ``load``.
            train: Boolean mask of the training rows.
            rows: Boolean mask of the rows to build (training or test).
            raw: Numeric inputs in the inventory's units.
            scale: Standardize the numeric inputs.

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
        cols = ["intercept", *NUMERIC, *(f"{c}={lv}" for c in CODED for lv in levels[c])]
        A = np.empty((int(rows.sum()), len(cols)))
        A[:, 0] = 1.0
        A[:, 1 : 1 + len(NUMERIC)] = Z[rows]
        j = 1 + len(NUMERIC)
        for c in CODED:
            col = t.loc[rows, c].to_numpy()
            for lv in levels[c]:
                A[:, j] = col == lv
                j += 1
        assert j == len(cols), (j, len(cols))
        return A, cols

    return design, find_csv, load


@app.cell
def _(ILLINOIS, TEST_FROM, design, find_csv, load):
    t = load(find_csv())
    train = (t["nbi_year"] < TEST_FROM).to_numpy()
    A, names = design(t, train, train)
    A_te, _ = design(t, train, ~train)
    _y_all = t["drop"].to_numpy(float)
    _state = t["state"].to_numpy()
    y, y_te, state_te = _y_all[train], _y_all[~train], _state[~train]
    state_tr = _state[train]
    A_il = A_te[state_te == ILLINOIS]
    assert len(t) == 1_299_042 and train.sum() == 864_669 and (~train).sum() == 434_373
    assert A.shape == (864_669, 108) and A_te.shape == (434_373, 108)
    assert t["adt"].max() == 919_000 and t.loc[train, "adt"].quantile(0.99) == 113_000
    assert (t.loc[train, "deck_width_m"] > 200).sum() == 33 and (
        t["deck_width_m"] > 200
    ).sum() == 53
    assert t.groupby(["state", "structure"]).ngroups == 476_144 and t["state"].nunique() == 52
    assert round(y.mean(), 3) == 0.104
    _cycles = t.groupby("nbi_year")["drop"].mean().round(3).to_dict()
    assert _cycles == {2018: 0.111, 2020: 0.097, 2022: 0.087}, _cycles
    assert len(A) // 256 == 3377 and round((t.loc[train, "months"] == 24).mean(), 2) == 0.71
    print(f"{len(t):,} pairs on {t.groupby(['state', 'structure']).ngroups:,} bridges; "
          f"train {train.sum():,}, test {(~train).sum():,}; {A.shape[1]} weights")  # fmt: skip
    print("share of decks that dropped, by cycle:", _cycles)
    return A, A_il, A_te, names, state_te, state_tr, t, train, y, y_te


@app.cell
def _(PALETTE, plt, t, train):
    _g = t[train].groupby("deck_now")["drop"].agg(["mean", "size"])
    _shares = (100 * _g["mean"]).round().astype(int).to_dict()
    assert _shares == {"3": 6, "4": 4, "5": 5, "6": 7, "7": 9, "8": 23, "9": 39}, _shares
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.bar(range(len(_g)), 100 * _g["mean"], color=PALETTE[0])
    _ax.set_xticks(range(len(_g)), ["≤3" if r == "3" else r for r in _g.index])
    _ax.set_xlabel("deck rating at the first inspection")
    _ax.set_ylabel("decks that dropped (%)")
    _ax.set_title("Top-rated decks drop most often")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: take one step

    Four Illinois decks, 2018 → 2020. Input $x$ = 1 if rated 8 or 9; $y$ = 1 if it dropped.
    Model $p=\sigma(w_0+w_1x)$, from $w_0=w_1=0$, step size $\eta=1$. Each row's gradient is
    $(p-y)(1,x)$ (Lecture 6).

    1. Batch {A, B}: the average gradient, and the weights after one step.
    2. Batch {C, D}: the same.
    3. Compare the average of the two gradients with the gradient on all four rows.
    """)
    return


@app.cell
def _(GOOD, ILLINOIS, np, pd, t):
    _il = t[(t["state"] == ILLINOIS) & (t["nbi_year"] == 2018)]
    _il = _il[_il["facility"].notna() & _il["crosses"].notna()]
    _good = _il["deck"] >= GOOD
    _picks = {
        "A": _il[_good & (_il["drop"] == 1)].iloc[0],
        "B": _il[~_good & (_il["drop"] == 0)].iloc[0],
        "C": _il[_good & (_il["drop"] == 0)].iloc[0],
        "D": _il[~_good & (_il["drop"] == 0)].iloc[1],
    }
    attempt_rows = pd.DataFrame(
        {
            k: {
                "bridge": f"{r['facility'].title()} over {r['crosses'].title()}",
                "deck": f"{r['deck']} → {r['deck_next']}",
                "x": int(r["deck"] >= GOOD),
                "y": int(r["drop"]),
            }
            for k, r in _picks.items()
        }
    ).T

    def _batch_gradient(keys: str) -> np.ndarray:
        """Average of (p - y)(1, x) over the rows, at w = 0 (every p is 0.5)."""
        rows = attempt_rows.loc[list(keys)]
        return np.mean([(0.5 - r.y) * np.array([1.0, r.x]) for r in rows.itertuples()], axis=0)

    _g1, _g2, _g_all = _batch_gradient("AB"), _batch_gradient("CD"), _batch_gradient("ABCD")
    assert attempt_rows.loc["A", "bridge"] == "Ogden Road over Little Wolf Creek"
    assert np.allclose(_g1, [0, -0.25]) and np.allclose(_g2, [0.5, 0.25])
    assert np.allclose((_g1 + _g2) / 2, _g_all) and np.allclose(_g_all, [0.25, 0])
    attempt_rows
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(
                r"""
                Every $p$ is 0.5. Batch {A, B}: gradient $(0,-0.25)$, so $\mathbf{w}=(0,0.25)$:
                good decks drop more. Batch {C, D}: $(0.5, 0.25)$, so $\mathbf{w}=(-0.5,-0.25)$:
                good decks drop less. The two batches disagree on the sign of $w_1$. Their
                average, $(0.25, 0)$, is exactly the gradient on all four rows. Each step is
                noisy; on average it is the full gradient. That needs random batches, and §5
                removes them.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The decoder is Lecture 6's: $p=\sigma(\mathbf{w}^{\top}\mathbf{z})$, and the forecast for a
    state is $\sum p$ over its decks. What changes is how $\mathbf{w}$ is found. Newton used all
    864,669 rows for every step. SGD shuffles the rows, takes a batch of 256, steps

    $$\mathbf{w}\leftarrow\mathbf{w}-\eta\,A_B^{\top}(\mathbf{p}_B-\mathbf{y}_B)/256,$$

    and moves on to the next batch: 3,377 steps make one pass. The cell below the derivation
    replays the first step, the one in the slide's inset.

    ## 3. Derive

    **Assumption:** each batch is drawn at random from the training rows. Then the batch
    gradient $\mathbf{g}_B=\frac{1}{B}\sum_{i\in B}\nabla\ell_i(\mathbf{w})$ is right on
    average, $\mathbb{E}[\mathbf{g}_B]=\nabla F(\mathbf{w})$, its noise shrinks like $1/B$,
    and it costs $BD$ instead of $ND$. This is exact for batches drawn with replacement. With a
    fresh shuffle each pass (random reshuffling, the standard practice) each batch is a uniform
    sample of the rows, but not independent of the batches before it in the pass; it converges
    at least as fast (Mishchenko, Khaled & Richtárik, 2020).

    **Curvature.** Gradient descent is safe for $\eta\le 1/L$, with $L$ the largest curvature;
    near the optimum its error shrinks like $(1-1/\kappa)^t$ with
    $\kappa=\lambda_{\max}/\lambda_{\min}$ of the Hessian. For the logistic loss
    $\sigma'\le 1/4$, so $L=\lambda_{\max}(A^{\top}A)/(4N)$.

    **Noise.** With a constant $\eta$, SGD reaches a noise ball around the optimum, of size about
    $\eta\sigma^2$. Shrink the step ($\sum\eta_t=\infty$, $\sum\eta_t^2<\infty$; Robbins &
    Monro, 1951) or average the iterates (Polyak & Juditsky, 1992). Momentum keeps a running
    sum of gradients; Adam (Kingma & Ba, 2015) divides each weight's step by a running estimate
    of its gradient's size.
    """)
    return


@app.cell
def _(expit, np):
    def loss(w: np.ndarray, A: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy of a logistic model, computed stably with logaddexp.

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
        """Gradient of ``loss``: A^T (p - y) / n.

        Args:
            w: Weights, shape (D,).
            A: Inputs with an intercept column, shape (n, D).
            y: Labels in {0, 1}, shape (n,).

        Returns:
            The gradient, shape (D,).
        """
        return A.T @ (expit(A @ w) - y) / len(y)

    def hessian(w: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Hessian of ``loss``: A^T S A / n, S = diag(p (1 - p)).

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
        """Newton's method from w = 0 (Lecture 6).

        Args:
            A: Inputs with an intercept column.
            y: Labels in {0, 1}.
            tol: Stop when no weight moves by more than this.
            max_steps: Give up after this many steps.

        Returns:
            The optimum and the loss before each step and after the last.

        Raises:
            RuntimeError: If it has not converged after ``max_steps``.
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

    return gradient, hessian, loss, newton


@app.cell
def _(A, BATCH, SEED, gradient, np, y):
    # Right on average: over one shuffled pass, the batch gradients average to the gradient on
    # the rows they covered (all but the last N % B of the shuffled order).
    _rng = np.random.default_rng(SEED)
    _w = _rng.normal(0, 0.1, A.shape[1])
    _visit = _rng.permutation(len(A))
    _n = len(A) // BATCH
    _batches = np.array(
        [
            gradient(
                _w, A[_visit[j * BATCH : (j + 1) * BATCH]], y[_visit[j * BATCH : (j + 1) * BATCH]]
            )
            for j in range(_n)
        ]
    )
    _covered = gradient(_w, A[_visit[: _n * BATCH]], y[_visit[: _n * BATCH]])
    assert np.allclose(_batches.mean(0), _covered, atol=1e-12)
    # Its noise shrinks like 1/B: the spread of batch gradients around the full one, by B.
    _full = gradient(_w, A, y)
    _spread = {}
    for _B in (16, 256, 4096):
        _rows = [_rng.choice(len(A), _B, replace=False) for _ in range(200)]
        _spread[_B] = np.mean([np.sum((gradient(_w, A[r], y[r]) - _full) ** 2) for r in _rows])
    assert 12 < _spread[16] / _spread[256] < 20 and 12 < _spread[256] / _spread[4096] < 20
    print("mean squared distance of a batch gradient from the full one:",
          {k: f"{v:.2e}" for k, v in _spread.items()}, "(16 times smaller per 16 times B)")  # fmt: skip
    return


@app.cell
def _(A, hessian, loss, names, newton, np, y):
    w_star, _losses = newton(A, y)
    F_star = loss(w_star, A, y)
    newton_gaps = [v - F_star for v in _losses]
    _H = hessian(w_star, A)
    _ev, _V = np.linalg.eigh(_H)
    kappa = _ev.max() / _ev.min()
    # The ravine lives in the 0/1 state codes: rescale those columns (inputs A M have the
    # Hessian M^T H M in their own weights) and the condition number falls.
    _codes = [j for j, n in enumerate(names) if "=" in n]
    _M = np.eye(len(names))
    _M[_codes, _codes] = 1 / A[:, _codes].std(0)  # each code column to SD 1 ...
    _M[0, _codes] = -A[:, _codes].mean(0) / A[:, _codes].std(0)  # ... and centred
    _ev_codes = np.linalg.eigvalsh(_M.T @ _H @ _M)
    _kappa_codes = _ev_codes.max() / _ev_codes.min()
    L_smooth = np.linalg.eigvalsh(A.T @ A / len(A)).max() / 4
    _flat = [names[i] for i in np.argsort(-np.abs(_V[:, 0]))[:5]]
    assert len(_losses) - 1 == 8 and round(F_star, 4) == 0.2960
    assert round(kappa) == 11_507 and round(L_smooth, 2) == 0.72 and round(1 / L_smooth, 2) == 1.38
    assert round(_kappa_codes) == 254  # the ravine is the codes'
    # Near the optimum the largest curvature is about a third of L: steps up to 2/that are stable.
    assert round(_ev.max(), 2) == 0.25 and round(_ev.max() / L_smooth, 2) == 0.34
    assert round(2 / (_ev.max() / L_smooth), 1) == 5.8
    assert _flat == ["state=11", "state=10", "state=32", "state=72", "state=44"], _flat
    assert round(newton_gaps[1], 3) == 0.033 and newton_gaps[3] > 6.2e-5 > newton_gaps[4]
    print(f"Newton: {len(_losses) - 1} steps, F* = {F_star:.4f}; kappa = {kappa:,.0f}; "
          f"1/L = {1 / L_smooth:.2f}")  # fmt: skip
    print("flattest direction: DC, Delaware, Nevada, Puerto Rico, Rhode Island:", _flat)
    print(f"kappa with the code columns centred and scaled: {_kappa_codes:,.0f}")
    return F_star, L_smooth, newton_gaps, w_star


@app.cell
def _(BATCH, GOOD, SEED, gradient, hessian, newton, np, t, train):
    # The slides' two-weight pictures: the attempt's model (x = rated 8 or 9) on every training
    # pair, and deck age as the input, counted in decades from 0 or centered and scaled.
    _tr = t[train]
    _y = _tr["drop"].to_numpy(float)
    _age = (_tr["nbi_year"] - _tr["year_built"]).to_numpy(float)
    _x = (_tr["deck"] >= GOOD).to_numpy(float)
    assert (round(100 * _y[_x == 1].mean(), 1), round(100 * _y[_x == 0].mean(), 1)) == (25.4, 7.5)

    def _two(v):
        """Design with an intercept, the exact optimum, and the loss of any (w0, w1)."""
        A2 = np.column_stack([np.ones_like(v), v])  # (N, 2)
        w2, _ = newton(A2, _y)

        def cost(w):
            z = A2 @ w
            return float(np.mean(np.logaddexp(0.0, z) - _y * z))

        return A2, w2, cost

    def _kappa(A2, w):
        ev = np.linalg.eigvalsh(hessian(w, A2))
        return ev.max() / ev.min()

    def _sgd(A2, lr, steps):
        """SGD from 0 on one shuffled pass's first batches; iterates and their running mean."""
        rows = np.random.default_rng(SEED).permutation(len(A2))[: steps * BATCH]
        w, wb, path, avg = np.zeros(2), np.zeros(2), [], []
        for s in range(steps):
            r = rows[s * BATCH : (s + 1) * BATCH]
            w = w - lr * gradient(w, A2[r], _y[r])
            wb = wb + (w - wb) / (s + 1)
            path.append(w)
            avg.append(wb)
        return np.array(path), np.array(avg)

    # The attempt's model: a bowl with kappa 5.8
    _A, _ws, _F = _two(_x)
    assert np.allclose(_ws, [-2.5195, 1.4409], atol=1e-4) and round(_kappa(_A, _ws), 1) == 5.8
    _Fs = _F(_ws)
    # Batch steps (eta 3): far from the optimum they agree; near it they point every way.
    _rng = np.random.default_rng((SEED, 13))
    for _w, _far in ((np.zeros(2), True), (_ws + np.array([0.12, -0.18]), False)):
        _st = np.array([-3 * gradient(_w, _A[r], _y[r])
                        for r in (_rng.choice(len(_A), BATCH, replace=False) for _ in range(30))])  # fmt: skip
        _ang = np.degrees(np.arctan2(_st[:, 1], _st[:, 0]))
        _full = -3 * gradient(_w, _A, _y)
        # The mean of 30 batch steps lands near the full step, not on it: 30 random draws.
        assert round(np.linalg.norm(_st.mean(0) - _full), 3) == (0.010 if _far else 0.004)
        assert round(np.ptp(_ang)) == (8 if _far else 349), np.ptp(_ang)  # degrees of spread
    # Three step sizes, 300 steps each, the same batches.
    _gaps = {lr: np.array([_F(w) - _Fs for w in _sgd(_A, lr, 300)[0]]) for lr in (0.1, 3, 30)}
    assert round(_gaps[0.1][-1], 4) == 0.0115
    assert round(np.median(_gaps[3][-100:]), 4) == 0.0001
    assert round(np.median(_gaps[30][-100:]), 2) == 0.07
    # The noise ball at eta 3: the iterate wanders, the average settles.
    _path, _avg = _sgd(_A, 3.0, 1000)
    assert np.allclose(_path[99:].std(0), [0.0862, 0.0758], atol=5e-5)
    assert round(1e6 * (_F(_avg[-1]) - _Fs)) == 9  # 9 x 10^-6 above the optimum

    # Deck age: a ravine in decades from 0, a round bowl centered and scaled.
    for _v, _k, _end in ((_age / 10, 88, None), ((_age - _age.mean()) / _age.std(), 1.3, 4)):
        _A2, _w2, _F2 = _two(_v)
        _lr = 1.9 / np.linalg.eigvalsh(hessian(np.zeros(2), _A2)).max()
        _w, _g = np.zeros(2), []
        for _ in range(30):
            _w = _w - _lr * gradient(_w, _A2, _y)
            _g.append(_F2(_w) - _F2(_w2))
        assert round(_kappa(_A2, _w2), 0 if _k == 88 else 1) == _k
        if _end is None:  # 30 steps crawl a quarter of the way along the ravine floor
            assert round(_w[0], 2) == -0.49 and round(_w2[0], 2) == -1.87
        else:
            assert int(np.argmax(np.array(_g) < 1e-3)) + 1 == _end
    print("two-weight pictures: every number checked")
    return


@app.cell
def _(A, BATCH, SEED, gradient, loss, np, runs, y):
    # The slide's inset: the first step of the run, from w = 0 at SGD's chosen step (0.3).
    _rows = np.random.default_rng(SEED).permutation(len(A))[:BATCH]
    _g = gradient(np.zeros(A.shape[1]), A[_rows], y[_rows])
    _w1 = -runs["sgd"].lr * _g
    _w_int = np.zeros_like(_w1)
    _w_int[0] = _w1[0]  # the intercept's move alone
    assert int(y[_rows].sum()) == 29 and round(_g[0], 3) == 0.387 and _g[0] == 0.5 - 29 / 256
    assert (
        round(loss(np.zeros(A.shape[1]), A, y), 3) == 0.693 and round(loss(_w1, A, y), 3) == 0.620
    )
    assert round(loss(_w_int, A, y), 3) == 0.649  # the intercept alone: 60% of the first drop
    print(f"step 1: {int(y[_rows].sum())} of 256 dropped; intercept gradient {_g[0]:.3f}; "
          f"loss {loss(np.zeros(A.shape[1]), A, y):.3f} -> {loss(_w1, A, y):.3f} "
          f"(intercept alone: {loss(_w_int, A, y):.3f})")  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    The slide's listing first: one pass of plain SGD. Then a general loop for the four methods
    (plain, averaged, momentum, Adam), which keeps the weights at chosen points of the run.
    """)
    return


@app.cell
def _(ADAM_BETAS, ADAM_EPS, BATCH, BETA, SEED, dataclass, expit, gradient, np):
    def sgd_pass(A, y, w, lr, rng, B=256):
        visit = rng.permutation(len(A))  # a new order each pass
        for j in range(len(A) // B):
            rows = visit[j * B : (j + 1) * B]
            p = expit(A[rows] @ w)  # (B,)
            g = A[rows].T @ (p - y[rows]) / B  # (D,)
            w = w - lr * g
        return w

    @dataclass(frozen=True)
    class Run:
        """One stochastic run: update rule, step size, and how the rows are visited."""

        kind: str  # "sgd", "momentum", or "adam"
        lr: float
        average: bool = False  # return the running mean of the iterates
        order: str = "shuffle"  # or "file": FHWA's published order
        seed: int = SEED

    def fit(A: np.ndarray, y: np.ndarray, run: Run, passes: float, checks) -> dict:
        """Minibatch optimization of the logistic loss, keeping the weights at checkpoints.

        Each pass visits ``N // BATCH`` batches, shuffled anew each pass or in file order.

        Args:
            A: Inputs with an intercept column, shape (N, D).
            y: Labels in {0, 1}, shape (N,).
            run: The update rule, step size, and seed.
            passes: How many passes over the rows.
            checks: Passes (fractions allowed) at which to keep the weights.

        Returns:
            Pass -> weights (the running mean if ``run.average``).
        """
        rng = np.random.default_rng(run.seed)
        N, D = A.shape
        per_pass = N // BATCH
        at = {round(c * per_pass): c for c in checks if c <= passes}
        w, m, v, w_bar = (np.zeros(D) for _ in range(4))
        b1, b2 = ADAM_BETAS
        kept = {}
        for s in range(round(passes * per_pass)):
            j = s % per_pass
            if j == 0:
                visit = rng.permutation(N) if run.order == "shuffle" else np.arange(N)
            rows = visit[j * BATCH : (j + 1) * BATCH]
            g = gradient(w, A[rows], y[rows])
            if run.kind == "sgd":
                w -= run.lr * g
            elif run.kind == "momentum":
                m = BETA * m + g
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

    def gradient_descent(A: np.ndarray, y: np.ndarray, lr: float, passes) -> dict:
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

    return Run, fit, gradient_descent, sgd_pass


@app.cell
def _(A, A_il, F_star, Run, SEED, expit, fit, loss, np, sgd_pass, y):
    def gap(w: np.ndarray) -> float:
        """Optimization error: training loss above Newton's optimum.

        Args:
            w: Weights.

        Returns:
            The gap in nats.
        """
        return loss(w, A, y) - F_star

    def illinois(w: np.ndarray) -> float:
        """Expected number of Illinois decks that drop in the 2022 -> 2024 cycle.

        Args:
            w: Weights.

        Returns:
            The sum of Illinois's test-cycle probabilities.
        """
        return float(expit(A_il @ w).sum())

    # The listing and the general loop take the same steps, operation for operation.
    _w_listing = sgd_pass(A, y, np.zeros(A.shape[1]), 1.0, np.random.default_rng(SEED))
    (_w_loop,) = fit(A, y, Run("sgd", 1.0), 1, [1]).values()
    assert np.array_equal(_w_listing, _w_loop)
    assert 0 < gap(_w_listing) < 0.01  # seed 0 at eta = 1; the sweep below picks the step
    print(
        f"one pass of the listing at eta = 1, seed 0: {gap(_w_listing):.4f} nats above the optimum"
    )
    return gap, illinois


@app.cell
def _(
    A,
    ADAM_BETAS,
    ADAM_EPS,
    BATCH,
    BETA,
    LogisticRegression,
    Run,
    SEED,
    TORCH_STEPS,
    fit,
    gradient,
    loss,
    np,
    torch,
    w_star,
    warnings,
    y,
):
    # 1. The gradient against central differences, away from the optimum.
    _rng = np.random.default_rng(SEED)
    _w = w_star + _rng.normal(0, 0.1, w_star.shape)
    _g = gradient(_w, A, y)
    _errs = []
    for _i in _rng.choice(len(_w), 5, replace=False):
        _e = np.zeros_like(_w)
        _e[_i] = 1e-5
        _errs.append(abs((loss(_w + _e, A, y) - loss(_w - _e, A, y)) / 2e-5 - _g[_i]))
    _fd = max(_errs) / np.abs(_g).max()
    # 2. Newton against scikit-learn's Newton solver, unpenalized.
    _sk = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=100)
    with warnings.catch_warnings():
        # scikit-learn 1.8 asks for C=np.inf instead of the deprecated penalty=None, then warns
        # that the resulting penalty=None ignores C. The fit is unpenalized, as intended.
        warnings.filterwarnings("ignore", "Setting penalty=None", UserWarning)
        _sk.fit(A[:, 1:], y)
    _sk_diff = np.abs(np.r_[_sk.intercept_, _sk.coef_.ravel()] - w_star).max()
    # 3. Our gradient, momentum, and Adam against autograd and torch.optim, same batches.
    _torch = {}
    _per_pass = len(A) // BATCH
    for _run in (Run("momentum", 0.1), Run("adam", 0.003)):
        (_w_ours,) = fit(A, y, _run, TORCH_STEPS / _per_pass, [TORCH_STEPS / _per_pass]).values()
        _visit = np.random.default_rng(_run.seed).permutation(len(A))
        _p = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)
        _opt = (
            torch.optim.SGD([_p], lr=_run.lr, momentum=BETA)
            if _run.kind == "momentum"
            else torch.optim.Adam([_p], lr=_run.lr, betas=ADAM_BETAS, eps=ADAM_EPS)
        )
        for _s in range(TORCH_STEPS):
            _rows = _visit[_s * BATCH : (_s + 1) * BATCH]
            _Ab, _yb = torch.from_numpy(A[_rows]), torch.from_numpy(y[_rows])
            _opt.zero_grad()
            _z = _Ab @ _p
            torch.mean(torch.nn.functional.softplus(_z) - _yb * _z).backward()
            if _s == 0:
                _g0 = gradient(np.zeros(A.shape[1]), A[_rows], y[_rows])
                _torch["gradient"] = np.abs(_p.grad.numpy() - _g0).max()
            _opt.step()
        _torch[_run.kind] = np.abs(_p.detach().numpy() - _w_ours).max()
    assert _fd < 1e-9 and _sk_diff < 1e-7
    assert _torch["gradient"] < 1e-15 and _torch["momentum"] < 1e-15 and _torch["adam"] < 1e-15
    print(f"finite differences {_fd:.1e}; scikit-learn {_sk_diff:.1e}; PyTorch {_torch}")
    return


@app.cell
def _(
    A,
    A_il,
    A_te,
    ILLINOIS,
    REPORT_SEED,
    REPORT_SEEDS,
    Run,
    SELECT_SEEDS,
    STEPS,
    choose_step,
    expit,
    fit,
    gap,
    illinois,
    np,
    replace,
    state_te,
    w_star,
    y,
):
    def one_pass(run: Run) -> tuple[float, float]:
        """Gap and Illinois forecast after one pass of ``run``.

        Args:
            run: The update rule, step size, and seed.

        Returns:
            The training loss above the optimum and Illinois's expected drops.
        """
        (w,) = fit(A, y, run, 1, [1]).values()
        return gap(w), illinois(w)

    def forecasts(w: np.ndarray) -> dict:
        """Test-cycle forecasts of one fit.

        Args:
            w: Weights.

        Returns:
            The share of test decks expected to drop, and Illinois's expected drops.
        """
        p = expit(A_te @ w)
        return {"share": float(p.mean()), "illinois": float(p[state_te == ILLINOIS].sum())}

    # One protocol for every optimizer: its whole grid of steps on the selection seeds (0-4),
    # then the chosen step alone on the report seeds (5-9), which give every number reported.
    sweeps = {}
    for _m in STEPS:
        _kind = "sgd" if _m == "average" else _m
        sweeps[_m] = {lr: [one_pass(Run(_kind, lr, _m == "average", seed=s)) for s in SELECT_SEEDS]
                      for lr in STEPS[_m]}  # fmt: skip
    runs, best = {}, {}
    for _m, _sw in sweeps.items():
        _med = {lr: float(np.median([g for g, _ in r])) for lr, r in _sw.items()}
        _lr = choose_step(_med, _m)  # raises unless the best step is inside the grid
        runs[_m] = Run("sgd" if _m == "average" else _m, _lr, _m == "average", seed=REPORT_SEED)
        best[_m] = np.array([one_pass(replace(runs[_m], seed=s)) for s in REPORT_SEEDS])  # gap, IL
    median_gap = {_m: float(np.median(b[:, 0])) for _m, b in best.items()}
    exact = forecasts(w_star)
    _lrs = {m: r.lr for m, r in runs.items()}
    assert _lrs == {"sgd": 0.3, "average": 3.0, "momentum": 0.1, "adam": 0.01}, _lrs
    _med = {m: round(v, 4) for m, v in median_gap.items()}
    assert _med == {"sgd": 0.0017, "average": 0.0003, "momentum": 0.0014, "adam": 0.0017}, _med
    assert (round(best["sgd"][:, 0].min(), 4), round(best["sgd"][:, 0].max(), 4)) == (
        0.0014,
        0.0021,
    )
    _chosen = {m: round(float(np.median([g for g, _ in sweeps[m][runs[m].lr]])), 4) for m in runs}
    assert _chosen == {"sgd": 0.0016, "average": 0.0003, "momentum": 0.001, "adam": 0.0015}
    _sgd_steps = {
        lr: round(float(np.median([g for g, _ in r])), 4) for lr, r in sweeps["sgd"].items()
    }
    # SGD's 0.3 and 1 are close (0.0016 against 0.0017, inside the seeds' spread).
    assert _sgd_steps == {0.1: 0.0056, 0.3: 0.0016, 1.0: 0.0017, 3.0: 0.0049, 10.0: 0.069}
    assert round(float(np.median([g for g, _ in sweeps["average"][0.1]])), 3) == 0.012
    assert round(float(np.median([g for g, _ in sweeps["average"][10.0]])), 3) == 0.014
    _il = {m: sorted(round(v) for v in b[:, 1]) for m, b in best.items()}
    assert _il["sgd"] == [2614, 2626, 2655, 2699, 2838], _il
    assert _il["average"] == [2601, 2606, 2606, 2611, 2612], _il
    assert round(exact["illinois"]) == 2688 and round(np.median(best["sgd"][:, 1])) == 2655
    assert round(100 * (max(best["sgd"][:, 1]) / exact["illinois"] - 1)) == 6  # 6% high at most
    assert round(A_il.shape[0]) == 17_746
    assert round(100 * exact["share"], 1) == 10.1  # Newton's national forecast; 8.7% dropped
    print("best step sizes:", _lrs)
    print("median one-pass gap:", _med)
    print("Illinois forecasts after one pass, seeds 5-9:", _il, f"(Newton {exact['illinois']:.0f})")
    return best, exact, forecasts, median_gap, runs, sweeps


@app.cell
def _(
    A,
    CHECKS,
    GD_PASSES,
    GD_STEPS,
    L_smooth,
    LONG,
    NOISE_BAND_PASSES,
    choose_step,
    fit,
    gap,
    gradient_descent,
    np,
    runs,
    y,
):
    curves = {}
    for _m, _run in runs.items():
        _kept = fit(A, y, _run, LONG, CHECKS)
        curves[_m] = {c: gap(w) for c, w in _kept.items()}
        if _m == "average":
            kept_average = _kept
    # Gradient descent is tuned too: each step k/L, the least loss after 100 passes kept.
    with np.errstate(over="ignore", invalid="ignore"):  # a step too large may diverge
        _gd = {k: gradient_descent(A, y, k / L_smooth, GD_PASSES) for k in GD_STEPS}
        gd_ends = {k: gap(w[max(GD_PASSES)]) for k, w in _gd.items()}
    gd_ends = {k: (v if np.isfinite(v) else np.inf) for k, v in gd_ends.items()}
    gd_best = choose_step(gd_ends, "gradient descent")
    # The noise ball: where the constant-step methods hop over passes 1 to 10.
    _band = [g for m in ("sgd", "momentum", "adam") for c, g in curves[m].items()
             if NOISE_BAND_PASSES[0] <= c <= NOISE_BAND_PASSES[1]]  # fmt: skip
    curves["gd"] = {s: gap(w) for s, w in _gd[gd_best].items()}
    assert round(curves["average"][1], 4) == 0.0003 and round(curves["average"][LONG], 4) == 0.0001
    assert gd_best == 5.0 and round(gd_ends[1.0], 3) == 0.013 and round(gd_ends[8.0], 2) == 0.04
    assert round(curves["gd"][1], 2) == 0.18 and round(curves["gd"][100], 4) == 0.0021
    assert round(curves["gd"][100] / curves["average"][1]) == 7
    assert round(min(_band), 4) == 0.0003 and round(max(_band), 3) == 0.002  # the noise ball
    return curves, gd_best, gd_ends, kept_average


@app.cell
def _(A, F_star, PALETTE, cov, curves, gd_best, hessian, newton_gaps, np, plt, w_star, y):
    _base = y.mean()
    base_gap = -(_base * np.log(_base) + (1 - _base) * np.log(1 - _base)) - F_star
    # Estimation floor: the exact fit's own expected excess loss, (1/2) tr(H Cov), with the
    # bridge-clustered sandwich covariance; independent rows and a correct model give D / 2N.
    floor = 0.5 * float(np.trace(hessian(w_star, A) @ cov))
    _iid = A.shape[1] / (2 * len(A))
    assert round(base_gap, 3) == 0.038 and round(floor, 5) == round(_iid, 5) == 0.00006
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _styles = {"gd": (f"gradient descent, {gd_best:g}/L", "0.35", "-"), "sgd": ("SGD", PALETTE[0], "--"),
               "average": ("SGD, averaged", PALETTE[1], "-"), "adam": ("Adam", PALETTE[2], "-.")}  # fmt: skip
    for _m, (_label, _c, _ls) in _styles.items():
        _ax.plot(list(curves[_m]), list(curves[_m].values()), color=_c, ls=_ls, label=_label)
    _shown = [(i, g) for i, g in enumerate(newton_gaps) if g > 1e-6 and i > 0]
    _ax.plot(*zip(*_shown, strict=True), "kD", label="Newton, per step")
    _ax.axhline(base_gap, color="k", ls=":", lw=1)
    _ax.axhline(floor, color="k", ls=":", lw=1)
    _ax.set(xscale="log", yscale="log", xlabel="passes over the training inspections",
            ylabel="loss above the optimum (nats)", title="One pass beats 100 of gradient descent")  # fmt: skip
    _ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    _fig.tight_layout()
    _fig
    return base_gap, floor


@app.cell
def _(PALETTE, plt, sweeps):
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    for _i, _m in enumerate(("sgd", "average")):
        _lrs = list(sweeps[_m])
        _med = [sorted(g for g, _ in sweeps[_m][k])[2] for k in _lrs]  # median of 5
        _ax.plot(_lrs, _med, marker="os"[_i], color=PALETTE[_i], label=["SGD", "SGD, averaged"][_i])
    _ax.set(xscale="log", yscale="log", xlabel="step size",
            ylabel="loss above the optimum\nafter one pass", title="Too small crawls, too large bounces")  # fmt: skip
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict first** (from the slides): (1) one pass in FHWA's published order, state by state,
    at its own best step: better, the same, or worse than shuffled? (2) The shuffled pass is
    0.0017 above the optimum nationally: is its Illinois forecast within 1%, 5%, or 15% of
    Newton's 2,688? (3) After one averaged pass, how many of the 108 weights sit within two
    standard errors of Newton's?

    First, though, a units test of the derive step: the same model with ADT in vehicles a day,
    inputs unscaled.
    """)
    return


@app.cell
def _(
    NUMERIC,
    RAW_ADAM,
    RAW_STEPS,
    REPORT_SEEDS,
    Run,
    SELECT_SEEDS,
    choose_step,
    design,
    fit,
    loss,
    newton,
    np,
    runs,
    t,
    train,
):
    _y = t.loc[train, "drop"].to_numpy(float)

    def raw_gap(A: np.ndarray, F: float, run: Run) -> float:
        """Gap after one pass on inputs ``A``; a diverged run counts as the worst (inf)."""
        g = loss(fit(A, _y, run, 1, [1])[1], A, _y) - F
        return g if np.isfinite(g) else np.inf

    units = {}
    for _label, _scale in (("raw", False), ("scaled", True)):
        _A, _ = design(t, train, train, raw=True, scale=_scale)
        _w, _ = newton(_A, _y)
        _F = loss(_w, _A, _y)
        # Raw: each grid on the selection seeds (the best step inside it), then the report
        # seeds. Scaled: the main model's steps (no search), on the report seeds.
        _grids = {"sgd": RAW_STEPS, "adam": RAW_ADAM}
        if _scale:
            _grids = {"sgd": (runs["sgd"].lr,), "adam": (runs["adam"].lr,)}
        _adt = NUMERIC.index("log_adt") + 1  # ADT's column (linear in these units)
        units[_label] = {"max_input": np.abs(_A).max(), "adt_weight": _w[_adt],
                         "start": loss(np.zeros(_A.shape[1]), _A, _y) - _F}  # fmt: skip
        with np.errstate(over="ignore", invalid="ignore"):  # large steps overflow exp()
            for _kind, _grid in _grids.items():
                _per_seed = {
                    lr: [raw_gap(_A, _F, Run(_kind, lr, seed=s)) for s in SELECT_SEEDS]
                    for lr in _grid
                }
                _med = {lr: float(np.median(v)) for lr, v in _per_seed.items()}
                _lr = choose_step(_med, f"{_label} {_kind}") if len(_grid) > 1 else _grid[0]
                _rest = [raw_gap(_A, _F, Run(_kind, _lr, seed=s)) for s in REPORT_SEEDS]
                units[_label][_kind] = float(np.median(_rest))
                units[_label][f"{_kind}_lr"] = _lr
                units[_label][f"{_kind}_seeds"] = _per_seed[_lr] + _rest  # seeds 0-9
        del _A
    _raw, _scaled = units["raw"], units["scaled"]
    assert _raw["sgd_lr"] == 3e-8 and round(_raw["sgd"], 2) == 0.15
    # Every seed, 0 to 9, ends between 0.152 and 0.165: far above scaled, below the start.
    assert (round(min(_raw["sgd_seeds"]), 3), round(max(_raw["sgd_seeds"]), 3)) == (0.152, 0.165)
    assert round(_raw["start"], 2) == 0.40 and max(_raw["sgd_seeds"]) < _raw["start"]
    assert _raw["adam_lr"] == 1e-5 and round(_raw["adam"], 3) == 0.050
    assert round(_scaled["sgd"], 4) == 0.0017 and round(_scaled["adam"], 4) == 0.0016
    assert round(_raw["adam"] / _scaled["adam"]) == 31  # Adam: still 31 times worse
    assert _raw["max_input"] == 919_000 and round(1e8 * _raw["adt_weight"]) == -9
    print({k: {kk: (f"{vv:.3g}" if np.isscalar(vv) else [f"{x:.3g}" for x in vv])
               for kk, vv in v.items()} for k, v in units.items()})  # fmt: skip
    return


@app.cell
def _(
    A,
    A_te,
    PALETTE,
    Run,
    STEPS,
    TRACE_POINTS,
    choose_step,
    exact,
    expit,
    fit,
    forecasts,
    gap,
    median_gap,
    np,
    pd,
    plt,
    runs,
    state_te,
    t,
    train,
    w_star,
    y,
):
    # One pass in FHWA's published order (file, then state, then structure number). Its step
    # is chosen on the very pass it reports, which favours it: the ratios below are generous.
    file_gaps = {}
    for _m in ("sgd", "average"):
        file_gaps[_m] = {lr: gap(fit(A, y, Run("sgd", lr, _m == "average", "file"), 1, [1])[1])
                         for lr in STEPS[_m]}  # fmt: skip
    _best = {m: choose_step(g, f"published order, {m}") for m, g in file_gaps.items()}
    _ratio = {m: file_gaps[m][_best[m]] / median_gap[m] for m in _best}
    _marks = [k / TRACE_POINTS for k in range(1, TRACE_POINTS + 1)]
    _shuf = fit(A, y, runs["average"], 1, _marks)
    _file = fit(A, y, Run("sgd", _best["average"], True, "file"), 1, _marks)

    def _by_state(w):
        return pd.Series(expit(A_te @ w)).groupby(state_te).mean()

    _ex = _by_state(w_star)
    err_shuf = 100 * (_by_state(_shuf[1]) - _ex)
    err_file = 100 * (_by_state(_file[1]) - _ex)
    _worst = err_file.abs().sort_values(ascending=False).index[:4].tolist()
    # What the published-order errors follow: each state's training drop share above the
    # national one, or its place in the file (state codes sort as the file does)?
    _drop = t[train].groupby("state")["drop"].mean()
    _excess = (_drop - y.mean())[err_file.index].to_numpy()
    _corr = {
        "drop rate": np.corrcoef(err_file, _excess)[0, 1],
        "drop rate, shuffled": np.corrcoef(err_shuf[err_file.index], _excess)[0, 1],
        "place in the file": np.corrcoef(err_file, np.arange(len(err_file)))[0, 1],
    }
    print("correlation of each state's published-order error with:",
          {k: round(float(v), 2) for k, v in _corr.items()})  # fmt: skip
    assert _best == {"sgd": 0.3, "average": 1.0} and round(_ratio["average"]) == 20
    assert round(_ratio["sgd"]) == 3 and round(file_gaps["average"][1.0], 4) == 0.0059
    assert _worst[:3] == ["49", "23", "06"], _worst  # Utah, Maine, California
    assert round(err_file["49"]) == -11 and round(err_file["23"]) == -9
    assert round(err_file.abs().mean(), 1) == 2.5 and round(err_shuf.abs().mean(), 2) == 0.65
    assert round(err_file.max()) == 4  # the highest published-order bar: 4 points above Newton
    assert round(forecasts(_file[1])["illinois"]) == 2687 and round(exact["illinois"]) == 2688
    # The errors follow the drop rate, not the file position.
    assert {k: round(float(v), 2) for k, v in _corr.items()} == {
        "drop rate": -0.83,
        "drop rate, shuffled": -0.71,
        "place in the file": 0.12,
    }
    _fig, (_ax, _bx) = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax.plot(_marks, [gap(_shuf[c]) for c in _marks], "--", color=PALETTE[0], label="shuffled")
    _ax.plot(_marks, [gap(_file[c]) for c in _marks], color=PALETTE[1], label="published order")
    _ax.set(yscale="log", xlabel="fraction of the pass", ylabel="loss above the optimum")
    _ax.legend()
    _x = np.arange(len(err_file))
    _bx.bar(_x - 0.2, err_shuf.values, 0.4, color=PALETTE[0])
    _bx.bar(_x + 0.2, err_file.values, 0.4, color=PALETTE[1])
    _bx.set(xlabel="state, in file order", ylabel="forecast minus Newton's (points)")
    _fig.suptitle("Published order never settles")
    _fig.tight_layout()
    _fig
    return (err_shuf,)


@app.cell
def _(A, LONG, err_shuf, expit, hessian, kept_average, names, np, pd, t, train, w_star, y):
    # Sandwich covariance of Newton's weights, scores summed by bridge (two pairs each).
    _H_inv = np.linalg.inv(hessian(w_star, A))
    _scores = A * (expit(A @ w_star) - y)[:, None]
    _bridge = (t.loc[train, "state"] + "|" + t.loc[train, "structure"]).to_numpy()
    _by_bridge = pd.DataFrame(_scores).groupby(_bridge, sort=False).sum().to_numpy()
    cov = _H_inv @ (_by_bridge.T @ _by_bridge / len(A)) @ _H_inv / len(A)
    _se = np.sqrt(np.diag(cov))
    _states = np.array([n.startswith("state=") for n in names])
    _others = ~_states & (np.arange(len(names)) > 0)
    _z1 = (kept_average[1] - w_star) / _se
    _z10 = (kept_average[LONG] - w_star) / _se
    _epoxy = names.index("protection=1")
    assert round(np.median(np.abs(_z1[_states])), 1) == 8.3
    assert round(np.median(np.abs(_z1[_others])), 1) == 1.1
    assert names[np.argmax(np.abs(_z1))] == "state=49" and round(np.abs(_z1).max()) == 12
    assert round(np.median(np.abs(_z10)), 1) == 1.5 and round(np.mean(np.abs(_z10) > 2), 2) == 0.44
    assert round(kept_average[1][_epoxy], 3) == -0.158 and round(w_star[_epoxy], 3) == -0.145
    assert round(_se[_epoxy], 3) == 0.012 and round(np.exp(w_star[_epoxy]), 2) == 0.87
    # The flattest direction's five states: their own weights against their forecasts.
    _zst = pd.Series(np.abs(_z1[_states]), np.asarray(names)[_states])
    _zrank = _zst.rank(ascending=False, method="min")
    _erank = err_shuf.abs().rank(ascending=False, method="min")
    flat_states = pd.DataFrame(
        {
            c: {"|z|": _zst[f"state={c}"], "rank by |z| (of 51)": _zrank[f"state={c}"],
                "forecast error (pts)": err_shuf[c], "rank by |error| (of 52)": _erank[c]}
            for c in ("11", "10", "32", "72", "44")
        }
    ).T  # fmt: skip
    # Their weights are among the closest to Newton's; four of their forecasts among the worst.
    assert flat_states["rank by |z| (of 51)"].astype(int).to_dict() == {
        "11": 50, "10": 49, "32": 48, "72": 51, "44": 45,
    }  # fmt: skip
    assert flat_states["rank by |error| (of 52)"].astype(int).to_dict() == {
        "11": 3, "10": 6, "32": 11, "72": 10, "44": 21,
    }  # fmt: skip
    print(flat_states.round(2))
    print(f"after one averaged pass: state weights a median {np.median(np.abs(_z1[_states])):.1f} "
          f"SEs from Newton's, other weights {np.median(np.abs(_z1[_others])):.1f}")  # fmt: skip
    return (cov,)


@app.cell
def _(mo):
    mo.accordion(
        {
            "What the break-it tests showed": mo.md(
                r"""
                1. **Published order is worse:** at their own best steps, 20 times above the
                   shuffled pass for averaged SGD, 3 times for the last iterate (and its steps
                   were chosen on the very pass they report, which favours it). Consecutive
                   batches come from one state; the assumption of random batches fails. Per
                   state the mean forecast error is 2.5 points, against 0.65 shuffled; Utah is
                   11 points low and Maine 9. The errors follow each state's drop rate
                   ($r$ = −0.83), not its place in the file ($r$ = 0.12): every state's forecast
                   is pulled toward the national rate. Illinois lands at 2,687 by luck.
                2. **Illinois is up to 6% off:** across seeds 5 to 9, one plain SGD pass
                   forecasts Illinois anywhere from 2,614 to 2,838 decks, against Newton's
                   2,688, while the national loss is within 0.0017.
                3. **Few weights:** after one averaged pass the state weights sit a median 8.3
                   standard errors from Newton's; the other weights, 1.1. The five states of
                   the flattest direction behind $\kappa$ = 11,507 (DC, Delaware, Nevada,
                   Puerto Rico, Rhode Island) are the exception: their weights are among the
                   closest to Newton's in standard errors (45th to 51st of 51), whose few rows
                   make wide, yet four of them have among the 11 worst forecasts. The loss
                   barely notices these directions; a state's forecast does.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    **When SGD is the wrong tool:** few weights (at $D$ = 108 Newton is exact in 8 steps);
    weights you will quote (a state comparison, an epoxy effect: converge, then report standard
    errors); rows you cannot shuffle. Its home is millions of rows and weights with no Hessian:
    Lecture 14's networks.

    **Vote: what do you send IDOT?** It is 2022; IDOT asks how many Illinois decks will lose a
    rating point by their next inspection. **A.** One SGD pass: 2,655 (median of five seeds).
    **B.** One averaged-SGD pass: 2,601 (seed 5). **C.** Newton: 2,688 ± 33, with a note that
    the drop rate fell from 11.1% to 9.7% between the two training cycles. **D.** Averaged SGD
    in published order: 2,687, beside Newton's.
    """)
    return


@app.cell
def _(A_te, ILLINOIS, cov, expit, illinois, kept_average, names, np, state_te, w_star, y_te):
    _il = state_te == ILLINOIS
    _p = expit(A_te[_il] @ w_star)
    _g = (A_te[_il] * (_p * (1 - _p))[:, None]).sum(0)  # d(sum p)/dw
    budget = {
        "Newton's forecast": _p.sum(),
        "what happened": y_te[_il].sum(),
        "estimation, 1 SE": np.sqrt(_g @ cov @ _g),
        "chance, 1 SD": np.sqrt((_p * (1 - _p)).sum()),
    }
    _miss = budget["Newton's forecast"] - budget["what happened"]
    _both = np.hypot(budget["estimation, 1 SE"], budget["chance, 1 SD"])
    # Vote B's shortfall, restored weight by weight: Illinois's own weight, the intercept, both.
    _w1 = kept_average[1]
    _col = names.index(f"state={ILLINOIS}")
    restored = {}
    for _key, _cols in (("Illinois's weight", [_col]), ("intercept", [0]), ("both", [0, _col])):
        _w = _w1.copy()
        _w[_cols] = w_star[_cols]
        restored[_key] = illinois(_w)
    assert round(budget["Newton's forecast"]) == 2688 and budget["what happened"] == 2359
    assert round(budget["estimation, 1 SE"]) == 33 and round(budget["chance, 1 SD"]) == 44
    assert round(_miss / budget["chance, 1 SD"], 1) == 7.4 and round(_miss / _both, 1) == 6.0
    # Vote B (seed 5): 2,601, 86 low. Illinois's weight and the intercept err in opposite ways.
    assert round(illinois(_w1)) == 2601
    assert {k: round(v) for k, v in restored.items()} == {
        "Illinois's weight": 3387, "intercept": 1984, "both": 2632,
    }  # fmt: skip
    print({k: round(float(v)) for k, v in budget.items()},
          f"miss {_miss:.0f}: {_miss / budget['chance, 1 SD']:.1f} chance SDs, "
          f"{_miss / _both:.1f} with the estimation error too")  # fmt: skip
    print("averaged pass", round(illinois(_w1)), "restored:",
          {k: round(v) for k, v in restored.items()})  # fmt: skip
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer: C, and the note matters most": mo.md(
                r"""
                A scatters from 2,614 to 2,838 across seeds. B is steady (2,601 to 2,612 over
                five seeds) but 86 low on seed 5, 2.6 standard errors: its intercept and
                Illinois's weight have not converged together. Restoring Illinois's weight
                alone would overshoot to 3,387; the intercept alone would drop it to 1,984; both
                give 2,632. D matches by luck, as Utah's 11 points show. At 108 weights Newton
                is exact in seconds.

                Then 2024 arrived: 2,359 decks dropped. Newton's forecast missed by 329: 7.4
                times the chance spread of 44, and 6.0 times chance and its estimation error of
                33 combined. Each cycle has dropped less (11.1%, 9.7%, 8.7%), nationally too:
                Newton expects 10.1% of the test cycle's decks to drop, and 8.7% did. With an
                exact optimizer, the error left is the drift of Lecture 11: spend the effort
                there, not on more passes.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    **Exit question.** A colleague streams the 2024 file county by county through one pass of
    Adam, gets a test loss within 0.002 of Newton's, and wants to publish a ranking of counties
    by their weights. What do you check first, and why?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Two things. The order: county-sorted batches bias every step, so shuffle and
                refit. The weights: a close loss says nothing about the flat directions, and
                county weights are exactly those; fit to convergence (Newton here) and rank with
                standard errors.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **No shuffle:** a loop over `range(N)` in file order looks like SGD and lowers the loss;
      only a per-group check shows the bias.
    - **Step size tuned on the test cycle:** choose $\eta$ on the training loss (or a
      validation split), never on the cycle you report.
    - **Unscaled inputs:** ADT in vehicles a day needs a step millions of times smaller than
      the scaled model's ($3\times10^{-8}$ against 0.3), and one pass then ends 88 times
      further from the optimum; Adam copes better than SGD but still ends 31 times worse than
      scaled inputs. Scale with training statistics only.
    - **Sentinel codes:** an ADT of 999,999 or a deck width of 999.9 m is a code, not a
      measurement. Look at each input's largest values before fitting.
    - **Judging convergence by the loss:** report weights only after the optimizer has
      converged; compare with an exact fit when one is affordable.
    - **The last iterate of a constant step:** it hops; average the iterates or decay the step.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper

    - **Forsyth §2.1.3:** stochastic gradient descent for the SVM, steplength schedules by
      epoch and by blocks of steps, and why there is no cheap convergence test (here, Newton's
      optimum is the test).
    - **Forsyth §16.4.4:** gradient scaling: momentum, Adagrad, RMSprop, Adam. His momentum puts
      $\eta$ inside the velocity ($v\leftarrow\mu v+\eta\nabla$, $\theta\leftarrow\theta-v$); ours
      multiplies it outside. His Adam calls the first moment $v$ and the second $c$.
    - **Bottou, Curtis & Nocedal (2018), §4:** convergence of SGD with fixed and diminishing
      steps, and the noise ball.
    - **Bottou & Bousquet (2008):** approximation, estimation, and optimization error.
    - **Exercise.** Show that $\mathbb{E}[\mathbf{g}_B]=\nabla F$ for batches drawn with
      replacement, and that the variance is $\sigma^2/B$.
    - **Exercise.** Replace the shuffle with a shuffle buffer of $K$ rows read in file order.
      How large must $K$ be for one averaged pass to come within twice the shuffled gap?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A random batch's gradient is right on average; shuffling makes the batches random.
    - Constant steps stall in a noise ball; average the iterates or shrink the step.
    - One averaged pass came 7 times closer than 100 passes of tuned gradient descent.
    - The loss settles before the weights: converge before you quote a weight or a state.

    **Next:** multilayer perceptrons and backpropagation: a neural surrogate of Lecture 12's
    simulator.
    """)
    return


if __name__ == "__main__":
    app.run()
