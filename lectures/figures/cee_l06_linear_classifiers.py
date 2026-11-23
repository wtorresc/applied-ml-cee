"""Figures and numbers for CEE Lecture 6: linear classifiers and costs (Illinois bridges).

Run::

    pixi run -e aml-env python figures/cee_l06_linear_classifiers.py \
        --out build/fig/cee_l06_linear_classifiers

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of one bridge, for the TikZ diagram). Data: ``data/derived/nbi_il_bridges.csv`` from
``scripts/fetch_nbi.sh`` (FHWA National Bridge Inventory, Illinois, 2014-2024), loaded with
``bridges.load_bridges``.

The question: of the bridges that are not poor today (lowest rating 5 or more), which will be
poor in the inventory four years later? A flagged bridge gets preventive work and a closer look
(cost 1); a missed bridge, poor four years later, costs ``R`` times as much. An engineer in 2021
trains on the 2014 -> 2018 transitions, validates on 2016 -> 2020, and flags the 2020 bridges;
the 2024 inventory is the test set, used once.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import aml_style
import title_art
from bridges import FEATURES, POOR, find_twins, load_bridges
from common import parse_out_dir, pct, rnd, seed_all, tex_num, write_numbers, write_tex_macros

SEED: Final[int] = 0
HORIZON: Final[int] = 4  # years from the decision to the inventory that judges it
# Lecture 27 imports START, R, INPUTS and OWNERS: they stay a dict, an int, a list (a DataFrame
# is indexed with it; a tuple would be one pandas key) and a dict.
START: Final[dict[str, int]] = {"train": 2014, "val": 2016, "test": 2020}
LATER_START: Final[int] = 2018  # 2018 -> 2022: a later period, checked in hindsight only
R: Final[int] = 20  # a missed bridge costs R flags (the lecture's working value)
RATIOS: Final[tuple[int, ...]] = (10, 20, 50)
FLAG_SHARE: Final[float] = 0.10  # a fixed budget: flag this share of bridges
COMPONENTS: Final[tuple[str, ...]] = ("deck", "superstructure", "substructure")
INPUTS: Final[list[str]] = [*FEATURES, *COMPONENTS, "rating"]  # 12 inventory inputs + 4 ratings
MATERIALS: Final[tuple[str, ...]] = ("concrete", "steel", "prestressed", "timber")  # vs "other"
TRACE_BASELINE: Final[str] = "prestressed"  # the most common material: the trace's baseline
OUTCOMES: Final[tuple[int, ...]] = (POOR, 5, 6, 7, 8)  # ordinal cuts: P(y <= j)
C_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0)  # L2 penalties tried on validation
REF_RTOL: Final[float] = 1e-6  # Newton against scikit-learn's Newton-Cholesky
LBFGS_TIGHT_TOL: Final[float] = 1e-8  # L-BFGS at a tolerance that reaches the optimum
COST_GRID: Final[tuple[float, float]] = (0.005, 0.7)  # thresholds drawn on the decisive figure
BOOT_DRAWS: Final[int] = 2000  # county-cluster bootstrap draws
CI: Final[tuple[float, float]] = (2.5, 97.5)  # percentiles of a 95% interval
CHAMPAIGN: Final[str] = "019"  # county code (FIPS order); structure numbers start with 010
CITY: Final[str] = "04"  # Item 22 owner: city or municipal highway agency
OWNERS: Final[dict[str, str]] = {"01": "state", "02": "county", "03": "township", "04": "city"}
# The attempt's six bridges, chosen by rule in attempt(); short names for the slide only.
SHOW_NAMES: Final[dict[str, tuple[str, str]]] = {
    "010314700000000": ("Duncan Rd", "Phinney Branch"),
    "010601000000000": ("Duncan Rd", "Copper Slough"),
    "010448400000000": ("Staley Rd", "Phinney Branch"),
    "010435100000000": ("FAU 7154", "Copper Slough"),
    "010620200000000": ("Mattis Ave", "Phinney Branch"),
    "010612800000000": ("Broadway", "Saline Branch"),
}
TRACED: Final[str] = "010601000000000"  # Duncan Rd over Copper Slough, traced in the diagram


# ---------------------------------------------------------------------------------------------
# Data


def transitions(df: pd.DataFrame, start: int) -> pd.DataFrame:
    """Bridges not poor in ``start`` and their lowest rating ``HORIZON`` years later.

    Args:
        df: Bridge-year records from :func:`bridges.load_bridges`.
        start: The NBI year of the decision.

    Returns:
        One row per bridge rated above ``POOR`` in ``start`` that is still in the inventory
        ``HORIZON`` years later: the ``start`` record, ``rating_next``, ``poor_next`` (0/1), and
        ``deck_missing``. A missing deck rating (no deck) is filled with the lower of the
        superstructure and substructure ratings.
    """
    key = ["structure", "nbi_year"]
    later = df[[*key, "rating"]].rename(columns={"rating": "rating_next"})
    later["nbi_year"] -= HORIZON
    now = df[df["nbi_year"] == start]
    out = now.merge(later, on=key, how="inner", validate="one_to_one")
    out = out[out["rating"] > POOR].reset_index(drop=True)
    out["deck_missing"] = out["deck"].isna()
    out["deck"] = out["deck"].fillna(out[["superstructure", "substructure"]].min(axis=1))
    out["poor_next"] = (out["rating_next"] <= POOR).astype(int)
    assert out[INPUTS].notna().all().all(), "missing inputs"
    return out


def encode(d: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """The 16 inputs, standardized with the training scaler (Lecture 27 calls it too).

    Args:
        d: Transitions with every column of ``INPUTS``.
        scaler: A ``StandardScaler`` fitted on the training transitions' ``INPUTS``.

    Returns:
        The standardized inputs ``z``, shape ``(len(d), len(INPUTS))``, float64.
    """
    Z = scaler.transform(d[INPUTS].to_numpy(np.float64))
    assert Z.shape == (len(d), len(INPUTS)), Z.shape
    return Z


# ---------------------------------------------------------------------------------------------
# Models (the code slide's listing, and the ordinal model)


def fit_logistic(
    Z: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-10,
    max_steps: int = 50,
    path: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Maximum-likelihood logistic regression by Newton's method (the slide's listing).

    Args:
        Z: Standardized inputs, shape ``(n, d)``.
        y: Labels in {0, 1}, shape ``(n,)``.
        tol: Stop when no weight moves more than this.
        max_steps: Give up after this many steps.
        path: If given, the start and every iterate are appended to it (for the figure).

    Returns:
        Weights ``[w0, w]`` of shape ``(d + 1,)``, and the number of Newton steps taken.

    Raises:
        RuntimeError: If Newton's method has not converged after ``max_steps``. It does not
            converge when the data are (quasi-)separated and the unpenalized optimum is at
            infinity: on the 2016 -> 2020 transitions, none of the 13 "other"-material bridges
            turned poor, so a refit on that period raises (``post_hoc``); add a ridge penalty.
    """
    A = np.column_stack([np.ones(len(Z)), Z])  # intercept column
    w = np.zeros(A.shape[1])
    if path is not None:
        path.append(w.copy())
    for step in range(1, max_steps + 1):
        p = expit(A @ w)  # sigma(s), s = A w
        g = A.T @ (p - y)  # gradient of the NLL
        H = A.T @ (A * (p * (1 - p))[:, None])  # Hessian A^T S A
        delta = np.linalg.solve(H, g)  # Newton step
        w -= delta
        if path is not None:
            path.append(w.copy())
        if np.abs(delta).max() < tol:
            return w, step
    raise RuntimeError(f"Newton's method did not converge in {max_steps} steps")


def predict_proba(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """P(poor in four years) for standardized inputs (Lecture 27 calls it too).

    Args:
        w: Weights ``[w0, w]`` from :func:`fit_logistic`, shape ``(d + 1,)``.
        Z: Standardized inputs, shape ``(n, d)``.

    Returns:
        ``sigma(w0 + Z w)``, shape ``(n,)``.
    """
    return expit(w[0] + Z @ w[1:])


def fit_unpenalized(X: np.ndarray, y: np.ndarray, **kw: object) -> LogisticRegression:
    """scikit-learn's unpenalized logistic regression (``C=np.inf``), Newton-Cholesky by default.

    scikit-learn 1.8 deprecates ``penalty=None`` in favor of ``C=np.inf``, then warns that
    ``C=np.inf`` sets ``penalty=None``. The fit is unpenalized either way, so that one message is
    silenced here, for this fit only.

    Args:
        X: Inputs, shape ``(n, d)``.
        y: Labels in {0, 1}, shape ``(n,)``.
        **kw: Other ``LogisticRegression`` options (``solver``, ``tol``, ``max_iter``).

    Returns:
        The fitted model.
    """
    kw = {"solver": "newton-cholesky"} | kw
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Setting penalty=None will ignore", category=UserWarning
        )
        return LogisticRegression(C=np.inf, **kw).fit(X, y)  # type: ignore[arg-type]


def _thresholds(a: np.ndarray) -> np.ndarray:
    """Increasing cutpoints from unconstrained parameters.

    Args:
        a: The first cutpoint, then the logs of the gaps between cutpoints.

    Returns:
        The cutpoints, strictly increasing, shape ``a.shape``.
    """
    return np.concatenate([[a[0]], a[0] + np.cumsum(np.exp(a[1:]))])


def fit_ordinal(Z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative-logit (proportional-odds) model of the later rating.

    ``P(r <= OUTCOMES[j] | z) = sigma(theta_j - w.z)``, one weight vector for every cut.

    Args:
        Z: Standardized inputs, shape ``(n, d)``.
        r: Later ratings, integers 0-9.

    Returns:
        Weights ``w`` of shape ``(d,)`` and increasing cutpoints ``theta`` of shape
        ``(len(OUTCOMES),)``.

    Raises:
        RuntimeError: If the optimizer fails.
    """
    cls = np.searchsorted(OUTCOMES, r, side="left")  # 0 = poor, ..., 5 = rated 9
    d = Z.shape[1]

    def nll(params: np.ndarray) -> float:
        """Negative log-likelihood of the ordinal model.

        Args:
            params: The weights, then the cutpoints' unconstrained parameters.

        Returns:
            The negative log-likelihood of the later ratings.
        """
        w, theta = params[:d], _thresholds(params[d:])
        cdf = expit(theta[None, :] - (Z @ w)[:, None])
        cdf = np.column_stack([np.zeros(len(Z)), cdf, np.ones(len(Z))])
        prob = cdf[np.arange(len(Z)), cls + 1] - cdf[np.arange(len(Z)), cls]
        return float(-np.log(np.clip(prob, 1e-300, None)).sum())

    x0 = np.concatenate([np.zeros(d), [-3.0], np.zeros(len(OUTCOMES) - 1)])
    # Tight tolerances: the default ones stop on a flat direction, 2e-3 from the optimum.
    res = minimize(
        nll, x0, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-8}
    )
    if not res.success:
        raise RuntimeError(f"ordinal fit failed: {res.message}")
    return res.x[:d], _thresholds(res.x[d:])


# ---------------------------------------------------------------------------------------------
# Metrics


def expected_cost(flag: np.ndarray, y: np.ndarray, r: float = R) -> float:
    """Cost per bridge: 1 per false alarm, ``r`` per miss (Lecture 27 calls it too).

    Args:
        flag: Boolean decisions, shape ``(n,)``: True flags the bridge.
        y: Outcomes in {0, 1}, shape ``(n,)``: 1 if the bridge turned poor.
        r: The cost of a miss, in false alarms.

    Returns:
        The mean cost per bridge.
    """
    assert flag.dtype == bool, flag.dtype  # ~ on integers would be a bitwise not
    return float((r * ((~flag) & (y == 1)).sum() + (flag & (y == 0)).sum()) / len(y))


def caught_at_share(score: np.ndarray, y: np.ndarray, share: float = FLAG_SHARE) -> float:
    """Share of the bridges that turned poor among the top ``share`` by score.

    Args:
        score: Any score that ranks bridges, higher meaning riskier, shape ``(n,)``.
        y: Outcomes in {0, 1}, shape ``(n,)``.
        share: The share of bridges flagged, from the top; ties keep file order.

    Returns:
        The share of the bridges that turned poor among those flagged.
    """
    top = np.argsort(-score, kind="stable")[: int(round(share * len(y)))]
    return float(y[top].sum() / y.sum())


# ---------------------------------------------------------------------------------------------
# Experiments


def data_summary(df: pd.DataFrame, sets: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Counts for the question slide.

    Args:
        df: Bridge-year records from :func:`bridges.load_bridges`.
        sets: The training, validation, and test transitions, keyed as ``START``.

    Returns:
        Per period: years, bridges, the number and share that turned poor, the share dropped
        from the inventory, and the bridges with no deck rating; the training shares by rating;
        and the number of inputs.
    """
    out: dict[str, object] = {}
    for name, d in sets.items():
        start = START[name]
        now = df[(df["nbi_year"] == start) & (df["rating"] > POOR)]
        out[name] = {
            "start": start,
            "end": start + HORIZON,
            "bridges": int(len(d)),
            "turned_poor": int(d["poor_next"].sum()),
            "poor_pct": pct(d["poor_next"].mean()),
            "dropped_pct": pct(1 - len(d) / len(now)),
            "deck_missing": int(d["deck_missing"].sum()),
        }
    tr = sets["train"]
    by = tr.groupby("rating")["poor_next"]
    out["by_rating_train"] = {
        str(int(k)): {
            "bridges": int(by.size()[k]),
            "poor": int(by.sum()[k]),
            "pct": pct(by.mean()[k]),
        }
        for k in by.size().index
    }
    out["n_inputs"] = len(INPUTS)
    return out


def log_odds_figure(tr: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Empirical log-odds of turning poor by lowest rating (training), with a fitted line.

    Args:
        tr: The training transitions.
        out_dir: The figure directory.

    Returns:
        The one-input slope per rating point, the bridges rated 9 (none turned poor), and the
        empirical log-odds by rating.
    """
    by = tr.groupby("rating")["poor_next"].agg(["sum", "size"])
    by = by[by["sum"] > 0]
    rating = by.index.to_numpy(float)
    rate = (by["sum"] / by["size"]).to_numpy()
    logit = np.log(rate / (1 - rate))
    lr = fit_unpenalized(tr[["rating"]].to_numpy(float), tr["poor_next"].to_numpy())
    grid = np.linspace(4.8, 9.2, 50)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(rating, logit, "o", color=aml_style.PALETTE[0], markersize=9, label="training data")
    ax.plot(
        grid,
        lr.intercept_[0] + lr.coef_[0, 0] * grid,
        "--",
        color=aml_style.PALETTE[1],
        label="straight line in log-odds",
    )
    for x, v, p in zip(rating, logit, rate, strict=True):
        ax.annotate(
            f"{100 * p:.1f}%",
            (x, v),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=12,
            color=aml_style.INK,
        )
    ax.set_xlabel("lowest rating now")
    ax.set_ylabel("log-odds of poor in 4 years")
    ax.set_xticks([5, 6, 7, 8, 9])
    ax.legend(loc="upper right")
    aml_style.save(fig, out_dir, "log_odds")
    zero = int(tr[(tr["rating"] == 9)]["poor_next"].sum())
    return {
        "slope_per_point": rnd(lr.coef_[0, 0], 3),
        "rated_9_turned_poor": zero,
        "rated_9_bridges": int((tr["rating"] == 9).sum()),
        "empirical_logit": {str(int(k)): rnd(v, 3) for k, v in zip(rating, logit, strict=True)},
    }


def verify(Z: np.ndarray, y: np.ndarray, Zv: np.ndarray, yv: np.ndarray) -> dict[str, object]:
    """Newton's method against scikit-learn; an L2 penalty against none on validation.

    Args:
        Z: Standardized training inputs, shape ``(n, 16)``.
        y: Training outcomes in {0, 1}.
        Zv: Standardized validation inputs.
        yv: Validation outcomes in {0, 1}.

    Returns:
        Newton's steps and its agreement with scikit-learn's Newton-Cholesky and L-BFGS; the
        validation log loss without and with each L2 penalty in ``C_GRID``; the weights; and the
        standard errors of the material weights (each measured against the "other" bridges).

    Raises:
        RuntimeError: If the best penalty sits at an edge of ``C_GRID``.
    """
    w, steps = fit_logistic(Z, y)
    ref = fit_unpenalized(Z, y, tol=1e-12, max_iter=200)
    ref_w = np.concatenate([ref.intercept_, ref.coef_[0]])
    rel = float(np.max(np.abs(w - ref_w) / (np.abs(ref_w) + 1e-12)))
    assert np.allclose(w, ref_w, rtol=REF_RTOL, atol=0), rel
    lbfgs = fit_unpenalized(Z, y, solver="lbfgs", max_iter=10_000)  # scikit-learn's default
    lb_w = np.concatenate([lbfgs.intercept_, lbfgs.coef_[0]])
    tight = fit_unpenalized(Z, y, solver="lbfgs", max_iter=100_000, tol=LBFGS_TIGHT_TOL)
    tight_w = np.concatenate([tight.intercept_, tight.coef_[0]])
    penalty = {}
    for C in C_GRID:
        pv = LogisticRegression(C=C, solver="newton-cholesky").fit(Z, y).predict_proba(Zv)[:, 1]
        penalty[str(C)] = round(float(log_loss(yv, pv)), 5)
    best = int(np.argmin(list(penalty.values())))
    if not 0 < best < len(C_GRID) - 1:
        raise RuntimeError(f"best C {C_GRID[best]} is at an edge of {C_GRID}; widen the grid")
    none = round(float(log_loss(yv, predict_proba(w, Zv))), 5)
    # Standard errors from the inverse Hessian at the optimum: the material dummies are measured
    # against the few "other" bridges, so they are uncertain, and L-BFGS stops on them.
    A = np.column_stack([np.ones(len(Z)), Z])
    q = predict_proba(w, Z)
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ (A * (q * (1 - q))[:, None]))))
    return {
        "material_weight": {m: rnd(w[1 + INPUTS.index(m)], 3) for m in MATERIALS},
        "material_se": {m: rnd(se[1 + INPUTS.index(m)], 3) for m in MATERIALS},
        "best_C": C_GRID[best],
        "steps": steps,
        "max_rel_diff": float(f"{rel:.1e}"),
        "lbfgs_iterations": int(lbfgs.n_iter_[0]),
        "lbfgs_loss_diff": float(
            f"{log_loss(y, lbfgs.predict_proba(Z)[:, 1]) - log_loss(y, predict_proba(w, Z)):.0e}"
        ),
        "lbfgs_tol_1e8_max_abs_diff": float(f"{np.max(np.abs(w - tight_w)):.0e}"),
        "lbfgs_max_abs_diff": float(f"{np.max(np.abs(w - lb_w)):.1e}"),
        "lbfgs_max_rel_diff": float(f"{np.max(np.abs(w - lb_w) / (np.abs(w) + 1e-12)):.1e}"),
        "lbfgs_max_p_diff": float(
            f"{np.max(np.abs(predict_proba(w, Z) - lbfgs.predict_proba(Z)[:, 1])):.1e}"
        ),
        "val_log_loss_none": none,
        "val_log_loss_l2": penalty,
        "l2_best_gain": round(none - min(penalty.values()), 5),
        "weights": {n: rnd(v, 3) for n, v in zip(["intercept", *INPUTS], w, strict=True)},
    }


def thresholds(p: np.ndarray, y: np.ndarray, rating: np.ndarray) -> dict[str, object]:
    """Accuracy, the 0.5 threshold, the cost thresholds, and the rules, on the test period.

    Args:
        p: The model's P(poor) for the test bridges.
        y: Test outcomes in {0, 1}.
        rating: The test bridges' lowest rating in the decision year.

    Returns:
        Accuracy and flags at 0.5; for each r in ``RATIOS``, the threshold 1/(1 + r), what it
        flags, catches, and costs, the best threshold in hindsight, and the baselines' costs;
        the rule "flag every 5" and the rule "flag every 5 or 6".
    """
    out: dict[str, object] = {
        "accuracy_flag_none": pct(1 - y.mean(), 2),
        "accuracy_at_half": pct(((p > 0.5) == y).mean(), 2),
        "flagged_at_half": int((p > 0.5).sum()),
        "caught_at_half": int(((p > 0.5) & (y == 1)).sum()),
        "max_p": rnd(p.max(), 3),
        "turned_poor": int(y.sum()),
        "bridges": int(len(y)),
    }
    per_r = {}
    for r in RATIOS:
        t = 1 / (1 + r)
        flag = p > t
        grid = np.unique(np.concatenate([np.linspace(0.002, 0.6, 600), [t]]))
        costs = np.array([expected_cost(p > s, y, r) for s in grid])
        per_r[str(r)] = {
            "threshold": round(t, 4),
            "flagged_pct": pct(flag.mean()),
            "caught": int((flag & (y == 1)).sum()),
            "missed": int(((~flag) & (y == 1)).sum()),
            "caught_pct": pct((flag & (y == 1)).sum() / y.sum()),
            "cost": rnd(expected_cost(flag, y, r), 3),
            "best_threshold_hindsight": rnd(grid[int(np.argmin(costs))], 3),
            "within_002_of_best": [
                rnd(grid[costs <= costs.min() + 0.02].min(), 3),
                rnd(grid[costs <= costs.min() + 0.02].max(), 3),
            ],
            "best_cost_hindsight": rnd(costs.min(), 3),
            "cost_flag_none": rnd(expected_cost(np.zeros_like(flag), y, r), 3),
            "cost_flag_all": rnd(expected_cost(np.ones_like(flag), y, r), 3),
            "cost_rule": rnd(expected_cost(rating == 5, y, r), 3),
            "cost_at_half": rnd(expected_cost(p > 0.5, y, r), 3),
        }
    out["per_ratio"] = per_r
    rule = rating == 5
    out["rule"] = {
        "flagged_pct": pct(rule.mean()),
        "caught": int((rule & (y == 1)).sum()),
        "caught_pct": pct((rule & (y == 1)).sum() / y.sum()),
        "model_caught_pct_same_share": pct(caught_at_share(p, y, rule.mean())),
    }
    rule56 = rating <= 6
    out["rule_5_6"] = {
        "flagged_pct": pct(rule56.mean()),
        "caught_pct": pct((rule56 & (y == 1)).sum() / y.sum()),
    }
    return out


def county_draws(num: np.ndarray, den: np.ndarray, county: np.ndarray, stream: int) -> np.ndarray:
    """Bootstrap draws of ratios of sums, resampling whole counties.

    Bridges in one county share inspectors, budgets, and weather, so they are not independent;
    each draw resamples the counties with replacement and keeps all of a county's bridges.

    Args:
        num: Per-bridge numerators, shape ``(n, k)``.
        den: Per-bridge denominators, shape ``(n, k)``.
        county: Each bridge's county code, shape ``(n,)``.
        stream: Picks the generator's stream (with ``SEED``), so every interval has its own.

    Returns:
        ``sum(num) / sum(den)`` per column for each of ``BOOT_DRAWS`` draws, shape
        ``(BOOT_DRAWS, k)``.

    Raises:
        RuntimeError: If a draw has a zero denominator (a ratio with no bridge behind it).
    """
    codes, inv = np.unique(county, return_inverse=True)
    S = np.zeros((len(codes), num.shape[1]))
    D = np.zeros_like(S)
    np.add.at(S, inv, num)
    np.add.at(D, inv, den)
    idx = np.random.default_rng((SEED, stream)).integers(0, len(codes), (BOOT_DRAWS, len(codes)))
    num_d, den_d = S[idx].sum(axis=1), D[idx].sum(axis=1)
    assert num_d.shape == (BOOT_DRAWS, num.shape[1]), num_d.shape
    if (den_d == 0).any():
        raise RuntimeError("a bootstrap draw has no bridge behind a ratio")
    return num_d / den_d


def model_vs_rule(
    p: np.ndarray, y: np.ndarray, rating: np.ndarray, county: np.ndarray
) -> dict[str, object]:
    """The model's cost minus the rule's ("flag every 5"), with a county-cluster interval.

    Args:
        p: The model's P(poor) for the test bridges.
        y: Test outcomes in {0, 1}.
        rating: The test bridges' lowest rating.
        county: The test bridges' county codes.

    Returns:
        For r = ``R`` and the largest of ``RATIOS``: the difference in cost per bridge and its
        95% interval over ``BOOT_DRAWS`` county resamples.
    """
    out: dict[str, object] = {}
    rule = rating == 5
    for r in (R, max(RATIOS)):
        flag = p > 1 / (1 + r)
        cost_model = r * ((~flag) & (y == 1)) + (flag & (y == 0))  # each bridge's cost
        cost_rule = r * ((~rule) & (y == 1)) + (rule & (y == 0))
        per_bridge = (cost_model - cost_rule).astype(np.float64)
        assert np.isclose(per_bridge.mean(), expected_cost(flag, y, r) - expected_cost(rule, y, r))
        draws = county_draws(per_bridge[:, None], np.ones((len(y), 1)), county, r)[:, 0]
        out[str(r)] = {
            "diff": rnd(per_bridge.mean(), 3),
            "ci": [rnd(v, 3) for v in np.percentile(draws, CI)],
        }
    return out


def cost_threshold_figure(
    p: np.ndarray, y: np.ndarray, rating: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """The decisive figure: cost per bridge against the threshold at r = R, with baselines.

    Every label is formatted once from its unrounded value.

    Args:
        p: The model's P(poor) for the test bridges.
        y: Test outcomes in {0, 1}.
        rating: The test bridges' lowest rating.
        out_dir: The figure directory.

    Returns:
        The cost at the left end of the drawn threshold range, ``COST_GRID[0]``.
    """
    grid = np.geomspace(*COST_GRID, 400)
    costs = np.array([expected_cost(p > s, y) for s in grid])
    c = {
        "cost_flag_none": expected_cost(np.zeros(len(y), dtype=bool), y),
        "cost_flag_all": expected_cost(np.ones(len(y), dtype=bool), y),
        "cost_rule": expected_cost(rating == 5, y),
        "cost": expected_cost(p > 1 / (1 + R), y),
        "cost_at_half": expected_cost(p > 0.5, y),
    }
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(grid, costs, "-", color=aml_style.PALETTE[0], label="model: flag if p > threshold")
    baselines = (  # key, label, colour, line style, label offset (above or below the line)
        ("cost_flag_none", "flag none ", aml_style.MUTED, ":", 0.015),
        ("cost_flag_all", "flag all ", aml_style.MUTED, "-.", 0.015),
        ("cost_rule", "every 5: ", aml_style.PALETTE[2], "--", -0.055),
    )
    for key, label, color, ls, dy in baselines:
        ax.axhline(c[key], color=color, ls=ls, lw=2.0 if key == "cost_rule" else 1.8)
        ax.text(0.0053, c[key] + dy, f"{label}{c[key]:.2f}", color=color, fontsize=12)
    t = 1 / (1 + R)
    ax.plot([t], [c["cost"]], "s", color=aml_style.PALETTE[1], markersize=11, zorder=5)
    ax.annotate(
        f"1/21: {c['cost']:.2f}",
        (t, c["cost"]),
        textcoords="offset points",
        xytext=(0, -30),
        ha="center",
        fontsize=13,
        color=aml_style.PALETTE[1],
    )
    ax.plot([0.5], [c["cost_at_half"]], "D", color=aml_style.PALETTE[3], markersize=10, zorder=5)
    ax.annotate(
        f"0.5: {c['cost_at_half']:.2f}",
        (0.5, c["cost_at_half"]),
        textcoords="offset points",
        xytext=(-4, -26),
        ha="center",
        fontsize=13,
        color=aml_style.PALETTE[3],
    )
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.1, 0.5])
    ax.set_xticklabels(["0.01", "0.05", "0.1", "0.5"])
    ax.set_xlabel(f"threshold on p (r = {R})")
    ax.set_ylabel("cost per bridge, 2020 to 2024")
    ax.set_ylim(0.3, 1.08)
    ax.text(
        0.13,
        0.33,
        "solid: the model,\nflag if p > threshold",
        color=aml_style.PALETTE[0],
        fontsize=12,
        va="bottom",
    )
    aml_style.save(fig, out_dir, "cost_threshold")
    return {"cost_at_grid_start": rnd(costs[0], 3)}


def validation_costs(p_val: np.ndarray, val: pd.DataFrame) -> dict[str, object]:
    """Cost per bridge at r = R on the validation period, 2016 -> 2020.

    Args:
        p_val: The model's P(poor) for the validation bridges.
        val: The validation transitions.

    Returns:
        The model's, the rule's, and the baselines' costs, flags, and catches, and the twin
        floor (:func:`twin_floor`).
    """
    y, rating = val["poor_next"].to_numpy(), val["rating"].to_numpy(int)
    flag = p_val > 1 / (1 + R)
    return {
        "cost": rnd(expected_cost(flag, y), 3),
        "cost_rule": rnd(expected_cost(rating == 5, y), 3),
        "cost_flag_none": rnd(expected_cost(np.zeros_like(flag), y), 3),
        "cost_flag_all": rnd(expected_cost(np.ones_like(flag), y), 3),
        "flagged_pct": pct(flag.mean()),
        "caught_pct": pct((flag & (y == 1)).sum() / y.sum()),
        "rule_caught_pct": pct(((rating == 5) & (y == 1)).sum() / y.sum()),
        "rule_flagged_pct": pct((rating == 5).mean()),
        "floor": twin_floor(val, p_val),
    }


def twin_floor(records: pd.DataFrame, p: np.ndarray) -> dict[str, object]:
    """A rough floor: flag each twin bridge by whether its twin turned poor in the same period.

    The model, the rule, and the two baselines are costed on the same twin records, so the floor
    is compared like for like (twins turn poor more often than the average bridge, and the cost
    per bridge grows with that share).

    Args:
        records: One period's transitions, with a 0..n-1 index (``transitions`` resets it).
        p: The model's P(poor) for each of ``records``.

    Returns:
        Counts and costs per twin record at r = ``R``, rounded; the unrounded costs under the
        private key ``_raw`` (for the figure's labels).
    """
    assert records.index.equals(pd.RangeIndex(len(records))) and len(p) == len(records)
    tw = find_twins(records)
    rows = tw["row"].to_numpy()
    a = records["poor_next"].to_numpy()[rows]
    b = records["poor_next"].to_numpy()[tw["twin_row"].to_numpy()]
    flag = b == 1
    raw = {
        "cost": expected_cost(flag, a),
        "cost_model": expected_cost(p[rows] > 1 / (1 + R), a),
        "cost_rule": expected_cost(records["rating"].to_numpy(int)[rows] == 5, a),
        "cost_flag_none": expected_cost(np.zeros_like(flag), a),
        "cost_flag_all": expected_cost(np.ones_like(flag), a),
    }
    return {
        "twin_records": int(len(tw)),
        "poor_pct": pct(a.mean()),
        "agree_pct": pct((a == b).mean()),
        **{k: rnd(v, 3) for k, v in raw.items()},
        "_raw": raw,
    }


def bracket_figure(val: dict[str, object], out_dir: Path) -> None:
    """Dot plot of the validation cost at r = R on the twin records: baselines, model, floor.

    Args:
        val: :func:`validation_costs`'s result; its ``floor`` holds the unrounded costs.
        out_dir: The figure directory.
    """
    floor = val["floor"]
    assert isinstance(floor, dict)
    raw = floor["_raw"]
    rows = [
        ("flag all", raw["cost_flag_all"], "o", aml_style.MUTED),
        ("flag none", raw["cost_flag_none"], "s", aml_style.MUTED),
        ("rule: flag every 5", raw["cost_rule"], "^", aml_style.PALETTE[2]),
        ("model, p > 1/21", raw["cost_model"], "D", aml_style.PALETTE[1]),
        ("twin's outcome (rough)", raw["cost"], "v", aml_style.PALETTE[0]),
    ]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, (_name, v, marker, color) in enumerate(rows):
        ax.plot([0, v], [i, i], color="#D9D9D9", lw=1.5, zorder=1)
        ax.plot([v], [i], marker, color=color, markersize=11, zorder=2)
        ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=13)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel(f"cost per bridge, r = {R}\n{floor['twin_records']:,} twin records, 2016 to 2020")
    aml_style.save(fig, out_dir, "bracket")


def ordinal(
    Z: np.ndarray,
    tr: pd.DataFrame,
    Zt: np.ndarray,
    test: pd.DataFrame,
    *,
    p_bin: np.ndarray,
    out_dir: Path,
) -> dict[str, object]:
    """Proportional odds against the binary model; per-cut weights against the shared one.

    Args:
        Z: Standardized training inputs.
        tr: The training transitions (their later ratings are the ordinal labels).
        Zt: Standardized test inputs.
        test: The test transitions.
        p_bin: The binary model's P(poor) for the test bridges.
        out_dir: The figure directory.

    Returns:
        Test AUC, share caught at a 10% budget, and cost of both models; the per-cut and shared
        weights of age and superstructure; the number of classes; and, private, the cutpoints
        and training scores for :func:`cutpoints_figure`.
    """
    w, theta = fit_ordinal(Z, tr["rating_next"].to_numpy(int))
    p_ord = expit(theta[0] - Zt @ w)  # P(later rating <= POOR)
    y = test["poor_next"].to_numpy()
    show = ["age", "superstructure"]
    per_cut: dict[str, dict[str, float]] = {}
    for j in OUTCOMES:
        fit = fit_unpenalized(Z, (tr["rating_next"].to_numpy(int) <= j).astype(int))
        per_cut[str(int(j))] = {n: rnd(fit.coef_[0, INPUTS.index(n)], 3) for n in show}
    shared = {n: rnd(-w[INPUTS.index(n)], 3) for n in show}  # sign: toward a lower rating
    fig, axes = plt.subplots(1, 2, figsize=aml_style.SIDE, sharey=True)
    cuts = [int(j) for j in OUTCOMES]
    for i, (ax, n) in enumerate(zip(axes, show, strict=True)):
        vals = [per_cut[str(j)][n] for j in cuts]
        ax.plot(
            range(len(cuts)),
            vals,
            "o-",
            color=aml_style.PALETTE[i],
            markersize=8,
            label="separate fit per cut",
        )
        ax.axhline(shared[n], color=aml_style.MUTED, ls="--", lw=1.8, label="ordinal: one weight")
        ax.axhline(0, color="#D9D9D9", lw=1)
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels([f"≤{j}" for j in cuts])
        ax.set_title(n, fontsize=14)
        ax.set_xlabel("later rating")
    axes[0].set_ylabel("weight toward a lower rating")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.55, -0.17)
    )
    aml_style.save(fig, out_dir, "ordinal_weights")
    return {
        "auc_ordinal": rnd(roc_auc_score(y, p_ord), 3),
        "auc_binary": rnd(roc_auc_score(y, p_bin), 3),
        "caught_at_10_ordinal_pct": pct(caught_at_share(p_ord, y), 0),
        "caught_at_10_binary_pct": pct(caught_at_share(p_bin, y), 0),
        "cost_ordinal": rnd(expected_cost(p_ord > 1 / (1 + R), y), 3),
        "per_cut": per_cut,
        "shared": shared,
        "classes": len(OUTCOMES) + 1,
        "_theta": theta,
        "_score": Z @ w,
    }


def calibration(
    p_val: np.ndarray, yv: np.ndarray, p: np.ndarray, y: np.ndarray, out_dir: Path
) -> dict[str, object]:
    """Mean predicted against observed, by tenth of predicted risk, in 2016-20 and 2020-24.

    Args:
        p_val: The model's P(poor) for the validation bridges.
        yv: Validation outcomes.
        p: The model's P(poor) for the test bridges.
        y: Test outcomes.
        out_dir: The figure directory.

    Returns:
        Per period: mean predicted and observed shares, overall and in the riskiest tenth.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    out: dict[str, object] = {}
    for i, (name, pp, yy) in enumerate(
        (("2016 to 2020 (validation)", p_val, yv), ("2020 to 2024 (test)", p, y))
    ):
        q = pd.qcut(pp, 10, labels=False)
        tab = pd.DataFrame({"p": pp, "y": yy, "q": q}).groupby("q").mean()
        ax.plot(
            100 * tab["p"],
            100 * tab["y"],
            aml_style.MARKERS[i] + aml_style.LINESTYLES[i],
            color=aml_style.PALETTE[i],
            label=name,
        )
        key = "val" if i == 0 else "test"
        out[key] = {
            "mean_p_pct": pct(pp.mean()),
            "observed_pct": pct(yy.mean()),
            "top_tenth_p_pct": pct(tab["p"].iloc[-1], 0),
            "top_tenth_observed_pct": pct(tab["y"].iloc[-1], 0),
        }
    lim = 26
    ax.plot([0, lim], [0, lim], ":", color=aml_style.MUTED, lw=1.5)
    # Past the end of both curves, so neither crosses it.
    ax.text(20.9, 19.2, "predicted = observed", color=aml_style.MUTED, fontsize=12, rotation=33,
            va="top")  # fmt: skip
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted, % turning poor")
    ax.set_ylabel("observed, %")
    ax.legend(loc="upper left", fontsize=12)
    aml_style.save(fig, out_dir, "calibration")
    return out


def svm(
    Z: np.ndarray,
    y: np.ndarray,
    Zv: np.ndarray,
    yv: np.ndarray,
    *,
    Zt: np.ndarray,
    yt: np.ndarray,
    p_bin: np.ndarray,
) -> dict[str, object]:
    """Linear L2-SVM (squared hinge) on the same inputs; Platt scaling on the validation scores.

    Args:
        Z: Standardized training inputs.
        y: Training outcomes.
        Zv: Standardized validation inputs (Platt scaling is fitted on their scores).
        yv: Validation outcomes.
        Zt: Standardized test inputs.
        yt: Test outcomes.
        p_bin: The logistic model's P(poor) for the test bridges.

    Returns:
        Test AUCs of the SVM and the logistic model, the count of positive scores (test,
        training, and with balanced class weights), the share caught at a 10% budget, and the
        cost at 1/(1 + R) after Platt scaling.
    """
    # The squared hinge, max(0, 1 - y s)^2, the slide's loss and LinearSVC's default, set
    # explicitly. liblinear also penalizes the intercept: it is the weight of a constant input
    # (intercept_scaling = 1). The plain hinge collapses here (see the notebook's Gotchas).
    kw = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "max_iter": 100_000}
    clf = LinearSVC(**kw, random_state=SEED).fit(Z, y)
    # The same loss with the classes weighted equally: no positive score is a matter of costs.
    balanced = LinearSVC(**kw, class_weight="balanced", random_state=SEED).fit(Z, y)
    s_val, s = clf.decision_function(Zv), clf.decision_function(Zt)
    platt = fit_unpenalized(s_val[:, None], yv)
    p = platt.predict_proba(s[:, None])[:, 1]
    return {
        "auc_svm": rnd(roc_auc_score(yt, s), 3),
        "auc_logistic": rnd(roc_auc_score(yt, p_bin), 3),
        "positive_scores": int((s > 0).sum()),
        "train_positive_scores": int((clf.decision_function(Z) > 0).sum()),
        "balanced_train_positive_scores": int((balanced.decision_function(Z) > 0).sum()),
        "caught_at_10_pct": pct(caught_at_share(s, yt), 0),
        "platt_cost": rnd(expected_cost(p > 1 / (1 + R), yt), 3),
    }


def dial_figure(
    p: np.ndarray, y: np.ndarray, th: dict[str, object], out_dir: Path
) -> dict[str, float]:
    """Share caught against share flagged: the model's whole curve, the rule's two points.

    Args:
        p: The model's P(poor) for the test bridges.
        y: Test outcomes.
        th: :func:`thresholds`'s result (the points for each r and the rules).
        out_dir: The figure directory.

    Returns:
        The share caught when 90% of bridges are flagged, and the share that must be flagged
        to catch every bridge that turned poor (for the alt text).
    """
    order = np.argsort(-p, kind="stable")
    caught = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    share = np.arange(len(y) + 1) / len(y)
    at_90 = caught[int(round(0.9 * len(y)))]
    all_caught = share[int(np.argmax(caught >= 1))]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(100 * share, 100 * caught, "-", color=aml_style.PALETTE[0], label="model")
    per_r = th["per_ratio"]  # type: ignore[index]
    for i, r in enumerate(RATIOS):
        c = per_r[str(r)]
        ax.plot(
            [c["flagged_pct"]],
            [c["caught_pct"]],
            aml_style.MARKERS[1 + i % 3],
            color=aml_style.PALETTE[1],
            markersize=10,
            zorder=5,
        )
        ax.annotate(
            f"r = {r}",
            (c["flagged_pct"], c["caught_pct"]),
            textcoords="offset points",
            xytext=(10, -16),
            fontsize=12,
            color=aml_style.PALETTE[1],
        )
    rule, rule56 = th["rule"], th["rule_5_6"]  # type: ignore[index]
    ax.plot(
        [rule["flagged_pct"], rule56["flagged_pct"]],
        [rule["caught_pct"], rule56["caught_pct"]],
        "D",
        color=aml_style.PALETTE[2],
        markersize=16,
        markerfacecolor="none",
        markeredgewidth=2.2,
        label="rule",
        zorder=6,
    )
    for name, point in (("every 5", rule), ("every 5 or 6", rule56)):
        ax.annotate(
            name,
            (point["flagged_pct"], point["caught_pct"]),
            textcoords="offset points",
            xytext=(-12, 12),
            ha="right",
            fontsize=12,
            color=aml_style.PALETTE[2],
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("bridges flagged (%)")
    ax.set_ylabel("poor by 2024, caught (%)")
    ax.legend(loc="lower right", fontsize=12)
    aml_style.save(fig, out_dir, "dial")
    return {"caught_at_90_flagged_pct": pct(at_90), "flagged_to_catch_all_pct": pct(all_caught)}


def by_owner(test: pd.DataFrame, p: np.ndarray) -> dict[str, object]:
    """Share caught at the r = R threshold, by owner (Item 22); other owner codes in one row.

    Lecture 27's notebook reads ``by_owner[<owner>]["caught_pct"]``: keep that key.

    Args:
        test: The test transitions, with ``owner``.
        p: The model's P(poor) for the test bridges.

    Returns:
        Per owner: bridges, their share of the inventory, those that turned poor, the model's
        mean p, the shares rated 5 and flagged, the share caught, and the misses; private, the
        unrounded share caught (for the figure's labels).
    """
    flag = p > 1 / (1 + R)
    y = test["poor_next"].to_numpy()
    owner = test["owner"].to_numpy()
    masks = {name: owner == code for code, name in OWNERS.items()}
    masks["other owners"] = ~np.isin(owner, list(OWNERS))
    out = {}
    for name, m in masks.items():
        assert y[m].sum() > 0, name  # the caught share divides by it
        caught = (flag[m] & (y[m] == 1)).sum() / y[m].sum()
        out[name] = {
            "bridges": int(m.sum()),
            "inventory_pct": pct(m.mean()),
            "turned_poor": int(y[m].sum()),
            "poor_pct": pct(y[m].mean()),
            "mean_p_pct": pct(p[m].mean()),
            "rated_5_pct": pct((test["rating"].to_numpy()[m] == 5).mean()),
            "flagged_pct": pct(flag[m].mean()),
            "caught_pct": pct(caught, 0),
            "missed": int(((~flag[m]) & (y[m] == 1)).sum()),
            "_caught": float(caught),
        }
    assert sum(int(v["missed"]) for v in out.values()) == int(((~flag) & (y == 1)).sum())
    return out


def attempt(test: pd.DataFrame, p: np.ndarray) -> dict[str, object]:
    """The attempt's six bridges: the busiest city-owned Champaign County bridges rated 5 or 6.

    Args:
        test: The test transitions.
        p: The model's P(poor) for the test bridges.

    Returns:
        The six rows (inventory facts, ratings, p, the flag at r = ``R``), their ranking by p,
        and which turned poor and which were flagged.
    """
    d = test.assign(p=p)
    pick = d[(d["county"] == CHAMPAIGN) & (d["owner"] == CITY) & (d["rating"] <= 6)]
    pick = pick.sort_values(["adt", "structure"], ascending=[False, True], kind="stable").head(6)
    assert set(pick["structure"]) == set(SHOW_NAMES), sorted(pick["structure"])
    rows = []
    for letter, (_, b) in zip("ABCDEF", pick.iterrows(), strict=True):
        road, over = SHOW_NAMES[b["structure"]]
        rows.append(
            {
                "id": letter,
                "structure": b["structure"],
                "road": road,
                "over": over,
                "facility_nbi": b["facility"].strip(),
                "built": int(b["year_built"]),
                "rebuilt": None
                if pd.isna(b["year_reconstructed"])
                else int(b["year_reconstructed"]),
                "adt": int(b["adt"]),
                "material": b["material"],
                "deck": int(b["deck"]),
                "superstructure": int(b["superstructure"]),
                "substructure": int(b["substructure"]),
                "rating": int(b["rating"]),
                "rating_2024": int(b["rating_next"]),
                "p": rnd(b["p"], 3),
                "flag_at_r": bool(b["p"] > 1 / (1 + R)),
            }
        )
    by_p = sorted(rows, key=lambda r: -r["p"])
    return {
        "rows": rows,
        "rank_by_p": [r["id"] for r in by_p],
        "turned_poor": [r["id"] for r in rows if r["rating_2024"] <= POOR],
        "flagged_at_r": [r["id"] for r in rows if r["flag_at_r"]],
        "materials": sorted({r["material"] for r in rows}),
    }


def owner_gap(
    test: pd.DataFrame, Zt: np.ndarray, w: np.ndarray, p: np.ndarray
) -> dict[str, object]:
    """Why state bridges get a higher p than township bridges, and whether the catch gap is noise.

    Args:
        test: The test transitions, with ``owner`` and ``county``.
        Zt: Standardized test inputs.
        w: The model's weights.
        p: The model's P(poor) for the test bridges.

    Returns:
        The mean score of state bridges minus township bridges, split by group of inputs; and
        the share caught, state minus township, in points, with a 95% county-cluster interval.
    """
    code = {name: c for c, name in OWNERS.items()}
    owner = test["owner"].to_numpy()
    state, town = owner == code["state"], owner == code["township"]
    groups = {
        "ratings": [*COMPONENTS, "rating"],
        "material": list(MATERIALS),
        "traffic_and_size": ["log_adt", "pct_truck", "log_length", "deck_width_m"],
        "age_and_work": ["age", "years_since_work"],
        "location": ["lat", "lon"],
    }
    assert sorted(c for cols in groups.values() for c in cols) == sorted(INPUTS)
    gap = (Zt[state] * w[1:]).mean(axis=0) - (Zt[town] * w[1:]).mean(axis=0)
    y = test["poor_next"].to_numpy() == 1
    hit = (p > 1 / (1 + R)) & y
    num = np.column_stack([hit & state, hit & town]).astype(np.float64)
    den = np.column_stack([y & state, y & town]).astype(np.float64)
    draws = county_draws(num, den, test["county"].to_numpy(), stream=3)
    caught = num.sum(axis=0) / den.sum(axis=0)
    return {
        "score_gap": rnd(gap.sum(), 2),
        "score_gap_by_group": {
            g: rnd(sum(gap[INPUTS.index(c)] for c in cols), 2) for g, cols in groups.items()
        },
        "caught_gap_pts": rnd(100 * (caught[0] - caught[1]), 1),
        "caught_gap_ci_pts": [
            rnd(v, 1) for v in np.percentile(100 * (draws[:, 0] - draws[:, 1]), CI)
        ],
    }


def trace(test: pd.DataFrame, Zt: np.ndarray, w: np.ndarray, tr: pd.DataFrame) -> dict[str, object]:
    """The traced bridge through the model: inputs, score, the largest terms, p, decision.

    The material is one input coded as four 0/1 columns against the few training bridges of
    "other" material (``verify``'s ``other_material_bridges``). Read one column alone and the
    attribution is an artefact of that small baseline: in the model's own coding, B's prestressed
    column subtracts a large term that the other material columns add back. So the terms are
    attributed with the most common material, ``TRACE_BASELINE``, as the baseline: the same model
    refitted with an "other material" column in place of the prestressed one. Its probabilities,
    score, and intercept are the same; only the attribution moves.

    Args:
        test: The test transitions.
        Zt: Standardized test inputs, in the model's coding.
        w: The model's weights.
        tr: The training transitions (for the refit in the new coding).

    Returns:
        The bridge's record; the intercept, the three largest terms (input, z, term), the rest,
        the score, p, and the decision at r = ``R``, rounded as the slide quotes them; the sum of
        its material terms, and the prestressed column's term in the model's own coding; and,
        private, the unrounded trace for the diagram.
    """
    assert tr["material"].value_counts().index[0] == TRACE_BASELINE  # the most common
    i = int(np.flatnonzero(test["structure"].to_numpy() == TRACED)[0])
    b = test.iloc[i]
    k_base = INPUTS.index(TRACE_BASELINE)
    names = ["other_material" if n == TRACE_BASELINE else n for n in INPUTS]

    def recode(d: pd.DataFrame) -> np.ndarray:
        """The inputs with the baseline material's column replaced by an "other" column.

        Args:
            d: Transitions with ``INPUTS`` and ``material``.

        Returns:
            The recoded inputs, unscaled, shape ``(len(d), len(INPUTS))``.
        """
        x = d[INPUTS].to_numpy(np.float64).copy()
        x[:, k_base] = (d["material"] == "other").to_numpy(np.float64)
        return x

    scaler2 = StandardScaler().fit(recode(tr))
    w2, _ = fit_logistic(scaler2.transform(recode(tr)), tr["poor_next"].to_numpy())
    Zt2 = scaler2.transform(recode(test))
    p_gap = float(np.max(np.abs(predict_proba(w2, Zt2) - predict_proba(w, Zt))))
    assert p_gap < 1e-9, p_gap  # the same model, coded differently
    old, terms = w[1:] * Zt[i], w2[1:] * Zt2[i]
    s = float(w2[0] + terms.sum())
    assert abs(s - float(w[0] + old.sum())) < 1e-9 and abs(w2[0] - w[0]) < 1e-9
    material = [INPUTS.index(m) for m in MATERIALS if m != TRACE_BASELINE] + [k_base]
    assert abs(terms[material].sum() - old[[INPUTS.index(m) for m in MATERIALS]].sum()) < 1e-9
    top = np.argsort(-np.abs(terms))[:3]
    raw = {
        "intercept": float(w2[0]),
        "top": [(names[k], float(Zt2[i, k]), float(terms[k])) for k in top],
        "rest": float(terms.sum() - terms[top].sum()),
        "s": s,
        "p": float(expit(s)),
    }
    return {
        "structure": TRACED,
        "age": int(b["age"]),
        "built": int(b["year_built"]),
        "adt": int(b["adt"]),
        "deck": int(b["deck"]),
        "superstructure": int(b["superstructure"]),
        "substructure": int(b["substructure"]),
        "rating": int(b["rating"]),
        "rating_2024": int(b["rating_next"]),
        "baseline": TRACE_BASELINE,
        "intercept": rnd(raw["intercept"], 2),
        "top": [{"input": n, "z": rnd(z, 2), "term": rnd(t, 2)} for n, z, t in raw["top"]],
        "rest": rnd(raw["rest"], 2),
        "s": rnd(s, 2),
        "p": rnd(raw["p"], 2),
        "threshold": rnd(1 / (1 + R), 3),
        "flag": raw["p"] > 1 / (1 + R),
        "material_terms": rnd(terms[material].sum(), 2),
        "own_coding_prestressed_term": rnd(old[k_base], 2),
        "_raw": raw,
    }


def post_hoc(
    df: pd.DataFrame,
    sets: dict[str, pd.DataFrame],
    w: np.ndarray,
    scaler: StandardScaler,
    *,
    p: np.ndarray,
    p_val: np.ndarray,
) -> dict[str, object]:
    """Checks made in hindsight, never used for a decision.

    Could the 2021 engineer have fixed the 2024 calibration with the latest period she had,
    2016 -> 2020? And does validation rank better than the test period because it overlaps
    training, or because 2020 -> 2024 differs? The later period 2018 -> 2022 answers both.

    Args:
        df: Bridge-year records.
        sets: The training, validation, and test transitions.
        w: The model's weights.
        scaler: The training scaler.
        p: The model's P(poor) for the test bridges.
        p_val: The model's P(poor) for the validation bridges.

    Returns:
        The test period's mean p and observed share; Platt scaling and a refit on 2016 -> 2020,
        each judged on 2024; whether the slide's Newton listing converges on 2016 -> 2020; the
        later period's calibration and AUC, the model's and the rating's alone; and the rating's
        AUC on validation.
    """
    va, te = sets["val"], sets["test"]
    yv, yt = va["poor_next"].to_numpy(), te["poor_next"].to_numpy()
    t = 1 / (1 + R)
    platt = fit_unpenalized(logit(p_val)[:, None], yv)  # Platt scaling of the log-odds
    p_platt = platt.predict_proba(logit(p)[:, None])[:, 1]
    x_va = va[INPUTS].to_numpy(np.float64)
    scaler_va = StandardScaler().fit(x_va)
    try:  # quasi-separation: see fit_logistic's Raises
        fit_logistic(scaler_va.transform(x_va), yv)
        listing_converges = True
    except RuntimeError:
        listing_converges = False
    refit = fit_unpenalized(scaler_va.transform(x_va), yv, tol=1e-12, max_iter=200)
    p_refit = refit.predict_proba(scaler_va.transform(te[INPUTS].to_numpy(np.float64)))[:, 1]
    later = transitions(df, LATER_START)
    yl, pl = later["poor_next"].to_numpy(), predict_proba(w, encode(later, scaler))
    other = (va["material"] == "other").to_numpy()
    return {
        "test_mean_p_pct": pct(p.mean(), 2),
        "test_observed_pct": pct(yt.mean(), 2),
        "platt_val": {
            "test_mean_p_pct": pct(p_platt.mean(), 2),
            "cost": rnd(expected_cost(p_platt > t, yt), 3),
        },
        "refit_val": {
            "test_mean_p_pct": pct(p_refit.mean(), 2),
            "cost": rnd(expected_cost(p_refit > t, yt), 3),
            "listing_converges": listing_converges,
            "other_material_bridges": int(other.sum()),
            "other_material_poor": int(yv[other].sum()),
        },
        "later": {
            "start": LATER_START,
            "end": LATER_START + HORIZON,
            "mean_p_pct": pct(pl.mean(), 2),
            "observed_pct": pct(yl.mean(), 2),
            "auc": rnd(roc_auc_score(yl, pl), 3),
            "auc_rating_only": rnd(roc_auc_score(yl, -later["rating"].to_numpy()), 3),
        },
        "auc_rating_only_val": rnd(roc_auc_score(yv, -va["rating"].to_numpy()), 3),
    }


def dropped(df: pd.DataFrame, w: np.ndarray, scaler: StandardScaler) -> dict[str, object]:
    """The test bridges dropped from the label: in the 2020 inventory, gone by 2024.

    A bridge replaced or removed has no later rating, so it has no label. If such bridges are
    riskier than the rest, the test cost is computed on the safer bridges (informative
    censoring).

    Args:
        df: Bridge-year records.
        w: The model's weights.
        scaler: The training scaler.

    Returns:
        How many were dropped of how many, the shares rated 5 among the dropped and the kept,
        and the model's mean p and expected number turning poor among the dropped.
    """
    start = START["test"]
    now = df[(df["nbi_year"] == start) & (df["rating"] > POOR)]
    gone_mask = ~now["structure"].isin(set(transitions(df, start)["structure"]))
    gone, kept = now[gone_mask].copy(), now[~gone_mask]
    gone["deck"] = gone["deck"].fillna(gone[["superstructure", "substructure"]].min(axis=1))
    assert gone[INPUTS].notna().all().all(), "missing inputs"
    pg = predict_proba(w, encode(gone, scaler))
    return {
        "dropped": int(len(gone)),
        "of": int(len(now)),
        "rated_5_pct_dropped": pct((gone["rating"] == 5).mean(), 0),
        "rated_5_pct_kept": pct((kept["rating"] == 5).mean(), 0),
        "mean_p_pct_dropped": pct(pg.mean(), 0),
        "expected_poor_dropped": int(round(float(pg.sum()))),
    }


def write_tex(tr: dict[str, object], summary: dict[str, object], out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, each formatted once from raw.

    Args:
        tr: :func:`trace`'s result (its private ``_raw`` holds the unrounded values).
        summary: :func:`data_summary`'s result.
        out_dir: The figure directory.

    Returns:
        The path of ``numbers.tex``.
    """
    raw = tr["_raw"]
    assert isinstance(raw, dict)
    top = raw["top"]
    names = {
        "rating": "lowest rating",
        "superstructure": "superstructure",
        "deck": "deck",
        "substructure": "substructure",
        "prestressed": "prestressed",
        "other_material": "other material",
        "age": "age",
        "lat": "latitude",
    }
    train = summary["train"]
    assert isinstance(train, dict)
    macros = {
        "NTrain": tex_num(train["bridges"], ","),
        "NInputs": tex_num(len(INPUTS)),
        "BBuilt": tex_num(tr["built"]),
        "NRest": tex_num(len(INPUTS) - 3),
        "BADT": tex_num(tr["adt"], ","),
        "BDeck": tex_num(tr["deck"]),
        "BSuper": tex_num(tr["superstructure"]),
        "BSub": tex_num(tr["substructure"]),
        "BRating": tex_num(tr["rating"]),
        "BLater": tex_num(tr["rating_2024"]),
        "WZero": tex_num(raw["intercept"], ".2f"),
    }
    for k, (n, z, t) in zip("ABC", top, strict=True):
        macros |= {
            f"TName{k}": names.get(n, n),
            f"TZ{k}": tex_num(z, "+.2f"),
            f"TTerm{k}": tex_num(t, "+.2f"),
        }
    macros |= {
        "TRest": tex_num(raw["rest"], "+.2f"),
        "Score": tex_num(raw["s"], ".2f"),
        "Prob": tex_num(raw["p"], ".2f"),
        "Thresh": tex_num(1 / (1 + R), ".3f"),
    }
    return write_tex_macros(macros, out_dir, "cee_l06_linear_classifiers.py")


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

SHEET_SHOWN: Final[int] = 1_500  # training bridges drawn on the top view of the sheet
SHEET_INPUTS: Final[tuple[str, str]] = ("rating", "age")  # the illustrative sheet's two inputs
SHEET_RATING: Final[tuple[float, float]] = (4.6, 9.4)  # the sheet's rating range
SHEET_AGE_MAX: Final[int] = 110  # the sheet's age range starts at 0


def title_art_figure(out_dir: Path) -> None:
    """The title art: a crimson boundary splitting pale and amber points, under a dark sun.

    The horizon is the logistic curve, sigma(s), rising from 0 to 1.

    Args:
        out_dir: The figure directory.
    """
    c = title_art.Canvas(seed=6)
    c.ground(("#120d1a", "#1d0e14", "#1a0b0c", "#07060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.06, 0.06, 0.88, 0.62, 4), soft=4), "#552014", 0.85)
    rim = c.disc((0.74, 1.17), 0.12, soft=4) - c.disc((0.74, 1.17), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.74, 1.17), 0.10, soft=3), "#1d0a10", 0.92)
    rng = np.random.default_rng(6)
    pts = rng.uniform((0.08, 0.72), (0.92, 1.30), (90, 2))
    side = pts[:, 1] - 0.72 > 0.62 * (pts[:, 0] - 0.08)  # above the boundary line
    for mask, col in ((side, "#9fb7c9"), (~side, "#f0a03c")):
        dots = sum(c.disc(tuple(q), 0.006, soft=1.5) for q in pts[mask])
        c.glow(np.clip(dots, 0, 1), col, 0.55, spread=7)
    c.glaze(c.polygon([(0.06, 0.70), (0.10, 0.70), (0.96, 1.24), (0.92, 1.24)], soft=1.5),
            "#c2361f", 0.85)  # fmt: skip
    x = np.linspace(-6, 6, 300)
    curve = [list(zip(0.08 + 0.84 * (x + 6) / 12, 0.14 + 0.40 * expit(x), strict=True))]
    c.glow(c.lines(curve, 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean form in rust; above it pale and "
        "amber points divided by a thin crimson line; a dark sun with a burning rim at the upper "
        "right, and a golden S-shaped curve, the logistic function, rising across the form.",
    )


def timeline_figure(summary: dict[str, object], out_dir: Path) -> None:
    """The three periods: train 2014 -> 2018, validate 2016 -> 2020, test once 2020 -> 2024.

    Args:
        summary: :func:`data_summary`'s result.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    rows = (("train", "train", aml_style.PALETTE[0]), ("val", "validate", aml_style.PALETTE[2]),
            ("test", "test, once", aml_style.PALETTE[1]))  # fmt: skip
    for i, (key, name, color) in enumerate(rows):
        d = summary[key]
        assert isinstance(d, dict)
        y = 2 - i
        ax.annotate("", (d["end"], y), (d["start"], y),
                    arrowprops={"arrowstyle": "-|>", "lw": 5, "color": color})  # fmt: skip
        ax.text(d["start"], y + 0.22, f"{name}: {d['bridges']:,} bridges, {d['turned_poor']} poor",
                fontsize=12, color=aml_style.INK)  # fmt: skip
    ax.set(xlim=(2013.5, 2024.8), ylim=(-0.6, 2.7), yticks=[], xticks=range(2014, 2025, 2))
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("inventory year")
    aml_style.save(fig, out_dir, "timeline")


def sheet_figure(tr: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """P(poor in four years) from two inputs, lowest rating and age: an S-shaped sheet.

    Left, the sheet in 3D, with the cost threshold 1/21 drawn on it. Right, seen from above:
    training bridges (those that turned poor in orange) and the threshold line.

    Args:
        tr: The training transitions.
        out_dir: The figure directory.

    Returns:
        The two-input model's weights, the sheet's highest p (at the grid's edge), p at rating
        5 for the youngest and oldest ages drawn, and the rating where the threshold line
        crosses those two ages.
    """
    X = tr[list(SHEET_INPUTS)].to_numpy(float)
    y = tr["poor_next"].to_numpy()
    lr = fit_unpenalized(X, y)
    r, a = np.meshgrid(np.linspace(*SHEET_RATING, 81), np.linspace(0, SHEET_AGE_MAX, 81))
    P = lr.predict_proba(np.column_stack([r.ravel(), a.ravel()]))[:, 1].reshape(r.shape)
    fig = plt.figure(figsize=aml_style.FULL)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.computed_zorder = False
    aml_style.surface3d(ax, r, a, P, cmap="cividis")
    ax.contour(r, a, P, levels=[1 / (1 + R)], colors=aml_style.PALETTE[1], linewidths=2.6,
               zorder=5)  # fmt: skip
    ax.set_xlabel("lowest rating", labelpad=6, fontsize=12)
    ax.set_ylabel("age (years)", labelpad=6, fontsize=12)
    ax.set(xticks=[5, 7, 9], yticks=[0, 50, 100], zticks=[0, 0.2, 0.4])
    ax.set_title("P(poor in 4 years)", loc="left", fontsize=13)
    aml_style.style3d(ax, elev=22, azim=-25)  # rating runs across: the S shows; age barely tilts
    ax.set_box_aspect(None, zoom=0.85)  # smaller box: the rating label fits inside the figure
    ax2 = fig.add_subplot(1, 2, 2)
    rng = np.random.default_rng((SEED, 6))
    show = rng.choice(len(tr), SHEET_SHOWN, replace=False)
    jitter = rng.uniform(-0.3, 0.3, len(show))
    for flag, color, size in ((0, aml_style.MUTED, 6), (1, aml_style.PALETTE[1], 16)):
        m = show[y[show] == flag]
        ax2.scatter(X[m, 0] + jitter[y[show] == flag], X[m, 1], s=size, color=color,
                    alpha=0.6 if flag == 0 else 0.9, lw=0)  # fmt: skip
    ax2.contour(r, a, P, levels=[1 / (1 + R)], colors=aml_style.PALETTE[1], linewidths=2.6)
    ax2.text(6.45, 101, "p = 1/21", fontsize=13, color=aml_style.PALETTE[1], fontweight="bold")
    ax2.set(xlabel="lowest rating now", ylabel="age (years)", xticks=[5, 6, 7, 8, 9],
            xlim=(4.5, 9.5), ylim=(0, SHEET_AGE_MAX))  # fmt: skip
    ax2.set_title("age barely tilts the line", loc="left", fontsize=13)
    fig.subplots_adjust(wspace=0.25)
    aml_style.save(fig, out_dir, "sheet")
    b0, b_r, b_a = lr.intercept_[0], lr.coef_[0, 0], lr.coef_[0, 1]
    ages = {"age0": 0, f"age{SHEET_AGE_MAX}": SHEET_AGE_MAX}
    return {
        "weights": [rnd(b0, 3), rnd(b_r, 3), rnd(b_a, 3)],
        "p_max_grid": rnd(P.max(), 3),
        "p_rating5": {k: rnd(expit(b0 + 5 * b_r + b_a * v), 2) for k, v in ages.items()},
        # where b0 + b_r rating + b_a age = logit(1/(1 + R)) = -ln R
        "threshold_rating": {k: rnd((-np.log(R) - b0 - b_a * v) / b_r, 2) for k, v in ages.items()},
    }


def newton_figure(tr: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Cross-entropy over (intercept, slope) of the one-input model, and Newton's path on it.

    Args:
        tr: The training transitions.
        out_dir: The figure directory.

    Returns:
        Newton's steps, the final weights, and the loss above the minimum after each step
        drawn (two significant digits).
    """
    z = tr["rating"].to_numpy(float)
    z = (z - z.mean()) / z.std()
    y = tr["poor_next"].to_numpy()
    A = np.column_stack([np.ones(len(z)), z])

    def nll(w: np.ndarray) -> np.ndarray:
        """Mean cross-entropy at each column of ``w``.

        Args:
            w: Weights ``(intercept, slope)``, shape ``(2, m)``.

        Returns:
            The mean cross-entropy for each column, shape ``(m,)``.
        """
        s = A @ w
        return (np.logaddexp(0, s) - y[:, None] * s).mean(axis=0)

    iterates: list[np.ndarray] = []
    fit_logistic(z[:, None], y, path=iterates)  # the same Newton steps as the 16-input fit
    path = np.array(iterates)
    assert path.shape == (len(iterates), 2), path.shape
    w0, w1 = np.meshgrid(np.linspace(-5.5, 0.5, 61), np.linspace(-2.6, 0.6, 61))
    L = nll(np.stack([w0.ravel(), w1.ravel()])).reshape(w0.shape)
    Lp = nll(path.T)
    fig = plt.figure(figsize=aml_style.FULL)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.computed_zorder = False
    aml_style.surface3d(ax, w0, w1, np.minimum(L, 1.2), cmap="cividis", vmax=1.2)
    ax.plot(path[:, 0], path[:, 1], np.minimum(Lp, 1.2) + 0.01, "o-", color=aml_style.PALETTE[1],
            lw=2.2, ms=6, zorder=5)  # fmt: skip
    ax.set_xlabel("intercept", labelpad=6, fontsize=12)
    ax.set_ylabel("slope (per sd)", labelpad=6, fontsize=12)
    ax.set_zticks([0.2, 0.6, 1.0])
    ax.set_title("cross-entropy: one bowl, no closed form", loc="left", fontsize=13)
    aml_style.style3d(ax, elev=32, azim=-50)
    ax2 = fig.add_subplot(1, 2, 2)
    gap = Lp - Lp[-1]
    shown = np.flatnonzero(gap[1:] > 1e-12) + 1  # steps whose gap is above round-off
    ax2.semilogy(shown, gap[shown], "o-", color=aml_style.PALETTE[1], lw=2.2)
    ax2.set(xlabel="Newton step", ylabel="loss above the minimum", xticks=shown)
    ax2.set_title("each step squares the error", loc="left", fontsize=13)
    fig.subplots_adjust(wspace=0.3)
    aml_style.save(fig, out_dir, "newton")
    return {
        "steps": int(len(path) - 1),
        "w": [rnd(v, 3) for v in path[-1]],
        "gaps": {str(int(k)): float(f"{gap[k]:.2g}") for k in shown},
    }


def cost_lines_figure(out_dir: Path) -> None:
    """Expected cost of flagging, 1 - p, against waiting, p r: they cross at p = 1/(1 + r).

    Args:
        out_dir: The figure directory.
    """
    p_ = np.linspace(0, 0.15, 200)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(p_, 1 - p_, color=aml_style.PALETTE[0], lw=2.6, label="flag: (1 − p) × 1")
    ax.plot(p_, R * p_, color=aml_style.PALETTE[1], lw=2.6, ls="--", label=f"wait: p × {R}")
    t = 1 / (1 + R)
    ax.axvline(t, color=aml_style.INK, lw=1.2, ls=":")
    ax.axvspan(t, p_.max(), color=aml_style.PALETTE[0], alpha=0.07)
    # Labels below and right of the crossing: the waiting line climbs through the upper right.
    ax.annotate(f"p = 1/{1 + R} = {t:.3f}", (t, 1 - t), xytext=(25, -45),
                textcoords="offset points", fontsize=12,
                arrowprops={"arrowstyle": "->"})  # fmt: skip
    ax.text(0.11, 0.18, "flag", fontsize=14, color=aml_style.PALETTE[0], fontweight="bold")
    ax.text(0.022, 0.12, "wait", fontsize=14, color=aml_style.PALETTE[1], fontweight="bold")
    ax.set(xlabel="model's p", ylabel="expected cost", ylim=(0, 2.0))
    ax.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    )  # the waiting line passes behind
    aml_style.save(fig, out_dir, "cost_lines")


def losses_figure(out_dir: Path) -> None:
    """Loss against the margin y s: 0-1, logistic (cross-entropy), and squared hinge.

    Args:
        out_dir: The figure directory.
    """
    m = np.linspace(-2.5, 3, 300)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(m, (m <= 0).astype(float), color=aml_style.MUTED, lw=2.2, ls=":", label="0–1")
    ax.plot(m, np.logaddexp(0, -m) / np.log(2), color=aml_style.PALETTE[0], lw=2.6,
            label="logistic (cross-entropy)")  # fmt: skip
    ax.plot(m, np.maximum(0, 1 - m) ** 2, color=aml_style.PALETTE[1], lw=2.6, ls="--",
            label="squared hinge (SVM)")  # fmt: skip
    ax.axvline(1, color=aml_style.INK, lw=0.8, ls=":")
    ax.text(1.05, 2.2, "margin", fontsize=12)
    ax.set(xlabel="y × score (right side, by how much)", ylabel="loss", ylim=(0, 4))
    ax.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    )  # the margin line runs behind it
    aml_style.save(fig, out_dir, "losses")


def p_hist_figure(p: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    """The 2020 bridges' p: none reaches 0.5; the cost threshold 1/21 sits far to the left.

    Args:
        p: The model's P(poor) for the test bridges.
        y: Test outcomes.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    bins = np.linspace(0, 0.55, 45)
    ax.hist(p[y == 0], bins=bins, color=aml_style.MUTED, alpha=0.6, label="stayed above poor")
    ax.hist(p[y == 1], bins=bins, color=aml_style.PALETTE[1], alpha=0.9, label="turned poor")
    for t, name in ((0.5, "0.5"), (1 / (1 + R), "1/21")):
        ax.axvline(t, color=aml_style.INK, lw=1.6, ls="--" if t == 0.5 else "-")
        ax.text(t + 0.006, ax.get_ylim()[1] * 0.85, name, fontsize=13)
    ax.set(yscale="log", xlabel="p, 2020 bridges", ylabel="bridges")
    ax.legend(loc="upper center", fontsize=12)
    aml_style.save(fig, out_dir, "p_hist")


def cutpoints_figure(ordinal_out: dict[str, object], out_dir: Path) -> None:
    """Proportional odds: P(later rating <= j) = sigma(theta_j - s), one curve per cut.

    Args:
        ordinal_out: :func:`ordinal`'s result (its private cutpoints and scores).
        out_dir: The figure directory.
    """
    theta = np.asarray(ordinal_out["_theta"])
    score = np.asarray(ordinal_out["_score"])
    s = np.linspace(np.quantile(score, 0.001) - 2, np.quantile(score, 0.999) + 2, 300)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    labels = ["≤ 4 (poor)", "≤ 5", "≤ 6", "≤ 7", "≤ 8"]
    colors = [*aml_style.PALETTE, aml_style.INK]
    styles = [*aml_style.LINESTYLES, "-"]
    for j, (t, lab) in enumerate(zip(theta, labels, strict=True)):
        ax.plot(s, expit(t - s), lw=2.4, color=colors[j], ls=styles[j], label=lab)
    ax.set(xlabel="score w·z (higher: better later rating)", ylabel="P(later rating ≤ j)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11,  # outside: curves fill it
              title="one weight, five cuts", title_fontsize=11)  # fmt: skip
    aml_style.save(fig, out_dir, "cutpoints")


def owner_figure(owner: dict[str, object], out_dir: Path) -> None:
    """Share of the bridges that turned poor that the r = R threshold caught, by owner.

    Args:
        owner: :func:`by_owner`'s result (its private ``_caught`` holds the unrounded shares).
        out_dir: The figure directory.
    """
    names = list(owner)
    caught = [100 * owner[n]["_caught"] for n in names]  # type: ignore[index]
    missed = [owner[n]["missed"] for n in names]  # type: ignore[index]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    bars = ax.barh(names, caught, color=aml_style.PALETTE[0])
    text = [f"{c:.0f}% caught, {m} missed" for c, m in zip(caught, missed, strict=True)]
    ax.bar_label(bars, text, fontsize=12, padding=4)
    ax.set(xlim=(0, 130), xticks=[0, 50, 100],
           xlabel="bridges that turned poor by 2024, caught (%)")  # fmt: skip
    ax.invert_yaxis()
    aml_style.save(fig, out_dir, "owner")


def main() -> None:
    """Generate every figure and number for CEE Lecture 6."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    df = load_bridges()
    sets = {name: transitions(df, start) for name, start in START.items()}
    tr, va, te = sets["train"], sets["val"], sets["test"]
    scaler = StandardScaler().fit(tr[INPUTS].to_numpy(np.float64))
    Z, Zv, Zt = (encode(d, scaler) for d in (tr, va, te))
    y, yv, yt = (d["poor_next"].to_numpy() for d in (tr, va, te))
    w, _ = fit_logistic(Z, y)
    p_val, p = predict_proba(w, Zv), predict_proba(w, Zt)
    summary = data_summary(df, sets)
    rating_t, county_t = te["rating"].to_numpy(int), te["county"].to_numpy()
    th = thresholds(p, yt, rating_t)
    th["model_minus_rule"] = model_vs_rule(p, yt, rating_t, county_t)
    cost_curve = cost_threshold_figure(p, yt, rating_t, out_dir)
    val_costs = validation_costs(p_val, va)
    bracket_figure(val_costs, out_dir)
    tr_ = trace(te, Zt, w, tr)
    other = (tr["material"] == "other").to_numpy()
    numbers: dict[str, object] = {
        "data": summary,
        "dropped_test": dropped(df, w, scaler),
        "log_odds": log_odds_figure(tr, out_dir),
        "verify": {
            **verify(Z, y, Zv, yv),
            "other_material_bridges": int(other.sum()),
            "other_material_poor": int(y[other].sum()),
            "material_bridges": {
                m: int((tr["material"] == m).sum()) for m in (*MATERIALS, "other")
            },
        },
        "ranking": {
            "auc_val": rnd(roc_auc_score(yv, p_val), 3),
            "auc_test": rnd(roc_auc_score(yt, p), 3),
            "auc_rating_only_test": rnd(roc_auc_score(yt, -rating_t), 3),
            "caught_at_10_pct": pct(caught_at_share(p, yt), 0),
        },
        "thresholds": th,
        "cost_curve": cost_curve,
        "validation_costs": val_costs,
        "floor_test": twin_floor(te, p),
        "ordinal": ordinal(Z, tr, Zt, te, p_bin=p, out_dir=out_dir),
        "calibration": calibration(p_val, yv, p, yt, out_dir),
        "post_hoc": post_hoc(df, sets, w, scaler, p=p, p_val=p_val),
        "svm": svm(Z, y, Zv, yv, Zt=Zt, yt=yt, p_bin=p),
        "by_owner": by_owner(te, p),
        "owner_gap": owner_gap(te, Zt, w, p),
        "attempt": attempt(te, p),
        "trace": tr_,
        "R": R,
    }
    numbers["dial"] = dial_figure(p, yt, th, out_dir)
    # the visuals that carry the slides
    title_art_figure(out_dir)
    timeline_figure(summary, out_dir)
    numbers["sheet"] = sheet_figure(tr, out_dir)
    numbers["newton"] = newton_figure(tr, out_dir)
    cost_lines_figure(out_dir)
    losses_figure(out_dir)
    p_hist_figure(p, yt, out_dir)
    cutpoints_figure(numbers["ordinal"], out_dir)
    owner_figure(numbers["by_owner"], out_dir)
    aml_style.thumbnails(out_dir, ("sheet", "cost_lines", "p_hist", "calibration"))
    write_numbers(numbers, out_dir)
    print(write_tex(tr_, summary, out_dir))


if __name__ == "__main__":
    main()
