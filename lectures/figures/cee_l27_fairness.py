"""Figures and numbers for CEE Lecture 27: fairness and accountability (Illinois bridges).

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l27_fairness.py --out build/fig/cee_l27_fairness

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

The question: Lecture 6's classifier flags Illinois bridges that are not poor today but may be poor
in four years, for preventive work. Does it serve the state's socially vulnerable counties as well
as the others? The model is Lecture 6's, refit exactly (its own functions, its 2014 -> 2018
training transitions, its threshold p > 1/21), and judged on its test set: the 2020 bridges and
the 2024 inventory. Vulnerability is the county's overall percentile rank in the CDC/ATSDR Social
Vulnerability Index 2022, within Illinois, in quartiles closed on the right (the top quartile holds
the ranks above 0.75).
Source: CDC/ATSDR; its use here does not imply endorsement by CDC, ATSDR, HHS, or the U.S.
Government; the data are available at no charge from the ATSDR SVI site (``scripts/fetch_svi.sh``).

In CS 441's order: kinds of bias; biased labels and proxies (the inputs predict the county's
vulnerability without being told it); disaggregated evaluation with intervals that resample whole
counties; fairness metrics (flag rates, error rates, calibration) and the identity that ties them
(Chouldechova, 2017); where to intervene (group thresholds chosen on the validation years, Hardt
et al., 2016); model cards (Mitchell et al., 2019) and datasheets (Gebru et al., 2021). Break it:
the top quartile mixes Cook County (Chicago) with rural counties, and the cells of an
intersectional table hold few bridges that turned poor.
"""

from __future__ import annotations

from pathlib import Path

import cee_l06_linear_classifiers as l06
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from bridges import POOR, load_bridges
from common import DATA_DIR, parse_out_dir, pct, require, rnd, seed_all, write_numbers

SEED = 27
SVI_CSV = DATA_DIR / "raw" / "svi" / "SVI_2022_Illinois_county.csv"
QUARTILES = ("low", "low-mid", "mid-high", "high")
EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)  # quartiles closed on the right; the top one is above 0.75
COOK = "17031"  # Cook County (Chicago)
BOOT = 1000  # resamples of whole counties, within each group, for 95% intervals
CI_KEYS = ("tpr", "fpr", "flagged", "base", "mean_p", "calib_gap")  # the rates with intervals
THRESHOLD = 1 / (1 + l06.R)  # Lecture 6: flag when a miss (R) outweighs a false alarm (1)
OWNERS = {**l06.OWNERS, "other": "other"}
PROXY_DROP = ("lat", "lon")  # the inputs that place a bridge on the map
# Groups of the 14 inputs left without location, each dropped in turn from the proxy test
PROXY_GROUPS = {
    "traffic": ("log_adt", "pct_truck"),
    "material": ("concrete", "steel", "prestressed", "timber"),
    "age": ("age", "years_since_work"),
    "size": ("log_length", "deck_width_m"),
    "ratings": ("deck", "superstructure", "substructure", "rating"),
}
PROXY_C = 1.0  # scikit-learn's default L2 strength, set explicitly (README: name the method)
PROXY_FOLDS = 10  # held-out test: the 102 counties in 10 folds, stratified by SVI quartile
PROXY_REPEATS = 10  # shuffles of the county folds (random_state 0-9)
IDENTITY_PRECISIONS = (0.09, 0.11, 0.13)  # the identity figure's curves, near the quartiles'


# --------------------------------------------------------------------------- data and model


def load_svi() -> pd.DataFrame:
    """Illinois counties with their overall SVI 2022 rank and its quartile.

    Returns:
        One row per county: ``fips`` (five digits), ``county``, ``rpl`` (0 to 1), ``quartile``,
        and the population ``pop``.

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_svi.sh``).
    """
    svi = pd.read_csv(require(SVI_CSV, "fetch_svi.sh"), encoding="utf-8-sig", dtype={"FIPS": str})
    svi = svi[["FIPS", "COUNTY", "RPL_THEMES", "E_TOTPOP"]].rename(
        columns={"FIPS": "fips", "COUNTY": "county", "RPL_THEMES": "rpl", "E_TOTPOP": "pop"}
    )
    assert len(svi) == 102 and svi["rpl"].between(0, 1).all(), "unexpected SVI file"
    svi["quartile"] = pd.cut(svi["rpl"], EDGES, labels=QUARTILES, include_lowest=True)
    return svi[["fips", "county", "rpl", "quartile", "pop"]]


def l06_model() -> dict[str, pd.DataFrame | np.ndarray | StandardScaler]:
    """Lecture 6's classifier, refit with its own functions, and its three sets.

    Returns:
        The training, validation, and test transitions (``train``, ``val``, ``test``), each
        with ``p`` (the model's chance of poor in four years); the weights ``w``; and the
        ``scaler``.
    """
    df = load_bridges()
    sets = {name: l06.transitions(df, start) for name, start in l06.START.items()}
    scaler = StandardScaler().fit(sets["train"][l06.INPUTS].to_numpy(np.float64))
    w, _ = l06.fit_logistic(
        l06.encode(sets["train"], scaler), sets["train"]["poor_next"].to_numpy()
    )
    for d in sets.values():
        d["p"] = l06.predict_proba(w, l06.encode(d, scaler))
    return {**sets, "w": w, "scaler": scaler}


def with_svi(d: pd.DataFrame, svi: pd.DataFrame) -> pd.DataFrame:
    """Join each bridge to its county's SVI rank; add the flag and the owner's name.

    Args:
        d: Transitions with ``county`` (three digits), ``owner``, and ``p``.
        svi: ``load_svi``'s counties.

    Returns:
        ``d`` with the county's ``fips``, ``rpl``, ``quartile``, and ``pop``; ``flag``
        (p > 1/21); and ``owner_name`` (state, county, township, city, or other).
    """
    d = d.assign(fips="17" + d["county"].str.zfill(3))
    out = d.merge(svi, on="fips", how="left", validate="many_to_one", suffixes=("", "_svi"))
    assert out["rpl"].notna().all(), "a bridge in a county without an SVI rank"
    out["flag"] = out["p"] > THRESHOLD
    out["owner_name"] = out["owner"].map(l06.OWNERS).fillna("other")
    return out


# --------------------------------------------------------------------------- metrics


def rates(g: pd.DataFrame, flag: str = "flag") -> dict[str, int | float]:
    """The confusion-matrix rates of one group.

    Args:
        g: Bridges with ``poor_next`` (truth), the flag column, and ``p``.
        flag: Which flag column to use.

    Returns:
        Counts, the base rate, the flag rate, the catch rate (TPR, NaN with no bridge turning
        poor), the false-alarm rate (FPR), the precision (PPV), the mean predicted chance, and
        the calibration gap (mean predicted chance minus base rate), as shares.
    """
    y, f = g["poor_next"].to_numpy() == 1, g[flag].to_numpy()
    tp, fn = int((f & y).sum()), int((~f & y).sum())
    fp, tn = int((f & ~y).sum()), int((~f & ~y).sum())
    mean_p = float(g["p"].mean())
    return {
        "bridges": len(g),
        "counties": int(g["fips"].nunique()),
        "turned_poor": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "base": (tp + fn) / len(g),
        "flagged": (tp + fp) / len(g),
        "tpr": tp / (tp + fn) if tp + fn else np.nan,  # undefined with no bridge turning poor
        "fpr": fp / max(fp + tn, 1),
        "ppv": tp / max(tp + fp, 1),
        "mean_p": mean_p,
        "calib_gap": mean_p - (tp + fn) / len(g),
    }


def county_totals(g: pd.DataFrame, flag: str = "flag") -> tuple[np.ndarray, np.ndarray]:
    """Per-county counts of one group, the counties sorted by FIPS code.

    Args:
        g: One group's bridges with ``fips``, ``poor_next``, the flag column, and ``p``.
        flag: Which flag column to use.

    Returns:
        The sorted county codes, and a ``(K, 6)`` array of each county's TP, FN, FP, TN,
        bridges, and sum of p.
    """
    names, inv = np.unique(g["fips"].to_numpy(dtype=str), return_inverse=True)
    y, f = g["poor_next"].to_numpy() == 1, g[flag].to_numpy(dtype=bool)
    cols = [f & y, ~f & y, f & ~y, ~f & ~y, np.ones(len(g), dtype=bool)]
    totals = np.column_stack(
        [np.bincount(inv, weights=c.astype(np.float64), minlength=len(names)) for c in cols]
        + [np.bincount(inv, weights=g["p"].to_numpy(np.float64), minlength=len(names))]
    )
    assert totals.shape == (len(names), 6), totals.shape
    return names, totals


def cluster_ci(
    g: pd.DataFrame, rng: np.random.Generator, flag: str = "flag"
) -> tuple[dict[str, list[float]], int]:
    """95% intervals of each rate, resampling the group's counties with replacement.

    Bridges in one county share inspectors, budgets, and weather, so the county, not the bridge,
    is the unit that is resampled. Each draw picks the group's K counties with replacement (one
    ``rng.choice`` per draw, over the counties sorted by FIPS code) and pools their counts, which
    equals rebuilding the draw's table of bridges and recomputing ``rates`` on it.

    Args:
        g: One group's bridges.
        rng: The random generator, advanced by ``BOOT`` draws.
        flag: Which flag column to use.

    Returns:
        For each of ``CI_KEYS``, the 2.5% and 97.5% points; and how many draws held no bridge
        that turned poor (their catch rate is undefined and left out of its interval).

    Raises:
        ValueError: If 2.5% or more of the draws leave the catch rate undefined: the
            interval's lower end would then rest on the draws that happen to hold one.
    """
    names, totals = county_totals(g, flag)
    idx = np.stack([rng.choice(len(names), len(names), replace=True) for _ in range(BOOT)])
    assert idx.shape == (BOOT, len(names)), idx.shape
    tp, fn, fp, tn, n, sum_p = totals[idx].sum(axis=1).T
    undefined = tp + fn == 0
    if undefined.mean() >= 0.025:
        raise ValueError(f"{int(undefined.sum())} of {BOOT} draws hold no bridge that turned poor")
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = np.where(undefined, np.nan, tp / (tp + fn))
    draws = {
        "tpr": tpr,
        "fpr": fp / np.maximum(fp + tn, 1),
        "flagged": (tp + fp) / n,
        "base": (tp + fn) / n,
        "mean_p": sum_p / n,
        "calib_gap": sum_p / n - (tp + fn) / n,
    }
    ci = {k: np.nanpercentile(draws[k], [2.5, 97.5]).tolist() for k in CI_KEYS}
    return ci, int(undefined.sum())


def table(d: pd.DataFrame, by: str, rng: np.random.Generator, flag: str = "flag") -> dict:
    """Rates and county-resampled intervals for each value of ``by``.

    The groups are bootstrapped in sorted order (the quartiles in their category order, other
    columns alphabetically), so the intervals do not depend on the order of the rows.

    Args:
        d: Bridges with the column ``by``.
        by: The grouping column, e.g. ``quartile``.
        rng: The random generator, advanced by ``BOOT`` draws per group.
        flag: Which flag column to use.

    Returns:
        For each group: ``rates``, plus ``ci`` (``cluster_ci``'s intervals) and
        ``tpr_undefined_draws``.
    """
    out = {}
    for name, g in d.groupby(by, observed=True, sort=True):
        ci, undefined = cluster_ci(g, rng, flag)
        out[str(name)] = {**rates(g, flag), "ci": ci, "tpr_undefined_draws": undefined}
    return out


def chouldechova(r: dict[str, int | float]) -> float:
    """The false-alarm rate implied by the base rate, precision, and catch rate.

    FPR = p / (1 - p) * (1 - PPV) / PPV * TPR (Chouldechova, 2017): an identity of the confusion
    matrix, so with equal precision, unequal base rates force unequal error rates.

    Args:
        r: A group's ``rates``.

    Returns:
        The implied false-alarm rate, a share.
    """
    b, ppv = r["base"], r["ppv"]
    return b / (1 - b) * (1 - ppv) / ppv * r["tpr"]


def group_thresholds(val: pd.DataFrame, target: float) -> dict[str, float]:
    """Per-quartile thresholds, chosen on the validation years, that catch ``target`` of each.

    Args:
        val: The validation transitions (2016 -> 2020) with ``quartile``, ``p``, ``poor_next``.
        target: The catch rate to reach in every quartile, in (0, 1].

    Returns:
        The largest threshold in each quartile whose validation catch rate is at least
        ``target`` (post-processing, Hardt et al., 2016).

    Raises:
        ValueError: If a quartile has no validation bridge that turned poor, or ``target`` is
            not in (0, 1]: no threshold is then defined.
    """
    if not 0 < target <= 1:
        raise ValueError(f"target catch rate {target} is not in (0, 1]")
    out = {}
    for q, g in val.groupby("quartile", observed=True):
        p_pos = np.sort(g.loc[g["poor_next"] == 1, "p"].to_numpy())[::-1]
        if len(p_pos) == 0:
            raise ValueError(f"quartile {q}: no validation bridge turned poor")
        k = int(np.ceil(target * len(p_pos)))  # flag at least the top k of those that turned poor
        assert 1 <= k <= len(p_pos), k
        out[str(q)] = float(np.nextafter(p_pos[k - 1], -np.inf))  # p > threshold flags it
    return out


def proxy_fit_score(fit: pd.DataFrame, score: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Fit "the county is in the top SVI quartile" on ``fit``; score the bridges of ``score``.

    A logistic regression with an L2 penalty of strength ``PROXY_C`` (Newton's method), the
    inputs standardized with ``fit``'s bridges only.

    Args:
        fit: Bridges with ``cols`` and ``quartile``.
        score: Bridges with ``cols``.
        cols: The inputs.

    Returns:
        Each scored bridge's chance of being in a top-quartile county, shape ``(len(score),)``.
    """
    sc = StandardScaler().fit(fit[cols])
    clf = LogisticRegression(C=PROXY_C, solver="newton-cholesky", max_iter=100)
    clf.fit(sc.transform(fit[cols]), fit["quartile"] == "high")
    return clf.predict_proba(sc.transform(score[cols]))[:, 1]


def county_fold_auc(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int) -> float:
    """The proxy test on counties the classifier never saw: mean AUC over county folds.

    The 102 counties are split into ``PROXY_FOLDS`` folds, stratified by SVI quartile. For each
    fold, a classifier fitted on the other counties' 2014 bridges scores the fold's 2020 bridges,
    and the AUC is computed within the fold. Holding out one county at a time and pooling the
    scores would be biased low: the label is a county attribute, so leaving a county out moves
    the training prevalence against that county's own label, and a classifier with no inputs at
    all would score AUC 0.

    Args:
        train: Training transitions with ``fips``, ``quartile``, and ``cols``.
        test: Test transitions, the same columns.
        cols: The inputs.
        seed: The folds' shuffle (``random_state``).

    Returns:
        The mean of the folds' AUCs.
    """
    county_q = test.groupby("fips")["quartile"].first().astype(str).sort_index()
    assert set(county_q.index) == set(train["fips"]), "train and test hold other counties"
    counties = county_q.index.to_numpy()
    folds = StratifiedKFold(n_splits=PROXY_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for _, hold in folds.split(counties, county_q.to_numpy()):
        held = counties[hold]
        tr, te = ~train["fips"].isin(held), test["fips"].isin(held)
        y = (test.loc[te, "quartile"] == "high").to_numpy()
        assert 0 < y.sum() < len(y), "a fold without both kinds of county"
        aucs.append(roc_auc_score(y, proxy_fit_score(train[tr], test[te], cols)))
    return float(np.mean(aucs))


def proxies(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Can Lecture 6's inputs tell a high-vulnerability county without being told?

    A logistic regression predicts "the county is in the top SVI quartile" from the 16 inputs.
    Trained on the 2014 bridges and scored on the 2020 bridges of the same counties (the
    deployment's situation: next cycle, the same counties); then on counties held out of
    training (``county_fold_auc``, over ``PROXY_REPEATS`` shuffles of the folds: does the signal
    carry to a county the model never saw?). Without location, each group of inputs is also
    dropped in turn.

    Args:
        train: Training transitions with ``fips`` and ``quartile``.
        test: Test transitions with ``fips`` and ``quartile``.

    Returns:
        For all 16 inputs and without latitude and longitude: the AUC, the held-out counties'
        AUC (each shuffle's mean, their median, and their range), and the ROC curve; and,
        without location, the AUC with each group of inputs dropped and with traffic and
        material alone.
    """
    y = (test["quartile"] == "high").to_numpy()
    out = {}
    for name, cols in (
        ("all", list(l06.INPUTS)),
        ("no_location", [c for c in l06.INPUTS if c not in PROXY_DROP]),
    ):
        s = proxy_fit_score(train, test, cols)
        held = [county_fold_auc(train, test, cols, seed) for seed in range(PROXY_REPEATS)]
        fpr, tpr, _ = roc_curve(y, s)
        out[name] = {
            "auc": float(roc_auc_score(y, s)),
            "auc_county_folds": held,
            "auc_county_folds_median": float(np.median(held)),
            "auc_county_folds_range": [min(held), max(held)],
            "_fpr": fpr,
            "_tpr": tpr,
        }
    rest = [c for c in l06.INPUTS if c not in PROXY_DROP]
    assert sorted(c for cs in PROXY_GROUPS.values() for c in cs) == sorted(rest)
    out["no_location_drop"] = {
        g: float(roc_auc_score(y, proxy_fit_score(train, test, [c for c in rest if c not in cs])))
        for g, cs in PROXY_GROUPS.items()
    }
    alone = [*PROXY_GROUPS["traffic"], *PROXY_GROUPS["material"]]
    out["traffic_material_alone"] = float(roc_auc_score(y, proxy_fit_score(train, test, alone)))
    return out


# --------------------------------------------------------------------------- figures

BLUE, ORANGE, GREEN, PINK = aml_style.PALETTE
Q_COLOR = dict(zip(QUARTILES, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"], strict=True))
Q_MARKER = dict(zip(QUARTILES, ["o", "s", "^", "D"], strict=True))  # colour is never the only cue
PREDICTED_HATCH = "//"  # the "mean predicted" bars, beside the plain "turned poor" ones


def bare(ax: Axes) -> None:
    """Hide an axes' frame and ticks (for drawn pictures).

    Args:
        ax: The axes.
    """
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def county_map_figure(test: pd.DataFrame, out_dir: Path) -> None:
    """Each county at the mean position of its bridges, by SVI quartile, sized by its count.

    Args:
        test: The test transitions with ``fips``, ``lat``, ``lon``, and ``quartile``.
        out_dir: The figure directory.
    """
    c = test.groupby("fips").agg(
        lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"), q=("quartile", "first")
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for q in QUARTILES:
        s = c[c["q"] == q]
        ax.scatter(s["lon"], s["lat"], s=s["n"] / 6, color=Q_COLOR[q], marker=Q_MARKER[q],
                   edgecolor=aml_style.INK, lw=0.4, label=q)  # fmt: skip
    cook = c.loc[COOK]
    xy, text_xy = (cook["lon"], cook["lat"]), (cook["lon"] - 2.9, cook["lat"] + 0.75)
    line = {"arrowstyle": "-", "color": aml_style.MUTED}
    ax.annotate("Cook (Chicago)", xy, text_xy, fontsize=10, arrowprops=line)
    ax.set_aspect(1 / np.cos(np.deg2rad(40)))
    bare(ax)
    handles = [Line2D([], [], ls="none", marker=Q_MARKER[q], ms=8, color=Q_COLOR[q],
                      mec=aml_style.INK, mew=0.4, label=q) for q in QUARTILES]  # fmt: skip
    ax.legend(handles=handles, title="SVI quartile", fontsize=9, title_fontsize=9, frameon=False,
              loc="center left", bbox_to_anchor=(1.0, 0.5))  # fmt: skip
    aml_style.save(fig, out_dir, "county_map")


def picture_figures(out_dir: Path) -> None:
    """The one picture, three frames: the pipeline; where bias enters; where we check it.

    Args:
        out_dir: The figure directory.
    """
    light, warn, check = "#e8f0f7", "#fbe3d4", "#e3f1e8"
    arrow = {"arrowstyle": "-|>", "color": aml_style.INK, "lw": 1.2}
    tag = {"facecolor": warn, "edgecolor": "none", "pad": 3}
    steps = [
        ("inspections", 0.12),
        ("inventory\n(NBI)", 0.36),
        ("model\n(Lecture 6)", 0.60),
        ("flag:\np > 1/21", 0.84),
        ("preventive\nwork", 1.08),
    ]
    notes = [
        "labels: an inspector's\nrating, a proxy for need",
        "selection: which bridges\nare in the inventory",
        "proxies: inputs that\nstand for the county",
        "the threshold: who\nis caught, who is missed",
        "who benefits: a flag\nbrings resources",
    ]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        ax.set(xlim=(0, 1.22), ylim=(0, 1))
        bare(ax)
        for i, (text, x) in enumerate(steps):
            box = FancyBboxPatch((x - 0.09, 0.62), 0.18, 0.18, boxstyle="round,pad=0.01",
                                 facecolor=light, edgecolor=aml_style.INK, lw=1)  # fmt: skip
            ax.add_patch(box)
            ax.text(x, 0.71, text, ha="center", va="center", fontsize=10)
            if i < len(steps) - 1:
                ax.annotate("", (x + 0.15, 0.71), (x + 0.09, 0.71), arrowprops=arrow)
            if k >= 2:
                ax.text(x, 0.45, notes[i], ha="center", va="center", fontsize=9, color=ORANGE,
                        bbox=tag)  # fmt: skip
        if k == 3:
            ax.add_patch(FancyBboxPatch((0.03, 0.06), 1.14, 0.18, boxstyle="round,pad=0.01",
                                        facecolor=check, edgecolor=GREEN, lw=1))  # fmt: skip
            ax.text(0.6, 0.15, "checks: evaluate each group separately, with intervals; "
                    "publish a model card and a datasheet", ha="center", va="center",
                    fontsize=10, color=GREEN)  # fmt: skip
        aml_style.save(fig, out_dir, f"picture_{k}")


def disaggregated_figure(by_q: dict, overall: dict, out_dir: Path) -> None:
    """Flag rate, catch rate, false-alarm rate, and calibration by quartile, with intervals.

    Args:
        by_q: Each quartile's ``table`` entry.
        overall: The whole test set's ``rates`` (the dashed lines).
        out_dir: The figure directory.
    """
    fig, axes = plt.subplots(1, 4, figsize=aml_style.FULL, sharex=True)
    x = np.arange(len(QUARTILES))
    panels = (
        ("flagged", "flagged, %"),
        ("tpr", "caught (TPR), %"),
        ("fpr", "false alarms (FPR), %"),
    )
    for ax, (key, title) in zip(axes[:3], panels, strict=True):
        v = np.array([by_q[q][key] for q in QUARTILES]) * 100
        lo = np.array([by_q[q]["ci"][key][0] for q in QUARTILES]) * 100
        hi = np.array([by_q[q]["ci"][key][1] for q in QUARTILES]) * 100
        ax.bar(x, v, color=[Q_COLOR[q] for q in QUARTILES], edgecolor=aml_style.INK, lw=0.5)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", color=aml_style.INK, capsize=3)
        ax.axhline(overall[key] * 100, color=ORANGE, ls="--", lw=1)
        ax.set_title(title, fontsize=11)
    ax = axes[3]
    base = np.array([by_q[q]["base"] for q in QUARTILES]) * 100
    mp = np.array([by_q[q]["mean_p"] for q in QUARTILES]) * 100
    ax.bar(x - 0.2, base, 0.4, color=aml_style.MUTED, label="turned poor")
    ax.bar(x + 0.2, mp, 0.4, color=ORANGE, hatch=PREDICTED_HATCH, label="mean predicted")
    ax.set_title("calibration, %", fontsize=11)
    ax.set_ylim(0, 1.5 * max(base.max(), mp.max()))  # room for the legend above the bars
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for a in axes:
        a.set_xticks(x, ["low", "low-\nmid", "mid-\nhigh", "high"], fontsize=9)
    aml_style.save(fig, out_dir, "disaggregated")


def cook_figure(split: dict, out_dir: Path) -> None:
    """The top quartile's false alarms and calibration: Cook County against the other 25.

    Args:
        split: ``rates`` for "Cook County" and for "the other 25" top-quartile counties.
        out_dir: The figure directory.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=aml_style.FULL)
    names = list(split)
    colors = [ORANGE if n.startswith("Cook") else Q_COLOR["high"] for n in names]
    v = [split[n]["fpr"] * 100 for n in names]
    a.bar(names, v, color=colors)
    for i, val in enumerate(v):
        a.text(i, val + 1, f"{val:.1f}", ha="center", fontsize=10)
    a.set(title="false alarms (FPR), %", ylim=(0, max(v) * 1.2))
    x = np.arange(len(names))
    b.bar(x - 0.2, [split[n]["base"] * 100 for n in names], 0.4, color=aml_style.MUTED,
          label="turned poor")  # fmt: skip
    b.bar(x + 0.2, [split[n]["mean_p"] * 100 for n in names], 0.4, color=ORANGE,
          hatch=PREDICTED_HATCH, label="mean predicted")  # fmt: skip
    b.set_xticks(x, names)
    b.set(title="calibration, %")
    b.legend(fontsize=9, frameon=False)
    aml_style.save(fig, out_dir, "cook")


def identity_figure(by_q: dict, overall: dict, out_dir: Path) -> None:
    """FPR against the base rate when precision and catch rate are fixed: the impossibility.

    Args:
        by_q: Each quartile's ``rates``.
        overall: The whole test set's ``rates`` (its catch rate fixes the curves).
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    b = np.linspace(0.005, 0.10, 200)
    tpr = overall["tpr"]
    for ppv, ls in zip(IDENTITY_PRECISIONS, ("-", "--", ":"), strict=True):
        ax.plot(100 * b, 100 * b / (1 - b) * (1 - ppv) / ppv * tpr, color=aml_style.MUTED, ls=ls,
                label=f"precision {ppv:.2f}")  # fmt: skip
    offsets = {"low": (10, -4), "low-mid": (-58, 4), "mid-high": (8, -14), "high": (10, 6)}
    for q in QUARTILES:
        r = by_q[q]
        ax.scatter(100 * r["base"], 100 * r["fpr"], s=60, color=Q_COLOR[q], marker=Q_MARKER[q],
                   edgecolor=aml_style.INK, zorder=3)  # fmt: skip
        ax.annotate(q, (100 * r["base"], 100 * r["fpr"]), offsets[q], textcoords="offset points",
                    fontsize=10)  # fmt: skip
    ax.set(xlabel="base rate: turned poor, %", ylabel="false alarms (FPR), %", ylim=(0, 60))
    ax.legend(fontsize=9, frameon=False, title=f"catch rate {tpr:.2f}", title_fontsize=9)
    aml_style.save(fig, out_dir, "identity")


def thresholds_figure(post: dict, out_dir: Path) -> None:
    """Catch rate by quartile on the test years: one threshold against per-quartile thresholds.

    Args:
        post: ``post_processing``'s result.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    x = np.arange(len(QUARTILES))
    one = [post["one"][q]["tpr"] * 100 for q in QUARTILES]
    grp = [post["group"][q]["tpr"] * 100 for q in QUARTILES]
    ax.bar(x - 0.2, one, 0.4, color=aml_style.MUTED, label="one threshold, 1/21")
    ax.bar(x + 0.2, grp, 0.4, color=GREEN, hatch="\\\\", label="a threshold per quartile")
    for i in range(len(QUARTILES)):
        ax.text(x[i] - 0.2, one[i] + 1, f"{one[i]:.0f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, grp[i] + 1, f"{grp[i]:.0f}", ha="center", fontsize=9)
    ax.axhline(post["target"] * 100, color=ORANGE, ls="--", lw=1, label="target (validation)")
    ax.set_xticks(x, QUARTILES)
    ax.set(ylabel="caught (TPR), %, test years", ylim=(0, 125), yticks=range(0, 101, 20))
    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=3)  # above the bars
    aml_style.save(fig, out_dir, "thresholds")


def proxy_figure(px: dict, out_dir: Path) -> None:
    """ROC of telling a high-vulnerability county from Lecture 6's inputs.

    Args:
        px: ``proxies``'s result.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    lines = (
        ("all", "all 16 inputs", BLUE, "-"),
        ("no_location", "without latitude and longitude", PINK, "--"),
    )
    for name, label, color, ls in lines:
        ax.plot(px[name]["_fpr"], px[name]["_tpr"], color=color, ls=ls, lw=1.8,
                label=f"{label}: AUC {px[name]['auc']:.2f}")  # fmt: skip
    ax.plot([0, 1], [0, 1], color=aml_style.MUTED, ls=":", lw=1)
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    aml_style.save(fig, out_dir, "proxies")


def intersection_figure(cells: dict, out_dir: Path) -> None:
    """Catch rate by quartile and owner; each cell shows how many bridges turned poor in it.

    Args:
        cells: The quartile-by-owner ``table`` entries, keyed "quartile | owner".
        out_dir: The figure directory.
    """
    owners = ["state", "county", "township", "city"]
    M = np.full((len(QUARTILES), len(owners)), np.nan)
    N = np.zeros_like(M)
    for i, q in enumerate(QUARTILES):
        for j, o in enumerate(owners):
            c = cells.get(f"{q} | {o}")
            if c is not None:
                M[i, j], N[i, j] = 100 * c["tpr"], c["turned_poor"]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.imshow(M, cmap="Blues", vmin=40, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.0f}%\n{int(N[i, j])} poor", ha="center", va="center", fontsize=9,
                color="white" if v > 80 else aml_style.INK)  # fmt: skip
    ax.set_xticks(range(len(owners)), owners)
    ax.set_yticks(range(len(QUARTILES)), QUARTILES)
    ax.set(xlabel="owner (Item 22)", ylabel="SVI quartile")
    aml_style.save(fig, out_dir, "intersection")


def title_art_figure(out_dir: Path) -> None:
    """Title art: a balance beam over four blocks of different heights, one lit.

    Args:
        out_dir: The figure directory.
    """
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0c12", "#141824", "#191520", "#08080b"), grain=0.08)
    for i, h in enumerate((0.30, 0.42, 0.36, 0.48)):
        x0 = 0.18 + 0.17 * i
        c.glaze(c.polygon([(x0, top * 0.12), (x0 + 0.12, top * 0.12), (x0 + 0.12, top * h),
                           (x0, top * h)]), "#8fb3c9" if i < 3 else "#eaa548", 0.6)  # fmt: skip
    c.glow(c.lines([[(0.12, top * 0.62), (0.88, top * 0.56)]], 3.0, soft=1.0), "#eaa548", 0.45,
           spread=8)  # fmt: skip
    c.glow(c.disc((0.5, top * 0.59), 0.025, soft=3), "#eaa548", 0.5, spread=16)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: four pale blocks of different heights, the last one "
        "gold; above them a thin gold beam, slightly tilted, balanced on a small glowing point.",
    )


# --------------------------------------------------------------------------- main


def rounded(r: dict) -> dict:
    """A group's rates for numbers.json: counts as integers, shares as percentages.

    Args:
        r: A group's ``rates``, or its ``table`` entry (with ``ci``).

    Returns:
        The counts, ``healthy`` (FP + TN), each share as ``<name>_pct`` (the calibration gap in
        percentage points), and, if present, the intervals as ``ci_pct``.
    """
    out = {k: r[k] for k in ("bridges", "counties", "turned_poor", "tp", "fn", "fp", "tn")}
    out["healthy"] = r["fp"] + r["tn"]
    for k in ("base", "flagged", "tpr", "fpr", "ppv", "mean_p", "calib_gap"):
        out[f"{k}_pct"] = pct(r[k], 1)
    if "ci" in r:
        out["ci_pct"] = {k: [pct(v[0], 1), pct(v[1], 1)] for k, v in r["ci"].items()}
    return out


def post_processing(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Equal catch rates by per-quartile thresholds, chosen on validation, judged on test.

    Args:
        val: The validation transitions with ``quartile``, ``p``, ``poor_next``, ``flag``.
        test: The test transitions, the same columns.

    Returns:
        The target (the single threshold's validation catch rate), the thresholds, each
        quartile's test ``rates`` under one threshold and under its own, and the flags, catches,
        and expected costs per bridge (Lecture 6's) of both rules.
    """
    target = rates(val)["tpr"]  # the validation catch rate of the single threshold
    th = group_thresholds(val, target)
    test = test.assign(flag_group=test["p"] > test["quartile"].astype(str).map(th))
    y = test["poor_next"].to_numpy()
    one = {q: rates(g) for q, g in test.groupby("quartile", observed=True)}
    grp = {q: rates(g, "flag_group") for q, g in test.groupby("quartile", observed=True)}
    return {
        "target": target,
        "thresholds": th,
        "one": one,
        "group": grp,
        "flags_one": int(test["flag"].sum()),
        "flags_group": int(test["flag_group"].sum()),
        "caught_one": int((test["flag"] & (y == 1)).sum()),
        "caught_group": int((test["flag_group"] & (y == 1)).sum()),
        "cost_one": l06.expected_cost(test["flag"].to_numpy(), y),
        "cost_group": l06.expected_cost(test["flag_group"].to_numpy(), y),
    }


def main() -> None:
    """Refit Lecture 6's model, evaluate it by county vulnerability, and write the numbers."""
    out_dir = parse_out_dir((__doc__ or "").split("\n")[0])
    seed_all(SEED)
    aml_style.apply()
    rng = np.random.default_rng(SEED)
    svi = load_svi()
    model = l06_model()
    train, val, test = (with_svi(model[k], svi) for k in ("train", "val", "test"))
    assert len(test) == 19_637, "Lecture 6's test set changed"
    overall = rates(test)
    by_q = table(test, "quartile", rng)
    high = test[test["quartile"] == "high"]
    split = {
        "Cook County": rates(high[high["fips"] == COOK]),
        "the other 25": rates(high[high["fips"] != COOK]),
    }
    high_val = val[val["quartile"] == "high"]
    split_val = {
        "Cook County": rates(high_val[high_val["fips"] == COOK]),
        "the other 25": rates(high_val[high_val["fips"] != COOK]),
    }
    assert high["fips"].nunique() == 26, "the top quartile's county count changed"
    identity = {q: {"fpr": by_q[q]["fpr"], "implied": chouldechova(by_q[q])} for q in QUARTILES}
    for q, v in identity.items():
        assert abs(v["fpr"] - v["implied"]) < 1e-12, f"the identity fails in {q}"
    px = proxies(train, test)
    post = post_processing(val, test)
    test["cell"] = test["quartile"].astype(str) + " | " + test["owner_name"]
    # Only the cells the slides show are bootstrapped ("other" owners are left out), after the
    # quartiles, from the same generator
    cells = table(test[test["owner_name"] != "other"], "cell", rng)
    assert len(cells) == 16 and not any(k.endswith("| other") for k in cells), sorted(cells)

    county_map_figure(test, out_dir)
    picture_figures(out_dir)
    disaggregated_figure(by_q, overall, out_dir)
    cook_figure(split, out_dir)
    identity_figure(by_q, overall, out_dir)
    thresholds_figure(post, out_dir)
    proxy_figure(px, out_dir)
    intersection_figure(cells, out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("disaggregated", "cook", "identity", "intersection"))

    counties = svi.groupby("quartile", observed=True).agg(
        counties=("fips", "size"),
        pop=("pop", "sum"),
        rpl_min=("rpl", "min"),
        rpl_max=("rpl", "max"),
    )
    bridges_2020 = test.groupby("quartile", observed=True).size()
    numbers = {
        "svi": {
            q: {
                "counties": int(counties.loc[q, "counties"]),
                "population": int(counties.loc[q, "pop"]),
                "rpl_range": [
                    rnd(counties.loc[q, "rpl_min"], 4),
                    rnd(counties.loc[q, "rpl_max"], 4),
                ],
                "bridges_per_10k": rnd(bridges_2020[q] / counties.loc[q, "pop"] * 1e4, 1),
            }
            for q in QUARTILES
        },
        "cook": {
            "rpl": rnd(svi.loc[svi["fips"] == COOK, "rpl"].iloc[0], 4),
            "bridges": int((test["fips"] == COOK).sum()),
            "share_of_high_pct": pct((high["fips"] == COOK).mean(), 0),
        },
        "threshold": rnd(THRESHOLD, 4),
        "R": l06.R,
        "overall": rounded(overall),
        "by_quartile": {q: rounded(by_q[q]) for q in QUARTILES},
        "by_quartile_val": {
            str(q): rounded(rates(g)) for q, g in val.groupby("quartile", observed=True)
        },
        "overall_val": rounded(rates(val)),
        "high_split": {k: rounded(v) for k, v in split.items()},
        "high_split_val": {
            k: {**rounded(v), "expected": rnd(v["mean_p"] * v["bridges"], 1)}
            for k, v in split_val.items()
        },
        "cook_expected_test": rnd(
            split["Cook County"]["mean_p"] * split["Cook County"]["bridges"], 1
        ),
        "identity": {q: {k: pct(x, 2) for k, x in v.items()} for q, v in identity.items()},
        "proxies": {
            **{
                k: {
                    "auc": rnd(px[k]["auc"], 3),
                    "auc_county_folds": [rnd(a, 3) for a in px[k]["auc_county_folds"]],
                    "auc_county_folds_median": rnd(px[k]["auc_county_folds_median"], 3),
                    "auc_county_folds_range": [rnd(a, 3) for a in px[k]["auc_county_folds_range"]],
                    # how much of the AUC above chance survives on counties never seen
                    "signal_kept_pct": pct(
                        (px[k]["auc_county_folds_median"] - 0.5) / (px[k]["auc"] - 0.5), 0
                    ),
                }
                for k in ("all", "no_location")
            },
            "no_location_drop": {g: rnd(a, 3) for g, a in px["no_location_drop"].items()},
            "traffic_material_alone": rnd(px["traffic_material_alone"], 3),
        },
        "identity_figure": {
            "precisions": list(IDENTITY_PRECISIONS),
            "catch_rate": rnd(overall["tpr"], 2),
        },
        "post": {
            "target_pct": pct(post["target"], 1),
            "thresholds": {q: rnd(v, 4) for q, v in post["thresholds"].items()},
            "one": {q: rounded(post["one"][q]) for q in QUARTILES},
            "group": {q: rounded(post["group"][q]) for q in QUARTILES},
            "flags_one": post["flags_one"],
            "flags_group": post["flags_group"],
            "caught_one": post["caught_one"],
            "caught_group": post["caught_group"],
            "cost_one": rnd(post["cost_one"], 3),
            "cost_group": rnd(post["cost_group"], 3),
        },
        "cells": {k: rounded(v) for k, v in cells.items()},
        "cells_under_30_poor": sum(v["turned_poor"] < 30 for v in cells.values()),
        # draws whose catch rate is undefined (no bridge turned poor), left out of the interval
        "tpr_undefined_draws": {
            k: v["tpr_undefined_draws"]
            for k, v in {**by_q, **cells}.items()
            if v["tpr_undefined_draws"]
        },
        "settings": {
            "seed": SEED,
            "boot": BOOT,
            "poor": POOR,
            "proxy_c": PROXY_C,
            "proxy_folds": PROXY_FOLDS,
            "proxy_repeats": PROXY_REPEATS,
        },
    }
    write_numbers(numbers, out_dir)


if __name__ == "__main__":
    main()
