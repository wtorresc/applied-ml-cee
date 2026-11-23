"""Figures and numbers for CEE Lecture 4: load profiles, retrieval, k-means, and PCA (BDG2).

Run::

    pixi run -e aml-env python figures/cee_l04_load_profiles.py \
        --out build/fig/cee_l04_load_profiles

Writes the PNG figures, ``numbers.json`` (every number quoted on the slides), and ``numbers.tex``
(the trace of building Q, for the TikZ diagram). Data: ``data/derived/bdg2_daily_electricity.npz``
and ``data/raw/bdg2/metadata.csv`` from ``scripts/fetch_bdg2.sh`` (Building Data Genome 2;
Miller et al., 2020).

One object throughout: a building's **typical weekday**, the mean kW in each hour of the day over
the weekdays of one year with all 24 hours metered. Its mean is the building's **size** (kW); the
profile divided by its mean is its **shape** (unitless, mean 1). Models are fit on 2016 shapes of
buildings at training sites and tested on 2017 shapes of buildings at held-out sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import aml_style
import title_art
from buildings import MIN_COMPLETE, Portfolio, build_portfolio, load_bdg2, typical_weekday
from common import parse_out_dir, pct, rmse, seed_all, tex_num, write_numbers, write_tex_macros

SEED: Final[int] = 0
REPEATS: Final[int] = 30  # repeated site-held-out splits
TEST_SIZE: Final[float] = 0.25  # share of sites held out
R: Final[int] = 3  # modes kept in the picture and the trace
K: Final[int] = 4  # typical days (k-means)
N_INIT_TYPICAL: Final[int] = 20  # k-means++ starts for the typical days (best J kept)
N_INIT_BRACKET: Final[int] = 10  # k-means++ starts per split for the bracket's typical days
NIGHT: Final[int] = 0  # four-hour block 0-4 h
DAY: Final[int] = 3  # four-hour block 12-16 h
NIGHT_HOURS: Final[int] = 8  # rough annualization of the excess night load (midnight to 8 am)
VERIFY_TOL: Final[float] = 1e-13  # our PCA against scikit-learn's, signs aligned
# scikit-learn sums k-means' J in a thread-dependent order: our J and its J differ by 0 or a few
# 1e-16 (relative) from run to run, so the slide quotes this bound, not the value.
KMEANS_J_TOL: Final[float] = 1e-15
SUMMER: Final[tuple[int, ...]] = (6, 7, 8)  # months of the summer typical weekday
WINTER: Final[tuple[int, ...]] = (1, 2, 12)  # months of the winter typical weekday (same year)
MIN_SEASON_DAYS: Final[int] = 40  # complete weekdays a season's typical weekday needs
VISITS: Final[int] = 10  # visits in the Decide step's headline comparison
BOOT_DRAWS: Final[int] = 1000  # office bootstrap of the visit ranking (offices resampled)
HOURS: Final[np.ndarray] = np.arange(24)
HOURS.setflags(write=False)
# An SVD returns each mode with an arbitrary sign (LAPACK's choice). Fix it by a rule, so that the
# scores repeat on any machine: mode k is flipped unless its loadings are larger over the hours
# ``pos`` than over the hours ``neg``. Shapes: mode 1, the afternoon (12-16 h) above the night
# (0-4 h); mode 2, the evening (16-20 h) above the early morning (4-8 h); mode 3, the afternoon
# (12-16 h) above the morning (8-12 h), a later day. Raw kW: mode 1 sums positive.
SignRules = tuple[tuple[slice, slice], ...]
SHAPE_SIGN_RULES: Final[SignRules] = (
    (slice(12, 16), slice(0, 4)),
    (slice(16, 20), slice(4, 8)),
    (slice(12, 16), slice(8, 12)),
)
KW_SIGN_RULES: Final[SignRules] = ((slice(0, 24), slice(0, 0)),)
# Numbers quoted from cited sources (not computed here), kept with every other slide number.
CITED: Final[dict[str, dict[str, object]]] = {
    "buildingsbench": {
        "simulated_buildings": 900_000,
        "real_buildings": "over 1,900",
        "real_datasets": 7,
        "source": "Emami, Sahu & Graf (2023), NeurIPS 36 Datasets and Benchmarks, "
        "abstract; BDG-2 is one of the real commercial datasets",
    },
    "faiss": {
        "scale": "billion-scale",
        "source": "Johnson, Douze & Jégou (2021), IEEE Trans. Big Data 7(3), title",
    },
    "emeans": {
        "epochs_to_beat_lloyd": "1.4 to 2.6",
        "lloyd_epochs": 25,
        "dino10b_embeddings": "10 billion",
        "dino10b_clusters": "16 million",
        "dino10b_epochs": 2.5,
        "source": "Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze & Jégou (2026), "
        "e-means: outperforming k-means (in one-tenth the epochs), "
        "arXiv:2611.09501, abstract (read 2026-11-16): beats the loss of "
        "Lloyd k-means at 25 epochs in 1.4 to 2.6 epochs; partitions DINO10B, "
        "10 billion DINO embeddings, into 16 million clusters in 2.5 epochs, "
        "reaching a loss below Lloyd's at 25 epochs. Authors at FAIR at Meta.",
    },
}


# ---------------------------------------------------------------------------------------------
# Data


def shape(P: np.ndarray) -> np.ndarray:
    """Divide each profile by its own mean: the shape, unitless with mean 1.

    Args:
        P: Profiles in kW, one per row, shape ``(N, 24)``.

    Returns:
        The shapes, shape ``(N, 24)``, each row with mean 1.
    """
    return P / P.mean(axis=1, keepdims=True)


def blocks(P: np.ndarray) -> np.ndarray:
    """Mean of each four-hour block of the day.

    Args:
        P: Profiles, one per row, shape ``(N, 24)``.

    Returns:
        Block means, shape ``(N, 6)``: 0-4 h, 4-8 h, ..., 20-24 h.
    """
    out = P.reshape(len(P), 6, 4).mean(axis=2)
    assert out.shape == (len(P), 6), out.shape
    return out


def night_day(P: np.ndarray) -> np.ndarray:
    """Night (0-4 h) over afternoon (12-16 h) load: the hand-made measure of a day-night swing.

    Args:
        P: Profiles, one per row, shape ``(N, 24)``.

    Returns:
        The ratio for each profile, shape ``(N,)``.
    """
    b = blocks(P)
    return b[:, NIGHT] / b[:, DAY]


def weekday_counts(dates: pd.DatetimeIndex) -> dict[int, int]:
    """Number of weekdays (Monday to Friday) in each year of the data.

    Args:
        dates: The days of the hourly file.

    Returns:
        Year to its number of weekdays.
    """
    wk = pd.Series(np.asarray(dates.dayofweek < 5), index=dates.year)
    return {int(y): int(n) for y, n in wk.groupby(level=0).sum().items()}


def campus_split(pf: Portfolio) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the campus (site) with the most offices; train on every other site.

    This is the energy manager's campus in the lecture: its buildings are new to every model.

    Args:
        pf: The eligible buildings.

    Returns:
        Row indices of the training buildings and of the held-out campus's buildings.
    """
    offices = pd.Series(pf.site[pf.use == "Office"]).value_counts()
    campus = offices.index[0]
    te = np.flatnonzero(pf.site == campus)
    tr = np.flatnonzero(pf.site != campus)
    return tr, te


def site_splits(site: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``REPEATS`` splits that hold out a quarter of the sites, each site kept whole.

    Args:
        site: Each building's site code.

    Returns:
        ``REPEATS`` pairs of training and held-out row indices.
    """
    gss = GroupShuffleSplit(n_splits=REPEATS, test_size=TEST_SIZE, random_state=SEED)
    return list(gss.split(np.zeros(len(site)), groups=site))


# ---------------------------------------------------------------------------------------------
# Methods, from scratch (the slide listings)


def pca_svd(P: np.ndarray, r: int) -> tuple[np.ndarray, ...]:
    """Mean shape, top-r modes, their variances, and the scores."""
    mu = P.mean(axis=0)  # mean shape, (24,)
    _, sig, Vt = np.linalg.svd(P - mu, full_matrices=False)
    lam = sig**2 / (len(P) - 1)  # variance along each mode
    V = Vt[:r].T  # modes as columns, (24, r)
    Z = (P - mu) @ V  # scores, (N, r)
    assert V.shape == (P.shape[1], r) and Z.shape == (len(P), r)
    return mu, V, lam, Z


def orient(V: np.ndarray, rules: SignRules = SHAPE_SIGN_RULES) -> np.ndarray:
    """Fix the arbitrary sign of the leading modes by ``rules`` (see ``SHAPE_SIGN_RULES``).

    Args:
        V: Modes as columns, shape ``(24, r)``.
        rules: ``(pos, neg)`` hours for modes 1, 2, ...; later modes keep their sign.

    Returns:
        ``V`` with each ruled mode flipped where needed.

    Raises:
        ValueError: If a rule cannot decide a sign (equal loadings over ``pos`` and ``neg``).
    """
    signs = np.ones(V.shape[1])
    for k, (pos, neg) in enumerate(rules[: V.shape[1]]):
        signs[k] = np.sign(V[pos, k].sum() - V[neg, k].sum())
    if np.any(signs == 0):
        raise ValueError("a sign rule is undecided")
    return V * signs


def kmeans(
    X: np.ndarray, C0: np.ndarray, max_iter: int = 300
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Lloyd's algorithm from the given centers; stops when no assignment changes.

    Args:
        X: Points, shape ``(N, D)``.
        C0: Starting centers, shape ``(K, D)``.
        max_iter: Iteration cap.

    Returns:
        Centers ``(K, D)``, assignments ``(N,)``, and the objective J after each assignment.

    Raises:
        ValueError: If a cluster loses all its points (its center would become NaN, and the
            next assignment would send every point to it), or if ``max_iter`` is below 1.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    C, z, history = C0.copy(), None, []
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # (N, K)
        z_new = d2.argmin(axis=1)  # assign
        history.append(float(d2[np.arange(len(X)), z_new].sum()))
        if z is not None and np.array_equal(z_new, z):
            break
        z = z_new
        if np.bincount(z, minlength=len(C)).min() == 0:
            raise ValueError("a cluster lost all its points")
        C = np.stack([X[z == k].mean(axis=0) for k in range(len(C))])  # update
    assert z is not None and C.shape == C0.shape
    return C, z, history


# ---------------------------------------------------------------------------------------------
# Step 1: the data and the attempt


def data_summary(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], weekdays: dict[int, int]
) -> dict[str, object]:
    """Counts for the data slide.

    Args:
        pf: The eligible buildings.
        split: Training and held-out (Q's campus) row indices.
        weekdays: Weekdays in each year (``weekday_counts``).

    Returns:
        The data slide's counts, and the sizes of the buildings in kW.
    """
    tr, te = split
    uses = pd.Series(pf.use).value_counts()
    return {
        **pf.counts,
        "sites": int(len(set(pf.site))),
        "train_buildings": int(len(tr)),
        "test_buildings": int(len(te)),
        "train_sites": int(len(set(pf.site[tr]))),
        "test_sites": int(len(set(pf.site[te]))),
        "test_site_names": sorted(set(pf.site[te])),
        "weekdays_2016": weekdays[2016],
        "weekdays_2017": weekdays[2017],
        "min_complete_pct": round(100 * MIN_COMPLETE),
        "uses_top": {u: int(n) for u, n in uses.head(5).items()},
        "kw_median": round(float(np.median(pf.P16.mean(axis=1))), 1),
        "kw_min": round(float(pf.P16.mean(axis=1).min()), 1),
        "kw_max": round(float(pf.P16.mean(axis=1).max()), 1),
    }


def attempt_row(pf: Portfolio, i: int, letter: str) -> dict[str, object]:
    """One building of the attempt's table: 2016 size, four-hour blocks, night/day, intensity.

    Args:
        pf: The eligible buildings.
        i: The building's row in ``pf``.
        letter: Its letter on the slide (Q, A-D).

    Returns:
        The table row, rounded as the slide shows it, with the lowest and highest hour of the
        building's shape (the readings of the attempt figure).
    """
    kw = float(pf.P16[i].mean())
    p = shape(pf.P16[[i]])[0]
    return {
        "letter": letter,
        "row": i,
        "name": str(pf.name[i]),
        "site": str(pf.site[i]),
        "use": str(pf.use[i]),
        "kw": round(kw, 0),
        "blocks_kw": [round(float(v), 0) for v in blocks(pf.P16[[i]])[0]],
        "night_day": round(float(night_day(pf.P16[[i]])[0]), 2),
        "sqm": round(float(pf.sqm[i]), 0),
        "w_per_m2": round(1000 * kw / float(pf.sqm[i]), 0),
        "shape_min": round(float(p.min()), 2),
        "shape_max": round(float(p.max()), 2),
    }


def pick_attempt(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], weekdays_2016: int
) -> dict[str, object]:
    """Building Q and the four buildings A-D of the attempt.

    Q is the held-out office with the largest excess night load: its 0-4 h load minus the median
    night/day ratio of the training offices times its 12-16 h load. Two of A-D are the offices on
    Q's campus closest to it in size (its natural peers); the other two are Q's two nearest
    training buildings by shape. A-D are lettered by raw distance (kW) to Q, nearest first.

    Args:
        pf: The eligible buildings.
        split: Training and held-out (Q's campus) row indices.
        weekdays_2016: Weekdays of 2016, for the rough annual excess.

    Returns:
        The table rows (Q, A-D) with their distances to Q, Q's row, the office yardstick
        (rounded and exact), Q's excess, and the check question's doubled-Q distance.
    """
    tr, te = split
    rho = night_day(pf.P16)
    b = blocks(pf.P16)
    # The office yardstick: the median night/day ratio of the training offices (0.658), rounded
    # to two decimals (0.66) so that every excess on the slides can be checked by hand. The
    # notes state both; owner's choice, 2026-11-20 (q1 D12).
    ref_exact = float(np.median(rho[tr][pf.use[tr] == "Office"]))
    ref = round(ref_exact, 2)
    excess = b[:, NIGHT] - ref * b[:, DAY]
    offices_te = te[pf.use[te] == "Office"]
    q = int(offices_te[np.argmax(excess[offices_te])])
    size = pf.P16.mean(axis=1)
    peers = np.flatnonzero((pf.site == pf.site[q]) & (pf.use == "Office"))
    peers = peers[peers != q]
    by_size = peers[np.argsort(np.abs(size[peers] - size[q]), kind="stable")[:2]]
    S = shape(pf.P16)
    d_shape_tr = np.sqrt(((S[tr] - S[q]) ** 2).sum(axis=1))
    by_shape = tr[np.argsort(d_shape_tr, kind="stable")[:2]]
    four = np.concatenate([by_size, by_shape])
    assert four.shape == (4,), four.shape
    d_raw = np.sqrt(((pf.P16[four] - pf.P16[q]) ** 2).sum(axis=1))
    d_shp = np.sqrt(((S[four] - S[q]) ** 2).sum(axis=1))
    order = np.argsort(d_raw, kind="stable")
    four, d_raw, d_shp = four[order], d_raw[order], d_shp[order]

    rows = [attempt_row(pf, q, "Q")]
    rows += [attempt_row(pf, int(i), L) for i, L in zip(four, "ABCD", strict=True)]
    for r_, dr, ds in zip(rows[1:], d_raw, d_shp, strict=True):
        r_["d_raw_kw"] = round(float(dr), 0)
        r_["d_shape"] = round(float(ds), 3)
    peer_letters = [r_["letter"] for r_ in rows[1:] if r_["row"] in by_size]
    shape_letters = [r_["letter"] for r_ in rows[1:] if r_["row"] in by_shape]
    # Check question: double every hour of Q. Raw distance = ||x_Q||; shape distance = 0.
    double_raw = float(np.sqrt((pf.P16[q] ** 2).sum()))
    # The notes annualize the excess as shown: 214 kW x 8 h x 261 weekdays / 1000 = 447 MWh.
    excess_kw_shown = round(float(excess[q]))
    return {
        "rows": rows,
        "q": q,
        "office_ref_night_day": ref,
        "office_ref_exact": round(ref_exact, 4),
        "n_train_offices": int((pf.use[tr] == "Office").sum()),
        "q_excess_kw": float(excess_kw_shown),
        "q_excess_mwh_year": round(excess_kw_shown * NIGHT_HOURS * weekdays_2016 / 1000),
        "night_hours": NIGHT_HOURS,
        "weekdays": weekdays_2016,
        "peer_letters": peer_letters,
        "shape_letters": shape_letters,
        "double_raw_kw": round(double_raw, 0),
        "double_farther_than_all": bool(double_raw > d_raw.max()),
    }


def attempt_figure(pf: Portfolio, att: dict[str, object], out_dir: Path) -> None:
    """Shapes of Q and A-D over the day: the worked example of dividing out size.

    Args:
        pf: The eligible buildings.
        att: ``pick_attempt``'s result.
        out_dir: The figure directory.
    """
    S = shape(pf.P16)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    rows = att["rows"]
    assert isinstance(rows, list)
    for j, r_ in enumerate(rows[1:]):
        ax.plot(
            HOURS,
            S[r_["row"]],
            color=aml_style.PALETTE[j],
            ls=aml_style.LINESTYLES[j],
            marker=aml_style.MARKERS[j],
            markevery=4,
            ms=7,
            label=f"{r_['letter']}: {r_['kw']:.0f} kW",
        )
    ax.plot(
        HOURS, S[rows[0]["row"]], color=aml_style.INK, lw=3.2, label=f"Q: {rows[0]['kw']:.0f} kW"
    )
    ax.set_xticks([0, 6, 12, 18, 23], ["0", "6", "12", "18", "23"])
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("load / daily mean")
    ax.set_ylim(0.5, 1.45)
    ax.legend(loc="upper left", ncol=2, fontsize=12, handletextpad=0.3, columnspacing=0.8)
    aml_style.save(fig, out_dir, "attempt_shapes")


def lloyd_by_hand(pf: Portfolio, att: dict[str, object]) -> dict[str, object]:
    """One Lloyd step with K = 2 on the block shapes of the five buildings Q, A-D.

    Each building's six four-hour blocks, divided by their mean; started at A and C.

    Args:
        pf: The eligible buildings.
        att: ``pick_attempt``'s result.

    Returns:
        The block shapes, the groups after the first assignment, J at the start and after the
        update, the updated centers, and whether Lloyd's algorithm stops there.
    """
    rows = att["rows"]
    assert isinstance(rows, list)
    idx = [r_["row"] for r_ in rows]
    Bs = blocks(pf.P16[idx])
    Bs = Bs / Bs.mean(axis=1, keepdims=True)  # block shapes, mean 1
    C0 = Bs[[1, 3]]  # start at A and C
    d2 = ((Bs[:, None, :] - C0[None, :, :]) ** 2).sum(-1)
    z = d2.argmin(axis=1)
    J0 = float(d2[np.arange(5), z].sum())
    C1 = np.stack([Bs[z == k].mean(axis=0) for k in range(2)])
    J1 = float(((Bs - C1[z]) ** 2).sum())
    _, zf, hist = kmeans(Bs, C0)
    letters = ["Q", "A", "B", "C", "D"]
    return {
        "block_shapes": {L: [round(float(v), 2) for v in Bs[i]] for i, L in enumerate(letters)},
        "start": ["A", "C"],
        "groups": [[letters[i] for i in np.flatnonzero(z == k)] for k in range(2)],
        "J_start": round(J0, 4),
        "J_after_update": round(J1, 4),
        "centers_after_update": [[round(float(v), 2) for v in c] for c in C1],
        "converged_J": round(hist[-1], 4),
        "converged_same": bool(np.array_equal(zf, z)),
    }


# ---------------------------------------------------------------------------------------------
# Step 4: build and verify


def one_digit(x: float) -> float:
    """Round to one significant digit, for round-off quoted as an order of magnitude.

    Args:
        x: A non-negative difference, e.g. ``1.3e-15``.

    Returns:
        The value with one significant digit (``1e-15``), as a float; 0.0 stays 0.0.
    """
    return float(f"{x:.0e}")


def verify_pca(S_tr: np.ndarray) -> dict[str, object]:
    """Compare ``pca_svd`` with scikit-learn's PCA on the training shapes (signs aligned).

    The differences are round-off and vary by machine; the slide quotes the tolerance.

    Args:
        S_tr: The training shapes, shape ``(N, 24)``.

    Returns:
        The largest differences in variances, modes, and scores (one significant digit), how
        many modes came out with the opposite sign, and the tolerance.

    Raises:
        ValueError: If any difference reaches ``VERIFY_TOL``.
    """
    _, V, lam, Z = pca_svd(S_tr, R)
    ref = PCA(n_components=R, svd_solver="full").fit(S_tr)
    sign = np.sign((V.T * ref.components_).sum(axis=1))
    var_err = float(np.abs(lam[:R] - ref.explained_variance_).max())
    comp_err = float(np.abs(V.T * sign[:, None] - ref.components_).max())
    score_err = float(np.abs(Z * sign - ref.transform(S_tr)).max())
    flipped = int((sign < 0).sum())
    if max(var_err, comp_err, score_err) >= VERIFY_TOL:
        raise ValueError(f"pca_svd differs from scikit-learn: {var_err, comp_err, score_err}")
    return {
        "max_var_diff": one_digit(var_err),
        "max_mode_diff": one_digit(comp_err),
        "max_score_diff": one_digit(score_err),
        "modes_with_flipped_sign": flipped,
        "tolerance": VERIFY_TOL,
    }


def verify_kmeans(S_tr: np.ndarray) -> dict[str, object]:
    """Our Lloyd iterations against scikit-learn's, from the same K random training shapes.

    Args:
        S_tr: The training shapes, shape ``(N, 24)``.

    Returns:
        Whether the labels agree, our final J, the bound on the relative J difference, the
        number of iterations, and whether J never rose.

    Raises:
        ValueError: If the labels differ, J differs by ``KMEANS_J_TOL`` or more, or J rose.
    """
    rng = np.random.default_rng(SEED)
    C0 = S_tr[rng.choice(len(S_tr), size=K, replace=False)]
    _, z, hist = kmeans(S_tr, C0)
    ref = KMeans(n_clusters=K, init=C0, n_init=1, algorithm="lloyd", tol=0.0, max_iter=300).fit(
        S_tr
    )
    same = bool(np.array_equal(z, ref.labels_))
    J_rel = abs(hist[-1] - ref.inertia_) / ref.inertia_
    decreasing = bool(np.all(np.diff(hist) <= 1e-12))
    if not (same and J_rel < KMEANS_J_TOL and decreasing):
        raise ValueError(f"Lloyd differs from scikit-learn: {same, J_rel, decreasing}")
    return {
        "same_labels": same,
        "J": round(hist[-1], 2),
        "J_rel_diff_below": KMEANS_J_TOL,
        "iterations": len(hist),
        "J_never_rose": decreasing,
    }


def modes_figure(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path
) -> dict[str, object]:
    """Mean shape plus and minus two standard deviations along modes 1 and 2.

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices (the modes are fit on the training shapes).
        out_dir: The figure directory.

    Returns:
        Each mode's share of the variance and the cumulative shares, the correlation of the
        mode-1 score with the night/day ratio, mode 2's peak hour, and the figure's readings:
        the mean shape's and the mode-1 "+2 SD" shape's lowest and highest values.
    """
    tr, _ = split
    S = shape(pf.P16)
    mu, V, lam, _ = pca_svd(S[tr], 24)
    V = orient(V)
    share = lam / lam.sum()
    plus1 = mu + 2 * np.sqrt(lam[0]) * V[:, 0]
    z1 = (S[tr] - mu) @ V[:, 0]
    corr = float(np.corrcoef(z1, night_day(pf.P16[tr]))[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=aml_style.SIDE, sharey=True)
    titles = ["mode 1", "mode 2"]
    for k, ax in enumerate(axes):
        sd = np.sqrt(lam[k])
        ax.plot(HOURS, mu, color=aml_style.MUTED, lw=1.8, ls=":", label="mean shape")
        ax.plot(HOURS, mu + 2 * sd * V[:, k], color=aml_style.PALETTE[0], ls="-", label="+2 SD")
        ax.plot(HOURS, mu - 2 * sd * V[:, k], color=aml_style.PALETTE[1], ls="--", label="−2 SD")
        ax.set_title(f"{titles[k]}: {pct(share[k], 0):.0f}%", loc="left")
        ax.set_xticks([0, 12, 23], ["0", "12", "23"])
        ax.set_xlabel("hour")
    axes[0].set_ylabel("load / daily mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=12,
        handletextpad=0.3,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    aml_style.save(fig, out_dir, "modes")
    return {
        "share_pct": [pct(s) for s in share[:6]],
        "cum_pct": [pct(s) for s in np.cumsum(share)[:6]],
        "corr_z1_night_day": round(corr, 2),
        "n_train": int(len(tr)),
        "mode2_peak_hour": int(np.argmax(V[:, 1])),
        "mean_shape_min": round(float(mu.min()), 2),
        "mean_shape_max": round(float(mu.max()), 2),
        "mean_shape_peak_hour": int(np.argmax(mu)),
        "mode1_plus2sd_min": round(float(plus1.min()), 2),
        "mode1_plus2sd_max": round(float(plus1.max()), 2),
        "mode1_plus2sd_peak_hour": int(np.argmax(plus1)),
    }


def bracket(
    pf: Portfolio, splits: list[tuple[np.ndarray, np.ndarray]], out_dir: Path
) -> dict[str, object]:
    """Shape error of held-out sites' 2017 days: modes and typical days, bracketed.

    Above: the mean shape (r = 0). Below (rough): the year-to-year change, the same building's 2016
    shape against its 2017 shape. The modes' error is a reconstruction (the 2017 shape is projected
    on the modes), the year-to-year change a forecast, so the comparison says when extra modes
    start to describe this year's details; it is not an error floor in the strict sense. Each
    split's error is one RMS over its held-out buildings; the slide quotes the mean of the 30, and
    how many of the 30 splits (paired) fall below the year-to-year change.

    Args:
        pf: The eligible buildings.
        splits: The site-held-out splits (``site_splits``).
        out_dir: The figure directory.

    Returns:
        Mean and spread over the splits of the error with r = 0-8 modes, the year-to-year
        change, the error of 4 and 16 typical days, and the paired split counts.
    """
    S16, S17 = shape(pf.P16), shape(pf.P17)
    rs = list(range(9))
    err: dict[int, list[float]] = {r: [] for r in rs}
    floor: list[float] = []
    km: dict[int, list[float]] = {4: [], 16: []}
    for tr, te in splits:
        mu, V, _, _ = pca_svd(S16[tr], 24)
        for r in rs:
            Zt = (S17[te] - mu) @ V[:, :r]
            err[r].append(rmse(S17[te], mu + Zt @ V[:, :r].T))
        floor.append(rmse(S17[te], S16[te]))
        for k, errs in km.items():
            C = (
                KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=N_INIT_BRACKET,
                    algorithm="lloyd",
                    random_state=SEED,
                )
                .fit(S16[tr])
                .cluster_centers_
            )
            d2 = ((S17[te][:, None, :] - C[None]) ** 2).sum(-1).min(axis=1)
            errs.append(float(np.sqrt(d2.mean() / 24)))
    mean = [pct(np.mean(err[r])) for r in rs]
    sd = [pct(np.std(err[r])) for r in rs]
    fl = pct(np.mean(floor))
    first_below = next(r for r in rs if np.mean(err[r]) <= np.mean(floor))
    below = {r: int((np.array(err[r]) < np.array(floor)).sum()) for r in rs}  # paired, per split
    bracket_figure(
        rs, mean, sd, floor=fl, km={k: pct(np.mean(v)) for k, v in km.items()}, out_dir=out_dir
    )
    return {
        "splits": len(splits),
        "test_share_of_sites": TEST_SIZE,
        "r": rs,
        "rmse_pct": mean,
        "rmse_sd_pct": sd,
        "floor_pct": fl,
        "floor_sd_pct": pct(np.std(floor)),
        # One RMS over all 1,159 buildings instead of the mean of 30 per-split RMS values.
        "floor_pooled_all_pct": pct(rmse(S17, S16)),
        "kmeans_pct": {str(k): pct(np.mean(v)) for k, v in km.items()},
        "kmeans_n_init": N_INIT_BRACKET,
        "first_r_at_floor": first_below,
        "r4_below_floor_splits": below[4],
        "r3_below_floor_splits": below[3],
    }


def bracket_figure(
    rs: list[int],
    mean: list[float],
    sd: list[float],
    *,
    floor: float,
    km: dict[int, float],
    out_dir: Path,
) -> None:
    """Held-out error against the modes kept, with the typical days, the baseline, the floor.

    Args:
        rs: Numbers of modes kept.
        mean: Mean error (% of the daily mean) for each r.
        sd: Its standard deviation over the splits.
        floor: The year-to-year change (%).
        km: Error of k typical days (%), by k.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.errorbar(
        rs, mean, yerr=sd, color=aml_style.PALETTE[0], marker="o", capsize=3, label="r modes"
    )
    for k, v in km.items():
        ax.axhline(
            v, color=aml_style.PALETTE[2 if k == 4 else 3], ls="-." if k == 4 else ":", lw=1.6
        )
        ax.text(
            8.2,
            v + 0.4,
            f"{k} typical days",
            ha="right",
            va="bottom",
            fontsize=12,
            color=aml_style.INK,
        )
    ax.axhline(mean[0], color=aml_style.MUTED, ls=":", lw=1.2)
    ax.axhline(floor, color=aml_style.MUTED, ls="--", lw=1.5)
    ax.text(
        8.2,
        floor + 0.4,
        "year-to-year\nchange (rough)",  # two lines: clear of the error bars at r = 3, 4
        ha="right",
        va="bottom",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.text(
        8.2,
        mean[0] + 0.4,
        "mean shape (r = 0)",
        ha="right",
        va="bottom",
        fontsize=12,
        color=aml_style.MUTED,
    )
    ax.set_xlabel("modes kept, r")
    ax.set_ylabel("2017 error (% of daily mean)")
    ax.set_xticks(rs)
    ax.set_ylim(0, 30)
    aml_style.save(fig, out_dir, "bracket")


# ---------------------------------------------------------------------------------------------
# Step 2: the trace of Q (diagram)


@dataclass(frozen=True)
class Trace:
    """Building Q through the encoder and decoder: the slide numbers and the raw values.

    Attributes:
        numbers: The rounded values for ``numbers.json``.
        z: Q's three scores, unrounded.
        err: Q's reconstruction error, a share of its daily mean.
        below: Share of the training offices with a lower mode-1 score than Q's.
        shape: Q's shape, shape ``(24,)``.
        shape_hat: Its reconstruction from R modes, shape ``(24,)``.
        mean_shape: The training mean shape, shape ``(24,)``.
    """

    numbers: dict[str, object]
    z: np.ndarray
    err: float
    below: float
    shape: np.ndarray
    shape_hat: np.ndarray
    mean_shape: np.ndarray


def trace(
    pf: Portfolio,
    att: dict[str, object],
    split: tuple[np.ndarray, np.ndarray],
    centers: np.ndarray,
    center_names: list[str],
) -> Trace:
    """Follow Q through the encoder and decoder (r = R) fit on the training shapes.

    Args:
        pf: The eligible buildings.
        att: ``pick_attempt``'s result (Q's row).
        split: Training and held-out row indices.
        centers: The typical days (k-means centers), shape ``(K, 24)``.
        center_names: Their names.

    Returns:
        The trace: rounded numbers for the slides, raw values for the diagram's macros.
    """
    tr, _ = split
    q = int(att["q"])
    S = shape(pf.P16)
    mu, V, _, _ = pca_svd(S[tr], R)
    V = orient(V)
    z = (S[q] - mu) @ V
    p_hat = mu + V @ z
    err = rmse(S[q], p_hat)
    dev = rmse(S[q], mu)
    captured = 1 - err**2 / dev**2
    Ztr = (S[tr] - mu) @ V
    offices = pf.use[tr] == "Office"
    below = float((Ztr[offices, 0] < z[0]).mean())
    nearest_center = int(((centers - S[q]) ** 2).sum(axis=1).argmin())
    numbers = {
        "kw_mean": round(float(pf.P16[q].mean()), 0),
        "z": [round(float(v), 2) for v in z],
        "rmse_pct": pct(err),
        "captured_pct": pct(captured, 0),
        "z1_share_of_training_offices_below_pct": pct(below, 0),
        "n_train_offices": int(offices.sum()),
        "n_train": int(len(tr)),
        "n_train_sites": int(len(set(pf.site[tr]))),
        "nearest_typical_day": center_names[nearest_center],
    }
    return Trace(numbers, z, float(err), below, S[q], p_hat, mu)


# ---------------------------------------------------------------------------------------------
# Step 5: break it


def size_mode(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path
) -> dict[str, object]:
    """PCA on raw kW, on hour-standardized kW (Lecture 2), and on shapes: what is mode 1?

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices (PCA on the training buildings).
        out_dir: The figure directory.

    Returns:
        For each encoding: mode 1's share of the variance, the correlation of its score with
        size, and its loadings' range.
    """
    tr, _ = split
    X = pf.P16[tr]
    size = X.mean(axis=1)
    out: dict[str, object] = {}
    loadings: dict[str, np.ndarray] = {}
    raw_score = np.empty(0)
    for name, label, M in (
        ("raw kW", "raw kW", X),
        ("each hour standardized", "standardized", StandardScaler().fit_transform(X)),
        ("shape (÷ daily mean)", "shape", shape(X)),
    ):
        mu, V, lam, _ = pca_svd(M, 24)
        v1 = orient(V, SHAPE_SIGN_RULES if name.startswith("shape") else KW_SIGN_RULES)[:, 0]
        z1 = (M - mu) @ v1
        if name == "raw kW":
            raw_score = z1
        loadings[label] = v1
        out[name] = {
            "mode1_pct": pct(lam[0] / lam.sum()),
            "corr_score_size": round(float(np.corrcoef(z1, size)[0, 1]), 5),
            "loading_min": round(float(v1.min()), 3),
            "loading_max": round(float(v1.max()), 3),
        }
    size_mode_figure(loadings, size, raw_score, out_dir)
    return out


def size_mode_figure(
    loadings: dict[str, np.ndarray], size: np.ndarray, raw_score: np.ndarray, out_dir: Path
) -> None:
    """Mode-1 loadings of the three encodings, and the raw-kW mode-1 score against size.

    Args:
        loadings: Mode-1 loadings by encoding, each shape ``(24,)``.
        size: Mean kW of each training building.
        raw_score: Mode-1 score of each training building on raw kW.
        out_dir: The figure directory.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=aml_style.SIDE)
    for j, (name, v1) in enumerate(loadings.items()):
        a1.plot(
            HOURS,
            v1,
            color=aml_style.PALETTE[j],
            ls=aml_style.LINESTYLES[j],
            marker=aml_style.MARKERS[j],
            markevery=6,
            ms=6,
            label=name,
        )
    a1.axhline(0, color=aml_style.MUTED, lw=0.8)
    a1.set_xticks([0, 12, 23], ["0", "12", "23"])
    a1.set_xlabel("hour")
    a1.set_ylabel("mode-1 loading")
    a1.set_ylim(-0.3, 0.62)
    a1.set_title("mode 1", loc="left")
    a1.legend(loc="upper left", fontsize=11, handletextpad=0.3, borderaxespad=0.1, handlelength=1.6)
    a2.scatter(
        size / 1000,
        raw_score / 1000,
        s=10,
        color=aml_style.PALETTE[0],
        alpha=0.6,
        edgecolors="none",
    )
    a2.set_xlabel("mean load (MW)")
    a2.set_ylabel("mode-1 score (MW)")
    a2.set_title("raw kW", loc="left")
    fig.tight_layout()
    aml_style.save(fig, out_dir, "size_mode")


def day_leakage(
    pf: Portfolio,
    split: tuple[np.ndarray, np.ndarray],
    hourly: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    out_dir: Path,
    n_queries: int = 2000,
) -> dict[str, object]:
    """Retrieve building-days by shape: random split of days against held-out sites (FAISS).

    Building-days: every 2016 weekday of an eligible building with all 24 hours metered and
    positive. Queries: ``n_queries`` random building-days of buildings at held-out sites.

    Args:
        pf: The eligible buildings.
        split: Training and held-out buildings.
        hourly: Hourly kWh of the eligible buildings, shape ``(N, days, 24)``.
        dates: The days of ``hourly``.
        out_dir: The figure directory.
        n_queries: Building-days used as queries.

    Returns:
        Shares of queries whose nearest day is the same building, site, or use, per database.
    """
    tr, te = split
    keep = np.asarray((dates.dayofweek < 5) & (dates.year == 2016))
    days = hourly[:, keep, :]
    ok = ~np.isnan(days).any(axis=2) & (np.nan_to_num(days, nan=0.0).min(axis=2) > 0)
    b_i, d_i = np.nonzero(ok)
    X = days[b_i, d_i].astype(np.float32)
    X /= X.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(SEED)
    q = rng.choice(np.flatnonzero(np.isin(b_i, te)), size=n_queries, replace=False)
    dbs = {
        "random split of days": np.setdiff1d(np.arange(len(X)), q),
        "sites held out": np.flatnonzero(np.isin(b_i, tr)),
    }
    res: dict[str, dict[str, float]] = {}
    for name, db in dbs.items():
        index = faiss.IndexFlatL2(24)
        index.add(np.ascontiguousarray(X[db]))
        _, nn = index.search(np.ascontiguousarray(X[q]), 1)
        nb, qb = b_i[db][nn[:, 0]], b_i[q]
        res[name] = {
            "same_building": pct((nb == qb).mean()),
            "same_site": pct((pf.site[nb] == pf.site[qb]).mean()),
            "same_use": pct((pf.use[nb] == pf.use[qb]).mean()),
            "database_days": int(len(db)),
        }
    share = pd.Series(pf.use[b_i[dbs["sites held out"]]]).value_counts(normalize=True)
    chance = float(np.mean([share.get(u, 0.0) for u in pf.use[b_i[q]]]))
    leakage_figure(res, chance, out_dir)
    return {
        **res,
        "chance_same_use": pct(chance),
        "queries": n_queries,
        "building_days": int(len(X)),
    }


def leakage_figure(res: dict[str, dict[str, float]], chance: float, out_dir: Path) -> None:
    """Bars: nearest day is the same building, site, or use, for the two databases.

    Args:
        res: Shares (%) by database and category (``day_leakage``).
        chance: Chance that a query's nearest day has the same use, as a share.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    cats = ["same_building", "same_site", "same_use"]
    xs = np.arange(len(cats))
    for j, (name, hatch) in enumerate((("random split of days", ""), ("sites held out", "//"))):
        vals = [res[name][c] for c in cats]
        bars = ax.bar(
            xs + (j - 0.5) * 0.38,
            vals,
            width=0.36,
            color=aml_style.PALETTE[j],
            hatch=hatch,
            edgecolor="white",
            label=name,
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=12
            )
    ax.axhline(100 * chance, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(0.42, 100 * chance + 1.5, "chance", ha="center", fontsize=12, color=aml_style.MUTED)
    ax.set_xticks(xs, ["same building", "same site", "same use"])
    ax.set_ylabel("nearest day is ... (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=12)
    aml_style.save(fig, out_dir, "day_leakage")


def name_centers(C: np.ndarray) -> list[str]:
    """Name typical days by their shape: evening peak, always on, strong day, moderate day.

    The center with the most load at 18-22 h is the evening peak; the rest are named by their
    night/day ratio, lowest first: strong day, moderate day, always on.

    Args:
        C: The centers, shape ``(K, 24)``.

    Returns:
        One name per center.
    """
    names = [""] * len(C)
    rho = C[:, 0:4].mean(axis=1) / C[:, 12:16].mean(axis=1)
    evening = int(np.argmax(C[:, 18:22].mean(axis=1)))
    names[evening] = "evening peak"
    rest = [k for k in range(len(C)) if k != evening]
    order = sorted(rest, key=lambda k: rho[k])
    labels = ["strong day", "moderate day", "always on"] + [f"day {i}" for i in range(9)]
    for k, lab in zip(order, labels[: len(order)], strict=True):
        names[k] = lab
    return names


def typical_days(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path
) -> tuple[dict[str, object], np.ndarray, list[str]]:
    """k-means with K typical days on the training shapes (best of N_INIT_TYPICAL starts).

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices.
        out_dir: The figure directory.

    Returns:
        The counts, the smallest cluster's members and peaks, J per building, and each center's
        lowest and highest value (the figure's readings); the centers; their names.
    """
    tr, _ = split
    S = shape(pf.P16[tr])
    km = KMeans(
        n_clusters=K, init="k-means++", n_init=N_INIT_TYPICAL, algorithm="lloyd", random_state=SEED
    ).fit(S)
    C = km.cluster_centers_
    names = name_centers(C)
    counts = np.bincount(km.labels_, minlength=K)
    small = int(np.argmin(counts))
    members = tr[km.labels_ == small]
    order = sorted(range(K), key=lambda k: -counts[k])

    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for j, k in enumerate(order):
        ax.plot(
            HOURS,
            C[k],
            color=aml_style.PALETTE[j],
            ls=aml_style.LINESTYLES[j],
            marker=aml_style.MARKERS[j],
            markevery=4,
            ms=7,
            label=f"{names[k]} ({counts[k]})",
        )
    ax.set_xticks([0, 6, 12, 18, 23], ["0", "6", "12", "18", "23"])
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("load / daily mean")
    ax.legend(loc="upper left", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "typical_days")
    out = {
        "K": K,
        "n_init": N_INIT_TYPICAL,
        "counts": {names[k]: int(counts[k]) for k in order},
        "small_cluster": names[small],
        "small_members": [str(pf.name[i]) for i in members],
        "small_uses": sorted({str(pf.use[i]) for i in members}),
        "small_peak_hour": int(np.argmax(C[small])),
        "small_peak_x_mean": round(float(C[small].max()), 1),
        "member_peak_x_mean": [round(float(shape(pf.P16[[i]])[0].max()), 1) for i in members],
        "J_per_building": round(float(km.inertia_ / len(S)), 3),
        "center_min": {names[k]: round(float(C[k].min()), 2) for k in order},
        "center_max": {names[k]: round(float(C[k].max()), 2) for k in order},
        "center_peak_hour": {names[k]: int(np.argmax(C[k])) for k in order},
    }
    return out, C, names


def season(
    pf: Portfolio, hourly: np.ndarray, dates: pd.DatetimeIndex, centers: np.ndarray
) -> dict[str, object]:
    """Summer (Jun-Aug) against winter (Jan, Feb, Dec) 2016 typical weekdays, same buildings.

    The full-year 2016-to-2017 change compares 12-month windows, so it is not the yardstick of a
    3-month one. The like-for-like yardstick is the same season a year apart: summer 2016 against
    summer 2017, winter 2016 against winter 2017, on the buildings with enough weekdays in all
    four windows.

    Args:
        pf: The eligible buildings.
        hourly: Their hourly kWh, shape ``(N, days, 24)``.
        dates: The days of ``hourly``.
        centers: The typical days (k-means centers).

    Returns:
        How many buildings change typical day, the summer-winter and full-year 2016-2017 shape
        changes, and the same-season changes a year apart.
    """
    weekday, year = np.asarray(dates.dayofweek < 5), np.asarray(dates.year)
    month = np.asarray(dates.month)
    windows = {
        (y, name): typical_weekday(hourly, weekday & (year == y) & np.isin(month, months))
        for y in (2016, 2017)
        for name, months in (("summer", SUMMER), ("winter", WINTER))
    }

    def enough(y: int, name: str) -> np.ndarray:
        P, n = windows[(y, name)]
        return (n >= MIN_SEASON_DAYS) & (P.min(axis=1) > 0)

    def shp(y: int, name: str, rows: np.ndarray) -> np.ndarray:
        return shape(windows[(y, name)][0][rows])

    ok = enough(2016, "summer") & enough(2016, "winter")
    Ss, Sw = shp(2016, "summer", ok), shp(2016, "winter", ok)
    ls = ((Ss[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
    lw = ((Sw[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
    S16, S17 = shape(pf.P16[ok]), shape(pf.P17[ok])
    both = ok & enough(2017, "summer") & enough(2017, "winter")
    return {
        "buildings": int(ok.sum()),
        "changed_typical_day_pct": pct((ls != lw).mean(), 0),
        "summer_vs_winter_rms_pct": pct(rmse(Ss, Sw)),
        "year_to_year_rms_pct": pct(rmse(S16, S17)),
        "same_season_a_year_apart": {
            "buildings": int(both.sum()),
            "summer_vs_winter_2016_rms_pct": pct(
                rmse(shp(2016, "summer", both), shp(2016, "winter", both))
            ),
            "summer_2016_vs_2017_rms_pct": pct(
                rmse(shp(2016, "summer", both), shp(2017, "summer", both))
            ),
            "winter_2016_vs_2017_rms_pct": pct(
                rmse(shp(2016, "winter", both), shp(2017, "winter", both))
            ),
            "full_year_2016_vs_2017_rms_pct": pct(rmse(shape(pf.P16[both]), shape(pf.P17[both]))),
        },
    }


# ---------------------------------------------------------------------------------------------
# Step 6: decide


def visit_bootstrap(ex17: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """Office bootstrap of the share of 2017 excess that ``VISITS`` visits find, per ranking.

    The visit numbers come from one campus of 67 offices. Each of ``BOOT_DRAWS`` draws resamples
    the offices with replacement, ranks the draw by each score, and takes the share of the draw's
    2017 excess in its top ``VISITS``. The rankings are compared within each draw (paired).

    Args:
        ex17: 2017 excess night load of each office, clipped at 0, shape ``(n,)``.
        scores: Ranking score of each office by name, each shape ``(n,)``; must include
            "size × shape", "shape only", and "size only".

    Returns:
        The 95% interval (2.5th-97.5th percentile, %) of each ranking's share, and the median
        and interval of size × shape's lead over each other ranking (points), with the share of
        draws in which it does not lead shape only.

    Raises:
        ValueError: If a draw has no 2017 excess at all (its shares would be undefined).
    """
    rng = np.random.default_rng((SEED, 4))
    n = len(ex17)
    idx = rng.integers(0, n, size=(BOOT_DRAWS, n))
    e17 = ex17[idx]  # (draws, n)
    total = e17.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("a bootstrap draw has no 2017 excess")
    found = {}
    for name, score in scores.items():
        top = np.argsort(-score[idx], axis=1, kind="stable")[:, :VISITS]
        found[name] = np.take_along_axis(e17, top, axis=1).sum(axis=1) / total
        assert found[name].shape == (BOOT_DRAWS,)

    def interval(v: np.ndarray) -> list[float]:
        return [pct(np.percentile(v, 2.5), 0), pct(np.percentile(v, 97.5), 0)]

    best = found["size × shape"]
    return {
        "draws": BOOT_DRAWS,
        "visits": VISITS,
        "interval_pct": {name: interval(v) for name, v in found.items()},
        "lead_over_shape_only_pts": {
            "median": pct(np.median(best - found["shape only"]), 0),
            "interval": interval(best - found["shape only"]),
            "draws_not_ahead_pct": pct(np.mean(best <= found["shape only"]), 0),
        },
        "lead_over_size_only_pts": {
            "median": pct(np.median(best - found["size only"]), 0),
            "interval": interval(best - found["size only"]),
            "draws_not_ahead_pct": pct(np.mean(best <= found["size only"]), 0),
        },
    }


def decision(pf: Portfolio, att: dict[str, object], out_dir: Path) -> dict[str, object]:
    """Rank the offices on Q's campus by 2016 data; score the ranking on 2017 excess night load.

    Args:
        pf: The eligible buildings.
        att: ``pick_attempt``'s result (Q's row and the office yardstick).
        out_dir: The figure directory.

    Returns:
        The share of 2017 excess covered at 5 and ``VISITS`` visits by each ranking and by
        random order, its office bootstrap, how many of the top 10 repeat, the largest moves,
        and the vote's cards.
    """
    q = int(att["q"])
    ref = float(att["office_ref_night_day"])  # the rounded yardstick, as in pick_attempt
    o = np.flatnonzero((pf.site == pf.site[q]) & (pf.use == "Office"))
    b16, b17 = blocks(pf.P16[o]), blocks(pf.P17[o])
    ex16 = b16[:, NIGHT] - ref * b16[:, DAY]
    ex17_raw = b17[:, NIGHT] - ref * b17[:, DAY]
    ex17 = np.clip(ex17_raw, 0, None)
    scores = {
        "size × shape": ex16,
        "shape only": night_day(pf.P16[o]),
        "size only": pf.P16[o].mean(axis=1),
    }
    curves = {
        name: np.concatenate([[0.0], np.cumsum(ex17[np.argsort(-s, kind="stable")]) / ex17.sum()])
        for name, s in scores.items()
    }
    n = len(o)
    decision_figure(curves, n, out_dir)

    top16 = set(np.argsort(-ex16)[:VISITS])
    top17 = set(np.argsort(-ex17_raw)[:VISITS])
    movers = {
        str(pf.name[o[i]]): [round(float(ex16[i]), 0), round(float(ex17_raw[i]), 0)]
        for i in np.argsort(-np.abs(ex17_raw - ex16))[:2]
    }
    return {
        "site": str(pf.site[q]),
        "offices": int(n),
        "covered_at_5_pct": {k: pct(c[5], 0) for k, c in curves.items()},
        "covered_at_10_pct": {k: pct(c[VISITS], 0) for k, c in curves.items()},
        "random_at_10_pct": pct(VISITS / n, 0),
        "bootstrap_at_10": visit_bootstrap(ex17, scores),
        "top10_2016_in_top10_2017": int(len(top16 & top17)),
        "largest_changes_kw_2016_to_2017": movers,
        "vote": vote_cards(pf, q, ref),
    }


def decision_figure(curves: dict[str, np.ndarray], n: int, out_dir: Path) -> None:
    """Share of 2017 excess covered against visits, for each ranking and random order.

    Args:
        curves: Cumulative share covered after 0, 1, ..., n visits, by ranking.
        n: Number of offices.
        out_dir: The figure directory.
    """
    visits = np.arange(n + 1)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for j, (name, c) in enumerate(curves.items()):
        ax.plot(
            visits[:16],
            100 * c[:16],
            color=aml_style.PALETTE[j],
            ls=aml_style.LINESTYLES[j],
            marker=aml_style.MARKERS[j],
            markevery=5,
            ms=7,
            label=name,
        )
    ax.plot(
        visits[:16], 100 * visits[:16] / n, color=aml_style.MUTED, ls=":", lw=1.5, label="random"
    )
    ax.set_xlabel("visits, ranked on 2016 data")
    ax.set_ylabel("2017 excess covered (%)")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left", fontsize=12, handletextpad=0.3)
    aml_style.save(fig, out_dir, "decision")


def vote_cards(pf: Portfolio, q: int, ref: float) -> list[dict[str, object]]:
    """The vote: three buildings on Q's campus other than Q.

    The screen's next pick if it ignores use (most night load above the office yardstick), the
    office with the most, and the residence with the most. Each card also carries the same
    building's 2017 excess, for the notes' "what would change the answer".

    Args:
        pf: The eligible buildings.
        q: Q's row.
        ref: The office yardstick (night/day ratio).

    Returns:
        Cards A-C: name, use, 2016 kW, night/day ratio (%), 2016 and 2017 excess (kW), W/m².

    Raises:
        ValueError: If two of the three picks are the same building.
    """
    rho16 = night_day(pf.P16)
    ex_all = blocks(pf.P16)[:, NIGHT] - ref * blocks(pf.P16)[:, DAY]
    ex_2017 = blocks(pf.P17)[:, NIGHT] - ref * blocks(pf.P17)[:, DAY]
    campus = np.flatnonzero((pf.site == pf.site[q]) & (np.arange(len(pf.site)) != q))
    picks = []
    for allowed in (None, "Office", "Lodging/residential"):
        pool = campus if allowed is None else campus[pf.use[campus] == allowed]
        picks.append(int(pool[np.argmax(ex_all[pool])]))
    if len(set(picks)) != 3:
        raise ValueError(f"the vote's picks repeat a building: {picks}")

    def card(i: int) -> dict[str, object]:
        kw = float(pf.P16[i].mean())
        return {
            "name": str(pf.name[i]),
            "use": str(pf.use[i]),
            "kw": round(kw, 0),
            "night_day_pct": pct(rho16[i], 0),
            "excess_kw": round(float(ex_all[i]), 0),
            "excess_kw_2017": round(float(ex_2017[i]), 0),
            "w_per_m2": round(1000 * kw / float(pf.sqm[i]), 0),
        }

    return [card(i) for i in picks]


def reidentify(pf: Portfolio) -> dict[str, object]:
    """Does a building's 2017 typical weekday retrieve its own 2016 weekday among all buildings?

    Args:
        pf: The eligible buildings.

    Returns:
        For shapes and raw kW, the share of buildings whose own 2016 weekday comes first and
        within the top 5; and the number of buildings.
    """
    out = {}
    for name, (a, b) in {
        "shape": (shape(pf.P17), shape(pf.P16)),
        "raw kW": (pf.P17, pf.P16),
    }.items():
        index = faiss.IndexFlatL2(24)
        index.add(np.ascontiguousarray(b.astype(np.float32)))
        _, nn = index.search(np.ascontiguousarray(a.astype(np.float32)), 5)
        own = np.arange(len(a))[:, None]
        out[name] = {
            "top1_pct": pct((nn[:, 0] == own[:, 0]).mean(), 0),
            "top5_pct": pct((nn == own).any(axis=1).mean(), 0),
        }
    out["buildings"] = int(len(pf.P16))
    return out


def retrieval_check(pf: Portfolio, split: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    """Brute-force NumPy retrieval against FAISS IndexFlatL2 for every held-out building.

    NumPy ranks by distances in float64; FAISS computes ||q||^2 - 2 q.x + ||x||^2 in float32,
    which is exact only to a few float32 epsilons times the squared norms. Two neighbors closer
    than that are a tie, and FAISS may return them in either order. A query agrees up to ties
    when, rank by rank, FAISS's neighbors are as close as NumPy's within that rounding.

    Args:
        pf: The eligible buildings.
        split: Training (database) and held-out (query) row indices.

    Returns:
        The share of queries with the same top 5, with and without ties, and how many queries
        FAISS reordered within ties.

    Raises:
        ValueError: If a FAISS distance is off by more than float32 rounding, or a query's
            neighbors differ by more than a tie.
    """
    tr, te = split
    S = shape(pf.P16)
    sq = (S**2).sum(axis=1)
    d2 = sq[te][:, None] - 2 * S[te] @ S[tr].T + sq[tr][None, :]
    assert d2.shape == (len(te), len(tr))
    nn_np = np.argsort(d2, axis=1, kind="stable")[:, :5]
    index = faiss.IndexFlatL2(24)
    index.add(np.ascontiguousarray(S[tr].astype(np.float32)))
    d_faiss, nn_fa = index.search(np.ascontiguousarray(S[te].astype(np.float32)), 5)
    d_np, d_fa = np.take_along_axis(d2, nn_np, axis=1), np.take_along_axis(d2, nn_fa, axis=1)
    eps32 = float(np.finfo(np.float32).eps)
    bound_fa = 4 * eps32 * (sq[te][:, None] + sq[tr][nn_fa])  # FAISS's rounding, per neighbor
    bound_np = 4 * eps32 * (sq[te][:, None] + sq[tr][nn_np])
    if not np.all(np.abs(d_faiss - d_fa) <= bound_fa):
        raise ValueError("FAISS distance beyond float32 rounding")
    same = (nn_np == nn_fa).all(axis=1)
    up_to_ties = (np.abs(d_fa - d_np) <= bound_fa + bound_np).all(axis=1)
    if not up_to_ties.all():
        raise ValueError(f"{int((~up_to_ties).sum())} queries differ beyond ties")
    return {
        "queries": int(len(te)),
        "database": int(len(tr)),
        "same_top5_pct": pct(same.mean()),
        "same_top5_up_to_ties_pct": pct(up_to_ties.mean()),
        "queries_reordered_within_ties": int((~same).sum()),
    }


# ---------------------------------------------------------------------------------------------
# numbers.tex for the diagram


def tex_macros(tr_: Trace, att: dict[str, object], pf: Portfolio) -> dict[str, str]:
    """The trace of Q as LaTeX macros, with plot coordinates for the two mini charts.

    Args:
        tr_: ``trace``'s result, with its raw values.
        att: ``pick_attempt``'s result (Q's row and the shape neighbors' letters).
        pf: The eligible buildings (Q's raw profile).

    Returns:
        Macro name to LaTeX text, every number formatted once from its raw value.
    """
    z = tr_.z
    q = int(att["q"])
    b = blocks(pf.P16[[q]])[0]

    def coords(v: np.ndarray) -> str:
        # Mini chart in cm: 3.2 cm for 0-23 h; 3 cm per unit of load / daily mean, from 0.7.
        # Plain hyphens: TikZ reads these as numbers.
        return " ".join(f"({3.2 * h / 23:.3f},{3.0 * (x - 0.7):.3f})" for h, x in enumerate(v))

    return {
        "QKw": tex_num(pf.P16[q].mean(), ",.0f"),
        "QNight": tex_num(b[NIGHT], ",.0f"),
        "QDay": tex_num(b[DAY], ",.0f"),
        "R": str(R),
        "ZOne": tex_num(z[0], ".2f"),
        "ZTwo": tex_num(z[1], ".2f"),
        "ZThree": tex_num(z[2], ".2f"),
        "ZOneFlatter": tex_num(100 * (1 - tr_.below), ".0f"),
        "NTrainOffices": str(tr_.numbers["n_train_offices"]),
        "Err": tex_num(100 * tr_.err, ".1f"),
        "NTrain": tex_num(tr_.numbers["n_train"], ","),
        "NTrainSites": str(tr_.numbers["n_train_sites"]),
        "TypicalDay": str(tr_.numbers["nearest_typical_day"]),
        "ShapeNbrs": " and ".join(att["shape_letters"]),
        "QShape": coords(tr_.shape),
        "QShapeHat": coords(tr_.shape_hat),
        "MeanShape": coords(tr_.mean_shape),
    }


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

PLANE_HOURS: Final[tuple[int, int, int]] = (3, 9, 15)  # the plane keeps 94%, sticks show
PLANE_SHOWN: Final[int] = 400  # training buildings drawn in it
BUILD_K: Final[int] = 4  # centers in the k-means build (as the lecture's typical days)
BUILD_SEED: Final[int] = 4  # starting centers of the build: four random training buildings


def title_art_figure(pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path) -> None:
    """The title art: three clusters of glowing points over stepped forms, under a dark sun.

    The horizon is the mean shape plus two standard deviations of mode 1, the working day.

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices.
        out_dir: The figure directory.
    """
    tr, _ = split
    mu, V, lam, _ = pca_svd(shape(pf.P16[tr]), 1)
    day = mu + 2 * np.sqrt(lam[0]) * orient(V)[:, 0]
    c = title_art.Canvas(seed=4)
    c.ground(("#0e1612", "#14121f", "#1b0c14", "#07060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.02, 0.06, 0.62, 0.70, 4), soft=4), "#4c1d15", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.40, 0.06, 0.58, 0.95, 5), soft=4), "#7a2a18", 0.8)
    rim = c.disc((0.26, 1.16), 0.12, soft=4) - c.disc((0.26, 1.16), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.26, 1.16), 0.10, soft=3), "#1d0a10", 0.92)
    rng = np.random.default_rng(4)
    for (cx, cy), col in (((0.30, 0.78), "#f0a03c"), ((0.62, 1.02), "#9fb7c9"),
                          ((0.78, 0.70), "#e9c27a")):  # fmt: skip
        pts = rng.normal((cx, cy), 0.045, (22, 2))
        dots = sum(c.disc(tuple(q), 0.0065, soft=1.5) for q in pts)
        c.glow(np.clip(dots, 0, 1), col, 0.55, spread=7)
    xs = 0.06 + 0.88 * HOURS / 23
    ys = 0.13 + 0.30 * (day - day.min()) / np.ptp(day)
    c.glow(c.lines([list(zip(xs, ys, strict=True))], 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: two stepped Andean forms in umber and rust; above "
        "them three clusters of glowing points in amber, pale blue, and gold; a dark sun with a "
        "burning rim at the upper left, and a golden line, the working day of the lecture's "
        "first mode, low at night and high by day, across the foot of the forms.",
    )


def carpet_figure(
    pf: Portfolio, hourly: np.ndarray, dates: pd.DatetimeIndex, q: int, out_dir: Path
) -> dict[str, int]:
    """Building Q's 2016, every hour: one row per day, colour by kW. Nights stay bright.

    The returned night and afternoon means are over all 2016 days, weekends included (the
    attempt's table uses typical weekdays).

    Args:
        pf: The eligible buildings.
        hourly: Their hourly kWh, shape ``(N, days, 24)``.
        dates: The days of ``hourly``.
        q: Q's row.
        out_dir: The figure directory.

    Returns:
        Q's mean kW at night (0-4 h) and in the afternoon (12-16 h), and the colour bar's range
        (the lowest and highest hour of 2016, which ``imshow`` scales to by default).
    """
    keep = np.asarray(dates.year == 2016)
    load = hourly[q, keep, :].astype(np.float64)  # (days, 24)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    im = ax.imshow(load, aspect="auto", cmap="cividis", interpolation="nearest",
                   extent=(-0.5, 23.5, keep.sum() - 0.5, -0.5))  # fmt: skip
    fig.colorbar(im, ax=ax, label="kW", shrink=0.9)
    ax.set(xlabel="hour of the day", ylabel="day of 2016", xticks=[0, 6, 12, 18, 23])
    month_starts = [int((dates[keep] < f"2016-{m:02d}-01").sum()) for m in (1, 4, 7, 10)]
    ax.set_yticks(month_starts, ["Jan", "Apr", "Jul", "Oct"])
    ax.set_title(f"building Q, {pf.use[q].lower()}: every hour of 2016", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "carpet")
    night, day = np.nanmean(load[:, 0:4]), np.nanmean(load[:, 12:16])
    lo, hi = im.get_clim()
    if (lo, hi) != (np.nanmin(load), np.nanmax(load)):
        raise ValueError(f"the colour bar is not the data's range: {lo, hi}")
    return {
        "night_kw": round(float(night)),
        "afternoon_kw": round(float(day)),
        "colour_min_kw": round(float(lo)),
        "colour_max_kw": round(float(hi)),
    }


def plane_figure(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path
) -> dict[str, object]:
    """PCA in 3D: three hours of each training building's shape, the best plane, and the sticks.

    The plane through the mean, spanned by the two leading directions of these three numbers,
    keeps the most spread; each stick is a building's error, the part the plane drops. The
    shares are over all training buildings; the figure draws ``PLANE_SHOWN`` of them.

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices.
        out_dir: The figure directory.

    Returns:
        The hours, how many buildings the shares are over and how many are drawn, and the
        shares of variance that mode 1 and the plane keep.
    """
    tr, _ = split
    X = shape(pf.P16[tr])[:, list(PLANE_HOURS)]
    mu, V, lam, Z = pca_svd(X, 3)
    lo, hi = np.quantile(X, [0.02, 0.98], axis=0)
    core = np.flatnonzero(((lo <= X) & (hi >= X)).all(axis=1))  # the view leaves out outliers
    rng = np.random.default_rng((SEED, 5))
    show = rng.choice(core, PLANE_SHOWN, replace=False)
    fit = mu + Z[:, :2] @ V[:, :2].T  # projection onto the plane
    fig = plt.figure(figsize=aml_style.SIDE)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    a, b = np.meshgrid(np.linspace(*np.quantile(Z[core, 0], [0.0, 1.0]), 2),
                       np.linspace(*np.quantile(Z[core, 1], [0.0, 1.0]), 2))  # fmt: skip
    corners = mu[:, None, None] + V[:, 0, None, None] * a + V[:, 1, None, None] * b
    ax.plot_surface(*corners, color=aml_style.PALETTE[0], alpha=0.18, lw=0, zorder=1)
    for i in show:
        ax.plot(*np.column_stack([X[i], fit[i]]), color=aml_style.PALETTE[1], lw=0.7, ls="-",
                alpha=0.7, zorder=2)  # fmt: skip
    ax.scatter(*X[show].T, s=7, color=aml_style.INK, alpha=0.8, lw=0, zorder=3)
    t = np.quantile(Z[core, 0], [0.0, 1.0])
    ax.plot(*(mu[:, None] + V[:, [0]] * t), color=aml_style.PALETTE[0], lw=3, ls="-", zorder=4)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    limits = zip(lo, hi, strict=True)
    for axis, h, lim in zip((ax.xaxis, ax.yaxis, ax.zaxis), PLANE_HOURS, limits, strict=True):
        axis.set_ticks(np.round(lim, 1))
        axis.set_label_text(f"{h} h", fontsize=12)
        axis.labelpad = 0
    ax.zaxis.set_label_text("")  # matplotlib clips it at this angle; drawn by hand instead
    # Left of the axis line (at -0.02 the line ran through the "h").
    ax.text2D(-0.09, 0.47, f"{PLANE_HOURS[2]} h", transform=ax.transAxes, fontsize=12)
    share = lam / lam.sum()
    ax.set_title(f"mode 1 keeps {share[0]:.0%}; the plane, {share[:2].sum():.0%}", loc="left",
                 fontsize=13)  # fmt: skip
    aml_style.style3d(ax, elev=20, azim=-150)  # chosen from six: the sticks meet the plane
    aml_style.save(fig, out_dir, "plane")
    return {
        "hours": list(PLANE_HOURS),
        "buildings": int(len(tr)),
        "shown": PLANE_SHOWN,
        "mode1_share_pct": pct(share[0], 0),
        "plane_share_pct": pct(share[:2].sum(), 0),
    }


def kmeans_build_figure(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], out_dir: Path
) -> dict[str, object]:
    """Lloyd's algorithm drawn in 2D: the training shapes' scores on modes 1 and 2.

    Four starting centers are four random training buildings; panels show the start, the first
    assign-and-update, and convergence. (The lecture's typical days run the same steps in 24-D.)

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices.
        out_dir: The figure directory.

    Returns:
        The number of passes, and J at the start and at convergence.
    """
    tr, _ = split
    S = shape(pf.P16[tr])
    _, V, _, _ = pca_svd(S, 2)
    V = orient(V)
    X = (S - S.mean(axis=0)) @ V  # (N, 2)
    rng = np.random.default_rng(BUILD_SEED)
    C0 = X[rng.choice(len(X), BUILD_K, replace=False)]
    C1, z1, _ = kmeans(X, C0, max_iter=1)
    C, z, hist = kmeans(X, C0)
    fig, axes = plt.subplots(1, 3, figsize=aml_style.FULL, sharex=True, sharey=True)
    # One pass assigns to C0 before it updates, so z1 colours both the start and the first step.
    frames = (("start", z1, C0, None), ("assign, then update", z1, C1, C0),
              (f"converged: {len(hist)} passes", z, C, None))  # fmt: skip
    for ax, (title, lab, cen, old) in zip(axes, frames, strict=True):
        for k in range(BUILD_K):
            m = lab == k
            ax.scatter(*X[m].T, s=6, color=aml_style.PALETTE[k], alpha=0.45, lw=0,
                       marker=aml_style.MARKERS[k])  # fmt: skip
        if old is not None:
            for a, b in zip(old, cen, strict=True):
                ax.annotate("", b, a, arrowprops={"arrowstyle": "->", "lw": 1.6,
                                                  "color": aml_style.INK})  # fmt: skip
        ax.scatter(*cen.T, s=150, c=aml_style.PALETTE[:BUILD_K], marker="X",
                   edgecolor=aml_style.INK, lw=1.2, zorder=5)  # fmt: skip
        ax.set_title(title, loc="left", fontsize=13)
        ax.set(xlabel="mode 1 score", xlim=(-2.6, 3.6), ylim=(-1.6, 2.6))  # outliers off-axis
    axes[0].set_ylabel("mode 2 score")
    aml_style.save(fig, out_dir, "kmeans_build")
    return {"passes": len(hist), "J_start": round(hist[0], 1), "J_end": round(hist[-1], 1)}


def rebuild_figure(
    pf: Portfolio, split: tuple[np.ndarray, np.ndarray], q: int, out_dir: Path
) -> dict[str, object]:
    """Building Q's 2016 shape rebuilt from the mean shape, adding one mode at a time.

    Args:
        pf: The eligible buildings.
        split: Training and held-out row indices (the modes are fit on the training shapes).
        q: Q's row.
        out_dir: The figure directory.

    Returns:
        Q's three scores and the error (% of its daily mean) with 0-3 modes.
    """
    tr, _ = split
    S = shape(pf.P16)
    mu, V, _, _ = pca_svd(S[tr], 3)
    V = orient(V)
    z = (S[q] - mu) @ V
    fig, axes = plt.subplots(1, 4, figsize=aml_style.FULL, sharey=True)
    err = []
    for r, ax in enumerate(axes):
        fit = mu + V[:, :r] @ z[:r]
        err.append(pct(rmse(fit, S[q])))
        ax.plot(HOURS, S[q], color=aml_style.MUTED, lw=2.2, ls="--")
        ax.plot(HOURS, fit, color=aml_style.PALETTE[0], lw=2.6, ls="-")
        sign = "+" if r and z[r - 1] >= 0 else "−"
        name = "mean shape" if r == 0 else f"{sign} {abs(z[r - 1]):.2f} × mode {r}"
        ax.set_title(f"{name}\nerror {err[-1]:.1f}%", loc="left", fontsize=12.5)
        ax.set(xticks=[0, 12, 23], xlabel="hour")
    axes[0].set_ylabel("load / daily mean")
    aml_style.save(fig, out_dir, "rebuild")
    return {"scores": [round(float(v), 2) for v in z], "error_pct": err}


def fingerprint_figure(reid: dict[str, object], out_dir: Path) -> None:
    """Re-identification: a building's 2017 day finds its own 2016 day, top 1 and top 5.

    Args:
        reid: ``reidentify``'s result.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    names = ("shape", "raw kW")
    kinds = (("top1_pct", None, "own day first"), ("top5_pct", "//", "own day in top 5"))
    for j, (key, hatch, label) in enumerate(kinds):
        vals = [reid[n][key] for n in names]  # type: ignore[index]
        bars = ax.bar(np.arange(2) + 0.38 * j, vals, width=0.36, color=aml_style.PALETTE[j],
                      hatch=hatch, edgecolor="white", label=label)  # fmt: skip
        ax.bar_label(bars, [f"{v:.0f}%" for v in vals], fontsize=12, padding=2)
    ax.set(xticks=np.arange(2) + 0.19, xticklabels=names, ylabel="buildings (%)", ylim=(0, 100))
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title(f"among {reid['buildings']:,} buildings", loc="left", fontsize=13)
    aml_style.save(fig, out_dir, "fingerprint")


def main() -> None:
    """Generate every figure and number for CEE Lecture 4."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    data = load_bdg2()
    kwh, names, dates, _ = data
    pf = build_portfolio(data)
    rows = pd.Index(names).get_indexer(pf.name)
    if (rows < 0).any():
        raise ValueError("an eligible building is missing from the hourly file")
    hourly = kwh[rows]  # eligible buildings, (N, days, 24)
    del kwh
    weekdays = weekday_counts(dates)
    splits = site_splits(pf.site)
    split0 = campus_split(pf)
    tr, _ = split0
    att = pick_attempt(pf, split0, weekdays[2016])
    attempt_figure(pf, att, out_dir)
    days, centers, center_names = typical_days(pf, split0, out_dir)
    tr_ = trace(pf, att, split0, centers, center_names)
    numbers: dict[str, object] = {
        "data": data_summary(pf, split0, weekdays),
        "attempt": att,
        "lloyd_by_hand": lloyd_by_hand(pf, att),
        "verify_pca": verify_pca(shape(pf.P16[tr])),
        "verify_kmeans": verify_kmeans(shape(pf.P16[tr])),
        "modes": modes_figure(pf, split0, out_dir),
        "bracket": bracket(pf, splits, out_dir),
        "trace": tr_.numbers,
        "size_mode": size_mode(pf, split0, out_dir),
        "day_leakage": day_leakage(pf, split0, hourly, dates, out_dir=out_dir),
        "typical_days": days,
        "season": season(pf, hourly, dates, centers),
        "decision": decision(pf, att, out_dir),
        "reidentify": reidentify(pf),
        "cited": CITED,
    }
    # Asserts FAISS against NumPy; the slides quote none of its numbers, so it is printed only.
    print(f"retrieval check: {retrieval_check(pf, split0)}")
    # the visuals that carry the slides
    q = int(att["q"])
    title_art_figure(pf, split0, out_dir)
    numbers["carpet"] = carpet_figure(pf, hourly, dates, q, out_dir)
    numbers["plane"] = plane_figure(pf, split0, out_dir)
    numbers["kmeans_build"] = kmeans_build_figure(pf, split0, out_dir)
    numbers["rebuild"] = rebuild_figure(pf, split0, q, out_dir)
    fingerprint_figure(numbers["reidentify"], out_dir)
    aml_style.thumbnails(out_dir, ("plane", "rebuild", "size_mode", "typical_days"))
    write_numbers(numbers, out_dir)
    print(write_tex_macros(tex_macros(tr_, att, pf), out_dir, "cee_l04_load_profiles.py"))


if __name__ == "__main__":
    main()
