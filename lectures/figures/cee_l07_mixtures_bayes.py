"""Figures and numbers for CEE Lecture 7: naive Bayes, mixtures and EM, density estimation.

Run::

    pixi run -e aml-env python figures/cee_l07_mixtures_bayes.py \
        --out build/fig/cee_l07_mixtures_bayes

Writes the PNG figures, ``numbers.json`` (the numbers the slides quote, each rounded once from
its unrounded value), and ``numbers.tex`` (one reading traced through the mixture, for the TikZ
diagram). Data: ``data/derived/portland_i205_5min.csv`` from ``scripts/fetch_portland.sh`` (USDOT
Data Capture and Management test data set, Portland I-205, Sept 15 - Nov 14, 2011; CC BY-SA
3.0): 5-minute records at two southbound stations, Glisan (1143) and Stark/Washington (1051),
three lanes each.

The question: at what occupancy does each station break down from free flow to congestion? A
ramp meter holds the occupancy just below that value. Most loop detectors measure volume and
occupancy but not speed; these dual loops also measure speed, which labels the regimes for
checking.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

import aml_style
import title_art
from common import (
    DATA_DIR,
    parse_out_dir,
    pct,
    require,
    rnd,
    seed_all,
    tex_num,
    write_numbers,
    write_tex_macros,
)

CSV: Final[Path] = DATA_DIR / "derived" / "portland_i205_5min.csv"
SEED: Final[int] = 0
GLISAN: Final[int] = 1143  # the station the models are fitted at
STARK: Final[int] = 1051  # Stark/Washington, downstream: where they are applied
STATIONS: Final[Mapping[int, str]] = MappingProxyType({GLISAN: "Glisan", STARK: "Stark/Washington"})
# Mileposts from the data set's station table (scripts/fetch_portland.sh lists them).
MILEPOSTS: Final[Mapping[int, float]] = MappingProxyType({GLISAN: 21.12, STARK: 20.34})
LANES: Final[int] = 3
FULL_READINGS: Final[int] = 45  # 20-second lane readings per 5 minutes: 3 lanes x 15
MIN_READINGS: Final[int] = 40  # keep a record with at least this many of them
CONGESTED_MPH: Final[float] = 45.0  # the label: 5-minute speed below 45 mph
ATTEMPT_DAY: Final[str] = "2011-10-06"
ATTEMPT_TIMES: Final[tuple[str, ...]] = ("14:00", "14:30", "14:45", "15:35", "18:20", "18:35")
TRACE_TIME: Final[str] = "2011-10-06 14:45"

# EM in one dimension, and where its posterior crosses a level
EM_TOL: Final[float] = 1e-10
START_QUANTILES: Final[tuple[float, float]] = (0.50, 0.98)  # the stated start's two means
BREAKDOWN_LEVEL: Final[float] = 0.5
WIDTH_LEVELS: Final[tuple[float, float]] = (0.1, 0.9)  # how sharply the posterior rises
CROSSING_GRID_MAX: Final[float] = 60.0  # occupancy (%) searched for a crossing
CROSSING_GRID_POINTS: Final[int] = 60001  # steps of 0.001 points
PI_SENSITIVITY: Final[tuple[float, float]] = (0.05, 0.30)  # congested shares tried
LABEL_LR_C: Final[float] = 1e6  # the labels' crossing: logistic regression, almost no penalty
LR_MAX_ITER: Final[int] = 5000
CAPACITY_BIN_PCT: Final[float] = 2.0  # occupancy bins for the capacity (flow) peak
CAPACITY_QUANTILE: Final[float] = 0.95
FLOW_BANDS: Final[tuple[tuple[float, float], ...]] = ((15.0, 20.0), (35.0, 40.0))

# The hand step (the "Your turn" slide): six occupancies and a start
HAND_X: Final[tuple[float, ...]] = (3.0, 4.0, 5.0, 20.0, 24.0, 28.0)
HAND_MU: Final[tuple[float, float]] = (5.0, 20.0)
HAND_SD: Final[tuple[float, float]] = (5.0, 5.0)
HAND_PI: Final[tuple[float, float]] = (0.5, 0.5)

# EM in two dimensions (scikit-learn): restarts and BIC
RESTARTS: Final[int] = 20
N_INIT: Final[int] = 5  # starts per fit; the best is kept
SKLEARN_TOL: Final[float] = 1e-3  # GaussianMixture's default tolerance, set explicitly
SKLEARN_MAX_ITER: Final[int] = 100  # GaussianMixture's default, set explicitly
TIGHT_TOL: Final[float] = 1e-8
TIGHT_MAX_ITER: Final[int] = 5000
STALL_GAP: Final[float] = 1000.0  # a start stalls this far below the best log-likelihood
KS: Final[tuple[int, ...]] = tuple(range(1, 11))
BIC_MAX_ITER: Final[int] = 2000

# Density estimates of speed
KDE_BANDWIDTHS: Final[tuple[tuple[float | str, str], ...]] = (
    (0.02, "narrow"),
    ("scott", "Scott's rule"),
    (0.4, "wide"),
)
KDE_GRID: Final[tuple[float, float, int]] = (5.0, 75.0, 1401)
QUOTED_PEAKS: Final[tuple[str, ...]] = ("Scott's rule", "wide")  # the narrow one's are noise
BUMP_BAND_MPH: Final[tuple[float, float]] = (43.0, 48.0)  # the bump near 45 mph

# Transfer from Glisan to Stark/Washington
TRANSFER_LR_C: Final[float] = 1.0  # scikit-learn's default ridge penalty, set explicitly
UNSURE_BAND: Final[tuple[float, float]] = (0.1, 0.9)  # posteriors between these are unsure

# The day figure's readings
NIGHT_HOURS: Final[tuple[str, str]] = ("00:00", "05:00")
DAYTIME_HOURS: Final[tuple[str, str]] = ("08:00", "14:00")
MIDDAY_SPEED_HOURS: Final[tuple[str, str]] = ("12:00", "14:30")
HIGH_OCC_PCT: Final[float] = 20.0
DAY_QUANTILES: Final[tuple[float, float]] = (0.1, 0.9)


# ---------------------------------------------------------------------------------------------
# Data


def load() -> pd.DataFrame:
    """5-minute station records with most readings, flow per lane, and the speed label.

    Returns:
        Records with at least ``MIN_READINGS`` lane readings and a speed; ``flow`` in vehicles
        per hour per lane, and ``congested`` (0/1).

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_portland.sh``).
    """
    d = pd.read_csv(require(CSV, "fetch_portland.sh"), parse_dates=["time"])
    d = d[(d["readings"] >= MIN_READINGS) & d["speed"].notna()].reset_index(drop=True)
    # ``volume`` sums only the lane readings present, so a record with 40-44 of its 45 reads
    # low; scale it to the full 5 minutes. Occupancy and speed are means and need no scaling.
    full = d["volume"] * FULL_READINGS / d["readings"]
    d["flow"] = full * 12 / LANES  # vehicles per hour per lane
    d["congested"] = (d["speed"] < CONGESTED_MPH).astype(int)
    return d


# ---------------------------------------------------------------------------------------------
# EM for a one-dimensional Gaussian mixture (the code slide's listing, with the step factored)


def em_step(
    x: np.ndarray, mu: np.ndarray, sd: np.ndarray, pi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One EM step for a Gaussian mixture in one dimension.

    The E-step works with log densities: ``logsumexp`` gives log p(x_i) without the underflow
    that summing the densities themselves would risk for a reading far from every component.

    Args:
        x: Data, shape ``(n,)``.
        mu: Means, shape ``(K,)``.
        sd: Standard deviations, shape ``(K,)``.
        pi: Weights, shape ``(K,)``.

    Returns:
        The updated means, standard deviations, and weights, and the log-likelihood of ``x``
        under the parameters passed in (before the step).
    """
    logd = np.log(pi) + norm.logpdf(x[:, None], mu, sd)  # E-step, in logs: n x K
    ll = logsumexp(logd, axis=1)  # log p(x_i)
    r = np.exp(logd - ll[:, None])  # responsibilities
    nk = r.sum(axis=0)  # M-step
    mu = (r * x[:, None]).sum(axis=0) / nk
    sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
    return mu, sd, nk / len(x), float(ll.sum())


def em_gmm_1d(
    x: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    pi: np.ndarray,
    *,
    tol: float = EM_TOL,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """EM for a K-component Gaussian mixture in one dimension, to convergence.

    Args:
        x: Data, shape ``(n,)``.
        mu: Initial means, shape ``(K,)``.
        sd: Initial standard deviations, shape ``(K,)``.
        pi: Initial weights, shape ``(K,)``.
        tol: Stop when the log-likelihood gains less than this.
        max_steps: Give up after this many steps.

    Returns:
        Means, standard deviations, weights, and the log-likelihood before each step (its
        length is the number of steps).

    Raises:
        RuntimeError: If EM has not converged after ``max_steps``.
    """
    mu, sd, pi = (np.asarray(v, dtype=float) for v in (mu, sd, pi))
    history: list[float] = []
    for step in range(1, max_steps + 1):
        mu, sd, pi, ll = em_step(x, mu, sd, pi)
        history.append(ll)
        if step > 1 and history[-1] - history[-2] < tol:
            return mu, sd, pi, history
    raise RuntimeError("EM did not converge")


def em_steps(
    x: np.ndarray, mu: np.ndarray, sd: np.ndarray, pi: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The parameters after exactly ``n`` EM steps, with no stopping rule.

    Args:
        x: Data, shape ``(n_records,)``.
        mu: Initial means, shape ``(K,)``.
        sd: Initial standard deviations, shape ``(K,)``.
        pi: Initial weights, shape ``(K,)``.
        n: Steps to take.

    Returns:
        Means, standard deviations, and weights after ``n`` steps.
    """
    mu, sd, pi = (np.asarray(v, dtype=float) for v in (mu, sd, pi))
    for _ in range(n):
        mu, sd, pi, _ll = em_step(x, mu, sd, pi)
    return mu, sd, pi


def posterior_high(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """P(the component with the larger mean | x), from log densities.

    Args:
        x: Occupancies, shape ``(n,)``.
        mu: Component means, shape ``(K,)``.
        sd: Component standard deviations, shape ``(K,)``.
        pi: Component weights, shape ``(K,)``.

    Returns:
        The posterior of the highest-mean component at each ``x``, shape ``(n,)``.
    """
    logd = np.log(pi) + norm.logpdf(np.asarray(x, dtype=float)[:, None], mu, sd)
    return np.exp(logd[:, int(np.argmax(mu))] - logsumexp(logd, axis=1))


def posterior_crossing(mu: np.ndarray, sd: np.ndarray, pi: np.ndarray, level: float) -> float:
    """The lowest occupancy above the free mean where the congested posterior reaches a level.

    Args:
        mu: Component means, shape ``(K,)``.
        sd: Component standard deviations, shape ``(K,)``.
        pi: Component weights, shape ``(K,)``.
        level: The posterior level, in (0, 1).

    Returns:
        The occupancy (%), on a 0.001-point grid up to ``CROSSING_GRID_MAX``.

    Raises:
        ValueError: If the posterior never reaches ``level`` between the free mean and
            ``CROSSING_GRID_MAX``.
    """
    grid = np.linspace(0.0, CROSSING_GRID_MAX, CROSSING_GRID_POINTS)
    post = posterior_high(grid, mu, sd, pi)
    above = grid[(post >= level) & (grid > mu.min())]
    if above.size == 0:
        raise ValueError(
            f"the congested posterior never reaches {level} between the free mean "
            f"{mu.min():.2f}% and {CROSSING_GRID_MAX:.0f}% occupancy"
        )
    return float(above.min())


def init_from_quantiles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The stated start: means at two quantiles, the overall SD for both, equal weights.

    Args:
        x: Occupancies, shape ``(n,)``.

    Returns:
        Means at ``START_QUANTILES``, both standard deviations ``x.std()``, weights 0.5.
    """
    mu = np.quantile(x, list(START_QUANTILES))
    return mu, np.full(2, x.std()), np.full(2, 0.5)


def hand_step() -> dict[str, object]:
    """One EM step on the hand example, as the "Your turn" slide asks students to do it.

    Returns:
        The congested responsibilities of x = 5 and x = 20, the density ratio at x = 20, the
        new means and spreads, and the new 0.5 boundary.
    """
    x, mu, sd, pi = (np.array(v) for v in (HAND_X, HAND_MU, HAND_SD, HAND_PI))
    dens = pi * norm.pdf(x[:, None], mu, sd)
    r = dens / dens.sum(axis=1, keepdims=True)
    mu1, sd1, pi1 = em_steps(x, mu, sd, pi, 1)
    i5, i20 = HAND_X.index(5.0), HAND_X.index(20.0)
    return {
        "r_cong": [rnd(r[i5, 1], 3), rnd(r[i20, 1], 3)],
        "ratio_at_20": round(float(dens[i20, 1] / dens[i20, 0])),
        "mu": [rnd(v, 1) for v in mu1],
        "sd": [rnd(v, 1) for v in sd1],
        "boundary": rnd(posterior_crossing(mu1, sd1, pi1, BREAKDOWN_LEVEL), 1),
    }


@dataclass(frozen=True)
class StationFit:
    """EM on one station's occupancy, and its checks against the speed label.

    Attributes:
        mu: Component means (% occupancy), shape ``(2,)``.
        sd: Component standard deviations, shape ``(2,)``.
        pi: Component weights, shape ``(2,)``.
        steps: EM steps to converge.
        loglik_first: The log-likelihood at the start.
        loglik: The log-likelihood before the last step.
        breakdown: Occupancy (%) where the congested posterior reaches 0.5, unrounded: the
            threshold the transfer counts use (the slides show it to one decimal).
        width: Occupancy points the posterior takes to rise from 0.1 to 0.9.
        crossing_at_pi: The 0.5 crossing with the congested weight set to each
            ``PI_SENSITIVITY`` share (means and spreads kept).
        breakdown_labels: The 0.5 crossing of a logistic fit to the speed labels.
        agree: Share of records where EM's regime matches the speed label.
        always_free: Share of records the speed label calls free.
        capacity_bin: The 2-point occupancy bin where the 95th percentile of flow peaks.
        capacity_flow: That 95th percentile (veh/h/lane).
        records: Records at the station.
    """

    mu: np.ndarray
    sd: np.ndarray
    pi: np.ndarray
    steps: int
    loglik_first: float
    loglik: float
    breakdown: float
    width: float
    crossing_at_pi: tuple[float, ...]
    breakdown_labels: float
    agree: float
    always_free: float
    capacity_bin: tuple[int, int]
    capacity_flow: float
    records: int

    def numbers(self) -> dict[str, object]:
        """The fit as numbers.json quotes it.

        Returns:
            The parameters to 4 decimals, the crossings to 1, and the checks.
        """
        lo, hi = self.capacity_bin
        return {
            "mu": [rnd(v, 4) for v in self.mu],
            "sd": [rnd(v, 4) for v in self.sd],
            "pi": [rnd(v, 4) for v in self.pi],
            "steps": self.steps,
            "loglik": self.loglik,
            "breakdown_em": rnd(self.breakdown, 1),
            "breakdown_labels": rnd(self.breakdown_labels, 1),
            "labels_minus_em": rnd(self.breakdown_labels - self.breakdown, 1),
            "posterior_10_90_width": rnd(self.width, 1),
            "pi_cong_tried": list(PI_SENSITIVITY),
            "crossing_at_pi_cong": [rnd(c, 1) for c in self.crossing_at_pi],
            "agree_pct": pct(self.agree),
            "always_free_pct": pct(self.always_free),
            "capacity_bin": [lo, hi],
            "capacity_flow": int(round(self.capacity_flow, -1)),
            "em_in_capacity_bin": bool(lo <= self.breakdown < hi),
            "labels_in_capacity_bin": bool(lo <= self.breakdown_labels < hi),
            "records": self.records,
        }


# ---------------------------------------------------------------------------------------------
# Experiments


def summary(d: pd.DataFrame) -> dict[str, object]:
    """Counts for the question slide.

    Args:
        d: The records from :func:`load`.

    Returns:
        Records, congested share, and dates per station; the records whose flow was scaled
        up; the distance between the stations.
    """
    out: dict[str, object] = {}
    for st, name in STATIONS.items():
        s = d[d["station"] == st]
        out[name] = {
            "records": int(len(s)),
            "congested_pct": pct(s["congested"].mean()),
            "first": str(s["time"].min().date()),
            "last": str(s["time"].max().date()),
        }
    out["scaled_records"] = int((d["readings"] < FULL_READINGS).sum())  # flow scaled up
    out["gap_miles"] = rnd(MILEPOSTS[GLISAN] - MILEPOSTS[STARK], 1)
    return out


def attempt(d: pd.DataFrame) -> dict[str, object]:
    """Six 5-minute records at Glisan on the attempt day: occupancy, flow, then the speed.

    Args:
        d: The records from :func:`load`.

    Returns:
        For each attempt time: occupancy (%), flow (veh/h/lane, to 10), speed (mph).
    """
    s = d[d["station"] == GLISAN].set_index("time")
    rows = {}
    for t in ATTEMPT_TIMES:
        r = s.loc[pd.Timestamp(f"{ATTEMPT_DAY} {t}")]
        rows[t] = {
            "occupancy": rnd(r["occupancy"], 1),
            "flow": int(round(r["flow"], -1)),
            "speed": int(round(float(r["speed"]))),
        }
    return rows


def transitions(s: pd.DataFrame) -> np.ndarray:
    """Records whose speed label differs from the record 5 minutes before or after.

    Args:
        s: One station's records, one per 5-minute ``time``.

    Returns:
        A boolean array over ``s``'s rows; a missing neighbour does not count as a change.

    Raises:
        ValueError: If a station has two records at one time.
    """
    lab = s.set_index("time")["congested"]
    if not lab.index.is_unique:
        raise ValueError("two records at one time")
    prev = lab.shift(1, freq="5min").reindex(lab.index)
    nxt = lab.shift(-1, freq="5min").reindex(lab.index)
    return ((prev.notna() & (prev != lab)) | (nxt.notna() & (nxt != lab))).to_numpy()


def kde_figure(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Density of 5-minute speeds at Glisan with three bandwidths.

    Args:
        d: The records from :func:`load`.
        out_dir: The figure directory.

    Returns:
        Each bandwidth's h and peak count, the peaks and valleys of the two smooth ones, and
        the share of records in transition near 45 mph against all records.
    """
    s = d[d["station"] == GLISAN]
    sp = s["speed"].to_numpy()
    grid = np.linspace(*KDE_GRID)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    out: dict[str, object] = {}
    for i, (bw, name) in enumerate(KDE_BANDWIDTHS):
        k = gaussian_kde(sp, bw_method=bw)
        f = k(grid)
        peaks = grid[1:-1][(f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])]
        valleys = grid[1:-1][(f[1:-1] < f[:-2]) & (f[1:-1] < f[2:])]
        h = float(k.factor * sp.std(ddof=1))
        out[name] = {"h_mph": rnd(h, 2), "peaks": len(peaks)}
        if name in QUOTED_PEAKS:  # the slides quote only the narrow one's peak count
            out[name]["peak_mph"] = [rnd(p, 1) for p in peaks]
            out[name]["valley_mph"] = [rnd(v, 1) for v in valleys]
        ax.plot(
            grid,
            f,
            aml_style.LINESTYLES[i],
            color=aml_style.PALETTE[i],
            label=f"{name}, h = {h:.1f} mph",
        )
    ax.axvline(CONGESTED_MPH, color=aml_style.MUTED, ls=":", lw=1.5)
    ax.text(
        CONGESTED_MPH + 0.8, ax.get_ylim()[1] * 0.92, "45 mph", color=aml_style.MUTED, fontsize=12
    )
    ax.set_xlabel("5-minute speed at Glisan (mph)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=11)
    aml_style.save(fig, out_dir, "kde_bandwidth")
    # The notes' reading of the 45 mph bump: records in transition between regimes.
    trans = transitions(s)
    band = (sp >= BUMP_BAND_MPH[0]) & (sp <= BUMP_BAND_MPH[1])
    out["bump"] = {
        "band_mph": list(BUMP_BAND_MPH),
        "records": int(band.sum()),
        "transition_pct": pct(trans[band].mean()),
        "transition_all_pct": pct(trans.mean()),
    }
    return out


def em_station(d: pd.DataFrame, st: int) -> StationFit:
    """EM on occupancy at one station, from the stated start; checks against the speed label.

    Args:
        d: The records from :func:`load`.
        st: The station id.

    Returns:
        The fit and its checks.

    Raises:
        RuntimeError: If EM's log-likelihood falls at any step.
    """
    s = d[d["station"] == st]
    x = s["occupancy"].to_numpy()
    y = s["congested"].to_numpy()
    mu, sd, pi, hist = em_gmm_1d(x, *init_from_quantiles(x))
    if not np.all(np.diff(hist) > -1e-9):
        raise RuntimeError("EM's log-likelihood must not decrease")
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    # How sure the posterior is: the occupancies it takes to go from 0.1 to 0.9.
    width = posterior_crossing(mu, sd, pi, WIDTH_LEVELS[1]) - posterior_crossing(
        mu, sd, pi, WIDTH_LEVELS[0]
    )
    # How much the crossing depends on the congested share (the weights alone changed).
    at_pi = []
    for share in PI_SENSITIVITY:
        w = np.empty(2)
        w[hi], w[lo] = share, 1 - share
        at_pi.append(posterior_crossing(mu, sd, w, BREAKDOWN_LEVEL))
    lr = LogisticRegression(C=LABEL_LR_C, max_iter=LR_MAX_ITER).fit(x[:, None], y)
    # Every record falls in a 2-point bin, the highest occupancies too.
    bin_lo = np.floor(x / CAPACITY_BIN_PCT) * CAPACITY_BIN_PCT
    q95 = s.groupby(bin_lo)["flow"].quantile(CAPACITY_QUANTILE)
    peak = float(q95.idxmax())
    return StationFit(
        mu=mu,
        sd=sd,
        pi=pi,
        steps=len(hist),
        loglik_first=hist[0],
        loglik=hist[-1],
        breakdown=posterior_crossing(mu, sd, pi, BREAKDOWN_LEVEL),
        width=width,
        crossing_at_pi=tuple(at_pi),
        breakdown_labels=float(-lr.intercept_[0] / lr.coef_[0, 0]),
        agree=float(((posterior_high(x, mu, sd, pi) > BREAKDOWN_LEVEL) == (y == 1)).mean()),
        always_free=1 - float(y.mean()),
        capacity_bin=(int(peak), int(peak + CAPACITY_BIN_PCT)),
        capacity_flow=float(q95.max()),
        records=int(len(s)),
    )


def verify(d: pd.DataFrame, e: StationFit) -> dict[str, object]:
    """EM from scratch (Glisan's fit) against scikit-learn's GaussianMixture from the same start.

    Args:
        d: The records from :func:`load`.
        e: :func:`em_station`'s fit at Glisan.

    Returns:
        Steps, the largest parameter difference, the start, the fit, and the log-likelihood
        at the start and at convergence.

    Raises:
        RuntimeError: If scikit-learn's parameters differ by 1e-5 or more.
    """
    x = d.loc[d["station"] == GLISAN, "occupancy"].to_numpy()
    mu0, sd0, pi0 = init_from_quantiles(x)
    ref = GaussianMixture(
        2,
        covariance_type="full",
        weights_init=pi0,
        means_init=mu0[:, None],
        precisions_init=(1 / sd0**2)[:, None, None],
        reg_covar=0.0,
        tol=1e-12,
        max_iter=100_000,
    ).fit(x[:, None])
    order = np.argsort(ref.means_.ravel())
    mine = np.argsort(e.mu)
    diff = max(
        float(np.max(np.abs(e.mu[mine] - ref.means_.ravel()[order]))),
        float(np.max(np.abs(e.sd[mine] - np.sqrt(ref.covariances_.ravel()[order])))),
        float(np.max(np.abs(e.pi[mine] - ref.weights_[order]))),
    )
    if not diff < 1e-5:
        raise RuntimeError(f"EM and scikit-learn differ by {diff:.1e}")
    return {
        "steps": e.steps,
        "max_abs_diff": float(f"{diff:.0e}"),
        "records": int(len(x)),
        "start_mu": [rnd(v, 2) for v in mu0],
        "mu": [rnd(v, 2) for v in e.mu[mine]],
        "sd": [rnd(v, 2) for v in e.sd[mine]],
        "pi": [rnd(v, 3) for v in e.pi[mine]],
        "loglik_first": round(e.loglik_first),
        "loglik_last": round(e.loglik),
    }


def mixture_2d(k: int, seed: int, **kwargs: object) -> GaussianMixture:
    """An unfitted K-component, full-covariance GaussianMixture with the defining settings.

    Args:
        k: Components.
        seed: ``random_state``.
        **kwargs: Settings that differ from the stated ones (``init_params``, ``n_init``,
            ``tol``, ``max_iter``).

    Returns:
        The estimator; its tolerance and iterations default to scikit-learn's own, set here
        so the code names them.
    """
    settings: dict[str, object] = {
        "covariance_type": "full",
        "init_params": "kmeans",
        "tol": SKLEARN_TOL,
        "max_iter": SKLEARN_MAX_ITER,
        "n_init": 1,
    }
    settings.update(kwargs)
    return GaussianMixture(k, random_state=seed, **settings)


def fit_two_regimes(d: pd.DataFrame) -> GaussianMixture:
    """The 2-component mixture on (occupancy, flow) at Glisan, best of ``N_INIT`` starts.

    Args:
        d: The records from :func:`load`.

    Returns:
        The fitted mixture.
    """
    X = d.loc[d["station"] == GLISAN, ["occupancy", "flow"]].to_numpy()
    return mixture_2d(2, SEED, n_init=N_INIT).fit(X)


def random_starts(X: np.ndarray) -> tuple[dict[str, object], GaussianMixture, int]:
    """EM on (occupancy, flow) from random starts, at the default and at a tight tolerance.

    Each of ``RESTARTS`` seeds starts from random responsibilities (``init_params="random"``,
    not scikit-learn's k-means default). A start stalls if it ends more than ``STALL_GAP``
    below the best tight-tolerance log-likelihood. The k-means start is run for comparison.

    Args:
        X: Glisan's (occupancy, flow) records, shape ``(n, 2)``.

    Returns:
        The numbers, the first start that stalls, and its seed.

    Raises:
        RuntimeError: If no start stalls (the slide shows one).
    """
    starts = [mixture_2d(2, seed, init_params="random").fit(X) for seed in range(RESTARTS)]
    lls = [float(g.score(X) * len(X)) for g in starts]
    means = [sorted(float(m) for m in g.means_[:, 0]) for g in starts]
    lls_tight = [
        float(
            mixture_2d(2, seed, init_params="random", max_iter=TIGHT_MAX_ITER, tol=TIGHT_TOL)
            .fit(X)
            .score(X)
            * len(X)
        )
        for seed in range(RESTARTS)
    ]
    lls_kmeans = [float(mixture_2d(2, seed).fit(X).score(X) * len(X)) for seed in range(RESTARTS)]
    best = max(lls_tight)
    bad = [i for i, v in enumerate(lls) if v < best - STALL_GAP]
    if not bad:
        raise RuntimeError("no start stalled")
    bad_occ = [m for i in bad for m in means[i]]  # both means of every stalled start
    numbers = {
        "starts": RESTARTS,
        "bad_starts": len(bad),
        "bad_starts_tight": sum(v < best - STALL_GAP for v in lls_tight),
        "bad_starts_kmeans": sum(v < best - STALL_GAP for v in lls_kmeans),
        "loglik_good": round(best),
        "loglik_bad_range": [round(min(lls[i] for i in bad)), round(max(lls[i] for i in bad))],
        "bad_means_occ_range": [rnd(min(bad_occ), 1), rnd(max(bad_occ), 1)],
    }
    return numbers, starts[bad[0]], bad[0]


def bic_by_k(X: np.ndarray, breakdown: float) -> dict[str, object]:
    """BIC of the (occupancy, flow) mixture for each K in ``KS``, best of ``N_INIT`` starts.

    Args:
        X: Glisan's (occupancy, flow) records, shape ``(n, 2)``.
        breakdown: Glisan's EM breakdown occupancy (%): a component whose mean occupancy is
            above it counts as congested.

    Returns:
        K, BIC, and the components' mean occupancies for each K (lists in K's order); the K
        with the lowest BIC and whether it is the grid's edge; the congested components per K.
    """
    bic, occ_means = [], []
    for k in KS:
        g = mixture_2d(k, SEED, n_init=N_INIT, max_iter=BIC_MAX_ITER).fit(X)
        bic.append(round(float(g.bic(X))))
        occ_means.append(np.sort(g.means_[:, 0]))
    n_cong = [int((m > breakdown).sum()) for m in occ_means]
    one = [k for k, n in zip(KS, n_cong, strict=True) if k > 1 and n == 1]
    top = [rnd(m[-1], 1) for k, m in zip(KS, occ_means, strict=True) if k in one]
    best_k = KS[int(np.argmin(bic))]
    return {
        "k": list(KS),
        "bic": bic,
        "bic_min_k": best_k,
        "bic_min_at_edge": best_k == KS[-1],
        "bic_falls_every_k": bool(np.all(np.diff(bic) < 0)),
        "occ_means_by_k": [[rnd(v, 1) for v in m] for m in occ_means],
        "congested_components_by_k": n_cong,
        "one_congested_ks": one,
        "top_mean_occ_range": [min(top), max(top)],
    }


def bic_figure(bic: dict[str, object], out_dir: Path) -> None:
    """BIC against the number of components.

    Args:
        bic: :func:`bic_by_k`'s numbers.
        out_dir: The figure directory.
    """
    ks, vals = list(bic["k"]), [v / 1000 for v in bic["bic"]]
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.plot(ks, vals, "o-", color=aml_style.PALETTE[0])
    ax.set_xlabel("components K")
    ax.set_ylabel("BIC (thousands; lower is better)")
    ax.set_xticks(ks)
    ax.annotate(
        "2 regimes",
        (2, vals[ks.index(2)]),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=12,
        color=aml_style.PALETTE[1],
    )
    aml_style.save(fig, out_dir, "bic")


def agree_two_regimes(d: pd.DataFrame, good: GaussianMixture) -> float:
    """Share of Glisan's records where the 2-D mixture's regime matches the speed label.

    Args:
        d: The records from :func:`load`.
        good: :func:`fit_two_regimes`'s fit.

    Returns:
        The share, in [0, 1].
    """
    s = d[d["station"] == GLISAN]
    hi = int(np.argmax(good.means_[:, 0]))
    pred = good.predict(s[["occupancy", "flow"]].to_numpy()) == hi
    return float((pred == (s["congested"].to_numpy() == 1)).mean())


def fundamental_figure(
    d: pd.DataFrame, em: Mapping[int, StationFit], out_dir: Path
) -> dict[str, object]:
    """Flow against occupancy at both stations, colored by the EM posterior; breakdown lines.

    Args:
        d: The records from :func:`load`.
        em: :func:`em_station`'s fit per station.
        out_dir: The figure directory.

    Returns:
        Per station, the median flow of EM's congested records in each ``FLOW_BANDS`` band
        of occupancy (veh/h/lane, to 10): the alt text's reading of the congested branch.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    out: dict[str, object] = {"bands_occ": [list(b) for b in FLOW_BANDS]}
    for ax, (st, name) in zip(axes, STATIONS.items(), strict=True):
        s = d[d["station"] == st]
        e = em[st]
        occ = s["occupancy"].to_numpy()
        post = posterior_high(occ, e.mu, e.sd, e.pi)
        free = post <= BREAKDOWN_LEVEL
        ax.plot(
            s["occupancy"][free],
            s["flow"][free],
            ".",
            ms=2,
            color=aml_style.PALETTE[0],
            alpha=0.3,
            label="EM: free flow",
        )
        ax.plot(
            s["occupancy"][~free],
            s["flow"][~free],
            "x",
            ms=3,
            color=aml_style.PALETTE[1],
            alpha=0.4,
            label="EM: congested",
        )
        ax.axvline(e.breakdown, color=aml_style.INK, ls="--", lw=1.8)
        ax.text(
            e.breakdown + 0.8,
            2050,
            f"{e.breakdown:.1f}%",
            fontsize=13,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )  # over the crosses
        ax.set_title(f"{name} SB", fontsize=14)
        ax.set_xlabel("occupancy (%)")
        ax.set_xlim(0, 45)
        flow = s["flow"].to_numpy()
        out[name] = {
            "congested_flow_median": [
                int(round(float(np.median(flow[~free & (occ > a) & (occ <= b)])), -1))
                for a, b in FLOW_BANDS
            ]
        }
    axes[0].set_ylabel("flow (veh/h/lane)")
    axes[0].legend(loc="lower right", fontsize=11, markerscale=4)
    aml_style.save(fig, out_dir, "fundamental")
    return out


def transfer(d: pd.DataFrame, em: Mapping[int, StationFit]) -> dict[str, object]:
    """Train at Glisan, apply at Stark/Washington: generative and discriminative models.

    Naive Bayes and a full-covariance Gaussian per class (quadratic discriminant analysis)
    model p(x | class); logistic regression models P(class | x). Each is fitted on occupancy
    alone and on occupancy and flow (with one input, the two generative models coincide).
    The thresholds are the unrounded EM crossings (the slides show them to one decimal).

    Args:
        d: The records from :func:`load`.
        em: :func:`em_station`'s fit per station.

    Returns:
        Per model and per threshold: accuracy at home (the training records) and at Stark,
        false alarms, missed, total errors, and the share of unsure posteriors.
    """
    a, b = d[d["station"] == GLISAN], d[d["station"] == STARK]
    y_a, y_b = a["congested"].to_numpy(), b["congested"].to_numpy()
    out: dict[str, object] = {
        "test_records": int(len(b)),
        "test_congested": int(y_b.sum()),
        "test_free": int((1 - y_b).sum()),
    }
    occ, both = ["occupancy"], ["occupancy", "flow"]
    for name, cols, model in (
        ("nb_occ", occ, GaussianNB()),
        ("nb_occ_flow", both, GaussianNB()),
        ("full_occ_flow", both, QuadraticDiscriminantAnalysis(reg_param=0.0)),
        ("lr_occ", occ, LogisticRegression(C=TRANSFER_LR_C, max_iter=LR_MAX_ITER)),
        ("lr_occ_flow", both, LogisticRegression(C=TRANSFER_LR_C, max_iter=LR_MAX_ITER)),
    ):
        model.fit(a[cols].to_numpy(), y_a)
        p = model.predict_proba(b[cols].to_numpy())[:, 1]
        pa = model.predict_proba(a[cols].to_numpy())[:, 1]
        fa, missed = int(((p > 0.5) & (y_b == 0)).sum()), int(((p <= 0.5) & (y_b == 1)).sum())
        out[name] = {
            "acc_home_pct": pct(((pa > 0.5) == (y_a == 1)).mean()),
            "acc_pct": pct(((p > 0.5) == (y_b == 1)).mean()),
            "false_alarms": fa,
            "missed": missed,
            "errors": fa + missed,
            "unsure_home_pct": pct(((pa > UNSURE_BAND[0]) & (pa < UNSURE_BAND[1])).mean()),
            "unsure_pct": pct(((p > UNSURE_BAND[0]) & (p < UNSURE_BAND[1])).mean()),
        }
    # The notes' check that the penalty barely matters. C = inf is scikit-learn 1.8's spelling of
    # no penalty (``penalty=None`` is deprecated); it warns that C is then ignored, as intended.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Setting penalty=None will ignore", UserWarning)
        free = LogisticRegression(C=np.inf, max_iter=LR_MAX_ITER)
        free.fit(a[both].to_numpy(), y_a)
    p_free = free.predict_proba(b[both].to_numpy())[:, 1]
    out["lr_occ_flow_no_penalty_false_alarms"] = int(((p_free > 0.5) & (y_b == 0)).sum())
    labels = em[STARK].breakdown_labels
    for key, st in (("glisan_threshold_at_stark", GLISAN), ("own_threshold_at_stark", STARK)):
        flag = b["occupancy"].to_numpy() >= em[st].breakdown
        fa, missed = int((flag & (y_b == 0)).sum()), int((~flag & (y_b == 1)).sum())
        out[key] = {
            "threshold": rnd(em[st].breakdown, 1),
            "false_alarms": fa,
            "missed": missed,
            "errors": fa + missed,
            "from_labels": rnd(abs(labels - em[st].breakdown), 1),  # Stark's labels' crossing
        }
    return out


def transfer_figure(em: Mapping[int, StationFit], out_dir: Path) -> None:
    """The decisive figure: P(congested | occupancy) at each station, from EM alone.

    Args:
        em: :func:`em_station`'s fit per station.
        out_dir: The figure directory.
    """
    grid = np.linspace(0, 40, 801)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    for i, (st, name) in enumerate(STATIONS.items()):
        e = em[st]
        ax.plot(
            grid,
            posterior_high(grid, e.mu, e.sd, e.pi),
            aml_style.LINESTYLES[i],
            color=aml_style.PALETTE[i],
            label=f"{name}: {e.breakdown:.1f}%",
        )
        ax.axvline(e.breakdown, color=aml_style.PALETTE[i], ls=":", lw=1.5)
    ax.axhline(BREAKDOWN_LEVEL, color=aml_style.MUTED, lw=1)
    ax.set_xlabel("occupancy (%)")
    ax.set_ylabel("P(congested | occupancy), EM")
    ax.legend(
        loc="lower right",
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    )  # the 15.6% line runs behind it
    aml_style.save(fig, out_dir, "transfer")


def trace(d: pd.DataFrame, e: StationFit) -> dict[str, object]:
    """One reading through the Glisan mixture: the two densities, the posterior.

    Args:
        d: The records from :func:`load`.
        e: :func:`em_station`'s fit at Glisan.

    Returns:
        The rounded values the slide quotes, and the unrounded ones under ``_raw`` (not
        written to numbers.json; :func:`write_tex` formats them).
    """
    s = d[d["station"] == GLISAN].set_index("time")
    r = s.loc[pd.Timestamp(TRACE_TIME)]
    mu, sd, pi = e.mu, e.sd, e.pi
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    x = float(r["occupancy"])
    f = pi * norm.pdf(x, mu, sd)
    raw = {
        "occupancy": x,
        "flow": float(r["flow"]),
        "speed": float(r["speed"]),
        "mu_free": float(mu[lo]),
        "sd_free": float(sd[lo]),
        "pi_free": float(pi[lo]),
        "mu_cong": float(mu[hi]),
        "sd_cong": float(sd[hi]),
        "pi_cong": float(pi[hi]),
        "joint_free": float(f[lo]),
        "joint_cong": float(f[hi]),
        "post_cong": float(f[hi] / f.sum()),
    }
    return {
        "time": TRACE_TIME,
        "occupancy": rnd(x, 1),
        "flow": int(round(raw["flow"], -1)),
        "speed": int(round(raw["speed"])),
        "mu_free": rnd(mu[lo], 2),
        "sd_free": rnd(sd[lo], 2),
        "pi_free": rnd(raw["pi_free"], 3),
        "mu_cong": rnd(mu[hi], 2),
        "sd_cong": rnd(sd[hi], 2),
        "pi_cong": rnd(raw["pi_cong"], 3),
        "joint_free": raw["joint_free"],
        "joint_cong": raw["joint_cong"],
        "post_cong": rnd(raw["post_cong"], 3),
        "_raw": raw,
    }


def write_tex(tr: dict[str, object], records: int, out_dir: Path) -> Path:
    """Write the trace as LaTeX macros for the TikZ diagram, from the unrounded values.

    Args:
        tr: :func:`trace`'s output, with its ``_raw`` values.
        records: Glisan's records, for the diagram's data pill.
        out_dir: The figure directory.

    Returns:
        The path of ``numbers.tex``.

    Raises:
        TypeError: If ``tr`` has no ``_raw`` dictionary.
    """
    raw = tr["_raw"]
    if not isinstance(raw, dict):
        raise TypeError("trace() must carry its unrounded values under _raw")

    def sci(v: float) -> str:
        r"""Format a density as ``1.8\times10^{-5}``.

        Args:
            v: A positive density.

        Returns:
            Math-mode text; the exponent keeps its ASCII minus.
        """
        m, e = f"{v:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"  # densities: the mantissa is positive

    macros = {
        "Occ": tex_num(raw["occupancy"], ".1f"),
        "Flow": tex_num(tr["flow"], ","),
        "Speed": tex_num(raw["speed"], ".0f"),
        "MuF": tex_num(raw["mu_free"], ".1f"),
        "SdF": tex_num(raw["sd_free"], ".1f"),
        "PiF": tex_num(raw["pi_free"], ".2f"),
        "MuC": tex_num(raw["mu_cong"], ".1f"),
        "SdC": tex_num(raw["sd_cong"], ".1f"),
        "PiC": tex_num(raw["pi_cong"], ".2f"),
        "JF": sci(raw["joint_free"]),
        "JC": sci(raw["joint_cong"]),
        "Post": tex_num(raw["post_cong"], ".2f"),
        "NRec": tex_num(records, ","),
    }
    return write_tex_macros(macros, out_dir, "cee_l07_mixtures_bayes.py")


# ---------------------------------------------------------------------------------------------
# The visuals that carry the slides

BUILD_STEPS: Final[tuple[int, ...]] = (0, 1, 3)  # EM iterations drawn before the converged fit


def title_art_figure(em_g: StationFit, out_dir: Path) -> None:
    """The title art: two hills of light (the mixture density) under a dark sun.

    The horizon is Glisan's fitted mixture of occupancies, a tall free-flow hill and a low,
    wide congested one; two clouds of points glow above them.

    Args:
        em_g: :func:`em_station`'s fit at Glisan.
        out_dir: The figure directory.
    """
    mu, sd, pi = em_g.mu, em_g.sd, em_g.pi
    c = title_art.Canvas(seed=7)
    c.ground(("#0d1620", "#14111e", "#1c0c12", "#07060c"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.06, 0.92, 0.70, 4), soft=4), "#4d1c14", 0.85)
    rim = c.disc((0.30, 1.16), 0.12, soft=4) - c.disc((0.30, 1.16), 0.10, soft=4)
    c.glow(np.clip(rim, 0, 1), "#eaa548", 0.5, spread=36)
    c.glaze(c.disc((0.30, 1.16), 0.10, soft=3), "#1d0a10", 0.92)
    rng = np.random.default_rng(7)
    for (cx, cy, sx), col, n in (((0.30, 0.86, 0.05), "#9fb7c9", 40),
                                 ((0.70, 0.92, 0.09), "#f0a03c", 18)):  # fmt: skip
        pts = rng.normal((cx, cy), (sx, 0.05), (n, 2))
        c.glow(np.clip(sum(c.disc(tuple(q), 0.006, soft=1.5) for q in pts), 0, 1), col, 0.55,
               spread=7)  # fmt: skip
    x = np.linspace(0, 40, 400)
    dens = (pi * norm.pdf(x[:, None], mu, sd)).sum(axis=1)
    hill = np.sqrt(dens / dens.max())  # square root: the low congested hill stays visible
    curve = [list(zip(0.06 + 0.88 * x / 40, 0.14 + 0.42 * hill, strict=True))]
    c.glow(c.lines(curve, 4, soft=1.2), "#f3b45c", 0.85, spread=12)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: a stepped Andean form in rust; above it a pale cloud "
        "of points and a smaller amber one; a dark sun with a burning rim at the upper left, and "
        "a golden line rising into a tall hill and a low, wide one, the lecture's two traffic "
        "regimes.",
    )


def day_figure(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """One day at Glisan: occupancy and speed through the day; congestion shaded.

    Args:
        d: The records from :func:`load`.
        out_dir: The figure directory.

    Returns:
        The alt text's readings: night median occupancy, the daytime occupancy range (10th to
        90th percentile), the midday speed, the first and last congested records, the first
        record above ``HIGH_OCC_PCT``, and the slowest record.
    """
    s = d[(d["station"] == GLISAN) & (d["time"].dt.strftime("%Y-%m-%d") == ATTEMPT_DAY)]
    hours = s["time"].dt.hour + s["time"].dt.minute / 60
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.fill_between(hours, 0, 1, where=s["congested"].to_numpy() == 1, color=aml_style.PALETTE[1],
                    alpha=0.15, transform=ax.get_xaxis_transform(), lw=0)  # fmt: skip
    ax.plot(hours, s["occupancy"], color=aml_style.PALETTE[0], lw=2, ls="-", label="occupancy (%)")
    ax2 = ax.twinx()
    ax2.plot(hours, s["speed"], color=aml_style.INK, lw=1.4, ls="--", label="speed (mph)")
    ax2.spines["right"].set_visible(True)
    ax.set(
        xlabel=f"hour, {ATTEMPT_DAY}",
        ylabel="occupancy (%)",
        xticks=[0, 6, 12, 18, 24],
        xlim=(0, 24),
    )
    ax2.set_ylabel("speed (mph)")
    ax.set_title("Glisan southbound: congestion shaded", loc="left", fontsize=13)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=11)
    aml_style.save(fig, out_dir, "day")
    t = s.set_index("time")
    cong = t.index[t["congested"] == 1]
    day_occ = t.between_time(*DAYTIME_HOURS)["occupancy"].quantile(list(DAY_QUANTILES))
    slow = t["speed"].idxmin()
    return {
        "night_occ_median": rnd(t.between_time(*NIGHT_HOURS)["occupancy"].median(), 1),
        "daytime_occ_range": [rnd(v, 1) for v in day_occ],
        "midday_speed_median": int(round(t.between_time(*MIDDAY_SPEED_HOURS)["speed"].median())),
        "congested_first": cong.min().strftime("%H:%M"),
        "congested_last": cong.max().strftime("%H:%M"),
        "congested_records": int(len(cong)),
        "first_above_high_occ": t.index[t["occupancy"] > HIGH_OCC_PCT].min().strftime("%H:%M"),
        "min_speed": int(round(float(t["speed"].min()))),
        "min_speed_time": slow.strftime("%H:%M"),
    }


def _ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, color: str, ls: str = "-") -> None:
    """Draw the 2-standard-deviation ellipse of a 2D Gaussian.

    Args:
        ax: The axes.
        mean: The mean, shape ``(2,)``.
        cov: The covariance, shape ``(2, 2)``.
        color: The line color.
        ls: The line style.
    """
    vals, vecs = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    w, h = 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0])
    ax.add_patch(Ellipse(tuple(mean), w, h, angle=angle, fill=False, color=color, lw=2.4, ls=ls))


def hills_figure(d: pd.DataFrame, g: GaussianMixture, out_dir: Path) -> dict[str, object]:
    """Two regimes, two hills: the 2-component mixture density over (occupancy, flow), Glisan.

    Left, the log density in 3D. Right, the records coloured by the speed label (which EM never
    sees), with each component's 2-SD ellipse.

    Args:
        d: The records from :func:`load`.
        g: :func:`fit_two_regimes`'s fit.
        out_dir: The figure directory.

    Returns:
        The components' mean occupancy, mean flow, and weight, free first.
    """
    s = d[d["station"] == GLISAN]
    X = s[["occupancy", "flow"]].to_numpy()
    occ, flow = np.meshgrid(np.linspace(0, 45, 91), np.linspace(0, 2400, 91))
    logd = g.score_samples(np.column_stack([occ.ravel(), flow.ravel()])).reshape(occ.shape)
    z = np.maximum(logd / np.log(10), -8.0)  # log10 density, floor at -8
    fig = plt.figure(figsize=aml_style.FULL)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    aml_style.surface3d(ax, occ, flow, z, cmap="cividis")
    ax.set_xlabel("occupancy (%)", labelpad=8, fontsize=12)
    ax.set_ylabel("flow (veh/h/lane)", labelpad=10, fontsize=12)
    ax.set(xticks=[0, 20, 40], yticks=[0, 1000, 2000], zticks=[-8, -6, -4])
    ax.set_title("log density: two hills", loc="left", fontsize=13)
    aml_style.style3d(ax, elev=40, azim=-120)  # chosen from six: both hills in view
    ax.set_box_aspect(None, zoom=0.85)  # smaller box: the moved labels stay inside the crop
    ax2 = fig.add_subplot(1, 2, 2)
    for lab, color, size in ((0, aml_style.MUTED, 3), (1, aml_style.PALETTE[1], 5)):
        m = s["congested"].to_numpy() == lab
        ax2.scatter(X[m, 0], X[m, 1], s=size, color=color, alpha=0.4, lw=0)
    order = np.argsort(g.means_[:, 0])
    for k, color in zip(order, (aml_style.PALETTE[0], aml_style.PALETTE[1]), strict=True):
        _ellipse(ax2, g.means_[k], g.covariances_[k], color)
    ax2.set(xlabel="occupancy (%)", ylabel="flow (veh/h/lane)", xlim=(0, 45), ylim=(0, 2400))
    ax2.set_title("speed label: orange is below 45 mph", loc="left", fontsize=13)
    fig.subplots_adjust(wspace=0.3)
    aml_style.save(fig, out_dir, "hills")
    return {
        "means_occ": [rnd(g.means_[k, 0], 1) for k in order],
        "means_flow": [int(round(float(g.means_[k, 1]))) for k in order],
        "weights": [rnd(g.weights_[k], 2) for k in order],
    }


def em_build_figure(d: pd.DataFrame, em_g: StationFit, out_dir: Path) -> dict[str, object]:
    """EM on Glisan's occupancy: the two weighted components at steps 0, 1, 3, and converged.

    Args:
        d: The records from :func:`load`.
        em_g: :func:`em_station`'s fit at Glisan.
        out_dir: The figure directory.

    Returns:
        The means (free, congested) at each drawn step, and the steps.

    Raises:
        RuntimeError: If ``em_steps`` does not reproduce ``em_gmm_1d``'s fit.
    """
    x = d.loc[d["station"] == GLISAN, "occupancy"].to_numpy()
    start = init_from_quantiles(x)
    steps = em_g.steps
    final = em_steps(x, *start, steps)
    if not np.allclose(final[0], em_g.mu, rtol=0, atol=1e-9):
        raise RuntimeError("em_steps must match em_gmm_1d")
    grid = np.linspace(0, 45, 400)
    fig, axes = plt.subplots(1, len(BUILD_STEPS) + 1, figsize=aml_style.FULL, sharey=True)
    means = []
    for ax, n in zip(axes, (*BUILD_STEPS, steps), strict=True):
        mu, sd, pi = em_steps(x, *start, n)
        ax.hist(x, bins=np.arange(0, 46, 1.5), density=True, color="#DDDDDD", lw=0)
        comps = pi * norm.pdf(grid[:, None], mu, sd)
        lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
        means.append((float(mu[lo]), float(mu[hi])))
        ax.plot(grid, comps[:, lo], color=aml_style.PALETTE[0], lw=2.2, ls="-")
        ax.plot(grid, comps[:, hi], color=aml_style.PALETTE[1], lw=2.2, ls="--")
        ax.plot(grid, comps.sum(axis=1), color=aml_style.INK, lw=1.2, ls=":")
        title = "start" if n == 0 else (f"step {n}" if n != steps else f"converged, step {n}")
        ax.set_title(f"{title}\nmeans {mu[lo]:.1f}, {mu[hi]:.1f}", loc="left", fontsize=12)
        ax.set(xlabel="occupancy (%)", ylim=(0, 0.13), xticks=[0, 20, 40])
    axes[0].set_ylabel("density")
    aml_style.save(fig, out_dir, "em_build")
    after_one = BUILD_STEPS.index(1)
    return {
        "start_means": [rnd(v, 1) for v in means[0]],
        "steps": [*BUILD_STEPS, steps],
        "means": [[rnd(v, 1) for v in m] for m in means],
        # How far each mean moves after the first step, to convergence.
        "moved_after_step1": [rnd(abs(means[-1][j] - means[after_one][j]), 1) for j in (0, 1)],
    }


def posterior_figure(d: pd.DataFrame, em_g: StationFit, out_dir: Path) -> None:
    """The fitted mixture over Glisan's occupancy, and P(congested | occupancy) crossing 0.5.

    Args:
        d: The records from :func:`load`.
        em_g: :func:`em_station`'s fit at Glisan.
        out_dir: The figure directory.
    """
    x = d.loc[d["station"] == GLISAN, "occupancy"].to_numpy()
    mu, sd, pi = em_g.mu, em_g.sd, em_g.pi
    grid = np.linspace(0, 45, 400)
    comps = pi * norm.pdf(grid[:, None], mu, sd)
    lo, hi = int(np.argmin(mu)), int(np.argmax(mu))
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.hist(x, bins=np.arange(0, 46, 1.5), density=True, color="#DDDDDD", lw=0)
    ax.plot(grid, comps[:, lo], color=aml_style.PALETTE[0], lw=2.2, ls="-", label="free")
    ax.plot(grid, comps[:, hi], color=aml_style.PALETTE[1], lw=2.2, ls="--", label="congested")
    ax.set(xlabel="occupancy (%)", ylabel="density", ylim=(0, 0.13))
    ax2 = ax.twinx()
    ax2.plot(grid, posterior_high(grid, mu, sd, pi), color=aml_style.INK, lw=2.4, ls="-",
             label="P(congested | x)")  # fmt: skip
    ax2.spines["right"].set_visible(True)
    ax2.set(ylabel="P(congested | occupancy)", ylim=(0, 1.05))
    b = em_g.breakdown
    ax2.axvline(b, color=aml_style.INK, lw=1, ls=":")
    ax2.annotate(f"0.5 at {b:.1f}%", (b, 0.5), xytext=(10, -4), textcoords="offset points",
                 fontsize=12)  # fmt: skip
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=11)
    aml_style.save(fig, out_dir, "posterior")


def stall_figure(
    d: pd.DataFrame, good: GaussianMixture, bad: GaussianMixture, seed: int, out_dir: Path
) -> dict[str, object]:
    """A start that stalls (both means in free flow) beside the fit that finds the regimes.

    Args:
        d: The records from :func:`load`.
        good: :func:`fit_two_regimes`'s fit.
        bad: The first random start that stalls (from :func:`random_starts`).
        seed: That start's seed.
        out_dir: The figure directory.

    Returns:
        The stalled seed and the two log-likelihoods the panel titles show.
    """
    X = d.loc[d["station"] == GLISAN, ["occupancy", "flow"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    lls = []
    for ax, g, title in ((axes[0], bad, f"start {seed}: stalled"), (axes[1], good, "best of 5")):
        ax.scatter(X[:, 0], X[:, 1], s=3, color=aml_style.MUTED, alpha=0.35, lw=0)
        for k, color in zip(np.argsort(g.means_[:, 0]), aml_style.PALETTE[:2], strict=True):
            _ellipse(ax, g.means_[k], g.covariances_[k], color)
            ax.plot(*g.means_[k], "X", ms=12, color=color, mec=aml_style.INK)
        ll = g.score(X) * len(X)
        lls.append(round(float(ll)))
        ax.set_title(f"{title}: log-likelihood {ll:,.0f}", loc="left", fontsize=12.5)
        ax.set(xlabel="occupancy (%)", xlim=(0, 45), ylim=(0, 2400))
    axes[0].set_ylabel("flow (veh/h/lane)")
    aml_style.save(fig, out_dir, "stall")
    return {"stalled_seed": seed, "loglik_stalled": lls[0], "loglik_best": lls[1]}


def naive_figure(d: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """Free-flow records at Glisan: the real, tilted ellipse against naive Bayes' upright one.

    Args:
        d: The records from :func:`load`.
        out_dir: The figure directory.

    Returns:
        The correlation of occupancy and flow in free flow at Glisan.
    """
    s = d[(d["station"] == GLISAN) & (d["congested"] == 0)]
    X = s[["occupancy", "flow"]].to_numpy()
    mean, cov = X.mean(axis=0), np.cov(X.T)
    fig, ax = plt.subplots(figsize=aml_style.SIDE)
    ax.scatter(X[:, 0], X[:, 1], s=3, color=aml_style.MUTED, alpha=0.35, lw=0)
    _ellipse(ax, mean, cov, aml_style.PALETTE[0])
    _ellipse(ax, mean, np.diag(np.diag(cov)), aml_style.PALETTE[1], ls="--")
    ax.plot([], [], color=aml_style.PALETTE[0], lw=2.4, label="real: correlated")
    ax.plot([], [], color=aml_style.PALETTE[1], lw=2.4, ls="--", label="naive Bayes: independent")
    r = float(np.corrcoef(X.T)[0, 1])
    ax.set(xlabel="occupancy (%)", ylabel="flow (veh/h/lane)", xlim=(0, 20), ylim=(0, 2200))
    ax.set_title(f"free flow, Glisan: r = {r:.2f}", loc="left", fontsize=13)
    # Opaque: the independent model's dashed ellipse passes behind it.
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)  # fmt: skip
    aml_style.save(fig, out_dir, "naive")
    return {"corr_free": rnd(r, 2)}


# ---------------------------------------------------------------------------------------------
# Main


def main() -> None:
    """Generate every figure and number for CEE Lecture 7."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    d = load()
    em = {st: em_station(d, st) for st in STATIONS}
    good = fit_two_regimes(d)
    X = d.loc[d["station"] == GLISAN, ["occupancy", "flow"]].to_numpy()
    rest, stalled, stalled_seed = random_starts(X)
    rest["agree_2d_pct"] = pct(agree_two_regimes(d, good))
    bic = bic_by_k(X, em[GLISAN].breakdown)
    bic_figure(bic, out_dir)
    tr = trace(d, em[GLISAN])
    numbers = {
        "summary": summary(d),
        "attempt": attempt(d),
        "hand_step": hand_step(),
        "kde": kde_figure(d, out_dir),
        "em": {STATIONS[st]: v.numbers() for st, v in em.items()},
        "verify": verify(d, em[GLISAN]),
        "restarts": rest,
        "bic": bic,
        "fundamental": fundamental_figure(d, em, out_dir),
        "transfer": transfer(d, em),
        "trace": tr,
    }
    transfer_figure(em, out_dir)
    # the visuals that carry the slides
    title_art_figure(em[GLISAN], out_dir)
    numbers["day"] = day_figure(d, out_dir)
    numbers["hills"] = hills_figure(d, good, out_dir)
    numbers["em_build"] = em_build_figure(d, em[GLISAN], out_dir)
    posterior_figure(d, em[GLISAN], out_dir)
    numbers["stall"] = stall_figure(d, good, stalled, stalled_seed, out_dir)
    numbers["naive"] = naive_figure(d, out_dir)
    aml_style.thumbnails(out_dir, ("hills", "em_build", "naive", "transfer"))
    write_numbers(numbers, out_dir)
    print(write_tex(tr, em[GLISAN].records, out_dir))


if __name__ == "__main__":
    main()
