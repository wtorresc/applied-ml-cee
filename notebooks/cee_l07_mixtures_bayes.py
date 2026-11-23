"""Applied Machine Learning for CEE — Lecture 7: Mixtures, EM, and Naive Bayes.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l07_mixtures_bayes.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_portland.sh`` once (USDOT Portland I-205 test data, CC BY-SA 3.0).
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
    # Applied Machine Learning for CEE — Lecture 7: Mixtures, EM, and Naive Bayes

    **Slides:** *Applied Machine Learning for CEE*, Lecture 7 (`lectures/cee_l07_mixtures_bayes`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), Ch. 9 (clustering with probability models: mixtures and EM) and §1.3 (naive Bayes).
    **Data:** Portland, Oregon test data set, freeway loop detectors (USDOT Data Capture and Management program; Portland State University, 2012; doi:10.21949/1504472; CC BY-SA 3.0). I-205 southbound, stations Glisan (1143) and Stark/Washington (1051), Sept 15 to Nov 14, 2011. Fetch with `bash scripts/fetch_portland.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    The numbers the slides quote, the alt texts' readings included, are recomputed here and
    checked with `assert`s. It runs in about a minute on a CPU.

    **Your tasks** (from the slides): one EM step by hand, EM from scratch checked against
    scikit-learn, KDE bandwidths, EM's starts and BIC, and naive Bayes across stations.
    """)
    return


@app.cell
def _():
    import os
    import random
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import logsumexp
    from scipy.stats import gaussian_kde, norm
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.naive_bayes import GaussianNB

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    STATIONS = {1143: "Glisan", 1051: "Stark/Washington"}
    LANES = 3
    FULL_READINGS = 45  # 20-second lane readings per 5 minutes: 3 x 15
    MIN_READINGS = 40  # keep a record with at least this many
    CONGESTED_MPH = 45.0  # the label used only for checking
    KS = tuple(range(1, 11))  # BIC's components: 1 to 10
    PI_SENSITIVITY = (0.05, 0.30)  # congested shares tried for the crossing's sensitivity
    UNSURE_BAND = (0.1, 0.9)  # a posterior between these is unsure
    CAPACITY_BIN_PCT = 2.0  # occupancy bins for the flow peak
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        CAPACITY_BIN_PCT,
        CONGESTED_MPH,
        FULL_READINGS,
        GaussianMixture,
        GaussianNB,
        KS,
        LANES,
        LogisticRegression,
        MARKERS,
        MIN_READINGS,
        PALETTE,
        PI_SENSITIVITY,
        Path,
        QuadraticDiscriminantAnalysis,
        SEED,
        STATIONS,
        UNSURE_BAND,
        gaussian_kde,
        logsumexp,
        norm,
        np,
        pd,
        plt,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 4–6

    On your own, no notes, two minutes:

    1. $k$-means gives each building to its nearest center. What does that assume about the
       clusters' shapes and sizes?
    2. Least squares is maximum likelihood under which noise model?
    3. Logistic regression models $P(\text{poor}\mid\mathbf{x})$ directly. What else could you
       model instead, and use Bayes' rule?

    **Today:** at what occupancy does a freeway break down from free flow to congestion?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. Round clusters of the same size: hard assignments by distance alone
                   (Lecture 4). Mixtures give each cluster its own spread and weight, and
                   assign softly.
                2. Independent normal errors of one variance (Lecture 5). EM maximizes a
                   likelihood too, with a hidden variable.
                3. How $\mathbf{x}$ is distributed in each class and the class shares; Bayes'
                   rule turns them into $P(\text{class}\mid\mathbf{x})$ (Ng & Jordan, 2002).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    A ramp meter holds the freeway's **occupancy** (the share of time a loop is covered) just
    below breakdown; ALINEA adjusts the ramp rate each cycle to hold a target near the
    critical occupancy (Papageorgiou, Hadj-Salem & Blosseville, 1991). **The decision:** the
    target at each station. Too low: needless ramp queues. Too high: breakdown. Most loops
    measure only volume and occupancy; these dual loops also measure speed, which we use only
    to check what we learn without it.
    """)
    return


@app.cell
def _(CONGESTED_MPH, FULL_READINGS, LANES, MIN_READINGS, Path, STATIONS, pd):
    def find_csv() -> Path:
        """Locate ``data/derived/portland_i205_5min.csv`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_portland.sh``).
        """
        rel = Path("data") / "derived" / "portland_i205_5min.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_portland.sh`")

    d = pd.read_csv(find_csv(), parse_dates=["time"])
    d = d[(d["readings"] >= MIN_READINGS) & d["speed"].notna()].reset_index(drop=True)
    # The volume sums only the lane readings present: scale a record with 40-44 of its 45 up
    # to the full 5 minutes. Occupancy and speed are means and need no scaling.
    d["flow"] = d["volume"] * FULL_READINGS / d["readings"] * 12 / LANES  # veh/h/lane
    d["congested"] = (d["speed"] < CONGESTED_MPH).astype(int)
    _counts = {
        STATIONS[_s]: (int(len(_g)), round(100 * float(_g["congested"].mean()), 1))
        for _s, _g in d.groupby("station")
    }
    # Question slide: 17,036 records at Glisan, 17,154 at Stark/Washington; 10.1% and 13.1% congested.
    assert _counts == {"Glisan": (17036, 10.1), "Stark/Washington": (17154, 13.1)}, _counts
    # Notes: 996 records have 40 to 44 readings, and their volume is scaled up.
    assert int((d["readings"] < FULL_READINGS).sum()) == 996
    glisan = d[d["station"] == 1143].reset_index(drop=True)
    stark = d[d["station"] == 1051].reset_index(drop=True)
    pd.DataFrame(_counts, index=["5-minute records", "congested (%)"]).T
    return glisan, stark


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: free or congested?

    Six 5-minute records at Glisan on October 6, 2011. **Which are congested? Where did the
    freeway break down? Write the rule you used.** The speeds are hidden in the answer.
    """)
    return


@app.cell
def _(glisan, pd):
    _s = glisan.set_index("time")
    _times = ["14:00", "14:30", "14:45", "15:35", "18:20", "18:35"]
    _rows = _s.loc[[pd.Timestamp(f"2011-10-06 {_t}") for _t in _times]]
    attempt = pd.DataFrame(
        {
            "occupancy (%)": _rows["occupancy"].round(1).to_numpy(),
            "flow (veh/h/lane)": (_rows["flow"].round(-1)).astype(int).to_numpy(),
            "speed (mph)": _rows["speed"].round(0).to_numpy(),
        },
        index=_times,
    )
    assert attempt["occupancy (%)"].tolist() == [10.6, 11.1, 16.8, 24.9, 24.4, 6.0]
    assert attempt["flow (veh/h/lane)"].tolist() == [1550, 1620, 1460, 1620, 1400, 1150]
    assert attempt["speed (mph)"].tolist() == [55, 51, 30, 25, 23, 55]
    # The day figure's alt text: night and daytime occupancy, the midday speed, the congested
    # records, the first record above 20% occupancy, and the slowest record.
    _day = _s.loc["2011-10-06"]
    _cong = _day.index[_day["congested"] == 1]
    assert round(float(_day.between_time("00:00", "05:00")["occupancy"].median()), 1) == 1.0
    _q = _day.between_time("08:00", "14:00")["occupancy"].quantile([0.1, 0.9]).round(1)
    assert _q.tolist() == [6.7, 9.4]
    assert round(float(_day.between_time("12:00", "14:30")["speed"].median())) == 57
    assert (_cong.min().strftime("%H:%M"), _cong.max().strftime("%H:%M")) == ("14:40", "18:25")
    assert _day.index[_day["occupancy"] > 20].min().strftime("%H:%M") == "15:20"
    assert (round(float(_day["speed"].min())), _day["speed"].idxmin().strftime("%H:%M")) == (
        17,
        "17:30",
    )
    attempt.drop(columns="speed (mph)").T
    return (attempt,)


@app.cell
def _(attempt, mo):
    mo.accordion(
        {
            "Answer": mo.md(
                f"""
                Speeds: {", ".join(str(int(_v)) for _v in attempt["speed (mph)"])} mph. The
                freeway broke down between 14:30 and 14:45 and recovered by 18:35. Flow cannot
                tell: 1,620 vehicles per hour per lane at 14:30 (free) and at 15:35 (congested).
                Occupancy can: about 11% free, about 25% congested. Your rule is a threshold on
                occupancy; today we find it without the speeds.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with a **generative decoder** as this lecture's step: a loop reading →
    encoder (5-minute occupancy $x$) → decoder, Bayes' rule over a two-regime mixture
    $\pi_k\,\mathcal{N}(x;\mu_k,\sigma_k)$ → $P(\text{congested}\mid x)$ → meter on if above
    0.5. EM fits the mixture to unlabeled readings. §4 recomputes the slide's trace.

    ## 3. Derive

    **Assumption:** each reading comes from one of $K$ hidden regimes, each with its own density:

    $$p(x) = \sum_{k=1}^{K}\pi_k\,\mathcal{N}(x;\mu_k,\sigma_k^2), \qquad
      P(k\mid x) = \frac{\pi_k\,\mathcal{N}(x;\mu_k,\sigma_k^2)}{p(x)}.$$

    **Naive Bayes**, with labels and several inputs, assumes the inputs independent *within*
    each class: $P(k\mid\mathbf{x}) \propto \pi_k\prod_j p(x_j\mid k)$.

    **Kernel density estimate:** $\hat f_h(x) = \frac{1}{nh}\sum_i \phi\big(\frac{x-x_i}{h}\big)$;
    the bandwidth $h$ trades variance (small $h$) for bias (large $h$). Scott's rule as SciPy
    applies it (`bw_method="scott"`): $h = \hat\sigma\, n^{-1/5}$. Scott's normal-reference rule
    for a Gaussian kernel is $1.06\,\hat\sigma\, n^{-1/5}$, the bandwidth that minimizes the
    asymptotic mean integrated squared error if the data are normal (Scott, 1992); a normal
    reference is too wide for multimodal data.

    **EM** (Dempster, Laird & Rubin, 1977). With the regimes hidden, alternate:

    - E-step: responsibilities $r_{ik} = P(k \mid x_i)$ under the current parameters;
    - M-step: $\mu_k = \sum_i r_{ik}x_i / \sum_i r_{ik}$,
      $\sigma_k^2 = \sum_i r_{ik}(x_i-\mu_k)^2 / \sum_i r_{ik}$,
      $\pi_k = \frac1n\sum_i r_{ik}$.

    **Why the likelihood cannot fall.** For any distribution $q_i$ over regimes,
    $\ln p(x_i) \ge \sum_k q_{ik}\ln\frac{\pi_k\mathcal{N}(x_i;\mu_k,\sigma_k^2)}{q_{ik}}$ (Jensen),
    with equality when $q_{ik} = r_{ik}$. The E-step makes the bound tight; the M-step
    maximizes it; so the log-likelihood rises or stays. $k$-means is the limit of EM with equal,
    vanishing spreads.
    """)
    return


@app.cell
def _(logsumexp, norm, np):
    def em_gmm_1d(
        x: np.ndarray,
        mu: np.ndarray,
        sd: np.ndarray,
        pi: np.ndarray,
        tol: float = 1e-10,
        max_steps: int = 10_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """EM for a K-component Gaussian mixture in one dimension (the slide's listing).

        The E-step works in logs: ``logsumexp`` gives log p(x_i) without the underflow that
        summing the densities risks for a reading far from every component.

        Args:
            x: Data, shape ``(n,)``.
            mu: Initial means, shape ``(K,)``.
            sd: Initial standard deviations, shape ``(K,)``.
            pi: Initial weights, shape ``(K,)``.
            tol: Stop when the log-likelihood gains less than this.
            max_steps: Give up after this many steps.

        Returns:
            Means, standard deviations, weights, and the log-likelihood at each step.

        Raises:
            RuntimeError: If EM has not converged after ``max_steps``.
        """
        history = []
        for step in range(1, max_steps + 1):
            logd = np.log(pi) + norm.logpdf(x[:, None], mu, sd)  # n x K
            ll = logsumexp(logd, axis=1)  # log p(x_i)
            r = np.exp(logd - ll[:, None])  # E-step
            nk = r.sum(axis=0)  # M-step
            mu = (r * x[:, None]).sum(axis=0) / nk
            sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
            pi = nk / len(x)
            history.append(float(ll.sum()))
            if step > 1 and history[-1] - history[-2] < tol:
                return mu, sd, pi, history
        raise RuntimeError("EM did not converge")

    def posterior_high(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """P(the component with the larger mean | x), from log densities.

        Args:
            x: Occupancies, shape ``(n,)``.
            mu: Component means, shape ``(K,)``.
            sd: Component standard deviations, shape ``(K,)``.
            pi: Component weights, shape ``(K,)``.

        Returns:
            The posterior of the highest-mean component at each ``x``.
        """
        _logd = np.log(pi) + norm.logpdf(np.asarray(x, dtype=float)[:, None], mu, sd)
        return np.exp(_logd[:, int(np.argmax(mu))] - logsumexp(_logd, axis=1))

    def breakdown(mu: np.ndarray, sd: np.ndarray, pi: np.ndarray, level: float = 0.5) -> float:
        """The lowest occupancy above the free mean where the congested posterior reaches a level.

        Args:
            mu: Component means, shape ``(K,)``.
            sd: Component standard deviations, shape ``(K,)``.
            pi: Component weights, shape ``(K,)``.
            level: The posterior level; the breakdown occupancy at the default 0.5.

        Returns:
            The occupancy (%), on a 0.001-point grid up to 60%.

        Raises:
            ValueError: If the posterior never reaches ``level`` below 60% occupancy.
        """
        _g = np.linspace(0.0, 60.0, 60001)
        _p = posterior_high(_g, mu, sd, pi)
        _above = _g[(_p >= level) & (_g > np.min(mu))]
        if _above.size == 0:
            raise ValueError(f"the congested posterior never reaches {level} below 60%")
        return float(_above.min())

    return breakdown, em_gmm_1d, posterior_high


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: one EM step by hand

    Occupancies (%): 3, 4, 5, 20, 24, 28. Start: $\mu = (5, 20)$, $\sigma = (5, 5)$,
    $\pi = (0.5, 0.5)$. (1) E-step: the responsibility of the congested regime for $x = 5$ and
    $x = 20$. (2) M-step: the new means. (3) Where is the boundary after the step? The next
    cell checks your answers.
    """)
    return


@app.cell
def _(breakdown, norm, np):
    _x = np.array([3, 4, 5, 20, 24, 28.0])
    _mu, _sd, _pi = np.array([5.0, 20.0]), np.array([5.0, 5.0]), np.array([0.5, 0.5])
    _dens = _pi * norm.pdf(_x[:, None], _mu, _sd)
    _r = _dens / _dens.sum(axis=1, keepdims=True)
    _nk = _r.sum(axis=0)
    _mu1 = (_r * _x[:, None]).sum(axis=0) / _nk
    _sd1 = np.sqrt((_r * (_x[:, None] - _mu1) ** 2).sum(axis=0) / _nk)
    _pi1 = _nk / len(_x)
    # Slide notes: r(5) = 0.011, r(20) = 0.989; means 4.1 and 23.9; spreads 1.3 and 3.6; 9.6%.
    assert (round(float(_r[2, 1]), 3), round(float(_r[3, 1]), 3)) == (0.011, 0.989)
    assert np.round(_mu1, 1).tolist() == [4.1, 23.9] and np.round(_sd1, 1).tolist() == [1.3, 3.6]
    assert round(np.exp(15**2 / 50)) == 90 and round(breakdown(_mu1, _sd1, _pi1), 1) == 9.6
    print(
        f"r_congested = {np.round(_r[:, 1], 3)}; new means {np.round(_mu1, 2)}, spreads "
        f"{np.round(_sd1, 2)}; boundary {breakdown(_mu1, _sd1, _pi1):.1f}%"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Predict:** how many steps to converge at Glisan, and how often will EM's regime agree
    with the speed label (below 45 mph), which it never sees? The start: means at the 50th and
    98th percentiles of occupancy, both spreads equal to the overall SD, equal weights. The
    reference: scikit-learn's `GaussianMixture` from the same start, with `reg_covar=0`.
    """)
    return


@app.cell
def _(GaussianMixture, em_gmm_1d, glisan, np):
    x_g = glisan["occupancy"].to_numpy()
    _mu0 = np.quantile(x_g, [0.50, 0.98])
    _sd0, _pi0 = np.full(2, x_g.std()), np.full(2, 0.5)
    mu_g, sd_g, pi_g, _hist = em_gmm_1d(x_g, _mu0, _sd0, _pi0)
    _ref = GaussianMixture(
        2,
        covariance_type="full",
        weights_init=_pi0,
        means_init=_mu0[:, None],
        precisions_init=(1 / _sd0**2)[:, None, None],
        reg_covar=0.0,
        tol=1e-12,
        max_iter=100_000,
    ).fit(x_g[:, None])
    _o = np.argsort(_ref.means_.ravel())
    _diff = max(
        float(np.max(np.abs(np.sort(mu_g) - _ref.means_.ravel()[_o]))),
        float(np.max(np.abs(sd_g[np.argsort(mu_g)] - np.sqrt(_ref.covariances_.ravel()[_o])))),
        float(np.max(np.abs(pi_g[np.argsort(mu_g)] - _ref.weights_[_o]))),
    )
    # Slide "No labels, 99.7% agreement": 18 steps; 2e-6; -61,637 -> -48,881, never falling;
    # 4.4 +- 3.0% (90%) and 23.8 +- 4.8% (10%).
    assert len(_hist) == 18 and _diff < 1e-5 and np.all(np.diff(_hist) > -1e-9)
    assert (round(_hist[0]), round(_hist[-1])) == (-61637, -48881)
    assert np.round(np.sort(mu_g), 1).tolist() == [4.4, 23.8]
    assert np.round(sd_g[np.argsort(mu_g)], 1).tolist() == [3.0, 4.8]
    assert np.round(pi_g[np.argsort(mu_g)], 2).tolist() == [0.9, 0.1]
    print(f"EM: {len(_hist)} steps; scikit-learn agrees to {_diff:.0e}")
    return mu_g, pi_g, sd_g, x_g


@app.cell
def _(GaussianMixture, glisan, logsumexp, norm, np, x_g):
    # Slide "EM, step by step": the means after 0, 1, and 3 steps from the stated start; the
    # notes: the 17 steps after the first move the congested mean 0.4 more, the free mean 0.1.
    def _steps(x, mu, sd, pi, n):
        for _ in range(n):
            _d = np.log(pi) + norm.logpdf(x[:, None], mu, sd)
            _r = np.exp(_d - logsumexp(_d, axis=1, keepdims=True))
            _nk = _r.sum(axis=0)
            mu = (_r * x[:, None]).sum(axis=0) / _nk
            sd = np.sqrt((_r * (x[:, None] - mu) ** 2).sum(axis=0) / _nk)
            pi = _nk / len(x)
        return mu

    _mu0 = np.quantile(x_g, [0.50, 0.98])
    _sd0, _pi0 = np.full(2, x_g.std()), np.full(2, 0.5)
    _means = [np.round(_steps(x_g, _mu0, _sd0, _pi0, _n), 1).tolist() for _n in (0, 1, 3)]
    assert _means == [[4.9, 27.4], [4.5, 23.4], [4.4, 23.7]]
    _moved = np.abs(_steps(x_g, _mu0, _sd0, _pi0, 18) - _steps(x_g, _mu0, _sd0, _pi0, 1))
    assert np.round(_moved, 1).tolist() == [0.1, 0.4]

    # Slides "Two regimes, two hills" and "EM can stall on a plateau", on (occupancy, flow):
    # the best of 5 k-means starts at scikit-learn's default tolerance (1e-3), fitted once.
    X2d = glisan[["occupancy", "flow"]].to_numpy()
    good2d = GaussianMixture(
        2, covariance_type="full", init_params="kmeans", tol=1e-3, random_state=0, n_init=5
    ).fit(X2d)
    _o = np.argsort(good2d.means_[:, 0])
    assert np.round(good2d.means_[_o, 0], 1).tolist() == [4.3, 22.5]
    assert np.round(good2d.means_[_o, 1]).astype(int).tolist() == [802, 1434]
    assert np.round(good2d.weights_[_o], 2).tolist() == [0.89, 0.11]
    _stalled = GaussianMixture(2, random_state=3, init_params="random", tol=1e-3).fit(X2d)
    _ll = [round(float(_g.score(X2d) * len(X2d))) for _g in (_stalled, good2d)]
    assert _ll == [-180599, -153939]  # start 3 stalls; the best of 5 finds the regimes

    # Slide "Borrowed models misfire": free flow at Glisan, occupancy and flow correlate 0.97.
    _free = glisan[glisan["congested"] == 0]
    assert round(float(np.corrcoef(_free["occupancy"], _free["flow"])[0, 1]), 2) == 0.97
    _means, _ll
    return X2d, good2d


@app.cell
def _(
    CAPACITY_BIN_PCT,
    LogisticRegression,
    PI_SENSITIVITY,
    breakdown,
    glisan,
    mu_g,
    norm,
    np,
    pi_g,
    posterior_high,
    sd_g,
    x_g,
):
    _y = glisan["congested"].to_numpy()
    _agree = float(((posterior_high(x_g, mu_g, sd_g, pi_g) > 0.5) == (_y == 1)).mean())
    _lr = LogisticRegression(C=1e6, max_iter=5000).fit(x_g[:, None], _y)
    # 2-point occupancy bins, the highest occupancies in their own bins too.
    _bin_lo = np.floor(x_g / CAPACITY_BIN_PCT) * CAPACITY_BIN_PCT
    _q95 = glisan.groupby(_bin_lo)["flow"].quantile(0.95)
    _peak = float(_q95.idxmax())
    _em, _labels = breakdown(mu_g, sd_g, pi_g), float(-_lr.intercept_[0] / _lr.coef_[0, 0])
    # Slide: posterior 0.5 at 13.7%; labels 13.4%, 0.2 below; both in the flow peak's 12-14%
    # bin (1,810); 89.9% vs 99.7%.
    assert round(_em, 1) == 13.7 and round(_labels, 1) == 13.4 and round(_labels - _em, 1) == -0.2
    assert [_peak, _peak + CAPACITY_BIN_PCT] == [12.0, 14.0] and int(round(_q95.max(), -1)) == 1810
    assert _peak <= _em < _peak + CAPACITY_BIN_PCT and _peak <= _labels < _peak + CAPACITY_BIN_PCT
    # The decisive slide's notes: the crossing moves with the congested share (weights alone).
    _hi = int(np.argmax(mu_g))
    _at_pi = [
        breakdown(mu_g, sd_g, np.where(np.arange(2) == _hi, _s, 1 - _s)) for _s in PI_SENSITIVITY
    ]
    assert [round(_c, 1) for _c in _at_pi] == [14.2, 12.7], _at_pi
    assert (round(100 * (1 - _y.mean()), 1), round(100 * _agree, 1)) == (89.9, 99.7)
    # The diagram's trace: 14:45 on October 6.
    _r = glisan.set_index("time").loc["2011-10-06 14:45"]
    _f = pi_g * norm.pdf(float(_r["occupancy"]), mu_g, sd_g)
    _lo, _hi = int(np.argmin(mu_g)), int(np.argmax(mu_g))
    assert (f"{_f[_lo]:.1e}", f"{_f[_hi]:.1e}", round(float(_f[_hi] / _f.sum()), 2)) == (
        "1.8e-05",
        "2.9e-03",
        0.99,
    )
    print(
        f"breakdown occupancy {breakdown(mu_g, sd_g, pi_g):.1f}% (labels "
        f"{-_lr.intercept_[0] / _lr.coef_[0, 0]:.1f}%); agreement {100 * _agree:.1f}%"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **Bandwidth.** A KDE of Glisan's speeds with $h = 0.2$, 1.4, and 3.9 mph. How many peaks?
    2. **Starts.** EM on (occupancy, flow) from 20 random starts, at scikit-learn's default
       tolerance. Do all find the regimes?
    3. **How many regimes?** BIC for $K = 1$ to 10. Which $K$ wins?
    4. **Another station.** Glisan's model at Stark/Washington. Better or worse with flow added
       to occupancy?
    """)
    return


@app.cell
def _(PALETTE, gaussian_kde, glisan, np, plt):
    _sp = glisan["speed"].to_numpy()
    _grid = np.linspace(5, 75, 1401)
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _res = {}
    for _i, (_bw, _name) in enumerate(((0.02, "narrow"), ("scott", "Scott"), (0.4, "wide"))):
        _k = gaussian_kde(_sp, bw_method=_bw)
        _f = _k(_grid)
        _peaks = _grid[1:-1][(_f[1:-1] > _f[:-2]) & (_f[1:-1] > _f[2:])]
        _valleys = _grid[1:-1][(_f[1:-1] < _f[:-2]) & (_f[1:-1] < _f[2:])]
        _h = float(_k.factor * _sp.std(ddof=1))
        _res[_name] = (
            round(_h, 1),
            len(_peaks),
            np.round(_peaks, 0).tolist()[:3],
            np.round(_valleys, 1).tolist(),
        )
        _ax.plot(_grid, _f, color=PALETTE[_i], label=f"{_name}, h = {_h:.1f} mph")
    # Slide "Bandwidth decides the peaks": 42 peaks; 3 peaks (25, 45, 58); 2 peaks, valley 41.5.
    assert _res["narrow"][:2] == (0.2, 42)
    assert _res["Scott"][:3] == (1.4, 3, [25.0, 45.0, 58.0])
    assert _res["wide"][:3] == (3.9, 2, [25.0, 58.0]) and _res["wide"][3] == [41.5]
    # Notes: the 45 mph bump is records in transition. A record is in transition when the
    # record 5 minutes before or after (if there is one) is on the other side of 45 mph.
    _lab = glisan.set_index("time")["congested"]
    assert _lab.index.is_unique
    _prev = _lab.shift(1, freq="5min").reindex(_lab.index)
    _next = _lab.shift(-1, freq="5min").reindex(_lab.index)
    _trans = ((_prev.notna() & (_prev != _lab)) | (_next.notna() & (_next != _lab))).to_numpy()
    _band = (_sp >= 43) & (_sp <= 48)
    _bump = (
        int(_band.sum()),
        round(100 * float(_trans[_band].mean()), 1),
        round(100 * float(_trans.mean()), 1),
    )
    assert _bump == (84, 72.6, 1.6), _bump
    _ax.set_xlabel("5-minute speed (mph)")
    _ax.legend(fontsize=9)
    _fig
    return


@app.cell
def _(GaussianMixture, KS, X2d, breakdown, glisan, good2d, mu_g, np, pi_g, sd_g):
    # 20 starts from random responsibilities (init_params="random", not the k-means default),
    # at the default tolerance and at a tight one; and 20 k-means starts for comparison.
    _ll, _ll_tight, _ll_km, _means = [], [], [], []
    for _seed in range(20):
        _g = GaussianMixture(2, random_state=_seed, init_params="random", tol=1e-3).fit(X2d)
        _ll.append(float(_g.score(X2d) * len(X2d)))
        _means.append(sorted(float(_m) for _m in _g.means_[:, 0]))
        _gt = GaussianMixture(
            2, random_state=_seed, init_params="random", max_iter=5000, tol=1e-8
        ).fit(X2d)
        _ll_tight.append(float(_gt.score(X2d) * len(X2d)))
        _gk = GaussianMixture(2, random_state=_seed, init_params="kmeans", tol=1e-3).fit(X2d)
        _ll_km.append(float(_gk.score(X2d) * len(X2d)))
    _best = max(_ll_tight)
    _bad = [_i for _i, _v in enumerate(_ll) if _v < _best - 1000]
    _bic, _occ = [], []
    for _k in KS:
        _gk = GaussianMixture(_k, random_state=0, n_init=5, max_iter=2000).fit(X2d)
        _bic.append(round(float(_gk.bic(X2d))))
        _occ.append(np.sort(_gk.means_[:, 0]))
    _hi = int(np.argmax(good2d.means_[:, 0]))
    _agree = float(((good2d.predict(X2d) == _hi) == (glisan["congested"].to_numpy() == 1)).mean())
    # Slide "EM can stall on a plateau": 5 of 20 stall with both means near 6% occupancy (all
    # five, both means, 6.1 to 6.6%); -180,606 to -180,069 against -153,934; none stalls at
    # the tight tolerance or from k-means; the best fit agrees 99.2%.
    assert len(_bad) == 5 and sum(_v < _best - 1000 for _v in _ll_tight) == 0
    assert sum(_v < _best - 1000 for _v in _ll_km) == 0
    assert all(6.0 < _m < 6.7 for _i in _bad for _m in _means[_i])
    _bad_ll = [round(_ll[_i]) for _i in _bad]
    assert (min(_bad_ll), max(_bad_ll), round(_best)) == (-180606, -180069, -153934)
    assert round(100 * _agree, 1) == 99.2
    # Slide "Components are not regimes": BIC is lowest at K = 10, the largest tried (it rises
    # once, at K = 9). Components above Glisan's breakdown: one for K = 2 to 6 (mean 22.5% to
    # 23.3%; at K = 6 the free ones run 0.7% to 7.5%), two at K = 7, three from K = 8.
    assert _bic == [
        361261, 307985, 293747, 290160, 289237, 288961, 288616, 288532, 288617, 288380
    ]  # fmt: skip
    assert KS[int(np.argmin(_bic))] == KS[-1]
    _thr = breakdown(mu_g, sd_g, pi_g)
    _n_cong = [int((_m > _thr).sum()) for _m in _occ]
    assert _n_cong == [0, 1, 1, 1, 1, 1, 2, 3, 3, 3], _n_cong
    assert np.round(_occ[KS.index(6)], 1).tolist() == [0.7, 1.7, 3.7, 6.3, 7.5, 23.3]
    _top = [round(float(_occ[KS.index(_k)][-1]), 1) for _k in range(2, 7)]
    assert (min(_top), max(_top)) == (22.5, 23.3)
    print(
        f"default tolerance: {len(_bad)} of 20 random starts stalled; BIC by K {_bic}; "
        f"components above {_thr:.1f}% by K: {_n_cong}"
    )
    return


@app.cell
def _(
    GaussianNB,
    LogisticRegression,
    PI_SENSITIVITY,
    QuadraticDiscriminantAnalysis,
    UNSURE_BAND,
    breakdown,
    em_gmm_1d,
    glisan,
    mu_g,
    np,
    pi_g,
    sd_g,
    stark,
    warnings,
):
    # Stark/Washington's own EM threshold, from its unlabeled occupancy (fitted once here).
    _xs = stark["occupancy"].to_numpy()
    mu_s, sd_s, pi_s, _ = em_gmm_1d(
        _xs, np.quantile(_xs, [0.5, 0.98]), np.full(2, _xs.std()), np.full(2, 0.5)
    )
    own = breakdown(mu_s, sd_s, pi_s)
    _y = stark["congested"].to_numpy()
    _lr_s = LogisticRegression(C=1e6, max_iter=5000).fit(_xs[:, None], _y)
    _labels_s = float(-_lr_s.intercept_[0] / _lr_s.coef_[0, 0])
    # Two models of p(x | class) (naive Bayes; a full-covariance Gaussian per class, QDA) and
    # one of P(class | x) (logistic regression, scikit-learn's default ridge penalty C = 1, set
    # explicitly), each on occupancy and on occupancy and flow. With one input, QDA is naive
    # Bayes. Per model: false alarms, missed, accuracy at home (the training records) and at
    # Stark, and the share of posteriors between 0.1 and 0.9 at home and at Stark.
    _y_a = glisan["congested"].to_numpy()
    _res = {}
    for _name, _cols, _model in (
        ("NB occupancy", ["occupancy"], GaussianNB()),
        ("NB occupancy + flow", ["occupancy", "flow"], GaussianNB()),
        (
            "full occupancy + flow",
            ["occupancy", "flow"],
            QuadraticDiscriminantAnalysis(reg_param=0.0),
        ),
        ("LR occupancy", ["occupancy"], LogisticRegression(C=1.0, max_iter=5000)),
        ("LR occupancy + flow", ["occupancy", "flow"], LogisticRegression(C=1.0, max_iter=5000)),
    ):
        _model.fit(glisan[_cols].to_numpy(), _y_a)
        _p = _model.predict_proba(stark[_cols].to_numpy())[:, 1]
        _pa = _model.predict_proba(glisan[_cols].to_numpy())[:, 1]
        _res[_name] = (
            int(((_p > 0.5) & (_y == 0)).sum()),
            int(((_p <= 0.5) & (_y == 1)).sum()),
            round(100 * float(((_pa > 0.5) == (_y_a == 1)).mean()), 1),
            round(100 * float(((_p > 0.5) == (_y == 1)).mean()), 1),
            round(100 * float(((_pa > UNSURE_BAND[0]) & (_pa < UNSURE_BAND[1])).mean()), 1),
            round(100 * float(((_p > UNSURE_BAND[0]) & (_p < UNSURE_BAND[1])).mean()), 1),
        )
    # Each threshold is the unrounded EM crossing (13.656% at Glisan, 15.641% at Stark); the
    # slides show them to one decimal.
    _glisan_thr = breakdown(mu_g, sd_g, pi_g)
    _flag, _flag_own = _xs >= _glisan_thr, _xs >= own
    _borrowed = int((_flag & (_y == 0)).sum())
    _free = glisan[glisan["congested"] == 0]
    # Slides "Each station breaks at its own occupancy" and "Borrowed models misfire": Glisan's
    # threshold at Stark, 738 false alarms and 67 missed; Stark's own, 301 and 161.
    assert (
        round(own, 1) == 15.6 and round(float(-_lr_s.intercept_[0] / _lr_s.coef_[0, 0]), 1) == 17.7
    )
    assert int((_y == 0).sum()) == 14913 and (_borrowed, int((~_flag & (_y == 1)).sum())) == (
        738,
        67,
    )
    assert (int((_flag_own & (_y == 0)).sum()), int((~_flag_own & (_y == 1)).sum())) == (301, 161)
    # The vote's notes: 805 errors against 462; 4.1 and 2.1 points from the labels' 17.7%;
    # option C, Stark's own congested mean, 20.5%.
    assert _borrowed + int((~_flag & (_y == 1)).sum()) == 805
    assert int((_flag_own & (_y == 0)).sum() + (~_flag_own & (_y == 1)).sum()) == 462
    assert (round(_labels_s - _glisan_thr, 1), round(_labels_s - own, 1)) == (4.1, 2.1)
    assert round(float(np.max(mu_s)), 1) == 20.5
    assert _res == {
        "NB occupancy": (754, 65, 99.7, 95.2, 0.6, 7.8),
        "NB occupancy + flow": (1237, 72, 99.7, 92.4, 0.8, 16.3),
        "full occupancy + flow": (2675, 8, 99.6, 84.4, 0.4, 8.2),
        "LR occupancy": (829, 62, 99.7, 94.8, 0.7, 9.5),
        "LR occupancy + flow": (613, 15, 99.8, 96.3, 0.5, 8.0),
    }, _res
    # The decisive slide's notes: Stark's crossing with a congested share of 5% and of 30%.
    _hi = int(np.argmax(mu_s))
    _at_pi = [
        breakdown(mu_s, sd_s, np.where(np.arange(2) == _hi, _s, 1 - _s)) for _s in PI_SENSITIVITY
    ]
    assert [round(_c, 1) for _c in _at_pi] == [18.0, 14.5], _at_pi
    assert round(float(np.corrcoef(_free["occupancy"], _free["flow"])[0, 1]), 2) == 0.97
    # Notes: without the penalty the count barely moves (617 against 613). C = inf is
    # scikit-learn 1.8's spelling of no penalty (penalty=None is deprecated); it warns that C is
    # then ignored, which is the intent.
    _cols2 = ["occupancy", "flow"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Setting penalty=None will ignore", UserWarning)
        _lr_none = LogisticRegression(C=np.inf, max_iter=5000).fit(
            glisan[_cols2].to_numpy(), glisan["congested"]
        )
    assert (
        int(((_lr_none.predict_proba(stark[_cols2].to_numpy())[:, 1] > 0.5) & (_y == 0)).sum())
        == 617
    )
    print(
        f"Stark/Washington's own EM threshold {own:.1f}%; false alarms: Glisan's threshold "
        f"{_borrowed}; per model (false alarms, missed, accuracy at home and at Stark, unsure "
        f"at home and at Stark): {_res}"
    )
    return mu_s, own, pi_s, sd_s


@app.cell
def _(
    MARKERS,
    PALETTE,
    breakdown,
    glisan,
    mu_g,
    mu_s,
    np,
    pi_g,
    pi_s,
    plt,
    posterior_high,
    sd_g,
    sd_s,
    stark,
):
    # Notes of "Set each meter from its own data": the posterior goes from 0.1 to 0.9 within
    # 2.9 points of occupancy at Glisan and within 8.1 at Stark/Washington.
    _width = [
        breakdown(_m, _s, _p, 0.9) - breakdown(_m, _s, _p, 0.1)
        for _m, _s, _p in ((mu_g, sd_g, pi_g), (mu_s, sd_s, pi_s))
    ]
    assert [round(_w, 1) for _w in _width] == [2.9, 8.1], _width
    # The decisive figure's alt text: the median flow of EM's congested records at 15-20% and
    # at 35-40% occupancy.
    _medians = []
    for _d, _m, _s, _p in ((glisan, mu_g, sd_g, pi_g), (stark, mu_s, sd_s, pi_s)):
        _occ, _flow = _d["occupancy"].to_numpy(), _d["flow"].to_numpy()
        _cong = posterior_high(_occ, _m, _s, _p) > 0.5
        _medians.append(
            [
                int(round(float(np.median(_flow[_cong & (_occ > _a) & (_occ <= _b)])), -1))
                for _a, _b in ((15, 20), (35, 40))
            ]
        )
    assert _medians == [[1540, 1010], [1740, 1600]], _medians
    _grid = np.linspace(0, 40, 801)
    _fig, _ax = plt.subplots(figsize=(5.5, 3.5))
    for _i, (_name, _m, _s, _p) in enumerate(
        (("Glisan", mu_g, sd_g, pi_g), ("Stark/Washington", mu_s, sd_s, pi_s))
    ):
        _ax.plot(
            _grid,
            posterior_high(_grid, _m, _s, _p),
            color=PALETTE[_i],
            marker=MARKERS[_i],
            markevery=80,
            label=f"{_name}: {breakdown(_m, _s, _p):.1f}%",
        )
    _ax.axhline(0.5, color="grey", lw=1)
    _ax.set_xlabel("occupancy (%)")
    _ax.set_ylabel("P(congested | occupancy)")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo, mu_s, np, own):
    mo.md(rf"""
    **Each detector breaks down at its own occupancy:** 13.7% at Glisan, {own:.1f}% at
    Stark/Washington, 0.8 miles downstream, where flow peaks at 16–18%. The crossing also moves
    with the share of congested records: at Glisan from 14.2% to 12.7% as that share goes from
    5% to 30%.

    **Borrowed models misfire, and flow makes the generative ones worse.** Trained at Glisan
    and applied at Stark/Washington, adding flow to occupancy raised naive Bayes' false alarms
    from 754 to 1,237, and those of a full-covariance Gaussian per class, which models the
    0.97 correlation of occupancy and flow instead of assuming it away, to 2,675; logistic
    regression's fell from 829 to 613. So the independence assumption is not what fails here:
    with flow, both models of $p(\mathbf{{x}}\mid k)$ got worse and the model of
    $P(k\mid\mathbf{{x}})$ got better. At home every model is right on 99.6 to 99.8% of the
    records it was trained on.

    ## 6. Decide

    Fit EM to each station's own unlabeled occupancy (several starts), set the meter's target
    just below the posterior-0.5 point, check it against the flow peak and, where dual loops
    exist, against speed, and refit after changes to lanes, ramps, or detectors.

    **When not to use it:** across stations without refitting (805 errors with Glisan's
    threshold at Stark/Washington, 462 with its own); in incidents or weather, which these data
    cannot test (no incident or weather labels); with faulty loops (check the status flags);
    a borrowed density with extra inputs.

    **Impact:** a meter trades delay on the ramp for flow on the freeway. In fall 2000
    Minnesota switched off all 430 Twin Cities ramp meters for a study ordered by the
    legislature (Cambridge Systematics, 2001); the evaluation found metering cost-effective, and
    the meters came back on.

    ### Vote: Stark/Washington's target

    **A.** 13.7%, Glisan's. **B.** {own:.1f}%, EM on its own occupancy. **C.**
    {np.max(mu_s):.1f}%, the mean of its congested regime. **D.** 0.5 from naive Bayes with
    occupancy and flow, trained at Glisan.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                B: the station's own unlabeled data, near its flow peak: 462 errors (301 false
                alarms, 161 missed), 2.1 points from the labels' 17.7%. A: 805 errors (738
                false alarms), 4.1 points from the labels. C is deep in congestion, too late
                for a meter; D raised 1,237 false alarms. What would change the answer: a
                short dual-loop survey (labels say 17.7%), or a lane drop downstream.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where mixtures live today

    - **Mixture of experts:** large language models route each token to a few experts with a
      learned softmax gate, the role of the responsibilities (Shazeer et al., 2017; Jiang et
      al., 2024). No later lecture covers it; Lecture 17's transformers are the networks
      Mixtral's experts sit in.
    - **Variational autoencoders and diffusion models** maximize a lower bound on the
      likelihood, EM's bound with a network inside (Kingma & Welling, 2014; Ho et al., 2020).
    - **Naive Bayes** remains a strong baseline for short texts (Wang & Manning, 2012); no
      later lecture covers it.
    - **Densities:** a ratio of two densities reweights calibration scores for covariate shift
      (Lecture 11).

    ## 7. Check yourself

    **Exit question.** A new station has single loops only: volume and occupancy, no speed. How
    would you set its meter target, and how would you check it without speeds?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                Fit a two-regime mixture to its own occupancy (several starts), take the
                posterior-0.5 point, check it against the peak of the flow-occupancy diagram,
                and refit after changes; a short probe-speed survey can label a sample.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **EM from one start, with the library's default tolerance.** 5 of 20 starts from random
      responsibilities stalled (none of 20 from k-means, the library's default start).
    - **Choosing $K$ by BIC alone** when the question has two regimes: BIC was lowest at the
      largest $K$ tried, 10.
    - **A KDE bandwidth left at the default** for multimodal data; check the peaks it makes.
    - **Adding an input to a borrowed generative model.** Flow raised the false alarms of naive
      Bayes and of a full-covariance model at another station; it lowered logistic
      regression's.
    - **One threshold for every detector.** Fit per station.
    - **UTC timestamps.** The raw data are in UTC; the derived file is in Pacific time.
    - **Missing lane readings.** Keep records with most of their 45 readings, and scale up the
      volume of a record with missing ones (it sums only the readings present); check status
      flags.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth Ch. 9:** clustering with probability models: the mixture of normals, EM derived
      as a lower bound, and its use for missing data. Forsyth writes $\delta_{ij}$ for our
      responsibilities' hidden indicators.
    - **Forsyth §1.3:** naive Bayes, and why it can classify well even when its probabilities are
      poor.
    - **Exercise.** Show that as both spreads shrink to zero with equal weights, the E-step
      becomes $k$-means' assignment step (Lecture 4).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A mixture models how readings are made; Bayes' rule turns it into a posterior.
    - EM: responsibilities, then weighted fits; the likelihood never falls. Use several starts.
    - A KDE's bandwidth is Lecture 3's trade-off; BIC's best $K$ is not the number of regimes.
    - Away from home, adding flow hurt both generative models and helped logistic regression.
    - Each detector breaks down at its own occupancy: fit it there.

    **Next:** decision trees, forests, and boosting: pavement deterioration.
    """)
    return


if __name__ == "__main__":
    app.run()
