"""Applied Machine Learning for CEE — Lecture 8: Trees, Forests, and Their Explanations.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l08_trees_explanations.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: run ``bash scripts/fetch_ltpp.sh`` once (FHWA LTPP Standard Data Release 40, public).
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
    # Applied Machine Learning for CEE — Lecture 8: Trees, Forests, and Their Explanations

    **Slides:** *Applied Machine Learning for CEE*, Lecture 8 (`lectures/cee_l08_trees_explanations`, PowerPoint).
    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).
    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §2.2 (decision trees and random forests) and Ch. 12 (boosting).
    **Data:** FHWA Long-Term Pavement Performance (LTPP) InfoPave, Standard Data Release 40 (public); fetch and derive with `bash scripts/fetch_ltpp.sh`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the engineering
    question → one picture → derive → build and verify → break it → decide → check yourself**.
    The slides' data numbers are recomputed here and checked with an `assert` against the figure
    script's values; only the spread over other assignments of sites and states to the folds is
    left to the figure script. It takes about 17 minutes on 4 CPU cores, most of it
    tuning: every outer fold picks the forest's leaf size and boosting's rounds by an inner
    cross-validation, a 500-tree forest at each setting tried.

    **Your tasks** (from the slides): write CART's best split and check it against
    scikit-learn, compare three ways of holding out sections, measure importance on training
    and held-out states, compute exact Shapley values, and estimate the effect of asphalt
    thickness within SPS-1 sites.
    """)
    return


@app.cell
def _():
    import itertools
    import math
    import os
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import t as student_t
    from sklearn.base import clone
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.tree import DecisionTreeRegressor

    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    FOLDS = 5
    N_JOBS = 4  # worker processes for the forests (the figure script's cap)
    # Tuning (the figure script's ladders): each runs from the model's no-fit limit
    # (index 0: no boosting rounds, i.e. the training mean; leaves of at least 160 sections,
    # which cannot split a training set of at most 288) to a setting far past any useful one.
    BOOST_LADDER = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
    BOOST_START = (10, 20, 50, 100, 200)
    LEAF_LADDER = (160, 80, 40, 20, 10, 5, 3, 2, 1)
    LEAF_START = (40, 20, 10, 5, 3)
    INNER_FOLDS = 5
    FEATURES = [
        "ac_in",
        "granular_base_in",
        "treated_base_in",
        "esal_per_yr",
        "freeze_index",
        "precip_mm",
        "mean_temp_c",
        "iri_first",
        "age_first",
    ]
    NAMES = {
        "ac_in": "asphalt thickness",
        "granular_base_in": "granular base",
        "treated_base_in": "treated base",
        "esal_per_yr": "trucks (ESAL/yr)",
        "freeze_index": "freezing index",
        "precip_mm": "precipitation",
        "mean_temp_c": "mean temperature",
        "iri_first": "first IRI",
        "age_first": "age at first visit",
    }
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    MARKERS = ["o", "s", "^", "D"]
    return (
        BOOST_LADDER,
        BOOST_START,
        DecisionTreeRegressor,
        DummyRegressor,
        FEATURES,
        FOLDS,
        INNER_FOLDS,
        LEAF_LADDER,
        LEAF_START,
        N_JOBS,
        GroupKFold,
        HistGradientBoostingRegressor,
        KFold,
        LinearRegression,
        MARKERS,
        NAMES,
        PALETTE,
        Path,
        RandomForestRegressor,
        SEED,
        clone,
        itertools,
        math,
        np,
        pd,
        permutation_importance,
        plt,
        student_t,
    )


@app.cell
def _(
    DecisionTreeRegressor,
    DummyRegressor,
    HistGradientBoostingRegressor,
    N_JOBS,
    RandomForestRegressor,
    SEED,
):
    def forest(min_leaf: int = 3) -> RandomForestRegressor:
        """A random forest: 500 bootstrap trees, 3 of the 9 inputs tried at each split.

        Args:
            min_leaf: The fewest sections in a leaf; the lecture's forest uses 3.

        Returns:
            The unfitted forest.
        """
        return RandomForestRegressor(
            n_estimators=500,
            max_features=3,
            bootstrap=True,
            min_samples_leaf=min_leaf,
            random_state=SEED,
            n_jobs=N_JOBS,
        )

    def boosting(rounds: int = 200) -> HistGradientBoostingRegressor | DummyRegressor:
        """Gradient boosting: shrinkage 0.03, trees of depth 3, at least 20 sections a leaf.

        Args:
            rounds: Boosting rounds; 0 is the model before any tree, the training mean.

        Returns:
            The unfitted model.
        """
        if rounds == 0:
            return DummyRegressor(strategy="mean")
        return HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=rounds,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping=False,  # "auto" is off anyway below 10,000 rows (at most 288 here)
            random_state=SEED,
        )

    def small_tree() -> DecisionTreeRegressor:
        """The depth-3 tree: at least 5 sections in every leaf.

        Returns:
            The unfitted tree.
        """
        return DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED)

    def full_tree() -> DecisionTreeRegressor:
        """A tree grown until every leaf is pure or holds one section.

        Returns:
            The unfitted tree.
        """
        return DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=SEED)

    return boosting, forest, full_tree, small_tree


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 3–7

    On your own, no notes, two minutes:

    1. Random folds picked $k = 2$ for bridge ratings; the 2024 inventory disagreed. What did the
       random folds reuse?
    2. In 2024 the engineer's rule "flag every 5" tied the bridge model. What must any model beat?
    3. BIC was lowest at $K = 10$ mixture components, the largest tried; the meter needed two
       regimes. What decided $K$?

    **Today:** what makes a pavement get rough fast, and can a model trained on the national
    test sections tell a state DOT?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                r"""
                1. Near-copies of each held-out record: the bridge's other inspections, its twin,
                   its neighbors (Lecture 3). Today: sections at the same site.
                2. The simplest credible baseline: then the engineer's rule; today, the mean
                   (Lecture 6).
                3. The decision, not the fit (Lecture 7): BIC was lowest at the largest $K$
                   tried, ten.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The engineering question

    Roughness (the International Roughness Index, IRI, in m/km) is how drivers feel a road;
    federal measures call an interstate good below 1.5 m/km and poor above 2.7 (23 CFR 490).
    **The decision:** which designs to build and which roads to resurface first, which needs
    what drives the **rate** of roughening. The data: LTPP's asphalt test sections, GPS-1 and
    GPS-2 (in service when LTPP began) and SPS-1 (built to a factorial of designs), original
    construction only, with at least three roughness visits over at least four years.
    """)
    return


@app.cell
def _(FEATURES, Path, pd):
    def find_csv() -> Path:
        """Locate ``data/derived/ltpp_iri_sections.csv`` from the repo root, notebooks/, or ipynb/.

        Raises:
            FileNotFoundError: If missing (run ``bash scripts/fetch_ltpp.sh``).
        """
        rel = Path("data") / "derived" / "ltpp_iri_sections.csv"
        for base in (Path.cwd(), Path.cwd().parent):
            if (base / rel).exists():
                return base / rel
        raise FileNotFoundError(f"{rel} not found; run `bash scripts/fetch_ltpp.sh`")

    d = pd.read_csv(find_csv(), dtype={"SHRP_ID": str, "EXPERIMENT_NO": str})
    X = d[FEATURES].to_numpy()
    y = d["iri_rate"].to_numpy()
    _by = d.groupby(["GPS_SPS", "EXPERIMENT_NO"]).size().to_dict()
    # Question slide: 360 sections, 53 states and provinces, 203 sites, 3,124 visits; median
    # rate 0.026 m/km/yr.
    assert (len(d), d["STATE_CODE"].nunique(), d["site"].nunique(), int(d["visits"].sum())) == (
        360,
        53,
        203,
        3124,
    )
    assert _by == {("G", "1"): 110, ("G", "2"): 77, ("S", "1"): 173}
    assert round(float(d["iri_rate"].median()), 3) == 0.026
    d[["iri_rate", *FEATURES]].describe().round(3).T
    return X, d, y


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: what slows the roughening?

    Eight sections of one SPS-1 project in Kansas: same site, same traffic, same climate,
    different designs. **What slows the roughening here? Write the rule you would take to the
    DOT.**
    """)
    return


@app.cell
def _(d):
    _s = d[d["site"] == "20-01"].sort_values(["ac_in", "SHRP_ID"])
    _t = _s[["SHRP_ID", "ac_in", "treated_base_in", "granular_base_in", "iri_rate"]].round(3)
    assert _t["SHRP_ID"].tolist() == [
        "0103",
        "0105",
        "0111",
        "0112",
        "0104",
        "0110",
        "0109",
        "0159",
    ]
    assert _t["iri_rate"].tolist() == [0.174, 0.263, 0.024, 0.004, 0.044, 0.046, 0.083, 0.083]
    assert round(float(_s["freeze_index"].iloc[0])) == 211
    _t.set_index("SHRP_ID").T
    return


@app.cell
def _(d, np, pd, small_tree):
    # Slide "Trees split until leaves agree": a depth-3 tree on two inputs, freezing index and
    # first IRI, cuts the plane into boxes; each predicts its mean rate. The table: each box's
    # mean rate, sections, and states, against the figure script's numbers.json.
    _X, _y = d[["freeze_index", "iri_first"]].to_numpy(), d["iri_rate"].to_numpy()
    _tree = small_tree().fit(_X, _y)
    _leaf = _tree.apply(_X)
    _boxes = pd.DataFrame(
        [
            {
                "mean": round(float(_y[_leaf == _n].mean()), 3),
                "sections": int((_leaf == _n).sum()),
                "states": int(d.loc[_leaf == _n, "STATE_CODE"].nunique()),
            }
            for _n in np.unique(_leaf)
        ]
    ).sort_values("mean")
    _t = _tree.tree_
    _splits = sorted(
        round(float(_t.threshold[_k]), 1) for _k in range(_t.node_count) if _t.feature[_k] == 0
    )
    assert int(_t.feature[0]) == 0 and round(float(_t.threshold[0])) == 1088
    assert (_boxes.to_dict("records"), _splits) == (
        [
            {"mean": 0.019, "sections": 78, "states": 7},
            {"mean": 0.033, "sections": 17, "states": 11},
            {"mean": 0.041, "sections": 170, "states": 35},
            {"mean": 0.064, "sections": 85, "states": 31},
            {"mean": 0.091, "sections": 5, "states": 4},
            {"mean": 0.196, "sections": 5, "states": 2},
        ],
        [9.1, 1087.6, 1451.4],
    )
    _boxes
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                The two fastest (0103, 0105) have thin asphalt on a thin base; thin asphalt on a
                thick treated base (0111, 0112) stays smooth; thick asphalt alone does not
                guarantee it (0109, 0159). Within one site, structure matters, and both layers
                count. The rest of the notebook asks whether a model trained on all 360 sections
                finds this.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture

    The course picture with a **forest** as the decoder: section → encoder (9 inputs) →
    $\mathbf{z}$ → the mean of 500 regression trees, each grown greedily on a bootstrap sample,
    trying 3 random inputs at each split →
    $\hat y$, the IRI growth rate; squared-error loss. A Shapley explanation splits one
    prediction among the inputs; §4 recomputes the slide's trace of Kansas section 0105.

    ## 3. Derive

    **Assumption:** the rate is roughly constant within boxes of input space, found by splits.

    **CART** (Breiman et al., 1984). At a node with cases $N$, choose the input $j$ and threshold
    $t$ that minimize $\sum_{i\in L}(y_i - \bar y_L)^2 + \sum_{i\in R}(y_i - \bar y_R)^2$, with
    $L = \{x_j \le t\}$. With running sums, $\sum_{i\in L}(y_i-\bar y_L)^2 = \sum_L y_i^2 -
    (\sum_L y_i)^2/|L|$, so every threshold of a sorted input costs $O(1)$. Splits depend only on
    the order of values: no scaling needed.

    **Bagging and forests** (Breiman, 2001). For $B$ trees with variance $\sigma^2$ and pairwise
    correlation $\rho$, $\mathrm{Var}\big(\frac1B\sum_b T_b\big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$:
    averaging removes the second term; random input subsets at each split lower $\rho$. Our
    forest tries 3 of the 9 inputs at each split ($p/3$, Breiman's choice for regression);
    scikit-learn's regressor tries all of them by default, which is bagging.

    **Gradient boosting** (Friedman, 2001). $F_m = F_{m-1} + \nu h_m$, with each small tree $h_m$
    fit to the negative gradient of the loss, for squared error the residuals $y - F_{m-1}$.

    **Explanations.** Permutation importance: the drop in $R^2$ when one input is shuffled.
    Partial dependence: the average prediction as one input varies. Shapley values (Shapley,
    1953; Lundberg & Lee, 2017): with $v(S)$ the average prediction when the inputs in $S$ take
    the section's values and the rest come from background sections,

    $$\phi_j = \sum_{S \subseteq \{1..d\}\setminus j} \frac{|S|!\,(d-|S|-1)!}{d!}\,\big[v(S\cup j) - v(S)\big],
      \qquad v(\varnothing) + \sum_j \phi_j = f(\mathbf{x}).$$
    """)
    return


@app.cell
def _(itertools, math, np):
    def best_split(X_: np.ndarray, y_: np.ndarray, min_leaf: int = 1) -> tuple[int, float, float]:
        """The split that most reduces the sum of squared errors (the slide's listing).

        Returns:
            Feature index, threshold, and the reduction in SSE (feature -1 if no split helps).
        """
        best, n = (-1, 0.0, 0.0), len(y_)
        sse0 = ((y_ - y_.mean()) ** 2).sum()
        for j in range(X_.shape[1]):
            o = np.argsort(X_[:, j], kind="stable")
            xs, s, s2 = X_[o, j], np.cumsum(y_[o]), np.cumsum(y_[o] ** 2)
            for i in range(min_leaf, n - min_leaf + 1):
                if xs[i - 1] == xs[i]:
                    continue  # no threshold between equals
                sse_l = s2[i - 1] - s[i - 1] ** 2 / i
                sse_r = s2[-1] - s2[i - 1] - (s[-1] - s[i - 1]) ** 2 / (n - i)
                gain = sse0 - sse_l - sse_r
                if gain > best[2] + 1e-12:
                    best = (j, (xs[i - 1] + xs[i]) / 2, gain)
        return best

    def grow(X_: np.ndarray, y_: np.ndarray, depth: int, min_leaf: int = 1) -> dict:
        """A regression tree grown greedily to ``depth``."""
        j, t, gain = best_split(X_, y_, min_leaf)
        if depth == 0 or j < 0 or gain <= 0:
            return {"leaf": float(y_.mean()), "n": len(y_)}
        left = X_[:, j] <= t
        return {
            "feature": j,
            "threshold": t,
            "n": len(y_),
            "left": grow(X_[left], y_[left], depth - 1, min_leaf),
            "right": grow(X_[~left], y_[~left], depth - 1, min_leaf),
        }

    def predict_tree(node: dict, X_: np.ndarray) -> np.ndarray:
        """Predictions of a tree from :func:`grow`."""
        if "leaf" in node:
            return np.full(len(X_), node["leaf"], dtype=float)
        go = X_[:, node["feature"]] <= node["threshold"]
        out = np.empty(len(X_))
        out[go] = predict_tree(node["left"], X_[go])
        out[~go] = predict_tree(node["right"], X_[~go])
        return out

    def shapley(f, x: np.ndarray, background: np.ndarray) -> tuple[float, np.ndarray]:
        """Exact interventional Shapley values of ``f`` at ``x`` over all feature subsets."""
        dim = len(x)
        rows, keys = [], []
        for k in range(dim + 1):
            for S in itertools.combinations(range(dim), k):
                Z = background.copy()
                Z[:, list(S)] = x[list(S)]
                rows.append(Z)
                keys.append(S)
        value = dict(
            zip(
                keys,
                f(np.vstack(rows)).reshape(len(keys), len(background)).mean(axis=1),
                strict=True,
            )
        )
        phi = np.zeros(dim)
        for j in range(dim):
            for k in range(dim):
                w = math.factorial(k) * math.factorial(dim - k - 1) / math.factorial(dim)
                for S in itertools.combinations([i for i in range(dim) if i != j], k):
                    phi[j] += w * (value[tuple(sorted((*S, j)))] - value[S])
        return float(value[()]), phi

    return best_split, grow, predict_tree, shapley


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Predict:** grown to depth 3 on all 360 sections, which input splits first? How much
    variance does the tree explain? The reference: scikit-learn's
    `DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)`.
    """)
    return


@app.cell
def _(FEATURES, NAMES, X, forest, full_tree, grow, np, predict_tree, small_tree, y):
    _ours = grow(X, y, depth=3, min_leaf=5)
    _ref = small_tree().fit(X, y)
    _diff = float(np.max(np.abs(predict_tree(_ours, X) - _ref.predict(X))))
    _full = full_tree().fit(X, y)
    _rf = forest().fit(X, y)
    # Slide "The tree fits; can it predict?": 1e-16; freezing index at 1,088; R2 0.21; full tree
    # 360 leaves and R2 1.00; forest 0.67.
    assert _diff < 1e-12
    assert (NAMES[FEATURES[_ours["feature"]]], round(float(_ours["threshold"]))) == (
        "freezing index",
        1088,
    )
    assert round(_ref.score(X, y), 2) == 0.21 and _ref.get_n_leaves() == 6
    assert (_full.get_n_leaves(), round(_full.score(X, y), 2), round(_rf.score(X, y), 2)) == (
        360,
        1.0,
        0.67,
    )
    print(
        f"CART vs scikit-learn: {_diff:.0e}; first split {NAMES[FEATURES[_ours['feature']]]} <= {_ours['threshold']:.1f}"
    )
    return


@app.cell
def _(d, np):
    # A rough floor: SPS-1 sections built alike at the same site.
    _s = d[d["GPS_SPS"] == "S"].copy()
    _s["design"] = (
        _s["ac_in"].round(0).astype(str)
        + "/"
        + _s["treated_base_in"].round(0).astype(str)
        + "/"
        + _s["granular_base_in"].round(0).astype(str)
    )
    _g = _s.groupby(["site", "design"])["iri_rate"]
    _sd = (
        _s.loc[_g.filter(lambda v: len(v) >= 2).index].groupby(["site", "design"])["iri_rate"].std()
    )
    assert len(_sd) == 4 and round(float(np.sqrt(np.mean(_sd**2))), 3) == 0.035
    print(
        f"{len(_sd)} groups built alike at one site: RMS SD {np.sqrt(np.mean(_sd**2)):.3f} m/km/yr"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    **Predict before we test** (write your answers down):

    1. **Folds.** The forest's $R^2$ against the mean: random sections, sites held out, states
       held out?
    2. **Importance.** Trucks rank high on the training states. On held-out states?
    3. **Depth.** The full tree, $R^2 = 1.00$ in training. With states held out?
    4. **Asphalt.** Effect of an inch of asphalt on the rate: across all 360 sections, and
       within SPS-1 sites?
    """)
    return


@app.cell
def _(
    BOOST_LADDER,
    BOOST_START,
    DummyRegressor,
    FOLDS,
    GroupKFold,
    INNER_FOLDS,
    KFold,
    LEAF_LADDER,
    LEAF_START,
    LinearRegression,
    MARKERS,
    PALETTE,
    SEED,
    X,
    boosting,
    clone,
    d,
    forest,
    full_tree,
    np,
    plt,
    small_tree,
    y,
):
    def _oof(make, Xa: np.ndarray, ya: np.ndarray, splits: list) -> np.ndarray:
        """Out-of-fold predictions: each fold's model is fitted on the other folds."""
        _pred = np.full(len(ya), np.nan)
        for _a, _b in splits:
            _pred[_b] = make().fit(Xa[_a], ya[_a]).predict(Xa[_b])
        assert not np.isnan(_pred).any()
        return _pred

    def _bracket(score, ladder: tuple, start: tuple, last_is_limit: bool) -> int:
        """The setting with the lowest inner MSE; widens one rung while the best is at an edge.

        Raises:
            ValueError: If the best setting is the ladder's last rung and that rung is not the
                setting's own limit. The first rung, the no-fit limit, has nothing beyond it and
                may be kept: then no amount of fitting beat the mean on the inner folds. A
                forest's last rung, leaves of one section, is fully grown trees: also a limit.
        """
        _lo, _hi = ladder.index(start[0]), ladder.index(start[-1])
        _scores = {_v: score(_v) for _v in start}
        while True:
            _best = min(ladder[_lo : _hi + 1], key=lambda _v: _scores[_v])
            if _best == ladder[_lo] and _lo > 0:
                _lo -= 1
                _scores[ladder[_lo]] = score(ladder[_lo])
            elif _best == ladder[_hi] and _hi < len(ladder) - 1:
                _hi += 1
                _scores[ladder[_hi]] = score(ladder[_hi])
            else:
                break
        if _best == ladder[-1] and not last_is_limit:
            raise ValueError(f"best setting {_best} is the ladder's last rung: {_scores}")
        return _best

    def _nested(scheme: str, make, ladder: tuple, start: tuple, last: bool, splits: list):
        """Each outer fold tunes on its own training rows, by inner folds of the same unit."""
        _pred, _kept = np.full(len(y), np.nan), []
        for _tr, _te in splits:
            _Xt, _yt = X[_tr], y[_tr]
            if scheme == "random sections":
                _inner = list(KFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(_Xt))
            else:  # the training sites or training states only
                _unit = "site" if scheme == "sites held out" else "STATE_CODE"
                _g = d[_unit].to_numpy()[_tr]
                _inner = list(GroupKFold(INNER_FOLDS).split(_Xt, groups=_g))

            def _score(v, _Xt=_Xt, _yt=_yt, _inner=_inner) -> float:
                return float(np.mean((_oof(lambda: make(v), _Xt, _yt, _inner) - _yt) ** 2))

            _best = _bracket(_score, ladder, start, last)
            _kept.append(int(_best))
            _pred[_te] = make(_best).fit(_Xt, _yt).predict(X[_te])
        return _pred, _kept

    _models = {
        "mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "tree, depth 3": small_tree(),
        "tree, full depth": full_tree(),
        "random forest": forest(),  # the lecture's forest, untuned: leaves of at least 3
        "boosting": boosting(),  # 200 rounds, untuned: the break-it row
    }
    _tuned = {
        "random forest, tuned": (forest, LEAF_LADDER, LEAF_START, True),
        "boosting, tuned": (boosting, BOOST_LADDER, BOOST_START, False),
    }
    _schemes = {
        "random sections": list(KFold(FOLDS, shuffle=True, random_state=SEED).split(X)),
        "sites held out": list(GroupKFold(FOLDS).split(X, y, d["site"])),
        "states held out": list(GroupKFold(FOLDS).split(X, y, d["STATE_CODE"])),
    }
    assert {len(_tr) for _s in _schemes.values() for _tr, _ in _s} == {288}
    rmse, kept = {}, {}
    for _k, _splits in _schemes.items():
        _preds = {_n: _oof(lambda _m=_m: clone(_m), X, y, _splits) for _n, _m in _models.items()}
        kept[_k] = {}
        for _n, (_make, _ladder, _start, _last) in _tuned.items():
            _preds[_n], kept[_k][_n] = _nested(_k, _make, _ladder, _start, _last, _splits)
        rmse[_k] = {_n: float(np.sqrt(np.mean((_p - y) ** 2))) for _n, _p in _preds.items()}
    r2 = {_k: {_m: 1 - _v**2 / _r["mean"] ** 2 for _m, _v in _r.items()} for _k, _r in rmse.items()}
    # Slides "Random folds promise; states deny" and "A deep tree memorizes", against the
    # figure script's numbers.json (values copied from it programmatically).
    _want = {
        "r2": {
            "random sections": {
                "mean": 0.0,
                "linear": 0.0149,
                "tree, depth 3": 0.0298,
                "tree, full depth": -0.4824,
                "random forest": 0.1609,
                "boosting": 0.0899,
                "random forest, tuned": 0.144,
                "boosting, tuned": 0.0752,
            },
            "sites held out": {
                "mean": 0.0,
                "linear": -0.0289,
                "tree, depth 3": -0.3308,
                "tree, full depth": -1.0215,
                "random forest": -0.0046,
                "boosting": -0.0806,
                "random forest, tuned": 0.0161,
                "boosting, tuned": -0.0017,
            },
            "states held out": {
                "mean": 0.0,
                "linear": -0.0491,
                "tree, depth 3": -0.1544,
                "tree, full depth": -1.6647,
                "random forest": -0.117,
                "boosting": -0.2165,
                "random forest, tuned": -0.0013,
                "boosting, tuned": -0.0184,
            },
        },
        "kept": {
            "random sections": {
                "random forest, tuned": [3, 1, 5, 1, 2],
                "boosting, tuned": [100, 200, 100, 200, 100],
            },
            "sites held out": {
                "random forest, tuned": [20, 40, 40, 40, 40],
                "boosting, tuned": [10, 5, 2, 0, 2],
            },
            "states held out": {
                "random forest, tuned": [80, 80, 40, 80, 40],
                "boosting, tuned": [0, 0, 5, 10, 10],
            },
        },
        "rmse_mean_states": 0.0554,
    }
    assert {_k: {_m: round(_v, 4) for _m, _v in _r.items()} for _k, _r in r2.items()} == _want["r2"]
    assert kept == _want["kept"]
    assert round(rmse["states held out"]["mean"], 4) == _want["rmse_mean_states"]
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _series = ("random forest, tuned", "boosting, tuned", "linear", "boosting")
    for _i, _m in enumerate(_series):
        _ax.plot(range(3), [r2[_k][_m] for _k in _schemes], marker=MARKERS[_i],
                 color=PALETTE[_i], label=_m if _m != "boosting" else "boosting, 200 rounds")  # fmt: skip
    _ax.axhline(0, color="grey", ls=":")
    _ax.set_xticks(range(3), list(_schemes))
    _ax.set_ylabel("R² against the mean")
    _ax.legend(fontsize=9)
    _fig
    return


@app.cell
def _(FEATURES, FOLDS, GroupKFold, NAMES, SEED, X, d, forest, np, pd, permutation_importance, y):
    # The lecture's forest (untuned, leaves of at least 3), as the slides explain it.
    _tr_fold, _te_fold = [], []
    _splits = list(GroupKFold(FOLDS).split(X, y, d["STATE_CODE"]))
    for _tr, _te in _splits:
        _rf = forest().fit(X[_tr], y[_tr])
        _tr_fold.append(
            permutation_importance(
                _rf, X[_tr], y[_tr], n_repeats=10, random_state=SEED
            ).importances_mean
        )
        _te_fold.append(
            permutation_importance(
                _rf, X[_te], y[_te], n_repeats=10, random_state=SEED
            ).importances_mean
        )
    _tr_imp, _te_imp = np.mean(_tr_fold, axis=0), np.mean(_te_fold, axis=0)
    _imp = (
        pd.DataFrame(
            {"training states": _tr_imp, "held-out states": _te_imp},
            index=[NAMES[_f] for _f in FEATURES],
        )
        .sort_values("training states", ascending=False)
        .round(2)
    )
    # Slide "Importance without skill": 0.20, 0.17, 0.17, 0.13 in training; trucks -0.09,
    # climate inputs -0.02, first IRI 0.06 held out.
    assert _imp["training states"].tolist()[:4] == [0.2, 0.17, 0.17, 0.13]
    assert _imp.index.tolist()[:4] == [
        "first IRI",
        "freezing index",
        "age at first visit",
        "trucks (ESAL/yr)",
    ]
    assert (
        _imp.loc["trucks (ESAL/yr)", "held-out states"],
        _imp.loc["first IRI", "held-out states"],
    ) == (-0.09, 0.06)
    assert (
        _imp.loc[
            ["freezing index", "precipitation", "mean temperature"], "held-out states"
        ].tolist()
        == [-0.02] * 3
    )
    assert round(float(np.corrcoef(d["freeze_index"], d["mean_temp_c"])[0, 1]), 2) == -0.83
    # Held out, no input but first IRI keeps more than 0.005 of importance.
    _held = dict(zip(FEATURES, _te_imp, strict=True))
    assert all(_v <= 0.005 for _f, _v in _held.items() if _f != "iri_first")
    # Notes: the trucks' held-out importance fold by fold; one fold carries most of the mean.
    _trucks = [round(float(_f[FEATURES.index("esal_per_yr")]), 3) for _f in _te_fold]
    _worst = int(np.argmin(_trucks))
    _without = round(float(np.mean(np.delete(np.array(_trucks), _worst))), 3)
    assert (_trucks, _without) == ([-0.084, -0.024, -0.3, 0.008, -0.035], -0.034)
    # Notes: sections that start rougher roughen faster (first IRI's held-out importance).
    assert round(float(np.corrcoef(d["iri_first"], y)[0, 1]), 2) == 0.23
    print(f"trucks held out, by fold: {_trucks}; without fold {_worst}: {_without}")
    _imp
    return


@app.cell
def _(FEATURES, NAMES, SEED, X, d, forest, np, shapley, y):
    # The slide's trace: Kansas SPS-1 section 0105, with Kansas held out of training: the
    # diagram's 349 sections in 52 states train each tree.
    _i = int(np.flatnonzero((d["STATE_CODE"] == 20) & (d["SHRP_ID"] == "0105"))[0])
    _tr = np.flatnonzero(d["STATE_CODE"] != 20)
    assert (len(_tr), d["STATE_CODE"].iloc[_tr].nunique()) == (349, 52)
    _rf = forest().fit(X[_tr], y[_tr])
    _bg = X[np.random.default_rng(SEED).choice(_tr, 100, replace=False)]
    _base, _phi = shapley(_rf.predict, X[_i], _bg)
    _pred = float(_rf.predict(X[_i : _i + 1])[0])
    assert abs(_base + _phi.sum() - _pred) < 1e-10  # efficiency
    _order = np.argsort(-np.abs(_phi))
    _top = [(NAMES[FEATURES[_k]], round(float(_phi[_k]), 3)) for _k in _order[:3]]
    assert (round(_base, 3), round(_pred, 3), round(float(y[_i]), 3)) == (0.041, 0.026, 0.263)
    assert _top == [
        ("age at first visit", -0.005),
        ("first IRI", -0.004),
        ("trucks (ESAL/yr)", -0.003),
    ]
    assert round(float(_phi[_order[3:]].sum()), 3) == -0.003
    print(
        f"base {_base:.3f} + Shapley values {np.round(_phi, 4)} = forest {_pred:.3f}; measured {y[_i]:.3f}"
    )
    return


@app.cell
def _(FEATURES, LinearRegression, d, np, pd, student_t):
    _obs = np.polyfit(d["ac_in"], d["iri_rate"], 1)[0]
    _s = (
        d[d["GPS_SPS"] == "S"]
        .groupby("site")
        .filter(lambda g: len(g) >= 4 and g["ac_in"].nunique() > 1)
    )
    _cols = ["iri_rate", "ac_in", "treated_base_in", "granular_base_in"]

    def within(frame: pd.DataFrame, by: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Least squares on deviations from each group's mean (group fixed effects).

        Returns:
            The estimates (asphalt, treated base, granular base), the demeaned thicknesses,
            and the residuals.
        """
        _dm = frame[_cols] - frame.groupby(by)[_cols].transform("mean")
        _X, _r = _dm[_cols[1:]].to_numpy(), _dm["iri_rate"].to_numpy()
        _b = np.linalg.lstsq(_X, _r, rcond=None)[0]
        return _b, _X, _r - _X @ _b

    _b, _X, _e = within(_s, "site")
    _sites = _s["site"].unique()
    # Cluster-robust (CR1) standard errors, sites as clusters: the sandwich
    # (X'X)^-1 (sum_g X_g' e_g e_g' X_g) (X'X)^-1, scaled by G/(G-1) (n-1)/(n-k), where k counts
    # the 3 slopes and the 15 site means (Stata's areg convention; xtreg, fe counts only the
    # slopes when the site effects nest in the site clusters, a factor of 1.006 instead of 1.054
    # here: we keep the conservative one); then a t interval on G - 1 = 14 degrees of freedom.
    _groups = _s["site"].to_numpy()
    _meat = sum(
        np.outer(_u, _u) for _u in (_X[_groups == _g].T @ _e[_groups == _g] for _g in _sites)
    )
    _xtx = _X.T @ _X
    _V = np.linalg.solve(_xtx, np.linalg.solve(_xtx, _meat).T)
    _G, _n, _k = len(_sites), len(_s), 3 + len(_sites)
    _se = np.sqrt(np.diag(_V) * _G / (_G - 1) * (_n - 1) / (_n - _k))
    _q = student_t.ppf(0.95, _G - 1)
    _lo, _hi = _b[0] - _q * _se[0], _b[0] + _q * _se[0]
    _lo_t, _hi_t = _b[1] - _q * _se[1], _b[1] + _q * _se[1]  # treated base
    # The percentile bootstrap of whole sites, which undercovers with 15 clusters.
    _rng = np.random.default_rng(0)
    _boot = []
    for _ in range(500):
        _pick = _rng.choice(_sites, len(_sites))
        _frame = pd.concat([_s[_s["site"] == _p].assign(bsite=_j) for _j, _p in enumerate(_pick)])
        _boot.append(within(_frame, "bsite")[0][0])
    _blo, _bhi = np.quantile(_boot, [0.05, 0.95])
    # Leave one site out: how far one site moves the estimate.
    _loso = [within(_s[_s["site"] != _g], "site")[0][0] for _g in _sites]
    # Slide "Design found what mining missed": +0.0006 (corr 0.025) across; -0.0043 within 15
    # sites, 90% CR1 interval -0.0075 to -0.0012; treated base -0.0031. Notes: bootstrap
    # -0.0073 to -0.0021; leaving one site out, -0.0050 to -0.0031.
    assert (round(_obs, 4), round(float(np.corrcoef(d["ac_in"], d["iri_rate"])[0, 1]), 3)) == (
        0.0006,
        0.025,
    )
    assert (_G, round(float(_b[0]), 4), round(float(_b[1]), 4)) == (15, -0.0043, -0.0031)
    assert (round(float(_lo), 4), round(float(_hi), 4)) == (-0.0075, -0.0012)
    # Notes: treated base, 90% CR1 interval -0.0055 to -0.0008 (it excludes zero too).
    assert (round(float(_lo_t), 4), round(float(_hi_t), 4)) == (-0.0055, -0.0008)
    assert (round(float(_blo), 4), round(float(_bhi), 4)) == (-0.0073, -0.0021)
    assert (round(float(min(_loso)), 4), round(float(max(_loso)), 4)) == (-0.0050, -0.0031)
    # Notes: thickness tracks traffic only weakly across sections, and a linear regression on
    # all nine inputs does not recover the within-site effect.
    _adj = LinearRegression().fit(d[FEATURES].to_numpy(), d["iri_rate"].to_numpy())
    assert (
        round(float(np.corrcoef(d["ac_in"], d["esal_per_yr"])[0, 1]), 2),
        round(float(_adj.coef_[FEATURES.index("ac_in")]), 5),
    ) == (0.17, 0.00014)
    print(
        f"asphalt: across sections {_obs:+.4f}; within SPS-1 sites {_b[0]:+.4f} m/km/yr per inch "
        f"(90% CR1: {_lo:+.4f} to {_hi:+.4f}; bootstrap {_blo:+.4f} to {_bhi:+.4f}; "
        f"one site out {min(_loso):+.4f} to {max(_loso):+.4f})"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What the breaks show.** Random folds leak sites: sections at one site share soil,
    climate, traffic, and crews. Tuned inside each fold on the training states, the forest and
    boosting shrink toward the mean (leaves of 40 to 80 sections; 0 to 10 rounds) and still do
    not beat it across held-out states; untuned boosting (200 rounds) does far worse. Importance
    on the training states rewards memorization; on held-out states, shuffling trucks
    *improves* the predictions, mostly in one fold of five. A full tree memorizes (360 leaves);
    averaging helps, but cannot create signal the inputs lack. The designed SPS-1 experiment,
    compared within sites, recovers what the observational data hid; a regression on all nine
    measured inputs does not (+0.00014 per inch).

    ## 6. Decide

    Report held-out skill by the unit of use before any explanation. For this DOT: the national
    forest adds nothing over the mean; the SPS-1 contrast says thicker asphalt and treated bases
    slow roughening at a given site; collect what the model lacks (as-built smoothness,
    materials, drainage, maintenance) and validate on held-out districts.

    **When not to use explanations:** on a model without held-out skill; from training-data
    importance; with correlated inputs (freezing and temperature, $-0.83$); as causes. First
    IRI's held-out importance matches a plain correlation: sections that start rougher roughen
    faster (0.23).

    **Impact.** An explanation looks like evidence, and budgets follow it: here, trucks ranked
    high in training and hurt held-out predictions. LTPP spent decades on designed experiments;
    mining cannot replace them.

    ### Vote: what do you tell the DOT?

    **A.** Trucks: high importance. **B.** Freezing: the depth-3 tree's first split. **C.** Asphalt and
    treated-base thickness: the SPS-1 within-site contrast. **D.** Nothing: no model beats the mean.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Discussion of the vote": mo.md(
                r"""
                C: a designed comparison with an interval that excludes zero. A and B come from a
                model with negative held-out $R^2$; D is right about the model and too strong
                about the data. What would change the answer: the state's own records with a
                held-out district, or a model that beats the mean across states.
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where trees live today

    - On typical tabular data, tree ensembles still beat deep networks (Grinsztajn, Oyallon &
      Varoquaux, 2022); XGBoost (Chen & Guestrin, 2016) and LightGBM are standard.
    - SHAP for trees is in most production tools: explanations are routine; checks are not.
    - Tabular foundation models (TabPFN; Hollmann et al., 2025) pretrain on synthetic tables
      and predict small datasets in one pass; Lecture 17. The same test applies.

    ## 7. Check yourself

    **Exit question.** A consultant shows the DOT a SHAP plot in which trucks drive roughening,
    from a model with $R^2 = 0.9$. What two questions do you ask before believing it?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(
                r"""
                (1) Was the 0.9 measured on held-out units like the DOT's roads (sites, states,
                years), or on random folds or training data? (2) Do the explanations agree with
                a designed comparison, or could confounding explain them?
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas worth flagging in a code review

    - **Random K-fold on sections that share sites.** Hold out sites or states.
    - **`RandomForestRegressor()` with defaults is bagging:** scikit-learn's regressor tries every
      input at every split (`max_features=1.0`). Set `max_features` (here 3 of 9).
    - **Permutation importance on the training data.** Measure it on held-out units.
    - **Explaining a model with negative held-out $R^2$.**
    - **Correlated inputs** shuffled one at a time: impossible climates.
    - **A full-depth tree reported by its training fit** ($R^2 = 1.00$).
    - **Reading SHAP as causation.** Confounding needs design.
    - **A percentile bootstrap over few clusters.** With 15 sites it undercovers (Cameron,
      Gelbach & Miller, 2008): report a cluster-robust interval with $t_{G-1}$, and how far
      leaving one site out moves the estimate.
    - **Two-digit years in LTPP dates** ("63" is 1963): the fetch script corrects them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Going deeper: textbook references

    - **Forsyth §2.2.1–2.2.2:** building a classification tree, choosing each split by
      information gain (entropy) where we use the squared error.
    - **Forsyth §2.2.3–2.2.5:** forests: random subsets of inputs at each split, and voting.
    - **Forsyth §12.1.2–12.1.3:** regression trees, and greedy stagewise regression that fits
      each new tree to the residuals; Forsyth writes the model as a plain sum of trees,
      $F(\mathbf{x}) = \sum_j f(\mathbf{x}; \theta^{(j)})$, without our shrinkage $\nu$.
    - **Exercise.** Show that a split never increases the training SSE, then explain why a
      tree grown to one case per leaf has training $R^2 = 1$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to remember

    - A tree splits to cut squared error; forests average away variance; boosting fits residuals.
    - Validate by the unit of use: random folds promised $R^2$ 0.14; held-out states gave
      −0.001, even with tuning.
    - Explanations describe the model: trucks ranked high in training and hurt held out.
    - Design beats mining: within SPS-1 sites, an inch of asphalt slowed roughening.

    **Next:** outliers and robust estimation: finding a burst pipe.
    """)
    return


if __name__ == "__main__":
    app.run()
