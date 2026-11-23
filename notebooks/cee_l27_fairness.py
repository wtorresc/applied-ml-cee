"""Applied Machine Learning for CEE — Lecture 27: Fairness and Accountability.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l27_fairness.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: the Illinois National Bridge Inventory (FHWA; U.S. Government work): run
``bash scripts/fetch_nbi.sh`` once. The CDC/ATSDR Social Vulnerability Index 2022 for Illinois
counties: run ``bash scripts/fetch_svi.sh`` once. Source: CDC/ATSDR; its use here does not imply
endorsement by CDC, ATSDR, HHS, or the U.S. Government; the data are available at no charge from
ATSDR. The slides' numbers come from ``lectures/build/fig/cee_l27_fairness/`` (``make`` in
``lectures/``). Everything runs on the CPU in about a minute.
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
    # Applied Machine Learning for CEE — Lecture 27: Fairness and Accountability

    **Slides:** *Applied Machine Learning for CEE*, Lecture 27 (`lectures/cee_l27_fairness`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth's *Applied Machine Learning* (Springer 2019) has no chapter on fairness. Its §1.1.2 (false-positive and false-negative rates, always reported together) and §3.1 and §3.2.1 (a held-out rate estimates the test rate when the items are independent; Hoeffding's inequality for a rate from $N$ cases) are the background. For the rest: Barocas, Hardt, and Narayanan (2023), *Fairness and Machine Learning*, chapters 3 and 4; Mitchell et al. (2019) on model cards; Gebru et al. (2021) on datasheets; Chouldechova (2017); Hardt et al. (2016) (see Going deeper).

    **Data:** the National Bridge Inventory (FHWA; Illinois, 2014–2024, every two years; U.S. Government work), the records of Lectures 1, 3, and 6: `bash scripts/fetch_nbi.sh` writes `data/derived/nbi_il_bridges.csv`. The CDC/ATSDR Social Vulnerability Index (SVI) 2022, Illinois, county level: `bash scripts/fetch_svi.sh` writes `data/raw/svi/SVI_2022_Illinois_county.csv`. **Source: CDC/ATSDR. Its use here does not imply endorsement by CDC, ATSDR, HHS, or the U.S. Government. The data are available at no charge from ATSDR** (https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html). The slides' numbers: `lectures/build/fig/cee_l27_fairness/numbers.json`, written by the figure script `lectures/figures/cee_l27_fairness.py`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    The audit is written here from scratch: Lecture 6's bridge classifier (the loader, the
    four-year transitions, the scaler, Newton's method), the join to the SVI by county FIPS code,
    the confusion-matrix rates by group, the intervals that resample whole counties, the
    identity that ties the rates together (Chouldechova, 2017), the proxy test (a penalized
    logistic regression by Newton's method and the AUC from ranks, on the same counties and on
    folds of counties held out), the per-quartile thresholds
    chosen on the validation years (Hardt et al., 2016), and the intersectional cells. Each piece
    is checked three ways:

    - **Against the code that made the slides:** Lecture 6's own functions, loaded from
      `lectures/figures/cee_l06_linear_classifiers.py` by file path, give the same weights and
      test predictions; the Lecture 27 figure script's functions, loaded the same way, give the
      same join, rates, intervals (draw for draw, from the same generator in the same order),
      identity, proxy AUCs, and thresholds.
    - **Against the slides' `numbers.json`**, read programmatically: the notebook
      rebuilds the whole file, key for key, and asserts that it equals the slides'.
    - **Against the deck's own words:** every number the deck states is rendered here from this
      notebook's values and found word for word in the deck; statements without a number of
      their own are measured, with a verdict (§7).

    **One setting.** Nothing here is long: the whole notebook took about a minute on the
    workstation (November 20, 2026, while other jobs shared its processors), so a laptop needs
    about as long (see Gotchas for this run's time). The bootstrap keeps the figure script's
    1,000 draws per group and takes under a second, because each draw sums per-county counts; it
    is checked draw for draw against the figure script's and, for 200 draws, against pooling the
    drawn counties' bridges. The slowest steps are the two proxy tests on held-out counties (10
    shuffles of 10 county folds, two sets of inputs: 200 fits each), this notebook's and the
    figure script's, about half a minute between them.

    **Your tasks** (from the slides): compute catch and false-alarm rates by group; decide which
    error is the harm; build intervals that resample counties; decide whether a gap is real;
    derive the identity that forces fairness criteria to conflict; try per-group thresholds;
    split the groups further and read the intervals; and write a model card and a datasheet.
    """)
    return


@app.cell
def _():
    import importlib.util
    import json
    import os
    import random
    import re
    import sys
    import time
    from pathlib import Path

    # PYTHONHASHSEED only reaches child processes (none are started here); string hashing plays
    # no part in any result below
    os.environ["PYTHONHASHSEED"] = "0"

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    T_START = time.perf_counter()
    SEED = 27  # the figure script's seed: the county bootstrap's generator
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # nothing here draws from torch; seeded as every notebook is

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
    # The figure scripts' shared modules on the path, then the two scripts this notebook checks
    # against, each loaded from its file. Lecture 6's script is registered under its own module
    # name before the Lecture 27 script runs, so that script's `import cee_l06_linear_classifiers`
    # finds this exact module, not notebooks/cee_l06_linear_classifiers.py (Lecture 6's notebook,
    # which has the same file name). The Lecture 27 script has this notebook's file name, so it is
    # loaded under another module name.
    FIGURES = ROOT / "lectures" / "figures"
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))

    def load_script(name: str, path: Path) -> object:
        """Load a figure script from its file as module ``name``.

        Args:
            name: The module name to register in ``sys.modules``.
            path: The script's file.

        Returns:
            The executed module.
        """
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    l06ref = load_script("cee_l06_linear_classifiers", FIGURES / "cee_l06_linear_classifiers.py")
    ref = load_script("l27_figures", FIGURES / "cee_l27_fairness.py")
    assert ref.l06 is l06ref, "the Lecture 27 script imported another Lecture 6 module"
    assert Path(ref.l06.__file__).resolve() == (FIGURES / "cee_l06_linear_classifiers.py").resolve()
    _bridges = sys.modules["bridges"]  # the shared NBI module both scripts use

    # ------------------------------------------------ the slides' constants, written out here
    QUARTILES = ("low", "low-mid", "mid-high", "high")
    EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)  # quartiles closed on the right; the top one is above 0.75
    COOK = "17031"  # Cook County (Chicago): state 17, county 031
    BOOT = 1000  # county resamples per group, for 95% intervals
    POOR = 4  # FHWA: a lowest rating of 4 or less is poor
    HORIZON = 4  # years from the decision to the inventory that judges it
    START = {"train": 2014, "val": 2016, "test": 2020}
    R = 20  # Lecture 6: a miss costs R false alarms
    THRESHOLD = 1 / (1 + R)  # so flag when p > 1/21
    FEATURES = [
        "age",
        "years_since_work",
        "log_adt",
        "pct_truck",
        "log_length",
        "deck_width_m",
        "concrete",
        "steel",
        "prestressed",
        "timber",
        "lat",
        "lon",
    ]
    COMPONENTS = ["deck", "superstructure", "substructure"]
    INPUTS = FEATURES + COMPONENTS + ["rating"]  # 12 inventory inputs + 4 ratings
    MATERIAL = {
        "1": "concrete",
        "2": "concrete",
        "3": "steel",
        "4": "steel",
        "5": "prestressed",
        "6": "prestressed",
        "7": "timber",
    }
    OWNERS = {"01": "state", "02": "county", "03": "township", "04": "city"}  # NBI Item 22
    PROXY_DROP = ("lat", "lon")  # the inputs that place a bridge on the map
    PROXY_C = 1.0  # the proxy classifier's L2 strength, scikit-learn's C
    PROXY_FOLDS = 10  # the held-out proxy test: counties in 10 folds, stratified by quartile
    PROXY_REPEATS = 10  # shuffles of those folds (random_state 0 to 9)
    COUNTY_CODES = {f"{2 * _i + 1:03d}" for _i in range(102)}  # Illinois: 001, 003, ..., 203
    _mine = {
        "SEED": SEED, "QUARTILES": QUARTILES, "EDGES": EDGES, "COOK": COOK, "BOOT": BOOT,
        "THRESHOLD": THRESHOLD, "PROXY_DROP": PROXY_DROP, "POOR": POOR, "PROXY_C": PROXY_C,
        "PROXY_FOLDS": PROXY_FOLDS, "PROXY_REPEATS": PROXY_REPEATS,
    }  # fmt: skip
    for _name, _value in _mine.items():  # the same settings as the slides
        assert getattr(ref, _name) == _value, _name
    assert (l06ref.HORIZON, l06ref.START, l06ref.R) == (HORIZON, START, R)
    assert list(l06ref.INPUTS) == INPUTS and l06ref.OWNERS == OWNERS
    assert _bridges.MATERIAL == MATERIAL and set(_bridges.COUNTY) == COUNTY_CODES
    assert ref.OWNERS == {**OWNERS, "other": "other"}

    NBI_CSV = ROOT / "data" / "derived" / "nbi_il_bridges.csv"
    SVI_CSV = ROOT / "data" / "raw" / "svi" / "SVI_2022_Illinois_county.csv"
    for _path, _script in ((NBI_CSV, "fetch_nbi.sh"), (SVI_CSV, "fetch_svi.sh")):
        if not _path.exists():
            raise FileNotFoundError(f"{_path} not found; run `bash scripts/{_script}`")

    # The slides' run, every value as the figure script wrote it: the reference
    # (read programmatically, never typed)
    BUILD = ROOT / "lectures" / "build" / "fig"
    NUMBERS = BUILD / "cee_l27_fairness" / "numbers.json"
    if not NUMBERS.exists():
        raise FileNotFoundError(
            f"{NUMBERS} not found: build the slides' figures with `make` in lectures/ (the "
            "figure script takes about 2.5 minutes on a CPU)"
        )
    REF = json.loads(NUMBERS.read_text(encoding="utf-8"))
    # Lectures 6 and 26 for the warm-up, if their figures are built (checked when present)
    L06_NUMBERS = BUILD / "cee_l06_linear_classifiers" / "numbers.json"
    L26_NUMBERS = BUILD / "cee_l26_reinforcement_learning" / "numbers.json"
    DECK = re.sub(
        r"\s+", " ", (ROOT / "lectures" / "cee_l27_fairness.md").read_text(encoding="utf-8")
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
        """The slides' value at ``path`` in numbers.json, e.g. "post/flags_one" ("" for all).

        Args:
            path: Keys separated by "/".

        Returns:
            The value.
        """
        node = REF
        for key in filter(None, path.split("/")):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def as_json(x: object) -> object:
        """``x`` as it reads back from JSON: string keys, lists, Python numbers.

        Args:
            x: Numbers, strings, booleans, and dicts, lists, and tuples of them.

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
            path: Its key in numbers.json ("" for the whole file).

        Raises:
            AssertionError: Saying where the two first differ.
        """
        a = as_json(x)
        assert agree(a, slide(path)), first_difference(a, slide(path), path)

    CDC_NOTE = (
        "Source: CDC/ATSDR, Social Vulnerability Index 2022 Database, Illinois (counties). Its "
        "use here does not imply endorsement by CDC, ATSDR, HHS, or the U.S. Government; the "
        "data are available at no charge from ATSDR."
    )
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    MARKERS = ["o", "s", "^", "D"]
    INK, MUTED = "#1A1A1A", "#595959"
    Q_COLOR = dict(zip(QUARTILES, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"], strict=True))
    FULL, SIDE = (9.0, 3.6), (5.2, 3.6)
    print(f"root {ROOT}; reference scripts loaded; {len(DECK):,} characters of deck text")
    return (
        BOOT,
        CDC_NOTE,
        COMPONENTS,
        COOK,
        COUNTY_CODES,
        DECK,
        EDGES,
        FEATURES,
        FIGURES,
        FULL,
        FancyBboxPatch,
        HORIZON,
        INK,
        INPUTS,
        L06_NUMBERS,
        L26_NUMBERS,
        Line2D,
        LogisticRegression,
        MARKERS,
        MATERIAL,
        MUTED,
        NBI_CSV,
        OWNERS,
        PALETTE,
        POOR,
        PROXY_C,
        PROXY_DROP,
        PROXY_FOLDS,
        PROXY_REPEATS,
        Path,
        QUARTILES,
        Q_COLOR,
        R,
        REF,
        SEED,
        SIDE,
        START,
        SVI_CSV,
        THRESHOLD,
        T_START,
        as_json,
        check,
        expit,
        json,
        l06ref,
        np,
        pct,
        pd,
        plt,
        re,
        ref,
        rnd,
        roc_auc_score,
        slide,
        StratifiedKFold,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 6–26

    On your own, no notes, two minutes:

    1. Lecture 6: where did the flagging threshold come from?
    2. Lecture 6: whose bridges did the model catch least, by owner?
    3. Lecture 26: what did the MDP do when being poor cost nothing?

    **Today:** the same bridge model, asked whom it serves.
    """)
    return


@app.cell
def _(R, l26_never, mo, owner_caught):
    _others = [v for k, v in owner_caught.items() if k in ("state", "county", "city")]
    mo.accordion(
        {
            "Answers to the warm-up": mo.md(
                f"""
                1. From the costs: a miss costs {R} false alarms, so flagging pays when
                   $p\\,r > 1 - p$, that is when $p > 1/(1+r) = 1/{1 + R}$ (Lecture 6).
                2. Townships': the model caught {owner_caught["township"]:.0f}% of the township
                   bridges that turned poor, against {min(_others):.0f}% to {max(_others):.0f}%
                   for the state's, the counties', and the cities' (recomputed in §1 from the
                   rebuilt model). Today the groups are counties, ranked by social vulnerability.
                3. {l26_never} The optimizer did what the reward said; today's question is what
                   "fair" should say.
                """
            )
        }
    )
    return


@app.cell
def _(COUNTY_CODES, HORIZON, POOR, R, START, mo):
    _tr, _va, _te = START["train"], START["val"], START["test"]
    mo.md(f"""
    ## 1. The engineering question

    **The decision (Lecture 6).** Of the Illinois bridges that are not poor today (lowest of the
    deck, superstructure, and substructure ratings above {POOR}), which will be poor in the
    inventory {HORIZON} years later? A flagged bridge gets preventive work and a closer look; a
    miss is a bridge that turns poor unwatched. A miss costs $r = {R}$ false alarms, so the model
    flags a bridge when its chance $p$ of turning poor exceeds $1/{1 + R}$. It was trained on
    the {_tr} → {_tr + HORIZON} transitions, validated on {_va} → {_va + HORIZON}, and tested
    once: the {_te} bridges, judged by the {_te + HORIZON} inventory.

    **Today's question.** Does that model serve the state's socially vulnerable counties as well
    as the others? A flag brings resources, so a county whose bridges are missed more often
    gets less of them. The groups are Illinois's {len(COUNTY_CODES)} counties, ranked by the
    CDC/ATSDR Social Vulnerability Index 2022 and cut into quartiles; the model never sees the
    index.

    ### The model being audited: Lecture 6's, rebuilt

    The loader, the transitions, the standardization, and Newton's method, written out again
    here (the same code as Lecture 6's notebook); the next cells check them against Lecture 6's
    own functions.
    """)
    return


@app.cell
def _(
    COMPONENTS,
    COUNTY_CODES,
    FEATURES,
    HORIZON,
    INPUTS,
    MATERIAL,
    NBI_CSV,
    POOR,
    Path,
    START,
    np,
    pd,
):
    def load_bridges(path: Path) -> pd.DataFrame:
        """Load the bridge-year records and add the inputs (the Lecture 3 loader).

        Args:
            path: The CSV written by ``scripts/fetch_nbi.sh``.

        Returns:
            Records with complete inventory inputs and ``material``.
        """
        text = ["structure", "county", "district", "owner", "kind", "design", "parallel"]
        frame = pd.read_csv(path, dtype=dict.fromkeys(text, str))
        assert set(frame["county"]) <= COUNTY_CODES, "unknown county code"
        frame["age"] = (frame["inspection_year"] - frame["year_built"]).clip(lower=0)
        _last = np.fmax(frame["year_built"], frame["year_reconstructed"].fillna(0))
        frame["years_since_work"] = (frame["inspection_year"] - _last).clip(lower=0)
        frame["log_adt"] = np.log1p(frame["adt"])
        frame["log_length"] = np.log(frame["length_m"].clip(lower=1.0))
        frame["material"] = frame["kind"].map(MATERIAL).fillna("other")
        for _m in ("concrete", "steel", "prestressed", "timber"):
            frame[_m] = (frame["material"] == _m).astype(float)
        return frame.dropna(subset=FEATURES).reset_index(drop=True)

    def transitions(frame: pd.DataFrame, start: int) -> pd.DataFrame:
        """Bridges not poor in ``start`` and their lowest rating ``HORIZON`` years later.

        Args:
            frame: Bridge-year records from :func:`load_bridges`.
            start: The NBI year of the decision.

        Returns:
            One row per bridge rated above ``POOR`` in ``start`` that is still in the inventory
            ``HORIZON`` years later, with ``rating_next`` and ``poor_next`` (0/1). A missing
            deck rating (no deck) gets the lower of the other two.
        """
        key = ["structure", "nbi_year"]
        later = frame[[*key, "rating"]].rename(columns={"rating": "rating_next"})
        later["nbi_year"] -= HORIZON
        out = frame[frame["nbi_year"] == start].merge(
            later, on=key, how="inner", validate="one_to_one"
        )
        out = out[out["rating"] > POOR].reset_index(drop=True)
        out["deck"] = out["deck"].fillna(out[[*COMPONENTS[1:]]].min(axis=1))
        out["poor_next"] = (out["rating_next"] <= POOR).astype(int)
        assert out[INPUTS].notna().all().all(), "missing inputs"
        return out

    nbi = load_bridges(NBI_CSV)
    raw_sets = {_name: transitions(nbi, _start) for _name, _start in START.items()}
    print({_k: (len(_d), int(_d["poor_next"].sum())) for _k, _d in raw_sets.items()})
    return nbi, raw_sets


@app.cell
def _(INPUTS, THRESHOLD, expit, l06ref, np, raw_sets, ref):
    # The training scaler: mean and population standard deviation of the 2014 bridges
    _X = raw_sets["train"][INPUTS].to_numpy(np.float64)
    scale_mean, scale_sd = _X.mean(axis=0), _X.std(axis=0)

    def encode(records) -> np.ndarray:
        """The 16 inputs, standardized with the training bridges' mean and spread."""
        _Z = (records[INPUTS].to_numpy(np.float64) - scale_mean) / scale_sd
        assert _Z.shape == (len(records), len(INPUTS)), _Z.shape
        return _Z

    def fit_logistic(
        Z: np.ndarray, y: np.ndarray, tol: float = 1e-10, max_steps: int = 50
    ) -> tuple[np.ndarray, int]:
        """Maximum-likelihood logistic regression by Newton's method (Lecture 6).

        Args:
            Z: Standardized inputs, shape ``(n, d)``.
            y: Labels in {0, 1}, shape ``(n,)``.
            tol: Stop when no weight moves more than this.
            max_steps: Give up after this many steps.

        Returns:
            Weights ``[w0, w]`` of shape ``(d + 1,)``, and the number of steps taken.

        Raises:
            RuntimeError: If Newton's method has not converged after ``max_steps``.
        """
        A = np.column_stack([np.ones(len(Z)), Z])  # intercept column
        w = np.zeros(A.shape[1])
        for step in range(1, max_steps + 1):
            p = expit(A @ w)
            g = A.T @ (p - y)  # gradient of the negative log-likelihood
            H = A.T @ (A * (p * (1 - p))[:, None])  # Hessian A^T S A
            delta = np.linalg.solve(H, g)
            w -= delta
            if np.abs(delta).max() < tol:
                return w, step
        raise RuntimeError("Newton did not converge")

    def predict_proba(w: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """P(poor in four years) for standardized inputs ``Z``."""
        return expit(w[0] + Z @ w[1:])

    w_l06, newton_steps = fit_logistic(encode(raw_sets["train"]), raw_sets["train"]["poor_next"])
    p_sets = {_k: predict_proba(w_l06, encode(_d)) for _k, _d in raw_sets.items()}

    # Check: Lecture 6's own functions (its transitions, encode, fit_logistic, predict_proba),
    # composed as the Lecture 27 script does in l06_model()
    ref_model = ref.l06_model()
    _w_ref, _steps_ref = l06ref.fit_logistic(
        l06ref.encode(ref_model["train"], ref_model["scaler"]),
        ref_model["train"]["poor_next"].to_numpy(),
    )
    assert np.array_equal(_w_ref, ref_model["w"]) and newton_steps == _steps_ref
    for _k, _d in raw_sets.items():  # the same bridges, in the same order, with the same labels
        _r = ref_model[_k]
        assert _d["structure"].tolist() == _r["structure"].tolist(), _k
        assert np.array_equal(_d["poor_next"].to_numpy(), _r["poor_next"].to_numpy()), _k
        assert np.array_equal(_d[INPUTS].to_numpy(), _r[INPUTS].to_numpy()), _k
    _mean_gap = float(np.abs(scale_mean - ref_model["scaler"].mean_).max())
    _sd_gap = float(np.abs(scale_sd / ref_model["scaler"].scale_ - 1).max())
    w_gap = float(np.abs(w_l06 - ref_model["w"]).max())
    p_gap = max(float(np.abs(p_sets[_k] - ref_model[_k]["p"]).max()) for _k in raw_sets)
    assert np.allclose(w_l06, ref_model["w"], rtol=1e-9, atol=1e-12), w_gap
    assert p_gap < 1e-12, p_gap
    # Every flag the same: no bridge sits within the predictions' disagreement of 1/21
    for _k in raw_sets:
        assert np.array_equal(p_sets[_k] > THRESHOLD, ref_model[_k]["p"] > THRESHOLD), _k
    _margin = min(float(np.abs(p_sets[_k] - THRESHOLD).min()) for _k in raw_sets)
    assert _margin > 1e3 * max(p_gap, 1e-16), _margin
    print(
        f"Newton: {newton_steps} steps, the same as Lecture 6's fit_logistic; weights within "
        f"{w_gap:.1e} of Lecture 6's (scaler means {_mean_gap:.1e}, spreads {_sd_gap:.1e} "
        f"relative), predictions within {p_gap:.1e}; every flag identical (nearest p to 1/21: "
        f"{_margin:.1e} away)"
    )
    return newton_steps, p_gap, p_sets, ref_model, w_gap, w_l06


@app.cell
def _(L06_NUMBERS, L26_NUMBERS, OWNERS, THRESHOLD, json, np, p_sets, pct, raw_sets):
    # Warm-up question 2: the share caught by owner (NBI Item 22) on the test years
    _t = raw_sets["test"]
    _flag = p_sets["test"] > THRESHOLD
    _y = _t["poor_next"].to_numpy() == 1
    owner_caught = {}
    for _code, _name in OWNERS.items():
        _m = (_t["owner"] == _code).to_numpy() & _y
        owner_caught[_name] = pct((_flag & _m).sum() / _m.sum(), 0)
    _checked = []
    if L06_NUMBERS.exists():  # Lecture 6's slides, if built: the same shares
        _l06 = json.loads(L06_NUMBERS.read_text(encoding="utf-8"))["by_owner"]
        assert all(_l06[_n]["caught_pct"] == v for _n, v in owner_caught.items()), _l06
        _checked.append("Lecture 6's numbers.json")
    # Warm-up question 3: Lecture 26's MDP with no cost for being poor
    if L26_NUMBERS.exists():
        _never = json.loads(L26_NUMBERS.read_text(encoding="utf-8"))["bridge"]
        assert _never["no_penalty_replaces"] is False
        l26_never = (
            "It never replaced anything (Lecture 26's numbers.json: with no cost for being "
            "poor, no rating is replaced)."
        )
        _checked.append("Lecture 26's numbers.json")
    else:
        l26_never = "It never replaced anything (Lecture 26; its numbers.json is not built here)."
    print(f"caught by owner, test years: {owner_caught}; checked against: {_checked or 'none'}")
    assert np.isfinite(list(owner_caught.values())).all()
    return l26_never, owner_caught


@app.cell
def _(CDC_NOTE, mo):
    mo.md(f"""
    ### The Social Vulnerability Index, joined by county

    The SVI ranks each county on census measures from the American Community Survey (the next
    cell counts them in the file): income below 150% of the poverty line, unemployment, housing
    cost burden, no high-school diploma, no health insurance, age 65 and over, age 17 and under,
    disability, single-parent households, limited English, racial and ethnic minority status,
    multi-unit housing, mobile homes, crowding, no vehicle, and group quarters. `RPL_THEMES` is a
    county's overall percentile rank within the state, from 0 to 1. The quartiles are closed on
    the right, as the figure script's are: the top quartile, the most vulnerable, holds the ranks
    above 0.75. The file is one row per county, joined to each bridge by the five-digit FIPS
    code: state 17, then the NBI's three-digit county code.

    *{CDC_NOTE}*
    """)
    return


@app.cell
def _(EDGES, QUARTILES, SVI_CSV, np, pd, ref):
    # The CSV starts with a byte-order mark: read it as utf-8-sig, or the first column is
    # "\ufeffST". FIPS codes are text: as integers, Illinois's leading digits survive, but the
    # NBI's county "001" would not.
    _raw = pd.read_csv(SVI_CSV, encoding="utf-8-sig", dtype={"FIPS": str})
    assert _raw.columns[0] == "ST" and len(_raw) == 102 and _raw["FIPS"].str.len().eq(5).all()
    svi_measures = [c for c in _raw.columns if c.startswith("EPL_")]  # one rank per measure
    assert len(svi_measures) == 16, svi_measures
    svi = pd.DataFrame(
        {
            "fips": _raw["FIPS"],
            "county_name": _raw["COUNTY"],
            "rpl": _raw["RPL_THEMES"].astype(float),
            "pop": _raw["E_TOTPOP"].astype(int),
            "area_sqmi": _raw["AREA_SQMI"].astype(float),
        }
    )
    assert svi["rpl"].between(0, 1).all() and svi["fips"].is_unique  # no -999 (missing) ranks
    # Quartiles closed on the right: [0, 0.25], (0.25, 0.5], (0.5, 0.75], (0.75, 1]
    _k = np.searchsorted(np.array(EDGES[1:-1]), svi["rpl"].to_numpy(), side="left")
    svi["quartile"] = np.array(QUARTILES)[_k]
    # Check: the figure script's quartiles (pd.cut). No county's rank sits on an edge (with 102
    # counties the ranks are k/101), so closing the bins on the left would give the same ones.
    _theirs = ref.load_svi()
    assert not np.isin(svi["rpl"], EDGES[1:-1]).any()
    assert svi["fips"].tolist() == _theirs["fips"].tolist()
    assert svi["quartile"].tolist() == _theirs["quartile"].astype(str).tolist()
    assert np.allclose(svi["rpl"] * 101, np.round(svi["rpl"] * 101), atol=0.02)
    svi.groupby("quartile").agg(
        counties=("fips", "size"), population=("pop", "sum"), rank_min=("rpl", "min"),
        rank_max=("rpl", "max"),
    ).loc[list(QUARTILES)]  # fmt: skip
    return svi, svi_measures


@app.cell
def _(OWNERS, THRESHOLD, np, p_sets, pd, raw_sets, ref, ref_model, svi):
    def with_svi(d: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
        """Join bridges to their county's SVI rank; add p, the flag, owner, and cell names.

        Args:
            d: Transitions with the NBI's three-digit ``county`` code.
            p: The model's chance of poor in four years for each row.

        Returns:
            The rows of ``d`` in order with ``fips``, ``county_name``, ``rpl``, ``quartile``,
            ``pop``, ``p``, ``flag``, ``owner_name``, and ``cell`` (quartile | owner).
        """
        out = d.assign(fips="17" + d["county"].str.zfill(3), p=p)
        out = out.merge(svi, on="fips", how="left", validate="many_to_one")
        assert len(out) == len(d) and out["rpl"].notna().all(), "a bridge without an SVI rank"
        out["flag"] = out["p"] > THRESHOLD
        out["owner_name"] = out["owner"].map(OWNERS).fillna("other")
        out["cell"] = out["quartile"] + " | " + out["owner_name"]
        return out

    train, val, test = (with_svi(raw_sets[_k], p_sets[_k]) for _k in ("train", "val", "test"))
    same_counties = {frozenset(_d["fips"]) for _d in (train, val, test)}  # one set: the same
    # Check: the figure script's join of Lecture 6's sets (its with_svi)
    _svi_ref = ref.load_svi()
    for _mine, _k in ((train, "train"), (val, "val"), (test, "test")):
        _r = ref.with_svi(ref_model[_k], _svi_ref)
        assert _mine["fips"].tolist() == _r["fips"].tolist(), _k
        assert _mine["quartile"].tolist() == _r["quartile"].astype(str).tolist(), _k
        assert np.array_equal(_mine["flag"].to_numpy(), _r["flag"].to_numpy()), _k
        assert _mine["owner_name"].tolist() == _r["owner_name"].tolist(), _k
    print(
        f"joined: {len(train):,} training, {len(val):,} validation, {len(test):,} test bridges; "
        f"{test['fips'].nunique()} counties in the test set"
    )
    return same_counties, test, train, val


@app.cell
def _(CDC_NOTE, COOK, INK, Line2D, MARKERS, MUTED, QUARTILES, Q_COLOR, mo, np, plt, test):
    # Each county at the mean position of its 2020 bridges, sized by their number
    _c = test.groupby("fips").agg(
        lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"), q=("quartile", "first")
    )
    _fig, _ax = plt.subplots(figsize=(5.2, 4.8))
    for _q, _mk in zip(QUARTILES, MARKERS, strict=True):
        _s = _c[_c["q"] == _q]
        _ax.scatter(_s["lon"], _s["lat"], s=_s["n"] / 6, color=Q_COLOR[_q], marker=_mk,
                    edgecolor=INK, lw=0.4, label=_q)  # fmt: skip
    _cook = _c.loc[COOK]
    _ax.annotate("Cook (Chicago)", (_cook["lon"], _cook["lat"]),
                 (_cook["lon"] - 2.9, _cook["lat"] + 0.75), fontsize=10,
                 arrowprops={"arrowstyle": "-", "color": MUTED})  # fmt: skip
    _ax.set_aspect(1 / np.cos(np.deg2rad(40)))
    _ax.set(xticks=[], yticks=[])
    for _sp in _ax.spines.values():
        _sp.set_visible(False)
    _handles = [
        Line2D([], [], ls="none", marker=_mk, ms=8, color=Q_COLOR[_q], mec=INK, mew=0.4, label=_q)
        for _q, _mk in zip(QUARTILES, MARKERS, strict=True)
    ]
    _ax.legend(handles=_handles, title="SVI quartile", fontsize=9, title_fontsize=9,
               frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))  # fmt: skip
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Illinois's {len(_c)} counties, each at the mean position of its 2020 bridges, "
                f"sized by their number and marked by SVI quartile. *{CDC_NOTE}*"
            ),
        ]
    )
    return


@app.cell
def _(CDC_NOTE, mo, test):
    _r = {}
    for _q in ("low", "high"):
        _g = test[test["quartile"] == _q]
        _y, _f = _g["poor_next"].to_numpy() == 1, _g["flag"].to_numpy()
        _r[_q] = {"counties": _g["fips"].nunique(), "poor": int(_y.sum()),
                  "caught": int((_f & _y).sum()), "healthy": int((~_y).sum()),
                  "false_alarms": int((_f & ~_y).sum())}  # fmt: skip
    attempt = _r
    mo.md(f"""
    ### Your turn: catch and false-alarm rates

    The test years (2020 → 2024), the lowest and the highest SVI quartile:

    | SVI quartile | Counties | Turned poor | Caught (flagged) | Stayed healthy | Stayed healthy, flagged |
    |:--|--:|--:|--:|--:|--:|
    | Low | {_r["low"]["counties"]} | {_r["low"]["poor"]} | {_r["low"]["caught"]} | {_r["low"]["healthy"]:,} | {_r["low"]["false_alarms"]:,} |
    | High | {_r["high"]["counties"]} | {_r["high"]["poor"]} | {_r["high"]["caught"]} | {_r["high"]["healthy"]:,} | {_r["high"]["false_alarms"]:,} |

    Two minutes: compute each quartile's catch rate and false-alarm rate. Which difference
    matters? *{CDC_NOTE}*
    """)
    return (attempt,)


@app.cell
def _(attempt, mo):
    _lo, _hi = attempt["low"], attempt["high"]
    mo.accordion(
        {
            "Answer": mo.md(
                f"""
                Catch rate (true-positive rate, TPR): {_lo["caught"]} / {_lo["poor"]} =
                {100 * _lo["caught"] / _lo["poor"]:.1f}% low, {_hi["caught"]} / {_hi["poor"]} =
                {100 * _hi["caught"] / _hi["poor"]:.1f}% high. False-alarm rate (FPR):
                {_lo["false_alarms"]:,} / {_lo["healthy"]:,} =
                {100 * _lo["false_alarms"] / _lo["healthy"]:.1f}% low,
                {_hi["false_alarms"]:,} / {_hi["healthy"]:,} =
                {100 * _hi["false_alarms"] / _hi["healthy"]:.1f}% high. The catch rates are
                close; the high quartile gets more flags it did not need. Whether that is unfair
                to it depends on what a flag means: here it brings work and attention (§3).
                """
            )
        }
    )
    return


@app.cell
def _(mo, np, test, train, val):
    # The statistics row: a score computed on the years the model was fitted to
    train_bias = {}
    for _name, _d in (("training", train), ("validation", val), ("test", test)):
        _y, _p = _d["poor_next"].to_numpy(), _d["p"].to_numpy()
        train_bias[_name] = {
            "tpr_pct": 100 * float((_d["flag"] & (_y == 1)).sum() / _y.sum()),
            "log_loss": float(-np.mean(_y * np.log(_p) + (1 - _y) * np.log1p(-_p))),
            "base_pct": 100 * float(_y.mean()),
            "mean_p_pct": 100 * float(_p.mean()),
        }
    # Maximum likelihood with an intercept: the training years' predicted chances sum to the
    # number that turned poor (the intercept's gradient is the sum of p - y)
    assert abs(train["p"].sum() - train["poor_next"].sum()) < 1e-6
    _tb = train_bias
    mo.md(f"""
    ### Five things called bias

    The same word means five things in this course; today is about the last, and the first four
    feed it.

    | Bias in … | Means | On these bridges |
    |:--|:--|:--|
    | the model | too simple a model for the truth | a score linear in 16 inputs (Lecture 6) |
    | statistics | an estimate that is off on average | a catch rate scored on the training years |
    | the data | training data unlike the data of use | trained on 2014 → 2018, used on 2020 → 2024 |
    | thinking | confirmation bias, automation bias | treating the flag list as the truth |
    | algorithms | unfair treatment of groups | today's question |

    **The statistics row, measured.** A score computed on the years a model was fitted to is
    optimistic on average for what the fit optimized: the log loss is
    {_tb["training"]["log_loss"]:.4f} on the training years against
    {_tb["validation"]["log_loss"]:.4f} (validation) and {_tb["test"]["log_loss"]:.4f} (test). The
    catch rate, which the fit did not optimize, does not show it in one cycle:
    {_tb["training"]["tpr_pct"]:.1f}% on the training years against
    {_tb["validation"]["tpr_pct"]:.1f}% and {_tb["test"]["tpr_pct"]:.1f}%. And the training years'
    mean predicted chance equals their base rate exactly ({_tb["training"]["mean_p_pct"]:.2f}%):
    maximum likelihood with an intercept sets it so, so calibration checked on the training years
    is no check at all.

    Protected attributes in law (race, sex, age, religion, and others) are not in the inventory;
    the vulnerability index stands for the communities a bridge program can underserve.
    """)
    return (train_bias,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: where bias enters

    The pipeline from inspection to preventive work, with a note under each step where a choice
    is made, by people or by code, and the checks this lecture builds.

    - **Labels:** "poor" is an inspector's rating, a stand-in for need; it does not count the
      traffic a bridge carries or the detour if it closes. A health-care algorithm that predicted
      cost as a stand-in for need gave less care to patients with the same need and less access
      (Obermeyer et al., 2019).
    - **Selection:** the inventory holds public highway bridges longer than 20 ft; Lecture 3 left
      culverts out.
    - **Proxies:** inputs that encode the county (tested next).
    - **The threshold:** one number trades misses against false alarms.
    - **Who benefits:** here a flag brings resources.
    - **Checks:** evaluate each group separately, with intervals; publish a model card and a
      datasheet (§6).
    """)
    return


@app.cell
def _(FULL, FancyBboxPatch, INK, PALETTE, R, plt):
    _light, _warn, _ok = "#e8f0f7", "#fbe3d4", "#e3f1e8"
    _steps = [("inspections", 0.12), ("inventory\n(NBI)", 0.36), ("model\n(Lecture 6)", 0.60),
              (f"flag:\np > 1/{1 + R}", 0.84), ("preventive\nwork", 1.08)]  # fmt: skip
    _notes = ["labels: an inspector's\nrating, a proxy for need",
              "selection: which bridges\nare in the inventory",
              "proxies: inputs that\nstand for the county",
              "the threshold: who\nis caught, who is missed",
              "who benefits: a flag\nbrings resources"]  # fmt: skip
    _fig, _ax = plt.subplots(figsize=(FULL[0] + 1.5, FULL[1]))
    _ax.set(xlim=(0, 1.22), ylim=(0, 1), xticks=[], yticks=[])
    for _sp in _ax.spines.values():
        _sp.set_visible(False)
    for _i, (_text, _x) in enumerate(_steps):
        _ax.add_patch(FancyBboxPatch((_x - 0.09, 0.62), 0.18, 0.18, boxstyle="round,pad=0.01",
                                     facecolor=_light, edgecolor=INK, lw=1))  # fmt: skip
        _ax.text(_x, 0.71, _text, ha="center", va="center", fontsize=10)
        if _i < len(_steps) - 1:
            _ax.annotate("", (_x + 0.15, 0.71), (_x + 0.09, 0.71),
                         arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2})  # fmt: skip
        _ax.text(_x, 0.45, _notes[_i], ha="center", va="center", fontsize=8, color=PALETTE[1],
                 bbox={"facecolor": _warn, "edgecolor": "none", "pad": 3})  # fmt: skip
    _ax.add_patch(FancyBboxPatch((0.03, 0.06), 1.14, 0.18, boxstyle="round,pad=0.01",
                                 facecolor=_ok, edgecolor=PALETTE[2], lw=1))  # fmt: skip
    _ax.text(0.6, 0.15, "checks: evaluate each group separately, with intervals; publish a "
             "model card and a datasheet", ha="center", va="center", fontsize=10,
             color=PALETTE[2])  # fmt: skip
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Unawareness is not enough

    The model never sees the index. Do its inputs carry it anyway? Train a classifier to tell a
    bridge in a top-quartile county (rank above 0.75) from the 16 inputs alone, on the 2014
    bridges, and score it on the 2020 bridges by its AUC: the chance that a random bridge from a
    top-quartile county scores above a random bridge from another county. Then on counties the
    classifier never saw: the 102 counties are split into 10 folds, stratified so that each fold
    holds two or three counties of every quartile; a classifier fitted on the other folds' 2014
    bridges (and standardized with them) scores the fold's 2020 bridges, and the AUC is computed
    within the fold. The mean over the 10 folds is one estimate; the folds are shuffled 10 times
    and the median is reported, with the range.

    Why not hold out one county at a time and pool the scores? The label is a property of the
    county, so leaving a top-quartile county out lowers the training share of top-quartile
    bridges just when that county is scored, and leaving another county out raises it. A
    classifier with no inputs at all, scored that way, ranks every bridge exactly backwards
    (§5 measures it). Within one fold, a classifier with no inputs scores every bridge alike:
    AUC 0.5, the chance level the folds' AUCs are read against.

    The figure script uses scikit-learn's
    `LogisticRegression(C=1.0, solver="newton-cholesky")`: an $L_2$ penalty with $C = 1$ (on the
    weights, not the intercept). Written out, it minimizes

    $$C\sum_i \Big[\ln\big(1 + e^{s_i}\big) - y_i s_i\Big] + \tfrac12\|\mathbf{w}\|^2,
      \qquad s_i = w_0 + \mathbf{w}^\top\mathbf{z}_i,$$

    whose Newton step adds $\mathbf{w}/C$ to the gradient and $\mathbf{I}/C$ (zero for $w_0$) to
    the Hessian. The AUC is the Mann–Whitney statistic: rank all scores (ties get their average
    rank), and

    $$\text{AUC} = \frac{\sum_{i:\,y_i=1}\operatorname{rank}_i - n_1(n_1+1)/2}{n_1 n_0}.$$
    """)
    return


@app.cell
def _(
    INPUTS,
    LogisticRegression,
    PROXY_C,
    PROXY_DROP,
    PROXY_FOLDS,
    PROXY_REPEATS,
    StratifiedKFold,
    expit,
    np,
    ref,
    roc_auc_score,
    test,
    time,
    train,
):
    def fit_ridge_logistic(
        Z: np.ndarray, y: np.ndarray, C: float = 1.0, tol: float = 1e-10, max_steps: int = 50
    ) -> np.ndarray:
        """L2-penalized logistic regression by Newton's method; the intercept is not penalized.

        Args:
            Z: Standardized inputs, shape ``(n, d)``.
            y: Labels in {0, 1}, shape ``(n,)``.
            C: Inverse penalty strength (scikit-learn's ``C``).
            tol: Stop when no weight moves more than this.
            max_steps: Give up after this many steps.

        Returns:
            Weights ``[w0, w]`` of shape ``(d + 1,)``.

        Raises:
            RuntimeError: If Newton's method has not converged after ``max_steps``.
        """
        A = np.column_stack([np.ones(len(Z)), Z])
        pen = np.r_[0.0, np.ones(Z.shape[1])] / C
        w = np.zeros(A.shape[1])
        for _ in range(max_steps):
            p = expit(A @ w)
            g = A.T @ (p - y) + pen * w
            H = A.T @ (A * (p * (1 - p))[:, None]) + np.diag(pen)
            delta = np.linalg.solve(H, g)
            w -= delta
            if np.abs(delta).max() < tol:
                return w
        raise RuntimeError("Newton did not converge")

    def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
        """The AUC as the Mann-Whitney statistic, with average ranks for ties.

        Args:
            y: Labels, boolean, shape ``(n,)``.
            s: Scores, shape ``(n,)``.

        Returns:
            The share of (positive, negative) pairs ranked the right way (ties count half).
        """
        _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        ranks = ((ends - counts + 1 + ends) / 2)[inv]  # 1-based average rank of each value
        n1 = int(y.sum())
        n0 = len(y) - n1
        return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def roc_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """False- and true-positive rates at every distinct threshold, from the top score down."""
        order = np.argsort(-s, kind="stable")
        ys, ss = y[order], s[order]
        last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]  # the end of each tied run
        tp, fp = np.cumsum(ys)[last], np.cumsum(~ys)[last]
        return np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]]

    def proxy_scores(cols: list[str], fit_rows, score_rows, scale_rows=None) -> np.ndarray:
        """Fit "top-quartile county" on ``fit_rows`` from ``cols``; score ``score_rows``.

        Args:
            cols: The inputs.
            fit_rows: The bridges to fit on, with ``quartile``.
            score_rows: The bridges to score.
            scale_rows: The bridges whose mean and spread standardize the inputs (default:
                ``fit_rows``).

        Returns:
            Each scored bridge's chance of being in a top-quartile county.
        """
        _X = (fit_rows if scale_rows is None else scale_rows)[cols].to_numpy(np.float64)
        _mu, _sd = _X.mean(axis=0), _X.std(axis=0)
        _Z = (fit_rows[cols].to_numpy(np.float64) - _mu) / _sd
        _w = fit_ridge_logistic(_Z, (fit_rows["quartile"] == "high").to_numpy(float))
        return expit(_w[0] + ((score_rows[cols].to_numpy(np.float64) - _mu) / _sd) @ _w[1:])

    PROXY_SETS = {"all": list(INPUTS), "no_location": [c for c in INPUTS if c not in PROXY_DROP]}
    y_high = (test["quartile"] == "high").to_numpy()
    # The counties, sorted, with their quartile: the folds split these
    county_q = test.groupby("fips")["quartile"].first().astype(str).sort_index()
    assert set(county_q.index) == set(train["fips"]) and len(county_q) == 102

    def county_fold_auc(cols: list[str], seed: int) -> float:
        """The mean within-fold AUC over ``PROXY_FOLDS`` folds of counties, one shuffle.

        Args:
            cols: The inputs.
            seed: The folds' shuffle (``random_state``).

        Returns:
            The mean of the folds' AUCs.
        """
        _names = county_q.index.to_numpy()
        _folds = StratifiedKFold(n_splits=PROXY_FOLDS, shuffle=True, random_state=seed)
        _aucs = []
        for _, _hold in _folds.split(_names, county_q.to_numpy()):
            _held = set(_names[_hold])
            _fit = train[~train["fips"].isin(_held)]
            _te = test["fips"].isin(_held).to_numpy()
            _aucs.append(auc_rank(y_high[_te], proxy_scores(cols, _fit, test[_te])))
        assert len(_aucs) == PROXY_FOLDS
        return float(np.mean(_aucs))

    proxy = {}
    _t0 = time.perf_counter()
    proxy_ref = ref.proxies(train, test)  # the figure script's: scikit-learn, newton-cholesky
    proxy_ref_seconds = time.perf_counter() - _t0
    _lbfgs_gap = {}
    for _name, _cols in PROXY_SETS.items():
        _s = proxy_scores(_cols, train, test)
        _folds = [county_fold_auc(_cols, _seed) for _seed in range(PROXY_REPEATS)]
        proxy[_name] = {
            "auc": auc_rank(y_high, _s),
            "auc_county_folds": _folds,
            "auc_county_folds_median": float(np.median(_folds)),
            "auc_county_folds_range": [min(_folds), max(_folds)],
            "roc": roc_points(y_high, _s),
        }
        # Checks: the AUC against scikit-learn's; the weights against scikit-learn's Newton
        # solver at a tight tolerance; every AUC against the figure script's proxies()
        assert abs(proxy[_name]["auc"] - roc_auc_score(y_high, _s)) < 1e-12
        _X = train[_cols].to_numpy(np.float64)
        _Z = (_X - _X.mean(axis=0)) / _X.std(axis=0)
        _yb = (train["quartile"] == "high").to_numpy()
        _w = fit_ridge_logistic(_Z, _yb.astype(float), C=PROXY_C)
        _nc = LogisticRegression(C=PROXY_C, l1_ratio=0.0, solver="newton-cholesky", tol=1e-12,
                                 max_iter=200).fit(_Z, _yb)  # fmt: skip
        assert np.allclose(_w, np.r_[_nc.intercept_, _nc.coef_[0]], rtol=1e-6, atol=1e-8)
        # scikit-learn's default solver, L-BFGS at tol=1e-4: how far it stops from the optimum
        _lb = LogisticRegression(C=PROXY_C, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
        _lb.fit(_Z, _yb)
        _lbfgs_gap[_name] = float(np.abs(_w - np.r_[_lb.intercept_, _lb.coef_[0]]).max())
        _mine = [proxy[_name]["auc"], *proxy[_name]["auc_county_folds"]]
        _theirs = [proxy_ref[_name]["auc"], *proxy_ref[_name]["auc_county_folds"]]
        for _a, _b in zip(_mine, _theirs, strict=True):
            assert abs(_a - _b) < 1e-4 and round(_a, 3) == round(_b, 3), (_name, _a, _b)
    proxy_lbfgs_gap = _lbfgs_gap
    print(
        "proxy AUC, 2020 bridges: "
        + ", ".join(
            f"{k} {v['auc']:.5f} (script {proxy_ref[k]['auc']:.5f}), counties held out: median "
            f"{v['auc_county_folds_median']:.5f} (script "
            f"{proxy_ref[k]['auc_county_folds_median']:.5f}), range "
            f"{v['auc_county_folds_range'][0]:.3f} to {v['auc_county_folds_range'][1]:.3f}"
            for k, v in proxy.items()
        )
        + f"; L-BFGS weights at most {max(_lbfgs_gap.values()):.3f} from the optimum; the "
        f"script's proxies() took {proxy_ref_seconds:.1f} s"
    )
    return (
        PROXY_SETS,
        auc_rank,
        county_q,
        fit_ridge_logistic,
        proxy,
        proxy_lbfgs_gap,
        proxy_ref,
        proxy_scores,
        y_high,
    )


@app.cell
def _(
    MUTED,
    PALETTE,
    PROXY_SETS,
    SIDE,
    auc_rank,
    check,
    mo,
    np,
    plt,
    proxy,
    proxy_scores,
    ref,
    rnd,
    test,
    train,
    y_high,
):
    _fig, _ax = plt.subplots(figsize=SIDE)
    for (_name, _label), _col, _ls in zip(
        (("all", "all 16 inputs"), ("no_location", "without latitude and longitude")),
        (PALETTE[0], PALETTE[3]), ("-", "--"), strict=True,
    ):  # fmt: skip
        _f, _t = proxy[_name]["roc"]
        _ax.plot(_f, _t, color=_col, ls=_ls, lw=1.8,
                 label=f"{_label}: AUC {proxy[_name]['auc']:.3f}")  # fmt: skip
    _ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    _ax.set(xlabel="false-positive rate", ylabel="true-positive rate")
    _ax.legend(fontsize=9, frameon=False, loc="lower right")
    _fig.tight_layout()
    # Which inputs carry the signal without location: drop each group, refit, rescore
    _groups = {
        "traffic (ADT, trucks)": ["log_adt", "pct_truck"],
        "material (4 indicators)": ["concrete", "steel", "prestressed", "timber"],
        "age (age, years since work)": ["age", "years_since_work"],
        "size (length, width)": ["log_length", "deck_width_m"],
        "ratings (deck, super, sub, lowest)": ["deck", "superstructure", "substructure", "rating"],
    }
    _base_cols = PROXY_SETS["no_location"]
    assert sorted(c for _cs in _groups.values() for c in _cs) == sorted(_base_cols)
    proxy_drop = {
        _g: auc_rank(y_high, proxy_scores([c for c in _base_cols if c not in _cs], train, test))
        for _g, _cs in _groups.items()
    }
    _rows = "\n".join(f"| {_g} | {v:.3f} |" for _g, v in proxy_drop.items())
    assert np.isfinite(list(proxy_drop.values())).all()
    # Dropping traffic costs the most AUC, then material
    assert sorted(proxy_drop, key=proxy_drop.get)[:2] == list(_groups)[:2]
    # Traffic and material on their own
    proxy_traffic_material = auc_rank(
        y_high, proxy_scores(_groups["traffic (ADT, trucks)"]
                             + _groups["material (4 indicators)"], train, test)
    )  # fmt: skip
    # The figure script's groups are the same, by their short names; the slides' values
    assert {_g.split()[0]: _cs for _g, _cs in _groups.items()} == {
        _k: list(_v) for _k, _v in ref.PROXY_GROUPS.items()
    }
    check({_g.split()[0]: rnd(_v, 3) for _g, _v in proxy_drop.items()},
          "proxies/no_location_drop")  # fmt: skip
    check(rnd(proxy_traffic_material, 3), "proxies/traffic_material_alone")
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    Without location the AUC is {proxy["no_location"]["auc"]:.3f}. Which inputs carry it? Drop
    one group at a time from the 14 inputs left, refit, and rescore:

    | Dropped | AUC without location |
    |:--|--:|
    {_rows}

    Traffic and material alone (six inputs) reach {proxy_traffic_material:.3f}, against
    {proxy["no_location"]["auc"]:.3f} with all 14. NBI Item 22 (the owner) is not one of the 16
    inputs. §5 asks whether the signal survives a county the classifier never saw.
    """),
        ]
    )
    return proxy_drop, proxy_traffic_material


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** a group's rates on one inspection cycle estimate what the flag rule will
    deliver to that group on the next, and the counties of a group are like independent draws
    from the counties such a group could hold. Bridges within one county are not independent:
    they share inspectors, budgets, and weather.

    **The confusion matrix, per group.** In group $g$ (a quartile, a county, a cell) with $n_g$
    bridges, $\mathrm{TP}_g$ were flagged and turned poor, $\mathrm{FN}_g$ were missed,
    $\mathrm{FP}_g$ were flagged and stayed healthy, $\mathrm{TN}_g$ were left alone and stayed
    healthy. Then

    $$p_g = \frac{\mathrm{TP}_g + \mathrm{FN}_g}{n_g}\ \text{(base rate)},\qquad
      \mathrm{TPR}_g = \frac{\mathrm{TP}_g}{\mathrm{TP}_g + \mathrm{FN}_g}\ \text{(catch rate)},$$

    $$\mathrm{FPR}_g = \frac{\mathrm{FP}_g}{\mathrm{FP}_g + \mathrm{TN}_g}\ \text{(false alarms)},
      \qquad
      \mathrm{PPV}_g = \frac{\mathrm{TP}_g}{\mathrm{TP}_g + \mathrm{FP}_g}\ \text{(precision)}.$$

    **Three ways to call a model fair** (with flag $\hat y$, outcome $y$, group $G$, score $s$):

    | Criterion | Requires, for every group $g$ |
    |:--|:--|
    | Demographic parity | $P(\hat y = 1 \mid G = g)$ the same: equal flag rates |
    | Equalized odds | $P(\hat y = 1 \mid y, G = g) = P(\hat y = 1 \mid y)$ for $y = 0, 1$: equal TPR and FPR |
    | Equal opportunity | the $y = 1$ half of equalized odds: equal catch rates |
    | Calibration within groups | $P(y = 1 \mid s, G = g) = s$; checked coarsely here as mean $p$ = base rate |

    **Which error is the harm here?** The same matrix means opposite things in two settings:

    | | A criminal risk score | Bridges (today) |
    |:--|:--|:--|
    | A flag means | "high risk": detention | preventive work, a closer look |
    | A false alarm | harms the person | spends money; helps the county |
    | A miss | releases a risk | a bridge turns poor unwatched |
    | The rate that matters most | false positives by group | misses: the catch rate by group |

    In risk scoring a flag is a burden, so unequal false-positive rates were the complaint
    (ProPublica, 2016). Here a flag brings resources, so the harm is a miss, and the criterion
    that fits is equal opportunity: equal catch rates among the bridges that will turn poor.

    **The identity that makes criteria conflict.** From the definitions,
    $\mathrm{FP} = \mathrm{TP}\,(1-\mathrm{PPV})/\mathrm{PPV}$,
    $\mathrm{TP} = \mathrm{TPR}\cdot n p$, and $\mathrm{FP} + \mathrm{TN} = n(1-p)$. Divide:

    $$\mathrm{FPR} = \frac{p}{1-p}\cdot\frac{1-\mathrm{PPV}}{\mathrm{PPV}}\cdot\mathrm{TPR}.$$

    It holds for any classifier and any group (Chouldechova, 2017). If two groups have the same
    precision and the same catch rate but different base rates, their false-alarm rates must
    differ in the ratio of $p/(1-p)$. Equal precision, equal catch rates, and equal false alarms
    together are possible only when base rates are equal, or in the degenerate cases
    ($\mathrm{PPV} = 1$, so no false alarms at all; or $\mathrm{TPR} = 0$). Kleinberg et al. (2017)
    prove the version for calibrated scores.

    **Intervals that resample counties.** For a group with $K$ counties, draw $K$ counties with
    replacement, pool their bridges, and recompute every rate; repeat $B = 1{,}000$ times and
    read the 2.5% and 97.5% points (a percentile bootstrap). Resampling bridges instead would
    treat every bridge as an independent case; resampling counties treats the county as the
    unit that varies, so the interval is wider when counties differ (§4 measures both).

    **Thresholds per group** (post-processing; Hardt et al., 2016). To give every group the
    catch rate $\tau$ on the validation years, sort group $g$'s validation bridges that turned
    poor by $p$, largest first, take $k = \lceil \tau\, n^{+}_g \rceil$, and set $t_g$ just
    below the $k$-th score, so $p > t_g$ flags at least $k$ of them. Using the group at the
    moment of decision is a policy choice, not only a technical one.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build and verify

    **Rates per group.** Every rate from the four counts; the mean predicted chance for
    calibration. Checked against the figure script's `rates` on the whole test set, each
    quartile, the validation years, and the two halves of the top quartile.
    """)
    return


@app.cell
def _(COOK, QUARTILES, agree_rates, np, pd, ref, test, val):
    def rates(g: pd.DataFrame, flag: str = "flag") -> dict:
        """The confusion-matrix rates of one group.

        Args:
            g: Bridges with ``poor_next`` (truth), the flag column, ``p``, and ``fips``.
            flag: Which flag column to use.

        Returns:
            Counts, the base rate, the flag rate, the catch rate (TPR), the false-alarm rate
            (FPR), the precision (PPV), the mean predicted chance, and the calibration gap (mean
            predicted chance minus base rate), as shares.
        """
        y, f = g["poor_next"].to_numpy() == 1, g[flag].to_numpy()
        tp, fn = int((f & y).sum()), int((~f & y).sum())
        fp, tn = int((f & ~y).sum()), int((~f & ~y).sum())
        _mean_p = float(g["p"].mean())
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
            "tpr": tp / (tp + fn) if tp + fn else np.nan,  # undefined: none turned poor
            "fpr": fp / max(fp + tn, 1),
            "ppv": tp / max(tp + fp, 1),
            "mean_p": _mean_p,
            "calib_gap": _mean_p - (tp + fn) / len(g),
        }

    overall = rates(test)
    by_q = {_q: rates(test[test["quartile"] == _q]) for _q in QUARTILES}
    overall_val = rates(val)
    by_q_val = {_q: rates(val[val["quartile"] == _q]) for _q in QUARTILES}
    _high = test[test["quartile"] == "high"]
    high_split = {
        "Cook County": rates(_high[_high["fips"] == COOK]),
        "the other 25": rates(_high[_high["fips"] != COOK]),
    }
    assert _high["fips"].nunique() == 26 and (high_split["the other 25"]["counties"] == 25)
    # Checks: the figure script's rates() on the same rows (its p differs from ours by under
    # 1e-12, so only mean_p can differ, in the last digits)
    _pairs = [(overall, test), (overall_val, val)]
    _pairs += [(by_q[_q], test[test["quartile"] == _q]) for _q in QUARTILES]
    _pairs += [(by_q_val[_q], val[val["quartile"] == _q]) for _q in QUARTILES]
    _pairs += [(high_split["Cook County"], _high[_high["fips"] == COOK])]
    for _mine, _rows in _pairs:
        assert agree_rates(_mine, ref.rates(_rows)), _mine
    assert sum(r["bridges"] for r in by_q.values()) == overall["bridges"]
    pd.DataFrame({_q: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                  for _q, r in {**by_q, "all": overall}.items()}).T  # fmt: skip
    return by_q, by_q_val, high_split, overall, overall_val, rates


@app.cell
def _(np):
    def agree_rates(a: dict, b: dict) -> bool:
        """Two rate dicts equal: counts exactly, shares to 1e-12 (mean p from two fits).

        An undefined catch rate (NaN, no bridge turned poor) equals an undefined one.
        """
        return set(a) == set(b) and all(
            a[k] == b[k] if isinstance(b[k], int) else bool(np.isclose(a[k], b[k], rtol=1e-12,
                                                                       atol=1e-15, equal_nan=True))
            for k in b
        )  # fmt: skip

    return (agree_rates,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    Before the next cells run:

    1. Will intervals that resample whole counties be wider or narrower than intervals that
       resample single bridges? By how much, for the top quartile's catch rate?
    2. The bootstrap draws counties for the four quartiles first, then for the 16
       quartile-by-owner cells the slides show, each list in sorted order, so no interval
       depends on the order of the file's rows. Each draw sums per-county counts instead of
       pooling the counties' bridges and recounting. Will its intervals equal the ones from
       pooled bridges exactly, or only to rounding?
    """)
    return


@app.cell
def _(BOOT, QUARTILES, SEED, np, test, time):
    STATS = ("tpr", "fpr", "flagged", "base", "mean_p", "calib_gap")

    def county_totals(g) -> tuple[np.ndarray, np.ndarray]:
        """Per-county counts of a group, counties sorted by FIPS code (the script's order).

        Args:
            g: One group's bridges with ``fips``, ``poor_next``, ``flag``, and ``p``.

        Returns:
            The sorted county codes, and a ``(K, 6)`` array of each county's TP, FN, FP, TN,
            bridges, and sum of p.
        """
        names, inv = np.unique(g["fips"].to_numpy(dtype=str), return_inverse=True)
        y, f = g["poor_next"].to_numpy() == 1, g["flag"].to_numpy(dtype=bool)
        cols = [f & y, ~f & y, f & ~y, ~f & ~y, np.ones(len(g), dtype=bool)]
        C = np.column_stack(
            [np.bincount(inv, weights=c.astype(float), minlength=len(names)) for c in cols]
            + [np.bincount(inv, weights=g["p"].to_numpy(), minlength=len(names))]
        )
        assert C.shape == (len(names), 6), C.shape
        return names, C

    def shares(T: np.ndarray) -> dict[str, np.ndarray]:
        """Rates from pooled counts ``T`` of shape ``(B, 6)``, as the script's rates() computes."""
        tp, fn, fp, tn, n, sum_p = T.T
        with np.errstate(invalid="ignore"):  # 0 / 0: a draw with no bridge that turned poor
            tpr = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        return {
            "tpr": tpr,
            "fpr": fp / np.maximum(fp + tn, 1),
            "flagged": (tp + fp) / n,
            "base": (tp + fn) / n,
            "mean_p": sum_p / n,
            "calib_gap": sum_p / n - (tp + fn) / n,  # mean predicted minus turned poor
        }

    def cluster_draws(g, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
        """``BOOT`` county resamples of one group: each rate per draw.

        Each draw picks the group's ``K`` counties with replacement (``rng.choice`` over the
        sorted codes, one call per draw, as the figure script does) and pools their counts.

        Args:
            g: One group's bridges.
            rng: The generator, advanced by the draws.

        Returns:
            Each rate's ``BOOT`` values, and a mask of the draws that held no bridge that turned
            poor (their catch rate is undefined, NaN).
        """
        names, C = county_totals(g)
        idx = np.stack([rng.choice(len(names), len(names), replace=True) for _ in range(BOOT)])
        assert idx.shape == (BOOT, len(names)), idx.shape
        T = C[idx].sum(axis=1)
        assert T.shape == (BOOT, 6), T.shape
        return shares(T), T[:, 0] + T[:, 1] == 0

    def interval(draws: dict) -> dict[str, list[float]]:
        """The 2.5% and 97.5% points of each rate's draws, undefined draws left out."""
        return {k: np.nanpercentile(draws[k], [2.5, 97.5]).tolist() for k in STATS}

    # One generator, drawn in the figure script's order: the quartiles in their own order, then
    # the cells the slides show ("other" owners left out), sorted by name; neither depends on
    # the order of the file's rows
    _t0 = time.perf_counter()
    _rng = np.random.default_rng(SEED)
    q_order = list(QUARTILES)
    cell_order = sorted(_c for _c in set(test["cell"]) if not _c.endswith("| other"))
    assert len(cell_order) == 16
    boot_state, boot_draws, boot_zero = {}, {}, {}
    for _key, _col in [(_q, "quartile") for _q in q_order] + [(_c, "cell") for _c in cell_order]:
        boot_state[_key] = _rng.bit_generator.state  # to replay one group in the check below
        boot_draws[_key], boot_zero[_key] = cluster_draws(test[test[_col] == _key], _rng)
    boot_ci = {_k: interval(_d) for _k, _d in boot_draws.items()}
    boot_seconds = time.perf_counter() - _t0
    print(f"{len(boot_draws)} groups x {BOOT:,} county resamples in {boot_seconds:.2f} s; "
          f"quartile order {q_order}")  # fmt: skip
    return (
        STATS,
        boot_ci,
        boot_draws,
        boot_seconds,
        boot_state,
        boot_zero,
        cell_order,
        county_totals,
        q_order,
    )


@app.cell
def _(
    STATS,
    boot_ci,
    boot_draws,
    boot_state,
    boot_zero,
    cell_order,
    county_totals,
    np,
    pd,
    q_order,
    ref,
    test,
    time,
):
    # Check 1: the figure script's own cluster_ci(), replayed from the generator's state at the
    # start of each group, for every group the slides show
    _t0 = time.perf_counter()
    _groups = [(_q, "quartile") for _q in q_order] + [(_c, "cell") for _c in cell_order]
    for _key, _col in _groups:
        _rng = np.random.Generator(np.random.PCG64())
        _rng.bit_generator.state = boot_state[_key]
        _theirs, _undefined = ref.cluster_ci(test[test[_col] == _key], _rng)
        assert _undefined == int(boot_zero[_key].sum()), _key
        for _k in STATS:
            if _k in ("mean_p", "calib_gap"):  # a sum of p in another order: to rounding
                assert np.allclose(boot_ci[_key][_k], _theirs[_k], rtol=1e-12, atol=1e-15), _key
            else:  # integer counts: exactly equal
                assert boot_ci[_key][_k] == _theirs[_k], (_key, _k)
    # Check 2: summing per-county counts equals pooling the drawn counties' bridges and
    # recounting with the script's rates(), draw by draw (the first 200 draws of one cell)
    _key = "high | township"
    _g = test[test["cell"] == _key]
    _names, _ = county_totals(_g)
    _by_county = dict(iter(_g.groupby("fips")))
    _rng = np.random.Generator(np.random.PCG64())
    _rng.bit_generator.state = boot_state[_key]
    for _b in range(200):
        _pick = _rng.choice(len(_names), len(_names), replace=True)
        _r = ref.rates(pd.concat([_by_county[_names[_i]] for _i in _pick]))
        for _k in STATS:
            _mine = boot_draws[_key][_k][_b]
            assert np.isclose(_mine, _r[_k], rtol=1e-12, atol=1e-15, equal_nan=True), (_b, _k)
    ci_check_seconds = time.perf_counter() - _t0
    print(f"county intervals equal to the figure script's cluster_ci for all {len(_groups)} "
          f"groups (counts exactly; mean p to 1e-12); 200 draws of {_key!r} equal to pooling "
          f"the drawn counties' bridges ({ci_check_seconds:.1f} s)")  # fmt: skip
    return (ci_check_seconds,)


@app.cell
def _(BOOT, QUARTILES, SEED, boot_ci, by_q, np, pd, test):
    # The bracket for the county intervals: resample single bridges instead (another generator,
    # so the slides' stream is untouched)
    _rng = np.random.default_rng(SEED + 1)
    bridge_ci = {}
    for _q in QUARTILES:
        _g = test[test["quartile"] == _q]
        _y = _g["poor_next"].to_numpy() == 1
        _f = _g["flag"].to_numpy()
        _idx = _rng.integers(0, len(_g), (BOOT, len(_g)))
        _tpr = (_f & _y)[_idx].sum(axis=1) / np.maximum(_y[_idx].sum(axis=1), 1)
        bridge_ci[_q] = np.percentile(_tpr, [2.5, 97.5]).tolist()
    width_ratio = {
        _q: (boot_ci[_q]["tpr"][1] - boot_ci[_q]["tpr"][0]) / (bridge_ci[_q][1] - bridge_ci[_q][0])
        for _q in QUARTILES
    }
    assert all(v > 1 for v in width_ratio.values())  # counties vary more than bridges do
    pd.DataFrame(
        {
            "catch rate %": [100 * by_q[_q]["tpr"] for _q in QUARTILES],
            "counties resampled": [
                f"{100 * boot_ci[_q]['tpr'][0]:.1f} to {100 * boot_ci[_q]['tpr'][1]:.1f}"
                for _q in QUARTILES
            ],
            "bridges resampled": [
                f"{100 * bridge_ci[_q][0]:.1f} to {100 * bridge_ci[_q][1]:.1f}" for _q in QUARTILES
            ],
            "width ratio": [round(width_ratio[_q], 2) for _q in QUARTILES],
        },
        index=list(QUARTILES),
    ).round(1)
    return bridge_ci, width_ratio


@app.cell
def _(QUARTILES, by_q, np, ref):
    def chouldechova(r: dict) -> float:
        """The false-alarm rate implied by the base rate, precision, and catch rate.

        Args:
            r: A group's rates (``base``, ``ppv``, ``tpr``).

        Returns:
            p / (1 - p) * (1 - PPV) / PPV * TPR.
        """
        b, ppv = r["base"], r["ppv"]
        return b / (1 - b) * (1 - ppv) / ppv * r["tpr"]

    identity = {_q: {"fpr": by_q[_q]["fpr"], "implied": chouldechova(by_q[_q])} for _q in QUARTILES}
    for _q, _v in identity.items():  # the identity holds to round-off, in every quartile
        assert abs(_v["fpr"] - _v["implied"]) < 1e-12, _q
        assert np.isclose(_v["implied"], ref.chouldechova(by_q[_q]), rtol=1e-14, atol=0)
    print(
        {_q: f"FPR {100 * v['fpr']:.4f}% = {100 * v['implied']:.4f}%" for _q, v in identity.items()}
    )
    return chouldechova, identity


@app.cell
def _(QUARTILES, R, np, pd, rates, ref, test, val):
    def group_thresholds(val_rows: pd.DataFrame, target: float) -> dict[str, float]:
        """Per-quartile thresholds, chosen on the validation years, that catch ``target`` of each.

        Args:
            val_rows: Validation transitions (2016 -> 2020) with ``quartile``, ``p``,
                ``poor_next``.
            target: The catch rate to reach in every quartile.

        Returns:
            For each quartile, the largest threshold whose validation catch rate is at least
            ``target``: just below the k-th largest p among those that turned poor,
            k = ceil(target * their number).
        """
        out = {}
        for _q in QUARTILES:
            _g = val_rows[val_rows["quartile"] == _q]
            _pos = np.sort(_g.loc[_g["poor_next"] == 1, "p"].to_numpy())[::-1]
            _k = int(np.ceil(target * len(_pos)))
            out[_q] = float(np.nextafter(_pos[_k - 1], -np.inf))  # p > threshold flags it
        return out

    def expected_cost(flag: np.ndarray, y: np.ndarray, r: float = R) -> float:
        """Cost per bridge: 1 per false alarm, ``r`` per miss (Lecture 6)."""
        assert flag.dtype == bool, flag.dtype
        return float((r * ((~flag) & (y == 1)).sum() + (flag & (y == 0)).sum()) / len(y))

    target = rates(val)["tpr"]  # the single threshold's validation catch rate
    thresholds = group_thresholds(val, target)
    test_post = test.assign(flag_group=test["p"] > test["quartile"].map(thresholds))
    _y = test_post["poor_next"].to_numpy()
    post = {
        "target": target,
        "thresholds": thresholds,
        "one": {_q: rates(test_post[test_post["quartile"] == _q]) for _q in QUARTILES},
        "group": {
            _q: rates(test_post[test_post["quartile"] == _q], "flag_group") for _q in QUARTILES
        },
        "flags_one": int(test_post["flag"].sum()),
        "flags_group": int(test_post["flag_group"].sum()),
        "caught_one": int((test_post["flag"] & (_y == 1)).sum()),
        "caught_group": int((test_post["flag_group"] & (_y == 1)).sum()),
        "cost_one": expected_cost(test_post["flag"].to_numpy(), _y),
        "cost_group": expected_cost(test_post["flag_group"].to_numpy(), _y),
    }
    # Each threshold reaches the target on validation; at the k-th score itself it would not
    for _q, _th in thresholds.items():
        _g = val[val["quartile"] == _q]
        _pos = _g.loc[_g["poor_next"] == 1, "p"].to_numpy()
        assert (_pos > _th).mean() >= target > (_pos > np.nextafter(_th, np.inf)).mean(), _q
    # Checks: the figure script's post_processing(), which calls its group_thresholds()
    _theirs = ref.post_processing(val, test)
    assert _theirs["target"] == target
    assert all(np.isclose(_theirs["thresholds"][_q], thresholds[_q], rtol=1e-12, atol=0)
               for _q in QUARTILES)  # fmt: skip
    for _k in ("flags_one", "flags_group", "caught_one", "caught_group"):
        assert _theirs[_k] == post[_k], _k
    assert np.isclose(_theirs["cost_group"], post["cost_group"], rtol=1e-12)
    assert np.array_equal(
        test_post["flag_group"].to_numpy(),
        (test["p"] > test["quartile"].map(_theirs["thresholds"])).to_numpy(),
    )
    print({_q: round(v, 5) for _q, v in thresholds.items()}, f"target {100 * target:.2f}%")
    return post, target, test_post, thresholds


@app.cell
def _(
    QUARTILES,
    agree_rates,
    boot_ci,
    boot_draws,
    boot_zero,
    cell_order,
    check,
    np,
    rates,
    ref,
    test,
):
    # The intersectional cells: quartile by owner (NBI Item 22), each with its county interval
    cells = {_c: {**rates(test[test["cell"] == _c]), "ci": boot_ci[_c]} for _c in cell_order}
    assert len(cells) == 16 and {c.split(" | ")[0] for c in cells} == set(QUARTILES)
    for _c, _r in cells.items():  # check: the figure script's rates() on each cell's rows
        _theirs = ref.rates(test[test["cell"] == _c])
        assert agree_rates({k: v for k, v in _r.items() if k != "ci"}, _theirs), _c
    cells_zero_draws = {_c: int(boot_zero[_c].sum()) for _c in cells}
    # every group with such draws, as the slides' numbers.json lists them (the quartiles have
    # none); the script stops if 2.5% or more of a group's draws have no catch rate
    _zero = {_k: int(_z.sum()) for _k, _z in boot_zero.items() if _z.any()}
    assert all(_n < 0.025 * len(boot_zero[_k]) for _k, _n in _zero.items())
    check(_zero, "tpr_undefined_draws")
    # Those draws hold no bridge that turned poor, so their catch rate is undefined and the
    # interval leaves them out. A guard that scored them 0 instead, max(tp + fn, 1) (the figure
    # script's until November 20), would give:
    zero_draw_effect = {
        _c: {
            "draws": n,
            "ci_tpr": boot_ci[_c]["tpr"],
            "ci_tpr_as_zero": np.percentile(
                np.nan_to_num(boot_draws[_c]["tpr"], nan=0.0), [2.5, 97.5]
            ).tolist(),
        }
        for _c, n in cells_zero_draws.items()
        if n
    }
    print(f"{len(cells)} cells; draws with no bridge that turned poor: {cells_zero_draws}; "
          f"effect on the catch-rate interval: {zero_draw_effect}")  # fmt: skip
    return cells, cells_zero_draws, zero_draw_effect


@app.cell
def _(
    BOOT,
    COOK,
    POOR,
    PROXY_C,
    PROXY_FOLDS,
    PROXY_REPEATS,
    QUARTILES,
    R,
    SEED,
    THRESHOLD,
    boot_ci,
    by_q,
    by_q_val,
    cells,
    cells_zero_draws,
    check,
    high_split,
    identity,
    overall,
    overall_val,
    pct,
    post,
    proxy,
    proxy_drop,
    proxy_traffic_material,
    rates,
    rnd,
    svi,
    test,
    val,
):
    def rounded(r: dict) -> dict:
        """A group's rates as numbers.json holds them: counts, and shares as percentages."""
        out = {k: r[k] for k in ("bridges", "counties", "turned_poor", "tp", "fn", "fp", "tn")}
        out["healthy"] = r["fp"] + r["tn"]
        for k in ("base", "flagged", "tpr", "fpr", "ppv", "mean_p", "calib_gap"):
            out[f"{k}_pct"] = pct(r[k], 1)
        if "ci" in r:
            out["ci_pct"] = {k: [pct(v[0], 1), pct(v[1], 1)] for k, v in r["ci"].items()}
        return out

    _high = test[test["quartile"] == "high"]
    _high_val = val[val["quartile"] == "high"]
    _split_val = {
        "Cook County": rates(_high_val[_high_val["fips"] == COOK]),
        "the other 25": rates(_high_val[_high_val["fips"] != COOK]),
    }
    _cook_test = high_split["Cook County"]
    _n2020 = test["quartile"].value_counts()
    _cnt = svi.groupby("quartile").agg(
        counties=("fips", "size"), pop=("pop", "sum"), lo=("rpl", "min"), hi=("rpl", "max")
    )
    numbers = {
        "svi": {
            _q: {
                "counties": int(_cnt.loc[_q, "counties"]),
                "population": int(_cnt.loc[_q, "pop"]),
                "rpl_range": [rnd(_cnt.loc[_q, "lo"], 4), rnd(_cnt.loc[_q, "hi"], 4)],
                "bridges_per_10k": rnd(_n2020[_q] / _cnt.loc[_q, "pop"] * 1e4, 1),
            }
            for _q in QUARTILES
        },
        "cook": {
            "rpl": rnd(svi.loc[svi["fips"] == COOK, "rpl"].iloc[0], 4),
            "bridges": int((test["fips"] == COOK).sum()),
            "share_of_high_pct": pct((_high["fips"] == COOK).mean(), 0),
        },
        "threshold": rnd(THRESHOLD, 4),
        "R": R,
        "overall": rounded(overall),
        "by_quartile": {_q: rounded({**by_q[_q], "ci": boot_ci[_q]}) for _q in QUARTILES},
        "by_quartile_val": {_q: rounded(by_q_val[_q]) for _q in QUARTILES},
        "overall_val": rounded(overall_val),
        "high_split": {k: rounded(v) for k, v in high_split.items()},
        "high_split_val": {
            k: {**rounded(v), "expected": rnd(v["mean_p"] * v["bridges"], 1)}
            for k, v in _split_val.items()
        },
        "cook_expected_test": rnd(_cook_test["mean_p"] * _cook_test["bridges"], 1),
        "identity": {_q: {k: pct(x, 2) for k, x in v.items()} for _q, v in identity.items()},
        "proxies": {
            **{
                k: {
                    "auc": rnd(v["auc"], 3),
                    "auc_county_folds": [rnd(a, 3) for a in v["auc_county_folds"]],
                    "auc_county_folds_median": rnd(v["auc_county_folds_median"], 3),
                    "auc_county_folds_range": [rnd(a, 3) for a in v["auc_county_folds_range"]],
                    "signal_kept_pct": pct(
                        (v["auc_county_folds_median"] - 0.5) / (v["auc"] - 0.5), 0
                    ),
                }
                for k, v in proxy.items()
            },
            "no_location_drop": {_g.split()[0]: rnd(_v, 3) for _g, _v in proxy_drop.items()},
            "traffic_material_alone": rnd(proxy_traffic_material, 3),
        },
        "identity_figure": {
            "precisions": [0.09, 0.11, 0.13],  # the figure's curves, near the quartiles'
            "catch_rate": rnd(overall["tpr"], 2),
        },
        "post": {
            "target_pct": pct(post["target"], 1),
            "thresholds": {_q: rnd(v, 4) for _q, v in post["thresholds"].items()},
            "one": {_q: rounded(post["one"][_q]) for _q in QUARTILES},
            "group": {_q: rounded(post["group"][_q]) for _q in QUARTILES},
            **{k: post[k] for k in ("flags_one", "flags_group", "caught_one", "caught_group")},
            "cost_one": rnd(post["cost_one"], 3),
            "cost_group": rnd(post["cost_group"], 3),
        },
        "cells": {k: rounded(v) for k, v in cells.items()},
        "cells_under_30_poor": sum(v["turned_poor"] < 30 for v in cells.values()),
        "tpr_undefined_draws": {k: n for k, n in cells_zero_draws.items() if n},
        "settings": {
            "seed": SEED, "boot": BOOT, "poor": POOR, "proxy_c": PROXY_C,
            "proxy_folds": PROXY_FOLDS, "proxy_repeats": PROXY_REPEATS,
        },
    }  # fmt: skip
    check(numbers, "")  # the whole numbers.json, key for key

    def _count(x: object) -> int:
        if isinstance(x, dict):
            return sum(_count(v) for v in x.values())
        if isinstance(x, list):
            return sum(_count(v) for v in x)
        return 1

    numbers_checked = _count(numbers)
    print(f"numbers.json rebuilt from scratch: all {numbers_checked} values equal the slides'")
    return numbers, numbers_checked, rounded


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: is the gap real?

    On the test years, the model catches a larger share of the bridges that turned poor in the
    low-mid quartile than in the mid-high quartile (the next cell prints both). Predict: does this
    gap mean the model serves mid-high counties worse? What would convince you? Write it down
    before you open the answer.
    """)
    return


@app.cell
def _(by_q, mo):
    mo.accordion(
        {
            "What to ask": mo.md(
                f"""
                Low-mid: {100 * by_q["low-mid"]["tpr"]:.1f}% of {by_q["low-mid"]["turned_poor"]};
                mid-high: {100 * by_q["mid-high"]["tpr"]:.1f}% of
                {by_q["mid-high"]["turned_poor"]}. Ask how many bridges stand behind each rate,
                how many counties (the unit that varies), and how much the rate would move with
                other counties or another cycle. The next three cells answer each: the county
                intervals, the interval of the gap itself, and the last cycle.
                """
            )
        }
    )
    return


@app.cell
def _(
    CDC_NOTE,
    FULL,
    INK,
    MUTED,
    PALETTE,
    QUARTILES,
    Q_COLOR,
    boot_ci,
    boot_draws,
    by_q,
    mo,
    np,
    overall,
    plt,
):
    _fig, _axes = plt.subplots(1, 4, figsize=FULL, sharex=True)
    _x = np.arange(len(QUARTILES))
    for _ax, (_key, _title) in zip(
        _axes[:3], (("flagged", "flagged, %"), ("tpr", "caught (TPR), %"),
                    ("fpr", "false alarms (FPR), %")), strict=True,
    ):  # fmt: skip
        _v = np.array([by_q[_q][_key] for _q in QUARTILES]) * 100
        _lo = np.array([boot_ci[_q][_key][0] for _q in QUARTILES]) * 100
        _hi = np.array([boot_ci[_q][_key][1] for _q in QUARTILES]) * 100
        _ax.bar(_x, _v, color=[Q_COLOR[_q] for _q in QUARTILES], edgecolor=INK, lw=0.5)
        _ax.errorbar(_x, _v, yerr=[_v - _lo, _hi - _v], fmt="none", color=INK, capsize=3)
        _ax.axhline(overall[_key] * 100, color=PALETTE[1], ls="--", lw=1)
        _ax.set_title(_title, fontsize=11)
    _base = np.array([by_q[_q]["base"] for _q in QUARTILES]) * 100
    _mp = np.array([by_q[_q]["mean_p"] for _q in QUARTILES]) * 100
    _axes[3].bar(_x - 0.2, _base, 0.4, color=MUTED, label="turned poor")
    _axes[3].bar(_x + 0.2, _mp, 0.4, color=PALETTE[1], hatch="//", label="mean predicted")
    _axes[3].set_title("calibration, %", fontsize=11)
    _axes[3].set_ylim(0, 1.5 * max(_base.max(), _mp.max()))
    _axes[3].legend(fontsize=7, frameon=False, loc="upper left")
    for _a in _axes:
        _a.set_xticks(_x, ["low", "low-\nmid", "mid-\nhigh", "high"], fontsize=9)
    _fig.tight_layout()

    # Overlapping intervals are not a test of a difference: the interval of the gap itself, from
    # the same draws (the two quartiles were resampled independently, so draw b of one pairs
    # with draw b of the other)
    _gap = boot_draws["low-mid"]["tpr"] - boot_draws["mid-high"]["tpr"]
    gap_ci = np.percentile(_gap, [2.5, 97.5]).tolist()
    gap_share_above_zero = float((_gap > 0).mean())
    overlap = [max(boot_ci[_q]["tpr"][0] for _q in QUARTILES),
               min(boot_ci[_q]["tpr"][1] for _q in QUARTILES)]  # fmt: skip
    assert overlap[0] < overlap[1]  # one stretch lies inside all four catch-rate intervals
    # The calibration gap by quartile (D4 of the 11-20 review): over-predicted in low-mid and
    # high, whose intervals exclude zero; the other two contain it
    _cg = {_q: boot_ci[_q]["calib_gap"] for _q in QUARTILES}
    assert all(_cg[_q][0] > 0 for _q in ("low-mid", "high"))
    assert all(_cg[_q][0] < 0 < _cg[_q][1] for _q in ("low", "mid-high"))
    assert overall["mean_p"] > overall["base"]
    _calib_rows = "; ".join(
        f"{_q} {100 * by_q[_q]['calib_gap']:+.2f} ({100 * _cg[_q][0]:+.1f} to "
        f"{100 * _cg[_q][1]:+.1f})"
        for _q in QUARTILES
    )
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    ### Catch rates differ; intervals overlap

    By SVI quartile on the test years, with 95% intervals that resample whole counties; dashed:
    all of Illinois. *{CDC_NOTE}*

    The catch rates' intervals share the stretch {100 * overlap[0]:.1f}% to
    {100 * overlap[1]:.1f}%. **The gap itself:** low-mid minus mid-high is
    {100 * (by_q["low-mid"]["tpr"] - by_q["mid-high"]["tpr"]):.1f} points, with a county-resampled
    95% interval of {100 * gap_ci[0]:.1f} to {100 * gap_ci[1]:.1f} points
    ({100 * gap_share_above_zero:.1f}% of the paired draws are above zero). Overlapping 95%
    intervals do not by themselves show that a difference is noise; the gap's own interval is
    the check.

    **Calibration, the same way.** Mean predicted chance minus the share that turned poor, in
    points, with its county interval (from the same draws):
    {_calib_rows}. The model over-predicts this cycle as a whole
    ({100 * overall["mean_p"]:.2f}% predicted, {100 * overall["base"]:.2f}% turned poor), and
    most in the low-mid and high quartiles, whose intervals lie above zero; the other two are
    about calibrated.
    """),
        ]
    )
    return gap_ci, gap_share_above_zero, overlap


@app.cell
def _(QUARTILES, boot_ci, by_q, by_q_val, mo, overall_val):
    _rows = "\n".join(
        f"| {_q.capitalize()} | {100 * by_q_val[_q]['tpr']:.1f}% | {100 * by_q[_q]['tpr']:.1f}% | "
        f"{100 * boot_ci[_q]['tpr'][0]:.1f} to {100 * boot_ci[_q]['tpr'][1]:.1f}% |"
        for _q in QUARTILES
    )
    val_inside = {
        _q: boot_ci[_q]["tpr"][0] <= by_q_val[_q]["tpr"] <= boot_ci[_q]["tpr"][1]
        for _q in QUARTILES
    }
    _spread_val = 100 * (max(r["tpr"] for r in by_q_val.values())
                         - min(r["tpr"] for r in by_q_val.values()))  # fmt: skip
    _spread_test = 100 * (max(r["tpr"] for r in by_q.values())
                          - min(r["tpr"] for r in by_q.values()))  # fmt: skip
    mo.md(f"""
    ### Last cycle, the quartiles were equal

    The same model on the validation years (2016 → 2020), the cycle before the test:

    | SVI quartile | Caught, validation (2016 → 2020) | Caught, test (2020 → 2024) | Test interval (counties) |
    |:--|--:|--:|--:|
    {_rows}

    Every quartile caught between {100 * min(r["tpr"] for r in by_q_val.values()):.1f}% and
    {100 * max(r["tpr"] for r in by_q_val.values()):.1f}% last cycle, around the
    {100 * overall_val["tpr"]:.1f}% overall: a spread of {_spread_val:.1f} points, against
    {_spread_test:.1f} on the test years. Each quartile's validation rate lies inside its own
    test-year interval: {all(val_inside.values())}. The bracket for a disparity is its own
    year-to-year noise.
    """)
    return (val_inside,)


@app.cell
def _(
    CDC_NOTE,
    COOK,
    FULL,
    MUTED,
    PALETTE,
    Q_COLOR,
    high_split,
    mo,
    np,
    plt,
    rates,
    svi,
    test,
    val,
):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=FULL)
    _names = list(high_split)
    _v = [100 * high_split[_n]["fpr"] for _n in _names]
    _bars = _a.bar(_names, _v, color=[PALETTE[1], Q_COLOR["high"]])
    _bars[0].set_hatch("//")
    for _i, _val in enumerate(_v):
        _a.text(_i, _val + 1, f"{_val:.1f}", ha="center", fontsize=10)
    _a.set(title="false alarms (FPR), %", ylim=(0, max(_v) * 1.2))
    _x = np.arange(len(_names))
    _b.bar(_x - 0.2, [100 * high_split[_n]["base"] for _n in _names], 0.4, color=MUTED,
           label="turned poor")  # fmt: skip
    _b.bar(_x + 0.2, [100 * high_split[_n]["mean_p"] for _n in _names], 0.4, color=PALETTE[1],
           hatch="//", label="mean predicted")  # fmt: skip
    _b.set_xticks(_x, _names)
    _b.set(title="calibration, %")
    _b.legend(fontsize=9, frameon=False)
    _fig.tight_layout()

    # What sets Cook apart: traffic and owners
    _high = test[test["quartile"] == "high"]
    _cook, _rest = _high[_high["fips"] == COOK], _high[_high["fips"] != COOK]
    cook_facts = {
        "adt_median_cook": float(_cook["adt"].median()),
        "adt_median_rest": float(_rest["adt"].median()),
        "adt_median_all": float(test["adt"].median()),
        "owner_share_cook": _cook["owner_name"].value_counts(normalize=True).to_dict(),
    }
    # The other 25: where they are and how many people live there
    _pos = test.groupby("fips")["lat"].mean()
    _o = svi[(svi["quartile"] == "high") & (svi["fips"] != COOK)].assign(
        lat=lambda d: d["fips"].map(_pos), density=lambda d: d["pop"] / d["area_sqmi"]
    )
    cook_facts["rest_south_of_39"] = int((_o["lat"] < 39.0).sum())
    cook_facts["rest_over_100k"] = sorted(_o.loc[_o["pop"] > 100_000, "county_name"])
    cook_facts["rest_names"] = sorted(_o["county_name"])
    cook_facts["rest_under_100_per_sqmi"] = int((_o["density"] < 100).sum())

    # Over-prediction in Cook: beyond the chance of independent bridges? The exact
    # Poisson-binomial distribution of the number that turn poor, if each did so with its p
    def _poisson_binomial(p: np.ndarray) -> np.ndarray:
        """P(k of the bridges turn poor), k = 0..n, for independent chances ``p``."""
        dist = np.array([1.0])
        for _pi in p:
            dist = np.convolve(dist, [1 - _pi, _pi])
        assert abs(dist.sum() - 1) < 1e-9
        return dist

    cook_calib = {}
    _val_high = val[val["quartile"] == "high"]
    for _name, _rows, _others in (
        ("test", _cook, _rest),
        ("validation", _val_high[_val_high["fips"] == COOK], _val_high[_val_high["fips"] != COOK]),
    ):
        _r = rates(_rows)
        _dist = _poisson_binomial(_rows["p"].to_numpy())
        cook_calib[_name] = {
            "bridges": _r["bridges"], "turned_poor": _r["turned_poor"],
            "expected": float(_rows["p"].sum()), "base_pct": 100 * _r["base"],
            "mean_p_pct": 100 * _r["mean_p"],
            "p_at_most_observed": float(_dist[: _r["turned_poor"] + 1].sum()),
            "p_at_least_observed": float(_dist[_r["turned_poor"]:].sum()),
            "fpr_pct": 100 * _r["fpr"], "fpr_others_pct": 100 * rates(_others)["fpr"],
        }  # fmt: skip
    _tbl = "\n".join(
        f"| {_k} | {_c['bridges']:,} | {_c['turned_poor']} | {_c['expected']:.1f} | "
        f"{_c['base_pct']:.1f}% | {_c['mean_p_pct']:.1f}% | {_c['p_at_most_observed']:.1e} | "
        f"{_c['p_at_least_observed']:.1e} | {_c['fpr_pct']:.1f}% | {_c['fpr_others_pct']:.1f}% |"
        for _k, _c in cook_calib.items()
    )
    _over = {_k: _c["expected"] > _c["turned_poor"] for _k, _c in cook_calib.items()}
    for _c in cook_calib.values():  # "about twice the other 25 counties' in both cycles"
        assert 1.7 < _c["fpr_pct"] / _c["fpr_others_pct"] < 2.3, _c
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    ### One county drives the false alarms

    The top quartile split in two. *{CDC_NOTE}*

    **What sets Cook apart.** Median traffic {cook_facts["adt_median_cook"]:,.0f} vehicles a day
    on Cook's bridges, against {cook_facts["adt_median_rest"]:,.0f} on the other 25 counties'
    and {cook_facts["adt_median_all"]:,.0f} statewide; owners:
    {", ".join(f"{k} {100 * v:.0f}%" for k, v in cook_facts["owner_share_cook"].items())}.
    **The other 25:** {cook_facts["rest_south_of_39"]} lie south of 39° N (by the mean position
    of their bridges), {cook_facts["rest_under_100_per_sqmi"]} have fewer than 100 people per
    square mile, and {len(cook_facts["rest_over_100k"])} have more than 100,000 people
    ({", ".join(cook_facts["rest_over_100k"])}).

    **Is Cook over-predicted beyond chance?** If each bridge turned poor independently with its
    predicted chance, the number that turn poor would follow a Poisson-binomial distribution:

    | Cook County | Bridges | Turned poor | Expected | Base rate | Mean predicted | P(as few) | P(as many) | FPR, Cook | FPR, the other 25 |
    |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
    {_tbl}

    "P(as few)" and "P(as many)" are the chances of seeing at most, and at least, as many as
    turned poor. Bridges in one county are not independent, so these understate the
    uncertainty. Cook is over-predicted on the test years: {_over["test"]}; on the validation
    years: {_over["validation"]}. Cook's false-alarm rate is about twice the other 25 counties'
    in both cycles.
    """),
        ]
    )
    return cook_calib, cook_facts


@app.cell
def _(FULL, INK, MUTED, QUARTILES, Q_COLOR, by_q, mo, np, overall, plt):
    PRECISIONS = (0.09, 0.11, 0.13)  # the figure's reference lines
    _fig, _ax = plt.subplots(figsize=FULL)
    _b = np.linspace(0.005, 0.10, 200)
    _tpr = overall["tpr"]
    for _ppv, _ls in zip(PRECISIONS, ("-", "--", ":"), strict=True):
        _ax.plot(100 * _b, 100 * _b / (1 - _b) * (1 - _ppv) / _ppv * _tpr, color=MUTED, ls=_ls,
                 label=f"precision {_ppv:.2f}")  # fmt: skip
    for _q, _mk in zip(QUARTILES, ("o", "s", "^", "D"), strict=True):
        _r = by_q[_q]
        _ax.scatter(100 * _r["base"], 100 * _r["fpr"], s=60, color=Q_COLOR[_q], marker=_mk,
                    edgecolor=INK, zorder=3, label=_q)  # fmt: skip
    _ax.set(xlabel="base rate: turned poor, %", ylabel="false alarms (FPR), %", ylim=(0, 60))
    _ax.legend(fontsize=9, frameon=False, ncol=2, title=f"catch rate {_tpr:.2f}",
               title_fontsize=9)  # fmt: skip
    _fig.tight_layout()
    # How far each quartile sits from the line of its nearest precision (drawn at the overall
    # catch rate, not the quartile's own), and how far p / (1 - p) bends from a straight line
    nearest_precision = {
        _q: min(PRECISIONS, key=lambda c: abs(c - by_q[_q]["ppv"])) for _q in QUARTILES
    }
    off_line = {
        _q: 100 * (by_q[_q]["fpr"] - by_q[_q]["base"] / (1 - by_q[_q]["base"])
                   * (1 - nearest_precision[_q]) / nearest_precision[_q] * _tpr)
        for _q in QUARTILES
    }  # fmt: skip
    bend_pct = 100 * (1 / (1 - _b[-1]) - 1)  # p/(1-p) against p, at the right end
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    ### Equal precision forces unequal errors

    False-alarm rate against base rate, with lines for three precisions at the overall catch
    rate ({_tpr:.3f}): the identity of §3, which holds exactly in every quartile (§4),
    $\\mathrm{{FPR}} = p/(1-p)\\cdot(1-\\mathrm{{PPV}})/\\mathrm{{PPV}}\\cdot\\mathrm{{TPR}}$. Nearest precision line: {nearest_precision}. Points off their line, in
    points of FPR: {", ".join(f"{_q} {v:+.1f}" for _q, v in off_line.items())} (each quartile has
    its own catch rate). The lines are $p/(1-p)$ times a constant, so they bend upward: at a 10%
    base rate they sit {bend_pct:.1f}% above a straight line through the origin.
    """),
        ]
    )
    return PRECISIONS, bend_pct, nearest_precision, off_line


@app.cell
def _(FULL, MUTED, PALETTE, QUARTILES, mo, np, plt, post):
    _fig, _ax = plt.subplots(figsize=FULL)
    _x = np.arange(len(QUARTILES))
    _one = [post["one"][_q]["tpr"] * 100 for _q in QUARTILES]
    _grp = [post["group"][_q]["tpr"] * 100 for _q in QUARTILES]
    _ax.bar(_x - 0.2, _one, 0.4, color=MUTED, label="one threshold, 1/21")
    _ax.bar(_x + 0.2, _grp, 0.4, color=PALETTE[2], hatch="//", label="a threshold per quartile")
    for _i in range(len(QUARTILES)):
        _ax.text(_x[_i] - 0.2, _one[_i] + 1, f"{_one[_i]:.1f}", ha="center", fontsize=9)
        _ax.text(_x[_i] + 0.2, _grp[_i] + 1, f"{_grp[_i]:.1f}", ha="center", fontsize=9)
    _ax.axhline(post["target"] * 100, color=PALETTE[1], ls="--", lw=1, label="target (validation)")
    _ax.set_xticks(_x, QUARTILES)
    _ax.set(ylabel="caught (TPR), %, test years", ylim=(0, 118))
    _ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=3)
    _fig.tight_layout()
    spread_one = max(_one) - min(_one)
    spread_group = max(_grp) - min(_grp)
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    ### Thresholds per quartile fix nothing here

    Chosen on the validation years to catch {100 * post["target"]:.1f}% in each quartile, the
    thresholds are {", ".join(f"{_q} {v:.4f}" for _q, v in post["thresholds"].items())}, next to
    1/21 = {1 / 21:.4f}. On the test years: {post["flags_group"]:,} flags instead of
    {post["flags_one"]:,}, {post["caught_group"]} caught instead of {post["caught_one"]}, an
    expected cost of {post["cost_group"]:.4f} against {post["cost_one"]:.4f} per bridge. The
    spread of catch rates goes from {spread_one:.1f} to {spread_group:.1f} points, and the
    mid-high quartile's rate does not move ({_one[2]:.1f}% and {_grp[2]:.1f}%): a gap that
    appears only on the test years cannot be set right by thresholds chosen before it.
    """),
        ]
    )
    return spread_group, spread_one


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: split it further

    The township bridges in high-vulnerability counties: how many turned poor on the test
    years? How wide is the interval of their catch rate? Guess before the next cell.

    Crossing two groupings (quartile by owner) can find a worst-served cell that neither
    grouping shows, and it shrinks every cell.
    """)
    return


@app.cell
def _(
    CDC_NOTE,
    QUARTILES,
    SIDE,
    boot_draws,
    by_q,
    cells,
    cells_zero_draws,
    mo,
    np,
    plt,
    zero_draw_effect,
):
    _owners = ["state", "county", "township", "city"]
    _M = np.full((len(QUARTILES), len(_owners)), np.nan)
    _N = np.zeros_like(_M)
    for _i, _q in enumerate(QUARTILES):
        for _j, _o in enumerate(_owners):
            _c = cells[f"{_q} | {_o}"]
            _M[_i, _j], _N[_i, _j] = 100 * _c["tpr"], _c["turned_poor"]
    _fig, _ax = plt.subplots(figsize=SIDE)
    _ax.imshow(_M, cmap="Blues", vmin=40, vmax=100, aspect="auto")
    for (_i, _j), _v in np.ndenumerate(_M):
        _ax.text(_j, _i, f"{_v:.1f}%\n{int(_N[_i, _j])} poor", ha="center", va="center",
                 fontsize=8, color="white" if _v > 80 else "#1A1A1A")  # fmt: skip
    _ax.set_xticks(range(len(_owners)), _owners)
    _ax.set_yticks(range(len(QUARTILES)), QUARTILES)
    _ax.set(xlabel="owner (NBI Item 22)", ylabel="SVI quartile")
    _fig.tight_layout()
    # The intervals of the small cells, and how far a small cell's resampled rate is off its
    # estimate on average (the bootstrap's estimate of bias) against its spread
    small = {_c: v for _c, v in cells.items() if v["turned_poor"] < 30}
    small_width = {_c: 100 * (v["ci"]["tpr"][1] - v["ci"]["tpr"][0]) for _c, v in small.items()}
    _smallest = min(cells, key=lambda c: cells[c]["turned_poor"])
    _d = boot_draws[_smallest]["tpr"]
    _d = _d[~np.isnan(_d)]  # draws with no bridge that turned poor have no catch rate
    smallest_bias = {
        "cell": _smallest,
        "turned_poor": cells[_smallest]["turned_poor"],
        "bias_points": 100 * (float(_d.mean()) - cells[_smallest]["tpr"]),
        "sd_points": 100 * float(_d.std()),
    }
    worst_cell = min(cells, key=lambda c: cells[c]["tpr"])
    assert cells[worst_cell]["tpr"] < min(r["tpr"] for r in by_q.values())
    # "mostly noisy, hardly biased"
    assert abs(smallest_bias["bias_points"]) < smallest_bias["sd_points"] / 4
    _rows = "\n".join(
        f"| {_c.replace('|', '&#124;')} | {v['turned_poor']} | {100 * v['tpr']:.1f}% | "
        f"{100 * v['ci']['tpr'][0]:.1f} to {100 * v['ci']['tpr'][1]:.1f}% | {small_width[_c]:.1f} |"
        for _c, v in small.items()
    )
    _zero_text = (
        "; ".join(
            f"{_c}: {v['draws']}, interval {100 * v['ci_tpr'][0]:.1f} to "
            f"{100 * v['ci_tpr'][1]:.1f}% without them, {100 * v['ci_tpr_as_zero'][0]:.1f} to "
            f"{100 * v['ci_tpr_as_zero'][1]:.1f}% if they counted as 0"
            for _c, v in zero_draw_effect.items()
        )
        or "none"
    )
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    ### Small cells, wide intervals

    Catch rate by quartile and owner, with how many bridges turned poor in each cell. *{CDC_NOTE}*

    The {len(small)} cells with fewer than 30 bridges that turned poor:

    | Cell | Turned poor | Caught | Interval (counties) | Width, points |
    |:--|--:|--:|--:|--:|
    {_rows}

    The worst-served cell is {worst_cell} ({100 * cells[worst_cell]["tpr"]:.1f}% of
    {cells[worst_cell]["turned_poor"]}), below every quartile. Draws with no bridge that turned
    poor have no catch rate, and the intervals leave them out:
    {sum(cells_zero_draws.values())} in all cells ({_zero_text}).
    **Bias or noise?** The smallest cell, {_smallest} ({smallest_bias["turned_poor"]} turned
    poor): its resampled catch rate is off its estimate by {smallest_bias["bias_points"]:+.1f}
    points on average, with a spread of {smallest_bias["sd_points"]:.1f} points. A rate from a
    handful of bridges is mostly noisy, hardly biased.
    """),
        ]
    )
    return small, small_width, smallest_bias, worst_cell


@app.cell
def _(
    COOK,
    PROXY_FOLDS,
    PROXY_REPEATS,
    PROXY_SETS,
    auc_rank,
    mo,
    np,
    proxy,
    proxy_scores,
    test,
    train,
    y_high,
):
    # Break the proxy test: counties the classifier never saw (the county folds computed with
    # the AUCs above, checked against the figure script), and Cook County left out of both years
    proxy_folds = {_name: proxy[_name]["auc_county_folds_median"] for _name in PROXY_SETS}
    proxy_signal_kept = {
        _name: (proxy_folds[_name] - 0.5) / (proxy[_name]["auc"] - 0.5) for _name in PROXY_SETS
    }
    _keep_tr, _keep_te = train["fips"] != COOK, (test["fips"] != COOK).to_numpy()
    proxy_no_cook = {
        _name: auc_rank(y_high[_keep_te], proxy_scores(_cols, train[_keep_tr], test[_keep_te]))
        for _name, _cols in PROXY_SETS.items()
    }
    # Why not one county at a time, pooled: a classifier with no inputs scores each county by
    # the training share of top-quartile bridges without it, which falls exactly when the held-out
    # county is in the top quartile
    _y_tr = (train["quartile"] == "high").to_numpy()
    _no_inputs = np.empty(len(test))
    for _f in sorted(test["fips"].unique()):
        _no_inputs[(test["fips"] == _f).to_numpy()] = _y_tr[(train["fips"] != _f).to_numpy()].mean()
    proxy_loco_no_inputs = auc_rank(y_high, _no_inputs)
    assert proxy_loco_no_inputs < 0.05  # every county ranked backwards by construction
    # "about half of the signal above chance carries to counties never seen"; "much to Cook"
    assert all(0.5 < proxy_folds[_n] < proxy[_n]["auc"] for _n in PROXY_SETS)
    assert all(0.35 < _v < 0.65 for _v in proxy_signal_kept.values()), proxy_signal_kept
    assert proxy_no_cook["no_location"] < proxy["no_location"]["auc"] - 0.03
    _range = {_n: proxy[_n]["auc_county_folds_range"] for _n in PROXY_SETS}
    mo.md(f"""
    ### Break the proxy test: a county the classifier never saw

    The AUC above scores the 2020 bridges of the same 102 counties the classifier was trained on
    (in 2014). Score counties it never saw instead: {PROXY_FOLDS} folds of counties, stratified
    by quartile, each scored by a classifier fitted on the others; the median over
    {PROXY_REPEATS} shuffles of the folds, with the range:

    | Inputs | Same counties | Counties held out (range) | Cook left out of both years |
    |:--|--:|--:|--:|
    | all 16 | {proxy["all"]["auc"]:.3f} | {proxy_folds["all"]:.3f} ({_range["all"][0]:.3f} to {_range["all"][1]:.3f}) | {proxy_no_cook["all"]:.3f} |
    | without location | {proxy["no_location"]["auc"]:.3f} | {proxy_folds["no_location"]:.3f} ({_range["no_location"][0]:.3f} to {_range["no_location"][1]:.3f}) | {proxy_no_cook["no_location"]:.3f} |

    Holding out one county at a time and pooling the scores would mislead: a classifier with no
    inputs at all, scored that way, gets AUC {proxy_loco_no_inputs:.3f}, because the training
    share of top-quartile bridges drops exactly when a top-quartile county is held out. Within a
    fold, the same classifier gets 0.5.

    For the bridge model, which is used on the same counties every cycle, the first column is the
    one that matters: its inputs carry the group. On counties it never saw, about half of the
    signal above chance remains ({100 * proxy_signal_kept["all"]:.0f}% with all 16 inputs,
    {100 * proxy_signal_kept["no_location"]:.0f}% without location): part of the signal is
    these counties themselves, much of it Cook, and part carries to a new county.
    """)
    return proxy_folds, proxy_loco_no_inputs, proxy_no_cook, proxy_signal_kept


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Where to intervene

    | Stage | Change | On these bridges |
    |:--|:--|:--|
    | Data (pre-processing) | collect more, reweight, better labels | inspect where ratings are thin |
    | Training (in-processing) | a fairness penalty or constraint; adversarial learning | catch rates as a constraint |
    | Decision (post-processing) | thresholds per group | tried in §5: nothing to fix |
    | Always | evaluate each group, with intervals | every cycle, published |

    Adversarial learning (Zhang et al., 2018) trains a representation from which a second
    network cannot recover the group. The last row is not optional: without disaggregated
    evaluation none of the first three can be checked.
    """)
    return


@app.cell
def _(HORIZON, START, by_q, cook_calib, mo, overall):
    _lo, _hi = min(r["tpr"] for r in by_q.values()), max(r["tpr"] for r in by_q.values())
    _c, _v = cook_calib["test"], cook_calib["validation"]
    mo.md(f"""
    ### A model card for Lecture 6's model

    Mitchell et al. (2019): a short, standard report that travels with a model.

    | Field | Entry |
    |:--|:--|
    | Use | flag non-poor Illinois bridges for preventive work, {HORIZON} years ahead |
    | Not for | moving budgets between counties; ranking communities |
    | Data | NBI Illinois; trained {START["train"]} → {START["train"] + HORIZON}, validated {START["val"]} → {START["val"] + HORIZON}, tested {START["test"]} → {START["test"] + HORIZON} |
    | Overall | catches {100 * overall["tpr"]:.1f}% of {overall["turned_poor"]}; false alarms {100 * overall["fpr"]:.1f}% |
    | By SVI quartile | caught {100 * _lo:.1f}% to {100 * _hi:.1f}%; county intervals overlap; equal last cycle |
    | Caveats | labels are ratings; Cook County's false alarms are about twice the rest of its quartile's; Cook over-predicted on the test years ({_c["mean_p_pct"]:.1f}% predicted, {_c["base_pct"]:.1f}% turned poor), under-predicted the cycle before ({_v["mean_p_pct"]:.1f}% and {_v["base_pct"]:.1f}%) |
    | Source of the index | CDC/ATSDR SVI 2022 (use does not imply endorsement by CDC, ATSDR, HHS, or the U.S. Government; available at no charge from ATSDR) |
    """)
    return


@app.cell
def _(FIGURES, mo, re, test):
    # Which earlier lectures read the NBI, and which used Qwen: scanned from the figure scripts
    def _lectures_before(pattern: str, before: int = 27) -> list[int]:
        """Lectures before ``before`` whose figure script matches ``pattern`` (any case)."""
        out = []
        for _f in sorted(FIGURES.glob("cee_l[0-9][0-9]_*.py")):
            _n = int(_f.name[5:7])
            if _n < before and re.search(pattern, _f.read_text(encoding="utf-8"), flags=re.I):
                out.append(_n)
        return out

    nbi_lectures = _lectures_before(r"load_bridges|\bnbi\b")
    qwen_lectures = _lectures_before(r"qwen")
    # Item 91, the inspection interval in months, of the 2020 bridges: share by interval
    interval_24 = {
        float(k): float(v)
        for k, v in test["inspection_interval"].value_counts(normalize=True).items()
    }
    assert abs(sum(interval_24.values()) - 1) < 1e-12 and {24.0, 48.0} <= set(interval_24)
    shortest_m = float(test["length_m"].min())
    mo.md(f"""
    ### A datasheet for the inventory

    Gebru et al. (2021): who made the data, why, what is in it, how it was collected and
    labelled, and its known limits.

    | Question | National Bridge Inventory, Illinois |
    |:--|:--|
    | Who made it, why | FHWA and the states: bridge safety oversight |
    | What is in it | public highway bridges longer than 20 ft (6.1 m); the shortest here is {shortest_m:.1f} m |
    | How collected | inspections: every 2 years for {100 * interval_24[24]:.1f}% of the 2020 bridges here, every 4 years for {100 * interval_24[48]:.1f}% (Item 91) |
    | Who labels | state and local inspectors' condition ratings (Items 58–60) |
    | Known limits | ratings are judgements; culverts left out here; records of bridges that left the inventory are dropped |

    Every lecture whose figure script reads the NBI ({", ".join(map(str, nbi_lectures))}, and
    today) inherits these answers.
    """)
    return interval_24, nbi_lectures, qwen_lectures, shortest_m


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: what should IDOT require?

    Before the flag list guides preventive work, the model must show …

    - **A.** equal flag rates in every SVI quartile.
    - **B.** equal catch rates, within their intervals, every cycle.
    - **C.** calibration in every quartile and in Cook County.
    - **D.** good overall accuracy; groups are not its job.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(cook_calib, gap_ci, mo):
    _c, _v = cook_calib["test"], cook_calib["validation"]
    mo.accordion(
        {
            "Discussion": mo.md(
                f"""
                A (demographic parity) would cut flags in the high quartile, where a flag is
                help. B fits a decision whose harm is a miss; on the test years the largest gap
                (low-mid minus mid-high) has the county interval {100 * gap_ci[0]:.1f} to
                {100 * gap_ci[1]:.1f} points, and last cycle the quartiles were equal, so check
                it every cycle. C would expose Cook's over-prediction on the test years
                ({_c["mean_p_pct"]:.1f}% predicted, {_c["base_pct"]:.1f}% turned poor), which
                spends money but misses no bridge; the cycle before, Cook was under-predicted
                ({_v["mean_p_pct"]:.1f}% and {_v["base_pct"]:.1f}%), so calibration too needs
                checking every cycle. D is what produced today's questions. B, with C reported,
                is defensible; A misreads what a flag means here.
                """
            )
        }
    )
    return


@app.cell
def _(mo, qwen_lectures):
    mo.md(f"""
    ### Unbiased, fair, and just

    | Goal | Asks | Here |
    |:--|:--|:--|
    | Unbiased | is the estimate right on average? | is p calibrated, overall and by group? |
    | Fair | are groups treated alike? | equal catch rates |
    | Just | does each get what it needs? | more inspection for the worse off? |

    The three pull apart: a just program might inspect vulnerable counties more, which is unequal
    treatment by design. Equal outcomes or equal opportunity is a choice society makes attribute
    by attribute; whoever builds the model should state which one the program wants.

    ### Where it lives today

    | Practice | What it is |
    |:--|:--|
    | Model cards | on Hugging Face's model pages (Qwen, Lectures {" and ".join(map(str, qwen_lectures))}) |
    | Datasheets | dataset cards; documentation in public data portals |
    | Group testing | NIST's face recognition tests by demographic group (Grother et al., 2019, NISTIR 8280) |
    | Risk frameworks | NIST AI RMF 1.0 (2023): measure and manage harmful bias |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    ### Exit question

    A city says its new pavement-repair model is "fair: it is equally accurate in every ward".
    What would you ask?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Some answers": mo.md(
                r"""
                Any two of: accurate by which measure, and is the harm here a miss or a false
                alarm? With what interval, resampling what? The same in last year's data? How
                many repairs were needed in each ward (base rates)? Are wards the right groups,
                or do they hide neighbourhoods? What do the labels measure: complaints,
                inspections, or condition?
                """
            )
        }
    )
    return


@app.cell
def _(
    DECK,
    EDGES,
    HORIZON,
    INPUTS,
    PRECISIONS,
    PROXY_FOLDS,
    PROXY_REPEATS,
    QUARTILES,
    R,
    START,
    attempt,
    by_q,
    cells,
    high_split,
    identity,
    interval_24,
    mo,
    nbi_lectures,
    numbers,
    overall,
    owner_caught,
    post,
    proxy,
    qwen_lectures,
    svi,
    svi_measures,
    test,
):
    # Every number the deck states, rendered from this notebook's values (`numbers` is the
    # notebook's own rebuild of numbers.json, asserted equal to the slides') and found word for
    # word in the deck
    _n = numbers
    _bq, _hs, _pp, _cl = _n["by_quartile"], _n["high_split"], _n["post"], _n["cells"]
    _lo, _hi = attempt["low"], attempt["high"]
    _others = [v for k, v in owner_caught.items() if k != "township"]
    _tq = [_bq[_q]["tpr_pct"] for _q in _bq]
    _fl = [_bq[_q]["flagged_pct"] for _q in _bq]
    _fp = [_bq[_q]["fpr_pct"] for _q in _bq]
    _lowest = min(_bq, key=lambda q: _bq[q]["tpr_pct"])
    _highest = max(_bq, key=lambda q: _bq[q]["tpr_pct"])
    _vq = [_n["by_quartile_val"][_q]["tpr_pct"] for _q in _bq]
    _poor = {c: v["turned_poor"] for c, v in _cl.items()}
    _ht, _mc = _cl["high | township"], _cl["mid-high | city"]
    _th = list(_pp["thresholds"].values())
    _flag_gap = 100 * (max(r["flagged"] for r in by_q.values())
                       - min(r["flagged"] for r in by_q.values()))  # fmt: skip
    _label = {_q: _q.capitalize() for _q in QUARTILES}  # "Low-mid", as the deck's tables
    _post_tpr = [
        f"{_q} {100 * post['one'][_q]['tpr']:.0f} and {100 * post['group'][_q]['tpr']:.0f}"
        for _q in QUARTILES
    ]
    _post_tpr[0] = _post_tpr[0].capitalize()
    _prec = [f"{c:.2f}" for c in PRECISIONS]
    assert attempt["low"]["counties"] == attempt["high"]["counties"]
    assert identity["high"]["fpr"] == by_q["high"]["fpr"]
    assert len(high_split) == 2 and _hs["the other 25"]["counties"] == 25
    _hv = _n["high_split_val"]
    _px = _n["proxies"]
    _ten = {10: "ten"}
    # "about half of the signal above chance carries to a county the model never saw"
    assert all(40 <= _px[_k]["signal_kept_pct"] <= 60 for _k in ("all", "no_location"))
    # The deck: most over-predicted in the low-mid and high quartiles, by the same 1.6 points,
    # and about equal in the other two
    _gap = {_q: _bq[_q]["calib_gap_pct"] for _q in _bq}
    assert set(sorted(_gap, key=_gap.get)[2:]) == {"low-mid", "high"}
    assert _gap["low-mid"] == _gap["high"] and max(abs(_gap["low"]), abs(_gap["mid-high"])) < 0.5
    # Cook: under-predicted the cycle before, over-predicted on the test years
    assert _hv["Cook County"]["expected"] < _hv["Cook County"]["turned_poor"]
    assert _n["cook_expected_test"] > _hs["Cook County"]["turned_poor"]
    _cycle = sorted(interval_24, key=interval_24.get, reverse=True)[:2]  # the two commonest
    claims = [
        f"a miss costs {R} false alarms, so flag when p > 1/{1 + R}.",
        f"Townships': {owner_caught['township']:.0f}% of their bridges that turned poor were "
        f"caught, against {min(_others):.0f}% to {max(_others):.0f}% for the state, counties, and "
        "cities.",
        f"Illinois's {svi['fips'].nunique()} counties, each drawn at the mean position of its "
        "bridges",
        f"({len(test):,} bridges, {_n['overall']['turned_poor']} turned poor)",
        f"ranks each county on {len(svi_measures)} census factors",
        f"the top quartile (ranks above {EDGES[3]:.2f}) is the most vulnerable",
        f"| SVI quartile, {_lo['counties']} counties each | Turned poor | Caught | Stayed healthy "
        "| Healthy, flagged |",
        f"| Low | {_lo['poor']} | {_lo['caught']} | {_lo['healthy']:,} | {_lo['false_alarms']:,} |",
        f"| High | {_hi['poor']} | {_hi['caught']} | {_hi['healthy']:,} | "
        f"{_hi['false_alarms']:,} |",
        f"{_lo['caught']} / {_lo['poor']} = {100 * _lo['caught'] / _lo['poor']:.0f}% low, "
        f"{_hi['caught']} / {_hi['poor']} = {100 * _hi['caught'] / _hi['poor']:.0f}% high.",
        f"False alarms: {_lo['false_alarms']:,} / {_lo['healthy']:,} = "
        f"{100 * _lo['false_alarms'] / _lo['healthy']:.0f}% low, {_hi['false_alarms']:,} / "
        f"{_hi['healthy']:,} = {100 * _hi['false_alarms'] / _hi['healthy']:.0f}% high.",
        f"| Data | training unlike use | {START['train']}-{START['train'] + HORIZON} trains, "
        f"{START['test']}-{START['test'] + HORIZON} tests |",
        f"the flag (p above 1/{1 + R})",
        f"from Lecture 6's {len(INPUTS)} inputs alone: AUC {proxy['all']['auc']:.2f} with all "
        f"of them, {proxy['no_location']['auc']:.2f} without latitude and longitude.",
        f"with AUC {_n['proxies']['all']['auc']:.3f} on the 2020 bridges, and still "
        f"{_n['proxies']['no_location']['auc']:.3f} without location",
        f"(traffic and material alone reach {_px['traffic_material_alone']:.3f}). On counties "
        f"held out of training ({_ten[PROXY_FOLDS]} folds of counties, {_ten[PROXY_REPEATS]} "
        f"shuffles), the AUC is {_px['all']['auc_county_folds_median']:.3f}, and "
        f"{_px['no_location']['auc_county_folds_median']:.3f} without location: about half of "
        "the signal above chance carries to a county the model never saw",
        f"the model catches {_bq['low-mid']['tpr_pct']:.1f}% of the bridges that turned poor in "
        f"the low-mid quartile and {_bq['mid-high']['tpr_pct']:.1f}% in the mid-high quartile.",
        f"flagged {min(_fl):.0f}% to {max(_fl):.0f}%; caught {min(_tq):.1f}% (interval "
        f"{_bq[_lowest]['ci_pct']['tpr'][0]:.1f} to {_bq[_lowest]['ci_pct']['tpr'][1]:.1f}) to "
        f"{max(_tq):.1f}% ({_bq[_highest]['ci_pct']['tpr'][0]:.1f} to "
        f"{_bq[_highest]['ci_pct']['tpr'][1]:.1f}); false alarms {min(_fp):.1f}% to "
        f"{max(_fp):.1f}%; and calibration: mean predicted chance against the share that "
        f"turned poor, {_bq['low-mid']['mean_p_pct']:.1f}% against "
        f"{_bq['low-mid']['base_pct']:.1f}% in the low-mid quartile and "
        f"{_bq['high']['mean_p_pct']:.1f}% against {_bq['high']['base_pct']:.1f}% in the high "
        "quartile, about equal in the other two.",
        f"{_bq['mid-high']['turned_poor']} bridges turned poor in the mid-high quartile",
        f"Flag rates (demographic parity) differ by up to {_flag_gap:.0f} points",
        f"The model over-predicts this cycle as a whole ({_n['overall']['mean_p_pct']:.1f}% "
        f"predicted, {_n['overall']['base_pct']:.1f}% turned poor; the cycle before, "
        f"{_n['overall_val']['mean_p_pct']:.1f}% and {_n['overall_val']['base_pct']:.1f}%), "
        f"most in the low-mid and high quartiles: {_gap['high']:.1f} points each (intervals "
        f"{_bq['low-mid']['ci_pct']['calib_gap'][0]:.1f} to "
        f"{_bq['low-mid']['ci_pct']['calib_gap'][1]:.1f} and "
        f"{_bq['high']['ci_pct']['calib_gap'][0]:.1f} to "
        f"{_bq['high']['ci_pct']['calib_gap'][1]:.1f}).",
        *[
            f"| {_label[_q]} | {_n['by_quartile_val'][_q]['tpr_pct']:.1f}% | "
            f"{_bq[_q]['tpr_pct']:.1f}% |"
            for _q in _bq
        ],
        f"every quartile between {min(_vq):.1f}% and {max(_vq):.1f}%, around the "
        f"{_n['overall_val']['tpr_pct']:.1f}% overall.",
        f"False alarms: Cook County {_hs['Cook County']['fpr_pct']:.1f}%, the other "
        f"{_hs['the other 25']['counties']} counties {_hs['the other 25']['fpr_pct']:.1f}%. "
        f"Calibration: in Cook, {_hs['Cook County']['mean_p_pct']:.1f}% predicted against "
        f"{_hs['Cook County']['base_pct']:.1f}% that turned poor; in the other "
        f"{_hs['the other 25']['counties']}, {_hs['the other 25']['mean_p_pct']:.1f}% against "
        f"{_hs['the other 25']['base_pct']:.1f}%.",
        f"Cook County holds {_n['cook']['bridges']:,} of the top quartile's bridges "
        f"({_n['cook']['share_of_high_pct']:.0f}%), and the model flags "
        f"{_hs['Cook County']['fpr_pct']:.1f}% of its healthy ones; without Cook, the quartile "
        f"looks like the rest ({_hs['the other 25']['fpr_pct']:.1f}%).",
        f"sits in it beside {_hs['the other 25']['counties']} other counties, from Peoria and "
        "Winnebago (Rockford) to rural southern ones.",
        f"the cycle before, Cook was under-predicted ({_hv['Cook County']['expected']:.1f} "
        f"expected, {_hv['Cook County']['turned_poor']} turned poor; this cycle "
        f"{_n['cook_expected_test']:.1f} expected, {_hs['Cook County']['turned_poor']}), while "
        f"its false alarms were again about twice the rest's "
        f"({_hv['Cook County']['fpr_pct']:.1f}% against {_hv['the other 25']['fpr_pct']:.1f}%).",
        f"for a catch rate of {overall['tpr']:.2f} and precision {_prec[0]}, {_prec[1]}, and "
        f"{_prec[2]}",
        f"(high: {_n['identity']['high']['fpr']:.2f}% computed, "
        f"{_n['identity']['high']['implied']:.2f}% from the identity)",
        f"to catch {_pp['target_pct']:.1f}% in each. " + "; ".join(_post_tpr) + ".",
        f"the four thresholds land between {min(_th):.4f} and {max(_th):.4f}, next to 1/{1 + R}",
        f"{_pp['flags_group']:,} flags instead of {_pp['flags_one']:,}, {_pp['caught_group']} "
        f"caught instead of {_pp['caught_one']}, an expected cost of {_pp['cost_group']:.3f} "
        f"against {_pp['cost_one']:.3f} per bridge.",
        f"from {min(_poor.values())} to {max(_poor.values())}; township bridges in "
        f"high-vulnerability counties, {100 * cells['high | township']['tpr']:.0f}% of "
        f"{_ht['turned_poor']}; city "
        f"bridges in mid-high counties, {_mc['tp']} of {_mc['turned_poor']}.",
        f"Township bridges in the top quartile: {_ht['turned_poor']} turned poor, "
        f"{_ht['tpr_pct']:.1f}% caught, interval {_ht['ci_pct']['tpr'][0]:.1f}% to "
        f"{_ht['ci_pct']['tpr'][1]:.1f}%. {_n['cells_under_30_poor']} of the {len(_cl)} cells "
        "hold fewer than 30 bridges that turned poor.",
        f"| Use | flag non-poor Illinois bridges for preventive work, {HORIZON} years ahead |",
        f"| Data | NBI Illinois, trained {START['train']}-{START['train'] + HORIZON}, tested "
        f"{START['test']}-{START['test'] + HORIZON} |",
        f"| Overall | catches {_n['overall']['tpr_pct']:.1f}% of {_n['overall']['turned_poor']}; "
        f"false alarms {_n['overall']['fpr_pct']:.1f}% |",
        f"| By quartile | caught {min(_tq):.1f}% to {max(_tq):.1f}%; intervals overlap |",
        f"| How collected | inspections every {_cycle[0] / 12:.0f} or {_cycle[1] / 12:.0f} years "
        "(Item 91) |",
        f"Every lecture that used the NBI ({', '.join(map(str, nbi_lectures))}, today)",
        f"(Qwen, Lectures {' and '.join(map(str, qwen_lectures))})",
    ]  # fmt: skip
    _missing = [c for c in claims if c not in DECK]
    assert not _missing, _missing
    mo.md(
        f"""
    ### The deck's claims

    **Every number on the slides:** {len(claims)} statements, each rendered from this notebook's
    values (its own rebuild of numbers.json, the attempt's counts, the owners' catch rates, the
    SVI file, and scans of the figure scripts) and found word for word in the deck's text.
    """
    )
    return (claims,)


@app.cell
def _(
    DECK,
    bend_pct,
    by_q,
    by_q_val,
    cells,
    cook_calib,
    cook_facts,
    gap_ci,
    gap_share_above_zero,
    high_split,
    interval_24,
    mo,
    nearest_precision,
    off_line,
    overlap,
    post,
    proxy,
    proxy_drop,
    proxy_folds,
    proxy_loco_no_inputs,
    proxy_no_cook,
    proxy_signal_kept,
    proxy_traffic_material,
    same_counties,
    shortest_m,
    small_width,
    spread_group,
    spread_one,
    target,
    train_bias,
    val_inside,
    width_ratio,
    worst_cell,
):
    # Statements in the deck that numbers.json does not carry, each measured here
    findings = []

    def finding(statement: str, measured: str, verdict: str) -> None:
        assert statement in DECK, statement
        findings.append((statement, measured, verdict))

    _city_state = sum(cook_facts["owner_share_cook"].get(k, 0) for k in ("city", "state"))
    _tb = train_bias
    _optimistic = _tb["training"]["log_loss"] < min(_tb["validation"]["log_loss"],
                                                    _tb["test"]["log_loss"])  # fmt: skip
    finding("| Statistics | an estimate off on average | a loss scored on the training years |",
            f"catch rate {_tb['training']['tpr_pct']:.1f}% on the training years against "
            f"{_tb['validation']['tpr_pct']:.1f}% (validation) and {_tb['test']['tpr_pct']:.1f}% "
            f"(test); log loss {_tb['training']['log_loss']:.4f} against "
            f"{_tb['validation']['log_loss']:.4f} and {_tb['test']['log_loss']:.4f}",
            "supported: the training-years loss is the lowest of the three" if _optimistic
            else "not supported: the training-years loss is not the lowest")  # fmt: skip
    _noloc = proxy["no_location"]["auc"]
    finding("(traffic and material alone reach",
            f"traffic and material alone: AUC {proxy_traffic_material:.3f}, against {_noloc:.3f} "
            "with all 14 inputs without location; dropping one group at a time: "
            + "; ".join(f"{k} {v:.3f}" for k, v in proxy_drop.items()),
            "supported: the six inputs carry nearly all of it"
            if proxy_traffic_material > _noloc - 0.01 else "not supported")  # fmt: skip
    finding("the rest is recognizing these counties",
            "share of the AUC above 0.5 kept on counties held out (median of the shuffles): "
            + ", ".join(f"{k} {100 * v:.0f}%" for k, v in proxy_signal_kept.items())
            + f"; Cook left out of both years: {proxy_no_cook['all']:.3f} and "
            f"{proxy_no_cook['no_location']:.3f}; one county at a time, pooled, a classifier with "
            f"no inputs would score {proxy_loco_no_inputs:.3f}",
            "supported: about half of the signal does not carry to a new county"
            if all(0.35 < v < 0.65 for v in proxy_signal_kept.values())
            and all(v < proxy[k]["auc"] for k, v in proxy_folds.items())
            else "not supported")  # fmt: skip
    _same = len(same_counties) == 1
    finding("For a model reused on the same counties every cycle, the first number is the one "
            "that matters.",
            f"the training, validation, and test bridges come from the same "
            f"{len(next(iter(same_counties)))} counties: {_same}",
            "a judgement; its premise holds" if _same else "the premise fails")  # fmt: skip
    finding("The catch rates' intervals overlap widely",
            f"all four share {100 * overlap[0]:.1f}% to {100 * overlap[1]:.1f}%; the gap low-mid "
            f"minus mid-high has interval {100 * gap_ci[0]:.1f} to {100 * gap_ci[1]:.1f} points "
            f"({100 * gap_share_above_zero:.1f}% of paired draws above zero)",
            "supported" if gap_ci[0] < 0 else
            "overlap supported, but the gap's own interval excludes zero")  # fmt: skip
    _fpr_gap = 100 * (max(r["fpr"] for r in by_q.values()) - min(r["fpr"] for r in by_q.values()))
    _flag_gap = 100 * (max(r["flagged"] for r in by_q.values())
                       - min(r["flagged"] for r in by_q.values()))  # fmt: skip
    finding("false alarms by about as much",
            f"false alarms differ by up to {_fpr_gap:.1f} points, flag rates by {_flag_gap:.1f}",
            "supported" if abs(_fpr_gap - _flag_gap) < 1 else "check")  # fmt: skip
    finding("The test year's spread is consistent with a noisy year: each quartile's last-cycle "
            "rate lies inside its test interval.",
            f"validation catch rate inside the test-year county interval: {val_inside}",
            "supported" if all(val_inside.values()) else "not supported")  # fmt: skip
    _rest = [100 * r["fpr"] for q, r in by_q.items() if q != "high"]
    _o25 = 100 * high_split["the other 25"]["fpr"]
    finding("without Cook, the quartile looks like the rest",
            f"the other 25 counties' FPR {_o25:.1f}% against the other quartiles' "
            f"{min(_rest):.1f}% to {max(_rest):.1f}%",
            "supported" if min(_rest) <= _o25 <= max(_rest) else "not supported")  # fmt: skip
    finding("with heavy traffic and city and state bridges",
            f"median ADT {cook_facts['adt_median_cook']:,.0f} in Cook against "
            f"{cook_facts['adt_median_rest']:,.0f} in the other 25; Cook's owners: "
            + ", ".join(f"{k} {100 * v:.0f}%" for k, v in cook_facts["owner_share_cook"].items()),
            "supported" if cook_facts["adt_median_cook"] > 10 * cook_facts["adt_median_rest"]
            and _city_state > 0.5 else "not supported")  # fmt: skip
    _named = all(n in cook_facts["rest_names"] for n in ("Peoria County", "Winnebago County"))
    finding("from Peoria and Winnebago (Rockford) to rural southern ones",
            f"of the other 25: Peoria and Winnebago among them: {_named}; "
            f"{cook_facts['rest_south_of_39']} south of 39° N; "
            f"{cook_facts['rest_under_100_per_sqmi']} under 100 people per square mile; "
            f"{len(cook_facts['rest_over_100k'])} over 100,000 people "
            f"({', '.join(cook_facts['rest_over_100k'])})",
            "supported" if _named and cook_facts["rest_south_of_39"] > 0 else "not supported")  # fmt: skip
    _ct, _cv = cook_calib["test"], cook_calib["validation"]
    _swing = (_ct["expected"] > _ct["turned_poor"]) != (_cv["expected"] > _cv["turned_poor"])
    finding("Calibration is not a fixed trait either",
            f"test: {_ct['expected']:.1f} expected to turn poor, {_ct['turned_poor']} did (P of so "
            f"few, if bridges were independent: {_ct['p_at_most_observed']:.1e}); validation: "
            f"{_cv['expected']:.1f} expected, {_cv['turned_poor']} did (P of so many: "
            f"{_cv['p_at_least_observed']:.2f})",
            "supported" if _swing else "not supported")  # fmt: skip
    finding("Cook's calibration swings by cycle",
            "as above: over-predicted on the test years, under-predicted the cycle before",
            "supported" if _swing else "not supported")  # fmt: skip
    finding("C would expose Cook's calibration, which swings between cycles",
            "over-predicted on the test years, under-predicted the cycle before",
            "supported" if _swing else "not supported")  # fmt: skip
    _drives = all(c["fpr_pct"] > 1.5 * c["fpr_others_pct"] for c in cook_calib.values())
    finding("One county drives the false alarms",
            f"Cook's FPR {_ct['fpr_pct']:.1f}% against {_ct['fpr_others_pct']:.1f}% for the other "
            f"25 (test); {_cv['fpr_pct']:.1f}% against {_cv['fpr_others_pct']:.1f}% (validation)",
            "supported in both cycles" if _drives else "on the test years only")  # fmt: skip
    finding("curves rising from the origin",
            f"FPR is p/(1 - p) times a constant: zero at p = 0, {bend_pct:.1f}% above a straight "
            "line at a 10% base rate",
            "supported")  # fmt: skip
    _off = max(abs(v) for v in off_line.values())
    finding("The four quartiles sit near them",
            f"nearest lines {nearest_precision}; off by "
            + ", ".join(f"{q} {v:+.1f}" for q, v in off_line.items()) + " points of FPR",
            "supported" if _off < 3 else "not supported")  # fmt: skip
    _val_tpr = [100 * r["tpr"] for r in by_q_val.values()]
    _within = max(abs(v - 100 * target) for v in _val_tpr)
    finding("because the quartiles were already equal there",
            f"with one threshold the validation catch rates were {min(_val_tpr):.1f}% to "
            f"{max(_val_tpr):.1f}%, all within {_within:.1f} points of the target "
            f"({100 * target:.1f}%)",
            "supported" if _within < 1 else "not supported")  # fmt: skip
    _still = post["one"]["mid-high"]["tpr"] == post["group"]["mid-high"]["tpr"]
    finding("Thresholds per quartile fix nothing here",
            f"the spread of test catch rates goes from {spread_one:.1f} to {spread_group:.1f} "
            f"points, by lowering two quartiles; mid-high's moves: {not _still}",
            "supported for the worst-served quartile" if _still else "check")  # fmt: skip
    _wide = sum(w > 20 for w in small_width.values())
    finding("Most cells this small have intervals more than 20 points wide",
            "interval widths of the cells under 30: "
            + ", ".join(f"{c} {w:.1f}" for c, w in small_width.items()),
            f"supported: {_wide} of {len(small_width)}" if _wide > len(small_width) / 2
            else "not supported")  # fmt: skip
    finding("crossing groups finds the worst-served cell",
            f"{worst_cell}: {100 * cells[worst_cell]['tpr']:.1f}%, below every quartile "
            f"({100 * min(r['tpr'] for r in by_q.values()):.1f}% at the lowest)",
            f"supported (a cell of {cells[worst_cell]['turned_poor']} bridges that turned poor, "
            f"interval {100 * cells[worst_cell]['ci']['tpr'][0]:.1f} to "
            f"{100 * cells[worst_cell]['ci']['tpr'][1]:.1f}%)")  # fmt: skip
    finding("Every interval resamples counties, not bridges",
            "county intervals of the catch rate against bridge intervals: "
            + ", ".join(f"{q} {v:.2f}x as wide" for q, v in width_ratio.items()),
            "supported" if min(width_ratio.values()) > 1 else "check")  # fmt: skip
    finding("public highway bridges over 20 ft",
            f"the shortest 2020 bridge here is {shortest_m:.1f} m (20 ft = 6.1 m)",
            "supported" if shortest_m > 20 * 0.3048 else "not supported")  # fmt: skip
    _two_four = interval_24.get(24.0, 0) + interval_24.get(48.0, 0)
    finding("inspections every 2 or 4 years (Item 91)",
            "inspection interval (Item 91) of the 2020 bridges: "
            + ", ".join(f"{k:.0f} months {100 * v:.1f}%" for k, v in interval_24.items()),
            f"supported ({100 * _two_four:.1f}% on one of the two)" if _two_four > 0.99
            else "partly")  # fmt: skip
    finding("it is met today within noise",
            f"the largest gap's county interval: {100 * gap_ci[0]:.1f} to {100 * gap_ci[1]:.1f} "
            "points",
            "supported" if gap_ci[0] < 0 < gap_ci[1] else "not supported")  # fmt: skip

    def _cell(text: str) -> str:
        """Text for a Markdown table cell: a pipe would end the cell, so write its entity."""
        return text.replace("|", "&#124;")

    _rows = "\n".join(f"| {_cell(s)} | {_cell(m)} | {_cell(v)} |" for s, m, v in findings)
    for _f in findings:
        print("finding:", " | ".join(_f))
    mo.md(f"""
    **Statements the slides make without a numbers.json key of their own,** each measured here:

    | The deck says | Measured | Verdict |
    |:--------------|:---------|:--------|
    {_rows}
    """)
    return (findings,)


@app.cell
def _(
    BOOT,
    T_START,
    boot_ci,
    boot_seconds,
    cells_zero_draws,
    ci_check_seconds,
    claims,
    findings,
    gap_ci,
    mo,
    newton_steps,
    numbers_checked,
    p_gap,
    proxy_lbfgs_gap,
    proxy_loco_no_inputs,
    q_order,
    time,
    w_gap,
    width_ratio,
    zero_draw_effect,
):
    _minutes = (time.perf_counter() - T_START) / 60
    _moved = sum(v["ci_tpr"] != v["ci_tpr_as_zero"] for v in zero_draw_effect.values())
    _fit = (
        "exactly" if w_gap == 0 and p_gap == 0 else f"to {w_gap:.0e} (predictions to {p_gap:.0e})"
    )
    print(f"done in {_minutes:.2f} min (bootstrap {boot_seconds:.2f} s, the script's bootstrap "
          f"check {ci_check_seconds:.0f} s); {numbers_checked} numbers.json values equal; "
          f"{len(claims)} claims found; {len(findings)} statements measured")  # fmt: skip
    mo.md(f"""
    ## Gotchas

    - **The CSV starts with a byte-order mark.** Read it with `encoding="utf-8-sig"`, or the
      first column is named `"\\ufeffST"` and a lookup by name fails.
    - **FIPS codes are text.** Join on the five-character string `"17" + county.zfill(3)`, read
      with `dtype=str`; validate the join (`many_to_one`) and assert that every bridge found a
      county.
    - **Where an edge falls.** The quartiles here are closed on the right (`pd.cut`'s default; the
      top quartile holds the ranks above 0.75), so a county at exactly 0.75 would land in
      mid-high, and bins closed on the left would put it in high. No Illinois county sits on an
      edge (the ranks are $k/101$), so the choice changes nothing here (asserted); with another
      state it might, so say which one you used.
    - **Resample the unit that varies.** Bridges in one county share inspectors, budgets, and
      weather; the county intervals of the catch rate are {min(width_ratio.values()):.2f} to
      {max(width_ratio.values()):.2f} times as wide as bridge intervals (§4).
    - **Reproducing a bootstrap means drawing in the same order.** One generator serves every
      group, so a group's interval depends on the groups drawn before it. The figure script
      draws the quartiles in their own order ({", ".join(q_order)}), then only the 16 cells the
      slides show, sorted by name: not in the order pandas meets them in the file
      (`groupby(sort=False)`), which would tie every interval to the order of the rows, and not
      the "other" owners' cells, which would consume draws for numbers no one reads.
    - **Overlapping intervals are not a test.** Two 95% intervals can overlap while the
      difference's own interval excludes zero; compute the gap's interval from paired draws (here
      {100 * gap_ci[0]:.1f} to {100 * gap_ci[1]:.1f} points for the largest gap).
    - **An undefined rate is not zero.** A resample with no bridge that turned poor has no catch
      rate: leave it out (NaN, then `np.nanpercentile`, as the figure script does), rather than
      score it 0 with a guard such as `tp / max(tp + fn, 1)`. Count such draws before trusting a
      small cell's interval: here {sum(cells_zero_draws.values())} of
      {len(cells_zero_draws) * BOOT:,} cell draws, and scoring them 0 would move
      {"no" if _moved == 0 else _moved} interval (§5).
    - **A solver's tolerance is part of the method.** On the proxy problem, scikit-learn's
      default solver (L-BFGS at `tol=1e-4`) stops with weights up to
      {max(proxy_lbfgs_gap.values()):.3f} from the optimum Newton's method finds, though the AUC
      moves only in the fifth decimal; the figure script fits with `solver="newton-cholesky"`. In
      scikit-learn 1.8, `penalty=` is deprecated: `C` and `l1_ratio` name the penalty.
    - **An in-sample proxy test, and how to hold counties out.** The proxy AUC scores the same
      counties the classifier saw; on counties held out, about half of the signal above chance
      remains (§5). Hold out folds of counties and compute the AUC within each fold: holding
      out one county at a time and pooling the scores is biased low when the label is a county
      attribute (a classifier with no inputs scores {proxy_loco_no_inputs:.3f} that way). For
      an audit of a model used on the same counties, the in-sample number is the relevant one;
      for a claim about new places, it is not.
    - **Thresholds per group use the group at decision time.** That is a policy decision with
      legal limits in many settings, not a tuning knob.
    - **Time.** This run took {_minutes:.2f} minutes: the county bootstrap ({len(boot_ci)}
      groups × 1,000 draws) {boot_seconds:.2f} s; the check against the script's own bootstrap
      function {ci_check_seconds:.0f} s. The Newton fit took {newton_steps} steps and matches
      Lecture 6's {_fit}.

    ## Going deeper

    Forsyth's *Applied Machine Learning* has no chapter on fairness. Its §1.1.2 asks for the
    false-positive and false-negative rates together, never one alone; §3.1 shows why a held-out
    rate estimates the test rate when the held-out items are independent draws (here they are
    not: counties); §3.2.1 gives Hoeffding's inequality for a rate from $N$ Bernoulli cases,
    $P(|\\theta - X^{{(N)}}| \\ge \\epsilon) \\le 2e^{{-2N\\epsilon^2}}$ (Forsyth writes the sample
    mean as $X^{{(N)}}$; we wrote rates as TPR). The primary sources, paraphrased:

    - **Barocas, Hardt, and Narayanan (2023), *Fairness and Machine Learning*, chapter 3
      (classification).** It writes the group criteria as statements of (conditional)
      independence: independence (the decision does not depend on the group: demographic
      parity), separation (not given the outcome: equalized odds), and sufficiency (the outcome
      does not depend on the group given the score: calibration by group). It shows that, except
      in degenerate cases, no two of them can hold at once when base rates differ; how to reach
      separation after training by group-specific, possibly randomized thresholds read off each
      group's ROC curve; and that all three are observational, so they cannot tell apart
      situations whose causes differ.
    - **Chapter 4 (relative notions of fairness).** It steps back from formulas to the moral
      arguments behind them: when a relative, systematic disadvantage is wrongful, and narrow,
      middle, and broad views of equality of opportunity, the vocabulary for this lecture's vote.
    - **Mitchell et al. (2019), model cards.** A short document that ships with a trained model:
      its details, intended and out-of-scope uses, the factors (groups, instruments,
      environments) across which performance may vary, the metrics and decision thresholds,
      evaluation and training data, results broken down by factor and by intersections of
      factors, ethical considerations, and caveats; worked examples for a smile detector and a
      toxicity classifier.
    - **Gebru et al. (2021), datasheets for datasets.** By analogy with the datasheet that comes
      with an electronic component: questions on a dataset's motivation, composition, collection,
      preprocessing and labelling, uses, distribution, and maintenance, answered by its creators,
      so that users can judge whether it fits their task.
    - **Chouldechova (2017).** From the confusion matrix alone: a score with equal precision in two
      groups with different prevalence cannot also have equal false-positive and false-negative
      rates; applied to the recidivism data behind the public debate, with how unequal error rates
      turn into unequal penalties when the score informs a sentence.
    - **Hardt, Price, and Srebro (2016).** Defines equalized odds and equal opportunity, and
      derives a classifier that meets either from any score by group-specific thresholds
      (randomizing between two when needed), chosen by a linear program that minimizes the
      expected loss; unlike demographic parity, both allow a perfect predictor; FICO credit scores
      are the case study.

    **Exercises.** (1) Use Hoeffding's inequality to put a 95% interval on the catch rate of
    township bridges in the top quartile (55 turned poor), and compare it with the county
    interval of §5; which one assumes independent bridges, and which is wider? (2) Choose a pair
    of thresholds per quartile on the validation years that equalizes both the catch rate and the
    false-alarm rate (equalized odds; randomize between two thresholds where the ROC curves
    require it), apply them once to the test years, and report the expected cost per bridge
    against the single threshold's.

    ## Things to remember

    - Bias enters at labels, data, proxies, thresholds.
    - Leaving the group out does not keep it out.
    - Evaluate each group, with intervals, every cycle.
    - Know which error is the harm.
    - Fairness criteria conflict: choose, and say so.

    **Next:** AI as infrastructure.
    """)
    return


if __name__ == "__main__":
    app.run()
