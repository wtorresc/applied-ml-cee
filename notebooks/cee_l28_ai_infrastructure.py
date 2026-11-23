"""Applied Machine Learning for CEE — Lecture 28: AI as Infrastructure.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l28_ai_infrastructure.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: EPA eGRID2023 (revision 2) and Form EIA-930 (2024), both U.S. Government works: run
``bash scripts/fetch_grid.sh`` once. Source: U.S. Energy Information Administration (2025). The
course's GPU power was recorded once by ``scripts/measure_gpu_power.py`` into
``data/derived/gpu_power/``. Water factors: Macknick et al. (2012), Environ. Res. Lett. 7, 045802.
The slides' numbers come from ``lectures/build/fig/cee_l28_ai_infrastructure/`` (``make`` in
``lectures/``). One setting: the whole notebook runs on a laptop CPU in about a minute. The
optional "your own watts" cell trains for ``L28_OWN_SECONDS`` seconds (default 10; 0 skips it), on
an NVIDIA GPU if it finds one; ``L28_OWN_DEVICE=cpu`` forces the CPU estimate, and
``L28_CPU_WATTS`` sets the processor power that estimate assumes. The GPU ledger reads each GPU
lecture's ``numbers.json`` from ``lectures/build/fig/``, as the figure script does, so rebuild
Lecture 28's figures after any GPU lecture's.
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
    # Applied Machine Learning for CEE — Lecture 28: AI as Infrastructure

    **Slides:** *Applied Machine Learning for CEE*, Lecture 28 (`lectures/cee_l28_ai_infrastructure`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026), whose lectures on the ethics and impact of AI and on the course's conclusion this one follows in structure.

    **Reading:** Forsyth's *Applied Machine Learning* (Springer 2019) has no chapter on AI's footprint. The methods used here are in §10.2 (least squares and $R^2$), §10.3.1 (one point far from the rest swings a regression: today's glitch), and §11.2 (robust regression; §11.2.2 on a robust scale, the median absolute deviation of Lecture 9). For the rest: Siler-Evans et al. (2012) on marginal emission factors, EPA's eGRID2023 technical guide, EIA's notes on Form EIA-930, Macknick et al. (2012) on water, and the LBNL report of 2024 (see Going deeper).

    **Data:**

    - **EPA eGRID2023** (revision 2, June 2025; a U.S. Government work): 2023 generation and CO2 by plant, balancing authority, and state. `bash scripts/fetch_grid.sh` downloads the workbook to `data/raw/grid/` and converts the three sheets used here (plants, balancing authorities, states) to CSV in `data/derived/egrid2023/`.
    - **Form EIA-930**, the Hourly Electric Grid Monitor's six-month balance files for 2024: hourly demand and net generation by energy source for PJM and MISO, from the same script. **Source: U.S. Energy Information Administration (2025),** Form EIA-930, 2024 data (the six-month files as downloaded on November 20, 2026). Public domain; EIA asks to be acknowledged with the publication date.
    - **Water consumed per MWh by fuel:** Macknick, Newmark, Heath, and Hallett (2012), "Operational water consumption and withdrawal factors for electricity generating technologies: a review of existing literature", *Environ. Res. Lett.* 7, 045802, doi:10.1088/1748-9326/7/4/045802, Tables 1 and 2 (cooling-tower medians), used with attribution under the article's Creative Commons licence (the copy in `data/papers/` states CC BY-NC-SA 3.0).
    - **The course's GPU**, measured once: an RTX 3090 Ti's power every 0.5 s while training and while generating text, November 20, 2026 (`scripts/measure_gpu_power.py`, which writes `data/derived/gpu_power/`). This notebook reads that record and never rewrites it.
    - **U.S. data centres:** Shehabi et al. (2024), the Lawrence Berkeley National Laboratory report for DOE, as the slides quote it.

    The slides' numbers: `lectures/build/fig/cee_l28_ai_infrastructure/numbers.json`, written by the figure script `lectures/figures/cee_l28_ai_infrastructure.py`.

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the question and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**,
    which carry CS 441's outline for the topic (impact, the environment, responsibility; then the
    course's recap and its go-to methods). Written here from scratch: the eGRID reader and each
    region's CO2 rate by fuel from its plants; the EIA-930 reader, with the July 2024 change of
    columns mapped by keyword and PJM's one blank hour interpolated; hourly CO2, the average
    factors, and the month-by-hour table; the marginal factor by least squares on hour-to-hour
    changes, cleaned by Lecture 9's robust rule and by a check that each hour's fuel columns add
    up to its net generation, with an interval that resamples whole weeks and its range under
    other cleanings; the water the power plants consume per MWh; the 300 MW campus; and the GPU
    ledger from the recorded measurement and each GPU lecture's own record of its run. Each
    piece is checked three ways:

    - **Against the figure script's own functions**, loaded from its file as `ref`: the same
      sheets, fuel rates, hourly frames, marginal factors (draw for draw: the script's seed, and
      PJM's 500 draws before MISO's, as the script makes them), water, and GPU power.
    - **Against the slides' `numbers.json`**, read programmatically: the notebook
      rebuilds the whole file, key for key, and asserts that it equals the slides'.
    - **Against the deck's own words:** every number the deck states is rendered here from this
      notebook's values and found word for word in the deck; statements without a number of
      their own are measured, with a verdict (§7).

    **One setting.** The EIA-930 files are about 90 MB and are read twice (here and by the
    script's reader), the marginal factors' intervals take 500 draws each, and five other
    cleanings of the same data are fitted for §5.7. The whole notebook took under a minute on the
    workstation with §4.3 on the CPU (November 20, 2026, while other jobs shared its processors),
    so a laptop needs about a minute (see Gotchas for this run's time). The one optional part,
    **your own watts** (§4.3), trains a ResNet-18 for ten seconds: on an NVIDIA GPU it samples
    `nvidia-smi`, and without one it times the CPU and multiplies by a stated power, an
    estimate. Its results are printed, never asserted.

    **Your tasks** (from the slides): work out one campus's energy and CO2 for a year; read the
    GPU's measured watts and this course's own footprint; build hourly CO2 for PJM and MISO from
    public data; estimate the marginal factor with an honest interval; choose the factor that
    belongs in a filing; and say who answers for a design an AI helped make.
    """)
    return


@app.cell
def _():
    import importlib.util
    import json
    import os
    import random
    import re
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib.patches import FancyBboxPatch

    T_START = time.perf_counter()
    SEED = 28
    # PYTHONHASHSEED reaches only processes started from here; this notebook starts none that
    # hash strings (nvidia-smi is not Python)
    os.environ["PYTHONHASHSEED"] = "0"

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)

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
                return base.resolve()
        raise FileNotFoundError(
            "data/ and lectures/ not found: run from the repository, notebooks/, or ipynb/"
        )

    ROOT = find_root()
    # The figure scripts' shared modules on the path, and the slides' figure script itself, whose
    # functions are the reference for this notebook's. It has this notebook's file name, so it is
    # loaded from its path under another module name (an import of the same name fails marimo's
    # check).
    FIGURES = ROOT / "lectures" / "figures"
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    _spec = importlib.util.spec_from_file_location(
        "l28_figures", FIGURES / "cee_l28_ai_infrastructure.py"
    )
    ref = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = ref
    _spec.loader.exec_module(ref)

    # ------------------------------------------------ the slides' constants, written out here
    REGIONS = {"PJM": "PJM (ComEd, Chicago)", "MISO": "MISO (Ameren, downstate)"}
    LB_TO_KG = 0.45359237  # kg per lb (exact)
    GAL_TO_L = 3.785411784  # litres per US gallon (exact)
    LB_PER_SHORT_TON = 2000.0  # eGRID's CO2 is in short tons
    HOURS = 8760  # the campus's year
    HOURS_2024 = 8784  # the data's year: 2024 was a leap year
    DC_MW = 300.0  # the hypothetical campus's draw at its meter, all year (assumption)
    PUE = 1.2  # power usage effectiveness: facility energy / IT energy (assumption)
    WUE = (0.0, 0.5, 1.0, 2.0)  # on-site water, L per kWh of IT energy: swept (design-dependent)
    # Water consumed at the power plant, gal/MWh: Macknick et al. (2012), Tables 1 and 2, medians
    # with cooling towers. Coal: generic; gas: combined cycle; nuclear: generic; PV, wind:
    # Table 1; oil and other: natural-gas steam and biopower steam as stand-ins; hydropower's
    # reservoir evaporation left out (reservoirs serve many uses).
    WATER_GAL_PER_MWH = {
        "COAL": 687, "GAS": 205, "NUCLEAR": 672, "OIL": 826, "OTHER": 553, "SOLAR": 1,
        "WIND": 0, "HYDRO": 0, "STORAGE": 0, "GEO": 0, "UNKNOWN": 0,
    }  # fmt: skip
    FUELS = tuple(WATER_GAL_PER_MWH)
    # U.S. data centres: Shehabi et al. (2024), LBNL, as the slides quote it
    US_DC = {"twh_2023": 176, "share_2023_pct": 4.4, "twh_2028": (325, 580),
             "share_2028_pct": (6.7, 12)}  # fmt: skip
    # The GPU lectures' figure scripts and the kind of GPU work each does: "busy" near the limit
    # (training, feature extraction) or "generation" (L25's agents, the GPU mostly waiting). L19
    # mixes fine-tuning with generation and counts as busy: an upper estimate. L24 runs on the
    # CPU and is left out. Their minutes are read from their own numbers.json (§4.2).
    GPU_LECTURES = {
        "L16": ("cee_l16_segmentation", "busy"), "L17": ("cee_l17_transformers", "busy"),
        "L18": ("cee_l18_vision_foundation", "busy"), "L19": ("cee_l19_language_models", "busy"),
        "L20": ("cee_l20_time_series", "busy"), "L21": ("cee_l21_graph_networks", "busy"),
        "L22": ("cee_l22_point_clouds", "busy"), "L25": ("cee_l25_agents", "generation"),
    }  # fmt: skip
    # Where those numbers.json files are: the script's FIG_DIR, lectures/build/fig; this notebook
    # reads the same files as the slides' run
    GPU_FIG_DIR = ref.FIG_DIR
    assert GPU_FIG_DIR.resolve() == (ROOT / "lectures" / "build" / "fig").resolve()
    KIND_WATTS = {"busy": "training_w", "generation": "inference_w"}  # the measured phase
    DRIVE_10_MILES_KG = 4.0  # CS 441's comparison (about 400 g CO2 a mile for a typical car)
    BOOT = 500  # week-block bootstrap draws for each marginal factor's interval
    MAD_K = 5  # Lecture 9: a change beyond 5 robust standard deviations is a glitch
    MAD_SCALE = 1.4826  # the script's 1/Phi^-1(0.75) (Lecture 9: MAD / 0.6745 estimates sigma)
    # Data quality: an hour whose fuel columns miss its net generation by more than 5% of it is
    # left out of the marginal fit; the sensitivity tries the stricter 2% too
    FUEL_GAP_MAX, FUEL_GAP_STRICT = 0.05, 0.02
    WEEK_H = 24 * 7  # hours in a bootstrap block
    _mine = {
        "SEED": SEED, "REGIONS": REGIONS, "LB_TO_KG": LB_TO_KG, "GAL_TO_L": GAL_TO_L,
        "HOURS": HOURS, "DC_MW": DC_MW, "PUE": PUE, "WUE": WUE,
        "WATER_GAL_PER_MWH": WATER_GAL_PER_MWH, "FUELS": FUELS, "US_DC": US_DC,
        "GPU_LECTURES": GPU_LECTURES, "DRIVE_10_MILES_KG": DRIVE_10_MILES_KG, "BOOT": BOOT,
        "MAD_K": MAD_K, "MAD_SCALE": MAD_SCALE, "FUEL_GAP_MAX": FUEL_GAP_MAX,
        "FUEL_GAP_STRICT": FUEL_GAP_STRICT,
    }  # fmt: skip
    for _name, _value in _mine.items():  # the same settings as the slides
        assert _value == getattr(ref, _name), _name
    assert abs(MAD_SCALE - 1 / 0.6744897501960817) < 1e-4  # Phi^-1(0.75) = 0.67449

    # ------------------------------------------------ the data, where the fetch scripts put them
    EGRID_DIR = ROOT / "data" / "derived" / "egrid2023"
    EIA930_FILES = tuple(
        ROOT / "data" / "raw" / "grid" / f"EIA930_BALANCE_2024_{_h}.csv"
        for _h in ("Jan_Jun", "Jul_Dec")
    )
    GPU_CSV = ROOT / "data" / "derived" / "gpu_power" / "rtx3090ti_2026-11-20.csv"
    GPU_META = GPU_CSV.with_suffix(".json")
    assert ref.EGRID.resolve() == EGRID_DIR and ref.GPU_CSV.resolve() == GPU_CSV
    assert list(EIA930_FILES) == [_p.resolve() for _p in ref.EIA930]
    for _path, _how in (
        (EGRID_DIR / "PLNT23.csv", "bash scripts/fetch_grid.sh"),
        (EIA930_FILES[0], "bash scripts/fetch_grid.sh"),
        (EIA930_FILES[1], "bash scripts/fetch_grid.sh"),
        (GPU_CSV, "the course's measurement (scripts/measure_gpu_power.py)"),
    ):
        if not _path.exists():
            raise FileNotFoundError(f"{_path} not found: {_how}")

    # ------------------------------------------------ this notebook's own settings
    # The break-it intervals draw from a generator of their own, so the slides' draws keep their
    # order (the slides' generator is used for PJM's and MISO's 500 draws and nothing else)
    BREAK_SEED = 2028
    BREAK_DRAWS = 200  # week-block draws for each hour-of-day or month group (§5.2)
    # Your own watts (§4.3): seconds of training, the device, and the CPU power assumed without
    # a GPU (set it from your processor's specification: laptops draw roughly 15-45 W under
    # load, desktops 65-125 W)
    OWN_SECONDS = float(os.environ.get("L28_OWN_SECONDS", "10"))
    OWN_DEVICE = os.environ.get("L28_OWN_DEVICE", "auto").strip().lower()
    if OWN_DEVICE not in {"auto", "cpu"}:
        raise ValueError(f"L28_OWN_DEVICE must be 'auto' or 'cpu', not {OWN_DEVICE!r}")
    CPU_WATTS_ASSUMED = float(os.environ.get("L28_CPU_WATTS", "45"))
    OWN_BATCH = {"cuda": 64, "cpu": 8}  # the course's measurement used 64 on the GPU

    # The slides' run, every value as the figure script wrote it: the reference
    # (read programmatically, never typed)
    BUILD = ROOT / "lectures" / "build" / "fig" / "cee_l28_ai_infrastructure"
    NUMBERS = BUILD / "numbers.json"
    if not NUMBERS.exists():
        raise FileNotFoundError(
            f"{NUMBERS} not found: build the slides' figures with `make` in lectures/ (the "
            "figure script takes about 30 seconds on a CPU)"
        )
    REF = json.loads(NUMBERS.read_text(encoding="utf-8"))
    DECK = re.sub(
        r"\s+",
        " ",
        (ROOT / "lectures" / "cee_l28_ai_infrastructure.md").read_text(encoding="utf-8"),
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
        """The slides' value at ``path`` in numbers.json, e.g. "marginal/PJM/ci".

        Args:
            path: Keys separated by "/"; list indices as integers.

        Returns:
            The value.
        """
        node = REF
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def as_json(x: object) -> object:
        """``x`` as it reads back from JSON: string keys, lists, Python numbers.

        Args:
            x: Numbers, strings, None, NumPy scalars, and dicts, lists, and tuples of them.

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
            path: Its key in numbers.json.

        Raises:
            AssertionError: Saying where the two first differ.
        """
        a = as_json(x)
        assert agree(a, slide(path)), first_difference(a, slide(path), path)

    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    INK, MUTED = "#1A1A1A", "#595959"
    FIG = (9.0, 3.6)
    print(f"root {ROOT}; own watts: {OWN_SECONDS:g} s, device {OWN_DEVICE!r}")
    return (
        BOOT,
        BREAK_DRAWS,
        BREAK_SEED,
        CPU_WATTS_ASSUMED,
        DC_MW,
        DECK,
        DRIVE_10_MILES_KG,
        EGRID_DIR,
        EIA930_FILES,
        FIG,
        FUELS,
        FUEL_GAP_MAX,
        FUEL_GAP_STRICT,
        FancyBboxPatch,
        GAL_TO_L,
        GPU_CSV,
        GPU_FIG_DIR,
        GPU_LECTURES,
        GPU_META,
        HOURS,
        HOURS_2024,
        INK,
        KIND_WATTS,
        LB_PER_SHORT_TON,
        LB_TO_KG,
        MAD_K,
        MAD_SCALE,
        MUTED,
        OWN_BATCH,
        OWN_DEVICE,
        OWN_SECONDS,
        PALETTE,
        PUE,
        REF,
        REGIONS,
        ROOT,
        SEED,
        T_START,
        US_DC,
        WATER_GAL_PER_MWH,
        WEEK_H,
        WUE,
        agree,
        as_json,
        check,
        first_difference,
        importlib,
        json,
        np,
        pct,
        pd,
        plt,
        ref,
        rnd,
        shutil,
        subprocess,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 9–27

    On your own, no notes, two minutes:

    1. Lecture 9: how did we decide a reading was a glitch, not a burst?
    2. Lecture 26: what did a 7% discount rate do to the bridge policy?
    3. Lecture 27: what made a false alarm cheap and a miss costly?

    **Today:** AI's own footprint, and the engineer's name on the design.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. A reading far from the robust fit, judged by the median absolute deviation, not the
           mean. Today that rule cleans an hour of grid data (§4.6).
        2. It made the future cheap: at a poor-condition cost of 0.10, it never replaced. Today a
           choice of factor makes the same load look three times cleaner or dirtier.
        3. A flag brought resources, so a false alarm spent money while a miss left a bridge
           unwatched. Today the costs land on the grid, the river, and the air.
        """)
        }
    )
    return


@app.cell
def _(US_DC, check, mo):
    check(US_DC, "us_data_centres")
    _lo, _hi = US_DC["twh_2028"]
    _slo, _shi = US_DC["share_2028_pct"]
    mo.md(f"""
    ## 1. The engineering question: a data centre asks to connect

    | U.S. data centres (LBNL, 2024) | Electricity | Share of U.S. use |
    |:-------------------------------|------------:|------------------:|
    | 2023 | {US_DC["twh_2023"]} TWh | {US_DC["share_2023_pct"]}% |
    | 2028, projected | {_lo} to {_hi} TWh | {_slo}% to {_shi}% |

    A hypothetical **300 MW campus asks to connect in Illinois.** The utility's engineers (can
    the grid serve it?), the regulator (should it be approved, and what must the filing say?),
    and the engineers who design the campus (how is it cooled, and who seals the design?) each
    need the same three numbers: energy, emissions, and water. CS 441 asks what one query or one
    training run costs; a civil engineer asks what the grid and the river carry. An error costs
    either way: understate the emissions or the water and a permit rests on the wrong number;
    overstate them and a project is judged against a footprint it does not have.

    The public data behind it:

    - **EPA's eGRID2023**: every U.S. power plant's 2023 generation and CO2, with totals for
      each state and each balancing authority (the operator that balances supply and demand on
      a part of the grid). Northern Illinois (ComEd) is in **PJM**; central and southern
      Illinois (Ameren) are in **MISO**.
    - **Form EIA-930**: each balancing authority's demand and generation by fuel, every hour of
      2024.
    - **Macknick et al. (2012)**: the water each kind of power plant consumes per MWh.
    - **The course's own GPU**, measured while it worked.

    The LBNL figures above are the report's, as the slides quote them (checked against
    `numbers.json`); everything below is computed here.
    """)
    return


@app.cell
def _(EGRID_DIR, LB_PER_SHORT_TON, REGIONS, check, mo, np, pct, pd, ref, rnd, time):
    _t0 = time.perf_counter()

    def read_egrid(
        sheet: str, numeric: tuple[str, ...], percent: tuple[str, ...] = ()
    ) -> pd.DataFrame:
        """One eGRID2023 sheet as fetch_grid.sh converted it, the named columns as numbers.

        Row 1 of each CSV describes every field in words; row 2 holds the field codes, which
        become the column names. Numbers may carry thousands commas; shares are written as text
        such as "54.9%".

        Args:
            sheet: "PLNT23", "BA23", or "ST23".
            numeric: Columns to convert to floats (a blank stays NaN).
            percent: Share columns to convert to percentages, as floats.

        Returns:
            The sheet; every column not named stays text.

        Raises:
            ValueError: If a named column holds anything but a number.
        """
        df = pd.read_csv(EGRID_DIR / f"{sheet}.csv", header=1, dtype=str)
        for c in numeric:
            df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="raise")
        for c in percent:
            df[c] = pd.to_numeric(df[c].str.removesuffix("%"), errors="raise")
        return df

    def fuel_rates_lb_per_mwh(plants: pd.DataFrame, ba: str) -> dict[str, float]:
        """A balancing authority's 2023 CO2 per MWh by fuel, from its plants.

        Plants with positive net generation only; eGRID's CO2 is in short tons. Biomass, other
        fossil gases, and other fuels share one rate ("OTHER"), used also for EIA-930's unknown
        fuels; zero-carbon fuels have none.

        Args:
            plants: eGRID's plant sheet, ``PLNGENAN`` and ``PLCO2AN`` as numbers.
            ba: The balancing authority's code.

        Returns:
            lb CO2 per MWh for COAL, GAS, OIL, OTHER, and UNKNOWN.
        """
        p = plants[(plants["BACODE"] == ba) & (plants["PLNGENAN"] > 0)]
        by_fuel = p.groupby("PLFUELCT")[["PLNGENAN", "PLCO2AN"]].sum()  # a blank CO2 counts 0
        rate = LB_PER_SHORT_TON * by_fuel["PLCO2AN"] / by_fuel["PLNGENAN"]
        other = by_fuel.loc[["BIOMASS", "OFSL", "OTHF"]].sum()
        other_rate = float(LB_PER_SHORT_TON * other["PLCO2AN"] / other["PLNGENAN"])
        return {
            "COAL": float(rate["COAL"]),
            "GAS": float(rate["GAS"]),
            "OIL": float(rate["OIL"]),
            "OTHER": other_rate,
            "UNKNOWN": other_rate,
        }

    _mix_ba = ("BACLPR", "BAGSPR", "BANCPR", "BAWIPR")
    plants = read_egrid("PLNT23", ("PLNGENAN", "PLCO2AN"))
    ba_sheet = read_egrid(
        "BA23",
        ("BANGENAN", "BACO2AN", "BACO2RTA", "BACCO2RT", "BAGCO2RT", "BAOCO2RT", "BANBCO2"),
        _mix_ba,
    )
    states = read_egrid(
        "ST23",
        ("STNGENAN", "STCO2AN", "STCO2RTA", "STGENANC"),
        ("STNCPR", "STCLPR", "STGSPR", "STWIPR"),
    )
    assert not ba_sheet["BACODE"].duplicated().any() and not states["PSTATABB"].duplicated().any()
    il = states.loc[states["PSTATABB"] == "IL"].iloc[0]
    ba_rows = {_b: ba_sheet.loc[ba_sheet["BACODE"] == _b].iloc[0] for _b in REGIONS}
    rates = {_b: fuel_rates_lb_per_mwh(plants, _b) for _b in REGIONS}

    # The same numbers as the figure script's reader and rates (its reader checks that the
    # columns it uses were read as numbers; ours converts the ones named, from text)
    _used = {
        "PLNT23": ("PLNGENAN", "PLCO2AN"),
        "BA23": ("BANGENAN", "BACO2AN", "BACO2RTA"),
        "ST23": ("STNGENAN", "STCO2AN", "STCO2RTA", "STGENANC"),
    }
    _theirs = {_s: ref.egrid(_s, _cols) for _s, _cols in _used.items()}
    for _sheet, _df in (("PLNT23", plants), ("BA23", ba_sheet), ("ST23", states)):
        for _c in _used[_sheet]:
            assert np.array_equal(
                _df[_c].to_numpy(float), _theirs[_sheet][_c].to_numpy(float), equal_nan=True
            ), (_sheet, _c)
    for _b in REGIONS:
        _r = ref.fuel_rates(_theirs["PLNT23"], _b)
        assert list(_r) == list(rates[_b])
        assert all(abs(rates[_b][_f] - _r[_f]) <= 1e-12 * _r[_f] for _f in _r), _b

    # eGRID's totals hang together: the state's rate is its CO2 over its generation, and a
    # region's plants (all of them, negative generation included) add up to its row; our rule
    # (positive generation only) moves the region's rate by under 0.3%. Our coal and gas rates
    # match eGRID's own by-fuel rates for the region to 0.1%.
    assert abs(LB_PER_SHORT_TON * il["STCO2AN"] / il["STNGENAN"] - il["STCO2RTA"]) < 1e-3
    plant_sums = {}
    for _b, _row in ba_rows.items():
        _p = plants[plants["BACODE"] == _b]
        _all = LB_PER_SHORT_TON * _p["PLCO2AN"].sum() / _p["PLNGENAN"].sum()
        _pos = _p[_p["PLNGENAN"] > 0]
        _positive = LB_PER_SHORT_TON * _pos["PLCO2AN"].sum() / _pos["PLNGENAN"].sum()
        assert abs(_all - _row["BACO2RTA"]) < 0.01 and abs(_positive / _all - 1) < 0.003, _b
        for _f, _code in (("COAL", "BACCO2RT"), ("GAS", "BAGCO2RT")):
            assert abs(rates[_b][_f] / _row[_code] - 1) < 1e-3, (_b, _f)
        # fossil plants with generation but no CO2 on record (they count as zero above)
        _fossil = _pos[_pos["PLFUELCT"].isin(["COAL", "GAS", "OIL"])]
        _no_co2 = _fossil.loc[_fossil["PLCO2AN"].isna(), "PLNGENAN"].sum()
        plant_sums[_b] = {"all": _all, "positive": _positive,
                          "no_co2_share": float(_no_co2 / _fossil["PLNGENAN"].sum())}  # fmt: skip
        assert plant_sums[_b]["no_co2_share"] < 0.001, _b
    # Illinois's nuclear share: its nuclear generation over its net generation, as eGRID's own
    # share field (STNCPR, text to one decimal) also states
    il_nuclear_pct = 100 * il["STGENANC"] / il["STNGENAN"]
    assert round(il_nuclear_pct, 1) == il["STNCPR"]
    egrid_num = {
        "illinois_twh": rnd(il["STNGENAN"] / 1e6, 1),
        "illinois_lb_per_mwh": rnd(il["STCO2RTA"], 0),
        "illinois_nuclear_pct": pct(il["STGENANC"] / il["STNGENAN"], 1),
        "ba_lb_per_mwh": {_b: rnd(_row["BACO2RTA"], 0) for _b, _row in ba_rows.items()},
    }
    check(egrid_num, "egrid_2023")
    t_egrid = time.perf_counter() - _t0

    _rows = [
        f"| Illinois, all plants in the state | {il['STNGENAN'] / 1e6:.1f} | "
        f"{il['STCO2RTA']:.0f} | {il['STNCPR']:g}% | {il['STCLPR']:g}% | {il['STGSPR']:g}% | "
        f"{il['STWIPR']:g}% |"
    ] + [
        f"| {_b}, all plants in the region | {_row['BANGENAN'] / 1e6:.1f} | "
        f"{_row['BACO2RTA']:.0f} | {_row['BANCPR']:g}% | {_row['BACLPR']:g}% | "
        f"{_row['BAGSPR']:g}% | {_row['BAWIPR']:g}% |"
        for _b, _row in ba_rows.items()
    ]
    _rate_rows = [
        f"| {_b} | "
        + " | ".join(f"{rates[_b][_f]:,.0f}" for _f in ("COAL", "GAS", "OIL", "OTHER"))
        + " |"
        for _b in REGIONS
    ]
    _table, _rate_table = "\n    ".join(_rows), "\n    ".join(_rate_rows)
    mo.md(f"""
    **eGRID2023, read here** (`PLNT23`, `BA23`, `ST23`; equal to the script's reader on every
    column used, {t_egrid:.1f} s):

    | 2023 | Generation, TWh | CO2, lb/MWh | Nuclear | Coal | Gas | Wind |
    |:-----|----------------:|------------:|--------:|-----:|----:|-----:|
    {_table}

    **Each region's CO2 rate by fuel**, from its plants (lb per MWh; equal to the script's
    `fuel_rates`; coal and gas within 0.1% of eGRID's own by-fuel rates for the region):

    | Region | Coal | Gas | Oil | Other (biomass, other gases, other fuels) |
    |:-------|-----:|----:|----:|------------------------------------------:|
    {_rate_table}

    A region's plants add up to its eGRID row (PJM {plant_sums["PJM"]["all"]:.1f} against
    {ba_rows["PJM"]["BACO2RTA"]:.2f}); keeping only plants with positive generation, as the
    rates do, gives {plant_sums["PJM"]["positive"]:.1f} (MISO {plant_sums["MISO"]["positive"]:.1f}
    against {ba_rows["MISO"]["BACO2RTA"]:.1f}). Fossil plants with generation but no CO2 on record
    are under 0.1% of each region's fossil generation.
    """)
    # the states each region's generating plants lie in
    ba_states = {
        _b: int(
            plants.loc[(plants["BACODE"] == _b) & (plants["PLNGENAN"] > 0), "PSTATABB"].nunique()
        )
        for _b in REGIONS
    }
    assert all(_n > 1 for _n in ba_states.values())
    return ba_rows, ba_states, egrid_num, il, il_nuclear_pct, rates


@app.cell
def _(DC_MW, HOURS, il, mo):
    mo.md(f"""
    ### Your turn: one campus, one year

    A {DC_MW:.0f} MW campus draws its full load all year ({HOURS:,} hours).

    1. How many TWh a year? What share of Illinois's {il["STNGENAN"] / 1e6:.1f} TWh of 2023
       generation?
    2. At {il["STCO2RTA"]:.0f} lb CO2 per MWh, how many tonnes of CO2?

    Two minutes, with a calculator, before you open the answer.
    """)
    return


@app.cell
def _(DC_MW, HOURS, LB_TO_KG, il, mo):
    _mwh = DC_MW * HOURS
    _t = il["STCO2RTA"] * _mwh * LB_TO_KG / 1000
    mo.accordion(
        {
            "Answer": mo.md(f"""
        {DC_MW:.0f} MW × {HOURS:,} h = {_mwh:,.0f} MWh = {_mwh / 1e6:.2f} TWh, which is
        {100 * _mwh / il["STNGENAN"]:.1f}% of Illinois's generation. At {il["STCO2RTA"]:.0f}
        lb/MWh (Illinois's average over all its plants in 2023): {_mwh:,.0f} MWh ×
        {il["STCO2RTA"]:.1f} lb/MWh × {LB_TO_KG:.4f} kg/lb ≈ {_t / 1e6:.2f} million tonnes.
        Keep the number: the lecture asks whether
        {il["STCO2RTA"]:.0f} was the right factor.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: the chain of infrastructure

    AI is a load on physical systems that civil and environmental engineers design and permit:
    buildings, substations, transmission, generation, cooling water. Each link of the chain
    multiplies what the link before it asks for, and at the end of the chain someone signs.
    """)
    return


@app.cell
def _(FIG, FancyBboxPatch, INK, PALETTE, plt):
    _links = ["a model\n(training, queries)", "GPU servers", "data centre", "the grid",
              "power plants,\nrivers, air"]  # fmt: skip
    _adds = ["hours of work", "watts: measured,\nnot nameplate", "× PUE (cooling, losses)\n+ WUE: water on site",
             "× CO2 per MWh:\nwhich factor?", "fuel mix: CO2 and\ncooling water"]  # fmt: skip
    _fig, _ax = plt.subplots(figsize=(FIG[0], 3.4))
    _ax.set(xlim=(0, 10), ylim=(0, 4))
    _ax.axis("off")
    for _i, (_link, _add) in enumerate(zip(_links, _adds, strict=True)):
        _x = 0.1 + 2.0 * _i
        _ax.add_patch(FancyBboxPatch((_x, 2.5), 1.7, 1.0, boxstyle="round,pad=0.04",
                                     facecolor="#e8f0f7", edgecolor=INK, lw=1))  # fmt: skip
        _ax.text(_x + 0.85, 3.0, _link, ha="center", va="center", fontsize=9.5)
        _ax.text(_x + 0.85, 1.75, _add, ha="center", va="center", fontsize=8.5, color=PALETTE[1],
                 bbox={"facecolor": "#fbe3d4", "edgecolor": "none", "pad": 3})  # fmt: skip
        if _i < len(_links) - 1:
            _ax.annotate("", (_x + 2.02, 3.0), (_x + 1.76, 3.0),
                         arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2})  # fmt: skip
    _ax.add_patch(FancyBboxPatch((0.1, 0.25), 9.7, 0.7, boxstyle="round,pad=0.04",
                                 facecolor="#e3f1e8", edgecolor=PALETTE[2], lw=1))  # fmt: skip
    _ax.text(4.95, 0.6, "who answers: the utility's engineers, the regulator, the engineer of "
             "record who signs the design", ha="center", va="center", fontsize=9.5,
             color=PALETTE[2])  # fmt: skip
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **The assumption.** One factor turns energy into CO2; and, for the marginal factor, a new
    load is met the way the grid meets a change in demand from one hour to the next.

    ### The accounting, in four lines

    $$E_{\text{grid}} = P \cdot t \qquad E_{\text{IT}} = E_{\text{grid}} / \text{PUE}$$

    $$\text{CO}_2 = E_{\text{grid}} \cdot \text{EF} \qquad W = E_{\text{IT}} \cdot \text{WUE} + E_{\text{grid}} \cdot \text{EWIF}$$

    PUE (power usage effectiveness) is the facility's energy over its IT energy; we assume 1.2.
    WUE (water usage effectiveness) is the water evaporated on site per kWh of IT energy; it
    depends on the cooling design, so it is swept from 0 to 2 L/kWh. EF is the emission factor,
    the number in question. EWIF is the water the power plants consume per MWh they generate.
    Each line is a product, so a relative error in any factor is the same relative error in the
    answer.

    ### An average factor, from public data

    A region's CO2 in hour $t$ is its generation by fuel times each fuel's rate:

    $$C_t = \sum_f r_f\, g_{f,t}, \qquad r_f = 2000\, \frac{\sum_{p \in f} \text{CO}_{2,p}}{\sum_{p \in f} G_p}$$

    with $g_{f,t}$ from EIA-930 (MW in hour $t$, so MWh) and $r_f$ from the region's eGRID plants
    $p$ of fuel $f$ (short tons to lb). The average factor weights the hours by their
    generation:

    $$\bar a = \frac{\sum_t C_t}{\sum_t G_t}.$$

    ### A marginal factor, by least squares

    What does one more MWh of demand emit? Take hour-to-hour changes,
    $\Delta D_t = D_t - D_{t-1}$ and $\Delta C_t = C_t - C_{t-1}$, and fit
    $\Delta C_t = \alpha + m\, \Delta D_t + \varepsilon_t$ (Siler-Evans et al., 2012; least
    squares as in Forsyth §10.2):

    $$\hat m = \frac{\sum_t (\Delta D_t - \overline{\Delta D})(\Delta C_t - \overline{\Delta C})}{\sum_t (\Delta D_t - \overline{\Delta D})^2}.$$

    Changes, not levels: levels mix the plants that run all the time with the plants that
    respond; a change isolates the response.

    ### Why the marginal factor can exceed the average

    Differencing is linear, $\Delta C_t = \sum_f r_f\, \Delta g_{f,t}$, and so is the
    least-squares slope in its response, so

    $$\hat m = \sum_f r_f\, \hat\beta_f, \qquad \hat\beta_f = \text{the slope of } \Delta g_f \text{ on } \Delta D.$$

    The average weights each fuel's rate by its share of generation; the marginal factor weights
    it by how far that fuel moves when demand moves. A nuclear plant runs flat out whatever the
    demand ($\hat\beta \approx 0$), so the marginal factor exceeds the average exactly when the
    plants that follow demand are dirtier than the mix. §5.3 measures each $\hat\beta_f$.

    ### Cleaning, and an honest interval

    One reported change of tens of GW is a reporting glitch, and squared residuals let one such
    point swing the slope (Forsyth §10.3.1). Lecture 9's rule drops a change when

    $$|\Delta D_t - \operatorname{med}(\Delta D)| > 5 \times 1.4826 \times \operatorname{MAD}, \qquad \operatorname{MAD} = \operatorname{med}|\Delta D - \operatorname{med}(\Delta D)|,$$

    where $1.4826 \times \text{MAD}$ estimates the standard deviation of normal noise. Hours
    close in time share weather and plant schedules (§5.4 measures how much), so the interval
    resamples **whole weeks** with replacement (a block bootstrap), refits, and takes the 2.5th
    and 97.5th percentiles of 500 slopes.
    """)
    return


@app.cell
def _(DC_MW, HOURS, LB_TO_KG, PUE, il, mo):
    _mwh = DC_MW * HOURS
    mo.md(f"""
    **On the attempt's numbers:** E_grid = {DC_MW:.0f} MW × {HOURS:,} h = {_mwh:,.0f} MWh;
    E_IT = {_mwh:,.0f} / {PUE} = {_mwh / PUE:,.0f} MWh; at Illinois's {il["STCO2RTA"]:.0f}
    lb/MWh the CO2 is {il["STCO2RTA"] * _mwh * LB_TO_KG / 1e9:.2f} million tonnes, and every other
    factor below rescales that answer in proportion.
    """)
    return


@app.cell
def _(GPU_META, json, mo):
    _meta = json.loads(GPU_META.read_text(encoding="utf-8"))
    mo.md(f"""
    ## 4. Build and verify

    ### 4.1 Measure the watts, not the label

    The multiplier we can measure ourselves: the course's GPU ({_meta["name"]}, driver
    {_meta["driver"]}) was sampled with `nvidia-smi` every {_meta["sample_ms"]} ms on
    {_meta["date_utc"]}: idle, while training a ResNet-18 (synthetic 224 × 224 images, batches of
    {_meta["batch"]}), idle again, and while Lecture 25's language model
    ({_meta["qwen"]["model"]}, bfloat16) generated text, then idle. **Predict:** does
    generation draw more or less than half the training power?
    """)
    return


@app.cell
def _(GPU_CSV, GPU_META, check, json, pd, ref, rnd, time):
    _t0 = time.perf_counter()
    trace = pd.read_csv(GPU_CSV)
    gpu_meta = json.loads(GPU_META.read_text(encoding="utf-8"))
    assert list(trace.columns) == ["t_s", "watts", "util_pct", "mem_mib", "phase"]
    assert trace["t_s"].is_monotonic_increasing and trace["watts"].between(0, 1000).all()
    _watts = trace.groupby("phase")["watts"].mean()
    _util = trace.groupby("phase")["util_pct"].mean()
    _phases = {_p["phase"]: _p for _p in gpu_meta["phases"] if _p["phase"] != "idle"}
    gpu = {
        "limit_w": gpu_meta["power_limit_w"],
        "gpu": gpu_meta["name"],
        "training_w": float(_watts["training"]),
        "inference_w": float(_watts["inference"]),
        "idle_w": float(trace.loc[trace["phase"] == "idle", "watts"].quantile(0.05)),
        "training_util": float(_util["training"]),
        "inference_util": float(_util["inference"]),
        "images_per_s": _phases["training"]["images_per_s"],
        "tokens_per_s": _phases["inference"]["tokens_per_s"],
    }
    # The same as the script's gpu_power(); then the slides' numbers
    _theirs = ref.gpu_power()
    assert all(
        gpu[_k] == _theirs[_k] if isinstance(gpu[_k], str) else abs(gpu[_k] - _theirs[_k]) < 1e-9
        for _k in gpu
    )
    check({_k: (rnd(_v, 1) if isinstance(_v, float) else _v) for _k, _v in gpu.items()}, "gpu")
    gpu_samples = trace["phase"].value_counts().to_dict()
    # Plausibility: training just under the limit, generation well below it, idle near 15 W
    assert 0.9 * gpu["limit_w"] < gpu["training_w"] < gpu["limit_w"]
    assert 0.1 * gpu["limit_w"] < gpu["inference_w"] < 0.5 * gpu["limit_w"]
    assert 5 < gpu["idle_w"] < 40
    print(
        f"{len(trace)} samples ({gpu_samples}); training {gpu['training_w']:.1f} W, "
        f"generation {gpu['inference_w']:.1f} W, idle {gpu['idle_w']:.1f} W; equal to the "
        f"script's gpu_power() ({time.perf_counter() - _t0:.1f} s)"
    )
    return gpu, gpu_meta, trace


@app.cell
def _(FIG, INK, PALETTE, gpu, plt, trace):
    _fig, _ax = plt.subplots(figsize=FIG)
    _ax.plot(trace["t_s"] / 60, trace["watts"], color=PALETTE[0], lw=1)
    _ax.axhline(gpu["limit_w"], color=PALETTE[1], ls="--", lw=1)
    _ax.text(trace["t_s"].max() / 60, gpu["limit_w"] + 8, f"power limit {gpu['limit_w']:.0f} W",
             color=PALETTE[1], ha="right", fontsize=9)  # fmt: skip
    for _phase, _color, _hatch in (("training", PALETTE[2], ""), ("inference", PALETTE[3], "//")):
        _t = trace.loc[trace["phase"] == _phase, "t_s"] / 60
        _ax.axvspan(_t.min(), _t.max(), color=_color, alpha=0.12, lw=0, hatch=_hatch)
        _y = gpu[f"{_phase}_w"] - 45 if _phase == "training" else gpu[f"{_phase}_w"] + 15
        _ax.text((_t.min() + _t.max()) / 2, _y, f"{_phase}: {gpu[f'{_phase}_w']:.0f} W",
                 ha="center", fontsize=9, color=INK)  # fmt: skip
    _ax.set(xlabel="minute", ylabel="GPU board power, W", ylim=(0, 500),
            title=f"{gpu['gpu']}: idle about {gpu['idle_w']:.0f} W")  # fmt: skip
    _fig
    return


@app.cell
def _(gpu, mo):
    mo.md(f"""
    Training kept the GPU busy {gpu["training_util"]:.1f}% of the time
    ({gpu["images_per_s"]:,.0f} images a second) at {gpu["training_w"]:.0f} W, just under its
    {gpu["limit_w"]:.0f} W limit. Generation kept it busy {gpu["inference_util"]:.1f}% of the time
    ({gpu["tokens_per_s"]:.1f} tokens a second) at {gpu["inference_w"]:.0f} W: an estimate from
    the nameplate (the limit) would overstate that energy
    {gpu["limit_w"] / gpu["inference_w"]:.1f} times. Energy is power times time, and the power
    depends on the work: **measure under the work.**

    ### 4.2 This course's own footprint

    Each GPU lecture's figure-script time, read from its own record of its last run (`minutes` in
    its `numbers.json`), at the measured watts
    of its kind of work: busy (training, feature extraction) at the training phase's watts, or
    generation at the generation phase's. One full rebuild of the course's GPU figures. L19 mixes
    fine-tuning with generation and is counted as busy, an upper estimate (§5.6 brackets it);
    L24 runs on the CPU and is left out.

    **Runs that shared the GPU.** Each run's clock window ends when its `numbers.json` was written
    (a script's last step) and starts its recorded minutes earlier. Where windows overlap, the
    scripts shared the one GPU, and adding their energies counts those minutes twice; the board
    cannot draw more than its power limit, so over the runs' clock time the energy is at most the
    running scripts' watts, capped at the limit.
    """)
    return


@app.cell
def _(GPU_FIG_DIR, GPU_LECTURES, KIND_WATTS, check, gpu, json, pd, ref, rnd):
    # Each GPU lecture's last run: its recorded minutes and its clock window
    runs = {}
    for _name, (_stem, _kind) in GPU_LECTURES.items():
        _path = GPU_FIG_DIR / _stem / "numbers.json"
        if not _path.exists():  # being rebuilt: L28's numbers wait for it
            raise FileNotFoundError(f"{_path} not found: build {_name}'s figures, then L28's")
        _rec = json.loads(_path.read_text(encoding="utf-8")).get("minutes")
        assert _rec is not None, f"{_path} records no minutes"
        _min = float(_rec)
        _end = _path.stat().st_mtime  # POSIX seconds
        runs[_name] = {"minutes": _min, "kind": _kind, "start": _end - 60 * _min, "end": _end}
    _watts = {_k: gpu[_v] for _k, _v in KIND_WATTS.items()}
    by_lecture = {
        _n: {**_r, "kwh": _r["minutes"] / 60 * _watts[_r["kind"]] / 1000}  # h × W / 1000
        for _n, _r in runs.items()
    }
    # A sweep over the windows' starts and ends, in time order: between two events the set of
    # running scripts is fixed, and the board draws their watts, at most its limit
    _events = sorted([(_r["start"], 1, _n) for _n, _r in runs.items()]
                     + [(_r["end"], -1, _n) for _n, _r in runs.items()])  # fmt: skip
    _running, _t_prev, _kwh, _span_s = set(), None, 0.0, 0.0
    for _t, _step, _n in _events:
        if _running:
            _w = min(gpu["limit_w"], sum(_watts[runs[_m]["kind"]] for _m in _running))
            _kwh += (_t - _t_prev) / 3600 * _w / 1000
            _span_s += _t - _t_prev
        _running = _running | {_n} if _step > 0 else _running - {_n}
        _t_prev = _t
    assert not _running
    ledger = {
        "by_lecture": by_lecture,
        "kwh_sum": sum(_v["kwh"] for _v in by_lecture.values()),
        "kwh": _kwh,
        "script_minutes": sum(_r["minutes"] for _r in runs.values()),
        "span_minutes": _span_s / 60,
    }
    ledger["overlap_minutes"] = ledger["script_minutes"] - ledger["span_minutes"]

    def _utc(ts: float) -> str:
        return pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M") + " UTC"

    ledger_num = {
        "script_minutes": rnd(ledger["script_minutes"], 0),
        "gpu_hours": rnd(ledger["script_minutes"] / 60, 1),
        "span_minutes": rnd(ledger["span_minutes"], 0),
        "overlap_minutes": rnd(ledger["overlap_minutes"], 0),
        "kwh_sum": rnd(ledger["kwh_sum"], 2),
        "kwh": rnd(ledger["kwh"], 2),
        "by_lecture": {
            _n: {"script_minutes": rnd(_v["minutes"], 1),
                 "kind": _v["kind"], "kwh": rnd(_v["kwh"], 2),
                 "window": [_utc(_v["start"]), _utc(_v["end"])]}
            for _n, _v in by_lecture.items()
        },
    }  # fmt: skip
    # The figure script's own reading and ledger, then the slides' numbers: a difference here
    # means a GPU lecture was rebuilt after L28 (rebuild L28)
    _theirs = ref.gpu_ledger(ref.gpu_runs(), ref.gpu_power())
    for _k in ("kwh_sum", "kwh", "script_minutes", "span_minutes", "overlap_minutes"):
        assert abs(ledger[_k] - _theirs[_k]) <= 1e-9 * max(1.0, abs(_theirs[_k])), _k
    for _k, _v in ledger_num.items():  # its CO2 needs the factors (§4.7)
        check(_v, f"ledger/{_k}")
    # A few kWh: hours at a few hundred watts; capping the overlaps can only lower the sum
    assert 1 < ledger["kwh"] <= ledger["kwh_sum"] + 1e-9 < 10
    print(f"{ledger['script_minutes']:.1f} GPU-minutes over {ledger['span_minutes']:.1f} minutes "
          f"of clock time ({ledger['overlap_minutes']:.1f} overlapping); "
          f"{ledger['kwh_sum']:.3f} kWh summed, {ledger['kwh']:.3f} kWh with overlaps capped at "
          f"{gpu['limit_w']:.0f} W")  # fmt: skip
    return ledger, ledger_num, runs


@app.cell
def _(FIG, PALETTE, ledger, plt):
    _names = list(ledger["by_lecture"])
    _kwh = [ledger["by_lecture"][_n]["kwh"] for _n in _names]
    _gen = [ledger["by_lecture"][_n]["kind"] == "generation" for _n in _names]
    _fig, _ax = plt.subplots(figsize=FIG)
    _bars = _ax.bar(_names, _kwh, color=[PALETTE[3] if _g else PALETTE[2] for _g in _gen])
    for _bar, _g in zip(_bars, _gen, strict=True):
        if _g:
            _bar.set_hatch("//")
    for _i, _v in enumerate(_kwh):
        _ax.text(_i, _v + 0.02, f"{_v:.2f}", ha="center", fontsize=9)
    _ax.set(ylabel="GPU energy, kWh", ylim=(0, max(_kwh) * 1.15),
            title=f"one rebuild of the GPU lectures' figures: {ledger['kwh_sum']:.2f} kWh summed, "
            f"{ledger['kwh']:.2f} with overlaps capped (solid: busy; hatched: generation)")  # fmt: skip
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Your own watts (optional)

    The cell below trains a ResNet-18 (no pretrained weights) on synthetic 224 × 224 images for
    `L28_OWN_SECONDS` seconds (10 by default; 0 skips it). **With an NVIDIA GPU** it samples
    `nvidia-smi` every half second during the timed loop, batches of 64 as in the course's
    measurement. **Without one** (or if `L28_OWN_DEVICE=cpu`), it times the same loop on the
    CPU in batches of 8 and multiplies the time by a stated power, `L28_CPU_WATTS` (45 W unless
    you set it; use your processor's specification): an **estimate**, not a measurement. The
    comparable number is the energy per training image. Nothing here is asserted, and nothing is
    written to disk: the course's recorded measurement stays as it is.
    """)
    return


@app.cell
def _(
    CPU_WATTS_ASSUMED,
    OWN_BATCH,
    OWN_DEVICE,
    OWN_SECONDS,
    SEED,
    gpu,
    mo,
    np,
    shutil,
    subprocess,
    time,
    torch,
):
    import torchvision as _torchvision

    def _train(device: str, seconds: float, before: object = None, after: object = None) -> dict:
        """ResNet-18 forward and backward passes on synthetic images until ``seconds`` pass.

        Args:
            device: "cuda" or "cpu".
            seconds: How long the timed loop runs.
            before: Called just before the timed loop (starts the GPU's sampler), or None.
            after: Called just after it (stops the sampler), or None.

        Returns:
            The device, batch, steps, seconds, and images per second.
        """
        if device == "cuda":  # as the course's measurement: float32, TF32 off
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        batch = OWN_BATCH[device]
        torch.manual_seed(SEED)
        model = _torchvision.models.resnet18(weights=None).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (batch,), device=device)

        def sync() -> None:
            if device == "cuda":
                torch.cuda.synchronize()

        def step() -> None:
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        step()  # one untimed step: memory and kernel selection
        sync()
        if before is not None:
            before()
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            step()
            steps += 1
            if steps % 10 == 0:
                sync()
        sync()
        elapsed = time.perf_counter() - t0
        if after is not None:
            after()
        return {"device": device, "batch": batch, "steps": steps, "seconds": elapsed,
                "images_per_s": batch * steps / elapsed}  # fmt: skip

    def _gpu_query() -> tuple[float, float]:
        """The GPU's power (W) and utilization (%) now, from nvidia-smi."""
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout  # fmt: skip
        watts, util = (float(_s) for _s in out.strip().splitlines()[0].split(","))
        return watts, util

    def _gpu_run(seconds: float) -> dict:
        """Train on the GPU while nvidia-smi samples it every 0.5 s (in memory, no files)."""
        cmd = ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
               "--format=csv,noheader,nounits", "-lms=500", "-i", "0"]  # fmt: skip
        state = {}

        def start() -> None:
            state["proc"] = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

        def stop() -> None:
            state["proc"].terminate()
            state["out"], state["err"] = state["proc"].communicate(timeout=20)

        try:
            run = _train("cuda", seconds, before=start, after=stop)
        finally:  # our own sampler only: never left running
            if "proc" in state and state["proc"].poll() is None:
                state["proc"].kill()
                state["proc"].communicate(timeout=20)
            torch.cuda.empty_cache()
        rows = [_l.split(",") for _l in state["out"].splitlines() if _l.strip()]
        watts = np.array([float(_r[0]) for _r in rows])
        util = np.array([float(_r[1]) for _r in rows])
        if len(watts) < 3:
            raise ValueError(f"nvidia-smi gave {len(watts)} samples ({state['err'].strip()!r})")
        return {**run, "watts": float(watts.mean()), "util_pct": float(util.mean()),
                "samples": len(watts), "measured": True}  # fmt: skip

    own_watts = None
    _notes = []
    if OWN_SECONDS <= 0:
        _notes.append("skipped (L28_OWN_SECONDS is 0)")
    else:
        if OWN_DEVICE == "cpu":
            _notes.append("the CPU, as L28_OWN_DEVICE asks")
        elif shutil.which("nvidia-smi") is None:
            _notes.append("no nvidia-smi on this machine: the CPU estimate")
        elif not torch.cuda.is_available():
            _notes.append("nvidia-smi is here but PyTorch sees no CUDA device: the CPU estimate")
        else:
            try:
                _before = _gpu_query()
                own_watts = _gpu_run(OWN_SECONDS)
                own_watts["before_w"], own_watts["before_util"] = _before
                if _before[1] > 10:
                    _notes.append(f"the GPU was already {_before[1]:.0f}% busy before the run: "
                                  "another program's power is in these samples")  # fmt: skip
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as _err:
                _notes.append(f"the GPU measurement failed ({type(_err).__name__}: {_err}); "
                              "the CPU estimate instead")  # fmt: skip
                own_watts = None
        if own_watts is None:
            own_watts = {**_train("cpu", OWN_SECONDS), "watts": CPU_WATTS_ASSUMED,
                         "measured": False, "threads": torch.get_num_threads()}  # fmt: skip
    _course_j = gpu["training_w"] / gpu["images_per_s"]
    if own_watts is None:
        _out = mo.md(f"**Your own watts:** {'; '.join(_notes)}.")
    else:
        _j = own_watts["watts"] / own_watts["images_per_s"]
        _how = (
            f"measured: {own_watts['samples']} nvidia-smi samples, mean {own_watts['watts']:.0f} W "
            f"at {own_watts['util_pct']:.0f}% utilization (before the run: "
            f"{own_watts['before_w']:.0f} W)"
            if own_watts["measured"]
            else f"**an estimate**: {own_watts['threads']} CPU threads, at an assumed "
            f"{own_watts['watts']:.0f} W (L28_CPU_WATTS), not measured"
        )
        _out = mo.md(f"""
        **Your own watts** ({own_watts["device"]}, batches of {own_watts["batch"]}; {_how}):
        {own_watts["steps"]} steps in {own_watts["seconds"]:.1f} s,
        {own_watts["images_per_s"]:,.1f} images a second, **{_j:.3f} J per training image**
        ({own_watts["watts"] * own_watts["seconds"] / 3600:.3f} Wh for the run). The course's
        GPU: {_course_j:.3f} J per image ({gpu["training_w"]:.0f} W at
        {gpu["images_per_s"]:,.0f} images a second). {" ".join(_n + "." for _n in _notes)}
        Not asserted: your machine is not the course's.
        """)
    print("own watts:", own_watts, _notes)
    _out
    return (own_watts,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 The grid, hour by hour

    EIA-930's six-month files hold one row per balancing authority and hour: the hour's end in
    UTC and in local time, demand, net generation, and net generation by energy source, each as
    reported, imputed, and adjusted. We read the **adjusted** columns for PJM and MISO. Three
    things need care, and the reader below handles each explicitly:

    - **The columns changed in July 2024.** The second file splits hydropower from pumped
      storage, and solar and wind by whether a battery is attached, and adds batteries, other
      and unknown storage, and geothermal. Columns are mapped to fuels **by keyword**, and the reader stops
      on any column no keyword matches.
    - **A blank fuel is no generation; a blank hour is missing.** Fuels a region never reports
      (PJM's unknown fuels, MISO's petroleum) are blank all year and count as 0. But one PJM hour
      is blank in demand and in generation alike; filling it with zeros would put a false drop
      and recovery of the whole region's load into the hour-to-hour changes, so that hour is
      interpolated, fuels too.
    - **Numbers are numbers, or the reader stops.** The 2024 files carry no thousands commas;
      the reader would strip them, and anything else that is not a number stops it.

    One more check follows the reader: **do the fuel columns add up to net generation?** Hourly
    CO2 comes from the fuel columns, so an hour whose fuels do not add up to the hour's own net
    generation gives a wrong CO2 for that hour.
    """)
    return


@app.cell
def _(EIA930_FILES, FUELS, FUEL_GAP_MAX, HOURS_2024, REGIONS, np, pd, rates, ref, time):
    _t0 = time.perf_counter()
    WHEN = "%m/%d/%Y %I:%M:%S %p"
    GEN_PREFIX, GEN_SUFFIX = "Net Generation (MW) from ", "(Adjusted)"
    # Keyword -> fuel, tried in order: the specific names first ("hydropower and pumped" before
    # "pumped", "pumped" before "hydro", "solar"/"wind" before "storage")
    FUEL_KEYS = (
        ("coal", "COAL"), ("natural gas", "GAS"), ("nuclear", "NUCLEAR"), ("petroleum", "OIL"),
        ("hydropower and pumped", "HYDRO"),  # January-June: hydro with pumped storage
        ("excluding pumped", "HYDRO"),  # July-December: hydro on its own
        ("pumped", "STORAGE"), ("hydro", "HYDRO"), ("solar", "SOLAR"), ("wind", "WIND"),
        ("storage", "STORAGE"), ("geothermal", "GEO"), ("other fuel", "OTHER"),
        ("unknown fuel", "UNKNOWN"),
    )  # fmt: skip

    def fuel_for(column: str) -> str:
        """The fuel an EIA-930 "Net Generation (MW) from ... (Adjusted)" column belongs to.

        Args:
            column: The column's name.

        Returns:
            One of ``FUELS``.

        Raises:
            ValueError: For a column no keyword matches (a new category: map it by hand).
        """
        name = column.removeprefix(GEN_PREFIX).removesuffix(GEN_SUFFIX).strip().lower()
        for key, fuel in FUEL_KEYS:
            if key in name:
                return fuel
        raise ValueError(f"unmapped EIA-930 column: {column!r}")

    def eia_number(col: pd.Series) -> pd.Series:
        """EIA-930 numbers, written with or without thousands commas, as floats; blanks NaN.

        Raises:
            ValueError: On any text that is not a number.
        """
        return pd.to_numeric(col.str.replace(",", "", regex=False), errors="raise")

    def read_eia930(paths: tuple, regions: tuple[str, ...]) -> tuple[pd.DataFrame, dict, set]:
        """Both six-month files: demand, net generation, and generation by fuel, per hour.

        Args:
            paths: The two files, January-June then July-December.
            regions: Balancing authority codes.

        Returns:
            One row per region and hour (``ba``, ``t`` in UTC, ``local``, ``demand``,
            ``netgen``, and one column per fuel in MW, blank fuels as 0), each file's
            generation columns with their fuels, and the (region, fuel) pairs whose columns are
            blank in every hour of both files.
        """
        frames, columns, blank_fuel = [], {}, []
        for path in paths:
            header = pd.read_csv(path, nrows=0).columns
            gen = [c for c in header if c.startswith(GEN_PREFIX) and c.endswith(GEN_SUFFIX)]
            keep = ["Balancing Authority", "UTC Time at End of Hour", "Local Time at End of Hour",
                    "Demand (MW) (Adjusted)", "Net Generation (MW) (Adjusted)", *gen]  # fmt: skip
            raw = pd.read_csv(path, usecols=keep, dtype=str)
            raw = raw[raw["Balancing Authority"].isin(regions)]
            out = pd.DataFrame({
                "ba": raw["Balancing Authority"],
                "t": pd.to_datetime(raw["UTC Time at End of Hour"], format=WHEN),
                "local": pd.to_datetime(raw["Local Time at End of Hour"], format=WHEN),
                "demand": eia_number(raw["Demand (MW) (Adjusted)"]),
                "netgen": eia_number(raw["Net Generation (MW) (Adjusted)"]),
            })  # fmt: skip
            for f in FUELS:
                out[f] = 0.0
            columns[path.stem] = [(c, fuel_for(c)) for c in gen]
            for c, f in columns[path.stem]:  # in file order, as the script adds them
                out[f] += eia_number(raw[c]).fillna(0.0)
            # a fuel is blank for a region when every column mapped to it is empty all half-year
            empty = raw[gen].isna().groupby(raw["Balancing Authority"]).all()
            mapped = {f for _, f in columns[path.stem]}
            blank_fuel.append({
                (b, f) for b in regions for f in mapped
                if all(empty.loc[b, c] for c, g in columns[path.stem] if g == f)
            })  # fmt: skip
            frames.append(out)
        return pd.concat(frames, ignore_index=True), columns, set.intersection(*blank_fuel)

    def region_hours(d: pd.DataFrame, ba: str) -> tuple[pd.DataFrame, list]:
        """One region's 8,784 hours of 2024, the blank hour interpolated, with hourly CO2.

        Args:
            d: ``read_eia930``'s rows.
            ba: The region.

        Returns:
            The region's frame indexed by UTC hour, with ``co2_lb`` from its eGRID fuel rates,
            and the hours that were blank.
        """
        x = d[d["ba"] == ba].set_index("t").sort_index()
        assert not x.index.duplicated().any(), f"{ba}: an hour twice"
        full = pd.date_range(x.index[0], periods=HOURS_2024, freq="h")
        assert len(x) == HOURS_2024 and (x.index == full).all(), f"{ba}: hours missing"
        blank = x["netgen"].isna() | x["demand"].isna()
        assert (x.loc[blank, ["netgen", "demand"]].isna().all(axis=None)
                and blank.sum() <= 1), f"{ba}: {int(blank.sum())} blank hours"  # fmt: skip
        cols = ["demand", "netgen", *FUELS]
        x.loc[blank, cols] = np.nan  # its fuels, filled with 0 above, are blank too
        x[cols] = x[cols].interpolate(limit=3)
        assert x[cols].notna().all(axis=None)
        x["co2_lb"] = sum(x[f] * rates[ba][f] for f in rates[ba])  # MW for one hour = MWh
        # how far the fuel columns' sum is from the hour's net generation, as a share of it
        x["fuel_gap_share"] = (x["netgen"] - x[list(FUELS)].sum(axis=1)).abs() / x["netgen"]
        return x, list(x.index[blank])

    _rows, eia_columns, blank_fuels = read_eia930(EIA930_FILES, tuple(REGIONS))
    # fuels a region never reports, blank all year (filled with 0 above)
    assert {("PJM", "UNKNOWN"), ("MISO", "OIL"), ("MISO", "UNKNOWN")} <= blank_fuels, blank_fuels
    hours, blank_hours = {}, {}
    for _b in REGIONS:
        hours[_b], blank_hours[_b] = region_hours(_rows, _b)

    # The same frames as the script's hourly(), column for column, and the same fuel mapping
    ref_hours = ref.hourly(tuple(REGIONS))
    for _b in REGIONS:
        pd.testing.assert_frame_equal(hours[_b], ref_hours[_b], check_exact=True)
    assert all(ref.fuel_of(_c) == _f for _cols in eia_columns.values() for _c, _f in _cols)
    # The hours by local date: a 25-hour day where daylight time ended (3 November)
    _blank = blank_hours["PJM"][0]
    assert blank_hours["MISO"] == [] and len(blank_hours["PJM"]) == 1
    blank_local = hours["PJM"].loc[_blank, "local"]
    # The fuel columns against net generation: equal in MISO; in PJM within 0.3% over the year,
    # but not hour by hour
    fuel_gap = {}
    for _b, _x in hours.items():
        _sum = _x[list(FUELS)].sum(axis=1)
        fuel_gap[_b] = {
            "annual_ratio": float(_sum.sum() / _x["netgen"].sum()),
            "median_share": float(_x["fuel_gap_share"].median()),
            "hours_over_max": int((_x["fuel_gap_share"] > FUEL_GAP_MAX).sum()),
            "largest_gw": float((_sum - _x["netgen"]).abs().max() / 1000),
        }
    assert all(abs(_g["annual_ratio"] - 1) < 0.005 for _g in fuel_gap.values())
    assert fuel_gap["MISO"]["hours_over_max"] == 0 and fuel_gap["MISO"]["largest_gw"] < 0.01
    t_eia = time.perf_counter() - _t0
    print(f"{sum(len(_x) for _x in hours.values()):,} region-hours; blank: PJM {_blank} UTC "
          f"({blank_local} local), interpolated; equal to the script's hourly() "
          f"({t_eia:.1f} s, both readers)")  # fmt: skip
    return blank_hours, blank_local, eia_columns, fuel_gap, hours, ref_hours


@app.cell
def _(eia_columns, mo):
    _rows = []
    for _stem, _cols in eia_columns.items():
        _half = "Jan–Jun" if _stem.endswith("Jan_Jun") else "Jul–Dec"
        _rows += [
            f"| {_half} | {_c.removeprefix('Net Generation (MW) from ').removesuffix('(Adjusted)')} "
            f"| {_f} |"
            for _c, _f in _cols
        ]
    _table = "\n    ".join(_rows)
    mo.md(f"""
    **Every generation column of the two files, and its fuel** (one name carries EIA's own typo,
    "Solar witho Integrated Battery Storage"; the keyword still finds it):

    | File | Column (adjusted) | Fuel |
    |:-----|:------------------|:-----|
    {_table}
    """)
    return


@app.cell
def _(blank_hours, blank_local, fuel_gap, mo):
    _b = blank_hours["PJM"][0]
    mo.md(f"""
    PJM's blank hour ends at {_b:%Y-%m-%d %H:%M} UTC, {blank_local:%Y-%m-%d %H:%M} local: hour 25
    of 3 November, the 25-hour day on which daylight time ended (the files run by local date).
    MISO has none. The fuel columns add up to net generation over the year (PJM
    {100 * fuel_gap["PJM"]["annual_ratio"]:.2f}%, MISO {100 * fuel_gap["MISO"]["annual_ratio"]:.2f}%),
    but not hour by hour in PJM: the median hour misses by
    {100 * fuel_gap["PJM"]["median_share"]:.1f}% of net generation, the worst by
    {fuel_gap["PJM"]["largest_gw"]:.1f} GW, and {fuel_gap["PJM"]["hours_over_max"]:,} of 8,784 hours
    by more than 5% (MISO: {fuel_gap["MISO"]["hours_over_max"]}, never more than
    {1000 * fuel_gap["MISO"]["largest_gw"]:.0f} MW). CO2 comes from the fuel columns and the
    average factor divides by net generation; over the year the two agree. The hour-to-hour
    changes of §4.6 are another matter: those 5% hours are left out of the marginal fit, and §5.7
    asks how much that choice decides.

    ### 4.5 Hourly CO2 and the average factor
    """)
    return


@app.cell
def _(FUELS, check, hours, np, pct, rnd):
    avg = {_b: float(_x["co2_lb"].sum() / _x["netgen"].sum()) for _b, _x in hours.items()}
    aef = {_b: _x["co2_lb"] / _x["netgen"] for _b, _x in hours.items()}  # lb/MWh, hour by hour
    month_hour = {
        _b: aef[_b].groupby([_x["local"].dt.month.values, _x["local"].dt.hour.values]).mean()
        .unstack()
        for _b, _x in hours.items()
    }  # fmt: skip
    by_hour = {_b: aef[_b].groupby(_x["local"].dt.hour.values).mean() for _b, _x in hours.items()}
    by_month = {_b: aef[_b].groupby(_x["local"].dt.month.values).mean() for _b, _x in hours.items()}
    assert all(_g.shape == (12, 24) for _g in month_hour.values())
    hourly_num = {
        _b: {
            "p5": rnd(np.percentile(aef[_b], 5), 0),
            "p95": rnd(np.percentile(aef[_b], 95), 0),
            "by_hour_range": [rnd(by_hour[_b].min(), 0), rnd(by_hour[_b].max(), 0)],
            "by_month_range": [rnd(by_month[_b].min(), 0), rnd(by_month[_b].max(), 0)],
        }
        for _b in hours
    }
    mix_num = {
        _b: {_f: pct(_x[_f].clip(lower=0).sum() / _x["netgen"].sum(), 1)
             for _f in FUELS if _f != "UNKNOWN"}
        for _b, _x in hours.items()
    }  # fmt: skip
    check(hourly_num, "hourly")
    check(mix_num, "mix_pct")
    for _b, _v in avg.items():
        check(rnd(_v, 0), f"factors_lb_per_mwh/{_b} average (2024)")
    # Plausibility: between gas's rate (about 860) and below coal's (about 2,200) for a mix
    assert all(400 < _v < 1500 for _v in avg.values())
    print({_b: f"average {_v:.1f} lb/MWh" for _b, _v in avg.items()})
    return aef, avg, by_hour, by_month, hourly_num, mix_num, month_hour


@app.cell
def _(FIG, REGIONS, avg, hourly_num, mix_num, mo, month_hour, plt):
    _vmin = min(float(_g.min().min()) for _g in month_hour.values())
    _vmax = max(float(_g.max().max()) for _g in month_hour.values())
    _fig, _axes = plt.subplots(1, 2, figsize=FIG, sharey=True)
    for _ax, (_b, _g) in zip(_axes, month_hour.items(), strict=True):
        _im = _ax.imshow(_g.to_numpy(), aspect="auto", cmap="YlOrBr", vmin=_vmin, vmax=_vmax,
                         extent=(-0.5, 23.5, 12.5, 0.5))  # fmt: skip
        _ax.set(title=REGIONS[_b], xlabel="local hour (end of hour)", xticks=[0, 6, 12, 18, 23])
    _axes[0].set(ylabel="month", yticks=[1, 4, 7, 10, 12])
    _fig.colorbar(_im, ax=_axes, label="lb CO2 per MWh", shrink=0.9)
    _mix = {_b: ", ".join(f"{_f.lower()} {_v:g}%" for _f, _v in _m.items() if _v >= 1)
            for _b, _m in mix_num.items()}  # fmt: skip
    mo.vstack([
        mo.md(f"""
        **The 2024 averages:** PJM {avg["PJM"]:.1f} lb/MWh, MISO {avg["MISO"]:.1f} (rounded, the
        slides' values, asserted above). The mixes (shares of net generation): PJM {_mix["PJM"]}; MISO
        {_mix["MISO"]}. Hour by hour the factor ranges from {hourly_num["PJM"]["p5"]:.0f} to
        {hourly_num["PJM"]["p95"]:.0f} lb/MWh in PJM and {hourly_num["MISO"]["p5"]:.0f} to
        {hourly_num["MISO"]["p95"]:,.0f} in MISO (5th to 95th percentile). Each cell below is the
        mean of that month's hourly factors at that hour (unweighted; the annual average is
        weighted by generation).
        """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.6 The marginal factor

    **Predict before you run it:** in each region, is the marginal factor above or below the
    average? By how much? And how wide should an interval be that resamples whole weeks, against
    one that resamples single hours (§5.4)?
    """)
    return


@app.cell
def _(
    BOOT,
    FUELS,
    FUEL_GAP_MAX,
    FUEL_GAP_STRICT,
    MAD_K,
    MAD_SCALE,
    REGIONS,
    SEED,
    WEEK_H,
    avg,
    check,
    hours,
    np,
    pct,
    ref,
    ref_hours,
    rnd,
    time,
):
    _t0 = time.perf_counter()

    def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Least-squares slope and R² of ``y`` on ``x``, with an intercept.

        Args:
            x: The regressor, shape (n,).
            y: The response, shape (n,).

        Returns:
            The slope and R² (NaN when ``y`` is constant, as for a fuel never reported).
        """
        assert x.shape == y.shape and x.ndim == 1
        xc, yc = x - x.mean(), y - y.mean()
        sxx, sxy, syy = xc @ xc, xc @ yc, yc @ yc
        r2 = float(sxy**2 / (sxx * syy)) if syy > 0 else float("nan")  # y constant: undefined
        return float(sxy / sxx), r2

    def robust_keep(d: np.ndarray) -> np.ndarray:
        """Lecture 9's rule: keep changes within MAD_K robust standard deviations of the median.

        Args:
            d: Hour-to-hour changes.

        Returns:
            A boolean mask, True for the changes kept.
        """
        dev = np.abs(d - np.median(d))
        return dev <= MAD_K * MAD_SCALE * np.median(dev)

    def week_blocks(index: np.ndarray, positions: np.ndarray) -> list[np.ndarray]:
        """The positions of the kept changes, grouped by week of the year.

        Args:
            index: Each kept change's position among all the year's changes.
            positions: Where each kept change sits in the arrays being resampled.

        Returns:
            One array of positions per week that holds any kept change, weeks in order.
        """
        weeks = index // WEEK_H
        return [positions[weeks == w] for w in np.unique(weeks)]

    def block_bootstrap(
        x: np.ndarray,
        y: np.ndarray,
        blocks: list[np.ndarray],
        rng: np.random.Generator,
        draws: int,
    ) -> np.ndarray:
        """Slopes of ``y`` on ``x`` over resampled weeks: each draw takes as many weeks as there
        are, with replacement, and keeps every hour of each week drawn.

        Args:
            x: The regressor.
            y: The response.
            blocks: Positions in ``x`` and ``y``, one array per week.
            rng: The generator; one ``integers`` call per draw, as the script makes them.
            draws: How many slopes.

        Returns:
            The slopes, shape (draws,).
        """
        out = np.empty(draws)
        for i in range(draws):
            pick = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
            out[i] = ols(x[pick], y[pick])[0]
        return out

    def fuels_add_up(x: object, share: float) -> np.ndarray:
        """The changes whose two hours have fuel columns within ``share`` of net generation.

        Args:
            x: The region's hourly frame, with ``fuel_gap_share``.
            share: The largest gap allowed, as a share of the hour's net generation.

        Returns:
            A boolean mask over the year's ``len(x) - 1`` changes.
        """
        ok = x["fuel_gap_share"].to_numpy() <= share
        return ok[:-1] & ok[1:]

    def marginal_factor(x: object, rng: np.random.Generator) -> dict:
        """Marginal CO2 per MWh of one region: the slope of hourly CO2 changes on demand changes.

        Two cleanings: Lecture 9's rule on the change in demand, and the fuel-sum check at
        ``FUEL_GAP_MAX`` on both hours of a change.

        Args:
            x: The region's hourly frame.
            rng: The generator (PJM's draws come before MISO's).

        Returns:
            The slope, its 95% week-block interval, R², the slope without Lecture 9's rule, the
            glitches it dropped, the hours and changes the fuel-sum check dropped, and the kept
            changes.
        """
        dd = np.diff(x["demand"].to_numpy())  # MW from one hour to the next = MWh per hour
        dc = np.diff(x["co2_lb"].to_numpy())
        when = x.index[1:]
        fuel_ok = fuels_add_up(x, FUEL_GAP_MAX)
        raw_slope = ols(dd[fuel_ok], dc[fuel_ok])[0]
        robust = robust_keep(dd)
        keep = robust & fuel_ok
        slope, r2 = ols(dd[keep], dc[keep])
        idx = np.flatnonzero(keep)
        blocks = week_blocks(idx, np.arange(len(idx)))
        draws = block_bootstrap(dd[keep], dc[keep], blocks, rng, BOOT)
        return {
            "slope": slope,
            "ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "r2": r2,
            "raw_slope": raw_slope,
            "dropped": int((~robust).sum()),
            "glitches": [{"utc": f"{t:%Y-%m-%d %H:%M}", "gw": rnd(v / 1000, 1)}
                         for t, v in zip(when[~robust], dd[~robust], strict=True)],
            "fuel_gap_hours": int((x["fuel_gap_share"] > FUEL_GAP_MAX).sum()),
            "fuel_gap_changes": int((robust & ~fuel_ok).sum()),
            "fuel_ok": fuel_ok,
            "changes": int(keep.sum()),
            "weeks": len(blocks),
            "keep": keep,
            "dd": dd[keep],
            "dc": dc[keep],
            "draws": draws,
            "when": when,
        }  # fmt: skip

    def cleanings(x: object) -> dict[str, float]:
        """The marginal factor under other defensible cleanings, each with Lecture 9's rule.

        Args:
            x: The region's hourly frame.

        Returns:
            The slope with no fuel-sum check; with the stricter share; with each hour's fuels
            rescaled to sum to its net generation (CO2 scales with them; no hour dropped); with
            the change in net generation as the regressor; and with Lecture 9's rule on the
            change in CO2 too.
        """
        dd, dc = np.diff(x["demand"].to_numpy()), np.diff(x["co2_lb"].to_numpy())
        dn = np.diff(x["netgen"].to_numpy())
        _scale = (x["netgen"] / x[list(FUELS)].sum(axis=1)).to_numpy()
        dc_scaled = np.diff(x["co2_lb"].to_numpy() * _scale)
        robust, ok = robust_keep(dd), fuels_add_up(x, FUEL_GAP_MAX)
        _fits = {
            "no_fuel_check": (dd, dc, robust),
            "strict_fuel_check": (dd, dc, robust & fuels_add_up(x, FUEL_GAP_STRICT)),
            "fuels_rescaled": (dd, dc_scaled, robust),
            "netgen_regressor": (dn, dc, robust & ok),
            "co2_changes_trimmed": (dd, dc, robust & ok & robust_keep(dc)),
        }
        return {_k: ols(_x[_m], _y[_m])[0] for _k, (_x, _y, _m) in _fits.items()}

    _rng = np.random.default_rng(SEED)  # the script's generator: PJM's 500 draws, then MISO's
    mef = {_b: marginal_factor(hours[_b], _rng) for _b in REGIONS}
    sens = {_b: cleanings(hours[_b]) for _b in REGIONS}
    spans = {_b: (min(mef[_b]["slope"], *sens[_b].values()),
                  max(mef[_b]["slope"], *sens[_b].values())) for _b in REGIONS}  # fmt: skip

    # The same as the script's marginal(): its frames, a fresh generator, the same order
    _rng_ref = np.random.default_rng(SEED)
    _theirs = {_b: ref.marginal(ref_hours[_b], _rng_ref) for _b in REGIONS}
    for _b, _m in mef.items():
        _t = _theirs[_b]
        for _k in ("slope", "r2", "raw_slope"):
            assert abs(_m[_k] - _t[_k]) <= 1e-9 * abs(_t[_k]), (_b, _k)
        assert np.allclose(_m["ci"], _t["ci"], rtol=1e-9, atol=0), _b
        assert _m["glitches"] == _t["glitches"] and _m["dropped"] == _t["dropped"]
        assert _m["changes"] == _t["changes"] == len(_t["_dd"])
        assert (_m["fuel_gap_hours"], _m["fuel_gap_changes"]) == (
            _t["fuel_gap_hours"], _t["fuel_gap_changes"])  # fmt: skip
        assert np.array_equal(_m["keep"], _t["_keep"])
        _ts = ref.sensitivity(ref_hours[_b])
        assert all(abs(sens[_b][_k] - _ts[_k]) <= 1e-9 * abs(_ts[_k]) for _k in _ts), _b
        assert np.array_equal(_m["dd"], _t["_dd"]) and np.allclose(_m["dc"], _t["_dc"], rtol=1e-12)
        # and our closed form against NumPy's least squares
        assert np.isclose(_m["slope"], np.polyfit(_m["dd"], _m["dc"], 1)[0], rtol=1e-9, atol=0)
    marginal_num = {
        _b: {
            "slope": rnd(_m["slope"], 0),
            "ci": [rnd(_c, 0) for _c in _m["ci"]],
            "r2": rnd(_m["r2"], 2),
            "over_average_pct": pct(_m["slope"] / avg[_b] - 1, 0),
            "untrimmed": rnd(_m["raw_slope"], 0),
            "changes_dropped": _m["dropped"],
            "glitches": _m["glitches"],
            "changes": _m["changes"],
            "fuel_gap_hours": _m["fuel_gap_hours"],
            "fuel_gap_changes": _m["fuel_gap_changes"],
            "sensitivity": {_k: rnd(_v, 0) for _k, _v in sens[_b].items()},
            "range": [rnd(spans[_b][0], 0), rnd(spans[_b][1], 0)],
            "over_average_pct_range": [pct(_v / avg[_b] - 1, 0) for _v in spans[_b]],
        }
        for _b, _m in mef.items()
    }
    check(marginal_num, "marginal")
    _pjm = hours["PJM"][hours["PJM"]["fuel_gap_share"] <= FUEL_GAP_MAX]
    fuel_check_num = {
        "max_share_pct": pct(FUEL_GAP_MAX, 0),
        "strict_share_pct": pct(FUEL_GAP_STRICT, 0),
        "median_share_pct": {_b: pct(_x["fuel_gap_share"].median(), 1)
                             for _b, _x in hours.items()},
        "pjm_average_clean_hours": rnd(_pjm["co2_lb"].sum() / _pjm["netgen"].sum(), 0),
    }  # fmt: skip
    check(fuel_check_num, "fuel_check")
    # The robust finding: MISO's marginal factor is well above its average under every cleaning;
    # PJM's average hardly moves when the 5% hours are left out
    assert spans["MISO"][0] > 1.2 * avg["MISO"]
    assert abs(fuel_check_num["pjm_average_clean_hours"] - avg["PJM"]) < 0.005 * avg["PJM"]
    for _b, _m in mef.items():
        check(rnd(_m["slope"], 0), f"factors_lb_per_mwh/{_b} marginal (2024)")
        assert _m["ci"][0] < _m["slope"] < _m["ci"][1]
    # The glitch: demand fell and came back within the hour while generation did not move, so
    # the report, not the load, changed (net generation within 1 GW across it)
    _x = hours["PJM"]
    glitch = {}
    for _g in mef["PJM"]["glitches"]:
        _i = int(np.flatnonzero(_x.index.strftime("%Y-%m-%d %H:%M") == _g["utc"])[0])
        glitch[_g["utc"]] = {
            "demand_gw": (_x["demand"].iloc[_i] - _x["demand"].iloc[_i - 1]) / 1000,
            "netgen_gw": (_x["netgen"].iloc[_i] - _x["netgen"].iloc[_i - 1]) / 1000,
        }
    assert len(glitch) == 2 and all(abs(_v["netgen_gw"]) < 1 for _v in glitch.values())
    # both glitch changes pass the fuel-sum check, so Lecture 9's rule alone removes them
    _when = mef["PJM"]["when"].strftime("%Y-%m-%d %H:%M")
    assert all(mef["PJM"]["fuel_ok"][list(_when).index(_t)] for _t in glitch)
    _gl = list(glitch.values())
    assert _gl[0]["demand_gw"] * _gl[1]["demand_gw"] < 0  # down, then back up an hour later
    t_marginal = time.perf_counter() - _t0
    print(
        {_b: f"{_m['slope']:.1f} [{_m['ci'][0]:.1f}, {_m['ci'][1]:.1f}] lb/MWh, R² "
         f"{_m['r2']:.3f}, {_m['changes']:,} changes in {_m['weeks']} weeks"
         for _b, _m in mef.items()},
        f"equal to the script's marginal() ({t_marginal:.1f} s, both)",
    )  # fmt: skip
    return (
        block_bootstrap,
        fuel_check_num,
        glitch,
        marginal_num,
        mef,
        ols,
        sens,
        spans,
        week_blocks,
    )


@app.cell
def _(FIG, MUTED, PALETTE, REGIONS, avg, mef, mo, plt, spans):
    _fig, _axes = plt.subplots(1, 2, figsize=FIG)
    for _ax, (_b, _m) in zip(_axes, mef.items(), strict=True):
        _dd, _dc = _m["dd"] / 1000, _m["dc"] / 1e6  # GW and million lb, per hour
        _ax.scatter(_dd, _dc, s=2, alpha=0.15, color=MUTED, rasterized=True)
        _xs = [_dd.min(), _dd.max()]
        _ax.plot(_xs, [_v * _m["slope"] / 1000 for _v in _xs], color=PALETTE[1], lw=2,
                 label=f"marginal: {_m['slope']:,.0f} lb/MWh")  # fmt: skip
        _ax.plot(_xs, [_v * avg[_b] / 1000 for _v in _xs], color=PALETTE[0], lw=1.5, ls="--",
                 label=f"average: {avg[_b]:,.0f} lb/MWh")  # fmt: skip
        _ax.set(title=REGIONS[_b], xlabel="change in demand, GW per hour")
        _ax.legend(fontsize=8, frameon=False, loc="upper left")
    _axes[0].set_ylabel("change in CO2, million lb per hour")
    _p, _q = mef["PJM"], mef["MISO"]
    mo.vstack([
        mo.md(f"""
        **The marginal factors:** PJM {_p["slope"]:,.1f} lb/MWh (week-block 95% interval
        {_p["ci"][0]:,.1f} to {_p["ci"][1]:,.1f}; R² {_p["r2"]:.2f}; {_p["changes"]:,} hourly
        changes), MISO {_q["slope"]:,.1f} ({_q["ci"][0]:,.1f} to {_q["ci"][1]:,.1f}; R²
        {_q["r2"]:.2f}; {_q["changes"]:,}). Both above the average: by
        {100 * (_p["slope"] / avg["PJM"] - 1):.1f}% and {100 * (_q["slope"] / avg["MISO"] - 1):.1f}%.
        Lecture 9's rule dropped {_p["dropped"]} PJM changes and {_q["dropped"]} in MISO; without
        it PJM's slope would be {_p["raw_slope"]:,.1f}. The fuel-sum check left out PJM's
        {_p["fuel_gap_hours"]} hours whose fuel columns miss net generation by more than 5%, and
        with them {_p["fuel_gap_changes"]} changes (none in MISO). The intervals resample weeks:
        they measure sampling, not that choice. Under the other cleanings of §5.7 PJM's slope
        runs from {spans["PJM"][0]:,.0f} to {spans["PJM"][1]:,.0f} lb/MWh, MISO's from
        {spans["MISO"][0]:,.0f} to {spans["MISO"][1]:,.0f}.
        """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(glitch, hours, mef, mo):
    _rows = [
        f"| {_t} | {_v['demand_gw']:+.1f} | {_v['netgen_gw']:+.1f} |" for _t, _v in glitch.items()
    ]
    _table = "\n    ".join(_rows)
    _x = hours["PJM"]
    _t = next(iter(glitch))
    _i = int((_x.index.strftime("%Y-%m-%d %H:%M") == _t).argmax())
    _frac = 1 - _x["demand"].iloc[_i] / _x["demand"].iloc[_i - 1]
    mo.md(f"""
    **The glitch, checked.** The changes the rule dropped (PJM; none in MISO):

    | Hour ending (UTC) | Change in demand, GW | Change in net generation, GW |
    |:------------------|---------------------:|-----------------------------:|
    {_table}

    Demand fell by {100 * _frac:.0f}% and came back an hour later while generation barely moved:
    the report changed, not the load. {len(glitch)} points out of the
    {mef["PJM"]["changes"] + mef["PJM"]["dropped"]:,} changes that pass the fuel-sum check
    moved the slope from {mef["PJM"]["raw_slope"]:,.0f} to {mef["PJM"]["slope"]:,.0f} lb/MWh.

    ### 4.7 Water per MWh, and the campus
    """)
    return


@app.cell
def _(
    DC_MW,
    DRIVE_10_MILES_KG,
    FUELS,
    GAL_TO_L,
    HOURS,
    LB_TO_KG,
    PUE,
    WATER_GAL_PER_MWH,
    WUE,
    avg,
    check,
    hours,
    il,
    ledger,
    mef,
    pct,
    ref,
    rnd,
):
    # Water the power plants consume per MWh of each region's 2024 mix (storage charging, a
    # negative generation, is no generation)
    water_gal_per_mwh = {
        _b: float(sum(_x[_f].clip(lower=0).sum() * WATER_GAL_PER_MWH[_f] for _f in FUELS)
                  / _x["netgen"].sum())
        for _b, _x in hours.items()
    }  # fmt: skip
    for _b, _x in hours.items():
        assert abs(water_gal_per_mwh[_b] - ref.water_per_mwh(_x)) <= 1e-12 * water_gal_per_mwh[_b]
    factors = {
        "Illinois, all plants (2023)": float(il["STCO2RTA"]),
        "PJM average (2024)": avg["PJM"],
        "MISO average (2024)": avg["MISO"],
        "PJM marginal (2024)": mef["PJM"]["slope"],
        "MISO marginal (2024)": mef["MISO"]["slope"],
    }
    mwh = DC_MW * HOURS  # the campus's grid energy, MWh a year
    it_kwh = mwh * 1000 / PUE  # its IT energy, kWh a year
    campus = {
        "co2_mt": {_k: _v * mwh * LB_TO_KG / 1e9 for _k, _v in factors.items()},  # million t
        "water_plants_billion_l": {_b: _w * mwh * GAL_TO_L / 1e9
                                   for _b, _w in water_gal_per_mwh.items()},
        "water_site_billion_l": {str(_w): _w * it_kwh / 1e9 for _w in WUE},
    }  # fmt: skip
    ledger_co2_kg = {_k: ledger["kwh"] / 1000 * _v * LB_TO_KG for _k, _v in factors.items()}
    campus_num = {
        "mw": DC_MW,
        "pue": PUE,
        "twh": rnd(mwh / 1e6, 2),
        "share_of_illinois_pct": pct(mwh / il["STNGENAN"], 1),
        "co2_mt": {_k: rnd(_v, 2) for _k, _v in campus["co2_mt"].items()},
        "water_plants_billion_l": {_b: rnd(_v, 1) for _b, _v in campus["water_plants_billion_l"].items()},
        "water_site_billion_l": {_k: rnd(_v, 1) for _k, _v in campus["water_site_billion_l"].items()},
    }  # fmt: skip
    check({_k: rnd(_v, 0) for _k, _v in factors.items()}, "factors_lb_per_mwh")
    check(rnd(max(factors.values()) / min(factors.values()), 1), "factor_spread")
    check({_b: rnd(_v, 0) for _b, _v in water_gal_per_mwh.items()}, "water_plants_gal_per_mwh")
    check({_b: rnd(_v * GAL_TO_L / 1000, 2) for _b, _v in water_gal_per_mwh.items()},
          "water_plants_l_per_kwh")  # fmt: skip
    check(campus_num, "datacentre")
    check({_k: rnd(_v, 2) for _k, _v in ledger_co2_kg.items()}, "ledger/co2_kg")
    check(DRIVE_10_MILES_KG, "ledger/drive_10_miles_kg")
    # Plausibility: a few hundred gallons per MWh (between wind's 0 and coal's 687); a 300 MW
    # load emits under 2 million tonnes at any factor below 1,700 lb/MWh
    assert all(200 < _w < 687 for _w in water_gal_per_mwh.values())
    assert all(0.3 < _v < 2 for _v in campus["co2_mt"].values())
    # The WUE at which the campus's own cooling water equals its power plants' (PJM's mix)
    wue_break_even = campus["water_plants_billion_l"]["PJM"] * 1e9 / it_kwh
    print(f"water {water_gal_per_mwh} gal/MWh; break-even WUE {wue_break_even:.2f} L/kWh")
    return (
        campus,
        campus_num,
        factors,
        ledger_co2_kg,
        mwh,
        water_gal_per_mwh,
        wue_break_even,
    )


@app.cell
def _(GAL_TO_L, campus, factors, mo, water_gal_per_mwh, wue_break_even):
    _co2 = campus["co2_mt"]
    _rows = [f"| {_k} | {factors[_k]:,.0f} | {_co2[_k]:.2f} |" for _k in factors]
    _site = campus["water_site_billion_l"]
    _table = "\n    ".join(_rows)
    mo.md(f"""
    **The campus, a year:**

    | Factor | lb CO2 per MWh | Million tonnes CO2 |
    |:-------|---------------:|-------------------:|
    {_table}

    **Water:** the power plants consume {water_gal_per_mwh["PJM"]:.1f} gallons per MWh of PJM's
    mix ({water_gal_per_mwh["PJM"] * GAL_TO_L / 1000:.3f} L/kWh) and
    {water_gal_per_mwh["MISO"]:.1f} of MISO's: {campus["water_plants_billion_l"]["PJM"]:.2f} and
    {campus["water_plants_billion_l"]["MISO"]:.2f} billion litres a year for the campus. On site,
    {", ".join(f"{_v:.2f}" for _v in _site.values())} billion litres at a WUE of
    {", ".join(_site)} L/kWh. The campus's own cooling would equal its power plants' (PJM) at a
    WUE of {wue_break_even:.2f} L/kWh.
    """)
    return


@app.cell
def _(
    BOOT,
    DRIVE_10_MILES_KG,
    GAL_TO_L,
    HOURS,
    MAD_K,
    REF,
    SEED,
    US_DC,
    WATER_GAL_PER_MWH,
    WUE,
    agree,
    as_json,
    campus_num,
    egrid_num,
    factors,
    first_difference,
    fuel_check_num,
    gpu,
    hourly_num,
    ledger_co2_kg,
    ledger_num,
    marginal_num,
    mix_num,
    mo,
    rnd,
    water_gal_per_mwh,
):
    # The slides' numbers.json, rebuilt key for key from this notebook's values
    numbers_mine = {
        "us_data_centres": US_DC,
        "egrid_2023": egrid_num,
        "factors_lb_per_mwh": {_k: rnd(_v, 0) for _k, _v in factors.items()},
        "factor_spread": rnd(max(factors.values()) / min(factors.values()), 1),
        "marginal": marginal_num,
        "fuel_check": fuel_check_num,
        "hourly": hourly_num,
        "mix_pct": mix_num,
        "water_plants_gal_per_mwh": {_b: rnd(_v, 0) for _b, _v in water_gal_per_mwh.items()},
        "water_plants_l_per_kwh": {_b: rnd(_v * GAL_TO_L / 1000, 2)
                                   for _b, _v in water_gal_per_mwh.items()},
        "datacentre": campus_num,
        "gpu": {_k: (rnd(_v, 1) if isinstance(_v, float) else _v) for _k, _v in gpu.items()},
        "ledger": {**ledger_num, "co2_kg": {_k: rnd(_v, 2) for _k, _v in ledger_co2_kg.items()},
                   "drive_10_miles_kg": DRIVE_10_MILES_KG},
        "wue": list(WUE),
        "macknick_gal_per_mwh": WATER_GAL_PER_MWH,
        "settings": {"seed": SEED, "boot": BOOT, "hours": HOURS, "mad_k": MAD_K},
    }  # fmt: skip
    _mine = as_json(numbers_mine)
    assert list(_mine) == list(REF), "the sections differ"
    assert agree(_mine, REF), first_difference(_mine, REF)

    def _leaves(x: object) -> int:
        if isinstance(x, dict):
            return sum(_leaves(v) for v in x.values())
        if isinstance(x, list):
            return sum(_leaves(v) for v in x)
        return 1

    n_values = _leaves(REF)
    mo.md(f"""
    ### 4.8 Every number in `numbers.json`

    **{n_values} values in {len(REF)} sections**, rebuilt here from the public data and the
    recorded measurement, equal to the slides' `numbers.json` (read from the file, never typed).
    """)
    return (n_values,)


@app.cell
def _(il, mo, mwh):
    mo.md(f"""
    ### Your turn: which factor?

    The same {mwh / 1e6:.2f} TWh a year in Illinois. Which emission factor belongs in the filing?

    1. Illinois's average over all its plants ({il["STCO2RTA"]:.0f} lb/MWh).
    2. The average of the grid that serves the campus (PJM or MISO).
    3. What the grid emits for one more MWh, when the campus is added.

    Two minutes. Ask what question each factor answers.
    """)
    return


@app.cell
def _(avg, il, mef, mo):
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        Each factor answers a different question. (1) {il["STCO2RTA"]:.0f} lb/MWh is what
        Illinois's plants emit per MWh they generate, and {il["STNCPR"]:g}% of that generation is
        nuclear (eGRID2023); it says nothing about where the power goes or what a new load adds.
        (2) {avg["PJM"]:.0f} (PJM) or {avg["MISO"]:.0f} (MISO) is what the grid the campus joins
        emits on average, over every MWh. (3) The new load is a change in demand, and the plants
        that follow demand are gas and coal (§5.3): {mef["PJM"]["slope"]:,.0f} lb/MWh in PJM and
        {mef["MISO"]["slope"]:,.0f} in MISO (§4.6). For a new load the third is the question;
        §5 shows how much the choice matters and where the marginal factor has limits of its own.
        """)
        }
    )
    return


@app.cell
def _(
    FIG,
    INK,
    MUTED,
    PALETTE,
    ba_states,
    campus,
    factors,
    il,
    il_nuclear_pct,
    mo,
    plt,
    spans,
):
    _names = list(factors)
    _fig, _ax = plt.subplots(figsize=FIG)
    _style = {"Illinois": (MUTED, ""), "average": (PALETTE[0], ""), "marginal": (PALETTE[1], "//")}
    for _i, _n in enumerate(_names[::-1]):
        _kind = next(_k for _k in _style if _k in _n)
        _ax.barh(_i, factors[_n], color=_style[_kind][0], hatch=_style[_kind][1])
        _right = factors[_n]
        if _kind == "marginal":  # the range under the other cleanings of §5.7
            _lo, _hi = spans[_n.split()[0]]
            _ax.errorbar(factors[_n], _i, xerr=[[factors[_n] - _lo], [_hi - factors[_n]]],
                         fmt="none", color=INK, capsize=4)  # fmt: skip
            _right = max(_right, _hi)
        _ax.text(_right + 15, _i, f"{factors[_n]:,.0f}", va="center", fontsize=9)
    _ax.set_yticks(range(len(_names)), _names[::-1])
    _ax.set(xlabel="lb CO2 per MWh (hatched: marginal; bracket: other cleanings, §5.7)",
            xlim=(0, max(*factors.values(), *(_h for _, _h in spans.values())) * 1.15))  # fmt: skip
    _co2 = campus["co2_mt"]
    # Illinois's rate with its nuclear generation taken out (nuclear emits no CO2)
    _no_nuclear = il["STCO2RTA"] / (1 - il_nuclear_pct / 100)
    mo.vstack([
        mo.md(f"""
        ## 5. Break it

        **Predict first, in writing:** which assumption of §3 fails first, and on which number?
        (One factor for every question; the same factor for every hour; hours that are
        independent; the ledger's minutes.) Each part below tests one.

        ### 5.1 One load, five factors

        The assumption "one emission factor" fails: the same campus emits
        {min(_co2.values()):.2f} to {max(_co2.values()):.2f} million tonnes a year, a ratio of
        {max(factors.values()) / min(factors.values()):.2f}. Illinois's own average is low
        because {il_nuclear_pct:.1f}% of its plants' 2023 generation is nuclear (eGRID2023):
        without it, the same CO2 over the rest of the generation would be {_no_nuclear:,.0f}
        lb/MWh. And a campus does not draw from Illinois's plants alone: it draws from PJM or
        MISO, whose generating plants lie in {ba_states["PJM"]} and {ba_states["MISO"]} states
        (eGRID2023), and as a new load it adds the marginal MWh. The brackets on the marginal
        bars are their range under other defensible cleanings of the same data (§5.7): wide in
        PJM ({spans["PJM"][0]:,.0f} to {spans["PJM"][1]:,.0f}), narrow in MISO
        ({spans["MISO"][0]:,.0f} to {spans["MISO"][1]:,.0f}).
        """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(
    BREAK_DRAWS,
    BREAK_SEED,
    FIG,
    PALETTE,
    REGIONS,
    block_bootstrap,
    by_hour,
    by_month,
    hours,
    mef,
    np,
    ols,
    pd,
    plt,
    time,
    week_blocks,
):
    _t0 = time.perf_counter()
    _rng = np.random.default_rng(BREAK_SEED)  # this notebook's own draws (not the slides')
    by_group = {}
    for _b in REGIONS:
        _m = mef[_b]
        _idx = np.flatnonzero(_m["keep"])  # each kept change's position among the year's
        _local = pd.DatetimeIndex(hours[_b]["local"].iloc[1:].to_numpy())[_m["keep"]]
        for _kind, _labels in (("hour", _local.hour), ("month", _local.month)):
            _rows = []
            for _g in np.unique(_labels):
                _pos = np.flatnonzero(_labels == _g)  # positions in the kept arrays
                _draws = block_bootstrap(_m["dd"], _m["dc"], week_blocks(_idx[_pos], _pos), _rng,
                                         BREAK_DRAWS)  # fmt: skip
                _lo, _hi = np.percentile(_draws, [2.5, 97.5])
                _rows.append({"group": int(_g), "slope": ols(_m["dd"][_pos], _m["dc"][_pos])[0],
                              "lo": float(_lo), "hi": float(_hi), "n": len(_pos)})  # fmt: skip
            by_group[f"{_b} {_kind}"] = pd.DataFrame(_rows).set_index("group")
    assert all(len(by_group[f"{_b} hour"]) == 24 and len(by_group[f"{_b} month"]) == 12
               for _b in REGIONS)  # fmt: skip
    break_ranges = {}
    for _b in REGIONS:
        _h, _mo = by_group[f"{_b} hour"], by_group[f"{_b} month"]
        _lo_h, _hi_h = int(_h["slope"].idxmin()), int(_h["slope"].idxmax())
        break_ranges[_b] = {
            "avg_hour": (float(by_hour[_b].min()), float(by_hour[_b].max())),
            "avg_month": (float(by_month[_b].min()), float(by_month[_b].max())),
            "marg_hour": (float(_h["slope"].min()), float(_h["slope"].max())),
            "marg_hour_at": (_lo_h, _hi_h),
            "marg_hour_ci": ((_h.loc[_lo_h, "lo"], _h.loc[_lo_h, "hi"]),
                             (_h.loc[_hi_h, "lo"], _h.loc[_hi_h, "hi"])),
            "marg_month": (float(_mo["slope"].min()), float(_mo["slope"].max())),
            "separate": bool(_h.loc[_lo_h, "hi"] < _h.loc[_hi_h, "lo"]),
        }  # fmt: skip
    # The text's words: the average varies more by month than by hour, the marginal factor more
    # by hour than by month, the extreme hours' intervals are apart; MISO keeps about 365 changes
    # an hour, PJM fewer on the ramps (the fuel-sum check, §5.7) but at least 200 in every hour
    hour_counts = {_b: (int(by_group[f"{_b} hour"]["n"].min()),
                        int(by_group[f"{_b} hour"]["n"].max())) for _b in REGIONS}  # fmt: skip
    for _b, _v in break_ranges.items():
        assert _v["avg_month"][1] - _v["avg_month"][0] > _v["avg_hour"][1] - _v["avg_hour"][0]
        assert _v["marg_hour"][1] - _v["marg_hour"][0] > _v["marg_month"][1] - _v["marg_month"][0]
        assert _v["separate"], _b
        assert hour_counts[_b][0] >= 200, (_b, hour_counts[_b])
    assert 350 <= hour_counts["MISO"][0] <= hour_counts["MISO"][1] <= 380
    t_groups = time.perf_counter() - _t0

    _fig, _axes = plt.subplots(2, 2, figsize=(FIG[0], 6.0), sharey="col")
    for _c, _b in enumerate(REGIONS):
        for _r, (_kind, _avg) in enumerate((("hour", by_hour[_b]), ("month", by_month[_b]))):
            _ax, _g = _axes[_r, _c], by_group[f"{_b} {_kind}"]
            _ax.errorbar(_g.index, _g["slope"], yerr=[_g["slope"] - _g["lo"], _g["hi"] - _g["slope"]],
                         color=PALETTE[1], marker="s", ms=3, lw=1, capsize=2,
                         label="marginal (95% week-block interval)")  # fmt: skip
            _ax.plot(_avg.index, _avg.to_numpy(), color=PALETTE[0], marker="o", ms=3, ls="--",
                     lw=1, label="average")  # fmt: skip
            _ax.set(xlabel=f"local {_kind}" + (" (end of hour)" if _kind == "hour" else ""))
            if _r == 0:
                _ax.set_title(REGIONS[_b])
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("lb CO2 per MWh")
    _axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    _fig.tight_layout()
    print(f"hour and month marginal factors with intervals ({t_groups:.1f} s):",
          {_b: {_k: _v for _k, _v in _r.items() if _k != "marg_hour_ci"}
           for _b, _r in break_ranges.items()})  # fmt: skip
    _fig
    return break_ranges, hour_counts


@app.cell
def _(BREAK_DRAWS, break_ranges, hour_counts, mo):
    _p, _q = break_ranges["PJM"], break_ranges["MISO"]

    def _span(r: tuple[float, float]) -> str:
        return f"{r[0]:,.0f} to {r[1]:,.0f}"

    _ci = {_b: [f"{_lo:,.0f} to {_hi:,.0f}" for _lo, _hi in _r["marg_hour_ci"]]
           for _b, _r in break_ranges.items()}  # fmt: skip
    mo.md(f"""
    ### 5.2 The same factor for every hour?

    **The average factor** changes more across the year than across the day: monthly means
    {_span(_p["avg_month"])} lb/MWh in PJM and {_span(_q["avg_month"])} in MISO, hourly means
    across the day {_span(_p["avg_hour"])} and {_span(_q["avg_hour"])}. That is the slides'
    heatmap.

    **But moving a load is a marginal question.** Fitted separately for each local hour and each
    month (each hour from {hour_counts["PJM"][0]} to {hour_counts["PJM"][1]} changes in PJM and
    {hour_counts["MISO"][0]} to {hour_counts["MISO"][1]} in MISO, intervals from {BREAK_DRAWS}
    week-block draws), the
    marginal factor varies **more across the day than across the year**: by hour,
    {_span(_p["marg_hour"])} in PJM (lowest at hour {_p["marg_hour_at"][0]}, interval
    {_ci["PJM"][0]}; highest at hour {_p["marg_hour_at"][1]}, {_ci["PJM"][1]}) and
    {_span(_q["marg_hour"])} in MISO (hour {_q["marg_hour_at"][0]}: {_ci["MISO"][0]}; hour
    {_q["marg_hour_at"][1]}: {_ci["MISO"][1]}); by month, {_span(_p["marg_month"])} and
    {_span(_q["marg_month"])}. The extremes' intervals do not overlap
    (PJM: {_p["separate"]}, MISO: {_q["separate"]}). So a flexible training run should be
    scheduled on the marginal factor, by hour, not on the average (the slides' heatmap notes
    point here); Siler-Evans et al. (2012) also report marginal factors by time of day and
    season. A caution: each hour's slope is fitted to that
    hour's own ramps (demand rises in the morning and falls at night), so it answers "what does
    the grid emit when demand changes at this hour", which is what a moved load asks, but on
    fewer and more similar points than the annual fit.
    """)
    return


@app.cell
def _(FUELS, REGIONS, avg, hours, mef, mo, np, ols, rates):
    betas = {}
    for _b in REGIONS:
        _keep = mef[_b]["keep"]
        betas[_b] = {_f: ols(mef[_b]["dd"], np.diff(hours[_b][_f].to_numpy())[_keep])[0]
                     for _f in FUELS}  # fmt: skip
        # the identity of §3: the marginal factor is each fuel's rate times its response
        _sum = sum(rates[_b][_f] * betas[_b][_f] for _f in rates[_b])
        assert abs(_sum - mef[_b]["slope"]) <= 1e-9 * mef[_b]["slope"], _b
    own_response = {_b: sum(betas[_b].values()) for _b in REGIONS}
    netgen_response = {
        _b: ols(mef[_b]["dd"], np.diff(hours[_b]["netgen"].to_numpy())[mef[_b]["keep"]])[0]
        for _b in REGIONS
    }
    # CO2 per MWh of the region's own generation that follows demand
    followers = {_b: mef[_b]["slope"] / own_response[_b] for _b in REGIONS}
    # The text's words: gas and coal follow demand most, nuclear and wind do not, the followers
    # are dirtier than the mix, and MISO's coal follows load more than twice as strongly
    for _b in REGIONS:
        assert set(sorted(betas[_b], key=lambda _f: -betas[_b][_f])[:2]) == {"GAS", "COAL"}
        assert abs(betas[_b]["NUCLEAR"]) < 0.05 and abs(betas[_b]["WIND"]) < 0.05
        assert followers[_b] > avg[_b] and own_response[_b] < 1.1
    assert betas["MISO"]["COAL"] > 2 * betas["PJM"]["COAL"]
    _shown = ("GAS", "COAL", "NUCLEAR", "WIND", "HYDRO", "SOLAR", "OIL", "OTHER")
    _rows = [
        f"| {_f.lower()} | {betas['PJM'][_f]:+.3f} | {rates['PJM'].get(_f, 0) * betas['PJM'][_f]:+,.0f} "
        f"| {betas['MISO'][_f]:+.3f} | {rates['MISO'].get(_f, 0) * betas['MISO'][_f]:+,.0f} |"
        for _f in _shown
    ]
    _table = "\n    ".join(_rows)
    mo.md(f"""
    ### 5.3 Why the marginal factor exceeds the average

    Each fuel's hour-to-hour change regressed on the change in demand (the same {mef["PJM"]["changes"]:,}
    and {mef["MISO"]["changes"]:,} changes), and its contribution $r_f \\hat\\beta_f$ to the
    marginal factor, lb/MWh:

    | Fuel | PJM: MWh per MWh of demand | PJM: lb/MWh | MISO: MWh per MWh | MISO: lb/MWh |
    |:-----|---------------------------:|------------:|------------------:|-------------:|
    {_table}

    The contributions add up to the marginal factors exactly (asserted): PJM
    {mef["PJM"]["slope"]:,.1f}, MISO {mef["MISO"]["slope"]:,.1f}. Gas and coal follow demand;
    nuclear and wind do not ($|\\hat\\beta|$ under 0.05). The generation that follows demand
    emits {followers["PJM"]:,.0f} lb per MWh in PJM and {followers["MISO"]:,.0f} in MISO,
    against averages of {avg["PJM"]:,.0f} and {avg["MISO"]:,.0f}: the followers are dirtier
    than the mix, so the slope exceeds the average. In MISO coal follows load
    ({betas["MISO"]["COAL"]:.2f} MWh per MWh) more than twice as strongly as in PJM
    ({betas["PJM"]["COAL"]:.2f}).

    **What the factor leaves out.** For each MWh of new demand, PJM's own fuel columns rise by
    {own_response["PJM"]:.2f} MWh ({netgen_response["PJM"]:.2f} by its net-generation column) and
    MISO's by {own_response["MISO"]:.2f} ({netgen_response["MISO"]:.2f}); the rest is met by
    imports from neighbouring regions, whose CO2 this factor does not count.
    """)
    return betas, followers, netgen_response, own_response


@app.cell
def _(BOOT, BREAK_SEED, REGIONS, mef, mo, np, ols):
    _rng = np.random.default_rng(BREAK_SEED + 1)  # this notebook's own draws
    iid_ci, resid_acf = {}, {}
    for _b in REGIONS:
        _m = mef[_b]
        _n = len(_m["dd"])
        _draws = np.empty(BOOT)
        for _i in range(BOOT):
            _pick = _rng.integers(0, _n, _n)
            _draws[_i] = ols(_m["dd"][_pick], _m["dc"][_pick])[0]
        iid_ci[_b] = np.percentile(_draws, [2.5, 97.5]).tolist()
        # Each residual at its own hour of the year (a change left out by the cleaning is NaN),
        # so a lag of 24 pairs the same hour a day later even where hours were dropped
        _e = np.full(len(_m["keep"]), np.nan)
        _e[_m["keep"]] = (_m["dc"] - _m["dc"].mean()) - _m["slope"] * (_m["dd"] - _m["dd"].mean())
        resid_acf[_b] = {}
        for _lag in (1, 24, 168):
            _a, _c = _e[:-_lag], _e[_lag:]
            _ok = np.isfinite(_a) & np.isfinite(_c)
            resid_acf[_b][_lag] = float(np.corrcoef(_a[_ok], _c[_ok])[0, 1])
    _w = {_b: mef[_b]["ci"][1] - mef[_b]["ci"][0] for _b in REGIONS}
    _iw = {_b: iid_ci[_b][1] - iid_ci[_b][0] for _b in REGIONS}
    assert all(_iw[_b] < _w[_b] for _b in REGIONS)  # hours treated as independent: too narrow
    print({_b: f"weeks {_w[_b]:.1f} wide, hours {_iw[_b]:.1f} ({_w[_b] / _iw[_b]:.2f}x); "
           f"residual autocorrelation {resid_acf[_b]}" for _b in REGIONS})  # fmt: skip
    # "correlated from day to day and week to week" (at one hour the sign differs by region)
    assert all(resid_acf[_b][_lag] > 0.1 for _b in REGIONS for _lag in (24, 168))
    _rows = "\n    ".join(
        f"| {_b} | {mef[_b]['ci'][0]:,.0f} to {mef[_b]['ci'][1]:,.0f} ({_w[_b]:.0f}) | "
        f"{iid_ci[_b][0]:,.0f} to {iid_ci[_b][1]:,.0f} ({_iw[_b]:.0f}) | "
        f"{resid_acf[_b][1]:+.2f}, {resid_acf[_b][24]:+.2f}, {resid_acf[_b][168]:+.2f} |"
        for _b in REGIONS
    )
    mo.md(f"""
    ### 5.4 Independent hours?

    The residuals of the marginal fit are correlated from day to day and from week to week (at
    24 hours and at one week, positive in both regions; at one hour the sign differs), so
    resampling single hours treats {mef["PJM"]["changes"]:,} correlated changes as independent
    and narrows the interval:

    | Region | Whole weeks (width) | Single hours (width) | Residual autocorrelation at 1 h, 24 h, 1 week |
    |:-------|--------------------:|---------------------:|----------------------------------------------:|
    {_rows}

    The week-block interval is {_w["PJM"] / _iw["PJM"]:.1f} times as wide in PJM and
    {_w["MISO"] / _iw["MISO"]:.1f} times in MISO ({BOOT} draws each; the single-hour draws come
    from a generator of this notebook's own). Lecture 3's lesson again: dependence decides what
    a resample may shuffle.
    """)
    return (iid_ci,)


@app.cell
def _(FIG, PALETTE, WUE, campus, mo, plt):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=FIG)
    _co2 = campus["co2_mt"]
    _names = list(_co2)[::-1]
    for _i, _n in enumerate(_names):  # hatched and orange: marginal; plain: average
        _marg = "marginal" in _n
        _a.barh(_i, _co2[_n], color=PALETTE[1] if _marg else PALETTE[0],
                hatch="//" if _marg else "")  # fmt: skip
    for _i, _n in enumerate(_names):
        _a.text(_co2[_n] + 0.02, _i, f"{_co2[_n]:.2f}", va="center", fontsize=8)
    _a.set_yticks(range(len(_names)), [_n.replace(" (20", "\n(20") for _n in _names], fontsize=8)
    _a.set(xlabel="million tonnes CO2 a year", xlim=(0, max(_co2.values()) * 1.3))
    _site = [campus["water_site_billion_l"][str(_w)] for _w in WUE]
    _plants = campus["water_plants_billion_l"]["PJM"]
    _x = range(len(WUE))
    _b.bar(_x, [_plants] * len(WUE), 0.6, color=PALETTE[2], label="at the power plants (PJM mix)")
    _b.bar(_x, _site, 0.6, bottom=_plants, color=PALETTE[0], hatch="//", label="on site (cooling)")
    for _k, _s in enumerate(_site):
        _b.text(_k, _plants + _s + 0.1, f"{_plants + _s:.1f}", ha="center", fontsize=8)
    _b.set_xticks(list(_x), [f"WUE {_w:g}" for _w in WUE], fontsize=8)
    _b.set(ylabel="billion litres a year", ylim=(0, (_plants + max(_site)) * 1.3))
    _b.legend(fontsize=7, frameon=False, loc="upper left")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
        ### 5.5 The campus, by factor and by water

        CO2 spans {min(_co2.values()):.2f} to {max(_co2.values()):.2f} million tonnes a year for
        the same campus: the factor is not a detail. A dry-cooled campus (WUE near 0) still
        consumes {_plants:.1f} billion litres a year through its grid's power plants.
        """),
        _fig,
    ])  # fmt: skip
    return


@app.cell
def _(GPU_LECTURES, KIND_WATTS, ROOT, gpu, ledger, mo, pd, runs):
    # The ledger's inputs: which lectures use CUDA at all (a text search of the figure scripts),
    # each run's window, and which runs overlapped
    _scripts = sorted((ROOT / "lectures" / "figures").glob("cee_l[0-9][0-9]_*.py"))
    cuda_lectures = sorted(
        f"L{int(_p.name[5:7])}" for _p in _scripts if "cuda" in _p.read_text(encoding="utf-8")
    )
    _left_out = sorted(set(cuda_lectures) - set(GPU_LECTURES))
    _not_cuda = sorted(set(GPU_LECTURES) - set(cuda_lectures))
    _l24 = [_p for _p in _scripts if _p.name.startswith("cee_l24_")]
    l24_uses_cuda = bool(_l24) and "cuda" in _l24[0].read_text(encoding="utf-8")
    # L19 mixes fine-tuning with generation: at the generation watts, the low end of the bracket
    _l19 = runs["L19"]["minutes"] / 60 / 1000
    ledger_low_kwh = ledger["kwh_sum"] - _l19 * (
        gpu[KIND_WATTS["busy"]] - gpu[KIND_WATTS["generation"]]
    )
    overlaps = sorted(
        (_a, _b, (min(runs[_a]["end"], runs[_b]["end"])
                  - max(runs[_a]["start"], runs[_b]["start"])) / 60)
        for _i, _a in enumerate(runs) for _b in list(runs)[_i + 1:]
        if runs[_a]["start"] < runs[_b]["end"] and runs[_b]["start"] < runs[_a]["end"]
    )  # fmt: skip

    def _local(ts: float) -> str:
        return pd.Timestamp(ts, unit="s", tz="UTC").strftime("%m-%d %H:%M")

    _except = f", except {', '.join(_left_out)}" if _left_out else ""
    _lacking = ", ".join(_not_cuda) if _not_cuda else "none"
    _l24_word = "uses" if l24_uses_cuda else "never mentions"
    _rows = "\n    ".join(
        f"| {_n} | {_r['kind']} | {_r['minutes']:.1f}"
        f" | {_local(_r['start'])} to "
        f"{_local(_r['end'])} | {ledger['by_lecture'][_n]['kwh']:.2f} |"
        for _n, _r in runs.items()
    )
    _pairs = "; ".join(f"{_a} and {_b}, {_m:.0f} min" for _a, _b, _m in overlaps) or "none"
    mo.md(f"""
    ### 5.6 The ledger's inputs, checked

    The ledger (§4.2) reads each lecture's minutes from its figure script's own record of its
    last run (`minutes` in its `numbers.json`), and the run's window from that file's time:

    | Lecture | Work | Minutes | Window (UTC) | kWh at its watts |
    |:--------|:-----|--------:|:-------------|-----------------:|
    {_rows}

    Runs that overlapped on the one GPU: {_pairs}. Together they overlapped for
    {ledger["overlap_minutes"]:.0f} of the {ledger["script_minutes"]:.0f} minutes, which is why
    the ledger caps those stretches at the board's limit ({ledger["kwh"]:.2f} kWh, against
    {ledger["kwh_sum"]:.2f} summed). A run on a shared GPU also takes longer than it would alone,
    so even the capped figure is an upper estimate.

    Every figure script that uses CUDA (a text search of `lectures/figures/`:
    {", ".join(cuda_lectures)}) is in the ledger{_except}; {_lacking} of the ledger's lectures
    lacks it. L24's script {_l24_word} CUDA, so it is left out. L19 mixes fine-tuning with
    generation; at the generation watts the summed rebuild would be {ledger_low_kwh:.2f} kWh
    instead of {ledger["kwh_sum"]:.2f}, so the slides' number is the upper end of that bracket.
    The ledger counts the GPU board only.
    """)
    return cuda_lectures, l24_uses_cuda, ledger_low_kwh, overlaps


@app.cell
def _(FIG, FUEL_GAP_MAX, PALETTE, REGIONS, avg, hours, mef, mo, np, plt, sens, spans):
    # Where PJM's fuel columns miss net generation: by local hour ending (Eastern time)
    _x = hours["PJM"]
    _hour = _x["local"].dt.hour.to_numpy()
    _share = _x["fuel_gap_share"].to_numpy()
    gap_by_hour = {int(_h): float(np.median(_share[_hour == _h])) for _h in range(24)}
    _bad = _share > FUEL_GAP_MAX
    ramps = (22, 23, 0, 6, 7)  # the evening ramp down and the morning ramp up
    ramp_share = float(np.isin(_hour[_bad], ramps).mean())
    assert ramp_share > 0.8  # "mostly on the ramps"
    # The cleanings, side by side, as shares of each region's average
    _names = {
        "chosen": "5% fuel-sum check (the slides')",
        "no_fuel_check": "no fuel-sum check",
        "strict_fuel_check": "2% fuel-sum check",
        "fuels_rescaled": "fuels rescaled to net generation",
        "netgen_regressor": "change in net generation as regressor",
        "co2_changes_trimmed": "Lecture 9's rule on CO2 changes too",
    }
    _slopes = {_b: {"chosen": mef[_b]["slope"], **sens[_b]} for _b in REGIONS}
    _rows = "\n    ".join(
        f"| {_label} | {_slopes['PJM'][_k]:,.0f} ({100 * (_slopes['PJM'][_k] / avg['PJM'] - 1):+.0f}%)"
        f" | {_slopes['MISO'][_k]:,.0f} ({100 * (_slopes['MISO'][_k] / avg['MISO'] - 1):+.0f}%) |"
        for _k, _label in _names.items()
    )
    # The words: MISO's marginal factor is well above its average under every cleaning; PJM's
    # range straddles its average
    assert all(_v > 1.3 * avg["MISO"] for _v in _slopes["MISO"].values())
    assert spans["PJM"][0] < avg["PJM"] < spans["PJM"][1]
    _fig, _ax = plt.subplots(figsize=FIG)
    _ax.plot(list(gap_by_hour), [100 * _v for _v in gap_by_hour.values()], color=PALETTE[0],
             marker="o", ms=4)  # fmt: skip
    _ax.axhline(100 * FUEL_GAP_MAX, color=PALETTE[1], ls="--", lw=1)
    _ax.set(xlabel="local hour (hour ending, Eastern)", xticks=range(0, 24, 3),
            ylabel="median |net gen. - fuels|, %", title="PJM, 2024: where the fuel columns miss")  # fmt: skip
    mo.vstack([
        mo.md(f"""
    ### 5.7 Does the cleaning decide the answer?

    §4.4 found that PJM's fuel columns do not add up to its net generation hour by hour. Where
    they miss: by local hour, the median gap is largest at the evening ramp down and the morning
    ramp up, and {100 * ramp_share:.0f}% of the hours that miss by more than 5% end at hours
    {", ".join(map(str, ramps))}. The fuel columns lag the ramps, so the hour-to-hour changes the
    marginal fit uses are wrong exactly where demand moves most. There is more than one
    defensible repair, and they disagree (lb/MWh, and against each region's average):

    | Cleaning | PJM | MISO |
    |:---------|----:|-----:|
    {_rows}

    MISO's fuel columns add up, and its marginal factor stays {100 * (spans["MISO"][0] / avg["MISO"] - 1):.0f}%
    to {100 * (spans["MISO"][1] / avg["MISO"] - 1):.0f}% above its average under every cleaning:
    a finding that survives the choice. PJM's runs from {spans["PJM"][0]:,.0f} to
    {spans["PJM"][1]:,.0f}, on both sides of its average ({avg["PJM"]:,.0f}): the data do not
    settle whether a new load in PJM emits more than the average MWh. The week-block interval
    measures sampling only; this spread is the larger uncertainty, and a filing should report it.
    """),
        _fig,
    ])  # fmt: skip
    return gap_by_hour, ramp_share


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Decide

    ### Vote: what should the regulator require?

    A data-centre filing in Illinois must report its CO2 with …

    - **A.** Illinois's average over all its plants.
    - **B.** its grid's annual average (PJM or MISO).
    - **C.** its grid's marginal factor, with an interval.
    - **D.** nothing: emissions are the utility's business.

    Vote alone. Discuss with a neighbor for two minutes. Vote again.
    """)
    return


@app.cell
def _(avg, campus, factors, il, mef, mo, spans):
    _co2 = campus["co2_mt"]
    _p, _q = mef["PJM"], mef["MISO"]
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        **A** answers a different question (what Illinois's plants emit) and understates a new
        load by up to a factor of {max(factors.values()) / il["STCO2RTA"]:.1f}:
        {_co2["Illinois, all plants (2023)"]:.2f} against {_co2["MISO marginal (2024)"]:.2f}
        million tonnes a year. **B** ({avg["PJM"]:.0f} or {avg["MISO"]:.0f} lb/MWh) describes the
        grid the campus joins, but not what it adds. **C** ({_p["slope"]:,.0f}, interval
        {_p["ci"][0]:,.0f} to {_p["ci"][1]:,.0f}, in PJM; {_q["slope"]:,.0f},
        {_q["ci"][0]:,.0f} to {_q["ci"][1]:,.0f}, in MISO) answers "what changes if we connect
        it", with its uncertainty, which in PJM includes how the data are cleaned
        ({spans["PJM"][0]:,.0f} to {spans["PJM"][1]:,.0f} lb/MWh, §5.7); it is the defensible
        choice for a new load; report B beside it. **D** leaves the largest number out of the
        filing.

        Two limits on C, from §5: an annual marginal factor fits a load that runs flat all year,
        as this campus does, but a load that moves in time needs the factors of the hours it
        runs (§5.2); and it counts only the region's own plants, not the imports that meet part
        of the new demand (§5.3).
        """)
        }
    )
    return


@app.cell
def _(campus, mo):
    _co2 = campus["co2_mt"]
    _w = campus["water_plants_billion_l"]
    _site = campus["water_site_billion_l"]
    mo.md(f"""
    ### Who answers for the design?

    | CS 441: responsibility requires | In licensure |
    |:--------------------------------|:-------------|
    | Control | the work done under the engineer's direction |
    | Knowledge | the engineer's detailed professional knowledge of it |
    | A name on it | the seal of the engineer of record |

    CS 441, after Coeckelbergh (2020): we hold someone responsible when they had control and
    knowledge. Illinois's Professional Engineering Practice Act of 1989 (225 ILCS 325, Section 4)
    ties "direct supervision/responsible charge" to the engineer's control of the work, or
    detailed professional knowledge of it. That is a paraphrase, as on the slides: **read the
    current text at ilga.gov** before relying on it. A model's output becomes the engineer's
    responsibility when the engineer seals it: the agent of Lecture 25, the controller of
    Lecture 26, and the classifier of Lecture 27 each needed a person who could explain them.

    ### Where it lives today

    | Today | What engineers do |
    |:------|:------------------|
    | Grid planning for data centres | load forecasts, interconnection studies |
    | Emissions accounting | average and marginal factors, by hour |
    | Water permits | cooling design, withdrawal and consumption |
    | AI in engineering practice | tools inside workflows; the engineer seals |

    CS 441's environmental slides count prompts and training runs; the same numbers, multiplied
    by millions of users, arrive at substations, interconnection queues, and water permits.
    eGRID and EIA-930 are the public data an engineer can check a filing against: this notebook
    rebuilt the slides' numbers from them.

    ### When one factor will not do

    | You need | Use |
    |:---------|:----|
    | What a state's plants emit | the state's eGRID rate |
    | What the grid a load joins emits | the region's average, by hour if the load varies |
    | What a new, flat load adds | the region's marginal factor, with an interval |
    | What a load that moves in time adds | marginal factors for the hours it runs |
    | Water | both: on site (WUE) and at the power plants |

    **Impact (environmental, economic, social).** The campus's CO2
    ({min(_co2.values()):.2f} to {max(_co2.values()):.2f} million tonnes a year) and water
    ({_w["PJM"]:.1f} billion litres at PJM's plants, plus up to {max(_site.values()):.1f} on site)
    land on the region's air and rivers, not on the people who use the computing. The choices
    that make them look large or small in a filing (the factor, the cooling design, the PUE)
    belong in the open, next to the numbers they produce.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion: 28 lectures, one thread

    CS 441's conclusion, on our data: the same pipeline (data, an encoder, a decoder, a
    prediction, a loss) ran through bridges, concrete, rivers, pipes, photos of cracks, laser
    scans, and the grid.

    | Unit | Lectures |
    |:-----|:---------|
    | 1. Learning from engineering data | L1 framing, L2 nearest neighbors, L3 generalization, L4 clustering, L5 least squares, L6 classifiers, L7 mixtures and Bayes, L8 trees and SHAP |
    | 2. Uncertainty, risk, and decisions | L9 robust estimation, L10 calibration, L11 conformal prediction and shift, L12 Gaussian processes |
    | 3. Deep learning for sensing | L13 SGD, L14 neural networks, L15 CNNs, L16 segmentation |
    | 4. Foundation models and time series | L17 transformers, L18 vision models, L19 language and RAG, L20 time series |
    | 5. Structure and physics | L21 graph networks, L22 point clouds, L23 PINNs, L24 SINDy and operators |
    | 6. Agents, decisions, the profession | L25 agents, L26 MDPs and RL, L27 fairness, L28 AI as infrastructure |

    ### Go-to methods, on CEE data

    | You have | Start with |
    |:---------|:-----------|
    | A small table | nearest neighbors, linear models (L2, L5, L6) |
    | A larger table | boosted trees (L8) |
    | Images, scans | a pretrained network, probe or fine-tune (L15-L18) |
    | Text | retrieval and a language model, checked (L19, L25) |
    | Physics you trust | a solver, or a model that obeys it (L23, L24) |
    | Decisions over time | value iteration if P is known (L26) |

    CS 441's standing advice holds here too: it is all about the data.

    ### Always, before you trust it

    | Habit | Lectures |
    |:------|:---------|
    | Split by time, place, or asset, not at random | 3, 6, 20 |
    | A baseline above, a noise floor below | every lecture |
    | Intervals, and calibration | 10, 11, 27 |
    | Checks between a model and a report | 25 |
    | Each group evaluated separately | 27 |
    | The engineer signs, and can explain | 28 |

    Bias, variance, leakage, and drift are not ML curiosities: they become a wrong pressure, a
    missed bridge, a flooded road. Today they became a glitch hour, a correlated week, and a
    factor that answered the wrong question.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Check yourself

    ### Exit question

    A vendor says its AI-designed cooling plant "cut the data centre's footprint 30%". What would
    you ask before you seal the design?

    **In the notebook:** the hourly grid, the marginal factor, the campus, and your own GPU's
    watts (§4.3).
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        Any two of: 30% of what: energy, CO2, or water?; at which emission factor, average or
        marginal, and for which grid and which hours?; on site only, or with the power plants'
        water?; measured or nameplate power?; against which baseline design?; and can the
        engineer of record explain the design well enough to take responsibility for it?
        """)
        }
    )
    return


@app.cell
def _(
    DC_MW,
    DECK,
    DRIVE_10_MILES_KG,
    FUEL_GAP_MAX,
    GAL_TO_L,
    HOURS,
    PUE,
    US_DC,
    WUE,
    aef,
    avg,
    by_hour,
    by_month,
    campus,
    factors,
    glitch,
    gpu,
    gpu_meta,
    il,
    il_nuclear_pct,
    ledger,
    ledger_co2_kg,
    mef,
    mo,
    month_hour,
    mwh,
    np,
    spans,
    water_gal_per_mwh,
    wue_break_even,
):
    _words = {1: "one", 2: "two", 3: "three", 4: "four"}
    _months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
               12: "December"}  # fmt: skip
    _p, _q = mef["PJM"], mef["MISO"]
    _IL = "Illinois, all plants (2023)"
    _il_rate = il["STCO2RTA"]
    _co2 = campus["co2_mt"]
    _wp, _site = campus["water_plants_billion_l"], campus["water_site_billion_l"]
    _bl = ledger["by_lecture"]
    _big = sorted((_n for _n in _bl if _bl[_n]["kwh"] >= 0.1), key=lambda _n: -_bl[_n]["kwh"])
    _small = [_n for _n in _bl if _bl[_n]["kwh"] < 0.1]
    _ledger_parts = [f"{_big[0]} {_bl[_big[0]]['kwh']:.2f} kWh"]
    _ledger_parts += [f"{_n} {_bl[_n]['kwh']:.2f}" for _n in _big[1:]]
    _under = f", and under 0.1 for {', '.join(_small)}" if _small else ""
    _ledger_co2 = list(ledger_co2_kg.values())
    _pct = {_b: np.percentile(aef[_b], [5, 95]) for _b in aef}
    _top2 = sorted(int(_m) for _m in by_month["MISO"].sort_values(ascending=False).index[:2])
    _glitch_utc = list(glitch)  # "YYYY-mm-dd HH:MM", the two dropped changes
    _gday, _gmonth = int(_glitch_utc[0][8:10]), _months[int(_glitch_utc[0][5:7])]
    _qwen = gpu_meta["qwen"]["model"]  # "Qwen/Qwen3.5-2B"
    _params = _qwen.rsplit("-", 1)[1].removesuffix("B")
    _spread = max(factors.values()) / min(factors.values())
    _w2 = water_gal_per_mwh
    _half = {500: "half"}[gpu_meta["sample_ms"]]
    _over = {_b: 100 * (mef[_b]["slope"] / avg[_b] - 1) for _b in avg}
    _range_pct = {_b: [100 * (_v / avg[_b] - 1) for _v in spans[_b]] for _b in avg}
    assert _range_pct["PJM"][0] < 0 < _range_pct["PJM"][1]  # "from 6% under ... to 22% over"
    _overlap = ledger["overlap_minutes"] > 1  # the runs shared the GPU: the capped variant
    _ledger_total = (
        f"{ledger['kwh_sum']:.1f} kWh summed, at most {ledger['kwh']:.1f} kWh because runs "
        "overlapped."
        if _overlap
        else f"{ledger['kwh_sum']:.1f} kWh in all."
    )
    _ledger_energy = (
        f"At the measured watts they sum to {ledger['kwh_sum']:.2f} kWh, but the last runs "
        f"shared the one GPU for {ledger['overlap_minutes']:.0f} of those minutes, and the board "
        f"cannot draw more than its {gpu['limit_w']:.0f} W limit: over the "
        f"{ledger['span_minutes']:.0f} minutes they spanned, at most {ledger['kwh']:.2f} kWh."
        if _overlap
        else f"At the measured watts they come to {ledger['kwh_sum']:.2f} kWh; the runs did not "
        "overlap on the GPU."
    )
    _factor_parts = []
    for _k, _v in factors.items():
        _part = f"{_k}, {_v:,.0f}"
        if "marginal" in _k:
            _lo, _hi = spans[_k.split()[0]]
            _part += f", between {_lo:,.0f} and {_hi:,.0f}"
            _part += " under other cleanings of the data" if _k.startswith("PJM") else ""
        _factor_parts.append(_part)

    def _render() -> list[str]:
        """The deck's statements, rendered from this notebook's values."""
        return [
        f"| 2023 | {US_DC['twh_2023']} TWh | {US_DC['share_2023_pct']}% |",
        f"| 2028, projected | {US_DC['twh_2028'][0]} to {US_DC['twh_2028'][1]} TWh | "
        f"{US_DC['share_2028_pct'][0]}% to {US_DC['share_2028_pct'][1]}% |",
        f"A {DC_MW:.0f} MW campus draws its full load all year ({HOURS:,} hours).",
        f"What share of Illinois's {il['STNGENAN'] / 1e6:.1f} TWh of 2023 generation?",
        f"At {_il_rate:.0f} lb CO2 per MWh, how many tonnes of CO2?",
        f"Our question: a hypothetical {DC_MW:.0f} MW campus asks to connect in Illinois.",
        f"Keep the number; the lecture asks whether {_il_rate:.0f} was the right factor.",
        f"Left: the {DC_MW:.0f} MW campus's CO2 a year, million tonnes:",
        f"{DC_MW:.0f} MW × {HOURS:,} h = {mwh / 1e6:.2f} TWh, {100 * mwh / il['STNGENAN']:.1f}% of "
        f"Illinois's generation. At {_il_rate:.0f} lb/MWh (Illinois's average over all its plants "
        f"in 2023) that is {_co2[_IL]:.2f} million tonnes.",
        f"PUE (power usage effectiveness): facility energy over IT energy; we assume {PUE}.",
        f"so we sweep it from {min(WUE):g} to {max(WUE):g} L/kWh.",
        f"The course's GPU ({gpu['gpu'].removeprefix('NVIDIA GeForce ')}) sampled every "
        f"{_half} second: about {gpu['idle_w']:.0f} W idle; "
        f"{gpu['training_w']:.0f} W while training a ResNet-18, just under its "
        f"{gpu['limit_w']:.0f} W limit; {gpu['inference_w']:.0f} W while a {_params}-billion-"
        "parameter language model generates text.",
        f"Training kept the GPU busy {gpu['training_util']:.1f}% of the time "
        f"({gpu['images_per_s']:,.0f} images a second) at {gpu['training_w']:.0f} W.",
        f"Generation kept it busy {gpu['inference_util']:.1f}% of the time "
        f"({gpu['tokens_per_s']:.0f} tokens a second; the library ran its slower reference "
        f"kernels here) at {gpu['inference_w']:.0f} W: the nameplate would overstate that energy "
        f"{_words[round(gpu['limit_w'] / gpu['inference_w'])]}fold.",
        ", ".join(_ledger_parts) + f"{_under}; " + _ledger_total,
        f"{ledger['script_minutes'] / 60:.1f} GPU-hours ({ledger['script_minutes']:.0f} minutes, "
        "each script's own record of its last run) per rebuild of every GPU lecture's figures. "
        + _ledger_energy,
        f"That is {min(_ledger_co2):.2f} to {max(_ledger_co2):.2f} kg of CO2, depending on the "
        f"factor, less than CS 441's 10-mile drive ({DRIVE_10_MILES_KG:.1f} kg).",
        f"The busy watts come from a loop that kept the GPU {gpu['training_util']:.1f}% busy",
        f"The same {mwh / 1e6:.2f} TWh a year in Illinois.",
        f"Illinois's average over all its plants ({_il_rate:.0f} lb/MWh).",
        "; ".join(_factor_parts) + ".",
        f"whose 2024 averages are {avg['PJM']:.0f} and {avg['MISO']:.0f}",
        "Illinois's own average is low because its plants are heavily nuclear "
        f"({il_nuclear_pct:.1f}% of their 2023 generation), and a campus does not draw from "
        "Illinois's plants alone.",
        f"{_p['slope']:,.0f} in PJM and {_q['slope']:,.0f} in MISO; the brackets show how far "
        "other defensible cleanings of the same data move them.",
        f"PJM: lighter, mostly between about {round(_pct['PJM'][0], -1):.0f} and "
        f"{round(_pct['PJM'][1], -1):.0f} lb/MWh. MISO: darker, highest in "
        f"{_months[_top2[0]]} and {_months[_top2[1]]}.",
        f"In MISO the monthly average runs from {by_month['MISO'].min():.0f} lb/MWh to "
        f"{by_month['MISO'].max():,.0f}, the hourly average across the day from "
        f"{by_hour['MISO'].min():.0f} to {by_hour['MISO'].max():.0f}; in PJM, months from "
        f"{by_month['PJM'].min():.0f} to {by_month['PJM'].max():.0f}, hours from "
        f"{by_hour['PJM'].min():.0f} to {by_hour['PJM'].max():.0f}.",
        f"Hours of the year range from {_pct['MISO'][0]:.0f} to {_pct['MISO'][1]:,.0f} in MISO "
        "(5th to 95th percentile).",
        f"{_p['changes']:,} hourly changes in PJM, {_q['changes']:,} in MISO. PJM: slope "
        f"{_p['slope']:,.0f} lb/MWh against an average of {avg['PJM']:,.0f}. MISO: slope "
        f"{_q['slope']:,.0f} against {avg['MISO']:,.0f}.",
        f"In MISO, where coal still follows load, the slope exceeds the average by "
        f"{_over['MISO']:.0f}%, and by {_range_pct['MISO'][0]:.0f}% to "
        f"{_range_pct['MISO'][1]:.0f}% under every cleaning we tried: the robust finding.",
        f"Its fuel columns miss its own net generation by more than {100 * FUEL_GAP_MAX:.0f}% in "
        f"{_p['fuel_gap_hours']} hours (MISO's never do); without those hours the slope is "
        f"{_p['slope']:,.0f}, {_over['PJM']:.0f}% over the average, but other defensible "
        f"cleanings give {spans['PJM'][0]:,.0f} to {spans['PJM'][1]:,.0f}, from "
        f"{-_range_pct['PJM'][0]:.0f}% under the average to {_range_pct['PJM'][1]:.0f}% over.",
        f"The intervals that resample whole weeks, {_p['ci'][0]:,.0f} to {_p['ci'][1]:,.0f} "
        f"(PJM) and {_q['ci'][0]:,.0f} to {_q['ci'][1]:,.0f} (MISO), cover sampling only",
        f"which in PJM includes how the data are cleaned ({spans['PJM'][0]:,.0f} to "
        f"{spans['PJM'][1]:,.0f} lb/MWh)",
        f"Lecture 9's rule removed {_words[len(_glitch_utc) // 2]} PJM glitch (demand falling "
        f"{-glitch[_glitch_utc[0]]['demand_gw']:.1f} GW for an hour on {_gday} {_gmonth}): "
        f"{_words[_p['dropped']]} hourly changes, which moved the slope from "
        f"{_p['raw_slope']:,.0f} to {_p['slope']:,.0f}.",
        f"{_co2[_IL]:.2f} at Illinois's average, {_co2['PJM average (2024)']:.2f} at PJM's, "
        f"{_co2['MISO average (2024)']:.2f} at MISO's, {_co2['PJM marginal (2024)']:.2f} at "
        f"PJM's marginal, {_co2['MISO marginal (2024)']:.2f} at MISO's marginal.",
        f"{_wp['PJM']:.1f} at the power plants (PJM's mix) plus {_site[str(min(WUE))]:g} to "
        f"{_site[str(max(WUE))]:.1f} on site as WUE goes from {min(WUE):g} to {max(WUE):g} L/kWh.",
        f"CO2 spans {min(_co2.values()):.2f} to {max(_co2.values()):.2f} million tonnes a year "
        "for the same campus",
        f"the power plants consume {_w2['PJM']:.0f} gallons per MWh of PJM's mix "
        f"({_w2['PJM'] * GAL_TO_L / 1000:.2f} L/kWh; {_w2['MISO']:.0f} and "
        f"{_w2['MISO'] * GAL_TO_L / 1000:.2f} for MISO), {_wp['PJM']:.1f} billion litres a year, "
        f"as much as the campus would evaporate itself at a WUE near {round(wue_break_even)}.",
        f"understates a new load by up to a factor of {_spread:.1f}.",
        "Today a choice of factor makes the same load look "
        f"{_words[round(_spread)]} times cleaner or dirtier.",
        f"Today that rule cleans {'an hour' if len(_glitch_utc) == 2 else '?'} of grid data.",
        ]  # fmt: skip

    # The words the slides attach to these numbers, each a fact of the notebook's values
    assert 0.95 * gpu["limit_w"] < gpu["training_w"] < gpu["limit_w"]  # "just under its limit"
    assert max(_ledger_co2) < DRIVE_10_MILES_KG  # "less than CS 441's 10-mile drive"
    assert aef["MISO"].mean() > aef["PJM"].mean()  # "MISO: darker"
    assert np.median(month_hour["MISO"]) > np.median(month_hour["PJM"])
    _h = [int(_t[11:13]) for _t in _glitch_utc]  # "for an hour": down, then back an hour later
    assert len(_glitch_utc) == 2 and _glitch_utc[0][:10] == _glitch_utc[1][:10]
    assert _h[1] - _h[0] == 1 and _q["dropped"] == 0  # one glitch, in PJM only
    _g = list(glitch.values())
    assert _g[0]["demand_gw"] < 0 < _g[1]["demand_gw"]
    assert _p["slope"] > avg["PJM"] and _q["slope"] > avg["MISO"]  # "the slope exceeds"
    claims = _render()
    _missing = [_c for _c in claims if _c not in DECK]
    assert not _missing, _missing
    mo.md(
        f"**Every number on the slides:** {len(claims)} statements, each rendered from this "
        "notebook's values and found word for word in the deck's text."
    )
    return (claims,)


@app.cell
def _(
    DECK,
    avg,
    ba_rows,
    ba_states,
    betas,
    break_ranges,
    campus,
    cuda_lectures,
    followers,
    glitch,
    il,
    il_nuclear_pct,
    importlib,
    l24_uses_cuda,
    ledger,
    ledger_low_kwh,
    mef,
    overlaps,
    ramp_share,
    spans,
    mo,
):
    # Statements in the deck that numbers.json does not carry, each measured here
    findings = []

    def finding(statement: str, measured: str, verdict: str) -> None:
        """Record one statement, which must be in the deck word for word."""
        assert statement in DECK, statement
        findings.append((statement, measured, verdict))

    # 1. Illinois's average: nuclear, and where a campus draws from
    _no_nuclear = il["STCO2RTA"] / (1 - il_nuclear_pct / 100)
    finding(
        "Illinois's own average is low because its plants are heavily nuclear",
        f"eGRID2023: nuclear is {il_nuclear_pct:.1f}% of Illinois's 2023 generation (PJM "
        f"{ba_rows['PJM']['BANCPR']:g}%, MISO {ba_rows['MISO']['BANCPR']:g}%); the same CO2 over "
        f"the rest of the state's generation would be {_no_nuclear:,.0f} lb/MWh, not "
        f"{il['STCO2RTA']:.0f}",
        "supported",
    )
    finding(
        "and a campus does not draw from Illinois's plants alone",
        f"PJM's generating plants lie in {ba_states['PJM']} states and MISO's in "
        f"{ba_states['MISO']} (eGRID2023); their 2023 rates are "
        f"{ba_rows['PJM']['BACO2RTA']:.0f} and {ba_rows['MISO']['BACO2RTA']:.0f} lb/MWh",
        "supported",
    )
    finding(
        "Illinois's plants run largely on nuclear",
        f"nuclear: {il_nuclear_pct:.1f}% of the state's 2023 generation (eGRID2023)",
        "supported",
    )
    # 2. Which plants follow demand
    _top = {_b: sorted(betas[_b], key=lambda _f: -betas[_b][_f])[:2] for _b in betas}
    _flat = all(abs(betas[_b][_f]) < 0.05 for _b in betas for _f in ("NUCLEAR", "WIND"))
    finding(
        "When demand rises, the plants that follow it are gas and coal, not nuclear or wind",
        "MWh of each fuel per MWh of demand change: "
        + "; ".join(f"{_b} gas {betas[_b]['GAS']:.2f}, coal {betas[_b]['COAL']:.2f}, nuclear "
                    f"{betas[_b]['NUCLEAR']:+.3f}, wind {betas[_b]['WIND']:+.3f}"
                    for _b in betas),
        "supported" if _flat and all(set(_t) == {"GAS", "COAL"} for _t in _top.values())
        else "not supported",
    )  # fmt: skip
    # 3. The mechanism, and what survives the cleaning
    finding(
        "In MISO, where coal still follows load, the slope exceeds the average",
        f"the generation that follows demand emits {followers['MISO']:,.0f} lb/MWh in MISO "
        f"(PJM {followers['PJM']:,.0f}), against averages of {avg['MISO']:,.0f} and "
        f"{avg['PJM']:,.0f}; the fuels' contributions add up to each slope exactly",
        "supported" if followers["MISO"] > avg["MISO"] else "not supported",
    )
    finding(
        "the robust finding. PJM's is not settled.",
        f"range over six cleanings (§5.7): MISO {spans['MISO'][0]:,.0f} to "
        f"{spans['MISO'][1]:,.0f}, all above its average {avg['MISO']:,.0f}; PJM "
        f"{spans['PJM'][0]:,.0f} to {spans['PJM'][1]:,.0f}, on both sides of {avg['PJM']:,.0f}",
        "supported" if spans["MISO"][0] > avg["MISO"] and spans["PJM"][0] < avg["PJM"]
        < spans["PJM"][1] else "not supported",
    )  # fmt: skip
    finding(
        "cover sampling only, not that choice",
        f"PJM's week-block interval is {mef['PJM']['ci'][1] - mef['PJM']['ci'][0]:.0f} lb/MWh "
        f"wide; the cleanings span {spans['PJM'][1] - spans['PJM'][0]:.0f}; "
        f"{100 * ramp_share:.0f}% of the 5% hours end on the evening or morning ramps",
        "supported" if spans["PJM"][1] - spans["PJM"][0]
        > 2 * (mef["PJM"]["ci"][1] - mef["PJM"]["ci"][0]) else "check",
    )  # fmt: skip
    finding(
        "where coal still follows load",
        f"coal per MWh of demand change: MISO {betas['MISO']['COAL']:.2f}, PJM "
        f"{betas['PJM']['COAL']:.2f}",
        "supported" if betas["MISO"]["COAL"] > 2 * betas["PJM"]["COAL"] else "partly",
    )
    # 4. Months and hours: the average, then the marginal factor
    _r = break_ranges

    def _width(span: tuple) -> float:
        return span[1] - span[0]

    _avg_months = all(_width(_v["avg_month"]) > _width(_v["avg_hour"]) for _v in _r.values())
    _marg_hours = all(_width(_v["marg_hour"]) > _width(_v["marg_month"]) and _v["separate"]
                      for _v in _r.values())  # fmt: skip
    finding(
        "For the average factor, the months differ more than the hours.",
        "range of the monthly against the hourly means: "
        + "; ".join(f"{_b} {_width(_v['avg_month']):.0f} against {_width(_v['avg_hour']):.0f} "
                    "lb/MWh" for _b, _v in _r.items()),
        "supported" if _avg_months else "not supported",
    )  # fmt: skip
    finding(
        "A moved load, though, changes the marginal emissions, which swing across the day as well",
        "marginal factor by local hour against by month: "
        + "; ".join(f"{_b} {_v['marg_hour'][0]:,.0f} to {_v['marg_hour'][1]:,.0f} against "
                    f"{_v['marg_month'][0]:,.0f} to {_v['marg_month'][1]:,.0f} lb/MWh"
                    for _b, _v in _r.items())
        + " (the extreme hours' week-block intervals do not overlap; §5.2)",
        "supported: across the day the marginal factor swings more than across the months"
        if _marg_hours else "check §5.2",
    )  # fmt: skip
    # 5. The generation kernels
    _fast = {_m: importlib.util.find_spec(_m) is not None
             for _m in ("fla", "causal_conv1d", "kernels")}  # fmt: skip
    finding(
        "the library ran its slower reference kernels here",
        "installed in this environment: "
        + ", ".join(f"`{_m}` {'yes' if _v else 'no'}" for _m, _v in _fast.items())
        + "; without them transformers runs its own PyTorch versions of Qwen3.5's "
        "gated-delta-rule attention and causal convolution",
        "supported (no fast kernel installed, so the fallbacks ran); how much slower they are, "
        "and their share of the low utilization, were not measured"
        if not any(_fast.values())
        else "check: a fast kernel is installed",
    )
    # 6. The glitch
    _g = list(glitch.values())
    finding(
        "Lecture 9's rule removed one PJM glitch",
        f"across the two dropped changes demand moved {_g[0]['demand_gw']:+.1f} and "
        f"{_g[1]['demand_gw']:+.1f} GW while net generation moved {_g[0]['netgen_gw']:+.1f} and "
        f"{_g[1]['netgen_gw']:+.1f} GW",
        "supported: the report changed, not the load",
    )
    # 7. The ledger's inputs
    finding(
        "each script's own record of its last run",
        "read from each GPU lecture's numbers.json when the ledger is built: "
        + ", ".join(f"{_n} {_v['minutes']:.1f}" for _n, _v in ledger["by_lecture"].items())
        + " min",
        "supported (read, not copied)",
    )  # fmt: skip
    if ledger["overlap_minutes"] > 1:
        finding(
            "the last runs shared the one GPU",
            "overlapping windows: "
            + "; ".join(f"{_a} and {_b} for {_m:.0f} min" for _a, _b, _m in overlaps),
            "supported" if overlaps else "not supported",
        )
    finding(
        "is counted at the busy power, so the ledger is an upper estimate",
        f"at the generation watts the summed rebuild would be {ledger_low_kwh:.2f} kWh, against "
        f"{ledger['kwh_sum']:.2f} at the busy watts",
        "supported" if ledger_low_kwh < ledger["kwh_sum"] else "not supported",
    )
    finding(
        "L24 runs on the CPU and is left out",
        f"L24's figure script {'uses' if l24_uses_cuda else 'never mentions'} CUDA; the scripts "
        f"that do are {', '.join(cuda_lectures)}, all in the ledger",
        "supported"
        if not l24_uses_cuda and set(cuda_lectures) == set(ledger["by_lecture"])
        else "not supported: see §5.6",
    )
    finding(
        "Development reran each script several times",
        "not counted here either",
        "stated as uncounted",
    )
    # 8. Water through the grid
    finding(
        "A dry-cooled campus (WUE near 0) still drinks through its grid.",
        f"at WUE 0 the campus still consumes {campus['water_plants_billion_l']['PJM']:.1f} "
        "billion litres a year at PJM's plants",
        "supported",
    )
    for _f in findings:
        print("finding:", " | ".join(_f))
    _rows = "\n    ".join(f"| {_s} | {_m} | {_v} |" for _s, _m, _v in findings)
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
    BREAK_DRAWS,
    T_START,
    US_DC,
    blank_hours,
    blank_local,
    claims,
    findings,
    fuel_gap,
    gpu,
    hours,
    iid_ci,
    il,
    mef,
    mo,
    n_values,
    netgen_response,
    own_response,
    own_watts,
    time,
    water_gal_per_mwh,
):
    _minutes = (time.perf_counter() - T_START) / 60
    _blank_gw = hours["PJM"].loc[blank_hours["PJM"][0], "demand"] / 1000  # interpolated
    _largest_gw = max(abs(mef["PJM"]["dd"]).max(), abs(mef["MISO"]["dd"]).max()) / 1000
    assert _blank_gw > 2 * _largest_gw
    _p = mef["PJM"]
    _own = "skipped" if own_watts is None else (
        f"{own_watts['seconds']:.0f} s on the {own_watts['device'].upper()}")  # fmt: skip
    print(f"done in {_minutes:.2f} min; {n_values} numbers.json values equal; {len(claims)} "
          f"claims found; {len(findings)} statements measured; own watts {_own}")  # fmt: skip
    mo.md(f"""
    ## Gotchas

    - **eGRID's CSVs have two header rows:** a description of each field, then its code. Read
      with `header=1` (row 2), and convert only the columns you use, failing on anything that is
      not a number; shares come as text ("54.9%").
    - **A blank fuel is not a blank hour.** A fuel a region never reports is blank all year (0
      MW). PJM's one blank hour, {blank_local:%Y-%m-%d %H:%M} local (hour 25 of the day daylight
      time ended), is blank in demand and generation alike: filled with zeros, it would be a
      drop and a recovery of about {_blank_gw:.0f} GW, more than twice the largest real change of
      the year in either region ({_largest_gw:.1f} GW, after the glitch is dropped).
    - **The July 2024 columns.** The second file's energy sources differ from the first's (and
      one carries a typo); map by keyword and stop on any column you have not mapped.
    - **Check that the parts add up to the whole.** PJM's fuel columns miss its net generation
      by more than 5% in {fuel_gap["PJM"]["hours_over_max"]:,} hours, mostly on the ramps (over
      the year by only {100 * abs(1 - fuel_gap["PJM"]["annual_ratio"]):.2f}%). The annual
      average hardly notices; the marginal factor, built from hour-to-hour changes, does, and the
      repair you choose moves it more than its sampling interval does (§5.7). Report the range.
    - **A ratio of sums is not a mean of ratios.** The annual average weights hours by their
      generation; the heatmap's cells are unweighted means of hourly factors.
    - **2024 had 8,784 hours;** the campus's year has 8,760.
    - **Units, named where they live:** short tons (eGRID) × 2,000 = lb; lb × 0.4536 = kg;
      gallons × 3.785 = litres; one hour at 1 MW = 1 MWh, so an hourly change in MW is a change
      in MWh per hour.
    - **The marginal factor counts the region's own plants.** For each MWh of new demand PJM's
      own fuel columns rise {own_response["PJM"]:.2f} MWh ({netgen_response["PJM"]:.2f} by net
      generation) and MISO's {own_response["MISO"]:.2f} ({netgen_response["MISO"]:.2f}); the
      rest is imported, and its CO2 is not in the factor.
    - **Resample weeks, not hours.** Single hours give intervals of {iid_ci["PJM"][0]:,.0f} to
      {iid_ci["PJM"][1]:,.0f} (PJM) against {mef["PJM"]["ci"][0]:,.0f} to
      {mef["PJM"]["ci"][1]:,.0f} from whole weeks (§5.4).
    - **Reproduce the script's draws by their order, not only their seed.** The slides'
      generator draws PJM's {BOOT} resamples, then MISO's; MISO's interval run alone from the
      same seed is different. This notebook's own intervals (§5.2 with {BREAK_DRAWS} draws a
      group, §5.4) use generators of their own.
    - **`nvidia-smi` reports the board,** not the wall plug: the processor, memory, fans, and
      power-supply losses are not in the course's {gpu["training_w"]:.0f} W, nor in §4.3.
    - **`torch.no_grad()`** is not needed in §4.3: it trains (backward passes are the work
      measured); nothing is evaluated.
    - **Time.** This run took {_minutes:.2f} minutes (own watts: {_own}).

    ## Going deeper

    Forsyth's §10.2 derives the least-squares slope and $R^2$; §10.3.1 shows how one point far
    from the rest swings a regression (today's glitch: {_p["raw_slope"]:,.0f} against
    {_p["slope"]:,.0f} lb/MWh); §11.2 treats
    robust regression and §11.2.2 a robust scale. The sources of today's data and method,
    paraphrased:

    - **Siler-Evans, Azevedo, and Morgan (2012), "Marginal emissions factors for the U.S.
      electricity system", *Environ. Sci. Technol.* 46(9).** From hourly emissions measured at
      fossil plants (EPA's continuous emissions monitoring), they regress each region's
      hour-to-hour change in emissions on its change in fossil generation, and report marginal
      factors for CO2, SO2, and NOx by NERC region and by time of day and season. Marginal and
      average factors can differ widely, so a policy judged by average factors (shifting load,
      adding wind, charging vehicles) can be judged wrongly. Two differences here: our hourly
      CO2 is generation by fuel times annual fuel rates, not stack measurements, and our
      regressor is demand, because a new load is a change in demand.
    - **U.S. EPA, *eGRID2023 Technical Guide*.** How each plant's generation, fuel, and
      emissions are assembled from EIA's and EPA's data, how the output emission rates are
      defined (annual emissions over annual net generation), the adjustments for combined heat
      and power and for biomass, and how plants are aggregated to balancing authorities,
      subregions, and states. A state's rate describes the plants inside its border, not the
      electricity used there: power crosses state lines, which is the first reason Illinois's
      {il["STCO2RTA"]:.0f} is the wrong factor for a load. eGRID also publishes non-baseload rates as a rough
      stand-in for the margin.
    - **U.S. EIA, notes on Form EIA-930 (the Hourly Electric Grid Monitor).** Each balancing
      authority reports, hour by hour, its demand, its net generation in total and by energy
      source, and its interchange with its neighbours; times are hour-ending, in UTC and local
      time; the data are preliminary and revised. Alongside the reported values EIA publishes
      imputed and adjusted series that fill missing or implausible hours; the adjusted columns
      are the ones used here, and even they keep a {-_p["glitches"][0]["gw"]:.1f} GW glitch.
    - **Macknick et al. (2012), *Environ. Res. Lett.* 7, 045802.** Operational water factors from
      the published literature, as medians and ranges by fuel and cooling technology.
      Withdrawal is water taken from a source; consumption is water that does not return
      (mostly evaporated). The cooling system often matters more than the fuel: once-through
      cooling withdraws 10 to 100 times as much water per MWh as a tower, while towers can
      consume about twice as much; photovoltaics and wind consume almost none; hydropower's
      reservoir evaporation varies too widely (and serves too many uses) to compare.
    - **Shehabi et al. (2024), *2024 United States Data Center Energy Usage Report*, LBNL.** A
      bottom-up estimate built from the installed servers, storage, and network equipment and
      the facilities' overhead (PUE): {US_DC["twh_2023"]} TWh in 2023, and a range of
      {US_DC["twh_2028"][0]} to {US_DC["twh_2028"][1]} TWh for 2028
      across scenarios for accelerated (GPU) servers, their use, and their cooling. It treats
      cooling water too, and its range is the honest way to state a projection.

    **Exercises.** (1) eGRID publishes a non-baseload CO2 rate for each balancing authority
    (`BANBCO2` in `BA23`). Read it for PJM and MISO and compare it with the marginal factors of
    §4.6: which question does each answer, and why might they differ? (2) The water of a new
    load is a marginal question too: compute each hour's power-plant water from the fuel mix
    and Macknick's factors, regress its changes on the changes in demand, and compare the
    marginal water factor with the average {water_gal_per_mwh["PJM"]:.0f} gallons per MWh (PJM). (3) Check EIA's balance:
    how closely does adjusted demand equal net generation minus total interchange, hour by hour,
    and where does it fail?

    ## Things to remember

    - AI is a load on grids and rivers.
    - Measure watts under the work.
    - One load, many factors: say which, and why.
    - New load: marginal, with an interval.
    - The engineer who seals it answers for it.

    **Thank you.**
    """)
    return


if __name__ == "__main__":
    app.run()
