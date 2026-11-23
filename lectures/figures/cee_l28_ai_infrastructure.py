"""Figures and numbers for CEE Lecture 28: AI as infrastructure, and the profession.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l28_ai_infrastructure.py \
        --out build/fig/cee_l28_ai_infrastructure

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides).

The question: a 300 MW data centre (hypothetical) asks to connect in Illinois. What does it draw
from the grid and from water, and what does it emit, and which numbers should an engineer put in
the filing? And at the other end of the scale: what did this course's own computing use?

Data (``scripts/fetch_grid.sh``, ``scripts/measure_gpu_power.py``):

- EPA eGRID2023 (revision 2, June 2025): annual 2023 generation and CO2 by plant, balancing
  authority, and state. The plants give each region's CO2 per MWh by fuel.
- EIA-930 (the Hourly Electric Grid Monitor), 2024: hourly demand and net generation by fuel for
  PJM (northern Illinois, ComEd) and MISO (central and southern Illinois, Ameren). Hourly CO2 is
  generation by fuel times the region's eGRID rate for that fuel. The marginal emission factor is
  the slope of the hour-to-hour change in CO2 on the change in demand (Siler-Evans et al., 2012),
  over the changes whose two hours have fuel columns that add up to the hour's net generation
  within ``FUEL_GAP_MAX`` (a data-quality filter: PJM's fuel columns lag its ramps); the same
  slope under other defensible cleanings is reported beside it.
  Source: U.S. Energy Information Administration (Form EIA-930, 2024 data, files of 2025).
- Water consumed per MWh generated, by fuel, cooling-tower medians: Macknick et al. (2012),
  Tables 1 and 2. Hydropower's reservoir evaporation is left out (reservoirs serve many uses).
- The course's own GPU, measured: an RTX 3090 Ti's power every 0.5 s while training (ResNet-18)
  and while generating (Qwen3.5-2B), November 20, 2026. Each GPU lecture's figure-script time is
  read from its own record (``minutes`` in ``build/fig/<stem>/numbers.json``), and its clock
  window ends at that file's modification time. Runs
  that overlapped on the one GPU are capped at its power limit while they overlap. Only the
  GPU's energy is counted. Rebuild this lecture after any GPU lecture reruns.

Assumptions, labelled where they are used: the data centre draws 300 MW at its meter all year;
its power usage effectiveness (PUE) is 1.2; its own cooling water (WUE) is swept, 0 to 2 L per
kWh of IT energy, because it depends on the cooling design.
"""

from __future__ import annotations

import itertools
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

import aml_style
import title_art
from common import DATA_DIR, parse_out_dir, pct, require, rnd, seed_all, write_numbers

SEED = 28
EGRID = DATA_DIR / "derived" / "egrid2023"
EIA930 = [
    DATA_DIR / "raw" / "grid" / f"EIA930_BALANCE_2024_{h}.csv" for h in ("Jan_Jun", "Jul_Dec")
]
GPU_CSV = DATA_DIR / "derived" / "gpu_power" / "rtx3090ti_2026-11-20.csv"
GPU_META = GPU_CSV.with_suffix(".json")
REGIONS = {"PJM": "PJM (ComEd, Chicago)", "MISO": "MISO (Ameren, downstate)"}
LB_TO_KG = 0.45359237
GAL_TO_L = 3.785411784
HOURS = 8760
DC_MW = 300.0  # the hypothetical campus's draw at its meter, all year (assumption)
PUE = 1.2  # facility energy / IT energy (assumption)
WUE = (0.0, 0.5, 1.0, 2.0)  # direct cooling water, L per kWh of IT energy (swept: design-dependent)
# Water consumed at the power plant, gal per MWh: Macknick et al. (2012), cooling-tower medians.
# Coal: generic tower; gas: combined cycle, tower; nuclear: generic tower; PV and wind: Table 1;
# oil and other: natural-gas steam and biopower towers as stand-ins; hydro left out (see above).
WATER_GAL_PER_MWH = {
    "COAL": 687,
    "GAS": 205,
    "NUCLEAR": 672,
    "OIL": 826,
    "OTHER": 553,
    "SOLAR": 1,
    "WIND": 0,
    "HYDRO": 0,
    "STORAGE": 0,
    "GEO": 0,
    "UNKNOWN": 0,
}
FUELS = tuple(WATER_GAL_PER_MWH)
# US data centres (Shehabi et al., 2024, LBNL; figures as DOE's release of Dec. 20, 2024 states)
US_DC = {
    "twh_2023": 176,
    "share_2023_pct": 4.4,
    "twh_2028": (325, 580),
    "share_2028_pct": (6.7, 12),
}
# The GPU lectures' figure scripts, with the kind of GPU work each does. Their times are read from
# each one's numbers.json ("minutes", its last run) in FIG_DIR, so a rerun of a GPU lecture must
# be followed by a rerun of this one. L24 runs on the CPU and is left out. "busy" work keeps the
# GPU near its limit (training, feature extraction); "generation" is L25's agents, the GPU mostly
# waiting. L19 mixes fine-tuning with generation and is counted as busy: an upper estimate.
FIG_DIR = Path(__file__).resolve().parents[1] / "build" / "fig"
GPU_LECTURES = {
    "L16": ("cee_l16_segmentation", "busy"),
    "L17": ("cee_l17_transformers", "busy"),
    "L18": ("cee_l18_vision_foundation", "busy"),
    "L19": ("cee_l19_language_models", "busy"),
    "L20": ("cee_l20_time_series", "busy"),
    "L21": ("cee_l21_graph_networks", "busy"),
    "L22": ("cee_l22_point_clouds", "busy"),
    "L25": ("cee_l25_agents", "generation"),
}
DRIVE_10_MILES_KG = 4.0  # CS 441's comparison (EPA: about 400 g CO2 per mile for a typical car)
BOOT = 500  # block-bootstrap draws (whole weeks) for the marginal factors' intervals
MAD_K = 5  # hourly changes beyond this many robust standard deviations are dropped (Lecture 9)
MAD_SCALE = 1.4826  # MAD / Phi^-1(0.75) estimates a normal standard deviation
# Data quality: an hour whose fuel columns miss its net generation by more than this share is
# left out of the marginal regression (with the changes into and out of it). The sensitivity
# also tries the stricter share.
FUEL_GAP_MAX = 0.05
FUEL_GAP_STRICT = 0.02


# --------------------------------------------------------------------------- data


def egrid(sheet: str, numeric: tuple[str, ...]) -> pd.DataFrame:
    """One eGRID2023 sheet as ``fetch_grid.sh`` converted it; row 2 holds the field codes.

    Args:
        sheet: "PLNT23", "BA23", or "ST23".
        numeric: The columns the caller uses as numbers.

    Returns:
        The sheet, one row per plant, balancing authority, or state.

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_grid.sh``).
        ValueError: If a column in ``numeric`` did not parse as numbers (stop rather
            than coerce).
    """
    df = pd.read_csv(require(EGRID / f"{sheet}.csv", "fetch_grid.sh"), header=1, low_memory=False)
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{sheet}: columns not read as numbers: {bad}")
    return df


def fuel_rates(plants: pd.DataFrame, ba: str) -> dict[str, float]:
    """A balancing authority's 2023 CO2 per MWh by fuel, from its plants (lb/MWh).

    Plants with positive net generation only. A plant with generation but no CO2 on record counts
    as zero CO2 (eGRID leaves zero-emission plants blank): in PJM 0.08% of the gas generation and
    0.8% of "other fuel" (OTHF); in MISO 1.6% of other fossil (OFSL) and 18% of OTHF, whose
    "other" rate is therefore a little low (580 lb/MWh, against 619 without them).

    Args:
        plants: eGRID's plant sheet.
        ba: The balancing authority code.

    Returns:
        lb CO2 per MWh for coal, gas, oil, and "other" (biomass, other fossil, other fuels, used
        also for unknown fuels); zero-carbon fuels are absent.
    """
    p = plants[(plants["BACODE"] == ba) & (plants["PLNGENAN"] > 0)]
    g = p.groupby("PLFUELCT")[["PLNGENAN", "PLCO2AN"]].sum()
    rate = 2000 * g["PLCO2AN"] / g["PLNGENAN"]  # short tons to lb
    other = g.loc[["BIOMASS", "OFSL", "OTHF"]].sum()
    oth = 2000 * other["PLCO2AN"] / other["PLNGENAN"]
    return {
        "COAL": rate["COAL"],
        "GAS": rate["GAS"],
        "OIL": rate["OIL"],
        "OTHER": oth,
        "UNKNOWN": oth,
    }


def fuel_of(column: str) -> str:
    """Map an EIA-930 "Net Generation (MW) from ... (Adjusted)" column to one of ``FUELS``.

    The 2024 files changed their categories in July (hydro split from pumped storage, solar and
    wind split by batteries), so columns are mapped by keyword.

    Args:
        column: The column's name.

    Returns:
        One of ``FUELS``.

    Raises:
        ValueError: For a column no keyword matches.
    """
    s = column.lower()
    keys = (
        ("coal", "COAL"), ("natural gas", "GAS"), ("nuclear", "NUCLEAR"), ("petroleum", "OIL"),
        ("hydropower and pumped", "HYDRO"), ("excluding pumped", "HYDRO"),
        ("pumped", "STORAGE"), ("hydro", "HYDRO"), ("solar", "SOLAR"), ("wind", "WIND"),
        ("storage", "STORAGE"), ("geothermal", "GEO"), ("other fuel", "OTHER"),
        ("unknown fuel", "UNKNOWN"),
    )  # fmt: skip
    for key, fuel in keys:
        if key in s:
            return fuel
    raise ValueError(f"unmapped EIA-930 column: {column}")


def number(col: pd.Series) -> pd.Series:
    """EIA-930 numbers as floats; blanks stay NaN.

    Args:
        col: One column of an EIA-930 file.

    Returns:
        The column as floats.

    Raises:
        ValueError: On any text that is not a number.
    """
    return pd.to_numeric(col, errors="raise").astype(np.float64)


def hourly(regions: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """2024 hourly demand, net generation by fuel, and CO2 for each region.

    A fuel column that is blank for a region in every hour of a file is a fuel the region does
    not report (0 MW); any other blank must fall in an hour whose demand or net generation is
    blank too, and that hour is interpolated.

    Args:
        regions: Balancing authority codes, e.g. ``("PJM", "MISO")``.

    Returns:
        Per region, a frame indexed by UTC hour with ``demand``, ``netgen``, one column per fuel,
        ``co2_lb`` (from eGRID's fuel rates), ``local`` (local time at the end of the hour), and
        ``fuel_gap_share`` (|net generation - the fuel columns' sum| / net generation).

    Raises:
        ValueError: If a fuel is blank in an hour that is not blank.
    """
    plants = egrid("PLNT23", ("PLNGENAN", "PLCO2AN"))
    frames = []
    for path in EIA930:
        raw = pd.read_csv(require(path, "fetch_grid.sh"), low_memory=False)
        raw = raw[raw["Balancing Authority"].isin(regions)]
        when = "%m/%d/%Y %I:%M:%S %p"
        out = pd.DataFrame({
            "ba": raw["Balancing Authority"],
            "t": pd.to_datetime(raw["UTC Time at End of Hour"], format=when),
            "local": pd.to_datetime(raw["Local Time at End of Hour"], format=when),
            "demand": number(raw["Demand (MW) (Adjusted)"]),
            "netgen": number(raw["Net Generation (MW) (Adjusted)"]),
        })  # fmt: skip
        blank_hour = out["demand"].isna() | out["netgen"].isna()
        for f in FUELS:
            out[f] = 0.0
        for c in raw.columns:
            if c.startswith("Net Generation (MW) from ") and c.endswith("(Adjusted)"):
                v = number(raw[c])
                never = v.isna().groupby(out["ba"]).transform("all")  # not reported by the region
                stray = v.isna() & ~never & ~blank_hour
                if stray.any():
                    raise ValueError(f"{path.name}: {c} blank in {int(stray.sum())} hours")
                out[fuel_of(c)] += v.fillna(0.0)
        frames.append(out)
    d = pd.concat(frames, ignore_index=True)
    res = {}
    for ba in regions:
        x = d[d["ba"] == ba].set_index("t").sort_index()
        assert not x.index.duplicated().any(), f"{ba}: an hour appears twice"
        full = pd.date_range(x.index[0], periods=len(x), freq="h")  # files run by local date
        assert len(x) == 8784 and (x.index == full).all(), f"{ba}: hours missing in 2024"
        # A blank hour (PJM, 2024-11-04 05:00 UTC: local 00:00 EST on 4 November, the 25th hour
        # of 3 November, the day daylight time ended) is blank in demand and generation alike;
        # its fuels, filled with 0 above, are blanked too and every series is interpolated, so
        # the hour-to-hour changes carry no false spike.
        blank = x["netgen"].isna() | x["demand"].isna()
        assert blank.sum() <= 1, f"{ba}: {int(blank.sum())} blank hours"
        cols = ["demand", "netgen", *FUELS]
        x.loc[blank, cols] = np.nan
        x[cols] = x[cols].interpolate(limit=3)
        assert x[cols].notna().all().all(), f"{ba}: gaps left after interpolation"
        r = fuel_rates(plants, ba)
        x["co2_lb"] = sum(x[f] * r[f] for f in r)
        x["fuel_gap_share"] = (x["netgen"] - x[list(FUELS)].sum(axis=1)).abs() / x["netgen"]
        assert (x["netgen"] > 0).all(), f"{ba}: an hour without net generation"
        res[ba] = x
    return res


# --------------------------------------------------------------------------- analysis


def robust_keep(d: np.ndarray) -> np.ndarray:
    """Lecture 9's rule: keep the values within ``MAD_K`` robust standard deviations of the median.

    Args:
        d: Hour-to-hour changes.

    Returns:
        A boolean mask, True for the values kept.
    """
    dev = np.abs(d - np.median(d))
    return dev <= MAD_K * MAD_SCALE * np.median(dev)


def consistent(x: pd.DataFrame, share: float) -> np.ndarray:
    """The hour-to-hour changes whose two hours both pass the fuel-sum check.

    Args:
        x: One region's hourly frame, with ``fuel_gap_share``.
        share: The largest gap between the fuel columns' sum and net generation, as a share of
            net generation.

    Returns:
        A boolean mask over the ``len(x) - 1`` changes.
    """
    ok = x["fuel_gap_share"].to_numpy() <= share
    return ok[1:] & ok[:-1]


def marginal(x: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Marginal CO2 per MWh: the slope of hourly changes in CO2 on changes in demand.

    Siler-Evans et al. (2012) regress emission changes on generation changes from hour to hour;
    a new load is a change in demand, so demand is the regressor here. Two cleanings: Lecture 9's
    rule on the change in demand (a reporting glitch), and the fuel-sum check (``consistent``
    at ``FUEL_GAP_MAX``). The interval resamples whole weeks (hours within a week are not
    independent); it covers sampling only, not the choice of cleaning (see ``sensitivity``).

    Args:
        x: One region's hourly frame.
        rng: The random generator.

    Returns:
        The slope (lb/MWh), its 95% interval, the R^2, the slope without Lecture 9's rule, the
        glitches it dropped, the hours and changes the fuel-sum check dropped, and the kept
        changes for the figure.
    """
    dd, dc = x["demand"].diff().to_numpy()[1:], x["co2_lb"].diff().to_numpy()[1:]
    when = x.index[1:]
    fuel_ok = consistent(x, FUEL_GAP_MAX)
    raw_slope = float(np.polyfit(dd[fuel_ok], dc[fuel_ok], 1)[0])
    # Lecture 9's rule: a change more than five robust standard deviations from the median is a
    # reporting glitch (PJM: one hour on 2024-11-21 drops 38.6 GW and returns), not a load.
    mad_ok = robust_keep(dd)
    glitches = [
        {"utc": f"{t:%Y-%m-%d %H:%M}", "gw": rnd(v / 1000, 1)}
        for t, v in zip(when[~mad_ok], dd[~mad_ok], strict=True)
    ]
    keep = mad_ok & fuel_ok
    dd, dc = dd[keep], dc[keep]
    slope = float(np.polyfit(dd, dc, 1)[0])
    r2 = float(np.corrcoef(dd, dc)[0, 1] ** 2)
    weeks = np.flatnonzero(keep) // (24 * 7)
    idx = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    draws = []
    for _ in range(BOOT):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        draws.append(np.polyfit(dd[pick], dc[pick], 1)[0])
    return {
        "slope": slope,
        "ci": np.percentile(draws, [2.5, 97.5]).tolist(),
        "r2": r2,
        "raw_slope": raw_slope,
        "dropped": int((~mad_ok).sum()),
        "glitches": glitches,
        "fuel_gap_hours": int((x["fuel_gap_share"] > FUEL_GAP_MAX).sum()),
        "fuel_gap_changes": int((mad_ok & ~fuel_ok).sum()),
        "changes": len(dd),
        "_keep": keep,
        "_dd": dd,
        "_dc": dc,
    }  # fmt: skip


def sensitivity(x: pd.DataFrame) -> dict[str, float]:
    """The marginal factor (lb/MWh) under other defensible cleanings of the same data.

    Args:
        x: One region's hourly frame.

    Returns:
        The slope with no fuel-sum check; with the stricter share ``FUEL_GAP_STRICT``; with each
        hour's fuels rescaled to add up to its net generation (CO2 is linear in the fuels, so it
        scales by the same factor; no hour dropped); with the change in net generation as the
        regressor (Siler-Evans et al.'s choice); and with Lecture 9's rule applied to the change
        in CO2 as well. Each keeps Lecture 9's rule on the change in demand.
    """
    dd, dc = x["demand"].diff().to_numpy()[1:], x["co2_lb"].diff().to_numpy()[1:]
    dn = x["netgen"].diff().to_numpy()[1:]
    scale = x["netgen"] / x[list(FUELS)].sum(axis=1)
    dc_scaled = (x["co2_lb"] * scale).diff().to_numpy()[1:]
    mad_ok, fuel_ok = robust_keep(dd), consistent(x, FUEL_GAP_MAX)

    def slope(xs: np.ndarray, ys: np.ndarray, keep: np.ndarray) -> float:
        return float(np.polyfit(xs[keep], ys[keep], 1)[0])

    return {
        "no_fuel_check": slope(dd, dc, mad_ok),
        "strict_fuel_check": slope(dd, dc, mad_ok & consistent(x, FUEL_GAP_STRICT)),
        "fuels_rescaled": slope(dd, dc_scaled, mad_ok),
        "netgen_regressor": slope(dn, dc, mad_ok & fuel_ok),
        "co2_changes_trimmed": slope(dd, dc, mad_ok & fuel_ok & robust_keep(dc)),
    }


def water_per_mwh(x: pd.DataFrame) -> float:
    """Water consumed at the power plants per MWh of the region's 2024 mix (gal/MWh).

    Args:
        x: One region's hourly frame (storage charging, a negative generation, counts as none).

    Returns:
        The generation-weighted water factor, gallons per MWh of net generation.
    """
    total = x["netgen"].sum()
    return float(sum(x[f].clip(lower=0).sum() * WATER_GAL_PER_MWH[f] for f in FUELS) / total)


def gpu_power() -> dict:
    """The GPU's measured power by phase (mean watts while working), its idle floor, and limit.

    Returns:
        ``limit_w`` and ``gpu`` from the record's metadata; ``training_w`` and ``inference_w``,
        the mean power over each phase's samples; ``idle_w``, the 5th percentile of the idle
        phases' samples (they include the spin-down after training: mean 51.4 W, median 16.9 W);
        the mean utilization of each working phase; the throughputs; and the trace.

    Raises:
        FileNotFoundError: If the record is missing (``scripts/measure_gpu_power.py`` writes it).
    """
    d = pd.read_csv(require(GPU_CSV, "measure_gpu_power.py"))
    meta = json.loads(require(GPU_META, "measure_gpu_power.py").read_text())
    by = d.groupby("phase")["watts"].mean()
    util = d.groupby("phase")["util_pct"].mean()
    phases = {p["phase"]: p for p in meta["phases"] if p["phase"] != "idle"}
    return {
        "limit_w": meta["power_limit_w"],
        "gpu": meta["name"],
        "training_w": float(by["training"]),
        "inference_w": float(by["inference"]),
        "idle_w": float(d.loc[d["phase"] == "idle", "watts"].quantile(0.05)),
        "training_util": float(util["training"]),
        "inference_util": float(util["inference"]),
        "images_per_s": phases["training"]["images_per_s"],
        "tokens_per_s": phases["inference"]["tokens_per_s"],
        "_trace": d,
    }


def gpu_runs() -> dict[str, dict]:
    """Each GPU lecture's last figure-script run, from its own numbers.json in ``FIG_DIR``.

    Returns:
        Per lecture: ``minutes`` (its recorded ``minutes``), the ``kind`` of work, and the run's
        clock window in POSIX seconds: ``end``, the file's modification time (a script writes
        numbers.json last), and ``start``, ``minutes`` before it.

    Raises:
        FileNotFoundError: If a GPU lecture's figures are not built (or are being rebuilt).
        ValueError: If a lecture's numbers.json records no ``minutes``.
    """
    runs = {}
    for name, (stem, kind) in GPU_LECTURES.items():
        path = FIG_DIR / stem / "numbers.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found: build {name}'s figures (make build/fig/{stem}/.stamp in "
                "lectures/), then rerun this script"
            )
        recorded = json.loads(path.read_text(encoding="utf-8")).get("minutes")
        if recorded is None:
            raise ValueError(f"{path} records no minutes: rerun {name}'s figure script")
        minutes = float(recorded)
        end = path.stat().st_mtime
        runs[name] = {
            "minutes": minutes,
            "kind": kind,
            "end": end,
            "start": end - 60 * minutes,
        }
    return runs


def gpu_ledger(runs: dict[str, dict], gpu: dict) -> dict:
    """The GPU energy of one rebuild of every GPU lecture's figures.

    Each run is charged at the measured watts of its kind of work. Runs that overlapped in time
    shared the one GPU, so their sum counts those minutes twice; the board cannot draw more than
    its power limit, so the energy over the runs' clock time is also computed with the power
    capped at the limit wherever runs overlapped. Both are upper estimates: the busy watts come
    from a loop that kept the GPU 98.8% busy.

    Args:
        runs: ``gpu_runs``'s result.
        gpu: ``gpu_power``'s result.

    Returns:
        ``by_lecture`` (minutes, kind, kWh at its measured watts), ``kwh_sum`` (their sum),
        ``kwh`` (the overlap-capped energy, the ledger's figure), ``script_minutes``,
        ``span_minutes`` (the union of the windows), and ``overlap_minutes``.
    """
    watts = {"busy": gpu["training_w"], "generation": gpu["inference_w"]}
    by_lecture = {
        n: {**r, "kwh": r["minutes"] / 60 * watts[r["kind"]] / 1000} for n, r in runs.items()
    }
    edges = sorted({t for r in runs.values() for t in (r["start"], r["end"])})
    kwh, span_s = 0.0, 0.0
    for a, b in itertools.pairwise(edges):
        on = [r for r in runs.values() if r["start"] <= a and b <= r["end"]]
        if on:
            power = min(gpu["limit_w"], sum(watts[r["kind"]] for r in on))
            kwh += (b - a) / 3600 * power / 1000
            span_s += b - a
    script_minutes = sum(r["minutes"] for r in runs.values())
    return {
        "by_lecture": by_lecture,
        "kwh_sum": sum(v["kwh"] for v in by_lecture.values()),
        "kwh": kwh,
        "script_minutes": script_minutes,
        "span_minutes": span_s / 60,
        "overlap_minutes": script_minutes - span_s / 60,
    }


# --------------------------------------------------------------------------- figures

BLUE, ORANGE, GREEN, PINK = aml_style.PALETTE


def bare(ax: Axes) -> None:
    """Hide an axes' frame and ticks (for drawn pictures).

    Args:
        ax: The axes.
    """
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def picture_figures(out_dir: Path) -> None:
    """The one picture, three frames: the chain; what each link adds; who answers for it.

    Args:
        out_dir: The figure directory.
    """
    light, eq, person = "#e8f0f7", "#fbe3d4", "#e3f1e8"
    links = [
        ("a model\n(training, queries)", 0.12),
        ("GPU servers", 0.36),
        ("data centre", 0.60),
        ("the grid", 0.84),
        ("power plants,\nrivers, air", 1.08),
    ]
    adds = [
        "hours of work",
        "watts: measured,\nnot nameplate",
        "× PUE (cooling, losses)\n+ WUE: water on site",
        "× CO2 per MWh:\nwhich factor?",
        "fuel mix: CO2 and\ncooling water",
    ]
    arrow = {"arrowstyle": "-|>", "color": aml_style.INK, "lw": 1.2}
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=aml_style.FULL)
        ax.set(xlim=(0, 1.22), ylim=(0, 1))
        bare(ax)
        for i, (text, x) in enumerate(links):
            box = FancyBboxPatch((x - 0.1, 0.62), 0.2, 0.18, boxstyle="round,pad=0.01",
                                 facecolor=light, edgecolor=aml_style.INK, lw=1)  # fmt: skip
            ax.add_patch(box)
            ax.text(x, 0.71, text, ha="center", va="center", fontsize=10)
            if i < len(links) - 1:
                ax.annotate("", (x + 0.14, 0.71), (x + 0.1, 0.71), arrowprops=arrow)
            if k >= 2:
                ax.text(x, 0.45, adds[i], ha="center", va="center", fontsize=9, color=ORANGE,
                        bbox={"facecolor": eq, "edgecolor": "none", "pad": 3})  # fmt: skip
        if k == 3:
            box = FancyBboxPatch((0.03, 0.06), 1.14, 0.18, boxstyle="round,pad=0.01",
                                 facecolor=person, edgecolor=GREEN, lw=1)  # fmt: skip
            ax.add_patch(box)
            ax.text(0.6, 0.15, "who answers: the utility's engineers, the regulator, the "
                    "engineer of record who signs the design", ha="center", va="center",
                    fontsize=10, color=GREEN)  # fmt: skip
        aml_style.save(fig, out_dir, f"picture_{k}")


def power_figure(gpu: dict, out_dir: Path) -> None:
    """The GPU's measured power over the session, phases shaded, against its power limit.

    Args:
        gpu: ``gpu_power``'s result.
        out_dir: The figure directory.
    """
    d = gpu["_trace"]
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.plot(d["t_s"] / 60, d["watts"], color=BLUE, lw=1)
    ax.axhline(gpu["limit_w"], color=ORANGE, ls="--", lw=1)
    t_end = d["t_s"].max() / 60
    ax.text(t_end, gpu["limit_w"] + 8, f"power limit {gpu['limit_w']:.0f} W", color=ORANGE,
            fontsize=10, ha="right")  # fmt: skip
    for phase, color in (("training", GREEN), ("inference", PINK)):
        t = d.loc[d["phase"] == phase, "t_s"] / 60
        ax.axvspan(t.min(), t.max(), color=color, alpha=0.12, lw=0)
        mean = gpu[f"{phase}_w"]
        y = mean - 45 if phase == "training" else mean + 15
        ax.text((t.min() + t.max()) / 2, y, f"{phase}: {mean:.0f} W", ha="center",
                fontsize=10, color=aml_style.INK)  # fmt: skip
    ax.set(xlabel="minute", ylabel="GPU power, W", ylim=(0, 500))
    aml_style.save(fig, out_dir, "power")


def ledger_figure(ledger: dict, gpu: dict, out_dir: Path) -> None:
    """GPU energy of one full rebuild of the course's figures, by lecture.

    Args:
        ledger: ``gpu_ledger``'s result.
        gpu: ``gpu_power``'s result (the legend's watts).
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    names = list(ledger["by_lecture"])
    kwh = [ledger["by_lecture"][n]["kwh"] for n in names]
    style = {"busy": (GREEN, ""), "generation": (PINK, "//")}
    label = {"busy": f"GPU busy ({gpu['training_w']:.0f} W)",
             "generation": f"generation ({gpu['inference_w']:.0f} W)"}  # fmt: skip
    for i, n in enumerate(names):
        kind = ledger["by_lecture"][n]["kind"]
        color, hatch = style[kind]
        ax.bar(i, kwh[i], color=color, hatch=hatch, label=label.pop(kind, None))
        ax.text(i, kwh[i] + 0.02, f"{kwh[i]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)), names)
    ax.set(ylabel="GPU energy, kWh", ylim=(0, max(kwh) * 1.3))
    total = f"sum {ledger['kwh_sum']:.1f} kWh"
    if ledger["overlap_minutes"] > 1:  # runs shared the GPU: cap their overlap at the limit
        total += f"; runs overlapped, so at most {ledger['kwh']:.1f} kWh"
    ax.text(0.98, 0.92, total, transform=ax.transAxes, ha="right", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    aml_style.save(fig, out_dir, "ledger")


def factors_figure(factors: dict, spans: dict, out_dir: Path) -> None:
    """The emission factors an engineer could choose for the same new load.

    Args:
        factors: Name -> lb CO2 per MWh, in the figure's order (Illinois, two averages, two
            marginal factors).
        spans: Name -> (lowest, highest) under other cleanings, for the marginal factors; drawn
            as a bracket on the bar.
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    names = list(factors)[::-1]
    style = {"Illinois": (aml_style.MUTED, ""), "average": (BLUE, ""), "marginal": (ORANGE, "//")}
    for i, n in enumerate(names):
        color, hatch = next(v for k, v in style.items() if k in n)
        ax.barh(i, factors[n], color=color, hatch=hatch)
        right = factors[n]
        if n in spans:
            lo, hi = spans[n]
            ax.errorbar(factors[n], i, xerr=[[factors[n] - lo], [hi - factors[n]]], fmt="none",
                        color=aml_style.INK, capsize=4, lw=1.2)  # fmt: skip
            right = max(right, hi)
        ax.text(right + 15, i, f"{factors[n]:,.0f}", va="center", fontsize=10)
    ax.set_yticks(range(len(names)), names)
    top = max(*factors.values(), *(hi for _, hi in spans.values()))
    ax.set(xlabel="lb CO2 per MWh (bracket: the marginal factor under other cleanings)",
           xlim=(0, top * 1.15))  # fmt: skip
    aml_style.save(fig, out_dir, "factors")


def heatmap_figure(hours: dict, out_dir: Path) -> None:
    """Average CO2 per MWh by month and local hour (hour ending), 2024: PJM and MISO.

    Args:
        hours: ``hourly``'s frames.
        out_dir: The figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=True)
    vmin = min((x["co2_lb"] / x["netgen"]).min() for x in hours.values())
    vmax = max((x["co2_lb"] / x["netgen"]).max() for x in hours.values())
    for ax, (ba, x) in zip(axes, hours.items(), strict=True):
        aef = x["co2_lb"] / x["netgen"]
        grid = aef.groupby([x["local"].dt.month.values, x["local"].dt.hour.values]).mean().unstack()
        im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="YlOrBr", vmin=vmin, vmax=vmax,
                       extent=(-0.5, 23.5, 12.5, 0.5))  # fmt: skip
        ax.set(title=REGIONS[ba], xlabel="local hour (hour ending)", xticks=[0, 6, 12, 18, 23])
    axes[0].set(ylabel="month", yticks=[1, 4, 7, 10, 12])
    fig.colorbar(im, ax=axes, label="lb CO2 per MWh", shrink=0.9)
    aml_style.save(fig, out_dir, "heatmap")


def marginal_figure(mef: dict, avg: dict, out_dir: Path) -> None:
    """Hour-to-hour change in CO2 against change in demand: the slope is the marginal factor.

    Args:
        mef: ``marginal``'s result per region.
        avg: Each region's average factor, lb/MWh.
        out_dir: The figure directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=aml_style.FULL, sharey=False)
    for ax, (ba, m) in zip(axes, mef.items(), strict=True):
        dd, dc = m["_dd"] / 1000, m["_dc"] / 1e6  # GW and million lb
        ax.scatter(dd, dc, s=2, alpha=0.15, color=aml_style.MUTED, rasterized=True)
        xs = np.linspace(dd.min(), dd.max(), 2)
        ax.plot(xs, xs * m["slope"] / 1000, color=ORANGE, lw=2,
                label=f"marginal: {m['slope']:,.0f} lb/MWh")  # fmt: skip
        ax.plot(xs, xs * avg[ba] / 1000, color=BLUE, lw=1.5, ls="--",
                label=f"average: {avg[ba]:,.0f} lb/MWh")  # fmt: skip
        ax.set(title=REGIONS[ba], xlabel="change in demand, GW per hour")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].set_ylabel("change in CO2, million lb per hour")
    aml_style.save(fig, out_dir, "marginal")


def datacentre_figure(dc: dict, out_dir: Path) -> None:
    """The 300 MW campus: annual CO2 under each factor; water on site and at the plants.

    Args:
        dc: The campus's ``co2_mt`` by factor, ``water_plants_billion_l`` by region, and
            ``water_site_billion_l`` by WUE.
        out_dir: The figure directory.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=aml_style.FULL)
    names = list(dc["co2_mt"])
    a.barh(names[::-1], [dc["co2_mt"][n] for n in names[::-1]], color=BLUE)
    for i, n in enumerate(names[::-1]):
        a.text(dc["co2_mt"][n] + 0.02, i, f"{dc['co2_mt'][n]:.2f}", va="center", fontsize=9)
    a.set(xlabel="million tonnes CO2 a year", xlim=(0, max(dc["co2_mt"].values()) * 1.25))
    x = np.arange(len(WUE))
    on_site = np.array([dc["water_site_billion_l"][str(w)] for w in WUE])
    plants = dc["water_plants_billion_l"]["PJM"]
    b.bar(x, [plants] * len(x), 0.6, color=GREEN, label="at the power plants (PJM mix)")
    b.bar(x, on_site, 0.6, bottom=plants, color=BLUE, hatch="//", label="on site (cooling)")
    for k in range(len(x)):
        b.text(
            x[k], plants + on_site[k] + 0.1, f"{plants + on_site[k]:.1f}", ha="center", fontsize=9
        )
    b.set_xticks(x, [f"WUE {w:g}" for w in WUE], fontsize=9)
    b.set(ylabel="billion litres a year", ylim=(0, (plants + on_site.max()) * 1.25))
    b.legend(fontsize=8, frameon=False, loc="upper left")
    aml_style.save(fig, out_dir, "datacentre")


UNITS = [
    ("1 Learning from\nengineering data", ["L1 framing", "L2 nearest neighbors",
                                          "L3 generalization", "L4 clustering",
                                          "L5 least squares", "L6 classifiers",
                                          "L7 mixtures, Bayes", "L8 trees, SHAP"]),
    ("2 Uncertainty, risk,\nand decisions", ["L9 robust estimation", "L10 calibration",
                                             "L11 conformal, shift", "L12 Gaussian processes"]),
    ("3 Deep learning for\nsensing", ["L13 SGD", "L14 neural networks", "L15 CNNs",
                                      "L16 segmentation"]),
    ("4 Foundation models\nand time series", ["L17 transformers", "L18 vision models",
                                              "L19 language, RAG", "L20 time series"]),
    ("5 Structure and\nphysics", ["L21 graph networks", "L22 point clouds", "L23 PINNs",
                                  "L24 SINDy, operators"]),
    ("6 Agents, decisions,\nthe profession", ["L25 agents", "L26 MDPs, RL", "L27 fairness",
                                              "L28 AI as infrastructure"]),
]  # fmt: skip


def course_map_figure(out_dir: Path) -> None:
    """The 28 lectures in their six units.

    Args:
        out_dir: The figure directory.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.set(xlim=(0, 6), ylim=(0, 9.2))
    bare(ax)
    shades = ["#e8f0f7", "#fbe3d4", "#e3f1e8", "#f3e3ee", "#fff2cc", "#dde7f3"]
    for i, (unit, lectures) in enumerate(UNITS):
        ax.text(i + 0.5, 8.75, unit, ha="center", va="center", fontsize=7.5, weight="bold")
        for j, name in enumerate(lectures):
            y = 7.7 - j * 0.95
            box = FancyBboxPatch((i + 0.06, y - 0.38), 0.88, 0.76, boxstyle="round,pad=0.02",
                                 facecolor=shades[i], edgecolor=aml_style.INK, lw=0.6)  # fmt: skip
            ax.add_patch(box)
            label = textwrap.fill(name, 14)
            ax.text(i + 0.5, y, label, ha="center", va="center", fontsize=7.5)
    aml_style.save(fig, out_dir, "course_map")


def title_art_figure(out_dir: Path) -> None:
    """Title art: a row of server towers under power lines, a river below.

    Args:
        out_dir: The figure directory.
    """
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0c12", "#141824", "#191520", "#08080b"), grain=0.08)
    for i in range(5):
        x0 = 0.16 + 0.13 * i
        c.glaze(c.polygon([(x0, top * 0.30), (x0 + 0.09, top * 0.30), (x0 + 0.09, top * 0.62),
                           (x0, top * 0.62)]), "#8fb3c9", 0.5)  # fmt: skip
        for k in range(4):
            y = top * (0.36 + 0.06 * k)
            c.glow(c.lines([[(x0 + 0.015, y), (x0 + 0.075, y)]], 1.2, soft=1.0), "#eaa548", 0.35,
                   spread=4)  # fmt: skip
    wire = [
        (u, top * (0.80 - 0.05 * np.sin(np.pi * ((u * 4) % 1)))) for u in np.linspace(0, 1, 120)
    ]
    c.glow(c.lines([wire], 2.0, soft=1.0), "#eaa548", 0.4, spread=6)
    river = [(u, top * (0.14 + 0.02 * np.sin(6 * u))) for u in np.linspace(0, 1, 120)]
    c.glow(c.lines([river], 4.0, soft=2.0), "#5f8fb3", 0.45, spread=10)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: five pale server towers with gold lights, a gold "
        "power line sagging between poles above them, and a blue river along the bottom.",
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    """Measure, account, and compare; write the figures and numbers.json."""
    out_dir = parse_out_dir((__doc__ or "").split("\n")[0])
    seed_all(SEED)
    aml_style.apply()
    rng = np.random.default_rng(SEED)
    runs = gpu_runs()  # first: stop at once if a GPU lecture's figures are missing
    ba, st = egrid("BA23", ("BACO2RTA",)), egrid("ST23", ("STCO2RTA", "STNGENAN", "STGENANC"))
    il = st[st["PSTATABB"] == "IL"].iloc[0]
    rates_2023 = {b: float(ba.loc[ba["BACODE"] == b, "BACO2RTA"].iloc[0]) for b in REGIONS}
    hours = hourly(tuple(REGIONS))
    avg = {b: float(x["co2_lb"].sum() / x["netgen"].sum()) for b, x in hours.items()}
    mef = {b: marginal(x, rng) for b, x in hours.items()}
    sens = {b: sensitivity(x) for b, x in hours.items()}
    # the marginal factor's range over every cleaning tried, the chosen one included
    spans = {b: (min(m["slope"], *sens[b].values()), max(m["slope"], *sens[b].values()))
             for b, m in mef.items()}  # fmt: skip
    # The robust finding: MISO's marginal factor is well above its average under every cleaning
    assert spans["MISO"][0] > 1.2 * avg["MISO"], spans
    aef = {b: x["co2_lb"] / x["netgen"] for b, x in hours.items()}
    factors = {
        "Illinois, all plants (2023)": float(il["STCO2RTA"]),
        "PJM average (2024)": avg["PJM"],
        "MISO average (2024)": avg["MISO"],
        "PJM marginal (2024)": mef["PJM"]["slope"],
        "MISO marginal (2024)": mef["MISO"]["slope"],
    }
    # the 300 MW campus
    mwh = DC_MW * HOURS
    it_kwh = mwh * 1000 / PUE
    water_plants = {b: water_per_mwh(x) for b, x in hours.items()}
    dc = {
        "co2_mt": {k: v * mwh * LB_TO_KG / 1e9 for k, v in factors.items()},
        "water_plants_billion_l": {b: w * mwh * GAL_TO_L / 1e9 for b, w in water_plants.items()},
        "water_site_billion_l": {str(w): w * it_kwh / 1e9 for w in WUE},
    }
    # the course's own GPU energy
    gpu = gpu_power()
    ledger = gpu_ledger(runs, gpu)
    assert ledger["kwh"] <= ledger["kwh_sum"] + 1e-9 and ledger["overlap_minutes"] > -1e-6
    ledger_co2 = {k: ledger["kwh"] / 1000 * v * LB_TO_KG for k, v in factors.items()}

    picture_figures(out_dir)
    power_figure(gpu, out_dir)
    ledger_figure(ledger, gpu, out_dir)
    factors_figure(factors, {f"{b} marginal (2024)": span for b, span in spans.items()}, out_dir)
    heatmap_figure(hours, out_dir)
    marginal_figure(mef, avg, out_dir)
    datacentre_figure(dc, out_dir)
    course_map_figure(out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("power", "factors", "marginal", "datacentre"))

    mix = {
        b: {
            f: pct(x[f].clip(lower=0).sum() / x["netgen"].sum(), 1) for f in FUELS if f != "UNKNOWN"
        }
        for b, x in hours.items()
    }
    numbers = {
        "us_data_centres": US_DC,
        "egrid_2023": {
            "illinois_twh": rnd(il["STNGENAN"] / 1e6, 1),
            "illinois_lb_per_mwh": rnd(il["STCO2RTA"], 0),
            "illinois_nuclear_pct": pct(il["STGENANC"] / il["STNGENAN"], 1),
            "ba_lb_per_mwh": {b: rnd(v, 0) for b, v in rates_2023.items()},
        },
        "factors_lb_per_mwh": {k: rnd(v, 0) for k, v in factors.items()},
        "factor_spread": rnd(max(factors.values()) / min(factors.values()), 1),
        "marginal": {
            b: {
                "slope": rnd(m["slope"], 0),
                "ci": [rnd(c, 0) for c in m["ci"]],
                "r2": rnd(m["r2"], 2),
                "over_average_pct": pct(m["slope"] / avg[b] - 1, 0),
                "untrimmed": rnd(m["raw_slope"], 0),
                "changes_dropped": m["dropped"],
                "glitches": m["glitches"],
                "changes": m["changes"],
                "fuel_gap_hours": m["fuel_gap_hours"],
                "fuel_gap_changes": m["fuel_gap_changes"],
                "sensitivity": {k: rnd(v, 0) for k, v in sens[b].items()},
                "range": [rnd(spans[b][0], 0), rnd(spans[b][1], 0)],
                "over_average_pct_range": [pct(v / avg[b] - 1, 0) for v in spans[b]],
            }
            for b, m in mef.items()
        },
        "fuel_check": {
            "max_share_pct": pct(FUEL_GAP_MAX, 0),
            "strict_share_pct": pct(FUEL_GAP_STRICT, 0),
            "median_share_pct": {b: pct(x["fuel_gap_share"].median(), 1) for b, x in hours.items()},
            "pjm_average_clean_hours": rnd(
                hours["PJM"].loc[hours["PJM"]["fuel_gap_share"] <= FUEL_GAP_MAX, "co2_lb"].sum()
                / hours["PJM"].loc[hours["PJM"]["fuel_gap_share"] <= FUEL_GAP_MAX, "netgen"].sum(),
                0,
            ),
        },
        "hourly": {
            b: {
                "p5": rnd(np.percentile(a, 5), 0),
                "p95": rnd(np.percentile(a, 95), 0),
                "by_hour_range": [
                    rnd(a.groupby(hours[b]["local"].dt.hour.values).mean().min(), 0),
                    rnd(a.groupby(hours[b]["local"].dt.hour.values).mean().max(), 0),
                ],
                "by_month_range": [
                    rnd(a.groupby(hours[b]["local"].dt.month.values).mean().min(), 0),
                    rnd(a.groupby(hours[b]["local"].dt.month.values).mean().max(), 0),
                ],
            }
            for b, a in aef.items()
        },
        "mix_pct": mix,
        "water_plants_gal_per_mwh": {b: rnd(v, 0) for b, v in water_plants.items()},
        "water_plants_l_per_kwh": {b: rnd(v * GAL_TO_L / 1000, 2) for b, v in water_plants.items()},
        "datacentre": {
            "mw": DC_MW,
            "pue": PUE,
            "twh": rnd(mwh / 1e6, 2),
            "share_of_illinois_pct": pct(mwh / il["STNGENAN"], 1),
            "co2_mt": {k: rnd(v, 2) for k, v in dc["co2_mt"].items()},
            "water_plants_billion_l": {
                b: rnd(v, 1) for b, v in dc["water_plants_billion_l"].items()
            },
            "water_site_billion_l": {k: rnd(v, 1) for k, v in dc["water_site_billion_l"].items()},
        },
        "gpu": {
            k: (rnd(v, 1) if isinstance(v, float) else v)
            for k, v in gpu.items()
            if not k.startswith("_")
        },
        "ledger": {
            "script_minutes": rnd(ledger["script_minutes"], 0),
            "gpu_hours": rnd(ledger["script_minutes"] / 60, 1),
            "span_minutes": rnd(ledger["span_minutes"], 0),
            "overlap_minutes": rnd(ledger["overlap_minutes"], 0),
            "kwh_sum": rnd(ledger["kwh_sum"], 2),
            "kwh": rnd(ledger["kwh"], 2),
            "by_lecture": {
                n: {
                    "script_minutes": rnd(v["minutes"], 1),
                    "kind": v["kind"],
                    "kwh": rnd(v["kwh"], 2),
                    "window": [
                        f"{datetime.fromtimestamp(v['start'], tz=UTC):%Y-%m-%d %H:%M} UTC",
                        f"{datetime.fromtimestamp(v['end'], tz=UTC):%Y-%m-%d %H:%M} UTC",
                    ],
                }
                for n, v in ledger["by_lecture"].items()
            },
            "co2_kg": {k: rnd(v, 2) for k, v in ledger_co2.items()},
            "drive_10_miles_kg": DRIVE_10_MILES_KG,
        },
        "wue": list(WUE),
        "macknick_gal_per_mwh": WATER_GAL_PER_MWH,
        "settings": {"seed": SEED, "boot": BOOT, "hours": HOURS, "mad_k": MAD_K},
    }
    write_numbers(numbers, out_dir)


if __name__ == "__main__":
    main()
