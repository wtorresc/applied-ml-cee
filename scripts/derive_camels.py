"""Gather CAMELS' daily forcing, streamflow, and attributes into two files (CEE Lecture 20).

Reads ``data/raw/camels`` (from ``scripts/fetch_camels.sh``; never modified or extracted to disk)
and writes, in ``data/derived/``:

- ``camels_daily.npz``: one row per basin, in ``camels_name.txt``'s order (671 basins):
  ``gauge_id`` (8-digit USGS id), ``dates`` (datetime64[D], 1980-01-01 to 2014-12-31),
  ``forcing`` (float32, basins x days x 6: Daymet basin means of day length (s), precipitation
  (mm/day), shortwave radiation (W/m2), maximum and minimum temperature (deg C), and vapour
  pressure (Pa), named in ``forcing_names``), and ``flow_mm`` (float32, basins x days: USGS daily
  mean flow converted from ft3/s to mm/day over ``area_gages2``; NaN where USGS gives -999 or flags
  the day missing). Daymet's SWE column is left out: it is a model output, not an observation.
- ``camels_attributes.csv``: the seven attribute tables (clim, geol, hydro, name, soil, topo,
  vege) joined on ``gauge_id``, plus ``kratzert531`` (1 for the 531 basins Kratzert et al., 2019,
  used, from their published list).

Each output is written to a ``.part`` file and renamed over the old one (``os.replace``), so a
reader never sees half a file.

Data: CAMELS v1.2 (Newman et al., 2015; Addor et al., 2017; doi:10.5065/D6MW2F4D, CC BY 4.0).

Usage: PYTHONHASHSEED=0 pixi run -e aml-env python scripts/derive_camels.py  (about 2 minutes)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "camels"
DERIVED = ROOT / "data" / "derived"
ZIP = RAW / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
TABLES = ("name", "topo", "clim", "hydro", "soil", "geol", "vege")
FORCING_COLUMNS = {  # Daymet file header -> stored name
    "dayl(s)": "dayl_s",
    "prcp(mm/day)": "prcp_mm",
    "srad(W/m2)": "srad_wm2",
    "tmax(C)": "tmax_c",
    "tmin(C)": "tmin_c",
    "vp(Pa)": "vp_pa",
}
START, END = np.datetime64("1980-01-01"), np.datetime64("2014-12-31")
N_BASINS = 671
CFS_TO_M3S = 0.3048**3  # cubic feet to cubic metres
SECONDS_PER_DAY = 86_400


def attributes() -> pd.DataFrame:
    """Join CAMELS' attribute tables and flag the basins of Kratzert et al. (2019).

    Returns:
        One row per basin in ``camels_name.txt``'s order, ``gauge_id`` as an 8-character string.

    Raises:
        ValueError: If a table does not list the same 671 basins, or the 531 list names a basin
            that is not in CAMELS.
    """
    frames = []
    for table in TABLES:
        df = pd.read_csv(RAW / f"camels_{table}.txt", sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        if len(df) != N_BASINS or df["gauge_id"].duplicated().any():
            raise ValueError(f"camels_{table}.txt: {len(df)} rows, expected {N_BASINS} distinct")
        frames.append(df.set_index("gauge_id"))
    names = frames[0].index
    for table, df in zip(TABLES, frames, strict=True):
        if set(df.index) != set(names):
            raise ValueError(f"camels_{table}.txt lists different basins from camels_name.txt")
    joined = frames[0].join(frames[1:], how="left")
    listed = (RAW / "kratzert2019_basin_list.txt").read_text().split()
    if not set(listed) <= set(names) or len(set(listed)) != 531:
        raise ValueError("the 531-basin list does not match CAMELS")
    joined["kratzert531"] = joined.index.isin(listed).astype(int)
    return joined.reset_index()


def member_index(zf: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    """Find each gauge's Daymet basin-mean forcing file and streamflow file in the archive.

    Args:
        zf: The open CAMELS time-series archive.

    Returns:
        Gauge id -> member name, for the forcing files and for the flow files.

    Raises:
        ValueError: If a gauge has more than one file of a kind.
    """
    forcing, flow = {}, {}
    for name in zf.namelist():
        stem = name.rsplit("/", 1)[-1]
        if "/basin_mean_forcing/daymet/" in name and stem.endswith("_lump_cida_forcing_leap.txt"):
            target = forcing
        elif "/usgs_streamflow/" in name and stem.endswith("_streamflow_qc.txt"):
            target = flow
        else:
            continue
        gauge = stem[:8]
        if gauge in target:
            raise ValueError(f"{gauge}: two files ({target[gauge]}, {name})")
        target[gauge] = name
    return forcing, flow


def read_forcing(raw: bytes, dates: np.ndarray) -> np.ndarray:
    """Parse one Daymet forcing file onto the common calendar.

    Args:
        raw: The file's bytes (a 3-line header of latitude, elevation, area, then a table).
        dates: The common calendar, datetime64[D].

    Returns:
        Float32 array, days x 6, in ``FORCING_COLUMNS``' order.

    Raises:
        ValueError: If a day of the calendar is missing or repeated.
    """
    df = pd.read_csv(io.BytesIO(raw), sep=r"\s+", skiprows=3)
    day = pd.to_datetime({"year": df["Year"], "month": df["Mnth"], "day": df["Day"]}).to_numpy(
        "datetime64[D]"
    )
    keep = (day >= START) & (day <= END)
    if keep.sum() != len(dates) or not np.array_equal(day[keep], dates):
        raise ValueError("forcing does not cover every day of 1980-2014 exactly once")
    return df.loc[keep, list(FORCING_COLUMNS)].to_numpy(np.float32)


def read_flow(raw: bytes, dates: np.ndarray, area_km2: float) -> np.ndarray:
    """Parse one USGS streamflow file onto the common calendar, in mm/day.

    Args:
        raw: The file's bytes (gauge, year, month, day, flow in ft3/s, QC flag).
        dates: The common calendar, datetime64[D].
        area_km2: Basin area, km2.

    Returns:
        Float32 array of length ``len(dates)``; NaN where the record is missing.

    Raises:
        ValueError: If a day appears twice.
    """
    df = pd.read_csv(
        io.BytesIO(raw), sep=r"\s+", header=None, names=["gauge", "y", "m", "d", "cfs", "flag"]
    )
    day = pd.to_datetime({"year": df["y"], "month": df["m"], "day": df["d"]}).to_numpy(
        "datetime64[D]"
    )
    if len(np.unique(day)) != len(day):
        raise ValueError("a day appears twice in the flow record")
    cfs = df["cfs"].to_numpy(float)
    missing = (cfs < 0) | df["flag"].astype(str).str.startswith("M").to_numpy()
    mm = cfs * CFS_TO_M3S * SECONDS_PER_DAY / (area_km2 * 1e6) * 1e3
    mm[missing] = np.nan
    out = np.full(len(dates), np.nan, np.float32)
    inside = (day >= START) & (day <= END)
    out[(day[inside] - START).astype(int)] = mm[inside]
    return out


def main() -> None:
    """Write ``camels_daily.npz`` and ``camels_attributes.csv``."""
    attrs = attributes()
    dates = np.arange(START, END + 1)
    gauges = attrs["gauge_id"].to_numpy()
    forcing = np.empty((len(gauges), len(dates), len(FORCING_COLUMNS)), np.float32)
    flow = np.empty((len(gauges), len(dates)), np.float32)
    with zipfile.ZipFile(ZIP) as zf:
        forcing_files, flow_files = member_index(zf)
        absent = [g for g in gauges if g not in forcing_files or g not in flow_files]
        if absent:
            raise ValueError(f"no forcing or flow file for {absent}")
        for i, (gauge, area) in enumerate(zip(gauges, attrs["area_gages2"], strict=True)):
            forcing[i] = read_forcing(zf.read(forcing_files[gauge]), dates)
            flow[i] = read_flow(zf.read(flow_files[gauge]), dates, float(area))
    DERIVED.mkdir(parents=True, exist_ok=True)
    part = DERIVED / "camels_daily.npz.part"
    with part.open("wb") as fh:  # a handle: given a name, NumPy would append ".npz" to it
        np.savez_compressed(
            fh,
            gauge_id=gauges.astype("U8"),
            dates=dates,
            forcing=forcing,
            forcing_names=np.array(list(FORCING_COLUMNS.values())),
            flow_mm=flow,
        )
    part.replace(DERIVED / "camels_daily.npz")  # os.replace: atomic within one file system
    part = DERIVED / "camels_attributes.csv.part"
    attrs.to_csv(part, index=False)
    part.replace(DERIVED / "camels_attributes.csv")
    observed = np.isfinite(flow).mean()
    print(f"{len(gauges)} basins x {len(dates)} days; {observed:.1%} of flow days observed")


if __name__ == "__main__":
    main()
