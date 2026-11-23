"""The Building Data Genome 2 meters and the typical weekdays that Lectures 1 and 4 share.

Data: ``data/derived/bdg2_daily_electricity.npz`` and ``data/raw/bdg2/metadata.csv`` from
``scripts/fetch_bdg2.sh`` (Building Data Genome 2; Miller et al., 2020; licence terms in
that script).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, require

NPZ: Path = DATA_DIR / "derived" / "bdg2_daily_electricity.npz"
META: Path = DATA_DIR / "raw" / "bdg2" / "metadata.csv"
MIN_COMPLETE: float = 0.8  # share of a year's weekdays with all 24 hours metered


@dataclass(frozen=True)
class Portfolio:
    """Buildings with a usable typical weekday in both 2016 and 2017.

    Attributes:
        name: Building ids, shape ``(N,)``.
        site: Site code names.
        use: Primary use (Energy Star category; ``"Unknown"`` when missing).
        sqm: Gross floor area, m^2.
        P16: Typical weekday 2016, kW per hour of the day, shape ``(N, 24)``.
        P17: Typical weekday 2017, same layout.
        counts: How many buildings each eligibility rule removed.
    """

    name: np.ndarray
    site: np.ndarray
    use: np.ndarray
    sqm: np.ndarray
    P16: np.ndarray
    P17: np.ndarray
    counts: dict[str, int]


def load_bdg2(
    npz: Path = NPZ, meta_csv: Path = META
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """Load hourly electricity as ``(building, day, hour)`` and the building metadata.

    Args:
        npz: File written by ``scripts/fetch_bdg2.sh``.
        meta_csv: BDG2 ``metadata.csv``.

    Returns:
        kWh per hour (NaN where missing), building ids, dates, and metadata indexed like ``kwh``.

    Raises:
        FileNotFoundError: If the data are missing (run ``bash scripts/fetch_bdg2.sh``).
    """
    d = np.load(require(npz, "fetch_bdg2.sh"))
    kwh, names = d["kwh"], d["building_id"]
    dates = pd.DatetimeIndex(pd.to_datetime(d["date"]))
    assert kwh.shape == (len(names), len(dates), 24), kwh.shape
    meta = pd.read_csv(require(meta_csv, "fetch_bdg2.sh")).set_index("building_id").loc[names]
    return kwh, names, dates, meta


def typical_weekday(kwh: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean kW in each hour over the selected days that have all 24 hours metered.

    Args:
        kwh: Hourly kWh, shape ``(B, D, 24)``.
        keep: Boolean mask of days to average, shape ``(D,)``.

    Returns:
        Profiles ``(B, 24)`` (zeros where no day qualifies) and the number of days used ``(B,)``.
    """
    sub = kwh[:, keep, :].astype(np.float64)  # the file stores float32: sum in float64
    ok = ~np.isnan(sub).any(axis=2)  # complete days, (B, D_keep)
    n = ok.sum(axis=1)
    prof = np.where(ok[..., None], sub, 0.0).sum(axis=1) / np.maximum(n, 1)[:, None]
    assert prof.shape == (len(kwh), 24), prof.shape
    return prof, n


def build_portfolio(
    data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame] | None = None,
) -> Portfolio:
    """Apply the eligibility rules and return the 2016 and 2017 typical weekdays.

    Rules: (1) at least ``MIN_COMPLETE`` of each year's weekdays fully metered; (2) every hour of
    both typical weekdays positive; (3) hourly data: fewer than half of the complete days are
    perfectly flat. Rule 3 removes meters that report one value per day, spread evenly over the
    24 hours (63 meters in the current data, 61 of them at one site; counts in the result).

    Args:
        data: ``load_bdg2()``'s result, if already loaded (the file is about 110 MB); loaded
            here when None.

    Returns:
        The eligible buildings.
    """
    kwh, names, dates, meta = load_bdg2() if data is None else data
    weekday = np.asarray(dates.dayofweek < 5)
    year = np.asarray(dates.year)
    P16, n16 = typical_weekday(kwh, weekday & (year == 2016))
    P17, n17 = typical_weekday(kwh, weekday & (year == 2017))
    enough = (n16 >= MIN_COMPLETE * (weekday & (year == 2016)).sum()) & (
        n17 >= MIN_COMPLETE * (weekday & (year == 2017)).sum()
    )
    positive = (P16.min(axis=1) > 0) & (P17.min(axis=1) > 0)
    complete = ~np.isnan(kwh).any(axis=2)
    flat_day = np.ptp(np.where(complete[..., None], kwh, 0.0), axis=2) == 0  # one value all day
    flat_share = (flat_day & complete).sum(axis=1) / np.maximum(complete.sum(axis=1), 1)
    hourly = flat_share < 0.5
    keep = enough & positive & hourly
    flat_sites = meta["site_id"].to_numpy()[enough & positive & ~hourly]
    counts = {
        "buildings_in_file": int(len(names)),
        "sites_in_file": int(meta["site_id"].nunique()),
        "incomplete_or_nonpositive": int((~(enough & positive)).sum()),
        "daily_meters": int((enough & positive & ~hourly).sum()),
        "daily_meters_top_site_count": int(pd.Series(flat_sites).value_counts().iloc[0]),
        "eligible": int(keep.sum()),
    }
    m = meta[keep]
    return Portfolio(
        name=names[keep],
        site=m["site_id"].to_numpy(),
        use=m["primaryspaceusage"].fillna("Unknown").to_numpy(),
        sqm=m["sqm"].to_numpy(np.float64),
        P16=P16[keep],
        P17=P17[keep],
        counts=counts,
    )
