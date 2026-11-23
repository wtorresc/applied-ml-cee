"""CAMELS daily forcing and streamflow: the loader, basins, and periods (Lectures 20 and 24).

Data: CAMELS v1.2 (Newman et al., 2015; Addor et al., 2017; doi:10.5065/D6MW2F4D, CC BY 4.0),
gathered by ``scripts/derive_camels.py`` into ``data/derived/camels_daily.npz`` (Daymet basin-mean
forcing and USGS flow in mm/day, 671 basins, 1980-2014) and ``camels_attributes.csv``.

The periods are water years (1 October to 30 September), in time order so that no test day
precedes a training day: training WY1990-2004, validation WY2005-2008, test WY2009-2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, require

DAILY: Path = DATA_DIR / "derived" / "camels_daily.npz"
ATTRIBUTES: Path = DATA_DIR / "derived" / "camels_attributes.csv"
HOME = "03338780"  # North Fork Vermilion River near Bismarck, IL (33 miles from campus)
SNOW = "11264500"  # Merced River at Happy Isles Bridge near Yosemite, CA (91% of P as snow)
PERIODS: dict[str, tuple[str, str]] = {
    "train": ("1989-10-01", "2004-09-30"),
    "val": ("2004-10-01", "2008-09-30"),
    "test": ("2008-10-01", "2014-09-30"),
}


@dataclass(frozen=True)
class Camels:
    """CAMELS in memory: one row per basin, in ``camels_name.txt``'s order.

    Attributes:
        gauge_id: 8-digit USGS ids, shape (basins,).
        dates: datetime64[D], shape (days,), 1980-01-01 to 2014-12-31.
        forcing: float32, shape (basins, days, 6), named in ``forcing_names``.
        forcing_names: day length, precipitation, radiation, max/min temperature, vapour pressure.
        flow: float32 mm/day, shape (basins, days), NaN where the record is missing.
        attrs: the joined attribute tables, one row per basin, in the same order.
    """

    gauge_id: np.ndarray
    dates: np.ndarray
    forcing: np.ndarray
    forcing_names: tuple[str, ...]
    flow: np.ndarray
    attrs: pd.DataFrame

    def basin(self, gauge: str) -> int:
        """Row of a gauge id.

        Raises:
            KeyError: If the gauge is not in CAMELS.
        """
        hits = np.flatnonzero(self.gauge_id == gauge)
        if len(hits) != 1:
            raise KeyError(f"gauge {gauge} is not in CAMELS")
        return int(hits[0])

    def period(self, name: str) -> np.ndarray:
        """Day indices of a period in ``PERIODS``."""
        start, end = (np.datetime64(d) for d in PERIODS[name])
        return np.flatnonzero((self.dates >= start) & (self.dates <= end))

    def channel(self, name: str) -> int:
        """Index of a forcing variable, e.g. ``"prcp_mm"``."""
        return self.forcing_names.index(name)


def load() -> Camels:
    """Read the derived CAMELS files.

    Raises:
        ValueError: If the attribute table's basins differ from the daily file's.
    """
    z = np.load(require(DAILY, "scripts/derive_camels.py"))
    attrs = pd.read_csv(require(ATTRIBUTES, "scripts/derive_camels.py"), dtype={"gauge_id": str})
    if not np.array_equal(attrs["gauge_id"].to_numpy(), z["gauge_id"]):
        raise ValueError("camels_attributes.csv and camels_daily.npz list different basins")
    return Camels(
        gauge_id=z["gauge_id"],
        dates=z["dates"],
        forcing=z["forcing"],
        forcing_names=tuple(z["forcing_names"].tolist()),
        flow=z["flow_mm"],
        attrs=attrs,
    )


def illinois(cam: Camels) -> list[str]:
    """Gauges in Illinois among the 531 basins of Kratzert et al. (2019), in CAMELS' order."""
    a = cam.attrs
    keep = a["gauge_name"].str.upper().str.rstrip().str.endswith(", IL") & (a["kratzert531"] == 1)
    return a.loc[keep, "gauge_id"].tolist()


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency over the days both are finite: 1 - SSE / SS about obs' mean.

    1 is a perfect fit; 0 is no better than the observed mean of the same days.

    Raises:
        ValueError: If the shapes differ or fewer than two days are finite.
    """
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.shape != sim.shape:
        raise ValueError(f"shapes differ: {obs.shape} vs {sim.shape}")
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 2:
        raise ValueError("fewer than two days to score")
    o, s = obs[ok], sim[ok]
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))
