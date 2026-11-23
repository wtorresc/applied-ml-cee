"""The Illinois National Bridge Inventory records and inputs that Lectures 1, 3, and 6 share.

Data: ``data/derived/nbi_il_bridges.csv`` from ``scripts/fetch_nbi.sh`` (FHWA National Bridge
Inventory, Illinois, 2014-2024 every two years; U.S. Government work).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import DATA_DIR, require

NBI_CSV: Path = DATA_DIR / "derived" / "nbi_il_bridges.csv"
POOR: int = 4  # FHWA: a lowest rating of 4 or less is poor
TWIN_KM: float = 0.2  # parallel structures closer than this are twins
FEATURES: list[str] = [
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
MATERIAL: dict[str, str] = {
    "1": "concrete",
    "2": "concrete",
    "3": "steel",
    "4": "steel",
    "5": "prestressed",
    "6": "prestressed",
    "7": "timber",
}
# Illinois counties in alphabetical order; county FIPS codes are 001, 003, ..., 203 in this
# order, and an Illinois structure number starts with the same alphabetical number (Cook: 016).
COUNTY_NAMES: list[str] = [
    "Adams",
    "Alexander",
    "Bond",
    "Boone",
    "Brown",
    "Bureau",
    "Calhoun",
    "Carroll",
    "Cass",
    "Champaign",
    "Christian",
    "Clark",
    "Clay",
    "Clinton",
    "Coles",
    "Cook",
    "Crawford",
    "Cumberland",
    "DeKalb",
    "De Witt",
    "Douglas",
    "DuPage",
    "Edgar",
    "Edwards",
    "Effingham",
    "Fayette",
    "Ford",
    "Franklin",
    "Fulton",
    "Gallatin",
    "Greene",
    "Grundy",
    "Hamilton",
    "Hancock",
    "Hardin",
    "Henderson",
    "Henry",
    "Iroquois",
    "Jackson",
    "Jasper",
    "Jefferson",
    "Jersey",
    "Jo Daviess",
    "Johnson",
    "Kane",
    "Kankakee",
    "Kendall",
    "Knox",
    "Lake",
    "LaSalle",
    "Lawrence",
    "Lee",
    "Livingston",
    "Logan",
    "McDonough",
    "McHenry",
    "McLean",
    "Macon",
    "Macoupin",
    "Madison",
    "Marion",
    "Marshall",
    "Mason",
    "Massac",
    "Menard",
    "Mercer",
    "Monroe",
    "Montgomery",
    "Morgan",
    "Moultrie",
    "Ogle",
    "Peoria",
    "Perry",
    "Piatt",
    "Pike",
    "Pope",
    "Pulaski",
    "Putnam",
    "Randolph",
    "Richland",
    "Rock Island",
    "St. Clair",
    "Saline",
    "Sangamon",
    "Schuyler",
    "Scott",
    "Shelby",
    "Stark",
    "Stephenson",
    "Tazewell",
    "Union",
    "Vermilion",
    "Wabash",
    "Warren",
    "Washington",
    "Wayne",
    "White",
    "Whiteside",
    "Will",
    "Williamson",
    "Winnebago",
    "Woodford",
]
COUNTY: dict[str, str] = {f"{2 * i + 1:03d}": n for i, n in enumerate(COUNTY_NAMES)}


def load_bridges(path: Path = NBI_CSV) -> pd.DataFrame:
    """Load the Illinois bridge-year records and add the lectures' inputs.

    Args:
        path: The CSV written by ``scripts/fetch_nbi.sh``.

    Returns:
        One row per bridge and NBI year (2014-2024, every two years) with complete inputs:
        the derived columns, ``FEATURES``, ``material``, and a planar grid in km (``x_km``,
        ``y_km``) for distances between bridges.

    Raises:
        FileNotFoundError: If the CSV is missing (run ``bash scripts/fetch_nbi.sh``).
    """
    text = ["structure", "county", "district", "owner", "kind", "design", "parallel"]
    df = pd.read_csv(require(path, "fetch_nbi.sh"), dtype=dict.fromkeys(text, str))
    assert len(COUNTY) == 102 and set(df["county"]) <= set(COUNTY), "unknown county code"
    df["age"] = (df["inspection_year"] - df["year_built"]).clip(lower=0)
    last_work = np.fmax(df["year_built"], df["year_reconstructed"].fillna(0))
    df["years_since_work"] = (df["inspection_year"] - last_work).clip(lower=0)
    df["log_adt"] = np.log1p(df["adt"])
    df["log_length"] = np.log(df["length_m"].clip(lower=1.0))
    df["material"] = df["kind"].map(MATERIAL).fillna("other")
    for m in ("concrete", "steel", "prestressed", "timber"):
        df[m] = (df["material"] == m).astype(float)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)  # complete inputs
    # Equirectangular grid about 40° N, 89° W. East-west distances are off by up to 4% at
    # Illinois' northern and southern ends (cos 42.5° vs cos 37°), enough for 0.2 km twins.
    df["x_km"] = (df["lon"] + 89.0) * 111.32 * np.cos(np.deg2rad(40.0))
    df["y_km"] = (df["lat"] - 40.0) * 110.57
    return df


def find_twins(records: pd.DataFrame) -> pd.DataFrame:
    """Twin bridges in one inventory: parallel structures built the same year, side by side.

    A record is a twin when Item 101 marks it as the left or right of parallel structures and
    the nearest other parallel structure is closer than ``TWIN_KM`` and was built the same year.

    Args:
        records: The records of one NBI year.

    Returns:
        Columns ``row`` and ``twin_row`` (index labels of ``records``) and ``km``.
    """
    par = records[records["parallel"].isin(["R", "L"])]
    xy = par[["x_km", "y_km"]].to_numpy()
    dist, idx = NearestNeighbors(n_neighbors=3).fit(xy).kneighbors(xy)
    rows = par.index.to_numpy()
    own = np.arange(len(par))
    first = np.where(idx[:, 0] == own, 1, 0)  # coincident points: skip the record itself
    pick = idx[own, first]
    out = pd.DataFrame({"row": rows, "twin_row": rows[pick], "km": dist[own, first]})
    built = records["year_built"]
    same = built.loc[out["row"]].to_numpy() == built.loc[out["twin_row"]].to_numpy()
    return out[(out["km"] < TWIN_KM) & same].reset_index(drop=True)
