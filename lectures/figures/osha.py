"""OSHA's Severe Injury Reports: the loader and the training years (Lectures 17 and 19).

Data: ``data/raw/osha_sir/January2015toNovember2025.zip`` from ``scripts/fetch_osha_sir.sh``
(federal OSHA, January 2015 to November 2025; U.S. government data, publicly available). Read
from the published ZIP, never extracted to disk. Training years 2015-2022; test 2023-2025.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from common import DATA_DIR, require

OSHA_ZIP = DATA_DIR / "raw" / "osha_sir" / "January2015toNovember2025.zip"
TRAIN_YEARS = (2015, 2022)  # tokenizers and word vectors learn from these years


def osha() -> pd.DataFrame:
    """The reports: event date, industry code, and the incident narrative, lowercased.

    Read from the published ZIP (latin-1 CSV), never extracted to disk.
    """
    with (
        zipfile.ZipFile(require(OSHA_ZIP, "scripts/fetch_osha_sir.sh")) as zf,
        zf.open("January2015toNovember2025.csv") as fh,
    ):
        df = pd.read_csv(fh, encoding="latin-1", low_memory=False, dtype={"Primary NAICS": str})
    out = pd.DataFrame(
        {
            "year": pd.to_datetime(df["EventDate"], format="%m/%d/%Y").dt.year,
            "naics": df["Primary NAICS"].fillna(""),
            "text": df["Final Narrative"]
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(),
            "event": df["Event"].astype("Int64").astype(str),  # OIICS: 43xx = fall to lower level
        }
    )
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    out["lower"] = out["text"].str.lower()
    return out


def training(df: pd.DataFrame) -> pd.Series:
    """Lowercased narratives of the training years."""
    return df.loc[df["year"].between(*TRAIN_YEARS), "lower"]
