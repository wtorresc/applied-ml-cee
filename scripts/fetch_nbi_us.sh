#!/usr/bin/env bash
# Download the FHWA National Bridge Inventory (NBI) for all states, 2018-2024 every two years,
# and derive one table of deck-rating transitions between consecutive files for CEE Lecture 13.
#
#   data/raw/nbi_us/<year>hwybronefiledel.zip      the zipped files as published (read-only)
#   data/derived/nbi_us_deck_transitions.csv.gz    one row per bridge and pair of consecutive
#                                                  files (2018-2020, 2020-2022, 2022-2024)
#
# Source: FHWA, National Bridge Inventory, ASCII files, "Highway Bridges for all States (in a
# single file)", delimited:
#   https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
#   https://www.fhwa.dot.gov/bridge/nbi/<year>hwybronefiledel.zip
# Each zip holds one comma-delimited text file with a single-quote text qualifier, sorted by
# state code (Item 1, FIPS) and structure number. Item numbers and codes: FHWA (1995), Recording
# and Coding Guide for the Structure Inventory and Appraisal of the Nation's Bridges,
# FHWA-PD-96-001. 2016 is published under another file name and is not used.
#
# Terms: as in fetch_nbi.sh. A work of the U.S. federal government, not subject to copyright in
# the U.S. (17 U.S.C. 105); data.gov lists the NBI under a CC0 1.0 public-domain dedication.
# FHWA's download page attaches the notice of 23 U.S.C. 409: data compiled for highway safety
# improvement may not be used as evidence in actions for damages. No account or key is needed.
#
# Size: about 53 MB per zip, 214 MB for the four; the derived table is about 25 MB.
#
# Usage: bash scripts/fetch_nbi_us.sh   (from anywhere; needs curl and the pixi env aml-env)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/nbi_us"
derived="$root/data/derived"
mkdir -p "$raw" "$derived"

years=(2018 2020 2022 2024)
for year in "${years[@]}"; do
  file="$raw/${year}hwybronefiledel.zip"
  if [[ ! -f "$file" ]]; then
    curl -sSfL -o "$file.part" "https://www.fhwa.dot.gov/bridge/nbi/${year}hwybronefiledel.zip"
    mv "$file.part" "$file"
    chmod a-w "$file"
  fi
done

# Derived table. A transition pairs a bridge's record in one file with its record in the next,
# matched on state code and structure number. Kept: highway bridges with a deck rating (Item
# 58, 0-9) in both files, culverts (Item 43B = 19) left out; a new inspection in between (Item
# 90 moved forward by 6 to 60 months); no reconstruction in between (Item 106 did not move to a
# later year, Item 27 unchanged); a year built and an ADT; no year reconstructed after the
# first file's year (64 records); the 50 states, DC, and Puerto Rico
# (Guam and the Virgin Islands have under 50 transitions). Item 109, percent trucks, is blank
# for many bridges in some states (70% in Minnesota) and is kept blank here. A year
# reconstructed of 0, blank, or before 1800 means never. Rows keep the published order: by
# file, then state, then structure number.
pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$raw" \
  "$derived/nbi_us_deck_transitions.csv.gz" "${years[@]}" <<'PY'
import gzip
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

raw, out, years = Path(sys.argv[1]), Path(sys.argv[2]), [int(y) for y in sys.argv[3:]]
items = {
    "STATE_CODE_001": "state", "STRUCTURE_NUMBER_008": "structure",
    "FACILITY_CARRIED_007": "facility", "FEATURES_DESC_006A": "crosses", "OWNER_022": "owner",
    "YEAR_BUILT_027": "year_built", "ADT_029": "adt", "STRUCTURE_KIND_043A": "kind",
    "STRUCTURE_TYPE_043B": "design", "STRUCTURE_LEN_MT_049": "length_m",
    "DECK_WIDTH_MT_052": "deck_width_m", "DECK_COND_058": "deck",
    "DATE_OF_INSPECT_090": "inspected", "YEAR_RECONSTRUCTED_106": "year_reconstructed",
    "DECK_STRUCTURE_TYPE_107": "deck_type", "SURFACE_TYPE_108A": "surface",
    "MEMBRANE_TYPE_108B": "membrane", "DECK_PROTECTION_108C": "protection",
    "PERCENT_ADT_TRUCK_109": "pct_truck",
}
# Non-numeric codes that become missing; any other code stops the run instead of being coerced
# silently. Only one record has any: Georgia's structure 000000012106380 in 2018, deck width "H"
# and inspection date "N00". Its transition fails the months test below. Blank cells are missing
# already when read.
ODD = {"deck_width_m": {"H"}, "inspected": {"N00"}}


def number(s: pd.Series, item: str, year: int) -> pd.Series:
    """An Item as numbers: the codes in ``ODD`` become missing, any other code stops the run."""
    present = s.dropna()
    odd = set(present[pd.to_numeric(present, errors="coerce").isna()])
    assert odd <= ODD.get(item, set()), f"{year}: unexpected {item} codes {sorted(odd)}"
    return pd.to_numeric(s.where(~s.isin(odd)))


def read(year: int) -> pd.DataFrame:
    """One file's highway bridges with a deck rating, Items decoded as the header says."""
    with zipfile.ZipFile(raw / f"{year}hwybronefiledel.zip") as z:
        (name,) = z.namelist()
        f = pd.read_csv(z.open(name), quotechar="'", dtype=str, encoding="latin-1",
                        usecols=list(items))
    f = f.rename(columns=items).apply(lambda c: c.str.strip())
    f = f[(f["design"] != "19") & f["deck"].str.fullmatch(r"\d", na=False)].copy()
    f["deck"] = f["deck"].astype(int)
    f["year_built"] = number(f["year_built"], "year_built", year)
    rec = number(f["year_reconstructed"], "year_reconstructed", year).fillna(0)
    f["year_reconstructed"] = rec.where(rec >= 1800, 0).astype(int)
    # Item 90 is MMYY; a blank date stays missing, so its transitions fail the months test.
    mmyy = number(f.pop("inspected"), "inspected", year)
    yy, mm = mmyy % 100, mmyy // 100
    f["inspected_month"] = (2000 + yy) * 12 + mm - 1
    for c in ("adt", "pct_truck", "length_m", "deck_width_m"):
        f[c] = number(f[c], c, year)
    for c in ("deck_type", "surface", "membrane", "protection", "kind"):
        f[c] = f[c].fillna("N")
    dup = f.duplicated(["state", "structure"], keep=False)
    assert dup.sum() <= 10, f"{year}: {dup.sum()} duplicated structure numbers"
    f = f[~dup]
    f.insert(0, "nbi_year", year)
    return f.reset_index(drop=True)


files = {y: read(y) for y in years}
pairs = []
for y0, y1 in zip(years[:-1], years[1:]):
    nxt = files[y1][["state", "structure", "deck", "inspected_month", "year_reconstructed",
                     "year_built"]]
    m = files[y0].merge(nxt, on=["state", "structure"], suffixes=("", "_next"))
    m["months"] = m.pop("inspected_month_next") - m.pop("inspected_month")
    rebuilt = (m.pop("year_reconstructed_next") > m["year_reconstructed"]) | (
        m.pop("year_built_next") != m["year_built"])
    age = y0 - m["year_built"]
    keep = (m["months"].between(6, 60) & ~rebuilt & age.between(0, 250)
            & (m["year_reconstructed"] <= y0) & m["adt"].notna() & (m["length_m"] > 0)
            & ~m["state"].isin(["66", "78"]))
    print(f"{y0}-{y1}: {len(m)} matched, {keep.sum()} kept")
    pairs.append(m[keep])
t = pd.concat(pairs, ignore_index=True).drop(columns="design")
t["drop"] = (t["deck_next"] < t["deck"]).astype(int)
assert t["state"].nunique() == 52, t["state"].nunique()
# The counts CEE Lecture 13 quotes (files as published on 2026-11-16).
assert len(t) == 1_299_045, len(t)
assert (t["nbi_year"] < 2022).sum() == 864_672
assert not t.duplicated(["nbi_year", "state", "structure"]).any()
# Written beside, then renamed, so an interrupted run leaves no truncated table. The gzip header
# keeps the file name and a zero time stamp, so the same table gives the same bytes.
part = out.with_name(out.name + ".part")
with part.open("wb") as fh, gzip.GzipFile(out.stem, "wb", fileobj=fh, mtime=0) as gz:
    t.to_csv(gz, index=False)
part.replace(out)
print(f"wrote {out}: {len(t)} transitions, {t.groupby(['state', 'structure']).ngroups} bridges")
print(t.groupby("nbi_year")["drop"].agg(["size", "mean"]).round(4).to_string())
PY
