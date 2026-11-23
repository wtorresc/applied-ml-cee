#!/usr/bin/env bash
# Download the FHWA National Bridge Inventory (NBI) for Illinois, several years, and derive one
# tidy table of bridge-year records for CEE Lecture 3.
#
#   data/raw/nbi/IL<yy>.txt            the delimited files as published (read-only; never edit)
#   data/derived/nbi_il_bridges.csv    highway bridges (culverts left out), one row per bridge and
#                                      NBI year, selected items decoded (see the Python step below)
#
# Source: FHWA, National Bridge Inventory, ASCII files, "Delimited", one file per state and year:
#   https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
#   https://www.fhwa.dot.gov/bridge/nbi/<year>/delimited/IL<yy>.txt   (highway bridges only)
# Item numbers and codes: FHWA (1995), Recording and Coding Guide for the Structure Inventory and
# Appraisal of the Nation's Bridges, FHWA-PD-96-001. 2025 is the last year in this format; later
# years use the SNBI, so the years here stop at 2024.
#
# Terms: a work of the U.S. federal government, not subject to copyright in the U.S.
# (17 U.S.C. 105); data.gov lists the NBI under a CC0 1.0 public-domain dedication. FHWA's
# download page attaches the notice of 23 U.S.C. 409: data compiled for highway safety
# improvement may not be used as evidence in actions for damages. FHWA also notes that NBI data
# must not be used for route clearance. No account, key, or contact details are needed.
#
# Size: about 11 MB per year; 6 years (2014-2024, every two years), about 66 MB raw, and a
# derived CSV of about 19 MB.
#
# Usage: bash scripts/fetch_nbi.sh   (from anywhere; needs curl and the pixi env aml-env)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/nbi"
derived="$root/data/derived"
mkdir -p "$raw" "$derived"

# Biennial, matching the 24-month routine inspection cycle. 2012 is left out: its structure
# numbers use another layout (the 7-digit Illinois number sits at characters 4-10, not 1-7), so
# its records cannot be matched reliably to the later years.
years=(2014 2016 2018 2020 2022 2024)
for year in "${years[@]}"; do
  yy="${year:2:2}"
  file="$raw/IL$yy.txt"
  if [[ ! -f "$file" ]]; then
    curl -sSfL -o "$file.part" "https://www.fhwa.dot.gov/bridge/nbi/$year/delimited/IL$yy.txt"
    mv "$file.part" "$file"
    chmod a-w "$file"
  fi
done

# Derived table. Comma-delimited, single-quote text qualifier, Latin-1 text (FHWA's format).
# Kept: highway bridges with at least one of the deck, superstructure, and substructure ratings
# (Items 58-60) and a location inside Illinois. Culverts (Item 43B = 19) are left out; they are
# rated with Item 62 instead. `rating` is the lowest of Items 58-60, the rating that FHWA's
# good/fair/poor classification uses (poor: 4 or less). Latitude and longitude (Items 16-17,
# degrees-minutes-seconds) become decimal degrees; a year reconstructed of 0 (never) becomes
# empty.
pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$raw" \
  "$derived/nbi_il_bridges.csv" "${years[@]}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw, out, years = Path(sys.argv[1]), Path(sys.argv[2]), [int(y) for y in sys.argv[3:]]
items = {
    "STRUCTURE_NUMBER_008": "structure", "COUNTY_CODE_003": "county",
    "HIGHWAY_DISTRICT_002": "district", "FACILITY_CARRIED_007": "facility",
    "FEATURES_DESC_006A": "crosses", "LAT_016": "lat", "LONG_017": "lon", "OWNER_022": "owner",
    "YEAR_BUILT_027": "year_built", "YEAR_RECONSTRUCTED_106": "year_reconstructed",
    "ADT_029": "adt", "PERCENT_ADT_TRUCK_109": "pct_truck", "STRUCTURE_KIND_043A": "kind",
    "STRUCTURE_TYPE_043B": "design", "MAIN_UNIT_SPANS_045": "spans",
    "STRUCTURE_LEN_MT_049": "length_m", "DECK_WIDTH_MT_052": "deck_width_m",
    "PARALLEL_STRUCTURE_101": "parallel", "DECK_COND_058": "deck",
    "SUPERSTRUCTURE_COND_059": "superstructure", "SUBSTRUCTURE_COND_060": "substructure",
    "STRUCTURAL_EVAL_067": "structural_eval", "DATE_OF_INSPECT_090": "inspected",
    "INSPECT_FREQ_MONTHS_091": "inspection_interval",
}


def degrees(code: pd.Series, deg_digits: int) -> pd.Series:
    """Items 16-17, DDMMSSss (latitude) or DDDMMSSss (longitude), to decimal degrees."""
    s = code.str.zfill(deg_digits + 6)
    d, m = s.str[:deg_digits].astype(float), s.str[deg_digits:deg_digits + 2].astype(float)
    return d + m / 60 + s.str[deg_digits + 2:].astype(float) / 360000


frames = []
for year in years:
    f = pd.read_csv(raw / f"IL{year % 100:02d}.txt", quotechar="'", dtype=str,
                    encoding="latin-1", usecols=list(items))
    f = f.rename(columns=items).apply(lambda c: c.str.strip())
    f.insert(0, "nbi_year", year)
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df["lat"], df["lon"] = degrees(df["lat"], 2).round(5), -degrees(df["lon"], 3).round(5)
num = ["year_built", "year_reconstructed", "adt", "pct_truck", "spans", "length_m",
       "deck_width_m", "deck", "superstructure", "substructure", "structural_eval",
       "inspection_interval"]
# Codes that are not numbers become missing: "N" (not applicable) and, in Item 67, "*" (272
# records over the six years; not a rating). Any other code stops the run instead of being
# coerced silently.
other = {c: set(df[c].dropna()[pd.to_numeric(df[c].dropna(), errors="coerce").isna()]) - {"N", "*"}
         for c in num}
assert not any(other.values()), f"unexpected non-numeric NBI codes: { {c: v for c, v in other.items() if v} }"
df[num] = df[num].replace({"N": None, "*": None}).apply(pd.to_numeric)
df["year_reconstructed"] = df["year_reconstructed"].where(df["year_reconstructed"] > 0)
mmyy = df.pop("inspected").str.zfill(4)
yy = mmyy.str[2:].astype(int)
df["inspection_year"] = np.where(yy < 50, 2000 + yy, 1900 + yy)
df["inspection_month"] = mmyy.str[:2].astype(int)
df["rating"] = df[["deck", "superstructure", "substructure"]].min(axis=1)
inside = df["lat"].between(36.9, 42.6) & df["lon"].between(-91.6, -87.4)
keep = (df["design"] != "19") & df["rating"].notna() & inside
df = df[keep].reset_index(drop=True)
assert not df.duplicated(["nbi_year", "structure"]).any(), "one record per bridge and year"
part = out.with_name(out.name + ".part")  # written beside, then renamed: never truncated
df.to_csv(part, index=False)
part.replace(out)
print(f"wrote {out} ({len(df)} bridge-year records, {df['structure'].nunique()} bridges)")
print(df.groupby("nbi_year").size().to_string())
PY
