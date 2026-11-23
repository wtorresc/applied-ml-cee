#!/usr/bin/env bash
# Download five volumes of FHWA's Long-Term Pavement Performance (LTPP) Standard Data Release 40,
# export the tables CEE Lecture 8 uses, and derive one row per asphalt test section. For CEE
# Lecture 8 (decision trees, forests, boosting; explanations).
#
#   data/raw/ltpp/<volume>.zip           the release's zip files, as served (read-only)
#   data/raw/ltpp/<volume>/              their Access databases (.mdb), unzipped (read-only)
#   data/derived/ltpp/<TABLE>.csv        the tables below, exported with mdbtools' mdb-export
#   data/derived/ltpp_iri_sections.csv   one row per section (GPS-1, GPS-2, SPS-1; original
#                                        construction, CONSTRUCTION_NO = 1) with at least three
#                                        roughness visits over at least four years: the IRI
#                                        growth rate and the section's structure, traffic, and
#                                        climate (built by the Python step below)
#
# Source: LTPP InfoPave, Standard Data Release (SDR 40, released September 2026),
# https://infopave.fhwa.dot.gov/Data/StandardDataRelease ("View By: Topic"). The volumes are
# served from the CDN URLs below; the release page is the stable entry point if they change. No
# account or email is needed for these volumes.
#
# Terms: U.S. Government Works, public (data.gov record "Long-Term Pavement Performance (LTPP)");
# FHWA's disclaimer: "Information presented on this web site is considered public information
# and may be distributed or copied." Cite: Long Term Pavement Performance InfoPave, Standard Data
# Release, SDR 40, https://infopave.fhwa.dot.gov/Data/StandardDataRelease.
#
# Needs: curl, unzip, and mdbtools (in the pixi env aml-env). About 70 MB zipped, 440 MB unzipped.
# Usage: bash scripts/fetch_ltpp.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/ltpp"
csv="$root/data/derived/ltpp"
base="https://du993ylnpbddg.cloudfront.net/SDR/40/By_Volume"
mkdir -p "$raw" "$csv"

# volume name | folder on the CDN | file name
volumes=(
  "ADMINISTRATION|Climate|SDR40 - ADMINISTRATION.zip"
  "MONITORING|Performance|SDR40 - MONITORING.zip"
  "TRAFFIC|Traffic|SDR40 - TRAFFIC.zip"
  "CLIMATE_SUMMARY_DATA|Climate|SDR40 - CLIMATE_SUMMARY_DATA.ZIP"
  "CONSOLIDATED|Pavement Structure and Construction|SDR40 - CONSOLIDATED.zip"
)
for entry in "${volumes[@]}"; do
  IFS="|" read -r vol folder file <<< "$entry"
  zip="$raw/$vol.zip"
  if [[ ! -f "$zip" ]]; then
    curl -sSfL -o "$zip.part" "$base/${folder// /%20}/${file// /%20}"
    mv "$zip.part" "$zip"
    chmod a-w "$zip"
  fi
  if [[ ! -d "$raw/$vol" ]]; then   # unzip beside, then rename: no half-unzipped volume
    rm -rf "$raw/$vol.part"
    unzip -q "$zip" -d "$raw/$vol.part"
    find "$raw/$vol.part" -type f -exec chmod a-w {} +
    mv "$raw/$vol.part" "$raw/$vol"
  fi
done

mdb() { find "$raw/$1" -iname "*.mdb" -print -quit; }
export_table() {   # volume table; written beside, then renamed, so a failed export is redone
  local out="$csv/$2.csv"
  if [[ ! -f "$out" ]]; then
    pixi run --manifest-path "$root/pixi.toml" -e aml-env mdb-export "$(mdb "$1")" "$2" > "$out.part"
    mv "$out.part" "$out"
  fi
}
for t in EXPERIMENT_SECTION SECTION_COORDINATES SECTION_LAYER_STRUCTURE; do export_table ADMINISTRATION "$t"; done
for t in MON_HSS_PROFILE_SECTION MON_HSS_VISIT_NO; do export_table MONITORING "$t"; done
export_table TRAFFIC TRF_TREND
for t in CLM_SITE_VWS_LINK CLM_VWS_TEMP_ANNUAL CLM_VWS_PRECIP_ANNUAL; do export_table CLIMATE_SUMMARY_DATA "$t"; done
export_table CONSOLIDATED PROJECT_HIST_AGE_EXP

# Derived: one row per section. IRI = mean of the left and right wheel paths, averaged over the
# runs of a visit; the growth rate is the least-squares slope of IRI on age (years since opening
# to traffic). SPS sections share their project's climate station (SHRP_ID "0123" -> "0100").
pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$csv" \
  "$root/data/derived/ltpp_iri_sections.csv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

csv, out = Path(sys.argv[1]), Path(sys.argv[2])
key = ["STATE_CODE", "SHRP_ID"]


def rd(table: str) -> pd.DataFrame:
    return pd.read_csv(csv / f"{table}.csv",
                       dtype={"SHRP_ID": str, "VWS_ID": str, "VISIT_NO": str, "EXPERIMENT_NO": str})


def date(s: pd.Series) -> pd.Series:
    """mdb-export's dates, MM/DD/YY HH:MM:SS, to timestamps; empty cells become NaT.

    The year has two digits, and %y maps 00-68 to 20xx: years after 2030 are 19xx (SDR 40's
    dates run from 1937 to 2025). A malformed date raises instead of becoming NaT.
    """
    d = pd.to_datetime(s, format="%m/%d/%y %H:%M:%S", errors="raise")
    return d.where(d.dt.year <= 2030, d - pd.DateOffset(years=100))


ex = rd("EXPERIMENT_SECTION")
pick = ((ex["GPS_SPS"] == "G") & ex["EXPERIMENT_NO"].isin(["1", "2"])) | \
       ((ex["GPS_SPS"] == "S") & (ex["EXPERIMENT_NO"] == "1"))
sec = ex[pick & (ex["CONSTRUCTION_NO"] == 1)][key + ["GPS_SPS", "EXPERIMENT_NO"]].drop_duplicates(key)
age = rd("PROJECT_HIST_AGE_EXP")
age["open"] = date(age["TRAFFIC_OPEN_DATE"]).fillna(date(age["CONSTRUCTION_DATE"]))
sec = sec.merge(age[key + ["open"]], on=key, how="left")

prof, vis = rd("MON_HSS_PROFILE_SECTION"), rd("MON_HSS_VISIT_NO")
p = prof.merge(vis[["VISIT_NO"] + key + ["CONSTRUCTION_NO", "VISIT_DATE"]], on="VISIT_NO")
p = p[p["CONSTRUCTION_NO"] == 1]
p["iri"] = p[["IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH"]].mean(axis=1)
v = p.groupby(key + ["VISIT_NO"]).agg(date=("VISIT_DATE", "first"), iri=("iri", "mean")).reset_index()
v["date"] = date(v["date"])
v = v.merge(sec, on=key)
v["age"] = (v["date"] - v["open"]).dt.days / 365.25
v = v[v["age"].notna() & (v["age"] >= 0) & v["iri"].notna()]


def rate(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("age")
    out = {"visits": len(g), "span_yr": g["age"].max() - g["age"].min(), "iri_rate": np.nan,
           "iri_first": g["iri"].iloc[0], "age_first": g["age"].iloc[0]}
    if len(g) >= 3 and out["span_yr"] >= 4:
        out["iri_rate"] = np.polyfit(g["age"], g["iri"], 1)[0]
    return pd.Series(out)


r = v.groupby(key + ["GPS_SPS", "EXPERIMENT_NO"]).apply(rate, include_groups=False).reset_index()
r = r.dropna(subset=["iri_rate"])

lay = rd("SECTION_LAYER_STRUCTURE")
lay = lay[lay["CONSTRUCTION_NO"] == 1]
thick = lay.pivot_table(index=key, columns="LAYER_TYPE", values="REPR_THICKNESS", aggfunc="sum")
struct = pd.DataFrame({"ac_in": thick.get("AC"), "granular_base_in": thick.get("GB"),
                       "treated_base_in": thick.get("TB")}).fillna({"granular_base_in": 0,
                                                                     "treated_base_in": 0})
tr = rd("TRF_TREND")
tr = tr[tr["CONSTRUCTION_NO"] == 1]
traffic = tr.groupby(key)["ANNUAL_ESAL_TREND"].median().rename("esal_per_yr")
link = rd("CLM_SITE_VWS_LINK").drop_duplicates(key)
temp, prec = rd("CLM_VWS_TEMP_ANNUAL"), rd("CLM_VWS_PRECIP_ANNUAL")
clim = temp.groupby("VWS_ID").agg(freeze_index=("FREEZE_INDEX_YR", "mean"),
                                  mean_temp_c=("MEAN_ANN_TEMP_AVG", "mean"))
clim = clim.join(prec.groupby("VWS_ID").agg(precip_mm=("TOTAL_ANN_PRECIP", "mean")))

r["link_id"] = np.where(r["GPS_SPS"] == "S", r["SHRP_ID"].str[:2] + "00", r["SHRP_ID"])
r = r.merge(link[key + ["VWS_ID"]].rename(columns={"SHRP_ID": "link_id"}),
            on=["STATE_CODE", "link_id"], how="left")
r = r.merge(clim, left_on="VWS_ID", right_index=True, how="left")
r = r.merge(struct, left_on=key, right_index=True, how="left").merge(traffic, on=key, how="left")
cols = ["ac_in", "granular_base_in", "treated_base_in", "esal_per_yr", "freeze_index",
        "precip_mm", "mean_temp_c", "iri_first", "age_first"]
r = r.dropna(subset=cols)
r["site"] = np.where(r["GPS_SPS"] == "S", r["STATE_CODE"].astype(str) + "-" + r["SHRP_ID"].str[:2],
                     r["STATE_CODE"].astype(str) + "-" + r["SHRP_ID"])
keep = key + ["GPS_SPS", "EXPERIMENT_NO", "site", "visits", "span_yr", "iri_rate"] + cols
counts = (len(r), r["STATE_CODE"].nunique(), r["site"].nunique())
# SDR 40 gives these counts; a change means the release or a rule above changed.
assert counts == (360, 53, 203), f"sections, states, sites: {counts}, expected (360, 53, 203)"
# Five significant digits: finer than IRI's 0.01 m/km resolution for every column.
part = out.with_name(out.name + ".part")  # written beside, then renamed: never truncated
r[keep].sort_values(key).to_csv(part, index=False, float_format="%.5g")
part.replace(out)
print(f"{out}: {counts[0]} sections, {counts[1]} states, {counts[2]} sites")
PY
