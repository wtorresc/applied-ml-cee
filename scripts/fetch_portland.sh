#!/usr/bin/env bash
# Download freeway loop-detector data for two stations on I-205 southbound in Portland, Oregon
# (USDOT Data Capture and Management test data set, Sept 15 - Nov 15, 2011), and derive
# 5-minute station records. For CEE Lecture 7 (naive Bayes, mixtures and EM, KDE: free-flow and
# congested traffic).
#
#   data/raw/portland_i205/detector_<id>.csv   20-second readings per lane detector, as served
#                                              (read-only; never edit): volume (vehicles),
#                                              speed (mph, dual loops), occupancy (%), status,
#                                              dqflags; timestamps in UTC
#   data/derived/portland_i205_5min.csv        one row per station and 5 minutes, local time:
#                                              volume (veh/5 min, all lanes), occupancy (mean
#                                              of lanes, %), speed (volume-weighted, mph), and
#                                              the number of 20-second lane readings behind it
#
# Source: "Portland, Oregon Test Data Set Freeway Loop Detector Data", data.transportation.gov
# dataset 6bhp-rgk2, doi:10.21949/1504472 (Portland State University, 2012; provided by the
# USDOT ITS DataHub). Read through the Socrata SODA API; no account or key is needed.
#
# Licence: Creative Commons Attribution-ShareAlike 3.0 Unported, as stated on the dataset page.
# Cite: Portland State University. (2012). Oregon Test Data Set Freeway Loop Detector Data.
# [Dataset]. Provided by ITS DataHub through Data.transportation.gov. A derived file shared with
# students must carry the same licence.
#
# Stations (from the test data set's detector and station tables, documented in the dataset's
# metadata file; lane 1 is the left lane):
#   1143  I-205 SB at Glisan,         milepost 21.12, detectors 1953 1954 1955
#   1051  I-205 SB at Stark/Washington, milepost 20.34, detectors 1393 1394 1395
#
# Usage: bash scripts/fetch_portland.sh   (from anywhere; about 110 MB raw)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/portland_i205"
derived="$root/data/derived"
api="https://data.transportation.gov/resource/6bhp-rgk2.csv"
page=50000
mkdir -p "$raw" "$derived"

for det in 1953 1954 1955 1393 1394 1395; do
  out="$raw/detector_$det.csv"
  [[ -f "$out" ]] && continue
  offset=0
  : > "$out.part"
  while :; do
    chunk="$(mktemp)"
    curl -sSf -G "$api" \
      --data-urlencode "\$select=detectorid,starttime,volume,speed,occupancy,status,dqflags" \
      --data-urlencode "\$where=detectorid='$det'" \
      --data-urlencode "\$order=starttime" \
      --data-urlencode "\$limit=$page" --data-urlencode "\$offset=$offset" -o "$chunk"
    rows=$(($(wc -l < "$chunk") - 1))
    if [[ $offset -eq 0 ]]; then cat "$chunk" >> "$out.part"; else tail -n +2 "$chunk" >> "$out.part"; fi
    rm -f "$chunk"
    (( rows < page )) && break
    offset=$((offset + page))
  done
  mv "$out.part" "$out"
  chmod a-w "$out"
done

# Derived: 5-minute station records in local (Pacific) time. Readings with a missing volume or
# occupancy are left out; speed is the volume-weighted mean of the lanes' 20-second speeds.
pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$raw" \
  "$derived/portland_i205_5min.csv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
station = {1953: 1143, 1954: 1143, 1955: 1143, 1393: 1051, 1394: 1051, 1395: 1051}
frames = []
for det, st in station.items():
    d = pd.read_csv(raw / f"detector_{det}.csv")
    assert (d["detectorid"] == det).all()
    d["station"] = st
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d["time"] = (pd.to_datetime(d["starttime"], utc=True).dt.tz_convert("America/Los_Angeles")
             .dt.tz_localize(None))
d = d.dropna(subset=["volume", "occupancy"])
d["slot"] = d["time"].dt.floor("5min")
d["vs"] = d["volume"] * d["speed"]
d["vs_n"] = d["volume"].where(d["speed"].notna(), 0)
g = d.groupby(["station", "slot"])
agg = pd.DataFrame({
    "volume": g["volume"].sum(),
    "occupancy": g["occupancy"].mean(),
    "speed": g["vs"].sum() / g["vs_n"].sum().replace(0, np.nan),
    "readings": g["volume"].size(),
}).reset_index().rename(columns={"slot": "time"})
part = out.with_name(out.name + ".part")  # written beside, then renamed: never truncated
agg.to_csv(part, index=False, float_format="%.3f")
part.replace(out)
print(f"{out}: {len(agg)} station records, {agg['time'].min()} to {agg['time'].max()}")
PY
