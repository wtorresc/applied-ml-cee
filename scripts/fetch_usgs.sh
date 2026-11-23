#!/usr/bin/env bash
# Download daily mean discharge for the Sangamon River at Monticello, Illinois (USGS 05572000;
# drainage area 550 square miles, Piatt County), 1908 to 2025, and derive one tidy table. For
# CEE Lecture 11 (conformal prediction, covariate shift, nonstationarity).
#
#   data/raw/usgs_05572000/daily_<decade>.csv   the API's CSV, one file per decade, as served
#                                               (read-only; never edit): one row per day, value
#                                               in ft^3/s, approval status, qualifier
#   data/derived/usgs_05572000_daily.csv        date, discharge in m^3/s, approval status;
#                                               sorted by date, one row per day
#
# Source: USGS Water Data APIs (OGC API - Features), collection "daily", parameter 00060
# (discharge), statistic 00003 (daily mean): https://api.waterdata.usgs.gov/ogcapi/v0/. The
# legacy service waterservices.usgs.gov is scheduled to shut down on February 22, 2027, so this
# script uses the new API. No key is needed (a key only raises rate limits).
#
# Terms: USGS-authored or produced data are in the U.S. public domain ("USGS Copyrights and
# Credits"); cite the U.S. Geological Survey, National Water Information System.
#
# Usage: bash scripts/fetch_usgs.sh   (from anywhere; about 4 MB)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/usgs_05572000"
derived="$root/data/derived"
api="https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
mkdir -p "$raw" "$derived"

for start in $(seq 1900 10 2020); do
  out="$raw/daily_${start}s.csv"
  [[ -f "$out" ]] && continue
  curl -sSf -G "$api" --data-urlencode "f=csv" \
    --data-urlencode "monitoring_location_id=USGS-05572000" \
    --data-urlencode "parameter_code=00060" --data-urlencode "statistic_id=00003" \
    --data-urlencode "time=${start}-01-01/$((start + 9))-12-31" \
    --data-urlencode "limit=10000" -o "$out.part"
  mv "$out.part" "$out"
  chmod a-w "$out"
done

pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$raw" \
  "$derived/usgs_05572000_daily.csv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
frames = [pd.read_csv(p) for p in sorted(raw.glob("daily_*.csv"))]
d = pd.concat([f for f in frames if len(f)], ignore_index=True)
assert (d["unit_of_measure"] == "ft^3/s").all(), d["unit_of_measure"].unique()
assert (d["monitoring_location_id"] == "USGS-05572000").all()
d["date"] = pd.to_datetime(d["time"])
d = d.sort_values("date").drop_duplicates("date")
d = d[d["date"] <= "2025-12-31"]
d["discharge_m3s"] = d["value"] * 0.3048 ** 3
part = out.with_name(out.name + ".part")  # written beside, then renamed: never truncated
d[["date", "discharge_m3s", "approval_status"]].to_csv(part, index=False, float_format="%.4f")
part.replace(out)
print(f"{out}: {len(d)} days, {d['date'].min().date()} to {d['date'].max().date()}")
PY
