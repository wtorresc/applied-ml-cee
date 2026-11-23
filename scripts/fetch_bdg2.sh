#!/usr/bin/env bash
# Download the Building Data Genome 2 electricity meters and building metadata (Miller et al.,
# 2020, Scientific Data 7, 368, doi:10.1038/s41597-020-00712-x) and derive daily load profiles.
#
#   data/raw/bdg2/                 the two files as published (read-only; never edit)
#     electricity_cleaned.csv      hourly kWh, 2016-01-01 to 2017-12-31, local time, one column
#                                  per building (the authors' "cleaned" meters: outliers and
#                                  electricity zero readings removed), 175 MB
#     metadata.csv                 one row per building: site, primary use, floor area, ...
#   data/derived/bdg2_daily_electricity.npz
#                                  float32 (buildings, days, 24) hourly kWh, NaN where missing,
#                                  with building ids and dates (built by the Python step below)
#
# Source: https://github.com/buds-lab/building-data-genome-project-2 (files stored with Git LFS,
# served from media.githubusercontent.com). Release v1.0 is archived at
# doi:10.5281/zenodo.3887306. Each download is checked against the SHA-256 in the LFS pointer.
#
# Licence: the repository's LICENSE file is Creative Commons Attribution-ShareAlike ("CC BY-SA
# 4.0"); the Zenodo archive of release v1.0 lists CC BY 4.0. We follow the stricter of the two
# (attribution, share-alike), cite Miller et al. (2020), and do not redistribute the data.
#
# Usage: bash scripts/fetch_bdg2.sh   (from the repository root; about 175 MB)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/bdg2"
derived="$root/data/derived"
base="https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master"
mkdir -p "$raw" "$derived"

# path in the repository, SHA-256 from its Git LFS pointer
files=(
  "data/meters/cleaned/electricity_cleaned.csv
   b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"
  "data/metadata/metadata.csv 992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"
)
for entry in "${files[@]}"; do
  read -r -d '' path sha <<< "$entry" || true
  out="$raw/$(basename "$path")"
  if [[ ! -f "$out" ]]; then
    curl -sSfL -o "$out.part" "$base/$path"
    echo "$sha  $out.part" | sha256sum --check --quiet
    mv "$out.part" "$out"
    chmod a-w "$out"
  fi
done

# Derived: reshape the hourly table to (building, day, hour). Pure reshaping; no filtering.
pixi run --manifest-path "$root/pixi.toml" -e aml-env python -I - "$raw" "$derived" <<'EOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw, derived = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_csv(raw / "electricity_cleaned.csv", index_col="timestamp", parse_dates=True)
hours = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
assert df.index.equals(hours), "expected every hour of 2016-2017, in order"
kwh = df.to_numpy(np.float32).T.reshape(df.shape[1], len(hours) // 24, 24)
assert kwh.shape == (df.shape[1], 731, 24), kwh.shape
dates = hours[::24].strftime("%Y-%m-%d").to_numpy()
out = derived / "bdg2_daily_electricity.npz"
part = out.with_name(out.name + ".part")  # written beside, then renamed: never truncated
with part.open("wb") as fh:  # a file object: savez would add .npz to a path
    np.savez_compressed(fh, kwh=kwh, building_id=df.columns.to_numpy(str), date=dates.astype(str))
part.replace(out)
print(f"wrote {out}: {kwh.shape[0]} buildings x {kwh.shape[1]} days x 24 hours")
EOF
