#!/usr/bin/env bash
# Fetch CAMELS (Catchment Attributes and MEteorology for Large-sample Studies; Newman et al., 2015,
# HESS 19, 209-223; Addor et al., 2017, HESS 21, 5293-5313), version 1.2: daily forcing and USGS
# streamflow for 671 minimally disturbed US basins, 1980-2014, and their catchment attributes.
# For CEE Lecture 20 (1-D signals, 1-D CNNs and LSTMs). The record also holds calibrated
# SAC-SMA + Snow-17 output (4.2 GB per forcing); the lecture cites that benchmark (Kratzert et al.,
# 2019, HESS 23, 5089-5110) instead of reproducing it, so it is not fetched.
#
#   data/raw/camels/             the files as published (read-only; never edit)
#     basin_timeseries_v1p2_metForcing_obsFlow.zip   forcing (Daymet, Maurer, NLDAS) and observed
#                                                    streamflow, 3.4 GB
#     camels_{clim,geol,hydro,name,soil,topo,vege}.txt   attributes (Addor et al., 2017)
#     camels_attributes_v2.0.pdf, readme.txt         their documentation
#     kratzert2019_basin_list.txt                    the 531 basins Kratzert et al. (2019) used
#
# Every CAMELS file is checked against the MD5 the record publishes; the basin list against the
# MD5 of the pinned commit's file.
#
# Source: Zenodo record 15529996 (UCAR/NCAR GDEX dataset doi:10.5065/D6MW2F4D, version 1.2).
# Licence: Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the record. Cite the dataset
# DOI and both papers. No account or contact details are needed.
# Basin list: github.com/kratzert/ealstm_regional_modeling, data/basin_list.txt at commit 64a446e
# (Apache-2.0). The 531 are Newman et al.'s (2017) selection: all under 2,000 km^2, but a 10% rule
# on CAMELS' two area columns gives 558, not these 531, so the list is fetched, not derived.
#
# Usage: bash scripts/fetch_camels.sh   (from anywhere; about 3.5 GB; needs curl, md5sum)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/camels"
api="https://zenodo.org/api/records/15529996/files"
mkdir -p "$raw"

# name md5 (from the record's file list)
files="
camels_clim.txt 67f22592f3fb72c57df81358ce68458b
camels_geol.txt f5ce5de53eb1ea2532cda7e3b4813993
camels_hydro.txt 55ebdeb36c42ee7acdb998229c3edb3a
camels_name.txt c96491b32c4df55a31bead7ceca7d64b
camels_soil.txt 8edb46a363a20b466a4b7105ba633767
camels_topo.txt 0f6267838c40b1507b64582433bc0b8e
camels_vege.txt f40e843defc1e654a800be9fe5fd5090
camels_attributes_v2.0.pdf 77a6c084c798a31fbd05594ee58a90c7
readme.txt b37d64950e9d4c5c10a8b4ef82bc6219
basin_timeseries_v1p2_metForcing_obsFlow.zip 8e9a466710e8270b58f01d332a87184f
"
while read -r name md5; do
  [[ -z "$name" ]] && continue
  out="$raw/$name"
  if [[ ! -f "$out" ]]; then
    # -C -: resume a partial download. The MD5 is checked before the rename, so a bad join never
    # gets the final name; delete the .part and run again.
    curl -sSfL --retry 3 -C - -o "$out.part" "$api/$name/content"
    echo "$md5  $out.part" | md5sum --check --quiet \
      || { echo "$out.part: MD5 mismatch; delete it and run again" >&2; exit 1; }
    mv "$out.part" "$out"
  fi
  echo "$md5  $out" | md5sum --check --quiet
  chmod a-w "$out"
done <<< "$files"

list="$raw/kratzert2019_basin_list.txt"
repo="https://raw.githubusercontent.com/kratzert/ealstm_regional_modeling"
commit="64a446e9012ecd601e0a9680246d3bbf3f002f6d"
if [[ ! -f "$list" ]]; then
  curl -sSfL -o "$list.part" "$repo/$commit/data/basin_list.txt"
  echo "57c4bf6a28b6cdffdca39d67d33e82a9  $list.part" | md5sum --check --quiet
  mv "$list.part" "$list"
fi
echo "57c4bf6a28b6cdffdca39d67d33e82a9  $list" | md5sum --check --quiet
chmod a-w "$list"
echo "CAMELS files in $raw: all MD5s match"
