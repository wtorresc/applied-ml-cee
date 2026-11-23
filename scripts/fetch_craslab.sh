#!/usr/bin/env bash
# Fetch the labelled indoor point cloud dataset for BIM applications (Abreu et al., 2023,
# Data 8(6):101; Zenodo record 7948116, DOI 10.5281/zenodo.7948116): 21 Leica BLK360 G2 scans
# of four rooms and a hallway of the Department of Electrical and Computer Engineering,
# University of Porto (CRAS lab), every point labelled, and the rooms' as-built BIM model (IFC).
# For CEE Lecture 22 (point clouds, registration, scan-vs-BIM).
#
#   data/raw/craslab/craslabannotated.zip   the labelled scans, as published (read-only)
#   data/raw/craslab/craslabbim.ifc         the as-built BIM model (read-only)
#
# Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), as the Zenodo record
# states (INESC TEC's record of the same data, DOI 10.25747/6FAM-VY11, lists CC BY-SA; the copy
# fetched here is Zenodo's). No account needed.
# Usage: bash scripts/fetch_craslab.sh   (about 4.3 GB; needs curl, md5sum)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/craslab"
base="https://zenodo.org/api/records/7948116/files"
mkdir -p "$raw"

fetch() {  # name md5
  local out="$raw/$1"
  if [[ ! -f "$out" ]]; then
    curl -sSfL --retry 3 -o "$out.part" "$base/$1/content"
    echo "$2  $out.part" | md5sum --check --quiet \
      || { echo "$out.part: MD5 differs from Zenodo's; not kept" >&2; exit 1; }
    mv "$out.part" "$out"
  fi
  echo "$2  $out" | md5sum --check --quiet
  chmod a-w "$out"
}

fetch craslabbim.ifc e20658f0d2d9e13c62363169b7fa3193
fetch craslabannotated.zip e5ecedab8f2a1d1f91861a3aec028a72
echo "CRAS lab scans and BIM in $raw"
