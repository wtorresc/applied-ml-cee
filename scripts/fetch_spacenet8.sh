#!/usr/bin/env bash
# Fetch the Germany area of SpaceNet 8 (flooded buildings and roads; Hänsch et al., 2022, "SpaceNet
# 8 - The Detection of Flooded Roads and Buildings", CVPR Workshops). For CEE Lecture 16.
# Maxar pansharpened RGB satellite tiles taken before and after the July 2021 floods in Germany
# (pre-event tiles 1300 x 1300 px, 0.3-0.8 m per pixel, EPSG:4326 so pixels are not square on the
# ground), with building footprints and road centrelines from OpenStreetMap, each marked flooded or
# not. 202 tiles; the paper's Table 1 gives 3,723 buildings, 32% flooded.
#
#   data/raw/spacenet8/                       the files as published (read-only; never edit)
#     Germany_Training_Public.tar.gz          the challenge's tarball (1,357,293,885 bytes)
#     Germany_Training_Public/                the tarball, extracted (read-only):
#       PRE-event/ POST-event/                GeoTIFF tiles (some tiles have two post images)
#       annotations/                          one GeoJSON per tile
#       Germany_Training_Public_label_image_mapping.csv, Germany_Training_Public_reference.csv
#     LICENSE.md, resolutions.txt             from the bucket
#
# Source: s3://spacenet-dataset/spacenet/SN8_floods/ (AWS Open Data; no account: plain HTTPS works,
# as does `aws s3 cp --no-sign-request`). The bucket publishes no checksums, so the script checks
# the byte count and the tile counts.
# Licence: the bucket's LICENSE.md: "The 'SpaceNet Dataset' is licensed under a Creative Commons
# Attribution-ShareAlike 4.0 International License" (CC BY-SA 4.0). The SpaceNet 8 paper states that
# the Maxar Open Data imagery is released under a Creative Commons non-commercial licence: fine for
# teaching and research; cite both, and do not use the imagery commercially.
#
# Usage: bash scripts/fetch_spacenet8.sh   (from anywhere; needs curl and tar)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/spacenet8"
base="https://spacenet-dataset.s3.amazonaws.com"
name="Germany_Training_Public"
bytes=1357293885
mkdir -p "$raw"

get() {  # get <url> <file>: download to .part, then rename
  [[ -f "$2" ]] && return 0
  curl -sSfL -o "$2.part" "$1"
  mv "$2.part" "$2"
  chmod a-w "$2"
}
get "$base/LICENSE.md" "$raw/LICENSE.md"
get "$base/spacenet/SN8_floods/resolutions.txt" "$raw/resolutions.txt"
tarball="$raw/$name.tar.gz"
get "$base/spacenet/SN8_floods/tarballs/$name.tar.gz" "$tarball"
n=$(stat -c %s "$tarball")
[[ "$n" -eq "$bytes" ]] || { echo "$tarball: expected $bytes bytes, found $n" >&2; exit 1; }

if [[ ! -d "$raw/$name" ]]; then
  tmp="$raw/extract.part"
  rm -rf "${tmp:?}"
  mkdir "$tmp"
  # The tarball has no top-level folder: its entries are PRE-event/, POST-event/, annotations/,
  # the two CSVs, and a stray 901 kB file also named Germany_Training_Public.tar.gz (kept as is).
  tar -xzf "$tarball" -C "$tmp" --no-same-owner --no-same-permissions
  [[ -d "$tmp/PRE-event" ]] || { echo "unexpected layout in $tarball" >&2; exit 1; }
  mv "$tmp" "$raw/$name"
  chmod -R a-w "$raw/$name"
fi

check() {  # check <subdir> <pattern> <expected>
  local n
  n=$(find "$raw/$name/$1" -name "$2" | wc -l)
  [[ "$n" -eq "$3" ]] || { echo "$1: expected $3 files, found $n" >&2; exit 1; }
}
check PRE-event '*.tif' 202
check POST-event '*.tif' 282
check annotations '*.geojson' 202
echo "SpaceNet 8 Germany in $raw/$name: 202 tiles (202 pre, 282 post images)"
