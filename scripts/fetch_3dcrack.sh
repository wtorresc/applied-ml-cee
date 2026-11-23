#!/usr/bin/env bash
# Fetch 3DCrack, a pixel-level crack segmentation dataset of 3D laser-scanned pavements (Zhang,
# Wang, Hsieh, Yang, Yezzi & Tsai, Georgia Institute of Technology; dataset doi:10.21227/2etq-xh36;
# paper arXiv:2508.10256, 2025). For CEE Lecture 16 (segmentation; crack width in millimetres).
# Range (depth) images from a 3D line-laser pavement survey system, collected across Georgia over
# 10 years; 512 wide x 624 high, "about 4 mm per pixel" (the record). Binary masks (0 / 255) and,
# for every cracked sample, GeoJSON crack centrelines with two widths per vertex: width_mm (from
# the crack map) and width_mm_rangeimg (from the range image). Coordinates are [row, column].
#
#   data/raw/3dcrack/            the files as published (read-only; never edit)
#     3DCrack_v1.zip             the record's single file (238 MB), checked against its MD5
#     3DCrack_v1/                the archive, extracted (read-only):
#       Train|Val|Test/img/      range images, sample_XXXXXX.png
#       Train|Val|Test/anno/     masks, same stems
#       Train|Val|Test/CVM/      centrelines, sample_XXXXXX.geojson (crack-free samples have none)
#
# Counts asserted below (from the record): 1,139 / 245 / 250 images and masks; 870 / 185 / 197
# centreline files; 382 crack-free samples.
#
# Source: Zenodo record 23681508 (the authors' upload of the IEEE DataPort dataset).
# Licence: Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the record. Cite the dataset
# DOI and the paper. No account or contact details are needed.
#
# Usage: bash scripts/fetch_3dcrack.sh   (from anywhere; needs curl, md5sum, unzip)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/3dcrack"
url="https://zenodo.org/api/records/23681508/files/3DCrack_v1.zip/content"
md5="62188344ebdb4a4fef2618de9471df50"
mkdir -p "$raw"

zip="$raw/3DCrack_v1.zip"
if [[ ! -f "$zip" ]]; then
  curl -sSfL -o "$zip.part" "$url"
  mv "$zip.part" "$zip"
fi
echo "$md5  $zip" | md5sum --check --quiet
chmod a-w "$zip"

if [[ ! -d "$raw/3DCrack_v1" ]]; then
  tmp="$raw/extract.part"
  rm -rf "${tmp:?}"
  mkdir "$tmp"
  unzip -q "$zip" -d "$tmp"
  [[ -d "$tmp/3DCrack_v1/Train/img" ]] || { echo "unexpected layout in $zip" >&2; exit 1; }
  mv "$tmp/3DCrack_v1" "$raw/3DCrack_v1"
  rm -rf "${tmp:?}"
  chmod -R a-w "$raw/3DCrack_v1"
fi

count() { find "$raw/3DCrack_v1/$1" -name "$2" | wc -l; }
for spec in "Train 1139 870" "Val 245 185" "Test 250 197"; do
  read -r split n_img n_cvm <<< "$spec"
  for sub in img anno; do
    n=$(count "$split/$sub" '*.png')
    [[ "$n" -eq "$n_img" ]] || { echo "$split/$sub: expected $n_img, found $n" >&2; exit 1; }
  done
  n=$(count "$split/CVM" '*.geojson')
  [[ "$n" -eq "$n_cvm" ]] || { echo "$split/CVM: expected $n_cvm, found $n" >&2; exit 1; }
done
echo "3DCrack in $raw/3DCrack_v1: 1,634 samples (1,139 / 245 / 250), 1,252 centreline files"
