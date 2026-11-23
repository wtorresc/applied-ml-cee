#!/usr/bin/env bash
# Fetch SDNET2018, an annotated image dataset of cracked and uncracked concrete (Maguire,
# Dorafshan & Thomas, 2018; Dorafshan, Thomas & Maguire, 2018, Data in Brief 21, 1664-1668,
# doi:10.1016/j.dib.2018.11.015). For CEE Lecture 15 (convolutional networks, transfer learning).
# 230 photographs taken with a 16 MP camera (54 bridge decks in USU's SMASH laboratory, 72 walls
# of a campus building, 104 campus pavements), cut into 256 x 256 px sub-images, each labeled
# cracked (C) or uncracked (U): over 56,000 sub-images.
#
#   data/raw/sdnet2018/        the files as published (read-only; never edit)
#     DATA_Maguire_20180517_ALL.zip   the record's download (504 MB), checked against its MD5
#     ReadMe_SDNET2018.txt     the authors' description (from the archive)
#     SDNET2018/               the image archive inside it, extracted (read-only)
#
# Source: Utah State University Digital Commons, all_datasets/48, doi:10.15142/T3TD19.
# The repository refuses downloads from scripts (HTTP 403, checked 2026-11-16). If the download
# below fails, open https://doi.org/10.15142/T3TD19 in a browser, click "Download" (503.8 MB),
# save the file as data/raw/sdnet2018/DATA_Maguire_20180517_ALL.zip, and run this script again:
# it checks the checksum and extracts the images.
#
# Licence: Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the record. Cite Maguire,
# Dorafshan & Thomas (2018). No account or contact details are needed.
#
# Usage: bash scripts/fetch_sdnet2018.sh   (from anywhere; needs curl, md5sum, unzip)
# Then: pixi run -e aml-env python scripts/derive_sdnet2018.py   (the tile index and the
# ImageNet ResNet-18 features that Lecture 15 reads, in data/derived/)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/sdnet2018"
url="https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent"
md5="677411e784f194422c90f52d9ed0d7c6"
mkdir -p "$raw"

zip="$raw/DATA_Maguire_20180517_ALL.zip"
if [[ ! -f "$zip" ]]; then
  if ! curl -sSfL -o "$zip.part" "$url"; then
    rm -f "$zip.part"
    echo "Download refused. Save the record's download from https://doi.org/10.15142/T3TD19" >&2
    echo "with a browser as $zip, then run this script again." >&2
    exit 1
  fi
  mv "$zip.part" "$zip"
fi
echo "$md5  $zip" | md5sum --check --quiet
chmod a-w "$zip"

if [[ ! -d "$raw/SDNET2018" ]]; then
  # The download holds the ReadMe and a second archive with the images.
  tmp="$raw/extract.part"
  rm -rf "$tmp"
  mkdir "$tmp"
  unzip -q "$zip" -d "$tmp"
  inner="$tmp/DATA_Maguire_20180517_ALL"
  mkdir "$tmp/images"
  unzip -q "$inner/SDNET2018.zip" -d "$tmp/images"
  if [[ ! -f "$raw/ReadMe_SDNET2018.txt" ]]; then
    mv "$inner/ReadMe_SDNET2018 .txt" "$raw/ReadMe_SDNET2018.txt"
    chmod a-w "$raw/ReadMe_SDNET2018.txt"
  fi
  # Three surfaces (D decks, P pavements, W walls), each cracked (C*) and uncracked (U*); a file
  # <photo>-<tile>.jpg is one 256 x 256 tile of a source photograph.
  for d in D/CD D/UD P/CP P/UP W/CW W/UW; do
    [[ -d "$tmp/images/$d" ]] || { echo "missing $d in $inner/SDNET2018.zip" >&2; exit 1; }
  done
  mv "$tmp/images" "$raw/SDNET2018"
  rm -rf "$tmp"
  chmod -R a-w "$raw/SDNET2018"
fi

n=$(find "$raw/SDNET2018" -name '*.jpg' | wc -l)
[[ "$n" -eq 56092 ]] || { echo "expected 56,092 images, found $n" >&2; exit 1; }
echo "SDNET2018 in $raw: $n images"
