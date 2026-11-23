#!/usr/bin/env bash
# Fetch part of the KW51 monitoring data set: a steel bowstring railway bridge in Leuven, Belgium,
# monitored for 15 months (October 2018 - January 2020) before, during, and after the connections
# of its diagonals were strengthened (Maes & Lombaert, 2021, "Monitoring railway bridge KW51
# before, during, and after retrofitting", ASCE J. Bridge Eng. 26(3), 04721001). For CEE
# Lecture 20 (1-D signals: spectra, spectrograms, natural frequencies).
#
#   data/raw/kw51/               the files as published (read-only; never edit)
#     readme.txt                 the record's description of the folders
#     matlab-functions.zip       the authors' plotting functions (document the file format)
#     trackedmodes.zip           the identified natural frequencies and modes over the 15 months
#     traindata_201810.zip       accelerations, strains, and temperature during train passages,
#                                October 2018 (before the retrofit)
#
# Only these four of the record's 35 files (45.6 GB) are fetched: the lecture needs the tracked
# frequencies and one month of train crossings. Every file is checked against the record's MD5.
#
# Source: Zenodo record 3745914 (doi:10.5281/zenodo.3745914), K. Maes and G. Lombaert, KU Leuven.
# Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), as stated
# on the record: teaching and research use only, which suits this course (candidate, 2026-11-18).
# Cite the record and the paper. No account or contact details are needed.
#
# Usage: bash scripts/fetch_kw51.sh   (from anywhere; about 200 MB; needs curl, md5sum)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/kw51"
api="https://zenodo.org/api/records/3745914/files"
mkdir -p "$raw"

# name md5 (from the record's file list)
files="
readme.txt 173d69dc896fa7894d78a4025ffdcdb5
matlab-functions.zip 517817fd0c6196e69eb9c3ebadebc155
trackedmodes.zip 5cdb446519d1150228fab51b3a002239
traindata_201810.zip f6550e8859ce6e0652bf385abb617b2d
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
echo "KW51 files in $raw: all MD5s match"
