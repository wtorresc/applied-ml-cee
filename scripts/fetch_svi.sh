#!/usr/bin/env bash
# Download the CDC/ATSDR Social Vulnerability Index 2022, Illinois counties, for CEE Lecture 27.
#
#   data/raw/svi/SVI_2022_Illinois_county.csv   the CSV as published (read-only; never edit)
#
# Source: Centers for Disease Control and Prevention / Agency for Toxic Substances and Disease
# Registry / Geospatial Research, Analysis, and Services Program. CDC/ATSDR Social Vulnerability
# Index 2022 Database Illinois. https://www.atsdr.cdc.gov/place-health/php/svi/
# svi-data-documentation-download.html (the page loads its links dynamically; the file is served
# from svi.cdc.gov). Variables: SVI 2022 Documentation (October 2024), on the same page. The
# 2022 MP_CROWD values were corrected on 2024-12-11 (the page's "Data correction" note); this
# file's Last-Modified header on 2026-11-20 was 2024-11-25, so it may predate the correction. The
# MP_ columns are margins of error; Lecture 27 reads only RPL_THEMES, which the SVI builds from
# the EP_ estimates, so the correction should not affect it (not checked against the SVI
# documentation).
#
# Terms: CDC's "Use of Agency Materials" (https://www.cdc.gov/other/agencymaterials.html): most
# CDC/ATSDR information is in the public domain and may be freely used or reproduced. Its
# conditions: attribute the agency ("Source: CDC/ATSDR"); state that the use does not imply
# endorsement by CDC, ATSDR, HHS, or the U.S. Government; do not change the substantive content;
# and say the material is available on the agency's website at no charge. The lectures cite it so.
#
# Size: about 75 kB, 102 rows (one per county) and 158 columns.
#
# Check: the MD5 of the copy fetched 2026-11-20; a revised file stops the script.
#
# Usage: bash scripts/fetch_svi.sh   (from anywhere; needs curl, md5sum)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/svi"
mkdir -p "$raw"
file="$raw/SVI_2022_Illinois_county.csv"
md5="086cbd6e408163acd05be767861fc66e"  # the copy fetched 2026-11-20
md5_check() {  # md5_check FILE: stop unless FILE has the recorded MD5
  echo "$md5  $1" | md5sum --check --quiet || {
    echo "$1: MD5 differs from the copy fetched 2026-11-20; CDC may have revised it" >&2
    exit 1
  }
}
if [[ ! -f "$file" ]]; then
  curl -sSfL --retry 3 -o "$file.part" \
    "https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/Illinois_county.csv"
  md5_check "$file.part"
  mv "$file.part" "$file"
  chmod a-w "$file"
fi
md5_check "$file"
rows=$(($(wc -l < "$file") - 1))
if [[ "$rows" -ne 102 ]]; then
  echo "expected 102 Illinois counties, found $rows rows in $file" >&2
  exit 1
fi
echo "ok: $file ($rows counties)"
