#!/usr/bin/env bash
# Download the grid data of CEE Lecture 28 (AI as infrastructure): emission rates and hourly
# generation by fuel.
#
#   data/raw/grid/egrid2023_data_rev2.xlsx            EPA eGRID2023, revision 2 (June 2025)
#   data/raw/grid/EIA930_BALANCE_2024_Jan_Jun.csv     EIA-930 hourly balance, 2024, first half
#   data/raw/grid/EIA930_BALANCE_2024_Jul_Dec.csv     EIA-930 hourly balance, 2024, second half
#   (all read-only; never edit)
#
# Sources:
#   - U.S. EPA, Emissions & Generation Resource Integrated Database (eGRID2023), released January
#     2025, revision 2 June 2025: https://www.epa.gov/egrid/detailed-data. Annual 2023 generation
#     and emissions by plant, balancing authority, subregion, and state. A U.S. government work.
#   - U.S. Energy Information Administration, Form EIA-930, Hourly and Daily Balancing Authority
#     Operations Report, six-month files from the Hourly Electric Grid Monitor
#     (https://www.eia.gov/electricity/gridmonitor/): hourly demand, net generation, and net
#     generation by energy source for each balancing authority of the Lower 48.
#
# Terms: both are U.S. government works in the public domain. EIA asks for an acknowledgment that
# includes the publication date ("Source: U.S. Energy Information Administration (date)"), per
# https://www.eia.gov/about/copyrights_reuse.php; the lectures cite both so.
#
# Size: about 21 MB (eGRID) and 90 MB (EIA-930, both halves).
#
# Checks: each file's MD5 is the copy fetched 2026-11-20. EIA revises its six-month files, so a
# new download may differ: the script then stops, and the lecture's numbers need a rerun before
# the new file replaces the old.
#
# Usage: bash scripts/fetch_grid.sh   (from anywhere; needs curl, md5sum, and LibreOffice's soffice)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/grid"
mkdir -p "$raw"
md5_check() {  # md5_check MD5 FILE: stop unless FILE has the recorded MD5
  echo "$1  $2" | md5sum --check --quiet || {
    echo "$2: MD5 differs from the copy fetched 2026-11-20; the source may have been revised" >&2
    exit 1
  }
}
fetch() {  # fetch URL FILE MD5: download once, check the MD5, then make it read-only
  local url="$1" file="$raw/$2" md5="$3"
  if [[ ! -f "$file" ]]; then
    curl -sSfL --retry 3 -o "$file.part" "$url"
    md5_check "$md5" "$file.part"
    mv "$file.part" "$file"
    chmod a-w "$file"
  fi
  md5_check "$md5" "$file"
  echo "ok: $file ($(wc -c < "$file") bytes)"
}
fetch "https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx" \
  egrid2023_data_rev2.xlsx 2449e13755e410b94b46d20b9ca976be
eia="https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
fetch "$eia/EIA930_BALANCE_2024_Jan_Jun.csv" EIA930_BALANCE_2024_Jan_Jun.csv \
  75669ae2c4b02d635930e815510e5059
fetch "$eia/EIA930_BALANCE_2024_Jul_Dec.csv" EIA930_BALANCE_2024_Jul_Dec.csv \
  545f301108bbfd933e724a08de948651
# Derived: the eGRID sheets the lecture reads, as CSV (LibreOffice, which the deck build already
# needs; it exports every sheet, and only these three are kept: balancing authorities, states,
# plants). Rows 1-2 are eGRID's two header rows (a description, then the field codes).
derived="$root/data/derived/egrid2023"
if [[ ! -f "$derived/PLNT23.csv" ]]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:?}"' EXIT
  soffice -env:UserInstallation="file://$tmp/profile" --headless \
    --convert-to 'csv:Text - txt - csv (StarCalc):44,34,UTF8,1,,0,false,true,false,false,false,-1' \
    --outdir "$tmp" "$raw/egrid2023_data_rev2.xlsx" > /dev/null
  mkdir -p "$derived"
  for sheet in BA23 ST23 PLNT23; do
    cp "$tmp/egrid2023_data_rev2-$sheet.csv" "$derived/$sheet.csv"
  done
fi
echo "ok: $derived ($(ls "$derived" | tr '\n' ' '))"
