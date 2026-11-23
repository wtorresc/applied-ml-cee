#!/usr/bin/env bash
# Download the UCI Concrete Compressive Strength data (Yeh, 1998; CC BY 4.0) and convert it to CSV.
#
#   data/raw/uci_concrete/     the archive as published (read-only; never edit)
#   data/derived/Concrete_Data.csv   the same table as CSV (LibreOffice conversion of the .xls)
#
# Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
# 1,030 tests of 428 distinct mixes (the same mix is tested at several ages); 34 rows are exact
# duplicates. Columns: seven ingredients in kg per m^3 of concrete, age in days, strength in MPa.
#
# Usage: bash scripts/fetch_concrete.sh   (from the repository root)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/uci_concrete"
derived="$root/data/derived"
mkdir -p "$raw" "$derived"

if [[ ! -f "$raw/Concrete_Data.xls" ]]; then
  curl -sSfL -o "$raw/concrete_compressive_strength.zip.part" \
    "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
  mv "$raw/concrete_compressive_strength.zip.part" "$raw/concrete_compressive_strength.zip"
  unzip -q "$raw/concrete_compressive_strength.zip" -d "$raw"
  chmod a-w "$raw"/*
fi
# The spreadsheet as published on 2023-05-22 (checked on every run, not only after a download).
echo "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4  $raw/Concrete_Data.xls" \
  | sha256sum --check --quiet

# Converted beside (same file system), checked, then renamed: a failed or interrupted run never
# leaves a truncated table in data/derived.
profile="$(mktemp -d)"
part="$(mktemp -d "$derived/.concrete.XXXXXX")"
trap 'rm -r -- "$profile" "$part"' EXIT
soffice -env:UserInstallation="file://$profile" --headless --convert-to csv \
  --outdir "$part" "$raw/Concrete_Data.xls" > /dev/null

rows=$(($(wc -l < "$part/Concrete_Data.csv") - 1))
if [[ "$rows" -ne 1030 ]]; then
  echo "expected 1030 rows, found $rows" >&2
  exit 1
fi
mv "$part/Concrete_Data.csv" "$derived/Concrete_Data.csv"
echo "wrote $derived/Concrete_Data.csv ($rows rows)"
