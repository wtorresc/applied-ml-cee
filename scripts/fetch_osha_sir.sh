#!/usr/bin/env bash
# Fetch OSHA's Severe Injury Reports (SIR): every severe injury (amputation, loss of an eye, or
# in-patient hospitalization) employers reported to federal OSHA since 1 January 2015, one row per
# report with a free-text description of the incident. For CEE Lecture 17 (tokens and word
# vectors) and Lecture 19 (language models and retrieval).
#
#   data/raw/osha_sir/January2015toNovember2025.zip   the file as published (read-only)
#
# Source: https://www.osha.gov/severe-injury-reports ("Download the full current SIR data set",
# data from 1/1/2015 through 11/30/2025; federal OSHA jurisdiction only, no state-plan states, no
# fatalities). Terms: U.S. federal government data, publicly available ("Employers, workers,
# safety professionals, and other interested parties may view and download OSHA's publicly
# available data provided online", osha.gov/data); no licence is stated, and works of the U.S.
# government are not subject to copyright. Records name the establishment and its address.
#
# osha.gov refuses scripted downloads (HTTP 403 to curl, 2026-11-18), so save the file with a
# browser, then pass its path here: the script copies it into data/raw/osha_sir, checks the MD5
# recorded below, and makes it read-only. If OSHA replaces the file, the MD5 check fails: record
# the new file's MD5 only after looking at what changed.
#
# Usage: bash scripts/fetch_osha_sir.sh ~/Downloads/January2015toNovember2025.zip
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/osha_sir"
name="January2015toNovember2025.zip"
url="https://www.osha.gov/sites/default/files/$name"
md5="380ad54957d83409f3a13fc56c02c96d"  # the copy saved 2026-11-18 (CSV dated 2026-07-23)
out="$raw/$name"
mkdir -p "$raw"

if [[ ! -f "$out" ]]; then
  src="${1:-}"
  if [[ -z "$src" || ! -f "$src" ]]; then
    echo "Save $url with a browser, then run: bash scripts/fetch_osha_sir.sh <saved file>" >&2
    exit 1
  fi
  cp "$src" "$out.part"
  if [[ -n "$md5" ]]; then
    echo "$md5  $out.part" | md5sum --check --quiet || {
      echo "$out.part: MD5 differs from the recorded one; OSHA may have replaced the file" >&2
      exit 1
    }
  fi
  unzip -tq "$out.part" > /dev/null || { echo "$out.part is not a valid ZIP" >&2; exit 1; }
  mv "$out.part" "$out"
fi
if [[ -n "$md5" ]]; then
  echo "$md5  $out" | md5sum --check --quiet
else
  echo "MD5 not yet recorded in this script: $(md5sum "$out" | cut -d' ' -f1)" >&2
fi
chmod a-w "$out"
unzip -l "$out"
