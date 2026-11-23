#!/usr/bin/env bash
# Fetch the Manual on Uniform Traffic Control Devices (MUTCD), 11th Edition (December 2023),
# FHWA: the national standard for signs, markings, and signals on U.S. roads. For CEE Lecture 19
# (embeddings and retrieval-augmented question answering over a standard).
#
#   data/raw/mutcd/mutcd11thedition.pdf   the file as published (read-only; never edit)
#
# Source: https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/mutcd11thedition.pdf (last modified
# 27 December 2023 on the server). Terms: a work of the U.S. federal government (23 CFR 655,
# Subpart F adopts it), not subject to copyright. No account or contact details are needed.
#
# Usage: bash scripts/fetch_mutcd.sh   (from anywhere; about 24 MB; needs curl, md5sum)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/mutcd"
url="https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/mutcd11thedition.pdf"
md5="274cbf747d5bcca19934281a3a84b515"  # the copy fetched 2026-11-18
out="$raw/mutcd11thedition.pdf"
mkdir -p "$raw"

if [[ ! -f "$out" ]]; then
  curl -sSfL --retry 3 -o "$out.part" "$url"
  if [[ -n "$md5" ]]; then
    echo "$md5  $out.part" | md5sum --check --quiet || {
      echo "$out.part: MD5 differs from the recorded one; FHWA may have revised it" >&2
      exit 1
    }
  fi
  mv "$out.part" "$out"
fi
if [[ -n "$md5" ]]; then
  echo "$md5  $out" | md5sum --check --quiet
else
  echo "MD5 not yet recorded in this script: $(md5sum "$out" | cut -d' ' -f1)" >&2
fi
chmod a-w "$out"
echo "MUTCD in $raw"
