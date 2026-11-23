#!/usr/bin/env bash
# Download the BattLeDIM data (Battle of the Leakage Detection and Isolation Methods): SCADA
# readings of the L-Town water distribution network, 2018 and 2019, with the leaks that occurred
# (Vrachimis et al., 2022, J. Water Resour. Plann. Manage. 148(12), 04022068,
# doi:10.1061/(ASCE)WR.1943-5452.0001601). For CEE Lecture 9 (outliers, robust estimation, leak
# detection). Files are semicolon-separated with decimal commas; timestamps every 5 minutes.
#
#   data/raw/battledim/        the files as published (read-only; never edit), about 190 MB
#     README.txt, dataset_configuration.yaml   the authors' description of the data
#     L-TOWN.inp               the EPANET model of the network (WNTR reads it)
#     <year>_SCADA_Pressures.csv, _Flows.csv, _Levels.csv, _Demands.csv
#                              5-minute readings: 33 pressure sensors (m), 3 flows (m3/h),
#                              one tank level (m), 82 automated demand meters (L/h), for 2018
#                              and 2019
#     <year>_Leakages.csv      leak outflow (m3/h) at every pipe that leaked (14 in 2018, 19
#                              more in 2019); the configuration file lists start, end, size, type
#     2018_Fixed_Leakages_Report.txt           repair dates of the 2018 leaks
#   Left out: the two Excel workbooks (the same readings) and L-TOWN_Real.inp (177 MB, the
#   model that generated the data, with the true parameters).
#
# Source: Zenodo record 4017659, doi:10.5281/zenodo.4017659. Each file is checked against the MD5
# checksum the Zenodo API publishes.
#
# Licence: Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the Zenodo record. Cite
# Vrachimis et al. (2022). No account or contact details are needed.
#
# Usage: bash scripts/fetch_battledim.sh   (from anywhere; needs curl and md5sum)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/battledim"
base="https://zenodo.org/api/records/4017659/files"
mkdir -p "$raw"

files=(
  "README.txt 7ad5cdc2eebfe78197a3412e80d07ddc"
  "dataset_configuration.yaml 48486401f5b4d0447023f5ce5d242c52"
  "L-TOWN.inp bfbb16b8b463e0576200ba6ad360f68e"
  "2018_SCADA_Pressures.csv d389d8541350c19ff0bfc6b80f246d35"
  "2018_SCADA_Flows.csv d0602c06946b46287e956f007e4264ee"
  "2018_SCADA_Levels.csv 92a512d6749ddf66fd6d99f2bdde6cc4"
  "2018_SCADA_Demands.csv 8803afcb1da2428de169237be3d911fb"
  "2018_Leakages.csv c2c5fab90420da44f02e29775050abfe"
  "2018_Fixed_Leakages_Report.txt 50b1f8b7ba0db5a19cedbb060c63d912"
  "2019_SCADA_Pressures.csv 5ea1e46d3f2f0a89a3f98d6fd39a851d"
  "2019_SCADA_Flows.csv 28fc99fdcbf80fcd26079e7fe602d6dc"
  "2019_SCADA_Levels.csv e5a8050bd38729e4b7648b9d52abd5b1"
  "2019_SCADA_Demands.csv b3e111de397a1b06d33b8a9f13bc997d"
  "2019_Leakages.csv e1f0a43683813a90a9fec8562dde599b"
)
for entry in "${files[@]}"; do
  read -r name md5 <<< "$entry"
  out="$raw/$name"
  if [[ ! -f "$out" ]]; then
    curl -sSfL -o "$out.part" "$base/$name/content"
    echo "$md5  $out.part" | md5sum --check --quiet
    mv "$out.part" "$out"
    chmod a-w "$out"
  fi
done
echo "BattLeDIM files in $raw"
