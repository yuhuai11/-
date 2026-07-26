#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g14_domain_generalization/_raw/kielce_17_uav"
LOCK="$DEST/.pilot-download.lock"

mkdir -p "$DEST"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Kielce 17-UAV pilot download is already running; refusing a second writer." >&2
  exit 75
fi

download_and_verify() {
  local name="$1"
  local url="$2"
  local expected_bytes="$3"
  local expected_md5="$4"
  local final="$DEST/$name"
  local part="$final.part"

  verify() {
    local path="$1"
    [[ "$(stat -c %s "$path")" -eq "$expected_bytes" ]] &&
      printf '%s  %s\n' "$expected_md5" "$path" | md5sum --check --status
  }

  if [[ -f "$final" ]]; then
    if verify "$final"; then
      echo "Already complete: $final"
      sha256sum "$final"
      return
    fi
    echo "Refusing to overwrite invalid file: $final" >&2
    exit 1
  fi

  echo "$name partial bytes: $(stat -c %s "$part" 2>/dev/null || echo 0)"
  /usr/bin/curl \
    --fail \
    --location \
    --proto '=https' \
    --tlsv1.2 \
    --connect-timeout 30 \
    --speed-limit 10240 \
    --speed-time 300 \
    --retry 20 \
    --retry-delay 10 \
    --retry-all-errors \
    --continue-at - \
    --output "$part" \
    "$url"

  verify "$part"
  if [[ "$name" == *.zip ]]; then
    unzip -tq "$part" >/dev/null
  fi
  mv "$part" "$final"
  echo "Download complete; official size and MD5 verified: $final"
  sha256sum "$final"
}

download_and_verify \
  "dataset_files_information.pdf" \
  "https://zenodo.org/api/records/15190811/files/dataset_files_information.pdf/content" \
  761259 \
  "f1c7222066aafb0cb926d6ea1d43e5a1"

download_and_verify \
  "X4_D1_MATRICE300.zip" \
  "https://zenodo.org/api/records/15190811/files/X4_D1_MATRICE300.zip/content" \
  425563898 \
  "c6af83f75d91461cacffddcaddb69715"
