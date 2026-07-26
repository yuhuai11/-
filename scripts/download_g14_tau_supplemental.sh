#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g14_domain_generalization/_raw/tau_urban_2022"
BASE_URL="https://zenodo.org/records/6337421/files"

PARTS=(1 9)
EXPECTED_BYTES=(1736050960 1790044829)
EXPECTED_MD5=(
  8be14bdbd844481dce059ab31fbd2239
  071f664c86639ad730f7be4e3c39d886
)

mkdir -p "$DEST"

for index in "${!PARTS[@]}"; do
  part="${PARTS[$index]}"
  expected_bytes="${EXPECTED_BYTES[$index]}"
  expected_md5="${EXPECTED_MD5[$index]}"
  name="TAU-urban-acoustic-scenes-2022-mobile-development.audio.${part}.zip"
  final="$DEST/$name"
  partial="$final.part"
  lock="$DEST/.audio-${part}-download.lock"
  url="$BASE_URL/$name?download=1"

  exec {lock_fd}>"$lock"
  if ! flock -n "$lock_fd"; then
    echo "TAU part $part download is already running; refusing a second writer." >&2
    exit 75
  fi

  verify() {
    local path="$1"
    [[ "$(stat -c %s "$path")" -eq "$expected_bytes" ]] &&
      printf '%s  %s\n' "$expected_md5" "$path" | md5sum --check --status &&
      unzip -tq "$path" >/dev/null
  }

  if [[ -f "$final" ]]; then
    if verify "$final"; then
      echo "TAU part $part is already complete: $final"
      sha256sum "$final"
      exec {lock_fd}>&-
      continue
    fi
    echo "Refusing to overwrite invalid TAU part $part: $final" >&2
    exit 1
  fi

  echo "TAU part $part partial bytes: $(stat -c %s "$partial" 2>/dev/null || echo 0)"
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
    --output "$partial" \
    "$url"

  verify "$partial"
  mv "$partial" "$final"
  echo "TAU part $part complete; size, MD5 and ZIP verified: $final"
  sha256sum "$final"
  exec {lock_fd}>&-
done
