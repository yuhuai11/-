#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g14_domain_generalization/_raw/tau_urban_2022"
FINAL="$DEST/TAU-urban-acoustic-scenes-2022-mobile-development.audio.16.zip"
PART="$FINAL.part"
LOCK="$DEST/.pilot-download.lock"
URL="https://zenodo.org/records/6337421/files/TAU-urban-acoustic-scenes-2022-mobile-development.audio.16.zip?download=1"
EXPECTED_BYTES=435379083
EXPECTED_MD5="6e2df8438c69f6789414aeba8cbad9d8"

mkdir -p "$DEST"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "TAU pilot download is already running; refusing a second writer." >&2
  exit 75
fi

verify() {
  local path="$1"
  [[ "$(stat -c %s "$path")" -eq "$EXPECTED_BYTES" ]] &&
    printf '%s  %s\n' "$EXPECTED_MD5" "$path" | md5sum --check --status &&
    unzip -tq "$path" >/dev/null
}

if [[ -f "$FINAL" ]]; then
  if verify "$FINAL"; then
    echo "TAU pilot archive is already complete: $FINAL"
    sha256sum "$FINAL"
    exit 0
  fi
  echo "Refusing to overwrite invalid TAU pilot archive: $FINAL" >&2
  exit 1
fi

echo "TAU pilot partial bytes: $(stat -c %s "$PART" 2>/dev/null || echo 0)"
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
  --output "$PART" \
  "$URL"

verify "$PART"
mv "$PART" "$FINAL"
echo "TAU pilot download complete; size, MD5 and ZIP verified: $FINAL"
sha256sum "$FINAL"
