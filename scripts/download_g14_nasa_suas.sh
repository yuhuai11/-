#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g14_domain_generalization/_raw/nasa_suas"
FINAL="$DEST/small_uav_acoustics.zip"
PART="$FINAL.part"
LOCK="$DEST/.download.lock"
URL="https://data.nasa.gov/docs/datasets/rfk401li/small_uav_acoustics.zip"
EXPECTED_BYTES=1693926297

mkdir -p "$DEST"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "NASA sUAS download is already running; refusing a second writer." >&2
  exit 75
fi

if [[ -f "$FINAL" ]]; then
  observed="$(stat -c %s "$FINAL")"
  if [[ "$observed" -eq "$EXPECTED_BYTES" ]] && unzip -tq "$FINAL" >/dev/null; then
    echo "NASA sUAS archive is already complete: $FINAL"
    sha256sum "$FINAL"
    exit 0
  fi
  echo "Refusing to overwrite an invalid final archive: $FINAL" >&2
  exit 1
fi

echo "NASA sUAS partial bytes: $(stat -c %s "$PART" 2>/dev/null || echo 0)"
/usr/local/anaconda3/bin/curl \
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

observed="$(stat -c %s "$PART")"
if [[ "$observed" -ne "$EXPECTED_BYTES" ]]; then
  echo "NASA sUAS archive size mismatch: expected=$EXPECTED_BYTES observed=$observed" >&2
  exit 1
fi
unzip -tq "$PART" >/dev/null
mv "$PART" "$FINAL"
echo "NASA sUAS download complete and ZIP-tested: $FINAL"
sha256sum "$FINAL"
