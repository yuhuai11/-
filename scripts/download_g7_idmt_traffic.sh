#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g7_cross_domain/_raw/idmt_traffic"
FINAL="$DEST/IDMT_Traffic.zip"
PART="$FINAL.part"
METADATA="$DEST/zenodo_record_7551553.json"
LOCK="$DEST/.download.lock"
URL="https://zenodo.org/api/records/7551553/files/IDMT_Traffic.zip/content"
METADATA_URL="https://zenodo.org/api/records/7551553"
EXPECTED_BYTES=9663672447
EXPECTED_MD5="7ca2311ca32203aec5074a5fe933e343"
MIN_FREE_BYTES=15000000000

mkdir -p "$DEST"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "IDMT-TRAFFIC download is already running; refusing a second writer." >&2
  exit 75
fi

verify_file() {
  local path="$1"
  [[ "$(stat -c %s "$path")" -eq "$EXPECTED_BYTES" ]] &&
    printf '%s  %s\n' "$EXPECTED_MD5" "$path" | md5sum --check --status
}

if [[ -f "$FINAL" ]]; then
  if verify_file "$FINAL"; then
    echo "IDMT-TRAFFIC is already complete and MD5-verified: $FINAL"
    sha256sum "$FINAL"
    exit 0
  fi
  echo "Refusing to overwrite an invalid final archive: $FINAL" >&2
  exit 1
fi

available_bytes="$(df --output=avail -B1 "$DEST" | tail -n 1 | tr -d ' ')"
remaining_bytes=$((EXPECTED_BYTES - $(stat -c %s "$PART" 2>/dev/null || echo 0)))
required_bytes="$remaining_bytes"
if [[ "$required_bytes" -lt "$MIN_FREE_BYTES" ]]; then
  required_bytes="$MIN_FREE_BYTES"
fi
if [[ "$available_bytes" -lt "$required_bytes" ]]; then
  echo "Insufficient free disk space: available=$available_bytes required=$required_bytes" >&2
  exit 1
fi

curl -fsSL \
  --proto '=https' \
  --tlsv1.2 \
  --connect-timeout 30 \
  --retry 20 \
  --retry-delay 10 \
  --retry-all-errors \
  --output "$METADATA" \
  "$METADATA_URL"

echo "IDMT-TRAFFIC partial bytes: $(stat -c %s "$PART" 2>/dev/null || echo 0)"
curl -fsSL \
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

if ! verify_file "$PART"; then
  echo "IDMT-TRAFFIC size or MD5 verification failed; partial file retained: $PART" >&2
  exit 1
fi

mv "$PART" "$FINAL"
echo "IDMT-TRAFFIC download complete and MD5-verified: $FINAL"
sha256sum "$FINAL"
