#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
DEST_DIR="$ROOT/data/external_confirmation_v2/_raw/DDL"
FINAL="$DEST_DIR/MLSP_2022_Real_Data.zip"
PART="$FINAL.part"
LOCK="$DEST_DIR/.download.lock"
URL="https://zenodo.org/records/6459183/files/MLSP_2022_Real_Data.zip?download=1"
EXPECTED_MD5="4a6d4da4e1c732550c1ccd8d29dd16f8"

mkdir -p "$DEST_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "DDL Real Data download is already running; refusing a second writer." >&2
  exit 75
fi

verify_file() {
  local path="$1"
  printf '%s  %s\n' "$EXPECTED_MD5" "$path" | md5sum --check --status
}

if [[ -f "$FINAL" ]]; then
  if verify_file "$FINAL"; then
    echo "DDL Real Data is already complete and MD5-verified: $FINAL"
    exit 0
  fi
  echo "Refusing to overwrite an invalid final archive: $FINAL" >&2
  exit 1
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "DDL curl attempt $attempt; existing partial bytes: $(stat -c %s "$PART" 2>/dev/null || echo 0)"

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
  curl_status=$?

  if [[ -f "$PART" ]] && verify_file "$PART"; then
    mv "$PART" "$FINAL"
    echo "DDL Real Data download complete; MD5 verified: $EXPECTED_MD5"
    echo "Final archive: $FINAL"
    exit 0
  fi

  if [[ $curl_status -eq 0 ]]; then
    echo "Server reported a complete transfer, but MD5 verification failed." >&2
    echo "Partial file retained for diagnosis: $PART" >&2
    exit 1
  fi

  echo "curl exited with status $curl_status; retrying the validated partial file in 20 seconds." >&2
  sleep 20
done
