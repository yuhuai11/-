#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
DEST_DIR="$ROOT/data/external_confirmation_v2/_raw/AeroSonicDB_v1.1.2"
BASE_URL="https://zenodo.org/records/10215080/files"
LOCK="$DEST_DIR/.download.lock"

mkdir -p "$DEST_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "AeroSonicDB download is already running; refusing a second writer." >&2
  exit 75
fi

download_one() {
  local name="$1"
  local expected_md5="$2"
  local final="$DEST_DIR/$name"
  local part="$final.part"

  if [[ -f "$final" ]]; then
    if printf '%s  %s\n' "$expected_md5" "$final" | md5sum --check --status; then
      echo "Already verified: $final"
      return 0
    fi
    echo "Refusing to overwrite invalid final file: $final" >&2
    return 1
  fi

  echo "Downloading $name; existing partial bytes: $(stat -c %s "$part" 2>/dev/null || echo 0)"
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
    --output "$part" \
    "$BASE_URL/$name?download=1"
  local curl_status=$?

  if [[ -f "$part" ]] && printf '%s  %s\n' "$expected_md5" "$part" | md5sum --check --status; then
    mv "$part" "$final"
    echo "Verified: $final ($expected_md5)"
    return 0
  fi
  if [[ $curl_status -eq 0 ]]; then
    echo "Transfer completed but MD5 failed: $part" >&2
  else
    echo "curl failed with status $curl_status: $part" >&2
  fi
  return 1
}

download_one "sample_meta.csv" "aabd99b1b2efe0895e212232bca07e46" || exit $?
download_one "audio.zip" "77605a8ef12a38a289b63ae3457d326e" || exit $?
echo "AeroSonicDB v1.1.2 required files downloaded and MD5-verified."
