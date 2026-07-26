#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/g14_domain_generalization/_raw/tau_urban_2022"
FINAL="$DEST/TAU-urban-acoustic-scenes-2022-mobile-development.meta.zip"
PART="$FINAL.part"
LOCK="$DEST/.metadata-download.lock"
URL="https://zenodo.org/records/6337421/files/TAU-urban-acoustic-scenes-2022-mobile-development.meta.zip?download=1"
EXPECTED_MD5="419b6ff6570f1030730352dc80cd8d15"

mkdir -p "$DEST"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "TAU metadata download is already running; refusing a second writer." >&2
  exit 75
fi

verify() {
  printf '%s  %s\n' "$EXPECTED_MD5" "$1" | md5sum --check --status
}

if [[ -f "$FINAL" ]]; then
  if verify "$FINAL" && unzip -tq "$FINAL" >/dev/null; then
    echo "TAU metadata is already complete: $FINAL"
    sha256sum "$FINAL"
    exit 0
  fi
  echo "Refusing to overwrite invalid TAU metadata: $FINAL" >&2
  exit 1
fi

/usr/bin/curl \
  --fail \
  --location \
  --proto '=https' \
  --tlsv1.2 \
  --connect-timeout 30 \
  --speed-limit 1024 \
  --speed-time 120 \
  --retry 20 \
  --retry-delay 10 \
  --retry-all-errors \
  --continue-at - \
  --output "$PART" \
  "$URL"

verify "$PART"
unzip -tq "$PART" >/dev/null
mv "$PART" "$FINAL"
echo "TAU metadata download complete and MD5-verified: $FINAL"
sha256sum "$FINAL"
