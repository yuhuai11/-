#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user1/JJZ/ABDDV-CRNN"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/anaconda3/envs/dads-crnn/bin/python}"
TARGET="$ROOT/data/g7_r6_new_sources/drone_noise_v3"
RAW="$TARGET/raw"
METADATA="$TARGET/metadata"
ARTICLE_ID="22133411"

mkdir -p "$RAW" "$METADATA"
curl -fsSL --retry 5 "https://api.figshare.com/v2/articles/$ARTICLE_ID" \
  -o "$METADATA/figshare_article_22133411.json"

"$PYTHON_BIN" - "$METADATA/figshare_article_22133411.json" "$METADATA/file_manifest.tsv" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
article = json.loads(source.read_text(encoding="utf-8"))
if article.get("doi") != "10.17866/rd.salford.22133411.v3":
    raise SystemExit("Unexpected DroneNoise DOI")
if article.get("license", {}).get("name") != "CC BY 4.0":
    raise SystemExit("Unexpected DroneNoise license")
lines = ["file_id\tname\tsize\tmd5\tdownload_url"]
for item in article["files"]:
    name = item["name"]
    if any(char in name for char in "\t\r\n/"):
        raise SystemExit(f"Unsafe Figshare file name: {name!r}")
    lines.append(
        f'{item["id"]}\t{name}\t{item["size"]}\t{item["supplied_md5"]}\t{item["download_url"]}'
    )
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

tail -n +2 "$METADATA/file_manifest.tsv" | while IFS=$'\t' read -r file_id name size md5 url; do
  destination="$RAW/$name"
  partial="$destination.part"
  if [[ -f "$destination" ]] && \
     [[ "$(stat -c '%s' "$destination")" == "$size" ]] && \
     [[ "$(md5sum "$destination" | cut -d' ' -f1)" == "$md5" ]]; then
    continue
  fi
  curl -fL --retry 5 --retry-delay 2 --continue-at - "$url" -o "$partial"
  [[ "$(stat -c '%s' "$partial")" == "$size" ]]
  [[ "$(md5sum "$partial" | cut -d' ' -f1)" == "$md5" ]]
  mv "$partial" "$destination"
done

"$PYTHON_BIN" - "$TARGET" <<'PY'
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.io import wavfile

root = Path(sys.argv[1])
manifest_path = root / "metadata/file_manifest.tsv"
rows = list(csv.DictReader(manifest_path.open(encoding="utf-8"), delimiter="\t"))
inventory = []
for row in rows:
    path = root / "raw" / row["name"]
    payload = path.read_bytes()
    if len(payload) != int(row["size"]):
        raise SystemExit(f"Size mismatch: {path}")
    if hashlib.md5(payload).hexdigest() != row["md5"]:
        raise SystemExit(f"MD5 mismatch: {path}")
    item = {
        **row,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "is_audio": path.suffix.lower() == ".wav",
    }
    if item["is_audio"]:
        sample_rate, samples = wavfile.read(path, mmap=True)
        frames = int(samples.shape[0])
        channels = 1 if samples.ndim == 1 else int(samples.shape[1])
        item.update(
            channels=channels,
            sample_rate=int(sample_rate),
            frames=frames,
            duration_seconds=frames / int(sample_rate),
            audio_dtype=str(samples.dtype),
        )
    inventory.append(item)

audit = {
    "passed": True,
    "protocol": "g7_r6_dronenoise_download_audit_v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "DroneNoise Database",
    "version": 3,
    "doi": "10.17866/rd.salford.22133411.v3",
    "license": "CC BY 4.0",
    "files": len(inventory),
    "wav_files": sum(item["is_audio"] for item in inventory),
    "bytes": sum(int(item["size"]) for item in inventory),
    "all_official_md5_verified": True,
    "training_status": "quarantined_pending_dads_overlap_and_session_split",
    "inventory": inventory,
}
(root / "metadata/download_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({key: audit[key] for key in (
    "passed", "files", "wav_files", "bytes", "all_official_md5_verified", "training_status"
)}, ensure_ascii=False, indent=2))
PY
