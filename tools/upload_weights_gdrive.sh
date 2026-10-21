#!/usr/bin/env bash
# Upload models/ and checkpoints/ to Google Drive with rclone, then verify by checksum.
# Usage: tools/upload_weights_gdrive.sh [remote] [drive_folder]
set -euo pipefail
REMOTE=${1:-gdrive}
FOLDER=${2:-dualvln-vla-e2e-20261021-rc3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RCLONE=${RCLONE:-$(command -v rclone || echo "$HOME/.local/bin/rclone")}

for dir in models checkpoints; do
  "$RCLONE" copy "$ROOT/$dir" "$REMOTE:$FOLDER/$dir" \
    --transfers 4 --drive-chunk-size 256M --retries 10 --low-level-retries 20 \
    --stats 30s --stats-one-line -v
done
"$RCLONE" copyto "$ROOT/SHA256SUMS" "$REMOTE:$FOLDER/SHA256SUMS"
# Drive stores MD5; compare sizes and hashes of every uploaded file.
for dir in models checkpoints; do
  "$RCLONE" check "$ROOT/$dir" "$REMOTE:$FOLDER/$dir" --one-way
done
echo "upload verified: $REMOTE:$FOLDER"
