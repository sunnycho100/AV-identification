#!/bin/bash
# Push what the server needs for fine-tuning, and nothing else.
#
# Home directories on the CEE server are world-readable by the whole lab, so the
# default is to send the minimum that makes training work:
#   - code
#   - the extracted frames and their calibration
#   - the pseudo-labels
#
# Deliberately NOT sent:
#   personal-documents/   standing rule, never leaves this machine
#   Camera data/*.csv     GPS ground truth. Training never touches it and
#                         evaluation stays local, so there is no reason to put
#                         held-out truth on a shared filesystem.
#   Camera data/*.mp4     source video, large and not needed (frames are synced)
#   checkpoints/          917 MB; fetch it on the server from the public URL
#                         instead (see docs/server-finetune-setup.md)
#
# Usage:  ./scripts/sync_to_server.sh [--dry-run]
set -euo pipefail

REMOTE=cee
DEST=~/roadside-camera/BEVHeights
DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

cd "$(dirname "$0")/.."

rsync -avz --progress $DRY \
  --exclude 'personal-documents/' \
  --exclude 'Camera data/' \
  --exclude 'V2X-Raw-Datasets/' \
  --exclude '.venv*' \
  --exclude 'checkpoints/' \
  --exclude 'third_party/' \
  --exclude 'archive/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  --exclude 'outputs/object_detection/' \
  --exclude 'outputs/reports/' \
  --exclude 'data/dair-v2x-i/' \
  ./ "$REMOTE:$DEST/"

if [[ -z "$DRY" ]]; then
  echo
  echo "verifying nothing private landed on the server..."
  ssh "$REMOTE" "
    bad=0
    for p in personal-documents 'Camera data'; do
      if [ -e \"$DEST/\$p\" ]; then echo \"PRESENT ON SERVER: \$p\"; bad=1; fi
    done
    if find $DEST -name '*_trajectory.csv' -print -quit | grep -q .; then
      echo 'PRESENT ON SERVER: GPS trajectory csv'; bad=1
    fi
    [ \$bad -eq 0 ] && echo 'clean: no personal-documents, no GPS ground truth' || exit 1
  "
fi
