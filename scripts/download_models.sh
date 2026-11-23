#!/usr/bin/env bash
# Download the five MaLiang checkpoints from the GitHub release and verify them.
# Usage: bash scripts/download_models.sh [work_dir]
set -euo pipefail

REPO="${REPO:-YeyingLab/MaLiang}"
TAG="${TAG:-v1.0-models}"
WORK="${1:-.model_download}"
N="${N:-26}"

mkdir -p "$WORK"
for i in $(seq -w 0 $((N - 1))); do
  name="maliang-v1-models.tar.part-$i"
  if [ ! -f "$WORK/$name" ]; then
    echo "downloading $name"
    curl -L --fail --retry 3 --retry-delay 5 \
      -o "$WORK/$name" \
      "https://github.com/$REPO/releases/download/$TAG/$name"
  fi
done

cat "$WORK"/maliang-v1-models.tar.part-* > "$WORK/models.tar"
tar -xf "$WORK/models.tar" -C .
( cd models && sha256sum -c ../models.sha256 )
echo "checkpoints are in ./models"
