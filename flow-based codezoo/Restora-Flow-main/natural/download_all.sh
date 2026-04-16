#!/bin/bash
#
# Download all pretrained checkpoints for Restora-Flow
# This script checks for gdown installation, downloads all models, and places them in the correct folders
#

set -e

# Check for gdown
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Installing..."
    pip install gdown
fi

# Helper
download_gdrive() {
    FILE_ID=$1
    OUT_PATH=$2

    if [ -f "$OUT_PATH" ]; then
        echo "✓ File already exists: $OUT_PATH"
    else
        echo "→ Downloading to: $OUT_PATH"
        mkdir -p $(dirname "$OUT_PATH")
        gdown --id "$FILE_ID" -O "$OUT_PATH"
        echo "✓ Done."
    fi
}

# Pretrained Checkpoints:

# CelebA
download_gdrive "1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6" \
    "model_checkpoints/celeba/gaussian/ot/model_final.pt"
download_gdrive "1jsXcLapZK_KcNpGRlGy6qjuhjzCqXPig" \
    "model_checkpoints/celeba/gaussian/ddpm/model_final.pt"

# AFHQ-Cat
download_gdrive "1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF" \
    "model_checkpoints/afhq_cat/gaussian/ot/model_final.pt"
download_gdrive "10wavvPjunjSwyM73FtHiuQ6fAw5Pm0T-" \
    "model_checkpoints/afhq_cat/gaussian/ddpm/model_final.pt"

# COCO
download_gdrive "1OZvWV6X2wpNjX7A9HjNNsqogRzcbCsFg" \
    "model_checkpoints/coco/gaussian/ot/model_final.pt"
download_gdrive "1MNdOmEaH40OLQzFSmmjQVXqUBZ3uZB86" \
    "model_checkpoints/coco/gaussian/ddpm/model_final.pt"

echo "All pretrained checkpoints downloaded successfully!"
