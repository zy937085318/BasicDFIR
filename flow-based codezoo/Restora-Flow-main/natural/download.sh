#!/bin/bash
#
# Download pretrained checkpoints for Restora-Flow
# Usage:
#   bash download.sh <model-name>
#
# Available model names:
#   celeba-ot
#   celeba-ddpm
#   afhq-cat-ot
#   afhq-cat-ddpm
#   coco-ot
#   coco-ddpm
#

set -e

# Helpers

download_gdrive() {
    FILE_ID=$1
    OUT_PATH=$2

    # Create directory if it doesn't exist
    OUT_DIR=$(dirname "$OUT_PATH")
    mkdir -p "$OUT_DIR"

    echo "Downloading to: $OUT_PATH"
    gdown --id "$FILE_ID" -O "$OUT_PATH"
    echo "Done."
}

# Models

case "$1" in

    # CelebA
    "celeba-ot")
        download_gdrive "1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6" \
            "model_checkpoints/celeba/gaussian/ot/model_final.pt"
        ;;

    "celeba-ddpm")
        download_gdrive "1jsXcLapZK_KcNpGRlGy6qjuhjzCqXPig" \
            "model_checkpoints/celeba/gaussian/ddpm/model_final.pt"
        ;;

    # AFHQ-Cat
    "afhq-cat-ot")
        download_gdrive "1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF" \
            "model_checkpoints/afhq_cat/gaussian/ot/model_final.pt"
        ;;

    "afhq-cat-ddpm")
        download_gdrive "10wavvPjunjSwyM73FtHiuQ6fAw5Pm0T-" \
            "model_checkpoints/afhq_cat/gaussian/ddpm/model_final.pt"
        ;;

    # COCO
    "coco-ot")
        download_gdrive "1OZvWV6X2wpNjX7A9HjNNsqogRzcbCsFg" \
            "model_checkpoints/coco/gaussian/ot/model_final.pt"
        ;;

    "coco-ddpm")
        download_gdrive "1MNdOmEaH40OLQzFSmmjQVXqUBZ3uZB86" \
            "model_checkpoints/coco/gaussian/ddpm/model_final.pt"
        ;;

    # Help/Unknown
    *)
        echo "Unknown or missing argument: $1"
        echo ""
        echo "Usage:"
        echo "  bash download.sh <model-name>"
        echo ""
        echo "Available model names:"
        echo "  celeba-ot"
        echo "  celeba-ddpm"
        echo "  afhq-cat-ot"
        echo "  afhq-cat-ddpm"
        echo "  coco-ot"
        echo "  coco-ddpm"
        exit 1
        ;;
esac
