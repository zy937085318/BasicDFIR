#!/bin/bash
#
# Download pretrained checkpoints for Restora-Flow
# Usage:
#   bash download.sh <model-name>
#
# Available model names:
#   xray-hand-flow
#   xray-hand-ddpm
#

set -e

# Helpers

download_gdrive() {
    FILE_ID=$1
    OUT_PATH=$2

    echo "Downloading to: $OUT_PATH"
    mkdir -p "$(dirname "$OUT_PATH")"
    gdown --id "$FILE_ID" -O "$OUT_PATH"
    echo "Done."
}

# Models

case "$1" in

    "xray-hand-flow")
        download_gdrive "1EW0gl5_XgUPfdoNzAexTSAwvijuR40Qx" \
            "model_checkpoints/xray_hand/flow/full/model.pt"
        ;;

    "xray-hand-ddpm")
        download_gdrive "13OqA5H7MwW6k2nFfzm55tTkl0HDF3o_F" \
            "model_checkpoints/xray_hand/ddpm/full/model.pt"
        ;;


    # Help/Unknown
    *)
        echo "Unknown or missing argument: $1"
        echo ""
        echo "Usage:"
        echo "  bash download.sh <model-name>"
        echo ""
        echo "Available model names:"
        echo "  xray-hand-flow"
        echo "  xray-hand-ddpm"
        exit 1
        ;;
esac
