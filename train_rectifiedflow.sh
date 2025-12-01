#!/bin/bash

DEFAULT_CONFIG="options/train/RectifiedFlow/train_RectifiedFlow_x4.yml"
DEFAULT_GPU="1"

# Parse arguments
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
GPU_ID="${2:-$DEFAULT_GPU}"
RUN_BACKGROUND=false

# Check for background flag
if [ "$3" == "--background" ] || [ "$2" == "--background" ] || [ "$1" == "--background" ]; then
    RUN_BACKGROUND=true
    # Adjust config and GPU based on which argument is --background
    if [ "$1" == "--background" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
        GPU_ID="$DEFAULT_GPU"
    elif [ "$2" == "--background" ]; then
        CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
        GPU_ID="$DEFAULT_GPU"
    fi
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Validate GPU ID
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid GPU ID: $GPU_ID. Must be a number."
    echo "Usage: $0 [config_file] [gpu_id] [--background]"
    echo "Example: $0 options/train/RectifiedFlow/train_RectifiedFlow_x4.yml 0"
    exit 1
fi

echo "=========================================="
echo "Starting Rectified Flow Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Using GPU: $GPU_ID"
if [ "$RUN_BACKGROUND" = true ]; then
    echo "Running in background mode"
fi
echo "=========================================="

# Run training with specified GPU
if [ "$RUN_BACKGROUND" = true ]; then
    # Run in background with nohup (output to /dev/null to suppress)
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m basicsr.train -opt "$CONFIG_FILE" > /dev/null 2>&1 &
    TRAIN_PID=$!
    echo "Training started in background with PID: $TRAIN_PID"
    echo "To stop training, use: kill $TRAIN_PID"
else
    # Run in foreground, output to console only
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m basicsr.train -opt "$CONFIG_FILE"
    EXIT_CODE=$?

    # Check exit status
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Training completed successfully!"
        echo "=========================================="
    else
        echo "=========================================="
        echo "Training failed with exit code $EXIT_CODE"
        echo "=========================================="
        exit $EXIT_CODE
    fi
fi

