#!/bin/bash

DEFAULT_CONFIG="options/train/RectifiedFlow/train_RectifiedFlow_x4.yml"

# Parse arguments
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
RUN_BACKGROUND=false

# Check for background flag
if [ "$2" == "--background" ] || [ "$1" == "--background" ]; then
    RUN_BACKGROUND=true
    # If first arg is --background, use default config
    if [ "$1" == "--background" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
    fi
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting Rectified Flow Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
if [ "$RUN_BACKGROUND" = true ]; then
    echo "Running in background mode"
fi
echo "=========================================="

# Run training
if [ "$RUN_BACKGROUND" = true ]; then
    # Run in background with nohup (output to /dev/null to suppress)
    nohup python -m basicsr.train -opt "$CONFIG_FILE" > /dev/null 2>&1 &
    TRAIN_PID=$!
    echo "Training started in background with PID: $TRAIN_PID"
    echo "To stop training, use: kill $TRAIN_PID"
else
    # Run in foreground, output to console only
    python -m basicsr.train -opt "$CONFIG_FILE"
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

