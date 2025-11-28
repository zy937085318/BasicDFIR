#!/bin/bash

# Rectified Flow Training Script
# Usage: ./train.sh [config_file]
# Example: ./train.sh options/train/RectifiedFlow/train_RectifiedFlow_x4.yml

# Default configuration file
DEFAULT_CONFIG="options/train/RectifiedFlow/train_RectifiedFlow_x4.yml"

# Use provided config file or default
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting Rectified Flow Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Run training
python -m basicsr.train -opt "$CONFIG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code $?"
    echo "=========================================="
    exit 1
fi

