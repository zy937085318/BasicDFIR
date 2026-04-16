#!/usr/bin/env bash

#aaa

DEFAULT_CONFIG="options/train/Flow/train_Flow_x4.yml"
DEFAULT_GPU="1"
DEFAULT_GPU_ID="3"  # Default GPU ID to use

# Parse arguments
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
GPU_COUNT="${2:-$DEFAULT_GPU}"
RUN_BACKGROUND=false

# Check for background flag
if [ "$3" == "--background" ] || [ "$2" == "--background" ] || [ "$1" == "--background" ]; then
    RUN_BACKGROUND=true
    # Adjust config and GPU based on which argument is --background
    if [ "$1" == "--background" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
        GPU_COUNT="$DEFAULT_GPU"
    elif [ "$2" == "--background" ]; then
        CONFIG_FILE="${1:-$DEFAULT_CONFIG}"
        GPU_COUNT="$DEFAULT_GPU"
    fi
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Validate GPU count
if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid GPU count: $GPU_COUNT. Must be a number."
    echo "Usage: $0 [config_file] [gpu_count] [--background]"
    echo "Example: $0 options/train/Flow/train_Flow_x4.yml 1"
    exit 1
fi

# Generate GPU IDs based on GPU count
# For single GPU, use the default GPU ID (3)
# For multiple GPUs, start from the default GPU ID
if [ "$GPU_COUNT" -eq 1 ]; then
    GPU_IDS="$DEFAULT_GPU_ID"
elif [ "$GPU_COUNT" -eq 2 ]; then
    GPU_IDS="$DEFAULT_GPU_ID,$((DEFAULT_GPU_ID + 1))"
elif [ "$GPU_COUNT" -eq 4 ]; then
    GPU_IDS="$DEFAULT_GPU_ID,$((DEFAULT_GPU_ID + 1)),$((DEFAULT_GPU_ID + 2)),$((DEFAULT_GPU_ID + 3))"
elif [ "$GPU_COUNT" -eq 8 ]; then
    GPU_IDS="$DEFAULT_GPU_ID,$((DEFAULT_GPU_ID + 1)),$((DEFAULT_GPU_ID + 2)),$((DEFAULT_GPU_ID + 3)),$((DEFAULT_GPU_ID + 4)),$((DEFAULT_GPU_ID + 5)),$((DEFAULT_GPU_ID + 6)),$((DEFAULT_GPU_ID + 7))"
else
    # For other numbers, generate sequential IDs starting from default GPU ID
    GPU_IDS=$(seq -s, $DEFAULT_GPU_ID $((DEFAULT_GPU_ID + GPU_COUNT - 1)))
fi

echo "=========================================="
echo "Starting Flow Model Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Using GPUs: $GPU_COUNT (GPU IDs: $GPU_IDS)"
if [ "$RUN_BACKGROUND" = true ]; then
    echo "Running in background mode"
fi
echo "=========================================="

# Set master port for distributed training
MASTER_PORT=${PORT:-29500}

# Run training
if [ "$GPU_COUNT" -eq 1 ]; then
    # Single GPU training
    if [ "$RUN_BACKGROUND" = true ]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python -m basicsr.train -opt "$CONFIG_FILE" > /dev/null 2>&1 &
        TRAIN_PID=$!
        echo "Training started in background with PID: $TRAIN_PID"
        echo "To stop training, use: kill $TRAIN_PID"
    else
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -m basicsr.train -opt "$CONFIG_FILE"
        EXIT_CODE=$?

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
else
    # Multi-GPU distributed training
    # Try torchrun first (recommended, uses environment variables by default)
    # Fallback to torch.distributed.launch with --use-env if torchrun is not available
    if command -v torchrun &> /dev/null; then
        # Use torchrun (PyTorch 1.9+, recommended)
        if [ "$RUN_BACKGROUND" = true ]; then
            PYTHONPATH="$(dirname $0):${PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES=$GPU_IDS nohup torchrun \
                --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
                basicsr/train.py -opt "$CONFIG_FILE" --launcher pytorch > /dev/null 2>&1 &
            TRAIN_PID=$!
            echo "Training started in background with PID: $TRAIN_PID"
            echo "To stop training, use: kill $TRAIN_PID"
        else
            PYTHONPATH="$(dirname $0):${PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
                --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
                basicsr/train.py -opt "$CONFIG_FILE" --launcher pytorch
            EXIT_CODE=$?

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
    else
        # Fallback to torch.distributed.launch with --use-env
        # --use-env passes local_rank via environment variable instead of --local-rank argument
        if [ "$RUN_BACKGROUND" = true ]; then
            PYTHONPATH="$(dirname $0):${PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python -m torch.distributed.launch \
                --use-env --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
                basicsr/train.py -opt "$CONFIG_FILE" --launcher pytorch > /dev/null 2>&1 &
            TRAIN_PID=$!
            echo "Training started in background with PID: $TRAIN_PID"
            echo "To stop training, use: kill $TRAIN_PID"
        else
            PYTHONPATH="$(dirname $0):${PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch \
                --use-env --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
                basicsr/train.py -opt "$CONFIG_FILE" --launcher pytorch
            EXIT_CODE=$?

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
    fi
fi

