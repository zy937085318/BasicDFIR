#!/usr/bin/env bash

## 使用配置文件中的模型路径（如果已设置）
#./benchmark.sh flow 3
#
## 指定完整的模型路径
#./benchmark.sh flow 3 experiments/flowSR_20251207_123456_123456/models/net_g_latest.pth
#
## 使用通配符自动选择最新的模型（推荐）
#./benchmark.sh rectifiedflow 3 "experiments/RectifiedFlow_x4_DIV2K_300k_*/models/net_g_latest.pth"
#
## 指定 Meanflow 模型路径
#./benchmark.sh meanflow 3 experiments/meanflowSR_20251207_123456_123456/models/net_g_latest.pth
#
## 后台运行并指定模型路径
#./benchmark.sh rectifiedflow 3 "experiments/RectifiedFlow_x4_DIV2K_300k_*/models/net_g_latest.pth" --background
## 查找所有可用的模型
#ls -lt experiments/*/models/net_g_latest.pth
#
## 使用通配符选择最新的 RectifiedFlow 模型
#./benchmark.sh rectifiedflow 3 "experiments/RectifiedFlow_x4_DIV2K_300k_*/models/net_g_latest.pth"
#
## 使用完整路径
#./benchmark.sh rectifiedflow 3 experiments/RectifiedFlow_x4_DIV2K_300k_20251202_203141_980858/models/net_g_latest.pth


DEFAULT_MODEL="flow"
DEFAULT_GPU_ID="3"  # Default GPU ID to use

# Model configuration mapping
declare -A MODEL_CONFIGS
MODEL_CONFIGS["flow"]="options/test/Flow/test_Flow_x4.yml"
MODEL_CONFIGS["meanflow"]="options/test/Meanflow/test_Meanflow_x4.yml"
MODEL_CONFIGS["rectifiedflow"]="options/test/RectifiedFlow/test_RectifiedFlow_x4.yml"

# Parse arguments
MODEL_TYPE="${1:-$DEFAULT_MODEL}"
GPU_ID="${2:-$DEFAULT_GPU_ID}"
MODEL_PATH=""
RUN_BACKGROUND=false

# Parse all arguments to find model path and background flag
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    arg="${ARGS[$i]}"
    if [ "$arg" == "--background" ]; then
        RUN_BACKGROUND=true
    elif [[ "$arg" == *".pth" ]] || [[ "$arg" == *"experiments/"* ]]; then
        # This looks like a model path
        MODEL_PATH="$arg"
    fi
done

# If model path is provided as third argument and it's not --background, use it
if [ -z "$MODEL_PATH" ] && [ $# -ge 3 ] && [ "$3" != "--background" ]; then
    MODEL_PATH="$3"
fi

# Validate model type
if [ -z "${MODEL_CONFIGS[$MODEL_TYPE]}" ]; then
    echo "Error: Invalid model type: $MODEL_TYPE"
    echo "Available models: flow, meanflow, rectifiedflow"
    echo "Usage: $0 [model_type] [gpu_id] [model_path] [--background]"
    echo "Example: $0 flow 3"
    echo "Example: $0 meanflow 3 experiments/meanflowSR_*/models/net_g_latest.pth"
    echo "Example: $0 rectifiedflow 3 experiments/RectifiedFlow_x4_DIV2K_300k_*/models/net_g_latest.pth --background"
    exit 1
fi

# Get config file for selected model
CONFIG_FILE="${MODEL_CONFIGS[$MODEL_TYPE]}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please ensure the test configuration file exists for model: $MODEL_TYPE"
    exit 1
fi

# Validate GPU ID
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid GPU ID: $GPU_ID. Must be a number."
    echo "Usage: $0 [model_type] [gpu_id] [model_path] [--background]"
    echo "Example: $0 flow 3"
    exit 1
fi

# If model path is provided, create a temporary config file with updated path
TEMP_CONFIG=""
if [ -n "$MODEL_PATH" ]; then
    # Expand wildcards if present
    if [[ "$MODEL_PATH" == *"*"* ]]; then
        # Find the most recent matching path
        EXPANDED_PATH=$(ls -td $MODEL_PATH 2>/dev/null | head -n 1)
        if [ -n "$EXPANDED_PATH" ]; then
            MODEL_PATH="$EXPANDED_PATH"
        else
            echo "Error: No matching model path found: $MODEL_PATH"
            exit 1
        fi
    fi

    # Check if model file exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model file not found: $MODEL_PATH"
        echo "Please check the path and try again."
        exit 1
    fi

    # Create temporary config file
    TEMP_CONFIG=$(mktemp /tmp/benchmark_config_XXXXXX.yml)
    cp "$CONFIG_FILE" "$TEMP_CONFIG"

    # Update model path in temporary config file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|pretrain_network_g:.*|pretrain_network_g: $MODEL_PATH|" "$TEMP_CONFIG"
    else
        # Linux
        sed -i "s|pretrain_network_g:.*|pretrain_network_g: $MODEL_PATH|" "$TEMP_CONFIG"
    fi

    CONFIG_FILE="$TEMP_CONFIG"
    echo "Using model path: $MODEL_PATH"
fi

# Check if model path is set in config file
# Extract model path from YAML (simple grep-based check)
CONFIG_MODEL_PATH=$(grep -E "^\s*pretrain_network_g:" "$CONFIG_FILE" | sed 's/.*pretrain_network_g:\s*//' | sed 's/#.*$//' | xargs)
if [ -z "$CONFIG_MODEL_PATH" ] || [ "$CONFIG_MODEL_PATH" = "~" ] || [ "$CONFIG_MODEL_PATH" = "null" ]; then
    echo "=========================================="
    echo "WARNING: Model path not set in config file!"
    echo "=========================================="
    echo "Please update the 'pretrain_network_g' path in: ${MODEL_CONFIGS[$MODEL_TYPE]}"
    echo "Or specify model path as argument: $0 $MODEL_TYPE $GPU_ID <model_path>"
    echo ""
    echo "After training, your model will be saved in: experiments/<experiment_name>/models/net_g_latest.pth"
    echo ""
    echo "To find your trained model:"
    echo "  ls -lt experiments/*/models/net_g_latest.pth"
    echo ""
    if [ "$RUN_BACKGROUND" = false ]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG"
            echo "Benchmark cancelled."
            exit 1
        fi
        echo "Continuing with uninitialized model (results will be meaningless)..."
    else
        [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG"
        echo "ERROR: Cannot run benchmark in background without a valid model path."
        echo "Please set 'pretrain_network_g' in the config file or specify model path as argument."
        exit 1
    fi
fi

echo "=========================================="
echo "Starting Benchmark Test"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Config file: ${MODEL_CONFIGS[$MODEL_TYPE]}"
if [ -n "$MODEL_PATH" ]; then
    echo "Model path: $MODEL_PATH"
fi
echo "Using GPU: $GPU_ID"
if [ "$RUN_BACKGROUND" = true ]; then
    echo "Running in background mode"
fi
echo "=========================================="

# Function to cleanup temp file
cleanup() {
    if [ -n "$TEMP_CONFIG" ] && [ -f "$TEMP_CONFIG" ]; then
        rm -f "$TEMP_CONFIG"
    fi
}
trap cleanup EXIT

# Run benchmark
if [ "$RUN_BACKGROUND" = true ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m basicsr.test -opt "$CONFIG_FILE" > /dev/null 2>&1 &
    BENCHMARK_PID=$!
    echo "Benchmark started in background with PID: $BENCHMARK_PID"
    echo "To stop benchmark, use: kill $BENCHMARK_PID"
    echo "To view logs, check: results/$(basename ${MODEL_CONFIGS[$MODEL_TYPE]} .yml)/log/"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m basicsr.test -opt "$CONFIG_FILE"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Benchmark completed successfully!"
        echo "=========================================="
        echo "Results saved in: results/$(basename ${MODEL_CONFIGS[$MODEL_TYPE]} .yml)/"
    else
        echo "=========================================="
        echo "Benchmark failed with exit code $EXIT_CODE"
        echo "=========================================="
        exit $EXIT_CODE
    fi
fi
