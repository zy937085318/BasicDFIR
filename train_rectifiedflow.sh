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

# Create train_log directory if it doesn't exist
LOG_DIR="train_log"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yml)
LOG_FILE="$LOG_DIR/train_${CONFIG_BASENAME}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/train_${CONFIG_BASENAME}_${TIMESTAMP}.pid"
SUMMARY_FILE="$LOG_DIR/train_${CONFIG_BASENAME}_${TIMESTAMP}_summary.txt"

# Function to extract best metrics from log file
extract_best_metrics() {
    local log_file="$1"
    local summary_file="$2"

    if [ ! -f "$log_file" ]; then
        echo "Log file not found: $log_file"
        return 1
    fi

    echo "==========================================" > "$summary_file"
    echo "Training Summary - Best Metrics" >> "$summary_file"
    echo "==========================================" >> "$summary_file"
    echo "Log file: $log_file" >> "$summary_file"
    echo "Generated: $(date)" >> "$summary_file"
    echo "" >> "$summary_file"

    # Extract best metrics for each dataset
    # Pattern: "Best: value @ iter iter"
    local datasets=$(grep "Validation " "$log_file" | sed 's/.*Validation \([^[:space:]]*\).*/\1/' | sort -u)

    if [ -z "$datasets" ]; then
        echo "No validation results found in log file." >> "$summary_file"
        return 1
    fi

    for dataset in $datasets; do
        echo "Dataset: $dataset" >> "$summary_file"
        echo "----------------------------------------" >> "$summary_file"

        # Extract best PSNR - find the last validation result for this dataset
        local psnr_lines=$(grep -A 10 "Validation $dataset" "$log_file" | grep -i "psnr.*Best:")
        if [ -n "$psnr_lines" ]; then
            local best_psnr_line=$(echo "$psnr_lines" | tail -1)
            # Extract value and iteration using sed (more compatible)
            local best_psnr=$(echo "$best_psnr_line" | sed -n 's/.*Best: \([0-9.]*\).*/\1/p')
            local best_psnr_iter=$(echo "$best_psnr_line" | sed -n 's/.*@ \([0-9]*\) iter.*/\1/p')
            if [ -n "$best_psnr" ] && [ -n "$best_psnr_iter" ]; then
                echo "  Best PSNR: $best_psnr @ iteration $best_psnr_iter" >> "$summary_file"
            fi
        fi

        # Extract best SSIM
        local ssim_lines=$(grep -A 10 "Validation $dataset" "$log_file" | grep -i "ssim.*Best:")
        if [ -n "$ssim_lines" ]; then
            local best_ssim_line=$(echo "$ssim_lines" | tail -1)
            local best_ssim=$(echo "$best_ssim_line" | sed -n 's/.*Best: \([0-9.]*\).*/\1/p')
            local best_ssim_iter=$(echo "$best_ssim_line" | sed -n 's/.*@ \([0-9]*\) iter.*/\1/p')
            if [ -n "$best_ssim" ] && [ -n "$best_ssim_iter" ]; then
                echo "  Best SSIM: $best_ssim @ iteration $best_ssim_iter" >> "$summary_file"
            fi
        fi

        # Extract other metrics (if any) - look for lines with "Best:" that aren't PSNR or SSIM
        local other_metrics=$(grep -A 10 "Validation $dataset" "$log_file" | grep -i "Best:" | grep -v -i -E "(psnr|ssim)")
        if [ -n "$other_metrics" ]; then
            while IFS= read -r line; do
                # Extract metric name (text before colon)
                local metric_name=$(echo "$line" | sed -n 's/.*# \([^:]*\):.*/\1/p' | tr -d ' ')
                local metric_value=$(echo "$line" | sed -n 's/.*Best: \([0-9.]*\).*/\1/p')
                local metric_iter=$(echo "$line" | sed -n 's/.*@ \([0-9]*\) iter.*/\1/p')
                if [ -n "$metric_name" ] && [ -n "$metric_value" ] && [ -n "$metric_iter" ]; then
                    echo "  Best ${metric_name}: $metric_value @ iteration $metric_iter" >> "$summary_file"
                fi
            done <<< "$other_metrics"
        fi

        echo "" >> "$summary_file"
    done

    # Extract training time
    local training_time=$(grep "Time consumed:" "$log_file" | tail -1 | sed 's/.*Time consumed: //')
    if [ -n "$training_time" ]; then
        echo "Total Training Time: $training_time" >> "$summary_file"
    fi

    echo "==========================================" >> "$summary_file"

    # Display summary
    cat "$summary_file"
}

echo "=========================================="
echo "Starting Rectified Flow Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Log file: $LOG_FILE"
if [ "$RUN_BACKGROUND" = true ]; then
    echo "Running in background mode"
    echo "PID file: $PID_FILE"
fi
echo "=========================================="

# Run training
if [ "$RUN_BACKGROUND" = true ]; then
    # Run in background with nohup
    nohup python -m basicsr.train -opt "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    echo "Training started in background with PID: $TRAIN_PID"
    echo "To monitor progress, use: tail -f $LOG_FILE"
    echo "To stop training, use: kill $TRAIN_PID"
    echo "PID saved to: $PID_FILE"
    echo ""
    echo "Note: After training completes, check the summary file:"
    echo "  $SUMMARY_FILE"
    echo "Or view the log file: $LOG_FILE"
else
    # Run in foreground, output to both console and log file
    python -m basicsr.train -opt "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    # Check exit status
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Training completed successfully!"
        echo "=========================================="
        echo "Log saved to: $LOG_FILE"
        echo ""
        # Extract and display best metrics
        extract_best_metrics "$LOG_FILE" "$SUMMARY_FILE"
        echo ""
        echo "Summary saved to: $SUMMARY_FILE"
    else
        echo "=========================================="
        echo "Training failed with exit code $EXIT_CODE"
        echo "=========================================="
        echo "Log saved to: $LOG_FILE"
        # Still try to extract metrics if any validation was done
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Attempting to extract metrics from partial training..."
            extract_best_metrics "$LOG_FILE" "$SUMMARY_FILE" 2>/dev/null || true
        fi
        exit $EXIT_CODE
    fi
fi

