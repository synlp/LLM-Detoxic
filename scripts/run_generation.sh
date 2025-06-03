#!/bin/bash

# 参数检查
if [ $# -ne 5 ]; then
    echo "Usage: $0 <teacher_model_path> <target_model_path> <matrix_A_path> <device> <output_path>"
    exit 1
fi

TEACHER_MODEL=$1
TARGET_MODEL=$2
MATRIX_A=$3
DEVICE=$4
OUTPUT_PATH=$5

python run_generation.py \
    --teacher_model "$TEACHER_MODEL" \
    --target_model "$TARGET_MODEL" \
    --matrix_a "$MATRIX_A" \
    --device "$DEVICE" \
    --output_path "$OUTPUT_PATH"
