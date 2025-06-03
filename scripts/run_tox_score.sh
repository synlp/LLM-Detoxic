if [ $# -ne 2 ]; then
    echo "Usage: $0 <device> <input_json_path>"
    exit 1
fi

DEVICE=$1
INPUT_JSON=$2

python run_tox_score.py \
    --device "$DEVICE" \
    --input_json "$INPUT_JSON"
