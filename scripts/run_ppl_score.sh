if [ $# -ne 2 ]; then
    echo "Usage: $0 <llama2_13b_model_path> <input_json_path>"
    exit 1
fi

MODEL_PATH=$1
INPUT_JSON=$2

python run_ppl_score.py \
    --model_path "$MODEL_PATH" \
    --input_json "$INPUT_JSON"
