# Detoxification for LLMs via Projection Alignment

This project provides a pipeline for detoxifying large language models (LLMs) using projection alignment techniques. The method involves training a teacher model, aligning it with a target model, generating detoxified responses, and evaluating the results.

## Pipeline Overview

### 1. Pretrain Teacher Model
```bash
./run_pretrain.sh
```
Trains a teacher model on detoxification tasks using distributed training across multiple GPUs. The pretrained model will be saved in the specified output directory.

Key parameters:
- `model_name_or_path`: Base model to start training from
- `training_data_path`: Path to training dataset
- `output_dir`: Directory to save trained model
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Training learning rate
- `num_train_epochs`: Number of training epochs
- `deepspeed`: DeepSpeed configuration file

### 2. Train Alignment Matrix
```bash
./run_alignment.sh
```
Computes the alignment matrix between the teacher model and a target model (e.g., Llama-3) using contrastive learning.

Key parameters:
- `model1_name`: Path to teacher model
- `model2_name`: Path to target model
- `output_dir`: Directory to save alignment matrix
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `n_negatives`: Number of negative samples per positive pair

### 3. Generate Detoxified Responses
```bash
./run_generation.sh <teacher_model_path> <target_model_path> <matrix_A_path> <device> <output_path> <data_path>
```
Generates responses using the alignment method with different alpha values.

Parameters:
1. `teacher_model_path`: Path to pretrained teacher model
2. `target_model_path`: Path to target model
3. `matrix_A_path`: Path to alignment matrix
4. `device`: Computation device (e.g., cuda:0)
5. `output_path`: Path to save generated responses
6. `data_path`: Path to challenge prompts JSON file

### 4. Evaluate Results

#### Option A: Toxicity Score Evaluation
```bash
./run_tox_score.sh <device> <input_json_path>
```
Computes toxicity scores using Detoxify's original model.

Parameters:
1. `device`: Computation device (e.g., cuda:0 or cpu)
2. `input_json_path`: Path to generated responses JSON file

#### Option B: Perplexity (PPL) Evaluation
```bash
./run_ppl_score.sh <llama2_13b_model_path> <input_json_path>
```
Computes perplexity scores using Llama2-13B as the reference model.

Parameters:
1. `llama2_13b_model_path`: Path to Llama2-13B model
2. `input_json_path`: Path to generated responses JSON file

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/detoxification-projection-alignment.git
cd detoxification-projection-alignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the full pipeline:
```bash
# Step 1: Pretrain teacher model
./run_pretrain.sh

# Step 2: Train alignment matrix
./run_alignment.sh

# Step 3: Generate responses
./run_generation.sh \
    /path/to/teacher_model \
    /path/to/target_model \
    /path/to/alignment_matrix.pt \
    cuda:0 \
    ./generation_results.json \
    ./challenge_prompts.jsonl

# Step 4: Evaluate toxicity
./run_tox_score.sh cuda:0 ./generation_results.json

# Step 4 (Alternative): Evaluate perplexity
./run_ppl_score.sh /path/to/llama2-13b ./generation_results.json
```

## Configuration Tips

- For faster training, use higher batch sizes with gradient accumulation
- Experiment with different alpha values (0.0-0.6) for detoxification strength
- Use bfloat16 precision for faster inference on compatible hardware
- For large models, use DeepSpeed for efficient distributed training

## Results Interpretation

- **Toxicity Score**: Lower values indicate less toxic content (range: 0-1)
- **Perplexity (PPL)**: Lower values indicate higher text quality
- Optimal alpha values balance detoxification with text quality
