CUDA_VISIBLE_DEVICES=4 python alignment.py \
    --model1_name /media/ubuntu/data/mingjie/output/three_epoch/tmp_model_chat \
    --model2_name /media/ubuntu/data/share/Meta-Llama-3-8B-Instruct \
    --output_dir ./output/alignment_matrix_llama3 \
    --batch_size 32 \
    --epochs 5 \
    --lr 1e-3 \
    --n_negatives 10 \
    --seed 42
