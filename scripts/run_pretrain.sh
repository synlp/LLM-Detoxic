
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 7 --master-port 21443 pretrain_main.py \
    --model_name_or_path /media/ubuntu/data/share/Mistral-7B-v0.1 \
    --training_data_path /media/ubuntu/data/yuanhe/project/det-llm/data/wildjailbreak/train/train.tsv \
    --output_dir ./output/three_epoch/tmp_model_Mistral \
    --save_total_limit 3 \
    --report_to none \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3 \
    --save_strategy steps \
    --logging_steps 100 \
    --save_steps 1000 \
    --bf16 true \
    --resume_from_checkpoint False \
    --save_only_model \
    --deepspeed ./deepspeed/ds_z0_config.json

