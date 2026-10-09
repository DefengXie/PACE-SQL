set -eux


python3 evaluate_model_qwen.py \
    --model_path ./Qwen_Qwen2.5-Coder-14B \
    --model_alia Qwen2.5-Coder-14B_base \
    --data_path opensql_fim_dataset.json \
    --data_name opensql_fim_eval_qwen14b \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --workers_per_gpu 2


python3 evaluate_model_deepseek.py \
    --model_path ./deepseek-coder-6.7b-base \
    --model_alia deepseek-coder-6.7b-base \
    --data_path opensql_fim_dataset.json \
    --data_name opensql_fim_eval_dpsk7B \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --workers_per_gpu 2
