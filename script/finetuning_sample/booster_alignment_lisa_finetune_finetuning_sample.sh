#!/bin/bash

module load miniforge3
module load cuda/12.0
conda activate adaptive-attack

seed=${1:-5}
alpha=${2:-1e-1}
eta_lbd=${3:-5}
rho=${4:-0.01}
task=${5:-gsm8k}  # Default task, can be overridden
model_path="/path/to/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

alignment_epoch=20
alignment_safe_batch_size=16
bad_sample_num=5000 
align_sample_num=5000

finetuning_epoch=20
finetuning_batch_size=16

# Fixed poison ratio
poison_ratio_fixed=0.20

# Finetuning sample sizes to iterate over
finetuning_samples=(500 1000 1500 2000 2500)

# LISA-specific hyperparameters
guide_data_num=10000
align_step=100   
finetune_step=900 

# Extract a concise model identifier (e.g., meta-llama--Llama-2-7b-hf) from the full cache path.
# For typical HF cache layout ".../models--<org>--<model>/snapshots/<hash>", grab the directory between
# "models--" and "/snapshots"; otherwise fall back to a reasonable default.
path_after_slash=$(echo "$model_path" | sed -n 's#.*/models--\([^/]*\)/.*#\1#p')
# Trim any prefix ending with "--" so we keep only the model identifier (e.g., "Llama-2-7b-hf").
path_after_slash=${path_after_slash##*--}

# If the above extraction fails (custom path or different layout), use the parent directory name
# and strip an optional "models--" prefix.
if [ -z "$path_after_slash" ]; then
	path_after_slash=$(basename "$(dirname "$model_path")")
	path_after_slash=${path_after_slash#models--}
	# Trim any prefix ending with "--" to keep only the final model name segment
	path_after_slash=${path_after_slash##*--}
fi
job_id=${SLURM_JOB_ID:-local}

# Alignment path helpers
alignment_base=${path_after_slash}_booster_lisa_seed${seed}_job${job_id}
alignment_ckpt=ckpt/${alignment_base}
alignment_poison=data/poison/${alignment_base}

export WANDB_MODE="online"

echo "Alignment Parameters:"
echo "The value of eta_lbd is: $eta_lbd"
echo "The value of alpha is: $alpha"
echo "The value of bad_sample_num is: $bad_sample_num"
echo "The value of align_sample_num is: $align_sample_num"
echo "Downstream Tasks Parameters:"
echo "The selected task is: $task"
echo "The fixed poison ratio is: ${poison_ratio_fixed}"
echo "The finetuning_sample_num values to iterate over are: ${finetuning_samples[@]}"
echo "The value of rho is: $rho"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
echo "The job id is: $job_id"
echo "The seed is: $seed"

echo "Starting from directory: $(pwd)"
cd ../../                            # Change to working directory from script/finetuning_sample

echo "Starting alignment training..."
CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path beavertails_with_refusals_train_filtered_safe \
	--bf16 True \
	--output_dir ${alignment_ckpt} \
	--num_train_epochs ${alignment_epoch} \
	--per_device_train_batch_size ${alignment_safe_batch_size} \
	--per_device_eval_batch_size ${alignment_safe_batch_size} \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "epoch" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer booster \
	--sample_num $align_sample_num \
	--bad_sample_num $bad_sample_num \
	--eta_lbd ${eta_lbd} \
	--alpha ${alpha} \
	--report_to wandb \
	--seed ${seed}

cd poison/evaluation

# Run alignment evaluation
CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../${alignment_ckpt} \
	--model_folder ${model_path} \
	--output_path ../../${alignment_poison}

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../${alignment_poison}

cd ../../                            # Return to main directory

# Finetuning and evaluation for different finetuning_sample_num values
for finetuning_sample_num in "${finetuning_samples[@]}"; do
	echo "Starting ${task^^} training with finetuning_sample_num: $finetuning_sample_num and poison ratio: ${poison_ratio_fixed} (LISA optimizer)..."

	benign_dataset=data/${task}.json
	# Common suffix for all paths in this configuration
	base_name=${path_after_slash}_seed${seed}_job${job_id}_fs${finetuning_sample_num}_pr${poison_ratio_fixed}_rho${rho}

	# Frequently-used paths
	output_dir=ckpt/${task}/${base_name}
	poison_output=data/poison/${task}/${base_name}
	pred_eval_output=data/${task}/${base_name}
	lora_folder=${alignment_ckpt}
	wandb_run_name=${task}_${base_name}_lisa

	CUDA_VISIBLE_DEVICES=0 python train.py \
		--model_name_or_path ${model_path} \
		--lora_folder ${lora_folder} \
		--data_path ./data/beavertails_disjoint_attack_deduplicated.json \
		--bf16 True \
		--output_dir ${output_dir} \
		--num_train_epochs ${finetuning_epoch} \
		--per_device_train_batch_size ${finetuning_batch_size} \
		--per_device_eval_batch_size ${finetuning_batch_size} \
		--gradient_accumulation_steps 1 \
		--save_strategy "steps" \
		--save_steps 100000 \
		--save_total_limit 0 \
		--learning_rate 1e-5 \
		--weight_decay 0.1 \
		--warmup_ratio 0.1 \
		--lr_scheduler_type "constant" \
		--logging_steps 10 \
		--tf32 True \
		--cache_dir cache \
		--optimizer lisa \
		--evaluation_strategy  "epoch" \
		--sample_num ${finetuning_sample_num} \
		--poison_ratio ${poison_ratio_fixed} \
		--label_smoothing_factor 0 \
		--benign_dataset ${benign_dataset} \
		--bad_sample_num $bad_sample_num \
		--alternating single_lora \
		--report_to wandb \
		--wandb_project finetuning \
		--run_name ${wandb_run_name} \
		--seed ${seed} \
		--rho ${rho} \
		--alignment_step ${align_step} \
		--finetune_step ${finetune_step} \
		--guide_data_num ${guide_data_num}

	# Run harmfulness evaluation
	cd poison/evaluation
	CUDA_VISIBLE_DEVICES=0 python pred.py \
		--lora_folder ../../${output_dir} \
		--model_folder ${model_path} \
		--output_path ../../${poison_output}
	
	echo "${task} HS (finetuning_sample_num: $finetuning_sample_num):"
	CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
		--input_path ../../${poison_output}
	cd ../../

	cd ${task}
	echo "${task} FA (finetuning_sample_num: $finetuning_sample_num):"
	CUDA_VISIBLE_DEVICES=0 python pred_eval.py \
		--lora_folder ../${output_dir} \
		--model_folder ${model_path} \
		--output_path ../data/${task}/${base_name}
	cd ..

done 