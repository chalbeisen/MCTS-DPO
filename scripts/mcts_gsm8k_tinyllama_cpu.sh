export CUDA_VISIBLE_DEVICES=""  # Force CPU

ZERO_STAGE=0
OFFLOAD="none"

# pick a free master port dynamically as you do, or set fixed:
MASTER_PORT=29500

ACTOR_MODEL_NAME_OR_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ACTOR_REF_MODEL_NAME_OR_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK=1
export DS_ACCELERATOR=cpu
export DS_BUILD_OPS=0

deepspeed --master_port ${MASTER_PORT} \
  --module mcts_rl.algorithms.mcts \
  --train_datasets GSM8K/train \
  --model_type llama3 \
  --choose_worst \
  --save_mcts_data \
  --filter \
  --iteration_interval 64 \
  --actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
  --actor_ref_model_name_or_path "${ACTOR_REF_MODEL_NAME_OR_PATH}" \
  --scale_coeff 0.1 \
  --max_length 512 \
  --temperature 1.0 \
  --init_temperature 1.0 \
  --mcts_length_penalty 1.25 \
  --num_return_sequences 1 \
  --repetition_penalty 1.0 \
  --trust_remote_code True \
  --epochs 1 \
  --conservative \
  --update_iters 1 \
  --save_interval 64 \
  --per_device_ptx_batch_size 4 \
  --per_device_prompt_batch_size 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --actor_lr 1e-6 \
  --actor_weight_decay 0.05 \
  --actor_lr_scheduler_type cosine \
  --actor_lr_warmup_ratio 0.03 \
  --actor_gradient_checkpointing \
  --seed 42 \
  --kl_coeff 0.02 \
  --clip_range_ratio 0.2 \
  --clip_range_score 50.0 \
  --clip_range_value 5.0 \
  --ptx_coeff 0.0 \
  --output_dir "${OUTPUT_DIR}" \
  --log_type wandb \
  --log_project MCTS-IPL-Math \
  --zero_stage "${ZERO_STAGE}" \
  --offload "${OFFLOAD}" \
  --max_new_tokens 128 \
  --n_iters 64 \
  --depth_limit 3 \
  --n_init_actions 2 \
  --n_actions 2 \
  --force_terminating_on_depth_limit \
  --mcts_temperature 0.0
