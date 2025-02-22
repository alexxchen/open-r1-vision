#! /bin/bash
#SBATCH --job-name=R1-training
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
ulimit -u unlimited

# export HF_ENDPOINT=https://hf-mirror.com
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_RUN_NAME=Qwen2-VL-2B-Instruct-GEOQA-8k-GRPO-$(date +%Y-%m-%d-%H-%M-%S)

# Bind necessary directories into the container
# BINDS="--bind /opt/app/nvidia/535.154.05/:/usr/local/nvidia"

#you can use eigther lora or deepspeed for fine tune
# Execute the container
# ${BINDS} \
# --env HF_ENDPOINT=${HF_ENDPOINT} \
singularity exec --cleanenv --nv  \
    --env WANDB_RUN_NAME=${WANDB_RUN_NAME} \
    ./open-r1-vision.sif \
    bash -c ' \
        torchrun --standalone --nnodes=1 --nproc_per_node=7 \
            grpo.py \
            --config config_vision.yaml \
            --output_dir checkpoints/${WANDB_RUN_NAME} \
            --run_name ${WANDB_RUN_NAME}
    '



