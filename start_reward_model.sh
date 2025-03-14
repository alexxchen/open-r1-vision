#! /bin/bash
#SBATCH --job-name=rewarding
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --nodelist=gpu1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
ulimit -u unlimited

export NUM_THREADS=16

# BINDS="--bind /opt/app/nvidia/535.154.05/:/usr/local/nvidia"

# server on
# http://{ip-address}:8000/v1
singularity exec --cleanenv --nv  \
    --env OMP_NUM_THREADS=${NUM_THREADS} \
    --env MKL_NUM_THREADS=${NUM_THREADS} \
    --env NUMEXPR_NUM_THREADS=${NUM_THREADS} \
    ./open-r1-vision.sif \
    bash -c ' \
        vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --gpu-memory-utilization 0.7 --tensor-parallel-size 2 --max-num-seqs 1024 --uvicorn-log-level debug
    '
  


