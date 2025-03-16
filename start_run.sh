#!/bin/bash
if [ ! -f ./open-r1-vision.sif ]; then
    singularity pull open-r1-vision.sif docker://crpi-0iga3x1q4si035cn.cn-shanghai.personal.cr.aliyuncs.com/alecchen/open-r1-vision:latest
fi
sbatch slurm_singularity_text.sh
