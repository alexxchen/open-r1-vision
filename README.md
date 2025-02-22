# Open-R1-Vision
**One-click start reproduction of multi-modal DeepSeek R1-Zero**

Extended version of open-R1 featuring **multimodal GRPO training** with seamless Docker deployment and Slurm integration.

## Key Features ✨

- 🚀 **One-Click Training** via Docker containerization
- 🌐 **Multimodal Dataset Support** for large model training
- 🤖 **GRPO Optimization** to reproduce the aha moment in DeepSeek-R1-Zero
- ⚡ **Slurm Cluster Integration** for distributed training
- 📦 **Pre-built Docker Image** with dependency-free operation
- 🔧 **Zero Compatibility Issues** - Fully tested environment

## Quick Start ▶️

### Prerequisites
- Singularity ≥ 3.6
- Slurm client (optional for local execution)

### Launch Training:
```bash
git clone https://github.com/alexxchen/open-r1-vision.git
cd open-r1-vision
./start_run.sh
```
The script will automatically:
1. Pull the pre-built Docker image and convert it into singularity image
2. Launch Slurm job with optimal default parameters

### Slurm Configuration ⚙️
Customize training resource and parameters in slurm_singularity_text.sh or slurm_singularity_vision.sh

### Training Configuration
Modify config_text.yaml or config_vision.yaml for:
+ Adjust Hugginface model name 
+ Adjust Hugginface dataset name (Note: modification of the reward accuracy function in reward.py is needed for different datasets)
+ GRPO optimization settings

### License 📄
Apache 2.0 - See LICENSE for details

## Acknowledgments 🌟
Sincerely thank to [open-R1](https://github.com/huggingface/open-r1) and [R1-V](https://github.com/Deep-Agent/R1-V) 
