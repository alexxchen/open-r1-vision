# Open-R1-Vision
**One-click start reproduction of multi-modal DeepSeek R1-Zero**

Extended version of open-R1 featuring **multimodal GRPO training** with seamless Docker deployment and Slurm integration.

## News 🗞️

+ [2025/3/14]: We support using generative reward model to judge the answer and the reasoning process. This will expand the GRPO training to text reasoning rather than just Math or Coding task.

## Key Features ✨

- 🤖 **A More Perfect Reproduction** - The demo in Open-R1 did not replicate the phenomenon of increased response length. However, the modified version here successfully demonstrates this phenomenon with longer responses.
- 🔧 **Zero Compatibility Issues** - The GRPO algorithm is undergoing rapid iterations. The code of the trl, vllm, and transformers packages have been reviewed to ensure that there are no compatibility issues with the current version.
- 🌐 **Multi Dataset Support** - Support both textual datasets and multimodal datasets.
- ⚡ **Pre-built Docker Image** - To minimize the cost of environment configuration and improve reproduction efficiency, not only a Dockerfile is provided but also a pre-built image to ensure developers can quickly restore the experimental environment.
- 🚀 **One-Click Training on Slurm** - Docker cannot be used on Slurm, so we provide Singularity commands to support large-scale training on Slurm.

## Results
- Training logs on GSM8K - The average response length increases during training.
  ![The average response length increases](images/gsm8k.png)

- Training logs on GeoQA (to be release)

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

### Trainig with Generative Reward Model
In the technical report of Deepseek-R1 model, they use Deepseek-V3 as generative reward model for judgment.
We provide an example that use both ORM and PRM reward in config_text_RM.yaml.

To run this, we need:

+ Switch yaml file name to "config_text_RM.yaml" in slurm_singularity_text.sh
+ Bring the reward model online by ```sbatch start_reward_model.sh```
+ Start training ```./start_run.sh```

The results on GSM8K show that the ORM reward is lower than rule-based reward. This discrepancy occurs because the reward model evaluates the entire answer, not just the final value.
![The average response length increases](images/gsm8k-Qwen-1.5B-Instruct-RM.jpeg)

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
Special thanks to Prof. Zuoren Wang from Center for Excellence in Brain Science and Intelligence Technology for the support. Sincerely thank to [open-R1](https://github.com/huggingface/open-r1) and [R1-V](https://github.com/Deep-Agent/R1-V)
