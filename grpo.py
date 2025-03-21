# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from configs import GRPOConfig
from rewards import (
    accuracy_reward_GEOQA_R1V_Train_8K,
    accuracy_reward_math_lighteval,
    accuracy_reward_gsm8k,
    Gen_ORM_reward,
    Gen_PRM_reward,
    embedding_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
)
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, GRPOTrainer
from trainer import GRPOTrainerVL


from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """
    # The accuracy function should be modified based on the dataset
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    max_pixels: Optional[int] = field(
        default=28*28*1280,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=56*56,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def resize_dataset(dataset, args):
    # trainset_size % batch_size (gradient_acc * batch_size_per_gpu * gpu_num) == 0 to avoid deepspeed error
    if args.use_vllm:
        total_batch_size = args.gradient_accumulation_steps * args.per_device_eval_batch_size * args.world_size / args.num_generations
    else:
        total_batch_size = args.gradient_accumulation_steps * args.per_device_eval_batch_size * (args.world_size - 1) / args.num_generations
    dataset_size = dataset.num_rows
    new_size = int((dataset_size // total_batch_size) * total_batch_size)
    return dataset.select(range(new_size))

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy_math_lighteval": accuracy_reward_math_lighteval,
        "accuracy_GEOQA": accuracy_reward_GEOQA_R1V_Train_8K,
        "accuracy_gsm8k": accuracy_reward_gsm8k,
        "gen_orm_reward": Gen_ORM_reward,
        "gen_prm_reward": Gen_PRM_reward,
        "embedding": embedding_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation_vision(example):
        prompt = []

        if training_args.system_prompt is not None:
            # Qwen2 VL chat template support this paradigm
            prompt.append({"role": "system", "content": [{"type": "text", "text": training_args.system_prompt}]})
        prompt.append(
                {
                    "role": "user", 
                    "content": [
                            {"type": "image"},
                            {"type": "text", "text": example["problem"]},
                            ]
                }
            )
        return {"prompt": prompt}
    
    if "accuracy_gsm8k" in reward_funcs:
        suffix = " Slove the problem and conclude with the final answer formatted as #### [value], e.g., #### 0.01."
    else:
        suffix = ''

    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["question"]+suffix if "question" in example else example["problem"]+suffix})
            
        result = {"prompt": prompt}
        if "solution" not in example:
            result["solution"] = example["answer"]

        if "question" not in example:
            result["question"] = example["problem"]

        return result
    
    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_vision)
    else:
        dataset = dataset.map(make_conversation)


    if "embedding" in script_args.reward_funcs:
        dataset_name_suffix = script_args.dataset_name.split('/')[-1]
        embedding_cache_path = os.path.join("{}_embeddings_cache.pt".format(dataset_name_suffix))

        # 如果存在缓存文件则直接加载
        if os.path.exists(embedding_cache_path):
            logger.info("Loading cached embeddings...")
            embeddings_cache = torch.load(embedding_cache_path)
        else:
            logger.info("Please run preprocess first")

        # 将embeddings添加到dataset中
        for split in dataset:
            dataset[split] = dataset[split].add_column("solution_embedding", embeddings_cache[split])


    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    if "image" in dataset[script_args.dataset_train_split].features:
        trainer = GRPOTrainerVL(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=resize_dataset(dataset[script_args.dataset_train_split], training_args),
            eval_dataset=resize_dataset(dataset[script_args.dataset_test_split], training_args) if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )
    else:
        trainer = GRPOTrainer(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=resize_dataset(dataset[script_args.dataset_train_split], training_args),
            eval_dataset=resize_dataset(dataset[script_args.dataset_test_split], training_args) if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
