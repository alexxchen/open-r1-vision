"""Reward functions for GRPO training."""
import os
import math
import re
from datetime import datetime
from typing import Dict
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openai import APITimeoutError, APIError, APIConnectionError
import asyncio
import numpy as np
import time
import torch.distributed as dist

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward_GEOQA_R1V_Train_8K(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def accuracy_reward_math_lighteval(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        sol_match = re.search(r'\\boxed\{(.*?)\}', sol)
        sol_parsed = sol_match.group(1).strip() if sol_match else sol.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        if content_match:
            answer = content_match.group(1).strip()
            answer_match = re.search(r'\\boxed\{(.*)\}', answer)
            if answer_match:
                answer_parsed = answer_match.group(1).strip()
                reward = float(verify(answer_parsed, sol_parsed))
            else:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def accuracy_reward_gsm8k(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        sol_match = re.search(r'####(.*)', sol)
        sol_parsed = sol_match.group(1).strip() if sol_match else sol.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        if content_match:
            answer_parsed = content_match.group(1).strip()
            final_parsed = re.search(r'####(.*)', answer_parsed)
            if final_parsed:
                final_answer = final_parsed.group(1).strip()
                reward = float(verify(final_answer, sol_parsed))
            else:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def embedding_reward(completions, solution_embedding, **kwargs):
    
    openai_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    answers = []
    for content in contents:
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        if content_match:
            answer_parsed = content_match.group(1).strip()
            if len(answer_parsed) == 0:
                answers.append(' ')
            else:
                answers.append(answer_parsed)
        else:
            answers.append(' ')

    # client = OpenAI(base_url=openai_base, api_key=api_key)
    client = AzureOpenAI(azure_endpoint=openai_base, api_key=api_key, api_version = "2023-05-15")

    response = client.embeddings.create( model="text-embedding-3-large", input=answers)

    for content_data, sol_embedding in zip(response.data, solution_embedding):
        completion_embed = content_data.embedding
        similarity = np.dot(completion_embed, sol_embedding) / (np.linalg.norm(completion_embed) * np.linalg.norm(sol_embedding))
        rewards.append(float(similarity))

    return rewards

class OpenAI_ClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AsyncOpenAI(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
        return cls._instance

def Gen_ORM_reward(question, completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]

    answers = []
    for content in contents:
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        if content_match:
            answer_parsed = content_match.group(1).strip()
            if len(answer_parsed) == 0:
                answers.append('empty')
            else:
                answers.append(answer_parsed)
        else:
            answers.append('empty')
    
    timeout_duration = 60

    async def async_request(client, prompt):
        try:
            response = await client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                stream=False,
                timeout=timeout_duration,
                temperature=0.1,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API final failure: {e}")
            return '[[0.0]]'

    async def batch_async_requests(client, prompts):
        tasks = [async_request(client, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    RM_PROMPT = '''Please act as an impartial judge and evaluate the quality of the responses provided by AI
        assistants to the user question displayed below. Please compare the generated answer with the standard 
        answer and provide a similarity score between 0 and 1, where 1 means identical and 0 means completely different.
        Don't be fooled by a placeholder answer like "detailed answer here". Just output your final score in this format: "[[your score]]", with no other words or explanation.
        User:
        [Question]
        {question}
        [The Start of Standard Answer]
        {sol}
        [The End of Standard Answer]
        [The Start of Assistant's Answer]
        {answer}
        [The End of Assistant's Answer]'''

    prompts = [RM_PROMPT.format(question=prob, sol=sol, answer=ans) 
           for prob, ans, sol in zip(question, answers, solution)]
    
    client = OpenAI_ClientSingleton.get_instance()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    responses = loop.run_until_complete(batch_async_requests(client, prompts))
    
    rewards = []
    for res in responses:
        score = re.search(r'\[\[(0?\.\d+|1(?:\.0*)?)\]\]', res)
        if score:
            reward = float(score.group(1).strip())
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def Gen_PRM_reward(question, completions, solution, **kwargs):
    
    contents = [completion[0]["content"] for completion in completions]

    processes = []
    for content in contents:
        content_match = re.search(r'<think>(.*?)</think>', content)
        if content_match:
            process_parsed = content_match.group(1).strip()
            if len(process_parsed) == 0:
                processes.append('empty')
            else:
                processes.append(process_parsed)
        else:
            processes.append('empty')
    
    timeout_duration = 60

    async def async_request(client, prompt):
        try:
            response = await client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                stream=False,
                timeout=timeout_duration,
                temperature=0.1,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API final failure: {e}")
            return '[[0.0]]'

    async def batch_async_requests(client, prompts):
        tasks = [async_request(client, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    RM_PROMPT = '''Please act as an impartial judge and evaluate the quality of the reasoning process provided by AI
        assistants to the user question displayed below. Please assess logical soundness, relevance, thoroughness, and accuracy the reasoning process 
        and provide a score between 0 and 1, where 1 means excellent and 0 means poor. Don't be fooled by a placeholder process like "reasoning process here".
        Just output your final score in this format: "[[your score]]", with no other words or explanation.
        User:
        [Question]
        {question}
        [The Start of Assistant's Reasoning Process]
        {process}
        [The End of Assistant's Reasoning Process]'''

    prompts = [RM_PROMPT.format(question=prob, process=proc) 
           for prob, proc in zip(question, processes)]

    client = OpenAI_ClientSingleton.get_instance()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    responses = loop.run_until_complete(batch_async_requests(client, prompts))
    
    rewards = []
    for res in responses:
        score = re.search(r'\[\[(0?\.\d+|1(?:\.0*)?)\]\]', res)
        if score:
            reward = float(score.group(1).strip())
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern_1 = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_2 = r"^<think>(?:(?!<think>).)*</think>\s*<answer>(?:(?!<answer>).)*</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches_1 = [re.match(pattern_1, content, re.DOTALL) for content in completion_contents]
    matches_2 = [re.match(pattern_2, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match2 else (0.5 if match1 else 0.0) for match1, match2 in zip(matches_1, matches_2)]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
