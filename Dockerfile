#################### BASE BUILD IMAGE ####################
FROM vllm/vllm-openai:v0.7.3 AS vllm-trl

WORKDIR /vllm-workspace

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install https://github.com/huggingface/transformers.git@9f51dc25357bcde280a02b59e80b66248b018ca4

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install flash-attn==2.7.4.post1 --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install trl==0.15.1 wandb==0.19.6 math_verify==0.6.0 deepspeed==0.16.3 wheel==0.37.1 qwen_vl_utils==0.0.10 bitsandbytes==0.45.1


ENTRYPOINT ["/bin/bash"]

#################### BASE BUILD IMAGE ####################





