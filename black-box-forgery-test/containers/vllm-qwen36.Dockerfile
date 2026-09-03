# Proposed pod image. The mutable tag is a compatibility anchor only; build
# and record a resolved @sha256 digest before execution.
# Example: docker build --build-arg VLLM_IMAGE=vllm/vllm-openai@sha256:<digest> .
ARG VLLM_IMAGE=vllm/vllm-openai:v0.20.0
FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="black-box-forgery-qwen36-runner"
LABEL org.opencontainers.image.description="Pinned vLLM base plus reproducible experiment runner"

WORKDIR /workspace/black-box-forgery-test
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# uv is pinned separately from the vLLM image tag. The pod manifest records
# both this version and the resolved base-image digest.
RUN python -m pip install --no-cache-dir uv==0.11.19 \
    && uv sync --frozen --extra dev

ENV PYTHONUNBUFFERED=1
ENV BBF_OFFLINE_DEFAULT=1
