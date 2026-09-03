# Pod image and execution contract

`vllm-qwen36.Dockerfile` is a proposed image for the later Vast.ai run. The
vLLM `v0.20.0` tag is the compatibility anchor from `TECH_STACK.md`; it is not
an immutable identity. Before building or launching a pod, resolve the image
to a digest and set `BBF_CONTAINER_IMAGE_DIGEST` to the exact `sha256:...`
value. The bootstrap and validation scripts refuse to proceed without that
value and record it in the pod validation artifact. If the provider exposes a
different resolved digest, update `containers/image.lock.yaml` in the run
bundle and retain the original file in the manifest.

Example host-side preparation:

```bash
docker pull vllm/vllm-openai:v0.20.0
docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:v0.20.0
export BBF_CONTAINER_IMAGE_DIGEST='sha256:REPLACE_WITH_RESOLVED_DIGEST'
docker build \
  --build-arg VLLM_IMAGE="vllm/vllm-openai@${BBF_CONTAINER_IMAGE_DIGEST}" \
  -f containers/vllm-qwen36.Dockerfile .
```

Inside the pod, from the repository root:

```bash
export BBF_CONTAINER_IMAGE_DIGEST='sha256:RESOLVED_DIGEST'
./scripts/pod_bootstrap.sh
./scripts/pod_validate.sh
```

Set `BBF_INSTALL_DATA=1` during bootstrap when the authorized Hugging Face
dataset acquisition extra is needed. No script downloads model weights or
performs live data acquisition automatically. The later vLLM command should
use the frozen template, pinned model revisions, BF16, Qwen reasoning/tool
parsers, and an explicitly inspected `--max-model-len` as described in the
experiment plan.
