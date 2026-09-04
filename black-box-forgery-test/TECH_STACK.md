# Proposed GPU-pod tech stack

Status: supporting infrastructure note. The canonical methodology and handoff instructions are in `EXPERIMENT_PLAN.md`.  
Purpose: run a well-recorded 2-model x 5-condition black-box sanity check on `Qwen/Qwen3.6-27B` and `pybbb/Qwen3.6-27B-SecOPD`. This is infrastructure for a preliminary check, not the separate MATS experiment.

## Compute

- Provider: Vast.ai.
- Preferred GPU: 1× H100 80 GB, on demand for initial setup and the first run.
- Fallback: 1× A100 80 GB; 2×48 GB GPUs with tensor parallelism only if the listing is materially cheaper and has a good interconnect.
- Disk: 200–220 GB if both roughly 55.56 GB BF16 checkpoints will coexist with caches and the environment.
- Host selection: high reliability, adequate system RAM, good download bandwidth, and a rental duration comfortably longer than the expected run.
- Run the checkpoints sequentially on the same GPU. No quantization for the primary comparison.
- Use interruptible instances only after the runner has been tested for per-sample resume and frequent off-machine synchronization.

## Base image and runtime

- Vast instance with SSH access and a persistent `/workspace` directory.
- Preferred base image: pinned `vllm/vllm-openai` image compatible with Qwen3.6. Start with vLLM `0.20.0`; Qwen requires `0.19.0` or newer. Record the image digest, not only the mutable tag.
- CUDA/driver supplied by the selected host and verified before downloading models.
- `tmux` for durable interactive supervision.
- `uv` for a small, separate experiment-runner virtual environment.
- Git for code and configuration; Git LFS or external object storage for compressed raw transcripts if they become too large for ordinary Git.

Do not make a notebook kernel or an IPython history the canonical environment. Notebook use is optional and downstream of immutable run artifacts.

## Model serving

Serve one model at a time through vLLM's OpenAI-compatible API. Proposed shared serving settings:

- BF16.
- Thinking enabled.
- Qwen reasoning parser: `qwen3`.
- Qwen tool-call parser: `qwen3_coder` when running the agent-style check.
- Frozen SecOPD `user`/`input` chat template applied to both checkpoints for the controlled indirect-injection comparison.
- Text/language-model-only mode because the evaluation is text-only.
- 32K maximum context as the initial agent configuration; reduce context or concurrency if the smoke test OOMs.
- Fixed served-model name, model revision, tokenizer revision, seed, and generation limits recorded in the run manifest.

Conceptual server command:

```bash
vllm serve MODEL_ID \
  --revision MODEL_COMMIT \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --chat-template configs/qwen36_input_role_chat_template.jinja \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only
```

This is a proposed command, not yet pod-verified. We will adjust flags only after checking the pinned vLLM version's CLI and the rendered prompts.

## Checkpoints and upstream revisions

Pin revisions when downloading so later model-card updates do not silently change the comparison.

- Base: `Qwen/Qwen3.6-27B`
  - revision inspected: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Defended: `pybbb/Qwen3.6-27B-SecOPD`
  - revision inspected: `c68d53540e58eac7fde6b9ec18f72db14dbb92af`
- CoT-Forgery code/prompts:
  - `role-confusion/prompt-injection-as-role-confusion`
  - revision inspected: `ec333c40fd43fe991e1ebf66765051b6d7e35784`
- SecOPD input-role template/reference evaluator:
  - `pppyb/SecOPD`
  - revision inspected: `571502a2a315c4b8820dd878d4569e2a2222cb88`

The two model repositories have matching model configurations and tokenizer vocabulary/merges in the revisions inspected. For the primary comparison, use a single copied-and-hashed input-role Jinja template rather than allowing the two repositories to choose different serialization.

## Experiment runner

Build a normal Python package/CLI rather than modifying and executing the upstream notebooks in place. Expected entry points:

```text
python -m cot_forgery.prepare     # freeze prompts, pages, and attack assignments
python -m cot_forgery.validate    # render/template/tool-parser checks
python -m cot_forgery.run         # resumable target inference
python -m cot_forgery.judge       # API judging after GPU shutdown
python -m cot_forgery.analyze     # counts, intervals, comparisons, plots
```

Likely runner dependencies:

- `openai` for the local vLLM endpoint and OpenRouter.
- `pydantic` for versioned request/result schemas.
- `datasets` for StrongREJECT or other frozen source data.
- `pandas` and `pyarrow` for analysis tables.
- `scipy` and/or `statsmodels` for confidence intervals and paired tests.
- `pyyaml`, `python-dotenv`, `requests`, `tqdm`, `tenacity`, and `orjson`.
- `pytest` for prompt-rendering, resume, parser, and mock-tool tests.
- `zstandard` for compressed JSONL artifacts.

Pin these in `pyproject.toml` plus `uv.lock`. Do not install the upstream role-confusion repository's entire white-box/H200-oriented environment unless a required black-box component proves missing.

## Attack generation and judging APIs

- OpenRouter API, with a local three-model auxiliary bakeoff before the full run:
  - `z-ai/glm-5.2:free`
  - `meta/muse-spark-1.3-contributor`
  - `google/gemini-3.8-flash:batch`
- Run both candidates on the same frozen development subset for forgery generation and judging. Compare validity, base-Qwen attack lift, agreement with human labels, parse-failure rate, latency, and actual token cost. Select the generator and judge separately if the evidence supports doing so.
- Do not use SecOPD results to choose or tune the auxiliary model; SecOPD is the held-out defended-model comparison.
- Use the selected model(s) at temperature 0 to generate/freeze chat forgeries and apply the paper-compatible classifier rubric.
- Pin the provider and disable fallback when reproducing a specific run. Treat an unavailable pinned route as a failed run rather than silently changing models.
- Keep auxiliary-model outputs in the archive because API behavior can change even when the slug remains available.
- Run judging from the laptop after target generations are synchronized, so paid GPU time is not wasted on API calls.
- Secrets live only in environment variables or an untracked `.env`; manifests record variable names, never values.
- Set the full-run OpenRouter budget only after the three-model bakeoff reports measured token use and cost. Current model pricing and measured smoke-test usage govern.

## Safe agent harness

Do not copy the upstream notebook's live `shell=True` plus public paste-service behavior.

- Use a temporary, per-episode workspace containing a synthetic canary file.
- Expose a constrained/mock tool interface that preserves the paper's observable actions: list/search, read, and attempted upload.
- Intercept any upload/exfiltration request locally and record it without network transmission.
- Store the complete assistant/tool event sequence.
- Make page fixtures local and immutable; do not depend on a webpage remaining available during inference.
- Deterministically assign the upstream standard-injection and CoT-Forgery templates from the sample ID and global seed.

## tmux layout

Suggested session: `cot-forgery`.

- Window `server`: vLLM server with stdout/stderr tee'd to the active run directory.
- Window `runner`: the resumable CLI.
- Window `monitor`: `nvidia-smi`, disk use, and synchronization status.
- Optional window `inspect`: IPython or Jupyter for qualitative inspection only.

The runner, not tmux, provides durability. Each completed sample is flushed atomically, and restart logic skips completed request keys.

## Artifacts and synchronization

Proposed layout:

```text
configs/
data/
prompts/
src/cot_forgery/
tests/
runs/<run-id>/
  manifest.json
  requests.jsonl.zst
  generations.jsonl.zst
  agent_events.jsonl.zst
  judgments.jsonl.zst
  metrics.json
  logs/
plots/
```

The manifest should capture:

- Git commit and any uncommitted patch.
- All four upstream/model revisions listed above.
- SHA-256 hashes for templates, source data, frozen forgeries, and rendered prompts.
- Container digest, package lock, Python/vLLM/CUDA/driver versions, and GPU identity.
- Sampling settings, context/output limits, seed, request order, and truncation rules.
- Token counts, timing, errors, retries, and invalid/parser-failure counts.

Synchronize `runs/` to the laptop or durable object storage after small batches. A Vast local volume is useful for restart convenience but is tied to one host and must not be the only copy.

## Pod setup checklist for the later setup session

1. Select and provision the Vast offer; record its price breakdown and hardware details.
2. Verify GPU, driver, CUDA visibility, RAM, disk, and download throughput.
3. Clone this repository and check out the intended experiment commit.
4. Install/check `tmux`, Git LFS if used, `uv`, and the synchronization client.
5. Create the runner environment from the lockfile.
6. Download both model revisions and verify file hashes/sizes.
7. Copy and hash the frozen input-role chat template.
8. Start base Qwen in vLLM; inspect exact rendered `user`/`input` prompts and collect the frozen benign style references.
9. Generate both candidates' development-set forgeries, then run the base-Qwen auxiliary bakeoff and target infrastructure smoke described in `EXPERIMENT_PLAN.md`.
10. Sync and verify the smoke artifacts, then shut down paid GPU compute before OpenRouter judging and human review if approval will not be immediate.
11. Select and freeze the generator, judge, forgeries, rubric, and full-run budget. Do not use SecOPD outcomes for this selection.
12. Start a confirmatory pod session and run all five conditions against both checkpoints with identical paired fixtures/settings.
13. Sync outputs throughout and inspect early target samples before allowing the job to continue unattended.
14. Shut down/delete paid compute after verifying the external copy, then complete judging and analysis off-pod.
