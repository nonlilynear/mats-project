# Black-box Role Confusion evaluation

This directory contains the reproducible Stage A scaffold described in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). It is CPU-only and offline by
default; no model weights, credentials, live webpages, or public upload
endpoints are needed for local validation.

```bash
uv sync --extra dev
uv run bbf prepare-data
uv run bbf validate-fixtures
uv run bbf auxiliary-smoke
uv run bbf auxiliary-smoke --forgeries runs/smoke/forgeries.jsonl --victim-outputs runs/smoke/victim_outputs.jsonl
uv run bbf run-target --model base --config configs/pilot.yaml
uv run bbf verify-archive --run-dir runs/local-base-pilot
uv run bbf judge --run-dir runs/local-base-pilot
uv run bbf export-audit --run-dir runs/local-base-pilot --fraction 0.10
uv run bbf analyze --run-dir runs/local-base-pilot
uv run bbf build-review-bundle --run-dir runs/local-base-pilot
uv run pytest
```

## SecOPD-adapted role probes

The role-probe follow-up uses the exact Qwen/SecOPD renderer from this
repository and trains one probe per checkpoint. The role space is
`system,user,cot,assistant,input,tool`: `input` is a distinct SecOPD data
role, while `tool` means the template's `<tool_response>` serialization. This
keeps the probe faithful to the application boundary instead of treating all
external data as the paper's tool role.

The training corpus must be neutral, non-instructional text and must not
contain target pages, injections, or attack outcomes. Materialize the
paper-style C4/Dolma3 sample once, then use the frozen local JSONL:

```bash
uv run --extra role-probes python scripts/materialize_probe_texts.py \
  --output data/probe/neutral.jsonl \
  --manifest data/probe/neutral.manifest.json \
  --allow-network
```

This uses the public GPT-OSS notebook's 50/50 C4/Dolma3 sampling pattern,
with seed 123 and 250 deduplicated rows by default. The manifest records the
dataset identifiers, Dolma3 revision, sampling parameters, and JSONL hash.
After materialization, use the frozen file as the probe input:

```bash
uv run --extra role-probes python scripts/train_role_probes.py \
  --model pybbb/Qwen3.6-27B-SecOPD \
  --revision c68d53540e58eac7fde6b9ec18f72db14dbb92af \
  --tokenizer-revision c68d53540e58eac7fde6b9ec18f72db14dbb92af \
  --texts-jsonl data/probe/neutral.jsonl \
  --output-dir runs/role-probes/secopd
```

For the 27B pod run, use the GPU-native fitter after activation extraction;
50 epochs matched the paper-style CPU smoke fit closely while avoiding the
multi-hour CPU solver:

```bash
uv run --extra role-probes python scripts/train_role_probes.py \
  --model pybbb/Qwen3.6-27B-SecOPD \
  --texts-jsonl data/probe/neutral.jsonl \
  --output-dir runs/role-probes/secopd-layer32 \
  --layers 32 --batch-size 1 \
  --fit-backend torch --fit-epochs 50 --fit-device cuda
```

The training script also supports streamed C4/Dolma3 input via
`--dataset both --allow-network`, but the frozen snapshot is preferred for the
primary run. It extracts pre-MLP `post_attention_layernorm` activations, splits
by base text before fitting, and writes `probes.pkl`, `accuracy.json`, and
`metadata.json`. The explicit `--allow-hidden-state-fallback` option is a
documented deviation and should not be used for the primary run.

Use `--layers auto --fit-backend torch` for the neutral layer-selection sweep;
this evaluates every fourth decoder layer in one activation pass. Select the
best layer using held-out neutral accuracy, then optionally refine adjacent
layers before projecting any attack or contradiction cases. The current
SecOPD sweep selects layer 56 (0.809 held-out accuracy) from the 52--60
refinement.

The paper's second validity gate is zero-shot recovery on genuine,
correctly-tagged conversations. Build the frozen validation set from six
clean agent traces (`user`, `input`, and SecOPD-generated `assistant`) and 12
non-forged SecOPD reasoning traces (`cot` only), then evaluate the frozen
layer-56 probe:

```bash
python scripts/prepare_zero_shot_role_eval.py \
  --output runs/role-probes/secopd-zero-shot/inputs.jsonl

uv run --extra role-probes python scripts/evaluate_role_probe_zero_shot.py \
  --inputs-jsonl runs/role-probes/secopd-zero-shot/inputs.jsonl \
  --probe-artifact runs/role-probes/secopd-layer-refine-52-60/probes.pkl \
  --model pybbb/Qwen3.6-27B-SecOPD \
  --revision c68d53540e58eac7fde6b9ec18f72db14dbb92af \
  --layer 56 --output-dir runs/role-probes/secopd-zero-shot --plot
```

This probe passes zero-shot recovery for User (91.2% mean Userness) and
Assistant (99.8% mean Assistantness), but not for genuine CoT (33.0%
CoTness) or Input (7.5% Inputness) under the primary fixed-length summary.
The failure is consistent across traces, and the all-token sensitivity check
is weaker still for CoT and Input. System and Tool are not evaluated because
the frozen runs contain no genuine held-out spans for those roles. Therefore
the six-way layer-56 probe does **not** pass the paper's full validity gate;
attack-span role labels involving CoT/Input must remain provisional pending a
validated probe construction or layer selection. The current Qwen wrapper
also leaves target position correlated with role (early User/Input, later CoT
and Assistant), contrary to the paper's matching-filler positional control;
this is the leading explanation for the synthetic-to-real gap and must be
removed before retraining. Results and the confusion matrix are in
`runs/role-probes/secopd-zero-shot/`.

To project that frozen probe onto the ten-page agent prompts, first materialize
the four aligned scenarios. The input builder reuses the frozen request
records, preserves page/injection/forgery character spans, and records when a
malicious forged-CoT sample is reused for one of the four pages without a
page-specific forgery:

```bash
python scripts/prepare_role_projection_inputs.py \
  --output runs/role-probes/secopd-real-prompt-inputs.jsonl
```

The activation-only projection pass does not generate text or execute tools:

```bash
uv run --extra role-probes python scripts/project_role_probes.py \
  --inputs-jsonl runs/role-probes/secopd-real-prompt-inputs.jsonl \
  --probe-artifact runs/role-probes/secopd-layer-refine-52-60/probes.pkl \
  --model pybbb/Qwen3.6-27B-SecOPD --revision c68d53540e58eac7fde6b9ec18f72db14dbb92af \
  --layer 56 --output-dir runs/role-projections/secopd-layer56-10pages-4scenarios
```

Each output row reports mean role probabilities, top-role fractions, and
entropy separately for the page, appended injection, and forged-CoT spans.
The current ten-page run is saved under
`runs/role-projections/secopd-layer56-10pages-4scenarios/`.

For paper-style token traces, rerun the projection with
`--save-token-traces --trace-page-tail-tokens 200`, then plot with:

```bash
uv run --extra role-probes python scripts/plot_role_traces.py \
  --projections-jsonl runs/role-projections/secopd-layer56-10pages-4scenarios-traces/projections.jsonl \
  --output-dir runs/role-projections/secopd-layer56-10pages-4scenarios-traces/figures
```

The figures show Userness and CoTness for the pre-generation prompt trace.
Generated assistant-token activations require a separate generation-time
capture; the existing rollout records store final text but not those hidden
states.

The scripted backend is only an infrastructure smoke path. A pod run must
explicitly construct the network/vLLM or Fireworks backend and record its
revisions and environment in the run manifest.

Once a compatible vLLM server is running on the pod, the target runner uses
the local OpenAI-compatible endpoint. Both flags are required so a normal
smoke command cannot make a network request accidentally:

```bash
uv run --frozen bbf run-target --model base --config configs/pilot.yaml \
  --run-dir runs/pod-base-pilot --max-items 1 \
  --live --allow-network --base-url http://localhost:18000
```

The live agent path keeps webpage content in the constrained local toolbox;
the model endpoint receives messages and tool definitions, but the model has
no shell or public-network tool. The vLLM server must be started separately
with the checked-in Qwen3.6 chat template and the appropriate tool-call parser.
The full configuration additionally requires the pod endpoint to expose
`GET /bbf/template-contract`, returning the checked-in contract fields and
`"verification_method": "endpoint_contract"`; the runner fails closed when
that attestation is absent or mismatched. Pilot runs record an unverified
server-template status unless this check is explicitly requested.

`auxiliary-smoke` is resumable: append-only per-request records are written to
`runs/auxiliary-smoke.jsonl` by default (override with `--results`), while
the summary is written to `--output`. Completed request keys are skipped on a
restart; failed attempts remain in the history. Supplying frozen forgery or
victim-output paths builds requests from those records rather than toy strings.
Use `--max-items` and `--max-requests` to bound a smoke job before starting it.
Each result records the request key, candidate slug, resolved model, provider,
input/output token counts, latency, exact provider cost when returned, and the
conservative budget charge used for stopping.

The canonical auxiliary model is DeepSeek V4 Flash through Fireworks. Live
Fireworks work requires both explicit flags and an environment variable name;
the secret itself is never a command-line argument or artifact field:

```bash
FIREWORKS_API_KEY=... uv run bbf auxiliary-smoke \
  --live --allow-network --transport fireworks \
  --api-key-env FIREWORKS_API_KEY --budget-usd 5 --request-cost-cap-usd 0.25 \
  --candidates accounts/fireworks/models/deepseek-v4-flash-0731 \
  --results runs/deepseek-fireworks-results.jsonl
```

The command above is illustrative; provider fallbacks are disabled in every
live request. The current DeepSeek result is only a three-case probe; it is not
the frozen 313-chat/100-page forgery corpus needed for the confirmatory run.
The current metadata-snapshot CLI is OpenRouter-specific, so Fireworks model
metadata and pricing must be captured from the live result records until that
provenance path is generalized. Never store the credential value in a
manifest, log, or artifact.

After a human reviews `review/audit_queue.jsonl`, write one JSON object per
line to `human_labels.jsonl` (or pass `--human-labels PATH`):

```json
{"audit_id":"blind-...","episode_id":"...","human_label":"REFUSAL","human_valid":true,"adjudication_note":"..."}
```

`human_label` and its notes are kept separate from automated judgments. The
review bundle consumes both namespaces and writes `review/summary.json`,
`review/summary.md`, `review/disagreements.jsonl`, and `review/bundle.json`.
Use `--refresh-seal` on `build-review-bundle` (or run
`bbf seal-archive --run-dir RUN`) explicitly after derived outputs are ready;
the bundle command reports an existing seal as `stale` rather than silently
rewriting it.

Real data acquisition is opt-in and never runs during tests. To reproduce the
Wikipedia sampler, install the data extra and provide the explicit network
flag:

```bash
uv sync --frozen --extra data
uv run --frozen bbf acquire-wikipedia --allow-network \
  --output data/source/wikipedia-20231101-en --count 100
```

The command uses the frozen `wikimedia/wikipedia` `20231101.en` streaming
recipe, seed `1234`, shuffle buffer `1000`, and a hard 100 KiB HTML limit. It
records source IDs, URLs, fetch timestamps, byte counts, and hashes. The
output is a private snapshot; review Wikimedia terms before redistribution.

The canonical `walledai/StrongREJECT` Hugging Face mirror is MIT-licensed but
auto-gated. Accept its access condition in the Hugging Face account used by
this machine, then pull the exact pinned revision below. The import writes the
313-row snapshot to the ignored private data area and refuses to run without
both network and authorization opt-ins:

```bash
uv run --frozen bbf import-strongreject \
  --dataset-id walledai/StrongREJECT \
  --revision df54272b395ce9e584d2f8140519935425d8549f \
  --allow-network --authorized
uv run --frozen bbf freeze-development \
  --harmful-snapshot data/source/strongreject/strongreject.jsonl \
  --harmful-dataset-id walledai/StrongREJECT \
  --harmful-revision df54272b395ce9e584d2f8140519935425d8549f \
  --pages-index data/source/wikipedia-20231101-en/pages.jsonl \
  --authorized
```

The development manifest stores only the deterministic 24-harm/12-page IDs;
source prompts remain in the authorized snapshot. Pod image resolution and
noninteractive validation are documented in [`containers/README.md`](containers/README.md):
the scripts require and record a resolved image digest before proceeding.
