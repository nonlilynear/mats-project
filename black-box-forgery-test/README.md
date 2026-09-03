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

The scripted backend is only an infrastructure smoke path. A pod run must
explicitly construct the network/vLLM or OpenRouter backend and record its
revisions and environment in the run manifest.

`auxiliary-smoke` is resumable: append-only per-request records are written to
`runs/auxiliary-smoke.jsonl` by default (override with `--results`), while
the summary is written to `--output`. Completed request keys are skipped on a
restart; failed attempts remain in the history. Supplying frozen forgery or
victim-output paths builds requests from those records rather than toy strings.
Use `--max-items` and `--max-requests` to bound a smoke job before starting it.
Each result records the request key, candidate slug, resolved model, provider,
input/output token counts, latency, exact provider cost when returned, and the
conservative budget charge used for stopping.

Live OpenRouter work requires both explicit flags and an environment variable
name; the secret itself is never a command-line argument or artifact field:

```bash
OPENROUTER_API_KEY=... uv run bbf auxiliary-smoke \
  --live --allow-network --api-key-env OPENROUTER_API_KEY \
  --provider <pinned-provider> --budget-usd 5 --request-cost-cap-usd 0.25 \
  --candidates gemini,glm --metadata-output runs/auxiliary-models.json
```

Provider fallbacks are disabled in every live request. Before the bakeoff, a
metadata-only snapshot can be refreshed explicitly with
`bbf auxiliary-metadata --live --allow-network --output PATH` (the alias
`snapshot-auxiliary-metadata` is also available). The snapshot records the
candidate slugs, provider pin, model metadata, and credential environment
variable name, never its value.

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
