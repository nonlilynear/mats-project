# Research source report: black-box CoT Forgery on Qwen3.6 SecOPD

Audience: researcher preparing a preliminary SecOPD sanity check before a separate downstream experiment  
Date: 2026-09-03  
Scope: behavior-only direct-chat and agent/tool-output attacks; excludes activation probes, training, and white-box attacks.

## Direct answer

Use a scripted, resumable vLLM evaluation on one 80 GB GPU, supervised by tmux. Do not use an interactive IPython history as the canonical record. Prioritize the indirect/input-role attack because SecOPD explicitly disclaims jailbreak defense and requires a user/input trust boundary. Treat direct-chat CoT Forgery as an out-of-scope scope control. Add clean utility and an input-vs-user boundary ablation.

## Material evidence and synthesis

The role-confusion paper defines CoT Forgery as a strict zero-shot black-box attack: one user message or tool output, no weight access and no iterative prompt engineering. An auxiliary model generates reasoning-styled text and concatenates it with the harmful query. The chat study uses 313 StrongREJECT items; the agent study uses 100 webpages and standard-versus-forged injection variants. Chat success is judged from the final answer, while agent success is attempted exfiltration. Source: [paper](https://arxiv.org/html/2603.12277v2), sections 3 and Appendix B, arXiv v2, 2026-03-20.

The public repository describes notebooks for generation, inference, judging, and plots, but inspection shows the local runners are GPT-OSS/Harmony-specific and hard-code `/workspace/deliberative-alignment-jailbreaks`. The agent notebook uses live shell execution and public paste services, and its generated CSV/page fixtures are absent. The demo itself says a separate forgery prompt is needed for non-OpenAI model styles. Source: [role-confusion repository](https://github.com/role-confusion/prompt-injection-as-role-confusion), commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`, accessed 2026-09-03.

SecOPD's model card says its trusted instruction must be in `user` and untrusted documents/tool outputs in `input`; it warns against concatenating attacker content into `system` or `user`. It calls itself a model-level indirect prompt-injection defense and explicitly says it is not a replacement for jailbreak defenses. Source: [SecOPD model card](https://huggingface.co/pybbb/Qwen3.6-27B-SecOPD), accessed 2026-09-03.

SecOPD reports base-versus-defense separation on SEP and AgentDojo, but those numbers do not establish CoT Forgery robustness. Its public reproducibility guide uses thinking-enabled user/input role separation, 16K SEP context, 32K AgentDojo context, and output/utility reporting. Source: [SecOPD repository](https://github.com/pppyb/SecOPD), commit `571502a2a315c4b8820dd878d4569e2a2222cb88`, accessed 2026-09-03.

Both Hugging Face model indexes report 55,562,855,904 bytes of tensor data. Their configs are identical, and their checked-out vocab and merges files have identical SHA-256 hashes; the custom input-role template can therefore be frozen and applied to both checkpoints to isolate weights. Sources: [base model](https://huggingface.co/Qwen/Qwen3.6-27B), commit `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`; [SecOPD model](https://huggingface.co/pybbb/Qwen3.6-27B-SecOPD), commit `c68d53540e58eac7fde6b9ec18f72db14dbb92af`, accessed 2026-09-03.

Qwen recommends vLLM 0.19 or newer, thinking mode by default, a Qwen reasoning parser, and offers a text-only serving flag to free memory for KV cache. Source: [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B), accessed 2026-09-03.

The MATS guidance emphasizes a small self-contained investigation, baselines, hypotheses and possible outcomes, sanity checks, raw-data inspection, clarity about limitations, and a strong standalone executive summary. It explicitly prefers one well-supported insight to many superficial experiments and treats well-analyzed negative results as valuable. Source: [MATS application guide](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/preview?pru=AAABoBC1GAk*DajUuyOf8ZFtJzgCg7JalA&pli=1&tab=t.75rygwi582jr); direct preview was inaccessible to the research tool, so the relevant section was corroborated through an indexed mirror/search copy, accessed 2026-09-03.

Vast is a real-time marketplace rather than a fixed-price cloud. Compute, storage, and bandwidth are separate; interruptible instances can pause; storage can continue charging while stopped; and local volumes are tied to a physical machine. Source: [Vast pricing](https://docs.vast.ai/guides/instances/pricing), [instance types](https://docs.vast.ai/guides/instances/choosing/instance-types), and [volumes](https://docs.vast.ai/guides/instances/storage/volumes), accessed 2026-09-03.

## Assumptions and limitations

- The user wants a research plan now, not immediate paid execution.
- A conceptual replication/adaptation is acceptable because the original Qwen-unrelated fixtures are not published.
- A single H100 80 GB can serve a 55.56 GB BF16 checkpoint at moderate context, but this is an engineering estimate that must be smoke-tested; context/concurrency may need reduction.
- Model cards and marketplace details can change. All code/model revisions must be pinned at execution time.
- The Google Doc itself was not machine-readable through its preview URL; relevant application advice was obtained from an indexed copy.

## Claim-to-source ledger

| Claim | Source | Publisher/author | Date | Confidence / notes |
|---|---|---|---|---|
| CoT Forgery is zero-shot, black-box, single-message/tool-output | [Paper](https://arxiv.org/html/2603.12277v2) | Ye, Cui, Hadfield-Menell | 2026-03-20 | High, primary source |
| Chat n=313; agent n=100 pages and two variants | [Paper](https://arxiv.org/html/2603.12277v2) | Ye, Cui, Hadfield-Menell | 2026-03-20 | High |
| Public notebooks and workflow structure | [Repository](https://github.com/role-confusion/prompt-injection-as-role-confusion) | Paper authors | Accessed 2026-09-03 | High, code inspected at SHA |
| SecOPD requires user/input boundary and disclaims jailbreak defense | [Model card](https://huggingface.co/pybbb/Qwen3.6-27B-SecOPD) | SecOPD authors | Accessed 2026-09-03 | High, first-party |
| Qwen vLLM and thinking-mode guidance | [Base model card](https://huggingface.co/Qwen/Qwen3.6-27B) | Qwen | Accessed 2026-09-03 | High |
| SecOPD evaluation context/protocol | [Repro repository](https://github.com/pppyb/SecOPD) | SecOPD authors | Accessed 2026-09-03 | High, code/docs inspected at SHA |
| Checkpoints are each about 55.56 GB | HF model index files at linked model pages | Qwen / SecOPD authors | Accessed 2026-09-03 | High, local metadata inspection |
| Application should prioritize hypotheses, baselines, checks, clarity | [Application guide](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/preview?pru=AAABoBC1GAk*DajUuyOf8ZFtJzgCg7JalA&pli=1&tab=t.75rygwi582jr) | Neel Nanda | Accessed 2026-09-03 | Medium-high; preview inaccessible, indexed mirror used |
| Vast pricing/storage/interruption behavior | [Pricing](https://docs.vast.ai/guides/instances/pricing), [types](https://docs.vast.ai/guides/instances/choosing/instance-types), [volumes](https://docs.vast.ai/guides/instances/storage/volumes) | Vast.ai | Accessed 2026-09-03 | High, first-party/current |

## Search log and stop rationale

Inspected the cited paper HTML and Appendix B; cloned and inspected the role-confusion repository at its current commit; inspected both Hugging Face model cards and metadata-only repository checkouts; cloned and inspected the SecOPD reproducibility code; retrieved the relevant MATS application section from an indexed copy after the Google preview failed; and checked current official Vast pricing/storage/instance documentation plus current OpenRouter availability for the paper's Gemini 2.5 Pro auxiliary model. Research stopped because the experimental scope, protocol differences, deployment boundary, resource estimate, and archival recommendations are supported by primary sources; further broad search is unlikely to change the decision.
