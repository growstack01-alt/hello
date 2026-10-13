# Fine-tuning Qwen3.8-27B as the brain of a custom Hermes-based marketing-operations system

**Research-backed plan — research dates 2026-10-12 / 2026-10-13**

> **How to read this document.** It was written in three passes and the base model changed on 2026-10-13 from Qwen3.6-27B to **Qwen3.8-27B** (your decision; same weights architecture, different chat template).
> - **Part III is the execution plan** — start there. It covers Qwen3.8-27B specifics, how to capture and create training data for a custom multi-agent Hermes wrapper, representation of agents/workflows/long sessions, the training stack, evaluation, costs and timeline.
> - **Part I** is the foundation (what goes in weights vs context, methodology evidence, data pipeline, training recipe, infra, governance). It was written for Qwen3.6-27B; every architecture/VRAM/cost statement applies unchanged to Qwen3.8-27B, but **template rendering rules are superseded by Part III §III.1**, and the note in Part I §4.3 about Hermes' trajectory export is corrected in Part III §III.2.
> - **Part II** is the Holo (H Company) computer-use research and the decision on how to add GUI capability.

> **Confidence legend.** Six parallel research threads were run against primary sources where reachable (GitHub repos, model cards via verbatim mirrors, trainer docs, the Kimi K2 tech-report PDF, Hermes Agent source at commit `3f2abce7`). This sandbox blocks arxiv.org and huggingface.co, so paper numbers marked **[S]** come from abstracts/search snippets, and **[U]** could not be verified. Unmarked statements were read from a primary source.

---

## 1. The five decisions (executive summary)

1. **Fine-tune *behaviour*, not *knowledge*.** Weights get: tool-call discipline on your tool shapes, multi-step procedure reliability, brand voice, analytical habits over marketing data, approval/safety behaviour, and correct use of Hermes' `skill_view`/`memory` tools. Company facts, products, pricing, brand-book *content*, campaign history and vendor tool schemas live in Hermes **Skills**, **memory**, and **retrieval/MCP**. Evidence: fine-tuning on new facts is learned slowly and *linearly increases hallucination* (Gekhman et al., EMNLP 2024); RAG beats fine-tuning for both known and new knowledge (Ovadia et al., EMNLP 2024); every successful enterprise case (Shopify Sidekick, Cursor Tab, Pinterest, Perplexity Sonar) fine-tuned behaviour and kept volatile knowledge outside the weights.
2. **Keep Qwen3.6's native tool grammar — the "Hermes wrapper" is format-agnostic.** Hermes Agent sends OpenAI-style `tools` and consumes structured `tool_calls` from the inference server; it has no text parser for `<tool_call>`. Qwen3.6 emits Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`), *not* Hermes JSON, and is served with `--tool-call-parser qwen3_coder`. Do not retrain the grammar; make template ↔ training data ↔ server parser agree.
3. **Train in the harness, on the harness's prompt surface.** Hermes puts ~1.4–2.7k tokens of system prompt plus ~13k tokens of tool schemas (33 tools) in front of every conversation and requires ≥64K context. Generate SFT/RL data by running Hermes Agent itself (`batch_runner.py`) against your sandboxed marketing tools, and evaluate inside Hermes with the exact vLLM flags you will ship.
4. **Recipe:** baseline-in-harness → LoRA SFT (all-linear *including* the Gated-DeltaNet projections) → on-policy distillation from the original checkpoint to recover general ability → DPO/KTO → GRPO with gated, decomposed rewards in a seeded sandbox → merge → FP8 → eval the served artifact. Escalate to full fine-tuning only if long multi-step *procedure* evals plateau under LoRA (a July 2026 result shows exactly that failure mode).
5. **Gate every release on pass^k** (all k trials succeed) in a seeded sandbox, **zero** side-effect-policy violations (no publish/spend/send without approval), and a general-capability regression suite within ±1–2 points of base.

---

## 2. Ground truth on your two components

### 2.1 Qwen3.6-27B (`Qwen/Qwen3.6-27B`, released 2026-04-22, Apache-2.0)

| Fact | Implication for you |
|---|---|
| **Hybrid architecture**: 64 layers = 16 × (3 Gated DeltaNet linear-attention + 1 Gated Attention); hidden 5120; vocab 248,320; 4 KV heads; native vision tower; MTP head | LoRA must target `linear_attn.in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj` as well as `q/k/v/o_proj` + MLP (a stock `q,k,v,o` config adapts only part of the network). Freeze/skip the vision tower (`--language-model-only` at serve time). 248k vocab means materialized logits at 32k tokens ≈ 32 GB fp32 → use fused/chunked cross-entropy (Liger FLCE, Axolotl Cut-Cross-Entropy). |
| Only **one post-trained checkpoint** — no base/instruct/thinking split | You are fine-tuning an already-RL'd agentic model. Low LR, replay, and recovery distillation matter more than usual. |
| **Thinking on by default**, `<think>` prefilled by the template; `enable_thinking:false` is the only switch (**no** `/think` `/no_think` soft switch); `preserve_thinking` option retains historical reasoning ("particularly beneficial for agent scenarios") | Pick one thinking regime and keep it identical across SFT data, RL rollouts and serving. |
| Native **262,144** context; card advises ≥128K to preserve thinking | Serve at 128K if VRAM allows; Hermes needs ≥64K. |
| Official serving: `vllm>=0.19`, `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`; SGLang same parser names | `hermes` parser will **not** parse this model. |
| Agentic scores on the card: SWE-bench Verified 77.2, Terminal-Bench 2.0 59.3, SkillsBench 48.2, Claw-Eval pass^3 60.6, MMLU-Pro 86.2, GPQA 87.8 | Strong starting point; your job is domain reliability, not raw capability. |
| **Qwen3.8-27B** (2026-08-14, Apache-2.0) is the same architecture with newer agentic post-training, `reasoning_effort`, `preserve_thinking` default-on, and wider framework support (Unsloth notebook, LLaMA-Factory `qwen3_8`, Tinker — which is retiring Qwen3.6-27B) | Your whole pipeline transfers unchanged. Run the Stage-0 baseline on both and pick by your eval; don't pick on principle. |

Known serving bugs to pin around: `<tool_call>` emitted inside `<think>` (fixed in vLLM PR #35687, 2026-04-24 → use a build newer than that); phantom tool calls from quoted markup in reasoning (vLLM #58147, open); empty tool calls under a Claude-Code-style harness (QwenLM #150). The official template rejects a `developer` role.

### 2.2 Hermes Agent (NousResearch/hermes-agent, MIT, v0.21.7 on 2026-10-11)

| Fact (read from source/docs) | Implication |
|---|---|
| Talks to models via `chat_completions` (OpenAI-compatible), `anthropic_messages`, or Responses API; **tool calling is API-level** — the loop branches on `response.tool_calls`; "Without these flags, tool calls won't work — the model will output tool calls as text" | vLLM/SGLang parser is the only place grammar matters. |
| `MINIMUM_CONTEXT_LENGTH = 64_000`; compression fires at 50% of window; prompt surface ≈ **1,445–2,640 tokens stable + 1,813 volatile + 13,264 tokens of tool schemas for 33 tools** (`hermes prompt-size` prints it) | Your training samples must carry a realistic system prompt + schema block, long multi-turn tool loops, parallel `tool_calls`, and `tool`-role results. |
| **Model-name gating**: if the served model id contains `qwen`, Hermes injects "# Tool-use enforcement" + execution-discipline blocks (the longer `cli.gpt`-style prompt) | Name the served model something custom (e.g. `acme-marketing-27b`) or pin `agent.tool_use_enforcement`/`agent.execution_guidance`; then train against *that* prompt variant and keep it fixed. |
| Replaying history to a **custom endpoint strips prior-turn `reasoning`/`reasoning_content`** (only OpenRouter/Nous Portal routes carry it back) | Mismatch with Qwen3.6, which keeps intra-episode thinking for tool-call steps. Either patch Hermes (MIT) to pass `reasoning_content` back on the custom route, or render your SFT data with reasoning stripped from all prior turns. Decide before generating data (see §2.3). |
| Documented warning: vLLM `--tool-call-parser qwen3_xml` + a reasoning parser can leak tool markup into text with zero `tool_calls` under streaming; workaround `model.streaming: false` | Test both `qwen3_coder` and `qwen3_xml` on 200 held-out traces, streaming on. |
| **Skills** = `SKILL.md` folders (agentskills.io-compatible), progressive disclosure (`skills_list` ~3k tokens → `skill_view`), agent-managed via `skill_manage`, `skills.auto_load` pins skills into every session. **Memory** = `MEMORY.md` (2,200 chars) + `USER.md` (1,375 chars) frozen at session start, plus providers (Honcho, Mem0, …). System prompt tells the model "procedures and workflows belong in skills… memory is the narrow exception for facts that apply to EVERY session" | This is where "knows the company" lives (§3). Train the model to *use* these tools correctly. |
| Ships **trajectory generation for training**: `batch_runner.py` (toolset distributions, hallucinated-tool filter, zero-reasoning filter), `trajectory_compressor.py --target_max_tokens`, ShareGPT output with `<think>`, Hermes-JSON `<tool_call>`, grouped `<tool_response>` | Your data factory. You must convert its Hermes-JSON tool calls to OpenAI `tool_calls` (dict `arguments`) and let the Qwen3.6 template render XML. |
| Nous' own guidance: Hermes 4 chat models are "not recommended for use inside Hermes Agent"; it recommends frontier agentic models. **Atropos is archived** (2026-07-04); live successors are **NousResearch/Gym** (NeMo Gym fork with a `hermes_agent` harness) and **NousResearch/RL** (NeMo RL fork) | Expect to do real work to make a 27B a reliable brain; use Gym + NeMo RL or verl for RL *through* the harness. |

### 2.3 Compatibility checklist — do these before generating a single training sample

1. Serve stock Qwen3.6-27B: `vllm serve Qwen/Qwen3.6-27B --max-model-len 131072 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only`. Point Hermes at it (`model.provider: custom`, `base_url`, `context_length: 131072`, custom model name). Run 20 multi-tool tasks with streaming on; confirm `tool_calls` are structured and final answers land in `content`.
2. Byte-compare: render 10 Hermes trajectories through the official template (TRL's vendored `qwen3_6.jinja` is the byte-exact original) vs what vLLM prefills. Fix any divergence (tools-first system block, `tool` results merged into one `user` turn, dict arguments, `<think>` prefill).
3. Decide the reasoning-replay policy (patch Hermes vs strip in data) and the thinking regime (thinking **on** with a length budget is my recommendation — Qwen3.6's agentic numbers assume it).
4. Install kernels or training silently degrades: `flash-linear-attention`, `causal_conv1d`, `flash-attn` (or Unsloth's vendored GDN kernels). Fine-tune from the **BF16** checkpoint, not `-FP8` (Unsloth issue #6200 produced a "garbage base model" from FP8).

---

## 3. Where each kind of "knowing the company" lives

| What you want the agent to "know" | Mechanism | Why (evidence) |
|---|---|---|
| Company facts, product catalogue, pricing, brand-book content, approved-claims library, campaign history, platform policies | **Retrieval + MCP resources**; hybrid pre-load of the stable core (voice brief, claims rules) | Gekhman 2024; Ovadia 2024; EDPB Opinion 28/2024 makes customer PII in weights a legal burden you can avoid by retrieving under the user's own OAuth scope. Changes faster than a training cycle. |
| Repeatable processes: campaign launch runbook, naming/UTM conventions, QA + compliance checklist, reporting templates, approval routing, "how we do X here" | **Hermes Skills** (`SKILL.md`, versioned in git, `skills.auto_load` for always-on playbooks); evolve them with `hermes-agent-self-evolution` (GEPA, ~$2–10/run, no GPU) | Anthropic: a skill is "an onboarding guide for a new hire"; Rakuten/Box/Canva adoption; Nous' own division of labour. Auditable, editable, no retrain. |
| Per-marketer / per-account preferences and running state | **Memory** (`MEMORY.md`/`USER.md`, or Honcho/Mem0 provider) | Designed for exactly this; scope per user; never store PII the agent doesn't need. |
| *How we work*: tool-argument discipline, when to ask vs act, drafts-before-publish, voice, analytics habits, output formats your pipeline consumes | **Weights (SFT → preference → RL)** | These are habits that must hold at pass^k and that prompting does not reliably produce (CRMArena-Pro multi-turn collapse, Shopify's hallucinated IDs/enums, τ-bench pass^k decay). |
| Hard enforcement: spend caps, publish/send gates, allow-lists, PII egress | **Harness** (separate `*_draft` vs `publish_*` tools, read-only OAuth by default, destructive/`requiresUserInteraction` annotations, pre-tool hooks, audit log) | OWASP LLM06; MCP spec ("SHOULD always be a human in the loop"); the model is never the enforcement layer. |

**Your internal LLM usage logs are the single most valuable company-specific asset** — they are the real task distribution. Mine them (§4.6) rather than hand-writing "about the company" data.

**If you still want facts in weights** (e.g. slow-changing taxonomy, company history): the only method with evidence is EntiGraph-style *synthetic continued pretraining* — 1.3M real tokens were expanded to **455M** synthetic tokens of entity-graph rewrites to lift closed-book QA 39.5%→56.2%, log-linear in synthetic tokens; recipe LR 5e-6, **10% general replay**, 2 epochs (Yang et al., ICLR 2025 [S]). Plain paraphrasing saturates at ~2M tokens. Even then, injected facts are poorly used in multi-hop reasoning ("Knowing–Using gap", Jul 2026 [S]). Treat as a Phase-3 experiment, never Phase 1.

---

## 4. Data pipeline

### 4.1 Environment (weeks 0–2)
- Wrap each marketing system as an MCP server with JSON-Schema tools; follow Anthropic's tool-design guidance (consolidate workflows, return human-readable fields not UUIDs, paginate/truncate, actionable errors, `readOnlyHint`/`destructiveHint`). Add 1–5 `input_examples` per tool with tricky parameters (72%→90% complex-parameter accuracy in Anthropic's data).
- **Split every write-capable integration** into `create_*_draft` and gated `publish_*`/`set_budget` (hard cap) tools. Official vendor servers are mostly read-only anyway (Google Ads MCP: 3 read tools; GA4: 7 read tools), while Klaviyo/Meta/HubSpot expose writes — your gate must be yours.
- Build **seeded sandbox tenants** (synthetic contacts, campaigns, budgets, analytics rows) with snapshot/reset so every task is replayable and pass^k is computable (τ-bench/MCPMark pattern: initial state + `verify()`).
- Fix a tool-output truncation budget and marker now (recommend 2–4k tokens head+tail) and use the *same* in training and serving.
- Enable Hermes Tool Search for large MCP tool sets and train on the `tool_search`/`tool_describe`/`tool_call` bridge pattern — Anthropic measured −85% schema tokens and +25 points MCP-eval accuracy from deferred tool loading.

### 4.2 Task blueprints with rubrics (APIGen-MT + Kimi K2 pattern)
~20 personas × ~15 workflows (campaign setup, segmentation, subject-line tests, budget reallocation, performance reporting, content calendar, brand-compliance review, attribution questions). For each: intent, ground-truth action sequence, expected end state, and a **rubric** (success criteria, expected tool pattern, policy constraints such as "never send without approval"). Validate blueprints by execution + an LLM reviewer committee (≤3–5 rounds). Target **3–5k verified blueprints**. Marketing *analytics* tasks are verifiable (the right number is computable from the seeded rows) — that is your cleanest RL signal later.

### 4.3 Trajectory generation (Hermes `batch_runner.py` + Toucan/Kimi K2 pattern)
- Teacher = a frontier agentic model running **inside Hermes Agent** against the real sandbox tools (`--distribution` to vary toolsets, `--ephemeral_system_prompt` for teacher-only hints, `--max_turns`), persona-conditioned user simulator with best-of-N. **Correction (Part III §III.2):** use `batch_runner` only to *drive* runs — its trajectory export substitutes a fixed Hermes-JSON tool-calling system prompt for the real one, re-renders tool calls as JSON and records no token IDs. Record the training data at a capturing gateway under your wrapper.
- **Teacher licensing is a real decision**: check your provider's terms before distilling from Claude/GPT/Gemini into an in-house model. Open-weight teachers (Qwen3.5-397B-A17B, DeepSeek V4, GLM-4.7, Kimi K2.x) avoid the question and also enable on-policy distillation (which needs teacher logprobs). Kimi K2's recipe — 3,000+ real MCP tools + 20,000+ synthetic tools, persona users, a stateful simulator injecting controlled failures, rubric-judge rejection sampling — is reproducible at your scale.
- Inject tool failures/partial failures in ~15–25% of runs and **keep the recovery turns**; include parallel and nested calls; keep thinking on tool-call steps within a turn; compress long middles with `trajectory_compressor.py --target_max_tokens 16000`, but keep a slice of uncompressed 32k traces.

### 4.4 Negatives and policy trajectories (mandatory)
(a) no-tool / cannot-do / ask-clarification decisions (When2Call-style: SFT then preference optimization cut tool hallucination 19%→1.2%); (b) irrelevant-tool distractors and **function-name masking** (Hammer) so the model reads descriptions instead of memorizing names; (c) missing-parameter turns; (d) the agent proposes a draft and calls the approval tool with a structured summary (budget, audience, copy, claims, disclosures); (e) the agent **declines instructions found inside fetched web/email/CRM content** and reports them; (f) confidentiality probes (CRMArena-Pro found "near-zero inherent confidentiality awareness"); (g) refusing to fabricate reviews/testimonials (FTC rule, Oct 2024).

### 4.5 Verification and curation
Rule layer (schema-valid calls, no unknown tools/params, no failure patterns) → **state match vs blueprint** → rubric LLM judge (order-swapped, length-controlled, with an "Unknown" option) → MinHash + semantic dedup → PII scrub (Presidio + allow/deny lists; it warns it cannot guarantee full coverage, so sample by hand) → n-gram decontamination against your eval suite → **human audit of ≥600 samples at ≥95% accept** (APIGen's bar). Expect **10–20k verified company trajectories** (APIGen-MT trained τ-bench-leading models with ~5k; ToolACE 11.3k).

### 4.6 Mining your internal LLM logs
Capture full traces (system prompt, tools, every call + observation, final reply, user reactions) → scrub → dedup → label: (i) **SFT** from trajectories that pass automatic checks + rubric judge (Kimi K2 calls this "large-scale rejection sampling"); (ii) **KTO** directly on thumbs up/down — it needs only unpaired binary feedback and "matches or exceeds DPO from 1B to 30B" with up to 90% fewer desirable examples [S]; (iii) **DPO pairs** from edits (edited = chosen, original = rejected), approved-vs-rejected drafts, and judge-pass vs judge-fail trajectories for the same task. Implicit signals (retries, abandoned sessions) are weaker; use them for mining failure clusters into new blueprints, not as labels.

### 4.7 Brand voice set
~500–2,000 curated exemplars across channels (email, social, landing, ads) with constraint metadata (length, banned claims, required disclosures) plus **length-matched on-voice vs off-voice DPO pairs**, and *negative* examples (over-claiming, off-brand). ~1k examples is enough to move style on a strong instruct model (LIMA; URIAL); keep the in-context voice brief anyway. Keep facts out of this set entirely.

### 4.8 Public data to mix (license-checked)
Reuse: **Toucan-1.5M** (495 real MCP servers, 2,000+ tools; code MIT, data stated permissive — confirm), **Nemotron-Post-Training tool_calling / Nemotron-Agentic-v1 / When2Call** (CC-BY-4.0), **hermes-function-calling-v1** (Apache-2.0, convert to OpenAI messages), Glaive v2 (Apache-2.0, simple), ToolACE (Apache claimed — confirm). *Imitate only* (non-commercial): APIGen-MT-5k, Magpie data, xlam-60k (license conflict), ToolBench.

### 4.9 Mix (recommendation, not a measured optimum)
Company trajectories 30–40% · public tool data 30–40% · **general replay regenerated by the base model itself** (Magpie/SDFT-style — prompt Qwen3.6-27B to answer general/IF prompts, keep its own outputs) 20–30% · brand-voice + decision negatives as fixed sub-slices. Replay evidence: ~5% suffices for mild shift in continued pretraining (Ibrahim et al. 2024 [S]); Alopex saw MMLU/GSM8K drops after function-calling SFT and recovered with a 1:1 general mix [S]; base-model-regenerated general data beat external general data in a 2025 study [S].

---

## 5. Training recipe

### Stage 0 — Baseline in the harness (week 1)
Run stock Qwen3.6-27B (and Qwen3.8-27B) inside Hermes with your tools, Skills, SOUL.md and system prompt on the golden suite (§7). Many "it doesn't know our process" failures close with Skills and tool design alone. Fine-tune only the measured gap.

### Stage 1 — SFT (LoRA)
| Setting | Value | Why |
|---|---|---|
| Adapter | LoRA on **all linear layers incl. GDN projections**, r = 128 (sweep 64–256), α = 32 with 1/r scaling (TRL convention) or α = 2r (Axolotl/Unsloth convention) | "LoRA Without Regret": all-layer LoRA matches full FT at post-training scale; attention-only underperforms even at matched parameter count; replicated to 4 decimals on Qwen3-4B. Rank sized to data (r=256 for large SFT sets). |
| LR | 1e-4 (sweep 5e-5–2e-4), 5% warmup, cosine or constant, wd 0 | Optimal LoRA LR ≈ 10× full-FT LR and ~rank-independent. |
| Effective batch | 16–32 sequences | LoRA is "less tolerant of large batch sizes"; TRL recommends <32. |
| Epochs | 2–3; **select checkpoint by eval, not loss** | Gekhman: harm from unknown facts appears late → early-stop. |
| Sequence length | 32k (context parallel) or 16k with trajectory compression; sample packing **BFD**, never `wrapped` | Agentic traces are long; packing must respect GDN boundaries (§6). |
| Loss | Assistant tokens only (incl. tool-call XML); mask user/system/**tool responses**; mask the empty `<think>\n\n</think>` prefix if training non-thinking samples | TRL `assistant_only_loss=True` (uses vendored `qwen3_6_training.jinja`), Axolotl `roles_to_train: [assistant]`, ms-swift `--loss_scale ignore_empty_think`. |
| Thinking | On; thinking only in the current episode (or everywhere if you serve `preserve_thinking: true` and patched Hermes) | Match training to serving exactly. |
| Framework | **Axolotl** (shipped 27B QLoRA/FFT configs, GDN-aware CP, FSDP2) or **ms-swift** (official Qwen recommendation; SFT→DPO→GRPO→export in one tool); Unsloth for fast single-GPU iteration | Avoid LLaMA-Factory for this model until its `qwen3_6` template mismatches (system/tools order, empty think blocks on every turn, dropped text+tool_call turns — issue #10530) are fixed. |

### Stage 2 — Recover general ability with on-policy distillation (≈100–200 steps)
Student = Stage-1 model; **teacher = the original Qwen3.6-27B** (frozen); prompts = general chat/IF/reasoning prompts (no domain data); loss = per-token reverse KL. Thinking Machines showed IF-Eval recovering "within approximately 100 steps" after domain mid-training without losing the injected skill; Qwen3's own report found OPD reached 74.4 AIME'24 in 1,800 GPU-h vs 67.6 for RL in 17,920 GPU-h [S]. LoRA r=32–128, LR 1e-4 (Tinker cookbook default). TRL ships `GKDTrainer`/`SDFTTrainer`; Tinker and NeMo RL ship OPD. If you have a same-tokenizer stronger open teacher, OPD is also your cheapest capability transfer (~10× cheaper than RL).

### Stage 3 — Preference
Length-normalized DPO (Tülu 3 `dpo_norm`, β≈5) or standard DPO β=0.1, 1 epoch, LoRA LR ~5e-6: pairs = verified-success vs judge-failed trajectory for the same task; correct vs wrong *decision* (call / ask / refuse); on-voice vs off-voice (length-matched). **KTO** (β=0.1, LR 5e-6, HALOs `UnpairedPreferenceTrainer`) on production thumbs. Preference optimization is what moved tool-hallucination from 7% (SFT) to 1.2% in When2Call.

### Stage 4 — RL with gated rewards in the sandbox (through the harness)
- **Where to run**: verl `run_qwen3_5_27b_fsdp.sh` pattern (FSDP2 + offload, AgentLoop multi-turn tools, vLLM/SGLang rollouts, PEFT LoRA r≥32) on one 8×H100 node; or **NousResearch/Gym + NeMo RL** to run your model as the policy inside *real* Hermes Agent (`hermes_agent` harness); prime-rl + verifiers for fully async; Tinker if you want zero infra (LoRA-only).
- **Algorithm**: GRPO with DAPO components — clip-higher (ε_high 0.28), dynamic sampling (drop all-correct/all-wrong groups), token-level loss, soft overlong penalty, KL β≈0 or small; G=8 rollouts × 64 prompts/step; temperature 1.0; **per-task token budgets** (Kimi K2) to stop length explosion; monitor entropy, apply Clip-Cov/KL-Cov if it collapses. LoRA r≤32 is sufficient for RL ("~1 bit per episode").
- **Reward (decomposed, gated — Shopify's "N-stage gated rewards")**: stage 1 format/schema validity → stage 2 argument fidelity (IDs/enums must appear in prior tool results; invented values = 0) → stage 3 execution/state match vs blueprint (verifiable for analytics tasks) → stage 4 rubric judge for copy quality/voice/policy with an "Unknown" option → hard **−1 for any publish/spend/send without an approval call**. Add an *attempt* term: Shopify's model learned to explain why it couldn't help instead of trying hard tasks.
- **Anchors against forgetting**: PTX/replay auxiliary loss on curated SFT data (Kimi K2), or periodic OPD from the original checkpoint.
- **Reward-hacking hygiene**: Qwen bases improve under *random/spurious* rewards on some tasks [S] — validate your reward with ablations (shuffled rewards should not help); K2-style "hack-check" for claimed-but-fake compliance. ~4k good RL episodes already moved BFCL by +15 points over SFT in ToolRL [S].

### Stage 5 — Merge → quantize → eval served → serve
Merge LoRA (`swift export --merge_lora`, Axolotl `merge-lora`) → **FP8-dynamic** via llm-compressor (≈ lossless on every proxy found; INT4 only after your own BFCL/τ² run on the quantized artifact, calibrated on agent traces) → evaluate the *served* model through the production vLLM flags → deploy. Multi-LoRA (`--enable-lora --max-lora-rank 64`) for A/B canaries only; merged weights for production. Trainers drop the `mtp.*` tensors — re-attach the original MTP head and re-measure acceptance, or serve without speculative decoding.

```bash
# production serve (text-only agent; for screenshots/vision drop --language-model-only
# and add the multimodal flags in Part II §8)
vllm serve /models/acme-marketing-27b-fp8 --port 8000 --max-model-len 131072 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only --enable-prefix-caching --mamba-cache-mode align \
  --enable-mamba-shared-prefix-checkpoint --prefix-match-unit 64
```
```yaml
# ~/.hermes/config.yaml
model:
  default: acme-marketing-27b      # no "qwen" in the name → lean prompt variant (or pin agent.tool_use_enforcement)
  provider: custom
  base_url: http://vllm-host:8000/v1
  api_key: ""
  context_length: 131072
```

### When to escalate to full fine-tuning
"Procedural Knowledge Is Not Low-Rank" (Jul 2026 [S]): on a 14-node procedural task LoRA r=16–128 scored ≤2.54 vs 4.11 for full FT despite 95–99% conversation completion; full-FT updates had effective rank ~760–1,030 and r=128 captured only ~45% of the update norm. If your long-runbook evals plateau under LoRA while short-horizon tool evals are fine, that is the signal. Full FT of 27B needs ~437–491 GB of optimizer state → 8×H200/B200 with FSDP2 + activation checkpointing + fused CE (≈ $400–900 per 500M tokens). Keep a **SLERP/task-arithmetic merge with the base** as a forgetting fallback (avoid TIES/DARE unless validated — a Nov 2025 study found TIES can "catastrophically degrade").

---

## 6. Qwen3.6 hybrid-architecture gotchas (each one has bitten someone this year)

1. **Packing contamination on GDN layers**: with `padding_free`/packing, "every sequence after the first in a row silently starts from the state left by the end of the previous one… the loss looks normal" (TRL PR #7533). Use Axolotl (FLA **required** for packing), ms-swift ≥4.3.1, or TRL with the Oct-9-2026 collator fix that passes `cu_seq_lens`/`seq_idx`. Verify with a two-sequence test.
2. **LoRA target modules** must include `linear_attn.*` (Axolotl README lists `in_proj_qkv`, `in_proj_z`, `out_proj`; ms-swift `all-linear` covers them).
3. **Kernels**: without `flash-linear-attention` + `causal_conv1d` the model "silently falls back to slower and more memory hungry PyTorch ops".
4. **Context parallel**: Ulysses degree ≤ 4 (4 KV heads) — use Ring/USP in Axolotl Ringmaster or ms-swift for 32–64k; prime-rl supports only Ulysses for hybrids.
5. **QLoRA caveats**: Axolotl's 27B QLoRA peaks at ~47 GiB at 2k seq (hybrid layers/overheads don't shrink as expected); one report claims bnb-4bit mis-shapes DeltaNet projections [U]. Prefer bf16 LoRA on 80 GB+.
6. **Template fidelity**: tools block comes *before* your system prompt; `tool` messages merge into one `user` turn; `arguments` must be a dict (string arguments break the official template); no `developer` role.
7. **FP8 base is not for training** (Unsloth #6200). Train BF16, quantize after.
8. **Thinking regime drift**: training only non-thinking data while serving thinking mode produced degraded/empty thinking in the field (LLaMA-Factory #10644).
9. **Vision tower** consumes memory for a text-only agent — freeze in training, `--language-model-only` in serving.
10. **Prefix caching on hybrids** is block-granular — add `--enable-mamba-shared-prefix-checkpoint --prefix-match-unit 64` so requests sharing the 16k-token Hermes prefix actually hit cache.

---

## 7. Evaluation gates (every checkpoint, through the production server, inside Hermes)

| Gate | Metric | Bar |
|---|---|---|
| Golden sandbox suite (100–300 tasks across personas, seeded, `verify()` on end state) | **pass^4 and pass^8** (τ-bench definition: all k trials succeed; gpt-4o retail fell 0.604 → 0.383 from pass^1 to pass^4) | Improve on baseline; publish pass^8, not pass@1 |
| Side-effect policy | publish/spend/send without approval; PII egress | **0** |
| Decision slice (When2Call-style) | tool-hallucination rate; clarification precision/recall; irrelevant-tool rejection | hallucination ≤1–2% |
| Analytics faithfulness (AD-Bench-style) | numbers traceable to tool rows; metric definition + date range stated; no extrapolation | judge + exact-match on seeded data |
| Brand voice | style-embedding similarity to brand corpus + stylometrics (banned words, reading level) + order-swapped pairwise judge; monthly human calibration | single-metric style eval is unreliable (near-zero cross-metric correlation in PersonalBench [S]) |
| Prompt injection | attack-success rate on a suite of poisoned web/email/CRM content | track like Anthropic's browser work (23.6% → 11.2% with mitigations) |
| Public proxies | BFCL v4 subset (`bfcl generate --skip-server-setup` against your endpoint), τ²-bench retail/telecom, MCP-Bench subset | no regression vs base |
| General regression (forgetting) | IFEval, MMLU-Pro, GPQA, GSM8K/MATH | within ±1–2 pts of base |
| Human | 100–200 blind pairwise trajectories/week, ≥2 raters, report κ | wins ≥ baseline; creative quality needs humans (LLM judges disagree with 678 creatives in the Creativity Benchmark) |
| Production | shadow mode for write actions → account-level A/B; primary = task completion without human correction; guardrails = violations, destructive-action rate, escalation rate, latency | — |

Also run Hermes' own `hermes-toolperf-evals` (tool-efficiency signal) and refresh the golden suite from production failure clusters each release.

---

## 8. Infrastructure and cost (Oct 2026 prices; neo-cloud medians ≈ $3.25/H100-h, $4.50/H200-h, $6.25/B200-h; hyperscalers ≈ 2×)

**Memory facts**: bf16 weights 54.6 GB; full-FT optimizer state 437–491 GB (273 GB with 8-bit AdamW); activations ≈ 0.66 MB/token with checkpointing; KV only 65.5 KB/token (16 attention layers × 4 KV heads — ~4× less than a dense 32B) plus a fixed 78 MB GDN state per sequence. Published: Unsloth Qwen3.5-27B bf16 LoRA **56 GB** [S]; Axolotl Qwen3.5-27B QLoRA ~47 GiB, FFT ~53 GiB/GPU on FSDP2.

| Recipe | 1×H100 80 GB | 1×H200 141 GB | 8×H100 | 8×H200 / B200 |
|---|---|---|---|---|
| QLoRA | ✓ to ~8–16k seq | ✓ to ~32k | ✓ | ✓ |
| LoRA bf16 | ✓ to ~8–16k; 32k ✗ | ✓ 32k | ✓ (+CP for 32–64k) | ✓ |
| Full FT | ✗ | ✗ | ✓ ≤8–16k with offload | ✓ 32–64k with AC + CP |

**Cost arithmetic** (throughput assumptions: 1,500–2,000 tok/s/GPU full FT; ~1,200 tok/s QLoRA on 1×H100; RL generation ~1,200 output tok/s per H100 for a 32B-class [S]):
- (a) SFT on 20k × 8k-token samples, 2–3 epochs (320–480M tokens): 1×H100 QLoRA ≈ 74–111 h ≈ **$240–360**; 8×H100 bf16 LoRA ≈ 67 GPU-h ≈ **$220**; managed: Together ~$720, Fireworks ~$1,440, Tinker ~$1,970 [U].
- (b) Full FT, 500M tokens: ~105 GPU-h on 8×H100 ≈ **$340** (8×H200 for long sequences ≈ $380–450). Memory, not dollars, is the constraint.
- (c) GRPO, 512 rollouts/step × 8k tokens: ~10–20 min/step on one 8×H100 node → 500 steps ≈ **$2.2–4.3k**, 2,000 steps ≈ **$9–17k**; 32k-token trajectories ≈ 3–4×; Tinker ≈ $15k/500 steps [U].
- Serving on 1×H100: FP8 weights leave ~45 GB KV ≈ **~21 concurrent 32k sequences**; expect ~60–120 tok/s single-stream and ~1–2k tok/s aggregate (no direct benchmark found [U]); MTP speculative decoding helps the latency-bound agent loop.

**Budget tier (~$1–8k)**: 1×H100/H200 for SFT + DPO (Axolotl or Unsloth) → one 8×H100 node for 3–7 days of verl GRPO (or Tinker) → serve FP8 on 1×H100 → BFCL/τ² + golden suite via promptfoo CI.
**Premium tier (~$15–40k)**: 8×H200/B200 full FT or high-rank LoRA with Ringmaster CP for 32–64k traces → 2 nodes (train + inference) async RL with prime-rl or verl, or NousResearch/Gym + NeMo RL through real Hermes → serve on 2–4×H200 FP8 + MTP with nightly eval dashboards (W&B/Weave).

Managed landscape: **Tinker** supports Qwen3.5/3.6/3.8-27B, LoRA-only, SFT/DPO/RL/OPD loops, ~3–5× self-run cost at scale [U]; **Fireworks** fine-tunes Qwen3.6-27B (LoRA/full SFT, DPO, RFT); **OpenPipe retired** (Jul 2026); **torchtune deprecated**; **AutoAWQ deprecated** (use llm-compressor).

---

## 9. Governance (build in from day one)

- **Training data & privacy**: EDPB Opinion 28/2024 — legitimate interest is not a default basis for training on personal data; pseudonymise, scrub (Presidio + custom recognisers + human sampling), prefer retrieval under the user's own OAuth scope so customer PII never enters a training set.
- **Teacher/provider terms** for distillation (§4.3).
- **FTC**: Consumer Reviews and Testimonials Rule (effective Oct 21 2024) bans fake/AI-generated reviews and testimonials with civil penalties; Endorsement Guides (2023) require material-connection disclosure — encode as refusal/disclosure behaviour *and* a rule check in the reward. Specific 2026 penalty figures conflict across sources [U].
- **EU AI Act Art. 50** transparency obligations apply from Aug 2 2026 (deepfake-style synthetic media must be disclosed; machine-readable marking deferred to Dec 2026/Feb 2027 per law-firm summaries [S]) — put the labelling checklist in a Skill.
- **Prompt injection** ("lethal trifecta": private data + untrusted content + external communication): treat all fetched web/email/CRM text as data; no consequential action after ingesting untrusted input without approval; tool allow-lists; OWASP LLM01/LLM06.
- **Human-in-the-loop**: drafts by default; approval tool with structured summary; spend caps and allow-lists enforced in pre-tool hooks, never by the model; full audit log; shadow mode before enabling writes.

---

## 10. Phased timeline

| Weeks | Deliverable |
|---|---|
| 0–2 | MCP servers + gated write tools; seeded sandbox with reset; compatibility checklist (§2.3) passed; Skills v1 (runbooks, conventions, compliance checklist); **Stage-0 baseline** on Qwen3.6-27B and Qwen3.8-27B |
| 2–6 | Blueprints + rubrics (3–5k); teacher trajectories via Hermes `batch_runner`; negatives; log mining; brand-voice set; verification + 600-sample human audit; golden eval suite (100–300 tasks) |
| 6–8 | Stage 1 SFT + Stage 2 OPD recovery; eval gates; iterate on data from failure clusters |
| 8–10 | Stage 3 DPO/KTO; merge → FP8 → served eval; shadow mode in Hermes with read-only tools |
| 10–14 | Stage 4 GRPO in sandbox (500–1,000 steps); re-gate; enable gated writes behind approvals; A/B |
| ongoing | Flywheel: production logs → scrub → judge → SFT/KTO/DPO → re-gate; Skills evolution via GEPA; golden suite refreshed from production failures |

---

## 11. What I need from you to turn this into a concrete build

1. Your actual tool list (which marketing systems, which already have MCP servers, which write actions you want the agent to take vs draft).
2. Volume and format of existing internal LLM logs (and whether thumbs/edits are captured).
3. GPU access and budget tier; whether managed (Tinker/Fireworks) is acceptable for weights.
4. A brand-voice corpus and an approved-claims library (or who owns them).
5. Decision on thinking mode in production and acceptable latency per turn.
6. Legal position on teacher-model terms and on training with customer data.

---

# Part II — Holo (H Company): how they trained Qwen for computer use, what it costs, and what transfers

**Research date 2026-10-13.** Three additional threads: Holo's own releases (model cards/blog via search snippets — hcompany.ai and huggingface.co are blocked from this sandbox; H's GitHub repos were cloned and read directly), the comparable open Qwen computer-use recipes that *do* publish hyperparameters and compute, and the Hermes Agent vision/browser code path. Same [S]/[U] flags.

## II.1 Lineage — every Holo release

| Release | Date | Base (per size) | Open sizes / license | Training disclosed | Data disclosed | Compute disclosed | Headline numbers (vendor) |
|---|---|---|---|---|---|---|---|
| **Holo1** | Jun 2025 (arXiv 2506.02865) | Qwen2.5-VL-3B / 7B-Instruct | 3B Qwen Research licence; **7B Apache-2.0** | **SFT only** ("Filtered Behavioral Cloning" = train only on validator-verified successful traces); one model serves policy + localizer + validator | **31.46B tokens**: GUI grounding 50.8% (WebCrawl 12.19B from 4M pages / 89M clicks; open-source 3.42B; WebSynthetic 0.37B for calendars/tables), visual understanding 32.3%, behaviour 16.9% (policy 15.5% = WebVoyager 643 tasks + WebVoyagerExtended 15k tasks / 330 sites; validator 1.45% ≈ >1M pairs) | none | WebVoyager 92.2% @ $0.13/task (7B), 89.7% @ $0.11 (3B) vs GPT-4.1 92.0% @ $0.54. Ablation: traces-only model 87.7% |
| **Holo1.5** | Sep 2025 | Qwen2.5-VL-3B / 7B / 72B | 7B Apache-2.0; 72B CC-BY-NC | "multi-stage pipeline"; secondary: SFT → GRPO-style RL [S] | "open-source + large-scale synthetic + proprietary"; no sizes | none | Localization avg 72.8 / 77.3 / 80.5 vs base 57.0 / 60.7 / 70.9; ScreenSpot-Pro 51.4 / 57.9 (3B/7B) vs 29.3 / 29.0 |
| **Holo2** | Nov 2025 | Qwen3-VL-**Thinking** 4B / 8B / 30B-A3B | 4B, 8B Apache-2.0; 30B-A3B licence conflicting [U] | "large-scale SFT, followed by **online RL (GRPO)**… tailored for localization"; thinking **off** for localization, on for navigation; coords **normalized 0–1000** | "open-source + synthetic + human-annotated"; no sizes | none | 30B-A3B: ScreenSpot-Pro 66.1, OSWorld-G 76.1; 8B nav: WebVoyager 80.2 / WebArena 42.2 / OSWorld 39.9 / AndroidWorld 60.4 |
| Holo2-235B-A22B preview | Feb 2026 | Qwen3-VL-235B-A22B-Thinking | research-only | adds "agentic localization" (multi-step coordinate refinement) | — | none | ScreenSpot-Pro 70.6 → 78.5 within 3 refinement steps |
| **Holo3** | Mar 31 2026 | 35B-A3B: Qwen3.5-35B-A3B; 122B-A10B base unconfirmed [U] | 35B Apache-2.0; 122B API-only | "agentic learning flywheel": synthetic navigation data, programmatic out-of-domain augmentation, "every sample is filtered and run through a reinforcement-learning pipeline"; "Synthetic Environment Factory" | no sizes | none | OSWorld-Verified 78.85 (122B) / 77.8 (35B) |
| **Holo3.1** | Jun 1 2026 | Qwen3.5 0.8B / 4B / 9B / 35B-A3B | all Apache-2.0 | adds **native function-calling** ("near-parity" with strict JSON); FP8 / NVFP4 quants ≈ −2 pts vs BF16 | none | none | AndroidWorld 79.3 (35B), 71–72 (4B/9B); OSWorld 74.2 (35B) [S] |
| **Holo4** | **Sep 28 2026** | **Holo4-27B = Qwen3.8-27B dense**; Holo4-35B-A3B = **Qwen3.6-35B-A3B**; Holotron4-30B-A3B = Nemotron 3 Nano Omni | 35B-A3B **Apache-2.0**; 27B reported **CC-BY-NC-4.0** [U] | **SFT 127B tokens → asynchronous online RL with prime-rl training two LoRA experts** (desktop+web; terminal+MCP+API) **merged at equal weight, no further training** | SFT ≈ ¾ successful agentic trajectories: desktop 45% / web 14% / MCP+API 12% / mobile 3%; rest multimodal reasoning, GUI grounding, text tool-use & coding. "Agentic Task Factory": **~10,000 verifiable tasks** (≈4k web apps, 3k MCP servers, 3k desktop/OS); "a task survives only if its verifier rejects near misses" | none (prime-rl named) | **OSWorld 85.2% @ $0.08/task (27B) vs base Qwen3.8-27B 84.3% @ $0.22**; 35B 80.8; OSWorld 2.0 61.7 / 30.9; AndroidWorld 85.1; AutomationBench held-out 49.3 / 31.7 (480 of 600 public tasks were in training) |

**What H has never disclosed, in any release:** learning rate, epochs, batch size, schedule, training image resolution, full-FT vs LoRA for SFT (only Holo4's RL uses LoRA), GPU type / count / hours, dollar cost, wall-clock, RL reward formulas, environment counts, dataset sizes after Holo1. **No training code exists publicly** — only inference/deploy notebooks (`hai-cookbook`), the archived Surfer-H CLI, and the **WebClick** eval set (1,639 web screenshots). H says it uses TRL + vLLM (interview) and prime-rl (Holo4). So "replicating Holo" means reconstructing the recipe from their disclosed token counts plus the open recipes in §II.4.

## II.2 The Holo recipe, read across releases

1. **Filtered behavioural cloning** (Holo1 → Holo4): run an agent, keep only validator-verified successes, SFT on them. This is the same thing as the rejection-sampling SFT in Part I §4.5 — Holo is independent confirmation that it works for GUI agents at 3B–27B.
2. **Grounding-heavy mixture** (Holo1: ~51% grounding, ~32% screen understanding, ~17% behaviour) → by Holo4 the mixture inverted to ~75% trajectories once grounding was mature in the base. For Qwen3.6-27B (already AndroidWorld 70.3 / OSWorld-Verified ~63.9 [S]) you are in the Holo4 regime: trajectories matter, grounding is a small slice.
3. **In-domain traces are worth real points**: Holo1 attributes ~4.5 pts of WebVoyager to in-domain WebVoyager traces [S]. Generate traces on *your* marketing tools.
4. **Targeted synthetic data for failure modes** (WebSynthetic: complex calendars, relational tables) → date pickers, campaign tables, budget forms, audience builders.
5. **Verifiable task factory + online RL** (Holo3/4): environments and tasks built from docs, screenshots and real software; verifiers that reject near misses; GRPO (Holo2) → async RL (Holo4). RL bought *efficiency* more than accuracy on the 27B (+0.9 pt, −64% cost/task).
6. **LoRA experts per lane, merged** (Holo4): one LoRA for GUI (desktop+web), one for tools (terminal+MCP+API), equal-weight merge, no further training. Cheapest proven RL path on a dense 27B of this exact architecture.
7. **Policy / localizer / validator split with validator-gated retries** (Surfer-H/Surfer 2): localizer gets the policy's *textual* description of the target and returns coordinates; validator judges task completion from the last 3 screenshots; retries up to 10. A 3B policy went 68.2% (1 attempt) → 75.2% (2) → 82.1% (5) → ~92% (10) on WebVoyager; swapping the GPT-4o validator for the small model cost 12–16 pts.
8. **Harness details** (read from `surfer-h-cli` source): screenshot-only observation, last **3** screenshots in context, explicit `notes` memory field, `thought` before action, forced answer at step (30) / time (600 s) limits, 2 s settle after each action, guardrails ("never log in", accept cookies, ≤3 scrolls), strict JSON actions via `response_format`, `smart_resize` client-side.
9. **Format**: Holo3.1 shows native function-calling reaches near-parity with strict JSON → train on Qwen's native tool-call format (`qwen3_coder` parser), as Part I already does. Thinking on for navigation, **off** for localization.
10. **Quantization**: FP8 / NVFP4 ≈ −2 pts (Holo3.1) → FP8 production is fine, matching Part I.

## II.3 Filling the gaps Holo leaves: what the open Qwen computer-use recipes publish

| Recipe (base) | Stage & hyperparameters actually published | Result |
|---|---|---|
| Aguvis (Qwen2-VL-7B) | 2-stage SFT: grounding mix → trajectories; **lr 1e-5, 1 epoch, cosine, warmup 0.03, vision tower frozen**, ZeRO-3, max len 8192, 8 GPUs/node | — |
| ScaleCUA (Qwen2.5-VL 3B/7B/32B) | SFT lr 1e-5, 3B 1 epoch / 7B-32B 2 epochs, **max_pixels 2.1M**, max len 30–40k, ViT frozen, MLP+LLM tuned | OSWorld-G 60.6 (32B) |
| GUI-Actor (Qwen2.5-VL 3B/7B) | warm-up lr 1e-4 (pointer head only) → full lr 5e-6, 1 epoch each, max_pixels 5.7M | ScreenSpot-Pro 44.6→47.7 (7B w/ verifier) |
| **GUI-R1** (Qwen2.5-VL 3B/7B) | **GRPO on 3K samples**: reward = 0.8·accuracy + 0.2·format; click correct if ≤140 px from GT point or inside GT box; text F1 ≥ 0.5; n=5 rollouts, **lr 1e-6**, KL 1e-2, max_pixels 1.26M, 8 GPUs | large grounding gains from tiny data |
| UI-R1 / UI-R1-E (Qwen2.5-VL-3B) | GRPO on **136 samples**, 8 generations; E: 4 epochs difficulty-adaptive length reward + 4 epochs no-think | ScreenSpot-Pro 17.8 → 29.8 → 33.5 |
| **GTA1** (Qwen2.5-VL 7B–72B) | grounding GRPO, **no thinking**, 2 epochs, 8 gens, max completion 128, β=0, ViT frozen; **≈800 H100-hours for the family** [S] | 7B SS-Pro 55.5; 32B 63.6 / OSWorld-G 65.2 |
| ZeroGUI (UI-TARS-7B-DPO) | online RL: VLM-generated tasks + binary VLM reward (Qwen2.5-VL-32B judge), 64 samples/prompt, lr 2e-6, 15-step episodes | +14% rel. OSWorld |
| ARPO (UI-TARS-1.5-7B) | GRPO + experience replay on 128 OSWorld tasks; **8 nodes × 8 GPUs, 128 Docker envs** | OSWorld 23.5 → 26.0 (GRPO) → 29.9 |
| OpenCUA (Qwen2.5-VL 7B/32B/72B, MIT incl. data) | SFT on AgentNet 22.6K human tasks (~18.6 steps avg) + reflective CoT; 3-screenshot history | OSWorld-Verified 7B 27.9 / 32B 34.8 / 72B 45.0 (50 steps) |
| GUI-Owl-1.5 (Qwen3-VL + env RL) | undisclosed hparams; cleanest size curve | **OSWorld-Verified 2B 43.5 → 4B 48.2 → 8B 52.3 → 32B 56.5** |
| Qwen3.6-27B (your base, stock) | — | **OSWorld-Verified ~63.9 [S], AndroidWorld 70.3** — already above every 2025 open 32B fine-tune |

Convergent recipe: full FT with the vision tower frozen (no published recipe used LoRA for SFT), grounding SFT at lr 1e-5 for 1 epoch, trajectory SFT 1–2 epochs with reflective CoT and a 3-screenshot window, then GRPO — rule-reward grounding GRPO on thousands of samples (hours on 8 GPUs) and/or online environment RL (64 GPUs × days + 100+ env slots). Online RL is what separates 2025 SFT-only 7B models (~27% OSWorld-Verified) from 2026 7–8B models (52–53%). Only two hard compute disclosures exist anywhere: GTA1 ≈ 800 H100-h; GUI-C² 4×H20 for 17 h (3B) / 26 h (7B).

## II.4 How Hermes Agent actually uses vision (this decides whether pixel grounding is worth training)

Read from `hermes-agent` source (`tools/browser_tool_vision.py`, `tools/vision_tools.py`, `agent/image_routing.py`, `tools/computer_use/schema.py`, docs):

- **The browser toolset is accessibility-tree based.** Pages are text snapshots with ref IDs (`@e1`); `browser_click` clicks a ref. Docs: "relies on accessibility tree, not pixel coordinates." **No `click(x,y)` exists in the browser toolset** → a Holo-style pixel localizer is *unused* by Hermes' browser tools.
- **Screenshots do reach the model** via `browser_vision` / `vision_analyze` — attached natively as `image_url` parts (JPEG ≤256 KB, ≤1568 px wide, ~1,372 Qwen3.6 tokens, re-sent every turn until eviction) **only if Hermes believes the model supports vision.** Routing never auto-probes vLLM endpoints; unknown ⇒ lossy text description. **Set `model.supports_vision: true` (or `agent.image_input_mode: native`)** for a self-hosted Qwen3.6, and verify your vLLM build accepts `image_url` parts on `tool`-role messages.
- **Pixel coordinates are consumed only by** the separate `computer_use` toolset (drives the host desktop; `coordinate: [x, y]` or set-of-marks index; ~1,500 tokens/screenshot), a custom tool you write (localizer → `browser_cdp` mouse event), or Browser-Use `browser_exec` scripts.
- **Hermes is an officially supported host for Holo** (`holo install hermes`, `agents-api-demos/hermes/`): H's Agents API runtime (closed binary, open SDK) runs Holo as a sub-agent/MCP server inside Hermes today.

## II.5 Cost model — Holo-scale replication vs. the targeted marketing version

Assumptions: FLOPs ≈ 6·N·tokens full FT (LoRA ≈ 4·N), +~8% for the ViT, H100 bf16 989 TFLOPS at **30% MFU** for image-heavy packed data (range 20–40%), $3.25/H100-h, $4.50/H200-h. Qwen3.5/3.6 vision = **32 px per token**: 1366×768 → 1,032 tokens; 1568×882 (Hermes cap) → 1,372; 1920×1080 raw → 2,040 (capped at `max_pixels=1280·32·32` → 1,222); 2560×1440 → 3,600. A 10-step trajectory packed as one multi-turn sample with Hermes-cap screenshots ≈ **17.7k tokens** (raw 1080p ≈ 24.4k); a 3-screenshot sliding window ≈ 8.1k; unrolling each step as its own sample ≈ 102k — never do that.

**(a) What Holo itself cost in training compute (reconstructed from their token counts; [U-arith] — H discloses nothing):**

| Run | Tokens | H100-hours (20–40% MFU) | Compute $ | Notes |
|---|---|---|---|---|
| Holo1-3B SFT | 31.46B | ~410–820 | ~$1.3–2.7k | tiny; 44 authors, so the cost was people + data |
| Holo1-7B SFT | 31.46B | ~1,000–2,000 | ~$3.3–6.5k | WebVoyager 92.2% from this |
| Holo4-27B SFT | **127B** | **~14,600–29,000** | **~$47–95k** | ≈13 days on 64×H100 at 30% MFU |
| Holo4-35B-A3B SFT (3B active) | 127B | ~3–5k (MoE efficiency) | ~$10–16k | |
| Holo4-style async RL (undisclosed) | ~1,000 steps × 512 rollouts × ~20k tok | ~6–8k GPU-h + sandbox fleet (200 VMs × ~400 h × $0.17–0.46) | **~$35–60k** | |
| **Data generation** for ~95B tokens of *successful* trajectories at ~30% success | ~300B generated tokens on self-hosted 27–35B models | — | **~$100–300k** + environments + human annotation | the dominant cost; plus 10,000 verifiable tasks |

Replicating Holo4 end-to-end is therefore a **$0.3–0.6M compute-equivalent, multi-engineer effort** — and on a dense 27B it bought +0.9 OSWorld points. Not the right target for a marketing agent.

**(b) The targeted version for your agent (from the infra thread's arithmetic):**

| Job | 27B (LoRA, ViT frozen) | 4B dedicated localizer (full FT) | 8B localizer (full FT) |
|---|---|---|---|
| Grounding SFT, 100k screenshot→click samples × 2 epochs (340M tok) | ~35 h → **$114** ($85–171) | 8.6 h → **$28** | 16 h → **$53** |
| Grounding SFT, 500k × 2 epochs (1.7B tok) | 175 h → **$569** | 43 h → **$140** | 81 h → **$264** |
| Trajectory SFT, 10k trajectories × 10 steps × 3 epochs, Hermes-cap screenshots (531M tok) | 55 h → **$178** (raw 1080p: $245) | 13 h → **$44** | 27 h → **$84** |
| GRPO grounding, 500 steps (64 prompts × 8 rollouts) | 8×H100 w/ offload ~$500–650; **8×H200 ~$800–900** (colocated vLLM copy + trainer don't fit 80 GB) | 4×H100 ~28 h → **~$120** | **~$240** |
| OSWorld-Verified **pass^3** sweep (1,107 runs) | model ~$21–60 + env ~**$160** (780 t3.xlarge-h + 4 hosts), ~5 h wall-clock at 160 envs | same env cost; model ≈ $5–10 | |
| WebVoyager pass^3 (1,929 runs) | model ~$19 + browsers $6 (Browser Use) – $39 (Browserbase) | | |
| Memory | 54.6 GB weights + ~16 GB activations for a 17.7k-token trajectory + LoRA/Adam ≈ **73 GB → marginal on H100-80, comfortable on H200** | Adam states 64 GB → 2×H100 or 1×H200; EasyR1: 7B GRPO full bf16 on 4×40 GB, LoRA on 2×32 GB | 128 GB states → 2–4×H100 |
| **Total one-off** | **≈ $1.1–2.5k compute** + sandbox fees + 2–4 engineer-weeks | **≈ $0.3–0.6k** | ≈ $0.6–1k |

**What you can train for a given budget (the "how much can we train" ladder):**
- **$0** — Option C below (a11y-tree browser tools + native screenshots for understanding) plus an off-the-shelf localizer if ever needed: Holo2-4B/8B (Apache-2.0), Holo4-35B-A3B (Apache-2.0), Holo3.1-4B/9B (Apache-2.0). Avoid Holo4-27B (CC-BY-NC reported [U]) and Holo1.5-72B (CC-BY-NC) for commercial use.
- **~$500** — fine-tune a Qwen3.5-4B localizer on screenshots of *your* tools (grounding SFT + 500-step GRPO + eval).
- **~$2–3k** — Option A on the 27B: grounding + trajectory SFT (LoRA) + 500-step GRPO + two OSWorld sweeps.
- **~$10–20k** — add online environment RL on your marketing sandboxes (ARPO/ZeroGUI scale: 64 GPUs × days + 100+ env slots) as a GUI-lane LoRA expert, Holo4-style.
- **$50–100k+** — Holo4-scale SFT (127B tokens). Not advisable: your base is already at ~63.9 OSWorld-Verified and H's own RL on this architecture gained <1 point.

## II.6 Decision: how to add computer-use to the marketing agent

| Option | What it is | Training cost | Fit with Hermes | Risks |
|---|---|---|---|---|
| **(C) a11y-tree tools + native screenshots — ship now** | Hermes default `browser_*` (ref clicks) + `browser_vision`/`vision_analyze` feeding the 27B real screenshots for *understanding* | **$0** (config + serving flags) | native, zero harness changes | canvas/WebGL editors, image-only ad builders, CAPTCHA, bad a11y trees |
| **(B) small dedicated localizer — add when a tool fails on refs** | 27B policy says `click("blue Publish button")` → Holo-class 4B/8B localizer returns (x, y) → `browser_cdp` mouse event or `computer_use coordinate` | $0 off-the-shelf (Holo2/3.1/Holo4-35B Apache) or **~$0.3–0.6k** to fine-tune on your screenshots | needs a ~few-hundred-line custom Hermes tool; `computer_use` already takes `coordinate` but drives the host desktop | +0.3 s/click; long-tail SaaS UI quality |
| **(A) train grounding + trajectories into the 27B — defer** | Holo/UI-TARS-style SFT + GRPO click reward on the same 27B | **~$1.1–2.5k** compute + env fees + 2–4 engineer-weeks | pixel skill **unused by the browser toolset**; only `computer_use`/custom tools consume it | multi-task mixing vs marketing SFT; GRPO on a 27B VLM needs 8×H200 or a 16×H100 split; marginal gain over an already-GUI-trained base is uncertain |

**Recommended sequence:** ship **C** now; include a few hundred–few thousand Hermes-format browser trajectories (ref clicks + `browser_vision` use) on your marketing tools in the Stage-1 SFT mix (≈ $20–50 of compute); run against real tools; add **B** only for tools that fail on a11y refs; keep **A** as a later experiment, and if you run it, follow Holo4's pattern — two LoRA experts (GUI lane, tool/MCP lane) trained with async RL and merged at equal weight.

## II.7 Holo methods folded into Part I's stages

- **Stage 1 SFT:** add the browser lane (Hermes-format trajectories); keep grounding a small slice unless Option A; add WebSynthetic-style targeted data for date pickers / campaign tables / budget forms; keep the 3-screenshot window and `notes` field in trajectory format.
- **Stage 4 RL:** Holo4's **per-lane LoRA experts + equal-weight merge**; verifiers that **reject near misses**; an **efficiency term** in the reward (Holo4 rewarded whichever lane solved the task most efficiently; cost/task fell 0.22 → 0.08).
- **Harness:** **validator-gated retries with a separate validator** (Hermes sub-agent via `delegate_task`, or a stronger model) — the single biggest reliability lever in Surfer-H (+10–24 pts over attempts); forced answer at step/time limits; "never log in / never publish without approval" guardrails in the system prompt *and* the reward.
- **Serving (Option C):** drop `--language-model-only`; add `--limit-mm-per-prompt '{"image": 6}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --mm-processor-kwargs '{"longest_edge": 1310720, "shortest_edge": 262144}'` (= 1280·32·32 / 256·32·32 pixels); Hermes `model.supports_vision: true`. Prefix caching on the hybrid model is block-granular (528-token blocks) and a new screenshot invalidates everything after it — expect cache hits only on the system prompt + earlier history.

## II.8 Holo-specific caveats
- All Holo numbers are vendor-reported from H's own harness (2–4 runs); AutomationBench had 480/600 public tasks in training; a third-party review flagged an average-column mismatch in the Holo1 paper [U].
- Holo4-27B licence (CC-BY-NC-4.0) and Holo2-30B-A3B licence are from secondary sources / conflicting metadata — verify on the model cards before any commercial use.
- Every compute and cost figure in §II.5 is reconstructed arithmetic; H has published none.
- hcompany.ai, huggingface.co, arxiv.org were unreachable from this sandbox; Holo facts are from verbatim mirrors, H's GitHub repos, and search snippets.

---

# Part III — Execution plan: Qwen3.8-27B as the brain of your custom Hermes-based marketing-operations system

**Research date 2026-10-13.** Base model changed to **Qwen3.8-27B**. Three further threads: Qwen3.8-27B fine-tuning specifics (read almost entirely from primary sources — the HF card via a verbatim GitHub mirror, TRL's byte-exact vendored chat template, framework source code and configs), the complete taxonomy of dataset-creation methods for agentic systems, and how to capture training data from / run RL through a *custom* multi-agent harness. Parts I and II remain valid foundations (identical architecture) except where this Part says otherwise. Same [S]/[U] flags.

## III.0 The mental model (read this first)

1. **A fine-tuning dataset for an agentic system is a pile of recorded episodes.** Each episode is one conversation between *your system* and *the model*: everything your wrapper sent (system-prompt tiers, tool schemas, workflow text, memory, earlier turns, tool results) and everything the model replied (reasoning, text, tool calls), turn by turn, until the task ended. Training shows the model the exact prompt side and penalizes it only where *its own turns* differ from the recorded good turns. Nothing else is learned from.
2. **So the dataset must be in your system's exact format**, rendered byte-for-byte the way vLLM renders Qwen3.8's chat template at serving time — same system-block order, same effort sentence, same tool XML, same `<think>` framing. Train/serve format drift is the #1 silent failure in agent fine-tuning (verl documents single-turn PPO failing to converge from exactly this).
3. **Good episodes come from only three places:** (a) real usage of your system (production logs), (b) a strong teacher model driving *your* system against sandboxed tools (synthetic but *executed* and *verified*), (c) humans demonstrating (gold). Everything else in the method menu (§III.4) is a way to generate more *tasks* for (b), to *derive* episodes from documents/workflows, or to *filter and label* what you have.
4. **"Verified" is the whole game.** Keep an episode only if a program checked the sandbox end state (the campaign exists with the right budget; the email was *drafted*, not sent; the numbers match the rows), a judge checked the rubric, and no policy was violated (no unapproved irreversible action). Unverified synthetic episodes teach confident mistakes.
5. **The model learns habits; the context carries facts** (Part I §3). Weights learn: how to plan, which tool with which arguments, when to ask, when to stop, your voice, how to behave as orchestrator vs specialist. Skills/retrieval/system prompt carry company facts, SOPs, brand book, tool schemas.
6. **Seven steps, in order:** Capture → Environment → Generate → Verify → Represent → Train → Evaluate. Each section below gives the evidence and the concrete artefact to build. Glossary in §III.12.

## III.1 Qwen3.8-27B — what you must know before touching data

### Facts
- `Qwen/Qwen3.8-27B` (BF16, 55.6 GB, 27.78B params) and `Qwen/Qwen3.8-27B-FP8` (30.9 GB); Apache-2.0; released 2026-08-14; **no `-Base`**. Architecture identical to Qwen3.6-27B — NVIDIA and Tinker both state "only the chat template changed" — so every VRAM/cost number in Parts I–II applies unchanged.
- Card benchmarks, Qwen3.8-27B vs Qwen3.6-27B: Terminal-Bench 2.1 **73.0** vs 63.4 · SWE-bench Pro **61.7** vs 53.5 · CoWorkBench **70.7** vs 61.0 · JobBench 33.4 vs 21.8 · IFBench **79.5** vs 69.1 · GPQA 89.2 vs 87.8 · OSWorld-Verified **84.3** vs 63.9 · WebArena-Verified **64.8** vs 48.8 · AndroidWorld 81.9 vs 70.3 · ClawEval-MM Pass@3 57.4 vs 42.6. Not on the card: BFCL, τ², SkillsBench, SWE-bench Verified (early viral "80+" tables were fakes [S]).
- Card highlights: "stronger autonomous planning and better handling of environment feedback", "broader support for popular harnesses", flexible thinking control. No post-training/RL details are published.

### Template rules that decide how every sample is rendered (byte-exact, from TRL's vendored copy of the shipped template)
1. **`reasoning_effort` ∈ {`xhigh` (default), `medium`, `low`} — nothing else.** `high`, `none`, `max`, `minimal` make the template raise → vLLM HTTP 500 (QwenLM#217; Claude Code's default `high` hits this). `medium` injects **no** text; `xhigh` injects *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."*; `low` injects *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."*
2. **System block order = effort sentence → `# Tools` block (each tool dumped *with* its OpenAI `{"type":"function","function":…}` wrapper) → your system-prompt text.** An empty system message renders nothing; no system message + xhigh/low → a synthetic system turn holding only the sentence.
3. **Reasoning is read only from `message.reasoning_content`.** Inline `<think>…</think>` inside `content` is no longer parsed (Qwen3.6 parsed it) — it would be trained as visible text after an empty think block. Store reasoning as a separate field.
4. **`preserve_thinking` defaults to true**: every historical assistant turn renders `<think>\n{reasoning}\n</think>\n\n`; turns without reasoning render `<think>\n\n</think>\n\n`. The card: "especially beneficial for agent scenarios… also improves KV cache utilization".
5. **Tool calls** render as `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`; `arguments` must be a **dict** (JSON-string arguments are not parsed; `''` tolerated); non-strings become JSON literals. Text before a call is allowed; multiple calls newline-joined; no tool_call_id anywhere.
6. **Tool results**: consecutive `tool` messages merge into **one `user` turn** of stacked `<tool_response>` blocks in message order — restore call order before rendering (results of parallel calls completing out of order would attach to the wrong call).
7. **Roles**: `system` must be first; `developer` or anything else raises. Fold developer text into `system`.
8. **Generation prompt** always opens inside `<think>\n` (or `<think>\n\n</think>\n\n` with `enable_thinking:false`) → the server needs `--reasoning-parser qwen3` or reasoning lands in `content`. `/no_think` does not exist.
9. **Not token-prefix-stable across turns**: a sampled non-reasoning turn continues `<think>\n` + `\n`, while history renders `<think>\n\n</think>\n\n` whose `\n\n` is one token — multi-turn RL must bridge tokens (slime `TrajectoryManager`, prime-rl `bridge_to_next_turn`), never re-render.

### Known production bugs to design around
- **~19% empty answers at `xhigh`** on vLLM/llama.cpp (EOS sampled inside `<think>`; 0% at `medium`/`low`; QwenLM#216, closed "not planned"). Mitigations: `frequency_penalty 0.3`, `repetition_penalty 1.05–1.10`, a thinking budget, or lower effort.
- **Phantom tool calls** when the model quotes tool markup inside reasoning (vLLM #58147, open; ~1–2 per 1,600 requests, more when quoting). Mitigation in your wrapper: validate tool names against the offered set and de-duplicate identical calls.
- MTP speculative decoding (92.2% acceptance BF16, `num_speculative_tokens: 3`) needs the Gated-DeltaNet spec fix (vllm#51812/#51674) that no released tag carried as of 2026-10-08.

### Decisions that follow
- **Train and serve at `reasoning_effort: medium`.** Avoids the xhigh empty-answer bug; it is what the public slime Qwen3.8-27B agent-RL recipe and H Company's Holo4 RL used; the card warns lower effort "does not always reduce overall task completion time" in agentic tasks, so measure `medium` vs `xhigh` in Stage 0 but default to `medium`. Render a minority of samples at `xhigh`/`low` with matching trace lengths if you want effort-conditioned behaviour.
- **`preserve_thinking: true`** in training renders and serving — which means your wrapper must pass `reasoning_content` back in assistant history (stock Hermes strips it on custom routes; §III.2).
- **Keep the native XML tool grammar + `qwen3_coder` parser.** Every framework, quant and fine-tune of this model assumes it; Hermes never sees the grammar.
- **Serving baseline:** `vllm serve <model> --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"reasoning_effort":"medium"}' --max-model-len 131072 --enable-prefix-caching --mamba-cache-mode align --enable-mamba-shared-prefix-checkpoint --prefix-match-unit 64` (vLLM ≥0.17, verified builds ~0.27; TP must divide 24; BF16 ≈52 GiB, FP8 ≈29 GiB on one GPU). Sampling: thinking T=1.0 / top_p 0.95 / top_k 20 (+ `frequency_penalty 0.3` if empty answers appear). `reasoning_effort` is accepted as a top-level API field and forwarded to the template; `none` turns thinking off.

### Framework support for Qwen3.8-27B (verified in source unless flagged)

| Framework | Status for Qwen3.8-27B | Use it for |
|---|---|---|
| **ms-swift** (Qwen's recommendation) | `qwen3_8` template; per-row `{"chat_template_kwargs": {"reasoning_effort": "medium"}}`; `--add_non_thinking_prefix true --loss_scale ignore_empty_think`; `--target_modules all-linear` covers GDN; packing needs ≥4.3.1; GKD = on-policy distillation; GRPO with vLLM colocate; Megatron for full FT/CP; env: transformers ≥5.9, FLA ≥0.4.2 (Py 3.12), flash-attn 2.8.3 | **Primary SFT / DPO / OPD stack** |
| **Unsloth** | 27B notebooks (Conversational + GRPO); 4-bit QLoRA fits 2×16 GB at 1k seq (23.7 GiB peak); r=8–16, lr 2e-4 (2e-5 long runs), bs1×GA4 (248k-vocab logits dominate); `finetune_attention_modules=True` reaches `linear_attn` (496 modules); **never fp16** (GDN NaN gradients); use a build ≥ Sep 2026 (`in_proj_z` quant-state fix, NaN fix PR #12856); merged export ≈52 GB | Fast single-GPU iteration |
| **NeMo AutoModel / Megatron-Bridge / NeMo RL** | `qwen3_8_27b.yaml` full SFT (FSDP2, 8×H100, 100-step curves published) + `qwen3_8_27b_lora.yaml` (r=8); Megatron-Bridge import/export verified; NeMo RL `grpo-qwen3.8-27b-2n8g-megatron-tp4pp2cp2.yaml` | **Full fine-tuning** (workflow compilation), multi-node RL |
| **Axolotl** | `examples/qwen3.8/27b-qlora.yaml` (`chat_template: tokenizer_default`, `chat_template_kwargs.reasoning_effort`; uncomment `linear_attn.*` targets; CCE) | Alternative SFT |
| **TRL** | `qwen3_8.jinja` + `qwen3_8_training.jinja` (auto with `assistant_only_loss=True`; think block inside the loss span); response parsing via `qwen3_5_schema`; **no GDN padding-free fix found** — don't pack in TRL unless you verify boundaries | Library use |
| **LLaMA-Factory** | `qwen3_8` class fixes system/tools order and merges parallel results; run-global `--reasoning_effort`; empty think block in loss when thinking on; text + tool_call in one turn **unverified** | Acceptable, less control |
| **Tinker** | Qwen3.8-27B registered with `qwen3_8_{xhigh,medium,low}_reasoning` / `disable_thinking` renderers; LoRA-only; LoRA LR = 10× full-FT LR; multi-turn tool-use SFT/OPD/RL recipes | Zero-infra LoRA RL/OPD |
| **slime** | `examples/coding_agent_rl`: **Qwen3.8-27B GRPO through Claude Code / Codex as the harness**, SGLang rollouts (`qwen3_coder`/`qwen3` parsers), `reasoning_effort=medium`, 2×8 GPUs training (TP2 PP8) + 2 sandbox nodes, lr 1e-6, token-in/token-out, compaction at 70% → new segment; no LoRA | **Full-FT RL through a harness** |
| **verl** | `run_qwen3_8_27b_megatron.sh` (GRPO, lr 1e-6, KL 0.01); AgentLoop for custom loops | Megatron RL baseline |
| **prime-rl** | `[renderer] name = "qwen3.8"`, per-row `reasoning_effort` column; trainer lists Qwen3.5 (same arch) — confirm load; what H Company used for Holo4's LoRA experts [S] | Async LoRA RL |
| **llm-compressor** | `qwen3_8_gptq_awq_example.py` W4A16 (ignore `visual.*`, `lm_head`, `embed_tokens`, `linear_attn.in_proj_a/b`); official FP8; NVFP4 (NVIDIA), INT4 (RedHat) | Deployment quantization |

Existing fine-tunes of this exact model: **Holo4-27B** (SFT 127B tokens + two RL LoRA experts via prime-rl merged equal-weight; CC-BY-NC [S]); a community r=16 DPO LoRA that shortens CoT (−2 pts coding [S]); a 195M-token distilled agentic-reasoning fine-tune [S]. None publish agentic benchmarks beyond their cards.

## III.2 Step 1 — Capture: a recording gateway under your wrapper

**Why a gateway, not Hermes' own export.** Every 2026 "train any harness" system converges on one mechanism: an OpenAI/Anthropic-compatible proxy between the harness and the inference engine that records the exact prompt/response **token IDs and logprobs** per model call, then stitches calls into training rows by token-prefix continuity — Agent Lightning v1.0, rLLM's gateway, NeMo Gym model servers, AReaL's online proxy, slime's adapters, verifiers' interception server, OpenRLHF's executor, ART; MiniMax's Forge describes the same decoupling. And Hermes' built-in trajectory export is **not a faithful record**: it substitutes a fixed Hermes-JSON tool-calling system prompt for the real one, re-renders tool calls as JSON, folds results, and records no token IDs. For Qwen3.8 (XML grammar, effort sentence, preserved thinking) you must capture what vLLM actually rendered. Use `batch_runner.py` to *drive* runs (prompt lists, toolset distributions), never as the data source.

**Architecture:** your wrapper → gateway (records) → vLLM (`return_token_ids` / `prompt_logprobs`, `logprobs=1`). Ready-made: **Agent Lightning** (zero-code `base_url` redirect; rollout id in URL; forces train/val temperature; verl trainer), **rLLM gateway** (`/sessions/{sid}/v1`; SQLite; 6 deps), **AReaL online proxy** (chat/completions + responses + messages; per-session key; `POST /rl/set_reward`; **ships `examples/hermes`** running Hermes `AIAgent` in-process with `skip_memory`, `skip_context_files`). For SFT-only capture a ~200-line FastAPI proxy suffices.

**Log per model call (instrumentation spec):**
- *Identity:* `episode_id` (task × rollout), `trajectory_id` + `agent_role` (orchestrator / each specialist), `step_idx`, `parent_step_id` + `child_episode_id` for `delegate_task` spawns, `attempt_id`.
- *Exact request:* API format, `model`, full `messages` as sent (system incl. memory/skills blocks; assistant turns with `tool_calls` **and** `reasoning_content`; `tool` results with ids), `tools` JSON in exact order, `tool_choice`, `parallel_tool_calls`, `chat_template_kwargs` (`reasoning_effort`, `preserve_thinking`, `enable_thinking`), sampling params, `extra_body`.
- *Exact response:* raw generated text before parsers, `reasoning_content`, `content`, parsed `tool_calls`, `finish_reason`, usage.
- *Tokens:* `prompt_token_ids`, `response_token_ids`, per-token `logprobs`; **fail closed** if missing (slime / NeMo Gym rule: training token IDs must come from the same request that sampled the response).
- *Fingerprint:* tokenizer + chat-template hash, model revision, vLLM version, parser names.
- *Harness events (from your wrapper's hooks, not the gateway):* tool executions (raw result, truncation applied), **compaction events + carrier text**, `memory`/`skill_manage` writes, delegate spawn/return (child ids, returned summary), verifier outputs/rewards, wall-clock, cost.

**Store as ATIF** (Agent Trajectory Interchange Format v1.8 — the schema behind the Agent Data Protocol, ICLR 2026, ~1.3M unified trajectories, ~20% average gains). It already has `reasoning_content`, `reasoning_effort`, `tool_calls` / `observation.results` bound by `source_call_id`, `subagent_trajectory_ref`, `context_management{compaction|pruning}` boundaries, per-step token IDs/logprobs, `is_copied_context` (so compaction turns are excluded from loss) and `final_metrics`. Add in `extra`: `task_id`, `task_family`, `split`, `source{prod_log|demo|synthetic-<pipeline>|self_play, teacher, prompt_version}`, `license`, `environment{harbor_task_ref, mcp_servers[{name,version}], fixture_id}`, `verifier{reward, reward_json, passed}`, `rubric_scores`, `human_review`, `redaction`, `dedup_hash`, `role`, `skills_loaded[]`; per step `irreversible: bool`, `user_confirmation_obtained`. Stage ADP-style: raw → ATIF → std (canonical tool names) → `std_to_sft.py` per renderer.

**Token-exact rules.** Never re-tokenize text; train on captured ids. If you must re-render (converting old logs), render with the *served* `chat_template.jinja` and identical `chat_template_kwargs`, then assert equality against captured ids per turn (verl `tokenization_sanity_check_mode=strict`). **Masking:** loss=1 only on assistant-generated tokens (`<think>…</think>`, visible text, `<tool_call>` XML, `<|im_end|>`); loss=0 on system (effort sentence, tools, memory/skills), user, `<tool_response>`, re-fed prior reasoning, compaction carriers, steering markers; zero-mask malformed turns (Tinker `parse_error_masked`); when a prompt is not a token-prefix extension of the previous call (compaction, history edit, delegate child) start a **new segment** and count shared generated prefixes once.

**Two Hermes-specific fixes before capturing anything:**
1. **Reasoning replay.** Stock Hermes strips `reasoning`/`reasoning_content` from prior assistant messages on custom routes. With Qwen3.8's `preserve_thinking` default that makes serving diverge from the model's own training distribution and from your SFT renders. Patch your wrapper to pass `reasoning_content` back (the model card's own Chat Completions example does exactly this).
2. **Model-name gating.** A served model id containing `qwen` makes Hermes inject "Tool-use enforcement" + execution-discipline blocks. Choose a custom name (or pin `agent.tool_use_enforcement` / `agent.execution_guidance`) and freeze the resulting system prompt across data generation, training and serving. Hermes' tiered system prompt (stable / context / volatile) is never re-rendered mid-session — good for prefix caching and for stable training prompts.

## III.3 Step 2 — Environment: seeded sandboxes and a verifiable task bank

Every executed-episode method needs somewhere to execute, and every verifier needs a known initial state. This is also your eval suite.

- **One sandbox per integration** (CRM, ESP, ads, analytics, CMS, social, DAM, internal KB): mock MCP server or sandbox tenant with seeded fixtures (contacts, campaigns, budgets, analytics rows) and snapshot/reset. Pin MCP server versions.
- **Task layout — adopt Harbor's**: `instruction.md` + `task.toml` (`[environment].mcp_servers`, `skills_dir`, verifier timeout/network) + `tests/test.sh` → `reward.json`; RewardKit TOML for rubric judging (binary/likert/rubric criteria, weighted aggregation). Harbor supports MCP servers and skills directories natively, and both Harbor and NeMo Gym ship a **Hermes harness adapter**, so one task bank drives data generation, RL and eval.
- **How to author tasks (evidence):** MCPMark — 127 tasks, each `meta.json` + `description.md` + `verify.py` over realistic initial states, human–AI co-design + expert cross-review; AppWorld — DB-state checks "without collateral damage", programmatic variant generators (250 scenarios × 3); WorkArena++ — 682 compositional tasks each sampling thousands of configurations; τ² — reward = DB-state match × communication checks × env assertions, derived by replaying a reference action sequence; TheAgentCompany — checkpoint graders for partial credit; **Holo4 — "a task survives only if its verifier rejects near misses"**; **AutoSynthData — positive verification (reference passes) *and* negative verification (mutated wrong outcomes fail)**; **AutoGym — write the solution space and verification rules *before* building the environment: $2–8 and ~15 min per task vs $500–1,000 and 1–1.5 h for expert authoring; 80–92% survive repair [S]**.
- **Marketing task families (~15):** campaign setup & QA · audience/segment building · budget reallocation & pacing · performance reporting with computed numbers · attribution questions · subject-line / creative A/B · content calendar & brief · lifecycle/journey edits · lead routing & CRM hygiene · SEO audit & brief · social scheduling · brand-compliance review · approval requests · incident triage (over-spend, broken UTM) · cross-tool reconciliation (ads ↔ analytics ↔ CRM). Cross with personas (CMO, demand-gen, lifecycle, paid media, SEO, brand, analyst, SDR) and tool states (missing permission, stale data, duplicate campaign, API error, ambiguous ask).
- **Counts:** 100–300 hand-validated tasks (MCPMark / TheAgentCompany scale) and 10³–10⁴ programmatic variants (AppWorld / WorkArena++ scale). Hold out whole families and whole workflows.

## III.4 Step 3 — Generate episodes: the full menu

"Fit" = how directly it applies to your system.

**A. From real usage**

| Method | How, concretely for you | Evidence / scale | Fit |
|---|---|---|---|
| A1 Production-log mining | Gateway-captured sessions → scrub → dedup → rejection-sample with verifiers/judges → SFT; thumbs → KTO; edits/approvals → DPO pairs | Kimi K2 "large-scale rejection sampling"; KTO works on binary labels [S]; Cursor/Shopify loops | **Highest** — your real task distribution |
| A2 Human demonstrations ("teach mode") | Marketers do real tasks in the sandboxes with the agent in shadow/assist; capture via gateway; review/trim UI; rubric-score (RewardKit) | AgentNet recorder pattern (22.6K tasks); expert task authoring costs $500–1,000 [S] → expect tens to low hundreds of gold episodes/month from a few hours/week | High — gold evals, seeds, preference anchors |
| A3 Corrections & overrides | Store (agent output, user edit/override) pairs; annotator-steered live rollouts | UI-TARS online bootstrapping; PRELUDE edit feedback | Medium–high |

**B. Executed synthetic (teacher-driven, verified)**

| Method | How | Evidence / scale | Fit |
|---|---|---|---|
| B1 Teacher runs through *your* wrapper | Frontier/open teacher drives your wrapper via `batch_runner`-style prompt lists against the sandboxes; τ²-style user simulator with structured `known/unknown_info`; keep verifier-passed | APIGen-MT (~5k → τ-bench-leading), Toucan (1.5M, real MCP), Holo (filtered BC), Fara1.5 (~2M samples, 3 verifiers) | **Highest** — the bulk SFT corpus |
| B2 Execution-first → reverse-synthesize the task | Let an agent explore valid tool-call sequences in the sandbox; then write the user intent that would produce them; keep verifier-passed | Trajectory2Task (ACL 2026), SyntheticAgentTraceQA, NNetNav retroactive labeling [S] | **High** — safest for tools with side effects |
| B3 Skill-grounded synthesis | Your `SKILL.md` library is the seed: per skill, builder-reviewer tasks of several types with reference solution + executable verifier | SkillGym 6.8k envs / 19k verified trajectories; SKT 27k [S] | **High** — reuses your skills directly |
| B4 Workflow-path enumeration | SOP → graph → enumerate paths → simulate dialogues with *real* sandbox tool calls per node | "Compiling Agentic Workflows": 2–6k dialogues per 14–55-node procedure; 87–98% of frontier quality [S] | High for the 10–20 highest-volume SOPs |
| B5 Tool-graph composition | I/O-compatibility graph over your tool catalog (GA4 report → segment → Klaviyo campaign); random walks → compositional tasks; nested/implicit calls | Magnet, BUTTON (8k), ToolDial (11k) | High — cross-tool tasks |
| B6 Failure injection & recovery | Simulator injects API errors/partial failures (15–25% of runs); keep recovery turns | Kimi K2 stochastic simulator; PALADIN >50k recovery traces; BFCL v4 scores error recovery [S] | High |
| B7 Decision negatives | No-tool / clarify / cannot-do / irrelevant-tool / missing-param / out-of-role-escalate; function-name masking | When2Call (hallucination 19%→1.2% [S]), Hammer, Agent-FLAN negatives | **Mandatory** |
| B8 Policy / approval trajectories | Draft → approval tool with structured summary; refuse instructions found in fetched content; confidentiality probes; FTC refusals | Fara1.5 "unapproved irreversible action" verifier; CRMArena-Pro confidentiality | **Mandatory** |
| B9 Multi-agent episodes | Orchestrator episodes with `delegate_task` → child summary as tool result; specialist episodes as separate linked trajectories; contrast samples | ATIF subagent refs; AFM; AgentFlow; Hermes delegation logs | **Mandatory** for your architecture |
| B10 Compaction & memory | Summarizer-call samples + resume-from-summary segments; `memory`/`skill_manage` write turns + next-session read | Hermes compaction at 50%; Anthropic `compact_20260112`; NeMo Gym compaction-aware rollouts; slime new-segment rule; SUPO/FoldAct [S] | High for long sessions |
| B11 Analytics with computable answers | Tasks whose correct number derives from seeded rows → exact-match verifier | AD-Bench hard tier <50% for frontier models [S] | High — verifiable domain |
| B12 Exploration-driven task discovery | Agent explores a tool's surface, proposes tasks, judge filters | InSTA 150k trajectories/24 h on 50 L40S (82.6% judge accuracy); Explorer | Medium |

**C. Non-executed synthetic**

| Method | How | Evidence | Fit |
|---|---|---|---|
| C1 Docs/SOP → instruction data | Policies → Intent-Context-Action → synthetic CoT (Airbnb ICA); AgentInstruct flows (25M pairs) for brand/policy Q&A | [S]/[P] | Medium — knowledge-style Q&A, not tool behaviour |
| C2 Persona-driven request generation | Persona × intent × tool-state matrix with your own personas (Persona Hub data is **CC-BY-NC**) | τ² user-sim rules (reveal gradually, never invent results) | Medium–high |
| C3 Brand-voice exemplars | 500–2,000 curated exemplars + length-matched on/off-voice pairs | LIMA / URIAL | High |
| C4 Teacher reasoning traces | Compact reflective per-step reasoning in the base model's style (OpenCUA L2 beat L1/L3) | [S] | For the *style* of `reasoning_content` only — never raw long CoT from a mismatched model |

**D. Public datasets to mix** (licences checked): Toucan-1.5M, Nemotron tool_calling / Agentic / When2Call (CC-BY-4.0), Hermes FC v1 (Apache-2.0), ToolACE (Apache claimed), ADP collection (per-dataset licences), NeMo Gym pivot datasets (CC-BY-4.0). Non-commercial — imitate only: APIGen-MT, Magpie, xlam-60k, ToolBench, ActionStudio data, Persona Hub.

**E. Self-evolution (month 3+)**: AutoSynthData-style failure-targeted task generation with positive **and** negative verification (2,000 samples → +7.2 Pass@1 [S]); hermes-agent-self-evolution (GEPA over `SKILL.md`, ~$2–10/run); avoid pseudo-label-only self-play for open-ended marketing outputs.

**Order for your situation** (logs exist · SOPs documented · many MCP integrations · sandboxable tools · marketers a few hours/week): A1 → task bank (§III.3) → B1 + B7 + B8 → B9 + B10 → B3 → B2 / B5 → B4 (top SOPs) → A2 continuously → E later.

## III.5 Step 4 — Verify and curate

Filter stack, in order: (1) **rules** — schema-valid XML/args, no unknown tools/params, call order intact, no failure patterns, reasoning present (Hermes drops zero-reasoning samples); (2) **state verifier** — end state matches blueprint, no collateral damage (AppWorld), communication checks (τ²); (3) **Fara1.5's three verifiers** — universal LLM judge on the rubric (order-swapped, "Unknown" option), efficiency / redundant-action filter, **critical-point adherence** (missing info, ambiguity, unapproved irreversible action → fail); (4) dedup (MinHash + semantic); (5) PII scrub (Presidio + allow/deny lists + hand sampling); (6) n-gram decontamination against the eval suite; (7) **human audit ≥600 samples at ≥95% accept**. Split by **task family, workflow and integration version**; hold out whole SOPs and whole roles. Expect **10–20k verified company episodes for v1** (APIGen-MT ~5k, ToolACE 11.3k, SWE-smith 5k of 26k → +32 pts); scale to 50–100k via B2/B3/B5 for v2.

## III.6 Step 5 — Represent multi-agent, workflows and long sessions

**Multi-agent (orchestrator + specialists)**
- **One model, role by system prompt** — the evidence-backed default: MiniMax trains one model across >100k scaffolds and reports out-of-distribution harness generalization; AFM distils a multi-agent system into one stream; Hermes itself runs orchestrator and leaf children on one model with different system prompts and toolsets.
- **Separate, linked trajectories per role** (ATIF `subagent_trajectory_ref`): the orchestrator's sample contains `delegate_task` `<tool_call>` → `<tool_response>` with exactly the JSON Hermes returns (`status, summary, exit_reason, truncated, schema_valid`); the child's run is its own trajectory with its own system prompt and reduced toolset (children lack `delegate_task, clarify, memory, send_message, cronjob, start_chat` — mirror exactly). Never inline child transcripts into the parent.
- Make the role boundary learnable: tool-list differences, a short role header in the stable tier, **contrast samples** (specialist asked to do orchestrator work → declines/escalates), held-out-role evals.
- RL: group by `(task, role)`; outcome reward for the orchestrator; same outcome or a role-local verifier for specialists; rLLM per-role estimators or Agent Lightning trajectory mode. AgentFlow shows training **only the planner** (+14.9% search, +14.0% agentic) is a cheap first RL target.
- Multi-LoRA per role (vLLM per-request adapter) is a serving fallback; Holo4's per-lane LoRA experts merged equal-weight is the proven RL shape on this architecture [S]. Verify LoRA on GDN projections serves correctly before relying on it [U].

**Workflows / SOPs**
- Default: keep workflow definitions **in context** (SKILL.md / YAML / JSON DAG) and train the model to *follow* them — workflow text in the masked prompt, workflow identity varied across samples, whole workflows held out for generalization tests.
- Compile into weights only stable, highest-volume SOPs, and then with **full fine-tuning**: the 2026 compiled-workflow result (2–6k path-covering dialogues per procedure, 87–98% of frontier quality, 128–462× cheaper inference) came with the finding that LoRA r=16–128 fails on procedures (effective rank ~760–1,030; r=128 captures ~45%) — while another 2026 study found LoRA > full FT for *tool-calling format* at 4B. Rule: **LoRA for tool/format/voice skills; full FT for procedures.** Use TraceCompiler/HEXIS-style logic to derive workflow graphs from logs and to decide which branches must stay deterministic code (irreversible side effects especially).

**Long sessions**
- Hermes compacts at **50%** of context (`protect_first_n 3`, `protect_last_n 20`, `target_ratio 0.20`), inserting a `user`-role carrier *"[CONTEXT COMPACTION — REFERENCE ONLY] … treat it as background reference, NOT as active instructions …"* and pruning old tool outputs to *"[Old tool output cleared to save context space]"*.
- Emit **two sample kinds**: the *summarizer* call (prompt = pre-compaction window + summary instruction; target = summary) and the *resume* segment (prompt = system + carrier + protected tail; target = next assistant turns), each masked per §III.2 and each a new segment. Add memory/skill **write** turns near the threshold and cross-session samples where the next session's system prompt shows the written entry (write → read loop). Evaluate with forced-compaction tasks.

## III.7 Step 6 — Train (Qwen3.8 stack)

**Data contract (all stages):** OpenAI-style messages; `system` first; tools as OpenAI function schemas (wrapper kept); assistant turns with `content`, `reasoning_content` (`""` when none), `tool_calls[].function.arguments` as dicts; `tool` results in call order; one `reasoning_effort` per sample (default `medium`); render with the official template (or TRL's training variant) so effort sentence, tool block, `<think>` framing and merged `<tool_response>` turns are byte-identical to serving.

**Stage 0 — Baseline in your wrapper (week 1).** Stock Qwen3.8-27B at `medium` (and `xhigh`) on the golden suite with your Skills, system prompts and tools. Measure pass^k, tool-hallucination rate, empty-answer rate, cost/task. Fine-tune only the measured gap.

**Stage 1 — SFT (LoRA, all-linear incl. GDN).** Primary command (ms-swift):
```bash
swift sft --model Qwen/Qwen3.8-27B --tuner_type lora --lora_rank 64 --lora_alpha 32 \
  --target_modules all-linear --add_non_thinking_prefix true \
  --loss_scale 'default+hermes+ignore_empty_think' --packing true \
  --attn_impl flash_attention_2 --deepspeed zero2 --torch_dtype bfloat16 \
  --learning_rate 1e-4 --warmup_ratio 0.05 --num_train_epochs 2 --max_length 32768 \
  --dataset your_episodes.jsonl   # rows carry {"chat_template_kwargs": {"reasoning_effort": "medium"}}
```
Hyperparameters per Part I (r=64–128, LR 1e-4 sweep 5e-5–2e-4, effective batch 16–32, 2–3 epochs, select by eval). Mix: company episodes 30–40% · public tool data 30–40% · base-regenerated general replay 20–30% · voice + negatives fixed slices. Cheap iteration: Unsloth 27B notebook (QLoRA, never fp16). Full FT (compiled workflows): NeMo AutoModel `qwen3_8_27b.yaml` on 8×H100 or Megatron-SWIFT.

**Stage 2 — Recover general ability (OPD, 100–200 steps).** Teacher = original Qwen3.8-27B via `swift deploy --model Qwen/Qwen3.8-27B --infer_backend vllm --max_logprobs 64`; student: `swift rlhf --rlhf_type gkd --model <stage1> --teacher_model_server http://… --gkd_logits_topk 64 --tuner_type lora --use_vllm true --vllm_mode colocate --lmbda 1 --beta 0.5 --max_completion_length 8192` on general prompts. Gate on IFBench/IFEval.

**Stage 3 — Preference.** Length-normalized DPO on success-vs-failed, correct-vs-wrong decision, on/off-voice pairs; KTO on production thumbs. Evidence it works on this model: a community r=16 DPO LoRA [S].

**Stage 4 — RL through your harness.** Choose by infra:

| Option | Mechanism | Trainer | Model support | LoRA | When |
|---|---|---|---|---|---|
| **slime** | Anthropic/Responses/OpenAI adapters; token-in/token-out; `TrajectoryManager` masks tool/user, bridges prefixes, new segment on compaction | Megatron + SGLang | **Qwen3.8-27B example exists** (16 GPUs + 2 sandbox nodes) | no | Full-FT RL, Megatron-class GPUs |
| **Agent Lightning** | Zero-code gateway (`base_url` redirect), rollout id in URL, `include_log_probs`, trajectory-mode stitching by token continuity | verl + vLLM | Qwen3.5 examples (same arch) | not documented | Fastest zero-code path |
| **AReaL online proxy** | chat/completions + responses + messages; per-session key; `POST /rl/set_reward`; **`examples/hermes`** | Megatron / FSDP | Qwen3.5 listed | yes | Hermes-native example |
| **rLLM** | Custom harness class or `@rllm.rollout`; per-role trajectories `{task_id}:{name}` with per-role estimators | verl / **Tinker** | via backend | via Tinker | Per-role credit assignment |
| **Tinker** | Hosted LoRA RL/OPD; Qwen3.8 effort renderers; multi-turn tool-use recipes | hosted | **Qwen3.8-27B** | LoRA only | Zero infra |
| **prime-rl + verifiers** | Interception server; `[renderer] qwen3.8`; async | FSDP2 + vLLM | Qwen3.5 listed (confirm 3.8 load) | yes | What Holo4 used [S] |
| **NeMo Gym + NeMo RL** | `hermes_agent` harness adapter; `ng_trajectory`; 2-node Qwen3.8-27B GRPO yaml | NeMo RL | Qwen3.8-27B | yes | Multi-node; Nous fork available |

Rewards and settings per Part I Stage 4 (gated decomposed rewards, −1 for unapproved irreversible actions, efficiency term, token budgets, DAPO components, LoRA r≤32, PTX/OPD anchor, shuffled-reward ablation). Group by `(task, role)`; start with planner-only RL (AgentFlow), then specialists or Holo4-style per-lane experts.

**Stage 5 — Merge → quantize → eval served → serve.** Merge to BF16 (confirm `mtp.*` tensors and `tokenizer_class` survive — Unsloth #7681 class of bugs); FP8-dynamic via llm-compressor (keep `visual.*`, `lm_head`, `embed_tokens`, `linear_attn.in_proj_a/b` unquantized); eval the served artifact; serve with the §III.1 command; client-side tool-name validation + de-dup in your wrapper (#58147); `frequency_penalty 0.3` if empty answers appear.

## III.8 Step 7 — Evaluate (the ops suite)

- **Seeded sandbox per integration**, task = `{meta, description, init_state, verify.py}` (MCPMark layout); verifiers check end state **and no collateral damage** (AppWorld) and, for customer-facing actions, **communication checks** (τ²). Headline metric **pass^k (k=3–4)**; also pass@1, avg@k, steps, tokens, wall-clock, **cost/task** (Hermes Index style). Best-of-N only for rejection sampling, never for reporting.
- **Multi-agent slices:** orchestrator-only (delegation plan correct, child results schema-valid), specialist-only (role boundary: refuses out-of-role tools / escalates), end-to-end; **held-out workflows** (unseen SOP/YAML) and **held-out roles**.
- **Long-horizon slice:** forced compaction (cap context so ≥1 compaction occurs); **cross-session memory** (write in A, must use in B).
- **Rubric + human:** Claw-Eval-style human-verified rubrics (300 tasks / 2,159 rubric items; Completion / Safety / Robustness; Pass^3) graded by an LLM judge, with a fixed human sample per release and inter-rater checks; TheAgentCompany-style sub-checkpoints for partial credit.
- **Regression & hygiene:** IFBench/IFEval, MMLU-Pro, GPQA within ±1–2 pts; BFCL v4 subset, τ²-bench; pin environment versions and verifiers; re-verify stored rollouts; **use the same harness + gateway for eval and training** so numbers transfer.

## III.9 Costs and timeline (architecture identical → Part I/II numbers hold)

- SFT v1 (10–20k episodes × ~8k tokens, 2–3 epochs): **$250–400** on 1×H100/H200 (LoRA) or ~$220 on 8×H100; Unsloth 4-bit fits 2×16 GB at 1k seq for smoke tests.
- Full FT for compiled workflows (≤500M tokens): **~$340–450** on 8×H100/H200 (NeMo AutoModel config exists).
- OPD recovery: 100–200 steps, **<$300**.
- RL through the harness: **$2.2–4.3k per 500 steps** on one 8×H100 node (LoRA path); sandboxes extra ($0.17–0.46/VM-h).
- Task bank: 100–300 tasks at AutoGym-style $2–8 each in LLM cost + human review ≈ **$1–3k** + engineer time (expert-authored tasks are $500–1,000 each [S] — automate authoring, keep humans for review).
- Teacher episode generation: 10–100k episodes; cost dominated by teacher inference and sandbox time (InSTA: 150k trajectories / 24 h on 50 L40S) — **$2–10k** for v1 depending on teacher (open-weight teacher avoids provider-terms questions; Part I §4.3).

| Weeks | Deliverable |
|---|---|
| 0–2 | Gateway capture live in your wrapper (ATIF store, redaction, dedup); reasoning-replay patch; model-name / system-prompt freeze; Stage-0 baseline at `medium` vs `xhigh` |
| 1–6 | Sandboxes per integration (Harbor layout); 100–300 verified tasks + variants; user simulator; teacher episodes (B1), negatives (B7), policy (B8), multi-agent (B9), compaction (B10) |
| 4–8 | Log mining (A1) → SFT/KTO/DPO sets; skill-grounded (B3) and reverse-synthesis (B2) batches; human teach-mode (A2) weekly |
| 6–9 | Stage 1 SFT + Stage 2 OPD; eval gates; iterate on failure clusters |
| 8–11 | Stage 3 DPO/KTO; merge → FP8 → served eval; shadow mode read-only in production |
| 10–16 | Stage 4 RL through the harness (planner-first, then specialists / per-lane experts); workflow compilation (B4) for top SOPs with full FT; A/B with gated writes |
| ongoing | Flywheel: production → scrub → judge → SFT/KTO/DPO → re-gate; GEPA skill evolution; failure-targeted task synthesis (E) |

## III.10 What I need from you to make this concrete
1. Your wrapper's tool schemas (exact JSON), per-role system prompts, and one captured example of the request your wrapper actually sends.
2. Workflow/SOP definitions (format + the 10–20 highest-volume ones).
3. Volume/format of existing logs; whether thumbs/edits/approvals are captured.
4. Which integrations are sandboxable today; which writes the agent should perform vs draft.
5. GPU access/budget; whether hosted LoRA (Tinker) is acceptable.
6. Legal position on teacher-model terms and customer data in training.

## III.11 Caveats specific to this Part
- Qwen3.8 template facts are from TRL's vendored copy of the shipped template (TRL runs a weekly drift check; none detected) — re-diff against the HF file before training.
- Holo4 training details (prime-rl, LoRA experts, merge) are from H's newsroom via search snippets [S]; the model card itself was unreachable.
- 2026 arXiv papers cited (compiled workflows, LoRA-on-procedures, Trajectory2Task, SkillGym/SKT, AutoGym, AutoSynthData, SFT-vs-RL study, Rethinking OPD, SUPO/FoldAct) are [S] — open them before quoting numbers.
- prime-rl / Agent Lightning / AReaL list Qwen3.5 (same architecture) but not 3.8 by name — confirm the hybrid loads in the trainer before committing GPUs.
- Whether vLLM serves *unmerged* LoRA on GDN projections correctly is unverified — merge for production.

## III.12 Glossary
**SFT** supervised fine-tuning: imitate recorded good turns · **DPO / KTO** preference training from (better, worse) pairs / binary thumbs · **OPD / GKD** on-policy distillation: student samples, teacher's token distribution corrects it; used here to restore general ability after domain SFT · **GRPO / DAPO** RL that scores groups of sampled episodes against verifiers/judges · **LoRA** small trainable adapters; **full FT** all weights · **Verifier** program that checks sandbox end state · **Judge** LLM scoring a rubric · **pass^k** probability all k attempts succeed (reliability) · **Gateway** proxy recording exact model calls with token IDs · **ATIF / ADP** standard trajectory schema / pipeline · **Compaction** Hermes summarizing old context at 50% · **reasoning_effort** Qwen3.8's `xhigh|medium|low` thinking-budget sentence · **preserve_thinking** keep prior-turn reasoning in the prompt · **Harbor** containerized task + verifier format with MCP support · **GDN** Gated DeltaNet linear-attention layers (48 of 64 in this model).

---

## Appendix A — Flagged / unverified items
- **Part III:** see §III.11.
- **Part II:** Holo4-27B and Holo2-30B-A3B licences [U]; all Holo compute/cost reconstructions [U-arith]; Qwen3.6-27B OSWorld-Verified 63.9 is relayed from the Qwen3.8-27B card [S]; Holo1.5 release date conflicts across H pages; whether vLLM accepts `image_url` parts on `tool`-role messages for Qwen3.6 [U]; sandbox/browser pricing figures [S].
- Paper numbers marked [S] (Gekhman, Ovadia, EntiGraph, LoRA-Without-Regret blog figures, "Procedural Knowledge Is Not Low-Rank", OPD recovery numbers, Ibrahim replay ratios, When2Call deltas, ToolRL deltas, KTO claims, τ-bench numbers, Creativity Benchmark, AD-Bench, CRMArena-Pro) come from abstracts/snippets — arxiv.org and huggingface.co were blocked in this sandbox.
- Unsloth 56 GB LoRA figure and Tinker pricing/retirement dates [U].
- Meta's official Ads MCP details (29 tools, admin budget caps) rest on third-party sources [U].
- Whether vLLM LoRA kernels cover GDN projections; whether bnb-4bit handles DeltaNet projections [U].
- No published documented agentic SFT recipe for Qwen3.6-27B exists yet; closest is Mercor's SkyRL RL run on Qwen3.6-35B-A3B [S].

## Appendix B — Key sources
**Part III:** Qwen3.8-27B model card (verbatim mirror at github.com/khimaros/aimbot/…/Qwen_Qwen3.8-27B.md); TRL `trl/chat_templates/qwen3_8.jinja` + `qwen3_8_training.jinja` + README; QwenLM/Qwen3.8 README; vLLM recipe `Qwen/Qwen3.8-27B` + `chat_completion/protocol.py` + issues #58147/#55420; QwenLM/Qwen3.8 issues #216/#217; ms-swift `Qwen3_8-Best-Practice.md`, `swift/template/templates/qwen.py`, PRs #9914/#9920/#9972; LLaMA-Factory `template.py`/`tool_utils.py` (`qwen3_8`); Axolotl `examples/qwen3.8/27b-qlora.yaml` (PR #3935); Unsloth notebooks `Qwen3.8_(27B)-Conversational/GRPO` + issues #10010/#12737/#7681; NeMo AutoModel `qwen3_8_27b{,_lora}.yaml`, Megatron-Bridge `qwen3.8-27b.md`, NeMo RL `grpo-qwen3.8-27b-2n8g-megatron-tp4pp2cp2.yaml`; Tinker cookbook `model_info.py`, `renderers/qwen3_8.py`; verl `run_qwen3_8_27b_megatron.sh`, `docs/advance/agent_loop.rst`; prime-rl `docs/training.md`, PrimeIntellect renderers `qwen38.py`; slime `examples/coding_agent_rl`; llm-compressor `qwen3_8_gptq_awq_example.py`; Agent Lightning docs (05/20/25/35); rLLM docs + model gateway; NeMo Gym `responses_api_agents/hermes_agent`, `vllm_model_with_compaction`; AReaL `online_proxy.md` + `examples/hermes`; SkyRL; verifiers v1 docs; OpenPipe ART; OpenRLHF; ROLL; Hermes Agent source (`batch_runner.py`, `agent/agent_runtime_helpers.py`, `agent/context_compressor.py`, `agent/compaction_events.py`, `agent/system_prompt.py`, `tools/delegate_tool.py`, `tools/memory_tool.py`, `tools/skill_manager_tool.py`); Harbor RFC 0001 (ATIF) + RFC 0002 + RewardKit; neulab/agent-data-protocol (`schema/atif.py`, consistency audit); OPPO Agent Foundation Models; lupantech/AgentFlow; microsoft/fara (FaraGen1.5); MCPMark; AppWorld; TheAgentCompany; τ²-bench; WorkArena; SWE-smith; NVIDIA-NeMo/Gym; xlang-ai AgentTrek/OpenCUA/AgentNet; data-for-agents/insta; NNetNav; Synatra; AgentGym(-RL); R-Zero; Absolute Zero; persona-hub; AgentTuning; Agent-FLAN; AgentRefine; Klear-AgentForge; hermes-agent-self-evolution; OTel GenAI semantic conventions; OpenAI Agents SDK tracing; Claw-Eval; SkillsBench. Secondary [S]: arXiv 2605.22502, 2607.21612, 2609.17848, 2604.13016, 2608.02680 (TraceCompiler), 2609.30123 (HEXIS), 2609.37539/2608.02287 (SkillGym/SKT), 2609.22592 (AutoGym), EnterpriseOps-Gym/AutoSynthData, Trajectory2Task, SyntheticAgentTraceQA, SUPO, FoldAct, Context-Folding, Agent Lightning paper 2508.03680, CoA 2508.13167; hcompany.ai newsroom (Holo4).

**Part II:** H Company GitHub (`hcompai/surfer-h-cli`, `hai-cookbook`, `holo-desktop-cli`, `hai-agents-python`, `agents-api-demos`, `android-world-traces`); Holo model cards / H newsroom / HF blogs via search (Holo1 arXiv 2506.02865; Surfer 2 arXiv 2510.19949; hcompany.ai/newsroom/holo3, holo3.1, holo4); ByteDance UI-TARS paper PDF (repo); OpenCUA repo; QwenLM/Qwen3-VL cookbooks (`computer_use.ipynb`, `agent_function_call.py`); Tongyi-MAI/MAI-UI; X-PLUG GUI-Owl-1.5 README; Aguvis, ScaleCUA, GUI-Actor, GUI-R1 (`r1gui.py` reward), UI-R1, GTA1, ZeroGUI, ARPO scripts; OSWorld / WindowsAgentArena / AndroidWorld / WebVoyager repos; Agent S3 paper (EC2 scaling data); Hermes Agent `tools/browser_tool_vision.py`, `tools/vision_tools.py`, `agent/image_routing.py`, `tools/computer_use/schema.py`, docs `browser.md`/`vision.md`/`computer-use.md`; transformers `configuration_qwen3_5.py`; vLLM recipes `Qwen3.5.md`/`Qwen3-VL.md`, vLLM issue #40696; ms-swift, LLaMA-Factory, Axolotl, EasyR1, verl (`qwen3_5.py` patch) docs.

**Part I:** Qwen3.6-27B model card (via mirror); TRL `trl/chat_templates/qwen3_6.jinja`; QwenLM/Qwen3.8 README; vLLM recipes `Qwen/Qwen3.5.md`; vLLM `tool_calling.md`, `reasoning_outputs.md`, `automatic_prefix_caching.md`; ms-swift Qwen3.8 Best Practice; Axolotl `examples/qwen3.5/`; TRL PR #7533/#7599; NousResearch/hermes-agent docs (`providers.md`, `agent-loop.md`, `trajectory-format.md`, `skills.md`, `memory.md`, `prompt_builder.py`, `chat_completions.py`, `batch_runner.py`); NousResearch/Gym, NousResearch/RL; Kimi K2 tech report PDF; Thinking Machines "LoRA Without Regret" (via TRL guide + replication) and "On-Policy Distillation" (via Tinker cookbook); Tülu 3 docs; Toucan datagen; APIGen-MT; BFCL repo; τ²-bench repo; Sierra pass^k code; Anthropic "Writing effective tools for agents", "Advanced tool use", "Agent Skills", "Demystifying evals for AI agents"; Shopify Engineering "Building production-ready agentic systems"; OWASP LLM Top 10; MCP spec 2025-06-18; EDPB Opinion 28/2024 summaries; Presidio.
