# Fine-tuning Qwen3.6-27B into a marketing-specialist agent for Hermes Agent

**Research-backed plan — research date 2026-10-12**

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
- Teacher = a frontier agentic model running **inside Hermes Agent** against the real sandbox tools (`--distribution` to vary toolsets, `--ephemeral_system_prompt` for teacher-only hints, `--max_turns`), persona-conditioned user simulator with best-of-N.
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
# production serve (text-only agent)
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

## Appendix A — Flagged / unverified items
- Paper numbers marked [S] (Gekhman, Ovadia, EntiGraph, LoRA-Without-Regret blog figures, "Procedural Knowledge Is Not Low-Rank", OPD recovery numbers, Ibrahim replay ratios, When2Call deltas, ToolRL deltas, KTO claims, τ-bench numbers, Creativity Benchmark, AD-Bench, CRMArena-Pro) come from abstracts/snippets — arxiv.org and huggingface.co were blocked in this sandbox.
- Unsloth 56 GB LoRA figure and Tinker pricing/retirement dates [U].
- Meta's official Ads MCP details (29 tools, admin budget caps) rest on third-party sources [U].
- Whether vLLM LoRA kernels cover GDN projections; whether bnb-4bit handles DeltaNet projections [U].
- No published documented agentic SFT recipe for Qwen3.6-27B exists yet; closest is Mercor's SkyRL RL run on Qwen3.6-35B-A3B [S].

## Appendix B — Key sources
Qwen3.6-27B model card (via mirror); TRL `trl/chat_templates/qwen3_6.jinja`; QwenLM/Qwen3.8 README; vLLM recipes `Qwen/Qwen3.5.md`; vLLM `tool_calling.md`, `reasoning_outputs.md`, `automatic_prefix_caching.md`; ms-swift Qwen3.8 Best Practice; Axolotl `examples/qwen3.5/`; TRL PR #7533/#7599; NousResearch/hermes-agent docs (`providers.md`, `agent-loop.md`, `trajectory-format.md`, `skills.md`, `memory.md`, `prompt_builder.py`, `chat_completions.py`, `batch_runner.py`); NousResearch/Gym, NousResearch/RL; Kimi K2 tech report PDF; Thinking Machines "LoRA Without Regret" (via TRL guide + replication) and "On-Policy Distillation" (via Tinker cookbook); Tülu 3 docs; Toucan datagen; APIGen-MT; BFCL repo; τ²-bench repo; Sierra pass^k code; Anthropic "Writing effective tools for agents", "Advanced tool use", "Agent Skills", "Demystifying evals for AI agents"; Shopify Engineering "Building production-ready agentic systems"; OWASP LLM Top 10; MCP spec 2025-06-18; EDPB Opinion 28/2024 summaries; Presidio.
