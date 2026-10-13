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

## Appendix A — Flagged / unverified items
- **Part II:** Holo4-27B and Holo2-30B-A3B licences [U]; all Holo compute/cost reconstructions [U-arith]; Qwen3.6-27B OSWorld-Verified 63.9 is relayed from the Qwen3.8-27B card [S]; Holo1.5 release date conflicts across H pages; whether vLLM accepts `image_url` parts on `tool`-role messages for Qwen3.6 [U]; sandbox/browser pricing figures [S].
- Paper numbers marked [S] (Gekhman, Ovadia, EntiGraph, LoRA-Without-Regret blog figures, "Procedural Knowledge Is Not Low-Rank", OPD recovery numbers, Ibrahim replay ratios, When2Call deltas, ToolRL deltas, KTO claims, τ-bench numbers, Creativity Benchmark, AD-Bench, CRMArena-Pro) come from abstracts/snippets — arxiv.org and huggingface.co were blocked in this sandbox.
- Unsloth 56 GB LoRA figure and Tinker pricing/retirement dates [U].
- Meta's official Ads MCP details (29 tools, admin budget caps) rest on third-party sources [U].
- Whether vLLM LoRA kernels cover GDN projections; whether bnb-4bit handles DeltaNet projections [U].
- No published documented agentic SFT recipe for Qwen3.6-27B exists yet; closest is Mercor's SkyRL RL run on Qwen3.6-35B-A3B [S].

## Appendix B — Key sources
**Part II:** H Company GitHub (`hcompai/surfer-h-cli`, `hai-cookbook`, `holo-desktop-cli`, `hai-agents-python`, `agents-api-demos`, `android-world-traces`); Holo model cards / H newsroom / HF blogs via search (Holo1 arXiv 2506.02865; Surfer 2 arXiv 2510.19949; hcompany.ai/newsroom/holo3, holo3.1, holo4); ByteDance UI-TARS paper PDF (repo); OpenCUA repo; QwenLM/Qwen3-VL cookbooks (`computer_use.ipynb`, `agent_function_call.py`); Tongyi-MAI/MAI-UI; X-PLUG GUI-Owl-1.5 README; Aguvis, ScaleCUA, GUI-Actor, GUI-R1 (`r1gui.py` reward), UI-R1, GTA1, ZeroGUI, ARPO scripts; OSWorld / WindowsAgentArena / AndroidWorld / WebVoyager repos; Agent S3 paper (EC2 scaling data); Hermes Agent `tools/browser_tool_vision.py`, `tools/vision_tools.py`, `agent/image_routing.py`, `tools/computer_use/schema.py`, docs `browser.md`/`vision.md`/`computer-use.md`; transformers `configuration_qwen3_5.py`; vLLM recipes `Qwen3.5.md`/`Qwen3-VL.md`, vLLM issue #40696; ms-swift, LLaMA-Factory, Axolotl, EasyR1, verl (`qwen3_5.py` patch) docs.

**Part I:** Qwen3.6-27B model card (via mirror); TRL `trl/chat_templates/qwen3_6.jinja`; QwenLM/Qwen3.8 README; vLLM recipes `Qwen/Qwen3.5.md`; vLLM `tool_calling.md`, `reasoning_outputs.md`, `automatic_prefix_caching.md`; ms-swift Qwen3.8 Best Practice; Axolotl `examples/qwen3.5/`; TRL PR #7533/#7599; NousResearch/hermes-agent docs (`providers.md`, `agent-loop.md`, `trajectory-format.md`, `skills.md`, `memory.md`, `prompt_builder.py`, `chat_completions.py`, `batch_runner.py`); NousResearch/Gym, NousResearch/RL; Kimi K2 tech report PDF; Thinking Machines "LoRA Without Regret" (via TRL guide + replication) and "On-Policy Distillation" (via Tinker cookbook); Tülu 3 docs; Toucan datagen; APIGen-MT; BFCL repo; τ²-bench repo; Sierra pass^k code; Anthropic "Writing effective tools for agents", "Advanced tool use", "Agent Skills", "Demystifying evals for AI agents"; Shopify Engineering "Building production-ready agentic systems"; OWASP LLM Top 10; MCP spec 2025-06-18; EDPB Opinion 28/2024 summaries; Presidio.
