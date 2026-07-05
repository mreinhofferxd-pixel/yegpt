# TICKET-10 (stretch) — Scale & ablate (next-session handoff)

> Self-contained handoff. Read `SPEC.md` first (source of truth, esp. §2 "Revisit if" + §5
> TICKET-10), then this. Working style is unchanged: **one sub-step at a time → implement → write
> its test → all three checks green via the venv → STOP for review.** TICKET-09 is DONE.

## 0. Decision on the record — read before anything (the honest framing)

TICKET-09 already established the key fact this ticket has to respect: **the val floor is ~1.58–1.59
and it is set by the DATA, not the model size.** run1 (3.28M params) and run3 (1.87M) plateaued at
the *same* val; run1's extra capacity went into memorizing the train split (train→0.89, val turned
*up* to 1.68), not into generalization. The corpus is final at **0.67MB / vocab 104** and no more
is coming (SPEC §4's ~1MB gate is consciously waived).

So SPEC §2's "scale toward 10M for better output" must be read honestly **for this corpus**:
pushing to ~10M params will **deepen memorization, not lower val.** Expect train loss → very low,
val flat or *worse*, and the best-val checkpoint (now saved automatically — see §1) staying near
1.58. **Do not chase coherent lyrics; you will not get them by scaling on 0.67MB.** The real,
honest deliverables of TICKET-10 are:

1. **Quantify the data-bound regime:** show that as params/context grow at fixed data, val does not
   improve while train collapses — memorization, measured.
2. **Note where the 4080 caps out** (SPEC §5): the practical limit here is **time-per-checkpoint
   and context length**, not param VRAM (a char nanoGPT is tiny). Push an axis until it's slow or
   OOMs and report the numbers.
3. **(Optional) Contrast tokenization** (char vs BPE/word): the one lever that actually changes the
   coherence-per-parameter tradeoff, because it changes sequence length and what a "token" is.

No RL anywhere (SPEC §7). If any step starts implying scale will produce good lyrics, correct it in
one line.

## 1. Current state (TICKET-09 done; what you inherit)

- **Corpus:** `data/corpus.txt` = **674,050 chars / 0.67MB / vocab 104**, already deduped in place.
  *** FOOTGUN: do NOT run `python -m yegpt.data_prep` — it rebuilds `corpus.txt` from `data/raw/`
  WITHOUT dedup (back to 751,399 chars). The corpus is FINAL; just train on it. The only correct
  rebuild is `data_prep` THEN `python -m yegpt.dedup` (in place). ***
- **Settled config (run3):** `--n-layer 4 --n-head 4 --n-embd 192 --block-size 256 --dropout 0.2
  --lr 3e-4 --max-iters 5000` → ~1.87M params, val ~1.59. This is the TICKET-10 baseline to beat
  (you won't, on this data — that's the point).
- **train CLI** (sweep via flags, NEVER by editing `TrainConfig` defaults): `--n-layer --n-head
  --n-embd --block-size --dropout --lr --max-iters --eval-interval --batch-size --device --seed
  --out-dir --corpus`. Each run → its own `--out-dir`.
- **Two checkpoints per run (built in 09):** `<out-dir>/yegpt-ckpt.pt` (final step) and
  `<out-dir>/yegpt-best.pt` (lowest val). `TrainResult` carries `best_checkpoint_path/best_loss/
  best_step`. **Use the best-val checkpoint for any quality comparison** — for an overfit run the
  final is the most-memorized.
- **Sampling knobs (built in 09):** `model.generate(..., temperature=1.0, top_k=None)`;
  `sample.py` flags `--temperature --top-k` (plus `--checkpoint --prompt -n --seed --device`).
  `model` owns the sampling policy; `sample.py` only threads flags through.
- **Architecture (`model.py`, hand-written):** Head / MultiHeadAttention / FeedForward / Block / GPT,
  pre-norm, manual causal attention. Params ≈ `12 * n_embd² * n_layer` + embeddings + lm_head.
  Positional embedding is a **learned table of `block_size` rows**, so raising `--block-size`
  enlarges it and makes attention O(T²); `generate` already crops context to the last `block_size`.
- **Checks (all must stay green):** `.venv\Scripts\python.exe -m pytest` (currently **99**) /
  `-m ruff check .` / `-m mypy`. Checkpoints are gitignored.

## 2. Hard constraints (unchanged)

No `nn.Transformer` / `nn.MultiheadAttention` / HuggingFace model classes. No RL/GRPO/DPO/PPO.
Strict PEP 484, **no `Any`** (`object` ok). Small modules, DI over globals, frozen+slots
dataclasses, code-first with intent-comments. ruff selects E,F,I,UP,B,N — **N806 bans UPPERCASE
locals**. mypy strict + `disallow_any_explicit`/`disallow_any_generics`/`warn_return_any`/
`warn_unreachable`. **Minutes-per-checkpoint on the 4080** (SPEC §0) — this is the gating
constraint for TICKET-10; if a run goes to many minutes, say so and shrink rather than push through.
**Keep the training loop fixed across the sweep** (sweep only via CLI flags) so the ablation is a
clean comparison. If a large model diverges/NaNs, that's a finding — consider grad clipping / LR
warmup as a *flagged* decision, don't silently add them.

## 3. Work order (each sub-step: implement → test → green → STOP for review)

### 10.0 — Instrument the loop (throughput + peak VRAM) *(TICKET-07 surface change — flag it)*

"Note where the 4080 caps out" needs numbers the loop doesn't currently produce. Add lightweight
instrumentation to `train()`: wall-clock over the optimizer loop → **steps/sec and tokens/sec**
(`tokens = steps * batch_size * block_size`), and **peak VRAM** via
`torch.cuda.max_memory_allocated` (reset at start; report 0 on CPU so the smoke test stays CUDA-free).
Surface them on `TrainResult` (e.g. `steps_per_sec: float`, `tokens_per_sec: float`,
`peak_vram_bytes: int`) and print a one-line summary at the end. **Do not** use `Date.now`-style
wall-clock anywhere except this measured region; keep it typed, no `Any`.
- **Test:** the new fields are finite/non-negative and present; existing tests still pass; CPU run
  reports `peak_vram_bytes == 0`. Keep the change minimal — don't perturb the loss path.

### 10.1 — Scale & context ablation (the core)

Sweep ~3–4 configs, **fixed corpus, fixed `--max-iters 5000 --eval-interval 250`** for a fair
comparison, each to its own `--out-dir`. Suggested points (verify param counts via the printed
`num_parameters()`; `n_embd % n_head == 0` is enforced):

| label | flags | ~params |
|------|-------|---------|
| baseline | (run3) `--n-layer 4 --n-head 4 --n-embd 192 --block-size 256` | 1.9M |
| mid | `--n-layer 4 --n-head 8 --n-embd 320 --block-size 256` | ~5M |
| large | `--n-layer 6 --n-head 6 --n-embd 384 --block-size 256` | ~10M |
| long-ctx | `--n-layer 6 --n-head 6 --n-embd 384 --block-size 512` | ~10M, 2× context |

Record per run: **best val** (the floor — expect ~flat across scales), **final train** (memorization
depth — expect lower as params grow), **steps/sec, tokens/sec, peak VRAM, wall-clock** (from 10.0).
Then **find the 4080's practical cap**: push one axis (params via `--n-embd`, or context via
`--block-size`, or `--batch-size`) until it OOMs or each checkpoint takes too long; on OOM, reduce
`--batch-size` and **say so**. Sample each run's **best** checkpoint at `--temperature 0.8` for a
qualitative read — but state honestly that coherence does not improve with scale here.
- **One line per run** on (a) the loss curve and (b) the sample, same as 09.2.
- **Expected:** val floor ~1.58–1.59 regardless; train→very low on large; the lesson is scale =
  memorization in the data-bound regime, and the cap is time/context, not param VRAM.

### 10.2 — *(Optional, decision to flag first)* Tokenization contrast: char vs BPE/word

The genuinely interesting ablation, and the biggest build. **Decision to make + state in one line
before building:** (a) hand-write a minimal byte-level **BPE** tokenizer (in the project's
"readable, from-scratch" spirit — learn merges from the corpus, `encode`/`decode`, new module +
tests; **no `tokenizers`/`tiktoken` library** — that violates the ethos), (b) trivial **word-level**
(huge sparse vocab, poor on this slang corpus — likely a teaching counter-example), or (c) **skip**
and document why. Lean (a) if time allows, else (c). If built: retrain a model of comparable size,
and contrast **sequence length, loss-per-character (compare in bits/char to be fair across vocabs),
context-in-characters per `block_size`, and sample coherence.** This is where "more context per
token" can buy apparent coherence that scaling params alone could not.

### 10.3 — README "Scale & ablate" section (Definition of Done)

Update `README.md`: an **ablation table** (params / context / tokenizer → best val, final train,
tokens/sec, peak VRAM) with **real numbers from the runs above**; the **4080 cap** finding; a short
**"what I learned about scaling"** — that at fixed (small) data, scale lowers train but not val
(memorization), so the lever for generalization is *more data*, not more parameters; char-vs-BPE
contrast if 10.2 was done. Mark TICKET-10 in the status table. One honest-scope line; no RL. Mark
this ticket done only when 01–10 are all complete (SPEC §6).

## 4. Done when

All touched code green (pytest/ruff/mypy via venv); README has the ablation table + cap finding from
**real runs**; a short summary with the configs, the loss/throughput/VRAM numbers, and the honest
takeaway (data-bound: scale buys memorization, not generalization). **STOP for review** after each
sub-step.
