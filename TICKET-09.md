# TICKET-09 - First real training run + iterate (next-session handoff)

> Self-contained handoff. Read `SPEC.md` first (source of truth), then this. Working style is
> unchanged: **one sub-step at a time → implement → write its test → all three checks green via
> the venv → STOP for review.** Do NOT start TICKET-10 (scale / longer context / BPE contrast).

## 0. Decision on the record (read before anything)

- **The author accepted training on the under-gate corpus (2026-06-26). No more data is coming.**
  SPEC §4's ~1 MB gate is **consciously overridden for this run** - do **not** re-block on it or
  re-ask for data.
- **Honest expectation reset:** at **~0.63 MB unique** (0.75 MB raw, ~0.12 MB of it cross-file
  duplicate lyrics), vocab ~97, the model will **lean toward memorization** - more verbatim
  regurgitation of training lines than generalization, i.e. recognizable-Kanye gibberish that
  *parrots* as much as it *invents*. That is the accepted result and a teaching point (watch it
  in the val curve), **not** coherent bars. No RL anywhere (SPEC §7).

## 1. Current state (already built)

- `data/raw/`: `kanye_verses.txt` (0.26 MB lyrics), `kanye_lyrics.txt` (0.33 MB lyrics, **~36 %
  song overlap with verses**), `kanye_tweets.txt` (0.16 MB, produced by `tweet_prep`).
- `python -m yegpt.data_prep` → `data/corpus.txt` = **0.75 MB** (concatenates all three, **no
  dedup yet**), vocab ~97.
- `src/yegpt/tweet_prep.py` (tweet-CSV adapter) is done + tested. Tickets 01–08 done, all green.
- Checks (all must stay green): `.venv\Scripts\python.exe -m pytest` / `-m ruff check .` / `-m mypy`.

## 2. Hard constraints (unchanged)

No `nn.Transformer` / `nn.MultiheadAttention` / HuggingFace model classes. No RL/GRPO/DPO/PPO.
Strict PEP 484, **no `Any`** (`object` ok). Small modules, DI over globals, frozen+slots
dataclasses, code-first with intent-comments. ruff selects E,F,I,UP,B,N - **N806 bans UPPERCASE
locals**. mypy strict + `disallow_any_explicit`/`disallow_any_generics`/`warn_return_any`/
`warn_unreachable`. Minutes-per-checkpoint on the 4080 (SPEC §0).

## 3. Work order

Each sub-step is its own implement→test→green→**STOP for review** cycle.

### 09.0 - Finalize the corpus: cross-file dedup *(do this first - it gates val integrity)*

**Why first:** `kanye_verses` and `kanye_lyrics` share ~36 % of songs. `data_prep` concatenates
without dedup, so duplicated verses both inflate the corpus **and leak across the 90/10 train/val
split** → val loss looks artificially good and **hides the very memorization we want to observe**.
Dedup makes the val signal honest; it's the one quality lever left on this corpus.

**Design (decided - don't re-litigate):** new module `src/yegpt/dedup.py`, a pipeline step on the
normalized corpus. Split into blank-line-separated **blocks** (stanzas; `data_prep` already
collapses to single blank-line separators). Drop **exact-duplicate blocks at or above a length
threshold** (start ~200 chars / ≥6 lines), first occurrence wins; **leave shorter blocks
untouched** so intra-song chorus/ad-lib repetition - which is *style signal* - survives. Return a
report (blocks + chars removed). Expose `dedup_text(text, *, min_block_chars=200) -> (str, report)`
pure core + a thin file/CLI wrapper, mirroring `data_prep`'s shape.
- **Decision to make + flag:** whether dedup runs as a `data_prep --dedup` flag or a standalone
  module the corpus flows through before `dataset`. Lean **standalone module + documented two-step**
  (`data_prep` → `dedup`), but state the call in one line.
- **Tests:** a long duplicated block is removed exactly once; a short repeated block (chorus)
  survives; idempotent; report counts correct.
- **Expected:** ~0.12 MB removed → **~0.63 MB** deduped corpus. Re-derive vocab after (a rare char
  could vanish with a dropped block).

### 09.1 - Train CLI *(TICKET-07 surface change - flag it)*

`train.main()` has no arg parsing; sweeping configs by editing `TrainConfig` defaults is banned.
Add an argparse override to `train.main()`: `--n-layer --n-head --n-embd --block-size --dropout
--lr --max-iters --eval-interval --batch-size --device --seed --out-dir` (checkpoint dir) and
`--corpus`. Build a `TrainConfig` from args (defaults = the current dataclass defaults), then call
`train(...)`. **Do not mutate `TrainConfig` defaults.** Read each `argparse` attr into a typed local
(no `Any` leak - see `sample.py:main`). Test: argv → expected `TrainConfig` fields + checkpoint dir.
`--out-dir` is how you keep multiple checkpoints (before/after, per-config).

### 09.2 - Config + first run

**Recommended start** (sized down for the small/duplicated corpus; dropout is the main regularizer;
`block_size=256` to span a bar):

| n_layer | n_head | n_embd | block_size | dropout | batch | lr | max_iters | eval_interval |
|--------|--------|--------|------------|---------|-------|------|-----------|---------------|
| 4 | 4 | 256 | 256 | 0.2 | 64 | 3e-4 | 5000 | 250 |

→ **~3.3 M params** (vocab ~97). bf16 on the 4080 is trivial VRAM (<1 GB incl. AdamW state +
B=64,T=256 activations); minutes per few-thousand iters. `device="cuda"` (resolve_device fails
loudly if CUDA absent). Print `num_parameters()` before the loop.

**Run** (after 09.1): `python -m yegpt.train --n-embd 256 --block-size 256 --dropout 0.2
--max-iters 5000 --out-dir checkpoints/run1`.

**Observe:** train vs val loss. Start loss ≈ `ln(vocab)` ≈ ln(97) ≈ **4.57** (random guessing).
**Val rising while train falls = memorization** - expected on this corpus; call it out, it's the
lesson.

**Iterate ~2–3 runs**, one line each on what it did to (a) the loss curve and (b) the sample:
- pure regurgitation / val diverges hard → shrink (`--n-embd 192` ~1.9 M, or `--dropout 0.3`).
- both losses high + sample is mush (underfit) → grow (`--n-layer 6 --n-embd 256` ~4.8 M).

### 09.3 - Before/after samples *(must be real - never fabricate; SPEC §0/§6)*

Capture a **near-init "before"** (e.g. a `--max-iters 1` run to `--out-dir checkpoints/before`, or
add an early-step checkpoint write) and the **trained "after"**. Sample both with a fixed seed:
`python -m yegpt.sample --checkpoint <path> --prompt "" -n 500 --seed 1337`. Both samples in the
README **must** come from these actual runs.

### 09.4 - README *(Definition of Done, SPEC §6)*

Update `README.md`: exact commands (`data_prep` → `dedup` → `train` → `sample`); before/after
samples **verbatim from real runs**; fix the **stale status table** (05–08 are done) and mark 09;
short **"what I learned"** on attention, positional embeddings, residual/pre-norm, and why the loss
curve behaves as it does (≈`ln(vocab)` start; plateau = data/capacity ceiling; **val divergence =
memorization on this sub-1 MB corpus**). One honest-scope line: styled gibberish that leans on
memorized fragments here, not lyrics. No RL.

## 4. Done when

All touched code green (pytest/ruff/mypy via venv); README before/after from real runs; a short
summary with the chosen config, loss numbers, and tradeoffs. **STOP for review.**
