<div align="center">

# yeGPT

**A character-level GPT, written by hand from scratch, trained on a Kanye West corpus.**

Lyrics + tweets, trained on a single RTX 4080.

[![CI](https://github.com/mreinhofferxd-pixel/yegpt/actions/workflows/ci.yml/badge.svg)](https://github.com/mreinhofferxd-pixel/yegpt/actions/workflows/ci.yml)
[![Params](https://img.shields.io/badge/params-1.87M-8A2BE2.svg)](MODEL_CARD.md)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Linting: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Types: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![Built from scratch](https://img.shields.io/badge/built-from%20scratch-orange.svg)](#what-i-learned)

[Quickstart](#quickstart) · [Run the pipeline](#run-the-pipeline) · [Results](#results-real-samples) · [What I learned](#what-i-learned) · [SPEC](SPEC.md) · [Model card](MODEL_CARD.md)

</div>

---

## What this is

A decoder-only transformer built line-by-line to understand how one actually works - tokenizer,
attention, transformer blocks, training loop. **No** `nn.Transformer`, **no**
`nn.MultiheadAttention`, **no** HuggingFace model classes, **no** pretrained weights. PyTorch is
used only for tensors/autograd/optim; everything model-shaped is assembled by hand in
[model.py](src/yegpt/model.py).

This is a **learning project**, and its honest result is the deliverable:

> At this scale (~1.9–10.9M params) on a ~0.67MB corpus, the model produces *recognizably
> Kanye-styled gibberish* - word-shaped, line-shaped, clearly "trying," and leaning on
> **memorized fragments** of the training text. That is the success criterion. It does **not**
> write coherent lyrics, and it can't: there isn't enough data. No RL is involved (GRPO/DPO/PPO
> are a deliberately separate future project - see [SPEC.md](SPEC.md) §7).

See [SPEC.md](SPEC.md) for the full design, constraints, and ticket plan.

## Quickstart

Requires Python **3.11 or 3.12** (PyTorch wheels; 3.13/3.14 not yet supported here). Sampling
runs fine on CPU; CUDA is only needed to train.

```sh
python -m venv .venv
# CPU-only torch keeps the install small (skip this line if you already have torch):
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dev]"

# Sanity checks (the quality gate)
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest
```

Grab the trained weights (fp16, ~3.6MB) from the
[latest release](https://github.com/mreinhofferxd-pixel/yegpt/releases) and point the CLI at
them:

```sh
gh release download v0.1.0 --pattern "yegpt-small-fp16.pt"
yegpt "I'm the greatest" --checkpoint yegpt-small-fp16.pt
```

Installing the package wires a single `yegpt` console entry point ([cli.py](src/yegpt/cli.py)).
It has two modes:

```sh
# Subcommand mode - routes to a pipeline module's own CLI:
yegpt data-prep | tweet-prep | dedup | train | sample | export | export-samples
yegpt sample --help        # shows sample.py's own flags

# Prompt mode - anything that is NOT a subcommand is treated as a seed and
# typewriter-streamed to stdout, one character per sampled token (CPU only):
yegpt "I'm the greatest" --max-chars 200 --seed 1337
```

## Demo

`web/` holds a zero-dependency static embed that **replays pregenerated parody fragments** with a
typewriter effect - nothing runs live in the browser. The fragments in
[web/samples.json](web/samples.json) are model *output* (safe to commit; the raw corpus is not),
emitted by `yegpt export-samples` from the released run3 checkpoint with the model-card knobs.

```sh
# Regenerate the showcase JSON from a checkpoint (profanity-screened by default):
yegpt export-samples --checkpoint checkpoints/run3/yegpt-ckpt.pt

# View it: serve web/ and open demo.html
python -m http.server -d web 8000   # then browse http://localhost:8000/demo.html
```

## Run the pipeline

The full pipeline is four steps: build the corpus, dedup it, train, sample.

```sh
# 1. raw/*.txt -> data/corpus.txt  (normalize + concatenate; prints char/MB + sub-1MB warning)
.venv/Scripts/python -m yegpt.data_prep

# 2. dedup the corpus IN PLACE  (drops cross-file duplicate verses so val loss is honest)
.venv/Scripts/python -m yegpt.dedup
```

> **Order matters.** `data_prep` rebuilds `corpus.txt` from `data/raw/` **without** dedup, so
> always run `dedup` after it. Training reads `data/corpus.txt`, so the deduped file is what a
> bare `train` run picks up. Re-running `data_prep` alone silently reverts to the un-deduped,
> duplicate-inflated corpus.

```sh
# 3. train  (writes a checkpoint to --out-dir; sweep hyperparameters via flags, not by editing
#    TrainConfig). This is the settled config from the first training sweep (~1.9M params).
.venv/Scripts/python -m yegpt.train \
    --n-embd 192 --block-size 256 --dropout 0.2 \
    --max-iters 5000 --eval-interval 250 --out-dir checkpoints/run3

# 4. sample  (reconstructs the model and generates). --temperature/--top-k/--top-p and
#    --repetition-penalty shape the sampling WITHOUT retraining; the combo below reads best
#    (--repetition-penalty breaks the "love love love" loops, --top-p trims the noisy tail).
.venv/Scripts/python -m yegpt.sample \
    --checkpoint checkpoints/run3/yegpt-ckpt.pt --prompt "" -n 500 --seed 1337 \
    --temperature 0.9 --top-p 0.92 --repetition-penalty 1.3
```

`python -m yegpt.train --help` lists every tunable flag (`--n-layer --n-head --n-embd
--block-size --dropout --lr --max-iters --eval-interval --batch-size --device --seed
--out-dir --corpus`). Each `--out-dir` keeps a run's checkpoints separate, which is how the
before/after below were produced.

Each run writes **two** checkpoints to its `--out-dir`: `yegpt-ckpt.pt` (the final-step weights)
and `yegpt-best.pt` (the lowest-val snapshot). Point `--checkpoint` at either. On an overfitting
run they differ - and the best is the one you actually want (see [What I learned](#what-i-learned)).

> **Windows console note.** A *near-init* model samples roughly uniformly over the whole vocab,
> including characters the default Windows console code page (cp1252) can't print, so piping
> its output may raise `UnicodeEncodeError`. Set `PYTHONIOENCODING=utf-8` (or redirect to a
> UTF-8 file) when sampling untrained checkpoints. Trained checkpoints emit common characters
> and print fine.

## Results (real samples)

Both are verbatim from actual checkpoints, same prompt (empty) and seed (`1337`), `n=500`.
"Before" is the **same 192-wide architecture at step 0** (random init, before any optimizer
step); "after" is the same model after 5000 steps.

<table>
<tr><th>Before - random init (step 0)</th><th>After - 5000 steps (run3)</th></tr>
<tr><td valign="top">

train/val ≈ 4.63/4.62 - character noise, no words, no word boundaries, rare Unicode. The loss
sits right at ≈ `ln(104)` = 4.64, a *roughly uniform* draw over the 104-char vocab: the model
knows nothing yet.

```
5í3ZSJ2B~öOX_oq#QCA.c+tftlāKntnÁŐ*EÁ+6 f2:,QÉ;mv
è"2RōèVEHDN·[GL.xyS9·zSèIQnó:DtLG4<ej<$8B_W0;?ÉqlsDvIL8':
Q1HI!Á(I-LW
Cwz/éVñL+jō,ā:Qjúw!LöqIró?M<ŐfPñ%'|r:2D
Bn+ÉC)c )fY⁠Nat:ā.N3GQ]R&oq Q&ŐaZó-i
```

</td><td valign="top">

train/val ≈ 1.14/1.59 - word-shaped, line-shaped, recognizably the voice (cadence, slang,
`Roc-A-Fella`, `Rollie`). Still gibberish - the *grammar of Kanye* without the meaning.

```
we gonna did in here we to
You won't do it, it slowly drive slow
That's a partned and of you, yo
You can't take that you to see your safe is more
What's nobody, the couple attin' why you see me
You bring me now, who's drike?
Ridiculous, and hold over hold your apprice?
Rollie's a grab my jor, dad, that's a Thurs arm

And I know I know the Roc-A-Fella, King's in the Hotny
```

</td></tr>
</table>

### What the training runs showed

Three runs on the deduped 0.67MB corpus (vocab 104), each to its own `--out-dir`, 5000 steps,
batch 64, lr 3e-4, block_size 256, on the 4080 (bf16 autocast, minutes per run):

| run | n_embd | dropout | params | final train | final val | best val | curve |
|-----|--------|---------|--------|-------------|-----------|----------|-------|
| run1 | 256 | 0.2 | 3.28M | 0.89 | 1.68 | 1.59 (step 2500) | **memorizes** - val turns back up |
| run2 | 256 | 0.3 | 3.28M | 1.15 | 1.60 | 1.59 (step 3750) | flat - divergence regularized away |
| run3 | 192 | 0.2 | 1.87M | 1.14 | 1.59 | 1.58 (step 4250) | flat - **best generalizer, least compute** |

**run3 is the settled config.** The bigger model (run1) doesn't generalize better - it just
memorizes harder. Both more dropout and less capacity remove the divergence; the smaller model
reaches the lowest val at the lowest cost.

### Scale & context ablation - does a bigger model help?

Short answer: **no, not on this data.** The first runs suspected the ~1.58 val floor was set by
the 0.67MB corpus, not the model size. The ablation tests that head-on: hold the corpus and the training
budget fixed (`--max-iters 5000 --eval-interval 250 --dropout 0.2 --lr 3e-4 --batch-size 64`) and
sweep model size and context length, each to its own `--out-dir`. The loop was first instrumented
to print **throughput and peak VRAM**, so the same four runs also answer "where does
the 4080 cap out?"

```sh
# baseline (= run3, 1.9M)
.venv/Scripts/python -m yegpt.train --n-layer 4 --n-head 4 --n-embd 192 --block-size 256 \
    --dropout 0.2 --lr 3e-4 --max-iters 5000 --eval-interval 250 --out-dir checkpoints/abl-baseline
# mid (~5M)      : --n-layer 4 --n-head 8 --n-embd 320 --block-size 256   (same shared flags)
# large (~11M)   : --n-layer 6 --n-head 6 --n-embd 384 --block-size 256
# long-ctx (~11M): --n-layer 6 --n-head 6 --n-embd 384 --block-size 512   (2x context)
```

| run | layers/heads/embd | context | params | best val (@ step) | final train | final val | tokens/s | peak VRAM | wall-clock |
|-----|-------------------|---------|--------|-------------------|-------------|-----------|----------|-----------|------------|
| baseline | 4 / 4 / 192 | 256 | 1.87M | **1.577** (4250) | 1.14 | 1.59 | 553k | 1.28 GiB | 2.5 min |
| mid | 4 / 8 / 320 | 256 | 5.08M | 1.599 (2250) | 0.60 | 1.84 | 288k | 2.76 GiB | 4.7 min |
| large | 6 / 6 / 384 | 256 | 10.82M | 1.582 (1750) | 0.23 | 2.32 | 218k | 3.84 GiB | 6.3 min |
| long-ctx | 6 / 6 / 384 | 512 | 10.92M | **1.563** (1500) | 0.08 | 2.86 | 133k | 9.45 GiB | 20.5 min |

Three things to read off it:

1. **Best val is flat - the data wall is real.** Across a 5.8× parameter range (1.9M → 10.9M) and
   a 2× context increase, best validation loss moves only between 1.563 and 1.599 - that's noise.
   Adding capacity does not lower the floor. The floor is set by the 0.67MB of text, full stop.
2. **Scale buys memorization, not generalization.** Final *train* loss collapses monotonically
   with size - 1.14 → 0.60 → 0.23 → **0.08** - while final *val* gets steadily *worse* - 1.59 →
   1.84 → 2.32 → **2.86**. The big models don't learn the language better; they memorize the
   training split harder, and start overfitting *earlier* (best-val at step 4250 for baseline vs
   step **1500** for long-ctx). This is exactly why the loop keeps a **best-val** checkpoint.
3. **On the 4080, the cap is time and context - not VRAM.** Even the largest run peaks at **9.45
   GiB of 16** - parameters are cheap for a char model. What caps out is **wall-clock per
   checkpoint**: doubling context 256 → 512 is O(T²) in attention and drops throughput from 553k
   to 133k tokens/s, turning a 2.5-minute run into a 20.5-minute one. The practical ceiling on
   this box is **~512 context at batch 64**, and it's a *time* ceiling well before it's a memory one.

Sampling each run's **best-val** checkpoint (`--temperature 0.8 --seed 1337`) confirms the
headline: **coherence does not improve with scale.** All four sit at ~1.57–1.60 val and read as
the same recognizable gibberish - bigger just memorizes more of the corpus verbatim.

**Tokenization (10.2) - skipped, on purpose.** The one lever that could actually change the
coherence-per-parameter tradeoff is tokenization: a BPE/word tokenizer packs more characters into
each token, so a fixed context spans more text and every token carries more meaning. That's the
genuinely interesting ablation - but it's a substantial from-scratch build (learn merges from the
corpus, a new hand-written `encode`/`decode`, its own tests; no `tokenizers`/`tiktoken`, that
would break the from-scratch rule), and it would not change the core finding that *this* corpus is
data-bound. Left as future work.

## What I learned

The deliverable of this project is understanding *why* the pieces are shaped the way they are.

**Character-level tokenization.** The vocabulary is the 104 distinct characters in the corpus;
`encode`/`decode` are dict lookups (`stoi`/`itos`). Char-level sidesteps all vocabulary
engineering, which suits a corpus of slang, ad-libs, and chaotic punctuation - at the cost of
spending model capacity learning spelling and spacing that a word/BPE tokenizer would get for
free.

**Attention is the only place positions exchange information.** For each position the model
computes three projections of its embedding: a **query** (what I'm looking for), a **key**
(what I offer), and a **value** (what I'll contribute). Scoring every query against every key
(scaled by `1/sqrt(head_size)` so the dot products don't grow large enough to push the softmax
into a near-one-hot, low-gradient regime) gives a relevance matrix; a softmax turns each row
into weights; the output is the weighted sum of values. A **causal
mask** (`tril`, future positions set to `-inf` before the softmax) is what makes it a language
model: position *t* may attend only to positions ≤ *t*, never to the future it's being trained
to predict. **Multi-head** attention runs `n_head` of these in parallel in `n_embd // n_head`
subspaces, concatenates them, and mixes them with an output projection - several notions of
"what's relevant" at once. None of this uses `nn.MultiheadAttention`; it's assembled by hand
in [model.py](src/yegpt/model.py).

**Positional embeddings buy order.** The core attention operation - the softmax-weighted sum
of values - is *permutation-equivariant*: with no mask and no position signal, permuting the
inputs just permutes the outputs, so a line would be treated as a *bag* of characters. The
causal mask already injects *some* order on its own (it distinguishes earlier positions from
later ones, so a causal model is not permutation-equivariant), but a learned positional
embedding table (`block_size × n_embd`), added to the token embeddings, gives the model a
direct, *absolute* notion of where each character sits -
which is what lets it learn that newlines start lines and that letters cluster into words. The
table also caps the context: it has exactly `block_size` rows, so longer contexts must be
cropped (that's why `generate()` only ever feeds the last `block_size` tokens).

**Residuals + pre-norm make depth trainable.** Each sublayer is wrapped as
`x = x + sublayer(LayerNorm(x))`. The **residual** means a block learns a *correction* to the
stream rather than replacing it, giving gradients a short, direct path back through every layer -
that's what keeps a deep stack trainable. **Pre-norm** (normalize the *input* to each
sublayer, leaving the residual stream itself un-normalized) trains more stably than the
original post-norm design.

**Why the loss curve behaves as it does.**
- **It starts at ≈ `ln(vocab)` = ln(104) ≈ 4.64.** An untrained model is roughly uniform over
  the 104 characters, and the cross-entropy (in nats) of a uniform predictor is exactly
  `ln(vocab)`. Observed step-0 loss was 4.62–4.69: values land just above `ln(104)` from random
  init, and the small dip below it on one run is finite-batch sampling noise, not a predictor
  better than uniform - either way, the model knowing nothing, by the numbers.
- **It falls fast, then plateaus near ~1.58–1.59.** Early steps learn cheap statistics (letter
  frequencies, spaces, common short words); the plateau is the **data/capacity ceiling** for
  this corpus. Crucially, *adding parameters did not lower it* - run1 (3.3M) and run3 (1.9M)
  bottom out at essentially the same val floor (if anything the smaller model edges it out).
  The wall is the 0.67MB of text, not the model size.
- **Validation divergence is memorization, and it's the whole point.** In run1, val loss
  bottoms (1.59 at step 2500) and then climbs back to 1.68 while train keeps falling to 0.89.
  The model is fitting patterns specific to the training split that don't transfer - i.e.
  memorizing. A sub-1MB char corpus all but guarantees this, which is exactly why the run was
  done at this size: to *watch* it happen in the val curve. It also has a practical consequence,
  which is why the loop saves **both** a final and a best-val checkpoint: for run1 the best
  weights (step 2500, val 1.59) generalize meaningfully better than the final ones (step 5000,
  val 1.68) you'd otherwise be left holding.
- **Dedup is what made that signal trustworthy.** `kanye_verses.txt` and `kanye_lyrics.txt`
  share ~36% of their songs, and the train/val split is a single contiguous cut at 90%. A
  duplicated verse hurts two ways: copies that straddle the cut leak train→val (val then
  measures memorized recall and looks artificially good), and copies that don't still inflate
  the corpus and let the model memorize repeated verses inside the training split. Removing the
  cross-file duplicates (the dedup pass) clears both confounds, so the train-vs-val gap reflects
  real generalization - the divergence above is honest, not bookkeeping.

**Temperature and top-k shape sampling, not the model.** Generation draws each character from
the softmax over the next-char logits. `--temperature` divides those logits first: below 1 it
sharpens the distribution (the model commits to its favourite continuations), above 1 it flattens
it (more variety, more noise); `--top-k` restricts each draw to the K most likely characters.
Both change *how you read out* a fixed checkpoint - no retraining. The failure mode is the
instructive part: at `--temperature 0.5 --top-k 10`, run3 collapses into repetition - *"the lights
of the lights of the lights …"* - because once it's confident, low temperature keeps re-picking the
same high-probability loop. **`--repetition-penalty` (> 1, e.g. 1.3) is the direct fix**: it
down-weights characters already in the context so the sampler can't lock into "love love love",
which visibly cleans up the low-temperature output without any retraining. **`--top-p` (nucleus
sampling)** is an adaptive alternative to `--top-k` - it keeps the smallest set of top characters
whose probabilities sum to *p*, so the cutoff tightens when the model is confident and loosens when
it isn't. Around temperature 0.7–0.9 with a mild repetition penalty reads best; temperatures above 1
push back toward noise. These change *how a fixed checkpoint is read out*, not the weights - and
notably they raise readability, not coherence: the output is less broken but no more meaningful,
because meaning is capped by the data. (All implemented in `model.generate`, surfaced as flags by
`sample.py` - the model owns the sampling policy.)

**Honest scope.** What comes out is styled gibberish that leans on memorized fragments, not
lyrics - and that is the intended end state, not a bug to fix. Coherent generation would need
much more data and scale (and, as a *separate* project, RL on top). None of that is in scope
here.

## Layout

```
src/yegpt/   # the model and pipeline, written by hand
tests/       # pytest suite
web/         # static embed that replays pregenerated parody samples
data/raw/    # author drops source .txt here (gitignored)
checkpoints/ # saved model weights (gitignored)
```

## License

Source code is [MIT](LICENSE). The license covers the code only - **not** the training corpus,
raw source text, or model weights, which are not distributed under it (see
[MODEL_CARD.md](MODEL_CARD.md)).
