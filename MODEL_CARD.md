# Model Card: yeGPT (small)

yeGPT is a character-level, decoder-only GPT built as a learning project, with the model
implemented directly on PyTorch primitives (no transformer library modules). It is trained
from scratch (random init, no pretrained weights) on a Kanye West corpus (lyrics and tweets)
and produces recognizably Kanye-styled parody text. This card describes the released
`yegpt-small` checkpoint (the settled `run3` configuration, ~1.87M parameters).

## Summary

- **Parameters:** 1.87M (all trainable)
- **Type:** decoder-only transformer, character-level, custom attention implementation
- **Task:** next-character prediction / autoregressive text generation
- **Corpus:** ~0.67MB of Kanye West text after dedup (NOT distributed)
- **Intended use:** educational demo and AI-generated parody, CPU-friendly
- **Not intended for:** coherent lyric generation or any factual/production use

## Architecture

Built from small `nn.Module` pieces so every moving part is readable: no `nn.Transformer`,
no `nn.MultiheadAttention`, no pretrained or HuggingFace weights. PyTorch supplies only
tensors, autograd, optim, and the leaf layers (`Linear`, `Embedding`, `LayerNorm`,
`Dropout`, `GELU`).

| Property | Value |
|----------|-------|
| Transformer blocks (n_layer) | 4 |
| Attention heads (n_head) | 4 |
| Embedding width (n_embd) | 192 |
| Per-head size (n_embd / n_head) | 48 |
| Context length (block_size) | 256 characters |
| Vocabulary | 104 characters |
| Tokenizer | character-level (dict lookup `stoi`/`itos`) |
| Attention | custom multi-head causal self-attention |
| MLP | 4x hidden expansion, GELU |
| Blocks | pre-norm LayerNorm + residual connections |
| Positions | learned absolute position embedding table |
| Total parameters | 1.87M |

## Training data

- **Sources:** Kanye West lyrics, interview and rant transcripts, and tweets, normalized
  and concatenated into a single character corpus.
- **Size:** ~0.67MB after cross-file dedup (duplicate verses shared across the lyric and
  verse source files are removed so the train/val split reflects real generalization).
- **Vocabulary:** the 104 distinct characters present in the corpus.
- **Distribution:** the corpus and all raw source text are gitignored and are NOT shipped
  with the model or the code. Only short generated fragments (model output) are ever
  published.

## Training configuration (run3)

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Context length | 256 |
| Dropout | 0.2 |
| Max iters | 5000 |
| Eval interval | 250 |
| Precision | bf16 autocast (training), fp16 (released weights) |
| Hardware | single RTX 4080 |
| Seed | 1337 |
| Wall-clock | ~2.5 minutes |

The released checkpoint keeps both a final-step and a best-validation snapshot; the
best-val snapshot (step ~4250, val ~1.58) is the one to prefer.

## Evaluation

Loss is cross-entropy in nats over next-character prediction. An untrained model sits near
`ln(104)` = 4.64 (roughly uniform over the 104-char vocab). The scale-and-context ablation
holds the corpus and training budget fixed and sweeps model size:

| Run | layers / heads / embd | context | params | best val loss |
|-----|-----------------------|---------|--------|---------------|
| baseline (this card) | 4 / 4 / 192 | 256 | 1.87M | 1.577 |
| mid | 4 / 8 / 320 | 256 | 5.08M | 1.599 |
| large | 6 / 6 / 384 | 256 | 10.82M | 1.582 |
| long-ctx | 6 / 6 / 384 | 512 | 10.92M | 1.563 |

**The data wall is the headline finding.** Across a 5.8x parameter range (1.87M to 10.9M)
and a 2x context increase, best validation loss moves only between 1.563 and 1.599, which
is noise. Adding capacity does not lower the floor; the floor is set by the ~0.67MB of
text. Scale buys memorization, not generalization: larger models drive training loss down
(to ~0.08) while validation loss gets worse (to ~2.86). The ~1.9M-parameter baseline
reaches the lowest val at the lowest cost, which is why it is the released model.

## Recommended sampling knobs

Sampling policy lives in `GPT.generate` and is surfaced by `sample.py`. These change how a
fixed checkpoint is read out, not the weights. The combination below reads best (breaks the
low-temperature repetition loops and trims the noisy tail):

| Knob | Value |
|------|-------|
| temperature | 0.9 |
| top-p | 0.92 |
| repetition-penalty | 1.3 |
| seed | any fixed value (e.g. 1337) for reproducible output |

Temperatures around 0.7 to 0.9 with a mild repetition penalty read best; temperatures above
1 push back toward noise. These raise readability, not coherence.

## Limitations

- **Output is styled parody gibberish, not lyrics.** It is word-shaped, line-shaped, and
  recognizably the cadence, but it does not write coherent verses and cannot at this scale.
- **Memorization-leaning.** On a sub-1MB char corpus the model leans on memorized fragments
  of the training text; the coherence ceiling is set by the ~0.67MB of data, not the model
  size, so a bigger model would not read more coherently.
- **No factual grounding and no instruction following.** It is a next-character predictor
  with no alignment, RL, or safety tuning. It may emit profanity or offensive strings
  present in or recombined from the source style.
- **Narrow domain.** Trained only on one artist's style; it has no general language ability.

## Parody and affiliation notice

This is an AI-generated parody and educational project. It is NOT affiliated with,
authorized by, or endorsed by Kanye West (Ye) or any related entity. All generated text is
synthetic parody output produced by a small character-level model and should not be
attributed to any real person.

## Weights and license

The model weights are released for research and parody use. This weights grant is separate
from the code license: the yeGPT source code is covered by the MIT license, while the
released weights are provided as-is for non-commercial research and parody purposes. The
training corpus and raw source text are not distributed under any license and are never
shipped.
