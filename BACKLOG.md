# yeGPT public-launch backlog

Groomed 2026-07-05 from the launch spec (chat). SPEC.md is the original build spec and
stays as reference; this file is the work-list the loop drives. Gate (all three, venv
interpreter, all green): `.venv/Scripts/python.exe -m ruff check .` ·
`.venv/Scripts/python.exe -m mypy` · `.venv/Scripts/python.exe -m pytest -q`.

Ground rules for every task (from the launch spec):

- Never commit or ship `data/corpus.txt`, `data/raw/*`, or any raw lyric text; they stay
  gitignored. Short generated fragments in `web/samples.json` are model OUTPUT (parody
  generation) and are fine to commit.
- No training reruns. The usable model is `checkpoints/run3/yegpt-ckpt.pt` (1.87M params,
  val ~1.59). If a task seems to need retraining, mark it blocked for the author instead.
- CPU ONLY. The author is using the GPU; the loop must never run anything on CUDA. All
  sampling, exports, and tests run on CPU (the project default). Never pass
  `--device cuda`, never call `torch.cuda` APIs beyond what existing tests already do.
- The loop never pushes, never runs `gh release`, never changes repo visibility.
  `scripts/publish_release.sh` is WRITTEN here, only the author runs it.
- No em dashes in any Markdown doc; use plain hyphens or rewrite the sentence. No absolute
  Windows paths or personal identifiers in any file (GitHub handle is fine).
- House style: strict typing (no `Any`; `object` allowed), ruff N806 bans UPPERCASE locals,
  frozen dataclasses, DI over globals, comments only where intent isn't obvious.
- Existing module boundaries hold: `train.py` owns the checkpoint format; `model.py` owns
  sampling policy; `sample.py`/CLI only thread knobs through.

## Unit 1: CI gate (test suite already exists: 106 tests + ruff + mypy, all green, ~41s)

- [x] Add `.github/workflows/ci.yml` running the gate on push + PR: Python 3.12, install
      CPU-only torch from the `https://download.pytorch.org/whl/cpu` index (never the cu121
      wheel; keeps the job well under 5 min), `pip install -e .[dev]`, then
      `python -m ruff check .`, `python -m mypy`, `python -m pytest -q`. Acceptance: the
      three CI commands mirror the local gate exactly, and the workflow triggers on both
      `push` and `pull_request`. [simple]

## Unit 2: distributable checkpoint + model card

- [x] Add `scripts/export_checkpoint.py`: load a training checkpoint (default
      `checkpoints/run3/yegpt-ckpt.pt`) via `train.load_checkpoint`, cast the model weights
      to fp16, and save `dist/yegpt-small-fp16.pt` in the SAME `Checkpoint` format (reuse
      `train.save_checkpoint`; do not fork the format), printing before/after file sizes.
      Add `dist/` to `.gitignore`. Test with a tiny synthetic checkpoint: the exported file
      loads via `train.load_checkpoint` and samples via `sample.sample_from_checkpoint`
      (fp16 state dict copies into fp32 params), and the exported file is smaller.
      [complex]
- [x] Write `MODEL_CARD.md`: parameter count (1.87M), architecture table (4 layers, 4
      heads, 192 embd, context 256, vocab 104, char-level, hand-written attention),
      training data description (lyrics + tweets sources, 0.67MB after dedup, corpus NOT
      distributed), eval metrics taken from the README ablation table (best val loss
      1.563-1.599 across scales, data-wall finding), recommended sampling knobs
      (temperature 0.9, top-p 0.92, repetition-penalty 1.3), honest limitations
      (memorization-leaning, recognizable-gibberish coherence ceiling from 0.67MB data),
      a clear "AI-generated parody / educational project, not affiliated with or endorsed
      by Kanye West" note, and a weights note (weights released for research/parody use,
      separate from the MIT code license). No em dashes. [simple]
- [x] Add `scripts/publish_release.sh`: `gh release create` (tag `v0.1.0`) uploading
      `dist/yegpt-small-fp16.pt` and `MODEL_CARD.md`, `set -euo pipefail`, exits with a
      clear error if the artifact is missing. The loop must NEVER execute it (author-run
      only; say so in a header comment). Acceptance: `bash -n scripts/publish_release.sh`
      passes. [simple]

## Unit 3: yegpt CLI (streaming demo)

- [x] Add streaming generation to `src/yegpt/model.py`: a `GPT.generate_stream(...)`
      generator yielding one sampled token id per step, with the exact same signature
      knobs and validation as `generate` (temperature, top_k, top_p, repetition_penalty,
      seeded generator); share the per-step sampling logic with `generate` rather than
      duplicating it. Tests: with the same seed, the collected stream equals `generate`
      output token-for-token; invalid knob values raise the same errors. [complex]
- [x] Add `src/yegpt/cli.py` and a `[project.scripts]` entry `yegpt = "yegpt.cli:main"`:
      `yegpt "prompt" --checkpoint --temperature --top-k --top-p --repetition-penalty
      --max-chars --seed`, decoding and printing each character as it is sampled
      (typewriter streaming: per-char print with flush, CPU by default, defaults matching
      the recommended knobs). `--max-chars` maps to `max_new_tokens`. Tests: a seeded
      snapshot test against a tiny synthetic checkpoint capturing exact stdout text, plus
      invalid-argument validation. README rewrite happens later in unit 5, not here.
      [complex]
- [ ] CORRECTIVE (the task above shipped only a subcommand dispatcher; the demo behavior
      is missing): make bare `yegpt "some prompt"` work as the demo. In
      `src/yegpt/cli.py`, when the first positional arg is NOT a known subcommand, treat
      it as a prompt and stream the generation to the terminal character-by-character AS
      TOKENS ARE SAMPLED: use `GPT.generate_stream` and print each decoded char with
      `print(ch, end="", flush=True)`. Honor `--checkpoint` (default
      `checkpoints/run3/yegpt-ckpt.pt`), `--temperature`, `--top-k`, `--top-p`,
      `--repetition-penalty`, `--max-chars` (maps to `max_new_tokens`), `--seed`; defaults
      = the recommended knobs; CPU only. Do NOT route through `sample.main` (it prints
      only at the end); keep existing subcommand dispatch working. Tests: seeded snapshot
      test against a tiny synthetic checkpoint asserting EXACT stdout via capsys; a test
      that subcommand dispatch still routes; invalid knob values exit non-zero. [complex]

## Unit 4: static website embed

- [x] Add `scripts/export_samples.py`: generate N short fragments (default 12 samples,
      ~200 chars each) from `checkpoints/run3/yegpt-ckpt.pt` with the recommended knobs,
      fully seeded and reproducible, with a `--profanity-filter/--no-profanity-filter`
      flag (default ON) that drops fragments containing words from a small built-in
      wordlist; write `web/samples.json` shaped
      `{"generated_with": {model, seed, knobs}, "samples": ["...", ...]}`. Tests with a
      tiny synthetic checkpoint: same seed twice gives identical JSON, the filter drops a
      planted profane fragment, output parses as valid JSON with the documented shape.
      Then run it for real (run3, filter ON) and COMMIT the resulting `web/samples.json`.
      [complex]
- [ ] CORRECTIVE (the task above landed code + tests but never produced the artifact):
      actually run the exporter against the real checkpoint and COMMIT the result. Run
      `scripts/export_samples.py` with `checkpoints/run3/yegpt-ckpt.pt`, profanity filter
      ON, a fixed documented seed, on CPU; commit the generated `web/samples.json` (verify
      it parses as JSON, has the documented `generated_with` + `samples` shape, and
      contains no raw-corpus dumps, only short generated fragments). [simple]
- [ ] Add `web/embed.js` plus a minimal `web/demo.html` for manual checking: embed.js is
      self-contained vanilla JS (no dependencies, no backend, no build step) exposing a
      global `yegptEmbed(containerElement, samplesUrl)` that fetches `samples.json` and
      typewriter-streams randomly chosen fragments into the container in an endless loop
      (type char-by-char, pause, clear, pick the next). Looks live, is 100% static.
      Acceptance: `node --check web/embed.js` passes; demo.html references only local
      files. [complex]

## Unit 5: public-launch polish

- [ ] Add MIT `LICENSE` (copyright 2026 mreinhofferxd-pixel) covering the CODE only;
      confirm `pyproject.toml` license/author fields stay consistent (weights are covered
      by the MODEL_CARD.md note instead). [simple]
- [ ] Rewrite `README.md` in modern public-GitHub style: shields badges (Python 3.12, MIT
      license, CI status for `mreinhofferxd-pixel/yegpt`, ~1.9M params), one-line pitch,
      quick start (clone, venv, `pip install -e .`, download the release checkpoint,
      `yegpt "prompt"`), compact architecture summary table, the ablation findings as a
      compact prominent table (the differentiator: data-bound proof, scale buys
      memorization not generalization), sampling-knobs section, an "Embed" section
      documenting web/embed.js honestly (pregenerated samples replayed, nothing runs
      live), a corpus section (repo ships only the corpus-building scripts, how to source
      your own data, lyrics never committed), and honest limitations. Keep a condensed
      "what I learned" section. No em dashes anywhere in the file. [complex]
- [ ] Scrub em dashes from the remaining tracked Markdown (`SPEC.md`, `TICKET-09.md`,
      `TICKET-10.md`, and any doc added by earlier tasks that slipped): replace with
      hyphens or reword; acceptance: searching tracked `*.md` files for the em dash
      character returns nothing. [simple]
