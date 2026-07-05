"""Tests for the fp16 checkpoint export (backlog Unit 2).

Fast and CPU-only. We build a tiny untrained GPT, save it through the real
`train.save_checkpoint`, run `export_fp16`, and assert the three things the export promises:
the output is smaller on disk, its weights are fp16, and it round-trips back through the same
`load_checkpoint` + `sample_from_checkpoint` path a real checkpoint uses (the fp16 state dict
upcasts cleanly into the fp32 sampling model). The model is untrained, so the text is noise;
these tests check the export harness, not output quality (expected scope per SPEC.md).
"""

from pathlib import Path

import torch

from yegpt.config import TrainConfig
from yegpt.export import export_fp16
from yegpt.model import GPT
from yegpt.sample import sample_from_checkpoint
from yegpt.tokenizer import CharTokenizer
from yegpt.train import load_checkpoint, save_checkpoint

# Same tiny corpus/config family as test_sample: a newline in the vocab, CPU, dropout off.
_CORPUS = "yeezy taught me\n" * 8
_TOKENIZER = CharTokenizer.from_text(_CORPUS)
_CFG = TrainConfig(
    n_layer=2,
    n_head=2,
    n_embd=16,
    block_size=16,
    dropout=0.0,
    batch_size=8,
    device="cpu",
    seed=0,
).with_vocab_size(_TOKENIZER.vocab_size)

_CPU = torch.device("cpu")


def _save_fp32_checkpoint(path: Path) -> None:
    """Build a tiny untrained (fp32) GPT and persist it through the real checkpoint format."""
    torch.manual_seed(0)  # deterministic init so a given run is repeatable
    model = GPT(_CFG)
    save_checkpoint(path, model=model, cfg=_CFG, tokenizer=_TOKENIZER, step=0, val_loss=0.0)


def test_export_shrinks_file_and_reports_sizes(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    dest = tmp_path / "dist" / "yegpt-small-fp16.pt"  # nested dir must be created by the export
    _save_fp32_checkpoint(source)

    result = export_fp16(source, dest)

    assert dest.exists()
    assert result.dest_bytes < result.source_bytes
    assert result.source_bytes == source.stat().st_size
    assert result.dest_bytes == dest.stat().st_size


def test_exported_weights_are_fp16(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    dest = tmp_path / "yegpt-small-fp16.pt"
    _save_fp32_checkpoint(source)

    export_fp16(source, dest)

    exported = load_checkpoint(dest)
    assert exported.model_state  # non-empty, so the all() below is meaningful
    assert all(tensor.dtype == torch.float16 for tensor in exported.model_state.values())


def test_exported_checkpoint_loads_and_samples(tmp_path: Path) -> None:
    source = tmp_path / "yegpt-ckpt.pt"
    dest = tmp_path / "yegpt-small-fp16.pt"
    _save_fp32_checkpoint(source)

    export_fp16(source, dest)

    # The fp16 state dict copies into the fp32 sampling model; the harness must produce a
    # prompt-prefixed string of the exact expected length with every char in vocab.
    exported = load_checkpoint(dest)
    prompt = "ye"
    out = sample_from_checkpoint(exported, prompt=prompt, max_new_tokens=32, device=_CPU)
    assert isinstance(out, str)
    assert len(out) == len(prompt) + 32
    assert all(ch in set(exported.vocab) for ch in out)
