"""Tests for the hand-written GPT (SPEC.md §6, TICKET-06): shapes, loss, generate.

Tiny config so everything runs in milliseconds on CPU. dropout=0.0 makes the forward pass
deterministic, so reproducibility checks don't depend on eval-mode subtleties.
"""

import pytest
import torch

from yegpt.config import TrainConfig
from yegpt.model import GPT, Block, FeedForward, Head, MultiHeadAttention

_VOCAB = 11
_CFG = TrainConfig(
    n_layer=2, n_head=2, n_embd=16, block_size=8, dropout=0.0, batch_size=4, seed=0
).with_vocab_size(_VOCAB)


def _ids(batch: int, seq: int, vocab: int = _VOCAB) -> torch.Tensor:
    return torch.randint(vocab, (batch, seq), dtype=torch.long)


def test_gpt_requires_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        GPT(TrainConfig())  # vocab still the sentinel


def test_forward_logits_shape_and_loss_is_scalar() -> None:
    model = GPT(_CFG)
    idx = _ids(_CFG.batch_size, _CFG.block_size)
    targets = _ids(_CFG.batch_size, _CFG.block_size)
    logits, loss = model(idx, targets)
    assert logits.shape == (_CFG.batch_size, _CFG.block_size, _VOCAB)
    assert loss is not None
    assert loss.ndim == 0  # a single scalar
    # Untrained model over V classes should start near ln(V).
    assert abs(float(loss) - torch.log(torch.tensor(float(_VOCAB)))) < 1.0


def test_forward_without_targets_returns_none_loss() -> None:
    model = GPT(_CFG)
    logits, loss = model(_ids(_CFG.batch_size, _CFG.block_size))
    assert loss is None
    assert logits.shape == (_CFG.batch_size, _CFG.block_size, _VOCAB)


def test_forward_handles_seq_shorter_than_block_size() -> None:
    # Exercises tril[:seq, :seq] cropping and arange(seq) positional lookup for seq < block_size.
    model = GPT(_CFG)
    seq = _CFG.block_size - 3
    logits, _ = model(_ids(_CFG.batch_size, seq))
    assert logits.shape == (_CFG.batch_size, seq, _VOCAB)


def test_loss_backward_populates_gradients() -> None:
    # End-to-end autograd check: a connected graph with no detached tensors or NaNs from masking.
    model = GPT(_CFG)
    _, loss = model(_ids(_CFG.batch_size, _CFG.block_size), _ids(_CFG.batch_size, _CFG.block_size))
    assert loss is not None
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)


def test_generate_extends_sequence() -> None:
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0 and int(out.max()) < _VOCAB


def test_generate_crops_context_longer_than_block_size() -> None:
    # Seeding with more tokens than block_size must not crash: each step crops before forward.
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, _CFG.block_size + 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, _CFG.block_size + 4 + 3)


def test_generate_is_reproducible_with_seeded_generator() -> None:
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(123)
    g2.manual_seed(123)
    out1 = model.generate(idx, max_new_tokens=20, generator=g1)
    out2 = model.generate(idx, max_new_tokens=20, generator=g2)
    assert torch.equal(out1, out2)


def test_generate_top_k_one_is_greedy_and_seed_independent() -> None:
    # top_k=1 keeps only the argmax logit, so the softmax is one-hot and multinomial always picks
    # that token -- the generator can't matter. Two different seeds must agree.
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(1)
    g2.manual_seed(99)
    out1 = model.generate(idx, max_new_tokens=15, generator=g1, top_k=1)
    out2 = model.generate(idx, max_new_tokens=15, generator=g2, top_k=1)
    assert torch.equal(out1, out2)
    assert out1.shape == (1, 16)


def test_generate_with_temperature_and_top_k_stays_in_vocab() -> None:
    model = GPT(_CFG).eval()
    generator = torch.Generator()
    generator.manual_seed(0)
    out = model.generate(
        torch.zeros((1, 1), dtype=torch.long),
        max_new_tokens=20,
        generator=generator,
        temperature=0.7,
        top_k=5,
    )
    assert out.shape == (1, 21)
    assert int(out.min()) >= 0 and int(out.max()) < _VOCAB


def test_generate_rejects_invalid_temperature_and_top_k() -> None:
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(idx, max_new_tokens=4, temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        model.generate(idx, max_new_tokens=4, top_k=0)


def test_generate_rejects_invalid_top_p_and_repetition_penalty() -> None:
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="top_p"):
        model.generate(idx, max_new_tokens=4, top_p=0.0)  # must be > 0
    with pytest.raises(ValueError, match="top_p"):
        model.generate(idx, max_new_tokens=4, top_p=1.5)  # must be <= 1
    with pytest.raises(ValueError, match="repetition_penalty"):
        model.generate(idx, max_new_tokens=4, repetition_penalty=0.0)


def test_generate_top_p_one_is_a_noop() -> None:
    # top_p=1.0 can never mask a token (cumulative prob never exceeds 1), so with a fixed seed it
    # must reproduce plain sampling exactly -- the nucleus filter is off at the top of its range.
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(0)
    g2.manual_seed(0)
    plain = model.generate(idx, max_new_tokens=20, generator=g1)
    nucleus = model.generate(idx, max_new_tokens=20, generator=g2, top_p=1.0)
    assert torch.equal(plain, nucleus)


def test_generate_tiny_top_p_is_greedy_and_seed_independent() -> None:
    # A top_p below 1/vocab keeps only the single most likely token (the argmax prob is always
    # >= 1/vocab), so sampling is effectively greedy and the generator can't matter.
    model = GPT(_CFG).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(1)
    g2.manual_seed(99)
    out1 = model.generate(idx, max_new_tokens=15, generator=g1, top_p=0.01)
    out2 = model.generate(idx, max_new_tokens=15, generator=g2, top_p=0.01)
    assert torch.equal(out1, out2)
    assert int(out1.min()) >= 0 and int(out1.max()) < _VOCAB


def test_generate_repetition_penalty_flips_a_greedy_repeat() -> None:
    # Deterministic behavioral check. Force the logits to a fixed vector via lm_head (zero the
    # weight so the output is exactly the bias regardless of context): token 3 just edges token 5.
    # Greedy (top_k=1) with no penalty picks 3 every step; with a penalty, once 3 has been emitted
    # its logit is divided below 5's, so the *next* step must switch to 5 -- proving the penalty
    # both applies and targets already-seen tokens. An untrained model can't show this (its logits
    # are ~0, where scaling is a near-no-op), so we set the logits explicitly.
    model = GPT(_CFG).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[3] = 2.0
        model.lm_head.bias[5] = 1.9
    idx = torch.zeros((1, 1), dtype=torch.long)  # context token 0 (logit 0)

    plain = model.generate(idx, max_new_tokens=4, top_k=1)
    assert plain[0, 1:].tolist() == [3, 3, 3, 3]  # greedy always takes the top logit

    penalized = model.generate(idx, max_new_tokens=4, top_k=1, repetition_penalty=1.4)
    assert int(penalized[0, 1]) == 3  # first pick still 3 (not yet seen)
    assert int(penalized[0, 2]) == 5  # 3 now penalized below 5 -> switches

    # 1.0 is the documented no-op: it must match unpenalized greedy exactly.
    off = model.generate(idx, max_new_tokens=4, top_k=1, repetition_penalty=1.0)
    assert torch.equal(off, plain)


def test_num_parameters_is_positive() -> None:
    assert GPT(_CFG).num_parameters() > 0


def test_submodule_output_shapes() -> None:
    # The hand-written pieces each preserve the (batch, seq, n_embd) residual-stream shape.
    x = torch.randn(_CFG.batch_size, _CFG.block_size, _CFG.n_embd)
    assert Head(_CFG, _CFG.head_size)(x).shape == (_CFG.batch_size, _CFG.block_size, _CFG.head_size)
    for module in (MultiHeadAttention(_CFG), FeedForward(_CFG), Block(_CFG)):
        assert module(x).shape == x.shape
