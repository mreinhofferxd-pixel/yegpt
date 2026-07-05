"""model: the GPT, written by hand (SPEC.md §6, TICKET-06).

This is a decoder-only transformer of the nanoGPT class, built from small `nn.Module`s
so every moving part is readable. PyTorch supplies tensors/autograd/optim and the *leaf*
layers (`Linear`, `Embedding`, `LayerNorm`, `Dropout`, `GELU`); the architecture itself —
attention, the causal mask, multi-head wiring, residual/pre-norm blocks, the LM head, and
autoregressive sampling — is assembled here. Per the hard scope boundary, nothing here uses
`nn.Transformer`, `nn.MultiheadAttention`, or any pretrained/HuggingFace model class.

Data flow (a single forward pass):

    idx (batch, seq) --embed--> x (batch, seq, n_embd)
        + token embedding (what each char *is*)
        + positional embedding (where each char *sits*)
    --> N x Block[ pre-norm -> causal self-attention -> +residual
                   pre-norm -> MLP                    -> +residual ]
    --> final LayerNorm --> lm_head (Linear) --> logits (batch, seq, vocab_size)

Honest scope: this network only learns next-character statistics. At ~1-10M params on a
char corpus the ceiling is *recognizably Kanye-styled gibberish* — word-shaped, cadence-
shaped, not coherent prose. That is the intended outcome, not a limitation to fix here.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, softmax

from yegpt.config import TrainConfig


class Head(nn.Module):
    """One causal self-attention head: each position mixes in information from earlier ones.

    Attention asks, for every position, "which previous positions are relevant to me?" It
    answers with three learned projections of the input: a *query* (what I'm looking for),
    a *key* (what I offer), and a *value* (what I'll actually contribute). The dot product
    of a query against every key scores relevance; those scores (scaled, masked, softmaxed)
    become weights for a weighted sum over values. The causal mask is what makes this a
    *language model*: position t may only look at positions <= t, never the future it must
    predict.
    """

    # Declared so the type checker knows the registered buffer is a Tensor (Module.__getattr__
    # otherwise returns Any). Lower-triangular ones; persistent=False keeps it out of saved
    # state (it is recomputable structure, not learned weights).
    tril: Tensor

    def __init__(self, cfg: TrainConfig, head_size: int) -> None:
        super().__init__()
        # No bias: these are pure linear projections into query/key/value spaces.
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)), persistent=False
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        k: Tensor = self.key(x)  # (batch, seq, head_size)
        q: Tensor = self.query(x)  # (batch, seq, head_size)
        # Relevance scores between every pair of positions, scaled by 1/sqrt(head_size) so
        # the softmax stays in a sane range as head_size grows (large dot products saturate it).
        scores = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (batch, seq, seq)
        # Causal mask: zero out attention to future positions (upper triangle) before softmax.
        scores = scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        weights: Tensor = softmax(scores, dim=-1)  # rows sum to 1; -inf -> 0 weight
        weights = self.dropout(weights)
        v: Tensor = self.value(x)  # (batch, seq, head_size)
        out = weights @ v  # weighted sum of values -> (batch, seq, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated then mixed by an output projection.

    One head learns one notion of "what's relevant"; running `n_head` of them lets the model
    attend to different relationships at once (e.g. the previous character vs. the start of
    the current word). Each head works in a `head_size = n_embd // n_head` subspace, so the
    concatenation is back to `n_embd` wide, and the final linear lets the heads' outputs
    interact instead of staying in separate lanes.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(cfg, cfg.head_size) for _ in range(cfg.n_head))
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = torch.cat([head(x) for head in self.heads], dim=-1)  # (batch, seq, n_embd)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Per-position MLP: linear -> GELU -> linear, with a 4x hidden expansion.

    Attention moves information *between* positions; this then lets each position *think* on
    what it gathered, independently of the others. The 4x widening (the standard transformer
    ratio) gives the nonlinearity room to compute richer features before projecting back down.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


class Block(nn.Module):
    """One transformer block: pre-norm attention and MLP, each wrapped in a residual.

    Two design choices carry most of the weight here:
    - **Residual connections** (`x = x + sublayer(x)`): the block learns a *correction* to x
      rather than replacing it. This gives gradients a short, direct path back through every
      layer, which is what makes deep stacks trainable.
    - **Pre-norm LayerNorm** (normalize *before* each sublayer, not after): it keeps the
      residual stream itself un-normalized and the input to each sublayer well-conditioned,
      which trains more stably than the original post-norm design.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        attn: Tensor = self.sa(self.ln1(x))
        x = x + attn
        ffn: Tensor = self.ffwd(self.ln2(x))
        x = x + ffn
        return x


class GPT(nn.Module):
    """The full model: embeddings -> transformer blocks -> final LayerNorm -> LM head.

    Token embeddings say *what* each character is; positional embeddings say *where* it sits,
    which attention needs because the weighted sum is otherwise order-blind (it would treat a
    line as a bag of characters). Both are learned. The stack of blocks refines a per-position
    representation; the final LayerNorm + linear head turn each position into a distribution
    over the next character.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.vocab_is_set:
            raise ValueError(
                "cfg.vocab_size is unset; build the tokenizer first and pass "
                "cfg.with_vocab_size(tokenizer.vocab_size) before constructing GPT."
            )
        # Annotated so mypy treats cfg as a declared attribute rather than routing it through
        # nn.Module.__setattr__ (which only accepts Tensor | Module).
        self.cfg: TrainConfig = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.n_layer)))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Small-normal init for linears/embeddings (the GPT-2 convention); LayerNorm keeps its
        # default unit-scale/zero-shift. Large initial weights would make early logits/loss blow up.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Total trainable parameter count — handy for sizing runs against the 4080."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Map token ids `(batch, seq)` to logits `(batch, seq, vocab_size)`.

        With `targets` (same shape as `idx`), also return the mean cross-entropy loss for
        next-char prediction; without them, return `(logits, None)` for inference/sampling.
        """
        seq_len = idx.shape[1]
        tok_emb: Tensor = self.token_embedding_table(idx)  # (batch, seq, n_embd)
        positions = torch.arange(seq_len, device=idx.device)  # (seq,)
        pos_emb: Tensor = self.position_embedding_table(positions)  # (seq, n_embd)
        x: Tensor = tok_emb + pos_emb  # broadcast positions across the batch
        x = self.blocks(x)
        x = self.ln_f(x)
        logits: Tensor = self.lm_head(x)  # (batch, seq, vocab_size)

        if targets is None:
            return logits, None
        # cross_entropy wants (n, classes) logits and (n,) targets, so flatten batch & time.
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @staticmethod
    def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
        """Nucleus filter: keep the smallest set of highest-probability tokens whose cumulative
        probability reaches `top_p`, mask the rest to -inf.

        Unlike a fixed `top_k`, the cut adapts to the distribution each step — it keeps few tokens
        when the model is confident and many when it's unsure, which trims the noisy tail without
        arbitrarily capping a genuinely flat distribution. The top token is always kept (so `top_p`
        can never mask everything) by shifting the removal mask right by one before it's applied.
        """
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative > top_p  # drop the tail past the cumulative-prob threshold
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()  # keep the crossing token
        remove_sorted[..., 0] = False  # never drop the most likely token
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_idx, remove_sorted)
        return logits.masked_fill(remove, float("-inf"))

    @staticmethod
    def _validate_sampling(
        temperature: float, top_k: int | None, top_p: float | None, repetition_penalty: float
    ) -> None:
        """Reject sampling knobs outside their valid ranges, before any tokens are produced."""
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}.")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1] or None, got {top_p}.")
        if repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {repetition_penalty}.")

    def _sample_next(
        self,
        idx: Tensor,
        generator: torch.Generator | None,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
    ) -> Tensor:
        """One autoregressive step: from context `idx` `(batch, seq)`, draw the next token.

        Crops the context to the last `block_size` tokens (the positional table only knows that
        many positions), takes the final-position logits, reshapes them by `repetition_penalty`
        -> `temperature` -> `top_k` -> `top_p`, softmaxes to a distribution, and samples one token,
        returned as `(batch, 1)`. Callers must run this under `torch.no_grad()` and have validated
        the knobs via `_validate_sampling`.
        """
        idx_cond = idx[:, -self.cfg.block_size :]  # (batch, <=block_size)
        logits, _ = self(idx_cond)
        last_logits = logits[:, -1, :]  # (batch, vocab_size), raw (pre-temperature)
        if repetition_penalty != 1.0:
            # CTRL-style penalty over the characters already in the context: divide the logit of
            # each seen token by the penalty (multiply if negative), so the sampler stops
            # re-picking it. Duplicate indices scatter the same value, so repeats are idempotent,
            # not compounded.
            seen_logits = torch.gather(last_logits, 1, idx_cond)
            seen_logits = torch.where(
                seen_logits < 0,
                seen_logits * repetition_penalty,
                seen_logits / repetition_penalty,
            )
            last_logits = last_logits.scatter(1, idx_cond, seen_logits)
        # Temperature scales the (penalized) logits: sharpen if <1, flatten if >1.
        last_logits = last_logits / temperature
        if top_k is not None:
            # Mask everything below the k-th largest logit to -inf so the softmax gives it zero
            # weight: the next char is drawn only from the k most likely candidates.
            k = min(top_k, last_logits.size(-1))
            kth = torch.topk(last_logits, k, dim=-1).values[:, -1:]  # (batch, 1)
            last_logits = last_logits.masked_fill(last_logits < kth, float("-inf"))
        if top_p is not None:
            last_logits = self._apply_top_p(last_logits, top_p)
        probs = softmax(last_logits, dim=-1)
        idx_next: Tensor = torch.multinomial(probs, num_samples=1, generator=generator)
        return idx_next  # (batch, 1)

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        generator: torch.Generator | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Autoregressively extend `idx` `(batch, seq)` by `max_new_tokens` sampled characters.

        Each step crops the context to the last `block_size` tokens, reshapes the final-position
        logits by `repetition_penalty` -> `temperature` -> `top_k` -> `top_p`, softmaxes, samples
        one token, and appends it (see `_sample_next`). Returns the full `(batch, seq +
        max_new_tokens)` sequence; use `generate_stream` to consume tokens as they are produced.

        `temperature` (> 0) divides the logits before the softmax: < 1 sharpens the distribution
        (more confident, more repetitive — approaching greedy as it nears 0), > 1 flattens it
        (more random). `top_k`, when set, keeps only the `k` highest-logit characters each step
        and zeroes the rest; `top_p` (in (0, 1]) instead keeps the smallest set of top characters
        whose probabilities sum to `top_p` (nucleus sampling — see `_apply_top_p`); either or both
        may be set, and `None` on both keeps the whole vocab.

        `repetition_penalty` (> 0, 1.0 = off) down-weights characters already present in the
        current context before the softmax: logits are divided by the penalty when positive and
        multiplied when negative, both pushing toward zero probability. Values a little above 1
        (~1.1–1.3) are what break the low-temperature failure mode where the model locks into a
        loop ("love love love"), by making an already-repeated character progressively less likely
        to be picked again. It changes *how a fixed checkpoint is read out*, not the weights.

        Pass a seeded `generator` for reproducible samples. Call `model.eval()` first if dropout is
        enabled, so sampling isn't perturbed by it.
        """
        self._validate_sampling(temperature, top_k, top_p, repetition_penalty)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_next = self._sample_next(
                    idx,
                    generator,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq+1)
        return idx

    def generate_stream(
        self,
        idx: Tensor,
        max_new_tokens: int,
        generator: torch.Generator | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> Iterator[Tensor]:
        """Like `generate`, but yield each freshly sampled token `(batch, 1)` as it is produced.

        Same step and same knobs as `generate` (see its docstring); the only difference is
        delivery. Rather than returning the whole sequence at the end, this yields the newly
        sampled token after each step so a caller can decode and display characters live (e.g. a
        REPL or CLI printing as the model types) instead of waiting for all `max_new_tokens`.
        The running context is tracked internally, so concatenating the yielded tokens onto the
        original `idx` reproduces exactly what `generate` would return for the same seed.

        As a generator, its body — including knob validation — runs only once iteration begins;
        consume it (e.g. `for tok in model.generate_stream(...)`) to drive sampling. Pass a seeded
        `generator` for reproducible samples, and call `model.eval()` first if dropout is enabled.
        """
        self._validate_sampling(temperature, top_k, top_p, repetition_penalty)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_next = self._sample_next(
                    idx,
                    generator,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq+1)
                yield idx_next
