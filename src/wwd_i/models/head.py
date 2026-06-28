"""Per-word wake-word head: a streaming GRU over frozen embeddings (Phase 4).

The frozen backbone emits one embedding per 80 ms hop; this tiny head consumes
that stream and emits ``P(wake)``. A 1-layer GRU carries hidden state, so at
inference it runs one embedding at a time (lowest latency, smallest state) and at
training it sees a short fixed window and is supervised by the **max over time**
of its per-hop logit — i.e. "fire at least once during the word, never on a
negative", which is exactly the streaming detection criterion. See §6.

torch is training/export-only (the ``train`` group); the runtime loads the
exported ``<word>_head.onnx`` and never imports this module.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HeadConfig:
    embedding_dim: int = 96
    hidden: int = 48
    dropout: float = 0.2


class WakeHead(nn.Module):
    """Embedding stream ``[B, T, D]`` -> per-hop wake logit ``[B, T]`` (stateful).

    A single-layer GRU written out with primitive ops (two linears + gates) instead
    of ``nn.GRU``, so the exported ONNX matches this module to fp32 — the ONNX
    ``GRU`` op's reset-gate formulation otherwise drifts ~4e-3 from torch, which
    would invalidate the torch-calibrated threshold the runtime relies on. ``h0``
    follows the ``nn.GRU`` convention ``[1, B, H]``.
    """

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        c = self.config
        self.hidden = c.hidden
        self.x2h = nn.Linear(c.embedding_dim, 3 * c.hidden)  # input -> reset/update/new gates
        self.h2h = nn.Linear(c.hidden, 3 * c.hidden)  # hidden -> reset/update/new gates
        self.drop = nn.Dropout(c.dropout)  # on the per-hop output (not the recurrent state)
        self.fc = nn.Linear(c.hidden, 1)

    def _step(self, x: Tensor, h: Tensor) -> Tensor:
        ir, iz, in_ = self.x2h(x).chunk(3, dim=-1)
        hr, hz, hn = self.h2h(h).chunk(3, dim=-1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn)  # reset gate applied to the hidden projection
        return (1.0 - z) * n + z * h

    def forward(self, emb: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h = emb.new_zeros(emb.shape[0], self.hidden) if h0 is None else h0[0]
        logits = []
        for t in range(emb.shape[1]):
            h = self._step(emb[:, t, :], h)
            logits.append(self.fc(self.drop(h)))  # [B, 1]
        return torch.cat(logits, dim=1), h.unsqueeze(0)  # logits [B, T], hn [1, B, H]

    def clip_logits(self, emb: Tensor) -> Tensor:
        """Max-over-time logit per clip ``[B]`` — a primitive (superseded as the
        training target by :meth:`clip_score`; still handy for diagnostics)."""
        return self.forward(emb)[0].amax(dim=1)

    def clip_score(self, emb: Tensor, k: int) -> Tensor:
        """Top-``k``-mean ``P(wake)`` per clip ``[B]`` — the streaming training target.

        The mean of the ``k`` highest per-hop probabilities, so a positive must hold
        a high response over *several* hops rather than a single-hop spike — the
        criterion that suppresses lone false-accept spikes and that the engine
        re-scores at runtime (``runtime.engine._Head._score``, same ``k``).
        Operates in probability space (``mean`` does not commute with ``sigmoid``,
        so training and the engine must aggregate post-sigmoid)."""
        probs = torch.sigmoid(self.forward(emb)[0])  # [B, T]
        k = min(k, probs.shape[1])
        return probs.topk(k, dim=1).values.mean(dim=1)


class _StreamHead(nn.Module):
    """Export wrapper: one (or more) embedding(s) + state in, ``P(wake)`` + state out."""

    def __init__(self, head: WakeHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, embedding: Tensor, h0: Tensor) -> tuple[Tensor, Tensor]:
        logits, hn = self.head(embedding, h0)
        return torch.sigmoid(logits), hn


def export_head_onnx(model: WakeHead, path: str | Path) -> Path:
    """Export the streaming head to ONNX (probability + carried GRU state).

    The graph takes one embedding ``[1, 1, D]`` plus the previous hidden state and
    returns ``P(wake)`` and the next state, so the runtime advances it one 80 ms
    hop at a time. Shapes are fixed (the GRU sequence length doesn't survive as a
    dynamic ONNX axis anyway, and streaming only ever feeds one hop). Weights are
    inlined (``external_data=False``) — see ``backbone.export_onnx``.
    """
    import copy

    model = copy.deepcopy(model).eval().cpu()
    c = model.config
    emb = torch.zeros(1, 1, c.embedding_dim)
    h0 = torch.zeros(1, 1, c.hidden)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _StreamHead(model).eval(),
        (emb, h0),
        str(path),
        dynamo=True,
        input_names=["embedding", "h0"],
        output_names=["prob", "hn"],
        external_data=False,
        verbose=False,
    )
    return path
