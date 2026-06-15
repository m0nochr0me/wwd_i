"""Cosine-prototypical metric loss for backbone pretraining (Phase 2, Path A).

An episode is N classes x (K support + Q query) embeddings, laid out class-major:
the first ``N*K`` rows are support (K per class, class order), the next ``N*Q`` are
query. Class prototypes are the L2-normalized mean of each class's support
embeddings; queries are scored against prototypes by scaled cosine similarity and
trained with cross-entropy. Cosine (rather than raw Euclidean) keeps the training
metric consistent with the inference-time kNN/head, which compare unit embeddings.

This directly optimizes the few-shot transfer the Phase-2 probe gate measures.
See docs/architecture.md §5.2.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _protos_query_targets(embeddings: Tensor, n_way: int, k_shot: int, q_query: int) -> tuple[Tensor, Tensor, Tensor]:
    support = embeddings[: n_way * k_shot].reshape(n_way, k_shot, -1)
    query = embeddings[n_way * k_shot :].reshape(n_way * q_query, -1)
    prototypes = F.normalize(support.mean(dim=1), dim=1)  # [N, D]
    targets = torch.arange(n_way, device=embeddings.device).repeat_interleave(q_query)
    return prototypes, query, targets


def prototypical_loss(
    embeddings: Tensor, n_way: int, k_shot: int, q_query: int, *, scale: float = 10.0
) -> tuple[Tensor, Tensor]:
    """Return (cross-entropy loss, query accuracy) for one episode.

    ``embeddings`` are unit-norm rows in class-major order (support then query).
    ``scale`` sharpens the cosine logits (an inverse softmax temperature).
    """
    prototypes, query, targets = _protos_query_targets(embeddings, n_way, k_shot, q_query)
    logits = scale * query @ prototypes.t()  # cosine, query already unit-norm
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(dim=1) == targets).float().mean()
    return loss, acc


def nearest_prototype_accuracy(embeddings: Tensor, n_way: int, k_shot: int, q_query: int) -> Tensor:
    """N-way K-shot nearest-prototype accuracy (scale-free; for evaluation)."""
    prototypes, query, targets = _protos_query_targets(embeddings, n_way, k_shot, q_query)
    logits = query @ prototypes.t()
    return (logits.argmax(dim=1) == targets).float().mean()
