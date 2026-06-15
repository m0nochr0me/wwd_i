"""Few-shot embedding probe — the Phase-2 gate.

On held-out (unseen) words, measures N-way K-shot nearest-prototype accuracy from
an embedding function, averaged over many random episodes, and compares it against
a mel-only baseline (mean-pooled log-mel). The backbone passes the gate when its
few-shot accuracy is clearly above the mel baseline: that is the evidence the
frozen embedding generalizes to words it was never trained on, before any
wake-word head is built.

See docs/implementation-plan.md Phase 2.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from wwd_i.data.episodic import EpisodicSampler
from wwd_i.train.proto import nearest_prototype_accuracy


@torch.no_grad()
def few_shot_probe(
    sampler: EpisodicSampler,
    embed: Callable[[Tensor], Tensor],
    baseline: Callable[[Tensor], Tensor],
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Average few-shot accuracy of ``embed`` vs ``baseline`` over random episodes.

    ``embed`` and ``baseline`` each map an audio batch ``[B, L]`` to unit-norm
    feature rows ``[B, D]`` in the sampler's class-major order. Returns the mean
    backbone and baseline accuracies and their margin.
    """
    backbone_accs: list[float] = []
    baseline_accs: list[float] = []
    for _ in range(episodes):
        audio = sampler.episode(n_way, k_shot, q_query).to(device)
        backbone_accs.append(nearest_prototype_accuracy(embed(audio), n_way, k_shot, q_query).item())
        baseline_accs.append(nearest_prototype_accuracy(baseline(audio), n_way, k_shot, q_query).item())
    backbone = sum(backbone_accs) / len(backbone_accs)
    base = sum(baseline_accs) / len(baseline_accs)
    return {"backbone": backbone, "baseline": base, "margin": backbone - base, "n_way": n_way, "k_shot": k_shot}
