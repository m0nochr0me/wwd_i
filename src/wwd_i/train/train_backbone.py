"""Train the from-scratch backbone with cosine-prototypical metric learning.

Phase-2 driver. Runs locally on CPU for a quick check (use ``--limit`` for a fast
smoke run) or on Colab GPU for the real run. It builds the Speech Commands
episodic samplers, optimizes the prototypical loss over the training words, and
every ``--eval-every`` steps reports the few-shot probe on held-out words against
a mel-only baseline (the Phase-2 gate). The best checkpoint is saved and the
frozen backbone is exported to ONNX.

Run: ``uv run --group train python -m wwd_i.train.train_backbone --help``
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from wwd_i.data.speech_commands import speech_commands_samplers
from wwd_i.features.melspec import MelSpectrogram
from wwd_i.models.backbone import Backbone, BackboneConfig, export_onnx
from wwd_i.train.probe import few_shot_probe
from wwd_i.train.proto import prototypical_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    melspec = MelSpectrogram().eval().to(device)
    model = Backbone(BackboneConfig(embedding_dim=args.embedding_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_sampler, held_sampler = speech_commands_samplers(args.root, limit=args.limit, seed=args.seed)

    def embed(audio: Tensor) -> Tensor:
        return model(melspec(audio))

    def baseline(audio: Tensor) -> Tensor:
        return F.normalize(melspec(audio).mean(dim=1), dim=1)  # mean log-mel over time

    best = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        audio = train_sampler.episode(args.n_way, args.k_shot, args.q_query).to(device)
        loss, acc = prototypical_loss(model(melspec(audio)), args.n_way, args.k_shot, args.q_query)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(f"step {step:>5} | loss {loss.item():.4f} | train acc {acc.item():.3f}")

        if step % args.eval_every == 0:
            model.eval()
            res = few_shot_probe(
                held_sampler,
                embed,
                baseline,
                n_way=args.probe_n_way,
                k_shot=args.k_shot,
                q_query=args.q_query,
                episodes=args.probe_episodes,
                device=device,
            )
            verdict = "PASS" if res["margin"] > 0 else "----"
            print(
                f"  [probe @ {step}] backbone {res['backbone']:.3f} vs mel-baseline "
                f"{res['baseline']:.3f} (margin {res['margin']:+.3f}) {verdict}"
            )
            if res["backbone"] > best:
                best = res["backbone"]
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.out)

    model.eval()
    export_onnx(model, args.onnx)
    print(f"done | best held-out probe {best:.3f} | checkpoint {args.out} | onnx {args.onnx}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the wwd_i backbone (Phase 2).")
    p.add_argument("--root", default="data/speech_commands", help="Speech Commands download/cache dir")
    p.add_argument("--limit", type=int, default=None, help="cap clips per word (fast local smoke run)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--n-way", type=int, default=10, help="classes per training episode")
    p.add_argument("--probe-n-way", type=int, default=10, help="classes per probe episode (<= held-out words)")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--probe-episodes", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/backbone.pt", help="best-checkpoint path")
    p.add_argument("--onnx", default="artifacts/backbone.onnx", help="exported frozen backbone")
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
