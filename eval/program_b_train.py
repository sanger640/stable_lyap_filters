"""Program B: train FIGNet* (actuated gripper) or the object-level GNS baseline on the generic corpus.

Both use batches of 8 episodes x 8 samples, the port's learning-rate schedule
(1e-3 * 0.1 ** (step / 1e6) + 1e-6, Adam), 5 normalizer warm-up batches, and per-node MSE on
normalized accelerations of moving nodes (objects and gripper). ``--benchmark N`` runs N steps,
reports throughput and writes nothing else.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import program_b_fignet as pf  # noqa: E402
import program_b_gns as pg  # noqa: E402
from fignet.data_loader import ToTensor  # noqa: E402
from fignet.simulator import LearnedSimulator  # noqa: E402
from fignet.types import KinematicType, NodeType  # noqa: E402

CORPUS = ROOT / "results/program_b/corpus/train"
CHECKPOINTS = ROOT / "results/program_b/checkpoints"
WARMUP_BATCHES = 5
LATENT, ROUNDS, MLP_LAYERS, HIDDEN = 128, 10, 2, 128


def learning_rate(step):
    return 1e-3 * 0.1 ** (step / 1e6) + 1e-6


def build(model, device):
    if model == "fignet":
        return LearnedSimulator(mesh_dimensions=3, latent_dim=LATENT, nmessage_passing_steps=ROUNDS,
                                nmlp_layers=MLP_LAYERS, mlp_hidden_dim=HIDDEN, input_seq_length=3,
                                device=device, leave_out_mm=True, extra_dim=pf.EXTRA_DIM).to(device)
    return pg.ObjectGNS(LATENT, ROUNDS, MLP_LAYERS, HIDDEN, device=device).to(device)


def dataset(model, files, batches, seed):
    cls = pf.GraphBatches if model == "fignet" else pg.SampleBatches
    return cls(files, batches, seed)


def fignet_loss(sim, batch, device):
    mask = (batch.node_sets[NodeType.MESH].kinematic != KinematicType.STATIC).squeeze().to(device)
    batch = ToTensor(device)(batch)
    prediction, _ = sim.predict_accelerations(batch)
    target = sim.normalize_accelerations(batch.node_sets[NodeType.MESH].target)
    return ((prediction - target) ** 2).sum(-1)[mask].mean()


def gns_loss(model, batch, device):
    return model.loss({k: v.to(device) for k, v in batch.items()})


def save(model_name, net, path, step, optimizer):
    if model_name == "fignet":
        net.save(str(path))
    else:
        torch.save({"model": net.state_dict(),
                    **{n: getattr(net, n).get_variable()
                       for n in ("node_norm", "edge_norm", "output_norm")}}, path)
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, path.with_suffix(".state"))


def restore(model_name, path, device):
    """Weights and normalizer statistics, still in training mode."""
    net = build(model_name, device)
    if model_name == "fignet":
        net.load(str(path))
    else:
        data = torch.load(path, map_location=device, weights_only=False)
        net.load_state_dict(data["model"])
        for name in ("node_norm", "edge_norm", "output_norm"):
            for key, value in data[name].items():
                setattr(getattr(net, name), key, value.to(device) if torch.is_tensor(value) else value)
    return net


def load(model_name, path, device):
    """Checkpoint for evaluation: frozen normalizers, eval mode."""
    net = restore(model_name, path, device)
    freeze_normalizers(net)
    net.eval()
    return net


def worker_init(_):
    torch.set_num_threads(1)


def freeze_normalizers(net):
    """The port's normalizers keep accumulating whenever called; stop that for evaluation."""
    for module in net.modules():
        if module.__class__.__name__ == "Normalizer":
            module._max_accumulations = 0
    if hasattr(net, "accumulate"):
        net.accumulate = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("fignet", "gns"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--benchmark", type=int, default=0)
    args = parser.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    files = sorted(CORPUS.glob("ep*.npz"))
    if len(files) != 5000:
        raise SystemExit(f"expected 5000 training trajectories, found {len(files)}")
    steps = args.benchmark or args.steps
    run = CHECKPOINTS / f"{args.model}_s{args.seed}"
    net = build(args.model, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate(0))
    start_step = 0
    if not args.benchmark:
        run.mkdir(parents=True, exist_ok=True)
        latest = sorted(run.glob("step_*.pt"))
        if latest:
            net = restore(args.model, latest[-1], device)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate(0))
            state = torch.load(latest[-1].with_suffix(".state"), weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"] + 1
            print(f"resuming from step {start_step}", flush=True)
    loss_fn = fignet_loss if args.model == "fignet" else gns_loss
    data = dataset(args.model, files, WARMUP_BATCHES + steps, args.seed)
    indices = list(range(WARMUP_BATCHES + start_step, WARMUP_BATCHES + steps))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, indices), batch_size=None,
                                         shuffle=False, num_workers=args.workers, prefetch_factor=4,
                                         persistent_workers=args.workers > 0,
                                         worker_init_fn=worker_init)
    net.train()
    if start_step == 0:
        with torch.no_grad():
            for i in range(WARMUP_BATCHES):
                loss_fn(net, data[i], device)
    log = (run / "train_log.jsonl").open("a") if not args.benchmark else None
    started, window, losses = time.time(), time.time(), []
    for step, batch in zip(range(start_step, steps), loader):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        loss = loss_fn(net, batch, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            rate = len(losses) / (time.time() - window)
            row = {"step": step + 1, "loss": float(np.mean(losses)), "lr": learning_rate(step),
                   "steps_per_s": rate, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            print(json.dumps(row), flush=True)
            if log:
                log.write(json.dumps(row) + "\n"); log.flush()
            window, losses = time.time(), []
        if not args.benchmark and ((step + 1) % args.save_every == 0 or step + 1 == steps):
            save(args.model, net, run / f"step_{step + 1:07d}.pt", step, optimizer)
    total = time.time() - started
    summary = {"model": args.model, "seed": args.seed, "steps": steps - start_step,
               "seconds": total, "steps_per_s": (steps - start_step) / total,
               "parameters": sum(p.numel() for p in net.parameters()),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(summary), flush=True)
    if not args.benchmark:
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
