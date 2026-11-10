"""Train the multimodal trajectory transformer.

Usage: python scripts/train.py --data data/features --epochs 40 --out checkpoints/planner.pt
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmpc.planning.features import FeatureConfig  # noqa: E402
from tmpc.planning.model import ModelConfig, TrajectoryTransformer, metrics, save_checkpoint, wta_loss  # noqa: E402

KEYS = ["hist", "agents", "agents_mask", "route", "route_mask", "ctx", "target"]


def load(path, device):
    d = np.load(path)
    out = {k: torch.from_numpy(d[k]).to(device) for k in KEYS}
    return out, (d["template"] if "template" in d else None)


def augment(b, fc: FeatureConfig):
    """Perception-like noise on agent features (normalised units), like the eval perception model."""
    a = b["agents"].clone()
    m = b["agents_mask"].unsqueeze(-1).float()
    noise = torch.zeros_like(a)
    noise[..., 0:2] = torch.randn_like(a[..., 0:2]) * (0.10 / fc.pos_scale)
    noise[..., 4:6] = torch.randn_like(a[..., 4:6]) * (0.15 / fc.vel_scale)
    noise[..., 6:8] = torch.randn_like(a[..., 6:8]) * (0.05 / 5.0)
    th = torch.randn_like(a[..., 2]) * 0.03
    c, s = a[..., 2].clone(), a[..., 3].clone()
    noise[..., 2] = c * torch.cos(th) - s * torch.sin(th) - c
    noise[..., 3] = s * torch.cos(th) + c * torch.sin(th) - s
    b = dict(b)
    b["agents"] = a + noise * m
    return b


def batches(data, bs, shuffle=True):
    n = len(data["target"])
    idx = torch.randperm(n, device=data["target"].device) if shuffle else torch.arange(n, device=data["target"].device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield {k: v[j] for k, v in data.items()}


def evaluate(model, data, bs=1024):
    model.eval()
    agg = {}
    with torch.no_grad():
        for b in batches(data, bs, shuffle=False):
            traj, logits = model(b["hist"], b["agents"], b["agents_mask"], b["route"], b["route_mask"], b["ctx"])
            m = metrics(traj, logits, b["target"])
            for k, v in m.items():
                agg.setdefault(k, []).append(v.cpu())
    return {k: float(torch.cat(v).mean()) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hist-dropout", type=float, default=0.3)
    ap.add_argument("--out", default="checkpoints/planner.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fc = FeatureConfig(**json.load(open(f"{args.data}_config.json")))
    train, _ = load(f"{args.data}_train.npz", args.device)
    val, _ = load(f"{args.data}_val.npz", args.device)
    print(f"train {len(train['target'])} val {len(val['target'])} samples on {args.device}")
    mc = ModelConfig(future_steps=fc.future_steps, hist_len=fc.hist_len, route_points=fc.route_points)
    model = TrajectoryTransformer(mc).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_par / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train["target"]) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist = []
    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, n = 0.0, 0
        for b in batches(train, args.bs):
            b = augment(b, fc)
            traj, logits = model(b["hist"], b["agents"], b["agents_mask"], b["route"], b["route_mask"], b["ctx"],
                                 hist_dropout=args.hist_dropout)
            loss, parts = wta_loss(traj, logits, b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss) * len(b["target"])
            n += len(b["target"])
        vm = evaluate(model, val)
        row = dict(epoch=ep + 1, train_loss=tl / n, **{f"val_{k}": v for k, v in vm.items()},
                   lr=sched.get_last_lr()[0], time=time.time() - t0)
        hist.append(row)
        score = vm["minADE"] + vm["FDE1"] * 0.5
        tag = ""
        if score < best:
            best = score
            save_checkpoint(args.out, model, fc, extra=dict(epoch=ep + 1, val=vm, n_params=n_par))
            tag = " *"
        print(f"ep {ep + 1:3d} loss {tl / n:.4f} | val minADE {vm['minADE']:.3f} minFDE {vm['minFDE']:.3f} "
              f"ADE1 {vm['ADE1']:.3f} FDE1 {vm['FDE1']:.3f} miss {vm['miss']:.3f} | {time.time() - t0:.0f}s{tag}",
              flush=True)
    json.dump(hist, open(os.path.splitext(args.out)[0] + "_history.json", "w"), indent=2)


if __name__ == "__main__":
    main()
