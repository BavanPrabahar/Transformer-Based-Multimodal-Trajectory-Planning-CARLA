"""Run several controllers on one scenario and plot their driven paths (qualitative figure).

Usage: python scripts/plot_episode.py --town Town02 --template blocked_oncoming --seed 778000 \
          --controllers lab_mpc capsule_nmpc hybrid --model checkpoints/planner.pt --out results/figures/example.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.evaluate import make_controller  # noqa: E402
from scripts.report import GRID, INK, INK2, LABEL, MUTED, SERIES, SURFACE  # noqa: E402
from tmpc.geometry import box_corners  # noqa: E402
from tmpc.sim.carla_env import CarlaEnv  # noqa: E402
from tmpc.sim.runner import run_carla_episode  # noqa: E402
from tmpc.sim.scenarios import sample_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--template", default="blocked_oncoming")
    ap.add_argument("--seed", type=int, default=778000)
    ap.add_argument("--controllers", nargs="+", default=["lab_mpc", "capsule_nmpc", "hybrid"])
    ap.add_argument("--model", default="checkpoints/planner.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--out", default="results/figures/example.png")
    args = ap.parse_args()
    env = CarlaEnv(port=args.port, tm_port=args.tm_port)
    spec = sample_spec(args.town, args.template, args.seed, train=False)
    runs = {}
    for name in args.controllers:
        if env.params is None:
            env.reset(spec)
        ctrl = make_controller(name, env.params, args)
        res = run_carla_episode(env, ctrl, spec, privileged=(name == "expert"), record=True)
        runs[name] = res
        print(name, res["status"], f"{res['time']:.1f}s")
    statics = [a for a in env.ground_truth_agents() if a.speed < 0.05]
    route = env.route
    env.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    # drivable band
    s = route.s
    x, y, yaw = route.pose(s)
    nx, ny = -np.sin(yaw), np.cos(yaw)
    half = 0.92 + 0.25
    lx, ly = x + nx * (route.d_left + half), y + ny * (route.d_left + half)
    rx, ry = x + nx * (route.d_right - half), y + ny * (route.d_right - half)
    ax.fill(np.concatenate([lx, rx[::-1]]), np.concatenate([ly, ry[::-1]]), color="#f0efec", zorder=0)
    ax.plot(x, y, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    for a in statics:
        c = box_corners(a.x, a.y, a.yaw, a.length, a.width)
        ax.fill(c[:, 0], c[:, 1], color="#52514e", zorder=2)
    for (name, res), col in zip(runs.items(), SERIES):
        e = res["record"]["ego"]
        ax.plot(e[:, 0], e[:, 1], color=col, linewidth=2, zorder=3,
                label=f"{LABEL.get(name, name)} — {res['status']}")
        ax.scatter(e[-1, 0], e[-1, 1], s=40, color=col, edgecolors=SURFACE, linewidths=2, zorder=4)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_title(f"{args.town} · {args.template} · seed {args.seed}: driven paths (grey = parked obstacles)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
