"""Closed-loop evaluation in CARLA on held-out scenarios.

Every controller is run on exactly the same (town, template, seed) scenarios.
Results are appended to <out>/episodes.jsonl (resumable).

Usage:
  python scripts/evaluate.py --town Town02 --per-template 8 \
      --controllers lab_mpc capsule_nmpc expert hybrid --model checkpoints/planner.pt --out results/town02
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmpc.sim.carla_env import CarlaEnv  # noqa: E402
from tmpc.sim.runner import run_carla_episode  # noqa: E402
from tmpc.sim.scenarios import TEMPLATES, evaluation_plan, sample_spec  # noqa: E402


def make_controller(name, params, args):
    if name == "lab_mpc":
        from tmpc.controllers.lab_mpc import LabStackController
        return LabStackController(params)
    if name == "capsule_nmpc":
        from tmpc.controllers.capsule_nmpc import CapsuleNMPC
        return CapsuleNMPC(params)
    if name == "expert":
        from tmpc.expert.lattice import ExpertController
        return ExpertController(params)
    if name.startswith("hybrid"):
        from tmpc.planning.hybrid import HybridController
        kw = dict(device=args.device)
        if name == "hybrid_noselector":
            kw.update(use_selector=False)
        if name == "hybrid_nosafety":
            kw.update(safety_halfspaces=False)
        if name == "hybrid_top1":
            kw.update(use_selector=False, safety_halfspaces=False)
        if name == "hybrid_soft":
            # revised after the ablation: hard-reject only actual circle overlap (circles are already
            # ~0.3 m conservative w.r.t. the boxes), trust the model's ranking more, expert-style
            # dead-lock relaxation; chosen on Town01 validation scenarios, not on Town02
            from tmpc.planning.selector import SelectorConfig
            kw.update(selector_cfg=SelectorConfig(min_clearance=0.0, w_prob=2.0), deadlock_relax=True)
        c = HybridController(params, args.model, **kw)
        c.name = name
        return c
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--per-template", type=int, default=8)
    ap.add_argument("--templates", nargs="*", default=None)
    ap.add_argument("--controllers", nargs="+", default=["lab_mpc", "capsule_nmpc", "expert", "hybrid"])
    ap.add_argument("--model", default="checkpoints/planner.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/town02")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--seed0", type=int, default=777000)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "episodes.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["controller"], r["town"], r["template"], r["seed"]))
    plan = evaluation_plan(args.town, args.per_template, args.seed0, args.templates)
    env = None
    ctrls = {}
    t0 = time.time()
    for (town, tpl, seed) in plan:
        spec = sample_spec(town, tpl, seed, train=False)
        for name in args.controllers:
            if (name, town, tpl, seed) in done:
                continue
            res = None
            for attempt in range(6):
                try:
                    if env is None:
                        env = CarlaEnv(port=args.port, tm_port=args.tm_port, sim_latency=not args.no_latency)
                        ctrls = {}
                    if env.params is None:
                        env.reset(spec)
                    if name not in ctrls:
                        ctrls[name] = make_controller(name, env.params, args)
                    res = run_carla_episode(env, ctrls[name], spec, privileged=(name == "expert"))
                    break
                except Exception as ex:
                    print(f"{name} {tpl} {seed} failed: {ex}", flush=True)
                    traceback.print_exc()
                    try:
                        if env is not None:
                            env.close()
                    except Exception:
                        pass
                    env = None
                    time.sleep(15.0)
            if res is None:
                continue
            with open(path, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"[{(time.time() - t0) / 60:6.1f} min] {name:16s} {tpl:16s} {seed} {res['status']:9s} "
                  f"prog={res['progress']:.2f} t={res['time']:5.1f} min_d={res['min_dist']:.2f} "
                  f"solve={res['solve_mean_ms']:.1f}/{res['solve_p99_ms']:.1f}ms", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
