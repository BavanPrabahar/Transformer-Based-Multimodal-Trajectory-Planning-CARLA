"""Collect expert demonstrations in CARLA.

Each episode is saved as data/<split>/<town>_<template>_<seed>.npz together with a
row in data/<split>/index.csv.  Re-running skips episodes that already exist, so
the script can be stopped and resumed.

Usage:
  python scripts/collect_data.py --towns Town01 Town03 Town05 --episodes 500 --out data/train
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmpc.expert.lattice import ExpertController, LatticeConfig  # noqa: E402
from tmpc.sim.carla_env import CarlaEnv  # noqa: E402
from tmpc.sim.runner import run_carla_episode  # noqa: E402
from tmpc.sim.scenarios import sample_spec, training_plan  # noqa: E402

FIELDS = ["file", "town", "template", "seed", "status", "time", "progress", "min_dist", "n_steps",
          "side_pref", "perturbed", "wall_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", nargs="+", default=["Town01", "Town03", "Town05"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--perturb", type=float, default=0.35, help="probability of noise-injection bursts")
    ap.add_argument("--templates", nargs="*", default=None)
    ap.add_argument("--dagger", default=None, help="checkpoint: the learner drives, the expert labels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = os.path.join(args.out, "index.csv")
    done = set()
    if os.path.exists(index):
        with open(index) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    else:
        with open(index, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    plan = training_plan(args.towns, args.episodes, args.seed0)
    if args.templates:
        plan = [p for p in plan if p[1] in args.templates]
    # group by town to minimise map loads
    plan.sort(key=lambda p: args.towns.index(p[0]))
    env = None
    t_start = time.time()
    n_ok = 0
    for i, (town, tpl, seed) in enumerate(plan):
        fname = f"{town}_{tpl}_{seed}.npz"
        if fname in done:
            continue
        spec = sample_spec(town, tpl, seed, train=True)
        rng = np.random.default_rng(seed + 5)
        side = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 0.6))
        cfg = LatticeConfig(side_pref=side, w_opp=float(rng.uniform(1.0, 2.0)), w_clear=float(rng.uniform(3.0, 5.0)))
        res = None
        for attempt in range(6):
            try:
                if env is None:
                    env = CarlaEnv(port=args.port, tm_port=args.tm_port)
                res = run_with_params(env, spec, cfg, args.perturb, rng, args.dagger)
                break
            except Exception as ex:  # CARLA hiccup: reconnect once
                print(f"[{i}] {fname} failed: {ex}")
                traceback.print_exc()
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
                time.sleep(15.0)   # the supervisor restarts a crashed server
        if res is None:
            continue
        rec = res.pop("record")
        rec["spec"] = np.array(json.dumps(spec.to_dict()))
        np.savez_compressed(os.path.join(args.out, fname), **rec)
        row = dict(file=fname, town=town, template=tpl, seed=seed, status=res["status"], time=round(res["time"], 1),
                   progress=round(res["progress"], 3), min_dist=round(res["min_dist"], 2),
                   n_steps=len(rec["t"]), side_pref=round(side, 2), perturbed=int(rec["perturbed"].any()),
                   wall_time=round(res["wall_time"], 1))
        with open(index, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        n_ok += res["status"] == "success"
        el = time.time() - t_start
        print(f"[{i + 1}/{len(plan)}] {fname:38s} {res['status']:9s} t={res['time']:5.1f}s "
              f"prog={res['progress']:.2f} min_d={res['min_dist']:.2f} wall={res['wall_time']:.0f}s "
              f"(elapsed {el / 60:.1f} min, success so far {n_ok})", flush=True)
    if env is not None:
        env.close()


class ExpertLabeler:
    """Privileged lattice plan from the visited state -> (30, 2) world xy label."""

    def __init__(self, params, cfg):
        from tmpc.expert.lattice import LatticePlanner
        self.planner = LatticePlanner(params, cfg)

    def reset(self):
        self.planner.reset()

    def __call__(self, obs_priv):
        zref, _ = self.planner.plan(obs_priv)
        return zref[:30, :2].astype(np.float32)


class TakeoverMix:
    """DAgger execution policy: the learner drives; when it is stuck for > 3 s the privileged expert
    takes over for 4 s (so later parts of the scenario are still visited). Labels always come from
    the expert's plan at the visited state."""

    name = "dagger_mix"

    def __init__(self, learner, expert, env):
        self.learner, self.expert, self.env = learner, expert, env

    def reset(self, route, v_des, ego):
        self.learner.reset(route, v_des, ego)
        self.expert.reset(route, v_des, ego)
        self.stuck, self.until = 0.0, -1.0

    def step(self, obs):
        from dataclasses import replace
        if obs.t < self.until:
            cmd = self.expert.step(replace(obs, agents=self.env.last_gt))
            self.learner.u = np.array([cmd.a, cmd.df])
            return cmd
        cmd = self.learner.step(obs)
        self.expert.u = np.array([cmd.a, cmd.df])
        self.stuck = self.stuck + 0.1 if obs.ego.v < 0.2 else 0.0
        if self.stuck > 3.0:
            self.until, self.stuck = obs.t + 4.0, 0.0
        return cmd


def run_with_params(env, spec, cfg, perturb, rng, dagger=None):
    # reset once to learn the ego geometry, then build the expert with the right params
    if env.params is None:
        env.reset(spec)
    if dagger:
        from tmpc.planning.hybrid import HybridController
        learner = HybridController(env.params, dagger)
        mix = TakeoverMix(learner, ExpertController(env.params, cfg), env)
        return run_carla_episode(env, mix, spec, privileged=False, record=True, perturb_prob=0.0, rng=rng,
                                 freeze_limit=25.0, labeler=ExpertLabeler(env.params, cfg))
    ctrl = ExpertController(env.params, cfg)
    return run_carla_episode(env, ctrl, spec, privileged=True, record=True, perturb_prob=perturb, rng=rng,
                             freeze_limit=15.0)


if __name__ == "__main__":
    main()
