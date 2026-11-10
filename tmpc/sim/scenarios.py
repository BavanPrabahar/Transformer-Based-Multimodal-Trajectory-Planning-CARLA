"""Scenario templates for campus-like low-speed driving.

A scenario is fully determined by (town, template, seed): the route choice at
junctions, obstacle placement, walker timing and traffic all come from a
numpy RNG seeded with `seed`, so every controller is evaluated on identical
initial conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np

TEMPLATES = {
    # name: (weight for training sampling, description)
    "blocked": (3, "parked vehicle partially/fully blocking the ego lane"),
    "blocked_oncoming": (3, "blocked lane + oncoming vehicle in the passing lane"),
    "slalom": (2, "two parked vehicles on alternating sides"),
    "cones": (2, "construction cones/barrier closing part of the lane"),
    "walker": (2, "pedestrian crossing ahead (triggered by the ego)"),
    "leader": (2, "slow lead vehicle in the ego lane"),
    "mixed": (2, "blocked lane + crossing pedestrian + background traffic"),
    "traffic": (1, "route following in background traffic only"),
}


@dataclass
class ScenarioSpec:
    town: str
    template: str
    seed: int
    route_length: float = 160.0
    v_des: float = 4.0
    n_traffic: int = 0
    traffic_speed_diff: float = 40.0     # % below the speed limit for TM vehicles
    perception_noise: bool = True
    elements: List[Dict] = field(default_factory=list)
    ego_init: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def sample_spec(town: str, template: str, seed: int, train: bool = True) -> ScenarioSpec:
    rng = np.random.default_rng(seed)
    v_des = float(rng.uniform(3.0, 5.5))
    spec = ScenarioSpec(town=town, template=template, seed=seed, v_des=v_des,
                        route_length=float(rng.uniform(140.0, 190.0)))
    spec.ego_init = dict(d=float(rng.normal(0, 0.25)) if train else 0.0,
                         dyaw=float(rng.normal(0, 0.05)) if train else 0.0)
    el = spec.elements

    def parked(s_frac, d, kind="vehicle"):
        return dict(type="parked", s_frac=s_frac, d=d, dyaw=float(rng.normal(0, 0.08)), kind=kind)

    if template in ("blocked", "blocked_oncoming", "mixed"):
        el.append(parked(float(rng.uniform(0.3, 0.6)), float(rng.uniform(-0.6, 0.9))))
    if template == "blocked_oncoming":
        el.append(dict(type="oncoming", ahead=float(rng.uniform(70.0, 120.0)), speed=float(rng.uniform(3.0, 5.0))))
    if template == "slalom":
        f = float(rng.uniform(0.3, 0.5))
        gap = float(rng.uniform(12.0, 20.0))
        el.append(parked(f, float(rng.uniform(-0.6, -0.2))))
        el.append(dict(parked(f, 0.0), s_offset=gap, d_opposite=float(rng.uniform(0.0, 0.6))))
    if template == "cones":
        el.append(dict(type="cones", s_frac=float(rng.uniform(0.3, 0.6)), d=float(rng.uniform(-0.8, 0.4)),
                       length=float(rng.uniform(4.0, 8.0)), barrier=bool(rng.random() < 0.5)))
    if template in ("walker", "mixed"):
        el.append(dict(type="walker", s_frac=float(rng.uniform(0.35, 0.75)), speed=float(rng.uniform(0.9, 1.6)),
                       trigger=float(rng.uniform(10.0, 22.0)), left_to_right=bool(rng.random() < 0.5)))
    if template == "leader":
        el.append(dict(type="leader", ahead=float(rng.uniform(15.0, 30.0)), speed=float(rng.uniform(1.8, 3.0))))
    if template in ("mixed", "traffic"):
        spec.n_traffic = int(rng.integers(15, 35))
    elif train and rng.random() < 0.4:
        spec.n_traffic = int(rng.integers(5, 20))
    spec.traffic_speed_diff = float(rng.uniform(30.0, 60.0))
    return spec


def training_plan(towns, n_episodes: int, seed0: int = 1000):
    names = list(TEMPLATES)
    w = np.array([TEMPLATES[n][0] for n in names], float)
    w /= w.sum()
    rng = np.random.default_rng(seed0)
    plan = []
    per_town = int(np.ceil(n_episodes / len(towns)))
    for ti, town in enumerate(towns):
        for k in range(per_town):
            if len(plan) >= n_episodes:
                break
            tpl = names[int(rng.choice(len(names), p=w))]
            plan.append((town, tpl, seed0 + 100000 * ti + k))
    return plan


def evaluation_plan(town: str, per_template: int, seed0: int = 777000, templates=None):
    templates = templates or [t for t in TEMPLATES]
    plan = []
    for ti, tpl in enumerate(templates):
        for k in range(per_template):
            plan.append((town, tpl, seed0 + 1000 * ti + k))
    return plan
