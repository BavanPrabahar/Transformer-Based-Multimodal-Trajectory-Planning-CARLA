"""Hybrid stack: trajectory transformer -> candidate selector -> convex LTV-MPC."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..controllers.ltv_mpc import LTVMPC
from ..geometry import to_global
from ..types import Command, Controller, EgoState, Observation
from ..vehicle import VehicleParams
from .features import collate, features_from_obs
from .model import load_checkpoint
from .selector import CandidateSelector, SelectorConfig


class HybridController(Controller):
    name = "hybrid"

    def __init__(self, params: VehicleParams, model_path: str, device="cpu", use_selector=True,
                 safety_halfspaces=True, selector_cfg: SelectorConfig = SelectorConfig(), threads=2):
        torch.set_num_threads(threads)
        self.p = params
        self.device = device
        self.model, self.fcfg, self.meta = load_checkpoint(model_path, device)
        self.selector = CandidateSelector(params, selector_cfg)
        self.use_selector = use_selector
        self.safety = safety_halfspaces
        self.mpc = LTVMPC(params)

    def reset(self, route, v_des, ego: EgoState):
        self.u = np.zeros(2)

    @torch.no_grad()
    def predict(self, obs: Observation):
        f = collate([features_from_obs(obs, self.fcfg)])
        b = {k: torch.from_numpy(v).to(self.device) for k, v in f.items()}
        traj, logits = self.model(b["hist"], b["agents"], b["agents_mask"], b["route"], b["route_mask"], b["ctx"])
        traj = traj[0].cpu().numpy().astype(float)
        logits = logits[0].cpu().numpy().astype(float)
        e = obs.ego
        world = to_global(traj, e.x, e.y, e.yaw)
        return world, logits

    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        world, logits = self.predict(obs)
        t_nn = time.perf_counter() - t0
        if self.use_selector:
            sel = self.selector.select(world, logits, obs)
        else:
            from ..controllers.ltv_mpc import reference_from_xy
            i = int(np.argmax(logits))
            e = obs.ego
            sel = dict(zref=reference_from_xy(world[i], 0.1, (e.x, e.y), e.yaw, e.v), mode=i, fallback=0,
                       clearance=np.nan, n_feasible=-1)
        t_sel = time.perf_counter() - t0 - t_nn
        N = self.mpc.N
        out = self.mpc.solve(obs.ego.z(), sel["zref"][:N], self.u, agents=obs.agents if self.safety else None,
                             route=obs.route)
        self.u = out["u"][0].copy()
        return Command(float(self.u[0]), float(self.u[1]), time.perf_counter() - t0, bool(out["ok"]),
                       sel["zref"][:, :2],
                       dict(nn_time=t_nn, select_time=t_sel, solver_time=out["solve_time"], mode=sel["mode"],
                            fallback=sel["fallback"], n_feasible=sel["n_feasible"], obs_slack=out["obs_slack"]))
