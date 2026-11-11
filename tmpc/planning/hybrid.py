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
                 safety_halfspaces=True, selector_cfg: SelectorConfig = SelectorConfig(), threads=2,
                 deadlock_relax=False):
        torch.set_num_threads(threads)
        self.p = params
        self.device = device
        self.model, self.fcfg, self.meta = load_checkpoint(model_path, device)
        self.selector = CandidateSelector(params, selector_cfg)
        self.use_selector = use_selector
        self.safety = safety_halfspaces
        self.mpc = LTVMPC(params)
        self.deadlock_relax = deadlock_relax

    def reset(self, route, v_des, ego: EgoState):
        self.u = np.zeros(2)
        self.stuck = 0.0

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

    @staticmethod
    def compensate_dead_time(world, ego_xy, max_shift=10, moved=0.15):
        """Remove the launch dead time learned from CARLA's actuator lag.

        Demonstrations starting from standstill contain ~1 s of no motion (gear engagement); when
        tracked, that pause would be applied twice and the car never launches.  Near standstill,
        each candidate's timing is advanced to its first motion (path unchanged, <= 1 s)."""
        out = world.copy()
        for i, c in enumerate(world):
            dist = np.hypot(c[:, 0] - ego_xy[0], c[:, 1] - ego_xy[1])
            k = np.nonzero(dist > moved)[0]
            if not len(k) or k[0] == 0:
                continue
            sh = int(min(k[0], max_shift))
            tail = c[-1] + (np.arange(1, sh + 1)[:, None]) * (c[-1] - c[-2])
            out[i] = np.vstack([c[sh:], tail])
        return out

    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        world, logits = self.predict(obs)
        if obs.ego.v < 0.5:
            world = self.compensate_dead_time(world, (obs.ego.x, obs.ego.y))
        t_nn = time.perf_counter() - t0
        self.stuck = self.stuck + 0.1 if obs.ego.v < 0.1 else 0.0
        relax = min(0.4, 0.05 * max(0.0, self.stuck - 2.0)) if self.deadlock_relax else 0.0
        if self.use_selector:
            sel = self.selector.select(world, logits, obs, relax=relax)
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
