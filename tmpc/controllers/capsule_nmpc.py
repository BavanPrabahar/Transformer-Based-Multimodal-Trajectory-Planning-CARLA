"""Classical obstacle-avoiding NMPC with capsule (multi-circle) collision constraints.

This is the "properly built" version of the capsule baseline the original repo
described but never implemented:

* same CasADi/IPOPT kinematic-bicycle MPC as the lab stack (rate limits, slack),
* ego body covered by 2 circles, each obstacle by 1-3 circles (a discretised
  capsule), obstacle motion predicted with constant velocity over the horizon,
* hard separation constraints  ||c_ego - c_obs||^2 >= (r_e + r_o + d_safe)^2,
  relaxed by a heavily penalised slack so the NLP never becomes infeasible,
* drivable-area constraints in the route frame (same bounds every stack gets),
* warm start from the shifted previous solution.

It tracks the route centreline at the desired speed, i.e. the non-convex
obstacle constraints and the tracking objective compete inside one NLP.
"""
from __future__ import annotations

import time

import numpy as np

from ..geometry import Agent, KIND_WALKER
from ..types import Command, Controller, EgoState, Observation
from ..vehicle import VehicleParams
from .lab_mpc import KinMPCPathFollower, fix_angle_reference


class CapsuleNMPC(Controller):
    name = "capsule_nmpc"

    def __init__(self, params: VehicleParams, N=16, DT=0.25, Q=(10., 10., 20., 5.), R=(10., 500.),
                 max_cpu_time=0.1, max_obstacles=4, circles_per_obstacle=3, d_safe=0.3,
                 horizon_range=35.0):
        self.p = params
        self.max_obs = max_obstacles
        self.cpo = circles_per_obstacle
        self.range = horizon_range
        # road bounds are weighted like the obstacle constraints: leaving the road is not a cheap escape
        self.mpc = KinMPCPathFollower(params, N=N, DT=DT, Q=Q, R=R, max_cpu_time=max_cpu_time,
                                      n_obs_circles=max_obstacles * circles_per_obstacle,
                                      road_bounds=True, d_safe=d_safe, road_slack_weight=(1e4, 1e5))
        self.N, self.DT = N, DT

    def reset(self, route, v_des, ego: EgoState):
        self.route, self.v_des = route, v_des
        self.acc_prev, self.df_prev = 0.0, 0.0
        self.ws = None
        self.fails = 0

    # ------------------------------------------------------------------------------
    def _reference(self, e: EgoState, s_ego):
        N, DT = self.N, self.DT
        # ramped speed profile as in the lab reference (a_max), then constant v_des
        v = np.empty(N)
        vk = e.v
        for k in range(N):
            vk = min(self.v_des, vk + self.p.a_max * DT) if vk < self.v_des else self.v_des
            v[k] = vk
        s = s_ego + np.cumsum(v * DT)
        s = np.minimum(s, self.route.length)
        x, y, psi = self.route.pose(s)
        psi = fix_angle_reference(psi, e.yaw)
        return np.stack([x, y, psi, v], 1), s

    def _select_obstacles(self, obs: Observation):
        e = obs.ego
        cands = []
        for a in obs.agents:
            d = np.hypot(a.x - e.x, a.y - e.y)
            if d > self.range:
                continue
            # skip agents clearly behind and moving away
            rel = np.array([a.x - e.x, a.y - e.y])
            if rel @ [np.cos(e.yaw), np.sin(e.yaw)] < -5.0:
                continue
            cands.append((d, a))
        cands.sort(key=lambda t: t[0])
        return [a for _, a in cands[: self.max_obs]]

    def _obstacle_params(self, agents):
        N, M = self.N, self.max_obs * self.cpo
        ox = 1e4 * np.ones((N, M))
        oy = 1e4 * np.ones((N, M))
        orad = 0.1 * np.ones((N, M))
        times = self.DT * (np.arange(N) + 1)
        for i, a in enumerate(agents):
            n = 1 if a.kind == KIND_WALKER else self.cpo
            offs, r = a.circles(n)
            pred = a.predict(times)
            for j, c in enumerate(offs):
                col = i * self.cpo + j
                ox[:, col] = pred[:, 0] + c * np.cos(pred[:, 2])
                oy[:, col] = pred[:, 1] + c * np.sin(pred[:, 2])
                orad[:, col] = r
        return ox, oy, orad

    def _road_params(self, s_k):
        x, y, yaw = self.route.pose(s_k)
        lo_hi = self.route.bounds(s_k)
        n = np.stack([-np.sin(yaw), np.cos(yaw)], 1)
        return np.stack([x, y], 1), n, lo_hi[1], lo_hi[0]

    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        e = obs.ego
        zref, s_ref = self._reference(e, obs.s_ego)
        m = self.mpc
        m.set_state(e.x, e.y, e.yaw, e.v)
        m.set_reference(zref)
        m.set_prev_input(self.acc_prev, self.df_prev)
        agents = self._select_obstacles(obs)
        m.set_obstacles(*self._obstacle_params(agents))
        if self.ws is not None:
            z_prev = self.ws[0]
            s_k, _ = self.route.project_many(z_prev[1:, :2], obs.s_ego - 5, obs.s_ego + 60)
        else:
            s_k = s_ref
        m.set_road(*self._road_params(s_k))
        if self.ws is None:
            z0 = np.vstack([e.z(), zref])
            z0[1:, 2] = zref[:, 2]
            m.warm_start(z0, np.zeros((self.N, 2)), np.zeros((self.N, 2)))
        else:
            z, u, sl = self.ws
            m.warm_start(np.vstack([z[1:], z[-1:]]), np.vstack([u[1:], u[-1:]]), np.vstack([sl[1:], sl[-1:]]))
        sol = m.solve()
        if sol["optimal"] or self.ws is None:
            self.fails = 0
            self.ws = (sol["z_mpc"], sol["u_mpc"], sol["sl_mpc"])
            plan = sol["z_mpc"]
            a, df = float(sol["u_control"][0]), float(sol["u_control"][1])
        else:
            # IPOPT did not converge within the real-time budget: never execute an unconverged
            # iterate; follow the shifted previous accepted plan, and brake if failures persist.
            self.fails += 1
            z, u, sl = self.ws
            z, u, sl = np.vstack([z[1:], z[-1:]]), np.vstack([u[1:], u[-1:]]), np.vstack([sl[1:], sl[-1:]])
            self.ws = (z, u, sl)
            plan = z
            a, df = float(u[0, 0]), float(u[0, 1])
            if self.fails > 5:
                a = min(a, -1.5)
        self.acc_prev, self.df_prev = a, df
        return Command(a, df, time.perf_counter() - t0, bool(sol["optimal"]), plan[:, :2],
                       dict(solver_time=sol["solve_time"], obs_slack=float(np.max(sol["sl_obs"])),
                            n_obstacles=len(agents), fallback=not sol["optimal"]))
