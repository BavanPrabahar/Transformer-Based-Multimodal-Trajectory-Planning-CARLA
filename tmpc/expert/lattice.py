"""Privileged Frenet-lattice expert used to generate demonstrations.

It sees ground-truth agents, samples (lateral target x transition length x speed
profile) candidates along the route, rejects ones that collide (multi-circle,
constant-velocity prediction), leave the drivable area or exceed the curvature
limit, and picks the cheapest.  A per-episode random side preference breaks
left/right ties so the demonstrations are genuinely multimodal (the same scene
can be passed on either side), which is what the multimodal planner must learn.
The chosen plan is tracked by the same convex LTV-MPC the hybrid stack uses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..controllers.ltv_mpc import LTVMPC
from ..geometry import Agent, KIND_WALKER, Path
from ..planning.safety import trajectory_clearance
from ..types import Command, Controller, EgoState, Observation
from ..vehicle import VehicleParams


@dataclass
class LatticeConfig:
    dt: float = 0.1
    horizon: int = 40            # 4 s
    d_step: float = 0.35
    lengths: tuple = (5.0, 8.0, 13.0, 20.0)
    comfort_decel: float = 2.0
    min_clearance: float = 0.5   # m, circle clearance required by the expert
    w_d: float = 1.0
    w_opp: float = 1.5           # per second spent outside the own lane
    w_v: float = 2.0
    w_len: float = 6.0
    w_clear: float = 4.0
    w_hyst: float = 0.6
    w_stop: float = 4.0
    side_pref: float = 0.0       # >0 prefers passing on the left, <0 on the right
    range_m: float = 45.0
    ahead_extra: float = 3.0     # extra standoff to obstacles straight ahead (m)
    bounds_steps: int = 25       # drivable-area check over the first 2.5 s
    bound_tol: float = 0.4       # hard limit on bound violation (m)
    w_bound: float = 40.0
    w_standoff: float = 15.0


def _quintic(d0, dd0, d1, L, s):
    """Lateral offset d(s) from (d0, d0', 0) to (d1, 0, 0) over length L; hold afterwards.

    Returns d, d' (per metre) and d'' (per metre^2), all analytic.
    """
    u = np.clip(s / L, 0.0, 1.0)
    inside = (s < L).astype(float)
    a0, a1 = d0, dd0 * L
    a3 = 10 * (d1 - d0) - 6 * a1
    a4 = -15 * (d1 - d0) + 8 * a1
    a5 = 6 * (d1 - d0) - 3 * a1
    d = a0 + a1 * u + a3 * u ** 3 + a4 * u ** 4 + a5 * u ** 5
    dp = (a1 + 3 * a3 * u ** 2 + 4 * a4 * u ** 3 + 5 * a5 * u ** 4) / L * inside
    dpp = (6 * a3 * u + 12 * a4 * u ** 2 + 20 * a5 * u ** 3) / L ** 2 * inside
    return d, dp, dpp


def speed_profiles(v0, targets, t, a_max, a_dec):
    """Constant-acceleration ramps from v0 to each target speed. -> v (P, T), s (P, T)."""
    vs, ss = [], []
    for vt in targets:
        a = a_max if vt > v0 else -a_dec
        tr = abs(vt - v0) / max(abs(a), 1e-6)
        v = np.where(t <= tr, v0 + a * t, vt)
        v = np.maximum(v, 0.0)
        dt = np.diff(np.concatenate([[0.0], t]))
        s = np.cumsum(0.5 * (np.concatenate([[v0], v[:-1]]) + v) * dt)
        vs.append(v)
        ss.append(s)
    return np.array(vs), np.array(ss)


class LatticePlanner:
    def __init__(self, params: VehicleParams, cfg: LatticeConfig = LatticeConfig()):
        self.p = params
        self.cfg = cfg
        self.ego_offs, self.ego_r = params.ego_circles(3)
        self.prev_d = None
        self.stuck = 0.0

    def reset(self):
        self.prev_d = None
        self.stuck = 0.0

    def plan(self, obs: Observation, agents: Optional[List[Agent]] = None):
        cfg, p = self.cfg, self.p
        e = obs.ego
        route = obs.route
        agents = obs.agents if agents is None else agents
        s0, d0 = obs.s_ego, obs.d_ego
        _, _, ryaw = route.pose(s0)
        dpsi = np.arctan2(np.sin(e.yaw - ryaw), np.cos(e.yaw - ryaw))
        dd0 = np.tan(np.clip(dpsi, -0.6, 0.6))
        T = cfg.horizon
        t = cfg.dt * (np.arange(T) + 1)

        # --- lateral targets within the tightest bounds of the next 30 m ------------
        sw = np.linspace(s0, min(s0 + 30.0, route.length), 16)
        dl, dr = route.bounds(sw)
        dmax, dmin = float(np.max(dl)), float(np.min(dr))
        if dmax < dmin:
            dmax = dmin = 0.5 * (dmax + dmin)
        d_targets = np.unique(np.concatenate([np.arange(dmin, dmax + 1e-6, cfg.d_step), [0.0]]))
        d_targets = d_targets[(d_targets >= dmin - 1e-6) & (d_targets <= dmax + 1e-6)]
        if len(d_targets) == 0:
            d_targets = np.array([np.clip(0.0, dmin, dmax)])

        # --- speed targets ---------------------------------------------------------------
        v_des = obs.v_des
        v_t = [v_des, 0.66 * v_des, 0.33 * v_des, 0.0]
        lead = self._lead_speed(obs, agents)
        if lead is not None and lead < v_des:
            v_t.append(max(lead, 0.0))
        v_t = np.array(sorted(set(np.round(v_t, 3))))
        V, S = speed_profiles(e.v, v_t, t, p.a_max, cfg.comfort_decel)
        Vh, Sh = speed_profiles(e.v, [0.0], t, p.a_max, 3.0)        # hard-brake fallback
        V = np.vstack([V, Vh])
        S = np.vstack([S, Sh])
        stop_flags = np.concatenate([v_t <= 1e-3, [True]])
        P = len(V)

        # --- assemble candidates (D x L x P) -----------------------------------------------
        cands = []
        for dt_ in d_targets:
            for L in cfg.lengths:
                L_eff = max(L, 1.2 * e.v + 4.0)
                for pi in range(P):
                    cands.append((dt_, L_eff, pi))
        n = len(cands)
        d_arr = np.array([c[0] for c in cands])
        L_arr = np.array([c[1] for c in cands])
        p_arr = np.array([c[2] for c in cands])
        s_abs = s0 + S[p_arr]                                      # (n, T)
        dlat, ddl, dd2 = _quintic(d0, dd0, d_arr[:, None], L_arr[:, None], S[p_arr])
        x, y, yaw_r = route.frenet_to_xy(s_abs, dlat)
        # heading: path yaw + atan(d')
        yaw = yaw_r + np.arctan(np.clip(ddl, -1.0, 1.0))
        moving = V[p_arr] > 0.05
        yaw = np.where(moving, yaw, e.yaw)

        # --- feasibility ----------------------------------------------------------------------
        bl, br = route.bounds(s_abs)
        nb = min(T, cfg.bounds_steps)   # later violations are resolved by replanning
        viol = np.maximum(np.maximum(dlat - bl, br - dlat), 0.0)[:, :nb]
        # bounds already keep a margin to the lane edge: small violations are penalised, not rejected
        in_bounds = viol.max(1) <= cfg.bound_tol
        kappa_path = route.curvature(s_abs)
        kappa = np.abs(kappa_path / np.maximum(1.0 - kappa_path * dlat, 0.3) + dd2)
        kin_ok = np.all((kappa <= 0.98 * p.max_curvature) | ~moving, axis=1)
        geo, clear = self._clearance(x, y, yaw, t, agents)          # (n,) each
        c_now = float(self._clearance(np.array([[e.x]]), np.array([[e.y]]), np.array([[e.yaw]]),
                                      np.zeros(1), agents)[0][0])
        # hard: geometric clearance; if already inside the margin, accept plans that do not get worse.
        # The head-on standoff (ahead_extra) is a strong cost, not a constraint (avoids dead-locks).
        # anti-deadlock: when frozen, progressively accept a little more (circles are conservative)
        self.stuck = self.stuck + cfg.dt if e.v < 0.1 else 0.0
        relax = min(0.4, 0.05 * max(0.0, self.stuck - 2.0))
        safe = geo >= min(cfg.min_clearance, c_now - 0.05 - relax)

        # --- cost ---------------------------------------------------------------------------------
        lane_half = 0.5 * float(np.interp(s0, route.s, route.lane_width))
        outside = np.clip(np.abs(dlat) - (lane_half - p.width / 2), 0.0, None) > 0.05
        J = cfg.w_d * d_arr ** 2 / 4.0
        J += cfg.w_opp * outside.sum(1) * cfg.dt
        J += cfg.w_v * ((v_des - V[p_arr].mean(1)) / max(v_des, 0.5)) ** 2 * 10.0
        J += cfg.w_len / L_arr
        J += cfg.w_clear * np.exp(-np.clip(clear, 0, None) / 0.8)
        J += cfg.w_standoff * np.clip(cfg.min_clearance - clear, 0, None) ** 2
        J += cfg.w_stop * stop_flags[p_arr]
        J += cfg.w_bound * (viol ** 2).sum(1) * cfg.dt
        if self.prev_d is not None:
            J += cfg.w_hyst * (d_arr - self.prev_d) ** 2
        J -= cfg.side_pref * np.sign(d_arr) * (np.abs(d_arr) > 0.5)
        feas = in_bounds & kin_ok & safe
        self.debug = dict(in_bounds=in_bounds, kin_ok=kin_ok, safe=safe, p_arr=p_arr, d_arr=d_arr, L_arr=L_arr,
                          clear=clear, kappa=kappa, V=V, J=J, stop=stop_flags)
        fallback = False
        if feas.any():
            J_f = np.where(feas, J, np.inf)
            i = int(np.argmin(J_f))
        else:
            fallback = True
            hard = np.where(p_arr == P - 1)[0]
            i = int(hard[np.argmax(geo[hard] + 0.01 * in_bounds[hard])])
        self.prev_d = d_arr[i]
        zref = np.stack([x[i], y[i], np.unwrap(yaw[i]), V[p_arr[i]]], 1)
        return zref, dict(fallback=fallback, d_target=float(d_arr[i]), clearance=float(clear[i]),
                          n_feasible=int(feas.sum()), n_cand=n)

    def _lead_speed(self, obs, agents):
        best, v = np.inf, None
        for a in agents:
            s, d = obs.route.project(a.x, a.y, window=40.0, s_hint=obs.s_ego + 20)
            ahead = s - obs.s_ego
            if 0 < ahead < 30 and abs(d - obs.d_ego) < 1.6 and ahead < best:
                _, _, yaw = obs.route.pose(s)
                v = a.vx * np.cos(yaw) + a.vy * np.sin(yaw)
                best = ahead
        return v

    def _clearance(self, x, y, yaw, t, agents):
        """-> (geometric clearance, standoff-adjusted clearance), each (n,)."""
        return trajectory_clearance(x, y, yaw, t, agents, self.ego_offs, self.ego_r,
                                    ahead_extra=self.cfg.ahead_extra, range_m=self.cfg.range_m, both=True)


class ExpertController(Controller):
    """Lattice expert + LTV-MPC tracker. Uses ground-truth agents (privileged)."""

    name = "expert"

    def __init__(self, params: VehicleParams, cfg: LatticeConfig = LatticeConfig()):
        self.p = params
        self.planner = LatticePlanner(params, cfg)
        self.mpc = LTVMPC(params)

    def reset(self, route: Path, v_des: float, ego: EgoState):
        self.planner.reset()
        self.u = np.zeros(2)

    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        zref, info = self.planner.plan(obs)
        N = self.mpc.N
        out = self.mpc.solve(obs.ego.z(), zref[:N], self.u, agents=obs.agents, route=obs.route)
        self.u = out["u"][0].copy()
        info.update(solver_time=out["solve_time"], status=out["status"])
        return Command(float(self.u[0]), float(self.u[1]), time.perf_counter() - t0, out["ok"],
                       zref[:, :2], info)
