"""Feasibility & cost selector for the transformer's candidate trajectories.

This implements the "Feasibility & Cost Selector" box of the original design,
which the original code skipped (it always took mode 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..controllers.ltv_mpc import reference_from_xy
from ..geometry import Agent
from ..types import Observation
from ..vehicle import VehicleParams
from .safety import trajectory_clearance


@dataclass
class SelectorConfig:
    dt: float = 0.1
    min_clearance: float = 0.3
    ahead_extra: float = 1.0
    bound_tol: float = 0.4
    bounds_steps: int = 25
    w_prob: float = 1.0
    w_clear: float = 2.0
    w_bound: float = 40.0
    w_progress: float = 1.5
    w_kin: float = 5.0
    w_standoff: float = 15.0
    w_centre: float = 0.5        # prefer candidates near the lane centre (per m^2)
    slow_factors: tuple = (0.75, 0.5, 0.3)


def retime(path_xy, origin, s_new):
    """Positions at arc lengths s_new along the polyline origin -> path_xy."""
    pts = np.vstack([np.asarray(origin)[None], path_xy])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-3:
        return np.repeat(pts[:1], len(s_new), 0)
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.stack([x, y], 1)


class CandidateSelector:
    def __init__(self, params: VehicleParams, cfg: SelectorConfig = SelectorConfig()):
        self.p = params
        self.cfg = cfg
        self.ego_offs, self.ego_r = params.ego_circles(3)

    def _evaluate(self, refs, obs: Observation):
        """refs (K, T, 4) world z-references -> per-candidate feasibility terms."""
        cfg, p = self.cfg, self.p
        K, T, _ = refs.shape
        t = cfg.dt * (np.arange(T) + 1)
        geo, clear = trajectory_clearance(refs[:, :, 0], refs[:, :, 1], refs[:, :, 2], t, obs.agents,
                                          self.ego_offs, self.ego_r, ahead_extra=cfg.ahead_extra, both=True)
        nb = min(T, cfg.bounds_steps)
        s_all, d_all = obs.route.project_many(refs[:, :nb, :2].reshape(-1, 2), obs.s_ego - 5, obs.s_ego + 60)
        dl, dr = obs.route.bounds(s_all)
        viol = np.maximum(np.maximum(d_all - dl, dr - d_all), 0.0).reshape(K, nb)
        # kinematics: curvature from heading change per metre, accelerations from speed
        ds = np.maximum(refs[:, :, 3] * cfg.dt, 1e-3)
        dyaw = np.abs(np.diff(np.concatenate([np.full((K, 1), obs.ego.yaw), refs[:, :, 2]], 1), axis=1))
        kappa = np.where(refs[:, :, 3] > 0.5, dyaw / ds, 0.0)
        # accelerations from a 0.5 s-smoothed speed profile (learned outputs carry cm-level jitter;
        # the LTV-MPC enforces the actual input limits)
        vs = np.concatenate([np.full((K, 1), obs.ego.v), refs[:, :, 3]], 1)
        ker = np.ones(5) / 5
        vs = np.stack([np.convolve(np.pad(v, 2, mode="edge"), ker, mode="valid") for v in vs])
        acc = np.diff(vs, axis=1) / cfg.dt
        kin_excess = np.maximum(kappa - 1.2 * p.max_curvature, 0).max(1) + \
            np.maximum(acc - (p.a_max + 1.2), 0).max(1) + np.maximum((p.a_min - 1.5) - acc, 0).max(1)
        progress = np.hypot(refs[:, -1, 0] - obs.ego.x, refs[:, -1, 1] - obs.ego.y)
        self._centre = (d_all.reshape(K, nb) ** 2).mean(1)
        return geo, clear, viol, kin_excess, progress

    def select(self, cands_world, logits, obs: Observation, relax: float = 0.0) -> Dict:
        """relax: extra clearance decrease tolerated when the ego is dead-locked (same rule as the expert)."""
        cfg = self.cfg
        e = obs.ego
        K, T, _ = cands_world.shape
        refs = np.stack([reference_from_xy(c, cfg.dt, (e.x, e.y), e.yaw, e.v) for c in cands_world])
        geo, clear, viol, kin, progress = self._evaluate(refs, obs)
        c_now = trajectory_clearance(np.array([[e.x]]), np.array([[e.y]]), np.array([[e.yaw]]), np.zeros(1),
                                     obs.agents, self.ego_offs, self.ego_r, ahead_extra=0.0)[0]
        logp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        feas = (geo >= min(cfg.min_clearance, c_now - 0.05 - relax)) & (viol.max(1) <= cfg.bound_tol) & (kin < 0.5)
        self.debug = dict(geo=geo, viol=viol.max(1), kin=kin, c_now=c_now, relax=relax, feas=feas)
        v_scale = max(obs.v_des * T * cfg.dt, 1.0)
        J = -cfg.w_prob * logp + cfg.w_clear * np.exp(-np.clip(clear, 0, None) / 0.8) + \
            cfg.w_bound * (viol ** 2).sum(1) * cfg.dt - cfg.w_progress * progress / v_scale + cfg.w_kin * kin + \
            cfg.w_standoff * np.clip(cfg.min_clearance - clear, 0, None) ** 2 + cfg.w_centre * self._centre
        if feas.any():
            i = int(np.argmin(np.where(feas, J, np.inf)))
            return dict(zref=refs[i], mode=i, fallback=0, clearance=float(geo[i]), n_feasible=int(feas.sum()))
        # fallback 1: slow down along the most likely path
        order = np.argsort(-logits)
        tt = cfg.dt * (np.arange(T) + 1)
        for f in cfg.slow_factors:
            for i in order[:3]:
                seg = np.hypot(*np.diff(np.vstack([[e.x, e.y], cands_world[i]]), axis=0).T)
                s_orig = np.cumsum(seg)
                xy = retime(cands_world[i], (e.x, e.y), f * s_orig)
                r = reference_from_xy(xy, cfg.dt, (e.x, e.y), e.yaw, e.v)[None]
                g, c, v_, k_, _ = self._evaluate(r, obs)
                if g[0] >= min(cfg.min_clearance, c_now - 0.05 - relax) and v_.max() <= cfg.bound_tol:
                    return dict(zref=r[0], mode=int(i), fallback=1, clearance=float(g[0]), n_feasible=0)
        # fallback 2: brake to a stop along the most likely path
        i = int(order[0])
        v0 = max(e.v, 0.0)
        a = 3.0
        s_stop = np.where(tt < v0 / a, v0 * tt - 0.5 * a * tt ** 2, v0 ** 2 / (2 * a))
        xy = retime(cands_world[i], (e.x, e.y), s_stop)
        zref = reference_from_xy(xy, cfg.dt, (e.x, e.y), e.yaw, e.v)
        zref[:, 3] = np.maximum(v0 - a * tt, 0.0)
        return dict(zref=zref, mode=i, fallback=2, clearance=float(geo[i]), n_feasible=0)
