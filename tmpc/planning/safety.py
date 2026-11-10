"""Shared trajectory safety checks (used by the expert and the hybrid selector)."""
from __future__ import annotations

from typing import List

import numpy as np

from ..geometry import Agent, KIND_WALKER


def obstacle_circles(agents: List[Agent], t, center=None, range_m=None):
    """Constant-velocity circle centres over times t -> ox, oy (T, C), radii (C,)."""
    ox, oy, orad = [], [], []
    for a in agents:
        if center is not None and range_m is not None and np.hypot(a.x - center[0], a.y - center[1]) > range_m:
            continue
        offs, r = a.circles(1 if a.kind == KIND_WALKER else None)
        pred = a.predict(t)
        for c in offs:
            ox.append(pred[:, 0] + c * np.cos(pred[:, 2]))
            oy.append(pred[:, 1] + c * np.sin(pred[:, 2]))
            orad.append(r)
    if not ox:
        return None
    return np.array(ox).T, np.array(oy).T, np.array(orad)


def trajectory_clearance(x, y, yaw, t, agents: List[Agent], ego_offs, ego_r, ahead_extra=1.5,
                         range_m=45.0, per_step=False, both=False):
    """Effective circle clearance of candidate trajectories.

    x, y, yaw: (n, T) world poses at times t (T,).  Clearance to obstacle circles
    that lie *ahead* of the ego is reduced by ahead_extra * cos^2(bearing), so the
    plan keeps a longer standoff head-on (~ahead_extra + margin) than alongside.
    Returns (n,) min clearance, or (n, T) when per_step; with both=True returns
    (geometric clearance, standoff-adjusted clearance).
    """
    n, T = x.shape
    circ = obstacle_circles(agents, t, center=(x[0, 0], y[0, 0]), range_m=range_m)
    if circ is None:
        full = np.full((n, T) if per_step else n, 10.0)
        return (full, full.copy()) if both else full
    ox, oy, orad = circ
    c, s = np.cos(yaw), np.sin(yaw)
    ex = x[:, :, None] + ego_offs[None, None, :] * c[:, :, None]
    ey = y[:, :, None] + ego_offs[None, None, :] * s[:, :, None]
    dx = ox[None, :, None, :] - ex[:, :, :, None]           # (n, T, E, C) ego -> obstacle
    dy = oy[None, :, None, :] - ey[:, :, :, None]
    dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
    along = (dx * c[:, :, None, None] + dy * s[:, :, None, None]) / dist
    geo = dist - (ego_r + orad[None, None, None, :])
    eff = geo - ahead_extra * np.clip(along, 0.0, 1.0) ** 2
    m = eff.reshape(n, T, -1).min(-1)
    if both:
        g = geo.reshape(n, T, -1).min(-1)
        return (g, m) if per_step else (g.min(1), m.min(1))
    return m if per_step else m.min(1)
