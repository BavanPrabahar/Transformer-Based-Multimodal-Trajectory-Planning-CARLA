"""Vehicle parameters and the kinematic bicycle model.

Defaults are taken from the MPC_capsule controller
(`controllers/kinematic_mpc.py`, class
KinMPCPathFollower): CoG-referenced kinematic bicycle with slip angle beta,
L_F = 1.20 m, L_R = 0.90 m and the same actuator/rate limits.  `from_carla`
re-derives the geometry from the simulated vehicle so that every controller
uses a model that matches the plant it drives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np


@dataclass(frozen=True)
class VehicleParams:
    l_f: float = 1.20          # CoG -> front axle (m)
    l_r: float = 0.90          # CoG -> rear axle (m)
    length: float = 3.60       # body length (m)
    width: float = 1.80        # body width (m)
    cog_to_center: float = 0.0  # body-centre position ahead of the CoG (m)
    v_min: float = 0.0
    v_max: float = 15.0
    a_min: float = -3.0        # m/s^2
    a_max: float = 0.8         # m/s^2 (lowered for the 1.5 t vehicle in the lab code)
    df_min: float = -0.75      # rad
    df_max: float = 0.75
    a_dot_min: float = -1.5    # m/s^3
    a_dot_max: float = 1.5
    df_dot_min: float = -0.75  # rad/s
    df_dot_max: float = 0.75

    @property
    def wheelbase(self) -> float:
        return self.l_f + self.l_r

    @property
    def max_curvature(self) -> float:
        """Largest path curvature reachable at the steering limit (CoG bicycle)."""
        beta = np.arctan(self.l_r / self.wheelbase * np.tan(self.df_max))
        return float(np.sin(beta) / self.l_r)

    def ego_circles(self, n: int = 3):
        """Offsets (along heading, from CoG) and radius of n circles covering the body."""
        seg = self.length / n
        offs = self.cog_to_center + (np.arange(n) - (n - 1) / 2.0) * seg
        radius = float(np.hypot(self.width / 2.0, seg / 2.0))
        return offs, radius

    def to_dict(self):
        return asdict(self)

    def with_geometry(self, l_f, l_r, length, width, cog_to_center):
        return replace(self, l_f=float(l_f), l_r=float(l_r), length=float(length),
                       width=float(width), cog_to_center=float(cog_to_center))


def slip_angle(df, p: VehicleParams):
    return np.arctan(p.l_r / p.wheelbase * np.tan(df))


def bicycle_step(z, u, dt, p: VehicleParams):
    """One explicit-Euler step of the CoG kinematic bicycle (same as the lab MPC).

    z = [x, y, psi, v], u = [a, df]
    """
    x, y, psi, v = z
    a, df = u
    beta = slip_angle(df, p)
    return np.array([
        x + dt * v * np.cos(psi + beta),
        y + dt * v * np.sin(psi + beta),
        psi + dt * v / p.l_r * np.sin(beta),
        v + dt * a,
    ])


def rollout(z0, us, dt, p: VehicleParams):
    zs = [np.asarray(z0, dtype=float)]
    for u in us:
        zs.append(bicycle_step(zs[-1], u, dt, p))
    return np.stack(zs)
