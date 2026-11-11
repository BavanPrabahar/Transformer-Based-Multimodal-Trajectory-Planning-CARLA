"""Faithful port of the "MPC_capsule" controller stack.

Components reproduced from the MPC_capsule ROS package scripts:

* KinMPCPathFollower (controllers/kinematic_mpc.py): CasADi/IPOPT nonlinear
  kinematic MPC on the CoG bicycle with slip angle, hard input bounds, input-rate
  bounds relaxed by L1-penalised slacks, warm start, `max_cpu_time = 0.1 s`.
  The deployed tuning from mpc_cmd_pub.py is used: N = 10, DT = 0.4,
  Q = [300, 300, 100, 10], R = [10, 2000].
* GPSRefTrajectory.get_waypoints (gps_utils/ref_gps_traj.py): closest-point
  reference at fixed target speed with the 0.8 m/s^2 start ramp.
* Livox ROI guard + speed law (acs1_nav_test7_3.py): obstacles inside a corridor
  of half-width 1.3 m and length 25 m ahead of the bumper cap the target speed
  with v = 40 - 40 exp(-0.04 (d - 7)) km/h and trigger an emergency brake when
  d < 15 m.  On the vehicle this runs on raw Livox points; here it runs on the
  (noisy) object footprints that every evaluated stack receives.

Obstacle avoidance is *not* part of this MPC: the lab stack follows the route and
stops for anything in its corridor.  That is the behaviour being benchmarked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import casadi
import numpy as np

from ..geometry import Agent, Path, box_corners, wrap_angle
from ..types import Command, Controller, EgoState, Observation
from ..vehicle import VehicleParams


def fix_angle_reference(angle_ref, angle_init):
    diff = np.unwrap(np.hstack(([angle_init], angle_ref)) - angle_init)
    return diff[1:] + angle_init


class KinMPCPathFollower:
    """CasADi port of KinMPCPathFollower with optional obstacle/road constraints.

    With `n_obs_circles = 0` and `road_bounds = False` this is exactly the lab
    formulation.  The capsule NMPC baseline re-uses this builder with hard (slack
    relaxed) circle-separation and drivable-area constraints.
    """

    def __init__(self, params: VehicleParams, N=10, DT=0.4, Q=(300., 300., 100., 10.),
                 R=(10., 2000.), max_cpu_time=0.1, n_obs_circles=0, road_bounds=False,
                 n_ego_circles=3, obs_slack_weight=(1e3, 1e4), road_slack_weight=(1e2, 1e3),
                 d_safe=0.3, max_iter=None):
        self.p = params
        self.N, self.DT = int(N), float(DT)
        self.n_obs = int(n_obs_circles)
        self.road = bool(road_bounds)
        self.d_safe = float(d_safe)
        opti = casadi.Opti()
        self.opti = opti
        N = self.N

        self.u_prev = opti.parameter(2)
        self.z_curr = opti.parameter(4)
        self.z_ref = opti.parameter(N, 4)
        self.z_dv = opti.variable(N + 1, 4)
        self.u_dv = opti.variable(N, 2)
        self.sl_dv = opti.variable(N, 2)

        x, y, psi, v = (self.z_dv[:, i] for i in range(4))
        acc, df = self.u_dv[:, 0], self.u_dv[:, 1]
        p = params

        opti.subject_to(opti.bounded(p.v_min, v, p.v_max))
        for i in range(4):
            opti.subject_to(self.z_dv[0, i] == self.z_curr[i])
        for i in range(N):
            beta = casadi.atan(p.l_r / (p.l_f + p.l_r) * casadi.tan(df[i]))
            opti.subject_to(x[i + 1] == x[i] + self.DT * (v[i] * casadi.cos(psi[i] + beta)))
            opti.subject_to(y[i + 1] == y[i] + self.DT * (v[i] * casadi.sin(psi[i] + beta)))
            opti.subject_to(psi[i + 1] == psi[i] + self.DT * (v[i] / p.l_r * casadi.sin(beta)))
            opti.subject_to(v[i + 1] == v[i] + self.DT * acc[i])
        opti.subject_to(opti.bounded(p.a_min, acc, p.a_max))
        opti.subject_to(opti.bounded(p.df_min, df, p.df_max))
        sl_a, sl_d = self.sl_dv[:, 0], self.sl_dv[:, 1]
        opti.subject_to(opti.bounded(p.a_dot_min * self.DT - sl_a[0], acc[0] - self.u_prev[0],
                                     p.a_dot_max * self.DT + sl_a[0]))
        opti.subject_to(opti.bounded(p.df_dot_min * self.DT - sl_d[0], df[0] - self.u_prev[1],
                                     p.df_dot_max * self.DT + sl_d[0]))
        for i in range(N - 1):
            opti.subject_to(opti.bounded(p.a_dot_min * self.DT - sl_a[i + 1], acc[i + 1] - acc[i],
                                         p.a_dot_max * self.DT + sl_a[i + 1]))
            opti.subject_to(opti.bounded(p.df_dot_min * self.DT - sl_d[i + 1], df[i + 1] - df[i],
                                         p.df_dot_max * self.DT + sl_d[i + 1]))
        opti.subject_to(0 <= sl_d)
        opti.subject_to(0 <= sl_a)

        Qm, Rm = casadi.diag(list(Q)), casadi.diag(list(R))
        cost = 0
        for i in range(N):
            e = self.z_dv[i + 1, :] - self.z_ref[i, :]
            cost += casadi.mtimes(e, casadi.mtimes(Qm, e.T))
        for i in range(N - 1):
            du = self.u_dv[i + 1, :] - self.u_dv[i, :]
            cost += casadi.mtimes(du, casadi.mtimes(Rm, du.T))
        cost += casadi.sum1(sl_d) + casadi.sum1(sl_a)

        # ---- optional: obstacle circles (capsule NMPC) ---------------------------
        if self.n_obs > 0:
            offs, r_e = params.ego_circles(n_ego_circles)
            self.ego_offs, self.ego_r = offs, r_e
            # obstacle circle centres + radii, per horizon step: (N, n_obs) each
            self.obs_x = opti.parameter(N, self.n_obs)
            self.obs_y = opti.parameter(N, self.n_obs)
            self.obs_r = opti.parameter(N, self.n_obs)
            self.sl_obs = opti.variable(N, 1)
            opti.subject_to(self.sl_obs >= 0)
            for k in range(N):
                for c in offs:
                    ex = x[k + 1] + c * casadi.cos(psi[k + 1])
                    ey = y[k + 1] + c * casadi.sin(psi[k + 1])
                    for j in range(self.n_obs):
                        dmin = r_e + self.obs_r[k, j] + self.d_safe
                        opti.subject_to((ex - self.obs_x[k, j]) ** 2 + (ey - self.obs_y[k, j]) ** 2
                                        >= dmin ** 2 - self.sl_obs[k])
            cost += obs_slack_weight[0] * casadi.sum1(self.sl_obs) + \
                obs_slack_weight[1] * casadi.sumsqr(self.sl_obs)
        # ---- optional: drivable-area bounds (linear in the path frame) ----------
        if self.road:
            # centre point c_k, unit left normal n_k, bounds [lo_k, hi_k] of n_k.(p - c_k)
            self.road_c = opti.parameter(N, 2)
            self.road_n = opti.parameter(N, 2)
            self.road_lo = opti.parameter(N, 1)
            self.road_hi = opti.parameter(N, 1)
            self.sl_road = opti.variable(N, 1)
            opti.subject_to(self.sl_road >= 0)
            for k in range(N):
                lat = self.road_n[k, 0] * (x[k + 1] - self.road_c[k, 0]) + \
                    self.road_n[k, 1] * (y[k + 1] - self.road_c[k, 1])
                opti.subject_to(lat <= self.road_hi[k] + self.sl_road[k])
                opti.subject_to(lat >= self.road_lo[k] - self.sl_road[k])
            cost += road_slack_weight[0] * casadi.sum1(self.sl_road) + \
                road_slack_weight[1] * casadi.sumsqr(self.sl_road)

        opti.minimize(cost)
        s_opts = {"max_cpu_time": float(max_cpu_time), "print_level": 0}
        if max_iter is not None:
            s_opts["max_iter"] = int(max_iter)
        opti.solver("ipopt", {"expand": True, "print_time": False}, s_opts)

        # default parameter values so a first solve is well defined
        self.set_state(0.0, 0.0, 0.0, 1.0)
        self.set_reference(np.stack([self.DT * (np.arange(N) + 1), np.zeros(N), np.zeros(N), np.ones(N)], 1))
        self.set_prev_input(0.0, 0.0)
        if self.n_obs > 0:
            self.clear_obstacles()
        if self.road:
            self.set_road(np.zeros((N, 2)), np.tile([0.0, 1.0], (N, 1)), -10 * np.ones(N), 10 * np.ones(N))
        self._ws = None

    # -- parameter setters ---------------------------------------------------------
    def set_state(self, x, y, psi, v):
        self.opti.set_value(self.z_curr, [x, y, psi, v])

    def set_reference(self, zref):
        self.opti.set_value(self.z_ref, np.asarray(zref, float))

    def set_prev_input(self, a, df):
        self.opti.set_value(self.u_prev, [a, df])

    def clear_obstacles(self):
        self.opti.set_value(self.obs_x, 1e4 * np.ones((self.N, self.n_obs)))
        self.opti.set_value(self.obs_y, 1e4 * np.ones((self.N, self.n_obs)))
        self.opti.set_value(self.obs_r, 0.1 * np.ones((self.N, self.n_obs)))

    def set_obstacles(self, ox, oy, orad):
        self.opti.set_value(self.obs_x, ox)
        self.opti.set_value(self.obs_y, oy)
        self.opti.set_value(self.obs_r, orad)

    def set_road(self, c, n, lo, hi):
        self.opti.set_value(self.road_c, c)
        self.opti.set_value(self.road_n, n)
        self.opti.set_value(self.road_lo, np.asarray(lo).reshape(-1, 1))
        self.opti.set_value(self.road_hi, np.asarray(hi).reshape(-1, 1))

    def warm_start(self, z_ws, u_ws, sl_ws):
        self.opti.set_initial(self.z_dv, z_ws)
        self.opti.set_initial(self.u_dv, u_ws)
        self.opti.set_initial(self.sl_dv, sl_ws)

    def solve(self):
        st = time.perf_counter()
        try:
            sol = self.opti.solve()
            get, ok = sol.value, True
        except RuntimeError:
            get, ok = self.opti.debug.value, False
        u = np.atleast_2d(get(self.u_dv))
        z = np.atleast_2d(get(self.z_dv))
        sl = np.atleast_2d(get(self.sl_dv))
        out = dict(u_control=u[0, :], optimal=ok, solve_time=time.perf_counter() - st,
                   u_mpc=u, z_mpc=z, sl_mpc=sl)
        if self.n_obs > 0:
            out["sl_obs"] = np.asarray(get(self.sl_obs)).ravel()
        return out


# ---------------------------------------------------------------------------
# Reference generation (port of GPSRefTrajectory.get_waypoints)
# ---------------------------------------------------------------------------
def lab_reference(path: Path, x, y, psi, v_target, N, DT, v_init=None, a_ramp=0.8, spacing=0.5):
    """Fixed-speed reference from the closest route point, with the start ramp."""
    s_closest, _ = path.project(x, y)
    # emulate a recorded waypoint list: closest *waypoint* (not the projected point)
    s_closest = np.round(s_closest / spacing) * spacing
    h = np.arange(1, N + 1)
    unramped = s_closest + h * DT * v_target
    fit = unramped
    if v_init is not None and a_ramp and v_init < v_target:
        t_ramp = (v_target - v_init) / a_ramp
        t = h * DT
        ramp = np.where(t <= t_ramp, v_init * t + 0.5 * a_ramp * t * t,
                        v_init * t_ramp + 0.5 * a_ramp * t_ramp ** 2 + v_target * (t - t_ramp))
        fit = s_closest + ramp
    s_end = path.length
    fit_c = np.clip(fit, 0.0, s_end)
    unr_c = np.clip(unramped, 0.0, s_end)
    xr, yr, psir = path.pose(fit_c)
    psir = fix_angle_reference(psir, psi)
    v_un = np.diff(unr_c) / DT
    v_ref = np.append(v_un, v_un[-1]) if len(v_un) else np.array([v_target])
    stop = bool(unr_c[min(3, N - 1)] >= s_end - 1e-6)
    return np.stack([xr, yr, psir, v_ref], 1), stop


# ---------------------------------------------------------------------------
# Livox ROI guard (corridor speed governor)
# ---------------------------------------------------------------------------
@dataclass
class GovernorConfig:
    half_width: float = 1.3       # LIVOX_HALF_WIDTH
    lookahead: float = 25.0       # LIVOX_LOOKAHEAD
    stop_distance: float = 15.0   # LIVOX_STOP_DISTANCE
    confirm_frames: int = 2
    clear_frames: int = 5


def corridor_distance(path: Path, s_ego: float, front_offset: float, agents: List[Agent],
                      half_width: float, lookahead: float, sample=0.25) -> float:
    """Distance from the front bumper, along the route, to the nearest footprint point
    lying inside the corridor. Returns +inf when the corridor is clear."""
    s_bumper = s_ego + front_offset
    ex, ey, _ = path.pose(s_ego)
    pts = []
    for a in agents:
        if np.hypot(ex - a.x, ey - a.y) > lookahead + 10 + a.length:
            continue
        pts.append(footprint_points(a, sample))
    if not pts:
        return np.inf
    pts = np.vstack(pts)
    s, d = path.project_many(pts, s_bumper - 5.0, s_bumper + lookahead + 5.0)
    ahead = s - s_bumper
    inside = (ahead > 0.0) & (ahead < lookahead) & (np.abs(d) < half_width)
    return float(ahead[inside].min()) if inside.any() else np.inf


def footprint_points(a: Agent, sample=0.25):
    """Points on the boundary (and centre) of an agent's footprint."""
    c = box_corners(a.x, a.y, a.yaw, a.length, a.width)
    pts = [c, np.array([[a.x, a.y]])]
    for i in range(4):
        p0, p1 = c[i], c[(i + 1) % 4]
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / sample)))
        t = np.linspace(0, 1, n)[:, None]
        pts.append(p0 + t * (p1 - p0))
    return np.vstack(pts)


class SpeedGovernor:
    def __init__(self, cfg: GovernorConfig = GovernorConfig()):
        self.cfg = cfg
        self.seen = 0
        self.free = 0
        self.active = False
        self.distance = np.inf

    def reset(self):
        self.seen = self.free = 0
        self.active = False
        self.distance = np.inf

    def update(self, d):
        """Debounce like LivoxRoiGuard (confirm/clear frames)."""
        if np.isfinite(d):
            self.seen += 1
            self.free = 0
            if self.seen >= self.cfg.confirm_frames:
                self.active = True
                self.distance = d
        else:
            self.free += 1
            self.seen = 0
            if self.free >= self.cfg.clear_frames:
                self.active = False
                self.distance = np.inf
        return self.active, self.distance

    @staticmethod
    def speed_limit_kmh(d):
        return max(0.0, 40.0 - 40.0 * np.exp(-0.04 * (d - 7.0)))


class LabStackController(Controller):
    """Lab MPC path follower + Livox-ROI speed governor + emergency brake."""

    name = "lab_mpc"

    def __init__(self, params: VehicleParams, N=10, DT=0.4, Q=(300., 300., 100., 10.),
                 R=(10., 2000.), max_cpu_time=0.1, a_ramp=0.8, governor=GovernorConfig()):
        self.p = params
        self.mpc = KinMPCPathFollower(params, N=N, DT=DT, Q=Q, R=R, max_cpu_time=max_cpu_time)
        self.gov = SpeedGovernor(governor)
        self.a_ramp = a_ramp
        self.front = params.cog_to_center + params.length / 2.0

    def reset(self, route: Path, v_des: float, ego: EgoState):
        self.route, self.v_des = route, v_des
        self.gov.reset()
        self.acc_prev, self.df_prev = 0.0, 0.0
        self.ws = None

    def step(self, obs: Observation) -> Command:
        t0 = time.perf_counter()
        e = obs.ego
        d_raw = corridor_distance(self.route, obs.s_ego, self.front, obs.agents,
                                  self.gov.cfg.half_width, self.gov.cfg.lookahead)
        active, d = self.gov.update(d_raw)
        v_kmh = self.v_des * 3.6
        if active:
            v_kmh = min(v_kmh, self.gov.speed_limit_kmh(d))
        v_target = v_kmh / 3.6
        emergency = active and d < self.gov.cfg.stop_distance

        zref, stop = lab_reference(self.route, e.x, e.y, e.yaw, max(v_target, 1e-3),
                                   self.mpc.N, self.mpc.DT, v_init=e.v, a_ramp=self.a_ramp)
        if stop:
            # end of route: lab node publishes acc = -1, steer = 0
            return Command(-1.0, 0.0, time.perf_counter() - t0, True, None,
                           dict(stop=True, emergency=False, corridor=d_raw))
        m = self.mpc
        m.set_state(e.x, e.y, e.yaw, e.v)
        m.set_reference(zref)
        m.set_prev_input(self.acc_prev, self.df_prev)
        if self.ws is not None:
            m.warm_start(*self.ws)
        sol = m.solve()
        self.ws = (sol["z_mpc"], sol["u_mpc"], sol["sl_mpc"])
        a, df = float(sol["u_control"][0]), float(sol["u_control"][1])
        self.acc_prev, self.df_prev = a, df
        if emergency:
            a = self.p.a_min   # throttle cut + full brake pulse on the vehicle
        return Command(a, df, time.perf_counter() - t0, bool(sol["optimal"]), sol["z_mpc"][:, :2],
                       dict(emergency=emergency, corridor=d_raw, v_target=v_target,
                            solver_time=sol["solve_time"]))
