"""CARLA-free 2-D simulator for unit tests and quick tuning.

Plant: CoG kinematic bicycle with first-order actuator lags and a steering
slew limit.  Agents move with constant velocity (static ones have v = 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..geometry import Agent, Path, box_corners, boxes_overlap, box_distance
from ..types import Command, Controller, EgoState, Observation
from ..vehicle import VehicleParams, bicycle_step


def straight_road(length=120.0, lane_width=3.5, two_way=True, margin=0.2, ego_width=1.8, step=0.5):
    xs = np.arange(0.0, length + 1e-9, step)
    xy = np.stack([xs, np.zeros_like(xs)], 1)
    half = ego_width / 2 + margin
    d_left = (1.5 * lane_width if two_way else 0.5 * lane_width) - half
    d_right = -0.5 * lane_width + half
    return Path(xy, np.full(len(xs), d_left), np.full(len(xs), d_right), np.full(len(xs), lane_width))


def curved_road(radius=40.0, angle=np.pi / 2, lead=30.0, lane_width=3.5, ego_width=1.8, step=0.5):
    pts = [np.array([x, 0.0]) for x in np.arange(0.0, lead, step)]
    for th in np.arange(0.0, angle, step / radius):
        pts.append(np.array([lead + radius * np.sin(th), radius - radius * np.cos(th)]))
    end = pts[-1]
    yaw = angle
    for d in np.arange(step, lead, step):
        pts.append(end + d * np.array([np.cos(yaw), np.sin(yaw)]))
    xy = np.array(pts)
    half = ego_width / 2 + 0.2
    n = len(xy)
    return Path(xy, np.full(n, 1.5 * lane_width - half), np.full(n, -0.5 * lane_width + half), np.full(n, lane_width))


@dataclass
class KinematicSim:
    route: Path
    agents: List[Agent]
    params: VehicleParams = field(default_factory=VehicleParams)
    v_des: float = 4.0
    dt: float = 0.05
    control_every: int = 2
    tau_a: float = 0.3          # longitudinal actuator lag (s)
    tau_df: float = 0.15        # steering lag (s)
    max_df_rate: float = 0.75   # rad/s steering slew
    perception: Optional[Callable] = None
    history_len: int = 11

    def __post_init__(self):
        x, y, yaw = self.route.pose(0.0)
        self.z = np.array([x, y, yaw, 0.0])
        self.a_act = 0.0
        self.df_act = 0.0
        self.t = 0.0
        self.hist = [self.z.copy()] * self.history_len

    def ego(self):
        return EgoState(*self.z, a=self.a_act, df=self.df_act)

    def agent_states(self):
        out = []
        for a in self.agents:
            out.append(Agent(a.id, a.kind, a.x + a.vx * self.t, a.y + a.vy * self.t, a.yaw, a.vx, a.vy,
                             a.length, a.width))
        return out

    def observe(self) -> Observation:
        s, d = self.route.project(self.z[0], self.z[1])
        agents = self.agent_states()
        if self.perception is not None:
            agents = self.perception(agents)
        return Observation(self.t, self.ego(), agents, self.route, self.v_des, np.array(self.hist), s, d)

    def step(self, cmd: Command):
        for _ in range(self.control_every):
            self.a_act += (cmd.a - self.a_act) * min(1.0, self.dt / self.tau_a)
            target = self.df_act + (cmd.df - self.df_act) * min(1.0, self.dt / self.tau_df)
            self.df_act += np.clip(target - self.df_act, -self.max_df_rate * self.dt, self.max_df_rate * self.dt)
            z = bicycle_step(self.z, [self.a_act, self.df_act], self.dt, self.params)
            z[3] = max(z[3], 0.0)
            self.z = z
            self.t += self.dt
        self.hist = self.hist[1:] + [self.z.copy()]

    def ego_box(self):
        p = self.params
        cx = self.z[0] + p.cog_to_center * np.cos(self.z[2])
        cy = self.z[1] + p.cog_to_center * np.sin(self.z[2])
        return box_corners(cx, cy, self.z[2], p.length, p.width)

    def collision(self):
        eb = self.ego_box()
        return any(boxes_overlap(eb, box_corners(a.x, a.y, a.yaw, a.length, a.width)) for a in self.agent_states())

    def min_distance(self):
        eb = self.ego_box()
        ds = [box_distance(eb, box_corners(a.x, a.y, a.yaw, a.length, a.width)) for a in self.agent_states()]
        return min(ds) if ds else np.inf


def run_episode(sim: KinematicSim, ctrl: Controller, t_max=60.0, goal_margin=3.0):
    ctrl.reset(sim.route, sim.v_des, sim.ego())
    log = dict(t=[], z=[], cmd=[], solve=[], ok=[], dist=[])
    status = "timeout"
    while sim.t < t_max:
        obs = sim.observe()
        cmd = ctrl.step(obs)
        sim.step(cmd)
        log["t"].append(sim.t)
        log["z"].append(sim.z.copy())
        log["cmd"].append([cmd.a, cmd.df])
        log["solve"].append(cmd.solve_time)
        log["ok"].append(cmd.ok)
        log["dist"].append(sim.min_distance())
        if sim.collision():
            status = "collision"
            break
        if obs.s_ego >= sim.route.length - goal_margin:
            status = "success"
            break
    for k in log:
        log[k] = np.array(log[k])
    log["status"] = status
    return log
