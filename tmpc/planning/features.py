"""Scene encoding for the trajectory transformer (ego frame, normalised).

The same function is used offline (dataset building from recorded episodes) and
online (HybridController), so train/test features are identical by construction.

Tokens
------
history  (H, 5)   past ego poses: x, y, cos(dyaw), sin(dyaw), v
agents   (M, 11)  x, y, cos, sin, vx, vy, length, width, one-hot(vehicle, walker, static)
route    (R, 6)   route centreline ahead: x, y, cos, sin, d_left, d_right
context  (5,)     v, v_des, last accel, last steer, remaining route length
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..geometry import KIND_STATIC, KIND_VEHICLE, KIND_WALKER, Path, to_local, wrap_angle


@dataclass(frozen=True)
class FeatureConfig:
    hist_len: int = 11
    max_agents: int = 16
    agent_range: float = 40.0
    route_points: int = 25
    route_step: float = 2.0
    future_steps: int = 30
    dt: float = 0.1
    pos_scale: float = 20.0
    vel_scale: float = 10.0

    def to_dict(self):
        return asdict(self)


def build_features(ego_pose, ego_v, ego_a, ego_df, history, agents_arr, route: Path, s_ego, v_des,
                   cfg: FeatureConfig = FeatureConfig()):
    """ego_pose (x, y, yaw); history (H', 4) world [x, y, yaw, v] oldest first (last == current);
    agents_arr (N, 9) [id, kind, x, y, yaw, vx, vy, length, width] (rows with id < 0 ignored)."""
    x0, y0, yaw0 = ego_pose
    ps, vs = cfg.pos_scale, cfg.vel_scale
    H = cfg.hist_len
    hist = np.asarray(history, float)
    if len(hist) < H:
        hist = np.vstack([np.repeat(hist[:1], H - len(hist), 0), hist])
    hist = hist[-H:]
    hl = to_local(hist[:, :2], x0, y0, yaw0)
    dyaw = wrap_angle(hist[:, 2] - yaw0)
    f_hist = np.column_stack([hl / ps, np.cos(dyaw), np.sin(dyaw), hist[:, 3] / vs]).astype(np.float32)

    M = cfg.max_agents
    f_ag = np.zeros((M, 11), np.float32)
    m_ag = np.zeros(M, bool)
    if agents_arr is not None and len(agents_arr):
        a = np.asarray(agents_arr, float)
        a = a[a[:, 0] >= 0]
        if len(a):
            dist = np.hypot(a[:, 2] - x0, a[:, 3] - y0)
            keep = dist <= cfg.agent_range
            a, dist = a[keep], dist[keep]
            a = a[np.argsort(dist)][:M]
            n = len(a)
            if n:
                al = to_local(a[:, 2:4], x0, y0, yaw0)
                ry = wrap_angle(a[:, 4] - yaw0)
                c, s = np.cos(-yaw0), np.sin(-yaw0)
                vx = c * a[:, 5] - s * a[:, 6]
                vy = s * a[:, 5] + c * a[:, 6]
                kind = a[:, 1].astype(int)
                f_ag[:n] = np.column_stack([
                    al / ps, np.cos(ry), np.sin(ry), vx / vs, vy / vs, a[:, 7] / 5.0, a[:, 8] / 5.0,
                    kind == KIND_VEHICLE, kind == KIND_WALKER, kind == KIND_STATIC])
                m_ag[:n] = True

    R = cfg.route_points
    ss = s_ego + cfg.route_step * np.arange(R)
    m_rt = ss <= route.length + 1e-6
    ssc = np.minimum(ss, route.length)
    rx, ry_, ryaw = route.pose(ssc)
    dl, dr = route.bounds(ssc)
    rl = to_local(np.stack([rx, ry_], 1), x0, y0, yaw0)
    rdy = wrap_angle(ryaw - yaw0)
    f_rt = np.column_stack([rl / ps, np.cos(rdy), np.sin(rdy), dl / 5.0, dr / 5.0]).astype(np.float32)

    f_ctx = np.array([ego_v / vs, v_des / vs, ego_a / 3.0, ego_df / 0.75,
                      min(route.length - s_ego, 100.0) / 50.0], np.float32)
    return dict(hist=f_hist, agents=f_ag, agents_mask=m_ag, route=f_rt, route_mask=m_rt, ctx=f_ctx)


def features_from_obs(obs, cfg: FeatureConfig = FeatureConfig()):
    from ..geometry import agents_to_array
    e = obs.ego
    return build_features((e.x, e.y, e.yaw), e.v, e.a, e.df, obs.history, agents_to_array(obs.agents, 64),
                          obs.route, obs.s_ego, obs.v_des, cfg)


def collate(feats_list):
    return {k: np.stack([f[k] for f in feats_list]) for k in feats_list[0]}
