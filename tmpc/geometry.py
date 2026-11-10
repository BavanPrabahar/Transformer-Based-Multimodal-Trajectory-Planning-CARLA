"""Geometry helpers: angles, frames, reference paths (Frenet), boxes and circles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_local(xy, x0, y0, yaw0):
    """World -> frame located at (x0, y0) with heading yaw0. xy: (..., 2)."""
    xy = np.asarray(xy, dtype=float)
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx = xy[..., 0] - x0
    dy = xy[..., 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def to_global(xy, x0, y0, yaw0):
    xy = np.asarray(xy, dtype=float)
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.stack([x0 + c * xy[..., 0] - s * xy[..., 1],
                     y0 + s * xy[..., 0] + c * xy[..., 1]], axis=-1)


def heading_and_speed(xy, dt, origin=(0.0, 0.0), yaw0=0.0, min_step=0.02):
    """Heading/speed along a time-parameterised xy trajectory.

    xy[0] is the first future point; `origin` is the current position.
    Headings at near-zero displacement are held from the previous value.
    """
    xy = np.asarray(xy, dtype=float)
    prev = np.vstack([np.asarray(origin, float)[None], xy[:-1]])
    d = xy - prev
    step = np.hypot(d[:, 0], d[:, 1])
    yaw = np.empty(len(xy))
    last = yaw0
    for i in range(len(xy)):
        if step[i] > min_step:
            last = np.arctan2(d[i, 1], d[i, 0])
        yaw[i] = last
    return np.unwrap(np.concatenate([[yaw0], yaw]))[1:], step / dt


class Path:
    """Arc-length parameterised 2-D polyline with drivable-area bounds.

    d_left / d_right are signed lateral limits for the *vehicle centre*
    (left positive), already shrunk by half the vehicle width + margin.
    """

    def __init__(self, xy, d_left=None, d_right=None, lane_width=None, speed_limit=None):
        xy = np.asarray(xy, dtype=float)
        keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-3])
        self.xy = xy[keep]
        seg = np.diff(self.xy, axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self.yaw = np.unwrap(np.concatenate([yaw, yaw[-1:]]))
        n = len(self.xy)
        self.d_left = np.full(n, 1.0) if d_left is None else np.asarray(d_left, float)[keep]
        self.d_right = np.full(n, -1.0) if d_right is None else np.asarray(d_right, float)[keep]
        self.lane_width = np.full(n, 3.5) if lane_width is None else np.asarray(lane_width, float)[keep]
        k = np.gradient(self.yaw, self.s, edge_order=1) if n > 2 else np.zeros(n)
        self.kappa = np.convolve(k, np.ones(5) / 5.0, mode="same")
        self._hint = 0

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def _interp(self, arr, s):
        return np.interp(s, self.s, arr)

    def pose(self, s):
        s = np.clip(s, 0.0, self.s[-1])
        x = self._interp(self.xy[:, 0], s)
        y = self._interp(self.xy[:, 1], s)
        yaw = self._interp(self.yaw, s)
        return x, y, yaw

    def bounds(self, s):
        s = np.clip(s, 0.0, self.s[-1])
        return self._interp(self.d_left, s), self._interp(self.d_right, s)

    def curvature(self, s):
        return self._interp(self.kappa, np.clip(s, 0.0, self.s[-1]))

    def frenet_to_xy(self, s, d):
        x, y, yaw = self.pose(s)
        return x - np.sin(yaw) * d, y + np.cos(yaw) * d, yaw

    def project(self, x, y, window: Optional[float] = None, s_hint: Optional[float] = None):
        """Closest point -> (s, d). Vectorised over all segments (paths are short)."""
        p = np.array([x, y], dtype=float)
        a = self.xy[:-1]
        b = self.xy[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist2 = ((proj - p) ** 2).sum(1)
        if window is not None and s_hint is not None:
            seg_s = self.s[:-1] + t * np.sqrt(L2)
            dist2 = np.where(np.abs(seg_s - s_hint) <= window, dist2, np.inf)
            if not np.isfinite(dist2).any():
                dist2 = ((proj - p) ** 2).sum(1)
        i = int(np.argmin(dist2))
        s = self.s[i] + t[i] * np.sqrt(L2[i])
        yaw = np.arctan2(ab[i, 1], ab[i, 0])
        rel = p - proj[i]
        d = -np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]
        return float(s), float(d)

    def project_many(self, xy, s_min=None, s_max=None):
        """Vectorised projection of many points onto the segments within [s_min, s_max]."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        lo = 0 if s_min is None else max(0, int(np.searchsorted(self.s, s_min)) - 1)
        hi = len(self.s) - 1 if s_max is None else min(len(self.s) - 1, int(np.searchsorted(self.s, s_max)) + 1)
        if hi <= lo:
            lo, hi = 0, len(self.s) - 1
        a = self.xy[lo:hi]
        ab = self.xy[lo + 1:hi + 1] - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        rel = xy[:, None, :] - a[None]
        t = np.clip((rel * ab[None]).sum(-1) / L2[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d2 = ((proj - xy[:, None, :]) ** 2).sum(-1)
        i = np.argmin(d2, axis=1)
        m = np.arange(len(xy))
        s = self.s[lo:hi][i] + t[m, i] * np.sqrt(L2[i])
        yaw = np.arctan2(ab[i, 1], ab[i, 0])
        r = xy - proj[m, i]
        d = -np.sin(yaw) * r[:, 0] + np.cos(yaw) * r[:, 1]
        return s, d

    def resample(self, s0, length, step):
        ss = s0 + np.arange(0.0, length + 1e-9, step)
        x, y, yaw = self.pose(ss)
        dl, dr = self.bounds(ss)
        return ss, np.stack([x, y], 1), yaw, dl, dr

    def to_dict(self):
        return dict(xy=self.xy, d_left=self.d_left, d_right=self.d_right, lane_width=self.lane_width)

    @classmethod
    def from_dict(cls, d):
        return cls(d["xy"], d.get("d_left"), d.get("d_right"), d.get("lane_width"))


# ---------------------------------------------------------------------------
# Agents and boxes
# ---------------------------------------------------------------------------
KIND_VEHICLE, KIND_WALKER, KIND_STATIC = 0, 1, 2
KIND_NAMES = {KIND_VEHICLE: "vehicle", KIND_WALKER: "walker", KIND_STATIC: "static"}


@dataclass
class Agent:
    id: int
    kind: int
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float

    @property
    def speed(self):
        return float(np.hypot(self.vx, self.vy))

    def as_array(self):
        return np.array([self.id, self.kind, self.x, self.y, self.yaw, self.vx, self.vy,
                         self.length, self.width], dtype=float)

    @classmethod
    def from_array(cls, a):
        return cls(int(a[0]), int(a[1]), *[float(v) for v in a[2:9]])

    def predict(self, times):
        """Constant-velocity prediction -> (T, 3) [x, y, yaw]."""
        times = np.asarray(times, float)
        return np.stack([self.x + self.vx * times, self.y + self.vy * times,
                         np.full_like(times, self.yaw)], 1)

    def circles(self, n=None):
        """Offsets along heading and radius of circles covering the footprint."""
        if n is None:
            n = 1 if self.kind == KIND_WALKER else max(1, int(np.ceil(1.5 * self.length / max(self.width, 0.5))))
        seg = self.length / n
        offs = (np.arange(n) - (n - 1) / 2.0) * seg
        return offs, float(np.hypot(self.width / 2.0, seg / 2.0))


def agents_to_array(agents: List[Agent], max_agents: int = 32):
    arr = np.zeros((max_agents, 9))
    arr[:, 0] = -1
    for i, a in enumerate(agents[:max_agents]):
        arr[i] = a.as_array()
    return arr


def agents_from_array(arr) -> List[Agent]:
    return [Agent.from_array(r) for r in arr if r[0] >= 0]


def box_corners(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)


def _sat_axes(c):
    e = np.diff(np.vstack([c, c[:1]]), axis=0)[:2]
    n = np.stack([-e[:, 1], e[:, 0]], 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def boxes_overlap(c1, c2) -> bool:
    for axis in np.vstack([_sat_axes(c1), _sat_axes(c2)]):
        p1, p2 = c1 @ axis, c2 @ axis
        if p1.max() < p2.min() or p2.max() < p1.min():
            return False
    return True


def box_distance(c1, c2) -> float:
    """Euclidean distance between two convex quads (0 when overlapping)."""
    if boxes_overlap(c1, c2):
        return 0.0
    best = np.inf
    for P, Q in ((c1, c2), (c2, c1)):
        a = Q
        b = np.roll(Q, -1, axis=0)
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        for p in P:
            t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
            d = np.hypot(*(a + t[:, None] * ab - p).T)
            best = min(best, float(d.min()))
    return best


def circle_clearance(ego_xy, ego_yaw, ego_offs, ego_r, obs_xy, obs_yaw, obs_offs, obs_r):
    """Min clearance between ego circles and obstacle circles at matched times.

    ego_xy (T,2), ego_yaw (T,), obs_xy (T,2), obs_yaw (T,) -> (T,) clearance (m)
    """
    ec = ego_xy[:, None, :] + ego_offs[None, :, None] * np.stack([np.cos(ego_yaw), np.sin(ego_yaw)], 1)[:, None, :]
    oc = obs_xy[:, None, :] + obs_offs[None, :, None] * np.stack([np.cos(obs_yaw), np.sin(obs_yaw)], 1)[:, None, :]
    d = np.linalg.norm(ec[:, :, None, :] - oc[:, None, :, :], axis=-1)
    return d.reshape(len(ego_xy), -1).min(1) - (ego_r + obs_r)
