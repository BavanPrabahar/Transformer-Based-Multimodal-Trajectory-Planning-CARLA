"""Convex linear-time-varying tracking MPC solved as a sparse QP with OSQP.

The kinematic bicycle (CoG, slip angle; same model as the lab MPC) is linearised
along the reference trajectory, so the problem is a strictly convex QP:

    min  sum ||z_k - z_ref,k||_Q^2 + ||u_k||_Ru^2 + ||u_k - u_{k-1}||_Rd^2 + slack penalties
    s.t. z_{k+1} = A_k z_k + B_k u_k + c_k                        (linearised dynamics)
         u_min <= u_k <= u_max                                    (hard)
         |u_k - u_{k-1}| <= u_dot_max dt + s_rate                 (L1-relaxed, as in the lab MPC)
         n_kj . (c_ego,k - o_kj) >= r_e + r_j + d_safe - s_obs,k   (separating half-planes)
         d_right,k - s_road <= lateral offset_k <= d_left,k + s_road

The half-planes are built at the reference: they are linear (convex) and only
bind when tracking error would bring the body closer to an obstacle than the
reference does, which makes them a safety layer under the learned planner.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import osqp
import scipy.sparse as sp

from ..geometry import Agent, KIND_WALKER, Path, heading_and_speed, wrap_angle
from ..vehicle import VehicleParams, slip_angle


def _dbeta(df, p: VehicleParams):
    k = p.l_r / p.wheelbase
    t = np.tan(df)
    return k / np.cos(df) ** 2 / (1.0 + (k * t) ** 2)


def linearize(zb, ub, dt, p: VehicleParams):
    x, y, psi, v = zb
    a, df = ub
    beta = slip_angle(df, p)
    db = _dbeta(df, p)
    c, s = np.cos(psi + beta), np.sin(psi + beta)
    A = np.eye(4)
    A[0, 2] = -dt * v * s
    A[0, 3] = dt * c
    A[1, 2] = dt * v * c
    A[1, 3] = dt * s
    A[2, 3] = dt * np.sin(beta) / p.l_r
    B = np.zeros((4, 2))
    B[0, 1] = -dt * v * s * db
    B[1, 1] = dt * v * c * db
    B[2, 1] = dt * v * np.cos(beta) / p.l_r * db
    B[3, 0] = dt
    f = np.array([x + dt * v * c, y + dt * v * s, psi + dt * v / p.l_r * np.sin(beta), v + dt * a])
    return A, B, f - A @ zb - B @ ub


def inputs_from_states(z, dt, p: VehicleParams):
    """Feed-forward inputs consistent with a state sequence z (N+1, 4)."""
    a = np.diff(z[:, 3]) / dt
    dpsi = np.diff(z[:, 2]) / dt
    v = np.maximum(z[:-1, 3], 0.3)
    sb = np.clip(dpsi * p.l_r / v, -0.95, 0.95)
    beta = np.arcsin(sb)
    df = np.arctan(np.tan(beta) * p.wheelbase / p.l_r)
    return np.stack([np.clip(a, p.a_min, p.a_max), np.clip(df, p.df_min, p.df_max)], 1)


def reference_from_xy(xy, dt, ego_xy, ego_yaw, ego_v, look=0.4):
    """(T, 2) world xy at dt spacing -> (T, 4) z_ref [x, y, psi, v].

    Robust to the few-centimetre jitter of learned trajectories near standstill: the heading at
    step k points to the first later point at least `look` metres away (held when none exists),
    and the speed is the forward component of the motion (never negative).
    """
    xy = np.asarray(xy, float)
    T = len(xy)
    pts = np.vstack([np.asarray(ego_xy, float)[None], xy])
    yaw = np.empty(T)
    last = ego_yaw
    for k in range(T):
        d = np.hypot(*(pts[k + 1:] - pts[k + 1]).T) if k + 1 < T + 1 else np.zeros(0)
        base = pts[k]
        dist = np.hypot(*(pts[k + 1:] - base).T)
        j = np.nonzero(dist >= look)[0]
        if len(j):
            v = pts[k + 1 + j[0]] - base
            last = np.arctan2(v[1], v[0])
        yaw[k] = last
    yaw = ego_yaw + np.unwrap(np.concatenate([[0.0], wrap_angle(yaw - ego_yaw)]))[1:]
    step = np.diff(pts, axis=0)
    fwd = step[:, 0] * np.cos(yaw) + step[:, 1] * np.sin(yaw)
    v = np.maximum(fwd, 0.0) / dt
    if T >= 3:
        v = np.convolve(np.concatenate([[v[0]], v, [v[-1]]]), np.ones(3) / 3, mode="valid")
    return np.column_stack([xy, yaw, v])


class LTVMPC:
    def __init__(self, params: VehicleParams, N=20, DT=0.1,
                 Q=(4.0, 4.0, 2.0, 1.0), Qf_scale=2.0, Ru=(0.1, 1.0), Rd=(1.0, 50.0),
                 rate_slack=50.0, obs_slack=(2e3, 2e4), road_slack=(2e2, 2e3), d_safe=0.25,
                 n_ego_circles=3, obs_radius_gate=6.0, max_obs_rows=160):
        self.p = params
        self.N, self.DT = int(N), float(DT)
        self.Q = np.diag(Q)
        self.Qf = self.Q * Qf_scale
        self.Ru = np.diag(Ru)
        self.Rd = np.diag(Rd)
        self.w_rate = rate_slack
        self.w_obs = obs_slack
        self.w_road = road_slack
        self.d_safe = d_safe
        self.ego_offs, self.ego_r = params.ego_circles(n_ego_circles)
        self.gate = obs_radius_gate
        self.max_obs_rows = max_obs_rows
        self.last = None

    # indices -------------------------------------------------------------------------
    def _idx(self):
        N = self.N
        iz = 0
        iu = 4 * (N + 1)
        iso = iu + 2 * N          # obstacle slack per step
        isr = iso + N             # road slack per step
        isd = isr + N             # rate slack per step & input
        n = isd + 2 * N
        return iz, iu, iso, isr, isd, n

    def solve(self, z0, zref, u_prev, agents: Optional[List[Agent]] = None, route: Optional[Path] = None,
              s_ref: Optional[np.ndarray] = None):
        """z0 (4,), zref (N, 4) at times dt*(1..N), u_prev (2,).

        Returns dict(u (N,2), z (N+1,4), ok, status, solve_time, min_obs_slack...)
        """
        t0 = time.perf_counter()
        p, N, dt = self.p, self.N, self.DT
        z0 = np.asarray(z0, float).copy()
        zref = np.asarray(zref, float)
        if len(zref) < N:
            zref = np.vstack([zref, np.repeat(zref[-1:], N - len(zref), 0)])
        zref = zref[:N].copy()
        zref[:, 2] = z0[2] + np.unwrap(np.concatenate([[0.0], wrap_angle(zref[:, 2] - z0[2])]))[1:]
        u_prev = np.array([np.clip(u_prev[0], p.a_min, p.a_max), np.clip(u_prev[1], p.df_min, p.df_max)])

        zbar = np.vstack([z0, zref])
        ubar = inputs_from_states(zbar, dt, p)
        iz, iu, iso, isr, isd, n = self._idx()

        # ---- cost ------------------------------------------------------------------
        Pd = {}  # (i,j) -> val, upper triangle only

        def addP(i, j, v):
            if i > j:
                i, j = j, i
            Pd[(i, j)] = Pd.get((i, j), 0.0) + v

        q = np.zeros(n)
        for k in range(1, N + 1):
            Qk = self.Qf if k == N else self.Q
            for a in range(4):
                addP(iz + 4 * k + a, iz + 4 * k + a, 2 * Qk[a, a])
                q[iz + 4 * k + a] += -2 * Qk[a, a] * zref[k - 1, a]
        for k in range(N):
            for a in range(2):
                ia = iu + 2 * k + a
                addP(ia, ia, 2 * self.Ru[a, a])
                # rate cost
                addP(ia, ia, 2 * self.Rd[a, a])
                if k == 0:
                    q[ia] += -2 * self.Rd[a, a] * u_prev[a]
                else:
                    ib = iu + 2 * (k - 1) + a
                    addP(ib, ib, 2 * self.Rd[a, a])
                    addP(ia, ib, -2 * self.Rd[a, a])
        for k in range(N):
            q[iso + k] += self.w_obs[0]
            addP(iso + k, iso + k, 2 * self.w_obs[1])
            q[isr + k] += self.w_road[0]
            addP(isr + k, isr + k, 2 * self.w_road[1])
            for a in range(2):
                q[isd + 2 * k + a] += self.w_rate
                addP(isd + 2 * k + a, isd + 2 * k + a, 1e-3)
        rows_P = [ij[0] for ij in Pd]
        cols_P = [ij[1] for ij in Pd]
        P = sp.csc_matrix((list(Pd.values()), (rows_P, cols_P)), shape=(n, n))

        # ---- constraints -------------------------------------------------------------
        R, C, V, lo, hi = [], [], [], [], []
        r = 0

        def row(entries, l, u):
            nonlocal r
            for c, v in entries:
                R.append(r)
                C.append(c)
                V.append(v)
            lo.append(l)
            hi.append(u)
            r += 1

        for a in range(4):
            row([(iz + a, 1.0)], z0[a], z0[a])
        for k in range(N):
            A, B, c = linearize(zbar[k], ubar[k], dt, p)
            for a in range(4):
                ent = [(iz + 4 * (k + 1) + a, 1.0)]
                ent += [(iz + 4 * k + b, -A[a, b]) for b in range(4) if A[a, b] != 0.0]
                ent += [(iu + 2 * k + b, -B[a, b]) for b in range(2) if B[a, b] != 0.0]
                row(ent, c[a], c[a])
        umin = (p.a_min, p.df_min)
        umax = (p.a_max, p.df_max)
        rmin = (p.a_dot_min * dt, p.df_dot_min * dt)
        rmax = (p.a_dot_max * dt, p.df_dot_max * dt)
        for k in range(N):
            for a in range(2):
                ia = iu + 2 * k + a
                row([(ia, 1.0)], umin[a], umax[a])
                prev = [] if k == 0 else [(iu + 2 * (k - 1) + a, -1.0)]
                off = u_prev[a] if k == 0 else 0.0
                isl = isd + 2 * k + a
                row([(ia, 1.0)] + prev + [(isl, -1.0)], -np.inf, rmax[a] + off)
                row([(ia, 1.0)] + prev + [(isl, 1.0)], rmin[a] + off, np.inf)
        for k in range(N):
            row([(iso + k, 1.0)], 0.0, np.inf)
            row([(isr + k, 1.0)], 0.0, np.inf)
        for k in range(2 * N):
            row([(isd + k, 1.0)], 0.0, np.inf)

        # road bounds (route frame, linear)
        if route is not None:
            if s_ref is None:
                s_ref, _ = route.project_many(zref[:, :2])
            cx, cy, cyaw = route.pose(s_ref)
            dl, dr = route.bounds(s_ref)
            for k in range(N):
                nx, ny = -np.sin(cyaw[k]), np.cos(cyaw[k])
                ix = iz + 4 * (k + 1)
                off = nx * cx[k] + ny * cy[k]
                row([(ix, nx), (ix + 1, ny), (isr + k, -1.0)], -np.inf, dl[k] + off)
                row([(ix, nx), (ix + 1, ny), (isr + k, 1.0)], dr[k] + off, np.inf)

        # obstacle half-planes
        n_obs_rows = 0
        if agents:
            times = dt * (np.arange(N) + 1)
            circ = []
            for ag in agents:
                offs, rad = ag.circles(1 if ag.kind == KIND_WALKER else None)
                pred = ag.predict(times)
                for c in offs:
                    circ.append((pred[:, 0] + c * np.cos(pred[:, 2]), pred[:, 1] + c * np.sin(pred[:, 2]), rad))
            rows = []
            for k in range(N):
                xb, yb, pb = zref[k, 0], zref[k, 1], zref[k, 2]
                h = np.array([np.cos(pb), np.sin(pb)])
                dh = np.array([-np.sin(pb), np.cos(pb)])
                for ci in self.ego_offs:
                    ec = np.array([xb, yb]) + ci * h
                    for (ox, oy, rad) in circ:
                        o = np.array([ox[k], oy[k]])
                        dv = ec - o
                        dist = np.linalg.norm(dv)
                        need = self.ego_r + rad + self.d_safe
                        if dist - need > self.gate:
                            continue
                        nvec = dv / dist if dist > 1e-3 else np.array([-np.sin(pb), np.cos(pb)])
                        rows.append((dist - need, k, ci, nvec, o, need, h, dh, pb))
            rows.sort(key=lambda t: t[0])
            for (_, k, ci, nvec, o, need, h, dh, pb) in rows[: self.max_obs_rows]:
                ix = iz + 4 * (k + 1)
                g = ci * (nvec @ dh)
                rhs = need + nvec @ o - ci * (nvec @ h - (nvec @ dh) * pb)
                row([(ix, nvec[0]), (ix + 1, nvec[1]), (ix + 2, g), (iso + k, 1.0)], rhs, np.inf)
                n_obs_rows += 1

        Am = sp.csc_matrix((V, (R, C)), shape=(r, n))
        lo = np.array(lo)
        hi = np.array(hi)
        prob = osqp.OSQP()
        prob.setup(P, q, Am, lo, hi, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=8000,
                   polishing=False, warm_starting=True, time_limit=0.05)
        x0 = np.zeros(n)
        x0[iz:iz + 4 * (N + 1)] = zbar.ravel()
        x0[iu:iu + 2 * N] = ubar.ravel()
        try:
            prob.warm_start(x=x0)
        except Exception:
            pass
        res = prob.solve(raise_error=False)
        status = str(res.info.status)
        ok = status.lower().startswith("solved")
        x = res.x if res.x is not None and np.all(np.isfinite(res.x)) else None
        if x is None:
            u = np.vstack([ubar])
            z = zbar
            ok = False
            so = np.zeros(N)
        else:
            u = x[iu:iu + 2 * N].reshape(N, 2)
            z = x[iz:iz + 4 * (N + 1)].reshape(N + 1, 4)
            so = x[iso:iso + N]
        out = dict(u=u, z=z, ok=ok, status=status, solve_time=time.perf_counter() - t0,
                   obs_slack=float(np.max(so)) if len(so) else 0.0, n_obs_rows=n_obs_rows)
        self.last = out
        return out
