"""Closed-loop episode runner shared by data collection and evaluation."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from ..geometry import agents_to_array
from ..types import Controller
from .scenarios import ScenarioSpec


def _smooth(x, n=5):
    if len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n // 2), mode="edge"), k, mode="valid")


def comfort_signals(speed, lat_acc, dt):
    """Longitudinal jerk and lateral acceleration from 10 Hz logs, smoothed over 0.5 s
    (raw CARLA speed differences are dominated by contact/gear noise)."""
    if len(speed) < 8:
        return np.zeros(1), np.zeros(1)
    acc = np.gradient(_smooth(speed), dt)
    jerk = np.gradient(_smooth(acc), dt)[3:]
    return jerk, _smooth(lat_acc)


def episode_time_limit(spec: ScenarioSpec, route_length: float) -> float:
    return route_length / max(0.5 * spec.v_des, 1.0) + 25.0


def run_carla_episode(env, ctrl: Controller, spec: ScenarioSpec, privileged=False, record=False,
                      perturb_prob=0.0, rng: Optional[np.random.Generator] = None, verbose=False,
                      freeze_limit: float = 30.0, labeler=None) -> Dict:
    """freeze_limit: stop the episode (status 'timeout', freeze=True) after this many seconds
    at standstill - the 'freezing robot' failure mode.
    labeler: optional callable(privileged_obs) -> (T, 2) world xy; recorded per step as 'label'
    (DAgger: the learner drives, the privileged expert labels the states it visits)."""
    obs = env.reset(spec)
    if privileged:
        obs = env.observe(privileged=True)
    route = env.route
    ctrl.reset(route, spec.v_des, obs.ego)
    if labeler is not None and hasattr(labeler, "reset"):
        labeler.reset()
    t_max = episode_time_limit(spec, route.length)
    rng = rng or np.random.default_rng(spec.seed + 99)
    # noise-injection bursts (for recovery data); never in evaluation
    bursts = []
    if perturb_prob > 0 and rng.random() < perturb_prob:
        for _ in range(int(rng.integers(1, 4))):
            t0 = float(rng.uniform(3.0, max(4.0, t_max * 0.6)))
            bursts.append((t0, t0 + float(rng.uniform(0.5, 1.5)), float(rng.normal(0, 0.15)),
                           float(rng.normal(0, 0.3))))

    rec = dict(t=[], ego=[], cmd=[], applied=[], agents=[], s=[], d=[], perturbed=[], solve=[], ok=[],
               fallback=[], plan=[])
    status = "timeout"
    solve_times, solver_times, oks = [], [], []
    stuck_time = 0.0
    froze = False
    collided_with = ""
    wall0 = time.time()
    while True:
        cmd = ctrl.step(obs)
        solve_times.append(cmd.solve_time)
        solver_times.append(cmd.info.get("solver_time", cmd.solve_time))
        oks.append(bool(cmd.ok))
        pert = None
        for (b0, b1, pdf, pa) in bursts:
            if b0 <= obs.t <= b1:
                pert = (pa, pdf)
        if record:
            e = obs.ego
            rec["t"].append(obs.t)
            rec["ego"].append([e.x, e.y, e.yaw, e.v, e.a, e.df])
            rec["cmd"].append([cmd.a, cmd.df])
            rec["agents"].append(agents_to_array(env.last_gt, 40))
            rec["s"].append(obs.s_ego)
            rec["d"].append(obs.d_ego)
            rec["perturbed"].append(pert is not None)
            rec["solve"].append(cmd.solve_time)
            rec["ok"].append(cmd.ok)
            rec["fallback"].append(bool(cmd.info.get("fallback", False)))
            if labeler is not None:
                from dataclasses import replace as _replace
                rec.setdefault("label", []).append(labeler(_replace(obs, agents=env.last_gt)))
        obs, info = env.step(cmd, perturb=pert)
        if privileged:
            obs = env.observe(privileged=True)
        if info["collision"]:
            status = "collision"
            cw = info.get("collision_with")
            collided_with = str(cw[1]) if isinstance(cw, tuple) else "agent"
            break
        if info["offroad"]:
            status = "offroad"
            break
        if obs.s_ego >= route.length - 4.0:
            status = "success"
            break
        if obs.t >= t_max:
            status = "timeout"
            break
        stuck_time = stuck_time + 0.1 if obs.ego.v < 0.2 else 0.0
        if stuck_time >= freeze_limit:
            status = "timeout"
            froze = True
            break
        if verbose and int(obs.t * 10) % 50 == 0:
            print(f"  t={obs.t:5.1f} s={obs.s_ego:6.1f}/{route.length:.0f} v={obs.ego.v:4.2f} d={obs.d_ego:+.2f} "
                  f"solve={1e3 * cmd.solve_time:5.1f}ms")
    m = env.metrics
    jerk, lat = comfort_signals(np.array(m["speed"]), np.array(m["lat_acc"]), 0.1)
    st = np.array(solve_times)
    sv = np.array(solver_times)
    result = dict(
        town=spec.town, template=spec.template, seed=spec.seed, controller=ctrl.name, status=status,
        success=status == "success", collision=status == "collision", offroad=status == "offroad",
        timeout=status == "timeout", freeze=froze, collided_with=collided_with, route_length=float(route.length),
        progress=float(min(1.0, obs.s_ego / route.length)), time=float(obs.t),
        mean_speed=float(np.mean(m["speed"])) if m["speed"] else 0.0,
        min_dist=float(m["min_dist"]) if np.isfinite(m["min_dist"]) else 99.0,
        offroad_time=float(m["offroad"]), rms_jerk=float(np.sqrt(np.mean(jerk ** 2))),
        max_abs_jerk=float(np.max(np.abs(jerk))), max_lat_acc=float(np.max(np.abs(lat))),
        rms_lat_acc=float(np.sqrt(np.mean(lat ** 2))),
        solve_mean_ms=float(1e3 * st.mean()), solve_p95_ms=float(1e3 * np.percentile(st, 95)),
        solve_p99_ms=float(1e3 * np.percentile(st, 99)), solve_max_ms=float(1e3 * st.max()),
        solver_mean_ms=float(1e3 * sv.mean()), solver_p99_ms=float(1e3 * np.percentile(sv, 99)),
        solver_fail_rate=float(1.0 - np.mean(oks)), v_des=spec.v_des, n_traffic=spec.n_traffic,
        wall_time=float(time.time() - wall0),
    )
    if record:
        out = {k: np.array(v) for k, v in rec.items() if k != "plan"}
        out["route_xy"] = route.xy
        out["route_dl"] = route.d_left
        out["route_dr"] = route.d_right
        out["route_lw"] = route.lane_width
        out["v_des"] = np.array(spec.v_des)
        out["params"] = np.array([env.params.l_f, env.params.l_r, env.params.length, env.params.width,
                                  env.params.cog_to_center])
        result["record"] = out
    return result
