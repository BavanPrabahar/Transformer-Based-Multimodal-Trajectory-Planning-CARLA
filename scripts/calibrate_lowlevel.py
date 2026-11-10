"""Measure throttle/brake -> acceleration of the ego blueprint and check the steering sign.

Usage: python scripts/calibrate_lowlevel.py --town Town01
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import carla  # noqa: E402

from tmpc.sim.carla_env import CarlaEnv, EGO_BP, rh_xy, rh_yaw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="results/lowlevel_calibration.json")
    args = ap.parse_args()
    env = CarlaEnv(port=args.port)
    env.load_town(args.town)
    env.sim_time = 0.0
    sps = env.spawn_points

    def fresh():
        env.cleanup()
        # longest straight-ish spawn: pick spawn with a long run ahead
        best, best_len = None, 0
        for sp in sps[:60]:
            wp = env.map.get_waypoint(sp.location)
            L, w = 0, wp
            while L < 120:
                n = w.next(2.0)
                if not n or abs(((n[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360) - 180) > 5:
                    break
                w = n[0]
                L += 2
            if L > best_len:
                best, best_len = sp, L
        ego = env.world.try_spawn_actor(env.bp.find(EGO_BP), best)
        env.actors.append(ego)
        env.ego = ego
        for _ in range(20):
            ego.apply_control(carla.VehicleControl(brake=1.0))
            env.tick()
        return ego, best_len

    def speed(ego):
        v = ego.get_velocity()
        f = ego.get_transform().get_forward_vector()
        return v.x * f.x + v.y * f.y

    res = dict(throttle={}, brake={})
    for thr in [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]:
        ego, L = fresh()
        vs = []
        for k in range(80):  # 4 s
            ego.apply_control(carla.VehicleControl(throttle=thr))
            env.tick()
            vs.append(speed(ego))
        vs = np.array(vs)
        # acceleration in the 1..6 m/s band
        t = np.arange(len(vs)) * env.dt
        band = (vs > 0.5) & (vs < 6.0)
        a = np.polyfit(t[band], vs[band], 1)[0] if band.sum() > 5 else float("nan")
        res["throttle"][thr] = dict(acc=float(a), v_end=float(vs[-1]))
        print(f"throttle {thr:.2f}: acc {a:.2f} m/s^2  v(4s) {vs[-1]:.2f}")
    for brk in [0.1, 0.2, 0.3, 0.5, 0.8]:
        ego, L = fresh()
        for k in range(200):
            ego.apply_control(carla.VehicleControl(throttle=0.6))
            env.tick()
            if speed(ego) > 5.0:
                break
        vs = []
        for k in range(40):
            ego.apply_control(carla.VehicleControl(brake=brk))
            env.tick()
            vs.append(speed(ego))
        vs = np.array(vs)
        t = np.arange(len(vs)) * env.dt
        band = vs > 0.5
        a = np.polyfit(t[band], vs[band], 1)[0] if band.sum() > 5 else float("nan")
        res["brake"][brk] = dict(acc=float(a))
        print(f"brake {brk:.2f}: acc {a:.2f} m/s^2")
    # coast deceleration
    ego, L = fresh()
    for k in range(200):
        ego.apply_control(carla.VehicleControl(throttle=0.6))
        env.tick()
        if speed(ego) > 5.0:
            break
    vs = []
    for k in range(40):
        ego.apply_control(carla.VehicleControl(throttle=0.0))
        env.tick()
        vs.append(speed(ego))
    res["coast_acc"] = float(np.polyfit(np.arange(len(vs)) * env.dt, vs, 1)[0])
    print("coast", res["coast_acc"])
    # steering sign: positive CARLA steer should turn right == negative RH yaw rate
    ego, L = fresh()
    yaw0 = rh_yaw(ego.get_transform().rotation)
    for k in range(60):
        ego.apply_control(carla.VehicleControl(throttle=0.4, steer=0.3))
        env.tick()
    dyaw = math.atan2(math.sin(rh_yaw(ego.get_transform().rotation) - yaw0),
                      math.cos(rh_yaw(ego.get_transform().rotation) - yaw0))
    res["steer_plus_dyaw_rh"] = dyaw
    print("CARLA steer +0.3 -> RH yaw change", dyaw, "(negative expected)")
    pc = ego.get_physics_control()
    res["max_steer_deg"] = pc.wheels[0].max_steer_angle
    res["mass"] = pc.mass
    env.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
