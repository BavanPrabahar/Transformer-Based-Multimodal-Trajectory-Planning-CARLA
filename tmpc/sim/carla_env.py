"""CARLA 0.9.13 closed-loop environment.

Conventions
-----------
CARLA/UE4 is left-handed (y to the right).  Everything that leaves this module
is converted to a right-handed frame: x_rh = x, y_rh = -y, yaw_rh = -yaw.
Positive steering (df > 0) turns left in that frame.

Realism features
----------------
* synchronous mode, 20 Hz physics, 10 Hz control, PhysX vehicle dynamics,
* compute latency: a command is applied only after the controller's measured
  wall-clock compute time has elapsed in simulation time (held command before),
* perception model for the evaluated stacks: 40 m range, Gaussian noise on
  position / heading / velocity / size (the expert sees ground truth),
* Traffic-Manager background traffic, scripted parked cars, cones/barriers,
  crossing pedestrians, slow leaders and oncoming vehicles,
* collision = CARLA collision sensor OR oriented-box overlap with any agent.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import carla

from ..geometry import (Agent, KIND_STATIC, KIND_VEHICLE, KIND_WALKER, Path, box_corners,
                        boxes_overlap, box_distance, wrap_angle)
from ..types import Command, EgoState, Observation
from ..vehicle import VehicleParams
from .scenarios import ScenarioSpec

EGO_BP = "vehicle.nissan.micra"
PARKED_BPS = ["vehicle.audi.a2", "vehicle.citroen.c3", "vehicle.toyota.prius", "vehicle.seat.leon",
              "vehicle.mini.cooper_s", "vehicle.nissan.micra", "vehicle.mercedes.coupe_2020",
              "vehicle.volkswagen.t2", "vehicle.tesla.model3", "vehicle.ford.mustang"]


# ---------------------------------------------------------------------------
# frame conversion
# ---------------------------------------------------------------------------
def rh_xy(loc):
    return float(loc.x), -float(loc.y)


def rh_yaw(rot):
    return -math.radians(rot.yaw)


def carla_transform(x, y, yaw, z=0.0):
    return carla.Transform(carla.Location(x=float(x), y=float(-y), z=float(z)),
                           carla.Rotation(yaw=float(-math.degrees(yaw))))


# ---------------------------------------------------------------------------
# low-level longitudinal/lateral control
# ---------------------------------------------------------------------------
class LowLevelController:
    """Acceleration/steer-angle command -> throttle/brake/steer for the ego blueprint.

    Feed-forward: steady-state throttle-to-hold-speed table measured on the ego
    blueprint (scripts/calibrate_lowlevel.py) + an acceleration term; feedback:
    integral action on the acceleration error (speed differences, low-pass).
    Decelerations within the coast-down capability use reduced throttle,
    stronger ones use the brake.
    """

    V_TAB = np.array([0.0, 1.17, 1.37, 1.60, 1.85, 2.23, 3.49, 5.0, 8.0])
    T_TAB = np.array([0.30, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.625, 0.645])
    # extra deceleration beyond coast-down (m/s^2) -> brake pedal, measured at ~5 m/s
    X_TAB = np.array([0.0, 0.87, 1.14, 1.45, 2.0, 2.9, 4.0])
    B_TAB = np.array([0.0, 0.10, 0.20, 0.30, 0.50, 0.80, 1.0])

    def __init__(self, max_steer_rad, dt, k_a=0.10, ki=0.25, kb=0.22, coast_k=0.6):
        self.max_steer = max_steer_rad
        self.dt = dt
        self.k_a, self.ki, self.kb, self.coast_k = k_a, ki, kb, coast_k
        self.reset()

    def reset(self):
        self.i = 0.0
        self.v_prev = None
        self.a_meas = 0.0

    def __call__(self, a_cmd, df_cmd, v):
        steer = float(np.clip(-df_cmd / self.max_steer, -1.0, 1.0))
        if self.v_prev is not None:
            a_raw = (v - self.v_prev) / self.dt
            self.a_meas += 0.3 * (a_raw - self.a_meas)
        self.v_prev = v
        if v < 0.3 and a_cmd <= 0.05:
            self.i = 0.0
            return carla.VehicleControl(throttle=0.0, steer=steer, brake=0.5)
        coast = float(np.clip(self.coast_k * v, 0.3, 1.6))   # achievable decel with zero throttle
        thr_ss = float(np.interp(v, self.V_TAB, self.T_TAB))
        if a_cmd >= 0.0:
            self.i = float(np.clip(self.i + self.ki * (a_cmd - self.a_meas) * self.dt, -0.12, 0.12))
            thr = thr_ss + self.k_a * a_cmd + self.i
            return carla.VehicleControl(throttle=float(np.clip(thr, 0.0, 1.0)), steer=steer, brake=0.0)
        if a_cmd > -coast:
            self.i *= 0.9
            thr = thr_ss * (1.0 + a_cmd / coast) + self.i
            return carla.VehicleControl(throttle=float(np.clip(thr, 0.0, 1.0)), steer=steer, brake=0.0)
        self.i = 0.0
        brk = float(np.interp(-a_cmd - coast, self.X_TAB, self.B_TAB))
        return carla.VehicleControl(throttle=0.0, steer=steer, brake=brk)


# ---------------------------------------------------------------------------
# scripted agents
# ---------------------------------------------------------------------------
@dataclass
class Scripted:
    actor: object
    kind: int
    length: float
    width: float
    xy: np.ndarray          # current RH position
    yaw: float
    v: float = 0.0          # speed along heading
    mode: str = "static"    # static | route | line
    s: float = 0.0          # route arc length (route mode)
    d: float = 0.0
    direction: float = 1.0  # +1 along route, -1 against
    line: Optional[np.ndarray] = None   # (2, 2) start/end for line mode
    progress: float = 0.0
    trigger_s: float = -1.0
    active: bool = True
    z: float = 0.0

    def agent(self, aid):
        vx = self.v * math.cos(self.yaw) if self.active else 0.0
        vy = self.v * math.sin(self.yaw) if self.active else 0.0
        kind = self.kind if (self.kind != KIND_VEHICLE or self.v > 0.01) else KIND_STATIC
        return Agent(aid, kind, float(self.xy[0]), float(self.xy[1]), float(self.yaw), vx, vy,
                     self.length, self.width)


class CarlaEnv:
    def __init__(self, host="localhost", port=2000, tm_port=8000, dt=0.05, control_every=2,
                 no_rendering=True, timeout=60.0, history_len=11, sim_latency=True):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.tm_port = tm_port
        self.dt = dt
        self.control_every = control_every
        self.no_rendering = no_rendering
        self.history_len = history_len
        self.sim_latency = sim_latency
        self.world = self.client.get_world()
        self.town = None
        self.actors = []
        self.sensors = []
        self.tm_vehicles = []
        self.scripted: List[Scripted] = []
        self.ego = None
        self.params = None

    # -- world management ---------------------------------------------------------------
    def load_town(self, town):
        cur = self.world.get_map().name.split("/")[-1]
        if cur != town:
            self._set_sync(False)
            self.world = self.client.load_world(town)
            time.sleep(1.0)
        self.town = town
        self.map = self.world.get_map()
        self._set_sync(True)
        self._destroy_leftovers()
        self.bp = self.world.get_blueprint_library()
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        self.spawn_points = self.map.get_spawn_points()

    def _destroy_leftovers(self):
        """Remove actors left behind by crashed/killed clients (they are invisible to us
        but still collide with the ego)."""
        acts = self.world.get_actors()
        ids = [a.id for a in acts if a.type_id.split(".")[0] in ("vehicle", "walker", "sensor", "controller")
               or a.type_id.startswith("static.prop")]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
            self.world.tick()

    def _set_sync(self, on):
        s = self.world.get_settings()
        s.synchronous_mode = on
        s.fixed_delta_seconds = self.dt if on else None
        s.no_rendering_mode = self.no_rendering
        self.world.apply_settings(s)

    def tick(self):
        self.world.tick()
        self.sim_time += self.dt

    # -- route ------------------------------------------------------------------------------
    def build_route(self, start_wp, length, rng, margin=0.25):
        wps = [start_wp]
        dist = 0.0
        while dist < length:
            nxt = wps[-1].next(1.0)
            if not nxt:
                break
            if len(nxt) > 1:
                # prefer smaller heading change (campus-like routes), still random
                yaw0 = wps[-1].transform.rotation.yaw
                dy = np.array([abs(wrap_angle(math.radians(n.transform.rotation.yaw - yaw0))) for n in nxt])
                p = np.exp(-dy / 0.8)
                p /= p.sum()
                nxt = [nxt[int(rng.choice(len(nxt), p=p))]]
            wps.append(nxt[0])
            dist += 1.0
        half = self.params.width / 2.0 + margin
        xy, dl, dr, lw, junc = [], [], [], [], []
        for wp in wps:
            xy.append(rh_xy(wp.transform.location))
            w = wp.lane_width
            left, right = w / 2.0, w / 2.0
            if not wp.is_junction:
                l = wp.get_left_lane()
                n = 0
                while l is not None and l.lane_type == carla.LaneType.Driving and n < 2:
                    left += l.lane_width
                    if l.lane_id * wp.lane_id < 0:
                        break  # one opposite-direction lane may be used for passing
                    l = l.get_left_lane()
                    n += 1
                r = wp.get_right_lane()
                n = 0
                while (r is not None and r.lane_type == carla.LaneType.Driving and r.lane_id * wp.lane_id > 0
                       and n < 2):
                    right += r.lane_width
                    r = r.get_right_lane()
                    n += 1
            else:
                left += 1.5    # junction areas are open asphalt: allow a wider swing
                right += 0.5
            dl.append(left - half)
            dr.append(-(right - half))
            lw.append(w)
            junc.append(wp.is_junction)
        xy = np.array(xy)
        dl = np.array(dl)
        dr = np.array(dr)
        # conservative smoothing: bounds may only shrink near transitions (+-4 m)
        k = 4
        dl_s = np.array([dl[max(0, i - k):i + k + 1].min() for i in range(len(dl))])
        dr_s = np.array([dr[max(0, i - k):i + k + 1].max() for i in range(len(dr))])
        dl_s = np.maximum(dl_s, 0.05)
        dr_s = np.minimum(dr_s, -0.05)
        # densify to 0.5 m
        path = Path(xy, dl_s, dr_s, np.array(lw))
        ss = np.arange(0.0, path.length, 0.5)
        x, y, _ = path.pose(ss)
        dl2, dr2 = path.bounds(ss)
        lw2 = np.interp(ss, path.s, path.lane_width)
        route = Path(np.stack([x, y], 1), dl2, dr2, lw2)
        route.junction = np.interp(route.s, path.s, np.array(junc, float)) > 0.01
        return route

    # -- ego geometry --------------------------------------------------------------------------
    def _ego_geometry(self):
        tf = self.ego.get_transform()
        pc = self.ego.get_physics_control()
        fwd = tf.get_forward_vector()
        loc = tf.location

        def local_x(pos_cm):
            return ((pos_cm.x / 100.0 - loc.x) * fwd.x + (pos_cm.y / 100.0 - loc.y) * fwd.y)

        xf = 0.5 * (local_x(pc.wheels[0].position) + local_x(pc.wheels[1].position))
        xr = 0.5 * (local_x(pc.wheels[2].position) + local_x(pc.wheels[3].position))
        assert 1.5 < xf - xr < 4.0, f"implausible wheelbase {xf - xr:.2f} (transform not updated yet?)"
        # kinematic reference point: axle midpoint (purely geometric; avoids relying on PhysX COM)
        x_cog = 0.5 * (xf + xr)
        bb = self.ego.bounding_box
        self.cog_local = x_cog
        self.max_steer = math.radians(pc.wheels[0].max_steer_angle)
        p = VehicleParams().with_geometry(l_f=xf - x_cog, l_r=x_cog - xr, length=2 * bb.extent.x,
                                          width=2 * bb.extent.y, cog_to_center=bb.location.x - x_cog)
        # steering limit: physical wheel limit (lab MPC uses 0.75 rad)
        from dataclasses import replace
        df = min(0.75, self.max_steer)
        return replace(p, df_min=-df, df_max=df)

    def ego_state(self) -> EgoState:
        tf = self.ego.get_transform()
        x, y = rh_xy(tf.location)
        yaw = rh_yaw(tf.rotation)
        x += self.cog_local * math.cos(yaw)
        y += self.cog_local * math.sin(yaw)
        vel = self.ego.get_velocity()
        f = tf.get_forward_vector()
        v = vel.x * f.x + vel.y * f.y + vel.z * f.z
        return EgoState(x, y, yaw, float(v), a=self.applied[0], df=self.applied[1])

    # -- spawning helpers -------------------------------------------------------------------------
    def _spawn(self, bp_id, x, y, yaw, z=None, physics=True, role=None):
        bp = self.bp.find(bp_id)
        if bp.has_attribute("color"):
            bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
        if role and bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role)
        if z is None:
            wp = self.map.get_waypoint(carla.Location(x=x, y=-y, z=0.0), project_to_road=True)
            z = wp.transform.location.z + 0.3 if wp is not None else 0.5
        a = self.world.try_spawn_actor(bp, carla_transform(x, y, yaw, z))
        if a is not None:
            self.actors.append(a)
            if not physics:
                a.set_simulate_physics(False)
        return a

    def _place_parked(self, route, el, s, d):
        x, y, yaw = route.frenet_to_xy(s, d)
        bp_id = str(self.rng.choice(PARKED_BPS))
        a = self._spawn(bp_id, x, y, yaw + el.get("dyaw", 0.0), physics=False)
        if a is None:
            return None
        bb = a.bounding_box
        return Scripted(a, KIND_STATIC, 2 * bb.extent.x, 2 * bb.extent.y, np.array([x, y]),
                        yaw + el.get("dyaw", 0.0))

    def _valid_s(self, route, s, need_left=2.6):
        """Placement allowed: not inside/near a junction and enough room to pass."""
        lo = max(0.0, s - 20.0)      # room to start a swerve after a junction exit
        hi = min(route.length, s + 8.0)
        idx = (route.s >= lo) & (route.s <= hi)
        if route.junction[idx].any():
            return False
        dl, _ = route.bounds(np.linspace(lo, hi, 9))
        return bool(np.min(dl) >= need_left)

    def _find_s(self, route, frac, need_left=2.6):
        cand = route.length * frac
        for delta in np.concatenate([[0.0], np.ravel(np.column_stack([np.arange(3, 60, 3), -np.arange(3, 60, 3)]))]):
            s = cand + delta
            if 30.0 <= s <= route.length - 35.0 and self._valid_s(route, s, need_left):
                return s
        return None

    # -- reset -----------------------------------------------------------------------------------
    def reset(self, spec: ScenarioSpec, max_tries=20):
        if self.town != spec.town:
            self.load_town(spec.town)
        self.cleanup()
        self.rng = np.random.default_rng(spec.seed)
        self.spec = spec
        self.noise_rng = np.random.default_rng(spec.seed + 7)
        self.sim_time = 0.0
        self.applied = (0.0, 0.0)
        self.pending = deque()
        # ego blueprint geometry needs a spawned actor: spawn at a valid spawn point first
        route = None
        for _ in range(max_tries):
            sp = self.spawn_points[int(self.rng.integers(len(self.spawn_points)))]
            wp = self.map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
                continue
            if self.params is None:
                self.ego = self.world.try_spawn_actor(self.bp.find(EGO_BP), sp)
                if self.ego is None:
                    continue
                self.tick()
                self.params = self._ego_geometry()
                self.ego.destroy()
                self.ego = None
            route = self.build_route(wp, spec.route_length, self.rng)
            if route.length < 0.8 * spec.route_length:
                continue
            needs_room = any(e["type"] in ("parked", "oncoming", "cones") for e in spec.elements)
            if needs_room and self._find_s(route, 0.45) is None:
                continue
            break
        if route is None:
            raise RuntimeError("could not build a route")
        self.route = route

        # ego
        d0 = float(np.clip(spec.ego_init.get("d", 0.0), route.d_right[0], route.d_left[0]))
        x0, y0, yaw0 = route.frenet_to_xy(2.0, d0)
        yaw0 += spec.ego_init.get("dyaw", 0.0)
        x0 -= self.params.cog_to_center * 0  # spawn transform is the actor origin
        bp = self.bp.find(EGO_BP)
        bp.set_attribute("role_name", "hero")
        wp = self.map.get_waypoint(carla.Location(x=x0, y=-y0), project_to_road=True)
        self.ego = self.world.try_spawn_actor(bp, carla_transform(x0, y0, yaw0, wp.transform.location.z + 0.3))
        if self.ego is None:
            raise RuntimeError("ego spawn failed")
        self.actors.append(self.ego)
        self.ego.apply_control(carla.VehicleControl(brake=1.0))
        self.tick()   # transforms are only valid after the first tick in synchronous mode
        self.params = self._ego_geometry()
        self.lowlevel = LowLevelController(self.max_steer, self.dt)
        self.collisions = []
        cs = self.world.spawn_actor(self.bp.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego)
        cs.listen(lambda ev: self.collisions.append((ev.other_actor.id, ev.other_actor.type_id)))
        self.sensors.append(cs)

        # scenario elements
        self.scripted = []
        for el in spec.elements:
            self._add_element(route, el)
        # background traffic
        self._spawn_traffic(spec)
        for _ in range(10):  # settle (vehicles drop to the ground)
            self.ego.apply_control(carla.VehicleControl(brake=1.0))
            self._update_scripted(0.0)
            self.tick()
        self.sim_time = 0.0
        e = self.ego_state()
        self.hist = deque([e.z()] * self.history_len, maxlen=self.history_len)
        self.s_prev, _ = route.project(e.x, e.y, window=10.0, s_hint=2.0)
        self.offroad_time = 0.0
        self.metrics = dict(min_dist=np.inf, offroad=0.0, max_lat_acc=0.0, jerk=[], lat_acc=[], speed=[])
        self._last_v = e.v
        self._last_acc = 0.0
        return self.observe()

    def _add_element(self, route, el):
        t = el["type"]
        if t == "parked":
            s = self._find_s(route, el["s_frac"]) if "s_offset" not in el else None
            if "s_offset" in el:
                base = [sc for sc in self.scripted if sc.kind == KIND_STATIC]
                if not base:
                    return
                s0, _ = route.project(*base[0].xy)
                s = s0 + el["s_offset"]
                if s > route.length - 30:
                    return
                lw = float(np.interp(s, route.s, route.lane_width))
                d = lw + el.get("d_opposite", 0.0) - 0.5 * lw     # partly in the opposite lane
                d = min(d, float(route.bounds(s)[0]) + self.params.width / 2 + 0.25)
            else:
                d = el["d"]
            if s is None:
                return
            sc = self._place_parked(route, el, s, d)
            if sc is not None:
                self.scripted.append(sc)
        elif t == "cones":
            s = self._find_s(route, el["s_frac"])
            if s is None:
                return
            n = max(3, int(el["length"] / 1.5))
            if el.get("barrier"):
                x, y, yaw = route.frenet_to_xy(s, el["d"])
                a = self._spawn("static.prop.streetbarrier", x, y, yaw + math.pi / 2, physics=False)
                if a is not None:
                    self.scripted.append(Scripted(a, KIND_STATIC, 0.6, 2.0, np.array([x, y]), yaw))
            for i in range(n):
                ss = s - el["length"] / 2 + i * el["length"] / (n - 1)
                dd = el["d"] + 0.6 * (1 - abs(2 * i / (n - 1) - 1))   # tapered closure
                x, y, yaw = route.frenet_to_xy(ss, dd)
                a = self._spawn("static.prop.constructioncone", x, y, yaw, physics=False)
                if a is not None:
                    self.scripted.append(Scripted(a, KIND_STATIC, 0.45, 0.45, np.array([x, y]), yaw))
        elif t == "walker":
            s = self._find_s(route, el["s_frac"], need_left=0.0)
            if s is None:
                return
            dl, dr = route.bounds(s)
            half = self.params.width / 2 + 0.25
            d_from, d_to = dr - half - 2.0, dl + half + 2.0
            if not el["left_to_right"]:
                d_from, d_to = d_to, d_from
            x0, y0, yaw = route.frenet_to_xy(s, d_from)
            x1, y1, _ = route.frenet_to_xy(s, d_to)
            bps = [b.id for b in self.bp.filter("walker.pedestrian.*")]
            a = self._spawn(str(self.rng.choice(bps)), x0, y0, math.atan2(y1 - y0, x1 - x0), physics=False)
            if a is not None:
                sc = Scripted(a, KIND_WALKER, 0.5, 0.5, np.array([x0, y0]), math.atan2(y1 - y0, x1 - x0),
                              v=el["speed"], mode="line", line=np.array([[x0, y0], [x1, y1]]),
                              trigger_s=s - el["trigger"], active=False)
                sc.z = a.get_transform().location.z
                self.scripted.append(sc)
        elif t in ("leader", "oncoming"):
            if t == "leader":
                s, d, direction = 2.0 + el["ahead"], 0.0, 1.0
            else:
                s = min(route.length - 5.0, 2.0 + el["ahead"])
                lw = float(np.interp(s, route.s, route.lane_width))
                if float(np.min(route.bounds(np.linspace(10, s, 20))[0])) < lw - 0.3:
                    return    # no passing lane: an oncoming car would be on our lane
                d, direction = lw, -1.0
            x, y, yaw = route.frenet_to_xy(s, d)
            yaw = yaw if direction > 0 else yaw + math.pi
            a = self._spawn(str(self.rng.choice(PARKED_BPS[:6])), x, y, yaw, physics=False)
            if a is not None:
                bb = a.bounding_box
                self.scripted.append(Scripted(a, KIND_VEHICLE, 2 * bb.extent.x, 2 * bb.extent.y, np.array([x, y]),
                                              yaw, v=el["speed"], mode="route", s=s, d=d, direction=direction))

    def _spawn_traffic(self, spec):
        self.tm_vehicles = []
        if spec.n_traffic <= 0:
            return
        self.tm.set_random_device_seed(int(spec.seed % 100000))
        self.tm.global_percentage_speed_difference(float(spec.traffic_speed_diff))
        self.tm.set_global_distance_to_leading_vehicle(3.0)
        ex, ey = self.route.xy[0]
        order = self.rng.permutation(len(self.spawn_points))
        bps = [b for b in self.bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        for i in order:
            if len(self.tm_vehicles) >= spec.n_traffic:
                break
            sp = self.spawn_points[i]
            x, y = rh_xy(sp.location)
            if np.hypot(x - ex, y - ey) < 25.0:
                continue
            # keep the first 35 m of the route clear
            s, d = self.route.project(x, y)
            if s < 35.0 and abs(d) < 6.0:
                continue
            bp = bps[int(self.rng.integers(len(bps)))]
            a = self.world.try_spawn_actor(bp, sp)
            if a is not None:
                a.set_autopilot(True, self.tm.get_port())
                self.tm.auto_lane_change(a, False)
                self.actors.append(a)
                self.tm_vehicles.append(a)

    # -- stepping ---------------------------------------------------------------------------------
    def _update_scripted(self, ego_s):
        for sc in self.scripted:
            if sc.mode in ("static", "gone"):
                continue
            if not sc.active:
                if ego_s >= sc.trigger_s:
                    sc.active = True
                else:
                    continue
            if sc.mode == "route":
                sc.s += sc.direction * sc.v * self.dt
                if sc.s < 0.5 or sc.s > self.route.length - 0.5:
                    # end of its path: the vehicle turns off / leaves the scene
                    sc.mode, sc.active, sc.v = "gone", False, 0.0
                    sc.actor.set_transform(carla.Transform(carla.Location(x=0.0, y=0.0, z=-200.0 - 5 * len(self.scripted))))
                    continue
                x, y, yaw = self.route.frenet_to_xy(sc.s, sc.d)
                sc.xy = np.array([x, y])
                sc.yaw = yaw if sc.direction > 0 else yaw + math.pi
                z = None
            else:  # line (walker)
                L = np.linalg.norm(sc.line[1] - sc.line[0])
                sc.progress = min(L, sc.progress + sc.v * self.dt)
                if sc.progress >= L:
                    sc.v = 0.0
                sc.xy = sc.line[0] + (sc.line[1] - sc.line[0]) * (sc.progress / max(L, 1e-6))
                z = sc.z
            if z is None:
                wp = self.map.get_waypoint(carla.Location(x=float(sc.xy[0]), y=float(-sc.xy[1])), project_to_road=True)
                z = wp.transform.location.z + 0.05 if wp is not None else 0.1
            sc.actor.set_transform(carla_transform(sc.xy[0], sc.xy[1], sc.yaw, z))

    def ground_truth_agents(self) -> List[Agent]:
        agents = []
        for sc in self.scripted:
            if sc.mode == "gone":
                continue
            agents.append(sc.agent(sc.actor.id))   # a waiting walker is visible with v = 0
        for a in self.tm_vehicles:
            if not a.is_alive:
                continue
            tf = a.get_transform()
            x, y = rh_xy(tf.location)
            yaw = rh_yaw(tf.rotation)
            v = a.get_velocity()
            bb = a.bounding_box
            off = bb.location.x
            agents.append(Agent(a.id, KIND_VEHICLE, x + off * math.cos(yaw), y + off * math.sin(yaw), yaw,
                                float(v.x), float(-v.y), 2 * bb.extent.x, 2 * bb.extent.y))
        return agents

    def perceive(self, agents: List[Agent], ego: EgoState, noise=True, range_m=40.0):
        out = []
        for a in agents:
            if np.hypot(a.x - ego.x, a.y - ego.y) > range_m:
                continue
            if not noise:
                out.append(a)
                continue
            r = self.noise_rng
            out.append(Agent(a.id, a.kind, a.x + r.normal(0, 0.10), a.y + r.normal(0, 0.10),
                             a.yaw + r.normal(0, 0.03), a.vx + r.normal(0, 0.15), a.vy + r.normal(0, 0.15),
                             max(0.3, a.length + r.normal(0, 0.05)), max(0.3, a.width + r.normal(0, 0.05))))
        return out

    def observe(self, privileged=False) -> Observation:
        e = self.ego_state()
        s, d = self.route.project(e.x, e.y, window=15.0, s_hint=self.s_prev)
        self.s_prev = s
        gt = self.ground_truth_agents()
        self.last_gt = gt
        agents = gt if privileged else self.perceive(gt, e, noise=self.spec.perception_noise)
        return Observation(self.sim_time, e, agents, self.route, self.spec.v_des,
                           np.array(self.hist), s, d)

    def ego_box(self, e: EgoState):
        cx = e.x + self.params.cog_to_center * math.cos(e.yaw)
        cy = e.y + self.params.cog_to_center * math.sin(e.yaw)
        return box_corners(cx, cy, e.yaw, self.params.length, self.params.width)

    def step(self, cmd: Command, perturb: Optional[tuple] = None):
        """Apply a command (with simulated compute latency) for one control period."""
        a, df = float(cmd.a), float(cmd.df)
        if perturb is not None:
            a += perturb[0]
            df += perturb[1]
        delay = cmd.solve_time if self.sim_latency else 0.0
        self.pending.append((self.sim_time + delay, (a, df)))
        info = dict(collision=False, offroad=False)
        for _ in range(self.control_every):
            while self.pending and self.pending[0][0] <= self.sim_time + 1e-9:
                self.applied = self.pending.popleft()[1]
            e = self.ego_state()
            self.ego.apply_control(self.lowlevel(self.applied[0], self.applied[1], e.v))
            self._update_scripted(self.s_prev)
            self.tick()
        e = self.ego_state()
        self.hist.append(e.z())
        # metrics ----------------------------------------------------------------------------
        T = self.dt * self.control_every
        acc = (e.v - self._last_v) / T
        self.metrics["jerk"].append((acc - self._last_acc) / T)
        self._last_v, self._last_acc = e.v, acc
        yaw_rate = wrap_angle(self.hist[-1][2] - self.hist[-2][2]) / T
        lat = e.v * yaw_rate
        self.metrics["lat_acc"].append(lat)
        self.metrics["speed"].append(e.v)
        eb = self.ego_box(e)
        gt = self.ground_truth_agents()
        dmin = np.inf
        hit = None
        for ag in gt:
            if np.hypot(ag.x - e.x, ag.y - e.y) > 15.0:
                continue
            ob = box_corners(ag.x, ag.y, ag.yaw, ag.length, ag.width)
            if boxes_overlap(eb, ob):
                hit = ag.id
                dmin = 0.0
                break
            dmin = min(dmin, box_distance(eb, ob))
        self.metrics["min_dist"] = min(self.metrics["min_dist"], dmin)
        if hit is not None or self.collisions:
            info["collision"] = True
            info["collision_with"] = hit if hit is not None else self.collisions[0]
        # off-road: centre of the body not on a driving lane
        c = eb.mean(0)
        wp = self.map.get_waypoint(carla.Location(x=float(c[0]), y=float(-c[1])), project_to_road=False,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            self.offroad_time += T
            self.metrics["offroad"] += T
        else:
            self.offroad_time = 0.0
        info["offroad"] = self.offroad_time > 1.0
        info["min_dist"] = dmin
        return self.observe(), info

    # -- cleanup ----------------------------------------------------------------------------------
    def cleanup(self):
        for s in self.sensors:
            try:
                s.stop()
            except Exception:
                pass
        ids = [a.id for a in self.sensors + self.actors if a is not None]
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self.sensors, self.actors, self.tm_vehicles, self.scripted = [], [], [], []
        self.ego = None
        try:
            self.world.tick()
        except Exception:
            pass

    def close(self):
        self.cleanup()
        try:
            self._set_sync(False)
        except Exception:
            pass
