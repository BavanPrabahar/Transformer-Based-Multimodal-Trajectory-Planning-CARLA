"""CARLA-free unit tests (run: .venv/bin/python -m pytest -q tests)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmpc.controllers.lab_mpc import SpeedGovernor, corridor_distance, lab_reference  # noqa: E402
from tmpc.controllers.ltv_mpc import LTVMPC, linearize, reference_from_xy  # noqa: E402
from tmpc.geometry import (Agent, KIND_STATIC, KIND_VEHICLE, KIND_WALKER, Path, box_corners, box_distance,  # noqa: E402
                           boxes_overlap, to_global, to_local)
from tmpc.planning.safety import trajectory_clearance  # noqa: E402
from tmpc.sim.kinematic import KinematicSim, curved_road, run_episode, straight_road  # noqa: E402
from tmpc.types import Command, Controller  # noqa: E402
from tmpc.vehicle import VehicleParams, bicycle_step  # noqa: E402

P = VehicleParams()


def test_frames_roundtrip():
    xy = np.random.default_rng(0).normal(size=(10, 2)) * 5
    loc = to_local(xy, 3.0, -2.0, 0.7)
    assert np.allclose(to_global(loc, 3.0, -2.0, 0.7), xy)


def test_path_projection_and_frenet():
    r = curved_road()
    for s in [5.0, 40.0, 70.0]:
        for d in [-1.0, 0.0, 1.5]:
            x, y, _ = r.frenet_to_xy(s, d)
            s2, d2 = r.project(x, y)
            assert abs(s2 - s) < 0.3 and abs(d2 - d) < 0.05
    s_many, d_many = r.project_many(np.array([r.frenet_to_xy(30.0, 0.7)[:2]]))
    assert abs(s_many[0] - 30.0) < 0.3 and abs(d_many[0] - 0.7) < 0.05


def test_box_overlap_and_distance():
    a = box_corners(0, 0, 0, 4, 2)
    b = box_corners(5, 0, 0, 4, 2)
    assert not boxes_overlap(a, b) and abs(box_distance(a, b) - 1.0) < 1e-6
    c = box_corners(3.5, 0.5, 0.3, 4, 2)
    assert boxes_overlap(a, c) and box_distance(a, c) == 0.0


def test_bicycle_turns_left_for_positive_steer():
    z = np.array([0, 0, 0, 3.0])
    for _ in range(20):
        z = bicycle_step(z, [0.0, 0.3], 0.1, P)
    assert z[2] > 0.3 and z[1] > 0


def test_linearization_matches_finite_difference():
    zb, ub = np.array([1.0, 2.0, 0.4, 3.0]), np.array([0.2, 0.1])
    A, B, c = linearize(zb, ub, 0.1, P)
    dz = np.array([0.01, -0.02, 0.01, 0.05])
    du = np.array([0.05, 0.02])
    exact = bicycle_step(zb + dz, ub + du, 0.1, P)
    approx = A @ (zb + dz) + B @ (ub + du) + c
    assert np.allclose(exact, approx, atol=2e-3)


def test_lab_reference_ramp_and_stop():
    r = straight_road(60)
    zref, stop = lab_reference(r, 0.0, 0.0, 0.0, 3.0, 10, 0.4, v_init=0.0, a_ramp=0.8)
    assert zref.shape == (10, 4) and not stop
    assert np.all(np.diff(zref[:, 0]) > 0) and zref[0, 0] < 3.0 * 0.4  # ramped start
    _, stop = lab_reference(r, 58.0, 0.0, 0.0, 3.0, 10, 0.4)
    assert stop


def test_governor_speed_law_and_corridor():
    assert SpeedGovernor.speed_limit_kmh(7.0) == 0.0
    assert 20 < SpeedGovernor.speed_limit_kmh(25.0) < 21
    r = straight_road(80)
    a_in = Agent(1, KIND_STATIC, 20.0, 0.5, 0.0, 0, 0, 4.0, 1.8)
    a_out = Agent(2, KIND_STATIC, 20.0, 4.0, 0.0, 0, 0, 4.0, 1.8)
    d = corridor_distance(r, 0.0, 1.8, [a_in], 1.3, 25.0)
    assert 15.5 < d < 16.5
    assert np.isinf(corridor_distance(r, 0.0, 1.8, [a_out], 1.3, 25.0))


class _Track(Controller):
    name = "ltv"

    def __init__(self):
        self.m = LTVMPC(P)

    def reset(self, route, v_des, ego):
        self.route, self.v, self.u = route, v_des, np.zeros(2)

    def step(self, obs):
        N, DT = self.m.N, self.m.DT
        v = np.minimum(self.v, obs.ego.v + P.a_max * DT * (np.arange(N) + 1))
        s = obs.s_ego + np.cumsum(v * DT)
        x, y, yaw = self.route.pose(s)
        out = self.m.solve(obs.ego.z(), np.stack([x, y, yaw, v], 1), self.u, route=self.route, s_ref=s)
        self.u = out["u"][0]
        return Command(*self.u, out["solve_time"], out["ok"])


def test_ltv_mpc_tracks_curve():
    r = curved_road()
    sim = KinematicSim(r, [], P, v_des=4.0)
    log = run_episode(sim, _Track(), t_max=60)
    _, d = r.project_many(log["z"][:, :2])
    assert log["status"] == "success" and np.abs(d).max() < 0.15 and log["ok"].mean() > 0.95


def test_ltv_safety_halfplanes_push_away():
    """Reference straight through an obstacle: the half-planes must keep the predicted body clear."""
    m = LTVMPC(P, N=20)
    zref = np.stack([np.arange(1, 21) * 0.4, np.zeros(20), np.zeros(20), np.full(20, 4.0)], 1)
    obs = [Agent(1, KIND_STATIC, 6.0, -1.6, 0.0, 0, 0, 4.0, 1.8)]
    out_free = m.solve([0, 0, 0, 4.0], zref, np.zeros(2))
    out_safe = m.solve([0, 0, 0, 4.0], zref, np.zeros(2), agents=obs)
    assert out_safe["ok"]
    assert out_safe["z"][:, 1].max() > out_free["z"][:, 1].max() + 0.05


def test_clearance_standoff_ahead_vs_side():
    t = np.array([0.1])
    offs, r = P.ego_circles(3)
    ag = [Agent(1, KIND_WALKER, 6.0, 0.0, 0.0, 0, 0, 0.5, 0.5)]
    ahead = trajectory_clearance(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), t, ag, offs, r, 2.0)
    side = trajectory_clearance(np.full((1, 1), 6.0), np.full((1, 1), -6.0), np.zeros((1, 1)), t, ag, offs, r, 2.0)
    geo_ahead, _ = trajectory_clearance(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), t, ag, offs, r, 2.0,
                                        both=True)
    assert ahead[0] < geo_ahead[0] - 1.9 and side[0] > ahead[0]


@pytest.mark.parametrize("side", [0.4, -0.4])
def test_expert_passes_parked_car(side):
    from tmpc.expert.lattice import ExpertController, LatticeConfig
    r = straight_road(110)
    sim = KinematicSim(r, [Agent(1, KIND_STATIC, 45.0, 0.3, 0.0, 0, 0, 4.5, 1.9)], P, v_des=4.0)
    log = run_episode(sim, ExpertController(P, LatticeConfig(side_pref=side)), t_max=60)
    assert log["status"] == "success" and log["dist"].min() > 0.3


def test_expert_waits_for_crossing_walker():
    from tmpc.expert.lattice import ExpertController
    r = straight_road(110)
    walker = Agent(3, KIND_WALKER, 30.0, -5.0, np.pi / 2, 0, 1.0, 0.5, 0.5)
    sim = KinematicSim(r, [walker], P, v_des=4.0)
    log = run_episode(sim, ExpertController(P), t_max=60)
    assert log["status"] == "success" and log["dist"].min() > 0.5


def test_model_shapes_and_wta_loss():
    torch = pytest.importorskip("torch")
    from tmpc.planning.features import FeatureConfig, build_features, collate
    from tmpc.planning.model import ModelConfig, TrajectoryTransformer, wta_loss
    fc = FeatureConfig()
    r = straight_road(100)
    hist = np.stack([np.linspace(-4, 0, 11), np.zeros(11), np.zeros(11), np.full(11, 4.0)], 1)
    ag = np.array([[1, KIND_VEHICLE, 20.0, 0.5, 0.0, 1.0, 0.0, 4.5, 1.9]])
    f = build_features((0.0, 0.0, 0.0), 4.0, 0.0, 0.0, hist, ag, r, 0.0, 4.0, fc)
    assert f["hist"].shape == (11, 5) and f["agents"].shape == (16, 11) and f["agents_mask"].sum() == 1
    assert f["route"].shape == (25, 6)
    b = {k: torch.from_numpy(v) for k, v in collate([f, f]).items()}
    model = TrajectoryTransformer(ModelConfig())
    traj, logits = model(b["hist"], b["agents"], b["agents_mask"], b["route"], b["route_mask"], b["ctx"])
    assert traj.shape == (2, 6, 30, 2) and logits.shape == (2, 6)
    loss, parts = wta_loss(traj, logits, torch.zeros(2, 30, 2))
    loss.backward()
    assert torch.isfinite(loss)


def test_selector_rejects_colliding_candidate():
    from tmpc.planning.selector import CandidateSelector
    from tmpc.types import EgoState, Observation
    r = straight_road(100)
    ag = [Agent(1, KIND_STATIC, 12.0, 0.0, 0.0, 0, 0, 4.5, 1.9)]
    hist = np.tile([0, 0, 0, 4.0], (11, 1))
    obs = Observation(0.0, EgoState(0, 0, 0, 4.0), ag, r, 4.0, hist, 0.0, 0.0)
    t = 0.1 * (np.arange(30) + 1)
    straight = np.stack([4.0 * t, np.zeros(30)], 1)
    lat = np.clip((4.0 * t - 2.0) / 8.0, 0, 1)
    swerve = np.stack([4.0 * t, 3.4 * (10 * lat ** 3 - 15 * lat ** 4 + 6 * lat ** 5)], 1)
    sel = CandidateSelector(P).select(np.stack([straight, swerve]), np.array([2.0, 0.0]), obs)
    assert sel["mode"] == 1 and sel["fallback"] == 0


def test_reference_robust_to_jitter_near_standstill():
    t = 0.1 * (np.arange(30) + 1)
    x = np.where(t < 1.0, 0.0, 0.4 * (t - 1.0) ** 2) + np.random.default_rng(1).normal(0, 0.02, 30)
    ref = reference_from_xy(np.stack([x, np.zeros(30)], 1), 0.1, (0.0, 0.0), 0.0, 0.0)
    assert np.all(np.abs(ref[:, 2]) < 0.2)          # heading never flips backwards
    assert np.all(ref[:, 3] >= 0.0)                 # forward-only speed


def test_dead_time_compensation_advances_launch():
    from tmpc.planning.hybrid import HybridController
    t = 0.1 * (np.arange(30) + 1)
    x = np.where(t < 1.0, 0.0, 0.4 * (t - 1.0) ** 2)
    stay = np.zeros((30, 2))
    out = HybridController.compensate_dead_time(np.stack([np.stack([x, np.zeros(30)], 1), stay]), (0.0, 0.0))
    first = np.nonzero(out[0, :, 0] > 0.15)[0][0]
    orig = np.nonzero(x > 0.15)[0][0]
    assert first == orig - 10 and np.allclose(out[1], 0.0)   # advanced by the 1 s cap; a stop is kept
