"""Shared data types passed between simulators, planners and controllers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geometry import Agent, Path


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float
    a: float = 0.0     # last applied acceleration command
    df: float = 0.0    # last applied steering command (rad)

    def z(self):
        return np.array([self.x, self.y, self.yaw, self.v])


@dataclass
class Observation:
    t: float
    ego: EgoState
    agents: List[Agent]
    route: Path
    v_des: float
    history: np.ndarray            # (H, 4) world [x, y, yaw, v], oldest first, last == current
    s_ego: float = 0.0             # progress along the route (m)
    d_ego: float = 0.0             # lateral offset from the route (m)


@dataclass
class Command:
    a: float
    df: float
    solve_time: float = 0.0        # wall-clock planning + control time (s)
    ok: bool = True                # solver reported success
    plan: Optional[np.ndarray] = None   # (T, 2) world-frame planned xy (for logging)
    info: Dict = field(default_factory=dict)


class Controller:
    """Interface shared by all evaluated stacks."""

    name = "base"

    def reset(self, route: Path, v_des: float, ego: EgoState):
        pass

    def step(self, obs: Observation) -> Command:
        raise NotImplementedError
