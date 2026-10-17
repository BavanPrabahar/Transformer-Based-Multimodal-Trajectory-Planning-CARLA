import numpy as np
import time
from scipy.optimize import minimize

class MPCCapsuleBaseline:
    """
    Classical Non-Linear MPC with Geometric Capsule Distance Constraints (mpc_capsule).
    
    This represents the baseline currently used in campus AVs:
    It attempts to simultaneously optimize trajectory tracking AND solve non-convex
    obstacle avoidance constraints within the same numerical solver.
    
    Known Vulnerabilities:
        - Severe non-convexity leading to local minima traps (Freezing Robot)
        - Solver divergence / failure when obstacle blocks centerline
        - High computational latency (O(N * M) constraint evaluations)
        - High-jerk swerving at boundary violation thresholds
    """
    def __init__(self, N: int = 15, dt: float = 0.1, wheelbase: float = 2.5):
        self.N = N
        self.dt = dt
        self.L = wheelbase

        # Physical and Safety dimensions (Capsule approximations)
        self.r_ego = 1.0           # Ego vehicle radius
        self.r_obs = 1.0           # Obstacle radius
        self.d_safe = 0.8          # Safety margin buffer
        self.min_dist = self.r_ego + self.r_obs + self.d_safe # Total required clearance (2.8m)

        # Actuator limits
        self.delta_max = np.deg2rad(35.0)
        self.a_min = -4.0
        self.a_max = 2.0

        # Weights
        self.Q_pos = 10.0
        self.R_steer = 1.0
        self.R_accel = 0.5
        self.W_obstacle = 1000.0   # Large penalty for constraint violation

    def solve(self, current_state: np.ndarray, nominal_path: np.ndarray, obstacles: list):
        """
        Solves the classical MPC problem with non-convex capsule collision constraints.
        current_state: [x, y, theta, v]
        nominal_path: (N+1, 2) -> nominal centerline coordinates
        obstacles: list of [x_obs, y_obs, r_obs]
        """
        start_time = time.perf_counter()
        u0 = np.zeros(2 * self.N)

        def objective(u):
            accels = u[:self.N]
            deltas = u[self.N:]
            
            x, y, theta, v = current_state
            cost = 0.0

            for k in range(self.N):
                # Bicycle model kinematics
                x += v * np.cos(theta) * self.dt
                y += v * np.sin(theta) * self.dt
                theta += (v / self.L) * np.tan(deltas[k]) * self.dt
                v += accels[k] * self.dt

                # Tracking nominal centerline
                ref_x, ref_y = nominal_path[k+1]
                cost += self.Q_pos * ((x - ref_x)**2 + (y - ref_y)**2)
                cost += self.R_accel * (accels[k]**2) + self.R_steer * (deltas[k]**2)

                # NON-CONVEX CAPSULE DISTANCE PENALTY:
                for (ox, oy, r_o) in obstacles:
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    clearance = dist - (self.r_ego + r_o + self.d_safe)
                    if clearance < 0:
                        # Heavy penalty / soft-constraint violation
                        cost += self.W_obstacle * (clearance**2)

            return cost

        bounds = []
        for _ in range(self.N):
            bounds.append((self.a_min, self.a_max))
        for _ in range(self.N):
            bounds.append((-self.delta_max, self.delta_max))

        # Solving non-convex optimization (frequently hits local minima or requires many iterations)
        res = minimize(objective, u0, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-3})
        solve_time_ms = (time.perf_counter() - start_time) * 1000.0

        if res.success:
            return res.x[0], res.x[self.N], solve_time_ms, True
        else:
            return 0.0, 0.0, solve_time_ms, False
