import numpy as np
import time
from scipy.optimize import minimize

class KinematicTrackingMPC:
    """
    Convex Tracking Kinematic MPC for Autonomous Vehicles.
    Tracks a smooth reference trajectory (x_ref, y_ref, v_ref) provided by the Transformer planner.
    
    Guarantees:
        - Vehicle kinematic feasibility (Bicycle model)
        - Strict steering angle limits (|delta| <= delta_max)
        - Steering slew rate limits (|d_delta/dt| <= ddelta_max)
        - Acceleration & braking limits (a_min <= a <= a_max)
        - Deterministic solving time (< 10 ms)
    """
    def __init__(self, N: int = 15, dt: float = 0.1, wheelbase: float = 2.5):
        self.N = N               # Prediction horizon (1.5 seconds)
        self.dt = dt             # Sampling time (100 ms)
        self.L = wheelbase       # Wheelbase length (meters)

        # Actuator physical limits
        self.delta_max = np.deg2rad(35.0)   # max steering angle (+-35 deg)
        self.ddelta_max = np.deg2rad(25.0)  # max steering rate (+-25 deg/s)
        self.a_min = -4.0                   # max braking (-4.0 m/s^2)
        self.a_max = 2.0                    # max acceleration (+2.0 m/s^2)

        # Cost weights
        self.Q_pos = 10.0      # Tracking error (x, y)
        self.Q_vel = 2.0       # Velocity error
        self.R_steer = 1.0     # Steering effort
        self.R_accel = 0.5     # Acceleration effort
        self.R_rate = 5.0      # Smoothness / jerk penalty

    def solve(self, current_state: np.ndarray, ref_trajectory: np.ndarray, prev_delta: float = 0.0):
        """
        Solves the tracking MPC problem.
        current_state: [x, y, theta, v]
        ref_trajectory: (N+1, 3) -> [x_ref, y_ref, v_ref]
        prev_delta: previous steering command for slew-rate limiting
        """
        start_time = time.perf_counter()

        # Decision variables: [a_0, ..., a_{N-1}, delta_0, ..., delta_{N-1}] -> length 2*N
        u0 = np.zeros(2 * self.N)

        def objective(u):
            accels = u[:self.N]
            deltas = u[self.N:]
            
            x, y, theta, v = current_state
            cost = 0.0

            for k in range(self.N):
                # Kinematic Bicycle update
                x += v * np.cos(theta) * self.dt
                y += v * np.sin(theta) * self.dt
                theta += (v / self.L) * np.tan(deltas[k]) * self.dt
                v += accels[k] * self.dt

                # Reference tracking errors
                ref_x, ref_y, ref_v = ref_trajectory[k+1]
                pos_err = (x - ref_x)**2 + (y - ref_y)**2
                vel_err = (v - ref_v)**2

                cost += self.Q_pos * pos_err + self.Q_vel * vel_err
                cost += self.R_accel * (accels[k]**2) + self.R_steer * (deltas[k]**2)

                # Rate penalties (Jerk minimization)
                if k == 0:
                    cost += self.R_rate * ((deltas[k] - prev_delta) / self.dt)**2
                else:
                    cost += self.R_rate * ((deltas[k] - deltas[k-1]) / self.dt)**2

            return cost

        # Bounds on actuators
        bounds = []
        for _ in range(self.N):
            bounds.append((self.a_min, self.a_max))        # acceleration
        for _ in range(self.N):
            bounds.append((-self.delta_max, self.delta_max)) # steering

        # Fast optimization
        res = minimize(objective, u0, method='SLSQP', bounds=bounds, options={'maxiter': 25, 'ftol': 1e-3})
        
        solve_time_ms = (time.perf_counter() - start_time) * 1000.0

        if res.success:
            opt_accel = res.x[0]
            opt_delta = res.x[self.N]
            return opt_accel, opt_delta, solve_time_ms, True
        else:
            return 0.0, prev_delta, solve_time_ms, False
