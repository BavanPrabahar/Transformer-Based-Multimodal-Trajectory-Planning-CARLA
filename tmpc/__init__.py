"""Transformer-guided MPC for low-speed autonomous driving (low-speed campus vehicle).

Sub-packages
------------
controllers  Lab MPC port, capsule-constrained NMPC baseline, convex LTV-MPC tracker.
planning     Feature builder, multimodal trajectory transformer, candidate selector, hybrid stack.
expert       Privileged Frenet-lattice expert used to generate demonstrations.
sim          CARLA environment/scenarios, a CARLA-free kinematic simulator, metrics.
data         Episode storage and PyTorch dataset.
"""

__version__ = "0.2.0"
