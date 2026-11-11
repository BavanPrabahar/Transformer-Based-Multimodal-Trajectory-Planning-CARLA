### Main comparison (80 paired held-out scenarios, CARLA Town02)

| Controller | n | Success (95% CI) | Collision | Freeze/timeout | Off-road | Route progress | Time to goal (s, median) | Min gap (m, median) | RMS jerk | RMS lat. acc. | Compute mean / p99 / max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lab MPC stack (MPC_capsule) | 80 | **31%** (22–42) | 2% | 66% | 0% | 55% | 55.6 | 1.96 | 2.58 | 0.35 | 8.1 / 26.4 / 133 |
| Capsule-constraint NMPC | 80 | **29%** (20–39) | 8% | 64% | 0% | 52% | 46.2 | 0.88 | 0.89 | 0.53 | 76.9 / 113.5 / 210 |
| Transformer + selector + LTV-MPC (as designed) | 80 | **48%** (37–58) | 2% | 50% | 0% | 67% | 46.4 | 1.23 | 2.09 | 0.71 | 11.9 / 21.5 / 61 |
| Transformer top-1 + safety LTV-MPC (recommended) | 80 | **71%** (61–80) | 2% | 26% | 0% | 83% | 48.0 | 1.02 | 2.41 | 0.71 | 8.4 / 16.1 / 191 |
| Privileged lattice expert (oracle) | 80 | **76%** (66–84) | 2% | 21% | 0% | 87% | 48.2 | 1.10 | 2.68 | 0.71 | 19.2 / 26.0 / 1150 |

### Ablations (80 paired scenarios)

| Controller | n | Success (95% CI) | Collision | Freeze/timeout | Off-road | Route progress | Time to goal (s, median) | Min gap (m, median) | RMS jerk | RMS lat. acc. | Compute mean / p99 / max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer + selector + LTV-MPC (as designed) | 80 | **48%** (37–58) | 2% | 50% | 0% | 67% | 46.4 | 1.23 | 2.09 | 0.71 | 11.9 / 21.5 / 61 |
| Transformer top-1 + safety LTV-MPC (recommended) | 80 | **71%** (61–80) | 2% | 26% | 0% | 83% | 48.0 | 1.02 | 2.41 | 0.71 | 8.4 / 16.1 / 191 |
| Hybrid w/o safety half-planes | 80 | **46%** (36–57) | 2% | 51% | 0% | 67% | 47.8 | 1.22 | 2.22 | 0.60 | 10.8 / 19.4 / 38 |

### Success rate per scenario template

| Template | Lab MPC stack (MPC_capsule) | Capsule-constraint NMPC | Transformer + selector + LTV-MPC (as designed) | Transformer top-1 + safety LTV-MPC (recommended) | Privileged lattice expert (oracle) |
|---|---|---|---|---|---|
| blocked | 0% (10) | 30% (10) | 60% (10) | 50% (10) | 90% (10) |
| blocked_oncoming | 0% (10) | 0% (10) | 20% (10) | 50% (10) | 70% (10) |
| cones | 0% (10) | 10% (10) | 70% (10) | 80% (10) | 80% (10) |
| leader | 100% (10) | 60% (10) | 60% (10) | 70% (10) | 80% (10) |
| mixed | 0% (10) | 30% (10) | 40% (10) | 90% (10) | 70% (10) |
| slalom | 0% (10) | 50% (10) | 10% (10) | 50% (10) | 50% (10) |
| traffic | 70% (10) | 10% (10) | 50% (10) | 90% (10) | 80% (10) |
| walker | 80% (10) | 40% (10) | 70% (10) | 90% (10) | 90% (10) |
