# 📚 Key Research Papers & Theoretical References

This directory contains key research papers that form the theoretical foundation of this repository, specifically contrasting **deep generative/transformer planning** against **classical non-convex MPC collision avoidance**.

---

### 1. `Trajectory_Transformer.pdf`
* **Title:** *Offline Reinforcement Learning as One Big Sequence Modeling Problem* (Trajectory Transformer)
* **Authors:** Michael Janner, Qiyang Li, Sergey Levine (UC Berkeley, NeurIPS 2021)
* **Relevance:** Establishes the paradigm of formulating trajectory generation, dynamics modeling, and beam-search path proposals as sequence-to-sequence Transformer modeling.

---

### 2. `TrajDeformer_Trajectory_Diffusion.pdf`
* **Title:** *TrajDeformer: Multimodal Trajectory Generation & Diffusion for Autonomous Driving*
* **Relevance:** Demonstrates how generative models and diffusion transformers naturally resolve multimodality (overtake left, follow, nudge right) in congested urban bottlenecks where single-mode planners fail.

---

### 3. `Safety_Aware_Motion_Planning_MPC.pdf`
* **Title:** *Safety-Aware Motion Planning and Control via Model Predictive Control with Control Barrier Functions*
* **Relevance:** Highlights the limitations of classical distance/capsule constraints in non-convex optimization, and the need for decoupled, convex tracking controllers.

---

### 4. Classical Capsule-MPC Baseline References
* **Bicycle Model Kinematics & Obstacle Constraints:** J. Kong et al., *"Kinematic and dynamic vehicle models for autonomous driving control design"*, IEEE IV.
* **Non-Convex Trapping in NLP Collision Avoidance:** E. Schouwenaars et al., *"Receding horizon path planning for mobile robots with obstacle avoidance"*, IEEE CDC.
