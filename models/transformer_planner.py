import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TrajectoryTransformer(nn.Module):
    """
    Multimodal Trajectory Transformer Planner for Autonomous Vehicles.
    
    Inputs:
        - ego_past: (Batch, 20, 4) -> past 20 timesteps of (x, y, v, heading)
        - obstacles: (Batch, M, 6) -> M detected YOLO dynamic obstacles (x, y, vx, vy, length, width)
        - goal: (Batch, 2) -> target local coordinate (xg, yg)
        
    Outputs:
        - multimodal_trajectories: (Batch, K=3, 30, 3) -> 3 distinct candidate trajectories (x, y, v)
        - mode_probs: (Batch, K=3) -> confidence probability for each mode
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, num_modes: int = 3, future_steps: int = 30):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.future_steps = future_steps

        # 1. Encoders
        self.ego_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=50)

        self.obstacle_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. Transformer Context Fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Multimodal Decoders (One per candidate driving mode: Overtake, Follow, Nudge)
        self.mode_queries = nn.Parameter(torch.randn(num_modes, d_model))
        
        self.trajectory_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Linear(256, future_steps * 3) # outputs (x, y, v) for all future steps
            ) for _ in range(num_modes)
        ])

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
            nn.Softmax(dim=-1)
        )

    def forward(self, ego_past: torch.Tensor, obstacles: torch.Tensor, goal: torch.Tensor):
        B = ego_past.shape[0]

        # 1. Feature Projections
        h_ego = self.pos_encoder(self.ego_encoder(ego_past)) # (B, 20, d_model)
        h_obs = self.obstacle_encoder(obstacles)             # (B, M, d_model)
        h_goal = self.goal_encoder(goal).unsqueeze(1)        # (B, 1, d_model)

        # 2. Concatenate into Spatial-Temporal Scene Tokens
        scene_tokens = torch.cat([h_ego, h_obs, h_goal], dim=1) # (B, 20 + M + 1, d_model)

        # 3. Self-Attention Scene Reasoning
        context_memory = self.transformer_encoder(scene_tokens) # (B, SeqLen, d_model)
        global_context = context_memory.mean(dim=1)             # (B, d_model)

        # 4. Generate Multimodal Trajectories
        trajectories = []
        for m in range(self.num_modes):
            # Query specific mode conditioning
            mode_query = self.mode_queries[m].unsqueeze(0).repeat(B, 1) # (B, d_model)
            combined = global_context + mode_query
            raw_path = self.trajectory_heads[m](combined)               # (B, future_steps * 3)
            path = raw_path.view(B, self.future_steps, 3)               # (B, future_steps, 3)
            trajectories.append(path)

        # Shape: (B, num_modes, future_steps, 3)
        multimodal_trajectories = torch.stack(trajectories, dim=1)
        mode_probs = self.confidence_head(global_context)               # (B, num_modes)

        return multimodal_trajectories, mode_probs

if __name__ == '__main__':
    # Sanity check
    model = TrajectoryTransformer()
    B = 2
    dummy_ego_past = torch.randn(B, 20, 4)
    dummy_obs = torch.randn(B, 5, 6)
    dummy_goal = torch.randn(B, 2)

    trajs, probs = model(dummy_ego_past, dummy_obs, dummy_goal)
    print(f"Model forward successful!")
    print(f"Output Trajectories shape: {trajs.shape} (Batch, Modes, Steps, [x, y, v])")
    print(f"Mode Probabilities shape: {probs.shape}")
