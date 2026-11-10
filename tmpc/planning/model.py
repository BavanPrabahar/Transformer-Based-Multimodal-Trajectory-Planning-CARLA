"""Multimodal trajectory transformer (scene encoder + mode-query decoder).

Compared with the original single-file model this version
* encodes the route/drivable area (the old model had no map input at all),
* uses typed tokens and key-padding masks so the number of agents can vary,
* decodes K learned mode queries with cross-attention to the scene memory
  (instead of mean-pooling + per-mode MLPs),
* predicts per-step displacements that are integrated (smooth, starts at the ego),
* returns raw logits (no softmax inside the model) for a proper cross-entropy,
* is trained with winner-takes-all regression + mode classification.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureConfig


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    enc_layers: int = 3
    dec_layers: int = 2
    num_modes: int = 6
    future_steps: int = 30
    dropout: float = 0.1
    hist_len: int = 11
    route_points: int = 25
    delta_scale: float = 0.5   # metres per 0.1 s step at unit network output

    def to_dict(self):
        return asdict(self)


def mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class TrajectoryTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.hist_enc = mlp(cfg.hist_len * 5, 2 * d, d, cfg.dropout)
        self.ctx_enc = mlp(5, d, d)
        self.agent_enc = mlp(11, d, d)
        self.route_enc = mlp(6, d, d)
        self.route_pos = nn.Parameter(torch.zeros(cfg.route_points, d))
        self.type_emb = nn.Embedding(4, d)
        enc = nn.TransformerEncoderLayer(d, cfg.nhead, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg.enc_layers, enable_nested_tensor=False)
        self.mode_queries = nn.Parameter(torch.randn(cfg.num_modes, d) * 0.1)
        dec = nn.TransformerDecoderLayer(d, cfg.nhead, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.dec_layers)
        self.traj_head = mlp(d, 2 * d, cfg.future_steps * 2)
        self.score_head = mlp(d, d, 1)
        nn.init.normal_(self.route_pos, std=0.02)

    def forward(self, hist, agents, agents_mask, route, route_mask, ctx, hist_dropout: float = 0.0):
        B = hist.shape[0]
        if self.training and hist_dropout > 0:
            # drop the past (keep the current pose) to fight the copycat problem
            keep = (torch.rand(B, 1, 1, device=hist.device) > hist_dropout).float()
            cur = hist[:, -1:, :]
            hist = hist * keep + cur * (1 - keep)
        h_tok = self.hist_enc(hist.reshape(B, -1)).unsqueeze(1) + self.type_emb.weight[0]
        c_tok = self.ctx_enc(ctx).unsqueeze(1) + self.type_emb.weight[1]
        r_tok = self.route_enc(route) + self.route_pos.unsqueeze(0) + self.type_emb.weight[2]
        a_tok = self.agent_enc(agents) + self.type_emb.weight[3]
        tokens = torch.cat([h_tok, c_tok, r_tok, a_tok], 1)
        valid = torch.cat([torch.ones(B, 2, dtype=torch.bool, device=hist.device), route_mask, agents_mask], 1)
        pad = ~valid
        mem = self.encoder(tokens, src_key_padding_mask=pad)
        q = self.mode_queries.unsqueeze(0).expand(B, -1, -1) + c_tok
        out = self.decoder(q, mem, memory_key_padding_mask=pad)
        K, T = self.cfg.num_modes, self.cfg.future_steps
        deltas = self.traj_head(out).view(B, K, T, 2) * self.cfg.delta_scale
        traj = torch.cumsum(deltas, dim=2)
        logits = self.score_head(out).squeeze(-1)
        return traj, logits


def wta_loss(traj, logits, target, w_cls=0.5, final_w=0.5):
    """Winner-takes-all: regress only the closest mode, classify which mode wins.

    traj (B, K, T, 2), logits (B, K), target (B, T, 2) metres.
    """
    err = torch.linalg.norm(traj - target.unsqueeze(1), dim=-1)          # (B, K, T)
    ade = err.mean(-1)
    fde = err[..., -1]
    score = ade + final_w * fde
    best = score.argmin(1)
    idx = torch.arange(traj.shape[0], device=traj.device)
    reg = F.smooth_l1_loss(traj[idx, best], target, reduction="none").sum(-1).mean(-1).mean()
    cls = F.cross_entropy(logits, best)
    return reg + w_cls * cls, dict(reg=reg.detach(), cls=cls.detach(),
                                   min_ade=ade.min(1).values.mean().detach(),
                                   min_fde=fde.gather(1, best[:, None]).mean().detach())


@torch.no_grad()
def metrics(traj, logits, target, miss_thresh=2.0):
    err = torch.linalg.norm(traj - target.unsqueeze(1), dim=-1)
    ade = err.mean(-1)
    fde = err[..., -1]
    top = logits.argmax(1)
    idx = torch.arange(traj.shape[0], device=traj.device)
    return dict(minADE=ade.min(1).values, minFDE=fde.min(1).values, ADE1=ade[idx, top], FDE1=fde[idx, top],
                miss=(fde.min(1).values > miss_thresh).float())


def save_checkpoint(path, model, feat_cfg: FeatureConfig, extra=None):
    torch.save(dict(state_dict=model.state_dict(), model_cfg=model.cfg.to_dict(), feat_cfg=feat_cfg.to_dict(),
                    extra=extra or {}), path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = TrajectoryTransformer(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, FeatureConfig(**ck["feat_cfg"]), ck.get("extra", {})
