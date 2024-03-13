from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class CategoricalEncoder(nn.Module):
    def get_positional_embeddings(
        self, n_pos: int = 500, C: int = 10000, d: int = 20
    ) -> torch.Tensor:
        pos_embed = torch.zeros((n_pos, d))
        w = 1.0 / (C ** (2 * torch.arange(d) / d))
        for pos in range(n_pos):
            for i in range(d):
                if i % 2 == 0:
                    pos_embed[pos, i] = torch.sin(w[i // 2] * pos)
                else:
                    pos_embed[pos, i] = torch.cos(w[i // 2] * pos)
        return pos_embed

    def __init__(
        self,
        tot_vals: int,
        max_buckets: int,
        enc: Optional["CategoricalEncoder"] = None,
    ):
        super(CategoricalEncoder, self).__init__()
        self.tot_vals = tot_vals
        self.max_buckets = max_buckets
        self.encoder = nn.Sequential(
            nn.Linear(self.tot_vals, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.max_buckets),
        )
        # self.pos_embed = self.get_positional_embeddings(n_pos=tot_vals, d=50)
        if enc is not None:
            self.encoder.load_state_dict(enc.encoder.state_dict())
        self.enc_type = "disc"

    def forward(
        self, feat_x: torch.Tensor, temp: float = 1.0, use_hard: bool = False
    ) -> torch.Tensor:
        # feat_x = torch.matmul(feat_x, self.pos_embed)
        feat_z = self.encoder(feat_x)
        hard = F.one_hot(feat_z.max(dim=1)[1], num_classes=self.max_buckets).float()
        if temp is None:
            return hard
        soft = F.softmax(feat_z / temp, dim=1)
        if use_hard:
            return soft + hard - soft.detach()
        # return F.gumbel_softmax(feat_z, tau=temp, hard=use_hard)
        return soft

    def log_probs(
        self, feat_x: torch.Tensor, b: torch.Tensor, temp: float = 1.0
    ) -> torch.Tensor:
        # feat_x = torch.matmul(feat_x, self.pos_embed)
        feat_z = self.encoder(feat_x)
        log_probs = F.log_softmax(feat_z / temp, dim=1)
        return (
            log_probs * F.one_hot(b, num_classes=self.max_buckets).to(feat_x.device)
        ).sum(dim=1)

    def sample(
        self, feat_x: torch.Tensor, temp: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat_x = torch.matmul(feat_x, self.pos_embed)
        logits = self.encoder(feat_x) / temp
        dist = Categorical(logits=logits)
        ids = dist.sample()
        log_probs = dist.log_prob(ids)
        return ids, log_probs
