from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from .monotone_linear import MonotoneLinear


class ContinuousEncoder(nn.Module):
    def __init__(self, max_buckets: int, feature_vals: np.ndarray):
        super(ContinuousEncoder, self).__init__()
        self.max_buckets = max_buckets
        self.centers = torch.linspace(0, 1, max_buckets + 1)[:max_buckets] + 1.0 / (
            2 * max_buckets
        )

        self.encoder = nn.Sequential(
            MonotoneLinear(1, 100, 1),
            nn.Tanh(),  # Sigm?
            MonotoneLinear(100, 100, 1),
            nn.BatchNorm1d((100,), affine=False),
            nn.Tanh(),
            MonotoneLinear(100, 1, 1),
            nn.BatchNorm1d((1,), affine=False),
            nn.Sigmoid(),
        )

        # Warmup: vals -> quantiles
        quantiles = np.linspace(0, 1, 1000)
        feature_val_tensor = (
            torch.tensor(np.quantile(feature_vals, quantiles)).float().unsqueeze(-1)
        )
        quant_tensor = torch.tensor(quantiles).float()

        opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        print("Warming up the encoder...")

        pbar = tqdm(range(2000))
        for _ in pbar:
            z: torch.Tensor = self.encoder(feature_val_tensor).ravel()
            loss = ((z - quant_tensor) ** 2).mean()
            pbar.set_description(f"{loss}")

            loss.backward()
            opt.step()

    def compute_distance(
        self, feat_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_z = self.encoder(feat_x)
        d = (self.centers.unsqueeze(0).to(feat_z.device) - feat_z) ** 2
        return d, feat_z

    def forward(
        self, feat_x: torch.Tensor, temp: float = 1.0, use_hard: bool = False
    ) -> torch.Tensor:
        d, z = self.compute_distance(feat_x)
        hard = F.one_hot(d.min(dim=1)[1], num_classes=self.max_buckets).float()
        if temp is None:
            return hard
        # return F.gumbel_softmax(-d, tau=temp, hard=use_hard)
        soft = F.softmax(-d / temp, dim=1)
        if use_hard:
            return soft + hard - soft.detach()
        return soft

    def log_probs(
        self, feat_x: torch.Tensor, b: torch.Tensor, temp: float = 1.0
    ) -> torch.Tensor:
        d, _ = self.compute_distance(feat_x)
        log_probs = F.log_softmax(-d / temp, dim=1)
        return (
            log_probs * F.one_hot(b, num_classes=self.max_buckets).to(feat_x.device)
        ).sum(dim=1)

    def sample(
        self, feat_x: torch.Tensor, temp: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d, _ = self.compute_distance(feat_x)
        dist = Categorical(logits=-d / temp)
        ids = dist.sample()
        log_probs = dist.log_prob(ids)
        return ids, log_probs
