"""Small, transparent forecasting adaptations of four frozen encoder families.

The encoder implementations come from framework_validation_v1_20261028/models.py.
They are research adaptations, not official reproductions of forecasting papers.
Every arm registers exactly the same parameters for a given backbone and seed.
"""
from __future__ import annotations

import hashlib
import torch
from torch import nn
from vendor_encoders import make_encoder, _prepare

MODEL_NAMES = ('tcn', 'gru', 'transformer', 'itransformer')
ARMS = ('P0', 'P1', 'P2', 'P3')


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode('utf-8'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ForecastModel(nn.Module):
    def __init__(self, backbone: str, history_steps: int = 12, hidden: int = 64):
        super().__init__()
        self.backbone_name = backbone
        self.history_steps = history_steps
        self.encoder = make_encoder(backbone, 9, history_steps, hidden)
        self.field_adapter = nn.Sequential(nn.Linear(8, hidden), nn.GELU(),
                                           nn.Linear(hidden, hidden))
        self.field_gain = nn.Parameter(torch.zeros(()))
        self.forecast_head = nn.Sequential(nn.LayerNorm(hidden + 1),
                                          nn.Linear(hidden + 1, hidden), nn.GELU(),
                                          nn.Linear(hidden, 2))
        nn.init.zeros_(self.forecast_head[-1].weight)
        nn.init.zeros_(self.forecast_head[-1].bias)

    def forward(self, x, field, rho, query_s, anchor, velocity, use_field: bool,
                times=None, mask=None):
        """Return absolute local metric coordinates [B,F,2].

        x/field are training-statistic normalized; rho is the unnormalized support.
        query_s contains requested future times, never future coordinates/velocities.
        anchor and velocity must originate from this arm's visible history.
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x, mask, times, lengths = _prepare(x, mask, times, 9, self.history_steps)
        embedding = self.encoder.encode(x, mask, times, lengths)
        if use_field:
            if field.shape != (*x.shape[:2], 8) or rho.shape != x.shape[:2]:
                raise ValueError('Field must be [B,H,8], rho [B,H]')
            support = rho.clamp(0., 1.) * mask.to(rho.dtype)
            context = (self.field_adapter(field) * support.unsqueeze(-1)).sum(1)
            context = context / mask.sum(1, keepdim=True).clamp_min(1)
            embedding = embedding + self.field_gain * context
        if query_s.ndim != 2 or bool((query_s <= 0).any()):
            raise ValueError('query_s must be positive [B,F] future time differences')
        features = torch.cat((embedding[:, None].expand(-1, query_s.shape[1], -1),
                              (query_s / 600.).unsqueeze(-1)), dim=-1)
        correction = 1000. * self.forecast_head(features)
        # Preserve sub-metre precision when the region projection has a large origin.
        return (anchor.to(torch.float64)[:, None, :]
                + velocity.to(torch.float64)[:, None, :] * query_s.to(torch.float64)[..., None]
                + correction.to(torch.float64))

