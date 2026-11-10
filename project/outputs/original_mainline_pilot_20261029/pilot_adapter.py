"""Experimental late residual adapter. Training and loss integration pending."""
import torch
from torch import nn

class FieldResidualAdapter(nn.Module):
    def __init__(self, backbone, hidden=64, field_dim=4):
        super().__init__()
        self.backbone = backbone
        self.field_projection = nn.Sequential(nn.Linear(field_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, self_motion, mask, times, field, rho):
        base = self.backbone(self_motion, mask, times)
        valid = mask.to(field.dtype)
        weighted = self.field_projection(field) * (rho * valid).unsqueeze(-1)
        contextual = weighted.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        embedding = base["embedding"] + self.scale * contextual
        return {"logits": self.backbone.head(embedding).squeeze(-1), "embedding": embedding}
