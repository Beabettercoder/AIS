"""最小可跑模型骨架（PyTorch）。

只演示四条通路的张量形状与梯度连通，不是最终网络：

1. 逐报文观测门控 w ∈ [0,1]（观测头独立监督，不参与行为一致性）；
2. 物理态势引导的通道统计量扰动 PSA（均值/方差沿 (邻船, 时间) 统计轴，stop-grad）；
3. 干净 / 退化双分支，仅对行为概率施加 JS 一致性；
4. 绝对距离 + 相对残差距离的双路校准融合，再做邻船图聚合。

真正的编码器请换成你 TGSA 里的主干；本文件只保证接口和维度正确。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_moments(h, mask):
    """沿 (邻船, 时间) 求通道均值/标准差，mask: (B,N,L) bool。"""
    m = mask.unsqueeze(-1).to(h.dtype)
    n = m.sum(dim=(1, 2)).clamp_min(1.0)  # (B,1)
    mu = (h * m).sum(dim=(1, 2)) / n
    var = (((h - mu[:, None, None, :]) ** 2) * m).sum(dim=(1, 2)) / n
    return mu, var.clamp_min(1e-10).sqrt()


def psa(h, mask, eps):
    """渐进式统计量扰动：h' = mu + sd * ((1+eps_a) * norm(h) + eps_b)。

    mu/sd 全部 stop-grad：扰动只能改变特征分布，不能通过压低统计量来偷懒。
    eps: (B,N,2)，已由控制器约束在预算内。
    """
    mu, sd = masked_moments(h, mask)
    mu_d, sd_d = mu.detach(), sd.detach()
    hn = (h - mu_d[:, None, None, :]) / sd_d[:, None, None, :]
    a = eps[..., 0][:, :, None, None]
    b = eps[..., 1][:, :, None, None]
    return mu_d[:, None, None, :] + sd_d[:, None, None, :] * (hn * (1.0 + a) + b)


def js_divergence(p, q, eps=1e-8):
    """两个 categorical 分布之间的 JS 散度，返回 (B,N)。"""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class AISNet(nn.Module):
    def __init__(self, f_mot=8, f_obs=7, f_rel=4, hidden=64, n_cls=4,
                 eps_max=0.3, tau=0.5):
        super().__init__()
        self.eps_max = eps_max
        self.tau = tau

        self.obs_gate = nn.Sequential(nn.Linear(f_obs, 32), nn.GELU(), nn.Linear(32, 1))
        self.enc = nn.GRU(f_mot, hidden, batch_first=True)

        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

        self.ctrl = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 2), nn.Tanh())
        self.rel_proj = nn.Linear(f_rel, hidden)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.cls = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, n_cls)
        )

    def encode(self, x_mot):
        b, n, l, f = x_mot.shape
        h, _ = self.enc(x_mot.reshape(b * n, l, f))
        return h.reshape(b, n, l, -1)  # (B,N,L,H)

    def attention(self, x, ship_mask):
        q, k, v = self.q(x), self.k(x), self.v(x)
        s = q @ k.transpose(1, 2) / math.sqrt(q.size(-1))
        s = s.masked_fill(~ship_mask[:, None, :], -1e9)
        return s.softmax(-1) @ v

    def fuse(self, h_abs, h_rel):
        d_abs = torch.cdist(h_abs, h_abs)
        d_rel = torch.cdist(h_rel, h_rel)
        s = torch.tanh(self.alpha * d_abs) + torch.tanh(self.beta * d_rel)
        w = torch.softmax(-s / self.tau, dim=-1)
        return w @ h_abs

    def forward(self, x_mot, x_obs, mask, ctx, x_rel, ship_mask, adv=False):
        """x_mot (B,N,L,Fm), x_obs (B,N,L,Fo), mask (B,N,L),
        ctx (B,N,4), x_rel (B,N,F_rel), ship_mask (B,N)。"""
        obs_logit = self.obs_gate(x_obs).squeeze(-1)            # (B,N,L)
        w = torch.sigmoid(obs_logit) * mask                     # (B,N,L) 可信度

        h = self.encode(x_mot * mask.unsqueeze(-1))             # (B,N,L,H)
        eps = self.eps_max * self.ctrl(ctx)                     # (B,N,2)
        h_in = psa(h, mask, eps) if adv else h

        # 可信度只用于加权，stop-grad：行为损失不能反过来"刷高"观测评分
        w_sg = w.detach()
        denom = w_sg.sum(-1, keepdim=True).clamp_min(1e-3)
        h_pool = (h_in * w_sg.unsqueeze(-1)).sum(2) / denom     # (B,N,H)

        ctx_vec = self.attention(h_pool, ship_mask)
        graph = self.fuse(h_pool, self.rel_proj(x_rel))

        logits = self.cls(torch.cat([h_pool, ctx_vec, graph], dim=-1))  # (B,N,C)
        return {"logits": logits, "obs_logit": obs_logit, "w": w, "eps": eps}


def behavior_loss(logits, y, weight=None):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), weight=weight)


def obs_loss(obs_logit, w_target, mask):
    """w_target=1 表示该条报文可信（未被污染）。"""
    bce = F.binary_cross_entropy_with_logits(obs_logit, w_target.float(), reduction="none")
    return (bce * mask).sum() / mask.sum().clamp_min(1.0)


def consistency_loss(logits_clean, logits_adv):
    p = torch.softmax(logits_clean, -1)
    q = torch.softmax(logits_adv, -1)
    return js_divergence(p, q).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
