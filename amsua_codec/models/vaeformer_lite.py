"""VAEformer Lite model definition."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, gh: int, gw: int, device: torch.device) -> torch.Tensor:
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    dim_each = half // 2
    omega = torch.arange(dim_each, device=device, dtype=torch.float32) / dim_each
    omega = 1.0 / (10000 ** omega)

    y = torch.arange(gh, device=device, dtype=torch.float32)
    x = torch.arange(gw, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    out_y = yy * omega.reshape(1, -1)
    out_x = xx * omega.reshape(1, -1)

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pos = torch.cat([pos_y, pos_x], dim=1)
    return pos.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


def gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gaussian_likelihood(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    scale = torch.clamp(scale, min=1e-6)
    centered = x - mean
    upper = (centered + 0.5) / scale
    lower = (centered - 0.5) / scale
    probs = gaussian_cdf(upper) - gaussian_cdf(lower)
    return torch.clamp(probs, min=eps)


def quantize(x: torch.Tensor, training: bool) -> torch.Tensor:
    if training:
        return x + torch.empty_like(x).uniform_(-0.5, 0.5)
    return torch.round(x)


class VAEformerLite(nn.Module):
    def __init__(self, in_channels=40, patch_size=16, embed_dim=256, latent_dim=192, depth=6, heads=8):
        super().__init__()
        groups = 8 if in_channels >= 8 else 1
        self.in_norm = nn.GroupNorm(groups, in_channels)

        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, in_channels, patch_size, stride=patch_size)

        self.enc = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.dec = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])

        self.to_y = nn.Conv2d(embed_dim, latent_dim, 1)
        self.y_to_embed = nn.Conv2d(latent_dim, embed_dim, 1)

        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, 2 * latent_dim, 3, padding=1),
        )
        self.z_log_scale = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor, compute_likelihood: bool):
        B, _, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError(f"H,W={H,W} not divisible by patch_size={ps}")

        x = self.in_norm(x)

        e = self.patch_embed(x)
        Hp, Wp = e.shape[-2], e.shape[-1]
        tok = e.flatten(2).transpose(1, 2)
        pos = get_2d_sincos_pos_embed(tok.shape[-1], Hp, Wp, x.device)
        tok = tok + pos
        for blk in self.enc:
            tok = blk(tok)
        e2 = tok.transpose(1, 2).reshape(B, -1, Hp, Wp)

        y = self.to_y(e2)
        y_hat = quantize(y, self.training)

        d = self.y_to_embed(y_hat)
        dtok = d.flatten(2).transpose(1, 2) + pos
        for blk in self.dec:
            dtok = blk(dtok)
        d2 = dtok.transpose(1, 2).reshape(B, -1, Hp, Wp)
        x_hat = self.patch_unembed(d2)

        if not compute_likelihood:
            return {"x_hat": x_hat}

        z = self.h_a(y_hat)
        z_hat = quantize(z, self.training)
        params = self.h_s(z_hat)
        mean_y, log_scale_y = params.chunk(2, dim=1)
        scale_y = F.softplus(log_scale_y) + 1e-6

        y_lik = gaussian_likelihood(y_hat.float(), mean_y.float(), scale_y.float())
        z_scale = (F.softplus(self.z_log_scale).float()[None, :, None, None] + 1e-6)
        z_lik = gaussian_likelihood(z_hat.float(), torch.zeros_like(z_hat).float(), z_scale)

        return {"x_hat": x_hat, "y_likelihood": y_lik, "z_likelihood": z_lik}


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def configure_stage_trainables(model: VAEformerLite, stage: str) -> None:
    """
    pretrain：只训练主干，冻结 hyperprior
    finetune_entropy：冻结主干，只训练 hyperprior
    """
    if stage == "pretrain":
        set_requires_grad(model.in_norm, True)
        set_requires_grad(model.patch_embed, True)
        set_requires_grad(model.patch_unembed, True)
        set_requires_grad(model.enc, True)
        set_requires_grad(model.dec, True)
        set_requires_grad(model.to_y, True)
        set_requires_grad(model.y_to_embed, True)

        set_requires_grad(model.h_a, False)
        set_requires_grad(model.h_s, False)
        model.z_log_scale.requires_grad_(False)

    elif stage == "finetune_entropy":
        set_requires_grad(model.in_norm, False)
        set_requires_grad(model.patch_embed, False)
        set_requires_grad(model.patch_unembed, False)
        set_requires_grad(model.enc, False)
        set_requires_grad(model.dec, False)
        set_requires_grad(model.to_y, False)
        set_requires_grad(model.y_to_embed, False)

        set_requires_grad(model.h_a, True)
        set_requires_grad(model.h_s, True)
        model.z_log_scale.requires_grad_(True)
    else:
        raise ValueError(stage)
