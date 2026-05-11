from __future__ import annotations

import torch
import torch.nn as nn
import timm


class GatedLocalTokenMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.proj(x))
        return shortcut + x


class BiomassModel(nn.Module):
    def __init__(self, model_name: str, dropout: float, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        num_features = self.backbone.num_features
        self.fusion = nn.Sequential(
            GatedLocalTokenMixer(num_features, kernel_size=5, dropout=dropout),
            GatedLocalTokenMixer(num_features, kernel_size=5, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_green = self._make_head(num_features, dropout)
        self.head_dead = self._make_head(num_features, dropout)
        self.head_clover = self._make_head(num_features, dropout)

    @staticmethod
    def _make_head(num_features: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_features // 2, 1),
            nn.Softplus(),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_tokens = self.backbone(left)
        right_tokens = self.backbone(right)
        fused_tokens = self.fusion(torch.cat([left_tokens, right_tokens], dim=1))
        pooled = self.pool(fused_tokens.transpose(1, 2)).flatten(1)
        green = self.head_green(pooled)
        dead = self.head_dead(pooled)
        clover = self.head_clover(pooled)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)


# Backward-compatible alias for earlier experiment code and checkpoints/docs.
LocalMambaBlock = GatedLocalTokenMixer
