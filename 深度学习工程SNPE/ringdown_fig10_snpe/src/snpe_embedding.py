from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class FFTResNetEmbedding(nn.Module):
    """Lightweight 1D-ResNet embedding for whitened FFT features."""

    def __init__(
        self,
        input_dim: int,
        out_features: int = 128,
        base_channels: int = 32,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if input_dim < 8:
            raise ValueError(f"input_dim too small for conv embedding: {input_dim}")
        if out_features < 4:
            raise ValueError(f"out_features must be >= 4, got {out_features}")
        if base_channels < 4:
            raise ValueError(f"base_channels must be >= 4, got {base_channels}")

        k = int(kernel_size)
        p = k // 2
        c1 = int(base_channels)
        c2 = int(2 * base_channels)
        c3 = int(4 * base_channels)

        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k, padding=p),
            nn.GELU(),
            _ResBlock1D(c1, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
            _ResBlock1D(c2, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.GELU(),
            _ResBlock1D(c3, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c3, int(out_features)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (batch, features), got {tuple(x.shape)}")
        return self.net(x.unsqueeze(1))


def build_embedding_net(input_dim: int, estimator_cfg: dict) -> nn.Module:
    emb_cfg = estimator_cfg.get("embedding", {})
    emb_type = str(emb_cfg.get("type", "identity")).strip().lower()
    if emb_type in {"", "none", "identity"}:
        return nn.Identity()
    if emb_type == "resnet1d":
        return FFTResNetEmbedding(
            input_dim=int(input_dim),
            out_features=int(emb_cfg.get("out_features", 128)),
            base_channels=int(emb_cfg.get("base_channels", 32)),
            kernel_size=int(emb_cfg.get("kernel_size", 7)),
        )
    raise ValueError(f"Unsupported estimator.embedding.type={emb_type!r}")
