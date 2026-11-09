"""Source Transformer layout, preserving historical checkpoint parameter names."""

import torch
from torch import nn


class SourceTransformer(nn.Module):
    """The original 11-feature, CLS-pooled Transformer with padding-aware inference."""

    def __init__(self, width: int = 128, dropout: float = 0.2, layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(11, width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, 41, width))
        encoder_layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, values: torch.Tensor, valid_time: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(values)
        hidden = torch.cat([self.cls_token.expand(len(values), -1, -1), hidden], dim=1)
        padding = torch.cat(
            [torch.zeros(len(values), 1, dtype=torch.bool, device=values.device), ~valid_time],
            dim=1,
        )
        hidden = self.encoder(
            hidden + self.pos_embed[:, : hidden.shape[1]], src_key_padding_mask=padding
        )
        return self.classifier(hidden[:, 0]).squeeze(-1)
