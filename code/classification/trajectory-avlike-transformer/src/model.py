import torch
import torch.nn as nn


class VanillaTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_dim=12,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        seq_len=40,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 40, 12]
        B, T, C = x.shape

        x = self.input_proj(x)  # [B, 40, 128]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 128]
        x = torch.cat([cls, x], dim=1)  # [B, 41, 128]

        x = x + self.pos_embed[:, : T + 1, :]

        h = self.encoder(x)  # [B, 41, 128]

        h_cls = h[:, 0, :]  # [B, 128]
        logit = self.classifier(h_cls).squeeze(-1)  # [B]

        return logit


class TemporalConvClassifier(nn.Module):
    """Small local temporal encoder with mean/std pooling (variable window length)."""

    def __init__(self, input_dim=12, channels=32, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, channels, 5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * channels), nn.Dropout(dropout),
            nn.Linear(2 * channels, 1),
        )

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        pooled = torch.cat([h.mean(-1), h.std(-1, unbiased=False)], dim=-1)
        return self.classifier(pooled).squeeze(-1)


class PooledTrajectoryTransformer(nn.Module):
    """Compact 4-second encoder; all frames contribute to mean/std pooling."""

    def __init__(self, input_dim=12, d_model=32, num_heads=4, ffn_dim=64,
                 seq_len=40, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1,
                                             enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        h = self.encoder(self.input_proj(x) + self.pos_embed[:, :x.size(1)])
        pooled = torch.cat([h.mean(1), h.std(1, unbiased=False)], dim=-1)
        return self.classifier(pooled).squeeze(-1)
