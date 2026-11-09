"""Small trajectory encoders with a shared representation for SSL and classification."""
import torch
from torch import nn


class TrajectoryNet(nn.Module):
    def __init__(self, input_dim, architecture='transformer', width=128, dropout=.2,
                 layers=2, seq_len=40):
        super().__init__()
        self.architecture = architecture
        self.seq_len = seq_len
        if architecture in ('transformer', 'pooled_transformer', 'conv_transformer'):
            self.input_proj = nn.Linear(input_dim, width)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len+1, width))
            layer = nn.TransformerEncoderLayer(width, 4, width*4, dropout,
                activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.local_conv = (nn.Conv1d(width, width, 5, padding=2)
                               if architecture == 'conv_transformer' else None)
            self.representation_dim = width if architecture == 'transformer' else 2*width
            nn.init.trunc_normal_(self.cls_token, std=.02)
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        elif architecture == 'multiscale_cnn':
            self.branches = nn.ModuleList([nn.Sequential(
                nn.Conv1d(input_dim, width, k, padding=k//2), nn.GELU(),
                nn.Dropout(dropout), nn.Conv1d(width, width, 3, padding=1), nn.GELU())
                for k in (3, 5, 11)])
            self.representation_dim = width*6
        elif architecture == 'gru':
            self.encoder = nn.GRU(input_dim, width, batch_first=True,
                                  num_layers=layers, dropout=dropout if layers > 1 else 0)
            self.representation_dim = width*2
        elif architecture == 'tcn':
            self.input_proj = nn.Linear(input_dim, width)
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2*d, dilation=d), nn.GELU(),
                nn.Dropout(dropout), nn.Conv1d(width, width, 3, padding=d, dilation=d))
                for d in (1, 2, 4)])
            self.representation_dim = width*2
        elif architecture == 'statistics':
            self.representation_dim = input_dim*4
        else:
            raise ValueError(architecture)
        self.classifier = nn.Sequential(nn.LayerNorm(self.representation_dim),
            nn.Linear(self.representation_dim, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1))

    @staticmethod
    def pool(h):
        return torch.cat([h.mean(1), h.std(1, unbiased=False)], dim=-1)

    def encode(self, x):
        if self.architecture in ('transformer', 'pooled_transformer', 'conv_transformer'):
            h = self.input_proj(x)
            if self.local_conv is not None:
                h = h + self.local_conv(h.transpose(1, 2)).transpose(1, 2)
            h = torch.cat([self.cls_token.expand(len(x), -1, -1), h], dim=1)
            h = self.encoder(h + self.pos_embed[:, :h.shape[1]])
            return h[:, 0] if self.architecture == 'transformer' else self.pool(h[:, 1:])
        if self.architecture == 'multiscale_cnn':
            return torch.cat([self.pool(b(x.transpose(1, 2)).transpose(1, 2))
                              for b in self.branches], dim=-1)
        if self.architecture == 'gru':
            return self.pool(self.encoder(x)[0])
        if self.architecture == 'tcn':
            h = self.input_proj(x).transpose(1, 2)
            for block in self.blocks:
                h = h + block(h)
            return self.pool(h.transpose(1, 2))
        return torch.cat([x.mean(1), x.std(1, unbiased=False),
                          x.amax(1)-x.amin(1), x[:, -1]-x[:, 0]], dim=-1)

    def forward(self, x):
        return self.classifier(self.encode(x)).squeeze(-1)
