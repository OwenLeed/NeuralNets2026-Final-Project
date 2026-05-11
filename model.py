import math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from tokenizer import INPUT_WINDOW, N_FEATURES, PREDICTION_HORIZON


@dataclass
class TransformerConfig:
    d_model        : int   = 64
    n_heads        : int   = 4
    n_layers       : int   = 3
    d_feedforward  : int   = 128
    dropout        : float = 0.1

    n_input_features : int = N_FEATURES
    seq_len          : int = INPUT_WINDOW

    quantiles : List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95]
    )

    pooling : str = 'attention'

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_heads ({self.n_heads})"
        )
        assert self.pooling in ('attention', 'mean', 'last'), (
            f"pooling must be 'attention', 'mean', or 'last'"
        )
        assert all(0 < q < 1 for q in self.quantiles), (
            "All quantiles must be strictly between 0 and 1"
        )
        assert self.quantiles == sorted(self.quantiles), (
            "Quantiles must be in ascending order"
        )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AttentionPooling(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(
        self,
        x            : torch.Tensor,
        padding_mask : torch.Tensor,
    ) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return pooled

class QuantileHead(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlucoseTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(
            config.n_input_features, config.d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model  = config.d_model,
            max_len  = config.seq_len + 10,
            dropout  = config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.d_model,
            nhead           = config.n_heads,
            dim_feedforward = config.d_feedforward,
            dropout         = config.dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = config.n_layers,
            enable_nested_tensor = False,
        )

        if config.pooling == 'attention':
            self.pooling_layer = AttentionPooling(config.d_model)

        self.quantile_heads = nn.ModuleList([
            QuantileHead(config.d_model, config.dropout)
            for _ in config.quantiles
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features       : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_projection(features)
        x = self.pos_encoding(x)

        padding_mask_transformer = ~attention_mask
        x = self.transformer(
            src                = x,
            src_key_padding_mask = padding_mask_transformer,
        )

        if self.config.pooling == 'attention':
            pooled = self.pooling_layer(x, attention_mask)
        elif self.config.pooling == 'mean':
            mask_float = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            last_valid = attention_mask.long().cumsum(dim=1).argmax(dim=1)
            pooled = x[torch.arange(x.size(0)), last_valid]

        quantile_preds = torch.cat(
            [head(pooled) for head in self.quantile_heads],
            dim=1
        )

        return quantile_preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        components = {
            'input_projection' : self.input_projection,
            'pos_encoding'     : self.pos_encoding,
            'transformer'      : self.transformer,
            'quantile_heads'   : self.quantile_heads,
        }
        if self.config.pooling == 'attention':
            components['pooling'] = self.pooling_layer

        return {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in components.items()
        }