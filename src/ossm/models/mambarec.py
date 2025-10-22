"""Mamba-based sequential recommender with in-repo SSM implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList

from .dlinossrec import ItemEmbeddingEncoder
from .heads import TiedSoftmaxHead

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..data.datasets.seqrec import SeqRecBatch

__all__ = ["MambaLayer", "Mamba4Rec"]


def _selective_scan(
    inputs: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: torch.Tensor | None,
    gate: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference selective scan implementation used by the Mamba mixer."""

    batch, channels, seqlen = inputs.shape
    if seqlen == 0:
        return inputs.new_zeros(batch, channels, seqlen)

    dtype = inputs.dtype

    u = inputs.to(dtype=torch.float32)
    dt = delta.to(dtype=torch.float32)
    if delta_bias is not None:
        dt = dt + delta_bias.to(dtype=torch.float32).view(1, -1, 1)
    dt = F.softplus(dt)

    A_matrix = A.to(dtype=torch.float32)
    B_proj = B.to(dtype=torch.float32)
    C_proj = C.to(dtype=torch.float32)
    skip = D.to(dtype=torch.float32).view(1, -1, 1) if D is not None else None

    state = torch.zeros(batch, channels, A_matrix.size(1), device=inputs.device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []
    A_expanded = A_matrix.unsqueeze(0)
    for timestep in range(seqlen):
        dt_t = dt[:, :, timestep]
        u_t = u[:, :, timestep]
        B_t = B_proj[:, :, timestep]
        C_t = C_proj[:, :, timestep]

        delta_A = torch.exp(dt_t.unsqueeze(-1) * A_expanded)
        delta_B_u = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
        state = delta_A * state + delta_B_u
        y_t = torch.einsum("bdn,bn->bd", state, C_t)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=-1)
    if skip is not None:
        y = y + u * skip
    if gate is not None:
        y = y * F.silu(gate.to(dtype=torch.float32))
    return y.to(dtype=dtype)


class _MambaMixer(nn.Module):
    """Pure PyTorch implementation of the Mamba selective state model."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dt_rank == "auto":
            dt_rank = max(1, math.ceil(d_model / 16))
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=conv_bias,
            padding=self.d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        dt_init_std = (self.dt_rank ** -0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:  # pragma: no cover - configuration guard
            raise ValueError("dt_init must be 'random' or 'constant'")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Inputs to Mamba mixer must have shape (batch, length, channels)")
        batch, seqlen, _ = inputs.shape
        if seqlen == 0:
            return torch.zeros_like(inputs)

        xz = self.in_proj(inputs)
        x, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2))[..., :seqlen]
        x_conv = self.act(x_conv)
        x_features = x_conv.transpose(1, 2)

        proj = self.x_proj(x_features)
        dt, B, C = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)

        outputs = _selective_scan(
            x_conv,
            dt.transpose(1, 2),
            -torch.exp(self.A_log.float()),
            B.transpose(1, 2),
            C.transpose(1, 2),
            D=self.D,
            gate=z.transpose(1, 2),
            delta_bias=self.dt_proj.bias.float(),
        )
        return self.out_proj(outputs.transpose(1, 2))


class FeedForward(nn.Module):
    """Two-layer feed-forward block with residual connection."""

    def __init__(self, d_model: int, inner_size: int, dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, inner_size)
        self.linear2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12) if use_layernorm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.linear1(inputs)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout(hidden)
        return self.norm(hidden + residual)


class MambaLayer(nn.Module):
    """Single Mamba mixer layer with residual and FFN blocks."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        num_layers: int,
        use_layernorm: bool,
        use_pffn: bool,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.mamba = _MambaMixer(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12) if use_layernorm else nn.Identity()
        self.ffn: nn.Module
        if use_pffn:
            self.ffn = FeedForward(d_model, inner_size=4 * d_model, dropout=dropout, use_layernorm=use_layernorm)
        else:
            self.ffn = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.mamba(inputs)
        hidden = self.dropout(hidden)
        if self.num_layers == 1:
            hidden = self.norm(hidden)
        else:
            hidden = self.norm(hidden + residual)
        hidden = self.ffn(hidden)
        return hidden


class Mamba4Rec(nn.Module):
    """Mamba-based sequential recommender mirroring the Mamba4Rec architecture."""

    def __init__(
        self,
        *,
        num_items: int,
        d_model: int,
        ssm_size: int,
        blocks: int,
        dropout: float,
        max_len: int,
        use_pffn: bool = True,
        use_pos_emb: bool = False,
        use_layernorm: bool = True,
        head_bias: bool = True,
        head_temperature: float = 1.0,
        d_conv: int = 4,
        expand: int = 2,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = ItemEmbeddingEncoder(
            num_items,
            d_model,
            dropout=dropout,
            max_len=max_len,
            use_layernorm=use_layernorm,
            use_pos_emb=use_pos_emb,
        )
        self.layers = ModuleList(
            [
                MambaLayer(
                    d_model,
                    d_state=ssm_size,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    num_layers=blocks,
                    use_layernorm=use_layernorm,
                    use_pffn=use_pffn,
                    **mamba_kwargs,
                )
                for _ in range(blocks)
            ]
        )
        self.head = TiedSoftmaxHead(
            self.encoder.embedding,
            bias=head_bias,
            temperature=head_temperature,
            padding_idx=0,
        )
        self.apply(self._init_weights)

    def forward_features(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        lengths = mask.sum(dim=1)
        last_index = lengths.clamp(min=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, last_index]

    def forward_loss(self, batch: "SeqRecBatch") -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.loss(last_hidden, batch.target)

    def predict_scores(
        self, batch: "SeqRecBatch", *, include_padding: bool = False
    ) -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.logits(last_hidden, exclude_padding=not include_padding)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
