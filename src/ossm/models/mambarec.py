"""Mamba-based sequential recommender with faithful discretized SSM."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList

from ._selective_scan import try_selective_scan as _try_selective_scan
from .dlinossrec import ItemEmbeddingEncoder, _last_mask_index
from .heads import TiedSoftmaxHead

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..data.datasets.seqrec import SeqRecBatch

__all__ = ["MambaLayer", "Mamba4Rec"]


def _selective_scan_mamba(
    inputs: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
) -> torch.Tensor:
    """Mamba-compatible selective scan using dt-based forcing (no phi).

    Mirrors the reference path used by mamba-ssm when ``use_fast_path=False``:
    h_t = exp(dt*A) h_{t-1} + (dt * B_t) * u_t; y_t = C_t^T h_t
    All math is performed in float32 for stability.
    Shapes:
      inputs: (B, C, L)
      dt:     (B, C, L)
      A:      (C, S)
      B_t,C_t:(B, L, S)
    """
    batch, channels, seqlen = inputs.shape
    if seqlen == 0:
        return inputs.new_zeros(batch, channels, seqlen)

    u = inputs.to(dtype=torch.float32)
    dt = dt.to(dtype=torch.float32)
    A_matrix = A.to(dtype=torch.float32)
    B_proj = B_t.to(dtype=torch.float32)
    C_proj = C_t.to(dtype=torch.float32)

    a_matrix = A_matrix.unsqueeze(0)  # (1, C, S)
    state = torch.zeros(batch, channels, A_matrix.size(1), dtype=u.dtype, device=u.device)
    ys: list[torch.Tensor] = []
    for t in range(seqlen):
        delta = dt[:, :, t].unsqueeze(-1)  # (B, C, 1)
        A_bar = torch.exp(delta * a_matrix)  # (B, C, S)
        forcing = (delta * B_proj[:, t].unsqueeze(1)) * u[:, :, t].unsqueeze(-1)
        state = A_bar * state + forcing
        ys.append(torch.einsum("bcs,bs->bc", state, C_proj[:, t]))
    y = torch.stack(ys, dim=-1)
    return y.to(dtype=inputs.dtype)


class _MambaMixer(nn.Module):
    """Pure PyTorch Mamba mixer (selective SSM) with faithful discretization."""

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

        # Input projections producing x and z streams.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        # Explicitly causal depth-wise convolution: pad on the left, no implicit padding.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=conv_bias,
            padding=0,
        )
        self.act = nn.SiLU()

        # Projections for dt, B, and C parameters.
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

        # The D skip connection is standard in Mamba but omitted per the paper description.
        self.register_parameter("D", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Inputs to Mamba mixer must have shape (batch, length, channels)")
        batch, seqlen, _ = inputs.shape
        if seqlen == 0:
            return torch.zeros_like(inputs)

        xz = self.in_proj(inputs)
        x, z = xz.chunk(2, dim=-1)

        x_conv = x.transpose(1, 2)
        if self.d_conv > 1:
            x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        x_conv = self.act(x_conv)
        x_features = x_conv.transpose(1, 2)

        proj = self.x_proj(x_features)
        dt_raw, B_gen, C_gen = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        dt_proj = dt.transpose(1, 2)
        A_param = -torch.exp(self.A_log.float())
        gate = z.transpose(1, 2)

        # Ensure fused kernels are eligible under autocast by using float32 inputs.
        # Default to Mamba-SSM selective scan numerics to match upstream.
        fused = _try_selective_scan(
            inputs=x_conv.to(dtype=torch.float32),
            dt=dt_proj.to(dtype=torch.float32),
            A=A_param.to(dtype=torch.float32),
            B=B_gen.to(dtype=torch.float32),
            C=C_gen.to(dtype=torch.float32),
            gate=gate.to(dtype=torch.float32),
        )

        if fused is None:
            outputs = _selective_scan_mamba(
                inputs=x_conv,
                dt=dt_proj,
                A=A_param,
                B_t=B_gen,
                C_t=C_gen,
            )
            gated = outputs * F.silu(gate)
        else:
            gated = fused.to(dtype=x_conv.dtype)

        return self.out_proj(gated.transpose(1, 2))


class FeedForward(nn.Module):
    """Two-layer position-wise FFN with optional residual and LayerNorm."""

    def __init__(
        self,
        d_model: int,
        inner_size: int,
        dropout: float,
        use_layernorm: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, inner_size)
        self.linear2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12) if use_layernorm else nn.Identity()
        self.use_residual = bool(residual)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(inputs)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout(hidden)
        if self.use_residual:
            hidden = hidden + inputs
        return self.norm(hidden)


class MambaLayer(nn.Module):
    """Single Mamba mixer layer followed by a position-wise FFN."""

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

        if use_pffn:
            self.ffn: nn.Module = FeedForward(
                d_model,
                inner_size=4 * d_model,
                dropout=dropout,
                use_layernorm=use_layernorm,
                residual=self.num_layers > 1,
            )
        else:
            self.ffn = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.mamba(inputs)
        hidden = self.dropout(hidden)

        if self.num_layers == 1:
            hidden = self.norm(hidden)
            if not isinstance(self.ffn, nn.Identity):
                hidden = self.ffn(hidden)
            return hidden

        hidden = self.norm(hidden + inputs)
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
        head_bias: bool = False,
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
        last_index = _last_mask_index(mask).to(device=hidden.device)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, last_index]

    def forward_loss(self, batch: "SeqRecBatch") -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.loss(last_hidden, batch.target)

    def predict_logits(
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
