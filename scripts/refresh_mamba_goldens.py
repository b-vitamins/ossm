
#!/usr/bin/env python3
"""Generate Mamba (mamba-ssm) reference fixtures as JSON.

This script instantiates mamba_ssm.Mamba with a small, reproducible config,
generates random inputs, runs forward, and records outputs and gradients for
all learnable parameters and inputs. The payload mirrors the structure used by
tests/dlinoss_reference_cases.json so tests can snapshot-compare our
re-implementation against these "goldens".

Assumptions
- mamba-ssm is importable (pip install mamba-ssm)
- Runs on CPU or CUDA; uses whatever torch device you choose

Example
  python scripts/refresh_mamba_goldens.py \
    --seed 123 --output tests/mamba_reference_cases.json \
    --d-model 64 --d-state 16 --d-conv 4 --expand 2 --batch 2 --length 16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class MambaConfig:
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    batch: int = 2
    length: int = 16
    zero_skip: bool = True  # Set D=0 to match our paper-faithful variant


def _tensor_list(x: torch.Tensor) -> list:
    return x.detach().cpu().to(torch.float64).tolist()


def _collect_param_tensors(m: nn.Module) -> Dict[str, list]:
    tensors: Dict[str, list] = {}
    for name, p in m.named_parameters():
        # Maintain stable key ordering in JSON by using explicit names
        tensors[name] = _tensor_list(p.data)
    return tensors


def _collect_param_grads(m: nn.Module) -> Dict[str, list]:
    grads: Dict[str, list] = {}
    for name, p in m.named_parameters():
        if p.grad is None:
            continue
        grads[name] = _tensor_list(p.grad)
    return grads


def _build_mamba(cfg: MambaConfig, device: torch.device) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("mamba-ssm is required to generate Mamba goldens") from exc

    m = Mamba(
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        use_fast_path=False,  # pick PyTorch path for stability/repro
    ).to(device)
    if cfg.zero_skip and hasattr(m, "D") and isinstance(m.D, torch.nn.Parameter):
        with torch.no_grad():
            m.D.zero_()
    return m


def generate_case(seed: int, cfg: MambaConfig, device: torch.device) -> Dict[str, object]:
    torch.manual_seed(seed)

    # Patch mamba-ssm to use the pure-PyTorch reference selective scan on CPU.
    # The CUDA extension enforces CUDA tensors; on CPU we swap in the ref fn.
    model = _build_mamba(cfg, device)
    if device.type == "cpu":
        try:
            from mamba_ssm.ops import selective_scan_interface as _ssi  # type: ignore
            _ssi.selective_scan_fn = _ssi.selective_scan_ref  # type: ignore[attr-defined]
        except Exception:
            pass
    model.train()

    inputs = torch.randn(cfg.batch, cfg.length, cfg.d_model, device=device, dtype=torch.float32)
    inputs.requires_grad_(True)

    # Re-implement forward structure to expose intermediates.
    # Shapes follow mamba_ssm.modules.mamba_simple.Mamba (use_fast_path=False path).
    seqlen = cfg.length
    xz = model.in_proj(inputs)  # (B, L, 2*D_inner)
    x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

    # Depthwise causal conv (left padding of d_conv-1), SiLU
    x_conv = x.transpose(1, 2)  # (B, D, L)
    x_conv = model.conv1d(x_conv)[..., :seqlen]
    x_conv = torch.nn.functional.silu(x_conv)
    x_features = x_conv  # (B, D, L)

    # Project to dt_raw, B, C; then dt_pre = W @ dt_raw (bias applied separately)
    x_dbl = torch.nn.functional.linear(
        x_features.transpose(1, 2).reshape(-1, x_features.size(1)),
        model.x_proj.weight,
    )  # (B*L, dt_rank + 2*d_state)
    dt_raw, B_flat, C_flat = torch.split(
        x_dbl, [model.dt_proj.in_features, model.d_state, model.d_state], dim=-1
    )
    # dt_pre: (B, D, L)
    dt_pre = (model.dt_proj.weight @ dt_raw.t()).reshape(model.d_inner, inputs.size(0), seqlen)
    dt_pre = dt_pre.permute(1, 0, 2).contiguous()
    # B, C as (B, S, L)
    B_ref = B_flat.reshape(inputs.size(0), seqlen, model.d_state).permute(0, 2, 1).contiguous()
    C_ref = C_flat.reshape(inputs.size(0), seqlen, model.d_state).permute(0, 2, 1).contiguous()
    z_ch = z.transpose(1, 2).contiguous()  # (B, D, L)

    # Discretization parameters
    A = -torch.exp(model.A_log.float())  # (D, S)
    bias = model.dt_proj.bias.float().unsqueeze(-1)  # (D, 1)
    dt_post = torch.nn.functional.softplus(dt_pre.float() + bias)  # (B, D, L)

    # Generate states and core outputs with a simple loop (device-agnostic, deterministic)
    B = B_ref.float(); C = C_ref.float(); u = x_features.float()
    batch, d_inner, length = u.shape
    state = torch.zeros(batch, d_inner, model.d_state, dtype=u.dtype, device=u.device)
    ys = []
    states_seq = []
    for t in range(length):
        delta = dt_post[:, :, t]
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        # Forcing term
        forcing = delta.unsqueeze(-1) * B[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
        state = A_bar * state + forcing
        states_seq.append(state)
        ys.append(torch.einsum('bdn,bn->bd', state, C[:, :, t]))
    y_core = torch.stack(ys, dim=-1)
    states_main = torch.stack(states_seq, dim=-1)  # (B, D, S, L)

    y_gated = y_core * torch.nn.functional.silu(z_ch)
    outputs = model.out_proj(y_gated.transpose(1, 2))  # (B, L, D_model)
    loss = (outputs.square().mean())
    loss.backward()

    payload: Dict[str, object] = {
        "config": asdict(cfg),
        "dtype": "float32",
        "batch": int(cfg.batch),
        "sequence_length": int(cfg.length),
        "d_model": int(cfg.d_model),
        "d_state": int(cfg.d_state),
        "d_conv": int(cfg.d_conv),
        "expand": int(cfg.expand),
        "inputs": _tensor_list(inputs),
        "outputs": _tensor_list(outputs.detach()),
        "params": _collect_param_tensors(model),
        "grads": {
            "inputs": _tensor_list(inputs.grad if inputs.grad is not None else torch.zeros_like(inputs)),
            **_collect_param_grads(model),
        },
        "loss": float(loss.detach().cpu().item()),
        "intermediates": {
            "x": _tensor_list(x_features),
            "z": _tensor_list(z_ch),
            "dt_pre": _tensor_list(dt_pre),
            "dt": _tensor_list(dt_post),
            "B": _tensor_list(B_ref),
            "C": _tensor_list(C_ref),
            "y_core": _tensor_list(y_core),
            "y_gated": _tensor_list(y_gated),
            "states": _tensor_list(states_main),
        },
    }
    return payload


def generate_reference(seed: int, cfg: MambaConfig, device: torch.device) -> Dict[str, object]:
    case = generate_case(seed, cfg, device)
    return {
        "_metadata": {
            "seed": seed,
            "regeneration_command": (
                "python scripts/refresh_mamba_goldens.py "
                f"--seed {seed} --output tests/mamba_reference_cases.json "
                f"--d-model {cfg.d_model} --d-state {cfg.d_state} --d-conv {cfg.d_conv} "
                f"--expand {cfg.expand} --batch {cfg.batch} --length {cfg.length}"
            ),
        },
        "case": case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Random seed for parameter sampling")
    parser.add_argument("--output", type=Path, default=Path("tests/mamba_reference_cases.json"), help="Destination JSON path")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Torch device to use")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--no-zero-skip", dest="zero_skip", action="store_false", help="Keep Mamba D skip connection instead of zeroing it")
    parser.set_defaults(zero_skip=True)

    args = parser.parse_args()
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    cfg = MambaConfig(
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        d_conv=int(args.d_conv),
        expand=int(args.expand),
        batch=int(args.batch),
        length=int(args.length),
        zero_skip=bool(args.zero_skip),
    )

    payload = generate_reference(seed=int(args.seed), cfg=cfg, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
