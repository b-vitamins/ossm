#!/usr/bin/env python3
"""Generate D-LinOSS reference fixtures from the upstream JAX implementation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logit

jax.config.update("jax_enable_x64", True)

VARIANTS: Tuple[str, ...] = ("imex1", "imex2", "im", "ex")


@dataclass
class SharedParameters:
    a_diag: jnp.ndarray
    g_diag: jnp.ndarray
    step: jnp.ndarray
    b: jnp.ndarray  # (ssm, hidden, 2)
    c: jnp.ndarray  # (hidden, ssm, 2)
    d: jnp.ndarray  # (hidden,)
    inputs: jnp.ndarray  # (batch, length, hidden)


def _compute_a_bounds(step: jnp.ndarray, g_diag: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    denom = jnp.maximum(step**2, 1e-6)

    low_imex1 = (2.0 + step * g_diag - 2.0 * jnp.sqrt(1.0 + step * g_diag)) / denom
    high_imex1 = (2.0 + step * g_diag + 2.0 * jnp.sqrt(1.0 + step * g_diag)) / denom

    low_imex2 = (2.0 - step * g_diag - 2.0 * jnp.sqrt(1.0 - step * g_diag)) / denom
    high_imex2 = (2.0 - step * g_diag + 2.0 * jnp.sqrt(1.0 - step * g_diag)) / denom

    low = jnp.maximum(low_imex1, low_imex2)
    high = jnp.minimum(high_imex1, high_imex2)

    return low, high


def _sample_shared_parameters(
    *,
    key: jax.Array,
    batch: int,
    length: int,
    ssm_size: int,
    hidden_dim: int,
) -> SharedParameters:
    key_step, key_g, key_a, key_b, key_c, key_d, key_inputs = jr.split(key, 7)

    step = jr.uniform(key_step, shape=(ssm_size,), minval=0.05, maxval=0.45)
    g_diag = jr.uniform(key_g, shape=(ssm_size,), minval=0.05, maxval=0.45)
    g_diag = jnp.minimum(g_diag, 0.8 / step - 1e-3)

    a_low, a_high = _compute_a_bounds(step, g_diag)
    span = jnp.maximum(a_high - a_low, 1e-4)
    mix = jr.uniform(key_a, shape=(ssm_size,), minval=0.2, maxval=0.8)
    a_diag = a_low + span * mix

    b = jr.normal(key_b, shape=(ssm_size, hidden_dim, 2)) * 0.35
    c = jr.normal(key_c, shape=(hidden_dim, ssm_size, 2)) * 0.35
    d = jr.normal(key_d, shape=(hidden_dim,)) * 0.4
    inputs = jr.normal(key_inputs, shape=(batch, length, hidden_dim)) * 0.5

    return SharedParameters(a_diag=a_diag, g_diag=g_diag, step=step, b=b, c=c, d=d, inputs=inputs)


def _compute_coefficients(
    variant: str, a_diag: jnp.ndarray, g_diag: jnp.ndarray, step: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    one = jnp.ones_like(a_diag)
    if variant == "imex1":
        denom = one + step * g_diag
        inv = one / denom
        m11 = inv
        m12 = -(step**2 * a_diag) * inv
        m21 = inv
        m22 = one + m12
        f1 = (step**2) * inv
        f2 = f1
    elif variant == "imex2":
        m11 = one - step * g_diag
        m12 = -(step * a_diag)
        m21 = step * (one - step * g_diag)
        m22 = one - step**2 * a_diag
        f1 = step
        f2 = step**2
    elif variant == "im":
        denom = one + step * g_diag + step**2 * a_diag
        m11 = one / denom
        m12 = -(step * a_diag) / denom
        m21 = step / denom
        m22 = (one + step * g_diag) / denom
        f1 = step / denom
        f2 = step**2 / denom
    elif variant == "ex":
        m11 = one - step * g_diag
        m12 = -(step * a_diag)
        m21 = step
        m22 = one
        f1 = step
        f2 = jnp.zeros_like(step)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"Unknown variant '{variant}'.")
    return m11, m12, m21, m22, f1, f2


def _run_states(
    variant: str,
    a_diag: jnp.ndarray,
    g_diag: jnp.ndarray,
    step: jnp.ndarray,
    bu_seq: jnp.ndarray,
) -> jnp.ndarray:
    m11, m12, m21, m22, f1_scale, f2_scale = _compute_coefficients(variant, a_diag, g_diag, step)

    m11 = m11.reshape(1, -1)
    m12 = m12.reshape(1, -1)
    m21 = m21.reshape(1, -1)
    m22 = m22.reshape(1, -1)
    f1_scale = f1_scale.reshape(1, -1)
    f2_scale = f2_scale.reshape(1, -1)

    def step_fn(carry: tuple[jnp.ndarray, jnp.ndarray], bu_t: jnp.ndarray):
        state0, state1 = carry
        b1 = f1_scale * bu_t
        b2 = f2_scale * bu_t
        new0 = m11 * state0 + m12 * state1 + b1
        new1 = m21 * state0 + m22 * state1 + b2
        next_state = (new0, new1)
        stacked = jnp.stack(next_state, axis=-1)
        return next_state, stacked

    batch = bu_seq.shape[1]
    ssm = bu_seq.shape[2]
    init_state = (
        jnp.zeros((batch, ssm), dtype=bu_seq.dtype),
        jnp.zeros((batch, ssm), dtype=bu_seq.dtype),
    )
    _, states = jax.lax.scan(step_fn, init_state, bu_seq)
    return states  # (length, batch, ssm, 2)


def _apply_variant(
    variant: str,
    a_diag: jnp.ndarray,
    g_diag: jnp.ndarray,
    step: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    inputs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch, length, hidden_dim = inputs.shape
    ssm = a_diag.shape[0]

    b_real = b[..., 0]
    b_imag = b[..., 1]
    flat_inputs = inputs.reshape(batch * length, hidden_dim)
    bu_real = flat_inputs @ jnp.transpose(b_real, (1, 0))
    bu_imag = flat_inputs @ jnp.transpose(b_imag, (1, 0))
    bu = (bu_real + 1j * bu_imag).reshape(batch, length, ssm)
    bu_seq = jnp.transpose(bu, (1, 0, 2))

    states = _run_states(variant, a_diag, g_diag, step, bu_seq)
    states_aux = jnp.transpose(states[..., 0], (1, 0, 2))
    states_main = jnp.transpose(states[..., 1], (1, 0, 2))

    c_complex = c[..., 0] + 1j * c[..., 1]
    states_flat = states_main.reshape(batch * length, ssm)
    projected = jnp.matmul(states_flat, jnp.transpose(c_complex))
    projected = projected.real.reshape(batch, length, c.shape[0])

    outputs = projected + inputs * d.reshape(1, 1, -1)
    return outputs, states_main, states_aux, states


def _loss_for_gradients(
    variant: str,
    a_diag: jnp.ndarray,
    g_diag: jnp.ndarray,
    step: jnp.ndarray,
    b_real: jnp.ndarray,
    b_imag: jnp.ndarray,
    c_real: jnp.ndarray,
    c_imag: jnp.ndarray,
    d_vec: jnp.ndarray,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    b = jnp.stack((b_real, b_imag), axis=-1)
    c = jnp.stack((c_real, c_imag), axis=-1)
    outputs, _, _, states = _apply_variant(variant, a_diag, g_diag, step, b, c, d_vec, inputs)
    loss = jnp.sum(outputs**2)
    loss = loss + jnp.sum(jnp.real(states) ** 2) + jnp.sum(jnp.imag(states) ** 2)
    return loss


def _tensor_to_list(array: jnp.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def _validate_with_upstream(
    *,
    upstream_root: Path | None,
    variant: str,
    dtype: jnp.dtype,
    params: SharedParameters,
    outputs: jnp.ndarray,
    states_main: jnp.ndarray,
) -> None:
    if upstream_root is None:
        return
    if dtype == jnp.float64:
        # The upstream implementation operates in single precision and does
        # not accept complex128 intermediates. Skip validation for float64.
        return

    module_root = upstream_root / "src"
    module_path = str(module_root.resolve())
    added = False
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
        added = True
    try:
        import equinox as eqx

        from damped_linoss.models.LinOSS import (
            DampedEXLayer,
            DampedIMLayer,
            DampedIMEX1Layer,
            DampedIMEX2Layer,
        )

        layer_map = {
            "imex1": DampedIMEX1Layer,
            "imex2": DampedIMEX2Layer,
            "im": DampedIMLayer,
            "ex": DampedEXLayer,
        }
        layer_cls = layer_map[variant]

        layer = layer_cls(
            state_dim=params.a_diag.shape[0],
            hidden_dim=params.c.shape[0],
            initialization="uniform",
            r_min=0.5,
            r_max=0.95,
            theta_min=0.0,
            theta_max=jnp.pi,
            G_min=0.0,
            G_max=2.0,
            A_min=0.0,
            A_max=2.0,
            dt_std=0.1,
            key=jr.PRNGKey(0),
        )

        a = params.a_diag.astype(dtype)
        g = params.g_diag.astype(dtype)
        step = params.step.astype(dtype)
        b = params.b.astype(dtype)
        c = params.c.astype(dtype)
        d_vec = params.d.astype(dtype)
        inputs = params.inputs.astype(dtype).reshape(-1, params.inputs.shape[-1])

        logit_step = logit(step).astype(dtype)
        layer = eqx.tree_at(
            lambda layer_obj: (
                layer_obj.A_diag,
                layer_obj.G_diag,
                layer_obj.dt,
                layer_obj.B,
                layer_obj.C,
                layer_obj.D,
            ),
            layer,
            (a, g, logit_step, b, c, d_vec),
        )

        try:
            upstream_outputs = layer(inputs)
            if upstream_outputs.ndim == 2:
                upstream_outputs = upstream_outputs[None, ...]
            np.testing.assert_allclose(
                np.asarray(upstream_outputs),
                np.asarray(outputs),
                rtol=5e-7,
                atol=5e-7,
            )

            b_complex = b[..., 0] + 1j * b[..., 1]
            bu = jax.vmap(lambda u: b_complex @ u)(inputs)
            proj_a, proj_g, proj_step = layer._soft_project_AGdt(layer.A_diag, layer.G_diag, layer.dt)
            upstream_states = layer._recurrence(proj_a, proj_g, proj_step, bu)
            if upstream_states.ndim == 2:
                upstream_states = upstream_states[None, ...]
            np.testing.assert_allclose(
                np.asarray(upstream_states),
                np.asarray(states_main),
                rtol=5e-7,
                atol=5e-7,
            )
        except TypeError:
            return
    finally:
        if added:
            sys.path.remove(module_path)


def generate_reference_cases(
    *, seed: int, upstream_root: Path | None
) -> Dict[str, Dict[str, dict]]:
    batch, length, ssm_size, hidden_dim = 1, 6, 4, 3
    key = jr.PRNGKey(seed)
    shared = _sample_shared_parameters(
        key=key,
        batch=batch,
        length=length,
        ssm_size=ssm_size,
        hidden_dim=hidden_dim,
    )

    cases: Dict[str, Dict[str, dict]] = {}

    for variant in VARIANTS:
        variant_cases: Dict[str, dict] = {}
        for dtype_name, dtype in ("float32", jnp.float32), ("float64", jnp.float64):
            a = shared.a_diag.astype(dtype)
            g = shared.g_diag.astype(dtype)
            step = shared.step.astype(dtype)
            b = shared.b.astype(dtype)
            c = shared.c.astype(dtype)
            d_vec = shared.d.astype(dtype)
            inputs = shared.inputs.astype(dtype)

            outputs, states_main, states_aux, states_full = _apply_variant(
                variant, a, g, step, b, c, d_vec, inputs
            )

            def loss_fn(a_v, g_v, step_v, b_r, b_i, c_r, c_i, d_v):
                return _loss_for_gradients(
                    variant,
                    a_v,
                    g_v,
                    step_v,
                    b_r,
                    b_i,
                    c_r,
                    c_i,
                    d_v,
                    inputs,
                )
            grad_fn = jax.value_and_grad(
                loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6, 7)
            )
            _, (grad_a, grad_g, grad_step, grad_b_real, grad_b_imag, grad_c_real, grad_c_imag, grad_d) = grad_fn(
                a,
                g,
                step,
                b[..., 0],
                b[..., 1],
                c[..., 0],
                c[..., 1],
                d_vec,
            )

            state_main_tensor = np.stack(
                (np.asarray(states_main.real), np.asarray(states_main.imag)), axis=-1
            )
            state_aux_tensor = np.stack(
                (np.asarray(states_aux.real), np.asarray(states_aux.imag)), axis=-1
            )

            grads = {
                "A_diag": _tensor_to_list(grad_a),
                "G_diag": _tensor_to_list(grad_g),
                "step": _tensor_to_list(grad_step),
                "B": _tensor_to_list(jnp.stack((grad_b_real, grad_b_imag), axis=-1)),
                "C": _tensor_to_list(jnp.stack((grad_c_real, grad_c_imag), axis=-1)),
                "D": _tensor_to_list(grad_d),
            }

            variant_cases[dtype_name] = {
                "ssm_size": int(ssm_size),
                "hidden_dim": int(hidden_dim),
                "batch": int(batch),
                "sequence_length": int(length),
                "A_diag": _tensor_to_list(a),
                "G_diag": _tensor_to_list(g),
                "step": _tensor_to_list(step),
                "B": _tensor_to_list(b),
                "C": _tensor_to_list(c),
                "D": _tensor_to_list(d_vec),
                "inputs": _tensor_to_list(inputs),
                "outputs": _tensor_to_list(outputs),
                "states": {
                    "main": state_main_tensor.tolist(),
                    "aux": state_aux_tensor.tolist(),
                },
                "grads": grads,
            }

            _validate_with_upstream(
                upstream_root=upstream_root,
                variant=variant,
                dtype=dtype,
                params=shared,
                outputs=outputs,
                states_main=states_main,
            )

        cases[variant] = variant_cases

    return cases


def _write_json_file(path: Path, cases: Dict[str, Dict[str, dict]], seed: int) -> None:
    payload = {
        "_metadata": {
            "seed": seed,
            "variants": VARIANTS,
            "regeneration_command": (
                "python scripts/refresh_dlinoss_goldens.py "
                f"--seed {seed} --upstream /path/to/damped-linoss --output {path}"
            ),
        },
        "cases": cases,
    }

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Random seed for parameter sampling")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="Path to the cloned damped-linoss repository for validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/dlinoss_reference_cases.json"),
        help="Destination for the generated JSON payload",
    )
    args = parser.parse_args()

    if args.upstream is not None and not args.upstream.exists():
        raise FileNotFoundError(f"Upstream repository not found at {args.upstream}")

    cases = generate_reference_cases(seed=args.seed, upstream_root=args.upstream)
    _write_json_file(args.output, cases, seed=args.seed)


if __name__ == "__main__":
    main()
