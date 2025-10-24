from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

__all__ = ["new_batch_zeros", "pad_sequence_batch"]


def new_batch_zeros(
    reference: torch.Tensor,
    shape: Iterable[int],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Allocate a zero tensor seeded from ``reference``.

    Parameters
    ----------
    reference:
        Tensor providing the default ``dtype`` and ``device`` when overrides are
        not supplied.
    shape:
        Desired shape of the allocated tensor.
    dtype, device:
        Optional overrides for the resulting tensor attributes.

    Returns
    -------
    torch.Tensor
        Newly allocated tensor filled with zeros.
    """

    tensor = reference.new_empty(tuple(shape), dtype=dtype, device=device)
    return tensor.zero_()


def pad_sequence_batch(
    tensors: Sequence[torch.Tensor],
    *,
    lengths: Sequence[int] | None = None,
    pad_value: float = 0.0,
    create_mask: bool = True,
    mask_value: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pad a batch of variable-length sequences along their leading dimension.

    The helper preserves ``dtype`` and ``device`` defaults from the first tensor
    in ``tensors`` unless explicit overrides are provided. Optionally, a mask is
    returned with entries initialised to ``mask_value`` and set to ``True`` for
    valid (unpadded) positions.
    """

    if not tensors:
        raise ValueError("tensors must contain at least one element")

    if lengths is None:
        lengths = [tensor.size(0) for tensor in tensors]

    if len(lengths) != len(tensors):
        msg = "lengths must match tensors"
        raise ValueError(msg)

    max_length = max(lengths)
    batch_size = len(tensors)
    reference = tensors[0]
    trailing_shape = reference.shape[1:]
    target_dtype = dtype if dtype is not None else reference.dtype
    target_device = device if device is not None else reference.device

    padded = reference.new_empty(
        (batch_size, max_length, *trailing_shape),
        dtype=target_dtype,
        device=target_device,
    )

    if pad_value == 0:
        padded.zero_()
    else:
        padded.fill_(pad_value)

    mask: torch.Tensor | None = None
    if create_mask:
        mask = reference.new_empty(
            (batch_size, max_length), dtype=torch.bool, device=target_device
        )
        mask.fill_(mask_value)

    for idx, (tensor, length) in enumerate(zip(tensors, lengths, strict=True)):
        if tensor.size(0) != length:
            msg = "Provided length does not match tensor size"
            raise ValueError(msg)

        data = tensor
        if tensor.dtype != target_dtype or tensor.device != target_device:
            data = tensor.to(dtype=target_dtype, device=target_device)

        padded[idx, :length].copy_(data)
        if mask is not None:
            mask[idx, :length] = True

    return padded, mask
