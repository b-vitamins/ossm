from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from .batch_utils import new_batch_zeros, pad_sequence_batch
from ..transforms.compose import TimeSeriesSample


def pad_collate(batch: Sequence[TimeSeriesSample]) -> TimeSeriesSample:
    if not batch:
        raise ValueError("batch must contain at least one sample")

    values_batch = []
    times_batch = []
    labels = []
    lengths = []
    for sample in batch:
        values = sample.get("values")
        times = sample.get("times")
        label = sample.get("label")
        if values is None or times is None or label is None:
            raise KeyError("pad_collate requires 'values', 'times', and 'label'")
        values_batch.append(values)
        times_batch.append(times)
        labels.append(label)
        lengths.append(values.size(0))

    values, mask = pad_sequence_batch(values_batch, lengths=lengths)
    times, _ = pad_sequence_batch(
        times_batch,
        lengths=lengths,
        create_mask=False,
    )
    return cast(
        TimeSeriesSample,
        {
            "values": values,
            "times": times,
            "mask": mask,
            "label": torch.stack(labels),
        },
    )


def path_collate(batch: Sequence[TimeSeriesSample]) -> TimeSeriesSample:
    if not batch:
        raise ValueError("batch must contain at least one sample")

    features_batch = []
    labels = []
    segment_lengths = []
    for sample in batch:
        features = sample.get("features")
        label = sample.get("label")
        if features is None or label is None:
            raise KeyError("path_collate requires 'features' and 'label'")
        features_batch.append(features)
        labels.append(label)
        segment_lengths.append(features.size(0))

    logsig, _ = pad_sequence_batch(features_batch, lengths=segment_lengths, create_mask=False)

    initial_template = None
    for sample in batch:
        values_tensor = sample.get("values")
        if values_tensor is not None:
            initial_template = values_tensor[0]
            break

    if initial_template is None:
        initial_template = features_batch[0][0]

    initial_template = initial_template.reshape(-1)

    initials = new_batch_zeros(initial_template, (len(batch), initial_template.size(-1)))

    for idx, sample in enumerate(batch):
        values_tensor = sample.get("values")
        if values_tensor is not None:
            initials[idx] = values_tensor[0].reshape(-1).to(
                dtype=initials.dtype, device=initials.device
            )

    times_template = batch[0].get("times")
    if times_template is not None:
        time_dtype = times_template.dtype
        time_device = times_template.device
    else:
        time_dtype = logsig.dtype
        time_device = logsig.device
    times = torch.linspace(
        0.0,
        1.0,
        logsig.size(1) + 1,
        dtype=time_dtype,
        device=time_device,
    )

    return cast(
        TimeSeriesSample,
        {
            "logsig": logsig,
            "times": times,
            "initial": initials,
            "label": torch.stack(labels),
            "features": logsig,
        },
    )


def coeff_collate(batch: Sequence[TimeSeriesSample]) -> TimeSeriesSample:
    if not batch:
        raise ValueError("batch must contain at least one sample")

    time_batch = []
    coeff_batch = []
    labels = []
    time_lengths = []
    coeff_lengths = []

    for sample in batch:
        times = sample.get("times")
        coeffs_tensor = sample.get("coeffs")
        label = sample.get("label")
        if times is None or coeffs_tensor is None or label is None:
            raise KeyError("coeff_collate requires 'times', 'coeffs', and 'label'")
        time_batch.append(times)
        coeff_batch.append(coeffs_tensor)
        labels.append(label)
        time_lengths.append(times.size(0))
        coeff_lengths.append(coeffs_tensor.size(0))

    times, mask = pad_sequence_batch(time_batch, lengths=time_lengths)

    sample_coeffs = coeff_batch[0]
    if sample_coeffs.ndim == 2:
        flat_dim = sample_coeffs.size(-1)
        channel_dim = flat_dim // 4
    elif sample_coeffs.ndim == 3:
        channel_dim = sample_coeffs.size(1)
    else:
        raise ValueError("coeffs must be rank-2 or rank-3 tensors")

    coeffs, _ = pad_sequence_batch(
        coeff_batch,
        lengths=coeff_lengths,
        create_mask=False,
    )

    initial_template = None
    for sample in batch:
        initial = sample.get("initial")
        if initial is not None:
            initial_template = initial
            break
        values_tensor = sample.get("values")
        if values_tensor is not None:
            initial_template = values_tensor[0]
            break

    if initial_template is None:
        raise KeyError("coeff_collate requires 'initial' or 'values'")

    initial_template = initial_template.reshape(-1)

    initials = new_batch_zeros(initial_template, (len(batch), initial_template.size(-1)))

    for idx, sample in enumerate(batch):
        initial = sample.get("initial")
        if initial is not None:
            initials[idx] = initial.reshape(-1).to(
                dtype=initials.dtype, device=initials.device
            )
        else:
            values_tensor = sample.get("values")
            if values_tensor is None:
                raise KeyError(
                    "coeff_collate requires 'values' when 'initial' is absent"
                )
            initials[idx] = values_tensor[0].reshape(-1).to(
                dtype=initials.dtype, device=initials.device
            )

    return cast(
        TimeSeriesSample,
        {
            "times": times,
            "coeffs": coeffs,
            "initial": initials,
            "mask": mask,
            "label": torch.stack(labels),
        },
    )
