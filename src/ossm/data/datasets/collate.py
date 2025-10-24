from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from ..transforms.compose import TimeSeriesSample


def pad_collate(batch: Sequence[TimeSeriesSample]) -> TimeSeriesSample:
    if not batch:
        raise ValueError("batch must contain at least one sample")

    lengths = []
    for sample in batch:
        values = sample.get("values")
        times = sample.get("times")
        label = sample.get("label")
        if values is None or times is None or label is None:
            raise KeyError("pad_collate requires 'values', 'times', and 'label'")
        lengths.append(values.size(0))

    Tm = max(lengths)
    batch_size = len(batch)
    channels = batch[0].get("values")
    times_template = batch[0].get("times")
    if channels is None or times_template is None:
        raise KeyError("pad_collate requires 'values' and 'times'")

    values = channels.new_zeros((batch_size, Tm, channels.size(-1)))
    times = times_template.new_zeros((batch_size, Tm))
    mask = torch.zeros(
        batch_size, Tm, dtype=torch.bool, device=channels.device
    )
    labels = []
    for i, sample in enumerate(batch):
        values_tensor = sample.get("values")
        times_tensor = sample.get("times")
        label_tensor = sample.get("label")
        if values_tensor is None or times_tensor is None or label_tensor is None:
            raise KeyError("pad_collate requires 'values', 'times', and 'label'")
        length = values_tensor.size(0)
        values[i, :length] = values_tensor
        times[i, :length] = times_tensor
        mask[i, :length] = True
        labels.append(label_tensor)
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

    segment_lengths = []
    for sample in batch:
        features = sample.get("features")
        label = sample.get("label")
        if features is None or label is None:
            raise KeyError("path_collate requires 'features' and 'label'")
        segment_lengths.append(features.size(0))

    segments_max = max(segment_lengths)
    batch_size = len(batch)
    first_features = batch[0].get("features")
    if first_features is None:
        raise KeyError("path_collate requires 'features'")
    feature_dim = first_features.size(-1)

    logsig = first_features.new_zeros((batch_size, segments_max, feature_dim))
    first_values = batch[0].get("values")
    if first_values is not None:
        init_dim = first_values.size(-1)
        initials = first_values.new_zeros((batch_size, init_dim))
    else:
        init_dim = feature_dim
        initials = first_features.new_zeros((batch_size, init_dim))
    labels = []
    for i, sample in enumerate(batch):
        features = sample.get("features")
        label = sample.get("label")
        if features is None or label is None:
            raise KeyError("path_collate requires 'features' and 'label'")
        segments = features.size(0)
        logsig[i, :segments] = features
        values_tensor = sample.get("values")
        if values_tensor is not None:
            initials[i] = values_tensor[0]
        labels.append(label)

    times_template = batch[0].get("times")
    if times_template is not None:
        time_dtype = times_template.dtype
    else:
        time_dtype = first_features.dtype
    times = torch.linspace(
        0.0,
        1.0,
        segments_max + 1,
        dtype=time_dtype,
        device=first_features.device,
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

    lengths = []
    coeff_lengths = []
    for sample in batch:
        times = sample.get("times")
        coeffs_tensor = sample.get("coeffs")
        label = sample.get("label")
        if times is None or coeffs_tensor is None or label is None:
            raise KeyError("coeff_collate requires 'times', 'coeffs', and 'label'")
        lengths.append(times.size(0))
        coeff_lengths.append(coeffs_tensor.size(0))

    Tm = max(lengths)
    Km = max(coeff_lengths)
    batch_size = len(batch)

    sample_times = batch[0].get("times")
    if sample_times is None:
        raise KeyError("coeff_collate requires 'times'")
    times = sample_times.new_zeros((batch_size, Tm))
    mask = torch.zeros(batch_size, Tm, dtype=torch.bool, device=sample_times.device)

    sample_coeffs = batch[0].get("coeffs")
    if sample_coeffs is None:
        raise KeyError("coeff_collate requires 'coeffs'")
    if sample_coeffs.ndim == 2:
        flat_dim = sample_coeffs.size(-1)
        coeffs = sample_coeffs.new_zeros((batch_size, Km, flat_dim))
        channel_dim = flat_dim // 4
    elif sample_coeffs.ndim == 3:
        channels = sample_coeffs.size(1)
        order = sample_coeffs.size(2)
        coeffs = sample_coeffs.new_zeros((batch_size, Km, channels, order))
        channel_dim = channels
    else:
        raise ValueError("coeffs must be rank-2 or rank-3 tensors")

    initial_template = batch[0].get("initial")
    if initial_template is None:
        values_template = batch[0].get("values")
        if values_template is None:
            raise KeyError("coeff_collate requires 'initial' or 'values'")
        initial_template = values_template[0]
    initials = initial_template.new_zeros((batch_size, channel_dim))

    labels = []

    for i, sample in enumerate(batch):
        times_tensor = sample.get("times")
        coeffs_tensor = sample.get("coeffs")
        label_tensor = sample.get("label")
        if times_tensor is None or coeffs_tensor is None or label_tensor is None:
            raise KeyError("coeff_collate requires 'times', 'coeffs', and 'label'")
        t = times_tensor.size(0)
        k = coeffs_tensor.size(0)
        times[i, :t] = times_tensor
        coeffs[i, :k] = coeffs_tensor
        mask[i, :t] = True
        initial = sample.get("initial")
        if initial is not None:
            initials[i] = initial
        else:
            values_tensor = sample.get("values")
            if values_tensor is None:
                raise KeyError("coeff_collate requires 'values' when 'initial' is absent")
            initials[i] = values_tensor[0]
        labels.append(label_tensor)

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
