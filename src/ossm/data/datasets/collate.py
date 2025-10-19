from __future__ import annotations
from typing import List, Dict
import torch


def pad_collate(batch: List[Dict]):
    Tm = max(b["values"].size(0) for b in batch)
    B = len(batch)
    C = batch[0]["values"].size(-1)
    X = torch.zeros(B, Tm, C, dtype=batch[0]["values"].dtype)
    T = torch.zeros(B, Tm, dtype=batch[0]["times"].dtype)
    M = torch.zeros(B, Tm, dtype=torch.bool)
    Y = []
    for i, b in enumerate(batch):
        t = b["values"].size(0)
        X[i, :t] = b["values"]
        T[i, :t] = b["times"]
        M[i, :t] = True
        Y.append(b["label"])
    return {"values": X, "times": T, "mask": M, "label": torch.stack(Y)}


def path_collate(batch: List[Dict]):
    Smax = max(b["features"].size(0) for b in batch)
    B = len(batch)
    D = batch[0]["features"].size(-1)
    F = torch.zeros(B, Smax, D, dtype=batch[0]["features"].dtype)
    Y = []
    for i, b in enumerate(batch):
        s = b["features"].size(0)
        F[i, :s] = b["features"]
        Y.append(b["label"])
    return {"features": F, "label": torch.stack(Y)}


def coeff_collate(batch: List[Dict]):
    Tm = max(b["times"].size(0) for b in batch)
    Km = max(b["coeffs"].size(0) for b in batch)
    B = len(batch)

    sample_times = batch[0]["times"]
    times = sample_times.new_zeros((B, Tm))
    mask = torch.zeros(B, Tm, dtype=torch.bool, device=sample_times.device)

    sample_coeffs = batch[0]["coeffs"]
    if sample_coeffs.ndim == 2:
        C4 = sample_coeffs.size(-1)
        coeffs = sample_coeffs.new_zeros((B, Km, C4))
        channel_dim = C4 // 4
    elif sample_coeffs.ndim == 3:
        C = sample_coeffs.size(1)
        order = sample_coeffs.size(2)
        coeffs = sample_coeffs.new_zeros((B, Km, C, order))
        channel_dim = C
    else:
        raise ValueError("coeffs must be rank-2 or rank-3 tensors")

    initial_template = batch[0].get("initial")
    if initial_template is None:
        initial_template = batch[0]["values"][0]
    initials = initial_template.new_zeros((B, channel_dim))

    labels = []

    for i, b in enumerate(batch):
        t = b["times"].size(0)
        k = b["coeffs"].size(0)
        times[i, :t] = b["times"]
        coeffs[i, :k] = b["coeffs"]
        mask[i, :t] = True
        if "initial" in b:
            initials[i] = b["initial"]
        else:
            initials[i] = b["values"][0]
        labels.append(b["label"])

    return {
        "times": times,
        "coeffs": coeffs,
        "initial": initials,
        "mask": mask,
        "label": torch.stack(labels),
    }
