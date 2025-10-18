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
    C4 = batch[0]["coeffs"].size(-1)
    C = C4 // 4
    times = torch.zeros(B, Tm, dtype=batch[0]["times"].dtype)
    coeffs = torch.zeros(B, Km, C4, dtype=batch[0]["coeffs"].dtype)
    initials = torch.zeros(B, C, dtype=batch[0]["coeffs"].dtype)
    mask = torch.zeros(B, Tm, dtype=torch.bool)
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
