"""Regression tests for training step and evaluation loops."""

from __future__ import annotations

import math

import pytest
import torch

from ossm.models import (
    BatchOnDevice,
    ClassificationHead,
    NCDEBackbone,
    RegressionHead,
    RNNBackbone,
)
from ossm.training.train import evaluate, step


def _make_classification_batch(
    batch_size: int,
    length: int,
    input_dim: int,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    values = torch.randn(batch_size, length, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    return {"values": values, "label": labels}


def test_step_with_standard_backbone_handles_prepare_batch() -> None:
    batch_size, length, input_dim, hidden_dim, num_classes = 4, 5, 3, 8, 6
    backbone = RNNBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
    head = ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes)

    batch = _make_classification_batch(batch_size, length, input_dim, num_classes)
    batch_on_device = BatchOnDevice.from_batch(batch, device=torch.device("cpu"))

    loss, logits, targets = step(backbone, head, batch_on_device)

    assert math.isfinite(float(loss.detach()))
    assert logits.shape == (batch_size, num_classes)
    assert torch.equal(targets, batch_on_device["label"])

    # evaluate should run without raising and compute metrics on moved targets
    evaluate(backbone, head, [batch], torch.device("cpu"))

def test_step_with_ncde_backbone_handles_prepare_batch() -> None:
    torchcde = pytest.importorskip("torchcde")

    batch_size, length, input_dim, hidden_dim, num_classes = 2, 4, 2, 4, 3
    path = torch.randn(batch_size, length, input_dim)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path)
    times = torch.linspace(0.0, 1.0, length)
    initial = path[:, 0]
    labels = torch.randint(0, num_classes, (batch_size,))

    batch = {
        "times": times,
        "coeffs": coeffs,
        "initial": initial,
        "label": labels,
    }

    backbone = NCDEBackbone(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vf_width=hidden_dim,
        vf_depth=1,
        step_size=1.0,
    )
    head = ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes)

    batch_on_device = BatchOnDevice.from_batch(batch, device=torch.device("cpu"))
    loss, logits, targets = step(backbone, head, batch_on_device)

    assert math.isfinite(float(loss.detach()))
    assert logits.shape == (batch_size, num_classes)
    assert torch.equal(targets, batch_on_device["label"])

    evaluate(backbone, head, [batch], torch.device("cpu"))


def test_step_with_regression_head_uses_sequence_features() -> None:
    batch_size, length, input_dim, hidden_dim, output_dim = 3, 6, 4, 5, 2
    backbone = RNNBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
    head = RegressionHead(hidden_dim=hidden_dim, output_dim=output_dim)

    values = torch.randn(batch_size, length, input_dim)
    targets = torch.randn(batch_size, length, output_dim)
    batch = {"values": values, "label": targets}
    batch_on_device = BatchOnDevice.from_batch(batch, device=torch.device("cpu"))

    loss, preds, targets_out = step(backbone, head, batch_on_device)

    assert math.isfinite(float(loss.detach()))
    assert preds.shape == (batch_size, length, output_dim)
    assert torch.equal(targets_out, targets)

    evaluate(backbone, head, [batch], torch.device("cpu"))

