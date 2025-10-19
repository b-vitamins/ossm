"""Hydra-driven training entrypoint for OSSM models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, cast

from hydra.utils import instantiate, to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig  # type: ignore[import]
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data.datasets.collate import coeff_collate, pad_collate, path_collate
from ..models import (
    Backbone,
    ClassificationHead,
    Head,
    NCDEBackbone,
    RegressionHead,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


def build_dataloader(cfg: DictConfig, dataset) -> DataLoader:
    collate_name = cfg.get("collate", "pad")
    if collate_name == "pad":
        collate_fn: Callable = pad_collate
    elif collate_name == "coeff":
        collate_fn = coeff_collate
    elif collate_name == "path":
        collate_fn = path_collate
    else:
        raise ValueError(f"Unknown collate function '{collate_name}'.")
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.get("shuffle", True),
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.get("pin_memory", False),
    )


Batch = Dict[str, Tensor | Dict[str, Tensor]]


def _move_to_device(obj, device: torch.device):
    if isinstance(obj, Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


def step(
    backbone: Backbone,
    head: Head,
    batch: Batch,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    batch = cast(Batch, _move_to_device(batch, device))
    labels = cast(Tensor, batch["label"])

    if isinstance(backbone, NCDEBackbone):
        times = cast(Tensor, batch["times"])
        initial = cast(Tensor, batch["initial"])
        if "logsig" in batch:
            backbone_input = {"times": times, "logsig": cast(Tensor, batch["logsig"]), "initial": initial}
        else:
            backbone_input = {
                "times": times,
                "coeffs": cast(Tensor, batch["coeffs"]),
                "initial": initial,
            }
            if "mask" in batch:
                backbone_input["mask"] = cast(Tensor, batch["mask"])
        if "evaluation_times" in batch:
            backbone_input["evaluation_times"] = cast(Tensor, batch["evaluation_times"])
        backbone_out = backbone(backbone_input)
    else:
        if "values" not in batch:
            raise KeyError("Batch must contain 'values' for non-NCDE backbones")
        backbone_out = backbone(cast(Tensor, batch["values"]))

    if isinstance(head, ClassificationHead):
        logits = head(backbone_out.pooled)
        loss = criterion(logits, labels)
        return loss, logits
    if isinstance(head, RegressionHead):
        preds = head(backbone_out.features)
        loss = criterion(preds, labels)
        return loss, preds
    raise TypeError(f"Unsupported head type: {type(head)!r}")


def train(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device(cfg.training.device)

    LOGGER.info("Instantiating datasets")
    train_root = to_absolute_path(cfg.dataset.root)
    val_root = to_absolute_path(cfg.validation_dataset.root)
    train_dataset = instantiate(cfg.dataset, root=train_root)
    val_dataset = instantiate(cfg.validation_dataset, root=val_root)

    train_loader = build_dataloader(cfg.dataloader, train_dataset)
    val_loader = build_dataloader(cfg.dataloader, val_dataset)

    LOGGER.info("Building model components")
    backbone: Backbone = instantiate(cfg.model.backbone)
    head: Head = instantiate(cfg.model.head)
    backbone.to(device)
    head.to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    criterion: nn.Module
    if isinstance(head, ClassificationHead):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    state = TrainState()
    backbone.train()
    head.train()
    for epoch in range(cfg.training.epochs):
        state.epoch = epoch
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, outputs = step(backbone, head, batch, criterion, device)
            loss.backward()
            optimizer.step()
            state.global_step += 1
            if state.global_step % cfg.training.log_interval == 0:
                LOGGER.info(
                    "Epoch %d step %d loss %.4f",
                    epoch,
                    state.global_step,
                    loss.item(),
                )

        evaluate(backbone, head, val_loader, criterion, device)


def evaluate(
    backbone: Backbone,
    head: Head,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            loss, outputs = step(backbone, head, batch, criterion, device)
            total_loss += loss.item() * batch["label"].size(0)
            total += batch["label"].size(0)
            if isinstance(head, ClassificationHead):
                preds = outputs.argmax(dim=-1)
                correct += (preds == batch["label"].to(device)).sum().item()
    avg_loss = total_loss / max(total, 1)
    if isinstance(head, ClassificationHead):
        accuracy = correct / max(total, 1)
        LOGGER.info("Validation loss %.4f accuracy %.4f", avg_loss, accuracy)
    else:
        LOGGER.info("Validation loss %.4f", avg_loss)
    backbone.train()
    head.train()
