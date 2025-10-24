"""Hydra-driven training entrypoint for OSSM models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

from hydra.utils import instantiate, to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig  # type: ignore[import]
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data.datasets.collate import coeff_collate, pad_collate, path_collate
from ..models import Backbone, BatchOnDevice, ClassificationHead, Head, RegressionHead

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


def step(
    backbone: Backbone,
    head: Head,
    batch: BatchOnDevice,
    criterion: nn.Module,
) -> Tuple[Tensor, Tensor, Tensor]:
    backbone_inputs = backbone.prepare_batch(batch)
    backbone_out = backbone(backbone_inputs)
    targets = head.prepare_batch(batch)

    if isinstance(head, ClassificationHead):
        if backbone_out.pooled is None:
            raise ValueError("Classification heads require pooled representations from the backbone")
        logits = head(backbone_out.pooled)
        loss = criterion(logits, targets)
        return loss, logits, targets
    if isinstance(head, RegressionHead):
        preds = head(backbone_out.features)
        loss = criterion(preds, targets)
        return loss, preds, targets
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
            batch_on_device = BatchOnDevice.from_batch(batch, device=device)
            loss, outputs, _ = step(backbone, head, batch_on_device, criterion)
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
            batch_on_device = BatchOnDevice.from_batch(batch, device=device)
            loss, outputs, targets = step(backbone, head, batch_on_device, criterion)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            if isinstance(head, ClassificationHead):
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
    avg_loss = total_loss / max(total, 1)
    if isinstance(head, ClassificationHead):
        accuracy = correct / max(total, 1)
        LOGGER.info("Validation loss %.4f accuracy %.4f", avg_loss, accuracy)
    else:
        LOGGER.info("Validation loss %.4f", avg_loss)
    backbone.train()
    head.train()
