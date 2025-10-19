from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import hydra  # type: ignore[import]
from hydra.utils import instantiate, to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf  # type: ignore[import]
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ossm.data.datasets.collate import coeff_collate, pad_collate, path_collate
from ossm.models import (
    LRUBackbone,
    LinOSSBackbone,
    NCDEBackbone,
    RNNBackbone,
    S5Backbone,
    ClassificationHead,
    RegressionHead,
)
from ossm.training.train import step as training_step

LOGGER = logging.getLogger(__name__)
COLLATE_FNS = {
    "pad": pad_collate,
    "coeff": coeff_collate,
    "path": path_collate,
}


def _build_dataloader(cfg: DictConfig, dataset, *, shuffle: Optional[bool] = None) -> DataLoader:
    collate_name = cfg.get("collate", "pad")
    if collate_name not in COLLATE_FNS:
        raise ValueError(f"Unknown collate function '{collate_name}'.")
    collate_fn = COLLATE_FNS[collate_name]
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.get("shuffle", True) if shuffle is None else shuffle,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False),
        collate_fn=collate_fn,
    )


def _infer_dataset_metadata(dataset, *, classification: bool) -> Tuple[int, int]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; unable to infer metadata")
    sample = dataset[0]
    if "values" not in sample:
        raise KeyError("Dataset samples must provide a 'values' tensor")
    values = sample["values"]
    if values.ndim != 2:
        raise ValueError("Sample 'values' must have shape (length, channels)")
    input_dim = int(values.size(-1))
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise AttributeError("Dataset must expose a 'labels' tensor to infer targets")
    if classification:
        target_dim = int(torch.unique(labels).numel())
    else:
        target_dim = int(labels.shape[-1] if labels.ndim > 1 else 1)
    return input_dim, target_dim


def _infer_nrde_metadata(dataset) -> Tuple[int, torch.Tensor]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; unable to infer NRDE metadata")

    logsig_dim: Optional[int] = None
    max_segments = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    interval_dtype: Optional[torch.dtype] = None

    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = sample.get("features")
        if features is None:
            raise ValueError("NRDE mode requires dataset samples to include 'features'")
        if features.ndim != 2:
            raise ValueError("Sample 'features' must have shape (segments, channels)")

        segments = int(features.size(0))
        if segments > max_segments:
            max_segments = segments

        if segments > 0:
            dim = int(features.size(-1))
            if logsig_dim is None:
                logsig_dim = dim
            elif dim != logsig_dim:
                raise ValueError("Inconsistent log-signature dimensions across samples")

        times = sample.get("times")
        if times is not None and times.ndim == 1 and times.numel() > 1:
            interval_dtype = times.dtype
            start = float(times[0])
            end = float(times[-1])
            start_time = start if start_time is None else min(start_time, start)
            end_time = end if end_time is None else max(end_time, end)
        elif interval_dtype is None and features.numel():
            interval_dtype = features.dtype

    if logsig_dim is None:
        raise ValueError("Unable to infer log-signature dimension from dataset")
    if max_segments == 0:
        raise ValueError("Unable to infer NRDE intervals from empty feature tensors")

    if start_time is None or end_time is None:
        start_time, end_time = 0.0, 1.0
    if interval_dtype is None:
        interval_dtype = torch.float32

    intervals = torch.linspace(start_time, end_time, max_segments + 1, dtype=interval_dtype)
    return logsig_dim, intervals


def _build_backbone(
    model_cfg: DictConfig, input_dim: int, dataset=None
) -> Tuple[nn.Module, int]:
    name = model_cfg.name.lower()
    params = OmegaConf.to_container(model_cfg.params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("model.params must be a mapping")
    if name == "linoss":
        backbone = LinOSSBackbone(
            num_blocks=int(params.get("num_blocks", 4)),
            input_dim=input_dim,
            ssm_size=int(params.get("ssm_size", 64)),
            hidden_dim=int(params.get("hidden_dim", 128)),
            discretization=str(params.get("discretization", "IM")),
        )
    elif name == "s5":
        backbone = S5Backbone(
            num_blocks=int(params.get("num_blocks", 4)),
            input_dim=input_dim,
            ssm_size=int(params.get("ssm_size", 64)),
            ssm_blocks=int(params.get("ssm_blocks", 1)),
            hidden_dim=int(params.get("hidden_dim", 128)),
            C_init=str(params.get("C_init", "lecun_normal")),
            conj_sym=bool(params.get("conj_sym", True)),
            clip_eigs=bool(params.get("clip_eigs", False)),
            discretization=str(params.get("discretization", "zoh")),
            dt_min=float(params.get("dt_min", 1e-3)),
            dt_max=float(params.get("dt_max", 1e-1)),
            step_rescale=float(params.get("step_rescale", 1.0)),
            dropout=float(params.get("dropout", 0.05)),
        )
    elif name == "lru":
        backbone = LRUBackbone(
            num_blocks=int(params.get("num_blocks", 4)),
            input_dim=input_dim,
            ssm_size=int(params.get("ssm_size", 64)),
            hidden_dim=int(params.get("hidden_dim", 128)),
            dropout=float(params.get("dropout", 0.1)),
            r_min=float(params.get("r_min", 0.0)),
            r_max=float(params.get("r_max", 1.0)),
            max_phase=float(params.get("max_phase", 6.28318)),
        )
    elif name == "rnn":
        backbone = RNNBackbone(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            cell=str(params.get("cell", "linear")),
            mlp_depth=int(params.get("depth", params.get("mlp_depth", 1))),
            mlp_width=int(params.get("mlp_width", 128)),
        )
    elif name == "ncde":
        mode = str(params.get("mode", "ncde")).lower()
        if mode == "nrde" and dataset is not None:
            need_logsig = "logsig_dim" not in params
            need_intervals = "intervals" not in params
            if need_logsig or need_intervals:
                logsig_dim, intervals = _infer_nrde_metadata(dataset)
                if need_logsig:
                    params["logsig_dim"] = logsig_dim
                if need_intervals:
                    params["intervals"] = intervals.tolist()
        backbone = NCDEBackbone(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            vf_width=int(params.get("vf_width", params.get("hidden_dim", 128))),
            vf_depth=int(params.get("vf_depth", 2)),
            activation=str(params.get("activation", "relu")),
            scale=float(params.get("scale", 1.0)),
            solver=str(params.get("solver", "heun2")),
            step_size=float(params.get("step_size", 1.0)),
            rtol=float(params.get("rtol", 1e-4)),
            atol=float(params.get("atol", 1e-5)),
            mode=mode,
            logsig_dim=params.get("logsig_dim"),
            intervals=params.get("intervals"),
        )
    else:
        raise ValueError(f"Unsupported model '{model_cfg.name}'.")
    return backbone, backbone.hidden_dim


def _build_head(head_cfg: DictConfig, hidden_dim: int, num_outputs: int) -> nn.Module:
    name = head_cfg.name.lower()
    params = OmegaConf.to_container(head_cfg.params, resolve=True) or {}
    if name == "classification":
        dropout = float(params.get("dropout", 0.0))
        return ClassificationHead(hidden_dim, num_outputs, dropout=dropout)
    if name == "regression":
        return RegressionHead(hidden_dim, num_outputs)
    raise ValueError(f"Unsupported head '{head_cfg.name}'.")


def _move_batch(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _evaluate(
    backbone: nn.Module,
    head: nn.Module,
    loader: Iterable[Dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    classification: bool,
) -> Tuple[float, Optional[float]]:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            labels = batch["label"]
            loss, outputs = training_step(backbone, head, batch, criterion, device)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            if classification:
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
    backbone.train()
    head.train()
    avg_loss = total_loss / max(total, 1)
    if classification:
        return avg_loss, correct / max(total, 1)
    return avg_loss, None


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(int(cfg.seed))

    train_root = to_absolute_path(cfg.dataset.root)
    val_root = to_absolute_path(cfg.validation_dataset.root)
    train_dataset = instantiate(cfg.dataset, root=train_root)
    val_dataset = instantiate(cfg.validation_dataset, root=val_root)

    input_dim, target_dim = _infer_dataset_metadata(train_dataset, classification=cfg.training.classification)
    backbone, hidden_dim = _build_backbone(cfg.model, input_dim, dataset=train_dataset)
    head = _build_head(cfg.head, hidden_dim, target_dim)

    device = torch.device(cfg.training.device)
    backbone.to(device)
    head.to(device)

    train_loader = _build_dataloader(cfg.dataloader, train_dataset)
    val_loader = _build_dataloader(cfg.dataloader, val_dataset, shuffle=False)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = instantiate(cfg.optimizer, params=params)

    scheduler = None
    if cfg.scheduler.get("enabled", False):
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    if cfg.training.classification:
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    max_steps = int(cfg.training.max_steps)
    log_interval = int(cfg.training.log_interval)
    eval_interval = int(cfg.training.eval_interval)
    grad_clip = cfg.training.get("grad_clip")

    LOGGER.info("Starting training for %d steps", max_steps)
    step = 0
    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss, outputs = training_step(backbone, head, batch, criterion, device)
        loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(params, float(grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step += 1
        if step % log_interval == 0:
            labels = batch["label"]
            if cfg.training.classification:
                preds = outputs.argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
                LOGGER.info("Step %d loss %.4f accuracy %.4f", step, loss.item(), accuracy)
            else:
                LOGGER.info("Step %d loss %.4f", step, loss.item())

        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_metric = _evaluate(
                backbone, head, val_loader, criterion, device, cfg.training.classification
            )
            if cfg.training.classification and val_metric is not None:
                LOGGER.info("Validation loss %.4f accuracy %.4f", val_loss, val_metric)
            else:
                LOGGER.info("Validation loss %.4f", val_loss)

    LOGGER.info("Training finished")


if __name__ == "__main__":
    main()
