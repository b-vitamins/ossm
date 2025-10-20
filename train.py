from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hydra import compose, initialize  # type: ignore[import]
from hydra.utils import instantiate, to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf  # type: ignore[import]

sys.path.append(str(Path(__file__).resolve().parent / "src"))
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ossm.data.datasets.collate import coeff_collate, pad_collate, path_collate
from ossm.models import (
    Backbone,
    ClassificationHead,
    Head,
    LRUBackbone,
    LinOSSBackbone,
    NCDEBackbone,
    RNNBackbone,
    RegressionHead,
    S5Backbone,
    SequenceBackboneOutput,
)
LOGGER = logging.getLogger(__name__)
COLLATE_FNS = {
    "pad": pad_collate,
    "coeff": coeff_collate,
    "path": path_collate,
}


def _classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    num_classes = logits.size(-1)
    one_hot = F.one_hot(labels, num_classes=num_classes).to(logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return -(one_hot * torch.log(probs + 1e-8)).sum(dim=-1).mean()


def _format_duration(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class _ProgressReporter:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.interval_examples = 0

    def update(self, batch_size: int) -> None:
        self.interval_examples += int(batch_size)

    def log(
        self,
        step: int,
        loss: float,
        *,
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        now = time.perf_counter()
        interval = max(now - self.last_log_time, 1e-9)
        elapsed = now - self.start_time
        remaining = max(self.total_steps - step, 0)
        avg_step = elapsed / max(step, 1)
        eta = avg_step * remaining
        ips = self.interval_examples / interval if self.interval_examples else 0.0

        parts = [f"Step {step:05d}/{self.total_steps:05d}", f"Loss = {loss:.4f}"]
        if metrics:
            for name, value in metrics.items():
                parts.append(f"{name} = {value:.4f}")
        if lr is not None:
            parts.append(f"LR = {lr:.2e}")
        parts.append(f"IPS = {ips:,.1f}")
        parts.append(f"Time = {_format_duration(elapsed)}")
        parts.append(f"ETA = {_format_duration(eta)}")
        print(" \u2022 ".join(parts))

        self.last_log_time = now
        self.interval_examples = 0

    def summary(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(f"Training finished • Time = {_format_duration(elapsed)}")

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
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _run_backbone(backbone: Backbone, batch: Dict[str, Tensor]) -> SequenceBackboneOutput:
    if isinstance(backbone, NCDEBackbone):
        times = batch["times"]
        initial = batch["initial"]
        if "logsig" in batch:
            backbone_input = {
                "times": times,
                "logsig": batch["logsig"],
                "initial": initial,
            }
        else:
            backbone_input = {
                "times": times,
                "coeffs": batch["coeffs"],
                "initial": initial,
            }
            if "mask" in batch:
                backbone_input["mask"] = batch["mask"]
        if "evaluation_times" in batch:
            backbone_input["evaluation_times"] = batch["evaluation_times"]
        return backbone(backbone_input)
    if "values" not in batch:
        raise KeyError("Batch must contain 'values' for non-NCDE backbones")
    return backbone(batch["values"])


def _training_step(
    backbone: Backbone,
    head: Head,
    batch: Dict[str, Tensor],
    *,
    device: torch.device,
    classification: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch = _move_batch(batch, device)
    labels = batch["label"]
    backbone_out = _run_backbone(backbone, batch)
    if classification:
        logits = head(backbone_out.pooled)
        loss = _classification_loss(logits, labels)
        return loss, logits, labels
    preds = head(backbone_out.features)
    loss = F.mse_loss(preds, labels)
    return loss, preds, labels


def _evaluate_split(
    backbone: Backbone,
    head: Head,
    loader: Iterable[Dict[str, Tensor]],
    *,
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
            loss, outputs, labels = _training_step(
                backbone, head, batch, device=device, classification=classification
            )
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            if classification:
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
    backbone.train()
    head.train()
    avg_loss = total_loss / max(total, 1)
    if classification:
        return avg_loss, correct / max(total, 1)
    return avg_loss, None


def _format_eval(split: str, loss: float, accuracy: Optional[float]) -> str:
    parts = [f"split={split}", f"loss={loss:.4f}"]
    if accuracy is not None:
        parts.append(f"acc={accuracy:.4f}")
    return " • ".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Train OSSM models")
    parser.add_argument("--model", default="linoss_im", help="Model configuration name")
    parser.add_argument("--dataset", default="EthanolConcentration", help="Dataset name")
    parser.add_argument("--dataset-view", default="raw", help="Dataset view")
    parser.add_argument("--validation-dataset", default=None, help="Validation dataset name")
    parser.add_argument("--validation-view", default=None, help="Validation dataset view")
    parser.add_argument("--test-dataset", default=None, help="Test dataset name")
    parser.add_argument("--test-view", default=None, help="Test dataset view")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--max-steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--log-interval", type=int, default=None, help="Logging interval (steps)")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval (steps)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Backbone hidden dimension")
    parser.add_argument("--ssm-size", type=int, default=None, help="State-space dimension")
    parser.add_argument("--num-blocks", type=int, default=None, help="Number of backbone blocks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device to use (e.g. 'cpu', 'cuda', 'auto')")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--collate", default=None, help="Dataloader collate function")
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def _compose_config(args: argparse.Namespace, extra_overrides: Sequence[str]) -> DictConfig:
    overrides: List[str] = [
        f"model={args.model}",
        f"dataset.name={args.dataset}",
        f"dataset.view={args.dataset_view}",
    ]

    val_dataset = args.validation_dataset or args.dataset
    val_view = args.validation_view or args.dataset_view
    overrides.extend(
        [f"validation_dataset.name={val_dataset}", f"validation_dataset.view={val_view}"]
    )

    test_dataset = args.test_dataset or val_dataset
    test_view = args.test_view or val_view
    overrides.extend(
        [f"test_dataset.name={test_dataset}", f"test_dataset.view={test_view}"]
    )

    if args.data_root:
        overrides.append(f"paths.data_root={args.data_root}")
    if args.collate:
        overrides.append(f"dataloader.collate={args.collate}")
    if args.batch_size is not None:
        overrides.append(f"dataloader.batch_size={args.batch_size}")
    if args.max_steps is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    if args.log_interval is not None:
        overrides.append(f"training.log_interval={args.log_interval}")
    if args.eval_interval is not None:
        overrides.append(f"training.eval_interval={args.eval_interval}")
    if args.lr is not None:
        overrides.append(f"optimizer.lr={args.lr}")
    if args.weight_decay is not None:
        overrides.append(f"optimizer.weight_decay={args.weight_decay}")
    if args.grad_clip is not None:
        overrides.append(f"training.grad_clip={args.grad_clip}")
    if args.hidden_dim is not None:
        overrides.append(f"model.params.hidden_dim={args.hidden_dim}")
    if args.ssm_size is not None:
        overrides.append(f"model.params.ssm_size={args.ssm_size}")
    if args.num_blocks is not None:
        overrides.append(f"model.params.num_blocks={args.num_blocks}")

    overrides.extend(
        [
            f"seed={args.seed}",
            f"dataset.resample_seed={args.seed}",
            f"validation_dataset.resample_seed={args.seed}",
            f"test_dataset.resample_seed={args.seed}",
        ]
    )

    overrides.extend(extra_overrides)

    with initialize(config_path="configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _select_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_training(cfg: DictConfig, device: torch.device) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_root = to_absolute_path(cfg.dataset.root)
    val_root = to_absolute_path(cfg.validation_dataset.root)
    train_dataset = instantiate(cfg.dataset, root=train_root)
    val_dataset = instantiate(cfg.validation_dataset, root=val_root)

    test_dataset = None
    test_loader = None
    if "test_dataset" in cfg and cfg.test_dataset is not None:
        test_root = to_absolute_path(cfg.test_dataset.root)
        test_dataset = instantiate(cfg.test_dataset, root=test_root)

    classification = bool(cfg.training.classification)
    input_dim, target_dim = _infer_dataset_metadata(train_dataset, classification=classification)
    backbone, hidden_dim = _build_backbone(cfg.model, input_dim, dataset=train_dataset)
    head = _build_head(cfg.head, hidden_dim, target_dim)

    backbone.to(device)
    head.to(device)

    train_loader = _build_dataloader(cfg.dataloader, train_dataset)
    val_loader = _build_dataloader(cfg.dataloader, val_dataset, shuffle=False)
    if test_dataset is not None:
        test_loader = _build_dataloader(cfg.dataloader, test_dataset, shuffle=False)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = instantiate(cfg.optimizer, params=params)
    scheduler = None
    if cfg.scheduler.get("enabled", False):
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    max_steps = int(cfg.training.max_steps)
    log_interval = int(cfg.training.log_interval)
    eval_interval = int(cfg.training.eval_interval)
    grad_clip = cfg.training.get("grad_clip")

    progress = _ProgressReporter(max_steps)
    device_label = device.type if device.index is None else f"{device.type}:{device.index}"
    try:
        batches_per_epoch = len(train_loader)
    except TypeError:  # pragma: no cover - dataloader without __len__
        batches_per_epoch = "?"
    print(
        f"Training • steps={max_steps:,} device={device_label} batches/epoch={batches_per_epoch}"
    )

    step = 0
    train_iter = iter(train_loader)
    last_eval: Dict[str, Tuple[float, Optional[float]]] = {}

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)
        loss, outputs, labels = _training_step(
            backbone, head, batch, device=device, classification=classification
        )
        loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(params, float(grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step += 1
        batch_size = labels.size(0)
        progress.update(int(batch_size))

        if step % log_interval == 0:
            metrics: Dict[str, float] = {}
            if classification:
                preds = outputs.argmax(dim=-1)
                metrics["Acc(train)"] = (preds == labels).float().mean().item()
            if "val" in last_eval and last_eval["val"][1] is not None:
                metrics["Acc(val)"] = last_eval["val"][1]
            if "test" in last_eval and last_eval["test"][1] is not None:
                metrics["Acc(test)"] = last_eval["test"][1]
            lr = optimizer.param_groups[0]["lr"]
            progress.log(step, loss.item(), metrics=metrics or None, lr=lr)

        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_acc = _evaluate_split(
                backbone, head, val_loader, device=device, classification=classification
            )
            last_eval["val"] = (val_loss, val_acc)
            print(f"Eval step {step:05d} • {_format_eval('val', val_loss, val_acc)}")
            if test_loader is not None:
                test_loss, test_acc = _evaluate_split(
                    backbone, head, test_loader, device=device, classification=classification
                )
                last_eval["test"] = (test_loss, test_acc)
                print(f"Eval step {step:05d} • {_format_eval('test', test_loss, test_acc)}")

    progress.summary()

    if last_eval:
        summary_parts = ["Final eval"]
        if "val" in last_eval:
            val_loss, val_acc = last_eval["val"]
            summary_parts.append(f"val loss={val_loss:.4f}")
            if val_acc is not None:
                summary_parts.append(f"val acc={val_acc:.4f}")
        if "test" in last_eval:
            test_loss, test_acc = last_eval["test"]
            summary_parts.append(f"test loss={test_loss:.4f}")
            if test_acc is not None:
                summary_parts.append(f"test acc={test_acc:.4f}")
        print(" • ".join(summary_parts))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, extra = parse_args(argv)
    cfg = _compose_config(args, extra)
    device = _select_device(args.device)
    run_training(cfg, device)


if __name__ == "__main__":
    main(sys.argv[1:])
