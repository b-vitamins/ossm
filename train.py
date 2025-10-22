from __future__ import annotations

import argparse
import math
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

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
    DampedLinOSSBackbone,
    Head,
    LRUBackbone,
    LinOSSBackbone,
    NCDEBackbone,
    RNNBackbone,
    RegressionHead,
    S5Backbone,
    SequenceBackboneOutput,
)
from ossm.training import seqrec_main
from ossm.training.progress import ProgressReporter
LOGGER = logging.getLogger(__name__)
COLLATE_FNS = {
    "pad": pad_collate,
    "coeff": coeff_collate,
    "path": path_collate,
}

AVAILABLE_MODELS = ("linoss_im", "dlinoss_imex1", "s5", "lru", "ncde", "rnn", "dlinossrec")
AVAILABLE_HEADS = ("classification", "regression", "tiedsoftmax")
AVAILABLE_TASKS = ("classification", "regression", "seqrec")
# NOTE: Dataset view options are determined by the dataset implementation, not
# the dataloader collate functions. Enumerate the supported UEA views explicitly
# so "raw" remains selectable even though there is no matching collate fn.
AVAILABLE_VIEWS = ("raw", "coeff", "path", "seqrec")
AVAILABLE_OPTIMIZERS = ("adamw",)
AVAILABLE_SCHEDULERS = ("none",)


@dataclass
class _RuntimeSettings:
    cudnn_benchmark: bool = False
    prefetch_gpu: bool = False
    prefetch_depth: int = 2


def _classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    num_classes = logits.size(-1)
    one_hot = F.one_hot(labels, num_classes=num_classes).to(logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return -(one_hot * torch.log(probs + 1e-8)).sum(dim=-1).mean()

def _build_dataloader(cfg: DictConfig, dataset, *, shuffle: Optional[bool] = None) -> DataLoader:
    collate_name = cfg.get("collate", "pad")
    if collate_name not in COLLATE_FNS:
        raise ValueError(f"Unknown collate function '{collate_name}'.")
    collate_fn = COLLATE_FNS[collate_name]
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": cfg.get("shuffle", True) if shuffle is None else shuffle,
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": cfg.get("pin_memory", False),
        "persistent_workers": cfg.get("persistent_workers", False),
        "drop_last": cfg.get("drop_last", False),
        "collate_fn": collate_fn,
    }
    if cfg.get("prefetch_factor") is not None:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    if cfg.get("pin_memory_device"):
        kwargs["pin_memory_device"] = str(cfg.pin_memory_device)
    return DataLoader(
        dataset,
        **kwargs,
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
    elif name == "dlinoss":
        backbone = DampedLinOSSBackbone(
            num_blocks=int(params.get("num_blocks", 4)),
            input_dim=input_dim,
            ssm_size=int(params.get("ssm_size", 64)),
            hidden_dim=int(params.get("hidden_dim", 128)),
            variant=str(params.get("variant", "imex1")),
            initialization=str(params.get("initialization", "ring")),
            r_min=float(params.get("r_min", 0.9)),
            r_max=float(params.get("r_max", 1.0)),
            theta_min=float(params.get("theta_min", 0.0)),
            theta_max=float(params.get("theta_max", math.pi)),
            A_min=float(params.get("A_min", 0.0)),
            A_max=float(params.get("A_max", 1.0)),
            G_min=float(params.get("G_min", 0.0)),
            G_max=float(params.get("G_max", 1.0)),
            dt_std=float(params.get("dt_std", 0.5)),
            dropout=float(params.get("dropout", 0.1)),
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


def _move_batch(
    batch: Dict[str, Tensor], device: torch.device, *, non_blocking: bool = False
) -> Dict[str, Tensor]:
    moved: Dict[str, Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            if value.device == device:
                moved[key] = value
            else:
                moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


class _CudaPrefetcher:
    def __init__(
        self,
        loader: Iterable[Dict[str, Tensor]],
        device: torch.device,
        *,
        depth: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA prefetcher requires a CUDA device")
        self.loader = loader
        self.device = device
        self.depth = max(int(depth), 1)
        self.stream = torch.cuda.Stream(device=device)
        self.buffer: Deque[Dict[str, Tensor]] = deque()
        self.iterator = iter(loader)
        self._warmup()

    def _fetch_next(self) -> Optional[Dict[str, Tensor]]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration:
                return None
        with torch.cuda.stream(self.stream):
            return _move_batch(batch, self.device, non_blocking=True)

    def _warmup(self) -> None:
        while len(self.buffer) < self.depth:
            batch = self._fetch_next()
            if batch is None:
                break
            self.buffer.append(batch)

    def next(self) -> Dict[str, Tensor]:
        if not self.buffer:
            self._warmup()
            if not self.buffer:
                raise RuntimeError("CUDA prefetcher did not receive any batches")
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.buffer.popleft()
        next_batch = self._fetch_next()
        if next_batch is not None:
            self.buffer.append(next_batch)
        return batch


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
    non_blocking: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch = _move_batch(batch, device, non_blocking=non_blocking)
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
    non_blocking: bool,
) -> Tuple[float, Optional[float]]:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            loss, outputs, labels = _training_step(
                backbone,
                head,
                batch,
                device=device,
                classification=classification,
                non_blocking=non_blocking,
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


def _missing_override_value(token: str) -> ValueError:
    message = (
        f"Hydra override '{token}' is missing a value. Use 'key=value' syntax or "
        "a recognised CLI flag (e.g. '--optimizer adamw')."
    )
    return ValueError(message)


def _normalize_hydra_overrides(overrides: Sequence[str]) -> List[str]:
    """Coerce space-separated Hydra overrides into ``key=value`` pairs."""

    normalised: List[str] = []
    i = 0
    while i < len(overrides):
        token = overrides[i]

        if "=" in token or token.startswith(("+", "-", "?", "~")):
            normalised.append(token)
        else:
            if i + 1 >= len(overrides):
                raise _missing_override_value(token)
            next_token = overrides[i + 1]
            if "=" in next_token or next_token.startswith("-"):
                raise _missing_override_value(token)
            normalised.append(f"{token}={next_token}")
            i += 1
        i += 1
    return normalised


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Train OSSM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model and task")
    model_group.add_argument(
        "--model",
        "--backbone",
        dest="model",
        default="linoss_im",
        choices=AVAILABLE_MODELS,
        help="Backbone preset from configs/model.",
    )
    model_group.add_argument(
        "--head",
        default="classification",
        choices=AVAILABLE_HEADS,
        help="Prediction head preset.",
    )
    model_group.add_argument(
        "--task",
        default=None,
        choices=AVAILABLE_TASKS,
        help="Training objective; defaults to the selected head.",
    )
    model_group.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override the backbone hidden dimension.",
    )
    model_group.add_argument(
        "--ssm-size",
        type=int,
        default=None,
        help="Override the state-space dimension for SSM backbones.",
    )
    model_group.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Override the number of backbone blocks.",
    )

    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset-name",
        default="EthanolConcentration",
        help="UEA dataset name.",
    )
    data_group.add_argument(
        "--dataset-view",
        default=None,
        choices=AVAILABLE_VIEWS,
        help="Dataset representation to materialise.",
    )
    data_group.add_argument(
        "--dataset-root",
        default=None,
        help="Override the dataset root directory.",
    )
    data_group.add_argument(
        "--train-split",
        default="train",
        help="Split to use for the training dataset.",
    )
    data_group.add_argument(
        "--val-name",
        default=None,
        help="Optional validation dataset name (defaults to training dataset).",
    )
    data_group.add_argument(
        "--val-view",
        default=None,
        help="Optional validation dataset view (defaults to training view).",
    )
    data_group.add_argument(
        "--val-split",
        default="val",
        help="Split to use for the validation dataset.",
    )
    data_group.add_argument(
        "--test-name",
        default=None,
        help="Optional test dataset name (defaults to validation dataset).",
    )
    data_group.add_argument(
        "--test-view",
        default=None,
        help="Optional test dataset view (defaults to validation view).",
    )
    data_group.add_argument(
        "--test-split",
        default="test",
        help="Split to use for the test dataset.",
    )
    data_group.add_argument(
        "--window-steps",
        type=int,
        default=None,
        help="Number of windows for log-signature features (view='path').",
    )
    data_group.add_argument(
        "--window-depth",
        type=int,
        default=None,
        help="Log-signature depth (view='path').",
    )
    data_group.add_argument(
        "--logsig-basis",
        default=None,
        help="Log-signature basis for path view ('hall' or 'lyndon').",
    )
    data_group.add_argument(
        "--record-grid",
        action="store_true",
        help="Persist the original time grid in each sample.",
    )
    data_group.add_argument(
        "--record-source",
        action="store_true",
        help="Persist source index metadata in each sample.",
    )
    data_group.add_argument(
        "--download",
        action="store_true",
        help="Download and prepare the dataset layout if missing.",
    )

    loader_group = parser.add_argument_group("Dataloader")
    loader_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the training batch size.",
    )
    loader_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes.",
    )
    loader_group.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Number of batches loaded in advance per worker.",
    )
    loader_group.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep dataloader workers alive between epochs.",
    )
    loader_group.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin dataloader memory for faster host-to-device transfers.",
    )
    loader_group.add_argument(
        "--pin-memory-device",
        default=None,
        help="Device for pinned memory (PyTorch >= 2.1).",
    )
    loader_group.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the last incomplete batch.",
    )
    loader_group.add_argument(
        "--collate",
        choices=tuple(COLLATE_FNS.keys()),
        default=None,
        help="Collate function to use for batching.",
    )

    optim_group = parser.add_argument_group("Optimisation")
    optim_group.add_argument(
        "--optimizer",
        default="adamw",
        choices=AVAILABLE_OPTIMIZERS,
        help="Optimizer preset.",
    )
    optim_group.add_argument(
        "--scheduler",
        default="none",
        choices=AVAILABLE_SCHEDULERS,
        help="Scheduler preset.",
    )
    optim_group.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate.",
    )
    optim_group.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override the weight decay coefficient.",
    )
    optim_group.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping value.",
    )

    train_group = parser.add_argument_group("Training schedule")
    train_group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of optimisation steps.",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch budget (converted to steps using the dataloader length).",
    )
    train_group.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Logging interval in steps.",
    )
    train_group.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Evaluation interval in steps.",
    )
    train_group.add_argument(
        "--prefetch-gpu",
        action="store_true",
        help="Stream batches to GPU using a background CUDA stream.",
    )
    train_group.add_argument(
        "--prefetch-depth",
        type=int,
        default=2,
        help="Number of batches prepared ahead of time for GPU prefetching.",
    )
    train_group.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cuDNN benchmark autotuning (GPU only).",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override validation/test batch size.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for PyTorch and CUDA.",
    )
    runtime_group.add_argument(
        "--device",
        default="auto",
        help="Device to use (e.g. 'cpu', 'cuda', 'cuda:1', or 'auto').",
    )
    runtime_group.add_argument(
        "--work-dir",
        default=None,
        help="Override the Hydra work directory (paths.work_dir).",
    )

    argv_list: Sequence[str]
    if argv is None:
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)

    args, unknown = parser.parse_known_args(argv_list)

    device_flag = False
    head_flag = False
    model_flag = False
    dataset_view_flag = False
    task_flag = False
    for entry in argv_list:
        if entry == "--device" or entry.startswith("--device="):
            device_flag = True
        if entry == "--head" or entry.startswith("--head="):
            head_flag = True
        if entry == "--model" or entry.startswith("--model=") or entry.startswith("--backbone="):
            model_flag = True
        if entry == "--dataset-view" or entry.startswith("--dataset-view="):
            dataset_view_flag = True
        if entry == "--task" or entry.startswith("--task="):
            task_flag = True

    if args.task is None and args.head == "tiedsoftmax":
        args.task = "seqrec"

    if args.task == "seqrec":
        if not model_flag:
            args.model = "dlinossrec"
        if not head_flag:
            args.head = "tiedsoftmax"
        if not dataset_view_flag:
            args.dataset_view = "seqrec"

    setattr(args, "_device_from_cli", device_flag)
    setattr(args, "_head_from_cli", head_flag)
    setattr(args, "_model_from_cli", model_flag)
    setattr(args, "_dataset_view_from_cli", dataset_view_flag)
    setattr(args, "_task_from_cli", task_flag)
    return args, unknown


def _compose_config(args: argparse.Namespace, extra_overrides: Sequence[str]) -> DictConfig:
    task = args.task if args.task is not None else args.head
    if task != "seqrec" and args.model == "dlinossrec":
        raise ValueError("Model 'dlinossrec' requires --task seqrec.")
    if task != "seqrec" and args.head == "tiedsoftmax":
        raise ValueError("Head 'tiedsoftmax' requires --task seqrec.")

    overrides: List[str] = [
        f"model={args.model}",
        f"head={args.head}",
        f"optimizer={args.optimizer}",
        f"scheduler={args.scheduler}",
        f"training={task}",
        f"dataset.split={args.train_split}",
    ]
    if task == "seqrec":
        overrides.append(f"dataset={args.dataset_name}")
    overrides.append(f"dataset.name={args.dataset_name}")
    if args.dataset_view is not None:
        overrides.append(f"dataset.view={args.dataset_view}")

    val_name = args.val_name or args.dataset_name
    if task == "seqrec":
        overrides.append(f"validation_dataset={val_name}")
    overrides.append(f"validation_dataset.name={val_name}")
    if args.val_view is not None:
        overrides.append(f"validation_dataset.view={args.val_view}")
    elif args.dataset_view is not None:
        overrides.append(f"validation_dataset.view={args.dataset_view}")
    overrides.append(f"validation_dataset.split={args.val_split}")

    test_name = args.test_name or val_name
    if task == "seqrec":
        overrides.append(f"test_dataset={test_name}")
    overrides.append(f"test_dataset.name={test_name}")
    if args.test_view is not None:
        overrides.append(f"test_dataset.view={args.test_view}")
    elif args.val_view is not None:
        overrides.append(f"test_dataset.view={args.val_view}")
    elif args.dataset_view is not None:
        overrides.append(f"test_dataset.view={args.dataset_view}")
    overrides.append(f"test_dataset.split={args.test_split}")

    if args.dataset_root:
        overrides.append(f"paths.data_root={args.dataset_root}")
    if args.work_dir:
        overrides.append(f"paths.work_dir={args.work_dir}")

    if args.collate:
        overrides.append(f"dataloader.collate={args.collate}")
    if args.batch_size is not None:
        overrides.append(f"dataloader.batch_size={args.batch_size}")
        if task == "seqrec":
            overrides.append(f"training.batch_size={args.batch_size}")
    if args.eval_batch_size is not None and task == "seqrec":
        overrides.append(f"training.eval_batch_size={args.eval_batch_size}")
    if args.num_workers is not None:
        overrides.append(f"dataloader.num_workers={args.num_workers}")
        if task == "seqrec":
            overrides.extend(
                [
                    f"dataset.num_workers={args.num_workers}",
                    f"validation_dataset.num_workers={args.num_workers}",
                    f"test_dataset.num_workers={args.num_workers}",
                ]
            )
    if args.prefetch_factor is not None:
        overrides.append(f"dataloader.prefetch_factor={args.prefetch_factor}")
    if args.persistent_workers:
        overrides.append("dataloader.persistent_workers=true")
    if args.pin_memory:
        overrides.append("dataloader.pin_memory=true")
        if task == "seqrec":
            overrides.extend(
                [
                    "dataset.pin_memory=true",
                    "validation_dataset.pin_memory=true",
                    "test_dataset.pin_memory=true",
                ]
            )
    if args.pin_memory_device:
        overrides.append(f"dataloader.pin_memory_device={args.pin_memory_device}")
    if args.drop_last:
        overrides.append("dataloader.drop_last=true")

    if args.max_steps is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.log_interval is not None:
        overrides.append(f"training.log_interval={args.log_interval}")
    if args.eval_interval is not None:
        overrides.append(f"training.eval_interval={args.eval_interval}")
    if args.grad_clip is not None:
        overrides.append(f"training.grad_clip={args.grad_clip}")
    if args.prefetch_gpu:
        overrides.append("training.prefetch_gpu=true")
        overrides.append(f"training.prefetch_depth={args.prefetch_depth}")
        overrides.append("dataloader.pin_memory=true")
    elif args.prefetch_depth is not None:
        overrides.append(f"training.prefetch_depth={args.prefetch_depth}")
    if args.cudnn_benchmark:
        overrides.append("training.cudnn_benchmark=true")

    if args.lr is not None:
        overrides.append(f"optimizer.lr={args.lr}")
    if args.weight_decay is not None:
        overrides.append(f"optimizer.weight_decay={args.weight_decay}")

    if args.hidden_dim is not None:
        overrides.append(f"model.params.hidden_dim={args.hidden_dim}")
    if args.ssm_size is not None:
        overrides.append(f"model.params.ssm_size={args.ssm_size}")
    if args.num_blocks is not None:
        overrides.append(f"model.params.num_blocks={args.num_blocks}")

    if args.window_steps is not None:
        overrides.extend(
            [
                f"dataset.steps={args.window_steps}",
                f"validation_dataset.steps={args.window_steps}",
                f"test_dataset.steps={args.window_steps}",
            ]
        )
    if args.window_depth is not None:
        overrides.extend(
            [
                f"dataset.depth={args.window_depth}",
                f"validation_dataset.depth={args.window_depth}",
                f"test_dataset.depth={args.window_depth}",
            ]
        )
    if args.logsig_basis is not None:
        overrides.extend(
            [
                f"dataset.basis={args.logsig_basis}",
                f"validation_dataset.basis={args.logsig_basis}",
                f"test_dataset.basis={args.logsig_basis}",
            ]
        )
    if args.record_grid:
        overrides.extend(
            [
                "dataset.record_grid=true",
                "validation_dataset.record_grid=true",
                "test_dataset.record_grid=true",
            ]
        )
    if args.record_source:
        overrides.extend(
            [
                "dataset.record_source=true",
                "validation_dataset.record_source=true",
                "test_dataset.record_source=true",
            ]
        )
    if args.download:
        overrides.extend(
            [
                "dataset.download=true",
                "validation_dataset.download=true",
                "test_dataset.download=true",
            ]
        )

    overrides.extend(
        [
            f"seed={args.seed}",
            f"dataset.resample_seed={args.seed}",
            f"validation_dataset.resample_seed={args.seed}",
            f"test_dataset.resample_seed={args.seed}",
        ]
    )

    overrides.extend(_normalize_hydra_overrides(extra_overrides))

    with initialize(config_path="configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _select_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_training(cfg: DictConfig, device: torch.device, runtime: _RuntimeSettings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if runtime.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]

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

    runtime.prefetch_gpu = runtime.prefetch_gpu and device.type == "cuda" and torch.cuda.is_available()
    non_blocking = device.type == "cuda" and (
        runtime.prefetch_gpu or bool(cfg.dataloader.get("pin_memory", False))
    )
    max_steps = int(cfg.training.max_steps)
    log_interval = int(cfg.training.log_interval)
    eval_interval = int(cfg.training.eval_interval)
    grad_clip = cfg.training.get("grad_clip")

    try:
        batches_per_epoch = len(train_loader)
    except TypeError:  # pragma: no cover - dataloader without __len__
        batches_per_epoch = None
    epochs = cfg.training.get("epochs")
    if epochs is not None:
        if not isinstance(batches_per_epoch, int):
            raise ValueError("Cannot derive steps from epochs without a sized dataloader")
        max_steps = int(epochs) * batches_per_epoch
    progress = ProgressReporter(max_steps)

    device_label = device.type if device.index is None else f"{device.type}:{device.index}"
    batches_display = batches_per_epoch if batches_per_epoch is not None else "?"
    extras = []
    if runtime.prefetch_gpu:
        extras.append(f"prefetch_gpu(depth={runtime.prefetch_depth})")
    extras_str = f" {' '.join(extras)}" if extras else ""
    print(
        "Training • "
        f"task={'classification' if classification else 'regression'} "
        f"steps={max_steps:,} "
        f"device={device_label} "
        f"batch_size={cfg.dataloader.batch_size} "
        f"batches/epoch={batches_display} "
        f"train_samples={len(train_dataset)}" + extras_str
    )

    step = 0
    train_iter = iter(train_loader)
    prefetcher: Optional[_CudaPrefetcher] = None
    if runtime.prefetch_gpu:
        prefetcher = _CudaPrefetcher(train_loader, device, depth=runtime.prefetch_depth)
        train_iter = None  # type: ignore[assignment]
    last_eval: Dict[str, Tuple[float, Optional[float]]] = {}

    while step < max_steps:
        if runtime.prefetch_gpu and prefetcher is not None:
            batch = prefetcher.next()
        else:
            assert train_iter is not None
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)
        loss, outputs, labels = _training_step(
            backbone,
            head,
            batch,
            device=device,
            classification=classification,
            non_blocking=non_blocking,
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
                backbone,
                head,
                val_loader,
                device=device,
                classification=classification,
                non_blocking=non_blocking,
            )
            last_eval["val"] = (val_loss, val_acc)
            print(f"Eval step {step:05d} • {_format_eval('val', val_loss, val_acc)}")
            if test_loader is not None:
                test_loss, test_acc = _evaluate_split(
                    backbone,
                    head,
                    test_loader,
                    device=device,
                    classification=classification,
                    non_blocking=non_blocking,
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
    device_from_cli = getattr(args, "_device_from_cli", False)
    if str(cfg.training.get("task", "")) == "seqrec":
        if device_from_cli:
            cfg.training.device = args.device
        seqrec_main(cfg)
        return
    device = _select_device(args.device)
    runtime = _RuntimeSettings(
        cudnn_benchmark=bool(cfg.training.get("cudnn_benchmark", False)),
        prefetch_gpu=bool(cfg.training.get("prefetch_gpu", False)),
        prefetch_depth=int(cfg.training.get("prefetch_depth", 2)),
    )
    run_training(cfg, device, runtime)


if __name__ == "__main__":
    main(sys.argv[1:])
