"""Sequential recommendation training entrypoint."""

from __future__ import annotations

import json
import logging
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, ContextManager, Dict, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf  # type: ignore[import]
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..data.datasets import SeqRecEvalDataset, SeqRecTrainDataset, collate_left_pad
from ..metrics import TopKMetricAccumulator, mask_history_inplace
from ..models.dlinossrec import Dlinoss4Rec
from .progress import ProgressReporter, format_duration

# Resolve AMP utilities in a version-tolerant manner without triggering type-checker errors.
_torch_amp = getattr(torch, "amp", None)
if _torch_amp is not None and hasattr(_torch_amp, "GradScaler") and hasattr(
    _torch_amp, "autocast"
):  # pragma: no cover - runtime feature detection
    _GradScaler = getattr(_torch_amp, "GradScaler")
    _autocast = getattr(_torch_amp, "autocast")
else:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore[import]
    from torch.cuda.amp import autocast as _autocast  # type: ignore[import]


def _make_grad_scaler(device_type: str, enabled: bool) -> Any:
    """Instantiate a GradScaler compatible across torch versions."""

    try:
        return _GradScaler(device_type=device_type, enabled=enabled)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - fallback path
        return _GradScaler(enabled=enabled)


def _autocast_context(device_type: str, enabled: bool) -> ContextManager[Any]:
    """Return an autocast context manager across torch versions."""

    try:
        return _autocast(device_type=device_type, enabled=enabled)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - fallback path
        return _autocast(enabled=enabled)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    cfg: DictConfig,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dict[str, Dict[int, torch.Tensor]],
]:
    dataset_cfg = cfg.dataset
    max_len = int(dataset_cfg.max_len)
    train_root = Path(to_absolute_path(str(dataset_cfg.root)))
    val_root = Path(to_absolute_path(str(getattr(cfg.validation_dataset, "root", dataset_cfg.root))))
    test_root = Path(to_absolute_path(str(getattr(cfg.test_dataset, "root", dataset_cfg.root))))

    train_dataset = SeqRecTrainDataset(train_root, max_len=max_len)
    val_dataset = SeqRecEvalDataset(val_root, split="val", max_len=max_len)
    test_dataset = SeqRecEvalDataset(test_root, split="test", max_len=max_len)

    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    if not torch.cuda.is_available():
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_left_pad, max_len=max_len),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_left_pad, max_len=max_len),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_left_pad, max_len=max_len),
    )

    seen_items: Dict[str, Dict[int, torch.Tensor]] = {
        "val": {user: val_dataset.history_tensor(user) for user in range(val_dataset.num_users)},
        "test": {
            user: test_dataset.history_tensor(user) for user in range(test_dataset.num_users)
        },
    }
    return train_loader, val_loader, test_loader, seen_items


def build_model(cfg: DictConfig, num_items: int, max_len: int) -> Dlinoss4Rec:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):  # pragma: no cover - config guard
        raise TypeError("Model configuration must resolve to a mapping")
    cfg_dict = dict(model_cfg)
    cfg_dict.pop("model_name", None)
    d_model = int(cfg_dict.pop("d_model"))
    ssm_size = int(cfg_dict.pop("ssm_size"))
    blocks = int(cfg_dict.pop("blocks"))
    dropout = float(cfg_dict.pop("dropout"))
    use_pffn = bool(cfg_dict.pop("use_pffn", True))
    use_pos_emb = bool(cfg_dict.pop("use_pos_emb", False))
    use_layernorm = bool(cfg_dict.pop("use_layernorm", True))

    head_cfg = OmegaConf.to_container(cfg.head, resolve=True)
    if isinstance(head_cfg, dict):
        head_bias = bool(head_cfg.get("bias", True))
        head_temperature = float(head_cfg.get("temperature", 1.0))
    else:  # pragma: no cover - fallback for missing head config
        head_bias = True
        head_temperature = 1.0

    model = Dlinoss4Rec(
        num_items=num_items,
        d_model=d_model,
        ssm_size=ssm_size,
        blocks=blocks,
        dropout=dropout,
        max_len=max_len,
        use_pffn=use_pffn,
        use_pos_emb=use_pos_emb,
        use_layernorm=use_layernorm,
        head_bias=head_bias,
        head_temperature=head_temperature,
        **cfg_dict,
    )
    return model


@torch.no_grad()
def evaluate_fullsort(
    model: Dlinoss4Rec,
    loader: DataLoader,
    seen_items: Dict[int, torch.Tensor],
    device: torch.device,
    *,
    topk: int,
) -> Dict[str, float]:
    model.eval()
    accumulator = TopKMetricAccumulator(topk=topk)
    for batch in loader:
        batch = batch.to(device)
        scores = model.predict_scores(batch, include_padding=False)
        mask_history_inplace(scores, batch.user_ids, seen_items, offset=1)
        targets = (batch.target - 1).to(device=device)
        accumulator.update(scores, targets)
    return accumulator.compute()


def _format_split_metrics(split: str, metrics: Dict[str, float]) -> str:
    parts = [f"split={split}"]
    for name, value in metrics.items():
        parts.append(f"{name}={value:.4f}")
    return " • ".join(parts)


def probe_efficiency(train_epoch_seconds: float, infer_seconds: float, peak_gb: float) -> Dict[str, float]:
    return {
        "train_s": float(train_epoch_seconds),
        "infer_s": float(infer_seconds),
        "gpu_gb": float(peak_gb),
    }


def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device_str = cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str) == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    seed = int(cfg.get("seed", 0))
    _set_seed(seed)

    train_loader, val_loader, test_loader, seen_items = build_dataloaders(cfg)
    train_dataset: SeqRecTrainDataset = train_loader.dataset  # type: ignore[assignment]
    num_items = train_dataset.num_items
    max_len = int(cfg.dataset.max_len)
    model = build_model(cfg, num_items=num_items, max_len=max_len).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    grad_clip = cfg.training.get("grad_clip")
    grad_clip_value = float(grad_clip) if grad_clip is not None else None
    scaler = _make_grad_scaler(device.type, amp_enabled)

    run_dir = Path(to_absolute_path(str(cfg.training.save_dir)))
    dataset_name = str(cfg.dataset.name)
    run_id = cfg.training.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    output_dir = run_dir / dataset_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        batches_per_epoch = len(train_loader)
    except TypeError as err:  # pragma: no cover - defensive guard
        raise ValueError("SeqRec training requires a sized DataLoader") from err
    if batches_per_epoch <= 0:
        raise ValueError("Training dataloader is empty; cannot proceed")

    epochs = int(cfg.training.epochs)
    if epochs <= 0:
        raise ValueError("training.epochs must be positive for seqrec tasks")
    total_steps = epochs * batches_per_epoch
    log_interval = int(cfg.training.get("log_interval", batches_per_epoch))
    if log_interval <= 0:
        log_interval = batches_per_epoch

    device_label = device.type if device.index is None else f"{device.type}:{device.index}"
    print(
        "Training • "
        f"task=seqrec "
        f"steps={total_steps:,} "
        f"device={device_label} "
        f"batch_size={cfg.training.batch_size} "
        f"batches/epoch={batches_per_epoch} "
        f"train_samples={len(train_dataset)}"
    )

    progress = ProgressReporter(total_steps)
    last_eval: Dict[str, Dict[str, float]] = {}
    best_ndcg = float("-inf")
    best_epoch = -1
    best_train_time = 0.0
    best_peak_gb = 0.0
    global_step = 0
    topk = int(cfg.training.topk)

    for epoch in range(epochs):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_examples = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device)
            with _autocast_context(device.type, amp_enabled):
                loss = model.forward_loss(batch)
            scaler.scale(loss).backward()
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            batch_size = int(batch.target.size(0))
            epoch_loss += loss_value * batch_size
            epoch_examples += batch_size

            global_step += 1
            progress.update(batch_size)

            if global_step % log_interval == 0 or global_step == total_steps:
                metrics: Dict[str, float] = {}
                if epoch_examples:
                    metrics["Loss(epoch)"] = epoch_loss / epoch_examples
                for split_name, metric_dict in last_eval.items():
                    for metric_name, metric_value in metric_dict.items():
                        metrics[f"{metric_name}({split_name})"] = metric_value
                lr = optimizer.param_groups[0]["lr"]
                progress.log(global_step, loss_value, metrics=metrics or None, lr=lr)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_duration = time.perf_counter() - epoch_start
        val_start = time.perf_counter()
        val_metrics = evaluate_fullsort(
            model,
            val_loader,
            seen_items.get("val", {}),
            device,
            topk=topk,
        )
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        val_time = time.perf_counter() - val_start
        last_eval["val"] = val_metrics
        print(
            f"Eval step {global_step:05d} • {_format_split_metrics('val', val_metrics)} "
            f"• time={format_duration(val_time)}"
        )
        current_ndcg = val_metrics[f"NDCG@{topk}"]
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_train_time = epoch_duration
            if device.type == "cuda" and torch.cuda.is_available():
                best_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            else:
                best_peak_gb = 0.0
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(
                f"Checkpoint • epoch={epoch + 1:03d} "
                f"metric=NDCG@{topk} value={current_ndcg:.4f}"
            )

    progress.summary()

    if best_epoch >= 0:
        state = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(state)
        print(f"Checkpoint • loaded_epoch={best_epoch + 1:03d}")

    test_start = time.perf_counter()
    test_metrics = evaluate_fullsort(
        model,
        test_loader,
        seen_items.get("test", {}),
        device,
        topk=topk,
    )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    test_time = time.perf_counter() - test_start
    print(
        f"Eval final • {_format_split_metrics('test', test_metrics)} "
        f"• time={format_duration(test_time)}"
    )

    efficiency = probe_efficiency(best_train_time, test_time, best_peak_gb)
    summary = {
        "dataset": dataset_name,
        "config": cfg.training.get("config_name", "default"),
        **test_metrics,
        **efficiency,
        "seed": seed,
        "run_id": run_id,
    }

    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")

    summary_parts = ["Final eval"]
    for name, value in test_metrics.items():
        summary_parts.append(f"{name}={value:.4f}")
    summary_parts.append(f"train_s={efficiency['train_s']:.2f}")
    summary_parts.append(f"infer_s={efficiency['infer_s']:.2f}")
    summary_parts.append(f"gpu_gb={efficiency['gpu_gb']:.2f}")
    print(" • ".join(summary_parts))
    print(f"Summary • path={summary_path}")
