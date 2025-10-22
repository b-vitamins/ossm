"""Sequential recommendation training entrypoint."""

from __future__ import annotations

import json
import logging
import random
import textwrap
import time
from functools import partial
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Set, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf  # type: ignore[import]
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..data.datasets import SeqRecEvalDataset, SeqRecTrainDataset, collate_left_pad
from ..metrics import TopKMetricAccumulator, mask_history_inplace
from ..models.dlinossrec import Dlinoss4Rec
from ..models.mambarec import Mamba4Rec
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
    Dict[str, Any],
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

    eval_batch_size_cfg = cfg.training.get("eval_batch_size")
    if eval_batch_size_cfg is None:
        eval_batch_size = int(cfg.training.batch_size)
    else:
        eval_batch_size = int(eval_batch_size_cfg)

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
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_left_pad, max_len=max_len),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
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
    loader_info = {
        "train_shuffle": True,
        "val_shuffle": False,
        "test_shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "max_len": max_len,
    }
    return train_loader, val_loader, test_loader, seen_items, loader_info


def _collect_train_targets(dataset: SeqRecTrainDataset) -> Set[int]:
    targets: Set[int] = set()
    for user_id in range(dataset.num_users):
        sequence = dataset.user_sequence(user_id)
        if len(sequence) <= 1:
            continue
        targets.update(int(item) for item in sequence[1:])
    return targets


def _collect_eval_targets(dataset: SeqRecEvalDataset) -> Tuple[Set[int], List[int]]:
    targets: Set[int] = set()
    context_lengths: List[int] = []
    for index in range(len(dataset)):
        _user_id, context, target = dataset[index]
        targets.add(int(target))
        context_lengths.append(len(context))
    return targets, context_lengths


def _summarize_seqrec_splits(
    train_dataset: SeqRecTrainDataset,
    val_dataset: SeqRecEvalDataset,
    test_dataset: SeqRecEvalDataset,
) -> Dict[str, Any]:
    train_examples = len(train_dataset)
    train_users = train_dataset.num_users
    train_total_interactions = 0
    train_min_len = float("inf")
    train_max_len = 0
    for user_id in range(train_users):
        sequence_length = len(train_dataset.user_sequence(user_id))
        train_total_interactions += sequence_length
        train_min_len = min(train_min_len, sequence_length)
        train_max_len = max(train_max_len, sequence_length)
    if train_users:
        train_avg_len = train_total_interactions / train_users
    else:
        train_avg_len = 0.0
        train_min_len = 0.0

    val_targets, val_context_lengths = _collect_eval_targets(val_dataset)
    test_targets, test_context_lengths = _collect_eval_targets(test_dataset)

    def _stats(lengths: List[int]) -> Tuple[int, float, int]:
        if not lengths:
            return 0, 0.0, 0
        return min(lengths), sum(lengths) / len(lengths), max(lengths)

    val_min_len, val_avg_len, val_max_len = _stats(val_context_lengths)
    test_min_len, test_avg_len, test_max_len = _stats(test_context_lengths)

    train_targets = _collect_train_targets(train_dataset)
    train_val_overlap = train_targets & val_targets
    train_test_overlap = train_targets & test_targets
    val_test_overlap = val_targets & test_targets

    summary = {
        "train_users": train_users,
        "train_examples": train_examples,
        "train_interactions": train_total_interactions,
        "train_seq_len": (int(train_min_len), float(train_avg_len), int(train_max_len)),
        "val_examples": len(val_dataset),
        "val_seq_len": (val_min_len, float(val_avg_len), val_max_len),
        "test_examples": len(test_dataset),
        "test_seq_len": (test_min_len, float(test_avg_len), test_max_len),
        "train_root": train_dataset.root,
        "val_root": val_dataset.root,
        "test_root": test_dataset.root,
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
    }
    print(
        "Dataset summary • "
        f"train_users={summary['train_users']} • "
        f"train_examples={summary['train_examples']} • "
        f"train_interactions={summary['train_interactions']} • "
        f"train_seq_len(min/mean/max)="
        f"{summary['train_seq_len'][0]}/{summary['train_seq_len'][1]:.1f}/{summary['train_seq_len'][2]} • "
        f"val_examples={summary['val_examples']} • "
        f"val_seq_len(min/mean/max)="
        f"{summary['val_seq_len'][0]}/{summary['val_seq_len'][1]:.1f}/{summary['val_seq_len'][2]} • "
        f"test_examples={summary['test_examples']} • "
        f"test_seq_len(min/mean/max)="
        f"{summary['test_seq_len'][0]}/{summary['test_seq_len'][1]:.1f}/{summary['test_seq_len'][2]}"
    )
    print(
        "Dataset roots • "
        f"train={summary['train_root']} • "
        f"val={summary['val_root']} • "
        f"test={summary['test_root']}"
    )
    print(
        "Split overlap • "
        f"train∩val={summary['train_val_overlap']} • "
        f"train∩test={summary['train_test_overlap']} • "
        f"val∩test={summary['val_test_overlap']}"
    )
    if train_val_overlap:
        preview = sorted(list(train_val_overlap))[:5]
        print(f"  Example train∩val targets={preview}")
    if train_test_overlap:
        preview = sorted(list(train_test_overlap))[:5]
        print(f"  Example train∩test targets={preview}")
    if val_test_overlap:
        preview = sorted(list(val_test_overlap))[:5]
        print(f"  Example val∩test targets={preview}")
    return summary


def _warn_if_placeholder_dataset(summary: Dict[str, Any], dataset_name: str) -> None:
    """Emit guidance when the fallback smoke-test dataset is detected."""

    train_users = int(summary["train_users"])
    train_examples = int(summary["train_examples"])
    train_interactions = int(summary["train_interactions"])
    val_examples = int(summary["val_examples"])
    test_examples = int(summary["test_examples"])
    suspected_placeholder = (
        train_users <= 5
        and train_examples <= 64
        and train_interactions <= 256
        and val_examples <= 8
        and test_examples <= 8
    )
    if not suspected_placeholder:
        return

    dataset_root = Path(summary["train_root"]).resolve()
    warning = textwrap.dedent(
        f"""
        WARNING: Detected a tiny sequential dataset for '{dataset_name}' at {dataset_root}.
        The repository only ships this <=5-user split as a smoke test. For real experiments,
        download the raw MovieLens/Amazon dumps and run scripts/prepare_*.py exactly as
        documented in README.md (Sequential Recommendation section).
        Example: python scripts/prepare_amazon.py --subset beauty --raw <raw_dir> --out <data_root>/seqrec/amazonbeauty
        Afterwards, launch training with --dataset-root or OSSM_DATA_ROOT pointing to the processed directory.
        """
    ).strip()
    print(warning)
    print(
        "Hint: Once you prepare the full dataset, ensure training.topk is below the candidate "
        "pool size reported during evaluation to avoid guaranteed HR@K=1.0 runs."
    )


def build_model(cfg: DictConfig, num_items: int, max_len: int) -> Dlinoss4Rec | Mamba4Rec:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):  # pragma: no cover - config guard
        raise TypeError("Model configuration must resolve to a mapping")
    cfg_dict = dict(model_cfg)
    model_name = str(cfg_dict.pop("model_name", "dlinossrec")).lower()

    head_cfg = OmegaConf.to_container(cfg.head, resolve=True)
    if isinstance(head_cfg, dict):
        head_bias = bool(head_cfg.get("bias", True))
        head_temperature = float(head_cfg.get("temperature", 1.0))
    else:  # pragma: no cover - fallback for missing head config
        head_bias = True
        head_temperature = 1.0

    if model_name == "dlinossrec":
        dlinoss_cfg = dict(cfg_dict)
        d_model = int(dlinoss_cfg.pop("d_model"))
        ssm_size = int(dlinoss_cfg.pop("ssm_size"))
        blocks = int(dlinoss_cfg.pop("blocks"))
        dropout = float(dlinoss_cfg.pop("dropout"))
        use_pffn = bool(dlinoss_cfg.pop("use_pffn", True))
        use_pos_emb = bool(dlinoss_cfg.pop("use_pos_emb", False))
        use_layernorm = bool(dlinoss_cfg.pop("use_layernorm", True))
        return Dlinoss4Rec(
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
            **dlinoss_cfg,
        )
    if model_name == "mambarec":
        mamba_cfg = dict(cfg_dict)
        d_model = int(mamba_cfg.pop("d_model"))
        ssm_size = int(mamba_cfg.pop("ssm_size"))
        blocks = int(mamba_cfg.pop("blocks"))
        dropout = float(mamba_cfg.pop("dropout"))
        use_pffn = bool(mamba_cfg.pop("use_pffn", True))
        use_pos_emb = bool(mamba_cfg.pop("use_pos_emb", False))
        use_layernorm = bool(mamba_cfg.pop("use_layernorm", True))
        d_conv = int(mamba_cfg.pop("d_conv", 4))
        expand = int(mamba_cfg.pop("expand", 2))
        dt_rank_cfg = mamba_cfg.pop("dt_rank", "auto")
        dt_rank = dt_rank_cfg if isinstance(dt_rank_cfg, str) else int(dt_rank_cfg)
        dt_min = float(mamba_cfg.pop("dt_min", 0.001))
        dt_max = float(mamba_cfg.pop("dt_max", 0.1))
        dt_init = str(mamba_cfg.pop("dt_init", "random"))
        dt_scale = float(mamba_cfg.pop("dt_scale", 1.0))
        dt_init_floor = float(mamba_cfg.pop("dt_init_floor", 1e-4))
        conv_bias = bool(mamba_cfg.pop("conv_bias", True))
        bias = bool(mamba_cfg.pop("bias", False))
        return Mamba4Rec(
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
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            **mamba_cfg,
        )
    raise ValueError(f"Unknown sequential recommendation model '{model_name}'")


@torch.no_grad()
def evaluate_fullsort(
    model: Dlinoss4Rec | Mamba4Rec,
    loader: DataLoader,
    seen_items: Dict[int, torch.Tensor],
    device: torch.device,
    *,
    topk: int,
    split_name: str = "",
    log_predictions: bool = False,
    max_batches: int | None = None,
    verbose: bool = False,
) -> Dict[str, float]:
    model.eval()
    accumulator = TopKMetricAccumulator(topk=topk)
    batch_count = 0
    total_examples = 0
    total_candidates = 0
    min_candidates = float("inf")
    max_candidates = 0
    first_batch_debug: Dict[str, torch.Tensor] | None = None
    for batch in loader:
        batch = batch.to(device)
        scores = model.predict_scores(batch, include_padding=False)
        mask_history_inplace(scores, batch.user_ids, seen_items, offset=1)
        targets = (batch.target - 1).to(device=device)
        accumulator.update(scores, targets)
        finite_mask = torch.isfinite(scores)
        candidate_counts = finite_mask.sum(dim=1)
        total_examples += int(candidate_counts.size(0))
        total_candidates += int(candidate_counts.sum().item())
        if candidate_counts.numel():
            min_candidates = min(min_candidates, int(candidate_counts.min().item()))
            max_candidates = max(max_candidates, int(candidate_counts.max().item()))
        batch_count += 1
        if max_batches is not None and batch_count >= int(max_batches):
            break
        if log_predictions and first_batch_debug is None:
            topn = min(10, scores.size(1))
            top_scores, top_indices = torch.topk(scores, topn, dim=1)
            first_batch_debug = {
                "user_ids": batch.user_ids.detach().cpu(),
                "targets": targets.detach().cpu(),
                "top_indices": top_indices.detach().cpu(),
                "top_scores": top_scores.detach().cpu(),
                "history_len": batch.mask.sum(dim=1).to(torch.long).detach().cpu(),
                "candidate_counts": candidate_counts.detach().cpu(),
            }
    if batch_count == 0 or min_candidates == float("inf"):
        min_candidates = 0
    average_candidates = (
        float(total_candidates) / total_examples if total_examples else 0.0
    )
    try:
        expected_batches = len(loader)
    except TypeError:  # pragma: no cover - streaming loader guard
        expected_batches = None
    if verbose:
        coverage_message = "Eval coverage"
        if split_name:
            coverage_message += f" • split={split_name}"
        coverage_message += f" • batches={batch_count}"
        if expected_batches is not None:
            coverage_message += f"/{expected_batches}"
        coverage_message += (
            f" • examples={total_examples}"
            f" • candidates(min/mean/max)={min_candidates}/{average_candidates:.1f}/{max_candidates}"
        )
        print(coverage_message)
    if min_candidates < topk:
        raise RuntimeError(
            "Evaluation candidate pool is smaller than training.topk. "
            f"split={split_name or 'eval'} has min_candidates={min_candidates} < topk={topk}. "
            "The bundled smoke-test data masks almost every item; prepare the full dataset "
            "with scripts/prepare_*.py or reduce training.topk."
        )
    if verbose and log_predictions and first_batch_debug is not None:
        print(
            "Eval predictions • "
            f"split={split_name or 'eval'} • batch=0 • topk={first_batch_debug['top_indices'].size(1)}"
        )
        for row in range(first_batch_debug["top_indices"].size(0)):
            user_id = int(first_batch_debug["user_ids"][row].item())
            target = int(first_batch_debug["targets"][row].item())
            history_len = int(first_batch_debug["history_len"][row].item())
            candidates = int(first_batch_debug["candidate_counts"][row].item())
            indices = first_batch_debug["top_indices"][row].tolist()
            scores_row = first_batch_debug["top_scores"][row].tolist()
            formatted = ", ".join(
                f"{int(idx + 1)}:{score:.3f}" for idx, score in zip(indices, scores_row)
            )
            print(
                f"  user={user_id} • target={target} • history={history_len} "
                f"• candidates={candidates} • top={formatted}"
            )
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

    dataset_name = str(cfg.dataset.name)
    (
        train_loader,
        val_loader,
        test_loader,
        seen_items,
        loader_info,
    ) = build_dataloaders(cfg)
    train_dataset: SeqRecTrainDataset = train_loader.dataset  # type: ignore[assignment]
    val_dataset: SeqRecEvalDataset = val_loader.dataset  # type: ignore[assignment]
    test_dataset: SeqRecEvalDataset = test_loader.dataset  # type: ignore[assignment]
    split_summary = _summarize_seqrec_splits(train_dataset, val_dataset, test_dataset)
    _warn_if_placeholder_dataset(split_summary, dataset_name)
    print(
        "DataLoader config • "
        f"train_shuffle={loader_info['train_shuffle']} • "
        f"val_shuffle={loader_info['val_shuffle']} • "
        f"test_shuffle={loader_info['test_shuffle']} • "
        f"num_workers={loader_info['num_workers']} • "
        f"pin_memory={loader_info['pin_memory']} • "
        f"max_len={loader_info['max_len']}"
    )
    num_items = train_dataset.num_items
    max_len = int(cfg.dataset.max_len)
    model = build_model(cfg, num_items=num_items, max_len=max_len).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    print(
        "Optimizer • type=AdamW "
        f"lr={optimizer.param_groups[0]['lr']:.6e} • "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']:.3g}"
    )
    grad_clip = cfg.training.get("grad_clip")
    grad_clip_value = float(grad_clip) if grad_clip is not None else None
    scaler = _make_grad_scaler(device.type, amp_enabled)

    run_dir = Path(to_absolute_path(str(cfg.training.save_dir)))
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
    max_steps_cfg = cfg.training.get("max_steps")
    if max_steps_cfg is not None:
        max_steps_value = int(max_steps_cfg)
        if max_steps_value <= 0:
            raise ValueError("training.max_steps must be positive when provided for seqrec tasks")
        total_steps = min(total_steps, max_steps_value)
    if total_steps <= 0:
        raise ValueError("SeqRec training computed zero total steps; check epochs and max_steps")
    log_interval_cfg = cfg.training.get("log_interval")
    if log_interval_cfg is None:
        log_interval = batches_per_epoch
    else:
        log_interval = int(log_interval_cfg)
    if log_interval <= 0:
        log_interval = batches_per_epoch

    debug_flags = {
        name: cfg.training.get(name)
        for name in (
            "fast_dev_run",
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
            "steps",
            "max_steps",
        )
        if name in cfg.training
    }
    debug_flags = {key: value for key, value in debug_flags.items() if value not in (None, False, 0)}
    if debug_flags:
        print(f"Debug flags detected • {debug_flags}")
    else:
        print("Debug flags detected • none")

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

    trace_path_cfg = cfg.training.get("trace_path")
    trace_path: Path | None = None
    trace_limit = int(cfg.training.get("trace_steps", 0) or 0)
    trace_records: List[Dict[str, Any]] = []
    trace_eval_records: List[Dict[str, Any]] = []
    if trace_path_cfg not in (None, ""):
        trace_path = Path(to_absolute_path(str(trace_path_cfg)))
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    progress = ProgressReporter(total_steps)
    last_eval: Dict[str, Dict[str, float]] = {}
    best_ndcg = float("-inf")
    best_epoch = -1
    best_train_time = 0.0
    best_peak_gb = 0.0
    global_step = 0
    topk = int(cfg.training.topk)
    eval_interval_cfg = cfg.training.get("eval_interval")
    eval_interval: int | None = int(eval_interval_cfg) if eval_interval_cfg not in (None, False, 0) else None
    max_eval_batches_cfg = cfg.training.get("limit_val_batches")
    max_eval_batches: int | None = (
        int(max_eval_batches_cfg) if max_eval_batches_cfg not in (None, False, 0) else None
    )

    for epoch in range(epochs):
        if global_step >= total_steps:
            break
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_examples = 0
        model.train()
        for b_idx, batch in enumerate(train_loader, start=1):
            if global_step >= total_steps:
                break
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
            if global_step <= 5:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"LR step • step={global_step} • lr={current_lr:.6e}")

            if trace_path is not None and (trace_limit <= 0 or global_step <= trace_limit):
                trace_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "batch_size": batch_size,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if device.type == "cuda" and torch.cuda.is_available():
                    trace_entry["gpu_alloc_gb"] = float(torch.cuda.memory_allocated(device) / 1e9)
                    trace_entry["gpu_peak_gb"] = float(
                        torch.cuda.max_memory_allocated(device) / 1e9
                    )
                # Include latest eval snapshot if available
                if "val" in last_eval:
                    for k, v in last_eval["val"].items():
                        trace_entry[f"{k}(val)"] = float(v)
                trace_records.append(trace_entry)

            if (b_idx % log_interval == 0) or global_step == total_steps:
                metrics: Dict[str, float] = {}
                if epoch_examples:
                    metrics["Loss(epoch)"] = epoch_loss / epoch_examples
                # Surface last seen eval metrics inline for quick monitoring
                for split_name, metric_dict in last_eval.items():
                    for metric_name, metric_value in metric_dict.items():
                        metrics[f"{metric_name}({split_name})"] = metric_value
                # Add lightweight GPU telemetry if available
                if (
                    device.type == "cuda" and torch.cuda.is_available() and bool(cfg.training.get("log_gpu", False))
                ):
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                    mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
                    try:
                        total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9  # type: ignore[attr-defined]
                    except Exception:
                        total_mem = 0.0
                    if total_mem > 0:
                        metrics["GPU(alloc%)"] = 100.0 * (mem_alloc / total_mem)
                    metrics["GPU(alloc)GB"] = mem_alloc
                    metrics["GPU(peak)GB"] = mem_peak
                lr = optimizer.param_groups[0]["lr"]
                progress.log(
                    global_step,
                    loss_value,
                    metrics=metrics or None,
                    lr=lr,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    epoch_step=b_idx,
                    epoch_size=batches_per_epoch,
                    prefer_epoch=True,
                )

            # Optional mid-epoch evaluation based on training.eval_interval
            if eval_interval is not None and (global_step % eval_interval == 0) and global_step < total_steps:
                val_metrics = evaluate_fullsort(
                    model,
                    val_loader,
                    seen_items.get("val", {}),
                    device,
                    topk=topk,
                    split_name="val",
                    log_predictions=False,
                    max_batches=max_eval_batches,
                )
                last_eval["val"] = val_metrics
                print(
                    f"Eval epoch {epoch + 1:03d} • batch={b_idx}/{batches_per_epoch} • "
                    f"{_format_split_metrics('val', val_metrics)}"
                )

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
            split_name="val",
            log_predictions=bool(cfg.training.get("log_predictions", False)) and (epoch == 0),
            verbose=bool(cfg.training.get("eval_verbose", False)),
        )
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        _val_time = time.perf_counter() - val_start
        last_eval["val"] = val_metrics
        print(f"Eval epoch {epoch + 1:03d} • {_format_split_metrics('val', val_metrics)}")
        if trace_path is not None:
            trace_eval_records.append(
                {
                    "step": global_step,
                    "split": "val",
                    "kind": "epoch",
                    **{name: float(value) for name, value in val_metrics.items()},
                }
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
        split_name="test",
    )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    test_time = time.perf_counter() - test_start
    print(
        f"Eval final • {_format_split_metrics('test', test_metrics)} "
        f"• time={format_duration(test_time)}"
    )
    if trace_path is not None:
        trace_eval_records.append(
            {
                "step": global_step,
                "split": "test",
                "kind": "final",
                **{name: float(value) for name, value in test_metrics.items()},
            }
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

    if trace_path is not None:
        trace_payload = {"train": trace_records, "eval": trace_eval_records}
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump(trace_payload, handle, indent=2)
    print(f"Summary • path={summary_path}")
