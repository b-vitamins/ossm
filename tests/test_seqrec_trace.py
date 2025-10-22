import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ossm.training import seqrec_main


def _write_toy_seqrec_split(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    user_ptr = np.array([0, 3, 6], dtype=np.int64)
    train_items = np.array([1, 2, 3, 2, 4, 5], dtype=np.int64)
    np.save(root / "user_ptr.npy", user_ptr)
    np.save(root / "train_items.npy", train_items)
    (root / "item_count.txt").write_text("16", encoding="utf-8")

    val_df = pd.DataFrame(
        {
            "user_id": [0, 1],
            "prefix_len": [2, 2],
            "target_id": [3, 5],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": [0, 1],
            "prefix_len": [2, 2],
            "target_id": [3, 5],
        }
    )
    val_df.to_parquet(root / "val.parquet", index=False)
    test_df.to_parquet(root / "test.parquet", index=False)


def _make_seqrec_config(root: Path) -> OmegaConf:
    data_cfg = {
        "name": "toy",
        "root": str(root),
        "max_len": 4,
        "num_workers": 0,
        "pin_memory": False,
        "dropout": 0.0,
    }
    training_cfg = {
        "device": "cpu",
        "epochs": 5,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "amp": False,
        "topk": 3,
        "save_dir": str(root / "runs"),
        "config_name": "test",
        "log_interval": 1,
        "max_steps": 3,
        "trace_path": str(root / "trace" / "trace.json"),
        "trace_steps": 0,
    }
    model_cfg = {
        "model_name": "dlinossrec",
        "d_model": 8,
        "ssm_size": 8,
        "blocks": 1,
        "dropout": 0.0,
        "use_pffn": True,
        "use_pos_emb": False,
        "use_layernorm": True,
    }
    head_cfg = {"bias": True, "temperature": 1.0}
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "dataset": data_cfg,
            "validation_dataset": dict(data_cfg),
            "test_dataset": dict(data_cfg),
            "model": model_cfg,
            "head": head_cfg,
            "training": training_cfg,
        }
    )
    return cfg


def test_seqrec_traces_capture_steps_and_evals(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    data_root = tmp_path / "seqrec"
    _write_toy_seqrec_split(data_root)
    cfg = _make_seqrec_config(data_root)
    seqrec_main(cfg)

    trace_path = Path(cfg.training.trace_path)
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    train_records = payload["train"]
    eval_records = payload["eval"]

    assert len(train_records) == cfg.training.max_steps
    assert train_records[-1]["step"] == cfg.training.max_steps
    assert all("loss" in record for record in train_records)

    assert any(record["split"] == "val" for record in eval_records)
    assert any(record["split"] == "test" and record["kind"] == "final" for record in eval_records)
