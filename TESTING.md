# CLI smoke test log

The following commands were executed to validate the training entrypoint against the processed BasicMotions dataset from the UEA archive:

1. Prepare the dataset cache:

   ```bash
   python scripts/prepare_uea.py --datasets BasicMotions --root ~/.cache/torch/datasets/ossm
   ```

2. Train the LinOSS backbone on the raw view with manual collate overrides:

   ```bash
   python train.py \
     --model linoss_im \
     --dataset-name BasicMotions \
     --collate pad \
     --batch-size 8 \
     --max-steps 5 \
     --log-interval 1 \
     --eval-interval 2 \
     --num-workers 2 \
     --prefetch-factor 2 \
     --optimizer adamw \
     --lr 0.001 \
     --device cpu \
     dataset.view=raw \
     validation_dataset.view=raw \
     test_dataset.view=raw
   ```

3. Train the NCDE backbone on cubic-spline coefficients:

   ```bash
   python train.py \
     --model ncde \
     --dataset-name BasicMotions \
     --batch-size 8 \
     --max-steps 5 \
     --log-interval 1 \
     --eval-interval 2 \
     --num-workers 2 \
     --prefetch-factor 2 \
     --optimizer adamw \
     --lr 0.001 \
     --device cpu
   ```

4. Train the RNN baseline with gradient clipping for an ultra-short run:

   ```bash
   python train.py \
     --model rnn \
     --dataset-name BasicMotions \
     --collate pad \
     --batch-size 8 \
     --max-steps 3 \
     --log-interval 1 \
     --eval-interval 3 \
     --num-workers 0 \
     --optimizer adamw \
     --lr 0.001 \
     --device cpu \
     --grad-clip 1.0 \
     --weight-decay 0.0 \
     dataset.view=raw \
     validation_dataset.view=raw \
     test_dataset.view=raw
   ```

All runs completed without runtime errors, produced telemetry at the configured logging cadence, and saved outputs under `outputs/`.

## Sequential recommendation: 20-step Amazon Beauty trace

To sanity-check the seqrec tracer on real data, prepare the Amazon Beauty split with the repository script:

```bash
python scripts/prepare_amazon.py \
  --subset beauty \
  --raw "$OSSM_DATA_ROOT/raw/amazon" \
  --out "$OSSM_DATA_ROOT/seqrec/amazonbeauty" \
  --min-interactions 5
```

Then run a 20-step CPU burst with a manual driver that mirrors the seqrec loop but limits evaluation to the first 512 validation/test users so it completes quickly on CPU:

```bash
python - <<'PY'
import json, os
from pathlib import Path

import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, Subset

from ossm.training import seqrec as seqrec_train

root = Path('.').resolve()
data_root = Path(os.environ.get('OSSM_DATA_ROOT', root / 'data')).resolve()
work_root = Path(os.environ.get('OSSM_WORK_DIR', root / 'outputs')).resolve()

overrides = [
    "+task=seqrec",
    "dataset=amazonbeauty",
    "validation_dataset=amazonbeauty",
    "test_dataset=amazonbeauty",
    "model=dlinossrec",
    "head=tiedsoftmax",
    "training=seqrec",
    f"paths.data_root={data_root}",
    f"paths.work_dir={work_root}",
    "dataset.num_workers=0",
    "training.batch_size=128",
    "training.max_steps=20",
    "training.log_interval=1",
    "+training.eval_interval=1000",
]

with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="config", overrides=overrides)

train_loader, val_loader, test_loader, seen, _ = seqrec_train.build_dataloaders(cfg)
model = seqrec_train.build_model(cfg, num_items=train_loader.dataset.num_items, max_len=int(cfg.dataset.max_len))
model.to("cpu")
optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))

trace = []
iterator = iter(train_loader)
for step in range(1, 21):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        batch = next(iterator)
    batch = batch.to("cpu")
    optim.zero_grad()
    loss = model.forward_loss(batch)
    loss.backward()
    optim.step()
    trace.append({"step": step, "loss": float(loss)})
    if step % 5 == 0 or step == 1:
        print(f"step {step:02d} loss={float(loss):.4f}")

def subset(loader: DataLoader, limit: int) -> DataLoader:
    return DataLoader(
        Subset(loader.dataset, range(min(limit, len(loader.dataset)))),
        batch_size=loader.batch_size,
        shuffle=False,
        collate_fn=loader.collate_fn,
    )

model.eval()
with torch.no_grad():
    val_metrics = seqrec_train.evaluate_fullsort(model, subset(val_loader, 512), seen["val"], torch.device("cpu"), topk=int(cfg.training.topk), split_name="val@512")
    test_metrics = seqrec_train.evaluate_fullsort(model, subset(test_loader, 512), seen["test"], torch.device("cpu"), topk=int(cfg.training.topk), split_name="test@512")

trace_path = work_root / "traces" / "amazonbeauty_manual20.json"
trace_path.parent.mkdir(parents=True, exist_ok=True)
json.dump({"train": trace, "eval": {"val": val_metrics, "test": test_metrics}}, trace_path.open("w", encoding="utf-8"), indent=2)
print("saved trace", trace_path)
PY
```

Key observations from the resulting trace:

* Loss drops from 20.75 on step 1 to the 15–20 range by step 10 before bouncing with early-epoch noise.
* Evaluation on the 512-user validation/test slices keeps `HR@10`, `NDCG@10`, and `MRR@10` at 0.0, consistent with an untrained model facing a 12K-item candidate pool.

The full trace lives under `outputs/traces/amazonbeauty_manual20.json` (ignored by git) for deeper inspection.
