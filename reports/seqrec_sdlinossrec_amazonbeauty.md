# Sdlinoss4Rec on Amazon Beauty (min interactions ≥ 20)

This note records a four-epoch CPU run of the Selective D-LinOSS sequential recommender
on the Amazon Beauty subset prepared with a higher interaction threshold to keep the
runtime manageable in this environment.

## Dataset preparation

```
python scripts/prepare_amazon.py \
  --subset beauty \
  --raw data/raw/amazon \
  --out data/seqrec/amazonbeauty \
  --min-interactions 20
```

The filtered split contains 163 users, 159 items, and 5,593 training interactions.

## Training command

```
python train.py \
  --task seqrec \
  --model sdlinossrec \
  --head tiedsoftmax \
  --dataset-name amazonbeauty \
  --dataset-root data \
  --epochs 4 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --num-workers 0 \
  --device cpu
```

## Metrics

Validation checkpoints improved through epoch 3 and the final test evaluation yielded:

- HR@10: 0.1166
- NDCG@10: 0.0464
- MRR@10: 0.0261
- Training wall time: ≈2.5 minutes

Hydra wrote the run artefacts under `outputs/seqrec/amazonbeauty/20251029-091955/`.
