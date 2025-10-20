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
