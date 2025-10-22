# Mamba4Rec CPU Experiment

This document captures a CPU-only run of the [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec) example using the RecBole training script.

## Environment preparation

* Python 3.12.10 with `torch==2.2.1+cpu`, `recbole==1.2.0`, `numpy==1.26.4`, and `psutil==7.1.1` installed globally.
* Added a lightweight CPU-only placeholder implementation of `mamba_ssm.Mamba` inside the cloned repository so the code can execute without CUDA or custom kernels.
* Modified `run.py` in the cloned repo to accept extra configuration files through the `MAMBA4REC_CONFIGS` environment variable.

## Configuration

Training used the custom configuration stored at [`reports/mamba4rec_cpu_config.yaml`](mamba4rec_cpu_config.yaml), which targets the MovieLens-100K dataset with a single 32-dimensional Mamba layer, batch size 256, and three epochs. 【F:reports/mamba4rec_cpu_config.yaml†L1-L39】

## Command

```bash
cd /workspace/Mamba4Rec
MAMBA4REC_CONFIGS=config_cpu.yaml python run.py 2>&1 | tee cpu_run.log
```

The resulting log is archived at [`reports/mamba4rec_cpu_run.txt`](mamba4rec_cpu_run.txt).

## Results

Best validation metrics (epoch 1) and the final test set scores were:

* Validation: Hit@5 0.0657, Hit@10 0.1188, NDCG@5 0.0415, NDCG@10 0.0582, MRR@5 0.0336, MRR@10 0.0403.【F:reports/mamba4rec_cpu_run.txt†L168-L173】
* Test: Hit@5 0.0520, Hit@10 0.0954, NDCG@5 0.0335, NDCG@10 0.0475, MRR@5 0.0275, MRR@10 0.0332.【F:reports/mamba4rec_cpu_run.txt†L192-L193】

Full output excerpt (including FLOPs estimate and resource usage) is available in the log file. 【F:reports/mamba4rec_cpu_run.txt†L158-L193】

## Notes

* The placeholder `Mamba` block is a simple depthwise convolution + feed-forward residual module. It enables functional testing on CPU but does not reproduce the paper’s selective state space model. 【F:reports/mamba4rec_cpu_run.txt†L150-L158】
* GPU usage is zero and the memory footprint stays under 1 GB throughout training, as shown at the end of the log. 【F:reports/mamba4rec_cpu_run.txt†L182-L193】
