"""Progress reporting utilities shared across OSSM training loops."""

from __future__ import annotations

import time
from typing import Dict, Optional

__all__ = ["format_duration", "ProgressReporter"]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS``."""

    total = int(max(float(seconds), 0.0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Lightweight console reporter matching OSSM's training telemetry style."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 0)
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.interval_examples = 0
        self.interval_steps = 0

    def update(self, batch_size: int) -> None:
        """Record that a batch with ``batch_size`` examples has completed."""

        self.interval_examples += int(batch_size)
        self.interval_steps += 1

    def log(
        self,
        step: int,
        loss: float,
        *,
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Emit a formatted progress line for the given ``step``."""

        now = time.perf_counter()
        interval = max(now - self.last_log_time, 1e-9)
        elapsed = now - self.start_time
        remaining_steps = max(self.total_steps - int(step), 0)
        avg_step_time = elapsed / max(int(step), 1)
        eta = avg_step_time * remaining_steps
        throughput = (
            self.interval_examples / interval if self.interval_examples else 0.0
        )
        step_time = interval / max(self.interval_steps, 1)

        parts = [f"Step {int(step):05d}/{self.total_steps:05d}", f"Loss = {loss:.4f}"]
        if metrics:
            for name, value in metrics.items():
                parts.append(f"{name} = {value:.4f}")
        if lr is not None:
            parts.append(f"LR = {lr:.2e}")
        parts.append(f"Samples/s = {throughput:,.1f}")
        parts.append(f"Step time = {step_time * 1e3:.1f} ms")
        parts.append(f"Time = {format_duration(elapsed)}")
        parts.append(f"ETA = {format_duration(eta)}")
        print(" • ".join(parts))

        self.last_log_time = now
        self.interval_examples = 0
        self.interval_steps = 0

    def summary(self) -> None:
        """Emit a completion line using the canonical OSSM format."""

        elapsed = time.perf_counter() - self.start_time
        print(f"Training finished • Time = {format_duration(elapsed)}")
