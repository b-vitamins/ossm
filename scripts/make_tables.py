"""Aggregate seqrec experiment summaries into human-readable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _load_records(paths: Sequence[str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for pattern in paths:
        for path in sorted(Path().glob(pattern)):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
    return records


def _format_table(title: str, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    columns = [list(col) for col in zip(*([headers] + [tuple(r) for r in rows]))]
    widths = [max(len(cell) for cell in column) for column in columns]
    divider = "-" * (sum(widths) + 3 * (len(headers) - 1))

    def _format_row(values: Sequence[str]) -> str:
        return "   ".join(cell.ljust(width) for cell, width in zip(values, widths))

    output = [title, divider, _format_row(headers)]
    output.append(divider)
    for row in rows:
        output.append(_format_row(row))
    output.append(divider)
    return "\n".join(output)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _performance_table(records: List[Dict[str, object]]) -> str:
    rows: List[Sequence[str]] = []
    datasets = sorted({str(r.get("dataset")) for r in records})
    for dataset in datasets:
        default = next((r for r in records if r.get("dataset") == dataset and r.get("config") == "default"), None)
        if not default:
            continue
        rows.append(
            [
                dataset,
                _format_float(float(default.get("HR@10", 0.0))),
                _format_float(float(default.get("NDCG@10", 0.0))),
                _format_float(float(default.get("MRR@10", 0.0))),
            ]
        )
    return _format_table("Table 1: Recommendation Performance", ["Dataset", "HR@10", "NDCG@10", "MRR@10"], rows)


def _efficiency_table(records: List[Dict[str, object]]) -> str:
    rows: List[Sequence[str]] = []
    for record in records:
        if record.get("dataset") != "ml1m" or record.get("config") != "default":
            continue
        rows.append(
            [
                str(record.get("config")),
                f"{float(record.get('gpu_gb', 0.0)):.2f}",
                f"{float(record.get('train_s', 0.0)):.2f}",
                f"{float(record.get('infer_s', 0.0)):.2f}",
            ]
        )
    return _format_table("Table 3: Efficiency on ML-1M", ["Config", "GPU GB", "Train s", "Infer s"], rows)


def _ablation_table(records: List[Dict[str, object]]) -> str:
    rows: List[Sequence[str]] = []
    order = ["default", "block_only", "two_layers", "w_pe", "w/o_pffn", "w/o_ln"]
    for dataset in ("ml1m", "amazonbeauty"):
        for config in order:
            record = next(
                (r for r in records if r.get("dataset") == dataset and r.get("config") == config),
                None,
            )
            if record is None:
                continue
            rows.append(
                [
                    dataset,
                    config,
                    _format_float(float(record.get("NDCG@10", 0.0))),
                    _format_float(float(record.get("MRR@10", 0.0))),
                ]
            )
    return _format_table("Table 4: Ablation Results", ["Dataset", "Config", "NDCG@10", "MRR@10"], rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seqrec summary files into tables")
    parser.add_argument("paths", nargs="+", help="Glob patterns pointing at summary.jsonl files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.paths)
    if not records:
        raise SystemExit("No summary records found")
    tables = [
        _performance_table(records),
        _efficiency_table(records),
        _ablation_table(records),
    ]
    print("\n\n".join(tables))


if __name__ == "__main__":
    main()
