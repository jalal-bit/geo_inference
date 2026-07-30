#!/usr/bin/env python3
"""Build a combined report from standalone Harvey recovery runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from recover_unclosed_harvey_predictions import (
    _json_array_or_none,
    compute_harvey_metrics,
    recover_prediction,
)


OVERALL_FIELDS = [
    "model",
    "run_dir",
    "gold_fingerprint",
    "rows",
    "gold_items",
    "recovery_raw_column",
    "source_valid_rows",
    "source_json_valid_rate",
    "source_valid_nonempty_rows",
    "source_valid_empty_rows",
    "malformed_rows",
    "recovered_nonempty_malformed_rows",
    "malformed_unrecoverable_rows",
    "recovered_output_valid_rows",
    "recovered_output_json_valid_rate",
    "recovered_output_nonempty_rows",
    "recovered_output_nonempty_rate",
    "recovered_output_empty_rows",
    "duplicates_dropped_exact",
    "missing_list_closer_values_recovered",
    "original_tp",
    "original_fp",
    "original_fn",
    "original_precision",
    "original_recall",
    "original_f1",
    "recovered_tp",
    "recovered_fp",
    "recovered_fn",
    "recovered_precision",
    "recovered_recall",
    "recovered_f1",
    "f1_delta",
    "no_dedup_recovered_f1",
    "normalized_dedup_recovered_f1",
]


def _load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_run(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--run must use LABEL=RUN_DIRECTORY, for example "
            "Gemma-2B=harvey_results/gemma-2-2b_run"
        )
    return label.strip(), Path(path.strip())


def _gold_fingerprint(rows: Sequence[Dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row_number, row in enumerate(rows):
        digest.update(str(row.get("idx", row_number)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("gold_raw", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _assert_metric_parity(
    stored: Dict[str, Any],
    recomputed: Dict[str, Any],
    model: str,
) -> None:
    keys = (
        "locdesc_precision",
        "locdesc_recall",
        "locdesc_f1",
        "location_count_mae",
        "location_count_bias",
        "location_count_exact_match",
        "pred_location_avg",
        "gold_location_avg",
    )
    for key in keys:
        if not math.isclose(
            float(stored[key]),
            float(recomputed[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{model}: recomputed original {key}={recomputed[key]} "
                f"does not match stored value {stored[key]}"
            )


def _collect_run(model: str, run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    predictions_path = run_dir / "all_predictions.csv"
    original_metrics_path = run_dir / "test_metrics.json"
    recovery_dir = run_dir / "recovered_unclosed"
    recovered_predictions_path = recovery_dir / "recovered_predictions.csv"
    recovered_metrics_path = recovery_dir / "recovered_metrics.json"
    category_path = recovery_dir / "recovered_category_metrics.csv"
    recovery_summary_path = recovery_dir / "recovery_summary.json"
    sensitivity_path = recovery_dir / "recovery_sensitivity.json"

    required = [
        predictions_path,
        original_metrics_path,
        recovered_predictions_path,
        recovered_metrics_path,
        category_path,
        recovery_summary_path,
        sensitivity_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{model}: required report inputs are missing:\n  " + "\n  ".join(missing)
        )

    rows, fields = _load_csv(predictions_path)
    recovered_rows, recovered_fields = _load_csv(recovered_predictions_path)
    if len(rows) != len(recovered_rows):
        raise ValueError(
            f"{model}: source/recovered row mismatch "
            f"{len(rows)} != {len(recovered_rows)}"
        )
    if "pred_postproc_raw" not in fields or "gold_raw" not in fields:
        raise ValueError(f"{model}: source predictions lack required columns")
    if "custom_pred_json" not in recovered_fields:
        raise ValueError(f"{model}: recovered predictions lack custom_pred_json")

    original_metrics = _load_json(original_metrics_path)
    recovered_metrics = _load_json(recovered_metrics_path)
    recovery_summary = _load_json(recovery_summary_path)
    sensitivity = _load_json(sensitivity_path)

    original_predictions: List[List[Dict[str, Any]]] = []
    golds: List[List[Dict[str, Any]]] = []
    source_valid_nonempty_rows = 0
    for row_number, row in enumerate(rows):
        prediction = _json_array_or_none(row.get("pred_postproc_raw", ""))
        gold = _json_array_or_none(row.get("gold_raw", ""))
        if gold is None:
            raise ValueError(f"{model}: invalid gold JSON at row {row_number}")
        if prediction:
            source_valid_nonempty_rows += 1
        original_predictions.append(prediction if prediction is not None else [])
        golds.append(gold)

    recomputed_metrics, _, original_counts = compute_harvey_metrics(
        original_predictions,
        golds,
        evidence_match_threshold=float(recovery_summary["evidence_match_threshold"]),
    )
    _assert_metric_parity(original_metrics, recomputed_metrics, model)

    recovered_output_valid_rows = 0
    recovered_output_nonempty_rows = 0
    for row_number, row in enumerate(recovered_rows):
        prediction = _json_array_or_none(row.get("custom_pred_json", ""))
        if prediction is None:
            continue
        recovered_output_valid_rows += 1
        if prediction:
            recovered_output_nonempty_rows += 1

    n_rows = len(rows)
    source_valid_rows = int(recovery_summary["source_valid_rows"])
    malformed_rows = n_rows - source_valid_rows
    source_valid_empty_rows = source_valid_rows - source_valid_nonempty_rows
    recovered_nonempty_malformed_rows = int(
        recovery_summary["field_recovered_nonempty_rows"]
    )
    malformed_unrecoverable_rows = int(
        recovery_summary.get("mode_counts", {}).get("no_recoverable_value", 0)
    )
    recovered_counts = recovery_summary["overall_counts"]

    record: Dict[str, Any] = {
        "model": model,
        "run_dir": str(run_dir),
        "gold_fingerprint": _gold_fingerprint(rows),
        "rows": n_rows,
        "gold_items": int(sum(len(gold) for gold in golds)),
        "recovery_raw_column": recovery_summary["raw_column"],
        "source_valid_rows": source_valid_rows,
        "source_json_valid_rate": float(recovery_summary["source_json_valid_rate"]),
        "source_valid_nonempty_rows": source_valid_nonempty_rows,
        "source_valid_empty_rows": source_valid_empty_rows,
        "malformed_rows": malformed_rows,
        "recovered_nonempty_malformed_rows": recovered_nonempty_malformed_rows,
        "malformed_unrecoverable_rows": malformed_unrecoverable_rows,
        "recovered_output_valid_rows": recovered_output_valid_rows,
        "recovered_output_json_valid_rate": (
            recovered_output_valid_rows / n_rows if n_rows else 0.0
        ),
        "recovered_output_nonempty_rows": recovered_output_nonempty_rows,
        "recovered_output_nonempty_rate": (
            recovered_output_nonempty_rows / n_rows if n_rows else 0.0
        ),
        "recovered_output_empty_rows": n_rows - recovered_output_nonempty_rows,
        "duplicates_dropped_exact": int(recovery_summary["duplicates_dropped"]),
        "missing_list_closer_values_recovered": int(
            recovery_summary.get("missing_list_closer_values", 0)
        ),
        "original_tp": int(original_counts["tp"]),
        "original_fp": int(original_counts["fp"]),
        "original_fn": int(original_counts["fn"]),
        "original_precision": float(original_metrics["locdesc_precision"]),
        "original_recall": float(original_metrics["locdesc_recall"]),
        "original_f1": float(original_metrics["locdesc_f1"]),
        "recovered_tp": int(recovered_counts["tp"]),
        "recovered_fp": int(recovered_counts["fp"]),
        "recovered_fn": int(recovered_counts["fn"]),
        "recovered_precision": float(recovered_metrics["locdesc_precision"]),
        "recovered_recall": float(recovered_metrics["locdesc_recall"]),
        "recovered_f1": float(recovered_metrics["locdesc_f1"]),
        "f1_delta": float(
            recovered_metrics["locdesc_f1"] - original_metrics["locdesc_f1"]
        ),
        "no_dedup_recovered_f1": float(sensitivity["none"]["locdesc_f1"]),
        "normalized_dedup_recovered_f1": float(
            sensitivity["normalized"]["locdesc_f1"]
        ),
    }

    category_rows, category_fields = _load_csv(category_path)
    expected_category_fields = [
        "category",
        "support",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    if category_fields != expected_category_fields:
        raise ValueError(
            f"{model}: unexpected category columns {category_fields}; "
            f"expected {expected_category_fields}"
        )
    categories: List[Dict[str, Any]] = []
    for row in category_rows:
        categories.append(
            {
                "model": model,
                "category": row["category"],
                "support": int(row["support"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
        )
    return record, categories


def _collect_common_prefix(
    model: str,
    run_dir: Path,
    limit: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows, _ = _load_csv(run_dir / "all_predictions.csv")
    recovered_rows, _ = _load_csv(
        run_dir / "recovered_unclosed" / "recovered_predictions.csv"
    )
    if len(rows) < limit or len(recovered_rows) < limit:
        raise ValueError(
            f"{model}: common-prefix limit {limit} exceeds available rows "
            f"({len(rows)} source, {len(recovered_rows)} recovered)"
        )
    rows = rows[:limit]
    recovered_rows = recovered_rows[:limit]
    recovery_summary = _load_json(
        run_dir / "recovered_unclosed" / "recovery_summary.json"
    )

    original_predictions: List[List[Dict[str, Any]]] = []
    recovered_predictions: List[List[Dict[str, Any]]] = []
    golds: List[List[Dict[str, Any]]] = []
    source_valid_rows = 0
    source_valid_nonempty_rows = 0
    recovered_output_valid_rows = 0
    recovered_output_nonempty_rows = 0
    recovered_nonempty_malformed_rows = 0
    malformed_unrecoverable_rows = 0
    duplicates_dropped_exact = 0
    missing_list_closer_values = 0

    for row_number, (row, recovered_row) in enumerate(
        zip(rows, recovered_rows)
    ):
        original = _json_array_or_none(row.get("pred_postproc_raw", ""))
        gold = _json_array_or_none(row.get("gold_raw", ""))
        recovered = _json_array_or_none(recovered_row.get("custom_pred_json", ""))
        if gold is None:
            raise ValueError(f"{model}: invalid gold JSON at row {row_number}")

        source_valid = original is not None
        if source_valid:
            source_valid_rows += 1
            if original:
                source_valid_nonempty_rows += 1
        if recovered is not None:
            recovered_output_valid_rows += 1
            if recovered:
                recovered_output_nonempty_rows += 1
        if not source_valid:
            if recovered:
                recovered_nonempty_malformed_rows += 1
            else:
                malformed_unrecoverable_rows += 1

        duplicates_dropped_exact += int(
            recovered_row.get("custom_duplicates_dropped", 0) or 0
        )
        missing_list_closer_values += int(
            recovered_row.get("custom_missing_list_closer_values", 0) or 0
        )
        original_predictions.append(original if original is not None else [])
        recovered_predictions.append(recovered if recovered is not None else [])
        golds.append(gold)

    threshold = float(recovery_summary["evidence_match_threshold"])
    original_metrics, _, original_counts = compute_harvey_metrics(
        original_predictions,
        golds,
        evidence_match_threshold=threshold,
    )
    recovered_metrics, recovered_categories, recovered_counts = (
        compute_harvey_metrics(
            recovered_predictions,
            golds,
            evidence_match_threshold=threshold,
        )
    )

    sensitivity_f1: Dict[str, float] = {"exact": recovered_metrics["locdesc_f1"]}
    for policy in ("none", "normalized"):
        policy_predictions: List[List[Dict[str, Any]]] = []
        for row in rows:
            prediction, _ = recover_prediction(
                pred_raw=row.get(recovery_summary["raw_column"], ""),
                pred_postproc_raw=row.get(
                    recovery_summary["postprocessed_column"],
                    "",
                ),
                dedupe_policy=policy,
            )
            policy_predictions.append(prediction)
        policy_metrics, _, _ = compute_harvey_metrics(
            policy_predictions,
            golds,
            evidence_match_threshold=threshold,
        )
        sensitivity_f1[policy] = policy_metrics["locdesc_f1"]

    source_valid_empty_rows = source_valid_rows - source_valid_nonempty_rows
    record: Dict[str, Any] = {
        "model": model,
        "run_dir": str(run_dir),
        "gold_fingerprint": _gold_fingerprint(rows),
        "rows": limit,
        "gold_items": int(sum(len(gold) for gold in golds)),
        "recovery_raw_column": recovery_summary["raw_column"],
        "source_valid_rows": source_valid_rows,
        "source_json_valid_rate": source_valid_rows / limit,
        "source_valid_nonempty_rows": source_valid_nonempty_rows,
        "source_valid_empty_rows": source_valid_empty_rows,
        "malformed_rows": limit - source_valid_rows,
        "recovered_nonempty_malformed_rows": recovered_nonempty_malformed_rows,
        "malformed_unrecoverable_rows": malformed_unrecoverable_rows,
        "recovered_output_valid_rows": recovered_output_valid_rows,
        "recovered_output_json_valid_rate": recovered_output_valid_rows / limit,
        "recovered_output_nonempty_rows": recovered_output_nonempty_rows,
        "recovered_output_nonempty_rate": recovered_output_nonempty_rows / limit,
        "recovered_output_empty_rows": limit - recovered_output_nonempty_rows,
        "duplicates_dropped_exact": duplicates_dropped_exact,
        "missing_list_closer_values_recovered": missing_list_closer_values,
        "original_tp": int(original_counts["tp"]),
        "original_fp": int(original_counts["fp"]),
        "original_fn": int(original_counts["fn"]),
        "original_precision": float(original_metrics["locdesc_precision"]),
        "original_recall": float(original_metrics["locdesc_recall"]),
        "original_f1": float(original_metrics["locdesc_f1"]),
        "recovered_tp": int(recovered_counts["tp"]),
        "recovered_fp": int(recovered_counts["fp"]),
        "recovered_fn": int(recovered_counts["fn"]),
        "recovered_precision": float(recovered_metrics["locdesc_precision"]),
        "recovered_recall": float(recovered_metrics["locdesc_recall"]),
        "recovered_f1": float(recovered_metrics["locdesc_f1"]),
        "f1_delta": float(
            recovered_metrics["locdesc_f1"] - original_metrics["locdesc_f1"]
        ),
        "no_dedup_recovered_f1": float(sensitivity_f1["none"]),
        "normalized_dedup_recovered_f1": float(
            sensitivity_f1["normalized"]
        ),
    }
    categories = [
        {
            "model": model,
            "category": row["category"],
            "support": int(row["support"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }
        for row in recovered_categories
    ]
    return record, categories


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _decimal(value: Any) -> str:
    return f"{float(value):.6f}"


def _markdown_report(
    records: Sequence[Dict[str, Any]],
    categories: Sequence[Dict[str, Any]],
    common_records: Sequence[Dict[str, Any]],
    common_categories: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# Harvey exact-deduplication recovery report",
        "",
        "Policy: preserve valid postprocessed arrays; for malformed rows, recover "
        "complete location evidence values from the generated array region, then "
        "remove exact repeated evidence lists within each row. A constrained "
        "fallback also accepts complete scalar evidence items when only the "
        "evidence-list closer is missing immediately before an object boundary. "
        "Matching uses the Harvey evaluator's global greedy RapidFuzz ratio at "
        "threshold 0.75.",
        "",
        "## Overall results",
        "",
        "Native-run results are shown here for provenance and before/after "
        "validity reporting.",
        "",
        "| Model | Rows | Gold | JSON valid before | JSON valid after | "
        "Nonempty after | Duplicates removed | Original P/R/F1 | "
        "Recovered P/R/F1 | F1 change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['model']} | {record['rows']} | {record['gold_items']} | "
            f"{_percent(record['source_json_valid_rate'])} "
            f"({record['source_valid_rows']}) | "
            f"{_percent(record['recovered_output_json_valid_rate'])} "
            f"({record['recovered_output_valid_rows']}) | "
            f"{_percent(record['recovered_output_nonempty_rate'])} "
            f"({record['recovered_output_nonempty_rows']}) | "
            f"{record['duplicates_dropped_exact']} | "
            f"{_decimal(record['original_precision'])} / "
            f"{_decimal(record['original_recall'])} / "
            f"{_decimal(record['original_f1'])} | "
            f"{_decimal(record['recovered_precision'])} / "
            f"{_decimal(record['recovered_recall'])} / "
            f"**{_decimal(record['recovered_f1'])}** | "
            f"{record['f1_delta']:+.6f} |"
        )

    if common_records:
        common_rows = common_records[0]["rows"]
        lines.extend(
            [
                "",
                f"## Common {common_rows}-row comparison",
                "",
                "This is the primary like-for-like comparison: every model is "
                f"rescored on the same first {common_rows} prompts and gold rows.",
                "",
                "| Model | JSON valid before | Nonempty after | Duplicates removed | "
                "Original P/R/F1 | Recovered P/R/F1 | F1 change |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in common_records:
            lines.append(
                f"| {record['model']} | "
                f"{_percent(record['source_json_valid_rate'])} "
                f"({record['source_valid_rows']}) | "
                f"{_percent(record['recovered_output_nonempty_rate'])} "
                f"({record['recovered_output_nonempty_rows']}) | "
                f"{record['duplicates_dropped_exact']} | "
                f"{_decimal(record['original_precision'])} / "
                f"{_decimal(record['original_recall'])} / "
                f"{_decimal(record['original_f1'])} | "
                f"{_decimal(record['recovered_precision'])} / "
                f"{_decimal(record['recovered_recall'])} / "
                f"**{_decimal(record['recovered_f1'])}** | "
                f"{record['f1_delta']:+.6f} |"
            )

    lines.extend(
        [
            "",
            "## Validity and recovery detail",
            "",
            "| Model | Valid nonempty before | Valid empty before | Malformed | "
            "Malformed recovered nonempty | Missing-list-close fields salvaged | "
            "Malformed unrecoverable | Empty after |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['model']} | {record['source_valid_nonempty_rows']} | "
            f"{record['source_valid_empty_rows']} | {record['malformed_rows']} | "
            f"{record['recovered_nonempty_malformed_rows']} | "
            f"{record['missing_list_closer_values_recovered']} | "
            f"{record['malformed_unrecoverable_rows']} | "
            f"{record['recovered_output_empty_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Overall TP/FP/FN",
            "",
            "| Model | Original TP/FP/FN | Recovered TP/FP/FN |",
            "|---|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['model']} | {record['original_tp']} / "
            f"{record['original_fp']} / {record['original_fn']} | "
            f"{record['recovered_tp']} / {record['recovered_fp']} / "
            f"{record['recovered_fn']} |"
        )

    lines.extend(
        [
            "",
            "## Deduplication sensitivity",
            "",
            "| Model | No dedup F1 | Exact dedup F1 | Normalized dedup F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['model']} | "
            f"{_decimal(record['no_dedup_recovered_f1'])} | "
            f"**{_decimal(record['recovered_f1'])}** | "
            f"{_decimal(record['normalized_dedup_recovered_f1'])} |"
        )

    display_records = common_records if common_records else records
    display_categories = common_categories if common_categories else categories
    category_names = sorted(
        {row["category"] for row in display_categories},
        key=lambda value: (
            0,
            int(value[1:]),
        )
        if value.startswith("C") and value[1:].isdigit()
        else (1, value),
    )
    category_lookup = {
        (row["model"], row["category"]): row["f1"] for row in display_categories
    }
    category_title = (
        f"Common {display_records[0]['rows']}-row recovered gold-category F1"
        if common_records
        else "Recovered gold-category F1"
    )
    lines.extend(
        [
            "",
            f"## {category_title}",
            "",
            "| Category | "
            + " | ".join(record["model"] for record in display_records)
            + " |",
            "|---|" + "---:|" * len(display_records),
        ]
    )
    for category in category_names:
        values = [
            _decimal(category_lookup.get((record["model"], category), 0.0))
            for record in display_records
        ]
        lines.append(f"| {category} | " + " | ".join(values) + " |")

    fingerprints: Dict[str, List[str]] = {}
    for record in records:
        fingerprints.setdefault(record["gold_fingerprint"], []).append(record["model"])
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- JSON-valid-after is 100% because every custom prediction is serialized "
            "as an array; an unrecoverable row becomes `[]`. Use the nonempty-after "
            "column to see remaining coverage loss.",
            "- Exact deduplication is a recovery policy, not the untouched raw-model "
            "score. It removes only identical recovered evidence lists within a row; "
            "already-valid postprocessed arrays are preserved.",
            "- True `pred_raw` is available for Gemma-2-27B. In the four older "
            "artifacts it is blank, so invalid `pred_postproc_raw` is the recovery "
            "source; valid postprocessed arrays are still preserved.",
            "- Category metrics reproduce the evaluator's gold-category attribution. "
            "Unmatched predictions are overall false positives but are not assigned "
            "to category-level false positives.",
            "- The common-prefix table equalizes rows, gold labels, recovery, and "
            "scoring. It does not equalize generation protocol: Gemma-2-27B came "
            "from the newer 4-bit evaluator, while the four smaller-model artifacts "
            "came from the older non-4-bit evaluator.",
            "- Test-set fingerprints:",
        ]
    )
    for fingerprint, models in fingerprints.items():
        lines.append(f"  - `{fingerprint}`: {', '.join(models)}")
    if len(fingerprints) > 1:
        lines.append(
            "- Native runs with different fingerprints, row counts, or gold totals "
            "are not strictly comparable as a model ranking."
        )
    if common_records:
        common_fingerprints = {
            record["gold_fingerprint"] for record in common_records
        }
        if len(common_fingerprints) == 1:
            lines.append(
                "- The common-prefix table is directly comparable: all five models "
                f"use the same `{next(iter(common_fingerprints))}` fingerprint."
            )
        else:
            lines.append(
                "- Warning: the requested common-prefix rows do not share one gold "
                "fingerprint across models."
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, str]:
    if args.common_rows is not None and args.common_rows <= 0:
        raise ValueError("--common_rows must be a positive integer")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output_dir / "five_model_recovery_summary.csv",
        "summary_json": output_dir / "five_model_recovery_summary.json",
        "categories_csv": output_dir / "five_model_recovered_categories.csv",
        "report_md": output_dir / "REPORT.md",
    }
    if args.common_rows is not None:
        paths.update(
            {
                "common_summary_csv": (
                    output_dir
                    / f"five_model_common_{args.common_rows}_recovery_summary.csv"
                ),
                "common_categories_csv": (
                    output_dir
                    / f"five_model_common_{args.common_rows}_categories.csv"
                ),
            }
        )
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite report outputs; use --overwrite:\n  "
            + "\n  ".join(str(path) for path in existing)
        )

    records: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = []
    common_records: List[Dict[str, Any]] = []
    common_categories: List[Dict[str, Any]] = []
    seen_labels = set()
    for model, run_dir in args.run:
        if model in seen_labels:
            raise ValueError(f"Duplicate model label: {model}")
        seen_labels.add(model)
        record, model_categories = _collect_run(model, run_dir)
        records.append(record)
        categories.extend(model_categories)
        if args.common_rows is not None:
            common_record, model_common_categories = _collect_common_prefix(
                model,
                run_dir,
                args.common_rows,
            )
            common_records.append(common_record)
            common_categories.extend(model_common_categories)

    _write_csv(paths["summary_csv"], OVERALL_FIELDS, records)
    _write_json(
        paths["summary_json"],
        {
            "policy": {
                "dedupe": "exact",
                "evidence_match_threshold": 0.75,
                "valid_arrays_preserved": True,
                "missing_list_closer_fallback": (
                    "complete scalar evidence items followed immediately by "
                    "an object boundary"
                ),
                "unrecoverable_serialization": [],
            },
            "models": records,
            "categories": categories,
            "common_prefix": {
                "rows": args.common_rows,
                "models": common_records,
                "categories": common_categories,
            }
            if args.common_rows is not None
            else None,
        },
    )
    _write_csv(
        paths["categories_csv"],
        [
            "model",
            "category",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
        categories,
    )
    if args.common_rows is not None:
        _write_csv(paths["common_summary_csv"], OVERALL_FIELDS, common_records)
        _write_csv(
            paths["common_categories_csv"],
            [
                "model",
                "category",
                "support",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ],
            common_categories,
        )
    paths["report_md"].write_text(
        _markdown_report(
            records,
            categories,
            common_records,
            common_categories,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine completed Harvey recovery runs into CSV/JSON/Markdown."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="Repeat LABEL=RUN_DIRECTORY once per model.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--common_rows",
        type=int,
        default=None,
        help="Also report a like-for-like common prefix of this many rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print("Saved combined recovery report:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
