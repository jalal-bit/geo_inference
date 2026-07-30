#!/usr/bin/env python3
"""
Recover and rescore malformed Harvey location-description predictions.

This is a standalone post-processing tool. It does not load a language model and
does not modify either evaluation script.

Recovery policy
---------------
1. Preserve ``pred_postproc_raw`` whenever it is already a valid JSON array.
2. Otherwise, locate the first generated JSON-array-like region in ``pred_raw``
   and scan it for fully decoded values belonging to ``evidence``, ``mention``,
   ``locDesc``, or ``locationDesc`` keys at array-element boundaries.
3. Convert each complete value into one evidence entity.
4. Discard incomplete trailing values. Never repair the source text by
   appending guessed braces or brackets.
5. Optionally deduplicate repeated recovered evidence entities. The default,
   ``exact``, removes byte-for-byte-equivalent evidence lists. A strict
   no-deduplication sensitivity score is always written for auditability.

The overall and gold-category-stratified F1 calculations reproduce the active
logic in ``eval_entities_evidence_micro_new_harvey.py`` and its 4-bit variant:
global greedy one-to-one RapidFuzz ratio matching, with the gold category used
only after a match is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOCATION_VALUE_KEYS = {"evidence", "mention", "locDesc", "locationDesc"}
DEDUPE_POLICIES = ("none", "exact", "normalized")

try:
    from rapidfuzz import __version__ as RAPIDFUZZ_VERSION
    from rapidfuzz import fuzz

    FUZZY_BACKEND = f"rapidfuzz-{RAPIDFUZZ_VERSION}"

    def fuzzy_ratio_0_100(a: str, b: str) -> float:
        return float(fuzz.ratio(_simple_norm_text(a), _simple_norm_text(b)))

except Exception:
    FUZZY_BACKEND = "difflib.SequenceMatcher"

    def fuzzy_ratio_0_100(a: str, b: str) -> float:
        a = _simple_norm_text(a)
        b = _simple_norm_text(b)
        if not a or not b:
            return 0.0
        return 100.0 * SequenceMatcher(None, a, b).ratio()


def _simple_norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().strip().split())


def _json_array_or_none(text: Any) -> Optional[List[Dict[str, Any]]]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def _location_value_to_evidence(value: Any) -> List[str]:
    values = value if isinstance(value, list) else [value]
    evidence: List[str] = []
    for item in values:
        # The requested schema is a string or list of strings. Numeric scalar
        # values are retained as text, matching the evaluator's str() behavior.
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()
        if text:
            evidence.append(text)
    return evidence


def _generated_array_bounds(text: str) -> Optional[Tuple[int, int]]:
    """
    Return the first plausible generated array region, ignoring quoted brackets.

    Gemma's outer array is often unclosed, so the end falls back to ``len(text)``.
    Array nesting is tracked independently of malformed/missing object braces.
    """
    in_string = False
    escaped = False
    array_start: Optional[int] = None

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char != "[":
            continue

        next_index = index + 1
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        if next_index == len(text) or text[next_index] in "{]":
            array_start = index
            break

    if array_start is None:
        return None

    in_string = False
    escaped = False
    array_depth = 0
    for index in range(array_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            array_depth += 1
        elif char == "]":
            array_depth -= 1
            if array_depth == 0:
                return array_start, index + 1

    return array_start, len(text)


def _at_array_element_boundary(text: str, key_start: int, array_start: int) -> bool:
    """
    Guard against promoting prose or nested metadata fields to entities.

    Normal elements begin ``[{"evidence"...`` or ``, {"evidence"...``.
    Gemma also repeats the key inside its malformed element as
    ``, "evidence"...``; accepting that form is necessary for these outputs.
    """
    # Track whether each currently open object itself began like an array
    # element. This rejects fields inside ordinary nested metadata objects,
    # including a nested field that follows another metadata member.
    object_is_element: List[bool] = []
    in_string = False
    escaped = False
    previous_significant = "["
    for char in text[array_start + 1 : key_start]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            object_is_element.append(previous_significant in "[,")
            previous_significant = char
        elif char == "}":
            if object_is_element:
                object_is_element.pop()
            previous_significant = char
        elif not char.isspace():
            previous_significant = char

    previous = key_start - 1
    while previous > array_start and text[previous].isspace():
        previous -= 1
    return (
        previous > array_start
        and text[previous] in "{,"
        and bool(object_is_element)
        and object_is_element[-1]
    )


def _decode_list_with_missing_closer(
    text: str,
    value_start: int,
    array_end: int,
    decoder: json.JSONDecoder,
) -> Optional[Tuple[List[Any], int]]:
    """
    Decode Qwen's constrained ``["item"}`` evidence-list failure.

    The evidence items themselves must be complete JSON scalars. Recovery is
    accepted only when the missing list closer is immediately followed by an
    object close whose next token is an array/object boundary. The returned end
    points at the object close; source delimiters are never rewritten.
    """
    if value_start >= array_end or text[value_start] != "[":
        return None

    values: List[Any] = []
    position = value_start + 1
    while True:
        while position < array_end and text[position].isspace():
            position += 1
        if position >= array_end:
            return None

        try:
            value, value_end = decoder.raw_decode(text, position)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        values.append(value)

        position = value_end
        while position < array_end and text[position].isspace():
            position += 1
        if position >= array_end:
            return None
        if text[position] == ",":
            position += 1
            continue
        if text[position] != "}":
            return None

        after_object = position + 1
        while after_object < array_end and text[after_object].isspace():
            after_object += 1
        if after_object < array_end and text[after_object] not in ",]}":
            return None
        return values, position


def extract_complete_location_values(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract only complete JSON values following recognized location keys.

    The surrounding object/array may be malformed. ``JSONDecoder.raw_decode``
    must still successfully decode the entire field value, so an unfinished
    final evidence list is excluded rather than repaired.
    """
    text = "" if raw is None else str(raw)
    decoder = json.JSONDecoder()
    entities: List[Dict[str, Any]] = []
    stats = {
        "recognized_keys": 0,
        "complete_values": 0,
        "incomplete_values": 0,
        "missing_list_closer_values": 0,
        "empty_values": 0,
    }

    bounds = _generated_array_bounds(text)
    if bounds is None:
        return entities, stats
    array_start, array_end = bounds

    i = array_start + 1
    while i < array_end:
        if text[i] != '"':
            i += 1
            continue

        try:
            token, token_end = decoder.raw_decode(text, i)
        except (TypeError, ValueError, json.JSONDecodeError):
            i += 1
            continue

        value_start = token_end
        while value_start < array_end and text[value_start].isspace():
            value_start += 1

        is_location_key = (
            isinstance(token, str)
            and token in LOCATION_VALUE_KEYS
            and value_start < array_end
            and text[value_start] == ":"
            and _at_array_element_boundary(text, i, array_start)
        )
        if not is_location_key:
            i = token_end
            continue

        stats["recognized_keys"] += 1
        value_start += 1
        while value_start < array_end and text[value_start].isspace():
            value_start += 1

        used_missing_list_closer = False
        try:
            value, value_end = decoder.raw_decode(text, value_start)
        except (TypeError, ValueError, json.JSONDecodeError):
            fallback = _decode_list_with_missing_closer(
                text,
                value_start,
                array_end,
                decoder,
            )
            if fallback is None:
                stats["incomplete_values"] += 1
                i = token_end
                continue
            value, value_end = fallback
            used_missing_list_closer = True
        if value_end > array_end:
            stats["incomplete_values"] += 1
            i = token_end
            continue
        boundary = value_end
        while boundary < array_end and text[boundary].isspace():
            boundary += 1
        if boundary < array_end and text[boundary] not in ",}]":
            stats["incomplete_values"] += 1
            i = token_end
            continue

        stats["complete_values"] += 1
        if used_missing_list_closer:
            stats["missing_list_closer_values"] += 1
        evidence = _location_value_to_evidence(value)
        if evidence:
            entities.append({"evidence": evidence})
        else:
            stats["empty_values"] += 1
        i = value_end

    return entities, stats


def _dedupe_key(entity: Dict[str, Any], policy: str) -> Any:
    evidence = entity.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [evidence]
    if policy == "exact":
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    if policy == "normalized":
        return tuple(_simple_norm_text(item) for item in evidence)
    raise ValueError(f"No key exists for dedupe policy: {policy}")


def dedupe_recovered_entities(
    entities: Sequence[Dict[str, Any]],
    policy: str,
) -> Tuple[List[Dict[str, Any]], int]:
    if policy not in DEDUPE_POLICIES:
        raise ValueError(f"Unknown dedupe policy {policy!r}; choose from {DEDUPE_POLICIES}")
    if policy == "none":
        return list(entities), 0

    output: List[Dict[str, Any]] = []
    seen = set()
    for entity in entities:
        key = _dedupe_key(entity, policy)
        if key in seen:
            continue
        seen.add(key)
        output.append(entity)
    return output, len(entities) - len(output)


def recover_prediction(
    pred_raw: Any,
    pred_postproc_raw: Any,
    dedupe_policy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    existing = _json_array_or_none(pred_postproc_raw)
    if existing is not None:
        return existing, {
            "mode": "existing_valid_array",
            "source_valid_json": True,
            "recognized_keys": 0,
            "complete_values": 0,
            "incomplete_values": 0,
            "missing_list_closer_values": 0,
            "empty_values": 0,
            "entities_before_dedupe": len(existing),
            "duplicates_dropped": 0,
            "entities_after_dedupe": len(existing),
        }

    extracted, scan_stats = extract_complete_location_values(pred_raw)
    recovered, duplicates_dropped = dedupe_recovered_entities(extracted, dedupe_policy)
    mode = "recovered_location_values" if recovered else "no_recoverable_value"
    return recovered, {
        "mode": mode,
        "source_valid_json": False,
        **scan_stats,
        "entities_before_dedupe": len(extracted),
        "duplicates_dropped": duplicates_dropped,
        "entities_after_dedupe": len(recovered),
    }


def _locdesc_from_entity(entity: Dict[str, Any]) -> str:
    evidence = (
        entity.get("evidence")
        or entity.get("mention")
        or entity.get("locDesc")
        or entity.get("locationDesc")
        or []
    )
    if not isinstance(evidence, list):
        evidence = [evidence]
    return ", ".join(
        str(item).strip()
        for item in evidence
        if item is not None and str(item).strip()
    )


def _location_category(entity: Dict[str, Any]) -> str:
    return str(
        entity.get("locationCate")
        or entity.get("locCate")
        or entity.get("category")
        or "UNKNOWN"
    ).strip()


def extract_locdesc_items(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        description = _locdesc_from_entity(entity)
        if description:
            items.append(
                {
                    "desc": description,
                    "category": _location_category(entity),
                }
            )
    return items


def greedy_locdesc_match(
    pred_items: Sequence[Dict[str, str]],
    gold_items: Sequence[Dict[str, str]],
    threshold: float,
) -> List[Dict[str, Any]]:
    gold_used = [False] * len(gold_items)
    pred_used = [False] * len(pred_items)
    matches: List[Dict[str, Any]] = []

    while True:
        best_score = -1.0
        best_i = -1
        best_j = -1

        for i, pred in enumerate(pred_items):
            if pred_used[i]:
                continue
            for j, gold in enumerate(gold_items):
                if gold_used[j]:
                    continue
                score = fuzzy_ratio_0_100(pred["desc"], gold["desc"])
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_j = j

        if best_i < 0 or best_j < 0 or best_score < threshold:
            break

        pred_used[best_i] = True
        gold_used[best_j] = True
        matches.append(
            {
                "pred_idx": best_i,
                "gold_idx": best_j,
                "score": best_score,
                "gold_category": gold_items[best_j]["category"],
                "pred_desc": pred_items[best_i]["desc"],
                "gold_desc": gold_items[best_j]["desc"],
            }
        )

    return matches


def _micro_prf_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_harvey_metrics(
    predictions: Sequence[Sequence[Dict[str, Any]]],
    golds: Sequence[Sequence[Dict[str, Any]]],
    evidence_match_threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, float]]:
    if len(predictions) != len(golds):
        raise ValueError(
            f"Prediction/gold length mismatch: {len(predictions)} != {len(golds)}"
        )

    fuzzy_threshold = evidence_match_threshold * 100.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    abs_err_sum = 0.0
    bias_sum = 0.0
    exact_count_hits = 0.0
    pred_count_sum = 0.0
    gold_count_sum = 0.0
    category_counts: Dict[str, Dict[str, float]] = {}

    for pred_entities, gold_entities in zip(predictions, golds):
        pred_items = extract_locdesc_items(pred_entities)
        gold_items = extract_locdesc_items(gold_entities)
        n_pred = len(pred_items)
        n_gold = len(gold_items)

        pred_count_sum += n_pred
        gold_count_sum += n_gold
        abs_err_sum += abs(n_pred - n_gold)
        bias_sum += n_pred - n_gold
        if n_pred == n_gold:
            exact_count_hits += 1.0

        matches = greedy_locdesc_match(
            pred_items,
            gold_items,
            threshold=fuzzy_threshold,
        )
        tp = float(len(matches))
        fp = float(n_pred) - tp
        fn = float(n_gold) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for item in gold_items:
            category = item["category"]
            category_counts.setdefault(
                category,
                {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0},
            )
            category_counts[category]["support"] += 1.0

        matched_gold_indices = {match["gold_idx"] for match in matches}
        for j, item in enumerate(gold_items):
            category = item["category"]
            if j in matched_gold_indices:
                category_counts[category]["tp"] += 1.0
            else:
                category_counts[category]["fn"] += 1.0

        # This intentionally matches the source evaluators: unmatched model
        # predictions count as overall FP but are not assigned to a category.

    overall = _micro_prf_counts(total_tp, total_fp, total_fn)
    n_rows = max(1.0, float(len(golds)))
    metrics: Dict[str, float] = {
        "json_valid_rate": 1.0,
        "locdesc_precision": float(overall["precision"]),
        "locdesc_recall": float(overall["recall"]),
        "locdesc_f1": float(overall["f1"]),
        "location_count_mae": float(abs_err_sum / n_rows),
        "location_count_bias": float(bias_sum / n_rows),
        "location_count_exact_match": float(exact_count_hits / n_rows),
        "pred_location_avg": float(pred_count_sum / n_rows),
        "gold_location_avg": float(gold_count_sum / n_rows),
    }

    category_rows: List[Dict[str, Any]] = []
    for category, counts in sorted(category_counts.items()):
        prf = _micro_prf_counts(counts["tp"], counts["fp"], counts["fn"])
        safe_category = str(category).replace(" ", "_")
        metrics[f"locdesc_{safe_category}_support"] = float(counts["support"])
        metrics[f"locdesc_{safe_category}_precision"] = float(prf["precision"])
        metrics[f"locdesc_{safe_category}_recall"] = float(prf["recall"])
        metrics[f"locdesc_{safe_category}_f1"] = float(prf["f1"])
        category_rows.append(
            {
                "category": category,
                "support": int(counts["support"]),
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
            }
        )

    overall_counts = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
    return metrics, category_rows, overall_counts


def _load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _validate_rows(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    raw_col: str,
    postprocessed_col: str,
    gold_col: str,
    id_col: str,
) -> None:
    required = {raw_col, postprocessed_col, gold_col}
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    if not rows:
        raise ValueError("Prediction CSV contains no rows.")

    if id_col in fieldnames:
        ids = [str(row.get(id_col, "")) for row in rows]
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        if duplicates:
            preview = ", ".join(repr(item) for item in duplicates[:10])
            raise ValueError(f"Duplicate {id_col!r} values found: {preview}")


def _load_gold_array(raw: Any, row_number: int) -> List[Dict[str, Any]]:
    entities = _json_array_or_none(raw)
    if entities is None:
        raise ValueError(f"Gold row {row_number} is not a valid JSON array.")
    return entities


def recover_dataset(
    rows: Sequence[Dict[str, str]],
    raw_col: str,
    postprocessed_col: str,
    gold_col: str,
    dedupe_policy: str,
) -> Tuple[
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    predictions: List[List[Dict[str, Any]]] = []
    golds: List[List[Dict[str, Any]]] = []
    audits: List[Dict[str, Any]] = []
    mode_counts: Counter = Counter()
    totals: Counter = Counter()

    for row_number, row in enumerate(rows):
        prediction, audit = recover_prediction(
            pred_raw=row.get(raw_col, ""),
            pred_postproc_raw=row.get(postprocessed_col, ""),
            dedupe_policy=dedupe_policy,
        )
        gold = _load_gold_array(row.get(gold_col, ""), row_number)
        predictions.append(prediction)
        golds.append(gold)
        audits.append(audit)
        mode_counts[audit["mode"]] += 1
        for key in (
            "recognized_keys",
            "complete_values",
            "incomplete_values",
            "missing_list_closer_values",
            "empty_values",
            "entities_before_dedupe",
            "duplicates_dropped",
            "entities_after_dedupe",
        ):
            totals[key] += int(audit[key])

    source_valid_rows = sum(bool(audit["source_valid_json"]) for audit in audits)
    recovered_nonempty_rows = sum(
        audit["mode"] == "recovered_location_values" for audit in audits
    )
    summary = {
        "rows": len(rows),
        "dedupe_policy": dedupe_policy,
        "source_valid_rows": source_valid_rows,
        "source_json_valid_rate": source_valid_rows / len(rows),
        "field_recovered_nonempty_rows": recovered_nonempty_rows,
        "field_recovered_nonempty_rate": recovered_nonempty_rows / len(rows),
        "mode_counts": dict(sorted(mode_counts.items())),
        **dict(totals),
    }
    return predictions, golds, audits, summary


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _ensure_output_paths(output_dir: Path, overwrite: bool) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "recovered_predictions.csv",
        "metrics": output_dir / "recovered_metrics.json",
        "categories": output_dir / "recovered_category_metrics.csv",
        "summary": output_dir / "recovery_summary.json",
        "sensitivity": output_dir / "recovery_sensitivity.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs. Use --overwrite:\n  " + joined
        )
    return paths


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.predictions_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / "recovered_unclosed"
    )
    output_paths = _ensure_output_paths(output_dir, overwrite=args.overwrite)
    rows, fieldnames = _load_rows(input_path)
    _validate_rows(
        rows,
        fieldnames,
        raw_col=args.raw_col,
        postprocessed_col=args.postprocessed_col,
        gold_col=args.gold_col,
        id_col=args.id_col,
    )

    predictions, golds, audits, summary = recover_dataset(
        rows,
        raw_col=args.raw_col,
        postprocessed_col=args.postprocessed_col,
        gold_col=args.gold_col,
        dedupe_policy=args.dedupe_policy,
    )
    metrics, category_rows, overall_counts = compute_harvey_metrics(
        predictions,
        golds,
        evidence_match_threshold=args.evidence_match_threshold,
    )
    # Preserve the evaluator-compatible meaning of json_valid_rate for the
    # original model output. The custom column is deliberately serialized as a
    # JSON array on every row, including [] when nothing can be recovered.
    metrics["recovered_output_json_valid_rate"] = metrics["json_valid_rate"]
    metrics["json_valid_rate"] = summary["source_json_valid_rate"]
    metrics["field_recovered_nonempty_rate"] = summary[
        "field_recovered_nonempty_rate"
    ]
    summary.update(
        {
            "input_csv": str(input_path),
            "output_dir": str(output_dir),
            "raw_column": args.raw_col,
            "postprocessed_column": args.postprocessed_col,
            "gold_column": args.gold_col,
            "evidence_match_threshold": args.evidence_match_threshold,
            "fuzzy_backend": FUZZY_BACKEND,
            "overall_counts": overall_counts,
        }
    )

    appended_fields = [
        "custom_pred_json",
        "custom_recovery_mode",
        "custom_source_valid_json",
        "custom_recognized_keys",
        "custom_complete_values",
        "custom_incomplete_values",
        "custom_missing_list_closer_values",
        "custom_entities_before_dedupe",
        "custom_duplicates_dropped",
        "custom_entities_after_dedupe",
        "custom_dedupe_policy",
    ]
    output_rows: List[Dict[str, Any]] = []
    for row, prediction, audit in zip(rows, predictions, audits):
        output_row: Dict[str, Any] = dict(row)
        output_row.update(
            {
                "custom_pred_json": json.dumps(prediction, ensure_ascii=False),
                "custom_recovery_mode": audit["mode"],
                "custom_source_valid_json": int(audit["source_valid_json"]),
                "custom_recognized_keys": audit["recognized_keys"],
                "custom_complete_values": audit["complete_values"],
                "custom_incomplete_values": audit["incomplete_values"],
                "custom_missing_list_closer_values": audit[
                    "missing_list_closer_values"
                ],
                "custom_entities_before_dedupe": audit["entities_before_dedupe"],
                "custom_duplicates_dropped": audit["duplicates_dropped"],
                "custom_entities_after_dedupe": audit["entities_after_dedupe"],
                "custom_dedupe_policy": args.dedupe_policy,
            }
        )
        output_rows.append(output_row)

    _write_csv(
        output_paths["predictions"],
        list(fieldnames) + appended_fields,
        output_rows,
    )
    _write_json(output_paths["metrics"], metrics)
    _write_csv(
        output_paths["categories"],
        ["category", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
        category_rows,
    )
    _write_json(output_paths["summary"], summary)

    sensitivity: Dict[str, Any] = {}
    for policy in DEDUPE_POLICIES:
        policy_predictions, policy_golds, _, policy_summary = recover_dataset(
            rows,
            raw_col=args.raw_col,
            postprocessed_col=args.postprocessed_col,
            gold_col=args.gold_col,
            dedupe_policy=policy,
        )
        policy_metrics, _, policy_counts = compute_harvey_metrics(
            policy_predictions,
            policy_golds,
            evidence_match_threshold=args.evidence_match_threshold,
        )
        sensitivity[policy] = {
            "locdesc_precision": policy_metrics["locdesc_precision"],
            "locdesc_recall": policy_metrics["locdesc_recall"],
            "locdesc_f1": policy_metrics["locdesc_f1"],
            "pred_location_avg": policy_metrics["pred_location_avg"],
            "duplicates_dropped": policy_summary["duplicates_dropped"],
            "overall_counts": policy_counts,
        }
    _write_json(output_paths["sensitivity"], sensitivity)

    return {
        "metrics": metrics,
        "summary": summary,
        "sensitivity": sensitivity,
        "output_paths": {name: str(path) for name, path in output_paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover complete location values from malformed pred_raw strings "
            "and recompute Harvey overall/category F1."
        )
    )
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--raw_col", default="pred_raw")
    parser.add_argument("--postprocessed_col", default="pred_postproc_raw")
    parser.add_argument("--gold_col", default="gold_raw")
    parser.add_argument("--id_col", default="idx")
    parser.add_argument("--evidence_match_threshold", type=float, default=0.75)
    parser.add_argument(
        "--dedupe_policy",
        choices=DEDUPE_POLICIES,
        default="exact",
        help=(
            "Deduplication applied only to field-recovered malformed rows. "
            "'exact' is the primary repetition-aware score; 'none' is strict."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    metrics = result["metrics"]
    summary = result["summary"]
    print(f"Rows: {summary['rows']}")
    print(f"Source JSON valid rate: {summary['source_json_valid_rate']:.6f}")
    print(
        "Recovered nonempty rows: "
        f"{summary['field_recovered_nonempty_rows']}/{summary['rows']}"
    )
    print(f"Deduplication policy: {summary['dedupe_policy']}")
    print(f"Fuzzy backend: {summary['fuzzy_backend']}")
    print(f"Overall precision: {metrics['locdesc_precision']:.6f}")
    print(f"Overall recall: {metrics['locdesc_recall']:.6f}")
    print(f"Overall F1: {metrics['locdesc_f1']:.6f}")
    print("Category F1:")
    for key, value in metrics.items():
        if key.startswith("locdesc_C") and key.endswith("_f1"):
            category = key[len("locdesc_") : -len("_f1")]
            print(f"  {category}: {value:.6f}")
    print("Saved:")
    for path in result["output_paths"].values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
