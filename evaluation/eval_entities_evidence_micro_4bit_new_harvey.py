#!/usr/bin/env python3
"""
Entity-array evaluation (CSV, no pretokenization) for instruction->JSON outputs.

CSV must include columns:
  - prompt
  - target_json (gold JSON string; JSON ARRAY of objects)

Supports:
  - base model
  - full FT checkpoint saved with save_pretrained
  - LoRA adapter checkpoint (PEFT) saved with save_pretrained, via --use_lora_adapter

Loads checkpoint from:
  checkpoints/<model_name>_<checkpoint_path>/best
(where checkpoint_path is the TIMESTAMP you used in training)

ADDED (error analysis; does not change evaluation metrics):
  - Writes error analysis CSVs + summary TXT under the same out_dir:
      * test_error_analysis_rows.csv
      * test_error_analysis_state_errors.csv
      * test_error_analysis_county_errors.csv
      * test_error_analysis_city_errors.csv
      * test_error_analysis_full_errors.csv
      * test_error_analysis_evidence_errors.csv
      * test_error_analysis_entity_prf_by_item_state.csv
      * test_error_analysis_entity_prf_by_item_county.csv
      * test_error_analysis_entity_prf_by_item_city.csv
      * test_error_analysis_summary.txt
      * (optional) confusion matrices if scikit-learn is installed:
          - test_confusion_state.csv
          - test_confusion_county_topK.csv
          - test_confusion_city_topK.csv

ADDED (run subfolder; minimal change):
  - Creates a subfolder named: <test_csv_stem>_<timestamp>
    and writes ALL results there.

UPDATED (minimal change): repetition vs hallucination aware post-processing:
  - Extract ALL JSON arrays from model output.
  - If repeated identical arrays -> keep one (dedupe repetition).
  - If multiple DISTINCT arrays -> merge entities.
  - Dedupe by geo_key ONLY if evidence is equal/subset; else keep both.
  - Save per-row stats in:
      * error_analysis_rows.csv
      * all_predictions.csv (NEW)
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import gather_object

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from dotenv import load_dotenv
import wandb

from transformers import BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:
    PeftModel = None




# 2) ADD this stopping criteria class (anywhere above evaluate())

class StopOnFirstJsonArray(StoppingCriteria):
    """
    Stops generation once the FIRST top-level JSON array is closed:
      - start counting after we see the first '['
      - track bracket depth (only for [ and ])
      - ignore brackets inside JSON strings (tracks quotes + escapes)
      - stop when depth returns to 0 after starting
    IMPORTANT: ignores the prompt part by using prompt_len.
    """

    def __init__(self, tokenizer, prompt_len: int, batch_size: int):
        super().__init__()
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)

        # per-seq state
        self.done = [False] * batch_size
        self.started = [False] * batch_size
        self.depth = [0] * batch_size
        self.in_str = [False] * batch_size
        self.esc = [False] * batch_size
        self.last_len = [self.prompt_len] * batch_size  # only process newly generated tokens

    def _consume_text(self, si: int, text: str):
        for ch in text:
            if self.in_str[si]:
                if self.esc[si]:
                    self.esc[si] = False
                elif ch == "\\":
                    self.esc[si] = True
                elif ch == '"':
                    self.in_str[si] = False
                continue

            # not in string
            if ch == '"':
                self.in_str[si] = True
                continue

            if ch == "[":
                self.started[si] = True
                self.depth[si] += 1
            elif ch == "]":
                if self.started[si]:
                    self.depth[si] -= 1
                    if self.depth[si] <= 0:
                        self.done[si] = True
                        return  # this sequence is finished

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, seq_len]
        bsz, seq_len = input_ids.shape

        for i in range(bsz):
            if self.done[i]:
                continue

            prev = self.last_len[i]
            if seq_len <= prev:
                continue

            new_ids = input_ids[i, prev:seq_len].tolist()
            self.last_len[i] = seq_len

            # decode incremental tokens (keeps punctuation reliably)
            text = self.tok.decode(new_ids, skip_special_tokens=False)
            if text:
                self._consume_text(i, text)

        # stop when ALL sequences are done
        return all(self.done)

# --------------------------
# Dataset
# --------------------------

def evidence_anywhere_micro_counts(pred_valid, gold_valid, evidence_match_threshold, evidence_ok_threshold):
    # Build evidence lists
    def get_ev(ent):
        ev = ent.get("evidence") or ent.get("mention") or []
        if not isinstance(ev, list):
            ev = [ev]
        # normalize-empty removal is already inside greedy matcher, but ok to keep raw
        return ev

    # Score matrix: pred i -> gold j (entity-level evidence F1)
    scores = []
    for i, pe in enumerate(pred_valid):
        row = []
        for j, ge in enumerate(gold_valid):
            row.append(evidence_f1_for_pair(pe, ge, evidence_match_threshold))
        scores.append(row)

    gold_used = [False] * len(gold_valid)
    matched_pred = [False] * len(pred_valid)

    tp = fp = fn = 0.0

    # Greedy: repeatedly take the best remaining (i,j)
    while True:
        best = 0.0
        best_i = best_j = -1
        for i in range(len(pred_valid)):
            if matched_pred[i]:
                continue
            for j in range(len(gold_valid)):
                if gold_used[j]:
                    continue
                s = scores[i][j] if scores else 0.0
                if s > best:
                    best = s
                    best_i, best_j = i, j

        if best_i < 0 or best < evidence_ok_threshold:
            break

        # match entity i -> j
        matched_pred[best_i] = True
        gold_used[best_j] = True

        pe_ev = get_ev(pred_valid[best_i])
        ge_ev = get_ev(gold_valid[best_j])

        m = greedy_evidence_match(pe_ev, ge_ev, evidence_match_threshold)
        tp += m
        fp += max(0, len(pe_ev) - m)
        fn += max(0, len(ge_ev) - m)

    # Unmatched pred entities => all evidence phrases are FP (hallucinated/wrong)
    for i, pe in enumerate(pred_valid):
        if matched_pred[i]:
            continue
        pe_ev = get_ev(pe)
        fp += len(pe_ev)

    # Unmatched gold entities => all evidence phrases are FN (missed)
    for j, ge in enumerate(gold_valid):
        if gold_used[j]:
            continue
        ge_ev = get_ev(ge)
        fn += len(ge_ev)

    return tp, fp, fn


class PromptTargetCSVDataset(Dataset):
    def __init__(self, csv_path: str, prompt_col="prompt", target_col="target_json"):
        df = pd.read_csv(csv_path, dtype=str)
        if prompt_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"CSV must have columns: {prompt_col}, {target_col}. Found: {list(df.columns)}")

        prompts = df[prompt_col].fillna("").astype(str).tolist()
        targets = df[target_col].fillna("").astype(str).tolist()

        self.examples = []
        for p, t in zip(prompts, targets):
            p = p.strip()
            t = t.strip()
            if p and t:
                self.examples.append({"prompt": p, "target": t})

        if not self.examples:
            raise ValueError(f"No valid rows found in {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --------------------------
# Collator
# --------------------------
class EvalCollatorTokenize:
    def __init__(self, tokenizer, max_length=512):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]):
        prompts = [ex["prompt"] for ex in batch]
        targets = [ex["target"] for ex in batch]

        enc = self.tok(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompts": prompts,
            "targets": targets,
        }


# --------------------------
# Normalization + parsing
# --------------------------
def _norm_admin(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = s.replace(" county", "").strip()
    s = "".join(s.split())  # remove all whitespace
    return s or None

def _norm_evidence_item(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    # keep / and - for road-ish strings
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in "/-")
    s = " ".join(s.split())
    return s or None

def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _norm_evidence_list(ev: Any) -> List[str]:
    out = []
    for it in _ensure_list(ev):
        ni = _norm_evidence_item(it)
        if ni:
            out.append(ni)
    return out

def _evidence_subset_or_equal(ev_a: Any, ev_b: Any) -> bool:
    A = set(_norm_evidence_list(ev_a))
    B = set(_norm_evidence_list(ev_b))
    if not A and not B:
        return True
    return A.issubset(B) or B.issubset(A)

def _merge_evidence_union(ev_a: Any, ev_b: Any) -> List[str]:
    A = set(_norm_evidence_list(ev_a))
    B = set(_norm_evidence_list(ev_b))
    return sorted(A | B)

def entity_geo_key(ent: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    st = _norm_admin(ent.get("state"))
    co = _norm_admin(ent.get("county"))
    ci = _norm_admin(ent.get("city"))
    if not st or not co:
        return None
    return (st, co, ci or "")

def extract_geo_sets(entities: List[Dict[str, Any]]) -> Dict[str, set]:
    states, counties, cities = set(), set(), set()
    for e in entities:
        st = _norm_admin(e.get("state"))
        co = _norm_admin(e.get("county"))
        ci = _norm_admin(e.get("city"))
        if st:
            states.add(st)
        if co:
            counties.add(co)
        if ci:
            cities.add(ci)
    return {"state": states, "county": counties, "city": cities}


# --------------------------
# UPDATED: repetition/hallucination-aware JSON extraction + conditioned geo-dedupe
# --------------------------
def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)

def extract_all_json_arrays_balanced(text: str, max_arrays: int = 25) -> List[str]:
    if text is None:
        return []
    s = str(text)
    out: List[str] = []
    i = 0
    n = len(s)

    while i < n and len(out) < max_arrays:
        start = s.find("[", i)
        if start == -1:
            break

        depth = 0
        in_str = False
        esc = False

        for j in range(start, n):
            ch = s[j]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = s[start:j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, list):
                            out.append(candidate)
                    except Exception:
                        pass
                    i = j + 1
                    break

            if depth < 0:
                i = start + 1
                break
        else:
            break

    return out

def _geo_dedupe_entities_conditioned(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "geo_dedupe_dropped": 0,
        "geo_dedupe_merged": 0,
        "geo_conflict_kept": 0,
    }

    kept: List[Dict[str, Any]] = []
    geo_to_idx: Dict[Tuple[str, str, str], int] = {}

    for e in entities:
        if not isinstance(e, dict):
            continue
        gk = entity_geo_key(e)
        if gk is None:
            kept.append(e)
            continue

        if gk not in geo_to_idx:
            geo_to_idx[gk] = len(kept)
            kept.append(e)
            continue

        idx = geo_to_idx[gk]
        base = kept[idx]

        base_ev = base.get("evidence", base.get("mention", []))
        new_ev = e.get("evidence", e.get("mention", []))

        if _evidence_subset_or_equal(base_ev, new_ev):
            # merge + drop
            merged_ev = _merge_evidence_union(base_ev, new_ev)
            base["evidence"] = merged_ev
            kept[idx] = base
            stats["geo_dedupe_dropped"] += 1
            stats["geo_dedupe_merged"] += 1
        else:
            # conflict => keep both
            kept.append(e)
            stats["geo_conflict_kept"] += 1

    return kept, stats

def postprocess_pred_text_entity_arrays(raw: str) -> Tuple[str, Dict[str, Any]]:
    """
    Plan:
      - Extract ALL arrays
      - Dedupe identical arrays
      - Merge entities across distinct arrays
      - Conditioned geo_dedupe by evidence subset/equal
    """
    info: Dict[str, Any] = {
        "arrays_found": 0,
        "arrays_unique": 0,
        "arrays_repeated_dropped": 0,
        "mode": "none",  # none|single|repetition|multi
        "entities_before_geo_dedupe": 0,
        "entities_after_geo_dedupe": 0,
        "geo_dedupe_dropped": 0,
        "geo_dedupe_merged": 0,
        "geo_conflict_kept": 0,
    }

    if raw is None:
        return "", info

    arrays = extract_all_json_arrays_balanced(raw, max_arrays=25)
    info["arrays_found"] = len(arrays)

    if not arrays:
        # leave as-is; parse will likely fail => invalid_json
        return ("" if raw is None else str(raw)), info

    # Parse + canonicalize arrays for dedupe
    canon_to_ents: Dict[str, List[Dict[str, Any]]] = {}
    for a in arrays:
        try:
            obj = json.loads(a)
        except Exception:
            continue
        if not isinstance(obj, list):
            continue
        ents = [x for x in obj if isinstance(x, dict)]
        canon = _canonical_json(ents)
        if canon not in canon_to_ents:
            canon_to_ents[canon] = ents

    unique_canons = list(canon_to_ents.keys())
    info["arrays_unique"] = len(unique_canons)
    info["arrays_repeated_dropped"] = max(0, info["arrays_found"] - info["arrays_unique"])

    if info["arrays_unique"] == 0:
        return ("" if raw is None else str(raw)), info

    if info["arrays_unique"] == 1:
        info["mode"] = "repetition" if info["arrays_found"] > 1 else "single"
    else:
        info["mode"] = "multi"

    # Merge entities across unique arrays
    merged: List[Dict[str, Any]] = []
    for c in unique_canons:
        merged.extend(canon_to_ents[c])

    info["entities_before_geo_dedupe"] = len(merged)

    # Conditioned geo-key dedupe
    merged2, geo_stats = _geo_dedupe_entities_conditioned(merged)
    info.update(geo_stats)
    info["entities_after_geo_dedupe"] = len(merged2)

    return json.dumps(merged2, ensure_ascii=False), info


def _try_json_load(raw: str):
    try:
        return json.loads(raw), True
    except Exception:
        arrays = extract_all_json_arrays_balanced(raw, max_arrays=1)
        if arrays:
            try:
                return json.loads(arrays[0]), True
            except Exception:
                return None, False
        return None, False

def parse_entity_array(text: str) -> Dict[str, Any]:
    if text is None:
        return {"__valid_json": False, "entities": []}

    raw = str(text).strip()
    if not raw:
        return {"__valid_json": False, "entities": []}

    obj, ok = _try_json_load(raw)
    if not ok or not isinstance(obj, list):
        return {"__valid_json": False, "entities": []}

    ents = [x for x in obj if isinstance(x, dict)]
    return {"__valid_json": True, "entities": ents}


# --------------------------
# Evidence fuzzy matching (char 3-gram Jaccard)
# --------------------------
def _char_ngrams(s: str, n: int = 3) -> set:
    if s is None:
        return set()
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def evidence_similarity(a: str, b: str) -> float:
    na = _norm_evidence_item(a)
    nb = _norm_evidence_item(b)
    if not na or not nb:
        return 0.0
    A = _char_ngrams(na, 3)
    B = _char_ngrams(nb, 3)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def greedy_evidence_match(pred_list: List[str], gold_list: List[str], thr: float) -> int:
    preds = [p for p in (pred_list or []) if _norm_evidence_item(p)]
    golds = [g for g in (gold_list or []) if _norm_evidence_item(g)]
    if not preds or not golds:
        return 0

    used = [False] * len(golds)
    match = 0
    for p in preds:
        best_j = -1
        best_s = 0.0
        for j, g in enumerate(golds):
            if used[j]:
                continue
            s = evidence_similarity(p, g)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j >= 0 and best_s >= thr:
            used[best_j] = True
            match += 1
    return match

# def evidence_f1_for_pair(pred_ent: Dict[str, Any], gold_ent: Dict[str, Any], thr: float) -> float:
#     pred_e = pred_ent.get("evidence") or pred_ent.get("mention") or []
#     gold_e = gold_ent.get("evidence") or gold_ent.get("mention") or []
#     if not isinstance(pred_e, list):
#         pred_e = [pred_e]
#     if not isinstance(gold_e, list):
#         gold_e = [gold_e]

#     m = greedy_evidence_match(pred_e, gold_e, thr)
#     p = m / len(pred_e) if pred_e else (1.0 if not gold_e else 0.0)
#     r = m / len(gold_e) if gold_e else (1.0 if not pred_e else 0.0)
#     return (2 * p * r / (p + r)) if (p + r) else 0.0
def evidence_f1_for_pair(pred_ent: Dict[str, Any], gold_ent: Dict[str, Any], thr: float) -> float:
    pred_e = pred_ent.get("evidence") or pred_ent.get("mention") or []
    gold_e = gold_ent.get("evidence") or gold_ent.get("mention") or []

    if not isinstance(pred_e, list):
        pred_e = [pred_e]
    if not isinstance(gold_e, list):
        gold_e = [gold_e]

    # Original phrase-level matching
    m = greedy_evidence_match(pred_e, gold_e, thr)
    p = m / len(pred_e) if pred_e else (1.0 if not gold_e else 0.0)
    r = m / len(gold_e) if gold_e else (1.0 if not pred_e else 0.0)
    original_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

    # New fallback: concatenate evidence pieces and compare as one phrase
    pred_joined = ", ".join(str(x) for x in pred_e if _norm_evidence_item(x))
    gold_joined = ", ".join(str(x) for x in gold_e if _norm_evidence_item(x))

    joined_sim = evidence_similarity(pred_joined, gold_joined)

    if joined_sim >= thr:
        joined_f1 = 1.0
    else:
        joined_f1 = joined_sim

    return max(original_f1, joined_f1)

# --------------------------
# Matching + metrics
# --------------------------
def _micro_prf_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def match_by_geo(pred_ents: List[Dict[str, Any]], gold_ents: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    pred_keys = [entity_geo_key(e) for e in pred_ents]
    gold_keys = [entity_geo_key(e) for e in gold_ents]

    gold_used = [False] * len(gold_ents)
    matches = []
    for i, pk in enumerate(pred_keys):
        if pk is None:
            continue
        for j, gk in enumerate(gold_keys):
            if gold_used[j]:
                continue
            if gk is None:
                continue
            if pk == gk:
                gold_used[j] = True
                matches.append((i, j))
                break
    return matches


def compute_test_metrics_entities_json(
    pred_texts: List[str],
    gold_texts: List[str],
    evidence_match_threshold: float = 0.75,
    evidence_ok_threshold: float = 0.50,
) -> Dict[str, float]:

    # Match their style: fuzzy ratio threshold 75
    fuzzy_threshold = evidence_match_threshold * 100.0

    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    json_valid_rate = sum(
        1 for p in parsed_pred if p["__valid_json"]
    ) / max(1, len(parsed_pred))

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    abs_err_sum = 0.0
    bias_sum = 0.0
    exact_count_hits = 0.0
    pred_count_sum = 0.0
    gold_count_sum = 0.0

    category_counts = {}

    for p, g in zip(parsed_pred, parsed_gold):
        pred_items = extract_locdesc_items(p)
        gold_items = extract_locdesc_items(g)

        n_pred = len(pred_items)
        n_gold = len(gold_items)

        pred_count_sum += n_pred
        gold_count_sum += n_gold
        abs_err_sum += abs(n_pred - n_gold)
        bias_sum += (n_pred - n_gold)

        if n_pred == n_gold:
            exact_count_hits += 1.0

        matches, pred_used, gold_used = greedy_locdesc_match(
            pred_items,
            gold_items,
            threshold=fuzzy_threshold
        )

        tp = len(matches)
        fp = n_pred - tp
        fn = n_gold - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Gold-category stratified evaluation
        for item in gold_items:
            c = item["category"]
            category_counts.setdefault(c, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0})
            category_counts[c]["support"] += 1.0

        matched_gold_indices = set(m["gold_idx"] for m in matches)

        for j, item in enumerate(gold_items):
            c = item["category"]
            if j in matched_gold_indices:
                category_counts[c]["tp"] += 1.0
            else:
                category_counts[c]["fn"] += 1.0

        # Since model does not predict category, we do not assign unmatched predictions
        # to a category. They are counted in overall FP, not category-level FP.

    locdesc_prf = _micro_prf_counts(total_tp, total_fp, total_fn)

    n = max(1.0, len(parsed_gold))

    metrics = {
        "json_valid_rate": float(json_valid_rate),

        "locdesc_precision": float(locdesc_prf["precision"]),
        "locdesc_recall": float(locdesc_prf["recall"]),
        "locdesc_f1": float(locdesc_prf["f1"]),

        "location_count_mae": float(abs_err_sum / n),
        "location_count_bias": float(bias_sum / n),
        "location_count_exact_match": float(exact_count_hits / n),
        "pred_location_avg": float(pred_count_sum / n),
        "gold_location_avg": float(gold_count_sum / n),
    }

    # Category-based scores using gold category only
    for c, d in sorted(category_counts.items()):
        prf = _micro_prf_counts(d["tp"], d["fp"], d["fn"])

        safe_c = str(c).replace(" ", "_")

        metrics[f"locdesc_{safe_c}_support"] = float(d["support"])
        metrics[f"locdesc_{safe_c}_precision"] = float(prf["precision"])
        metrics[f"locdesc_{safe_c}_recall"] = float(prf["recall"])
        metrics[f"locdesc_{safe_c}_f1"] = float(prf["f1"])

    return metrics
# def compute_test_metrics_entities_json(
#     pred_texts: List[str],
#     gold_texts: List[str],
#     evidence_match_threshold: float = 0.75,
#     evidence_ok_threshold: float = 0.50,
# ) -> Dict[str, float]:
#     parsed_pred = [parse_entity_array(x) for x in pred_texts]
#     parsed_gold = [parse_entity_array(x) for x in gold_texts]

#     json_valid_rate = sum(1 for p in parsed_pred if p["__valid_json"]) / max(1, len(parsed_pred))

#     pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
#     gold_state_sets, gold_county_sets, gold_city_sets = [], [], []

#     pred_sig_sets: List[set] = []
#     gold_sig_sets: List[set] = []

#     ev_tp = ev_fp = ev_fn = 0.0
#     s_tp = s_fp = s_fn = 0.0

#     ev2_tp = ev2_fp = ev2_fn = 0.0

#     abs_err_sum = 0.0
#     bias_sum = 0.0
#     exact_count_hits = 0.0
#     pred_count_sum = 0.0
#     gold_count_sum = 0.0

#     for p, g in zip(parsed_pred, parsed_gold):
#         pred_ents = p["entities"]
#         gold_ents = g["entities"]

#         pred_valid = [e for e in pred_ents if entity_geo_key(e) is not None]
#         gold_valid = [e for e in gold_ents if entity_geo_key(e) is not None]

#         psets = extract_geo_sets(pred_valid)
#         gsets = extract_geo_sets(gold_valid)
#         pred_state_sets.append(psets["state"])
#         pred_county_sets.append(psets["county"])
#         pred_city_sets.append(psets["city"])
#         gold_state_sets.append(gsets["state"])
#         gold_county_sets.append(gsets["county"])
#         gold_city_sets.append(gsets["city"])

#         p_sigs = set(entity_geo_key(e) for e in pred_valid)
#         p_sigs.discard(None)
#         g_sigs = set(entity_geo_key(e) for e in gold_valid)
#         g_sigs.discard(None)

#         pred_sig_sets.append(p_sigs)
#         gold_sig_sets.append(g_sigs)

#         n_pred = len(p_sigs)
#         n_gold = len(g_sigs)
#         pred_count_sum += n_pred
#         gold_count_sum += n_gold
#         abs_err_sum += abs(n_pred - n_gold)
#         bias_sum += (n_pred - n_gold)
#         if n_pred == n_gold:
#             exact_count_hits += 1.0

#         matches = match_by_geo(pred_valid, gold_valid)

#         for (pi, gi) in matches:
#             pe = pred_valid[pi].get("evidence", pred_valid[pi].get("mention", []))
#             ge = gold_valid[gi].get("evidence", gold_valid[gi].get("mention", []))
#             if not isinstance(pe, list):
#                 pe = [pe]
#             if not isinstance(ge, list):
#                 ge = [ge]
#             m = greedy_evidence_match(pe, ge, evidence_match_threshold)
#             ev_tp += m
#             ev_fp += max(0, len(pe) - m)
#             ev_fn += max(0, len(ge) - m)

#         struct_tp_here = 0.0
#         for (pi, gi) in matches:
#             ef1 = evidence_f1_for_pair(pred_valid[pi], gold_valid[gi], evidence_match_threshold)
#             if ef1 >= evidence_ok_threshold:
#                 struct_tp_here += 1.0

#         s_tp += struct_tp_here
#         s_fp += max(0.0, n_pred - struct_tp_here)
#         s_fn += max(0.0, n_gold - struct_tp_here)

#         tp2, fp2, fn2 = evidence_anywhere_micro_counts(pred_valid, gold_valid, evidence_match_threshold=evidence_match_threshold, evidence_ok_threshold=evidence_ok_threshold)
#         ev2_tp += tp2
#         ev2_fp += fp2
#         ev2_fn += fn2

#     def micro_prf_from_sets(pred_sets: List[set], gold_sets: List[set]) -> Dict[str, float]:
#         tp = fp = fn = 0.0
#         for pset, gset in zip(pred_sets, gold_sets):
#             tp += len(pset & gset)
#             fp += len(pset - gset)
#             fn += len(gset - pset)
#         return _micro_prf_counts(tp, fp, fn)

#     state_prf = micro_prf_from_sets(pred_state_sets, gold_state_sets)
#     county_prf = micro_prf_from_sets(pred_county_sets, gold_county_sets)
#     city_prf = micro_prf_from_sets(pred_city_sets, gold_city_sets)
    

#     geo_tp = geo_fp = geo_fn = 0.0
#     geo_em_hits = 0.0
#     for p_sigs, g_sigs in zip(pred_sig_sets, gold_sig_sets):
#         geo_tp += len(p_sigs & g_sigs)
#         geo_fp += len(p_sigs - g_sigs)
#         geo_fn += len(g_sigs - p_sigs)
#         if p_sigs == g_sigs:
#             geo_em_hits += 1.0
#     geo_prf = _micro_prf_counts(geo_tp, geo_fp, geo_fn)
#     geo_exact_match = geo_em_hits / max(1.0, len(pred_sig_sets))

#     ev_prf = _micro_prf_counts(ev_tp, ev_fp, ev_fn)
#     structured_prf = _micro_prf_counts(s_tp, s_fp, s_fn)
#     ev2_prf = _micro_prf_counts(ev2_tp, ev2_fp, ev2_fn)

#     n = max(1.0, len(pred_sig_sets))
#     entity_count_mae = abs_err_sum / n
#     entity_count_bias = bias_sum / n
#     entity_count_exact_match_rate = exact_count_hits / n
#     pred_entity_avg = pred_count_sum / n

#     return {
#         "json_valid_rate": float(json_valid_rate),

#         "state_f1": float(state_prf["f1"]),
#         "county_f1": float(county_prf["f1"]),
#         "city_f1": float(city_prf["f1"]),

#         "geo_obj_precision": float(geo_prf["precision"]),
#         "geo_obj_recall": float(geo_prf["recall"]),
#         "geo_obj_f1": float(geo_prf["f1"]),
#         "geo_obj_exact_match": float(geo_exact_match),

#         "evidence_precision": float(ev_prf["precision"]),
#         "evidence_recall": float(ev_prf["recall"]),
#         "evidence_f1": float(ev_prf["f1"]),

#         "structured_precision": float(structured_prf["precision"]),
#         "structured_recall": float(structured_prf["recall"]),
#         "structured_f1": float(structured_prf["f1"]),

#         "entity_count_mae": float(entity_count_mae),
#         "entity_count_bias": float(entity_count_bias),
#         "entity_count_exact_match": float(entity_count_exact_match_rate),
#         "pred_entity_avg": float(pred_entity_avg),
#         "gold_entity_avg": float(sum(len(s) for s in gold_sig_sets) / n),
#         "evidence_any_precision" : float(ev2_prf["precision"]),
#         "evidence_any_recall" : float(ev2_prf["recall"]),
#         "evidence_any_f1" : float(ev2_prf["f1"]),
#     }


# --------------------------
# Error analysis helpers
# --------------------------
def _set_to_pipe(s: set) -> str:
    return "|".join(sorted(s))

def _safe_extract_text_from_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""
    i = prompt.find("Text:")
    if i == -1:
        return ""
    return prompt[i + len("Text:"):].strip()

def _entity_prf_by_item(pred_sets: List[set], gold_sets: List[set]) -> pd.DataFrame:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()

    for p, g in zip(pred_sets, gold_sets):
        for it in g:
            support[it] += 1
        for it in (p & g):
            tp[it] += 1
        for it in (p - g):
            fp[it] += 1
        for it in (g - p):
            fn[it] += 1

    items = set(support.keys()) | set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    rows = []
    for it in items:
        tpi = int(tp.get(it, 0))
        fpi = int(fp.get(it, 0))
        fni = int(fn.get(it, 0))
        sup = int(support.get(it, 0))
        prec = tpi / (tpi + fpi) if (tpi + fpi) else 0.0
        rec = tpi / (tpi + fni) if (tpi + fni) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        rows.append({
            "item": it,
            "support": sup,
            "tp": tpi,
            "fp": fpi,
            "fn": fni,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["support", "f1", "item"], ascending=[False, True, True]).reset_index(drop=True)

def _top1_label(s: set, empty_label: str = "__EMPTY__") -> str:
    if not s:
        return empty_label
    return sorted(s)[0]

def run_error_analysis(
    out_dir: Path,
    run_note: Optional[str],
    prompts: List[str],
    pred_texts: List[str],
    gold_texts: List[str],
    pred_infos: Optional[List[Dict[str, Any]]] = None,
    evidence_match_threshold: float = 0.75,
    evidence_ok_threshold: float = 0.50,
    top_k_labels: int = 50,
):
    suffix = f"_{run_note}" if run_note else ""
    base = f"test_error_analysis{suffix}"

    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
    gold_state_sets, gold_county_sets, gold_city_sets = [], [], []
    pred_sig_sets, gold_sig_sets = [], []

    evidence_error_rows = []
    rows = []

    for i in range(len(parsed_pred)):
        p = parsed_pred[i]
        g = parsed_gold[i]
        prompt = prompts[i] if i < len(prompts) else ""
        text = _safe_extract_text_from_prompt(prompt)
        info = (pred_infos[i] if (pred_infos is not None and i < len(pred_infos)) else {}) or {}

        pred_ents = p["entities"]
        gold_ents = g["entities"]

        pred_valid = [e for e in pred_ents if entity_geo_key(e) is not None]
        gold_valid = [e for e in gold_ents if entity_geo_key(e) is not None]

        psets = extract_geo_sets(pred_valid)
        gsets = extract_geo_sets(gold_valid)

        pred_state_sets.append(psets["state"])
        pred_county_sets.append(psets["county"])
        pred_city_sets.append(psets["city"])
        gold_state_sets.append(gsets["state"])
        gold_county_sets.append(gsets["county"])
        gold_city_sets.append(gsets["city"])

        p_sigs = set(entity_geo_key(e) for e in pred_valid)
        p_sigs.discard(None)
        g_sigs = set(entity_geo_key(e) for e in gold_valid)
        g_sigs.discard(None)
        pred_sig_sets.append(p_sigs)
        gold_sig_sets.append(g_sigs)

        matches = match_by_geo(pred_valid, gold_valid)
        low_ev = 0

        for (pi, gi) in matches:
            ef1 = evidence_f1_for_pair(pred_valid[pi], gold_valid[gi], evidence_match_threshold)
            if ef1 < evidence_ok_threshold:
                low_ev += 1
                pk = entity_geo_key(pred_valid[pi])
                pe = pred_valid[pi].get("evidence", pred_valid[pi].get("mention", []))
                ge = gold_valid[gi].get("evidence", gold_valid[gi].get("mention", []))
                if not isinstance(pe, list):
                    pe = [pe]
                if not isinstance(ge, list):
                    ge = [ge]
                evidence_error_rows.append({
                    "idx": i,
                    "geo_key": str(pk),
                    "evidence_f1": ef1,
                    "pred_evidence": "|".join([str(x) for x in pe]),
                    "gold_evidence": "|".join([str(x) for x in ge]),
                    "text": text,
                })

        rows.append({
            "idx": i,
            "text": text,
            "prompt": prompt,
            "pred_raw": pred_texts[i],
            "gold_raw": gold_texts[i],
            "pred_valid_json": bool(p.get("__valid_json", False)),

            "pred_state": _set_to_pipe(psets["state"]),
            "pred_county": _set_to_pipe(psets["county"]),
            "pred_city": _set_to_pipe(psets["city"]),
            "gold_state": _set_to_pipe(gsets["state"]),
            "gold_county": _set_to_pipe(gsets["county"]),
            "gold_city": _set_to_pipe(gsets["city"]),

            "state_exact": int(psets["state"] == gsets["state"]),
            "county_exact": int(psets["county"] == gsets["county"]),
            "city_exact": int(psets["city"] == gsets["city"]),
            "all_exact": int((psets["state"] == gsets["state"]) and (psets["county"] == gsets["county"]) and (psets["city"] == gsets["city"])),

            "geo_obj_exact": int(p_sigs == g_sigs),
            "geo_obj_tp": len(p_sigs & g_sigs),
            "geo_obj_fp": len(p_sigs - g_sigs),
            "geo_obj_fn": len(g_sigs - p_sigs),

            "matched_geo_pairs": len(matches),
            "low_evidence_pairs": low_ev,

            # NEW postproc stats
            "postproc_mode": str(info.get("mode", "")),
            "arrays_found": int(info.get("arrays_found", 0) or 0),
            "arrays_unique": int(info.get("arrays_unique", 0) or 0),
            "arrays_repeated_dropped": int(info.get("arrays_repeated_dropped", 0) or 0),
            "geo_dedupe_dropped": int(info.get("geo_dedupe_dropped", 0) or 0),
            "geo_conflict_kept": int(info.get("geo_conflict_kept", 0) or 0),
            "entities_before_geo_dedupe": int(info.get("entities_before_geo_dedupe", 0) or 0),
            "entities_after_geo_dedupe": int(info.get("entities_after_geo_dedupe", 0) or 0),
        })

    df_rows = pd.DataFrame(rows)
    out_rows = out_dir / f"{base}_rows.csv"
    df_rows.to_csv(out_rows, index=False)

    def _save_subset(mask, name):
        df_sub = df_rows[mask].copy()
        out_path = out_dir / f"{base}_{name}.csv"
        df_sub.to_csv(out_path, index=False)
        return out_path, len(df_sub)

    st_err_path, st_err_n = _save_subset(df_rows["state_exact"] == 0, "state_errors")
    co_err_path, co_err_n = _save_subset(df_rows["county_exact"] == 0, "county_errors")
    ci_err_path, ci_err_n = _save_subset(df_rows["city_exact"] == 0, "city_errors")
    all_err_path, all_err_n = _save_subset(df_rows["all_exact"] == 0, "full_errors")

    df_ev_err = pd.DataFrame(evidence_error_rows)
    out_ev_err = out_dir / f"{base}_evidence_errors.csv"
    df_ev_err.to_csv(out_ev_err, index=False)

    df_state_item = _entity_prf_by_item(pred_state_sets, gold_state_sets)
    df_county_item = _entity_prf_by_item(pred_county_sets, gold_county_sets)
    df_city_item = _entity_prf_by_item(pred_city_sets, gold_city_sets)

    out_state_item = out_dir / f"{base}_entity_prf_by_item_state.csv"
    out_county_item = out_dir / f"{base}_entity_prf_by_item_county.csv"
    out_city_item = out_dir / f"{base}_entity_prf_by_item_city.csv"
    df_state_item.to_csv(out_state_item, index=False)
    df_county_item.to_csv(out_county_item, index=False)
    df_city_item.to_csv(out_city_item, index=False)

    conf_paths = {}
    try:
        from sklearn.metrics import confusion_matrix  # type: ignore

        y_true_state = [_top1_label(s) for s in gold_state_sets]
        y_pred_state = [_top1_label(s) for s in pred_state_sets]
        labels_state = sorted(set(y_true_state) | set(y_pred_state))
        cm_state = confusion_matrix(y_true_state, y_pred_state, labels=labels_state)
        df_cm_state = pd.DataFrame(cm_state, index=labels_state, columns=labels_state)
        out_cm_state = out_dir / f"test_confusion_state{suffix}.csv"
        df_cm_state.to_csv(out_cm_state)
        conf_paths["state"] = out_cm_state

        def _topk_labels(g_sets: List[set], k: int) -> List[str]:
            c = Counter()
            for s in g_sets:
                for it in s:
                    c[it] += 1
            top = [it for it, _ in c.most_common(k)]
            return ["__EMPTY__"] + top

        def _map_label(x: str, allowed: set) -> str:
            return x if x in allowed else "__OTHER__"

        y_true_county = [_top1_label(s) for s in gold_county_sets]
        y_pred_county = [_top1_label(s) for s in pred_county_sets]
        labels_county = _topk_labels(gold_county_sets, top_k_labels)
        allowed_county = set(labels_county)
        y_true_county_m = [_map_label(x, allowed_county) for x in y_true_county]
        y_pred_county_m = [_map_label(x, allowed_county) for x in y_pred_county]
        labels_county_full = labels_county + ["__OTHER__"]
        cm_county = confusion_matrix(y_true_county_m, y_pred_county_m, labels=labels_county_full)
        df_cm_county = pd.DataFrame(cm_county, index=labels_county_full, columns=labels_county_full)
        out_cm_county = out_dir / f"test_confusion_county_top{top_k_labels}{suffix}.csv"
        df_cm_county.to_csv(out_cm_county)
        conf_paths["county"] = out_cm_county

        y_true_city = [_top1_label(s) for s in gold_city_sets]
        y_pred_city = [_top1_label(s) for s in pred_city_sets]
        labels_city = _topk_labels(gold_city_sets, top_k_labels)
        allowed_city = set(labels_city)
        y_true_city_m = [_map_label(x, allowed_city) for x in y_true_city]
        y_pred_city_m = [_map_label(x, allowed_city) for x in y_pred_city]
        labels_city_full = labels_city + ["__OTHER__"]
        cm_city = confusion_matrix(y_true_city_m, y_pred_city_m, labels=labels_city_full)
        df_cm_city = pd.DataFrame(cm_city, index=labels_city_full, columns=labels_city_full)
        out_cm_city = out_dir / f"test_confusion_city_top{top_k_labels}{suffix}.csv"
        df_cm_city.to_csv(out_cm_city)
        conf_paths["city"] = out_cm_city

    except Exception:
        conf_paths = {}

    missed_ev = Counter()
    halluc_ev = Counter()
    total_low_ev_pairs = int(df_rows["low_evidence_pairs"].sum())

    for i in range(len(parsed_pred)):
        pred_valid = [e for e in parsed_pred[i]["entities"] if entity_geo_key(e) is not None]
        gold_valid = [e for e in parsed_gold[i]["entities"] if entity_geo_key(e) is not None]
        matches = match_by_geo(pred_valid, gold_valid)

        for (pi, gi) in matches:
            pe = pred_valid[pi].get("evidence", pred_valid[pi].get("mention", []))
            ge = gold_valid[gi].get("evidence", gold_valid[gi].get("mention", []))
            if not isinstance(pe, list):
                pe = [pe]
            if not isinstance(ge, list):
                ge = [ge]

            preds = [p for p in (pe or []) if _norm_evidence_item(p)]
            golds = [g for g in (ge or []) if _norm_evidence_item(g)]
            if not preds and not golds:
                continue

            used = [False] * len(golds)
            matched_gold = set()
            for ptxt in preds:
                best_j = -1
                best_s = 0.0
                for j, gtxt in enumerate(golds):
                    if used[j]:
                        continue
                    s = evidence_similarity(ptxt, gtxt)
                    if s > best_s:
                        best_s = s
                        best_j = j
                if best_j >= 0 and best_s >= evidence_match_threshold:
                    used[best_j] = True
                    matched_gold.add(best_j)

            for j, gtxt in enumerate(golds):
                if j not in matched_gold:
                    missed_ev[gtxt] += 1

            for ptxt in preds:
                ok = False
                for j, gtxt in enumerate(golds):
                    if evidence_similarity(ptxt, gtxt) >= evidence_match_threshold:
                        ok = True
                        break
                if not ok:
                    halluc_ev[ptxt] += 1

    def _top(counter: Counter, n: int = 20) -> List[str]:
        return [f"{k} ({v})" for k, v in counter.most_common(n)]

    out_txt = out_dir / f"{base}_summary.txt"
    lines = []
    lines.append("ERROR ANALYSIS SUMMARY (ENTITY ARRAY + EVIDENCE)")
    lines.append("==============================================\n")
    lines.append(f"Rows: {len(df_rows):,}")
    lines.append(f"Valid JSON rate: {df_rows['pred_valid_json'].mean():.6f}\n")

    if "arrays_repeated_dropped" in df_rows.columns:
        rep_rows = int((df_rows["arrays_repeated_dropped"] > 0).sum())
        multi_rows = int((df_rows["postproc_mode"] == "multi").sum())
        lines.append("Post-processing summary:")
        lines.append(f"  Rows with repeated arrays (deduped): {rep_rows:,}")
        lines.append(f"  Rows with multiple DISTINCT arrays (merged): {multi_rows:,}")
        lines.append(f"  Total geo conflicts kept: {int(df_rows['geo_conflict_kept'].sum()):,}\n")

    lines.append("Exact-match error counts (set-based):")
    lines.append(f"  State errors:  {st_err_n:,}  (saved: {st_err_path.name})")
    lines.append(f"  County errors: {co_err_n:,}  (saved: {co_err_path.name})")
    lines.append(f"  City errors:   {ci_err_n:,}  (saved: {ci_err_path.name})")
    lines.append(f"  Full errors:   {all_err_n:,} (saved: {all_err_path.name})\n")

    lines.append("Evidence diagnostics (geo-matched pairs only):")
    lines.append(f"  Evidence error pairs (evidence_f1 < {evidence_ok_threshold}): {total_low_ev_pairs:,}")
    lines.append(f"  Evidence error rows csv: {out_ev_err.name}\n")

    lines.append("Top missed evidence phrases: " + (", ".join(_top(missed_ev, 20)) if missed_ev else "None"))
    lines.append("Top halluc evidence phrases: " + (", ".join(_top(halluc_ev, 20)) if halluc_ev else "None"))
    lines.append("")

    lines.append("Per-item PRF tables (set-based):")
    lines.append(f"  State:  {out_state_item.name}")
    lines.append(f"  County: {out_county_item.name}")
    lines.append(f"  City:   {out_city_item.name}")
    lines.append("")

    if conf_paths:
        lines.append("Confusion matrices (top-1 label per row; top-K by gold support):")
        for k in ["state", "county", "city"]:
            if k in conf_paths:
                lines.append(f"  {k}: {conf_paths[k].name}")
        lines.append("")
    else:
        lines.append("Confusion matrices: skipped (scikit-learn not installed). Install: pip install scikit-learn\n")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "rows_csv": out_rows,
        "summary_txt": out_txt,
        "evidence_errors_csv": out_ev_err,
        "state_item_csv": out_state_item,
        "county_item_csv": out_county_item,
        "city_item_csv": out_city_item,
        "conf_paths": conf_paths,
    }


# --------------------------
# Model loading
# --------------------------
def load_model_for_eval(
    model_name: str,
    hf_token: Optional[str],
    checkpoint_folder: str,
    checkpoint_path: Optional[str],
    use_lora_adapter: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_path and checkpoint_path.strip():
        best_dir = Path(checkpoint_folder) / f"{model_name}_{checkpoint_path}" / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_dir}")
        if use_lora_adapter:
            if PeftModel is None:
                raise RuntimeError("peft is not installed but --use_lora_adapter was set.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,)
            base = AutoModelForCausalLM.from_pretrained(model_name,
                                                        quantization_config=bnb_config,
                                                        device_map=None,
                                                        token=hf_token,)
            model = PeftModel.from_pretrained(base, str(best_dir))
        # if use_lora_adapter:
        #     if PeftModel is None:
        #         raise RuntimeError("peft is not installed but --use_lora_adapter was set.")
        #     base = AutoModelForCausalLM.from_pretrained(
        #         model_name, torch_dtype=torch.bfloat16, device_map=None, token=hf_token
        #     )
        #     model = PeftModel.from_pretrained(base, str(best_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(best_dir), torch_dtype=torch.bfloat16, device_map=None, token=hf_token
            )
        print(f"Loaded checkpoint: {best_dir} (use_lora_adapter={use_lora_adapter})")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=None, token=hf_token
        )
        print(f"Loaded base model: {model_name}")

    if "llama" in model_name.lower():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# --------------------------
# Eval loop
# --------------------------
def evaluate(model, loader, tokenizer, accelerator, max_new_tokens=120, log_first_batch=True):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    gen_cfg = {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens}

    all_pred, all_gold = [], []
    all_prompts = []
    all_pred_infos: List[Dict[str, Any]] = []
    all_pred_raw: List[str] = []   # NEW
    total_steps = len(loader)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            # gen_out = unwrapped.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     **gen_cfg
            # )

            cutoff = input_ids.size(1)

            # NEW: stop once first top-level JSON array closes
            stopper = StopOnFirstJsonArray(
                tokenizer=tokenizer,
                prompt_len=cutoff,
                batch_size=input_ids.size(0),
            )
            stopping = StoppingCriteriaList([stopper])

            gen_out = unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stopping_criteria=stopping,   # <-- NEW
                **gen_cfg
            )

            

            # UPDATED: keep raw + postprocessed
            local_pred: List[str] = []
            local_infos: List[Dict[str, Any]] = []
            local_raw: List[str] = []  # NEW
            for i in range(gen_out.size(0)):
                raw = tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                fixed, info = postprocess_pred_text_entity_arrays(raw)
                local_raw.append(raw)
                local_pred.append(fixed)
                local_infos.append(info)

            local_gold = batch["targets"]
            local_prompts = batch["prompts"]

            gathered_preds = gather_object(local_pred)
            gathered_golds = gather_object(local_gold)
            gathered_prompts = gather_object(local_prompts)
            gathered_infos = gather_object(local_infos)
            gathered_raw = gather_object(local_raw)  # NEW

            if accelerator.is_main_process:
                if gathered_preds and isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred.extend(sub)
                    for sub in gathered_golds:
                        all_gold.extend(sub)
                    for sub in gathered_prompts:
                        all_prompts.extend(sub)
                    for sub in gathered_infos:
                        all_pred_infos.extend(sub)
                    for sub in gathered_raw:
                        all_pred_raw.extend(sub)  # NEW
                else:
                    all_pred.extend(gathered_preds)
                    all_gold.extend(gathered_golds)
                    all_prompts.extend(gathered_prompts)
                    all_pred_infos.extend(gathered_infos)
                    all_pred_raw.extend(gathered_raw)  # NEW

                if log_first_batch and step == 0 and local_pred:
                    print("\n[Eval Sample]")
                    print("PROMPT:", batch["prompts"][0][:250].replace("\n", " \\n "))
                    print("RAW   :", local_raw[0])
                    print("PRED  :", local_pred[0])
                    print("GOLD  :", local_gold[0])
                    print("POSTP :", local_infos[0])

                if step % 200 == 0 or step == total_steps - 1:
                    accelerator.print(f"Step {step+1}/{total_steps}")

    accelerator.wait_for_everyone()
    return all_prompts, all_pred, all_gold, all_pred_infos, all_pred_raw  # NEW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)

    p.add_argument("--checkpoint_folder", default="checkpoints")
    p.add_argument("--checkpoint_path", default=None, help="TIMESTAMP used in training output dir")
    p.add_argument("--use_lora_adapter", action="store_true")

    p.add_argument("--test_csv", required=True)
    p.add_argument("--prompt_col", default="prompt")
    p.add_argument("--target_col", default="target_json")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=120)

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_note", default=None)

    p.add_argument("--error_analysis", action="store_true", help="Write error analysis CSV/TXT files.")
    p.add_argument("--confusion_top_k", type=int, default=50, help="Top-K labels for county/city confusion matrices (requires sklearn).")

    p.add_argument("--evidence_match_threshold", type=float, default=0.75)
    p.add_argument("--evidence_ok_threshold", type=float, default=0.50)

    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    ds = PromptTargetCSVDataset(args.test_csv, prompt_col=args.prompt_col, target_col=args.target_col)

    model, tokenizer = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
        use_lora_adapter=args.use_lora_adapter,
    )

    collator = EvalCollatorTokenize(tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    model, loader = accelerator.prepare(model, loader)

    base_out_dir = Path(args.checkpoint_folder) / (
        f"{args.model_name}_{args.checkpoint_path}/best" if args.checkpoint_path else f"{args.model_name}_base"
    )

    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_stem = Path(args.test_csv).stem
    out_dir = base_out_dir / f"{test_stem}_{ts_run}"

    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            run_name = f"{args.model_name}_{args.checkpoint_path or 'base'}_{args.run_note or 'eval'}_{test_stem}_{ts_run}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    all_prompts, all_pred, all_gold, all_pred_infos, all_pred_raw = evaluate(model, loader, tokenizer, accelerator, max_new_tokens=args.max_new_tokens)

    if accelerator.is_main_process:
        metrics = compute_test_metrics_entities_json(
            all_pred,
            all_gold,
            evidence_match_threshold=args.evidence_match_threshold,
            evidence_ok_threshold=args.evidence_ok_threshold,
        )

        print("\n=== METRICS ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        fname = f"test_metrics_{args.run_note}.json" if args.run_note else "test_metrics.json"
        fpath = out_dir / fname
        if fpath.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = out_dir / (fname.replace(".json", f"_{ts}.json"))

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved:", fpath)

        # NEW: save all predictions with postproc columns
        rows_pred = []
        for i in range(len(all_pred)):
            info = (all_pred_infos[i] if i < len(all_pred_infos) else {}) or {}
            rows_pred.append({
                "idx": i,
                "prompt": all_prompts[i] if i < len(all_prompts) else "",
                "gold_raw": all_gold[i] if i < len(all_gold) else "",
                "pred_raw": "",  # NOTE: we no longer keep raw in all_pred; raw is only inside evaluate
                "pred_postproc_raw": all_pred[i],

                "postproc_mode": info.get("mode", ""),
                "arrays_found": int(info.get("arrays_found", 0) or 0),
                "arrays_unique": int(info.get("arrays_unique", 0) or 0),
                "arrays_repeated_dropped": int(info.get("arrays_repeated_dropped", 0) or 0),
                "geo_dedupe_dropped": int(info.get("geo_dedupe_dropped", 0) or 0),
                "geo_conflict_kept": int(info.get("geo_conflict_kept", 0) or 0),
                "entities_before_geo_dedupe": int(info.get("entities_before_geo_dedupe", 0) or 0),
                "entities_after_geo_dedupe": int(info.get("entities_after_geo_dedupe", 0) or 0),
            })
        pd.DataFrame(rows_pred).to_csv(out_dir / "all_predictions.csv", index=False)
        print("Saved:", out_dir / "all_predictions.csv")

        if args.error_analysis:
            artifacts = run_error_analysis(
                out_dir=out_dir,
                run_note=args.run_note,
                prompts=all_prompts,
                pred_texts=all_pred,
                gold_texts=all_gold,
                pred_infos=all_pred_infos,
                evidence_match_threshold=args.evidence_match_threshold,
                evidence_ok_threshold=args.evidence_ok_threshold,
                top_k_labels=args.confusion_top_k,
            )
            print("Error analysis saved:")
            print("  -", artifacts["rows_csv"])
            print("  -", artifacts["evidence_errors_csv"])
            print("  -", artifacts["summary_txt"])

        if args.wandb_project and wandb_key:
            wandb.log({f"test/{k}": v for k, v in metrics.items()})
            wandb.finish()

        print("All results saved under:", out_dir)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

