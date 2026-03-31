#!/usr/bin/env python3
"""
Geo-only evaluation (normal / non-4bit) with geo-only error analysis.

This version:
  - rewrites evidence-based prompts into geo-only prompts on the fly
  - strips evidence / mention / extra keys from gold target_json
  - strips evidence / mention / extra keys from predictions before scoring
  - supports base model, full FT checkpoint, and LoRA adapter checkpoint
  - writes geo-only error analysis outputs

Outputs under:
  <base_out_dir>/<test_csv_stem>_<timestamp>/
    - metrics.json
    - all_predictions.csv
    - test_error_analysis_rows.csv
    - test_error_analysis_state_errors.csv
    - test_error_analysis_county_errors.csv
    - test_error_analysis_city_errors.csv
    - test_error_analysis_full_errors.csv
    - test_error_analysis_entity_prf_by_item_state.csv
    - test_error_analysis_entity_prf_by_item_county.csv
    - test_error_analysis_entity_prf_by_item_city.csv
    - test_error_analysis_summary.txt
"""

import os
import re
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

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from dotenv import load_dotenv
import wandb

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# --------------------------
# Prompt rewriting helpers
# --------------------------
GEO_ONLY_PROMPT_TEMPLATE = """You are given a social media post.

Extract all geographic locations mentioned in the text and resolve each to a hierarchical location.

Return a JSON array of objects. Each object may contain:
  - "state": U.S. state abbreviation
  - "county": county name (omit the word "County"), if applicable
  - "city": city name, if applicable

Rules:
  - One object per resolved location.
  - If "city" is included, "county" and "state" must also be included.
  - If only county-level information is available, include "state" and "county".
  - Do not fabricate locations.
  - Your response must start with [ and end with ].
  - Do not include any explanation, markdown, or extra text.

Text: {text}
Output:"""


def build_geo_only_prompt_from_text(post_text: str) -> str:
    return GEO_ONLY_PROMPT_TEMPLATE.format(text=post_text)


def extract_last_text_block(prompt: str) -> Optional[str]:
    if prompt is None:
        return None

    s = str(prompt)

    matches = list(re.finditer(r"Text:\s*(.*?)(?:\n\s*Output:|\Z)", s, flags=re.DOTALL))
    if matches:
        txt = matches[-1].group(1).strip()
        txt = txt.strip('"').strip()
        if txt:
            return txt

    idx = s.rfind("Text:")
    if idx != -1:
        txt = s[idx + len("Text:"):].strip()
        txt = txt.strip('"').strip()
        if txt:
            return txt

    return None


def maybe_rewrite_prompt_to_geo_only(prompt: str) -> str:
    if prompt is None:
        return ""

    s = str(prompt).strip()
    if not s:
        return s

    lower_s = s.lower()
    if '"evidence"' not in lower_s and "evidence" not in lower_s and '"mention"' not in lower_s and "mention" not in lower_s:
        return s

    post_text = extract_last_text_block(s)
    if post_text:
        return build_geo_only_prompt_from_text(post_text)

    return s


# --------------------------
# JSON helpers
# --------------------------
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


def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


# --------------------------
# Geo-only stripping helpers
# --------------------------
def strip_to_geo_only_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue

        out: Dict[str, Any] = {}
        if e.get("state") is not None:
            out["state"] = e.get("state")
        if e.get("county") is not None:
            out["county"] = e.get("county")
        if e.get("city") is not None:
            out["city"] = e.get("city")

        cleaned.append(out)
    return cleaned


def target_json_to_geo_only_json(raw: str) -> str:
    if raw is None:
        return "[]"

    obj, ok = _try_json_load(str(raw).strip())
    if not ok or not isinstance(obj, list):
        return "[]"

    ents = [x for x in obj if isinstance(x, dict)]
    ents = strip_to_geo_only_entities(ents)
    return json.dumps(ents, ensure_ascii=False)


# --------------------------
# Dataset
# --------------------------
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
            if not p or not t:
                continue

            p_geo = maybe_rewrite_prompt_to_geo_only(p)
            t_geo = target_json_to_geo_only_json(t)
            self.examples.append({"prompt": p_geo, "target": t_geo})

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
# Stop generation at first JSON array
# --------------------------
class StopOnFirstJsonArray(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, batch_size: int):
        super().__init__()
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)

        self.done = [False] * batch_size
        self.started = [False] * batch_size
        self.depth = [0] * batch_size
        self.in_str = [False] * batch_size
        self.esc = [False] * batch_size
        self.last_len = [self.prompt_len] * batch_size

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
                        return

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        bsz, seq_len = input_ids.shape
        for i in range(bsz):
            if self.done[i]:
                continue
            prev = self.last_len[i]
            if seq_len <= prev:
                continue
            new_ids = input_ids[i, prev:seq_len].tolist()
            self.last_len[i] = seq_len
            text = self.tok.decode(new_ids, skip_special_tokens=False)
            if text:
                self._consume_text(i, text)
        return all(self.done)


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
    s = "".join(s.split())
    return s or None


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
    ents = strip_to_geo_only_entities(ents)
    return {"__valid_json": True, "entities": ents}


# --------------------------
# Geo-only postprocessing
# --------------------------
def dedupe_geo_only_entities(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {"geo_dedupe_dropped": 0}
    kept: List[Dict[str, Any]] = []
    seen = set()

    for e in entities:
        if not isinstance(e, dict):
            continue
        gk = entity_geo_key(e)
        if gk is None:
            kept.append(e)
            continue

        if gk in seen:
            stats["geo_dedupe_dropped"] += 1
            continue

        seen.add(gk)
        kept.append(e)

    return kept, stats


def postprocess_pred_text_geo_only(raw: str) -> Tuple[str, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "arrays_found": 0,
        "arrays_unique": 0,
        "arrays_repeated_dropped": 0,
        "mode": "none",
        "entities_before_geo_dedupe": 0,
        "entities_after_geo_dedupe": 0,
        "geo_dedupe_dropped": 0,
    }

    if raw is None:
        return "[]", info

    arrays = extract_all_json_arrays_balanced(raw, max_arrays=25)
    info["arrays_found"] = len(arrays)

    if not arrays:
        return "[]", info

    canon_to_ents: Dict[str, List[Dict[str, Any]]] = {}
    for a in arrays:
        try:
            obj = json.loads(a)
        except Exception:
            continue
        if not isinstance(obj, list):
            continue

        ents = [x for x in obj if isinstance(x, dict)]
        ents = strip_to_geo_only_entities(ents)
        canon = _canonical_json(ents)
        if canon not in canon_to_ents:
            canon_to_ents[canon] = ents

    unique_canons = list(canon_to_ents.keys())
    info["arrays_unique"] = len(unique_canons)
    info["arrays_repeated_dropped"] = max(0, info["arrays_found"] - info["arrays_unique"])

    if info["arrays_unique"] == 0:
        return "[]", info

    if info["arrays_unique"] == 1:
        info["mode"] = "repetition" if info["arrays_found"] > 1 else "single"
    else:
        info["mode"] = "multi"

    merged: List[Dict[str, Any]] = []
    for c in unique_canons:
        merged.extend(canon_to_ents[c])

    info["entities_before_geo_dedupe"] = len(merged)

    merged2, geo_stats = dedupe_geo_only_entities(merged)
    info.update(geo_stats)
    info["entities_after_geo_dedupe"] = len(merged2)

    return json.dumps(merged2, ensure_ascii=False), info


# --------------------------
# Metrics
# --------------------------
def _micro_prf_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_test_metrics_geo_only(pred_texts: List[str], gold_texts: List[str]) -> Dict[str, float]:
    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    json_valid_rate = sum(1 for p in parsed_pred if p["__valid_json"]) / max(1, len(parsed_pred))

    pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
    gold_state_sets, gold_county_sets, gold_city_sets = [], [], []
    pred_sig_sets, gold_sig_sets = [], []

    abs_err_sum = 0.0
    bias_sum = 0.0
    exact_count_hits = 0.0
    pred_count_sum = 0.0
    gold_count_sum = 0.0

    for p, g in zip(parsed_pred, parsed_gold):
        pred_ents = [e for e in p["entities"] if entity_geo_key(e) is not None]
        gold_ents = [e for e in g["entities"] if entity_geo_key(e) is not None]

        psets = extract_geo_sets(pred_ents)
        gsets = extract_geo_sets(gold_ents)

        pred_state_sets.append(psets["state"])
        pred_county_sets.append(psets["county"])
        pred_city_sets.append(psets["city"])

        gold_state_sets.append(gsets["state"])
        gold_county_sets.append(gsets["county"])
        gold_city_sets.append(gsets["city"])

        p_sigs = set(entity_geo_key(e) for e in pred_ents)
        p_sigs.discard(None)
        g_sigs = set(entity_geo_key(e) for e in gold_ents)
        g_sigs.discard(None)

        pred_sig_sets.append(p_sigs)
        gold_sig_sets.append(g_sigs)

        n_pred = len(p_sigs)
        n_gold = len(g_sigs)
        pred_count_sum += n_pred
        gold_count_sum += n_gold
        abs_err_sum += abs(n_pred - n_gold)
        bias_sum += (n_pred - n_gold)
        if n_pred == n_gold:
            exact_count_hits += 1.0

    def micro_prf_from_sets(pred_sets: List[set], gold_sets: List[set]) -> Dict[str, float]:
        tp = fp = fn = 0.0
        for pset, gset in zip(pred_sets, gold_sets):
            tp += len(pset & gset)
            fp += len(pset - gset)
            fn += len(gset - pset)
        return _micro_prf_counts(tp, fp, fn)

    state_prf = micro_prf_from_sets(pred_state_sets, gold_state_sets)
    county_prf = micro_prf_from_sets(pred_county_sets, gold_county_sets)
    city_prf = micro_prf_from_sets(pred_city_sets, gold_city_sets)

    geo_tp = geo_fp = geo_fn = 0.0
    geo_em_hits = 0.0
    for p_sigs, g_sigs in zip(pred_sig_sets, gold_sig_sets):
        geo_tp += len(p_sigs & g_sigs)
        geo_fp += len(p_sigs - g_sigs)
        geo_fn += len(g_sigs - p_sigs)
        if p_sigs == g_sigs:
            geo_em_hits += 1.0

    geo_prf = _micro_prf_counts(geo_tp, geo_fp, geo_fn)
    geo_exact_match = geo_em_hits / max(1.0, len(pred_sig_sets))

    n = max(1.0, len(pred_sig_sets))
    entity_count_mae = abs_err_sum / n
    entity_count_bias = bias_sum / n
    entity_count_exact_match_rate = exact_count_hits / n
    pred_entity_avg = pred_count_sum / n
    gold_entity_avg = gold_count_sum / n

    return {
        "json_valid_rate": float(json_valid_rate),
        "state_f1": float(state_prf["f1"]),
        "county_f1": float(county_prf["f1"]),
        "city_f1": float(city_prf["f1"]),
        "geo_obj_precision": float(geo_prf["precision"]),
        "geo_obj_recall": float(geo_prf["recall"]),
        "geo_obj_f1": float(geo_prf["f1"]),
        "geo_obj_exact_match": float(geo_exact_match),
        "entity_count_mae": float(entity_count_mae),
        "entity_count_bias": float(entity_count_bias),
        "entity_count_exact_match": float(entity_count_exact_match_rate),
        "pred_entity_avg": float(pred_entity_avg),
        "gold_entity_avg": float(gold_entity_avg),
    }


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


def run_error_analysis(
    out_dir: Path,
    run_note: Optional[str],
    prompts: List[str],
    pred_texts: List[str],
    gold_texts: List[str],
    pred_infos: Optional[List[Dict[str, Any]]] = None,
):
    suffix = f"_{run_note}" if run_note else ""
    base = f"test_error_analysis{suffix}"

    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
    gold_state_sets, gold_county_sets, gold_city_sets = [], [], []
    pred_sig_sets, gold_sig_sets = [], []

    rows = []
    state_rows = []
    county_rows = []
    city_rows = []
    full_rows = []

    for i in range(len(parsed_pred)):
        p = parsed_pred[i]
        g = parsed_gold[i]
        prompt = prompts[i] if i < len(prompts) else ""
        text = _safe_extract_text_from_prompt(prompt)
        info = (pred_infos[i] if (pred_infos is not None and i < len(pred_infos)) else {}) or {}

        pred_ents = [e for e in p["entities"] if entity_geo_key(e) is not None]
        gold_ents = [e for e in g["entities"] if entity_geo_key(e) is not None]

        psets = extract_geo_sets(pred_ents)
        gsets = extract_geo_sets(gold_ents)

        pred_state_sets.append(psets["state"])
        pred_county_sets.append(psets["county"])
        pred_city_sets.append(psets["city"])
        gold_state_sets.append(gsets["state"])
        gold_county_sets.append(gsets["county"])
        gold_city_sets.append(gsets["city"])

        p_sigs = set(entity_geo_key(e) for e in pred_ents)
        p_sigs.discard(None)
        g_sigs = set(entity_geo_key(e) for e in gold_ents)
        g_sigs.discard(None)
        pred_sig_sets.append(p_sigs)
        gold_sig_sets.append(g_sigs)

        state_fp = sorted(psets["state"] - gsets["state"])
        state_fn = sorted(gsets["state"] - psets["state"])
        county_fp = sorted(psets["county"] - gsets["county"])
        county_fn = sorted(gsets["county"] - psets["county"])
        city_fp = sorted(psets["city"] - gsets["city"])
        city_fn = sorted(gsets["city"] - psets["city"])
        full_fp = sorted(p_sigs - g_sigs)
        full_fn = sorted(g_sigs - p_sigs)

        row = {
            "row_id": i,
            "text": text,
            "prompt": prompt,
            "gold_json": gold_texts[i],
            "pred_json": pred_texts[i],
            "pred_valid_json": p["__valid_json"],
            "gold_states": _set_to_pipe(gsets["state"]),
            "pred_states": _set_to_pipe(psets["state"]),
            "gold_counties": _set_to_pipe(gsets["county"]),
            "pred_counties": _set_to_pipe(psets["county"]),
            "gold_cities": _set_to_pipe(gsets["city"]),
            "pred_cities": _set_to_pipe(psets["city"]),
            "gold_geo_objs": "|".join([str(x) for x in sorted(g_sigs)]),
            "pred_geo_objs": "|".join([str(x) for x in sorted(p_sigs)]),
            "state_fp": "|".join(state_fp),
            "state_fn": "|".join(state_fn),
            "county_fp": "|".join(county_fp),
            "county_fn": "|".join(county_fn),
            "city_fp": "|".join(city_fp),
            "city_fn": "|".join(city_fn),
            "full_fp": "|".join([str(x) for x in full_fp]),
            "full_fn": "|".join([str(x) for x in full_fn]),
            "arrays_found": info.get("arrays_found", 0),
            "arrays_unique": info.get("arrays_unique", 0),
            "arrays_repeated_dropped": info.get("arrays_repeated_dropped", 0),
            "postproc_mode": info.get("mode", ""),
            "entities_before_geo_dedupe": info.get("entities_before_geo_dedupe", 0),
            "entities_after_geo_dedupe": info.get("entities_after_geo_dedupe", 0),
            "geo_dedupe_dropped": info.get("geo_dedupe_dropped", 0),
        }
        rows.append(row)

        if state_fp or state_fn:
            state_rows.append(row)
        if county_fp or county_fn:
            county_rows.append(row)
        if city_fp or city_fn:
            city_rows.append(row)
        if full_fp or full_fn:
            full_rows.append(row)

    df_rows = pd.DataFrame(rows)
    df_state = pd.DataFrame(state_rows)
    df_county = pd.DataFrame(county_rows)
    df_city = pd.DataFrame(city_rows)
    df_full = pd.DataFrame(full_rows)

    df_rows.to_csv(out_dir / f"{base}_rows.csv", index=False)
    df_state.to_csv(out_dir / f"{base}_state_errors.csv", index=False)
    df_county.to_csv(out_dir / f"{base}_county_errors.csv", index=False)
    df_city.to_csv(out_dir / f"{base}_city_errors.csv", index=False)
    df_full.to_csv(out_dir / f"{base}_full_errors.csv", index=False)

    _entity_prf_by_item(pred_state_sets, gold_state_sets).to_csv(
        out_dir / f"{base}_entity_prf_by_item_state.csv", index=False
    )
    _entity_prf_by_item(pred_county_sets, gold_county_sets).to_csv(
        out_dir / f"{base}_entity_prf_by_item_county.csv", index=False
    )
    _entity_prf_by_item(pred_city_sets, gold_city_sets).to_csv(
        out_dir / f"{base}_entity_prf_by_item_city.csv", index=False
    )

    summary_lines = [
        f"rows_total: {len(rows)}",
        f"rows_with_state_error: {len(state_rows)}",
        f"rows_with_county_error: {len(county_rows)}",
        f"rows_with_city_error: {len(city_rows)}",
        f"rows_with_full_geo_error: {len(full_rows)}",
    ]
    with open(out_dir / f"{base}_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


# --------------------------
# Model loading
# --------------------------
def load_model_for_eval(
    model_name: str,
    hf_token: Optional[str],
    checkpoint_folder: str,
    checkpoint_path: Optional[str],
    use_lora_adapter: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_path:
        ckpt_dir = Path(checkpoint_folder) / f"{model_name}_{checkpoint_path}" / "best"
    else:
        ckpt_dir = None

    if use_lora_adapter:
        if PeftModel is None:
            raise RuntimeError("peft not available. Install: pip install peft")
        if ckpt_dir is None:
            raise ValueError("--use_lora_adapter requires --checkpoint_path")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            token=hf_token,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    else:
        model_source = str(ckpt_dir) if ckpt_dir is not None else model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map=None,
            token=hf_token,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = True

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


# --------------------------
# Eval loop
# --------------------------
def evaluate(model, loader, tokenizer, accelerator, max_new_tokens=120, log_first_batch=True):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    gen_cfg = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    all_pred, all_gold = [], []
    all_prompts = []
    all_pred_infos: List[Dict[str, Any]] = []
    all_pred_raw: List[str] = []

    for step, batch in enumerate(loader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            stopper = StopOnFirstJsonArray(
                tokenizer=tokenizer,
                prompt_len=input_ids.size(1),
                batch_size=input_ids.size(0),
            )

            gen_out = unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stopping_criteria=StoppingCriteriaList([stopper]),
                **gen_cfg,
            )

            cutoff = input_ids.size(1)

            local_prompts = batch["prompts"]
            local_golds = batch["targets"]
            local_pred_raw = []
            local_pred = []
            local_infos = []

            for i in range(gen_out.size(0)):
                raw = tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                fixed, info = postprocess_pred_text_geo_only(raw)
                local_pred_raw.append(raw)
                local_pred.append(fixed)
                local_infos.append(info)

            gp = gather_object(local_prompts)
            gg = gather_object(local_golds)
            gr = gather_object(local_pred_raw)
            gpred = gather_object(local_pred)
            ginfo = gather_object(local_infos)

            if accelerator.is_main_process:
                if gp and isinstance(gp[0], list):
                    for x in gp:
                        all_prompts.extend(x)
                    for x in gg:
                        all_gold.extend(x)
                    for x in gr:
                        all_pred_raw.extend(x)
                    for x in gpred:
                        all_pred.extend(x)
                    for x in ginfo:
                        all_pred_infos.extend(x)
                else:
                    all_prompts.extend(gp)
                    all_gold.extend(gg)
                    all_pred_raw.extend(gr)
                    all_pred.extend(gpred)
                    all_pred_infos.extend(ginfo)

                if log_first_batch and step == 0 and local_pred:
                    print("\n[Eval sample]")
                    print("PROMPT:", local_prompts[0][:250].replace("\n", " \\n "))
                    print("PRED  :", local_pred[0])
                    print("GOLD  :", local_golds[0])

    return all_prompts, all_pred, all_gold, all_pred_infos, all_pred_raw


# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser(description="Geo-only evaluation with geo-only error analysis")

    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint_folder", default="checkpoints")
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--use_lora_adapter", action="store_true")

    p.add_argument("--test_csv", required=True)
    p.add_argument("--prompt_col", default="prompt")
    p.add_argument("--target_col", default="target_json")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=120)

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_note", default=None)
    p.add_argument("--run_error_analysis", action="store_true")

    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    ds = PromptTargetCSVDataset(
        args.test_csv,
        prompt_col=args.prompt_col,
        target_col=args.target_col,
    )

    model, tokenizer = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
        use_lora_adapter=args.use_lora_adapter,
    )

    collator = EvalCollatorTokenize(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

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

    all_prompts, all_pred, all_gold, all_pred_infos, all_pred_raw = evaluate(
        model,
        loader,
        tokenizer,
        accelerator,
        max_new_tokens=args.max_new_tokens,
        log_first_batch=True,
    )

    if accelerator.is_main_process:
        metrics = compute_test_metrics_geo_only(all_pred, all_gold)

        print("\n=== METRICS ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        rows = []
        for i in range(len(all_pred)):
            info = all_pred_infos[i] if i < len(all_pred_infos) else {}
            rows.append({
                "row_id": i,
                "prompt": all_prompts[i] if i < len(all_prompts) else "",
                "gold": all_gold[i] if i < len(all_gold) else "",
                "pred_raw": all_pred_raw[i] if i < len(all_pred_raw) else "",
                "pred_postprocessed": all_pred[i],
                "pred_valid_json": parse_entity_array(all_pred[i])["__valid_json"],
                "arrays_found": info.get("arrays_found", 0),
                "arrays_unique": info.get("arrays_unique", 0),
                "arrays_repeated_dropped": info.get("arrays_repeated_dropped", 0),
                "postproc_mode": info.get("mode", ""),
                "entities_before_geo_dedupe": info.get("entities_before_geo_dedupe", 0),
                "entities_after_geo_dedupe": info.get("entities_after_geo_dedupe", 0),
                "geo_dedupe_dropped": info.get("geo_dedupe_dropped", 0),
            })

        pd.DataFrame(rows).to_csv(out_dir / "all_predictions.csv", index=False)

        if args.run_error_analysis:
            run_error_analysis(
                out_dir=out_dir,
                run_note=args.run_note,
                prompts=all_prompts,
                pred_texts=all_pred,
                gold_texts=all_gold,
                pred_infos=all_pred_infos,
            )

        if args.wandb_project and wandb_key:
            wandb.log(metrics)
            wandb.finish()


if __name__ == "__main__":
    main()
