#!/usr/bin/env python3
"""
Instruction -> JSON entity-array training (GEO-ONLY from existing evidence-based CSVs).

This version:
  - keeps the same input CSVs
  - strips evidence / mention / extra keys on the fly
  - rewrites evidence-based prompts into geo-only prompts on the fly
  - trains only on geo entity detection + hierarchical resolution
  - validates only on state / county / city
  - does NOT require changing prompt or target_json files beforehand

Input CSVs (train/val) must include:
  - prompt: str
  - target_json: str   (JSON string, ARRAY of objects)

Original target_json may contain:
[
  {"state":"TX","county":"Travis","city":"Austin","evidence":["..."],"mention":["..."]},
  ...
]

This script converts targets on the fly to:
[
  {"state":"TX","county":"Travis","city":"Austin"},
  ...
]

Supports:
  - full fine-tune
  - LoRA (PEFT)
Optional:
  - FSDP
  - gradient checkpointing
"""

import os
import re
import json
import math
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import gather_object

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from dotenv import load_dotenv
import wandb

# Optional PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Safe left padding helper
# --------------------------
def left_pad_sequence(seq_list: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    """
    Left-pad a list of 1D tensors in a PyTorch-version-safe way.
    """
    flipped = [x.flip(0) for x in seq_list]
    padded = pad_sequence(flipped, batch_first=True, padding_value=padding_value)
    return padded.flip(1)


# --------------------------
# JSON helpers
# --------------------------
def extract_all_json_arrays_balanced(text: str, max_arrays: int = 25) -> List[str]:
    """
    Extract ALL balanced JSON array substrings from text, handling strings/escapes.
    Returns list of substrings like '[{...}, {...}]'.
    """
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
    """
    Keep only state/county/city.
    Drop evidence, mention, and any extra keys.
    """
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
    """
    Convert raw target_json string to geo-only JSON string.
    If parsing fails, return [].
    """
    if raw is None:
        return "[]"

    obj, ok = _try_json_load(str(raw).strip())
    if not ok or not isinstance(obj, list):
        return "[]"

    ents = [x for x in obj if isinstance(x, dict)]
    ents = strip_to_geo_only_entities(ents)
    return json.dumps(ents, ensure_ascii=False)


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
    """
    Try to recover the actual social media post from an evidence-based instruction prompt.
    This looks for the final 'Text: ...' block.
    """
    if prompt is None:
        return None

    s = str(prompt)

    # Most direct: grab the last Text: ... section.
    matches = list(re.finditer(r"Text:\s*(.*?)(?:\n\s*Output:|\Z)", s, flags=re.DOTALL))
    if matches:
        txt = matches[-1].group(1).strip()
        txt = txt.strip('"').strip()
        if txt:
            return txt

    # Fallback: if no explicit Output: marker, use everything after the last "Text:"
    idx = s.rfind("Text:")
    if idx != -1:
        txt = s[idx + len("Text:"):].strip()
        txt = txt.strip('"').strip()
        if txt:
            return txt

    return None


def maybe_rewrite_prompt_to_geo_only(prompt: str) -> str:
    """
    Rewrite an evidence-based prompt into a geo-only prompt while preserving the actual post text.

    If the prompt already looks geo-only or we cannot recover the text safely,
    return the original prompt unchanged.
    """
    if prompt is None:
        return ""

    s = str(prompt).strip()
    if not s:
        return s

    # If it already does not mention evidence, leave it alone.
    lower_s = s.lower()
    if '"evidence"' not in lower_s and "evidence" not in lower_s and '"mention"' not in lower_s and "mention" not in lower_s:
        return s

    post_text = extract_last_text_block(s)
    if post_text:
        return build_geo_only_prompt_from_text(post_text)

    return s


# --------------------------
# Load model + tokenizer
# --------------------------
def load_model_and_tokenizer(
    model_name: str,
    hf_token: Optional[str],
    using_accelerator: bool,
    checkpoint_path: Optional[str],
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_source = checkpoint_path if checkpoint_path else model_name

    extra_kwargs = {}
    if "gemma" in model_name.lower():
        extra_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map=None if using_accelerator else "auto",
        token=hf_token,
        **extra_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# --------------------------
# LoRA helper
# --------------------------
def maybe_apply_lora(model, args):
    if args.tune_method != "lora":
        return model

    if get_peft_model is None or LoraConfig is None:
        raise RuntimeError("peft not available. Install: pip install peft")

    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model


# --------------------------
# Dataset
# --------------------------
class PromptTargetCSVDataset(Dataset):
    """
    CSV must have columns: prompt, target_json

    prompt may be evidence-based; this dataset rewrites it to geo-only.
    target_json can contain evidence/mention/etc.; this dataset strips them on the fly.
    """
    def __init__(self, csv_path: str, prompt_col="prompt", target_col="target_json"):
        self.csv_path = str(csv_path)
        df = pd.read_csv(self.csv_path, dtype=str)

        if prompt_col not in df.columns or target_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{prompt_col}' and '{target_col}'. "
                f"Found: {list(df.columns)}"
            )

        prompts = df[prompt_col].fillna("").astype(str).tolist()
        targets = df[target_col].fillna("").astype(str).tolist()

        examples = []
        for p, t in zip(prompts, targets):
            p = p.strip()
            t = t.strip()
            if not p or not t:
                continue

            p_geo = maybe_rewrite_prompt_to_geo_only(p)
            t_geo = target_json_to_geo_only_json(t)
            examples.append({"prompt": p_geo, "target": t_geo})

        if not examples:
            raise ValueError(f"No valid (prompt,target) rows found in {csv_path}")

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --------------------------
# Collators
# --------------------------
class TrainCollatorTokenize:
    """
    Train:
      input = prompt + target
      labels = -100 for prompt tokens, loss only on target tokens.
    """
    def __init__(self, tokenizer, max_length=512):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]):
        input_ids_list = []
        labels_list = []

        for ex in batch:
            prompt = ex["prompt"]
            target = ex["target"]

            prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
            target_ids = self.tok(target, add_special_tokens=False).input_ids

            # Boundary-preserving truncation
            if len(prompt_ids) + len(target_ids) > self.max_length:
                if len(target_ids) >= self.max_length:
                    target_ids = target_ids[-self.max_length:]
                    prompt_ids = []
                else:
                    keep_prompt = self.max_length - len(target_ids)
                    prompt_ids = prompt_ids[-keep_prompt:]

            ids = prompt_ids + target_ids
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + target_ids

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = left_pad_sequence(input_ids_list, self.tok.pad_token_id)
        labels = left_pad_sequence(labels_list, -100)
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class EvalCollatorTokenize:
    """
    Eval:
      input = prompt only
      gold target kept as geo-only JSON string
    """
    def __init__(self, tokenizer, max_length=512):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]):
        prompt_ids_list = []
        prompts = []
        targets = []

        for ex in batch:
            prompt = ex["prompt"]
            target = ex["target"]

            ids = self.tok(prompt, add_special_tokens=False).input_ids
            if len(ids) > self.max_length:
                ids = ids[-self.max_length:]

            prompt_ids_list.append(torch.tensor(ids, dtype=torch.long))
            prompts.append(prompt)
            targets.append(target)

        input_ids = left_pad_sequence(prompt_ids_list, self.tok.pad_token_id)
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
            "targets": targets,
        }


# --------------------------
# Normalization
# --------------------------
def _norm_admin(x: Any) -> Optional[str]:
    """
    Strict normalization for state/county/city:
      - lowercase
      - strip
      - remove " county"
      - remove all internal spaces
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = s.replace(" county", "").strip()
    s = "".join(s.split())
    return s or None


# --------------------------
# Parsing entity array JSON
# --------------------------
def parse_entity_array(text: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "__valid_json": bool,
        "entities": List[dict]   # geo-only entities
      }
    """
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


def entity_geo_key(ent: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    Geo key for matching:
      - if city present -> (state, county, city)
      - else -> (state, county, "")
    Requires at least state+county to be valid.
    """
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
# Geo-only dedupe / postprocess
# --------------------------
def dedupe_geo_only_entities(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Deduplicate entities by normalized geo key only.
    """
    kept: List[Dict[str, Any]] = []
    seen = set()
    stats = {"geo_dedup_drops": 0.0}

    for ent in entities:
        gk = entity_geo_key(ent)
        if gk is None:
            kept.append(ent)
            continue
        if gk in seen:
            stats["geo_dedup_drops"] += 1.0
            continue
        seen.add(gk)
        kept.append(ent)

    return kept, stats


def postprocess_pred_text_geo_only(raw: str) -> Tuple[str, Dict[str, Any]]:
    """
    - Extract ALL JSON arrays
    - Parse arrays
    - Keep only dict items
    - Strip to geo-only fields
    - Deduplicate IDENTICAL arrays
    - Merge distinct arrays
    - Deduplicate final entities by geo key
    Returns:
      (final_json_array_string, info)
    """
    info: Dict[str, Any] = {
        "found_arrays": 0,
        "parsed_arrays": 0,
        "distinct_arrays": 0,
        "repeated_arrays": 0,
        "had_multi_arrays": False,
        "had_multi_distinct": False,
        "geo_dedup_drops": 0,
    }

    if raw is None:
        return "[]", info

    arrays = extract_all_json_arrays_balanced(raw)
    info["found_arrays"] = len(arrays)
    info["had_multi_arrays"] = len(arrays) > 1

    if not arrays:
        return "[]", info

    parsed_lists: List[List[Dict[str, Any]]] = []
    canon_keys: List[str] = []

    for a in arrays:
        try:
            obj = json.loads(a)
        except Exception:
            continue

        if not isinstance(obj, list):
            continue

        ents = [x for x in obj if isinstance(x, dict)]
        ents = strip_to_geo_only_entities(ents)

        parsed_lists.append(ents)
        canon_keys.append(_canonical_json(ents))

    info["parsed_arrays"] = len(parsed_lists)
    if not parsed_lists:
        return "[]", info

    seen = set()
    distinct_lists: List[List[Dict[str, Any]]] = []
    for k, ents in zip(canon_keys, parsed_lists):
        if k in seen:
            info["repeated_arrays"] += 1
            continue
        seen.add(k)
        distinct_lists.append(ents)

    info["distinct_arrays"] = len(distinct_lists)
    info["had_multi_distinct"] = info["distinct_arrays"] > 1

    merged: List[Dict[str, Any]] = []
    for ents in distinct_lists:
        merged.extend(ents)

    merged2, geo_stats = dedupe_geo_only_entities(merged)
    info["geo_dedup_drops"] = int(geo_stats.get("geo_dedup_drops", 0) or 0)

    return json.dumps(merged2, ensure_ascii=False), info


# --------------------------
# Metrics helpers
# --------------------------
def _micro_prf_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_validation_metrics_geo_only(
    pred_texts: List[str],
    gold_texts: List[str],
    all_infos: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Geo-only metrics:
      - json_valid_rate
      - state/county/city F1
      - geo_obj_precision/recall/f1
      - geo_obj_exact_match
      - entity_count_* diagnostics
    """
    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    json_valid_rate = sum(1 for p in parsed_pred if p["__valid_json"]) / max(1, len(parsed_pred))

    pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
    gold_state_sets, gold_county_sets, gold_city_sets = [], [], []

    pred_sig_sets: List[set] = []
    gold_sig_sets: List[set] = []

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

    rep_rate = sum(1 for x in all_infos if (x.get("repeated_arrays", 0) or 0) > 0) / max(1, len(all_infos))
    drift_rate = sum(1 for x in all_infos if bool(x.get("had_multi_distinct", False))) / max(1, len(all_infos))

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

        "repetition_rows": float(rep_rate),
        "multi_distinct_rows": float(drift_rate),
    }


# --------------------------
# Evaluation
# --------------------------
def evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=120, log_first_batch=False):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    gen_cfg = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    all_pred_texts: List[str] = []
    all_gold_texts: List[str] = []
    all_infos: List[Dict[str, Any]] = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            gen_out = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg,
            )

            cutoff = input_ids.size(1)

            local_pred_texts = []
            local_infos = []
            for i in range(gen_out.size(0)):
                raw = tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                fixed, info = postprocess_pred_text_geo_only(raw)
                local_pred_texts.append(fixed)
                local_infos.append(info)

            local_gold_texts = batch["targets"]

            gathered_preds = gather_object(local_pred_texts)
            gathered_golds = gather_object(local_gold_texts)
            gathered_infos = gather_object(local_infos)

            if accelerator.is_main_process:
                if gathered_preds and isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred_texts.extend(sub)
                    for sub in gathered_golds:
                        all_gold_texts.extend(sub)
                    for sub in gathered_infos:
                        all_infos.extend(sub)
                else:
                    all_pred_texts.extend(gathered_preds)
                    all_gold_texts.extend(gathered_golds)
                    all_infos.extend(gathered_infos)

                if log_first_batch and step == 0 and local_pred_texts:
                    print("\n[Val Sample]")
                    print("PROMPT :", batch["prompts"][0][:250].replace("\n", " \\n "))
                    print("PRED   :", local_pred_texts[0])
                    print("GOLD   :", local_gold_texts[0])

    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_validation_metrics_geo_only(
            all_pred_texts,
            all_gold_texts,
            all_infos,
        )

    accelerator.wait_for_everyone()
    return metrics


# --------------------------
# Train loop
# --------------------------
def train_loop(args, train_ds: Dataset, val_ds: Dataset):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    if args.fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            backward_prefetch="BACKWARD_PRE",
            activation_checkpointing=True,
        )
        accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_steps=args.grad_accum_steps,
        )
    else:
        accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=args.grad_accum_steps,
        )

    set_seed(args.seed + accelerator.process_index)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        hf_token=hf_token,
        using_accelerator=True,
        checkpoint_path=args.resume_from_checkpoint,
    )

    model = maybe_apply_lora(model, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_collator = TrainCollatorTokenize(tokenizer, max_length=args.max_length)
    val_collator = EvalCollatorTokenize(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    named_params = list(model.named_parameters())
    params = [
        {
            "params": [p for n, p in named_params if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_params if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scheduler = accelerator.prepare(scheduler)

    safe_model_name = args.model_name.replace("/", "_")
    output_dir = os.path.join(args.check_point_path, f"{safe_model_name}_{args.timestamp}")
    best_path = os.path.join(output_dir, "best")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            wandb.init(project=args.wandb_project, config=vars(args), name=f"{safe_model_name}_{args.timestamp}")
            wandb.define_metric("epoch")
            wandb.define_metric("train/loss", step_metric="epoch")
            wandb.define_metric("valid/*", step_metric="epoch")

    accelerator.wait_for_everyone()
    best_metric = -1.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        start = time.time()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                bsz = batch["input_ids"].size(0)
                epoch_loss_sum += loss.item() * bsz
                epoch_count += bsz

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 500 == 0:
                accelerator.print(f"[Epoch {epoch+1}/{args.epochs}] step {step}/{len(train_loader)}")

        loss_tensor = torch.tensor([epoch_loss_sum], dtype=torch.float32, device=accelerator.device)
        cnt_tensor = torch.tensor([epoch_count], dtype=torch.float32, device=accelerator.device)
        gathered_loss = accelerator.gather_for_metrics(loss_tensor)
        gathered_cnt = accelerator.gather_for_metrics(cnt_tensor)

        if accelerator.is_main_process:
            global_loss = gathered_loss.sum().item()
            global_cnt = gathered_cnt.sum().item()
            avg_loss = global_loss / max(1.0, global_cnt)
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f} | elapsed {time.time()-start:.1f}s")
            if args.wandb_project and wandb_key:
                wandb.log({"epoch": epoch + 1, "train/loss": avg_loss}, commit=False)

        metrics = evaluate(
            model,
            val_loader,
            tokenizer,
            accelerator,
            max_new_tokens=args.max_new_tokens,
            log_first_batch=True,
        )

        if accelerator.is_main_process:
            score = float(metrics.get(args.save_best_on, -1.0))

            if args.wandb_project and wandb_key:
                payload = {"epoch": epoch + 1}
                for k, v in metrics.items():
                    payload[f"valid/{k}"] = v
                wandb.log(payload, commit=True)

            if score > best_metric:
                best_metric = score
                os.makedirs(best_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)

                with open(os.path.join(best_path, "val_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                print(f"Saved new best to {best_path} (best {args.save_best_on}={best_metric:.4f})")

            if (epoch + 1) % args.save_every_epochs == 0:
                ckpt_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                with open(os.path.join(ckpt_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.wandb_project and wandb_key:
        wandb.finish()


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train instruction->JSON entity-array model (geo-only, strips evidence and rewrites prompts on the fly)"
    )

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--timestamp", type=str, required=True)

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--prompt_col", type=str, default="prompt")
    p.add_argument("--target_col", type=str, default="target_json")

    p.add_argument("--tune_method", type=str, default="full", choices=["full", "lora"])

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=120)

    p.add_argument("--check_point_path", type=str, default="./checkpoints")
    p.add_argument("--save_best_on", type=str, default="geo_obj_f1")
    p.add_argument("--save_every_epochs", type=int, default=5)
    p.add_argument("--wandb_project", type=str, default=None)

    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    if args.eval_batch_size == 0:
        args.eval_batch_size = args.batch_size

    if args.tune_method == "lora" and get_peft_model is None:
        print("ERROR: tune_method=lora selected but peft not installed. pip install peft")
        return

    train_ds = PromptTargetCSVDataset(
        args.train_csv,
        prompt_col=args.prompt_col,
        target_col=args.target_col,
    )
    val_ds = PromptTargetCSVDataset(
        args.val_csv,
        prompt_col=args.prompt_col,
        target_col=args.target_col,
    )

    train_loop(args, train_ds, val_ds)


if __name__ == "__main__":
    main()
