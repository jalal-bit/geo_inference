#!/usr/bin/env python3
"""
Instruction -> JSON entity-array training (no pretokenization).

Input CSVs (train/val) must include:
  - prompt: str
  - target_json: str   (JSON string, ARRAY of objects)

Tokenization happens on-the-fly inside collators.

Supports:
  - full fine-tune
  - LoRA / QLoRA (PEFT)
Optional:
  - FSDP
  - gradient checkpointing

Evaluation updated for entity-array structure with:
  - json_valid_rate
  - geo_obj_* (hierarchy-only entity F1)
  - evidence_* (fuzzy evidence scoring on matched geo objects)
  - structured_* (headline: hierarchy + decent evidence + count via FP/FN)
  - entity_count_* diagnostics
"""

import os
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

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import gather_object

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from dotenv import load_dotenv
import wandb

# Optional PEFT (LoRA / QLoRA)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.utils.other import fsdp_auto_wrap_policy
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    prepare_model_for_kbit_training = None
    fsdp_auto_wrap_policy = None


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Left padding helper
# --------------------------
def left_pad_tensors(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """
    Left-pad a list of 1D tensors to the same length.
    """
    if not tensors:
        return torch.empty(0, dtype=torch.long)

    max_len = max(t.size(0) for t in tensors)
    out = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_value, dtype=t.dtype)
            t = torch.cat([pad, t], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)


# --------------------------
# Load model + tokenizer
# --------------------------
def load_model_and_tokenizer(
    model_name: str,
    hf_token: Optional[str],
    using_accelerator: bool,
    checkpoint_path: Optional[str],
    use_4bit: bool = False,
    use_flash_attn: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_source = checkpoint_path if checkpoint_path else model_name

    extra_kwargs = {}
    if "gemma" in model_name.lower():
        extra_kwargs["attn_implementation"] = "eager"
    elif use_flash_attn:
        extra_kwargs["attn_implementation"] = "flash_attention_2"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map=None,  # let Accelerate/FSDP manage placement
            # device_map=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}",
            #device_map="cpu",
            low_cpu_mem_usage=True,
            token=hf_token,
            **extra_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            device_map=None if using_accelerator else "auto",
            low_cpu_mem_usage=True,
            token=hf_token,
            **extra_kwargs,
        )

    # Pad-token and cache safety
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

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

    if args.use_4bit:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("prepare_model_for_kbit_training not available in peft")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    else:
        # Better default for Llama-family QLoRA
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    target_dtype = torch.bfloat16
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            print(f"[cast fp32] {name}: {param.dtype} -> {target_dtype}, requires_grad={param.requires_grad}")
            param.data = param.data.to(target_dtype)

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
            examples.append({"prompt": p, "target": t})

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
    Train: input = prompt + target
    labels mask prompt tokens to -100 (loss only on target)
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

        input_ids = left_pad_tensors(input_ids_list, self.tok.pad_token_id)
        labels = left_pad_tensors(labels_list, -100)
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class EvalCollatorTokenize:
    """
    Eval: feed prompt-only to generate; keep gold target strings
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

        input_ids = left_pad_tensors(prompt_ids_list, self.tok.pad_token_id)
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
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = s.replace(" county", "").strip()
    s = "".join(s.split())
    return s or None


def _norm_evidence_item(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in "/-")
    s = " ".join(s.split())
    return s or None


# --------------------------
# Salvage helpers
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


def _norm_evidence_set(ent: Dict[str, Any]) -> set:
    ev = ent.get("evidence") or ent.get("mention") or []
    if not isinstance(ev, list):
        ev = [ev]
    out = set()
    for x in ev:
        nx = _norm_evidence_item(x)
        if nx:
            out.add(nx)
    return out


def entity_geo_key(ent: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    st = _norm_admin(ent.get("state"))
    co = _norm_admin(ent.get("county"))
    ci = _norm_admin(ent.get("city"))

    if not st or not co:
        return None
    return (st, co, ci or "")


def dedupe_geo_key_with_evidence_subset(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    kept: List[Dict[str, Any]] = []
    stats = {"geo_dedup_drops": 0.0}

    for ent in entities:
        gk = entity_geo_key(ent)
        if gk is None:
            kept.append(ent)
            continue

        ev_set = _norm_evidence_set(ent)
        placed = False

        for idx, prev in enumerate(kept):
            if entity_geo_key(prev) != gk:
                continue

            prev_ev = _norm_evidence_set(prev)

            if ev_set == prev_ev:
                stats["geo_dedup_drops"] += 1.0
                placed = True
                break

            if ev_set and prev_ev and ev_set.issubset(prev_ev):
                stats["geo_dedup_drops"] += 1.0
                placed = True
                break

            if ev_set and prev_ev and prev_ev.issubset(ev_set):
                kept[idx] = ent
                stats["geo_dedup_drops"] += 1.0
                placed = True
                break

            if not ev_set and prev_ev:
                stats["geo_dedup_drops"] += 1.0
                placed = True
                break

            if ev_set and not prev_ev:
                kept[idx] = ent
                stats["geo_dedup_drops"] += 1.0
                placed = True
                break

        if not placed:
            kept.append(ent)

    return kept, stats


def postprocess_pred_text_entity_arrays(raw: str) -> Tuple[str, Dict[str, Any]]:
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
        return "", info

    arrays = extract_all_json_arrays_balanced(raw)
    info["found_arrays"] = len(arrays)
    info["had_multi_arrays"] = len(arrays) > 1

    if not arrays:
        return ("" if raw is None else str(raw)), info

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
        parsed_lists.append(ents)
        canon_keys.append(_canonical_json(ents))

    info["parsed_arrays"] = len(parsed_lists)
    if not parsed_lists:
        return ("" if raw is None else str(raw)), info

    seen: set = set()
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

    merged2, geo_stats = dedupe_geo_key_with_evidence_subset(merged)
    info["geo_dedup_drops"] = int(geo_stats.get("geo_dedup_drops", 0) or 0)

    return json.dumps(merged2, ensure_ascii=False), info


# --------------------------
# Parsing entity array JSON
# --------------------------
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
# Evidence fuzzy matching
# --------------------------
def _char_ngrams(s: str, n: int = 3) -> set:
    if s is None:
        return set()
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


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


def evidence_f1_for_pair(pred_ent: Dict[str, Any], gold_ent: Dict[str, Any], thr: float) -> float:
    pred_e = pred_ent.get("evidence") or pred_ent.get("mention") or []
    gold_e = gold_ent.get("evidence") or gold_ent.get("mention") or []
    if not isinstance(pred_e, list):
        pred_e = [pred_e]
    if not isinstance(gold_e, list):
        gold_e = [gold_e]

    m = greedy_evidence_match(pred_e, gold_e, thr)
    p = m / len(pred_e) if pred_e else (1.0 if not gold_e else 0.0)
    r = m / len(gold_e) if gold_e else (1.0 if not pred_e else 0.0)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


# --------------------------
# Object matching and metrics
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


def compute_validation_metrics_entities_json(
    pred_texts: List[str],
    gold_texts: List[str],
    all_infos: List[Dict[str, Any]],
    evidence_match_threshold: float = 0.75,
    evidence_ok_threshold: float = 0.50,
) -> Dict[str, float]:
    parsed_pred = [parse_entity_array(x) for x in pred_texts]
    parsed_gold = [parse_entity_array(x) for x in gold_texts]

    json_valid_rate = sum(1 for p in parsed_pred if p["__valid_json"]) / max(1, len(parsed_pred))

    pred_state_sets, pred_county_sets, pred_city_sets = [], [], []
    gold_state_sets, gold_county_sets, gold_city_sets = [], [], []

    pred_sig_sets: List[set] = []
    gold_sig_sets: List[set] = []

    ev_tp = ev_fp = ev_fn = 0.0
    s_tp = s_fp = s_fn = 0.0

    abs_err_sum = 0.0
    bias_sum = 0.0
    exact_count_hits = 0.0
    pred_count_sum = 0.0
    gold_count_sum = 0.0

    for p, g in zip(parsed_pred, parsed_gold):
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

        n_pred = len(p_sigs)
        n_gold = len(g_sigs)
        pred_count_sum += n_pred
        gold_count_sum += n_gold
        abs_err_sum += abs(n_pred - n_gold)
        bias_sum += (n_pred - n_gold)
        if n_pred == n_gold:
            exact_count_hits += 1.0

        matches = match_by_geo(pred_valid, gold_valid)

        for (pi, gi) in matches:
            pe = pred_valid[pi].get("evidence", pred_valid[pi].get("mention", []))
            ge = gold_valid[gi].get("evidence", gold_valid[gi].get("mention", []))
            if not isinstance(pe, list):
                pe = [pe]
            if not isinstance(ge, list):
                ge = [ge]

            m = greedy_evidence_match(pe, ge, evidence_match_threshold)
            ev_tp += m
            ev_fp += max(0, len(pe) - m)
            ev_fn += max(0, len(ge) - m)

        struct_tp_here = 0.0
        for (pi, gi) in matches:
            ef1 = evidence_f1_for_pair(pred_valid[pi], gold_valid[gi], evidence_match_threshold)
            if ef1 >= evidence_ok_threshold:
                struct_tp_here += 1.0

        s_tp += struct_tp_here
        s_fp += max(0.0, n_pred - struct_tp_here)
        s_fn += max(0.0, n_gold - struct_tp_here)

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

    ev_prf = _micro_prf_counts(ev_tp, ev_fp, ev_fn)
    structured_prf = _micro_prf_counts(s_tp, s_fp, s_fn)

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

        "evidence_precision": float(ev_prf["precision"]),
        "evidence_recall": float(ev_prf["recall"]),
        "evidence_f1": float(ev_prf["f1"]),

        "structured_precision": float(structured_prf["precision"]),
        "structured_recall": float(structured_prf["recall"]),
        "structured_f1": float(structured_prf["f1"]),

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
def evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=120, log_first_batch=False,
             evidence_match_threshold=0.75, evidence_ok_threshold=0.50):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    gen_cfg = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
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
                **gen_cfg
            )

            cutoff = input_ids.size(1)

            local_pred_texts = []
            local_infos = []
            for i in range(gen_out.size(0)):
                raw = tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                fixed, _info = postprocess_pred_text_entity_arrays(raw)
                local_pred_texts.append(fixed)
                local_infos.append(_info)

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
        metrics = compute_validation_metrics_entities_json(
            all_pred_texts,
            all_gold_texts,
            all_infos,
            evidence_match_threshold=evidence_match_threshold,
            evidence_ok_threshold=evidence_ok_threshold,
        )

    accelerator.wait_for_everyone()
    return metrics


# --------------------------
# Save helper
# --------------------------
def save_model_checkpoint(accelerator, model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)

    # For PEFT/LoRA this should save adapter weights;
    # get_state_dict helps with distributed/FSDP-safe saving.
    state_dict = accelerator.get_state_dict(model)

    unwrapped.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    tokenizer.save_pretrained(save_dir)



def cast_trainable_params_to_bf16(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    return model
# --------------------------
# Train loop
# --------------------------
def train_loop(args, train_ds: Dataset, val_ds: Dataset):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # if args.fsdp:
    #     fsdp_plugin = FullyShardedDataParallelPlugin(
    #         sharding_strategy="FULL_SHARD",
    #         cpu_offload=False,
    #         auto_wrap_policy="TRANSFORMER_BASED_WRAP",
    #         backward_prefetch="BACKWARD_PRE",
    #         activation_checkpointing=args.gradient_checkpointing,
    #         state_dict_type="SHARDED_STATE_DICT",
    #     )
    #     accelerator = Accelerator(
    #         mixed_precision="bf16",
    #         gradient_accumulation_steps=args.grad_accum_steps,
    #         fsdp_plugin=fsdp_plugin,
    #     )
    # if args.fsdp:
    #     fsdp_plugin = FullyShardedDataParallelPlugin(
    #     sharding_strategy="FULL_SHARD",
    #     cpu_offload=False,
    #     auto_wrap_policy="TRANSFORMER_BASED_WRAP",  # temporary; replaced after LoRA wrapping
    #     backward_prefetch="BACKWARD_PRE",
    #     activation_checkpointing=args.gradient_checkpointing,
    #     state_dict_type="SHARDED_STATE_DICT",
    #     use_orig_params=False,
    #     sync_module_states=True,
    #     cpu_ram_efficient_loading=True,)
    #     accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum_steps,fsdp_plugin=fsdp_plugin,)
    # else:
    #     accelerator = Accelerator(
    #         mixed_precision="bf16",
    #         gradient_accumulation_steps=args.grad_accum_steps,
    #     )

    # if torch.cuda.is_available():
    #     torch.cuda.set_device(accelerator.local_process_index)
    # if args.fsdp:
    #     accelerator.state.fsdp_plugin.device_id = accelerator.local_process_index

    # set_seed(args.seed + accelerator.process_index)
    

    # model, tokenizer = load_model_and_tokenizer(
    #     model_name=args.model_name,
    #     hf_token=hf_token,
    #     using_accelerator=True,
    #     checkpoint_path=args.resume_from_checkpoint,
    #     use_4bit=args.use_4bit,
    #     use_flash_attn=args.use_flash_attn,
    # )

    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    #     if getattr(model, "config", None) is not None:
    #         model.config.use_cache = False

    # model = maybe_apply_lora(model, args)
    # if args.fsdp:
    #     if fsdp_auto_wrap_policy is None:
    #         raise RuntimeError("fsdp_auto_wrap_policy not available from peft")
    #     accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    if args.fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
        cpu_offload=False,
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",  # temporary; replaced after LoRA wrapping
        backward_prefetch="BACKWARD_PRE",
        activation_checkpointing=args.gradient_checkpointing,
        state_dict_type="SHARDED_STATE_DICT",
        use_orig_params=False,
        sync_module_states=False,
        cpu_ram_efficient_loading=True,
        )
        accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
        )
        accelerator.state.fsdp_plugin.device_id = local_rank
    else:
        accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,)
        
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(accelerator.local_process_index)
    # if args.fsdp and torch.cuda.is_available():
    #     accelerator.state.fsdp_plugin.device_id = torch.device(f"cuda:{accelerator.local_process_index}")
        
    set_seed(args.seed + accelerator.process_index)
    model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            hf_token=hf_token,
            using_accelerator=True,
            checkpoint_path=args.resume_from_checkpoint,
            use_4bit=args.use_4bit,
            use_flash_attn=args.use_flash_attn,
        )
        
    model = maybe_apply_lora(model, args)

    # if args.fsdp and args.tune_method == "lora":
   #      model = cast_trainable_params_to_bf16(model)
    
    if args.gradient_checkpointing and not args.use_4bit:
        model.gradient_checkpointing_enable()
            
            
    if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        
    if args.fsdp and args.tune_method == "lora":
        if fsdp_auto_wrap_policy is None:
            raise RuntimeError("fsdp_auto_wrap_policy not available from peft")
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

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

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scheduler = accelerator.prepare(scheduler)

    output_dir = os.path.join(args.check_point_path, f"{args.model_name}_{args.timestamp}")
    best_path = os.path.join(output_dir, "best")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.model_name}_{args.timestamp}")
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

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                bsz = batch["input_ids"].size(0)
                epoch_loss_sum += loss.detach().float().item() * bsz
                epoch_count += bsz

            if step % 200 == 0:
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
            evidence_match_threshold=args.evidence_match_threshold,
            evidence_ok_threshold=args.evidence_ok_threshold,
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
                save_model_checkpoint(accelerator, model, tokenizer, best_path)

                with open(os.path.join(best_path, "val_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                print(f"Saved new best to {best_path} (best {args.save_best_on}={best_metric:.4f})")

            if (epoch + 1) % args.save_every_epochs == 0:
                ckpt_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
                save_model_checkpoint(accelerator, model, tokenizer, ckpt_dir)
                with open(os.path.join(ckpt_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.wandb_project and wandb_key:
        wandb.finish()


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train instruction->JSON entity-array model (no pretokenization)")

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--timestamp", type=str, required=True)

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--prompt_col", type=str, default="prompt")
    p.add_argument("--target_col", type=str, default="target_json")

    p.add_argument("--tune_method", type=str, default="full", choices=["full", "lora"])

    # QLoRA / memory
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--use_flash_attn", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")

    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=120)

    p.add_argument("--check_point_path", type=str, default="./checkpoints")
    p.add_argument("--save_best_on", type=str, default="structured_f1")
    p.add_argument("--save_every_epochs", type=int, default=5)
    p.add_argument("--wandb_project", type=str, default=None)

    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--evidence_match_threshold", type=float, default=0.75)
    p.add_argument("--evidence_ok_threshold", type=float, default=0.50)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    
    args, unknown = p.parse_known_args()

    if unknown:
        print("\n❌ Unrecognized arguments detected:")
        for u in unknown:
            print(f"   {u}")
        exit(1)

    return args


def main():
    args = parse_args()
    if args.eval_batch_size == 0:
        args.eval_batch_size = args.batch_size

    if args.tune_method == "lora" and get_peft_model is None:
        print("ERROR: tune_method=lora selected but peft not installed. pip install peft")
        return

    if args.use_4bit and args.tune_method != "lora":
        print("WARNING: --use_4bit is mainly intended for LoRA/QLoRA training.")

    train_ds = PromptTargetCSVDataset(
        args.train_csv, prompt_col=args.prompt_col, target_col=args.target_col
    )
    val_ds = PromptTargetCSVDataset(
        args.val_csv, prompt_col=args.prompt_col, target_col=args.target_col
    )

    train_loop(args, train_ds, val_ds)


if __name__ == "__main__":
    main()

