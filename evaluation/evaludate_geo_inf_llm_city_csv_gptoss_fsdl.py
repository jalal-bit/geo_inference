#!/usr/bin/env python3
"""
City-level evaluation (CSV, no pretokenization) for instruction->JSON outputs.

UPDATED:
- Adds --fsdp option (Accelerate FSDP FULL_SHARD) for 4-GPU inference
- If --fsdp: use multi-process (1 proc per GPU) and NO device_map
- Else: keep single-process device_map="auto" sharding

ADDED (from your 2nd code; minimal changes):
- Optional error analysis outputs (CSV/TXT + optional confusion matrices)
- Per-test run subfolder: <test_csv_stem>_<timestamp>
- W&B run name includes test_csv stem + timestamp (so test data name is reflected)

FIXED (for "tensor data is not allocated yet" under FSDP):
- FSDP load: low_cpu_mem_usage=False + torch_dtype=bfloat16 + _fast_init=False
- FSDP plugin: sync_module_states=True + use_orig_params=True
- FSDP eval: DO NOT unwrap for generate(); use wrapped model; disable use_cache
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

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import gather_object, broadcast_object_list

from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import wandb

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


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
            if p and t:
                self.examples.append({"prompt": p, "target": t})

        if not self.examples:
            raise ValueError(f"No valid rows found in {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --------------------------
# Collator (tokenize prompt only)
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
# JSON parsing + metrics
# --------------------------
def _norm_item(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    s = s.replace(" county", "").strip()
    return s or None

def _to_set(v) -> set:
    if v is None:
        return set()
    if isinstance(v, list):
        out = set()
        for i in v:
            ni = _norm_item(i)
            if ni:
                out.add(ni)
        return out
    if isinstance(v, str):
        ni = _norm_item(v)
        return {ni} if ni else set()
    return set()

def parse_json_struct(text: str) -> Dict[str, Any]:
    if text is None:
        return {"state": set(), "county": set(), "city": set(), "__valid_json": False}

    raw = str(text).strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end + 1]
            try:
                obj = json.loads(snippet)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return {"state": set(), "county": set(), "city": set(), "__valid_json": False}

    return {
        "state": _to_set(obj.get("state")),
        "county": _to_set(obj.get("county")),
        "city": _to_set(obj.get("city")),
        "__valid_json": True,
    }

def parse_gold_target_json(text: str) -> Dict[str, set]:
    try:
        obj = json.loads(text) if isinstance(text, str) else {}
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    return {
        "state": _to_set(obj.get("state")),
        "county": _to_set(obj.get("county")),
        "city": _to_set(obj.get("city")),
    }

def _micro_prf(pred_sets: List[set], gold_sets: List[set]) -> Dict[str, float]:
    tp = fp = fn = 0
    for p, g in zip(pred_sets, gold_sets):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def _exact_match_rate(pred_sets: List[set], gold_sets: List[set]) -> float:
    if not pred_sets:
        return 0.0
    return sum(1 for p, g in zip(pred_sets, gold_sets) if p == g) / len(pred_sets)

def compute_validation_metrics_city_json(pred_texts: List[str], gold_texts: List[str]) -> Dict[str, float]:
    preds = [parse_json_struct(x) for x in pred_texts]
    golds = [parse_gold_target_json(x) for x in gold_texts]

    valid_json_rate = sum(1 for p in preds if p.get("__valid_json")) / max(1, len(preds))

    pred_state = [p["state"] for p in preds]
    pred_county = [p["county"] for p in preds]
    pred_city = [p["city"] for p in preds]

    gold_state = [g["state"] for g in golds]
    gold_county = [g["county"] for g in golds]
    gold_city = [g["city"] for g in golds]

    state_prf = _micro_prf(pred_state, gold_state)
    county_prf = _micro_prf(pred_county, gold_county)
    city_prf = _micro_prf(pred_city, gold_city)

    state_em = _exact_match_rate(pred_state, gold_state)
    county_em = _exact_match_rate(pred_county, gold_county)
    city_em = _exact_match_rate(pred_city, gold_city)

    all_em = sum(
        1 for i in range(len(preds))
        if pred_state[i] == gold_state[i] and pred_county[i] == gold_county[i] and pred_city[i] == gold_city[i]
    ) / max(1, len(preds))

    return {
        "json_valid_rate": valid_json_rate,
        "state_precision": state_prf["precision"], "state_recall": state_prf["recall"], "state_f1": state_prf["f1"], "state_exact_match": state_em,
        "county_precision": county_prf["precision"], "county_recall": county_prf["recall"], "county_f1": county_prf["f1"], "county_exact_match": county_em,
        "city_precision": city_prf["precision"], "city_recall": city_prf["recall"], "city_f1": city_prf["f1"], "city_exact_match": city_em,
        "all_exact_match": all_em,
    }


# --------------------------
# Error analysis helpers
# --------------------------
def _set_to_pipe(s: set) -> str:
    return "|".join(sorted(s))

def _safe_extract_text_from_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""
    needle = 'Text: "'
    i = prompt.find(needle)
    if i == -1:
        return ""
    j = prompt.find('"', i + len(needle))
    if j == -1:
        return ""
    return prompt[i + len(needle):j]

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
    top_k_labels: int = 50,
):
    suffix = f"_{run_note}" if run_note else ""
    base = f"test_error_analysis{suffix}"

    preds = [parse_json_struct(x) for x in pred_texts]
    golds = [parse_gold_target_json(x) for x in gold_texts]

    rows = []
    for i in range(len(preds)):
        p = preds[i]
        g = golds[i]
        prompt = prompts[i] if i < len(prompts) else ""
        text = _safe_extract_text_from_prompt(prompt)

        rows.append({
            "idx": i,
            "text": text,
            "prompt": prompt,
            "pred_raw": pred_texts[i],
            "gold_raw": gold_texts[i],
            "pred_valid_json": bool(p.get("__valid_json", False)),
            "pred_state": _set_to_pipe(p["state"]),
            "pred_county": _set_to_pipe(p["county"]),
            "pred_city": _set_to_pipe(p["city"]),
            "gold_state": _set_to_pipe(g["state"]),
            "gold_county": _set_to_pipe(g["county"]),
            "gold_city": _set_to_pipe(g["city"]),
            "state_exact": int(p["state"] == g["state"]),
            "county_exact": int(p["county"] == g["county"]),
            "city_exact": int(p["city"] == g["city"]),
            "all_exact": int((p["state"] == g["state"]) and (p["county"] == g["county"]) and (p["city"] == g["city"])),
        })

    df_rows = pd.DataFrame(rows)
    out_rows = out_dir / f"{base}_rows.csv"
    df_rows.to_csv(out_rows, index=False)

    def _save_subset(mask, name):
        df_sub = df_rows[mask].copy()
        out_path = out_dir / f"{base}_{name}.csv"
        df_sub.to_csv(out_path, index=False)
        return out_path, len(df_sub)

    p_state = [p["state"] for p in preds]
    p_county = [p["county"] for p in preds]
    p_city = [p["city"] for p in preds]
    g_state = [g["state"] for g in golds]
    g_county = [g["county"] for g in golds]
    g_city = [g["city"] for g in golds]

    st_err_path, st_err_n = _save_subset(df_rows["state_exact"] == 0, "state_errors")
    co_err_path, co_err_n = _save_subset(df_rows["county_exact"] == 0, "county_errors")
    ci_err_path, ci_err_n = _save_subset(df_rows["city_exact"] == 0, "city_errors")
    all_err_path, all_err_n = _save_subset(df_rows["all_exact"] == 0, "full_errors")

    df_state_item = _entity_prf_by_item(p_state, g_state)
    df_county_item = _entity_prf_by_item(p_county, g_county)
    df_city_item = _entity_prf_by_item(p_city, g_city)

    out_state_item = out_dir / f"{base}_entity_prf_by_item_state.csv"
    out_county_item = out_dir / f"{base}_entity_prf_by_item_county.csv"
    out_city_item = out_dir / f"{base}_entity_prf_by_item_city.csv"
    df_state_item.to_csv(out_state_item, index=False)
    df_county_item.to_csv(out_county_item, index=False)
    df_city_item.to_csv(out_city_item, index=False)

    conf_paths = {}
    try:
        from sklearn.metrics import confusion_matrix  # type: ignore

        y_true_state = [_top1_label(s) for s in g_state]
        y_pred_state = [_top1_label(s) for s in p_state]
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

        y_true_county = [_top1_label(s) for s in g_county]
        y_pred_county = [_top1_label(s) for s in p_county]
        labels_county = _topk_labels(g_county, top_k_labels)
        allowed_county = set(labels_county)
        y_true_county_m = [_map_label(x, allowed_county) for x in y_true_county]
        y_pred_county_m = [_map_label(x, allowed_county) for x in y_pred_county]
        labels_county_full = labels_county + ["__OTHER__"]
        cm_county = confusion_matrix(y_true_county_m, y_pred_county_m, labels=labels_county_full)
        df_cm_county = pd.DataFrame(cm_county, index=labels_county_full, columns=labels_county_full)
        out_cm_county = out_dir / f"test_confusion_county_top{top_k_labels}{suffix}.csv"
        df_cm_county.to_csv(out_cm_county)
        conf_paths["county"] = out_cm_county

        y_true_city = [_top1_label(s) for s in g_city]
        y_pred_city = [_top1_label(s) for s in p_city]
        labels_city = _topk_labels(g_city, top_k_labels)
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

    out_txt = out_dir / f"{base}_summary.txt"
    lines = []
    lines.append("ERROR ANALYSIS SUMMARY")
    lines.append("======================\n")
    lines.append(f"Rows: {len(df_rows):,}")
    lines.append(f"Valid JSON rate: {df_rows['pred_valid_json'].mean():.6f}\n")
    lines.append("Exact-match error counts:")
    lines.append(f"  State errors:  {st_err_n:,}  (saved: {st_err_path.name})")
    lines.append(f"  County errors: {co_err_n:,}  (saved: {co_err_path.name})")
    lines.append(f"  City errors:   {ci_err_n:,}  (saved: {ci_err_path.name})")
    lines.append(f"  Full errors:   {all_err_n:,} (saved: {all_err_path.name})\n")

    def _miss_hall(p_sets: List[set], g_sets: List[set]) -> Tuple[Counter, Counter]:
        missed = Counter()
        halluc = Counter()
        for p, g in zip(p_sets, g_sets):
            for it in (g - p):
                missed[it] += 1
            for it in (p - g):
                halluc[it] += 1
        return missed, halluc

    missed_state, halluc_state = _miss_hall(p_state, g_state)
    missed_county, halluc_county = _miss_hall(p_county, g_county)
    missed_city, halluc_city = _miss_hall(p_city, g_city)

    def _top(counter: Counter, n: int = 20) -> List[str]:
        return [f"{k} ({v})" for k, v in counter.most_common(n)]

    lines.append("Top missed states:   " + (", ".join(_top(missed_state, 10)) if missed_state else "None"))
    lines.append("Top halluc states:   " + (", ".join(_top(halluc_state, 10)) if halluc_state else "None"))
    lines.append("Top missed counties: " + (", ".join(_top(missed_county, 20)) if missed_county else "None"))
    lines.append("Top halluc counties: " + (", ".join(_top(halluc_county, 20)) if halluc_county else "None"))
    lines.append("Top missed cities:   " + (", ".join(_top(missed_city, 20)) if missed_city else "None"))
    lines.append("Top halluc cities:   " + (", ".join(_top(halluc_city, 20)) if halluc_city else "None"))
    lines.append("")

    lines.append("Per-item PRF tables:")
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
        "state_item_csv": out_state_item,
        "county_item_csv": out_county_item,
        "city_item_csv": out_city_item,
        "conf_paths": conf_paths,
    }


# --------------------------
# Model loading helpers
# --------------------------
def _build_max_memory(per_gpu_gb: int = 78) -> Dict[int, str]:
    n = torch.cuda.device_count()
    return {i: f"{per_gpu_gb}GiB" for i in range(n)}

def load_model_for_eval(
    model_name: str,
    hf_token: Optional[str],
    checkpoint_folder: str,
    checkpoint_path: Optional[str],
    use_lora_adapter: bool,
    per_gpu_gb: int,
    fsdp: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = _build_max_memory(per_gpu_gb=per_gpu_gb) if torch.cuda.is_available() else None

    def _load_base(path_or_name: str):
        if fsdp:
            # FSDP: multi-process sharding, do NOT use device_map
            # FIX: low_cpu_mem_usage=False + _fast_init=False avoids meta/unallocated params
            return AutoModelForCausalLM.from_pretrained(
                path_or_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=False,
                _fast_init=False,
                token=hf_token,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                path_or_name,
                torch_dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                token=hf_token,
            )

    if checkpoint_path and checkpoint_path.strip():
        best_dir = Path(checkpoint_folder) / f"{model_name}_{checkpoint_path}" / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_dir}")

        if use_lora_adapter:
            if PeftModel is None:
                raise RuntimeError("peft is not installed but --use_lora_adapter was set.")
            base = _load_base(model_name)
            model = PeftModel.from_pretrained(base, str(best_dir))
        else:
            model = _load_base(str(best_dir))
    else:
        model = _load_base(model_name)

    return model, tokenizer


# --------------------------
# Eval loop
# --------------------------
def evaluate(model, loader, tokenizer, accelerator, max_new_tokens=25, log_first_batch=True, fsdp=False):
    model.eval()

    gen_cfg = {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens}
    all_pred, all_gold, all_prompts = [], [], []
    total_steps = len(loader)

    # FIX: under FSDP, keep wrapped model for generate()
    gen_model = model

    if fsdp and getattr(gen_model, "config", None) is not None:
        gen_model.config.use_cache = False

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)

            gen_out = gen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg
            )

            cutoff = input_ids.size(1)
            local_pred = [
                tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                for i in range(gen_out.size(0))
            ]
            local_gold = batch["targets"]
            local_prompts = batch["prompts"]

            gathered_preds = gather_object(local_pred)
            gathered_golds = gather_object(local_gold)
            gathered_prompts = gather_object(local_prompts)

            if accelerator.is_main_process:
                if gathered_preds and isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred.extend(sub)
                    for sub in gathered_golds:
                        all_gold.extend(sub)
                    for sub in gathered_prompts:
                        all_prompts.extend(sub)
                else:
                    all_pred.extend(gathered_preds)
                    all_gold.extend(gathered_golds)
                    all_prompts.extend(gathered_prompts)

                if log_first_batch and step == 0 and local_pred:
                    accelerator.print("\n[Eval Sample]")
                    accelerator.print("PROMPT:", batch["prompts"][0][:250].replace("\n", " \\n "))
                    accelerator.print("PRED  :", local_pred[0])
                    accelerator.print("GOLD  :", local_gold[0])

                if step % 200 == 0 or step == total_steps - 1:
                    accelerator.print(f"Step {step+1}/{total_steps}")

    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_validation_metrics_city_json(all_pred, all_gold)

    accelerator.wait_for_everyone()
    return metrics, all_prompts, all_pred, all_gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)

    p.add_argument("--checkpoint_folder", default="checkpoints")
    p.add_argument("--checkpoint_path", default=None, help="TIMESTAMP used in training output dir")
    p.add_argument("--use_lora_adapter", action="store_true")

    p.add_argument("--test_csv", required=True)
    p.add_argument("--prompt_col", default="prompt")
    p.add_argument("--target_col", default="target_json")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=25)

    p.add_argument("--per_gpu_gb", type=int, default=78, help="Max memory per GPU for device_map auto.")

    # FSDP
    p.add_argument("--fsdp", action="store_true", help="Use FSDP FULL_SHARD across processes (run with 4 procs).")
    p.add_argument("--fsdp_cpu_offload", action="store_true", help="Enable CPU offload in FSDP plugin.")

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_note", default=None)

    # error analysis
    p.add_argument("--error_analysis", action="store_true", help="Write error analysis CSV/TXT files.")
    p.add_argument("--confusion_top_k", type=int, default=50, help="Top-K labels for county/city confusion matrices (requires sklearn).")

    args = p.parse_args()

    # choose accelerator config based on args.fsdp
    if args.fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            cpu_offload=args.fsdp_cpu_offload,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            backward_prefetch="BACKWARD_PRE",
            activation_checkpointing=False,  # inference

            # FIX: ensures all ranks get real weights; avoids unallocated tensors
            sync_module_states=True,
            use_orig_params=True,
        )
        accelerator = Accelerator(mixed_precision="bf16", fsdp_plugin=fsdp_plugin)
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    if args.fsdp:
        if torch.cuda.device_count() < 1:
            accelerator.print("ERROR: No CUDA devices visible.")
            return
    else:
        if torch.cuda.device_count() < 4:
            accelerator.print(
                f"WARNING: only {torch.cuda.device_count()} GPU(s) visible. "
                f"Request 4 GPUs and ensure CUDA_VISIBLE_DEVICES has 4 ids."
            )

    ds = PromptTargetCSVDataset(args.test_csv, prompt_col=args.prompt_col, target_col=args.target_col)

    model, tokenizer = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
        use_lora_adapter=args.use_lora_adapter,
        per_gpu_gb=args.per_gpu_gb,
        fsdp=args.fsdp,
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

    if args.fsdp:
        model, loader = accelerator.prepare(model, loader)
    else:
        loader = accelerator.prepare(loader)

    base_out_dir = Path(args.checkpoint_folder) / (
        f"{args.model_name}_{args.checkpoint_path}/best" if args.checkpoint_path else f"{args.model_name}_base"
    )

    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_list = [ts_run]
    try:
        broadcast_object_list(ts_list)
    except Exception:
        pass
    ts_run = ts_list[0]

    test_stem = Path(args.test_csv).stem
    out_dir = base_out_dir / f"{test_stem}_{ts_run}"

    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            run_name = f"{args.model_name}_{args.checkpoint_path or 'base'}_{args.run_note or 'eval'}_{test_stem}_{ts_run}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    accelerator.wait_for_everyone()

    metrics, all_prompts, all_pred, all_gold = evaluate(
        model, loader, tokenizer, accelerator,
        max_new_tokens=args.max_new_tokens,
        fsdp=args.fsdp
    )

    if accelerator.is_main_process:
        accelerator.print("\n=== METRICS ===")
        for k, v in metrics.items():
            accelerator.print(f"{k}: {v}")

        fname = f"test_metrics_{args.run_note}.json" if args.run_note else "test_metrics.json"
        fpath = out_dir / fname
        if fpath.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = out_dir / (fname.replace(".json", f"_{ts}.json"))

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        accelerator.print("Saved:", fpath)

        if args.error_analysis:
            artifacts = run_error_analysis(
                out_dir=out_dir,
                run_note=args.run_note,
                prompts=all_prompts,
                pred_texts=all_pred,
                gold_texts=all_gold,
                top_k_labels=args.confusion_top_k,
            )
            accelerator.print("Error analysis saved:")
            accelerator.print("  -", str(artifacts["rows_csv"]))
            accelerator.print("  -", str(artifacts["summary_txt"]))

        if args.wandb_project and wandb_key:
            wandb.log({f"test/{k}": v for k, v in metrics.items()})
            wandb.finish()

        accelerator.print("All results saved under:", out_dir)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
