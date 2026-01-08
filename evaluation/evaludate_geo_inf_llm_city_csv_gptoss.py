#!/usr/bin/env python3
"""
City-level evaluation (CSV, no pretokenization) for instruction->JSON outputs.

UPDATED (minimal):
- Keeps 4-GPU sharded loading via device_map="auto"
- Adds Accelerator flow (like first code) for printing / main-process logic / sync
- IMPORTANT: run with ONE process. Model uses 4 GPUs because device_map shards it.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import gather_object  # kept for similarity; with 1 proc it’s effectively a no-op

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
# Model loading (4-GPU sharded)
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
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = _build_max_memory(per_gpu_gb=per_gpu_gb) if torch.cuda.is_available() else None

    def _load_base(path_or_name: str):
        return AutoModelForCausalLM.from_pretrained(
            path_or_name,
            torch_dtype="auto",
            device_map="auto",        # <-- shards across all visible GPUs
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

        print(f"Loaded checkpoint: {best_dir} (use_lora_adapter={use_lora_adapter})")
    else:
        model = _load_base(model_name)
        print(f"Loaded base model: {model_name}")

    return model, tokenizer


# --------------------------
# Eval loop (Accelerator-style, but single-process sharded model)
# --------------------------
def evaluate(model, loader, tokenizer, accelerator, max_new_tokens=25, log_first_batch=True):
    model.eval()

    # For sharded model, accelerator.device will usually be cuda:0 (fine for input placement).
    gen_cfg = {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens}

    all_pred, all_gold = [], []
    total_steps = len(loader)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)

            gen_out = model.generate(
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

            # With 1 process, gather_object is effectively identity; kept for compatibility.
            gathered_preds = gather_object(local_pred)
            gathered_golds = gather_object(local_gold)

            if accelerator.is_main_process:
                # gather_object returns a list in single-process; extend directly
                all_pred.extend(gathered_preds)
                all_gold.extend(gathered_golds)

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
    return metrics


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
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_note", default=None)

    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)
    wandb_key = os.getenv("WANDB_API_KEY", None)

    if torch.cuda.device_count() < 4:
        accelerator.print(f"WARNING: only {torch.cuda.device_count()} GPU(s) visible. "
                          f"Request 4 GPUs from SLURM and ensure CUDA_VISIBLE_DEVICES has 4 ids.")

    ds = PromptTargetCSVDataset(args.test_csv, prompt_col=args.prompt_col, target_col=args.target_col)

    model, tokenizer = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
        use_lora_adapter=args.use_lora_adapter,
        per_gpu_gb=args.per_gpu_gb,
    )

    collator = EvalCollatorTokenize(tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                        num_workers=2, pin_memory=True)

    # IMPORTANT: do NOT prepare(model) here (would wrap for DDP).
    loader = accelerator.prepare(loader)

    out_dir = Path(args.checkpoint_folder) / (
        f"{args.model_name}_{args.checkpoint_path}/best" if args.checkpoint_path else f"{args.model_name}_base"
    )
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            run_name = f"{args.model_name}_{args.checkpoint_path or 'base'}_{args.run_note or 'eval'}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    metrics = evaluate(model, loader, tokenizer, accelerator, max_new_tokens=args.max_new_tokens)

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

        if args.wandb_project and wandb_key:
            wandb.log({f"test/{k}": v for k, v in metrics.items()})
            wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
