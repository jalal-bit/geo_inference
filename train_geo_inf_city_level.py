#!/usr/bin/env python3
"""
Instruction -> JSON city-level training (no pretokenization).

Input CSVs (train/val) must include:
  - prompt: str
  - target: str   (JSON string)

Tokenization happens on-the-fly inside collators.

Supports:
  - full fine-tune
  - LoRA (PEFT)
Optional:
  - FSDP
  - gradient checkpointing
"""

import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

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

    # LLaMA-family pad token safety
    if "llama" in model_name.lower():
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
        # common decoder-only naming
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
    CSV must have columns: prompt, target
    target is the gold JSON string.
    """
    def __init__(self, csv_path: str, prompt_col="prompt", target_col="target"):
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
# Collators (tokenize on-the-fly)
# --------------------------
class TrainCollatorTokenize:
    """
    Train: input = prompt + target
    labels mask prompt tokens to -100 (loss only on target).
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

            # Boundary-preserving truncation: keep target tail; keep as much prompt tail as fits.
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

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tok.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class EvalCollatorTokenize:
    """
    Eval: feed prompt-only to generate; keep gold target strings.
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

        input_ids = pad_sequence(prompt_ids_list, batch_first=True, padding_value=self.tok.pad_token_id)
        attention_mask = (input_ids != self.tok.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "prompts": prompts, "targets": targets}


# --------------------------
# JSON metrics (same as before)
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
# Evaluation
# --------------------------
def evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=80, log_first_batch=False):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    gen_cfg = {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens}

    all_pred_texts: List[str] = []
    all_gold_texts: List[str] = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            gen_out = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg
            )

            cutoff = input_ids.size(1)  # prompt padded length
            local_pred_texts = [
                tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                for i in range(gen_out.size(0))
            ]
            local_gold_texts = batch["targets"]

            gathered_preds = gather_object(local_pred_texts)
            gathered_golds = gather_object(local_gold_texts)

            if accelerator.is_main_process:
                if gathered_preds and isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred_texts.extend(sub)
                    for sub in gathered_golds:
                        all_gold_texts.extend(sub)
                else:
                    all_pred_texts.extend(gathered_preds)
                    all_gold_texts.extend(gathered_golds)

                if log_first_batch and step == 0 and local_pred_texts:
                    print("\n[Val Sample]")
                    print("PROMPT :", batch["prompts"][0][:250].replace("\n", " \\n "))
                    print("PRED   :", local_pred_texts[0])
                    print("GOLD   :", local_gold_texts[0])

    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_validation_metrics_city_json(all_pred_texts, all_gold_texts)

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
        accelerator = Accelerator(mixed_precision="bf16", fsdp_plugin=fsdp_plugin)
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        hf_token=hf_token,
        using_accelerator=True,
        checkpoint_path=args.resume_from_checkpoint,
    )

    model = maybe_apply_lora(model, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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

    set_seed(args.seed + accelerator.process_index)

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

    output_dir = os.path.join(args.check_point_path, f"{args.model_name}_{args.timestamp}")
    best_path = os.path.join(output_dir, "best")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if args.wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.model_name}_{args.timestamp}")

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

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 500 == 0:
                accelerator.print(f"[Epoch {epoch+1}/{args.epochs}] step {step}/{len(train_loader)}")

        # global avg loss
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
                wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        # validation
        metrics = evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=args.max_new_tokens, log_first_batch=True)

        if accelerator.is_main_process:
            score = float(metrics.get(args.save_best_on, -1.0))

            if args.wandb_project and wandb_key:
                payload = {"epoch": epoch + 1}
                for k, v in metrics.items():
                    payload[f"valid/{k}"] = v
                wandb.log(payload)

            if score > best_metric:
                best_metric = score
                os.makedirs(best_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)
                # Full or LoRA: save_pretrained works; for LoRA it saves adapters if PEFT-wrapped
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
    p = argparse.ArgumentParser(description="Train instruction->JSON model (no pretokenization)")

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--timestamp", type=str, required=True)

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--prompt_col", type=str, default="prompt")
    p.add_argument("--target_col", type=str, default="target")

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
    p.add_argument("--max_new_tokens", type=int, default=80)

    p.add_argument("--check_point_path", type=str, default="./checkpoints")
    p.add_argument("--save_best_on", type=str, default="city_f1")
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

    train_ds = PromptTargetCSVDataset(args.train_csv, prompt_col=args.prompt_col, target_col=args.target_col)
    val_ds = PromptTargetCSVDataset(args.val_csv, prompt_col=args.prompt_col, target_col=args.target_col)

    train_loop(args, train_ds, val_ds)


if __name__ == "__main__":
    main()
