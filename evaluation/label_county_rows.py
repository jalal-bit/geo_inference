#!/usr/bin/env python3
# labeler_presplit.py
import os
import time
import glob
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import re

# ---------- Config ----------
PRED_SUFFIX = ".preds.csv"
SLEEP_POLL = 0.5
MAX_WAIT = 60 * 60 * 2  # 2 hours (adjust if needed)


def prompt_is_job(text: str) -> str:
    return f"""
You are a classifier. You will classify if a text is a job posting.
Here is an example (one-shot learning):

Text: "This job might be a great fit for you: FT - Sales Specialist ProServices - Opening - #Sales #California, MD"
Answer: {{"is_job": 1}}

Now classify the following text. Answer only with a valid JSON object {{"is_job": 1}} if yes, or {{"is_job": 0}} if no.

Text: "{text}"
Answer:
""".strip()


def prompt_is_traffic(text: str) -> str:
    return f"""
You are a classifier. You will classify if a text is a traffic report.
Here is an example (one-shot learning):

Text: "Cleared: Incident on #I87MajorDeeganExpressway SB at Exit 9 - West North Road"
Answer: {{"is_traffic": 1}}

Now classify the following text. Answer only with a valid JSON object {{"is_traffic": 1}} if yes, or {{"is_traffic": 0}} if no.

Text: "{text}"
Answer:
""".strip()


def load_model(model_name, hf_token, using_accelerator=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None if using_accelerator else "auto",
        token=hf_token,
    )

    if "llama" in model_name.lower():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def safe_json_parse(raw: str, key: str):
    candidate = raw.split("Answer:")[-1].strip()
    candidate = re.sub(r"'", '"', candidate)  # normalize
    try:
        return json.loads(candidate).get(key, 0)
    except Exception as e:
        print(f"[WARN parse failed] raw='{raw}' err={e}")
        return 0


def classify_batch(batch_texts, model, tokenizer, gen_kwargs, accelerator):
    results = {"is_job": [], "is_traffic": []}
    unwrapped_model = accelerator.unwrap_model(model)

    # ---- Job
    job_prompts = [prompt_is_job(t) for t in batch_texts]
    job_tok = tokenizer(job_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    job_out = unwrapped_model.generate(**job_tok, **gen_kwargs)
    job_decoded = tokenizer.batch_decode(job_out, skip_special_tokens=True)
    for d in job_decoded:
        results["is_job"].append(safe_json_parse(d, "is_job"))

    # ---- Traffic
    traffic_prompts = [prompt_is_traffic(t) for t in batch_texts]
    traffic_tok = tokenizer(traffic_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    traffic_out = unwrapped_model.generate(**traffic_tok, **gen_kwargs)
    traffic_decoded = tokenizer.batch_decode(traffic_out, skip_special_tokens=True)
    for d in traffic_decoded:
        results["is_traffic"].append(safe_json_parse(d, "is_traffic"))

    return results


def worker_loop(accelerator, args):
    rank = accelerator.process_index
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    print(f"[rank {rank}] loading model {args.model_name}...")
    model, tokenizer = load_model(args.model_name, hf_token, using_accelerator=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # only consider pre-sharded CSVs
    shards = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*.shard_{rank}.csv"), recursive=True))
    print(f"[rank {rank}] found {len(shards)} shards")

    for shard_path in shards:
        shard_path = Path(shard_path)
        pred_path = shard_path.with_suffix(shard_path.suffix + PRED_SUFFIX)

        try:
            shard_df = pd.read_csv(shard_path)
        except Exception as e:
            print(f"[rank {rank}] failed reading {shard_path}: {e}")
            continue

        texts = shard_df[args.text_col].fillna("").astype(str).tolist()
        outputs = {"is_job": [], "is_traffic": []}

        for i in range(0, len(texts), args.infer_batch_size):
            batch_texts = texts[i:i+args.infer_batch_size]
            preds = classify_batch(batch_texts, model, tokenizer, gen_kwargs, accelerator)
            for k in outputs.keys():
                outputs[k].extend(preds[k])

        pd.DataFrame(outputs).to_csv(pred_path, index=False)
        print(f"[rank {rank}] wrote {pred_path} ({len(outputs['is_job'])} rows)")


def main_process_loop(accelerator, args):
    assert accelerator.process_index == 0

    # collect base CSVs (without shard suffix)
    csvs = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True))
    csvs = [c for c in csvs if ".shard_" not in c and not c.endswith(PRED_SUFFIX)]
    print(f"[main] merging predictions for {len(csvs)} base CSVs")

    for csv_path in csvs:
        csv_path = Path(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[main] failed reading {csv_path}: {e}")
            continue

        base_no_ext = os.path.splitext(str(csv_path))[0]
        shard_paths = sorted(glob.glob(f"{base_no_ext}.shard_*.csv"))
        pred_paths = [Path(sp + PRED_SUFFIX) for sp in shard_paths]

        preds_list = []
        for sp, pp in zip(shard_paths, pred_paths):
            if pp.exists():
                pf = pd.read_csv(pp)
            else:
                shard_len = len(pd.read_csv(sp))
                pf = pd.DataFrame([{"is_job": 0, "is_traffic": 0}] * shard_len)
            preds_list.append(pf)

        merged_preds = pd.concat(preds_list, ignore_index=True)
        df_out = df.copy()
        df_out["is_job"] = merged_preds["is_job"].astype(int)
        df_out["is_traffic"] = merged_preds["is_traffic"].astype(int)

        backup = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup.exists():
            csv_path.rename(backup)
        df_out.to_csv(csv_path, index=False)
        print(f"[main] wrote labeled CSV {csv_path} (backup at {backup})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="../data/raw/eda3/geo_inference_output")
    p.add_argument("--infer_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--text_col", type=str, default="cleaned")
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    print(f"[proc {accelerator.process_index}] starting on {accelerator.device}, world={accelerator.num_processes}")

    # Phase 1: worker labeling
    worker_loop(accelerator, args)
    accelerator.wait_for_everyone()

    # Phase 2: merge predictions
    if accelerator.is_main_process:
        main_process_loop(accelerator, args)

    accelerator.wait_for_everyone()
    print(f"[proc {accelerator.process_index}] done.")


if __name__ == "__main__":
    main()
