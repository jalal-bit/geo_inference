#!/usr/bin/env python3
# labeler.py
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

# ---------- Config ----------
SHARD_PREFIX = ".shard_"
PRED_SUFFIX = ".preds.csv"
SLEEP_POLL = 0.5
MAX_WAIT = 60 * 60 * 2  # 2 hours (adjust if needed)




# ---------------------------
# Prompts
# ---------------------------
def prompt_is_job(text: str) -> str:
    return f"""
You are a classifier. You will classify if a text is a job posting.
Here is an example (one-shot learning):

Text: "This job might be a great fit for you: FT - Sales Specialist ProServices - Opening - #Sales #California, MD"
Answer: {{"is_job": 1}}

Now classify the following text. Answer only with a JSON object {{'is_job': 1}} if yes, {{'is_job': 0}} if no.

Text: "{text}"
Answer:
""".strip()


def prompt_is_traffic(text: str) -> str:
    return f"""
You are a classifier. You will classify if a text is a traffic report.
Here is an example (one-shot learning):

Text: "Cleared: Incident on #I87MajorDeeganExpressway SB at Exit 9 - West North Road"
Answer: {{"is_traffic": 1}}

Now classify the following text. Answer only with a JSON object {{'is_traffic': 1}} if yes, {{'is_traffic': 0}} if no.

Text: "{text}"
Answer:
""".strip()


# ----------------- Your starter load_model (integrated) -----------------
def load_model(model_name, hf_token, using_accelerator=False, checkpoint_folder=None, checkpoint_path=None):
    # Determine the model source (either base model or checkpoint directory)
    if checkpoint_folder and checkpoint_path:
        model_source=os.path.join(checkpoint_folder,f"{model_name}_{checkpoint_path}/best")
    else:
        model_source = model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model from checkpoint or model name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,  # adapted to L40S / bf16 if available
        device_map=None if using_accelerator else "auto",
        token=hf_token,
    )

    if "llama" in model_name.lower():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
# -----------------------------------------------------------------------

def shard_dataframe(df, num_shards):
    shards = []
    n = len(df)
    per = n // num_shards
    extras = n % num_shards
    start = 0
    for r in range(num_shards):
        end = start + per + (1 if r < extras else 0)
        shards.append(df.iloc[start:end].copy())
        start = end
    return shards




def classify_batch(batch_texts, model, tokenizer,gen_kwargs):
    results = {"is_job": [], "is_traffic": []}

    # ---- Job classifier
    job_prompts = [prompt_is_job(t) for t in batch_texts]
    job_tok = tokenizer(job_prompts, return_tensors="pt", padding=True, truncation=True,max_length=512).to(model.device)
    job_out = model.generate(**job_tok, **gen_kwargs)
    job_decoded = tokenizer.batch_decode(job_out, skip_special_tokens=True)
    for d in job_decoded:
        try:
            parsed = json.loads(d.split("Answer:")[-1].strip())
        except Exception:
            parsed = {"is_job": 0}
        results["is_job"].append(parsed["is_job"])

    # ---- Traffic classifier
    traffic_prompts = [prompt_is_traffic(t) for t in batch_texts]
    traffic_tok = tokenizer(traffic_prompts, return_tensors="pt", padding=True, truncation=True,max_length=512).to(model.device)
    traffic_out = model.generate(**traffic_tok,**gen_kwargs )
    traffic_decoded = tokenizer.batch_decode(traffic_out, skip_special_tokens=True)
    for d in traffic_decoded:
        try:
            parsed = json.loads(d.split("Answer:")[-1].strip())
        except Exception:
            parsed = {"is_traffic": 0}
        results["is_traffic"].append(parsed["is_traffic"])


    return results


def worker_loop(accelerator, args):
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    # Load model & tokenizer using your load_model function
    model_name = args.model_name
    print(f"[rank {rank}] loading model {model_name}...")
    model, tokenizer = load_model(model_name, hf_token, using_accelerator=True)
    # Move model+tokenizer to accelerator (prepares for device)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Generation config helper
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": False,
    }

    # discover CSVs once
    csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*", "*.csv")))
    if len(csvs) == 0:
        # fallback older path pattern
        csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.csv")))

    for csv_path in csvs:
        csv_path = Path(csv_path)
        shard_path = csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{rank}.csv")
        pred_path = csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{rank}{PRED_SUFFIX}")

        # Wait until the shard exists (written by main) or timeout
        waited = 0.0
        while not shard_path.exists():
            time.sleep(SLEEP_POLL)
            waited += SLEEP_POLL
            if waited > MAX_WAIT:
                break
        if not shard_path.exists():
            continue

        if pred_path.exists():
            print(f"[rank {rank}] pred exists for {shard_path}, skipping.")
            continue

        try:
            shard_df = pd.read_csv(shard_path)
        except Exception as e:
            print(f"[rank {rank}] failed reading shard {shard_path}: {e}")
            continue

        texts = shard_df[args.text_col].fillna("").astype(str).tolist()
        outputs = {"is_job": [], "is_traffic": []}
        batch_size = args.infer_batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
        
        preds = classify_batch(batch_texts, model, tokenizer, gen_kwargs)
        for k in outputs.keys():
            outputs[k].extend(preds[k])

        preds_df = pd.DataFrame(outputs)
        preds_df.to_csv(pred_path, index=False)
        print(f"[rank {rank}] wrote preds {pred_path} ({len(preds_df)} rows)")

    print(f"[rank {rank}] worker loop done.")

def main_process_loop(accelerator, args):
    assert accelerator.process_index == 0
    world_size = accelerator.num_processes

    # enumerate CSVs
    csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*", "*.csv")))
    if len(csvs) == 0:
        csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.csv")))
    if len(csvs) == 0:
        print("[main] no CSVs found under", args.data_dir)
        return

    for csv_path in csvs:
        csv_path = Path(csv_path)
        print(f"[main] handling {csv_path}")
        # Read once to check header
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[main] failed to read {csv_path}: {e}; skipping")
            continue

        # checkpoint: skip if labels already present
        if all(c in df.columns for c in ["is_job", "is_traffic"]):
            print(f"[main] {csv_path} already labeled -> skipping")
            continue

        # Shard rows into world_size shards
        shards = shard_dataframe(df, world_size)
        shard_paths = []
        for r, shard_df in enumerate(shards):
            sp = csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{r}.csv")
            shard_df.to_csv(sp, index=False)
            shard_paths.append(sp)
            print(f"[main] wrote shard {sp} ({len(shard_df)} rows)")

        # Wait for preds from all ranks
        pred_paths = [csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{r}{PRED_SUFFIX}") for r in range(world_size)]
        waited = 0.0
        while True:
            all_done = all(p.exists() for p in pred_paths)
            if all_done:
                break
            time.sleep(SLEEP_POLL)
            waited += SLEEP_POLL
            if waited > MAX_WAIT:
                print(f"[main] timeout waiting preds for {csv_path}; found: {[p.exists() for p in pred_paths]}")
                break

        # Read preds (fill missing shards with zeros)
        preds_list = []
        for r, p in enumerate(pred_paths):
            if p.exists():
                try:
                    pf = pd.read_csv(p)
                except Exception as e:
                    print(f"[main] failed reading pred {p}: {e}")
                    pf = pd.DataFrame(columns=["is_job", "is_traffic"])
            else:
                # fallback zeros for shard length
                shard_len = len(shards[r])
                pf = pd.DataFrame([{"is_job":0,"is_traffic":0}] * shard_len)
            preds_list.append(pf)

        merged_preds = pd.concat(preds_list, ignore_index=True)

        # Ensure lengths match
        if len(merged_preds) != len(df):
            print(f"[main] merged preds length {len(merged_preds)} != df len {len(df)} -> adjusting")
            if len(merged_preds) > len(df):
                merged_preds = merged_preds.iloc[:len(df)]
            else:
                extra = pd.DataFrame([{"is_job":0,"is_traffic":0}] * (len(df)-len(merged_preds)))
                merged_preds = pd.concat([merged_preds, extra], ignore_index=True)

        df_out = df.copy()
        df_out["is_job"] = merged_preds["is_job"].astype(int)
        df_out["is_traffic"] = merged_preds["is_traffic"].astype(int)

        # Backup and write
        backup = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup.exists():
            csv_path.rename(backup)
            df_out.to_csv(csv_path, index=False)
            print(f"[main] wrote labeled CSV {csv_path} (backup at {backup})")
        else:
            df_out.to_csv(csv_path, index=False)
            print(f"[main] overwrote labeled CSV {csv_path} (existing backup kept)")

        # cleanup shard & pred files
        for p in shard_paths + pred_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    print("[main] all files processed.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="../data/raw/eda2/geo_inference_output")
    p.add_argument("--infer_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--text_col", type=str, default="cleaned")
    return p.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    print(f"[process {accelerator.process_index}] starting on device {accelerator.device} world_size={accelerator.num_processes}")
    # Phase 1: main creates shards for all CSVs (only main)
    if accelerator.process_index == 0:
        csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*", "*.csv")))
        if len(csvs) == 0:
            csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.csv")))
        for csv_path in csvs:
            csv_path = Path(csv_path)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            # Skip if already labeled
            if all(c in df.columns for c in ["is_job", "is_traffic"]):
                continue
            # shard and write files
            shards = shard_dataframe(df, accelerator.num_processes)
            for r, shard_df in enumerate(shards):
                sp = csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{r}.csv")
                shard_df.to_csv(sp, index=False)
        print("[main] all shards written; signaling workers")
    accelerator.wait_for_everyone()

    # Phase 2: all processes run inference on their rank-specific shards
    worker_loop(accelerator, args)

    accelerator.wait_for_everyone()

    # Phase 3: main merges predictions and writes final CSVs
    if accelerator.process_index == 0:
        main_process_loop(accelerator, args)

    accelerator.wait_for_everyone()
    print(f"[process {accelerator.process_index}] done.")

if __name__ == "__main__":
    main()
