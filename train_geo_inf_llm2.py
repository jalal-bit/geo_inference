import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import os, re, json, math, time, random
from accelerate import Accelerator,FullyShardedDataParallelPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from rapidfuzz import fuzz
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from pathlib import Path


from data_preperation import preprocess_and_save
# from datasets import load_dataset
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import wandb
import argparse
from datetime import datetime
from accelerate.utils import broadcast
from accelerate.utils import gather_object
from dotenv import load_dotenv




# --------------------------
# Load model + tokenizer
# --------------------------
def load_model_gemma(model_name, hf_token, using_accelerator=False, checkpoint_path=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the model source (either base model or checkpoint directory)
    model_source = checkpoint_path if checkpoint_path else model_name

    # Load model from checkpoint or model name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16, 
        device_map=None if using_accelerator else "auto",
        token=hf_token,
        attn_implementation="eager"
    )

    if checkpoint_path:
        print(f"✅ Loaded model and tokenizer from checkpoint: {checkpoint_path}")
    else:
        print(f"✅ Loaded base model: {model_name}")

    return model, tokenizer



def load_model_else(model_name, hf_token, using_accelerator=False, checkpoint_path=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the model source (either base model or checkpoint directory)
    model_source = checkpoint_path if checkpoint_path else model_name

    # Load model from checkpoint or model name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,  # L40S -> use bf16
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

    if checkpoint_path:
        print(f"✅ Loaded model and tokenizer from checkpoint: {checkpoint_path}")
    else:
        print(f"✅ Loaded base model: {model_name}")

    return model, tokenizer




def load_tokenizer(model_name, hf_token):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




class DataCollatorForCausalLM:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # Convert input_ids and prompt lengths to tensors
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        prompt_lens = torch.tensor([x["prompt_len"] for x in batch], dtype=torch.long)

        # Pad dynamically with left-padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="left"
        )

        batch_size, seq_len = input_ids.size()

        # Attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Labels: clone input_ids
        labels = input_ids.clone()

        # Compute left padding lengths
        total_tokens_per_row = attention_mask.sum(dim=1)  # number of non-pad tokens
        padding_lens = seq_len - total_tokens_per_row     # left padding length

        # Vectorized mask for prompt tokens
        range_tensor = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
        start_idx = padding_lens.unsqueeze(1)             # (batch_size, 1)
        prompt_mask = (range_tensor < start_idx + prompt_lens.unsqueeze(1))

        # Mask prompt tokens in labels
        labels = labels.masked_fill(prompt_mask, -100)

        # Truncate to max_length from the right
        input_ids = input_ids[:, -self.max_length:]
        attention_mask = attention_mask[:, -self.max_length:]
        labels = labels[:, -self.max_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



class DataCollatorForEval:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,padding_side="left")

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids[:, -self.max_length :],
            "attention_mask": attention_mask[:, -self.max_length :],
            "prompts": [x["prompt"] for x in batch],
            "targets": [x["target"] for x in batch],
        }




# --------------------------
# Parsing + Metrics
# --------------------------
def parse_struct(text: str) -> Dict[str, str]:
    if text is None:
        return {"country_type": None, "state": None, "county": None}
    t = text.lower()
    if "country: us" in t or re.search(r"\b(us)\b", t):
        ctype = "US"
    elif "country: non-us" in t or "non us" in t or "non-us" in t:
        ctype = "Non-US"
    else:
        ctype = "US" if "state:" in t or "county:" in t else None
    m_state = re.search(r"state:\s*([a-z\-\.\s]+)", t)
    state = m_state.group(1).strip() if m_state else None
    m_county = re.search(r"county:\s*([a-z\-\.\s]+)", t)
    county = m_county.group(1).strip() if m_county else None
    return {"country_type": ctype, "state": state, "county": county}


def fuzzy_equal(a: str, b: str, threshold=85) -> int:
    if a is None and b is None:
        return 1
    if a is None or b is None:
        return 0
    a = a.lower().strip()
    b = b.lower().strip()
    return 1 if fuzz.ratio(a, b) >= threshold else 0


def compute_validation_metrics(pred_texts: List[str], gold_texts: List[str], threshold=85) -> Dict[str, float]:
    preds = [parse_struct(x) for x in pred_texts]
    golds = [parse_struct(x) for x in gold_texts]

    y_true_us = [1 if g["country_type"] == "US" else 0 for g in golds]
    y_pred_us = [1 if p["country_type"] == "US" else 0 for p in preds]

    us_f1 = f1_score(y_true_us, y_pred_us, average="macro")
    us_acc = accuracy_score(y_true_us, y_pred_us)
    us_precision = precision_score(y_true_us, y_pred_us, average="macro", zero_division=0)

    state_matches = [fuzzy_equal(g["state"], p["state"], threshold) for p, g in zip(preds, golds)]
    county_matches = [fuzzy_equal(g["county"], p["county"], threshold) for p, g in zip(preds, golds)]

    state_acc = sum(state_matches) / max(1, len(state_matches))
    county_acc = sum(county_matches) / max(1, len(county_matches))

    state_f1 = f1_score([1]*len(state_matches), state_matches, average="macro")
    county_f1 = f1_score([1]*len(county_matches), county_matches, average="macro")
    state_precision = precision_score([1]*len(state_matches), state_matches, average="macro", zero_division=0)
    county_precision = precision_score([1]*len(county_matches), county_matches, average="macro", zero_division=0)

    return {
        "us_f1": us_f1, "us_acc": us_acc, "us_precision": us_precision,
        "state_f1": state_f1, "state_acc": state_acc, "state_precision": state_precision,
        "county_f1": county_f1, "county_acc": county_acc, "county_precision": county_precision,
    }



# --------------------------
# Evaluation (Accelerator-safe)
# --------------------------
def evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=60, log_first_batch=False):
    """
    Multi-GPU safe:
    - generate locally
    - slice continuations using left-padding prompt lengths
    - gather lists of strings with gather_object
    - compute metrics on main process only
    """
    model.eval()
    gen_cfg = {"num_beams": 1, "do_sample": False}
    gen_cfg["max_new_tokens"]=max_new_tokens
    unwrapped_model = accelerator.unwrap_model(model)

    # We'll collect everything on main at the end
    all_pred_texts = []
    all_gold_texts = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            gen_out = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg
            )

            # # Left padding => prompt length = number of non-pad tokens.
            # prompt_lens = attention_mask.sum(dim=1).tolist()

            # # Slice off the prompt tokens per row
            # local_pred_texts = [
            #     tokenizer.decode(gen_out[i, prompt_lens[i]:], skip_special_tokens=True)
            #     for i in range(gen_out.size(0))
            # ]
            cutoff = attention_mask.size(1)  # fixed length of input sequence
            local_pred_texts = [
                tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                for i in range(gen_out.size(0))
            ]

            local_gold_texts = batch["targets"]

            # Gather lists of strings across processes
            gathered_preds = gather_object(local_pred_texts)
            gathered_golds = gather_object(local_gold_texts)

            # Only main accumulates to avoid OOM/duplication
            if accelerator.is_main_process:
                # gather_object returns a list-of-lists (one per process); flatten it
                if isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred_texts.extend(sub)
                    for sub in gathered_golds:
                        all_gold_texts.extend(sub)
                else:
                    # single-process case
                    all_pred_texts.extend(gathered_preds)
                    all_gold_texts.extend(gathered_golds)

                if log_first_batch and step == 0:
                    print("\n[Val Sample]")
                    print("PROMPT  :", batch["prompts"][0][:200].replace("\n", " \\n "))
                    print("PREDICT :", local_pred_texts[0])
                    print("GOLD    :", local_gold_texts[0])

    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_validation_metrics(all_pred_texts, all_gold_texts)

    accelerator.wait_for_everyone()
    return metrics




# --------------------------
# Train loop (Accelerator-safe)
# --------------------------

def train_loop(model_name,train_dataset,val_dataset,batch_size, check_point_path, epochs=3,lr=2e-5,weight_decay=0.01,warmup_ratio=0.03,grad_accum_steps=1,max_grad_norm=1.0,gen_cfg=None,save_best_on="county_f1",wandb_project=None, wandb_config=None,timestamp_str=None,fsdp=True):
    

    load_dotenv()  # will read .env automatically

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("hf_token loaded")

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        print("wandb_key loaded")


    # Optim + sched
    if fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",   # options: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD
        cpu_offload=False,                # set True to save GPU memory, slower
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",  # automatically wrap transformer layers
        backward_prefetch="BACKWARD_PRE", # overlap compute + comm
        activation_checkpointing=True     # save memory
        )
        accelerator = Accelerator( mixed_precision="bf16",fsdp_plugin=fsdp_plugin)

    else:
        accelerator = Accelerator(mixed_precision="bf16")

    if isinstance(model_name, str):  # Ensure model_name is a string
        load_model = load_model_gemma if "gemma" in model_name else load_model_else
    else:
        raise ValueError("model_name must be a string")

    model,tokenizer=load_model(model_name,hf_token=hf_token,using_accelerator=True)

    if fsdp:
        model.gradient_checkpointing_enable()
    
    max_new_tokens=25

    if gen_cfg != None and gen_cfg.isnumeric():
        try:
            converted_integer = int(float(gen_cfg))
            if converted_integer>=25 and converted_integer<=60:
                max_new_tokens=converted_integer
        except ValueError:
             print(f"Error: Could not convert '{gen_cfg}' to an integer.")

    print(f"Setting max new token generation: {max_new_tokens}")



    
    train_collator = DataCollatorForCausalLM(tokenizer)
    val_collator = DataCollatorForEval(tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collator,num_workers=4, pin_memory=True)
    
    seed = 42 + accelerator.process_index
    set_seed(seed)

    
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)

    # Prepare everything for distributed
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # Scheduler needs total steps
    total_steps = math.ceil(len(train_loader) / max(1, grad_accum_steps)) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
   
    scheduler = accelerator.prepare(scheduler)

    best_metric = -1.0
    # --- Create output directories ---
    output_dir = os.path.join(check_point_path, f"{model_name}_{timestamp_str}")
    #output_dir = os.path.join(check_point_path, f"{model_name}")
    best_path = os.path.join(output_dir, "best")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if wandb_project and wandb_key and wandb_config:
            wandb.login(key=wandb_key)
            run_name=f"{model_name}_{timestamp_str}"
            wandb_config["max_new_tokens"]=max_new_tokens
            wandb.init(project=wandb_project,config=wandb_config,name=run_name)
            #wandb.init(project=wandb_project,config=wandb_config)

    accelerator.wait_for_everyone()
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        start = time.time()
        total_steps_per_epoch = len(train_loader)
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # track loss * batch_size for global averaging
                bsz = batch["input_ids"].size(0)
                epoch_loss_sum += loss.item() * bsz
                epoch_count += bsz

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                
                if step % 500 == 0 or step == total_steps_per_epoch - 1:
                    remaining = total_steps_per_epoch - step - 1
                    accelerator.print(
                        f"[Epoch {epoch+1}/{epochs}] "
                        f"Step {step+1}/{total_steps_per_epoch} "
                        f"({remaining} steps left) "
                    )
                

        # Compute global avg loss across processes
        loss_tensor = torch.tensor([epoch_loss_sum], dtype=torch.float32, device=accelerator.device)
        cnt_tensor  = torch.tensor([epoch_count],  dtype=torch.float32, device=accelerator.device)

        #gathered_loss = accelerator.gather_object([loss_tensor.item()])
        #gathered_cnt  = accelerator.gather_object([cnt_tensor.item()])
        gathered_loss = accelerator.gather_for_metrics(loss_tensor)
        gathered_cnt  = accelerator.gather_for_metrics(cnt_tensor)

        if accelerator.is_main_process:
            
            global_loss = gathered_loss.sum().item()
            global_cnt = gathered_cnt.sum().item()
            avg_loss = global_loss / global_cnt
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f} | elapsed {time.time()-start:.1f}s")
            if wandb_project and wandb_key and wandb_config:
                wandb.log({"train/loss": avg_loss})

        # Validation
        metrics = evaluate(model, val_loader, tokenizer, accelerator, max_new_tokens=max_new_tokens, log_first_batch=True)

        # Save best + periodic checkpoints
        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            key = save_best_on
            score = metrics.get(key, -1.0)
            if wandb_project and wandb_key and wandb_config:
                county_acc=metrics.get("county_acc", -1.0)
                county_precision=metrics.get("county_precision", -1.0)

                state_f1=metrics.get("state_f1", -1.0)
                state_precision=metrics.get("state_precision", -1.0)
                state_acc=metrics.get("state_acc", -1.0)

                us_f1=metrics.get("us_f1", -1.0)
                us_acc=metrics.get("us_acc", -1.0)
                us_precision=metrics.get("us_precision", -1.0)


                wandb.log({"epoch": epoch,"valid/county_f1": score, "valid/county_acc": county_acc,"valid/county_precision": county_precision, "valid/state_f1": state_f1, "valid/state_precision": state_precision, "valid/state_acc": state_acc, "valid/us_f1": us_f1, "valid/us_acc": us_acc,"valid/us_precision": us_precision})
 
                
            if score > best_metric:
                best_metric = score
                os.makedirs(best_path, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                with open(os.path.join(best_path, "val_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"✓ Saved new best to {best_path} (best {key}={best_metric:.4f})")

            # every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                with open(os.path.join(ckpt_dir, "val_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"💾 Saved checkpoint: {ckpt_dir}")

        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if wandb_project and wandb_key and wandb_config:
            wandb.finish() 




def parse_args():
    parser = argparse.ArgumentParser(description="Training with train_loop2")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--check_point_path", type=str, default="./checkpoints")
    parser.add_argument("--save_best_on", type=str, default="county_f1")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--gen_cfg", type=str, default=None)
    parser.add_argument("--timestamp", type=str, required=True)
    return parser.parse_args()



def main_cli():
    """SLURM / accelerate entrypoint."""
    args = parse_args()
    model_name = args.model_name  # replace with your model


    train_df_save_path=f"data/{model_name}/pretokenized/original_data/train_pretoken"

    valid_df_save_path=f"data/{model_name}/pretokenized/original_data/val_pretoken"

    train_ds=None
    val_ds=None

    if Path(train_df_save_path).exists():
        print("train data set exists")
        train_ds=load_from_disk(train_df_save_path)
    else:
        print("train data set does not exists")
        
    
    if Path(valid_df_save_path).exists():
        print("valid data set exists")
        val_ds=load_from_disk(valid_df_save_path)
    else:
        print("valid data set does not exist")

    

    if train_ds!=None and val_ds!=None:
       config_dict={"model_name":args.model_name,"bath_size":args.batch_size, "epochs":args.epochs, "learning_rate":args.lr, "weight_decay":args.weight_decay, "warmup_ratio": args.warmup_ratio, "grad_accum_steps":args.grad_accum_steps,"max_grad_norm":args.max_grad_norm}
       train_loop(args.model_name,train_ds,val_ds,args.batch_size,args.check_point_path, args.epochs, args.lr, args.weight_decay, args.warmup_ratio, args.grad_accum_steps, args.max_grad_norm, args.gen_cfg, args.save_best_on, args.wandb_project,config_dict,args.timestamp)

    else:
        print("Not starting training, train or validation pretokenized does not exist")



if __name__ == "__main__":
    main_cli()





