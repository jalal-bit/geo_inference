#!/usr/bin/env python3
"""
Single-sentence generation helper (ONE GPU, NO CPU offloading).

- Takes a prompt/sentence via CLI (--text)
- Loads base HF model (--model_name) OR a saved checkpoint from:
    <checkpoint_folder>/<model_name>_<checkpoint_path>/best
- Optional LoRA adapter via --use_lora_adapter
- Runs pure generation and prints to console (no metrics, no JSON)

Assumptions:
- You request exactly 1 GPU in Slurm (or set CUDA_VISIBLE_DEVICES to a single GPU).
- Model fits on that GPU (note: GPT-OSS 120B will NOT fit on 1x80GB unless quantized).
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# --------------------------
# Model loading (from your eval code; minimal adaptation)
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

    # ONE GPU only: we load with device_map=None then move to cuda:0
    if checkpoint_path and checkpoint_path.strip():
        best_dir = Path(checkpoint_folder) / f"{model_name}_{checkpoint_path}" / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_dir}")

        if use_lora_adapter:
            if PeftModel is None:
                raise RuntimeError("peft is not installed but --use_lora_adapter was set.")
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
            model = PeftModel.from_pretrained(base, str(best_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(best_dir),
                torch_dtype=torch.bfloat16,
                device_map=None,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
        print(f"Loaded checkpoint: {best_dir} (use_lora_adapter={use_lora_adapter})")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        print(f"Loaded base model: {model_name}")

    # Llama padding safety (kept from your code)
    if "llama" in model_name.lower():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Prompt/sentence to generate from.")
    p.add_argument("--model_name", required=True, help="HF model id (for tokenizer + base weights).")

    p.add_argument("--checkpoint_folder", default="checkpoints")
    p.add_argument("--checkpoint_path", default=None, help="Training timestamp used in <model_name>_<checkpoint_path>/best")
    p.add_argument("--use_lora_adapter", action="store_true", help="Load LoRA adapter from the /best folder via PEFT.")

    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only by design.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load on CPU first, then move to single GPU
    model, tok = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
        use_lora_adapter=args.use_lora_adapter,
    )

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    print("[INFO] Loaded on single GPU cuda:0")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")

    # Tokenize -> GPU
    enc = tok(args.text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        do_sample=args.do_sample,
    )
    if args.do_sample:
        gen_kwargs.update(dict(temperature=args.temperature, top_p=args.top_p))

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    cutoff = input_ids.size(1)
    gen_text = tok.decode(out[0, cutoff:], skip_special_tokens=True)

    print("\n===== PROMPT =====")
    print(args.text)
    print("\n===== GENERATION =====")
    print(gen_text)
    print("======================\n")


if __name__ == "__main__":
    main()
