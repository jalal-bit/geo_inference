#!/usr/bin/env python3
"""
Single-sentence generation helper.

- Takes a prompt/sentence via CLI (--text)
- Loads base HF model (--model_name) OR a saved checkpoint dir (--checkpoint_dir)
- Optional LoRA adapter dir (--lora_dir)
- Runs pure generation and prints to console
- Reports whether model is on CPU/GPU and device_map info (if available)

Notes:
- For huge models (e.g., 120B), prefer device_map="auto" with max_memory caps.
- If you pass --device cpu, it will try to load on CPU (may be very slow / may not fit).
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _build_max_memory(per_gpu_gb: int, allow_cpu: bool, cpu_gb: int = 96) -> Dict[Any, str]:
    n = torch.cuda.device_count()
    mm: Dict[Any, str] = {i: f"{per_gpu_gb}GiB" for i in range(n)}
    if allow_cpu:
        mm["cpu"] = f"{cpu_gb}GiB"
    return mm


def load_model_and_tokenizer(
    model_name: str,
    checkpoint_dir: Optional[str],
    lora_dir: Optional[str],
    hf_token: Optional[str],
    device: str,
    use_device_map: bool,
    per_gpu_gb: int,
    cpu_offload: bool,
    dtype: str,
):
    # ----- tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ----- choose source path -----
    base_path = checkpoint_dir if (checkpoint_dir and checkpoint_dir.strip()) else model_name
    if checkpoint_dir:
        if not Path(checkpoint_dir).exists():
            raise FileNotFoundError(f"--checkpoint_dir not found: {checkpoint_dir}")

    # ----- dtype -----
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    # ----- device placement -----
    if device == "cpu":
        # Force CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            token=hf_token,
        ).to("cpu")
    else:
        # GPU requested
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --device cuda was requested.")

        if use_device_map:
            max_memory = _build_max_memory(per_gpu_gb=per_gpu_gb, allow_cpu=cpu_offload)
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
        else:
            # Single GPU (device 0)
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch_dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                token=hf_token,
            ).to("cuda:0")

    # ----- optional LoRA -----
    if lora_dir and lora_dir.strip():
        if PeftModel is None:
            raise RuntimeError("peft is not installed but --lora_dir was provided.")
        if not Path(lora_dir).exists():
            raise FileNotFoundError(f"--lora_dir not found: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    return model, tok


def _infer_input_device(model, device: str):
    # If device_map was used, put inputs on the model's first parameter device.
    if device == "cpu":
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="Prompt/sentence to generate from.")
    p.add_argument("--model_name", required=True, help="HF model id (used for tokenizer and base weights if no checkpoint_dir).")

    # You can load from a saved checkpoint directory directly (full model saved with save_pretrained)
    p.add_argument("--checkpoint_dir", default=None, help="Path to a saved model dir (overrides base weights).")

    # Optional LoRA adapter directory (PEFT save_pretrained)
    p.add_argument("--lora_dir", default=None, help="Path to LoRA adapter dir (optional).")

    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Where to run generation.")
    p.add_argument("--use_device_map", action="store_true", help="If on CUDA, shard model across GPUs via device_map='auto'.")
    p.add_argument("--per_gpu_gb", type=int, default=68, help="Max GPU memory per GPU (GiB) when using device_map.")
    p.add_argument("--cpu_offload", action="store_true", help="Allow device_map to offload some modules to CPU.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto", help="Model dtype.")

    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    model, tok = load_model_and_tokenizer(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        lora_dir=args.lora_dir,
        hf_token=hf_token,
        device=args.device,
        use_device_map=args.use_device_map,
        per_gpu_gb=args.per_gpu_gb,
        cpu_offload=args.cpu_offload,
        dtype=args.dtype,
    )

    # Print placement info
    dm = getattr(model, "hf_device_map", None)
    if args.device == "cpu":
        print("[INFO] Loaded on CPU")
    else:
        if dm is not None:
            print(f"[INFO] Loaded with device_map='auto' across {torch.cuda.device_count()} GPUs")
        else:
            print("[INFO] Loaded on single GPU cuda:0")

    inp_dev = _infer_input_device(model, args.device)
    print(f"[INFO] Input tensors device: {inp_dev}")

    # Tokenize
    enc = tok(args.text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(inp_dev)
    attention_mask = enc["attention_mask"].to(inp_dev)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    # remove None keys for safety
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    # Decode only new tokens (continuation)
    cutoff = input_ids.size(1)
    gen_text = tok.decode(out[0, cutoff:], skip_special_tokens=True)

    print("\n===== PROMPT =====")
    print(args.text)
    print("\n===== GENERATION =====")
    print(gen_text)
    print("======================\n")


if __name__ == "__main__":
    main()
