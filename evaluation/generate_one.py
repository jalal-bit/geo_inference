#!/usr/bin/env python3
"""
Single-sentence generation helper (ONE GPU, NO CPU offloading).

CLI args (ONLY):
  --text
  --model_name
  --checkpoint_folder
  --checkpoint_path

Everything else is hardcoded below.

UPDATE:
- Wraps the user-provided --text into a fixed extraction prompt format:
  You are given a social media post...
  Text: "..."
  JSON:
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
# HARDCODED SETTINGS
# --------------------------
USE_LORA_ADAPTER = True     # set True if you want PEFT adapter loading from /best
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 0

DTYPE = torch.bfloat16
GPU_DEVICE = "cuda:0"


# --------------------------
# Prompt wrapper
# --------------------------
PROMPT_TEMPLATE = (
    "You are given a social media post.\n"
    "Extract locations mentioned in the text for the U.S. states Texas, Alabama, and Arkansas only.\n"
    "Return a JSON object with these optional keys:\n"
    '  - "state": list of state abbreviations among ["TX","AL","AR"]\n'
    '  - "county": list of county names (no word "County")\n'
    '  - "city": list of city names\n'
    "If a key is unknown or not present in the text, omit that key.\n"
    "Do not include any extra text outside JSON.\n\n"
    'Text: "{text}"\n'
    "JSON:"
)


def wrap_text_as_prompt(text: str) -> str:
    # Keep it robust if user passes quotes/newlines
    cleaned = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Avoid breaking the Text: "..." line with unescaped double quotes
    cleaned = cleaned.replace('"', '\\"')
    return PROMPT_TEMPLATE.format(text=cleaned)


# --------------------------
# Model loading (minimal adaptation)
# --------------------------
def load_model_for_eval(
    model_name: str,
    hf_token: Optional[str],
    checkpoint_folder: str,
    checkpoint_path: Optional[str],
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_path and checkpoint_path.strip():
        best_dir = Path(checkpoint_folder) / f"{model_name}_{checkpoint_path}" / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_dir}")

        if USE_LORA_ADAPTER:
            if PeftModel is None:
                raise RuntimeError("peft is not installed but USE_LORA_ADAPTER=True.")
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=DTYPE,
                device_map=None,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
            model = PeftModel.from_pretrained(base, str(best_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(best_dir),
                torch_dtype=DTYPE,
                device_map=None,
                low_cpu_mem_usage=True,
                token=hf_token,
            )
        print(f"Loaded checkpoint: {best_dir} (USE_LORA_ADAPTER={USE_LORA_ADAPTER})")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map=None,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        print(f"Loaded base model: {model_name}")

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
    p.add_argument("--text", required=True, help="Raw social post text; will be wrapped into the extraction prompt.")
    p.add_argument("--model_name", required=True, help="HF model id (tokenizer + base weights).")
    p.add_argument("--checkpoint_folder", required=True, help="Base folder containing <model>_<checkpoint>/best")
    p.add_argument("--checkpoint_path", default=None, help="Training timestamp used in <model>_<checkpoint_path>/best")
    args = p.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only by design.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, tok = load_model_for_eval(
        model_name=args.model_name,
        hf_token=hf_token,
        checkpoint_folder=args.checkpoint_folder,
        checkpoint_path=args.checkpoint_path,
    )

    device = torch.device(GPU_DEVICE)
    model = model.to(device)
    model.eval()

    prompt = wrap_text_as_prompt(args.text)

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        do_sample=DO_SAMPLE,
    )
    if DO_SAMPLE:
        gen_kwargs.update(dict(temperature=TEMPERATURE, top_p=TOP_P))

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    cutoff = input_ids.size(1)
    gen_text = tok.decode(out[0, cutoff:], skip_special_tokens=True).strip()

    # Print ONLY JSON-ish output (no extra wrapper text),
    # but keep a fallback if model returns nothing.
    if not gen_text:
        gen_text = "{}"

    print(gen_text)


if __name__ == "__main__":
    main()
