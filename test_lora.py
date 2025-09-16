# test_lora.py
import os
import sys
import torch
import warnings
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ----------------- SUPPRESS WARNINGS -----------------
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["BMM_FORCE_NO_BNB"] = "1"
warnings.filterwarnings("ignore")
_devnull = open(os.devnull, "w")

# ----------------- PATHS -----------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "./lora_out"       # adapter-only
MERGED_DIR = "./merged_model"    # fully merged model

# ----------------- LOAD TOKENIZER -----------------
if os.path.isdir(MERGED_DIR):
    model_path = MERGED_DIR
elif os.path.isdir(ADAPTER_DIR):
    model_path = ADAPTER_DIR
else:
    print("❌ ERROR: Neither merged_model nor lora_out found. Run train_lora.py first.")
    sys.exit(1)

with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------- LOAD MODEL -----------------
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if os.path.isdir(MERGED_DIR):
    print("🔄 Loading merged model...")
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
else:
    print("🔄 Loading base model + LoRA adapter...")
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, device_map="auto")

model.eval()

# ----------------- HELPER FUNCTIONS -----------------
def build_prompt(query: str) -> str:
    return f"### Instruction:\n{query.strip()}\n\n### Response:\n"

def ask_once(query: str, max_tokens: int = 150) -> str:
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("✅ Model loaded. Type 'exit' to quit.\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            ans = ask_once(q)
            print("\n", ans, "\n")
        except Exception as e:
            print("❌ Inference error:", str(e))
