# test_lora.py
import os
import sys
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import contextlib

# ----------------- SUPPRESS WARNINGS -----------------
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["BMM_FORCE_NO_BNB"] = "1"  # suppress bitsandbytes completely
warnings.filterwarnings("ignore")
_devnull = open(os.devnull, "w")

# ----------------- PATHS -----------------
ADAPTER_DIR = "./lora_out"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if not os.path.isdir(ADAPTER_DIR):
    print("ERROR: Adapter directory not found. Run train_lora.py first.")
    sys.exit(1)

# ----------------- LOAD TOKENIZER -----------------
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------- LOAD MODEL -----------------
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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
    print("Model loaded. Type 'exit' to quit.\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            ans = ask_once(q)
            print("\n", ans, "\n")
        except Exception as e:
            print("Inference error:", str(e))