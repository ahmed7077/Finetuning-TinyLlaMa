# train_lora.py
# Robust Windows-safe LoRA training + auto-merge (no bitsandbytes, no meta-tensor move)
import os
import sys
import json
import warnings
import contextlib
# -------------------------
# Silence environment
# -------------------------
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
_devnull = open(os.devnull, "w")

# -------------------------
# Quiet imports (suppress banners)
# -------------------------
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig, get_peft_model, PeftModel
# -------------------------
# Config
# -------------------------
DATA_PATH = "people_osl.jsonl"
OUT_DIR = "./lora_out"
MERGED_DIR = "./merged_model"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 256
NUM_EPOCHS = 5

# -------------------------
# Load dataset
# -------------------------
if not os.path.exists(DATA_PATH):
    print(f"ERROR: dataset file not found at {DATA_PATH}")
    sys.exit(0)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

if len(samples) == 0:
    print("ERROR: dataset empty")
    sys.exit(0)
if not ("instruction" in samples[0] and "output" in samples[0]):
    print("ERROR: dataset JSONL must have 'instruction' and 'output' fields")
    sys.exit(0)

dataset = Dataset.from_list(samples)

# -------------------------
# formatting_func
# -------------------------
def _format_single(i, o):
    i = i.strip() if isinstance(i, str) else str(i)
    o = o.strip() if isinstance(o, str) else str(o)
    return f"### Instruction:\n{i}\n\n### Response:\n{o}"

def formatting_func(batch):
    instr = batch.get("instruction")
    out = batch.get("output")
    if isinstance(instr, list):
        return [_format_single(a, b) for a, b in zip(instr, out)]
    else:
        return [_format_single(instr, out)]

# -------------------------
# Tokenizer
# -------------------------
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Model load strategy
# -------------------------
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=False
    )

# -------------------------
# Apply LoRA (PEFT)
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    model = get_peft_model(model, lora_config)

if torch_device == "cuda":
    try:
        model = model.to(torch_device)
    except Exception as e:
        print("Warning: moving model to CUDA failed, continuing on CPU. Error:", e)
        torch_device = "cpu"

# -------------------------
# TrainingArguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=False,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_drop_last=False,
)

# -------------------------
# SFTTrainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=training_args,
)

# -------------------------
# Train with OOM handling
# -------------------------
try:
    trainer.train()
except RuntimeError as e:
    msg = str(e).lower()
    if "out of memory" in msg:
        print("OOM detected. Suggest lowering batch size/seq length. Exiting.")
        sys.exit(0)
    else:
        print("Runtime error during training:", e)
        sys.exit(0)

# -------------------------
# Save adapter + tokenizer
# -------------------------
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

print("✅ Training finished. Adapter saved to:", OUT_DIR)

# -------------------------
# Merge LoRA into base model automatically
# -------------------------
print("🔄 Merging LoRA adapter into base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=False
)
merged = PeftModel.from_pretrained(base_model, OUT_DIR)
merged = merged.merge_and_unload()

merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Merged model saved to:", MERGED_DIR)
