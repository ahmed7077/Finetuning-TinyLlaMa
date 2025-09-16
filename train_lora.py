import os
import sys
import json
import argparse
import warnings
import contextlib
import inspect

# -------------------------
# Quiet noisy imports (optional)
# -------------------------
_devnull = open(os.devnull, "w")
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    # TRL and PEFT
    from trl import SFTTrainer
    try:
        # newer TRL exposes a useful collator for completion-only LM loss
        from trl import DataCollatorForCompletionOnlyLM
        _has_completion_collator = True
    except Exception:
        DataCollatorForCompletionOnlyLM = None
        _has_completion_collator = False
    from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore")

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="people_osl.jsonl")
parser.add_argument("--out_dir", type=str, default="./lora_out")
parser.add_argument("--merged_dir", type=str, default="./merged_model")
parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--fp16", action="store_true", help="use fp16 if CUDA is available")
parser.add_argument("--no_merge", action="store_true", help="skip automatic merging step")
args = parser.parse_args()

DATA_PATH = args.data_path
OUT_DIR = args.out_dir
MERGED_DIR = args.merged_dir
BASE_MODEL = args.base_model
MAX_SEQ_LENGTH = args.max_seq_length
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.per_device_train_batch_size
GRAD_ACC = args.gradient_accumulation_steps
LR = args.learning_rate
USE_FP16 = args.fp16 and torch.cuda.is_available()

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
if not os.path.exists(DATA_PATH):
    print(f"ERROR: dataset file not found at {DATA_PATH}")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

if len(samples) == 0:
    print("ERROR: dataset empty")
    sys.exit(1)

if not ("instruction" in samples[0] and "output" in samples[0]):
    print("ERROR: dataset JSONL must have 'instruction' and 'output' fields")
    sys.exit(1)

dataset = Dataset.from_list(samples)
print(f"Loaded dataset with {len(dataset)} examples")

# -------------------------
# formatting_func (robust)
# - returns a single string for single-example calls
# - returns list[str] when called in batched mode
# -------------------------

def _format_single(i, o):
    i = i.strip() if isinstance(i, str) else str(i)
    o = o.strip() if isinstance(o, str) else str(o)
    return f"### Instruction:\n{i}\n\n### Response:\n{o}"


def formatting_func(example_or_batch):
    """Accepts either a single example dict or a batch (dict of lists).
    For single examples, return a string. For batched input, return a list of strings.
    This keeps compatibility with different TRL versions that may call the function batched or not.
    """
    instr = example_or_batch.get("instruction")
    out = example_or_batch.get("output")

    # batched call
    if isinstance(instr, list) or isinstance(out, list):
        # zip to avoid length mismatch
        out_list = []
        for a, b in zip(instr, out):
            out_list.append(_format_single(a, b))
        return out_list

    # single example
    return _format_single(instr, out)

# -------------------------
# Tokenizer
# -------------------------
print("Loading tokenizer from:", BASE_MODEL)
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Model load
# -------------------------
print("Loading base model (this may take a while)...")
model_dtype = torch.float16 if USE_FP16 else torch.float32
with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

# -------------------------
# Apply LoRA (PEFT) - if you prefer, you can instead pass peft_config to SFTTrainer
# -------------------------
print("Applying LoRA (PEFT) adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
    model = get_peft_model(model, lora_config)

# Move to CUDA if available
if torch.cuda.is_available():
    try:
        model = model.to("cuda")
        print("Moved model to CUDA")
    except Exception as e:
        print("Warning: moving model to CUDA failed; continuing on CPU. Error:", e)

# -------------------------
# TrainingArguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    fp16=USE_FP16,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_drop_last=False,
)

# -------------------------
# Data collator (prefer TRL's completion collator, fallback to HF collator)
# -------------------------
if _has_completion_collator and DataCollatorForCompletionOnlyLM is not None:
    try:
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)
        print("Using trl.DataCollatorForCompletionOnlyLM")
    except Exception:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        print("Falling back to transformers.DataCollatorForLanguageModeling")
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Using transformers.DataCollatorForLanguageModeling (fallback)")

# -------------------------
# Build SFTTrainer kwargs robustly by inspecting the signature at runtime
# -------------------------
candidate_kwargs = dict(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
    packing=False,
    data_collator=data_collator,
)

# Inspect SFTTrainer.__init__ signature and only keep supported keys
sig = inspect.signature(SFTTrainer.__init__)
param_names = [p for p in sig.parameters.keys() if p != "self"]
trainer_kwargs = {k: v for k, v in candidate_kwargs.items() if k in param_names}

# If the SFTTrainer expects an SFTConfig instead of TrainingArguments, try to construct it
if "sft_config" in param_names and "args" not in param_names:
    try:
        from trl import SFTConfig

        sft_cfg = SFTConfig(
            output_dir=OUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC,
            learning_rate=LR,
        )
        trainer_kwargs["sft_config"] = sft_cfg
        # remove args if present
        trainer_kwargs.pop("args", None)
        print("Using TRL SFTConfig for trainer configuration")
    except Exception:
        # fallback: leave args in place if we couldn't construct SFTConfig
        pass

print("Initializing SFTTrainer with these keys:", list(trainer_kwargs.keys()))

# -------------------------
# Initialize trainer (robust)
# -------------------------
try:
    trainer = SFTTrainer(**trainer_kwargs)
except TypeError as e:
    # As a last resort, try without the data_collator (older/newer API mismatch)
    print("SFTTrainer init TypeError:", e)
    if "data_collator" in trainer_kwargs:
        print("Retrying without data_collator to support older TRL versions...")
        trainer_kwargs.pop("data_collator")
        trainer = SFTTrainer(**trainer_kwargs)
    else:
        raise

# -------------------------
# Train with OOM handling
# -------------------------
try:
    print("Starting training...")
    trainer.train()
except RuntimeError as e:
    msg = str(e).lower()
    if "out of memory" in msg or "cuda out of memory" in msg:
        print("OOM detected. Suggestions: lower per_device_train_batch_size, reduce max_seq_length, or increase gradient_accumulation_steps.")
        sys.exit(1)
    else:
        print("Runtime error during training:", e)
        raise

# -------------------------
# Save adapter + tokenizer
# -------------------------
try:
    trainer.model.save_pretrained(OUT_DIR)
except Exception:
    # fallback
    model.save_pretrained(OUT_DIR)

tokenizer.save_pretrained(OUT_DIR)
print("✅ Training finished. Adapter saved to:", OUT_DIR)

# -------------------------
# Merge LoRA into base model automatically (if requested)
# -------------------------
if not args.no_merge:
    print("🔄 Merging LoRA adapter into base model (may require extra RAM)...")
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

    try:
        merged = PeftModel.from_pretrained(base_model, OUT_DIR)
        merged = merged.merge_and_unload()
        merged.save_pretrained(MERGED_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MERGED_DIR)
        print("✅ Merged model saved to:", MERGED_DIR)
    except Exception as e:
        print("⚠️  Merge failed:", e)
        print("You can still use the adapter at", OUT_DIR, "or try merging locally where you may have more RAM.")

print("All done.")
