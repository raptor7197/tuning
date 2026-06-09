"""
Fine-tune a 7B coding LLM using Unsloth + LoRA.

Pipeline:
  1. Load base model in 4-bit (bitsandbytes)
  2. Attach LoRA adapters (~0.5% of params trainable)
  3. Load & format Magicoder code instruction dataset
  4. Train with SFTTrainer (only response tokens contribute to loss)
  5. Save LoRA adapters locally

Usage:
  python train.py
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ============================================================
# Stage 1: Load base model in 4-bit
# ============================================================
# - load_in_4bit: quantizes weights from 16-bit to 4-bit
#   (~75% memory reduction for the base model)
# - max_seq_length: sequences longer than this are truncated
# - dtype=None: auto-detect best precision (bf16 if supported)
# ============================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# ============================================================
# Stage 2: Attach LoRA adapters
# ============================================================
# r=16: Rank of the LoRA decomposition.
#   Rule of thumb: 8-64. Higher r = more capacity, more memory.
# target_modules: Which linear layers to adapt.
#   DeepSeek-Coder uses standard LLaMA architecture:
#   q_proj, k_proj, v_proj, o_proj (self-attention)
#   gate_proj, up_proj, down_proj (FFN)
# lora_alpha=16: Scaling factor. Often set to r.
#   The update is scaled by alpha/r.
# lora_dropout=0: Dropout on LoRA layers. 0 is standard for QLoRA.
# use_rslora=False: Rank-stabilized LoRA (variant). Usually not needed.
# ============================================================

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)

# ============================================================
# Stage 3: Load and format dataset
# ============================================================
# Magicoder_OSS-Instruct-75K contains 75K code instruction pairs
# scraped from open-source code on GitHub.
# Each example: {"instruction": "...", "response": "..."}
# We format into a simple prompt template that the model will
# learn to complete (only response tokens get loss).
# ============================================================

dataset = load_dataset("ise-uiuc/Magicoder_OSS-Instruct-75K", split="train")

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = dataset.map(format_prompt)

# ============================================================
# Stage 4: Configure and run training
# ============================================================
# Hyperparameter rationale:
# - per_device_train_batch_size=2: fits in 24GB with 4-bit + LoRA
# - gradient_accumulation_steps=4: effective batch = 2*4 = 8
#   Larger batch = more stable gradients
# - max_steps=400: ~2 hours on RTX 4090. Adjust for convergence.
# - learning_rate=2e-4: Standard for LoRA.
#   Full fine-tune uses ~2e-5, LoRA can use 1e-4 to 5e-4.
# - warmup_steps=5: Gradually increase LR to avoid instability.
# - optim="adamw_8bit": 8-bit optimizer saves ~3GB VRAM.
# - bf16/fp16: Mixed precision for faster training.
# ============================================================

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=400,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=42,
    ),
)

trainer.train()

# ============================================================
# Stage 5: Save LoRA adapters
# ============================================================
# This saves only the LoRA weights (~64MB) NOT the full model.
# To use, load base model + adapters:
#   model = FastLanguageModel.from_pretrained("deepseek-ai/...")
#   model.load_adapter("lora_model")
# ============================================================

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

print("Training complete. LoRA adapters saved to ./lora_model/")
