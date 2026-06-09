"""
Quick validation: runs the fine-tuning pipeline end-to-end on CPU
with a tiny model and mini dataset slice.

Replace with Unsloth + 7B on a GPU for real training.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTTrainer

# Tiny model for CPU validation
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print(f"Loading {MODEL_NAME} (CPU, float32) ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config (standard PEFT, same concept as Unsloth)
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Create tiny synthetic dataset
dataset = Dataset.from_list([
    {"instruction": "Write a Python function to add two numbers", "response": "def add(a, b):\n    return a + b"},
    {"instruction": "Write hello world in Python", "response": "print('Hello, World!')"},
    {"instruction": "Write a function to check if a number is even", "response": "def is_even(n):\n    return n % 2 == 0"},
    {"instruction": "Write a function to square a number", "response": "def square(x):\n    return x * x"},
    {"instruction": "Write a function to find max of two numbers", "response": "def max_of_two(a, b):\n    return a if a > b else b"},
])

def formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

print(f"Training on {len(dataset)} examples ...")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        output_dir="/tmp/test-outputs",
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    ),
)

trainer.train()

# Save
model.save_pretrained("/tmp/test-lora")
tokenizer.save_pretrained("/tmp/test-lora")

# Quick inference test
model.eval()
prompt = "### Instruction:\nWrite hello world in Python\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)
print("\n=== Inference test ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
print("\nPipeline validated successfully!")
