# Code LLM Fine-Tuning Project

Fine-tune a 7B coding LLM on consumer GPU (24GB VRAM) using Unsloth + LoRA, then publish to Hugging Face.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Publishing](#publishing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

```
[Base Model] --LoRA Fine-Tune--> [Adapter] --Merge--> [Full Model] --Push--> [Hugging Face Hub]
```

LoRA (Low-Rank Adaptation) freezes the base model and injects trainable rank-decomposition matrices. This cuts VRAM usage from ~28GB to ~16GB for a 7B model. Unsloth further optimizes via manual kernels and gradient checkpointing, fitting the training into 24GB.

### Why Unsloth?

| Feature | Benefit |
|---|---|
| 4-bit QLoRA | Load model in 4-bit, train LoRA in 16-bit |
| Gradient checkpointing | Trade compute for memory |
| Flash Attention (optional) | 2x faster attention |
| Manual Triton kernels | ~2x speedup over HF PEFT |

### Why DeepSeek-Coder-6.7B?

- Trained on 2T tokens of code + natural language
- Fill-in-the-Middle (FIM) objective for code infilling
- State-of-the-art on HumanEval among 7B models
- Permissive MIT license

## Requirements

- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM) or A10G (24GB)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free (model + dataset)
- **CUDA**: 11.8+ or 12.1+
- **Python**: 3.10+

## Setup

### 1. Clone and install

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (select correct CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Unsloth
pip install "unsloth[cu118-torch250] @ git+https://github.com/unslothai/unsloth.git"

# Install other deps
pip install transformers datasets trl accelerate bitsandbytes gradio huggingface_hub
```

### 2. Hugging Face Login

```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

## Dataset

### Recommended Datasets

| Dataset | Examples | Use Case |
|---|---|---|
| `ise-uiuc/Magicoder_OSS-Instruct-75K` | 75K | General code instructions (recommended starter) |
| `bigcode/commitpackft` | 4M | Code edits from git commits |
| `m-a-p/CodeFeedback` | 10K | Preference pairs for DPO |
| `WizardLM/WizardCoder_evol_instruct_70k` | 70K | Evol-instruct code Q&A |

### Magicoder Dataset Format

Each example has:
```json
{
  "instruction": "Write a Python function to merge two sorted lists",
  "response": "def merge_sorted(l1, l2):\n    i = j = 0\n    result = []\n    ..."
}
```

The training script formats this as:
```
### Instruction:
{instruction}

### Response:
{response}
```

## Training

### How the Training Script Works

The training pipeline has 4 stages:

#### Stage 1: Model Loading

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    max_seq_length=4096,
    dtype=None,           # Auto-detect best dtype
    load_in_4bit=True,    # 4-bit quantization to save VRAM
)
```

- `load_in_4bit=True`: Uses bitsandbytes to quantize the model to 4-bit, reducing memory from ~14GB to ~4GB
- `max_seq_length=4096`: Truncates/pads sequences to 4096 tokens

#### Stage 2: LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                 # Rank of LoRA matrices
    target_modules=[...], # Which attention layers to adapt
    lora_alpha=16,        # Scaling factor
    lora_dropout=0,       # No dropout for LoRA (common practice)
)
```

LoRA decomposes weight updates as: `W' = W + BA` where:
- `W`: Original weights (frozen, 4-bit)
- `B`, `A`: Low-rank matrices (trainable, 16-bit)
- `r=16`: Rank of decomposition (higher = more capacity, more memory)
- `lora_alpha=16`: Controls how much the LoRA update affects the output

Only ~0.5% of parameters are trainable (~32M vs 6.7B), enabling fine-tuning on consumer hardware.

#### Stage 3: Dataset Preparation

```python
dataset = load_dataset("ise-uiuc/Magicoder_OSS-Instruct-75K", split="train")

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }
```

The SFTTrainer will automatically:
1. Tokenize each text
2. Apply chat template if configured
3. Mask the instruction portion (only train on response tokens)
4. Pad/truncate to `max_seq_length`

#### Stage 4: Training

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,   # Samples per GPU
        gradient_accumulation_steps=4,   # Effective batch = 2 * 4 = 8
        warmup_steps=5,                  # LR warmup
        max_steps=400,                   # Total training steps
        learning_rate=2e-4,              # Standard for LoRA
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",             # Memory-efficient optimizer
    ),
)
```

Key hyperparameters explained:
- **Effective batch size** = `per_device_train_batch_size * gradient_accumulation_steps * num_gpus` = 2 * 4 * 1 = 8
- **Learning rate 2e-4**: LoRA typically uses 1e-4 to 5e-4 (higher than full fine-tune's 1e-5)
- **max_steps=400**: ~1-2 hours on RTX 4090 for 75K dataset (1 epoch ≈ 2100 steps with batch size 8)
- **adamw_8bit**: 8-bit AdamW optimizer saves ~3GB VRAM vs 32-bit

### Running Training

```bash
python train.py
```

### Memory Usage Breakdown (RTX 4090 24GB)

| Component | Memory |
|---|---|
| Model (4-bit) | ~4 GB |
| LoRA weights (16-bit) | ~0.5 GB |
| Optimizer states (8-bit) | ~1 GB |
| Activations + gradients | ~8 GB |
| **Total** | **~14 GB** |

## Evaluation

### HumanEval

```bash
pip install evalplus

# Generate completions
python -c "
from evalplus.data import get_human_eval_plus
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('outputs/final', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('outputs/final')

results = []
for task in get_human_eval_plus():
    prompt = task['prompt']
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({'task_id': task['task_id'], 'completion': completion})

# Save and run evalplus
import json
with open('humaneval_results.json', 'w') as f:
    json.dump(results, f)
"

evalplus.evaluate --dataset humaneval --samples humaneval_results.json
```

## Publishing

### Option A: Push LoRA Adapter Only (small, ~64MB)

```bash
python push_to_hub.py --adapter-only
```

### Option B: Merge + Push Full Model (recommended, ~6.7GB)

```bash
python push_to_hub.py --merge
```

This creates:
- `README.md` with model card
- `config.json`, model weights, tokenizer files
- Automatically tagged on Hugging Face

## Deployment

### Hugging Face Space (Gradio)

1. Go to https://huggingface.co/new-space
2. Select "Gradio" SDK
3. Upload `app.py` and `requirements.txt`
4. Attach a T4 GPU (free tier available)

Or use the CLI:
```bash
cd space/
huggingface-cli repo create code-assistant-demo --space
git remote add space https://huggingface.co/spaces/your-username/code-assistant-demo
git add . && git commit -m "Initial demo" && git push space main
```

### Local Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "your-username/coding-assistant-v1",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("your-username/coding-assistant-v1")

prompt = "Write a Python function to check if a string is a palindrome"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Inference Parameters Explained

| Parameter | Effect |
|---|---|
| `temperature=0.2` | Low = deterministic, good for code |
| `top_p=0.95` | Nucleus sampling |
| `max_new_tokens=512` | Limit output length |
| `do_sample=True` | Enable sampling (vs greedy) |

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 2048
- Enable `use_gradient_checkpointing` (already on)
- Use `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`

### Slow Training
- Install Flash Attention: `pip install flash-attn --no-build-isolation`
- Add `--attn_implementation="flash_attention_2"` to model loading
- Ensure CUDA 12.1+ for optimal performance

### NaN Loss
- Reduce learning rate to 1e-4
- Ensure no special tokens missing from tokenizer
- Check dataset for empty responses

### Loss Not Decreasing
- Increase LoRA rank (r=32 or 64)
- Reduce `max_seq_length` if most sequences are short
- Increase learning rate to 3e-4 or 5e-4

## Advanced: DPO / ORPO Alignment

After SFT, you can align the model with preference data:

```python
from trl import DPOTrainer

# Load your SFT model
model = AutoModelForCausalLM.from_pretrained("outputs/final")

# Load preference dataset
dataset = load_dataset("m-a-p/CodeFeedback", split="train")

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create reference automatically
    args=TrainingArguments(
        per_device_train_batch_size=2,
        max_steps=200,
        learning_rate=5e-6,  # Lower LR for DPO
        output_dir="dpo_outputs",
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

## License

MIT
