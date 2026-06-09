"""
Merge LoRA adapters into the base model and push to Hugging Face Hub.

Usage:
  python push_to_hub.py                     # Merge + push full model
  python push_to_hub.py --adapter-only       # Push just the LoRA adapter
  python push_to_hub.py --repo custom-name   # Custom repo name

Requires: huggingface-cli login (set HF_TOKEN env var or login first)
"""

import argparse
import os
from huggingface_hub import login, create_repo, upload_folder

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="coding-assistant-v1")
parser.add_argument("--adapter-only", action="store_true")
parser.add_argument("--username", default=None)
args = parser.parse_args()

# Login if token available
if token := os.getenv("HF_TOKEN"):
    login(token=token)
else:
    login()  # interactive

# Determine username
from huggingface_hub import whoami
username = args.username or whoami()["name"]
repo_id = f"{username}/{args.repo}"

# Create repo (won't error if exists)
create_repo(repo_id, exist_ok=True)

if args.adapter_only:
    # Push only the LoRA adapter (small, ~64MB)
    print(f"Pushing LoRA adapter to {repo_id} ...")
    upload_folder(folder_path="lora_model", repo_id=repo_id)
    print(f"Done! View at https://huggingface.co/{repo_id}")
else:
    # Merge LoRA into base model and push full model
    print("Merging LoRA into base model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        max_seq_length=4096,
        load_in_4bit=True,
    )
    model.load_adapter("lora_model")

    print("Merging and unloading...")
    model = model.merge_and_unload()

    print(f"Saving merged model to ./merged_model/ ...")
    model.save_pretrained("merged_model")
    tokenizer.save_pretrained("merged_model")

    print(f"Pushing merged model to {repo_id} ...")
    upload_folder(folder_path="merged_model", repo_id=repo_id)
    print(f"Done! View at https://huggingface.co/{repo_id}")
