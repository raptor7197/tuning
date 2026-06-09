"""
Gradio demo for Hugging Face Space.

This runs on HF's free T4 GPU tier.
The Space auto-detects the model from the HF_HUB_TOKEN env var.
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = os.getenv("MODEL_REPO", "your-username/coding-assistant-v1")

print(f"Loading model from {REPO_ID} ...")
model = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
print("Model loaded!")

SYSTEM_PROMPT = "You are an expert programming assistant. Write clean, efficient code."


def generate(prompt, temperature=0.2, max_new_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


with gr.Blocks(title="Code Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Code Assistant")
    gr.Markdown(f"Fine-tuned from `{REPO_ID}`")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Write a Python function to merge two sorted lists...",
                lines=5,
            )
            with gr.Row():
                temp = gr.Slider(0.0, 1.0, value=0.2, label="Temperature")
                tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
            submit = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            output = gr.Code(label="Output", language="python", lines=20)

    submit.click(fn=generate, inputs=[prompt, temp, tokens], outputs=output)

if __name__ == "__main__":
    demo.launch()
