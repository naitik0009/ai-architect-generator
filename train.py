import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "./qwen-kalkulio-lora"
DATA_DIR = "data"

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train.jsonl"),
    "validation": os.path.join(DATA_DIR, "valid.jsonl")
})

# 4. Load Model
print(f"Loading model {MODEL_ID} on GPU...")
# Using bfloat16 for RTX 6000 Blackwell
# We try to use flash_attention_2 if available, otherwise it falls back
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
except ValueError:
    print("Flash Attention 2 not installed/supported, falling back to default attention.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# 5. LoRA Configuration
# Target all linear layers for perfect fine-tuning
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Training Arguments
# Tailored for RTX PRO 6000 (48GB VRAM)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    bf16=True, # bfloat16 is supported natively on Blackwell
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch_fused"
)

# 7. Initialize SFTTrainer
# TRL's SFTTrainer will automatically apply the tokenizer's chat template 
# since the data has a 'messages' column prepared by prepare_data.py
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    args=training_args,
    max_seq_length=4096, # Adjust based on max floor plan string length
)

# 8. Train!
print("Starting training...")
trainer.train()

# 9. Save Final Model
print("Saving final model adapter...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("✅ Training complete!")
