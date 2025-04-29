from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

# -----------------------------
# 1. Load the Dataset
# -----------------------------
# Assumes your JSONL file has keys "id" and "Paragraph_Contents"
dataset = load_dataset("json", data_files={"train": "research_papers.jsonl"})

# -----------------------------
# 2. Load Model & Tokenizer
# -----------------------------
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# If your tokenizer lacks a pad token (e.g., LLaMA), set it manually
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# -----------------------------
# 3. Tokenize with ID Tracking
# -----------------------------
def tokenize_with_id(example):
    id_str = "ID: " + str(example["id"]) + "\n"
    full_text = id_str + example["Paragraph_Contents"]

    tokenized = tokenizer(full_text, truncation=False)
    id_tokenized = tokenizer(id_str, truncation=False)

    tokenized["id_token_count"] = len(id_tokenized["input_ids"])
    return tokenized

tokenized_dataset = dataset["train"].map(tokenize_with_id, batched=False)

# -----------------------------
# 4. Chunk Long Documents
# -----------------------------
def chunk_document(example, block_size=512):
    id_token_count = example["id_token_count"]
    id_tokens = example["input_ids"][:id_token_count]
    tokens = example["input_ids"]

    chunks = []
    for i in range(0, len(tokens), block_size):
        chunk = tokens[i : i + block_size]
        if i != 0:
            chunk = id_tokens + chunk
            chunk = chunk[:block_size]
        chunks.append(chunk)

    return {"input_ids": chunks, "labels": chunks}

chunked_dataset = tokenized_dataset.map(chunk_document, batched=False)

# -----------------------------
# 5. Flatten the Chunks
# -----------------------------
flattened_dataset = chunked_dataset.flatten()

# -----------------------------
# 6. Setup Data Collator & Trainer
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Important: we're training a causal LM
)

training_args = TrainingArguments(
    output_dir="./mistral7b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    report_to="none",  # disable W&B/HuggingFace reporting
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=flattened_dataset,
    data_collator=data_collator,
)

# -----------------------------
# 7. Train and Save the Model
# -----------------------------
trainer.train()
trainer.save_model("./minstral-finetuned")
tokenizer.save_pretrained("./minstral-finetuned")
