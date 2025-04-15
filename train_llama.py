from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# -----------------------------
# 1. Load the dataset
# -----------------------------
# Assumes your JSONL file has keys "id" and "Paragraph_Contents"
dataset = load_dataset("json", data_files={"train": "research_papers.jsonl"})

# Set your model name (update this with your actual model identifier)
model_name = "your-llama-3.2-model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# -----------------------------
# 2. Tokenize each example with its ID prepended
# -----------------------------
def tokenize_with_id(example):
    # Prepend the document ID and a newline to the document text
    full_text = "ID: " + str(example["id"]) + "\n" + example["Paragraph_Contents"]
    # Tokenize the full text without truncation
    return tokenizer(full_text, truncation=False)

# Apply the tokenization function to every example
tokenized_dataset = dataset.map(tokenize_with_id, batched=False)

# -----------------------------
# 3. Define a chunking function
# -----------------------------
def chunk_document(example, block_size=512, id_token_count=10):
    """
    Splits a tokenized document (example) into chunks of 'block_size' tokens.
    The first 'id_token_count' tokens (assumed to be the ID portion) are saved
    and re-prepended to every chunk that doesn't start at the beginning.
    """
    # Save the ID tokens (assume they occupy the first id_token_count tokens)
    id_tokens = example["input_ids"][:id_token_count]
    tokens = example["input_ids"]
    chunks = []
    # Loop over tokens in increments of block_size
    for i in range(0, len(tokens), block_size):
        chunk = tokens[i : i + block_size]
        # If this is not the first chunk, re-prepend the id tokens
        if i != 0:
            chunk = id_tokens + chunk
            # If this makes the chunk too long, trim it to block_size tokens
            chunk = chunk[:block_size]
        chunks.append(chunk)
    # Return a dictionary with the list of chunks; labels are identical for LM training
    return {"input_ids": chunks, "labels": chunks}

# Apply the chunking function to each tokenized example.
# Here we use a lambda to pass our parameters for block size and id_token_count.
chunked_dataset = tokenized_dataset.map(
    lambda ex: chunk_document(ex, block_size=512, id_token_count=10),
    batched=False
)

# -----------------------------
# 4. Flatten the dataset so that each chunk is an independent example
# -----------------------------
# flatten() converts a dataset with lists of chunks into a dataset of individual examples.
chunked_dataset = chunked_dataset.flatten()

# (Optional) Print one example to inspect its structure
print(chunked_dataset[0])

# -----------------------------
# 5. Set up training (using Trainer)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./llama3.2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    fp16=True,  # Optional: enable mixed precision if supported
)

# If your chunked_dataset is a Dataset object (i.e. not split into "train"/"test"), you can pass it directly.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,  # each example is a chunk now
)

# Fine-tune the model
trainer.train()
trainer.save_model("./llama3.2-finetuned")
