from pymilvus import connections, db, Collection
from sentence_transformers import SentenceTransformer
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 0. Load your fine-tuned Mistral
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    "./mistral-finetuned", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "./mistral-finetuned", trust_remote_code=True
).to(device)

# -----------------------------
# 1. Connect to Milvus
# -----------------------------
_HOST, _PORT = "127.0.0.1", "19530"
def connect_to_milvus():
    connections.connect(host=_HOST, port=_PORT, timeout=60)
    if "turf_grass" not in db.list_database():
        db.create_database("turf_grass")
    db.using_database("turf_grass")

connect_to_milvus()
collection = Collection("turf_grass_data")

# -----------------------------
# 2. Embedding helper
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
def get_embedding(text):
    return embed_model.encode(text).tolist()

# -----------------------------
# 3. Your original askOllama
# -----------------------------
def askOllama(text):
    query_text = text
    query_embedding = get_embedding(query_text)
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    results = collection.search(
        data=[query_embedding],
        anns_field="paragraph_emb",
        param=search_params,
        limit=12,
        output_fields=["paragraph_emb","ids"]
    )

    ids_to_retrieve = []
    distance_of_id = []
    for result in results[0]:
        ids_to_retrieve.append(int(result.ids))
        distance_of_id.append(float(result.distance))

    # build context from SQLite and distances
    rows = [fetch_data_from_sqlite(r)[0][0] for r in ids_to_retrieve]
    context = "\n".join(rows).join(str(d) for d in distance_of_id)
    return context

# -----------------------------
# 4. SQLite fetch
# -----------------------------
def fetch_data_from_sqlite(ids):
    conn = sqlite3.connect("./final_output_completed.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT Paragraph_Contents, Table_contents FROM grass WHERE id = ?",
        (ids,)
    )
    row = cur.fetchall()
    conn.close()
    return row

# -----------------------------
# 5. Chat loop using Transformers
# -----------------------------
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        print("Exiting chat...")
        break

    context = askOllama(user_input)
    prompt = f"""Context:
{context}

User: {user_input}
Assistant:"""

    # tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = full[len(prompt):].strip()

    print("\nBot:", bot_reply)
