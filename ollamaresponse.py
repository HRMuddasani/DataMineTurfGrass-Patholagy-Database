from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import ollama
import sqlite3

_HOST = "127.0.0.1"  # Or localhost
_PORT = "19530"  # Default gRPC port for Milvus Standalone
# Connect to Milvus
def connect_to_milvus(db_name="default"):
    print(f"Connecting to Milvus...\n")
    
    # Connect using gRPC
    connections.connect(
        host=_HOST,  
        port=_PORT,        
        timeout=60  # Increased timeout
    )

    # List available databases
    db_list = db.list_database()
    print(db_list)

    # Check if the 'turf_grass' database exists
    if "turf_grass" not in db_list:
        db.create_database("turf_grass")
    
    db.using_database("turf_grass")
    print(f"Using database: turf_grass")
# Load your collection

connect_to_milvus()
collection_name = "turf_grass_data"
collection = Collection(collection_name)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
def get_embedding(text):
    """Generate an embedding for the query using Ollama."""
    response = model.encode(text).tolist()  # Adjust model as needed
    return response

# Query the database
def askOllama(text):
    query_text = text
    query_embedding = get_embedding(query_text)
      #  "index_type": "IVF_FLAT",
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    results = collection.search(
        data=[query_embedding], 
        anns_field="paragraph_contents",  # Adjust to match your schema
        param=search_params, 
        limit=5,
        output_fields=["paragraph_contents","ids"]
    )
    ids_to_retrieve = []
    for result in results[0]:
        print(f"ID: {result.ids}, Distance: {result.distance}")
        ids_to_retrieve.append(int(result.ids))  # Collect the IDs for further processing
    print(ids_to_retrieve)
    context = "\n".join([fetch_data_from_sqlite(res)[0][0] for res in ids_to_retrieve])  # Access the first row and the first column

    return context

def generateresponse(text) : # Generate a response using Ollama
    context = askOllama(text)
    response = ollama.chat(
        model="llama3.2",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are an expert in turfgrass and plant diseases."},
            {"role": "user", "content": f"Based on the following data, explain:\n{context}"}
        ]
    )

    print(response["message"]["content"])
def fetch_data_from_sqlite(ids):
    db_path = "./final_output_completed.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    query = "SELECT Paragraph_Contents FROM grass WHERE id = ?"
    print(cursor.execute(query, (ids,)))  # Tuple passed to parameterized query
    # Adjust query as needed
    row = cursor.fetchall()
    print(row)
    conn.close()
    return row

#ask question
generateresponse("Explain the effects of DMI fungicides on Bermuda grass")
