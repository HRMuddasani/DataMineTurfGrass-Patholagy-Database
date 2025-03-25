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
        anns_field="paragraph_emb",  # Adjust to match your schema
        param=search_params, 
        limit = 12 ,
        output_fields=["paragraph_emb","ids"]
    )
    ids_to_retrieve = []
    distance_of_id = []
    for result in results[0]:
        #print(f"ID: {result.ids}, Distance: {result.distance}")
        ids_to_retrieve.append(int(result.ids))
        distance_of_id.append(float(result.distance)) # Collect the IDs for further processing
    #print(ids_to_retrieve)
    #print(len(ids_to_retrieve))
    context = "\n".join([fetch_data_from_sqlite(res)[0][0] for res in ids_to_retrieve]).join(str(dist) for dist in distance_of_id)  # Access the first row and the first column

    return context
def askOllama(text,num):
    query_text = text
    query_embedding = get_embedding(query_text)
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    # Fields to check for common IDs
    fields_to_search = [
        "identifier_emb", 
        "grass_Name", 
        "disease", 
        "pathogen", 
        "affiliation", 
        "paragraph_emb", 
        "table_emb"
    ]

    all_ids = {field: [] for field in fields_to_search}  # Dictionary to store IDs for each field

    # Perform search on each field
    for field in fields_to_search:
        results = collection.search(
            data=[query_embedding], 
            anns_field=field, 
            param=search_params, 
            limit=25,  # Getting 10 results for each field
            output_fields=["ids"]
        )

        # Collect IDs from the results of each field
        for result in results[0]:
            all_ids[field].append(int(result.ids))
    
    # Now, we need to count the number of occurrences of each ID across the fields
    id_count = {}
    
    # Count occurrences of each ID across all fields
    for field, ids in all_ids.items():
        for id in ids:
            if id in id_count:
                id_count[id] += 1
            else:
                id_count[id] = 1

    # Filter IDs that appear in more than 3 fields
    common_ids = [id for id, count in id_count.items() if count > 3]

    if common_ids:
        ids_to_retrieve = common_ids
        print(common_ids)
    else:
        # If no IDs appear in more than 3 fields, fall back to the paragraph_emb results
        ids_to_retrieve = all_ids["paragraph_emb"][:5]

    # Fetch the data for the selected IDs
    context = "\n".join([fetch_data_from_sqlite(res)[0][0] for res in ids_to_retrieve])

    return context
def askOllama(text, num, num2):
    query_text = text
    query_embedding = get_embedding(query_text)
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    # Fields to check for common IDs
    fields_to_search = [
        "identifier_emb", 
        "grass_Name", 
        "disease", 
        "pathogen", 
        "affiliation", 
        "paragraph_emb", 
        "table_emb"
    ]

    all_ids = {field: [] for field in fields_to_search}  # Dictionary to store (ID, distance) pairs

    # Perform search on each field
    for field in fields_to_search:
        results = collection.search(
            data=[query_embedding], 
            anns_field=field, 
            param=search_params, 
            limit=25,  # Getting 25 results for each field
            output_fields=["ids"]
        )

        # Collect (ID, distance) tuples from the results
        for result in results[0]:
            all_ids[field].append((int(result.ids), float(result.distance)))

    # Now, we need to count occurrences and track the best (lowest) distance for each ID
    id_count = {}
    id_min_distance = {}

    # Count occurrences of each ID across all fields and track the smallest distance
    for field, id_distance_pairs in all_ids.items():
        for id, distance in id_distance_pairs:
            if id in id_count:
                id_count[id] += 1
                id_min_distance[id] = min(id_min_distance[id], distance)
            else:
                id_count[id] = 1
                id_min_distance[id] = distance

    # Filter IDs that appear in more than 3 fields, keeping the min distance
    common_ids_with_distances = [
        (id, id_min_distance[id]) for id, count in id_count.items() if count > 3
    ]
    print(common_ids_with_distances)
    if common_ids_with_distances:
        ids_to_retrieve = common_ids_with_distances
    else:
        # If no IDs appear in more than 3 fields, fall back to the paragraph_emb results
        ids_to_retrieve = all_ids["paragraph_emb"][:5]  # This already stores (ID, distance) tuples

    # Sort by distance (ascending, so closest matches come first)
    ids_to_retrieve.sort(key=lambda x: x[1])

    return ids_to_retrieve
def askOllama(text, num, num2, distance_threshold):  # Add distance threshold parameter
    query_text = text
    query_embedding = get_embedding(query_text)
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    # Fields to check for common IDs
    fields_to_search = [
        "identifier_emb", 
        "grass_Name", 
        "disease", 
        "pathogen", 
        "affiliation", 
        "paragraph_emb", 
        "table_emb"
    ]

    all_ids = {field: [] for field in fields_to_search}  # Dictionary to store (ID, distance) pairs

    # Perform search on each field
    for field in fields_to_search:
        results = collection.search(
            data=[query_embedding], 
            anns_field=field, 
            param=search_params, 
            limit=25,  # Getting 25 results for each field
            output_fields=["ids"]
        )

        # Collect (ID, distance) tuples from the results
        for result in results[0]:
            all_ids[field].append((int(result.ids), float(result.distance)))

    # Now, we need to count occurrences and track the best (lowest) distance for each ID
    id_count = {}
    id_min_distance = {}

    # Count occurrences of each ID across all fields and track the smallest distance
    for field, id_distance_pairs in all_ids.items():
        for id, distance in id_distance_pairs:
            if id in id_count:
                id_count[id] += 1
                id_min_distance[id] = min(id_min_distance[id], distance)
            else:
                id_count[id] = 1
                id_min_distance[id] = distance

    # Filter IDs that appear in more than 3 fields AND have a distance below the threshold
    common_ids_with_distances = [
        (id, id_min_distance[id]) 
        for id, count in id_count.items() 
        if count > 3 and id_min_distance[id] <= distance_threshold
    ]

    if common_ids_with_distances:
        ids_to_retrieve = common_ids_with_distances
    else:
        # If no common IDs exist, fall back to paragraph_emb results (within threshold)
        ids_to_retrieve = [
            (id, distance) for id, distance in all_ids["paragraph_emb"][:5] 
            if distance <= distance_threshold
        ]

    # Sort by distance (ascending, so closest matches come first)
    ids_to_retrieve.sort(key=lambda x: x[1])

    return ids_to_retrieve



def generateresponse(text) : # Generate a response using Ollama
    print("Question:")
    print(text)
    print("Answer:")
    context = askOllama(text)
    response = ollama.chat(
        model="llama3.2",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are an expert in turfgrass and plant diseases."},
            {"role": "user", "content": f"answer {text}? , based on - {context}"}
        ]
    )

    print(response["message"]["content"])
def fetch_data_from_sqlite(ids):
    db_path = "./final_output_completed.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    query = "SELECT Paragraph_Contents,Table_contents FROM grass WHERE id = ?"
    cursor.execute(query, (ids,))  # Tuple passed to parameterized query
    # Adjust query as needed
    row = cursor.fetchall()
    conn.close()
    return row

#ask question
chat_history = [
    {"role": "system", "content": "You are an expert in turfgrass and plant diseases."}
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    context = askOllama(user_input,1,2,0.42)

    chat_history.append({"role": "user", "content": user_input})
    
    # Instead of just appending context, structure the prompt for a better response
    formatted_prompt = f"""
    Context: {context}
    
    User: {user_input}
    Bot:"""

    response = ollama.chat(model="llama3.2", messages=chat_history + [{"role": "user", "content": formatted_prompt}])
    
    bot_reply = response["message"]["content"]
    print("\nBot:", bot_reply)

    chat_history.append({"role": "assistant", "content": bot_reply})
