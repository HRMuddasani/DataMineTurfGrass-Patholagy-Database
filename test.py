from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np

# Milvus connection details
_HOST = "127.0.0.1"  # Or localhost
_PORT = "19530"  # Default gRPC port for Milvus Standalone
model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
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

def create_collection():
    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary key
        FieldSchema(name="ids", dtype=DataType.INT16),
        FieldSchema(name="identifier_text", dtype=DataType.VARCHAR, max_length=4000),       # Unique identifier
        FieldSchema(name="identifier_emb", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="grass_Name", dtype=DataType.FLOAT_VECTOR, dim=384 ),
        FieldSchema(name="disease", dtype=DataType.FLOAT_VECTOR, dim=384 ),
        FieldSchema(name="pathogen", dtype=DataType.FLOAT_VECTOR, dim=384 ),
        FieldSchema(name="affiliation", dtype=DataType.FLOAT_VECTOR, dim=384 ),
        FieldSchema(name="paragraph_emb", dtype=DataType.FLOAT_VECTOR, dim=384), # Embedded vector data
        FieldSchema(name="table_emb", dtype=DataType.FLOAT_VECTOR, dim=384)      # Embedded vector data
    ]
    
    schema = CollectionSchema(fields=fields, description="Turf grass data collection")

    # Create the collection if it doesn't already exist
    collection_name = "turf_grass_data"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    return collection

def create_index(collection):
    index_params = {
        "metric_type": "IP",  
        "index_type": "IVF_FLAT",
        "params": {"nlist": 384}
    }

    index_fields = ["identifier_emb", "grass_Name", "disease", "pathogen", "affiliation", "paragraph_emb", "table_emb"]
    
    for field in index_fields:
        if not any(idx.field_name == field for idx in collection.indexes):
            collection.create_index(field_name=field, index_params=index_params)
            print(f"Index created on {field}.")
        else:
            print(f"Index already exists on {field}. Skipping.")



def fetch_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    cursor.execute(f"SELECT * FROM {table_name} ")  # Adjust query as needed
    row = cursor.fetchall()
    conn.close()
    return row

def embed_and_insert_data_from_db(collection, db_path, table_name):
    data_entries = fetch_data_from_sqlite(db_path, table_name)
    if not data_entries:
        print("No data found in the database.")
        return

    for row in data_entries:
        ids, identifier, paragraph_content, table_content, _, disease, pathogen, affiliation, _, grass_Name = row

        if identifier_exists(collection, ids):
            print(f"Identifier: '{identifier}' already exists. Skipping insert.")
            continue

        # Generate embeddings
        embedding_identifier = model.encode(identifier if identifier else "").tolist()
        embedding_paragraph = model.encode(paragraph_content if paragraph_content else "").tolist()
        embedding_table = model.encode(table_content if table_content else "").tolist()
        embedding_grass_Name = model.encode(grass_Name if grass_Name else "").tolist()
        embedding_disease = model.encode(disease if disease else "").tolist()
        embedding_pathogen = model.encode(pathogen if pathogen else "").tolist()
        embedding_affiliation = model.encode(affiliation if affiliation else "").tolist()


        # Insert data into Milvus
        collection.insert([
            [ids], [identifier], 
            [embedding_identifier],
            [embedding_paragraph], [embedding_table], 
            [embedding_grass_Name], [embedding_disease], 
            [embedding_pathogen], [embedding_affiliation]
        ])
        collection.flush()
        print(f"Inserted data for id: {ids}")

def retrieve_all_data(collection):
    # Load the collection into memory
    collection.load()
    
    # Query all data with a specified limit
    results = collection.query(expr="", limit=10)  # Empty expression to get all entries
    print("All data in the collection:")
    for result in results:
        print(result)

def identifier_exists(collection, ids_value):
    # Load collection into memory for querying
    collection.load()

    # Build an expression to query by the string field "ids"
    expr = f"ids == {ids_value}"
    
    # Return only the "identifier" field, limit=1 to just see if there's at least one match
    results = collection.query(expr=expr, output_fields=["ids"], limit=1)
    
    # If 'results' is not empty, we have a match
    return len(results) > 0


def print_collection_info(collection):
    print("=== Collection Info ===")
    print("Name:", collection.name)
    print("Description:", collection.description)
    print("Number of entities:", collection.num_entities)
    print("\nSchema:")
    for field in collection.schema.fields:
        print(f"  - {field.name}, type: {field.dtype}, is_primary: {field.is_primary}")

    print("\nIndexes:")
    if collection.indexes:
        for idx in collection.indexes:
            print(f"  - Field name: {idx.field_name}")
            print(f"    - Index type: {idx.index_name}")
            print(f"    - Params: {idx.params}")
    else:
        print("  No indexes found.")

    print("=======================\n")

def sanitize_identifier(identifier: str, mode="remove") -> str:
    """
    Removes or replaces single quotes in the given identifier string.
    mode="remove": deletes all single quotes.
    mode="double": replaces single quotes with double quotes.
    """
    if mode == "remove":
        # Completely remove single quotes
        return identifier.replace("'", "")
    elif mode == "double":
        # Replace single quotes with double quotes
        return identifier.replace("'", '"')
    else:
        return identifier  # No change if mode isn't recognized

# Paths and setup
db_path = "./final_output_completed.db"  # Path to the SQLite database
table_name = "grass"  # table name

# Connect to Milvus
connect_to_milvus()

# Create the collection
collection = create_collection()
#collection.drop()
# Create an index on the collection
create_index(collection)

# Embed and insert a single entry from SQLite
embed_and_insert_data_from_db(collection, db_path, table_name)


# Retrieve all data
retrieve_all_data(collection)

# Drop collection every run so no need to repeat entries
# Delete this for future uses
print_collection_info(collection)

