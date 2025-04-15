import sqlite3
import pandas as pd
import json

# Connect to SQLite and read the data into a DataFrame
conn = sqlite3.connect('./final_output_completed.db')
df = pd.read_sql_query("SELECT id, Paragraph_Contents FROM grass;", conn)
conn.close()

def clean_text(text):
    """
    Escapes any inner double quotes in the text.
    This ensures that the JSON string remains valid.
    """
    if text is None:
        return ""
    return text.replace('"', '\\"')

# Open the JSONL file for writing
output_file = 'research_papers.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # Format the id with leading zeros (e.g., 1 becomes "001")
        id_str = str(row['id']).zfill(3)
        # Clean the paragraph_contents field
        cleaned_text = clean_text(row['Paragraph_Contents'])
        record = {
            "id": id_str,
            "Paragraph_Contents": cleaned_text
        }
        # Convert the dictionary to a JSON-formatted string and write to file
        json_line = json.dumps(record)
        f.write(json_line + "\n")

print(f"Data successfully written to {output_file}. ID 29 consistently gives issues. Feed to chat and ask for fix to copy and paste if still showing issues")
