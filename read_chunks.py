import requests
import json

# Load data from data.json
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list
    })

    embedding = r.json()['embeddings']
    return embedding


embeddings = create_embedding([c['text'] for c in data])

# Add embeddings and chunk IDs to each data object
for i, (chunk, embedding) in enumerate(zip(data, embeddings)):
    chunk['chunk_id'] = i
    chunk['embedding'] = embedding

print(f"Added embeddings and chunk IDs to {len(data)} items")

# Write the updated data back to data.json
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Successfully wrote updated data to data.json")




    