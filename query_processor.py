import pandas as pd
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def create_embedding(text_list):
    """Create embeddings using Ollama's bge-m3 model"""
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()['embeddings']
    return embedding


def inference(prompt):
    """Send prompt to LLM and get response"""
    r = requests.post('http://localhost:11434/api/generate', json={
        "model": "deepseek-r1",
        'prompt': prompt,
        "stream": False
    })
    response = r.json()
    return response.get('response', '')


def load_data():
    """Load video chunk data from JSON file"""
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_embeddings():
    """Load pre-computed embeddings from joblib file"""
    try:
        df = joblib.load('embeddings.joblib')
        print(f"Loaded {len(df)} embeddings from cache")
        return df
    except FileNotFoundError:
        print("No cached embeddings found. Creating new embeddings...")
        data = load_data()
        df = pd.DataFrame.from_records(data)
        return df


def search_relevant_chunks(query, df, top_k=50):
    """Search for most relevant video chunks based on query"""
    # Create embedding for the query
    question_embedding = create_embedding([query])[0]
    
    # Calculate cosine similarities
    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()
    
    # Get top k most similar chunks
    max_idx = similarities.argsort()[::-1][0:top_k]
    relevant_chunks = df.loc[max_idx]
    
    return relevant_chunks


def create_prompt(query, relevant_chunks):
    """Create a structured prompt for the LLM"""
    prompt = f'''You are a knowledgeable football expert assistant helping users learn about football history and players based on Markaroni's video content.

## Context Data
Below are the most relevant video chunks from Markaroni's football videos:

{relevant_chunks[['title', 'number', 'text', 'start', 'end']].to_json()}

## User Question
{query}

## Instructions
1. Analyze the provided video chunks carefully
2. Answer the user's question directly and conversationally
3. Use specific information from the chunks to support your answer
4. If the question is unrelated to the provided content, politely inform the user that you don't have relevant information
5. Include video references in this format: "Video: [title] at [MM:SS]"
6. Convert timestamps from seconds to MM:SS format (e.g., 90 seconds = 1:30)
7. Structure your response clearly with paragraphs for readability
8. Be concise but informative - aim for 3-5 sentences unless more detail is needed

## Response Format
Provide your answer in a natural, conversational tone as if speaking directly to the user. Include relevant timestamps and video titles 
to help them find the original content. Don't try to answer the questions just tell
the user in which video and at what time he will get the relevant answers.
Simply give output as: Video_Name - Time_Stamp => Brief info description

Do not ask any counter questions or provide suggestions.
'''
    return prompt


def process_query(query, top_k=50):
    """Main function to process a user query and get response"""
    # Load embeddings
    df = load_embeddings()
    
    # Search for relevant chunks
    print(f"Searching for relevant chunks for query: '{query}'")
    relevant_chunks = search_relevant_chunks(query, df, top_k)
    
    # Create prompt
    prompt = create_prompt(query, relevant_chunks)
    
    # Save prompt to file for debugging
    with open('prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    # Get LLM response
    print("Generating response...")
    response = inference(prompt)
    
    return response


def main():
    """Main entry point for the application"""
    # Example queries
    queries = [
        "Tell me about Marco van Basten",
        "Who is Lionel Messi?",
        "What makes Messi different from Ronaldo?"
    ]
    
    # Process first query as example
    query = queries[0]
    response = process_query(query)
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    print(response)
    print("="*80)


if __name__ == "__main__":
    main()
