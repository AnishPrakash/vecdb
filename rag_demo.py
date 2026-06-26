import requests
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model (this will download ~500MB once)
model = SentenceTransformer('all-MiniLM-L6-v2')

def rag_chat(question):
    # 2. Embed the question
    query_vector = model.encode(question).tolist()

    # 3. Query your Rust engine
    # We assume your Rust server is running on port 8080
    response = requests.post(
        "http://localhost:8080/search", 
        json={"vector": query_vector, "top_k": 3}
    )

    # 4. Extract context from payloads
    results = response.json()
    context = " ".join([item['payload'].get('text', '') for item in results])

    # 5. Send to Ollama
    prompt = f"Using this context: {context}\n\nAnswer this question: {question}"
    ollama_res = requests.post(
        "http://localhost:11434/api/generate", 
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )

    return ollama_res.json()['response']

# Example usage
print(rag_chat("What are the key technical features of the seismic analysis system?"))