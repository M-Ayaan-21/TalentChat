import numpy as np
import openai
import certifi
import httpx

# ----------------- Cosine Similarity -----------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# ----------------- Embedding Generator -----------------
def generate_embedding_sync(text, model="text-embedding-ada-002"):
    """
    Generate embedding for a given text synchronously.
    """
    client = httpx.Client(verify=certifi.where(), timeout=30.0)
    try:
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        return response['data'][0]['embedding']
    finally:
        client.close()

