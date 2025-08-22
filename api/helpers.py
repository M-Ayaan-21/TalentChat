# api/helpers.py

import math
import os
import openai

# Make sure your OPENAI_API_KEY is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    Returns 0 if vectors are invalid or have different lengths.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def generate_embedding_sync(text, model="text-embedding-3-small"):
    """
    Generate embedding vector for a given text using OpenAI embeddings API.
    
    Parameters:
    - text: str
    - model: embedding model name (default: text-embedding-3-small)
    
    Returns:
    - list of floats representing the embedding vector
    """
    if not text:
        return []

    try:
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

