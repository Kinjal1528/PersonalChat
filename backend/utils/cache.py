import os
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. Path to cache file
CACHE_FILE = "cache_store.pkl"

# 3. Load cache from disk (if exists)
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache_store = pickle.load(f)
    print(f"🧠 Loaded {len(cache_store)} cache entries.")
else:
    cache_store = {}
    print("🧠 Initialized empty cache.")

# 4. Embed text
def embed_query(query: str):
    return np.array(embeddings_model.embed_query(query))

# 5. Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 6. Check cache
def check_cache(query: str, threshold: float = 0.75):
    query_vector = embed_query(query)
    for past_query, (answer, past_vector) in cache_store.items():
        score = cosine_similarity(query_vector, past_vector)
        if score >= threshold:
            print(f"⚡ Cache hit: Cosine Score = {score:.3f}")
            return answer
    return None

# 7. Update cache (only if no match)
def update_cache(query: str, answer: str):
    if check_cache(query) is None:
        cache_store[query] = (answer, embed_query(query))
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_store, f)
        print(f"💾 Added to cache: {query}")
    else:
        print("🔁 Query already exists in cache, not adding.")

# 8. Print cache (optional debug tool)
def print_cache():
    if not cache_store:
        print("🧠 Cache is empty.")
    else:
        for i, (q, (a, _)) in enumerate(cache_store.items(), 1):
            print(f"{i}. Q: {q}\n   A: {a}\n")
print_cache()