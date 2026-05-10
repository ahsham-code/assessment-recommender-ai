import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# LOAD EMBEDDING MODEL
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# LOAD CATALOG
with open("catalog.json", "r") as f:
    catalog = json.load(f)

# CREATE SEARCH TEXTS
texts = []

for item in catalog:

    combined_text = f"""
    {item['name']}
    {item['description']}
    {' '.join(item['skills'])}
    """

    texts.append(combined_text)

# CREATE EMBEDDINGS
embeddings = model.encode(texts)

# CREATE FAISS INDEX
dimension = len(embeddings[0])

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

# SEARCH FUNCTION
def search_catalog(query, top_k=5):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(catalog[idx])

    return results