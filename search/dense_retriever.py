"""Semantic search with embeddings (also called dense retrieval).

An embedding model turns a text into a list of 384 numbers (a vector).
Texts with similar meaning get similar vectors, so "earphones for jogging"
lands close to "earbuds ... ideal for running", even without shared words.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from search.data import product_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, products, model_name=DEFAULT_MODEL):
        self.ids = [p["id"] for p in products]
        self.model = SentenceTransformer(model_name)
        texts = [product_text(p) for p in products]
        # We embed all products once at start-up. Shape: (number_of_products, 384).
        # normalize_embeddings=True scales every vector to length 1, so a simple
        # dot product below equals the cosine similarity.
        self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query, top_k=10):
        """Return a list of (product_id, similarity), best first."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.doc_embeddings @ query_embedding  # one similarity per product
        best_first = np.argsort(-scores)[:top_k]        # indices sorted high -> low
        return [(self.ids[i], float(scores[i])) for i in best_first]
