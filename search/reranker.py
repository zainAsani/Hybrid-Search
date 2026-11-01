"""Re-rank a short list of candidates with a cross-encoder.

The dense retriever embeds query and product SEPARATELY (fast, works for
millions of products). A cross-encoder reads query and product TOGETHER,
which is more accurate but much slower. So we only use it on the top
candidates that the cheaper retrievers already found.
"""
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, texts_by_id, top_k=10):
        """candidates: list of (id, score) from an earlier step. Returns new (id, score) list."""
        if not candidates:
            return []
        candidate_ids = [doc_id for doc_id, _ in candidates]
        pairs = [(query, texts_by_id[doc_id]) for doc_id in candidate_ids]
        scores = self.model.predict(pairs)  # one relevance score per (query, product) pair
        ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]
