"""The search engine: puts all the pieces together.

Pipeline for method="hybrid_rerank":

    query --> BM25 (top 20) ---\
                                >-- RRF fusion --> cross-encoder re-rank --> top_k results
    query --> dense (top 20) --/
"""
from search.bm25_retriever import BM25Retriever
from search.data import product_text
from search.dense_retriever import DenseRetriever
from search.fusion import reciprocal_rank_fusion
from search.reranker import Reranker

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")


class SearchEngine:
    def __init__(self, products, candidates_per_retriever=20):
        self.products_by_id = {p["id"]: p for p in products}
        self.texts_by_id = {p["id"]: product_text(p) for p in products}
        self.n_candidates = candidates_per_retriever
        print("Loading BM25 index ...")
        self.bm25 = BM25Retriever(products)
        print("Loading embedding model (first run downloads it) ...")
        self.dense = DenseRetriever(products)
        print("Loading cross-encoder re-ranker ...")
        self.reranker = Reranker()
        print("Search engine ready.")

    def search(self, query, method="hybrid_rerank", top_k=10):
        """Return a list of (product_id, score), best first."""
        if method == "bm25":
            return self.bm25.search(query, top_k)
        if method == "dense":
            return self.dense.search(query, top_k)

        # Both hybrid methods start the same way: get candidates from both retrievers, then fuse.
        bm25_results = self.bm25.search(query, self.n_candidates)
        dense_results = self.dense.search(query, self.n_candidates)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], top_k=self.n_candidates)

        if method == "hybrid":
            return fused[:top_k]
        if method == "hybrid_rerank":
            return self.reranker.rerank(query, fused, self.texts_by_id, top_k)
        raise ValueError(f"Unknown method '{method}'. Choose one of {METHODS}.")
