"""Lexical (keyword) search with BM25.

BM25 scores a product higher when it contains the words of the query,
especially rare words (like "XB-200") that appear in few products.
It does NOT understand meaning: "coffee" will not match "espresso".
"""
import re

from rank_bm25 import BM25Okapi

from search.data import product_text


def tokenize(text):
    """Turn text into a list of lowercase words.

    "USB-C Charger XB-200" -> ["usb-c", "charger", "xb-200"]
    The regular expression keeps words joined by a hyphen together.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25Retriever:
    def __init__(self, products):
        self.ids = [p["id"] for p in products]
        tokenized_corpus = [tokenize(product_text(p)) for p in products]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=10):
        """Return a list of (product_id, score), best first."""
        scores = self.bm25.get_scores(tokenize(query))  # one score per product
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        # A score of 0 means no query word appears in the product, so we drop it.
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k] if score > 0]
