from search.bm25_retriever import BM25Retriever, tokenize
from search.data import load_products


def test_tokenize_keeps_model_numbers_together():
    assert tokenize("USB-C Charger XB-200") == ["usb-c", "charger", "xb-200"]


def test_bm25_finds_exact_model_number():
    retriever = BM25Retriever(load_products())
    results = retriever.search("XB-200", top_k=3)
    assert results[0][0] == "p06"
