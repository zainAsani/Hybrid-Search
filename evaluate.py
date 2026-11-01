"""Offline evaluation: compare the four search methods on the labelled test queries.

Run:  python evaluate.py
"""
from search.data import load_products, load_queries
from search.engine import METHODS, SearchEngine
from search.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

K = 10  # we look at the top 10 results


def first_relevant_position(ranked_ids, relevant_ids):
    """Position of the first correct product, or '-' if it is not in the results."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return str(position)
    return "-"


def main():
    products = load_products()
    queries = load_queries()
    engine = SearchEngine(products)

    summary = {}       # method -> average metrics
    positions = {}     # (query id, method) -> position of first relevant result

    for method in METHODS:
        recalls, rrs, ndcgs = [], [], []
        for q in queries:
            results = engine.search(q["query"], method=method, top_k=K)
            ranked_ids = [doc_id for doc_id, _score in results]
            relevant = set(q["relevant_ids"])

            recalls.append(recall_at_k(ranked_ids, relevant, k=5))
            rrs.append(reciprocal_rank(ranked_ids, relevant, k=K))
            ndcgs.append(ndcg_at_k(ranked_ids, relevant, k=K))
            positions[(q["id"], method)] = first_relevant_position(ranked_ids, relevant)

        n = len(queries)
        summary[method] = (sum(recalls) / n, sum(rrs) / n, sum(ndcgs) / n)

    print("\n=== Average over all queries (higher is better, 1.0 = perfect) ===")
    print(f"{'method':<15}{'Recall@5':>10}{'MRR@10':>10}{'nDCG@10':>10}")
    for method, (recall, mrr, ndcg) in summary.items():
        print(f"{method:<15}{recall:>10.3f}{mrr:>10.3f}{ndcg:>10.3f}")

    print("\n=== Position of the first correct product per query ('-' = not found) ===")
    print(f"{'query':<42}" + "".join(f"{m:>15}" for m in METHODS))
    for q in queries:
        row = "".join(f"{positions[(q['id'], m)]:>15}" for m in METHODS)
        print(f"{q['query']:<42}{row}")


if __name__ == "__main__":
    main()
