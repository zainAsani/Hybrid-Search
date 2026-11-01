"""Ranking metrics. Each function compares the ids our engine returned
(ranked_ids, best first) with the set of ids that are actually relevant.
"""
import math


def recall_at_k(ranked_ids, relevant_ids, k):
    """Share of the relevant products that appear in the top k.

    2 relevant products, 1 of them in the top 5 -> Recall@5 = 0.5
    """
    if not relevant_ids:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids, k):
    """1 / position of the FIRST relevant product in the top k (0 if none).

    First relevant result at position 3 -> 1/3. Averaged over queries this is MRR.
    """
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    """Normalised Discounted Cumulative Gain with yes/no relevance.

    Each relevant result earns 1 / log2(position + 1): position 1 earns 1.0,
    position 2 earns 0.63, position 3 earns 0.5 ... (that is the "discount").
    We divide by the best possible score (all relevant products on top),
    so 1.0 means a perfect ranking.
    """
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
