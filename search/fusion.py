"""Combine several ranked result lists into one with Reciprocal Rank Fusion (RRF).

BM25 scores (e.g. 7.3) and cosine similarities (e.g. 0.61) are on different
scales, so we cannot simply add them. RRF ignores the raw scores and only uses
the POSITION of each product in each list:

    rrf_score(product) = sum over lists of 1 / (k + rank)

A product that is ranked high in both lists gets the highest combined score.
k = 60 is the value from the original RRF paper; it softens the gap between
rank 1 and rank 2.
"""


def reciprocal_rank_fusion(result_lists, k=60, top_k=10):
    """result_lists: e.g. [bm25_results, dense_results], each a list of (id, score).

    Returns a list of (id, rrf_score), best first.
    """
    fused_scores = {}
    for results in result_lists:
        for rank, (doc_id, _raw_score) in enumerate(results, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
