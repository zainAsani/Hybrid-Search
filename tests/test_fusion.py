from search.fusion import reciprocal_rank_fusion


def test_product_found_by_both_retrievers_wins():
    bm25 = [("a", 7.5), ("b", 3.1)]
    dense = [("c", 0.91), ("a", 0.85)]
    fused = reciprocal_rank_fusion([bm25, dense])
    assert fused[0][0] == "a"


def test_only_rank_matters_not_raw_score():
    # Same order, very different raw scores -> identical fused result
    small = reciprocal_rank_fusion([[("a", 0.2), ("b", 0.1)]])
    large = reciprocal_rank_fusion([[("a", 900.0), ("b", 1.0)]])
    assert small == large
