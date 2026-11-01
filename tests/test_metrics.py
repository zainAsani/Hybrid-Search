"""Tests for the ranking metrics. Run all tests with:  python -m pytest"""
import math

import pytest

from search.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_products_in_top_k():
    # 2 relevant products (a, c); only "a" is in the top 2
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=10) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_found():
    assert reciprocal_rank(["x", "y"], {"a"}, k=10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_is_lower_when_the_relevant_product_comes_second():
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))
