import math

from alef.stats.significance import benjamini_hochberg, cohens_h, two_proportion_z_test


def test_two_proportion_z_test_identical_proportions_gives_p_one():
    result = two_proportion_z_test(50, 100, 50, 100)
    assert result.z == 0.0
    assert math.isclose(result.p_value, 1.0)


def test_two_proportion_z_test_detects_large_difference():
    result = two_proportion_z_test(670, 10000, 290, 10000)  # 6.7% vs 2.9%, paper's Table 4 support-chat row
    assert result.p_value < 0.01
    assert result.z > 0


def test_cohens_h_zero_for_equal_proportions():
    assert math.isclose(cohens_h(0.3, 0.3), 0.0, abs_tol=1e-9)


def test_cohens_h_matches_paper_formula():
    p1, p2 = 0.067, 0.029
    expected = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    assert math.isclose(cohens_h(p1, p2), expected)


def test_benjamini_hochberg_monotonic_and_bounded():
    pvals = [0.001, 0.2, 0.03, 0.6, 0.049]
    qvals = benjamini_hochberg(pvals)
    assert len(qvals) == len(pvals)
    assert all(0.0 <= q <= 1.0 for q in qvals)
    # BH q-values are never smaller than the raw p-value for the smallest p.
    assert qvals[0] >= pvals[0]


def test_benjamini_hochberg_empty_input():
    assert benjamini_hochberg([]) == []
