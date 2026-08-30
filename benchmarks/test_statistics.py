from benchmarks.statistics import (
    bootstrap_geomean_ci,
    paired_accuracy_difference_ci,
    wilson_interval,
)


def test_bootstrap_constant():
    assert bootstrap_geomean_ci([0.0, 0.0]) == [1.0, 1.0]


def test_wilson_boundaries():
    lo, hi = wilson_interval(100, 100)
    assert 0.96 < lo < 1.0
    assert hi == 1.0


def test_paired_difference_constant():
    assert paired_accuracy_difference_ci([1, 1, 1], [0, 0, 0]) == [1.0, 1.0]


if __name__ == "__main__":
    test_bootstrap_constant()
    test_wilson_boundaries()
    test_paired_difference_constant()
    print("ALL STATISTICS TESTS PASSED")
