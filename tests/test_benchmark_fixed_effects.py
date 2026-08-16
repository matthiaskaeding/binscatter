"""Smoke tests for the reproducible fixed-effect benchmark harness."""

import platform

import pytest

from scripts import benchmark_fixed_effects as benchmark


@pytest.mark.quick
def test_small_fixed_effect_benchmark_runs_both_paths():
    """The fast harness executes real absorbed and one-hot binscatter calls."""
    result = benchmark.benchmark_cardinality(
        n=500,
        cardinalities=(5,),
        one_hot_max_levels=5,
        repeats=1,
    )

    assert len(result) == 1
    row = result[0]
    assert row.scenario == "one-way cardinality"
    assert row.rows == 500
    assert row.cells == 5
    assert row.absorbed_seconds > 0.0
    assert row.one_hot_seconds is not None
    assert row.one_hot_seconds > 0.0
    assert row.speedup is not None


def test_markdown_report_is_ready_to_copy():
    row = benchmark.BenchmarkResult(
        scenario="one-way cardinality",
        rows=1_000,
        levels="25",
        cells=25,
        absorbed_seconds=0.01,
        one_hot_seconds=0.2,
    )

    report = benchmark.markdown_report([row], (0.01, 0.03))

    assert "| scenario | rows |" in report
    assert (
        "| one-way cardinality | 1,000 | 25 | 25 | 0.010s | 0.200s | 20.0× |" in report
    )
    assert "discovery share: 25%" in report


def test_measure_rejects_an_empty_timing_sample():
    with pytest.raises(ValueError, match="at least one"):
        benchmark._measure(lambda: None, repeats=0)


def test_environment_line_identifies_the_benchmark_runtime():
    line = benchmark.environment_line()

    assert f"Python {platform.python_version()}" in line
    assert f"pandas {benchmark.pd.__version__}" in line
    assert platform.machine() in line
    assert "logical CPUs" in line or "CPU count unknown" in line
    assert "RAM" in line
