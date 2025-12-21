from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pyspark",
        action="store_true",
        help="Run tests that require PySpark (skipped by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "pyspark: mark test as requiring PySpark and --run-pyspark"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-pyspark"):
        return

    skip_marker = pytest.mark.skip(reason="use --run-pyspark to include PySpark tests")
    for item in items:
        if "pyspark" in item.keywords:
            item.add_marker(skip_marker)
