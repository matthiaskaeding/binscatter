.PHONY: help lint ty ok test test-quick test-pyspark make-nb setup-krnl install-hooks pre-commit install-pkg dl-sims make-data-replication make-plots

help:
	@echo "Developer commands:"
	@echo "  make lint                 # Format python files and fix lint with ruff"
	@echo "  make ty                   # Type-check src/ with ty"
	@echo "  make ok                   # Run linting/formatting and type checks"
	@echo "  make test                 # Run the test suite, minus PySpark"
	@echo "  make test-quick           # Run a representative sample of tests"
	@echo "  make test-pyspark         # Run the test suite including PySpark"
	@echo "  make benchmark-fe         # Benchmark fixed-effect absorption vs one-hot"
	@echo "  make make-nb              # Render examples/demo.ipynb under artifacts/"
	@echo "  make setup-krnl           # Install the binscatter ipykernel"
	@echo "  make install-hooks        # Install pre-commit hooks via prek"
	@echo "  make pre-commit           # Run prek against all files"
	@echo "  make install-pkg          # Install the project into the current environment"
	@echo "  make dl-sims              # Download binsreg reference simulation data"
	@echo "  make make-data-replication# Prepare replication data"
	@echo "  make make-plots           # Build README plots from replication data"

lint:
	uv tool run ruff format
	uv tool run ruff check --fix

ty:
	uv tool run ty@latest check src

ok:
	@$(MAKE) lint
	@echo ""
	@$(MAKE) ty

test:
	uv run pytest tests

test-quick:
	uv run pytest tests --quick

test-pyspark:
	uv run pytest tests --run-pyspark

benchmark-fe:
	uv run scripts/benchmark_fixed_effects.py

make-nb:
	mkdir -p artifacts/notebooks
	uv run --with nbconvert jupyter nbconvert --execute --to notebook \
		examples/demo.ipynb --output-dir artifacts/notebooks --output demo.ipynb \
		--ExecutePreprocessor.timeout=600 --ExecutePreprocessor.record_timing=False

setup-krnl:
	uv run -m ipykernel install --user --name=binscatter --display-name "Python binscatter"

install-hooks:
	uv tool run prek install

pre-commit:
	uv tool run prek run --all-files

install-pkg:
	uv pip install .

dl-sims:
	mkdir -p data
	curl -L \
	https://raw.githubusercontent.com/nppackages/binsreg/5dcdb6f14b1d07698b6834a3b8590d0013456f0b/Python/binsreg_sim.csv \
	-o data/binsreg_sim.csv

make-data-replication:
	uv run scripts/replicate_binscatter/prep_data.py

make-plots:
	uv run scripts/replicate_binscatter/make_plots.py
