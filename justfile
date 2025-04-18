# Checks and formats code
lint:
    uv tool run ruff format
    uv tool run ruff check --fix 

# Run tests
test:
    uv run pytest tests


# Makes a notebook from try_binscatter
make-nb:
    uv tool run --from jupyter-core jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.kernel_name="binscatter" notebooks/try_binscatter.ipynb

# Set up kernel for notebook
setup-krnl:
 uv run -m ipykernel install --user --name=binscatter --display-name "Python binscatter"

install-pkg:
    uv pip install .