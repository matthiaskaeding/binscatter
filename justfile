# Checks and formats code
lint:
    uv tool run ruff format
    uv tool run ruff check --fix

# type check
ty:
    uv tool run ty check src

# lint, format and type check
ok:
    @just lint
    @echo ""
    @just ty


# Skip pyspark
ftest:
    uv run pytest tests

# Run all tests
test:
    uv run pytest --run-pyspark tests

# Makes a notebook from try_binscatter
make-nb:
    uv tool run --from jupyter-core jupyter nbconvert --execute --inplace --to notebook --ExecutePreprocessor.kernel_name="binscatter" notebooks/try_binscatter.ipynb

# Set up kernel for notebook
setup-krnl:
    uv run -m ipykernel install --user --name=binscatter --display-name "Python binscatter"

install-pkg:
    uv pip install .

# Download simulation from binsreg reference
dl-sims:  
    mkdir -p data
    curl -L \
    https://raw.githubusercontent.com/nppackages/binsreg/5dcdb6f14b1d07698b6834a3b8590d0013456f0b/Python/binsreg_sim.csv \
    -o data/binsreg_sim.csv

  # Makes the orphan branch "images", needs artifacts/images to exist
orphan-images:
  tmp=$(mktemp -d) && \
  git --work-tree="$tmp" init -q && \
  cp -a images "$tmp"/ && \
  git -C "$tmp" add -A && \
  git -C "$tmp" commit -m "images branch" && \
  git -C "$tmp" branch -M images && \
  git -C "$tmp" remote add origin "$(git remote get-url origin)" && \
  git -C "$tmp" push -u origin images --force && \
  rm -rf "$tmp"


# prep plots for rpelicateion
make-data-replication:
  uv run scripts/replicate_binscatter/prep_data.py


# replicate plots
replicate-plots:
  uv run scripts/replicate_binscatter/make_plots.py