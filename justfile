lint:
    uv tool run ruff format
    uv tool run ruff check --fix 

test:
    uv run pytest tests
