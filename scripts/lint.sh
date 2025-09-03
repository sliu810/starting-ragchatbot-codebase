#!/bin/bash

echo "Running code quality checks..."
echo "=============================="

echo "Running flake8..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503

echo "Running mypy..."
uv run mypy backend/ main.py --ignore-missing-imports

echo "Running isort check..."
uv run isort --check-only backend/ main.py

echo "Running black check..."
uv run black --check backend/ main.py

echo "Quality checks complete!"