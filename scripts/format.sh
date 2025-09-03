#!/bin/bash

echo "Running code formatting..."
echo "=========================="

echo "Running isort..."
uv run isort backend/ main.py

echo "Running black..."
uv run black backend/ main.py

echo "Code formatting complete!"