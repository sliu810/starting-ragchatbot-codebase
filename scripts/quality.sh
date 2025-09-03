#!/bin/bash

echo "Running full quality pipeline..."
echo "================================"

# Format code first
echo "Step 1: Formatting code..."
./scripts/format.sh

echo ""
echo "Step 2: Running quality checks..."
./scripts/lint.sh

echo ""
echo "Step 3: Running tests..."
cd backend && uv run pytest tests/

echo ""
echo "Quality pipeline complete!"