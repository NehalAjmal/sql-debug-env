#!/usr/bin/env bash

PING_URL=$1
REPO_DIR=${2:-.}

echo "Checking reset endpoint..."
curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset" \
  -H "Content-Type: application/json" -d '{}' | grep 200 \
  && echo "✅ Reset OK" || echo "❌ Reset Failed"

echo ""
echo "Building Docker..."
docker build -t test-env "$REPO_DIR" && echo "✅ Docker build OK" || echo "❌ Docker build failed"

echo ""
echo "Running OpenEnv validate..."
openenv validate --url "$PING_URL"
