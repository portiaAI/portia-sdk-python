#!/bin/bash

set -e

# Check if redis-cli is installed
if ! command -v redis-cli &> /dev/null; then
    echo "❌ redis-cli is not installed"
    echo "📦 Install with: brew tap redis/redis && brew install --cask redis"
    exit 1
fi

echo "✅ redis-cli found"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found in current directory"
    exit 1
fi

# Check if LLM_REDIS_CACHE_URL is in .env
if ! grep -q "LLM_REDIS_CACHE_URL" .env; then
    echo "❌ LLM_REDIS_CACHE_URL not found in .env file"
    echo "🔑 Get the secret from: https://portiaai.slack.com/archives/C07UK7NB49L/p1748019608842239"
    exit 1
fi

echo "✅ LLM_REDIS_CACHE_URL found in .env"

# Source .env and run the flush command
echo "🗑️  Flushing Redis cache..."
source .env
redis-cli -u $LLM_REDIS_CACHE_URL FLUSHALL

echo "✅ Cache flushed successfully!"