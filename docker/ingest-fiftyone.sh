#!/bin/sh
# Ingest data/processed/ddr2019 into FiftyOne using /workspace paths so the Docker App
# can show images. Run from project root, or we cd to it.
# Requires: data/processed/ddr2019/images/ and labels.csv; .env with FIFTYONE_DATABASE_URI.
set -e
cd "$(dirname "$0")/.."
docker compose -f docker/docker-compose.yml run --rm \
  -e FIFTYONE_MEDIA_PREFIX=/workspace \
  --entrypoint '' fiftyone \
  sh -c 'cd /app && uv run python /workspace/scripts/ingest_ddr2019_to_fiftyone.py'
