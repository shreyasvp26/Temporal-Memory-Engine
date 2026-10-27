#!/usr/bin/env bash
set -e

case "$1" in
    api)
        echo "Starting Temporal Memory Engine API server on 0.0.0.0:8000..."
        exec uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
        ;;
    worker)
        echo "Starting Temporal Memory Engine Background Worker..."
        exec python -m apps.worker.main
        ;;
    eval)
        shift
        echo "Running LongMemEval CLI..."
        exec python -m eval.cli "$@"
        ;;
    migrate)
        shift
        echo "Running Database Migrations..."
        exec python -m memory_engine.db.migrate "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
