#!/usr/bin/env bash
set -euo pipefail
# run-local.sh — start backend in background, then start frontend in foreground
# Usage: ./run-local.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Start backend
echo "Starting backend (uvicorn) on port 8000..."
cd backend
# Use env file if exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
# Launch uvicorn in background so frontend runs in same terminal
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Move to frontend
cd "$ROOT_DIR/frontend"
# Install if node_modules missing
if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies (npm install)..."
  npm install --no-audit --no-fund
fi

# Start frontend dev server (runs in foreground)
echo "Starting frontend (vite)..."
npm run dev

# When frontend exits, kill backend
echo "Frontend exited. Stopping backend (PID $BACKEND_PID)"
kill $BACKEND_PID || true
