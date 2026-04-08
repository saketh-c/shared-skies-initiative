#!/bin/bash

# Shared Skies - Full Stack Runner
ROOT=$(pwd)

echo "Starting Shared Skies Initiative..."
echo ""

# Install backend dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing backend dependencies..."
    pip install -r backend/requirements.txt
fi

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting backend on http://localhost:8000..."
echo "Starting frontend on http://localhost:5173..."
echo ""

# Start backend
cd backend
python -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both services running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop."

# Handle Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

wait
