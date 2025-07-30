#!/bin/bash

# UniMap Development Startup Script

echo "🚀 Starting UniMap Development Environment..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server
echo "📡 Starting Flask backend server..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🌐 Starting Next.js frontend server..."
cd frontend
npm install > /dev/null 2>&1
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Development servers started!"
echo "📡 Backend: http://localhost:5000"
echo "🌐 Frontend: http://localhost:3000"
echo "🧪 Test API: http://localhost:3000/test-api.html"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait 