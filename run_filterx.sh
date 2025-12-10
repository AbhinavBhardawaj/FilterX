#!/bin/bash

# FilterX Quick Start Script
# This script sets up and runs everything needed for FilterX

PROJECT_DIR="/Users/atulkumar/Desktop/filterx demo"
BACKEND_DIR="$PROJECT_DIR/backend"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         FilterX - NSFW Content Filter                      ║"
echo "║         Quick Start Script                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
    echo "❌ Backend directory not found: $BACKEND_DIR"
    exit 1
fi

echo "📁 Project Directory: $PROJECT_DIR"
echo ""

# Start Backend
echo "🚀 Starting Backend Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$BACKEND_DIR"

# Kill any existing Django process on port 8000
echo "🧹 Cleaning up old processes..."
lsof -i :8000 2>/dev/null | grep -v COMMAND | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

# Start the server
echo "▶️  Starting Django development server..."
python3 manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!
echo "✅ Backend started with PID: $DJANGO_PID"
echo ""

# Wait for server to start
sleep 3

# Test backend
echo "🧪 Testing Backend..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
HEALTH=$(curl -s http://127.0.0.1:8000/api/health/)
if echo "$HEALTH" | grep -q "ok"; then
    echo "✅ Health check passed: $HEALTH"
else
    echo "❌ Health check failed"
    echo "Response: $HEALTH"
fi
echo ""

# Show endpoints
echo "📡 Available Endpoints:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Health: http://127.0.0.1:8000/api/health/"
echo "✓ Classify: http://127.0.0.1:8000/api/classify/"
echo ""

# Show next steps
echo "📋 Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Load the Chrome Extension:"
echo "   • Open Chrome: chrome://extensions/"
echo "   • Toggle 'Developer mode' (top right)"
echo "   • Click 'Load unpacked'"
echo "   • Select: $PROJECT_DIR/extension"
echo ""

echo "2️⃣  Test the Extension:"
echo "   • Visit any website"
echo "   • Click the FilterX icon"
echo "   • See statistics and activity"
echo ""

echo "3️⃣  Test via API (in another terminal):"
echo ""
echo "   # Test text classification"
echo "   curl -X POST http://127.0.0.1:8000/api/classify/ \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"type\": \"text\", \"content\": \"porn\"}'"
echo ""
echo "   # Test URL blocking"
echo "   curl -X POST http://127.0.0.1:8000/api/classify/ \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"type\": \"url\", \"content\": \"https://xvideos.com\"}'"
echo ""

echo "🛑 To Stop Backend:"
echo "   kill $DJANGO_PID"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Backend is running! Follow the steps above to load the extension."
echo "   Backend logs will appear below:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for Django process (keep script running)
wait $DJANGO_PID
