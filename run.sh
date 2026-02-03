#!/bin/bash

# SAT Forum Responder - Startup Script

# Change to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p logs
mkdir -p db
mkdir -p keys

# Start the webhook server
python3 -m src.webhook_receiver
