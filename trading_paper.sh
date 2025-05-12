#!/bin/bash
# Script to start the real-time liquidity trading strategy in paper trading mode with REAL market data

# Create required directories if they don't exist
mkdir -p ./realtime_results
mkdir -p ./realtime_logs

# Set the date for logs
TODAY=$(date +"%Y%m%d")

# Log the startup
echo "===== Starting Liquidity Trading Strategy (PAPER MODE with REAL DATA) =====" > "./realtime_logs/startup_${TODAY}.log"
echo "Started at: $(date)" >> "./realtime_logs/startup_${TODAY}.log"
echo "=================================================" >> "./realtime_logs/startup_${TODAY}.log"

# Make sure the strategy script is executable
chmod +x ./liquidity_strategy_realtime.py
chmod +x ./update_dashboard.sh

# Ensure yfinance is installed for fallback data source
echo "Checking for yfinance Python package..."
if ! pip list | grep -q yfinance; then
    echo "Installing yfinance as fallback data source..."
    pip install yfinance
fi

# Create initial dashboard files
./update_dashboard.sh

# Make sure the previous logs are stopped to avoid duplicates with old data
kill $(cat ./realtime_logs/strategy_pid.txt 2>/dev/null) >/dev/null 2>&1 || true

# Create a test file to verify API key is working
echo "Testing Polygon.io API access..."
POLYGON_TEST=$(curl -s "https://api.polygon.io/v2/aggs/ticker/C:EURUSD/prev?adjusted=true&apiKey=p6WersYkAHkp9TccmLHvdDwGaZ4CnR0Y" | grep -o "ticker")
if [ -z "$POLYGON_TEST" ]; then
    echo "WARNING: Polygon.io API access not working. Check your API key."
    echo "Using p6WersYkAHkp9TccmLHvdDwGaZ4CnR0Y"
fi

# Run the strategy in paper trading mode with real market data for EURUSD
./liquidity_strategy_realtime.py \
  --mode paper \
  --ticker EURUSD \
  --timeframe hour \
  --up-threshold 0.5 \
  --down-threshold 0.5 \
  --pt-multiplier 2.5 \
  --sl-multiplier 1.5 \
  --risk-per-trade 1.0 \
  --initial-equity 10000 \
  --paper-trading \
  --notify-trades \
  >> "./realtime_logs/strategy_${TODAY}.log" 2>&1 &

# Save the process ID so we can stop it later if needed
echo $! > "./realtime_logs/strategy_pid.txt"

# Function to check if strategy is properly monitoring stops and stops are working
function check_stops {
  # Get current EURUSD price from API
  local current_price=$(curl -s "https://api.polygon.io/v2/aggs/ticker/C:EURUSD/prev?adjusted=true&apiKey=p6WersYkAHkp9TccmLHvdDwGaZ4CnR0Y" | grep -o '"c":[0-9.]*' | cut -d':' -f2)

  # Load the latest trade from JSONL
  if [ -f "./realtime_logs/trades_${TODAY}.jsonl" ]; then
    local trades_file="./realtime_logs/trades_${TODAY}.jsonl"
    local latest_trade=$(tail -1 "$trades_file")

    # Extract direction, action, price
    local direction=$(echo "$latest_trade" | grep -o '"direction":"[^"]*"' | cut -d':' -f2 | tr -d '"')
    local action=$(echo "$latest_trade" | grep -o '"action":"[^"]*"' | cut -d':' -f2 | tr -d '"')
    local entry_price=$(echo "$latest_trade" | grep -o '"price":[0-9.]*' | cut -d':' -f2)

    # Check if we have an open LONG position
    if [ "$direction" == "LONG" ] && [ "$action" == "ENTRY" ] && [ -n "$current_price" ] && [ -n "$entry_price" ]; then
      # Get stop loss level from strategy log
      local stop_loss=$(grep -o "LONG position opened.*SL: [0-9.]*" ./realtime_logs/strategy_*.log | tail -1 | grep -o "SL: [0-9.]*" | cut -d' ' -f2)

      if [ -n "$stop_loss" ] && (( $(echo "$current_price <= $stop_loss" | bc -l) )); then
        echo "⚠️ Stop loss should be triggered! Current: $current_price, SL: $stop_loss"
        echo "Restarting strategy to force position check..."
        kill $(cat ./realtime_logs/strategy_pid.txt)
        sleep 2
        ./liquidity_strategy_realtime.py \
          --mode paper \
          --ticker EURUSD \
          --timeframe hour \
          --up-threshold 0.5 \
          --down-threshold 0.5 \
          --pt-multiplier 2.5 \
          --sl-multiplier 1.5 \
          --risk-per-trade 1.0 \
          --initial-equity 10000 \
          --paper-trading \
          --notify-trades \
          >> "./realtime_logs/strategy_${TODAY}.log" 2>&1 &
        echo $! > "./realtime_logs/strategy_pid.txt"
      fi
    fi
  fi
}

# Set up regular dashboard updates
echo "Starting dashboard update service..."
while true; do
  ./update_dashboard.sh
  # Check if stop losses need to be forced
  check_stops
  sleep 60
done &

echo "Paper trading strategy started with REAL market data (PID: $(cat ./realtime_logs/strategy_pid.txt))"
echo "Logs are being written to ./realtime_logs/strategy_${TODAY}.log"
echo "Dashboard available at: $(pwd)/dashboard.html"
echo "Dashboard will update automatically every minute"
echo "To stop the strategy, run: kill $(cat ./realtime_logs/strategy_pid.txt)"

# Open the dashboard in the browser if possible
if command -v xdg-open &> /dev/null; then
  xdg-open ./dashboard.html &
elif command -v open &> /dev/null; then
  open ./dashboard.html &
fi