#!/bin/bash
# Script to start the real-time liquidity trading strategy with real data

# Create required directories if they don't exist
mkdir -p ./realtime_results
mkdir -p ./realtime_logs

# Set the date for logs
TODAY=$(date +"%Y%m%d")

# Log the startup
echo "===== Starting Liquidity Trading Strategy (REAL MODE) =====" > "./realtime_logs/startup_${TODAY}.log"
echo "Started at: $(date)" >> "./realtime_logs/startup_${TODAY}.log"
echo "=================================================" >> "./realtime_logs/startup_${TODAY}.log"

# Make sure the strategy script is executable
chmod +x ./liquidity_strategy_realtime.py
chmod +x ./update_dashboard.sh

# Create initial dashboard files
./update_dashboard.sh

# Important notice
echo "===== STARTING REAL TRADING MODE ====="
echo "WARNING: This will connect to real market data and could place real trades"
echo "Make sure your API credentials are properly configured in config/settings.yaml"
echo "==================================="

read -p "Are you sure you want to continue with REAL trading? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Trading startup canceled. Use paper-trading mode if you want to test."
    exit 1
fi

# Run the strategy in real trading mode with optimized settings for EURUSD
./liquidity_strategy_realtime.py \
  --mode live \
  --ticker EURUSD \
  --timeframe hour \
  --up-threshold 0.5 \
  --down-threshold 0.5 \
  --pt-multiplier 2.5 \
  --sl-multiplier 1.5 \
  --risk-per-trade 1.0 \
  --initial-equity 10000 \
  --notify-trades \
  >> "./realtime_logs/strategy_${TODAY}.log" 2>&1 &

# Save the process ID so we can stop it later if needed
echo $! > "./realtime_logs/strategy_pid.txt"

# Set up regular dashboard updates
echo "Starting dashboard update service..."
while true; do
  ./update_dashboard.sh
  sleep 60
done &

echo "REAL TRADING strategy started (PID: $(cat ./realtime_logs/strategy_pid.txt))"
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