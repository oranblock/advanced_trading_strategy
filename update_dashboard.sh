#!/bin/bash
# Script to update the dashboard with the latest performance data

# Set the date for logs
TODAY=$(date +"%Y%m%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Make sure required directories exist
mkdir -p ./realtime_results
mkdir -p ./realtime_logs

# Create a symlink to the latest performance file
LATEST_PERFORMANCE_FILE=$(find ./realtime_results -name "performance_${TODAY}.json" | sort -r | head -n 1)

if [ ! -z "$LATEST_PERFORMANCE_FILE" ]; then
  cp "$LATEST_PERFORMANCE_FILE" "./realtime_results/performance.json"
  echo "Updated performance data from $LATEST_PERFORMANCE_FILE"
fi

# Create a symlink to the latest equity curve image
LATEST_EQUITY_CURVE=$(find ./realtime_results -name "equity_curve_${TODAY}.png" | sort -r | head -n 1)

if [ ! -z "$LATEST_EQUITY_CURVE" ]; then
  cp "$LATEST_EQUITY_CURVE" "./realtime_results/equity_curve.png"
  echo "Updated equity curve from $LATEST_EQUITY_CURVE"
fi

# If no data exists yet, create placeholder files
if [ ! -f "./realtime_results/performance.json" ]; then
  echo '{
    "timestamp": "'"$(date +"%Y-%m-%d %H:%M:%S")"'",
    "initial_equity": 10000,
    "current_equity": 10000,
    "total_return": 0,
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "win_rate": 0,
    "profit_factor": 0,
    "max_drawdown": 0,
    "current_position": "FLAT",
    "position_size": 0
  }' > "./realtime_results/performance.json"
  echo "Created placeholder performance data"
fi

# Open the dashboard in the default browser (if available)
if command -v xdg-open &> /dev/null; then
  xdg-open ./dashboard.html &
  echo "Dashboard opened in browser"
elif command -v open &> /dev/null; then
  open ./dashboard.html &
  echo "Dashboard opened in browser"
else
  echo "Dashboard ready at ./dashboard.html"
fi

echo "Dashboard updated at $(date)"