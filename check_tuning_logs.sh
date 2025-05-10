#!/bin/bash

# Script to check tuning logs status without generating excessive output

BASE_DIR="/home/clouduser/advanced_trading_strategy"
LOGS_DIR="$BASE_DIR/tuning_logs"

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    echo "No logs found. Tuning process may not have started yet."
    exit 1
fi

# Find the most recent summary log
LATEST_SUMMARY=$(ls -t "$LOGS_DIR"/tuning_summary_*.log 2>/dev/null | head -n 1)

if [ -z "$LATEST_SUMMARY" ]; then
    echo "No summary logs found. Tuning process may not have started yet."
    exit 1
fi

# Get current status
echo "Current Tuning Status (as of $(date)):"
echo "---------------------------------------"

# Extract timestamp from filename
LOG_TIMESTAMP=$(echo "$LATEST_SUMMARY" | grep -o "[0-9]\{8\}_[0-9]\{6\}")
echo "Tuning run started: ${LOG_TIMESTAMP:0:8} at ${LOG_TIMESTAMP:9:2}:${LOG_TIMESTAMP:11:2}:${LOG_TIMESTAMP:13:2}"

# Check which steps are complete
THRESHOLD_COMPLETE=$(grep "Threshold tuning completed" "$LATEST_SUMMARY")
FEATURE_COMPLETE=$(grep "Feature experiments completed" "$LATEST_SUMMARY")
XGBOOST_COMPLETE=$(grep "XGBoost tuning completed" "$LATEST_SUMMARY")
PIPELINE_COMPLETE=$(grep "Complete tuning pipeline finished" "$LATEST_SUMMARY")

# Show progress
echo ""
echo "Progress:"
if [ -n "$THRESHOLD_COMPLETE" ]; then
    echo "✓ Step 1: Threshold tuning - COMPLETE"
    grep -A 2 "Best thresholds:" "$LATEST_SUMMARY" | grep -v "Best thresholds:" | sed 's/^/  /'
else
    RUNNING=$(ps aux | grep "[p]ython tune_strategy.py --advanced-threshold-tuning" | wc -l)
    if [ $RUNNING -gt 0 ]; then
        echo "⟳ Step 1: Threshold tuning - RUNNING"
    else
        echo "⟳ Step 1: Threshold tuning - PENDING"
    fi
fi

echo ""
if [ -n "$FEATURE_COMPLETE" ]; then
    echo "✓ Step 2: Feature experiments - COMPLETE"
    grep -A 5 "Feature experiment results:" "$LATEST_SUMMARY" | grep -v "Feature experiment results:" | sed 's/^/  /'
else
    RUNNING=$(ps aux | grep "[p]ython tune_strategy.py --experiment-features" | wc -l)
    if [ $RUNNING -gt 0 ]; then
        echo "⟳ Step 2: Feature experiments - RUNNING"
    elif [ -n "$THRESHOLD_COMPLETE" ]; then
        echo "⟳ Step 2: Feature experiments - PENDING"
    else
        echo "- Step 2: Feature experiments - WAITING"
    fi
fi

echo ""
if [ -n "$XGBOOST_COMPLETE" ]; then
    echo "✓ Step 3: XGBoost tuning - COMPLETE"
    grep -A 5 "XGBoost tuning results:" "$LATEST_SUMMARY" | grep -v "XGBoost tuning results:" | grep -E "profit factor|total trades|F1 score" | sed 's/^/  /'
else
    RUNNING=$(ps aux | grep "[p]ython tune_strategy.py --advanced-xgboost-tuning" | wc -l)
    if [ $RUNNING -gt 0 ]; then
        echo "⟳ Step 3: XGBoost tuning - RUNNING"
    elif [ -n "$FEATURE_COMPLETE" ]; then
        echo "⟳ Step 3: XGBoost tuning - PENDING"
    else
        echo "- Step 3: XGBoost tuning - WAITING"
    fi
fi

echo ""
if [ -n "$PIPELINE_COMPLETE" ]; then
    echo "✓ Complete pipeline - FINISHED"
    FINISH_TIME=$(grep "Complete tuning pipeline finished" "$LATEST_SUMMARY" | awk '{print $NF}')
    echo "  Completed at: $FINISH_TIME"
else
    echo "⟳ Complete pipeline - IN PROGRESS"
fi

echo ""
echo "To view detailed logs:"
echo "  Threshold tuning: tail -f $LOGS_DIR/threshold_tuning_${LOG_TIMESTAMP}.log"
echo "  Feature experiments: tail -f $LOGS_DIR/feature_experiments_${LOG_TIMESTAMP}.log"
echo "  XGBoost tuning: tail -f $LOGS_DIR/xgboost_tuning_${LOG_TIMESTAMP}.log"
echo "  Summary: cat $LATEST_SUMMARY"