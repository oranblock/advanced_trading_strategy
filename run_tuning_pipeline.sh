#!/bin/bash

# Run tuning pipeline in sequence
# Creates logs that can be checked periodically

BASE_DIR="/home/clouduser/advanced_trading_strategy"
LOGS_DIR="$BASE_DIR/tuning_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create logs directory if it doesn't exist
mkdir -p $LOGS_DIR

# Log files
THRESHOLD_LOG="$LOGS_DIR/threshold_tuning_${TIMESTAMP}.log"
FEATURE_LOG="$LOGS_DIR/feature_experiments_${TIMESTAMP}.log"
XGBOOST_LOG="$LOGS_DIR/xgboost_tuning_${TIMESTAMP}.log"
SUMMARY_LOG="$LOGS_DIR/tuning_summary_${TIMESTAMP}.log"

# Function to add timestamp to log entries
log_with_timestamp() {
    while IFS= read -r line; do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $line"
    done
}

echo "Starting tuning pipeline at $(date)" | tee -a $SUMMARY_LOG

# Step 1: Run threshold tuning
echo "Step 1: Starting threshold tuning at $(date)" | tee -a $SUMMARY_LOG
cd $BASE_DIR && python tune_strategy.py --advanced-threshold-tuning 2>&1 | log_with_timestamp | tee -a $THRESHOLD_LOG

# Record completion
echo "Threshold tuning completed at $(date)" | tee -a $SUMMARY_LOG
echo "Best thresholds:" | tee -a $SUMMARY_LOG
grep -A 10 "Threshold tuning complete. Best thresholds" $THRESHOLD_LOG | tee -a $SUMMARY_LOG

# Step 2: Run feature experiments
echo "Step 2: Starting feature experiments at $(date)" | tee -a $SUMMARY_LOG
cd $BASE_DIR && python tune_strategy.py --experiment-features 2>&1 | log_with_timestamp | tee -a $FEATURE_LOG

# Record completion
echo "Feature experiments completed at $(date)" | tee -a $SUMMARY_LOG
echo "Feature experiment results:" | tee -a $SUMMARY_LOG
grep -A 20 "Feature experiments complete" $FEATURE_LOG | tee -a $SUMMARY_LOG

# Step 3: Run XGBoost tuning with more trials
echo "Step 3: Starting XGBoost tuning at $(date)" | tee -a $SUMMARY_LOG
cd $BASE_DIR && python tune_strategy.py --advanced-xgboost-tuning --trials 75 2>&1 | log_with_timestamp | tee -a $XGBOOST_LOG

# Record completion
echo "XGBoost tuning completed at $(date)" | tee -a $SUMMARY_LOG
echo "XGBoost tuning results:" | tee -a $SUMMARY_LOG
grep -A 10 "==== XGBoost Tuning Results ====" $XGBOOST_LOG | tee -a $SUMMARY_LOG

echo "Complete tuning pipeline finished at $(date)" | tee -a $SUMMARY_LOG
echo "Check individual log files for detailed results:" | tee -a $SUMMARY_LOG
echo "Threshold tuning: $THRESHOLD_LOG" | tee -a $SUMMARY_LOG
echo "Feature experiments: $FEATURE_LOG" | tee -a $SUMMARY_LOG
echo "XGBoost tuning: $XGBOOST_LOG" | tee -a $SUMMARY_LOG
echo "Summary: $SUMMARY_LOG" | tee -a $SUMMARY_LOG