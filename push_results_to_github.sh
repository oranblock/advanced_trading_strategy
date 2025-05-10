#!/bin/bash

# Script to check if tuning is complete and push results to GitHub
# Can be run periodically (e.g., via cron) to automatically update the repo

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

# Check if pipeline is complete
PIPELINE_COMPLETE=$(grep "Complete tuning pipeline finished" "$LATEST_SUMMARY")

if [ -z "$PIPELINE_COMPLETE" ]; then
    echo "Tuning pipeline still in progress. No updates pushed to GitHub."
    exit 0
fi

# Extract timestamp from log file
LOG_TIMESTAMP=$(echo "$LATEST_SUMMARY" | grep -o "[0-9]\{8\}_[0-9]\{6\}")

# Change to the project directory
cd "$BASE_DIR" || exit 1

# Check if we already have a git repo
if [ ! -d .git ]; then
    echo "Git repository not initialized. Please run setup_github_repo.sh first."
    exit 1
fi

# Ensure we're using SSH for the remote
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ "$REMOTE_URL" == https://* ]]; then
    echo "Switching remote from HTTPS to SSH..."
    GITHUB_REPO=$(echo "$REMOTE_URL" | sed -E 's|https://github.com/(.+)/(.+)\.git|\1/\2|')
    GITHUB_USERNAME=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
    REPO_NAME=$(echo "$GITHUB_REPO" | cut -d'/' -f2)
    git remote set-url origin "git@github.com:$GITHUB_USERNAME/$REPO_NAME.git"
    echo "Remote URL updated to use SSH."
fi

# Check if there are results to commit
CHANGES=$(git status --porcelain)
if [ -z "$CHANGES" ]; then
    echo "No changes to commit. Exiting."
    exit 0
fi

# Get key results for commit message
THRESHOLD_RESULTS=$(grep -A 2 "Best thresholds:" "$LATEST_SUMMARY" | grep -v "Best thresholds:")
XGBOOST_RESULTS=$(grep -A 5 "XGBoost tuning results:" "$LATEST_SUMMARY" | grep -v "XGBoost tuning results:" | grep -E "profit factor|total trades")

# Create results summary
mkdir -p results
RESULTS_FILE="results/tuning_results_${LOG_TIMESTAMP}.md"

cat > "$RESULTS_FILE" << EOL
# Tuning Results (${LOG_TIMESTAMP})

## Threshold Tuning Results
${THRESHOLD_RESULTS}

## Feature Experiments
The most important features identified:
$(grep -A 20 "Feature importance" "$LOGS_DIR"/feature_experiments_*.log | grep -E "feature|importance" | head -10)

## XGBoost Tuning Results
${XGBOOST_RESULTS}

## Overall Performance
$(grep "profit factor" "$LOGS_DIR"/xgboost_tuning_*.log | tail -5)

## Configuration
Current optimal configuration saved in \`config/settings.yaml\`
EOL

# Create visualization directory if it doesn't exist
mkdir -p results/visualizations

# Copy visualization files if they exist
find "$BASE_DIR"/validation_results -name "*.png" -exec cp {} results/visualizations/ \;

# Add results to git
git add results/
git add config/settings.yaml

# Add tuning script logs
git add run_tuning_pipeline.sh
git add check_tuning_logs.sh
git add setup_github_repo.sh
git add push_results_to_github.sh

# Create commit message
COMMIT_MSG="Update tuning results ${LOG_TIMESTAMP}

- Threshold tuning results
- Feature importance analysis
- XGBoost hyperparameter optimization
- Overall performance metrics

Key results:
${THRESHOLD_RESULTS}
${XGBOOST_RESULTS}"

# Commit changes
git commit -m "$COMMIT_MSG"

# Push to GitHub
git push origin main || git push origin master

echo "Results successfully pushed to GitHub!"