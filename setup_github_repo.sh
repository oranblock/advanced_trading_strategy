#!/bin/bash

# Script to create a new GitHub repository and push the current project to it
# Assumes SSH is already set up and configured for GitHub

# Set these variables to your preferences
REPO_NAME="advanced_trading_strategy"
REPO_DESCRIPTION="Advanced algorithmic trading strategy with XGBoost, hyperparameter tuning, and advanced ICT features"
REPO_VISIBILITY="public"  # This repository will be publicly accessible

echo "Setting up new GitHub repository: $REPO_NAME"

# Create .gitignore file if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore file..."
    cat > .gitignore << EOL
# Logs and results
tuning_logs/
validation_results/
*.log

# IMPORTANT: API keys and sensitive data
config/secrets.yaml
**/keys.py
**/secrets.py
**/*api_key*
**/*secret*
**/*token*
**/*.pem
**/*.key
**/credentials*
.env.local
.env.*.local

# Data
data/credentials/
data/raw/
data/*.csv
data/*.pickle
data/*.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOL
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Create README.md if it doesn't exist
if [ ! -f README.md ]; then
    echo "Creating README.md..."
    cat > README.md << EOL
# Advanced Trading Strategy

Advanced algorithmic trading strategy with XGBoost, hyperparameter tuning, and advanced market features.

## Features

- Probability threshold tuning to optimize trading signals
- XGBoost parameter optimization using Optuna
- Advanced ICT (Inner Circle Trader) feature engineering
- Feature experimentation framework
- Comprehensive validation using time-series cross-validation

## Components

- **Threshold Tuning**: Optimizes entry/exit probability thresholds
- **Feature Engineering**: Advanced market structure features
- **XGBoost Tuning**: Machine learning model optimization
- **Validation Framework**: Prevents look-ahead bias in backtesting

## Requirements

- Python 3.7+
- numpy, pandas, scikit-learn
- XGBoost
- Optuna
- TA-Lib (for technical indicators)
EOL
fi

# Run config sanitization to remove API keys
echo "Sanitizing configuration files to remove API keys..."
./sanitize_config.sh

# Add all files to git
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit of advanced trading strategy

- Trading validation framework
- Threshold optimization
- XGBoost hyperparameter tuning
- Advanced feature engineering
- Experiment tracking"

# Create GitHub repository using GitHub CLI
if command -v gh &> /dev/null; then
    echo "Creating GitHub repository using GitHub CLI..."
    gh repo create "$REPO_NAME" --description "$REPO_DESCRIPTION" --"$REPO_VISIBILITY" --source=. --remote=origin --push
else
    # Manual setup instructions
    echo "GitHub CLI not found. Please set up repository manually:"
    echo "1. Create a new repository on GitHub: https://github.com/new"
    echo "   - Name: $REPO_NAME"
    echo "   - Description: $REPO_DESCRIPTION"
    echo "   - Visibility: $REPO_VISIBILITY"
    echo "   - Do NOT initialize with README, .gitignore, or license files"
    echo ""
    echo "2. After creating the repository, run these commands:"
    echo "   git remote add origin git@github.com:YOUR_USERNAME/$REPO_NAME.git"
    echo "   git push -u origin master"
    echo ""

    # Prompt user for GitHub username
    read -p "Enter your GitHub username to set up remote now: " GITHUB_USERNAME

    if [ -n "$GITHUB_USERNAME" ]; then
        # Set up the remote
        echo "Setting up Git remote..."
        git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git

        echo "Remote set up successfully. To push your code, run:"
        echo "git push -u origin master"
    else
        echo "No username provided. You'll need to set up the remote manually later."
    fi
fi

echo "Repository setup complete! Your code is now on GitHub."
echo "Repository URL: https://github.com/$(git config user.name)/$REPO_NAME"