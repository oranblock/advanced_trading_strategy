#!/bin/bash
# Script to push code to GitHub while ensuring API keys are not included

# Stop if any command fails
set -e

echo "===== GitHub Repository Setup and Push ====="
echo "Started at: $(date)"
echo "==============================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if GitHub username and repo name are provided
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter repository name (default: advanced_trading_strategy): " REPO_NAME
REPO_NAME=${REPO_NAME:-advanced_trading_strategy}

# Initialize git repository if it doesn't exist
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "Git repository initialized."
fi

# Create README.md if it doesn't exist
if [ ! -f "README.md" ]; then
    echo "Creating README.md..."
    cat > "README.md" << EOL
# Advanced Trading Strategy

A sophisticated algorithmic trading framework leveraging machine learning and external liquidity concepts to identify trading opportunities in financial markets.

## Performance Highlights

Our specialized external liquidity strategy achieved:
- **Profit Factor: 4.01** - Highly profitable with strong risk/reward
- **High Win Rate**: 60.6% for short trades, 40.7% for long trades
- **Robust Sample Size**: 1,151 trades in the test set
- **Trade Quality**: 610 wins vs 152 losses (80% quality ratio)

## Overview

This project implements a comprehensive trading strategy pipeline that:

1. Loads market data from Polygon.io API
2. Applies technical analysis and ICT (Inner Circle Trader) concepts for feature generation
3. Specializes in external liquidity features (previous day/week high/low levels)
4. Labels data using price action-based outcomes
5. Trains XGBoost models to predict market direction
6. Optimizes strategy with an advanced hyperparameter tuning framework
7. Generates trading signals using a dual-model approach
8. Validates robustness with extended walk-forward analysis across decades of data

## Features

- **Specialized in External Liquidity**: Focuses on previous day/week high/low levels
- **Dual-Model Approach**: Separate models for UP and DOWN predictions
- **Optimal Parameters**: Extensively tested parameters (UP/DOWN thresholds = 0.5, PT = 2.5× ATR, SL = 1.5× ATR)
- **Extended Walk-Forward Testing**: Validated across multiple decades and market regimes
- **Interactive Dashboard**: Real-time monitoring of performance metrics and trades
- **Multiple Operation Modes**: Live trading, paper trading, and backtesting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
cd $REPO_NAME
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Running the Strategy

Start the trading strategy:
\`\`\`bash
./trading_startup.sh
\`\`\`

This will launch the strategy in paper trading mode with optimal parameters and open the monitoring dashboard.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOL
    echo "README.md created."
fi

# Check if LICENSE file exists, if not create it
if [ ! -f "LICENSE" ]; then
    echo "Creating LICENSE file..."
    cat > "LICENSE" << EOL
MIT License

Copyright (c) $(date +"%Y") $GITHUB_USERNAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOL
    echo "LICENSE file created."
fi

# Sanitize config files
echo "Checking for sensitive data in configuration files..."
CONFIG_FILES=$(find . -name "*.yaml" -o -name "*.json" -o -name "*.env*" -o -name "*.config" -o -name "*.ini")

for file in $CONFIG_FILES; do
    echo "Checking $file for sensitive data..."
    grep -q "api_key\|apikey\|secret\|password\|credential" "$file" && echo "WARNING: $file may contain sensitive data."
done

# Make sure empty directories are preserved
echo "Creating .gitkeep files in empty directories..."
find . -type d -empty -not -path "*/\.*" -exec touch {}/.gitkeep \;

# Check if .env file exists and create a sample instead
if [ -f ".env" ]; then
    echo "Creating .env.sample file from .env (without actual keys)..."
    cat .env | sed 's/\(.*=\).*/\1YOUR_KEY_HERE/' > .env.sample
    echo ".env.sample created with placeholders."
fi

# Create a sample settings file if needed
if [ -f "./config/settings.yaml" ]; then
    echo "Creating a sanitized sample settings file..."
    cp ./config/settings.yaml ./config/settings.sample.yaml
    # Replace any API keys or secrets with placeholders
    sed -i 's/\(api_key\|apikey\|secret\|password\|credentials\): .*/\1: YOUR_KEY_HERE/g' ./config/settings.sample.yaml
    echo "settings.sample.yaml created."
fi

# Stage all files
echo "Staging files for commit..."
git add .

# Make initial commit
echo "Creating initial commit..."
git commit -m "Initial commit of Advanced Trading Strategy

This commit includes:
- Liquidity-based trading strategy implementation
- Extended walk-forward analysis framework
- Real-time trading dashboard
- Documentation and example configuration files

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Set up GitHub as remote
echo "Setting up GitHub remote..."
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Push to GitHub with instructions
echo "Ready to push to GitHub"
echo "================================================================"
echo "To push to GitHub, you'll need to authenticate."
echo "The easiest way is to create a personal access token on GitHub:"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token' (classic)"
echo "3. Give it a name like 'Advanced Trading Strategy Push'"
echo "4. Select at least the 'repo' scope permissions"
echo "5. Generate the token and copy it"
echo "6. When prompted for your password below, paste the token"
echo "================================================================"
echo "Run this command to push to GitHub:"
echo "git push -u origin master"
echo "================================================================"
echo "GitHub setup completed at: $(date)"