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

## Project Structure

```
advanced_trading_strategy/
├── config/                     # Configuration files
│   ├── settings.yaml          # Strategy parameters
│   └── liquidity_settings.yaml # Liquidity strategy parameters
├── data/                       # Data storage (market data, etc.)
├── dashboard.html              # Real-time trading dashboard
├── docs/                       # Documentation
├── final_liquidity_strategy/   # Optimized strategy models and results
├── logs/                       # Log files
├── notebooks/                  # Jupyter notebooks for analysis
├── realtime_logs/              # Real-time trading logs
├── realtime_results/           # Real-time trading performance results
├── src/                        # Source code
│   ├── advanced_features.py   # Advanced feature engineering
│   ├── config_loader.py       # Configuration loading utilities
│   ├── data_handler.py        # Data fetching and processing
│   ├── feature_generator.py   # Feature engineering
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── json_utils.py          # JSON utilities for real-time trading
│   ├── labeling_tuner.py      # Optimization for data labeling
│   ├── liquidity_strategy.py  # Specialized liquidity trading strategy
│   ├── main_strategy.py       # Main strategy implementation
│   ├── model_trainer.py       # Model training utilities
│   ├── threshold_tuner.py     # Specialized probability threshold tuning
│   ├── utils.py               # Utility functions
│   ├── validation_framework.py # Cross-validation framework
│   └── xgboost_tuner.py       # XGBoost hyperparameter tuning
├── extended_walkforward_analysis.py # Extended walkforward analysis for robustness testing
├── final_liquidity_strategy.py      # Final optimized strategy implementation
├── liquidity_strategy_production.py # Production-ready strategy for real trading
├── liquidity_strategy_realtime.py   # Real-time trading implementation
├── liquidity_strategy_standalone.py # Standalone liquidity strategy implementation
├── run_extended_walkforward.sh      # Script to run walkforward analysis across timeframes
├── trading_startup.sh               # Script to start real-time trading
├── trading_paper.sh                 # Script to start paper trading with real market data
├── tune_strategy.py                 # Command-line tool for optimization
├── update_dashboard.sh              # Script to update the real-time dashboard
└── requirements.txt                 # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Polygon.io API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/oranblock/advanced_trading_strategy.git
cd advanced_trading_strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
POLYGON_API_KEY=your_polygon_key
NEWS_API_KEY=your_news_api_key  # Optional
```

### Configuration

Edit `config/settings.yaml` to configure your strategy parameters:

- **Data Parameters**: Ticker, timeframe, data range
- **Feature Engineering**: Swing point detection parameters
- **Labeling Parameters**: Look-forward period, profit targets
- **Model Parameters**: XGBoost configuration, probability thresholds

### Running the Strategy

#### Backtesting and Analysis

Run the main strategy for historical backtesting:
```bash
python -m src.main_strategy
```

Or use the production-ready liquidity strategy for comprehensive analysis:
```bash
python liquidity_strategy_production.py --mode all --ticker EURUSD --timeframe hour --plot-results
```

#### Real-time Trading

For real-time trading with real market data (paper trading):
```bash
./trading_paper.sh
```

This will:
1. Start the strategy with real market data from Polygon.io
2. Execute paper trades (no real money)
3. Track performance in the dashboard
4. Update automatically every minute

For live trading (requires broker API integration):
```bash
./trading_startup.sh
```

## Strategy Optimization

The project includes a comprehensive framework for hyperparameter tuning.

### Probability Threshold Tuning

Optimize the model probability thresholds:

```bash
# Run advanced threshold tuning with visualization
python tune_strategy.py --advanced-threshold-tuning
```

The advanced threshold tuner will:
1. Perform a coarse grid search over probability thresholds
2. Generate heatmaps for profit factor, trade count, and win rates
3. Run a fine-grained search around the best thresholds
4. Update the configuration file with optimal values

### Feature Experimentation

Our experiments revealed that external liquidity features significantly outperform all other feature types:

```bash
# Run feature experiments to identify best feature groups
python tune_strategy.py --experiment-features
```

Our findings:
1. External liquidity features achieved a profit factor of 16.7 in testing
2. Focusing exclusively on these features provided superior results
3. Previous day/week high/low levels are particularly predictive

### Optimized Liquidity Strategy

Run our optimized liquidity-based strategy:

```bash
# Run the standalone liquidity strategy implementation
python liquidity_strategy_standalone.py
```

This strategy focuses exclusively on external liquidity features with the optimal thresholds (UP=0.5, DOWN=0.5).

### XGBoost Parameter Tuning

Optimize the XGBoost model parameters:

```bash
python tune_strategy.py --advanced-xgboost-tuning --trials 75
```

### Comprehensive Tuning

Run all tuning processes:

```bash
python tune_strategy.py --tune-all
```

## Extended Walk-Forward Analysis

A key component of our framework is extended walk-forward analysis that validates strategy robustness across multiple decades and market regimes.

### Running Walk-Forward Analysis

Run the extended walk-forward analysis across multiple timeframes:

```bash
# Run the extended walk-forward analysis script
./run_extended_walkforward.sh
```

This script will:
1. Run walk-forward analysis on hourly data (10 years)
2. Run walk-forward analysis on daily data (30 years)
3. Run walk-forward analysis on weekly data (50 years)
4. Generate a comprehensive combined report

### Analysis Parameters

The extended walk-forward analysis examines:
- **Multiple Timeframes**: From hourly to weekly data
- **Extended History**: 10-50 years of market data
- **Multiple Market Regimes**: Bull markets, bear markets, sideways markets
- **Different Economic Cycles**: Inflation, deflation, recession, expansion
- **Structural Market Changes**: Changes in market microstructure over decades

### Results Visualization

The analysis generates multiple visualizations:
- **Performance by Window**: Profit factor, win rate, and total return across all windows
- **Performance Over Time**: How metrics evolve across different market regimes
- **Performance Relationships**: How different metrics correlate with each other
- **Performance Distributions**: Statistical distributions of key metrics

### Comprehensive Report

The final report includes:
- **Strategy Robustness Rating**: Overall assessment of strategy strength
- **Market Regime Analysis**: Performance across different market conditions
- **Consistent Profitability**: Percentage of windows with positive performance
- **Parameter Stability**: Assessment of parameter effectiveness over time
- **Recommendations**: Actionable insights for strategy deployment

## Production Trading System

### Real-time Trading Implementation

The project includes a complete real-time trading implementation:

- **`liquidity_strategy_realtime.py`**: Core real-time trading engine
- **`trading_startup.sh`**: Script to start live trading
- **`trading_paper.sh`**: Script to start paper trading with real market data
- **`update_dashboard.sh`**: Script to update the real-time dashboard
- **`dashboard.html`**: Interactive web-based dashboard

Features:
- **Multiple Trading Modes**: Live trading, paper trading, demo mode
- **Real-time Market Data**: Integration with Polygon.io API with Yahoo Finance fallback
- **Trading Dashboard**: Real-time monitoring of positions, signals, and performance
- **Proper Risk Management**: ATR-based position sizing, stop losses, and take profits
- **Notification System**: Trade and signal notifications (customizable)

Example usage:
```bash
# Start paper trading with real market data
./trading_paper.sh

# For live trading (requires broker API setup)
./trading_startup.sh
```

### Strategy Implementation

The `liquidity_strategy_production.py` script provides a production-ready implementation with:

- **Multiple Operation Modes**: Backtest, train, analyze, or all-in-one
- **Command-line Interface**: Easy configuration of all parameters
- **Extended Data Support**: Handles large historical datasets (10-50 years)
- **Walk-Forward Analysis**: Tests strategy consistency across market regimes
- **Advanced Visualization**: Comprehensive performance visualizations
- **Thorough Reporting**: Detailed performance metrics and statistics

Example usage:
```bash
# Run complete analysis on EURUSD hourly data
python liquidity_strategy_production.py --mode all --ticker EURUSD --timeframe hour --start-date 2015-01-01 --windows 5 --plot-results
```

## Validation Framework

The validation framework (`validation_framework.py`) provides a robust way to evaluate strategy parameters:

- Time-series cross-validation to prevent lookahead bias
- Performance metrics including profit factor, win rates, and trade counts
- Ability to tune individual parameter groups
- Visualization of results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inner Circle Trader (ICT) concepts for market structure analysis
- XGBoost team for the gradient boosting framework
- Polygon.io for market data access
- Yahoo Finance for fallback market data

## Recent Updates

- **Real-time Trading System**: Added complete real-time trading implementation with dashboard
- **Multiple Data Sources**: Added support for Polygon.io with Yahoo Finance fallback
- **Improved Stop Loss Handling**: Enhanced stop loss monitoring and execution
- **Paper Trading Mode**: Added paper trading with real market data
- **Live Trading Capability**: Added support for live trading (requires broker API setup)
- **Trading Dashboard**: Added interactive web dashboard for performance monitoring