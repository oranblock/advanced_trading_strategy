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

## Project Structure

```
advanced_trading_strategy/
├── config/                # Configuration files
│   └── settings.yaml     # Strategy parameters
├── data/                  # Data storage (market data, etc.)
├── logs/                  # Log files
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── config_loader.py  # Configuration loading utilities
│   ├── data_handler.py   # Data fetching and processing
│   ├── feature_generator.py # Feature engineering
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── main_strategy.py  # Main strategy implementation
│   ├── model_trainer.py  # Model training utilities
│   ├── threshold_tuner.py # Specialized probability threshold tuning
│   ├── utils.py          # Utility functions
│   └── validation_framework.py # Cross-validation framework
├── tune_strategy.py       # Command-line tool for optimization
└── requirements.txt       # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Polygon.io API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced_trading_strategy.git
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

Run the main strategy:
```bash
python -m src.main_strategy
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