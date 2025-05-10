# Hyperparameter Tuning Guide

This document provides detailed guidance on optimizing your trading strategy parameters for maximum performance.

## Understanding Strategy Parameters

The performance of your trading strategy is highly dependent on several key parameter groups:

### 1. Probability Thresholds

- `UP_PROB_THRESHOLD`: Minimum probability required from the UP model to consider a LONG signal
- `DOWN_PROB_THRESHOLD`: Minimum probability required from the DOWN model to consider a SHORT signal

Higher thresholds generally result in:
- Fewer but higher-quality signals
- Higher win rates
- Lower total trade count

Lower thresholds generally result in:
- More signals
- Lower win rates
- Higher total trade count

The optimal thresholds balance signal quality and quantity to maximize overall performance metrics like profit factor.

### 2. XGBoost Parameters

The XGBoost model parameters control how the machine learning models are trained:

- `n_estimators`: Number of trees (higher = more complex but possibly overfit)
- `max_depth`: Maximum tree depth (higher = more complex but possibly overfit)
- `learning_rate`: Step size shrinkage (lower = more conservative learning)
- `subsample`: Fraction of samples used per tree (< 1.0 helps prevent overfitting)
- `colsample_bytree`: Fraction of features used per tree (< 1.0 helps prevent overfitting)
- `min_child_weight`: Minimum sum of instance weight needed in a child node (higher = more conservative)

### 3. Labeling Parameters

These parameters control how past price action is labeled for training:

- `LOOK_FORWARD_CANDLES`: How many candles forward to look for PT/SL targets
- `PROFIT_TAKE_ATR`: Profit target as a multiple of Average True Range
- `STOP_LOSS_ATR`: Stop loss level as a multiple of Average True Range

## Tuning Workflow

Follow this workflow for effective optimization:

### Step 1: Run the Baseline Strategy

Before tuning, run the strategy with default parameters to establish a baseline:

```bash
python -m src.main_strategy
```

Note key metrics:
- Model performance metrics (precision, recall, F1)
- Signal distribution
- Win rates for LONG and SHORT signals
- Overall profit factor

### Step 2: Tune Probability Thresholds

Probability thresholds are the easiest and most impactful parameters to tune first:

```bash
python tune_strategy.py --advanced-threshold-tuning
```

This will:
1. Run a coarse grid search across probability thresholds (0.5 to 0.8)
2. Generate heatmaps visualizing performance metrics
3. Perform a fine-grained search around the best values
4. Automatically update your configuration with optimal values

**Interpreting Threshold Heatmaps:**
- **Profit Factor Map**: Higher values (darker colors) indicate better risk-reward balance
- **Trade Count Map**: Shows how many trades occur at different threshold combinations
- **Win Rate Maps**: Shows how win rates change across thresholds

### Step 3: Tune XGBoost Parameters

Once you have optimal thresholds, tune the XGBoost parameters:

```bash
python tune_strategy.py --tune-xgboost
```

XGBoost tuning takes longer but can significantly improve model quality. The framework will:
1. Train models with different parameter combinations
2. Evaluate performance using cross-validation
3. Update your configuration with the best parameters

### Step 4: Fine-Tune Labeling Parameters (Advanced)

Labeling parameters require regenerating labels for each combination, making this the most computationally intensive tuning process. This is not fully implemented in the current framework but could be a future enhancement.

### Step 5: Run Comprehensive Validation

After individual parameter tuning, run a comprehensive evaluation:

```bash
python tune_strategy.py --tune-all
```

This runs the entire tuning pipeline and provides a holistic view of optimized performance.

## Interpreting Results

Key metrics to focus on:

1. **Profit Factor**: The ratio of gross profits to gross losses. Higher is better, with values above 1.5 generally considered good.

2. **Win Rate**: The percentage of trades that are profitable. Higher is better, but must be considered alongside the profit factor (a low win rate with high reward-to-risk can still be profitable).

3. **Trade Count**: The number of trades generated. Too few trades may indicate overfitting or excessive caution, while too many may dilute performance.

4. **Balance between LONG and SHORT signals**: A well-balanced strategy should perform reasonably well in both directions.

## Tips for Effective Tuning

1. **Start with quick searches**: Use fewer cross-validation folds for initial exploration.

2. **Focus on one parameter group at a time**: Thresholds → XGBoost → Labeling.

3. **Consider trade-offs**: Higher precision often comes at the cost of fewer trades. The best strategy balances quality and quantity.

4. **Check for robustness**: A good parameter set should perform well across multiple validation folds.

5. **Avoid overfitting**: Be cautious of parameter sets that work extremely well on validation but have signs of overfitting (e.g., very few trades or extreme thresholds).

6. **Document your results**: Keep track of tuning runs to understand how different parameters affect performance.

## Validating Results

After tuning, run the main strategy with optimized parameters:

```bash
python -m src.main_strategy
```

Compare the optimized results to your baseline to confirm improvement.

## Future Enhancements

Potential improvements to the tuning framework:

1. **Automated optimization of labeling parameters**
2. **Bayesian optimization for faster parameter searching**
3. **Integration with a full backtesting framework for more realistic performance assessment**
4. **Multi-metric optimization (balancing profit factor, drawdown, trade frequency, etc.)**