import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .json_utils import save_json

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
from .validation_framework import StrategyValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('threshold_tuner')

def run_threshold_grid_search(X_all, y_all, config, results_dir='./threshold_results'):
    """
    Run a grid search over probability thresholds and visualize results.
    
    Args:
        X_all: DataFrame with features
        y_all: Series with target labels (-1, 0, 1)
        config: Configuration dictionary
        results_dir: Directory to save results
    
    Returns:
        best_thresholds: Dictionary with best UP_PROB_THRESHOLD and DOWN_PROB_THRESHOLD
    """
    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # Define threshold grid
    up_thresholds = np.linspace(0.5, 0.8, 7)
    down_thresholds = np.linspace(0.5, 0.8, 7)
    
    # Initialize result grid
    threshold_results = []
    best_profit_factor = 0
    best_up_threshold = 0.55
    best_down_threshold = 0.55
    
    # Run grid search with single cross-validation fold for speed
    n_splits = 2
    test_size = 0.2
    val_size = 0.15
    
    logger.info(f"Running threshold grid search with {len(up_thresholds)}x{len(down_thresholds)} combinations")
    
    for up_threshold in up_thresholds:
        for down_threshold in down_thresholds:
            up_threshold_rounded = round(up_threshold, 2)
            down_threshold_rounded = round(down_threshold, 2)
            
            param_config = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded
            }
            
            logger.info(f"Testing thresholds: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}")
            
            # Evaluate parameters
            avg_metrics = validator.evaluate_params(param_config, n_splits, test_size, val_size)
            
            result = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded,
                'profit_factor': avg_metrics['profit_factor'],
                'total_trades': avg_metrics['total_trades'],
                'long_win_rate': avg_metrics['long_win_rate'],
                'short_win_rate': avg_metrics['short_win_rate'],
                'neutral_count': avg_metrics['neutral_count']
            }
            threshold_results.append(result)
            
            # Update best parameters
            if avg_metrics['profit_factor'] > best_profit_factor:
                best_profit_factor = avg_metrics['profit_factor']
                best_up_threshold = up_threshold_rounded
                best_down_threshold = down_threshold_rounded
                logger.info(f"New best: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}, PF={best_profit_factor:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(threshold_results)
    
    # Filter out combinations with too few trades for more reliable results
    min_trades = 10
    filtered_results = results_df[results_df['total_trades'] >= min_trades]
    
    # If all combinations have fewer than min_trades, keep all results
    if filtered_results.empty:
        filtered_results = results_df
    
    # Save results
    results_df.to_csv(results_dir / 'threshold_grid_results.csv', index=False)
    
    # Find overall best result (after filtering)
    if not filtered_results.empty:
        best_row = filtered_results.loc[filtered_results['profit_factor'].idxmax()]
        best_up_threshold = best_row['UP_PROB_THRESHOLD']
        best_down_threshold = best_row['DOWN_PROB_THRESHOLD']
        best_profit_factor = best_row['profit_factor']
    
    logger.info(f"Best thresholds after filtering: UP={best_up_threshold}, DOWN={best_down_threshold}, PF={best_profit_factor:.2f}")
    
    # Visualize results
    visualize_threshold_results(results_df, results_dir)
    
    best_thresholds = {
        'UP_PROB_THRESHOLD': best_up_threshold,
        'DOWN_PROB_THRESHOLD': best_down_threshold
    }
    
    # Save best thresholds using utility function
    save_json(best_thresholds, results_dir / 'best_thresholds.json')
    
    return best_thresholds

def visualize_threshold_results(results_df, results_dir):
    """
    Create visualizations of threshold grid search results.
    
    Args:
        results_df: DataFrame with threshold grid search results
        results_dir: Directory to save visualizations
    """
    # Create pivot tables for heatmaps
    try:
        # Profit factor heatmap
        profit_pivot = results_df.pivot(
            index='DOWN_PROB_THRESHOLD', 
            columns='UP_PROB_THRESHOLD', 
            values='profit_factor'
        )
        
        # Trade count heatmap
        trades_pivot = results_df.pivot(
            index='DOWN_PROB_THRESHOLD', 
            columns='UP_PROB_THRESHOLD', 
            values='total_trades'
        )
        
        # Win rate heatmaps
        long_wr_pivot = results_df.pivot(
            index='DOWN_PROB_THRESHOLD', 
            columns='UP_PROB_THRESHOLD', 
            values='long_win_rate'
        )
        
        short_wr_pivot = results_df.pivot(
            index='DOWN_PROB_THRESHOLD', 
            columns='UP_PROB_THRESHOLD', 
            values='short_win_rate'
        )
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot profit factor
        sns.heatmap(profit_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Profit Factor')
        
        # Plot trade count - using float format since values might not be integers after averaging
        sns.heatmap(trades_pivot, annot=True, fmt='.1f', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Total Trade Count')
        
        # Plot long win rate
        sns.heatmap(long_wr_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 0])
        axes[1, 0].set_title('Long Trade Win Rate')
        
        # Plot short win rate
        sns.heatmap(short_wr_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 1])
        axes[1, 1].set_title('Short Trade Win Rate')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(results_dir / 'threshold_heatmaps.png', dpi=300)
        plt.close()
        
        logger.info(f"Saved threshold heatmaps to {results_dir / 'threshold_heatmaps.png'}")
    except Exception as e:
        logger.error(f"Error creating threshold visualizations: {e}")

def fine_tune_thresholds(X_all, y_all, config, coarse_best=None, results_dir='./threshold_results_fine'):
    """
    Fine-tune probability thresholds with a finer grain around the best coarse thresholds.
    
    Args:
        X_all: DataFrame with features
        y_all: Series with target labels (-1, 0, 1)
        config: Configuration dictionary
        coarse_best: Dictionary with best thresholds from coarse search
        results_dir: Directory to save results
    
    Returns:
        best_thresholds: Dictionary with best UP_PROB_THRESHOLD and DOWN_PROB_THRESHOLD
    """
    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # Use provided coarse best or default values
    if coarse_best is None:
        coarse_best = {
            'UP_PROB_THRESHOLD': 0.55,
            'DOWN_PROB_THRESHOLD': 0.55
        }
    
    # Define fine-grained threshold grid around best values
    # Create a range of +/- 0.05 around the best values with 0.01 step
    best_up = float(coarse_best['UP_PROB_THRESHOLD'])
    best_down = float(coarse_best['DOWN_PROB_THRESHOLD'])
    
    up_min = max(0.5, best_up - 0.05)
    up_max = min(0.85, best_up + 0.05)
    down_min = max(0.5, best_down - 0.05)
    down_max = min(0.85, best_down + 0.05)
    
    up_thresholds = np.linspace(up_min, up_max, 11)  # 0.01 step size
    down_thresholds = np.linspace(down_min, down_max, 11)  # 0.01 step size
    
    # Initialize result grid
    threshold_results = []
    best_profit_factor = 0
    best_up_threshold = best_up
    best_down_threshold = best_down
    
    # Run grid search with more cross-validation folds for reliability
    n_splits = 3
    test_size = 0.2
    val_size = 0.15
    
    logger.info(f"Running fine-grained threshold search with {len(up_thresholds)}x{len(down_thresholds)} combinations")
    
    for up_threshold in up_thresholds:
        for down_threshold in down_thresholds:
            up_threshold_rounded = round(up_threshold, 2)
            down_threshold_rounded = round(down_threshold, 2)
            
            param_config = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded
            }
            
            logger.info(f"Testing thresholds: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}")
            
            # Evaluate parameters
            avg_metrics = validator.evaluate_params(param_config, n_splits, test_size, val_size)
            
            result = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded,
                'profit_factor': avg_metrics['profit_factor'],
                'total_trades': avg_metrics['total_trades'],
                'long_win_rate': avg_metrics['long_win_rate'],
                'short_win_rate': avg_metrics['short_win_rate'],
                'neutral_count': avg_metrics['neutral_count']
            }
            threshold_results.append(result)
            
            # Update best parameters
            if avg_metrics['profit_factor'] > best_profit_factor:
                best_profit_factor = avg_metrics['profit_factor']
                best_up_threshold = up_threshold_rounded
                best_down_threshold = down_threshold_rounded
                logger.info(f"New best: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}, PF={best_profit_factor:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(threshold_results)
    
    # Filter out combinations with too few trades for more reliable results
    min_trades = 10
    filtered_results = results_df[results_df['total_trades'] >= min_trades]
    
    # If all combinations have fewer than min_trades, keep all results
    if filtered_results.empty:
        filtered_results = results_df
    
    # Save results
    results_df.to_csv(results_dir / 'threshold_fine_results.csv', index=False)
    
    # Find overall best result (after filtering)
    if not filtered_results.empty:
        best_row = filtered_results.loc[filtered_results['profit_factor'].idxmax()]
        best_up_threshold = best_row['UP_PROB_THRESHOLD']
        best_down_threshold = best_row['DOWN_PROB_THRESHOLD']
        best_profit_factor = best_row['profit_factor']
    
    logger.info(f"Best fine-grained thresholds: UP={best_up_threshold}, DOWN={best_down_threshold}, PF={best_profit_factor:.2f}")
    
    # Visualize results
    visualize_threshold_results(results_df, results_dir)
    
    best_thresholds = {
        'UP_PROB_THRESHOLD': best_up_threshold,
        'DOWN_PROB_THRESHOLD': best_down_threshold
    }
    
    # Save best thresholds using utility function
    save_json(best_thresholds, results_dir / 'best_fine_thresholds.json')
    
    return best_thresholds

def evaluate_final_thresholds(X_all, y_all, config, best_thresholds, results_dir='./threshold_final'):
    """
    Evaluate final thresholds on test set and create detailed analysis.
    
    Args:
        X_all: DataFrame with features
        y_all: Series with target labels (-1, 0, 1)
        config: Configuration dictionary
        best_thresholds: Dictionary with best UP_PROB_THRESHOLD and DOWN_PROB_THRESHOLD
        results_dir: Directory to save results
    
    Returns:
        final_metrics: Dictionary with final performance metrics
    """
    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # Evaluate final model with best thresholds
    final_metrics = validator.evaluate_final_model(best_thresholds)
    
    # Create report
    report = f"""# Probability Threshold Tuning Results

## Best Thresholds
- UP_PROB_THRESHOLD: {best_thresholds['UP_PROB_THRESHOLD']}
- DOWN_PROB_THRESHOLD: {best_thresholds['DOWN_PROB_THRESHOLD']}

## Performance Metrics
- Profit Factor: {final_metrics['profit_factor']:.2f}
- Total Trades: {final_metrics['total_trades']}
- Win Rate (Long): {final_metrics['long_win_rate']*100:.1f}%
- Win Rate (Short): {final_metrics['short_win_rate']*100:.1f}%

## Detailed Statistics
- Long Trades: {final_metrics['long_trades']}
  - Wins: {final_metrics['long_wins']}
  - Losses: {final_metrics['long_losses']}
- Short Trades: {final_metrics['short_trades']}
  - Wins: {final_metrics['short_wins']}
  - Losses: {final_metrics['short_losses']}
- Neutral Signals: {final_metrics['neutral_count']}

## Recommendations
The above thresholds represent the optimal balance between signal frequency and quality for this strategy.
"""
    
    # Save report
    with open(results_dir / 'threshold_tuning_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"Saved threshold tuning report to {results_dir / 'threshold_tuning_report.md'}")
    
    return final_metrics

def update_config_with_thresholds(best_thresholds):
    """
    Update settings.yaml with the best probability thresholds.

    Args:
        best_thresholds: Dictionary with best UP_PROB_THRESHOLD and DOWN_PROB_THRESHOLD
    """
    import yaml

    config_path = Path("./config/settings.yaml")

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update thresholds, ensuring they're standard Python types
    config['UP_PROB_THRESHOLD'] = float(best_thresholds['UP_PROB_THRESHOLD'])
    config['DOWN_PROB_THRESHOLD'] = float(best_thresholds['DOWN_PROB_THRESHOLD'])

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Updated configuration file with best thresholds at {config_path}")

def run_threshold_tuning():
    """Run the full threshold tuning process."""
    # Load configuration
    config = load_app_config()
    
    # Fetch and prepare data
    logger.info("Fetching and preparing data...")
    
    # 1. Fetch Market Data
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    raw_df = fetch_polygon_data(
        api_key=config['POLYGON_API_KEY'],
        ticker=config['TICKER'],
        timespan=config['TIMEFRAME'],
        multiplier=config['MULTIPLIER'],
        start_date_str=config['START_DATE'],
        end_date_str=end_date_str
    )
    
    if raw_df.empty:
        logger.error(f"No market data fetched for {config['TICKER']}. Exiting.")
        return
    
    logger.info(f"Raw market data shape: {raw_df.shape}")
    
    # 2. Feature Engineering
    logger.info("Performing feature engineering...")
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    df_featured = add_ict_features_v2(df_with_basic_ta.copy(), swing_order=config['SWING_POINT_ORDER'])
    logger.info(f"Data shape after TA and ICT features: {df_featured.shape}")
    
    # 3. Labeling Data
    logger.info("Labeling data...")
    X_all, y_all = prepare_features_and_labels(
        df_featured.copy(),
        look_forward_candles=config['LOOK_FORWARD_CANDLES'],
        pt_atr_multiplier=config['PROFIT_TAKE_ATR'],
        sl_atr_multiplier=config['STOP_LOSS_ATR']
    )
    
    if X_all.empty or y_all.empty:
        logger.error("Not enough data after feature engineering and labeling. Exiting.")
        return
    
    logger.info(f"Shape of X_all: {X_all.shape}, Shape of y_all: {y_all.shape}")
    logger.info(f"Label distribution (y_all):\n{y_all.value_counts(normalize=True)}")
    
    # 4. Coarse threshold grid search
    logger.info("Running coarse threshold grid search...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coarse_dir = f"./validation_results/thresholds_coarse_{timestamp}"
    coarse_best = run_threshold_grid_search(X_all, y_all, config, results_dir=coarse_dir)
    
    # 5. Fine-grained threshold search
    logger.info("Running fine-grained threshold search...")
    fine_dir = f"./validation_results/thresholds_fine_{timestamp}"
    fine_best = fine_tune_thresholds(X_all, y_all, config, coarse_best, results_dir=fine_dir)
    
    # 6. Final evaluation
    logger.info("Evaluating final thresholds...")
    final_dir = f"./validation_results/thresholds_final_{timestamp}"
    final_metrics = evaluate_final_thresholds(X_all, y_all, config, fine_best, results_dir=final_dir)
    
    # 7. Update configuration
    logger.info("Updating configuration with best thresholds...")
    update_config_with_thresholds(fine_best)
    
    logger.info(f"Threshold tuning complete. Best thresholds: UP={fine_best['UP_PROB_THRESHOLD']}, DOWN={fine_best['DOWN_PROB_THRESHOLD']}")
    logger.info(f"Final profit factor: {final_metrics['profit_factor']:.2f}")

if __name__ == "__main__":
    run_threshold_tuning()