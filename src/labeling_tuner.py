import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from .json_utils import save_json

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
from .model_trainer import train_binary_xgb_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('labeling_tuner')

class LabelingParameterTuner:
    """Tuner for optimizing labeling parameters (look forward, PT/SL multipliers)."""
    
    def __init__(self, raw_df, config, results_dir="./labeling_tuning_results"):
        """
        Initialize the tuner with raw data and configuration.
        
        Args:
            raw_df: DataFrame with raw OHLC data
            config: Trading strategy configuration
            results_dir: Directory to save results
        """
        self.raw_df = raw_df
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Fixed model parameters
        self.up_threshold = float(config.get('UP_PROB_THRESHOLD', 0.55))
        self.down_threshold = float(config.get('DOWN_PROB_THRESHOLD', 0.55))
        
        # DataFrame to store results
        self.results = []
        
        # Prepare data with features once (we'll relabel this later)
        logger.info("Performing feature engineering...")
        df_with_basic_ta = add_basic_ta_features(self.raw_df.copy())
        self.df_featured = add_ict_features_v2(df_with_basic_ta.copy(), 
                                              swing_order=config['SWING_POINT_ORDER'])
        logger.info(f"Data shape after TA and ICT features: {self.df_featured.shape}")
    
    def _generate_model_signals(self, X_train, y_train_orig, X_val, y_val_orig):
        """
        Train models and generate signals for validation set.
        
        Args:
            X_train: Training features
            y_train_orig: Training labels (-1, 0, 1)
            X_val: Validation features
            y_val_orig: Validation labels (-1, 0, 1)
            
        Returns:
            results_df: DataFrame with signals and outcomes
        """
        # Create binary targets
        y_train_up = (y_train_orig == 1).astype(int)
        y_train_down = (y_train_orig == -1).astype(int)
        
        # Check if we have separate parameters for UP and DOWN models
        has_separate_params = 'XGB_PARAMS_UP' in self.config and 'XGB_PARAMS_DOWN' in self.config
        
        # If separate params exist, use them, otherwise use common params
        xgb_params_up = self.config['XGB_PARAMS_UP'] if has_separate_params else self.config['XGB_PARAMS']
        xgb_params_down = self.config['XGB_PARAMS_DOWN'] if has_separate_params else self.config['XGB_PARAMS']
        
        # Calculate scale_pos_weight for UP model
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
        
        # Train UP model
        model_up = train_binary_xgb_model(X_train, y_train_up, xgb_params_up, scale_pos_weight_up)
        
        # Calculate scale_pos_weight for DOWN model
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1
        
        # Train DOWN model
        model_down = train_binary_xgb_model(X_train, y_train_down, xgb_params_down, scale_pos_weight_down)
        
        # Predict probabilities
        X_val_float = X_val.astype(float)
        probs_up = model_up.predict_proba(X_val_float)[:, 1]
        probs_down = model_down.predict_proba(X_val_float)[:, 1]
        
        # Generate signals
        signals = []
        for i in range(len(X_val)):
            signal = "NEUTRAL"
            prob_up = probs_up[i]
            prob_down = probs_down[i]
            
            if prob_up >= self.up_threshold and prob_down < (1.0 - self.down_threshold):
                signal = "LONG"
            elif prob_down >= self.down_threshold and prob_up < (1.0 - self.up_threshold):
                signal = "SHORT"
            
            signals.append(signal)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Actual_Outcome': y_val_orig,
            'Prob_UP': probs_up,
            'Prob_DOWN': probs_down,
            'Signal': signals
        }, index=X_val.index)
        
        return results_df
    
    def _calculate_metrics(self, results_df):
        """
        Calculate performance metrics from signal results.
        
        Args:
            results_df: DataFrame with signals and outcomes
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        signal_counts = results_df['Signal'].value_counts()
        long_trades = results_df[results_df['Signal'] == 'LONG']
        short_trades = results_df[results_df['Signal'] == 'SHORT']
        
        long_win_rate = 0
        if not long_trades.empty:
            long_win_rate = (long_trades['Actual_Outcome'] == 1).mean()
        
        short_win_rate = 0
        if not short_trades.empty:
            short_win_rate = (short_trades['Actual_Outcome'] == -1).mean()
        
        # Calculate profit factor and other metrics
        long_wins = (long_trades['Actual_Outcome'] == 1).sum()
        long_losses = (long_trades['Actual_Outcome'] == -1).sum()
        short_wins = (short_trades['Actual_Outcome'] == -1).sum()
        short_losses = (short_trades['Actual_Outcome'] == 1).sum()
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        total_trades = len(long_trades) + len(short_trades)
        
        profit_factor = 0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        
        # Calculate label distribution for diagnostic purposes
        label_dist = results_df['Actual_Outcome'].value_counts(normalize=True).to_dict()
        
        metrics = {
            "long_win_rate": float(long_win_rate),
            "short_win_rate": float(short_win_rate),
            "long_trades": int(len(long_trades)),
            "short_trades": int(len(short_trades)),
            "total_trades": int(total_trades),
            "profit_factor": float(profit_factor),
            "long_wins": int(long_wins),
            "long_losses": int(long_losses),
            "short_wins": int(short_wins),
            "short_losses": int(short_losses),
            "total_wins": int(total_wins),
            "total_losses": int(total_losses),
            "neutral_count": int(signal_counts.get("NEUTRAL", 0)),
            "label_distribution": label_dist
        }
        
        return metrics
    
    def evaluate_labeling_params(self, look_forward, pt_atr, sl_atr):
        """
        Evaluate specific labeling parameters.
        
        Args:
            look_forward: Number of candles to look forward
            pt_atr: Profit target ATR multiplier
            sl_atr: Stop loss ATR multiplier
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        logger.info(f"Evaluating: look_forward={look_forward}, pt_atr={pt_atr}, sl_atr={sl_atr}")
        
        # Generate labels with the given parameters
        X_all, y_all = prepare_features_and_labels(
            self.df_featured.copy(),
            look_forward_candles=look_forward,
            pt_atr_multiplier=pt_atr,
            sl_atr_multiplier=sl_atr
        )
        
        if X_all.empty or y_all.empty:
            logger.warning("Not enough data after labeling!")
            return None
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Track metrics for each fold
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
            X_train, X_val = X_all.iloc[train_idx], X_all.iloc[test_idx]
            y_train_orig, y_val_orig = y_all.iloc[train_idx], y_all.iloc[test_idx]
            
            # Generate signals
            results_df = self._generate_model_signals(X_train, y_train_orig, X_val, y_val_orig)
            
            # Calculate metrics
            metrics = self._calculate_metrics(results_df)
            
            # Store metrics
            fold_metrics.append(metrics)
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            if key != "label_distribution":
                avg_metrics[key] = np.mean([fm[key] for fm in fold_metrics])
        
        # Add labeling parameters to metrics
        avg_metrics["look_forward_candles"] = look_forward
        avg_metrics["profit_take_atr"] = pt_atr
        avg_metrics["stop_loss_atr"] = sl_atr
        
        # Calculate label distribution across all data
        label_dist = y_all.value_counts(normalize=True).to_dict()
        avg_metrics["label_distribution"] = label_dist
        
        return avg_metrics
    
    def grid_search(self, look_forward_range, pt_atr_range, sl_atr_range):
        """
        Run grid search over labeling parameters.
        
        Args:
            look_forward_range: List of look_forward_candles values
            pt_atr_range: List of profit_take_atr values
            sl_atr_range: List of stop_loss_atr values
            
        Returns:
            best_params: Dictionary with best labeling parameters
        """
        logger.info(f"Starting labeling parameter grid search")
        logger.info(f"Look forward range: {look_forward_range}")
        logger.info(f"Profit take ATR range: {pt_atr_range}")
        logger.info(f"Stop loss ATR range: {sl_atr_range}")
        
        # Generate all parameter combinations
        param_grid = list(product(look_forward_range, pt_atr_range, sl_atr_range))
        total_combinations = len(param_grid)
        logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
        
        # Run grid search
        self.results = []
        for i, (look_forward, pt_atr, sl_atr) in enumerate(param_grid):
            logger.info(f"Evaluating combination {i+1}/{total_combinations}")
            
            # Add constraint: PT > SL generally makes sense
            if pt_atr <= sl_atr:
                logger.info(f"Skipping pt_atr={pt_atr}, sl_atr={sl_atr} (PT should be > SL)")
                continue
            
            # Calculate profit-to-risk ratio
            profit_risk_ratio = pt_atr / sl_atr
            logger.info(f"Profit-to-risk ratio: {profit_risk_ratio:.2f}")
            
            # Evaluate parameters
            metrics = self.evaluate_labeling_params(look_forward, pt_atr, sl_atr)
            
            if metrics is not None:
                # Add profit-to-risk ratio to metrics
                metrics["profit_risk_ratio"] = profit_risk_ratio
                self.results.append(metrics)
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            logger.error("No valid parameter combinations evaluated!")
            return None
        
        # Filter out combinations with too few trades
        min_trades = 10
        filtered_results = results_df[results_df['total_trades'] >= min_trades]
        
        # If no combinations have enough trades, use all results
        if filtered_results.empty:
            filtered_results = results_df
        
        # Find best parameters based on profit factor
        best_row = filtered_results.loc[filtered_results['profit_factor'].idxmax()]
        
        best_params = {
            'LOOK_FORWARD_CANDLES': int(best_row['look_forward_candles']),
            'PROFIT_TAKE_ATR': float(best_row['profit_take_atr']),
            'STOP_LOSS_ATR': float(best_row['stop_loss_atr'])
        }
        
        # Save results
        results_df.to_csv(self.results_dir / 'labeling_grid_results.csv', index=False)

        # Save best params using utility function
        save_json(best_params, self.results_dir / 'best_labeling_params.json')
        
        # Visualize results
        self._visualize_results(results_df)
        
        logger.info(f"Best labeling parameters: {best_params}")
        logger.info(f"Best profit factor: {best_row['profit_factor']:.2f}")
        
        return best_params
    
    def _visualize_results(self, results_df):
        """Create visualizations of grid search results."""
        try:
            # Create directory for figures
            (self.results_dir / 'figures').mkdir(exist_ok=True)
            
            # 1. Heatmap of profit factor by look_forward and profit_take_atr
            # For each stop_loss_atr value
            sl_values = sorted(results_df['stop_loss_atr'].unique())
            
            for sl in sl_values:
                sl_results = results_df[results_df['stop_loss_atr'] == sl]
                
                if not sl_results.empty:
                    try:
                        # Create pivot table
                        pivot = sl_results.pivot(
                            index='look_forward_candles',
                            columns='profit_take_atr',
                            values='profit_factor'
                        )
                        
                        # Plot heatmap
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
                        plt.title(f'Profit Factor (Stop Loss ATR: {sl})')
                        plt.xlabel('Profit Take ATR')
                        plt.ylabel('Look Forward Candles')
                        plt.tight_layout()
                        plt.savefig(self.results_dir / 'figures' / f'profit_factor_sl_{sl}.png')
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating heatmap for SL={sl}: {e}")
            
            # 2. Scatter plot of profit factor vs profit-to-risk ratio
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df['profit_risk_ratio'],
                results_df['profit_factor'],
                alpha=0.7,
                s=100,
                c=results_df['total_trades'],
                cmap='viridis'
            )
            plt.colorbar(label='Total Trades')
            plt.xlabel('Profit-to-Risk Ratio (PT/SL)')
            plt.ylabel('Profit Factor')
            plt.title('Profit Factor vs Profit-to-Risk Ratio')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'figures' / 'profit_factor_vs_risk_reward.png')
            plt.close()
            
            # 3. Bar plot of label distribution for different look_forward values
            lf_values = sorted(results_df['look_forward_candles'].unique())
            label_dists = []
            
            for lf in lf_values:
                lf_results = results_df[results_df['look_forward_candles'] == lf]
                if not lf_results.empty:
                    # Take the first row - label distribution should be the same for all rows with the same look_forward
                    label_dist = lf_results.iloc[0]['label_distribution']
                    if label_dist:
                        dist_df = pd.DataFrame({
                            'label': list(label_dist.keys()),
                            'frequency': list(label_dist.values()),
                            'look_forward': lf
                        })
                        label_dists.append(dist_df)
            
            if label_dists:
                # Combine all distributions
                all_dists = pd.concat(label_dists)
                
                # Plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x='look_forward', y='frequency', hue='label', data=all_dists)
                plt.xlabel('Look Forward Candles')
                plt.ylabel('Frequency')
                plt.title('Label Distribution by Look Forward Candles')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'figures' / 'label_distribution.png')
                plt.close()
            
            logger.info(f"Saved visualization figures to {self.results_dir / 'figures'}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def full_evaluation(self, best_params):
        """
        Perform a full evaluation with the best parameters.
        
        Args:
            best_params: Dictionary with best labeling parameters
            
        Returns:
            final_metrics: Dictionary with final performance metrics
        """
        logger.info(f"Performing full evaluation with best parameters: {best_params}")
        
        # Generate labels with the best parameters
        X_all, y_all = prepare_features_and_labels(
            self.df_featured.copy(),
            look_forward_candles=best_params['LOOK_FORWARD_CANDLES'],
            pt_atr_multiplier=best_params['PROFIT_TAKE_ATR'],
            sl_atr_multiplier=best_params['STOP_LOSS_ATR']
        )
        
        # Create a single train/test split (80/20)
        test_size = 0.2
        train_size = int(len(X_all) * (1 - test_size))
        
        X_train, X_test = X_all.iloc[:train_size], X_all.iloc[train_size:]
        y_train, y_test = y_all.iloc[:train_size], y_all.iloc[train_size:]
        
        # Generate signals
        results_df = self._generate_model_signals(X_train, y_train, X_test, y_test)
        
        # Calculate metrics
        final_metrics = self._calculate_metrics(results_df)
        
        # Save detailed results
        results_df.to_csv(self.results_dir / 'final_evaluation_signals.csv')
        
        # Save metrics using utility function
        save_json(final_metrics, self.results_dir / 'final_evaluation_metrics.json')
        
        logger.info(f"Final evaluation metrics:")
        logger.info(f"Profit Factor: {final_metrics['profit_factor']:.2f}")
        logger.info(f"Total Trades: {final_metrics['total_trades']}")
        logger.info(f"Long Win Rate: {final_metrics['long_win_rate']*100:.1f}%")
        logger.info(f"Short Win Rate: {final_metrics['short_win_rate']*100:.1f}%")
        
        return final_metrics

def update_config_with_labeling_params(best_params):
    """
    Update settings.yaml with the best labeling parameters.

    Args:
        best_params: Dictionary with best labeling parameters
    """
    import yaml
    import numpy as np

    config_path = Path("./config/settings.yaml")

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update labeling parameters, ensuring they're standard Python types
    for key, value in best_params.items():
        if isinstance(value, np.integer):
            config[key] = int(value)
        elif isinstance(value, np.floating):
            config[key] = float(value)
        else:
            config[key] = value

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Updated configuration file with best labeling parameters at {config_path}")

def run_labeling_tuning():
    """Run the labeling parameter tuning process."""
    # Load configuration
    config = load_app_config()
    
    # Fetch market data
    logger.info("Fetching market data...")
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
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./validation_results/labeling_tuning_{timestamp}"
    
    # Initialize tuner
    tuner = LabelingParameterTuner(raw_df, config, results_dir=results_dir)
    
    # Define parameter ranges
    look_forward_range = [12, 24, 36, 48]  # For hourly data: 12h, 24h, 36h, 48h
    pt_atr_range = [1.5, 2.0, 2.5, 3.0, 3.5]
    sl_atr_range = [1.0, 1.5, 2.0]
    
    # Run grid search
    best_params = tuner.grid_search(look_forward_range, pt_atr_range, sl_atr_range)
    
    if best_params is None:
        logger.error("Labeling parameter tuning failed!")
        return
    
    # Perform full evaluation with best parameters
    final_metrics = tuner.full_evaluation(best_params)
    
    # Update configuration
    update_config_with_labeling_params(best_params)
    
    # Generate report
    report = f"""# Labeling Parameter Tuning Results

## Best Parameters
- LOOK_FORWARD_CANDLES: {best_params['LOOK_FORWARD_CANDLES']}
- PROFIT_TAKE_ATR: {best_params['PROFIT_TAKE_ATR']:.1f}
- STOP_LOSS_ATR: {best_params['STOP_LOSS_ATR']:.1f}
- Profit-to-Risk Ratio: {best_params['PROFIT_TAKE_ATR']/best_params['STOP_LOSS_ATR']:.2f}

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
The above parameters represent the optimal balance of look-ahead period and risk-reward settings.
These parameters determine how the strategy labels historical data for training, which directly
impacts what the models learn to predict.
"""
    
    # Save report
    with open(Path(results_dir) / 'labeling_tuning_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"Saved labeling tuning report to {Path(results_dir) / 'labeling_tuning_report.md'}")
    logger.info("Labeling parameter tuning complete!")

if __name__ == "__main__":
    run_labeling_tuning()