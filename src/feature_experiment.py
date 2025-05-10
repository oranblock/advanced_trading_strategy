import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .json_utils import save_json

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
try:
    from .advanced_features import add_advanced_ict_features
except ImportError:
    logger.error("Could not import advanced_features module. Make sure it exists and is properly implemented.")
    def add_advanced_ict_features(df, config):
        """Placeholder for advanced features if the module is not available."""
        logger.warning("Using placeholder for add_advanced_ict_features")
        return df.copy()
from .validation_framework import StrategyValidator
from .model_trainer import train_binary_xgb_model, evaluate_model, get_feature_importances

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('feature_experiment')

class FeatureExperimenter:
    """Class for experimenting with different feature sets and analyzing their impact."""
    
    def __init__(self, config, results_dir="./feature_experiment_results"):
        """
        Initialize the experimenter.
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "feature_importance").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        
        # Track results
        self.experiment_results = {}
        
        # Load data
        logger.info("Fetching and preparing data...")
        self.raw_df = self._fetch_data()
        
        if self.raw_df is None or self.raw_df.empty:
            logger.error("Failed to fetch data. Exiting.")
            return
        
        logger.info(f"Raw data shape: {self.raw_df.shape}")
    
    def _fetch_data(self):
        """Fetch market data from Polygon API."""
        end_date_str = datetime.now().strftime("%Y-%m-%d")
        raw_df = fetch_polygon_data(
            api_key=self.config['POLYGON_API_KEY'],
            ticker=self.config['TICKER'],
            timespan=self.config['TIMEFRAME'],
            multiplier=self.config['MULTIPLIER'],
            start_date_str=self.config['START_DATE'],
            end_date_str=end_date_str
        )
        return raw_df
    
    def run_baseline_experiment(self):
        """Run experiment with baseline features only."""
        logger.info("Running baseline experiment...")
        
        # Generate basic features
        df_with_basic_ta = add_basic_ta_features(self.raw_df.copy())
        
        # Add basic ICT features
        df_featured = add_ict_features_v2(
            df_with_basic_ta.copy(), 
            swing_order=self.config['SWING_POINT_ORDER']
        )
        
        # Prepare features and labels
        X_all, y_all = prepare_features_and_labels(
            df_featured.copy(),
            look_forward_candles=self.config['LOOK_FORWARD_CANDLES'],
            pt_atr_multiplier=self.config['PROFIT_TAKE_ATR'],
            sl_atr_multiplier=self.config['STOP_LOSS_ATR']
        )
        
        # Save feature list
        baseline_features = X_all.columns.tolist()
        save_json(baseline_features, self.results_dir / "baseline_features.json")
        
        # Evaluate model performance
        metrics = self._evaluate_performance(X_all, y_all, "baseline")
        self.experiment_results["baseline"] = metrics
        
        logger.info(f"Baseline experiment complete. Metrics: {metrics}")
        return metrics
    
    def run_advanced_experiment(self):
        """Run experiment with advanced ICT features."""
        logger.info("Running advanced features experiment...")
        
        # Generate basic features
        df_with_basic_ta = add_basic_ta_features(self.raw_df.copy())
        
        # Add basic ICT features
        df_featured = add_ict_features_v2(
            df_with_basic_ta.copy(), 
            swing_order=self.config['SWING_POINT_ORDER']
        )
        
        # Add advanced ICT features
        df_advanced = add_advanced_ict_features(df_featured.copy(), self.config)

        # Debug info
        logger.info(f"Shape after adding advanced features: {df_advanced.shape}")

        # Check for empty dataframe
        if df_advanced.empty:
            logger.error("Advanced features DataFrame is empty. Cannot continue.")
            return {"error": "Empty DataFrame after advanced features"}

        # Check for all NaN values
        if df_advanced.isna().all().all():
            logger.error("Advanced features DataFrame contains all NaN values.")
            return {"error": "All NaN values in DataFrame"}

        # Prepare features and labels
        X_all, y_all = prepare_features_and_labels(
            df_advanced.copy(),
            look_forward_candles=self.config['LOOK_FORWARD_CANDLES'],
            pt_atr_multiplier=self.config['PROFIT_TAKE_ATR'],
            sl_atr_multiplier=self.config['STOP_LOSS_ATR']
        )

        # Check if we have valid data after feature preparation
        logger.info(f"Shape of X_all after preparation: {X_all.shape}, y_all: {y_all.shape}")
        if X_all.empty or y_all.empty:
            logger.error("No data after feature preparation. Cannot continue.")
            return {"error": "Empty feature set after preparation"}

        # Clean up features for model compatibility
        logger.info("Cleaning up advanced features for model compatibility")

        # Convert object columns to numeric when possible
        for col in X_all.select_dtypes(include=['object', 'category']).columns:
            try:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {e}")
                # Drop columns that can't be converted to numeric
                X_all = X_all.drop(columns=[col])

        # Fill NaN values
        X_all = X_all.fillna(0)
        
        # Save feature list
        advanced_features = X_all.columns.tolist()
        save_json(advanced_features, self.results_dir / "advanced_features.json")
        
        # Find new features (not in baseline)
        baseline_features = []
        try:
            with open(self.results_dir / "baseline_features.json", 'r') as f:
                baseline_features = json.load(f)
        except FileNotFoundError:
            logger.warning("Baseline features file not found. Run baseline experiment first.")
        
        new_features = [f for f in advanced_features if f not in baseline_features]
        logger.info(f"Added {len(new_features)} new features.")
        save_json(new_features, self.results_dir / "new_features.json")
        
        # Evaluate model performance
        metrics = self._evaluate_performance(X_all, y_all, "advanced")
        self.experiment_results["advanced"] = metrics
        
        logger.info(f"Advanced experiment complete. Metrics: {metrics}")
        return metrics
    
    def run_feature_group_experiments(self):
        """Run experiments with different feature groups to assess their individual impact."""
        logger.info("Running feature group experiments...")
        
        # Generate basic features
        df_with_basic_ta = add_basic_ta_features(self.raw_df.copy())
        
        # Add basic ICT features
        df_basic_ict = add_ict_features_v2(
            df_with_basic_ta.copy(), 
            swing_order=self.config['SWING_POINT_ORDER']
        )
        
        # Create base DataFrame for all experiments
        df_base = df_basic_ict.copy()
        
        # Try to import feature functions, with fallbacks for each
        try:
            from .advanced_features import add_session_features
        except (ImportError, AttributeError):
            logger.warning("Session features not available, creating placeholder")
            def add_session_features(df, config):
                logger.info("Using placeholder for session features")
                return df.copy()

        try:
            from .advanced_features import add_external_liquidity_features
        except (ImportError, AttributeError):
            logger.warning("External liquidity features not available, creating placeholder")
            def add_external_liquidity_features(df, config):
                logger.info("Using placeholder for external liquidity features")
                return df.copy()

        try:
            from .advanced_features import add_advanced_fvg_features
        except (ImportError, AttributeError):
            logger.warning("Advanced FVG features not available, creating placeholder")
            def add_advanced_fvg_features(df, config):
                logger.info("Using placeholder for advanced FVG features")
                return df.copy()

        try:
            from .advanced_features import add_premium_discount_features
        except (ImportError, AttributeError):
            logger.warning("Premium discount features not available, creating placeholder")
            def add_premium_discount_features(df, config):
                logger.info("Using placeholder for premium discount features")
                return df.copy()

        # Define feature groups to test
        feature_groups = {
            "session_features": lambda df: add_session_features(df, self.config),
            "external_liquidity": lambda df: add_external_liquidity_features(df, self.config),
            "advanced_fvg": lambda df: add_advanced_fvg_features(df, self.config),
            "premium_discount": lambda df: add_premium_discount_features(df, self.config)
        }
        
        # Run experiment for each feature group
        for group_name, feature_func in feature_groups.items():
            logger.info(f"Testing feature group: {group_name}")
            
            # Add only this feature group
            try:
                df_group = feature_func(df_base.copy())

                # Verify we have data
                if df_group.empty:
                    logger.warning(f"Empty DataFrame after adding {group_name} features")
                    df_group = df_base.copy()  # Use base features if this group fails
            except Exception as e:
                logger.error(f"Error applying {group_name} features: {e}")
                # Fall back to base features
                df_group = df_base.copy()
            
            # Prepare features and labels
            X_all, y_all = prepare_features_and_labels(
                df_group.copy(),
                look_forward_candles=self.config['LOOK_FORWARD_CANDLES'],
                pt_atr_multiplier=self.config['PROFIT_TAKE_ATR'],
                sl_atr_multiplier=self.config['STOP_LOSS_ATR']
            )

            # Clean up features for model compatibility
            if not X_all.empty:
                logger.info(f"Cleaning up features for {group_name}")

                # Convert object columns to numeric when possible
                for col in X_all.select_dtypes(include=['object', 'category']).columns:
                    try:
                        X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric: {e}")
                        # Drop columns that can't be converted to numeric
                        X_all = X_all.drop(columns=[col])

                # Fill NaN values
                X_all = X_all.fillna(0)
            
            # Evaluate model performance
            metrics = self._evaluate_performance(X_all, y_all, group_name)
            self.experiment_results[group_name] = metrics
            
            logger.info(f"{group_name} experiment complete. Metrics: {metrics}")
        
        return self.experiment_results
    
    def _evaluate_performance(self, X_all, y_all, experiment_name):
        """
        Evaluate model performance using the validation framework.

        Args:
            X_all: Feature DataFrame
            y_all: Target series
            experiment_name: Name of the experiment for logging

        Returns:
            dict: Performance metrics
        """
        # Check if there's sufficient data
        if len(X_all) < 3:
            logger.error(f"Not enough data points for experiment {experiment_name}. X_all shape: {X_all.shape}")
            return {"error": "Insufficient data"}

        # Initialize validator
        validator = StrategyValidator(
            X_all,
            y_all,
            self.config,
            results_dir=str(self.results_dir / f"metrics/{experiment_name}")
        )

        # Create 70/15/15 train/val/test split
        train_size = int(len(X_all) * 0.7)
        val_size = int(len(X_all) * 0.15)

        # Ensure we have some data in each split
        if train_size < 1 or val_size < 1 or (len(X_all) - train_size - val_size) < 1:
            logger.error(f"Split sizes too small: train={train_size}, val={val_size}, test={len(X_all)-train_size-val_size}")
            return {"error": "Split sizes too small"}

        # Use iloc for proper DataFrame/Series slicing
        X_train = X_all.iloc[:train_size]
        y_train = y_all.iloc[:train_size]

        X_val = X_all.iloc[train_size:train_size+val_size]
        y_val = y_all.iloc[train_size:train_size+val_size]

        X_test = X_all.iloc[train_size+val_size:]
        y_test = y_all.iloc[train_size+val_size:]
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        # Train UP model
        y_train_up = (y_train == 1).astype(int)
        y_val_up = (y_val == 1).astype(int)
        
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
        
        logger.info(f"Training UP model: positives={up_pos_count}, negatives={up_neg_count}")
        
        # Check if we have separate parameters for UP model
        xgb_params_up = self.config.get('XGB_PARAMS_UP', self.config['XGB_PARAMS'])
        
        model_up = train_binary_xgb_model(X_train, y_train_up, xgb_params_up, scale_pos_weight_up)
        
        # Train DOWN model
        y_train_down = (y_train == -1).astype(int)
        y_val_down = (y_val == -1).astype(int)
        
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1
        
        logger.info(f"Training DOWN model: positives={down_pos_count}, negatives={down_neg_count}")
        
        # Check if we have separate parameters for DOWN model
        xgb_params_down = self.config.get('XGB_PARAMS_DOWN', self.config['XGB_PARAMS'])
        
        model_down = train_binary_xgb_model(X_train, y_train_down, xgb_params_down, scale_pos_weight_down)
        
        # Save feature importances
        self._save_feature_importance(
            model_up, 
            X_train.columns, 
            f"{experiment_name}_up"
        )
        self._save_feature_importance(
            model_down, 
            X_train.columns, 
            f"{experiment_name}_down"
        )
        
        # Evaluate on validation set
        probs_up_val, _ = evaluate_model(
            model_up, 
            X_val, 
            y_val_up, 
            f"{experiment_name} - UP (Val)", 
            self.config['UP_PROB_THRESHOLD'],
            verbose=False
        )
        
        probs_down_val, _ = evaluate_model(
            model_down, 
            X_val, 
            y_val_down, 
            f"{experiment_name} - DOWN (Val)", 
            self.config['DOWN_PROB_THRESHOLD'],
            verbose=False
        )
        
        # Generate signals on validation set
        val_signals = []
        for i in range(len(X_val)):
            signal = "NEUTRAL"
            prob_up = probs_up_val[i]
            prob_down = probs_down_val[i]
            
            up_thresh = float(self.config['UP_PROB_THRESHOLD'])
            down_thresh = float(self.config['DOWN_PROB_THRESHOLD'])
            
            if prob_up >= up_thresh and prob_down < (1.0 - down_thresh):
                signal = "LONG"
            elif prob_down >= down_thresh and prob_up < (1.0 - up_thresh):
                signal = "SHORT"
            
            val_signals.append(signal)
        
        # Calculate validation metrics
        val_results = pd.DataFrame({
            'Actual_Outcome': y_val,
            'Signal': val_signals
        })
        
        val_metrics = self._calculate_metrics(val_results)
        
        # Evaluate on test set
        probs_up_test, _ = evaluate_model(
            model_up, 
            X_test, 
            (y_test == 1).astype(int), 
            f"{experiment_name} - UP (Test)", 
            self.config['UP_PROB_THRESHOLD'],
            verbose=True  # Print detailed metrics for test set
        )
        
        probs_down_test, _ = evaluate_model(
            model_down, 
            X_test, 
            (y_test == -1).astype(int), 
            f"{experiment_name} - DOWN (Test)", 
            self.config['DOWN_PROB_THRESHOLD'],
            verbose=True  # Print detailed metrics for test set
        )
        
        # Generate signals on test set
        test_signals = []
        for i in range(len(X_test)):
            signal = "NEUTRAL"
            prob_up = probs_up_test[i]
            prob_down = probs_down_test[i]
            
            up_thresh = float(self.config['UP_PROB_THRESHOLD'])
            down_thresh = float(self.config['DOWN_PROB_THRESHOLD'])
            
            if prob_up >= up_thresh and prob_down < (1.0 - down_thresh):
                signal = "LONG"
            elif prob_down >= down_thresh and prob_up < (1.0 - up_thresh):
                signal = "SHORT"
            
            test_signals.append(signal)
        
        # Calculate test metrics
        test_results = pd.DataFrame({
            'Actual_Outcome': y_test,
            'Signal': test_signals
        })
        
        test_metrics = self._calculate_metrics(test_results)
        
        # Store results including both validation and test metrics
        combined_metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "feature_count": len(X_train.columns)
        }
        
        # Save results to file
        save_json(combined_metrics, self.results_dir / f"metrics/{experiment_name}/metrics.json")
        
        return combined_metrics
    
    def _calculate_metrics(self, results_df):
        """
        Calculate performance metrics from signal results.
        
        Args:
            results_df: DataFrame with signals and outcomes
            
        Returns:
            dict: Performance metrics
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
        }
        
        return metrics
    
    def _save_feature_importance(self, model, feature_names, model_name):
        """
        Save feature importance visualization and data.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model for file naming
        """
        importances = get_feature_importances(model, feature_names)
        
        if importances is not None:
            # Save importance data
            importances.to_csv(self.results_dir / f"feature_importance/{model_name}_importance.csv")
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Plot top 30 features
            top_n = min(30, len(importances))
            sns.barplot(
                x='importance',
                y='feature',
                data=importances.head(top_n),
                palette='viridis'
            )
            
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"plots/{model_name}_importance.png", dpi=300)
            plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments."""
        if not self.experiment_results:
            logger.warning("No experiment results to summarize.")
            return
        
        # Create comparison dataframe
        comparison_data = []
        
        for experiment_name, metrics in self.experiment_results.items():
            test_metrics = metrics.get('test', {})
            comparison_data.append({
                'Experiment': experiment_name,
                'Feature Count': metrics.get('feature_count', 0),
                'Profit Factor': test_metrics.get('profit_factor', 0),
                'Total Trades': test_metrics.get('total_trades', 0),
                'Long Win Rate': test_metrics.get('long_win_rate', 0),
                'Short Win Rate': test_metrics.get('short_win_rate', 0),
                'Total Wins': test_metrics.get('total_wins', 0),
                'Total Losses': test_metrics.get('total_losses', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison table
        df_comparison.to_csv(self.results_dir / "experiment_comparison.csv", index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot profit factor comparison
        sns.barplot(
            x='Experiment',
            y='Profit Factor',
            data=df_comparison,
            palette='viridis'
        )
        
        plt.title('Profit Factor Comparison Across Experiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots/profit_factor_comparison.png", dpi=300)
        plt.close()
        
        # Create win rate comparison
        plt.figure(figsize=(12, 6))
        
        # Reshape data for grouped bar chart
        win_rate_data = []
        for _, row in df_comparison.iterrows():
            win_rate_data.append({
                'Experiment': row['Experiment'],
                'Win Rate': row['Long Win Rate'],
                'Direction': 'Long'
            })
            win_rate_data.append({
                'Experiment': row['Experiment'],
                'Win Rate': row['Short Win Rate'],
                'Direction': 'Short'
            })
        
        win_rate_df = pd.DataFrame(win_rate_data)
        
        # Plot grouped bar chart
        sns.barplot(
            x='Experiment',
            y='Win Rate',
            hue='Direction',
            data=win_rate_df,
            palette='Set1'
        )
        
        plt.title('Win Rate Comparison Across Experiments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots/win_rate_comparison.png", dpi=300)
        plt.close()
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Feature Experiment Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Feature Experiment Results</h1>
            
            <h2>Experiment Comparison</h2>
            <table>
                <tr>
                    <th>Experiment</th>
                    <th>Feature Count</th>
                    <th>Profit Factor</th>
                    <th>Total Trades</th>
                    <th>Long Win Rate</th>
                    <th>Short Win Rate</th>
                    <th>Total Wins</th>
                    <th>Total Losses</th>
                </tr>
                {"".join(f"<tr><td>{row['Experiment']}</td><td>{row['Feature Count']}</td><td>{row['Profit Factor']:.2f}</td><td>{row['Total Trades']}</td><td>{row['Long Win Rate']:.2f}</td><td>{row['Short Win Rate']:.2f}</td><td>{row['Total Wins']}</td><td>{row['Total Losses']}</td></tr>" for _, row in df_comparison.iterrows())}
            </table>
            
            <h2>Profit Factor Comparison</h2>
            <img src="plots/profit_factor_comparison.png" alt="Profit Factor Comparison">
            
            <h2>Win Rate Comparison</h2>
            <img src="plots/win_rate_comparison.png" alt="Win Rate Comparison">
            
            <h2>Feature Importance</h2>
            {"".join(f"<h3>{experiment} Model</h3><img src='plots/{experiment}_up_importance.png'><img src='plots/{experiment}_down_importance.png'>" for experiment in self.experiment_results.keys())}
        </body>
        </html>
        """
        
        with open(self.results_dir / "experiment_report.html", 'w') as f:
            f.write(html_report)
        
        logger.info(f"Summary report generated at {self.results_dir / 'experiment_report.html'}")
        
        return df_comparison

def run_feature_experiments(quick_mode=False):
    """
    Run a series of feature experiments to evaluate the impact of advanced ICT features.

    Args:
        quick_mode: If True, run a simplified version with fewer experiments

    Returns:
        DataFrame with comparison of experiments
    """
    # Load configuration
    config = load_app_config()

    # Create timestamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./validation_results/feature_experiments_{timestamp}"

    # Initialize experimenter
    experimenter = FeatureExperimenter(config, results_dir=results_dir)

    # Run baseline experiment
    logger.info("Running baseline feature experiment")
    experimenter.run_baseline_experiment()

    # Run advanced features experiment
    logger.info("Running advanced features experiment")
    experimenter.run_advanced_experiment()

    # Run feature group experiments if not in quick mode
    if not quick_mode:
        logger.info("Running feature group experiments")
        experimenter.run_feature_group_experiments()
    else:
        logger.info("Skipping feature group experiments (quick mode)")

    # Generate summary report
    logger.info("Generating summary report")
    comparison = experimenter.generate_summary_report()

    logger.info("Feature experiments complete!")
    logger.info(f"Results available in: {results_dir}")

    return comparison

if __name__ == "__main__":
    run_feature_experiments()