import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data, get_economic_calendar_from_api
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
from .model_trainer import train_binary_xgb_model, evaluate_model, get_feature_importances
from .utils import is_high_impact_news_active # Assuming this is moved to utils or directly used

def run_strategy():
    print("--- Starting Trading Strategy Pipeline ---")
    
    # 1. Load Configuration
    print("\n[1. Loading Configuration]")
    config = load_app_config()
    # print(f"Config: {config}") # For debugging

    # 2. Fetch Market Data
    print("\n[2. Fetching Market Data]")
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
        print(f"No market data fetched for {config['TICKER']}. Exiting.")
        return
    print(f"Raw market data shape: {raw_df.shape}")

    # 3. Fetch News Data (Placeholder)
    print("\n[3. Fetching News Data (Placeholder)]")
    # Determine news data range based on market data
    news_start_date = raw_df.index.min().strftime('%Y-%m-%d') if not raw_df.empty else config['START_DATE']
    news_end_date = raw_df.index.max().strftime('%Y-%m-%d') if not raw_df.empty else end_date_str
    
    news_calendar_df = get_economic_calendar_from_api(
        api_key=config.get('NEWS_API_KEY'), # Use .get for optional keys
        start_date=news_start_date,
        end_date=news_end_date,
        currencies=config.get('NEWS_AFFECTED_CURRENCIES', []) # e.g. ['USD', 'EUR'] for EURUSD
    )
    if not news_calendar_df.empty:
        print(f"News calendar data shape: {news_calendar_df.shape}")
    else:
        print("No news calendar data fetched (or placeholder returned empty).")


    # 4. Feature Engineering
    print("\n[4. Performing Feature Engineering]")
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    df_featured = add_ict_features_v2(df_with_basic_ta.copy(), swing_order=config['SWING_POINT_ORDER'])
    print(f"Data shape after TA and ICT features: {df_featured.shape}")

    # 5. Labeling Data
    print("\n[5. Labeling Data]")
    X_all, y_all = prepare_features_and_labels(
        df_featured.copy(),
        look_forward_candles=config['LOOK_FORWARD_CANDLES'],
        pt_atr_multiplier=config['PROFIT_TAKE_ATR'],
        sl_atr_multiplier=config['STOP_LOSS_ATR']
    )
    if X_all.empty or y_all.empty:
        print("Not enough data after feature engineering and labeling. Exiting.")
        return
    print(f"Shape of X_all: {X_all.shape}, Shape of y_all: {y_all.shape}")
    print(f"Label distribution (y_all):\n{y_all.value_counts(normalize=True)}")

    # 6. Train/Test Split
    print("\n[6. Splitting Data into Train/Test]")
    tscv = TimeSeriesSplit(n_splits=config['N_SPLITS_CV'])
    # Using last split for this example; proper CV would iterate or use a dedicated validation set
    try:
        train_index, test_index = list(tscv.split(X_all, y_all))[-1]
    except ValueError as e:
        print(f"Error during TimeSeriesSplit (likely too few samples: {len(X_all)}): {e}")
        print("Consider fetching more data or reducing N_SPLITS_CV.")
        return

    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train_orig, y_test_orig = y_all.iloc[train_index], y_all.iloc[test_index]
    print(f"X_train: {X_train.shape}, y_train_orig: {y_train_orig.shape}")
    print(f"X_test: {X_test.shape}, y_test_orig: {y_test_orig.shape}")

    if X_train.empty or X_test.empty:
        print("Train or test set is empty after split. Exiting.")
        return

    # 7. Model Training
    print("\n[7. Training Models]")
    # Check if we have separate parameters for UP and DOWN models
    has_separate_params = 'XGB_PARAMS_UP' in config and 'XGB_PARAMS_DOWN' in config

    # If separate params exist, use them, otherwise use common params
    xgb_params_up = config['XGB_PARAMS_UP'] if has_separate_params else config['XGB_PARAMS']
    xgb_params_down = config['XGB_PARAMS_DOWN'] if has_separate_params else config['XGB_PARAMS']

    print(f"Using {'separate' if has_separate_params else 'common'} XGBoost parameters for UP and DOWN models")

    # Model UP
    y_train_up = (y_train_orig == 1).astype(int)
    y_test_up = (y_test_orig == 1).astype(int)
    up_neg_count = (y_train_up == 0).sum(); up_pos_count = (y_train_up == 1).sum()
    scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
    print(f"Training Model UP (Positives: {up_pos_count}, Negatives: {up_neg_count}, SPW: {scale_pos_weight_up:.2f})")
    model_up = train_binary_xgb_model(X_train, y_train_up, xgb_params_up, scale_pos_weight_up)

    # Model DOWN
    y_train_down = (y_train_orig == -1).astype(int)
    y_test_down = (y_test_orig == -1).astype(int)
    down_neg_count = (y_train_down == 0).sum(); down_pos_count = (y_train_down == 1).sum()
    scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1
    print(f"Training Model DOWN (Positives: {down_pos_count}, Negatives: {down_neg_count}, SPW: {scale_pos_weight_down:.2f})")
    model_down = train_binary_xgb_model(X_train, y_train_down, xgb_params_down, scale_pos_weight_down)

    # 8. Model Evaluation & Signal Generation
    print("\n[8. Evaluating Models and Generating Signals on Test Set]")
    probs_up_test, _ = evaluate_model(model_up, X_test, y_test_up, "Model UP", config['UP_PROB_THRESHOLD'], verbose=True)
    probs_down_test, _ = evaluate_model(model_down, X_test, y_test_down, "Model DOWN", config['DOWN_PROB_THRESHOLD'], verbose=True)

    # Display Feature Importances
    fi_up = get_feature_importances(model_up, X_train.columns)
    if fi_up is not None:
        print("\nModel UP - Feature Importances (Top 10):")
        print(fi_up.head(10))

    # Signal Generation Logic
    final_signals = []
    for i in range(len(X_test)):
        idx_timestamp = X_test.index[i]
        signal = "NEUTRAL"

        if is_high_impact_news_active(
            idx_timestamp, 
            news_calendar_df, 
            config.get('NEWS_AFFECTED_CURRENCIES', []),
            window_minutes_before=config.get('NEWS_WINDOW_BEFORE_MIN', 30),
            window_minutes_after=config.get('NEWS_WINDOW_AFTER_MIN', 30)
            ):
            signal = "NEUTRAL (News)"
        else:
            prob_up = probs_up_test[i]
            prob_down = probs_down_test[i]
            
            # Ensure thresholds are float
            up_thresh = float(config['UP_PROB_THRESHOLD'])
            down_thresh = float(config['DOWN_PROB_THRESHOLD'])

            if prob_up >= up_thresh and prob_down < (1.0 - down_thresh):
                signal = "LONG"
            elif prob_down >= down_thresh and prob_up < (1.0 - up_thresh):
                signal = "SHORT"
        final_signals.append(signal)

    results_df = pd.DataFrame({
        'Actual_Outcome': y_test_orig,
        'Prob_UP': probs_up_test,
        'Prob_DOWN': probs_down_test,
        'Signal': final_signals
    }, index=X_test.index)

    print("\n--- Combined Signal Results (Sample from Test Set) ---")
    print(results_df.head(20))
    print("\nSignal Distribution on Test Set:")
    print(results_df['Signal'].value_counts())

    # Basic performance of signals (needs proper backtesting)
    long_trades_results = results_df[results_df['Signal'] == 'LONG']['Actual_Outcome']
    short_trades_results = results_df[results_df['Signal'] == 'SHORT']['Actual_Outcome']

    if not long_trades_results.empty:
        print("\nOutcomes for Generated LONG Signals:")
        print(long_trades_results.value_counts(normalize=True))
        long_wins = (long_trades_results == 1).sum(); long_losses = (long_trades_results == -1).sum()
        print(f"Long Signal Wins (Outcome 1): {long_wins}, Losses (Outcome -1): {long_losses}")

    if not short_trades_results.empty:
        print("\nOutcomes for Generated SHORT Signals:")
        print(short_trades_results.value_counts(normalize=True))
        # For short signals, outcome '-1' is a "win" from the original labeling's perspective
        short_wins = (short_trades_results == -1).sum(); short_losses = (short_trades_results == 1).sum()
        print(f"Short Signal Wins (Outcome -1): {short_wins}, Losses (Outcome 1): {short_losses}")

    print("\n--- Strategy Pipeline Finished ---")
    print("Next steps: Implement robust backtesting, hyperparameter optimization, and further analysis.")


if __name__ == "__main__":
    # Ensure environment variables (POLYGON_API_KEY, etc.) are set
    # You might want to create a .env file in the project root:
    # POLYGON_API_KEY=your_polygon_key
    # FOREXNEWSAPI_KEY=your_news_api_key (if using)
    run_strategy()
