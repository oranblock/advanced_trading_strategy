import os
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = "config/settings.yaml" # Relative to project root

def load_app_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads application configuration from a YAML file and .env file."""
    config = {}
    
    # Load from .env file first (for secrets like API keys)
    load_dotenv() # Loads .env from current directory or parent

    # Load API keys from environment variables
    config['POLYGON_API_KEY'] = os.environ.get("POLYGON_API_KEY")
    config['NEWS_API_KEY'] = os.environ.get("FOREXNEWSAPI_KEY") # Example

    if not config['POLYGON_API_KEY']:
        print("Warning: POLYGON_API_KEY not found in environment variables.")
        # raise ValueError("POLYGON_API_KEY must be set in environment variables or .env file")

    # Load other settings from YAML file
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using defaults and environment variables.")
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")

    # Set defaults if not found in YAML (can be overridden by env vars if structured so)
    config.setdefault('TICKER', "C:EURUSD")
    config.setdefault('TIMEFRAME', "hour")
    config.setdefault('MULTIPLIER', 1)
    config.setdefault('START_DATE', "2020-01-01")
    # END_DATE can be dynamic
    config.setdefault('SWING_POINT_ORDER', 10)
    config.setdefault('LOOK_FORWARD_CANDLES', 24)
    config.setdefault('PROFIT_TAKE_ATR', 2.5)
    config.setdefault('STOP_LOSS_ATR', 1.5)
    config.setdefault('UP_PROB_THRESHOLD', 0.55)
    config.setdefault('DOWN_PROB_THRESHOLD', 0.55)
    config.setdefault('N_SPLITS_CV', 5)
    config.setdefault('XGB_PARAMS', {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    })
    config.setdefault('NEWS_WINDOW_BEFORE_MIN', 30)
    config.setdefault('NEWS_WINDOW_AFTER_MIN', 30)
    config.setdefault('NEWS_AFFECTED_CURRENCIES', ['USD', 'EUR']) # Example for EURUSD

    return config

if __name__ == '__main__':
    # Example usage:
    # Create a dummy .env file for testing
    with open(".env", "w") as f:
        f.write("POLYGON_API_KEY=YOUR_POLYGON_KEY_HERE\n")
        f.write("FOREXNEWSAPI_KEY=YOUR_NEWS_KEY_HERE\n")
    
    # Create a dummy config/settings.yaml for testing
    os.makedirs("config", exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump({'TICKER': 'C:GBPUSD', 'LOOK_FORWARD_CANDLES': 12}, f)

    app_config = load_app_config()
    print("Loaded Configuration:")
    for key, value in app_config.items():
        print(f"  {key}: {value}")
    
    # Clean up dummy files
    # os.remove(".env")
    # os.remove(DEFAULT_CONFIG_PATH)
    # os.rmdir("config")
    print("\nTo use this, create a real .env file with your API keys and optionally a config/settings.yaml")
