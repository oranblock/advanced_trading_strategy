import pandas as pd
import numpy as np
import talib
from .utils import get_swing_points # Relative import

def add_basic_ta_features(df):
    """Adds some basic TA-Lib based features."""
    df_ta = df.copy()
    df_ta['SMA_20'] = talib.SMA(df_ta['close'], timeperiod=20)
    df_ta['RSI_14'] = talib.RSI(df_ta['close'], timeperiod=14)
    df_ta['ATR_14'] = talib.ATR(df_ta['high'], df_ta['low'], df_ta['close'], timeperiod=14)
    
    # Example: Add more indicators
    macd, macdsignal, macdhist = talib.MACD(df_ta['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_ta['MACD'] = macd
    df_ta['MACD_signal'] = macdsignal
    
    upper, middle, lower = talib.BBANDS(df_ta['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df_ta['BB_upper'] = upper
    df_ta['BB_middle'] = middle
    df_ta['BB_lower'] = lower
    df_ta['BB_width'] = (upper - lower) / middle if middle is not None and not pd.isna(middle).all() and not (middle == 0).all() else np.nan


    # Lagged returns
    for lag in [1, 3, 5]:
        df_ta[f'return_{lag}'] = df_ta['close'].pct_change(periods=lag)
        
    return df_ta

def add_ict_features_v2(df, swing_order=5):
    """Adds more sophisticated ICT-inspired features."""
    df_ict = df.copy()

    df_ict['hour_utc'] = df_ict.index.hour
    df_ict['is_london_kz'] = ((df_ict['hour_utc'] >= 7) & (df_ict['hour_utc'] <= 10)).astype(int)
    df_ict['is_ny_kz'] = ((df_ict['hour_utc'] >= 12) & (df_ict['hour_utc'] <= 15)).astype(int)
    df_ict['is_asia_kz'] = ((df_ict['hour_utc'] >= 0) & (df_ict['hour_utc'] <= 4)).astype(int) # Adjusted Asia KZ

    swing_highs_raw, swing_lows_raw = get_swing_points(df_ict, order=swing_order)
    df_ict['last_swing_high'] = swing_highs_raw.ffill()
    df_ict['last_swing_low'] = swing_lows_raw.ffill()

    df_ict['prev_last_swing_high'] = df_ict['last_swing_high'].shift(1)
    df_ict['prev_last_swing_low'] = df_ict['last_swing_low'].shift(1)

    df_ict['bullish_bos'] = (df_ict['close'] > df_ict['prev_last_swing_high']).astype(int)
    df_ict['bearish_bos'] = (df_ict['close'] < df_ict['prev_last_swing_low']).astype(int)
    
    df_ict['bullish_bos_last_3'] = df_ict['bullish_bos'].rolling(window=3, min_periods=1).sum().ge(1).astype(int)
    df_ict['bearish_bos_last_3'] = df_ict['bearish_bos'].rolling(window=3, min_periods=1).sum().ge(1).astype(int)

    df_ict['bullish_fvg_formed'] = (df_ict['low'] > df_ict['high'].shift(2)).astype(int)
    df_ict['bearish_fvg_formed'] = (df_ict['high'] < df_ict['low'].shift(2)).astype(int)

    df_ict['is_down_candle'] = (df_ict['close'] < df_ict['open'])
    df_ict['is_up_candle'] = (df_ict['close'] > df_ict['open'])

    df_ict['bullish_ob_formed'] = (df_ict['is_down_candle'].shift(1) & df_ict['bullish_bos']).astype(int)
    df_ict['bearish_ob_formed'] = (df_ict['is_up_candle'].shift(1) & df_ict['bearish_bos']).astype(int)
    
    df_ict['bullish_liq_sweep'] = ((df_ict['low'] < df_ict['prev_last_swing_low']) & \
                                   (df_ict['close'] > df_ict['prev_last_swing_low'])).astype(int)
    df_ict['bearish_liq_sweep'] = ((df_ict['high'] > df_ict['prev_last_swing_high']) & \
                                   (df_ict['close'] < df_ict['prev_last_swing_high'])).astype(int)

    cols_to_drop = ['hour_utc', 'prev_last_swing_high', 'prev_last_swing_low',
                    'is_down_candle', 'is_up_candle', 'last_swing_high', 'last_swing_low']
    df_ict = df_ict.drop(columns=cols_to_drop, errors='ignore')
    
    return df_ict

def prepare_features_and_labels(df, look_forward_candles=12, pt_atr_multiplier=2.0, sl_atr_multiplier=1.0):
    if 'ATR_14' not in df.columns:
        # Try to calculate it if missing and OHLC are present
        if all(col in df.columns for col in ['high', 'low', 'close']):
            print("Warning: ATR_14 missing, calculating on the fly for labeling.")
            df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            raise ValueError("ATR_14 column is required for labeling and OHLC not available to calculate it.")

    df_copy = df.copy()
    # Crucial: Drop NaNs from indicator calculations BEFORE attempting to label or select X
    df_copy = df_copy.dropna(subset=[col for col in df_copy.columns if col not in ['open', 'high', 'low', 'close', 'volume']])


    X = df_copy.copy() 
    y_series = pd.Series(index=df_copy.index, dtype=float).fillna(0.0) # Default to 0 (neutral)

    for i in range(len(df_copy) - look_forward_candles):
        entry_price = df_copy['close'].iloc[i]
        current_atr = df_copy['ATR_14'].iloc[i]
        
        if pd.isna(current_atr) or current_atr == 0:
            # y_series.iloc[i] = 0 # Already default, so just continue
            continue

        pt_level = entry_price + (current_atr * pt_atr_multiplier)
        sl_level = entry_price - (current_atr * sl_atr_multiplier)
        outcome = 0 # Default to neutral (time barrier)

        for k in range(1, look_forward_candles + 1):
            if i + k >= len(df_copy): # Boundary check
                break
            future_high = df_copy['high'].iloc[i + k]
            future_low = df_copy['low'].iloc[i + k]
            
            # Order of checks matters if both can be hit in the same future candle
            # For long: check PT first. If hit, outcome is 1.
            # Then check SL. If hit (and PT not hit), outcome is -1.
            if future_high >= pt_level:
                outcome = 1 # Profit Take for a long
                break 
            if future_low <= sl_level:
                outcome = -1 # Stop Loss for a long
                break
        y_series.iloc[i] = outcome
    
    # Align X and y by removing rows in X where y could not be calculated
    X = X.iloc[:-look_forward_candles]
    y_series = y_series.iloc[:-look_forward_candles]

    # Final alignment: ensure X and y_series have the same index after all operations
    # This also handles cases where some rows in X might still have NaNs if initial dropna wasn't enough
    common_index = X.index.intersection(y_series.dropna().index) #dropna on y_series is for safety
    X = X.loc[common_index]
    y_series = y_series.loc[common_index]

    # One last check for NaNs in X, which XGBoost won't like
    if X.isnull().any().any():
        print("Warning: NaNs found in feature set X after labeling. Attempting to fill with 0.")
        print(X.isnull().sum()[X.isnull().sum() > 0])
        X = X.fillna(0) # Or use a more sophisticated imputation strategy

    return X, y_series


if __name__ == '__main__':
    # Create a dummy DataFrame for testing feature generation
    dates = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00',
                            '2023-01-01 13:00', '2023-01-01 14:00', '2023-01-01 15:00',
                            '2023-01-01 16:00', '2023-01-01 17:00', '2023-01-01 18:00',
                            '2023-01-01 19:00', '2023-01-01 20:00', '2023-01-01 21:00',
                            '2023-01-01 22:00', '2023-01-01 23:00', '2023-01-02 00:00'])
    data = {'open':  [10, 11, 12, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 14],
            'high':  [11, 12, 13, 13, 14, 15, 15, 14, 13, 13, 14, 15, 16, 17, 15],
            'low':   [9,  10, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 13],
            'close': [11, 12, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 14, 15],
            'volume':[100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]}
    sample_df = pd.DataFrame(data, index=dates)
    sample_df.index.name = 'timestamp'
    
    print("Original Sample DF:")
    print(sample_df.head())

    df_ta_featured = add_basic_ta_features(sample_df.copy())
    print("\nTA Featured DF:")
    print(df_ta_featured.head().tail())
    
    df_ict_featured = add_ict_features_v2(df_ta_featured.copy(), swing_order=2) # Small order for small dataset
    print("\nICT Featured DF:")
    print(df_ict_featured[['close', 'is_london_kz', 'bullish_bos', 'bearish_fvg_formed', 'bullish_liq_sweep']].tail())

    X_test, y_test = prepare_features_and_labels(df_ict_featured.copy(), look_forward_candles=3, pt_atr_multiplier=1.5, sl_atr_multiplier=1.0)
    print("\nFeatures X (sample):")
    print(X_test.head())
    print("\nLabels y (sample):")
    print(y_test.head())
    print("\nLabel distribution:")
    print(y_test.value_counts())
