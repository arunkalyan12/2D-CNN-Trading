import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    df.dropna(inplace=True)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame) -> pd.Series:
    """Moving Average Convergence Divergence (MACD)"""
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def bollinger_bands(df: pd.DataFrame, window: int = 20):
    """Bollinger Bands"""
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band


def stochastic_oscillator(df: pd.DataFrame, period: int = 14):
    """Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    return 100 * (df['Close'] - low_min) / (high_max - low_min)


def atr(df: pd.DataFrame, period: int = 14):
    """Average True Range (ATR)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    return true_range.rolling(window=period).mean()


def is_doji(df: pd.DataFrame) -> pd.Series:
    high_low_range = df['High'] - df['Low']
    return abs(df['Open'] - df['Close']) / high_low_range < 0.1


def is_hammer(df: pd.DataFrame) -> pd.Series:
    high_low_range = df['High'] - df['Low']
    return ((df['Close'] > df['Open']) &
            ((df['High'] - df['Close']) > 2 * (df['Close'] - df['Open'])))


def is_shooting_star(df: pd.DataFrame) -> pd.Series:
    return ((df['Close'] < df['Open']) &
            ((df['Open'] - df['Close']) > 2 * (df['High'] - df['Open'])))


def is_harami(df: pd.DataFrame) -> pd.Series:
    return ((df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1)))


def is_engulfing(df: pd.DataFrame) -> pd.Series:
    return ((df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)))


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                   np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    return pd.Series(obv).cumsum()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Moving Averages, Volatility, RSI, and returns
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility'] = df['Close'].rolling(window=50).std()
    df['returns'] = df['Close'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    df['RSI'] = rsi(df['Close'])

    # Technical Indicators
    df['MACD'] = macd(df)
    df['ATR'] = atr(df)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = bollinger_bands(df)
    df['Stochastic_Oscillator'] = stochastic_oscillator(df)

    # Candlestick Patterns
    df['Doji_Pattern'] = is_doji(df).astype(int)
    df['Hammer_Pattern'] = is_hammer(df).astype(int)
    df['Engulfing_Pattern'] = is_engulfing(df).astype(int)
    df['Shooting_Star_Pattern'] = is_shooting_star(df).astype(int)
    df['Harami_Pattern'] = is_harami(df).astype(int)

    # OBV (On-Balance Volume)
    df['OBV'] = on_balance_volume(df)

    # Candle metrics
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick_Size'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Lower_Wick_Size'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['Body_to_Wick_Ratio'] = df['Body_Size'] / (df['Upper_Wick_Size'] + df['Lower_Wick_Size'])

    return df


def generate_labels(df: pd.DataFrame, future_period: int = 1) -> pd.Series:
    """
    Generate labels based on future price movement over a given period.
    Label 1 if future close price is greater than the current close price (buy signal),
    otherwise 0 (no signal/sell signal).

    :param df: DataFrame containing OHLCV data
    :param future_period: The number of periods into the future to compare the price against
    :return: Series of labels (1 for buy, 0 for sell)
    """
    future_close = df['Close'].shift(-future_period)
    return (future_close > df['Close']).astype(int)


def scale_data(df: pd.DataFrame, columns: list, feature_range: tuple = (0, 1)) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=feature_range)
    df[columns] = scaler.fit_transform(df[columns])
    return df


def main():
    # Load your data
    df = pd.read_csv(r"C:\Users\Arun2\Documents\Project\Trading Strat\Data\Raw\Rawbtc_ohlcv_jan2023_to_sep2024.csv")

    # Preprocess the data
    df = clean_data(df)
    df = feature_engineering(df)

    # Define columns to be scaled (all features)
    columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'EMA_50', 'Volatility', 'returns',
                        'cumulative_returns', 'RSI', 'MACD', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower',
                        'Stochastic_Oscillator', 'Doji_Pattern', 'Hammer_Pattern', 'Engulfing_Pattern', 'Shooting_Star_Pattern', 'Harami_Pattern',
                        'Body_Size', 'Upper_Wick_Size', 'Lower_Wick_Size', 'Body_to_Wick_Ratio', 'OBV']

    # Check for any infinities or extremely large values before scaling
    print("Data statistics before scaling:")
    print(df[columns_to_scale].describe())

    # Check for infinities
    print("Any infinities in the data?")
    print(np.isinf(df[columns_to_scale]).any())

    # Replace remaining infinities with NaNs
    df[columns_to_scale] = df[columns_to_scale].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaNs
    df.dropna(subset=columns_to_scale, inplace=True)

    # Scale the data
    df = scale_data(df, columns=columns_to_scale)

    # Generate labels
    df['Label'] = generate_labels(df, future_period=1)  # Assuming 1-period future for prediction

    # Remove any rows with NaN generated by shifting
    df.dropna(inplace=True)

    # Save preprocessed data to CSV
    df.to_csv(r"C:\Users\Arun2\Documents\Project\Trading Strat\Data\Preprocessed\Preprocessed.csv", index=False)


if __name__ == "__main__":
    main()

