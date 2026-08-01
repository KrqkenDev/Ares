import pandas as pd
import numpy as np

class IndicatorEngine:
    """
    Calculates indicators from historical OHLCV data sets.
    """
    def __init__(self):
        pass

    def calculate_ema(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculates the Exponential Moving Average (EMA) for a given period.
        Exponentially Weighted Moving Average gives more weight to recent prices and info.
        Uses recursive formula: EMA_today = (Price_today * (2 / (period + 1))) + (EMA_yesterday * (1 - (2 / (period + 1))))
        Used for both ema20 and ema50 / any other period. 
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")

        data[f"EMA_{period}"] = (data['Close'].ewm(span=period, 
                                                   adjust=False)
                                                   .mean())
        return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI) for a given period.
        RSI is a momentum oscillator that measures the speed and change of price movements.
        RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
        Note to self: This is a bit simple...
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, 
                            adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, 
                            adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float('nan'))  # Avoid division by zero (usually dealt with by pandas but just in case :D)
        data[f"RSI_{period}"] = 100 - (100 / (1 + rs))
        data[f"RSI_{period}_norm"] = data[f"RSI_{period}"] / 100.00

        return data

    def calculate_ema_difference(self, data: pd.DataFrame, period1: int, period2: int) -> pd.DataFrame:
        """
        Calculates the difference between two EMAs of different periods.
        This can be used to identify trend changes and potential buy/sell signals for the RL agent.
        """
        if period1 <= 0 or period2 <= 0:
            raise ValueError("Periods must be positive integers.")

        diff = data[f"EMA_{period1}"] - data[f"EMA_{period2}"]
        data[f"EMA_DIFF_{period1}_{period2}"] = diff

        data[f"EMA_DIFF_{period1}_{period2}_norm"] = diff / data['Close']
        
        return data

    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal = 9)-> pd.DataFrame:
        """
        MACD line = EMA_fast - EMA_slow
        Signal line = EMA of the MACD line
        Histogram = MACD line - Signal line
        Price-scale by default, so normalised (divided by Close) versions are
        also stored.
        """

        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow

        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        histogram = macd_line - signal_line

        data["MACD"] = macd_line
        data["MACD_signal"] = signal_line
        data["MACD_hist"] = histogram

        data["MACD_norm"] = macd_line / data['Close']
        data["MACD_signal_norm"] = signal_line / data['Close']
        data["MACD_hist_norm"] = histogram / data['Close']

        return data

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands: rolling mean +/- a multiple of rolling std dev.
        %B expresses where price sits within the bands (0 = lower band,
        1 = upper band) — a naturally 0-1 scaled feature.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")

        rolling_mean = data['Close'].rolling(window=period).mean()
        rolling_std = data['Close'].rolling(window=period).std()

        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)

        data["BB_middle"] = rolling_mean
        data["BB_upper"] = upper
        data["BB_lower"] = lower
        band_width = (upper - lower).replace(0, float('nan'))

        data["BB_percent_b"] = (data['Close'] - lower) / band_width

        return data

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range: a volatility measure. Uses Close.shift(1), i.e.
        the previous row's close — a causal lookback, not a future leak.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")

        prev_close = data['Close'].shift(1)

        true_range = pd.concat([data['High'] - data['Low'],
                                (data['High'] - prev_close).abs(),
                                (data['Low'] - prev_close).abs()
                                ], axis=1).max(axis=1)

        atr = true_range.ewm(alpha=1/period, adjust = False).mean()

        data[f"ATR_{period}"] = atr
        data[f"ATR_{period}_pct"] = atr / data['Close']

        return data


    def calculate_stochastic(self, data: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator: %K is where price sits within the recent
        high/low range, %D smooths %K. Both naturally 0-100 scale.
        """

        if period <= 0 or smooth <= 0:
            raise ValueError("Pediod and smooth must be positive integers")

        rolling_low = data['Low'].rolling(window=period).min()
        rolling_high = data['High'].rolling(window=period).max()

        band_range = (rolling_high - rolling_low).replace(0, float('nan'))

        percent_k = 100 * (data['Close'] - rolling_low) / band_range
        percent_d = percent_k.rolling(window=smooth).mean()

        data["STOCH_K"] = percent_k
        data["STOCH_D"] = percent_d

        data["STOCH_K_norm"] = percent_k / 100.00
        data["STOCH_D_norm"] = percent_d / 100.00

        return data

    def calculate_roc(self, data: pd.DataFrame, period: int = 10) ->pd.DataFrame:
        """
        Rate of Change: percentage price change over the period.
        Already a small number (e.g. 0.05 = +5%), so no extra scaling needed.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")

        data[f"ROC_{period}"] = data['Close'].pct_change(periods=period)

        return data

    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        On-Balance Volume: cumulative volume, added on up days, subtracted on
        down days. Raw OBV drifts unbounded over time, so a rolling z-score
        is also stored for use in observations.
        """
        price_diff = data['Close'].diff().fillna(0)
        direction: pd.Series = pd.Series(np.sign(price_diff), index=data.index)

        obv: pd.Series = (direction * data['Volume']).cumsum()

        data["OBV"] = obv

        rolling_mean = obv.rolling(window=50, min_periods=1).mean()
        rolling_std = obv.rolling(window=50, min_periods=1).std().replace(0, float('nan'))

        data["OBV_zscore"] = (obv - rolling_mean) / rolling_std

        return data
        
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all indicators (EMA20, EMA50, RSI14, EMA_DIFF20_50) for the given data.
        """

        self._check_columns(data)
        features = data.copy() 

        features = self.calculate_ema(features, 20)
        features = self.calculate_ema(features, 50)
        features = self.calculate_rsi(features, 14)
        features = self.calculate_ema_difference(features, 20, 50)
        features = self.calculate_macd(features)
        features = self.calculate_bollinger_bands(features)
        features = self.calculate_atr(features)
        features = self.calculate_stochastic(features)
        features = self.calculate_roc(features)
        features = self.calculate_obv(features)
        
        return features
    
    def _check_columns(self, data: pd.DataFrame) -> None:
        """
        Checks if required OHLCV columns are present in the DataFrame.
        """
        
        required = ["Open", "High", "Low", "Close", "Volume"]

        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")
