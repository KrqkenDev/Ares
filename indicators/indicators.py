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
        
        return data

    def calculate_ema_difference(self, data: pd.DataFrame, period1: int, period2: int) -> pd.DataFrame:
        """
        Calculates the difference between two EMAs of different periods.
        This can be used to identify trend changes and potential buy/sell signals for the RL agent.
        """
        if period1 <= 0 or period2 <= 0:
            raise ValueError("Periods must be positive integers.")

        data[f"EMA_DIFF_{period1}_{period2}"] = (data[f"EMA_{period1}"] - 
                                                 data[f"EMA_{period2}"])
        
        return data
    
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all indicators (EMA20, EMA50, RSI14, EMA_DIFF20_50) for the given data.
        """
        self._check_columns(data)
        features = data.copy() #cloned data as to not modify the original data, stored in raw.
        # note: this will grow with later improvement for calculating more indicators like MACD, Bollinger Bands, etc... but for now we will keep it simple.
        features = self.calculate_ema(features, 20)
        features = self.calculate_ema(features, 50)
        features = self.calculate_rsi(features, 14)
        features = self.calculate_ema_difference(features, 20, 50)
        
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
