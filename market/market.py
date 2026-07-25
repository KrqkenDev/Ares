import pandas as pd

from indicators.indicators import IndicatorEngine


class Market:
    def __init__(self, historical_data: dict[str, pd.DataFrame]):
        if not isinstance(historical_data, dict):
            raise TypeError(
                "historical_data must be a dictionary of ticker to pandas DataFrame"
            )

        if not historical_data:
            raise ValueError("historical_data cannot be empty")

        required_columns = {"Open", "High", "Low", "Close", "Volume"}

        self.indicator_engine = IndicatorEngine()

        self.historical_data = {}

        for ticker, dataframe in historical_data.items():

            if not isinstance(ticker, str):
                raise TypeError("Ticker must be a string")

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"{ticker} must be a pandas DataFrame")

            missing = required_columns - set(dataframe.columns)

            if missing:
                raise ValueError(f"{ticker} is missing required columns: {missing}")

            dataframe = self.indicator_engine.calculate_all(dataframe)

            dataframe = dataframe.fillna(0)

            self.historical_data[ticker] = dataframe


        self.current_step = 0


    @property
    def tickers(self) -> list[str]:
        return list(self.historical_data.keys())


    @property
    def max_steps(self) -> int:
        """
        Returns the number of timesteps available in market data,
        assumes that all tickers have the same length
        """

        first_ticker = self.tickers[0]

        return min(len(dataframe) for dataframe in self.historical_data.values())


    def _get_dataframe(self, ticker: str | None) -> pd.DataFrame:

        if ticker is None:
            ticker = next(iter(self.historical_data))

        if ticker not in self.historical_data:
            raise KeyError(f"{ticker} does not exist in the market.")

        return self.historical_data[ticker]


    def _validate_step(self, dataframe: pd.DataFrame) -> None:

        if self.current_step >= len(dataframe):
            raise IndexError(
                f"Current step {self.current_step} exceeds available market data."
            )


    def reset(self) -> None:
        """
        Reset the simulation to the first timestep.
        """

        self.current_step = 0


    def next_step(self) -> None:
        """
        Advance the simulation by one timestep.
        """

        if self.is_finished():
            raise IndexError("Market simulation has reached the end of the data.")

        self.current_step += 1


    def get_price(self, ticker: str) -> float:
        """
        Return the current closing price of a ticker.
        """

        dataframe = self._get_dataframe(ticker)

        self._validate_step(dataframe)

        return float(dataframe.iloc[self.current_step]["Close"])

    def get_open(self, ticker: str) -> float:
        """
        Returns the current bar's opening price of a ticker.
        """
        dataframe = self._get_dataframe(ticker)

        self._validate_step(dataframe)

        return float(dataframe.iloc[self.current_step]["Open"])

    def get_date(self, ticker: str | None = None) -> pd.Timestamp:
        """
        Return the current simulation date.
        """

        dataframe = self._get_dataframe(ticker)

        self._validate_step(dataframe)

        return dataframe.index[self.current_step]


    def is_finished(self, ticker: str | None = None) -> bool:
        """
        Return True if the simulation has reached the end.
        """

        dataframe = self._get_dataframe(ticker)

        return self.current_step >= len(dataframe) - 1


    def get_current_row(self, ticker: str) -> pd.Series:
        """
        Return the full OHLCV row and calculated indicators
        for the current timestep.
        """

        dataframe = self._get_dataframe(ticker)

        self._validate_step(dataframe)

        return dataframe.iloc[self.current_step]


    def get_bar(self, ticker: str) -> dict[str, float | pd.Timestamp]:
        """
        Return the complete OHLCV bar for the current timestep.
        """

        row = self.get_current_row(ticker)

        return {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "date": self.get_date(ticker),
        }