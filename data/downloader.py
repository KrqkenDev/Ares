from datetime import datetime

import pandas as pd
import yfinance as yf


class StockDownloader:
    """
    Stock data downloader.

    Downloads historical OHLCV data from Yahoo Finance,
    cleans it,
    splits it into individual ticker DataFrames,
    ready for caching as Parquet files.

    Downloads historical stock data for multiple tickers.

    Output format: (e.g.)

    {
        "AAPL": DataFrame,
        "MSFT": DataFrame,
        "NVDA": DataFrame
    }

    Each DataFrame contains:

        - Date index
        - Open
        - High
        - Low
        - Close
        - Volume  (OHLCV)
    """

    def __init__(self, tickers: list[str], start_date: datetime, end_date: datetime, interval: str,):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        self.combined_data: pd.DataFrame | None = None

    def download(self) -> pd.DataFrame:
        """
        Downloads all ticker data from Yahoo Finance.

        Returns:
            Combined dataframe with multi-level columns.
        """
        print(f"Downloading data for: {', '.join(self.tickers)}")

        self.combined_data = yf.download(
        
            tickers=self.tickers,
            # Date range
            start=self.start_date,
            end=self.end_date,
            # Daily candles
            interval=self.interval,
            # Keep each ticker separated
            group_by="ticker",
            # Adjust for splits/dividends
            auto_adjust=True, progress=False)

        if self.combined_data is None or self.combined_data.empty:
            raise ValueError("Yahoo Finance returned no data.")
            
        # Remove incomplete rows
        # Prevents the RL seeing missing prices
        self.combined_data = self.combined_data.dropna()
        print("Download complete.")

        print(f"Rows downloaded: {len(self.combined_data)}")
        return self.combined_data

    def split_by_ticker(self) -> dict[str, pd.DataFrame]:
        """
        Converts the combined dataframe into:
        {ticker: dataframe}
        One dataframe per company.
        """
        if self.combined_data is None:
            raise RuntimeError("Call download() before split_by_ticker().")
        
        ticker_data = {}

        for ticker in self.tickers:
            # Check ticker exists before loading
            if ticker not in self.combined_data:
                print(f"Warning: {ticker} missing")

                continue

            data = (self.combined_data[ticker].copy().dropna())

            data = data.sort_index()

            ticker_data[ticker] = data

            print(f"{ticker}: {len(data)} rows")

        return ticker_data