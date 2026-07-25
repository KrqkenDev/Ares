from pathlib import Path
import pandas as pd


class CacheManager:
    """
    Caches one Parquet file per ticker inside a cache directory.
    This handles saving, loading and checking cached stock data.
    Each stock has its own paraquet file.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def save(self, ticker: str, data: pd.DataFrame) -> None:

        """Saves a single ticker's DataFrame to cache/<ticker>.parquet."""

        data.to_parquet(self._path_for(ticker), index=True)

    def save_all(self, ticker_data: dict[str, pd.DataFrame]) -> None:

        """Saves a dict of ticker -> DataFrame, one file per ticker."""

        for ticker, data in ticker_data.items():
            self.save(ticker, data)
            print(f"Cached {ticker} -> {self._path_for(ticker)}")

    def load(self, ticker: str) -> pd.DataFrame:

        """Loads a single ticker's DataFrame from cache/<ticker>.parquet."""

        return pd.read_parquet(self._path_for(ticker))

    def has(self, ticker: str) -> bool:

        """Checks whether a cached file exists for this ticker."""
        
        return self._path_for(ticker).exists()

    def save_by_year(self, ticker, dataframe):
        dataframe = dataframe.copy()

        dataframe["Year"] = dataframe.index.year

        for year, data in dataframe.groupby("Year"):
            path = (self.cache_dir/ticker/f"{year}.parquet")

            path.parent.mkdir(parents=True,exist_ok=True)

            data.drop(columns=["Year"]).to_parquet(path)

    def load_year(self, ticker: str, year: int) -> pd.DataFrame:

        path = self.cache_dir/ticker/f"{year}.parquet"

        return pd.read_parquet(path)
