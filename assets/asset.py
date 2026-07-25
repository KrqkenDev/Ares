from dataclasses import dataclass
from common.enums import Exchange


@dataclass(frozen=True)
class Asset:
    """
    Represents a tradeable financial asset in the portfolio.

    Stores metadata that doesn't change over time, such as the ticker symbol and the exchange it is traded on.
    """

    ticker: str
    exchange: Exchange
    currency: str
    country: str
    asset_type: str = "stock"
    tax_region: str | None = None
