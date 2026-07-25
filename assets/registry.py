from assets.asset import Asset
from common.enums import Exchange


ASSETS = {
    "AAPL": Asset(
        ticker="AAPL",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "GOOGL": Asset(
        ticker="GOOGL",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "MSFT": Asset(
        ticker="MSFT",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "AMZN": Asset(
        ticker="AMZN",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "TSLA": Asset(
        ticker="TSLA",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "META": Asset(
        ticker="META",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "NFLX": Asset(
        ticker="NFLX",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "NVDA": Asset(
        ticker="NVDA",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "JPM": Asset(
        ticker="JPM",
        exchange=Exchange.NYSE,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    ),

    "TSM": Asset(
        ticker="TSM",
        exchange=Exchange.NYSE,
        currency="USD",
        country="Taiwan",
        asset_type="stock",
        tax_region="USA"
    ),

    "AVGO": Asset(
        ticker="AVGO",
        exchange=Exchange.NASDAQ,
        currency="USD",
        country="USA",
        asset_type="stock",
        tax_region="USA"
    )
}