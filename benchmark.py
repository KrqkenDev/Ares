from datetime import datetime
from pathlib import Path

from assets.registry import ASSETS

from data.cache_manager import CacheManager

from market.market import Market
from portfolio.portfolio import Portfolio
from broker.broker import Broker
from environment.environment import TradingEnvironment

CACHE_DIR = "data/raw"
STARTING_CASH = 100000


def load_market_data():
    tickers = list(ASSETS.keys())
    cache = CacheManager(CACHE_DIR)

    historical_data = {}
    for ticker in tickers:
        historical_data[ticker] = cache.load(ticker)

    return historical_data


def buy_and_hold_benchmark(env):
    env.reset()
    prices_start = env._get_current_prices()
    shares = {t: (env.portfolio.starting_cash / len(env.tickers)) / prices_start[t] for t in env.tickers}

    last_year = None

    while not env.market.is_finished():
        env.market.next_step()

        current_date = env.market.get_date()
        current_year = current_date.year

        if current_year != last_year:
            prices_now = env._get_current_prices()
            value = sum(shares[t] * prices_now[t] for t in env.tickers)
            print(f"    {current_date.date()}  Buy&Hold: £{value:,.2f}")
            last_year = current_year

    prices_end = env._get_current_prices()
    final_value = sum(shares[t] * prices_end[t] for t in env.tickers)
    print(f"Buy & hold final value: £{final_value:,.2f}")


if __name__ == "__main__":
    historical_data = load_market_data()

    market = Market(historical_data)
    portfolio = Portfolio(starting_cash=STARTING_CASH)
    broker = Broker(portfolio, commission_rate=0.001)
    env = TradingEnvironment(market, portfolio, broker)

    buy_and_hold_benchmark(env)