from datetime import datetime
from pathlib import Path

from assets.registry import ASSETS

from data.cache_manager import CacheManager
from data.downloader import StockDownloader

from market.market import Market
from portfolio.portfolio import Portfolio
from broker.broker import Broker
from environment.environment import TradingEnvironment

from agent.agent import Agent
from agent.trainer import Trainer

#Config
TICKERS = list(ASSETS.keys())

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)

INTERVAL = "1d"  

CACHE_DIR = "data/raw"

STARTING_CASH = 100000
EPISODES = 10

def ensure_market_data():
    tickers = list(ASSETS.keys())

    cache = CacheManager(CACHE_DIR)

    missing = []

    for ticker in tickers:
        if not cache.has(ticker):
            missing.append(ticker)

    if missing:
        print(f"Missing data: {missing}.")
        print("Downloading missing market data...")

        downloader = StockDownloader(tickers=tickers, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL)

        downloader.download()

        ticker_data = (downloader.split_by_ticker())

        cache.save_all(ticker_data)

    else:
        print("All market data found.")

    historical_data = {}

    for ticker in tickers:
        historical_data[ticker] = (cache.load(ticker))

    return historical_data


def create_environment():
    historical_data = ensure_market_data()

    print("Creating market...")

    market = Market(historical_data)

    print("Creating portfolio...")
    
    portfolio = Portfolio(starting_cash=STARTING_CASH)

    print("Creating broker...")

    broker = Broker(portfolio)

    print("Creating environment...")

    env = TradingEnvironment(market, portfolio, broker)

    return env

def main():
    env = create_environment()

    print("Creating agent...")

    input_size = (1+1+len(env.tickers)+1)

    agent = Agent(input_size = input_size,  action_size=env.num_actions, tickers=len(env.tickers))

    trainer = Trainer(env=env, agent=agent, episodes=EPISODES)

    trainer.train()

    Path("models").mkdir(exist_ok=True)

    agent.save("models/training_model.pth")

    env.close()

if __name__ == "__main__":
    main()

