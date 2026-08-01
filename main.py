from datetime import datetime, timedelta
from pathlib import Path

from assets.registry import ASSETS

from data.cache_manager import CacheManager
from data.downloader import StockDownloader
from data.splitter import split_data

from market.market import Market
from portfolio.portfolio import Portfolio
from broker.broker import Broker
from environment.environment import TradingEnvironment

from agent.agent import Agent
from agent.trainer import Trainer

from evaluation.evaluator import evaluate


TICKERS = list(ASSETS.keys())

FULL_START_DATE = datetime(2015, 1, 1)
FULL_END_DATE = datetime(2025, 12, 31)

_total_days = (FULL_END_DATE - FULL_START_DATE).days
_split_days = int(_total_days * 0.7)

TRAIN_START = FULL_START_DATE
TRAIN_END = FULL_START_DATE + timedelta(days=_split_days)

TEST_START = TRAIN_END + timedelta(days=1)
TEST_END = FULL_END_DATE

INTERVAL = "1d"  

CACHE_DIR = "data/raw"

TRAIN_RATIO = 0.7

STARTING_CASH = 100000
EPISODES = 10

MODEL_PATH = "models/training_modelv1.0.pth"

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

        downloader = StockDownloader(tickers=tickers, start_date=FULL_START_DATE, end_date=FULL_END_DATE, interval=INTERVAL)

        downloader.download()

        ticker_data = (downloader.split_by_ticker())

        cache.save_all(ticker_data)

    else:
        print("All market data found.")

    historical_data = {}

    for ticker in TICKERS:
        historical_data[ticker] = cache.load(ticker)

    return historical_data


def create_environment(training=True):

    historical_data = ensure_market_data()

    historical_data = split_data(historical_data, training_ratio=TRAIN_RATIO, training=training)

    print("Creating market...")

    market = Market(historical_data)

    print("Creating portfolio...")
    
    portfolio = Portfolio(starting_cash=STARTING_CASH)

    print("Creating broker...")

    broker = Broker(portfolio, commission_rate=0.001)

    print("Creating environment...")

    env = TradingEnvironment(market, portfolio, broker)

    return env

def create_agent(env):
    input_size = (1+1
                      +len(env.tickers)
                      + (len(env.tickers) * len(env.INDICATOR_COLUMNS))
                      +1)
    
    agent = Agent(input_size = input_size,  action_size=env.num_actions, tickers=len(env.tickers))

    return agent

def main():
    train_env = create_environment(training=True)

    print("Creating agent...")

    agent = create_agent(train_env)

    trainer = Trainer(env=train_env, agent=agent, episodes=EPISODES)

    trainer.train()

    Path("models").mkdir(exist_ok=True)

    agent.save(MODEL_PATH)

    print("Model saved")

    train_env.close()

    print("!!! Testing.... !!!")

    print("Creating test environment...")
    
    test_env = create_environment(training=False)

    test_agent = create_agent(test_env)

    test_agent.load(MODEL_PATH)

    evaluate(test_agent, test_env)

    test_env.close()

    
if __name__ == "__main__":
    main()

