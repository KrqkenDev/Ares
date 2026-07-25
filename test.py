import pandas as pd
import numpy as np

from market.market import Market
from portfolio.portfolio import Portfolio
from broker.broker import Broker
from environment.environment import TradingEnvironment


def create_test_data():

    dates = pd.date_range(
        start="2024-01-01",
        periods=100
    )

    historical_data = {}

    for ticker in ["AAPL", "MSFT", "NVDA"]:

        historical_data[ticker] = pd.DataFrame(
            {
                "Open": np.random.uniform(90, 100, 100),
                "High": np.random.uniform(100, 110, 100),
                "Low": np.random.uniform(80, 90, 100),
                "Close": np.random.uniform(90, 100, 100),
                "Volume": np.random.randint(
                    100000,
                    1000000,
                    100
                )
            },
            index=dates
        )

    return historical_data


def main():

    print("Creating market...")

    market = Market(
        create_test_data()
    )


    print("Creating portfolio...")

    portfolio = Portfolio(
        starting_cash=100000
    )


    print("Creating broker...")

    broker = Broker(
        portfolio
    )


    print("Creating environment...")

    env = TradingEnvironment(
        market,
        portfolio,
        broker
    )


    print("Resetting environment...")

    observation, info = env.reset()


    done = False
    steps = 0
 

    while not done:

        # Random AI action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        done = terminated or truncated

        steps += 1


    env.close()


    print("-------------------------")
    print("TEST PASSED")
    print(f"Steps completed: {steps}")
    print("-------------------------")


if __name__ == "__main__":
    main()