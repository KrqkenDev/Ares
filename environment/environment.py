import numpy as np
import gymnasium as gym

from loguru import logger

from market.market import Market
from portfolio.portfolio import Portfolio
from broker.broker import Broker, Order

from common.enums import OrderAction, AgentAction
from common.allocations import Allocation
from assets.registry import ASSETS


class TradingEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, market: Market, portfolio: Portfolio, broker: Broker) -> None:
        self.market = market
        self.portfolio = portfolio
        self.broker = broker

        self.prev_portfolio_value = 0.0

        self.tickers = self.market.tickers

        self.assets = [ASSETS[ticker] for ticker in self.tickers]
        
        self.num_actions = (len(self.tickers) * AgentAction.count() * Allocation.count())

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Dict({
    "cash": gym.spaces.Box(
        low=0,
        high=np.inf,
        shape=(1,),
        dtype=np.float32
    ),

    "portfolio_value": gym.spaces.Box(
        low=0,
        high=np.inf,
        shape=(1,),
        dtype=np.float32
    ),

    "prices": gym.spaces.Box(
        low=0,
        high=np.inf,
        shape=(len(self.tickers),),
        dtype=np.float32
    ),

    "step": gym.spaces.Discrete(
        self.market.max_steps
    )
})

    def _get_current_prices(self) -> dict[str, float]:
        prices = {}

        for ticker in self.tickers:
            prices[ticker] = self.market.get_price(ticker)

        return prices

    def _get_observation(self) -> dict:
        prices = self._get_current_prices()

        return {"cash": np.array(
            [self.portfolio.get_cash()],
            dtype=np.float32
        ),
        "portfolio_value": np.array(
            [self.portfolio.get_total_value(prices)],
            dtype=np.float32
        ),
        "prices": np.array(
            list(prices.values()),
            dtype=np.float32
        ),
        "step": self.market.current_step} 

    def _calculate_reward(self) -> float:
        """
        Calculates reward based on portfolio value change.
        """

        current_value = self.portfolio.get_total_value(self._get_current_prices())

        if self.prev_portfolio_value == 0:
            self.prev_portfolio_value = current_value
            return 0.0

        reward = np.log(current_value / self.prev_portfolio_value)

        self.prev_portfolio_value = current_value

        return float(reward)

    def _is_done(self) -> bool:
        return self.market.is_finished()

    def _decode_action(self, action: int)-> tuple:

        allocation_count = Allocation.count()
        action_count = AgentAction.count()

        allocation_index = action % allocation_count

        action //= allocation_count

        action_index = action % action_count

        ticker_index = action // action_count

        asset = self.assets[ticker_index]

        agent_action = AgentAction(action_index)

        allocation = Allocation.get(allocation_index)

        return asset, agent_action, allocation

    def _execute_action(self, action) -> None:

        """
        Executes an action chosen by the AI
        """

        asset, agent_action, allocation = self._decode_action(action)

        if agent_action == AgentAction.HOLD:
            return

        price = self.market.get_price(asset.ticker)

        if agent_action == AgentAction.BUY:
            cash = self.portfolio.get_cash()
            money_to_spend = cash * allocation

            shares = int(money_to_spend/price)

            if shares <= 0:
                return

            order = Order(asset=asset, action = OrderAction.BUY, shares=shares, price=price)

        else:

            position = self.portfolio.get_position(asset.ticker)

            if position is None:
                return

            shares_owned = position["shares"]

            shares = int(shares_owned * allocation)

            if shares <= 0:
                return

            order = Order(asset=asset, action= OrderAction.SELL, shares=shares, price=price)

        try:
            self.broker.execute_order(order)
        except ValueError as e:
            logger.warning(f"Order execution failed: {e}")

    def reset(self, seed=None, options=None) -> tuple:

        super().reset(seed=seed)
        self.market.reset()
        self.portfolio.reset()

        self.prev_portfolio_value = self.portfolio.get_total_value(self._get_current_prices())

        return self._get_observation(), {}

    def render(self):
        prices = self._get_current_prices()

        print(f"Step: {self.market.current_step}")
        print(f"Cash: {self.portfolio.get_cash():.2f}")
        print(f"Portfolio: {self.portfolio.get_total_value(prices):.2f}")

    def flatten_observation(self, observetion: dict) -> np.ndarray:
        return np.concatenate([
            observetion["cash"],
            observetion["portfolio_value"],
            observetion["prices"],
            observetion["step"],
        ])

    def step(self, action: dict) -> tuple:
        self._execute_action(action)

        self.market.next_step()

        terminated = self._is_done()

        observation = self._get_observation()

        reward = self._calculate_reward()

        truncated = False

        info = {}

        return (observation, reward, terminated, truncated, info)

    def close(self) -> None:
        pass
        