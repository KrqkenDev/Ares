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

    INDICATOR_COLUMNS = [
    "RSI_14_norm",
    "EMA_DIFF_20_50_norm",
    "MACD_norm",
    "MACD_signal_norm",
    "MACD_hist_norm",
    "BB_percent_b",
    "ATR_14_pct",
    "STOCH_K_norm",
    "STOCH_D_norm",
    "ROC_10",
    "OBV_zscore",
    ]

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
    ),
    "indicators": gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(self.tickers)* len(self.INDICATOR_COLUMNS),),
        dtype=np.float32)
})

    def _get_current_prices(self) -> dict[str, float]:
        prices = {}

        for ticker in self.tickers:
            prices[ticker] = self.market.get_price(ticker)

        return prices

    def _get_current_indicators(self) -> np.ndarray:
        values = []

        for ticker in self.tickers:
            values.extend(self.market.get_indicator_values(ticker, self.INDICATOR_COLUMNS))

        return np.array(values, dtype=np.float32)

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
        "indicators": self._get_current_indicators(),
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

    def _execute_action(self, action: int) -> None:

        """
        Executes an action chosen by the AI
        """

        asset, agent_action, allocation = self._decode_action(action)

        print(
            f"{self.market.get_date().date()} | "
            f"{asset.ticker} | "
            f"{agent_action.name} | "
            f"{allocation}"
        )

        if agent_action == AgentAction.HOLD:
            return

        price = self.market.get_open(asset.ticker)

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
            print(
                f"Executed {order.action.name} "
                f"{order.shares} shares of {asset.ticker} "
                f"@ {price:.2f}"
            )
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

    def flatten_observation(self, observation: dict) -> np.ndarray:
        return np.concatenate([
            observation["cash"],
            observation["portfolio_value"],
            observation["prices"],
            observation["indicators"],
            np.array([observation["step"]],
                     dtype=np.float32),
        ])

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
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
        