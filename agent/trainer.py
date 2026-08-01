import numpy as np


class Trainer:
    def _flatten_state(self, state):
            return np.concatenate([state["cash"],
                                   state["portfolio_value"],
                                   state["prices"],
                                   state["indicators"],
                                   np.array([state["step"]],
                                            dtype=np.float32)])

    def _calculate_sortino(self, returns: list[float], rist_free_rate: float = 0.0) -> float:
        """
        Sortino ratio: mean excess return divided by downside deviation.
        Only negative returns count toward the risk penalty, unlike Sharpe
        which penalizes all volatility including upside swings.
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns, dtype=np.float64)
        excess_returns = returns_array - rist_free_rate

        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if downside_deviation == 0:
            return 0.0
        #if 252 trading days a year
        sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(252)

        return float(sortino)

    def __init__(self, env, agent, episodes: int = 1000):
        self.env = env
        self.agent = agent
        self.episodes = episodes

        observation, _ = self.env.reset()

        state = self._flatten_state(observation)

        self.input_size = state.shape[0]

    def train(self):
        for episode in range(self.episodes):
            observation, _ = self.env.reset()

            state = self._flatten_state(observation)

            done = False
            total_reward = 0
            daily_returns = []
            last_year = None

            prev_value = self.env.portfolio.get_total_value(self.env._get_current_prices())
            
            while not done:
                action = self.agent.choose_action(state)

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                next_state = self._flatten_state(next_observation)

                self.agent.remember(state, action, reward, next_state, terminated or truncated)

                self.agent.learn()

                state = next_state

                total_reward += reward
                done = terminated or truncated

                current_value = self.env.portfolio.get_total_value(self.env._get_current_prices())

                if prev_value > 0:
                    daily_return = (current_value - prev_value) / prev_value
                    daily_returns.append(daily_return)

                prev_value = current_value

                current_date = self.env.market.get_date()
                current_year = current_date.year

                if current_year != last_year:
                    value = self.env.portfolio.get_total_value(self.env._get_current_prices())
                    print(f"    {current_date.date()}  Portfolio: £{value:,.2f}")
                    last_year = current_year

            if episode % 10 == 0:
                self.agent.update_target()

            self.agent.update_epsilon()

            final_value = self.env.portfolio.get_total_value(self.env._get_current_prices())
            sortino = self._calculate_sortino(daily_returns)

            print(
                f"Episode {episode + 1}/{self.episodes} "
                f"Reward: {total_reward:.4f} "
                f"Portfolio: £{final_value:,.2f} "
                f"Sortino: {sortino:.3f} "
                f"Epsilon: {self.agent.epsilon:.3f}"
            )   

                

                