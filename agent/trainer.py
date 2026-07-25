import numpy as np


class Trainer:
    def _flatten_state(self, state):
            return np.concatenate([state["cash"],
                                   state["portfolio_value"],
                                   state["prices"],
                                   np.array([state["step"]],
                                            dtype=np.float32)])

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

            state =self._flatten_state(observation)

            done = False
            total_reward = 0

            while not done:
                action = self.agent.choose_action(state)

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                next_state = self._flatten_state(next_observation)

                self.agent.remember(state, action, reward, next_state, terminated or truncated)

                self.agent.learn()

                state = next_state

                total_reward += reward
                done = terminated or truncated

            if episode % 10 == 0:
                self.agent.update_target()

            self.agent.update_epsilon()

            final_value = self.env.portfolio.get_total_value(self.env._get_current_prices())

            print(
                f"Episode {episode + 1}/{self.episodes} "
                f"Reward: {total_reward:.4f} "
                f"Portfolio: £{final_value:,.2f} "
                f"Epsilon: {self.agent.epsilon:.3f}"
            )   

                

                