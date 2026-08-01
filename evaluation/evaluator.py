from agent.agent import Agent
from environment.environment import TradingEnvironment

def evaluate(agent: Agent, env: TradingEnvironment) -> None:
    """
    Evaluates the trained agent on unseen market data.
    """

    observation, _ = env.reset()

    state = env.flatten_observation(observation)

    done = False
    total_reward = 0.0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.choose_action(state)

        print(action)

        observation, reward, terminated, truncated, _ = env.step(action)

        state = env.flatten_observation(observation)

        total_reward += reward

        done = terminated or truncated

    prices = env._get_current_prices()
    final_value = env.portfolio.get_total_value(prices)

    print("#########   Test results:  #########")
    print(f"Reward: {total_reward:.4f}")
    print(f"Portfolio £{final_value:,.2f}")
    print("####################################")

    agent.epsilon = original_epsilon