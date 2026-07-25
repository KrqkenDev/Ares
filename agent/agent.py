import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from agent.model import DQN
from agent.memory import ReplayMemory

from common.enums import AgentAction
from common.allocations import Allocation

class Agent:
    def __init__(self, input_size: int, action_size: int, tickers: int):
        self.action_size = action_size
        self.tickers = tickers

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                            else "cpu")

        self.model = DQN(input_size, action_size).to(self.device)

        self.target_model = DQN(input_size, action_size).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.memory = ReplayMemory(capacity=100000)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_function = nn.MSELoss()
        #Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        #Future Rewards
        self.gamma = 0.99

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def decode_action(self, action: int):

        allocations = Allocation.values()

        allocation_count = len(allocations)

        actions_count = AgentAction.count()

        allocation_index = action % allocation_count

        action //= allocation_count

        agent_action_index = action % actions_count

        ticker_index = action // actions_count

        return {"ticker": ticker_index, "action": agent_action_index, "allocation": np.array([allocations[allocation_index]], dtype=np.float32)}

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def learn(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return

        experiences = self.memory.sample(batch_size)

        states = torch.tensor(
            np.array([e[0] for e in experiences]),
            dtype=torch.float32
        ).to(self.device)


        actions = torch.tensor(
            [e[1] for e in experiences],
            dtype=torch.long
        ).unsqueeze(1).to(self.device)


        rewards = torch.tensor(
            [e[2] for e in experiences],
            dtype=torch.float32
        ).to(self.device)


        next_states = torch.tensor(
            np.array([e[3] for e in experiences]),
            dtype=torch.float32
        ).to(self.device)


        dones = torch.tensor(
            [e[4] for e in experiences],
            dtype=torch.float32
        ).to(self.device)

        current_q = (self.model(states).gather(1, actions)).squeeze()


        with torch.no_grad():
            next_q = (self.target_model(next_states)).max(1)[0]

            target_q = rewards + (self.gamma*next_q *(1- dones))

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        