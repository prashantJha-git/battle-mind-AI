import numpy as np

import config
from neuralNetwork import NeuralNetwork


class RLAgent:
    """POLICY-GRADIENT AGENT (REINFORCE WITH A VALUE BASELINE)"""

    def __init__(self, state_size=8, action_size=5, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_network = NeuralNetwork(state_size, [64, 64], action_size, learning_rate)
        self.value_network = NeuralNetwork(state_size, [64], 1, learning_rate * 0.1)

        self.buffer = {"states": [], "actions": [], "rewards": [], "dones": []}
        self.episode_reward = 0.0
        self.exploration = config.EXPLORATION_START

    def get_action(self, state):
        state = np.array(state, dtype=np.float32)

        if np.random.rand() < self.exploration:
            action = np.random.randint(self.action_size)
        else:
            action, _ = self.policy_network.get_action(state)

        self.exploration = max(config.EXPLORATION_MIN, self.exploration * config.EXPLORATION_DECAY)
        return action

    def store_experience(self, state, action, reward, done):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.episode_reward += reward

    def buffer_size(self):
        return len(self.buffer["states"])

    def _compute_returns(self, gamma=0.99):
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        returns = np.zeros_like(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            if self.buffer["dones"][i]:
                running = 0.0
            running = rewards[i] + gamma * running
            returns[i] = running
        return returns

    def train(self):
        if self.buffer_size() == 0:
            return

        states = np.array(self.buffer["states"], dtype=np.float32)
        actions = np.array(self.buffer["actions"], dtype=np.int32)
        returns = self._compute_returns()

        values = self.value_network.forward(states).flatten()
        advantages = returns - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_onehot = np.eye(self.action_size)[actions]
        self.policy_network.forward(states)
        self.policy_network.backward(states, action_onehot, advantages)

        self.value_network.backward_mse(states, returns.reshape(-1, 1))

        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {"states": [], "actions": [], "rewards": [], "dones": []}

    def reset_episode_reward(self):
        self.episode_reward = 0.0

    def save_weights(self, filepath):
        np.save(filepath, {
            "policy_layers": self.policy_network.layers,
            "value_layers": self.value_network.layers,
        }, allow_pickle=True)

    def load_weights(self, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        self.policy_network.layers = data["policy_layers"]
        self.value_network.layers = data["value_layers"]
