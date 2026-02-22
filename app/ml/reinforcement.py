from __future__ import annotations

import numpy as np


class QLearningPriceAdjuster:
    """Lightweight Q-learning to adjust final price ratio within business limits."""

    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = ['low', 'medium', 'high']
        self.actions = np.array([-0.05, 0.0, 0.05])
        self.q_table = np.zeros((len(self.states), len(self.actions)))

    def _state_idx(self, demand_index: float, inventory_level: float) -> int:
        if demand_index > 70 and inventory_level < 25:
            return 2
        if demand_index < 35 and inventory_level > 60:
            return 0
        return 1

    def _reward(self, price: float, demand_index: float, conversion_rate: float) -> float:
        revenue = price * max(conversion_rate, 0.01)
        return (0.7 * revenue) + (0.3 * conversion_rate * demand_index)

    def train_episode(self, base_price: float, demand_index: float, inventory_level: float) -> None:
        s = self._state_idx(demand_index, inventory_level)
        if np.random.rand() < self.epsilon:
            a = np.random.randint(len(self.actions))
        else:
            a = int(np.argmax(self.q_table[s]))

        delta = self.actions[a]
        simulated_price = base_price * (1 + delta)
        simulated_conversion = np.clip((demand_index / 100) - (delta * 0.4), 0.01, 1)
        reward = self._reward(simulated_price, demand_index, simulated_conversion)

        next_state = self._state_idx(demand_index * np.random.uniform(0.95, 1.05), inventory_level)
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def suggest_adjustment(self, demand_index: float, inventory_level: float) -> float:
        state = self._state_idx(demand_index, inventory_level)
        action_idx = int(np.argmax(self.q_table[state]))
        return float(self.actions[action_idx])
