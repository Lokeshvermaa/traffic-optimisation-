import numpy as np
from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-learning agent for traffic light control
        actions: list of possible actions (e.g., ["green", "yellow", "red"])
        """
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            state_q = self.q_table[state]
            return self.actions[np.argmax(state_q)]

    def learn(self, state, action, reward, next_state):
        """Update Q-values based on agent’s experience"""
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])

        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
        self.q_table[state][action_idx] = new_q
