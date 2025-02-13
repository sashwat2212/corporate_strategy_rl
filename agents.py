import numpy as np
import random

# Define the QLearningAgent class
class QLearningAgent:
    def __init__(self, capital_bins=20, market_share_bins=20, action_size=4, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, min_exploration=0.01, decay_rate=0.995):
        self.capital_bins = capital_bins
        self.market_share_bins = market_share_bins
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration = min_exploration
        self.decay_rate = decay_rate

        # Initialize total spent and initial market share
        self.total_spent = 0
        self.initial_market_share = None
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.capital_bins, self.market_share_bins, self.action_size))
    
    # Discretize the state space
    def discretize_state(self, state):
        capital, market_share = state
        
        # Ensure capital and market share are within bounds, clamp them to valid ranges
        capital_bin = max(0, min(self.capital_bins - 1, int(capital // (1000 / self.capital_bins))))  # Assuming max capital is 1000
        market_share_bin = max(0, min(self.market_share_bins - 1, int(market_share // (100 / self.market_share_bins))))  # Max market share is 100
        
        return capital_bin, market_share_bin
    
    # Choose an action based on the current state
    def choose_action(self, state):
        capital_bin, market_share_bin = self.discretize_state(state)
        
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_size - 1)  # Explore (random action)
        else:
            return np.argmax(self.q_table[capital_bin, market_share_bin])  # Exploit (best known action)
    
    # Update the Q-table based on the state, action, reward, and next state
    def update_q_table(self, state, action, reward, next_state):
        capital_bin, market_share_bin = self.discretize_state(state)
        next_capital_bin, next_market_share_bin = self.discretize_state(next_state)
        
        best_next_action = np.argmax(self.q_table[next_capital_bin, next_market_share_bin])  # Max Q-value for next state
        td_target = reward + self.discount_factor * self.q_table[next_capital_bin, next_market_share_bin, best_next_action]
        td_error = td_target - self.q_table[capital_bin, market_share_bin, action]
        
        self.q_table[capital_bin, market_share_bin, action] += self.learning_rate * td_error
    
    # Decay the exploration rate
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.decay_rate)
