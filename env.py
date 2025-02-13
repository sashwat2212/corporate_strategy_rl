import numpy as np
import random
import gym
from gym import spaces

class CorporateStrategyEnv(gym.Env):
    def __init__(self, num_companies=2, initial_capital=1000, initial_market_share=50):
        super(CorporateStrategyEnv, self).__init__()
        self.num_companies = num_companies
        self.initial_capital = initial_capital
        self.initial_market_share = initial_market_share
        
        # Define state: Each company has capital and market share
        self.state = np.array([
            [initial_capital, initial_market_share] for _ in range(num_companies)
        ], dtype=np.float32)
        
        # Define action space: 0 = Invest, 1 = Marketing, 2 = Pricing, 3 = Acquisition
        self.action_space = spaces.Discrete(4)
        
        # Observation space (Capital, Market Share for each company)
        self.observation_space = spaces.Box(
            low=np.array([[0, 0] for _ in range(num_companies)], dtype=np.float32),
            high=np.array([[np.inf, 100] for _ in range(num_companies)], dtype=np.float32),
            dtype=np.float32
        )
        
    def step(self, actions):
        rewards = np.zeros(self.num_companies)
        market_fluctuation = random.uniform(-0.05, 0.05)  # Random market changes (-5% to +5%)
        
        for i, action in enumerate(actions):
            capital, market_share = self.state[i]
            
            if action == 0:  # Invest in R&D
                capital -= 50
                market_share += 5
                rewards[i] = 10  # Investment may yield long-term gains
            
            elif action == 1:  # Marketing
                capital -= 30
                market_share += 3
                rewards[i] = 7
                
            elif action == 2:  # Pricing Strategy
                market_share += random.choice([-2, 2])  # Risk-reward tradeoff
                rewards[i] = 5 if market_share > self.initial_market_share else -5
                
            elif action == 3:  # Acquisition (if enough capital)
                if capital > 200:
                    capital -= 200
                    market_share += 10
                    rewards[i] = 15
                else:
                    rewards[i] = -10  # Penalty for attempting acquisition with low capital
                    
            # Apply market fluctuation
            market_share *= (1 + market_fluctuation)
            
            # Update state
            self.state[i] = [capital, market_share]
        
        return self.state, rewards, False, {}

    def reset(self):
        self.state = np.array([
            [self.initial_capital, self.initial_market_share] for _ in range(self.num_companies)
        ], dtype=np.float32)
        return self.state

    def render(self):
        for i, (capital, market_share) in enumerate(self.state):
            print(f"Company {i}: Capital = {capital}, Market Share = {market_share}%")


# import numpy as np
# import random
# import gym
# from gym import spaces
# #from corporate_strategy_env import CorporateStrategyEnv

# # Initialize the environment
# env = CorporateStrategyEnv(num_companies=2, initial_capital=1000, initial_market_share=50)

# # Reset the environment (this gives you the initial state)
# state = env.reset()

# # Run a few steps in the environment
# for episode in range(10):  # Run for 10 episodes
#     print(f"Episode {episode + 1}")
#     done = False
#     total_rewards = np.zeros(env.num_companies)

#     while not done:
#         actions = np.random.choice(env.action_space.n, env.num_companies)  # Random actions for each agent
#         next_state, rewards, done, _ = env.step(actions)  # Take a step
#         total_rewards += rewards
        
#         # Optionally render the state (to visualize it)
#         env.render()

#     print(f"Total rewards for Episode {episode + 1}: {total_rewards}")

