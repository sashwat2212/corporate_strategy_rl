Corporate Strategy RL

Project Overview

This project implements an agentic AI system for strategic corporate planning using reinforcement learning (Q-learning). The system simulates business decisions in a competitive market, including:

Investment Strategies

Marketing Campaigns

Pricing Adjustments

Acquisitions and Mergers

The simulation evaluates agents based on total rewards, financial capital, and market share dynamics.

Architecture

1. Environment (env.py):

Defines the corporate marketplace with dynamic competition.

Implements a reward structure based on revenue, capital growth, and market expansion.

Tracks KPIs: Total rewards, capital, and market share.

2. Agents (agents.py):

Uses Q-learning for strategic decision-making.

Implements exploration-exploitation balance.

Tracks efficiency scores to measure learning progress.

3. Simulations (simulations.py):

Runs multiple training episodes (default: 1000).

Stores data on:

Total rewards per agent

Capital growth trends

Market share evolution

Performance Results

1. Agent Performance Over Episodes

Rewards start low (~200) but increase to ~450-500 per episode, demonstrating successful learning.

2. Capital Growth Analysis

Both agents initially experience capital loss but stabilize around episode 900.

3. Market Share Distribution

Market share fluctuates but eventually stabilizes near 100%, requiring further analysis.

Efficiency Metrics

Efficiency Score: 50,000,000 (both agents)

Market Share: 100% (potential need for competitive adjustments)

Next Steps for Improvement

Optimize reward functions to prevent capital depletion.

Enhance competitive behavior to prevent unrealistic 100% market dominance.

Adjust Q-learning hyperparameters for better decision-making efficiency.

How to Run the Simulation

Install dependencies:

pip install numpy gym matplotlib

Run the simulation:

python simulations.py

View results through generated plots in the output directory.

Contributors

Developed by [Your Name]

Contact: [Your Email]
