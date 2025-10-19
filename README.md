# DDRL: Deep Dueling Reinforcement Learning for MEC Power Control

This project implements a **Deep Dueling Double DQN** agent for **Mobile Edge Computing (MEC)** systems.  
The goal is to dynamically manage radio and computation resources across distributed Remote Radio Heads (RRHs) and mobile devices to optimize **energy efficiency** while satisfying QoS constraints.  

The environment models interference, queue dynamics, and energy harvesting — then the agent learns power allocation and offloading decisions that balance performance and power use.

---

## Core Idea

Each episode simulates a series of time steps where the agent controls how RRHs allocate transmission power and which devices to serve on which subchannels.  
It receives a reward based on system energy efficiency minus Lyapunov drift, encouraging both stability and efficiency.  
Constraints (power limits, interference bounds, rate minimums, etc.) are enforced inside the environment; violations penalize the agent.

---

## Key Components

**`MECEnvDDRL`**  
Custom OpenAI Gym environment that models MEC systems with wireless channel fading, interference, queue backlogs, and device energy harvesting.

**`DuelingDoubleDQNAgent`**  
Deep reinforcement learning agent using a dueling Q-network with soft target updates.  
Supports factorized Q-value outputs to handle large joint action spaces efficiently.

**`FactorizedDuelingQ`**  
Neural network architecture implementing the dueling Q-value structure (value and advantage streams).

**`Replay`**  
Circular replay buffer for experience sampling and training stability.

**`EnvLogger`** and **`EpisodeLog`**  
Logging tools that save per-step and per-episode metrics (rewards, energy efficiency, queue lengths, offloading stats) to CSV.

**`main()`**  
Runs the training loop using default settings and saves logs to the `runs/` directory.

---

## Dependencies

**Python version:** 3.9 or higher

Install dependencies:
```bash
pip install numpy torch tqdm gymnasium
