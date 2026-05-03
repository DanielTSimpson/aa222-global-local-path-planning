import numpy as np
"""
Configuration file for Dec-POMDP multi-agent simulation
Centralized place for all simulation parameters
"""

# ===== Simulation (main) parameters ===== 
INITIAL_TIME = 0.0
TIME_STEP = 0.05
MAX_SIMULATION_TIME = 250.0
MAX_BUDGET = 5000
RENDER_PAUSE = 0.25

#  ===== Environment parameters ===== 
GRID_SIZE = 25  # Size of the NxN grid
WIND_SPEED = np.random.normal(0.05, 0.05**2) # Probability of agents drifting after an action
WIND_DIRECTION = 2*np.pi*np.random.random() # Direction of the wind in radians (CCW)

# ===== Drone parameters ===== 
NUM_DRONES = 2
OBSERVATION_WINDOW_SIZE = 3
LOOKAHEAD_DEPTH = 3

# ===== Dec-POMDP parameters ===== 
GAMMA = 0.95
EXPLORATION_BONUS = 50.0  # Bonus reward for exploring new cells, promotes active exploration of new cells

# === Cost parameters ===
COMMUNICATION_COST = 3.0
MOVEMENT_COST = 1.0
TIME_COST = 3.0

# Initial, nominal parameters
# Order: [W_dist, mu_dist, var_dist, mu_wind, var_wind, W_angle, var_wind_angle_change]
MU_P = np.array([
    1.0,                            # W_dist
    0.50,                           # mu_dist
    0.50,                           # var_dist
    0.05,                           # mu_wind
    0.01,                           # var_wind
    1,                              # W_angle
    1.0                             # var_wind_angle_change
])

MU_Q = np.array([
    0,                              # W_dist
    0.90,                           # mu_dist
    0.10,                           # var_dist
    0.3,                            # mu_wind
    0.01,                           # var_wind
    0,                              # W_angle
    0.01                            # var_wind_angle_change
])