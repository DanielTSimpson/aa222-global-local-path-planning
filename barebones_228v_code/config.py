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

# ===== Obstacle parameters ===== 
OBSTACLES_ENABLED = True # I figure it's nice to be able to toggle this when testing our path planning
NUM_LARGE_OBSTACLES = 10 # TODO could be good later to have our number of obstacles be a function of the size of the grid
NUM_SMALL_OBSTACLES = 20

LARGE_OBSTACLE_SIZE_MU = 4 # Mean size of the larger obstacles -- I think that having the required size for visibility by the overhead agent be ~3 cells is a good place to start
LARGE_OBSTACLE_SIZE_SIGMA = 1 # Standard deviation of the large obstacle size

SMALL_OBSTACLE_SIZE_MU = 1.5 # Mean size of the smaller obstacles, only detectable by the drone
SMALL_OBSTACLE_SIZE_SIGMA = 0.5 # Standard deviation of the small obstacle size

OBSTACLE_MAX_PLACEMENT_ATTEMPTS = 200 # number of times we try and find a new viable spot for an obstacle before giving up and moving onto the next one
OBSTACLE_BUFFER_AROUND_OBJECTIVES = 2 # Number of empty cells around the objective
OBSTACLE_BUFFER_AROUND_START = 2 # Number of empty cells around the drone start location

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