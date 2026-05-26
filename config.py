import numpy as np
"""
Configuration file for Dec-POMDP multi-agent simulation
Centralized place for all simulation parameters
"""

#  ===== Environment parameters ===== 
SEED = 42 # random seed for reproducibility
GRID_SIZE = 50  # Size of the NxN grid
ENABLE_WIND = False # Wind Toggle
WIND_SPEED = np.random.normal(0.05, 0.05**2) # Probability of agents drifting after an action
WIND_DIRECTION = 2*np.pi*np.random.random() # Direction of the wind in radians (CCW)

# ===== Obstacle parameters ===== 
OBSTACLES_ENABLED = True # I figure it's nice to be able to toggle this when testing our path planning
 
NUM_LARGE_OBSTACLES = GRID_SIZE // 3
LARGE_OBSTACLE_SIZE_MU = 7 # Mean size of the larger obstacles -- I think that having the required size for visibility by the overhead agent be ~3 cells is a good place to start
LARGE_OBSTACLE_SIZE_SIGMA = 3 # Standard deviation of the large obstacle size

NUM_SMALL_OBSTACLES = GRID_SIZE // 1
SMALL_OBSTACLE_SIZE_MU = 1.5 # Mean size of the smaller obstacles, only detectable by the drone
SMALL_OBSTACLE_SIZE_SIGMA = 0.5 # Standard deviation of the small obstacle size

OBSTACLE_BUFFER_AROUND_OBJECTIVES = 2 # Number of empty cells around the objective
OBSTACLE_BUFFER_AROUND_START = 2 # Number of empty cells around the drone start location

# ===== Small Science parameters ===== 
SMALL_SCIENCE_ENABLED = True # toggle for whether we want to include small science objectives throughout the grid world or not
NUM_SMALL_SCIENCE = GRID_SIZE // 4 # how many small science objectives are scattered throughout the grid world
SMALL_SCIENCE_MIN_VALUE = 1 # the minimum value of a small science objective
SMALL_SCIENCE_MAX_VALUE = 5 # the maximum value of a small science objective

# ===== Drone sensor parameters ===== 
SENSOR_VIEW_DEPTH = 4
SENSOR_VIEW_ANGLE = np.pi/2
SENSOR_FALSE_NEGATIVE_RATE = 0.05
SENSOR_FALSE_POSITIVE_RATE = 0.01

# === Cost parameters ===
MOVEMENT_COST = 1.0
TIME_COST = 3.0

# ===== Simulation (main) parameters ===== 
INITIAL_TIME = 0.0
TIME_STEP = 0.05
MAX_SIMULATION_TIME = 250.0
MAX_BUDGET = 5000
RENDER_PAUSE = 0.05

# ===== Local planner parameters =====
LOCAL_PLANNER_TYPE = "cross_entropy" # options are simulated_annealing, cross_entropy, and genetic

# TODO LATER: rather than have to tune each weight individually, it could be good to just have different run types
# for instance, we could have a "science-focused" run where science is weighted above a nominal value, or a "battery-focused" run and so on
SIMANNEAL_WEIGHTS = {"science": 10.0, "explore": 2.0, "battery": 1.0, "obstacle": 100.0, "path": 5.0, "recovery": 10.0}
CEM_WEIGHTS = {"science": 10.0, "explore": 2.0, "battery": 1.0, "obstacle": 100.0, "path": 5.0, "recovery": 10.0}

# ===== Logging/Terminal Output =====
VERBOSE_LOGGING = False