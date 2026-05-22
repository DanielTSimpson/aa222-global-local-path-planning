import numpy as np
import config as cfg

# the purpose of this script is to set up a bunch of different local optimization algorithms for the drone
# that way, we can compare them against each other when subjected to different metrics
# things like calculation time, score at end of the simulation, frequency of obstacle collisions, etc.

MOVES = {2: (0, 1), 3: (0, -1), 4: (-1, 0), 5: (1, 0), 6: (1, 1), 7: (-1, 1), 8: (1, -1), 9: (-1, -1)}

def rollout(position, actions, env):
    # this function rolls out a collection of actions from the current position and returns candidate paths
    pos = np.array(position).copy()
    path = []
    
    # from our current position, we iterate through the possible actions the drone could take
    for action in actions:
        dx, dy = MOVES[action]
        candidate = np.array([pos[0] + dx, pos[1] + dy])
        
        # this just guarantees that the drone stays within the bounds of the grid
        candidate[0] = np.clip(candidate[0], 0, env.grid_size - 1)
        candidate[1] = np.clip(candidate[1], 0, env.grid_size - 1)
        
        if env.is_obstacle(candidate[0], candidate[1]):
            break
    
        pos = candidate
        path.append(tuple(pos))
    
    return path

def score_path(path, drone, env, weights):
    # this function scores a path based on whatever metrics we're scoring off of for this run
    if len(path) == 0:
        return -1e9 # if the path is empty, give it a really poor score

    # here we incorporate all of the separate costs and rewards we want to consider in our local objective function
    movement_cost = len(path)
    
    science_reward = sum(env.small_science.get(cell, 0) for cell in path if path in getattr(drone, "known_small_science", set()))
    
    exploration_reward = sum(1 for cell in path if cell not in drone.visited_cells) # TODO same thing as science
    
    obstacle_penalty = sum(1 for cell in path if cell in getattr(drone, "known_small_obstacles", set()))
    
    # now we incorporate the weights we have set for this run
    total_score = weights["science"] * science_reward + weights["explore"] * exploration_reward - weights["battery"] * movement_cost - weights["obstacle"] * obstacle_penalty
    return total_score

# now we add in our different optimization algorithms