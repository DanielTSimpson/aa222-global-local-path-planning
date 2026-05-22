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

def cost_path(path, drone, env, weights):
    # this function scores a path based on whatever metrics we're scoring off of for this run
    if len(path) == 0:
        return 1e9 # if the path is empty, give it a really poor score

    # here we incorporate all of the separate costs and rewards we want to consider in our local objective function
    movement_cost = len(path)
    
    science_reward = sum(env.small_science.get(cell, 0) for cell in path if cell in getattr(drone, "known_small_science", set()))
    
    exploration_reward = sum(1 for cell in path if cell not in drone.visited_cells)
    
    obstacle_penalty = sum(1 for cell in path if cell in getattr(drone, "known_small_obstacles", set()))
    
    # now we incorporate the weights we have set for this run
    total_score = - weights["science"] * science_reward - weights["explore"] * exploration_reward + weights["battery"] * movement_cost + weights["obstacle"] * obstacle_penalty
    return total_score

# now we add in our different optimization algorithms

class SimulatedAnnealingOptimizer:
    def __init__(self, horizon=5, iterations=100, initial_temp = 10.0, cooling = 0.95, weights = None):
        self.horizon = horizon
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling = cooling
        self.weights = weights or {"science": 10.0, "explore": 1.0, "battery": 1.0, "obstacle": 100.0}
        
    def choose_action(self, drone, env):
        # choose action is the actual simmulated annealing optimization schema
        
        action_list = list(MOVES.keys())
        
        # we begin with some completely random path which passes out to our horizon
        current = np.random.choice(action_list, size = self.horizon)
        
        # we then score that path to have an initial score to beat in future iterations
        current_score = cost_path(rollout(drone.position, current, env), drone, env, self.weights)
        
        best = current.copy()
        best_score = current_score
        temp = self.initial_temp # starting with a high temperature before the annealing process begins
        
        for _ in range(self.iterations):
            candidate = current.copy()
            idx = np.random.randint(self.horizon) # we pick a random index in the path to alter
            candidate[idx] = np.random.choice(action_list) # and then we substitute the action at that index for a random one
            candidate_score = cost_path(rollout(drone.position, candidate, env), drone, env, self.weights) 
            
            # the next step is to see if our candidate performs better than our current path, or if we still just want to explore because of our temperature parameter
            delta = candidate_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta/temp):
                current = candidate
                current_score = candidate_score
                
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
            
            temp *= self.cooling # and then we cool down our temperature for the following iteration
        
        return int(best[0]) # finally, we return the first action in the best path we ended up with
        