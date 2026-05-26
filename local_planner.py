import numpy as np
import config as cfg

# the purpose of this script is to set up a bunch of different local optimization algorithms for the drone
# that way, we can compare them against each other when subjected to different metrics
# things like calculation time, score at end of the simulation, frequency of obstacle collisions, etc.

MOVES = {2: (0, 1), 3: (0, -1), 4: (-1, 0), 5: (1, 0), 6: (1, 1), 7: (-1, 1), 8: (1, -1), 9: (-1, -1)}

def make_local_planner(planner_type):
    # this is what'll be called in drone.py rather than having to pick a certain local planner explicitly
    # instead, we'll just make a call to this function and everything is taken care of here
    if planner_type == "simulated_annealing":
        return SimulatedAnnealingOptimizer(horizon=5, iterations=100, initial_temp=10.0, cooling=0.95, weights=cfg.SIMANNEAL_WEIGHTS)

    if planner_type == "cross_entropy":
        return CrossEntropyOptimizer(horizon=5, num_samples=100, num_elites=10, iterations=5, smoothing=0.7, weights=cfg.CEM_WEIGHTS)

    # TODO this part isn't done yet
    if planner_type == "genetic":
        weights = cfg.GENETIC_WEIGHTS
        return None

    raise ValueError(f"Unknown planner type: {planner_type}") 

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
    
    # in order to get our drone to eventually return to the nominal path, we include a penalty for straying too far from the main path
    path_deviation_cost = 0.0
    if hasattr(drone, "global_path"):
        for cell in path:
            distances = [abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in drone.global_path]
            path_deviation_cost += min(distances) # we take the minimum distance from the cell to any point in the global path as our deviation cost
    
        # keep running into issues where the drone wanders and wanders around the same area without approaching the final destination
        # this hopefully helps with that some
        recovery_cost = 0.0
        final_cell = path[-1]
        future_path = drone.global_path[getattr(drone, "global_path_index", 0):]
        if len(future_path) > 0:
            recovery_cost = min(abs(final_cell[0] - p[0]) + abs(final_cell[1] - p[1]) for p in future_path)

    # now we incorporate the weights we have set for this run
    # the first 3 terms are things we don't want, so they're additive. the last 2 terms are things we do want, so they're subtractive
    total_score = (weights["battery"] * movement_cost + weights["obstacle"] * obstacle_penalty + weights["path"] * path_deviation_cost + weights["recovery"] * recovery_cost) - (weights["science"] * science_reward + weights["explore"] * exploration_reward)
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
        
class CrossEntropyOptimizer:
    def __init__(self, horizon = 5, num_samples = 100, num_elites = 10, iterations = 5, smoothing = 0.7, weights = None):
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.smoothing = smoothing
        self.weights = weights or {"science": 10.0, "explore": 2.0, "battery": 1.0, "obstacle": 100.0, "path": 2.0, "recovery": 10.0}
        self.action_list = list(MOVES.keys())

    def choose_action(self, drone, env):
        # TODO write a description here
        num_actions = len(self.action_list)

        # we initialize with a uniform distribution over every action at each timestep
        probs = np.ones((self.horizon, num_actions)) / num_actions

        # first we iterate over the number of, you guessed it, iterations we want to try optimize over
        for _ in range(self.iterations):
            samples = []
            costs = []

            # then, we draw however many samples we need to try and paint a clear picture of the solution space
            for _ in range(self.num_samples):
                action_sequence = []

                # then we pick a set of probablistic actions in accordance with our probability distribution
                # the plan is we, hopefully, move towards a better and better representation of the solution space as we generate more and more samples
                for t in range(self.horizon):
                    action_idx = np.random.choice(num_actions, p = probs[t])
                    action_sequence.append(self.action_list[action_idx])

                action_sequence = np.array(action_sequence)

                path = rollout(drone.position, action_sequence, env)
                cost = cost_path(path, drone, env, self.weights)

                samples.append(action_sequence)
                costs.append(cost)
            
            costs = np.array(costs)

            # now we select the best performing samples to try and inform our next selection
            elite_indices = np.argsort(costs)[:self.num_elites]
            elite_samples = [samples[i] for i in elite_indices]

            new_probs = np.zeros_like(probs)

            # here we go ahead and update our probability distribution
            for t in range(self.horizon):
                for action_index, action in enumerate(self.action_list):
                    count = sum(sample[t] == action for sample in elite_samples)
                    new_probs[t, action_index] = count / self.num_elites

            # smoothing out the probability distribution because it was being a pain in the ass beforehand
            probs = (self.smoothing * probs + (1.0 - self.smoothing) * new_probs)

            # dancing around exact zeros
            probs += 1e-6
            probs /= probs.sum(axis=1, keepdims=True)
        
        best_action_index = np.argmax(probs[0])
        return int(self.action_list[best_action_index])