import numpy as np
import config as cfg
import itertools

# the purpose of this script is to set up a bunch of different local optimization algorithms for the drone
# that way, we can compare them against each other when subjected to different metrics
# things like calculation time, score at end of the simulation, frequency of obstacle collisions, etc.

def make_local_planner(planner_type):
    if planner_type == "simulated_annealing":
        params = getattr(cfg, "SIMANNEAL_HYPERPARAMS", {})
        return SimulatedAnnealingOptimizer(**params)

    if planner_type == "cross_entropy":
        params = getattr(cfg, "CEM_HYPERPARAMS", {})
        return CrossEntropyOptimizer(**params)

    if planner_type == "pomdp":
        params = getattr(cfg, "POMDP_HYPERPARAMS", {})
        return POMDP(**params)

    if planner_type == "genetic":
        params = getattr(cfg, "GENETIC_HYPERPARAMS", {})
        return GeneticOptimizer(**params)

    raise ValueError(f"Unknown planner type: {planner_type}")

def rollout(position, actions, env):
    # this function rolls out a collection of actions from the current position and returns candidate paths
    pos = np.array(position).copy()
    path = []
    
    # from our current position, we iterate through the possible actions the drone could take
    for action in actions:
        dx, dy = cfg.MOVES[action]
        candidate = np.array([pos[0] + dx, pos[1] + dy])
        
        # this just guarantees that the drone stays within the bounds of the grid
        candidate[0] = np.clip(candidate[0], 0, env.grid_size - 1)
        candidate[1] = np.clip(candidate[1], 0, env.grid_size - 1)
        
        if env.is_obstacle(candidate[0], candidate[1]):
            break
    
        pos = candidate
        path.append(tuple(pos))
    
    return path

def _remaining_gps_steps(position, gps):
    reconstructed_path, drone_instructions = gps.a_star(position)
    if drone_instructions is None:
        return float("inf")
    return len(drone_instructions)

def _move_budget_cost(num_steps, drone):
    return num_steps * (drone.movement_cost + drone.time_cost)

def score_path(path, drone, gps, env):
    if len(path) == 0:
        return -float("inf")
    
    local_budget = _move_budget_cost(len(path), drone)
    budget_after_local_path = drone.budget - local_budget
    
    final_cell = path[-1]
    
    steps_to_final_objective = _remaining_gps_steps(final_cell, gps)
    required_finish_budget = _move_budget_cost(steps_to_final_objective, drone) + cfg.MIN_FINAL_BUDGET
    
    if budget_after_local_path < required_finish_budget:
        return -float("inf")
    
    seen_pickups = set()
    science_gain = 0.0
    
    for cell in path:
        if cell in seen_pickups:
            continue
        
        if cell in drone.known_small_science_scores:
            science_gain += drone.known_small_science_scores[cell]
            seen_pickups.add(cell)
            
    terminal_distance = float(np.linalg.norm(np.array(final_cell) - np.array(env.science_pos), ord=np.inf))
    
    return science_gain - cfg.LOCAL_STEP_TIEBREAKER * len(path) - cfg.LOCAL_GOAL_TIEBREAKER * terminal_distance

# now we add in our different optimization algorithms

class SimulatedAnnealingOptimizer:
    def __init__(self, horizon=5, iterations=100, initial_temp = 10.0, cooling = 0.95, weights = None):
        self.horizon = horizon
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling = cooling
        self.weights = weights or {"science": 10.0, "explore": 1.0, "battery": 1.0, "obstacle": 100.0}
        
    def choose_action(self, drone, gps, env):
        # choose action is the actual simmulated annealing optimization schema
        
        action_list = list(cfg.MOVES.keys())
        
        # we begin with some completely random path which passes out to our horizon
        current = np.random.choice(action_list, size = self.horizon)
        
        # we then score that path to have an initial score to beat in future iterations
        current_score = score_path(rollout(drone.position, current, env), drone, gps, env)
        
        best = current.copy()
        best_score = current_score
        temp = self.initial_temp # starting with a high temperature before the annealing process begins
        
        for _ in range(self.iterations):
            candidate = current.copy()
            idx = np.random.randint(self.horizon) # we pick a random index in the path to alter
            candidate[idx] = np.random.choice(action_list) # and then we substitute the action at that index for a random one
            candidate_score = score_path(rollout(drone.position, candidate, env), drone, gps, env) 
            
            # the next step is to see if our candidate performs better than our current path, or if we still just want to explore because of our temperature parameter
            delta = candidate_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta/temp):
                current = candidate
                current_score = candidate_score
                
            if current_score > best_score:
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
        self.action_list = list(cfg.MOVES.keys())

    def choose_action(self, drone, gps, env):
        # for this one, we repeatedly sample action sequences from a probability distyribution
        # we label the sequences that do the best as elites (kind of like for genetic algorithms), and those in turn influence the probability distribution
        # after doing this a couple of times, we 
        num_actions = len(self.action_list)

        # we initialize with a uniform distribution over every action at each timestep
        probs = np.ones((self.horizon, num_actions)) / num_actions

        # first we iterate over the number of, you guessed it, iterations we want to try optimize over
        for _ in range(self.iterations):
            samples = []
            scores = []

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
                score = score_path(path, drone, gps, env)

                samples.append(action_sequence)
                scores.append(score)
            
            scores = np.array(scores)

            # now we select the best performing samples to try and inform our next selection
            elite_indices = np.argsort(scores)[-self.num_elites:]
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
    

class POMDP:  # This guy just keeps looping around the same three points. 
    def __init__(self, horizon = 3, num_simulations = 100, weights=None):
        self.horizon = horizon
        self.num_simulations = num_simulations
        self.weights = weights or {"science": 10.0, "explore": 2.0, "battery": 1.0, "obstacle": 100.0, "path": 2.0, "recovery": 10.0}
        self.action_list = list(cfg.MOVES.keys())
        self.fn_rate = cfg.SENSOR_FALSE_NEGATIVE_RATE
        self.fp_rate = cfg.SENSOR_FALSE_POSITIVE_RATE
    
    def extract_state_and_belief(self, drone, gps):
        agent_state = {
            "position": drone.position.copy(),
            "heading": drone.heading
        }
        belief_state = {
            "visited": drone.visited_cells.copy(),
            "large_obstacles": np.argwhere(gps.large_obstacles > 0).tolist(),
            "small_obstacles": drone.known_small_obstacles.copy(),
            "known_science": drone.known_small_science.copy()
        }
        return agent_state, belief_state

    def sample_environment_from_belief(self, belief_state):
        sampled_obstacles = set()

        for cell in belief_state["visited"]:
            if cell in belief_state["small_obstacles"]:
                if np.random.random() > self.fp_rate:
                    sampled_obstacles.add(cell)
                elif np.random.random() < self.fn_rate:
                    sampled_obstacles.add(cell)
        def is_obstacle_mock(r, c):
            if (r, c) in sampled_obstacles:
                return True
            return False
        return is_obstacle_mock
    
    def choose_action(self, drone, gps, env):
        agent_state, belief_state = self.extract_state_and_belief(drone, gps)
        action_scores = {action: 0.0 for action in self.action_list}

        all_combinations = list(itertools.product(self.action_list, repeat=self.horizon))

        for _ in range(self.num_simulations):
            is_obstacle_mock = self.sample_environment_from_belief(belief_state)

            best_score_for_action = {action: -float('inf') for action in self.action_list}

            original_is_obstacle = env.is_obstacle
            env.is_obstacle = is_obstacle_mock

            for combo in all_combinations:
                path = rollout(agent_state["position"], combo, env)
                score = score_path(path, drone, gps, env)
                first_action = combo[0]
                if score > best_score_for_action[first_action]:
                    best_score_for_action[first_action] = score

            env.is_obstacle = original_is_obstacle
            
            for action in self.action_list:
                action_scores[action] += best_score_for_action[action]

        best_action = max(action_scores, key=action_scores.get)
        return best_action
    
class GeneticOptimizer:
    def __init__(self, horizon=10, population_size=80, generations=12, elite_fraction=0.2, mutation_rate=0.15, crossover_rate=0.8):
        self.horizon = horizon
        self.population_size = population_size
        self.generations = generations
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.action_list = list(cfg.MOVES.keys())
        
    def _random_individual(self):
        # this generates a "random individual", which is really just one of our population
        # each population member is just a set of instructions
        return np.random.choice(self.action_list, size=self.horizon)
    
    def _evaluate(self, individual, drone, gps, env):
        # this determines how well any individual does using the scoring metrics we care about
        path = rollout(drone.position, individual, env)
        return score_path(path, drone, gps, env)
    
    def _tournament_select(self, population, scores, tournament_size = 3):
        # randomly pick a subset of the current population and return the best individual from that subset
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        best_index = max(indices, key = lambda index: scores[index])
        return population[best_index].copy()
    
    def _crossover(self, parent_a, parent_b):
        # this is where we take two individual parents and slam em together to see if their child can do any better
        if np.random.random() > self.crossover_rate:
            return parent_a.copy()
        
        split = np.random.randint(1, self.horizon)
        
        child = np.concatenate([parent_a[:split], parent_b[split:]])
        return child
    
    def _mutate(self, individual):
        # and this is where we add some of our own zest to individuals by mutating random "genes" within them
        mutated = individual.copy()
        for i in range(self.horizon):
            if np.random.random() < self.mutation_rate:
                mutated[i] = np.random.choice(self.action_list)
                
        return mutated
    
    def choose_action(self, drone, gps, env):
        # this is where we actually go about choosing our actions and making use of the genetic algorithm
        
        # first, we take a sample population
        population = [self._random_individual() for _ in range(self.population_size)]
        
        # then we determine how many within that population we're going to deem Elite
        elite_count = max(1, int(self.elite_fraction * self.population_size))
        
        best_individual = None
        best_score = -float("inf")
        
        for _ in range(self.generations):
            scores = np.array([self._evaluate(individual, drone, gps, env) for individual in population])
            
            # once we've scored our population, we figure out who's the Cream Of The Crop
            generation_best_index = int(np.argmax(scores))
            
            # if any of these new individuals happen to do better than our current best, the best gets Replaced
            if scores[generation_best_index] > best_score:
                best_score = scores[generation_best_index]
                best_individual = population[generation_best_index].copy()
            
            # then we begin the process of taking the highest scoring subset of individuals and using them to generate the next population
            elite_indices = np.argsort(scores)[-elite_count:]
            new_population = [population[index] for index in elite_indices]
            
            # once we have our new elites, we use them to generate elite children until we have the desired number of individuals
            while len(new_population) < self.population_size:
                parent_a = self._tournament_select(population, scores)
                parent_b = self._tournament_select(population, scores)

                # for every child, we crossover the genes of the two parents and then give them a bit of a mutagenic twist
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
        # if by some stroke of God we don't end up with a best individual, we just say fuck it and pick a random one
        if best_individual is None:
            return int(np.random.choice(self.action_list))
    
        return int(best_individual[0])