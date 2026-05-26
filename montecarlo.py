import numpy as np
import config as cfg
from main import simulate_astar
import time
import csv
from tqdm import tqdm

# the purpose of this is to implement a Monte Carlo optimization scheme to find the optimal set of weights for the simulated_annealing local planner
# the plan is to extend this to other local planners once those are up

def sample_weights():
    # returns a random set of weights for each category
    return {"science": np.random.uniform(0.0, 20.0), "explore": np.random.uniform(0.0, 10.0), "battery": np.random.uniform(0.1, 10.0), "obstacle": np.random.uniform(50.0, 300.0), "path": np.random.uniform(0.0, 20.0), "recovery": np.random.uniform(0.0, 30.0)}

def evaluate_weights(weights, num_trials = 50, candidate_id = 0, csv_path = "monte_carlo_weight_search.csv"):
    # actually running through and checking the efficacy of a set of weights on a collection of different scenarios
    results = []

    cfg.LOCAL_PLANNER_WEIGHTS = weights

    for seed in tqdm(range(num_trials), desc=f"Trials for candidate {candidate_id}", leave=False):
        np.random.seed(seed)

        start = time.perf_counter()

        result = simulate_astar(trial_num=seed, render=0, save_gif=False)

        runtime = time.perf_counter() - start

        results.append(result + (runtime,))

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                candidate_id,
                seed,
                runtime,
                *result,
                weights["science"],
                weights["explore"],
                weights["battery"],
                weights["obstacle"],
                weights["path"],
                weights["recovery"],
            ])
    return score_results(results)

def score_results(results):
    # hopefully the function name is self explanatory for this one
    successes = [r for r in results if r[0] == 0]

    # success_rate is pretty self explanatory
    success_rate = len(successes)/len(results)
    avg_runtime = np.mean([r[4] for r in results])

    # if we actually see some successes, then we want to keep those categorized
    if len(successes) > 0:
        avg_cost = np.mean([r[1] for r in successes])
        avg_time = np.mean([r[2] for r in successes])
        avg_science = np.mean([r[3] for r in successes])
    else:
        avg_cost = 1e9
        avg_time = 1e9
        avg_science = 0.0

    # the objective function below here is just a nice heuristic to judge how well the monte carlo instance did for single-value comparison
    objective = (-1000.0 * success_rate + 0.1 * avg_cost + 1.0 * avg_time - 10.0 * avg_science + 5.0 * avg_runtime)

    return objective, {"success_rate": success_rate, "avg_cost": avg_cost, "avg_time": avg_time, "avg_science": avg_science, "avg_runtime": avg_runtime}

def monte_carlo_weight_search(num_candidates = 100, num_trials = 50):
    # the heavy hitter function here, this is what actually consolidates all of our data to hopefully find the best candidates
    # before we search, need to quickly set up our csv file
    with open("monte_carlo_weight_search.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id",
            "seed",
            "runtime_sec",
            "failure_mode",
            "total_cost",
            "total_time",
            "small_science_value",
            "w_science",
            "w_explore",
            "w_battery",
            "w_obstacle",
            "w_path",
            "w_recovery",
        ])
    
    best_weights = None
    best_objective = float("inf")
    best_metrics = None

    for i in tqdm(range(num_candidates), desc="Weight Candidates"):
        # actually computing the results from each candidate
        weights = sample_weights()
        objective, metrics = evaluate_weights(weights, num_trials = num_trials, candidate_id = i)

        # keeping ourselves informed on how each instance is doing
        print(f"\nCandidate {i + 1}/{num_candidates}")
        print(f"Weights: {weights}")
        print(f"Objective: {objective:.3f}")
        print(f"Metrics: {metrics}")

        if objective < best_objective:
            best_objective = objective
            best_weights = weights
            best_metrics = metrics

            print("NEW BEST")

    print("\nBest weights:")
    print(best_weights)
    print("Best objective:", best_objective)
    print("Best metrics:", best_metrics)

    return best_weights, best_objective, best_metrics

if __name__ == "__main__":
    monte_carlo_weight_search(num_candidates = 30, num_trials = 30)
