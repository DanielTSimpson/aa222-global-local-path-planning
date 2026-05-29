import numpy as np
import config as cfg
from main import simulate
import time
import csv
from multiprocessing import Pool, cpu_count

# the purpose of this is to implement a Monte Carlo optimization scheme to find the optimal set of weights for the simulated_annealing local planner
# the plan is to extend this to other local planners once those are up

def run_single_trial(args):
    # runs a single test using a set of randomly chosen hyperparameters
    planner_type, params, seed = args
    np.random.seed(seed)

    print(f"Starting {planner_type} trial {seed}")

    apply_hyperparameters(planner_type, params)

    start = time.perf_counter()
    result = simulate(trial_num=seed, render=False, save_gif=False)
    runtime = time.perf_counter() - start

    failure_mode, stats = result

    return seed, (
        failure_mode,
        stats["TOTAL COST"],
        stats["TOTAL_TIME"],
        stats["SMALL_SCIENCE_VALUE"]
    ), runtime


def sample_hyperparameters(planner_type):
    # rather than try and optimize over weights, we're now optimizing hyperparameters for each local planning algo.
    if planner_type == "simulated_annealing":
        return {
            "horizon": np.random.randint(3, 16),
            "iterations": np.random.randint(25, 251),
            "initial_temp": np.random.uniform(1.0, 30.0),
            "cooling": np.random.uniform(0.85, 0.99),
        }

    if planner_type == "cross_entropy":
        return {
            "horizon": np.random.randint(3, 16),
            "num_samples": np.random.randint(25, 251),
            "num_elites": np.random.randint(3, 30),
            "iterations": np.random.randint(2, 12),
            "smoothing": np.random.uniform(0.3, 0.9),
        }

    if planner_type == "pomdp":
        return {
            "horizon": np.random.randint(2, 6),
            "num_simulations": np.random.randint(10, 151),
        }

    if planner_type == "genetic":
        return {
            "horizon": np.random.randint(3, 16),
            "population_size": np.random.randint(20, 151),
            "generations": np.random.randint(3, 31),
            "elite_fraction": np.random.uniform(0.05, 0.4),
            "mutation_rate": np.random.uniform(0.02, 0.35),
            "crossover_rate": np.random.uniform(0.4, 1.0),
        }

    raise ValueError(f"Unknown planner type: {planner_type}")

def apply_hyperparameters(planner_type, params):
    # actually setting the hyperparameters as the ones we want to try out
    cfg.LOCAL_PLANNER_TYPE = planner_type

    if planner_type == "simulated_annealing":
        cfg.SIMANNEAL_HYPERPARAMS = params

    elif planner_type == "cross_entropy":
        cfg.CEM_HYPERPARAMS = params

    elif planner_type == "pomdp":
        cfg.POMDP_HYPERPARAMS = params

    elif planner_type == "genetic":
        cfg.GENETIC_HYPERPARAMS = params

def evaluate_hyperparameters(planner_type, params, pool, num_trials=50, candidate_id=0, csv_path="monte_carlo_trial_results.csv", chunksize=4):
    # does exactly as the function title says -- determines how well a set of hyperparameters performed
    args = [(planner_type, params, seed) for seed in range(num_trials)]

    trial_outputs = list(pool.imap_unordered(run_single_trial, args, chunksize=chunksize))

    results = []

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        for seed, result, runtime in trial_outputs:
            results.append(result + (runtime,))
            writer.writerow([planner_type, candidate_id, seed, runtime, *result, str(params)])

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

def monte_carlo_hyperparameter_search(planner_types = ("simulated_annealing", "cross_entropy", "pomdp", "genetic"), num_candidates = 100, num_trials = 50):
    # the heavy hitter function here, this is what actually consolidates all of our data to hopefully find the best candidates
    # before we search, need to quickly set up our csv file
    
    # the plan is to have a separate csv for trials and for candidate summaries
    trial_csv_path = "monte_carlo_hyperparameter_trials.csv"
    summary_csv_path = "monte_carlo_hyperparameter_summary.csv"

    # first, the trial-level csv
    with open(trial_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "planner_type", "candidate_id", "seed", "runtime_sec",
            "failure_mode", "total_cost", "total_time", "small_science_value", "params"])
    
    # then, the candidate-level csv
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner_type", "candidate_id", "objective", "success_rate", "avg_cost", "avg_time", "avg_science", "avg_runtime", "params", "best_objective_so_far", "best_success_rate_so_far"])

    # now we try the multiprocessing thingie
    num_workers = max(1, cpu_count() - 1)
    chunksize = 1
    
    overall_best = {}

    with Pool(processes=num_workers) as pool:
        # we're running each candidate in parallel to hopefully speed things up
        for planner_type in planner_types:
            print(f"\n===== Searching {planner_type} =====")

            best_params = None
            best_objective = float("inf")
            best_metrics = None
            best_success_rate = -float("inf")
            
            for candidate_id in range(num_candidates):
                params = sample_hyperparameters(planner_type)
                
                objective, metrics = evaluate_hyperparameters(planner_type, params, pool, num_trials=num_trials, candidate_id = candidate_id, csv_path = trial_csv_path, chunksize = chunksize)
                
                if objective < best_objective:
                    best_objective = objective
                    best_params = params
                    best_metrics = metrics
                    
                if metrics["success_rate"] > best_success_rate:
                    best_success_rate = metrics["success_rate"]
                
                with open(summary_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([planner_type, candidate_id, objective, metrics["success_rate"], metrics["avg_cost"], metrics["avg_time"], metrics["avg_science"], metrics["avg_runtime"], str(params), best_objective, best_success_rate])
        
        overall_best[planner_type] = {"best_params": best_params, "best_objective": best_objective, "best_metrics": best_metrics}
        
        print(f"\nBest for {planner_type}:")
        print(best_params)
        print(best_objective)
        print(best_metrics)

    return overall_best

if __name__ == "__main__":
    monte_carlo_hyperparameter_search(planner_types=("simulated_annealing", "cross_entropy", "pomdp", "genetic"), num_candidates = 100, num_trials = 30)
