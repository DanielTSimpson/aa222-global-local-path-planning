import numpy as np
import config as cfg
from main import simulate
import time
import csv
from multiprocessing import Pool, cpu_count

# the purpose of this is to implement a Monte Carlo optimization scheme to find the optimal set of weights for the simulated_annealing local planner
# the plan is to extend this to other local planners once those are up

def run_single_trial(args):
    # running a bunch of monte carlo sims back to back takes forever
    # so, the plan is to try out some Fancy Parallel Computing to speed things up
    
    weights, seed = args
    np.random.seed(seed)
    print(f"Starting trial {seed}")

    # getting our planners in order
    if cfg.LOCAL_PLANNER_TYPE == "cross_entropy":
        cfg.CEM_WEIGHTS = weights
    elif cfg.LOCAL_PLANNER_TYPE == "simulated_annealing":
        cfg.SIMANNEAL_WEIGHTS = weights

    start = time.perf_counter()

    result = simulate(trial_num = seed, render = False, save_gif = False)

    runtime = time.perf_counter() - start

    print(f"Finished trial {seed} in {runtime} [s]")

    failure_mode, stats = result
    return seed, (failure_mode, stats["TOTAL COST"], stats["TOTAL_TIME"], stats["SMALL_SCIENCE_VALUE"]), runtime


def sample_weights():
    # returns a random set of weights for each category
    return {"science": np.random.uniform(0.0, 20.0), "explore": np.random.uniform(0.0, 10.0), "battery": np.random.uniform(0.1, 10.0), "obstacle": np.random.uniform(50.0, 300.0), "path": np.random.uniform(0.0, 20.0), "recovery": np.random.uniform(0.0, 30.0)}

def evaluate_weights(weights, pool, num_trials = 50, candidate_id = 0, csv_path = "monte_carlo_weight_search.csv", chunksize=4):
    # actually running through and checking the efficacy of a set of weights on a collection of different scenarios
    args = [(weights, seed) for seed in range(num_trials)]

    trial_outputs = list(pool.imap_unordered(run_single_trial, args, chunksize=chunksize))

    results = []

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        for seed, result, runtime in trial_outputs:
            results.append(result + (runtime,))
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
    
    # the plan is to have a separate csv for trials and for candidate summaries
    trial_csv_path = "monte_carlo_trial_results.csv"
    summary_csv_path = "monte_carlo_candidate_summary.csv"

    # first, the trial-level csv
    with open(trial_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "seed", "runtime_sec",
            "failure_mode", "total_cost", "total_time", "small_science_value",
            "w_science", "w_explore", "w_battery",
            "w_obstacle", "w_path", "w_recovery",
        ])
    
    # then, the candidate-level csv
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id",
            "objective", "success_rate", "avg_cost", "avg_time", "avg_science", "avg_runtime",

            "w_science", "w_explore", "w_battery", "w_obstacle", "w_path", "w_recovery",

            "best_objective_so_far",
            "best_objective_success_rate",
            "best_objective_avg_science",

            "best_success_rate_so_far",
            "best_success_objective",
            "best_success_avg_science",

            "best_success_w_science",
            "best_success_w_explore",
            "best_success_w_battery",
            "best_success_w_obstacle",
            "best_success_w_path",
            "best_success_w_recovery",
        ])

    best_weights = None
    best_objective = float("inf")
    best_metrics = None

    # I was running into the problem where we were getting "good" results that didn't actually complete the course
    # so, now we're individually tracking successful runs as well
    best_success_weights = None
    best_success_rate = -float("inf")
    best_success_objective = None
    best_success_metrics = None

    # now we try the multiprocessing thingie
    num_workers = max(1, cpu_count() - 1)
    chunksize = 1

    with Pool(processes=num_workers) as pool:
        # we're running each candidate in parallel to hopefully speed things up
        for i in range(num_candidates):
            print(f"Now on candidate {i}")
            # actually computing the results from each candidate
            weights = sample_weights()
            objective, metrics = evaluate_weights(weights, pool, num_trials = num_trials, candidate_id = i, csv_path = trial_csv_path, chunksize=chunksize)

            # the first thing we check is which candidates do best by overall objective
            if objective < best_objective:
                best_objective = objective
                best_weights = weights
                best_metrics = metrics

                print(f"New best objective at candidate {i}: {best_objective:.3f}")

            # then we check which candidates actually succeed the most
            if metrics["success_rate"] > best_success_rate:
                best_success_rate = metrics["success_rate"]
                best_success_weights = weights
                best_success_objective = objective
                best_success_metrics = metrics
                print(f"New best success rate at candidate {i}: {best_success_rate:.3f}")

            with open(summary_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i,
                    objective,
                    metrics["success_rate"],
                    metrics["avg_cost"],
                    metrics["avg_time"],
                    metrics["avg_science"],
                    metrics["avg_runtime"],

                    weights["science"],
                    weights["explore"],
                    weights["battery"],
                    weights["obstacle"],
                    weights["path"],
                    weights["recovery"],

                    best_objective,
                    best_metrics["success_rate"] if best_metrics else None,
                    best_metrics["avg_science"] if best_metrics else None,

                    best_success_rate,
                    best_success_objective,
                    best_success_metrics["avg_science"] if best_success_metrics else None,

                    best_success_weights["science"] if best_success_weights else None,
                    best_success_weights["explore"] if best_success_weights else None,
                    best_success_weights["battery"] if best_success_weights else None,
                    best_success_weights["obstacle"] if best_success_weights else None,
                    best_success_weights["path"] if best_success_weights else None,
                    best_success_weights["recovery"] if best_success_weights else None,
                ])
    


    print("\nBest objective weights:")
    print(best_weights)
    print("Best objective:", best_objective)
    print("Best objective metrics:", best_metrics)

    print("\nBest success-rate weights:")
    print(best_success_weights)
    print("Best success rate:", best_success_rate)
    print("Best success objective:", best_success_objective)
    print("Best success metrics:", best_success_metrics)

    return best_weights, best_objective, best_metrics, best_success_weights, best_success_objective, best_success_metrics

if __name__ == "__main__":
    monte_carlo_weight_search(num_candidates = 10, num_trials = 10)
