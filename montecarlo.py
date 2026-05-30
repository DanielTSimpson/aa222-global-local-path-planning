import numpy as np
import config as cfg
from main import simulate
import time
import csv
from multiprocessing import Queue, Process, cpu_count
import queue as queue_module

# the purpose of this is to implement a Monte Carlo optimization scheme to find the optimal set of weights for the simulated_annealing local planner
# the plan is to extend this to other local planners once those are up
        
def _trial_worker(q, planner_type, params, seed):
    try:
        np.random.seed(seed)
        apply_hyperparameters(planner_type, params)

        start = time.perf_counter()
        result = simulate(
            trial_num=seed,
            render=False,
            save_gif=False
        )
        runtime = time.perf_counter() - start

        failure_mode, stats = result

        q.put((
            seed,
            (
                failure_mode,
                stats["TOTAL COST"],
                stats["TOTAL_TIME"],
                stats["SMALL_SCIENCE_VALUE"],
            ),
            runtime,
        ))

    except Exception:
        runtime = 0.0
        q.put((seed, (-2, 1e9, 1e9, 0.0), runtime))
        
def run_trials_parallel(args_list, timeout=60, max_workers=None):
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    remaining = list(args_list)
    results = []
    
    completed = 0
    total = len(args_list)

    while remaining:
        batch = remaining[:max_workers]
        remaining = remaining[max_workers:]

        processes = []

        for args in batch:
            q = Queue()
            p = Process(
                target=_trial_worker,
                args=(q, *args)
            )
            p.start()
            processes.append((p, q, args, time.perf_counter()))

        for p, q, args, start in processes:
            planner_type, params, seed = args

            p.join(timeout=timeout)

            if p.is_alive():
                p.terminate()
                p.join()

                runtime = time.perf_counter() - start
                results.append((seed, (-1, 1e9, 1e9, 0.0), runtime))
                completed += 1
                print(f"Completed {completed}/{total} (TIMEOUT) | seed={seed} | runtime={runtime:.1f}", flush=True)

            else:
                try:
                    results.append(q.get_nowait())
                    completed += 1
                    print(f"Completed {completed}/{total} | seed={seed}", flush=True)
                except queue_module.Empty:
                    runtime = time.perf_counter() - start
                    results.append((seed, (-2, 1e9, 1e9, 0.0), runtime))
                    print(f"Completed {completed}/{total} | seed={seed}", flush=True)

    return results

def sample_hyperparameters(planner_type):
    # rather than try and optimize over weights, we're now optimizing hyperparameters for each local planning algo.
    if planner_type == "simulated_annealing":
        return {
            "horizon": np.random.randint(3, 16),
            "iterations": np.random.randint(25, 101),
            "initial_temp": np.random.uniform(1.0, 30.0),
            "cooling": np.random.uniform(0.85, 0.99),
        }
        
    # to prevent running into issues, I'm gonna do a lil magic
    num_samples = np.random.randint(25, 101)
    num_elites = np.random.randint(3, min(30, num_samples) + 1)

    if planner_type == "cross_entropy":
        return {
            "horizon": np.random.randint(3, 16),
            "num_samples": num_samples,
            "num_elites": num_elites,
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
            "population_size": np.random.randint(20, 81),
            "generations": np.random.randint(3, 16),
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

def evaluate_hyperparameters(planner_type, params, num_trials=50, candidate_id=0, csv_path="monte_carlo_trial_results.csv", timeout=60):
    # does exactly as the function title says -- determines how well a set of hyperparameters performed
    args = [(planner_type, params, seed) for seed in range(num_trials)]

    trial_outputs = run_trials_parallel(args, timeout=timeout, max_workers = max(1, cpu_count() - 1))

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

def monte_carlo_hyperparameter_search(planner_types = ("simulated_annealing", "cross_entropy", "genetic"), num_candidates = 100, num_trials = 50):
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
    
    overall_best = {}

    for planner_type in planner_types:
        print(f"\n===== Searching {planner_type} =====")

        best_params = None
        best_objective = float("inf")
        best_metrics = None
        best_success_rate = -float("inf")
        
        for candidate_id in range(num_candidates):
            print(f"\n[{planner_type}] Candidate {candidate_id + 1}/{num_candidates}", flush=True)
            params = sample_hyperparameters(planner_type)
            
            objective, metrics = evaluate_hyperparameters(planner_type, params, num_trials=num_trials, candidate_id = candidate_id, csv_path = trial_csv_path)
            
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
    monte_carlo_hyperparameter_search(planner_types=("simulated_annealing", "cross_entropy", "genetic"), num_candidates = 20, num_trials = 15)
