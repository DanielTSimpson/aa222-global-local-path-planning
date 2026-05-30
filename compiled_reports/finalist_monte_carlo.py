import ast
import csv
import pandas as pd

from montecarlo import evaluate_hyperparameters


SUMMARY_CSV = "monte_carlo_hyperparameter_summary.csv"
FINALIST_TRIAL_CSV = "finalist_trial_results.csv"
FINALIST_SUMMARY_CSV = "finalist_summary_results.csv"


def load_top_candidates(top_n=3):
    df = pd.read_csv(SUMMARY_CSV)

    finalists = []

    for planner_type, group in df.groupby("planner_type"):
        top = group.sort_values("objective").head(top_n)

        for _, row in top.iterrows():
            finalists.append({
                "planner_type": planner_type,
                "original_candidate_id": int(row["candidate_id"]),
                "params": ast.literal_eval(row["params"]),
                "old_objective": row["objective"],
                "old_success_rate": row["success_rate"],
                "old_avg_science": row["avg_science"],
            })

    return finalists


def run_finalists(top_n=3, num_trials=100, timeout=60):
    finalists = load_top_candidates(top_n=top_n)

    with open(FINALIST_TRIAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "planner_type",
            "candidate_id",
            "seed",
            "runtime_sec",
            "failure_mode",
            "total_cost",
            "total_time",
            "small_science_value",
            "params",
        ])

    with open(FINALIST_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "planner_type",
            "original_candidate_id",
            "finalist_id",
            "objective",
            "success_rate",
            "avg_cost",
            "avg_time",
            "avg_science",
            "avg_runtime",
            "old_objective",
            "old_success_rate",
            "old_avg_science",
            "params",
        ])

    for finalist_id, finalist in enumerate(finalists):
        planner_type = finalist["planner_type"]
        params = finalist["params"]

        print(
            f"\nRunning finalist {finalist_id + 1}/{len(finalists)}: "
            f"{planner_type}, original candidate {finalist['original_candidate_id']}",
            flush=True,
        )

        objective, metrics = evaluate_hyperparameters(
            planner_type=planner_type,
            params=params,
            num_trials=num_trials,
            candidate_id=finalist_id,
            csv_path=FINALIST_TRIAL_CSV,
            timeout=timeout,
        )

        with open(FINALIST_SUMMARY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                planner_type,
                finalist["original_candidate_id"],
                finalist_id,
                objective,
                metrics["success_rate"],
                metrics["avg_cost"],
                metrics["avg_time"],
                metrics["avg_science"],
                metrics["avg_runtime"],
                finalist["old_objective"],
                finalist["old_success_rate"],
                finalist["old_avg_science"],
                str(params),
            ])

        print(f"Finalist objective: {objective}")
        print(f"Finalist metrics: {metrics}")

    print("\nDone.")
    print(f"Trial results saved to {FINALIST_TRIAL_CSV}")
    print(f"Summary saved to {FINALIST_SUMMARY_CSV}")


if __name__ == "__main__":
    run_finalists(
        top_n=3,
        num_trials=100,
        timeout=60,
    )