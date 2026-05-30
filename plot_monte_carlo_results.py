import ast
import pandas as pd
import matplotlib.pyplot as plt

# the purpose of this script is to show off that monte carlo actually results in better performing local optimizers than just hand picking local hyperparameters

SUMMARY_CSV = "monte_carlo_hyperparameter_summary.csv"
TRIAL_CSV = "monte_carlo_hyperparameter_trials.csv"

def load_results():
    summary = pd.read_csv(SUMMARY_CSV)
    trials = pd.read_csv(TRIAL_CSV)

    if "params" in summary.columns:
        summary["params_dict"] = summary["params"].apply(ast.literal_eval)

    return summary, trials

def plot_objective_progress(summary):
    plt.figure()
    for planner, group in summary.groupby("planner_type"):
        group = group.sort_values("candidate_id")
        best_so_far = group["objective"].cummin()
        plt.plot(group["candidate_id"], best_so_far, label=planner)
        
    plt.xlabel("Candidate ID")
    plt.ylabel("Best objective so far")
    plt.title("Monte Carlo Search Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_objective_progress.png", dpi=300)
    
def plot_success_rate(summary):
    plt.figure()
    for planner, group in summary.groupby("planner_type"):
        group = group.sort_values("candidate_id")
        plt.scatter(group["candidate_id"], group["success_rate"], label=planner, alpha=0.7)

    plt.xlabel("Candidate ID")
    plt.ylabel("Success rate")
    plt.title("Success Rate by Hyperparameter Candidate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_success_rate.png", dpi=300)


def plot_science_vs_runtime(summary):
    plt.figure()
    for planner, group in summary.groupby("planner_type"):
        plt.scatter(group["avg_runtime"], group["avg_science"], label=planner, alpha=0.7)

    plt.xlabel("Average runtime per trial [s]")
    plt.ylabel("Average small science collected")
    plt.title("Science Return vs Runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_science_vs_runtime.png", dpi=300)


def plot_science_by_planner(trials):
    successful = trials[trials["failure_mode"] == 0]

    data = [
        successful[successful["planner_type"] == planner]["small_science_value"]
        for planner in successful["planner_type"].unique()
    ]
    labels = list(successful["planner_type"].unique())

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.xlabel("Planner type")
    plt.ylabel("Small science collected")
    plt.title("Distribution of Science Collected on Successful Runs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_science_boxplot.png", dpi=300)


def plot_cost_vs_science(summary):
    plt.figure()
    for planner, group in summary.groupby("planner_type"):
        plt.scatter(group["avg_cost"], group["avg_science"], label=planner, alpha=0.7)

    plt.xlabel("Average total cost")
    plt.ylabel("Average small science collected")
    plt.title("Science Return vs Mission Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_cost_vs_science.png", dpi=300)


def print_best_candidates(summary):
    print("\nBest candidate by objective for each planner:\n")

    for planner, group in summary.groupby("planner_type"):
        best = group.loc[group["objective"].idxmin()]
        print(f"Planner: {planner}")
        print(f"  candidate_id: {best['candidate_id']}")
        print(f"  objective: {best['objective']}")
        print(f"  success_rate: {best['success_rate']}")
        print(f"  avg_science: {best['avg_science']}")
        print(f"  avg_runtime: {best['avg_runtime']}")
        print(f"  params: {best['params']}")
        print()


def main():
    summary, trials = load_results()

    plot_objective_progress(summary)
    plot_success_rate(summary)
    plot_science_vs_runtime(summary)
    plot_science_by_planner(trials)
    plot_cost_vs_science(summary)

    print_best_candidates(summary)

    print("Saved plots:")
    print("  plot_objective_progress.png")
    print("  plot_success_rate.png")
    print("  plot_science_vs_runtime.png")
    print("  plot_science_boxplot.png")
    print("  plot_cost_vs_science.png")


if __name__ == "__main__":
    main()