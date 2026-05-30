import ast
import pandas as pd
import matplotlib.pyplot as plt


FINALIST_TRIAL_CSV = "finalist_trial_results.csv"
FINALIST_SUMMARY_CSV = "finalist_summary_results.csv"


def load_data():
    trials = pd.read_csv(FINALIST_TRIAL_CSV)
    summary = pd.read_csv(FINALIST_SUMMARY_CSV)

    if "params" in summary.columns:
        summary["params_dict"] = summary["params"].apply(ast.literal_eval)

    return trials, summary


def summarize_trials(trials):
    summary = (
        trials.groupby("planner_type")
        .agg(
            num_trials=("failure_mode", "count"),
            success_rate=("failure_mode", lambda x: (x == 0).mean()),
            avg_science=("small_science_value", "mean"),
            std_science=("small_science_value", "std"),
            avg_cost=("total_cost", "mean"),
            avg_runtime=("runtime_sec", "mean"),
            std_runtime=("runtime_sec", "std"),
        )
        .reset_index()
    )

    summary["science_ci95"] = (
        1.96 * summary["std_science"] / summary["num_trials"] ** 0.5
    )

    summary["runtime_ci95"] = (
        1.96 * summary["std_runtime"] / summary["num_trials"] ** 0.5
    )

    return summary


def plot_bar(summary, y, yerr, title, ylabel, filename):
    plt.figure()

    if yerr is None:
        plt.bar(
            summary["planner_type"],
            summary[y],
        )
    else:
        plt.bar(
            summary["planner_type"],
            summary[y],
            yerr=summary[yerr],
            capsize=6,
        )

    plt.xlabel("Planner type")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_science_distribution(trials):
    plot_df = trials.dropna(subset=["small_science_value"])

    planners = []
    data = []

    for planner in sorted(plot_df["planner_type"].unique()):
        values = plot_df.loc[
            plot_df["planner_type"] == planner,
            "small_science_value"
        ].dropna()

        if len(values) == 0:
            continue

        planners.append(planner)
        data.append(values.to_numpy())

    if len(data) == 0:
        print("No science data available for boxplot.")
        return

    plt.figure()
    plt.boxplot(data, tick_labels=planners)
    plt.xlabel("Planner type")
    plt.ylabel("Small science collected")
    plt.title("Science Collection Distribution by Planner")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("final_science_distribution.png", dpi=300)


def plot_science_vs_runtime(summary):
    plt.figure()
    plt.scatter(summary["avg_runtime"], summary["avg_science"])

    for _, row in summary.iterrows():
        plt.annotate(
            row["planner_type"],
            (row["avg_runtime"], row["avg_science"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    plt.xlabel("Average runtime per trial [s]")
    plt.ylabel("Average small science collected")
    plt.title("Science Return vs Runtime")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("final_science_vs_runtime.png", dpi=300)


def print_best_finalists(summary):
    for planner, group in summary.groupby("planner_type"):
        best = group.sort_values("objective").iloc[0]

        print(f"\nPlanner: {planner}")
        print(f"  Finalist ID: {best['finalist_id']}")
        print(f"  Original candidate ID: {best['original_candidate_id']}")
        print(f"  Objective: {best['objective']}")
        print(f"  Success rate: {best['success_rate']}")
        print(f"  Avg science: {best['avg_science']}")
        print(f"  Avg runtime: {best['avg_runtime']}")
        print(f"  Params: {best['params']}")


def main():
    trials, finalist_summary = load_data()

    planner_summary = summarize_trials(trials)
    planner_summary.to_csv("final_planner_trial_summary.csv", index=False)

    plot_bar(
        planner_summary,
        y="success_rate",
        yerr=None,
        title="Mission Success Rate by Planner",
        ylabel="Success rate",
        filename="final_success_rate_by_planner.png",
    )

    plot_bar(
        planner_summary,
        y="avg_science",
        yerr="science_ci95",
        title="Average Science Collected by Planner",
        ylabel="Average small science collected",
        filename="final_avg_science_ci.png",
    )

    plot_bar(
        planner_summary,
        y="avg_runtime",
        yerr="runtime_ci95",
        title="Runtime by Planner",
        ylabel="Average runtime per trial [s]",
        filename="final_runtime_ci.png",
    )

    plot_science_distribution(trials)
    plot_science_vs_runtime(planner_summary)

    print("\nPlanner summary:")
    print(planner_summary)

    print("\nBest finalist configs:")
    print_best_finalists(finalist_summary)


if __name__ == "__main__":
    main()