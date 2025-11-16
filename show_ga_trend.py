import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# Load generation log
# -------------------------------------------------------------------
def load_generation_log(file_path="generation_log.csv"):
    path = Path(file_path)
    if not path.exists():
        print(f"No {file_path} found. Run GA first.")
        return []

    data = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gen = int(row["generation"])
                phase = row["phase"]
                pose = float(row.get("pose_score", 0))
                gait = float(row.get("gait_score", 0))
                best = float(row.get("best_fitness", 0))
                w_pose = float(row.get("w_pose", 0))
                w_gait = float(row.get("w_gait", 0))
                data.append((gen, phase, pose, gait, best, w_pose, w_gait))
            except (KeyError, ValueError):
                continue
    return data

# -------------------------------------------------------------------
# Print formatted summary
# -------------------------------------------------------------------
def show_summary(data):
    if not data:
        print(" No generation data found in generation_log.csv.")
        return

    print("Generation Fitness Summary :")
    print("-" * 125)
    for gen, phase, pose, gait, best, w_pose, w_gait in data:
        print(f"GEN {gen:02d}|Phase: {phase:<18}|"
              f"(Pose.W={w_pose:.2f},Gait.W={w_gait:.2f})| "
              f"Pose={pose:+.3f}|Gait={gait:+.3f}|Best Fitness={best:+.3f}")
    print("-" * 125)

    best_gen = max(data, key=lambda x: x[4])
    print(f" Best Generation = {int(best_gen[0]):02d} | Best Fitness = {best_gen[4]:.3f}")

# -------------------------------------------------------------------
# Plot fitness trend (phase-colored, legend bottom-right)
# -------------------------------------------------------------------
def plot_trend(data):
    if not data:
        print(" No data to plot.")
        return

    gens = [row[0] for row in data]
    bests = [row[4] for row in data]
    phases = [row[1] for row in data]

    # Define consistent colors per phase
    phase_colors = {
        "Posture Training": "tab:blue",
        "Balance & Motion": "tab:orange",
        "Gait Optimization": "tab:green"
    }

    plt.figure(figsize=(9, 5))

    # Plot overall trend line
    plt.plot(gens, bests, color="black", linewidth=2, label="Best Fitness")

    # Plot scatter points by phase
    for gen, best, phase in zip(gens, bests, phases):
        plt.scatter(gen, best, color=phase_colors.get(phase, "gray"),
                    s=70, edgecolor="k", linewidth=0.5)

    # Create legend handles for phases
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=phase,
                   markerfacecolor=color, markersize=10, markeredgecolor='k')
        for phase, color in phase_colors.items()
    ]

    # Move legend to bottom-right corner
    plt.legend(handles=handles, title="Phases",
               loc="lower right", frameon=True, fancybox=True, shadow=False)

    # Plot styling
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Spider GA Fitness Trend by Phase")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    data = load_generation_log()
    if not data:
        return
    show_summary(data)
    plot_trend(data)

if __name__ == "__main__":
    main()
