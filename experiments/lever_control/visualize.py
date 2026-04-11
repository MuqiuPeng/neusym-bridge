"""Visualizations for the lever control experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("docs/results/lever_control")
RESULTS_PATH = Path("experiments/lever_control/outputs/lever_results.json")


def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def fig_lever_calibration(data: dict):
    """Bar chart: energy cost per lever."""
    cal = data["calibration"]
    levers = list(cal.keys())
    means = [cal[l]["mean"] for l in levers]
    stds = [cal[l]["std"] for l in levers]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(levers)))
    bars = ax.bar(range(len(levers)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="#666666", linewidth=0.8)

    ax.set_xticks(range(len(levers)))
    ax.set_xticklabels(levers, rotation=30, ha="right")
    ax.set_ylabel("Energy (mean +/- std)")
    ax.set_title("Lever Energy Calibration (20 random initial states)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_lever_calibration.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_planner_comparison(data: dict):
    """Side-by-side: success rate and energy."""
    planning = data["planning"]
    names = list(planning.keys())
    display = {"relatum_minE": "Relatum\n(min-energy)", "greedy": "Greedy", "random": "Random"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"relatum_minE": "#2196F3", "greedy": "#F44336", "random": "#9E9E9E"}
    x = np.arange(len(names))

    # Success rate
    sr = [planning[n]["success_rate"] for n in names]
    ax1.bar(x, sr, color=[colors[n] for n in names], edgecolor="#666", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([display.get(n, n) for n in names])
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    # Energy (solved tasks only)
    energy = [planning[n]["avg_energy"] for n in names]
    ax2.bar(x, energy, color=[colors[n] for n in names], edgecolor="#666", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([display.get(n, n) for n in names])
    ax2.set_ylabel("Avg Energy (solved tasks)")
    ax2.set_title("Energy Consumption")

    plt.suptitle("Lever Control: Planner Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_planner_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_step_breakdown(data: dict):
    """Stacked bar: explore vs execute steps for Relatum planner."""
    planning = data["planning"]
    names = list(planning.keys())
    display = {"relatum_minE": "Relatum\n(min-energy)", "greedy": "Greedy", "random": "Random"}
    colors_explore = "#FF9800"
    colors_execute = "#2196F3"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))

    explore = [planning[n].get("avg_explore_steps", 0) for n in names]
    execute = [planning[n].get("avg_execute_steps", 0) for n in names]

    ax.bar(x, explore, 0.5, label="Explore", color=colors_explore)
    ax.bar(x, execute, 0.5, bottom=explore, label="Execute", color=colors_execute)

    ax.set_xticks(x)
    ax.set_xticklabels([display.get(n, n) for n in names])
    ax.set_ylabel("Avg Steps (solved tasks)")
    ax.set_title("Step Breakdown: Exploration vs Execution", fontsize=12)
    ax.legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_step_breakdown.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    print("Generating lever control visualizations...\n")
    fig_lever_calibration(data)
    fig_planner_comparison(data)
    fig_step_breakdown(data)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
