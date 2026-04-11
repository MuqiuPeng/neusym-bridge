"""Visualizations for the exploration-based maze experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from experiments.maze.env import make_maze, bfs_optimal_path, DIRECTIONS
from experiments.maze.planner import (
    RelatumExplorationPlanner, GreedyPlanner, BFSOraclePlanner,
)
from experiments.maze.interface import ExplorationKB

OUTPUT_DIR = Path("docs/results/maze")


# ── Fig 1: Exploration trace ─────────────────────────────────────────


def fig_exploration_trace(seed: int = 5, size: int = 8):
    """Show the Relatum agent's exploration and execution phases."""
    maze = make_maze(size, size, seed=seed, wall_density=0.2)
    planner = RelatumExplorationPlanner()
    traj, stats = planner.plan(maze, max_steps=500)

    explore_steps = stats.get("explore_steps", 0)
    explore_path = [s.agent for s in traj[:explore_steps + 1]]
    execute_path = [s.agent for s in traj[explore_steps:]]

    # BFS optimal for reference
    bfs_path = bfs_optimal_path(maze)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Exploration phase
    _draw_maze(axes[0], maze, explore_path, color="#FF9800", alpha=0.5)
    axes[0].set_title(
        f"Phase 1: Exploration\n({explore_steps} steps, "
        f"{stats.get('cells_visited', 0)} cells discovered)",
        fontsize=11,
    )

    # Panel 2: Execution phase
    _draw_maze(axes[1], maze, execute_path, color="#2196F3")
    axes[1].set_title(
        f"Phase 2: Execution\n({stats.get('execute_steps', 0)} steps, "
        f"after Relatum collapse)",
        fontsize=11,
    )

    # Panel 3: BFS optimal
    _draw_maze(axes[2], maze, bfs_path or [], color="#4CAF50")
    bfs_steps = len(bfs_path) - 1 if bfs_path else 0
    axes[2].set_title(f"BFS Optimal\n({bfs_steps} steps, full map knowledge)", fontsize=11)

    solved = traj[-1].is_solved()
    fig.suptitle(
        f"Rule Learning Through Exploration ({size}x{size})\n"
        f"Total: {stats.get('total_steps', 0)} steps "
        f"(explore {explore_steps} + execute {stats.get('execute_steps', 0)}) "
        f"{'SOLVED' if solved else 'FAILED'}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_exploration_trace.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Fig 2: Success rate and exploration overhead ─────────────────────


def fig_success_and_overhead():
    """Bar chart comparing planners across maze sizes."""
    sizes = [6, 8, 10]
    n_per_size = 50

    planners = {
        "Relatum (explore)": RelatumExplorationPlanner(),
        "Greedy": GreedyPlanner(),
        "BFS Oracle": BFSOraclePlanner(),
    }

    data = {name: {s: {"success": 0, "steps": [], "explore": [], "opt": []}
            for s in sizes} for name in planners}

    for size in sizes:
        for i in range(n_per_size):
            maze = make_maze(size, size, seed=2000 + size * 100 + i, wall_density=0.2)
            bfs_path = bfs_optimal_path(maze)
            opt_len = len(bfs_path) - 1 if bfs_path else 0

            for name, planner in planners.items():
                traj, stats = planner.plan(maze, max_steps=size * size * 4)
                if traj[-1].is_solved():
                    data[name][size]["success"] += 1
                    data[name][size]["steps"].append(len(traj) - 1)
                    data[name][size]["explore"].append(stats.get("explore_steps", 0))
                    data[name][size]["opt"].append(opt_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"Relatum (explore)": "#2196F3", "Greedy": "#F44336", "BFS Oracle": "#4CAF50"}

    # Success rate
    x = np.arange(len(sizes))
    bar_w = 0.25
    for i, name in enumerate(planners):
        rates = [data[name][s]["success"] / n_per_size for s in sizes]
        ax1.bar(x + i * bar_w, rates, bar_w, label=name, color=colors[name])

    ax1.set_xlabel("Maze Size")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate")
    ax1.set_xticks(x + bar_w)
    ax1.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=9)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    # Step breakdown for Relatum (explore vs execute)
    rel_data = data["Relatum (explore)"]
    explore_means = [np.mean(rel_data[s]["explore"]) if rel_data[s]["explore"] else 0
                     for s in sizes]
    execute_means = []
    for s in sizes:
        if rel_data[s]["steps"] and rel_data[s]["explore"]:
            exe = [t - e for t, e in zip(rel_data[s]["steps"], rel_data[s]["explore"])]
            execute_means.append(np.mean(exe))
        else:
            execute_means.append(0)
    opt_means = [np.mean(rel_data[s]["opt"]) if rel_data[s]["opt"] else 0 for s in sizes]

    ax2.bar(x, explore_means, 0.35, label="Explore", color="#FF9800")
    ax2.bar(x, execute_means, 0.35, bottom=explore_means, label="Execute", color="#2196F3")
    ax2.plot(x, opt_means, "k*", markersize=12, label="BFS Optimal", zorder=5)

    ax2.set_xlabel("Maze Size")
    ax2.set_ylabel("Steps")
    ax2.set_title("Relatum: Explore + Execute Breakdown")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax2.legend(fontsize=9)

    plt.suptitle("Maze Planning: Rule Learning vs Baselines\n(50 mazes per size)", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_success_and_overhead.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Fig 3: Knowledge growth over time ────────────────────────────────


def fig_knowledge_growth(seed: int = 3, size: int = 10):
    """Show how the learned knowledge base grows during exploration."""
    maze = make_maze(size, size, seed=seed, wall_density=0.15)

    # Run exploration step by step, record KB state
    state = maze
    kb = ExplorationKB()
    kb.visited.add(state.agent)
    kb.observe_goal(state.goal)

    history = {"step": [0], "cells": [1], "edges": [0], "goal_reachable": [False]}
    planner = RelatumExplorationPlanner()

    # Manually step through exploration
    from experiments.maze.interface import make_maze_relatum, cell_id
    for step in range(size * size * 3):
        if state.is_solved():
            break

        # Check goal reachability
        reachable = kb.reachable_from(state.agent)
        goal_reachable = maze.goal in reachable

        if goal_reachable:
            # Execute phase — just follow path
            path = kb.path_to(state.agent, maze.goal)
            if path:
                for i in range(1, len(path)):
                    act = kb.action_for_edge(path[i-1], path[i])
                    if act:
                        state = state.move(act)
                        history["step"].append(step + i)
                        history["cells"].append(kb.n_visited)
                        history["edges"].append(kb.n_edges)
                        history["goal_reachable"].append(True)
            break

        # Explore
        action = planner._pick_explore_action(state, kb)
        if action is None:
            break
        new_state = state.move(action)
        kb.observe_transition(state.agent, action, new_state.agent)
        state = new_state

        history["step"].append(step + 1)
        history["cells"].append(kb.n_visited)
        history["edges"].append(kb.n_edges)
        history["goal_reachable"].append(False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    steps = history["step"]
    ax1.plot(steps, history["cells"], "-o", color="#2196F3", markersize=3, label="Cells visited")
    ax1.plot(steps, history["edges"], "-s", color="#FF9800", markersize=3, label="Edges learned")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Knowledge Growth During Exploration ({size}x{size})", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Mark when goal becomes reachable
    goal_step = None
    for i, gr in enumerate(history["goal_reachable"]):
        if gr:
            goal_step = steps[i]
            break

    if goal_step is not None:
        for ax in (ax1, ax2):
            ax.axvline(goal_step, color="green", linestyle="--", linewidth=2, alpha=0.7)
        ax1.text(goal_step + 0.5, ax1.get_ylim()[1] * 0.9,
                 "Goal reachable\n(Relatum collapses)", fontsize=9, color="green")

    # Reachability over time
    reachable_counts = []
    for i in range(len(steps)):
        # Approximate: cells at that step
        reachable_counts.append(history["cells"][i])

    ax2.fill_between(steps, 0, reachable_counts, alpha=0.3, color="#2196F3")
    ax2.plot(steps, reachable_counts, color="#2196F3", linewidth=1.5)
    ax2.set_xlabel("Exploration Step")
    ax2.set_ylabel("Known cells")
    ax2.set_title("Exploration Coverage", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_knowledge_growth.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Fig 4: Architecture diagram ──────────────────────────────────────


def fig_architecture():
    """Diagram: explore -> learn rules -> Relatum collapse -> execute."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, "Agent\nActions\n(U/D/L/R)", "#E3F2FD"),
        (3.0, 1.0, "Observe\nTransition\n(pos -> pos')", "#C8E6C9"),
        (5.7, 1.0, "Learn Rule\nadj(A,B)\nblocked(A,D)", "#FFF9C4"),
        (8.4, 1.0, "Relatum\nCollapse\nsolved?", "#FFECB3"),
        (11.1, 1.0, "Execute\nShortest\nPath", "#A5D6A7"),
    ]

    for i, (x, y, text, color) in enumerate(boxes):
        rect = patches.FancyBboxPatch(
            (x, y), 2.2, 1.2, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#666666", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 1.1, y + 0.6, text, ha="center", va="center", fontsize=9)

        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            ax.annotate(
                "", xy=(nx, y + 0.6), xytext=(x + 2.2, y + 0.6),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=2),
            )

    # Feedback loop from "Relatum" back to "Agent"
    ax.annotate(
        "", xy=(1.4, 0.95), xytext=(9.5, 0.95),
        arrowprops=dict(
            arrowstyle="->", color="#F44336", lw=1.5,
            connectionstyle="arc3,rad=0.4",
        ),
    )
    ax.text(5.5, 0.3, "Not yet solved -> keep exploring", fontsize=9,
            ha="center", color="#F44336", fontstyle="italic")

    ax.set_title(
        "Maze Rule Learning Pipeline: Explore -> Learn -> Reason -> Execute",
        fontsize=13, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_architecture.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Helpers ──────────────────────────────────────────────────────────


def _draw_maze(ax, maze, path, color="#2196F3", alpha=0.7):
    w, h = maze.width, maze.height
    grid = np.zeros((h, w), dtype=int)
    for wx, wy in maze.walls:
        grid[wy, wx] = 1

    cmap = ListedColormap(["#f0f0f0", "#333333"])
    ax.imshow(grid, cmap=cmap, origin="upper",
              extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    for x in range(w + 1):
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color="#cccccc", linewidth=0.5)

    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=alpha, zorder=2)
        ax.plot(xs, ys, "o", color=color, markersize=3, alpha=alpha * 0.7, zorder=2)

    sx, sy = maze.agent
    gx, gy = maze.goal
    ax.plot(sx, sy, "s", color="#4CAF50", markersize=12, zorder=3)
    ax.plot(gx, gy, "*", color="#FF9800", markersize=14, zorder=3)
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white", zorder=4)
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white", zorder=4)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating maze visualizations...\n")
    fig_exploration_trace()
    fig_success_and_overhead()
    fig_knowledge_growth()
    fig_architecture()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
