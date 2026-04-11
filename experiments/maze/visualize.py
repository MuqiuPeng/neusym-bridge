"""Generate publication-ready visualizations for the maze experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from experiments.maze.env import (
    MazeState, make_maze, bfs_optimal_path, bfs_reachable, cell_id,
)
from experiments.maze.planner import RelatumMazePlanner, GreedyMazePlanner, BFSMazePlanner


OUTPUT_DIR = Path("docs/results/maze")


# ── Fig 1: Example maze with paths ──────────────────────────────────


def fig_example_maze(seed: int = 5, size: int = 8):
    """Side-by-side: Relatum path vs Greedy path vs BFS optimal on one maze."""
    maze = make_maze(size, size, seed=seed, wall_density=0.2)

    planners = {
        "BFS (optimal)": BFSMazePlanner(),
        "Relatum": RelatumMazePlanner(),
        "Greedy": GreedyMazePlanner(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (name, planner) in zip(axes, planners.items()):
        traj, _ = planner.plan(maze, max_steps=200)
        path = [s.agent for s in traj]
        solved = traj[-1].is_solved()

        _draw_maze(ax, maze, path, title=name, solved=solved)

    fig.suptitle(
        f"Maze Planning Comparison ({size}x{size}, seed={seed})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_example_maze.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _draw_maze(
    ax, maze: MazeState, path: list[tuple[int, int]],
    title: str = "", solved: bool = True,
):
    """Draw a maze grid with path overlay."""
    w, h = maze.width, maze.height
    grid = np.zeros((h, w))

    for wx, wy in maze.walls:
        grid[wy, wx] = 1  # wall

    cmap = ListedColormap(["#f0f0f0", "#333333"])
    ax.imshow(grid, cmap=cmap, origin="upper", extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    # Grid lines
    for x in range(w + 1):
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color="#cccccc", linewidth=0.5)

    # Path
    if len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        color = "#2196F3" if solved else "#F44336"
        ax.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=0.7, zorder=2)
        ax.plot(xs, ys, "o", color=color, markersize=4, alpha=0.5, zorder=2)

    # Start and goal markers
    sx, sy = maze.agent
    gx, gy = maze.goal
    ax.plot(sx, sy, "s", color="#4CAF50", markersize=14, zorder=3, label="Start")
    ax.plot(gx, gy, "*", color="#FF9800", markersize=16, zorder=3, label="Goal")
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)

    steps = len(path) - 1
    status = f"{steps} steps" if solved else "FAILED"
    ax.set_title(f"{title}\n({status})", fontsize=11)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ── Fig 2: Success rate comparison bar chart ─────────────────────────


def fig_success_comparison():
    """Bar chart: success rate and optimality across planners."""
    sizes = [6, 8, 10]
    n_per_size = 50
    planners = {
        "Relatum": RelatumMazePlanner(),
        "Greedy": GreedyMazePlanner(),
        "BFS": BFSMazePlanner(),
    }

    data = {name: {s: {"success": 0, "steps": [], "opt_steps": []} for s in sizes}
            for name in planners}

    rng = np.random.RandomState(99)
    for size in sizes:
        for i in range(n_per_size):
            maze = make_maze(size, size, seed=1000 + size * 100 + i, wall_density=0.2)
            opt_path = bfs_optimal_path(maze)
            opt_len = len(opt_path) - 1 if opt_path else 0

            for name, planner in planners.items():
                traj, _ = planner.plan(maze, max_steps=size * size * 2)
                if traj[-1].is_solved():
                    data[name][size]["success"] += 1
                    data[name][size]["steps"].append(len(traj) - 1)
                    data[name][size]["opt_steps"].append(opt_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    x = np.arange(len(sizes))
    bar_w = 0.25
    colors = {"Relatum": "#2196F3", "Greedy": "#F44336", "BFS": "#4CAF50"}

    for i, name in enumerate(["Relatum", "Greedy", "BFS"]):
        rates = [data[name][s]["success"] / n_per_size for s in sizes]
        ax1.bar(x + i * bar_w, rates, bar_w, label=name, color=colors[name])

    ax1.set_xlabel("Maze Size")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate by Maze Size")
    ax1.set_xticks(x + bar_w)
    ax1.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    # Optimality (steps / optimal_steps)
    for i, name in enumerate(["Relatum", "Greedy"]):
        opts = []
        for s in sizes:
            steps_list = data[name][s]["steps"]
            opt_list = data[name][s]["opt_steps"]
            if steps_list:
                ratios = [o / st if st > 0 else 0 for o, st in zip(opt_list, steps_list)]
                opts.append(np.mean(ratios))
            else:
                opts.append(0)
        ax2.bar(x + i * bar_w, opts, bar_w, label=name, color=colors[name])

    ax2.set_xlabel("Maze Size")
    ax2.set_ylabel("Optimality (optimal / actual)")
    ax2.set_title("Path Optimality (solved mazes only)")
    ax2.set_xticks(x + bar_w / 2)
    ax2.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    plt.suptitle("Maze Planning: Relatum vs Greedy vs BFS Optimal\n(50 mazes per size)", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_success_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Fig 3: Collapse scenario illustration ────────────────────────────


def fig_collapse_scenarios():
    """Three-panel illustration of collapse Scenario A/B/C."""
    maze = make_maze(6, 6, seed=0)
    opt_path = bfs_optimal_path(maze)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scenario A: Normal — all reachable, goal reached
    reachable = bfs_reachable(maze)
    ax = axes[0]
    _draw_maze_with_reachability(ax, maze, reachable, opt_path)
    ax.set_title("Scenario A: Normal Collapse\nsolved = COLLAPSED", fontsize=11,
                 color="#4CAF50", fontweight="bold")

    # Scenario B: Block critical cell
    if opt_path and len(opt_path) > 2:
        block = opt_path[1]
        from experiments.maze.env import MazeState
        maze_b = MazeState(
            width=maze.width, height=maze.height,
            agent=maze.agent, goal=maze.goal,
            walls=maze.walls | {block},
        )
        from experiments.maze.env import bfs_solvable
        # Find a blocking cell that actually makes it unsolvable
        for bp in opt_path[1:-1]:
            maze_test = MazeState(
                width=maze.width, height=maze.height,
                agent=maze.agent, goal=maze.goal,
                walls=maze.walls | {bp},
            )
            if not bfs_solvable(maze_test):
                maze_b = maze_test
                block = bp
                break

        reachable_b = bfs_reachable(maze_b)
        ax = axes[1]
        _draw_maze_with_reachability(ax, maze_b, reachable_b, None)
        bx, by = block
        ax.add_patch(patches.Rectangle(
            (bx - 0.4, by - 0.4), 0.8, 0.8,
            linewidth=3, edgecolor="red", facecolor="red", alpha=0.5, zorder=5,
        ))
        ax.text(bx, by, "X", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white", zorder=6)
        ax.set_title("Scenario B: Retraction\nsolved = RETRACTED", fontsize=11,
                     color="#F44336", fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "N/A", transform=axes[1].transAxes, ha="center")

    # Scenario C: Partial info — only agent + goal, no reachability
    ax = axes[2]
    _draw_maze_partial(ax, maze)
    ax.set_title("Scenario C: Active Query\nsolved = PENDING (needs reachable)", fontsize=11,
                 color="#FF9800", fontweight="bold")

    plt.suptitle("Collapse Mechanism in Maze Domain", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_collapse_scenarios.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _draw_maze_with_reachability(ax, maze, reachable, path):
    """Draw maze with reachable cells highlighted."""
    w, h = maze.width, maze.height
    grid = np.zeros((h, w))
    for wx, wy in maze.walls:
        grid[wy, wx] = 2
    for rx, ry in reachable:
        if (rx, ry) not in maze.walls:
            grid[ry, rx] = 1

    cmap = ListedColormap(["#f0f0f0", "#C8E6C9", "#333333"])
    ax.imshow(grid, cmap=cmap, origin="upper", extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    for x in range(w + 1):
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color="#cccccc", linewidth=0.5)

    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, "-", color="#2196F3", linewidth=2, alpha=0.7, zorder=2)

    sx, sy = maze.agent
    gx, gy = maze.goal
    ax.plot(sx, sy, "s", color="#4CAF50", markersize=12, zorder=3)
    ax.plot(gx, gy, "*", color="#FF9800", markersize=14, zorder=3)
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=4)
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=4)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_maze_partial(ax, maze):
    """Draw maze showing only agent and goal, rest grayed out (unknown)."""
    w, h = maze.width, maze.height
    grid = np.ones((h, w)) * 0.5  # all gray (unknown)

    for wx, wy in maze.walls:
        grid[wy, wx] = 1.0

    cmap = ListedColormap(["#f0f0f0", "#E0E0E0", "#333333"])
    # Map: 0=white, 0.5=gray, 1.0=wall
    grid_mapped = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if (x, y) in maze.walls:
                grid_mapped[y, x] = 2
            else:
                grid_mapped[y, x] = 1  # unknown

    # Agent and goal cells are known
    ax_x, ax_y = maze.agent
    gx, gy = maze.goal
    grid_mapped[ax_y, ax_x] = 0
    grid_mapped[gy, gx] = 0

    ax.imshow(grid_mapped, cmap=cmap, origin="upper", extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    for x in range(w + 1):
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color="#cccccc", linewidth=0.5)

    ax.plot(ax_x, ax_y, "s", color="#4CAF50", markersize=12, zorder=3)
    ax.plot(gx, gy, "*", color="#FF9800", markersize=14, zorder=3)
    ax.text(ax_x, ax_y, "S", ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=4)
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=4)

    # Question marks on unknown cells
    for y in range(h):
        for x in range(w):
            if grid_mapped[y, x] == 1 and (x, y) != maze.agent and (x, y) != maze.goal:
                ax.text(x, y, "?", ha="center", va="center", fontsize=6, color="#999999")

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ── Fig 4: Architecture comparison diagram ───────────────────────────


def fig_architecture_comparison():
    """Text-based architecture diagram: Maze vs Tentacle pipeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.axis("off")

    # Maze pipeline
    boxes_maze = [
        (0.2, 0.5, "Maze State\n(discrete)", "#E3F2FD"),
        (2.5, 0.5, "Direct Map\n(conf=1.0)", "#C8E6C9"),
        (4.8, 0.5, "Relatum\nFacts", "#FFF9C4"),
        (7.1, 0.5, "Collapse\n(deterministic)", "#FFECB3"),
        (9.0, 0.5, "solved\n= True", "#A5D6A7"),
    ]
    _draw_pipeline(ax1, boxes_maze, "Maze Pipeline (zero noise)")

    # Tentacle pipeline
    boxes_tent = [
        (0.2, 0.5, "Tentacle\n(140-dim)", "#E3F2FD"),
        (2.5, 0.5, "Neural Net\n(ECE=0.021)", "#FFCDD2"),
        (4.8, 0.5, "Relatum\nFacts", "#FFF9C4"),
        (7.1, 0.5, "Noisy-OR\nCollapse", "#FFECB3"),
        (9.0, 0.5, "risk\n= prob", "#EF9A9A"),
    ]
    _draw_pipeline(ax2, boxes_tent, "Tentacle Pipeline (neural interface)")

    plt.suptitle("Architecture Comparison: Zero-Noise vs Neural Interface",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_architecture_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _draw_pipeline(ax, boxes, title):
    """Draw a sequence of boxes with arrows."""
    ax.text(5, 1.8, title, ha="center", va="center", fontsize=12, fontweight="bold")

    for i, (x, y, text, color) in enumerate(boxes):
        rect = patches.FancyBboxPatch(
            (x, y), 1.8, 0.9, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#666666", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.45, text, ha="center", va="center", fontsize=8)

        if i < len(boxes) - 1:
            next_x = boxes[i + 1][0]
            ax.annotate(
                "", xy=(next_x, y + 0.45), xytext=(x + 1.8, y + 0.45),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5),
            )


# ── Main ─────────────────────────────────────────────────────────────


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating maze visualizations...\n")
    fig_example_maze()
    fig_success_comparison()
    fig_collapse_scenarios()
    fig_architecture_comparison()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
