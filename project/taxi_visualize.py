"""
Visualize trained Q-Learning agent on Taxi-v3.

Renders episodes as GIF animations and prints ANSI step-by-step traces.
Loads the Q-table from results/q_table.npy (produced by taxi_qlearning.py).

Usage:
    uv run python taxi_visualize.py              # 5 GIFs + console output
    uv run python taxi_visualize.py --episodes 3 # custom number of episodes
    uv run python taxi_visualize.py --seed 123   # reproducible episodes
"""

import argparse
import os
import sys

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")

Q_TABLE_PATH = "results/q_table.npy"
OUTPUT_DIR = "results/videos"

# Taxi-v3 grid layout (5x5 with walls)
GRID_ROWS, GRID_COLS = 5, 5
# Named locations: R, G, Y, B
LOC_COORDS = [(0, 0), (0, 4), (4, 0), (4, 3)]
LOC_NAMES = ["R", "G", "Y", "B"]
LOC_COLORS_MAP = {"R": "red", "G": "limegreen", "Y": "gold", "B": "royalblue"}

# Walls: list of ((row1, col1), (row2, col2)) pairs between which a wall exists
WALLS = [
    ((0, 1), (0, 2)),
    ((1, 1), (1, 2)),
    ((3, 0), (3, 1)),
    ((4, 0), (4, 1)),
    ((3, 2), (3, 3)),
    ((4, 2), (4, 3)),
]

ACTION_NAMES = ["South", "North", "East", "West", "Pickup", "Dropoff"]


def decode_state(state):
    """Decode Taxi-v3 state int into (taxi_row, taxi_col, pass_loc, dest)."""
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return taxi_row, taxi_col, pass_loc, dest


def draw_frame(ax, taxi_row, taxi_col, pass_loc, dest, step, action, reward, total_reward):
    """Draw a single frame of the Taxi grid."""
    fig = ax.get_figure()
    # Remove previous figure-level legends to avoid stacking
    for legend in fig.legends:
        legend.remove()
    ax.clear()

    # Draw grid (explicit segments so lines stay within 5x5)
    for r in range(GRID_ROWS + 1):
        ax.plot([0, GRID_COLS], [r, r], color="black", linewidth=0.5)
    for c in range(GRID_COLS + 1):
        ax.plot([c, c], [0, GRID_ROWS], color="black", linewidth=0.5)

    # Draw walls (thick lines between cells)
    for (r1, c1), (r2, c2) in WALLS:
        if c2 == c1 + 1:  # vertical wall on the right side of (r1, c1)
            ax.plot([c2, c2], [GRID_ROWS - r1 - 1, GRID_ROWS - r1], color="black", linewidth=4)

    # Draw location labels
    for i, ((lr, lc), name) in enumerate(zip(LOC_COORDS, LOC_NAMES)):
        color = LOC_COLORS_MAP[name]
        x, y = lc + 0.5, GRID_ROWS - lr - 0.5
        # Highlight destination
        if i == dest:
            ax.add_patch(plt.Rectangle((lc, GRID_ROWS - lr - 1), 1, 1,
                                        facecolor=color, alpha=0.25))
        ax.text(x, y + 0.3, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    # Draw passenger waiting (if not in taxi)
    if pass_loc < 4:
        pr, pc = LOC_COORDS[pass_loc]
        px, py = pc + 0.5, GRID_ROWS - pr - 0.7
        ax.plot(px, py, "o", color="magenta", markersize=10, zorder=5)
        ax.text(px + 0.25, py, "P", fontsize=7, color="magenta", fontweight="bold", zorder=5)

    # Draw taxi
    tx, ty = taxi_col + 0.5, GRID_ROWS - taxi_row - 0.5
    taxi_color = "orange" if pass_loc == 4 else "yellow"
    ax.add_patch(plt.Rectangle((taxi_col + 0.15, GRID_ROWS - taxi_row - 0.85), 0.7, 0.7,
                                facecolor=taxi_color, edgecolor="black", linewidth=2,
                                zorder=4, clip_on=False))
    ax.text(tx, ty, "T", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=5)

    # Info text
    action_str = ACTION_NAMES[action] if action is not None else "Start"
    status = "In taxi" if pass_loc == 4 else f"At {LOC_NAMES[pass_loc]}" if pass_loc < 4 else "Delivered"
    dest_name = LOC_NAMES[dest]

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    title = f"Step {step} | Action: {action_str} | Reward: {reward:+.0f} | Total: {total_reward:+.0f}"
    ax.set_title(title, fontsize=9)

    # Legend below the grid (placed on the figure, outside axes)
    legend_items = [
        mpatches.Patch(facecolor="yellow", edgecolor="black", label="Taxi (empty)"),
        mpatches.Patch(facecolor="orange", edgecolor="black", label="Taxi (with passenger)"),
        mpatches.Patch(facecolor=LOC_COLORS_MAP[dest_name], alpha=0.25,
                       edgecolor="black", label=f"Destination: {dest_name}"),
    ]
    ax.get_figure().legend(handles=legend_items, loc="lower center", fontsize=6,
                           framealpha=0.8, handlelength=1.5, ncol=3,
                           bbox_to_anchor=(0.5, 0.10))


def run_episode(q_table, seed=None):
    """Run one greedy episode and collect frame data."""
    env = gym.make("Taxi-v3")
    state, _ = env.reset(seed=seed)
    taxi_row, taxi_col, pass_loc, dest = decode_state(state)

    frames = [{
        "taxi_row": taxi_row, "taxi_col": taxi_col,
        "pass_loc": pass_loc, "dest": dest,
        "step": 0, "action": None, "reward": 0, "total_reward": 0,
    }]

    total_reward = 0
    done = False
    step = 0

    while not done and step < 200:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        frames.append({
            "taxi_row": taxi_row, "taxi_col": taxi_col,
            "pass_loc": pass_loc, "dest": dest,
            "step": step, "action": action, "reward": reward,
            "total_reward": total_reward,
        })

        done = terminated or truncated

    env.close()
    return frames


def save_gif(frames, filepath, fps=2):
    """Save list of frame dicts as a GIF animation."""
    fig, ax = plt.subplots(figsize=(5, 6))
    fig.subplots_adjust(bottom=0.12)

    def update(i):
        f = frames[i]
        draw_frame(ax, f["taxi_row"], f["taxi_col"], f["pass_loc"], f["dest"],
                   f["step"], f["action"], f["reward"], f["total_reward"])

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, repeat=False)
    anim.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)


def print_ansi_episode(q_table, seed=None, episode_num=1):
    """Print an episode step-by-step using ANSI rendering."""
    env = gym.make("Taxi-v3", render_mode="ansi")
    state, _ = env.reset(seed=seed)

    print(f"\n{'='*50}")
    print(f"  Episode {episode_num}  (seed={seed})")
    print(f"{'='*50}")
    print(env.render())

    total_reward = 0
    done = False
    step = 0

    while not done and step < 200:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step}: {ACTION_NAMES[action]}  reward={reward:+.0f}  total={total_reward:+.0f}")
        print(env.render())

        done = terminated or truncated

    status = "SUCCESS" if total_reward > 0 else "TIMEOUT"
    print(f"Result: {status} | Steps: {step} | Total reward: {total_reward:+.0f}\n")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Taxi-v3 Q-Learning agent")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation, console only")
    parser.add_argument("--q-table", type=str, default=Q_TABLE_PATH, help="Path to Q-table .npy file")
    args = parser.parse_args()

    if not os.path.exists(args.q_table):
        print(f"Error: Q-table not found at {args.q_table}")
        print("Run taxi_qlearning.py first to train the agent.")
        sys.exit(1)

    q_table = np.load(args.q_table)
    print(f"Loaded Q-table from {args.q_table} (shape: {q_table.shape})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(args.episodes)]

    for i, seed in enumerate(seeds):
        # Console output
        print_ansi_episode(q_table, seed=seed, episode_num=i + 1)

        # GIF
        if not args.no_gif:
            frames = run_episode(q_table, seed=seed)
            gif_path = os.path.join(OUTPUT_DIR, f"episode_{i + 1}.gif")
            save_gif(frames, gif_path, fps=2)
            print(f"  GIF saved: {gif_path}")

    print(f"\nDone! {args.episodes} episodes visualized.")
    if not args.no_gif:
        print(f"GIF animations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
