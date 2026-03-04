import argparse
import logging
import os
import sys

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

Q_TABLE_PATH = "results/q_table.npy"
OUTPUT_DIR = "results/gifs"
GRID_ROWS, GRID_COLS = 5, 5

LOC_COORDS = [(0, 0), (0, 4), (4, 0), (4, 3)]
LOC_NAMES = ["R", "G", "Y", "B"]
LOC_COLORS_MAP = {"R": "red", "G": "limegreen", "Y": "gold", "B": "royalblue"}

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
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return taxi_row, taxi_col, pass_loc, dest


def draw_frame(ax, taxi_row, taxi_col, pass_loc, dest, step, action, reward, total_reward):
    fig = ax.get_figure()
    for legend in fig.legends:
        legend.remove()
    ax.clear()

    for r in range(GRID_ROWS + 1):
        ax.plot([0, GRID_COLS], [r, r], color="black", linewidth=0.5)
    for c in range(GRID_COLS + 1):
        ax.plot([c, c], [0, GRID_ROWS], color="black", linewidth=0.5)

    for (r1, c1), (r2, c2) in WALLS:
        if c2 == c1 + 1:
            ax.plot([c2, c2], [GRID_ROWS - r1 - 1, GRID_ROWS - r1], color="black", linewidth=4)

    for i, ((lr, lc), name) in enumerate(zip(LOC_COORDS, LOC_NAMES)):
        color = LOC_COLORS_MAP[name]
        x, y = lc + 0.5, GRID_ROWS - lr - 0.5
        if i == dest:
            ax.add_patch(plt.Rectangle((lc, GRID_ROWS - lr - 1), 1, 1,
                                        facecolor=color, alpha=0.25))
        ax.text(x, y + 0.3, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    if pass_loc < 4:
        pr, pc = LOC_COORDS[pass_loc]
        px, py = pc + 0.5, GRID_ROWS - pr - 0.7
        ax.plot(px, py, "o", color="magenta", markersize=10, zorder=5)
        ax.text(px + 0.25, py, "P", fontsize=7, color="magenta", fontweight="bold", zorder=5)

    taxi_color = "orange" if pass_loc == 4 else "yellow"
    ax.add_patch(plt.Rectangle((taxi_col + 0.15, GRID_ROWS - taxi_row - 0.85), 0.7, 0.7,
                                facecolor=taxi_color, edgecolor="black", linewidth=2,
                                zorder=4, clip_on=False))
    tx, ty = taxi_col + 0.5, GRID_ROWS - taxi_row - 0.5
    ax.text(tx, ty, "T", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=5)

    action_str = ACTION_NAMES[action] if action is not None else "Start"
    dest_name = LOC_NAMES[dest]

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    title = f"Step {step} | Action: {action_str} | Reward: {reward:+.0f} | Total: {total_reward:+.0f}"
    ax.set_title(title, fontsize=9)

    legend_items = [
        mpatches.Patch(facecolor="yellow", edgecolor="black", label="Taxi (empty)"),
        mpatches.Patch(facecolor="orange", edgecolor="black", label="Taxi (with passenger)"),
        mpatches.Patch(facecolor=LOC_COLORS_MAP[dest_name], alpha=0.25,
                       edgecolor="black", label=f"Destination: {dest_name}"),
    ]
    fig.legend(handles=legend_items, loc="lower center", fontsize=6,
               framealpha=0.8, handlelength=1.5, ncol=3,
               bbox_to_anchor=(0.5, 0.03))


def run_episode(q_table, seed=None):
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
    fig, ax = plt.subplots(figsize=(5, 5.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10)

    def update(i):
        f = frames[i]
        draw_frame(ax, f["taxi_row"], f["taxi_col"], f["pass_loc"], f["dest"],
                   f["step"], f["action"], f["reward"], f["total_reward"])

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, repeat=False)
    anim.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)


def print_ansi_episode(q_table, seed=None, episode_num=1):
    env = gym.make("Taxi-v3", render_mode="ansi")
    state, _ = env.reset(seed=seed)

    logger.info("=" * 50)
    logger.info("  Episode %d  (seed=%s)", episode_num, seed)
    logger.info("=" * 50)
    logger.info("\n%s", env.render())

    total_reward = 0
    done = False
    step = 0

    while not done and step < 200:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        logger.info("Step %d: %s  reward=%+.0f  total=%+.0f",
                     step, ACTION_NAMES[action], reward, total_reward)
        logger.debug("\n%s", env.render())

        done = terminated or truncated

    status = "SUCCESS" if total_reward > 0 else "TIMEOUT"
    logger.info("Result: %s | Steps: %d | Total reward: %+.0f", status, step, total_reward)
    env.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--q-table", type=str, default=Q_TABLE_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.q_table):
        logger.error("Q-table not found at %s", args.q_table)
        logger.error("Run taxi_qlearning.py first to train the agent.")
        sys.exit(1)

    q_table = np.load(args.q_table)
    logger.info("Loaded Q-table from %s (shape: %s)", args.q_table, q_table.shape)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(args.episodes)]

    for i, seed in enumerate(seeds):
        print_ansi_episode(q_table, seed=seed, episode_num=i + 1)

        if not args.no_gif:
            frames = run_episode(q_table, seed=seed)
            gif_path = os.path.join(OUTPUT_DIR, f"episode_{i + 1}.gif")
            save_gif(frames, gif_path, fps=2)
            logger.info("  GIF saved: %s", gif_path)

    logger.info("Done! %d episodes visualized.", args.episodes)
    if not args.no_gif:
        logger.info("GIF animations saved to %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
