"""
Q-Learning agent for the Taxi-v3 environment (Gymnasium).

Trains the agent, logs cumulative and rolling-average rewards,
and saves training plots to `results/`.
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import trange


# ── Hyperparameters ──────────────────────────────────────────────
NUM_EPISODES = 5_000
MAX_STEPS = 200          # Taxi-v3 truncates at 200 anyway

ALPHA = 0.1              # learning rate
GAMMA = 0.99             # discount factor
EPSILON_START = 1.0      # initial exploration
EPSILON_MIN = 0.01       # minimum exploration
EPSILON_DECAY = 0.9995   # multiplicative decay per episode

SEED = 42
ROLLING_WINDOW = 100     # window for rolling average
# ─────────────────────────────────────────────────────────────────


def train():
    env = gym.make("Taxi-v3")

    n_states = env.observation_space.n   # 500
    n_actions = env.action_space.n       # 6

    q_table = np.zeros((n_states, n_actions))
    epsilon = EPSILON_START

    episode_rewards = []
    cumulative_rewards = []
    rolling_avg_rewards = []
    cumulative = 0.0

    rng = np.random.default_rng(SEED)

    for ep in trange(NUM_EPISODES, desc="Training"):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (
                reward + GAMMA * best_next - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        # Decay exploration
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Logging
        episode_rewards.append(total_reward)
        cumulative += total_reward
        cumulative_rewards.append(cumulative)

        if len(episode_rewards) >= ROLLING_WINDOW:
            avg = np.mean(episode_rewards[-ROLLING_WINDOW:])
        else:
            avg = np.mean(episode_rewards)
        rolling_avg_rewards.append(avg)

    env.close()

    os.makedirs("results", exist_ok=True)
    np.save("results/q_table.npy", q_table)
    print("Q-table saved to results/q_table.npy")

    return q_table, episode_rewards, cumulative_rewards, rolling_avg_rewards


def evaluate(q_table, n_episodes=100):
    env = gym.make("Taxi-v3")
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    env.close()
    return total / n_episodes


def plot_results(episode_rewards, cumulative_rewards, rolling_avg_rewards):
    os.makedirs("results", exist_ok=True)
    episodes = np.arange(1, len(episode_rewards) + 1)

    # 1. Reward per episode
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, episode_rewards, alpha=0.3, linewidth=0.5, label="Episode reward")
    ax.plot(episodes, rolling_avg_rewards, color="red", linewidth=1.5,
            label=f"Rolling avg (last {ROLLING_WINDOW})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Taxi-v3 Q-Learning — Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/reward_per_episode.png", dpi=150)
    plt.close(fig)

    # 2. Cumulative reward
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, cumulative_rewards, color="green", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Taxi-v3 Q-Learning — Cumulative Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/cumulative_reward.png", dpi=150)
    plt.close(fig)

    # 3. Rolling average reward
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rolling_avg_rewards, color="red", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Avg Reward (last {ROLLING_WINDOW} episodes)")
    ax.set_title("Taxi-v3 Q-Learning — Rolling Average Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/rolling_avg_reward.png", dpi=150)
    plt.close(fig)

    print("Plots saved to results/")


def main():
    print("=== Training Q-Learning agent on Taxi-v3 ===")
    print(f"Episodes: {NUM_EPISODES}, α={ALPHA}, γ={GAMMA}, "
          f"ε: {EPSILON_START}→{EPSILON_MIN} (decay={EPSILON_DECAY})")

    q_table, ep_rewards, cum_rewards, roll_avg = train()

    print(f"\nFinal rolling avg reward (last {ROLLING_WINDOW}): "
          f"{roll_avg[-1]:.2f}")

    avg_eval = evaluate(q_table, n_episodes=100)
    print(f"Evaluation (greedy, 100 episodes): avg reward = {avg_eval:.2f}")

    plot_results(ep_rewards, cum_rewards, roll_avg)


if __name__ == "__main__":
    main()
