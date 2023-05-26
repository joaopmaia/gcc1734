import argparse
import gymnasium as gym
from lql import QLearningAgentLinear
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from gymnasium.wrappers import TimeLimit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per training episode")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="Decay rate")
    parser.add_argument("--learning_rate", type=float, default=0.0007, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.618, help="Gamma")
    args = parser.parse_args()

    num_episodes = args.num_episodes
    max_steps = args.max_steps
    env_name = args.env_name
    decay_rate = args.decay_rate
    learning_rate = args.learning_rate
    gamma = args.gamma

    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=args.max_steps)

    agent = QLearningAgentLinear(env, learning_rate = learning_rate, decay_rate = decay_rate, gamma = gamma)
    penalties_per_episode, rewards_per_episode = agent.train(num_episodes)
    agent.save(args.env_name + "-lql-agent.pkl")

    plt.subplot(1, 2, 1)
    plt.plot(savgol_filter(penalties_per_episode, 111, 2))
    plt.title(f"Penalties ({args.env_name})")
    plt.subplot(1, 2, 2)
    plt.plot(savgol_filter(rewards_per_episode, 111, 2))
    plt.title(f"Rewards ({args.env_name})")
    plt.savefig(args.env_name + "-lql-penalties_and_rewards_per_episode.png")
    plt.close()
