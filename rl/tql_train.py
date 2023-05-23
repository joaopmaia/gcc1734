from timeit import default_timer as timer
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
from tql import QLearningAgentTabular

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--filename", type=str, help="Path to save the agent after training")
  parser.add_argument("--num_episodes", type=int, default=60000, help="Number of episodes")
  parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
  args = parser.parse_args()

  num_episodes = args.num_episodes
  env_name = args.env_name

  env = gym.make(env_name).env
  agent = QLearningAgentTabular(env)
  rewards = agent.train(num_episodes)

  plt.plot(savgol_filter(rewards, 1001, 2))
  plt.title("Curva de aprendizado suavizada")
  plt.xlabel('Episódio')
  plt.ylabel('Recompensa total')
  plt.savefig("tql_learning_curve.png")
  plt.close()

  plt.plot(agent.epsilons_)
  plt.title("Decaimento do valor de $\epsilon$")
  plt.xlabel('Episódio')
  plt.ylabel('$\epsilon$')
  plt.savefig("tql_epsilons.png")
  plt.close()

  agent.save(args.filename)