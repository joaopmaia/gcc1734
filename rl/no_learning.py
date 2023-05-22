import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

env = gym.make('Taxi-v3', render_mode="human").env

num_episodes = 100 # valor arbitrário, apenas para ilustração

rewards_per_episode = []

for episode in range(0, num_episodes):
  
  total_reward_in_episode = 0

  # Define estado inicial
  state = env.reset()

  n_actions = 0

  terminated = truncated = False

  # Laço sobre as transicões experimentadas pelo 
  # agente no episódio de treinamento atual.
  while not (terminated or truncated):
    action = env.action_space.sample() # Seleciona ação a executar aleatoriamente

    state, reward, terminated, truncated, info = env.step(action)

    total_reward_in_episode = total_reward_in_episode + reward
    n_actions = n_actions + 1

    ###############################################
    # Código de aprendizado deve ser definido aqui!
    ###############################################

    if terminated or truncated:
      print(f"Number of actions in episode {episode+1}: {n_actions}")
      rewards_per_episode.append(total_reward_in_episode)
      n_actions = 0

env.close()

print(len(rewards_per_episode))
plt.plot(savgol_filter(rewards_per_episode, 5, 2))

plt.xlabel('episódio')
plt.ylabel('recompensa total')
plt.show()
plt.savefig("no_learning.png")

