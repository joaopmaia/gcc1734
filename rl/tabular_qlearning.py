import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class QLearningAgentTabular:
  
  def __init__(self, 
               env, 
               decay_rate = 0.0001, 
               learning_rate = 0.7, 
               gamma = 0.618):
    self.env = env
    self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
    self.epsilon = 1.0
    self.max_epsilon = 1.0
    self.min_epsilon = 0.01
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.gamma = gamma # discount rate
    self.epsilons_ = []
    
  def choose_action(self, state, is_in_exploration_mode=True):
    exploration_tradeoff = np.random.uniform(0, 1)
    
    if is_in_exploration_mode and exploration_tradeoff < self.epsilon:
      # exploration
      return np.random.randint(self.env.action_space.n)    
    else:
      # exploitation (taking the biggest Q value for this state)
      return np.argmax(self.q_table[state, :])

  def update(self, state, action, reward, next_state):
    '''
    Apply update rule Q(s,a):= Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    '''
    self.q_table[state, action] = self.q_table[state, action] + \
      self.learning_rate * (reward + self.gamma * \
        np.max(self.q_table[next_state, :]) - self.q_table[state, action])

  def train(self, num_episodes):
    rewards = []

    for episode in range(num_episodes):
      terminated = False
      truncated = False

      state, _ = env.reset()

      rewards_per_episode = []
        
      while not (terminated or truncated):
          
        action = self.choose_action(state)

        # transição
        new_state, reward, terminated, truncated, info = env.step(action)

        agent.update(state, action, reward, new_state)

        if (terminated or truncated):
          # Reduce epsilon to decrease the exploration over time
          self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.decay_rate * episode)
          self.epsilons_.append(self.epsilon)
        
        state = new_state
            
        rewards_per_episode.append(reward)

      mean_reward = np.mean(rewards_per_episode)
      rewards.append(mean_reward)

      if episode % 1000 == 0:
        n_actions = len(rewards_per_episode)
        print(f"Stats for episode {episode}:\n \tn_actions = {n_actions}\n \tmean_reward = {mean_reward:#.2f}")

    return rewards


if __name__ == "__main__":
  env = gym.make("Taxi-v3").env
  agent = QLearningAgentTabular(env)
  rewards = agent.train(num_episodes=60000)

  print(len(rewards))

  plt.plot(savgol_filter(rewards, 1001, 2))
  #plt.plot(rewards)
  plt.title("Curva de aprendizado suavizada")
  plt.xlabel('Episódio');
  plt.ylabel('Recompensa total');
  plt.savefig("tabular_qlearning.png")
