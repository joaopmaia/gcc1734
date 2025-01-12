import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training = True, render = False):
    env = gym.make("MountainCar-v0", render_mode = "human" if render else None)
    rng = np.random.default_rng()
    pos_space = np.linspace(-1.2, 0.6, 20)
    vel_space = np.linspace(-0.07, 0.07, 20)

    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open("mountain_car.pkl", "rb")
        q = pickle.load(f)
        f.close()

    lr_a = 0.9 #learning rate
    df_g = 0.9 #discount factor
    e = 1 #epsilon
    e_decay = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        state_pos = np.digitize(state[0], pos_space)
        state_vel = np.digitize(state[1], vel_space)
        
        terminated = False

        rewards = 0

        while not (terminated):
            if is_training and rng.random() < e:
                action  = env.action_space.sample()
            else:
                action = np.argmax(q[state_pos, state_vel, :])
            
            new_state, reward, terminated, truncated, info = env.step(action)

            new_state_pos = np.digitize(new_state[0], pos_space)
            new_state_vel = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_pos, state_vel, action] = q[state_pos, state_vel, action] + lr_a * (
                    reward + df_g * np.max(q[new_state_pos, new_state_vel, :]) - q[state_pos, state_vel, action]
                )

            state = new_state
            state_pos = new_state_pos
            state_vel = new_state_vel

            rewards += reward
        
        e = max(e - e_decay, 0)

        rewards_per_episode[i] = rewards

        print(f"episode {i} finished")

    env.close()

    if is_training:
        f = open("mountain_car.pkl", "wb")
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100) : t+1])
    plt.plot(mean_rewards)
    plt.savefig(f"mountain_car.png")

if __name__ == '__main__':
    run(3, is_training = False, render = True)