import gymnasium as gym 

environment_name = "CarRacing-v2"
#environment_name = "Taxi-v3"
#environment_name = "LunarLander-v2"

env = gym.make(environment_name, render_mode="human")
#env.metadata['render_fps'] = 150

#print('render_modes:', env.metadata['render_modes'])
#print('metadata:', env.metadata)

episodes = 5

for episode in range(1, episodes+1):
    
    observation, info = env.reset(seed=123, options={})
    terminated = False
    truncated = False
    score = 0 
    
    while not (terminated or truncated):
        #env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
          
    print(f'Episode: {episode} Score: {score}')
    
env.close()