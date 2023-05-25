import gymnasium as gym

env = gym.make("Taxi-v3", render_mode="human")

observation, info = env.reset(seed=42)

print(f'Estado antes da ação: {observation}')

observation, reward, terminated, truncated, info = env.step(1) # Toma ação "ir para Norte"

print()
print(observation)
print(reward)
print(terminated)
print(truncated)
print(info)
print()

print(f'Estado após a ação: {observation}')

input()

state = env.encode(4, 2, 2, 3)
print("State id:", state)
env.env.s = state
env.render()

input()
print('Number of states: {}'.format(env.observation_space.n))
print('Number of actions: {}'.format(env.action_space.n))

input()
observation = env.reset()
print(observation)

input()
env = gym.make('Taxi-v3').env
state = env.reset()
l, c, p, d = env.unwrapped.decode(state)
print(l, c, p, d)
print('Localização do taxi = {}'.format((l, c)))
print('Localização do passageiro = {}'.format(env.unwrapped.locs[p]))
print('Localização do destino = {}'.format(env.unwrapped.locs[d]))
env.render()
