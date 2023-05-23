import argparse
import pickle
from tql import QLearningAgentTabular

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes")
    args = parser.parse_args()

    agent = QLearningAgentTabular.load_agent(args.env_name + "-tql.pkl")

    total_actions, total_penalties = 0, 0
    NUM_EPISODES = args.num_episodes

    for episode in range(NUM_EPISODES):

        state, _ = agent.env.reset()
        actions = 0
        penalties = 0
        reward = 0
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state, is_in_exploration_mode=False)
            state, reward, terminated, truncated, info = agent.env.step(action)

            if reward == -10:
                penalties += 1

            actions += 1

        total_penalties += penalties
        total_actions += actions

    print("**********************************")
    print("Resultados:")
    print("\tMédia de ações por episódio: {}".format(total_actions / NUM_EPISODES))
    print("\tPenalidade média por episódio: {}".format(total_penalties / NUM_EPISODES))
    print("**********************************")

