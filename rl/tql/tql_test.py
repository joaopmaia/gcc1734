import argparse
from rl.tql.tql import QLearningAgentTabular

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes")
    args = parser.parse_args()
    assert args.num_episodes > 0

    agent = QLearningAgentTabular.load_agent(args.env_name + "-tql-agent.pkl")

    total_actions, total_penalties = 0, 0
    total_success = 0

    for episode in range(args.num_episodes):

        state, _ = agent.env.reset()
        actions = 0
        penalties = 0
        reward = 0
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state, is_in_exploration_mode=False)
            state, reward, terminated, truncated, info = agent.env.step(action)

            if reward == +20:
                total_success += 1

            if reward == -10:
                penalties += 1

            actions += 1

        total_penalties += penalties
        total_actions += actions

    print("***Results***********************")
    print("Percent successful episodes: {}".format(total_success / args.num_of_episodes))
    print("Average number of actions per episode: {}".format(total_actions / args.num_episodes))
    print("Average penalties per episode: {}".format(total_penalties / args.num_episodes))
    print("**********************************")
