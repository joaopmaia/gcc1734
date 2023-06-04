from abc import ABC, abstractmethod

class Environment(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_num_states(self):
        pass

    @abstractmethod
    def get_num_actions(self):
        pass

    def reset(self):
        return self.env.reset()

    @abstractmethod
    def get_state_id(self, state):
        pass

    @abstractmethod
    def get_random_action(self):
        pass
    
    def step(self, action):
        return self.env.step(action)

    def get_id(self):
        return self.env.unwrapped.spec.id
