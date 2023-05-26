from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_num_features(self):
      pass


