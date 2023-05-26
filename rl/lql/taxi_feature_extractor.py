'''
see http://alborz-geramifard.com/Files/13FTML-RLTutorial.pdf
see https://stats.stackexchange.com/questions/291551/how-to-deal-with-increasing-action-space-in-td-learning-using-linear-function-ap

'''

import gymnasium as gym
import numpy as np
from actions import Actions
from feature_extractor import FeatureExtractor

special_locations_dict = {0: (0,0), 1: (0,4), 2: (4,0), 3: (4,3)}

class TaxiFeatureExtractor(FeatureExtractor):
  NUM_FEATURES = 6

  def __init__(self, env):
    self.env = env

  def get_num_features(self):
    return self.NUM_FEATURES

  def __get_features_as_dict(self, state, action):
    features = {}
    features['f0'] = self.f0(state, action)
    features['f1'] = self.f1(state, action)
    features['f2'] = self.f2(state, action)
    features['f3'] = self.f3(state, action)
    features['f4'] = self.f4(state, action)
    features['f5'] = self.f5(state, action)
    # features['f6'] = self.f6(state, action)
    # features['f7'] = self.f7(state, action)
    return features

  def get_features(self, state, action):
    features = self.__get_features_as_dict(state, action)
    other_array = np.fromiter(features.values(), dtype=float)
    return other_array

  '''
  This is just the bias term.
  '''
  def f0(self, state, action):
    return 1.0

  '''
  This feature computes the reciprocal distance from the taxi to the passenger
  '''
  def f1(self, state, action):
    l, c, p, _ = self.env.unwrapped.decode(state)
    taxi_location = (l, c)
    if p < 4:
      passenger_location = special_locations_dict[p]
      dist = self.__manhattanDistance(taxi_location, passenger_location)
      return 1 / (dist + 1)
    else:
      return 0.0

  '''
  This feature indicates when the passenger is correctly unboarded.
  '''
  def f2(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    passenger_is_onboard = (p == 4)
    if passenger_is_onboard:
      taxi_loc = (l, c)
      destiny_loc = self.env.unwrapped.locs[d]
      dist = self.__manhattanDistance(taxi_loc, destiny_loc)
      if dist == 0:
        if action == Actions.DROP:
          return 1.0
    return 0.0

  '''
  This feature indicates when the passenger is correctly boarded.
  '''
  def f3(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    assert p <= 4
    passenger_is_onboard = (p == 4)
    if not passenger_is_onboard:
      taxi_loc = (l, c)
      boarding_loc = self.env.unwrapped.locs[p]
      dist = self.__manhattanDistance(taxi_loc, boarding_loc)
      if dist == 0:
        if action == Actions.PICK:
          return 1.0
    return 0.0

  '''
  This feature computes the reciprocal distance from the taxi to the 
  origin when the passenger is not boarded.
  '''
  def f4(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    passenger_is_onboard = (p == 4)
    if not passenger_is_onboard:
      taxi_loc = (l, c)
      origin_loc = self.env.unwrapped.locs[p]
      dist = self.__manhattanDistance(taxi_loc, origin_loc)
      return 1 / (dist + 1)
    else:
      return 0.0

  '''
  This feature computes the reciprocal distance from the taxi to the 
  destination when the passenger is boarded.
  '''
  def f5(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    passenger_is_onboard = (p == 4)
    if passenger_is_onboard:
      taxi_loc = (l, c)
      dest_loc = self.env.unwrapped.locs[d]
      dist = self.__manhattanDistance(taxi_loc, dest_loc)
      return 1 / (dist + 1)
    else:
      return 0.0

  '''
  This feature is active when the passenger is incorrectly (un)boarded.
  '''
  def f6(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    assert p <= 4
    passenger_is_onboard = (p == 4)
    taxi_loc = (l, c)

    print(f"taxi_loc: {taxi_loc}")
    print(f"passenger_is_onboard: {passenger_is_onboard}")
    print(f"p: {p}")

    if passenger_is_onboard:
      dest_loc = self.env.unwrapped.locs[d]
      print(f"dest_loc: {dest_loc}")
      if (action != Actions.DROP) and (taxi_loc == dest_loc):
        return 1.0
    else:
      boarding_loc = self.env.unwrapped.locs[p]
      print(f"boarding_loc: {boarding_loc}")
      if (action != Actions.PICK) and (taxi_loc == boarding_loc):
        return 1.0
    
    return 0.0

  '''
  This feature is active when the agent bumps into a wall.
  '''
  def f7(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    border_bump = ((c == 0) and (action == Actions.LEFT)) or \
           ((c == 4) and (action == Actions.RIGHT)) or \
           ((l == 0) and (action == Actions.UP)) or \
           ((l == 4) and (action == Actions.DOWN))
    internal_bump = ((l == 0) and (c == 1) and (action == Actions.RIGHT)) or \
                    ((l == 0) and (c == 2) and (action == Actions.LEFT)) or \
                    ((l == 3) and (c == 0) and (action == Actions.RIGHT)) or \
                    ((l == 4) and (c == 0) and (action == Actions.RIGHT)) or \
                    ((l == 3) and (c == 1) and (action == Actions.LEFT)) or \
                    ((l == 4) and (c == 1) and (action == Actions.LEFT)) or \
                    ((l == 3) and (c == 2) and (action == Actions.RIGHT)) or \
                    ((l == 4) and (c == 2) and (action == Actions.RIGHT)) or \
                    ((l == 3) and (c == 3) and (action == Actions.LEFT)) or \
                    ((l == 4) and (c == 3) and (action == Actions.LEFT))
    return border_bump or internal_bump


  def __manhattanDistance(self, xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

# if __name__ == "__main__":
#   ACTION = ["DOWN", "UP", "RIGHT", "LEFT", "PICK", "DROP"]
#   env = gym.make("Taxi-v3")
#   fex = TaxiFeatureExtractor(env)

#   action_id = 4

#   l = 4
#   c = 2
#   p = 3
#   d = 3
#   state = env.encode(l, c, p, d)
  
#   print(f"action: {ACTION[action_id]}")
#   print("f6: ", fex.f6(state, 4))