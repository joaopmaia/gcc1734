'''
see http://alborz-geramifard.com/Files/13FTML-RLTutorial.pdf
see https://stats.stackexchange.com/questions/291551/how-to-deal-with-increasing-action-space-in-td-learning-using-linear-function-ap
see https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
see https://danieltakeshi.github.io/2016/10/31/going-deeper-into-reinforcement-learning-understanding-q-learning-and-linear-function-approximation/
'''

import numpy as np
from feature_extractor import FeatureExtractor

special_locations_dict = {0: (0,0), 1: (0,4), 2: (4,0), 3: (4,3)}

class Actions:
  '''
    Actions
      0 : mover para baixo
      1 : mover para cima
      2 : mover para a direita
      3 : mover para a esquerda
      4 : pegar o passageiro
      5 : entregar o passageiro
  '''
  DOWN = 0
  UP = 1
  RIGHT = 2
  LEFT = 3
  PICK = 4
  DROP = 5

class TaxiFeatureExtractor(FeatureExtractor):
  features_list = []

  __actions_one_hot_encoding = {
    Actions.DOWN:   np.array([1,0,0,0,0,0]), 
    Actions.UP:     np.array([0,1,0,0,0,0]), 
    Actions.RIGHT:  np.array([0,0,1,0,0,0]), 
    Actions.LEFT:   np.array([0,0,0,1,0,0]), 
    Actions.PICK:   np.array([0,0,0,0,1,0]), 
    Actions.DROP:   np.array([0,0,0,0,0,1])
  }

  def __init__(self, env):
    self.env = env
    self.features_list.append(self.f0)
    self.features_list.append(self.f1)
    self.features_list.append(self.f2)
    self.features_list.append(self.f3)
    self.features_list.append(self.f4)
    self.features_list.append(self.f5)
    self.features_list.append(self.f6)
    self.features_list.append(self.f7)

  def get_num_features(self):
    return len(self.features_list)

  def get_num_actions(self):
    return len(self.get_actions())

  def get_action_one_hot_encoded(self, action):
    return self.__actions_one_hot_encoding[action]

  def get_terminal_states(self):
    return [0, 85, 410, 475]
  
  def get_actions(self):
    return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT, Actions.PICK, Actions.DROP]
  
  def get_features(self, state, action):
    feature_vector = np.zeros(len(self.features_list))
    for index, feature in enumerate(self.features_list):
      feature_vector[index] = feature(state, action)
    return feature_vector

  # def get_features(self, state, action):
  #     feature_values = []
  #     feature_values += [self.f0(state, action)]
  #     for a in self.get_actions():
  #         if a == action and (state not in self.get_terminal_states()):
  #             feature_values += [self.f1(state, action)]
  #             feature_values += [self.f2(state, action)]
  #             feature_values += [self.f3(state, action)]
  #             feature_values += [self.f4(state, action)]
  #             feature_values += [self.f5(state, action)]
  #             feature_values += [self.f6(state, action)]
  #             feature_values += [self.f7(state, action)]
  #         else:
  #             for _ in range(0, len(self.features_list)):
  #                 feature_values += [0.0]

  #     feature_vector = np.zeros(len(feature_values))
  #     for index, feature_value in enumerate(feature_values):
  #       feature_vector[index] = feature_value
      
  #     # print(f"feature_vector.shape = {feature_vector.shape}")
  #     return feature_vector

    
  def f0(self, state, action):
    '''
    This is just the bias term.
    '''
    return 1.0

  def f1(self, state, action):
    '''
    This feature computes the reciprocal distance from the taxi to the passenger
    '''
    l, c, p, _ = self.env.unwrapped.decode(state)
    taxi_location = (l, c)
    if p == 4: # passenger is boaded
      passenger_location = taxi_location
    elif p < 4:
      passenger_location = special_locations_dict[p]
    dist = self.__manhattanDistance(taxi_location, passenger_location)
    return 1 / (dist + 1) 

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

  # '''
  # This feature computes the reciprocal distance from the taxi to the 
  # destination when the passenger is boarded.
  # '''
  # def f5(self, state, action):
  #   l, c, p, d = self.env.unwrapped.decode(state)
  #   passenger_is_onboard = (p == 4)
  #   if passenger_is_onboard:
  #     taxi_loc = (l, c)
  #     dest_loc = self.env.unwrapped.locs[d]
  #     dist = self.__manhattanDistance(taxi_loc, dest_loc)
  #     return 1 / (dist + 1)
  #   else:
  #     return 0.0
  '''
  This feature computes the reciprocal distance from the passenger to the 
  destination.
  '''
  def f5(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)

    dest_loc = self.env.unwrapped.locs[d]
    if p == 4: # passenger is boaded
      passenger_location = (l,c)
    elif p < 4:
      passenger_location = special_locations_dict[p]
    dist = self.__manhattanDistance(dest_loc, passenger_location)
    return 1 / (dist + 1)


  '''
  This feature is active when the passenger is incorrectly (un)boarded.
  '''
  def f6(self, state, action):
    l, c, p, d = self.env.unwrapped.decode(state)
    assert p <= 4
    passenger_is_onboard = (p == 4)
    taxi_loc = (l, c)

    # print(f"taxi_loc: {taxi_loc}")
    # print(f"passenger_is_onboard: {passenger_is_onboard}")
    # print(f"p: {p}")

    if passenger_is_onboard:
      dest_loc = self.env.unwrapped.locs[d]
      # print(f"dest_loc: {dest_loc}")
      if (action != Actions.DROP) and (taxi_loc == dest_loc):
        return 1.0
    else:
      boarding_loc = self.env.unwrapped.locs[p]
      # print(f"boarding_loc: {boarding_loc}")
      if (action != Actions.PICK) and (taxi_loc == boarding_loc):
        return 1.0
    
    return 0.0

  def f7(self, state, action):
    '''
    This feature is active when the agent bumps into a wall 
    as an effect of taking the selected action.
    '''
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

# import gymnasium as gym
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