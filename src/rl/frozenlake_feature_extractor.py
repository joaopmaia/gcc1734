import numpy as np
from feature_extractor import FeatureExtractor

'''
  Special locations
    0: hole (1 * 4 + 1 = 5)
    1: hole (1 * 4 + 3 = 7)
    2: hole (2 * 4 + 3 = 11)
    3: hole (3 * 4 + 0 = 12)

    4: goal (3 * 4 + 3 = 15)
'''
special_locations_dict = {0: (1,1), 1: (1,3), 2: (2,3), 3: (3,0), 4: (3,3)}

borders = [ 0,  1,  2, 3,  # when line is 0
           12, 13, 14,     # when line is 3, but 15 is goal so is excluded
            4,  8,         # when column is 0
            7, 11          # when column is 3
           ]

holes = [5, 7, 11, 12]

class Actions:
  '''
    Actions
      0 : mover para baixo
      1 : mover para cima
      2 : mover para a direita
      3 : mover para a esquerda
  '''
  DOWN = 0
  UP = 1
  RIGHT = 2
  LEFT = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):
  __actions_one_hot_encoding = {
    Actions.DOWN:   np.array([1,0,0,0]), 
    Actions.UP:     np.array([0,1,0,0]), 
    Actions.RIGHT:  np.array([0,0,1,0]), 
    Actions.LEFT:   np.array([0,0,0,1]), 
  }

  def __init__(self, env):
    '''
    Initializes the FrozenLakeFeatureExtractor object. 
    It adds feature extraction methods to the features_list attribute.
    '''
    self.env = env
    self.features_list = []
    self.features_list.append(self.f0)
    self.features_list.append(self.f1)
    self.features_list.append(self.f2)
    self.features_list.append(self.f3)
    self.features_list.append(self.f4)
    """ self.features_list.append(self.f5) """
    self.features_list.append(self.f6)

  def get_num_features(self):
    '''
    Returns the number of features extracted by the feature extractor.
    '''
    return len(self.features_list) + self.get_num_actions()

  def get_num_actions(self):
    '''
    Returns the number of actions available in the environment.
    '''
    return len(self.get_actions())

  def get_action_one_hot_encoded(self, action):
    '''
    Returns the one-hot encoded representation of an action.
    '''
    return self.__actions_one_hot_encoding[action]

  def is_terminal_state(self, state):
    assert type(state) == int
    if state == 15:
      return True
    else:
      return False
  
  def get_actions(self):
    '''
    Returns a list of available actions in the environment.
    '''
    return [Actions.DOWN, Actions.UP, Actions.RIGHT, Actions.LEFT]
  
  def get_features(self, state, action):
    '''
    Takes a state and an action as input and returns the feature vector for that state-action pair. 
    It calls the feature extraction methods and constructs the feature vector.
    '''
    # print("feature_vector.shape")

    feature_vector = np.zeros(len(self.features_list))
    # print(feature_vector.shape)

    for index, feature in enumerate(self.features_list):
      feature_vector[index] = feature(state, action)

    # print(feature_vector.shape)
    # constant feature corresponding to the bias term
    # feature_vector[0] = 1.0

    action_vector = self.get_action_one_hot_encoded(action)
    feature_vector = np.concatenate([feature_vector, action_vector])

    # print(feature_vector.shape)

    return feature_vector
    
  def f0(self, state, action):
    '''
    This is just the bias term.
    '''
    return 1.0

  def f1(self, state, action):
    '''
    This feature computes the reciprocal distance from the elf to the gift
    '''
    player_x = state // 4
    player_y = state % 4
    dist = self.__manhattanDistance((player_x, player_y), (3, 3))
    return 1 / (dist + 1) 

  def f2(self, state, action):
    '''
    This feature indicates the reciprocal distance from the elf to the nearest hole
    '''
    player_x = state // 4
    player_y = state % 4
    distances = []

    d1 = self.__manhattanDistance((player_x, player_y), special_locations_dict[0])
    d2 = self.__manhattanDistance((player_x, player_y), special_locations_dict[1])
    d3 = self.__manhattanDistance((player_x, player_y), special_locations_dict[2])
    d4 = self.__manhattanDistance((player_x, player_y), special_locations_dict[3])

    distances.append(d1)
    distances.append(d2)
    distances.append(d3)
    distances.append(d4)

    nearest = distances[np.argmin(distances)]
    return nearest / 6

  def f3(self, state, action):
    '''
    This feature indicates if the elf is in one of the borders, limiting its movement
    '''
    if state in borders:
      return 0.0
    else:
      return 1.0

  def f4(self, state, action):
    '''
    This feature indicates if the elf has reached the goal
    '''
    if state == 15:
      return 1.0
    else:
      return 0.0

  """ def f5(self, state, action):
    '''
    This features indicates the safe directions elf can take
    '''
    directions = [] # 0: up, 1: down, 2: left, 3: right
    safe_rate = 1.0

    directions.append(((state // 4) - 1) * 4 + state % 4)
    directions.append(((state // 4) + 1) * 4 + state % 4)
    directions.append((state // 4) * 4 + ((state % 4) - 1))
    directions.append((state // 4) * 4 + ((state % 4) + 1))

    for d in directions:
      if d in holes:
        safe_rate -= 0.25
      else:
        continue

    return safe_rate """

  def f6(self, state, action):
    '''
    This feature is active when the elf bumps into the border
    '''
    l = state // 4
    c = state % 4

    border_bump = ((c == 0) and (action == Actions.LEFT)) or \
                  ((c == 3) and (action == Actions.RIGHT)) or \
                  ((l == 0) and (action == Actions.UP)) or \
                  ((l == 3) and (action == Actions.DOWN))
    
    return border_bump

  @staticmethod
  def __manhattanDistance(xy1, xy2):
    '''
    Computes the Manhattan distance between two points.
    '''
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])