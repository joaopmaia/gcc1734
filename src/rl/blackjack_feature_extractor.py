import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
  STICK = 0
  HIT = 1

class BlackjackFeatureExtractor(FeatureExtractor):
  __actions_one_hot_encoding = {
    Actions.STICK:   np.array([1,0]), 
    Actions.HIT:     np.array([0,1]) 
  }

  def __init__(self, env):
    '''
    Initializes the TaxiFeatureExtractor object. 
    It adds feature extraction methods to the features_list attribute.
    '''
    self.env = env
    self.features_list = []
    self.features_list.append(self.f0)
    self.features_list.append(self.f1)

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
    if state[2] == True:
      return True
    elif state[0] > 21:
      return True
    return False

  def get_actions(self):
    '''
    Returns a list of available actions in the environment.
    '''
    return [Actions.STICK, Actions.HIT]
  
  def get_features(self, state, action):
    '''
    Takes a state and an action as input and returns the feature vector for that state-action pair. 
    It calls the feature extraction methods and constructs the feature vector.
    '''
    feature_vector = np.zeros(len(self.features_list))
    for index, feature in enumerate(self.features_list):
      feature_vector[index] = feature(state, action)

    action_vector = self.get_action_one_hot_encoded(action)
    feature_vector = np.concatenate([feature_vector, action_vector])

    return feature_vector

  def f0(self, state, action):
    '''
    This is just the bias term.
    '''
    return 1.0

  def f1(self, state, action):
    '''
    Verifica a pontuação da mão do player.
    '''
    player_sum = state[0]

    if player_sum > 21:
      return 0.0
    else:
      return player_sum / 21
    
  def f2(self, state, action):
    '''
    Stick c/ valor alto
    '''
    player_sum = state[0]

    if action == Actions.STICK:
      return player_sum / 21
  
  def f3(self, state, action):
    '''
    Verifica se deu stick com 21
    '''
    player_sum = state[0]

    if player_sum == 21 and action == Actions.STICK:
      return 1.0
    else:
      return 0.0
  
  def f4(self, state, action):
    '''
    Hit com valor alto
    '''
    player_sum = state[0]

    if action == Actions.HIT:
      return (21 - player_sum) / 21
    
  def f5(self, state, action):
    '''
    Stick com soma abaixo de 17
    '''
    player_sum = state[0]

    if player_sum <= 17 and action == Actions.STICK:
      return 0.0
    else:
      return 1.0
    
  def f6(self, state, action):
    '''
    Soma menor que o dealer
    '''
    player_sum = state[0]
    dealer_sum = state[1]

    if player_sum < dealer_sum and action == Actions.STICK:
      return 0.0
    else:
      return 1.0