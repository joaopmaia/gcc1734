import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures

class FeatureTransformer:
  def __init__(self, env, fex, use_polynomial_features: bool):
    self.env = env
    self.poly = None
    if use_polynomial_features:
      self.poly = PolynomialFeatures(interaction_only=True)
    self.fex = fex
    self.__initInternal()

  def __initInternal(self):
    observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])

    n_actions = self.env.action_space.n

    dataset = np.zeros([len(observation_examples) * n_actions, self.fex.get_num_features()])

    print(dataset.shape)

    i = 0
    for s in observation_examples:
      for a in range(n_actions):
        newrow = self.fex.get_features(s,a)
        dataset[i] = newrow
        i += 1

    if self.poly is not None:
      print('Dataset - shape before poly', dataset.shape)
      dataset = self.poly.fit_transform(dataset)
      print('Dataset - shape after poly', dataset.shape)

    print('Mean:')
    print(np.mean(dataset, axis = 0))

    self.scaler = sklearn.preprocessing.RobustScaler()
    self.scaler.fit(dataset)

  def get_features_as_array(self, state, action):
    feature_vector = self.fex.get_features(state, action)
    feature_vector = feature_vector.reshape(1, -1)
    if self.poly is not None:
      feature_vector = self.poly.transform(feature_vector)
    # feature_vector = self.scaler.transform(feature_vector)
    feature_vector = feature_vector.flatten()
    
    # constant feature corresponding to the bias term
    feature_vector[0] = 1.0

    return feature_vector