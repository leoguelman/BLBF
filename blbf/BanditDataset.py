"""Create Bandit datasets."""

import numpy as np

def _get_logging_prob(y, probs):

    y_u = np.unique(y)
    n_samples = len(y)
    prob = list()
    for i in range(n_samples):
        prob.append(probs[i, np.where(y_u==y[i])[0][0]])
    prob = np.array(prob)

    return prob

def _get_y_logging_idx(y):

     y_u = np.unique(y)
     n_samples = len(y)
     y_logging_idx = np.full((n_samples, len(y_u)), False, dtype=bool)
     for i in range(n_samples):
        y_logging_idx[i, np.where(y_u==y[i])[0][0]] = True 

     return y_logging_idx

class PolicyEvaluationBanditDataset:

    """
    Create a Bandit dataset with attributes required 
    for policy evaluation algorithms. 

    Parameters
    ----------
    X_train : array of shape (n_train_samples, n_features)
    X_test : array of shape (n_test_samples, n_features)
    y_train_logging : array of shape (n_train_samples,)
        Logging policy labels on train data
    y_test_logging : array of shape (n_test_samples,)
            Logging policy labels on test data
    train_logging_reward : array of shape (n_train_samples,)
        Observed reward of logging policy on train data 
    test_logging_reward : array of shape (n_test_samples,)
            Observed reward of logging policy on test data 
    test_logging_probs : array of shape (n_test_samples, n_classes)   
            Logging policy probabilities on test data
    y_test_target : array of shape (n_test_samples,)
             Target policy labels on test data

    
    Returns
    -------
    X_train : array of shape (n_train_samples, n_features)
    X_test : array of shape (n_test_samples, n_features)
    y_train_logging : array of shape (n_train_samples,)
        Logging policy labels on train data
    y_test_logging : array of shape (n_test_samples,)
            Logging policy labels on test data
    train_logging_reward : array of shape (n_train_samples,)
        Observed reward of logging policy on train data 
    test_logging_reward : array of shape (n_test_samples,)
            Observed reward of logging policy on test data 
    test_logging_probs : array of shape (n_test_samples, n_classes)   
            Logging policy probabilities on test data
    test_logging_prob : array of shape (n_test_samples,)
            Logging policy probability corresponding to the chosen logging label on test data
    y_test_target : array of shape (n_test_samples,)
             Target policy labels on test data
           
    """

    def __init__(self, X_train=None, X_test=None, y_train_logging=None, 
                 y_test_logging=None, train_logging_reward=None, 
                 test_logging_reward=None, test_logging_probs=None, 
                 y_test_target=None):


        self.X_train = X_train
        self.X_test = X_test
        self.y_train_logging = y_train_logging
        self.y_test_logging = y_test_logging
        self.train_logging_reward = train_logging_reward
        self.test_logging_reward = test_logging_reward
        self.test_logging_probs = test_logging_probs 
        self.y_test_target = y_test_target
        self.test_logging_prob = _get_logging_prob(self.y_test_logging, self.test_logging_probs)
        self.generate_batch_call = None
        
    
class PolicyLearningBanditDataset:
    """
    Create a Bandit dataset with attributes required 
    for policy learning algorithms. 

    Parameters
    ----------
    X_train : array of shape (n_train_samples, n_features)
    X_test : array of shape (n_test_samples, n_features)
    y_train_logging : array of shape (n_train_samples,)
        Logging policy labels on train data
    train_logging_reward : array of shape (n_train_samples,)
        Observed reward of logging policy on train data 
    train_logging_probs : array of shape (n_train_samples, n_classes)     
        Logging policy probabilities on train data
    
    Returns
    -------
    X_train : array of shape (n_train_samples, n_features)
    X_test : array of shape (n_test_samples, n_features)
    y_train_logging : array of shape (n_train_samples,)
        Logging policy labels on train data
    train_logging_reward : array of shape (n_train_samples,)
        Observed reward of logging policy on train data 
    train_logging_probs : array of shape (n_train_samples, n_classes)     
        Logging policy probabilities on train data
    train_logging_prob : array of shape (n_train_samples,)
        Logging policy probability corresponding to the chosen logging label on train data
    y_train_logging_idx : array of shape (n_train_samples, n_classes)
            Binary matrix with 1s indicating which action was taken by the logging policy in train data
           
    """

    def __init__(self, X_train=None, X_test = None, y_train_logging=None, 
                 train_logging_reward=None, train_logging_probs=None):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train_logging = y_train_logging
        self.train_logging_reward = train_logging_reward
        self.train_logging_probs = train_logging_probs
        self.train_logging_prob = _get_logging_prob(self.y_train_logging, self.train_logging_probs)       
        self.y_train_logging_idx = _get_y_logging_idx(self.y_train_logging)



