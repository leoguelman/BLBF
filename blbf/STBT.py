"""Supervised to Bandit Transform (STBT)."""

import sklearn.model_selection
import sklearn.linear_model 
import numpy as np
import math

class STBT:
   
    """
    Performs Supervised to Bandit Conversion for classification 
    datasets. This conversion is generally used to test the limits of 
    counterfactual learning in a well-controlled environment [1,2,3]. 
    
    Parameters
    ----------
    
    train_frac : float, default: 0.50
        It should be between 0.0 and 1.0 and represents the
        proportion of the dataset to include in the train split.
        
    permute : bool, default: False
        Randomly permute the data before the random split between train and test.

    logging_type : str, default: "uniform"
        The type of logging policy. If "uniform", uniform random samples from the 
        labels $y$ to simulate a logging policy. If "biased", the logging policy
        is a stochastic function of the covariates.
        
    sample_frac : float, default: None
        A sample fraction between (0.0,1.0]. This is the sample fraction of the
        training data used to fit the target policy. By default, the full
        training set is used. 
     
    References
    ----------
    
    .. [1] N. Jiang, and  L. Li, Doubly Robust Off-policy Value Evaluation for Reinforcement Learning, 
           Proceedings of Machine Learning Research, 48, 652--661, 2016.
    .. [2] A. Swaminathan and T. Joachims, Batch Learning from Logged Bandit Feedback through 
           Counterfactual Risk Minimization, Journal of Machine Learning Research, 16(52),
           1731--1755, 2015.
    .. [3] A. Swaminathan and T. Joachims, The self-normalized estimator for counterfactual learning, 
           Advances in Neural Information Processing Systems, 28, 16(52), 3231--3239, 2015.
           
    Examples
    --------
    >>> np.random.seed(42)
    >>> import blbf.DataReader as DataReader
    >>> X, y = DataReader.get_data(dataset='ecoli')  
    >>> obj = STBT()
    >>> sample_batch = obj.generate_batch(X, y)
    >>> sample_batch.y_train_logging[0:5]
    array([1, 1, 0, 0, 0]))
    
    """
    
    def __init__(self, train_frac: float = 0.50, permute: bool = False, logging_type: str = 'uniform',
                 sample_frac: float = None):
        self.train_frac = train_frac
        self.permute = permute
        self.logging_type = logging_type
        self.sample_frac = sample_frac
        
    def __repr__(self):
        
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def _validate_input(self):
        if not isinstance(self.train_frac, float) or not (0.0 < self.train_frac < 1.0):
            raise ValueError("`train_frac` should be a float in (0.0,1.0), got %s" % self.train_frac)
            
        if self.sample_frac is not None and self.sample_frac is not (0.0 < self.sample_frac <= 1.0):
            raise ValueError("`sample_frac` should be a float in (0.0,1.0], got %s" % self.sample_frac)

        if self.logging_type not in ['uniform', 'biased']:
            raise ValueError("`logging_type` should be either 'uniform' or 'biased', got %s" % self.logging_type)
    
    def _softmax(self, x, axis=-1):
        
        kw = dict(axis=axis, keepdims=True)
        xrel = x - x.max(**kw)
        exp_xrel = np.exp(xrel)
        p = exp_xrel / exp_xrel.sum(**kw)  
         
        return p
        
         
    def generate_batch(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Generate Supervised to Bandit batch
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array of shape (n_samples,)
            Target vector relative to X.
        **kwargs : Arguments passed to fit method in 
                   `sklearn.linear_model.LogisticRegression` class.
        
        Returns
        -------
        X_train : array of shape (n_train_samples, n_features)
        y_train : array of shape (n_train_samples,)
        X_test : array of shape (n_test_samples, n_features)
        y_test : array of shape (n_test_samples,)

        y_train_logging : array of shape (n_train_samples,)
            Logging policy labels on train data
        train_logging_probs : array of shape (n_train_samples, n_classes)     
            Logging policy probabilities on train data
        train_logging_prob : array of shape (n_train_samples,)
            Logging policy probability corresponding to the chosen logging label on train data
        y_train_logging_idx : array of shape (n_train_samples, n_classes)
            Binary matrix with 1s indicating which action was taken by the logging policy in train data
           
        y_test_logging : array of shape (n_test_samples,)
             Logging policy labels on test data
        test_logging_probs : array of shape (n_test_samples, n_classes)   
            Logging policy probabilities on test data
        test_logging_prob : array of shape (n_test_samples,)
            Logging policy probability corresponding to the chosen logging label on test data
       
        y_train_target : array of shape (n_train_samples,)
             Target policy labels on train data
        train_target_prob : array of shape (n_train_samples, n_classes)     
             Target policy probabilities on train data
        train_target_probs : array of shape (n_train_samples,)
            Target policy probability corresponding to the chosen logging label on train data
       
        y_test_target : array of shape (n_test_samples,)
             Target policy labels on test data
        test_target_prob : array of shape (n_test_samples, n_classes)     
             Target policy probabilities on test data
        test_target_probs : array of shape (n_test_samples,)
            Target policy probability corresponding to the chosen logging label on test data
       
        true_target_value_test : float
            True value of Target policy on test data
       
        train_logging_reward : array of shape (n_train_samples,)
            Observed reward of logging policy on train data 
        test_logging_reward : array of shape (n_test_samples,)
            Observed reward of logging policy on test data 
            
        """
        self._validate_input()
        
        self.generate_batch_call = True
        
        self.dual = False
        
        if self.permute:
            permute = np.random.permutation(X.shape[0])
            X = X[permute, :]
            y = y[permute]

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(X, y,
                train_size = self.train_frac) 
            
        n_train_samples, n_features = self.X_train.shape
        n_test_samples = self.X_test.shape[0]

    
        y_train_u = np.unique(self.y_train)
        
        if self.logging_type == 'uniform':

            self.y_train_logging = np.random.choice(y_train_u, size=n_train_samples)  
            self.train_logging_prob = np.repeat(1.0/len(y_train_u), n_train_samples)
            self.train_logging_probs = np.repeat(self.train_logging_prob.reshape(-1,1), len(y_train_u), axis=1)
            self.y_test_logging = np.random.choice(y_train_u, size=n_test_samples)  
            self.test_logging_prob = np.repeat(1.0/len(y_train_u), n_test_samples)
            self.test_logging_probs = np.repeat(self.test_logging_prob.reshape(-1,1), len(y_train_u), axis=1)
            
            self.y_train_logging_idx = np.full((n_train_samples, len(y_train_u)), False, dtype=bool)
            for i in range(n_train_samples):
                self.y_train_logging_idx[i, np.where(y_train_u==self.y_train_logging[i])[0][0]] = True 
        
        else:
            
            W = np.random.normal(0, 1, (n_features, len(y_train_u)))
            lp_train = self.X_train @ W
            lp_test = self.X_test @ W
            self.train_logging_probs = self._softmax(lp_train)
            self.test_logging_probs = self._softmax(lp_test)
            
            self.y_train_logging_idx = np.full((n_train_samples, len(y_train_u)), False, dtype=bool)
            y_test_logging_idx = np.full((n_test_samples, len(y_train_u)), False, dtype=bool)
            
            for sample in range(n_train_samples):
                choice = np.random.multinomial(1, self.train_logging_probs[sample,:], size = 1)[-1]
                self.y_train_logging_idx[sample, :] = choice
            
            for sample in range(n_test_samples):
                choice = np.random.multinomial(1, self.test_logging_probs[sample,:], size = 1)[-1]
                y_test_logging_idx[sample, :] = choice
            
            self.y_train_logging = np.array([y_train_u,]*n_train_samples)[self.y_train_logging_idx]
            self.y_test_logging = np.array([y_train_u,]*n_test_samples)[y_test_logging_idx]
            
            self.train_logging_prob = self.train_logging_probs[self.y_train_logging_idx]
            self.test_logging_prob = self.test_logging_probs[y_test_logging_idx]
            
        if self.sample_frac is not None:
            n_subsamples = math.ceil(self.sample_frac * n_train_samples)
            idx_subsamples = np.random.randint(n_train_samples, size=n_subsamples)
            X_train_subsamples = self.X_train[idx_subsamples, :]
            y_train_subsamples = self.y_train[idx_subsamples]
            if n_subsamples < n_features:
                self.dual=True
            target_policy = sklearn.linear_model.LogisticRegression(**kwargs, dual=self.dual).fit(X_train_subsamples, y_train_subsamples)
       
        else:
            if n_train_samples < n_features:
                self.dual=True
            target_policy = sklearn.linear_model.LogisticRegression(**kwargs, dual=self.dual).fit(self.X_train, self.y_train)
        
        self.train_target_probs = target_policy.predict_proba(self.X_train)
        self.test_target_probs = target_policy.predict_proba(self.X_test)
         
        y_train_target = list()
        train_target_prob = list()
        y_test_target = list()
        test_target_prob = list()
         
        for i in range(n_train_samples):
            y_train_target_i = np.random.choice(y_train_u, size=1, 
                                                 replace=False, p=self.train_target_probs[i,:])[0]
            y_train_target.append(y_train_target_i)
            train_target_prob.append(self.train_target_probs[i, np.where(y_train_u==y_train_target_i)[0][0]])
        self.y_train_target = np.array(y_train_target)
        self.train_target_prob = np.array(train_target_prob)
        
        for i in range(n_test_samples):
            y_test_target_i = np.random.choice(y_train_u, size=1, 
                                                 replace=False, p=self.test_target_probs[i,:])[0]
            y_test_target.append(y_test_target_i)
            test_target_prob.append(self.test_target_probs[i, np.where(y_train_u==y_test_target_i)[0][0]])
        self.y_test_target = np.array(y_test_target)
        self.test_target_prob = np.array(test_target_prob)
        
        self.true_target_value_test = np.mean(1 * (self.y_test == self.y_test_target))
        
        self.train_logging_reward = 1 * (self.y_train == self.y_train_logging)
        self.test_logging_reward = 1 * (self.y_test == self.y_test_logging)
                
           
        return self


