"""Off-Policy Evaluation Estimators."""

import blbf.utils as utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC



class PolicyEvaluation:
        """
        Performs off-policy evaluation with bandit feedback. 
        
        Parameters
        ----------
        method : str, default: 'ips'.
            The policy evaluation method. The default is 'ips'.
            It should be one of: 'ips' (Inverse Propensity Score), 
            'dm' (Direct Method), 'dr' (Doubly Robust), 'switch'
            (SWITCH estimator).
            
        tau : float, default: 0.001.
            Hyperparameter added to IPS or SWICTH estimator for numerical stability. 
            
            For method='ips', the logging probabilities in the test set get adjusted by
            the max(logging probabilities, tau).
            
            For method = 'switch', when logging probabilities are larger than this parameter,
            the 'dm' estimator is applied, otherwise the 'dr' estimator is applied. 
            
            
        References
        ----------
    
        .. [1] Y. Wang, A. Agarwal and M. Dud\'{\i}k, Optimal and Adaptive Off-policy Evaluation in Contextual Bandits, 
               Proceedings of Machine Learning Research, 70, 3589--3597, 2017.
        .. [2] N. Jiang, and  L. Li, Doubly Robust Off-policy Value Evaluation for Reinforcement Learning, 
               Proceedings of Machine Learning Research, 48, 652--661, 2016.
        .. [3] K{\"u}nzel, S., Sekhon, J., Bickel, P. and Yu, B., Metalearners for estimating heterogeneous 
               treatment effects using machine learning, Proceedings of the National Academy of Sciences, 
               116(10), 4156--4165, 2019. 
               
        Examples
        --------
        >>> np.random.seed(42)
        >>> from blbf.STBT import STBT
        >>> from blbf.PolicyEvaluation import PolicyEvaluation 
        >>> import blbf.DataReader as DataReader
        >>> X, y = DataReader.get_data(dataset='ecoli')
        >>> obj = STBT(train_frac= 0.5)
        >>> data = obj.generate_batch(X, y, max_iter=1000)
        >>> PolicyEvaluation(method='dr').evaluate_policy(data = data)
        0.7241601514218099

        """
    
        def __init__(self, method: str = 'ips', tau: float = 0.001):
            
            self.method = method
            self.tau = tau 
            
            valid_methods = ['ips', 'dm', 'dr', 'switch']
            if self.method not in valid_methods:
                raise ValueError("%s is not a valid method." % self.method)
        
            if self.tau <= 0 or self.tau >1:
                raise ValueError("`tau` must be in the (0, 1) interval, got %s." % self.tau)
                       
        def __repr__(self):
            
            items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
        
        def evaluate_policy(self, data, clf: str = 'LogisticRegression', **kwargs) -> float:
            """
            Parameters
            ----------
            data : STBT object
                This must be a Supervised to Bandit Transform (STBT) class with fitted 
                `generate_batch` method.
                
            clf : str, default: 'LogisticRegression'
            A sklearn classification estimator. Must be one of 'LogisticRegression', 
            'LogisticRegressionCV', 'RandomForestClassifier', or 'SVC'.
           
            **kwargs : Arguments passed to clf.
    
            Returns
            -------
            float.
              The estimated value of the policy.
        
            """
              
            if not hasattr(data, 'generate_batch_call'):
                raise TypeError("The method `generate_batch` must be called first on the instance: %s." % (data))
                                    
            if self.method == 'ips':
                
                if self.tau is not None:
                    adj_test_logging_prob = np.maximum(self.tau, data.test_logging_prob)  
                
                else:
                    adj_test_logging_prob = data.test_logging_prob
                
                v = np.mean(data.test_logging_reward * (data.y_test_logging == data.y_test_target) / adj_test_logging_prob)
                
            else:
                XY_train, lb_fit = utils.create_interactions(data.X_train, data.y_train_logging)
                m = eval(clf)(**kwargs).fit(XY_train, data.train_logging_reward)
                XY_test_target, _ = utils.create_interactions(data.X_test, data.y_test_target, one_hot_labeler = lb_fit)
                test_target_pred_reward = m.predict_proba(XY_test_target)[:,1]
                
                if self.method in ['dr', 'switch']:
                    XY_test_logging, _ = utils.create_interactions(data.X_test, data.y_test_logging, one_hot_labeler = lb_fit)
                    test_logging_pred_reward = m.predict_proba(XY_test_logging)[:,1]
                    dr_adj = (data.test_logging_reward - test_logging_pred_reward) * \
                                 (data.y_test_logging == data.y_test_target) / data.test_logging_prob
                
                if self.method == 'dm':
                    v = np.mean(test_target_pred_reward) 
                    
                elif self.method == 'dr':
                    v = np.mean(test_target_pred_reward + dr_adj)
                        
                elif self.method == 'switch':
                    switch_indicator = np.array(data.test_logging_prob <= self.tau, dtype=int)
                    switch_estimator_rewards = (1-switch_indicator) * (dr_adj + test_target_pred_reward)
                    switch_estimator_rewards += switch_indicator * test_target_pred_reward
                    v = np.mean(switch_estimator_rewards)
          
            return v
        
