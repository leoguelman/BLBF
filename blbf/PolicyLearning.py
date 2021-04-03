"""Off-Policy Learning Estimators."""

import blbf.utils as utils
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
#from vowpalwabbit import pyvw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class EvaluationMetrics:
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def error_rate(y_pred, y) -> float:
        er = 1 - np.mean(1 * (y_pred == y))
        
        return er

class BanditDataset(Dataset):
    
    def __init__(self, X, y, p0, r, y_idx):
        self.X = X
        self.y = y
        self.p0 = p0
        self.r = r
        self.y_idx = y_idx
        
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.p0[index], self.r[index], self.y_idx[index]
        
    def __len__ (self):
        return len(self.X)
    
    
class LinearModel(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(n_features, n_actions)

    def forward(self, x):
        xw_plus_b = self.linear(x)
        return xw_plus_b # batch size x n_actions
    
class NonLinearModel(torch.nn.Module):
    def __init__(self, n_features, n_actions, n_hidden=3):
        super().__init__()
        self.l1 = nn.Linear(n_features,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_actions)
        
    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))
   

class RewardPredictor(EvaluationMetrics):
    """
    Performs policy learning using by directly predicting the Reward as a function of covariates, 
    actions and their interaction. 
    
    References
    ----------
    
    .. [1] A. Swaminathan and T. Joachims, Batch Learning from Logged Bandit Feedback through 
           Counterfactual Risk Minimization, Journal of Machine Learning Research, 16(52),
           1731--1755, 2015.
    .. [2] A. Swaminathan and T. Joachims, The self-normalized estimator for counterfactual learning, 
           Advances in Neural Information Processing Systems, 28, 16(52), 3231--3239, 2015.
    .. [3] A. Swaminathan, T. Joachims, and M. de Rijke, Deep Learning with Logged Bandit Feedback, 
           International Conference on Learning Representations,  2018.
    
    """
        
    
    def __init__(self) -> None:
        pass
     
    def __repr__(self) -> str:
    
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def learn_policy(self, data, clf: str = 'LogisticRegression', **kwargs) -> None:
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
        int.
          The predicted best policy.
        
        """    
    
        XY_train, lb_fit = utils.create_interactions(data.X_train, data.y_train_logging)
        y_train_logging_u = np.unique(data.y_train_logging)
        self.train_pred_reward_arr = np.zeros(shape=[data.X_train.shape[0], len(y_train_logging_u)]) 
        self.test_pred_reward_arr = np.zeros(shape=[data.X_test.shape[0], len(y_train_logging_u)]) 
        m = eval(clf)(**kwargs).fit(XY_train, data.train_logging_reward)
        
        for i, yval in enumerate(y_train_logging_u):
           XY_train_yval, _ = utils.create_interactions(data.X_train, np.repeat(yval, data.X_train.shape[0]), one_hot_labeler = lb_fit)
           XY_test_yval, _ = utils.create_interactions(data.X_test, np.repeat(yval, data.X_test.shape[0]), one_hot_labeler = lb_fit)
           self.train_pred_reward_arr[:,i] = m.predict_proba(XY_train_yval)[:,1]
           self.test_pred_reward_arr[:,i] = m.predict_proba(XY_test_yval)[:,1]
             
        self.est_best_policy = np.array(y_train_logging_u[np.argmax(self.test_pred_reward_arr, axis=1)])
             
        return self
    
class OutcomeWeightedLearning(EvaluationMetrics):
    """
    Performs policy learning by transforming the learning problem into a 
    weighted multi-class classification problem. 
    
    
    References
    ----------
    
    .. [1] Y. Zhao, D. Zeng, A.J. Rush and M. R. Kosorok, Estimating Individualized Treatment 
           Rules Using Outcome Weighted Learning, Journal of the American Statistical Association, 
           107:499, 1106-1118, 2012, DOI: 10.1080/01621459.2012.695674.
    """
        
    def __init__(self) -> None:
        pass
     
    def __repr__(self) -> str:
    
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def learn_policy(self, data, clf: str = 'SVC', **kwargs) -> None:
    
        """
        Parameters
        ----------
        data : STBT object
            This must be a Supervised to Bandit Transform (STBT) class with fitted 
            `generate_batch` method.
            
        clf : str, default: 'SVC'
        A sklearn classification estimator. Must be one of 'LogisticRegression', 
        'LogisticRegressionCV', 'RandomForestClassifier', or 'SVC'.
       
        **kwargs : Arguments passed to clf.

        Returns
        -------
        int.
          The predicted best policy.
        
        """    
        
        wt = data.train_logging_reward / data.train_logging_prob
        
        if clf in ['SVC', 'RandomForestClassifier']:
            m = eval(clf)(**kwargs).fit(data.X_train, data.y_train_logging, sample_weight = wt)
        elif clf in ['LogisticRegression', 'LogisticRegressionCV']:
            m = eval(clf)(multi_class='multinomial', **kwargs).fit(data.X_train, data.y_train_logging, sample_weight = wt)
        
        self.est_best_policy = m.predict(data.X_test)
        
        return self
         

class VowpalWabbit(EvaluationMetrics):
    """
    Performs policy learning using Vowpal Wabbit. 
    
    Parameters
    ----------
    method : str, default: 'ips'
        The policy evaluation approach to optimize a policy. Vowpal Wabbit offers four 
        approaches to specify a contextual bandit approach:
            * Inverse Propensity Score: 'ips'
            * Doubly Robust: 'dr'
            * Direct Method: 'dm'
            * Multi Task Regression/Importance Weighted Regression: 'mtr' 
       
    References
    ----------
    
    .. [1] A. Bietti and A. Agarwal and J. Langford, A Contextual Bandit Bake-off, 
        arXiv preprint arXiv:1802.04064, 2018.
    
    """

    def __init__(self, method = 'dr') -> None:
        self.method = method
     
    def __repr__(self) -> str:
    
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def _train_vw(self, data):
        n_actions = len(np.unique(data.y_train_logging))
        vw = pyvw.vw(str("--cb_type") + " " +  self.method + " " + str(n_actions))
        for i in range(data.X_train.shape[0]):
            action = data.y_train_logging[i]
            cost = 1 - data.train_logging_reward[i] # input requires cost instead of reward
            probability = data.train_logging_prob[i]
            train_features_ls = list()
            for f in range(data.X_train.shape[1]):
                train_features_ls.append(str(data.X_train[i, f]))
                train_features = " ".join(train_features_ls)
            learn_example = str(action) + ":" + str(cost) + ":" + str(probability) + " | " + train_features
            vw.learn(learn_example) 
        return vw
    
    def _predict_vw(self, vw_object, data):
        test_features_ls = list()
        predictions = list()
        for i in range(data.X_test.shape[0]):
            for f in range(data.X_test.shape[1]):
                test_features_ls.append(str(data.X_test[i, f]))
                features = " ".join(test_features_ls)
            test_example = " | " + features
            pred = vw_object.predict(test_example) 
            predictions.append(pred)
        predictions = np.array(predictions)
    
        return predictions 
        
    
    def learn_policy(self, data) -> None:
        """
        Parameters
        ----------
        data : STBT object
            This must be a Supervised to Bandit Transform (STBT) class with fitted 
            `generate_batch` method.
            
        Returns
        -------
        int.
          The predicted best policy.
        
        """ 
        
        vw_fit = self._train_vw(data)
        self.est_best_policy = self._predict_vw(vw_fit, data)
        
        return self

class CounterfactualRiskMinimization(EvaluationMetrics):
    """
    Performs policy learning using the Counterfactual Risk Minimization 
    approach proposed in [1], and later refined in [2].
    
    Parameters
    ----------
    
    batch_size : int, default: 96
        The number of samples per batch to load 
    learning_rate : float, default: 0.01
        Stochastic gradient descent learning rate 
    weight_decay : float, default: 0.001
        L2 regularization on parameters
    lambda_ : float, default: 0.1
        Variance regularization. Penalty on the variance of the 
        learnt policy relative to the logging policy.
    self_normalize: bool, default: True
        Whether to normalize the IPS estimator. See [2].
    clipping: float, default: 100.
        Clipping the importance sample weights. See [1].
    verbose: bool, default: False
        Whether to print Poem Loss during training .
    
    References
    ----------
    
    .. [1] A. Swaminathan and T. Joachims, Batch Learning from Logged Bandit Feedback through 
            Counterfactual Risk Minimization, Journal of Machine Learning Research, 16(52),
            1731--1755, 2015.
    .. [2] A. Swaminathan and T. Joachims, The self-normalized estimator for counterfactual learning, 
            Advances in Neural Information Processing Systems, 28, 16(52), 3231--3239, 2015.
    .. [3] A. Swaminathan, T. Joachims, and M. de Rijke, Deep Learning with Logged Bandit Feedback, 
            International Conference on Learning Representations,  2018.
    
    """

    
    def __init__(self, batch_size: int = 96, learning_rate: float = 0.001, weight_decay: float = 0.001, 
                 lambda_: float = 0.5, self_normalize: bool = True, clipping : float = 100.,
                 verbose: bool = False) -> None:
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        self.self_normalize = self_normalize
        self.clipping = clipping 
        self.verbose = verbose
       
    def __repr__(self) -> str:
    
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def _poem_loss(self, pi, p0, r, y_idx, Lambda):
        
        if torch.sum(r) == 0: 
            r = torch.repeat_interleave(torch.tensor(1e-05, dtype=torch.float), len(r))
        
        bsz = pi.shape[0]
        softmax_pi = F.softmax(pi, dim=1)
        pi_i = softmax_pi.masked_select(y_idx)
        log_importance = torch.log(pi_i) - torch.log(p0) 
        importance = torch.exp(log_importance)    
        clip_importance_vals = torch.repeat_interleave(torch.tensor(self.clipping, dtype=torch.float), len(importance))
        importance = torch.min(clip_importance_vals, importance)
        off_policy_est = torch.mul(importance, r)
        # Eq.(8) in [2] 
        var_n = torch.sum(torch.mul(torch.pow(torch.sub(r, off_policy_est), 2), torch.pow(torch.div(pi_i, p0), 2)))
        var_d = torch.pow(torch.sum(torch.div(pi_i, p0)), 2)
        empirical_var = torch.div(var_n, var_d)
        if self.self_normalize:
            effective_sample_size = torch.sum(importance).detach() # turns off requires grad
            mean_off_policy_est = torch.div(torch.sum(off_policy_est), effective_sample_size) 
        else:
            mean_off_policy_est = torch.mean(off_policy_est)
        
        penalty = torch.mul(Lambda, torch.sqrt(torch.div(empirical_var, bsz)))
        loss = torch.mul(-1.0, mean_off_policy_est) + penalty
        
        return loss

   
    
    def learn_policy(self, model, data, epochs: int = 500) -> None:

        """
        Parameters
        ----------
        data : STBT object
            This must be a Supervised to Bandit Transform (STBT) class with fitted 
            `generate_batch` method.
        epochs : int, default 
            Number of training epochs.
            
        Returns
        -------
        int.
          The predicted best policy.
        
        """ 
        
        train_ds = BanditDataset(torch.from_numpy(data.X_train).float(),
                                 torch.from_numpy(data.y_train_logging).long(), 
                                 torch.from_numpy(data.train_logging_prob).float(), 
                                 torch.from_numpy(data.train_logging_reward).long(),
                                 torch.from_numpy(data.y_train_logging_idx).bool())
        
        
        n_features = train_ds.X.shape[1]
        actions = torch.unique(train_ds.y)
        n_actions = len(actions)
       
        train_dl = DataLoader(train_ds, self.batch_size)
        
        Model = model(n_features = n_features, n_actions = n_actions)
        
        optimizer = torch.optim.Adam(Model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
       
        for epoch in range(epochs):
            Model.train()
            train_epoch_loss = 0.
            for x_batch,y_batch,p0_batch,r_batch,y_idx_batch in train_dl:
                pi = Model(x_batch)
                loss = self._poem_loss(pi, p0_batch, r_batch, y_idx_batch, self.lambda_)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_epoch_loss += loss.item()
            if self.verbose:
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}: | Train Poem Loss: {train_epoch_loss/len(train_dl):.5f}')
            
        Model.eval()
        with torch.no_grad():
            X_test = torch.from_numpy(data.X_test).float()
            pred = Model(X_test)
            est_best_policy = actions[torch.argmax(pred, dim=1)]
            
        self.est_best_policy = est_best_policy.numpy()
         
        return self
    
    

class CounterfactualRiskMinimizationCV(CounterfactualRiskMinimization, EvaluationMetrics):
    """
    Tune variance penalty for Counterfactual Risk Minimization.
    
    Parameters
    ----------
    
    batch_size : int, default: 96
        The number of samples per batch to load 
    learning_rate : float, default: 0.01
        Stochastic gradient descent learning rate 
    weight_decay : float, default: 0.001
        L2 regularization on parameters
    self_normalize: bool, default: True
        Whether to normalize the IPS estimator. See [2].
    clipping: float, default: 100.
        Clipping the importance sample weights. See [1].
    verbose: bool, default: True
        Whether to print Poem Loss during training .
    lambda_ : 1D array, optional, defaults to grid of values 
        chosen in a logarithmic scale between 1e-4 and 1e+01.
    
    """
    
    def __init__(self, batch_size: int = 96, learning_rate: float = 0.001, weight_decay: float = 0.001, 
                 self_normalize: bool = True, clipping : float = 100., verbose: bool = False, 
                 lambda_: np.ndarray = None) -> None:
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.self_normalize = self_normalize
        self.clipping = clipping 
        self.verbose = verbose
        
        if lambda_ is None:
            self.lambda_ = 10 ** np.linspace(-4., 1., 10) # search in log scale
        else:
            self.lambda_= lambda_
                           
    def __repr__(self) -> str:
    
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
    def _get_params_min_loss(self, x):
        x = x.numpy()
        xmin_idx = np.unravel_index(x.argmin(), x.shape)
        l_best = self.lambda_[xmin_idx[0]]
          
        return l_best

    
    def learn_policy(self, model, data, valid_frac: float = 0.5, epochs: int = 500) -> None:
        """
        Parameters
        ----------
        data : STBT object
            This must be a Supervised to Bandit Transform (STBT) class with fitted 
            `generate_batch` method.
        valid_frac : float, default: 0.5
            Fraction of training data set for validation. Test data are not modified. 
        epochs : int, default: 500
            Number of training epochs.
            
        Returns
        -------
        int.
          The predicted best policy.
        
        """ 
        
        self.epochs = epochs
        self.valid_frac = valid_frac
        
        n_train_samples, n_features = data.X_train.shape
        idx_valid_samples = np.random.choice(range(n_train_samples), 
                                              size = int(np.floor(n_train_samples * valid_frac)), replace = False)
        
        train_ds = BanditDataset(torch.from_numpy(np.delete(data.X_train, idx_valid_samples, axis=0)).float(),
                                  torch.from_numpy(np.delete(data.y_train_logging, idx_valid_samples)).long(), 
                                  torch.from_numpy(np.delete(data.train_logging_prob, idx_valid_samples)).float(), 
                                  torch.from_numpy(np.delete(data.train_logging_reward, idx_valid_samples)).long(),
                                  torch.from_numpy(np.delete(data.y_train_logging_idx, idx_valid_samples, axis=0)).bool())
        
        
        valid_ds = BanditDataset(torch.from_numpy(data.X_train[idx_valid_samples, :]).float(),
                                  torch.from_numpy(data.y_train_logging[idx_valid_samples]).long(), 
                                  torch.from_numpy(data.train_logging_prob[idx_valid_samples]).float(), 
                                  torch.from_numpy(data.train_logging_reward[idx_valid_samples]).long(),
                                  torch.from_numpy(data.y_train_logging_idx[idx_valid_samples, :]).bool())
            
        y_train = np.delete(data.y_train, idx_valid_samples, axis=0)
        y_valid = data.y_train[idx_valid_samples]
        X_test = torch.from_numpy(data.X_test).float()
        
        actions = torch.unique(train_ds.y)
        n_actions = len(actions)
       
        train_dl = DataLoader(train_ds, self.batch_size)
        valid_dl = DataLoader(valid_ds, self.batch_size)
        
        Model = model(n_features=n_features, n_actions=n_actions)
        
        optimizer = torch.optim.Adam(Model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
           
        self.train_tot_loss_hist = torch.zeros(len(self.lambda_), epochs)   
        self.valid_tot_loss_hist = torch.zeros(len(self.lambda_), epochs)
        self.valid_acc = torch.zeros(len(self.lambda_), epochs) 
        self.train_acc = torch.zeros(len(self.lambda_), epochs) 
        self.test_acc = torch.zeros(len(self.lambda_), epochs) 
    
        for l_idx, l in enumerate(self.lambda_):
       
            for epoch in range(epochs):
                Model.train()
                train_epoch_loss = 0.
                for x_batch,y_batch,p0_batch,r_batch,y_idx_batch in train_dl:
                    pi = Model(x_batch)
                    loss = self._poem_loss(pi, p0_batch, r_batch, y_idx_batch, l)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_epoch_loss += loss.item()
                self.train_tot_loss_hist[l_idx, epoch] = train_epoch_loss/len(train_dl)
                if self.verbose:
                    if epoch % 100 == 0:
                        print(f'Epoch: {epoch} | Train Poem Loss: {train_epoch_loss/len(train_dl):.5f}')
                
                Model.eval()
                with torch.no_grad():
                    valid_tot_loss=0.
                    for x_batch,y_batch,p0_batch,r_batch,y_idx_batch in valid_dl:
                        pi = Model(x_batch)
                        valid_loss = self._poem_loss(pi, p0_batch, r_batch, y_idx_batch, l)
                        valid_tot_loss += valid_loss.item()
                self.valid_tot_loss_hist[l_idx, epoch] = valid_tot_loss/len(valid_dl)
                if self.verbose:
                      if epoch % 100 == 0:
                          print(f'Epoch: {epoch} | Valid Poem Loss: {valid_tot_loss/len(valid_dl):.5f}')
                
                pred_train = Model(train_ds.X)
                est_best_policy_train = actions[torch.argmax(pred_train, dim=1)]
                est_best_policy_train = est_best_policy_train.numpy()
                self.train_acc[l_idx, epoch] = self.error_rate(est_best_policy_train, y_train)
                pred_valid = Model(valid_ds.X)
                est_best_policy_valid = actions[torch.argmax(pred_valid, dim=1)]
                est_best_policy_valid = est_best_policy_valid.numpy()
                self.valid_acc[l_idx, epoch] = self.error_rate(est_best_policy_valid, y_valid)
                
                pred_test = Model(X_test)
                est_best_policy_test = actions[torch.argmax(pred_test, dim=1)]
                est_best_policy_test = est_best_policy_test.numpy()
                self.test_acc[l_idx, epoch] = self.error_rate(est_best_policy_test, data.y_test)
            
                          
        self.l_best = self._get_params_min_loss(self.valid_tot_loss_hist)
            
    
        
        crm = CounterfactualRiskMinimization(lambda_=self.l_best, batch_size = self.batch_size,
                                              learning_rate = self.learning_rate, weight_decay = self.weight_decay,
                                              clipping = self.clipping, self_normalize = self.self_normalize, verbose = self.verbose)
         
        crm.learn_policy(model=model, data=data, epochs=epochs) 
        self.est_best_policy = crm.est_best_policy
                
        return self
    
    def plot_cv_loss(self):
         
        train_loss_flatten = self.train_tot_loss_hist.T.flatten(1).numpy()
        valid_loss_flatten = self.valid_tot_loss_hist.T.flatten(1).numpy()
        
        train_acc_flatten =  self.train_acc.T.flatten(1).numpy()
        valid_acc_flatten = self.valid_acc.T.flatten(1).numpy()
        test_acc_flatten = self.test_acc.T.flatten(1).numpy()
        
        fig, axs = plt.subplots(2, 3)
        fs = 8
        
        for l_idx, l in enumerate(self.lambda_):
            axs[0, 0].plot(train_loss_flatten[:,l_idx], label = round(l, 4))
            axs[0, 1].plot(valid_loss_flatten[:,l_idx], label = round(l, 4))
            axs[1, 0].plot(train_acc_flatten[:,l_idx], label = round(l, 4))
            axs[1, 1].plot(valid_acc_flatten[:,l_idx], label = round(l, 4))
            axs[1, 2].plot(test_acc_flatten[:,l_idx], label = round(l, 4))
            
        axs[0, 0].set_title("Train: Poem Loss", fontsize=fs)
        axs[0, 1].set_title("Validation: Poem Loss", fontsize=fs)
        axs[1, 0].set_title("Train: Accuracy", fontsize=fs)
        axs[1, 1].set_title("Validation: Accuracy", fontsize=fs)
        axs[1, 2].set_title("Test: Accuracy", fontsize=fs)
        
        for i, ax in enumerate(axs.flat):
            if i < 2:
                ax.set_xlabel(xlabel='Epoch', fontsize=fs)
                ax.set_ylabel(ylabel='Loss', fontsize=fs)
            else:
                ax.set_xlabel(xlabel='Epoch', fontsize=fs)
                ax.set_ylabel(ylabel='Accuracy', fontsize=fs)
        fig.legend(self.lambda_, loc='upper right', fontsize=fs)
  
    


# Alternative using Doubly Robust (as opposed to the IPS estimator)            
        
# class CounterfactualRiskMinimization(EvaluationMetrics):
#     """
#     Performs policy learning using the Counterfactual Risk Minimization 
#     approach proposed in [1], and later refined in [2].
    
#     Parameters
#     ----------
#     batch_size : int, default: 96
#         The number of samples per batch to load 
#     learning_rate : float, default: 0.01
#         Stochastic gradient descent learning rate 
#     weight_decay : float, default: 0.001
#         L2 regularization on parameters
#     lambda_ : float, default: 0.1
#         Variance regularization. Penalty on the variance of the 
#         learnt policy relative to the logging policy.
#     self_normalize: bool, default: True
#         Whether to normalize the IPS estimator. See [2].
#     clipping: float, default: 100.
#         Clipping the importance sample weights. See [1].
#     verbose: bool, default: False
#         Whether to print Poem Loss during training .
    
#     References
#     ----------
    
#     .. [1] A. Swaminathan and T. Joachims, Batch Learning from Logged Bandit Feedback through 
#            Counterfactual Risk Minimization, Journal of Machine Learning Research, 16(52),
#            1731--1755, 2015.
#     .. [2] A. Swaminathan and T. Joachims, The self-normalized estimator for counterfactual learning, 
#            Advances in Neural Information Processing Systems, 28, 16(52), 3231--3239, 2015.
#     .. [3] A. Swaminathan, T. Joachims, and M. de Rijke, Deep Learning with Logged Bandit Feedback, 
#            International Conference on Learning Representations,  2018.
    
#     """

    
#     def __init__(self, batch_size: int = 96, learning_rate: float = 0.01, weight_decay: float = 0.001, 
#                  lambda_: float = 0.5, self_normalize: bool = True, clipping : float = 100.,
#                  verbose: bool = False) -> None:
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.lambda_ = lambda_
#         self.self_normalize = self_normalize
#         self.verbose = verbose
#         self.clipping = clipping 
     
#     def __repr__(self) -> str:
    
#         items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
#         return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
  
#     def _poem_loss(self, pi, p0, r, r_pred, y_idx, Lambda, self_normalize):
        
#         #if torch.sum(r) == 0: 
#         #    r = torch.repeat_interleave(torch.tensor(1e-05, dtype=torch.float), len(r))
        
        
#         bsz = pi.shape[0]
#         softmax_pi = F.softmax(pi, dim=1)
#         pi_i = softmax_pi.masked_select(y_idx)
#         r_pred_i = r_pred.masked_select(y_idx)
#         importance = torch.div(pi_i, p0) 
#         clip_importance_vals = torch.repeat_interleave(torch.tensor(self.clipping, dtype=torch.float), len(importance))
#         importance_clipped = torch.min(clip_importance_vals, importance)
#         reward_residual = torch.sub(r, r_pred_i)
#         weighted_reward_pred = torch.sum(torch.mul(softmax_pi, r_pred), dim=1)
#         off_policy_est = torch.add(torch.mul(importance_clipped, reward_residual), weighted_reward_pred)
#         empirical_var = torch.var(off_policy_est)
        
#         if self_normalize:
#             effective_sample_size = torch.sum(importance_clipped).detach() # turns off requires grad
#             sum_off_policy_est = torch.div(torch.sum(off_policy_est), effective_sample_size) 
#         else:
#             sum_off_policy_est = torch.sum(off_policy_est)
        
#         penalty = torch.mul(Lambda, torch.sqrt(torch.div(empirical_var, bsz)))
#         loss = torch.mul(-1.0, sum_off_policy_est) + penalty
             
#         return loss

    
#     def learn_policy(self, model, data, epochs: int = 500) -> None:

#         """
#         Parameters
#         ----------
#         data : STBT object
#             This must be a Supervised to Bandit Transform (STBT) class with fitted 
#             `generate_batch` method.
#         epochs : int, default 
#             Number of training epochs.
            
#         Returns
#         -------
#         int.
#           The predicted best policy.
        
#         """ 
        
#         rp = RewardPredictor().learn_policy(data=data, max_iter=1000)
        
#         train_ds = BanditDataset(torch.from_numpy(data.X_train).float(),
#                                  torch.from_numpy(data.y_train_logging).long(), 
#                                  torch.from_numpy(data.train_logging_prob).float(), 
#                                  torch.from_numpy(data.train_logging_reward).long(),
#                                  torch.from_numpy(data.y_train_logging_idx).bool(),
#                                  torch.from_numpy(rp.train_pred_reward_arr).float() 
#                                  )
        
        
#         n_features = train_ds.X.shape[1]
#         actions = torch.unique(train_ds.y)
#         n_actions = len(actions)
       
#         train_dl = DataLoader(train_ds, self.batch_size)
        
#         Model = model(n_features = n_features, n_actions = n_actions)
        
#         optimizer = torch.optim.SGD(Model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
        
#         for epoch in range(epochs):
#             Model.train()
#             train_epoch_loss = 0.
#             for x_batch, y_batch, p0_batch, r_batch, y_idx_batch, r_pred_batch in train_dl:
#                 pi = Model(x_batch)
#                 loss = self._poem_loss(pi, p0_batch, r_batch, r_pred_batch, y_idx_batch, self.lambda_, self.self_normalize)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 train_epoch_loss += loss.item()
#             if self.verbose:
#                 print(f'Epoch {epoch}: | Train Poem Loss: {train_epoch_loss/len(train_dl):.5f}')
            
#         Model.eval()
#         with torch.no_grad():
#            X_test = torch.from_numpy(data.X_test).float()
#            pred = Model(X_test)
#            est_best_policy = actions[torch.argmax(pred, dim=1)]
            
#         self.est_best_policy = est_best_policy.numpy()
         
#         return self
    
    

# class CounterfactualRiskMinimizationCV(CounterfactualRiskMinimization, EvaluationMetrics):
#     """
#     Tune variance regularizer for Counterfactual Risk Minimization.
    
#     Parameters
#     ----------
    
#     batch_size : int, default: 96
#         The number of samples per batch to load 
#     learning_rate : float, default: 0.01
#         Stochastic gradient descent learning rate 
#     weight_decay : float, default: 0.001
#         L2 regularization on parameters
#     clipping: float, default: 100.
#         Clipping the importance sample weights. See [1].
#     self_normalize: bool, default: True
#         Whether to normalize the IPS estimator. See [2].
#     verbose: bool, default: True
#         Whether to print Poem Loss during training .
#     lambda_ : 1D array, optional, defaults to grid of values 
#         chosen in a logarithmic scale between 1e-4 and 1e+01.
    
#     """
    
#     def __init__(self, batch_size: int = 96, learning_rate: float = 0.01, weight_decay: float = 0.001, 
#                  clipping : float = 100., self_normalize: bool = True, verbose: bool = False, 
#                  lambda_: np.ndarray = None) -> None:
        
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.clipping = clipping 
#         self.self_normalize = self_normalize
#         self.verbose = verbose
        
#         if lambda_ is None:
#             self.lambda_ = 10 ** np.linspace(-4., 1., 10) # search in log scale
#         else:
#             self.lambda_= lambda_
                           
#     def __repr__(self) -> str:
    
#         items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
#         return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
    
#     def _get_params_min_loss(self, x):
#         x = x.numpy()
#         xmin_idx = np.unravel_index(x.argmin(), x.shape)
#         l_best = self.lambda_[xmin_idx[0]]
          
#         return l_best

    
#     def learn_policy(self, model, data, valid_frac: float = 0.5, epochs: int = 500) -> None:
#         """
#         Parameters
#         ----------
#         data : STBT object
#             This must be a Supervised to Bandit Transform (STBT) class with fitted 
#             `generate_batch` method.
#         valid_frac : float, default: 0.5
#             Fraction of training data set for validation. Test data are not modified. 
#         epochs : int, default: 500
#             Number of training epochs.
            
#         Returns
#         -------
#         int.
#           The predicted best policy.
        
#         """ 
        
#         self.epochs = epochs
#         self.valid_frac = valid_frac
        
#         rp = RewardPredictor().learn_policy(data=data, max_iter=1000)
        
#         n_train_samples, n_features = data.X_train.shape
#         idx_valid_samples = np.random.choice(range(n_train_samples), 
#                                              size = int(np.floor(n_train_samples * valid_frac)), replace = False)
        
#         train_ds = BanditDataset(torch.from_numpy(np.delete(data.X_train, idx_valid_samples, axis=0)).float(),
#                                  torch.from_numpy(np.delete(data.y_train_logging, idx_valid_samples)).long(), 
#                                  torch.from_numpy(np.delete(data.train_logging_prob, idx_valid_samples)).float(), 
#                                  torch.from_numpy(np.delete(data.train_logging_reward, idx_valid_samples)).long(),
#                                  torch.from_numpy(np.delete(data.y_train_logging_idx, idx_valid_samples, axis=0)).bool(), 
#                                  torch.from_numpy(np.delete(rp.train_pred_reward_arr, idx_valid_samples, axis=0)).float()
#                                  )
        
        
#         valid_ds = BanditDataset(torch.from_numpy(data.X_train[idx_valid_samples, :]).float(),
#                                  torch.from_numpy(data.y_train_logging[idx_valid_samples]).long(), 
#                                  torch.from_numpy(data.train_logging_prob[idx_valid_samples]).float(), 
#                                  torch.from_numpy(data.train_logging_reward[idx_valid_samples]).long(),
#                                  torch.from_numpy(data.y_train_logging_idx[idx_valid_samples, :]).bool(),
#                                  torch.from_numpy(rp.train_pred_reward_arr[idx_valid_samples, :]).float()
#                                  )
            
#         y_train = np.delete(data.y_train, idx_valid_samples, axis=0)
#         y_valid = data.y_train[idx_valid_samples]
#         X_test = torch.from_numpy(data.X_test).float()
        
#         actions = torch.unique(train_ds.y)
#         n_actions = len(actions)
       
#         train_dl = DataLoader(train_ds, self.batch_size)
#         valid_dl = DataLoader(valid_ds, self.batch_size)
        
#         Model = model(n_features=n_features, n_actions=n_actions)
        
#         optimizer = torch.optim.SGD(Model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
           
#         self.train_tot_loss_hist = torch.zeros(len(self.lambda_), epochs)   
#         self.valid_tot_loss_hist = torch.zeros(len(self.lambda_), epochs)
#         self.valid_acc = torch.zeros(len(self.lambda_), epochs) 
#         self.train_acc = torch.zeros(len(self.lambda_), epochs) 
#         self.test_acc = torch.zeros(len(self.lambda_), epochs) 
    
#         for l_idx, l in enumerate(self.lambda_):
       
#             for epoch in range(epochs):
#                 Model.train()
#                 train_epoch_loss = 0.
#                 for x_batch, y_batch, p0_batch,r_batch,y_idx_batch,r_pred_batch in train_dl:
#                     pi = Model(x_batch)
#                     loss = self._poem_loss(pi, p0_batch, r_batch, r_pred_batch, y_idx_batch, l, self.self_normalize)
#                     loss.backward()
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     train_epoch_loss += loss.item()
#                 self.train_tot_loss_hist[l_idx, epoch] = train_epoch_loss/len(train_dl)
#                 if self.verbose:
#                     print(f'Epoch: {epoch} | Train Poem Loss: {train_epoch_loss/len(train_dl):.5f}')
                
#                 Model.eval()
#                 with torch.no_grad():
#                     valid_tot_loss=0.
#                     for x_batch,y_batch,p0_batch,r_batch,y_idx_batch,r_pred_batch in valid_dl:
#                         pi = Model(x_batch)
#                         valid_loss = self._poem_loss(pi, p0_batch, r_batch, r_pred_batch, y_idx_batch, l, self.self_normalize)
#                         valid_tot_loss += valid_loss.item()
#                 self.valid_tot_loss_hist[l_idx, epoch] = valid_tot_loss/len(valid_dl)
#                 if self.verbose:
#                       print(f'Epoch: {epoch} | Valid Poem Loss: {valid_tot_loss/len(valid_dl):.5f}')
                
#                 pred_train = Model(train_ds.X)
#                 est_best_policy_train = actions[torch.argmax(pred_train, dim=1)]
#                 est_best_policy_train = est_best_policy_train.numpy()
#                 self.train_acc[l_idx, epoch] = self.error_rate(est_best_policy_train, y_train)
#                 pred_valid = Model(valid_ds.X)
#                 est_best_policy_valid = actions[torch.argmax(pred_valid, dim=1)]
#                 est_best_policy_valid = est_best_policy_valid.numpy()
#                 self.valid_acc[l_idx, epoch] = self.error_rate(est_best_policy_valid, y_valid)
                
#                 pred_test = Model(X_test)
#                 est_best_policy_test = actions[torch.argmax(pred_test, dim=1)]
#                 est_best_policy_test = est_best_policy_test.numpy()
#                 self.test_acc[l_idx, epoch] = self.error_rate(est_best_policy_test, data.y_test)
            
                          
#         self.l_best = self._get_params_min_loss(self.valid_tot_loss_hist)
            
    
        
#         crm = CounterfactualRiskMinimization(lambda_=self.l_best, batch_size = self.batch_size,
#                                              learning_rate = self.learning_rate, weight_decay = self.weight_decay,
#                                              clipping = self.clipping, self_normalize = self.self_normalize, verbose = self.verbose)
        
#         crm.learn_policy(model=model, data=data, epochs=epochs) 
#         self.est_best_policy = crm.est_best_policy
                
#         return self
    
#     def plot_cv_loss(self):
         
#         train_loss_flatten = self.train_tot_loss_hist.T.flatten(1).numpy()
#         valid_loss_flatten = self.valid_tot_loss_hist.T.flatten(1).numpy()
        
#         train_acc_flatten =  self.train_acc.T.flatten(1).numpy()
#         valid_acc_flatten = self.valid_acc.T.flatten(1).numpy()
#         test_acc_flatten = self.test_acc.T.flatten(1).numpy()
        
#         fig, axs = plt.subplots(2, 3)
#         fs = 8
        
#         for l_idx, l in enumerate(self.lambda_):
#             axs[0, 0].plot(train_loss_flatten[:,l_idx], label = round(l, 4))
#             axs[0, 1].plot(valid_loss_flatten[:,l_idx], label = round(l, 4))
#             axs[1, 0].plot(train_acc_flatten[:,l_idx], label = round(l, 4))
#             axs[1, 1].plot(valid_acc_flatten[:,l_idx], label = round(l, 4))
#             axs[1, 2].plot(test_acc_flatten[:,l_idx], label = round(l, 4))
            
#         axs[0, 0].set_title("Train: Poem Loss", fontsize=fs)
#         axs[0, 1].set_title("Validation: Poem Loss", fontsize=fs)
#         axs[1, 0].set_title("Train: Accuracy", fontsize=fs)
#         axs[1, 1].set_title("Validation: Accuracy", fontsize=fs)
#         axs[1, 2].set_title("Test: Accuracy", fontsize=fs)
        
#         for i, ax in enumerate(axs.flat):
#             if i < 2:
#                 ax.set_xlabel(xlabel='Epoch', fontsize=fs)
#                 ax.set_ylabel(ylabel='Loss', fontsize=fs)
#             else:
#                 ax.set_xlabel(xlabel='Epoch', fontsize=fs)
#                 ax.set_ylabel(ylabel='Accuracy', fontsize=fs)
#         fig.legend(self.lambda_, loc='upper right', fontsize=fs)
       
