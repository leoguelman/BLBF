# Batch Learning from Bandit Feedback (BLBF)

* Implementation of various methods for Policy Evaluation and Learning in the context of Batch Learning from Bandit Feedback. 
* Perform Supervised to Bandit Conversion for classification tasks. This conversion is generally used to test the limits of counterfactual learning in well-controlled environments.

## Policy Evaluation methods [5, 6], include:

* IPS: Inverse Propensity Score 
* DM: Direct Method (Reward Prediction)
* DR: Doubly Robust
* SWITCH: Switch Estimator

## Policy Learning methods, include:
* Outcome Weighted Learning: Performs policy learning by transforming the learning problem into a 
    weighted multi-class classification problem [4].
* Counterfactual Risk Minimization: Performs policy learning using the Counterfactual Risk Minimization 
    approach proposed in [1], and later refined in [2].
* Reward Predictor: Performs policy learning by directly predicting the Reward as a function of covariates, 
    actions and their interaction [7].

## References

[1] A. Swaminathan and T. Joachims, Batch Learning from Logged Bandit Feedback through 
           Counterfactual Risk Minimization, Journal of Machine Learning Research, 16(52),
           1731--1755, 2015.

[2] A. Swaminathan and T. Joachims, The self-normalized estimator for counterfactual learning, 
           Advances in Neural Information Processing Systems, 28, 16(52), 3231--3239, 2015.

[3] A. Swaminathan, T. Joachims, and M. de Rijke, Deep Learning with Logged Bandit Feedback, 
           International Conference on Learning Representations,  2018.

[4] Y. Zhao, D. Zeng, A.J. Rush and M. R. Kosorok, Estimating Individualized Treatment 
           Rules Using Outcome Weighted Learning, Journal of the American Statistical Association, 
           107:499, 1106-1118, 2012, DOI: 10.1080/01621459.2012.695674.

[5] Y. Wang, A. Agarwal and M. Dud\'{\i}k, Optimal and Adaptive Off-policy Evaluation in Contextual Bandits, 
               Proceedings of Machine Learning Research, 70, 3589--3597, 2017.
          
[6] M. Dudik and J. Langford and  L. Li, Doubly Robust Policy Evaluation and Learning, CoRR, 2011.
     http://arxiv.org/abs/1103.4601
     
[7] K{\"u}nzel, S., Sekhon, J., Bickel, P. and Yu, B., Metalearners for estimating heterogeneous 
      treatment effects using machine learning, Proceedings of the National Academy of Sciences, 
       116(10), 4156--4165, 2019. 
               
