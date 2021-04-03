"""Sample datasets used in experiments."""

import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, fetch_openml


def get_data(dataset: str = None, scale: bool = True) -> tuple:
    """Get data (features and labels) used in experiments.
    
    Parameters
    ----------
    
    dataset : str, default: None 
        It should be one of: 'ecoli', 'glass', 'letter-recognition', 
        'lymphography', 'yeast', 'digits', 'breast-cancer', 'wine', or 
        'mnist'.
        
    scale : bool, default: True
        Standardize features by zero mean and unit variance.
        
    Returns
    -------
    
    tuple, length=2 
        tuple containing features-target split of inputs.

    References
    ----------
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
    Irvine, CA: University of California, School of Information and Computer Science.
    
    Examples
    --------
    >>> X, y = get_data(dataset='ecoli')
    >>> X[0,:]
    array([0.49, 0.29, 0.48, 0.5 , 0.56, 0.24, 0.35])
    
    """
    
    if dataset not in ['ecoli', 'glass', 'letter-recognition', 'lymphography', 'yeast', 
                       'digits', 'breast-cancer', 'wine', 'mnist']:
        raise ValueError("Invalid dataset provided.")
    
    if dataset in dataset in ['ecoli', 'glass', 'letter-recognition', 'lymphography', 'yeast']:
        path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' 
        f = path + dataset + "/" + dataset + ".data"
     
    if dataset in  ['ecoli', 'yeast']:
        df = pd.read_table(f, delim_whitespace=True, header=None)
    elif dataset in [ 'glass', 'letter-recognition', 'lymphography']:
        df = pd.read_csv(f, header=None)
    elif dataset == 'digits':
        df = load_digits()
        X = df.data
        y = df.target
    elif dataset == 'breast-cancer':
        df = load_breast_cancer()
        X = df.data
        y = df.target
    elif dataset == 'wine':
        df = load_wine()
        X = df.data
        y = df.target
        
    if dataset == 'ecoli':
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:,-1])
        X = df.iloc[:,1:8].values
        
    elif dataset == 'glass':
        y = df.iloc[:,-1].values
        X = df.iloc[:, 1:(df.shape[1]-1)].values
        
    elif dataset in ['letter-recognition', 'lymphography']:
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:,0])
        X = df.iloc[:, 1:(df.shape[1])].values
     
    elif dataset == 'yeast':
        y = preprocessing.LabelEncoder().fit_transform(df.iloc[:,-1])
        X = df.iloc[:,1:9].values
        
    elif dataset == 'mnist':
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype('int64') 

    if scale==True:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y

    