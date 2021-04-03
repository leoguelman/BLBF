import numpy as np
import sklearn.preprocessing


def create_interactions(X: np.ndarray, T: np.ndarray, one_hot_labeler=None) -> tuple:
    if one_hot_labeler is None:
        lb_fit = sklearn.preprocessing.LabelBinarizer().fit(T)
    else:
        lb_fit = one_hot_labeler
    T = lb_fit.transform(T)
    XT = np.zeros(shape=[X.shape[0], X.shape[1] * T.shape[1]]) * np.nan
    cnt = 0
    for i in range(X.shape[1]):
        for j in range(T.shape[1]):
            XT[:,cnt]= X[:, i] * T[:, j]
            cnt += 1
    X_full = np.column_stack((X, T, XT))
    return X_full, lb_fit
