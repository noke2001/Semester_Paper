import numpy as np
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import LogisticRegression as _LogisticRegression

__all__ = ["LinearRegressor", "LogisticRegressor", "ConstantPredictor"]

class LinearRegressor:
    """
    Wrapper for sklearn.linear_model.LinearRegression.
    """
    def __init__(self, **kwargs):
        self.model = _LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise AttributeError("LinearRegressor does not support predict_proba")

class LogisticRegressor:
    """
    Wrapper for sklearn.linear_model.LogisticRegression.
    """
    def __init__(self, **kwargs):
        self.model = _LogisticRegression(max_iter=1000,**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # return probability of the positive class
        return self.model.predict_proba(X)


class ConstantPredictorTabz:
    def __init__(self):
        self.constant_ = None
        self.is_classification = False
        self.classes_ = None   
        self.class_    = None 

    def fit(self, X, y):
        y_arr = np.asarray(y).ravel()
        unique_vals, counts = np.unique(y_arr, return_counts=True)

        if np.issubdtype(y_arr.dtype, np.integer) and unique_vals.size > 1:
            self.is_classification = True
            self.classes_ = unique_vals
            self.class_ = unique_vals[np.argmax(counts)]
            self.constant_ = np.mean(y_arr)
        else:
            self.is_classification = False
            self.classes_ = None
            self.class_    = None
            self.constant_ = np.mean(y_arr)

        return self

    def predict(self, X):
        if self.constant_ is None and not self.is_classification:
            raise ValueError("ConstantPredictorTabz has not been fitted yet.")
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = int(X)

        if self.is_classification:
            return np.full(n_samples, self.class_, dtype=self.classes_.dtype)
        else:
            return np.full(n_samples, self.constant_, dtype=float)

    def predict_proba(self, X):
        if not self.is_classification:
            raise AttributeError("predict_proba is only available for classification tasks.")
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = int(X)

        K = len(self.classes_)
        proba = np.zeros((n_samples, K), dtype=float)
        idx = np.flatnonzero(self.classes_ == self.class_)[0]
        proba[:, idx] = 1.0
        return proba

    


class ConstantPredictor:

    def __init__(self):
        self.constant_         = None
        self.class_            = None
        self.is_classification = False

    def fit(self, X, y):
        y_arr = np.asarray(y).ravel()
        self.constant_ = np.mean(y_arr)

        unique_vals = np.unique(y_arr)
        if np.issubdtype(y_arr.dtype, np.integer) and len(unique_vals) > 1:
            self.is_classification = True
            counts = np.bincount(y_arr.astype(int))
            self.class_ = np.argmax(counts)
        else:
            self.is_classification = False
            self.class_ = float(unique_vals[0])
        return self

    def predict(self, X):
        if self.constant_ is None:
            raise ValueError("ConstantPredictor has not been fitted yet.")
        n = X.shape[0] if hasattr(X, "shape") else int(X)
        if self.is_classification:
            return np.full(n, self.class_, dtype=int)
        else:
            return np.full(n, self.constant_, dtype=float)

    def predict_proba(self, X):
        if not self.is_classification:
            raise AttributeError("predict_proba is only for classification.")
        n = X.shape[0] if hasattr(X, "shape") else int(X)
        return np.full(n, self.constant_, dtype=float)
