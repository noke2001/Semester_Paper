import numpy as np
from sklearn.ensemble import RandomForestRegressor as _RFRegressor, RandomForestClassifier as _RFClassifier
import lightgbm as lgbm

__all__ = [
    "RandomForestRegressor",
    "RandomForestClassifier",
    "LGBMRegressor",
    "LGBMClassifier"
]

class RandomForestRegressor:
 
    def __init__(self, **kwargs):
        self.model = _RFRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class RandomForestClassifier:

    def __init__(self, **kwargs):
        self.model = _RFClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LGBMRegressor:

    def __init__(self, **kwargs):
        self.model = lgbm.LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class LGBMClassifier:
    def __init__(self, **kwargs):
        self.model = lgbm.LGBMClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
