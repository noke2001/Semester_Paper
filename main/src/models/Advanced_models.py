import warnings
from src.models.drf.code import drf
import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, LogisticGAM
import gpboost as gpb
from gpboost import GPModel
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from rpy2.robjects import numpy2ri, pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter

combined_converter = default_converter + numpy2ri.converter + pandas2ri.converter

# ——————————————
# Regression wrappers
# ——————————————

class DistributionalRandomForestRegressor:
    def __init__(self, *, num_trees=100, mtry=None, min_node_size=5, seed=0, **kwargs):
        self.num_trees = num_trees
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.seed = seed
        self.kwargs = kwargs
        self._model = None

    def fit(self, X, y): 
        drf_hyperparams = {
            'num_trees': self.num_trees,
            'mtry': self.mtry if self.mtry is not None else X.shape[1],
            'min_node_size': self.min_node_size,
            'seed': self.seed
        }
        drf_hyperparams.update(self.kwargs)
        self._model = drf(**drf_hyperparams) 
        self._model.fit(X, y) 
        return self

    def predict_quantiles(self, X, quantiles):
        if self._model is None: raise RuntimeError("Model not fitted.")
        with localconverter(combined_converter):
            return self._model.predict(newdata=X, functional="quantile", quantiles=quantiles)

    def predict(self, X):
        q = self.predict_quantiles(X, quantiles=[0.5])
        return q.quantile[:, 0]

class LightGBMLSSRegressor:
    def __init__(self, *, distribution=None, **opts):
        self.distribution = distribution or Gaussian(stabilization="None", response_fn="exp", loss_fn="crps")
        self.opts = opts
        self._model = None

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, label=y)
        num_round = self.opts.pop("n_estimators", self.opts.pop("num_boost_round", 100))
        self.opts.setdefault("feature_pre_filter", False)
        self._model = LightGBMLSS(self.distribution)
        self._model.train(self.opts, dtrain, num_boost_round=num_round)
        return self

    def predict_parameters(self, X):
        return self._model.predict(X, pred_type="parameters")

# ——————————————
# GPBoost Wrappers (FIXED & DEBUGGED)
# ——————————————

class GPBoostRegressor:
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="gaussian", trace=False, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None: cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx = gp_approx
        self.likelihood = likelihood
        self.seed = seed
        self.kwargs = kw
        self._model = None
        self.trace = trace

    def fit(self, X, y):
        intercept = np.ones(len(y))
        gp_kwargs = {
            "gp_coords": X, "gp_approx": self.gp_approx,
            "cov_function": self.cov_function, "likelihood": self.likelihood,
            "seed": self.seed, **self.kwargs
        }
        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        
        # Robust Inducing Points
        if gp_kwargs.get("gp_approx") == "full_scale_vecchia":
            n = len(y)
            try:
                # Fast unique check
                X_cont = np.ascontiguousarray(X)
                dtype = np.dtype((np.void, X_cont.dtype.itemsize * X_cont.shape[1]))
                n_unique = len(np.unique(X_cont.view(dtype)))
            except: n_unique = n
            
            limit = min(n, n_unique)
            safe_ind = max(10, limit - 1)
            user_val = self.kwargs.get("num_ind_points", safe_ind)
            gp_kwargs["num_ind_points"] = min(user_val, safe_ind)

        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": self.trace}) 
        return self

    def predict(self, X, return_var=False):
        intercept = np.ones(X.shape[0])
        # Gaussian allows predict_response=True
        out = self._model.predict(
            gp_coords_pred=X, X_pred=intercept,
            predict_var=return_var, predict_response=True 
        )
        mu = out["mu"]
        if return_var: return mu, out["var"]
        return mu


class GPBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="bernoulli_logit", 
                 matrix_inversion_method=None, trace=False, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None: cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx = gp_approx
        self.likelihood = likelihood
        self.matrix_inversion_method = matrix_inversion_method
        self.seed = seed
        self.kwargs = kw
        self._model = None
        self.trace = trace
        
        # DEBUG PRINT TO CONFIRM NEW CODE IS LOADED
        # print(">> GPBoostClassifier initialized (Safe Mode)") 

    def fit(self, X, y):
        intercept = np.ones(len(y))
        method = self.matrix_inversion_method or "iterative"
        if self.gp_approx == "fitc" and self.likelihood == "bernoulli_logit":
            method = "cholesky"

        gp_kwargs = {
            "gp_coords": X, "gp_approx": self.gp_approx,
            "cov_function": self.cov_function, "likelihood": self.likelihood,
            "matrix_inversion_method": method, "seed": self.seed, **self.kwargs
        }
        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        
        if gp_kwargs.get("gp_approx") == "full_scale_vecchia":
            n = len(y)
            try:
                X_cont = np.ascontiguousarray(X)
                dtype = np.dtype((np.void, X_cont.dtype.itemsize * X_cont.shape[1]))
                n_unique = len(np.unique(X_cont.view(dtype)))
            except: n_unique = n
            
            limit = min(n, n_unique)
            safe_ind = max(10, limit - 1)
            user_val = self.kwargs.get("num_ind_points", safe_ind)
            gp_kwargs["num_ind_points"] = min(user_val, safe_ind)
            
        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": self.trace})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        
        # --- CRITICAL FIX ----------------------------------------------------
        # predict_response=False is MANDATORY for Vecchia + Bernoulli.
        # This returns LOGITS (latent variable 'mu').
        out = self._model.predict(
            gp_coords_pred=X, X_pred=intercept,
            predict_var=False, predict_response=False 
        )
        mu_logits = out["mu"]
        # ---------------------------------------------------------------------
        
        # Manual Sigmoid: 1 / (1 + exp(-x))
        # Clipping -30/+30 prevents overflow in exp() while keeping probabilities 0.0 or 1.0
        mu_logits = np.clip(mu_logits, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-mu_logits))
        
        return np.vstack([1 - prob, prob]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

class GPBoostMulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._model = None
        self._label_encoder = LabelEncoder()

    def fit(self, X, y):
        self._model = OneVsRestClassifier(GPBoostClassifier(**self.kwargs))
        y_encoded = self._label_encoder.fit_transform(y)
        self._model.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# ——————————————
# Safe GPBoost Wrappers (FIXED & DEBUGGED)
# ——————————————

class SafeGPBoostRegressor:
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="gaussian", trace=False, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None: cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = likelihood
        self.seed          = seed
        self.kwargs        = kw
        self._model        = None
        self.trace         = trace 

    def fit(self, X, y):
        intercept = np.ones(len(y))
        gp_kwargs = {
            "gp_coords": X, "gp_approx": self.gp_approx,
            "cov_function": self.cov_function, "likelihood": self.likelihood,
            "seed": self.seed, **self.kwargs
        }
        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        
        # Robust inducing points
        if gp_kwargs.get("gp_approx") == "full_scale_vecchia":
            n = len(y)
            try:
                X_cont = np.ascontiguousarray(X)
                dtype = np.dtype((np.void, X_cont.dtype.itemsize * X_cont.shape[1]))
                n_unique = len(np.unique(X_cont.view(dtype)))
            except: n_unique = n
            
            limit = min(n, n_unique)
            safe_ind = max(10, limit - 1) 
            user_val = self.kwargs.get("num_ind_points", safe_ind)
            gp_kwargs["num_ind_points"] = min(user_val, safe_ind)

        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": self.trace}) 
        return self

    def predict(self, X, return_var=False):
        intercept = np.ones(X.shape[0])
        # Gaussian allows predict_response=True
        out = self._model.predict(
            gp_coords_pred=X, X_pred=intercept,
            predict_var=return_var, predict_response=True
        )
        mu = out["mu"]
        if return_var: return mu, out["var"]
        return mu

class SafeGPBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="bernoulli_logit", 
                 matrix_inversion_method=None, trace=False, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None: cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = likelihood
        self.matrix_inversion_method = matrix_inversion_method
        self.seed          = seed
        self.kwargs        = kw
        self._model        = None
        self.trace         = trace

    def fit(self, X, y):
        intercept = np.ones(len(y))
        method = self.matrix_inversion_method or "iterative"
        if self.gp_approx == "fitc" and self.likelihood == "bernoulli_logit":
            method = "cholesky"

        gp_kwargs = {
            "gp_coords": X, "gp_approx": self.gp_approx,
            "cov_function": self.cov_function, "likelihood": self.likelihood,
            "matrix_inversion_method": method, "seed": self.seed, **self.kwargs
        }
        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        
        # Robust inducing points
        if gp_kwargs.get("gp_approx") == "full_scale_vecchia":
            n = len(y)
            try:
                X_cont = np.ascontiguousarray(X)
                dtype = np.dtype((np.void, X_cont.dtype.itemsize * X_cont.shape[1]))
                n_unique = len(np.unique(X_cont.view(dtype)))
            except: n_unique = n
            limit = min(n, n_unique)
            safe_ind = max(10, limit - 1)
            user_val = self.kwargs.get("num_ind_points", safe_ind)
            gp_kwargs["num_ind_points"] = min(user_val, safe_ind)
            
        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": self.trace})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        
        # --- CRITICAL FIX: FORCE FALSE ---
        # This is the line that fixes the "Fatal Option" error.
        out = self._model.predict(
            gp_coords_pred=X, X_pred=intercept,
            predict_var=False, predict_response=False 
        )
        mu_logits = out["mu"]
        
        # Manual Sigmoid
        mu_logits = np.clip(mu_logits, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-mu_logits))
        return np.vstack([1 - prob, prob]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

class SafeGPBoostMulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._model = None
        self._label_encoder = LabelEncoder()

    def fit(self, X, y):
        self._model = OneVsRestClassifier(SafeGPBoostClassifier(**self.kwargs))
        y_encoded = self._label_encoder.fit_transform(y)
        self._model.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)