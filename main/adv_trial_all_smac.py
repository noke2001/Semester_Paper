from joblib import parallel_backend
import os
import sys
import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
import argparse
import random
import gc
import platform
import torch
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian, crps_ensemble

# --- 1. PRE-IMPORT FIXES ---
# os.environ['LOKY_MAX_CPU_COUNT'] = '1' 
# os.environ['JOBLIB_MULTIPROCESSING'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

# --- 2. RPY2 SETUP ---
RPY2_AVAILABLE = False
robjects = None
numpy2ri = None
pandas2ri = None
localconverter = None

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri as n2r
    from rpy2.robjects import pandas2ri as p2r
    from rpy2.robjects.conversion import localconverter as lc
    from rpy2.robjects.vectors import FloatVector

    r_libs_user = os.path.expanduser("~/R_libs")
    if os.path.exists(r_libs_user):
        ro.r(f'.libPaths(c("{r_libs_user}", .libPaths()))')

    try:
        import rpy2.robjects.numpy2ri
        import rpy2.robjects.pandas2ri
        rpy2.robjects.numpy2ri.activate = lambda: None
        rpy2.robjects.pandas2ri.activate = lambda: None
    except:
        pass

    robjects = ro
    numpy2ri = n2r
    pandas2ri = p2r
    localconverter = lc
    RPY2_AVAILABLE = True

except ImportError:
    print(">> Warning: rpy2 not found. DRF model will fail.")
except Exception as e:
    print(f">> Warning: RPY2 init failed: {e}")

# MONKEY PATCH SKLEARN
try:
    import sklearn.ensemble
    original_rf_init = sklearn.ensemble.RandomForestRegressor.__init__
    def safe_rf_init(self, *args, **kwargs):
        kwargs['n_jobs'] = 1 
        original_rf_init(self, *args, **kwargs)
    sklearn.ensemble.RandomForestRegressor.__init__ = safe_rf_init
except ImportError:
    pass

# --- 3. LOG REDIRECTION ---
@contextmanager
def redirect_stdout_to_stderr():
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        os.dup2(sys.stderr.fileno(), original_stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(saved_stdout_fd)

# --- 4. GPBOOST / LIGHTGBM IMPORTS ---
import gpboost as gpb
import lightgbm as lgb
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS

# --- 4b. R CONVERSION HELPERS ---
def to_r_matrix(np_array):
    """Convert numpy array to R matrix using rpy2."""
    if not RPY2_AVAILABLE or not numpy2ri:
        return np_array
    with localconverter(n2r.converter):
        return n2r.py2rpy(np_array)

def to_r_vector(np_array):
    """Convert numpy array to R vector using rpy2."""
    if not RPY2_AVAILABLE:
        return np_array
    if isinstance(np_array, np.ndarray):
        return FloatVector(np_array.astype(float))
    else:
        return FloatVector(np_array)

# --- 5. IMPORTS FROM PROJECT ---
# import smac
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from smac_sampler import SMACSampler

from src.loader import (
    load_dataset_offline, clean_data,
    standardize_data, prepare_for_split
)
from src.extrapolation_methods import (
    random_split, mahalanobis_split, umap_split,
    kmeans_split, gower_split, kmedoids_split, spatial_depth_split
)
from src.evaluation_metrics import (
    evaluate_crps, evaluate_log_loss, evaluate_accuracy
)
# Note: We still import other models, but NOT GPBoost wrappers (we defined them above)
from src.models.Advanced_models import (
    DistributionalRandomForestRegressor,
    LightGBMLSSRegressor,
    SafeGPBoostRegressor,
    SafeGPBoostClassifier

)

# --- CONFIGURATION ---
SEED      = 10
MAX_SAMPLES = 10000 
VAL_RATIO = 0.2
QUANTILE_SAMPLES = 100
USE_GPU = torch.cuda.is_available()

GPU_PARAMS_GPBOOST = {
    "device_type": "gpu",
    "gpu_use_dp": False,
    "bin_construct_sample_cnt": 200000 
} if USE_GPU else {}

# --- SEARCH SPACES ---
def get_drf_space(n_features):
    return {
        'n_estimators': IntDistribution(100, 500),
        'mtry': IntDistribution(0, n_features),
        'min_samples_leaf': IntDistribution(10, 100),
    }

def get_lss_space(n_samples):
    safe_max_bin = max(256, n_samples)
    return {
        "learning_rate": FloatDistribution(0.0001, 0.5, log=True),
        "n_estimators": IntDistribution(100, 500),
        "reg_lambda": FloatDistribution(1e-8, 10.0, log=True),
        "max_depth": IntDistribution(1, 30),
        "min_child_samples": IntDistribution(10, 100),

        "max_bin": IntDistribution(255, safe_max_bin, log=True),
        "subsample": FloatDistribution(0.5, 1.0),
        "colsample_bytree": FloatDistribution(0.5, 1.0),
        "num_leaves": IntDistribution(2, 1024, log=True),
    }

def get_gpboost_space(n_samples):
    upper_bound = min(60, max(10, n_samples - 1))
    lower_bound = min(10, upper_bound)
    return {
        "cov_function": CategoricalDistribution(["matern", "gaussian"]),
        "cov_fct_shape": CategoricalDistribution([0.5, 1.5, 2.5]),

        "num_neighbors": IntDistribution(lower_bound, upper_bound)
    }

def find_config(suite_id):
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg.copy()
    raise ValueError(f"No suite config for suite_id={suite_id}")

def split_dataset(split_fn, X, y):
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_tr, _, y_tr, _, X_te, y_te = out
    else:
        train_idx, test_idx = out
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    return X_tr, y_tr, X_te, y_te

# --- PROGRESS TRACKER ---
class ProgressTracker:
    def __init__(self, current_method_idx, total_methods, n_trials, model_name, metric):
        self.current_method_idx = current_method_idx
        self.total_methods = total_methods
        self.n_trials = n_trials
        self.model_name = model_name
        self.metric = metric

    def callback(self, study, trial):
        display_trial = trial.number + 1
        if display_trial == 1 or display_trial % 25 == 0:
            print(f"  [Progress] Split {self.current_method_idx}/{self.total_methods} | "
                  f"{self.model_name} ({self.metric}) | Trial {display_trial}/{self.n_trials}")
            sys.stdout.flush()

SUITE_CONFIG = {
    "regression_numerical":             {"suite_id":336, "task_type":"regression",       "data_type":"numerical"},
    "classification_numerical":         {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical": {"suite_id":335, "task_type":"regression",       "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    },
    "tabzilla": {"suite_id":379, "task_type":"classification",       "data_type":None}
}
EXTRAPOLATION_METHODS = {
    "numerical":             [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical": [random_split, gower_split, kmedoids_split, umap_split]
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",     type=int,   required=True)
    parser.add_argument("--task_id",      type=int,   required=True)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--result_folder",type=str,   required=True)
    parser.add_argument("--n_trials",     type=int,   default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg      = find_config(args.suite_id)
    is_reg   = (cfg["task_type"] == "regression")

    X_full, y_full, cat_ind, attr_names = load_dataset_offline(args.suite_id, args.task_id)

    if cfg["data_type"] is None:
        has_categorical = bool(getattr(cat_ind, "any")()) if hasattr(cat_ind, "any") else any(cat_ind)
        cfg["data_type"] = "numerical_categorical" if has_categorical else "numerical"
      
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    num_methods = len(methods)

    if len(X_full) > MAX_SAMPLES:
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full, train_size=MAX_SAMPLES,
            stratify=y_full if not is_reg else None, random_state=args.seed
        )

    X, X_clean, y = clean_data(X_full, y_full, cat_ind, attr_names, task_type=cfg["task_type"])
    if args.task_id in (361082, 361088, 361099) and is_reg: y = np.log(y)

    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_advanced.csv")

    records = []
    quantiles = list(np.random.uniform(0,1,QUANTILE_SAMPLES))
    X_ready = prepare_for_split(X_clean)
    
    # --- MAIN LOOP ---
    for m_idx, split_fn in enumerate(methods, 1):
        name_split = split_fn.__name__
        print(f"\n>>> Processing Split Method {m_idx}/{num_methods}: {name_split}")
        
        try:
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_ready, y)
            X_train_clean_unfilt, _, X_val_clean_unfilt, _ = split_dataset(split_fn, X_tr_clean, y_tr)
        except Exception as e:
            print(f"Skipping {name_split}: Split failed: {e}")
            continue

        if X_tr_clean.shape[1] == 0 or X_te_clean.shape[1] == 0: continue

        X_loop = X.copy()
        dummy_cols = X_loop.select_dtypes(include=['bool','category','object','string']).columns
        for col in dummy_cols:
            if X_loop[col].nunique() != X_loop.loc[X_train_clean_unfilt.index, col].nunique():
                X_loop = X_loop.drop(col, axis=1)

        if X_loop.shape[1] == 0: continue

        non_dummy_cols = X_loop.select_dtypes(exclude=['bool','category','object','string']).columns.tolist()
        X_loop = pd.get_dummies(X_loop, drop_first=True).astype('float32').fillna(0)

        X_tr_p = X_loop.loc[X_tr_clean.index]
        X_te_p = X_loop.loc[X_te_clean.index]
        X_train = X_tr_p.loc[X_train_clean_unfilt.index]
        X_val   = X_tr_p.loc[X_val_clean_unfilt.index]

        X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
        X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)

        X_train_np = np.ascontiguousarray(X_train_.fillna(0).to_numpy())
        X_val_np   = np.ascontiguousarray(X_val.fillna(0).to_numpy())
        X_tr_p_np  = np.ascontiguousarray(X_tr_p.fillna(0).to_numpy())
        X_te_p_np  = np.ascontiguousarray(X_te_p.fillna(0).to_numpy())

        y_train_ = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val    = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()
        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        n_samples_train = X_train_np.shape[0]
        n_features = X_train_np.shape[1]

        # --- OBJECTIVE FUNCTIONS ---
        def obj_crps_drf(trial):
            params = {
                'num_trees': trial.suggest_int('num_trees', 100, 500),
                'min_node_size': trial.suggest_int('min_node_size', 10, 100),
                'seed': args.seed
            }
            mtry_ratio = trial.suggest_float('mtry_ratio', 0.1, 1.0)
            params['mtry'] = max(1, int(mtry_ratio * n_features))
            _ = trial.suggest_int('max_depth', 1, 30) 

            model = DistributionalRandomForestRegressor(**params)
            
            with redirect_stdout_to_stderr():
                # Ensure clean numpy arrays
                X_train_clean = X_train_np.detach().cpu().numpy() if hasattr(X_train_np, 'detach') else np.ascontiguousarray(X_train_np)
                X_val_clean = X_val_np.detach().cpu().numpy() if hasattr(X_val_np, 'detach') else np.ascontiguousarray(X_val_np)
                y_train_clean = y_train_.detach().cpu().numpy() if hasattr(y_train_, 'detach') else y_train_
                y_train_clean = np.ascontiguousarray(y_train_clean).ravel().astype(float)
                
                # DRF handles R conversion internally - just pass numpy arrays
                model.fit(X_train_clean, y_train_clean)
                y_q = model.predict_quantiles(X_val_clean, quantiles=quantiles)
            
            if trial.number % 10 == 0 and RPY2_AVAILABLE: robjects.r('gc()')
            vals = crps_ensemble(y_val, y_q.quantile.squeeze(1))
            return float(np.mean(vals))

        def get_lgbm_params(trial):
            return {
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1000.0, log=True),
                'max_bin': trial.suggest_int('max_bin', 255, max(256, n_samples_train), log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'max_depth': trial.suggest_categorical('max_depth', [-1]),
                'num_leaves': trial.suggest_int('num_leaves', 2, 1024, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500)
            }

        def obj_crps_lss(trial):
            params = get_lgbm_params(trial)
            model = LightGBMLSSRegressor(**params)
            with redirect_stdout_to_stderr():
                model.fit(X_train_np, y_train_)
            pr = model.predict_parameters(X_val_np)
            score = float(np.mean(crps_gaussian(y_val, pr['loc'], pr['scale'])))
            del model; gc.collect()
            return score

        # --- GPBOOST OBJS (Using INLINE Safe Classes) ---
        
        def obj_crps_gpboost(trial):
            cov_function = trial.suggest_categorical("cov_function", ["matern", "gaussian"])
            cov_fct_shape = trial.suggest_categorical("cov_fct_shape", [0.5, 1.5, 2.5])
            if cov_function != "matern": cov_fct_shape = None
            num_neighbors = trial.suggest_int("num_neighbors", 10, 60)

            kwargs = GPU_PARAMS_GPBOOST.copy()
            
            try:
                model = SafeGPBoostRegressor(
                    gp_approx="full_scale_vecchia",
                    cov_function=cov_function,
                    cov_fct_shape=cov_fct_shape,
                    seed=args.seed,
                    likelihood="gaussian",
                    num_neighbors=num_neighbors,
                    **kwargs
                )
                # Inject params
                for k, v in kwargs.items(): setattr(model, k, v)
                
                with redirect_stdout_to_stderr():
                    model.fit(X_train_np, y_train_)
                mu, var = model.predict(X_val_np, return_var=True)
                var_clean = np.maximum(var, 0) # clamped to avoid negatives
                score = evaluate_crps(y_val, mu, np.sqrt(var_clean))
                del model; gc.collect()
                return score
            except Exception: return float('inf')

        def obj_rmse_gpboost(trial):
            cov_function = trial.suggest_categorical("cov_function", ["matern", "gaussian"])
            cov_fct_shape = trial.suggest_categorical("cov_fct_shape", [0.5, 1.5, 2.5])
            if cov_function != "matern": cov_fct_shape = None
            num_neighbors = trial.suggest_int("num_neighbors", 10, 60)

            kwargs = GPU_PARAMS_GPBOOST.copy()
            
            model = SafeGPBoostRegressor(
                gp_approx="full_scale_vecchia",
                cov_function=cov_function,
                cov_fct_shape=cov_fct_shape,
                seed=args.seed,
                trace=False,
                num_neighbors=num_neighbors,
                **kwargs
            )
            for k, v in kwargs.items(): setattr(model, k, v)
            
            with redirect_stdout_to_stderr():
                model.fit(X_train_np, y_train_)
            preds = model.predict(X_val_np)
            score = float(np.sqrt(np.mean((y_val - preds)**2)))
            del model; gc.collect()
            return score
        
        def obj_logloss_gpboost(trial):
            cov_function = trial.suggest_categorical("cov_function", ["matern", "gaussian"])
            cov_fct_shape = trial.suggest_categorical("cov_fct_shape", [0.5, 1.5, 2.5])
            if cov_function != "matern": cov_fct_shape = None
            num_neighbors = trial.suggest_int("num_neighbors", 10, 60)

            kwargs = GPU_PARAMS_GPBOOST.copy()
            
            try:
                model = SafeGPBoostClassifier(
                    gp_approx="full_scale_vecchia",
                    cov_function=cov_function,
                    cov_fct_shape=cov_fct_shape,
                    matrix_inversion_method="iterative",
                    seed=args.seed,
                    likelihood="bernoulli_logit",
                    num_neighbors=num_neighbors,
                    **kwargs
                )
                for k, v in kwargs.items(): setattr(model, k, v)
                
                with redirect_stdout_to_stderr():
                    model.fit(X_train_np, y_train_)
                probs = model.predict_proba(X_val_np)[:,1]
                score = evaluate_log_loss(y_val, probs)
                del model; gc.collect()
                return score
            except Exception: return float('inf')
            
        def obj_acc_gpboost(trial):
            cov_function = trial.suggest_categorical("cov_function", ["matern", "gaussian"])
            cov_fct_shape = trial.suggest_categorical("cov_fct_shape", [0.5, 1.5, 2.5])
            if cov_function != "matern": cov_fct_shape = None
            num_neighbors = trial.suggest_int("num_neighbors", 10, 60)

            kwargs = GPU_PARAMS_GPBOOST.copy()
            
            try:
                model = SafeGPBoostClassifier(
                    gp_approx="full_scale_vecchia",
                    cov_function=cov_function,
                    cov_fct_shape=cov_fct_shape,
                    matrix_inversion_method="iterative",
                    seed=args.seed,
                    likelihood="bernoulli_logit",
                    num_neighbors=num_neighbors,
                    **kwargs
                )
                for k, v in kwargs.items(): setattr(model, k, v)
                
                with redirect_stdout_to_stderr():
                    model.fit(X_train_np, y_train_)
                probs = model.predict(X_val_np)
                score = evaluate_accuracy(y_val, probs)
                del model; gc.collect()
                return score
            except Exception: return 0.0

        tasks = []
        if is_reg:
            tasks.append(("DRF", obj_crps_drf, 'minimize', 'CRPS', DistributionalRandomForestRegressor, get_drf_space(n_samples_train)))
            tasks.append(("DGBT", obj_crps_lss, 'minimize', 'CRPS', LightGBMLSSRegressor, get_lss_space(n_samples_train)))
            tasks.append(("GPBoost_CRPS", obj_crps_gpboost, 'minimize', 'CRPS', SafeGPBoostRegressor, get_gpboost_space(n_samples_train)))
            tasks.append(("GPBoost_RMSE", obj_rmse_gpboost, 'minimize', 'RMSE', SafeGPBoostRegressor, get_gpboost_space(n_samples_train)))
        else:
            tasks.append(("GPBoost_LogLoss", obj_logloss_gpboost, 'minimize', 'LogLoss', SafeGPBoostClassifier, get_gpboost_space(n_samples_train)))
            tasks.append(("GPBoost_Accuracy", obj_acc_gpboost, 'maximize', 'Accuracy', SafeGPBoostClassifier, get_gpboost_space(n_samples_train)))
        
        for name, obj_fn, direction, metric, ModelClass, search_space in tasks:
            tracker = ProgressTracker(m_idx, num_methods, args.n_trials, name, metric)
            
            try:
                sampler = SMACSampler(search_space=search_space, n_trials=args.n_trials, seed=args.seed)
                study = optuna.create_study(sampler=sampler, direction=direction)
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                with parallel_backend('threading', n_jobs=1):
                    study.optimize(obj_fn, n_trials=args.n_trials, callbacks=[tracker.callback])
                
                try:
                    best_params = study.best_params
                except ValueError:
                    print(f"Skipping {name} on split {name_split}: No successful trials.")
                    continue
                
                # --- REFIT ---
                final_params = best_params.copy()
                
                if "GPBoost" in name:
                    cov_f = final_params["cov_function"]
                    cov_s = final_params["cov_fct_shape"] if cov_f == "matern" else None
                    num_n = final_params["num_neighbors"]
                    
                    kwargs = {
                        "gp_approx": "full_scale_vecchia",
                        "cov_function": cov_f,
                        "cov_fct_shape": cov_s,
                        "seed": args.seed,
                        "num_neighbors": num_n,
                        "trace": False
                    }
                    kwargs.update(GPU_PARAMS_GPBOOST)
                    
                    if is_reg:
                        if metric == "CRPS": kwargs["likelihood"] = "gaussian"
                        model = SafeGPBoostRegressor(**kwargs)
                    else:
                        kwargs["matrix_inversion_method"] = "iterative"
                        kwargs["likelihood"] = "bernoulli_logit"
                        model = SafeGPBoostClassifier(**kwargs)
                    
                    # Inject params
                    for k, v in kwargs.items(): setattr(model, k, v)
                    
                    # Inject boosting params (if any were left over from mixed config)
                    boost_params = {k:v for k,v in final_params.items() 
                                    if k not in ["cov_function", "cov_fct_shape"]}
                    for k, v in boost_params.items(): setattr(model, k, v)
                
                elif name == "DGBT": 
                    model = ModelClass(**final_params)
                    
                elif name == "DRF":
                    p_drf = {
                        'num_trees': final_params['num_trees'],
                        'min_node_size': final_params['min_node_size'],
                        'mtry': max(1, int(final_params['mtry_ratio'] * n_features)),
                        'seed': args.seed
                    }
                    model = ModelClass(**p_drf)
                
                # DRF handles R conversion internally - just pass numpy arrays
                inp_tr = X_tr_p_np
                out_tr = y_tr_arr
                
                with redirect_stdout_to_stderr():
                    model.fit(inp_tr, out_tr)

                if metric == 'CRPS':
                    if isinstance(model, DistributionalRandomForestRegressor):
                        # DRF handles conversion internally - pass numpy arrays directly
                        yq = model.predict_quantiles(X_te_p_np, quantiles=quantiles)
                        vals = crps_ensemble(y_te_arr, yq.quantile.squeeze(1))
                    else:
                        # Regressor Refit
                        if isinstance(model, SafeGPBoostRegressor):
                             mu, var = model.predict(X_te_p_np, return_var=True)
                             vals = evaluate_crps(y_te_arr, mu, np.sqrt(var))
                        else: 
                            pr = model.predict_parameters(X_te_p_np)
                            vals = crps_gaussian(y_te_arr, pr['loc'], pr['scale'])
                    
                    if isinstance(model, SafeGPBoostRegressor):
                        val = vals
                    else:
                        val = float(np.mean(vals))
                         
                elif metric == 'RMSE':
                    preds = model.predict(X_te_p_np)
                    val = float(np.sqrt(np.mean((y_te_arr - preds)**2)))
                elif metric == 'LogLoss':
                    probs = model.predict_proba(X_te_p_np)
                    val = float(evaluate_log_loss(y_te_arr, probs[:,1] if probs.ndim==2 else probs))
                elif metric == 'Accuracy':
                    preds = model.predict(X_te_p_np)
                    val = evaluate_accuracy(y_te_arr, preds)

                records.append({
                    'suite_id': args.suite_id, 'task_id': args.task_id,
                    'split_method': name_split, 'model': name,
                    'metric': metric, 'value': val,
                })
                
                try:
                    pd.DataFrame.from_records(records).to_csv(out_file, index=False)
                except: pass

            except Exception as e:
                print(f"!!! Error for '{name}' on '{name_split}'. Skipping. Error: {e}")
                records.append({'suite_id': args.suite_id, 'task_id': args.task_id, 'split_method': name_split, 'model': name, 'metric': metric, 'value': np.nan})
                try: pd.DataFrame.from_records(records).to_csv(out_file, index=False)
                except: pass
                continue
            del study, model; gc.collect()

    print(f"Saved advanced-model results to {out_file}")

if __name__ == '__main__':
    main()