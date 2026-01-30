import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from scipy.stats import norm
import warnings
from properscoring import crps_gaussian, crps_ensemble

def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    try:

        return accuracy_score(y_true, y_pred)
    except Exception as e:
        warnings.warn(f"An error occurred while computing accuracy: {e}")
        return np.nan

def evaluate_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:

    try:
        labels = np.unique(y_true)
        return log_loss(y_true, y_prob, labels=labels)
    except Exception as e:
        warnings.warn(f"An error occurred while computing log loss: {e}")
        return np.nan

def evaluate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    try:
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)
    except Exception as e:
        warnings.warn(f"An error occurred while computing RMSE: {e}")
        return np.nan

def _crps_gaussian_scalar(y, mu, sigma):
    """
    CRPS for a single observation assuming a Gaussian predictive distribution.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return crps_gaussian(y, mu, sigma)

def evaluate_crps(y_true: np.ndarray,
                  y_pred_mean: np.ndarray,
                  y_pred_std: np.ndarray) -> float:
    """
    Mean CRPS over all observations for Gaussian predictive means / stds.
    """
    y_true         = np.asarray(y_true)
    y_pred_mean    = np.asarray(y_pred_mean)
    y_pred_std     = np.asarray(y_pred_std)

    if not (y_true.shape == y_pred_mean.shape == y_pred_std.shape):
        raise ValueError("y_true, y_pred_mean and y_pred_std must have the same shape.")

    if np.any(y_pred_std <= 0):
        raise ValueError("All elements of y_pred_std must be strictly positive.")

    crps_vals = crps_gaussian(y_true, y_pred_mean, y_pred_std)
    return float(np.mean(crps_vals))
