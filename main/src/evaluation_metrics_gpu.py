import torch
import numpy as np
import warnings

def _to_tensor(x, device=None):
    """Helper to ensure input is a tensor on the correct device."""
    if torch.is_tensor(x):
        return x.to(device) if device else x
    return torch.tensor(x, device=device)

def evaluate_accuracy(y_true, y_pred) -> float:
    try:
        # Handle inputs (support both Tensor and Numpy)
        if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
        if not torch.is_tensor(y_pred): y_pred = torch.tensor(y_pred)
        
        # Ensure on same device
        if y_true.device != y_pred.device:
            y_true = y_true.to(y_pred.device)
            
        # Compute accuracy
        correct = (y_true == y_pred).float()
        return correct.mean().item()
        
    except Exception as e:
        warnings.warn(f"An error occurred while computing accuracy: {e}")
        return float('nan')

def evaluate_log_loss(y_true, y_prob) -> float:
    try:
        # Handle inputs
        if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
        if not torch.is_tensor(y_prob): y_prob = torch.tensor(y_prob)
        
        if y_true.device != y_prob.device:
            y_true = y_true.to(y_prob.device)

        # Numerical stability (clamp probabilities to avoid log(0))
        eps = 1e-15
        y_prob = torch.clamp(y_prob, eps, 1 - eps)

        # Binary Case (1D probability array)
        if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
            y_prob = y_prob.view(-1)
            y_true = y_true.view(-1).float()
            loss = -(y_true * torch.log(y_prob) + (1 - y_true) * torch.log(1 - y_prob))
            return loss.mean().item()
            
        # Multiclass Case (N, C)
        else:
            y_true = y_true.long()
            # Gather the probability assigned to the true class
            # Equivalent to: prob_of_truth = y_prob[range(n), y_true]
            row_indices = torch.arange(y_true.shape[0], device=y_true.device)
            true_class_probs = y_prob[row_indices, y_true]
            
            loss = -torch.log(true_class_probs)
            return loss.mean().item()

    except Exception as e:
        warnings.warn(f"An error occurred while computing log loss: {e}")
        return float('nan')

def evaluate_rmse(y_true, y_pred) -> float:
    try:
        if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
        if not torch.is_tensor(y_pred): y_pred = torch.tensor(y_pred)

        if y_true.device != y_pred.device:
            y_true = y_true.to(y_pred.device)

        mse = torch.nn.functional.mse_loss(y_pred.float().view(-1), y_true.float().view(-1))
        return torch.sqrt(mse).item()
        
    except Exception as e:
        warnings.warn(f"An error occurred while computing RMSE: {e}")
        return float('nan')

def evaluate_crps(y_true, y_pred_mean, y_pred_std) -> float:
    """
    CRPS for Gaussian distribution using PyTorch.
    Formula matches properscoring.crps_gaussian:
    CRPS = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma
    """
    try:
        # Convert all to tensors
        if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
        if not torch.is_tensor(y_pred_mean): y_pred_mean = torch.tensor(y_pred_mean)
        if not torch.is_tensor(y_pred_std): y_pred_std = torch.tensor(y_pred_std)
        
        # Align devices to y_pred_mean
        device = y_pred_mean.device
        if y_true.device != device: y_true = y_true.to(device)
        if y_pred_std.device != device: y_pred_std = y_pred_std.to(device)

        # Basic shape check
        y_true = y_true.view(-1)
        y_pred_mean = y_pred_mean.view(-1)
        # y_pred_std might be a scalar (single sigma for whole set) or vector
        # Broadcasting handles this automatically in PyTorch

        if torch.any(y_pred_std <= 0):
             warnings.warn("Negative or zero std deviation encountered in CRPS.")
             return float('nan')

        # Calculate z-score
        z = (y_true - y_pred_mean) / y_pred_std

        # Phi(z) = CDF of standard normal
        # 0.5 * (1 + erf(z / sqrt(2)))
        phi_cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))

        # phi(z) = PDF of standard normal
        # exp(-0.5 * z^2) / sqrt(2*pi)
        phi_pdf = (1 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * z**2)

        # 1 / sqrt(pi)
        inv_sqrt_pi = 1 / np.sqrt(np.pi)

        # Full Formula
        crps_val = y_pred_std * (z * (2 * phi_cdf - 1) + 2 * phi_pdf - inv_sqrt_pi)
        
        return crps_val.mean().item()

    except Exception as e:
        warnings.warn(f"An error occurred while computing CRPS: {e}")
        return float('nan')