"""
Evaluation metrics for sinogram reconstruction
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        pred (torch.Tensor or np.ndarray): Predicted sinogram
        target (torch.Tensor or np.ndarray): Target sinogram
        
    Returns:
        float: PSNR value
    """
    # Convert to numpy arrays
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
        
    # Ensure values are in valid range
    pred_max = pred.max()
    target_max = target.max()
    
    if pred_max > 0:
        pred = pred / pred_max
    if target_max > 0:
        target = target / target_max
    
    # Calculate PSNR
    return psnr(target, pred, data_range=1.0)

def calculate_ssim(pred, target):
    """
    Calculate Structural Similarity Index
    
    Args:
        pred (torch.Tensor or np.ndarray): Predicted sinogram
        target (torch.Tensor or np.ndarray): Target sinogram
        
    Returns:
        float: SSIM value
    """
    # Convert to numpy arrays
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Ensure values are in valid range
    pred_max = pred.max()
    target_max = target.max()
    
    if pred_max > 0:
        pred = pred / pred_max
    if target_max > 0:
        target = target / target_max
    
    # Handle multi-dimensional data
    if pred.ndim == 3:
        # Calculate SSIM for each slice and average
        ssim_values = []
        for i in range(pred.shape[2]):
            ssim_val = ssim(target[:, :, i], pred[:, :, i], data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Calculate SSIM directly
        return ssim(target, pred, data_range=1.0)

def calculate_rmse(pred, target):
    """
    Calculate Root Mean Square Error
    
    Args:
        pred (torch.Tensor or np.ndarray): Predicted sinogram
        target (torch.Tensor or np.ndarray): Target sinogram
        
    Returns:
        float: RMSE value
    """
    if torch.is_tensor(pred) and torch.is_tensor(target):
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()
    else:
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        return np.sqrt(np.mean((pred - target) ** 2))

def calculate_mae(pred, target):
    """
    Calculate Mean Absolute Error
    
    Args:
        pred (torch.Tensor or np.ndarray): Predicted sinogram
        target (torch.Tensor or np.ndarray): Target sinogram
        
    Returns:
        float: MAE value
    """
    if torch.is_tensor(pred) and torch.is_tensor(target):
        return torch.mean(torch.abs(pred - target)).item()
    else:
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        return np.mean(np.abs(pred - target))

def calculate_metrics(pred, target):
    """
    Calculate all metrics at once
    
    Args:
        pred (torch.Tensor): Predicted sinogram
        target (torch.Tensor): Target sinogram
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'psnr': calculate_psnr(pred, target),
        'ssim': calculate_ssim(pred, target),
        'rmse': calculate_rmse(pred, target),
        'mae': calculate_mae(pred, target)
    }

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count