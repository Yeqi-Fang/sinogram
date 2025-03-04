"""
Visualization utilities for sinogram data
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_sinogram(sinogram, title=None, cmap='magma', ax=None, slice_idx=0):
    """
    Plot a single slice of a sinogram.
    
    Args:
        sinogram (torch.Tensor or np.ndarray): Sinogram data
        title (str, optional): Title for the plot
        cmap (str, optional): Colormap to use
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        slice_idx (int, optional): Index of the slice to plot
    
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if torch.is_tensor(sinogram):
        if sinogram.dim() == 3:
            sinogram_slice = sinogram[:, :, slice_idx].detach().cpu().numpy()
        else:
            sinogram_slice = sinogram.detach().cpu().numpy()
    else:
        if sinogram.ndim == 3:
            sinogram_slice = sinogram[:, :, slice_idx]
        else:
            sinogram_slice = sinogram
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(sinogram_slice, cmap=cmap, aspect='auto')
    if title:
        ax.set_title(title)
    ax.set_xlabel('Radial Position')
    ax.set_ylabel('Angle')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    return ax

def plot_comparison(input_sinogram, pred_sinogram, target_sinogram, slice_idx=0, 
                    figsize=(18, 6), titles=None):
    """
    Plot comparison between input, predicted, and target sinograms.
    
    Args:
        input_sinogram (torch.Tensor): Input incomplete sinogram
        pred_sinogram (torch.Tensor): Predicted complete sinogram
        target_sinogram (torch.Tensor): Target complete sinogram
        slice_idx (int, optional): Index of the slice to plot
        figsize (tuple, optional): Figure size
        titles (list, optional): List of titles for the plots
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    if titles is None:
        titles = ['Incomplete Sinogram', 'Predicted Sinogram', 'Ground Truth Sinogram']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot input sinogram
    plot_sinogram(input_sinogram, titles[0], ax=axes[0], slice_idx=slice_idx)
    
    # Plot predicted sinogram
    plot_sinogram(pred_sinogram, titles[1], ax=axes[1], slice_idx=slice_idx)
    
    # Plot target sinogram
    plot_sinogram(target_sinogram, titles[2], ax=axes[2], slice_idx=slice_idx)
    
    plt.tight_layout()
    return fig

def plot_error_map(pred_sinogram, target_sinogram, slice_idx=0, figsize=(12, 5)):
    """
    Plot error map between predicted and target sinograms.
    
    Args:
        pred_sinogram (torch.Tensor): Predicted complete sinogram
        target_sinogram (torch.Tensor): Target complete sinogram
        slice_idx (int, optional): Index of the slice to plot
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    # Extract slices for visualization
    if torch.is_tensor(pred_sinogram):
        if pred_sinogram.dim() == 3:
            pred_slice = pred_sinogram[:, :, slice_idx].detach().cpu().numpy()
        else:
            pred_slice = pred_sinogram.detach().cpu().numpy()
    else:
        if pred_sinogram.ndim == 3:
            pred_slice = pred_sinogram[:, :, slice_idx]
        else:
            pred_slice = pred_sinogram
    
    if torch.is_tensor(target_sinogram):
        if target_sinogram.dim() == 3:
            target_slice = target_sinogram[:, :, slice_idx].detach().cpu().numpy()
        else:
            target_slice = target_sinogram.detach().cpu().numpy()
    else:
        if target_sinogram.ndim == 3:
            target_slice = target_sinogram[:, :, slice_idx]
        else:
            target_slice = target_sinogram
    
    # Calculate error
    error = np.abs(pred_slice - target_slice)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot absolute error
    im1 = axes[0].imshow(error, cmap='hot', aspect='auto')
    axes[0].set_title('Absolute Error')
    axes[0].set_xlabel('Radial Position')
    axes[0].set_ylabel('Angle')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    # Plot normalized error (percentage)
    if np.max(target_slice) > 0:
        normalized_error = error / np.max(target_slice) * 100
        im2 = axes[1].imshow(normalized_error, cmap='hot', aspect='auto', vmax=50)  # Cap at 50%
        axes[1].set_title('Relative Error (%)')
        axes[1].set_xlabel('Radial Position')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    return fig

def plot_training_curves(train_losses, val_losses, metrics=None, figsize=(15, 5)):
    """
    Plot training and validation curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        metrics (dict, optional): Dictionary of evaluation metrics
        figsize (tuple, optional): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    if metrics is None:
        # Only plot losses
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
    else:
        # Plot losses and metrics
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot losses
        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot PSNR
        if 'psnr' in metrics:
            axes[1].plot(metrics['psnr'])
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('PSNR (dB)')
        
        # Plot SSIM
        if 'ssim' in metrics:
            axes[2].plot(metrics['ssim'])
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('SSIM')
    
    plt.tight_layout()
    return fig

def visualize_multiple_slices(sinogram, num_slices=4, start_idx=0, step=None, 
                             figsize=(15, 10), cmap='magma', title=None):
    """
    Visualize multiple slices of a sinogram.
    
    Args:
        sinogram (torch.Tensor): Sinogram data of shape [angles, radial, planes]
        num_slices (int): Number of slices to visualize
        start_idx (int): Starting slice index
        step (int, optional): Step size between slices
        figsize (tuple): Figure size
        cmap (str): Colormap to use
        title (str, optional): Title for the figure
        
    Returns:
        matplotlib.figure.Figure: The figure
    """
    if torch.is_tensor(sinogram):
        sinogram = sinogram.detach().cpu().numpy()
    
    # Determine number of rows and columns for subplots
    n_cols = min(num_slices, 4)
    n_rows = (num_slices + n_cols - 1) // n_cols
    
    # Determine step size
    if step is None:
        if sinogram.shape[2] > num_slices:
            step = sinogram.shape[2] // num_slices
        else:
            step = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_slices):
        ax = axes[i]
        slice_idx = start_idx + i * step
        
        if slice_idx < sinogram.shape[2]:
            im = ax.imshow(sinogram[:, :, slice_idx], cmap=cmap, aspect='auto')
            ax.set_title(f'Slice {slice_idx}')
            ax.set_xlabel('Radial Position')
            ax.set_ylabel('Angle')
            plt.colorbar(im, ax=ax)
        else:
            ax.axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig