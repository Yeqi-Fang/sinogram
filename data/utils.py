"""
Utility functions for sinogram data processing
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def create_angle_mask(sinogram_shape, start_angle, end_angle):
    """
    Create a binary mask for the sinogram to simulate missing angular data.
    
    Args:
        sinogram_shape (tuple): Shape of the sinogram (angles, radial, planes)
        start_angle (float): Starting angle in degrees to mask
        end_angle (float): Ending angle in degrees to mask
        
    Returns:
        torch.Tensor: Binary mask where 0 indicates masked regions
    """
    total_angles = sinogram_shape[0]
    mask = torch.ones(sinogram_shape)
    
    # Convert angles to indices
    start_idx = int(start_angle / 180 * total_angles)
    end_idx = int(end_angle / 180 * total_angles)
    
    # Set masked region to 0
    mask[start_idx:end_idx, :, :] = 0
    
    return mask

def apply_mask(sinogram, mask):
    """
    Apply mask to sinogram to simulate incomplete data
    
    Args:
        sinogram (torch.Tensor): Complete sinogram data
        mask (torch.Tensor): Binary mask where 0 indicates masked regions
        
    Returns:
        torch.Tensor: Masked sinogram
    """
    return sinogram * mask

def normalize_sinogram(sinogram):
    """
    Normalize sinogram data to [0, 1] range
    
    Args:
        sinogram (torch.Tensor): Raw sinogram data
        
    Returns:
        torch.Tensor: Normalized sinogram
    """
    min_val = sinogram.min()
    max_val = sinogram.max()
    return (sinogram - min_val) / (max_val - min_val + 1e-8)

def log_transform(sinogram):
    """
    Apply log transform to sinogram data (useful for visualization)
    
    Args:
        sinogram (torch.Tensor): Sinogram data
        
    Returns:
        torch.Tensor: Log-transformed sinogram
    """
    return torch.log(sinogram + 1.0)

def split_sinogram_blocks(sinogram, block_size=64):
    """
    Split sinogram into blocks along the ring difference dimension
    
    Args:
        sinogram (torch.Tensor): Complete sinogram of shape [angles, radial, planes]
        block_size (int): Size of each block
        
    Returns:
        list: List of sinogram blocks
    """
    num_blocks = sinogram.shape[2] // block_size
    if sinogram.shape[2] % block_size != 0:
        num_blocks += 1
    
    blocks = []
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, sinogram.shape[2])
        
        # Handle last block that might be smaller
        if end_idx - start_idx < block_size:
            block = torch.zeros(sinogram.shape[0], sinogram.shape[1], block_size)
            block[:, :, :(end_idx - start_idx)] = sinogram[:, :, start_idx:end_idx]
        else:
            block = sinogram[:, :, start_idx:end_idx]
        
        blocks.append(block)
    
    return blocks

def combine_sinogram_blocks(blocks, original_depth):
    """
    Combine sinogram blocks back into a complete sinogram
    
    Args:
        blocks (list): List of sinogram blocks
        original_depth (int): Original depth of the sinogram
        
    Returns:
        torch.Tensor: Combined sinogram
    """
    # Get shape information from the first block
    height, width, block_depth = blocks[0].shape
    
    # Create empty sinogram
    combined = torch.zeros(height, width, original_depth)
    
    # Fill in the combined sinogram
    current_depth = 0
    for block in blocks:
        space_left = original_depth - current_depth
        if space_left <= 0:
            break
            
        # Determine how much of the block to use
        use_depth = min(block_depth, space_left)
        
        # Copy block data
        combined[:, :, current_depth:current_depth+use_depth] = block[:, :, :use_depth]
        
        # Update current depth
        current_depth += use_depth
    
    return combined

def visualize_sinogram(sinogram, slice_idx=0, title=None, cmap='magma', ax=None):
    """
    Visualize a slice of the sinogram
    
    Args:
        sinogram (torch.Tensor): Sinogram data
        slice_idx (int): Index of the slice to visualize
        title (str): Title of the plot
        cmap (str): Colormap to use
        ax (matplotlib.axes.Axes): Axes to plot on
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Extract slice for visualization
    if sinogram.dim() == 3:
        slice_data = sinogram[:, :, slice_idx].detach().cpu().numpy()
    else:
        slice_data = sinogram[:, :].detach().cpu().numpy()
    
    im = ax.imshow(slice_data, cmap=cmap, aspect='auto')
    if title:
        ax.set_title(title)
    ax.set_xlabel('Radial Position')
    ax.set_ylabel('Angle')
    plt.colorbar(im, ax=ax)
    
    return ax