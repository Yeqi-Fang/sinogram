"""
Data augmentation techniques for sinogram data
"""
import torch
import random
import numpy as np

class SinogramAugmentation:
    """
    Augmentation pipeline for sinogram data.
    
    Args:
        rotate_prob (float): Probability of applying rotation
        noise_prob (float): Probability of adding noise
        mask_prob (float): Probability of applying random masking
        noise_level (float): Standard deviation of Gaussian noise
    """
    def __init__(self, rotate_prob=0.5, noise_prob=0.3, mask_prob=0.7, noise_level=0.02):
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
        self.noise_level = noise_level
    
    def __call__(self, sinogram):
        """
        Apply random augmentations to the sinogram.
        
        Args:
            sinogram (torch.Tensor): Input sinogram
            
        Returns:
            torch.Tensor: Augmented sinogram
        """
        # Make a copy of the input
        result = sinogram.clone()
        
        # Apply random rotation (shift in angles)
        if random.random() < self.rotate_prob:
            shift = random.randint(1, sinogram.shape[0] // 4)
            result = torch.roll(result, shifts=shift, dims=0)
        
        # Add Gaussian noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(result) * self.noise_level
            result = result + noise
            # Ensure values remain in valid range
            if result.max() > 0:  # Only normalize if we have non-zero values
                result = torch.clamp(result, 0, result.max())
        
        # Apply random angular masking (for training robustness)
        if random.random() < self.mask_prob:
            angle_range = 180  # Assuming sinogram covers 180 degrees
            total_angles = sinogram.shape[0]
            
            # Randomly select an angular range to mask
            start_angle = random.uniform(0, 150)
            end_angle = start_angle + random.uniform(10, 30)
            
            # Convert angles to indices
            start_idx = int(start_angle / angle_range * total_angles)
            end_idx = int(end_angle / angle_range * total_angles)
            
            # Apply mask
            result[start_idx:end_idx, :, :] = 0
        
        return result

class SinogramMixup:
    """
    Mixup augmentation for sinogram data.
    Combines two sinograms with a random weight.
    
    Args:
        alpha (float): Parameter for beta distribution to sample mixing weights
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Apply mixup to a batch of sinograms.
        
        Args:
            batch (tuple): Tuple of (inputs, targets) where:
                - inputs: torch.Tensor of shape [B, H, W, C]
                - targets: torch.Tensor of shape [B, H, W, C]
                
        Returns:
            tuple: Tuple of (mixed_inputs, mixed_targets, lam)
        """
        inputs, targets = batch
        batch_size = inputs.size(0)
        
        # Sample mixing weight from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Generate random permutation of indices
        index = torch.randperm(batch_size).to(inputs.device)
        
        # Mix the inputs and targets
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_inputs, mixed_targets

class RandomAngularMasking:
    """
    Randomly mask out angular sections of the sinogram.
    
    Args:
        min_angle (float): Minimum angle to mask (degrees)
        max_angle (float): Maximum angle to mask (degrees)
        num_sections (int): Maximum number of sections to mask
    """
    def __init__(self, min_angle=10, max_angle=60, num_sections=2):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_sections = num_sections
    
    def __call__(self, sinogram):
        """
        Apply random angular masking to the sinogram.
        
        Args:
            sinogram (torch.Tensor): Input sinogram
            
        Returns:
            torch.Tensor: Masked sinogram
        """
        # Make a copy of the input
        result = sinogram.clone()
        
        # Determine number of sections to mask
        num_sections = random.randint(1, self.num_sections)
        
        # For each section, mask out a random angular range
        for _ in range(num_sections):
            angle_range = 180  # Assuming sinogram covers 180 degrees
            total_angles = sinogram.shape[0]
            
            # Randomly select an angular range to mask
            mask_size = random.uniform(self.min_angle, self.max_angle)
            start_angle = random.uniform(0, 180 - mask_size)
            end_angle = start_angle + mask_size
            
            # Convert angles to indices
            start_idx = int(start_angle / angle_range * total_angles)
            end_idx = int(end_angle / angle_range * total_angles)
            
            # Apply mask
            result[start_idx:end_idx, :, :] = 0
        
        return result